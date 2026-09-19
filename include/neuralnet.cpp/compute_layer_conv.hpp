#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_layer_base.hpp"
#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// conv_engine — Conv2D / MaxPool2D 共用的布局转换助手（纯引擎原语组合）
//
// 引擎的 rearrange_3d 只能做 (M, B*N) ↔ (B*M, N)，无法一步完成
// (C, B*P) ↔ (C*P, B)（那是 (c,b,p) 三维转置）。这里用
//   rearrange_3d  +  gather_rows/scatter_add_rows（配一份形状相关的置换索引）
// 组合出这两个方向，置换索引与数据无关，由各层缓存一次即可。
// ══════════════════════════════════════════════════════════════════════════
namespace conv_engine {

// 构造置换索引 perm（长度 C*P，一列）：perm[c*P + p] = p*C + c
// 说明：rearrange_3d(Z, C, P, B) 得到行序 (p*C + c)，按 perm 用 gather 重排即得 (c*P + p)。
[[nodiscard]] inline Result<Tensor> make_layout_perm(
    ComputeEngine& engine, std::size_t C, std::size_t P)
{
    if (C == 0 || P == 0)
        return std::unexpected(Error{"make_layout_perm: C/P must be > 0"});
    Matrix m(C * P, 1);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t p = 0; p < P; ++p)
            m.set_value_unchecked(c * P + p, 0, static_cast<Scalar>(p * C + c));
    return engine.from_matrix(m);
}

// (C, B*P) → (C*P, B)：cols 布局转 samples 布局（Conv/Pool 层输出）
[[nodiscard]] inline Result<Tensor> cols_to_samples(
    ComputeEngine& engine, const Tensor& Z,
    std::size_t C, std::size_t P, std::size_t B, const Tensor& perm)
{
    auto w = engine.rearrange_3d(Z, C, P, B, /*inverse=*/false);   // (P*C, B)
    if (!w) return std::unexpected(w.error());
    return engine.gather_rows(*w, perm);                          // (C*P, B)
}

// (C*P, B) → (C, B*P)：samples 布局转回 cols 布局（Conv/Pool 层反向入口）
[[nodiscard]] inline Result<Tensor> samples_to_cols(
    ComputeEngine& engine, const Tensor& g,
    std::size_t C, std::size_t P, std::size_t B, const Tensor& perm)
{
    Tensor x2 = engine.create_tensor(P * C, B);
    if (!x2.valid())
        return std::unexpected(Error{"samples_to_cols: 张量分配失败"});
    { auto r = engine.zero(x2); if (!r) return std::unexpected(r.error()); }
    { auto r = engine.scatter_add_rows(x2, perm, g);
      if (!r) return std::unexpected(r.error()); }
    return engine.rearrange_3d(x2, C, P, B, /*inverse=*/true);    // (C, B*P)
}

} // namespace conv_engine

class Conv2D final : public Layer
{
private:
    std::size_t in_channels_, out_channels_;
    std::size_t kernel_, stride_, padding_;
    std::size_t in_h_, in_w_;   // 输入空间尺寸（方形外也可用，用于 im2col）
    std::size_t out_h_, out_w_; // 输出空间尺寸（构造时计算）

    Tensor w_;        // (C_out, C_in*k*k)
    Tensor b_;        // (C_out, 1)
    Tensor grad_w_;
    Tensor grad_b_;

    // 引擎侧缓存（全设备驻留，无 CPU 中间矩阵）
    Tensor col_cache_;    // im2col 输出 (C_in*k*k, P*B)，供 backward
    Tensor perm_cache_;   // 布局置换索引 (C_out*P, 1)：形状相关、与数据无关
    bool shape_invalid_ = false; // 构造期守卫：kernel 过大（无符号下溢）→ init 报错

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    // 惰性构建布局置换索引（形状不变则跨 forward/backward 复用）
    [[nodiscard]] Result<void> ensure_perm_(ComputeEngine& engine)
    {
        if (perm_cache_.valid()) return {};
        auto p = conv_engine::make_layout_perm(engine, out_channels_, out_h_ * out_w_);
        if (!p) return std::unexpected(p.error());
        perm_cache_ = std::move(*p);
        return {};
    }

public:
    Conv2D(std::size_t in_channels, std::size_t out_channels,
           std::size_t kernel, std::size_t stride = 1, std::size_t padding = 0,
           std::size_t in_h = 0, std::size_t in_w = 0)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_(kernel), stride_(stride != 0 ? stride : 1), padding_(padding),
          in_h_(in_h), in_w_(in_w)
    {
        // API 级守卫：kernel 过大时 (in + 2*pad - kernel) 无符号下溢 →
        // out_h_/out_w_ 巨值 → 分配 abort。记录无效，init() 时报错。
        if (in_h_ + 2 * padding_ < kernel_ || in_w_ + 2 * padding_ < kernel_)
        {
            shape_invalid_ = true;
            out_h_ = 0;
            out_w_ = 0;
        }
        else
        {
            out_h_ = (in_h_ + 2 * padding_ - kernel_) / stride_ + 1;
            out_w_ = (in_w_ + 2 * padding_ - kernel_) / stride_ + 1;
        }
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        if (shape_invalid_)
            return std::unexpected(Error{"Conv2D: kernel 过大 (kernel > in + 2*padding)"});
        const std::size_t fan_in = in_channels_ * kernel_ * kernel_;

        // 权重 (C_out, C_in*k*k) — He 风格均匀初始化
        Matrix w_cpu(out_channels_, fan_in);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(fan_in + out_channels_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_channels_, 1);  // 零初始化

        auto w_res = engine.from_matrix(w_cpu);
        if (!w_res) return std::unexpected(w_res.error());
        w_ = std::move(*w_res);
        auto b_res = engine.from_matrix(b_cpu);
        if (!b_res) return std::unexpected(b_res.error());
        b_ = std::move(*b_res);

        grad_w_ = engine.create_tensor(out_channels_, fan_in);
        grad_b_ = engine.create_tensor(out_channels_, 1);
        { auto r1 = engine.zero(grad_w_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_b_); if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override { return {w_, b_}; }
    std::vector<TensorRef> param_gradients() override { return {grad_w_, grad_b_}; }

    void clear_cache() override { col_cache_ = Tensor{}; }

    // 可作为"重计算单元"：本层无子层，基类 forward_recompute 的默认实现
    // （临时关 checkpoint_mode_ → 重跑 forward → 恢复）即可重建 col_cache_。
    // 注意：需要外部 driver 调用 forward_recompute——CNN 是扁平 Layer 列表，
    // 没有块级 driver，故 Model::set_checkpoint_every 对本层仍是 no-op（见 backward
    // 的缓存校验：未重算就反向会明确报错，而不是静默错值）。
    [[nodiscard]] bool recompute_supported() const override { return true; }

    // ── forward: Z = W × im2col(x) + b，再转 samples 布局 ───────────────
    // 全程引擎原语 + DSL：窗口展开在 im2col 原语内完成（CPU/GPU 各自实现），
    // 无 to_matrix/from_matrix 往返、无 CPU 标量循环。
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (shape_invalid_)
            return std::unexpected(Error{"Conv2D: kernel 过大 (kernel > in + 2*padding)"});
        if (input.rows() != in_channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"conv forward: input shape mismatch"});
        const std::size_t batch = input.cols();
        if (batch == 0)
            return std::unexpected(Error{"conv forward: batch must be > 0"});
        { auto r = ensure_perm_(engine); if (!r) return std::unexpected(r.error()); }

        const std::size_t P = out_h_ * out_w_;

        // 1) 窗口展开（引擎原语）：x (C_in*H*W, B) → col (C_in*k*k, P*B)
        auto col = engine.im2col(input, in_channels_, in_h_, in_w_,
                                 kernel_, stride_, padding_, out_h_, out_w_);
        if (!col) return std::unexpected(col.error());
        // checkpoint 模式不驻留：显式清空，避免 size 相同导致静默用陈旧 im2col
        col_cache_ = checkpoint_mode_ ? Tensor{} : *col;

        // 2) Z = W × col + b → (C_out, P*B)：matmul 段与行广播偏置**融合为单次
        //    dispatch**（GPU 上 2 → 1，且不物化 matmul 中间结果；与
        //    Linear::forward / matmul_with_bias 同一结构）。
        auto Z = dsl::compute(engine,
            dsl::matmul(w_, *col, false, false) + dsl::row_broadcast(b_),
            out_channels_, batch * P);
        if (!Z) return std::unexpected(Z.error());

        // 3) 布局转换 (C_out, P*B) → (C_out*P, B)（rearrange_3d + gather 置换）
        return conv_engine::cols_to_samples(engine, *Z, out_channels_, P, batch, perm_cache_);
    }

    // ── backward ───────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (shape_invalid_)
            return std::unexpected(Error{"Conv2D: kernel 过大 (kernel > in + 2*padding)"});
        if (grad_output.rows() != out_channels_ * out_h_ * out_w_)
            return std::unexpected(Error{"conv backward: grad_output shape mismatch"});
        const std::size_t batch = grad_output.cols();
        if (batch == 0)
            return std::unexpected(Error{"conv backward: batch must be > 0"});
        const std::size_t kk = kernel_ * kernel_;
        const std::size_t P = out_h_ * out_w_;
        // 缓存前置校验：checkpoint 模式（尚未 forward_recompute）下 forward 不驻留
        // col_cache_；不校验会拿空/上一 batch 的陈旧 im2col 静默算出错误梯度。
        if (col_cache_.rows() != in_channels_ * kk || col_cache_.cols() != P * batch)
            return std::unexpected(Error{
                "conv backward: im2col 缓存缺失或不匹配"
                "（checkpoint 模式需先 forward_recompute；batch 变化后需重新 forward）"});
        { auto r = ensure_perm_(engine); if (!r) return std::unexpected(r.error()); }

        // 1) grad_output (C_out*P, B) → gZ (C_out, P*B)
        auto gZ = conv_engine::samples_to_cols(engine, grad_output,
                                               out_channels_, P, batch, perm_cache_);
        if (!gZ) return std::unexpected(gZ.error());

        // 2) grad_W += gZ × col^T → (C_out, C_in*k*k)：matmul 段与累加**融合为单次
        //    dispatch**并原地写入 grad_w_（GPU 上 2 → 1，且不物化 gw）
        auto r1 = dsl::compute_into(engine,
            dsl::leaf(grad_w_) + dsl::matmul(*gZ, col_cache_, false, true), grad_w_);
        if (!r1) return std::unexpected(r1.error());

        // 3) grad_b += row_reduce_sum(gZ) → (C_out, 1)（对全部位置与样本求和）
        auto gb = engine.row_reduce_sum(*gZ);
        if (!gb) return std::unexpected(gb.error());
        auto r2 = engine.add_inplace(grad_b_, *gb);
        if (!r2) return std::unexpected(r2.error());

        // 4) gcol = W^T × gZ → (C_in*k*k, P*B) ；5) grad_x = col2im(gcol)
        auto gcol = engine.matmul(w_, *gZ, true, false);
        if (!gcol) return std::unexpected(gcol.error());
        return engine.col2im(*gcol, in_channels_, in_h_, in_w_,
                             kernel_, stride_, padding_, out_h_, out_w_);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MaxPool2D — 二维最大池化（记录 argmax 供 backward）
//
// 布局约定（batch-major 列布局）：
//   输入  Tensor: (C * H * W, batch)
//   输出  Tensor: (C * Hp * Wp, batch)，Hp=(H-pool)/stride+1，Wp 同理
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  每个 (pool×pool) 窗口取最大值，记录 argmax 位置
//   backward: 把梯度散射回 argmax 位置（其余位置为 0）
//
// 实现（全引擎化，2026-09-20）：
//   forward:  im2col(k=pool, stride=stride, pad=0) 展开窗口
//             → 每通道 col_reduce_max → (C, P*B) → rearrange_3d + gather 转 samples 布局
//   backward: (C*P, B) → scatter_add + rearrange_3d 回 (C, P*B)
//             → 每通道 mask=(窗口 == 窗口 max) 的单次融合 DSL → col2im 散射
// 语义说明：反向按「窗口内并列最大值**均分**该窗口梯度」处理（总梯度守恒）。
//   无并列最大值时与原 argmax-first 实现逐位一致；出现并列最大值时按 1/cnt 均分
//   （合法次梯度；PyTorch 取首个 argmax，TensorFlow 历史上给所有并列元素全量梯度，
//   三者只在并列时不同）。maxpool_gradcheck 同时覆盖 tie-free 与并列用例。
// ══════════════════════════════════════════════════════════════════════════
class MaxPool2D final : public Layer
{
private:
    std::size_t channels_, in_h_, in_w_;
    std::size_t pool_, stride_;
    std::size_t out_h_, out_w_;
    // 引擎侧缓存（全设备驻留，无 CPU 中间矩阵/索引 vector）
    Tensor col_cache_;      // 展开窗口 (C*pool*pool, P*B)
    Tensor pooled_cache_;   // 窗口 max (C, P*B)：backward 的 mask 判据
    Tensor perm_cache_;     // 布局置换索引 (C*P, 1)：形状相关、与数据无关
    Tensor expand_cache_;   // 组内广播索引 (C*pool*pool, 1)：每组行号重复 pool*pool 次
    bool shape_invalid_ = false; // 构造期守卫：pool 过大（无符号下溢）→ forward/backward 报错

    // 惰性构建布局置换索引（形状不变则跨 forward/backward 复用）
    [[nodiscard]] Result<void> ensure_perm_(ComputeEngine& engine)
    {
        if (perm_cache_.valid()) return {};
        auto p = conv_engine::make_layout_perm(engine, channels_, out_h_ * out_w_);
        if (!p) return std::unexpected(p.error());
        perm_cache_ = std::move(*p);
        return {};
    }

    // 惰性构建组内广播索引：gather_rows(pooled (C,N), expand_idx) → (C*kk, N)，
    // 即每个通道的窗口 max 复制 kk 次（供 mask 比较使用；避免再加一个
    // "分组广播"原语——gather_rows 已经是通用的行复制）。
    [[nodiscard]] Result<void> ensure_expand_(ComputeEngine& engine)
    {
        if (expand_cache_.valid()) return {};
        const std::size_t kk = pool_ * pool_;
        Matrix m(channels_ * kk, 1);
        for (std::size_t c = 0; c < channels_; ++c)
            for (std::size_t i = 0; i < kk; ++i)
                m.set_value_unchecked(c * kk + i, 0, static_cast<Scalar>(c));
        auto t = engine.from_matrix(m);
        if (!t) return std::unexpected(t.error());
        expand_cache_ = std::move(*t);
        return {};
    }

public:
    MaxPool2D(std::size_t channels, std::size_t in_h, std::size_t in_w,
              std::size_t pool = 2, std::size_t stride = 0)
        : channels_(channels), in_h_(in_h), in_w_(in_w),
          pool_(pool), stride_(stride != 0 ? stride : (pool != 0 ? pool : 1))
    {
        // API 级守卫：pool 过大（或 pool=0）时 (in - pool) 无符号下溢 →
        // out_h_/out_w_ 巨值 → 分配 abort。记录无效，forward/backward 时报错。
        if (pool_ == 0 || in_h_ < pool_ || in_w_ < pool_)
        {
            shape_invalid_ = true;
            out_h_ = 0;
            out_w_ = 0;
        }
        else
        {
            out_h_ = (in_h_ - pool_) / stride_ + 1;
            out_w_ = (in_w_ - pool_) / stride_ + 1;
        }
    }

    void clear_cache() override
    {
        col_cache_ = Tensor{};
        pooled_cache_ = Tensor{};
    }

    // 同 Conv2D：无子层，基类 forward_recompute 默认实现即可重建窗口/max 缓存；
    // 但 CNN 扁平层列表没有块级 driver，未重算就反向会在 backward 明确报错。
    [[nodiscard]] bool recompute_supported() const override { return true; }

    // ── forward ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (shape_invalid_)
            return std::unexpected(Error{"MaxPool2D: pool 窗口大于输入尺寸"});
        if (input.rows() != channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"maxpool forward: input shape mismatch"});
        const std::size_t batch = input.cols();
        if (batch == 0)
            return std::unexpected(Error{"maxpool forward: batch must be > 0"});
        { auto r = ensure_perm_(engine); if (!r) return std::unexpected(r.error()); }

        const std::size_t kk = pool_ * pool_;
        const std::size_t P = out_h_ * out_w_;

        // 1) 窗口展开（引擎原语）：x (C*H*W, B) → col (C*kk, P*B)
        auto col = engine.im2col(input, channels_, in_h_, in_w_,
                                 pool_, stride_, 0, out_h_, out_w_);
        if (!col) return std::unexpected(col.error());

        // 2) 分组归约求窗口 max：每 kk 行一组 → (C, P*B)。
        //    用 grouped_reduce_max 单次原语完成（早期版本按通道循环
        //    slice_rows + col_reduce_max + insert_rows，C 次 dispatch）。
        auto pooled = engine.grouped_reduce_max(*col, channels_, kk);
        if (!pooled) return std::unexpected(pooled.error());

        // 3) checkpoint 模式不驻留：显式清空（避免 size 相同静默用陈旧数据）
        if (checkpoint_mode_)
        {
            col_cache_ = Tensor{};
            pooled_cache_ = Tensor{};
        }
        else
        {
            col_cache_ = *col;
            pooled_cache_ = *pooled;
        }

        // 4) 布局转换 (C, P*B) → (C*P, B)
        return conv_engine::cols_to_samples(engine, *pooled, channels_, P, batch, perm_cache_);
    }

    // ── backward: 按窗口 mask 散射（重叠窗在 col2im 内累加）──────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (shape_invalid_)
            return std::unexpected(Error{"MaxPool2D: pool 窗口大于输入尺寸"});
        if (grad_output.rows() != channels_ * out_h_ * out_w_)
            return std::unexpected(Error{"maxpool backward: grad_output shape mismatch"});
        const std::size_t batch = grad_output.cols();
        if (batch == 0)
            return std::unexpected(Error{"maxpool backward: batch must be > 0"});
        const std::size_t kk = pool_ * pool_;
        const std::size_t P = out_h_ * out_w_;
        // 缓存前置校验：checkpoint（尚未 forward_recompute）或 clear_cache 后缓存为空，
        // 明确报错而不是静默错值。
        if (col_cache_.rows() != channels_ * kk || col_cache_.cols() != P * batch ||
            pooled_cache_.rows() != channels_ || pooled_cache_.cols() != P * batch ||
            !perm_cache_.valid())
            return std::unexpected(Error{
                "maxpool backward: 展开/max 缓存缺失或不匹配"
                "（checkpoint 模式需先 forward_recompute；clear_cache 后需重新 forward）"});

        // 1) grad_output (C*P, B) → g_pooled (C, P*B)
        auto g_pooled = conv_engine::samples_to_cols(engine, grad_output,
                                                     channels_, P, batch, perm_cache_);
        if (!g_pooled) return std::unexpected(g_pooled.error());

        // 2) 并列均分（无逐通道循环，全部为整张量原语 + DSL）
        //    mx_exp = 组内广播(pooled)：gather_rows 把每个通道的 max 复制 kk 次
        //    总梯度守恒（Σ share = 窗口梯度）；无并列时与 argmax 散射逐位一致。
        { auto r = ensure_expand_(engine); if (!r) return std::unexpected(r.error()); }

        auto mx_exp = engine.gather_rows(pooled_cache_, expand_cache_);   // (C*kk, P*B)
        if (!mx_exp) return std::unexpected(mx_exp.error());

        // 并列个数 cnt = 分组求和([x == 窗口 max]) → (C, P*B)
        auto eq = dsl::compute(engine,
            dsl::select(dsl::leaf(col_cache_) == dsl::leaf(*mx_exp),
                        Scalar{1}, Scalar{0}),
            channels_ * kk, P * batch);                                   // (C*kk, P*B)
        if (!eq) return std::unexpected(eq.error());
        auto cnt = engine.grouped_reduce_sum(*eq, channels_, kk);
        if (!cnt) return std::unexpected(cnt.error());

        // 每个并列元素分到的梯度 g/cnt → 再广播回窗口内
        auto gdiv = dsl::compute(engine,
            dsl::leaf(*g_pooled) / dsl::leaf(*cnt),
            channels_, P * batch);                                        // (C, P*B)
        if (!gdiv) return std::unexpected(gdiv.error());
        auto gdiv_exp = engine.gather_rows(*gdiv, expand_cache_);         // (C*kk, P*B)
        if (!gdiv_exp) return std::unexpected(gdiv_exp.error());

        auto gc = dsl::compute(engine,
            dsl::select(dsl::leaf(col_cache_) == dsl::leaf(*mx_exp),
                        dsl::leaf(*gdiv_exp), Scalar{0}),
            channels_ * kk, P * batch);                                   // (C*kk, P*B)
        if (!gc) return std::unexpected(gc.error());

        // 3) 反向散射到输入像素（重叠窗口在 col2im 内累加）
        return engine.col2im(*gc, channels_, in_h_, in_w_,
                             pool_, stride_, 0, out_h_, out_w_);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Softmax — 按行 softmax（用于注意力权重）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  row_max[r] = max_c x[r][c]               (row_reduce_max)
//             shifted[r][c] = x[r][c] - row_max[r]    (broadcast_row Sub)
//             exp_shift[r][c] = exp(shifted[r][c])     (unary Exp)
//             row_sum[r] = Σ_c exp_shift[r][c]         (row_reduce_sum)
//             out[r][c] = exp_shift[r][c] / row_sum[r] (broadcast_row Div)
//   backward: grad_x = out ⊙ (grad_out - row_dot(out ⊙ grad_out))
//             row_dot[r] = Σ_c out[r][c] * grad_out[r][c]
// ══════════════════════════════════════════════════════════════════════════
} // namespace nn

