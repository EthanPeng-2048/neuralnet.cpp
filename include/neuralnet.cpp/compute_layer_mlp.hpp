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
// Linear — 全连接层
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = W × x + b
//   backward: grad_x = W^T × grad_out
//             grad_W += grad_out × x^T
//             grad_b += Σ_batch grad_out
// ══════════════════════════════════════════════════════════════════════════
class Linear final : public Layer
{
private:
    std::size_t in_features_;
    std::size_t out_features_;
    Tensor w_;           // 权重 (out_features, in_features)
    Tensor b_;           // 偏置 (out_features, 1)
    Tensor grad_w_;      // 权重梯度
    Tensor grad_b_;      // 偏置梯度
    Tensor input_cache_; // forward 输入缓存（供 backward 使用）

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

public:
    Linear(std::size_t in_features, std::size_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // ── 在 CPU 上初始化权重（Xavier 均匀分布） ──
        Matrix w_cpu(out_features_, in_features_);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features_ + out_features_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_features_, 1);  // 零初始化

        // ── 通过 engine 上传到目标设备（精度 = p_.param，§9.2）──────────────
        // 全 F32 配置下与迁移前逐字节一致（零回归）；f16 配置 = 权重存储减半
        auto w_res = engine.from_matrix(w_cpu, p_.param);
        if (!w_res) return std::unexpected(w_res.error());
        w_ = std::move(*w_res);

        auto b_res = engine.from_matrix(b_cpu, p_.param);
        if (!b_res) return std::unexpected(b_res.error());
        b_ = std::move(*b_res);

        // ── 梯度张量精度 = p_.param（与参数同精度；§8.3 存储精度不可变）──
        grad_w_ = engine.create_tensor(out_features_, in_features_, p_.param);
        grad_b_ = engine.create_tensor(out_features_, 1, p_.param);
        auto r1 = engine.zero(grad_w_);
        auto r2 = engine.zero(grad_b_);
        if (!r1) return std::unexpected(r1.error());
        if (!r2) return std::unexpected(r2.error());
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {w_, b_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_w_, grad_b_};
    }

    void clear_cache() override { input_cache_ = Tensor{}; }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward: 一行代码，精度由 p_.compute 决定 ──────────────────────
    // 引擎内部处理 matmul + broadcast bias 的精度问题
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != w_.cols())
            return std::unexpected(Error{"linear forward: input shape mismatch"});

        if (!checkpoint_mode_)
            input_cache_ = input;

        // 计算精度 = p_.compute（引擎内部处理 matmul + broadcast bias 的精度：
        // f16 存储经 PrecisionEngine 边界 cast 走 f32 GEMM）
        return engine.matmul_with_bias(w_, input, b_, false, false, p_.compute);
    }

    // ── backward: 同样简洁，精度由引擎处理 ────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (grad_output.rows() != w_.rows())
            return std::unexpected(Error{"linear backward: grad_output shape mismatch"});

        // 计算精度 = p_.compute（in-place 累加的目标精度 = grad_w_ 的存储精度，
        // compute_into 无需 P：§8.3 in-place 存储精度不可变）
        auto grad_input = engine.matmul(w_, grad_output, true, false, p_.compute);
        if (!grad_input) return std::unexpected(grad_input.error());
        nn_dbg_scan("lin.grad_out", engine, grad_output);
        nn_dbg_scan("lin.cache", engine, input_cache_);
        nn_dbg_scan("lin.grad_w(pre)", engine, grad_w_);
        nn_dbg_scan("lin.grad_b(pre)", engine, grad_b_);
        nn_dbg_scan("lin.grad_in", engine, *grad_input);

        // grad_w += grad_output × input^T：matmul 段与累加**融合为单次 dispatch**
        // 并原地写入 grad_w_（GPU 上 1 个融合 kernel：不物化 gw (out,in)，也不额外
        // 分配输出缓冲；取代 matmul + accumulate 两次 dispatch）。
        // k（求和维度 = batch 大小）是形状参数，不进 key → 同一 shader 适配任意 batch。
        auto grad_w_acc = dsl::compute_into(engine,
            dsl::leaf(grad_w_) + dsl::matmul(grad_output, input_cache_, false, true),
            grad_w_);
        if (!grad_w_acc) return std::unexpected(grad_w_acc.error());
        nn_dbg_scan("lin.grad_w(post-accum)", engine, grad_w_);

        // grad_b += Σ grad_output（行归约，默认 f32）
        // 注：此项**无法**并入表达式——归约向量输出契约要求输出链只经归约/
        // 广播视图访问输入，而此处必须同时引用外部累加张量 grad_b_（Linear
        // 视图），两者的语义冲突（见 eval_expr_reduce 的前置校验）。故保留
        // "归约原语 + 累加原语"两步。
        auto gb = engine.row_reduce_sum(grad_output, p_.compute);
        if (!gb) return std::unexpected(gb.error());
        nn_dbg_scan("lin.row_sum(grad_out)", engine, *gb);
        auto r2 = engine.accumulate(grad_b_, *gb);
        if (!r2) return std::unexpected(r2.error());
        nn_dbg_scan("lin.grad_b(post-accum)", engine, grad_b_);

        return grad_input;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ReLU — ReLU 激活函数
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = max(x, 0)
//   backward: grad_x = (x > 0) ? grad_out : 0
// ══════════════════════════════════════════════════════════════════════════
class ReLU final : public Layer
{
private:
    Tensor input_cache_;

public:
    ReLU() = default;

    void clear_cache() override { input_cache_ = Tensor{}; }

    // ── forward: out = max(x, 0) ──────────────────────────────────────────
    // ReLU 算法由 Layer 表达为 Max 原语 + 标量 0
    // Engine/Shader 只提供 Max 原语，不知道 "ReLU" 是什么
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (!checkpoint_mode_)
            input_cache_ = input;
        return dsl::compute(engine,
            dsl::max(dsl::leaf(input), Scalar{0}),
            input.rows(), input.cols(), p_.compute);
    }

    // ── backward: grad_x = (x > 0) ? grad_out : 0 ────────────────────────
    // ReLU 反向算法由 Layer 表达为 Select + Gt 原语
    // Engine/Shader 只提供 Select/Gt 原语，不知道 "ReLU backward" 是什么
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            return std::unexpected(Error{"relu backward: shape mismatch"});

        return dsl::compute(engine,
            dsl::select(dsl::leaf(input_cache_) > Scalar{0},
                        dsl::leaf(grad_output), Scalar{0}),
            grad_output.rows(), grad_output.cols(), p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GeLU — QuickGeLU 激活函数（单表达式 DSL 融合）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = x * sigmoid(β * x) = x / (1 + exp(-β·x))
//   backward: grad_x = grad_out * s * (1 + βx * (1 - s)),  s = sigmoid(βx)
//
// 用 dsl::compute 单表达式融合：sigmoid(βx) = 1/(1+exp(-βx)) 全部折叠为单个
// GPU 融合 kernel（仅 input/output 落显存，无中间 Tensor）。backward 用
// input_cache_ 重算 sigmoid（不缓存，省显存）。AOT 收集由 scan_exprs dry-run
// 本层完成（GPU 闭合世界两端一致）。
// ══════════════════════════════════════════════════════════════════════════
class GeLU final : public Layer
{
private:
    static constexpr Scalar BETA = 1.702f;
    Tensor input_cache_;

public:
    GeLU() = default;

    void clear_cache() override
    {
        input_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward: out = x * sigmoid(β * x) ────────────────────────────────
    // 单表达式 DSL 融合（M 融合）：sigmoid(βx) = 1/(1+exp(-βx))，全部折叠为
    // 单个 GPU 融合 kernel（仅 input/output 落显存，无中间 Tensor）。表达式
    // 文本只写在本 Layer；AOT 收集由 scan_exprs dry-run 本方法完成。
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (!checkpoint_mode_)
            input_cache_ = input;
        const Scalar beta = BETA;
        // out = x * sigmoid(βx) = x / (1 + exp(-β·x))
        return dsl::compute(engine,
            dsl::leaf(input) / (Scalar{1} + dsl::exp(-(dsl::leaf(input) * beta))),
            input.rows(), input.cols(), p_.compute);
    }

    // ── backward: grad_x = grad_out * factor ─────────────────────────────
    // factor = s * (1 + βx * (1 - s)),  s = sigmoid(βx) = 1/(1+exp(-βx))
    // 单表达式 DSL 融合：sigmoid 用 input_cache_ 重算（不缓存，省显存），
    // βx 子表达式由 DSL 的 CSE 复用，消除中间 Tensor。
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            return std::unexpected(Error{"gelu backward: shape mismatch"});

        const Scalar beta = BETA;
        // s = 1/(1+exp(-βx))；factor = s * (1 + βx*(1-s))；out = grad_out * factor
        auto bx = dsl::leaf(input_cache_) * beta;                    // βx
        auto s  = Scalar{1} / (Scalar{1} + dsl::exp(-bx));          // sigmoid(βx)
        auto factor = s * (Scalar{1} + bx * (Scalar{1} - s));       // 1+βx(1-s)
        return dsl::compute(engine,
            dsl::leaf(grad_output) * factor,
            grad_output.rows(), grad_output.cols(), p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SwiGLU — Swish-Gated Linear Unit（LLaMA/Mistral 风格 FFN 激活）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  输入 (2*d_ff, batch)，前 d_ff 行为 gate，后 d_ff 行为 up
//             gate = slice_rows(x, 0, d_ff)
//             up   = slice_rows(x, d_ff, d_ff)
//             sw = SiLU(gate) = gate * sigmoid(gate)
//             out = sw ⊙ up                       → (d_ff, batch)
//   backward: grad_up   = grad_out ⊙ sw
//             grad_sw   = grad_out ⊙ up
//             grad_gate = grad_sw * (σ(g) + g*σ(g)*(1-σ(g)))
//             grad_input 把 grad_gate 写回行 [0, d_ff)、grad_up 写回 [d_ff, 2*d_ff)
//
// 原语分解（Engine/Shader 只知道标量原语）：
//   SiLU forward:  Neg → Exp → Add(1) → Div(1/x) → Mul(x*s)
//   SiLU backward: Sub → Mul → Mul → Add → Mul
//   split/merge 用 slice_rows / insert_rows
// ══════════════════════════════════════════════════════════════════════════
class SwiGLU final : public Layer
{
private:
    std::size_t d_ff_ = 0;
    Tensor input_cache_;  // 前向输入 (2*d_ff, batch)：backward 据此重算 gate/up/s（全融合，不缓存中间张量）

public:
    SwiGLU() = default;
    explicit SwiGLU(std::size_t d_ff) : d_ff_(d_ff) {}

    void clear_cache() override
    {
        input_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward: out = SiLU(gate) ⊙ up = gate·σ(gate)·up ────────────────────
    // 单表达式 DSL 融合：gate/up 用 RowAccess 行视图读取同一 (2*d_ff, batch)
    // 输入（gate = row_access(in, 0, d_ff)，up = row_access(in, d_ff, d_ff)），
    // 不再 slice_rows 物化半张量 → 消去 2 次 D2D 拷贝。输出 (d_ff, batch)。
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        if (!checkpoint_mode_)
            input_cache_ = input;
        const std::size_t cols = input.cols();
        // gate[r] = in[r % d_ff]（r<d_ff 即 in[r]）；up[r] = in[d_ff + r % d_ff]
        const auto gv = dsl::row_access(input, 0u, dff);
        const auto uv = dsl::row_access(input, dff, dff);
        // out = gate · σ(gate) · up；σ(g) = 1/(1+exp(-g))
        return dsl::compute(engine,
            gv * (Scalar{1} / (Scalar{1} + dsl::exp(-gv))) * uv,
            d_ff_, cols, p_.compute);
    }

    // ── backward: 单 kernel 融合，消去 create+zero+2×insert_rows ──────────
    //
    // 数学：
    //   s(node)     = σ(gate[node])
    //   grad_gate   = grad_out ⊙ up ⊙ s ⊙ (1 + gate ⊙ (1 − s))
    //   grad_up     = grad_out ⊙ gate ⊙ s
    //
    // 输出 grad_input (2*d_ff, batch) 分两半写回：
    //   r ∈ [0, d_ff)       → grad_gate[r]
    //   r ∈ [d_ff, 2*d_ff)  → grad_up[r − d_ff]
    // 用 select(Row() < d_ff, grad_gate_expr, grad_up_expr) 分半；gate/up/s/go
    // 均经 RowAccess(offset=0|d_ff, mod=d_ff) 行视图按 r % d_ff 定位到对应半，
    // 两半表达式对"错误"半只会算出越界内但被 Select 丢弃的值，正确性无虞。
    // （共享的 gate/s 子表达式在树型 DSL 中会重复折叠，故 EXPR_MAX_REGS/INPUTS
    //   已相应放宽，换取零中间张量、零拷贝。）
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        const std::size_t rows = 2 * d_ff_;
        const std::size_t cols = grad_output.cols();

        if (input_cache_.rows() != rows || input_cache_.cols() != cols)
            return std::unexpected(Error{"swiglu backward: input_cache shape mismatch"});
        if (grad_output.rows() != d_ff_ || grad_output.cols() != cols)
            return std::unexpected(Error{"swiglu backward: grad_output shape mismatch"});

        const auto go   = dsl::row_access(grad_output, 0u, dff);   // go[r % d_ff]
        const auto gate = dsl::row_access(input_cache_, 0u, dff);   // in[r % d_ff]
        const auto up   = dsl::row_access(input_cache_, dff, dff);  // in[d_ff + r % d_ff]
        const auto s    = Scalar{1} / (Scalar{1} + dsl::exp(-gate)); // σ(gate)

        // factor = s·(1 + gate·(1−s))；grad_gate = go·up·factor；grad_up = go·gate·s
        const auto factor = s * (Scalar{1} + gate * (Scalar{1} - s));
        const auto gg = go * up * factor;
        const auto gu = go * gate * s;
        return dsl::compute(engine,
            dsl::select(dsl::row() < dsl::rparam(static_cast<nn::Scalar>(d_ff_)), gg, gu),
            rows, cols, p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// LayerNorm — 层归一化
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  mean = (1/F) * Σ_f x[f][b]              (col_reduce_sum + scale)
//             diff = x - mean                          (broadcast_col Sub)
//             var = (1/F) * Σ_f diff²                  (elementwise Mul + col_reduce_sum + scale)
//             std_inv = 1 / sqrt(var + eps)            (binary_scalar Add + unary Rsqrt)
//             normalized = diff * std_inv              (broadcast_col Mul)
//             out = gamma * normalized + beta          (broadcast_row Mul + broadcast_row Add)
//
//   backward: gy = grad_out * gamma                    (broadcast_row Mul)
//             mean_g = (1/F) * Σ_f gy                  (col_reduce_sum + scale)
//             gy_norm = gy * normalized                (elementwise Mul)
//             mean_gn = (1/F) * Σ_f gy_norm            (col_reduce_sum + scale)
//             grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
//             grad_gamma += Σ_b gy_norm                (row_reduce_sum)
//             grad_beta += Σ_b grad_out                (row_reduce_sum)
// ══════════════════════════════════════════════════════════════════════════
class LayerNorm final : public Layer
{
private:
    std::size_t normalized_shape_;
    Scalar epsilon_;

    // 可学习参数
    Tensor gamma_;      // (normalized_shape, 1)
    Tensor beta_;       // (normalized_shape, 1)
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // backward 缓存
    Tensor normalized_cache_;  // (features, batch)
    Tensor std_cache_;         // (1, batch) — std_inv

    static constexpr Scalar EPSILON = 1e-5;

public:
    explicit LayerNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // gamma 初始化为 1, beta 初始化为 0
        Matrix gamma_cpu(normalized_shape_, 1, Scalar{1});
        Matrix beta_cpu(normalized_shape_, 1, Scalar{0});

        auto g = engine.from_matrix(gamma_cpu, p_.param);
        if (!g) return std::unexpected(g.error());
        gamma_ = std::move(*g);

        auto bv = engine.from_matrix(beta_cpu, p_.param);
        if (!bv) return std::unexpected(bv.error());
        beta_ = std::move(*bv);

        grad_gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param);
        grad_beta_ = engine.create_tensor(normalized_shape_, 1, p_.param);
        { auto r1 = engine.zero(grad_gamma_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_beta_);  if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {gamma_, beta_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_gamma_, grad_beta_};
    }

    void clear_cache() override
    {
        normalized_cache_ = Tensor{};
        std_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (normalized_cache_.valid()) r.emplace_back(normalized_cache_);
        if (std_cache_.valid()) r.emplace_back(std_cache_);
        return r;
    }

    // ── forward ───────────────────────────────────────────────────────────
    // M3 融合（算法公式不变，diff_sq (F,B) 由归约 kernel 内部消解）：
    //   融合表达式保持 F 无关结构（不含 1/F、ε 常量）；形状相关标量在
    //   (1,B) 小向量上用引擎原语施加：
    //   1. mean_raw = col_reduce_sum(x)                    → (1,B) 归约向量输出
    //   2. mean     = mean_raw*(1/F)
    //   3. diff     = x - mean (col 广播)                  → (F,B) 融合逐元素
    //   4. var_raw  = col_reduce_sum(diff²)                → (1,B) 归约向量输出
    //   5. std_inv  = rsqrt(var_raw*(1/F) + ε)             → (1,B) 原语
    //   6. normalized=diff * std_inv (col 广播)            → (F,B) 融合逐元素
    //   7. out      = normalized*gamma + beta (row 广播)   → (F,B) 融合逐元素
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != normalized_shape_)
            return std::unexpected(Error{"layernorm forward: input shape mismatch"});

        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. mean_raw = col_reduce_sum(x) → (1,B)
        auto mean_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input)), F, B, p_.stable);
        if (!mean_raw) return std::unexpected(mean_raw.error());

        // 2. mean = mean_raw*(1/F) → (1,B)
        // （1/F 是形状相关标量，由 RParam 承载：**值不进 expr_spec_key**，
        //   融合表达式保持 F 无关 → 不同归一化维度共享同一 AOT 融合 shader）
        auto mean = dsl::compute(engine,
            dsl::leaf(*mean_raw) * dsl::rparam(inv_features),
            mean_raw->rows(), mean_raw->cols(), p_.stable);
        if (!mean) return std::unexpected(mean.error());

        // 3. diff = x - mean (col 广播) → (F,B)
        auto diff = dsl::compute(engine,
            dsl::leaf(input) - dsl::col_broadcast(*mean), F, B, p_.stable);
        if (!diff) return std::unexpected(diff.error());

        // 4. var_raw = col_reduce_sum(diff²) → (1,B)
        auto var_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(*diff) * dsl::leaf(*diff)), F, B, p_.stable);
        if (!var_raw) return std::unexpected(var_raw.error());

        // 5. std_inv = rsqrt(var_raw*(1/F) + ε) → (1,B)
        // （"乘 1/F → 加 ε → rsqrt"三步塌成单表达式、单次遍历；1/F、ε 由
        //   RParam 承载，表达式结构仍与 F / ε 取值无关）
        auto std_inv = dsl::compute(engine,
            dsl::rsqrt(dsl::leaf(*var_raw) * dsl::rparam(inv_features)
                       + dsl::rparam(epsilon_)),
            var_raw->rows(), var_raw->cols(), p_.stable);
        if (!std_inv) return std::unexpected(std_inv.error());
        Tensor std_inv_t = std::move(*std_inv);
        if (!checkpoint_mode_)
            std_cache_ = std_inv_t;

        // 6. normalized = diff * std_inv (col 广播) → (F,B)
        auto normalized = dsl::compute(engine,
            dsl::leaf(*diff) * dsl::col_broadcast(std_inv_t), F, B, p_.stable);
        if (!normalized) return std::unexpected(normalized.error());
        Tensor normalized_t = std::move(*normalized);
        if (!checkpoint_mode_)
            normalized_cache_ = normalized_t;

        // 7. out = normalized*gamma + beta (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
            + dsl::row_broadcast(beta_),
            F, B, p_.stable);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
    //   gy = grad_out * gamma
    //   mean_g  = col_reduce_sum(gy) * invF
    //   mean_gn = col_reduce_sum(gy * normalized) * invF
    // grad_gamma += row_sum(gy ⊙ normalized)
    // grad_beta  += row_sum(grad_out)
    // M3 融合（融合表达式 F 无关，1/F 在 (1,B) 上用原语施加）：
    //   mean_g/mean_gn 为列归约向量输出；(F,B) 全尺寸中间量由融合 kernel 消解。
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. mean_g_raw = col_reduce_sum(gy) → (1,B)
        auto mg_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)),
            F, B, p_.stable);
        if (!mg_raw) return std::unexpected(mg_raw.error());
        auto mean_g = dsl::compute(engine,
            dsl::leaf(*mg_raw) * dsl::rparam(inv_features),
            mg_raw->rows(), mg_raw->cols(), p_.stable);
        if (!mean_g) return std::unexpected(mean_g.error());

        // 2. mean_gn_raw = col_reduce_sum(gy ⊙ normalized) → (1,B)
        auto mgn_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        if (!mgn_raw) return std::unexpected(mgn_raw.error());
        auto mean_gn = dsl::compute(engine,
            dsl::leaf(*mgn_raw) * dsl::rparam(inv_features),
            mgn_raw->rows(), mgn_raw->cols(), p_.stable);
        if (!mean_gn) return std::unexpected(mean_gn.error());

        // 3. grad_x = (gy - mean_g - normalized*mean_gn) * std_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*mean_g)
             - dsl::leaf(normalized_cache_) * dsl::col_broadcast(*mean_gn))
            * dsl::col_broadcast(std_cache_),
            F, B, p_.stable);
        if (!grad_x) return std::unexpected(grad_x.error());

        // 4. grad_gamma += row_reduce_sum(gy ⊙ normalized) → (F,1)
        //    （∂L/∂γ_f = Σ_b gy·n：γ 是 out 的线性因子，导数不含 γ）
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output)
                * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        if (!gg) return std::unexpected(gg.error());
        auto grad_gamma_acc = dsl::compute_into(engine,
            dsl::leaf(grad_gamma_) + dsl::leaf(*gg), grad_gamma_);
        if (!grad_gamma_acc) return std::unexpected(grad_gamma_acc.error());

        // 5. grad_beta += row_reduce_sum(grad_out) → (F,1)
        auto gb = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)), F, B, p_.stable);
        if (!gb) return std::unexpected(gb.error());
        auto grad_beta_acc = dsl::compute_into(engine,
            dsl::leaf(grad_beta_) + dsl::leaf(*gb), grad_beta_);
        if (!grad_beta_acc) return std::unexpected(grad_beta_acc.error());

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RMSNorm — Root Mean Square 归一化（LLaMA/Mistral 风格）
//
// 与 LayerNorm 的差异：不减去均值、无 beta 偏置，只按均方根归一化。
// 每层少 2 次列归约 + 1 次广播，计算量更小，训练更稳。
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  mean_sq = (1/F) * Σ_f x²             (elementwise Mul + col_reduce_sum + scale)
//             rms_inv = 1 / sqrt(mean_sq + eps)    (binary_scalar Add + unary Rsqrt)
//             normed  = x * rms_inv                (broadcast_col Mul)
//             out     = normed * gamma             (broadcast_row Mul)
//   backward: gy       = grad_out * gamma          (broadcast_row Mul)
//             gy_norm  = gy ⊙ normed               (elementwise Mul)
//             m        = (1/F) * Σ_f gy_norm       (col_reduce_sum + scale)
//             grad_x   = (gy - m*normed) * rms_inv (broadcast_col Mul + Sub + Mul)
//             grad_gamma += row_reduce_sum(gy_norm) (row_reduce_sum)
// ══════════════════════════════════════════════════════════════════════════
class RMSNorm final : public Layer
{
private:
    std::size_t normalized_shape_;
    Scalar epsilon_;

    Tensor gamma_;        // (normalized_shape, 1)
    Tensor grad_gamma_;

    // backward 缓存
    Tensor normed_cache_;   // (features, batch)
    Tensor rms_inv_cache_;  // (1, batch)

    static constexpr Scalar EPSILON = 1e-5;

public:
    explicit RMSNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // gamma 初始化为 1（无 beta）
        Matrix gamma_cpu(normalized_shape_, 1, Scalar{1});
        auto g = engine.from_matrix(gamma_cpu, p_.param);
        if (!g) return std::unexpected(g.error());
        gamma_ = std::move(*g);

        grad_gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param);
        { auto r1 = engine.zero(grad_gamma_); if (!r1) return std::unexpected(r1.error()); }
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {gamma_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_gamma_};
    }

    void clear_cache() override
    {
        normed_cache_ = Tensor{};
        rms_inv_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (normed_cache_.valid()) r.emplace_back(normed_cache_);
        if (rms_inv_cache_.valid()) r.emplace_back(rms_inv_cache_);
        return r;
    }

    // ── forward ───────────────────────────────────────────────────────────
    // M3 融合（算法公式不变，中间 x_sq (F,B) 由归约 kernel 内部消解）：
    //   融合表达式保持 F 无关结构（不含 1/F、ε 常量，避免闭合世界 key 随
    //   归一化维度漂移）；形状相关标量在 (1,B) 小向量上用引擎原语施加：
    //   1. s_raw  = col_reduce_sum(x*x)                    → (1,B) 归约向量输出
    //   2. rms_inv= rsqrt(s_raw*(1/F) + ε)                 → (1,B) 原语
    //   3. normed = x * rms_inv (col 广播)                 → (F,B) 融合逐元素
    //   4. out    = normed * gamma (row 广播)              → (F,B) 融合逐元素
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != normalized_shape_)
            return std::unexpected(Error{"rmsnorm forward: input shape mismatch"});

        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. s_raw = col_reduce_sum(x*x) → (1,B)
        auto s_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input) * dsl::leaf(input)), F, B, p_.stable);
        if (!s_raw) return std::unexpected(s_raw.error());

        // 2. rms_inv = rsqrt(s_raw*(1/F) + ε) → (1,B)
        // （"乘 1/F → 加 ε → rsqrt"三步塌成单表达式、单次遍历；1/F、ε 由
        //   RParam 承载：值不进 expr_spec_key → 表达式结构保持 F/ε 无关）
        auto rms_inv = dsl::compute(engine,
            dsl::rsqrt(dsl::leaf(*s_raw) * dsl::rparam(inv_features)
                       + dsl::rparam(epsilon_)),
            s_raw->rows(), s_raw->cols(), p_.stable);
        if (!rms_inv) return std::unexpected(rms_inv.error());
        Tensor rms_inv_t = std::move(*rms_inv);
        if (!checkpoint_mode_)
            rms_inv_cache_ = rms_inv_t;

        // 3. normed = x * rms_inv (col 广播) → (F,B)
        auto normed = dsl::compute(engine,
            dsl::leaf(input) * dsl::col_broadcast(rms_inv_t), F, B, p_.stable);
        if (!normed) return std::unexpected(normed.error());
        Tensor normed_t = std::move(*normed);
        if (!checkpoint_mode_)
            normed_cache_ = normed_t;

        // 4. out = normed * gamma (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normed_t) * dsl::row_broadcast(gamma_), F, B, p_.stable);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // M3 融合（融合表达式 F 无关，1/F 在 (1,B) 小向量上用原语施加）：
    //   gy       = grad * gamma
    //   m_raw    = col_reduce_sum(gy ⊙ normed)             → (1,B) 归约向量输出
    //   m        = m_raw * (1/F)
    //   grad_x   = (gy - m*normed) * rms_inv               → (F,B) 融合逐元素
    //   grad_gamma += row_reduce_sum(gy ⊙ normed)          → (F,1) 归约向量输出
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. m_raw = col_reduce_sum(gy ⊙ normed) → (1,B)
        auto m_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normed_cache_)),
            F, B, p_.stable);
        if (!m_raw) return std::unexpected(m_raw.error());
        auto m = dsl::compute(engine,
            dsl::leaf(*m_raw) * dsl::rparam(inv_features),
            m_raw->rows(), m_raw->cols(), p_.stable);
        if (!m) return std::unexpected(m.error());

        // 2. grad_x = (gy - m*normed) * rms_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*m) * dsl::leaf(normed_cache_))
            * dsl::col_broadcast(rms_inv_cache_),
            F, B, p_.stable);
        if (!grad_x) return std::unexpected(grad_x.error());

        // 3. grad_gamma += row_reduce_sum(gy ⊙ normed) → (F,1)
        //    （∂L/∂γ_f = Σ_b gy·n：γ 是 out 的线性因子，导数不含 γ）
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output)
                * dsl::leaf(normed_cache_)),
            F, B, p_.stable);
        if (!gg) return std::unexpected(gg.error());
        auto grad_gamma_acc = dsl::compute_into(engine,
            dsl::leaf(grad_gamma_) + dsl::leaf(*gg), grad_gamma_);
        if (!grad_gamma_acc) return std::unexpected(grad_gamma_acc.error());

        return grad_x;
    }
};

// ── 归一化层工厂：按 NormType 创建 LayerNorm 或 RMSNorm ──────────────────
[[nodiscard]] inline std::unique_ptr<Layer> make_norm_layer(
    std::size_t d_model, NormType norm_type)
{
    if (norm_type == NormType::RMSNorm)
        return std::make_unique<RMSNorm>(d_model);
    return std::make_unique<LayerNorm>(d_model);
}

// ══════════════════════════════════════════════════════════════════════════
// Conv2D — 二维卷积层（引擎化：im2col + matmul 复用矩阵乘内核）
//
// 布局约定（与项目 batch-major 列布局一致）：
//   输入  Tensor: (C_in * H_in * W_in, batch)
//   输出  Tensor: (C_out * H_out * W_out, batch)
//   权重  w_: (C_out, C_in * k * k)
//   偏置  b_: (C_out, 1)
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  im2col(x) → col (C_in*k*k, batch*OH*OW)
//             Z = W × col + b  → (C_out, batch*OH*OW)
//             重排 → (C_out*OH*OW, batch)   （恢复 batch-major 列布局）
//   backward: grad_W += gZ × col^T
//             grad_b += row_reduce_sum(gZ)
//             grad_col = W^T × gZ
//             col2im → (C_in*H*W, batch)
//
// 说明：im2col/col2im 涉及复杂重排，沿用 PatchEmbedding 的先例在 CPU 端完成
//       （to_matrix/from_matrix），GEMM 仍复用引擎 matmul 内核。
//       MNIST 尺度下 CPU↔设备往返开销可忽略；GPU 融合卷积内核留作后续优化。
} // namespace nn

