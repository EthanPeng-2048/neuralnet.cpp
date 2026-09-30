#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_attention.hpp"
#include "compute_layer_feedforward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{

class PositionalEncoding final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t max_len_;
    std::size_t tile_size_;   // >0: 启用 tiling（每个样本的序列长度）
    Tensor encoding_cache_;
    std::size_t cached_total_ = 0;

    [[nodiscard]] Result<void> rebuild_encoding(ComputeEngine& engine, std::size_t total_len)
    {
        const std::size_t base_len = (tile_size_ > 0) ? tile_size_ : total_len;
        const std::size_t batch    = (tile_size_ > 0) ? total_len / tile_size_ : 1;

        std::vector<Scalar> enc(d_model_ * total_len);   // 宿主桥（17 §3 D11）
        const std::size_t half = d_model_ / 2;
        std::vector<Scalar> freqs(half);
        for (std::size_t i = 0; i < half; ++i)
            freqs[i] = Scalar{1} / std::pow(Scalar{10000},
                static_cast<Scalar>(2 * i) / static_cast<Scalar>(d_model_));

        for (std::size_t b = 0; b < batch; ++b)
        {
            const std::size_t col_off = b * base_len;
            for (std::size_t pos = 0; pos < base_len; ++pos)
            {
                const Scalar pos_d = static_cast<Scalar>(pos);
                for (std::size_t i = 0; i < half; ++i)
                {
                    const Scalar angle = pos_d * freqs[i];
                    enc[(2 * i)     * total_len + col_off + pos] = std::sin(angle);
                    enc[(2 * i + 1) * total_len + col_off + pos] = std::cos(angle);
                }
                if (d_model_ % 2 == 1)
                {
                    const Scalar freq_last = Scalar{1} / std::pow(Scalar{10000},
                        static_cast<Scalar>(2 * half) / static_cast<Scalar>(d_model_));
                    enc[(d_model_ - 1) * total_len + col_off + pos] = std::sin(pos_d * freq_last);
                }
            }
        }
        NN_TRY(r, detail::upload_span(engine, d_model_, total_len, p_.param, std::span(enc)));
        encoding_cache_ = std::move(*r);
        cached_total_ = total_len;
        return {};
    }

public:
    PositionalEncoding(std::size_t d_model, std::size_t max_len = 5000,
                       std::size_t tile_size = 0)
        : d_model_(d_model), max_len_(max_len), tile_size_(tile_size)
    {
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != d_model_)
            return std::unexpected(Error{"PE forward: d_model mismatch"});
        const std::size_t total_len = input.cols();
        if (tile_size_ > 0 && total_len % tile_size_ != 0)
            return std::unexpected(Error{"PE forward: total_len not divisible by tile_size"});
        if (tile_size_ == 0 && total_len > max_len_)
            return std::unexpected(Error{"PE forward: seq_len exceeds max_len"});

        if (cached_total_ != total_len)
        {
            NN_TRY(r, rebuild_encoding(engine, total_len));
        }

        return dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(encoding_cache_),
            input.rows(), input.cols(), p_.compute);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        return grad_output;  // 位置编码不可学习，梯度直接穿透
    }
};

// TransformerEncoderLayer — Pre-Norm 编码器层（批量化）
//
// 算法（只在此处，不在 Engine/Shader）：
//   x = x + SelfAttn(LN₁(x))
//   x = x + FFN(LN₂(x))
//
// 输入/输出形状：(d_model, batch * seq_len)
// seq_len > 0 时启用 MHA 批量化路径（消除 per-head 和 per-sample 循环）。
// ══════════════════════════════════════════════════════════════════════════
class TransformerEncoderLayer final : public Layer
{
private:
    MultiHeadAttention self_attn_;
    LayerNorm norm1_;
    FeedForward ff_;
    LayerNorm norm2_;

    Tensor residual2_cache_;

public:
    TransformerEncoderLayer(std::size_t d_model, std::size_t num_heads,
                            std::size_t d_ff, std::size_t seq_len = 0)
        : self_attn_(d_model, num_heads, seq_len),
          norm1_(d_model),
          ff_(d_model, d_ff),
          norm2_(d_model) {}

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, self_attn_.init(engine));
        NN_TRY(r2, norm1_.init(engine));
        NN_TRY(r3, ff_.init(engine));
        NN_TRY(r4, norm2_.init(engine));
        return {};
    }

    // ── D7：精度配置下传（§9.2）：复合层必须把 profile 给到全部子层 ──────
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        self_attn_.set_precision_profile(profile);
        norm1_.set_precision_profile(profile);
        ff_.set_precision_profile(profile);
        norm2_.set_precision_profile(profile);
    }

    std::vector<TensorRef> parameters() override
    {
        return collect_refs(self_attn_.parameters(),
                            norm1_.parameters(),
                            ff_.parameters(),
                            norm2_.parameters());
    }

    std::vector<TensorRef> param_gradients() override
    {
        return collect_refs(self_attn_.param_gradients(),
                            norm1_.param_gradients(),
                            ff_.param_gradients(),
                            norm2_.param_gradients());
    }

    // 梯度检查点：把模式传播给内部注意力/归一化/FFN
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        self_attn_.set_checkpoint_mode(enabled);
        norm1_.set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
        norm2_.set_checkpoint_mode(enabled);
    }

    [[nodiscard]] bool recompute_supported() const override { return true; }

    [[nodiscard]] Result<Tensor> forward_recompute(
        const Tensor& saved_input) override
    {
        // 临时关闭本层及其子层的 checkpoint 模式，使 forward 重建缓存
        set_checkpoint_mode(false);
        auto r = forward(saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    void clear_cache() override
    {
        self_attn_.clear_cache();
        norm1_.clear_cache();
        ff_.clear_cache();
        norm2_.clear_cache();
        residual2_cache_ = Tensor{};
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        // 残差分支 1（Pre-Norm + 自注意力），与 GPTBlock/RAPTBlock 同骨架
        NN_TRY(r2, prenorm_residual_forward_(engine, input, norm1_, self_attn_, p_.compute));
        Tensor res2 = std::move(*r2);
        if (!checkpoint_mode_)
            residual2_cache_ = res2;
        // 残差分支 2（Pre-Norm + FFN）
        return prenorm_residual_forward_(engine, res2, norm2_, ff_, p_.compute);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // 两处残差分流 + 子层反向（与 GPTBlock / RAPTBlock 同一骨架）
        return prenorm_residual_backward_(engine, grad_output,
                                          norm1_, self_attn_, norm2_, ff_,
                                          p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// TransformerEncoder — ViT 风格的 Transformer 编码器（全批量化）
//
// 算法（只在此处，不在 Engine/Shader）：
//   1. 输入 (d_model, batch * num_patches) — PatchEmbedding 输出，batch-major 列
//   2. + tiled PE → (d_model, batch * num_patches)
//   3. 一次性通过 N 个 EncoderLayer（MHA 内部用 rearrange_3d 批量化）
//   4. 全局平均池化（按样本聚合 num_patches 维度）→ (d_model, batch)
//
// 全 GPU 批量化策略：
//   - 整个 batch 同时通过所有层，无 per-sample 循环
//   - 池化用 rearrange_3d + row_reduce_sum + rearrange_3d，纯 GPU 原语
//   - 反向池化用 rearrange_3d + matmul(grad, ones_row_)，纯 GPU 原语
//   - 无 batch 边界 PCIe 传输（输入输出均为 GPU 张量流）
//
// 输入: (d_model, batch * num_patches)  — PatchEmbedding 输出
// 输出: (d_model, batch)                 — 池化后的序列表示
// ══════════════════════════════════════════════════════════════════════════
class TransformerEncoder final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_patches_;
    Scalar inv_num_patches_;
    std::vector<TransformerEncoderLayer> layers_;
    PositionalEncoding pos_encoding_;

    std::size_t batch_size_ = 0;
    Tensor ones_row_;  // (1, num_patches) 全1，用于 backward 池化梯度广播

public:
    TransformerEncoder(std::size_t d_model, std::size_t num_heads,
                       std::size_t d_ff, std::size_t num_layers,
                       std::size_t num_patches)
        : d_model_(d_model), num_patches_(num_patches),
          inv_num_patches_(Scalar{1} / static_cast<Scalar>(num_patches)),
          // PE 启用 tiling：每个样本独立使用 (d_model, num_patches) 的编码
          pos_encoding_(d_model, num_patches, num_patches)
    {
        // 所有 EncoderLayer 传入 seq_len=num_patches，启用 MHA 批量化路径
        for (std::size_t i = 0; i < num_layers; ++i)
            layers_.emplace_back(d_model, num_heads, d_ff, num_patches);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        for (auto& layer : layers_)
        {
            NN_TRY(r, layer.init(engine));
        }
        // M6 段 C：pos_encoding_ 是子 Layer，engine 由 init 绑定
        {
            NN_TRY(r, pos_encoding_.init(engine));
        }
        // 预创建 ones_row_ (1, num_patches) 全1，用于 backward 广播
        // （M2 声明式：引擎填数，原 from_matrix 口径 = F32）
        ones_row_ = engine.create_tensor(1, num_patches_, Precision::F32, InitSpec::constant(1));
        if (!ones_row_.valid())
            return std::unexpected(Error{"TransformerEncoder: ones_row_ 初始化失败"});
        return {};
    }

    // ── D7：精度配置下传（§9.2）：逐层 EncoderLayer + 位置编码 ──────────
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        for (auto& l : layers_) l.set_precision_profile(profile);
        pos_encoding_.set_precision_profile(profile);
    }

    std::vector<TensorRef> parameters() override
    {
        return collect_block_refs_(layers_, &TransformerEncoderLayer::parameters);
    }

    std::vector<TensorRef> param_gradients() override
    {
        return collect_block_refs_(layers_, &TransformerEncoderLayer::param_gradients);
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        // input: (d_model, batch * num_patches)
        if (input.rows() != d_model_)
            return std::unexpected(Error{"TransformerEncoder: row count mismatch"});
        if (input.cols() % num_patches_ != 0)
            return std::unexpected(Error{"TransformerEncoder: cols not divisible by num_patches"});
        batch_size_ = input.cols() / num_patches_;

        // 1. 添加 tiled 位置编码 → (d_model, batch * num_patches)
        NN_TRY(pe, pos_encoding_.forward(input));
        Tensor x = std::move(*pe);

        // 2. 一次性通过所有 EncoderLayer [全批量化 GPU]
        for (auto& layer : layers_)
        {
            NN_TRY(lr, layer.forward(x));
            x = std::move(*lr);
        }
        // x: (d_model, batch * num_patches)

        // 3. 全局平均池化（按样本聚合 num_patches 维度）→ (d_model, batch)
        //   使用 rearrange_3d 把 (d_model, batch*num_patches) 重排为
        //   (batch*d_model, num_patches)，每行块对应一个样本，
        //   row_reduce_sum 后得到 (batch*d_model, 1)，
        //   再 rearrange_3d inverse 回 (d_model, batch)。
        NN_TRY(re, engine.rearrange_3d(x, d_model_, batch_size_, num_patches_, false));
        // 归约步走 dsl::compute_reduce（输出与输入同形）
        auto row_sum = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(*re)), re->rows(), re->cols(), p_.compute);
        NN_TRY_CHECK(row_sum);
        NN_TRY(rs_re, engine.rearrange_3d(*row_sum, d_model_, batch_size_, 1, true));
        // 原地缩放走 dsl::compute_into（inv_num_patches_ 走 rparam，值不进 key）
        auto r = dsl::compute_into(engine,
            dsl::leaf(*rs_re) * dsl::rparam(inv_num_patches_), *rs_re);
        NN_TRY_CHECK(r);
        return *rs_re;
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // grad_output: (d_model, batch) — 来自下游 Linear.backward

        // 1. 反向池化：(d_model, batch) → (d_model, batch * num_patches)
        //   每个样本的池化梯度广播到 num_patches 列：
        //   grad_x[d, b*num_patches + p] = grad_out[d, b] * inv_num_patches
        //   实现：rearrange_3d(grad, d_model, batch, 1, false) → (batch*d_model, 1)
        //         原地 scale(inv_n) → matmul(grad_col_vec, ones_row_) → (batch*d_model, num_patches)
        //         rearrange_3d(result, d_model, batch, num_patches, true) → (d_model, batch*num_patches)
        NN_TRY(g_re, engine.rearrange_3d(grad_output, d_model_, batch_size_, 1, false));
        // 原地缩放走 dsl::compute_into（inv_num_patches_ 走 rparam，值不进 key）
        auto r = dsl::compute_into(engine,
            dsl::leaf(*g_re) * dsl::rparam(inv_num_patches_), *g_re);
        NN_TRY_CHECK(r);
        // (*g_re): (batch*d_model, 1) × ones_row_ (1, num_patches) → (batch*d_model, num_patches)
        // 纯 matmul 走 DSL 直写（结构经 scan 的 TransformerEncoderLayer dry-run 登记；
        // 注意该调用在 pooling 路径，scan 的 enc dry-run 覆盖 forward/backward）
        auto unpooled = dsl::compute(engine,
            dsl::matmul(*g_re, ones_row_, false, false),
            g_re->rows(), ones_row_.cols(), p_.compute);
        NN_TRY_CHECK(unpooled);
        NN_TRY(grad, engine.rearrange_3d(*unpooled, d_model_, batch_size_, num_patches_, true));
        Tensor grad_x = std::move(*grad);

        // 2. 反向通过所有 EncoderLayer [全批量化 GPU]
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        {
            NN_TRY(br, it->backward(grad_x));
            grad_x = std::move(*br);
        }

        // 3. 反向 PE（梯度直接穿透）
        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// PatchEmbedding — 图像 patch 嵌入
//
// 算法（只在此处，不在 Engine/Shader）：
//   1. 将 (img_size², batch) 展平图像提取 num_patches 个不重叠 patch
//   2. 每个展平 patch (patch_size²) 经 Linear 投影到 d_model 维
//   3. 重排为 (d_model, batch * num_patches) —— batch-major 列布局
//      [r, b * num_patches + p] = 样本 b 的 patch p 的第 r 维特征
//      这样下游 TransformerEncoder 可整体批量化（消除 per-sample 循环）。
//
// 注意: patch 提取涉及复杂重排，此处用 to_matrix/from_matrix 在 CPU 端
//       完成（batch 边界，PCIe 传输符合纯 GPU 架构约定）。
// ══════════════════════════════════════════════════════════════════════════
class PatchEmbedding final : public Layer
{
private:
    std::size_t img_size_;
    std::size_t patch_size_;
    std::size_t grid_size_;
    std::size_t num_patches_;
    std::size_t patch_dim_;
    std::size_t d_model_;
    Linear projection_;
    Tensor input_cache_;

public:
    PatchEmbedding(std::size_t img_size, std::size_t patch_size,
                   std::size_t d_model)
        : img_size_(img_size), patch_size_(patch_size),
          grid_size_(img_size / patch_size),
          num_patches_(grid_size_ * grid_size_),
          patch_dim_(patch_size * patch_size),
          d_model_(d_model),
          projection_(patch_dim_, d_model)
    {
        NN_ASSERT(img_size % patch_size == 0,
                  "PatchEmbedding: img_size must be divisible by patch_size");
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        return projection_.init(engine);
    }

    // ── D7：精度配置下传（§9.2）：投影层是复合层的唯一子层 ───────────────
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        projection_.set_precision_profile(profile);
    }

    [[nodiscard]] std::size_t num_patches() const noexcept { return num_patches_; }
    [[nodiscard]] std::size_t d_model()     const noexcept { return d_model_; }

    std::vector<TensorRef> parameters() override
    { return projection_.parameters(); }

    std::vector<TensorRef> param_gradients() override
    { return projection_.param_gradients(); }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        input_cache_ = input;
        const std::size_t batch = input.cols();

        // Step 1: CPU 端提取 patches，直接生成 batch-major 列布局
        //   all_patches[flat, b * num_patches + p] = image_b[pixel(flat, p)]
        //   样本 b 占连续列块 [b*num_patches, (b+1)*num_patches)，
        //   便于下游 MHA 的 rearrange_3d 按 batch 切分列块。
        // （patch 提取是 PatchEmbedding 的算法职责，无对应 op-level 原语；
        //  此处为 batch 边界的合法 CPU 预处理，与 GPTModel 的 gather_rows 同性质；
        //  宿主桥走 span，不经 Matrix——17 §3 D11）
        NN_TRY(in_v, detail::download_vector(engine, input));
        const std::size_t in_cols = input.cols();

        std::vector<Scalar> all_patches(patch_dim_ * batch * num_patches_);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t p = 0; p < num_patches_; ++p)
            {
                const std::size_t gr = (p / grid_size_) * patch_size_;
                const std::size_t gc = (p % grid_size_) * patch_size_;
                const std::size_t col_idx = b * num_patches_ + p;  // batch-major
                for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    for (std::size_t pc = 0; pc < patch_size_; ++pc)
                    {
                        const std::size_t flat = pr * patch_size_ + pc;
                        const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                        all_patches[flat * (batch * num_patches_) + col_idx] =
                            (*in_v)[pix * in_cols + b];
                    }
            }
        }

        auto ap_t = detail::upload_span(engine, patch_dim_, batch * num_patches_,
                                        Precision::F32, std::span(all_patches));
        NN_TRY_CHECK(ap_t);

        // Step 2: 投影 → (d_model, batch * num_patches) — 已是 batch-major，无需重排
        return projection_.forward(*ap_t);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // grad_output: (d_model, batch * num_patches) — batch-major
        const std::size_t batch = grad_output.cols() / num_patches_;

        // Step 1: 投影层反向 → (patch_dim, batch * num_patches) — batch-major
        NN_TRY(bp, projection_.backward(grad_output));

        // Step 2: 散射梯度回输入 → (img_size², batch)
        auto gp_v = detail::download_vector(engine, *bp);   // 宿主桥（17 §3 D11）
        NN_TRY_CHECK(gp_v);
        const std::size_t gp_cols = bp->cols();

        std::vector<Scalar> grad_input(img_size_ * img_size_ * batch);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t p = 0; p < num_patches_; ++p)
            {
                const std::size_t gr = (p / grid_size_) * patch_size_;
                const std::size_t gc = (p % grid_size_) * patch_size_;
                const std::size_t col_idx = b * num_patches_ + p;  // batch-major
                for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    for (std::size_t pc = 0; pc < patch_size_; ++pc)
                    {
                        const std::size_t flat = pr * patch_size_ + pc;
                        const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                        const std::size_t gi   = pix * batch + b;
                        grad_input[gi] = grad_input[gi]
                                       + (*gp_v)[flat * gp_cols + col_idx];
                    }
            }
        }
        return detail::upload_span(engine, img_size_ * img_size_, batch,
                                   Precision::F32, std::span(grad_input));
    }
};

} // namespace nn

