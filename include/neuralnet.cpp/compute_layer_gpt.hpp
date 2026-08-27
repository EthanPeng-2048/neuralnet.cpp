#ifndef NN_COMPUTE_LAYER_GPT_HPP
#define NN_COMPUTE_LAYER_GPT_HPP

#include "compute_layer_base.hpp"
#include "compute_layer_attention.hpp"

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
class GPTBlock final : public Layer
{
private:
    CausalSelfAttention self_attn_;
    std::unique_ptr<Layer> norm1_;
    FeedForward ff_;
    std::unique_ptr<Layer> norm2_;

    Tensor residual2_cache_;

    // ── activation offload（L1-offload）状态 ──
    bool offload_enabled_ = false;      // GPTModel 是否启用 offload
    bool offloaded_ = false;            // 当前激活是否已导出到 host
    Tensor offload_slab_;               // 持久 host-visible 缓冲（跨 step 复用）
    std::vector<TensorRef> offload_refs_;        // 需 offload 的缓存成员引用（稳定地址）
    std::vector<std::pair<std::size_t, std::size_t>> offload_shapes_;  // 各缓存形状
    std::vector<std::size_t> offload_offsets_;   // 各激活在 slab 中的 float 偏移

public:
    GPTBlock(std::size_t d_model, std::size_t num_heads,
             std::size_t d_ff, std::size_t max_len = 1024,
             std::size_t seq_len = 0,
             PosEncodingType pos_enc = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm)
        : self_attn_(d_model, num_heads, max_len, seq_len, pos_enc),
          norm1_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation),
          norm2_(make_norm_layer(d_model, norm_type)) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = self_attn_.init(engine); if (!r1) return std::unexpected(r1.error());
        if (norm1_) { auto r = norm1_->init(engine); if (!r) return std::unexpected(r.error()); }
        auto r2 = ff_.init(engine); if (!r2) return std::unexpected(r2.error());
        if (norm2_) { auto r = norm2_->init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = self_attn_.parameters();
        auto n1 = norm1_->parameters();
        auto f  = ff_.parameters();
        auto n2 = norm2_->parameters();
        p.insert(p.end(), n1.begin(), n1.end());
        p.insert(p.end(), f.begin(), f.end());
        p.insert(p.end(), n2.begin(), n2.end());
        return p;
    }

    // 文档感知：把每样本文档 id 转发给内部自注意力（用于块对角掩码）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        self_attn_.set_doc_ids(ids);
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = self_attn_.param_gradients();
        auto gn1 = norm1_->param_gradients();
        auto gf  = ff_.param_gradients();
        auto gn2 = norm2_->param_gradients();
        g.insert(g.end(), gn1.begin(), gn1.end());
        g.insert(g.end(), gf.begin(), gf.end());
        g.insert(g.end(), gn2.begin(), gn2.end());
        return g;
    }

    // 梯度检查点：把模式传播给内部注意力/归一化/FFN
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        self_attn_.set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
    }

    // GPTBlock 可作为“重计算单元”：从保存的块输入重算 forward 重建缓存
    [[nodiscard]] bool recompute_supported() const override { return true; }

    [[nodiscard]] Result<Tensor> forward_recompute(
        ComputeEngine& engine, const Tensor& saved_input) override
    {
        // 临时关闭本块及其子层的 checkpoint 模式，使 forward 重建缓存
        set_checkpoint_mode(false);
        auto r = forward(engine, saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    void clear_cache() override
    {
        self_attn_.clear_cache();
        norm1_->clear_cache();
        ff_.clear_cache();
        norm2_->clear_cache();
        residual2_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = self_attn_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto n1 = norm1_->activation_cache(); r.insert(r.end(), n1.begin(), n1.end());
        auto f = ff_.activation_cache(); r.insert(r.end(), f.begin(), f.end());
        auto n2 = norm2_->activation_cache(); r.insert(r.end(), n2.begin(), n2.end());
        if (residual2_cache_.valid()) r.emplace_back(residual2_cache_);
        return r;
    }

    // ── activation offload（L1-offload）────────────────────────────────
    void set_offload_enabled(bool enabled)
    {
        offload_enabled_ = enabled;
        if (!enabled) offload_slab_ = Tensor{};  // 释放持久缓冲
    }
    [[nodiscard]] bool offload_enabled() const noexcept { return offload_enabled_; }

    // 导出：把本块 backward 所需的中间激活逐个写入持久 host slab（释放 GPU 显存）。
    // offload_refs_ 记录各缓存成员地址（forward 后成员地址稳定），导入时复用。
    [[nodiscard]] Result<void> export_activations(ComputeEngine& engine)
    {
        if (!offload_enabled_ || offloaded_) return {};
        offload_refs_ = activation_cache();
        offload_offsets_.clear();
        offload_shapes_.clear();
        // 惰性创建持久 slab（大小 = 本块激活总 float 数，跨 step 复用）
        if (!offload_slab_.valid())
        {
            std::size_t total = 0;
            for (auto& ref : offload_refs_)
                if (ref.get().valid()) total += ref.get().size();
            // 无有效激活可导出（如混合模式下 checkpoint 块 forward 不驻留缓存）
            if (total == 0)
            {
                offloaded_ = false;
                return {};
            }
            auto slab = engine.create_offload_buffer(total);
            if (!slab) return std::unexpected(slab.error());
            offload_slab_ = std::move(*slab);
        }
        std::size_t offset = 0;
        for (auto& ref : offload_refs_)
        {
            if (!ref.get().valid()) continue;
            auto r = engine.offload_save(offload_slab_, offset, ref.get());
            if (!r) return std::unexpected(r.error());
            offload_shapes_.push_back({ref.get().rows(), ref.get().cols()});
            offload_offsets_.push_back(offset);
            offset += ref.get().size();
            ref.get() = Tensor{};  // 释放 GPU 版（数据已在 host slab）
        }
        offloaded_ = true;
        return {};
    }

    // 导入：从 host slab 恢复激活到缓存成员（backward 前调用，替代重计算）
    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        if (!offloaded_) return {};
        for (std::size_t i = 0; i < offload_offsets_.size(); ++i)
        {
            auto t = engine.offload_restore(offload_slab_, offload_offsets_[i],
                                            offload_shapes_[i].first,
                                            offload_shapes_[i].second);
            if (!t) return std::unexpected(t.error());
            offload_refs_[i].get() = std::move(*t);
        }
        offloaded_ = false;
        return {};
    }

    // 实际 slab 字节数（诊断用；未创建时 0）
    [[nodiscard]] std::size_t offload_slab_bytes() const noexcept
    {
        return offload_slab_.valid() ? offload_slab_.size() * sizeof(float) : 0;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;

        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        auto r2 = dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(*a),
            input.rows(), input.cols());
        if (!r2) return std::unexpected(r2.error());
        Tensor res2 = std::move(*r2);
        if (!checkpoint_mode_)
            residual2_cache_ = res2;

        auto n2 = norm2_->forward(engine, res2);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return dsl::compute(engine,
            dsl::leaf(res2) + dsl::leaf(*f),
            res2.rows(), res2.cols());
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // activation offload：从 host 恢复激活再反向（替代重计算）
        if (offloaded_)
        {
            auto im = import_activations(engine);
            if (!im) return std::unexpected(im.error());
        }

        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;

        auto grad_r1 = dsl::compute(engine,
            dsl::leaf(grad_output) + dsl::leaf(*b_n2),
            grad_output.rows(), grad_output.cols());
        if (!grad_r1) return std::unexpected(grad_r1.error());

        auto b_sa = self_attn_.backward(engine, *grad_r1);
        if (!b_sa) return b_sa;
        auto b_n1 = norm1_->backward(engine, *b_sa);
        if (!b_n1) return b_n1;

        return dsl::compute(engine,
            dsl::leaf(*grad_r1) + dsl::leaf(*b_n1),
            grad_r1->rows(), grad_r1->cols());
    }

    // ── 增量推理（KV cache）──────────────────────────────────────────
    // Pre-Norm 单 token 前向：
    //   x = x_new + CausalSelfAttn(LN₁(x_new), kv_cache)
    //   x = x + FFN(LN₂(x))
    // 输入: x_new (d_model, 1)
    // 输出: (d_model, 1)
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        const Tensor& x_new,
        Tensor& k_cache,
        Tensor& v_cache,
        std::size_t cur_len)
    {
        auto n1 = norm1_->forward(engine, x_new);
        if (!n1) return n1;

        auto a = self_attn_.forward_step(engine, *n1, k_cache, v_cache, cur_len);
        if (!a) return a;

        auto r2 = dsl::compute(engine,
            dsl::leaf(x_new) + dsl::leaf(*a),
            x_new.rows(), x_new.cols());
        if (!r2) return std::unexpected(r2.error());

        auto n2 = norm2_->forward(engine, *r2);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return dsl::compute(engine,
            dsl::leaf(*r2) + dsl::leaf(*f),
            r2->rows(), r2->cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════════════
// PositionEncoder — 位置编码抽象基类
//
// 把 GPTModel 中原本按 PosEncodingType 散落的 if-else 位置编码逻辑
// （可学习 / 正弦波 / 无）抽离为独立的多态层次，GPTModel 通过基类指针使用，
// 消除 use_pos_emb_ / pos_emb_learnable_ 等标志位分支的耦合。
//
// 接口：
//   - apply(engine, token_emb_T, batch, seq)   全量前向，返回已加位置信息的 x
//   - apply_step(engine, x, pos)               增量单 token 前向
//   - backward(engine, grad_T, batch, seq)     累计位置梯度（仅可学习有意义）
//   - parameters() / param_gradients()         可学习参数（仅 Learned 返回非空）
//
// 注意：ALiBi / RoPE 的位置信息由注意力层注入（CausalSelfAttention 线性偏置 /
//        AttentionBase 的 RotaryEmbedding），故其编码器为 no-op。
// ══════════════════════════════════════════════════════════════════════════
class PositionEncoder
{
public:
    virtual ~PositionEncoder() = default;

    // 全量前向：token_emb_T 为 (d_model, batch*seq)，返回 x = token_emb_T (+ pos_emb)
    [[nodiscard]] virtual Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& token_emb_T,
        std::size_t batch, std::size_t seq) = 0;

    // 增量前向：x 为 (d_model, 1)，返回 x + pos_emb[pos]
    [[nodiscard]] virtual Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& x, std::size_t pos) = 0;

    // 反向：累计位置梯度到 grad_pos_emb_（默认 no-op）
    [[nodiscard]] virtual Result<void> backward(
        ComputeEngine& engine, const Tensor& grad_T,
        std::size_t batch, std::size_t seq) = 0;

    // 初始化（引擎相关操作，如创建张量等）
    [[nodiscard]] virtual Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    [[nodiscard]] virtual std::vector<TensorRef> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<TensorRef> param_gradients() { return {}; }
};

// 加性位置编码基类（Learned / Sinusoidal 共用）：
// 通过 pos_emb_ 张量按位置 gather 并加到 token 嵌入上。
class AdditivePositionEncoder : public PositionEncoder
{
protected:
    bool learnable_ = false;

    Tensor pos_emb_;        // (seq_len, d_model)
    Tensor grad_pos_emb_;   // (seq_len, d_model)，仅 learnable_ 有效

    // pos_indices 缓存（避免每 step 重建）— (total, 1) 值为 [0,..,0,1,..,1,...,seq-1,..]
    Tensor pos_indices_cache_;
    std::size_t pos_indices_batch_ = 0;  // 缓存键：batch_size
    std::size_t pos_indices_seq_ = 0;    // 缓存键：seq_len

    // 用给定的位置编码矩阵初始化 pos_emb_（learnable 时额外分配梯度）
    [[nodiscard]] Result<void> init_(ComputeEngine& engine, Matrix&& pe, bool learnable)
    {
        const std::size_t rows = pe.rows();
        const std::size_t cols = pe.cols();
        learnable_ = learnable;
        auto pe_r = engine.from_matrix(pe);
        if (!pe_r) return std::unexpected(pe_r.error());
        pos_emb_ = std::move(*pe_r);
        if (learnable_)
        {
            grad_pos_emb_ = engine.create_tensor(rows, cols);
            auto r = engine.zero(grad_pos_emb_);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // 确保 pos_indices 缓存有效（batch-major：i = b*seq + t → position=t）
    [[nodiscard]] Result<void> ensure_pos_indices_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        if (pos_indices_batch_ == batch && pos_indices_seq_ == seq)
            return {};
        Matrix pidx_m(batch * seq, 1);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t t = 0; t < seq; ++t)
                pidx_m.set_value_unchecked(b * seq + t, 0,
                    static_cast<Scalar>(t));
        auto pidx_t = engine.from_matrix(pidx_m);
        if (!pidx_t) return std::unexpected(pidx_t.error());
        pos_indices_cache_ = std::move(*pidx_t);
        pos_indices_batch_ = batch;
        pos_indices_seq_ = seq;
        return {};
    }

public:
    AdditivePositionEncoder() = default;

    [[nodiscard]] bool learnable() const noexcept { return learnable_; }

    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& token_emb_T,
        std::size_t batch, std::size_t seq) override
    {
        auto ci = ensure_pos_indices_(engine, batch, seq);
        if (!ci) return std::unexpected(ci.error());
        auto pos_gathered = engine.gather_rows(pos_emb_, pos_indices_cache_);
        if (!pos_gathered) return std::unexpected(pos_gathered.error());
        auto pos_T = engine.transpose(*pos_gathered);
        if (!pos_T) return std::unexpected(pos_T.error());
        auto x_with_pos = dsl::compute(engine,
            dsl::leaf(token_emb_T) + dsl::leaf(*pos_T),
            token_emb_T.rows(), token_emb_T.cols());
        if (!x_with_pos) return std::unexpected(x_with_pos.error());
        return std::move(*x_with_pos);
    }

    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& x, std::size_t pos) override
    {
        Matrix pos_m(1, 1);
        pos_m.set_value_unchecked(0, 0, static_cast<Scalar>(pos));
        auto pos_t = engine.from_matrix(pos_m);
        if (!pos_t) return std::unexpected(pos_t.error());
        auto pos_emb_g = engine.gather_rows(pos_emb_, *pos_t);
        if (!pos_emb_g) return std::unexpected(pos_emb_g.error());
        auto pos_T = engine.transpose(*pos_emb_g);
        if (!pos_T) return std::unexpected(pos_T.error());
        auto x_wp = dsl::compute(engine,
            dsl::leaf(x) + dsl::leaf(*pos_T),
            x.rows(), x.cols());
        if (!x_wp) return std::unexpected(x_wp.error());
        return std::move(*x_wp);
    }

    [[nodiscard]] Result<void> backward(
        ComputeEngine& engine, const Tensor& grad_T,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    {
        if (!learnable_) return {};
        // pos_indices 缓存由 apply() 建立，backward 直接复用（batch/seq 一致）。
        auto pr = engine.scatter_add_rows(grad_pos_emb_, pos_indices_cache_, grad_T);
        if (!pr) return std::unexpected(pr.error());
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        if (!learnable_) return {};
        return { pos_emb_ };
    }
    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        if (!learnable_) return {};
        return { grad_pos_emb_ };
    }
};

// 可学习位置编码（GPT 默认）：N(0, 0.02) 随机初始化。
// 与 token_emb_ 共享同一 rng 序列（保持与旧实现完全一致的可复现性）。
class LearnedPositionEncoder final : public AdditivePositionEncoder
{
    std::size_t d_model_;
    std::size_t seq_len_;
    // 自持 RNG（seed=42、N(0,0.02)）：init() 延迟到构造之后调用，
    // 不能持有指向构造函数局部变量的非拥有指针（会悬空）。
    std::mt19937_64 rng_{42};
    std::normal_distribution<Scalar> dist_{0.0, 0.02};

public:
    LearnedPositionEncoder(std::size_t d_model, std::size_t seq_len)
        : d_model_(d_model), seq_len_(seq_len) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix pe(seq_len_, d_model_);
        auto pe_s = pe.span();
        for (std::size_t i = 0; i < pe.size(); ++i) pe_s[i] = dist_(rng_);
        return init_(engine, std::move(pe), /*learnable=*/true);
    }
};

// 正弦波固定位置编码（冻结，不参与训练）：
//   PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
class SinusoidalPositionEncoder final : public AdditivePositionEncoder
{
    std::size_t d_model_;
    std::size_t seq_len_;

public:
    SinusoidalPositionEncoder(std::size_t d_model, std::size_t seq_len)
        : d_model_(d_model), seq_len_(seq_len) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix pe(seq_len_, d_model_);
        auto pe_s = pe.span();
        for (std::size_t pos = 0; pos < seq_len_; ++pos)
            for (std::size_t i = 0; i < d_model_; ++i)
            {
                Scalar angle = static_cast<Scalar>(pos) /
                    std::pow(Scalar{10000}, static_cast<Scalar>(2 * (i / 2)) / static_cast<Scalar>(d_model_));
                pe_s[pos * d_model_ + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        return init_(engine, std::move(pe), /*learnable=*/false);
    }
};

// 无位置编码（ALiBi / RoPE）：位置信息由注意力层注入，此处为 no-op。
class NoPositionEncoder final : public PositionEncoder
{
public:
    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& /*engine*/, const Tensor& token_emb_T,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    { return token_emb_T; }

    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& /*engine*/, const Tensor& x, std::size_t /*pos*/) override
    { return x; }

    [[nodiscard]] Result<void> backward(
        ComputeEngine& /*engine*/, const Tensor& /*grad_T*/,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    { return {}; }
};

// GPTModel — Decoder-only Transformer 语言模型
//
// 算法（只在此处，不在 Engine/Shader）：
//   组件: TokenEmb [+ PosEmb] + N × GPTBlock + LayerNorm + LM Head
//   输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
//   输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
//
// 通过 PosEncodingType 参数支持三种位置编码模式：
//   - Learned:    可学习位置嵌入（默认）
//   - Sinusoidal: 正弦波固定位置编码（冻结）
//   - ALiBi:      无位置嵌入，通过 CausalSelfAttention 的线性偏置注入位置信息
//
// 注意: token embedding 查表 + 位置 embedding 相加涉及按 token ID 的
//       稀疏写入，此处用 to_matrix/from_matrix 在 CPU 端完成
//       （batch 边界，PCIe 传输符合纯 GPU 架构约定）。
// ══════════════════════════════════════════════════════════════════════════
class GPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;

    // 嵌入
    Tensor token_emb_;       // (vocab_size, d_model)
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;  // 位置编码（多态：Learned/Sinusoidal/无）

    std::vector<GPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    // 反向缓存
    Tensor stored_tokens_tensor_;          // token IDs 的 Tensor 版本 (total, 1)
    std::size_t batch_size_ = 0;

    // 文档感知：当前 step 每样本文档 id（batch-major b*seq+t → doc id），
    // 由调用方在 forward 前 set_doc_ids 设置，转发给各 GPTBlock。
    std::vector<std::size_t> doc_ids_;

    // batch 录制粒度：每隔 flush_interval_ 个 Transformer block 提交一次
    // 0 = 不在 block 间 flush（默认），>0 = 每 N 个 block flush 一次
    std::size_t flush_interval_ = 0;

    // 梯度检查点（激活重计算 L1）：每隔 checkpoint_every_ 个 GPTBlock 保存一次
    // 块输入，backward 时重算以省去驻留整层激活。0 = 不启用。
    std::size_t checkpoint_every_ = 0;
    std::vector<Tensor> checkpoint_inputs_;  // 各 checkpoint 块的输入 (d_model, batch*seq)

    // activation offload（L1-offload）：把每块内部激活搬 host-visible，backward 拷回
    bool activation_offload_ = false;

public:
    GPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
             std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
             PosEncodingType pos_enc_type = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, seq_len,
                                 pos_enc_type, activation, norm_type);

        // 初始化位置编码器（Learned / Sinusoidal / ALiBi / RoPE）
        switch (pos_enc_type)
        {
            case PosEncodingType::Learned:
                pos_encoder_ = std::make_unique<LearnedPositionEncoder>(
                    d_model, seq_len);
                break;
            case PosEncodingType::Sinusoidal:
                pos_encoder_ = std::make_unique<SinusoidalPositionEncoder>(
                    d_model, seq_len);
                break;
            default:  // ALiBi / RoPE：位置信息由注意力层注入
                pos_encoder_ = std::make_unique<NoPositionEncoder>();
                break;
        }
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // 初始化 token_emb_
        Matrix te(vocab_size_, d_model_);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);

        auto te_r = engine.from_matrix(te);
        if (!te_r) return std::unexpected(te_r.error());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_);
        { auto r1 = engine.zero(grad_token_emb_); if (!r1) return std::unexpected(r1.error()); }

        // 初始化子层
        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        for (auto& block : blocks_)
        {
            auto r = block.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        if (ln_f_)
        {
            auto r = ln_f_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        { auto r = lm_head_.init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        auto gp = pos_encoder_->param_gradients();
        g.insert(g.end(), gp.begin(), gp.end());
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_->param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    // 文档感知：设置当前 step 每样本文档 id（batch-major b*seq+t → doc id）。
    // 传入空 span 清除文档感知（退化为纯因果）。在 forward 前调用。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq_len = input.rows();
        batch_size_ = input.cols();
        // ── 1. gather 所有 token 的 embedding（统一采用 batch-major 列序） ──
        // 下方注意力（AttentionBase::forward 的 rearrange_3d + batched_matmul + 因果掩码）
        // 假定扁平列为 batch-major（b*seq + t）。因此先把输入 (seq, batch) 转置为
        // (batch, seq)，使 gather_rows 的 flat 序即为 batch-major：i = b*seq + t。
        auto input_T = engine.transpose(input);   // (batch, seq)，flat 索引 = b*seq+t
        if (!input_T) return std::unexpected(input_T.error());

        // gather_rows(token_emb_, input_T) → (batch*seq, d_model)
        //   row i = token_emb[input_T[i]], i 是 batch-major 索引 b*seq+t
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());

        // ── 2. 保存 token IDs 的 Tensor 拷贝（供 backward 的 scatter_add_rows） ──
        // 用 batch-major 序的 input_T，使 scatter 行号与 grad_T（transpose(grad_x)）对齐。
        // 全程 GPU：clone 在 GPU 内执行，无 PCIe 传输
        auto st_t = engine.clone(*input_T);
        if (!st_t) return std::unexpected(st_t.error());
        stored_tokens_tensor_ = std::move(*st_t);

        // ── 3. 构造 x: (d_model, batch*seq)（batch-major 列序） ──
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        // ── 3. 施加位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE 为 no-op） ──
        auto x_result = pos_encoder_->apply(engine, *all_T, batch_size_, seq_len);
        if (!x_result) return std::unexpected(x_result.error());
        // ── 4. 通过 Transformer 块（全批量化，无 per-sample 循环） ──
        Tensor x = std::move(*x_result);
        checkpoint_inputs_.clear();
        const bool ckpt = (checkpoint_every_ > 0);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            // 文档感知：把本 step 每样本文档 id 传给各 block 的注意力
            blocks_[bi].set_doc_ids(doc_ids_);
            // 梯度检查点：每 checkpoint_every_ 个块保存一次输入；
            // 该块及其子层以 checkpoint 模式运行（不驻留中间激活）
            if (ckpt && (bi % checkpoint_every_ == 0))
            {
                auto save = engine.clone(x);
                if (!save) return std::unexpected(save.error());
                checkpoint_inputs_.push_back(std::move(*save));
                blocks_[bi].set_checkpoint_mode(true);
            }
            else
            {
                blocks_[bi].set_checkpoint_mode(false);
            }
            auto r = blocks_[bi].forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
            // activation offload：forward 后把本块内部激活搬 host-visible，释放 GPU 显存。
            // 混合模式（offload + checkpoint 共存）：checkpoint 块 forward 不驻留激活
            // （checkpoint_mode=true），无可导出的缓存，必须跳过 export（否则会创建
            // 空 slab 而失败）；非 checkpoint 块才导出。
            if (activation_offload_ && !blocks_[bi].checkpoint_mode())
            {
                auto ex = blocks_[bi].export_activations(engine);
                if (!ex) return std::unexpected(ex.error());
            }
            // 按间隔 flush，将大录制拆分为多个小提交（防 TDR）
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // ── 5. 最终 LayerNorm/RMSNorm ──
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        x = std::move(*ln);

        // ── 6. LM Head → (vocab_size, seq*batch) batch-major ──
        auto lm_out = lm_head_.forward(engine, x);
        return lm_out;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;

        // ⚠ 注意：不用在这里 zero grad_token_emb_！
        //   grad_token_emb_ 已注册到优化器的 param_gradients() 中，
        //   由优化器的 zero_grad() 统一清零。这里如果额外清零会破坏
        //   梯度积累（accum_steps > 1 时前几轮的梯度信号全部丢失）。
        //   位置编码梯度同理（pos_encoder_->param_gradients() 已注册，勿在此清零）。
        // (void)engine.zero(grad_token_emb_);

        // ── 1. LM Head 反向 → (d_model, seq*batch) ──
        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        Tensor grad_x = std::move(*b_lm);

        // ── 2. LayerNorm/RMSNorm 反向 ──
        auto b_ln = ln_f_->backward(engine, grad_x);
        if (!b_ln) return b_ln;
        grad_x = std::move(*b_ln);

        // ── 3. 逐块反向（全批量化） ──
        {
            const std::size_t n = blocks_.size();
            for (std::size_t bi = 0; bi < n; ++bi)
            {
                const std::size_t idx = n - 1 - bi;
                // 梯度检查点：若是 checkpoint 块，先重算 forward 重建缓存再反向
                if (checkpoint_every_ > 0 && (idx % checkpoint_every_ == 0))
                {
                    const std::size_t seg = idx / checkpoint_every_;
                    NN_ASSERT(seg < checkpoint_inputs_.size(),
                              "GPTModel backward: checkpoint input missing");
                    auto cr = blocks_[idx].forward_recompute(engine, checkpoint_inputs_[seg]);
                    if (!cr) return cr;
                }
                auto br = blocks_[idx].backward(engine, grad_x);
                if (!br) return br;
                grad_x = std::move(*br);
                // 梯度检查点：backward 后立即释放该块的重算激活缓存，
                // 避免跨块累积（否则所有块缓存会在 backward 末尾同时驻留，
                // 抵消检查点的显存收益）。非 checkpoint 模式下不清理。
                if (checkpoint_every_ > 0)
                    blocks_[idx].clear_cache();
                // activation offload：backward 后释放恢复的激活缓存（掩码常驻不清理）
                if (activation_offload_)
                    blocks_[idx].clear_cache();
                if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
                {
                    auto fr = engine.flush_batch();
                    if (!fr) return std::unexpected(fr.error());
                }
            }
        }

        // ── 3.5 重计算完成：释放保存的块输入（供 L2 整块归还） ──
        checkpoint_inputs_.clear();

        // ── 4. 转置 grad_x + pos_grad GPU 计算 ──
        //   grad_x: (d_model, batch*seq)（batch-major 列序）
        //   grad_T = transpose(grad_x) → (total, d_model) — 用于 scatter_add_rows
        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());

        // 位置编码反向（Learned 累计 grad_pos_emb_；Sinusoidal/ALiBi/RoPE no-op）
        auto pr = pos_encoder_->backward(engine, *grad_T, batch_size_, seq_len_);
        if (!pr) return std::unexpected(pr.error());

        // ── 5. scatter_add_rows: grad_token_emb_[tokens] += grad_T ──
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        // grad_input: token IDs 无梯度，返回零张量（仅用于接口一致性）
        Matrix grad_input(seq_len, batch_size_, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── batch 录制粒度控制 ──
    void set_flush_interval(std::size_t interval) override { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 梯度检查点（激活重计算 L1）粒度控制 ──
    // stride：每 N 个 GPTBlock 保存一次块输入，backward 时重算该块 forward。
    // 0 = 不启用（默认）。stride=1 时显存收益最大（仅保留块输入 + 单块激活）。
    void set_checkpoint_every(std::size_t stride) override { checkpoint_every_ = stride; }
    [[nodiscard]] std::size_t checkpoint_every() const noexcept { return checkpoint_every_; }

    // ── activation offload（L1-offload）开关 ──
    // 启用后：forward 把每块内部激活搬 host-visible（释放 device-local VRAM），
    // backward 拷回再反向（不重算，FLOPs 保持 1.0×，代价是 PCIe 传输）。
    void set_activation_offload(bool enabled) override
    {
        activation_offload_ = enabled;
        for (auto& b : blocks_) b.set_offload_enabled(enabled);
    }
    [[nodiscard]] bool activation_offload() const noexcept { return activation_offload_; }

    // 实际 offload RAM 字节数：各块已创建 slab 大小之和（诊断用）
    [[nodiscard]] std::size_t offload_ram_bytes() override
    {
        std::size_t total = 0;
        for (auto& b : blocks_)
            total += b.offload_slab_bytes();
        return total;
    }

    // 梯度检查点：把模式传播给所有块与末级归一化/LM Head
    // （当本 GPTModel 整体作为 Model 的一层被置于 checkpoint 模式时生效）
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        checkpoint_inputs_.clear();
    }

    // ── 增量推理：单 token 前向（KV cache）──────────────────────────
    // 流程: token_emb[token] [+ pos_emb[pos]] → N × GPTBlock.forward_step
    //       → ln_f → lm_head → (vocab_size, 1)
    // 每层的 KV cache 由调用方维护，forward_step 只负责写入和计算。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        std::size_t token_id,
        std::size_t pos,
        std::vector<Tensor>& k_caches,
        std::vector<Tensor>& v_caches,
        std::size_t cur_len)
    {
        // 1. token embedding 查表 → (1, d_model) → transpose → (d_model, 1)
        Matrix id_m(1, 1);
        id_m.set_value_unchecked(0, 0, static_cast<Scalar>(token_id));
        auto id_t = engine.from_matrix(id_m);
        if (!id_t) return std::unexpected(id_t.error());
        auto emb = engine.gather_rows(token_emb_, *id_t);
        if (!emb) return std::unexpected(emb.error());
        auto x_new = engine.transpose(*emb);
        if (!x_new) return std::unexpected(x_new.error());

        // 2. 位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE no-op）
        auto x_wp = pos_encoder_->apply_step(engine, *x_new, pos);
        if (!x_wp) return std::unexpected(x_wp.error());
        x_new = std::move(*x_wp);

        // 3. 逐块增量前向
        Tensor x = std::move(*x_new);
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            auto r = blocks_[i].forward_step(
                engine, x, k_caches[i], v_caches[i], cur_len);
            if (!r) return r;
            x = std::move(*r);
            // 按 flush_interval 拆分提交（防 TDR）
            if (flush_interval_ > 0 &&
                (i + 1) % flush_interval_ == 0 && i + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // 4. 最终 LayerNorm + LM Head → (vocab_size, 1)
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    // ── 采样生成（KV cache 增量推理）────────────────────────────────
    // 性能策略（P0 + P1 + KV cache）：
    //   P0: begin_batch/end_batch 包裹 forward_step，单次 GPU 提交。
    //   P1: 每步只上传 1 个 token ID，无需重传整个 seq_len。
    //   KV cache: 每步 attention 只计算 Q×K_history（O(seq_len) 而非 O(seq_len²)），
    //             历文 K/V 不再重复投影。
    //   logits 直接是 (vocab_size, 1)，无需 transpose+slice。
    //
    // 滑动窗口: 当 cur_len 达到 seq_len_ 时，丢弃最旧 token 重建 cache
    //           （保留最后 seq_len_-1 个 token 作为新上下文）。
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
             const std::vector<std::size_t>& prompt,
             std::size_t max_new_tokens,
             Scalar temperature = 1.0,
             std::size_t eos_token_id = static_cast<std::size_t>(-1),
             std::size_t min_new_tokens = 0)
    {
        std::vector<std::size_t> context(prompt);
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

        // ── 预分配 KV cache: 每层一对 (k, v)，形状 (seq_len_, d_model) ──
        // H*d_k = d_model（因为 d_k = d_model / num_heads）
        std::vector<Tensor> k_caches, v_caches;
        k_caches.reserve(blocks_.size());
        v_caches.reserve(blocks_.size());
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            k_caches.push_back(engine.create_tensor(seq_len_, d_model_));
            v_caches.push_back(engine.create_tensor(seq_len_, d_model_));
        }

        // ── prefill: 截断到 seq_len_ 长度（滑动窗口初始） ──────────────
        std::size_t start_init = 0;
        if (context.size() > seq_len_)
            start_init = context.size() - seq_len_;
        std::size_t cur_len = 0;
        Tensor last_logits_t;

        // ── 逐 token 填充 KV cache（prefill 与滑动窗口重建共用） ──────
        // 从 context[start..end) 逐个 forward_step，更新 cur_len 与 last_logits_t
        auto fill_cache_ = [&](std::size_t start) -> Result<void>
        {
            for (std::size_t i = start; i < context.size(); ++i)
            {
                auto r = forward_step(engine, context[i], cur_len,
                                      k_caches, v_caches, cur_len);
                if (!r) return std::unexpected(r.error());
                last_logits_t = *r;
                ++cur_len;
            }
            return {};
        };

        {
            auto r = fill_cache_(start_init);
            if (!r) return std::unexpected(r.error());
        }

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            // Sliding window: rebuild cache when full (keep last seq_len_-1 tokens)
            if (cur_len >= seq_len_)
            {
                for (auto& kc : k_caches) { auto r = engine.zero(kc); if (!r) return std::unexpected(r.error()); }
                for (auto& vc : v_caches) { auto r = engine.zero(vc); if (!r) return std::unexpected(r.error()); }
                cur_len = 0;
                const std::size_t keep = seq_len_ - 1;
                const std::size_t start_new = (context.size() > keep) ? (context.size() - keep) : 0;
                auto r = fill_cache_(start_new);
                if (!r) return std::unexpected(r.error());
            }

            // Sample from last_logits_t (from prefill or previous step)
            auto logits_m = engine.to_matrix(last_logits_t);
            if (!logits_m) return std::unexpected(logits_m.error());

            std::vector<Scalar> last_logits(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last_logits[v] = logits_m->at_unchecked(v, 0);

            // temperature
            if (temperature > 0.0 && temperature != 1.0)
                for (auto& v : last_logits) v /= temperature;

            // softmax（数值稳定）
            Scalar max_val = last_logits[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, last_logits[v]);
            Scalar sum_exp = 0.0;
            for (auto& v : last_logits)
            {
                v = std::exp(v - max_val);
                sum_exp += v;
            }
            for (auto& v : last_logits) v /= sum_exp;

            // 采样
            // temperature > 0: 随机采样（temperature 仅影响缩放，1.0 = 不缩放但仍然采样）
            // temperature == 0: 贪心解码（argmax）
            std::size_t next_token;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cumulative = 0.0;
                next_token = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cumulative += last_logits[v];
                    if (r <= cumulative) { next_token = v; break; }
                }
            }
            else
            {
                next_token = 0;
                Scalar best = last_logits[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (last_logits[v] > best) { best = last_logits[v]; next_token = v; }
            }

            context.push_back(next_token);

            if (step >= min_new_tokens && next_token == eos_token_id)
                break;

            generated.push_back(next_token);

            // Run forward_step for next_token to write KV cache and get new logits
            auto br = engine.begin_batch();
            if (!br) return std::unexpected(br.error());

            auto logits_res = forward_step(engine, next_token, cur_len,
                                           k_caches, v_caches, cur_len);
            if (!logits_res) return std::unexpected(logits_res.error());

            auto er = engine.end_batch();
            if (!er) return std::unexpected(er.error());

            last_logits_t = std::move(*logits_res);
            ++cur_len;
        }
        return generated;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// concat_cols — 沿列方向拼接两个张量（列主序激活：拼接序列/记忆维度）
//   a: (rows, c1), b: (rows, c2) → out: (rows, c1+c2)
// 用 transpose + insert_rows + transpose 组合实现（纯引擎原语，CPU/GPU 均可，
// 无需新增 shader 或 AOT 融合）。用于 ZiPT 阶段二把记忆 K/V 与局部 K/V 拼接。
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline Result<Tensor> concat_cols(
    ComputeEngine& engine, const Tensor& a, const Tensor& b)
{
    if (a.rows() != b.rows())
        return std::unexpected(Error{"concat_cols: row count mismatch"});
    auto aT = engine.transpose(a);   // (c1, rows)
    if (!aT) return std::unexpected(aT.error());
    auto bT = engine.transpose(b);   // (c2, rows)
    if (!bT) return std::unexpected(bT.error());
    Tensor dstT = engine.create_tensor(a.cols() + b.cols(), a.rows());
    auto r1 = engine.insert_rows(dstT, 0, *aT);
    if (!r1) return std::unexpected(r1.error());
    auto r2 = engine.insert_rows(dstT, a.cols(), *bT);
    if (!r2) return std::unexpected(r2.error());
    return engine.transpose(dstT);
}

// ══════════════════════════════════════════════════════════════════════════
// CrossAttention — 阶段一：全局上下文重要性压缩（AttnZip Memory Queries）
//
// 算法（AttnZip 文档 §3.2，单头，忠实还原）：
//   K = X·W_K, V = X·W_V                     X: (d_model, batch·L)
//   A = Softmax(P·K^T / sqrt(d))             P: (d_model, M) 可学习记忆查询
//   C = A·V                                  C: (d_model, batch·M)
//
// 批量化（batch=batch，单头，d_k = d_model）：
//   Q_re = P 沿 batch 平铺 → (batch·d_model, M)
//   K_re = rearrange_3d(K, d_model, batch, L) → (batch·d_model, L)
//   S    = batched_matmul(Q_re, K_re, batch, transA=true, alpha=scale) → (batch·M, L)
//   A    = softmax(S)
//   C_re = batched_matmul(V_re, A, batch, false, true) → (batch·d_model, M)
//   C    = rearrange_3d(C_re, d_model, batch, M, true) → (d_model, batch·M)
//
// 参数：P (d_model, M) 可学习 + w_k + w_v（无 w_o：C = A·V 直接输出）
// 复杂度：对序列长度 L 线性（O(M·L·d)），是 AttnZip 消除 O(L²) 的核心。
// ══════════════════════════════════════════════════════════════════════════
} // namespace nn

#endif // NN_COMPUTE_LAYER_GPT_HPP
