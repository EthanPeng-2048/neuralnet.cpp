#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_feedforward.hpp"
#include "compute_layer_attention.hpp"
#include "compute_layer_gpt.hpp"

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

namespace nn {
class CrossAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t memory_;       // M：记忆 token 数
    std::size_t seq_len_;      // L：上下文序列长度（0 = 单样本，cols 即 L）
    Scalar scale_;
    Tensor P_;                 // 记忆查询 (d_model, M) 可学习
    Tensor grad_P_;            // 记忆查询梯度
    Linear w_k_;
    Linear w_v_;
    Softmax softmax_;

    // forward 缓存（供 backward）
    Tensor Q_re_cache_;        // (batch·d_model, M) P 平铺
    Tensor K_re_cache_;        // (batch·d_model, L)
    Tensor V_re_cache_;        // (batch·d_model, L)
    std::size_t batch_cache_ = 0;
    std::size_t seq_cache_    = 0;

    // 把 P 沿 batch 平铺为 (batch·d_model, M)：每个 batch 块 = P
    [[nodiscard]] Result<Tensor> build_q_re_(ComputeEngine& engine, std::size_t batch)
    {
        Tensor q = engine.create_tensor(batch * d_model_, memory_);
        for (std::size_t b = 0; b < batch; ++b)
        {
            auto r = engine.insert_rows(q, b * d_model_, P_);
            if (!r) return std::unexpected(r.error());
        }
        return q;
    }

public:
    CrossAttention(std::size_t d_model, std::size_t memory,
                   std::size_t seq_len = 0)
        : d_model_(d_model), memory_(memory), seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model))),
          w_k_(d_model, d_model), w_v_(d_model, d_model) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // 初始化记忆查询 P（列主序 (d_model, M)，每列一个记忆查询）
        Matrix p_cpu(d_model_, memory_);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(d_model_ + memory_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        std::mt19937_64 rng{std::random_device{}()};
        auto sp = p_cpu.span();
        for (std::size_t i = 0; i < p_cpu.size(); ++i) sp[i] = dist(rng);
        auto pr = engine.from_matrix(p_cpu);
        if (!pr) return std::unexpected(pr.error());
        P_ = std::move(*pr);

        grad_P_ = engine.create_tensor(d_model_, memory_);
        { auto r = engine.zero(grad_P_); if (!r) return std::unexpected(r.error()); }

        auto r1 = w_k_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_v_.init(engine); if (!r2) return std::unexpected(r2.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> out;
        out.push_back(P_);
        auto p = w_k_.parameters();
        auto v = w_v_.parameters();
        out.insert(out.end(), p.begin(), p.end());
        out.insert(out.end(), v.begin(), v.end());
        return out;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> out;
        out.push_back(grad_P_);
        auto gk = w_k_.param_gradients();
        auto gv = w_v_.param_gradients();
        out.insert(out.end(), gk.begin(), gk.end());
        out.insert(out.end(), gv.begin(), gv.end());
        return out;
    }

    void clear_cache() override
    {
        Q_re_cache_ = Tensor{}; K_re_cache_ = Tensor{}; V_re_cache_ = Tensor{};
        softmax_.clear_cache(); batch_cache_ = 0; seq_cache_ = 0;
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Q_re_cache_.valid()) r.emplace_back(Q_re_cache_);
        if (K_re_cache_.valid()) r.emplace_back(K_re_cache_);
        if (V_re_cache_.valid()) r.emplace_back(V_re_cache_);
        auto s = softmax_.activation_cache();
        r.insert(r.end(), s.begin(), s.end());
        return r;
    }

    // 前向：输入 X (d_model, batch·L) → 输出 C (d_model, batch·M)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t total = input.cols();
        const std::size_t L = (seq_len_ > 0) ? seq_len_ : total;
        const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;

        auto K = w_k_.forward(engine, input);   // (d_model, batch·L)
        if (!K) return K;
        auto V = w_v_.forward(engine, input);
        if (!V) return V;

        auto Q_re = build_q_re_(engine, batch);              // (batch·d_model, M)
        if (!Q_re) return std::unexpected(Q_re.error());
        auto K_re = engine.rearrange_3d(*K, d_model_, batch, L, false);  // (batch·d_model, L)
        if (!K_re) return std::unexpected(K_re.error());
        auto V_re = engine.rearrange_3d(*V, d_model_, batch, L, false);
        if (!V_re) return std::unexpected(V_re.error());

        // S = batched_matmul(Q_re, K_re, batch, transA=true, alpha=scale) → (batch·M, L)
        auto S = engine.batched_matmul(*Q_re, *K_re, batch, true, false, scale_);
        if (!S) return std::unexpected(S.error());
        // A = softmax(S)
        auto attn = softmax_.forward(engine, *S);
        if (!attn) return std::unexpected(attn.error());
        // C_re = batched_matmul(V_re, A, batch, false, true) → (batch·d_model, M)
        auto C_re = engine.batched_matmul(*V_re, *attn, batch, false, true);
        if (!C_re) return std::unexpected(C_re.error());
        // C = rearrange back → (d_model, batch·M)
        auto C = engine.rearrange_3d(*C_re, d_model_, batch, memory_, true);
        if (!C) return std::unexpected(C.error());

        if (!checkpoint_mode_)
        {
            Q_re_cache_ = std::move(*Q_re);
            K_re_cache_ = std::move(*K_re);
            V_re_cache_ = std::move(*V_re);
            batch_cache_ = batch;
            seq_cache_   = L;
        }
        return C;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // grad_output: (d_model, batch·M)
        const std::size_t batch = batch_cache_;
        const std::size_t L = seq_cache_;

        auto grad_C_re = engine.rearrange_3d(grad_output, d_model_, batch, memory_, false);
        if (!grad_C_re) return std::unexpected(grad_C_re.error());

        // A 由 softmax.output_cache() 单一持有
        const Tensor& A = softmax_.output_cache();

        // grad_V_re = batched_matmul(grad_C, A, batch, false, false) → (batch·d_model, L)
        auto grad_V_re = engine.batched_matmul(*grad_C_re, A, batch, false, false);
        if (!grad_V_re) return std::unexpected(grad_V_re.error());
        // grad_A = batched_matmul(grad_C, V, batch, true, false) → (batch·M, L)
        auto grad_A = engine.batched_matmul(*grad_C_re, V_re_cache_, batch, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());
        // grad_S = softmax.backward(grad_A)
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());

        // grad_Q_re = batched_matmul(K, grad_S, batch, false, true, scale) → (batch·d_model, M)
        auto grad_Q_re = engine.batched_matmul(K_re_cache_, *grad_S, batch, false, true, scale_);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        // grad_P = Σ_b grad_Q_re[b]（Q 是 P 沿 batch 的平铺，需按 batch 累加）
        for (std::size_t b = 0; b < batch; ++b)
        {
            auto slice = engine.slice_rows(*grad_Q_re, b * d_model_, d_model_);
            if (!slice) return std::unexpected(slice.error());
            auto ar = engine.add_inplace(grad_P_, *slice);
            if (!ar) return std::unexpected(ar.error());
        }
        // grad_K_re = batched_matmul(Q, grad_S, batch, false, false, scale) → (batch·d_model, L)
        auto grad_K_re = engine.batched_matmul(Q_re_cache_, *grad_S, batch, false, false, scale_);
        if (!grad_K_re) return std::unexpected(grad_K_re.error());

        // 还原到 (d_model, batch·L) 并喂给 w_k / w_v
        auto grad_K = engine.rearrange_3d(*grad_K_re, d_model_, batch, L, true);
        if (!grad_K) return std::unexpected(grad_K.error());
        auto grad_V = engine.rearrange_3d(*grad_V_re, d_model_, batch, L, true);
        if (!grad_V) return std::unexpected(grad_V.error());
        auto gk = w_k_.backward(engine, *grad_K);
        if (!gk) return gk;
        auto gv = w_v_.backward(engine, *grad_V);
        if (!gv) return gv;
        return dsl::compute(engine,
            dsl::leaf(*gk) + dsl::leaf(*gv),
            gk->rows(), gk->cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ZiPTBlock — 阶段二：局部-全局联合注意力 + FFN（Pre-Norm 解码器块）
//
// 算法（AttnZip 文档 §3.3）：
//   n1 = LN₁(Y)
//   Q_y = n1·W_Q,  K_y = n1·W_K,  V_y = n1·W_V         局部 Y (d_model, batch·W)
//   K_c = C·W_Kc,  V_c = C·W_Vc                        记忆 C (d_model, batch·M)
//   K_cat = [K_c, K_y],  V_cat = [V_c, V_y]            序列维拼接
//   O = Softmax(Q_y·K_cat^T / sqrt(d_k)) · V_cat
//   r2 = Y + O·W_O； out = r2 + FFN(LN₂(r2))
//
// 掩码：记忆 token 对局部全部可见（无掩码），局部 token 之间因果
//   （token t 可见记忆全部 + 局部 ≤ t；局部 > t 掩掉）。
// 批量化（batch*H，materialized 路径）：Q/K/V 经 rearrange 后 batched_matmul。
// 复杂度：对长上下文 L 为常数（记忆 M 与窗口 W 均 ≪ L）。
// ══════════════════════════════════════════════════════════════════════════
class ZiPTBlock final : public Layer
{
private:
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t window_;       // W：局部窗口（模型 seq_len）
    std::size_t memory_;       // M：记忆 token 数
    Scalar scale_;
    std::unique_ptr<Layer> norm1_;
    Linear w_q_, w_k_, w_v_, w_o_;   // 局部投影
    Linear w_kc_, w_vc_;             // 记忆 K/V 投影（独立，避免共享 Linear 缓存串扰）
    Softmax softmax_;
    std::unique_ptr<Layer> norm2_;
    FeedForward ff_;

    Tensor residual2_cache_;
    // forward 缓存（联合注意力反向）
    Tensor Q_re_cache_;        // (batch*H*d_k, W)
    Tensor K_cat_cache_;       // (batch*H*d_k, M+W)
    Tensor V_cat_cache_;       // (batch*H*d_k, M+W)
    std::size_t batch_cache_ = 0;
    // 掩码缓存（常数，跨 step 复用；按 batch 键控）
    Tensor mask_cache_;
    std::size_t mask_batch_ = 0;
    // 文档感知：每窗口位置 doc id（batch-major，size = batch*window_）；空=无。
    // 用于在局部因果掩码之上叠加文档边界隔离（含 [PAD] 隔离，PAD 的 doc=0）。
    std::vector<std::size_t> doc_ids_;
    bool has_doc_ids_ = false;

    // 构建记忆-局部联合掩码 (batch*H*W, M+W)：
    //   行 (bb, t)：列 [0, M) 记忆全可见；列 [M, M+t] 局部因果；列 [M+t+1, M+W) = -1e30
    //   文档感知（has_doc_ids_）：局部列再加 doc[j]==doc[t] 约束（跨文档/PAD 屏蔽），
    //   记忆列仍全可见（记忆=压缩历史，不含 PAD，按 AttnZip 设计保持全局可见）。
    //   有 doc_ids 时掩码每 step 变化，不缓存；否则按 batch 缓存复用。
    [[nodiscard]] Result<Tensor> build_mask_(ComputeEngine& engine, std::size_t batch)
    {
        if (!has_doc_ids_ && mask_cache_.valid() && mask_batch_ == batch)
            return mask_cache_;
        const std::size_t BH = batch * num_heads_;
        const std::size_t total_keys = memory_ + window_;
        Matrix mask(BH * window_, total_keys, Scalar{0});
        const Scalar NEG = Scalar{-1e30};
        for (std::size_t bb = 0; bb < BH; ++bb)
        {
            const std::size_t b = bb / num_heads_;
            for (std::size_t t = 0; t < window_; ++t)
            {
                // 局部因果：未来位置（j_local > t）屏蔽
                for (std::size_t jl = t + 1; jl < window_; ++jl)
                    mask.set_value_unchecked(bb * window_ + t, memory_ + jl, NEG);
                // 文档边界：跨文档（含 PAD，doc=0）屏蔽
                if (has_doc_ids_)
                    for (std::size_t jl = 0; jl < window_; ++jl)
                        if (doc_ids_[b * window_ + jl] != doc_ids_[b * window_ + t])
                            mask.set_value_unchecked(bb * window_ + t, memory_ + jl, NEG);
            }
        }
        auto mr = engine.from_matrix(mask);
        if (!mr) return std::unexpected(mr.error());
        mask_cache_ = std::move(*mr);
        mask_batch_ = batch;
        return mask_cache_;
    }

public:
    ZiPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
              std::size_t window, std::size_t memory,
              NormType norm_type = NormType::LayerNorm,
              ActivationType activation = ActivationType::GeLU)
        : num_heads_(num_heads),
          d_k_(d_model / num_heads), window_(window), memory_(memory),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          norm1_(make_norm_layer(d_model, norm_type)),
          w_q_(d_model, d_model), w_k_(d_model, d_model),
          w_v_(d_model, d_model), w_o_(d_model, d_model),
          w_kc_(d_model, d_model), w_vc_(d_model, d_model),
          norm2_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ZiPTBlock: d_model must be divisible by num_heads");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = norm1_->init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_q_.init(engine);    if (!r2) return std::unexpected(r2.error());
        auto r3 = w_k_.init(engine);    if (!r3) return std::unexpected(r3.error());
        auto r4 = w_v_.init(engine);    if (!r4) return std::unexpected(r4.error());
        auto r5 = w_o_.init(engine);    if (!r5) return std::unexpected(r5.error());
        auto r6 = w_kc_.init(engine);   if (!r6) return std::unexpected(r6.error());
        auto r7 = w_vc_.init(engine);   if (!r7) return std::unexpected(r7.error());
        auto r8 = norm2_->init(engine); if (!r8) return std::unexpected(r8.error());
        auto r9 = ff_.init(engine);     if (!r9) return std::unexpected(r9.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> out;
        auto n1 = norm1_->parameters(); out.insert(out.end(), n1.begin(), n1.end());
        auto q = w_q_.parameters();     out.insert(out.end(), q.begin(), q.end());
        auto k = w_k_.parameters();     out.insert(out.end(), k.begin(), k.end());
        auto v = w_v_.parameters();     out.insert(out.end(), v.begin(), v.end());
        auto o = w_o_.parameters();     out.insert(out.end(), o.begin(), o.end());
        auto kc = w_kc_.parameters();   out.insert(out.end(), kc.begin(), kc.end());
        auto vc = w_vc_.parameters();   out.insert(out.end(), vc.begin(), vc.end());
        auto n2 = norm2_->parameters(); out.insert(out.end(), n2.begin(), n2.end());
        auto f = ff_.parameters();      out.insert(out.end(), f.begin(), f.end());
        return out;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> out;
        auto n1 = norm1_->param_gradients(); out.insert(out.end(), n1.begin(), n1.end());
        auto q = w_q_.param_gradients();     out.insert(out.end(), q.begin(), q.end());
        auto k = w_k_.param_gradients();     out.insert(out.end(), k.begin(), k.end());
        auto v = w_v_.param_gradients();     out.insert(out.end(), v.begin(), v.end());
        auto o = w_o_.param_gradients();     out.insert(out.end(), o.begin(), o.end());
        auto kc = w_kc_.param_gradients();   out.insert(out.end(), kc.begin(), kc.end());
        auto vc = w_vc_.param_gradients();   out.insert(out.end(), vc.begin(), vc.end());
        auto n2 = norm2_->param_gradients(); out.insert(out.end(), n2.begin(), n2.end());
        auto f = ff_.param_gradients();      out.insert(out.end(), f.begin(), f.end());
        return out;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
        w_kc_.set_checkpoint_mode(enabled);
        w_vc_.set_checkpoint_mode(enabled);
        softmax_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        norm1_->clear_cache(); w_q_.clear_cache(); w_k_.clear_cache();
        w_v_.clear_cache(); w_o_.clear_cache(); w_kc_.clear_cache();
        w_vc_.clear_cache(); softmax_.clear_cache(); norm2_->clear_cache();
        ff_.clear_cache();
        residual2_cache_ = Tensor{};
        Q_re_cache_ = Tensor{}; K_cat_cache_ = Tensor{}; V_cat_cache_ = Tensor{};
        batch_cache_ = 0;
    }

    // 文档感知：记录本块输入（局部窗口）每位置的文档 id（batch-major）。
    // 传空 span 清除文档感知，退化为纯因果局部掩码。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty())
        {
            doc_ids_.clear();
            has_doc_ids_ = false;
            mask_cache_ = Tensor{};   // 使掩码回到按 batch 缓存
            return;
        }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

    // 基类单输入 forward/backward 不适用于双输入块（仅经 ZiPTModel 组合使用）
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& /*input*/) override
    {
        (void)engine;
        return std::unexpected(Error{"ZiPTBlock: use forward(engine, input, memory_input)"});
    }
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& /*grad_output*/) override
    {
        (void)engine;
        return std::unexpected(Error{"ZiPTBlock: use backward(engine, grad, grad_C_out)"});
    }

    // 前向：局部 Y (d_model, batch·W) + 记忆 C (d_model, batch·M) → (d_model, batch·W)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input, const Tensor& memory_input)
    {
        const std::size_t total = input.cols();
        const std::size_t batch = (window_ > 0) ? (total / window_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;

        auto Q = w_q_.forward(engine, *n1);   // (d_model, batch·W)
        if (!Q) return Q;
        auto Ky = w_k_.forward(engine, *n1);
        if (!Ky) return Ky;
        auto Vy = w_v_.forward(engine, *n1);
        if (!Vy) return Vy;
        auto Kc = w_kc_.forward(engine, memory_input);   // (d_model, batch·M)
        if (!Kc) return Kc;
        auto Vc = w_vc_.forward(engine, memory_input);
        if (!Vc) return Vc;

        // rearrange 为 (batch*H*d_k, ·) 批量化布局
        auto Q_re = engine.rearrange_3d(*Q, H_dk, batch, window_, false);
        if (!Q_re) return std::unexpected(Q_re.error());
        auto Ky_re = engine.rearrange_3d(*Ky, H_dk, batch, window_, false);
        if (!Ky_re) return std::unexpected(Ky_re.error());
        auto Vy_re = engine.rearrange_3d(*Vy, H_dk, batch, window_, false);
        if (!Vy_re) return std::unexpected(Vy_re.error());
        auto Kc_re = engine.rearrange_3d(*Kc, H_dk, batch, memory_, false);
        if (!Kc_re) return std::unexpected(Kc_re.error());
        auto Vc_re = engine.rearrange_3d(*Vc, H_dk, batch, memory_, false);
        if (!Vc_re) return std::unexpected(Vc_re.error());

        // 序列维拼接：记忆在前，局部在后
        auto K_cat = concat_cols(engine, *Kc_re, *Ky_re);   // (batch*H*d_k, M+W)
        if (!K_cat) return std::unexpected(K_cat.error());
        auto V_cat = concat_cols(engine, *Vc_re, *Vy_re);
        if (!V_cat) return std::unexpected(V_cat.error());

        // S = batched_matmul(Q_re, K_cat, BH, transA=true, alpha=scale) → (batch*H*W, M+W)
        auto S = engine.batched_matmul(*Q_re, *K_cat, BH, true, false, scale_);
        if (!S) return std::unexpected(S.error());
        // 施加记忆-局部联合掩码
        auto mask = build_mask_(engine, batch);
        if (!mask) return std::unexpected(mask.error());
        auto ma = engine.add_inplace(*S, *mask);
        if (!ma) return std::unexpected(ma.error());
        // A = softmax(S)
        auto attn = softmax_.forward(engine, *S);
        if (!attn) return std::unexpected(attn.error());
        // O_re = batched_matmul(V_cat, A, BH, false, true) → (batch*H*d_k, W)
        auto O_re = engine.batched_matmul(*V_cat, *attn, BH, false, true);
        if (!O_re) return std::unexpected(O_re.error());
        // O = rearrange back → (d_model, batch·W)
        auto O = engine.rearrange_3d(*O_re, H_dk, batch, window_, true);
        if (!O) return std::unexpected(O.error());
        auto out_attn = w_o_.forward(engine, *O);
        if (!out_attn) return out_attn;

        auto r2 = dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(*out_attn),
            input.rows(), input.cols());
        if (!r2) return std::unexpected(r2.error());
        if (!checkpoint_mode_)
            residual2_cache_ = *r2;

        auto n2 = norm2_->forward(engine, *r2);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        if (!checkpoint_mode_)
        {
            Q_re_cache_ = std::move(*Q_re);
            K_cat_cache_ = std::move(*K_cat);
            V_cat_cache_ = std::move(*V_cat);
            batch_cache_ = batch;
        }
        return dsl::compute(engine,
            dsl::leaf(*r2) + dsl::leaf(*f),
            r2->rows(), r2->cols());
    }

    // 反向：grad_output (d_model, batch·W)；grad_C_out 累加记忆 C 的梯度
    // 返回对局部输入 Y 的梯度 (d_model, batch·W)
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output, Tensor& grad_C_out)
    {
        const std::size_t batch = batch_cache_;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;
        auto grad_r1 = dsl::compute(engine,
            dsl::leaf(grad_output) + dsl::leaf(*b_n2),
            grad_output.rows(), grad_output.cols());
        if (!grad_r1) return std::unexpected(grad_r1.error());

        // ── 联合注意力反向（materialized 路径，镜像 AttentionBase 旧路径） ──
        auto grad_O = w_o_.backward(engine, *grad_r1);   // (d_model, batch·W)
        if (!grad_O) return grad_O;
        auto grad_concat_re = engine.rearrange_3d(*grad_O, H_dk, batch, window_, false);
        if (!grad_concat_re) return std::unexpected(grad_concat_re.error());
        const Tensor& A = softmax_.output_cache();

        // grad_V_cat = batched_matmul(grad_O, A, BH, false, false) → (batch*H*d_k, M+W)
        auto grad_V_cat = engine.batched_matmul(*grad_concat_re, A, BH, false, false);
        if (!grad_V_cat) return std::unexpected(grad_V_cat.error());
        // grad_A = batched_matmul(grad_O, V_cat, BH, true, false) → (batch*H*W, M+W)
        auto grad_A = engine.batched_matmul(*grad_concat_re, V_cat_cache_, BH, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());
        // grad_Q_re = batched_matmul(K_cat, grad_S, BH, false, true, scale) → (batch*H*d_k, W)
        auto grad_Q_re = engine.batched_matmul(K_cat_cache_, *grad_S, BH, false, true, scale_);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        // grad_K_cat = batched_matmul(Q_re, grad_S, BH, false, false, scale) → (batch*H*d_k, M+W)
        auto grad_K_cat = engine.batched_matmul(Q_re_cache_, *grad_S, BH, false, false, scale_);
        if (!grad_K_cat) return std::unexpected(grad_K_cat.error());

        // 沿序列维拆分 grad_K/grad_V 的记忆与局部部分（transpose → slice_rows → transpose）
        auto split_cols_ = [&](const Tensor& t)
            -> Result<std::pair<Tensor, Tensor>>
        {
            auto tT = engine.transpose(t);   // (M+W, batch*H*d_k)
            if (!tT) return std::unexpected(tT.error());
            auto memT = engine.slice_rows(*tT, 0, memory_);       // (M, batch*H*d_k)
            if (!memT) return std::unexpected(memT.error());
            auto locT = engine.slice_rows(*tT, memory_, window_); // (W, batch*H*d_k)
            if (!locT) return std::unexpected(locT.error());
            auto mem = engine.transpose(*memT);   // (batch*H*d_k, M)
            if (!mem) return std::unexpected(mem.error());
            auto loc = engine.transpose(*locT);   // (batch*H*d_k, W)
            if (!loc) return std::unexpected(loc.error());
            return std::pair<Tensor, Tensor>{std::move(*mem), std::move(*loc)};
        };

        auto k_split = split_cols_(*grad_K_cat);
        if (!k_split) return std::unexpected(k_split.error());
        auto v_split = split_cols_(*grad_V_cat);
        if (!v_split) return std::unexpected(v_split.error());

        // ── 局部路径梯度 ──
        auto grad_Q = engine.rearrange_3d(*grad_Q_re, H_dk, batch, window_, true);
        if (!grad_Q) return std::unexpected(grad_Q.error());
        auto gq = w_q_.backward(engine, *grad_Q);
        if (!gq) return gq;
        auto grad_Ky = engine.rearrange_3d(k_split->second, H_dk, batch, window_, true);
        if (!grad_Ky) return std::unexpected(grad_Ky.error());
        auto gk = w_k_.backward(engine, *grad_Ky);
        if (!gk) return gk;
        auto grad_Vy = engine.rearrange_3d(v_split->second, H_dk, batch, window_, true);
        if (!grad_Vy) return std::unexpected(grad_Vy.error());
        auto gv = w_v_.backward(engine, *grad_Vy);
        if (!gv) return gv;
        auto g_attn = dsl::compute(engine,
            dsl::leaf(*gq) + dsl::leaf(*gk),
            gq->rows(), gq->cols());
        if (!g_attn) return std::unexpected(g_attn.error());
        auto grad_n1 = dsl::compute(engine,
            dsl::leaf(*g_attn) + dsl::leaf(*gv),
            g_attn->rows(), g_attn->cols());
        if (!grad_n1) return std::unexpected(grad_n1.error());
        auto b_n1 = norm1_->backward(engine, *grad_n1);
        if (!b_n1) return b_n1;
        auto grad_x = dsl::compute(engine,
            dsl::leaf(*grad_r1) + dsl::leaf(*b_n1),
            grad_r1->rows(), grad_r1->cols());
        if (!grad_x) return std::unexpected(grad_x.error());

        // ── 记忆路径梯度（累加进 grad_C_out） ──
        auto grad_Kc = engine.rearrange_3d(k_split->first, H_dk, batch, memory_, true);
        if (!grad_Kc) return std::unexpected(grad_Kc.error());
        auto gkc = w_kc_.backward(engine, *grad_Kc);
        if (!gkc) return gkc;
        auto grad_Vc = engine.rearrange_3d(v_split->first, H_dk, batch, memory_, true);
        if (!grad_Vc) return std::unexpected(grad_Vc.error());
        auto gvc = w_vc_.backward(engine, *grad_Vc);
        if (!gvc) return gvc;
        { auto a1 = engine.add_inplace(grad_C_out, *gkc); if (!a1) return std::unexpected(a1.error()); }
        { auto a2 = engine.add_inplace(grad_C_out, *gvc); if (!a2) return std::unexpected(a2.error()); }

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ZiPTModel — AttnZip 解码器（zipt：zip + GPT）
//
//   token_emb → (+pos_enc) → [历史 H 压缩为记忆 C] → N × ZiPTBlock(窗口 W, C)
//   → LN → LM Head
//
// 两种模式（由 window 参数决定，W=seq_len 为旧行为向后兼容）：
//   * W = L（window==seq_len，旧行为）：压缩整条序列 X → C，块对 [C;X] 联合注意力。
//     注意：此时块注意力键 = M+L，复杂度 O(L²)，实际上「无压缩」。
//   * W < L（L = W + C，新行为）：把输入 X(长度 L) 按 batch 拆为
//       历史 H = X[0 : L-W]（长度 C = L-W）→ 压缩为记忆 C (M 个 token)
//       窗口 W_seq = X[L-W : L]（长度 W）→ 块直接注意力
//     每块对 [M 记忆 ; W 窗口] 联合注意力，键数 = M+W（注意力预算）。
//     复杂度：压缩 O(M·C·d) 线性于 L；块注意力 O(W·(M+W)) 对 L 为常数。
//     即：用「传统上下文 (M+W)」的注意力预算处理「总长度 L」的上下文。
//
// 训练范式：对完整序列 X 一次性压缩历史为记忆 C，随后每块对 [C ; 窗口] 联合
// 注意力，loss 仅在窗口 W 上计算（历史被压缩器"消费"，不直接预测）。
// ══════════════════════════════════════════════════════════════════════════
class ZiPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;    // L：总上下文长度
    std::size_t window_;     // W：局部窗口（=seq_len 时为旧行为 W=L 无压缩）
    std::size_t hist_len_;   // C = L - W：历史长度（被压缩，split 模式 >0）
    std::size_t memory_;     // M
    bool split_;             // true = W<L 新行为（历史/窗口分离）；false = 旧行为 W=L

    Tensor token_emb_;
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;
    CrossAttention compressor_;
    std::vector<ZiPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    Tensor stored_tokens_tensor_;
    std::size_t batch_size_ = 0;
    // 文档感知：每位置文档 id（batch-major，size = batch*seq_len_）；空=无。
    // forward 时切出块输入（窗口）对应的子段下发给各块做局部文档掩码。
    std::vector<std::size_t> doc_ids_;

    // 沿 batch-major 列序把 x (d_model, batch·L) 拆为 历史 H (d_model, batch·C)
    // 与 窗口 W_seq (d_model, batch·W)。列序 i=b·L+t（batch 在列方向）。
    [[nodiscard]] Result<std::pair<Tensor, Tensor>>
    split_history_window_(ComputeEngine& engine, const Tensor& x,
                          std::size_t batch, std::size_t L,
                          std::size_t C, std::size_t W)
    {
        auto x_re = engine.rearrange_3d(x, d_model_, batch, L, false); // (batch·d, L)
        if (!x_re) return std::unexpected(x_re.error());
        auto xT = engine.transpose(*x_re);                            // (L, batch·d)
        if (!xT) return std::unexpected(xT.error());
        auto hT = engine.slice_rows(*xT, 0, C);                       // (C, batch·d)
        if (!hT) return std::unexpected(hT.error());
        auto wT = engine.slice_rows(*xT, C, W);                       // (W, batch·d)
        if (!wT) return std::unexpected(wT.error());
        auto h_re = engine.transpose(*hT);                            // (batch·d, C)
        if (!h_re) return std::unexpected(h_re.error());
        auto w_re = engine.transpose(*wT);                            // (batch·d, W)
        if (!w_re) return std::unexpected(w_re.error());
        auto H = engine.rearrange_3d(*h_re, d_model_, batch, C, true);   // (d, batch·C)
        if (!H) return std::unexpected(H.error());
        auto Ws = engine.rearrange_3d(*w_re, d_model_, batch, W, true);  // (d, batch·W)
        if (!Ws) return std::unexpected(Ws.error());
        return std::pair<Tensor, Tensor>{std::move(*H), std::move(*Ws)};
    }

    // 合并 历史梯度 (d, batch·C) 与 窗口梯度 (d, batch·W) → (d, batch·L)
    [[nodiscard]] Result<Tensor>
    merge_history_window_(ComputeEngine& engine, const Tensor& gradH,
                          const Tensor& gradW, std::size_t batch,
                          std::size_t L, std::size_t C, std::size_t W)
    {
        auto h_re = engine.rearrange_3d(gradH, d_model_, batch, C, false); // (batch·d, C)
        if (!h_re) return std::unexpected(h_re.error());
        auto w_re = engine.rearrange_3d(gradW, d_model_, batch, W, false); // (batch·d, W)
        if (!w_re) return std::unexpected(w_re.error());
        auto hT = engine.transpose(*h_re);   // (C, batch·d)
        if (!hT) return std::unexpected(hT.error());
        auto wT = engine.transpose(*w_re);   // (W, batch·d)
        if (!wT) return std::unexpected(wT.error());
        Tensor dstT = engine.create_tensor(L, batch * d_model_);
        auto r1 = engine.insert_rows(dstT, 0, *hT);
        if (!r1) return std::unexpected(r1.error());
        auto r2 = engine.insert_rows(dstT, C, *wT);
        if (!r2) return std::unexpected(r2.error());
        auto dst = engine.transpose(dstT);   // (batch·d, L)
        if (!dst) return std::unexpected(dst.error());
        return engine.rearrange_3d(*dst, d_model_, batch, L, true);  // (d, batch·L)
    }

public:
    ZiPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
              std::size_t window,
              std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
              std::size_t memory_tokens,
              PosEncodingType pos_enc_type = PosEncodingType::Learned,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          window_(window), memory_(memory_tokens),
          compressor_(d_model, memory_tokens,
                      (window < seq_len) ? (seq_len - window) : seq_len),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        // 归约 W：W==0 或 W>seq_len → 回退到 W=L（旧行为）
        if (window_ == 0 || window_ > seq_len_) window_ = seq_len_;
        hist_len_ = seq_len_ - window_;
        split_    = (window_ < seq_len_);
        const std::size_t block_window = split_ ? window_ : seq_len_;

        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, block_window, memory_tokens,
                                 norm_type, activation);
        switch (pos_enc_type)
        {
            case PosEncodingType::Learned:
                pos_encoder_ = std::make_unique<LearnedPositionEncoder>(d_model, seq_len);
                break;
            case PosEncodingType::Sinusoidal:
                pos_encoder_ = std::make_unique<SinusoidalPositionEncoder>(d_model, seq_len);
                break;
            default:
                pos_encoder_ = std::make_unique<NoPositionEncoder>();
                break;
        }
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
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
        { auto r = engine.zero(grad_token_emb_); if (!r) return std::unexpected(r.error()); }

        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        auto r1 = compressor_.init(engine);
        if (!r1) return std::unexpected(r1.error());
        for (auto& b : blocks_)
        {
            auto r = b.init(engine);
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

    // ZiPT 未实现 forward_recompute（压缩器 CrossAttention 与 ZiPTBlock 都无
    // 重计算路径）。梯度检查点（激活重计算）需要 forward_recompute 才能在
    // backward 时重建缓存；若被请求，明确中止而非静默产生错误结果。
    // 已知限制：见 docs/12-innovative-designs.md §10。
    void set_checkpoint_every(std::size_t /*stride*/) override
    {
        std::fprintf(stderr, "FATAL: ZiPTModel does not support gradient checkpointing "
                             "(no forward_recompute); requested stride=%zu\n",
                     static_cast<std::size_t>(-1));
        std::abort();
    }

    // 文档感知：记录每位置文档 id（batch-major，size = batch*seq_len_）。
    // forward 时按窗口切段下发给各块，使块内局部注意力在文档边界（含 [PAD]）隔离。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        auto cp = compressor_.parameters();
        p.insert(p.end(), cp.begin(), cp.end());
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
        auto cg = compressor_.param_gradients();
        g.insert(g.end(), cg.begin(), cg.end());
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

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq_len = seq_len_;   // ZiPT 固定 seq_len_；输入必须等于配置长度
        const std::size_t batch = input.cols();
        if (input.rows() != seq_len_)
            return std::unexpected(Error{"ZiPTModel forward: input rows != seq_len_"});
        batch_size_ = batch;

        auto input_T = engine.transpose(input);   // (batch, seq) batch-major
        if (!input_T) return std::unexpected(input_T.error());
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());
        auto st = engine.clone(*input_T);
        if (!st) return std::unexpected(st.error());
        stored_tokens_tensor_ = std::move(*st);
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        auto x_res = pos_encoder_->apply(engine, *all_T, batch, seq_len);
        if (!x_res) return std::unexpected(x_res.error());
        Tensor x = std::move(*x_res);

        // 阶段一：历史压缩 → 记忆 C；阶段二：块对窗口联合注意力
        Tensor C;
        Tensor block_input;
        if (split_)
        {
            auto split = split_history_window_(engine, x, batch, seq_len_, hist_len_, window_);
            if (!split) return std::unexpected(split.error());
            auto c_res = compressor_.forward(engine, split->first);   // H → C (d, batch·M)
            if (!c_res) return c_res;
            C = std::move(*c_res);
            block_input = std::move(split->second);                   // W_seq (d, batch·W)
        }
        else
        {
            auto c_res = compressor_.forward(engine, x);              // 旧行为：压缩整条 X
            if (!c_res) return c_res;
            C = std::move(*c_res);
            block_input = x;
        }

        // 阶段二：逐块联合注意力（对 [C ; 窗口]）
        // 文档感知：把块输入（窗口）对应的 doc 子段下发给各块做局部文档/PAD 掩码。
        if (!doc_ids_.empty())
        {
            const std::size_t win_cols = split_ ? window_ : seq_len_;
            std::vector<std::size_t> block_doc_ids(batch * win_cols);
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t t = 0; t < win_cols; ++t)
                    block_doc_ids[b * win_cols + t] =
                        doc_ids_[b * seq_len_ + (split_ ? hist_len_ + t : t)];
            for (auto& blk : blocks_) blk.set_doc_ids(block_doc_ids);
        }
        for (auto& b : blocks_)
        {
            auto r = b.forward(engine, block_input, C);
            if (!r) return r;
            block_input = std::move(*r);
        }

        auto ln = ln_f_->forward(engine, block_input);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;
        const std::size_t batch = batch_size_;

        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        auto b_ln = ln_f_->backward(engine, *b_lm);
        if (!b_ln) return b_ln;
        Tensor grad_w = std::move(*b_ln);   // (d_model, batch·W) 块输出梯度

        // 记忆 C 梯度：跨块累加
        Tensor grad_C = engine.create_tensor(d_model_, batch * memory_);
        { auto z = engine.zero(grad_C); if (!z) return std::unexpected(z.error()); }

        for (std::size_t i = blocks_.size(); i-- > 0;)
        {
            auto br = blocks_[i].backward(engine, grad_w, grad_C);
            if (!br) return br;
            grad_w = std::move(*br);
        }

        Tensor grad_x;
        if (split_)
        {
            // 压缩器反向：grad_C → 对历史 H 的梯度 (d, batch·C)
            auto gxh = compressor_.backward(engine, grad_C);
            if (!gxh) return gxh;
            // 合并 [grad_H ; grad_W] → (d, batch·L)
            auto merged = merge_history_window_(engine, *gxh, grad_w,
                                                batch, seq_len_, hist_len_, window_);
            if (!merged) return std::unexpected(merged.error());
            grad_x = std::move(*merged);
        }
        else
        {
            // 旧行为：压缩器对整条 X 的梯度与块梯度同形状，直接累加
            auto gxc = compressor_.backward(engine, grad_C);
            if (!gxc) return gxc;
            auto gx_sum = dsl::compute(engine,
                dsl::leaf(grad_w) + dsl::leaf(*gxc),
                grad_w.rows(), grad_w.cols());
            if (!gx_sum) return std::unexpected(gx_sum.error());
            grad_x = std::move(*gx_sum);
        }

        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());
        auto pr = pos_encoder_->backward(engine, *grad_T, batch, seq_len);
        if (!pr) return std::unexpected(pr.error());
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        Matrix grad_input(seq_len, batch, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── 采样生成（重计算式，无 KV cache） ────────────────────────
    // 每步把「当前上下文（prompt + 已生成，滑动窗口截断到 seq_len）」
    // 重新 forward（内部重做压缩 C + 联合注意力），取末位 logits 采样。
    // 复杂度 O(seq²)（每步 O(seq)），对首个 CLI 接入足够；KV cache 优化留待后续。
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

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            // 滑动窗口截断到 seq_len_
            std::size_t start = 0;
            if (context.size() > seq_len_) start = context.size() - seq_len_;
            const std::size_t n = context.size() - start;

            // 输入 (seq_len_, 1)：使 forward 始终处理整窗（压缩机/块按固定长度构建）。
            // 旧行为 (W=L)：前 n 个为上下文，其余 pad_id=0 填充；末位 logits 取 n-1。
            // 新行为 (W<L)：右对齐，末位真实 token 落到窗口末（位置 seq_len_-1），
            //   forward 输出仅覆盖窗口 (vocab, W)，logits 取窗口末列 window_-1。
            const std::size_t pred_col = split_ ? (window_ - 1) : (n - 1);
            Matrix in(seq_len_, 1);
            if (split_)
            {
                const std::size_t pad_head = seq_len_ - n;
                for (std::size_t i = 0; i < seq_len_; ++i)
                    in.set_value_unchecked(i, 0,
                        static_cast<Scalar>((i >= pad_head) ? context[start + (i - pad_head)] : 0));
            }
            else
            {
                for (std::size_t i = 0; i < seq_len_; ++i)
                    in.set_value_unchecked(i, 0,
                        static_cast<Scalar>((i < n) ? context[start + i] : 0));
            }
            auto in_t = engine.from_matrix(in);
            if (!in_t) return std::unexpected(in_t.error());
            auto logits = forward(engine, *in_t);
            if (!logits) return std::unexpected(logits.error());
            auto lm = engine.to_matrix(*logits);
            if (!lm) return std::unexpected(lm.error());

            // 末位（最后真实位置）logits
            std::vector<Scalar> last(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last[v] = lm->at_unchecked(v, pred_col);

            if (temperature > 0.0 && temperature != 1.0)
                for (auto& x : last) x /= temperature;

            Scalar max_val = last[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, last[v]);
            Scalar sum_exp = 0;
            for (auto& x : last) { x = std::exp(x - max_val); sum_exp += x; }
            for (auto& x : last) x /= sum_exp;

            std::size_t next;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cum = 0;
                next = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cum += last[v];
                    if (r <= cum) { next = v; break; }
                }
            }
            else
            {
                next = 0;
                Scalar best = last[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (last[v] > best) { best = last[v]; next = v; }
            }

            context.push_back(next);
            if (step >= min_new_tokens && next == eos_token_id) break;
            generated.push_back(next);
        }
        return generated;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ReLULinearAttention — ReLU 线性注意力（RLA / RAPT 核心层）
//
// 算法（docs/15-rapt-algorithm.md §3，causal 版；bidirectional 亦支持）：
//   q' = ReLU(RoPE(q)), k' = ReLU(RoPE(k)), v = V·W_v（V 不做 ReLU）
//   分子 num_t  = Σ_{i∈S_t} (q'_t·k'_i) v_i  = B_t · q'_t,   B_t = Σ_{i∈S_t} v_i k'_i^T
//   分母 den_t  = sqrt( Σ_{i∈S_t} (q'_t·k'_i)^2 + eps )
//               = sqrt( q'_t^T A_t q'_t + eps ),  A_t = Σ_{i∈S_t} k'_i k'_i^T
//   out_t = num_t / den_t
//   其中 S_t = { i<=t }（causal）或 { 全部 }（bidirectional）。
//
// 关键性质：
//   * 复杂度 O(L·d_k²)，无 O(L²) 物化 —— 分子/分母都靠运行态前缀和。
//     分母 L2 恒等式：Σ_i (q'·k'_i)² = q'^T (Σ k'_i k'_i^T) q'（避免显式 L×L 分数矩阵）。
//   * 施加顺序必须 RoPE → ReLU（RoPE 若在 ReLU 之后，旋转产生的负值被截断 → 丢位置信息）。
//   * 数值：分母 L2 归一化（余弦式约束）；禁 1/sqrt(d_k) 缩放；稳定性靠 LayerNorm + eps。
//
// 实现说明：causal 的前缀和递推本质是序列化的，无法折叠为 matmul 批次，
//   故核心扫描用 CPU Matrix 标量循环，经 to_matrix/from_matrix 适配任意后端
//   （GPU 走 staging 往返，v1 正确性优先）。
// ══════════════════════════════════════════════════════════════════════════
} // namespace nn

