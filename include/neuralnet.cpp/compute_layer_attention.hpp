#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_softmax.hpp"

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

class RotaryEmbedding
{
private:
    std::size_t d_k_ = 0;
    std::size_t seq_cached_ = 0;   // 缓存表的列数（全表应用）
    std::size_t pos_offset_ = 0;   // 绝对位置偏移（滑动窗生成时用，0=从 0 起）
    Tensor cos_cache_;             // (d_k, seq_cached_)
    Tensor sin_cache_;             // (d_k, seq_cached_)

    // 把单个位置 pos 的 cos/sin 写入指定列（rebuild/apply_step 共用）。
    // LLaMA 式：cos 沿 d 维 = cat(freqs, freqs)（前后半相同），
    // 配合 rotate_half（前后半交换+前半取负）构成 2×2 旋转块。
    void fill_pos_column_(Matrix& c, Matrix& s, std::size_t pos, std::size_t col) const
    {
        const std::size_t half = d_k_ / 2;
        const Scalar pd = static_cast<Scalar>(pos);
        for (std::size_t j = 0; j < half; ++j)
        {
            const Scalar theta = pd / std::pow(Scalar{10000},
                static_cast<Scalar>(2 * j) / static_cast<Scalar>(d_k_));
            const Scalar cv = std::cos(theta);
            const Scalar sv = std::sin(theta);
            c.set_value_unchecked(j,        col, cv);
            c.set_value_unchecked(half + j, col, cv);
            s.set_value_unchecked(j,        col, sv);
            s.set_value_unchecked(half + j, col, sv);
        }
    }

    // 重建全表 (d_k, seq)：cos/sin 按维度对交错重复；位置 = pos + pos_offset_
    // （绝对位置偏移：滑动窗生成时，输入被截断到窗口，但位置应从真实起点算起）
    [[nodiscard]] Result<void> rebuild(ComputeEngine& engine, std::size_t seq)
    {
        Matrix c(d_k_, seq), s(d_k_, seq);
        for (std::size_t pos = 0; pos < seq; ++pos)
            fill_pos_column_(c, s, pos + pos_offset_, pos);
        auto cr = engine.from_matrix(c);
        if (!cr) return std::unexpected(cr.error());
        cos_cache_ = std::move(*cr);
        auto sr = engine.from_matrix(s);
        if (!sr) return std::unexpected(sr.error());
        sin_cache_ = std::move(*sr);
        seq_cached_ = seq;
        return {};
    }

    // 构造 RoPE 内联表达式（forward 用 Add 结尾，backward 用 Sub 结尾）
    // 表达式文本只写在本 Layer（apply/apply_step），AOT 收集（scan_exprs）
    // 用同一段代码 dry-run 折叠出结构并合成融合 shader——绝不漂移。

public:
    RotaryEmbedding() = default;
    explicit RotaryEmbedding(std::size_t d_k)
        : d_k_(d_k)
    {
    }

    [[nodiscard]] Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    [[nodiscard]] std::size_t d_k() const noexcept { return d_k_; }

    // 设置绝对位置偏移（滑动窗生成：输入截断到窗口，位置从真实起点算起）。
    // 改变后强制下次 apply 重建 cos/sin 表。
    void set_position_offset(std::size_t off)
    {
        if (off != pos_offset_)
        {
            pos_offset_ = off;
            seq_cached_ = 0;   // 使下次 apply 触发 rebuild
        }
    }

    // 全表应用：q 为 (batch*H*d_k, seq)，cos/sin 为 (d_k, seq) 短表
    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& q,
        std::size_t seq, bool backward)
    {
        if (d_k_ == 0 || d_k_ % 2 != 0)
            return std::unexpected(Error{"RotaryEmbedding::apply: d_k must be positive and even"});
        if (seq != seq_cached_)
        {
            auto r = rebuild(engine, seq);
            if (!r) return std::unexpected(r.error());
        }
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        // 内联 RoPE 表达式（LLaMA half-swap；backward 为旋转正交，逆 = 反角）：
        //   forward:  out = q·cos + rotate_half(q)·sin
        //   backward: out = q·cos − rotate_half(q)·sin
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
                q.rows(), q.cols());
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
            q.rows(), q.cols());
    }

    // 增量推理：q 为 (H*d_k, 1)，位置 = pos（cur_len）
    // 直接生成 (d_k,1) 位置表，避免对全表切片取列。
    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& q, std::size_t pos, bool backward)
    {
        if (d_k_ == 0 || d_k_ % 2 != 0)
            return std::unexpected(Error{"RotaryEmbedding::apply_step: d_k must be positive and even"});
        Matrix c(d_k_, 1), s(d_k_, 1);
        fill_pos_column_(c, s, pos, 0);
        auto cr = engine.from_matrix(c);
        if (!cr) return std::unexpected(cr.error());
        auto sr = engine.from_matrix(s);
        if (!sr) return std::unexpected(sr.error());
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(*cr, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
                q.rows(), q.cols());
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(*cr, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
            q.rows(), q.cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// AttentionBase — 多头注意力基类（批量化：消除 per-head 和 per-sample 循环）
//
// 提取 MultiHeadAttention 与 CausalSelfAttention 的公共逻辑：
//   - 完全相同的成员变量、参数/梯度接口
//   - forward/backward 仅在「scores 之后、softmax 之前」是否施加掩码上有差异
//
// 算法（只在此处，不在 Engine/Shader）：
//   Q = W_q × x, K = W_k × x, V = W_v × x  (三个 Linear 投影)
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//
//   批量化关键：用 rearrange_3d 把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
//   使 batched_matmul 能按 batch*H 切分行块，单次 dispatch 处理所有样本和所有头。
//
//   Q_re = rearrange_3d(Q, H*d_k, batch, seq) → (batch*H*d_k, seq)
//   S = batched_matmul(Q_re, K_re, batch*H, transA=true, alpha=scale) → (batch*H*seq, seq)
//     （scale = 1/sqrt(d_k) 通过 matmul 的 alpha 系数折进写出，省去独立 scale pass）
//   S = apply_mask_(S)         ← 子类钩子（默认 no-op = MHA 行为）
//   A = softmax(S)  — 行级归一化，堆叠布局下天然正确
//   O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
//   O = rearrange_3d(O_re, H*d_k, batch, seq, inverse=true) → (H*d_k, batch*seq)
//   out = W_o × O
//
// 输入形状: (d_model, batch * seq_len)，输出形状: (d_model, batch * seq_len)
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断
// ══════════════════════════════════════════════════════════════════════════
class AttentionBase : public Layer
{
protected:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;   // 单样本序列长度（0 = 单样本，cols 即 seq）
    Scalar scale_;

    Linear w_q_;
    Linear w_k_;
    Linear w_v_;
    Linear w_o_;
    Softmax softmax_;

    // RoPE（pos_enc == RoPE 时启用）：作用在 Q/K 的 d_k 维（每头），
    // 施加点在 Q/K 完成 rearrange 之后（列=position）。
    bool use_rope_ = false;
    RotaryEmbedding rope_;

    // forward 缓存（rearranged 版本，供 backward 直接使用）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged
    Tensor attn_cache_;                    // (batch*H*seq, seq) 旧路径缓存

    // 两趟式缓存（M6→S7）：m/l 替代 attn_cache_（不物化得分矩阵）；
    // W_cache_（S7）为物化的 softmax 权重 (BH*seq, seq)，backward 复用
    Tensor m_cache_, l_cache_;   // (batch*H*seq, 1)：行 max / softmax 分母
    Tensor W_cache_;             // (batch*H*seq, seq)：softmax 权重（S7 物化）
    bool two_pass_active_ = false;  // forward 是否走了两趟式路径（backward 读取）

    // ── S7：掩码输入张量钩子（IR 掩码表达式用；空 = 无该分量）──────────
    //   mask_slopes_   — ALiBi 按头斜率 (1, num_heads)，batch_mod(num_heads) 索引
    //   mask_doc_col_  — 每行文档 id (BH*seq, 1)，row_broadcast（doc_col[b*H*seq+h*seq+i]）
    //   mask_doc_ids_  — 每位置文档 id (1, batch*seq)，batch_col(seq) 切片
    [[nodiscard]] virtual const Tensor* mask_slopes_() const { return nullptr; }
    [[nodiscard]] virtual const Tensor* mask_doc_col_() const { return nullptr; }
    [[nodiscard]] virtual const Tensor* mask_doc_ids_() const { return nullptr; }

    // ── S7：掩码 DSL 表达式（causal 恒有 + alibi/doc 按钩子）──────────
    // 4 种组合（causal / causal+alibi / causal+doc / causal+alibi+doc）各自
    // 生成确定结构（scan dry-run 覆盖全部组合，闭合世界两端一致）。
    // blocked≠0 的位置屏蔽（-inf）。
    const Scalar kNegInf_ = -std::numeric_limits<Scalar>::infinity();
    [[nodiscard]] bool use_alibi_mask_() const { return mask_slopes_() != nullptr; }
    [[nodiscard]] bool use_doc_mask_() const { return mask_doc_col_() != nullptr; }
    template <typename E>
    auto masked_causal_(const E& scores, std::size_t /*seq*/) const
    {
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        return scores + dsl::select(causal != Scalar{0}, kNegInf_, Scalar{0});
    }
    template <typename E>
    auto masked_alibi_(const E& scores, std::size_t /*seq*/) const
    {
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        const auto alibi = dsl::batch_mod(*mask_slopes_(), num_heads_)
                         * (dsl::col() - dsl::row());
        return scores + dsl::select(causal != Scalar{0}, kNegInf_, alibi);
    }
    template <typename E>
    auto masked_doc_(const E& scores, std::size_t seq) const
    {
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        const auto blocked = causal + dsl::select(
            dsl::row_broadcast(*mask_doc_col_()) != dsl::batch_col(*mask_doc_ids_(), seq),
            Scalar{1}, Scalar{0});
        return scores + dsl::select(blocked != Scalar{0}, kNegInf_, Scalar{0});
    }
    template <typename E>
    auto masked_alibi_doc_(const E& scores, std::size_t seq) const
    {
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        const auto blocked = causal + dsl::select(
            dsl::row_broadcast(*mask_doc_col_()) != dsl::batch_col(*mask_doc_ids_(), seq),
            Scalar{1}, Scalar{0});
        const auto alibi = dsl::batch_mod(*mask_slopes_(), num_heads_)
                         * (dsl::col() - dsl::row());
        return scores + dsl::select(blocked != Scalar{0}, kNegInf_, alibi);
    }

    // ── 两趟式（M6→S7）决策钩子 ──────────────────────────────────────
    // 决定是否用两趟式注意力（S7 起恒 true：IR 融合路径不物化得分矩阵；
    // 掩码分量经 mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子读取）。
    struct TwoPassMask
    {
        bool use_two_pass = true;
    };
    [[nodiscard]] virtual Result<TwoPassMask> two_pass_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        (void)engine; (void)batch; (void)seq;
        return TwoPassMask{true};  // MHA：无偏置，两趟式
    }

    // 掩码钩子：子类重写以施加掩码，默认 no-op（MHA 行为）
    // 在 forward 中 scores（已含 alpha=scale 缩放）之后、softmax 之前调用
    [[nodiscard]] virtual Result<Tensor> apply_mask_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t batch, std::size_t seq)
    {
        (void)engine; (void)batch; (void)seq;
        return std::move(scores);
    }

    // 增量推理掩码钩子：在 forward_step 中 scale 之后、softmax 之前调用。
    // 默认 no-op：MHA 无掩码；CSA 因果掩码在增量推理中天然满足
    // （新 token 只能看到 cache 中的前文，无未来位置可掩）。
    // ALiBi 子类重写以施加线性距离偏置 -slope*(cur_len - j)。
    [[nodiscard]] virtual Result<Tensor> apply_mask_step_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t cur_len)
    {
        (void)engine; (void)cur_len;
        return std::move(scores);
    }

    // 在 backward 中 softmax 反向之后调用（掩码为常数，梯度直接穿透，默认 no-op）
    virtual void mask_backward_(
        ComputeEngine& /*engine*/, Tensor& /*grad_S*/,
        std::size_t /*batch*/, std::size_t /*seq*/) {}

public:
    AttentionBase(std::size_t d_model, std::size_t num_heads,
                  std::size_t seq_len = 0,
                  PosEncodingType pos_enc = PosEncodingType::Learned)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(d_model, d_model),
          w_k_(d_model, d_model),
          w_v_(d_model, d_model),
          w_o_(d_model, d_model),
          use_rope_(pos_enc == PosEncodingType::RoPE),
          rope_(d_model / num_heads)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "AttentionBase: d_model must be divisible by num_heads");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = w_q_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_k_.init(engine); if (!r2) return std::unexpected(r2.error());
        auto r3 = w_v_.init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = w_o_.init(engine); if (!r4) return std::unexpected(r4.error());
        if (use_rope_)
        {
            auto r5 = rope_.init(engine); if (!r5) return std::unexpected(r5.error());
        }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = w_q_.parameters();
        auto k = w_k_.parameters();
        auto v = w_v_.parameters();
        auto o = w_o_.parameters();
        p.insert(p.end(), k.begin(), k.end());
        p.insert(p.end(), v.begin(), v.end());
        p.insert(p.end(), o.begin(), o.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = w_q_.param_gradients();
        auto k = w_k_.param_gradients();
        auto v = w_v_.param_gradients();
        auto o = w_o_.param_gradients();
        g.insert(g.end(), k.begin(), k.end());
        g.insert(g.end(), v.begin(), v.end());
        g.insert(g.end(), o.begin(), o.end());
        return g;
    }

    // 梯度检查点：把模式传播给内部投影层与 softmax
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
        softmax_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        Q_cache_ = Tensor{};
        K_cache_ = Tensor{};
        V_cache_ = Tensor{};
        attn_cache_ = Tensor{};
        m_cache_ = Tensor{};
        l_cache_ = Tensor{};
        // 掩码/偏置描述子（two_pass_bias_ 指向子类 doc_ids/slopes 缓存）小而常驻，
        // 不随激活清理
        w_q_.clear_cache();
        w_k_.clear_cache();
        w_v_.clear_cache();
        w_o_.clear_cache();
        softmax_.clear_cache();
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Q_cache_.valid()) r.emplace_back(Q_cache_);
        if (K_cache_.valid()) r.emplace_back(K_cache_);
        if (V_cache_.valid()) r.emplace_back(V_cache_);
        if (m_cache_.valid()) r.emplace_back(m_cache_);
        if (l_cache_.valid()) r.emplace_back(l_cache_);
        auto wq = w_q_.activation_cache(); r.insert(r.end(), wq.begin(), wq.end());
        auto wk = w_k_.activation_cache(); r.insert(r.end(), wk.begin(), wk.end());
        auto wv = w_v_.activation_cache(); r.insert(r.end(), wv.begin(), wv.end());
        auto wo = w_o_.activation_cache(); r.insert(r.end(), wo.begin(), wo.end());
        auto sm = softmax_.activation_cache(); r.insert(r.end(), sm.begin(), sm.end());
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"AttentionBase forward: input shape mismatch"});

        const std::size_t total_seq = input.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            return std::unexpected(Error{"AttentionBase forward: cols not divisible by seq_len"});

        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        //    使 batched_matmul 能按 batch*H 切分行块
        //    局部 Q/K/V 承载 forward 计算；仅在非 checkpoint 模式下写入成员缓存。
        const std::size_t H_dk = num_heads_ * d_k_;
        Tensor Q, K, V;  // (batch*H*d_k, seq) rearranged
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false);
            if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false);
            if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false);
            if (!vr) return std::unexpected(vr.error());
            V = std::move(*vr);
        }
        else
        {
            // batch=1: rearrange 是恒等拷贝，跳过
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // 2.5 RoPE：对 Q/K 施加旋转位置编码（rearrange 后列=position，
        //    每 d_k 行一个头，cos/sin 短表按 RowMod 平铺）
        if (use_rope_)
        {
            auto qr2 = rope_.apply(engine, Q, seq, /*backward=*/false);
            if (!qr2) return std::unexpected(qr2.error());
            Q = std::move(*qr2);
            auto kr2 = rope_.apply(engine, K, seq, /*backward=*/false);
            if (!kr2) return std::unexpected(kr2.error());
            K = std::move(*kr2);
        }
        // 2.6 S7：scale（1/sqrt(d_k)）折进 Q（Q *= scale）：注意力表达式不含
        // scale 常量 → 结构与 d_k 无关（不同 d_k 共享融合 shader，闭合世界
        // key 稳定）。backward 的 grad_Q 相应补乘 scale。
        { auto qs = engine.scale_inplace(Q, scale_); if (!qs) return std::unexpected(qs.error()); }

        // 3-7. 注意力主体：两趟式 vs 旧路径（由掩码钩子决策）
        const std::size_t BH = batch * num_heads_;
        auto tpm = two_pass_mask_(engine, batch, seq);
        if (!tpm) return std::unexpected(tpm.error());
        Tensor concat_out;  // (batch*H*d_k, seq)
        if (tpm->use_two_pass)
        {
            // ── S7 IR 融合路径（M4-M6 → matmul 段 + 归约 + 普通 batched_matmul）──
            //   m = rowmax(scale·Q·K^T + mask)        → (BH*seq, 1)（不物化 QK^T）
            //   l = Σ_j exp(scale·Q·K^T + mask − m)   → (BH*seq, 1)
            //   W = softmax 归一化权重（物化 (BH*seq, seq)，backward 复用）
            //   O = W·V_t                              → (BH*seq, d_k)
            
            const bool use_slopes = use_alibi_mask_();
            const bool use_doc = use_doc_mask_();
            // 掩码组合选择器：同一选择用于 m/l/W 三个表达式（结构一致）。
            // 每个分支返回 Result<Tensor>（统一返回类型，内部表达式各异）。
            const auto compute_m = [&]() -> Result<Tensor> {
                if (use_slopes && use_doc)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_max(masked_alibi_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)),
                        BH * seq, seq);
                if (use_slopes)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_max(masked_alibi_(
                            dsl::matmul(Q, K, true, false, BH), seq)),
                        BH * seq, seq);
                if (use_doc)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_max(masked_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)),
                        BH * seq, seq);
                return dsl::compute_reduce(engine,
                    dsl::row_reduce_max(masked_causal_(
                        dsl::matmul(Q, K, true, false, BH), seq)),
                    BH * seq, seq);
            };
            const auto compute_l = [&](const Tensor& m_t) -> Result<Tensor> {
                if (use_slopes && use_doc)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_sum(dsl::exp(masked_alibi_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t))),
                        BH * seq, seq);
                if (use_slopes)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_sum(dsl::exp(masked_alibi_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t))),
                        BH * seq, seq);
                if (use_doc)
                    return dsl::compute_reduce(engine,
                        dsl::row_reduce_sum(dsl::exp(masked_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t))),
                        BH * seq, seq);
                return dsl::compute_reduce(engine,
                    dsl::row_reduce_sum(dsl::exp(masked_causal_(
                        dsl::matmul(Q, K, true, false, BH), seq)
                        - dsl::row_broadcast(m_t))),
                    BH * seq, seq);
            };
            const auto compute_W = [&](const Tensor& m_t, const Tensor& l_t) -> Result<Tensor> {
                if (use_slopes && use_doc)
                    return dsl::compute(engine,
                        dsl::exp(masked_alibi_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t)) / dsl::row_broadcast(l_t),
                        BH * seq, seq);
                if (use_slopes)
                    return dsl::compute(engine,
                        dsl::exp(masked_alibi_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t)) / dsl::row_broadcast(l_t),
                        BH * seq, seq);
                if (use_doc)
                    return dsl::compute(engine,
                        dsl::exp(masked_doc_(
                            dsl::matmul(Q, K, true, false, BH), seq)
                            - dsl::row_broadcast(m_t)) / dsl::row_broadcast(l_t),
                        BH * seq, seq);
                return dsl::compute(engine,
                    dsl::exp(masked_causal_(
                        dsl::matmul(Q, K, true, false, BH), seq)
                        - dsl::row_broadcast(m_t)) / dsl::row_broadcast(l_t),
                    BH * seq, seq);
            };
            // m = row_max(scale·Q·K^T + mask)
            auto m = compute_m();
            if (!m) return std::unexpected(m.error());
            // l = row_sum(exp(scale·Q·K^T + mask − m))
            auto l = compute_l(*m);
            if (!l) return std::unexpected(l.error());
            // W = exp(scale·Q·K^T + mask − m) / l（物化，backward 复用）
            auto W = compute_W(*m, *l);
            if (!W) return std::unexpected(W.error());
            // V 需 (BH*seq, d_k) 布局：V (BH*d_k, seq) 是 per-batch (d_k, seq)，
            // 按 batch 转置：transpose → (seq, BH*d_k) → rearrange_3d → (BH*seq, d_k)
            auto V_T_full = engine.transpose(V);
            if (!V_T_full) return std::unexpected(V_T_full.error());
            auto V_t = engine.rearrange_3d(*V_T_full, seq, BH, d_k_, false);
            if (!V_t) return std::unexpected(V_t.error());
            // O = W × V_t（普通 batched_matmul 原语）
            auto O_t = engine.batched_matmul(*W, *V_t, BH, false, false);
            if (!O_t) return std::unexpected(O_t.error());
            // O_t: (BH*seq, d_k) → 按 batch 转置回 (BH*d_k, seq) 供后续 rearrange：
            //   transpose → (d_k, BH*seq) → rearrange_3d(d_k, BH, seq) → (BH*d_k, seq)
            auto O_T_full = engine.transpose(*O_t);
            if (!O_T_full) return std::unexpected(O_T_full.error());
            auto co = engine.rearrange_3d(*O_T_full, d_k_, BH, seq, false);
            if (!co) return std::unexpected(co.error());
            concat_out = std::move(*co);
            if (!checkpoint_mode_)
            {
                Q_cache_ = std::move(Q);
                K_cache_ = std::move(K);
                V_cache_ = std::move(V);
                m_cache_ = std::move(*m);
                l_cache_ = std::move(*l);
                W_cache_ = std::move(*W);
            }
            two_pass_active_ = true;
        }
        else
        {
            // ── 旧路径：物化得分矩阵（ALiBi/doc_ids 等共享掩码不适用时回退） ──
            two_pass_active_ = false;
            // S = batched_matmul(Q, K, batch*H, transA=true) → (batch*H*seq, seq)
            // scale (1/sqrt(d_k)) 通过 alpha 系数折进 matmul 写出（cuBLAS sgemm 语义），
            // 省去一次全矩阵 scale pass + 额外 barrier
            auto scores = engine.batched_matmul(
                Q, K, BH, true, false, scale_);
            if (!scores) return std::unexpected(scores.error());
            // 施加掩码（钩子：MHA 默认 no-op，CSA 施加因果/ALiBi 掩码）
            auto masked = apply_mask_(engine, std::move(*scores), batch, seq);
            if (!masked) return std::unexpected(masked.error());
            // A = softmax(S_masked)
            auto attn = softmax_.forward(engine, *masked);
            if (!attn) return std::unexpected(attn.error());
            Tensor A = std::move(*attn);
            // O_re = batched_matmul(V, A, batch*H, transB=true) → (batch*H*d_k, seq)
            // 标准 attention: O[:,i] = sum_j V[:,j] * A[i,j] = (V × A^T)[:,i]
            auto co = engine.batched_matmul(V, A, BH, false, true);
            if (!co) return std::unexpected(co.error());
            concat_out = std::move(*co);
            if (!checkpoint_mode_)
            {
                // 注意力概率由 softmax_.output_cache() 单一持有（A 与其共享 buffer），
                // 不再另存 attn_cache_（避免旧路径重复存一份 (BH*seq, seq) 大矩阵）
                Q_cache_ = std::move(Q);
                K_cache_ = std::move(K);
                V_cache_ = std::move(V);
            }
        }

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(concat_out, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(concat_out);
        }

        // 9. 输出投影
        return w_o_.forward(engine, concat);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // 推断 batch/seq（与 forward 一致）
        const std::size_t total_seq = grad_output.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        // 1. 输出投影反向 → grad_concat: (H*d_k, batch*seq)
        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;

        // 2. rearrange grad_concat → (batch*H*d_k, seq)
        Tensor grad_concat_re;
        if (batch > 1)
        {
            auto gcr = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!gcr) return std::unexpected(gcr.error());
            grad_concat_re = std::move(*gcr);
        }
        else
        {
            grad_concat_re = std::move(*gc);
        }

        // 3-7. 注意力反向：两趟式（M6，重算 W 不物化 grad_S）vs 旧路径
        Tensor grad_Q_re, grad_K_re, grad_V_re;  // 均 (batch*H*d_k, seq)
        if (two_pass_active_)
        {
            // P = grad_A = batched_matmul(grad_concat^T, V, BH, true, false)
            // forward: O = V × A^T → grad_A = grad_O^T × V（两趟式反向的 P 输入）
            auto grad_A = engine.batched_matmul(
                grad_concat_re, V_cache_, BH, true, false);
            if (!grad_A) return std::unexpected(grad_A.error());
            // G = grad_concat_re^T 按 batch 转置 → (BH*seq, d_k)，
            // 供 grad_V[j][k] = Σ_i W·G[i][k]（同 V 的布局转换）
            auto G_T_full = engine.transpose(grad_concat_re);
            if (!G_T_full) return std::unexpected(G_T_full.error());
            auto G = engine.rearrange_3d(*G_T_full, seq, BH, d_k_, false);
            if (!G) return std::unexpected(G.error());
            // ── S7 IR 路径（M6 → R/X 表达式 + 普通 batched_matmul）──
            //   R  = row_sum(W·P)                     → (BH*seq, 1)
            //   X  = scale·W·(P − R)                  → (BH*seq, seq)（物化）
            //   grad_Q = K × X^T；grad_K = Q × X；grad_V = W^T × G
            auto R = dsl::compute_reduce(engine,
                dsl::row_reduce_sum(dsl::leaf(W_cache_) * dsl::leaf(*grad_A)),
                BH * seq, seq);
            if (!R) return std::unexpected(R.error());
            auto X = dsl::compute(engine,
                dsl::leaf(W_cache_)
                    * (dsl::leaf(*grad_A) - dsl::row_broadcast(*R)),
                BH * seq, seq);
            if (!X) return std::unexpected(X.error());
            // grad_Q = K × X^T（K_b (d_k,seq)，X_b (seq,seq) 按 X^T 使用）
            auto gq = engine.batched_matmul(K_cache_, *X, BH, false, true);
            if (!gq) return std::unexpected(gq.error());
            { auto gqs = engine.scale_inplace(*gq, scale_); if (!gqs) return std::unexpected(gqs.error()); }
            if (!gq) return std::unexpected(gq.error());
            // grad_K = Q × X
            auto gk = engine.batched_matmul(Q_cache_, *X, BH, false, false);
            if (!gk) return std::unexpected(gk.error());
            // grad_V = W^T × G（W_b (seq,seq) 按 W^T 使用，G_b (seq,d_k)）→ (BH*seq, d_k)
            auto gv_t = engine.batched_matmul(W_cache_, *G, BH, true, false);
            if (!gv_t) return std::unexpected(gv_t.error());
            // grad_V 转置回 (BH*d_k, seq)（与 forward 的 V_t→V 逆变换一致）
            auto gv_T = engine.transpose(*gv_t);
            if (!gv_T) return std::unexpected(gv_T.error());
            auto gv_re = engine.rearrange_3d(*gv_T, d_k_, BH, seq, false);
            if (!gv_re) return std::unexpected(gv_re.error());
            grad_Q_re = std::move(*gq);
            grad_K_re = std::move(*gk);
            grad_V_re = std::move(*gv_re);
        }
        else
        {
            // grad_V_re = batched_matmul(grad_concat, A, BH, false, false)
            // forward: O = V × A^T → grad_V = grad_O × A
            // A 由 softmax.output_cache() 单一持有（避免 attn_cache_ 重复存一份）
            auto gvr = engine.batched_matmul(
                grad_concat_re, softmax_.output_cache(), BH, false, false);
            if (!gvr) return std::unexpected(gvr.error());
            grad_V_re = std::move(*gvr);
            // grad_A = batched_matmul(grad_concat^T, V, BH, true, false)
            // forward: O = V × A^T → grad_A = grad_O^T × V
            auto grad_A = engine.batched_matmul(
                grad_concat_re, V_cache_, BH, true, false);
            if (!grad_A) return std::unexpected(grad_A.error());
            // grad_S = softmax.backward(grad_A) — 掩码/偏置为常数，梯度直接穿透
            auto grad_S = softmax_.backward(engine, *grad_A);
            if (!grad_S) return std::unexpected(grad_S.error());
            mask_backward_(engine, *grad_S, batch, seq);
            // grad_Q_re = batched_matmul(K, grad_S, BH, false, true) × scale
            // 前向 S = scale·Q^T·K → ∂L/∂Q = scale·K·grad_S^T，
            // scale 通过 alpha 折进 matmul 写出（省去两次全矩阵 scale pass）
            auto gq = engine.batched_matmul(
                K_cache_, *grad_S, BH, false, true, scale_);
            if (!gq) return std::unexpected(gq.error());
            grad_Q_re = std::move(*gq);
            // grad_K_re = batched_matmul(Q, grad_S, BH, false, false) × scale
            // ∂L/∂K = scale·Q·grad_S，同样折进 alpha
            auto gk = engine.batched_matmul(
                Q_cache_, *grad_S, BH, false, false, scale_);
            if (!gk) return std::unexpected(gk.error());
            grad_K_re = std::move(*gk);
        }

        // 7.5 RoPE backward：对 Q/K 梯度施加反角旋转
        //    （forward 的旋转矩阵正交，逆 = 转置 = 反角：grad*cos − rot(grad)*sin）
        if (use_rope_)
        {
            auto gq2 = rope_.apply(engine, grad_Q_re, seq, /*backward=*/true);
            if (!gq2) return std::unexpected(gq2.error());
            grad_Q_re = std::move(*gq2);
            auto gk2 = rope_.apply(engine, grad_K_re, seq, /*backward=*/true);
            if (!gk2) return std::unexpected(gk2.error());
            grad_K_re = std::move(*gk2);
        }

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            auto gq = engine.rearrange_3d(grad_Q_re, H_dk, batch, seq, true);
            if (!gq) return std::unexpected(gq.error());
            grad_Q = std::move(*gq);
            auto gk = engine.rearrange_3d(grad_K_re, H_dk, batch, seq, true);
            if (!gk) return std::unexpected(gk.error());
            grad_K = std::move(*gk);
            auto gv = engine.rearrange_3d(grad_V_re, H_dk, batch, seq, true);
            if (!gv) return std::unexpected(gv.error());
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(grad_Q_re);
            grad_K = std::move(grad_K_re);
            grad_V = std::move(grad_V_re);
        }

        // 9. 投影层反向 + 累加输入梯度
        auto giq = w_q_.backward(engine, grad_Q);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);

        auto gik = w_k_.backward(engine, grad_K);
        if (!gik) return gik;
        auto r1 = engine.add_inplace(grad_input, *gik);
        if (!r1) return std::unexpected(r1.error());

        auto giv = w_v_.backward(engine, grad_V);
        if (!giv) return giv;
        auto r2 = engine.add_inplace(grad_input, *giv);
        if (!r2) return std::unexpected(r2.error());

        return grad_input;
    }

    // ── 增量推理（KV cache）──────────────────────────────────────────
    // 处理单个新 token，复用历史 K/V 缓存，避免重复计算前文。
    //
    // KV cache 布局: (max_len, H*d_k)，row=position，col=head×dim
    //   - 投影得到 k_new/v_new: (H*d_k, 1)
    //   - transpose → (1, H*d_k) 后 insert_rows 到 cache 第 cur_len 行
    //   - slice_rows 取前 new_len 行参与 attention
    //
    // 无需因果掩码：新 token 天然只能看到自身及之前的位置（cache 中只有前文）。
    //
    // 输入: x_new (d_model, 1) 单个新 token 的嵌入
    // 输出: (d_model, 1) attention 输出
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        const Tensor& x_new,
        Tensor& k_cache,
        Tensor& v_cache,
        std::size_t cur_len)
    {
        if (x_new.rows() != d_model_ || x_new.cols() != 1)
            return std::unexpected(Error{"AttentionBase forward_step: x_new must be (d_model, 1)"});

        // 1. Q/K/V 投影 → (H*d_k, 1)
        auto q_res = w_q_.forward(engine, x_new);
        if (!q_res) return q_res;
        auto k_new = w_k_.forward(engine, x_new);
        if (!k_new) return k_new;
        auto v_new = w_v_.forward(engine, x_new);
        if (!v_new) return v_new;

        // 1.5 RoPE：对 Q/K 施加当前位置 (cur_len) 的旋转后写入 KV cache
        //    （cache 中的历史 K 已在各自 step 旋转过，相对位置自然成立）
        if (use_rope_)
        {
            auto qr = rope_.apply_step(engine, *q_res, cur_len, /*backward=*/false);
            if (!qr) return std::unexpected(qr.error());
            q_res = std::move(*qr);
            auto kr = rope_.apply_step(engine, *k_new, cur_len, /*backward=*/false);
            if (!kr) return std::unexpected(kr.error());
            k_new = std::move(*kr);
        }

        // 2. transpose → (1, H*d_k)，匹配 cache 的行布局
        auto k_new_T = engine.transpose(*k_new);
        if (!k_new_T) return std::unexpected(k_new_T.error());
        auto v_new_T = engine.transpose(*v_new);
        if (!v_new_T) return std::unexpected(v_new_T.error());

        // 3. 追加到 KV cache（就地写入第 cur_len 行）
        auto r1 = engine.insert_rows(k_cache, cur_len, *k_new_T);
        if (!r1) return std::unexpected(r1.error());
        auto r2 = engine.insert_rows(v_cache, cur_len, *v_new_T);
        if (!r2) return std::unexpected(r2.error());

        // 4. 取有效区间 [0, new_len) 并 transpose 为 (H*d_k, new_len) 布局
        //    batched_matmul 要求 rows 能被 batch 整除:
        //    (new_len, H*d_k) 的 rows=new_len 不保证整除 num_heads
        //    transpose 后 (H*d_k, new_len) 的 rows=H*d_k=d_model 必然整除
        const std::size_t new_len = cur_len + 1;
        auto k_valid = engine.slice_rows(k_cache, 0, new_len);
        if (!k_valid) return std::unexpected(k_valid.error());
        auto v_valid = engine.slice_rows(v_cache, 0, new_len);
        if (!v_valid) return std::unexpected(v_valid.error());
        auto K_T = engine.transpose(*k_valid);   // (H*d_k, new_len)
        if (!K_T) return std::unexpected(K_T.error());
        auto V_T = engine.transpose(*v_valid);   // (H*d_k, new_len)
        if (!V_T) return std::unexpected(V_T.error());

        // 5. scores = batched_matmul(Q, K_T, H, transA=T, transB=F)
        //    Q: (H*d_k, 1) — 每头 (d_k, 1)，转置后 (1, d_k)，M=1
        //    K_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transB=F，N=new_len
        //    每头: (1, d_k) × (d_k, new_len) = (1, new_len)
        //    堆叠: (H, new_len)
        //    scale (1/sqrt(d_k)) 通过 alpha 折进 matmul 写出
        auto scores = engine.batched_matmul(
            *q_res, *K_T, num_heads_, true, false, scale_);
        if (!scores) return std::unexpected(scores.error());

        // 6. 施加增量推理掩码钩子（默认 no-op；ALiBi 施加线性偏置）
        auto masked = apply_mask_step_(engine, std::move(*scores), cur_len);
        if (!masked) return std::unexpected(masked.error());

        // 7. softmax（行级归一化，每头独立）
        auto attn = softmax_.forward(engine, *masked);
        if (!attn) return std::unexpected(attn.error());

        // 8. attn_out = batched_matmul(V_T, attn, H, transA=F, transB=T)
        //    V_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transA=F，M=d_k
        //    attn: (H, new_len) — 每头 (1, new_len)，转置后 (new_len, 1)，N=1
        //    每头: (d_k, new_len) × (new_len, 1) = (d_k, 1)
        //    堆叠: (H*d_k, 1)
        auto attn_out = engine.batched_matmul(
            *V_T, *attn, num_heads_, false, true);
        if (!attn_out) return std::unexpected(attn_out.error());

        // 9. 输出投影 → (d_model, 1)
        return w_o_.forward(engine, *attn_out);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MultiHeadAttention — 无掩码多头注意力（极简子类：沿用 AttentionBase 默认行为）
// ══════════════════════════════════════════════════════════════════════════
class MultiHeadAttention final : public AttentionBase
{
public:
    MultiHeadAttention(std::size_t d_model, std::size_t num_heads,
                       std::size_t seq_len = 0)
        : AttentionBase(d_model, num_heads, seq_len) {}
};

// ══════════════════════════════════════════════════════════════════════════
// PositionalEncoding — 正弦波固定位置编码（支持 batch tiling）
//
// 算法（只在此处，不在 Engine/Shader）：
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//   forward: out = input + PE   (elementwise Add)
//   backward: 梯度直接穿透（编码不可学习）
//
// Tiling 模式（tile_size > 0）：
//   当输入为 (d_model, batch * tile_size) 时，生成 (d_model, tile_size) 的
//   基础编码，沿列方向 tile 为 (d_model, batch * tile_size)。
//   每个 tile_size 列块为同一份编码，对应一个样本。
// ══════════════════════════════════════════════════════════════════════════

// ── 扁平化注意力掩码构建（纯函数，可独立单测） ─────────────────────────
//
// 返回 (BH*seq, seq) 的掩码，其中 BH = batch*num_heads。
// 行 (b, h, t) 位于 (b*num_heads + h)*seq + t，列 j = key 位置（0..seq-1）。
//
// doc_ids（可选，batch-major：b*seq+t → 文档 id）非空时启用块对角文档感知：
//   - 跨文档位置禁止相互注意（值 = -inf）
//   - 同文档内仍施加因果掩码（未来位置 j>i 为 -inf）
// doc_ids 为空时退化为纯因果掩码（与现有行为一致）。
//
// use_alibi 时，在允许注意的位置叠加 ALiBi 线性偏置 -m_h*(i-j)。
[[nodiscard]] Matrix build_attention_mask(
    std::size_t batch, std::size_t seq_len, std::size_t num_heads,
    bool use_alibi, std::span<const Scalar> slopes,
    std::span<const std::size_t> doc_ids = {})
{
    NN_ASSERT(!use_alibi || slopes.size() >= num_heads,
              "build_attention_mask: slopes too small for num_heads");
    NN_ASSERT(doc_ids.empty() || doc_ids.size() >= batch * seq_len,
              "build_attention_mask: doc_ids too small for batch*seq_len");

    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    const std::size_t BH = batch * num_heads;
    Matrix mask(BH * seq_len, seq_len);
    for (std::size_t b = 0; b < batch; ++b)
    {
        for (std::size_t h = 0; h < num_heads; ++h)
        {
            const std::size_t head_idx = b * num_heads + h;
            const Scalar slope = use_alibi ? slopes[h] : Scalar{0};

            for (std::size_t i = 0; i < seq_len; ++i)
            {
                for (std::size_t j = 0; j < seq_len; ++j)
                {
                    bool allowed = (j <= i);
                    if (allowed && !doc_ids.empty())
                    {
                        // 块对角文档感知：仅同文档内的因果位置可注意
                        allowed = (doc_ids[b * seq_len + i]
                                   == doc_ids[b * seq_len + j]);
                    }
                    if (allowed)
                    {
                        const Scalar bias = use_alibi
                            ? -slope * static_cast<Scalar>(i - j)
                            : Scalar{0};
                        mask.set_value_unchecked(head_idx * seq_len + i, j, bias);
                    }
                    else
                    {
                        mask.set_value_unchecked(head_idx * seq_len + i, j, neg_inf);
                    }
                }
            }
        }
    }
    return mask;
}

// ══════════════════════════════════════════════════════════════════════════
// CausalSelfAttention — 因果自注意力（继承 AttentionBase，仅重写掩码钩子）
//
// 通过 PosEncodingType 参数支持两种掩码模式：
//   - Learned / Sinusoidal: 仅施加因果掩码（mask[i][j] = 0 if j<=i, -inf if j>i）
//   - ALiBi: 因果掩码 + ALiBi 线性偏置
//
// ALiBi (Attention with Linear Biases) 原理：
//   不使用位置嵌入，而是在注意力分数上添加线性偏置：
//   attention_score = Q*K^T + bias
//   其中 bias[i][j] = -m_h * (i - j) for j <= i
//   斜率 m_h = 2^(-8h/H)，h 是头索引，H 是总头数
//
// 优点（ALiBi 模式）：
//   1. 无需位置嵌入，减少参数
//   2. 天然支持长度外推（训练短序列，推理长序列）
//   3. 计算开销极小（仅添加预计算的偏置）
//
// 算法差异（相对于 AttentionBase）：
//   在 forward 的 scale 之后、softmax 之前，施加预计算的掩码：
//     S += mask (batch*H*seq, seq) — 因果掩码（按 batch*H 平铺，ALiBi 模式下含线性偏置）
//   backward 无需特殊处理（掩码为常数，softmax.backward 已处理梯度穿透）
//
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断。
//   seq_len=0 表示单样本模式（cols 即 seq），保持向后兼容。
// ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention final : public AttentionBase
{
private:
    bool use_alibi_;        // true = ALiBi 模式（因果掩码 + 线性偏置）

    // 掩码缓存（平铺为 batch*H*seq × seq 以匹配堆叠 scores 布局）
    // ALiBi 模式下包含因果掩码 + 线性偏置；普通模式下仅因果掩码
    Tensor mask_cache_;
    std::size_t mask_cached_batch_ = 0;  // 缓存键：batch（独立字段，避免位打包溢出）
    std::size_t mask_cached_seq_ = 0;    // 缓存键：seq_len

    // 两趟式（M6→S7）组合偏置小张量缓存（AttnBias 描述子指向它们；
    // S7 IR 掩码表达式经 mask_slopes_/mask_doc_col_/mask_doc_ids_ 读取）
    Tensor slopes_cache_;   // (1, num_heads) ALiBi 按头斜率（惰性构建）
    Tensor doc_ids_cache_;  // (1, batch*seq) 每位置文档 id（每步重建）
    Tensor doc_col_;        // (BH*seq, 1) 每行文档 id（每步重建，S7 掩码用）

    // ALiBi 斜率：m_h = 2^(-8h/H)（仅 use_alibi_ = true 时使用）
    std::vector<Scalar> slopes_;

    // 文档感知掩码：当前 step 每样本的文档 id（batch-major b*seq+t → doc id）
    // 非空时启用块对角文档感知（跨文档禁止注意），每步重建（不缓存）。
    std::vector<std::size_t> doc_ids_;
    bool has_doc_ids_ = false;

    void init_slopes_()
    {
        slopes_.resize(num_heads_);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            // m_h = 2^(-8h/H)
            slopes_[h] = std::pow(Scalar{2}, -Scalar{8} * h / num_heads_);
        }
    }

    [[nodiscard]] Result<void> ensure_mask_(ComputeEngine& engine, std::size_t batch, std::size_t seq_len)
    {
        // 用 (batch, seq_len) 作为缓存键
        if (mask_cached_batch_ == batch && mask_cached_seq_ == seq_len)
            return {};

        // 纯因果掩码：委托 build_attention_mask（doc_ids 为空）
        auto mask = build_attention_mask(batch, seq_len, num_heads_,
                                         use_alibi_, slopes_);
        auto r = engine.from_matrix(mask);
        if (!r) return std::unexpected(r.error());
        mask_cache_ = std::move(*r);
        mask_cached_batch_ = batch;
        mask_cached_seq_ = seq_len;
        return {};
    }

public:
    // 设置当前 step 的每样本文档 id（batch-major b*seq+t → doc id）。
    // 传入空 span 会清除文档感知，退化为纯因果掩码。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty())
        {
            doc_ids_.clear();
            has_doc_ids_ = false;
            return;
        }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

protected:
    // 重写掩码钩子：在 scale 之后、softmax 之前施加因果/ALiBi 掩码
    [[nodiscard]] Result<Tensor> apply_mask_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t batch, std::size_t seq) override
    {
        if (has_doc_ids_)
        {
            // 文档感知：doc_ids_ 每 batch 变化，掩码不可缓存，每步重建
            // 块对角（跨文档 -inf）∧ 因果（同文档内未来 -inf）
            auto m = build_attention_mask(batch, seq, num_heads_,
                                          use_alibi_, slopes_, doc_ids_);
            auto mt = engine.from_matrix(m);
            if (!mt) return std::unexpected(mt.error());
            return dsl::compute(engine,
                dsl::leaf(scores) + dsl::leaf(*mt),
                scores.rows(), scores.cols());
        }
        {
            auto r = ensure_mask_(engine, batch, seq);
            if (!r) return std::unexpected(r.error());
        }
        return dsl::compute(engine,
            dsl::leaf(scores) + dsl::leaf(mask_cache_),
            scores.rows(), scores.cols());
    }

    // 重写两趟式决策：组合式 AttnBias 描述子统一 因果/ALiBi/doc_ids（及其组合），
    // 全部走两趟式（不物化 (BH·seq, seq) 得分矩阵），不再回退旧路径。
    [[nodiscard]] Result<TwoPassMask> two_pass_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq) override
    {
        // 掩码分量由 mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子读取（IR 表达式）
        if (use_alibi_)
        {
            if (!slopes_cache_.valid())
            {
                Matrix s(1, num_heads_);
                for (std::size_t h = 0; h < num_heads_; ++h)
                    s.set_value_unchecked(0, h, slopes_[h]);
                auto t = engine.from_matrix(s);
                if (!t) return std::unexpected(t.error());
                slopes_cache_ = std::move(*t);
            }
        }
        if (has_doc_ids_)
        {
            // 文档感知：doc_ids_ 每 batch 变化，每步重建（小张量 O(BH*seq)）。
            // S7：doc_ids_cache_ 为 (1, BH*seq) 布局——每 (b,h) 块重复
            // doc_ids[b*seq..]（BatchCol 视图的 batch 下标 = BH 网格下标；
            // 旧 M5 AttnBias 的 (1, batch*seq) 布局 S7 起不再适用）
            Matrix d(1, batch * num_heads_ * seq);
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t h = 0; h < num_heads_; ++h)
                    for (std::size_t i = 0; i < seq; ++i)
                        d.set_value_unchecked(0, (b * num_heads_ + h) * seq + i,
                                              static_cast<Scalar>(doc_ids_[b * seq + i]));
            auto t = engine.from_matrix(d);
            if (!t) return std::unexpected(t.error());
            doc_ids_cache_ = std::move(*t);

            // S7：doc_col_（(BH*seq,1)）每行文档 id：doc_col[b*H*seq+h*seq+i]
            // = doc_ids[b*seq+i]（跨 head 重复；IR 掩码按行广播读取）
            {
                const std::size_t BH = batch * num_heads_;
                Matrix dc(BH * seq, 1);
                for (std::size_t b = 0; b < batch; ++b)
                    for (std::size_t h = 0; h < num_heads_; ++h)
                        for (std::size_t i = 0; i < seq; ++i)
                            dc.set_value_unchecked(
                                (b * num_heads_ + h) * seq + i, 0,
                                static_cast<Scalar>(doc_ids_[b * seq + i]));
                auto tc = engine.from_matrix(dc);
                if (!tc) return std::unexpected(tc.error());
                doc_col_ = std::move(*tc);
            }

        }
        return TwoPassMask{/*use_two_pass=*/true};
    }

    // ── S7 掩码输入张量钩子（IR 掩码表达式读取）──────────────────────
    [[nodiscard]] const Tensor* mask_slopes_() const override
    { return use_alibi_ ? &slopes_cache_ : nullptr; }
    [[nodiscard]] const Tensor* mask_doc_col_() const override
    { return has_doc_ids_ ? &doc_col_ : nullptr; }
    [[nodiscard]] const Tensor* mask_doc_ids_() const override
    { return has_doc_ids_ ? &doc_ids_cache_ : nullptr; }

    // 重写增量推理掩码钩子：ALiBi 模式下施加线性距离偏置。
    // 普通因果模式（use_alibi_ = false）：no-op，因 cache 只含前文，因果天然满足。
    // ALiBi 模式：scores[h, j] += -slope[h] * (cur_len - j)，j ∈ [0, cur_len]。
    //   scores 布局: (H, new_len)，new_len = cur_len + 1
    //   query 在位置 cur_len，key 在位置 j，距离 = cur_len - j
    [[nodiscard]] Result<Tensor> apply_mask_step_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t cur_len) override
    {
        if (!use_alibi_) return std::move(scores);

        const std::size_t new_len = cur_len + 1;
        // 构建 ALiBi 偏置矩阵 (H, new_len)
        Matrix bias(num_heads_, new_len);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            const Scalar slope = slopes_[h];
            for (std::size_t j = 0; j < new_len; ++j)
            {
                // j ∈ [0, cur_len]，距离 cur_len - j ∈ [0, cur_len]
                const std::size_t dist = cur_len - j;
                bias.set_value_unchecked(h, j,
                    -slope * static_cast<Scalar>(dist));
            }
        }
        auto bias_t = engine.from_matrix(bias);
        if (!bias_t) return std::unexpected(bias_t.error());
        return dsl::compute(engine,
            dsl::leaf(scores) + dsl::leaf(*bias_t),
            scores.rows(), scores.cols());
    }

public:
    CausalSelfAttention(std::size_t d_model, std::size_t num_heads,
                       std::size_t /*max_len*/ = 1024,
                       std::size_t seq_len = 0,
                       PosEncodingType pos_enc = PosEncodingType::Learned)
        : AttentionBase(d_model, num_heads, seq_len, pos_enc),
          use_alibi_(pos_enc == PosEncodingType::ALiBi)
    {
        if (use_alibi_)
            init_slopes_();
    }
};

} // namespace nn

