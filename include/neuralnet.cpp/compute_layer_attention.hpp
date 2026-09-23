#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_softmax.hpp"
#include "expr_fold.hpp"

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
//   - MHA/CSA 的差异只在 fold 掩码变体（fold_mask_variant_：MHA=Plain 双向
//     无掩码 / CSA=causal[+alibi/doc]），掩码恒在 fold body 内表达、绝不物化
//
// 算法（只在此处，不在 Engine/Shader）：
//   Q = W_q × x, K = W_k × x, V = W_v × x  (三个 Linear 投影)
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//
//   批量化关键：用 rearrange_3d 把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
//   单次处理所有样本和所有头。
//
//   forward（P-C2-7 单 fold 路径，S 不物化）：
//   Q/K = rearrange_3d → (batch*H*d_k, seq)；[RoPE]；Q *= scale（S7 折进 Q）
//   V_t = transpose + rearrange → (BH*seq, d_k)
//   O_t = eval_expr(make_fold_attn_o(...), {Q, K, V_t, [掩码输入]}, BH*seq, d_k)
//     —— QKᵀ/掩码/online softmax/ΣwV 全在单个 fold kernel 内逐块完成
//   O = transpose + rearrange 回 (H*d_k, batch*seq)；out = W_o × O
//   backward：W 由 recompute_W_ 按同掩码树重算，R/X 表达式 + batched_matmul
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

    // forward 缓存（rearranged 版本，供 backward 直接使用；得分矩阵类缓存
    // 随单 fold 路径全部取消——W/m/l/attn 均不物化，W 在 backward 重算）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged

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

    // ── fold 掩码变体选择（P-C2-7 正确性修复）──────────────────────────
    //   类级语义，fail-safe 默认 = 无掩码：MHA（基类默认）双向不掩蔽——
    //   与迁移前 apply_mask_ 默认 no-op 同义；CSA override 返回四分支
    //   掩码树（causal 恒有 + alibi/doc 按钩子）。forward fold 与
    //   recompute_W_ 的掩码树都经此选择（两者同构同序）。
    [[nodiscard]] virtual nn::expr::FoldAttnMask fold_mask_variant_() const
    {
        return nn::expr::FoldAttnMask::Plain;
    }
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

    // ── P0-5：从 Q/K 重算 W（P-C2-7：m/l 缓存已随 forward fold 迁移删除，
    //   softmax 单表达式在 kernel 内部归一化，m/l 不再外溢）。两步均走既有
    //   快路径：1) S = masked(Q·Kᵀ)（generate_glsl_matmul 分块；掩码树与
    //   forward fold 的变体同构）2) W = softmax(S)（M3 归约快路径）。
    // 返回 (BH*seq, seq) 的 softmax 归一化权重；attention FLOPs ×1.5–2
    // （QK^T 在 backward 重算），训练可接受。
    [[nodiscard]] Result<Tensor> recompute_W_(
        ComputeEngine& engine,
        const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq) const
    {
        const bool use_slopes = use_alibi_mask_();
        const bool use_doc = use_doc_mask_();
        // 1) S = masked(Q·Kᵀ)（四分支掩码树与 forward fold 的变体同构；
        //    generate_glsl_matmul 分块快路径，S 瞬时物化一次）
        Result<Tensor> sres = [&]() -> Result<Tensor> {
            // MHA（Plain）：S 不施加任何掩码——与 forward fold 的 Plain
            //   变体同构；裸 matmul 表达式由 scan_exprs 的 MHA dry-run
            //   注册（闭合世界：该结构此前从未被 dry-run 覆盖）
            if (fold_mask_variant_() == nn::expr::FoldAttnMask::Plain)
                return dsl::compute(engine,
                    dsl::matmul(Q, K, true, false, BH), BH * seq, seq);
            if (use_slopes && use_doc)
                return dsl::compute(engine,
                    masked_alibi_doc_(dsl::matmul(Q, K, true, false, BH), seq),
                    BH * seq, seq);
            if (use_slopes)
                return dsl::compute(engine,
                    masked_alibi_(dsl::matmul(Q, K, true, false, BH), seq),
                    BH * seq, seq);
            if (use_doc)
                return dsl::compute(engine,
                    masked_doc_(dsl::matmul(Q, K, true, false, BH), seq),
                    BH * seq, seq);
            return dsl::compute(engine,
                masked_causal_(dsl::matmul(Q, K, true, false, BH), seq),
                BH * seq, seq);
        }();
        if (!sres) return std::unexpected(sres.error());
        Tensor S = std::move(*sres);
        // 2) W = softmax(S)（M3 单表达式归约快路径——与 Softmax::forward
        //    同构：ReduceRef 直接参与算术按行广播；m/l 在 kernel 内部归一化，
        //    不再作为输入/缓存存在）
        return dsl::compute(engine,
            dsl::exp(dsl::leaf(S) - dsl::row_reduce_max(S))
            / dsl::row_reduce_sum(
                dsl::exp(dsl::leaf(S) - dsl::row_reduce_max(S))),
            BH * seq, seq);
    }

    // ── 掩码输入准备钩子（P-C2-7 前身 = two_pass_mask_ 决策钩子）─────────
    // forward 恒走单 fold 路径（旧物化/两趟分支已删，不保留旧路径）；本钩子
    // 仅做"掩码输入张量准备"（CSA 覆写构建 slopes/doc_col/doc_ids 缓存），
    // 掩码分量经 mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子供 fold/recompute
    // 表达式读取。默认 no-op（MHA/AttentionBase 无掩码分量）。
    [[nodiscard]] virtual Result<void> prepare_mask_inputs_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        (void)engine; (void)batch; (void)seq;
        return {};
    }

    // （P-C2-7：apply_mask_ 物化路径钩子已删除——掩码恒在 fold body /
    //   recompute_W_ 掩码树内表达，绝不物化 (BH·seq, seq) 掩码矩阵；
    //   增量推理的 apply_mask_step_ 独立保留，见下。）

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
        // 掩码/偏置描述子（指向子类 doc_ids/slopes 缓存）小而常驻，
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
        // 注：保留就地原语——DSL 表达式会多分配一整块 (BH·d_k, seq) 缓冲，
        // 实测该点比 scale_inplace 慢（见 build/perfprobe 的原地/表达式对照）。
        { auto qs = engine.scale_inplace(Q, scale_); if (!qs) return std::unexpected(qs.error()); }

        // ── 注意力主体：单 fold 路径（P-C2-7，不保留旧路径）────────────
        const std::size_t BH = batch * num_heads_;
        //   掩码输入张量准备（CSA 覆写钩子构建 slopes/doc_col/doc_ids）
        {
            auto pm = prepare_mask_inputs_(engine, batch, seq);
            if (!pm) return std::unexpected(pm.error());
        }
        // 掩码变体（类级语义：MHA=Plain 双向无掩码 / CSA=四分支树，见
        //   fold_mask_variant_；与 recompute_W_ 的掩码树同构同序）
        const auto fmask = fold_mask_variant_();
        Tensor concat_out;  // (batch*H*d_k, seq)——fold 输出 O_t 经转置/重排得到
        // V 需 (BH*seq, d_k) 布局：V_t 构建（原位于 W 之后，fold 需前置）
        auto V_T_full = engine.transpose(V);
        if (!V_T_full) return std::unexpected(V_T_full.error());
        auto V_t = engine.rearrange_3d(*V_T_full, seq, BH, d_k_, false);
        if (!V_t) return std::unexpected(V_t.error());
        // ── 单 fold 表达式：S 不物化、online 单遍 ──────────────────────
        //   6 趟 (BH·seq, seq) 物化流量归零；掩码在 fold body 内逐块生效，
        //   causal_skip 把被屏蔽块钳成空转（被跳过的恰是 -inf/0 恒等项 →
        //   与全量计算逐位一致，见 FoldSpec::causal_skip 注释）。
        //   inputs 顺序 = make_fold_attn_o 的 views 顺序：Q,K,V_t,[slope],[dc],[ids]
        std::vector<Tensor> fold_in{Q, K, *V_t};
        if (const Tensor* sl = mask_slopes_())  fold_in.push_back(*sl);
        if (const Tensor* dc = mask_doc_col_()) fold_in.push_back(*dc);
        if (const Tensor* di = mask_doc_ids_()) fold_in.push_back(*di);
        const nn::ExprSpec fold_spec =
            nn::expr::make_fold_attn_o(seq, d_k_, BH, fmask);
        if (auto fv = nn::validate_expr_spec(fold_spec, fold_in.size()); !fv)
            return std::unexpected(fv.error());
        // scale 已折进 Q（scale_inplace，S7 教训）——fold 的 mm 段直接消费
        auto O_t_r = engine.eval_expr(fold_spec, fold_in, BH * seq, d_k_);
        if (!O_t_r) return std::unexpected(O_t_r.error());
        Tensor O_t = std::move(*O_t_r);
        // O_t: (BH*seq, d_k) → 按 batch 转置回 (BH*d_k, seq) 供后续 rearrange：
        //   transpose → (d_k, BH*seq) → rearrange_3d(d_k, BH, seq) → (BH*d_k, seq)
        auto O_T_full = engine.transpose(O_t);
        if (!O_T_full) return std::unexpected(O_T_full.error());
        auto co = engine.rearrange_3d(*O_T_full, d_k_, BH, seq, false);
        if (!co) return std::unexpected(co.error());
        concat_out = std::move(*co);
        if (!checkpoint_mode_)
        {
            Q_cache_ = std::move(Q);
            K_cache_ = std::move(K);
            V_cache_ = std::move(V);
            // P-C2-7：m/l 不再缓存（recompute_W_ 内部 softmax 归一化，见 P0-5 区）
        }
        // 掩码恒在 fold body 内生效（fold_mask_variant_ 选变体），绝不物化

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

        // 3-7. 注意力反向：S7 R/X 路径（P-C2-7：旧 softmax.backward 分支不保留）
        Tensor grad_Q_re, grad_K_re, grad_V_re;  // 均 (batch*H*d_k, seq)
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
            //   从 Q/K 重算 W（P0-5：不缓存 W；P-C2-7 起 m/l 也不缓存——
            //   recompute_W_ 内部 softmax 单表达式归一化）；
            //   FLOPs ×1.5-2（QK^T 重算），训练可接受。
            auto W_re = recompute_W_(engine, Q_cache_, K_cache_, BH, seq);
            if (!W_re) return std::unexpected(W_re.error());
            //   R  = row_sum(W·P)                     → (BH*seq, 1)
            //   X  = scale·W·(P − R)                  → (BH*seq, seq)（物化）
            //   grad_Q = K × X^T；grad_K = Q × X；grad_V = W^T × G
            auto R = dsl::compute_reduce(engine,
                dsl::row_reduce_sum(dsl::leaf(*W_re) * dsl::leaf(*grad_A)),
                BH * seq, seq);
            if (!R) return std::unexpected(R.error());
            auto X = dsl::compute(engine,
                dsl::leaf(*W_re)
                    * (dsl::leaf(*grad_A) - dsl::row_broadcast(*R)),
                BH * seq, seq);
            if (!X) return std::unexpected(X.error());
            // grad_Q = K × X^T（K_b (d_k,seq)，X_b (seq,seq) 按 X^T 使用）
            auto gq = engine.batched_matmul(K_cache_, *X, BH, false, true);
            if (!gq) return std::unexpected(gq.error());
            // grad_Q 补乘 scale（forward 把 scale 折进了 Q）：用就地原语（同 forward
            // 的说明——DSL 表达式要多分配一整块缓冲，实测更慢）
            { auto gqs = engine.scale_inplace(*gq, scale_); if (!gqs) return std::unexpected(gqs.error()); }
            // grad_K = Q × X
            auto gk = engine.batched_matmul(Q_cache_, *X, BH, false, false);
            if (!gk) return std::unexpected(gk.error());
            // grad_V = W^T × G（W_b (seq,seq) 按 W^T 使用，G_b (seq,d_k)）→ (BH*seq, d_k)
            auto gv_t = engine.batched_matmul(*W_re, *G, BH, true, false);
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
        auto gik = w_k_.backward(engine, grad_K);
        if (!gik) return gik;
        auto giv = w_v_.backward(engine, grad_V);
        if (!giv) return giv;

        // grad_input = grad_Q + grad_K + grad_V：三路累加**原地**融合为单趟
        // （目标传递，不额外分配；取代 clone + 两次 add_inplace）。加法结合
        // 顺序与原实现一致（(giq + gik) + giv）→ 逐字节等价。
        auto acc = dsl::compute_into(engine,
            dsl::leaf(*giq) + dsl::leaf(*gik) + dsl::leaf(*giv), *giq);
        if (!acc) return std::unexpected(acc.error());
        return giq;
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

    // 掩码输入张量缓存（fold body / recompute_W_ 掩码树经
    // mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子读取）
    // ALiBi 斜率表 (1, batch*num_heads)：按 (b,h) 块重复 slopes_[h]——fold 的
    //   batch_mod(BH) 按网格下标 b*H+h 直读（旧 (1,H) 表 batch≥2 时 b≥1 越界
    //   读、ALiBi 静默错；batch=1 恰好掩蔽）。旧 recompute 的 batch_mod(%H)
    //   与增量推理按 h∈[0,H) 读均落首块 → 语义不变。batch 变化时重建。
    Tensor slopes_cache_;
    std::size_t slopes_cached_batch_ = 0;  // slopes_cache_ 形状键（batch 变更重建）
    Tensor doc_ids_cache_;  // (1, batch*H*seq) 每位置文档 id 按 (b,h) 块重复（每步重建）
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
    // 掩码输入张量准备（原 two_pass_mask_ 的构建体原样保留、职责纯化）：
    //   slopes_cache_ / doc_ids_cache_ / doc_col_ 由本钩子构建，
    //   掩码分量经 mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子读取。
    [[nodiscard]] Result<void> prepare_mask_inputs_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq) override
    {
        // 掩码分量由 mask_slopes_/mask_doc_col_/mask_doc_ids_ 钩子读取（IR 表达式）
        if (use_alibi_)
        {
            // fold 契约：batch_mod(BH) 直读 b*H+h → 表长必须 = batch*H。
            //   旧 (1,num_heads) 表在 batch≥2 越界读/ALiBi 静默丢（batch=1
            //   掩蔽，历史单样本测试未暴露）；按 (b,h) 块重复 slopes_[h]。
            if (!slopes_cache_.valid() || slopes_cached_batch_ != batch)
            {
                Matrix s(1, batch * num_heads_);
                for (std::size_t b = 0; b < batch; ++b)
                    for (std::size_t h = 0; h < num_heads_; ++h)
                        s.set_value_unchecked(0, b * num_heads_ + h, slopes_[h]);
                auto t = engine.from_matrix(s);
                if (!t) return std::unexpected(t.error());
                slopes_cache_ = std::move(*t);
                slopes_cached_batch_ = batch;
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
        return {};
    }

    // ── S7 掩码输入张量钩子（IR 掩码表达式读取）──────────────────────
    [[nodiscard]] const Tensor* mask_slopes_() const override
    { return use_alibi_ ? &slopes_cache_ : nullptr; }
    [[nodiscard]] const Tensor* mask_doc_col_() const override
    { return has_doc_ids_ ? &doc_col_ : nullptr; }
    [[nodiscard]] const Tensor* mask_doc_ids_() const override
    { return has_doc_ids_ ? &doc_ids_cache_ : nullptr; }

    // fold 掩码变体 override：CSA = 因果恒有 + alibi/doc 按钩子（四分支
    //   与旧 apply_mask_ override 语义同构）
    [[nodiscard]] nn::expr::FoldAttnMask fold_mask_variant_() const override
    {
        return use_doc_mask_()
            ? (use_alibi_mask_() ? nn::expr::FoldAttnMask::AlibiDoc
                                : nn::expr::FoldAttnMask::Doc)
            : (use_alibi_mask_() ? nn::expr::FoldAttnMask::Alibi
                                : nn::expr::FoldAttnMask::Causal);
    }

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

