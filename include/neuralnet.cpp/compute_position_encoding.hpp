#pragma once

// ───────────────────────────────────────────────────────────────────────────
//  compute_position_encoding.hpp — 位置编码策略族（L2 辅助对象，**非 Layer**）
//
//  位置编码 = 把"位置"信息注入表征的策略。不同策略的**注入点**不同，
//  但统一为一个多态基类 `PositionEncoder`，而不是散落的 PosEncodingType 判断：
//
//    ┌────────────────────┬──────────────────────────────────────────────┐
//    │ Learned            │ 嵌入侧：x += pos_emb[pos]（gather + 加性融合） │
//    │ Sinusoidal         │ 嵌入侧：x += 固定正弦表[pos]                  │
//    │ RoPE               │ 注意力侧：对 Q/K 施加旋转（正交变换）          │
//    │ ALiBi              │ 注意力侧：对注意力分数加线性距离偏置           │
//    │ NoPositionEncoder  │ 恒等（无位置编码）                            │
//    └────────────────────┴──────────────────────────────────────────────┘
//
//  所有权按**注入点**划分（谁拥有 = 谁负责）：
//    · 模型侧（GPTModel/ZiPTModel/RAPTModel）持有嵌入侧编码器，构造期
//      `make_embedding_position_encoder(type, d_model, seq_len)`；forward 开头
//      `apply(...)` / 增量 `apply_step(...)` 注入（RoPE/ALiBi 映射为恒等）。
//    · 注意力层（CausalSelfAttention / ReLULinearAttention）**自持**注意力侧
//      编码器，构造期 `make_attention_position_encoder(type, d_k, num_heads)`；
//      在 `apply_qk()`（RoPE 旋转）与 `bias_slopes()`（ALiBi 斜率表）上取数
//      （Learned/Sinusoidal 映射为恒等）。
//  于是 PosEncodingType 的分发只剩这两个工厂，forward/backward 热路径内
//  **不存在任何按编码类型的分支**。
//
//  AOT 闭合世界：本文件内的内联表达式（RoPE 旋转、加性融合、ALiBi 偏置）
//  由 `tools/scan_exprs.cpp` 的 dry-run 登记（文本只写在这里，不在 Engine）。
// ───────────────────────────────────────────────────────────────────────────

#include "compute_layer_base.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  RotaryEmbedding — 旋转位置编码的 cos/sin 短表（(d_k, seq)，LLaMA half-swap）
//
//  纯数据对象（表构建 + 内联 RoPE 表达式），不作策略分发；由
//  `RopePositionEncoder` 持有。独立保留是因为扫描/测试直接对拍该表。
// ═══════════════════════════════════════════════════════════════════════════
class RotaryEmbedding
{
private:
    std::size_t d_k_ = 0;
    std::size_t seq_cached_ = 0;   // 缓存表的列数（全表应用）
    std::size_t pos_offset_ = 0;   // 绝对位置偏移（滑动窗生成时用，0=从 0 起）
    Tensor cos_cache_;             // (d_k, seq_cached_)
    Tensor sin_cache_;             // (d_k, seq_cached_)
    PrecisionProfile p_;           // 多精度（§9.2）：由持有它的位置编码器下传

    // 把单个位置 pos 的 cos/sin 写入指定列（rebuild/apply_step 共用）。
    // LLaMA 式：cos 沿 d 维 = cat(freqs, freqs)（前后半相同），
    // 配合 rotate_half（前后半交换+前半取负）构成 2×2 旋转块。
    // c/s 为行主序标量缓冲（cols 列），宿主桥（17 §3 D11）——不经 Matrix。
    void fill_pos_column_(std::span<Scalar> c, std::span<Scalar> s, std::size_t cols,
                          std::size_t pos, std::size_t col) const
    {
        const std::size_t half = d_k_ / 2;
        const Scalar pd = static_cast<Scalar>(pos);
        for (std::size_t j = 0; j < half; ++j)
        {
            const Scalar theta = pd / std::pow(Scalar{10000},
                static_cast<Scalar>(2 * j) / static_cast<Scalar>(d_k_));
            const Scalar cv = std::cos(theta);
            const Scalar sv = std::sin(theta);
            c[j        * cols + col] = cv;
            c[(half + j) * cols + col] = cv;
            s[j        * cols + col] = sv;
            s[(half + j) * cols + col] = sv;
        }
    }

    // 重建全表 (d_k, seq)：cos/sin 按维度对交错重复；位置 = pos + pos_offset_
    // （绝对位置偏移：滑动窗生成时，输入被截断到窗口，但位置应从真实起点算起）
    [[nodiscard]] Result<void> rebuild(ComputeEngine& engine, std::size_t seq)
    {
        std::vector<Scalar> c(d_k_ * seq), s(d_k_ * seq);
        for (std::size_t pos = 0; pos < seq; ++pos)
            fill_pos_column_(c, s, seq, pos + pos_offset_, pos);
        NN_TRY(cr, detail::upload_span(engine, d_k_, seq, Precision::F32, std::span(c)));
        cos_cache_ = std::move(*cr);
        NN_TRY(sr, detail::upload_span(engine, d_k_, seq, Precision::F32, std::span(s)));
        sin_cache_ = std::move(*sr);
        seq_cached_ = seq;
        return {};
    }

    // 构造 RoPE 内联表达式（forward 用 Add 结尾，backward 用 Sub 结尾）
    // 表达式文本只写在本类（apply/apply_step），AOT 收集（scan_exprs）
    // 用同一段代码 dry-run 折叠出结构并合成融合 shader——绝不漂移。

public:
    RotaryEmbedding() = default;
    explicit RotaryEmbedding(std::size_t d_k)
        : d_k_(d_k)
    {
    }

    [[nodiscard]] Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    // 多精度（§9.2）：RoPE 是辅助对象（非 Layer），由持有它的位置编码器下传
    // profile（否则 RoPE 的 DSL 求值退回 F32，f16 配置下 Q/K 存储不减半）
    void set_precision_profile(const PrecisionProfile& p) { p_ = p; }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

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
            NN_TRY(r, rebuild(engine, seq));
        }
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        // 内联 RoPE 表达式（LLaMA half-swap；backward 为旋转正交，逆 = 反角）：
        //   forward:  out = q·cos + rotate_half(q)·sin
        //   backward: out = q·cos − rotate_half(q)·sin
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
                q.rows(), q.cols(), p_.compute);
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
            q.rows(), q.cols(), p_.compute);
    }

    // 增量推理：q 为 (H*d_k, 1)，位置 = pos（cur_len）
    // 直接生成 (d_k,1) 位置表，避免对全表切片取列。
    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& q, std::size_t pos, bool backward)
    {
        if (d_k_ == 0 || d_k_ % 2 != 0)
            return std::unexpected(Error{"RotaryEmbedding::apply_step: d_k must be positive and even"});
        std::vector<Scalar> c(d_k_), s(d_k_);       // 宿主桥（17 §3 D11）
        fill_pos_column_(c, s, 1, pos, 0);
        NN_TRY(cr, detail::upload_span(engine, d_k_, 1, Precision::F32, std::span(c)));
        NN_TRY(sr, detail::upload_span(engine, d_k_, 1, Precision::F32, std::span(s)));
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(*cr, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
                q.rows(), q.cols(), p_.compute);
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(*cr, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
            q.rows(), q.cols(), p_.compute);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  PositionEncoder — 位置编码策略基类（统一接口）
//
//  接口按**注入点**分三组，各具体策略只覆写自己那组，其余走基类 no-op：
//    · 嵌入侧   apply / apply_step / backward / parameters / param_gradients
//    · Q/K 侧   apply_qk / apply_qk_step / set_position_offset
//    · 分数侧   has_score_bias / prepare_bias / bias_slopes / apply_bias_step
//  所有方法在**构造期或每调用一次**的粒度上使用（不在块内循环），具体策略由
//  工厂在构造期选定 → 热路径零类型判断。
// ═══════════════════════════════════════════════════════════════════════════
class PositionEncoder
{
protected:
    // 多精度（§9.2）：由持有它的模型下传（否则其内部 DSL 求值退回 F32，
    // f16 配置下静默丢失存储收益）。本类是辅助对象（非 Layer），故单独 setter。
    PrecisionProfile p_;

    PositionEncoder() = default;

public:
    virtual ~PositionEncoder() = default;

    PositionEncoder(const PositionEncoder&) = delete;
    PositionEncoder& operator=(const PositionEncoder&) = delete;

    virtual void set_precision_profile(const PrecisionProfile& p) { p_ = p; }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

    // 引擎相关初始化（创建张量等）
    [[nodiscard]] virtual Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    // ── 嵌入侧注入（Learned / Sinusoidal 覆写；其余 no-op）──────────────
    // apply：token_emb_T 为 (d_model, batch*seq)，返回 x = token_emb_T (+ pos_emb)
    [[nodiscard]] virtual Result<Tensor> apply(
        ComputeEngine& /*engine*/, const Tensor& token_emb_T,
        std::size_t /*batch*/, std::size_t /*seq*/)
    { return token_emb_T; }

    // 增量前向：x 为 (d_model, 1)，返回 x + pos_emb[pos]
    [[nodiscard]] virtual Result<Tensor> apply_step(
        ComputeEngine& /*engine*/, const Tensor& x, std::size_t /*pos*/)
    { return x; }

    // 反向：累计位置梯度到 grad_pos_emb_（默认 no-op）
    [[nodiscard]] virtual Result<void> backward(
        ComputeEngine& /*engine*/, const Tensor& /*grad_T*/,
        std::size_t /*batch*/, std::size_t /*seq*/)
    { return {}; }

    [[nodiscard]] virtual std::vector<TensorRef> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<TensorRef> param_gradients() { return {}; }

    // ── 注意力 Q/K 侧（RoPE 覆写；其余 no-op）─────────────────────────
    // 全表：qk 为 (batch*H*d_k, seq)，列 = position；**原地替换**
    [[nodiscard]] virtual Result<void> apply_qk(
        ComputeEngine& /*engine*/, Tensor& /*qk*/, std::size_t /*seq*/, bool /*backward*/)
    { return {}; }

    // 增量：qk 为 (H*d_k, 1)，位置 = pos
    [[nodiscard]] virtual Result<void> apply_qk_step(
        ComputeEngine& /*engine*/, Tensor& /*qk*/, std::size_t /*pos*/, bool /*backward*/)
    { return {}; }

    // 绝对位置偏移（滑动窗生成；仅 RoPE 有意义）
    virtual void set_position_offset(std::size_t /*off*/) {}

    // ── 注意力**分数侧**（ALiBi 覆写；其余 no-op）────────────────────
    // 与 apply_qk 对称的另一条注入链：RoPE 旋转 Q/K，ALiBi 给分数加偏置。
    // 掩码（因果 / 文档）不属于位置编码，由注意力自己的掩码策略负责——
    // 两者正交，各自只写自己的那一项。
    [[nodiscard]] virtual bool has_score_bias() const noexcept { return false; }

    // 每步准备分数偏置表（batch/seq 变化时重建）
    [[nodiscard]] virtual Result<void> prepare_score_bias(
        ComputeEngine& /*engine*/, std::size_t /*batch*/, std::size_t /*seq*/)
    { return {}; }

    // fold 输入追加（斜率表）。调用方必须**先**调本函数、**再**追加掩码输入，
    // 顺序与 make_fold_attn_o 的 views 顺序一致（slopes → doc_col → doc_ids）。
    virtual void append_score_bias_inputs(std::vector<Tensor>& /*io*/) const {}

    // 分数偏置斜率表 (1, batch*H)（batch_mod(BH) 视图直读）
    [[nodiscard]] virtual const Tensor* bias_slopes() const noexcept { return nullptr; }

    // 反向：把位置偏置**原地**叠加到已掩码的 S 上（S += bias(S)；
    //   无偏置 = no-op）。为什么是"掩码之后、softmax 之前"独立一步：
    //   掩码与偏置是两件事，融成一条 select 会让掩码策略知道位置编码的存在。
    //   两种写法数值等价（被屏蔽处 -inf 加有限值仍是 -inf；未屏蔽处 s+0 精确），
    //   代价是反向多一个逐元素原地 kernel。
    [[nodiscard]] virtual Result<void> apply_score_bias(
        ComputeEngine& /*engine*/, Tensor& /*score*/,
        std::size_t /*BH*/, std::size_t /*seq*/)
    { return {}; }

    // 增量推理的分数偏置（scores 为 (H, new_len)）
    [[nodiscard]] virtual Result<Tensor> apply_bias_step(
        ComputeEngine& /*engine*/, Tensor&& scores, std::size_t /*cur_len*/,
        const PrecisionProfile& /*p*/) const
    { return std::move(scores); }
};

// ── 加性位置编码基类（Learned / Sinusoidal 共用）─────────────────────────
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

    // 用给定的位置编码张量初始化 pos_emb_（learnable 时额外分配梯度）。
    // M2：入参为 Tensor——调用方经引擎创建（InitSpec 填数或宿主公式 + span 上传），
    // 本函数只负责装配与梯度缓冲。
    [[nodiscard]] Result<void> init_(ComputeEngine& engine, Tensor&& pe, bool learnable)
    {
        const std::size_t rows = pe.rows();
        const std::size_t cols = pe.cols();
        learnable_ = learnable;
        pos_emb_ = std::move(pe);
        if (!pos_emb_.valid())
            return std::unexpected(Error{"PositionEncoder: 位置编码创建失败"});
        if (learnable_)
        {
            grad_pos_emb_ = engine.create_tensor(rows, cols, p_.param, InitSpec::zero());
            if (!grad_pos_emb_.valid())
                return std::unexpected(Error{"PositionEncoder: 梯度缓冲初始化失败"});
        }
        return {};
    }

    // 确保 pos_indices 缓存有效（batch-major：i = b*seq + t → position=t）
    [[nodiscard]] Result<void> ensure_pos_indices_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        if (pos_indices_batch_ == batch && pos_indices_seq_ == seq)
            return {};
        std::vector<Scalar> pidx(batch * seq);        // 宿主桥（17 §3 D11）：不经 Matrix
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t t = 0; t < seq; ++t)
                pidx[b * seq + t] = static_cast<Scalar>(t);
        auto pidx_t = detail::upload_span(engine, batch * seq, 1, Precision::F32,
                                          std::span(pidx));
        NN_TRY_CHECK(pidx_t);
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
        NN_TRY(ci, ensure_pos_indices_(engine, batch, seq));
        NN_TRY(pos_gathered, engine.gather_rows(pos_emb_, pos_indices_cache_));
        NN_TRY(pos_T, engine.transpose(*pos_gathered));
        auto x_with_pos = dsl::compute(engine,
            dsl::leaf(token_emb_T) + dsl::leaf(*pos_T),
            token_emb_T.rows(), token_emb_T.cols(), p_.compute);
        NN_TRY_CHECK(x_with_pos);
        return std::move(*x_with_pos);
    }

    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& x, std::size_t pos) override
    {
        std::vector<Scalar> pos_v(1);                 // 宿主桥（17 §3 D11）
        pos_v[0] = static_cast<Scalar>(pos);
        NN_TRY(pos_t, detail::upload_span(engine, 1, 1, Precision::F32, std::span(pos_v)));
        NN_TRY(pos_emb_g, engine.gather_rows(pos_emb_, *pos_t));
        NN_TRY(pos_T, engine.transpose(*pos_emb_g));
        auto x_wp = dsl::compute(engine,
            dsl::leaf(x) + dsl::leaf(*pos_T),
            x.rows(), x.cols(), p_.compute);
        NN_TRY_CHECK(x_wp);
        return std::move(*x_wp);
    }

    [[nodiscard]] Result<void> backward(
        ComputeEngine& engine, const Tensor& grad_T,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    {
        if (!learnable_) return {};
        // pos_indices 缓存由 apply() 建立，backward 直接复用（batch/seq 一致）。
        NN_TRY(pr, engine.scatter_add_rows(grad_pos_emb_, pos_indices_cache_, grad_T));
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

// ── 可学习位置编码（GPT 默认）：N(0, 0.02) 随机初始化 ─────────────────────
class LearnedPositionEncoder final : public AdditivePositionEncoder
{
    std::size_t d_model_;
    std::size_t seq_len_;

public:
    LearnedPositionEncoder(std::size_t d_model, std::size_t seq_len)
        : d_model_(d_model), seq_len_(seq_len) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // N(0, 0.02) 随机初始化——M2 声明式：层算分布参数、引擎填数，
        // 分布 seed 显式传（U1）；不再自持 RNG。
        auto pe = engine.create_tensor(seq_len_, d_model_, p_.param,
                                       InitSpec::normal(0, 0.02, kInitSeed));
        if (!pe.valid())
            return std::unexpected(Error{"LearnedPositionEncoder: 初始化失败"});
        return init_(engine, std::move(pe), /*learnable=*/true);
    }
};

// ── 正弦波固定位置编码（冻结，不参与训练）───────────────────────────────
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
        std::vector<Scalar> pe(seq_len_ * d_model_);   // 宿主桥（17 §3 D11）
        for (std::size_t pos = 0; pos < seq_len_; ++pos)
            for (std::size_t i = 0; i < d_model_; ++i)
            {
                Scalar angle = static_cast<Scalar>(pos) /
                    std::pow(Scalar{10000}, static_cast<Scalar>(2 * (i / 2)) / static_cast<Scalar>(d_model_));
                pe[pos * d_model_ + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        // 闭式公式（非分布/常数）仍宿主计算 → span 上传进引擎（M2 裁定：InitSpec
        // 只收分布与常数初始化；公式数据自 M4 起经 detail::upload_span，不经 Matrix）
        auto pe_t = detail::upload_span(engine, seq_len_, d_model_, p_.param,
                                        std::span(pe));
        NN_TRY_CHECK(pe_t);
        return init_(engine, std::move(*pe_t), /*learnable=*/false);
    }
};

// ── 无位置编码：真正的恒等策略（默认在所有注入点都不动作）────────────────
class NoPositionEncoder final : public PositionEncoder
{
public:
    NoPositionEncoder() = default;
};

// ── RoPE：注意力 Q/K 侧旋转位置编码 ───────────────────────────────────────
class RopePositionEncoder final : public PositionEncoder
{
private:
    RotaryEmbedding rope_;

public:
    explicit RopePositionEncoder(std::size_t d_k) : rope_(d_k) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        return rope_.init(engine);
    }

    void set_precision_profile(const PrecisionProfile& p) override
    {
        PositionEncoder::set_precision_profile(p);
        rope_.set_precision_profile(p);
    }

    [[nodiscard]] Result<void> apply_qk(
        ComputeEngine& engine, Tensor& qk, std::size_t seq, bool backward) override
    {
        NN_TRY(r, rope_.apply(engine, qk, seq, backward));
        qk = std::move(*r);
        return {};
    }

    [[nodiscard]] Result<void> apply_qk_step(
        ComputeEngine& engine, Tensor& qk, std::size_t pos, bool backward) override
    {
        NN_TRY(r, rope_.apply_step(engine, qk, pos, backward));
        qk = std::move(*r);
        return {};
    }

    void set_position_offset(std::size_t off) override { rope_.set_position_offset(off); }
    [[nodiscard]] std::size_t d_k() const noexcept { return rope_.d_k(); }
};

// ── ALiBi：注意力分数侧线性距离偏置 ──────────────────────────────────────
//   score[i][j] += −m_h · (i − j)，m_h = 2^(−8h/H)（h 为头下标）
//   斜率表按 (b,h) 块重复为 (1, batch*H)：掩码表达式的 batch_mod(BH) 视图
//   按下标 b*H+h 直读；若用 (1,H) 表，batch≥2 时越界读且**静默错**
//   （batch=1 时下标恒 <H，恰好掩盖该错误）——故 batch 变化时必须重建。
class AlibiPositionEncoder final : public PositionEncoder
{
private:
    std::size_t num_heads_;
    std::vector<Scalar> slopes_;        // m_h
    Tensor slopes_cache_;               // (1, batch*num_heads_) —— fold 输入（batch_mod 视图）
    Tensor slope_row_cache_;            // (B*H*seq, 1) —— 反向：每行斜率 m_{h(row)}
    Tensor pos_row_cache_;              // (B*H*seq, 1) —— 反向：批内位置 row % seq
    std::size_t cached_batch_ = 0;
    std::size_t cached_seq_ = 0;

public:
    explicit AlibiPositionEncoder(std::size_t num_heads) : num_heads_(num_heads)
    {
        slopes_.resize(num_heads);
        for (std::size_t h = 0; h < num_heads; ++h)
            slopes_[h] = std::pow(Scalar{2}, -Scalar{8} * h / num_heads);
    }

    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] bool has_score_bias() const noexcept override { return true; }

    [[nodiscard]] Result<void> prepare_score_bias(
        ComputeEngine& engine, std::size_t batch, std::size_t seq) override
    {
        if (slopes_cache_.valid() && cached_batch_ == batch && cached_seq_ == seq)
            return {};
        const std::size_t BH = batch * num_heads_;
        const std::size_t rows = BH * seq;
        // ① fold 输入 (1, BH)：fold 的 batch_mod(bh) 按下标 b*H+h 直读
        std::vector<Scalar> s(BH);   // 宿主桥（17 §3 D11）
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t h = 0; h < num_heads_; ++h)
                s[b * num_heads_ + h] = slopes_[h];
        NN_TRY(t, detail::upload_span(engine, 1, BH, Precision::F32, std::span(s)));
        slopes_cache_ = std::move(*t);
        // ② 反向行表 (rows,1)：row_broadcast 按全局行号直读 → 不需要 grid 的
        //    batch 分解（不能依赖 matmul 段，见 apply_score_bias 注释）
        std::vector<Scalar> sr(rows), pr(rows);
        for (std::size_t r = 0; r < rows; ++r)
        {
            sr[r] = slopes_[(r / seq) % num_heads_];
            pr[r] = static_cast<Scalar>(r % seq);
        }
        NN_TRY(tr, detail::upload_span(engine, rows, 1, Precision::F32, std::span(sr)));
        slope_row_cache_ = std::move(*tr);
        NN_TRY(tp, detail::upload_span(engine, rows, 1, Precision::F32, std::span(pr)));
        pos_row_cache_ = std::move(*tp);
        cached_batch_ = batch;
        cached_seq_ = seq;
        return {};
    }

    void append_score_bias_inputs(std::vector<Tensor>& io) const override
    { io.push_back(slopes_cache_); }

    [[nodiscard]] const Tensor* bias_slopes() const noexcept override
    { return &slopes_cache_; }

    // ALiBi 项：m_h · (Col − Row)（BatchMod 按下标 b*H+h 直读斜率表）
    // 反向：S += m_h · (Col − Row)，写回 S。
    // 调用点在掩码之后、softmax 之前；非 ALiBi 策略走基类 no-op。
    //
    // ⚠ 为什么用 row_broadcast(行表) 而不是 dsl::col() - dsl::row()：
    //   `row()` 是"批内行号"，其分解来自同一 ExprSpec 里的 matmul 段
    //   （MatmulSpec.batch）。旧实现把偏置项融进掩码表达式，那个表达式里恰好有
    //   `dsl::matmul(..., BH)`，所以 row() 正确；把它拆成独立一步后 spec 里没有
    //   matmul → batch 退化为 1 → row() 变成**全局行号** → 偏置静默错值
    //   （实测 |bias| 达 27，而理论界为 (seq−1)·m_0≈7，alibi gradcheck 直接失败）。
    //   (rows,1) 行表 + row_broadcast 按**全局行号**直读，语义与网格分解无关，
    //   因而独立成步也正确。表长 O(B·H·seq)，与掩码的 doc 缓存同量级。
    [[nodiscard]] Result<void> apply_score_bias(
        ComputeEngine& engine, Tensor& score,
        std::size_t /*BH*/, std::size_t /*seq*/) override
    {
        NN_ASSERT(slope_row_cache_.valid() && pos_row_cache_.valid(),
                  "AlibiPositionEncoder: prepare_score_bias 未被调用");
        NN_TRY(res, dsl::compute(engine,
            dsl::leaf(score)
                + dsl::row_broadcast(slope_row_cache_)
                      * (dsl::col() - dsl::row_broadcast(pos_row_cache_)),
            score.rows(), score.cols(), score.precision()));
        score = std::move(*res);
        return {};
    }

    // 增量推理：scores 布局 (H, new_len)，new_len = cur_len + 1；
    // query 在位置 cur_len，key 在位置 j → 偏置 = −slope[h] · (cur_len − j)
    [[nodiscard]] Result<Tensor> apply_bias_step(
        ComputeEngine& engine, Tensor&& scores, std::size_t cur_len,
        const PrecisionProfile& p) const override
    {
        const std::size_t new_len = cur_len + 1;
        std::vector<Scalar> bias(num_heads_ * new_len);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            const Scalar slope = slopes_[h];
            for (std::size_t j = 0; j < new_len; ++j)
            {
                const std::size_t dist = cur_len - j;
                bias[h * new_len + j] = -slope * static_cast<Scalar>(dist);
            }
        }
        auto bias_t = detail::upload_span(engine, num_heads_, new_len, Precision::F32,
                                          std::span(bias));
        NN_TRY_CHECK(bias_t);
        return dsl::compute(engine,
            dsl::leaf(scores) + dsl::leaf(*bias_t),
            scores.rows(), scores.cols(), p.compute);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  工厂 —— **全仓 PosEncodingType 分发的两个落点**（按注入点一分为二）
//
//  位置编码的两类注入点归属不同主体，故工厂也分开；同一策略类型只在一个
//  工厂里是"实做"，在另一个里映射为无位置编码（NoPositionEncoder）：
//
//    · make_embedding_position_encoder（**模型侧**，GPTModel/ZiPTModel/RAPTModel
//      持有并在嵌入侧 apply）：Learned / Sinusoidal 实做；ALiBi / RoPE → 恒等。
//    · make_attention_position_encoder（**注意力层自持**，CausalSelfAttention /
//      ReLULinearAttention 持有）：ALiBi / RoPE 实做；Learned / Sinusoidal → 恒等。
//
//  这样"谁拥有 = 谁负责"：嵌入侧位置信息归模型，Q/K 旋转与分数偏置归注意力层，
//  每个注入点只构造自己那一半，不会有"模型级对象有时给注意力用"的混合状态。
//
//  注意（RoPE 的表）：分给注意力层意味着**每层各持一份 cos/sin 表**——
//  每层 2·d_k·seq·4B（本仓默认 d_model=128/H=4/layers=4/seq=256 ≈ 64KB/层，
//  全模型 256KB）。层数很多时才需重新考虑共享。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::unique_ptr<PositionEncoder> make_embedding_position_encoder(
    PosEncodingType type, std::size_t d_model, std::size_t seq_len)
{
    switch (type)
    {
        case PosEncodingType::Learned:
            return std::make_unique<LearnedPositionEncoder>(d_model, seq_len);
        case PosEncodingType::Sinusoidal:
            return std::make_unique<SinusoidalPositionEncoder>(d_model, seq_len);
        default:   // ALiBi / RoPE：位置信息在注意力层注入，嵌入侧无动作
            return std::make_unique<NoPositionEncoder>();
    }
}

[[nodiscard]] inline std::unique_ptr<PositionEncoder> make_attention_position_encoder(
    PosEncodingType type, std::size_t d_k, std::size_t num_heads)
{
    switch (type)
    {
        case PosEncodingType::ALiBi:
            return std::make_unique<AlibiPositionEncoder>(num_heads);
        case PosEncodingType::RoPE:
            return std::make_unique<RopePositionEncoder>(d_k);
        default:   // Learned / Sinusoidal：位置信息在嵌入侧注入，注意力层无动作
            return std::make_unique<NoPositionEncoder>();
    }
}

} // namespace nn
