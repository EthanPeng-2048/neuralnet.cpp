#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_softmax.hpp"
#include "compute_position_encoding.hpp"
#include "expr_spec.hpp"

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

// ═══════════════════════════════════════════════════════════════════════════
//  注意力 fold 构造（双域旗舰样例）：fold attention 输出 O
//  （S 不物化，online 单遍）
//
//  这是**注意力 Layer 自己的表达式文本**（AOT 收集原则：表达式只出现在
//  Layer）：本文件是 make_fold_attn_o 构造与 AttnMaskKind 掩码种类的定义
//  来源，expr_fold.hpp 只含与注意力无关的通用折叠样例。
//  scan_exprs dry-run、fused_gpu/expr_cpu 对拍与 AttentionBase::forward
//  共用本构造 → key 一致。
//
//  数学：O = softmax_masked(Q·Kᵀ) · V_t
//    键域每块：S_tile = mm(Q,K)（内层 dk 收缩）→ [掩码] →
//      m' = max(m, 块 max)；α = e^{m−m'}；e = e^{S−m'}；
//      l = l·α + Σ_块 e；（块尾 vecacc：O *= α；O += Σ_块 e·V_t）
//    向量域：out[d] = O[d] / l
//  布局：Q/K (bh·dk, seq) 行主序；V_t (bh·seq, dk)；输出 (bh·seq, dk)。
//  掩码语义与掩码策略类的 masked_scores() 对齐（Row/Col 按 batched
//  网格：Row = 行%(rows/batch) = 查询位置 i、Col = 全局键位置 j）。
// ═══════════════════════════════════════════════════════════════════════════
namespace expr
{

// 注意力**掩码**语义 —— 只管键的可达性，不含任何位置信息。
//   Plain     无掩码（仅测试/数学对拍用——Layer 恒因果系）
//   Causal    因果（j>i → -inf）
//   CausalDoc 因果 + 文档块对角（doc_col 经 RowBroadcast、doc_ids 经 BatchCol）
// 位置偏置（ALiBi）不属于掩码：它由位置编码策略给出，作为独立的第二个维度
// 传入（score_bias）。两者正交 —— 掩码策略不必知道位置编码的存在。
enum class AttnMaskKind
{
    Plain,
    Causal,
    CausalDoc,
};

// fold 构造：掩码 × 位置偏置两个正交维度；生成的 IR 指令顺序固定为
//   掩码（causal → doc）→ ALiBi 斜率项，与拆分前的 5 个融合变体逐指令一致。
[[nodiscard]] inline ExprSpec make_fold_attn_o(std::uint32_t seq, std::uint32_t dk,
                                               std::uint32_t bh, AttnMaskKind mask,
                                               bool score_bias)
{
    ExprSpec s;
    FoldSpec f;
    f.k             = seq;
    f.num_state     = 2;                       // 0 = m, 1 = l
    f.inits         = { -std::numeric_limits<Scalar>::infinity(), Scalar{0} };
    f.vec_state_len = dk;
    f.matmul        = MatmulSpec{ 0, 1, 1, 0, dk, bh };  // A=Q(trans), B=K, k=dk, batch=bh
    // 常量：c0 = 0（拷贝/中性元）、c1 = -inf（掩码屏蔽）
    s.consts = { Scalar{0}, -std::numeric_limits<Scalar>::infinity() };
    // 输入/视图槽位（与掩码策略 append_fold_inputs 的追加顺序约定一致）：
    //   0=Q 1=K 2=V_t；[slopes (1,bh)→BatchMod]；[doc_col (BH*seq,1)→RowBroadcast]；
    //   [doc_ids (1,bh*seq)→BatchCol(seq)]
    const bool has_slope = score_bias;
    const bool has_doc   = (mask == AttnMaskKind::CausalDoc);
    const bool has_causal = (mask != AttnMaskKind::Plain);
    f.tri_skip = has_causal;   // 生成器据此钳 valid（整块/边界跳过 -inf 区）
    s.views = { linear(), linear(), linear() };
    std::uint8_t slot_slope = 0, slot_dc = 0, slot_ids = 0;
    if (has_slope)
    {
        slot_slope = static_cast<std::uint8_t>(s.views.size());
        s.views.push_back(batch_mod(bh));
    }
    if (has_doc)
    {
        slot_dc = static_cast<std::uint8_t>(s.views.size());
        s.views.push_back(row_broadcast());          // doc_col (rows,1)：b[row]
        slot_ids = static_cast<std::uint8_t>(s.views.size());
        s.views.push_back(batch_col(seq));           // doc_ids：b[batch*seq + col]
    }

    const auto ins = [](ExprOp op, std::uint8_t dst, ExprOperand a,
                        ExprOperand b = {}) {
        ExprInstr i;
        i.op  = static_cast<uint8_t>(op);
        i.dst = dst;
        i.a   = a;
        i.b   = b;
        return i;
    };
    std::uint8_t nreg = 2;                     // 状态占 0..1，临时从 2 递增
    std::vector<ExprInstr> body;

    // ── 键域前段：S = masked(mm) ──
    const uint8_t s0 = nreg++;                 // S 经 nreg 分配（防与掩码临时撞号）
    body.push_back(ins(ExprOp::Add, s0, matmul_op(), cst(0)));  // r{s0} = S
    std::uint8_t S = s0;
    if (has_causal)
    {
        const uint8_t gt = nreg++;             // Gt(Col, Row)
        body.push_back(ins(ExprOp::Gt, gt, col(), row()));
        const uint8_t sel = nreg++;            // Select(gt, c1=-inf, c0=0)
        {
            ExprInstr isel;                     // 三操作数指令：手填 c 字段
            isel.op  = static_cast<uint8_t>(ExprOp::Select);
            isel.dst = sel;
            isel.a   = reg(gt);
            isel.b   = cst(1);
            isel.c   = cst(0);
            body.push_back(isel);
        }
        const uint8_t sm = nreg++;             // S + 屏蔽项
        body.push_back(ins(ExprOp::Add, sm, reg(S), reg(sel)));
        S = sm;
    }
    if (has_doc)
    {
        // 文档块对角：Ne(doc_col[row], doc_ids[batch,col]) → 屏蔽 -inf
        //   （与 CausalDocScoreMask::masked_scores 同构：两层 select 合并因果后统一 -inf）
        const uint8_t ne = nreg++;
        body.push_back(ins(ExprOp::Ne, ne, input(slot_dc), input(slot_ids)));
        const uint8_t ds = nreg++;
        {
            ExprInstr isel;
            isel.op  = static_cast<uint8_t>(ExprOp::Select);
            isel.dst = ds;
            isel.a   = reg(ne);
            isel.b   = cst(1);
            isel.c   = cst(0);
            body.push_back(isel);
        }
        const uint8_t sm = nreg++;
        body.push_back(ins(ExprOp::Add, sm, reg(S), reg(ds)));
        S = sm;
    }
    if (has_slope)
    {
        const uint8_t off = nreg++;            // slope[batch] * (Col - Row)
        body.push_back(ins(ExprOp::Sub, off, col(), row()));
        const uint8_t sl = nreg++;
        body.push_back(ins(ExprOp::Mul, sl, input(slot_slope), reg(off)));
        const uint8_t sa = nreg++;
        body.push_back(ins(ExprOp::Add, sa, reg(S), reg(sl)));
        S = sa;
    }

    // ── 键域中段：online m / α / e / l ──
    const uint8_t m_old = nreg++;              // m 拷贝
    body.push_back(ins(ExprOp::Add, m_old, reg(0), cst(0)));
    const uint8_t blk_m = nreg++;              // 块内 S max（归约）
    body.push_back(ins(ExprOp::RowMax, blk_m, reg(S)));
    body.push_back(ins(ExprOp::Max, 0, reg(m_old), reduce(blk_m)));   // m ← 状态更新
    const uint8_t dm = nreg++;                 // m_old − m'
    body.push_back(ins(ExprOp::Sub, dm, reg(m_old), reg(0)));
    const uint8_t alpha = nreg++;              // α（行标量 → vecacc.scale）
    body.push_back(ins(ExprOp::Exp, alpha, reg(dm)));
    const uint8_t es = nreg++;                 // S − m'
    body.push_back(ins(ExprOp::Sub, es, reg(S), reg(0)));
    const uint8_t weight = nreg++;             // e = e^{S−m'}（元素 → vecacc.weight）
    body.push_back(ins(ExprOp::Exp, weight, reg(es)));
    const uint8_t blk_l = nreg++;              // 块内 Σe（归约）
    body.push_back(ins(ExprOp::RowSum, blk_l, reg(weight)));
    const uint8_t la = nreg++;                 // l·α
    body.push_back(ins(ExprOp::Mul, la, reg(1), reg(alpha)));
    body.push_back(ins(ExprOp::Add, 1, reg(la), reduce(blk_l)));      // l ← 状态更新

    // ── vecacc：O *= α；O += Σ_块 e·V_t ──
    f.vecacc  = VecAccSpec{ 0, weight, 2, alpha, 1 };
    f.body    = std::move(body);
    // ── 向量域 finalize：out[d] = O[d] / l ──
    // dst 必须用**临时寄存器**（新号）：向量域逐列循环执行，若 dst 复用被读
    // 状态（如 l），第 1 列的输出会覆盖除数 → 第 2 列起读到被污染的值
    // （实测 0.4 = 10/25：l=4 被首列输出 25 覆盖）。
    {
        const uint8_t fout = nreg++;
        f.finalize = { ins(ExprOp::Div, fout, nn::expr::vec_state(0), reg(1)) };
    }
    s.fold     = std::move(f);
    s.num_regs = nreg;
    return s;
}

} // namespace expr

// ═══════════════════════════════════════════════════════════════════════════
//  AttnScoreMask — 注意力**掩码策略**（一种具体策略 = 一种掩码语义）
//
//  职责边界（本轮拆分的要点）：**只做掩码**，不含任何位置信息。
//    · 因果掩码 / 文档块对角是"键能不能被看到"——注意力自己的事；
//    · ALiBi 是"位置带来的分数偏移"——位置编码的事（AlibiPositionEncoder）。
//  两者正交：`masked_scores()` 产出**掩码后**的 S，位置偏置由
//  `PositionEncoder::apply_score_bias()` 在"掩码之后、softmax 之前"独立叠加。
//  于是本族不必知道 `PositionEncoder` 的存在——旧版的 `CausalAlibiScoreMask`
//  / `CausalAlibiDocScoreMask` 正是把两者融成一条 select 的产物，已删除。
//
//  为什么是策略对象而不是基类里的一串 if：
//    掩码语义（无掩码 / 因果 / 因果+文档）在**构造期**就确定，旧实现却把判定
//    留在热路径（`recompute_W_` 每次反向重跑组合链）。现在：具体策略类在配置期
//    选定（工厂 `make_score_mask_()` 在构造 / `set_doc_ids` 里跑一次），
//    forward/backward 只做一次虚调用。
//
//  接口（全部在"每调用一次"粒度，不在块内循环）：
//    mask_kind()          → forward fold kernel 的掩码种类（类级常量）
//    prepare()            → 构建本策略的掩码输入张量（文档 id / 列）
//    append_fold_inputs() → 按 make_fold_attn_o 的 views 顺序追加 fold 输入
//    masked_scores()      → 反向重算 S 的 DSL 文本（与 fold 掩码语义同构）
//
//  AOT 闭合世界：每条 masked_scores() 的表达式文本都必须被 scan_exprs 的
//  对应掩码 dry-run 覆盖。
// ═══════════════════════════════════════════════════════════════════════════
class AttnScoreMask
{
protected:
    AttnScoreMask() = default;

public:
    virtual ~AttnScoreMask() = default;
    AttnScoreMask(const AttnScoreMask&) = delete;
    AttnScoreMask& operator=(const AttnScoreMask&) = delete;

    // forward fold kernel 的掩码种类（类级常量，构造期定型）
    [[nodiscard]] virtual expr::AttnMaskKind mask_kind() const noexcept = 0;

    // 每步准备掩码输入张量（无掩码分量 = no-op）
    [[nodiscard]] virtual Result<void> prepare(
        ComputeEngine& /*engine*/, std::size_t /*batch*/, std::size_t /*seq*/)
    { return {}; }

    // fold 输入追加：顺序必须与 make_fold_attn_o 的 views 顺序一致
    // （Q,K,V_t 之后；位置偏置输入由 PositionEncoder 先行追加，故本函数在它之后调）
    virtual void append_fold_inputs(std::vector<Tensor>& /*io*/) const {}

    // 反向：S = mask(Q·Kᵀ)（DSL 文本 —— 与 forward fold 掩码语义同构；
    //   位置偏置由位置编码策略另行叠加，见 PositionEncoder::apply_score_bias）
    [[nodiscard]] virtual Result<Tensor> masked_scores(
        ComputeEngine& engine, const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq, const PrecisionProfile& p) const = 0;

    // 文档 id 注入（仅文档感知策略覆写；其余 no-op = 无文档掩码）
    virtual void set_doc_ids(std::span<const std::size_t> /*ids*/) {}
};

// ── 无掩码（双向）：MHA ──────────────────────────────────────────────────
class PlainScoreMask final : public AttnScoreMask
{
public:
    [[nodiscard]] expr::AttnMaskKind mask_kind() const noexcept override
    { return expr::AttnMaskKind::Plain; }

    [[nodiscard]] Result<Tensor> masked_scores(
        ComputeEngine& engine, const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq, const PrecisionProfile& p) const override
    {
        return dsl::compute(engine, dsl::matmul(Q, K, true, false, BH),
                            BH * seq, seq, p.compute);
    }
};

// ── 因果掩码（j > i → -inf）──────────────────────────────────────────────
// 表达式文本逐字保留（AOT key 不变）；文档变体继承本类（因果是底座）。
class CausalScoreMask : public AttnScoreMask
{
public:
    static constexpr Scalar kNegInf = -std::numeric_limits<Scalar>::infinity();

    [[nodiscard]] expr::AttnMaskKind mask_kind() const noexcept override
    { return expr::AttnMaskKind::Causal; }

    [[nodiscard]] Result<Tensor> masked_scores(
        ComputeEngine& engine, const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq, const PrecisionProfile& p) const override
    {
        const auto scores = dsl::matmul(Q, K, true, false, BH);
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        return dsl::compute(engine,
            scores + dsl::select(causal != Scalar{0}, kNegInf, Scalar{0}),
            BH * seq, seq, p.compute);
    }
};

// ── 因果 + 文档块对角（跨文档禁止注意）──────────────────────────────────
class CausalDocScoreMask final : public CausalScoreMask
{
protected:
    std::size_t num_heads_;
    std::vector<std::size_t> doc_ids_;   // 批内每位置文档 id（batch-major）
    bool has_doc_ids_ = false;
    Tensor doc_ids_cache_;   // (1, BH*seq)：每 (b,h) 块重复 doc_ids[b*seq..]
    Tensor doc_col_;         // (BH*seq, 1)：每行文档 id（跨 head 重复）

public:
    explicit CausalDocScoreMask(std::size_t num_heads) : num_heads_(num_heads) {}

    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); has_doc_ids_ = false; return; }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

    [[nodiscard]] bool doc_mask_enabled() const noexcept { return has_doc_ids_; }

    [[nodiscard]] expr::AttnMaskKind mask_kind() const noexcept override
    { return expr::AttnMaskKind::CausalDoc; }

    [[nodiscard]] Result<void> prepare(
        ComputeEngine& engine, std::size_t batch, std::size_t seq) override
    {
        // 文档感知：doc_ids_ 每 step 变化，每步重建（小张量 O(BH*seq)）。
        // doc_ids_cache_ 为 (1, BH*seq) 布局——每 (b,h) 块重复 doc_ids[b*seq..]
        //   （BatchCol 视图的 batch 下标 = BH 网格下标，故不能用 (1,batch*seq)）
        // doc_col_（(BH*seq,1)）每行文档 id（跨 head 重复；IR 掩码按行广播读取）。
        //   两者数值相同，共用一份宿主缓冲、按不同形状上传两次。
        const std::size_t BH = batch * num_heads_;
        std::vector<Scalar> d(BH * seq);   // 宿主桥（17 §3 D11）
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t h = 0; h < num_heads_; ++h)
                for (std::size_t i = 0; i < seq; ++i)
                    d[(b * num_heads_ + h) * seq + i] =
                        static_cast<Scalar>(doc_ids_[b * seq + i]);
        auto t = detail::upload_span(engine, 1, BH * seq, Precision::F32, std::span(d));
        NN_TRY_CHECK(t);
        doc_ids_cache_ = std::move(*t);
        auto tc = detail::upload_span(engine, BH * seq, 1, Precision::F32, std::span(d));
        NN_TRY_CHECK(tc);
        doc_col_ = std::move(*tc);
        return {};
    }

    void append_fold_inputs(std::vector<Tensor>& io) const override
    {
        io.push_back(doc_col_);
        io.push_back(doc_ids_cache_);
    }

    [[nodiscard]] Result<Tensor> masked_scores(
        ComputeEngine& engine, const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq, const PrecisionProfile& p) const override
    {
        const auto scores = dsl::matmul(Q, K, true, false, BH);
        const auto causal = dsl::select(dsl::col() > dsl::row(),
                                        Scalar{1}, Scalar{0});
        const auto blocked = causal + dsl::select(
            dsl::row_broadcast(doc_col_) != dsl::batch_col(doc_ids_cache_, seq),
            Scalar{1}, Scalar{0});
        return dsl::compute(engine,
            scores + dsl::select(blocked != Scalar{0}, kNegInf, Scalar{0}),
            BH * seq, seq, p.compute);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  AttentionBase — 多头注意力**核心**（批量化：消除 per-head / per-sample 循环）
//
//  职责只剩"注意力数学本身"，两类可变量各由**一个构造期定型的策略对象**承载，
//  基类里不再有 use_rope_ / use_alibi_ / use_doc_ids_ 之类的标志位：
//
//    · PositionEncoder（位置编码，见 compute_position_encoding.hpp）
//        - apply_qk()/apply_qk_step() → RoPE 旋转 Q/K
//        - apply_score_bias()/apply_bias_step() → ALiBi 分数偏置（原地叠加）
//        - 其余编码类型在注意力侧为 no-op → 热路径无任何判断
//    · AttnScoreMask（**掩码**，见上；与位置编码正交）
//        - 构造期/配置期由 make_score_mask_() 选定，前/反向各一次虚调用
//
//  算法（只在此处，不在 Engine/Shader）：
//    Q = W_q × x, K = W_k × x, V = W_v × x  (三个 Linear 投影)
//    Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//    rearrange_3d → (batch*H*d_k, seq)，单次处理所有样本和所有头。
//
//    forward（单 fold 路径，S 不物化）：
//      Q/K = rearrange_3d → (batch*H*d_k, seq)；[位置编码]；Q *= scale（折进 Q）
//      V_t = transpose + rearrange → (BH*seq, d_k)
//      O_t = eval_expr(make_fold_attn_o(..., mask_kind, has_score_bias),
//                      {Q, K, V_t, [slopes], [掩码输入]}, BH*seq, d_k)
//        —— QKᵀ/掩码/位置偏置/online softmax/ΣwV 全在单个 fold kernel 内逐块完成
//      O = transpose + rearrange 回 (H*d_k, batch*seq)；out = W_o × O
//    backward：S = mask(Q·Kᵀ) → S += bias(S)（原地）→ softmax，随后 R/X 表达式
//      + batched_matmul。两条链各一次虚调用、互相独立。
//
//  输入形状: (d_model, batch * seq_len)，输出形状: (d_model, batch * seq_len)
//    seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断
// ═══════════════════════════════════════════════════════════════════════════
class AttentionBase : public Layer
{
public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "Attention"; }

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

    // ── 位置编码策略（**注意力层自持**，构造期由子类经
    //   install_position_encoder() 装入）────────────────────────────────
    //   注意力侧只有 RoPE（Q/K 旋转）与 ALiBi（分数偏置）会实做，其余类型由
    //   工厂映射为恒等策略 → 热路径里没有任何按编码类型的分支。
    //   默认恒等：AttentionBase/MultiHeadAttention 不装可用的编码器也对。
    //   （嵌入侧的 Learned/Sinusoidal 归模型所有，见
    //   make_embedding_position_encoder。）
    std::unique_ptr<PositionEncoder> pos_ = std::make_unique<NoPositionEncoder>();

    // ── 分数掩码策略（构造期/配置期定型；热路径只调用，不判断变体）──
    std::unique_ptr<AttnScoreMask> mask_;

    // ── SubLN（BitNet b1.58 的"子层归一化"，docs/development/22 §3.1）──────
    // **子层内部**的额外归一化：挂在 attention 输出（宽 d_model）与输出投影
    // w_o_ 之间。默认 nullptr = 关（既有路径逐位不变）。类型跟随模型的
    // norm_type（2B4T = RMSNorm）——与 GPTBlock 的 norm1_/norm2_ 同一工厂。
    //
    // 位置：注意力输出 → **attn_sub_norm_ → w_o_**。不能挂到 block 级：
    // block 只看到 w_o_ 之后的 (d_model, N)，看不到投影前的 concat。
    std::unique_ptr<Layer> attn_sub_norm_;

    // forward 缓存（rearranged 版本，供 backward 直接使用；得分矩阵类量
    // 不作缓存——W/m/l/attn 均不物化，W 在 backward 重算）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged

    // ── 掩码策略工厂 ────────────────────────────────────────────────────
    //  子类按自己的掩码语义覆写；只在**配置期**调用（构造 / set_doc_ids），
    //  不在 forward/backward 里调用。
    //  fail-safe 默认 = 无掩码（双向），与旧 fold_mask_variant_ 默认 Plain 一致。
    [[nodiscard]] virtual std::unique_ptr<AttnScoreMask> make_score_mask_() const
    { return std::make_unique<PlainScoreMask>(); }

    void install_score_mask_() { mask_ = make_score_mask_(); }

    [[nodiscard]] AttnScoreMask& score_mask() noexcept
    {
        NN_ASSERT(mask_ != nullptr, "AttentionBase: score mask not installed");
        return *mask_;
    }

    // ── 配置期（子类构造器内）：装入注意力侧位置编码策略 ─────────────────
    //  注意力侧的位置编码（RoPE 的 Q/K 旋转、ALiBi 的分数偏置）**由本层自持**；
    //  嵌入侧（Learned/Sinusoidal）归模型，经 make_embedding_position_encoder
    //  在模型侧施加。
    //  注意：本函数**不**再定型掩码策略——掩码（因果/文档）与位置偏置已经正交，
    //  掩码策略由子类独立安装（`install_score_mask_()`），不再依赖位置编码。
    void install_position_encoder(std::unique_ptr<PositionEncoder> enc)
    {
        pos_ = std::move(enc);
        NN_ASSERT(pos_ != nullptr, "AttentionBase: position encoder required");
        pos_->set_precision_profile(p_);
    }

public:
    AttentionBase(std::size_t d_model, std::size_t num_heads,
                  std::size_t seq_len = 0,
                  bool subln = false,
                  NormType subln_norm_type = NormType::LayerNorm)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(d_model, d_model),
          w_k_(d_model, d_model),
          w_v_(d_model, d_model),
          w_o_(d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "AttentionBase: d_model must be divisible by num_heads");
        mask_ = std::make_unique<PlainScoreMask>();
        if (subln)
            attn_sub_norm_ = make_norm_layer(d_model, subln_norm_type);
    }

    [[nodiscard]] PositionEncoder& position_encoder() noexcept { return *pos_; }
    [[nodiscard]] const PositionEncoder& position_encoder() const noexcept { return *pos_; }

    // ── D7：精度配置下传（§9.2）──────────────────────────────────────────
    // 注意力是复合层（4 个投影 Linear + Softmax + 位置编码）；子层/辅助对象必须
    // 一起拿到 profile，否则它们的 p_ 停在默认 F32：参数仍按 F32 创建、
    // DSL 求值退回 F32 —— f16 配置下静默失效。
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        w_q_.set_precision_profile(profile);
        w_k_.set_precision_profile(profile);
        w_v_.set_precision_profile(profile);
        w_o_.set_precision_profile(profile);
        softmax_.set_precision_profile(profile);
        pos_->set_precision_profile(profile);   // 位置编码（RoPE 表 / ALiBi 斜率）
        if (attn_sub_norm_) attn_sub_norm_->set_precision_profile(profile);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, w_q_.init(engine));
        NN_TRY(r2, w_k_.init(engine));
        NN_TRY(r3, w_v_.init(engine));
        NN_TRY(r4, w_o_.init(engine));
        // M6 段 C：softmax_ 是子 Layer（engine 由 init 绑定），否则 forward 内 fail-fast
        { NN_TRY(r6, softmax_.init(engine)); }
        // 注意力侧位置编码（RoPE/ALiBi 的 init 是 no-op；恒等策略同样 no-op）——
        // 按 M6 段 C 的不变量"复合层 init_impl 必须 init 全部子对象"统一调用。
        if (attn_sub_norm_) { NN_TRY(r7, attn_sub_norm_->init(engine)); }
        return pos_->init(engine);
    }

    std::vector<TensorRef> parameters() override
    {
        auto r = collect_refs(w_q_.parameters(), w_k_.parameters(),
                              w_v_.parameters(), w_o_.parameters());
        if (attn_sub_norm_)
        {
            auto s = attn_sub_norm_->parameters();
            r.insert(r.end(), s.begin(), s.end());
        }
        return r;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto r = collect_refs(w_q_.param_gradients(), w_k_.param_gradients(),
                              w_v_.param_gradients(), w_o_.param_gradients());
        if (attn_sub_norm_)
        {
            auto s = attn_sub_norm_->param_gradients();
            r.insert(r.end(), s.begin(), s.end());
        }
        return r;
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
        if (attn_sub_norm_) attn_sub_norm_->set_checkpoint_mode(enabled);
    }

    // 文档感知：转发给掩码策略（非文档策略 = no-op，与旧"无 doc 钩子"一致）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        score_mask().set_doc_ids(ids);
    }

    void clear_cache() override
    {
        Q_cache_ = Tensor{};
        K_cache_ = Tensor{};
        V_cache_ = Tensor{};
        // 掩码/偏置描述子（掩码策略持有的 doc/slopes 缓存）小而常驻，
        // 不随激活清理
        w_q_.clear_cache();
        w_k_.clear_cache();
        w_v_.clear_cache();
        w_o_.clear_cache();
        softmax_.clear_cache();
        if (attn_sub_norm_) attn_sub_norm_->clear_cache();
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
        if (attn_sub_norm_)
        {
            auto sn = attn_sub_norm_->activation_cache();
            r.insert(r.end(), sn.begin(), sn.end());
        }
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != d_model_)
            NN_FAIL("AttentionBase forward: input shape mismatch");
        const std::size_t total_seq = input.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            NN_FAIL("AttentionBase forward: cols not divisible by seq_len");
        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        NN_TRY(q_res, w_q_.forward(input));
        NN_TRY(k_res, w_k_.forward(input));
        NN_TRY(v_res, w_v_.forward(input));

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        //    使 batched_matmul 能按 batch*H 切分行块
        //    局部 Q/K/V 承载 forward 计算；仅在非 checkpoint 模式下写入成员缓存。
        const std::size_t H_dk = num_heads_ * d_k_;
        Tensor Q, K, V;  // (batch*H*d_k, seq) rearranged
        if (batch > 1)
        {
            NN_TRY(qr, engine.rearrange_3d(*q_res, H_dk, batch, seq, false));
            Q = std::move(*qr);
            NN_TRY(kr, engine.rearrange_3d(*k_res, H_dk, batch, seq, false));
            K = std::move(*kr);
            NN_TRY(vr, engine.rearrange_3d(*v_res, H_dk, batch, seq, false));
            V = std::move(*vr);
        }
        else
        {
            // batch=1: rearrange 是恒等拷贝，跳过
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // 2.5 位置编码：对 Q/K 施加旋转位置编码（rearrange 后列=position，
        //     每 d_k 行一个头，cos/sin 短表按 RowMod 平铺）。
        //     非 RoPE 编码在注意力侧是恒等策略 → 这里无标志位判断。
        {
            NN_TRY(qe, pos_->apply_qk(engine, Q, seq, /*backward=*/false));
            NN_TRY(ke, pos_->apply_qk(engine, K, seq, /*backward=*/false));
        }
        // 2.6 scale（1/sqrt(d_k)）折进 Q（Q *= scale）：注意力表达式不含
        // scale 常量 → 结构与 d_k 无关（不同 d_k 共享融合 shader，闭合世界
        // key 稳定）。backward 的 grad_Q 相应补乘 scale。
        // 写法用 dsl::compute_into（零分配原地写）；dsl::compute 是分配版
        // （另分配一整块缓冲），此处不适用。
        // scale_ 用 rparam 承载 → 值不进 expr_spec_key（融合二期教训 1）。
        {
            auto qs = dsl::compute_into(engine,
                dsl::leaf(Q) * dsl::rparam(scale_), Q);
            NN_TRY_CHECK(qs);
        }

        // ── 注意力主体：单 fold 路径 ────────────────────────────────────
        const std::size_t BH = batch * num_heads_;
        //   分数输入准备：位置偏置表（ALiBi 斜率；非 ALiBi = no-op）
        NN_TRY(pb, pos_->prepare_score_bias(engine, batch, seq));
        //   掩码输入张量准备（掩码策略自己构建 doc_col/doc_ids）
        NN_TRY(pm, mask_->prepare(engine, batch, seq));
        Tensor concat_out;  // (batch*H*d_k, seq)——fold 输出 O_t 经转置/重排得到
        // V 需 (BH*seq, d_k) 布局：V_t 构建须在 fold 求值之前完成（fold 直接消费 V_t）
        NN_TRY(V_T_full, engine.transpose(V));
        NN_TRY(V_t, engine.rearrange_3d(*V_T_full, seq, BH, d_k_, false));
        // ── 单 fold 表达式：S 不物化、online 单遍 ──────────────────────
        //   无 (BH·seq, seq) 中间张量流量；掩码在 fold body 内逐块生效，
        //   tri_skip 把被屏蔽块钳成空转（被跳过的恰是 -inf/0 恒等项 →
        //   与全量计算逐位一致，见 FoldSpec::tri_skip 注释）。
        //   inputs 顺序 = make_fold_attn_o 的 views 顺序：Q,K,V_t,[slope],[dc],[ids]
        //   即**先位置偏置输入、后掩码输入**（两条链正交，各自只追加自己的）。
        std::vector<Tensor> fold_in{Q, K, *V_t};
        pos_->append_score_bias_inputs(fold_in);
        mask_->append_fold_inputs(fold_in);
        const nn::ExprSpec fold_spec = nn::expr::make_fold_attn_o(
            seq, d_k_, BH, mask_->mask_kind(), pos_->has_score_bias());
        NN_TRY(fv, nn::validate_expr_spec(fold_spec, fold_in.size()));
        // scale 已折进 Q（见上 2.6；运行时值不进 expr_spec_key）——fold 的 mm 段直接消费
        NN_TRY(O_t_r, engine.eval_expr(fold_spec, fold_in, BH * seq, d_k_, p_.compute));
        Tensor O_t = std::move(*O_t_r);
        // O_t: (BH*seq, d_k) → 按 batch 转置回 (BH*d_k, seq) 供后续 rearrange：
        //   transpose → (d_k, BH*seq) → rearrange_3d(d_k, BH, seq) → (BH*d_k, seq)
        NN_TRY(O_T_full, engine.transpose(O_t));
        NN_TRY(co, engine.rearrange_3d(*O_T_full, d_k_, BH, seq, false));
        concat_out = std::move(*co);
        if (!checkpoint_mode_)
        {
            Q_cache_ = std::move(Q);
            K_cache_ = std::move(K);
            V_cache_ = std::move(V);
            // m/l 不作缓存（backward 内 softmax 归一化，见 recompute_W_ 注释）
        }
        // 掩码恒在 fold body 内生效（mask_->mask_kind() 选掩码种类），绝不物化；
        // 位置偏置作为正交的第二维一并交给 fold 构造（pos_->has_score_bias()）

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            NN_TRY(cb, engine.rearrange_3d(concat_out, H_dk, batch, seq, true));
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(concat_out);
        }

        // 9. SubLN（子层内部归一化；关 = 直通，逐位不变）→ 输出投影
        Tensor attn_out = std::move(concat);
        if (attn_sub_norm_)
        {
            NN_TRY(sn, attn_sub_norm_->forward(attn_out));
            attn_out = std::move(*sn);
        }
        return w_o_.forward(attn_out);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // 推断 batch/seq（与 forward 一致）
        const std::size_t total_seq = grad_output.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        // 1. SubLN 反向（关 = 直通）→ 输出投影反向 → grad_concat: (H*d_k, batch*seq)
        const Tensor* p_grad_o = &grad_output;
        Tensor grad_prenorm;
        if (attn_sub_norm_)
        {
            NN_TRY(gsn, attn_sub_norm_->backward(grad_output));
            grad_prenorm = std::move(*gsn);
            p_grad_o = &grad_prenorm;
        }
        NN_TRY(gc, w_o_.backward(*p_grad_o));
        nn_dbg_scan("attn.gc", engine, *gc);

        // 2. rearrange grad_concat → (batch*H*d_k, seq)
        Tensor grad_concat_re;
        if (batch > 1)
        {
            NN_TRY(gcr, engine.rearrange_3d(*gc, H_dk, batch, seq, false));
            grad_concat_re = std::move(*gcr);
        }
        else
        {
            grad_concat_re = std::move(*gc);
        }

        // 3-7. 注意力反向：R/X 路径
        Tensor grad_Q_re, grad_K_re, grad_V_re;  // 均 (batch*H*d_k, seq)
        {
            // P = grad_A = batched_matmul(grad_concat^T, V, BH, true, false)
            // forward: O = V × A^T → grad_A = grad_O^T × V（后续 R/X 分解的 P 输入）
            // dsl::matmul(batch)（结构经 scan 的 CSA/MHA backward dry-run 登记）；
            // 输出形状：transA=true → rows=BH*A.cols()；transB=false → cols=B.cols()
            auto grad_A = dsl::compute(engine,
                dsl::matmul(grad_concat_re, V_cache_, true, false, BH),
                BH * grad_concat_re.cols(), V_cache_.cols(), p_.compute);
            NN_TRY_CHECK(grad_A);
            nn_dbg_scan("attn.grad_A", engine, *grad_A);
            // G = grad_concat_re^T 按 batch 转置 → (BH*seq, d_k)，
            // 供 grad_V[j][k] = Σ_i W·G[i][k]（同 V 的布局转换）
            NN_TRY(G_T_full, engine.transpose(grad_concat_re));
            NN_TRY(G, engine.rearrange_3d(*G_T_full, seq, BH, d_k_, false));
            // ── IR 路径（R/X 表达式 + batched matmul）──
            //   从 Q/K 重算 W（W 与 m/l 均不缓存——recompute_W_ 内部
            //   softmax 单表达式归一化）；
            //   FLOPs ×1.5-2（QK^T 重算），训练可接受。
            NN_TRY(W_re, recompute_W_(engine, Q_cache_, K_cache_, BH, seq));
            nn_dbg_scan("attn.W_re", engine, *W_re);
            //   R  = row_sum(W·P)                     → (BH*seq, 1)
            //   X  = scale·W·(P − R)                  → (BH*seq, seq)（物化）
            //   grad_Q = K × X^T；grad_K = Q × X；grad_V = W^T × G
            auto R = dsl::compute_reduce(engine,
                dsl::row_reduce_sum(dsl::leaf(*W_re) * dsl::leaf(*grad_A)),
                BH * seq, seq);
            NN_TRY_CHECK(R);
            nn_dbg_scan("attn.R", engine, *R);
            auto X = dsl::compute(engine,
                dsl::leaf(*W_re)
                    * (dsl::leaf(*grad_A) - dsl::row_broadcast(*R)),
                BH * seq, seq, p_.compute);
            NN_TRY_CHECK(X);
            nn_dbg_scan("attn.X", engine, *X);
            // grad_Q = K × X^T（K_b (d_k,seq)，X_b (seq,seq) 按 X^T 使用）
            // dsl::matmul(batch)（transA=F,transB=T → rows=K.rows(), cols=X.rows()/BH）
            auto gq = dsl::compute(engine,
                dsl::matmul(K_cache_, *X, false, true, BH),
                K_cache_.rows(), X->rows() / BH, p_.compute);
            NN_TRY_CHECK(gq);
            nn_dbg_scan("attn.gq(pre-scale)", engine, *gq);
            // grad_Q 补乘 scale（forward 把 scale 折进了 Q）：compute_into 原地
            // （同 forward 2.6 的写法；rparam 承载 scale 值不进 key）
            {
                auto gqs = dsl::compute_into(engine,
                    dsl::leaf(*gq) * dsl::rparam(scale_), *gq);
                NN_TRY_CHECK(gqs);
            }
            // grad_K = Q × X（dsl::matmul(batch)：false,false → rows=Q.rows(), cols=X.cols()）
            auto gk = dsl::compute(engine,
                dsl::matmul(Q_cache_, *X, false, false, BH),
                Q_cache_.rows(), X->cols(), p_.compute);
            NN_TRY_CHECK(gk);
            nn_dbg_scan("attn.gk", engine, *gk);
            // grad_V = W^T × G（W_b (seq,seq) 按 W^T 使用，G_b (seq,d_k)）→ (BH*seq, d_k)
            // dsl::matmul(batch)：transA=T,transB=F → rows=BH*W.cols(), cols=G.cols()
            auto gv_t = dsl::compute(engine,
                dsl::matmul(*W_re, *G, true, false, BH),
                BH * W_re->cols(), G->cols(), p_.compute);
            NN_TRY_CHECK(gv_t);
            // grad_V 转置回 (BH*d_k, seq)（与 forward 的 V_t→V 逆变换一致）
            NN_TRY(gv_T, engine.transpose(*gv_t));
            NN_TRY(gv_re, engine.rearrange_3d(*gv_T, d_k_, BH, seq, false));
            nn_dbg_scan("attn.gv_re", engine, *gv_re);
            grad_Q_re = std::move(*gq);
            grad_K_re = std::move(*gk);
            grad_V_re = std::move(*gv_re);
        }

        // 7.5 位置编码 backward：对 Q/K 梯度施加反角旋转
        //     （RoPE 的旋转矩阵正交，逆 = 转置 = 反角：grad*cos − rot(grad)*sin；
        //      非 RoPE 策略 = 恒等，无标志位判断）
        {
            NN_TRY(gq2, pos_->apply_qk(engine, grad_Q_re, seq, /*backward=*/true));
            NN_TRY(gk2, pos_->apply_qk(engine, grad_K_re, seq, /*backward=*/true));
        }

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            NN_TRY(gq, engine.rearrange_3d(grad_Q_re, H_dk, batch, seq, true));
            grad_Q = std::move(*gq);
            NN_TRY(gk, engine.rearrange_3d(grad_K_re, H_dk, batch, seq, true));
            grad_K = std::move(*gk);
            NN_TRY(gv, engine.rearrange_3d(grad_V_re, H_dk, batch, seq, true));
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(grad_Q_re);
            grad_K = std::move(grad_K_re);
            grad_V = std::move(grad_V_re);
        }

        // 9. 投影层反向 + 累加输入梯度
        NN_TRY(giq, w_q_.backward(grad_Q));
        nn_dbg_scan("attn.giq", engine, *giq);
        NN_TRY(gik, w_k_.backward(grad_K));
        nn_dbg_scan("attn.gik", engine, *gik);
        NN_TRY(giv, w_v_.backward(grad_V));
        nn_dbg_scan("attn.giv", engine, *giv);

        // grad_input = grad_Q + grad_K + grad_V：三路累加**原地**融合为单趟
        // （dsl::compute_into 目标传递，不额外分配）。结合顺序固定为
        // (giq + gik) + giv —— 求和顺序确定，结果可复现。
        auto acc = dsl::compute_into(engine,
            dsl::leaf(*giq) + dsl::leaf(*gik) + dsl::leaf(*giv), *giq);
        NN_TRY_CHECK(acc);
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
    // 因果掩码无需施加：新 token 天然只能看到自身及之前的位置（cache 中只有前文）。
    // ALiBi 等分数偏置由掩码策略转发给位置编码器（apply_bias_step）。
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
            NN_FAIL("AttentionBase forward_step: x_new must be (d_model, 1)");
        // 1. Q/K/V 投影 → (H*d_k, 1)
        NN_TRY(q_res, w_q_.forward(x_new));
        NN_TRY(k_new, w_k_.forward(x_new));
        NN_TRY(v_new, w_v_.forward(x_new));

        // 1.5 位置编码：对 Q/K 施加当前位置 (cur_len) 的旋转后写入 KV cache
        //     （cache 中的历史 K 已在各自 step 旋转过，相对位置自然成立；
        //      非 RoPE 策略 = 恒等）
        {
            NN_TRY(qr, pos_->apply_qk_step(engine, *q_res, cur_len, /*backward=*/false));
            NN_TRY(kr, pos_->apply_qk_step(engine, *k_new, cur_len, /*backward=*/false));
        }

        // 2. transpose → (1, H*d_k)，匹配 cache 的行布局
        NN_TRY(k_new_T, engine.transpose(*k_new));
        NN_TRY(v_new_T, engine.transpose(*v_new));

        // 3. 追加到 KV cache（就地写入第 cur_len 行）
        NN_TRY(r1, engine.insert_rows(k_cache, cur_len, *k_new_T));
        NN_TRY(r2, engine.insert_rows(v_cache, cur_len, *v_new_T));

        // 4. 取有效区间 [0, new_len) 并 transpose 为 (H*d_k, new_len) 布局
        //    batched_matmul 要求 rows 能被 batch 整除:
        //    (new_len, H*d_k) 的 rows=new_len 不保证整除 num_heads
        //    transpose 后 (H*d_k, new_len) 的 rows=H*d_k=d_model 必然整除
        const std::size_t new_len = cur_len + 1;
        NN_TRY(k_valid, engine.slice_rows(k_cache, 0, new_len));
        NN_TRY(v_valid, engine.slice_rows(v_cache, 0, new_len));
        auto K_T = engine.transpose(*k_valid);   // (H*d_k, new_len)
        NN_TRY_CHECK(K_T);
        auto V_T = engine.transpose(*v_valid);   // (H*d_k, new_len)
        NN_TRY_CHECK(V_T);

        // 5. scores = batched_matmul(Q, K_T, H, transA=T, transB=F)
        //    Q: (H*d_k, 1) — 每头 (d_k, 1)，转置后 (1, d_k)，M=1
        //    K_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transB=F，N=new_len
        //    每头: (1, d_k) × (d_k, new_len) = (1, new_len)
        //    堆叠: (H, new_len)
        //    scale (1/sqrt(d_k)) 经 rparam 尾链乘在 matmul 结果上（不进 expr_spec_key）
        auto scores = dsl::compute(engine,
            dsl::matmul(*q_res, *K_T, true, false, num_heads_) * dsl::rparam(scale_),
            num_heads_ * q_res->cols(), K_T->cols(), p_.compute);
        NN_TRY_CHECK(scores);

        // 6. 施加增量推理的位置偏置（ALiBi；非 ALiBi = 恒等）。
        //    因果无需施加：KV cache 里只有前文，新 token 天然看不到未来。
        NN_TRY(masked, pos_->apply_bias_step(engine, std::move(*scores), cur_len, p_));

        // 7. softmax（行级归一化，每头独立）
        NN_TRY(attn, softmax_.forward(*masked));

        // 8. attn_out = batched_matmul(V_T, attn, H, transA=F, transB=T)
        //    V_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transA=F，M=d_k
        //    attn: (H, new_len) — 每头 (1, new_len)，转置后 (new_len, 1)，N=1
        //    每头: (d_k, new_len) × (new_len, 1) = (d_k, 1)
        //    堆叠: (H*d_k, 1)
        // dsl::matmul(batch)：纯 {0,1} 结构（attention backward 已登记同 key）
        auto attn_out = dsl::compute(engine,
            dsl::matmul(*V_T, *attn, false, true, num_heads_),
            V_T->rows(), attn->rows() / num_heads_, p_.compute);
        NN_TRY_CHECK(attn_out);

        // 9. SubLN（子层内部归一化；关 = 直通）→ 输出投影 → (d_model, 1)
        Tensor ao = std::move(*attn_out);
        if (attn_sub_norm_)
        {
            NN_TRY(sn, attn_sub_norm_->forward(ao));
            ao = std::move(*sn);
        }
        return w_o_.forward(ao);
    }

protected:
    // ── 从 Q/K 重算 S 与 W（backward 用）───────────────────────────────
    //   W 与 m/l 均不作缓存：softmax 单表达式在 kernel 内部归一化，m/l 不外溢。
    //   S 由两条**正交**的链依次得到（各一次虚调用，无变体判断链）：
    //     1) S = mask(Q·Kᵀ)              —— 掩码策略（因果 / 文档）
    //     2) S += bias(S)（原地）         —— 位置编码策略（ALiBi；其余 no-op）
    //     3) W = softmax(S)（归约快路径）
    // 返回 (BH*seq, seq) 的 softmax 归一化权重；attention FLOPs ×1.5–2
    // （QK^T 在 backward 重算），训练可接受。
    [[nodiscard]] Result<Tensor> recompute_W_(
        ComputeEngine& engine,
        const Tensor& Q, const Tensor& K,
        std::size_t BH, std::size_t seq) const
    {
        NN_TRY(sres, mask_->masked_scores(engine, Q, K, BH, seq, p_));
        Tensor S = std::move(*sres);
        // 位置偏置：原地叠加（零额外分配）；非 ALiBi 策略是 no-op
        NN_TRY_CHECK(pos_->apply_score_bias(engine, S, BH, seq));
        // W = softmax(S)：单表达式归约快路径——与 Softmax::forward
        // 同构：ReduceRef 直接参与算术按行广播；m/l 在 kernel 内部归一化，
        // 不作为输入/缓存存在。
        return dsl::compute(engine,
            dsl::exp(dsl::leaf(S) - dsl::row_reduce_max(S))
            / dsl::row_reduce_sum(
                dsl::exp(dsl::leaf(S) - dsl::row_reduce_max(S))),
            BH * seq, seq, p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MultiHeadAttention — 无掩码双向多头注意力
//
// 掩码语义 = PlainScoreMask（基类默认工厂即可，无需再塞标志位）；
// 位置编码默认恒等（TransformerEncoderLayer 的位置信息在嵌入侧注入）。
// ══════════════════════════════════════════════════════════════════════════
class MultiHeadAttention final : public AttentionBase
{
public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "MultiHeadAttention"; }

    MultiHeadAttention(std::size_t d_model, std::size_t num_heads,
                       std::size_t seq_len = 0)
        : AttentionBase(d_model, num_heads, seq_len) {}
};

// ══════════════════════════════════════════════════════════════════════════
// CausalSelfAttention — 因果自注意力
//
// 两个**互相独立**的策略对象，各有自己的配置期工厂：
//   · 掩码（本类自己）：Plain→Causal 由本类选定，文档掩码按运行期数据开关
//       · 无文档        → CausalScoreMask
//       · 文档块对角    → CausalDocScoreMask
//   · 位置编码（由本层持有，见 compute_position_encoding.hpp）：
//       · RoPE   → pos_->apply_qk() 旋转 Q/K
//       · ALiBi  → pos_->apply_score_bias() 给分数加线性偏置
//  掩码工厂**不查询**位置编码（旧版 `pos_->has_score_bias()` 分支已删除）：
//  掩码只管"键能不能被看到"，位置偏置只管"位置带来的偏移"，两者正交。
//  forward/backward 里没有 use_alibi_/use_doc_ids_ 之类的判断。
//
// 文档感知（set_doc_ids）：每 step 的文档 id 是**运行期数据**，其"有/无"在
//   set_doc_ids 这一**配置调用**里裁定（模式不变时零开销），而不是每题一次
//   forward 再判一遍。空 span = 关闭文档掩码（退化为纯因果）。
//
// 算法差异（相对于 AttentionBase）：无——全部差异由两个策略对象承载。
//   seq_len=0 表示单样本模式（cols 即 seq）。
//
// 注意：max_len 参数保留仅为向后兼容签名，当前实现不使用。
// ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention final : public AttentionBase
{
public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "CausalSelfAttention"; }

private:
    bool doc_enabled_ = false;

    [[nodiscard]] std::unique_ptr<AttnScoreMask> make_score_mask_() const override
    {
        // 配置期工厂：这是**唯一**决定掩码种类的地方（只依赖文档掩码的开关，
        // 与位置编码无关）。forward/backward 里没有任何变体判断。
        if (doc_enabled_)
            return std::make_unique<CausalDocScoreMask>(num_heads_);
        return std::make_unique<CausalScoreMask>();
    }

public:
    CausalSelfAttention(std::size_t d_model, std::size_t num_heads,
                        std::size_t /*max_len*/ = 1024,
                        std::size_t seq_len = 0,
                        PosEncodingType pos_enc = PosEncodingType::Learned,
                        bool subln = false,
                        NormType subln_norm_type = NormType::LayerNorm)
        : AttentionBase(d_model, num_heads, seq_len, subln, subln_norm_type)
    {
        // ① 位置编码：注意力侧**由本层自持**（RoPE 的 cos/sin 表随层构建；
        //    Learned/Sinusoidal 在注意力侧是恒等 → 交回模型侧施加）。
        install_position_encoder(
            make_attention_position_encoder(pos_enc, d_k_, num_heads_));
        // ② 掩码：本类的因果语义（与位置编码正交，故独立安装）
        install_score_mask_();
    }

    // 文档感知：ids 非空 = 开启块对角文档掩码。空 span = 关闭（纯因果）。
    // 配置期调用（每 step 至多一次）：模式变化时才换装掩码策略。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        const bool want = !ids.empty();
        if (want != doc_enabled_)
        {
            doc_enabled_ = want;
            install_score_mask_();   // 配置期换装：掩码策略随模式切换
        }
        score_mask().set_doc_ids(ids);
    }
};

} // namespace nn
