#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_fold.hpp — 分块状态归约（FoldSpec）样例构造：P-C1 地基
//
//  这些构造函数是 scan_exprs（AOT 收集 dry-run）、fused_gpu_test（GPU 对拍）
//  与注意力 Layer（AttentionBase::forward 直调 make_fold_attn_o，P-C2-7 起
//  单 fold 路径）的**共享唯一来源**——各方必须构造出结构完全一致的 spec
//  （key 一致），否则闭合世界查表未命中。rowmax/rowsum/softmax_denom 三个
//  P-C1 样例不被 Layer 使用，保留为状态语义回归锚点。
//
//  共同形状契约：输入 (rows, K)，输出 (rows, 1)；fold.k = K（形状参数）。
//  三个样例的语义增量：
//    1. rowmax  —— 单状态 max 累加（对拍 engine.row_reduce_max）
//    2. rowsum  —— 单状态 sum 累加（对拍 engine.row_reduce_sum）
//    3. softmax_denom —— 双状态 online rescale（m 跨块更新 + l = l*α +
//       Σexp(x−m')）：验证状态读写序、拷贝旧状态技巧、body 链内状态广播、
//       两个块归约指令——P-C1 状态语义的全覆盖锚点。
// ═══════════════════════════════════════════════════════════════════════════

#include <limits>

#include "expr_spec.hpp"

namespace nn::expr
{

// 末指令 = 状态自复制（dst 自身 Add 常量 0），把"输出 = 末指令 dst"落到状态上
[[nodiscard]] inline ExprInstr fold_emit_state(std::uint8_t state_reg,
                                               std::uint8_t const_zero_slot)
{
    ExprInstr ins;
    ins.op  = static_cast<uint8_t>(ExprOp::Add);
    ins.dst = state_reg;
    ins.a   = reg(state_reg);
    ins.b   = cst(const_zero_slot);
    return ins;
}

// ── 1) fold rowmax：m = max_k x(row, k) ────────────────────────────────────
// body:  [0] blk = RowMax(x)          （沿块内 kb 归约）
//        [1] m   = max(m, Reduce(blk))（写回状态 0）
// finalize: r0 = r0 + c0（状态自复制 → 输出）
[[nodiscard]] inline ExprSpec make_fold_rowmax(std::uint32_t k)
{
    ExprSpec s;
    FoldSpec f;
    f.k          = k;
    f.num_state  = 1;
    f.inits      = { -std::numeric_limits<Scalar>::infinity() };
    ExprInstr b0;                    // blk = RowMax(input 0)
    b0.op  = static_cast<uint8_t>(ExprOp::RowMax);
    b0.dst = 1;
    b0.a   = input(0);
    b0.b   = {};
    ExprInstr b1;                    // m = max(m, Reduce(blk))
    b1.op  = static_cast<uint8_t>(ExprOp::Max);
    b1.dst = 0;
    b1.a   = reg(0);
    b1.b   = reduce(1);
    f.body     = { b0, b1 };
    f.finalize = { fold_emit_state(0, 0) };
    s.fold     = std::move(f);
    s.consts   = { Scalar{0} };
    s.num_regs = 2;                  // 状态 0 + 临时 1
    s.views    = { linear() };
    return s;
}

// ── 2) fold rowsum：s = Σ_k x(row, k) ──────────────────────────────────────
[[nodiscard]] inline ExprSpec make_fold_rowsum(std::uint32_t k)
{
    ExprSpec s;
    FoldSpec f;
    f.k          = k;
    f.num_state  = 1;
    f.inits      = { Scalar{0} };
    ExprInstr b0;
    b0.op  = static_cast<uint8_t>(ExprOp::RowSum);
    b0.dst = 1;
    b0.a   = input(0);
    ExprInstr b1;
    b1.op  = static_cast<uint8_t>(ExprOp::Add);
    b1.dst = 0;
    b1.a   = reg(0);
    b1.b   = reduce(1);
    f.body     = { b0, b1 };
    f.finalize = { fold_emit_state(0, 0) };
    s.fold     = std::move(f);
    s.consts   = { Scalar{0} };
    s.num_regs = 2;
    s.views    = { linear() };
    return s;
}

// ── 3) fold softmax 分母（online 双状态）──────────────────────────────────
// 状态：0 = m（行 max，初值 -inf）、1 = l（移位和，初值 0）
// body（块内网格 x(row, kb)），指令序即状态更新序：
//   [0]  dst2 = m_old 拷贝（m 更新前留存旧值，供 α 与回看）
//   [1]  dst3 = RowMax(x)                       —— 块内行 max
//   [2]  m    = max(dst2, Reduce(dst3))         —— 状态 0 跨块更新
//   [3]  dst4 = dst2 − m                        —— m_old − m'
//   [4]  dst5 = exp(dst4)                       —— rescale 因子 α
//   [5]  dst7 = x − m'（广播）
//   [6]  dst8 = exp(dst7)
//   [7]  dst6 = RowSum(dst8)                    —— 块内 Σexp(x−m')
//   [8]  dst9 = l · α
//   [9]  l    = dst9 + Reduce(dst6)             —— 状态 1 跨块更新
// finalize: 输出 l（状态自复制）
// 参考语义：l_final = Σ_k exp(x(row,k) − max_k x)（浮点结合序容差内等价）。
[[nodiscard]] inline ExprSpec make_fold_softmax_denom(std::uint32_t k)
{
    const auto ins = [](ExprOp op, std::uint8_t dst, ExprOperand a,
                        ExprOperand b = {}) {
        ExprInstr i;
        i.op  = static_cast<uint8_t>(op);
        i.dst = dst;
        i.a   = a;
        i.b   = b;
        return i;
    };
    ExprSpec s;
    FoldSpec f;
    f.k         = k;
    f.num_state = 2;
    f.inits     = { -std::numeric_limits<Scalar>::infinity(), Scalar{0} };
    f.body = {
        ins(ExprOp::Add,    2, reg(0), cst(0)),   // m_old 拷贝
        ins(ExprOp::RowMax, 3, input(0)),         // 块内 blk_m
        ins(ExprOp::Max,    0, reg(2), reduce(3)),// m' ← 状态更新
        ins(ExprOp::Sub,    4, reg(2), reg(0)),   // m_old − m'
        ins(ExprOp::Exp,    5, reg(4)),           // α
        ins(ExprOp::Sub,    7, input(0), reg(0)), // x − m'
        ins(ExprOp::Exp,    8, reg(7)),           // exp(x − m')
        ins(ExprOp::RowSum, 6, reg(8)),           // 块内 blk_l
        ins(ExprOp::Mul,    9, reg(1), reg(5)),   // l · α
        ins(ExprOp::Add,    1, reg(9), reduce(6)),// l' ← 状态更新
    };
    f.finalize = { fold_emit_state(1, 0) };
    s.fold     = std::move(f);
    s.consts   = { Scalar{0} };
    s.num_regs = 10;                             // 状态 0..1 + 临时 2..9
    s.views    = { linear() };
    return s;
}

} // namespace nn::expr

// ═══════════════════════════════════════════════════════════════════════════
//  P-C2 双域旗舰样例：fold attention 输出 O（S 不物化，online 单遍）
//
//  数学：O = softmax_masked(Q·Kᵀ) · V_t
//    键域每块：S_tile = mm(Q,K)（内层 dk 收缩）→ [掩码] →
//      m' = max(m, 块 max)；α = e^{m−m'}；e = e^{S−m'}；
//      l = l·α + Σ_块 e；（块尾 vecacc：O *= α；O += Σ_块 e·V_t）
//    向量域：out[d] = O[d] / l
//  布局：Q/K (bh·dk, seq) 行主序；V_t (bh·seq, dk)；输出 (bh·seq, dk)。
//  掩码语义与既有 masked_causal_ / masked_alibi_ 对齐（Row/Col 按 batched
//  网格：Row = 行%(rows/batch) = 查询位置 i、Col = 全局键位置 j）。
//  scan_exprs dry-run、fused_gpu/expr_cpu 对拍与 AttentionBase::forward
//  共用本构造 → key 一致。
// ═══════════════════════════════════════════════════════════════════════════
namespace nn::expr
{

enum class FoldAttnMask
{
    Plain,      // 无掩码（仅测试/数学对拍用——Layer 恒因果系）
    Causal,     // 因果（j>i → -inf）
    Alibi,      // 因果 + ALiBi 线性偏置（slopes 经 BatchMod 视图）
    Doc,        // 因果 + 文档块对角（doc_col 经 RowBroadcast、doc_ids 经 BatchCol）
    AlibiDoc,   // 因果 + ALiBi + 文档
};

[[nodiscard]] inline ExprSpec make_fold_attn_o(std::uint32_t seq, std::uint32_t dk,
                                               std::uint32_t bh, FoldAttnMask mask)
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
    // 输入/视图槽位（与 Layer 组包顺序约定一致）：
    //   0=Q 1=K 2=V_t；[slopes (1,bh)→BatchMod]；[doc_col (BH*seq,1)→RowBroadcast]；
    //   [doc_ids (1,bh*seq)→BatchCol(seq)]
    const bool has_slope = (mask == FoldAttnMask::Alibi ||
                            mask == FoldAttnMask::AlibiDoc);
    const bool has_doc   = (mask == FoldAttnMask::Doc ||
                            mask == FoldAttnMask::AlibiDoc);
    const bool has_causal = (mask != FoldAttnMask::Plain);
    f.causal_skip = has_causal;   // 生成器据此钳 valid（整块/边界跳过 -inf 区）
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
        //   （与既有 masked_doc_ 同构：两层 select 合并因果后统一 -inf）
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

} // namespace nn::expr
