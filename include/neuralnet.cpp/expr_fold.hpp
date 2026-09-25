#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_fold.hpp — 分块状态归约（FoldSpec）通用样例构造：P-C1 地基
//
//  本头只承载**与注意力无关**的通用 fold 样例，是 scan_exprs（AOT 收集
//  dry-run）与 fused_gpu_test / expr_fold_test 对拍的**共享唯一来源**——
//  各方必须构造出结构完全一致的 spec（key 一致），否则闭合世界查表未命中。
//  rowmax/rowsum/softmax_denom 三个 P-C1 样例不被 Layer 使用，保留为状态
//  语义回归锚点。
//
//  注意力的 fold 构造（make_fold_attn_o / FoldAttnMask）属于 Layer 侧的
//  表达式文本（与 AOT 收集原则"表达式文本只出现在 Layer"一致），定义在
//  compute_layer_attention.hpp，本头不包含任何注意力专属语义。
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
