#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_opt.hpp — ExprSpec 规范化与优化 pass（IR-A + IR-B）
//
//  对应文档 `docs/11-ir-optimization.md`：
//    - IR-A：canonicalize_expr_spec（DCE / 常量折叠 / 代数化简 / 稳定重编号）
//    - IR-B：CSE（公共子表达式消除）+ 寄存器分配（liveness 线性扫描）
//
//  canonical IR = canonicalize_expr_spec(spec) 的输出，是 key 计算、去重、
//  shader 合成的唯一依据。约定：
//      expr_spec_key(spec) ≡ expr_spec_key(canonicalize_expr_spec(spec))
//  （scan 构建期与 runtime 运行期都必须先 canonicalize 再算 key，两端一致。）
//
//  关键不变量（架构红线内，参考 09/11 文档）：
//    1. **不改变 views/inputs**（顺序、内容）——只优化 instrs/consts/num_regs。
//       运行时 dispatch 的输入绑定顺序不变（融合 shader binding 布局不变），
//       gpu_inputs 按原 inputs 传递即可。
//    2. **不改变求值语义**（保守常量折叠 + 代数化简；不折叠超越函数
//       exp/log/sqrt/rsqrt/tanh，避免 CPU libm 与 GPU 实现的 ulp 差异）。
//    3. **确定性铁律**：全部 pass 遍历顺序固定（指令序从前到后、寄存器按号、
//       常量按出现序），无依赖未定义求值顺序的逻辑（C++ 实参求值顺序坑）。
//    4. 输出指令（最后一条）恒为真实寄存器指令（glsl_gen 输出 `r<last_dst>`），
//       故输出指令不做"整体折叠为常量/替换为 Input"的化简。
//    5. 归约指令不折叠（结果依赖数据）；CSE/寄存器分配保持归约语义
//       （Reduce 操作数仍只引用归约指令 dst，由 dst 重映射保证）。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "expr_spec.hpp"

namespace nn
{

// ── 常量折叠求值（仅 IEEE 754 基本运算；不折叠超越函数/归约）────────────
// 返回 false 表示该算子不参与折叠（调用方保留原指令）。
[[nodiscard]] inline bool expr_eval_const_fold(ExprOp op, Scalar a, Scalar b,
                                               Scalar c, Scalar& out_v)
{
    switch (op)
    {
    case ExprOp::Neg:    out_v = -a;                 return true;
    case ExprOp::Abs:    out_v = std::fabs(a);       return true;
    case ExprOp::Add:    out_v = a + b;              return true;
    case ExprOp::Sub:    out_v = a - b;              return true;
    case ExprOp::Mul:    out_v = a * b;              return true;
    case ExprOp::Div:    out_v = a / b;              return true;
    case ExprOp::Max:    out_v = std::max(a, b);     return true;
    case ExprOp::Min:    out_v = std::min(a, b);     return true;
    case ExprOp::Lt:     out_v = (a <  b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Le:     out_v = (a <= b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Gt:     out_v = (a >  b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Ge:     out_v = (a >= b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Eq:     out_v = (a == b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Ne:     out_v = (a != b) ? Scalar{1} : Scalar{0}; return true;
    case ExprOp::Select: out_v = (a != Scalar{0}) ? b : c;         return true;
    default:             return false;  // 超越函数 / 归约不折叠
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  代数化简辅助（IR-A）
//
//  尝试把单条指令化简为更简单的等价操作数；返回 nullopt 表示不化简。
//  保守策略（不改变求值语义）：
//    neg(neg(x))→x、x+0→x、x-0→x、x*1→x、x*0→0、x/1→x、
//    max(x,x)→x、min(x,x)→x、select(常量,b,c)→b/c、select(cond,x,x)→x
//  注意：0-x→neg(x) 需生成新指令，不在此处理（调用方负责）。
//  需要向常量池写入时（x*0→0）经 consts 就地追加（确定性：保持出现序）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::optional<ExprOperand> expr_try_algebraic_simplify(
    ExprOp op, const ExprOperand& a, const ExprOperand& b, const ExprOperand& c,
    std::vector<Scalar>& consts,
    const std::vector<ExprOp>& def_op, const std::vector<ExprOperand>& def_a)
{
    const auto is_const = [&](const ExprOperand& o, Scalar v)
    {
        return o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const) &&
               consts[o.idx] == v;
    };
    const auto add_const_op = [&](Scalar v) -> ExprOperand
    {
        for (std::size_t j = 0; j < consts.size(); ++j)
            if (consts[j] == v)
                return {static_cast<std::uint8_t>(ExprOperandKind::Const),
                        static_cast<std::uint8_t>(j)};
        const std::uint8_t idx = static_cast<std::uint8_t>(consts.size());
        consts.push_back(v);
        return {static_cast<std::uint8_t>(ExprOperandKind::Const), idx};
    };
    const auto is_reg_like = [](const ExprOperand& o)
    {
        return o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
               o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout);
    };

    if (op == ExprOp::Neg)
    {
        // neg(neg(x)) → x
        if (is_reg_like(a) && a.idx < def_op.size() && def_op[a.idx] == ExprOp::Neg)
            return def_a[a.idx];
    }
    else if (op == ExprOp::Add)
    {
        if (is_const(a, Scalar{0})) return b;
        if (is_const(b, Scalar{0})) return a;
    }
    else if (op == ExprOp::Sub)
    {
        if (is_const(b, Scalar{0})) return a;
        // 0-x → neg(x) 需生成指令，由调用方处理
    }
    else if (op == ExprOp::Mul)
    {
        if (is_const(a, Scalar{1})) return b;
        if (is_const(b, Scalar{1})) return a;
        if (is_const(a, Scalar{0}) || is_const(b, Scalar{0})) return add_const_op(Scalar{0});
    }
    else if (op == ExprOp::Div)
    {
        if (is_const(b, Scalar{1})) return a;
    }
    else if (op == ExprOp::Max || op == ExprOp::Min)
    {
        if (a == b) return a;  // max(x,x)/min(x,x) → x
    }
    else if (op == ExprOp::Select)
    {
        if (a.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
            return (consts[a.idx] != Scalar{0}) ? b : c;
        if (b == c) return b;  // select(cond, x, x) → x
    }
    return std::nullopt;
}

// ═══════════════════════════════════════════════════════════════════════════
//  IR pass 1：常量池去重 + 常量折叠 + 代数化简（IR-A）
//
//  逐条指令（从前到后）：
//    - 重映射操作数：Const 用去重后索引；Reg/Fanout 若已被化简成更简操作数
//      （replace 表）则替换。
//    - 归约指令：只重映射操作数（结果依赖数据，不折叠/不 alias）。
//    - 非输出指令：若操作数全为常量 → 常量折叠（expr_eval_const_fold）；
//      否则尝试代数化简（expr_try_algebraic_simplify；0-x 单独生成 Neg 指令）。
//    - 输出指令（最后一条）：只重映射操作数，不做任何折叠/替换
//      （保证输出恒为真实寄存器，glsl_gen 输出 `r<last_dst>`）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec fold_constants_and_algebra(const ExprSpec& spec)
{
    ExprSpec out;
    out.views  = spec.views;   // 不变量：views 不变
    out.matmul = spec.matmul;  // 不变量：matmul 段不变（pass 只优化 instrs/consts）

    // 1) 常量池去重（相同值合并，保持出现序）
    std::vector<int> const_remap(spec.consts.size(), -1);
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
    {
        int hit = -1;
        for (std::size_t j = 0; j < out.consts.size(); ++j)
            if (out.consts[j] == spec.consts[i]) { hit = static_cast<int>(j); break; }
        if (hit < 0)
        {
            hit = static_cast<int>(out.consts.size());
            out.consts.push_back(spec.consts[i]);
        }
        const_remap[i] = hit;
    }

    const auto is_const_zero = [&](const ExprOperand& o)
    {
        return o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const) &&
               out.consts[o.idx] == Scalar{0};
    };

    // replace[r]：寄存器 r 化简后等价的操作数（非空则引用处替换）
    std::vector<std::optional<ExprOperand>> replace(spec.num_regs);
    // 定义 r 的指令（化简后）的 op 与 a 操作数（用于 neg(neg(x)) 化简）
    std::vector<ExprOp>   def_op(spec.num_regs, ExprOp::Add);
    std::vector<ExprOperand> def_a(spec.num_regs, ExprOperand{});

    const auto remap_op = [&](const ExprOperand& op) -> ExprOperand
    {
        if (op.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
            return {static_cast<std::uint8_t>(ExprOperandKind::Const),
                    static_cast<std::uint8_t>(const_remap[op.idx])};
        if ((op.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
             op.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout)) &&
            op.idx < spec.num_regs && replace[op.idx])
            return *replace[op.idx];
        return op;
    };

    const auto emit = [&](std::uint8_t opc, std::uint8_t dst,
                          const ExprOperand& a, const ExprOperand& b,
                          const ExprOperand& c) -> void
    {
        ExprInstr ni; ni.op = opc; ni.dst = dst; ni.a = a; ni.b = b; ni.c = c;
        out.instrs.push_back(ni);
        def_op[dst] = static_cast<ExprOp>(opc);
        def_a[dst] = a;
    };

    for (std::size_t idx = 0; idx < spec.instrs.size(); ++idx)
    {
        const ExprInstr& ins = spec.instrs[idx];
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::size_t dst = ins.dst;
        const bool is_output = (idx + 1 == spec.instrs.size());

        // 只重映射实际使用的操作数（未使用的 b/c 是默认哨兵 {0,0}，
        // 不得当 Reg(0) 重编号——记忆库 operator-fusion M1 坑 5）
        const std::size_t nops = expr_instr_num_operands(op);
        const ExprOperand a = remap_op(ins.a);
        const ExprOperand b = (nops >= 2) ? remap_op(ins.b) : ins.b;
        const ExprOperand c = (nops >= 3) ? remap_op(ins.c) : ins.c;

        // 归约指令：只重映射操作数，不折叠 / 不 alias
        if (expr_op_is_reduce(op))
        {
            emit(ins.op, ins.dst, a, b, c);
            continue;
        }
        // 输出指令：只重映射操作数（保证输出恒为真实寄存器）
        if (is_output)
        {
            emit(ins.op, ins.dst, a, b, c);
            continue;
        }

        // ── 常量折叠（操作数全为常量）──────────────────────────────────
        {
            bool all_const = true;
            const std::size_t nops = expr_instr_num_operands(op);
            const ExprOperand* ops[3] = {&a, &b, &c};
            for (std::size_t oi = 0; oi < nops; ++oi)
                if (ops[oi]->kind != static_cast<std::uint8_t>(ExprOperandKind::Const))
                { all_const = false; break; }
            if (all_const)
            {
                Scalar v = Scalar{0};
                if (expr_eval_const_fold(op, out.consts[a.idx], out.consts[b.idx],
                                         out.consts[c.idx], v))
                {
                    // 常量入池（去重）
                    std::optional<ExprOperand> cv;
                    for (std::size_t j = 0; j < out.consts.size(); ++j)
                        if (out.consts[j] == v) { cv = {static_cast<std::uint8_t>(ExprOperandKind::Const),
                                                        static_cast<std::uint8_t>(j)}; break; }
                    if (!cv)
                    {
                        const std::uint8_t ci = static_cast<std::uint8_t>(out.consts.size());
                        out.consts.push_back(v);
                        cv = {static_cast<std::uint8_t>(ExprOperandKind::Const), ci};
                    }
                    replace[dst] = *cv;
                    continue;
                }
            }
        }

        // ── 代数化简 ────────────────────────────────────────────────────
        // 0 - x → neg(x)：生成 Neg 指令（输出指令已提前 continue）
        if (op == ExprOp::Sub && is_const_zero(a) && !is_const_zero(b))
        {
            emit(static_cast<std::uint8_t>(ExprOp::Neg), ins.dst, b, {}, {});
            continue;
        }
        if (auto repl = expr_try_algebraic_simplify(op, a, b, c, out.consts,
                                                    def_op, def_a))
        {
            replace[dst] = *repl;
            continue;
        }

        // 生成化简后指令
        emit(ins.op, ins.dst, a, b, c);
    }

    out.num_regs = spec.num_regs;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  IR pass 2：死代码消除（DCE，IR-A）
//
//  从输出（最后一条指令 dst）反向标记存活，删除不影响输出的指令。
//  views/consts/num_regs 保持不变（后续 renumber 处理空洞）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec dead_code_elimination(const ExprSpec& spec)
{
    if (spec.instrs.empty())
        return spec;
    const std::size_t n = spec.instrs.size();
    std::vector<std::uint8_t> live_reg(spec.num_regs, 0);
    std::vector<std::uint8_t> live_instr(n, 0);
    live_reg[spec.instrs.back().dst] = 1;  // 输出必存活

    for (std::size_t i = n; i-- > 0;)
    {
        const ExprInstr& ins = spec.instrs[i];
        if (ins.dst < spec.num_regs && live_reg[ins.dst])
            live_instr[i] = 1;
        if (!live_instr[i])
            continue;
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
        const std::size_t nops = expr_instr_num_operands(op);
        for (std::size_t oi = 0; oi < nops; ++oi)
        {
            const ExprOperand& o = *ops[oi];
            if ((o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce)) &&
                o.idx < spec.num_regs)
                live_reg[o.idx] = 1;
        }
    }

    ExprSpec out;
    out.views   = spec.views;
    out.consts  = spec.consts;
    out.num_regs = spec.num_regs;
    out.matmul  = spec.matmul;
    out.instrs.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (live_instr[i])
            out.instrs.push_back(spec.instrs[i]);
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  IR pass 3：稳定重编号（IR-A）
//
//  寄存器按指令序连续重编号（0..k-1，消除 DCE/折叠产生的空洞），同时清理
//  未使用常量池项。views 不变。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec renumber_registers(const ExprSpec& spec)
{
    ExprSpec out;
    out.views  = spec.views;
    out.matmul = spec.matmul;

    // 常量池清理：只保留被引用的（保持出现序）
    std::vector<int> const_remap(spec.consts.size(), -1);
    for (const auto& in : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(in.op);
        const ExprOperand* ops[3] = {&in.a, &in.b, &in.c};
        const std::size_t nops = expr_instr_num_operands(op);
        for (std::size_t oi = 0; oi < nops; ++oi)
        {
            const ExprOperand& o = *ops[oi];
            if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const) &&
                const_remap[o.idx] < 0)
            {
                const_remap[o.idx] = static_cast<int>(out.consts.size());
                out.consts.push_back(spec.consts[o.idx]);
            }
        }
    }

    // 寄存器连续重编号
    std::vector<int> new_reg(spec.num_regs, -1);
    std::uint8_t next = 0;
    for (const auto& in : spec.instrs)
        new_reg[in.dst] = next++;

    out.instrs.reserve(spec.instrs.size());
    for (const auto& in : spec.instrs)
    {
        ExprInstr ni = in;
        ni.dst = static_cast<std::uint8_t>(new_reg[in.dst]);
        const auto remap = [&](ExprOperand o) -> ExprOperand
        {
            if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
            {
                o.idx = static_cast<std::uint8_t>(const_remap[o.idx]);
            }
            else if ((o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                      o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                      o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce)) &&
                     o.idx < spec.num_regs)
            {
                o.idx = static_cast<std::uint8_t>(new_reg[o.idx]);
            }
            return o;
        };
        // 只重映射实际使用的操作数（未用 b/c 是默认哨兵，不得当 Reg(0) 重编号）
        const std::size_t nops =
            expr_instr_num_operands(static_cast<ExprOp>(in.op));
        ni.a = remap(in.a);
        if (nops >= 2) ni.b = remap(in.b);
        if (nops >= 3) ni.c = remap(in.c);
        out.instrs.push_back(ni);
    }
    out.num_regs = next;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  IR pass 4：公共子表达式消除（CSE，IR-B）
//
//  按指令序，对 (op, a, b, c) 哈希（Fanout 归一化为 Reg；操作数经 dst 重映射），
//  命中则复用其 dst（不生成指令），否则分配新连续 dst。归约指令同样参与
//  CSE——Reduce 操作数引用经 dst_map 保持指向归约指令的规范 dst。
//  views/consts 不变。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec common_subexpression_elimination(const ExprSpec& spec)
{
    ExprSpec out;
    out.views  = spec.views;
    out.consts = spec.consts;
    out.matmul = spec.matmul;

    // key: op(8) + a.kind(8) + a.idx(8) + b.kind(8) + b.idx(8) + c.kind(8) + c.idx(8)
    std::unordered_map<std::uint64_t, std::uint8_t> cse;
    std::vector<int> dst_map(spec.num_regs, -1);
    std::uint8_t next = 0;

    const auto normalize = [](ExprOperand o) -> ExprOperand
    {
        if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout))
            o.kind = static_cast<std::uint8_t>(ExprOperandKind::Reg);  // Fanout 语义同 Reg
        return o;
    };

    out.instrs.reserve(spec.instrs.size());
    for (const auto& in : spec.instrs)
    {
        const auto remap = [&](ExprOperand o) -> ExprOperand
        {
            o = normalize(o);
            if ((o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce)) &&
                o.idx < spec.num_regs && dst_map[o.idx] >= 0)
                o.idx = static_cast<std::uint8_t>(dst_map[o.idx]);
            return o;
        };
        // 只重映射实际使用的操作数（未用 b/c 是默认哨兵；key 中保持哨兵不变）
        const std::size_t nops =
            expr_instr_num_operands(static_cast<ExprOp>(in.op));
        const ExprOperand a = remap(in.a);
        const ExprOperand b = (nops >= 2) ? remap(in.b) : in.b;
        const ExprOperand c = (nops >= 3) ? remap(in.c) : in.c;

        std::uint64_t key = static_cast<std::uint64_t>(in.op);
        key |= static_cast<std::uint64_t>(a.kind) << 8;
        key |= static_cast<std::uint64_t>(a.idx)  << 16;
        key |= static_cast<std::uint64_t>(b.kind) << 24;
        key |= static_cast<std::uint64_t>(b.idx)  << 32;
        key |= static_cast<std::uint64_t>(c.kind) << 40;
        key |= static_cast<std::uint64_t>(c.idx)  << 48;

        const auto it = cse.find(key);
        if (it != cse.end())
        {
            dst_map[in.dst] = it->second;  // 复用已存在子表达式
            continue;
        }
        const std::uint8_t nd = next++;
        cse.emplace(key, nd);
        dst_map[in.dst] = nd;
        ExprInstr ni; ni.op = in.op; ni.dst = nd; ni.a = a; ni.b = b; ni.c = c;
        out.instrs.push_back(ni);
    }
    out.num_regs = next;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  IR pass 5：寄存器分配（liveness 线性扫描，IR-B）
//
//  计算每条指令 dst 的最后使用位置（last_use），线性扫描复用已释放寄存器，
//  降低 num_regs（让更多表达式落在 EXPR_MAX_REGS=16 内）。确定性贪心：
//  每步释放所有 last_use < i 的寄存器，分配时取最小可用号。
//  输出寄存器（最后一条指令 dst）的 last_use 设为 n-1 防复用（值需保留到输出）。
//
//  区段划分（关键）：逐元素指令经 liveness 复用（区段 [0, next_elem)）；
//  归约指令**独立分配**（区段 [next_elem, ...)，不复用）——validate_expr_spec
//  要求 reduce_dst/elem_dst 按寄存器号互斥（CPU 求值的 reduce_vec[]/regs[]
//  也按号索引），归约 dst 不得与逐元素共享号。
//  views/consts 不变。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec allocate_registers_liveness(const ExprSpec& spec)
{
    if (spec.instrs.empty())
        return spec;
    const std::size_t n = spec.instrs.size();

    // def_at[reg] = 定义该寄存器的指令下标
    std::vector<int> def_at(spec.num_regs, -1);
    for (std::size_t i = 0; i < n; ++i)
        def_at[spec.instrs[i].dst] = static_cast<int>(i);

    // last_use[i]：指令 i 的 dst 最后一次被引用的指令下标
    std::vector<int> last_use(n, -1);
    for (std::size_t i = 0; i < n; ++i)
    {
        const ExprOp op = static_cast<ExprOp>(spec.instrs[i].op);
        const ExprOperand* ops[3] = {&spec.instrs[i].a, &spec.instrs[i].b, &spec.instrs[i].c};
        const std::size_t nops = expr_instr_num_operands(op);
        for (std::size_t oi = 0; oi < nops; ++oi)
        {
            const ExprOperand& o = *ops[oi];
            if ((o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce)) &&
                o.idx < spec.num_regs && def_at[o.idx] >= 0)
                last_use[def_at[o.idx]] =
                    std::max(last_use[def_at[o.idx]], static_cast<int>(i));
        }
    }
    last_use[n - 1] = static_cast<int>(n - 1);  // 输出寄存器保护

    // 分配（确定性贪心；逐元素 liveness 复用，归约独立区段）
    std::vector<int> reg_of(n, -1);
    std::vector<int> release_pos;        // 逐元素区段：已分配寄存器 -> 可释放位置
    std::vector<std::uint8_t> free_mark; // 逐元素区段：已释放标记
    for (std::size_t i = 0; i < n; ++i)
    {
        const ExprOp op = static_cast<ExprOp>(spec.instrs[i].op);
        if (expr_op_is_reduce(op))
            continue;  // 归约指令第二遍独立分配
        // 释放所有 last_use < i 的逐元素寄存器
        for (std::size_t r = 0; r < release_pos.size(); ++r)
            if (release_pos[r] >= 0 && release_pos[r] < static_cast<int>(i) && !free_mark[r])
                free_mark[r] = 1;
        // 取最小可用
        int reg = -1;
        for (std::size_t r = 0; r < release_pos.size(); ++r)
            if (free_mark[r]) { reg = static_cast<int>(r); free_mark[r] = 0; break; }
        if (reg < 0)
        {
            reg = static_cast<int>(release_pos.size());
            release_pos.push_back(-1);
            free_mark.push_back(0);
        }
        reg_of[i] = reg;
        release_pos[reg] = last_use[i];
    }
    const std::uint32_t next_elem = static_cast<std::uint32_t>(release_pos.size());
    std::uint32_t next_red = next_elem;
    for (std::size_t i = 0; i < n; ++i)
    {
        const ExprOp op = static_cast<ExprOp>(spec.instrs[i].op);
        if (expr_op_is_reduce(op))
            reg_of[i] = static_cast<int>(next_red++);
    }
    const std::uint32_t num_regs = next_red;

    ExprSpec out;
    out.views  = spec.views;
    out.consts = spec.consts;
    out.matmul = spec.matmul;
    out.instrs.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        ExprInstr ni = spec.instrs[i];
        ni.dst = static_cast<std::uint8_t>(reg_of[i]);
        const auto remap = [&](ExprOperand o) -> ExprOperand
        {
            if ((o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                 o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce)) &&
                o.idx < spec.num_regs && def_at[o.idx] >= 0)
                o.idx = static_cast<std::uint8_t>(reg_of[def_at[o.idx]]);
            return o;
        };
        // 只重映射实际使用的操作数（未用 b/c 是默认哨兵，不得当 Reg(0) 重编号）
        const std::size_t nops =
            expr_instr_num_operands(static_cast<ExprOp>(spec.instrs[i].op));
        ni.a = remap(ni.a);
        if (nops >= 2) ni.b = remap(ni.b);
        if (nops >= 3) ni.c = remap(ni.c);
        out.instrs.push_back(ni);
    }
    out.num_regs = num_regs;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  canonicalize_expr_spec — 完整规范化链（IR-A + IR-B）
//
//  canonical IR = 本函数输出。链序固定（确定性）：
//      fold(DCE/常量折叠/代数化简) → renumber → CSE → 寄存器分配
//  key / 去重 / shader 合成一律基于 canonical IR（两端一致）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline ExprSpec canonicalize_expr_spec(const ExprSpec& spec)
{
    ExprSpec s = fold_constants_and_algebra(spec);
    s = dead_code_elimination(s);
    s = renumber_registers(s);
    s = common_subexpression_elimination(s);
    s = allocate_registers_liveness(s);
    return s;
}

} // namespace nn

