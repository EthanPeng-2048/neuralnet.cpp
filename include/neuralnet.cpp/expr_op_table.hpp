#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_op_table.hpp — 算子元数据单一真源（RV-07）
//
//  以 `algebra_ops.hpp` 的 `nn::ops::*`（唯一算子语义源，`apply` 即 CPU 标量
//  语义）为基，为每个 `ExprOp` 提供**一行**元数据：元数 / 归约语义 / 可折叠性 /
//  CPU 标量函数指针 / GLSL 发射形态 + 符号。
//
//  消费者全部查表（此前同一个算子散落在约 9 处 switch）：
//    ① `compute_cpu_engine.hpp`——标量求值 + 两处寄存器融合（3 处 → 1 处）；
//    ② `expr_opt.hpp`——常量折叠；
//    ③ `expr_glsl_gen.hpp`——shader 指令发射（6 处 → 1 处）。
//
//  新增一个逐元素算子只需两处：在 `expr_spec.hpp` 的 `ExprOp` 加枚举值，
//  在本表加一行（元数非默认 2 时再同步 `expr_instr_num_operands`——由本文件末尾
//  的 `consteval` 一致性断言强制核对，写漏即编译失败）。
//
//  ⚠ 不变量：本表覆盖 `ExprOp` 的**全部**取值（含 4 条归约指令，emit=None）；
//  归约指令不进 CPU 标量 ALU / GLSL 逐元素发射，由调用方在查表前分流。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <iterator>

#include "algebra_ops.hpp"  // nn::ops::*（apply = CPU 标量语义）+ expr_spec（ExprOp）

namespace nn::detail
{

// GLSL 一条指令的字符串骨架由"发射形态"决定，符号 / 函数名由表提供——
// 这样"新增一个算子"不必改发射器，只需选一个已有形态 + 填符号。
enum class ExprEmitKind : std::uint8_t
{
    None,     // 归约指令：不经逐元素发射
    Infix,    // a <sym> b
    Func2,    // sym(a, b)            （如 max/min）
    Compare,  // (a <sym> b) ? 1 : 0  （vec4 形态用 sym_vec：vec4(sym_vec(a, b))）
    Neg,      // -(a)
    Func1,    // sym(a)
    Select,   // (a != 0) ? b : c     （vec4 形态：mix(c, b, notEqual(a, vec4(0.0)))）
};

// CPU 标量语义函数指针（比较类返回 1.0/0.0 的 Scalar，与写回寄存器的语义一致）
using ExprUnaryFn  = Scalar (*)(Scalar) noexcept;
using ExprBinaryFn = Scalar (*)(Scalar, Scalar) noexcept;

struct ExprOpInfo
{
    std::uint8_t arity;         // 实际使用操作数个数（== expr_instr_num_operands）
    bool         is_reduce;     // dst 是隐式"每行/每列一个标量"的归约向量
    bool         reduces_cols;  // 归约方向：true=按列 → (1,cols)，false=按行 → (rows,1)
    bool         foldable;      // 常量折叠（仅 IEEE 754 基本运算；不含超越函数/归约）
    ExprUnaryFn  unary;         // arity==1 且非归约：unary(a)
    ExprBinaryFn binary;        // arity==2：binary(a,b)
    ExprEmitKind emit;          // GLSL 发射形态
    const char*  sym;           // 中缀符号 / 函数名
    const char*  sym_vec;       // Compare 的 vec4 内建名（否则 ""）
};

// 比较算子：`ops::Lt` 等的 apply 返回 bool（供条件表达式），寄存器写回要 Scalar
//  1.0/0.0——此处做唯一一次桥接。
namespace cmp
{
[[nodiscard]] inline Scalar lt(Scalar a, Scalar b) noexcept { return ops::Lt::apply(a, b) ? Scalar{1} : Scalar{0}; }
[[nodiscard]] inline Scalar le(Scalar a, Scalar b) noexcept { return ops::Le::apply(a, b) ? Scalar{1} : Scalar{0}; }
[[nodiscard]] inline Scalar gt(Scalar a, Scalar b) noexcept { return ops::Gt::apply(a, b) ? Scalar{1} : Scalar{0}; }
[[nodiscard]] inline Scalar ge(Scalar a, Scalar b) noexcept { return ops::Ge::apply(a, b) ? Scalar{1} : Scalar{0}; }
[[nodiscard]] inline Scalar eq(Scalar a, Scalar b) noexcept { return ops::Eq::apply(a, b) ? Scalar{1} : Scalar{0}; }
[[nodiscard]] inline Scalar ne(Scalar a, Scalar b) noexcept { return ops::Ne::apply(a, b) ? Scalar{1} : Scalar{0}; }
}  // namespace cmp

// 行构造器（字段全列——避开 -Wmissing-designated-field-initializers）
namespace row
{
[[nodiscard]] consteval ExprOpInfo unary(ExprUnaryFn f, ExprEmitKind e,
                                         const char* sym, bool fold = false) noexcept
{
    return ExprOpInfo{1, false, false, fold, f, nullptr, e, sym, ""};
}
[[nodiscard]] consteval ExprOpInfo binary(ExprBinaryFn f, ExprEmitKind e,
                                          const char* sym, const char* sym_vec = "",
                                          bool fold = true) noexcept
{
    return ExprOpInfo{2, false, false, fold, nullptr, f, e, sym, sym_vec};
}
[[nodiscard]] consteval ExprOpInfo select_op() noexcept
{
    return ExprOpInfo{3, false, false, true, nullptr, nullptr, ExprEmitKind::Select, "", ""};
}
[[nodiscard]] consteval ExprOpInfo reduce(bool cols) noexcept
{
    return ExprOpInfo{1, true, cols, false, nullptr, nullptr, ExprEmitKind::None, "", ""};
}
}  // namespace row

// 表按下标 == static_cast<uint8_t>(ExprOp) 直接索引（顺序不得改动）
inline constexpr ExprOpInfo kExprOpTable[] = {
    /* Neg   */ row::unary(&ops::Neg::apply,   ExprEmitKind::Neg,   "-",           /*fold=*/true),
    /* Exp   */ row::unary(&ops::Exp::apply,   ExprEmitKind::Func1, "exp"),
    /* Log   */ row::unary(&ops::Log::apply,   ExprEmitKind::Func1, "log"),
    /* Sqrt  */ row::unary(&ops::Sqrt::apply,  ExprEmitKind::Func1, "sqrt"),
    /* Rsqrt */ row::unary(&ops::Rsqrt::apply, ExprEmitKind::Func1, "inversesqrt"),
    /* Abs   */ row::unary(&ops::Abs::apply,   ExprEmitKind::Func1, "abs",         /*fold=*/true),
    /* Tanh  */ row::unary(&ops::Tanh::apply,  ExprEmitKind::Func1, "tanh"),
    /* Add   */ row::binary(&ops::Add::apply,  ExprEmitKind::Infix,   "+"),
    /* Sub   */ row::binary(&ops::Sub::apply,  ExprEmitKind::Infix,   "-"),
    /* Mul   */ row::binary(&ops::Mul::apply,  ExprEmitKind::Infix,   "*"),
    /* Div   */ row::binary(&ops::Div::apply,  ExprEmitKind::Infix,   "/"),
    /* Max   */ row::binary(&ops::Max::apply,  ExprEmitKind::Func2,   "max"),
    /* Min   */ row::binary(&ops::Min::apply,  ExprEmitKind::Func2,   "min"),
    /* Lt    */ row::binary(&cmp::lt,          ExprEmitKind::Compare, "<",  "lessThan"),
    /* Le    */ row::binary(&cmp::le,          ExprEmitKind::Compare, "<=", "lessThanEqual"),
    /* Gt    */ row::binary(&cmp::gt,          ExprEmitKind::Compare, ">",  "greaterThan"),
    /* Ge    */ row::binary(&cmp::ge,          ExprEmitKind::Compare, ">=", "greaterThanEqual"),
    /* Eq    */ row::binary(&cmp::eq,          ExprEmitKind::Compare, "==", "equal"),
    /* Ne    */ row::binary(&cmp::ne,          ExprEmitKind::Compare, "!=", "notEqual"),
    /* Select*/ row::select_op(),
    /* ColSum*/ row::reduce(/*cols=*/true),
    /* ColMax*/ row::reduce(/*cols=*/true),
    /* RowSum*/ row::reduce(/*cols=*/false),
    /* RowMax*/ row::reduce(/*cols=*/false),
};

[[nodiscard]] inline constexpr std::size_t expr_op_count() noexcept { return std::size(kExprOpTable); }

[[nodiscard]] inline constexpr const ExprOpInfo& expr_op_info(ExprOp op) noexcept
{
    return kExprOpTable[static_cast<std::uint8_t>(op)];
}

// CPU 标量求值（一元 / 二元 / Select 三元）；**归约指令不得走本函数**（调用方分流）。
[[nodiscard]] inline Scalar expr_alu_scalar(ExprOp op, Scalar a, Scalar b, Scalar c) noexcept
{
    const ExprOpInfo& info = expr_op_info(op);
    if (info.arity == 1) return info.unary(a);
    if (info.arity == 2) return info.binary(a, b);
    return (a != Scalar{0}) ? b : c;   // Select
}

// ── 与 expr_spec.hpp 的辅助函数强制一致（写漏同步即编译失败）──────────────
[[nodiscard]] consteval bool expr_op_table_consistent() noexcept
{
    for (std::size_t i = 0; i < expr_op_count(); ++i)
    {
        const ExprOp op = static_cast<ExprOp>(i);
        if (kExprOpTable[i].arity        != expr_instr_num_operands(op)) return false;
        if (kExprOpTable[i].is_reduce    != expr_op_is_reduce(op))       return false;
        if (kExprOpTable[i].reduces_cols != expr_op_reduces_cols(op))    return false;
    }
    return true;
}
static_assert(expr_op_table_consistent(),
              "expr_op_table.hpp 与 expr_spec.hpp 的算子辅助函数不一致："
              "新增/改动算子时须同步本表（arity/reduce/reduces_cols）");

}  // namespace nn::detail
