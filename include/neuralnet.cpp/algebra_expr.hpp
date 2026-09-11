#pragma once

// ── expr.hpp — 表达式模板核心：编译期 AST 节点类型系统 ───────────────────
// 上层用普通 C++ 运算符写表达式，编译器自动构建编译期 AST。
// CPU 路径：递归求值 AST（零开销，和手写循环等价）。
//
// 算法永远在上层，计算永远在底层，中间靠 AST 传递。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <type_traits>

#include "core_config.hpp"
#include "algebra_ops.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// 表达式概念（C++20 Concepts）
// ══════════════════════════════════════════════════════════════════════════

/// 表达式：可按索引求值为 Scalar 的类型
template <typename T>
concept Expression = requires(const T &t, std::size_t i) {
    { t.eval(i) } -> std::convertible_to<Scalar>;
};

/// 布尔表达式：可按索引求值为 bool 的类型（用于条件选择）
template <typename T>
concept BoolExpression = requires(const T &t, std::size_t i) {
    { t.eval(i) } -> std::convertible_to<bool>;
};

// ══════════════════════════════════════════════════════════════════════════
// 叶子节点：常量值
// ══════════════════════════════════════════════════════════════════════════

struct Val
{
    Scalar value;

    [[nodiscard]] constexpr Scalar eval(std::size_t /*i*/) const noexcept { return value; }
};

// ══════════════════════════════════════════════════════════════════════════
// 一元表达式
// ══════════════════════════════════════════════════════════════════════════

template <typename Child, typename Op>
struct UnaryExpr
{
    Child child;

    [[nodiscard]] constexpr auto eval(std::size_t i) const noexcept
    {
        return Op::apply(child.eval(i));
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 二元表达式
// ══════════════════════════════════════════════════════════════════════════

template <typename Left, typename Right, typename Op>
struct BinaryExpr
{
    Left left;
    Right right;

    [[nodiscard]] constexpr auto eval(std::size_t i) const noexcept
    {
        return Op::apply(left.eval(i), right.eval(i));
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 三元表达式（条件选择：cond ? then_expr : else_expr）
// ══════════════════════════════════════════════════════════════════════════

template <typename Cond, typename Then, typename Else>
struct TernaryExpr
{
    Cond cond;
    Then then_expr;
    Else else_expr;

    [[nodiscard]] constexpr auto eval(std::size_t i) const noexcept
    {
        return cond.eval(i) ? then_expr.eval(i) : else_expr.eval(i);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 自由函数：一元数学函数
// ══════════════════════════════════════════════════════════════════════════

/// 取反：-x
template <Expression Expr>
[[nodiscard]] constexpr auto neg(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Neg>{expr};
}

/// 绝对值：|x|
template <Expression Expr>
[[nodiscard]] auto abs(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Abs>{expr};
}

/// 指数：e^x
template <Expression Expr>
[[nodiscard]] auto exp(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Exp>{expr};
}

/// 自然对数：ln(x)
template <Expression Expr>
[[nodiscard]] auto log(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Log>{expr};
}

/// 平方根：√x
template <Expression Expr>
[[nodiscard]] auto sqrt(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Sqrt>{expr};
}

/// 平方根倒数：1/√x —— LayerNorm 标准差倒数专用
template <Expression Expr>
[[nodiscard]] auto rsqrt(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Rsqrt>{expr};
}

/// Sigmoid：1 / (1 + e^(-x))
template <Expression Expr>
[[nodiscard]] auto sigmoid(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Sigmoid>{expr};
}

/// 双曲正切：tanh(x)
template <Expression Expr>
[[nodiscard]] auto tanh(const Expr &expr)
{
    return UnaryExpr<Expr, ops::Tanh>{expr};
}

/// ReLU：max(x, 0)
template <Expression Expr>
[[nodiscard]] auto relu(const Expr &expr)
{
    return UnaryExpr<Expr, ops::ReLU>{expr};
}

// ══════════════════════════════════════════════════════════════════════════
// 自由函数：二元值运算
// ══════════════════════════════════════════════════════════════════════════

/// 最大值：max(a, b)
template <Expression L, Expression R>
[[nodiscard]] constexpr auto max(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Max>{l, r};
}

template <Expression Expr>
[[nodiscard]] constexpr auto max(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Max>{expr, Val{s}};
}

template <Expression Expr>
[[nodiscard]] constexpr auto max(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Max>{Val{s}, expr};
}

/// 最小值：min(a, b)
template <Expression L, Expression R>
[[nodiscard]] constexpr auto min(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Min>{l, r};
}

template <Expression Expr>
[[nodiscard]] constexpr auto min(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Min>{expr, Val{s}};
}

template <Expression Expr>
[[nodiscard]] constexpr auto min(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Min>{Val{s}, expr};
}

// ══════════════════════════════════════════════════════════════════════════
// 自由函数：条件选择
// ══════════════════════════════════════════════════════════════════════════

/// select(cond, then_expr, else_expr) — 等价于 cond ? then : else
/// 用法：select(x > Scalar{0}, x, x * 0.01f)  // LeakyReLU
template <BoolExpression Cond, Expression Then, Expression Else>
[[nodiscard]] constexpr auto select(const Cond &cond, const Then &then_expr, const Else &else_expr)
{
    return TernaryExpr<Cond, Then, Else>{cond, then_expr, else_expr};
}

/// select 重载：else 分支为标量
/// 用法：select(x > Scalar{0}, go, Scalar{0})  // ReLU 反向
template <BoolExpression Cond, Expression Then>
[[nodiscard]] constexpr auto select(const Cond &cond, const Then &then_expr, Scalar else_val)
{
    return TernaryExpr<Cond, Then, Val>{cond, then_expr, Val{else_val}};
}

/// select 重载：then 分支为标量
template <BoolExpression Cond, Expression Else>
[[nodiscard]] constexpr auto select(const Cond &cond, Scalar then_val, const Else &else_expr)
{
    return TernaryExpr<Cond, Val, Else>{cond, Val{then_val}, else_expr};
}

/// select 重载：then/else 均为标量
template <BoolExpression Cond>
[[nodiscard]] constexpr auto select(const Cond &cond, Scalar then_val, Scalar else_val)
{
    return TernaryExpr<Cond, Val, Val>{cond, Val{then_val}, Val{else_val}};
}

// ══════════════════════════════════════════════════════════════════════════
// 模板运算符：Scalar op Expression / Expression op Scalar
// 用于复合表达式中的标量与任意表达式类型的运算
// 例如：1.0f + exp(x) → BinaryExpr<Val, UnaryExpr<...>, ops::Add>
//
// 注：C++ 模板推导要求 Scalar/Expr 左右位置分别定义，无法合并。
// 模板实参推导不会自动尝试 (Scalar, Expr) ↔ (Expr, Scalar) 的对称转换，
// 因此 operator+(Scalar, Expr) 和 operator+(Expr, Scalar) 必须分别定义。
// 下面 22 个运算符重载按 Scalar op Expr、Expr op Scalar、Expr op Expr 三组排列。
// ══════════════════════════════════════════════════════════════════════════

// ── Scalar + Expression ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator+(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Add>{Val{s}, expr};
}

// ── Scalar - Expression ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator-(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Sub>{Val{s}, expr};
}

// ── Scalar * Expression ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator*(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Mul>{Val{s}, expr};
}

// ── Scalar / Expression ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator/(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Div>{Val{s}, expr};
}

// ── Scalar > Expression → bool 表达式 ────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator>(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Gt>{Val{s}, expr};
}

// ── Scalar < Expression → bool 表达式 ────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator<(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Lt>{Val{s}, expr};
}

// ── Scalar >= Expression → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator>=(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Ge>{Val{s}, expr};
}

// ── Scalar <= Expression → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator<=(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Le>{Val{s}, expr};
}

// ── Scalar == Expression → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator==(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Eq>{Val{s}, expr};
}

// ── Scalar != Expression → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator!=(Scalar s, const Expr &expr)
{
    return BinaryExpr<Val, Expr, ops::Ne>{Val{s}, expr};
}

// ── Expression + Scalar ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator+(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Add>{expr, Val{s}};
}

// ── Expression - Scalar ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator-(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Sub>{expr, Val{s}};
}

// ── Expression * Scalar ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator*(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Mul>{expr, Val{s}};
}

// ── Expression / Scalar ──────────────────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator/(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Div>{expr, Val{s}};
}

// ── Expression > Scalar → bool 表达式 ────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator>(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Gt>{expr, Val{s}};
}

// ── Expression < Scalar → bool 表达式 ────────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator<(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Lt>{expr, Val{s}};
}

// ── Expression >= Scalar → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator>=(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Ge>{expr, Val{s}};
}

// ── Expression <= Scalar → bool 表达式 ───────────────────────────────────
template <Expression Expr>
[[nodiscard]] constexpr auto operator<=(const Expr &expr, Scalar s)
{
    return BinaryExpr<Expr, Val, ops::Le>{expr, Val{s}};
}

// ── Expression + Expression ──────────────────────────────────────────────
template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator+(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Add>{l, r};
}

// ── Expression - Expression ──────────────────────────────────────────────
template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator-(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Sub>{l, r};
}

// ── Expression * Expression ──────────────────────────────────────────────
template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator*(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Mul>{l, r};
}

// ── Expression / Expression ──────────────────────────────────────────────
template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator/(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Div>{l, r};
}

// ═══════════════════════════════════════════════════════════════════════════
// Expression 与 Expression 的比较运算（返回 bool 表达式 AST）
// ═══════════════════════════════════════════════════════════════════════════

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator>(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Gt>{l, r};
}

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator<(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Lt>{l, r};
}

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator>=(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Ge>{l, r};
}

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator<=(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Le>{l, r};
}

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator==(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Eq>{l, r};
}

template <Expression L, Expression R>
[[nodiscard]] constexpr auto operator!=(const L &l, const R &r)
{
    return BinaryExpr<L, R, ops::Ne>{l, r};
}

} // namespace nn

