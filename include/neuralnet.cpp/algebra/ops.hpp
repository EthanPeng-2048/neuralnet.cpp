#ifndef NN_ALGEBRA_OPS_HPP
#define NN_ALGEBRA_OPS_HPP

// ── ops.hpp — 表达式模板操作策略定义 ──────────────────────────────────────
// 每个 Op 定义 CPU 行为（apply），用于构建编译期 AST，
// 实现零开销抽象的逐元素运算。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>

#include "../nn_config.hpp"

namespace nn::ops
{

// ══════════════════════════════════════════════════════════════════════════
// 二元算术运算
// ══════════════════════════════════════════════════════════════════════════

struct Add
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a + b; }
};

struct Sub
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a - b; }
};

struct Mul
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a * b; }
};

struct Div
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a / b; }
};

// ══════════════════════════════════════════════════════════════════════════
// 一元运算
// ══════════════════════════════════════════════════════════════════════════

struct Neg
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a) noexcept { return -a; }
};

struct Abs
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::abs(a); }
};

struct Exp
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::exp(a); }
};

struct Log
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::log(a); }
};

struct Sigmoid
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return Scalar{1} / (Scalar{1} + std::exp(-a)); }
};

struct Tanh
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::tanh(a); }
};

struct ReLU
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a) noexcept { return a > Scalar{0} ? a : Scalar{0}; }
};

// ══════════════════════════════════════════════════════════════════════════
// 比较运算（返回 bool，用于条件表达式）
// ══════════════════════════════════════════════════════════════════════════

struct Gt
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a > b; }
};

struct Lt
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a < b; }
};

struct Ge
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a >= b; }
};

struct Le
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a <= b; }
};

struct Eq
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a == b; }
};

// ══════════════════════════════════════════════════════════════════════════
// 二元值运算（返回 Scalar）
// ══════════════════════════════════════════════════════════════════════════

struct Max
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return std::max(a, b); }
};

struct Min
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return std::min(a, b); }
};

} // namespace nn::ops

#endif // NN_ALGEBRA_OPS_HPP
