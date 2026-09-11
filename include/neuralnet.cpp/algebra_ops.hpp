#ifndef NN_ALGEBRA_OPS_HPP
#define NN_ALGEBRA_OPS_HPP

// ── ops.hpp — 表达式模板操作策略定义 ──────────────────────────────────────
// 每个 Op 定义 CPU 行为（apply），用于构建编译期 AST，
// 实现零开销抽象的逐元素运算。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>

#include "core_config.hpp"
#include "expr_spec.hpp"  // ExprOp（单一算子来源：ops 是唯一事实来源）

namespace nn::ops
{

// ══════════════════════════════════════════════════════════════════════════
// 二元算术运算
// ══════════════════════════════════════════════════════════════════════════

struct Add
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a + b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Add; }
};

struct Sub
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a - b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Sub; }
};

struct Mul
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a * b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Mul; }
};

struct Div
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return a / b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Div; }
};

// ══════════════════════════════════════════════════════════════════════════
// 一元运算
// ══════════════════════════════════════════════════════════════════════════

struct Neg
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a) noexcept { return -a; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Neg; }
};

struct Abs
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::abs(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Abs; }
};

struct Exp
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::exp(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Exp; }
};

struct Log
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::log(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Log; }
};

struct Sqrt
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::sqrt(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Sqrt; }
};

struct Rsqrt
{
    /// 1/sqrt(x) —— LayerNorm 标准差倒数专用，单次调用避免除法
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return Scalar{1} / std::sqrt(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Rsqrt; }
};

struct Sigmoid
{
    // ── 数值稳定实现：大正数直接返回 1，大负数返回 0，避免 exp 上溢 ──
    // 原 1/(1+exp(-a)) 在 a 为大负数时 exp(-a) 会溢出为 inf → 结果为 0
    // 但 NaN/Inf 可能传播；此处分支保证全区间有界且无 NaN。
    [[nodiscard]] static Scalar apply(Scalar a) noexcept
    {
        if (a >= Scalar{0})
        {
            const Scalar z = std::exp(-a);
            return Scalar{1} / (Scalar{1} + z);
        }
        else
        {
            const Scalar z = std::exp(a);
            return z / (Scalar{1} + z);
        }
    }
};

struct Tanh
{
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::tanh(a); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Tanh; }
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
    static constexpr ExprOp op_id() noexcept { return ExprOp::Gt; }
};

struct Lt
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a < b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Lt; }
};

struct Ge
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a >= b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Ge; }
};

struct Le
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a <= b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Le; }
};

struct Eq
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a == b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Eq; }
};

struct Ne
{
    [[nodiscard]] static constexpr bool apply(Scalar a, Scalar b) noexcept { return a != b; }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Ne; }
};

// ══════════════════════════════════════════════════════════════════════════
// 二元值运算（返回 Scalar）
// ══════════════════════════════════════════════════════════════════════════

struct Max
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return std::max(a, b); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Max; }
};

struct Min
{
    [[nodiscard]] static constexpr Scalar apply(Scalar a, Scalar b) noexcept { return std::min(a, b); }
    static constexpr ExprOp op_id() noexcept { return ExprOp::Min; }
};

} // namespace nn::ops

#endif // NN_ALGEBRA_OPS_HPP