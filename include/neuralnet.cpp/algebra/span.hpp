#ifndef NN_ALGEBRA_SPAN_HPP
#define NN_ALGEBRA_SPAN_HPP

// ── span.hpp — 可构建 AST 的智能视图 ─────────────────────────────────────
// 替代 std::span<Scalar> 作为逐元素操作的载体。
// Span 同时是：
//   1. 数据视图（指向 Matrix 内部数据）
//   2. 表达式树叶子节点（eval(i) 返回 data_[i]）
//   3. compute::apply 的输出目标（被写入）
//
// 运算符重载在幕后构建 AST，上层与具体执行路径解耦。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <span>

#include "../nn_config.hpp"
#include "expr.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Span：可构建 AST 的智能视图
// ══════════════════════════════════════════════════════════════════════════

class Span
{
private:
    Scalar *data_;
    std::size_t size_;

public:
    // ── 构造函数 ──────────────────────────────────────────────────────────
    constexpr Span() noexcept : data_(nullptr), size_(0) {}
    constexpr Span(Scalar *data, std::size_t size) noexcept : data_(data), size_(size) {}

    /// 从 std::span<Scalar> 隐式构造（兼容 Matrix::span()）
    constexpr Span(std::span<Scalar> sp) noexcept : data_(sp.data()), size_(sp.size()) {}

    // ── 访问器 ───────────────────────────────────────────────────────────
    [[nodiscard]] constexpr Scalar *data() noexcept { return data_; }
    [[nodiscard]] constexpr const Scalar *data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    // ── 索引访问 ─────────────────────────────────────────────────────────
    [[nodiscard]] constexpr Scalar &operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] constexpr Scalar operator[](std::size_t i) const noexcept { return data_[i]; }

    // ── 表达式树叶子节点的 eval ──────────────────────────────────────────
    [[nodiscard]] constexpr Scalar eval(std::size_t i) const noexcept { return data_[i]; }

    // ── 子视图 ───────────────────────────────────────────────────────────
    [[nodiscard]] constexpr Span subspan(std::size_t offset, std::size_t count) const noexcept
    {
        return Span{data_ + offset, count};
    }

    [[nodiscard]] constexpr Span subspan(std::size_t offset) const noexcept
    {
        return Span{data_ + offset, size_ - offset};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 运算符重载：Span OP Scalar → BinaryExpr
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] constexpr auto operator+(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Add>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator-(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Sub>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator*(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Mul>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator/(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Div>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator>(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Gt>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator<(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Lt>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator>=(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Ge>{*this, Val{s}};
    }

    [[nodiscard]] constexpr auto operator<=(Scalar s) const
    {
        return BinaryExpr<Span, Val, ops::Le>{*this, Val{s}};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 运算符重载：Span OP Span → BinaryExpr
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] constexpr auto operator+(const Span &other) const
    {
        return BinaryExpr<Span, Span, ops::Add>{*this, other};
    }

    [[nodiscard]] constexpr auto operator-(const Span &other) const
    {
        return BinaryExpr<Span, Span, ops::Sub>{*this, other};
    }

    [[nodiscard]] constexpr auto operator*(const Span &other) const
    {
        return BinaryExpr<Span, Span, ops::Mul>{*this, other};
    }

    [[nodiscard]] constexpr auto operator/(const Span &other) const
    {
        return BinaryExpr<Span, Span, ops::Div>{*this, other};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 一元运算符
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] constexpr auto operator-() const
    {
        return UnaryExpr<Span, ops::Neg>{*this};
    }
};

} // namespace nn

#endif // NN_ALGEBRA_SPAN_HPP
