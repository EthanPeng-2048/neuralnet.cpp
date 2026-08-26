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

#include "core_config.hpp"
#include "algebra_expr.hpp"

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

    // 注：Span 不定义成员运算符，所有 AST 构造由 expr.hpp 中的
    // 自由函数模板 operator+/-/*/>/< 等（基于 Expression 概念）统一处理。
    // 这保证 Span 与 ConstSpan/Val/任意 Expression 都能自然组合。
};

// ══════════════════════════════════════════════════════════════════════════
// ConstSpan：只读 AST 叶子节点
//
// 与 Span 的区别：
//   - 内部存储 const Scalar*（不可写入）
//   - 可从 const Matrix::span() 构造（std::span<const Scalar>）
//   - 可从 Span 隐式构造（Span → ConstSpan 单向转换）
//   - 仅满足 Expression 概念（作为 AST 输入），不能作为 compute::apply 输出
//
// 设计理由：
//   上层 Layer/Loss/Optimizer 在 const Matrix& 上调用 .span() 时，
//   得到的是 std::span<const Scalar>，无法构造可写 Span。
//   ConstSpan 提供类型安全的只读 AST 叶子，避免 const_cast 滥用。
// ══════════════════════════════════════════════════════════════════════════

class ConstSpan
{
private:
    const Scalar *data_;
    std::size_t size_;

public:
    // ── 构造函数 ──────────────────────────────────────────────────────────
    constexpr ConstSpan() noexcept : data_(nullptr), size_(0) {}
    constexpr ConstSpan(const Scalar *data, std::size_t size) noexcept : data_(data), size_(size) {}

    /// 从 std::span<const Scalar> 隐式构造（兼容 const Matrix::span()）
    constexpr ConstSpan(std::span<const Scalar> sp) noexcept : data_(sp.data()), size_(sp.size()) {}

    /// 从 std::span<Scalar> 隐式构造（兼容 mutable Matrix::span()）
    /// 注：标准上 std::span<Scalar> 应可隐式转 std::span<const Scalar>，
    ///     但部分 libc++ 实验/C++26 模式下该隐式转换受限，故显式提供此重载。
    constexpr ConstSpan(std::span<Scalar> sp) noexcept : data_(sp.data()), size_(sp.size()) {}

    /// 从 Span 隐式构造（Span 可作为只读输入）
    constexpr ConstSpan(const Span &s) noexcept : data_(s.data()), size_(s.size()) {}

    // ── 访问器 ───────────────────────────────────────────────────────────
    [[nodiscard]] constexpr const Scalar *data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    // ── 索引访问（只读） ─────────────────────────────────────────────────
    [[nodiscard]] constexpr Scalar operator[](std::size_t i) const noexcept { return data_[i]; }

    // ── 表达式树叶子节点的 eval ──────────────────────────────────────────
    [[nodiscard]] constexpr Scalar eval(std::size_t i) const noexcept { return data_[i]; }

    // ── 子视图 ───────────────────────────────────────────────────────────
    [[nodiscard]] constexpr ConstSpan subspan(std::size_t offset, std::size_t count) const noexcept
    {
        return ConstSpan{data_ + offset, count};
    }

    [[nodiscard]] constexpr ConstSpan subspan(std::size_t offset) const noexcept
    {
        return ConstSpan{data_ + offset, size_ - offset};
    }

    // 注：ConstSpan 不定义成员运算符，所有 AST 构造由 expr.hpp 中的
    // 自由函数模板 operator+/-/*/>/< 等（基于 Expression 概念）统一处理。
    // 这保证 ConstSpan 与 Span/Val/任意 Expression 都能自然组合。
};

} // namespace nn

#endif // NN_ALGEBRA_SPAN_HPP