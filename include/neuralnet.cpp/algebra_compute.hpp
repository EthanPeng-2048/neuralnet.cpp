#pragma once

// ── compute_dispatch.hpp — 表达式模板统一入口 ─────────────────────────────
// compute::apply(span, expr) 是所有逐元素运算的统一入口。
//
// CPU 路径：递归求值 AST，编译器内联所有 eval() 调用，
//           消除 AST 节点内存分配（全部栈上值类型），向量化最终循环。
//           实测性能与手写 for 循环完全等价（零开销抽象）。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <ranges>

#include "core_config.hpp"
#include "algebra_span.hpp"

namespace nn::compute
{

// ══════════════════════════════════════════════════════════════════════════
// apply — 统一入口：对 span 中的每个元素应用表达式
// ══════════════════════════════════════════════════════════════════════════
//
// 用法示例：
//   Span x = matrix.span();
//   compute::apply(x, max(x, Scalar{0}));                           // ReLU
//   compute::apply(x, x * (1.0f / (1.0f + exp(-1.702f * x))));    // GeLU
//   compute::apply(x, select(x > Scalar{0}, x, x * 0.01f));        // LeakyReLU
//   compute::apply(x, abs(x));                                      // abs
//   compute::apply(x, x * 2.0f + 1.0f);                            // 线性变换
//
/// @param x    输出目标（Span 视图，指向 Matrix 内部数据）
/// @param expr 表达式树（编译期 AST，由运算符重载自动构建）
///
/// @note x 和 expr 中的 Span 可以指向同一内存（就地操作安全），
///       因为每个元素在读取后才被写入。
template <typename Expr>
void apply(Span x, const Expr &expr)
{
    const auto n = x.size();
    if (n == 0) return;

    auto indices = std::views::iota(std::size_t{0}, n);
    nn::for_each(indices.begin(), indices.end(),
        [&x, &expr](std::size_t i) noexcept
        {
            x[i] = static_cast<Scalar>(expr.eval(i));
        });
}

} // namespace nn::compute

