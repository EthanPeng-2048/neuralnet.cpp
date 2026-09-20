#pragma once

// ── compute_dispatch.hpp — 表达式模板统一入口 ─────────────────────────────
// compute::apply(span, expr) 是所有逐元素运算的统一入口。
//
// CPU 路径：递归求值 AST，编译器内联所有 eval() 调用，
//           消除 AST 节点内存分配（全部栈上值类型），向量化最终循环。
//           实测性能与手写 for 循环完全等价（零开销抽象）。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <algorithm>
#include <iterator>
#include <ranges>
#include <thread>

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
///
/// 实现与 `nn::dsl::eval_into_span`（DSL 模板路径）保持**同一循环结构 +
/// 同一向量化提示 + 同一分块并行**：这两条路此前已分叉——本函数仅用
/// `nn::for_each`（无 `NN_VECTORIZE_PRAGMA`，块内层由线程池统一处理），
/// 导致 eager 逐元素原语比等价的 DSL 表达式明显慢。二者现在同构。
template <typename Expr>
void apply(Span x, const Expr &expr)
{
    const auto n = x.size();
    if (n == 0) return;

    Scalar* d = x.data();
    if (n < PARALLEL_THRESHOLD)
    {
        NN_VECTORIZE_PRAGMA
        for (std::size_t i = 0; i < n; ++i)
            d[i] = static_cast<Scalar>(expr.eval(i));
        return;
    }

    const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(),
                                                 std::size_t{1});
    constexpr std::size_t TARGET_CHUNK = std::size_t{1} << 16;  // ~64K 元素/块
    const std::size_t n_chunks = std::clamp(n / TARGET_CHUNK, std::size_t{1}, hw * 4);
    const std::size_t base = n / n_chunks;
    const std::size_t rem  = n % n_chunks;

    nn::parallel_for_samples(n_chunks,
        [d, &expr, base, rem](std::size_t chunk) noexcept
        {
            const std::size_t off = chunk * base + std::min(chunk, rem);
            const std::size_t len = base + (chunk < rem ? 1 : 0);
            Scalar* dc = d + off;
            NN_VECTORIZE_PRAGMA
            for (std::size_t k = 0; k < len; ++k)
                dc[k] = static_cast<Scalar>(expr.eval(off + k));
        });
}

} // namespace nn::compute

