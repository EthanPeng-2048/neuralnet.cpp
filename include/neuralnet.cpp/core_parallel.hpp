#pragma once

// ══════════════════════════════════════════════════════════════════════════
//  core_parallel.hpp — 顶层并行算法（线程池 + 阈值串行回落的薄封装）
//
//  与 core_config.hpp 的关系：core_config 只保留标量/阈值等**编译期常量**，
//  不再依赖线程池；本头把"常量阈值 + global_thread_pool 调度"组合成对齐
//  std::for_each / std::transform 的顶层并行算法。分层动机见 dev/separate_repos：
//  core_config 作为 nn-base 叶子头必须摆脱对 core_threadpool 的依赖，
//  使"只用常量（Scalar/BLOCK_SIZE/EPSILON）"的消费方不再传递拉入线程池。
//
//  依赖：core_config.hpp（PARALLEL_THRESHOLD 常量）+ core_threadpool.hpp（调度）。
//  使用方：algebra_matrix / compute_cpu_engine / expr_dsl / domain_tokenizer_base。
// ══════════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <iterator>  // for std::ranges::distance
#include <utility>   // for std::forward

#include "core_config.hpp"     // PARALLEL_THRESHOLD
#include "core_threadpool.hpp" // global_thread_pool

namespace nn
{
    // ── 顶层并行算法函数 ─────────────────────────────────────────────────────
    // 用法与 std::for_each / std::transform 完全一致，内部自动选择串行/并行。
    // 替代 std::execution::par 的线程创建开销，也替代手写 for 循环。

    // for_each
    template<typename Iterator, typename Func>
    inline void for_each(Iterator first, Iterator last, Func&& func)
    {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (n >= PARALLEL_THRESHOLD)
        {
            global_thread_pool().parallel_for_each(first, last, std::forward<Func>(func));
        }
        else
        {
            for (auto it = first; it != last; ++it)
                func(*it);
        }
    }

    // transform（一元）
    template<typename InputIt, typename OutputIt, typename UnaryOp>
    inline void transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op)
    {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (n >= PARALLEL_THRESHOLD)
        {
            global_thread_pool().parallel_transform(first, last, d_first, std::forward<UnaryOp>(op));
        }
        else
        {
            for (; first != last; ++first, ++d_first)
                *d_first = op(*first);
        }
    }

    // transform（二元）
    template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    inline void transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                          OutputIt d_first, BinaryOp&& op)
    {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first1, last1));
        if (n >= PARALLEL_THRESHOLD)
        {
            global_thread_pool().parallel_transform(first1, last1, first2, d_first,
                                                     std::forward<BinaryOp>(op));
        }
        else
        {
            for (; first1 != last1; ++first1, ++first2, ++d_first)
                *d_first = op(*first1, *first2);
        }
    }

    // transform_reduce（一元 transform）
    template<typename InputIt, typename T, typename BinaryReduceOp, typename UnaryTransformOp>
    inline T transform_reduce(InputIt first, InputIt last, T init,
                              BinaryReduceOp&& reduce_op, UnaryTransformOp&& transform_op)
    {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (n >= PARALLEL_THRESHOLD)
        {
            return global_thread_pool().parallel_transform_reduce(
                first, last, init,
                std::forward<BinaryReduceOp>(reduce_op),
                std::forward<UnaryTransformOp>(transform_op));
        }
        else
        {
            T acc = init;
            for (; first != last; ++first)
                acc = reduce_op(acc, transform_op(*first));
            return acc;
        }
    }

    // transform_reduce（二元 transform）
    template<typename InputIt1, typename InputIt2, typename T,
             typename BinaryReduceOp, typename BinaryTransformOp>
    inline T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                              T init, BinaryReduceOp&& reduce_op,
                              BinaryTransformOp&& transform_op)
    {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first1, last1));
        if (n >= PARALLEL_THRESHOLD)
        {
            return global_thread_pool().parallel_transform_reduce(
                first1, last1, first2, init,
                std::forward<BinaryReduceOp>(reduce_op),
                std::forward<BinaryTransformOp>(transform_op));
        }
        else
        {
            T acc = init;
            for (; first1 != last1; ++first1, ++first2)
                acc = reduce_op(acc, transform_op(*first1, *first2));
            return acc;
        }
    }

    // parallel_for_blocks（块级并行，无阈值保护）
    template<typename Iterator, typename Func>
    inline void parallel_for_blocks(Iterator first, Iterator last, Func&& func)
    {
        global_thread_pool().parallel_for_blocks(first, last, std::forward<Func>(func));
    }

    // parallel_for_samples（样本级并行）
    template<typename Func>
    inline void parallel_for_samples(std::size_t num_samples, Func&& func)
    {
        if (num_samples <= 1)
        {
            for (std::size_t i = 0; i < num_samples; ++i)
                func(i);
        }
        else
        {
            global_thread_pool().parallel_for_samples(num_samples, std::forward<Func>(func));
        }
    }
} // namespace nn
