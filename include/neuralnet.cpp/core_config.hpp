#ifndef NN_CORE_CONFIG_HPP
#define NN_CORE_CONFIG_HPP

#include <cstddef>
#include <iterator>  // for std::distance
#include <thread>    // for std::thread::hardware_concurrency

#include "core_threadpool.hpp"
#include "core_errors.hpp"

// ── 缓存分块大小 ─────────────────────────────────────────────────────────────
// 64×64×8 = 32 KB，安全装入大多数 CPU 的 L1 缓存
// 矩阵乘法、转置等所有分块操作共用此值
// 修改时需同步评估 b_block 栈占用（BLOCK_SIZE² × 8 字节）
namespace nn
{
    // ── 标量类型（float 加速 / double 精度） ─────────────────────────
    using Scalar = float;

    inline constexpr std::size_t BLOCK_SIZE = 64;
    static_assert(BLOCK_SIZE * BLOCK_SIZE * sizeof(Scalar) <= 65536,
                  "BLOCK_SIZE too large: b_block would exceed 64KB stack budget");
    // ── 数值常量 ─────────────────────────────────────────────────────────────
    // 使用 constexpr 避免运行时计算
    inline constexpr Scalar EPSILON = 1e-6f;

    // ── 并行阈值 ─────────────────────────────────────────────────────────────
    // 元素数 >= 此值时启用线程池并行；低于则串行执行（避免调度开销）。
    // 线程池调度开销约 50-200μs（Windows mutex+cv），仅当计算量足够时并行才有收益。
    //
    // bench_thresholds 实测结果（32 核 CPU, Release -O3, 2026-07-25）：
    //   元素数   串行μs  并行μs  加速比   建议
    //   16384     9.6    9.8     0.98x   无差别
    //   32768    18.8   19.9     0.94x   无差别
    //   65536    38.7  121.7     0.32x   串行优（调度开销显现）
    //   131072   76.4  182.8     0.42x   串行优
    //   262144  151.6  196.0     0.77x   串行优
    //   524288  308.7  201.5     1.53x   ← 首次稳定 > 1.5x
    //   1048576 620.3  186.7     3.32x   并行优（稳定）
    //   2097152 1239.5 217.3     5.70x   并行优
    //   4194304 2657.3 340.6     7.80x   并行优
    // 取 524288 (512K) 作为阈值：在加速比首次稳定 > 1.5x 的位置。
    // 注：1024-4096 元素数测出的高加速比是亚微秒级测量噪声，不可靠。
    inline constexpr std::size_t PARALLEL_THRESHOLD = 524288;

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

#endif // NN_CORE_CONFIG_HPP