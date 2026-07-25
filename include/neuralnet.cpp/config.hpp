#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <cstddef>
#include <execution>
#include <iterator>  // for std::distance
#include <thread>    // for std::thread::hardware_concurrency

#include "core_threadpool.hpp"
#include "core_errors.hpp"

// ── 执行策略 ────────────────────────────────────────────────────────────────
// 默认并行+向量化；编译时可通过 -DNN_EXEC_POLICY=std::execution::seq 覆盖
#ifndef NN_EXEC_POLICY
#define NN_EXEC_POLICY std::execution::par_unseq
#endif

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
    inline constexpr std::size_t PARALLEL_THRESHOLD = 131072;

    // ── 顶层并行算法函数 ─────────────────────────────────────────────────────
    // 用法与 std::for_each / std::transform 完全一致，内部自动选择串行/并行。
    // 替代 std::execution::par 的线程创建开销，也替代手写 for 循环。

    // for_each
    template<typename Iterator, typename Func>
    inline void for_each(Iterator first, Iterator last, Func&& func)
    {
        const auto n = static_cast<std::size_t>(std::distance(first, last));
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
        const auto n = static_cast<std::size_t>(std::distance(first, last));
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
        const auto n = static_cast<std::size_t>(std::distance(first1, last1));
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
        const auto n = static_cast<std::size_t>(std::distance(first, last));
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
        const auto n = static_cast<std::size_t>(std::distance(first1, last1));
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

    // ── 智能执行策略（兼容旧代码，全部转发到顶层 nn:: 函数）───────────────
    // 新代码请直接使用 nn::for_each / nn::transform 等顶层函数。
    struct SmartPolicy {
        // 阈值（与顶层 PARALLEL_THRESHOLD 保持一致，供旧代码读取）
        inline static constexpr std::size_t PARALLEL_THRESHOLD = nn::PARALLEL_THRESHOLD;

        // GPU 加速阈值：矩阵面积超过此值时自动走 GPU（默认 64×64 = 4096 元素）
        inline static constexpr std::size_t GPU_THRESHOLD = 64 * 64;

        // 是否启用 GPU 后端（运行时开关，默认关闭）
        inline static bool gpu_enabled = false;

        // GPU 操作计数（性能统计）
        inline static std::atomic<uint64_t> gpu_matmul_count{0};
        inline static std::atomic<uint64_t> cpu_matmul_count{0};

        // 全部转发到 nn:: 顶层函数
        template<typename Iterator, typename Func>
        static void for_each(Iterator first, Iterator last, Func&& func) {
            nn::for_each(first, last, std::forward<Func>(func));
        }

        template<typename InputIt, typename OutputIt, typename UnaryOp>
        static void transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op) {
            nn::transform(first, last, d_first, std::forward<UnaryOp>(op));
        }

        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        static void transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                              OutputIt d_first, BinaryOp&& op) {
            nn::transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
        }

        template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt first, InputIt last, T init,
                                  BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            return nn::transform_reduce(first, last, init,
                                        std::forward<BinaryOp>(reduce_op),
                                        std::forward<UnaryOp>(transform_op));
        }

        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                  T init, BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            return nn::transform_reduce(first1, last1, first2, init,
                                        std::forward<BinaryOp>(reduce_op),
                                        std::forward<UnaryOp>(transform_op));
        }

        template<typename Iterator, typename Func>
        static void parallel_for_blocks(Iterator first, Iterator last, Func&& func) {
            nn::parallel_for_blocks(first, last, std::forward<Func>(func));
        }

        template<typename Func>
        static void parallel_for_samples(std::size_t num_samples, Func&& func) {
            nn::parallel_for_samples(num_samples, std::forward<Func>(func));
        }
    };
} // namespace nn

#endif // NN_CONFIG_HPP