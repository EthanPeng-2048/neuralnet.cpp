#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <cstddef>
#include <execution>
#include <expected>
#include <iterator>  // for std::distance
#include <string>
#include <thread>    // for std::thread::hardware_concurrency
#include "thread_pool.hpp"

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
    // ── 错误类型（C++23 std::expected）─────────────────────────────────────
    // 所有公共 API 使用 Result<T> 返回错误，不抛异常。
    struct Error {
        std::string message;
    };
    template <typename T>
    using Result = std::expected<T, Error>;

    inline constexpr std::size_t BLOCK_SIZE = 64;
    static_assert(BLOCK_SIZE * BLOCK_SIZE * sizeof(double) <= 65536,
                  "BLOCK_SIZE too large: b_block would exceed 64KB stack budget");
    // ── 数值常量 ─────────────────────────────────────────────────────────────
    // 使用 constexpr 避免运行时计算
    inline constexpr double EPSILON = 1e-8;
    
    // ── 智能执行策略 ─────────────────────────────────────────────────────────
    // 根据元素数量自动选择串行或使用全局线程池并行。
    // 线程池采用 latch + 调用者参与设计：一次批量入队 + 原子计数器，
    // 消除了旧版 N×submit + N×future 的堆分配和同步开销。
    struct SmartPolicy {
        // 阈值 = 每核最小分块 × 2，确保至少有 2 个分块可并行。
        // chunk_count() 已自动按核心数缩放（min(cores, total/4096)），
        // 阈值只需设最低门槛，核心利用随数据量自然增长。
        inline static constexpr std::size_t PARALLEL_THRESHOLD = 4096 * 2;  // 8192
        
        // for_each 版本
        template<typename Iterator, typename Func>
        static void for_each(Iterator first, Iterator last, Func&& func) {
            const auto n = static_cast<std::size_t>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                global_thread_pool().parallel_for_each(first, last, std::forward<Func>(func));
            } else {
                for (auto it = first; it != last; ++it)
                    func(*it);
            }
        }
        
        // transform 版本（一元）
        template<typename InputIt, typename OutputIt, typename UnaryOp>
        static void transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op) {
            const auto n = static_cast<std::size_t>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                global_thread_pool().parallel_transform(first, last, d_first, std::forward<UnaryOp>(op));
            } else {
                std::transform(first, last, d_first, std::forward<UnaryOp>(op));
            }
        }
        
        // transform 版本（二元）
        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        static void transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp&& op) {
            const auto n = static_cast<std::size_t>(std::distance(first1, last1));
            if (n >= PARALLEL_THRESHOLD) {
                global_thread_pool().parallel_transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
            } else {
                std::transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
            }
        }
        
        // transform_reduce 版本（一元 transform）
        template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt first, InputIt last, T init, BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            const auto n = static_cast<std::size_t>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                return global_thread_pool().parallel_transform_reduce(
                    first, last, init,
                    std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            } else {
                return std::transform_reduce(std::execution::seq, first, last, init, 
                                            std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            }
        }
        
        // transform_reduce 版本（二元 transform，两个输入范围）
        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            const auto n = static_cast<std::size_t>(std::distance(first1, last1));
            if (n >= PARALLEL_THRESHOLD) {
                return global_thread_pool().parallel_transform_reduce(first1, last1, first2, init,
                                                                    std::forward<BinaryOp>(reduce_op),
                                                                    std::forward<UnaryOp>(transform_op));
            } else {
                return std::transform_reduce(std::execution::seq, first1, last1, first2, init,
                                            std::forward<BinaryOp>(reduce_op),
                                            std::forward<UnaryOp>(transform_op));
            }
        }
    };
} // namespace nn

#endif // NN_CONFIG_HPP