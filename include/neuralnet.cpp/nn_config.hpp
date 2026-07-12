#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <cstddef>
#include <execution>
#include <iterator>  // for std::distance

// ── 执行策略 ────────────────────────────────────────────────────────────────
// 默认并行+向量化；编译时可通过 -DNN_EXEC_POLICY=std::execution::seq 覆盖
#ifndef NN_EXEC_POLICY
#define NN_EXEC_POLICY std::execution::par_unseq
#endif

// ── 缓存分块大小 ─────────────────────────────────────────────────────────────
// 32×32×8 = 8 KB，安全装入大多数 CPU 的 L1 缓存
// 矩阵乘法、转置等所有分块操作共用此值
// 修改时需同步评估 b_block 栈占用（BLOCK_SIZE² × 8 字节）
namespace nn
{
    inline constexpr std::size_t BLOCK_SIZE = 64;
    
    // ── 数值常量 ─────────────────────────────────────────────────────────────
    // 使用 constexpr 避免运行时计算
    inline constexpr double EPSILON = 1e-8;
    
    // ── 智能执行策略 ─────────────────────────────────────────────────────────
    // 根据元素数量自动选择串行或并行策略，避免小矩阵的线程池调度开销
    struct SmartPolicy {
        static constexpr long long PARALLEL_THRESHOLD = 100000; // 100K元素
        
        // for_each 版本
        template<typename Iterator, typename Func>
        static void for_each(Iterator first, Iterator last, Func&& func) {
            const auto n = static_cast<long long>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                std::for_each(std::execution::par_unseq, first, last, std::forward<Func>(func));
            } else {
                std::for_each(std::execution::seq, first, last, std::forward<Func>(func));
            }
        }
        
        // transform 版本（一元）
        template<typename InputIt, typename OutputIt, typename UnaryOp>
        static void transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op) {
            const auto n = static_cast<long long>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                std::transform(std::execution::par_unseq, first, last, d_first, std::forward<UnaryOp>(op));
            } else {
                std::transform(std::execution::seq, first, last, d_first, std::forward<UnaryOp>(op));
            }
        }
        
        // transform 版本（二元）
        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        static void transform(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp&& op) {
            const auto n = static_cast<long long>(std::distance(first1, last1));
            if (n >= PARALLEL_THRESHOLD) {
                std::transform(std::execution::par_unseq, first1, last1, first2, d_first, std::forward<BinaryOp>(op));
            } else {
                std::transform(std::execution::seq, first1, last1, first2, d_first, std::forward<BinaryOp>(op));
            }
        }
        
        // transform_reduce 版本（一元 transform）
        template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt first, InputIt last, T init, BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            const auto n = static_cast<long long>(std::distance(first, last));
            if (n >= PARALLEL_THRESHOLD) {
                return std::transform_reduce(std::execution::par_unseq, first, last, init, 
                                            std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            } else {
                return std::transform_reduce(std::execution::seq, first, last, init, 
                                            std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            }
        }
        
        // transform_reduce 版本（二元 transform，两个输入范围）
        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        static T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp&& reduce_op, UnaryOp&& transform_op) {
            const auto n = static_cast<long long>(std::distance(first1, last1));
            if (n >= PARALLEL_THRESHOLD) {
                return std::transform_reduce(std::execution::par_unseq, first1, last1, first2, init, 
                                            std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            } else {
                return std::transform_reduce(std::execution::seq, first1, last1, first2, init, 
                                            std::forward<BinaryOp>(reduce_op), std::forward<UnaryOp>(transform_op));
            }
        }
    };
} // namespace nn

#endif // NN_CONFIG_HPP