#ifndef NN_CONFIG_HPP
#define NN_CONFIG_HPP

#include <atomic>
#include <cstddef>
#include <execution>
#include <expected>
#include <iostream>
#include <iterator>  // for std::distance
#include <string>
#include <thread>    // for std::thread::hardware_concurrency

#include "core/thread_pool.hpp"
#include "core/errors.hpp"

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
    
    // ── 智能执行策略 ─────────────────────────────────────────────────────────
    // 根据元素数量自动选择串行或使用全局线程池并行。
    // 线程池采用 latch + 调用者参与设计：一次批量入队 + 原子计数器，
    // 消除了旧版 N×submit + N×future 的堆分配和同步开销。
    struct SmartPolicy {
        // 阈值 = 1024 元素：即使 MNIST 小隐藏层（64×64=4096）也能触发并行。
        // chunk_count() 已自动按核心数缩放（min(cores, total/1024)），
        // 阈值只需设最低门槛，核心利用随数据量自然增长。
        inline static constexpr std::size_t PARALLEL_THRESHOLD = 1024;
        
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

        // ── 块级并行（供矩阵乘法/转置的分块循环使用）─────────────────
        // 与 for_each 不同：每个"元素"本身就是重量级计算（如 64×64×K 矩阵乘），
        // 因此不适用 PARALLEL_THRESHOLD 保护，直接按"块数"分发。
        template<typename Iterator, typename Func>
        static void parallel_for_blocks(Iterator first, Iterator last, Func&& func) {
            global_thread_pool().parallel_for_blocks(first, last, std::forward<Func>(func));
        }

        // ── 样本级并行（供 batch 维度并行使用）─────────────────────────
        // 每个"样本"包含完整的前向/反向传播子任务，样本间完全独立。
        // 用于 GPT/Transformer 的 batch 循环并行化。
        template<typename Func>
        static void parallel_for_samples(std::size_t num_samples, Func&& func) {
            if (num_samples <= 1) {
                for (std::size_t i = 0; i < num_samples; ++i)
                    func(i);
            } else {
                global_thread_pool().parallel_for_samples(num_samples, std::forward<Func>(func));
            }
        }

        // ── GPU 加速配置 ─────────────────────────────────────────────────
#ifdef NN_HAS_VULKAN
        // GPU 加速阈值：矩阵输出面积（M*N）超过此值时自动走 GPU。
        // 设为 1024（32×32）以覆盖 per-sample transformer 操作（M*N=2048），
        // 同时过滤过小矩阵避免 GPU kernel 启动开销超过计算收益。
        inline static constexpr std::size_t GPU_THRESHOLD = 1024;

        // 是否启用 GPU 后端（运行时开关，默认关闭）
        // 用户需调用 nn::GpuBackend::instance().initialize() 初始化后设为 true
        inline static bool gpu_enabled = false;

        // GPU 调度统计（用于诊断）
        inline static std::atomic<std::size_t> gpu_matmul_count{0};
        inline static std::atomic<std::size_t> cpu_matmul_count{0};

        static void reset_matmul_stats() noexcept {
            gpu_matmul_count.store(0);
            cpu_matmul_count.store(0);
        }

        static void print_matmul_stats() {
            auto gpu = gpu_matmul_count.load();
            auto cpu = cpu_matmul_count.load();
            auto total = gpu + cpu;
            std::cout << "[GPU Stats] matmul 总计: " << total
                      << "，GPU: " << gpu
                      << "，CPU: " << cpu;
            if (total > 0)
                std::cout << " (" << (100.0 * gpu / total) << "% GPU)";
            std::cout << "\n";
        }
#endif
    };
} // namespace nn

#endif // NN_CONFIG_HPP