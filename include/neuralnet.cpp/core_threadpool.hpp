#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <algorithm>  // for std::transform (serial fallback)
#include <numeric>    // for std::transform_reduce (serial fallback)
#include <iterator>   // for std::distance
#include <type_traits>

#include "core_assert.hpp"

// ── CPU pause 自旋原语：GCC/Clang 用 __builtin_ia32_pause，MSVC 用 _mm_pause ──
#if defined(_MSC_VER)
#include <immintrin.h>
#define NN_CPU_PAUSE() _mm_pause()
#else
#define NN_CPU_PAUSE() __builtin_ia32_pause()
#endif

namespace nn
{
    // ── 简易线程池（latch 零分配设计）──────────────────────────────────────
    // 核心改进：将原来的 "N 次 submit + N 个 future + N 次加锁" 替换为
    //   "1 次批量入队 + 1 个原子计数器 + 调用者参与处理"
    // 消除每分块一次 shared_ptr<packaged_task> 堆分配和 future 同步开销
    class ThreadPool
    {
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;

        std::mutex queue_mutex_;
        std::condition_variable condition_;
        std::atomic<bool> stop_{false};

    public:
        // ── worker 线程索引（每个 worker 在构造时绑定唯一 ID） ──────
        // 用于 parallel_for_each 内部识别当前执行线程，支持 per-worker 状态
        // （如 BPE 训练的并行合并 delta 映射）。
        static thread_local std::size_t tl_worker_index;

        explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
        {
            if (num_threads == 0) num_threads = 1;
            workers_.reserve(num_threads);
            for (std::size_t i = 0; i < num_threads; ++i)
            {
                workers_.emplace_back([this, i]
                {
                    tl_worker_index = i;
                    for (;;)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock lock(queue_mutex_);
                            condition_.wait(lock, [this]
                            {
                                return stop_.load(std::memory_order_acquire) || !tasks_.empty();
                            });
                            if (stop_.load(std::memory_order_acquire) && tasks_.empty())
                                return;
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }

        // ── 通用单任务提交（submit）已删除：全库无调用方（审查 P1-3），
        //    且每次调用 make_shared<packaged_task> 堆分配，违背本池
        //    "零分配 latch" 设计。需要 future 语义时应在调用方分块后
        //    用 parallel_* 系列原语。 ──────────────────────────────────

    private:
        // ── 分块辅助：计算合理的分块数 ──────────────────────────────────
        // 每分块至少 1024 个元素（Scalar=float 时约 4 KB），可装入 L1 缓存。
        // 4096→1024：降低阈值使 MNIST 小隐藏层（64×batch）也能触发多核并行。
        [[nodiscard]] std::size_t chunk_count(std::size_t total) const noexcept
        {
            constexpr std::size_t MIN_CHUNK = 1024;
            const auto nw = workers_.size();
            if (nw <= 1 || total < MIN_CHUNK * 2)
                return 1;
            auto n = std::min(nw, total / MIN_CHUNK);
            return n < 1 ? 1 : n;
        }

        // ── 分片完成：递减 latch；最后一个完成者唤醒 cv 等待者 ──────────
        // 等待方（wait_for_latch 的无超时 wait / worker 主循环）都挂在
        // condition_ 上，latch 归零必须通知，否则等待者会永久睡眠。
        // 用 notify_all：等待者可能同时包含调用者与空闲 worker。
        //
        // ⚠ 必须持 queue_mutex_ 再 notify（丢唤醒 / lost wakeup 修复）：
        // wait_for_latch 用 wait(lock, [&]{ return latch == 0; }) 等待，谓词
        // 在同一把锁下求值。若归零+notify 不持锁，就会出现
        //   等待者判谓词=false → 通知者置零并 notify（此刻无注册等待者，信号丢失）
        //   → 等待者真正进入 wait() → 永久阻塞
        // 空并行区压测（build/perfprobe/probe_pool2.cpp，32 核）实测：不持锁
        // 数万次 region 内必死锁；持锁后 3×50000 次稳定通过。
        void finish_chunk(std::atomic<int>& latch) noexcept
        {
            if (latch.fetch_sub(1, std::memory_order_release) == 1)
            {
                std::lock_guard lock(queue_mutex_);
                condition_.notify_all();
            }
        }

        // ── work-stealing 等待：调用者不空转，帮忙处理队列任务 ─────────
        // 优化（依据性能审查报告）：
        //   - 旧实现：spin 64 次 + yield，64 次 spin 中反复原子读取消耗电量
        //     CPU 占用率显示 100% 但实际有效计算比例低（调用者空转）
        //   - 新实现：
        //     1) 短自旋（16 次 pause）快速检测 latch 归零——典型情况无 yield 开销
        //     2) 自旋失败后 work-steal：尝试从队列取任务执行（参与计算）
        //     3) 队列为空时阻塞等待 condition_variable（无超时轮询），
        //        由 finish_chunk 在 latch 归零时通知唤醒
        // 线程索引约定：调用者线程帮忙执行偷来的任务时，临时将
        // tl_worker_index 置为 workers_.size()（调用者 slot），避免
        // parallel_for_each_indexed 的任务体与 worker 0 撞同一 slot；
        // 执行完恢复原值（嵌套并行时 worker 保留自己的索引）。
        void wait_for_latch(std::atomic<int>& latch)
        {
            // 阶段 1：短自旋（16 次 pause）——应对 latch 即将归零的快路径
            for (int spin = 0; spin < 16; ++spin)
            {
                if (latch.load(std::memory_order_acquire) == 0)
                    return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                NN_CPU_PAUSE();
#endif
            }

            // 阶段 2：work-stealing + cv 等待
            while (latch.load(std::memory_order_acquire) > 0)
            {
                std::function<void()> task;
                {
                    std::unique_lock lock(queue_mutex_);
                    if (!tasks_.empty())
                    {
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                }
                if (task)
                {
                    // 借用"调用者 slot 索引"执行，执行完恢复原值
                    const std::size_t saved = tl_worker_index;
                    tl_worker_index = workers_.size();
                    task();
                    tl_worker_index = saved;

                    // 执行完一个任务后回到阶段 1 短自旋
                    for (int spin = 0; spin < 16; ++spin)
                    {
                        if (latch.load(std::memory_order_acquire) == 0)
                            return;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                        NN_CPU_PAUSE();
#endif
                    }
                }
                else
                {
                    // 队列为空：阻塞等待（无超时）；
                    // 完成通知来自 finish_chunk，虚假唤醒由谓词兜底
                    std::unique_lock lock(queue_mutex_);
                    condition_.wait(lock,
                        [&latch]() {
                            return latch.load(std::memory_order_acquire) == 0;
                        });
                }
            }
        }

    public:
        // ── 并行 for_each（latch + 调用者参与） ─────────────────────────
        // 与旧版相比：零 future 分配、一次加锁入队、调用者不空等
        template<typename Iterator, typename Func>
        void parallel_for_each(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (total == 0) return;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                for (auto it = first; it != last; ++it)
                    func(*it);
                return;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;

            // 原子计数器：初始值 = n_chunks
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            // 将前 n_chunks-1 个分块批量入队（仅一次加锁）；
            // 每入队一个任务 notify_one，避免高核机器上唤醒全部 worker
            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([this, beg, end, &func, &latch]()
                    {
                        for (auto it = beg; it != end; ++it)
                            func(*it);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            // 调用者处理最后一个分块（不经过队列，零分配）
            {
                const std::size_t c = n_chunks - 1;
                const std::size_t len = base + (c < rem ? 1 : 0);
                const std::size_t off = total - len;
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                for (auto it = beg; it != last; ++it)
                    func(*it);
                finish_chunk(latch);
            }

            // work-stealing 等待：调用者帮忙处理队列任务而非空转
            wait_for_latch(latch);
        }

        // ── 带 worker 索引的并行 for_each ────────────────────────────
        // 与 parallel_for_each 相同的分区和调度策略，但回调额外接收
        // worker_index 参数（worker 线程 = 0..size()-1，调用者线程 = size()）。
        // 用于需要 per-worker 状态（如 BPE 训练的 local delta）的场景，
        // 调用者可安全使用 index = n_threads 作为独立 slot，避免与 worker 0 冲突。
        template<typename Iterator, typename Func>
        void parallel_for_each_indexed(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (total == 0) return;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                for (auto it = first; it != last; ++it)
                    func(*it, 0);  // 单线程：index = 0
                return;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;

            std::atomic<int> latch{static_cast<int>(n_chunks)};

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([this, beg, end, &func, &latch]()
                    {
                        for (auto it = beg; it != end; ++it)
                            func(*it, tl_worker_index);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            // 调用者处理最后一个分块，index = workers_.size()（不与任何 worker 冲突）
            {
                const std::size_t c = n_chunks - 1;
                const std::size_t len = base + (c < rem ? 1 : 0);
                const std::size_t off = total - len;
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                for (auto it = beg; it != last; ++it)
                    func(*it, workers_.size());
                finish_chunk(latch);
            }

            wait_for_latch(latch);
        }

        // ── 块级并行 for_each（供矩阵乘法/转置的分块循环使用）─────────
        // 与普通 parallel_for_each 不同：每个"元素"本身就是一次重量级计算
        // （如 64×64×K 的矩阵乘法分块），因此不适用 MIN_CHUNK 保护。
        // 直接按块数分给各线程，1 块 = 1 分片。
        template<typename Iterator, typename Func>
        void parallel_for_blocks(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (total <= 1)
            {
                for (auto it = first; it != last; ++it)
                    func(*it);
                return;
            }

            const auto n_chunks = std::min(workers_.size(), total);
            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([this, beg, end, &func, &latch]()
                    {
                        for (auto it = beg; it != end; ++it)
                            func(*it);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                for (auto it = beg; it != last; ++it)
                    func(*it);
                finish_chunk(latch);
            }

            wait_for_latch(latch);
        }

        // ── 并行 for_samples（独立样本级并行）───────────────────────────
        // 与 for_each（按元素分块）不同：每个"样本"包含一个完整的计算子任务
        // （如矩阵乘法、前向传播），样本间完全独立、无数据竞争。
        // 用于 GPT/Transformer batch 维度的并行：1 样本 = 1 分片。
        template<typename Func>
        void parallel_for_samples(std::size_t num_samples, Func&& func)
        {
            if (num_samples <= 1)
            {
                for (std::size_t i = 0; i < num_samples; ++i)
                    func(i);
                return;
            }

            const auto n_chunks = std::min(workers_.size(), num_samples);
            const std::size_t base = num_samples / n_chunks;
            const std::size_t rem  = num_samples % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    const std::size_t start = off;
                    const std::size_t end   = off + len;
                    off += len;

                    tasks_.emplace([this, start, end, &func, &latch]()
                    {
                        for (std::size_t i = start; i < end; ++i)
                            func(i);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            // 调用者处理最后一个分片
            {
                const std::size_t c = n_chunks - 1;
                const std::size_t start = num_samples - (base + (c < rem ? 1 : 0));
                for (std::size_t i = start; i < num_samples; ++i)
                    func(i);
                finish_chunk(latch);
            }

            wait_for_latch(latch);
        }

        // ── 并行 transform（一元）───────────────────────────────────────
        template<typename InputIt, typename OutputIt, typename UnaryOp>
        void parallel_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (total == 0) return;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                std::transform(first, last, d_first, std::forward<UnaryOp>(op));
                return;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto in_beg = first;
                    auto out_beg = d_first;
                    std::ranges::advance(in_beg,  static_cast<std::ptrdiff_t>(off));
                    std::ranges::advance(out_beg, static_cast<std::ptrdiff_t>(off));
                    auto in_end = in_beg;
                    std::ranges::advance(in_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([this, in_beg, in_end, out_beg, &op, &latch]()
                    {
                        auto in = in_beg;
                        auto out = out_beg;
                        for (; in != in_end; ++in, ++out)
                            *out = op(*in);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto in_beg = first;
                auto out_beg = d_first;
                std::ranges::advance(in_beg,  static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(out_beg, static_cast<std::ptrdiff_t>(off));
                for (; in_beg != last; ++in_beg, ++out_beg)
                    *out_beg = op(*in_beg);
                finish_chunk(latch);
            }

            wait_for_latch(latch);
        }

        // ── 并行 transform（二元）───────────────────────────────────────
        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        void parallel_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                OutputIt d_first, BinaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first1, last1));
            if (total == 0) return;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                std::transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
                return;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto i1 = first1;
                    auto i2 = first2;
                    auto o = d_first;
                    std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                    std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                    std::ranges::advance(o,  static_cast<std::ptrdiff_t>(off));
                    auto i1_end = i1;
                    std::ranges::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([this, i1, i1_end, i2, o, &op, &latch]()
                    {
                        auto it1 = i1;
                        auto it2 = i2;
                        auto out = o;
                        for (; it1 != i1_end; ++it1, ++it2, ++out)
                            *out = op(*it1, *it2);
                        finish_chunk(latch);
                    });
                    condition_.notify_one();
                }
            }

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto i1 = first1;
                auto i2 = first2;
                auto o = d_first;
                std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(o,  static_cast<std::ptrdiff_t>(off));
                for (; i1 != last1; ++i1, ++i2, ++o)
                    *o = op(*i1, *i2);
                finish_chunk(latch);
            }

            wait_for_latch(latch);
        }

        // ── 归约分块（确定性契约，铁律 8）──────────────────────────────────
        // 分段边界必须**只由 total 决定**，与 worker 数 / 机器核数无关：
        // 旧实现 n_chunks = chunk_count(total) 依赖 workers_.size()，
        // 同一输入在 1-worker 与 N-worker 下走不同折叠结构 → 浮点非结合律
        // 导致字节不一致（跨机也不一致）。修复后 1-worker 与 N-worker、
        // 任何机器都走完全相同的分段，部分和按**固定块下标**存放与合并，
        // 线程只决定"谁算哪块"，不参与边界与合并顺序。
        //
        // 借鉴 ATen::parallel_reduce（torch/include/ATen/Parallel-inl.h）：
        // 其部分和按 results[tid] 存放、边界由 get_num_threads() 决定 →
        // 跨线程数不可复现（PyTorch 亦声明 CPU 归约不保证跨线程数一致）；
        // 本库铁律 8 要求逐字节一致，故与 ATen 反向：边界固定、弃用
        // per-thread 部分和下标。
        //
        // 分段语义（串/并行共用的唯一定义）：
        //   块 0 以 init 为种子，块 c>0 以块内首元素为种子（init 恰好计入
        //   一次，旧实现每块都加 init、合并时再加一次，init≠0 时数学错误）；
        //   块内从左到右折叠；结果按块下标升序合并。
        //   n_chunks==1 时退化为纯左折叠，与 n < PARALLEL_THRESHOLD 的
        //   串行路径（core_config::transform_reduce）逐字节一致。
        // 块大小下限保证 n_chunks≥2 时每块至少 MIN_CHUNK 个元素（非空，
        // c>0 块取首元素作种子安全）；MAX_CHUNKS 限制任务/partials 数量，
        // 任务开销 ∝ 块数，实际调用方（text_train 梯度统计，log 间隔一次）
        // 对任务数不敏感。
        [[nodiscard]] static std::size_t reduce_chunk_count(std::size_t total) noexcept
        {
            constexpr std::size_t MIN_CHUNK  = 16384;   // 元素数（64KB f32）
            constexpr std::size_t MAX_CHUNKS = 512;
            if (total < MIN_CHUNK * 2) return 1;
            return std::min(MAX_CHUNKS, total / MIN_CHUNK);
        }

        // ── 并行 transform_reduce（一元 transform）──────────────────────
        template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
        T parallel_transform_reduce(InputIt first, InputIt last, T init,
                                    BinaryOp&& reduce_op, UnaryOp&& transform_op)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (total == 0) return init;

            const auto n_chunks = reduce_chunk_count(total);
            if (n_chunks == 1)
            {
                // 纯左折叠（= 单块语义）。显式循环而非 std::transform_reduce：
                // 后者归约顺序标准未定义，不能作确定性基准。
                T local = init;
                for (auto it = first; it != last; ++it)
                    local = reduce_op(local, transform_op(*it));
                return local;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            // 部分和按固定块下标存放（与调度无关 → 确定性）
            std::vector<T> partials(n_chunks);

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::ranges::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    if (c == 0)
                    {
                        // 块 0：以 init 为种子（init 在整个归约恰好计入一次）
                        tasks_.emplace([this, beg, end, &reduce_op, &transform_op, &partials, &latch, init]()
                        {
                            T local = init;
                            for (auto it = beg; it != end; ++it)
                                local = reduce_op(local, transform_op(*it));
                            partials[0] = std::move(local);
                            finish_chunk(latch);
                        });
                    }
                    else
                    {
                        // 块 c>0：以块内首元素为种子（len ≥ MIN_CHUNK ≥ 1 保证非空）
                        tasks_.emplace([this, beg, end, &reduce_op, &transform_op, &partials, &latch, c]()
                        {
                            auto it = beg;
                            T local = transform_op(*it);
                            for (++it; it != end; ++it)
                                local = reduce_op(local, transform_op(*it));
                            partials[c] = std::move(local);
                            finish_chunk(latch);
                        });
                    }
                    condition_.notify_one();
                }
            }

            {
                // 末段由调用者执行（n_chunks ≥ 2 ⇒ 末块下标 ≥ 1 ⇒ 首元素种子）
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto beg = first;
                std::ranges::advance(beg, static_cast<std::ptrdiff_t>(off));
                auto it = beg;
                T local = transform_op(*it);
                for (++it; it != last; ++it)
                    local = reduce_op(local, transform_op(*it));
                partials[c] = std::move(local);
                finish_chunk(latch);
            }

            wait_for_latch(latch);

            T result = std::move(partials[0]);
            for (std::size_t c = 1; c < n_chunks; ++c)
                result = reduce_op(result, partials[c]);
            return result;
        }

        // ── 并行 transform_reduce（二元输入范围）────────────────────────
        // 与一元版共享同一确定性契约（铁律 8）：分段边界只由 total 决定、
        // 块 0 以 init 为种子、其余块以首元素为种子、按块下标升序合并。
        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        T parallel_transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                    T init, BinaryOp&& reduce_op, UnaryOp&& transform_op)
        {
            const auto total = static_cast<std::size_t>(std::ranges::distance(first1, last1));
            if (total == 0) return init;

            const std::size_t n_chunks = reduce_chunk_count(total);
            if (n_chunks == 1)
            {
                T local = init;
                auto i1 = first1;
                auto i2 = first2;
                for (; i1 != last1; ++i1, ++i2)
                    local = reduce_op(local, transform_op(*i1, *i2));
                return local;
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};
            // 部分和按固定块下标存放（边界 = f(total)，与调度无关）
            std::vector<T> partials(n_chunks);

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto i1 = first1;
                    auto i2 = first2;
                    std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                    std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                    auto i1_end = i1;
                    std::ranges::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    if (c == 0)
                    {
                        // 块 0：以 init 为种子（init 在整个归约恰好计入一次）
                        tasks_.emplace([this, i1, i1_end, i2, &reduce_op, &transform_op, &partials, &latch, init]()
                        {
                            T local = init;
                            auto it1 = i1;
                            auto it2 = i2;
                            for (; it1 != i1_end; ++it1, ++it2)
                                local = reduce_op(local, transform_op(*it1, *it2));
                            partials[0] = std::move(local);
                            finish_chunk(latch);
                        });
                    }
                    else
                    {
                        // 块 c>0：以块内首元素为种子（len ≥ MIN_CHUNK ≥ 1 保证非空）
                        tasks_.emplace([this, i1, i1_end, i2, &reduce_op, &transform_op, &partials, &latch, c]()
                        {
                            auto it1 = i1;
                            auto it2 = i2;
                            T local = transform_op(*it1, *it2);
                            for (++it1, ++it2; it1 != i1_end; ++it1, ++it2)
                                local = reduce_op(local, transform_op(*it1, *it2));
                            partials[c] = std::move(local);
                            finish_chunk(latch);
                        });
                    }
                    condition_.notify_one();
                }
            }

            {
                // 末段由调用者执行（n_chunks ≥ 2 ⇒ 末块下标 ≥ 1 ⇒ 首元素种子）
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto i1 = first1;
                auto i2 = first2;
                std::ranges::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::ranges::advance(i2, static_cast<std::ptrdiff_t>(off));
                T local = transform_op(*i1, *i2);
                for (++i1, ++i2; i1 != last1; ++i1, ++i2)
                    local = reduce_op(local, transform_op(*i1, *i2));
                partials[c] = std::move(local);
                finish_chunk(latch);
            }

            wait_for_latch(latch);

            T result = std::move(partials[0]);
            for (std::size_t c = 1; c < n_chunks; ++c)
                result = reduce_op(result, partials[c]);
            return result;
        }

        // ── 析构 ────────────────────────────────────────────────────────
        ~ThreadPool()
        {
            stop_.store(true, std::memory_order_release);
            condition_.notify_all();
            for (auto& worker : workers_)
            {
                if (worker.joinable())
                    worker.join();
            }
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

        // 返回当前线程的 worker 索引（0..size()-1）。
        // 非 worker 线程默认返回 0；当调用者线程在 wait_for_latch 中帮忙
        // 执行偷来的任务、或处理 parallel_for_each_indexed 的调用者分片时，
        // 返回 size()（调用者 slot，不与任何 worker 冲突）。
        [[nodiscard]] static std::size_t worker_index() noexcept { return tl_worker_index; }
    };

    // thread_local 定义（inline，ODR-safe）
    inline thread_local std::size_t ThreadPool::tl_worker_index = 0;

    // ── 全局线程池单例 ─────────────────────────────────────────────────────
    inline ThreadPool& global_thread_pool()
    {
        static ThreadPool pool{std::thread::hardware_concurrency()};
        return pool;
    }

} // namespace nn

