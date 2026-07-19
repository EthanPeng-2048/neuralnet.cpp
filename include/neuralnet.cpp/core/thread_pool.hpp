#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

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
        explicit ThreadPool(std::size_t num_threads = std::thread::hardware_concurrency())
        {
            if (num_threads == 0) num_threads = 1;
            workers_.reserve(num_threads);
            for (std::size_t i = 0; i < num_threads; ++i)
            {
                workers_.emplace_back([this]
                {
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

        // ── 通用单任务提交（保留给非性能敏感路径） ──────────────────────
        template<typename F, typename... Args>
        [[nodiscard]] auto submit(F&& f, Args&&... args)
            -> std::future<std::invoke_result_t<F, Args...>>
        {
            using return_type = std::invoke_result_t<F, Args...>;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> result = task->get_future();
            {
                std::unique_lock lock(queue_mutex_);
                assert(!stop_.load(std::memory_order_acquire) && "submit on stopped ThreadPool");
                tasks_.emplace([task]() { (*task)(); });
            }
            condition_.notify_one();
            return result;
        }

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

        // ── work-stealing 等待：调用者不空转，帮忙处理队列任务 ─────────
        void wait_for_latch(std::atomic<int>& latch)
        {
            while (latch.load(std::memory_order_acquire) > 0)
            {
                std::function<void()> task;
                {
                    std::lock_guard lock(queue_mutex_);
                    if (!tasks_.empty())
                    {
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                }
                if (task)
                    task();
                else
                    std::this_thread::yield();
            }
        }

    public:
        // ── 并行 for_each（latch + 调用者参与） ─────────────────────────
        // 与旧版相比：零 future 分配、一次加锁入队、调用者不空等
        template<typename Iterator, typename Func>
        void parallel_for_each(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
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

            // 将前 n_chunks-1 个分块批量入队（仅一次加锁）
            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([beg, end, &func, &latch]()
                    {
                        for (auto it = beg; it != end; ++it)
                            func(*it);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            // 调用者处理最后一个分块（不经过队列，零分配）
            {
                const std::size_t c = n_chunks - 1;
                const std::size_t len = base + (c < rem ? 1 : 0);
                const std::size_t off = total - len;
                auto beg = first;
                std::advance(beg, static_cast<std::ptrdiff_t>(off));
                for (auto it = beg; it != last; ++it)
                    func(*it);
                latch.fetch_sub(1, std::memory_order_release);
            }

            // work-stealing 等待：调用者帮忙处理队列任务而非空转
            wait_for_latch(latch);
        }

        // ── 块级并行 for_each（供矩阵乘法/转置的分块循环使用）─────────
        // 与普通 parallel_for_each 不同：每个"元素"本身就是一次重量级计算
        // （如 64×64×K 的矩阵乘法分块），因此不适用 MIN_CHUNK 保护。
        // 直接按块数分给各线程，1 块 = 1 分片。
        template<typename Iterator, typename Func>
        void parallel_for_blocks(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
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
                    std::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([beg, end, &func, &latch]()
                    {
                        for (auto it = beg; it != end; ++it)
                            func(*it);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto beg = first;
                std::advance(beg, static_cast<std::ptrdiff_t>(off));
                for (auto it = beg; it != last; ++it)
                    func(*it);
                latch.fetch_sub(1, std::memory_order_release);
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

                    tasks_.emplace([start, end, &func, &latch]()
                    {
                        for (std::size_t i = start; i < end; ++i)
                            func(i);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            // 调用者处理最后一个分片
            {
                const std::size_t c = n_chunks - 1;
                const std::size_t start = num_samples - (base + (c < rem ? 1 : 0));
                for (std::size_t i = start; i < num_samples; ++i)
                    func(i);
                latch.fetch_sub(1, std::memory_order_release);
            }

            wait_for_latch(latch);
        }

        // ── 并行 transform（一元）───────────────────────────────────────
        template<typename InputIt, typename OutputIt, typename UnaryOp>
        void parallel_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
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
                    std::advance(in_beg,  static_cast<std::ptrdiff_t>(off));
                    std::advance(out_beg, static_cast<std::ptrdiff_t>(off));
                    auto in_end = in_beg;
                    std::advance(in_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([in_beg, in_end, out_beg, &op, &latch]()
                    {
                        auto in = in_beg;
                        auto out = out_beg;
                        for (; in != in_end; ++in, ++out)
                            *out = op(*in);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto in_beg = first;
                auto out_beg = d_first;
                std::advance(in_beg,  static_cast<std::ptrdiff_t>(off));
                std::advance(out_beg, static_cast<std::ptrdiff_t>(off));
                for (; in_beg != last; ++in_beg, ++out_beg)
                    *out_beg = op(*in_beg);
                latch.fetch_sub(1, std::memory_order_release);
            }

            wait_for_latch(latch);
        }

        // ── 并行 transform（二元）───────────────────────────────────────
        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        void parallel_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                OutputIt d_first, BinaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first1, last1));
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
                    std::advance(i1, static_cast<std::ptrdiff_t>(off));
                    std::advance(i2, static_cast<std::ptrdiff_t>(off));
                    std::advance(o,  static_cast<std::ptrdiff_t>(off));
                    auto i1_end = i1;
                    std::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([i1, i1_end, i2, o, &op, &latch]()
                    {
                        auto it1 = i1;
                        auto it2 = i2;
                        auto out = o;
                        for (; it1 != i1_end; ++it1, ++it2, ++out)
                            *out = op(*it1, *it2);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto i1 = first1;
                auto i2 = first2;
                auto o = d_first;
                std::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::advance(i2, static_cast<std::ptrdiff_t>(off));
                std::advance(o,  static_cast<std::ptrdiff_t>(off));
                for (; i1 != last1; ++i1, ++i2, ++o)
                    *o = op(*i1, *i2);
                latch.fetch_sub(1, std::memory_order_release);
            }

            wait_for_latch(latch);
        }

        // ── 并行 transform_reduce（一元 transform）──────────────────────
        template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
        T parallel_transform_reduce(InputIt first, InputIt last, T init,
                                    BinaryOp&& reduce_op, UnaryOp&& transform_op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
            if (total == 0) return init;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                return std::transform_reduce(first, last, init,
                                            std::forward<BinaryOp>(reduce_op),
                                            std::forward<UnaryOp>(transform_op));
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};

            // 预分配结果数组（栈上小 vector），避免 future 堆分配
            std::vector<T> partials(n_chunks);

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto beg = first;
                    std::advance(beg, static_cast<std::ptrdiff_t>(off));
                    auto end = beg;
                    std::advance(end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([beg, end, &reduce_op, &transform_op, &partials, &latch, c]()
                    {
                        T local{};  // 要求 T{} 是 reduce_op 的单位元
                        for (auto it = beg; it != end; ++it)
                            local = reduce_op(local, transform_op(*it));
                        partials[c] = std::move(local);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto beg = first;
                std::advance(beg, static_cast<std::ptrdiff_t>(off));
                T local{};
                for (auto it = beg; it != last; ++it)
                    local = reduce_op(local, transform_op(*it));
                partials[c] = std::move(local);
                latch.fetch_sub(1, std::memory_order_release);
            }

            wait_for_latch(latch);

            T result = init;
            for (auto& p : partials)
                result = reduce_op(result, p);
            return result;
        }

        // ── 并行 transform_reduce（二元输入范围）────────────────────────
        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        T parallel_transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                    T init, BinaryOp&& reduce_op, UnaryOp&& transform_op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first1, last1));
            if (total == 0) return init;

            const auto n_chunks = chunk_count(total);
            if (n_chunks <= 1)
            {
                return std::transform_reduce(first1, last1, first2, init,
                                            std::forward<BinaryOp>(reduce_op),
                                            std::forward<UnaryOp>(transform_op));
            }

            const std::size_t base = total / n_chunks;
            const std::size_t rem  = total % n_chunks;
            std::atomic<int> latch{static_cast<int>(n_chunks)};
            std::vector<T> partials(n_chunks);

            {
                std::lock_guard lock(queue_mutex_);
                std::size_t off = 0;
                for (std::size_t c = 0; c < n_chunks - 1; ++c)
                {
                    const std::size_t len = base + (c < rem ? 1 : 0);
                    auto i1 = first1;
                    auto i2 = first2;
                    std::advance(i1, static_cast<std::ptrdiff_t>(off));
                    std::advance(i2, static_cast<std::ptrdiff_t>(off));
                    auto i1_end = i1;
                    std::advance(i1_end, static_cast<std::ptrdiff_t>(len));
                    off += len;

                    tasks_.emplace([i1, i1_end, i2, &reduce_op, &transform_op, &partials, &latch, c]()
                    {
                        T local{};
                        auto it1 = i1;
                        auto it2 = i2;
                        for (; it1 != i1_end; ++it1, ++it2)
                            local = reduce_op(local, transform_op(*it1, *it2));
                        partials[c] = std::move(local);
                        latch.fetch_sub(1, std::memory_order_release);
                    });
                }
            }
            condition_.notify_all();

            {
                const std::size_t c = n_chunks - 1;
                const std::size_t off = total - (base + (c < rem ? 1 : 0));
                auto i1 = first1;
                auto i2 = first2;
                std::advance(i1, static_cast<std::ptrdiff_t>(off));
                std::advance(i2, static_cast<std::ptrdiff_t>(off));
                T local{};
                for (; i1 != last1; ++i1, ++i2)
                    local = reduce_op(local, transform_op(*i1, *i2));
                partials[c] = std::move(local);
                latch.fetch_sub(1, std::memory_order_release);
            }

            wait_for_latch(latch);

            T result = init;
            for (auto& p : partials)
                result = reduce_op(result, p);
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
    };

    // ── 全局线程池单例 ─────────────────────────────────────────────────────
    inline ThreadPool& global_thread_pool()
    {
        static ThreadPool pool{std::thread::hardware_concurrency()};
        return pool;
    }

} // namespace nn

#endif // NN_CORE_THREAD_POOL_HPP