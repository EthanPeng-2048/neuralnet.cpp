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
#include <stdexcept>
#include <type_traits>

namespace nn
{
    // ── 简易线程池 ─────────────────────────────────────────────────────────────
    // 训练开始时创建固定数量线程，避免反复创建/销毁的调度开销
    // 支持提交任意可调用对象，返回 std::future 以获取结果
    class ThreadPool
    {
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;

        std::mutex queue_mutex_;
        std::condition_variable condition_;
        std::atomic<bool> stop_{false};

    public:
        // 构造：启动指定数量的工作线程
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

        // 提交任务，返回 future
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
                if (stop_.load(std::memory_order_acquire))
                    throw std::runtime_error("submit on stopped ThreadPool");
                tasks_.emplace([task]()
                {
                    (*task)();
                });
            }
            condition_.notify_one();
            return result;
        }

        // 并行 for_each：将 [first, last) 均匀分块，每块提交一个任务
        template<typename Iterator, typename Func>
        void parallel_for_each(Iterator first, Iterator last, Func&& func)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
            if (total == 0) return;

            const auto num_threads = workers_.size();
            if (num_threads <= 1 || total < num_threads)
            {
                // 串行回退
                for (auto it = first; it != last; ++it)
                    func(*it);
                return;
            }

            const std::size_t chunk_size = total / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);

            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto chunk_start = first;
                std::advance(chunk_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                auto chunk_end = (t == num_threads - 1) ? last : chunk_start;
                if (t < num_threads - 1)
                    std::advance(chunk_end, static_cast<std::ptrdiff_t>(chunk_size));

                futures.push_back(submit([&func, chunk_start, chunk_end]()
                {
                    for (auto it = chunk_start; it != chunk_end; ++it)
                        func(*it);
                }));
            }

            // 等待所有块完成
            for (auto& fut : futures)
                fut.get();
        }

        // 并行 transform：将 [first1, last1) 均匀分块
        template<typename InputIt, typename OutputIt, typename UnaryOp>
        void parallel_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first, last));
            if (total == 0) return;

            const auto num_threads = workers_.size();
            if (num_threads <= 1 || total < num_threads)
            {
                std::transform(first, last, d_first, std::forward<UnaryOp>(op));
                return;
            }

            const std::size_t chunk_size = total / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);

            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto chunk_start = first;
                auto out_start = d_first;
                std::advance(chunk_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                std::advance(out_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                
                auto chunk_end = (t == num_threads - 1) ? last : chunk_start;
                auto out_end = (t == num_threads - 1) ? d_first : out_start;
                if (t < num_threads - 1)
                {
                    std::advance(chunk_end, static_cast<std::ptrdiff_t>(chunk_size));
                    std::advance(out_end, static_cast<std::ptrdiff_t>(chunk_size));
                }

                futures.push_back(submit([&op, chunk_start, chunk_end, out_start]()
                {
                    auto in_it = chunk_start;
                    auto out_it = out_start;
                    for (; in_it != chunk_end; ++in_it, ++out_it)
                        *out_it = op(*in_it);
                }));
            }

            for (auto& fut : futures)
                fut.get();
        }

        // 并行 transform（二元版本）
        template<typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
        void parallel_transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                OutputIt d_first, BinaryOp&& op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first1, last1));
            if (total == 0) return;

            const auto num_threads = workers_.size();
            if (num_threads <= 1 || total < num_threads)
            {
                std::transform(first1, last1, first2, d_first, std::forward<BinaryOp>(op));
                return;
            }

            const std::size_t chunk_size = total / num_threads;
            std::vector<std::future<void>> futures;
            futures.reserve(num_threads);

            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto in1_start = first1;
                auto in2_start = first2;
                auto out_start = d_first;
                std::advance(in1_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                std::advance(in2_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                std::advance(out_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                
                auto in1_end = (t == num_threads - 1) ? last1 : in1_start;
                auto out_end = (t == num_threads - 1) ? d_first : out_start;
                if (t < num_threads - 1)
                {
                    std::advance(in1_end, static_cast<std::ptrdiff_t>(chunk_size));
                    std::advance(out_end, static_cast<std::ptrdiff_t>(chunk_size));
                }

                futures.push_back(submit([&op, in1_start, in1_end, in2_start, out_start]()
                {
                    auto it1 = in1_start;
                    auto it2 = in2_start;
                    auto out_it = out_start;
                    for (; it1 != in1_end; ++it1, ++it2, ++out_it)
                        *out_it = op(*it1, *it2);
                }));
            }

            for (auto& fut : futures)
                fut.get();
        }

        // 并行 transform_reduce（二元输入范围）
        template<typename InputIt1, typename InputIt2, typename T, typename BinaryOp, typename UnaryOp>
        T parallel_transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                                    T init, BinaryOp&& reduce_op, UnaryOp&& transform_op)
        {
            const auto total = static_cast<std::size_t>(std::distance(first1, last1));
            if (total == 0) return init;

            const auto num_threads = workers_.size();
            if (num_threads <= 1 || total < num_threads)
            {
                return std::transform_reduce(first1, last1, first2, init,
                                            std::forward<BinaryOp>(reduce_op),
                                            std::forward<UnaryOp>(transform_op));
            }

            const std::size_t chunk_size = total / num_threads;
            std::vector<std::future<T>> futures;
            futures.reserve(num_threads);

            for (std::size_t t = 0; t < num_threads; ++t)
            {
                auto in1_start = first1;
                auto in2_start = first2;
                std::advance(in1_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                std::advance(in2_start, static_cast<std::ptrdiff_t>(t * chunk_size));
                
                auto in1_end = (t == num_threads - 1) ? last1 : in1_start;
                if (t < num_threads - 1)
                    std::advance(in1_end, static_cast<std::ptrdiff_t>(chunk_size));

                futures.push_back(submit([&reduce_op, &transform_op, in1_start, in1_end, in2_start]()
                {
                    T local_sum{};
                    auto it1 = in1_start;
                    auto it2 = in2_start;
                    for (; it1 != in1_end; ++it1, ++it2)
                        local_sum = reduce_op(local_sum, transform_op(*it1, *it2));
                    return local_sum;
                }));
            }

            // 收集各块结果
            T result = init;
            for (auto& fut : futures)
                result = reduce_op(result, fut.get());
            return result;
        }

        // 析构：等待所有任务完成后关闭线程
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

        // 禁止拷贝/移动
        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;
        ThreadPool(ThreadPool&&) = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }
    };

    // ── 全局线程池单例（懒初始化） ─────────────────────────────────────────────
    // 训练开始时自动创建，程序结束时自动销毁，避免反复创建/销毁
    inline ThreadPool& global_thread_pool()
    {
        static ThreadPool pool{std::thread::hardware_concurrency()};
        return pool;
    }
} // namespace nn

#endif // THREAD_POOL_HPP