#ifndef NN_ALGEBRA_MATRIX_HPP
#define NN_ALGEBRA_MATRIX_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "../core/errors.hpp"
#include "../core/assert.hpp"
#include "../core/thread_pool.hpp"
#include "../nn_config.hpp"

// GPU 加速支持（可选，由 CMake 的 NN_HAS_VULKAN 宏控制）
#ifdef NN_HAS_VULKAN
#include "../backend/vk_backend.hpp"
#endif

namespace nn
{
    class Matrix
    {
    private:
        std::vector<Scalar> data_{};
        std::size_t rows_{0};
        std::size_t cols_{0};

        [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept
        {
            return row * cols_ + col;
        }

        static void require_same_shape(const Matrix &lhs, const Matrix &rhs, std::string_view message)
        {
            if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_)
            {
                assert(false && message.data()); // NOLINT
            }
        }

        // 内部构造函数（无校验，由工厂函数 create() 保证前置条件）
        Matrix(std::vector<Scalar> data, std::size_t rows, std::size_t cols)
            : data_(std::move(data)), rows_(rows), cols_(cols) {}

    public:
        Matrix() = default;

        explicit Matrix(std::size_t rows, std::size_t cols)
            : data_(rows * cols), rows_(rows), cols_(cols) {}

        // 工厂函数（策略1）：外部输入校验，返回 Result 而非 assert
        [[nodiscard]] static Result<Matrix> create(std::vector<Scalar> data, std::size_t rows, std::size_t cols)
        {
            if (data.size() != rows * cols)
                return std::unexpected(Error{"data size mismatch"});
            return Matrix(std::move(data), rows, cols);
        }

        // 从标量值初始化矩阵
        Matrix(std::size_t rows, std::size_t cols, Scalar value)
            : data_(rows * cols, value), rows_(rows), cols_(cols) {}
        Matrix(const Matrix &) = default;
        Matrix(Matrix &&) noexcept = default;
        Matrix &operator=(const Matrix &) = default;
        Matrix &operator=(Matrix &&) noexcept = default;
        ~Matrix() = default;

        // ── 就地调整大小（复用已有内存） ──────────────────────────────────
        void resize(std::size_t rows, std::size_t cols)
        {
            if (rows_ == rows && cols_ == cols) return; // 尺寸不变，零开销
            rows_ = rows;
            cols_ = cols;
            data_.resize(rows * cols);
        }

        // ── std::span 访问（C++20 现代接口，推荐使用） ────────────────────
        // 零开销抽象：编译后等价于裸指针 + 大小，可替代所有 data_ptr() 场景
        [[nodiscard]] std::span<const Scalar> span() const noexcept { return {data_.data(), data_.size()}; }
        [[nodiscard]] std::span<Scalar> span() noexcept { return {data_.data(), data_.size()}; }

        // 访问器
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] constexpr std::size_t size() const noexcept { return data_.size(); }
        [[nodiscard]] constexpr bool empty() const noexcept { return data_.empty(); }
        [[nodiscard]] Scalar at(std::size_t row, std::size_t col) const
        {
            if (row >= rows_ || col >= cols_)
            {
                assert(false && "Matrix index out of range");
            }
            return data_[index(row, col)];
        }
        void set_value(std::size_t row, std::size_t col, Scalar value)
        {
            if (row >= rows_ || col >= cols_)
            {
                assert(false && "Matrix index out of range");
            }
            data_[index(row, col)] = value;
        }
        [[nodiscard]] constexpr Scalar at_unchecked(std::size_t row, std::size_t col) const noexcept { return data_[index(row, col)]; } // 无校验
        constexpr void set_value_unchecked(std::size_t row, std::size_t col, Scalar value) noexcept { data_[index(row, col)] = value; } // 无校验
        [[nodiscard]] const std::vector<Scalar> &data() const noexcept { return data_; }
        [[nodiscard]] std::vector<Scalar> &data() noexcept { return data_; }
        [[nodiscard]] std::vector<std::vector<Scalar>> get_data() const
        {
            std::vector<std::vector<Scalar>> result(rows_, std::vector<Scalar>(cols_, 0.0));
            for (std::size_t row = 0; row < rows_; ++row)
            {
                for (std::size_t col = 0; col < cols_; ++col)
                {
                    result[row][col] = data_[index(row, col)];
                }
            }
            return result;
        }

        // 工厂函数（策略1）：外部输入校验，返回 Result 而非 assert
        [[nodiscard]] Result<void> set_data(const std::vector<std::vector<Scalar>> &new_data)
        {
            if (new_data.empty())
            {
                rows_ = 0;
                cols_ = 0;
                data_.clear();
                return {};
            }

            const std::size_t new_rows = new_data.size();
            const std::size_t new_cols = new_data.front().size();
            for (const auto &row : new_data)
            {
                if (row.size() != new_cols)
                {
                    return std::unexpected(Error{"all rows must have the same number of columns"});
                }
            }

            rows_ = new_rows;
            cols_ = new_cols;
            data_.resize(rows_ * cols_);
            for (std::size_t row = 0; row < rows_; ++row)
            {
                for (std::size_t col = 0; col < cols_; ++col)
                {
                    data_[index(row, col)] = new_data[row][col];
                }
            }
            return {};
        }

        // ── 转置（返回新矩阵） ─────────────────────────────────────────────
        [[nodiscard]] Matrix transpose() const
        {
            Matrix result(cols_, rows_);
            transpose_to(result);
            return result;
        }

        // ── 转置到预分配缓冲区（零分配热路径） ─────────────────────────────
        void transpose_to(Matrix &result) const
        {
            assert(&result != this && "transpose_to: self-referencing not supported");
            result.resize(cols_, rows_);
            if (rows_ == 0 || cols_ == 0) return;

            const auto src = span();
            auto dst = result.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            const std::size_t i_blocks = (R + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // 通过 SmartPolicy 分发块级并行：分块数很少（1~几十），但其内循环计算量大。
            const auto n_blocks = i_blocks * j_blocks;
            if (n_blocks <= 1)
            {
                for (std::size_t i = 0; i < R; ++i)
                    for (std::size_t j = 0; j < C; ++j)
                        dst[j * R + i] = src[i * C + j];
            }
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                SmartPolicy::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [src, dst, R, C, j_blocks](std::size_t block_idx) noexcept
                    {
                        const std::size_t ib = block_idx / j_blocks;
                        const std::size_t jb = block_idx % j_blocks;
                        const std::size_t i0 = ib * BLOCK_SIZE;
                        const std::size_t j0 = jb * BLOCK_SIZE;
                        const std::size_t i1 = std::min(i0 + BLOCK_SIZE, R);
                        const std::size_t j1 = std::min(j0 + BLOCK_SIZE, C);
                        for (std::size_t i = i0; i < i1; ++i)
                            for (std::size_t j = j0; j < j1; ++j)
                                dst[j * R + i] = src[i * C + j];
                    });
            }
        }

        [[nodiscard]] Matrix operator+(const Matrix &other) const
        {
            require_same_shape(*this, other, "addition dimension mismatch");
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::plus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator-(const Matrix &other) const
        {
            require_same_shape(*this, other, "subtraction dimension mismatch");
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::minus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator*(Scalar scalar) const
        {
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(), result.data().begin(),
                           [scalar](Scalar value) noexcept { return value * scalar; });
            return result;
        }

        friend Matrix operator*(Scalar scalar, const Matrix &mat)
        {
            return mat * scalar;
        }

        // ── 矩阵乘法（返回新矩阵） ─────────────────────────────────────────
        [[nodiscard]] Matrix operator*(const Matrix &other) const
        {
            if (cols_ != other.rows_)
                assert(false && "matrix multiplication dimension mismatch"); // NOLINT
            Matrix result(rows_, other.cols_);
            multiply_to(result, other);
            return result;
        }

        // ── 矩阵乘法到预分配缓冲区（零分配热路径） ─────────────────────────
        // 使用 std::span（C++20）提供类型安全的非拥有视图
        void multiply_to(Matrix &result, const Matrix &other) const
        {
            assert(&result != this && "multiply_to: self-referencing not supported");
            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

#ifdef NN_HAS_VULKAN
            // ── GPU 加速路径 ─────────────────────────────────────────────
            // 矩阵面积超过阈值时自动走 GPU，失败则静默 fallback 到 CPU
            if (SmartPolicy::gpu_enabled && M * N >= SmartPolicy::GPU_THRESHOLD)
            {
                auto& backend = GpuBackend::instance();
                // 快速路径：已初始化时跳过全局锁
                if (backend.is_initialized() || backend.initialize())
                {
                    auto mm = backend.matmul_direct(
                        span(), other.span(), result.span(), M, N, K);
                    if (mm) {
                        SmartPolicy::gpu_matmul_count.fetch_add(1, std::memory_order_relaxed);
                        return;  // GPU 成功，直接返回
                    }
                }
                // GPU 失败，静默 fallback 到 CPU 路径
            }
            SmartPolicy::cpu_matmul_count.fetch_add(1, std::memory_order_relaxed);
#endif

            // 清零结果矩阵（使用 RAII 封装的方法）
            result.zero();

            // std::span 非拥有视图：从 std::vector 借出，生命周期由调用栈保证
            const auto a = span();
            const auto b = other.span();
            auto r = result.span();

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // 通过 SmartPolicy 分发块级并行：分块数可能很少但其内循环计算量巨大（三层循环 O(n³)）。
            const auto n_blocks = i_blocks * j_blocks;
            if (n_blocks <= 1)
            {
                // 单块串行——与下方 lambda 逻辑相同，但避免不必要的线程池调用
                const std::size_t i_start = 0, i_end = M;
                const std::size_t j_start = 0, j_end = N;
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                    const std::size_t k_len = k_end - k_start;
                    const std::size_t j_len = j_end - j_start;
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                    for (std::size_t jj = 0; jj < j_len; ++jj)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            b_block[jj * k_len + kk] = b[(k_start + kk) * N + (j_start + jj)];
                    const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);
                    for (std::size_t i = i_start; i < i_end; ++i)
                    {
                        const auto a_row = a.subspan(i * K + k_start);
                        auto r_row = r.subspan(i * N + j_start);
                        for (std::size_t j = 0; j < j_len; ++j)
                        {
                            const auto b_col = b_block_span.subspan(j * k_len, k_len);
                            Scalar sum = 0.0;
                            for (std::size_t kk = 0; kk < k_len; ++kk)
                                sum += a_row[kk] * b_col[kk];
                            r_row[j] += sum;
                        }
                    }
                }
            }
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                SmartPolicy::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [a, b, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
                    {
                        const std::size_t i_block = block_idx / j_blocks;
                        const std::size_t j_block = block_idx % j_blocks;
                        const std::size_t i_start = i_block * BLOCK_SIZE;
                        const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                        const std::size_t j_start = j_block * BLOCK_SIZE;
                        const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);

                        for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                        {
                            const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                            const std::size_t k_len = k_end - k_start;
                            const std::size_t j_len = j_end - j_start;

                            std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                            for (std::size_t jj = 0; jj < j_len; ++jj)
                                for (std::size_t kk = 0; kk < k_len; ++kk)
                                    b_block[jj * k_len + kk] = b[(k_start + kk) * N + (j_start + jj)];
                            const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);

                            for (std::size_t i = i_start; i < i_end; ++i)
                            {
                                const auto a_row = a.subspan(i * K + k_start);
                                auto r_row = r.subspan(i * N + j_start);
                                for (std::size_t j = 0; j < j_len; ++j)
                                {
                                    const auto b_col = b_block_span.subspan(j * k_len, k_len);
                                    Scalar sum = 0.0;
                                    for (std::size_t kk = 0; kk < k_len; ++kk)
                                        sum += a_row[kk] * b_col[kk];
                                    r_row[j] += sum;
                                }
                            }
                        }
                    });
            }
        }

        void scale_inplace(Scalar scalar) noexcept
        {
            SmartPolicy::for_each(data_.begin(), data_.end(),
                           [scalar](Scalar &value) noexcept { value *= scalar; });
        }

        // 逐元素加法 inplace
        void add_inplace(const Matrix &other)
        {
            require_same_shape(*this, other, "add_inplace dimension mismatch");
            SmartPolicy::transform(data_.begin(), data_.end(), other.data_.begin(),
                           data_.begin(), std::plus<>{});
        }

        // 填充零
        void zero() noexcept
        {
            std::fill(data_.begin(), data_.end(), 0.0);
        }

        // ── 逐元素一元变换（返回新矩阵） ────────────────────────────────
        // out[i] = func(in[i])，内部自动选择串行/并行。
        template <typename F>
        [[nodiscard]] Matrix apply(F&& func) const
        {
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(),
                           result.data_.begin(), std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（返回新矩阵） ────────────────────────────────
        // out[i] = func(a[i], b[i])
        template <typename F>
        [[nodiscard]] Matrix binary_apply(const Matrix& other, F&& func) const
        {
            require_same_shape(*this, other, "binary_apply dimension mismatch");
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(),
                           other.data_.begin(), result.data_.begin(),
                           std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（就地修改） ──────────────────────────────────
        template <typename F>
        void binary_apply_inplace(const Matrix& other, F&& func)
        {
            require_same_shape(*this, other, "binary_apply_inplace dimension mismatch");
            SmartPolicy::transform(data_.begin(), data_.end(),
                           other.data_.begin(), data_.begin(),
                           std::forward<F>(func));
        }

        // ── 归约操作 ────────────────────────────────────────────────────
        // result = reduce_op(init, transform_op(data[0]), transform_op(data[1]), ...)
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] T reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            return SmartPolicy::transform_reduce(data_.begin(), data_.end(), init,
                std::forward<ReduceOp>(reduce_op), std::forward<TransformOp>(transform_op));
        }

        // ── 批量 SGD 更新（供 Optimizer 使用） ──────────────────────────
        // params[i] -= lr * grads[i]，逐矩阵串行、矩阵内部并行
        static void batch_sgd_update(
            std::span<std::reference_wrapper<Matrix>> params,
            std::span<std::reference_wrapper<Matrix>> grads,
            Scalar lr)
        {
            for (std::size_t i = 0; i < params.size(); ++i)
            {
                auto& p = params[i].get();
                auto& g = grads[i].get();
                p.binary_apply_inplace(g,
                    [lr](Scalar pv, Scalar gv) noexcept { return pv - lr * gv; });
            }
        }

        // ── 批量 Adam 更新（供 Optimizer 使用） ─────────────────────────
        static void batch_adam_update(
            std::span<std::reference_wrapper<Matrix>> params,
            std::span<std::reference_wrapper<Matrix>> grads,
            std::span<std::reference_wrapper<Matrix>> m_bufs,
            std::span<std::reference_wrapper<Matrix>> v_bufs,
            Scalar lr, Scalar beta1, Scalar beta2, Scalar eps, Scalar bc1, Scalar bc2)
        {
            for (std::size_t i = 0; i < params.size(); ++i)
            {
                auto& p = params[i].get();
                auto& g = grads[i].get();
                auto& m = m_bufs[i].get();
                auto& v = v_bufs[i].get();

                const auto total = p.size();
                auto idx = std::views::iota(std::size_t{0}, total);
                SmartPolicy::for_each(idx.begin(), idx.end(),
                    [&p, &g, &m, &v, lr, beta1, beta2, eps, bc1, bc2](std::size_t j) noexcept {
                        auto& pv = p.data_[j];
                        auto  gv = g.data_[j];
                        auto& mv = m.data_[j];
                        auto& vv = v.data_[j];

                        mv = beta1 * mv + (1.0 - beta1) * gv;
                        vv = beta2 * vv + (1.0 - beta2) * gv * gv;
                        pv -= lr * (mv / bc1) / (std::sqrt(vv / bc2) + eps);
                    });
            }
        }

        // ── 批量梯度清零 ────────────────────────────────────────────────
        static void batch_zero_grad(std::span<std::reference_wrapper<Matrix>> grads)
        {
            for (auto& g_ref : grads)
            {
                auto& g = g_ref.get();
                std::fill(g.data_.begin(), g.data_.end(), Scalar{0});
            }
        }

        // ── Broadcast bias 加法（in-place） ───────────────────────────
        // this[i][j] += bias[i][0]，bias 形状必须为 (rows_, 1)
        void add_bias_broadcast_inplace(const Matrix& bias)
        {
            assert(bias.rows_ == rows_ && bias.cols_ == 1 && "bias broadcast dimension mismatch");
            const std::size_t batch = cols_;
            SmartPolicy::for_each(data_.begin(), data_.end(),
                [&bias, batch, i = std::size_t{0}](Scalar& v) mutable noexcept {
                    v += bias.data_[i / batch];
                    ++i;
                });
        }

        // ── SGD+Momentum 批量更新 ─────────────────────────────────────
        // velocities[i] = beta * velocities[i] + (1-beta) * grads[i]
        // params[i] -= lr * velocities[i]
        static void batch_sgd_momentum_update(
            std::span<std::reference_wrapper<Matrix>> params,
            std::span<std::reference_wrapper<Matrix>> grads,
            std::span<std::reference_wrapper<Matrix>> velocities,
            Scalar lr, Scalar beta)
        {
            const auto n = params.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                auto& p = params[i].get();
                auto& g = grads[i].get();
                auto& v = velocities[i].get();
                v.binary_apply_inplace(g,
                    [beta](Scalar vv, Scalar gv) noexcept { return beta * vv + (1.0 - beta) * gv; });
                p.binary_apply_inplace(v,
                    [lr](Scalar pv, Scalar vv) noexcept { return pv - lr * vv; });
            }
        }

        // ── Softmax 按行（前向） ──────────────────────────────────────
        // input: (rows, cols)，output 必须已预分配相同形状
        static void softmax_rows(Matrix& output, const Matrix& input)
        {
            const std::size_t rows = input.rows_;
            const std::size_t cols = input.cols_;
            assert(output.rows_ == rows && output.cols_ == cols);

            const auto in_span = input.span();
            auto out_span = output.span();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const auto total = rows * cols;

            auto process_row = [in_span, out_span, cols](std::size_t r) noexcept {
                const std::size_t offset = r * cols;
                const auto row_in = in_span.subspan(offset, cols);
                auto row_out = out_span.subspan(offset, cols);

                Scalar max_val = row_in[0];
                for (std::size_t c = 1; c < cols; ++c)
                    max_val = std::max(max_val, row_in[c]);

                Scalar sum = Scalar{0};
                for (std::size_t c = 0; c < cols; ++c) {
                    Scalar e = std::exp(row_in[c] - max_val);
                    row_out[c] = e;
                    sum += e;
                }

                const Scalar inv_sum = Scalar{1} / sum;
                for (std::size_t c = 0; c < cols; ++c)
                    row_out[c] *= inv_sum;
            };

            if (total >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(row_indices.begin(), row_indices.end(), process_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    process_row(r);
        }

        // ── Softmax 按行（反向） ──────────────────────────────────────
        // grad_input[i][j] = softmax_out[i][j] * (grad_out[i][j] - Σ_k softmax_out[i][k] * grad_out[i][k])
        static void softmax_rows_backward(Matrix& grad_input,
                                          const Matrix& grad_output,
                                          const Matrix& softmax_output)
        {
            const std::size_t rows = softmax_output.rows_;
            const std::size_t cols = softmax_output.cols_;
            assert(grad_input.rows_ == rows && grad_input.cols_ == cols);
            assert(grad_output.rows_ == rows && grad_output.cols_ == cols);

            const auto go_span = grad_output.span();
            const auto out_span = softmax_output.span();
            auto gi_span = grad_input.span();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const auto total = rows * cols;

            auto process_row = [go_span, out_span, gi_span, cols](std::size_t r) noexcept {
                const std::size_t offset = r * cols;
                const auto row_go = go_span.subspan(offset, cols);
                const auto row_out = out_span.subspan(offset, cols);
                auto row_gi = gi_span.subspan(offset, cols);

                Scalar dot = Scalar{0};
                for (std::size_t c = 0; c < cols; ++c)
                    dot += row_out[c] * row_go[c];

                for (std::size_t c = 0; c < cols; ++c)
                    row_gi[c] = row_out[c] * (row_go[c] - dot);
            };

            if (total >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(row_indices.begin(), row_indices.end(), process_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    process_row(r);
        }

        // ── LayerNorm 前向传播 ────────────────────────────────────────
        // input: (features, batch_size), gamma/beta: (features, 1)
        // mean/std_inv 形状: (1, batch_size), normalized/output: (features, batch_size)
        // 所有输出矩阵必须已预分配
        static void layer_norm_forward(Matrix& output,
                                       Matrix& mean_out,
                                       Matrix& std_inv_out,
                                       Matrix& normalized_out,
                                       const Matrix& input,
                                       const Matrix& gamma,
                                       const Matrix& beta,
                                       Scalar epsilon)
        {
            const std::size_t features = input.rows_;
            const std::size_t batch_size = input.cols_;
            assert(gamma.rows_ == features && gamma.cols_ == 1);
            assert(beta.rows_ == features && beta.cols_ == 1);

            const auto in_span = input.span();
            const auto gamma_span = gamma.span();
            const auto beta_span = beta.span();
            auto mean_span = mean_out.span();
            auto std_span = std_inv_out.span();
            auto norm_span = normalized_out.span();
            auto res_span = output.span();

            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);

            auto process_sample = [in_span, gamma_span, beta_span, mean_span, std_span,
                                   norm_span, res_span, features, batch_size,
                                   epsilon](std::size_t b) noexcept {
                // 均值
                Scalar sum = Scalar{0};
                for (std::size_t f = 0; f < features; ++f)
                    sum += in_span[f * batch_size + b];
                Scalar mean = sum / static_cast<Scalar>(features);
                mean_span[b] = mean;

                // 方差→标准差倒数
                Scalar var_sum = Scalar{0};
                for (std::size_t f = 0; f < features; ++f) {
                    Scalar diff = in_span[f * batch_size + b] - mean;
                    var_sum += diff * diff;
                }
                Scalar variance = var_sum / static_cast<Scalar>(features);
                Scalar std_inv = Scalar{1} / std::sqrt(variance + epsilon);
                std_span[b] = std_inv;

                // 归一化 + 仿射变换
                for (std::size_t f = 0; f < features; ++f) {
                    Scalar normalized = (in_span[f * batch_size + b] - mean) * std_inv;
                    norm_span[f * batch_size + b] = normalized;
                    res_span[f * batch_size + b] = gamma_span[f] * normalized + beta_span[f];
                }
            };

            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(), process_sample);
            else
                for (std::size_t b = 0; b < batch_size; ++b)
                    process_sample(b);
        }

        // ── LayerNorm 反向传播 ────────────────────────────────────────
        // 计算 grad_input, 并累加到 grad_gamma, grad_beta
        static void layer_norm_backward(Matrix& grad_input,
                                        Matrix& grad_gamma,
                                        Matrix& grad_beta,
                                        const Matrix& grad_output,
                                        const Matrix& normalized_cache,
                                        const Matrix& std_inv_cache,
                                        const Matrix& gamma)
        {
            const std::size_t features = grad_output.rows_;
            const std::size_t batch_size = grad_output.cols_;

            const auto go_span = grad_output.span();
            const auto norm_span = normalized_cache.span();
            const auto std_span = std_inv_cache.span();
            const auto gamma_span = gamma.span();
            auto gi_span = grad_input.span();
            auto gg_span = grad_gamma.span();
            auto gb_span = grad_beta.span();

            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);

            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD) {
                // 并行计算 dL/dx（无数据竞争），串行累加 dL/dγ 和 dL/dβ
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(),
                    [go_span, norm_span, std_span, gamma_span, gi_span,
                     features, batch_size](std::size_t b) noexcept {
                        Scalar std_inv = std_span[b];
                        Scalar sum_grad = Scalar{0};
                        Scalar sum_grad_norm = Scalar{0};
                        for (std::size_t f = 0; f < features; ++f) {
                            Scalar g = go_span[f * batch_size + b] * gamma_span[f];
                            sum_grad += g;
                            sum_grad_norm += g * norm_span[f * batch_size + b];
                        }
                        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(features);
                        for (std::size_t f = 0; f < features; ++f) {
                            Scalar g = go_span[f * batch_size + b] * gamma_span[f];
                            gi_span[f * batch_size + b] = (g - sum_grad * inv_features -
                                norm_span[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
                        }
                    });
                // 串行累加 grad_gamma, grad_beta
                for (std::size_t b = 0; b < batch_size; ++b) {
                    for (std::size_t f = 0; f < features; ++f) {
                        Scalar grad_out = go_span[f * batch_size + b];
                        gg_span[f] += grad_out * norm_span[f * batch_size + b];
                        gb_span[f] += grad_out;
                    }
                }
            } else {
                for (std::size_t b = 0; b < batch_size; ++b) {
                    Scalar std_inv = std_span[b];
                    Scalar sum_grad = Scalar{0};
                    Scalar sum_grad_norm = Scalar{0};
                    for (std::size_t f = 0; f < features; ++f) {
                        Scalar g = go_span[f * batch_size + b] * gamma_span[f];
                        sum_grad += g;
                        sum_grad_norm += g * norm_span[f * batch_size + b];
                    }
                    const Scalar inv_features = Scalar{1} / static_cast<Scalar>(features);
                    for (std::size_t f = 0; f < features; ++f) {
                        Scalar grad_out = go_span[f * batch_size + b];
                        Scalar g = grad_out * gamma_span[f];
                        gg_span[f] += grad_out * norm_span[f * batch_size + b];
                        gb_span[f] += grad_out;
                        gi_span[f * batch_size + b] = (g - sum_grad * inv_features -
                            norm_span[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
                    }
                }
            }
        }

        // ── Cross Entropy Loss 前向传播 ───────────────────────────────
        // 计算逐样本 softmax + CE loss，写入 grad_input（= softmax - target）
        // 返回平均损失
        static Scalar cross_entropy_forward(Matrix& grad_input,
                                            const Matrix& logits,
                                            const Matrix& target_onehot)
        {
            const std::size_t classes = logits.rows_;
            const std::size_t batch = logits.cols_;
            assert(grad_input.rows_ == classes && grad_input.cols_ == batch);

            std::atomic<Scalar> total_loss{0.0};

            auto process_sample = [&](std::size_t i) {
                // 数值稳定 softmax
                Scalar max_val = logits.at_unchecked(0, i);
                for (std::size_t c = 1; c < classes; ++c) {
                    Scalar val = logits.at_unchecked(c, i);
                    if (val > max_val) max_val = val;
                }

                // 栈上 128 类，超过用堆
                std::array<Scalar, 128> exp_vals_fixed{};
                std::vector<Scalar> exp_vals_heap;
                std::span<Scalar> exp_vals;
                if (classes <= 128) {
                    exp_vals = exp_vals_fixed;
                } else {
                    exp_vals_heap.resize(classes);
                    exp_vals = exp_vals_heap;
                }

                Scalar sum_exp = Scalar{0};
                for (std::size_t c = 0; c < classes; ++c) {
                    Scalar e = std::exp(logits.at_unchecked(c, i) - max_val);
                    exp_vals[c] = e;
                    sum_exp += e;
                }

                // 查找真实类别
                std::size_t true_class = 0;
                for (std::size_t c = 0; c < classes; ++c) {
                    if (target_onehot.at_unchecked(c, i) > 0.5) {
                        true_class = c;
                        break;
                    }
                }

                const Scalar prob_true = exp_vals[true_class] / sum_exp;
                Scalar expected = total_loss.load(std::memory_order_relaxed);
                while (!total_loss.compare_exchange_weak(expected, expected - std::log(prob_true),
                                                          std::memory_order_relaxed)) {}

                // 梯度 = softmax - target
                for (std::size_t c = 0; c < classes; ++c) {
                    const Scalar softmax_c = exp_vals[c] / sum_exp;
                    grad_input.set_value_unchecked(c, i,
                        softmax_c - target_onehot.at_unchecked(c, i));
                }
            };

            if (batch >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::parallel_for_samples(batch, process_sample);
            else
                for (std::size_t i = 0; i < batch; ++i)
                    process_sample(i);

            return total_loss.load() / static_cast<Scalar>(batch);
        }
    };
} // namespace nn

// ── GpuTensor 方法实现（需要 Matrix 完整定义） ──────────────────────────
#ifdef NN_HAS_VULKAN
#include "../backend/gpu_tensor_impl.hpp"
#endif

#endif // NN_ALGEBRA_MATRIX_HPP