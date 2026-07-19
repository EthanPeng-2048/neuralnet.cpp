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
#include "span.hpp"
#include "compute_dispatch.hpp"

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
        Matrix(const Matrix &other)
            : data_(other.data_), rows_(other.rows_), cols_(other.cols_)
        {
        }
        Matrix(Matrix &&other) noexcept
            : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_)
        {
        }
        Matrix &operator=(const Matrix &other)
        {
            if (this != &other) {
                data_ = other.data_;
                rows_ = other.rows_;
                cols_ = other.cols_;
            }
            return *this;
        }
        Matrix &operator=(Matrix &&other) noexcept
        {
            if (this != &other) {
                data_ = std::move(other.data_);
                rows_ = other.rows_;
                cols_ = other.cols_;
            }
            return *this;
        }
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
        [[nodiscard]] std::span<const Scalar> span() const
        {
            return {data_.data(), data_.size()};
        }
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
            auto s = span();
            auto o = other.span();
            SmartPolicy::transform(s.begin(), s.end(), o.begin(),
                           result.data_.begin(), std::plus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator-(const Matrix &other) const
        {
            require_same_shape(*this, other, "subtraction dimension mismatch");
            Matrix result(rows_, cols_);
            auto s = span();
            auto o = other.span();
            SmartPolicy::transform(s.begin(), s.end(), o.begin(),
                           result.data_.begin(), std::minus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator*(Scalar scalar) const
        {
            Matrix result(rows_, cols_);
            auto s = span();
            SmartPolicy::transform(s.begin(), s.end(), result.data().begin(),
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
            auto o = other.span();
            SmartPolicy::transform(data_.begin(), data_.end(), o.begin(),
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
            auto s = span();
            SmartPolicy::transform(s.begin(), s.end(),
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
            auto s = span();
            auto o = other.span();
            SmartPolicy::transform(s.begin(), s.end(),
                           o.begin(), result.data_.begin(),
                           std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（就地修改） ──────────────────────────────────
        template <typename F>
        void binary_apply_inplace(const Matrix& other, F&& func)
        {
            require_same_shape(*this, other, "binary_apply_inplace dimension mismatch");
            auto o = other.span();
            SmartPolicy::transform(data_.begin(), data_.end(),
                           o.begin(), data_.begin(),
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

        // ── 按行归约（通用数学原语，不是算法） ──────────────────────────
        // 对每一行独立归约，返回 (rows, 1) 矩阵。
        //   result[r][0] = reduce_op(init, transform_op(this[r][0]), ..., transform_op(this[r][cols-1]))
        // 上层可基于此表达 softmax 行最大值/行求和、按行范数等算法。
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] Matrix row_reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            Matrix result(rows_, 1);
            if (rows_ == 0) return result;

            const auto self = span();
            auto out = result.span();
            const std::size_t C = cols_;

            auto row_indices = std::views::iota(std::size_t{0}, rows_);
            const std::size_t total = rows_ * cols_;

            auto process_row = [self, out, C, init,
                                reduce_op = std::forward<ReduceOp>(reduce_op),
                                transform_op = std::forward<TransformOp>(transform_op)](std::size_t r) noexcept {
                const auto row = self.subspan(r * C, C);
                T acc = init;
                for (std::size_t c = 0; c < C; ++c)
                    acc = reduce_op(acc, transform_op(row[c]));
                out[r] = static_cast<Scalar>(acc);
            };

            if (total >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(row_indices.begin(), row_indices.end(), process_row);
            else
                for (std::size_t r = 0; r < rows_; ++r)
                    process_row(r);
            return result;
        }

        // ── 按列归约（通用数学原语，不是算法） ──────────────────────────
        // 对每一列独立归约，返回 (1, cols) 矩阵。
        //   result[0][c] = reduce_op(init, transform_op(this[0][c]), ..., transform_op(this[rows-1][c]))
        // 上层可基于此表达 LayerNorm 列均值/列方差等算法。
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] Matrix col_reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            Matrix result(1, cols_);
            if (cols_ == 0) return result;

            const auto self = span();
            auto out = result.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            // 先归约到标量数组，再写回 result
            std::vector<T> accs(C, init);
            for (std::size_t r = 0; r < R; ++r) {
                for (std::size_t c = 0; c < C; ++c)
                    accs[c] = reduce_op(accs[c], transform_op(self[r * C + c]));
            }
            for (std::size_t c = 0; c < C; ++c)
                out[c] = static_cast<Scalar>(accs[c]);
            return result;
        }

        // ── 按行广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], row_vec[r][0])，row_vec 形状必须为 (rows_, 1)
        // 上层可基于此表达 softmax 减行最大值、除行求和等算法。
        template <typename F>
        void broadcast_row_inplace(const Matrix& row_vec, F&& op)
        {
            assert(row_vec.rows_ == rows_ && row_vec.cols_ == 1 && "row_vec shape mismatch");
            const auto v = row_vec.span();
            const std::size_t C = cols_;
            auto idx = std::views::iota(std::size_t{0}, data_.size());
            SmartPolicy::for_each(idx.begin(), idx.end(),
                [d = data_.data(), &v, C, op = std::forward<F>(op)](std::size_t i) noexcept {
                    d[i] = static_cast<Scalar>(op(d[i], v[i / C]));
                });
        }

        // ── 按列广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], col_vec[0][c])，col_vec 形状必须为 (1, cols_)
        // 上层可基于此表达 LayerNorm 减列均值、乘列标准差等算法。
        template <typename F>
        void broadcast_col_inplace(const Matrix& col_vec, F&& op)
        {
            assert(col_vec.rows_ == 1 && col_vec.cols_ == cols_ && "col_vec shape mismatch");
            const auto v = col_vec.span();
            const std::size_t C = cols_;
            auto idx = std::views::iota(std::size_t{0}, data_.size());
            SmartPolicy::for_each(idx.begin(), idx.end(),
                [d = data_.data(), &v, C, op = std::forward<F>(op)](std::size_t i) noexcept {
                    d[i] = static_cast<Scalar>(op(d[i], v[i % C]));
                });
        }

        // ── Broadcast bias 加法（in-place） ───────────────────────────
        // 通用广播原语：this[i][j] += bias[i][0]，bias 形状必须为 (rows_, 1)
        // 这是通用数学原语（按行广播加法），不是算法。
        void add_bias_broadcast_inplace(const Matrix& bias)
        {
            assert(bias.rows_ == rows_ && bias.cols_ == 1 && "bias broadcast dimension mismatch");
            const std::size_t batch = cols_;
            auto b = bias.span();
            auto idx = std::views::iota(std::size_t{0}, data_.size());
            SmartPolicy::for_each(idx.begin(), idx.end(),
                [d = data_.data(), &b, batch](std::size_t i) noexcept {
                    d[i] += b[i / batch];
                });
        }
    };
} // namespace nn


#endif // NN_ALGEBRA_MATRIX_HPP
