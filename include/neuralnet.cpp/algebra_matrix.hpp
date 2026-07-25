#ifndef NN_ALGEBRA_MATRIX_HPP
#define NN_ALGEBRA_MATRIX_HPP

#include <algorithm>
#include <array>
#include <atomic>
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

#include "core_errors.hpp"
#include "core_assert.hpp"
#include "core_threadpool.hpp"
#include "config.hpp"
#include "algebra_span.hpp"
#include "algebra_compute.hpp"

// GPU 加速支持（可选，由 CMake 的 NN_HAS_VULKAN 宏控制）
#ifdef NN_HAS_VULKAN
#include "backend/vk_backend.hpp"
#endif

namespace nn
{
    // ═══════════════════════════════════════════════════════════════════════
    //  Matrix 类 — 矩阵存储与运算原语
    //
    //  错误处理约定（与上层 L2 计算层不同）：
    //    - L1 代数层使用 NN_ASSERT 进行形状校验（编程错误检查）
    //    - L2+ 计算层使用 Result<T> 进行运行时错误处理（用户输入校验）
    //    - 原因：Matrix 是底层原语，assert 不影响 Release 性能；
    //            上层需要向用户报告错误，故使用 Result<T>
    //    - 参见 DEVELOPMENT_STANDARDS.md "分层职责单一" 章节
    // ═══════════════════════════════════════════════════════════════════════
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

        static void require_same_shape(const Matrix &lhs, const Matrix &rhs, [[maybe_unused]] std::string_view message)
        {
            if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_)
            {
                NN_ASSERT(false, message.data());
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
                NN_ASSERT(false, "Matrix index out of range");
            }
            return data_[index(row, col)];
        }
        void set_value(std::size_t row, std::size_t col, Scalar value)
        {
            if (row >= rows_ || col >= cols_)
            {
                NN_ASSERT(false, "Matrix index out of range");
            }
            data_[index(row, col)] = value;
        }
        [[nodiscard]] constexpr Scalar at_unchecked(std::size_t row, std::size_t col) const noexcept { return data_[index(row, col)]; } // 无校验
        constexpr void set_value_unchecked(std::size_t row, std::size_t col, Scalar value) noexcept { data_[index(row, col)] = value; } // 无校验
        // 注：原 data() 方法已移除——它返回 std::vector<Scalar>& 泄露内部存储类型，
        // 违反"不穿透接口"规范。所有外部访问应通过 span() 获取视图。
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
            NN_ASSERT(&result != this, "transpose_to: self-referencing not supported");
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
                nn::parallel_for_blocks(
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
            auto r = result.span();
            nn::transform(s.begin(), s.end(), o.begin(),
                           r.begin(), std::plus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator-(const Matrix &other) const
        {
            require_same_shape(*this, other, "subtraction dimension mismatch");
            Matrix result(rows_, cols_);
            auto s = span();
            auto o = other.span();
            auto r = result.span();
            nn::transform(s.begin(), s.end(), o.begin(),
                           r.begin(), std::minus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator*(Scalar scalar) const
        {
            Matrix result(rows_, cols_);
            auto s = span();
            auto r = result.span();
            nn::transform(s.begin(), s.end(), r.begin(),
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
                NN_ASSERT(false, "matrix multiplication dimension mismatch");
            Matrix result(rows_, other.cols_);
            multiply_to(result, other);
            return result;
        }

        // ── 基于 span 的矩阵乘法（零拷贝，供 batched_matmul 等场景使用） ──
        // 从 a/b 的子区间直接计算，无需构造临时 Matrix 拷贝
        static void multiply_to_span(
            std::span<Scalar> r, std::size_t M, std::size_t N,
            std::span<const Scalar> a, std::size_t /*a_rows*/, std::size_t a_cols,
            std::span<const Scalar> b, std::size_t b_rows, std::size_t b_cols)
        {
            NN_ASSERT(a_cols == b_rows, "multiply_to_span: inner dimension mismatch");
            (void)b_rows;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_cols;
            if (M == 0 || N == 0 || K == 0) return;

            // 直接复用 blocked matmul 内核
            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, b, r, M, N, K, a_cols, b_cols, j_blocks](std::size_t block_idx) noexcept
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
                            b_block[jj * k_len + kk] = b[(k_start + kk) * b_cols + (j_start + jj)];
                    const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);
                    for (std::size_t i = i_start; i < i_end; ++i)
                    {
                        const auto a_row = a.subspan(i * a_cols + k_start);
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
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 基于 span 的矩阵乘法（B 转置，零拷贝） ─────────────────────────
        static void multiply_transposed_to_span(
            std::span<Scalar> r, std::size_t M, std::size_t N,
            std::span<const Scalar> a, std::size_t /*a_rows*/, std::size_t a_cols,
            std::span<const Scalar> bt, std::size_t /*bt_rows*/, std::size_t bt_cols)
        {
            NN_ASSERT(a_cols == bt_cols, "multiply_transposed_to_span: inner dimension mismatch");
            (void)bt_cols;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_cols;
            if (M == 0 || N == 0 || K == 0) return;

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, bt, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
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
                            b_block[jj * k_len + kk] = bt[(j_start + jj) * K + (k_start + kk)];
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
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 基于 span 的矩阵乘法（A 转置，零拷贝） ─────────────────────────
        static void transpose_multiply_to_span(
            std::span<Scalar> r, std::size_t M, std::size_t N,
            std::span<const Scalar> a, std::size_t a_rows, std::size_t a_cols,
            std::span<const Scalar> b, std::size_t b_rows, std::size_t b_cols)
        {
            NN_ASSERT(a_rows == b_rows, "transpose_multiply_to_span: inner dimension mismatch");
            (void)b_rows;  // NN_ASSERT 在 Release 模式下展开为空，参数仅用于断言
            const std::size_t K = a_rows;  // a is (K, M) stored
            if (M == 0 || N == 0 || K == 0) return;

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            auto kernel = [a, b, r, M, N, K, a_cols, b_cols, j_blocks](std::size_t block_idx) noexcept
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
                    const std::size_t i_len = i_end - i_start;
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> a_block{};
                    for (std::size_t ii = 0; ii < i_len; ++ii)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            a_block[ii * k_len + kk] = a[(k_start + kk) * a_cols + (i_start + ii)];
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                    for (std::size_t jj = 0; jj < j_len; ++jj)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            b_block[jj * k_len + kk] = b[(k_start + kk) * b_cols + (j_start + jj)];
                    const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);
                    for (std::size_t i = 0; i < i_len; ++i)
                    {
                        const auto a_row = std::span<const Scalar>(a_block.data() + i * k_len, k_len);
                        auto r_row = r.subspan((i_start + i) * N + j_start);
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
            };

            if (n_blocks <= 1)
                kernel(0);
            else
            {
                auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
                nn::parallel_for_blocks(block_indices.begin(), block_indices.end(), kernel);
            }
        }

        // ── 矩阵乘法到预分配缓冲区（零分配热路径） ─────────────────────────
        // 使用 std::span（C++20）提供类型安全的非拥有视图
        void multiply_to(Matrix &result, const Matrix &other) const
        {
            NN_ASSERT(&result != this, "multiply_to: self-referencing not supported");
            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

#ifdef NN_HAS_VULKAN
            // ── GPU 加速路径 ─────────────────────────────────────────────────
            // 矩阵面积超过阈值时自动走 GPU，失败则静默 fallback 到 CPU
            if (SmartPolicy::gpu_enabled && M * N >= SmartPolicy::GPU_THRESHOLD)
            {
                auto& backend = GpuBackend::instance();
                // 快速路径：已初始化时跳过全局锁
                if (backend.is_initialized() || backend.initialize())
                {
                    auto mm = backend.matmul_direct(
                        span(), other.span(), result.span(), M, N, K);
                    if (mm)
                    {
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
#pragma clang loop vectorize(assume_safety)
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
                nn::parallel_for_blocks(
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
#pragma clang loop vectorize(assume_safety)
                                    for (std::size_t kk = 0; kk < k_len; ++kk)
                                        sum += a_row[kk] * b_col[kk];
                                    r_row[j] += sum;
                                }
                            }
                        }
                    });
            }
        }

        // ── 矩阵乘法（B 转置）到预分配缓冲区 ─────────────────────────
        // 计算 result = this * B^T，其中 B 存储为 (N, K) 行主序
        // 无需实际转置 B，直接从 B 的行读取列（节省 O(N*K) 的拷贝）
        // 
        // 维度要求：this=(M,K), b_trans=(N,K) → result=(M,N)
        // 即 C[m][n] = Σ_k A[m][k] * B[n][k]
        void multiply_transposed_to(Matrix &result, const Matrix &b_trans) const
        {
            NN_ASSERT(&result != this && &result != &b_trans, "multiply_transposed_to: self-referencing not supported");
            NN_ASSERT(cols_ == b_trans.cols_, "multiply_transposed_to: inner dimensions mismatch");
            const std::size_t M = rows_;
            const std::size_t K = cols_;
            const std::size_t N = b_trans.rows_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

            result.zero();

            const auto a = span();
            const auto bt = b_trans.span();
            auto r = result.span();

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            if (n_blocks <= 1)
            {
                const std::size_t i_start = 0, i_end = M;
                const std::size_t j_start = 0, j_end = N;
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                    const std::size_t k_len = k_end - k_start;
                    const std::size_t j_len = j_end - j_start;
                    // B^T 块加载：B[n][k] 从 bt[n * K + k] 读取
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                    for (std::size_t jj = 0; jj < j_len; ++jj)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            b_block[jj * k_len + kk] = bt[(j_start + jj) * K + (k_start + kk)];
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
                nn::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [a, bt, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
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
                                    b_block[jj * k_len + kk] = bt[(j_start + jj) * K + (k_start + kk)];
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

        // ── 矩阵乘法（A 转置）到预分配缓冲区 ─────────────────────────
        // 计算 result = this^T * B，其中 this 存储为 (K, M) 行主序
        // 无需实际转置 this，直接从 this 的列读取行（节省 O(K*M) 的拷贝）
        //
        // 维度要求：this=(K,M), b=(K,N) → result=(M,N)
        // 即 C[m][n] = Σ_k A[k][m] * B[k][n]
        void transpose_multiply_to(Matrix &result, const Matrix &b) const
        {
            NN_ASSERT(&result != this && &result != &b, "transpose_multiply_to: self-referencing not supported");
            NN_ASSERT(rows_ == b.rows_, "transpose_multiply_to: inner dimensions mismatch");
            const std::size_t K = rows_;  // this is (K, M)
            const std::size_t M = cols_;
            const std::size_t N = b.cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

            result.zero();

            const auto a = span();  // this stored as (K, M)
            const auto b_data = b.span();  // b stored as (K, N)
            auto r = result.span();

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            if (n_blocks <= 1)
            {
                const std::size_t i_start = 0, i_end = M;
                const std::size_t j_start = 0, j_end = N;
                for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                {
                    const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                    const std::size_t k_len = k_end - k_start;
                    const std::size_t j_len = j_end - j_start;
                    const std::size_t i_len = i_end - i_start;

                    // A^T 块加载：A^T[m][k] = A[k][m] 从 a[k * M + m] 读取
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> a_block{};
                    for (std::size_t ii = 0; ii < i_len; ++ii)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            a_block[ii * k_len + kk] = a[(k_start + kk) * M + (i_start + ii)];

                    // B 块加载：B[k][n] 从 b_data[k * N + n] 读取
                    std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                    for (std::size_t jj = 0; jj < j_len; ++jj)
                        for (std::size_t kk = 0; kk < k_len; ++kk)
                            b_block[jj * k_len + kk] = b_data[(k_start + kk) * N + (j_start + jj)];
                    const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);

                    for (std::size_t i = 0; i < i_len; ++i)
                    {
                        const auto a_row = std::span<const Scalar>(a_block.data() + i * k_len, k_len);
                        auto r_row = r.subspan((i_start + i) * N + j_start);
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
                nn::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [a, b_data, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
                    {
                        const std::size_t i_block = block_idx / j_blocks;
                        const std::size_t j_block = block_idx % j_blocks;
                        const std::size_t i_start = i_block * BLOCK_SIZE;
                        const std::size_t i_end = std::min(i_start + BLOCK_SIZE, M);
                        const std::size_t j_start = j_block * BLOCK_SIZE;
                        const std::size_t j_end = std::min(j_start + BLOCK_SIZE, N);
                        const std::size_t i_len = i_end - i_start;

                        for (std::size_t k_start = 0; k_start < K; k_start += BLOCK_SIZE)
                        {
                            const std::size_t k_end = std::min(k_start + BLOCK_SIZE, K);
                            const std::size_t k_len = k_end - k_start;
                            const std::size_t j_len = j_end - j_start;

                            // A^T 块加载
                            std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> a_block{};
                            for (std::size_t ii = 0; ii < i_len; ++ii)
                                for (std::size_t kk = 0; kk < k_len; ++kk)
                                    a_block[ii * k_len + kk] = a[(k_start + kk) * M + (i_start + ii)];

                            // B 块加载
                            std::array<Scalar, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                            for (std::size_t jj = 0; jj < j_len; ++jj)
                                for (std::size_t kk = 0; kk < k_len; ++kk)
                                    b_block[jj * k_len + kk] = b_data[(k_start + kk) * N + (j_start + jj)];
                            const auto b_block_span = std::span<const Scalar>(b_block.data(), k_len * j_len);

                            for (std::size_t i = 0; i < i_len; ++i)
                            {
                                const auto a_row = std::span<const Scalar>(a_block.data() + i * k_len, k_len);
                                auto r_row = r.subspan((i_start + i) * N + j_start);
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

        // ── 累加矩阵乘法（A * B^T，结果累加到 result） ─────────────
        // 计算 result += this * B^T
        // 用于梯度累加：grad_w += grad_output * input^T
        void multiply_transposed_add_to(Matrix &result, const Matrix &b_trans) const
        {
            NN_ASSERT(&result != this && &result != &b_trans, "multiply_transposed_add_to: self-referencing");
            NN_ASSERT(cols_ == b_trans.cols_, "multiply_transposed_add_to: inner dimensions mismatch");
            NN_ASSERT(result.rows() == rows_ && result.cols() == b_trans.rows_, "multiply_transposed_add_to: result shape mismatch");
            const std::size_t M = rows_;
            const std::size_t K = cols_;
            const std::size_t N = b_trans.rows_;
            if (M == 0 || N == 0 || K == 0) return;

            const auto a = span();
            const auto bt = b_trans.span();
            auto r = result.span();

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto n_blocks = i_blocks * j_blocks;

            if (n_blocks <= 1)
            {
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
                            b_block[jj * k_len + kk] = bt[(j_start + jj) * K + (k_start + kk)];
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
                nn::parallel_for_blocks(
                    block_indices.begin(), block_indices.end(),
                    [a, bt, r, M, N, K, j_blocks](std::size_t block_idx) noexcept
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
                                    b_block[jj * k_len + kk] = bt[(j_start + jj) * K + (k_start + kk)];
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
            auto s = span();
            nn::for_each(s.begin(), s.end(),
                           [scalar](Scalar &value) noexcept { value *= scalar; });
        }

        // 逐元素加法 inplace
        void add_inplace(const Matrix &other)
        {
            require_same_shape(*this, other, "add_inplace dimension mismatch");
            auto s = span();
            auto o = other.span();
            nn::transform(s.begin(), s.end(), o.begin(),
                           s.begin(), std::plus<>{});
        }

        // 填充零
        void zero() noexcept
        {
            std::fill(span().begin(), span().end(), 0.0);
        }


        // ── 逐元素一元变换（返回新矩阵） ────────────────────────────────
        // out[i] = func(in[i])，内部自动选择串行/并行。
        template <typename F>
        [[nodiscard]] Matrix apply(F&& func) const
        {
            Matrix result(rows_, cols_);
            auto s = span();
            auto r = result.span();
            nn::transform(s.begin(), s.end(),
                           r.begin(), std::forward<F>(func));
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
            auto r = result.span();
            nn::transform(s.begin(), s.end(),
                           o.begin(), r.begin(),
                           std::forward<F>(func));
            return result;
        }

        // ── 逐元素二元变换（就地修改） ──────────────────────────────────
        template <typename F>
        void binary_apply_inplace(const Matrix& other, F&& func)
        {
            require_same_shape(*this, other, "binary_apply_inplace dimension mismatch");
            auto s = span();
            auto o = other.span();
            nn::transform(s.begin(), s.end(),
                           o.begin(), s.begin(),
                           std::forward<F>(func));
        }

        // ── 归约操作 ────────────────────────────────────────────────────
        // result = reduce_op(init, transform_op(data[0]), transform_op(data[1]), ...)
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] T reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            auto s = span();
            return nn::transform_reduce(s.begin(), s.end(), init,
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

            auto process_row = [self, out, C, init,
                                reduce_op = std::forward<ReduceOp>(reduce_op),
                                transform_op = std::forward<TransformOp>(transform_op)](std::size_t r) noexcept {
                const auto row = self.subspan(r * C, C);
                T acc = init;
                for (std::size_t c = 0; c < C; ++c)
                    acc = reduce_op(acc, transform_op(row[c]));
                out[r] = static_cast<Scalar>(acc);
            };

            nn::for_each(row_indices.begin(), row_indices.end(), process_row);
            return result;
        }

        // ── 按列归约（通用数学原语，不是算法） ──────────────────────────
        // 对每一列独立归约，返回 (1, cols) 矩阵。
        //   result[0][c] = reduce_op(init, transform_op(this[0][c]), ..., transform_op(this[rows-1][c]))
        // 上层可基于此表达 LayerNorm 列均值/列方差等算法。
        //
        // 实现策略（cache-friendly blocked + 行块并行）：
        //   - 旧实现按列扫描，跨行 stride=C 访问，cache miss 率 ~100%
        //   - 单线程行主序扫描：每行的 C 个元素累加到 out[c]，cache 友好
        //   - 行块并行：R 足够大且 C 不至于让累加器溢出 L2 时，按行分块并行
        //
        // bench_thresholds 单线程 vs 按列并行实测（32 核 CPU, Release -O3, 2026-07-25）：
        //   形状        naiveμs   blockedμs   加速比
        //   32×128        1.8        0.3       6.00x
        //   128×128      12.7        1.3       9.77x
        //   1K×1K       2328.9      141.4      16.47x
        //   4K×4K      208753.7    4827.7      43.24x
        //   32×10K      151.7       51.9       2.92x   (naive SIMD 优势最小)
        //   1K×10K    48048.4     2203.6      21.80x
        //   64×64K    47890.8     1287.6      37.19x
        //   8K×64      1248.9       75.6      16.52x
        //   64×8K      1089.6       83.1      13.11x
        //   结论：所有形状下 blocked 全面胜出。
        //
        // 按列并行测试结果（cache miss 严重，仅 128×10K 受益）：
        //   形状        串行μs    并行μs    加速比
        //   1K×1K       592.8    2326.6    0.25x   ← stride=C 跨行访问
        //   4K×4K     11039.1   54651.3    0.20x   ← 同上
        //   128×10K     719.8     273.5    2.63x   ← 唯一按列并行受益场景
        //
        // 行块并行策略（本实现）：
        //   - 按行分块：T 个行块，每线程处理 [r0, r_end) 区间，本地累加器累加该区间所有行
        //   - 每线程内部仍是行主序扫描，cache 行为与单线程版完全一致
        //   - 归并阶段：串行 O(T*C) 合并各线程累加器到 out[c]
        //   - 启用条件：R >= COL_REDUCE_PARALLEL_ROWS 且 C*T*sizeof(Scalar) <= L2_BUDGET
        //   - 否则回退到单线程行主序扫描
        template <typename T, typename ReduceOp, typename TransformOp>
        [[nodiscard]] Matrix col_reduce(T init, ReduceOp&& reduce_op, TransformOp&& transform_op) const
        {
            Matrix result(1, cols_);
            if (cols_ == 0) return result;

            const auto self = span();
            auto out = result.span();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            // 极小矩阵直接 naive（避免清零开销）
            if (R * C < 64)
            {
                for (std::size_t c = 0; c < C; ++c)
                {
                    T acc = init;
                    for (std::size_t r = 0; r < R; ++r)
                        acc = reduce_op(acc, transform_op(self[r * C + c]));
                    out[c] = static_cast<Scalar>(acc);
                }
                return result;
            }

            // 行块并行启用条件：
            //   1. R 足够大（保证每线程分到足够行块摊销同步开销）
            //   2. R * C >= PARALLEL_THRESHOLD（与全局并行阈值一致）
            //
            // bench_thresholds 实测（32 核 CPU, Release -O3, 2026-07-25）：
            //   形状         串行μs   行块并μs  加速比
            //   1K×1K         592.3    229.3     2.58x
            //   1K×10K       6598.7   1671.9     3.95x
            //   4K×4K       11347.5   1497.4     7.58x
            //   8K×128        662.1    335.7     1.97x
            //   64K×64       2881.7    759.5     3.79x
            //   4K×512       1239.8    398.9     3.11x
            //   128×128       10.1     40.1     0.25x   ← R<1024，同步开销主导
            //   64×8K        300.3    937.6     0.32x   ← R<1024
            //   128×10K      719.0   1085.3     0.66x   ← R<1024，归并成本主导
            //   结论：R >= 1024 是行块并行有效性的硬门槛；C 的上限不严
            //         （累加器 1K×10K*32*4=1.28MB 溢出 L2 进 L3，但 cache 行为仍优于单线程串行扫描）
            constexpr std::size_t COL_REDUCE_PARALLEL_ROWS = 1024;     // 行数门槛
            const std::size_t hw_threads = std::thread::hardware_concurrency();
            const std::size_t n_threads = (hw_threads == 0) ? 1 : hw_threads;
            const bool use_parallel =
                R >= COL_REDUCE_PARALLEL_ROWS &&
                R * C >= PARALLEL_THRESHOLD &&
                n_threads > 1;

            if (!use_parallel)
            {
                // ── 单线程行主序扫描 ──
                for (std::size_t c = 0; c < C; ++c)
                    out[c] = static_cast<Scalar>(init);
                for (std::size_t r = 0; r < R; ++r)
                {
                    const Scalar* row = self.data() + r * C;
                    for (std::size_t c = 0; c < C; ++c)
                    {
                        Scalar v = static_cast<Scalar>(transform_op(row[c]));
                        out[c] = static_cast<Scalar>(reduce_op(static_cast<T>(out[c]), v));
                    }
                }
                return result;
            }

            // ── 行块并行路径 ──
            // 分配 n_threads 组本地累加器（连续存储，cache 友好）
            std::vector<T> local_acc(n_threads * C);
            for (std::size_t t = 0; t < n_threads; ++t)
                for (std::size_t c = 0; c < C; ++c)
                    local_acc[t * C + c] = init;

            // 按行分块并行扫描
            auto& pool = global_thread_pool();
            const std::size_t base = R / n_threads;
            const std::size_t rem = R % n_threads;
            auto row_blocks = std::views::iota(std::size_t{0}, n_threads);
            pool.parallel_for_blocks(row_blocks.begin(), row_blocks.end(),
                [self, &local_acc, &reduce_op, &transform_op, C, base, rem](std::size_t t) noexcept {
                    const std::size_t r0 = t * base + std::min(t, rem);
                    const std::size_t r_end = (t + 1) * base + std::min(t + 1, rem);
                    auto* acc = local_acc.data() + t * C;
                    for (std::size_t r = r0; r < r_end; ++r)
                    {
                        const Scalar* row = self.data() + r * C;
                        for (std::size_t c = 0; c < C; ++c)
                        {
                            Scalar v = static_cast<Scalar>(transform_op(row[c]));
                            acc[c] = reduce_op(acc[c], v);
                        }
                    }
                });

            // 归并阶段：串行合并 n_threads 组累加器到 out[c]
            // 第 0 组直接写入，其余组归并进来（reduce_op 满足结合律，结果与单线程一致）
            for (std::size_t c = 0; c < C; ++c)
                out[c] = static_cast<Scalar>(local_acc[c]);  // 组 0
            for (std::size_t t = 1; t < n_threads; ++t)
            {
                const auto* acc = local_acc.data() + t * C;
                for (std::size_t c = 0; c < C; ++c)
                    out[c] = static_cast<Scalar>(reduce_op(static_cast<T>(out[c]), acc[c]));
            }
            return result;
        }

        // ── 按行广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], row_vec[r][0])，row_vec 形状必须为 (rows_, 1)
        // 上层可基于此表达 softmax 减行最大值、除行求和等算法。
        template <typename F>
        void broadcast_row_inplace(const Matrix& row_vec, F&& op)
        {
            NN_ASSERT(row_vec.rows_ == rows_ && row_vec.cols_ == 1, "row_vec shape mismatch");
            const auto v = row_vec.span();
            const std::size_t C = cols_;
            auto d = span();
            auto idx = std::views::iota(std::size_t{0}, d.size());
            nn::for_each(idx.begin(), idx.end(),
                [&d, &v, C, op = std::forward<F>(op)](std::size_t i) noexcept {
                    d[i] = static_cast<Scalar>(op(d[i], v[i / C]));
                });
        }

        // ── 按列广播（通用数学原语，不是算法） ──────────────────────────
        // this[r][c] = op(this[r][c], col_vec[0][c])，col_vec 形状必须为 (1, cols_)
        // 上层可基于此表达 LayerNorm 减列均值、乘列标准差等算法。
        template <typename F>
        void broadcast_col_inplace(const Matrix& col_vec, F&& op)
        {
            NN_ASSERT(col_vec.rows_ == 1 && col_vec.cols_ == cols_, "col_vec shape mismatch");
            const auto v = col_vec.span();
            const std::size_t C = cols_;
            auto d = span();
            auto idx = std::views::iota(std::size_t{0}, d.size());
            nn::for_each(idx.begin(), idx.end(),
                [&d, &v, C, op = std::forward<F>(op)](std::size_t i) noexcept {
                    d[i] = static_cast<Scalar>(op(d[i], v[i % C]));
                });
        }

        // ── Broadcast bias 加法（in-place） ───────────────────────────
        // 通用广播原语：this[i][j] += bias[i][0]，bias 形状必须为 (rows_, 1)
        // 这是通用数学原语（按行广播加法），不是算法。
        void add_bias_broadcast_inplace(const Matrix& bias)
        {
            NN_ASSERT(bias.rows_ == rows_ && bias.cols_ == 1, "bias broadcast dimension mismatch");
            const std::size_t batch = cols_;
            auto b = bias.span();
            auto d = span();
            auto idx = std::views::iota(std::size_t{0}, d.size());
            nn::for_each(idx.begin(), idx.end(),
                [&d, &b, batch](std::size_t i) noexcept {
                    d[i] += b[i / batch];
                });
        }
    };
} // namespace nn

// ── GpuTensor 方法实现（需要 Matrix 和 GpuBackend 的完整定义）──────────
#ifdef NN_HAS_VULKAN
#include "backend/gpu_tensor_impl.hpp"
#endif

#endif // NN_ALGEBRA_MATRIX_HPP