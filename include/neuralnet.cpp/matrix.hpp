#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <neuralnet.cpp/nn_config.hpp>

namespace nn
{
    class Matrix
    {
    private:
        std::vector<double> data_{};
        std::size_t rows_{0};
        std::size_t cols_{0};

        [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept
        {
            return row * cols_ + col;
        }

        static void require_same_shape(const Matrix &lhs, const Matrix &rhs, const char *message)
        {
            if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_)
            {
                throw std::invalid_argument(message);
            }
        }

    public:
        Matrix() = default;

        explicit Matrix(std::size_t rows, std::size_t cols)
            : data_(rows * cols), rows_(rows), cols_(cols) {}

        Matrix(std::vector<double> data, std::size_t rows, std::size_t cols)
            : data_(std::move(data)), rows_(rows), cols_(cols)
        {
            if (data_.size() != rows_ * cols_)
            {
                throw std::invalid_argument("data size mismatch");
            }
        }
        
        // 从标量值初始化矩阵
        Matrix(std::size_t rows, std::size_t cols, double value)
            : data_(rows * cols, value), rows_(rows), cols_(cols) {}
        Matrix(const Matrix &) = default;
        Matrix(Matrix &&) noexcept = default;
        Matrix &operator=(const Matrix &) = default;
        Matrix &operator=(Matrix &&) noexcept = default;
        ~Matrix() = default;

        // ── 就地调整大小（复用已有内存） ──────────────────────────────────
        void resize(std::size_t rows, std::size_t cols) noexcept
        {
            if (rows_ == rows && cols_ == cols) return; // 尺寸不变，零开销
            rows_ = rows;
            cols_ = cols;
            data_.resize(rows * cols);
        }

        // ── 原始指针访问（供 SIMD / 内核使用） ────────────────────────────
        [[nodiscard]] const double *data_ptr() const noexcept { return data_.data(); }
        [[nodiscard]] double *data_ptr() noexcept { return data_.data(); }

        // 访问器
        [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] constexpr std::size_t size() const noexcept { return data_.size(); }
        [[nodiscard]] constexpr bool empty() const noexcept { return data_.empty(); }
        [[nodiscard]] double at(std::size_t row, std::size_t col) const
        {
            if (row >= rows_ || col >= cols_)
            {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[index(row, col)];
        }
        void set_value(std::size_t row, std::size_t col, double value)
        {
            if (row >= rows_ || col >= cols_)
            {
                throw std::out_of_range("Matrix index out of range");
            }
            data_[index(row, col)] = value;
        }
        [[nodiscard]] constexpr double at_unchecked(std::size_t row, std::size_t col) const noexcept { return data_[index(row, col)]; } // 无校验
        constexpr void set_value_unchecked(std::size_t row, std::size_t col, double value) noexcept { data_[index(row, col)] = value; } // 无校验
        [[nodiscard]] const std::vector<double> &data() const noexcept { return data_; }
        [[nodiscard]] std::vector<double> &data() noexcept { return data_; }
        [[nodiscard]] std::vector<std::vector<double>> get_data() const
        {
            std::vector<std::vector<double>> result(rows_, std::vector<double>(cols_, 0.0));
            for (std::size_t row = 0; row < rows_; ++row)
            {
                for (std::size_t col = 0; col < cols_; ++col)
                {
                    result[row][col] = data_[index(row, col)];
                }
            }
            return result;
        }

        void set_data(const std::vector<std::vector<double>> &new_data)
        {
            if (new_data.empty())
            {
                rows_ = 0;
                cols_ = 0;
                data_.clear();
                return;
            }

            const std::size_t new_rows = new_data.size();
            const std::size_t new_cols = new_data.front().size();
            for (const auto &row : new_data)
            {
                if (row.size() != new_cols)
                {
                    throw std::invalid_argument("all rows must have the same number of columns");
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
        }

        // ── 转置（返回新矩阵） ─────────────────────────────────────────────
        [[nodiscard]] Matrix transpose() const
        {
            Matrix result(cols_, rows_);
            transpose_to(result);
            return result;
        }

        // ── 转置到预分配缓冲区（零分配热路径） ─────────────────────────────
        void transpose_to(Matrix &result) const noexcept
        {
            result.resize(cols_, rows_);
            if (rows_ == 0 || cols_ == 0) return;

            const auto *src = data_ptr();
            auto *dst = result.data_ptr();
            const std::size_t R = rows_;
            const std::size_t C = cols_;

            const std::size_t i_blocks = (R + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (C + BLOCK_SIZE - 1) / BLOCK_SIZE;

            auto block_indices = std::views::iota(std::size_t{0}, i_blocks * j_blocks);

            SmartPolicy::for_each(block_indices.begin(), block_indices.end(),
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

        [[nodiscard]] Matrix operator*(double scalar) const
        {
            Matrix result(rows_, cols_);
            SmartPolicy::transform(data_.begin(), data_.end(), result.data().begin(),
                           [scalar](double value) noexcept { return value * scalar; });
            return result;
        }

        friend Matrix operator*(double scalar, const Matrix &mat) noexcept
        {
            return mat * scalar;
        }

        // ── 矩阵乘法（返回新矩阵） ─────────────────────────────────────────
        [[nodiscard]] Matrix operator*(const Matrix &other) const
        {
            if (cols_ != other.rows_)
                throw std::invalid_argument("matrix multiplication dimension mismatch");
            Matrix result(rows_, other.cols_);
            multiply_to(result, other);
            return result;
        }

        // ── 矩阵乘法到预分配缓冲区（零分配热路径） ─────────────────────────
        // 使用原始指针 + restrict 提示，帮助编译器自动向量化
        void multiply_to(Matrix &result, const Matrix &other) const noexcept
        {
            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;
            result.resize(M, N);
            if (M == 0 || N == 0 || K == 0) return;

            // 清零结果矩阵
            auto *r_ptr = result.data_ptr();
            std::fill(r_ptr, r_ptr + M * N, 0.0);

            const double *a_ptr = data_ptr();
            const double *b_ptr = other.data_ptr();

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            auto block_indices = std::views::iota(std::size_t{0}, i_blocks * j_blocks);
            SmartPolicy::for_each(block_indices.begin(), block_indices.end(),
                          [a_ptr, b_ptr, r_ptr, M, N, K, j_blocks](std::size_t block_idx) noexcept
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

                                  // 加载 B 的子块到栈数组并转置，改善访问局部性
                                  std::array<double, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                                  for (std::size_t jj = 0; jj < j_len; ++jj)
                                      for (std::size_t kk = 0; kk < k_len; ++kk)
                                          b_block[jj * k_len + kk] = b_ptr[(k_start + kk) * N + (j_start + jj)];

                                  // 累加当前 K 块对 C 块的贡献
                                  for (std::size_t i = i_start; i < i_end; ++i)
                                  {
                                      const double *a_row = a_ptr + i * K + k_start;
                                      double *r_row = r_ptr + i * N + j_start;
                                      for (std::size_t j = 0; j < j_len; ++j)
                                      {
                                          const double *b_col = b_block.data() + j * k_len;
                                          double sum = 0.0;
                                          for (std::size_t kk = 0; kk < k_len; ++kk)
                                              sum += a_row[kk] * b_col[kk];
                                          r_row[j] += sum;
                                      }
                                  }
                              }
                          });
        }

        void scale_inplace(double scalar) noexcept
        {
            SmartPolicy::for_each(data_.begin(), data_.end(),
                          [scalar](double &value) noexcept { value *= scalar; });
        }

        // 逐元素加法 inplace
        void add_inplace(const Matrix &other) noexcept
        {
            if (rows_ != other.rows_ || cols_ != other.cols_) return;
            SmartPolicy::transform(data_.begin(), data_.end(), other.data_.begin(),
                           data_.begin(), std::plus<>{});
        }

        // 逐元素减法 inplace
        void subtract_inplace(const Matrix &other) noexcept
        {
            if (rows_ != other.rows_ || cols_ != other.cols_) return;
            SmartPolicy::transform(data_.begin(), data_.end(), other.data_.begin(),
                           data_.begin(), std::minus<>{});
        }

        // 填充零
        void zero() noexcept
        {
            std::fill(data_.begin(), data_.end(), 0.0);
        }
    };
}

#endif