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

        Matrix(std::size_t rows, std::size_t cols)
            : data_(rows * cols, 0.0), rows_(rows), cols_(cols) {}

        Matrix(std::vector<double> data, std::size_t rows, std::size_t cols)
            : data_(std::move(data)), rows_(rows), cols_(cols)
        {
            if (data_.size() != rows_ * cols_)
            {
                throw std::invalid_argument("data size mismatch");
            }
        }
        Matrix(const Matrix &) = default;
        Matrix(Matrix &&) noexcept = default;
        Matrix &operator=(const Matrix &) = default;
        Matrix &operator=(Matrix &&) noexcept = default;
        ~Matrix() = default;

        [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
        [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
        [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
        [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

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

        [[nodiscard]] double at_unchecked(std::size_t row, std::size_t col) const noexcept
        {
            return data_[index(row, col)];
        }

        void set_value_unchecked(std::size_t row, std::size_t col, double value) noexcept
        {
            data_[index(row, col)] = value;
        }

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

        [[nodiscard]] Matrix transpose() const
        {
            Matrix result(cols_, rows_);

            // 块内行列数，32×32×8byte = 8KB，安全装入 L1

            const std::size_t i_blocks = (rows_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (cols_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

            auto block_indices = std::views::iota(
                std::size_t{0}, i_blocks * j_blocks);

            std::for_each(NN_EXEC_POLICY,
                          block_indices.begin(), block_indices.end(),
                          [&](std::size_t block_idx)
                          {
                              const std::size_t ib = block_idx / j_blocks;
                              const std::size_t jb = block_idx % j_blocks;

                              const std::size_t i0 = ib * BLOCK_SIZE;
                              const std::size_t j0 = jb * BLOCK_SIZE;
                              const std::size_t i1 = std::min(i0 + BLOCK_SIZE, rows_); // 边界截断
                              const std::size_t j1 = std::min(j0 + BLOCK_SIZE, cols_);

                              // 块内转置：A[i][j] -> R[j][i]
                              // 两层循环都在小块内，全部命中缓存
                              for (std::size_t i = i0; i < i1; ++i)
                                  for (std::size_t j = j0; j < j1; ++j)
                                      result.data_[j * rows_ + i] = data_[i * cols_ + j];
                          });

            return result;
        }

        [[nodiscard]] Matrix operator+(const Matrix &other) const
        {
            require_same_shape(*this, other, "addition dimension mismatch");
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::plus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator-(const Matrix &other) const
        {
            require_same_shape(*this, other, "subtraction dimension mismatch");
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                           result.data_.begin(), std::minus<>{});
            return result;
        }

        [[nodiscard]] Matrix operator*(double scalar) const
        {
            Matrix result(rows_, cols_);
            std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), result.data_.begin(),
                           [scalar](double value)
                           { return value * scalar; });
            return result;
        }

        [[nodiscard]] Matrix operator*(const Matrix &other) const
        {
            if (cols_ != other.rows_)
            {
                throw std::invalid_argument("matrix multiplication dimension mismatch");
            }

            const std::size_t M = rows_;
            const std::size_t N = other.cols_;
            const std::size_t K = cols_;

            Matrix result(M, N);

            const std::size_t i_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const std::size_t j_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

            auto block_indices = std::views::iota(std::size_t{0}, i_blocks * j_blocks);
            std::for_each(NN_EXEC_POLICY, block_indices.begin(), block_indices.end(),
                          [&](std::size_t block_idx)
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

                                  // 加载 B 的子块到栈数组并转置：b_block[jj * k_len + kk] = B(k_start+kk, j_start+jj)
                                  std::array<double, BLOCK_SIZE * BLOCK_SIZE> b_block{};
                                  for (std::size_t jj = 0; jj < j_len; ++jj)
                                  {
                                      for (std::size_t kk = 0; kk < k_len; ++kk)
                                      {
                                          b_block[jj * k_len + kk] = other.data_[other.index(k_start + kk, j_start + jj)];
                                      }
                                  }

                                  // 累加当前 K 块对 C 块的贡献
                                  for (std::size_t i = i_start; i < i_end; ++i)
                                  {
                                      const std::size_t a_base = i * K + k_start;
                                      for (std::size_t j = j_start; j < j_end; ++j)
                                      {
                                          double sum = 0.0;
                                          const std::size_t b_base = (j - j_start) * k_len;
                                          for (std::size_t kk = 0; kk < k_len; ++kk)
                                          {
                                              sum += data_[a_base + kk] * b_block[b_base + kk];
                                          }
                                          result.data_[result.index(i, j)] += sum;
                                      }
                                  }
                              }
                          });

            return result;
        }

        void scale_inplace(double scalar) noexcept
        {
            std::for_each(NN_EXEC_POLICY, data_.begin(), data_.end(),
                          [scalar](double &value)
                          { value *= scalar; });
        }
    };
}

#endif