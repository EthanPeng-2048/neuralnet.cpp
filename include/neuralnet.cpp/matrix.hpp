#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
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

#ifndef NN_EXEC_POLICY
#define NN_EXEC_POLICY std::execution::par_unseq
#endif

namespace nn {
class Matrix {
private:
    std::vector<double> data_{};
    std::size_t rows_{0};
    std::size_t cols_{0};

    [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const noexcept {
        return row * cols_ + col;
    }

    static void require_same_shape(const Matrix &lhs, const Matrix &rhs, const char *message) {
        if (lhs.rows_ != rhs.rows_ || lhs.cols_ != rhs.cols_) {
            throw std::invalid_argument(message);
        }
    }

public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols)
        : data_(rows * cols, 0.0), rows_(rows), cols_(cols) {}

    Matrix(std::vector<double> data, std::size_t rows, std::size_t cols)
        : data_(std::move(data)), rows_(rows), cols_(cols) {
        if (data_.size() != rows_ * cols_) {
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

    [[nodiscard]] double at(std::size_t row, std::size_t col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data_[index(row, col)];
    }

    void set_value(std::size_t row, std::size_t col, double value) {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        data_[index(row, col)] = value;
    }

    [[nodiscard]] double at_unchecked(std::size_t row, std::size_t col) const noexcept {
        return data_[index(row, col)];
    }

    void set_value_unchecked(std::size_t row, std::size_t col, double value) noexcept {
        data_[index(row, col)] = value;
    }

    [[nodiscard]] const std::vector<double> &data() const noexcept { return data_; }
    [[nodiscard]] std::vector<double> &data() noexcept { return data_; }

    [[nodiscard]] std::vector<std::vector<double>> get_data() const {
        std::vector<std::vector<double>> result(rows_, std::vector<double>(cols_, 0.0));
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t col = 0; col < cols_; ++col) {
                result[row][col] = data_[index(row, col)];
            }
        }
        return result;
    }

    void set_data(const std::vector<std::vector<double>> &new_data) {
        if (new_data.empty()) {
            rows_ = 0;
            cols_ = 0;
            data_.clear();
            return;
        }

        const std::size_t new_rows = new_data.size();
        const std::size_t new_cols = new_data.front().size();
        for (const auto &row : new_data) {
            if (row.size() != new_cols) {
                throw std::invalid_argument("all rows must have the same number of columns");
            }
        }

        rows_ = new_rows;
        cols_ = new_cols;
        data_.resize(rows_ * cols_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t col = 0; col < cols_; ++col) {
                data_[index(row, col)] = new_data[row][col];
            }
        }
    }

    [[nodiscard]] Matrix transpose() const {
        Matrix result(cols_, rows_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t col = 0; col < cols_; ++col) {
                result.set_value_unchecked(col, row, data_[index(row, col)]);
            }
        }
        return result;
    }

    [[nodiscard]] Matrix operator+(const Matrix &other) const {
        require_same_shape(*this, other, "addition dimension mismatch");
        Matrix result(rows_, cols_);
        std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                       result.data_.begin(), std::plus<>{});
        return result;
    }

    [[nodiscard]] Matrix operator-(const Matrix &other) const {
        require_same_shape(*this, other, "subtraction dimension mismatch");
        Matrix result(rows_, cols_);
        std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), other.data_.begin(),
                       result.data_.begin(), std::minus<>{});
        return result;
    }

    [[nodiscard]] Matrix operator*(double scalar) const {        Matrix result(rows_, cols_);
        std::transform(NN_EXEC_POLICY, data_.begin(), data_.end(), result.data_.begin(),
                       [scalar](double value) { return value * scalar; });
        return result;
    }

    [[nodiscard]] Matrix operator*(const Matrix &other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("matrix multiplication dimension mismatch");
        }

        Matrix result(rows_, other.cols_);
        const Matrix other_t = other.transpose();

        auto indices = std::views::iota(std::size_t{0}, rows_ * other.cols_);
        std::for_each(NN_EXEC_POLICY, indices.begin(), indices.end(),
                      [&](std::size_t idx) {
                          const std::size_t row = idx / other.cols_;
                          const std::size_t col = idx % other.cols_;
                          double sum = 0.0;
                          for (std::size_t k = 0; k < cols_; ++k) {
                              sum += data_[index(row, k)] * other_t.data_[other_t.index(col, k)];
                          }
                          result.data_[idx] = sum;
                      });

        return result;
    }

    void scale_inplace(double scalar) noexcept {
        std::for_each(NN_EXEC_POLICY, data_.begin(), data_.end(),
                      [scalar](double &value) { value *= scalar; });
    }
};
}

#endif