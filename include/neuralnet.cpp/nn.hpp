#ifndef NN_HPP
#define NN_HPP

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

// 默认使用并行非序列化执行策略。若编译环境未链接 Intel TBB，可改为 std::execution::seq
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
            rows_ = 0;            cols_ = 0;
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
        data_.resize(rows_ * cols_); // 优化：避免 assign 的零初始化开销
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

        // 优化：使用 C++20 views::iota 替代临时 vector，消除堆分配
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

[[nodiscard]] inline Matrix one_hot(const std::vector<std::size_t> &true_i, std::size_t mat_size) {
    const std::size_t batch_size = true_i.size();
    Matrix result(mat_size, batch_size);

    for (std::size_t i = 0; i < batch_size; ++i) {
        if (true_i[i] >= mat_size) {
            throw std::out_of_range("one_hot index out of range");
        }
        result.set_value_unchecked(true_i[i], i, 1.0);
    }

    return result;
}
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix &input) = 0;
    virtual Matrix backward(const Matrix &grad_output) = 0;
    virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
    virtual std::vector<std::reference_wrapper<Matrix>> param_gradients() { return {}; }
};

class Linear final : public Layer {
private:
    Matrix W_;
    Matrix b_;
    Matrix grad_W_;
    Matrix grad_b_;
    Matrix input_cache_;

    // 修复：使用 thread_local 保证多线程构造 Layer 时的线程安全
    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

public:
    Linear(std::size_t in_features, std::size_t out_features)
        : W_(out_features, in_features),
          b_(out_features, 1),
          grad_W_(out_features, in_features),   // 添加
          grad_b_(out_features, 1),             // 添加
          input_cache_() {
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        std::ranges::generate(W_.data(), [&] { return dist(rng_); });
    }
    
    std::vector<std::reference_wrapper<Matrix>> parameters() override {
        return {std::ref(W_), std::ref(b_)};
    }
    
    std::vector<std::reference_wrapper<Matrix>> param_gradients() override {
        return {std::ref(grad_W_), std::ref(grad_b_)};
    }
    
    Matrix forward(const Matrix &input) override {
        if (input.rows() != W_.cols()) {
            throw std::invalid_argument("linear forward input shape mismatch");
        }

        input_cache_ = input;
        const Matrix product = W_ * input;
        Matrix result(product.rows(), product.cols());

        for (std::size_t row = 0; row < product.rows(); ++row) {
            const double bias_value = b_.at_unchecked(row, 0);
            const auto begin = product.data().begin() + static_cast<std::ptrdiff_t>(row * product.cols());
            const auto end = begin + static_cast<std::ptrdiff_t>(product.cols());
            const auto out_begin = result.data().begin() + static_cast<std::ptrdiff_t>(row * product.cols());
            std::transform(begin, end, out_begin,
                           [bias_value](double value) { return value + bias_value; });
        }

        return result;
    }

    Matrix backward(const Matrix &grad_output) override {
        if (grad_output.rows() != W_.rows()) {
            throw std::invalid_argument("linear backward grad_output shape mismatch");
        }
        if (input_cache_.rows() != W_.cols() || input_cache_.cols() != grad_output.cols()) {            throw std::invalid_argument("linear backward cache/input shape mismatch");
        }

        const std::size_t in_feat = W_.cols();
        const std::size_t out_feat = W_.rows();
        const std::size_t batch = grad_output.cols();

        // 计算 grad_input
        Matrix grad_input(in_feat, batch);
        auto grad_in_indices = std::views::iota(std::size_t{0}, in_feat * batch);
        std::for_each(NN_EXEC_POLICY, grad_in_indices.begin(), grad_in_indices.end(),
                      [&](std::size_t idx) {
                          const std::size_t input_feature = idx / batch;
                          const std::size_t batch_index = idx % batch;
                          double sum = 0.0;
                          for (std::size_t out_feature = 0; out_feature < out_feat; ++out_feature) {
                              sum += W_.at_unchecked(out_feature, input_feature) *
                                     grad_output.at_unchecked(out_feature, batch_index);
                          }
                          grad_input.set_value_unchecked(input_feature, batch_index, sum);
                      });

        // 计算 grad_W（累加模式）
        auto grad_w_indices = std::views::iota(std::size_t{0}, out_feat * in_feat);
        std::for_each(NN_EXEC_POLICY, grad_w_indices.begin(), grad_w_indices.end(),
                      [&](std::size_t idx) {
                          const std::size_t out_feature = idx / in_feat;
                          const std::size_t input_feature = idx % in_feat;
                          double sum = 0.0;
                          for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
                              sum += grad_output.at_unchecked(out_feature, batch_index) *
                                     input_cache_.at_unchecked(input_feature, batch_index);
                          }
                          // 累加而不是覆盖
                          double old = grad_W_.at_unchecked(out_feature, input_feature);
                          grad_W_.set_value_unchecked(out_feature, input_feature, old + sum);
                      });
        
        // 计算 grad_b（累加模式）
        auto out_indices = std::views::iota(std::size_t{0}, out_feat);
        auto batch_indices = std::views::iota(std::size_t{0}, batch);
        std::for_each(NN_EXEC_POLICY, out_indices.begin(), out_indices.end(),
                      [&](std::size_t out_feature) {
                          const double sum = std::transform_reduce(
                              NN_EXEC_POLICY,
                              batch_indices.begin(), batch_indices.end(),
                              0.0,
                              std::plus<>{},
                              [&](std::size_t batch_index) {
                                  return grad_output.at_unchecked(out_feature, batch_index);
                              });
                          // 累加而不是覆盖
                          double old = grad_b_.at_unchecked(out_feature, 0);
                          grad_b_.set_value_unchecked(out_feature, 0, old + sum);
                      });
        return grad_input;
    }
};

class ReLU final : public Layer {
private:
    Matrix input_cache_;

public:
    ReLU() = default;

    Matrix forward(const Matrix &input) override {
        input_cache_ = input;
        Matrix result(input.rows(), input.cols());
        std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                       result.data().begin(), [](double value) { return std::max(0.0, value); });
        return result;
    }
    
    Matrix backward(const Matrix &grad_output) override {
        if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols()) {
            throw std::invalid_argument("relu backward shape mismatch");
        }

        Matrix grad_input(grad_output.rows(), grad_output.cols());
        std::transform(NN_EXEC_POLICY,
                       input_cache_.data().begin(), input_cache_.data().end(),
                       grad_output.data().begin(),
                       grad_input.data().begin(),
                       [](double input_value, double grad_value) {
                           return input_value > 0.0 ? grad_value : 0.0;
                       });        return grad_input;
    }
};

class MSELoss final {
private:
    Matrix grad_input_;

public:
    MSELoss() = default;

    [[nodiscard]] double forward(const Matrix &pred, const Matrix &target) {
        if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
            throw std::invalid_argument("mse loss shape mismatch");
        }
        if (pred.empty()) {
            throw std::invalid_argument("mse loss cannot be computed on an empty matrix");
        }

        grad_input_ = Matrix(pred.rows(), pred.cols());
        const auto total = static_cast<double>(pred.size());

        const double sum_sq = std::transform_reduce(
            NN_EXEC_POLICY,
            pred.data().begin(), pred.data().end(),
            target.data().begin(),
            0.0,
            std::plus<>{},
            [](double prediction, double actual) {
                const double diff = prediction - actual;
                return diff * diff;
            });

        const double loss = sum_sq / total;
        const double factor = 2.0 / total;

        std::transform(NN_EXEC_POLICY,
                       pred.data().begin(), pred.data().end(),
                       target.data().begin(),
                       grad_input_.data().begin(),
                       [factor](double prediction, double actual) {
                           return factor * (prediction - actual);
                       });

        return loss;
    }

    [[nodiscard]] Matrix backward() const { return grad_input_; }
};

// 优化器基类
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

// SGD 优化器
class SGD : public Optimizer {
private:
    double lr_;
    std::vector<std::reference_wrapper<Matrix>> params_;
    std::vector<std::reference_wrapper<Matrix>> grads_;

public:
    SGD(std::vector<std::reference_wrapper<Matrix>> params,
        std::vector<std::reference_wrapper<Matrix>> grads,
        double lr)
        : lr_(lr), params_(std::move(params)), grads_(std::move(grads)) {
        if (params_.size() != grads_.size()) {
            throw std::invalid_argument("params and grads must have same size");
        }
    }

    void step() override {
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto &p = params_[i].get();
            auto &g = grads_[i].get();
            // p = p - lr * g
            std::transform(NN_EXEC_POLICY,
                           p.data().begin(), p.data().end(),
                           g.data().begin(),
                           p.data().begin(),
                           [this](double p_val, double g_val) { return p_val - lr_ * g_val; });
        }
    }

    void zero_grad() override {
        for (auto &g_ref : grads_) {
            auto &g = g_ref.get();
            std::fill(g.data().begin(), g.data().end(), 0.0);
        }
    }
};
} // namespace nn

#endif // NN_HPP
