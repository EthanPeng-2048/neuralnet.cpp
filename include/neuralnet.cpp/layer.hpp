#ifndef LAYER_HPP
#define LAYER_HPP

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
#include "matrix.hpp"

namespace nn
{
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Matrix forward(const Matrix &input) = 0;
        virtual Matrix backward(const Matrix &grad_output) = 0;
        virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
        virtual std::vector<std::reference_wrapper<Matrix>> param_gradients() { return {}; }
    };

    class Linear final : public Layer
    {
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
              grad_W_(out_features, in_features), // 添加
              grad_b_(out_features, 1),           // 添加
              input_cache_()
        {
            // Xavier 均匀初始化：适合 tanh/sigmoid，对 ReLU 也可用
            const double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
            std::uniform_real_distribution<double> dist(-limit, limit);
            std::ranges::generate(W_.data(), [&]
                                  { return dist(rng_); });
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(W_), std::ref(b_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(grad_W_), std::ref(grad_b_)};
        }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != W_.cols())
            {
                throw std::invalid_argument("linear forward input shape mismatch");
            }

            input_cache_ = input;
            const Matrix product = W_ * input;
            Matrix result(product.rows(), product.cols());

            for (std::size_t row = 0; row < product.rows(); ++row)
            {
                const double bias_value = b_.at_unchecked(row, 0);
                const auto begin = product.data().begin() + static_cast<std::ptrdiff_t>(row * product.cols());
                const auto end = begin + static_cast<std::ptrdiff_t>(product.cols());
                const auto out_begin = result.data().begin() + static_cast<std::ptrdiff_t>(row * product.cols());
                std::transform(begin, end, out_begin,
                               [bias_value](double value)
                               { return value + bias_value; });
            }

            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != W_.rows())
            {
                throw std::invalid_argument("linear backward grad_output shape mismatch");
            }
            if (input_cache_.rows() != W_.cols() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("linear backward cache/input shape mismatch");
            }

            const std::size_t in_feat = W_.cols();
            const std::size_t out_feat = W_.rows();
            const std::size_t batch = grad_output.cols();

            // 计算 grad_input
            Matrix grad_input(in_feat, batch);
            auto grad_in_indices = std::views::iota(std::size_t{0}, in_feat * batch);
            std::for_each(NN_EXEC_POLICY, grad_in_indices.begin(), grad_in_indices.end(),
                          [&](std::size_t idx)
                          {
                              const std::size_t input_feature = idx / batch;
                              const std::size_t batch_index = idx % batch;
                              double sum = 0.0;
                              for (std::size_t out_feature = 0; out_feature < out_feat; ++out_feature)
                              {
                                  sum += W_.at_unchecked(out_feature, input_feature) *
                                         grad_output.at_unchecked(out_feature, batch_index);
                              }
                              grad_input.set_value_unchecked(input_feature, batch_index, sum);
                          });

            // 计算 grad_W
            auto grad_w_indices = std::views::iota(std::size_t{0}, out_feat * in_feat);
            std::for_each(NN_EXEC_POLICY, grad_w_indices.begin(), grad_w_indices.end(),
                          [&](std::size_t idx)
                          {
                              const std::size_t out_feature = idx / in_feat;
                              const std::size_t input_feature = idx % in_feat;
                              double sum = 0.0;
                              for (std::size_t batch_index = 0; batch_index < batch; ++batch_index)
                              {
                                  sum += grad_output.at_unchecked(out_feature, batch_index) *
                                         input_cache_.at_unchecked(input_feature, batch_index);
                              }
                              double old = grad_W_.at_unchecked(out_feature, input_feature);
                              grad_W_.set_value_unchecked(out_feature, input_feature, old + sum);
                          });

            // 计算 grad_b
            auto out_indices = std::views::iota(std::size_t{0}, out_feat);
            auto batch_indices = std::views::iota(std::size_t{0}, batch);
            std::for_each(NN_EXEC_POLICY, out_indices.begin(), out_indices.end(),
                          [&](std::size_t out_feature)
                          {
                              const double sum = std::transform_reduce(
                                  NN_EXEC_POLICY,
                                  batch_indices.begin(), batch_indices.end(),
                                  0.0,
                                  std::plus<>{},
                                  [&](std::size_t batch_index)
                                  {
                                      return grad_output.at_unchecked(out_feature, batch_index);
                                  });
                              double old = grad_b_.at_unchecked(out_feature, 0);
                              grad_b_.set_value_unchecked(out_feature, 0, old + sum);
                          });
            return grad_input;
        }
    };

    class ReLU final : public Layer
    {
    private:
        Matrix input_cache_;

    public:
        ReLU() = default;

        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            std::transform(NN_EXEC_POLICY, input.data().begin(), input.data().end(),
                           result.data().begin(), [](double value)
                           { return std::max(0.0, value); });
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
            {
                throw std::invalid_argument("relu backward shape mismatch");
            }

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            std::transform(NN_EXEC_POLICY,
                           input_cache_.data().begin(), input_cache_.data().end(),
                           grad_output.data().begin(),
                           grad_input.data().begin(),
                           [](double input_value, double grad_value)
                           {
                               return input_value > 0.0 ? grad_value : 0.0;
                           });
            return grad_input;
        }
    };
}

#endif