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

#include <neuralnet.cpp/nn_config.hpp>

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
        
        // 添加参数更新辅助方法，避免虚函数调用开销
        virtual void update_params(double /*lr*/) noexcept {}
        virtual void zero_grad() noexcept {}
    };

    class Linear final : public Layer
    {
    private:
        Matrix W_;
        Matrix b_;
        Matrix grad_W_;
        Matrix grad_b_;
        Matrix input_cache_;

        // ── 预分配缓冲区：避免 forward/backward 热路径反复分配内存 ──────
        Matrix product_buf_;   // W * input 的中间结果
        Matrix grad_WT_buf_;   // backward: W^T

        // 修复：使用 thread_local 保证多线程构造 Layer 时的线程安全
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    public:
        Linear(std::size_t in_features, std::size_t out_features)
            : W_(out_features, in_features),
              b_(out_features, 1),
              grad_W_(out_features, in_features),
              grad_b_(out_features, 1),
              input_cache_(),
              product_buf_(out_features, 1),
              grad_WT_buf_(in_features, out_features)
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
                throw std::invalid_argument("linear forward input shape mismatch");

            input_cache_ = input;

            // 融合矩阵乘法 + bias 加法，减少一次完整遍历
            const std::size_t out_feat = W_.rows();
            const std::size_t batch = input.cols();

            // product = W * input（写入预分配缓冲区，避免分配）
            W_.multiply_to(product_buf_, input);

            // result = product + bias（返回新矩阵，NRVO 优化）
            Matrix result(out_feat, batch);
            const double *prod_ptr = product_buf_.data_ptr();
            const double *bias_ptr = b_.data_ptr();
            double *res_ptr = result.data_ptr();
            const auto total = static_cast<std::size_t>(out_feat * batch);

            auto indices = std::views::iota(std::size_t{0}, total);
            SmartPolicy::for_each(indices.begin(), indices.end(),
                          [prod_ptr, bias_ptr, res_ptr, batch](std::size_t idx) noexcept
                          {
                              res_ptr[idx] = prod_ptr[idx] + bias_ptr[idx / batch];
                          });

            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != W_.rows())
                throw std::invalid_argument("linear backward grad_output shape mismatch");
            if (input_cache_.rows() != W_.cols() || input_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("linear backward cache/input shape mismatch");

            const std::size_t in_feat = W_.cols();
            const std::size_t out_feat = W_.rows();
            const std::size_t batch = grad_output.cols();

            // grad_input = W^T * grad_output
            W_.transpose_to(grad_WT_buf_);
            Matrix grad_input(in_feat, batch);
            grad_WT_buf_.multiply_to(grad_input, grad_output);

            // grad_W += grad_output * input_cache_^T（逐元素累加，避免临时矩阵）
            // 手动计算：grad_W[i][j] += sum_k(grad_output[i][k] * input_cache_[j][k])
            {
                const double *go_ptr = grad_output.data_ptr();
                const double *ic_ptr = input_cache_.data_ptr();
                double *gw_ptr = grad_W_.data_ptr();
                for (std::size_t of = 0; of < out_feat; ++of)
                {
                    for (std::size_t inf = 0; inf < in_feat; ++inf)
                    {
                        double sum = 0.0;
                        for (std::size_t b = 0; b < batch; ++b)
                            sum += go_ptr[of * batch + b] * ic_ptr[inf * batch + b];
                        gw_ptr[of * in_feat + inf] += sum;
                    }
                }
            }

            // grad_b += sum(grad_output, dim=batch)
            {
                const double *go_ptr = grad_output.data_ptr();
                double *gb_ptr = grad_b_.data_ptr();
                for (std::size_t of = 0; of < out_feat; ++of)
                {
                    double sum = 0.0;
                    for (std::size_t b = 0; b < batch; ++b)
                        sum += go_ptr[of * batch + b];
                    gb_ptr[of] += sum;
                }
            }

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

            const double *in_ptr = input.data_ptr();
            const auto n = static_cast<long long>(input.size());

            Matrix result(input.rows(), input.cols());
            double *out_ptr = result.data_ptr();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(0LL, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_ptr, out_ptr](long long i) noexcept
                    { out_ptr[i] = in_ptr[i] > 0.0 ? in_ptr[i] : 0.0; });
            } else {
                for (long long i = 0; i < n; ++i)
                    out_ptr[i] = in_ptr[i] > 0.0 ? in_ptr[i] : 0.0;
            }
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("relu backward shape mismatch");

            const double *in_ptr = input_cache_.data_ptr();
            const double *go_ptr = grad_output.data_ptr();
            const auto n = static_cast<long long>(grad_output.size());

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            double *out_ptr = grad_input.data_ptr();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(0LL, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_ptr, go_ptr, out_ptr](long long i) noexcept
                    { out_ptr[i] = in_ptr[i] > 0.0 ? go_ptr[i] : 0.0; });
            } else {
                for (long long i = 0; i < n; ++i)
                    out_ptr[i] = in_ptr[i] > 0.0 ? go_ptr[i] : 0.0;
            }
            return grad_input;
        }
    };

    class GeLU final : public Layer
    {
    private:
        Matrix input_cache_;
        Matrix sigmoid_cache_;  // 缓存 sigmoid(1.702 * x) 用于反向传播
        static constexpr double BETA = 1.702;

    public:
        GeLU() = default;

        // QuickGeLU: x * sigmoid(1.702 * x)
        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            sigmoid_cache_ = Matrix(input.rows(), input.cols());

            const double *in_ptr = input.data_ptr();
            double *sig_ptr = sigmoid_cache_.data_ptr();
            const auto n = static_cast<long long>(input.size());

            Matrix result(input.rows(), input.cols());
            double *out_ptr = result.data_ptr();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(0LL, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_ptr, out_ptr, sig_ptr](long long i) noexcept
                    {
                        double sigmoid_input = BETA * in_ptr[i];
                        double sigmoid_val = 1.0 / (1.0 + std::exp(-sigmoid_input));
                        sig_ptr[i] = sigmoid_val;
                        out_ptr[i] = in_ptr[i] * sigmoid_val;
                    });
            } else {
                for (long long i = 0; i < n; ++i) {
                    double sigmoid_input = BETA * in_ptr[i];
                    double sigmoid_val = 1.0 / (1.0 + std::exp(-sigmoid_input));
                    sig_ptr[i] = sigmoid_val;
                    out_ptr[i] = in_ptr[i] * sigmoid_val;
                }
            }
            return result;
        }

        // d/dx [x * sigmoid(βx)] = sigmoid(βx) + x * β * sigmoid(βx) * (1 - sigmoid(βx))
        //                        = sigmoid(βx) * [1 + βx * (1 - sigmoid(βx))]
        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("gelu backward shape mismatch");

            const double *in_ptr = input_cache_.data_ptr();
            const double *sig_ptr = sigmoid_cache_.data_ptr();
            const double *go_ptr = grad_output.data_ptr();
            const auto n = static_cast<long long>(grad_output.size());

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            double *out_ptr = grad_input.data_ptr();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(0LL, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_ptr, sig_ptr, go_ptr, out_ptr](long long i) noexcept
                    {
                        double s = sig_ptr[i];
                        out_ptr[i] = go_ptr[i] * s * (1.0 + BETA * in_ptr[i] * (1.0 - s));
                    });
            } else {
                for (long long i = 0; i < n; ++i) {
                    double s = sig_ptr[i];
                    out_ptr[i] = go_ptr[i] * s * (1.0 + BETA * in_ptr[i] * (1.0 - s));
                }
            }
            return grad_input;
        }
    };
}

#endif