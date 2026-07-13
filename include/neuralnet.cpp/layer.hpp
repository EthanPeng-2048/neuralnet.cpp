#ifndef LAYER_HPP
#define LAYER_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
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
#include "nn_config.hpp"

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
            const auto n = static_cast<long long>(input.size());
            input_cache_ = input;
            sigmoid_cache_.resize(input.rows(), input.cols());

            const double *in_ptr = input.data_ptr();
            double *sig_ptr = sigmoid_cache_.data_ptr();

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

    class LayerNorm final : public Layer
    {
    private:
        std::size_t normalized_shape_;
        double epsilon_;
        
        // 可学习参数
        Matrix gamma_;      // 缩放参数 (normalized_shape_, 1)
        Matrix beta_;       // 偏移参数 (normalized_shape_, 1)
        Matrix grad_gamma_; // gamma 梯度
        Matrix grad_beta_;  // beta 梯度
        
        // 缓存用于反向传播
        Matrix input_cache_;
        Matrix normalized_cache_;  // 归一化后的值
        Matrix std_cache_;         // 标准差倒数 (1/sqrt(σ² + ε))
        Matrix mean_cache_;        // 均值

        // 数值稳定性
        static constexpr double EPSILON = 1e-5;

    public:
        explicit LayerNorm(std::size_t normalized_shape, double epsilon = EPSILON)
            : normalized_shape_(normalized_shape),
              epsilon_(epsilon),
              gamma_(normalized_shape, 1, 1.0),  // 初始化为1
              beta_(normalized_shape, 1, 0.0),   // 初始化为0
              grad_gamma_(normalized_shape, 1),
              grad_beta_(normalized_shape, 1)
        {
            // 参数初始化已在构造函数中完成
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(gamma_), std::ref(beta_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(grad_gamma_), std::ref(grad_beta_)};
        }

        Matrix forward(const Matrix &input) override
        {
            // 输入形状: (normalized_shape_, batch_size)
            if (input.rows() != normalized_shape_)
                throw std::invalid_argument("layer_norm forward input shape mismatch");

            input_cache_ = input;
            const std::size_t batch_size = input.cols();
            const std::size_t features = input.rows();

            // 缓存中间结果（复用已有内存）
            mean_cache_.resize(1, batch_size);
            std_cache_.resize(1, batch_size);
            normalized_cache_.resize(features, batch_size);

            Matrix result(features, batch_size);

            const double *in_ptr = input.data_ptr();
            const double *gamma_ptr = gamma_.data_ptr();
            const double *beta_ptr = beta_.data_ptr();
            double *mean_ptr = mean_cache_.data_ptr();
            double *std_ptr = std_cache_.data_ptr();
            double *norm_ptr = normalized_cache_.data_ptr();
            double *res_ptr = result.data_ptr();

            // 对每个样本（列）独立处理
            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);
            
            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD) {
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(),
                    [in_ptr, gamma_ptr, beta_ptr, mean_ptr, std_ptr, norm_ptr, res_ptr, 
                     features, batch_size, epsilon = epsilon_](std::size_t b) noexcept
                    {
                        // 计算均值
                        double sum = 0.0;
                        for (std::size_t f = 0; f < features; ++f)
                            sum += in_ptr[f * batch_size + b];
                        double mean = sum / static_cast<double>(features);
                        mean_ptr[b] = mean;

                        // 计算方差
                        double var_sum = 0.0;
                        for (std::size_t f = 0; f < features; ++f) {
                            double diff = in_ptr[f * batch_size + b] - mean;
                            var_sum += diff * diff;
                        }
                        double variance = var_sum / static_cast<double>(features);
                        double std_inv = 1.0 / std::sqrt(variance + epsilon);
                        std_ptr[b] = std_inv;

                        // 归一化和仿射变换
                        for (std::size_t f = 0; f < features; ++f) {
                            double normalized = (in_ptr[f * batch_size + b] - mean) * std_inv;
                            norm_ptr[f * batch_size + b] = normalized;
                            res_ptr[f * batch_size + b] = gamma_ptr[f] * normalized + beta_ptr[f];
                        }
                    });
            } else {
                for (std::size_t b = 0; b < batch_size; ++b) {
                    // 计算均值
                    double sum = 0.0;
                    for (std::size_t f = 0; f < features; ++f)
                        sum += in_ptr[f * batch_size + b];
                    double mean = sum / static_cast<double>(features);
                    mean_ptr[b] = mean;

                    // 计算方差
                    double var_sum = 0.0;
                    for (std::size_t f = 0; f < features; ++f) {
                        double diff = in_ptr[f * batch_size + b] - mean;
                        var_sum += diff * diff;
                    }
                    double variance = var_sum / static_cast<double>(features);
                    double std_inv = 1.0 / std::sqrt(variance + epsilon_);
                    std_ptr[b] = std_inv;

                    // 归一化和仿射变换
                    for (std::size_t f = 0; f < features; ++f) {
                        double normalized = (in_ptr[f * batch_size + b] - mean) * std_inv;
                        norm_ptr[f * batch_size + b] = normalized;
                        res_ptr[f * batch_size + b] = gamma_ptr[f] * normalized + beta_ptr[f];
                    }
                }
            }

            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != normalized_shape_)
                throw std::invalid_argument("layer_norm backward shape mismatch");

            const std::size_t features = normalized_shape_;
            const std::size_t batch_size = grad_output.cols();

            Matrix grad_input(features, batch_size);

            const double *go_ptr = grad_output.data_ptr();
            const double *norm_ptr = normalized_cache_.data_ptr();
            const double *std_ptr = std_cache_.data_ptr();
            const double *gamma_ptr = gamma_.data_ptr();
            double *gi_ptr = grad_input.data_ptr();
            double *gg_ptr = grad_gamma_.data_ptr();
            double *gb_ptr = grad_beta_.data_ptr();

            // 计算梯度
            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);
            
            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD) {
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(),
                    [go_ptr, norm_ptr, std_ptr, gamma_ptr, gi_ptr, gg_ptr, gb_ptr,
                     features, batch_size](std::size_t b) noexcept
                    {
                        double std_inv = std_ptr[b];
                        
                        // 预计算统计量：O(N) 而非 O(N²)
                        double sum_grad = 0.0;
                        double sum_grad_norm = 0.0;
                        for (std::size_t f = 0; f < features; ++f) {
                            double g = go_ptr[f * batch_size + b] * gamma_ptr[f];
                            sum_grad += g;
                            sum_grad_norm += g * norm_ptr[f * batch_size + b];
                        }

                        // 单次遍历：同时计算 dL/dγ、dL/dβ、dL/dx
                        const double inv_features = 1.0 / static_cast<double>(features);
                        for (std::size_t f = 0; f < features; ++f) {
                            double grad_out = go_ptr[f * batch_size + b];
                            double g = grad_out * gamma_ptr[f];
                            gg_ptr[f] += grad_out * norm_ptr[f * batch_size + b];
                            gb_ptr[f] += grad_out;
                            gi_ptr[f * batch_size + b] = (g - sum_grad * inv_features -
                                   norm_ptr[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
                        }
                    });
            } else {
                for (std::size_t b = 0; b < batch_size; ++b) {
                    double std_inv = std_ptr[b];
                    
                    // 预计算统计量：O(N) 而非 O(N²)
                    double sum_grad = 0.0;
                    double sum_grad_norm = 0.0;
                    for (std::size_t f = 0; f < features; ++f) {
                        double g = go_ptr[f * batch_size + b] * gamma_ptr[f];
                        sum_grad += g;
                        sum_grad_norm += g * norm_ptr[f * batch_size + b];
                    }

                    // 单次遍历：同时计算 dL/dγ、dL/dβ、dL/dx
                    const double inv_features = 1.0 / static_cast<double>(features);
                    for (std::size_t f = 0; f < features; ++f) {
                        double grad_out = go_ptr[f * batch_size + b];
                        double g = grad_out * gamma_ptr[f];
                        gg_ptr[f] += grad_out * norm_ptr[f * batch_size + b];
                        gb_ptr[f] += grad_out;
                        gi_ptr[f * batch_size + b] = (g - sum_grad * inv_features -
                               norm_ptr[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
                    }
                }
            }

            return grad_input;
        }

    };

    // ── Softmax 激活层 ─────────────────────────────────────────────────────
    // 对每一行独立应用 softmax：out[i][j] = exp(in[i][j]) / Σ_k exp(in[i][k])
    // 输入形状: (rows, cols)，输出形状相同。
    // 使用最大值减法保证数值稳定性。
    // 反向传播: grad[i][j] = out[i][j] * (grad_out[i][j] - Σ_k out[i][k] * grad_out[i][k])
    // ────────────────────────────────────────────────────────────────────────
    class Softmax final : public Layer
    {
    private:
        Matrix output_cache_;

    public:
        Softmax() = default;

        Matrix forward(const Matrix &input) override
        {
            const std::size_t rows = input.rows();
            const std::size_t cols = input.cols();
            output_cache_.resize(rows, cols);

            Matrix result(rows, cols);
            const double *in_ptr = input.data_ptr();
            double *out_ptr = result.data_ptr();
            double *cache_ptr = output_cache_.data_ptr();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const std::size_t total = rows * cols;

            auto process_row = [in_ptr, out_ptr, cache_ptr, cols](std::size_t r) noexcept
            {
                const std::size_t offset = r * cols;
                const double *row_in = in_ptr + offset;
                double *row_out = out_ptr + offset;
                double *row_cache = cache_ptr + offset;

                // 数值稳定：减去行内最大值
                double max_val = row_in[0];
                for (std::size_t c = 1; c < cols; ++c)
                    max_val = std::max(max_val, row_in[c]);

                // 计算 exp 和求和
                double sum = 0.0;
                for (std::size_t c = 0; c < cols; ++c)
                {
                    double e = std::exp(row_in[c] - max_val);
                    row_out[c] = e;
                    row_cache[c] = e;
                    sum += e;
                }

                // 归一化
                const double inv_sum = 1.0 / sum;
                for (std::size_t c = 0; c < cols; ++c)
                {
                    row_out[c] *= inv_sum;
                    row_cache[c] *= inv_sum;
                }
            };

            if (total >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(row_indices.begin(), row_indices.end(), process_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    process_row(r);

            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t rows = output_cache_.rows();
            const std::size_t cols = output_cache_.cols();

            if (grad_output.rows() != rows || grad_output.cols() != cols)
                throw std::invalid_argument("softmax backward shape mismatch");

            Matrix grad_input(rows, cols);

            const double *go_ptr = grad_output.data_ptr();
            const double *out_ptr = output_cache_.data_ptr();
            double *gi_ptr = grad_input.data_ptr();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const std::size_t total = rows * cols;

            auto process_row = [go_ptr, out_ptr, gi_ptr, cols](std::size_t r) noexcept
            {
                const std::size_t offset = r * cols;
                const double *row_go = go_ptr + offset;
                const double *row_out = out_ptr + offset;
                double *row_gi = gi_ptr + offset;

                // dot = Σ_k out[k] * grad_out[k]
                double dot = 0.0;
                for (std::size_t c = 0; c < cols; ++c)
                    dot += row_out[c] * row_go[c];

                // grad = out * (grad_out - dot)
                for (std::size_t c = 0; c < cols; ++c)
                    row_gi[c] = row_out[c] * (row_go[c] - dot);
            };

            if (total >= SmartPolicy::PARALLEL_THRESHOLD)
                SmartPolicy::for_each(row_indices.begin(), row_indices.end(), process_row);
            else
                for (std::size_t r = 0; r < rows; ++r)
                    process_row(r);

            return grad_input;
        }
    };

    // ── 多头注意力层 ──────────────────────────────────────────────────────
    // 来源: "Attention Is All You Need" (Vaswani et al., 2017)
    //
    //   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    //   head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    //   Attention(Q, K, V) = softmax(QK^T / √d_k) V
    //
    // 输入形状: (d_model, seq_len)，输出形状: (d_model, seq_len)
    // ─────────────────────────────────────────────────────────────────────
    class MultiHeadAttention final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t num_heads_;
        std::size_t d_k_;
        double scale_;  // 1.0 / sqrt(d_k)

        // 投影层
        Linear W_q_;
        Linear W_k_;
        Linear W_v_;
        Linear W_o_;

        // 前向传播缓存
        Matrix Q_cache_;           // (d_model, seq_len)
        Matrix K_cache_;           // (d_model, seq_len)
        Matrix V_cache_;           // (d_model, seq_len)
        std::vector<Matrix> attn_; // num_heads_ × (seq_len, seq_len) — softmax 输出

        // Per-head 切片缓存（forward 提取，backward 使用）
        std::vector<Matrix> Q_heads_;  // num_heads_ × (d_k_, seq_len)
        std::vector<Matrix> K_heads_;
        std::vector<Matrix> V_heads_;
        std::vector<Matrix> O_heads_;  // num_heads_ × (d_k_, seq_len) — 前向输出缓存

        // 辅助缓冲区（避免循环内重复分配）
        Matrix grad_scores_buf_;  // (seq_len, seq_len)
        Matrix grad_A_buf_;       // (seq_len, seq_len)
        Matrix grad_O_h_buf_;     // (d_k_, seq_len) — backward 中复用
        Softmax softmax_;         // 复用，避免每头构造/析构

        // ── 从矩阵中提取行切片到 dst ────────────────────────────────────
        static void extract_rows(const Matrix &src, std::size_t row_start,
                                 std::size_t row_count, Matrix &dst)
        {
            const std::size_t cols = src.cols();
            dst.resize(row_count, cols);
            const double *src_ptr = src.data_ptr();
            double *dst_ptr = dst.data_ptr();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_ptr + (row_start + r) * cols, cols, dst_ptr + r * cols);
        }

        // ── 将 src 写入 dst 的指定行范围 ────────────────────────────────
        static void insert_rows(Matrix &dst, std::size_t row_start,
                                const Matrix &src)
        {
            const std::size_t row_count = src.rows();
            const std::size_t cols = src.cols();
            double *dst_ptr = dst.data_ptr();
            const double *src_ptr = src.data_ptr();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_ptr + r * cols, cols, dst_ptr + (row_start + r) * cols);
        }

        // ── 逐元素缩放 ──────────────────────────────────────────────────
        static void scale_inplace(Matrix &m, double s)
        {
            double *ptr = m.data_ptr();
            const auto n = static_cast<long long>(m.size());
            if (n >= SmartPolicy::PARALLEL_THRESHOLD)
            {
                auto indices = std::views::iota(0LL, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [ptr, s](long long i) noexcept { ptr[i] *= s; });
            }
            else
            {
                for (long long i = 0; i < n; ++i)
                    ptr[i] *= s;
            }
        }

    public:
        MultiHeadAttention(std::size_t d_model, std::size_t num_heads)
            : d_model_(d_model),
              num_heads_(num_heads),
              d_k_(d_model / num_heads),
              scale_(1.0 / std::sqrt(static_cast<double>(d_model / num_heads))),
              W_q_(d_model, d_model),
              W_k_(d_model, d_model),
              W_v_(d_model, d_model),
              W_o_(d_model, d_model)
        {
            if (d_model % num_heads != 0)
                throw std::invalid_argument("MultiHeadAttention: d_model must be divisible by num_heads");
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = W_q_.parameters();
            auto wk = W_k_.parameters();
            auto wv = W_v_.parameters();
            auto wo = W_o_.parameters();
            params.insert(params.end(), wk.begin(), wk.end());
            params.insert(params.end(), wv.begin(), wv.end());
            params.insert(params.end(), wo.begin(), wo.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = W_q_.param_gradients();
            auto gk = W_k_.param_gradients();
            auto gv = W_v_.param_gradients();
            auto go = W_o_.param_gradients();
            grads.insert(grads.end(), gk.begin(), gk.end());
            grads.insert(grads.end(), gv.begin(), gv.end());
            grads.insert(grads.end(), go.begin(), go.end());
            return grads;
        }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                throw std::invalid_argument("MultiHeadAttention forward input shape mismatch");

            const std::size_t seq_len = input.cols();

            // ── 1. 线性投影 Q, K, V ──────────────────────────────────────
            Q_cache_ = W_q_.forward(input);
            K_cache_ = W_k_.forward(input);
            V_cache_ = W_v_.forward(input);

            // ── 2. 分配 per-head 缓冲区 ──────────────────────────────────
            Q_heads_.resize(num_heads_);
            K_heads_.resize(num_heads_);
            V_heads_.resize(num_heads_);
            O_heads_.resize(num_heads_);
            attn_.resize(num_heads_);

            // ── 3. 逐头计算缩放点积注意力 ────────────────────────────────
            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t row_start = h * d_k_;

                // 提取 Q_h, K_h, V_h → (d_k_, seq_len)
                extract_rows(Q_cache_, row_start, d_k_, Q_heads_[h]);
                extract_rows(K_cache_, row_start, d_k_, K_heads_[h]);
                extract_rows(V_cache_, row_start, d_k_, V_heads_[h]);

                // S_h = Q_h^T @ K_h → (seq_len, seq_len)
                Matrix QhT = Q_heads_[h].transpose();
                attn_[h].resize(seq_len, seq_len);
                QhT.multiply_to(attn_[h], K_heads_[h]);

                // S_h *= 1/√d_k
                scale_inplace(attn_[h], scale_);

                // A_h = softmax(S_h) — 按行 softmax（复用成员 softmax_）
                attn_[h] = softmax_.forward(attn_[h]);

                // O_h = V_h @ A_h → (d_k_, seq_len)，缓存到 O_heads_
                O_heads_[h].resize(d_k_, seq_len);
                V_heads_[h].multiply_to(O_heads_[h], attn_[h]);
            }

            // ── 4. 拼接所有头的输出并投影 ────────────────────────────────
            Matrix output(d_model_, seq_len);
            for (std::size_t h = 0; h < num_heads_; ++h)
                insert_rows(output, h * d_k_, O_heads_[h]);

            return W_o_.forward(output);
        }

        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t seq_len = grad_output.cols();

            // ── 1. 输出投影反向 ──────────────────────────────────────────
            Matrix grad_concat = W_o_.backward(grad_output);  // (d_model, seq_len)

            // ── 2. 初始化各头梯度累加矩阵 ────────────────────────────────
            Matrix grad_Q_all(d_model_, seq_len);
            Matrix grad_K_all(d_model_, seq_len);
            Matrix grad_V_all(d_model_, seq_len);
            {
                const auto n = static_cast<long long>(d_model_ * seq_len);
                std::fill(grad_Q_all.data_ptr(), grad_Q_all.data_ptr() + n, 0.0);
                std::fill(grad_K_all.data_ptr(), grad_K_all.data_ptr() + n, 0.0);
                std::fill(grad_V_all.data_ptr(), grad_V_all.data_ptr() + n, 0.0);
            }

            // ── 3. 逐头计算注意力反向 ────────────────────────────────────
            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t row_start = h * d_k_;
                const Matrix &Q_h = Q_heads_[h];    // (d_k_, seq_len)
                const Matrix &K_h = K_heads_[h];    // (d_k_, seq_len)
                const Matrix &V_h = V_heads_[h];    // (d_k_, seq_len)
                const Matrix &A_h = attn_[h];       // (seq_len, seq_len)

                // 提取 grad_O_h（复用成员缓冲区）
                extract_rows(grad_concat, row_start, d_k_, grad_O_h_buf_);

                // grad_V_h = grad_O_h @ A_h^T → (d_k_, seq_len)
                Matrix AhT = A_h.transpose();
                Matrix grad_V_h(d_k_, seq_len);
                grad_O_h_buf_.multiply_to(grad_V_h, AhT);

                // grad_A_h = V_h^T @ grad_O_h → (seq_len, seq_len)
                Matrix VhT = V_h.transpose();
                grad_A_buf_.resize(seq_len, seq_len);
                VhT.multiply_to(grad_A_buf_, grad_O_h_buf_);

                // grad_S_h = A_h ⊙ (grad_A_h - row_sum(A_h ⊙ grad_A_h))
                grad_scores_buf_.resize(seq_len, seq_len);
                {
                    const double *a_ptr = A_h.data_ptr();
                    const double *ga_ptr = grad_A_buf_.data_ptr();
                    double *gs_ptr = grad_scores_buf_.data_ptr();

                    for (std::size_t i = 0; i < seq_len; ++i)
                    {
                        double dot = 0.0;
                        for (std::size_t j = 0; j < seq_len; ++j)
                            dot += a_ptr[i * seq_len + j] * ga_ptr[i * seq_len + j];
                        for (std::size_t j = 0; j < seq_len; ++j)
                            gs_ptr[i * seq_len + j] =
                                a_ptr[i * seq_len + j] * (ga_ptr[i * seq_len + j] - dot);
                    }
                }

                // grad_Q_h = K_h @ grad_S_h^T * scale → (d_k_, seq_len)
                Matrix gsT = grad_scores_buf_.transpose();
                Matrix grad_Q_h(d_k_, seq_len);
                K_h.multiply_to(grad_Q_h, gsT);
                scale_inplace(grad_Q_h, scale_);

                // grad_K_h = Q_h @ grad_S_h * scale → (d_k_, seq_len)
                Matrix grad_K_h(d_k_, seq_len);
                Q_h.multiply_to(grad_K_h, grad_scores_buf_);
                scale_inplace(grad_K_h, scale_);

                // 累加到全局梯度（每个头写入不同行，无重叠）
                insert_rows(grad_Q_all, row_start, grad_Q_h);
                insert_rows(grad_K_all, row_start, grad_K_h);
                insert_rows(grad_V_all, row_start, grad_V_h);
            }

            // ── 4. 投影层反向，累加输入梯度 ──────────────────────────────
            Matrix grad_input = W_q_.backward(grad_Q_all);
            grad_input = grad_input + W_k_.backward(grad_K_all);
            grad_input = grad_input + W_v_.backward(grad_V_all);

            return grad_input;
        }
    };

    // ── 正弦波固定位置编码 ───────────────────────────────────────────────
    // 来源: "Attention Is All You Need" (Vaswani et al., 2017)
    //
    //   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    //   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    //
    // 输入形状: (d_model, seq_len)，输出形状相同。
    // 编码矩阵在构造时一次性预计算，前向传播仅做逐元素加法。
    // 反向传播直接穿透（编码不可学习）。
    // ─────────────────────────────────────────────────────────────────────
    class PositionalEncoding final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t max_len_;
        Matrix encoding_;   // (d_model, max_len) — 预计算的正弦波编码

    public:
        PositionalEncoding(std::size_t d_model, std::size_t max_len = 5000)
            : d_model_(d_model),
              max_len_(max_len),
              encoding_(d_model, max_len)
        {
            // ── 一次性预计算频率与编码 ──
            // 先计算每个特征对的角频率（避免在 position 循环中重复 pow）
            const std::size_t half = d_model / 2;
            std::vector<double> freqs(half);
            for (std::size_t i = 0; i < half; ++i)
                freqs[i] = 1.0 / std::pow(10000.0, static_cast<double>(2 * i) / d_model);

            double *e_ptr = encoding_.data_ptr();
            for (std::size_t pos = 0; pos < max_len; ++pos)
            {
                const double pos_d = static_cast<double>(pos);
                for (std::size_t i = 0; i < half; ++i)
                {
                    const double angle = pos_d * freqs[i];
                    e_ptr[(2 * i)       * max_len + pos] = std::sin(angle);
                    e_ptr[(2 * i + 1)   * max_len + pos] = std::cos(angle);
                }
                // 奇数维度：最后一个特征仅使用 sin
                if (d_model % 2 == 1)
                {
                    const double angle = pos_d * freqs[half];
                    e_ptr[(d_model - 1) * max_len + pos] = std::sin(angle);
                }
            }
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override { return {}; }
        std::vector<std::reference_wrapper<Matrix>> param_gradients() override { return {}; }

        Matrix forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                throw std::invalid_argument("positional encoding forward: d_model mismatch");

            const std::size_t seq_len = input.cols();

            if (seq_len > max_len_)
                throw std::invalid_argument("positional encoding forward: sequence length exceeds max_len");

            // result = input + encoding[:, 0:seq_len]
            Matrix result(d_model_, seq_len);
            const double *in_ptr = input.data_ptr();
            const double *e_ptr = encoding_.data_ptr();
            double *out_ptr = result.data_ptr();

            const auto total = static_cast<std::size_t>(d_model_ * seq_len);
            const std::size_t max_len = max_len_;
            auto indices = std::views::iota(std::size_t{0}, total);
            SmartPolicy::for_each(indices.begin(), indices.end(),
                [in_ptr, e_ptr, out_ptr, seq_len, max_len](std::size_t idx) noexcept
                {
                    // idx = row * seq_len + col
                    const std::size_t row = idx / seq_len;
                    const std::size_t col = idx % seq_len;
                    out_ptr[idx] = in_ptr[idx] + e_ptr[row * max_len + col];
                });

            return result;
        }

        // 位置编码为固定值，梯度直接穿透
        Matrix backward(const Matrix &grad_output) override { return grad_output; }
    };
}

#endif