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
#include <span>
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
            const auto prod_span = product_buf_.span();
            const auto bias_span = b_.span();
            auto res_span = result.span();
            const auto total = static_cast<std::size_t>(out_feat * batch);

            auto indices = std::views::iota(std::size_t{0}, total);
            SmartPolicy::for_each(indices.begin(), indices.end(),
                          [prod_span, bias_span, res_span, batch](std::size_t idx) noexcept
                          {
                              res_span[idx] = prod_span[idx] + bias_span[idx / batch];
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
                const auto go_span = grad_output.span();
                const auto ic_span = input_cache_.span();
                auto gw_span = grad_W_.span();
                const std::size_t gw_size = out_feat * in_feat;
                auto gw_indices = std::views::iota(std::size_t{0}, gw_size);
                SmartPolicy::for_each(gw_indices.begin(), gw_indices.end(),
                    [go_span, ic_span, gw_span, in_feat, batch](std::size_t idx) noexcept
                    {
                        const std::size_t of = idx / in_feat;
                        const std::size_t inf = idx % in_feat;
                        double sum = 0.0;
                        for (std::size_t b = 0; b < batch; ++b)
                            sum += go_span[of * batch + b] * ic_span[inf * batch + b];
                        gw_span[idx] += sum;
                    });
            }

            // grad_b += sum(grad_output, dim=batch)
            {
                const auto go_span = grad_output.span();
                auto gb_span = grad_b_.span();
                for (std::size_t of = 0; of < out_feat; ++of)
                {
                    double sum = 0.0;
                    for (std::size_t b = 0; b < batch; ++b)
                        sum += go_span[of * batch + b];
                    gb_span[of] += sum;
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

            const auto in_span = input.span();
            const auto n = input.size();

            Matrix result(input.rows(), input.cols());
            auto out_span = result.span();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(std::size_t{0}, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_span, out_span](std::size_t i) noexcept
                    { out_span[i] = in_span[i] > 0.0 ? in_span[i] : 0.0; });
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    out_span[i] = in_span[i] > 0.0 ? in_span[i] : 0.0;
            }
            return result;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
                throw std::invalid_argument("relu backward shape mismatch");

            const auto in_span = input_cache_.span();
            const auto go_span = grad_output.span();
            const auto n = grad_output.size();

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            auto out_span = grad_input.span();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(std::size_t{0}, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_span, go_span, out_span](std::size_t i) noexcept
                    { out_span[i] = in_span[i] > 0.0 ? go_span[i] : 0.0; });
            } else {
                for (std::size_t i = 0; i < n; ++i)
                    out_span[i] = in_span[i] > 0.0 ? go_span[i] : 0.0;
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
            const auto n = input.size();
            input_cache_ = input;
            sigmoid_cache_.resize(input.rows(), input.cols());

            const auto in_span = input.span();
            auto sig_span = sigmoid_cache_.span();

            Matrix result(input.rows(), input.cols());
            auto out_span = result.span();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(std::size_t{0}, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_span, out_span, sig_span](std::size_t i) noexcept
                    {
                        double sigmoid_input = BETA * in_span[i];
                        double sigmoid_val = 1.0 / (1.0 + std::exp(-sigmoid_input));
                        sig_span[i] = sigmoid_val;
                        out_span[i] = in_span[i] * sigmoid_val;
                    });
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    double sigmoid_input = BETA * in_span[i];
                    double sigmoid_val = 1.0 / (1.0 + std::exp(-sigmoid_input));
                    sig_span[i] = sigmoid_val;
                    out_span[i] = in_span[i] * sigmoid_val;
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

            const auto in_span = input_cache_.span();
            const auto sig_span = sigmoid_cache_.span();
            const auto go_span = grad_output.span();
            const auto n = grad_output.size();

            Matrix grad_input(grad_output.rows(), grad_output.cols());
            auto out_span = grad_input.span();

            if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
                auto indices = std::views::iota(std::size_t{0}, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [in_span, sig_span, go_span, out_span](std::size_t i) noexcept
                    {
                        double s = sig_span[i];
                        out_span[i] = go_span[i] * s * (1.0 + BETA * in_span[i] * (1.0 - s));
                    });
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    double s = sig_span[i];
                    out_span[i] = go_span[i] * s * (1.0 + BETA * in_span[i] * (1.0 - s));
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

            const auto in_span = input.span();
            const auto gamma_span = gamma_.span();
            const auto beta_span = beta_.span();
            auto mean_span = mean_cache_.span();
            auto std_span = std_cache_.span();
            auto norm_span = normalized_cache_.span();
            auto res_span = result.span();

            // 对每个样本（列）独立处理
            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);
            
            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD) {
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(),
                    [in_span, gamma_span, beta_span, mean_span, std_span, norm_span, res_span, 
                     features, batch_size, epsilon = epsilon_](std::size_t b) noexcept
                    {
                        // 计算均值
                        double sum = 0.0;
                        for (std::size_t f = 0; f < features; ++f)
                            sum += in_span[f * batch_size + b];
                        double mean = sum / static_cast<double>(features);
                        mean_span[b] = mean;

                        // 计算方差
                        double var_sum = 0.0;
                        for (std::size_t f = 0; f < features; ++f) {
                            double diff = in_span[f * batch_size + b] - mean;
                            var_sum += diff * diff;
                        }
                        double variance = var_sum / static_cast<double>(features);
                        double std_inv = 1.0 / std::sqrt(variance + epsilon);
                        std_span[b] = std_inv;

                        // 归一化和仿射变换
                        for (std::size_t f = 0; f < features; ++f) {
                            double normalized = (in_span[f * batch_size + b] - mean) * std_inv;
                            norm_span[f * batch_size + b] = normalized;
                            res_span[f * batch_size + b] = gamma_span[f] * normalized + beta_span[f];
                        }
                    });
            } else {
                for (std::size_t b = 0; b < batch_size; ++b) {
                    // 计算均值
                    double sum = 0.0;
                    for (std::size_t f = 0; f < features; ++f)
                        sum += in_span[f * batch_size + b];
                    double mean = sum / static_cast<double>(features);
                    mean_span[b] = mean;

                    // 计算方差
                    double var_sum = 0.0;
                    for (std::size_t f = 0; f < features; ++f) {
                        double diff = in_span[f * batch_size + b] - mean;
                        var_sum += diff * diff;
                    }
                    double variance = var_sum / static_cast<double>(features);
                    double std_inv = 1.0 / std::sqrt(variance + epsilon_);
                    std_span[b] = std_inv;

                    // 归一化和仿射变换
                    for (std::size_t f = 0; f < features; ++f) {
                        double normalized = (in_span[f * batch_size + b] - mean) * std_inv;
                        norm_span[f * batch_size + b] = normalized;
                        res_span[f * batch_size + b] = gamma_span[f] * normalized + beta_span[f];
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

            const auto go_span = grad_output.span();
            const auto norm_span = normalized_cache_.span();
            const auto std_span = std_cache_.span();
            const auto gamma_span = gamma_.span();
            auto gi_span = grad_input.span();
            auto gg_span = grad_gamma_.span();
            auto gb_span = grad_beta_.span();

            // 计算梯度
            auto batch_indices = std::views::iota(std::size_t{0}, batch_size);
            
            if (batch_size >= SmartPolicy::PARALLEL_THRESHOLD) {
                // 先并行计算每个样本的 dL/dx（无竞争），
                // 再串行累加 dL/dγ 和 dL/dβ（避免数据竞争）
                SmartPolicy::for_each(batch_indices.begin(), batch_indices.end(),
                    [go_span, norm_span, std_span, gamma_span, gi_span,
                     features, batch_size](std::size_t b) noexcept
                    {
                        double std_inv = std_span[b];
                        double sum_grad = 0.0;
                        double sum_grad_norm = 0.0;
                        for (std::size_t f = 0; f < features; ++f) {
                            double g = go_span[f * batch_size + b] * gamma_span[f];
                            sum_grad += g;
                            sum_grad_norm += g * norm_span[f * batch_size + b];
                        }
                        const double inv_features = 1.0 / static_cast<double>(features);
                        for (std::size_t f = 0; f < features; ++f) {
                            double grad_out = go_span[f * batch_size + b];
                            double g = grad_out * gamma_span[f];
                            gi_span[f * batch_size + b] = (g - sum_grad * inv_features -
                                   norm_span[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
                        }
                    });
                // 串行累加 dL/dγ 和 dL/dβ
                for (std::size_t b = 0; b < batch_size; ++b) {
                    for (std::size_t f = 0; f < features; ++f) {
                        double grad_out = go_span[f * batch_size + b];
                        gg_span[f] += grad_out * norm_span[f * batch_size + b];
                        gb_span[f] += grad_out;
                    }
                }
            } else {
                for (std::size_t b = 0; b < batch_size; ++b) {
                    double std_inv = std_span[b];
                    
                    // 预计算统计量：O(N) 而非 O(N²)
                    double sum_grad = 0.0;
                    double sum_grad_norm = 0.0;
                    for (std::size_t f = 0; f < features; ++f) {
                        double g = go_span[f * batch_size + b] * gamma_span[f];
                        sum_grad += g;
                        sum_grad_norm += g * norm_span[f * batch_size + b];
                    }

                    // 单次遍历：同时计算 dL/dγ、dL/dβ、dL/dx
                    const double inv_features = 1.0 / static_cast<double>(features);
                    for (std::size_t f = 0; f < features; ++f) {
                        double grad_out = go_span[f * batch_size + b];
                        double g = grad_out * gamma_span[f];
                        gg_span[f] += grad_out * norm_span[f * batch_size + b];
                        gb_span[f] += grad_out;
                        gi_span[f * batch_size + b] = (g - sum_grad * inv_features -
                               norm_span[f * batch_size + b] * sum_grad_norm * inv_features) * std_inv;
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
            const auto in_span = input.span();
            auto out_span = result.span();
            auto cache_span = output_cache_.span();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const std::size_t total = rows * cols;

            auto process_row = [in_span, out_span, cache_span, cols](std::size_t r) noexcept
            {
                const std::size_t offset = r * cols;
                const auto row_in = in_span.subspan(offset, cols);
                auto row_out = out_span.subspan(offset, cols);
                auto row_cache = cache_span.subspan(offset, cols);

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

            const auto go_span = grad_output.span();
            const auto out_span = output_cache_.span();
            auto gi_span = grad_input.span();

            auto row_indices = std::views::iota(std::size_t{0}, rows);
            const std::size_t total = rows * cols;

            auto process_row = [go_span, out_span, gi_span, cols](std::size_t r) noexcept
            {
                const std::size_t offset = r * cols;
                const auto row_go = go_span.subspan(offset, cols);
                const auto row_out = out_span.subspan(offset, cols);
                auto row_gi = gi_span.subspan(offset, cols);

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
            const auto src_span = src.span();
            auto dst_span = dst.span();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_span.begin() + (row_start + r) * cols, cols, dst_span.begin() + r * cols);
        }

        // ── 将 src 写入 dst 的指定行范围 ────────────────────────────────
        static void insert_rows(Matrix &dst, std::size_t row_start,
                                const Matrix &src)
        {
            const std::size_t row_count = src.rows();
            const std::size_t cols = src.cols();
            auto dst_span = dst.span();
            const auto src_span = src.span();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_span.begin() + r * cols, cols, dst_span.begin() + (row_start + r) * cols);
        }

        // ── 逐元素缩放 ──────────────────────────────────────────────────
        static void scale_inplace(Matrix &m, double s)
        {
            auto m_span = m.span();
            const auto n = m.size();
            if (n >= SmartPolicy::PARALLEL_THRESHOLD)
            {
                auto indices = std::views::iota(std::size_t{0}, n);
                SmartPolicy::for_each(indices.begin(), indices.end(),
                    [m_span, s](std::size_t i) noexcept { m_span[i] *= s; });
            }
            else
            {
                for (std::size_t i = 0; i < n; ++i)
                    m_span[i] *= s;
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
                grad_Q_all.zero();
                grad_K_all.zero();
                grad_V_all.zero();
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
                    const auto a_span = A_h.span();
                    const auto ga_span = grad_A_buf_.span();
                    auto gs_span = grad_scores_buf_.span();

                    for (std::size_t i = 0; i < seq_len; ++i)
                    {
                        double dot = 0.0;
                        for (std::size_t j = 0; j < seq_len; ++j)
                            dot += a_span[i * seq_len + j] * ga_span[i * seq_len + j];
                        for (std::size_t j = 0; j < seq_len; ++j)
                            gs_span[i * seq_len + j] =
                                a_span[i * seq_len + j] * (ga_span[i * seq_len + j] - dot);
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

            auto e_span = encoding_.span();
            for (std::size_t pos = 0; pos < max_len; ++pos)
            {
                const double pos_d = static_cast<double>(pos);
                for (std::size_t i = 0; i < half; ++i)
                {
                    const double angle = pos_d * freqs[i];
                    e_span[(2 * i)       * max_len + pos] = std::sin(angle);
                    e_span[(2 * i + 1)   * max_len + pos] = std::cos(angle);
                }
                // 奇数维度：最后一个特征仅使用 sin
                if (d_model % 2 == 1)
                {
                    const double freq_last = 1.0 / std::pow(10000.0, static_cast<double>(2 * half) / d_model);
                    const double angle = pos_d * freq_last;
                    e_span[(d_model - 1) * max_len + pos] = std::sin(angle);
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
            const auto in_span = input.span();
            const auto e_span = encoding_.span();
            auto out_span = result.span();

            const auto total = static_cast<std::size_t>(d_model_ * seq_len);
            const std::size_t max_len = max_len_;
            auto indices = std::views::iota(std::size_t{0}, total);
            SmartPolicy::for_each(indices.begin(), indices.end(),
                [in_span, e_span, out_span, seq_len, max_len](std::size_t idx) noexcept
                {
                    // idx = row * seq_len + col
                    const std::size_t row = idx / seq_len;
                    const std::size_t col = idx % seq_len;
                    out_span[idx] = in_span[idx] + e_span[row * max_len + col];
                });

            return result;
        }

        // 位置编码为固定值，梯度直接穿透
        Matrix backward(const Matrix &grad_output) override { return grad_output; }
    };

    // ── 前馈网络 (Feed-Forward Network) ────────────────────────────────
    // FFN(x) = Linear(d_ff → d_model)(GeLU(Linear(d_model → d_ff)(x)))
    // 输入/输出形状: (d_model, seq_len)
    class FeedForward final : public Layer
    {
    private:
        Linear fc1_;       // (d_ff, d_model)
        Linear fc2_;       // (d_model, d_ff)
        GeLU activation_;
        Matrix input_cache_;

    public:
        FeedForward(std::size_t d_model, std::size_t d_ff)
            : fc1_(d_model, d_ff), fc2_(d_ff, d_model)
        {}

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = fc1_.parameters();
            auto p2 = fc2_.parameters();
            params.insert(params.end(), p2.begin(), p2.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = fc1_.param_gradients();
            auto g2 = fc2_.param_gradients();
            grads.insert(grads.end(), g2.begin(), g2.end());
            return grads;
        }

        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            auto x = activation_.forward(fc1_.forward(input));
            return fc2_.forward(x);
        }

        Matrix backward(const Matrix &grad_output) override
        {
            auto grad = fc2_.backward(grad_output);
            grad = activation_.backward(grad);
            return fc1_.backward(grad);
        }
    };

    // ── Transformer 编码器层 (Pre-Norm 架构) ──────────────────────────
    //   x = x + SelfAttn(LayerNorm₁(x))    ← 残差连接
    //   x = x + FFN(LayerNorm₂(x))         ← 残差连接
    // 输入/输出形状: (d_model, seq_len)
    class TransformerEncoderLayer final : public Layer
    {
    private:
        MultiHeadAttention self_attn_;
        LayerNorm norm1_;
        FeedForward ff_;
        LayerNorm norm2_;

        // ── 反向传播缓存 ──
        Matrix residual1_cache_;   // 第一个残差连接前的原始输入
        Matrix residual2_cache_;   // 第二个残差连接前的输入 (= residual1)

    public:
        TransformerEncoderLayer(std::size_t d_model, std::size_t num_heads, std::size_t d_ff)
            : self_attn_(d_model, num_heads),
              norm1_(d_model),
              ff_(d_model, d_ff),
              norm2_(d_model)
        {}

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = self_attn_.parameters();
            auto n1 = norm1_.parameters();
            auto f  = ff_.parameters();
            auto n2 = norm2_.parameters();
            params.insert(params.end(), n1.begin(), n1.end());
            params.insert(params.end(), f.begin(), f.end());
            params.insert(params.end(), n2.begin(), n2.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = self_attn_.param_gradients();
            auto gn1 = norm1_.param_gradients();
            auto gf  = ff_.param_gradients();
            auto gn2 = norm2_.param_gradients();
            grads.insert(grads.end(), gn1.begin(), gn1.end());
            grads.insert(grads.end(), gf.begin(), gf.end());
            grads.insert(grads.end(), gn2.begin(), gn2.end());
            return grads;
        }

        Matrix forward(const Matrix &input) override
        {
            // 子层1: Self-Attention + 残差
            residual1_cache_ = input;
            auto norm1_out = norm1_.forward(input);
            auto attn_out  = self_attn_.forward(norm1_out);
            residual2_cache_ = input + attn_out;  // 残差连接

            // 子层2: FFN + 残差
            auto norm2_out = norm2_.forward(residual2_cache_);
            auto ff_out    = ff_.forward(norm2_out);
            return residual2_cache_ + ff_out;
        }

        Matrix backward(const Matrix &grad_output) override
        {
            // ── 反向传播第2个残差连接: grad_r2 = grad_output (分流到两条路径) ──
            //   路径A: 直接流向 residual2 (= residual1)
            //   路径B: 流向 FFN
            Matrix grad_residual1 = grad_output;   // 路径A
            Matrix grad_ff_out    = grad_output;   // 路径B

            // 反向 FFN → LayerNorm₂
            auto grad_norm2 = ff_.backward(grad_ff_out);
            grad_residual1 = grad_residual1 + norm2_.backward(grad_norm2);

            // ── 反向传播第1个残差连接 ──
            //   路径A: 直接流向输入 x
            //   路径B: 流向 Self-Attention
            Matrix grad_input = grad_residual1;    // 路径A
            Matrix grad_attn_out = grad_residual1; // 路径B

            // 反向 Self-Attention → LayerNorm₁
            auto grad_norm1 = self_attn_.backward(grad_attn_out);
            grad_input = grad_input + norm1_.backward(grad_norm1);

            return grad_input;
        }
    };

    // ── Transformer 编码器 ──────────────────────────────────────────────
    // 堆叠 N 个 TransformerEncoderLayer，内含位置编码 + 全局平均池化
    // 输入: (d_model × num_patches, batch_size)  — PatchEmbedding 的输出
    // 输出: (d_model, batch_size)                 — 池化后的序列表示
    //
    // 反向传播策略: 对每个样本 re-forward 以重建内部缓存 (checkpointing)
    // ────────────────────────────────────────────────────────────────────
    class TransformerEncoder final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t num_patches_;

        std::vector<TransformerEncoderLayer> layers_;
        PositionalEncoding pos_encoding_;

        // ── 反向传播缓存: 每个样本在位置编码后的输入（供 re-forward 重建） ──
        std::vector<Matrix> stored_inputs_;  // [sample] — 每个样本 PE 后的输入
        std::size_t batch_size_{0};

    public:
        TransformerEncoder(std::size_t d_model, std::size_t num_heads,
                           std::size_t d_ff, std::size_t num_layers,
                           std::size_t num_patches)
            : d_model_(d_model), num_patches_(num_patches),
              pos_encoding_(d_model, num_patches)
        {
            for (std::size_t i = 0; i < num_layers; ++i)
                layers_.emplace_back(d_model, num_heads, d_ff);
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            std::vector<std::reference_wrapper<Matrix>> params;
            for (auto &layer : layers_)
            {
                auto lp = layer.parameters();
                params.insert(params.end(), lp.begin(), lp.end());
            }
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            std::vector<std::reference_wrapper<Matrix>> grads;
            for (auto &layer : layers_)
            {
                auto lg = layer.param_gradients();
                grads.insert(grads.end(), lg.begin(), lg.end());
            }
            return grads;
        }

        // ── 前向传播 ────────────────────────────────────────────────────
        Matrix forward(const Matrix &input) override
        {
            batch_size_ = input.cols();
            stored_inputs_.resize(batch_size_);

            Matrix output(d_model_, batch_size_);

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                // 从展平向量中提取样本 b → (d_model, num_patches)
                Matrix x(d_model_, num_patches_);
                for (std::size_t r = 0; r < d_model_; ++r)
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        x.set_value_unchecked(r, c,
                            input.at_unchecked(r * num_patches_ + c, b));

                // 添加位置编码
                x = pos_encoding_.forward(x);

                // 缓存 PE 后的输入，供 backward 中 re-forward 重建缓存
                stored_inputs_[b] = x;

                // 依次通过编码器层
                for (std::size_t l = 0; l < layers_.size(); ++l)
                    x = layers_[l].forward(x);

                // 全局平均池化: (d_model, num_patches) → (d_model,)
                const double inv_n = 1.0 / static_cast<double>(num_patches_);
                for (std::size_t r = 0; r < d_model_; ++r)
                {
                    double sum = 0.0;
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        sum += x.at_unchecked(r, c);
                    output.set_value_unchecked(r, b, sum * inv_n);
                }
            }
            return output;
        }

        // ── 反向传播 ────────────────────────────────────────────────────
        // 对每个样本: re-forward 重建缓存 → 反向传播 → 累加参数梯度
        Matrix backward(const Matrix &grad_output) override
        {
            Matrix grad_input(d_model_ * num_patches_, batch_size_);
            grad_input.zero();

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                // ── Re-forward: 重建该样本的全部缓存 ──
                // stored_inputs_[b] 已含位置编码，无需重复施加
                Matrix x = stored_inputs_[b];
                for (std::size_t l = 0; l < layers_.size(); ++l)
                    x = layers_[l].forward(x);    // 重建每层缓存

                // ── 全局平均池化梯度: 展开 ──
                const double inv_n = 1.0 / static_cast<double>(num_patches_);
                Matrix grad(d_model_, num_patches_);
                for (std::size_t r = 0; r < d_model_; ++r)
                {
                    double g = grad_output.at_unchecked(r, b) * inv_n;
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        grad.set_value_unchecked(r, c, g);
                }

                // ── 反向传播编码器层 ──
                for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l)
                    grad = layers_[l].backward(grad);

                // ── 写入 grad_input ──
                for (std::size_t r = 0; r < d_model_; ++r)
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        grad_input.set_value_unchecked(r * num_patches_ + c, b,
                            grad.at_unchecked(r, c));
            }
            return grad_input;
        }
    };

    // ── Patch 嵌入层 ──────────────────────────────────────────────────
    // 将展平的图像分割为不重叠的 patch 并投影到 d_model 维空间
    // 输入: (img_size², batch_size)          — 展平的 28×28 图像
    // 输出: (d_model × num_patches, batch_size) — 展平的 patch 序列
    // ────────────────────────────────────────────────────────────────────
    class PatchEmbedding final : public Layer
    {
    private:
        std::size_t img_size_;
        std::size_t patch_size_;
        std::size_t grid_size_;      // img_size / patch_size
        std::size_t num_patches_;    // grid_size²
        std::size_t patch_dim_;      // patch_size²
        std::size_t d_model_;
        Linear projection_;          // (patch_dim, d_model)
        Matrix input_cache_;

    public:
        PatchEmbedding(std::size_t img_size, std::size_t patch_size, std::size_t d_model)
            : img_size_(img_size), patch_size_(patch_size),
              grid_size_(img_size / patch_size),
              num_patches_((img_size / patch_size) * (img_size / patch_size)),
              patch_dim_(patch_size * patch_size),
              d_model_(d_model),
              projection_(patch_dim_, d_model)
        {
            if (img_size % patch_size != 0)
                throw std::invalid_argument(
                    "PatchEmbedding: img_size must be divisible by patch_size");
        }

        [[nodiscard]] std::size_t num_patches() const noexcept { return num_patches_; }
        [[nodiscard]] std::size_t d_model()     const noexcept { return d_model_; }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return projection_.parameters();
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return projection_.param_gradients();
        }

        // ── 前向传播 ────────────────────────────────────────────────────
        // 批量提取 patch 并一次性投影
        Matrix forward(const Matrix &input) override
        {
            input_cache_ = input;
            const std::size_t batch = input.cols();

            // Step 1: 提取所有 patch → (patch_dim, num_patches × batch)
            // 布局: patch p 的 sample b 位于列 p × batch + b
            Matrix all_patches(patch_dim_, num_patches_ * batch);

            for (std::size_t b = 0; b < batch; ++b)
            {
                for (std::size_t p = 0; p < num_patches_; ++p)
                {
                    const std::size_t gr = (p / grid_size_) * patch_size_;
                    const std::size_t gc = (p % grid_size_) * patch_size_;
                    const std::size_t col_idx = p * batch + b;

                    for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    {
                        for (std::size_t pc = 0; pc < patch_size_; ++pc)
                        {
                            const std::size_t flat = pr * patch_size_ + pc;
                            const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                            all_patches.set_value_unchecked(flat, col_idx,
                                input.at_unchecked(pix, b));
                        }
                    }
                }
            }

            // Step 2: 投影 → (d_model, num_patches × batch)
            Matrix projected = projection_.forward(all_patches);

            // Step 3: 重排为 (d_model × num_patches, batch)
            Matrix output(d_model_ * num_patches_, batch);
            for (std::size_t r = 0; r < d_model_; ++r)
                for (std::size_t p = 0; p < num_patches_; ++p)
                    for (std::size_t b = 0; b < batch; ++b)
                        output.set_value_unchecked(r * num_patches_ + p, b,
                            projected.at_unchecked(r, p * batch + b));

            return output;
        }

        // ── 反向传播 ────────────────────────────────────────────────────
        Matrix backward(const Matrix &grad_output) override
        {
            const std::size_t batch = grad_output.cols();

            // Step 1: 重排梯度 → (d_model, num_patches × batch)
            Matrix grad_projected(d_model_, num_patches_ * batch);
            for (std::size_t r = 0; r < d_model_; ++r)
                for (std::size_t p = 0; p < num_patches_; ++p)
                    for (std::size_t b = 0; b < batch; ++b)
                        grad_projected.set_value_unchecked(r, p * batch + b,
                            grad_output.at_unchecked(r * num_patches_ + p, b));

            // Step 2: 投影层反向 → (patch_dim, num_patches × batch)
            Matrix grad_patches = projection_.backward(grad_projected);

            // Step 3: 散射梯度回输入 → (img_size², batch)
            Matrix grad_input(img_size_ * img_size_, batch);
            grad_input.zero();

            for (std::size_t b = 0; b < batch; ++b)
            {
                for (std::size_t p = 0; p < num_patches_; ++p)
                {
                    const std::size_t gr = (p / grid_size_) * patch_size_;
                    const std::size_t gc = (p % grid_size_) * patch_size_;
                    const std::size_t col_idx = p * batch + b;

                    for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    {
                        for (std::size_t pc = 0; pc < patch_size_; ++pc)
                        {
                            const std::size_t flat = pr * patch_size_ + pc;
                            const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                            const double val = grad_input.at_unchecked(pix, b)
                                             + grad_patches.at_unchecked(flat, col_idx);
                            grad_input.set_value_unchecked(pix, b, val);
                        }
                    }
                }
            }
            return grad_input;
        }
    };
}

#endif