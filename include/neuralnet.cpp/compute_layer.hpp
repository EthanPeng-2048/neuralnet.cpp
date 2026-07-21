#ifndef NN_COMPUTE_LAYER_HPP
#define NN_COMPUTE_LAYER_HPP

#include <functional>
#include <limits>
#include <random>

#include "algebra_matrix.hpp"
#include "config.hpp"
#include "algebra_span.hpp"
#include "algebra_compute.hpp"

namespace nn
{
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Result<Matrix> forward(const Matrix &input) = 0;
        virtual Result<Matrix> backward(const Matrix &grad_output) = 0;
        virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
        virtual std::vector<std::reference_wrapper<Matrix>> param_gradients() { return {}; }

        // ── 序列生成（仅 GPTModel 等自回归模型实现） ─────────────────────
        // 默认实现返回错误，避免 L5 层使用 dynamic_cast 向下转型。
        // L5 层应通过 Model::generate() 调用此方法。
        [[nodiscard]] virtual Result<std::vector<std::size_t>>
        generate(const std::vector<std::size_t> & /*prompt*/,
                 std::size_t /*max_new_tokens*/,
                 Scalar /*temperature*/)
        {
            return std::unexpected(Error{"Layer::generate not supported by this layer type"});
        }
    };

    class Linear final : public Layer
    {
    private:
        Matrix w_;
        Matrix b_;
        Matrix grad_w_;
        Matrix grad_b_;
        Matrix input_cache_;

        // ── 预分配缓冲区：避免 forward/backward 热路径反复分配内存 ──────
        Matrix product_buf_;   // w * input 的中间结果
        Matrix grad_wt_buf_;   // backward: w^T

        // 使用 thread_local 保证多线程构造 Layer 时的线程安全
        inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    public:
        Linear(std::size_t in_features, std::size_t out_features)
            : w_(out_features, in_features),
              b_(out_features, 1),
              grad_w_(out_features, in_features),
              grad_b_(out_features, 1),
              input_cache_(),
              product_buf_(out_features, 1),
              grad_wt_buf_(in_features, out_features)
        {
            // Xavier 均匀初始化：适合 tanh/sigmoid，对 ReLU 也可用
            const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features + out_features));
            std::uniform_real_distribution<Scalar> dist(-limit, limit);
            auto w_span = w_.span();
            for (std::size_t i = 0; i < w_.size(); ++i)
                w_span[i] = dist(rng_);
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            return {std::ref(w_), std::ref(b_)};
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            return {std::ref(grad_w_), std::ref(grad_b_)};
        }

        Result<Matrix> forward(const Matrix &input) override
        {
            if (input.rows() != w_.cols())
                return std::unexpected(Error{"linear forward input shape mismatch"});

            input_cache_ = input;

            // product = w * input（写入预分配缓冲区，避免分配）
            w_.multiply_to(product_buf_, input);

            // result = product + bias
            Matrix result = product_buf_;
            result.add_bias_broadcast_inplace(b_);

            return result;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != w_.rows())
                return std::unexpected(Error{"linear backward grad_output shape mismatch"});
            if (input_cache_.rows() != w_.cols() || input_cache_.cols() != grad_output.cols())
                return std::unexpected(Error{"linear backward cache/input shape mismatch"});

            const std::size_t in_feat = w_.cols();
            const std::size_t out_feat = w_.rows();
            const std::size_t batch = grad_output.cols();

            // grad_input = w^T * grad_output
            w_.transpose_to(grad_wt_buf_);
            Matrix grad_input(in_feat, batch);
            grad_wt_buf_.multiply_to(grad_input, grad_output);

            // grad_w += grad_output * input_cache_^T（通过 Matrix 语义方法）
            {
                Matrix input_T = input_cache_.transpose();
                Matrix grad_w_accum(out_feat, in_feat);
                grad_output.multiply_to(grad_w_accum, input_T);
                grad_w_.add_inplace(grad_w_accum);
            }

            // grad_b += sum(grad_output, dim=batch) —— 按行归约后再累加
            // grad_b_accum[of][0] = Σ_b grad_output[of][b]
            Matrix grad_b_accum = grad_output.row_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });
            grad_b_.add_inplace(grad_b_accum);

            return grad_input;
        }
    };

    class ReLU final : public Layer
    {
    private:
        Matrix input_cache_;

    public:
        ReLU() = default;

        // forward: out = max(x, 0)
        // 通过 AST 入口 compute::apply 表达 ReLU 算法，底层自动并行
        Result<Matrix> forward(const Matrix &input) override
        {
            input_cache_ = input;
            Matrix result(input.rows(), input.cols());
            ConstSpan in = input.span();
            Span out = result.span();
            compute::apply(out, max(in, Scalar{0}));
            return result;
        }

        // backward: out = (x > 0) ? grad_output : 0
        // 通过 AST select() 表达条件选择
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
                return std::unexpected(Error{"relu backward shape mismatch"});

            Matrix result(grad_output.rows(), grad_output.cols());
            Span x = input_cache_.span();
            ConstSpan go = grad_output.span();
            Span out = result.span();
            compute::apply(out, select(x > Scalar{0}, go, Scalar{0}));
            return result;
        }
    };

    class GeLU final : public Layer
    {
    private:
        Matrix input_cache_;
        Matrix sigmoid_cache_;
        static constexpr Scalar BETA = 1.702;

    public:
        GeLU() = default;

        // QuickGeLU: out = x * sigmoid(β * x)
        // 通过 AST 表达 QuickGeLU 算法：sigmoid(βx) 与 x*sigmoid(βx) 均由 AST 求值
        Result<Matrix> forward(const Matrix &input) override
        {
            input_cache_ = input;
            // sigmoid_cache_[i] = 1 / (1 + exp(-β * x[i]))
            sigmoid_cache_.resize(input.rows(), input.cols());
            ConstSpan x_in = input.span();
            Span s_out = sigmoid_cache_.span();
            compute::apply(s_out, sigmoid(BETA * x_in));

            // out[i] = x[i] * sigmoid_cache_[i]
            Matrix result(input.rows(), input.cols());
            Span x = input_cache_.span();
            Span s = sigmoid_cache_.span();
            Span out = result.span();
            compute::apply(out, x * s);
            return result;
        }

        // d/dx [x * sigmoid(βx)] = sigmoid(βx) * [1 + βx * (1 - sigmoid(βx))]
        // factor = s * (1 + βx * (1 - s));  out = go * factor
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            if (input_cache_.rows() != grad_output.rows() || input_cache_.cols() != grad_output.cols())
                return std::unexpected(Error{"gelu backward shape mismatch"});

            // factor[i] = s[i] * (1 + BETA * x[i] * (1 - s[i]))
            Matrix factor(input_cache_.rows(), input_cache_.cols());
            Span x = input_cache_.span();
            Span s = sigmoid_cache_.span();
            Span f_out = factor.span();
            compute::apply(f_out, s * (Scalar{1} + BETA * x * (Scalar{1} - s)));

            // out[i] = go[i] * factor[i]
            Matrix result(grad_output.rows(), grad_output.cols());
            ConstSpan go = grad_output.span();
            Span f_in = factor.span();
            Span r_out = result.span();
            compute::apply(r_out, go * f_in);
            return result;
        }
    };

    class LayerNorm final : public Layer
    {
    private:
        std::size_t normalized_shape_;
        Scalar epsilon_;

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
        static constexpr Scalar EPSILON = 1e-5;

    public:
        explicit LayerNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
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

        // ── 前向传播：对每个 batch 的所有 features 归一化 + 仿射变换 ──
        // input: (features, batch_size)
        // mean/std: (1, batch_size)  ← 按列归约（每列对应一个 batch）
        // normalized/output: (features, batch_size)
        Result<Matrix> forward(const Matrix &input) override
        {
            if (input.rows() != normalized_shape_)
                return std::unexpected(Error{"layer_norm forward input shape mismatch"});

            input_cache_ = input;
            const std::size_t features = input.rows();
            const std::size_t batch_size = input.cols();
            const Scalar inv_features = Scalar{1} / static_cast<Scalar>(features);

            mean_cache_.resize(1, batch_size);
            std_cache_.resize(1, batch_size);
            normalized_cache_.resize(features, batch_size);

            // mean[b] = (1/features) * Σ_f input[f][b]  （按列归约）
            mean_cache_ = input.col_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });
            mean_cache_.scale_inplace(inv_features);

            // diff = input - mean（按列广播减法，复用 Matrix 通用原语）
            Matrix diff = input;
            diff.broadcast_col_inplace(mean_cache_,
                [](Scalar x, Scalar m) noexcept { return x - m; });

            // diff_sq = diff²  (AST 逐元素平方)
            Matrix diff_sq(features, batch_size);
            Span d_in = diff.span();
            Span ds_out = diff_sq.span();
            compute::apply(ds_out, d_in * d_in);

            // var[b] = (1/features) * Σ_f diff_sq[f][b]  （按列归约）
            Matrix var = diff_sq.col_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });
            var.scale_inplace(inv_features);

            // std_inv[b] = 1 / sqrt(var[b] + ε)  (AST: rsqrt(var + ε))
            const Scalar eps = epsilon_;
            Span v_in = var.span();
            Span s_out = std_cache_.span();
            compute::apply(s_out, rsqrt(v_in + eps));

            // normalized = diff * std_inv（按列广播乘法）
            normalized_cache_ = diff;
            normalized_cache_.broadcast_col_inplace(std_cache_,
                [](Scalar d, Scalar s) noexcept { return d * s; });

            // output = gamma * normalized + beta（按行广播：gamma/beta 形状 (features, 1)）
            Matrix output = normalized_cache_;
            output.broadcast_row_inplace(gamma_,
                [](Scalar n, Scalar g) noexcept { return n * g; });
            output.broadcast_row_inplace(beta_,
                [](Scalar o, Scalar b) noexcept { return o + b; });
            return output;
        }

        // ── 反向传播 ──
        // dL/dγ[f] = Σ_b dL/dy[f][b] * normalized[f][b]
        // dL/dβ[f] = Σ_b dL/dy[f][b]
        // dL/dx[f][b] = (dL/dy[f][b] * γ[f] - mean_g - normalized[f][b] * mean_gn) * std_inv[b]
        //   其中 mean_g = (1/features) * Σ_f dL/dy[f][b] * γ[f]
        //        mean_gn = (1/features) * Σ_f dL/dy[f][b] * γ[f] * normalized[f][b]
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            if (grad_output.rows() != normalized_shape_)
                return std::unexpected(Error{"layer_norm backward shape mismatch"});

            const std::size_t features = normalized_shape_;
            const std::size_t batch_size = grad_output.cols();
            const Scalar inv_features = Scalar{1} / static_cast<Scalar>(features);

            // gy_gamma[f][b] = grad_output[f][b] * gamma[f]（按行广播乘法）
            Matrix gy_gamma = grad_output;
            gy_gamma.broadcast_row_inplace(gamma_,
                [](Scalar gy, Scalar g) noexcept { return gy * g; });

            // gy_gamma_norm = gy_gamma ⊙ normalized  (AST 逐元素乘)
            Matrix gy_gamma_norm(features, batch_size);
            Span gy = gy_gamma.span();
            Span n = normalized_cache_.span();
            Span gn_out = gy_gamma_norm.span();
            compute::apply(gn_out, gy * n);

            // grad_gamma[f] = Σ_b gy_gamma_norm[f][b]  （按行归约）
            Matrix grad_gamma_row = gy_gamma_norm.row_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });

            // grad_beta[f] = Σ_b grad_output[f][b]  （按行归约）
            Matrix grad_beta_row = grad_output.row_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });

            grad_gamma_.add_inplace(grad_gamma_row);
            grad_beta_.add_inplace(grad_beta_row);

            // mean_g[b] = (1/features) * Σ_f gy_gamma[f][b]  （按列归约）
            Matrix mean_g = gy_gamma.col_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });
            mean_g.scale_inplace(inv_features);

            // mean_gn[b] = (1/features) * Σ_f gy_gamma_norm[f][b]  （按列归约）
            Matrix mean_gn = gy_gamma_norm.col_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });
            mean_gn.scale_inplace(inv_features);

            // grad_input[f][b] = (gy_gamma[f][b] - mean_g[b] - normalized[f][b] * mean_gn[b]) * std_inv[b]
            // 步骤1: t1 = gy_gamma - mean_g（按列广播减法）
            Matrix t1 = gy_gamma;
            t1.broadcast_col_inplace(mean_g,
                [](Scalar gy, Scalar m) noexcept { return gy - m; });

            // 步骤2: t2 = normalized * mean_gn（按列广播乘法）
            Matrix t2 = normalized_cache_;
            t2.broadcast_col_inplace(mean_gn,
                [](Scalar n, Scalar m) noexcept { return n * m; });

            // 步骤3: diff = t1 - t2  (AST 逐元素减)
            Matrix diff(features, batch_size);
            Span t1_s = t1.span();
            Span t2_s = t2.span();
            Span d_out = diff.span();
            compute::apply(d_out, t1_s - t2_s);

            // 步骤4: grad_input = diff * std_inv（按列广播乘法）
            Matrix grad_input = diff;
            grad_input.broadcast_col_inplace(std_cache_,
                [](Scalar d, Scalar s) noexcept { return d * s; });
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

        // forward: 对每一行独立做 softmax
        // 算法步骤：
        //   1. row_max[r] = max_c input[r][c]               （按行归约）
        //   2. shifted[r][c] = input[r][c] - row_max[r]      （按行广播减法）
        //   3. exp_shifted[r][c] = exp(shifted[r][c])        （AST: exp(span)）
        //   4. row_sum[r] = sum_c exp_shifted[r][c]          （按行归约）
        //   5. output[r][c] = exp_shifted[r][c] / row_sum[r] （按行广播除法）
        Result<Matrix> forward(const Matrix &input) override
        {
            // 1. 求每行最大值
            Matrix row_max = input.row_reduce(
                std::numeric_limits<Scalar>::lowest(),
                [](Scalar a, Scalar b) noexcept { return std::max(a, b); },
                [](Scalar x) noexcept { return x; });

            // 2. shifted = input - row_max（按行广播）
            Matrix shifted = input;
            shifted.broadcast_row_inplace(row_max,
                [](Scalar x, Scalar m) noexcept { return x - m; });

            // 3. exp_shifted = exp(shifted)  (AST 逐元素 exp)
            Matrix exp_shifted(shifted.rows(), shifted.cols());
            Span sh = shifted.span();
            Span es_out = exp_shifted.span();
            compute::apply(es_out, exp(sh));

            // 4. 求每行 exp 之和
            Matrix row_sum = exp_shifted.row_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });

            // 5. 归一化: output = exp_shifted / row_sum（按行广播）
            Matrix output = exp_shifted;
            output.broadcast_row_inplace(row_sum,
                [](Scalar e, Scalar s) noexcept { return e / s; });

            output_cache_ = output;  // 缓存 softmax 输出供 backward 使用
            return output;
        }

        // backward: grad_input[i][j] = out[i][j] * (grad_out[i][j] - dot(out[i], grad_out[i]))
        //   dot[r] = Σ_c out[r][c] * grad_out[r][c]  （按行归约）
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            const std::size_t rows = output_cache_.rows();
            const std::size_t cols = output_cache_.cols();

            if (grad_output.rows() != rows || grad_output.cols() != cols)
                return std::unexpected(Error{"softmax backward shape mismatch"});

            // elementwise_product = out ⊙ grad_output  (AST 逐元素乘)
            Matrix elementwise_product(rows, cols);
            Span o = output_cache_.span();
            ConstSpan g = grad_output.span();
            Span ep_out = elementwise_product.span();
            compute::apply(ep_out, o * g);

            // dot[r] = Σ_c elementwise_product[r][c]
            Matrix dot = elementwise_product.row_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });

            // grad_input[r][c] = out[r][c] * (grad_output[r][c] - dot[r])
            // 步骤：先按行广播减法（grad_output - dot），再乘以 out
            Matrix go_minus_dot = grad_output;
            go_minus_dot.broadcast_row_inplace(dot,
                [](Scalar g, Scalar d) noexcept { return g - d; });

            // grad_input = out ⊙ go_minus_dot  (AST 逐元素乘)
            Matrix grad_input(rows, cols);
            Span o2 = output_cache_.span();
            Span gmd = go_minus_dot.span();
            Span gi_out = grad_input.span();
            compute::apply(gi_out, o2 * gmd);
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
        Scalar scale_;  // 1.0 / sqrt(d_k)

        // 投影层
        Linear w_q_;
        Linear w_k_;
        Linear w_v_;
        Linear w_o_;

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
        Softmax softmax_;         // 仅在 forward() 中调用 forward()，不调用 backward()（注意力反向手动计算）

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

        // ── 逐元素缩放（委托 Matrix 语义方法） ──────────────────────────
        static void scale_inplace(Matrix &m, Scalar s)
        {
            m.scale_inplace(s);
        }

    public:
        MultiHeadAttention(std::size_t d_model, std::size_t num_heads)
            : d_model_(d_model),
              num_heads_(num_heads),
              d_k_(d_model / num_heads),
              scale_(1.0 / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
              w_q_(d_model, d_model),
              w_k_(d_model, d_model),
              w_v_(d_model, d_model),
              w_o_(d_model, d_model)
        {
            if (d_model % num_heads != 0)
                NN_ASSERT(false, "MultiHeadAttention: d_model must be divisible by num_heads");
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = w_q_.parameters();
            auto wk = w_k_.parameters();
            auto wv = w_v_.parameters();
            auto wo = w_o_.parameters();
            params.insert(params.end(), wk.begin(), wk.end());
            params.insert(params.end(), wv.begin(), wv.end());
            params.insert(params.end(), wo.begin(), wo.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = w_q_.param_gradients();
            auto gk = w_k_.param_gradients();
            auto gv = w_v_.param_gradients();
            auto go = w_o_.param_gradients();
            grads.insert(grads.end(), gk.begin(), gk.end());
            grads.insert(grads.end(), gv.begin(), gv.end());
            grads.insert(grads.end(), go.begin(), go.end());
            return grads;
        }

        Result<Matrix> forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                return std::unexpected(Error{"MultiHeadAttention forward input shape mismatch"});

            const std::size_t seq_len = input.cols();

            // ── 1. 线性投影 Q, K, V ──────────────────────────────────────
            auto q_res = w_q_.forward(input);
            if (!q_res) return q_res;
            Q_cache_ = *q_res;
            auto k_res = w_k_.forward(input);
            if (!k_res) return k_res;
            K_cache_ = *k_res;
            auto v_res = w_v_.forward(input);
            if (!v_res) return v_res;
            V_cache_ = *v_res;

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
                auto sm_res = softmax_.forward(attn_[h]);
                if (!sm_res) return sm_res;
                attn_[h] = *sm_res;

                // O_h = V_h @ A_h → (d_k_, seq_len)，缓存到 O_heads_
                O_heads_[h].resize(d_k_, seq_len);
                V_heads_[h].multiply_to(O_heads_[h], attn_[h]);
            }

            // ── 4. 拼接所有头的输出并投影 ────────────────────────────────
            Matrix output(d_model_, seq_len);
            for (std::size_t h = 0; h < num_heads_; ++h)
                insert_rows(output, h * d_k_, O_heads_[h]);

            auto wo_res = w_o_.forward(output);
            if (!wo_res) return wo_res;
            return *wo_res;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            const std::size_t seq_len = grad_output.cols();

            // ── 1. 输出投影反向 ──────────────────────────────────────────
            auto gc_res = w_o_.backward(grad_output);
            if (!gc_res) return gc_res;
            Matrix grad_concat = *gc_res;  // (d_model, seq_len)

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
                // 通过 AST + Matrix 原语表达：
                //   1. ep = A ⊙ grad_A                (AST 逐元素乘)
                //   2. dot[r] = Σ_c ep[r][c]          (按行归约)
                //   3. gmd = grad_A - dot             (按行广播减)
                //   4. grad_S = A ⊙ gmd              (AST 逐元素乘)
                grad_scores_buf_.resize(seq_len, seq_len);
                Matrix ep(seq_len, seq_len);
                ConstSpan a_s = A_h.span();
                Span ga_s = grad_A_buf_.span();
                Span ep_s = ep.span();
                compute::apply(ep_s, a_s * ga_s);

                Matrix dot = ep.row_reduce(Scalar{0},
                    [](Scalar x, Scalar y) noexcept { return x + y; },
                    [](Scalar x) noexcept { return x; });

                Matrix gmd = grad_A_buf_;
                gmd.broadcast_row_inplace(dot,
                    [](Scalar ga, Scalar d) noexcept { return ga - d; });

                Span gmd_s = gmd.span();
                Span gs_s = grad_scores_buf_.span();
                compute::apply(gs_s, a_s * gmd_s);

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
            auto giq_res = w_q_.backward(grad_Q_all);
            if (!giq_res) return giq_res;
            Matrix grad_input = *giq_res;
            auto gik_res = w_k_.backward(grad_K_all);
            if (!gik_res) return gik_res;
            grad_input = grad_input + *gik_res;
            auto giv_res = w_v_.backward(grad_V_all);
            if (!giv_res) return giv_res;
            grad_input = grad_input + *giv_res;

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
            std::vector<Scalar> freqs(half);
            for (std::size_t i = 0; i < half; ++i)
                freqs[i] = 1.0 / std::pow(10000.0, static_cast<Scalar>(2 * i) / d_model);

            auto e_span = encoding_.span();
            for (std::size_t pos = 0; pos < max_len; ++pos)
            {
                const Scalar pos_d = static_cast<Scalar>(pos);
                for (std::size_t i = 0; i < half; ++i)
                {
                    const Scalar angle = pos_d * freqs[i];
                    e_span[(2 * i)       * max_len + pos] = std::sin(angle);
                    e_span[(2 * i + 1)   * max_len + pos] = std::cos(angle);
                }
                // 奇数维度：最后一个特征仅使用 sin
                if (d_model % 2 == 1)
                {
                    const Scalar freq_last = 1.0 / std::pow(10000.0, static_cast<Scalar>(2 * half) / d_model);
                    const Scalar angle = pos_d * freq_last;
                    e_span[(d_model - 1) * max_len + pos] = std::sin(angle);
                }
            }
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override { return {}; }
        std::vector<std::reference_wrapper<Matrix>> param_gradients() override { return {}; }

        Result<Matrix> forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                return std::unexpected(Error{"positional encoding forward: d_model mismatch"});

            const std::size_t seq_len = input.cols();

            if (seq_len > max_len_)
                return std::unexpected(Error{"positional encoding forward: sequence length exceeds max_len"});

            // 提取编码切片: encoding_ 形状为 (d_model_, max_len_)，
            // 仅取前 seq_len 列
            Matrix encoding_slice(d_model_, seq_len);
            for (std::size_t r = 0; r < d_model_; ++r)
                for (std::size_t c = 0; c < seq_len; ++c)
                    encoding_slice.set_value_unchecked(r, c,
                        encoding_.at_unchecked(r, c));

            // result = input + encoding（通过 Matrix operator+，内部自动并行）
            return input + encoding_slice;
        }

        // 位置编码为固定值，梯度直接穿透
        Result<Matrix> backward(const Matrix &grad_output) override { return grad_output; }
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

        Result<Matrix> forward(const Matrix &input) override
        {
            input_cache_ = input;
            auto fc1_res = fc1_.forward(input);
            if (!fc1_res) return fc1_res;
            auto act_res = activation_.forward(*fc1_res);
            if (!act_res) return act_res;
            auto fc2_res = fc2_.forward(*act_res);
            if (!fc2_res) return fc2_res;
            return *fc2_res;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            auto b_fc2 = fc2_.backward(grad_output);
            if (!b_fc2) return b_fc2;
            auto b_act = activation_.backward(*b_fc2);
            if (!b_act) return b_act;
            auto b_fc1 = fc1_.backward(*b_act);
            if (!b_fc1) return b_fc1;
            return *b_fc1;
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

        Result<Matrix> forward(const Matrix &input) override
        {
            // 子层1: Self-Attention + 残差
            residual1_cache_ = input;
            auto n1_res = norm1_.forward(input);
            if (!n1_res) return n1_res;
            auto sa_res = self_attn_.forward(*n1_res);
            if (!sa_res) return sa_res;
            residual2_cache_ = input + *sa_res;  // 残差连接

            // 子层2: FFN + 残差
            auto n2_res = norm2_.forward(residual2_cache_);
            if (!n2_res) return n2_res;
            auto ff_res = ff_.forward(*n2_res);
            if (!ff_res) return ff_res;
            return residual2_cache_ + *ff_res;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            // ── 反向传播第2个残差连接: grad_r2 = grad_output (分流到两条路径) ──
            //   路径A: 直接流向 residual2 (= residual1)
            //   路径B: 流向 FFN
            Matrix grad_residual1 = grad_output;   // 路径A
            Matrix grad_ff_out    = grad_output;   // 路径B

            // 反向 FFN → LayerNorm₂
            auto b_ff = ff_.backward(grad_ff_out);
            if (!b_ff) return b_ff;
            auto b_n2 = norm2_.backward(*b_ff);
            if (!b_n2) return b_n2;
            grad_residual1 = grad_residual1 + *b_n2;

            // ── 反向传播第1个残差连接 ──
            //   路径A: 直接流向输入 x
            //   路径B: 流向 Self-Attention
            Matrix grad_input = grad_residual1;    // 路径A
            Matrix grad_attn_out = grad_residual1; // 路径B

            // 反向 Self-Attention → LayerNorm₁
            auto b_sa = self_attn_.backward(grad_attn_out);
            if (!b_sa) return b_sa;
            auto b_n1 = norm1_.backward(*b_sa);
            if (!b_n1) return b_n1;
            grad_input = grad_input + *b_n1;

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
        Result<Matrix> forward(const Matrix &input) override
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
                auto pe_res = pos_encoding_.forward(x);
                if (!pe_res) return pe_res;
                x = *pe_res;

                // 缓存 PE 后的输入，供 backward 中 re-forward 重建缓存
                stored_inputs_[b] = x;

                // 依次通过编码器层
                for (std::size_t l = 0; l < layers_.size(); ++l)
                {
                    auto ly_res = layers_[l].forward(x);
                    if (!ly_res) return ly_res;
                    x = *ly_res;
                }

                // 全局平均池化: (d_model, num_patches) → (d_model,)
                const Scalar inv_n = 1.0 / static_cast<Scalar>(num_patches_);
                for (std::size_t r = 0; r < d_model_; ++r)
                {
                    Scalar sum = 0.0;
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        sum += x.at_unchecked(r, c);
                    output.set_value_unchecked(r, b, sum * inv_n);
                }
            }
            return output;
        }

        // ── 反向传播 ──
        // 对每个样本: re-forward 重建缓存 → 反向传播 → 累加参数梯度
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            Matrix grad_input(d_model_ * num_patches_, batch_size_);
            grad_input.zero();

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                // ── Re-forward: 重建该样本的全部缓存 ──
                // stored_inputs_[b] 已含位置编码，无需重复施加
                Matrix x = stored_inputs_[b];
                for (std::size_t l = 0; l < layers_.size(); ++l)
                {
                    auto ly_res = layers_[l].forward(x);    // 重建每层缓存
                    if (!ly_res) return ly_res;
                    x = *ly_res;
                }

                // ── 全局平均池化梯度: 展开 ──
                const Scalar inv_n = 1.0 / static_cast<Scalar>(num_patches_);
                Matrix grad(d_model_, num_patches_);
                for (std::size_t r = 0; r < d_model_; ++r)
                {
                    Scalar g = grad_output.at_unchecked(r, b) * inv_n;
                    for (std::size_t c = 0; c < num_patches_; ++c)
                        grad.set_value_unchecked(r, c, g);
                }

                // ── 反向传播编码器层 ──
                for (int l = static_cast<int>(layers_.size()) - 1; l >= 0; --l)
                {
                    auto bl_res = layers_[l].backward(grad);
                    if (!bl_res) return bl_res;
                    grad = *bl_res;
                }

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
                NN_ASSERT(false, "PatchEmbedding: img_size must be divisible by patch_size");
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
        Result<Matrix> forward(const Matrix &input) override
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
            auto proj_res = projection_.forward(all_patches);
            if (!proj_res) return proj_res;
            Matrix projected = *proj_res;

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
        Result<Matrix> backward(const Matrix &grad_output) override
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
            auto bp_res = projection_.backward(grad_projected);
            if (!bp_res) return bp_res;
            Matrix grad_patches = *bp_res;

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
                            const Scalar val = grad_input.at_unchecked(pix, b)
                                             + grad_patches.at_unchecked(flat, col_idx);
                            grad_input.set_value_unchecked(pix, b, val);
                        }
                    }
                }
            }
            return grad_input;
        }
    };

    // ── 因果自注意力层 (Causal Self-Attention) ──────────────────────────
    // 用于 GPT 风格的自回归语言模型。
    // 与 MultiHeadAttention 的区别：在 softmax 前施加上三角因果掩码，
    // 保证位置 i 只能关注位置 ≤ i 的 token。
    //
    // 输入形状: (d_model, seq_len)，输出形状: (d_model, seq_len)
    // ─────────────────────────────────────────────────────────────────────
    class CausalSelfAttention final : public Layer
    {
    private:
        std::size_t d_model_;
        std::size_t num_heads_;
        std::size_t d_k_;
        Scalar scale_;

        // 投影层
        Linear w_q_;
        Linear w_k_;
        Linear w_v_;
        Linear w_o_;

        // 前向传播缓存
        Matrix Q_cache_;
        Matrix K_cache_;
        Matrix V_cache_;
        std::vector<Matrix> attn_;           // num_heads_ × (seq_len, seq_len)
        std::vector<Matrix> Q_heads_;
        std::vector<Matrix> K_heads_;
        std::vector<Matrix> V_heads_;
        std::vector<Matrix> O_heads_;

        // 因果掩码 (上三角为 -inf)
        // mask_data_ 以 max_len_ 为行间距存储 (i * max_len_ + j)。
        // 直接用 Span(mask_data_.data(), seq_len * seq_len) 读取会越界错位，
        // 故提供 mask_buf_ 作为按当前 seq_len 重建的连续缓冲区。
        std::vector<Scalar> mask_data_;     // max_len_ × max_len_
        std::size_t   max_len_ = 0;
        Matrix        mask_buf_;            // 连续的 seq_len × seq_len 掩码
        std::size_t   mask_buf_seq_len_ = 0; // mask_buf_ 当前对应的 seq_len

        // 辅助缓冲区
        Matrix grad_scores_buf_;
        Matrix grad_A_buf_;
        Matrix grad_O_h_buf_;
        Softmax softmax_;         // 仅在 forward() 中调用 forward()，不调用 backward()（注意力反向手动计算）

        // ── 按需构建连续 seq_len × seq_len 掩码缓冲区 ────────────────────
        // 修复：原代码以 max_len_ 行间距存储 mask，但读取时按连续 seq_len×seq_len
        // 解释，导致 seq_len < max_len_ 时掩码完全错位（因果性失效）。
        // 此处当 seq_len 变化时重建连续缓冲区，后续即可安全使用 AST apply。
        Span ensure_mask_buf(std::size_t seq_len)
        {
            if (mask_buf_seq_len_ != seq_len)
            {
                mask_buf_.resize(seq_len, seq_len);
                auto dst = mask_buf_.span();
                const std::size_t ml = max_len_;
                const Scalar *src = mask_data_.data();
                for (std::size_t i = 0; i < seq_len; ++i)
                {
                    const Scalar *src_row = src + i * ml;
                    Scalar *dst_row = dst.data() + i * seq_len;
                    for (std::size_t j = 0; j < seq_len; ++j)
                        dst_row[j] = src_row[j];
                }
                mask_buf_seq_len_ = seq_len;
            }
            return Span{mask_buf_.span()};
        }

        static void extract_rows(const Matrix &src, std::size_t row_start,
                                 std::size_t row_count, Matrix &dst)
        {
            const std::size_t cols = src.cols();
            dst.resize(row_count, cols);
            const auto src_span = src.span();
            auto dst_span = dst.span();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_span.begin() + (row_start + r) * cols, cols,
                            dst_span.begin() + r * cols);
        }

        static void insert_rows(Matrix &dst, std::size_t row_start,
                                const Matrix &src)
        {
            const std::size_t row_count = src.rows();
            const std::size_t cols = src.cols();
            auto dst_span = dst.span();
            const auto src_span = src.span();
            for (std::size_t r = 0; r < row_count; ++r)
                std::copy_n(src_span.begin() + r * cols, cols,
                            dst_span.begin() + (row_start + r) * cols);
        }

        // ── 逐元素缩放（委托 Matrix 语义方法） ──────────────────────────
        static void scale_inplace(Matrix &m, Scalar s)
        {
            m.scale_inplace(s);
        }

    public:
        CausalSelfAttention(std::size_t d_model, std::size_t num_heads,
                            std::size_t max_len = 1024)
            : d_model_(d_model),
              num_heads_(num_heads),
              d_k_(d_model / num_heads),
              scale_(1.0 / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
              w_q_(d_model, d_model),
              w_k_(d_model, d_model),
              w_v_(d_model, d_model),
              w_o_(d_model, d_model),
              mask_data_(max_len * max_len, 0.0),
              max_len_(max_len)
        {
            if (d_model % num_heads != 0)
                NN_ASSERT(false, "CausalSelfAttention: d_model must be divisible by num_heads");

            // 预计算因果掩码: mask[i][j] = 0 if j <= i else -inf
            // 使用 -infinity() 而非 -1e30，确保 softmax 后未来位置权重严格为 0。
            const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
            for (std::size_t i = 0; i < max_len; ++i)
                for (std::size_t j = 0; j < max_len; ++j)
                    mask_data_[i * max_len + j] = (j <= i) ? 0.0 : neg_inf;
        }

        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            auto params = w_q_.parameters();
            auto wk = w_k_.parameters();
            auto wv = w_v_.parameters();
            auto wo = w_o_.parameters();
            params.insert(params.end(), wk.begin(), wk.end());
            params.insert(params.end(), wv.begin(), wv.end());
            params.insert(params.end(), wo.begin(), wo.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            auto grads = w_q_.param_gradients();
            auto gk = w_k_.param_gradients();
            auto gv = w_v_.param_gradients();
            auto go = w_o_.param_gradients();
            grads.insert(grads.end(), gk.begin(), gk.end());
            grads.insert(grads.end(), gv.begin(), gv.end());
            grads.insert(grads.end(), go.begin(), go.end());
            return grads;
        }

        Result<Matrix> forward(const Matrix &input) override
        {
            if (input.rows() != d_model_)
                return std::unexpected(Error{"CausalSelfAttention forward input shape mismatch"});

            const std::size_t seq_len = input.cols();

            // 1. 线性投影
            auto q_res = w_q_.forward(input);
            if (!q_res) return q_res;
            Q_cache_ = *q_res;
            auto k_res = w_k_.forward(input);
            if (!k_res) return k_res;
            K_cache_ = *k_res;
            auto v_res = w_v_.forward(input);
            if (!v_res) return v_res;
            V_cache_ = *v_res;

            // 2. 分配 per-head 缓冲区
            Q_heads_.resize(num_heads_);
            K_heads_.resize(num_heads_);
            V_heads_.resize(num_heads_);
            O_heads_.resize(num_heads_);
            attn_.resize(num_heads_);

            // 3. 逐头计算因果自注意力
            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t row_start = h * d_k_;

                extract_rows(Q_cache_, row_start, d_k_, Q_heads_[h]);
                extract_rows(K_cache_, row_start, d_k_, K_heads_[h]);
                extract_rows(V_cache_, row_start, d_k_, V_heads_[h]);

                // S = Q_h^T @ K_h → (seq_len, seq_len)
                Matrix QhT = Q_heads_[h].transpose();
                attn_[h].resize(seq_len, seq_len);
                QhT.multiply_to(attn_[h], K_heads_[h]);

                // 施加因果掩码 (AST 逐元素加: a += m)
                // 修复 stride bug：使用按当前 seq_len 重建的连续 mask_buf_，
                // 而非错误的 Span(mask_data_.data(), seq_len*seq_len)。
                {
                    Span a_s = attn_[h].span();
                    Span m_s = ensure_mask_buf(seq_len);
                    compute::apply(a_s, a_s + m_s);
                }

                // 缩放
                scale_inplace(attn_[h], scale_);

                // softmax
                auto sm_res = softmax_.forward(attn_[h]);
                if (!sm_res) return sm_res;
                attn_[h] = *sm_res;

                // O_h = V_h @ A_h
                O_heads_[h].resize(d_k_, seq_len);
                V_heads_[h].multiply_to(O_heads_[h], attn_[h]);
            }

            // 4. 拼接 + 输出投影
            Matrix output(d_model_, seq_len);
            for (std::size_t h = 0; h < num_heads_; ++h)
                insert_rows(output, h * d_k_, O_heads_[h]);

            auto wo_res = w_o_.forward(output);
            if (!wo_res) return wo_res;
            return *wo_res;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            const std::size_t seq_len = grad_output.cols();

            // 1. 输出投影反向
            auto gc_res = w_o_.backward(grad_output);
            if (!gc_res) return gc_res;
            Matrix grad_concat = *gc_res;

            // 2. 初始化全局梯度
            Matrix grad_Q_all(d_model_, seq_len);
            Matrix grad_K_all(d_model_, seq_len);
            Matrix grad_V_all(d_model_, seq_len);
            grad_Q_all.zero();
            grad_K_all.zero();
            grad_V_all.zero();

            // 3. 逐头反向
            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t row_start = h * d_k_;
                const Matrix &Q_h = Q_heads_[h];
                const Matrix &K_h = K_heads_[h];
                const Matrix &V_h = V_heads_[h];
                const Matrix &A_h = attn_[h];

                extract_rows(grad_concat, row_start, d_k_, grad_O_h_buf_);

                // grad_V_h = grad_O_h @ A_h^T
                Matrix AhT = A_h.transpose();
                Matrix grad_V_h(d_k_, seq_len);
                grad_O_h_buf_.multiply_to(grad_V_h, AhT);

                // grad_A_h = V_h^T @ grad_O_h
                Matrix VhT = V_h.transpose();
                grad_A_buf_.resize(seq_len, seq_len);
                VhT.multiply_to(grad_A_buf_, grad_O_h_buf_);

                // grad_S_h = A_h ⊙ (grad_A_h - row_sum(A_h ⊙ grad_A_h))
                // 通过 AST + Matrix 原语表达（与 MHA.backward 相同算法）：
                //   1. ep = A ⊙ grad_A                (AST 逐元素乘)
                //   2. dot[r] = Σ_c ep[r][c]          (按行归约)
                //   3. gmd = grad_A - dot             (按行广播减)
                //   4. grad_S = A ⊙ gmd              (AST 逐元素乘)
                grad_scores_buf_.resize(seq_len, seq_len);
                Matrix ep(seq_len, seq_len);
                ConstSpan a_s = A_h.span();
                Span ga_s = grad_A_buf_.span();
                Span ep_s = ep.span();
                compute::apply(ep_s, a_s * ga_s);

                Matrix dot = ep.row_reduce(Scalar{0},
                    [](Scalar x, Scalar y) noexcept { return x + y; },
                    [](Scalar x) noexcept { return x; });

                Matrix gmd = grad_A_buf_;
                gmd.broadcast_row_inplace(dot,
                    [](Scalar ga, Scalar d) noexcept { return ga - d; });

                Span gmd_s = gmd.span();
                Span gs_s = grad_scores_buf_.span();
                compute::apply(gs_s, a_s * gmd_s);

                // grad_Q_h = K_h @ grad_S_h^T * scale
                Matrix gsT = grad_scores_buf_.transpose();
                Matrix grad_Q_h(d_k_, seq_len);
                K_h.multiply_to(grad_Q_h, gsT);
                scale_inplace(grad_Q_h, scale_);

                // grad_K_h = Q_h @ grad_S_h * scale
                Matrix grad_K_h(d_k_, seq_len);
                Q_h.multiply_to(grad_K_h, grad_scores_buf_);
                scale_inplace(grad_K_h, scale_);

                insert_rows(grad_Q_all, row_start, grad_Q_h);
                insert_rows(grad_K_all, row_start, grad_K_h);
                insert_rows(grad_V_all, row_start, grad_V_h);
            }

            // 4. 投影层反向
            auto giq_res = w_q_.backward(grad_Q_all);
            if (!giq_res) return giq_res;
            Matrix grad_input = *giq_res;
            auto gik_res = w_k_.backward(grad_K_all);
            if (!gik_res) return gik_res;
            grad_input = grad_input + *gik_res;
            auto giv_res = w_v_.backward(grad_V_all);
            if (!giv_res) return giv_res;
            grad_input = grad_input + *giv_res;

            return grad_input;
        }
    };

    // ── GPT Transformer 块 (Pre-Norm Decoder Block) ──────────────────
    //   x = x + CausalSelfAttn(LayerNorm₁(x))
    //   x = x + FFN(LayerNorm₂(x))
    // 输入/输出形状: (d_model, seq_len)
    class GPTBlock final : public Layer
    {
    private:
        CausalSelfAttention self_attn_;
        LayerNorm norm1_;
        FeedForward ff_;
        LayerNorm norm2_;

        Matrix residual1_cache_;
        Matrix residual2_cache_;

    public:
        GPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
                 std::size_t max_len = 1024)
            : self_attn_(d_model, num_heads, max_len),
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

        Result<Matrix> forward(const Matrix &input) override
        {
            // 子层1: CausalSelfAttention + 残差
            residual1_cache_ = input;
            auto n1_res = norm1_.forward(input);
            if (!n1_res) return n1_res;
            auto sa_res = self_attn_.forward(*n1_res);
            if (!sa_res) return sa_res;
            residual2_cache_ = input + *sa_res;

            // 子层2: FFN + 残差
            auto n2_res = norm2_.forward(residual2_cache_);
            if (!n2_res) return n2_res;
            auto ff_res = ff_.forward(*n2_res);
            if (!ff_res) return ff_res;
            return residual2_cache_ + *ff_res;
        }

        Result<Matrix> backward(const Matrix &grad_output) override
        {
            Matrix grad_residual1 = grad_output;
            Matrix grad_ff_out    = grad_output;

            auto b_ff = ff_.backward(grad_ff_out);
            if (!b_ff) return b_ff;
            auto b_n2 = norm2_.backward(*b_ff);
            if (!b_n2) return b_n2;
            grad_residual1 = grad_residual1 + *b_n2;

            Matrix grad_input = grad_residual1;
            Matrix grad_attn_out = grad_residual1;

            auto b_sa = self_attn_.backward(grad_attn_out);
            if (!b_sa) return b_sa;
            auto b_n1 = norm1_.backward(*b_sa);
            if (!b_n1) return b_n1;
            grad_input = grad_input + *b_n1;

            return grad_input;
        }
    };

    // ── GPT 语言模型 ─────────────────────────────────────────────────
    // Decoder-only Transformer 用于自回归文本生成。
    //
    // 组件: TokenEmbedding + PositionalEncoding + N × GPTBlock + LayerNorm + LM Head
    //
    // 输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
    // 输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
    //
    // 传播流程:
    //   输入: (seq_len, batch) token IDs
    //   → Embed: (d_model, seq_len × batch)
    //   → + PE: (d_model, seq_len × batch)
    //   → N × GPTBlock: (d_model, seq_len × batch)
    //   → LayerNorm: (d_model, seq_len × batch)
    //   → LM Head: (vocab_size, seq_len × batch)
    // ────────────────────────────────────────────────────────────────────
    class GPTModel final : public Layer
    {
    private:
        std::size_t vocab_size_;
        std::size_t d_model_;
        std::size_t seq_len_;

        // 嵌入层
        Matrix token_emb_;      // (vocab_size, d_model) — 查找表
        Matrix grad_token_emb_;

        // 位置编码（可学习）
        Matrix pos_emb_;        // (max_seq_len, d_model)
        Matrix grad_pos_emb_;

        // Transformer 块
        std::vector<GPTBlock> blocks_;

        // 最终 LayerNorm
        LayerNorm ln_f_;

        // LM Head（权重与 token_emb_ 共享或独立）
        Linear lm_head_;        // (vocab_size, d_model)

        // 反向传播缓存
        std::vector<Matrix> stored_inputs_;   // 每个样本经 PE 后的输入
        std::vector<std::vector<std::size_t>> stored_tokens_; // token IDs
        std::size_t batch_size_{0};

    public:
        GPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
                 std::size_t num_heads, std::size_t d_ff, std::size_t num_layers)
            : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
              token_emb_(vocab_size, d_model),
              grad_token_emb_(vocab_size, d_model),
              pos_emb_(seq_len, d_model),
              grad_pos_emb_(seq_len, d_model),
              ln_f_(d_model),
              lm_head_(d_model, vocab_size)
        {
            // Xavier-like 初始化嵌入
            {
                constexpr Scalar emb_init_std = 0.02;
                std::mt19937_64 rng{42};
                std::normal_distribution<Scalar> dist(0.0, emb_init_std);
                auto te = token_emb_.span();
                for (std::size_t i = 0; i < token_emb_.size(); ++i)
                    te[i] = dist(rng);
                auto pe = pos_emb_.span();
                for (std::size_t i = 0; i < pos_emb_.size(); ++i)
                    pe[i] = dist(rng);
            }

            for (std::size_t i = 0; i < num_layers; ++i)
                blocks_.emplace_back(d_model, num_heads, d_ff, seq_len);
        }

        // ── 可学习嵌入参数 ─────────────────────────────────────────
        std::vector<std::reference_wrapper<Matrix>> parameters() override
        {
            std::vector<std::reference_wrapper<Matrix>> params;
            params.push_back(std::ref(token_emb_));
            params.push_back(std::ref(pos_emb_));
            for (auto &b : blocks_)
            {
                auto bp = b.parameters();
                params.insert(params.end(), bp.begin(), bp.end());
            }
            auto lp = ln_f_.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
            auto hp = lm_head_.parameters();
            params.insert(params.end(), hp.begin(), hp.end());
            return params;
        }

        std::vector<std::reference_wrapper<Matrix>> param_gradients() override
        {
            std::vector<std::reference_wrapper<Matrix>> grads;
            grads.push_back(std::ref(grad_token_emb_));
            grads.push_back(std::ref(grad_pos_emb_));
            for (auto &b : blocks_)
            {
                auto bg = b.param_gradients();
                grads.insert(grads.end(), bg.begin(), bg.end());
            }
            auto lg = ln_f_.param_gradients();
            grads.insert(grads.end(), lg.begin(), lg.end());
            auto hg = lm_head_.param_gradients();
            grads.insert(grads.end(), hg.begin(), hg.end());
            return grads;
        }

        // ── 前向传播 ────────────────────────────────────────────────
        // input: (seq_len, batch_size) — token IDs 作为 Scalar
        Result<Matrix> forward(const Matrix &input) override
        {
            const std::size_t seq_len = input.rows();
            batch_size_ = input.cols();
            stored_inputs_.resize(batch_size_);
            stored_tokens_.resize(batch_size_);

            // 清空缓存，防止跨 forward 累积
            for (auto &v : stored_tokens_)
                v.clear();

            // 输出: (vocab_size, seq_len × batch_size)
            Matrix output(vocab_size_, seq_len * batch_size_);

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                stored_tokens_[b].reserve(seq_len);

                // 1. Token 嵌入 + 位置嵌入 → (d_model, seq_len)
                Matrix x(d_model_, seq_len);
                for (std::size_t t = 0; t < seq_len; ++t)
                {
                    auto token_id = static_cast<std::size_t>(input.at_unchecked(t, b));
                    if (token_id >= vocab_size_)
                        return std::unexpected(Error{
                            "GPTModel::forward token id out of range: " +
                            std::to_string(token_id) +
                            " (vocab_size=" + std::to_string(vocab_size_) + ")"});
                    stored_tokens_[b].push_back(token_id);

                    const auto emb_span = token_emb_.span();
                    const auto pos_span = pos_emb_.span();
                    auto out_span = x.span();
                    for (std::size_t d = 0; d < d_model_; ++d)
                        out_span[d * seq_len + t] =
                            emb_span[token_id * d_model_ + d] +
                            pos_span[t * d_model_ + d];
                }

                stored_inputs_[b] = x;

                // 2. 通过 Transformer 块
                for (std::size_t l = 0; l < blocks_.size(); ++l)
                {
                    auto blk_res = blocks_[l].forward(x);
                    if (!blk_res) return blk_res;
                    x = *blk_res;
                }

                // 3. 最终 LayerNorm
                auto ln_res = ln_f_.forward(x);
                if (!ln_res) return ln_res;
                x = *ln_res;

                // 4. LM Head: (d_model, seq_len) → (vocab_size, seq_len)
                auto lm_res = lm_head_.forward(x);
                if (!lm_res) return lm_res;
                Matrix logits = *lm_res;

                // 5. 写入输出 (vocab_size, seq_len × batch)
                //    布局: 列 t * batch_size + b 对应样本 b 的位置 t
                auto log_span = logits.span();
                auto out_span2 = output.span();
                for (std::size_t r = 0; r < vocab_size_; ++r)
                    for (std::size_t t = 0; t < seq_len; ++t)
                        out_span2[r * (seq_len * batch_size_) + t * batch_size_ + b] =
                            log_span[r * seq_len + t];
            }

            return output;
        }

        // ── 反向传播 ────────────────────────────────────────────────
        // grad_output: (vocab_size, seq_len × batch_size)
        Result<Matrix> backward(const Matrix &grad_output) override
        {
            const std::size_t seq_len = seq_len_;
            grad_token_emb_.zero();
            grad_pos_emb_.zero();

            Matrix grad_input(seq_len, batch_size_);
            grad_input.zero();

            for (std::size_t b = 0; b < batch_size_; ++b)
            {
                // 提取该样本的梯度: (vocab_size, seq_len)
                Matrix grad_logits(vocab_size_, seq_len);
                auto go_span = grad_output.span();
                auto gl_span = grad_logits.span();
                for (std::size_t r = 0; r < vocab_size_; ++r)
                    for (std::size_t t = 0; t < seq_len; ++t)
                        gl_span[r * seq_len + t] =
                            go_span[r * (seq_len * batch_size_) + t * batch_size_ + b];

                // Re-forward 重建缓存
                Matrix x = stored_inputs_[b];
                for (std::size_t l = 0; l < blocks_.size(); ++l)
                {
                    auto blk_res = blocks_[l].forward(x);
                    if (!blk_res) return blk_res;
                    x = *blk_res;
                }
                auto ln_fwd = ln_f_.forward(x);
                if (!ln_fwd) return ln_fwd;
                x = *ln_fwd;

                // LM Head 反向 → (d_model, seq_len)
                auto b_lm = lm_head_.backward(grad_logits);
                if (!b_lm) return b_lm;
                Matrix grad_ln = *b_lm;

                // LayerNorm 反向
                auto b_ln = ln_f_.backward(grad_ln);
                if (!b_ln) return b_ln;
                grad_ln = *b_ln;

                // 逐块反向
                for (int l = static_cast<int>(blocks_.size()) - 1; l >= 0; --l)
                {
                    auto b_blk = blocks_[l].backward(grad_ln);
                    if (!b_blk) return b_blk;
                    grad_ln = *b_blk;
                }

                // 累加嵌入梯度
                const auto tokens = stored_tokens_[b];
                for (std::size_t t = 0; t < seq_len; ++t)
                {
                    const std::size_t tid = tokens[t];
                    // token embedding 梯度
                    auto gte = grad_token_emb_.span();
                    for (std::size_t d = 0; d < d_model_; ++d)
                        gte[tid * d_model_ + d] +=
                            grad_ln.at_unchecked(d, t);
                    // positional embedding 梯度
                    auto gpe = grad_pos_emb_.span();
                    for (std::size_t d = 0; d < d_model_; ++d)
                        gpe[t * d_model_ + d] +=
                            grad_ln.at_unchecked(d, t);
                    // grad_input (token IDs 无梯度，此处仅用于接口一致性)
                    grad_input.set_value_unchecked(t, b, 0.0);
                }
            }

            return grad_input;
        }

        // ── 采样生成（支持温度采样 + 贪心） ────────────────────────
        [[nodiscard]] Result<std::vector<std::size_t>>
        generate(const std::vector<std::size_t> &prompt,
                 std::size_t max_new_tokens,
                 Scalar temperature = 1.0) override
        {
            std::vector<std::size_t> context(prompt);
            std::vector<std::size_t> generated;
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

            for (std::size_t step = 0; step < max_new_tokens; ++step)
            {
                // 截取最后 seq_len 个 token
                std::size_t start = 0;
                if (context.size() > seq_len_)
                    start = context.size() - seq_len_;

                std::size_t cur_len = context.size() - start;
                Matrix input(cur_len, 1);
                for (std::size_t t = 0; t < cur_len; ++t)
                    input.set_value_unchecked(t, 0,
                        static_cast<Scalar>(context[start + t]));

                auto logits_res = forward(input); // (vocab_size, cur_len)
                if (!logits_res)
                    return std::unexpected(std::move(logits_res).error());
                auto logits = *logits_res;

                // 取最后一个位置的 logits
                std::vector<Scalar> last_logits(vocab_size_);
                for (std::size_t v = 0; v < vocab_size_; ++v)
                    last_logits[v] = logits.at_unchecked(v, cur_len - 1);

                // temperature
                if (temperature > 0.0 && temperature != 1.0)
                {
                    for (auto &v : last_logits)
                        v /= temperature;
                }

                // softmax（数值稳定）
                Scalar max_val = last_logits[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    max_val = std::max(max_val, last_logits[v]);
                Scalar sum_exp = 0.0;
                for (auto &v : last_logits)
                {
                    v = std::exp(v - max_val);
                    sum_exp += v;
                }
                for (auto &v : last_logits)
                    v /= sum_exp;

                // 采样
                std::size_t next_token;
                if (temperature > 0.0 && temperature != 1.0)
                {
                    // 概率采样
                    Scalar r = dist(rng);
                    Scalar cumulative = 0.0;
                    next_token = vocab_size_ - 1;
                    for (std::size_t v = 0; v < vocab_size_; ++v)
                    {
                        cumulative += last_logits[v];
                        if (r <= cumulative)
                        {
                            next_token = v;
                            break;
                        }
                    }
                }
                else
                {
                    // 贪心
                    next_token = 0;
                    Scalar best = last_logits[0];
                    for (std::size_t v = 1; v < vocab_size_; ++v)
                    {
                        if (last_logits[v] > best)
                        {
                            best = last_logits[v];
                            next_token = v;
                        }
                    }
                }

                context.push_back(next_token);
                generated.push_back(next_token);
            }

            return generated;
        }
    };
}

#endif // NN_COMPUTE_LAYER_HPP