#ifndef NN_COMPUTE_LAYER_HPP
#define NN_COMPUTE_LAYER_HPP

// ── compute_layer.hpp — 引擎化计算层 ───────────────────────────────────────
//
// 架构铁律：
//   1. Layer 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。不再有 forward_gpu / backward_gpu。
//   2. 算法只在 Layer（通过组合 engine 原语表达），绝不在 Engine/Shader 中。
//   3. Engine/Shader 只提供 op-level 原语（matmul, add, exp, max, reduce 等）。
//
// 算法表达示例：
//   ReLU forward  = max(x, 0)                    → engine.elementwise_binary_scalar(Max, x, 0)
//   ReLU backward = (x > 0) ? grad : 0           → engine.elementwise_select_scalar_cond(Gt, x, 0, grad, 0)
//   Linear fwd    = W×x + b                      → engine.matmul(W, x) + engine.broadcast_row_add(out, b)
//   Linear bwd    = W^T×grad, grad_W += grad×x^T, grad_b += row_sum(grad)
//   GeLU fwd      = x * sigmoid(β*x)             → 5 次原语组合
//   LayerNorm fwd = (x - mean) * rsqrt(var+eps) * gamma + beta
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "tensor.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Layer — 引擎化计算层基类
// ══════════════════════════════════════════════════════════════════════════
class Layer
{
public:
    virtual ~Layer() = default;

    // forward/backward 接收 ComputeEngine 引用，自动适配 CPU/GPU
    // 只有一套实现，不再有 forward_gpu / backward_gpu
    [[nodiscard]] virtual Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) = 0;

    [[nodiscard]] virtual Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) = 0;

    // 参数访问（供 optimizer 使用）
    [[nodiscard]] virtual std::vector<Tensor*> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<Tensor*> param_gradients() { return {}; }

    // 梯度清零（每个训练 step 开始前调用）
    virtual void zero_grad(ComputeEngine& engine)
    {
        for (auto* grad : param_gradients())
            (void)engine.zero(*grad);
    }
};

// ── 辅助：深拷贝 Tensor（通过 engine.clone()，无 PCIe 传输） ──────────────
// 用于需要修改中间结果但不影响原 Tensor 的场景（如 LayerNorm 中的 diff）
[[nodiscard]] inline Result<Tensor> clone_tensor(
    ComputeEngine& engine, const Tensor& src)
{
    return engine.clone(src);
}

// ══════════════════════════════════════════════════════════════════════════
// Linear — 全连接层
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = W × x + b
//   backward: grad_x = W^T × grad_out
//             grad_W += grad_out × x^T
//             grad_b += Σ_batch grad_out
// ══════════════════════════════════════════════════════════════════════════
class Linear final : public Layer
{
private:
    Tensor w_;           // 权重 (out_features, in_features)
    Tensor b_;           // 偏置 (out_features, 1)
    Tensor grad_w_;      // 权重梯度
    Tensor grad_b_;      // 偏置梯度
    Tensor input_cache_; // forward 输入缓存（供 backward 使用）

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

public:
    Linear(ComputeEngine& engine,
           std::size_t in_features, std::size_t out_features)
    {
        // ── 在 CPU 上初始化权重（Xavier 均匀分布） ──
        Matrix w_cpu(out_features, in_features);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features + out_features));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_features, 1);  // 零初始化

        // ── 通过 engine 上传到目标设备 ──
        auto w_res = engine.from_matrix(w_cpu);
        NN_ASSERT(w_res, "Linear: failed to create weight tensor");
        w_ = std::move(*w_res);

        auto b_res = engine.from_matrix(b_cpu);
        NN_ASSERT(b_res, "Linear: failed to create bias tensor");
        b_ = std::move(*b_res);

        // ── 创建梯度张量（零初始化） ──
        grad_w_ = engine.create_tensor(out_features, in_features);
        grad_b_ = engine.create_tensor(out_features, 1);
        (void)engine.zero(grad_w_);
        (void)engine.zero(grad_b_);
    }

    [[nodiscard]] std::vector<Tensor*> parameters() override
    {
        return {&w_, &b_};
    }

    [[nodiscard]] std::vector<Tensor*> param_gradients() override
    {
        return {&grad_w_, &grad_b_};
    }

    // ── forward: out = W × x + b ──────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != w_.cols())
            return std::unexpected(Error{"linear forward: input shape mismatch"});

        // 缓存输入供 backward 使用
        input_cache_ = input;

        // out = W × x
        auto out = engine.matmul(w_, input);
        if (!out) return std::unexpected(out.error());

        // out += b（按行广播加法：out[f][b] += bias[f]）
        auto r = engine.broadcast_row_inplace(*out, b_, BinaryOp::Add);
        if (!r) return std::unexpected(r.error());

        return out;
    }

    // ── backward: grad_x = W^T × grad_out, 累积 grad_W / grad_b ──────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (grad_output.rows() != w_.rows())
            return std::unexpected(Error{"linear backward: grad_output shape mismatch"});

        // grad_input = W^T × grad_output
        auto grad_input = engine.matmul(w_, grad_output, true, false);
        if (!grad_input) return std::unexpected(grad_input.error());

        // grad_W += grad_output × input_cache^T
        auto gw = engine.matmul(grad_output, input_cache_, false, true);
        if (!gw) return std::unexpected(gw.error());
        auto r1 = engine.add_inplace(grad_w_, *gw);
        if (!r1) return std::unexpected(r1.error());

        // grad_b += Σ_batch grad_output（按行归约求和）
        auto gb = engine.row_reduce_sum(grad_output);
        if (!gb) return std::unexpected(gb.error());
        auto r2 = engine.add_inplace(grad_b_, *gb);
        if (!r2) return std::unexpected(r2.error());

        return grad_input;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ReLU — ReLU 激活函数
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = max(x, 0)
//   backward: grad_x = (x > 0) ? grad_out : 0
// ══════════════════════════════════════════════════════════════════════════
class ReLU final : public Layer
{
private:
    Tensor input_cache_;

public:
    ReLU() = default;

    // ── forward: out = max(x, 0) ──────────────────────────────────────────
    // ReLU 算法由 Layer 表达为 Max 原语 + 标量 0
    // Engine/Shader 只提供 Max 原语，不知道 "ReLU" 是什么
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        input_cache_ = input;
        return engine.elementwise_binary_scalar(BinaryOp::Max, input, Scalar{0});
    }

    // ── backward: grad_x = (x > 0) ? grad_out : 0 ────────────────────────
    // ReLU 反向算法由 Layer 表达为 Select + Gt 原语
    // Engine/Shader 只提供 Select/Gt 原语，不知道 "ReLU backward" 是什么
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            return std::unexpected(Error{"relu backward: shape mismatch"});

        return engine.elementwise_select_scalar_cond(
            CompareOp::Gt,           // 条件：x > 0
            input_cache_,            // 条件操作数 A = x
            Scalar{0},               // 条件操作数 B = 0 (标量)
            grad_output,             // then = grad_output (张量)
            Scalar{0});              // else = 0 (标量)
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GeLU — QuickGeLU 激活函数（复杂算法的原语组合）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = x * sigmoid(β * x)
//             = x * (1 / (1 + exp(-β * x)))
//   backward: factor = s * (1 + βx * (1 - s)),  s = sigmoid(βx)
//             grad_x = grad_out * factor
//
// 原语分解：
//   forward: 5 次原语调用（Mul→Neg→Exp→Add→Div→Mul）
//   backward: 5 次原语调用（Sub→Mul→Mul→Add→Mul→Mul）
//   shader 只有 Mul/Neg/Exp/Add/Div 原语，不知道 "GeLU" 是什么
// ══════════════════════════════════════════════════════════════════════════
class GeLU final : public Layer
{
private:
    static constexpr Scalar BETA = 1.702f;
    Tensor input_cache_;
    Tensor sigmoid_cache_;  // 缓存 sigmoid(βx) 供 backward 使用

public:
    GeLU() = default;

    // ── forward: out = x * sigmoid(β * x) ────────────────────────────────
    // 分解为原语：
    //   t1 = x * β                    (binary_scalar Mul)
    //   t2 = -t1                      (unary Neg)  → -βx
    //   t3 = exp(t2)                  (unary Exp)  → exp(-βx)
    //   t4 = t3 + 1                   (binary_scalar Add)
    //   s  = 1 / t4                   (binary_scalar Div, scalar_first)  → sigmoid(βx)
    //   out = x * s                   (binary Mul)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        input_cache_ = input;

        // t1 = x * β
        auto t1 = engine.elementwise_binary_scalar(BinaryOp::Mul, input, BETA);
        if (!t1) return std::unexpected(t1.error());

        // t2 = -t1 (= -βx)
        auto t2 = engine.elementwise_unary(UnaryOp::Neg, *t1);
        if (!t2) return std::unexpected(t2.error());

        // t3 = exp(t2)
        auto t3 = engine.elementwise_unary(UnaryOp::Exp, *t2);
        if (!t3) return std::unexpected(t3.error());

        // t4 = t3 + 1
        auto t4 = engine.elementwise_binary_scalar(BinaryOp::Add, *t3, Scalar{1});
        if (!t4) return std::unexpected(t4.error());

        // s = 1 / t4  (scalar_first: 1.0 / t4)
        auto s = engine.elementwise_binary_scalar(BinaryOp::Div, *t4, Scalar{1}, true);
        if (!s) return std::unexpected(s.error());

        sigmoid_cache_ = *s;  // 缓存供 backward

        // out = x * s
        return engine.elementwise_binary(BinaryOp::Mul, input, *s);
    }

    // ── backward: grad_x = grad_out * factor ─────────────────────────────
    // factor = s * (1 + βx * (1 - s)),  s = sigmoid(βx)
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            return std::unexpected(Error{"gelu backward: shape mismatch"});

        // s = sigmoid_cache_
        // one_minus_s = 1 - s
        auto one_minus_s = engine.elementwise_binary_scalar(
            BinaryOp::Sub, sigmoid_cache_, Scalar{1}, true);
        if (!one_minus_s) return std::unexpected(one_minus_s.error());

        // βx = input * β
        auto bx = engine.elementwise_binary_scalar(BinaryOp::Mul, input_cache_, BETA);
        if (!bx) return std::unexpected(bx.error());

        // inner = βx * (1 - s)
        auto inner = engine.elementwise_binary(BinaryOp::Mul, *bx, *one_minus_s);
        if (!inner) return std::unexpected(inner.error());

        // paren = 1 + inner
        auto paren = engine.elementwise_binary_scalar(BinaryOp::Add, *inner, Scalar{1});
        if (!paren) return std::unexpected(paren.error());

        // factor = s * paren
        auto factor = engine.elementwise_binary(BinaryOp::Mul, sigmoid_cache_, *paren);
        if (!factor) return std::unexpected(factor.error());

        // grad_input = grad_out * factor
        return engine.elementwise_binary(BinaryOp::Mul, grad_output, *factor);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// LayerNorm — 层归一化
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  mean = (1/F) * Σ_f x[f][b]              (col_reduce_sum + scale)
//             diff = x - mean                          (broadcast_col Sub)
//             var = (1/F) * Σ_f diff²                  (elementwise Mul + col_reduce_sum + scale)
//             std_inv = 1 / sqrt(var + eps)            (binary_scalar Add + unary Rsqrt)
//             normalized = diff * std_inv              (broadcast_col Mul)
//             out = gamma * normalized + beta          (broadcast_row Mul + broadcast_row Add)
//
//   backward: gy = grad_out * gamma                    (broadcast_row Mul)
//             mean_g = (1/F) * Σ_f gy                  (col_reduce_sum + scale)
//             gy_norm = gy * normalized                (elementwise Mul)
//             mean_gn = (1/F) * Σ_f gy_norm            (col_reduce_sum + scale)
//             grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
//             grad_gamma += Σ_b gy_norm                (row_reduce_sum)
//             grad_beta += Σ_b grad_out                (row_reduce_sum)
// ══════════════════════════════════════════════════════════════════════════
class LayerNorm final : public Layer
{
private:
    std::size_t normalized_shape_;
    Scalar epsilon_;

    // 可学习参数
    Tensor gamma_;      // (normalized_shape, 1)
    Tensor beta_;       // (normalized_shape, 1)
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // backward 缓存
    Tensor normalized_cache_;  // (features, batch)
    Tensor std_cache_;         // (1, batch) — std_inv

    static constexpr Scalar EPSILON = 1e-5;

public:
    explicit LayerNorm(ComputeEngine& engine,
                       std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon)
    {
        // gamma 初始化为 1, beta 初始化为 0
        Matrix gamma_cpu(normalized_shape, 1, Scalar{1});
        Matrix beta_cpu(normalized_shape, 1, Scalar{0});

        auto g = engine.from_matrix(gamma_cpu);
        NN_ASSERT(g, "LayerNorm: failed to create gamma tensor");
        gamma_ = std::move(*g);

        auto b = engine.from_matrix(beta_cpu);
        NN_ASSERT(b, "LayerNorm: failed to create beta tensor");
        beta_ = std::move(*b);

        grad_gamma_ = engine.create_tensor(normalized_shape, 1);
        grad_beta_ = engine.create_tensor(normalized_shape, 1);
        (void)engine.zero(grad_gamma_);
        (void)engine.zero(grad_beta_);
    }

    [[nodiscard]] std::vector<Tensor*> parameters() override
    {
        return {&gamma_, &beta_};
    }

    [[nodiscard]] std::vector<Tensor*> param_gradients() override
    {
        return {&grad_gamma_, &grad_beta_};
    }

    // ── forward ───────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != normalized_shape_)
            return std::unexpected(Error{"layernorm forward: input shape mismatch"});

        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);

        // mean = col_reduce_sum(input) * inv_features → (1, batch)
        auto mean = engine.col_reduce_sum(input);
        if (!mean) return std::unexpected(mean.error());
        auto r = engine.scale_inplace(*mean, inv_features);
        if (!r) return std::unexpected(r.error());

        // diff = input - mean (broadcast_col Sub) — 需要深拷贝 input
        auto diff = clone_tensor(engine, input);
        if (!diff) return std::unexpected(diff.error());
        r = engine.broadcast_col_inplace(*diff, *mean, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // diff_sq = diff * diff
        auto diff_sq = engine.elementwise_binary(BinaryOp::Mul, *diff, *diff);
        if (!diff_sq) return std::unexpected(diff_sq.error());

        // var = col_reduce_sum(diff_sq) * inv_features → (1, batch)
        auto var = engine.col_reduce_sum(*diff_sq);
        if (!var) return std::unexpected(var.error());
        r = engine.scale_inplace(*var, inv_features);
        if (!r) return std::unexpected(r.error());

        // var_eps = var + eps
        auto var_eps = engine.elementwise_binary_scalar(BinaryOp::Add, *var, epsilon_);
        if (!var_eps) return std::unexpected(var_eps.error());

        // std_inv = rsqrt(var + eps) → (1, batch)
        auto std_inv = engine.elementwise_unary(UnaryOp::Rsqrt, *var_eps);
        if (!std_inv) return std::unexpected(std_inv.error());

        // normalized = diff * std_inv (broadcast_col Mul) — 就地修改 diff
        r = engine.broadcast_col_inplace(*diff, *std_inv, BinaryOp::Mul);
        if (!r) return std::unexpected(r.error());

        // 缓存 normalized 和 std_inv 供 backward 使用
        normalized_cache_ = *diff;
        std_cache_ = *std_inv;

        // output = gamma * normalized + beta (broadcast_row)
        // 先深拷贝 normalized，再就地广播
        auto output = clone_tensor(engine, *diff);
        if (!output) return std::unexpected(output.error());

        r = engine.broadcast_row_inplace(*output, gamma_, BinaryOp::Mul);
        if (!r) return std::unexpected(r.error());

        r = engine.broadcast_row_inplace(*output, beta_, BinaryOp::Add);
        if (!r) return std::unexpected(r.error());

        return output;
    }

    // ── backward ──────────────────────────────────────────────────────────
    // grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
    // gy = grad_out * gamma
    // mean_g = mean(gy), mean_gn = mean(gy * normalized)
    // grad_gamma += row_sum(gy * normalized)
    // grad_beta += row_sum(grad_out)
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);

        // gy = grad_out * gamma (broadcast_row Mul) — 深拷贝 grad_output
        auto gy = clone_tensor(engine, grad_output);
        if (!gy) return std::unexpected(gy.error());
        auto r = engine.broadcast_row_inplace(*gy, gamma_, BinaryOp::Mul);
        if (!r) return std::unexpected(r.error());

        // mean_g = col_reduce_sum(gy) * inv_features → (1, batch)
        auto mean_g = engine.col_reduce_sum(*gy);
        if (!mean_g) return std::unexpected(mean_g.error());
        r = engine.scale_inplace(*mean_g, inv_features);
        if (!r) return std::unexpected(r.error());

        // gy_norm = gy * normalized_cache_ (elementwise Mul)
        auto gy_norm = engine.elementwise_binary(BinaryOp::Mul, *gy, normalized_cache_);
        if (!gy_norm) return std::unexpected(gy_norm.error());

        // mean_gn = col_reduce_sum(gy_norm) * inv_features → (1, batch)
        auto mean_gn = engine.col_reduce_sum(*gy_norm);
        if (!mean_gn) return std::unexpected(mean_gn.error());
        r = engine.scale_inplace(*mean_gn, inv_features);
        if (!r) return std::unexpected(r.error());

        // t1 = gy - mean_g (broadcast_col Sub) — 就地修改 gy
        r = engine.broadcast_col_inplace(*gy, *mean_g, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // t2 = normalized * mean_gn (broadcast_col Mul) — 深拷贝 normalized
        auto t2 = clone_tensor(engine, normalized_cache_);
        if (!t2) return std::unexpected(t2.error());
        r = engine.broadcast_col_inplace(*t2, *mean_gn, BinaryOp::Mul);
        if (!r) return std::unexpected(r.error());

        // grad_input = (t1 - t2) * std_inv
        // 先 t1 -= t2 (elementwise Sub)
        auto t2_neg = engine.elementwise_binary(BinaryOp::Sub, *gy, *t2);
        if (!t2_neg) return std::unexpected(t2_neg.error());

        // 上面创建新 Tensor 作为 (gy - t2)，再乘 std_inv (broadcast_col Mul)
        r = engine.broadcast_col_inplace(*t2_neg, std_cache_, BinaryOp::Mul);
        if (!r) return std::unexpected(r.error());

        // grad_gamma += row_reduce_sum(gy_norm) → (features, 1)
        auto gg = engine.row_reduce_sum(*gy_norm);
        if (!gg) return std::unexpected(gg.error());
        r = engine.add_inplace(grad_gamma_, *gg);
        if (!r) return std::unexpected(r.error());

        // grad_beta += row_reduce_sum(grad_output) → (features, 1)
        auto gb = engine.row_reduce_sum(grad_output);
        if (!gb) return std::unexpected(gb.error());
        r = engine.add_inplace(grad_beta_, *gb);
        if (!r) return std::unexpected(r.error());

        return t2_neg;
    }
};

} // namespace nn

#endif // NN_COMPUTE_LAYER_HPP
