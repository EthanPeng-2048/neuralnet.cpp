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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "tensor.hpp"
#include "model_spec.hpp"

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

    // 参数访问（供 optimizer 使用）— 使用 reference_wrapper 替代裸指针，明确表达非拥有语义
    [[nodiscard]] virtual std::vector<TensorRef> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<TensorRef> param_gradients() { return {}; }

    // 梯度清零（每个训练 step 开始前调用）
    [[nodiscard]] virtual Result<void> zero_grad(ComputeEngine& engine)
    {
        for (auto& grad : param_gradients())
        {
            auto r = engine.zero(grad);
            if (!r) return r;
        }
        return {};
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
        NN_ASSERT(w_res, w_res ? "" : w_res.error().message.c_str());
        w_ = std::move(*w_res);

        auto b_res = engine.from_matrix(b_cpu);
        NN_ASSERT(b_res, b_res ? "" : b_res.error().message.c_str());
        b_ = std::move(*b_res);

        // ── 创建梯度张量（零初始化） ──
        grad_w_ = engine.create_tensor(out_features, in_features);
        grad_b_ = engine.create_tensor(out_features, 1);
        { auto r1 = engine.zero(grad_w_); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }
        { auto r2 = engine.zero(grad_b_); NN_ASSERT(r2, r2 ? "" : r2.error().message.c_str()); }
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {w_, b_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_w_, grad_b_};
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
        NN_ASSERT(g, g ? "" : g.error().message.c_str());
        gamma_ = std::move(*g);

        auto bv = engine.from_matrix(beta_cpu);
        NN_ASSERT(bv, bv ? "" : bv.error().message.c_str());
        beta_ = std::move(*bv);

        grad_gamma_ = engine.create_tensor(normalized_shape, 1);
        grad_beta_ = engine.create_tensor(normalized_shape, 1);
        { auto r1 = engine.zero(grad_gamma_); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }
        { auto r2 = engine.zero(grad_beta_);  NN_ASSERT(r2, r2 ? "" : r2.error().message.c_str()); }
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {gamma_, beta_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_gamma_, grad_beta_};
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

// ══════════════════════════════════════════════════════════════════════════
// Softmax — 按行 softmax（用于注意力权重）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  row_max[r] = max_c x[r][c]               (row_reduce_max)
//             shifted[r][c] = x[r][c] - row_max[r]    (broadcast_row Sub)
//             exp_shift[r][c] = exp(shifted[r][c])     (unary Exp)
//             row_sum[r] = Σ_c exp_shift[r][c]         (row_reduce_sum)
//             out[r][c] = exp_shift[r][c] / row_sum[r] (broadcast_row Div)
//   backward: grad_x = out ⊙ (grad_out - row_dot(out ⊙ grad_out))
//             row_dot[r] = Σ_c out[r][c] * grad_out[r][c]
// ══════════════════════════════════════════════════════════════════════════
class Softmax final : public Layer
{
private:
    Tensor output_cache_;

public:
    Softmax() = default;

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        // 1. row_max = max per row
        auto row_max = engine.row_reduce_max(input);
        if (!row_max) return std::unexpected(row_max.error());

        // 2. shifted = input - row_max (broadcast_row Sub) — 深拷贝 input
        auto shifted = clone_tensor(engine, input);
        if (!shifted) return std::unexpected(shifted.error());
        auto r = engine.broadcast_row_inplace(*shifted, *row_max, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // 3. exp_shift = exp(shifted)
        auto exp_shift = engine.elementwise_unary(UnaryOp::Exp, *shifted);
        if (!exp_shift) return std::unexpected(exp_shift.error());

        // 4. row_sum = Σ_c exp_shift[r][c]
        auto row_sum = engine.row_reduce_sum(*exp_shift);
        if (!row_sum) return std::unexpected(row_sum.error());

        // 5. output = exp_shift / row_sum (broadcast_row Div) — 深拷贝 exp_shift
        auto output = clone_tensor(engine, *exp_shift);
        if (!output) return std::unexpected(output.error());
        r = engine.broadcast_row_inplace(*output, *row_sum, BinaryOp::Div);
        if (!r) return std::unexpected(r.error());

        output_cache_ = *output;
        return output;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // ep = out ⊙ grad_output
        auto ep = engine.elementwise_binary(BinaryOp::Mul, output_cache_, grad_output);
        if (!ep) return std::unexpected(ep.error());

        // dot[r] = Σ_c ep[r][c]
        auto dot = engine.row_reduce_sum(*ep);
        if (!dot) return std::unexpected(dot.error());

        // gmd = grad_output - dot (broadcast_row Sub) — 深拷贝 grad_output
        auto gmd = clone_tensor(engine, grad_output);
        if (!gmd) return std::unexpected(gmd.error());
        auto r = engine.broadcast_row_inplace(*gmd, *dot, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // grad_input = out ⊙ gmd
        return engine.elementwise_binary(BinaryOp::Mul, output_cache_, *gmd);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MultiHeadAttention — 多头注意力层（批量化：消除 per-head 和 per-sample 循环）
//
// 算法（只在此处，不在 Engine/Shader）：
//   Q = W_q × x, K = W_k × x, V = W_v × x  (三个 Linear 投影)
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//
//   批量化关键：用 rearrange_3d 把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
//   使 batched_matmul 能按 batch*H 切分行块，单次 dispatch 处理所有样本和所有头。
//
//   Q_re = rearrange_3d(Q, H*d_k, batch, seq) → (batch*H*d_k, seq)
//   S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
//   S *= scale
//   A = softmax(S)  — 行级归一化，堆叠布局下天然正确
//   O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
//   O = rearrange_3d(O_re, H*d_k, batch, seq, inverse=true) → (H*d_k, batch*seq)
//   out = W_o × O
//
// 输入形状: (d_model, batch * seq_len)，输出形状: (d_model, batch * seq_len)
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断
// ══════════════════════════════════════════════════════════════════════════
class MultiHeadAttention : public Layer
{
protected:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;   // 单样本序列长度（0 = 单样本，cols 即 seq）
    Scalar scale_;

    Linear w_q_;
    Linear w_k_;
    Linear w_v_;
    Linear w_o_;
    Softmax softmax_;

    // forward 缓存（rearranged 版本，供 backward 直接使用）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged
    Tensor attn_cache_;                    // (batch*H*seq, seq)

public:
    MultiHeadAttention(ComputeEngine& engine,
                       std::size_t d_model, std::size_t num_heads,
                       std::size_t seq_len = 0)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(engine, d_model, d_model),
          w_k_(engine, d_model, d_model),
          w_v_(engine, d_model, d_model),
          w_o_(engine, d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "MHA: d_model must be divisible by num_heads");
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = w_q_.parameters();
        auto k = w_k_.parameters();
        auto v = w_v_.parameters();
        auto o = w_o_.parameters();
        p.insert(p.end(), k.begin(), k.end());
        p.insert(p.end(), v.begin(), v.end());
        p.insert(p.end(), o.begin(), o.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = w_q_.param_gradients();
        auto k = w_k_.param_gradients();
        auto v = w_v_.param_gradients();
        auto o = w_o_.param_gradients();
        g.insert(g.end(), k.begin(), k.end());
        g.insert(g.end(), v.begin(), v.end());
        g.insert(g.end(), o.begin(), o.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"MHA forward: input shape mismatch"});

        const std::size_t total_seq = input.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            return std::unexpected(Error{"MHA forward: cols not divisible by seq_len"});

        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        //    使 batched_matmul 能按 batch*H 切分行块
        const std::size_t H_dk = num_heads_ * d_k_;
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false);
            if (!qr) return std::unexpected(qr.error());
            Q_cache_ = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false);
            if (!kr) return std::unexpected(kr.error());
            K_cache_ = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false);
            if (!vr) return std::unexpected(vr.error());
            V_cache_ = std::move(*vr);
        }
        else
        {
            // batch=1: rearrange 是恒等拷贝，跳过
            Q_cache_ = std::move(*q_res);
            K_cache_ = std::move(*k_res);
            V_cache_ = std::move(*v_res);
        }

        // 3. S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
        const std::size_t BH = batch * num_heads_;
        auto scores = engine.batched_matmul(
            Q_cache_, K_cache_, BH, true, false);
        if (!scores) return std::unexpected(scores.error());

        // 4. S *= scale
        auto r = engine.scale_inplace(*scores, scale_);
        if (!r) return std::unexpected(r.error());

        // 5. A = softmax(S)
        auto attn = softmax_.forward(engine, *scores);
        if (!attn) return std::unexpected(attn.error());
        attn_cache_ = *attn;

        // 6. O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
        auto concat_out = engine.batched_matmul(
            V_cache_, attn_cache_, BH, false, false);
        if (!concat_out) return std::unexpected(concat_out.error());

        // 7. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(*concat_out, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(*concat_out);
        }

        // 8. 输出投影
        return w_o_.forward(engine, concat);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // 推断 batch/seq（与 forward 一致）
        const std::size_t total_seq = grad_output.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        // 1. 输出投影反向 → grad_concat: (H*d_k, batch*seq)
        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;

        // 2. rearrange grad_concat → (batch*H*d_k, seq)
        Tensor grad_concat_re;
        if (batch > 1)
        {
            auto gcr = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!gcr) return std::unexpected(gcr.error());
            grad_concat_re = std::move(*gcr);
        }
        else
        {
            grad_concat_re = std::move(*gc);
        }

        // 3. grad_V_re = batched_matmul(grad_concat, A, BH, false, true)
        auto grad_V_re = engine.batched_matmul(
            grad_concat_re, attn_cache_, BH, false, true);
        if (!grad_V_re) return std::unexpected(grad_V_re.error());

        // 4. grad_A = batched_matmul(V, grad_concat, BH, true, false)
        auto grad_A = engine.batched_matmul(
            V_cache_, grad_concat_re, BH, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());

        // 5. grad_S = softmax.backward(grad_A)
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());

        // 6. grad_Q_re = batched_matmul(K, grad_S, BH, false, true) × scale
        auto grad_Q_re = engine.batched_matmul(
            K_cache_, *grad_S, BH, false, true);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        auto rq = engine.scale_inplace(*grad_Q_re, scale_);
        if (!rq) return std::unexpected(rq.error());

        // 7. grad_K_re = batched_matmul(Q, grad_S, BH, false, false) × scale
        auto grad_K_re = engine.batched_matmul(
            Q_cache_, *grad_S, BH, false, false);
        if (!grad_K_re) return std::unexpected(grad_K_re.error());
        auto rk = engine.scale_inplace(*grad_K_re, scale_);
        if (!rk) return std::unexpected(rk.error());

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            auto gq = engine.rearrange_3d(*grad_Q_re, H_dk, batch, seq, true);
            if (!gq) return std::unexpected(gq.error());
            grad_Q = std::move(*gq);
            auto gk = engine.rearrange_3d(*grad_K_re, H_dk, batch, seq, true);
            if (!gk) return std::unexpected(gk.error());
            grad_K = std::move(*gk);
            auto gv = engine.rearrange_3d(*grad_V_re, H_dk, batch, seq, true);
            if (!gv) return std::unexpected(gv.error());
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(*grad_Q_re);
            grad_K = std::move(*grad_K_re);
            grad_V = std::move(*grad_V_re);
        }

        // 9. 投影层反向 + 累加输入梯度
        auto giq = w_q_.backward(engine, grad_Q);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);

        auto gik = w_k_.backward(engine, grad_K);
        if (!gik) return gik;
        auto r1 = engine.add_inplace(grad_input, *gik);
        if (!r1) return std::unexpected(r1.error());

        auto giv = w_v_.backward(engine, grad_V);
        if (!giv) return giv;
        auto r2 = engine.add_inplace(grad_input, *giv);
        if (!r2) return std::unexpected(r2.error());

        return grad_input;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// PositionalEncoding — 正弦波固定位置编码（支持 batch tiling）
//
// 算法（只在此处，不在 Engine/Shader）：
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//   forward: out = input + PE   (elementwise Add)
//   backward: 梯度直接穿透（编码不可学习）
//
// Tiling 模式（tile_size > 0）：
//   当输入为 (d_model, batch * tile_size) 时，生成 (d_model, tile_size) 的
//   基础编码，沿列方向 tile 为 (d_model, batch * tile_size)。
//   每个 tile_size 列块为同一份编码，对应一个样本。
// ══════════════════════════════════════════════════════════════════════════
class PositionalEncoding final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t max_len_;
    std::size_t tile_size_;   // >0: 启用 tiling（每个样本的序列长度）
    Tensor encoding_cache_;
    std::size_t cached_total_ = 0;

    void rebuild_encoding(ComputeEngine& engine, std::size_t total_len)
    {
        const std::size_t base_len = (tile_size_ > 0) ? tile_size_ : total_len;
        const std::size_t batch    = (tile_size_ > 0) ? total_len / tile_size_ : 1;

        Matrix enc(d_model_, total_len);
        const std::size_t half = d_model_ / 2;
        std::vector<Scalar> freqs(half);
        for (std::size_t i = 0; i < half; ++i)
            freqs[i] = Scalar{1} / std::pow(Scalar{10000},
                static_cast<Scalar>(2 * i) / static_cast<Scalar>(d_model_));

        for (std::size_t b = 0; b < batch; ++b)
        {
            auto span = enc.span();
            const std::size_t col_off = b * base_len;
            for (std::size_t pos = 0; pos < base_len; ++pos)
            {
                const Scalar pos_d = static_cast<Scalar>(pos);
                for (std::size_t i = 0; i < half; ++i)
                {
                    const Scalar angle = pos_d * freqs[i];
                    span[(2 * i)     * total_len + col_off + pos] = std::sin(angle);
                    span[(2 * i + 1) * total_len + col_off + pos] = std::cos(angle);
                }
                if (d_model_ % 2 == 1)
                {
                    const Scalar freq_last = Scalar{1} / std::pow(Scalar{10000},
                        static_cast<Scalar>(2 * half) / static_cast<Scalar>(d_model_));
                    span[(d_model_ - 1) * total_len + col_off + pos] = std::sin(pos_d * freq_last);
                }
            }
        }
        auto r = engine.from_matrix(enc);
        NN_ASSERT(r, r ? "" : r.error().message.c_str());
        encoding_cache_ = std::move(*r);
        cached_total_ = total_len;
    }

public:
    PositionalEncoding(ComputeEngine& /*engine*/,
                       std::size_t d_model, std::size_t max_len = 5000,
                       std::size_t tile_size = 0)
        : d_model_(d_model), max_len_(max_len), tile_size_(tile_size)
    {
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"PE forward: d_model mismatch"});
        const std::size_t total_len = input.cols();
        if (tile_size_ > 0 && total_len % tile_size_ != 0)
            return std::unexpected(Error{"PE forward: total_len not divisible by tile_size"});
        if (tile_size_ == 0 && total_len > max_len_)
            return std::unexpected(Error{"PE forward: seq_len exceeds max_len"});

        if (cached_total_ != total_len)
            rebuild_encoding(engine, total_len);

        return engine.elementwise_binary(BinaryOp::Add, input, encoding_cache_);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& /*engine*/, const Tensor& grad_output) override
    {
        return grad_output;  // 位置编码不可学习，梯度直接穿透
    }
};

// ══════════════════════════════════════════════════════════════════════════
// FeedForward — 前馈网络 (FFN(x) = Linear₂(GeLU(Linear₁(x))))
// ══════════════════════════════════════════════════════════════════════════
class FeedForward final : public Layer
{
private:
    Linear fc1_;  // (d_model → d_ff)
    Linear fc2_;  // (d_ff → d_model)
    GeLU  gelu_;

public:
    FeedForward(ComputeEngine& engine,
                std::size_t d_model, std::size_t d_ff)
        : fc1_(engine, d_model, d_ff), fc2_(engine, d_ff, d_model) {}

    std::vector<TensorRef> parameters() override
    {
        auto p = fc1_.parameters();
        auto p2 = fc2_.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = fc1_.param_gradients();
        auto g2 = fc2_.param_gradients();
        g.insert(g.end(), g2.begin(), g2.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        auto h1 = fc1_.forward(engine, input);
        if (!h1) return h1;
        auto h2 = gelu_.forward(engine, *h1);
        if (!h2) return h2;
        return fc2_.forward(engine, *h2);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto b2 = fc2_.backward(engine, grad_output);
        if (!b2) return b2;
        auto bg = gelu_.backward(engine, *b2);
        if (!bg) return bg;
        return fc1_.backward(engine, *bg);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// TransformerEncoderLayer — Pre-Norm 编码器层（批量化）
//
// 算法（只在此处，不在 Engine/Shader）：
//   x = x + SelfAttn(LN₁(x))
//   x = x + FFN(LN₂(x))
//
// 输入/输出形状：(d_model, batch * seq_len)
// seq_len > 0 时启用 MHA 批量化路径（消除 per-head 和 per-sample 循环）。
// ══════════════════════════════════════════════════════════════════════════
class TransformerEncoderLayer final : public Layer
{
private:
    MultiHeadAttention self_attn_;
    LayerNorm norm1_;
    FeedForward ff_;
    LayerNorm norm2_;

    Tensor residual1_cache_;
    Tensor residual2_cache_;

public:
    TransformerEncoderLayer(ComputeEngine& engine,
                            std::size_t d_model, std::size_t num_heads,
                            std::size_t d_ff, std::size_t seq_len = 0)
        : self_attn_(engine, d_model, num_heads, seq_len),
          norm1_(engine, d_model),
          ff_(engine, d_model, d_ff),
          norm2_(engine, d_model) {}

    std::vector<TensorRef> parameters() override
    {
        auto p = self_attn_.parameters();
        auto n1 = norm1_.parameters();
        auto f  = ff_.parameters();
        auto n2 = norm2_.parameters();
        p.insert(p.end(), n1.begin(), n1.end());
        p.insert(p.end(), f.begin(), f.end());
        p.insert(p.end(), n2.begin(), n2.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = self_attn_.param_gradients();
        auto gn1 = norm1_.param_gradients();
        auto gf  = ff_.param_gradients();
        auto gn2 = norm2_.param_gradients();
        g.insert(g.end(), gn1.begin(), gn1.end());
        g.insert(g.end(), gf.begin(), gf.end());
        g.insert(g.end(), gn2.begin(), gn2.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        residual1_cache_ = input;

        // x1 = LN₁(input)
        auto n1 = norm1_.forward(engine, input);
        if (!n1) return n1;

        // a = SelfAttn(x1)
        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        // r2 = input + a
        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r2) return std::unexpected(r2.error());
        residual2_cache_ = *r2;

        // x2 = LN₂(r2)
        auto n2 = norm2_.forward(engine, residual2_cache_);
        if (!n2) return n2;

        // f = FFN(x2)
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        // out = r2 + f
        return engine.elementwise_binary(BinaryOp::Add, residual2_cache_, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // 残差2 反向: 分流到 residual1 + FFN
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_.backward(engine, *grad_ff);
        if (!b_n2) return b_n2;

        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());

        // 残差1 反向: 分流到 input + SelfAttn
        auto b_sa = self_attn_.backward(engine, *grad_r1);
        if (!b_sa) return b_sa;
        auto b_n1 = norm1_.backward(engine, *b_sa);
        if (!b_n1) return b_n1;

        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// TransformerEncoder — ViT 风格的 Transformer 编码器（全批量化）
//
// 算法（只在此处，不在 Engine/Shader）：
//   1. 输入 (d_model, batch * num_patches) — PatchEmbedding 输出，batch-major 列
//   2. + tiled PE → (d_model, batch * num_patches)
//   3. 一次性通过 N 个 EncoderLayer（MHA 内部用 rearrange_3d 批量化）
//   4. 全局平均池化（按样本聚合 num_patches 维度）→ (d_model, batch)
//
// 全 GPU 批量化策略：
//   - 整个 batch 同时通过所有层，无 per-sample 循环
//   - 池化用 rearrange_3d + row_reduce_sum + rearrange_3d，纯 GPU 原语
//   - 反向池化用 rearrange_3d + matmul(grad, ones_row_)，纯 GPU 原语
//   - 无 batch 边界 PCIe 传输（输入输出均为 GPU 张量流）
//
// 输入: (d_model, batch * num_patches)  — PatchEmbedding 输出
// 输出: (d_model, batch)                 — 池化后的序列表示
// ══════════════════════════════════════════════════════════════════════════
class TransformerEncoder final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_patches_;
    Scalar inv_num_patches_;
    std::vector<TransformerEncoderLayer> layers_;
    PositionalEncoding pos_encoding_;

    std::size_t batch_size_ = 0;
    Tensor ones_row_;  // (1, num_patches) 全1，用于 backward 池化梯度广播

public:
    TransformerEncoder(ComputeEngine& engine,
                       std::size_t d_model, std::size_t num_heads,
                       std::size_t d_ff, std::size_t num_layers,
                       std::size_t num_patches)
        : d_model_(d_model), num_patches_(num_patches),
          inv_num_patches_(Scalar{1} / static_cast<Scalar>(num_patches)),
          // PE 启用 tiling：每个样本独立使用 (d_model, num_patches) 的编码
          pos_encoding_(engine, d_model, num_patches, num_patches)
    {
        // 所有 EncoderLayer 传入 seq_len=num_patches，启用 MHA 批量化路径
        for (std::size_t i = 0; i < num_layers; ++i)
            layers_.emplace_back(engine, d_model, num_heads, d_ff, num_patches);

        // 预创建 ones_row_ (1, num_patches) 全1，用于 backward 广播
        Matrix ones_cpu(1, num_patches_, Scalar{1});
        auto or_t = engine.from_matrix(ones_cpu);
        NN_ASSERT(or_t, or_t ? "" : or_t.error().message.c_str());
        ones_row_ = std::move(*or_t);
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        for (auto& l : layers_)
        {
            auto lp = l.parameters();
            p.insert(p.end(), lp.begin(), lp.end());
        }
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        for (auto& l : layers_)
        {
            auto lg = l.param_gradients();
            g.insert(g.end(), lg.begin(), lg.end());
        }
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        // input: (d_model, batch * num_patches)
        if (input.rows() != d_model_)
            return std::unexpected(Error{"TransformerEncoder: row count mismatch"});
        if (input.cols() % num_patches_ != 0)
            return std::unexpected(Error{"TransformerEncoder: cols not divisible by num_patches"});
        batch_size_ = input.cols() / num_patches_;

        // 1. 添加 tiled 位置编码 → (d_model, batch * num_patches)
        auto pe = pos_encoding_.forward(engine, input);
        if (!pe) return pe;
        Tensor x = std::move(*pe);

        // 2. 一次性通过所有 EncoderLayer [全批量化 GPU]
        for (auto& layer : layers_)
        {
            auto lr = layer.forward(engine, x);
            if (!lr) return lr;
            x = std::move(*lr);
        }
        // x: (d_model, batch * num_patches)

        // 3. 全局平均池化（按样本聚合 num_patches 维度）→ (d_model, batch)
        //   使用 rearrange_3d 把 (d_model, batch*num_patches) 重排为
        //   (batch*d_model, num_patches)，每行块对应一个样本，
        //   row_reduce_sum 后得到 (batch*d_model, 1)，
        //   再 rearrange_3d inverse 回 (d_model, batch)。
        auto re = engine.rearrange_3d(x, d_model_, batch_size_, num_patches_, false);
        if (!re) return std::unexpected(re.error());
        auto row_sum = engine.row_reduce_sum(*re);
        if (!row_sum) return std::unexpected(row_sum.error());
        auto rs_re = engine.rearrange_3d(*row_sum, d_model_, batch_size_, 1, true);
        if (!rs_re) return std::unexpected(rs_re.error());
        auto r = engine.scale_inplace(*rs_re, inv_num_patches_);
        if (!r) return std::unexpected(r.error());
        return *rs_re;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // grad_output: (d_model, batch) — 来自下游 Linear.backward

        // 1. 反向池化：(d_model, batch) → (d_model, batch * num_patches)
        //   每个样本的池化梯度广播到 num_patches 列：
        //   grad_x[d, b*num_patches + p] = grad_out[d, b] * inv_num_patches
        //   实现：rearrange_3d(grad, d_model, batch, 1, false) → (batch*d_model, 1)
        //         scale(inv_n) → matmul(ones_row_, result_T) ... 直接：
        //         matmul(grad_col_vec, ones_row_) → (batch*d_model, num_patches)
        //         rearrange_3d(result, d_model, batch, num_patches, true) → (d_model, batch*num_patches)
        auto g_re = engine.rearrange_3d(grad_output, d_model_, batch_size_, 1, false);
        if (!g_re) return std::unexpected(g_re.error());
        auto r = engine.scale_inplace(*g_re, inv_num_patches_);
        if (!r) return std::unexpected(r.error());
        // (*g_re): (batch*d_model, 1) × ones_row_ (1, num_patches) → (batch*d_model, num_patches)
        auto unpooled = engine.matmul(*g_re, ones_row_, false, false);
        if (!unpooled) return std::unexpected(unpooled.error());
        auto grad = engine.rearrange_3d(*unpooled, d_model_, batch_size_, num_patches_, true);
        if (!grad) return std::unexpected(grad.error());
        Tensor grad_x = std::move(*grad);

        // 2. 反向通过所有 EncoderLayer [全批量化 GPU]
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        {
            auto br = it->backward(engine, grad_x);
            if (!br) return br;
            grad_x = std::move(*br);
        }

        // 3. 反向 PE（梯度直接穿透）
        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// PatchEmbedding — 图像 patch 嵌入
//
// 算法（只在此处，不在 Engine/Shader）：
//   1. 将 (img_size², batch) 展平图像提取 num_patches 个不重叠 patch
//   2. 每个展平 patch (patch_size²) 经 Linear 投影到 d_model 维
//   3. 重排为 (d_model, batch * num_patches) —— batch-major 列布局
//      [r, b * num_patches + p] = 样本 b 的 patch p 的第 r 维特征
//      这样下游 TransformerEncoder 可整体批量化（消除 per-sample 循环）。
//
// 注意: patch 提取涉及复杂重排，此处用 to_matrix/from_matrix 在 CPU 端
//       完成（batch 边界，PCIe 传输符合纯 GPU 架构约定）。
// ══════════════════════════════════════════════════════════════════════════
class PatchEmbedding final : public Layer
{
private:
    std::size_t img_size_;
    std::size_t patch_size_;
    std::size_t grid_size_;
    std::size_t num_patches_;
    std::size_t patch_dim_;
    std::size_t d_model_;
    Linear projection_;
    Tensor input_cache_;

public:
    PatchEmbedding(ComputeEngine& engine,
                   std::size_t img_size, std::size_t patch_size,
                   std::size_t d_model)
        : img_size_(img_size), patch_size_(patch_size),
          grid_size_(img_size / patch_size),
          num_patches_(grid_size_ * grid_size_),
          patch_dim_(patch_size * patch_size),
          d_model_(d_model),
          projection_(engine, patch_dim_, d_model)
    {
        NN_ASSERT(img_size % patch_size == 0,
                  "PatchEmbedding: img_size must be divisible by patch_size");
    }

    [[nodiscard]] std::size_t num_patches() const noexcept { return num_patches_; }
    [[nodiscard]] std::size_t d_model()     const noexcept { return d_model_; }

    std::vector<TensorRef> parameters() override
    { return projection_.parameters(); }

    std::vector<TensorRef> param_gradients() override
    { return projection_.param_gradients(); }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        input_cache_ = input;
        const std::size_t batch = input.cols();

        // Step 1: CPU 端提取 patches，直接生成 batch-major 列布局
        //   all_patches[flat, b * num_patches + p] = image_b[pixel(flat, p)]
        //   样本 b 占连续列块 [b*num_patches, (b+1)*num_patches)，
        //   便于下游 MHA 的 rearrange_3d 按 batch 切分列块。
        // （patch 提取是 PatchEmbedding 的算法职责，无对应 op-level 原语；
        //  此处为 batch 边界的合法 CPU 预处理，与 GPTModel 的 gather_rows 同性质）
        auto in_m = engine.to_matrix(input);
        if (!in_m) return std::unexpected(in_m.error());

        Matrix all_patches(patch_dim_, batch * num_patches_);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t p = 0; p < num_patches_; ++p)
            {
                const std::size_t gr = (p / grid_size_) * patch_size_;
                const std::size_t gc = (p % grid_size_) * patch_size_;
                const std::size_t col_idx = b * num_patches_ + p;  // batch-major
                for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    for (std::size_t pc = 0; pc < patch_size_; ++pc)
                    {
                        const std::size_t flat = pr * patch_size_ + pc;
                        const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                        all_patches.set_value_unchecked(flat, col_idx,
                            in_m->at_unchecked(pix, b));
                    }
            }
        }

        auto ap_t = engine.from_matrix(all_patches);
        if (!ap_t) return std::unexpected(ap_t.error());

        // Step 2: 投影 → (d_model, batch * num_patches) — 已是 batch-major，无需重排
        return projection_.forward(engine, *ap_t);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // grad_output: (d_model, batch * num_patches) — batch-major
        const std::size_t batch = grad_output.cols() / num_patches_;

        // Step 1: 投影层反向 → (patch_dim, batch * num_patches) — batch-major
        auto bp = projection_.backward(engine, grad_output);
        if (!bp) return bp;

        // Step 2: 散射梯度回输入 → (img_size², batch)
        auto gp_m = engine.to_matrix(*bp);
        if (!gp_m) return std::unexpected(gp_m.error());

        Matrix grad_input(img_size_ * img_size_, batch);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t p = 0; p < num_patches_; ++p)
            {
                const std::size_t gr = (p / grid_size_) * patch_size_;
                const std::size_t gc = (p % grid_size_) * patch_size_;
                const std::size_t col_idx = b * num_patches_ + p;  // batch-major
                for (std::size_t pr = 0; pr < patch_size_; ++pr)
                    for (std::size_t pc = 0; pc < patch_size_; ++pc)
                    {
                        const std::size_t flat = pr * patch_size_ + pc;
                        const std::size_t pix  = (gr + pr) * img_size_ + (gc + pc);
                        const Scalar val = grad_input.at_unchecked(pix, b)
                                         + gp_m->at_unchecked(flat, col_idx);
                        grad_input.set_value_unchecked(pix, b, val);
                    }
            }
        }
        return engine.from_matrix(grad_input);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// CausalSelfAttention — 因果自注意力（全批量化：消除 per-head 和 per-sample 循环）
//
// 算法（只在此处，不在 Engine/Shader）：
//   与 MultiHeadAttention 相同的批量化策略（rearrange_3d + batched_matmul），
//   但在 softmax 前施加上三角 -inf 因果掩码：mask[i][j] = 0 if j<=i, -inf if j>i
//
//   输入: (d_model, batch * seq_len) — batch-major 列布局
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//   Q_re = rearrange_3d(Q, H*d_k, batch, seq) → (batch*H*d_k, seq)
//   S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
//   S *= scale
//   S += mask (batch*H*seq, seq) — 因果掩码按 batch*H 平铺
//   A = softmax(S)
//   O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
//   O = rearrange_3d(O_re, H*d_k, batch, seq, inverse=true) → (H*d_k, batch*seq)
//   out = W_o × O
//
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断。
//   seq_len=0 表示单样本模式（cols 即 seq），保持向后兼容。
// ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;   // 单样本序列长度（0 = 单样本，cols 即 seq）
    Scalar scale_;

    Linear w_q_;
    Linear w_k_;
    Linear w_v_;
    Linear w_o_;
    Softmax softmax_;

    // forward 缓存（rearranged 版本，供 backward 直接使用）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged
    Tensor attn_cache_;                    // (batch*H*seq, seq)

    // 因果掩码（平铺为 batch*H*seq × seq 以匹配堆叠 scores 布局）
    Tensor mask_cache_;
    std::size_t mask_cached_key_ = 0;  // 编码 (batch << 16) | seq_len

    void ensure_mask(ComputeEngine& engine, std::size_t batch, std::size_t seq_len)
    {
        // 用 (batch, seq_len) 组合作为缓存键
        const std::size_t key = (batch << 16) | seq_len;
        if (mask_cached_key_ == key) return;

        const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
        const std::size_t BH = batch * num_heads_;
        // 平铺掩码: (BH * seq_len, seq_len)，每个 head 一份相同的 (seq, seq) 掩码
        Matrix mask(BH * seq_len, seq_len);
        for (std::size_t h = 0; h < BH; ++h)
            for (std::size_t i = 0; i < seq_len; ++i)
                for (std::size_t j = 0; j < seq_len; ++j)
                    mask.set_value_unchecked(h * seq_len + i, j,
                        (j <= i) ? Scalar{0} : neg_inf);
        auto r = engine.from_matrix(mask);
        NN_ASSERT(r, r ? "" : r.error().message.c_str());
        mask_cache_ = std::move(*r);
        mask_cached_key_ = key;
    }

public:
    CausalSelfAttention(ComputeEngine& engine,
                       std::size_t d_model, std::size_t num_heads,
                       std::size_t /*max_len*/ = 1024,
                       std::size_t seq_len = 0)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(engine, d_model, d_model),
          w_k_(engine, d_model, d_model),
          w_v_(engine, d_model, d_model),
          w_o_(engine, d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "CausalSelfAttention: d_model must be divisible by num_heads");
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = w_q_.parameters();
        auto k = w_k_.parameters();
        auto v = w_v_.parameters();
        auto o = w_o_.parameters();
        p.insert(p.end(), k.begin(), k.end());
        p.insert(p.end(), v.begin(), v.end());
        p.insert(p.end(), o.begin(), o.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = w_q_.param_gradients();
        auto k = w_k_.param_gradients();
        auto v = w_v_.param_gradients();
        auto o = w_o_.param_gradients();
        g.insert(g.end(), k.begin(), k.end());
        g.insert(g.end(), v.begin(), v.end());
        g.insert(g.end(), o.begin(), o.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"CausalSelfAttention forward: input shape mismatch"});

        const std::size_t total_seq = input.cols();
        const std::size_t seq   = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            return std::unexpected(Error{"CausalSelfAttention forward: cols not divisible by seq_len"});

        ensure_mask(engine, batch, seq);

        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        //    使 batched_matmul 能按 batch*H 切分行块
        const std::size_t H_dk = num_heads_ * d_k_;
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false);
            if (!qr) return std::unexpected(qr.error());
            Q_cache_ = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false);
            if (!kr) return std::unexpected(kr.error());
            K_cache_ = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false);
            if (!vr) return std::unexpected(vr.error());
            V_cache_ = std::move(*vr);
        }
        else
        {
            // batch=1: rearrange 是恒等拷贝，跳过
            Q_cache_ = std::move(*q_res);
            K_cache_ = std::move(*k_res);
            V_cache_ = std::move(*v_res);
        }

        // 3. S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
        const std::size_t BH = batch * num_heads_;
        auto scores = engine.batched_matmul(
            Q_cache_, K_cache_, BH, true, false);
        if (!scores) return std::unexpected(scores.error());

        // 4. S *= scale（掩码值为 0/-inf，缩放不影响）
        auto r = engine.scale_inplace(*scores, scale_);
        if (!r) return std::unexpected(r.error());

        // 5. 施加因果掩码 S += mask (batch*H*seq, seq)
        auto masked = engine.elementwise_binary(BinaryOp::Add, *scores, mask_cache_);
        if (!masked) return std::unexpected(masked.error());

        // 6. A = softmax(S_masked)
        auto attn = softmax_.forward(engine, *masked);
        if (!attn) return std::unexpected(attn.error());
        attn_cache_ = *attn;

        // 7. O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
        auto concat_out = engine.batched_matmul(
            V_cache_, attn_cache_, BH, false, false);
        if (!concat_out) return std::unexpected(concat_out.error());

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(*concat_out, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(*concat_out);
        }

        // 9. 输出投影
        return w_o_.forward(engine, concat);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // 推断 batch/seq（与 forward 一致）
        const std::size_t total_seq = grad_output.cols();
        const std::size_t seq   = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        // 1. 输出投影反向 → grad_concat: (H*d_k, batch*seq)
        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;

        // 2. rearrange grad_concat → (batch*H*d_k, seq)
        Tensor grad_concat_re;
        if (batch > 1)
        {
            auto gcr = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!gcr) return std::unexpected(gcr.error());
            grad_concat_re = std::move(*gcr);
        }
        else
        {
            grad_concat_re = std::move(*gc);
        }

        // 3. grad_V_re = batched_matmul(grad_concat, A, BH, false, true)
        auto grad_V_re = engine.batched_matmul(
            grad_concat_re, attn_cache_, BH, false, true);
        if (!grad_V_re) return std::unexpected(grad_V_re.error());

        // 4. grad_A = batched_matmul(V, grad_concat, BH, true, false)
        auto grad_A = engine.batched_matmul(
            V_cache_, grad_concat_re, BH, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());

        // 5. grad_S = softmax.backward(grad_A) — 掩码为常数，梯度直接穿透
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());

        // 6. grad_Q_re = batched_matmul(K, grad_S, BH, false, true) × scale
        auto grad_Q_re = engine.batched_matmul(
            K_cache_, *grad_S, BH, false, true);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        auto rq = engine.scale_inplace(*grad_Q_re, scale_);
        if (!rq) return std::unexpected(rq.error());

        // 7. grad_K_re = batched_matmul(Q, grad_S, BH, false, false) × scale
        auto grad_K_re = engine.batched_matmul(
            Q_cache_, *grad_S, BH, false, false);
        if (!grad_K_re) return std::unexpected(grad_K_re.error());
        auto rk = engine.scale_inplace(*grad_K_re, scale_);
        if (!rk) return std::unexpected(rk.error());

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            auto gq = engine.rearrange_3d(*grad_Q_re, H_dk, batch, seq, true);
            if (!gq) return std::unexpected(gq.error());
            grad_Q = std::move(*gq);
            auto gk = engine.rearrange_3d(*grad_K_re, H_dk, batch, seq, true);
            if (!gk) return std::unexpected(gk.error());
            grad_K = std::move(*gk);
            auto gv = engine.rearrange_3d(*grad_V_re, H_dk, batch, seq, true);
            if (!gv) return std::unexpected(gv.error());
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(*grad_Q_re);
            grad_K = std::move(*grad_K_re);
            grad_V = std::move(*grad_V_re);
        }

        // 9. 投影层反向 + 累加输入梯度
        auto giq = w_q_.backward(engine, grad_Q);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);

        auto gik = w_k_.backward(engine, grad_K);
        if (!gik) return gik;
        auto r1 = engine.add_inplace(grad_input, *gik);
        if (!r1) return std::unexpected(r1.error());

        auto giv = w_v_.backward(engine, grad_V);
        if (!giv) return giv;
        auto r2 = engine.add_inplace(grad_input, *giv);
        if (!r2) return std::unexpected(r2.error());

        return grad_input;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ALiBiCausalSelfAttention — 带线性偏置的因果自注意力
//
// ALiBi (Attention with Linear Biases) 原理：
//   不使用位置嵌入，而是在注意力分数上添加线性偏置：
//   attention_score = Q*K^T + bias
//   其中 bias[i][j] = -m_h * (i - j) for j <= i
//   斜率 m_h = 2^(-8h/H)，h 是头索引，H 是总头数
//
// 优点：
//   1. 无需位置嵌入，减少参数
//   2. 天然支持长度外推（训练短序列，推理长序列）
//   3. 计算开销极小（仅添加预计算的偏置）
//
// 算法（只在此处，不在 Engine/Shader）：
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//   Q_re = rearrange_3d(Q, H*d_k, batch, seq) → (batch*H*d_k, seq)
//   S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
//   S *= scale
//   S += alibi_bias (batch*H*seq, seq) — ALiBi 偏置按 batch*H 平铺
//   A = softmax(S)
//   O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
//   O = rearrange_3d(O_re, H*d_k, batch, seq, inverse=true) → (H*d_k, batch*seq)
//   out = W_o × O
// ══════════════════════════════════════════════════════════════════════════
class ALiBiCausalSelfAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;
    Scalar scale_;

    Linear w_q_;
    Linear w_k_;
    Linear w_v_;
    Linear w_o_;
    Softmax softmax_;

    // forward 缓存
    Tensor Q_cache_, K_cache_, V_cache_;
    Tensor attn_cache_;

    // ALiBi 偏置缓存（合并因果掩码 + 线性偏置）
    Tensor alibi_bias_cache_;
    std::size_t alibi_bias_cached_key_ = 0;

    // ALiBi 斜率：m_h = 2^(-8h/H)
    std::vector<Scalar> slopes_;

    void init_slopes()
    {
        slopes_.resize(num_heads_);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            // m_h = 2^(-8h/H)
            slopes_[h] = std::pow(Scalar{2}, -Scalar{8} * h / num_heads_);
        }
    }

    void ensure_alibi_bias(ComputeEngine& engine, std::size_t batch, std::size_t seq_len)
    {
        const std::size_t key = (batch << 16) | seq_len;
        if (alibi_bias_cached_key_ == key) return;

        const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
        const std::size_t BH = batch * num_heads_;

        // ALiBi 偏置: (BH * seq_len, seq_len)
        // 对于每个头 h，偏置矩阵为：
        //   bias[i][j] = -m_h * (i - j) if j <= i
        //   bias[i][j] = -inf if j > i (因果掩码)
        Matrix bias(BH * seq_len, seq_len);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t h = 0; h < num_heads_; ++h)
            {
                const std::size_t head_idx = b * num_heads_ + h;
                const Scalar slope = slopes_[h];

                for (std::size_t i = 0; i < seq_len; ++i)
                {
                    for (std::size_t j = 0; j < seq_len; ++j)
                    {
                        if (j <= i)
                        {
                            // ALiBi 线性偏置：-m_h * (i - j)
                            bias.set_value_unchecked(
                                head_idx * seq_len + i, j,
                                -slope * static_cast<Scalar>(i - j));
                        }
                        else
                        {
                            // 因果掩码：未来位置为 -inf
                            bias.set_value_unchecked(
                                head_idx * seq_len + i, j,
                                neg_inf);
                        }
                    }
                }
            }
        }

        auto r = engine.from_matrix(bias);
        NN_ASSERT(r, r ? "" : r.error().message.c_str());
        alibi_bias_cache_ = std::move(*r);
        alibi_bias_cached_key_ = key;
    }

public:
    ALiBiCausalSelfAttention(ComputeEngine& engine,
                             std::size_t d_model, std::size_t num_heads,
                             std::size_t /*max_len*/ = 1024,
                             std::size_t seq_len = 0)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(engine, d_model, d_model),
          w_k_(engine, d_model, d_model),
          w_v_(engine, d_model, d_model),
          w_o_(engine, d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ALiBiCausalSelfAttention: d_model must be divisible by num_heads");
        init_slopes();
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = w_q_.parameters();
        auto k = w_k_.parameters();
        auto v = w_v_.parameters();
        auto o = w_o_.parameters();
        p.insert(p.end(), k.begin(), k.end());
        p.insert(p.end(), v.begin(), v.end());
        p.insert(p.end(), o.begin(), o.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = w_q_.param_gradients();
        auto k = w_k_.param_gradients();
        auto v = w_v_.param_gradients();
        auto o = w_o_.param_gradients();
        g.insert(g.end(), k.begin(), k.end());
        g.insert(g.end(), v.begin(), v.end());
        g.insert(g.end(), o.begin(), o.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"ALiBiCausalSelfAttention forward: input shape mismatch"});

        const std::size_t total_seq = input.cols();
        const std::size_t seq   = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            return std::unexpected(Error{"ALiBiCausalSelfAttention forward: cols not divisible by seq_len"});

        ensure_alibi_bias(engine, batch, seq);

        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        const std::size_t H_dk = num_heads_ * d_k_;
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false);
            if (!qr) return std::unexpected(qr.error());
            Q_cache_ = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false);
            if (!kr) return std::unexpected(kr.error());
            K_cache_ = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false);
            if (!vr) return std::unexpected(vr.error());
            V_cache_ = std::move(*vr);
        }
        else
        {
            Q_cache_ = std::move(*q_res);
            K_cache_ = std::move(*k_res);
            V_cache_ = std::move(*v_res);
        }

        // 3. S = batched_matmul(Q_re, K_re, batch*H, transA=true) → (batch*H*seq, seq)
        const std::size_t BH = batch * num_heads_;
        auto scores = engine.batched_matmul(
            Q_cache_, K_cache_, BH, true, false);
        if (!scores) return std::unexpected(scores.error());

        // 4. S *= scale
        auto r = engine.scale_inplace(*scores, scale_);
        if (!r) return std::unexpected(r.error());

        // 5. 施加 ALiBi 偏置 S += alibi_bias (batch*H*seq, seq)
        auto biased = engine.elementwise_binary(BinaryOp::Add, *scores, alibi_bias_cache_);
        if (!biased) return std::unexpected(biased.error());

        // 6. A = softmax(S_biased)
        auto attn = softmax_.forward(engine, *biased);
        if (!attn) return std::unexpected(attn.error());
        attn_cache_ = *attn;

        // 7. O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
        auto concat_out = engine.batched_matmul(
            V_cache_, attn_cache_, BH, false, false);
        if (!concat_out) return std::unexpected(concat_out.error());

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(*concat_out, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(*concat_out);
        }

        // 9. 输出投影
        return w_o_.forward(engine, concat);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // 推断 batch/seq（与 forward 一致）
        const std::size_t total_seq = grad_output.cols();
        const std::size_t seq   = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        // 1. 输出投影反向 → grad_concat: (H*d_k, batch*seq)
        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;

        // 2. rearrange grad_concat → (batch*H*d_k, seq)
        Tensor grad_concat_re;
        if (batch > 1)
        {
            auto gcr = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!gcr) return std::unexpected(gcr.error());
            grad_concat_re = std::move(*gcr);
        }
        else
        {
            grad_concat_re = std::move(*gc);
        }

        // 3. grad_V_re = batched_matmul(grad_concat, A, BH, false, true)
        auto grad_V_re = engine.batched_matmul(
            grad_concat_re, attn_cache_, BH, false, true);
        if (!grad_V_re) return std::unexpected(grad_V_re.error());

        // 4. grad_A = batched_matmul(V, grad_concat, BH, true, false)
        auto grad_A = engine.batched_matmul(
            V_cache_, grad_concat_re, BH, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());

        // 5. grad_S = softmax.backward(grad_A) — ALiBi 偏置为常数，梯度直接穿透
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());

        // 6. grad_Q_re = batched_matmul(K, grad_S, BH, false, true) × scale
        auto grad_Q_re = engine.batched_matmul(
            K_cache_, *grad_S, BH, false, true);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        auto rq = engine.scale_inplace(*grad_Q_re, scale_);
        if (!rq) return std::unexpected(rq.error());

        // 7. grad_K_re = batched_matmul(Q, grad_S, BH, false, false) × scale
        auto grad_K_re = engine.batched_matmul(
            Q_cache_, *grad_S, BH, false, false);
        if (!grad_K_re) return std::unexpected(grad_K_re.error());
        auto rk = engine.scale_inplace(*grad_K_re, scale_);
        if (!rk) return std::unexpected(rk.error());

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            auto gq = engine.rearrange_3d(*grad_Q_re, H_dk, batch, seq, true);
            if (!gq) return std::unexpected(gq.error());
            grad_Q = std::move(*gq);
            auto gk = engine.rearrange_3d(*grad_K_re, H_dk, batch, seq, true);
            if (!gk) return std::unexpected(gk.error());
            grad_K = std::move(*gk);
            auto gv = engine.rearrange_3d(*grad_V_re, H_dk, batch, seq, true);
            if (!gv) return std::unexpected(gv.error());
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(*grad_Q_re);
            grad_K = std::move(*grad_K_re);
            grad_V = std::move(*grad_V_re);
        }

        // 9. 投影层反向 + 累加输入梯度
        auto giq = w_q_.backward(engine, grad_Q);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);

        auto gik = w_k_.backward(engine, grad_K);
        if (!gik) return gik;
        auto r1 = engine.add_inplace(grad_input, *gik);
        if (!r1) return std::unexpected(r1.error());

        auto giv = w_v_.backward(engine, grad_V);
        if (!giv) return giv;
        auto r2 = engine.add_inplace(grad_input, *giv);
        if (!r2) return std::unexpected(r2.error());

        return grad_input;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GPTBlock — Pre-Norm 解码器块
//
// 算法（只在此处，不在 Engine/Shader）：
//   x = x + CausalSelfAttn(LN₁(x))
//   x = x + FFN(LN₂(x))
// ══════════════════════════════════════════════════════════════════════════
class GPTBlock final : public Layer
{
private:
    CausalSelfAttention self_attn_;
    LayerNorm norm1_;
    FeedForward ff_;
    LayerNorm norm2_;

    Tensor residual1_cache_;
    Tensor residual2_cache_;

public:
    GPTBlock(ComputeEngine& engine,
             std::size_t d_model, std::size_t num_heads,
             std::size_t d_ff, std::size_t max_len = 1024,
             std::size_t seq_len = 0)
        : self_attn_(engine, d_model, num_heads, max_len, seq_len),
          norm1_(engine, d_model),
          ff_(engine, d_model, d_ff),
          norm2_(engine, d_model) {}

    std::vector<TensorRef> parameters() override
    {
        auto p = self_attn_.parameters();
        auto n1 = norm1_.parameters();
        auto f  = ff_.parameters();
        auto n2 = norm2_.parameters();
        p.insert(p.end(), n1.begin(), n1.end());
        p.insert(p.end(), f.begin(), f.end());
        p.insert(p.end(), n2.begin(), n2.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = self_attn_.param_gradients();
        auto gn1 = norm1_.param_gradients();
        auto gf  = ff_.param_gradients();
        auto gn2 = norm2_.param_gradients();
        g.insert(g.end(), gn1.begin(), gn1.end());
        g.insert(g.end(), gf.begin(), gf.end());
        g.insert(g.end(), gn2.begin(), gn2.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        residual1_cache_ = input;

        auto n1 = norm1_.forward(engine, input);
        if (!n1) return n1;

        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r2) return std::unexpected(r2.error());
        residual2_cache_ = *r2;

        auto n2 = norm2_.forward(engine, residual2_cache_);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return engine.elementwise_binary(BinaryOp::Add, residual2_cache_, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_.backward(engine, *grad_ff);
        if (!b_n2) return b_n2;

        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());

        auto b_sa = self_attn_.backward(engine, *grad_r1);
        if (!b_sa) return b_sa;
        auto b_n1 = norm1_.backward(engine, *b_sa);
        if (!b_n1) return b_n1;

        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ALiBiGPTBlock — 使用 ALiBi 的 Pre-Norm 解码器块
//
// 算法（只在此处，不在 Engine/Shader）：
//   x = x + ALiBiCausalSelfAttn(LN₁(x))
//   x = x + FFN(LN₂(x))
// ══════════════════════════════════════════════════════════════════════════
class ALiBiGPTBlock final : public Layer
{
private:
    ALiBiCausalSelfAttention self_attn_;
    LayerNorm norm1_;
    FeedForward ff_;
    LayerNorm norm2_;

    Tensor residual1_cache_;
    Tensor residual2_cache_;

public:
    ALiBiGPTBlock(ComputeEngine& engine,
                  std::size_t d_model, std::size_t num_heads,
                  std::size_t d_ff, std::size_t max_len = 1024,
                  std::size_t seq_len = 0)
        : self_attn_(engine, d_model, num_heads, max_len, seq_len),
          norm1_(engine, d_model),
          ff_(engine, d_model, d_ff),
          norm2_(engine, d_model) {}

    std::vector<TensorRef> parameters() override
    {
        auto p = self_attn_.parameters();
        auto n1 = norm1_.parameters();
        auto f  = ff_.parameters();
        auto n2 = norm2_.parameters();
        p.insert(p.end(), n1.begin(), n1.end());
        p.insert(p.end(), f.begin(), f.end());
        p.insert(p.end(), n2.begin(), n2.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = self_attn_.param_gradients();
        auto gn1 = norm1_.param_gradients();
        auto gf  = ff_.param_gradients();
        auto gn2 = norm2_.param_gradients();
        g.insert(g.end(), gn1.begin(), gn1.end());
        g.insert(g.end(), gf.begin(), gf.end());
        g.insert(g.end(), gn2.begin(), gn2.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        residual1_cache_ = input;

        auto n1 = norm1_.forward(engine, input);
        if (!n1) return n1;

        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r2) return std::unexpected(r2.error());
        residual2_cache_ = *r2;

        auto n2 = norm2_.forward(engine, residual2_cache_);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return engine.elementwise_binary(BinaryOp::Add, residual2_cache_, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_.backward(engine, *grad_ff);
        if (!b_n2) return b_n2;

        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());

        auto b_sa = self_attn_.backward(engine, *grad_r1);
        if (!b_sa) return b_sa;
        auto b_n1 = norm1_.backward(engine, *b_sa);
        if (!b_n1) return b_n1;

        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GPTModel — Decoder-only Transformer 语言模型
//
// 算法（只在此处，不在 Engine/Shader）：
//   组件: TokenEmb + PosEmb + N × GPTBlock + LayerNorm + LM Head
//   输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
//   输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
//
// 注意: token embedding 查表 + 位置 embedding 相加涉及按 token ID 的
//       稀疏写入，此处用 to_matrix/from_matrix 在 CPU 端完成
//       （batch 边界，PCIe 传输符合纯 GPU 架构约定）。
// ══════════════════════════════════════════════════════════════════════════
class GPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;
    bool pos_emb_learnable_;  // true=learned, false=sinusoidal(冻结)

    // 嵌入
    Tensor token_emb_;       // (vocab_size, d_model)
    Tensor grad_token_emb_;
    Tensor pos_emb_;         // (seq_len, d_model)
    Tensor grad_pos_emb_;

    std::vector<GPTBlock> blocks_;
    LayerNorm ln_f_;
    Linear lm_head_;

    // 反向缓存
    Tensor stored_x_;                      // forward 输入 (d_model, seq*batch)
    std::vector<std::size_t> stored_tokens_flat_;  // 所有 token IDs (seq*batch)
    Tensor stored_tokens_tensor_;          // token IDs 的 Tensor 版本 (total, 1)
    std::size_t batch_size_ = 0;

    // pos_indices 缓存（避免每 step 重建）
    Tensor pos_indices_cache_;             // (total, 1) 值为 [0,0,..,1,1,..,...,seq-1,..]
    std::size_t pos_indices_key_ = 0;      // 缓存键: (batch_size << 16) | seq_len

    // batch 录制粒度：每隔 flush_interval_ 个 Transformer block 提交一次
    // 0 = 不在 block 间 flush（默认），>0 = 每 N 个 block flush 一次
    std::size_t flush_interval_ = 0;

public:
    GPTModel(ComputeEngine& engine,
             std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
             std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
             PosEncodingType pos_enc_type = PosEncodingType::Learned)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          pos_emb_learnable_(pos_enc_type == PosEncodingType::Learned),
          ln_f_(engine, d_model),
          lm_head_(engine, d_model, vocab_size)
    {
        // 初始化 token_emb_ 和 pos_emb_
        Matrix te(vocab_size, d_model);
        Matrix pe(seq_len, d_model);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);

        // 位置编码初始化
        if (pos_enc_type == PosEncodingType::Sinusoidal)
        {
            // 正弦波固定位置编码: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
            auto pe_s = pe.span();
            for (std::size_t pos = 0; pos < seq_len; ++pos)
                for (std::size_t i = 0; i < d_model; ++i)
                {
                    Scalar angle = static_cast<Scalar>(pos) /
                        std::pow(Scalar{10000}, static_cast<Scalar>(2 * (i / 2)) / static_cast<Scalar>(d_model));
                    pe_s[pos * d_model + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
                }
        }
        else
        {
            // 可学习位置编码: N(0, 0.02) 随机初始化
            auto pe_s = pe.span();
            for (std::size_t i = 0; i < pe.size(); ++i) pe_s[i] = dist(rng);
        }

        auto te_r = engine.from_matrix(te);
        NN_ASSERT(te_r, te_r ? "" : te_r.error().message.c_str());
        token_emb_ = std::move(*te_r);
        auto pe_r = engine.from_matrix(pe);
        NN_ASSERT(pe_r, pe_r ? "" : pe_r.error().message.c_str());
        pos_emb_ = std::move(*pe_r);

        grad_token_emb_ = engine.create_tensor(vocab_size, d_model);
        grad_pos_emb_ = engine.create_tensor(seq_len, d_model);
        { auto r1 = engine.zero(grad_token_emb_); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }
        { auto r2 = engine.zero(grad_pos_emb_);    NN_ASSERT(r2, r2 ? "" : r2.error().message.c_str()); }

        // GPTBlock 传入 seq_len，启用 CausalSelfAttention 批量化路径
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(engine, d_model, num_heads, d_ff, seq_len, seq_len);
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        if (pos_emb_learnable_)
            p.push_back(pos_emb_);
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        if (pos_emb_learnable_)
            g.push_back(grad_pos_emb_);
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_.param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq_len = input.rows();
        batch_size_ = input.cols();
        const std::size_t total = seq_len * batch_size_;

        // ── 1. gather 所有 token 的 embedding ──
        // input: (seq, batch) 标量 token IDs（row-major: i = t * batch + b）
        // gather_rows(token_emb_, input) → (seq*batch, d_model)
        //   row i = token_emb[input[i]], i 是 input 的 row-major 索引
        auto all_emb = engine.gather_rows(token_emb_, input);
        if (!all_emb) return std::unexpected(all_emb.error());

        // ── 2. 保存 token IDs 的 Tensor 拷贝（供 backward 的 scatter_add_rows） ──
        // 全程 GPU：clone 在 GPU 内执行，无 PCIe 传输
        auto st_t = engine.clone(input);
        if (!st_t) return std::unexpected(st_t.error());
        stored_tokens_tensor_ = std::move(*st_t);

        // ── 2.5 确保 pos_indices 缓存有效 ──
        {
            const std::size_t key = (batch_size_ << 16) | seq_len;
            if (pos_indices_key_ != key)
            {
                Matrix pidx_m(total, 1);
                for (std::size_t t = 0; t < seq_len; ++t)
                    for (std::size_t b = 0; b < batch_size_; ++b)
                        pidx_m.set_value_unchecked(t * batch_size_ + b, 0,
                            static_cast<Scalar>(t));
                auto pidx_t = engine.from_matrix(pidx_m);
                if (!pidx_t) return std::unexpected(pidx_t.error());
                pos_indices_cache_ = std::move(*pidx_t);
                pos_indices_key_ = key;
            }
        }

        // ── 3. 构造 x: (d_model, seq*batch) 用原语组合 ──
        //   x = transpose(gather(token_emb, input)) + transpose(gather(pos_emb, pos_idx))
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        auto pos_gathered = engine.gather_rows(pos_emb_, pos_indices_cache_);
        if (!pos_gathered) return std::unexpected(pos_gathered.error());

        auto pos_T = engine.transpose(*pos_gathered);
        if (!pos_T) return std::unexpected(pos_T.error());

        auto x_result = engine.elementwise_binary(BinaryOp::Add, *all_T, *pos_T);
        if (!x_result) return std::unexpected(x_result.error());
        stored_x_ = *x_result;  // 缓存供 backward（GPTBlock 内部也有自己的缓存）

        // ── 4. 通过 Transformer 块（全批量化，无 per-sample 循环） ──
        Tensor x = std::move(*x_result);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            auto r = blocks_[bi].forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
            // 按间隔 flush，将大录制拆分为多个小提交（防 TDR）
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // ── 5. 最终 LayerNorm ──
        auto ln = ln_f_.forward(engine, x);
        if (!ln) return ln;
        x = std::move(*ln);

        // ── 6. LM Head → (vocab_size, seq*batch) batch-major ──
        return lm_head_.forward(engine, x);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;

        (void)engine.zero(grad_token_emb_);
        if (pos_emb_learnable_)
            (void)engine.zero(grad_pos_emb_);

        // ── 1. LM Head 反向 → (d_model, seq*batch) ──
        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        Tensor grad_x = std::move(*b_lm);

        // ── 2. LayerNorm 反向 ──
        auto b_ln = ln_f_.backward(engine, grad_x);
        if (!b_ln) return b_ln;
        grad_x = std::move(*b_ln);

        // ── 3. 逐块反向（全批量化） ──
        {
            const std::size_t n = blocks_.size();
            for (std::size_t bi = 0; bi < n; ++bi)
            {
                auto br = blocks_[n - 1 - bi].backward(engine, grad_x);
                if (!br) return br;
                grad_x = std::move(*br);
                if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
                {
                    auto fr = engine.flush_batch();
                    if (!fr) return std::unexpected(fr.error());
                }
            }
        }

        // ── 4. 转置 grad_x + pos_grad GPU 计算 ──
        //   grad_x: (d_model, seq*batch)
        //   grad_T = transpose(grad_x) → (total, d_model) — 用于 scatter_add_rows
        //   pos_grad = rearrange_3d(grad_x) + reduce → 全 GPU，无 PCIe 传输
        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());

        if (pos_emb_learnable_)
        {
            // rearrange_3d(grad_x, d_model, seq, batch) → (seq*d_model, batch)
            // 然后 row_reduce_sum → (seq*d_model, 1) = pos_grad 扁平化
            auto rearranged = engine.rearrange_3d(
                grad_x, d_model_, seq_len, batch_size_, false);
            if (!rearranged) return std::unexpected(rearranged.error());

            auto reduced = engine.row_reduce_sum(*rearranged);
            if (!reduced) return std::unexpected(reduced.error());

            // reduced 是 (seq*d_model, 1)，数据布局为 pos_grad[t*d_model+d]
            // reshape 为 (seq, d_model)
            Tensor pos_grad_t;
            if (reduced->is_gpu())
            {
                // GPU 路径：零拷贝 reshape（共享底层 buffer）
                auto reshaped_gt = reduced->gpu_tensor().with_shape(seq_len, d_model_);
                pos_grad_t = Tensor::from_gpu(std::move(reshaped_gt));
            }
            else
            {
                // CPU 路径：数据布局已正确，重新解释形状
                const auto& src = reduced->cpu_matrix();
                Matrix dst(seq_len, d_model_, Scalar{0});
                std::memcpy(dst.span().data(), src.span().data(),
                            seq_len * d_model_ * sizeof(Scalar));
                pos_grad_t = Tensor::from_matrix(std::move(dst));
            }

            auto ar = engine.add_inplace(grad_pos_emb_, pos_grad_t);
            if (!ar) return std::unexpected(ar.error());
        }

        // ── 5. scatter_add_rows: grad_token_emb_[tokens] += grad_T ──
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        // grad_input: token IDs 无梯度，返回零张量（仅用于接口一致性）
        Matrix grad_input(seq_len, batch_size_, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── batch 录制粒度控制 ──
    void set_flush_interval(std::size_t interval) { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 采样生成（支持温度采样 + 贪心） ────────────────────────────
    // 注意: 此方法需要在 CPU 端采样，因此 forward 后会下载 logits。
    // min_new_tokens: 前这么多个 token 内不检查 EOS（避免立刻停止）
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
              const std::vector<std::size_t>& prompt,
              std::size_t max_new_tokens,
              Scalar temperature = 1.0,
              std::size_t eos_token_id = static_cast<std::size_t>(-1),
              std::size_t min_new_tokens = 0)
    {
        std::vector<std::size_t> context(prompt);
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            std::size_t start = 0;
            if (context.size() > seq_len_)
                start = context.size() - seq_len_;

            const std::size_t cur_len = context.size() - start;
            // 始终 pad 到 seq_len_，确保 CausalSelfAttention 中 total_seq % seq_len_ == 0
            Matrix input(seq_len_, 1);
            input.zero();
            for (std::size_t t = 0; t < cur_len; ++t)
                input.set_value_unchecked(t, 0,
                    static_cast<Scalar>(context[start + t]));

            auto in_t = engine.from_matrix(input);
            if (!in_t) return std::unexpected(in_t.error());
            auto logits_res = forward(engine, *in_t);
            if (!logits_res) return std::unexpected(logits_res.error());

            auto logits_m = engine.to_matrix(*logits_res);
            if (!logits_m) return std::unexpected(logits_m.error());

            // 取最后一个位置的 logits
            std::vector<Scalar> last_logits(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last_logits[v] = logits_m->at_unchecked(v, cur_len - 1);

            // temperature
            if (temperature > 0.0 && temperature != 1.0)
                for (auto& v : last_logits) v /= temperature;

            // softmax（数值稳定）
            Scalar max_val = last_logits[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, last_logits[v]);
            Scalar sum_exp = 0.0;
            for (auto& v : last_logits)
            {
                v = std::exp(v - max_val);
                sum_exp += v;
            }
            for (auto& v : last_logits) v /= sum_exp;

            // 采样
            // temperature > 0: 随机采样（temperature 仅影响缩放，1.0 = 不缩放但仍然采样）
            // temperature == 0: 贪心解码（argmax）
            std::size_t next_token;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cumulative = 0.0;
                next_token = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cumulative += last_logits[v];
                    if (r <= cumulative) { next_token = v; break; }
                }
            }
            else
            {
                next_token = 0;
                Scalar best = last_logits[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (last_logits[v] > best) { best = last_logits[v]; next_token = v; }
            }

            context.push_back(next_token);

            // 遇到 EOS 停止生成（不将 EOS 加入输出）
            // 但前 min_new_tokens 个 token 内不检查，避免模型一上来就输出 EOS
            if (step >= min_new_tokens && next_token == eos_token_id)
                break;

            generated.push_back(next_token);
        }
        return generated;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ALiBiGPTModel — 使用 ALiBi 的 Decoder-only Transformer 语言模型
//
// 算法（只在此处，不在 Engine/Shader）：
//   组件: TokenEmb + N × ALiBiGPTBlock + LayerNorm + LM Head
//   输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
//   输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
//
// 与 GPTModel 的区别：
//   1. 不使用位置嵌入，通过 ALiBi 在注意力分数上添加线性偏置
//   2. 支持长度外推（训练短序列，推理长序列）
//   3. 参数更少（无 pos_emb_）
// ══════════════════════════════════════════════════════════════════════════
class ALiBiGPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t seq_len_;

    // 可学习嵌入（仅 token，无位置嵌入）
    Tensor token_emb_;       // (vocab_size, d_model)
    Tensor grad_token_emb_;

    std::vector<ALiBiGPTBlock> blocks_;
    LayerNorm ln_f_;
    Linear lm_head_;

    // 反向缓存
    Tensor stored_x_;                      // forward 输入 (d_model, seq*batch)
    Tensor stored_tokens_tensor_;          // token IDs 的 Tensor 版本 (GPU clone)
    std::size_t batch_size_ = 0;

    // batch 录制粒度：每隔 flush_interval_ 个 Transformer block 提交一次
    std::size_t flush_interval_ = 0;

public:
    ALiBiGPTModel(ComputeEngine& engine,
                  std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
                  std::size_t num_heads, std::size_t d_ff, std::size_t num_layers)
        : vocab_size_(vocab_size), seq_len_(seq_len),
          ln_f_(engine, d_model),
          lm_head_(engine, d_model, vocab_size)
    {
        // 初始化 token_emb_（无位置嵌入）
        Matrix te(vocab_size, d_model);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);

        auto te_r = engine.from_matrix(te);
        NN_ASSERT(te_r, te_r ? "" : te_r.error().message.c_str());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size, d_model);
        { auto r1 = engine.zero(grad_token_emb_); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }

        // ALiBiGPTBlock 传入 seq_len，启用批量化路径
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(engine, d_model, num_heads, d_ff, seq_len, seq_len);
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_.param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        batch_size_ = input.cols();

        // ── 1. gather 所有 token 的 embedding ──
        // input: (seq, batch) 标量 token IDs（row-major: i = t * batch + b）
        // gather_rows(token_emb_, input) → (seq*batch, d_model)
        auto all_emb = engine.gather_rows(token_emb_, input);
        if (!all_emb) return std::unexpected(all_emb.error());

        // ── 2. 保存 token IDs 的 Tensor 拷贝（供 backward 的 scatter_add_rows） ──
        // 全程 GPU：clone 在 GPU 内执行，无 PCIe 传输
        auto st_t = engine.clone(input);
        if (!st_t) return std::unexpected(st_t.error());
        stored_tokens_tensor_ = std::move(*st_t);

        // ── 3. 构造 x: (d_model, seq*batch) ──
        //   ALiBi 不需要位置嵌入，直接使用 token embedding
        auto x_result = engine.transpose(*all_emb);
        if (!x_result) return std::unexpected(x_result.error());
        stored_x_ = *x_result;

        // ── 4. 通过 Transformer 块（全批量化，无 per-sample 循环） ──
        Tensor x = std::move(*x_result);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            auto r = blocks_[bi].forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // ── 5. 最终 LayerNorm ──
        auto ln = ln_f_.forward(engine, x);
        if (!ln) return ln;
        x = std::move(*ln);

        // ── 6. LM Head → (vocab_size, seq*batch) batch-major ──
        return lm_head_.forward(engine, x);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;

        (void)engine.zero(grad_token_emb_);

        // ── 1. LM Head 反向 → (d_model, seq*batch) ──
        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        Tensor grad_x = std::move(*b_lm);

        // ── 2. LayerNorm 反向 ──
        auto b_ln = ln_f_.backward(engine, grad_x);
        if (!b_ln) return b_ln;
        grad_x = std::move(*b_ln);

        // ── 3. 逐块反向（全批量化） ──
        {
            const std::size_t n = blocks_.size();
            for (std::size_t bi = 0; bi < n; ++bi)
            {
                auto br = blocks_[n - 1 - bi].backward(engine, grad_x);
                if (!br) return br;
                grad_x = std::move(*br);
                if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
                {
                    auto fr = engine.flush_batch();
                    if (!fr) return std::unexpected(fr.error());
                }
            }
        }

        // ── 4. 转置 grad_x + CPU 辅助 ──
        //   grad_x: (d_model, seq*batch)
        //   grad_T = transpose(grad_x) → (total, d_model) — 用于 scatter_add_rows
        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());

        // ── 5. scatter_add_rows: grad_token_emb_[tokens] += grad_T ──
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        // grad_input: token IDs 无梯度，返回零张量（仅用于接口一致性）
        Matrix grad_input(seq_len, batch_size_, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── batch 录制粒度控制 ──
    void set_flush_interval(std::size_t interval) { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 采样生成（支持温度采样 + 贪心） ────────────────────────────
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
              const std::vector<std::size_t>& prompt,
              std::size_t max_new_tokens,
              Scalar temperature = 1.0,
              std::size_t eos_token_id = static_cast<std::size_t>(-1),
              std::size_t min_new_tokens = 0)
    {
        std::vector<std::size_t> context(prompt);
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            std::size_t start = 0;
            if (context.size() > seq_len_)
                start = context.size() - seq_len_;

            const std::size_t cur_len = context.size() - start;
            Matrix input(seq_len_, 1);
            input.zero();
            for (std::size_t t = 0; t < cur_len; ++t)
                input.set_value_unchecked(t, 0,
                    static_cast<Scalar>(context[start + t]));

            auto in_t = engine.from_matrix(input);
            if (!in_t) return std::unexpected(in_t.error());
            auto logits_res = forward(engine, *in_t);
            if (!logits_res) return std::unexpected(logits_res.error());

            auto logits_m = engine.to_matrix(*logits_res);
            if (!logits_m) return std::unexpected(logits_m.error());

            // 取最后一个位置的 logits
            std::vector<Scalar> last_logits(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last_logits[v] = logits_m->at_unchecked(v, cur_len - 1);

            // temperature
            if (temperature > 0.0 && temperature != 1.0)
                for (auto& v : last_logits) v /= temperature;

            // softmax（数值稳定）
            Scalar max_val = last_logits[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, last_logits[v]);
            Scalar sum_exp = 0.0;
            for (auto& v : last_logits)
            {
                v = std::exp(v - max_val);
                sum_exp += v;
            }
            for (auto& v : last_logits) v /= sum_exp;

            // 采样
            // temperature > 0: 随机采样（temperature 仅影响缩放，1.0 = 不缩放但仍然采样）
            // temperature == 0: 贪心解码（argmax）
            std::size_t next_token;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cumulative = 0.0;
                next_token = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cumulative += last_logits[v];
                    if (r <= cumulative) { next_token = v; break; }
                }
            }
            else
            {
                next_token = 0;
                Scalar best = last_logits[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (last_logits[v] > best) { best = last_logits[v]; next_token = v; }
            }

            context.push_back(next_token);

            if (step >= min_new_tokens && next_token == eos_token_id)
                break;

            generated.push_back(next_token);
        }
        return generated;
    }
};

} // namespace nn

#endif // NN_COMPUTE_LAYER_HPP
