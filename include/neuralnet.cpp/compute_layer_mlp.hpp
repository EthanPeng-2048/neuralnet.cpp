#ifndef NN_COMPUTE_LAYER_MLP_HPP
#define NN_COMPUTE_LAYER_MLP_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{
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
    std::size_t in_features_;
    std::size_t out_features_;
    Tensor w_;           // 权重 (out_features, in_features)
    Tensor b_;           // 偏置 (out_features, 1)
    Tensor grad_w_;      // 权重梯度
    Tensor grad_b_;      // 偏置梯度
    Tensor input_cache_; // forward 输入缓存（供 backward 使用）

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

public:
    Linear(std::size_t in_features, std::size_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // ── 在 CPU 上初始化权重（Xavier 均匀分布） ──
        Matrix w_cpu(out_features_, in_features_);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features_ + out_features_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_features_, 1);  // 零初始化

        // ── 通过 engine 上传到目标设备 ──
        auto w_res = engine.from_matrix(w_cpu);
        if (!w_res) return std::unexpected(w_res.error());
        w_ = std::move(*w_res);

        auto b_res = engine.from_matrix(b_cpu);
        if (!b_res) return std::unexpected(b_res.error());
        b_ = std::move(*b_res);

        // ── 创建梯度张量（零初始化） ──
        grad_w_ = engine.create_tensor(out_features_, in_features_);
        grad_b_ = engine.create_tensor(out_features_, 1);
        { auto r1 = engine.zero(grad_w_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_b_); if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {w_, b_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_w_, grad_b_};
    }

    void clear_cache() override { input_cache_ = Tensor{}; }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward: out = W × x + b ──────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != w_.cols())
            return std::unexpected(Error{"linear forward: input shape mismatch"});

        // 缓存输入供 backward 使用（checkpoint 模式下不保留，由重计算重建）
        if (!checkpoint_mode_)
            input_cache_ = input;

        // out = W × x
        auto out = engine.matmul(w_, input, false, false);
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

    void clear_cache() override { input_cache_ = Tensor{}; }

    // ── forward: out = max(x, 0) ──────────────────────────────────────────
    // ReLU 算法由 Layer 表达为 Max 原语 + 标量 0
    // Engine/Shader 只提供 Max 原语，不知道 "ReLU" 是什么
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (!checkpoint_mode_)
            input_cache_ = input;
        return dsl::compute(engine,
            dsl::max(dsl::leaf(input), Scalar{0}),
            input.rows(), input.cols());
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

        return dsl::compute(engine,
            dsl::select(dsl::leaf(input_cache_) > Scalar{0},
                        dsl::leaf(grad_output), Scalar{0}),
            grad_output.rows(), grad_output.cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GeLU — QuickGeLU 激活函数（单表达式 DSL 融合）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = x * sigmoid(β * x) = x / (1 + exp(-β·x))
//   backward: grad_x = grad_out * s * (1 + βx * (1 - s)),  s = sigmoid(βx)
//
// 用 dsl::compute 单表达式融合：sigmoid(βx) = 1/(1+exp(-βx)) 全部折叠为单个
// GPU 融合 kernel（仅 input/output 落显存，无中间 Tensor）。backward 用
// input_cache_ 重算 sigmoid（不缓存，省显存）。AOT 收集由 scan_exprs dry-run
// 本层完成（GPU 闭合世界两端一致）。
// ══════════════════════════════════════════════════════════════════════════
class GeLU final : public Layer
{
private:
    static constexpr Scalar BETA = 1.702f;
    Tensor input_cache_;

public:
    GeLU() = default;

    void clear_cache() override
    {
        input_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward: out = x * sigmoid(β * x) ────────────────────────────────
    // 单表达式 DSL 融合（M 融合）：sigmoid(βx) = 1/(1+exp(-βx))，全部折叠为
    // 单个 GPU 融合 kernel（仅 input/output 落显存，无中间 Tensor）。表达式
    // 文本只写在本 Layer；AOT 收集由 scan_exprs dry-run 本方法完成。
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (!checkpoint_mode_)
            input_cache_ = input;
        const Scalar beta = BETA;
        // out = x * sigmoid(βx) = x / (1 + exp(-β·x))
        return dsl::compute(engine,
            dsl::leaf(input) / (Scalar{1} + dsl::exp(-(dsl::leaf(input) * beta))),
            input.rows(), input.cols());
    }

    // ── backward: grad_x = grad_out * factor ─────────────────────────────
    // factor = s * (1 + βx * (1 - s)),  s = sigmoid(βx) = 1/(1+exp(-βx))
    // 单表达式 DSL 融合：sigmoid 用 input_cache_ 重算（不缓存，省显存），
    // βx 子表达式由 DSL 的 CSE 复用，消除中间 Tensor。
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            return std::unexpected(Error{"gelu backward: shape mismatch"});

        const Scalar beta = BETA;
        // s = 1/(1+exp(-βx))；factor = s * (1 + βx*(1-s))；out = grad_out * factor
        auto bx = dsl::leaf(input_cache_) * beta;                    // βx
        auto s  = Scalar{1} / (Scalar{1} + dsl::exp(-bx));          // sigmoid(βx)
        auto factor = s * (Scalar{1} + bx * (Scalar{1} - s));       // 1+βx(1-s)
        return dsl::compute(engine,
            dsl::leaf(grad_output) * factor,
            grad_output.rows(), grad_output.cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SwiGLU — Swish-Gated Linear Unit（LLaMA/Mistral 风格 FFN 激活）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  输入 (2*d_ff, batch)，前 d_ff 行为 gate，后 d_ff 行为 up
//             gate = slice_rows(x, 0, d_ff)
//             up   = slice_rows(x, d_ff, d_ff)
//             sw = SiLU(gate) = gate * sigmoid(gate)
//             out = sw ⊙ up                       → (d_ff, batch)
//   backward: grad_up   = grad_out ⊙ sw
//             grad_sw   = grad_out ⊙ up
//             grad_gate = grad_sw * (σ(g) + g*σ(g)*(1-σ(g)))
//             grad_input 把 grad_gate 写回行 [0, d_ff)、grad_up 写回 [d_ff, 2*d_ff)
//
// 原语分解（Engine/Shader 只知道标量原语）：
//   SiLU forward:  Neg → Exp → Add(1) → Div(1/x) → Mul(x*s)
//   SiLU backward: Sub → Mul → Mul → Add → Mul
//   split/merge 用 slice_rows / insert_rows
// ══════════════════════════════════════════════════════════════════════════
class SwiGLU final : public Layer
{
private:
    std::size_t d_ff_ = 0;
    Tensor gate_cache_;    // (d_ff, batch)
    Tensor sigmoid_cache_; // σ(gate) (d_ff, batch)
    Tensor up_cache_;      // (d_ff, batch)

public:
    SwiGLU() = default;
    explicit SwiGLU(std::size_t d_ff) : d_ff_(d_ff) {}

    void clear_cache() override
    {
        gate_cache_ = Tensor{};
        sigmoid_cache_ = Tensor{};
        up_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (gate_cache_.valid()) r.emplace_back(gate_cache_);
        if (sigmoid_cache_.valid()) r.emplace_back(sigmoid_cache_);
        if (up_cache_.valid()) r.emplace_back(up_cache_);
        return r;
    }

    // ── forward: out = SiLU(gate) ⊙ up ────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        // gate = 前 d_ff 行，up = 后 d_ff 行
        auto gate = engine.slice_rows(input, 0, d_ff_);
        if (!gate) return std::unexpected(gate.error());
        auto up = engine.slice_rows(input, d_ff_, d_ff_);
        if (!up) return std::unexpected(up.error());

        // s = sigmoid(gate) = 1 / (1 + exp(-gate))
        auto s = dsl::compute(engine,
            Scalar{1} / (Scalar{1} + dsl::exp(-dsl::leaf(*gate))),
            gate->rows(), gate->cols());
        if (!s) return std::unexpected(s.error());

        if (!checkpoint_mode_)
        {
            gate_cache_ = *gate;
            sigmoid_cache_ = *s;
            up_cache_ = *up;
        }

        // out = SiLU(gate) ⊙ up = gate * sigmoid(gate) * up
        return dsl::compute(engine,
            dsl::leaf(*gate) * dsl::leaf(*s) * dsl::leaf(*up),
            gate->rows(), gate->cols());
    }

    // ── backward: 用 eval_expr（表达式 DSL）融合逐元素计算 ──────────────────
    //
    // 数学：
    //   sw   = gate ⊙ sigmoid
    //   grad_up   = grad_out ⊙ sw = grad_out ⊙ gate ⊙ sigmoid
    //   factor    = sigmoid ⊙ (1 + gate ⊙ (1 − sigmoid))
    //             = sigmoid + sigmoid ⊙ gate ⊙ (1 − sigmoid)
    //   grad_gate = grad_out ⊙ up ⊙ factor
    //
    // 两条 eval_expr（CPU 一次遍历融合、无中间 Tensor）：
    //   expr1: grad_gate（6 regs, 6 instrs）
    //     inputs=[grad_out, s, gate, up], consts=[1.0]
    //     r0 = 1 − s          (Sub, cst(0), input(1))
    //     r1 = gate * (1−s)   (Mul, input(2), fanout(0))
    //     r2 = 1 + r1         (Add, cst(0), fanout(1))
    //     r3 = s * r2         (Mul, input(1), fanout(2))                  // factor
    //     r4 = up * r3        (Mul, input(3), fanout(3))
    //     r5 = grad * r4      (Mul, input(0), fanout(4))                  // grad_gate
    //
    //   expr2: grad_up（2 regs, 2 instrs）
    //     inputs=[grad_out, sigmoid, gate]
    //     i0: r0 = grad * gate    (Mul, input(0), input(2))
    //     i1: r1 = r0 * sigmoid   (Mul, fanout(0), input(1))               // grad_up
    //
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t rows = d_ff_;
        const std::size_t cols = grad_output.cols();

        // ── 统一表达式 DSL（内联数学式，融合；CPU 单次遍历 + SIMD）──
        //   grad_gate = grad_out ⊙ up ⊙ s ⊙ (1 + gate ⊙ (1 − s))
        //   grad_up   = grad_out ⊙ gate ⊙ s
        const nn::Scalar one{1};
        auto grad_gate = dsl::compute(engine,
            dsl::leaf(grad_output) * dsl::leaf(up_cache_)
              * (dsl::leaf(sigmoid_cache_)
                 * (one + dsl::leaf(gate_cache_) * (one - dsl::leaf(sigmoid_cache_)))),
            rows, cols);
        if (!grad_gate) return std::unexpected(grad_gate.error());
        auto grad_up = dsl::compute(engine,
            dsl::leaf(grad_output) * dsl::leaf(gate_cache_) * dsl::leaf(sigmoid_cache_),
            rows, cols);
        if (!grad_up) return std::unexpected(grad_up.error());

        // 合并：grad_input = (grad_gate; grad_up) → (2*d_ff, batch)
        auto grad_input = engine.create_tensor(2 * d_ff_, cols);
        auto rz = engine.zero(grad_input);
        if (!rz) return std::unexpected(rz.error());
        auto ri0 = engine.insert_rows(grad_input, 0, *grad_gate);
        if (!ri0) return std::unexpected(ri0.error());
        auto ri1 = engine.insert_rows(grad_input, d_ff_, *grad_up);
        if (!ri1) return std::unexpected(ri1.error());

        return grad_input;
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
    explicit LayerNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // gamma 初始化为 1, beta 初始化为 0
        Matrix gamma_cpu(normalized_shape_, 1, Scalar{1});
        Matrix beta_cpu(normalized_shape_, 1, Scalar{0});

        auto g = engine.from_matrix(gamma_cpu);
        if (!g) return std::unexpected(g.error());
        gamma_ = std::move(*g);

        auto bv = engine.from_matrix(beta_cpu);
        if (!bv) return std::unexpected(bv.error());
        beta_ = std::move(*bv);

        grad_gamma_ = engine.create_tensor(normalized_shape_, 1);
        grad_beta_ = engine.create_tensor(normalized_shape_, 1);
        { auto r1 = engine.zero(grad_gamma_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_beta_);  if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {gamma_, beta_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_gamma_, grad_beta_};
    }

    void clear_cache() override
    {
        normalized_cache_ = Tensor{};
        std_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (normalized_cache_.valid()) r.emplace_back(normalized_cache_);
        if (std_cache_.valid()) r.emplace_back(std_cache_);
        return r;
    }

    // ── forward ───────────────────────────────────────────────────────────
    // M3 融合（算法公式不变，diff_sq (F,B) 由归约 kernel 内部消解）：
    //   融合表达式保持 F 无关结构（不含 1/F、ε 常量）；形状相关标量在
    //   (1,B) 小向量上用引擎原语施加：
    //   1. mean_raw = col_reduce_sum(x)                    → (1,B) 归约向量输出
    //   2. mean     = mean_raw*(1/F)
    //   3. diff     = x - mean (col 广播)                  → (F,B) 融合逐元素
    //   4. var_raw  = col_reduce_sum(diff²)                → (1,B) 归约向量输出
    //   5. std_inv  = rsqrt(var_raw*(1/F) + ε)             → (1,B) 原语
    //   6. normalized=diff * std_inv (col 广播)            → (F,B) 融合逐元素
    //   7. out      = normalized*gamma + beta (row 广播)   → (F,B) 融合逐元素
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != normalized_shape_)
            return std::unexpected(Error{"layernorm forward: input shape mismatch"});

        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. mean_raw = col_reduce_sum(x) → (1,B)
        auto mean_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input)), F, B);
        if (!mean_raw) return std::unexpected(mean_raw.error());

        // 2. mean = mean_raw*(1/F) → (1,B)
        // （1/F 形状相关标量在 (1,B) 小向量上用原语施加，保持融合表达式 F 无关）
        auto mean = engine.elementwise_binary_scalar(BinaryOp::Mul, *mean_raw, inv_features);
        if (!mean) return std::unexpected(mean.error());

        // 3. diff = x - mean (col 广播) → (F,B)
        auto diff = dsl::compute(engine,
            dsl::leaf(input) - dsl::col_broadcast(*mean), F, B);
        if (!diff) return std::unexpected(diff.error());

        // 4. var_raw = col_reduce_sum(diff²) → (1,B)
        auto var_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(*diff) * dsl::leaf(*diff)), F, B);
        if (!var_raw) return std::unexpected(var_raw.error());

        // 5. std_inv = rsqrt(var_raw*(1/F) + ε) → (1,B)
        // （1/F、ε 形状相关标量在 (1,B) 小向量上用原语施加，保持融合表达式 F 无关）
        auto var = engine.elementwise_binary_scalar(BinaryOp::Mul, *var_raw, inv_features);
        if (!var) return std::unexpected(var.error());
        auto var_eps = engine.elementwise_binary_scalar(BinaryOp::Add, *var, epsilon_);
        if (!var_eps) return std::unexpected(var_eps.error());
        auto std_inv = engine.elementwise_unary(UnaryOp::Rsqrt, *var_eps);
        if (!std_inv) return std::unexpected(std_inv.error());
        Tensor std_inv_t = std::move(*std_inv);
        if (!checkpoint_mode_)
            std_cache_ = std_inv_t;

        // 6. normalized = diff * std_inv (col 广播) → (F,B)
        auto normalized = dsl::compute(engine,
            dsl::leaf(*diff) * dsl::col_broadcast(std_inv_t), F, B);
        if (!normalized) return std::unexpected(normalized.error());
        Tensor normalized_t = std::move(*normalized);
        if (!checkpoint_mode_)
            normalized_cache_ = normalized_t;

        // 7. out = normalized*gamma + beta (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
            + dsl::row_broadcast(beta_),
            F, B);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
    //   gy = grad_out * gamma
    //   mean_g  = col_reduce_sum(gy) * invF
    //   mean_gn = col_reduce_sum(gy * normalized) * invF
    // grad_gamma += row_sum(gy ⊙ normalized)
    // grad_beta  += row_sum(grad_out)
    // M3 融合（融合表达式 F 无关，1/F 在 (1,B) 上用原语施加）：
    //   mean_g/mean_gn 为列归约向量输出；(F,B) 全尺寸中间量由融合 kernel 消解。
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. mean_g_raw = col_reduce_sum(gy) → (1,B)
        auto mg_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)),
            F, B);
        if (!mg_raw) return std::unexpected(mg_raw.error());
        auto mean_g = engine.elementwise_binary_scalar(BinaryOp::Mul, *mg_raw, inv_features);
        if (!mean_g) return std::unexpected(mean_g.error());

        // 2. mean_gn_raw = col_reduce_sum(gy ⊙ normalized) → (1,B)
        auto mgn_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normalized_cache_)),
            F, B);
        if (!mgn_raw) return std::unexpected(mgn_raw.error());
        auto mean_gn = engine.elementwise_binary_scalar(BinaryOp::Mul, *mgn_raw, inv_features);
        if (!mean_gn) return std::unexpected(mean_gn.error());

        // 3. grad_x = (gy - mean_g - normalized*mean_gn) * std_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*mean_g)
             - dsl::leaf(normalized_cache_) * dsl::col_broadcast(*mean_gn))
            * dsl::col_broadcast(std_cache_),
            F, B);
        if (!grad_x) return std::unexpected(grad_x.error());

        // 4. grad_gamma += row_reduce_sum(gy ⊙ normalized) → (F,1)
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normalized_cache_)),
            F, B);
        if (!gg) return std::unexpected(gg.error());
        auto r = engine.add_inplace(grad_gamma_, *gg);
        if (!r) return std::unexpected(r.error());

        // 5. grad_beta += row_reduce_sum(grad_out) → (F,1)
        auto gb = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)), F, B);
        if (!gb) return std::unexpected(gb.error());
        r = engine.add_inplace(grad_beta_, *gb);
        if (!r) return std::unexpected(r.error());

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RMSNorm — Root Mean Square 归一化（LLaMA/Mistral 风格）
//
// 与 LayerNorm 的差异：不减去均值、无 beta 偏置，只按均方根归一化。
// 每层少 2 次列归约 + 1 次广播，计算量更小，训练更稳。
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  mean_sq = (1/F) * Σ_f x²             (elementwise Mul + col_reduce_sum + scale)
//             rms_inv = 1 / sqrt(mean_sq + eps)    (binary_scalar Add + unary Rsqrt)
//             normed  = x * rms_inv                (broadcast_col Mul)
//             out     = normed * gamma             (broadcast_row Mul)
//   backward: gy       = grad_out * gamma          (broadcast_row Mul)
//             gy_norm  = gy ⊙ normed               (elementwise Mul)
//             m        = (1/F) * Σ_f gy_norm       (col_reduce_sum + scale)
//             grad_x   = (gy - m*normed) * rms_inv (broadcast_col Mul + Sub + Mul)
//             grad_gamma += row_reduce_sum(gy_norm) (row_reduce_sum)
// ══════════════════════════════════════════════════════════════════════════
class RMSNorm final : public Layer
{
private:
    std::size_t normalized_shape_;
    Scalar epsilon_;

    Tensor gamma_;        // (normalized_shape, 1)
    Tensor grad_gamma_;

    // backward 缓存
    Tensor normed_cache_;   // (features, batch)
    Tensor rms_inv_cache_;  // (1, batch)

    static constexpr Scalar EPSILON = 1e-5;

public:
    explicit RMSNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // gamma 初始化为 1（无 beta）
        Matrix gamma_cpu(normalized_shape_, 1, Scalar{1});
        auto g = engine.from_matrix(gamma_cpu);
        if (!g) return std::unexpected(g.error());
        gamma_ = std::move(*g);

        grad_gamma_ = engine.create_tensor(normalized_shape_, 1);
        { auto r1 = engine.zero(grad_gamma_); if (!r1) return std::unexpected(r1.error()); }
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {gamma_};
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_gamma_};
    }

    void clear_cache() override
    {
        normed_cache_ = Tensor{};
        rms_inv_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (normed_cache_.valid()) r.emplace_back(normed_cache_);
        if (rms_inv_cache_.valid()) r.emplace_back(rms_inv_cache_);
        return r;
    }

    // ── forward ───────────────────────────────────────────────────────────
    // M3 融合（算法公式不变，中间 x_sq (F,B) 由归约 kernel 内部消解）：
    //   融合表达式保持 F 无关结构（不含 1/F、ε 常量，避免闭合世界 key 随
    //   归一化维度漂移）；形状相关标量在 (1,B) 小向量上用引擎原语施加：
    //   1. s_raw  = col_reduce_sum(x*x)                    → (1,B) 归约向量输出
    //   2. rms_inv= rsqrt(s_raw*(1/F) + ε)                 → (1,B) 原语
    //   3. normed = x * rms_inv (col 广播)                 → (F,B) 融合逐元素
    //   4. out    = normed * gamma (row 广播)              → (F,B) 融合逐元素
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != normalized_shape_)
            return std::unexpected(Error{"rmsnorm forward: input shape mismatch"});

        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. s_raw = col_reduce_sum(x*x) → (1,B)
        auto s_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input) * dsl::leaf(input)), F, B);
        if (!s_raw) return std::unexpected(s_raw.error());

        // 2. rms_inv = rsqrt(s_raw*(1/F) + ε) → (1,B)
        // （1/F、ε 形状相关标量在 (1,B) 小向量上用原语施加，保持融合表达式 F 无关）
        auto scaled = engine.elementwise_binary_scalar(BinaryOp::Mul, *s_raw, inv_features);
        if (!scaled) return std::unexpected(scaled.error());
        auto var_eps = engine.elementwise_binary_scalar(BinaryOp::Add, *scaled, epsilon_);
        if (!var_eps) return std::unexpected(var_eps.error());
        auto rms_inv = engine.elementwise_unary(UnaryOp::Rsqrt, *var_eps);
        if (!rms_inv) return std::unexpected(rms_inv.error());
        Tensor rms_inv_t = std::move(*rms_inv);
        if (!checkpoint_mode_)
            rms_inv_cache_ = rms_inv_t;

        // 3. normed = x * rms_inv (col 广播) → (F,B)
        auto normed = dsl::compute(engine,
            dsl::leaf(input) * dsl::col_broadcast(rms_inv_t), F, B);
        if (!normed) return std::unexpected(normed.error());
        Tensor normed_t = std::move(*normed);
        if (!checkpoint_mode_)
            normed_cache_ = normed_t;

        // 4. out = normed * gamma (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normed_t) * dsl::row_broadcast(gamma_), F, B);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // M3 融合（融合表达式 F 无关，1/F 在 (1,B) 小向量上用原语施加）：
    //   gy       = grad * gamma
    //   m_raw    = col_reduce_sum(gy ⊙ normed)             → (1,B) 归约向量输出
    //   m        = m_raw * (1/F)
    //   grad_x   = (gy - m*normed) * rms_inv               → (F,B) 融合逐元素
    //   grad_gamma += row_reduce_sum(gy ⊙ normed)          → (F,1) 归约向量输出
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. m_raw = col_reduce_sum(gy ⊙ normed) → (1,B)
        auto m_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normed_cache_)),
            F, B);
        if (!m_raw) return std::unexpected(m_raw.error());
        auto m = engine.elementwise_binary_scalar(BinaryOp::Mul, *m_raw, inv_features);
        if (!m) return std::unexpected(m.error());

        // 2. grad_x = (gy - m*normed) * rms_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*m) * dsl::leaf(normed_cache_))
            * dsl::col_broadcast(rms_inv_cache_),
            F, B);
        if (!grad_x) return std::unexpected(grad_x.error());

        // 3. grad_gamma += row_reduce_sum(gy ⊙ normed) → (F,1)
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normed_cache_)),
            F, B);
        if (!gg) return std::unexpected(gg.error());
        auto r = engine.add_inplace(grad_gamma_, *gg);
        if (!r) return std::unexpected(r.error());

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// FusedChainLayer — IR-C 演示：begin_expr/end_expr 多表达式链式融合
//
// forward 用 begin_expr/end_expr 把"无法写进一行的多步逐元素变换"写成多个
// 表达式（t、u 为中间量，不逃逸——融合的语义约束）：
//     t   = x * 2                    (Linear 输入 + 常量)
//     u   = t + 3                    (消费 t：Linear 引用 → 内联为寄存器)
//     out = u * gamma                (row 广播参数 (F,1))
// 引擎在 end_expr 时做融合分析（IR-C，expr_graph.hpp）：三段拼接成单个
// kernel，中间结果 t/u 留在寄存器（不落显存、不额外 dispatch）。
//   - GPU：begin_expr 录制 → end_expr 融合 → AOT 融合 shader 匹配 dispatch
//     （闭合世界：scan_exprs 必须 dry-run 本 Layer 登记融合后的复合 spec）。
//   - CPU：begin/end 为 no-op，各表达式直接求值（参考实现）。
//   - scan：NN_EXPR_SCAN 下 end_expr 登记融合后的复合 spec（两端一致）。
// ══════════════════════════════════════════════════════════════════════════
class FusedChainLayer final : public Layer
{
private:
    std::size_t features_;
    Tensor gamma_;   // (F, 1) 按行广播系数

public:
    FusedChainLayer(std::size_t features) : features_(features) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix g(features_, 1, Scalar{0.5f});  // 演示用固定系数
        auto g_t = engine.from_matrix(g);
        if (!g_t) return std::unexpected(g_t.error());
        gamma_ = std::move(*g_t);
        return {};
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t F = input.rows();
        const std::size_t B = input.cols();

        auto b = engine.begin_expr();
        if (!b) return std::unexpected(b.error());
        auto t = dsl::compute(engine, dsl::leaf(input) * Scalar{2}, F, B);
        if (!t) return std::unexpected(t.error());
        auto u = dsl::compute(engine, dsl::leaf(*t) + Scalar{3}, F, B);
        if (!u) return std::unexpected(u.error());
        auto out = dsl::compute(engine,
            dsl::leaf(*u) * dsl::row_broadcast(gamma_), F, B);
        if (!out) return std::unexpected(out.error());
        auto e = engine.end_expr();
        if (!e) return std::unexpected(e.error());
        return out;
    }

    // out = ((x*2)+3)*gamma → dout/dx = 2*gamma。
    // 非 IR-C 演示路径：单表达式直接求值（scan 单独登记，GPU 独立 AOT）。
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t F = grad_output.rows();
        const std::size_t B = grad_output.cols();
        return dsl::compute(engine,
            dsl::leaf(grad_output) * dsl::row_broadcast(gamma_) * Scalar{2},
            F, B);
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override { return {}; }
    [[nodiscard]] std::vector<TensorRef> param_gradients() override { return {}; }
};

// ── 归一化层工厂：按 NormType 创建 LayerNorm 或 RMSNorm ──────────────────
[[nodiscard]] inline std::unique_ptr<Layer> make_norm_layer(
    std::size_t d_model, NormType norm_type)
{
    if (norm_type == NormType::RMSNorm)
        return std::make_unique<RMSNorm>(d_model);
    return std::make_unique<LayerNorm>(d_model);
}

// ══════════════════════════════════════════════════════════════════════════
// Conv2D — 二维卷积层（引擎化：im2col + matmul 复用矩阵乘内核）
//
// 布局约定（与项目 batch-major 列布局一致）：
//   输入  Tensor: (C_in * H_in * W_in, batch)
//   输出  Tensor: (C_out * H_out * W_out, batch)
//   权重  w_: (C_out, C_in * k * k)
//   偏置  b_: (C_out, 1)
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  im2col(x) → col (C_in*k*k, batch*OH*OW)
//             Z = W × col + b  → (C_out, batch*OH*OW)
//             重排 → (C_out*OH*OW, batch)   （恢复 batch-major 列布局）
//   backward: grad_W += gZ × col^T
//             grad_b += row_reduce_sum(gZ)
//             grad_col = W^T × gZ
//             col2im → (C_in*H*W, batch)
//
// 说明：im2col/col2im 涉及复杂重排，沿用 PatchEmbedding 的先例在 CPU 端完成
//       （to_matrix/from_matrix），GEMM 仍复用引擎 matmul 内核。
//       MNIST 尺度下 CPU↔设备往返开销可忽略；GPU 融合卷积内核留作后续优化。
} // namespace nn

#endif // NN_COMPUTE_LAYER_MLP_HPP
