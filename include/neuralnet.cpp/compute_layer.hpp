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
#include <memory>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Layer — 引擎化计算层基类
// ══════════════════════════════════════════════════════════════════════════
class Layer
{
protected:
    // 梯度检查点模式：为 true 时 forward 不保留逐层中间激活（供激活重计算）。
    // 该模式由支持重计算的复合层（GPTBlock/TransformerEncoderLayer）在
    // forward 中按 checkpoint 边界设置，子层（Linear/GeLU/Norm/Attention 等）
    // 据此决定是否跳过缓存写入。
    bool checkpoint_mode_ = false;

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

    // batch 录制粒度控制（默认 no-op，仅 GPTModel override）
    virtual void set_flush_interval(std::size_t /*interval*/) {}

    // 文档感知：设置当前 step 每样本文档 id（默认 no-op，GPTModel override）
    virtual void set_doc_ids(std::span<const std::size_t> /*ids*/) {}

    // 训练/推理模式切换（默认 no-op，BatchNorm 等需要 override）
    virtual void set_training(bool /*training*/) {}

    // 非可学习状态收集（默认空，BatchNorm 的 running 统计量等需要 override）
    [[nodiscard]] virtual std::vector<TensorRef> extra_state() { return {}; }

    // 引擎相关初始化（创建/上传权重张量），替换构造函数中的 NN_ASSERT 模式。
    // 默认实现空操作；各层在构造后由 Model::add<T>() 调用。
    // 返回 Result 以正确传播引擎错误，而非在 Release 下吞掉。
    [[nodiscard]] virtual Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    // 梯度检查点（激活重计算）契约 ──────────────────────────────────
    // checkpoint_mode_ = true 时，forward 不保留中间激活（供 L1 激活重计算）；
    // forward_recompute 重算 forward 并重建缓存（供 backward 使用）。
    virtual void set_checkpoint_mode(bool enabled) { checkpoint_mode_ = enabled; }
    [[nodiscard]] bool checkpoint_mode() const noexcept { return checkpoint_mode_; }

    // 释放本层为 backward 保留的中间激活缓存（清空成员缓存 Tensor，归还显存）。
    // 由 GPTModel 在 checkpoint 块 backward 之后调用，避免重算的激活跨块累积
    // （否则所有块缓存会在 backward 末尾同时驻留，抵消检查点的显存收益）。
    // 默认 no-op；各缓存持有层 override。
    virtual void clear_cache() {}

    // 返回本层 backward 所需的中间激活缓存引用（供 activation offload 导出/导入）。
    // 仅返回 valid 的张量；掩码等小而常驻的缓存不在此列（不参与 offload，保持常驻）。
    [[nodiscard]] virtual std::vector<TensorRef> activation_cache() { return {}; }

    // 该层是否可作为“重计算单元”（即 forward_recompute 有实际意义）
    [[nodiscard]] virtual bool recompute_supported() const { return false; }

    // 从保存的输入重算 forward，重建本层 backward 所需的中间缓存。
    // 默认实现：临时关闭 checkpoint 模式重跑 forward（保留缓存）再恢复。
    // 复合层（GPTBlock 等）override 以同时关闭子层的 checkpoint 模式。
    [[nodiscard]] virtual Result<Tensor> forward_recompute(
        ComputeEngine& engine, const Tensor& saved_input)
    {
        const bool prev = checkpoint_mode_;
        checkpoint_mode_ = false;
        auto r = forward(engine, saved_input);
        checkpoint_mode_ = prev;
        return r;
    }

    // 梯度检查点粒度（默认 0 = 不启用；由 GPTModel 等 override）
    virtual void set_checkpoint_every(std::size_t /*stride*/) {}

    // activation offload（L1-offload）开关（默认 no-op；GPTModel override）
    virtual void set_activation_offload(bool /*enabled*/) {}

    // 理论 offload RAM 字节数（各层累计；默认 0，GPTModel override）
    [[nodiscard]] virtual std::size_t offload_ram_bytes() { return 0; }
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

    void clear_cache() override
    {
        input_cache_ = Tensor{};
        sigmoid_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        if (sigmoid_cache_.valid()) r.emplace_back(sigmoid_cache_);
        return r;
    }

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
        if (!checkpoint_mode_)
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

        if (!checkpoint_mode_)
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
        auto t1 = engine.elementwise_unary(UnaryOp::Neg, *gate);
        if (!t1) return std::unexpected(t1.error());
        auto t2 = engine.elementwise_unary(UnaryOp::Exp, *t1);
        if (!t2) return std::unexpected(t2.error());
        auto t3 = engine.elementwise_binary_scalar(BinaryOp::Add, *t2, Scalar{1});
        if (!t3) return std::unexpected(t3.error());
        auto s = engine.elementwise_binary_scalar(BinaryOp::Div, *t3, Scalar{1}, true);
        if (!s) return std::unexpected(s.error());

        if (!checkpoint_mode_)
        {
            gate_cache_ = *gate;
            sigmoid_cache_ = *s;
            up_cache_ = *up;
        }

        // sw = gate * s，out = sw ⊙ up
        auto sw = engine.elementwise_binary(BinaryOp::Mul, *gate, *s);
        if (!sw) return std::unexpected(sw.error());
        return engine.elementwise_binary(BinaryOp::Mul, *sw, *up);
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
// ══════════════════════════════════════════════════════════════════════════
class Conv2D final : public Layer
{
private:
    std::size_t in_channels_, out_channels_;
    std::size_t kernel_, stride_, padding_;
    std::size_t in_h_, in_w_;   // 输入空间尺寸（方形外也可用，用于 im2col）
    std::size_t out_h_, out_w_; // 输出空间尺寸（构造时计算）

    Tensor w_;        // (C_out, C_in*k*k)
    Tensor b_;        // (C_out, 1)
    Tensor grad_w_;
    Tensor grad_b_;
    Matrix col_cache_;           // im2col 输出 (C_in*k*k, batch*OH*OW)，供 backward

    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    // ── im2col：input (C_in*H*W, batch) → col (C_in*k*k, batch*OH*OW) ──
    static Matrix im2col_(const Matrix& input,
                          std::size_t C_in, std::size_t H_in, std::size_t W_in,
                          std::size_t batch,
                          std::size_t k, std::size_t stride, std::size_t pad,
                          std::size_t H_out, std::size_t W_out)
    {
        Matrix col(C_in * k * k, batch * H_out * W_out);
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t oh = 0; oh < H_out; ++oh)
            {
                for (std::size_t ow = 0; ow < W_out; ++ow)
                {
                    for (std::size_t ci = 0; ci < C_in; ++ci)
                    {
                        for (std::size_t kh = 0; kh < k; ++kh)
                        {
                            for (std::size_t kw = 0; kw < k; ++kw)
                            {
                                const long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(pad);
                                const long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(pad);
                                Scalar v = Scalar{0};
                                if (ih >= 0 && iw >= 0 &&
                                    ih < static_cast<long>(H_in) && iw < static_cast<long>(W_in))
                                {
                                    v = input.at_unchecked(ci * H_in * W_in + ih * W_in + iw, b);
                                }
                                const std::size_t r = ci * k * k + kh * k + kw;
                                const std::size_t c = b * H_out * W_out + oh * W_out + ow;
                                col.set_value_unchecked(r, c, v);
                            }
                        }
                    }
                }
            }
        }
        return col;
    }

    // ── col2im：col (C_in*k*k, batch*OH*OW) → (C_in*H*W, batch) ──
    static Matrix col2im_(const Matrix& col,
                          std::size_t C_in, std::size_t H_in, std::size_t W_in,
                          std::size_t batch,
                          std::size_t k, std::size_t stride, std::size_t pad,
                          std::size_t H_out, std::size_t W_out)
    {
        Matrix out(C_in * H_in * W_in, batch, Scalar{0});
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t oh = 0; oh < H_out; ++oh)
            {
                for (std::size_t ow = 0; ow < W_out; ++ow)
                {
                    for (std::size_t ci = 0; ci < C_in; ++ci)
                    {
                        for (std::size_t kh = 0; kh < k; ++kh)
                        {
                            for (std::size_t kw = 0; kw < k; ++kw)
                            {
                                const long ih = static_cast<long>(oh * stride + kh) - static_cast<long>(pad);
                                const long iw = static_cast<long>(ow * stride + kw) - static_cast<long>(pad);
                                if (ih < 0 || iw < 0 ||
                                    ih >= static_cast<long>(H_in) || iw >= static_cast<long>(W_in))
                                    continue;
                                const std::size_t r = ci * k * k + kh * k + kw;
                                const std::size_t c = b * H_out * W_out + oh * W_out + ow;
                                const std::size_t orow = ci * H_in * W_in + ih * W_in + iw;
                                out.set_value_unchecked(orow, b,
                                    out.at_unchecked(orow, b) + col.at_unchecked(r, c));
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    // ── 布局重排：Z (C_out, batch*OH*OW) → out (C_out*OH*OW, batch) ──
    static Matrix cols_to_samples_(const Matrix& Z, std::size_t C_out,
                                   std::size_t OH, std::size_t OW, std::size_t batch)
    {
        Matrix out(C_out * OH * OW, batch);
        for (std::size_t co = 0; co < C_out; ++co)
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t p = 0; p < OH * OW; ++p)
                    out.set_value_unchecked(co * OH * OW + p, b,
                        Z.at_unchecked(co, b * OH * OW + p));
        return out;
    }

    // ── 布局重排：out (C_out*OH*OW, batch) → Z (C_out, batch*OH*OW) ──
    static Matrix samples_to_cols_(const Matrix& out, std::size_t C_out,
                                   std::size_t OH, std::size_t OW, std::size_t batch)
    {
        Matrix Z(C_out, batch * OH * OW);
        for (std::size_t co = 0; co < C_out; ++co)
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t p = 0; p < OH * OW; ++p)
                    Z.set_value_unchecked(co, b * OH * OW + p,
                        out.at_unchecked(co * OH * OW + p, b));
        return Z;
    }

public:
    Conv2D(std::size_t in_channels, std::size_t out_channels,
           std::size_t kernel, std::size_t stride = 1, std::size_t padding = 0,
           std::size_t in_h = 0, std::size_t in_w = 0)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_(kernel), stride_(stride), padding_(padding),
          in_h_(in_h), in_w_(in_w)
    {
        out_h_ = (in_h_ + 2 * padding_ - kernel_) / stride_ + 1;
        out_w_ = (in_w_ + 2 * padding_ - kernel_) / stride_ + 1;
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        const std::size_t fan_in = in_channels_ * kernel_ * kernel_;

        // 权重 (C_out, C_in*k*k) — He 风格均匀初始化
        Matrix w_cpu(out_channels_, fan_in);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(fan_in + out_channels_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        auto w_span = w_cpu.span();
        for (std::size_t i = 0; i < w_cpu.size(); ++i)
            w_span[i] = dist(rng_);

        Matrix b_cpu(out_channels_, 1);  // 零初始化

        auto w_res = engine.from_matrix(w_cpu);
        if (!w_res) return std::unexpected(w_res.error());
        w_ = std::move(*w_res);
        auto b_res = engine.from_matrix(b_cpu);
        if (!b_res) return std::unexpected(b_res.error());
        b_ = std::move(*b_res);

        grad_w_ = engine.create_tensor(out_channels_, fan_in);
        grad_b_ = engine.create_tensor(out_channels_, 1);
        { auto r1 = engine.zero(grad_w_); if (!r1) return std::unexpected(r1.error()); }
        { auto r2 = engine.zero(grad_b_); if (!r2) return std::unexpected(r2.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override { return {w_, b_}; }
    std::vector<TensorRef> param_gradients() override { return {grad_w_, grad_b_}; }

    void clear_cache() override { col_cache_ = Matrix{}; }

    // ── forward: Z = W × im2col(x) + b，重排回 batch-major ──────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != in_channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"conv forward: input shape mismatch"});
        const std::size_t batch = input.cols();

        // 下载输入 → CPU im2col
        auto in_r = engine.to_matrix(input);
        if (!in_r) return std::unexpected(in_r.error());
        Matrix col = im2col_(*in_r, in_channels_, in_h_, in_w_, batch,
                             kernel_, stride_, padding_, out_h_, out_w_);
        if (!checkpoint_mode_)
            col_cache_ = col;

        auto col_t = engine.from_matrix(col);
        if (!col_t) return std::unexpected(col_t.error());

        // Z = W × col → (C_out, batch*OH*OW)
        auto Z = engine.matmul(w_, *col_t, false, false);
        if (!Z) return std::unexpected(Z.error());
        auto rb = engine.broadcast_row_inplace(*Z, b_, BinaryOp::Add);
        if (!rb) return std::unexpected(rb.error());

        // 重排 → (C_out*OH*OW, batch)
        auto Z_cpu = engine.to_matrix(*Z);
        if (!Z_cpu) return std::unexpected(Z_cpu.error());
        Matrix out_cpu = cols_to_samples_(*Z_cpu, out_channels_, out_h_, out_w_, batch);
        return engine.from_matrix(out_cpu);
    }

    // ── backward ───────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t batch = grad_output.cols();

        // grad_output (C_out*OH*OW, batch) → gZ (C_out, batch*OH*OW)
        auto g_cpu = engine.to_matrix(grad_output);
        if (!g_cpu) return std::unexpected(g_cpu.error());
        Matrix gZ_cpu = samples_to_cols_(*g_cpu, out_channels_, out_h_, out_w_, batch);
        auto gZ = engine.from_matrix(gZ_cpu);
        if (!gZ) return std::unexpected(gZ.error());

        // grad_W += gZ × col^T → (C_out, C_in*k*k)
        auto col_t = engine.from_matrix(col_cache_);
        if (!col_t) return std::unexpected(col_t.error());
        auto gw = engine.matmul(*gZ, *col_t, false, true);
        if (!gw) return std::unexpected(gw.error());
        auto r1 = engine.add_inplace(grad_w_, *gw);
        if (!r1) return std::unexpected(r1.error());

        // grad_b += row_reduce_sum(gZ) → (C_out, 1)
        auto gb = engine.row_reduce_sum(*gZ);
        if (!gb) return std::unexpected(gb.error());
        auto r2 = engine.add_inplace(grad_b_, *gb);
        if (!r2) return std::unexpected(r2.error());

        // grad_col = W^T × gZ → (C_in*k*k, batch*OH*OW)
        auto gcol = engine.matmul(w_, *gZ, true, false);
        if (!gcol) return std::unexpected(gcol.error());
        auto gcol_cpu = engine.to_matrix(*gcol);
        if (!gcol_cpu) return std::unexpected(gcol_cpu.error());
        Matrix gin_cpu = col2im_(*gcol_cpu, in_channels_, in_h_, in_w_, batch,
                                 kernel_, stride_, padding_, out_h_, out_w_);
        return engine.from_matrix(gin_cpu);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MaxPool2D — 二维最大池化（记录 argmax 供 backward）
//
// 布局约定（batch-major 列布局）：
//   输入  Tensor: (C * H * W, batch)
//   输出  Tensor: (C * Hp * Wp, batch)，Hp=(H-pool)/stride+1，Wp 同理
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  每个 (pool×pool) 窗口取最大值，记录 argmax 位置
//   backward: 把梯度散射回 argmax 位置（其余位置为 0）
//
// 说明：池化涉及窗口 max + argmax 追踪，在 CPU 端完成（to_matrix/from_matrix），
//       与 Conv2D 的 im2col 策略一致；GPU 融合池化内核留作后续优化。
// ══════════════════════════════════════════════════════════════════════════
class MaxPool2D final : public Layer
{
private:
    std::size_t channels_, in_h_, in_w_;
    std::size_t pool_, stride_;
    std::size_t out_h_, out_w_;
    std::vector<std::size_t> max_indices_;  // (channels*out_h*out_w, batch) 扁平 argmax 行索引

public:
    MaxPool2D(std::size_t channels, std::size_t in_h, std::size_t in_w,
              std::size_t pool = 2, std::size_t stride = 0)
        : channels_(channels), in_h_(in_h), in_w_(in_w),
          pool_(pool), stride_(stride != 0 ? stride : pool)
    {
        out_h_ = (in_h_ - pool_) / stride_ + 1;
        out_w_ = (in_w_ - pool_) / stride_ + 1;
    }

    void clear_cache() override { max_indices_.clear(); }

    // ── forward ─────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != channels_ * in_h_ * in_w_)
            return std::unexpected(Error{"maxpool forward: input shape mismatch"});
        const std::size_t batch = input.cols();

        auto in_r = engine.to_matrix(input);
        if (!in_r) return std::unexpected(in_r.error());
        const Matrix& in = *in_r;

        Matrix out(channels_ * out_h_ * out_w_, batch);
        if (!checkpoint_mode_)
            max_indices_.assign(channels_ * out_h_ * out_w_ * batch, 0);

        const std::size_t out_area = out_h_ * out_w_;
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t c = 0; c < channels_; ++c)
            {
                for (std::size_t oh = 0; oh < out_h_; ++oh)
                {
                    for (std::size_t ow = 0; ow < out_w_; ++ow)
                    {
                        Scalar best = -std::numeric_limits<Scalar>::infinity();
                        std::size_t best_idx = 0;
                        for (std::size_t dh = 0; dh < pool_; ++dh)
                        {
                            for (std::size_t dw = 0; dw < pool_; ++dw)
                            {
                                const std::size_t ih = oh * stride_ + dh;
                                const std::size_t iw = ow * stride_ + dw;
                                const std::size_t r = c * in_h_ * in_w_ + ih * in_w_ + iw;
                                const Scalar v = in.at_unchecked(r, b);
                                if (v > best) { best = v; best_idx = r; }
                            }
                        }
                        const std::size_t orr = c * out_area + oh * out_w_ + ow;
                        out.set_value_unchecked(orr, b, best);
                        if (!checkpoint_mode_)
                            max_indices_[b * channels_ * out_area + orr] = best_idx;
                    }
                }
            }
        }
        return engine.from_matrix(out);
    }

    // ── backward: 散射梯度到 argmax 位置 ────────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t batch = grad_output.cols();
        const std::size_t out_area = out_h_ * out_w_;

        auto g_r = engine.to_matrix(grad_output);
        if (!g_r) return std::unexpected(g_r.error());
        const Matrix& g = *g_r;

        Matrix gin(channels_ * in_h_ * in_w_, batch, Scalar{0});
        for (std::size_t b = 0; b < batch; ++b)
        {
            for (std::size_t c = 0; c < channels_; ++c)
            {
                for (std::size_t oh = 0; oh < out_h_; ++oh)
                {
                    for (std::size_t ow = 0; ow < out_w_; ++ow)
                    {
                        const std::size_t orr = c * out_area + oh * out_w_ + ow;
                        const std::size_t idx = max_indices_[b * channels_ * out_area + orr];
                        gin.set_value_unchecked(idx, b,
                            gin.at_unchecked(idx, b) + g.at_unchecked(orr, b));
                    }
                }
            }
        }
        return engine.from_matrix(gin);
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

    void clear_cache() override { output_cache_ = Tensor{}; }

    // 供组合层（AttentionBase）读取/复用 softmax 输出，避免重复存一份
    Tensor& output_cache() { return output_cache_; }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (output_cache_.valid()) r.emplace_back(output_cache_);
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        // 行 softmax（数值稳定）：out = exp(x - row_max) / Σ_c exp(x - row_max)
        // 单表达式融合（M3）：row_max/row_sum 为归约视图/归约指令，中间全尺寸
        // Tensor（shifted/exp_shift/row_max/row_sum 的物化）由融合 kernel 消解，
        // 仅 input 与 output 落显存。算法公式与旧多次原语完全一致。
        // 表达式文本只写在本 Layer；AOT 收集由 scan_exprs dry-run 本方法完成。
        auto out = dsl::compute(engine,
            dsl::exp(dsl::leaf(input) - dsl::row_reduce_max(input))
            / dsl::row_reduce_sum(
                dsl::exp(dsl::leaf(input) - dsl::row_reduce_max(input))),
            input.rows(), input.cols());
        if (!out) return std::unexpected(out.error());
        if (checkpoint_mode_)
            return out;
        // 单缓冲：把结果移入 output_cache_（唯一持有者），返回共享同一 buffer
        output_cache_ = std::move(*out);
        return output_cache_;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // grad_x = out ⊙ (grad_output - row_dot(out ⊙ grad_output))
        // 单表达式融合（M3）：row_dot 为归约指令，消除 ep/gmd 等全尺寸中间 Tensor。
        auto out = dsl::compute(engine,
            dsl::leaf(output_cache_)
            * (dsl::leaf(grad_output)
               - dsl::row_reduce_sum(dsl::leaf(output_cache_)
                                     * dsl::leaf(grad_output))),
            output_cache_.rows(), output_cache_.cols());
        return out;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RotaryEmbedding — 旋转位置编码（RoPE）缓存与应用
//
// 约定（LLaMA/Mistral/DeepSeek half-swap 风格，便于与参考实现交叉验证）：
//   q_rot = q·cos + rotate_half(q)·sin
//   rotate_half(x) = [-x_后半, x_前半]   —— 维度对 (j, j+d_k/2) 的 2×2 旋转块
//   cos/sin 表 (d_k, seq)：cos[j]=cos[j+d_k/2]=cos(t·θ_j)，
//     即沿 d 维 = cat(freqs, freqs)（前后半相同，LLaMA 约定），θ_j = t / 10000^(2j/d_k)
//
// 施加方式（统一表达式 DSL，见 expr_spec.hpp）：
//   inputs: [q, q, cos, sin]
//   views:  [Linear, RotateHalf(block=d_k, 前半取负), RowMod(d_k), RowMod(d_k)]
//   forward: r0=q*cos; r1=rot(q)*sin; out=r0+r1
//   backward（旋转正交，逆=反角）: out=r0−r1
//   —— 单次 eval_expr 完成，CPU 一次遍历、无中间 Tensor。
//
// 位置约定：应用点在 Q/K 完成 rearrange 之后（列=position），
//   Q/K 形状 (batch*H*d_k, seq)，cos/sin 短表 (d_k, seq) 按 RowMod 平铺。
// ══════════════════════════════════════════════════════════════════════════
class RotaryEmbedding
{
private:
    std::size_t d_k_ = 0;
    std::size_t seq_cached_ = 0;   // 缓存表的列数（全表应用）
    std::size_t pos_offset_ = 0;   // 绝对位置偏移（滑动窗生成时用，0=从 0 起）
    Tensor cos_cache_;             // (d_k, seq_cached_)
    Tensor sin_cache_;             // (d_k, seq_cached_)

    // 把单个位置 pos 的 cos/sin 写入指定列（rebuild/apply_step 共用）。
    // LLaMA 式：cos 沿 d 维 = cat(freqs, freqs)（前后半相同），
    // 配合 rotate_half（前后半交换+前半取负）构成 2×2 旋转块。
    void fill_pos_column_(Matrix& c, Matrix& s, std::size_t pos, std::size_t col) const
    {
        const std::size_t half = d_k_ / 2;
        const Scalar pd = static_cast<Scalar>(pos);
        for (std::size_t j = 0; j < half; ++j)
        {
            const Scalar theta = pd / std::pow(Scalar{10000},
                static_cast<Scalar>(2 * j) / static_cast<Scalar>(d_k_));
            const Scalar cv = std::cos(theta);
            const Scalar sv = std::sin(theta);
            c.set_value_unchecked(j,        col, cv);
            c.set_value_unchecked(half + j, col, cv);
            s.set_value_unchecked(j,        col, sv);
            s.set_value_unchecked(half + j, col, sv);
        }
    }

    // 重建全表 (d_k, seq)：cos/sin 按维度对交错重复；位置 = pos + pos_offset_
    // （绝对位置偏移：滑动窗生成时，输入被截断到窗口，但位置应从真实起点算起）
    [[nodiscard]] Result<void> rebuild(ComputeEngine& engine, std::size_t seq)
    {
        Matrix c(d_k_, seq), s(d_k_, seq);
        for (std::size_t pos = 0; pos < seq; ++pos)
            fill_pos_column_(c, s, pos + pos_offset_, pos);
        auto cr = engine.from_matrix(c);
        if (!cr) return std::unexpected(cr.error());
        cos_cache_ = std::move(*cr);
        auto sr = engine.from_matrix(s);
        if (!sr) return std::unexpected(sr.error());
        sin_cache_ = std::move(*sr);
        seq_cached_ = seq;
        return {};
    }

    // 构造 RoPE 内联表达式（forward 用 Add 结尾，backward 用 Sub 结尾）
    // 表达式文本只写在本 Layer（apply/apply_step），AOT 收集（scan_exprs）
    // 用同一段代码 dry-run 折叠出结构并合成融合 shader——绝不漂移。

public:
    RotaryEmbedding() = default;
    explicit RotaryEmbedding(std::size_t d_k)
        : d_k_(d_k)
    {
    }

    [[nodiscard]] Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    [[nodiscard]] std::size_t d_k() const noexcept { return d_k_; }

    // 设置绝对位置偏移（滑动窗生成：输入截断到窗口，位置从真实起点算起）。
    // 改变后强制下次 apply 重建 cos/sin 表。
    void set_position_offset(std::size_t off)
    {
        if (off != pos_offset_)
        {
            pos_offset_ = off;
            seq_cached_ = 0;   // 使下次 apply 触发 rebuild
        }
    }

    // 全表应用：q 为 (batch*H*d_k, seq)，cos/sin 为 (d_k, seq) 短表
    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& q,
        std::size_t seq, bool backward)
    {
        if (d_k_ == 0 || d_k_ % 2 != 0)
            return std::unexpected(Error{"RotaryEmbedding::apply: d_k must be positive and even"});
        if (seq != seq_cached_)
        {
            auto r = rebuild(engine, seq);
            if (!r) return std::unexpected(r.error());
        }
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        // 内联 RoPE 表达式（LLaMA half-swap；backward 为旋转正交，逆 = 反角）：
        //   forward:  out = q·cos + rotate_half(q)·sin
        //   backward: out = q·cos − rotate_half(q)·sin
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
                q.rows(), q.cols());
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(cos_cache_, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(sin_cache_, dk),
            q.rows(), q.cols());
    }

    // 增量推理：q 为 (H*d_k, 1)，位置 = pos（cur_len）
    // 直接生成 (d_k,1) 位置表，避免对全表切片取列。
    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& q, std::size_t pos, bool backward)
    {
        if (d_k_ == 0 || d_k_ % 2 != 0)
            return std::unexpected(Error{"RotaryEmbedding::apply_step: d_k must be positive and even"});
        Matrix c(d_k_, 1), s(d_k_, 1);
        fill_pos_column_(c, s, pos, 0);
        auto cr = engine.from_matrix(c);
        if (!cr) return std::unexpected(cr.error());
        auto sr = engine.from_matrix(s);
        if (!sr) return std::unexpected(sr.error());
        const std::uint32_t dk = static_cast<std::uint32_t>(d_k_);
        if (backward)
            return dsl::compute(engine,
                dsl::leaf(q) * dsl::row_mod(*cr, dk)
                - dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
                q.rows(), q.cols());
        return dsl::compute(engine,
            dsl::leaf(q) * dsl::row_mod(*cr, dk)
            + dsl::rotate_half(q, dk) * dsl::row_mod(*sr, dk),
            q.rows(), q.cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// AttentionBase — 多头注意力基类（批量化：消除 per-head 和 per-sample 循环）
//
// 提取 MultiHeadAttention 与 CausalSelfAttention 的公共逻辑：
//   - 完全相同的成员变量、参数/梯度接口
//   - forward/backward 仅在「scores 之后、softmax 之前」是否施加掩码上有差异
//
// 算法（只在此处，不在 Engine/Shader）：
//   Q = W_q × x, K = W_k × x, V = W_v × x  (三个 Linear 投影)
//   Q/K/V: (H*d_k, batch*seq) — 头维度在行方向，batch 在列方向
//
//   批量化关键：用 rearrange_3d 把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
//   使 batched_matmul 能按 batch*H 切分行块，单次 dispatch 处理所有样本和所有头。
//
//   Q_re = rearrange_3d(Q, H*d_k, batch, seq) → (batch*H*d_k, seq)
//   S = batched_matmul(Q_re, K_re, batch*H, transA=true, alpha=scale) → (batch*H*seq, seq)
//     （scale = 1/sqrt(d_k) 通过 matmul 的 alpha 系数折进写出，省去独立 scale pass）
//   S = apply_mask_(S)         ← 子类钩子（默认 no-op = MHA 行为）
//   A = softmax(S)  — 行级归一化，堆叠布局下天然正确
//   O_re = batched_matmul(V_re, A, batch*H) → (batch*H*d_k, seq)
//   O = rearrange_3d(O_re, H*d_k, batch, seq, inverse=true) → (H*d_k, batch*seq)
//   out = W_o × O
//
// 输入形状: (d_model, batch * seq_len)，输出形状: (d_model, batch * seq_len)
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断
// ══════════════════════════════════════════════════════════════════════════
class AttentionBase : public Layer
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

    // RoPE（pos_enc == RoPE 时启用）：作用在 Q/K 的 d_k 维（每头），
    // 施加点在 Q/K 完成 rearrange 之后（列=position）。
    bool use_rope_ = false;
    RotaryEmbedding rope_;

    // forward 缓存（rearranged 版本，供 backward 直接使用）
    Tensor Q_cache_, K_cache_, V_cache_;  // (batch*H*d_k, seq) rearranged
    Tensor attn_cache_;                    // (batch*H*seq, seq) 旧路径缓存

    // 两趟式缓存（M6）：m/l 替代 attn_cache_（不物化得分矩阵）
    Tensor m_cache_, l_cache_;   // (batch*H*seq, 1)：行 max / softmax 分母
    AttnBias two_pass_bias_;     // 两趟式组合偏置描述子（指针指向子类缓存张量）
    bool two_pass_active_ = false;  // forward 是否走了两趟式路径（backward 读取）

    // ── 两趟式（M6）决策钩子 ──────────────────────────────────────────
    // 决定是否用两趟式注意力（不物化 (BH*seq, seq) 得分/概率矩阵，显存峰值
    // 从 ~3-4×BH·seq² 降至 ~1×）。返回 {use_two_pass, bias}：
    //   - use_two_pass=true：两趟式；bias 为组合式 AttnBias 描述子
    //     （causal / doc_ids / slopes，见 compute_engine.hpp；空 = 无偏置/掩码）
    //   - use_two_pass=false：回退旧路径
    struct TwoPassMask
    {
        bool use_two_pass = true;
        AttnBias bias;
    };
    [[nodiscard]] virtual Result<TwoPassMask> two_pass_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        (void)engine; (void)batch; (void)seq;
        return TwoPassMask{true, AttnBias{}};  // MHA：无偏置，两趟式
    }

    // 掩码钩子：子类重写以施加掩码，默认 no-op（MHA 行为）
    // 在 forward 中 scores（已含 alpha=scale 缩放）之后、softmax 之前调用
    [[nodiscard]] virtual Result<Tensor> apply_mask_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t batch, std::size_t seq)
    {
        (void)engine; (void)batch; (void)seq;
        return std::move(scores);
    }

    // 增量推理掩码钩子：在 forward_step 中 scale 之后、softmax 之前调用。
    // 默认 no-op：MHA 无掩码；CSA 因果掩码在增量推理中天然满足
    // （新 token 只能看到 cache 中的前文，无未来位置可掩）。
    // ALiBi 子类重写以施加线性距离偏置 -slope*(cur_len - j)。
    [[nodiscard]] virtual Result<Tensor> apply_mask_step_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t cur_len)
    {
        (void)engine; (void)cur_len;
        return std::move(scores);
    }

    // 在 backward 中 softmax 反向之后调用（掩码为常数，梯度直接穿透，默认 no-op）
    virtual void mask_backward_(
        ComputeEngine& /*engine*/, Tensor& /*grad_S*/,
        std::size_t /*batch*/, std::size_t /*seq*/) {}

public:
    AttentionBase(std::size_t d_model, std::size_t num_heads,
                  std::size_t seq_len = 0,
                  PosEncodingType pos_enc = PosEncodingType::Learned)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          w_q_(d_model, d_model),
          w_k_(d_model, d_model),
          w_v_(d_model, d_model),
          w_o_(d_model, d_model),
          use_rope_(pos_enc == PosEncodingType::RoPE),
          rope_(d_model / num_heads)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "AttentionBase: d_model must be divisible by num_heads");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = w_q_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_k_.init(engine); if (!r2) return std::unexpected(r2.error());
        auto r3 = w_v_.init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = w_o_.init(engine); if (!r4) return std::unexpected(r4.error());
        if (use_rope_)
        {
            auto r5 = rope_.init(engine); if (!r5) return std::unexpected(r5.error());
        }
        return {};
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

    // 梯度检查点：把模式传播给内部投影层与 softmax
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
        softmax_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        Q_cache_ = Tensor{};
        K_cache_ = Tensor{};
        V_cache_ = Tensor{};
        attn_cache_ = Tensor{};
        m_cache_ = Tensor{};
        l_cache_ = Tensor{};
        // 掩码/偏置描述子（two_pass_bias_ 指向子类 doc_ids/slopes 缓存）小而常驻，
        // 不随激活清理
        w_q_.clear_cache();
        w_k_.clear_cache();
        w_v_.clear_cache();
        w_o_.clear_cache();
        softmax_.clear_cache();
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Q_cache_.valid()) r.emplace_back(Q_cache_);
        if (K_cache_.valid()) r.emplace_back(K_cache_);
        if (V_cache_.valid()) r.emplace_back(V_cache_);
        if (m_cache_.valid()) r.emplace_back(m_cache_);
        if (l_cache_.valid()) r.emplace_back(l_cache_);
        auto wq = w_q_.activation_cache(); r.insert(r.end(), wq.begin(), wq.end());
        auto wk = w_k_.activation_cache(); r.insert(r.end(), wk.begin(), wk.end());
        auto wv = w_v_.activation_cache(); r.insert(r.end(), wv.begin(), wv.end());
        auto wo = w_o_.activation_cache(); r.insert(r.end(), wo.begin(), wo.end());
        auto sm = softmax_.activation_cache(); r.insert(r.end(), sm.begin(), sm.end());
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"AttentionBase forward: input shape mismatch"});

        const std::size_t total_seq = input.cols();
        const std::size_t seq      = (seq_len_ > 0) ? seq_len_ : total_seq;
        const std::size_t batch    = (seq_len_ > 0) ? (total_seq / seq_len_) : 1;
        if (total_seq != batch * seq)
            return std::unexpected(Error{"AttentionBase forward: cols not divisible by seq_len"});

        // 1. 线性投影 → Q/K/V: (H*d_k, batch*seq)
        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        // 2. rearrange: (H*d_k, batch*seq) → (batch*H*d_k, seq)
        //    使 batched_matmul 能按 batch*H 切分行块
        //    局部 Q/K/V 承载 forward 计算；仅在非 checkpoint 模式下写入成员缓存。
        const std::size_t H_dk = num_heads_ * d_k_;
        Tensor Q, K, V;  // (batch*H*d_k, seq) rearranged
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false);
            if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false);
            if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false);
            if (!vr) return std::unexpected(vr.error());
            V = std::move(*vr);
        }
        else
        {
            // batch=1: rearrange 是恒等拷贝，跳过
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // 2.5 RoPE：对 Q/K 施加旋转位置编码（rearrange 后列=position，
        //    每 d_k 行一个头，cos/sin 短表按 RowMod 平铺）
        if (use_rope_)
        {
            auto qr2 = rope_.apply(engine, Q, seq, /*backward=*/false);
            if (!qr2) return std::unexpected(qr2.error());
            Q = std::move(*qr2);
            auto kr2 = rope_.apply(engine, K, seq, /*backward=*/false);
            if (!kr2) return std::unexpected(kr2.error());
            K = std::move(*kr2);
        }

        // 3-7. 注意力主体：两趟式 vs 旧路径（由掩码钩子决策）
        const std::size_t BH = batch * num_heads_;
        auto tpm = two_pass_mask_(engine, batch, seq);
        if (!tpm) return std::unexpected(tpm.error());
        Tensor concat_out;  // (batch*H*d_k, seq)
        if (tpm->use_two_pass)
        {
            // ── 两趟式（M6）：不物化 (BH*seq, seq) 得分/概率矩阵 ──
            //   m = rowmax(scale·Q·K^T + mask)        → (BH*seq, 1)
            //   l = Σ_j exp(scale·Q·K^T + mask − m)   → (BH*seq, 1)
            //   O = W·V_t（W = softmax 归一化权重）    → (BH*seq, d_k)
            two_pass_bias_ = tpm->bias;
            auto m = engine.batched_matmul_reduce(
                Q, K, BH, ReduceOp::Max, true, false,
                scale_, /*reduce_cols=*/true, two_pass_bias_);
            if (!m) return std::unexpected(m.error());
            auto l = engine.batched_matmul_softmax_denom(
                Q, K, *m, BH, true, false, scale_, two_pass_bias_);
            if (!l) return std::unexpected(l.error());
            // V 需 (BH*seq, d_k) 布局（apply 原语的 V 约定）：V (BH*d_k, seq)
            // 是 per-batch (d_k, seq)，需按 batch 转置：
            //   transpose → (seq, BH*d_k) → rearrange_3d(seq, BH, d_k) → (BH*seq, d_k)
            auto V_T_full = engine.transpose(V);
            if (!V_T_full) return std::unexpected(V_T_full.error());
            auto V_t = engine.rearrange_3d(*V_T_full, seq, BH, d_k_, false);
            if (!V_t) return std::unexpected(V_t.error());
            auto O_t = engine.batched_matmul_softmax_apply(
                Q, K, *V_t, *m, *l, BH, true, false, scale_, two_pass_bias_);
            if (!O_t) return std::unexpected(O_t.error());
            // O_t: (BH*seq, d_k) → 按 batch 转置回 (BH*d_k, seq) 供后续 rearrange：
            //   transpose → (d_k, BH*seq) → rearrange_3d(d_k, BH, seq) → (BH*d_k, seq)
            auto O_T_full = engine.transpose(*O_t);
            if (!O_T_full) return std::unexpected(O_T_full.error());
            auto co = engine.rearrange_3d(*O_T_full, d_k_, BH, seq, false);
            if (!co) return std::unexpected(co.error());
            concat_out = std::move(*co);
            if (!checkpoint_mode_)
            {
                Q_cache_ = std::move(Q);
                K_cache_ = std::move(K);
                V_cache_ = std::move(V);
                m_cache_ = std::move(*m);
                l_cache_ = std::move(*l);
            }
            two_pass_active_ = true;
        }
        else
        {
            // ── 旧路径：物化得分矩阵（ALiBi/doc_ids 等共享掩码不适用时回退） ──
            two_pass_active_ = false;
            // S = batched_matmul(Q, K, batch*H, transA=true) → (batch*H*seq, seq)
            // scale (1/sqrt(d_k)) 通过 alpha 系数折进 matmul 写出（cuBLAS sgemm 语义），
            // 省去一次全矩阵 scale pass + 额外 barrier
            auto scores = engine.batched_matmul(
                Q, K, BH, true, false, scale_);
            if (!scores) return std::unexpected(scores.error());
            // 施加掩码（钩子：MHA 默认 no-op，CSA 施加因果/ALiBi 掩码）
            auto masked = apply_mask_(engine, std::move(*scores), batch, seq);
            if (!masked) return std::unexpected(masked.error());
            // A = softmax(S_masked)
            auto attn = softmax_.forward(engine, *masked);
            if (!attn) return std::unexpected(attn.error());
            Tensor A = std::move(*attn);
            // O_re = batched_matmul(V, A, batch*H, transB=true) → (batch*H*d_k, seq)
            // 标准 attention: O[:,i] = sum_j V[:,j] * A[i,j] = (V × A^T)[:,i]
            auto co = engine.batched_matmul(V, A, BH, false, true);
            if (!co) return std::unexpected(co.error());
            concat_out = std::move(*co);
            if (!checkpoint_mode_)
            {
                // 注意力概率由 softmax_.output_cache() 单一持有（A 与其共享 buffer），
                // 不再另存 attn_cache_（避免旧路径重复存一份 (BH*seq, seq) 大矩阵）
                Q_cache_ = std::move(Q);
                K_cache_ = std::move(K);
                V_cache_ = std::move(V);
            }
        }

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(concat_out, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(concat_out);
        }

        // 9. 输出投影
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

        // 3-7. 注意力反向：两趟式（M6，重算 W 不物化 grad_S）vs 旧路径
        Tensor grad_Q_re, grad_K_re, grad_V_re;  // 均 (batch*H*d_k, seq)
        if (two_pass_active_)
        {
            // P = grad_A = batched_matmul(grad_concat^T, V, BH, true, false)
            // forward: O = V × A^T → grad_A = grad_O^T × V（两趟式反向的 P 输入）
            auto grad_A = engine.batched_matmul(
                grad_concat_re, V_cache_, BH, true, false);
            if (!grad_A) return std::unexpected(grad_A.error());
            // G = grad_concat_re^T 按 batch 转置 → (BH*seq, d_k)，
            // 供 grad_V[j][k] = Σ_i W·G[i][k]（同 V 的布局转换）
            auto G_T_full = engine.transpose(grad_concat_re);
            if (!G_T_full) return std::unexpected(G_T_full.error());
            auto G = engine.rearrange_3d(*G_T_full, seq, BH, d_k_, false);
            if (!G) return std::unexpected(G.error());
            // Pass 1：R[i] = Σ_j W·P；grad_Q[:,i] = scale·Σ_j W·(P−R)·K_b[:,j]
            // （kernel 内部重算 W，不物化 (BH*seq, seq)）
            Tensor R;
            auto gq = engine.batched_matmul_softmax_backward_q(
                Q_cache_, K_cache_, *grad_A, m_cache_, l_cache_,
                BH, true, false, scale_, R, two_pass_bias_);
            if (!gq) return std::unexpected(gq.error());
            // Pass 2：grad_K[:,j] = scale·Σ_i W·(P−R)·Q_b[:,i]；
            //          grad_V[j][k] = Σ_i W·G[i][k]
            auto gkv = engine.batched_matmul_softmax_backward_kv(
                Q_cache_, K_cache_, *grad_A, *G, R, m_cache_, l_cache_,
                BH, true, false, scale_, grad_V_re, two_pass_bias_);
            if (!gkv) return std::unexpected(gkv.error());
            grad_Q_re = std::move(*gq);
            grad_K_re = std::move(*gkv);
        }
        else
        {
            // grad_V_re = batched_matmul(grad_concat, A, BH, false, false)
            // forward: O = V × A^T → grad_V = grad_O × A
            // A 由 softmax.output_cache() 单一持有（避免 attn_cache_ 重复存一份）
            auto gvr = engine.batched_matmul(
                grad_concat_re, softmax_.output_cache(), BH, false, false);
            if (!gvr) return std::unexpected(gvr.error());
            grad_V_re = std::move(*gvr);
            // grad_A = batched_matmul(grad_concat^T, V, BH, true, false)
            // forward: O = V × A^T → grad_A = grad_O^T × V
            auto grad_A = engine.batched_matmul(
                grad_concat_re, V_cache_, BH, true, false);
            if (!grad_A) return std::unexpected(grad_A.error());
            // grad_S = softmax.backward(grad_A) — 掩码/偏置为常数，梯度直接穿透
            auto grad_S = softmax_.backward(engine, *grad_A);
            if (!grad_S) return std::unexpected(grad_S.error());
            mask_backward_(engine, *grad_S, batch, seq);
            // grad_Q_re = batched_matmul(K, grad_S, BH, false, true) × scale
            // 前向 S = scale·Q^T·K → ∂L/∂Q = scale·K·grad_S^T，
            // scale 通过 alpha 折进 matmul 写出（省去两次全矩阵 scale pass）
            auto gq = engine.batched_matmul(
                K_cache_, *grad_S, BH, false, true, scale_);
            if (!gq) return std::unexpected(gq.error());
            grad_Q_re = std::move(*gq);
            // grad_K_re = batched_matmul(Q, grad_S, BH, false, false) × scale
            // ∂L/∂K = scale·Q·grad_S，同样折进 alpha
            auto gk = engine.batched_matmul(
                Q_cache_, *grad_S, BH, false, false, scale_);
            if (!gk) return std::unexpected(gk.error());
            grad_K_re = std::move(*gk);
        }

        // 7.5 RoPE backward：对 Q/K 梯度施加反角旋转
        //    （forward 的旋转矩阵正交，逆 = 转置 = 反角：grad*cos − rot(grad)*sin）
        if (use_rope_)
        {
            auto gq2 = rope_.apply(engine, grad_Q_re, seq, /*backward=*/true);
            if (!gq2) return std::unexpected(gq2.error());
            grad_Q_re = std::move(*gq2);
            auto gk2 = rope_.apply(engine, grad_K_re, seq, /*backward=*/true);
            if (!gk2) return std::unexpected(gk2.error());
            grad_K_re = std::move(*gk2);
        }

        // 8. rearrange back: (batch*H*d_k, seq) → (H*d_k, batch*seq)
        Tensor grad_Q, grad_K, grad_V;
        if (batch > 1)
        {
            auto gq = engine.rearrange_3d(grad_Q_re, H_dk, batch, seq, true);
            if (!gq) return std::unexpected(gq.error());
            grad_Q = std::move(*gq);
            auto gk = engine.rearrange_3d(grad_K_re, H_dk, batch, seq, true);
            if (!gk) return std::unexpected(gk.error());
            grad_K = std::move(*gk);
            auto gv = engine.rearrange_3d(grad_V_re, H_dk, batch, seq, true);
            if (!gv) return std::unexpected(gv.error());
            grad_V = std::move(*gv);
        }
        else
        {
            grad_Q = std::move(grad_Q_re);
            grad_K = std::move(grad_K_re);
            grad_V = std::move(grad_V_re);
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

    // ── 增量推理（KV cache）──────────────────────────────────────────
    // 处理单个新 token，复用历史 K/V 缓存，避免重复计算前文。
    //
    // KV cache 布局: (max_len, H*d_k)，row=position，col=head×dim
    //   - 投影得到 k_new/v_new: (H*d_k, 1)
    //   - transpose → (1, H*d_k) 后 insert_rows 到 cache 第 cur_len 行
    //   - slice_rows 取前 new_len 行参与 attention
    //
    // 无需因果掩码：新 token 天然只能看到自身及之前的位置（cache 中只有前文）。
    //
    // 输入: x_new (d_model, 1) 单个新 token 的嵌入
    // 输出: (d_model, 1) attention 输出
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        const Tensor& x_new,
        Tensor& k_cache,
        Tensor& v_cache,
        std::size_t cur_len)
    {
        if (x_new.rows() != d_model_ || x_new.cols() != 1)
            return std::unexpected(Error{"AttentionBase forward_step: x_new must be (d_model, 1)"});

        // 1. Q/K/V 投影 → (H*d_k, 1)
        auto q_res = w_q_.forward(engine, x_new);
        if (!q_res) return q_res;
        auto k_new = w_k_.forward(engine, x_new);
        if (!k_new) return k_new;
        auto v_new = w_v_.forward(engine, x_new);
        if (!v_new) return v_new;

        // 1.5 RoPE：对 Q/K 施加当前位置 (cur_len) 的旋转后写入 KV cache
        //    （cache 中的历史 K 已在各自 step 旋转过，相对位置自然成立）
        if (use_rope_)
        {
            auto qr = rope_.apply_step(engine, *q_res, cur_len, /*backward=*/false);
            if (!qr) return std::unexpected(qr.error());
            q_res = std::move(*qr);
            auto kr = rope_.apply_step(engine, *k_new, cur_len, /*backward=*/false);
            if (!kr) return std::unexpected(kr.error());
            k_new = std::move(*kr);
        }

        // 2. transpose → (1, H*d_k)，匹配 cache 的行布局
        auto k_new_T = engine.transpose(*k_new);
        if (!k_new_T) return std::unexpected(k_new_T.error());
        auto v_new_T = engine.transpose(*v_new);
        if (!v_new_T) return std::unexpected(v_new_T.error());

        // 3. 追加到 KV cache（就地写入第 cur_len 行）
        auto r1 = engine.insert_rows(k_cache, cur_len, *k_new_T);
        if (!r1) return std::unexpected(r1.error());
        auto r2 = engine.insert_rows(v_cache, cur_len, *v_new_T);
        if (!r2) return std::unexpected(r2.error());

        // 4. 取有效区间 [0, new_len) 并 transpose 为 (H*d_k, new_len) 布局
        //    batched_matmul 要求 rows 能被 batch 整除:
        //    (new_len, H*d_k) 的 rows=new_len 不保证整除 num_heads
        //    transpose 后 (H*d_k, new_len) 的 rows=H*d_k=d_model 必然整除
        const std::size_t new_len = cur_len + 1;
        auto k_valid = engine.slice_rows(k_cache, 0, new_len);
        if (!k_valid) return std::unexpected(k_valid.error());
        auto v_valid = engine.slice_rows(v_cache, 0, new_len);
        if (!v_valid) return std::unexpected(v_valid.error());
        auto K_T = engine.transpose(*k_valid);   // (H*d_k, new_len)
        if (!K_T) return std::unexpected(K_T.error());
        auto V_T = engine.transpose(*v_valid);   // (H*d_k, new_len)
        if (!V_T) return std::unexpected(V_T.error());

        // 5. scores = batched_matmul(Q, K_T, H, transA=T, transB=F)
        //    Q: (H*d_k, 1) — 每头 (d_k, 1)，转置后 (1, d_k)，M=1
        //    K_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transB=F，N=new_len
        //    每头: (1, d_k) × (d_k, new_len) = (1, new_len)
        //    堆叠: (H, new_len)
        //    scale (1/sqrt(d_k)) 通过 alpha 折进 matmul 写出
        auto scores = engine.batched_matmul(
            *q_res, *K_T, num_heads_, true, false, scale_);
        if (!scores) return std::unexpected(scores.error());

        // 6. 施加增量推理掩码钩子（默认 no-op；ALiBi 施加线性偏置）
        auto masked = apply_mask_step_(engine, std::move(*scores), cur_len);
        if (!masked) return std::unexpected(masked.error());

        // 7. softmax（行级归一化，每头独立）
        auto attn = softmax_.forward(engine, *masked);
        if (!attn) return std::unexpected(attn.error());

        // 8. attn_out = batched_matmul(V_T, attn, H, transA=F, transB=T)
        //    V_T: (H*d_k, new_len) — 每头 (d_k, new_len)，transA=F，M=d_k
        //    attn: (H, new_len) — 每头 (1, new_len)，转置后 (new_len, 1)，N=1
        //    每头: (d_k, new_len) × (new_len, 1) = (d_k, 1)
        //    堆叠: (H*d_k, 1)
        auto attn_out = engine.batched_matmul(
            *V_T, *attn, num_heads_, false, true);
        if (!attn_out) return std::unexpected(attn_out.error());

        // 9. 输出投影 → (d_model, 1)
        return w_o_.forward(engine, *attn_out);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// MultiHeadAttention — 无掩码多头注意力（极简子类：沿用 AttentionBase 默认行为）
// ══════════════════════════════════════════════════════════════════════════
class MultiHeadAttention final : public AttentionBase
{
public:
    MultiHeadAttention(std::size_t d_model, std::size_t num_heads,
                       std::size_t seq_len = 0)
        : AttentionBase(d_model, num_heads, seq_len) {}
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

    [[nodiscard]] Result<void> rebuild_encoding(ComputeEngine& engine, std::size_t total_len)
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
        if (!r) return std::unexpected(r.error());
        encoding_cache_ = std::move(*r);
        cached_total_ = total_len;
        return {};
    }

public:
    PositionalEncoding(std::size_t d_model, std::size_t max_len = 5000,
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
        {
            auto r = rebuild_encoding(engine, total_len);
            if (!r) return std::unexpected(r.error());
        }

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
    Linear fc1_;   // GeLU: (d_model → d_ff); SwiGLU: (d_model → 2*d_ff)
    Linear fc2_;   // (d_ff → d_model)
    GeLU  gelu_;
    SwiGLU swiglu_;  // SwiGLU 激活（含 split/merge）
    bool use_swiglu_ = false;

public:
    FeedForward(std::size_t d_model, std::size_t d_ff,
                ActivationType activation = ActivationType::GeLU)
        : fc1_(d_model,
               activation == ActivationType::SwiGLU ? 2 * d_ff : d_ff),
          fc2_(d_ff, d_model),
          swiglu_(d_ff),
          use_swiglu_(activation == ActivationType::SwiGLU) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = fc1_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = fc2_.init(engine); if (!r2) return std::unexpected(r2.error());
        return {};
    }

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

    // 梯度检查点：把模式传播给内部 fc1/fc2/gelu/swiglu
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        fc1_.set_checkpoint_mode(enabled);
        fc2_.set_checkpoint_mode(enabled);
        gelu_.set_checkpoint_mode(enabled);
        swiglu_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        fc1_.clear_cache();
        fc2_.clear_cache();
        gelu_.clear_cache();
        swiglu_.clear_cache();
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = fc1_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto b = fc2_.activation_cache(); r.insert(r.end(), b.begin(), b.end());
        auto g = gelu_.activation_cache(); r.insert(r.end(), g.begin(), g.end());
        auto s = swiglu_.activation_cache(); r.insert(r.end(), s.begin(), s.end());
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        auto h1 = fc1_.forward(engine, input);
        if (!h1) return h1;
        if (use_swiglu_)
        {
            auto h2 = swiglu_.forward(engine, *h1);
            if (!h2) return h2;
            return fc2_.forward(engine, *h2);
        }
        auto h2 = gelu_.forward(engine, *h1);
        if (!h2) return h2;
        return fc2_.forward(engine, *h2);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto b2 = fc2_.backward(engine, grad_output);
        if (!b2) return b2;
        if (use_swiglu_)
        {
            auto bg = swiglu_.backward(engine, *b2);
            if (!bg) return bg;
            return fc1_.backward(engine, *bg);
        }
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

    Tensor residual2_cache_;

public:
    TransformerEncoderLayer(std::size_t d_model, std::size_t num_heads,
                            std::size_t d_ff, std::size_t seq_len = 0)
        : self_attn_(d_model, num_heads, seq_len),
          norm1_(d_model),
          ff_(d_model, d_ff),
          norm2_(d_model) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = self_attn_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = norm1_.init(engine); if (!r2) return std::unexpected(r2.error());
        auto r3 = ff_.init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = norm2_.init(engine); if (!r4) return std::unexpected(r4.error());
        return {};
    }

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

    // 梯度检查点：把模式传播给内部注意力/归一化/FFN
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        self_attn_.set_checkpoint_mode(enabled);
        norm1_.set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
        norm2_.set_checkpoint_mode(enabled);
    }

    [[nodiscard]] bool recompute_supported() const override { return true; }

    [[nodiscard]] Result<Tensor> forward_recompute(
        ComputeEngine& engine, const Tensor& saved_input) override
    {
        // 临时关闭本层及其子层的 checkpoint 模式，使 forward 重建缓存
        set_checkpoint_mode(false);
        auto r = forward(engine, saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    void clear_cache() override
    {
        self_attn_.clear_cache();
        norm1_.clear_cache();
        ff_.clear_cache();
        norm2_.clear_cache();
        residual2_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = self_attn_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto n1 = norm1_.activation_cache(); r.insert(r.end(), n1.begin(), n1.end());
        auto f = ff_.activation_cache(); r.insert(r.end(), f.begin(), f.end());
        auto n2 = norm2_.activation_cache(); r.insert(r.end(), n2.begin(), n2.end());
        if (residual2_cache_.valid()) r.emplace_back(residual2_cache_);
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        // x1 = LN₁(input)
        auto n1 = norm1_.forward(engine, input);
        if (!n1) return n1;

        // a = SelfAttn(x1)
        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        // r2 = input + a
        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r2) return std::unexpected(r2.error());
        Tensor res2 = std::move(*r2);
        if (!checkpoint_mode_)
            residual2_cache_ = res2;

        // x2 = LN₂(res2)
        auto n2 = norm2_.forward(engine, res2);
        if (!n2) return n2;

        // f = FFN(x2)
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        // out = res2 + f
        return engine.elementwise_binary(BinaryOp::Add, res2, *f);
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
    TransformerEncoder(std::size_t d_model, std::size_t num_heads,
                       std::size_t d_ff, std::size_t num_layers,
                       std::size_t num_patches)
        : d_model_(d_model), num_patches_(num_patches),
          inv_num_patches_(Scalar{1} / static_cast<Scalar>(num_patches)),
          // PE 启用 tiling：每个样本独立使用 (d_model, num_patches) 的编码
          pos_encoding_(d_model, num_patches, num_patches)
    {
        // 所有 EncoderLayer 传入 seq_len=num_patches，启用 MHA 批量化路径
        for (std::size_t i = 0; i < num_layers; ++i)
            layers_.emplace_back(d_model, num_heads, d_ff, num_patches);
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        for (auto& layer : layers_)
        {
            auto r = layer.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        // 预创建 ones_row_ (1, num_patches) 全1，用于 backward 广播
        Matrix ones_cpu(1, num_patches_, Scalar{1});
        auto or_t = engine.from_matrix(ones_cpu);
        if (!or_t) return std::unexpected(or_t.error());
        ones_row_ = std::move(*or_t);
        return {};
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
    PatchEmbedding(std::size_t img_size, std::size_t patch_size,
                   std::size_t d_model)
        : img_size_(img_size), patch_size_(patch_size),
          grid_size_(img_size / patch_size),
          num_patches_(grid_size_ * grid_size_),
          patch_dim_(patch_size * patch_size),
          d_model_(d_model),
          projection_(patch_dim_, d_model)
    {
        NN_ASSERT(img_size % patch_size == 0,
                  "PatchEmbedding: img_size must be divisible by patch_size");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        return projection_.init(engine);
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

// ── 扁平化注意力掩码构建（纯函数，可独立单测） ─────────────────────────
//
// 返回 (BH*seq, seq) 的掩码，其中 BH = batch*num_heads。
// 行 (b, h, t) 位于 (b*num_heads + h)*seq + t，列 j = key 位置（0..seq-1）。
//
// doc_ids（可选，batch-major：b*seq+t → 文档 id）非空时启用块对角文档感知：
//   - 跨文档位置禁止相互注意（值 = -inf）
//   - 同文档内仍施加因果掩码（未来位置 j>i 为 -inf）
// doc_ids 为空时退化为纯因果掩码（与现有行为一致）。
//
// use_alibi 时，在允许注意的位置叠加 ALiBi 线性偏置 -m_h*(i-j)。
[[nodiscard]] Matrix build_attention_mask(
    std::size_t batch, std::size_t seq_len, std::size_t num_heads,
    bool use_alibi, std::span<const Scalar> slopes,
    std::span<const std::size_t> doc_ids = {})
{
    NN_ASSERT(!use_alibi || slopes.size() >= num_heads,
              "build_attention_mask: slopes too small for num_heads");
    NN_ASSERT(doc_ids.empty() || doc_ids.size() >= batch * seq_len,
              "build_attention_mask: doc_ids too small for batch*seq_len");

    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    const std::size_t BH = batch * num_heads;
    Matrix mask(BH * seq_len, seq_len);
    for (std::size_t b = 0; b < batch; ++b)
    {
        for (std::size_t h = 0; h < num_heads; ++h)
        {
            const std::size_t head_idx = b * num_heads + h;
            const Scalar slope = use_alibi ? slopes[h] : Scalar{0};

            for (std::size_t i = 0; i < seq_len; ++i)
            {
                for (std::size_t j = 0; j < seq_len; ++j)
                {
                    bool allowed = (j <= i);
                    if (allowed && !doc_ids.empty())
                    {
                        // 块对角文档感知：仅同文档内的因果位置可注意
                        allowed = (doc_ids[b * seq_len + i]
                                   == doc_ids[b * seq_len + j]);
                    }
                    if (allowed)
                    {
                        const Scalar bias = use_alibi
                            ? -slope * static_cast<Scalar>(i - j)
                            : Scalar{0};
                        mask.set_value_unchecked(head_idx * seq_len + i, j, bias);
                    }
                    else
                    {
                        mask.set_value_unchecked(head_idx * seq_len + i, j, neg_inf);
                    }
                }
            }
        }
    }
    return mask;
}

// ══════════════════════════════════════════════════════════════════════════
// CausalSelfAttention — 因果自注意力（继承 AttentionBase，仅重写掩码钩子）
//
// 通过 PosEncodingType 参数支持两种掩码模式：
//   - Learned / Sinusoidal: 仅施加因果掩码（mask[i][j] = 0 if j<=i, -inf if j>i）
//   - ALiBi: 因果掩码 + ALiBi 线性偏置
//
// ALiBi (Attention with Linear Biases) 原理：
//   不使用位置嵌入，而是在注意力分数上添加线性偏置：
//   attention_score = Q*K^T + bias
//   其中 bias[i][j] = -m_h * (i - j) for j <= i
//   斜率 m_h = 2^(-8h/H)，h 是头索引，H 是总头数
//
// 优点（ALiBi 模式）：
//   1. 无需位置嵌入，减少参数
//   2. 天然支持长度外推（训练短序列，推理长序列）
//   3. 计算开销极小（仅添加预计算的偏置）
//
// 算法差异（相对于 AttentionBase）：
//   在 forward 的 scale 之后、softmax 之前，施加预计算的掩码：
//     S += mask (batch*H*seq, seq) — 因果掩码（按 batch*H 平铺，ALiBi 模式下含线性偏置）
//   backward 无需特殊处理（掩码为常数，softmax.backward 已处理梯度穿透）
//
//   seq_len 由构造函数指定，batch = input.cols() / seq_len 在 forward 时推断。
//   seq_len=0 表示单样本模式（cols 即 seq），保持向后兼容。
// ══════════════════════════════════════════════════════════════════════════
class CausalSelfAttention final : public AttentionBase
{
private:
    bool use_alibi_;        // true = ALiBi 模式（因果掩码 + 线性偏置）

    // 掩码缓存（平铺为 batch*H*seq × seq 以匹配堆叠 scores 布局）
    // ALiBi 模式下包含因果掩码 + 线性偏置；普通模式下仅因果掩码
    Tensor mask_cache_;
    std::size_t mask_cached_batch_ = 0;  // 缓存键：batch（独立字段，避免位打包溢出）
    std::size_t mask_cached_seq_ = 0;    // 缓存键：seq_len

    // 两趟式（M6）组合偏置小张量缓存（AttnBias 描述子指向它们）
    Tensor slopes_cache_;   // (1, num_heads) ALiBi 按头斜率（惰性构建）
    Tensor doc_ids_cache_;  // (1, batch*seq) 每位置文档 id（每步重建）

    // ALiBi 斜率：m_h = 2^(-8h/H)（仅 use_alibi_ = true 时使用）
    std::vector<Scalar> slopes_;

    // 文档感知掩码：当前 step 每样本的文档 id（batch-major b*seq+t → doc id）
    // 非空时启用块对角文档感知（跨文档禁止注意），每步重建（不缓存）。
    std::vector<std::size_t> doc_ids_;
    bool has_doc_ids_ = false;

    void init_slopes_()
    {
        slopes_.resize(num_heads_);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            // m_h = 2^(-8h/H)
            slopes_[h] = std::pow(Scalar{2}, -Scalar{8} * h / num_heads_);
        }
    }

    [[nodiscard]] Result<void> ensure_mask_(ComputeEngine& engine, std::size_t batch, std::size_t seq_len)
    {
        // 用 (batch, seq_len) 作为缓存键
        if (mask_cached_batch_ == batch && mask_cached_seq_ == seq_len)
            return {};

        // 纯因果掩码：委托 build_attention_mask（doc_ids 为空）
        auto mask = build_attention_mask(batch, seq_len, num_heads_,
                                         use_alibi_, slopes_);
        auto r = engine.from_matrix(mask);
        if (!r) return std::unexpected(r.error());
        mask_cache_ = std::move(*r);
        mask_cached_batch_ = batch;
        mask_cached_seq_ = seq_len;
        return {};
    }

public:
    // 设置当前 step 的每样本文档 id（batch-major b*seq+t → doc id）。
    // 传入空 span 会清除文档感知，退化为纯因果掩码。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty())
        {
            doc_ids_.clear();
            has_doc_ids_ = false;
            return;
        }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

protected:
    // 重写掩码钩子：在 scale 之后、softmax 之前施加因果/ALiBi 掩码
    [[nodiscard]] Result<Tensor> apply_mask_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t batch, std::size_t seq) override
    {
        if (has_doc_ids_)
        {
            // 文档感知：doc_ids_ 每 batch 变化，掩码不可缓存，每步重建
            // 块对角（跨文档 -inf）∧ 因果（同文档内未来 -inf）
            auto m = build_attention_mask(batch, seq, num_heads_,
                                          use_alibi_, slopes_, doc_ids_);
            auto mt = engine.from_matrix(m);
            if (!mt) return std::unexpected(mt.error());
            return engine.elementwise_binary(BinaryOp::Add, scores, *mt);
        }
        {
            auto r = ensure_mask_(engine, batch, seq);
            if (!r) return std::unexpected(r.error());
        }
        return engine.elementwise_binary(BinaryOp::Add, scores, mask_cache_);
    }

    // 重写两趟式决策：组合式 AttnBias 描述子统一 因果/ALiBi/doc_ids（及其组合），
    // 全部走两趟式（不物化 (BH·seq, seq) 得分矩阵），不再回退旧路径。
    [[nodiscard]] Result<TwoPassMask> two_pass_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq) override
    {
        AttnBias bias;
        bias.causal = true;
        bias.num_heads = num_heads_;
        if (use_alibi_)
        {
            if (!slopes_cache_.valid())
            {
                Matrix s(1, num_heads_);
                for (std::size_t h = 0; h < num_heads_; ++h)
                    s.set_value_unchecked(0, h, slopes_[h]);
                auto t = engine.from_matrix(s);
                if (!t) return std::unexpected(t.error());
                slopes_cache_ = std::move(*t);
            }
            bias.slopes = &slopes_cache_;
        }
        if (has_doc_ids_)
        {
            // 文档感知：doc_ids_ 每 batch 变化，每步重建（小张量 O(batch*seq)）
            Matrix d(1, batch * seq);
            for (std::size_t p = 0; p < batch * seq; ++p)
                d.set_value_unchecked(0, p, static_cast<Scalar>(doc_ids_[p]));
            auto t = engine.from_matrix(d);
            if (!t) return std::unexpected(t.error());
            doc_ids_cache_ = std::move(*t);
            bias.doc_ids = &doc_ids_cache_;
        }
        return TwoPassMask{/*use_two_pass=*/true, bias};
    }

    // 重写增量推理掩码钩子：ALiBi 模式下施加线性距离偏置。
    // 普通因果模式（use_alibi_ = false）：no-op，因 cache 只含前文，因果天然满足。
    // ALiBi 模式：scores[h, j] += -slope[h] * (cur_len - j)，j ∈ [0, cur_len]。
    //   scores 布局: (H, new_len)，new_len = cur_len + 1
    //   query 在位置 cur_len，key 在位置 j，距离 = cur_len - j
    [[nodiscard]] Result<Tensor> apply_mask_step_(
        ComputeEngine& engine, Tensor&& scores,
        std::size_t cur_len) override
    {
        if (!use_alibi_) return std::move(scores);

        const std::size_t new_len = cur_len + 1;
        // 构建 ALiBi 偏置矩阵 (H, new_len)
        Matrix bias(num_heads_, new_len);
        for (std::size_t h = 0; h < num_heads_; ++h)
        {
            const Scalar slope = slopes_[h];
            for (std::size_t j = 0; j < new_len; ++j)
            {
                // j ∈ [0, cur_len]，距离 cur_len - j ∈ [0, cur_len]
                const std::size_t dist = cur_len - j;
                bias.set_value_unchecked(h, j,
                    -slope * static_cast<Scalar>(dist));
            }
        }
        auto bias_t = engine.from_matrix(bias);
        if (!bias_t) return std::unexpected(bias_t.error());
        return engine.elementwise_binary(BinaryOp::Add, scores, *bias_t);
    }

public:
    CausalSelfAttention(std::size_t d_model, std::size_t num_heads,
                       std::size_t /*max_len*/ = 1024,
                       std::size_t seq_len = 0,
                       PosEncodingType pos_enc = PosEncodingType::Learned)
        : AttentionBase(d_model, num_heads, seq_len, pos_enc),
          use_alibi_(pos_enc == PosEncodingType::ALiBi)
    {
        if (use_alibi_)
            init_slopes_();
    }
};

// ══════════════════════════════════════════════════════════════════════════
// GPTBlock — Pre-Norm 解码器块
//
// 通过 pos_enc 参数选择注意力模式（Learned/Sinusoidal/ALiBi）：
//   x = x + CausalSelfAttn(LN₁(x), pos_enc)
//   x = x + FFN(LN₂(x))
// ══════════════════════════════════════════════════════════════════════════
class GPTBlock final : public Layer
{
private:
    CausalSelfAttention self_attn_;
    std::unique_ptr<Layer> norm1_;
    FeedForward ff_;
    std::unique_ptr<Layer> norm2_;

    Tensor residual2_cache_;

    // ── activation offload（L1-offload）状态 ──
    bool offload_enabled_ = false;      // GPTModel 是否启用 offload
    bool offloaded_ = false;            // 当前激活是否已导出到 host
    Tensor offload_slab_;               // 持久 host-visible 缓冲（跨 step 复用）
    std::vector<TensorRef> offload_refs_;        // 需 offload 的缓存成员引用（稳定地址）
    std::vector<std::pair<std::size_t, std::size_t>> offload_shapes_;  // 各缓存形状
    std::vector<std::size_t> offload_offsets_;   // 各激活在 slab 中的 float 偏移

public:
    GPTBlock(std::size_t d_model, std::size_t num_heads,
             std::size_t d_ff, std::size_t max_len = 1024,
             std::size_t seq_len = 0,
             PosEncodingType pos_enc = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm)
        : self_attn_(d_model, num_heads, max_len, seq_len, pos_enc),
          norm1_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation),
          norm2_(make_norm_layer(d_model, norm_type)) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = self_attn_.init(engine); if (!r1) return std::unexpected(r1.error());
        if (norm1_) { auto r = norm1_->init(engine); if (!r) return std::unexpected(r.error()); }
        auto r2 = ff_.init(engine); if (!r2) return std::unexpected(r2.error());
        if (norm2_) { auto r = norm2_->init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = self_attn_.parameters();
        auto n1 = norm1_->parameters();
        auto f  = ff_.parameters();
        auto n2 = norm2_->parameters();
        p.insert(p.end(), n1.begin(), n1.end());
        p.insert(p.end(), f.begin(), f.end());
        p.insert(p.end(), n2.begin(), n2.end());
        return p;
    }

    // 文档感知：把每样本文档 id 转发给内部自注意力（用于块对角掩码）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        self_attn_.set_doc_ids(ids);
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto g = self_attn_.param_gradients();
        auto gn1 = norm1_->param_gradients();
        auto gf  = ff_.param_gradients();
        auto gn2 = norm2_->param_gradients();
        g.insert(g.end(), gn1.begin(), gn1.end());
        g.insert(g.end(), gf.begin(), gf.end());
        g.insert(g.end(), gn2.begin(), gn2.end());
        return g;
    }

    // 梯度检查点：把模式传播给内部注意力/归一化/FFN
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        self_attn_.set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
    }

    // GPTBlock 可作为“重计算单元”：从保存的块输入重算 forward 重建缓存
    [[nodiscard]] bool recompute_supported() const override { return true; }

    [[nodiscard]] Result<Tensor> forward_recompute(
        ComputeEngine& engine, const Tensor& saved_input) override
    {
        // 临时关闭本块及其子层的 checkpoint 模式，使 forward 重建缓存
        set_checkpoint_mode(false);
        auto r = forward(engine, saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    void clear_cache() override
    {
        self_attn_.clear_cache();
        norm1_->clear_cache();
        ff_.clear_cache();
        norm2_->clear_cache();
        residual2_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = self_attn_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto n1 = norm1_->activation_cache(); r.insert(r.end(), n1.begin(), n1.end());
        auto f = ff_.activation_cache(); r.insert(r.end(), f.begin(), f.end());
        auto n2 = norm2_->activation_cache(); r.insert(r.end(), n2.begin(), n2.end());
        if (residual2_cache_.valid()) r.emplace_back(residual2_cache_);
        return r;
    }

    // ── activation offload（L1-offload）────────────────────────────────
    void set_offload_enabled(bool enabled)
    {
        offload_enabled_ = enabled;
        if (!enabled) offload_slab_ = Tensor{};  // 释放持久缓冲
    }
    [[nodiscard]] bool offload_enabled() const noexcept { return offload_enabled_; }

    // 导出：把本块 backward 所需的中间激活逐个写入持久 host slab（释放 GPU 显存）。
    // offload_refs_ 记录各缓存成员地址（forward 后成员地址稳定），导入时复用。
    [[nodiscard]] Result<void> export_activations(ComputeEngine& engine)
    {
        if (!offload_enabled_ || offloaded_) return {};
        offload_refs_ = activation_cache();
        offload_offsets_.clear();
        offload_shapes_.clear();
        // 一次性明细诊断（仅首个块首次导出打印一次）
        {
            static bool dumped = false;
            if (!dumped)
            {
                std::size_t sum = 0;
                for (auto& ref : offload_refs_)
                {
                    if (!ref.get().valid()) continue;
                    sum += ref.get().size();
                    std::cout << "[offload-diag] tensor " << ref.get().rows() << "x"
                              << ref.get().cols() << " = " << (ref.get().size() * 4 / (1024*1024)) << "MB\n";
                }
                std::cout << "[offload-diag] block0 slab sum = " << (sum * 4 / (1024*1024)) << "MB\n";
                dumped = true;
            }
        }
        // 惰性创建持久 slab（大小 = 本块激活总 float 数，跨 step 复用）
        if (!offload_slab_.valid())
        {
            std::size_t total = 0;
            for (auto& ref : offload_refs_)
                if (ref.get().valid()) total += ref.get().size();
            // 无有效激活可导出（如混合模式下 checkpoint 块 forward 不驻留缓存）
            if (total == 0)
            {
                offloaded_ = false;
                return {};
            }
            auto slab = engine.create_offload_buffer(total);
            if (!slab) return std::unexpected(slab.error());
            offload_slab_ = std::move(*slab);
        }
        std::size_t offset = 0;
        for (auto& ref : offload_refs_)
        {
            if (!ref.get().valid()) continue;
            auto r = engine.offload_save(offload_slab_, offset, ref.get());
            if (!r) return std::unexpected(r.error());
            offload_shapes_.push_back({ref.get().rows(), ref.get().cols()});
            offload_offsets_.push_back(offset);
            offset += ref.get().size();
            ref.get() = Tensor{};  // 释放 GPU 版（数据已在 host slab）
        }
        offloaded_ = true;
        return {};
    }

    // 导入：从 host slab 恢复激活到缓存成员（backward 前调用，替代重计算）
    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        if (!offloaded_) return {};
        for (std::size_t i = 0; i < offload_offsets_.size(); ++i)
        {
            auto t = engine.offload_restore(offload_slab_, offload_offsets_[i],
                                            offload_shapes_[i].first,
                                            offload_shapes_[i].second);
            if (!t) return std::unexpected(t.error());
            offload_refs_[i].get() = std::move(*t);
        }
        offloaded_ = false;
        return {};
    }

    // 实际 slab 字节数（诊断用；未创建时 0）
    [[nodiscard]] std::size_t offload_slab_bytes() const noexcept
    {
        return offload_slab_.valid() ? offload_slab_.size() * sizeof(float) : 0;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;

        auto a = self_attn_.forward(engine, *n1);
        if (!a) return a;

        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r2) return std::unexpected(r2.error());
        Tensor res2 = std::move(*r2);
        if (!checkpoint_mode_)
            residual2_cache_ = res2;

        auto n2 = norm2_->forward(engine, res2);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return engine.elementwise_binary(BinaryOp::Add, res2, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // activation offload：从 host 恢复激活再反向（替代重计算）
        if (offloaded_)
        {
            auto im = import_activations(engine);
            if (!im) return std::unexpected(im.error());
        }

        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;

        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());

        auto b_sa = self_attn_.backward(engine, *grad_r1);
        if (!b_sa) return b_sa;
        auto b_n1 = norm1_->backward(engine, *b_sa);
        if (!b_n1) return b_n1;

        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }

    // ── 增量推理（KV cache）──────────────────────────────────────────
    // Pre-Norm 单 token 前向：
    //   x = x_new + CausalSelfAttn(LN₁(x_new), kv_cache)
    //   x = x + FFN(LN₂(x))
    // 输入: x_new (d_model, 1)
    // 输出: (d_model, 1)
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        const Tensor& x_new,
        Tensor& k_cache,
        Tensor& v_cache,
        std::size_t cur_len)
    {
        auto n1 = norm1_->forward(engine, x_new);
        if (!n1) return n1;

        auto a = self_attn_.forward_step(engine, *n1, k_cache, v_cache, cur_len);
        if (!a) return a;

        auto r2 = engine.elementwise_binary(BinaryOp::Add, x_new, *a);
        if (!r2) return std::unexpected(r2.error());

        auto n2 = norm2_->forward(engine, *r2);
        if (!n2) return n2;

        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        return engine.elementwise_binary(BinaryOp::Add, *r2, *f);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════════════
// PositionEncoder — 位置编码抽象基类
//
// 把 GPTModel 中原本按 PosEncodingType 散落的 if-else 位置编码逻辑
// （可学习 / 正弦波 / 无）抽离为独立的多态层次，GPTModel 通过基类指针使用，
// 消除 use_pos_emb_ / pos_emb_learnable_ 等标志位分支的耦合。
//
// 接口：
//   - apply(engine, token_emb_T, batch, seq)   全量前向，返回已加位置信息的 x
//   - apply_step(engine, x, pos)               增量单 token 前向
//   - backward(engine, grad_T, batch, seq)     累计位置梯度（仅可学习有意义）
//   - parameters() / param_gradients()         可学习参数（仅 Learned 返回非空）
//
// 注意：ALiBi / RoPE 的位置信息由注意力层注入（CausalSelfAttention 线性偏置 /
//        AttentionBase 的 RotaryEmbedding），故其编码器为 no-op。
// ══════════════════════════════════════════════════════════════════════════
class PositionEncoder
{
public:
    virtual ~PositionEncoder() = default;

    // 全量前向：token_emb_T 为 (d_model, batch*seq)，返回 x = token_emb_T (+ pos_emb)
    [[nodiscard]] virtual Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& token_emb_T,
        std::size_t batch, std::size_t seq) = 0;

    // 增量前向：x 为 (d_model, 1)，返回 x + pos_emb[pos]
    [[nodiscard]] virtual Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& x, std::size_t pos) = 0;

    // 反向：累计位置梯度到 grad_pos_emb_（默认 no-op）
    [[nodiscard]] virtual Result<void> backward(
        ComputeEngine& engine, const Tensor& grad_T,
        std::size_t batch, std::size_t seq) = 0;

    // 初始化（引擎相关操作，如创建张量等）
    [[nodiscard]] virtual Result<void> init(ComputeEngine& /*engine*/) { return {}; }

    [[nodiscard]] virtual std::vector<TensorRef> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<TensorRef> param_gradients() { return {}; }
};

// 加性位置编码基类（Learned / Sinusoidal 共用）：
// 通过 pos_emb_ 张量按位置 gather 并加到 token 嵌入上。
class AdditivePositionEncoder : public PositionEncoder
{
protected:
    bool learnable_ = false;

    Tensor pos_emb_;        // (seq_len, d_model)
    Tensor grad_pos_emb_;   // (seq_len, d_model)，仅 learnable_ 有效

    // pos_indices 缓存（避免每 step 重建）— (total, 1) 值为 [0,..,0,1,..,1,...,seq-1,..]
    Tensor pos_indices_cache_;
    std::size_t pos_indices_batch_ = 0;  // 缓存键：batch_size
    std::size_t pos_indices_seq_ = 0;    // 缓存键：seq_len

    // 用给定的位置编码矩阵初始化 pos_emb_（learnable 时额外分配梯度）
    [[nodiscard]] Result<void> init_(ComputeEngine& engine, Matrix&& pe, bool learnable)
    {
        const std::size_t rows = pe.rows();
        const std::size_t cols = pe.cols();
        learnable_ = learnable;
        auto pe_r = engine.from_matrix(pe);
        if (!pe_r) return std::unexpected(pe_r.error());
        pos_emb_ = std::move(*pe_r);
        if (learnable_)
        {
            grad_pos_emb_ = engine.create_tensor(rows, cols);
            auto r = engine.zero(grad_pos_emb_);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // 确保 pos_indices 缓存有效（batch-major：i = b*seq + t → position=t）
    [[nodiscard]] Result<void> ensure_pos_indices_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        if (pos_indices_batch_ == batch && pos_indices_seq_ == seq)
            return {};
        Matrix pidx_m(batch * seq, 1);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t t = 0; t < seq; ++t)
                pidx_m.set_value_unchecked(b * seq + t, 0,
                    static_cast<Scalar>(t));
        auto pidx_t = engine.from_matrix(pidx_m);
        if (!pidx_t) return std::unexpected(pidx_t.error());
        pos_indices_cache_ = std::move(*pidx_t);
        pos_indices_batch_ = batch;
        pos_indices_seq_ = seq;
        return {};
    }

public:
    AdditivePositionEncoder() = default;

    [[nodiscard]] bool learnable() const noexcept { return learnable_; }

    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& engine, const Tensor& token_emb_T,
        std::size_t batch, std::size_t seq) override
    {
        auto ci = ensure_pos_indices_(engine, batch, seq);
        if (!ci) return std::unexpected(ci.error());
        auto pos_gathered = engine.gather_rows(pos_emb_, pos_indices_cache_);
        if (!pos_gathered) return std::unexpected(pos_gathered.error());
        auto pos_T = engine.transpose(*pos_gathered);
        if (!pos_T) return std::unexpected(pos_T.error());
        auto x_with_pos = engine.elementwise_binary(BinaryOp::Add, token_emb_T, *pos_T);
        if (!x_with_pos) return std::unexpected(x_with_pos.error());
        return std::move(*x_with_pos);
    }

    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& engine, const Tensor& x, std::size_t pos) override
    {
        Matrix pos_m(1, 1);
        pos_m.set_value_unchecked(0, 0, static_cast<Scalar>(pos));
        auto pos_t = engine.from_matrix(pos_m);
        if (!pos_t) return std::unexpected(pos_t.error());
        auto pos_emb_g = engine.gather_rows(pos_emb_, *pos_t);
        if (!pos_emb_g) return std::unexpected(pos_emb_g.error());
        auto pos_T = engine.transpose(*pos_emb_g);
        if (!pos_T) return std::unexpected(pos_T.error());
        auto x_wp = engine.elementwise_binary(BinaryOp::Add, x, *pos_T);
        if (!x_wp) return std::unexpected(x_wp.error());
        return std::move(*x_wp);
    }

    [[nodiscard]] Result<void> backward(
        ComputeEngine& engine, const Tensor& grad_T,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    {
        if (!learnable_) return {};
        // pos_indices 缓存由 apply() 建立，backward 直接复用（batch/seq 一致）。
        auto pr = engine.scatter_add_rows(grad_pos_emb_, pos_indices_cache_, grad_T);
        if (!pr) return std::unexpected(pr.error());
        return {};
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        if (!learnable_) return {};
        return { pos_emb_ };
    }
    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        if (!learnable_) return {};
        return { grad_pos_emb_ };
    }
};

// 可学习位置编码（GPT 默认）：N(0, 0.02) 随机初始化。
// 与 token_emb_ 共享同一 rng 序列（保持与旧实现完全一致的可复现性）。
class LearnedPositionEncoder final : public AdditivePositionEncoder
{
    std::size_t d_model_;
    std::size_t seq_len_;
    // 自持 RNG（seed=42、N(0,0.02)）：init() 延迟到构造之后调用，
    // 不能持有指向构造函数局部变量的非拥有指针（会悬空）。
    std::mt19937_64 rng_{42};
    std::normal_distribution<Scalar> dist_{0.0, 0.02};

public:
    LearnedPositionEncoder(std::size_t d_model, std::size_t seq_len)
        : d_model_(d_model), seq_len_(seq_len) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix pe(seq_len_, d_model_);
        auto pe_s = pe.span();
        for (std::size_t i = 0; i < pe.size(); ++i) pe_s[i] = dist_(rng_);
        return init_(engine, std::move(pe), /*learnable=*/true);
    }
};

// 正弦波固定位置编码（冻结，不参与训练）：
//   PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
class SinusoidalPositionEncoder final : public AdditivePositionEncoder
{
    std::size_t d_model_;
    std::size_t seq_len_;

public:
    SinusoidalPositionEncoder(std::size_t d_model, std::size_t seq_len)
        : d_model_(d_model), seq_len_(seq_len) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix pe(seq_len_, d_model_);
        auto pe_s = pe.span();
        for (std::size_t pos = 0; pos < seq_len_; ++pos)
            for (std::size_t i = 0; i < d_model_; ++i)
            {
                Scalar angle = static_cast<Scalar>(pos) /
                    std::pow(Scalar{10000}, static_cast<Scalar>(2 * (i / 2)) / static_cast<Scalar>(d_model_));
                pe_s[pos * d_model_ + i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
        return init_(engine, std::move(pe), /*learnable=*/false);
    }
};

// 无位置编码（ALiBi / RoPE）：位置信息由注意力层注入，此处为 no-op。
class NoPositionEncoder final : public PositionEncoder
{
public:
    [[nodiscard]] Result<Tensor> apply(
        ComputeEngine& /*engine*/, const Tensor& token_emb_T,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    { return token_emb_T; }

    [[nodiscard]] Result<Tensor> apply_step(
        ComputeEngine& /*engine*/, const Tensor& x, std::size_t /*pos*/) override
    { return x; }

    [[nodiscard]] Result<void> backward(
        ComputeEngine& /*engine*/, const Tensor& /*grad_T*/,
        std::size_t /*batch*/, std::size_t /*seq*/) override
    { return {}; }
};

// GPTModel — Decoder-only Transformer 语言模型
//
// 算法（只在此处，不在 Engine/Shader）：
//   组件: TokenEmb [+ PosEmb] + N × GPTBlock + LayerNorm + LM Head
//   输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
//   输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
//
// 通过 PosEncodingType 参数支持三种位置编码模式：
//   - Learned:    可学习位置嵌入（默认）
//   - Sinusoidal: 正弦波固定位置编码（冻结）
//   - ALiBi:      无位置嵌入，通过 CausalSelfAttention 的线性偏置注入位置信息
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

    // 嵌入
    Tensor token_emb_;       // (vocab_size, d_model)
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;  // 位置编码（多态：Learned/Sinusoidal/无）

    std::vector<GPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    // 反向缓存
    Tensor stored_tokens_tensor_;          // token IDs 的 Tensor 版本 (total, 1)
    std::size_t batch_size_ = 0;

    // 文档感知：当前 step 每样本文档 id（batch-major b*seq+t → doc id），
    // 由调用方在 forward 前 set_doc_ids 设置，转发给各 GPTBlock。
    std::vector<std::size_t> doc_ids_;

    // batch 录制粒度：每隔 flush_interval_ 个 Transformer block 提交一次
    // 0 = 不在 block 间 flush（默认），>0 = 每 N 个 block flush 一次
    std::size_t flush_interval_ = 0;

    // 梯度检查点（激活重计算 L1）：每隔 checkpoint_every_ 个 GPTBlock 保存一次
    // 块输入，backward 时重算以省去驻留整层激活。0 = 不启用。
    std::size_t checkpoint_every_ = 0;
    std::vector<Tensor> checkpoint_inputs_;  // 各 checkpoint 块的输入 (d_model, batch*seq)

    // activation offload（L1-offload）：把每块内部激活搬 host-visible，backward 拷回
    bool activation_offload_ = false;

public:
    GPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
             std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
             PosEncodingType pos_enc_type = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, seq_len,
                                 pos_enc_type, activation, norm_type);

        // 初始化位置编码器（Learned / Sinusoidal / ALiBi / RoPE）
        switch (pos_enc_type)
        {
            case PosEncodingType::Learned:
                pos_encoder_ = std::make_unique<LearnedPositionEncoder>(
                    d_model, seq_len);
                break;
            case PosEncodingType::Sinusoidal:
                pos_encoder_ = std::make_unique<SinusoidalPositionEncoder>(
                    d_model, seq_len);
                break;
            default:  // ALiBi / RoPE：位置信息由注意力层注入
                pos_encoder_ = std::make_unique<NoPositionEncoder>();
                break;
        }
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // 初始化 token_emb_
        Matrix te(vocab_size_, d_model_);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);

        auto te_r = engine.from_matrix(te);
        if (!te_r) return std::unexpected(te_r.error());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_);
        { auto r1 = engine.zero(grad_token_emb_); if (!r1) return std::unexpected(r1.error()); }

        // 初始化子层
        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        for (auto& block : blocks_)
        {
            auto r = block.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        if (ln_f_)
        {
            auto r = ln_f_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        { auto r = lm_head_.init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }

    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        auto gp = pos_encoder_->param_gradients();
        g.insert(g.end(), gp.begin(), gp.end());
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_->param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    // 文档感知：设置当前 step 每样本文档 id（batch-major b*seq+t → doc id）。
    // 传入空 span 清除文档感知（退化为纯因果）。在 forward 前调用。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq_len = input.rows();
        batch_size_ = input.cols();
        // ── 1. gather 所有 token 的 embedding（统一采用 batch-major 列序） ──
        // 下方注意力（AttentionBase::forward 的 rearrange_3d + batched_matmul + 因果掩码）
        // 假定扁平列为 batch-major（b*seq + t）。因此先把输入 (seq, batch) 转置为
        // (batch, seq)，使 gather_rows 的 flat 序即为 batch-major：i = b*seq + t。
        auto input_T = engine.transpose(input);   // (batch, seq)，flat 索引 = b*seq+t
        if (!input_T) return std::unexpected(input_T.error());

        // gather_rows(token_emb_, input_T) → (batch*seq, d_model)
        //   row i = token_emb[input_T[i]], i 是 batch-major 索引 b*seq+t
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());

        // ── 2. 保存 token IDs 的 Tensor 拷贝（供 backward 的 scatter_add_rows） ──
        // 用 batch-major 序的 input_T，使 scatter 行号与 grad_T（transpose(grad_x)）对齐。
        // 全程 GPU：clone 在 GPU 内执行，无 PCIe 传输
        auto st_t = engine.clone(*input_T);
        if (!st_t) return std::unexpected(st_t.error());
        stored_tokens_tensor_ = std::move(*st_t);

        // ── 3. 构造 x: (d_model, batch*seq)（batch-major 列序） ──
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        // ── 3. 施加位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE 为 no-op） ──
        auto x_result = pos_encoder_->apply(engine, *all_T, batch_size_, seq_len);
        if (!x_result) return std::unexpected(x_result.error());
        // ── 4. 通过 Transformer 块（全批量化，无 per-sample 循环） ──
        Tensor x = std::move(*x_result);
        checkpoint_inputs_.clear();
        const bool ckpt = (checkpoint_every_ > 0);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            // 文档感知：把本 step 每样本文档 id 传给各 block 的注意力
            blocks_[bi].set_doc_ids(doc_ids_);
            // 梯度检查点：每 checkpoint_every_ 个块保存一次输入；
            // 该块及其子层以 checkpoint 模式运行（不驻留中间激活）
            if (ckpt && (bi % checkpoint_every_ == 0))
            {
                auto save = engine.clone(x);
                if (!save) return std::unexpected(save.error());
                checkpoint_inputs_.push_back(std::move(*save));
                blocks_[bi].set_checkpoint_mode(true);
            }
            else
            {
                blocks_[bi].set_checkpoint_mode(false);
            }
            auto r = blocks_[bi].forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
            // activation offload：forward 后把本块内部激活搬 host-visible，释放 GPU 显存。
            // 混合模式（offload + checkpoint 共存）：checkpoint 块 forward 不驻留激活
            // （checkpoint_mode=true），无可导出的缓存，必须跳过 export（否则会创建
            // 空 slab 而失败）；非 checkpoint 块才导出。
            if (activation_offload_ && !blocks_[bi].checkpoint_mode())
            {
                auto ex = blocks_[bi].export_activations(engine);
                if (!ex) return std::unexpected(ex.error());
            }
            // 按间隔 flush，将大录制拆分为多个小提交（防 TDR）
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // ── 5. 最终 LayerNorm/RMSNorm ──
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        x = std::move(*ln);

        // ── 6. LM Head → (vocab_size, seq*batch) batch-major ──
        auto lm_out = lm_head_.forward(engine, x);
        return lm_out;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;

        // ⚠ 注意：不用在这里 zero grad_token_emb_！
        //   grad_token_emb_ 已注册到优化器的 param_gradients() 中，
        //   由优化器的 zero_grad() 统一清零。这里如果额外清零会破坏
        //   梯度积累（accum_steps > 1 时前几轮的梯度信号全部丢失）。
        //   位置编码梯度同理（pos_encoder_->param_gradients() 已注册，勿在此清零）。
        // (void)engine.zero(grad_token_emb_);

        // ── 1. LM Head 反向 → (d_model, seq*batch) ──
        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        Tensor grad_x = std::move(*b_lm);

        // ── 2. LayerNorm/RMSNorm 反向 ──
        auto b_ln = ln_f_->backward(engine, grad_x);
        if (!b_ln) return b_ln;
        grad_x = std::move(*b_ln);

        // ── 3. 逐块反向（全批量化） ──
        {
            const std::size_t n = blocks_.size();
            for (std::size_t bi = 0; bi < n; ++bi)
            {
                const std::size_t idx = n - 1 - bi;
                // 梯度检查点：若是 checkpoint 块，先重算 forward 重建缓存再反向
                if (checkpoint_every_ > 0 && (idx % checkpoint_every_ == 0))
                {
                    const std::size_t seg = idx / checkpoint_every_;
                    NN_ASSERT(seg < checkpoint_inputs_.size(),
                              "GPTModel backward: checkpoint input missing");
                    auto cr = blocks_[idx].forward_recompute(engine, checkpoint_inputs_[seg]);
                    if (!cr) return cr;
                }
                auto br = blocks_[idx].backward(engine, grad_x);
                if (!br) return br;
                grad_x = std::move(*br);
                // 梯度检查点：backward 后立即释放该块的重算激活缓存，
                // 避免跨块累积（否则所有块缓存会在 backward 末尾同时驻留，
                // 抵消检查点的显存收益）。非 checkpoint 模式下不清理。
                if (checkpoint_every_ > 0)
                    blocks_[idx].clear_cache();
                // activation offload：backward 后释放恢复的激活缓存（掩码常驻不清理）
                if (activation_offload_)
                    blocks_[idx].clear_cache();
                if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
                {
                    auto fr = engine.flush_batch();
                    if (!fr) return std::unexpected(fr.error());
                }
            }
        }

        // ── 3.5 重计算完成：释放保存的块输入（供 L2 整块归还） ──
        checkpoint_inputs_.clear();

        // ── 4. 转置 grad_x + pos_grad GPU 计算 ──
        //   grad_x: (d_model, batch*seq)（batch-major 列序）
        //   grad_T = transpose(grad_x) → (total, d_model) — 用于 scatter_add_rows
        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());

        // 位置编码反向（Learned 累计 grad_pos_emb_；Sinusoidal/ALiBi/RoPE no-op）
        auto pr = pos_encoder_->backward(engine, *grad_T, batch_size_, seq_len_);
        if (!pr) return std::unexpected(pr.error());

        // ── 5. scatter_add_rows: grad_token_emb_[tokens] += grad_T ──
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        // grad_input: token IDs 无梯度，返回零张量（仅用于接口一致性）
        Matrix grad_input(seq_len, batch_size_, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── batch 录制粒度控制 ──
    void set_flush_interval(std::size_t interval) override { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 梯度检查点（激活重计算 L1）粒度控制 ──
    // stride：每 N 个 GPTBlock 保存一次块输入，backward 时重算该块 forward。
    // 0 = 不启用（默认）。stride=1 时显存收益最大（仅保留块输入 + 单块激活）。
    void set_checkpoint_every(std::size_t stride) override { checkpoint_every_ = stride; }
    [[nodiscard]] std::size_t checkpoint_every() const noexcept { return checkpoint_every_; }

    // ── activation offload（L1-offload）开关 ──
    // 启用后：forward 把每块内部激活搬 host-visible（释放 device-local VRAM），
    // backward 拷回再反向（不重算，FLOPs 保持 1.0×，代价是 PCIe 传输）。
    void set_activation_offload(bool enabled) override
    {
        activation_offload_ = enabled;
        for (auto& b : blocks_) b.set_offload_enabled(enabled);
    }
    [[nodiscard]] bool activation_offload() const noexcept { return activation_offload_; }

    // 实际 offload RAM 字节数：各块已创建 slab 大小之和（诊断用）
    [[nodiscard]] std::size_t offload_ram_bytes() override
    {
        std::size_t total = 0;
        for (auto& b : blocks_)
            total += b.offload_slab_bytes();
        return total;
    }

    // 梯度检查点：把模式传播给所有块与末级归一化/LM Head
    // （当本 GPTModel 整体作为 Model 的一层被置于 checkpoint 模式时生效）
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        checkpoint_inputs_.clear();
    }

    // ── 增量推理：单 token 前向（KV cache）──────────────────────────
    // 流程: token_emb[token] [+ pos_emb[pos]] → N × GPTBlock.forward_step
    //       → ln_f → lm_head → (vocab_size, 1)
    // 每层的 KV cache 由调用方维护，forward_step 只负责写入和计算。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        std::size_t token_id,
        std::size_t pos,
        std::vector<Tensor>& k_caches,
        std::vector<Tensor>& v_caches,
        std::size_t cur_len)
    {
        // 1. token embedding 查表 → (1, d_model) → transpose → (d_model, 1)
        Matrix id_m(1, 1);
        id_m.set_value_unchecked(0, 0, static_cast<Scalar>(token_id));
        auto id_t = engine.from_matrix(id_m);
        if (!id_t) return std::unexpected(id_t.error());
        auto emb = engine.gather_rows(token_emb_, *id_t);
        if (!emb) return std::unexpected(emb.error());
        auto x_new = engine.transpose(*emb);
        if (!x_new) return std::unexpected(x_new.error());

        // 2. 位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE no-op）
        auto x_wp = pos_encoder_->apply_step(engine, *x_new, pos);
        if (!x_wp) return std::unexpected(x_wp.error());
        x_new = std::move(*x_wp);

        // 3. 逐块增量前向
        Tensor x = std::move(*x_new);
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            auto r = blocks_[i].forward_step(
                engine, x, k_caches[i], v_caches[i], cur_len);
            if (!r) return r;
            x = std::move(*r);
            // 按 flush_interval 拆分提交（防 TDR）
            if (flush_interval_ > 0 &&
                (i + 1) % flush_interval_ == 0 && i + 1 < blocks_.size())
            {
                auto fr = engine.flush_batch();
                if (!fr) return std::unexpected(fr.error());
            }
        }

        // 4. 最终 LayerNorm + LM Head → (vocab_size, 1)
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    // ── 采样生成（KV cache 增量推理）────────────────────────────────
    // 性能策略（P0 + P1 + KV cache）：
    //   P0: begin_batch/end_batch 包裹 forward_step，单次 GPU 提交。
    //   P1: 每步只上传 1 个 token ID，无需重传整个 seq_len。
    //   KV cache: 每步 attention 只计算 Q×K_history（O(seq_len) 而非 O(seq_len²)），
    //             历文 K/V 不再重复投影。
    //   logits 直接是 (vocab_size, 1)，无需 transpose+slice。
    //
    // 滑动窗口: 当 cur_len 达到 seq_len_ 时，丢弃最旧 token 重建 cache
    //           （保留最后 seq_len_-1 个 token 作为新上下文）。
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

        // ── 预分配 KV cache: 每层一对 (k, v)，形状 (seq_len_, d_model) ──
        // H*d_k = d_model（因为 d_k = d_model / num_heads）
        std::vector<Tensor> k_caches, v_caches;
        k_caches.reserve(blocks_.size());
        v_caches.reserve(blocks_.size());
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            k_caches.push_back(engine.create_tensor(seq_len_, d_model_));
            v_caches.push_back(engine.create_tensor(seq_len_, d_model_));
        }

        // ── prefill: 截断到 seq_len_ 长度（滑动窗口初始） ──────────────
        std::size_t start_init = 0;
        if (context.size() > seq_len_)
            start_init = context.size() - seq_len_;
        std::size_t cur_len = 0;
        Tensor last_logits_t;

        // ── 逐 token 填充 KV cache（prefill 与滑动窗口重建共用） ──────
        // 从 context[start..end) 逐个 forward_step，更新 cur_len 与 last_logits_t
        auto fill_cache_ = [&](std::size_t start) -> Result<void>
        {
            for (std::size_t i = start; i < context.size(); ++i)
            {
                auto r = forward_step(engine, context[i], cur_len,
                                      k_caches, v_caches, cur_len);
                if (!r) return std::unexpected(r.error());
                last_logits_t = *r;
                ++cur_len;
            }
            return {};
        };

        {
            auto r = fill_cache_(start_init);
            if (!r) return std::unexpected(r.error());
        }

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            // Sliding window: rebuild cache when full (keep last seq_len_-1 tokens)
            if (cur_len >= seq_len_)
            {
                for (auto& kc : k_caches) { auto r = engine.zero(kc); if (!r) return std::unexpected(r.error()); }
                for (auto& vc : v_caches) { auto r = engine.zero(vc); if (!r) return std::unexpected(r.error()); }
                cur_len = 0;
                const std::size_t keep = seq_len_ - 1;
                const std::size_t start_new = (context.size() > keep) ? (context.size() - keep) : 0;
                auto r = fill_cache_(start_new);
                if (!r) return std::unexpected(r.error());
            }

            // Sample from last_logits_t (from prefill or previous step)
            auto logits_m = engine.to_matrix(last_logits_t);
            if (!logits_m) return std::unexpected(logits_m.error());

            std::vector<Scalar> last_logits(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last_logits[v] = logits_m->at_unchecked(v, 0);

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

            // Run forward_step for next_token to write KV cache and get new logits
            auto br = engine.begin_batch();
            if (!br) return std::unexpected(br.error());

            auto logits_res = forward_step(engine, next_token, cur_len,
                                           k_caches, v_caches, cur_len);
            if (!logits_res) return std::unexpected(logits_res.error());

            auto er = engine.end_batch();
            if (!er) return std::unexpected(er.error());

            last_logits_t = std::move(*logits_res);
            ++cur_len;
        }
        return generated;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// concat_cols — 沿列方向拼接两个张量（列主序激活：拼接序列/记忆维度）
//   a: (rows, c1), b: (rows, c2) → out: (rows, c1+c2)
// 用 transpose + insert_rows + transpose 组合实现（纯引擎原语，CPU/GPU 均可，
// 无需新增 shader 或 AOT 融合）。用于 ZiPT 阶段二把记忆 K/V 与局部 K/V 拼接。
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline Result<Tensor> concat_cols(
    ComputeEngine& engine, const Tensor& a, const Tensor& b)
{
    if (a.rows() != b.rows())
        return std::unexpected(Error{"concat_cols: row count mismatch"});
    auto aT = engine.transpose(a);   // (c1, rows)
    if (!aT) return std::unexpected(aT.error());
    auto bT = engine.transpose(b);   // (c2, rows)
    if (!bT) return std::unexpected(bT.error());
    Tensor dstT = engine.create_tensor(a.cols() + b.cols(), a.rows());
    auto r1 = engine.insert_rows(dstT, 0, *aT);
    if (!r1) return std::unexpected(r1.error());
    auto r2 = engine.insert_rows(dstT, a.cols(), *bT);
    if (!r2) return std::unexpected(r2.error());
    return engine.transpose(dstT);
}

// ══════════════════════════════════════════════════════════════════════════
// CrossAttention — 阶段一：全局上下文重要性压缩（AttnZip Memory Queries）
//
// 算法（AttnZip 文档 §3.2，单头，忠实还原）：
//   K = X·W_K, V = X·W_V                     X: (d_model, batch·L)
//   A = Softmax(P·K^T / sqrt(d))             P: (d_model, M) 可学习记忆查询
//   C = A·V                                  C: (d_model, batch·M)
//
// 批量化（batch=batch，单头，d_k = d_model）：
//   Q_re = P 沿 batch 平铺 → (batch·d_model, M)
//   K_re = rearrange_3d(K, d_model, batch, L) → (batch·d_model, L)
//   S    = batched_matmul(Q_re, K_re, batch, transA=true, alpha=scale) → (batch·M, L)
//   A    = softmax(S)
//   C_re = batched_matmul(V_re, A, batch, false, true) → (batch·d_model, M)
//   C    = rearrange_3d(C_re, d_model, batch, M, true) → (d_model, batch·M)
//
// 参数：P (d_model, M) 可学习 + w_k + w_v（无 w_o：C = A·V 直接输出）
// 复杂度：对序列长度 L 线性（O(M·L·d)），是 AttnZip 消除 O(L²) 的核心。
// ══════════════════════════════════════════════════════════════════════════
class CrossAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t memory_;       // M：记忆 token 数
    std::size_t seq_len_;      // L：上下文序列长度（0 = 单样本，cols 即 L）
    Scalar scale_;
    Tensor P_;                 // 记忆查询 (d_model, M) 可学习
    Tensor grad_P_;            // 记忆查询梯度
    Linear w_k_;
    Linear w_v_;
    Softmax softmax_;

    // forward 缓存（供 backward）
    Tensor Q_re_cache_;        // (batch·d_model, M) P 平铺
    Tensor K_re_cache_;        // (batch·d_model, L)
    Tensor V_re_cache_;        // (batch·d_model, L)
    std::size_t batch_cache_ = 0;
    std::size_t seq_cache_    = 0;

    // 把 P 沿 batch 平铺为 (batch·d_model, M)：每个 batch 块 = P
    [[nodiscard]] Result<Tensor> build_q_re_(ComputeEngine& engine, std::size_t batch)
    {
        Tensor q = engine.create_tensor(batch * d_model_, memory_);
        for (std::size_t b = 0; b < batch; ++b)
        {
            auto r = engine.insert_rows(q, b * d_model_, P_);
            if (!r) return std::unexpected(r.error());
        }
        return q;
    }

public:
    CrossAttention(std::size_t d_model, std::size_t memory,
                   std::size_t seq_len = 0)
        : d_model_(d_model), memory_(memory), seq_len_(seq_len),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model))),
          w_k_(d_model, d_model), w_v_(d_model, d_model) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        // 初始化记忆查询 P（列主序 (d_model, M)，每列一个记忆查询）
        Matrix p_cpu(d_model_, memory_);
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(d_model_ + memory_));
        std::uniform_real_distribution<Scalar> dist(-limit, limit);
        std::mt19937_64 rng{std::random_device{}()};
        auto sp = p_cpu.span();
        for (std::size_t i = 0; i < p_cpu.size(); ++i) sp[i] = dist(rng);
        auto pr = engine.from_matrix(p_cpu);
        if (!pr) return std::unexpected(pr.error());
        P_ = std::move(*pr);

        grad_P_ = engine.create_tensor(d_model_, memory_);
        { auto r = engine.zero(grad_P_); if (!r) return std::unexpected(r.error()); }

        auto r1 = w_k_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_v_.init(engine); if (!r2) return std::unexpected(r2.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> out;
        out.push_back(P_);
        auto p = w_k_.parameters();
        auto v = w_v_.parameters();
        out.insert(out.end(), p.begin(), p.end());
        out.insert(out.end(), v.begin(), v.end());
        return out;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> out;
        out.push_back(grad_P_);
        auto gk = w_k_.param_gradients();
        auto gv = w_v_.param_gradients();
        out.insert(out.end(), gk.begin(), gk.end());
        out.insert(out.end(), gv.begin(), gv.end());
        return out;
    }

    void clear_cache() override
    {
        Q_re_cache_ = Tensor{}; K_re_cache_ = Tensor{}; V_re_cache_ = Tensor{};
        softmax_.clear_cache(); batch_cache_ = 0; seq_cache_ = 0;
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Q_re_cache_.valid()) r.emplace_back(Q_re_cache_);
        if (K_re_cache_.valid()) r.emplace_back(K_re_cache_);
        if (V_re_cache_.valid()) r.emplace_back(V_re_cache_);
        auto s = softmax_.activation_cache();
        r.insert(r.end(), s.begin(), s.end());
        return r;
    }

    // 前向：输入 X (d_model, batch·L) → 输出 C (d_model, batch·M)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t total = input.cols();
        const std::size_t L = (seq_len_ > 0) ? seq_len_ : total;
        const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;

        auto K = w_k_.forward(engine, input);   // (d_model, batch·L)
        if (!K) return K;
        auto V = w_v_.forward(engine, input);
        if (!V) return V;

        auto Q_re = build_q_re_(engine, batch);              // (batch·d_model, M)
        if (!Q_re) return std::unexpected(Q_re.error());
        auto K_re = engine.rearrange_3d(*K, d_model_, batch, L, false);  // (batch·d_model, L)
        if (!K_re) return std::unexpected(K_re.error());
        auto V_re = engine.rearrange_3d(*V, d_model_, batch, L, false);
        if (!V_re) return std::unexpected(V_re.error());

        // S = batched_matmul(Q_re, K_re, batch, transA=true, alpha=scale) → (batch·M, L)
        auto S = engine.batched_matmul(*Q_re, *K_re, batch, true, false, scale_);
        if (!S) return std::unexpected(S.error());
        // A = softmax(S)
        auto attn = softmax_.forward(engine, *S);
        if (!attn) return std::unexpected(attn.error());
        // C_re = batched_matmul(V_re, A, batch, false, true) → (batch·d_model, M)
        auto C_re = engine.batched_matmul(*V_re, *attn, batch, false, true);
        if (!C_re) return std::unexpected(C_re.error());
        // C = rearrange back → (d_model, batch·M)
        auto C = engine.rearrange_3d(*C_re, d_model_, batch, memory_, true);
        if (!C) return std::unexpected(C.error());

        if (!checkpoint_mode_)
        {
            Q_re_cache_ = std::move(*Q_re);
            K_re_cache_ = std::move(*K_re);
            V_re_cache_ = std::move(*V_re);
            batch_cache_ = batch;
            seq_cache_   = L;
        }
        return C;
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        // grad_output: (d_model, batch·M)
        const std::size_t batch = batch_cache_;
        const std::size_t L = seq_cache_;

        auto grad_C_re = engine.rearrange_3d(grad_output, d_model_, batch, memory_, false);
        if (!grad_C_re) return std::unexpected(grad_C_re.error());

        // A 由 softmax.output_cache() 单一持有
        const Tensor& A = softmax_.output_cache();

        // grad_V_re = batched_matmul(grad_C, A, batch, false, false) → (batch·d_model, L)
        auto grad_V_re = engine.batched_matmul(*grad_C_re, A, batch, false, false);
        if (!grad_V_re) return std::unexpected(grad_V_re.error());
        // grad_A = batched_matmul(grad_C, V, batch, true, false) → (batch·M, L)
        auto grad_A = engine.batched_matmul(*grad_C_re, V_re_cache_, batch, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());
        // grad_S = softmax.backward(grad_A)
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());

        // grad_Q_re = batched_matmul(K, grad_S, batch, false, true, scale) → (batch·d_model, M)
        auto grad_Q_re = engine.batched_matmul(K_re_cache_, *grad_S, batch, false, true, scale_);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        // grad_P = Σ_b grad_Q_re[b]（Q 是 P 沿 batch 的平铺，需按 batch 累加）
        for (std::size_t b = 0; b < batch; ++b)
        {
            auto slice = engine.slice_rows(*grad_Q_re, b * d_model_, d_model_);
            if (!slice) return std::unexpected(slice.error());
            auto ar = engine.add_inplace(grad_P_, *slice);
            if (!ar) return std::unexpected(ar.error());
        }
        // grad_K_re = batched_matmul(Q, grad_S, batch, false, false, scale) → (batch·d_model, L)
        auto grad_K_re = engine.batched_matmul(Q_re_cache_, *grad_S, batch, false, false, scale_);
        if (!grad_K_re) return std::unexpected(grad_K_re.error());

        // 还原到 (d_model, batch·L) 并喂给 w_k / w_v
        auto grad_K = engine.rearrange_3d(*grad_K_re, d_model_, batch, L, true);
        if (!grad_K) return std::unexpected(grad_K.error());
        auto grad_V = engine.rearrange_3d(*grad_V_re, d_model_, batch, L, true);
        if (!grad_V) return std::unexpected(grad_V.error());
        auto gk = w_k_.backward(engine, *grad_K);
        if (!gk) return gk;
        auto gv = w_v_.backward(engine, *grad_V);
        if (!gv) return gv;
        return engine.elementwise_binary(BinaryOp::Add, *gk, *gv);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ZiPTBlock — 阶段二：局部-全局联合注意力 + FFN（Pre-Norm 解码器块）
//
// 算法（AttnZip 文档 §3.3）：
//   n1 = LN₁(Y)
//   Q_y = n1·W_Q,  K_y = n1·W_K,  V_y = n1·W_V         局部 Y (d_model, batch·W)
//   K_c = C·W_Kc,  V_c = C·W_Vc                        记忆 C (d_model, batch·M)
//   K_cat = [K_c, K_y],  V_cat = [V_c, V_y]            序列维拼接
//   O = Softmax(Q_y·K_cat^T / sqrt(d_k)) · V_cat
//   r2 = Y + O·W_O； out = r2 + FFN(LN₂(r2))
//
// 掩码：记忆 token 对局部全部可见（无掩码），局部 token 之间因果
//   （token t 可见记忆全部 + 局部 ≤ t；局部 > t 掩掉）。
// 批量化（batch*H，materialized 路径）：Q/K/V 经 rearrange 后 batched_matmul。
// 复杂度：对长上下文 L 为常数（记忆 M 与窗口 W 均 ≪ L）。
// ══════════════════════════════════════════════════════════════════════════
class ZiPTBlock final : public Layer
{
private:
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t window_;       // W：局部窗口（模型 seq_len）
    std::size_t memory_;       // M：记忆 token 数
    Scalar scale_;
    std::unique_ptr<Layer> norm1_;
    Linear w_q_, w_k_, w_v_, w_o_;   // 局部投影
    Linear w_kc_, w_vc_;             // 记忆 K/V 投影（独立，避免共享 Linear 缓存串扰）
    Softmax softmax_;
    std::unique_ptr<Layer> norm2_;
    FeedForward ff_;

    Tensor residual2_cache_;
    // forward 缓存（联合注意力反向）
    Tensor Q_re_cache_;        // (batch*H*d_k, W)
    Tensor K_cat_cache_;       // (batch*H*d_k, M+W)
    Tensor V_cat_cache_;       // (batch*H*d_k, M+W)
    std::size_t batch_cache_ = 0;
    // 掩码缓存（常数，跨 step 复用；按 batch 键控）
    Tensor mask_cache_;
    std::size_t mask_batch_ = 0;

    // 构建记忆-局部联合掩码 (batch*H*W, M+W)：
    //   行 (bb, t)：列 [0, M) 记忆全可见；列 [M, M+t] 局部因果；列 [M+t+1, M+W) = -1e30
    [[nodiscard]] Result<Tensor> build_mask_(ComputeEngine& engine, std::size_t batch)
    {
        if (mask_cache_.valid() && mask_batch_ == batch)
            return mask_cache_;
        const std::size_t BH = batch * num_heads_;
        const std::size_t total_keys = memory_ + window_;
        Matrix mask(BH * window_, total_keys, Scalar{0});
        const Scalar NEG = Scalar{-1e30};
        for (std::size_t bb = 0; bb < BH; ++bb)
            for (std::size_t t = 0; t < window_; ++t)
                for (std::size_t j = memory_ + t + 1; j < total_keys; ++j)
                    mask.set_value_unchecked(bb * window_ + t, j, NEG);
        auto mr = engine.from_matrix(mask);
        if (!mr) return std::unexpected(mr.error());
        mask_cache_ = std::move(*mr);
        mask_batch_ = batch;
        return mask_cache_;
    }

public:
    ZiPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
              std::size_t window, std::size_t memory,
              NormType norm_type = NormType::LayerNorm,
              ActivationType activation = ActivationType::GeLU)
        : num_heads_(num_heads),
          d_k_(d_model / num_heads), window_(window), memory_(memory),
          scale_(Scalar{1} / std::sqrt(static_cast<Scalar>(d_model / num_heads))),
          norm1_(make_norm_layer(d_model, norm_type)),
          w_q_(d_model, d_model), w_k_(d_model, d_model),
          w_v_(d_model, d_model), w_o_(d_model, d_model),
          w_kc_(d_model, d_model), w_vc_(d_model, d_model),
          norm2_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ZiPTBlock: d_model must be divisible by num_heads");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = norm1_->init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_q_.init(engine);    if (!r2) return std::unexpected(r2.error());
        auto r3 = w_k_.init(engine);    if (!r3) return std::unexpected(r3.error());
        auto r4 = w_v_.init(engine);    if (!r4) return std::unexpected(r4.error());
        auto r5 = w_o_.init(engine);    if (!r5) return std::unexpected(r5.error());
        auto r6 = w_kc_.init(engine);   if (!r6) return std::unexpected(r6.error());
        auto r7 = w_vc_.init(engine);   if (!r7) return std::unexpected(r7.error());
        auto r8 = norm2_->init(engine); if (!r8) return std::unexpected(r8.error());
        auto r9 = ff_.init(engine);     if (!r9) return std::unexpected(r9.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> out;
        auto n1 = norm1_->parameters(); out.insert(out.end(), n1.begin(), n1.end());
        auto q = w_q_.parameters();     out.insert(out.end(), q.begin(), q.end());
        auto k = w_k_.parameters();     out.insert(out.end(), k.begin(), k.end());
        auto v = w_v_.parameters();     out.insert(out.end(), v.begin(), v.end());
        auto o = w_o_.parameters();     out.insert(out.end(), o.begin(), o.end());
        auto kc = w_kc_.parameters();   out.insert(out.end(), kc.begin(), kc.end());
        auto vc = w_vc_.parameters();   out.insert(out.end(), vc.begin(), vc.end());
        auto n2 = norm2_->parameters(); out.insert(out.end(), n2.begin(), n2.end());
        auto f = ff_.parameters();      out.insert(out.end(), f.begin(), f.end());
        return out;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> out;
        auto n1 = norm1_->param_gradients(); out.insert(out.end(), n1.begin(), n1.end());
        auto q = w_q_.param_gradients();     out.insert(out.end(), q.begin(), q.end());
        auto k = w_k_.param_gradients();     out.insert(out.end(), k.begin(), k.end());
        auto v = w_v_.param_gradients();     out.insert(out.end(), v.begin(), v.end());
        auto o = w_o_.param_gradients();     out.insert(out.end(), o.begin(), o.end());
        auto kc = w_kc_.param_gradients();   out.insert(out.end(), kc.begin(), kc.end());
        auto vc = w_vc_.param_gradients();   out.insert(out.end(), vc.begin(), vc.end());
        auto n2 = norm2_->param_gradients(); out.insert(out.end(), n2.begin(), n2.end());
        auto f = ff_.param_gradients();      out.insert(out.end(), f.begin(), f.end());
        return out;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
        w_kc_.set_checkpoint_mode(enabled);
        w_vc_.set_checkpoint_mode(enabled);
        softmax_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        norm1_->clear_cache(); w_q_.clear_cache(); w_k_.clear_cache();
        w_v_.clear_cache(); w_o_.clear_cache(); w_kc_.clear_cache();
        w_vc_.clear_cache(); softmax_.clear_cache(); norm2_->clear_cache();
        ff_.clear_cache();
        residual2_cache_ = Tensor{};
        Q_re_cache_ = Tensor{}; K_cat_cache_ = Tensor{}; V_cat_cache_ = Tensor{};
        batch_cache_ = 0;
    }

    // 基类单输入 forward/backward 不适用于双输入块（仅经 ZiPTModel 组合使用）
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& /*input*/) override
    {
        (void)engine;
        return std::unexpected(Error{"ZiPTBlock: use forward(engine, input, memory_input)"});
    }
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& /*grad_output*/) override
    {
        (void)engine;
        return std::unexpected(Error{"ZiPTBlock: use backward(engine, grad, grad_C_out)"});
    }

    // 前向：局部 Y (d_model, batch·W) + 记忆 C (d_model, batch·M) → (d_model, batch·W)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input, const Tensor& memory_input)
    {
        const std::size_t total = input.cols();
        const std::size_t batch = (window_ > 0) ? (total / window_) : 1;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;

        auto Q = w_q_.forward(engine, *n1);   // (d_model, batch·W)
        if (!Q) return Q;
        auto Ky = w_k_.forward(engine, *n1);
        if (!Ky) return Ky;
        auto Vy = w_v_.forward(engine, *n1);
        if (!Vy) return Vy;
        auto Kc = w_kc_.forward(engine, memory_input);   // (d_model, batch·M)
        if (!Kc) return Kc;
        auto Vc = w_vc_.forward(engine, memory_input);
        if (!Vc) return Vc;

        // rearrange 为 (batch*H*d_k, ·) 批量化布局
        auto Q_re = engine.rearrange_3d(*Q, H_dk, batch, window_, false);
        if (!Q_re) return std::unexpected(Q_re.error());
        auto Ky_re = engine.rearrange_3d(*Ky, H_dk, batch, window_, false);
        if (!Ky_re) return std::unexpected(Ky_re.error());
        auto Vy_re = engine.rearrange_3d(*Vy, H_dk, batch, window_, false);
        if (!Vy_re) return std::unexpected(Vy_re.error());
        auto Kc_re = engine.rearrange_3d(*Kc, H_dk, batch, memory_, false);
        if (!Kc_re) return std::unexpected(Kc_re.error());
        auto Vc_re = engine.rearrange_3d(*Vc, H_dk, batch, memory_, false);
        if (!Vc_re) return std::unexpected(Vc_re.error());

        // 序列维拼接：记忆在前，局部在后
        auto K_cat = concat_cols(engine, *Kc_re, *Ky_re);   // (batch*H*d_k, M+W)
        if (!K_cat) return std::unexpected(K_cat.error());
        auto V_cat = concat_cols(engine, *Vc_re, *Vy_re);
        if (!V_cat) return std::unexpected(V_cat.error());

        // S = batched_matmul(Q_re, K_cat, BH, transA=true, alpha=scale) → (batch*H*W, M+W)
        auto S = engine.batched_matmul(*Q_re, *K_cat, BH, true, false, scale_);
        if (!S) return std::unexpected(S.error());
        // 施加记忆-局部联合掩码
        auto mask = build_mask_(engine, batch);
        if (!mask) return std::unexpected(mask.error());
        auto ma = engine.add_inplace(*S, *mask);
        if (!ma) return std::unexpected(ma.error());
        // A = softmax(S)
        auto attn = softmax_.forward(engine, *S);
        if (!attn) return std::unexpected(attn.error());
        // O_re = batched_matmul(V_cat, A, BH, false, true) → (batch*H*d_k, W)
        auto O_re = engine.batched_matmul(*V_cat, *attn, BH, false, true);
        if (!O_re) return std::unexpected(O_re.error());
        // O = rearrange back → (d_model, batch·W)
        auto O = engine.rearrange_3d(*O_re, H_dk, batch, window_, true);
        if (!O) return std::unexpected(O.error());
        auto out_attn = w_o_.forward(engine, *O);
        if (!out_attn) return out_attn;

        auto r2 = engine.elementwise_binary(BinaryOp::Add, input, *out_attn);
        if (!r2) return std::unexpected(r2.error());
        if (!checkpoint_mode_)
            residual2_cache_ = *r2;

        auto n2 = norm2_->forward(engine, *r2);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;

        if (!checkpoint_mode_)
        {
            Q_re_cache_ = std::move(*Q_re);
            K_cat_cache_ = std::move(*K_cat);
            V_cat_cache_ = std::move(*V_cat);
            batch_cache_ = batch;
        }
        return engine.elementwise_binary(BinaryOp::Add, *r2, *f);
    }

    // 反向：grad_output (d_model, batch·W)；grad_C_out 累加记忆 C 的梯度
    // 返回对局部输入 Y 的梯度 (d_model, batch·W)
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output, Tensor& grad_C_out)
    {
        const std::size_t batch = batch_cache_;
        const std::size_t H_dk = num_heads_ * d_k_;
        const std::size_t BH = batch * num_heads_;

        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;
        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());

        // ── 联合注意力反向（materialized 路径，镜像 AttentionBase 旧路径） ──
        auto grad_O = w_o_.backward(engine, *grad_r1);   // (d_model, batch·W)
        if (!grad_O) return grad_O;
        auto grad_concat_re = engine.rearrange_3d(*grad_O, H_dk, batch, window_, false);
        if (!grad_concat_re) return std::unexpected(grad_concat_re.error());
        const Tensor& A = softmax_.output_cache();

        // grad_V_cat = batched_matmul(grad_O, A, BH, false, false) → (batch*H*d_k, M+W)
        auto grad_V_cat = engine.batched_matmul(*grad_concat_re, A, BH, false, false);
        if (!grad_V_cat) return std::unexpected(grad_V_cat.error());
        // grad_A = batched_matmul(grad_O, V_cat, BH, true, false) → (batch*H*W, M+W)
        auto grad_A = engine.batched_matmul(*grad_concat_re, V_cat_cache_, BH, true, false);
        if (!grad_A) return std::unexpected(grad_A.error());
        auto grad_S = softmax_.backward(engine, *grad_A);
        if (!grad_S) return std::unexpected(grad_S.error());
        // grad_Q_re = batched_matmul(K_cat, grad_S, BH, false, true, scale) → (batch*H*d_k, W)
        auto grad_Q_re = engine.batched_matmul(K_cat_cache_, *grad_S, BH, false, true, scale_);
        if (!grad_Q_re) return std::unexpected(grad_Q_re.error());
        // grad_K_cat = batched_matmul(Q_re, grad_S, BH, false, false, scale) → (batch*H*d_k, M+W)
        auto grad_K_cat = engine.batched_matmul(Q_re_cache_, *grad_S, BH, false, false, scale_);
        if (!grad_K_cat) return std::unexpected(grad_K_cat.error());

        // 沿序列维拆分 grad_K/grad_V 的记忆与局部部分（transpose → slice_rows → transpose）
        auto split_cols_ = [&](const Tensor& t)
            -> Result<std::pair<Tensor, Tensor>>
        {
            auto tT = engine.transpose(t);   // (M+W, batch*H*d_k)
            if (!tT) return std::unexpected(tT.error());
            auto memT = engine.slice_rows(*tT, 0, memory_);       // (M, batch*H*d_k)
            if (!memT) return std::unexpected(memT.error());
            auto locT = engine.slice_rows(*tT, memory_, window_); // (W, batch*H*d_k)
            if (!locT) return std::unexpected(locT.error());
            auto mem = engine.transpose(*memT);   // (batch*H*d_k, M)
            if (!mem) return std::unexpected(mem.error());
            auto loc = engine.transpose(*locT);   // (batch*H*d_k, W)
            if (!loc) return std::unexpected(loc.error());
            return std::pair<Tensor, Tensor>{std::move(*mem), std::move(*loc)};
        };

        auto k_split = split_cols_(*grad_K_cat);
        if (!k_split) return std::unexpected(k_split.error());
        auto v_split = split_cols_(*grad_V_cat);
        if (!v_split) return std::unexpected(v_split.error());

        // ── 局部路径梯度 ──
        auto grad_Q = engine.rearrange_3d(*grad_Q_re, H_dk, batch, window_, true);
        if (!grad_Q) return std::unexpected(grad_Q.error());
        auto gq = w_q_.backward(engine, *grad_Q);
        if (!gq) return gq;
        auto grad_Ky = engine.rearrange_3d(k_split->second, H_dk, batch, window_, true);
        if (!grad_Ky) return std::unexpected(grad_Ky.error());
        auto gk = w_k_.backward(engine, *grad_Ky);
        if (!gk) return gk;
        auto grad_Vy = engine.rearrange_3d(v_split->second, H_dk, batch, window_, true);
        if (!grad_Vy) return std::unexpected(grad_Vy.error());
        auto gv = w_v_.backward(engine, *grad_Vy);
        if (!gv) return gv;
        auto g_attn = engine.elementwise_binary(BinaryOp::Add, *gq, *gk);
        if (!g_attn) return std::unexpected(g_attn.error());
        auto grad_n1 = engine.elementwise_binary(BinaryOp::Add, *g_attn, *gv);
        if (!grad_n1) return std::unexpected(grad_n1.error());
        auto b_n1 = norm1_->backward(engine, *grad_n1);
        if (!b_n1) return b_n1;
        auto grad_x = engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
        if (!grad_x) return std::unexpected(grad_x.error());

        // ── 记忆路径梯度（累加进 grad_C_out） ──
        auto grad_Kc = engine.rearrange_3d(k_split->first, H_dk, batch, memory_, true);
        if (!grad_Kc) return std::unexpected(grad_Kc.error());
        auto gkc = w_kc_.backward(engine, *grad_Kc);
        if (!gkc) return gkc;
        auto grad_Vc = engine.rearrange_3d(v_split->first, H_dk, batch, memory_, true);
        if (!grad_Vc) return std::unexpected(grad_Vc.error());
        auto gvc = w_vc_.backward(engine, *grad_Vc);
        if (!gvc) return gvc;
        { auto a1 = engine.add_inplace(grad_C_out, *gkc); if (!a1) return std::unexpected(a1.error()); }
        { auto a2 = engine.add_inplace(grad_C_out, *gvc); if (!a2) return std::unexpected(a2.error()); }

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ZiPTModel — AttnZip 解码器（zipt：zip + GPT）
//
//   token_emb → (+pos_enc) → [历史 H 压缩为记忆 C] → N × ZiPTBlock(窗口 W, C)
//   → LN → LM Head
//
// 两种模式（由 window 参数决定，W=seq_len 为旧行为向后兼容）：
//   * W = L（window==seq_len，旧行为）：压缩整条序列 X → C，块对 [C;X] 联合注意力。
//     注意：此时块注意力键 = M+L，复杂度 O(L²)，实际上「无压缩」。
//   * W < L（L = W + C，新行为）：把输入 X(长度 L) 按 batch 拆为
//       历史 H = X[0 : L-W]（长度 C = L-W）→ 压缩为记忆 C (M 个 token)
//       窗口 W_seq = X[L-W : L]（长度 W）→ 块直接注意力
//     每块对 [M 记忆 ; W 窗口] 联合注意力，键数 = M+W（注意力预算）。
//     复杂度：压缩 O(M·C·d) 线性于 L；块注意力 O(W·(M+W)) 对 L 为常数。
//     即：用「传统上下文 (M+W)」的注意力预算处理「总长度 L」的上下文。
//
// 训练范式：对完整序列 X 一次性压缩历史为记忆 C，随后每块对 [C ; 窗口] 联合
// 注意力，loss 仅在窗口 W 上计算（历史被压缩器"消费"，不直接预测）。
// ══════════════════════════════════════════════════════════════════════════
class ZiPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;    // L：总上下文长度
    std::size_t window_;     // W：局部窗口（=seq_len 时为旧行为 W=L 无压缩）
    std::size_t hist_len_;   // C = L - W：历史长度（被压缩，split 模式 >0）
    std::size_t memory_;     // M
    bool split_;             // true = W<L 新行为（历史/窗口分离）；false = 旧行为 W=L

    Tensor token_emb_;
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;
    CrossAttention compressor_;
    std::vector<ZiPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    Tensor stored_tokens_tensor_;
    std::size_t batch_size_ = 0;

    // 沿 batch-major 列序把 x (d_model, batch·L) 拆为 历史 H (d_model, batch·C)
    // 与 窗口 W_seq (d_model, batch·W)。列序 i=b·L+t（batch 在列方向）。
    [[nodiscard]] Result<std::pair<Tensor, Tensor>>
    split_history_window_(ComputeEngine& engine, const Tensor& x,
                          std::size_t batch, std::size_t L,
                          std::size_t C, std::size_t W)
    {
        auto x_re = engine.rearrange_3d(x, d_model_, batch, L, false); // (batch·d, L)
        if (!x_re) return std::unexpected(x_re.error());
        auto xT = engine.transpose(*x_re);                            // (L, batch·d)
        if (!xT) return std::unexpected(xT.error());
        auto hT = engine.slice_rows(*xT, 0, C);                       // (C, batch·d)
        if (!hT) return std::unexpected(hT.error());
        auto wT = engine.slice_rows(*xT, C, W);                       // (W, batch·d)
        if (!wT) return std::unexpected(wT.error());
        auto h_re = engine.transpose(*hT);                            // (batch·d, C)
        if (!h_re) return std::unexpected(h_re.error());
        auto w_re = engine.transpose(*wT);                            // (batch·d, W)
        if (!w_re) return std::unexpected(w_re.error());
        auto H = engine.rearrange_3d(*h_re, d_model_, batch, C, true);   // (d, batch·C)
        if (!H) return std::unexpected(H.error());
        auto Ws = engine.rearrange_3d(*w_re, d_model_, batch, W, true);  // (d, batch·W)
        if (!Ws) return std::unexpected(Ws.error());
        return std::pair<Tensor, Tensor>{std::move(*H), std::move(*Ws)};
    }

    // 合并 历史梯度 (d, batch·C) 与 窗口梯度 (d, batch·W) → (d, batch·L)
    [[nodiscard]] Result<Tensor>
    merge_history_window_(ComputeEngine& engine, const Tensor& gradH,
                          const Tensor& gradW, std::size_t batch,
                          std::size_t L, std::size_t C, std::size_t W)
    {
        auto h_re = engine.rearrange_3d(gradH, d_model_, batch, C, false); // (batch·d, C)
        if (!h_re) return std::unexpected(h_re.error());
        auto w_re = engine.rearrange_3d(gradW, d_model_, batch, W, false); // (batch·d, W)
        if (!w_re) return std::unexpected(w_re.error());
        auto hT = engine.transpose(*h_re);   // (C, batch·d)
        if (!hT) return std::unexpected(hT.error());
        auto wT = engine.transpose(*w_re);   // (W, batch·d)
        if (!wT) return std::unexpected(wT.error());
        Tensor dstT = engine.create_tensor(L, batch * d_model_);
        auto r1 = engine.insert_rows(dstT, 0, *hT);
        if (!r1) return std::unexpected(r1.error());
        auto r2 = engine.insert_rows(dstT, C, *wT);
        if (!r2) return std::unexpected(r2.error());
        auto dst = engine.transpose(dstT);   // (batch·d, L)
        if (!dst) return std::unexpected(dst.error());
        return engine.rearrange_3d(*dst, d_model_, batch, L, true);  // (d, batch·L)
    }

public:
    ZiPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
              std::size_t window,
              std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
              std::size_t memory_tokens,
              PosEncodingType pos_enc_type = PosEncodingType::Learned,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          window_(window), memory_(memory_tokens),
          compressor_(d_model, memory_tokens,
                      (window < seq_len) ? (seq_len - window) : seq_len),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        // 归约 W：W==0 或 W>seq_len → 回退到 W=L（旧行为）
        if (window_ == 0 || window_ > seq_len_) window_ = seq_len_;
        hist_len_ = seq_len_ - window_;
        split_    = (window_ < seq_len_);
        const std::size_t block_window = split_ ? window_ : seq_len_;

        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, block_window, memory_tokens,
                                 norm_type, activation);
        switch (pos_enc_type)
        {
            case PosEncodingType::Learned:
                pos_encoder_ = std::make_unique<LearnedPositionEncoder>(d_model, seq_len);
                break;
            case PosEncodingType::Sinusoidal:
                pos_encoder_ = std::make_unique<SinusoidalPositionEncoder>(d_model, seq_len);
                break;
            default:
                pos_encoder_ = std::make_unique<NoPositionEncoder>();
                break;
        }
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix te(vocab_size_, d_model_);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);
        auto te_r = engine.from_matrix(te);
        if (!te_r) return std::unexpected(te_r.error());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_);
        { auto r = engine.zero(grad_token_emb_); if (!r) return std::unexpected(r.error()); }

        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        auto r1 = compressor_.init(engine);
        if (!r1) return std::unexpected(r1.error());
        for (auto& b : blocks_)
        {
            auto r = b.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        if (ln_f_)
        {
            auto r = ln_f_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        { auto r = lm_head_.init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        auto cp = compressor_.parameters();
        p.insert(p.end(), cp.begin(), cp.end());
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        auto gp = pos_encoder_->param_gradients();
        g.insert(g.end(), gp.begin(), gp.end());
        auto cg = compressor_.param_gradients();
        g.insert(g.end(), cg.begin(), cg.end());
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_->param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq_len = input.rows();
        const std::size_t batch = input.cols();
        batch_size_ = batch;

        auto input_T = engine.transpose(input);   // (batch, seq) batch-major
        if (!input_T) return std::unexpected(input_T.error());
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());
        auto st = engine.clone(*input_T);
        if (!st) return std::unexpected(st.error());
        stored_tokens_tensor_ = std::move(*st);
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        auto x_res = pos_encoder_->apply(engine, *all_T, batch, seq_len);
        if (!x_res) return std::unexpected(x_res.error());
        Tensor x = std::move(*x_res);

        // 阶段一：历史压缩 → 记忆 C；阶段二：块对窗口联合注意力
        Tensor C;
        Tensor block_input;
        if (split_)
        {
            auto split = split_history_window_(engine, x, batch, seq_len_, hist_len_, window_);
            if (!split) return std::unexpected(split.error());
            auto c_res = compressor_.forward(engine, split->first);   // H → C (d, batch·M)
            if (!c_res) return c_res;
            C = std::move(*c_res);
            block_input = std::move(split->second);                   // W_seq (d, batch·W)
        }
        else
        {
            auto c_res = compressor_.forward(engine, x);              // 旧行为：压缩整条 X
            if (!c_res) return c_res;
            C = std::move(*c_res);
            block_input = x;
        }

        // 阶段二：逐块联合注意力（对 [C ; 窗口]）
        for (auto& b : blocks_)
        {
            auto r = b.forward(engine, block_input, C);
            if (!r) return r;
            block_input = std::move(*r);
        }

        auto ln = ln_f_->forward(engine, block_input);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq_len = seq_len_;
        const std::size_t batch = batch_size_;

        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        auto b_ln = ln_f_->backward(engine, *b_lm);
        if (!b_ln) return b_ln;
        Tensor grad_w = std::move(*b_ln);   // (d_model, batch·W) 块输出梯度

        // 记忆 C 梯度：跨块累加
        Tensor grad_C = engine.create_tensor(d_model_, batch * memory_);
        { auto z = engine.zero(grad_C); if (!z) return std::unexpected(z.error()); }

        for (std::size_t i = blocks_.size(); i-- > 0;)
        {
            auto br = blocks_[i].backward(engine, grad_w, grad_C);
            if (!br) return br;
            grad_w = std::move(*br);
        }

        Tensor grad_x;
        if (split_)
        {
            // 压缩器反向：grad_C → 对历史 H 的梯度 (d, batch·C)
            auto gxh = compressor_.backward(engine, grad_C);
            if (!gxh) return gxh;
            // 合并 [grad_H ; grad_W] → (d, batch·L)
            auto merged = merge_history_window_(engine, *gxh, grad_w,
                                                batch, seq_len_, hist_len_, window_);
            if (!merged) return std::unexpected(merged.error());
            grad_x = std::move(*merged);
        }
        else
        {
            // 旧行为：压缩器对整条 X 的梯度与块梯度同形状，直接累加
            auto gxc = compressor_.backward(engine, grad_C);
            if (!gxc) return gxc;
            auto gx_sum = engine.elementwise_binary(BinaryOp::Add, grad_w, *gxc);
            if (!gx_sum) return std::unexpected(gx_sum.error());
            grad_x = std::move(*gx_sum);
        }

        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());
        auto pr = pos_encoder_->backward(engine, *grad_T, batch, seq_len);
        if (!pr) return std::unexpected(pr.error());
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        Matrix grad_input(seq_len, batch, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── 采样生成（重计算式，无 KV cache） ────────────────────────
    // 每步把「当前上下文（prompt + 已生成，滑动窗口截断到 seq_len）」
    // 重新 forward（内部重做压缩 C + 联合注意力），取末位 logits 采样。
    // 复杂度 O(seq²)（每步 O(seq)），对首个 CLI 接入足够；KV cache 优化留待后续。
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
            // 滑动窗口截断到 seq_len_
            std::size_t start = 0;
            if (context.size() > seq_len_) start = context.size() - seq_len_;
            const std::size_t n = context.size() - start;

            // 输入 (seq_len_, 1)：使 forward 始终处理整窗（压缩机/块按固定长度构建）。
            // 旧行为 (W=L)：前 n 个为上下文，其余 pad_id=0 填充；末位 logits 取 n-1。
            // 新行为 (W<L)：右对齐，末位真实 token 落到窗口末（位置 seq_len_-1），
            //   forward 输出仅覆盖窗口 (vocab, W)，logits 取窗口末列 window_-1。
            const std::size_t pred_col = split_ ? (window_ - 1) : (n - 1);
            Matrix in(seq_len_, 1);
            if (split_)
            {
                const std::size_t pad_head = seq_len_ - n;
                for (std::size_t i = 0; i < seq_len_; ++i)
                    in.set_value_unchecked(i, 0,
                        static_cast<Scalar>((i >= pad_head) ? context[start + (i - pad_head)] : 0));
            }
            else
            {
                for (std::size_t i = 0; i < seq_len_; ++i)
                    in.set_value_unchecked(i, 0,
                        static_cast<Scalar>((i < n) ? context[start + i] : 0));
            }
            auto in_t = engine.from_matrix(in);
            if (!in_t) return std::unexpected(in_t.error());
            auto logits = forward(engine, *in_t);
            if (!logits) return std::unexpected(logits.error());
            auto lm = engine.to_matrix(*logits);
            if (!lm) return std::unexpected(lm.error());

            // 末位（最后真实位置）logits
            std::vector<Scalar> last(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last[v] = lm->at_unchecked(v, pred_col);

            if (temperature > 0.0 && temperature != 1.0)
                for (auto& x : last) x /= temperature;

            Scalar max_val = last[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, last[v]);
            Scalar sum_exp = 0;
            for (auto& x : last) { x = std::exp(x - max_val); sum_exp += x; }
            for (auto& x : last) x /= sum_exp;

            std::size_t next;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cum = 0;
                next = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cum += last[v];
                    if (r <= cum) { next = v; break; }
                }
            }
            else
            {
                next = 0;
                Scalar best = last[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (last[v] > best) { best = last[v]; next = v; }
            }

            context.push_back(next);
            if (step >= min_new_tokens && next == eos_token_id) break;
            generated.push_back(next);
        }
        return generated;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ReLULinearAttention — ReLU 线性注意力（RLA / RAPT 核心层）
//
// 算法（RLA.md §3，causal 版；bidirectional 亦支持）：
//   q' = ReLU(RoPE(q)), k' = ReLU(RoPE(k)), v = V·W_v（V 不做 ReLU）
//   分子 num_t  = Σ_{i∈S_t} (q'_t·k'_i) v_i  = B_t · q'_t,   B_t = Σ_{i∈S_t} v_i k'_i^T
//   分母 den_t  = sqrt( Σ_{i∈S_t} (q'_t·k'_i)^2 + eps )
//               = sqrt( q'_t^T A_t q'_t + eps ),  A_t = Σ_{i∈S_t} k'_i k'_i^T
//   out_t = num_t / den_t
//   其中 S_t = { i<=t }（causal）或 { 全部 }（bidirectional）。
//
// 关键性质：
//   * 复杂度 O(L·d_k²)，无 O(L²) 物化 —— 分子/分母都靠运行态前缀和。
//     分母 L2 恒等式：Σ_i (q'·k'_i)² = q'^T (Σ k'_i k'_i^T) q'（避免显式 L×L 分数矩阵）。
//   * 施加顺序必须 RoPE → ReLU（RoPE 若在 ReLU 之后，旋转产生的负值被截断 → 丢位置信息）。
//   * 数值：分母 L2 归一化（余弦式约束）；禁 1/sqrt(d_k) 缩放；稳定性靠 LayerNorm + eps。
//
// 实现说明：causal 的前缀和递推本质是序列化的，无法折叠为 matmul 批次，
//   故核心扫描用 CPU Matrix 标量循环，经 to_matrix/from_matrix 适配任意后端
//   （GPU 走 staging 往返，v1 正确性优先）。
// ══════════════════════════════════════════════════════════════════════════
class ReLULinearAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;    // 单样本序列长度（0 = 单样本，cols 即 seq）
    bool causal_;            // true=因果前缀和；false=全量双向
    bool use_rope_;          // RoPE 施加在 Q/K 进 ReLU 之前（RLA 强约束）
    RotaryEmbedding rope_;
    Linear w_q_, w_k_, w_v_, w_o_;

    // forward 缓存（backward 用）
    Tensor Qp_cache_;        // (batch*H*d_k, seq) ReLU(RoPE(Q))
    Tensor Kp_cache_;        // (batch*H*d_k, seq) ReLU(RoPE(K))
    Tensor V_re_cache_;      // (batch*H*d_k, seq) V（不 ReLU、不 RoPE）
    std::size_t batch_cache_ = 0;
    std::size_t seq_cache_   = 0;

    // 文档感知：每位置文档 id（batch-major，size = batch*seq）；空=无文档感知
    std::vector<std::size_t> doc_ids_;
    bool has_doc_ids_ = false;

    // 由 doc_ids_ 构建文档边界向量（size = batch*seq）：boundary[b*seq+t]=1
    // 表示位置 t 是文档起点（t==0 或与前一位置文档不同）。
    [[nodiscard]] std::vector<uint8_t> build_boundary_(
        std::size_t batch, std::size_t seq) const
    {
        std::vector<uint8_t> boundary;
        if (!has_doc_ids_) return boundary;
        boundary.assign(batch * seq, 0);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t t = 0; t < seq; ++t)
                if (t == 0 || doc_ids_[b * seq + t] != doc_ids_[b * seq + t - 1])
                    boundary[b * seq + t] = 1;
        return boundary;
    }

    // ── 标量线性代数助手（行主序向量化，dk×dk 矩阵） ──────────────────
    static void matvec_(const std::vector<Scalar>& M,
                        const std::vector<Scalar>& x,
                        std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t i = 0; i < dk; ++i)
        {
            Scalar acc{0};
            for (std::size_t j = 0; j < dk; ++j) acc += M[i * dk + j] * x[j];
            y[i] = acc;
        }
    }
    static void matvec_T_(const std::vector<Scalar>& M,
                          const std::vector<Scalar>& x,
                          std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t j = 0; j < dk; ++j)
        {
            Scalar acc{0};
            for (std::size_t i = 0; i < dk; ++i) acc += M[i * dk + j] * x[i];
            y[j] = acc;
        }
    }
    static void add_outer_(std::vector<Scalar>& M,
                           const std::vector<Scalar>& x,
                           const std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t i = 0; i < dk; ++i)
            for (std::size_t j = 0; j < dk; ++j)
                M[i * dk + j] += x[i] * y[j];
    }

    // 构造 RLA 掩码 (BH*seq, seq) 0/1 矩阵：mask[b*H+h][t, j] = 1 iff 允许 q_t 看 k_j
    //   causal:       j <= t
    //   文档感知:     j <= t 且 doc[j] == doc[t]（仅 causal 模式生效）
    //   bidirectional: 全 1（无掩码）
    [[nodiscard]] Result<Tensor> build_rla_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        Matrix mask(BH * seq, seq, Scalar{1});
        if (causal_)
        {
            for (std::size_t bh = 0; bh < BH; ++bh)
            {
                const std::size_t b = bh / num_heads_;
                for (std::size_t t = 0; t < seq; ++t)
                    for (std::size_t j = 0; j < seq; ++j)
                    {
                        bool ok = (j <= t);
                        if (ok && has_doc_ids_ &&
                            doc_ids_[b * seq + j] != doc_ids_[b * seq + t])
                            ok = false;
                        if (!ok) mask.set_value_unchecked(bh * seq + t, j, Scalar{0});
                    }
            }
        }
        return engine.from_matrix(mask);
    }

    // ── 原语组合版 RLA 前向（GPU 上运行，消除 PCIe 往返） ──────────────
    // 用现有引擎原语（batched_matmul / elementwise / reduce / broadcast）
    // 组合出 RLA 计算，不落 CPU。代价：物化 (BH*seq, seq) 分数矩阵（O(L²)），
    // 作为过渡实现；后续可被 IR 张量复用优化改进为 O(L·d²) 运行态。
    //
    // 输入 Qp/Kp/V: (BH*dk, seq)；返回 (BH*dk, seq)
    //   S[i,j] = q_i·k_j  (非负，Qp/Kp 经 ReLU)
    //   num = S·Vᵀ；denom = sqrt(Σ_j S² + eps)；out = num/denom
    [[nodiscard]] Result<Tensor> forward_matmul(
        ComputeEngine& engine, const Tensor& Qp, const Tensor& Kp,
        const Tensor& V, std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        const std::size_t dk = d_k_;
        auto S = engine.batched_matmul(Qp, Kp, BH, true, false, Scalar{1});   // (BH*seq, seq)
        if (!S) return std::unexpected(S.error());
        auto mask = build_rla_mask_(engine, batch, seq);                      // (BH*seq, seq)
        if (!mask) return std::unexpected(mask.error());
        auto Sm = engine.elementwise_binary(BinaryOp::Mul, *S, *mask);
        if (!Sm) return std::unexpected(Sm.error());
        auto num = engine.batched_matmul(*Sm, V, BH, false, true);            // Sm @ Vᵀ → (BH*seq, dk)
        if (!num) return std::unexpected(num.error());
        auto S2 = engine.elementwise_binary(BinaryOp::Mul, *Sm, *Sm);
        if (!S2) return std::unexpected(S2.error());
        auto denom2 = engine.row_reduce_sum(*S2);                             // (BH*seq, 1)
        if (!denom2) return std::unexpected(denom2.error());
        auto denom_e = engine.elementwise_binary_scalar(BinaryOp::Add, *denom2, Scalar{1e-6}, false);
        if (!denom_e) return std::unexpected(denom_e.error());
        auto denom = engine.elementwise_unary(UnaryOp::Sqrt, *denom_e);
        if (!denom) return std::unexpected(denom.error());
        auto out = engine.clone(*num);                                        // (BH*seq, dk)
        if (!out) return std::unexpected(out.error());
        { auto r = engine.broadcast_row_inplace(*out, *denom, BinaryOp::Div);
          if (!r) return std::unexpected(r.error()); }
        auto O_T = engine.transpose(*out);                                    // (dk, BH*seq)
        if (!O_T) return std::unexpected(O_T.error());
        return engine.rearrange_3d(*O_T, dk, BH, seq, false);                 // (BH*dk, seq)
    }

    // ── 原语组合版 RLA 反向 ──────────────────────────────────────────
    // 返回 {grad_Qp, grad_Kp, grad_V}，均 (BH*dk, seq)。与 forward_matmul 对应。
    [[nodiscard]] Result<std::array<Tensor, 3>> backward_matmul(
        ComputeEngine& engine,
        const Tensor& Qp, const Tensor& Kp, const Tensor& V,
        const Tensor& grad_out_re,   // (BH*dk, seq)
        std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        // ── 重算前向量 ──
        auto S = engine.batched_matmul(Qp, Kp, BH, true, false, Scalar{1});
        if (!S) return std::unexpected(S.error());
        auto mask = build_rla_mask_(engine, batch, seq);
        if (!mask) return std::unexpected(mask.error());
        auto Sm = engine.elementwise_binary(BinaryOp::Mul, *S, *mask);
        if (!Sm) return std::unexpected(Sm.error());
        auto num = engine.batched_matmul(*Sm, V, BH, false, true);            // Sm @ Vᵀ → (BH*seq, dk)
        if (!num) return std::unexpected(num.error());
        auto S2 = engine.elementwise_binary(BinaryOp::Mul, *Sm, *Sm);
        if (!S2) return std::unexpected(S2.error());
        auto denom2 = engine.row_reduce_sum(*S2);
        if (!denom2) return std::unexpected(denom2.error());
        auto denom_e = engine.elementwise_binary_scalar(BinaryOp::Add, *denom2, Scalar{1e-6}, false);
        if (!denom_e) return std::unexpected(denom_e.error());
        auto denom = engine.elementwise_unary(UnaryOp::Sqrt, *denom_e);
        if (!denom) return std::unexpected(denom.error());

        // ── dO_t: (BH*seq, dk) ──
        auto dO_T = engine.transpose(grad_out_re);
        if (!dO_T) return std::unexpected(dO_T.error());
        auto dO_t = engine.rearrange_3d(*dO_T, seq, BH, d_k_, false);
        if (!dO_t) return std::unexpected(dO_t.error());

        // dnum = dO / denom（按行广播）
        auto dnum = engine.clone(*dO_t);
        if (!dnum) return std::unexpected(dnum.error());
        { auto r = engine.broadcast_row_inplace(*dnum, *denom, BinaryOp::Div);
          if (!r) return std::unexpected(r.error()); }

        // ddenom = -rowsum(dO·num) / denom²
        auto prod = engine.elementwise_binary(BinaryOp::Mul, *dO_t, *num);
        if (!prod) return std::unexpected(prod.error());
        auto dot = engine.row_reduce_sum(*prod);
        if (!dot) return std::unexpected(dot.error());
        auto negdot = engine.elementwise_unary(UnaryOp::Neg, *dot);
        if (!negdot) return std::unexpected(negdot.error());
        auto denom_sq = engine.elementwise_binary(BinaryOp::Mul, *denom, *denom);
        if (!denom_sq) return std::unexpected(denom_sq.error());
        auto ddenom = engine.elementwise_binary(BinaryOp::Div, *negdot, *denom_sq);
        if (!ddenom) return std::unexpected(ddenom.error());
        // ddenom2 = ddenom / (2·denom)
        auto two_denom = engine.elementwise_binary_scalar(BinaryOp::Mul, *denom, Scalar{2}, false);
        if (!two_denom) return std::unexpected(two_denom.error());
        auto ddenom2 = engine.elementwise_binary(BinaryOp::Div, *ddenom, *two_denom);
        if (!ddenom2) return std::unexpected(ddenom2.error());

        // dSm = dSm_num + dSm_den
        //   dSm_num = dnum @ V；dSm_den = 2·Sm·ddenom2（按行广播）
        auto dSm_num = engine.batched_matmul(*dnum, V, BH, false, false);
        if (!dSm_num) return std::unexpected(dSm_num.error());
        auto dSm_den = engine.clone(*Sm);
        if (!dSm_den) return std::unexpected(dSm_den.error());
        { auto r = engine.broadcast_row_inplace(*dSm_den, *ddenom2, BinaryOp::Mul);
          if (!r) return std::unexpected(r.error()); }
        dSm_den = engine.elementwise_binary_scalar(BinaryOp::Mul, *dSm_den, Scalar{2}, false);
        if (!dSm_den) return std::unexpected(dSm_den.error());
        auto dSm = engine.elementwise_binary(BinaryOp::Add, *dSm_num, *dSm_den);
        if (!dSm) return std::unexpected(dSm.error());
        // dS = dSm · mask（掩码是常数 0/1 乘法）
        auto dS = engine.elementwise_binary(BinaryOp::Mul, *dSm, *mask);
        if (!dS) return std::unexpected(dS.error());

        // ── 梯度 Q/K/V ──
        auto grad_Qp = engine.batched_matmul(Kp, *dS, BH, false, true);    // Kp @ dSᵀ (BH*dk, seq)
        if (!grad_Qp) return std::unexpected(grad_Qp.error());
        auto grad_Kp = engine.batched_matmul(Qp, *dS, BH, false, false);   // Qp @ dS  (BH*dk, seq)
        if (!grad_Kp) return std::unexpected(grad_Kp.error());
        auto grad_V = engine.batched_matmul(*dnum, *Sm, BH, true, false);  // dnumᵀ @ Sm → (BH*dk, seq)
        if (!grad_V) return std::unexpected(grad_V.error());

        return std::array<Tensor, 3>{std::move(*grad_Qp), std::move(*grad_Kp), std::move(*grad_V)};
    }

    // ── 前向扫描：out[bh*dk+j][t] = RLA 输出 ──────────────────────────
    // boundary（可选，空=无文档感知）：size = batch*seq，boundary[b*seq+t]=1
    // 表示位置 t 是文档起点。causal 模式下文档边界处重置运行态 A/B，
    // 使每个 token 只聚合本文档内前缀（文档感知 = 运行态重置于边界）。
    static void scan_forward_(const Matrix& Q, const Matrix& K, const Matrix& V,
                              Matrix& out, std::size_t dk, std::size_t BH,
                              std::size_t seq, bool causal,
                              std::size_t num_heads,
                              const std::vector<uint8_t>& boundary,
                              Scalar eps)
    {
        std::vector<Scalar> A(dk * dk), B(dk * dk), qv(dk), kv(dk), vv(dk),
                            num(dk), Aq(dk);
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            const std::size_t r0 = bh * dk;
            const std::size_t batch = bh / num_heads;
            const auto doc_reset = [&](std::size_t t) {
                return !boundary.empty() && boundary[batch * seq + t] != 0;
            };
            std::fill(A.begin(), A.end(), Scalar{0});
            std::fill(B.begin(), B.end(), Scalar{0});
            if (!causal)  // 双向：先求全集 A, B
            {
                for (std::size_t t = 0; t < seq; ++t)
                {
                    for (std::size_t j = 0; j < dk; ++j)
                    {
                        kv[j] = K.at_unchecked(r0 + j, t);
                        vv[j] = V.at_unchecked(r0 + j, t);
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
            }
            for (std::size_t t = 0; t < seq; ++t)
            {
                for (std::size_t j = 0; j < dk; ++j)
                {
                    qv[j] = Q.at_unchecked(r0 + j, t);
                    kv[j] = K.at_unchecked(r0 + j, t);
                    vv[j] = V.at_unchecked(r0 + j, t);
                }
                if (causal)  // 前缀和含自身 i<=t；文档边界处重置
                {
                    if (doc_reset(t))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
                matvec_(B, qv, num, dk);
                matvec_(A, qv, Aq, dk);
                Scalar s{0};
                for (std::size_t j = 0; j < dk; ++j) s += qv[j] * Aq[j];
                const Scalar denom = std::sqrt(s + eps);
                for (std::size_t j = 0; j < dk; ++j)
                    out.set_value_unchecked(r0 + j, t, num[j] / denom);
            }
        }
    }

    // ── 反向扫描：G 为 grad_out_re，写 gQ/gK/gV（均 (BH*dk, seq)） ────
    // 前向重算运行态；dL/dA_t=dL/ds_t·q_tq_t^T，dL/dB_t=outer(dL/dnum_t,q_t)；
    // 后缀和得 dL/dk_i = 2·SA·k_i + SB^T·v_i，dL/dv_i = SB·k_i。
    static void scan_backward_(const Matrix& Q, const Matrix& K, const Matrix& V,
                               const Matrix& G, Matrix& gQ, Matrix& gK, Matrix& gV,
                               std::size_t dk, std::size_t BH, std::size_t seq,
                               bool causal, std::size_t num_heads,
                               const std::vector<uint8_t>& boundary, Scalar eps)
    {
        std::vector<Scalar> A(dk * dk), B(dk * dk), qv(dk), kv(dk), vv(dk),
                            num(dk), Aq(dk), gnum(dk), tmp(dk), tmp2(dk);
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            const std::size_t r0 = bh * dk;
            const std::size_t batch = bh / num_heads;
            const auto doc_reset = [&](std::size_t t) {
                return !boundary.empty() && boundary[batch * seq + t] != 0;
            };
            std::vector<Scalar> store_q(seq * dk), store_dq(seq * dk),
                                store_dA(seq * dk * dk), store_dB(seq * dk * dk);
            std::vector<Scalar> dA_total(dk * dk, Scalar{0}),
                                dB_total(dk * dk, Scalar{0});
            std::fill(A.begin(), A.end(), Scalar{0});
            std::fill(B.begin(), B.end(), Scalar{0});
            if (!causal)
            {
                for (std::size_t t = 0; t < seq; ++t)
                {
                    for (std::size_t j = 0; j < dk; ++j)
                    {
                        kv[j] = K.at_unchecked(r0 + j, t);
                        vv[j] = V.at_unchecked(r0 + j, t);
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
            }
            // Pass 1：重算运行态，存每位置 q_t、dL/dq_t、dL/dA_t、dL/dB_t
            for (std::size_t t = 0; t < seq; ++t)
            {
                for (std::size_t j = 0; j < dk; ++j)
                {
                    qv[j] = Q.at_unchecked(r0 + j, t);
                    kv[j] = K.at_unchecked(r0 + j, t);
                    vv[j] = V.at_unchecked(r0 + j, t);
                }
                if (causal)
                {
                    if (doc_reset(t))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
                matvec_(B, qv, num, dk);
                matvec_(A, qv, Aq, dk);
                Scalar s{0};
                for (std::size_t j = 0; j < dk; ++j) s += qv[j] * Aq[j];
                const Scalar denom = std::sqrt(s + eps);
                const Scalar denom_inv = Scalar{1} / denom;
                Scalar gdot{0};
                for (std::size_t j = 0; j < dk; ++j)
                {
                    gnum[j] = G.at_unchecked(r0 + j, t) * denom_inv;
                    gdot   += G.at_unchecked(r0 + j, t) * num[j];
                }
                const Scalar ddenom = -gdot / (denom * denom);
                const Scalar ds = ddenom * (Scalar{0.5} * denom_inv);
                matvec_T_(B, gnum, tmp, dk);   // tmp = B^T·gnum
                for (std::size_t j = 0; j < dk; ++j)
                {
                    store_q[t * dk + j]  = qv[j];
                    store_dq[t * dk + j] = tmp[j] + Scalar{2} * ds * Aq[j];
                }
                for (std::size_t a = 0; a < dk; ++a)
                    for (std::size_t b = 0; b < dk; ++b)
                    {
                        const Scalar dA_ab = ds * qv[a] * qv[b];
                        const Scalar dB_ab = gnum[a] * qv[b];
                        if (causal)
                        {
                            store_dA[(t * dk + a) * dk + b] = dA_ab;
                            store_dB[(t * dk + a) * dk + b] = dB_ab;
                        }
                        else
                        {
                            // 双向：A、B 为全集常数，dL/dA、dL/dB 是所有 t 求和
                            dA_total[a * dk + b] += dA_ab;
                            dB_total[a * dk + b] += dB_ab;
                        }
                    }
            }
            // Pass 2：反向
            std::fill(A.begin(), A.end(), Scalar{0});  // 复用 A/B 作累积
            std::fill(B.begin(), B.end(), Scalar{0});
            for (std::size_t i = seq; i-- > 0;)
            {
                if (causal)
                {
                    // 后缀和：SA = Σ_{t>=i} dL/dA_t，SB = Σ_{t>=i} dL/dB_t
                    // 文档感知：跨入前一文档时重置后缀和（只统计本文档内 t>=i）
                    if (i + 1 < seq && doc_reset(i + 1))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    for (std::size_t idx = 0; idx < dk * dk; ++idx)
                    {
                        A[idx] += store_dA[i * dk * dk + idx];
                        B[idx] += store_dB[i * dk * dk + idx];
                    }
                }
                else
                {
                    // 双向：A、B 即全集梯度 dL/dA、dL/dB（对每个 i 相同）
                    std::copy(dA_total.begin(), dA_total.end(), A.begin());
                    std::copy(dB_total.begin(), dB_total.end(), B.begin());
                }
                for (std::size_t j = 0; j < dk; ++j)
                {
                    kv[j] = K.at_unchecked(r0 + j, i);
                    vv[j] = V.at_unchecked(r0 + j, i);
                }
                matvec_(A, kv, tmp, dk);      // SA·k_i
                matvec_T_(B, vv, tmp2, dk);   // SB^T·v_i
                for (std::size_t j = 0; j < dk; ++j)
                    gK.set_value_unchecked(r0 + j, i, Scalar{2} * tmp[j] + tmp2[j]);
                matvec_(B, kv, tmp2, dk);     // SB·k_i
                for (std::size_t j = 0; j < dk; ++j)
                    gV.set_value_unchecked(r0 + j, i, tmp2[j]);
                for (std::size_t j = 0; j < dk; ++j)
                    gQ.set_value_unchecked(r0 + j, i, store_dq[i * dk + j]);
            }
        }
    }

public:
    ReLULinearAttention(std::size_t d_model, std::size_t num_heads,
                        std::size_t seq_len = 0,
                        bool causal = true,
                        PosEncodingType pos_enc = PosEncodingType::RoPE)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len), causal_(causal),
          use_rope_(pos_enc == PosEncodingType::RoPE),
          rope_(d_model / num_heads),
          w_q_(d_model, d_model), w_k_(d_model, d_model),
          w_v_(d_model, d_model), w_o_(d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ReLULinearAttention: d_model must be divisible by num_heads");
        NN_ASSERT(d_model % num_heads == 0 && (d_model / num_heads) % 2 == 0,
                  "ReLULinearAttention: RoPE requires even d_k");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = w_q_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_k_.init(engine); if (!r2) return std::unexpected(r2.error());
        auto r3 = w_v_.init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = w_o_.init(engine); if (!r4) return std::unexpected(r4.error());
        if (use_rope_)
        {
            auto r5 = rope_.init(engine); if (!r5) return std::unexpected(r5.error());
        }
        return {};
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

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        Qp_cache_ = Tensor{};
        Kp_cache_ = Tensor{};
        V_re_cache_ = Tensor{};
        batch_cache_ = 0;
        seq_cache_ = 0;
        w_q_.clear_cache(); w_k_.clear_cache();
        w_v_.clear_cache(); w_o_.clear_cache();
    }

    // 文档感知：记录每位置文档 id（batch-major），forward/backward 据此
    // 在文档边界重置运行态（每个 token 只聚合本文档内前缀）。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); has_doc_ids_ = false; return; }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

    // 绝对位置偏移（滑动窗生成用）：转发给 RoPE，使重计算式生成的
    // 位置从真实起点算起而非每次从 0 重置。
    void set_position_offset(std::size_t off)
    {
        if (use_rope_) rope_.set_position_offset(off);
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Qp_cache_.valid()) r.emplace_back(Qp_cache_);
        if (Kp_cache_.valid()) r.emplace_back(Kp_cache_);
        if (V_re_cache_.valid()) r.emplace_back(V_re_cache_);
        auto wq = w_q_.activation_cache(); r.insert(r.end(), wq.begin(), wq.end());
        auto wk = w_k_.activation_cache(); r.insert(r.end(), wk.begin(), wk.end());
        auto wv = w_v_.activation_cache(); r.insert(r.end(), wv.begin(), wv.end());
        auto wo = w_o_.activation_cache(); r.insert(r.end(), wo.begin(), wo.end());
        return r;
    }

    // 前向：输入 X (d_model, batch·seq) → 输出 (d_model, batch·seq)
    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"ReLULinearAttention forward: input shape mismatch"});

        const std::size_t total = input.cols();
        const std::size_t seq = (seq_len_ > 0) ? seq_len_ : total;
        const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;
        if (total != batch * seq)
            return std::unexpected(Error{"ReLULinearAttention forward: cols not divisible by seq_len"});
        const std::size_t H_dk = num_heads_ * d_k_;

        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        Tensor Q, K, V;   // (BH*dk, seq) rearranged
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false); if (!vr) return std::unexpected(vr.error());
            V = std::move(*vr);
        }
        else
        {
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // RoPE → ReLU（顺序必须：先旋转后截断，否则丢位置信息）
        if (use_rope_)
        {
            auto qr = rope_.apply(engine, Q, seq, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply(engine, K, seq, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = engine.elementwise_binary_scalar(BinaryOp::Max, Q, Scalar{0}, false);
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = engine.elementwise_binary_scalar(BinaryOp::Max, K, Scalar{0}, false);
        if (!Kp) return std::unexpected(Kp.error());

        // 核心扫描（CPU 标量；任意后端经 staging）
        // 原语组合版 RLA 前向（GPU 上运行，消除 PCIe 往返）
        auto out_t = forward_matmul(engine, *Qp, *Kp, V, batch, seq);
        if (!out_t) return std::unexpected(out_t.error());

        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(*out_t, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(*out_t);
        }

        Qp_cache_ = std::move(*Qp);
        Kp_cache_ = std::move(*Kp);
        V_re_cache_ = std::move(V);
        batch_cache_ = batch;
        seq_cache_   = seq;

        return w_o_.forward(engine, concat);
    }

    // 反向：grad_output (d_model, batch·seq) → 输入梯度 (d_model, batch·seq)
    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq = seq_cache_;
        const std::size_t batch = batch_cache_;
        const std::size_t H_dk = num_heads_ * d_k_;

        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;
        Tensor gcr;
        if (batch > 1)
        {
            auto g = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!g) return std::unexpected(g.error());
            gcr = std::move(*g);
        }
        else
        {
            gcr = std::move(*gc);
        }

        // 原语组合版 RLA 反向（GPU 上运行）
        auto gres = backward_matmul(engine, Qp_cache_, Kp_cache_, V_re_cache_,
                                    gcr, batch, seq);
        if (!gres) return std::unexpected(gres.error());
        auto gQt = std::move((*gres)[0]);
        auto gKt = std::move((*gres)[1]);
        auto gVt = std::move((*gres)[2]);

        // ReLU 反向：(x>0) ? g : 0（x>0 与 Qp/Kp>0 同号）
        auto gq_relu = engine.elementwise_select_scalar_cond(
            CompareOp::Gt, Qp_cache_, Scalar{0}, gQt, Scalar{0});
        if (!gq_relu) return std::unexpected(gq_relu.error());
        auto gk_relu = engine.elementwise_select_scalar_cond(
            CompareOp::Gt, Kp_cache_, Scalar{0}, gKt, Scalar{0});
        if (!gk_relu) return std::unexpected(gk_relu.error());

        // RoPE 反向（旋转正交，逆 = 反角）
        if (use_rope_)
        {
            auto gq = rope_.apply(engine, *gq_relu, seq, true);
            if (!gq) return std::unexpected(gq.error());
            gq_relu = std::move(*gq);
            auto gk = rope_.apply(engine, *gk_relu, seq, true);
            if (!gk) return std::unexpected(gk.error());
            gk_relu = std::move(*gk);
        }

        Tensor gq_r, gk_r, gv_r;
        if (batch > 1)
        {
            auto a = engine.rearrange_3d(*gq_relu, H_dk, batch, seq, true); if (!a) return std::unexpected(a.error());
            gq_r = std::move(*a);
            auto b = engine.rearrange_3d(*gk_relu, H_dk, batch, seq, true); if (!b) return std::unexpected(b.error());
            gk_r = std::move(*b);
            auto c = engine.rearrange_3d(gVt, H_dk, batch, seq, true); if (!c) return std::unexpected(c.error());
            gv_r = std::move(*c);
        }
        else
        {
            gq_r = std::move(*gq_relu);
            gk_r = std::move(*gk_relu);
            gv_r = std::move(gVt);
        }

        auto giq = w_q_.backward(engine, gq_r);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);
        auto gik = w_k_.backward(engine, gk_r);
        if (!gik) return gik;
        { auto r = engine.add_inplace(grad_input, *gik); if (!r) return std::unexpected(r.error()); }
        auto giv = w_v_.backward(engine, gv_r);
        if (!giv) return giv;
        { auto r = engine.add_inplace(grad_input, *giv); if (!r) return std::unexpected(r.error()); }
        return grad_input;
    }

    // ── 增量推理（运行态 KV cache） ──────────────────────────────
    // 处理单个新 token，增量更新运行态 A/B（RLA 的"KV cache"），O(d_k²) 每步，
    // 使自回归生成真正线性于上下文长度（无需逐窗重算）。
    //   A_state/B_state: (num_heads*d_k, d_k)，行块 h 分别 = Σ k'_h k'_h^T、Σ v_h k'_h^T
    //   pos: 绝对位置（RoPE 施加于 Q/K 进 ReLU 前）
    // 输入 input: (d_model, 1) 单 token 隐藏态；返回 (d_model, 1) 经 w_o 输出。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Matrix& A_state, Matrix& B_state, std::size_t pos)
    {
        auto q_res = w_q_.forward(engine, input);   // (d_model, 1) = (H*dk, 1)
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;
        Tensor Q = std::move(*q_res), K = std::move(*k_res), V = std::move(*v_res);
        if (use_rope_)
        {
            auto qr = rope_.apply_step(engine, Q, pos, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply_step(engine, K, pos, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = engine.elementwise_binary_scalar(BinaryOp::Max, Q, Scalar{0}, false);
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = engine.elementwise_binary_scalar(BinaryOp::Max, K, Scalar{0}, false);
        if (!Kp) return std::unexpected(Kp.error());
        auto Qm = engine.to_matrix(*Qp); if (!Qm) return std::unexpected(Qm.error());
        auto Km = engine.to_matrix(*Kp); if (!Km) return std::unexpected(Km.error());
        auto Vm = engine.to_matrix(V);   if (!Vm) return std::unexpected(Vm.error());
        const std::size_t H = num_heads_;
        const std::size_t dk = d_k_;
        Matrix out_m(H * dk, 1, Scalar{0});
        std::vector<Scalar> num(dk), Aq(dk);
        for (std::size_t h = 0; h < H; ++h)
        {
            const std::size_t r0 = h * dk;
            // 更新运行态：A[h] += k'k'^T，B[h] += v k'^T（因果：先累积再用）
            for (std::size_t a = 0; a < dk; ++a)
                for (std::size_t b = 0; b < dk; ++b)
                {
                    const Scalar ka = Km->at_unchecked(r0 + a, 0);
                    const Scalar kb = Km->at_unchecked(r0 + b, 0);
                    A_state.set_value_unchecked(r0 + a, b,
                        A_state.at_unchecked(r0 + a, b) + ka * kb);
                    B_state.set_value_unchecked(r0 + a, b,
                        B_state.at_unchecked(r0 + a, b) + Vm->at_unchecked(r0 + a, 0) * kb);
                }
            // num = B[h]·q'；Aq = A[h]·q'；s = q'·Aq；out = num / sqrt(s+eps)
            for (std::size_t i = 0; i < dk; ++i)
            {
                Scalar n{0}, aq{0};
                for (std::size_t b = 0; b < dk; ++b)
                {
                    n  += B_state.at_unchecked(r0 + i, b) * Qm->at_unchecked(r0 + b, 0);
                    aq += A_state.at_unchecked(r0 + i, b) * Qm->at_unchecked(r0 + b, 0);
                }
                num[i] = n; Aq[i] = aq;
            }
            Scalar s{0};
            for (std::size_t i = 0; i < dk; ++i) s += Qm->at_unchecked(r0 + i, 0) * Aq[i];
            const Scalar denom = std::sqrt(s + Scalar{1e-6});
            for (std::size_t i = 0; i < dk; ++i)
                out_m.set_value_unchecked(r0 + i, 0, num[i] / denom);
        }
        auto out_t = engine.from_matrix(out_m);
        if (!out_t) return std::unexpected(out_t.error());
        return w_o_.forward(engine, *out_t);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTBlock — RAPT 解码器块（GPT 风格：Norm → RLA 注意力 → 残差 → FFN → 残差）
// ══════════════════════════════════════════════════════════════════════════
class RAPTBlock final : public Layer
{
private:
    std::unique_ptr<Layer> norm1_;
    ReLULinearAttention attn_;
    std::unique_ptr<Layer> norm2_;
    FeedForward ff_;

public:
    RAPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
              std::size_t seq_len, PosEncodingType pos_enc,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true)
        : norm1_(make_norm_layer(d_model, norm_type)),
          attn_(d_model, num_heads, seq_len, causal, pos_enc),
          norm2_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation)
    {
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = norm1_->init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = attn_.init(engine);   if (!r2) return std::unexpected(r2.error());
        auto r3 = norm2_->init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = ff_.init(engine);     if (!r4) return std::unexpected(r4.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        auto n1 = norm1_->parameters(); p.insert(p.end(), n1.begin(), n1.end());
        auto a = attn_.parameters();    p.insert(p.end(), a.begin(), a.end());
        auto n2 = norm2_->parameters(); p.insert(p.end(), n2.begin(), n2.end());
        auto f = ff_.parameters();      p.insert(p.end(), f.begin(), f.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        auto n1 = norm1_->param_gradients(); g.insert(g.end(), n1.begin(), n1.end());
        auto a = attn_.param_gradients();    g.insert(g.end(), a.begin(), a.end());
        auto n2 = norm2_->param_gradients(); g.insert(g.end(), n2.begin(), n2.end());
        auto f = ff_.param_gradients();      g.insert(g.end(), f.begin(), f.end());
        return g;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        attn_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        norm1_->clear_cache(); attn_.clear_cache();
        norm2_->clear_cache(); ff_.clear_cache();
    }

    // 文档感知：转发给内部 RLA 注意力（文档边界处重置运行态）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        attn_.set_doc_ids(ids);
    }

    // 绝对位置偏移：转发给内部 RLA 注意力（滑动窗生成时 RoPE 用绝对位置）
    void set_position_offset(std::size_t off)
    {
        attn_.set_position_offset(off);
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;
        auto a = attn_.forward(engine, *n1);
        if (!a) return a;
        auto r1 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
        return engine.elementwise_binary(BinaryOp::Add, *r1, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;
        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());
        auto grad_a = attn_.backward(engine, *grad_r1);
        if (!grad_a) return grad_a;
        auto b_n1 = norm1_->backward(engine, *grad_a);
        if (!b_n1) return b_n1;
        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }

    // 增量推理：单 token 经 norm1 → RLA 运行态注意力 → 残差 → norm2 → FFN → 残差
    // A/B_state: RLA 运行态 KV cache（见 ReLULinearAttention::forward_step）
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Matrix& A_state, Matrix& B_state, std::size_t pos)
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;
        auto a = attn_.forward_step(engine, *n1, A_state, B_state, pos);
        if (!a) return a;
        auto r1 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
        return engine.elementwise_binary(BinaryOp::Add, *r1, *f);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTModel — RAPT 解码器（ReLU 激活线性注意力语言模型）
//
//   token_emb → (+pos_enc，RoPE 在注意力内部施加) → N × RAPTBlock → LN → LM Head
//
// RLA 约束：位置编码必须用 RoPE（或 ALiBi），且 RoPE 施加在 Q/K 进 ReLU 之前；
// 本实现强制 RoPE（v1 不支持 ALiBi），输入侧用 NoPositionEncoder（无位置嵌入）。
// ══════════════════════════════════════════════════════════════════════════
class RAPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;
    std::size_t num_heads_;   // 用于运行态 KV cache 尺寸（d_k = d_model/num_heads）
    // （d_ff_/num_layers_/causal_ 仅构造时用，不存成员避免 -Wunused）

    Tensor token_emb_;
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;
    std::vector<RAPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    Tensor stored_tokens_tensor_;
    std::size_t batch_size_ = 0;
    std::vector<std::size_t> doc_ids_;   // 文档感知：每位置文档 id（batch-major）

public:
    RAPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
              std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
              PosEncodingType pos_enc = PosEncodingType::RoPE,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          num_heads_(num_heads),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, pos_enc,
                                 activation, norm_type, causal);
        // RLA 强约束：必须 RoPE（或 ALiBi）。v1 强制 RoPE，输入侧无位置嵌入。
        pos_encoder_ = std::make_unique<NoPositionEncoder>();
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix te(vocab_size_, d_model_);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);
        auto te_r = engine.from_matrix(te);
        if (!te_r) return std::unexpected(te_r.error());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_);
        { auto r = engine.zero(grad_token_emb_); if (!r) return std::unexpected(r.error()); }

        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        for (auto& b : blocks_)
        {
            auto r = b.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        if (ln_f_)
        {
            auto r = ln_f_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        { auto r = lm_head_.init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        auto gp = pos_encoder_->param_gradients();
        g.insert(g.end(), gp.begin(), gp.end());
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_->param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        token_emb_ = Tensor{};
        grad_token_emb_ = Tensor{};
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        stored_tokens_tensor_ = Tensor{};
        batch_size_ = 0;
    }

    // 文档感知：记录 doc_ids，forward 时下发给各块（文档边界处重置 RLA 运行态）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    // 绝对位置偏移：下发给各块（滑动窗生成时 RoPE 用绝对位置，非 0..seq-1 重置）
    void set_position_offset(std::size_t off)
    {
        for (auto& b : blocks_) b.set_position_offset(off);
    }

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq = input.rows();
        const std::size_t batch = input.cols();
        batch_size_ = batch;

        auto input_T = engine.transpose(input);   // (batch, seq)
        if (!input_T) return std::unexpected(input_T.error());
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());
        auto st = engine.clone(*input_T);
        if (!st) return std::unexpected(st.error());
        stored_tokens_tensor_ = std::move(*st);
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        auto x_res = pos_encoder_->apply(engine, *all_T, batch, seq);
        if (!x_res) return std::unexpected(x_res.error());
        Tensor x = std::move(*x_res);

        // 文档感知：每块 forward 前下发本步 doc_ids（边界重置运行态）
        for (auto& b : blocks_)
        {
            if (!doc_ids_.empty()) b.set_doc_ids(doc_ids_);
            auto r = b.forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
        }
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq = seq_len_;
        const std::size_t batch = batch_size_;

        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        auto b_ln = ln_f_->backward(engine, *b_lm);
        if (!b_ln) return b_ln;
        Tensor grad_x = std::move(*b_ln);

        for (std::size_t i = blocks_.size(); i-- > 0;)
        {
            auto br = blocks_[i].backward(engine, grad_x);
            if (!br) return br;
            grad_x = std::move(*br);
        }

        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());
        auto pr = pos_encoder_->backward(engine, *grad_T, batch, seq);
        if (!pr) return std::unexpected(pr.error());
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        Matrix grad_input(seq, batch, Scalar{0});
        return engine.from_matrix(grad_input);
    }

    // ── 采样生成（增量运行态，KV cache） ────────────────────────
    // 利用 RLA 的运行态前缀和（A_t、B_t）作为天然 KV cache：逐 token 增量更新，
    // 每步 O(d²)（不重算前文），使生成真正线性于上下文长度，且无滑动窗截断
    // （绝对位置经 pos 计数器自然递增，RoPE 位置恒正确）。
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
             const std::vector<std::size_t>& prompt,
             std::size_t max_new_tokens,
             Scalar temperature = 1.0,
             std::size_t eos_token_id = static_cast<std::size_t>(-1),
             std::size_t min_new_tokens = 0)
    {
        if (prompt.empty())
            return std::unexpected(Error{"RAPT generate: empty prompt"});
        const std::size_t dk = d_model_ / num_heads_;
        // 每块一个运行态 KV cache（A/B，形状 (d_model, d_k)，行块按头划分）
        std::vector<Matrix> statesA, statesB;
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            statesA.emplace_back(d_model_, dk, Scalar{0});
            statesB.emplace_back(d_model_, dk, Scalar{0});
        }

        // 处理单个 token：embed → 各块 forward_step → LN → LM head → 返回 logits 列
        auto step_one = [&](std::size_t tok, std::size_t pos)
            -> Result<std::vector<Scalar>>
        {
            Matrix idx(1, 1);
            idx.set_value_unchecked(0, 0, static_cast<Scalar>(tok));
            auto idx_t = engine.from_matrix(idx);
            if (!idx_t) return std::unexpected(idx_t.error());
            auto emb = engine.gather_rows(token_emb_, *idx_t);   // (1, d_model)
            if (!emb) return std::unexpected(emb.error());
            auto emb_t = engine.transpose(*emb);                  // (d_model, 1)
            if (!emb_t) return std::unexpected(emb_t.error());
            Tensor h = std::move(*emb_t);
            for (std::size_t i = 0; i < blocks_.size(); ++i)
            {
                auto r = blocks_[i].forward_step(engine, h, statesA[i], statesB[i], pos);
                if (!r) return std::unexpected(r.error());
                h = std::move(*r);
            }
            auto ln = ln_f_->forward(engine, h);
            if (!ln) return std::unexpected(ln.error());
            auto logits = lm_head_.forward(engine, *ln);
            if (!logits) return std::unexpected(logits.error());
            auto lm = engine.to_matrix(*logits);
            if (!lm) return std::unexpected(lm.error());
            std::vector<Scalar> last(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last[v] = lm->at_unchecked(v, 0);
            return last;
        };

        // 逐 token 处理 prompt（建立运行态）；保留最后一步 logits（预测下一 token）
        std::size_t pos = 0;
        std::vector<Scalar> last;
        for (std::size_t i = 0; i < prompt.size(); ++i)
        {
            auto r = step_one(prompt[i], pos++);
            if (!r) return std::unexpected(r.error());
            last = std::move(*r);
        }

        // 采样生成
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            std::vector<Scalar> lastv = last;
            if (temperature > 0.0 && temperature != 1.0)
                for (auto& x : lastv) x /= temperature;

            Scalar max_val = lastv[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, lastv[v]);
            Scalar sum_exp = 0;
            for (auto& x : lastv) { x = std::exp(x - max_val); sum_exp += x; }
            for (auto& x : lastv) x /= sum_exp;

            std::size_t next;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cum = 0;
                next = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cum += lastv[v];
                    if (r <= cum) { next = v; break; }
                }
            }
            else
            {
                next = 0;
                Scalar best = lastv[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (lastv[v] > best) { best = lastv[v]; next = v; }
            }

            if (step >= min_new_tokens && next == eos_token_id) break;
            generated.push_back(next);

            auto r = step_one(next, pos++);
            if (!r) return std::unexpected(r.error());
            last = std::move(*r);
        }
        return generated;
    }
};

} // namespace nn

#endif // NN_COMPUTE_LAYER_HPP
