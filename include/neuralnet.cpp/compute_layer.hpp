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
        const std::size_t fan_in = in_channels * kernel * kernel;
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

    // 重建全表 (d_k, seq)：cos/sin 按维度对交错重复
    [[nodiscard]] Result<void> rebuild(ComputeEngine& engine, std::size_t seq)
    {
        Matrix c(d_k_, seq), s(d_k_, seq);
        for (std::size_t pos = 0; pos < seq; ++pos)
            fill_pos_column_(c, s, pos, pos);
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
    std::mt19937_64* rng_;           // 非拥有指针
    std::normal_distribution<Scalar>* dist_;  // 非拥有指针

public:
    LearnedPositionEncoder(std::size_t d_model, std::size_t seq_len,
                           std::mt19937_64& rng,
                           std::normal_distribution<Scalar>& dist)
        : d_model_(d_model), seq_len_(seq_len), rng_(&rng), dist_(&dist) {}

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix pe(seq_len_, d_model_);
        auto pe_s = pe.span();
        for (std::size_t i = 0; i < pe.size(); ++i) pe_s[i] = (*dist_)(*rng_);
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
                {
                    std::mt19937_64 rng{42};
                    std::normal_distribution<Scalar> dist(0.0, 0.02);
                    pos_encoder_ = std::make_unique<LearnedPositionEncoder>(
                        d_model, seq_len, rng, dist);
                }
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

} // namespace nn

#endif // NN_COMPUTE_LAYER_HPP
