#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_layer_base.hpp"
#include "compute_layer_bitlinear.hpp"
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

public:
    Linear(std::size_t in_features, std::size_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    [[nodiscard]] const char* layer_name() const noexcept override { return "Linear"; }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // ── 声明式初始化（M2，17 §4.4）：层算分布参数，引擎填数 ──────────
        // Xavier 均匀分布；精度 = p_.param（§9.2：f16 配置按 p_.param 舍入
        // 存储、体积减半，f32 配置保持原值）。分布 seed 显式传（U1）。
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features_ + out_features_));
        w_ = engine.create_tensor(out_features_, in_features_, p_.param,
                                  InitSpec::uniform(-limit, limit, kInitSeed));
        if (!w_.valid()) NN_FAIL("Linear: 权重初始化失败");
        b_ = engine.create_tensor(out_features_, 1, p_.param, InitSpec::zero());
        if (!b_.valid()) NN_FAIL("Linear: 偏置初始化失败");
        grad_w_ = engine.create_tensor(out_features_, in_features_, p_.param, InitSpec::zero());
        grad_b_ = engine.create_tensor(out_features_, 1, p_.param, InitSpec::zero());
        if (!grad_w_.valid() || !grad_b_.valid())
            NN_FAIL("Linear: 梯度缓冲初始化失败");
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

    // ── forward: 一行代码，精度由 p_.compute 决定 ──────────────────────
    // 引擎内部处理 matmul + broadcast bias 的精度问题
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != w_.cols())
            NN_FAIL("linear forward: input shape mismatch");
        if (!checkpoint_mode_)
            input_cache_ = input;

        // 计算精度 = p_.compute。Layer 直写该表达式：matmul 段 + row_broadcast
        // bias 融合为单 kernel（与 engine.matmul_with_bias 的 DSL 融合结构一致，
        // 见 compute_cpu_engine.hpp）；f16 存储经基类边界 cast 入口（NVI）
        // （边界 cast 或 in-kernel f16 变体，内部按 f32 计算、输出按目标精度舍入）。
        return dsl::compute(engine,
            dsl::matmul(w_, input, false, false) + dsl::row_broadcast(b_),
            w_.rows(), input.cols(), p_.compute);
    }

    // ── backward: 同样简洁，精度由引擎处理 ────────────────────────────
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (grad_output.rows() != w_.rows())
            NN_FAIL("linear backward: grad_output shape mismatch");
        // 计算精度 = p_.compute（in-place 累加的目标精度 = grad_w_ 的存储精度，
        // compute_into 无需 P：§8.3 in-place 存储精度不可变）
        // grad_input = W^T × grad_output：纯 matmul 段（无尾链），由 dsl::compute
        // 直写（scan 的 Linear backward dry-run 自动登记该结构）。
        auto grad_input = dsl::compute(engine,
            dsl::matmul(w_, grad_output, true, false),
            w_.cols(), grad_output.cols(), p_.compute);
        NN_TRY_CHECK(grad_input);
        nn_dbg_scan("lin.grad_out", engine, grad_output);
        nn_dbg_scan("lin.cache", engine, input_cache_);
        nn_dbg_scan("lin.grad_w(pre)", engine, grad_w_);
        nn_dbg_scan("lin.grad_b(pre)", engine, grad_b_);
        nn_dbg_scan("lin.grad_in", engine, *grad_input);

        // grad_w += grad_output × input^T：matmul 段与累加**融合为单次 dispatch**
        // 并原地写入 grad_w_（GPU 上 1 个融合 kernel：不物化 gw (out,in)，
        // 也不额外分配输出缓冲）。
        // k（求和维度 = batch 大小）是形状参数，不进 key → 同一 shader 适配任意 batch。
        auto grad_w_acc = dsl::compute_into(engine,
            dsl::leaf(grad_w_) + dsl::matmul(grad_output, input_cache_, false, true),
            grad_w_);
        NN_TRY_CHECK(grad_w_acc);
        nn_dbg_scan("lin.grad_w(post-accum)", engine, grad_w_);

        // grad_b += Σ grad_output（行归约，默认 f32）
        // 注：归约与累加**无法并入同一表达式**——归约向量输出契约要求输出链
        // 只经归约/广播视图访问输入，而此处必须同时引用外部累加张量 grad_b_
        // （Linear 视图），两者的语义冲突（见 eval_expr_reduce 的前置校验）。
        // 故保留"归约 + 累加"两步：累加步用 dsl::compute_into（原地 dst += src，
        // f16 src 由叶子精度签名处理），归约步用 dsl::compute_reduce（输出
        // (out,1) 归约向量同形）。
        auto gb = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)),
            grad_output.rows(), grad_output.cols(), p_.compute);
        NN_TRY_CHECK(gb);
        nn_dbg_scan("lin.row_sum(grad_out)", engine, *gb);
        auto r2 = dsl::compute_into(engine,
            dsl::leaf(grad_b_) + dsl::leaf(*gb), grad_b_);
        NN_TRY_CHECK(r2);
        nn_dbg_scan("lin.grad_b(post-accum)", engine, grad_b_);

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
    [[nodiscard]] const char* layer_name() const noexcept override { return "ReLU"; }

    ReLU() = default;

    void clear_cache() override { input_cache_ = Tensor{}; }

    // ── forward: out = max(x, 0) ──────────────────────────────────────────
    // ReLU 算法由 Layer 表达为 Max 原语 + 标量 0
    // Engine/Shader 只提供 Max 原语，不知道 "ReLU" 是什么
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (!checkpoint_mode_)
            input_cache_ = input;
        return dsl::compute(engine,
            dsl::max(dsl::leaf(input), Scalar{0}),
            input.rows(), input.cols(), p_.compute);
    }

    // ── backward: grad_x = (x > 0) ? grad_out : 0 ────────────────────────
    // ReLU 反向算法由 Layer 表达为 Select + Gt 原语
    // Engine/Shader 只提供 Select/Gt 原语，不知道 "ReLU backward" 是什么
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            NN_FAIL("relu backward: shape mismatch");
        return dsl::compute(engine,
            dsl::select(dsl::leaf(input_cache_) > Scalar{0},
                        dsl::leaf(grad_output), Scalar{0}),
            grad_output.rows(), grad_output.cols(), p_.compute);
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
    [[nodiscard]] const char* layer_name() const noexcept override { return "GeLU"; }

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
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (!checkpoint_mode_)
            input_cache_ = input;
        const Scalar beta = BETA;
        // out = x * sigmoid(βx) = x / (1 + exp(-β·x))
        return dsl::compute(engine,
            dsl::leaf(input) / (Scalar{1} + dsl::exp(-(dsl::leaf(input) * beta))),
            input.rows(), input.cols(), p_.compute);
    }

    // ── backward: grad_x = grad_out * factor ─────────────────────────────
    // factor = s * (1 + βx * (1 - s)),  s = sigmoid(βx) = 1/(1+exp(-βx))
    // 单表达式 DSL 融合：sigmoid 用 input_cache_ 重算（不缓存，省显存），
    // βx 子表达式由 DSL 的 CSE 复用，消除中间 Tensor。
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (input_cache_.rows() != grad_output.rows() ||
            input_cache_.cols() != grad_output.cols())
            NN_FAIL("gelu backward: shape mismatch");
        const Scalar beta = BETA;
        // s = 1/(1+exp(-βx))；factor = s * (1 + βx*(1-s))；out = grad_out * factor
        auto bx = dsl::leaf(input_cache_) * beta;                    // βx
        auto s  = Scalar{1} / (Scalar{1} + dsl::exp(-bx));          // sigmoid(βx)
        auto factor = s * (Scalar{1} + bx * (Scalar{1} - s));       // 1+βx(1-s)
        return dsl::compute(engine,
            dsl::leaf(grad_output) * factor,
            grad_output.rows(), grad_output.cols(), p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SwiGLU — Swish-Gated Linear Unit（LLaMA/Mistral 风格 FFN 激活）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  输入 (2*d_ff, batch)，前 d_ff 行为 gate，后 d_ff 行为 up
//             gate/up 经 row_access 行视图读取（零拷贝，不物化半张量）
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
//   split/merge：gate/up/s 经 RowAccess 行视图定位，两半由 select 一次写出
// ══════════════════════════════════════════════════════════════════════════
class SwiGLU final : public Layer
{
private:
    std::size_t d_ff_ = 0;
    Tensor input_cache_;  // 前向输入 (2*d_ff, batch)：backward 据此重算 gate/up/s（全融合，不缓存中间张量）

public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "SwiGLU"; }

    SwiGLU() = default;
    explicit SwiGLU(std::size_t d_ff) : d_ff_(d_ff) {}

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

    // ── forward: out = SiLU(gate) ⊙ up = gate·σ(gate)·up ────────────────────
    // 单表达式 DSL 融合：gate/up 用 RowAccess 行视图读取同一 (2*d_ff, batch)
    // 输入（gate = row_access(in, 0, d_ff)，up = row_access(in, d_ff, d_ff)），
    // 行视图零拷贝、不物化半张量（无 D2D 拷贝）。输出 (d_ff, batch)。
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        if (!checkpoint_mode_)
            input_cache_ = input;
        const std::size_t cols = input.cols();
        // gate[r] = in[r % d_ff]（r<d_ff 即 in[r]）；up[r] = in[d_ff + r % d_ff]
        const auto gv = dsl::row_access(input, 0u, dff);
        const auto uv = dsl::row_access(input, dff, dff);
        // out = gate · σ(gate) · up；σ(g) = 1/(1+exp(-g))
        return dsl::compute(engine,
            gv * (Scalar{1} / (Scalar{1} + dsl::exp(-gv))) * uv,
            d_ff_, cols, p_.compute);
    }

    // ── backward: 单表达式融合，直接写出整张 grad_input (2*d_ff, batch) ──────
    //
    // 数学：
    //   s(node)     = σ(gate[node])
    //   grad_gate   = grad_out ⊙ up ⊙ s ⊙ (1 + gate ⊙ (1 − s))
    //   grad_up     = grad_out ⊙ gate ⊙ s
    //
    // 输出 grad_input (2*d_ff, batch) 分两半写回：
    //   r ∈ [0, d_ff)       → grad_gate[r]
    //   r ∈ [d_ff, 2*d_ff)  → grad_up[r − d_ff]
    // 用 select(Row() < d_ff, grad_gate_expr, grad_up_expr) 分半；gate/up/s/go
    // 均经 RowAccess(offset=0|d_ff, mod=d_ff) 行视图按 r % d_ff 定位到对应半，
    // 两半表达式对"错误"半只会算出越界内但被 Select 丢弃的值，正确性无虞。
    // （共享的 gate/s 子表达式在树型 DSL 中会重复折叠，故 EXPR_MAX_REGS/INPUTS
    //   已相应放宽，换取零中间张量、零拷贝。）
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        const std::size_t rows = 2 * d_ff_;
        const std::size_t cols = grad_output.cols();

        if (input_cache_.rows() != rows || input_cache_.cols() != cols)
            NN_FAIL("swiglu backward: input_cache shape mismatch");
        if (grad_output.rows() != d_ff_ || grad_output.cols() != cols)
            NN_FAIL("swiglu backward: grad_output shape mismatch");
        const auto go   = dsl::row_access(grad_output, 0u, dff);   // go[r % d_ff]
        const auto gate = dsl::row_access(input_cache_, 0u, dff);   // in[r % d_ff]
        const auto up   = dsl::row_access(input_cache_, dff, dff);  // in[d_ff + r % d_ff]
        const auto s    = Scalar{1} / (Scalar{1} + dsl::exp(-gate)); // σ(gate)

        // factor = s·(1 + gate·(1−s))；grad_gate = go·up·factor；grad_up = go·gate·s
        const auto factor = s * (Scalar{1} + gate * (Scalar{1} - s));
        const auto gg = go * up * factor;
        const auto gu = go * gate * s;
        return dsl::compute(engine,
            dsl::select(dsl::row() < dsl::rparam(static_cast<nn::Scalar>(d_ff_)), gg, gu),
            rows, cols, p_.compute);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ReLU2GLU — 门控平方 ReLU（BitNet b1.58 2B4T 的 FFN 激活）
//
// 算法（只在此处，不在 Engine/Shader）：
//   forward:  out = relu(gate)² ⊙ up            （relu(g) = max(g, 0)）
//   backward: grad_gate = grad_out ⊙ up ⊙ 2·relu(gate)
//             grad_up   = grad_out ⊙ relu(gate)²
//
// 与 SwiGLU **同接线**（输入 (2·d_ff, batch)，gate = 前 d_ff 行、up = 后 d_ff 行，
// RowAccess 行视图零拷贝、单表达式全融合、输出 (d_ff, batch)），只把逐元素函数
// σ(g) 换成 relu(g)。两者不可互相替代（2B4T 用 relu2，LLaMA 用 silu）。
// ══════════════════════════════════════════════════════════════════════════
class ReLU2GLU final : public Layer
{
private:
    std::size_t d_ff_ = 0;
    Tensor input_cache_;  // 前向输入 (2*d_ff, batch)：backward 据此重算 gate

public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "ReLU2GLU"; }

    ReLU2GLU() = default;
    explicit ReLU2GLU(std::size_t d_ff) : d_ff_(d_ff) {}

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

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        if (!checkpoint_mode_)
            input_cache_ = input;
        const std::size_t cols = input.cols();
        const auto gv = dsl::row_access(input, 0u, dff);      // gate = in[r]
        const auto uv = dsl::row_access(input, dff, dff);     // up   = in[d_ff + r]
        const auto r  = dsl::relu(gv);                        // max(g, 0)
        return dsl::compute(engine, r * r * uv, d_ff_, cols, p_.compute);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const std::uint32_t dff = static_cast<std::uint32_t>(d_ff_);
        const std::size_t rows = 2 * d_ff_;
        const std::size_t cols = grad_output.cols();

        if (input_cache_.rows() != rows || input_cache_.cols() != cols)
            NN_FAIL("relu2glu backward: input_cache shape mismatch");
        if (grad_output.rows() != d_ff_ || grad_output.cols() != cols)
            NN_FAIL("relu2glu backward: grad_output shape mismatch");

        const auto go   = dsl::row_access(grad_output, 0u, dff);
        const auto gate = dsl::row_access(input_cache_, 0u, dff);
        const auto up   = dsl::row_access(input_cache_, dff, dff);
        const auto r    = dsl::relu(gate);
        // d/dg relu²(g) = 2·relu(g)（g=0 处不可导，取次梯度 0）
        const auto gg = go * up * (Scalar{2} * r);
        const auto gu = go * r * r;
        return dsl::compute(engine,
            dsl::select(dsl::row() < dsl::rparam(static_cast<nn::Scalar>(d_ff_)), gg, gu),
            rows, cols, p_.compute);
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
    [[nodiscard]] const char* layer_name() const noexcept override { return "LayerNorm"; }

    explicit LayerNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // gamma 初始化为 1, beta 初始化为 0（M2 声明式：引擎填数）
        gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::constant(1));
        beta_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::zero());
        if (!gamma_.valid() || !beta_.valid())
            NN_FAIL("LayerNorm: 参数初始化失败");
        grad_gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::zero());
        grad_beta_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::zero());
        if (!grad_gamma_.valid() || !grad_beta_.valid())
            NN_FAIL("LayerNorm: 梯度缓冲初始化失败");
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
    // 单表达式融合（算法公式不变，diff_sq (F,B) 由归约 kernel 内部消解）：
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
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != normalized_shape_)
            NN_FAIL("layernorm forward: input shape mismatch");
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. mean_raw = col_reduce_sum(x) → (1,B)
        auto mean_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input)), F, B, p_.stable);
        NN_TRY_CHECK(mean_raw);

        // 2. mean = mean_raw*(1/F) → (1,B)
        // （1/F 是形状相关标量，由 RParam 承载：**值不进 expr_spec_key**，
        //   融合表达式保持 F 无关 → 不同归一化维度共享同一 AOT 融合 shader）
        auto mean = dsl::compute(engine,
            dsl::leaf(*mean_raw) * dsl::rparam(inv_features),
            mean_raw->rows(), mean_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean);

        // 3. diff = x - mean (col 广播) → (F,B)
        auto diff = dsl::compute(engine,
            dsl::leaf(input) - dsl::col_broadcast(*mean), F, B, p_.stable);
        NN_TRY_CHECK(diff);

        // 4. var_raw = col_reduce_sum(diff²) → (1,B)
        auto var_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(*diff) * dsl::leaf(*diff)), F, B, p_.stable);
        NN_TRY_CHECK(var_raw);

        // 5. std_inv = rsqrt(var_raw*(1/F) + ε) → (1,B)
        // （"乘 1/F → 加 ε → rsqrt"三步塌成单表达式、单次遍历；1/F、ε 由
        //   RParam 承载，表达式结构仍与 F / ε 取值无关）
        auto std_inv = dsl::compute(engine,
            dsl::rsqrt(dsl::leaf(*var_raw) * dsl::rparam(inv_features)
                       + dsl::rparam(epsilon_)),
            var_raw->rows(), var_raw->cols(), p_.stable);
        NN_TRY_CHECK(std_inv);
        Tensor std_inv_t = std::move(*std_inv);
        if (!checkpoint_mode_)
            std_cache_ = std_inv_t;

        // 6. normalized = diff * std_inv (col 广播) → (F,B)
        auto normalized = dsl::compute(engine,
            dsl::leaf(*diff) * dsl::col_broadcast(std_inv_t), F, B, p_.stable);
        NN_TRY_CHECK(normalized);
        Tensor normalized_t = std::move(*normalized);
        if (!checkpoint_mode_)
            normalized_cache_ = normalized_t;

        // 7. out = normalized*gamma + beta (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
            + dsl::row_broadcast(beta_),
            F, B, p_.stable);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // grad_x = (gy - mean_g - normalized*mean_gn) * std_inv
    //   gy = grad_out * gamma
    //   mean_g  = col_reduce_sum(gy) * invF
    //   mean_gn = col_reduce_sum(gy * normalized) * invF
    // grad_gamma += row_sum(gy ⊙ normalized)
    // grad_beta  += row_sum(grad_out)
    // 单表达式融合（融合表达式 F 无关，1/F 在 (1,B) 上用原语施加）：
    //   mean_g/mean_gn 为列归约向量输出；(F,B) 全尺寸中间量由融合 kernel 消解。
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. mean_g_raw = col_reduce_sum(gy) → (1,B)
        auto mg_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)),
            F, B, p_.stable);
        NN_TRY_CHECK(mg_raw);
        auto mean_g = dsl::compute(engine,
            dsl::leaf(*mg_raw) * dsl::rparam(inv_features),
            mg_raw->rows(), mg_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean_g);

        // 2. mean_gn_raw = col_reduce_sum(gy ⊙ normalized) → (1,B)
        auto mgn_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(mgn_raw);
        auto mean_gn = dsl::compute(engine,
            dsl::leaf(*mgn_raw) * dsl::rparam(inv_features),
            mgn_raw->rows(), mgn_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean_gn);

        // 3. grad_x = (gy - mean_g - normalized*mean_gn) * std_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*mean_g)
             - dsl::leaf(normalized_cache_) * dsl::col_broadcast(*mean_gn))
            * dsl::col_broadcast(std_cache_),
            F, B, p_.stable);
        NN_TRY_CHECK(grad_x);

        // 4. grad_gamma += row_reduce_sum(gy ⊙ normalized) → (F,1)
        //    （∂L/∂γ_f = Σ_b gy·n：γ 是 out 的线性因子，导数不含 γ）
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output)
                * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(gg);
        auto grad_gamma_acc = dsl::compute_into(engine,
            dsl::leaf(grad_gamma_) + dsl::leaf(*gg), grad_gamma_);
        NN_TRY_CHECK(grad_gamma_acc);

        // 5. grad_beta += row_reduce_sum(grad_out) → (F,1)
        auto gb = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)), F, B, p_.stable);
        NN_TRY_CHECK(gb);
        auto grad_beta_acc = dsl::compute_into(engine,
            dsl::leaf(grad_beta_) + dsl::leaf(*gb), grad_beta_);
        NN_TRY_CHECK(grad_beta_acc);

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
    [[nodiscard]] const char* layer_name() const noexcept override { return "RMSNorm"; }

    explicit RMSNorm(std::size_t normalized_shape, Scalar epsilon = EPSILON)
        : normalized_shape_(normalized_shape), epsilon_(epsilon) {}

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // gamma 初始化为 1（无 beta）——M2 声明式：引擎填数
        gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::constant(1));
        if (!gamma_.valid())
            NN_FAIL("RMSNorm: 参数初始化失败");
        grad_gamma_ = engine.create_tensor(normalized_shape_, 1, p_.param, InitSpec::zero());
        if (!grad_gamma_.valid())
            NN_FAIL("RMSNorm: 梯度缓冲初始化失败");
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
    // 单表达式融合（算法公式不变，中间 x_sq (F,B) 由归约 kernel 内部消解）：
    //   融合表达式保持 F 无关结构（不含 1/F、ε 常量，避免闭合世界 key 随
    //   归一化维度漂移）；形状相关标量在 (1,B) 小向量上用引擎原语施加：
    //   1. s_raw  = col_reduce_sum(x*x)                    → (1,B) 归约向量输出
    //   2. rms_inv= rsqrt(s_raw*(1/F) + ε)                 → (1,B) 原语
    //   3. normed = x * rms_inv (col 广播)                 → (F,B) 融合逐元素
    //   4. out    = normed * gamma (row 广播)              → (F,B) 融合逐元素
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != normalized_shape_)
            NN_FAIL("rmsnorm forward: input shape mismatch");
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = input.cols();

        // 1. s_raw = col_reduce_sum(x*x) → (1,B)
        auto s_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(input) * dsl::leaf(input)), F, B, p_.stable);
        NN_TRY_CHECK(s_raw);

        // 2. rms_inv = rsqrt(s_raw*(1/F) + ε) → (1,B)
        // （"乘 1/F → 加 ε → rsqrt"三步塌成单表达式、单次遍历；1/F、ε 由
        //   RParam 承载：值不进 expr_spec_key → 表达式结构保持 F/ε 无关）
        auto rms_inv = dsl::compute(engine,
            dsl::rsqrt(dsl::leaf(*s_raw) * dsl::rparam(inv_features)
                       + dsl::rparam(epsilon_)),
            s_raw->rows(), s_raw->cols(), p_.stable);
        NN_TRY_CHECK(rms_inv);
        Tensor rms_inv_t = std::move(*rms_inv);
        if (!checkpoint_mode_)
            rms_inv_cache_ = rms_inv_t;

        // 3. normed = x * rms_inv (col 广播) → (F,B)
        auto normed = dsl::compute(engine,
            dsl::leaf(input) * dsl::col_broadcast(rms_inv_t), F, B, p_.stable);
        NN_TRY_CHECK(normed);
        Tensor normed_t = std::move(*normed);
        if (!checkpoint_mode_)
            normed_cache_ = normed_t;

        // 4. out = normed * gamma (row 广播) → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normed_t) * dsl::row_broadcast(gamma_), F, B, p_.stable);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // 单表达式融合（融合表达式 F 无关，1/F 在 (1,B) 小向量上用原语施加）：
    //   gy       = grad * gamma
    //   m_raw    = col_reduce_sum(gy ⊙ normed)             → (1,B) 归约向量输出
    //   m        = m_raw * (1/F)
    //   grad_x   = (gy - m*normed) * rms_inv               → (F,B) 融合逐元素
    //   grad_gamma += row_reduce_sum(gy ⊙ normed)          → (F,1) 归约向量输出
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const Scalar inv_features = Scalar{1} / static_cast<Scalar>(normalized_shape_);
        const std::size_t F = normalized_shape_;
        const std::size_t B = grad_output.cols();

        // 1. m_raw = col_reduce_sum(gy ⊙ normed) → (1,B)
        auto m_raw = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normed_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(m_raw);
        auto m = dsl::compute(engine,
            dsl::leaf(*m_raw) * dsl::rparam(inv_features),
            m_raw->rows(), m_raw->cols(), p_.stable);
        NN_TRY_CHECK(m);

        // 2. grad_x = (gy - m*normed) * rms_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::col_broadcast(*m) * dsl::leaf(normed_cache_))
            * dsl::col_broadcast(rms_inv_cache_),
            F, B, p_.stable);
        NN_TRY_CHECK(grad_x);

        // 3. grad_gamma += row_reduce_sum(gy ⊙ normed) → (F,1)
        //    （∂L/∂γ_f = Σ_b gy·n：γ 是 out 的线性因子，导数不含 γ）
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output)
                * dsl::leaf(normed_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(gg);
        auto grad_gamma_acc = dsl::compute_into(engine,
            dsl::leaf(grad_gamma_) + dsl::leaf(*gg), grad_gamma_);
        NN_TRY_CHECK(grad_gamma_acc);

        return grad_x;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// BatchNorm — 批归一化（沿 batch 维；NormType::BatchNorm，MLP/CNN/ViT 三路径接入）
//
// 训练/推理双态（`set_training` 切换；`Model::set_training` 逐层转发）：
//   训练态：用当前 batch 的统计量归一化，并以 EMA 更新 running 统计；
//   推理态：用 running 统计量归一化（batch=1 也安全）。
// running_mean/running_var 是**非可学习状态**：经 `extra_state()` 序列化、
// 不进 `parameters()`（优化器不碰它们）。
//
// 口径说明（与 PyTorch 的差异）：running_var 的 EMA 用**有偏**方差（1/B）
// ——归一化与统计更新同口径，B=1 无除零，结果确定（铁律 #8）。
// momentum_ = 新 batch 的权重（PyTorch 同口径，默认 0.1）。
//
// 算法（只在此处，不在 Engine/Shader；归约方向 = 沿 batch，与 LayerNorm
// 的沿特征方向互为翻转，其余公式同形）：
//   训练态 forward:
//     mean      = (1/B) Σ_b x[f][b]        (row_reduce_sum + 缩放)
//     diff      = x - mean                  (row 广播 Sub)
//     var       = (1/B) Σ_b diff²           (row_reduce_sum + 缩放)
//     std_inv   = 1/√(var + ε)              (rsqrt)
//     normalized= diff * std_inv            (row 广播 Mul)
//     out       = γ⊙normalized + β          (row 广播 Mul/Add)
//     running_mean ← (1-m)·running_mean + m·mean        (EMA)
//     running_var  ← (1-m)·running_var  + m·var         (EMA)
//   推理态 forward:
//     inv_std   = 1/√(running_var + ε)      (rsqrt)
//     normalized= (x - running_mean)·inv_std
//     out       = γ⊙normalized + β
//   backward（训练态；均值/归约全部沿 batch 维）:
//     gy         = grad_out * γ
//     mean_g     = (1/B) Σ_b gy
//     mean_gn    = (1/B) Σ_b gy⊙normalized
//     grad_x     = (gy - mean_g - normalized·mean_gn) * std_inv
//     grad_gamma += Σ_b grad_out⊙normalized   (row_reduce_sum)
//     grad_beta  += Σ_b grad_out               (row_reduce_sum)
//   backward（推理态；前向对 x 逐元素、跨 b 无耦合，故无均值修正项）:
//     grad_x     = grad_out * γ * inv_std     （γ/β 归约项与训练态同形）
//
// 接入范围：MNIST MLP（`mnist_train --norm batchnorm` / gui.py）、CNN（`--norm-place
//     conv/head/both`，层是 Model 直接子层）、ViT（pre-norm 槽位与 `--norm-place final`；
//     `TransformerEncoder{,Layer}` 已逐级转发 set_training）。
// GPT/RAPT 复合层不转发 set_training、text CLI 也不提供该选项；
// 梯度检查点场景由 forward_recompute 抑制 running 统计的重复 EMA。
// ══════════════════════════════════════════════════════════════════════════
class BatchNorm final : public Layer
{
private:
    std::size_t features_;
    Scalar epsilon_;
    Scalar momentum_;              // EMA 的新 batch 权重（默认 0.1）
    bool training_ = true;         // 训练/推理双态（set_training 切换）
    bool stat_update_suppress_ = false;  // forward_recompute 重算时抑制 EMA

    // 可学习参数
    Tensor gamma_;      // (features, 1)
    Tensor beta_;       // (features, 1)
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // 非可学习状态（extra_state → 序列化；初值 mean=0 / var=1）
    Tensor running_mean_;   // (features, 1)
    Tensor running_var_;    // (features, 1)

    // backward 缓存
    Tensor normalized_cache_;  // (features, batch)
    Tensor std_cache_;         // (features, 1) — 训练态 std_inv / 推理态 inv_std

    static constexpr Scalar EPSILON = 1e-5;
    static constexpr Scalar MOMENTUM = 0.1f;

public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "BatchNorm"; }

    explicit BatchNorm(std::size_t features, Scalar epsilon = EPSILON,
                       Scalar momentum = MOMENTUM)
        : features_(features), epsilon_(epsilon), momentum_(momentum) {}

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // γ=1、β=0（同 LayerNorm）；running 统计 mean=0、var=1（初值下
        // 推理态 = 恒等缩放，可直接训练早期评估）——M2 声明式：引擎填数。
        gamma_ = engine.create_tensor(features_, 1, p_.param, InitSpec::constant(1));
        beta_ = engine.create_tensor(features_, 1, p_.param, InitSpec::zero());
        if (!gamma_.valid() || !beta_.valid())
            NN_FAIL("BatchNorm: 参数初始化失败");
        grad_gamma_ = engine.create_tensor(features_, 1, p_.param, InitSpec::zero());
        grad_beta_ = engine.create_tensor(features_, 1, p_.param, InitSpec::zero());
        if (!grad_gamma_.valid() || !grad_beta_.valid())
            NN_FAIL("BatchNorm: 梯度缓冲初始化失败");
        running_mean_ = engine.create_tensor(features_, 1, p_.stable, InitSpec::zero());
        running_var_ = engine.create_tensor(features_, 1, p_.stable,
                                             InitSpec::constant(1));
        if (!running_mean_.valid() || !running_var_.valid())
            NN_FAIL("BatchNorm: running 统计初始化失败");
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

    // 非可学习状态：running 统计（不进 parameters → 优化器不更新）
    [[nodiscard]] std::vector<TensorRef> extra_state() override
    {
        std::vector<TensorRef> r;
        if (running_mean_.valid()) r.emplace_back(running_mean_);
        if (running_var_.valid())  r.emplace_back(running_var_);
        return r;
    }

    // 训练/推理双态切换（Model::set_training → 本层）
    void set_training(bool training) override { training_ = training; }

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
    // 训练态（步骤与 LayerNorm 一一对应，归约/广播方向沿 batch）：
    //   1. mean_raw = row_reduce_sum(x)                     → (F,1)
    //   2. mean     = mean_raw*(1/B)                        → (F,1)
    //   3. diff     = x - mean (row 广播)                   → (F,B)
    //   4. var_raw  = row_reduce_sum(diff²)                 → (F,1)
    //   5. std_inv  = rsqrt(var_raw*(1/B) + ε)              → (F,1)
    //   6. normalized = diff * std_inv (row 广播)           → (F,B)
    //   7. running 统计 EMA（compute_into 原地；recompute 抑制）
    //   8. out      = normalized*γ + β (row 广播)           → (F,B)
    //（1/B、ε、momentum 都是 RParam：值不进 expr_spec_key）
    // 推理态：inv_std + 归一化 + 出参三步，无归约链（不同结构，scan 必须
    // 单独覆盖——见 tools/scan_exprs.cpp 模型 pass 的推理态 run）。
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != features_)
            NN_FAIL("batchnorm forward: input shape mismatch");
        const std::size_t F = features_;
        const std::size_t B = input.cols();

        // ── 推理态：用 running 统计（batch=1 安全）──
        if (!training_)
        {
            // 1. inv_std = rsqrt(running_var + ε) → (F,1)
            auto inv_std = dsl::compute(engine,
                dsl::rsqrt(dsl::leaf(running_var_) + dsl::rparam(epsilon_)),
                F, 1, p_.stable);
            NN_TRY_CHECK(inv_std);
            Tensor inv_std_t = std::move(*inv_std);
            if (!checkpoint_mode_)
                std_cache_ = inv_std_t;

            // 2. normalized = (x - running_mean) * inv_std → (F,B)
            auto normalized = dsl::compute(engine,
                (dsl::leaf(input) - dsl::row_broadcast(running_mean_))
                * dsl::row_broadcast(inv_std_t), F, B, p_.stable);
            NN_TRY_CHECK(normalized);
            Tensor normalized_t = std::move(*normalized);
            if (!checkpoint_mode_)
                normalized_cache_ = normalized_t;

            // 3. out = normalized*γ + β（与训练态同构，共享融合结构）
            return dsl::compute(engine,
                dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
                + dsl::row_broadcast(beta_),
                F, B, p_.stable);
        }

        // ── 训练态 ──
        const Scalar inv_batch = Scalar{1} / static_cast<Scalar>(B);

        // 1. mean_raw = row_reduce_sum(x) → (F,1)
        auto mean_raw = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(input)), F, B, p_.stable);
        NN_TRY_CHECK(mean_raw);

        // 2. mean = mean_raw*(1/B) → (F,1)
        auto mean = dsl::compute(engine,
            dsl::leaf(*mean_raw) * dsl::rparam(inv_batch),
            mean_raw->rows(), mean_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean);

        // 3. diff = x - mean (row 广播) → (F,B)
        auto diff = dsl::compute(engine,
            dsl::leaf(input) - dsl::row_broadcast(*mean), F, B, p_.stable);
        NN_TRY_CHECK(diff);

        // 4. var_raw = row_reduce_sum(diff²) → (F,1)
        auto var_raw = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(*diff) * dsl::leaf(*diff)),
            F, B, p_.stable);
        NN_TRY_CHECK(var_raw);

        // 5. std_inv = rsqrt(var_raw*(1/B) + ε) → (F,1)
        auto std_inv = dsl::compute(engine,
            dsl::rsqrt(dsl::leaf(*var_raw) * dsl::rparam(inv_batch)
                       + dsl::rparam(epsilon_)),
            var_raw->rows(), var_raw->cols(), p_.stable);
        NN_TRY_CHECK(std_inv);
        Tensor std_inv_t = std::move(*std_inv);
        if (!checkpoint_mode_)
            std_cache_ = std_inv_t;

        // 6. normalized = diff * std_inv (row 广播) → (F,B)
        auto normalized = dsl::compute(engine,
            dsl::leaf(*diff) * dsl::row_broadcast(std_inv_t), F, B, p_.stable);
        NN_TRY_CHECK(normalized);
        Tensor normalized_t = std::move(*normalized);
        if (!checkpoint_mode_)
            normalized_cache_ = normalized_t;

        // 7. running 统计 EMA（原地 compute_into；推理态无此步，
        //    forward_recompute 重算时抑制——否则检查点场景会双倍更新）
        if (!stat_update_suppress_)
        {
            auto ema_m = dsl::compute_into(engine,
                dsl::leaf(running_mean_) * dsl::rparam(1 - momentum_)
                + dsl::leaf(*mean) * dsl::rparam(momentum_),
                running_mean_);
            NN_TRY_CHECK(ema_m);
            // 有偏 var（1/B 口径）直接以 var_raw*(1/B·m) 进 EMA，省一次分配
            auto ema_v = dsl::compute_into(engine,
                dsl::leaf(running_var_) * dsl::rparam(1 - momentum_)
                + dsl::leaf(*var_raw) * dsl::rparam(inv_batch * momentum_),
                running_var_);
            NN_TRY_CHECK(ema_v);
        }

        // 8. out = normalized*γ + β → (F,B)
        return dsl::compute(engine,
            dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
            + dsl::row_broadcast(beta_),
            F, B, p_.stable);
    }

    // ── backward ──────────────────────────────────────────────────────────
    // 训练态：与 LayerNorm backward 同形（归约方向翻转为 batch、1/F → 1/B）；
    // 推理态：前向对 x 逐元素（无跨 b 耦合）→ grad_x 无均值修正项。
    // γ/β 归约项两种模式同构（normalized_cache_ 语义一致）。
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (!normalized_cache_.valid() || !std_cache_.valid())
            NN_FAIL("batchnorm backward: 缺少 forward 缓存（先调 forward）");
        const std::size_t F = features_;
        const std::size_t B = grad_output.cols();
        const Scalar inv_batch = Scalar{1} / static_cast<Scalar>(B);

        // ── 推理态：grad_x = gy * γ * inv_std ──
        if (!training_)
        {
            auto grad_x = dsl::compute(engine,
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::row_broadcast(std_cache_),
                F, B, p_.stable);
            NN_TRY_CHECK(grad_x);
            NN_TRY(grad_gamma_acc, accumulate_gamma_(engine, grad_output));
            NN_TRY(grad_beta_acc, accumulate_beta_(engine, grad_output));
            return grad_x;
        }

        // ── 训练态 ──
        // 1. mean_g_raw = row_reduce_sum(gy) → (F,1)
        auto mg_raw = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)),
            F, B, p_.stable);
        NN_TRY_CHECK(mg_raw);
        auto mean_g = dsl::compute(engine,
            dsl::leaf(*mg_raw) * dsl::rparam(inv_batch),
            mg_raw->rows(), mg_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean_g);

        // 2. mean_gn_raw = row_reduce_sum(gy ⊙ normalized) → (F,1)
        auto mgn_raw = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
                * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(mgn_raw);
        auto mean_gn = dsl::compute(engine,
            dsl::leaf(*mgn_raw) * dsl::rparam(inv_batch),
            mgn_raw->rows(), mgn_raw->cols(), p_.stable);
        NN_TRY_CHECK(mean_gn);

        // 3. grad_x = (gy - mean_g - normalized*mean_gn) * std_inv → (F,B)
        auto grad_x = dsl::compute(engine,
            (dsl::leaf(grad_output) * dsl::row_broadcast(gamma_)
             - dsl::row_broadcast(*mean_g)
             - dsl::leaf(normalized_cache_) * dsl::row_broadcast(*mean_gn))
            * dsl::row_broadcast(std_cache_),
            F, B, p_.stable);
        NN_TRY_CHECK(grad_x);

        // 4. grad_gamma += row_reduce_sum(grad_out ⊙ normalized) → (F,1)
        // 5. grad_beta  += row_reduce_sum(grad_out)             → (F,1)
        NN_TRY(grad_gamma_acc, accumulate_gamma_(engine, grad_output));
        NN_TRY(grad_beta_acc, accumulate_beta_(engine, grad_output));

        return grad_x;
    }

    // 梯度检查点重算：重建缓存，但**不**重复更新 running 统计
    //（原 forward 已在真实前向里更新过一次）。
    [[nodiscard]] Result<Tensor> forward_recompute(const Tensor& saved_input) override
    {
        const bool prev_ckpt = checkpoint_mode_;
        const bool prev_sup  = stat_update_suppress_;
        checkpoint_mode_ = false;
        stat_update_suppress_ = true;
        auto r = forward(saved_input);
        checkpoint_mode_ = prev_ckpt;
        stat_update_suppress_ = prev_sup;
        return r;
    }

private:
    // grad_gamma += row_reduce_sum(grad_out ⊙ normalized)（训练/推理态共用）
    [[nodiscard]] Result<void> accumulate_gamma_(
        ComputeEngine& engine, const Tensor& grad_output)
    {
        const std::size_t F = features_;
        const std::size_t B = grad_output.cols();
        auto gg = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(
                dsl::leaf(grad_output) * dsl::leaf(normalized_cache_)),
            F, B, p_.stable);
        NN_TRY_CHECK(gg);
        return dsl::compute_into(engine,
            dsl::leaf(grad_gamma_) + dsl::leaf(*gg), grad_gamma_);
    }

    // grad_beta += row_reduce_sum(grad_out)（训练/推理态共用）
    [[nodiscard]] Result<void> accumulate_beta_(
        ComputeEngine& engine, const Tensor& grad_output)
    {
        const std::size_t F = features_;
        const std::size_t B = grad_output.cols();
        auto gb = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)), F, B, p_.stable);
        NN_TRY_CHECK(gb);
        return dsl::compute_into(engine,
            dsl::leaf(grad_beta_) + dsl::leaf(*gb), grad_beta_);
    }
};

// ── 归一化层工厂：按 NormType 创建 LayerNorm / RMSNorm / BatchNorm ────────
[[nodiscard]] inline std::unique_ptr<Layer> make_norm_layer(
    std::size_t d_model, NormType norm_type)
{
    if (norm_type == NormType::RMSNorm)
        return std::make_unique<RMSNorm>(d_model);
    if (norm_type == NormType::BatchNorm)
        return std::make_unique<BatchNorm>(d_model);
    return std::make_unique<LayerNorm>(d_model);
}

// ── 线性层工厂：按 WeightQuant 创建 Linear / BitLinear（P1.5，docs 21 §4.8.3）──
// 与 make_norm_layer 同一先例：复合层（Attention / FeedForward / GPT / RAPT）
// 用**运行期开关**选线性层类型，成员类型统一为 `std::unique_ptr<Layer>`。
// BitLinear 与 Linear 的参数形状/顺序完全一致（(out,in) 权重 + (out,1) 偏置），
// 因此序列化与优化器侧零改动。
[[nodiscard]] inline std::unique_ptr<Layer> make_linear_layer(
    std::size_t in_features, std::size_t out_features,
    WeightQuant quant = WeightQuant::None)
{
    if (quant == WeightQuant::T1_58)
        return std::make_unique<BitLinear>(in_features, out_features);
    return std::make_unique<Linear>(in_features, out_features);
}
} // namespace nn
