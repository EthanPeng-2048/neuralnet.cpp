#pragma once

// ── compute_layer_bitlinear.hpp — 三值（1.58-bit）线性层 BitLinear ──────────
//
// 设计：docs/development/21-quantized-weights.md §4.4（P1 形态）。
// 前置证据：research/ternary_ste/REPORT.md（同一套 DSL 组合的 STE 训练 PoC）、
//           research/ternary_scale/REPORT.md（尺度 = 逐行 absmean 的实测依据）。
//
// 名词（§4.3.2 语义，务必先读）：
//   · `param = T1_58` 表示"本层 forward 的**有效权重精度**是三值"，
//     **不是**"参数张量的存储布局是三值" —— 所以：
//       latent_w_ : f32 参数（优化器更新对象 + STE 梯度落点），**不落盘为三值**；
//       wq_       : f16 量化缓冲（{-1,0,+1} 在 f16 下精确可表示），每步重算；
//       tau_      : (out,1) f32 逐行尺度（absmean），同样是派生物。
//   · forward 用三值权重 + 尺度做**去量化**点积；backward 用 STE
//     （把量化器当恒等映射求导）把梯度送回 f32 latent。
//
// 算法（只在此处，不在 Engine/Shader）：
//   τ_o   = (1/K)·Σ_k |W[o,k]|                      （逐输出通道 absmean）
//   wq    = RoundClip(W/τ, -1, 1)                   （DSL 无 round → 阈值 0.5 的 select 等价）
//   Y     = (wq · X)·τ + b                          （τ 逐行广播，b 逐行广播）
//   dX    = wqᵀ · (dY∘τ)                            （等价于 (τ∘wq)ᵀ·dY）
//   dW    = (dY · Xᵀ)∘τ                             （STE：∂wq/∂W ≡ 1，含去量化尺度）
//   db    = Σ_batch dY
//
// 与 f16/f32 的**正交性**（§8）：三值是"换一种权重表示"（离散化 + 需要 γ），
// 不是"换更窄的容器"（舍入）。因此它只占 param 槽，且不进提升序
//（precision_rank 无定义）——见 precision.hpp。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstddef>
#include <utility>

#include "compute_engine.hpp"
#include "compute_layer_base.hpp"
#include "compute_tensor.hpp"
#include "expr_dsl.hpp"
#include "precision.hpp"

namespace nn
{

class BitLinear final : public Layer
{
private:
    std::size_t in_features_;
    std::size_t out_features_;

    // ── 可训练参数（f32 latent；优化器与 STE 的作用点）────────────────────
    Tensor latent_w_;    // (out_features, in_features)
    Tensor b_;           // (out_features, 1) —— 偏置不做三值量化（BitNet 同款）
    Tensor grad_w_;      // 与 latent_w_ 同形
    Tensor grad_b_;      // 与 b_ 同形

    // ── 量化派生物（每步重算；不落盘、不进序列化）────────────────────────
    Tensor wq_;          // (out_features, in_features) f16，值 ∈ {-1,0,+1}
    Tensor tau_;         // (out_features, 1) f32，逐行 absmean

    Tensor input_cache_; // forward 输入缓存（backward 用）

public:
    BitLinear(std::size_t in_features, std::size_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    [[nodiscard]] const char* layer_name() const noexcept override { return "BitLinear"; }

    // ── 能力声明（R1，§4.2）────────────────────────────────────────────
    // param **只**接受 T1_58（T1_58 层面"该层权重是三值"的语义锚）；
    // compute/stable/optimizer 仍是 {f16,f32}——激活与数值敏感链绝不能被三值化。
    [[nodiscard]] PrecisionSupport precision_support() const override
    {
        PrecisionSupport sup = PrecisionSupport::rounding_all_slots();
        sup.param = PrecisionSet::of({Precision::T1_58});
        return sup;
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // Xavier 均匀分布（与 Linear 同款）；latent/偏置**恒 f32**
        //（§4.3.2：param=T1_58 只是一句声明，绝不拿它去 create_tensor）。
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features_ + out_features_));
        latent_w_ = engine.create_tensor(out_features_, in_features_, Precision::F32,
                                         InitSpec::uniform(-limit, limit, kInitSeed));
        if (!latent_w_.valid()) NN_FAIL("BitLinear: latent 权重初始化失败");
        b_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        if (!b_.valid()) NN_FAIL("BitLinear: 偏置初始化失败");
        grad_w_ = engine.create_tensor(out_features_, in_features_, Precision::F32,
                                       InitSpec::zero());
        grad_b_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        if (!grad_w_.valid() || !grad_b_.valid())
            NN_FAIL("BitLinear: 梯度缓冲初始化失败");
        // 量化缓冲：wq_ 恒 f16（{-1,0,1} 精确可表示；P2 起才改打包存储）
        wq_ = engine.create_tensor(out_features_, in_features_, Precision::F16,
                                   InitSpec::zero());
        if (!wq_.valid()) NN_FAIL("BitLinear: 量化缓冲初始化失败");
        tau_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        if (!tau_.valid()) NN_FAIL("BitLinear: 尺度缓冲初始化失败");
        return {};
    }

    // ── 参数视图：latent（f32）+ 偏置（f32）──────────────────────────────
    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {latent_w_, b_};
    }
    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        return {grad_w_, grad_b_};
    }

    // ── 量化派生物只读视图（诊断 / 测试 / 序列化对拍用）────────────────────
    [[nodiscard]] const Tensor& latent_weights() const noexcept { return latent_w_; }
    [[nodiscard]] const Tensor& quantized_weights() const noexcept { return wq_; }
    [[nodiscard]] const Tensor& row_scales() const noexcept { return tau_; }

    void clear_cache() override
    {
        input_cache_ = Tensor{};
        tau_ = Tensor{};   // 派生物：重算即可（wq_ 属参数派生物，保留缓冲）
    }

    [[nodiscard]] std::vector<TensorRef> activation_cache() override
    {
        // backward 需要 input_cache_（X）与 tau_（尺度）→ 两者都参与 offload/重算
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        if (tau_.valid()) r.emplace_back(tau_);
        return r;
    }

    // ── forward：量化（τ + wq）→ 去量化点积 ─────────────────────────────
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != in_features_)
            NN_FAIL("bitlinear forward: input shape mismatch");
        if (!checkpoint_mode_)
            input_cache_ = input;
        if (auto q = quantize_weights_(); !q)
        {
            // τ / wq 重算失败：直接上抛（不静默沿用上一步的旧缓冲）
            return std::unexpected(q.error());
        }
        return dsl::compute(engine,
            dsl::matmul(wq_, input, false, false) * dsl::row_broadcast(tau_)
                + dsl::row_broadcast(b_),
            out_features_, input.cols(), p_.compute);
    }

    // ── backward：STE（把 wq 当常数；梯度回 f32 latent）──────────────────
    [[nodiscard]] Result<Tensor> backward(const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (grad_output.rows() != out_features_)
            NN_FAIL("bitlinear backward: grad_output shape mismatch");
        if (!tau_.valid() || !input_cache_.valid())
            NN_FAIL("bitlinear backward: 必须先 forward（τ/输入缓存未建立）");

        // dX = wqᵀ·(dY∘τ)：τ 是逐输出通道的，必须先落到 dY 的行上再左乘
        //（不能提到 matmul 外面——τ_o 在求和维内）。
        NN_TRY(dy_scaled, dsl::compute(engine,
            dsl::leaf(grad_output) * dsl::row_broadcast(tau_),
            out_features_, grad_output.cols(), p_.compute));
        auto grad_input = dsl::compute(engine,
            dsl::matmul(wq_, *dy_scaled, /*transA=*/true, false),
            in_features_, grad_output.cols(), p_.compute);
        NN_TRY_CHECK(grad_input);

        // dW += (dY·Xᵀ)∘τ（STE：∂wq/∂W ≡ 1，再乘去量化尺度 τ）
        // matmul 段 + 逐行缩放 + 原地累加融合为单次 dispatch（与 Linear 同构）。
        auto grad_w_acc = dsl::compute_into(engine,
            dsl::leaf(grad_w_)
                + dsl::matmul(grad_output, input_cache_, false, true)
                      * dsl::row_broadcast(tau_),
            grad_w_);
        NN_TRY_CHECK(grad_w_acc);

        // db += Σ_batch dY（归约向量 + 原地累加两步；与 Linear 同款理由：
        // 归约向量输出契约要求输出链只经归约/广播视图访问输入）
        NN_TRY(gb, dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)),
            grad_output.rows(), grad_output.cols(), p_.compute));
        auto r2 = dsl::compute_into(engine,
            dsl::leaf(grad_b_) + dsl::leaf(*gb), grad_b_);
        NN_TRY_CHECK(r2);

        return grad_input;
    }

private:
    // ── 量化：τ = 逐行 absmean；wq = RoundClip(W/τ, -1, 1) ───────────────
    // 全程在引擎内（铁律 #12：L2+ 不得把权重读回宿主）：
    //   1) τ_raw = row_reduce_sum(|W|)          —— 独立一趟归约（GLSL emitter
    //      不支持把归约视图内联进逐元素 kernel，故用 compute_reduce 单独出向量）；
    //   2) τ = τ_raw·(1/K)                      —— RParam 承载 1/K（值不进 key）；
    //   3) wq ← select(|W|>0.5τ 的正负两侧)      —— compute_into 原地写 f16 缓冲。
    [[nodiscard]] Result<void> quantize_weights_()
    {
        ComputeEngine& engine = engine_ref();
        const Scalar inv_in = Scalar{1} / static_cast<Scalar>(in_features_);

        NN_TRY(tau_raw, dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::abs(dsl::leaf(latent_w_))),
            out_features_, in_features_, Precision::F32));
        NN_TRY(tau, dsl::compute(engine,
            dsl::leaf(*tau_raw) * dsl::rparam(inv_in),
            out_features_, 1, Precision::F32));
        tau_ = std::move(*tau);

        // RoundClip(W/τ, -1, 1)：DSL 没有 round/floor，但在 [-1,1] 区间内
        // 与"阈值 0.5τ 的两侧 select"等价（BitNet 的定义形式）。
        auto lo = dsl::row_broadcast(tau_) * dsl::rparam(Scalar{-0.5});
        auto hi = dsl::row_broadcast(tau_) * dsl::rparam(Scalar{0.5});
        return dsl::compute_into(engine,
            dsl::select(dsl::leaf(latent_w_) > hi,
                        Scalar{1},
                        dsl::select(dsl::leaf(latent_w_) < lo, Scalar{-1}, Scalar{0})),
            wq_);
    }
};

} // namespace nn
