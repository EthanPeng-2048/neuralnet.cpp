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
#include "compute_layer_mlp.hpp"
#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn {

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

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        auto r1 = fc1_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = fc2_.init(engine); if (!r2) return std::unexpected(r2.error());
        // M6 段 C：激活子层也是 Layer（engine 由 init 绑定），必须一并 init——
        // 否则 forward 里调 gelu_/swiglu_.forward() 会在 engine_ref() 处 fail-fast。
        { auto r = gelu_.init(engine);   if (!r) return std::unexpected(r.error()); }
        { auto r = swiglu_.init(engine); if (!r) return std::unexpected(r.error()); }
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

    // ── D7：精度配置下传（§9.2）──────────────────────────────────────────
    // FeedForward 是复合层（fc1/fc2 + GeLU/SwiGLU）：必须把 profile 下传给
    // 所有子层，否则子层 p_ 停在默认全 F32 —— FFN 是 d_ff=4·d_model 量级的
    // 最大激活生产者，f16 配置下静默失效。
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        fc1_.set_precision_profile(profile);
        fc2_.set_precision_profile(profile);
        gelu_.set_precision_profile(profile);
        swiglu_.set_precision_profile(profile);
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
        const Tensor& input) override
    {
        auto h1 = fc1_.forward(input);
        if (!h1) return h1;
        if (use_swiglu_)
        {
            auto h2 = swiglu_.forward(*h1);
            if (!h2) return h2;
            return fc2_.forward(*h2);
        }
        auto h2 = gelu_.forward(*h1);
        if (!h2) return h2;
        return fc2_.forward(*h2);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        auto b2 = fc2_.backward(grad_output);
        if (!b2) return b2;
        if (use_swiglu_)
        {
            auto bg = swiglu_.backward(*b2);
            if (!bg) return bg;
            return fc1_.backward(*bg);
        }
        auto bg = gelu_.backward(*b2);
        if (!bg) return bg;
        return fc1_.backward(*bg);
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

} // namespace nn

