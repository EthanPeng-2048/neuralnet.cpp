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

} // namespace nn

