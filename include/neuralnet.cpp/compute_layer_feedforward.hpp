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
    // 线性子层走工厂（P1.5 docs 21 §4.8.3）：weight_quant=T1_58 → BitLinear。
    // 成员类型 = std::unique_ptr<Layer>（与 norm1_/sub_norm_ 同一形态）。
    std::unique_ptr<Layer> fc1_;  // GeLU: (d_model → d_ff); SwiGLU: (d_model → 2*d_ff)
    std::unique_ptr<Layer> fc2_;  // (d_ff → d_model)
    // 三值模式（构造期定型；决定 fc1_/fc2_ 的**具体类型**与 profile 分流）
    WeightQuant weight_quant_ = WeightQuant::None;
    GeLU  gelu_;
    SwiGLU swiglu_;  // SwiGLU 激活（含 split/merge）
    ReLU2GLU relu2glu_;  // 门控平方 ReLU 激活（BitNet 2B4T；同 split/merge 接线）
    bool use_swiglu_ = false;
    bool use_relu2_ = false;

    // ── SubLN（BitNet b1.58 的"子层归一化"，docs/development/22 §3.1）──────
    // **子层内部**的额外归一化：挂在中间激活（宽 d_ff）与输出投影 fc2 之间。
    // 默认 nullptr = 关（既有路径逐位不变）。类型跟随模型的 norm_type
    // （2B4T = RMSNorm）——经 make_norm_layer 构造，与 GPTBlock 的 norm1_/norm2_
    // 同一工厂、同一类型来源。
    //
    // 位置：`h2 = act(fc1(x))` → **sub_norm_ → fc2**。不能挂到 block 级：
    // block 看不到这个 (d_ff, N) 的中间张量（这正是"子层内部"的含义）。
    std::unique_ptr<Layer> sub_norm_;

public:
    [[nodiscard]] const char* layer_name() const noexcept override { return "FeedForward"; }

    FeedForward(std::size_t d_model, std::size_t d_ff,
                ActivationType activation = ActivationType::GeLU,
                bool subln = false,
                NormType subln_norm_type = NormType::LayerNorm,
                WeightQuant weight_quant = WeightQuant::None)
        : fc1_(make_linear_layer(d_model,
               (activation == ActivationType::SwiGLU ||
                activation == ActivationType::ReLU2) ? 2 * d_ff : d_ff, weight_quant)),
          fc2_(make_linear_layer(d_ff, d_model, weight_quant)),
          weight_quant_(weight_quant),
          swiglu_(d_ff),
          relu2glu_(d_ff),
          use_swiglu_(activation == ActivationType::SwiGLU),
          use_relu2_(activation == ActivationType::ReLU2)
    {
        if (subln)
            sub_norm_ = make_norm_layer(d_ff, subln_norm_type);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, fc1_->init(engine));
        NN_TRY(r2, fc2_->init(engine));
        // M6 段 C：激活子层也是 Layer（engine 由 init 绑定），必须一并 init——
        // 否则 forward 里调 gelu_/swiglu_.forward() 会在 engine_ref() 处 fail-fast。
        { NN_TRY(r, gelu_.init(engine)); }
        { NN_TRY(r, swiglu_.init(engine)); }
        { NN_TRY(r, relu2glu_.init(engine)); }
        if (sub_norm_) { NN_TRY(r, sub_norm_->init(engine)); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        auto r = collect_refs(fc1_->parameters(), fc2_->parameters());
        if (sub_norm_)
        {
            auto s = sub_norm_->parameters();
            r.insert(r.end(), s.begin(), s.end());
        }
        return r;
    }

    std::vector<TensorRef> param_gradients() override
    {
        auto r = collect_refs(fc1_->param_gradients(), fc2_->param_gradients());
        if (sub_norm_)
        {
            auto s = sub_norm_->param_gradients();
            r.insert(r.end(), s.begin(), s.end());
        }
        return r;
    }

    // 梯度检查点：把模式传播给内部 fc1/fc2/gelu/swiglu
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        fc1_->set_checkpoint_mode(enabled);
        fc2_->set_checkpoint_mode(enabled);
        gelu_.set_checkpoint_mode(enabled);
        swiglu_.set_checkpoint_mode(enabled);
        relu2glu_.set_checkpoint_mode(enabled);
        if (sub_norm_) sub_norm_->set_checkpoint_mode(enabled);
    }

    // ── D7：精度配置下传（§9.2）──────────────────────────────────────────
    // FeedForward 是复合层（fc1/fc2 + GeLU/SwiGLU）：必须把 profile 下传给
    // 所有子层，否则子层 p_ 停在默认全 F32 —— FFN 是 d_ff=4·d_model 量级的
    // 最大激活生产者，f16 配置下静默失效。
    //
    // ── P1.5 三值分流（§4.8.3）───────────────────────────────────────────
    // `weight_quant_ = T1_58` 时：fc1_/fc2_ 是 BitLinear，拿 param=T1_58 的
    // profile；本层自身与激活层拿 param 归一化为 f32 的那一份（自身不持有
    // 三值权重）。非三值模式逐位不变（split 退化为恒等）。
    void set_precision_profile(const PrecisionProfile& profile) override
    {
        const TernaryProfileSplit sp = split_ternary_profile(
            layer_name(), profile, weight_quant_ == WeightQuant::T1_58);
        Layer::set_precision_profile(sp.self);
        fc1_->set_precision_profile(sp.linear);
        fc2_->set_precision_profile(sp.linear);
        gelu_.set_precision_profile(sp.self);
        swiglu_.set_precision_profile(sp.self);
        relu2glu_.set_precision_profile(sp.self);
        if (sub_norm_) sub_norm_->set_precision_profile(sp.self);
    }

    void clear_cache() override
    {
        fc1_->clear_cache();
        fc2_->clear_cache();
        gelu_.clear_cache();
        swiglu_.clear_cache();
        relu2glu_.clear_cache();
        if (sub_norm_) sub_norm_->clear_cache();
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = fc1_->activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto b = fc2_->activation_cache(); r.insert(r.end(), b.begin(), b.end());
        auto g = gelu_.activation_cache(); r.insert(r.end(), g.begin(), g.end());
        auto s = swiglu_.activation_cache(); r.insert(r.end(), s.begin(), s.end());
        auto r2 = relu2glu_.activation_cache(); r.insert(r.end(), r2.begin(), r2.end());
        if (sub_norm_)
        {
            auto sn = sub_norm_->activation_cache();
            r.insert(r.end(), sn.begin(), sn.end());
        }
        return r;
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        NN_TRY(h1, fc1_->forward(input));
        Tensor h2_out;
        if (use_swiglu_)
        {
            NN_TRY(h2, swiglu_.forward(*h1));
            h2_out = std::move(*h2);
        }
        else if (use_relu2_)
        {
            NN_TRY(h2, relu2glu_.forward(*h1));
            h2_out = std::move(*h2);
        }
        else
        {
            NN_TRY(h2, gelu_.forward(*h1));
            h2_out = std::move(*h2);
        }
        // SubLN：中间激活归一化后再进输出投影（关 = 直通，逐位不变）
        if (sub_norm_)
        {
            NN_TRY(sn, sub_norm_->forward(h2_out));
            h2_out = std::move(*sn);
        }
        return fc2_->forward(h2_out);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        NN_TRY(b2, fc2_->backward(grad_output));
        Tensor g = std::move(*b2);
        if (sub_norm_)
        {
            NN_TRY(gs, sub_norm_->backward(g));
            g = std::move(*gs);
        }
        if (use_swiglu_)
        {
            NN_TRY(bg, swiglu_.backward(g));
            return fc1_->backward(*bg);
        }
        if (use_relu2_)
        {
            NN_TRY(bg, relu2glu_.backward(g));
            return fc1_->backward(*bg);
        }
        NN_TRY(bg, gelu_.backward(g));
        return fc1_->backward(*bg);
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
