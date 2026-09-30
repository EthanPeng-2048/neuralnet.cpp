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
#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Softmax — 按行 softmax（用于注意力权重）
//
// 算法（只在此处，不在 Engine/Shader），两次调用各为单个融合表达式：
//   forward:  out[r][c] = exp(x[r][c] - row_max[r]) / row_sum[r]
//             其中 row_max[r] = max_c x[r][c]，row_sum[r] = Σ_c exp(x[r][c]-row_max[r])
//   backward: grad_x[r][c] = out[r][c] * (grad_out[r][c] - row_dot[r])
//             其中 row_dot[r] = Σ_c out[r][c] * grad_out[r][c]
// 中间量（shifted/exp/row_max/row_sum）全部由融合 kernel 消解，不落显存。
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
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        // 行 softmax（数值稳定）：out = exp(x - row_max) / Σ_c exp(x - row_max)
        // 单表达式融合：row_max/row_sum 为归约视图/归约指令，中间全尺寸
        // Tensor（shifted/exp_shift/row_max/row_sum 的物化）由融合 kernel 消解，
        // 仅 input 与 output 落显存。
        // 表达式文本只写在本 Layer；AOT 收集由 scan_exprs dry-run 本方法完成。
        auto out = dsl::compute(engine,
            dsl::exp(dsl::leaf(input) - dsl::row_reduce_max(input))
            / dsl::row_reduce_sum(
                dsl::exp(dsl::leaf(input) - dsl::row_reduce_max(input))),
            input.rows(), input.cols(), p_.stable);
        NN_TRY_CHECK(out);
        if (checkpoint_mode_)
            return out;
        // 单缓冲：把结果移入 output_cache_（唯一持有者），返回共享同一 buffer
        output_cache_ = std::move(*out);
        return output_cache_;
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // grad_x = out ⊙ (grad_output - row_dot(out ⊙ grad_output))
        // 单表达式融合：row_dot 为归约指令，消除 ep/gmd 等全尺寸中间 Tensor。
        auto out = dsl::compute(engine,
            dsl::leaf(output_cache_)
            * (dsl::leaf(grad_output)
               - dsl::row_reduce_sum(dsl::leaf(output_cache_)
                                     * dsl::leaf(grad_output))),
            output_cache_.rows(), output_cache_.cols(), p_.stable);
        return out;
    }
};

} // namespace nn

