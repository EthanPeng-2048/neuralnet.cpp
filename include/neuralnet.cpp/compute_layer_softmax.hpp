#ifndef NN_COMPUTE_LAYER_SOFTMAX_HPP
#define NN_COMPUTE_LAYER_SOFTMAX_HPP

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

} // namespace nn

#endif // NN_COMPUTE_LAYER_SOFTMAX_HPP
