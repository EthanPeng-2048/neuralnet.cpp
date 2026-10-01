#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  examples/fusion_custom_layer.hpp — 「自研层」示例（A2 验收）
//
//  模拟一个**库外**使用者的自定义层：表达式结构在库内不存在，只在本头里写。
//  配合 `nn_enable_gpu_fusion(<target>)`，使用者**只做两件事**：
//    ① 写 `dsl::compute`（本文件）；
//    ② 在 CMake 里加一行 `nn_enable_gpu_fusion(<target>)`。
//  不需要写 dry-run 条目、不需要精度回填清单、不需要碰库内任何文件。
//
//  注意：库内规则「表达式文本只出现在 Layer」对**库外使用者**的自然对应物
//  就是"只出现在自己的层/算子头里"。
// ═══════════════════════════════════════════════════════════════════════════

#include <neuralnet.cpp/nn.hpp>

namespace nn_example
{

// 一个刻意与库内任何结构都不同的融合表达式：
//   out = exp(|x| * alpha) / (sqrt(|y|) + alpha)
// 两个运行时标量参数（alpha）→ 值不进 key，同结构共享 shader。
// 入参只读：`dsl::leaf` 按值收 Tensor（句柄拷贝），表达式不写输入。
[[nodiscard]] inline nn::Result<nn::Tensor> fused_custom_op(
    nn::ComputeEngine& engine, const nn::Tensor& x, const nn::Tensor& y, float alpha,
    nn::Precision P = nn::Precision::F32)
{
    return nn::dsl::compute(engine,
        nn::dsl::exp(nn::dsl::abs(nn::dsl::leaf(x)) * nn::dsl::rparam(alpha))
            / (nn::dsl::sqrt(nn::dsl::abs(nn::dsl::leaf(y)))
               + nn::dsl::rparam(alpha)),
        x.rows(), x.cols(), P);
}

// ── (b) 归一化形态：列归约 + 广播 + rsqrt ──────────────────────────────
// 与库内 LayerNorm / RMSNorm 的 std_inv 表达式**同形**（raxis=1、
// viewkinds=[0,8]）。库里这一类属于实测"只有执行 Layer 才拿得到"的
// 25 条之一——即**锚点的符号实例与运行期实例折叠出不同结构**。
// 库外自定义层没有 dry-run/模型 pass，因此本函数是对 `nn_enable_gpu_fusion`
// **可靠性**的直接检验：注册表若漏了运行期那一条，GPU 会闭合世界硬报错。
[[nodiscard]] inline nn::Result<nn::Tensor> fused_custom_norm(
    nn::ComputeEngine& engine, const nn::Tensor& x, float inv_cols, float eps,
    nn::Precision P = nn::Precision::F32)
{
    return nn::dsl::compute(engine,
        nn::dsl::rsqrt(nn::dsl::col_reduce_sum(nn::dsl::leaf(x))
                           * nn::dsl::rparam(inv_cols) + nn::dsl::rparam(eps)),
        x.rows(), x.cols(), P);
}

} // namespace nn_example
