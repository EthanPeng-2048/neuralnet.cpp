#pragma once

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

} // namespace nn

