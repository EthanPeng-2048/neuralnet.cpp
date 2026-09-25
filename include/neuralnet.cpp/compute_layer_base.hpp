#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <utility>
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
    bool checkpoint_mode_ = false;

    // D7：精度配置（§9.2，Model 在 add/构造 Layer 时注入，每层一份）
    // 默认全 F32 = 现状行为，零回归（G5）
    PrecisionProfile p_;

public:
    virtual ~Layer() = default;

    // ── D7：精度配置注入（§9.2）─────────────────────────────────────────
    // virtual：复合层/持有辅助对象（RoPE、位置编码器）的层需要把 profile 继续
    // 下传（否则辅助对象内的 DSL 求值退回 F32，静默丢掉 f16 存储收益）。
    virtual void set_precision_profile(const PrecisionProfile& profile) { p_ = profile; }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

    // forward/backward 接收 ComputeEngine 引用，自动适配 CPU/GPU
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
// ══════════════════════════════════════════════════════════════════════════
// ActivationOffloader — 通用激活 offload（L1-offload）
//
// 把「本层 backward 所需的中间激活」（由 Layer::activation_cache() 枚举）
// 逐个写进一块持久 host-visible slab，随后把成员张量置空以释放 device-local
// 显存；backward 入口再从 slab 恢复。与具体层无关：GPTBlock / RAPTBlock 共用
// 同一实现（避免两份易漂移的拷贝）。
//
// 用法：
//   offloader.set_enabled(true);
//   ... forward ...
//   offloader.export_activations(engine, activation_cache());  // 导出并释放
//   ... backward 入口 ...
//   offloader.import_activations(engine);                      // 恢复
//
// 注意：export 依赖 forward 后成员地址稳定（refs_ 持有成员引用）；
//       无有效缓存时（如 checkpoint 模式 forward 不驻留激活）静默跳过。
// ══════════════════════════════════════════════════════════════════════════
class ActivationOffloader
{
private:
    bool enabled_ = false;
    bool offloaded_ = false;
    Tensor slab_;                                   // 持久 host-visible 缓冲（跨 step 复用）
    std::vector<TensorRef> refs_;                   // 各激活成员引用（地址稳定）
    std::vector<std::pair<std::size_t, std::size_t>> shapes_;  // 各激活形状
    std::vector<std::size_t> offsets_;              // 各激活在 slab 中的 float 偏移

public:
    void set_enabled(bool enabled) noexcept
    {
        enabled_ = enabled;
        if (!enabled) { slab_ = Tensor{}; offloaded_ = false; }  // 释放持久缓冲
    }
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] bool offloaded() const noexcept { return offloaded_; }

    // 导出：逐个写入 slab，写入后置空成员张量（数据已在 host slab）
    [[nodiscard]] Result<void> export_activations(
        ComputeEngine& engine, std::vector<TensorRef> refs)
    {
        if (!enabled_ || offloaded_) return {};
        refs_ = std::move(refs);
        offsets_.clear();
        shapes_.clear();
        // slab 容量校验（缺陷修复）：原先只在 !slab_.valid() 时按**当时**的
        // 激活总量分配，之后永不增长。若后续 step 的激活总量更大（批大小/
        // 序列长度变化、--resume 后续训、最后一个不满 batch 之后的 step 等），
        // offload_save 会按新 offset 越界写 slab → 缓冲区破坏/设备丢失。
        // 现在每次导出都按当前总量校验，不足则重建。
        std::size_t needed = 0;
        for (auto& ref : refs_)
            if (ref.get().valid()) needed += ref.get().size();
        if (needed == 0) { offloaded_ = false; return {}; }
        if (!slab_.valid() || slab_.size() < needed)
        {
            auto slab = engine.create_offload_buffer(needed);
            if (!slab) return std::unexpected(slab.error());
            slab_ = std::move(*slab);
        }
        std::size_t offset = 0;
        for (auto& ref : refs_)
        {
            if (!ref.get().valid()) continue;
            auto r = engine.offload_save(slab_, offset, ref.get());
            if (!r) return std::unexpected(r.error());
            shapes_.push_back({ref.get().rows(), ref.get().cols()});
            offsets_.push_back(offset);
            offset += ref.get().size();
            ref.get() = Tensor{};   // 释放 GPU 版（数据已在 host slab）
        }
        offloaded_ = true;
        return {};
    }

    // 导入：从 host slab 恢复激活到缓存成员（backward 前调用，替代重计算）
    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        if (!offloaded_) return {};
        for (std::size_t i = 0; i < offsets_.size(); ++i)
        {
            auto t = engine.offload_restore(slab_, offsets_[i],
                                            shapes_[i].first, shapes_[i].second);
            if (!t) return std::unexpected(t.error());
            refs_[i].get() = std::move(*t);
        }
        offloaded_ = false;
        return {};
    }

    // 实际 slab 字节数（诊断用；未创建时 0）
    [[nodiscard]] std::size_t slab_bytes() const noexcept
    {
        return slab_.valid() ? slab_.size() * sizeof(float) : 0;
    }
};

[[nodiscard]] inline Result<Tensor> clone_tensor(
    ComputeEngine& engine, const Tensor& src)
{
    return engine.clone(src);
}

} // namespace nn

