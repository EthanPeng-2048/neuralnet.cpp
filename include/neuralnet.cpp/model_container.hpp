#ifndef NN_MODEL_CONTAINER_HPP
#define NN_MODEL_CONTAINER_HPP

// ── model_container.hpp — 引擎化模型容器 ───────────────────────────────────
//
// 架构铁律：
//   1. Model 持有 observer_ptr<ComputeEngine>（非拥有），所有 forward/backward 委托给
//      Layer::forward(engine, Tensor) / Layer::backward(engine, Tensor)。
//   2. Model 不包含任何 GPU-resident 中间结果缓存或 forward_gpu/backward_gpu
//      路径 —— 设备分发完全由 ComputeEngine 实现负责。
//   3. forward/backward 操作 Tensor，单套实现适配 CPU/GPU。
//
// 数据流（GPU 训练为例）：
//   Matrix(x_batch) ──engine.from_matrix──▶ Tensor[GPU]
//     ──forward──▶ Tensor[GPU] ──loss.forward──▶ Scalar
//     ──loss.backward──▶ Tensor[GPU] ──backward──▶ (丢弃)
//     ──optimizer.step──▶ 参数就地更新（全程在 GPU）
//   仅 evaluate 时 engine.to_matrix 下载到 CPU 做 argmax。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "core_assert.hpp"
#include "compute_layer.hpp"
#include "model_spec.hpp"

namespace nn
{

class Model
{
private:
    observer_ptr<ComputeEngine> engine_;
    std::vector<std::unique_ptr<Layer>> layers_;
    // 可选架构规格：由 build_*_from_spec 工厂在构建时设置，供 load_model
    // 校验文件头部与模型架构是否一致。为空表示未记录（跳过校验，向后兼容）。
    std::optional<ModelSpec> spec_;

public:
    Model() = default;
    explicit Model(ComputeEngine& engine) : engine_(engine) {}

    // 不可拷贝（unique_ptr 语义），可移动
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    // ── 引擎绑定 ─────────────────────────────────────────────────────────
    void set_engine(ComputeEngine& engine) { engine_.reset(&engine); }
    [[nodiscard]] ComputeEngine& engine() const noexcept
    {
        NN_ASSERT(engine_, "Model: engine not bound");
        return *engine_;
    }

    // ── 构建网络（通用模板接口） ────────────────────────────────────────
    // 用法：model.add<Linear>(engine, in, out) / model.add<ReLU>() / ...
    // 注意：MultiHeadAttention 的构造函数是 protected（设计为基类，
    //       由 TransformerEncoderLayer / CausalSelfAttention 内部组合），
    //       因此不直接通过 add 添加。
    template <typename LayerType, typename... Args>
    Model& add(Args&&... args)
    {
        layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
        return *this;
    }

    // 添加已由工厂构造的 Layer（如 make_norm_layer 按 NormType 创建归一化层）
    Model& add_layer(std::unique_ptr<Layer> layer)
    {
        layers_.emplace_back(std::move(layer));
        return *this;
    }

    // ── 访问 ─────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

    [[nodiscard]] Layer& layer_at(std::size_t index) noexcept
    {
        NN_ASSERT(index < layers_.size(), "Model::layer_at index out of range");
        return *layers_[index];
    }

    // ── 架构规格（可选）：记录本模型对应的 ModelSpec ────────────────────
    // build_*_from_spec 工厂自动设置；也可手动 set_spec 以启用 load_model 校验。
    void set_spec(const ModelSpec& spec) { spec_ = spec; }
    [[nodiscard]] const std::optional<ModelSpec>& spec() const noexcept { return spec_; }

    // ── batch 录制粒度：在 Transformer block 间按间隔 flush ──
    // 通过 Layer 基类虚函数分发：GPTModel override 生效，
    // 其他层类型默认 no-op。
    void set_flush_interval(std::size_t interval)
    {
        for (auto& layer : layers_)
            layer->set_flush_interval(interval);
    }

    // ── 梯度检查点（激活重计算 L1）：按间隔配置各层 ──
    // 通过 Layer 基类虚函数分发：GPTModel override 生效，
    // 其他层类型默认 no-op。
    void set_checkpoint_every(std::size_t stride)
    {
        for (auto& layer : layers_)
            layer->set_checkpoint_every(stride);
    }

    // ── activation offload（L1-offload）：把激活搬 host-visible ──
    // 通过 Layer 基类虚函数分发：GPTModel override 生效，
    // 其他层类型默认 no-op。
    void set_activation_offload(bool enabled)
    {
        for (auto& layer : layers_)
            layer->set_activation_offload(enabled);
    }

    // 理论 offload RAM 字节数（各层累计，用于诊断）
    [[nodiscard]] std::size_t offload_ram_bytes()
    {
        std::size_t total = 0;
        for (auto& layer : layers_)
            total += layer->offload_ram_bytes();
        return total;
    }

    // ── 训练/推理模式切换：转发给各 Layer（BatchNorm 等需要区分） ──
    void set_training(bool training)
    {
        for (auto& layer : layers_)
            layer->set_training(training);
    }

    // ── 文档感知：设置当前 step 每样本文档 id（转发给各 Layer） ─────────
    // GPTModel override 生效；其他层默认 no-op。
    void set_doc_ids(std::span<const std::size_t> ids)
    {
        for (auto& layer : layers_)
            layer->set_doc_ids(ids);
    }

    // ── 前向传播：Tensor → Tensor（全程不离开 engine 设备） ───────────────
    [[nodiscard]] Result<Tensor> forward(const Tensor& input)
    {
        if (layers_.empty())
            return std::unexpected(Error{"Model::forward: no layers"});
        if (engine_ == nullptr)
            return std::unexpected(Error{"Model::forward: engine not bound"});

        Tensor x = input;
        for (auto& layer : layers_)
        {
            auto r = layer->forward(*engine_, x);
            if (!r) return std::unexpected(r.error());
            x = std::move(*r);
        }
        return x;
    }

    // ── 反向传播：Tensor → Tensor（从最后一层到第一层） ───────────────────
    [[nodiscard]] Result<Tensor> backward(const Tensor& grad_output)
    {
        if (layers_.empty())
            return std::unexpected(Error{"Model::backward: no layers"});
        if (engine_ == nullptr)
            return std::unexpected(Error{"Model::backward: engine not bound"});

        Tensor g = grad_output;
        for (std::size_t i = layers_.size(); i-- > 0;)
        {
            auto r = layers_[i]->backward(*engine_, g);
            if (!r) return std::unexpected(r.error());
            g = std::move(*r);
        }
        return g;
    }

    // ── 参数收集：返回 TensorRef 供 Optimizer 使用 ──────────────────────────
    [[nodiscard]] std::vector<TensorRef> parameters()
    {
        std::vector<TensorRef> result;
        for (auto& layer : layers_)
        {
            auto layer_params = layer->parameters();
            result.insert(result.end(), layer_params.begin(), layer_params.end());
        }
        return result;
    }

    // ── 非可学习状态收集（如 BatchNorm 的 running 统计量，供序列化） ────────
    [[nodiscard]] std::vector<TensorRef> extra_state()
    {
        std::vector<TensorRef> result;
        for (auto& layer : layers_)
        {
            auto layer_extra = layer->extra_state();
            result.insert(result.end(), layer_extra.begin(), layer_extra.end());
        }
        return result;
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients()
    {
        std::vector<TensorRef> result;
        for (auto& layer : layers_)
        {
            auto layer_grads = layer->param_gradients();
            result.insert(result.end(), layer_grads.begin(), layer_grads.end());
        }
        return result;
    }

    // ── 梯度清零（每个训练 step 开头调用） ────────────────────────────────
    [[nodiscard]] Result<void> zero_grad()
    {
        for (auto& layer : layers_)
        {
            auto r = layer->zero_grad(*engine_);
            if (!r) return r;
        }
        return {};
    }
};

} // namespace nn

#endif // NN_MODEL_CONTAINER_HPP
