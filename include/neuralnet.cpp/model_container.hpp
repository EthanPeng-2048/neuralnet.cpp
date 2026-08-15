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
#include <vector>

#include "core_assert.hpp"
#include "compute_layer.hpp"

namespace nn
{

class Model
{
private:
    observer_ptr<ComputeEngine> engine_;
    std::vector<std::unique_ptr<Layer>> layers_;

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

    // ── 访问 ─────────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

    [[nodiscard]] Layer& layer_at(std::size_t index) noexcept
    {
        NN_ASSERT(index < layers_.size(), "Model::layer_at index out of range");
        return *layers_[index];
    }

    // ── batch 录制粒度：在 Transformer block 间按间隔 flush ──
    // 通过 Layer 基类虚函数分发：GPTModel override 生效，
    // 其他层类型默认 no-op。
    void set_flush_interval(std::size_t interval)
    {
        for (auto& layer : layers_)
            layer->set_flush_interval(interval);
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
