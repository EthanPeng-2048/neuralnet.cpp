#ifndef NN_MODEL_CONTAINER_HPP
#define NN_MODEL_CONTAINER_HPP

// ── model_container.hpp — 引擎化模型容器 ───────────────────────────────────
//
// 架构铁律：
//   1. Model 持有 ComputeEngine*（非拥有），所有 forward/backward 委托给
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
    ComputeEngine* engine_ = nullptr;
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    Model() = default;
    explicit Model(ComputeEngine& engine) : engine_(&engine) {}

    // 不可拷贝（unique_ptr 语义），可移动
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    // ── 引擎绑定 ─────────────────────────────────────────────────────────
    // Model 必须在 add_* 之前绑定引擎（Linear/LayerNorm 构造需要 engine
    // 来创建同设备权重张量）。
    void set_engine(ComputeEngine& engine) { engine_ = &engine; }
    [[nodiscard]] ComputeEngine& engine() const noexcept
    {
        NN_ASSERT(engine_ != nullptr, "Model: engine not bound");
        return *engine_;
    }

    // ── 构建网络（MLP 专用，L4 调用） ────────────────────────────────────
    Model& add_linear(std::size_t in_features, std::size_t out_features)
    {
        layers_.emplace_back(std::make_unique<Linear>(*engine_, in_features, out_features));
        return *this;
    }

    Model& add_relu()
    {
        layers_.emplace_back(std::make_unique<ReLU>());
        return *this;
    }

    Model& add_gelu()
    {
        layers_.emplace_back(std::make_unique<GeLU>());
        return *this;
    }

    Model& add_layer_norm(std::size_t normalized_shape, Scalar epsilon = 1e-5)
    {
        layers_.emplace_back(std::make_unique<LayerNorm>(*engine_, normalized_shape, epsilon));
        return *this;
    }

    // 通用模板接口（供 gpu_test 等内部测试使用）
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

    // ── 参数收集：返回 Tensor* 供 Optimizer 使用 ──────────────────────────
    [[nodiscard]] std::vector<Tensor*> parameters()
    {
        std::vector<Tensor*> result;
        for (auto& layer : layers_)
        {
            auto layer_params = layer->parameters();
            result.insert(result.end(), layer_params.begin(), layer_params.end());
        }
        return result;
    }

    [[nodiscard]] std::vector<Tensor*> param_gradients()
    {
        std::vector<Tensor*> result;
        for (auto& layer : layers_)
        {
            auto layer_grads = layer->param_gradients();
            result.insert(result.end(), layer_grads.begin(), layer_grads.end());
        }
        return result;
    }

    // ── 梯度清零（每个训练 step 开头调用） ────────────────────────────────
    void zero_grad()
    {
        for (auto& layer : layers_)
            layer->zero_grad(*engine_);
    }
};

} // namespace nn

#endif // NN_MODEL_CONTAINER_HPP
