#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nn
{
    class Model
    {
    private:
        std::vector<std::unique_ptr<Layer>> layers_;

    public:
        Model() = default;

        // 不可拷贝（unique_ptr 语义），只能移动
        Model(const Model &) = delete;
        Model &operator=(const Model &) = delete;
        Model(Model &&) noexcept = default;
        Model &operator=(Model &&) noexcept = default;

        // ── 构建网络 ────────────────────────────────────────────────────────
        // 用法：model.add<nn::Linear>(784, 64).add<nn::ReLU>()...
        template <typename LayerType, typename... Args>
        Model &add(Args &&...args)
        {
            layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
            return *this;
        }

        [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

        // ── 前向传播 ────────────────────────────────────────────────────────
        [[nodiscard]] Matrix forward(const Matrix &input)
        {
            if (layers_.empty())
            {
                throw std::runtime_error("Model has no layers");
            }
            Matrix out = layers_.front()->forward(input);
            for (std::size_t i = 1; i < layers_.size(); ++i)
            {
                out = layers_[i]->forward(out);
            }
            return out;
        }

        // ── 反向传播 ────────────────────────────────────────────────────────
        // 传入 loss 对最后一层输出的梯度，返回对输入的梯度（通常不需要）
        Matrix backward(const Matrix &grad_output)
        {
            if (layers_.empty())
            {
                throw std::runtime_error("Model has no layers");
            }
            Matrix grad = layers_.back()->backward(grad_output);
            for (std::size_t i = layers_.size() - 1; i-- > 0;)
            {
                grad = layers_[i]->backward(grad);
            }
            return grad;
        }

        // ── 参数收集 ────────────────────────────────────────────────────────
        // 聚合所有层的可训练参数，供 Optimizer 使用
        [[nodiscard]] std::vector<std::reference_wrapper<Matrix>> parameters()
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &layer : layers_)
            {
                for (auto &p : layer->parameters())
                {
                    result.push_back(p);
                }
            }
            return result;
        }

        [[nodiscard]] std::vector<std::reference_wrapper<Matrix>> param_gradients()
        {
            std::vector<std::reference_wrapper<Matrix>> result;
            for (auto &layer : layers_)
            {
                for (auto &g : layer->param_gradients())
                {
                    result.push_back(g);
                }
            }
            return result;
        }
    };

} // namespace nn

#endif // MODEL_HPP