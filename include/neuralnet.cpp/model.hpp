#ifndef MODEL_HPP
#define MODEL_HPP

#include <cassert>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "nn_config.hpp"
#include "layer.hpp"

namespace nn
{
    class Model
    {
    private:
        std::vector<std::unique_ptr<Layer>> layers_;

    public:
        Model() = default;

        // 访问指定层（用于向下转型等场景）
        // 使用 assert 保护边界，编程错误在 debug 模式捕获
        [[nodiscard]] Layer &layer_at(std::size_t index) noexcept
        {
            assert(index < layers_.size() && "Model::layer_at index out of range");
            return *layers_[index];
        }

        // 不可拷贝（unique_ptr 语义），只能移动
        Model(const Model &) = delete;
        Model &operator=(const Model &) = delete;
        Model(Model &&) noexcept = default;
        Model &operator=(Model &&) noexcept = default;

        // ── 构建网络（模板接口，仅供 L3 内部使用）──────────────────────
        template <typename LayerType, typename... Args>
        Model &add(Args &&...args)
        {
            layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
            return *this;
        }

        // ── L4 构建层接口（非模板，L4 无需知道 Layer 类型）──────────────
        // L4 通过这些方法组装模型，不直接接触 L2 的层类型。
        Model &add_linear(std::size_t in_features, std::size_t out_features);
        Model &add_relu();
        Model &add_gelu();
        Model &add_layer_norm(std::size_t normalized_shape, Scalar epsilon = 1e-5);
        Model &add_softmax();
        Model &add_patch_embedding(std::size_t img_size, std::size_t patch_size, std::size_t d_model);
        Model &add_transformer_encoder(std::size_t d_model, std::size_t num_heads,
                                       std::size_t d_ff, std::size_t num_layers,
                                       std::size_t num_patches);
        Model &add_gpt_model(std::size_t vocab_size, std::size_t d_model,
                             std::size_t seq_len, std::size_t num_heads,
                             std::size_t d_ff, std::size_t num_layers);

        [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

        // ── 前向传播 ────────────────────────────────────────────────────────
        // 当 GPU 启用且可用时，自动使用双轨制 GPU 流水线：
        //   Input → Upload → [Layer1.forward_gpu → Layer2.forward_gpu → ...] → Download → Output
        // 中间所有层的矩阵乘法和激活函数全程在 GPU 显存中流转，零 PCIe 中间传输。
        [[nodiscard]] Result<Matrix> forward(const Matrix &input)
        {
            if (layers_.empty())
            {
                return std::unexpected(Error{"Model has no layers"});
            }

#ifdef NN_HAS_VULKAN
            // ── GPU 流水线路径 ─────────────────────────────────────────
            if (SmartPolicy::gpu_enabled)
            {
                auto& backend = GpuBackend::instance();
                if (backend.is_initialized() || backend.initialize())
                {
                    // 1. Upload 输入到 GPU（唯一一次 CPU→GPU 传输）
                    auto gpu_in = GpuTensor::from_matrix(input, backend);
                    if (gpu_in)
                    {
                        // 2. 逐层 forward_gpu（全程 GPU 显存流转）
                        GpuTensor gpu_out = std::move(*gpu_in);
                        bool gpu_ok = true;
                        for (std::size_t i = 0; i < layers_.size(); ++i)
                        {
                            auto layer_res = layers_[i]->forward_gpu(gpu_out, backend);
                            if (!layer_res) { gpu_ok = false; break; }
                            gpu_out = std::move(*layer_res);
                        }
                        if (gpu_ok)
                        {
                            // 3. Download 输出到 CPU（唯一一次 GPU→CPU 传输）
                            auto cpu_out = gpu_out.to_matrix(backend);
                            if (cpu_out) return cpu_out;
                        }
                    }
                    // GPU 路径失败，静默 fallback 到 CPU 路径
                }
            }
#endif

            // ── CPU 路径（原始逻辑）────────────────────────────────────
            auto result = layers_.front()->forward(input);
            if (!result) return std::unexpected(result.error());
            Matrix out = std::move(*result);
            for (std::size_t i = 1; i < layers_.size(); ++i)
            {
                result = layers_[i]->forward(out);
                if (!result) return std::unexpected(result.error());
                out = std::move(*result);
            }
            return out;
        }

        // ── 反向传播 ────────────────────────────────────────────────────────
        // 传入 loss 对最后一层输出的梯度，返回对输入的梯度（通常不需要）
        [[nodiscard]] Result<Matrix> backward(const Matrix &grad_output)
        {
            if (layers_.empty())
            {
                return std::unexpected(Error{"Model has no layers"});
            }
            auto result = layers_.back()->backward(grad_output);
            if (!result) return std::unexpected(result.error());
            Matrix grad = std::move(*result);
            for (std::size_t i = layers_.size() - 1; i-- > 0;)
            {
                result = layers_[i]->backward(grad);
                if (!result) return std::unexpected(result.error());
                grad = std::move(*result);
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

    // ════════════════════════════════════════════════════════════════════════
    // L4 构建层接口实现（inline，L4 只调用这些方法，不接触 Layer 类型）
    // ════════════════════════════════════════════════════════════════════════

    inline Model &Model::add_linear(std::size_t in_features, std::size_t out_features)
    {
        return add<Linear>(in_features, out_features);
    }

    inline Model &Model::add_relu()
    {
        return add<ReLU>();
    }

    inline Model &Model::add_gelu()
    {
        return add<GeLU>();
    }

    inline Model &Model::add_layer_norm(std::size_t normalized_shape, Scalar epsilon)
    {
        return add<LayerNorm>(normalized_shape, epsilon);
    }

    inline Model &Model::add_softmax()
    {
        return add<Softmax>();
    }

    inline Model &Model::add_patch_embedding(std::size_t img_size, std::size_t patch_size,
                                              std::size_t d_model)
    {
        return add<PatchEmbedding>(img_size, patch_size, d_model);
    }

    inline Model &Model::add_transformer_encoder(std::size_t d_model, std::size_t num_heads,
                                                  std::size_t d_ff, std::size_t num_layers,
                                                  std::size_t num_patches)
    {
        return add<TransformerEncoder>(d_model, num_heads, d_ff, num_layers, num_patches);
    }

    inline Model &Model::add_gpt_model(std::size_t vocab_size, std::size_t d_model,
                                        std::size_t seq_len, std::size_t num_heads,
                                        std::size_t d_ff, std::size_t num_layers)
    {
        return add<GPTModel>(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers);
    }

} // namespace nn

#endif // MODEL_HPP