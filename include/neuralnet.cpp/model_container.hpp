#ifndef NN_MODEL_CONTAINER_HPP
#define NN_MODEL_CONTAINER_HPP

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "core_assert.hpp"
#include "config.hpp"
#include "compute_layer.hpp"

// GPU 加速支持（可选）
#ifdef NN_HAS_VULKAN
#include "backend/vk_backend.hpp"
#endif

namespace nn
{
    class Model
    {
    private:
        std::vector<std::unique_ptr<Layer>> layers_;

#ifdef NN_HAS_VULKAN
        // ── GPU 中间结果缓存（延迟下载：backward 时按需下载）────────────
        // forward 时保留所有中间 GpuTensor，backward 时按需下载每层输入
        mutable std::vector<GpuTensor> gpu_intermediates_;
        mutable GpuBackend *gpu_backend_ = nullptr;
#endif

    public:
        Model() = default;

        // 访问指定层（用于向下转型等场景）
        // 使用 NN_ASSERT 保护边界，编程错误在 debug 模式捕获
        [[nodiscard]] Layer &layer_at(std::size_t index) noexcept
        {
            NN_ASSERT(index < layers_.size(), "Model::layer_at index out of range");
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

        // ── 序列生成（仅对包含 GPTModel 等自回归层的模型有效） ─────────
        // 通过 Layer 虚函数 dispatch 调用，避免 L5 层使用 dynamic_cast。
        // 委托给模型的第一层（GPT 通常是容器内唯一层）。
        [[nodiscard]] Result<std::vector<std::size_t>>
        generate(const std::vector<std::size_t> &prompt,
                 std::size_t max_new_tokens,
                 Scalar temperature = 1.0)
        {
            if (layers_.empty())
                return std::unexpected(Error{"Model has no layers"});
            return layers_.front()->generate(prompt, max_new_tokens, temperature);
        }

        // ── 前向传播 ────────────────────────────────────────────────────────
        // 通过 Matrix 语义方法（multiply_to）和表达式模板（compute::apply）自动分派，
        // 上层代码完全符合 L(N)→L(N-1) 分层调用规则。
        //
        // 当 GPU 启用且可用时，自动使用 GPU-resident 流水线：
        //   Input → Upload → [Layer1.forward_gpu → Layer2.forward_gpu → ...] → Download → Output
        // 中间所有层的矩阵乘法和激活函数全程在 GPU 显存中流转，零 PCIe 中间传输。
        [[nodiscard]] Result<Matrix> forward(const Matrix &input)
        {
            if (layers_.empty())
            {
                return std::unexpected(Error{"Model has no layers"});
            }

#ifdef NN_HAS_VULKAN
            // ── GPU-resident 流水线路径 ─────────────────────────────────────
            if (SmartPolicy::gpu_enabled)
            {
                auto &backend = GpuBackend::instance();
                if (backend.is_initialized() || backend.initialize())
                {
                    // 1. Upload 输入到 GPU（唯一一次 CPU→GPU 传输）
                    auto gpu_in = GpuTensor::from_matrix(input, backend);
                    if (gpu_in)
                    {
                        // 2. 开始 batch 模式：所有层录制到同一个 command buffer
                        auto batch_r = backend.begin_batch();
                        if (batch_r)
                        {
                            // 保留所有中间 GpuTensor 的所有权
                            gpu_intermediates_.clear();
                            gpu_intermediates_.reserve(layers_.size() + 1);
                            gpu_intermediates_.push_back(std::move(*gpu_in));

                            bool gpu_ok = true;
                            for (std::size_t i = 0; i < layers_.size(); ++i)
                            {
                                auto layer_res = layers_[i]->forward_gpu(
                                    gpu_intermediates_.back(), backend);
                                if (!layer_res)
                                {
                                    gpu_ok = false;
                                    break;
                                }
                                gpu_intermediates_.push_back(std::move(*layer_res));
                            }
                            if (gpu_ok)
                            {
                                // 3. 提交 batch（一次提交、一次 fence wait）
                                auto end_r = backend.end_batch();
                                if (end_r)
                                {
                                    // 4. 保存 backend 指针供 backward 使用
                                    gpu_backend_ = &backend;

                                    // 5. Download 最终输出到 CPU（唯一一次 GPU→CPU 传输）
                                    auto cpu_out = gpu_intermediates_.back().to_matrix(backend);
                                    if (cpu_out) return cpu_out;
                                }
                            }
                            else
                            {
                                (void)backend.end_batch();
                            }
                        }
                    }
                    // GPU 路径失败，静默 fallback 到 CPU 路径
                    gpu_intermediates_.clear();
                }
            }
#endif

            // ── CPU 路径（原始逻辑）──────────────────────────────────────
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
        //
        // 当 GPU 中间结果可用时，使用 GPU-resident backward：
        //   - Linear: 两次 matmul 在 GPU（transA/transB），仅下载 grad_w/grad_b
        //   - ReLU: elementwise 在 GPU
        //   - 其他层: fallback 到 CPU
        [[nodiscard]] Result<Matrix> backward(const Matrix &grad_output)
        {
            if (layers_.empty())
            {
                return std::unexpected(Error{"Model has no layers"});
            }

#ifdef NN_HAS_VULKAN
            // ── GPU-resident backward 路径 ────────────────────────────────
            if (!gpu_intermediates_.empty() && gpu_backend_ &&
                gpu_intermediates_.size() == layers_.size() + 1)
            {
                auto &backend = *gpu_backend_;

                // 1. Upload grad_output 到 GPU
                auto gpu_grad = GpuTensor::from_matrix(grad_output, backend);
                if (gpu_grad)
                {
                    // 2. 逐层 backward_gpu（从最后一层到第一层）
                    GpuTensor grad = std::move(*gpu_grad);
                    bool gpu_ok = true;
                    for (std::size_t i = layers_.size(); i-- > 0;)
                    {
                        auto res = layers_[i]->backward_gpu(
                            grad, gpu_intermediates_[i], backend);
                        if (!res)
                        {
                            gpu_ok = false;
                            break;
                        }
                        grad = std::move(*res);
                    }

                    // 3. 清理 GPU 中间结果
                    gpu_intermediates_.clear();
                    gpu_backend_ = nullptr;

                    if (gpu_ok)
                    {
                        // 4. 下载最终 grad_input（通常不需要，但保持接口一致）
                        auto cpu_grad = grad.to_matrix(backend);
                        if (cpu_grad) return cpu_grad;
                    }
                }
                // GPU 路径失败，fallback 到 CPU
                gpu_intermediates_.clear();
                gpu_backend_ = nullptr;
            }
#endif

            // ── CPU 路径 ───────────────────────────────────────────────────
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

#ifdef NN_HAS_VULKAN
        // ── 使所有层的 GPU 权重缓存失效 ──────────────────────────────────
        // 训练时，在 optimizer.step() 更新 CPU 端权重后调用，
        // 确保下一次 forward_gpu 重新上传最新权重到 GPU。
        void invalidate_gpu_caches()
        {
            for (auto &layer : layers_)
                layer->invalidate_gpu_cache();
        }
#endif
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

#endif // NN_MODEL_CONTAINER_HPP