#ifndef MNIST_COMMON_HPP
#define MNIST_COMMON_HPP

#include <cstddef>
#include <string>
#include <expected>
#include <vector>

#include "model.hpp"
#include "model_spec.hpp"

namespace nn {

// ── MNIST 常量 ──────────────────────────────────────────────────────────────
inline constexpr std::size_t MNIST_INPUT_DIM = 784;
inline constexpr std::size_t MNIST_NUM_CLASSES = 10;

// 默认 MLP 网络架构：输入层 -> 隐藏层1 -> 隐藏层2 -> 隐藏层3 -> 输出层
inline const std::vector<std::size_t> MNIST_LAYER_DIMS = {
    MNIST_INPUT_DIM, 512, 256, 128, 64, MNIST_NUM_CLASSES
};

// 默认 Transformer 超参数
inline constexpr std::size_t TRANSFORMER_D_MODEL   = 64;
inline constexpr std::size_t TRANSFORMER_NUM_HEADS  = 4;
inline constexpr std::size_t TRANSFORMER_D_FF       = 128;
inline constexpr std::size_t TRANSFORMER_NUM_LAYERS = 2;
inline constexpr std::size_t TRANSFORMER_PATCH_SIZE = 7;   // 28 / 7 = 4 → 4×4 = 16 patches

// ── 构建 MLP 模型 ──────────────────────────────────────────────────────────
// 可指定自定义层维度，也可使用默认 MNIST_LAYER_DIMS
[[nodiscard]] inline Result<Model> build_mnist_mlp_model(
    const std::vector<std::size_t> &layer_dims = MNIST_LAYER_DIMS)
{
    if (layer_dims.size() < 2)
        return std::unexpected(Error{"MLP layer_dims must have at least 2 elements"});

    Model model;
    for (std::size_t i = 0; i < layer_dims.size() - 1; ++i)
    {
        std::size_t in_dim  = layer_dims[i];
        std::size_t out_dim = layer_dims[i + 1];

        model.add_linear(in_dim, out_dim);

        if (i < layer_dims.size() - 2)
        {
            model.add_layer_norm(out_dim)
                 .add_gelu();
        }
    }
    return model;
}

// ── 构建 Transformer 模型 (ViT-like) ──────────────────────────────────────
[[nodiscard]] inline Result<Model> build_mnist_transformer_model(
    std::size_t d_model    = TRANSFORMER_D_MODEL,
    std::size_t num_heads  = TRANSFORMER_NUM_HEADS,
    std::size_t d_ff       = TRANSFORMER_D_FF,
    std::size_t num_layers = TRANSFORMER_NUM_LAYERS,
    std::size_t patch_size = TRANSFORMER_PATCH_SIZE)
{
    constexpr std::size_t img_size = 28;
    const std::size_t num_patches = (img_size / patch_size) * (img_size / patch_size);

    Model model;
    model.add_patch_embedding(img_size, patch_size, d_model)
         .add_transformer_encoder(d_model, num_heads, d_ff, num_layers, num_patches)
         .add_linear(d_model, MNIST_NUM_CLASSES);
    return model;
}

// ── 统一构建入口（字符串参数） ─────────────────────────────────────────────
[[nodiscard]] inline Result<Model> build_mnist_model(const std::string &model_type = "mlp")
{
    if (model_type == "mlp")
        return build_mnist_mlp_model();
    else if (model_type == "transformer")
        return build_mnist_transformer_model();
    else
        return std::unexpected(Error{"Unknown model type: " + model_type
                                    + " (available: mlp, transformer)"});
}

// ── 从 ModelSpec 构建模型 ─────────────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构
[[nodiscard]] inline Result<Model> build_mnist_model_from_spec(const ModelSpec &spec)
{
    if (spec.is_mlp())
    {
        return build_mnist_mlp_model(spec.layer_dims);
    }
    else if (spec.is_transformer())
    {
        return build_mnist_transformer_model(
            spec.d_model, spec.num_heads, spec.d_ff,
            spec.num_layers, spec.patch_size);
    }
    else
    {
        return std::unexpected(Error{
            "Invalid ModelSpec type for MNIST: expected MLP or Transformer"});
    }
}

} // namespace nn

#endif // MNIST_COMMON_HPP