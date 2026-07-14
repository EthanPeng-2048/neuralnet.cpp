#ifndef MNIST_COMMON_HPP
#define MNIST_COMMON_HPP

#include <cstddef>
#include <string>
#include <stdexcept>
#include <vector>

#include "model.hpp"

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
// 根据 MNIST_LAYER_DIMS 自动构建网络 (Linear + LayerNorm + GeLU)
[[nodiscard]] inline Model build_mnist_mlp_model()
{
    Model model;
    for (std::size_t i = 0; i < MNIST_LAYER_DIMS.size() - 1; ++i)
    {
        std::size_t in_dim = MNIST_LAYER_DIMS[i];
        std::size_t out_dim = MNIST_LAYER_DIMS[i + 1];
        
        model.add<Linear>(in_dim, out_dim);
        
        if (i < MNIST_LAYER_DIMS.size() - 2)
        {
            model.add<LayerNorm>(out_dim)
                 .add<GeLU>();
        }
    }
    return model;
}

// ── 构建 Transformer 模型 (ViT-like) ──────────────────────────────────────
// 架构: PatchEmbedding → TransformerEncoder → Linear(分类头)
//
// 输入: (784, batch) — 展平的 28×28 图像
// PatchEmbedding: 28×28 → 16 个 7×7 patch → 投影到 d_model
// TransformerEncoder: N 层 Self-Attention + FFN (Pre-Norm) + 位置编码 + 池化
// 分类头: Linear(d_model → 10)
//
// 传播流程:
//   (784, batch)
//   → PatchEmbedding: (d_model×16, batch)
//   → TransformerEncoder: (d_model, batch)
//   → Linear: (10, batch)
// ────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline Model build_mnist_transformer_model(
    std::size_t d_model   = TRANSFORMER_D_MODEL,
    std::size_t num_heads = TRANSFORMER_NUM_HEADS,
    std::size_t d_ff      = TRANSFORMER_D_FF,
    std::size_t num_layers = TRANSFORMER_NUM_LAYERS,
    std::size_t patch_size = TRANSFORMER_PATCH_SIZE)
{
    constexpr std::size_t img_size = 28;
    const std::size_t num_patches = (img_size / patch_size) * (img_size / patch_size);

    Model model;
    model.add<PatchEmbedding>(img_size, patch_size, d_model)
         .add<TransformerEncoder>(d_model, num_heads, d_ff, num_layers, num_patches)
         .add<Linear>(d_model, MNIST_NUM_CLASSES);
    return model;
}

// ── 统一构建入口 ────────────────────────────────────────────────────────────
// model_type: "mlp" 或 "transformer"
[[nodiscard]] inline Model build_mnist_model(const std::string &model_type = "mlp")
{
    if (model_type == "mlp")
        return build_mnist_mlp_model();
    else if (model_type == "transformer")
        return build_mnist_transformer_model();
    else
        throw std::invalid_argument("Unknown model type: " + model_type
                                    + " (available: mlp, transformer)");
}

} // namespace nn

#endif // MNIST_COMMON_HPP