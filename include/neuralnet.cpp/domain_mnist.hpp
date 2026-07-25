#ifndef NN_DOMAIN_MNIST_HPP
#define NN_DOMAIN_MNIST_HPP

// ── domain_mnist.hpp — MNIST 领域构建层 ────────────────────────────────────
//
// 基于（引擎化架构）提供两种模型构建：
//   1. MLP           —— build_mnist_mlp_model
//   2. Transformer   —— build_mnist_transformer_model（ViT 风格）
//
// 依赖：Model + ComputeEngine（model_container.hpp / compute_engine.hpp）
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include "core_errors.hpp"
#include "compute_engine.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"

namespace nn {

// ── MNIST 常量 ──────────────────────────────────────────────────────────────
inline constexpr std::size_t MNIST_INPUT_DIM = 784;
inline constexpr std::size_t MNIST_NUM_CLASSES = 10;

// ── MNIST Transformer (ViT) 默认超参数 ──────────────────────────────────────
// 图像 28×28，patch_size=7 → 4×4=16 个 patch，patch_dim=49
inline constexpr std::size_t MNIST_IMG_SIZE    = 28;
inline constexpr std::size_t MNIST_PATCH_SIZE  = 7;
inline constexpr std::size_t MNIST_TF_D_MODEL  = 64;
inline constexpr std::size_t MNIST_TF_NUM_HEADS = 4;
inline constexpr std::size_t MNIST_TF_D_FF     = 128;
inline constexpr std::size_t MNIST_TF_NUM_LAYERS = 2;

// ── 共享 CSV 行解析工具 ────────────────────────────────────────────────────
// 解析单行 CSV（逗号分隔的浮点数）为 std::vector<Scalar>
[[nodiscard]] inline Result<std::vector<Scalar>>
parse_csv_line(const std::string &line)
{
    std::vector<Scalar> values;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        auto v = parse_number<Scalar>(token);
        if (!v)
            return std::unexpected(Error{"CSV 含无效数字 '" + token + "': " + v.error().message});
        values.push_back(*v);
    }
    return values;
}

// ── 从 CSV 行加载单张 MNIST 图片 ───────────────────────────────────────────
// 输入：784 个逗号分隔的像素值（0~255 或归一化后的 0~1）
// 输出：(784, 1) 列向量 Matrix
[[nodiscard]] inline Result<Matrix>
load_image_from_csv_line(const std::string &csv_line)
{
    auto values = parse_csv_line(csv_line);
    if (!values)
        return std::unexpected(std::move(values).error());
    if (values->size() != MNIST_INPUT_DIM)
        return std::unexpected(Error{"CSV 必须包含恰好 " + std::to_string(MNIST_INPUT_DIM) +
                                    " 个值，实际: " + std::to_string(values->size())});

    Matrix img(MNIST_INPUT_DIM, 1);
    for (std::size_t i = 0; i < MNIST_INPUT_DIM; ++i)
        img.set_value_unchecked(i, 0, (*values)[i]);
    return img;
}

// 默认 MLP 网络架构：输入层 → 隐藏层1 → 隐藏层2 → 隐藏层3 → 输出层
inline const std::vector<std::size_t> MNIST_LAYER_DIMS = {
    MNIST_INPUT_DIM, 512, 256, 128, 64, MNIST_NUM_CLASSES
};

// ── 构建 MLP 模型 ──────────────────────────────────────────────────────────
// 通过指定 ComputeEngine 创建同设备的权重张量；可指定自定义层维度。
// 结构：Linear → LayerNorm → GeLU × (N-1) + Linear（最后一层）
[[nodiscard]] inline Result<Model> build_mnist_mlp_model(
    ComputeEngine& engine,
    const std::vector<std::size_t> &layer_dims = MNIST_LAYER_DIMS)
{
    if (layer_dims.size() < 2)
        return std::unexpected(Error{"MLP layer_dims must have at least 2 elements"});

    Model model(engine);
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

// ── 构建 MNIST Transformer (ViT 风格) 模型 ──────────────────────────────────
// 结构：PatchEmbedding → TransformerEncoder → Linear(分类头)
//   PatchEmbedding: img 28×28 → 16 个 7×7 patch → Linear(49, d_model) 投影
//   TransformerEncoder: PE + N×EncoderLayer + 全局平均池化
//   Linear: d_model → 10（分类头）
[[nodiscard]] inline Result<Model> build_mnist_transformer_model(
    ComputeEngine& engine,
    std::size_t img_size   = MNIST_IMG_SIZE,
    std::size_t patch_size = MNIST_PATCH_SIZE,
    std::size_t d_model    = MNIST_TF_D_MODEL,
    std::size_t num_heads  = MNIST_TF_NUM_HEADS,
    std::size_t d_ff       = MNIST_TF_D_FF,
    std::size_t num_layers = MNIST_TF_NUM_LAYERS)
{
    if (img_size % patch_size != 0)
        return std::unexpected(Error{"MNIST Transformer: img_size must be divisible by patch_size"});
    if (d_model == 0 || num_heads == 0 || d_ff == 0 || num_layers == 0)
        return std::unexpected(Error{"MNIST Transformer: parameters must be positive"});
    if (d_model % num_heads != 0)
        return std::unexpected(Error{"MNIST Transformer: d_model must be divisible by num_heads"});

    const std::size_t grid_size  = img_size / patch_size;
    const std::size_t num_patches = grid_size * grid_size;

    Model model(engine);
    model.add_patch_embedding(img_size, patch_size, d_model);
    model.add_transformer_encoder(d_model, num_heads, d_ff, num_layers, num_patches);
    model.add_linear(d_model, MNIST_NUM_CLASSES);
    return model;
}

// ── 构造 MNIST Transformer ModelSpec ───────────────────────────────────────
[[nodiscard]] inline ModelSpec make_mnist_transformer_spec(
    std::size_t img_size   = MNIST_IMG_SIZE,
    std::size_t patch_size = MNIST_PATCH_SIZE,
    std::size_t d_model    = MNIST_TF_D_MODEL,
    std::size_t num_heads  = MNIST_TF_NUM_HEADS,
    std::size_t d_ff       = MNIST_TF_D_FF,
    std::size_t num_layers = MNIST_TF_NUM_LAYERS)
{
    (void)img_size;  // img_size 固定为 MNIST_IMG_SIZE，不写入 spec
    ModelSpec spec;
    spec.type       = ModelType::Transformer;
    spec.d_model    = d_model;
    spec.num_heads  = num_heads;
    spec.d_ff       = d_ff;
    spec.num_layers = num_layers;
    spec.patch_size = patch_size;
    // vocab_size / seq_len 不用于 Transformer (MNIST ViT)
    return spec;
}

// ── 从 ModelSpec 构建模型 ─────────────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构（支持 MLP 和 Transformer）
[[nodiscard]] inline Result<Model> build_mnist_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec)
{
    if (spec.is_mlp())
        return build_mnist_mlp_model(engine, spec.layer_dims);

    if (spec.is_transformer())
    {
        const std::size_t patch_size = spec.patch_size != 0 ? spec.patch_size : MNIST_PATCH_SIZE;
        return build_mnist_transformer_model(
            engine, MNIST_IMG_SIZE, patch_size,
            spec.d_model, spec.num_heads, spec.d_ff, spec.num_layers);
    }

    return std::unexpected(Error{
        "Invalid ModelSpec type for MNIST: expected MLP or Transformer"});
}

} // namespace nn

#endif // NN_DOMAIN_MNIST_HPP
