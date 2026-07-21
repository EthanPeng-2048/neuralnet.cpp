#ifndef NN_DOMAIN_MNIST_HPP
#define NN_DOMAIN_MNIST_HPP

#include <cstddef>
#include <string>
#include <expected>
#include <sstream>
#include <vector>

#include "core_errors.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"

namespace nn {

// ── MNIST 常量 ──────────────────────────────────────────────────────────────
inline constexpr std::size_t MNIST_INPUT_DIM = 784;
inline constexpr std::size_t MNIST_NUM_CLASSES = 10;

// ── 共享 CSV 行解析工具 ────────────────────────────────────────────────────
// 解析单行 CSV（逗号分隔的浮点数）为 std::vector<Scalar>
// 异常安全：使用 nn::parse_number（基于 std::from_chars），不抛异常。
// mnist_train 与 mnist_infer 共用此工具，消除重复实现。
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

    // ── 参数校验 ───────────────────────────────────────────────────────
    if (patch_size == 0)
        return std::unexpected(Error{"patch_size must be > 0"});
    if (img_size % patch_size != 0)
        return std::unexpected(Error{"img_size (28) must be divisible by patch_size (" +
                                    std::to_string(patch_size) + ")"});
    if (d_model == 0)
        return std::unexpected(Error{"d_model must be > 0"});
    if (num_heads == 0)
        return std::unexpected(Error{"num_heads must be > 0"});
    if (d_model % num_heads != 0)
        return std::unexpected(Error{"d_model (" + std::to_string(d_model) +
                                    ") must be divisible by num_heads (" +
                                    std::to_string(num_heads) + ")"});
    if (d_ff == 0)
        return std::unexpected(Error{"d_ff must be > 0"});
    if (num_layers == 0)
        return std::unexpected(Error{"num_layers must be > 0"});

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

#endif // NN_DOMAIN_MNIST_HPP