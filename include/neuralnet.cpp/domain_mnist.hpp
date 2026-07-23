#ifndef NN_DOMAIN_MNIST_HPP
#define NN_DOMAIN_MNIST_HPP

// ── domain_mnist.hpp — MNIST 领域构建层 ────────────────────────────────────
//
// 仅保留 MLP 构建（基于新引擎化架构）。
// Transformer / GPT 等其他模型类型已移除（新架构仅提供 Linear/ReLU/GeLU/
// LayerNorm 原语层，不支持 PatchEmbedding/TransformerEncoder/GPTModel）。
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

// ── 从 ModelSpec 构建模型 ─────────────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构（仅支持 MLP）
[[nodiscard]] inline Result<Model> build_mnist_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec)
{
    if (spec.is_mlp())
        return build_mnist_mlp_model(engine, spec.layer_dims);

    return std::unexpected(Error{
        "Invalid ModelSpec type for MNIST: expected MLP "
        "(Transformer/GPT not supported by engine-based architecture)"});
}

} // namespace nn

#endif // NN_DOMAIN_MNIST_HPP
