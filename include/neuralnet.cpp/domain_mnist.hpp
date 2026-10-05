#pragma once

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
#include "domain_cnn.hpp"

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
            NN_FAIL("CSV 含无效数字 '" + token + "': " + v.error().message);
        values.push_back(*v);
    }
    return values;
}

// ── 从 CSV 行加载单张 MNIST 图片 ───────────────────────────────────────────
// 输入：784 个逗号分隔的像素值（0~255 或归一化后的 0~1）
// 输出：(784, 1) 列向量 Matrix
// 底层解析实现；上层代码应优先使用 load_image_tensor_from_csv_line
// （上层统一用 Tensor）。
[[nodiscard]] inline Result<Matrix>
load_image_from_csv_line(const std::string &csv_line)
{
    auto values = parse_csv_line(csv_line);
    NN_TRY_CHECK(values);
    if (values->size() != MNIST_INPUT_DIM)
        NN_FAIL("CSV 必须包含恰好 " + std::to_string(MNIST_INPUT_DIM) +
                                    " 个值，实际: " + std::to_string(values->size()));
    Matrix img(MNIST_INPUT_DIM, 1);
    for (std::size_t i = 0; i < MNIST_INPUT_DIM; ++i)
        img.set_value_unchecked(i, 0, (*values)[i]);
    return img;
}

// ── 从 CSV 行加载单张 MNIST 图片为 Tensor ────────────────────────────────
// 上层统一使用 Tensor：内部调用 load_image_from_csv_line 得到 Matrix 后，
// 通过 engine.from_matrix 上传至对应设备。
[[nodiscard]] inline Result<Tensor>
load_image_tensor_from_csv_line(const std::string &csv_line, ComputeEngine& engine)
{
    auto mat_r = load_image_from_csv_line(csv_line);
    NN_TRY_CHECK(mat_r);
    return engine.from_matrix(*mat_r);
}

// 默认 MLP 网络架构：输入层 → 隐藏层1 → 隐藏层2 → 隐藏层3 → 输出层
inline const std::vector<std::size_t> MNIST_LAYER_DIMS = {
    MNIST_INPUT_DIM, 512, 256, 128, 64, MNIST_NUM_CLASSES
};

// ── 构建 MLP 模型 ──────────────────────────────────────────────────────────
// 通过指定 ComputeEngine 创建同设备的权重张量；可指定自定义层维度。
// 结构：Linear → Norm → GeLU × (N-1) + Linear（最后一层）
// norm_type 决定归一化层：LayerNorm / RMSNorm / BatchNorm。
[[nodiscard]] inline Result<Model> build_mnist_mlp_model(
    ComputeEngine& engine,
    const std::vector<std::size_t> &layer_dims = MNIST_LAYER_DIMS,
    NormType norm_type = NormType::LayerNorm,
    PrecisionProfile precision = PrecisionProfile{})
{
    if (layer_dims.size() < 2)
        NN_FAIL("MLP layer_dims must have at least 2 elements");
    Model model(engine);
    model.set_default_precision_profile(precision);   // 须在 add 之前：权重按精度创建
    for (std::size_t i = 0; i < layer_dims.size() - 1; ++i)
    {
        std::size_t in_dim  = layer_dims[i];
        std::size_t out_dim = layer_dims[i + 1];

        {
            auto r = model.add<Linear>(in_dim, out_dim);
            NN_TRY_CHECK(r);
        }

        if (i < layer_dims.size() - 2)
        {
            {
                auto r = model.add_layer(make_norm_layer(out_dim, norm_type));
                NN_TRY_CHECK(r);
            }
            {
                auto r = model.add<GeLU>();
                NN_TRY_CHECK(r);
            }
        }
    }
    return model;
}

// ── 构建 MNIST Transformer (ViT 风格) 模型 ──────────────────────────────────
// 结构：PatchEmbedding → TransformerEncoder → Linear(分类头)
//   PatchEmbedding: img 28×28 → 16 个 7×7 patch → Linear(49, d_model) 投影
//   TransformerEncoder: PE + N×EncoderLayer（块内 pre-norm，类型 = norm_type）
//     + final norm（默认开，NormPlace::Final = 池化前的 ln_f）+ 全局平均池化
//   Linear: d_model → 10（分类头）
// norm_place：Final（默认，原版 ViT 的 ln_f）/ None（显式关闭 = 旧结构）；
//   conv/head/both 属 CNN。旧模型文件缺 norm_place 键 → None → 不加。
[[nodiscard]] inline Result<Model> build_mnist_transformer_model(
    ComputeEngine& engine,
    std::size_t img_size   = MNIST_IMG_SIZE,
    std::size_t patch_size = MNIST_PATCH_SIZE,
    std::size_t d_model    = MNIST_TF_D_MODEL,
    std::size_t num_heads  = MNIST_TF_NUM_HEADS,
    std::size_t d_ff       = MNIST_TF_D_FF,
    std::size_t num_layers = MNIST_TF_NUM_LAYERS,
    PrecisionProfile precision = PrecisionProfile{},
    NormType norm_type   = NormType::LayerNorm,
    NormPlace norm_place = NormPlace::Final)
{
    if (img_size % patch_size != 0)
        NN_FAIL("MNIST Transformer: img_size must be divisible by patch_size");
    if (d_model == 0 || num_heads == 0 || d_ff == 0 || num_layers == 0)
        NN_FAIL("MNIST Transformer: parameters must be positive");
    if (d_model % num_heads != 0)
        NN_FAIL("MNIST Transformer: d_model must be divisible by num_heads");
    if (norm_place != NormPlace::None && norm_place != NormPlace::Final)
        NN_FAIL("MNIST Transformer: norm_place must be none/final (conv/head/both are CNN-only)");
    const std::size_t grid_size  = img_size / patch_size;
    const std::size_t num_patches = grid_size * grid_size;

    Model model(engine);
    model.set_default_precision_profile(precision);   // 须在 add 之前：权重按精度创建
    {
        auto r = model.add<PatchEmbedding>(img_size, patch_size, d_model);
        NN_TRY_CHECK(r);
    }
    {
        auto r = model.add<TransformerEncoder>(d_model, num_heads, d_ff, num_layers,
                                               num_patches, norm_type,
                                               norm_place == NormPlace::Final);
        NN_TRY_CHECK(r);
    }
    {
        auto r = model.add<Linear>(d_model, MNIST_NUM_CLASSES);
        NN_TRY_CHECK(r);
    }
    return model;
}

// ── 构造 MNIST Transformer ModelSpec ───────────────────────────────────────
// img_size 固定为 MNIST_IMG_SIZE（28×28），故不入参、不写入 spec。
// norm_type/norm_place 描述归一化：类型（pre-norm 槽位）与 final norm 开关；
// 默认 Final（原版 ViT = 块 pre-norm + 末端 ln_f，4 epoch A/B 实测优于无 final）。
// 旧模型文件缺 norm_place 键 → spec 读回 None → 不加（旧布局逐位一致）。
[[nodiscard]] inline ModelSpec make_mnist_transformer_spec(
    std::size_t patch_size = MNIST_PATCH_SIZE,
    std::size_t d_model    = MNIST_TF_D_MODEL,
    std::size_t num_heads  = MNIST_TF_NUM_HEADS,
    std::size_t d_ff       = MNIST_TF_D_FF,
    std::size_t num_layers = MNIST_TF_NUM_LAYERS,
    NormType norm_type   = NormType::LayerNorm,
    NormPlace norm_place = NormPlace::Final)
{
    ModelSpec spec;
    spec.type       = ModelType::Transformer;
    spec.d_model    = d_model;
    spec.num_heads  = num_heads;
    spec.d_ff       = d_ff;
    spec.num_layers = num_layers;
    spec.patch_size = patch_size;
    spec.norm_type  = norm_type;
    spec.norm_place = norm_place;
    // vocab_size / seq_len 不用于 Transformer (MNIST ViT)
    return spec;
}

// ── 从 ModelSpec 构建模型 ─────────────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构（支持 MLP 和 Transformer）
[[nodiscard]] inline Result<Model> build_mnist_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec,
    PrecisionProfile precision = PrecisionProfile{})
{
    if (spec.is_mlp())
    {
        auto model = build_mnist_mlp_model(engine, spec.layer_dims, spec.norm_type,
                                           precision);
        if (model)
            model->set_spec(spec);  // 记录架构规格，供 load_model 校验
        return model;
    }

    if (spec.is_transformer())
    {
        const std::size_t patch_size = spec.patch_size != 0 ? spec.patch_size : MNIST_PATCH_SIZE;
        auto model = build_mnist_transformer_model(
            engine, MNIST_IMG_SIZE, patch_size,
            spec.d_model, spec.num_heads, spec.d_ff, spec.num_layers, precision,
            spec.norm_type, spec.norm_place);
        if (model)
            model->set_spec(spec);  // 记录架构规格，供 load_model 校验
        return model;
    }

    if (spec.is_cnn())
    {
        auto model = build_cnn_model_from_spec(engine, spec);
        if (model)
            model->set_spec(spec);  // 记录架构规格，供 load_model 校验
        return model;
    }

    NN_FAIL("Invalid ModelSpec type for MNIST: expected MLP, Transformer, or CNN");
}

} // namespace nn

