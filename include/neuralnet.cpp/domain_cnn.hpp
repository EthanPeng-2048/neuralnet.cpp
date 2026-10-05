#pragma once

// ── domain_cnn.hpp — CNN 领域构建层（引擎化架构） ──────────────────────────
//
// 依赖：Model + ComputeEngine + ModelSpec（model_container.hpp / compute_engine.hpp）
//
// 结构（LeNet 风格，batch-major 列布局 (C*H*W, batch)）：
//   Conv2D → [MaxPool2D] → [Norm] × N  → 展平 (C*H*W, batch)
//   → Linear(H1) → [Norm] → ReLU → ... → Linear(num_classes)
// 方括号项可选：由 CnnConfig::norm_place（NormPlace）控制，
// 默认 BatchNorm@Conv（现代 LeNet 形态；显式置 None = 无 norm 的旧结构）。
//
// 布局约定：卷积/池化层输入输出均为 (C*H*W, batch) 列布局（与项目一致），
//   展平后直接喂给 Linear，无需独立 Flatten 层。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <string>
#include <vector>

#include "core_errors.hpp"
#include "compute_engine.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"

namespace nn
{

// ── 单个卷积层规格 ────────────────────────────────────────────────────────
struct CnnConvSpec
{
    std::size_t out_channels;  // 输出通道数
    std::size_t kernel;        // 核大小（方形）
    std::size_t stride = 1;    // 步长
    std::size_t padding = 0;   // 填充
};

// ── CNN 配置结构体 ────────────────────────────────────────────────────────
// 将 build_cnn_model 的多个位置参数收拢为一个结构体，避免调用方签名过长。
struct CnnConfig
{
    std::size_t in_channels = 1;          // 输入通道数（MNIST=1）
    std::size_t in_size     = 28;         // 输入空间尺寸（方形）
    std::size_t pool        = 2;          // 每个卷积后的 MaxPool 窗口（0=无池化）
    std::vector<CnnConvSpec> convs;       // 卷积层列表
    // 展平后的全连接头（首元素 = 展平后第一隐藏层宽度，末位 = 类别数）。
    // 例如 {120, 10} → Linear(flatten→120) + ReLU + Linear(120→10)。
    std::vector<std::size_t> fc_dims;
    // ── 归一化（默认 CNN 规范形态 = BatchNorm@Conv；显式置 None 可关）──────
    // norm_type 决定归一化层种类，norm_place 决定挂在哪（见 model_spec.hpp
    // 的 NormPlace 注释）：Conv = 每个卷积（+池化）后按 (C*H*W, batch) 归一化
    // （BatchNorm 下即“逐特征跨 batch”的经典 Conv→BN 语义）；Head = 全连接头
    // 隐藏层 Linear→Norm→ReLU（MLP 同款）。默认值由 4 epoch A/B 实测选定。
    // 旧模型文件缺 norm_place 键 → spec 读回 None → 不加（旧布局逐位一致）。
    NormType  norm_type  = NormType::BatchNorm;
    NormPlace norm_place = NormPlace::Conv;
};

// 默认 MNIST CNN（LeNet-5 风格）：
//   Conv(1→6, k5) + Pool2 → Conv(6→16, k5) + Pool2 → flatten(16*4*4=256)
//   → Linear(256,120) → ReLU → Linear(120,10)
inline const std::vector<CnnConvSpec> MNIST_CNN_CONVS = {
    {6, 5},   // out_channels=6, kernel=5
    {16, 5},  // out_channels=16, kernel=5
};
inline const std::vector<std::size_t> MNIST_CNN_FC = {120, 10};

// ── 卷积输出尺寸（不含池化） ─────────────────────────────────────────────
[[nodiscard]] inline std::size_t conv_out_size(
    std::size_t in, std::size_t k, std::size_t s, std::size_t p)
{
    return (in + 2 * p - k) / s + 1;
}

// ── 构建 CNN 模型（CnnConfig 版本，推荐使用） ────────────────────────────
[[nodiscard]] inline Result<Model> build_cnn_model(
    ComputeEngine& engine, const CnnConfig& cfg)
{
    if (cfg.convs.empty())
        NN_FAIL("CNN: convs must not be empty");
    if (cfg.fc_dims.size() < 2)
        NN_FAIL("CNN: fc_dims must have at least 2 elements");
    if (cfg.in_channels == 0 || cfg.in_size == 0)
        NN_FAIL("CNN: in_channels/in_size must be positive");
    if (cfg.norm_place != NormPlace::None && cfg.norm_place != NormPlace::Conv &&
        cfg.norm_place != NormPlace::Head && cfg.norm_place != NormPlace::Both)
        NN_FAIL("CNN: norm_place must be none/conv/head/both (final is ViT-only)");
    Model model(engine);
    std::size_t c = cfg.in_channels;
    std::size_t h = cfg.in_size;
    std::size_t w = cfg.in_size;

    for (std::size_t i = 0; i < cfg.convs.size(); ++i)
    {
        const CnnConvSpec& cv = cfg.convs[i];
        if (cv.out_channels == 0 || cv.kernel == 0)
            NN_FAIL("CNN: conv out_channels/kernel must be positive");
        if (cv.kernel > h + 2 * cv.padding || cv.kernel > w + 2 * cv.padding)
            NN_FAIL("CNN: kernel larger than input spatial size");
        {
            auto r = model.add<Conv2D>(c, cv.out_channels, cv.kernel,
                                       cv.stride, cv.padding, h, w);
            NN_TRY_CHECK(r);
        }
        h = conv_out_size(h, cv.kernel, cv.stride, cv.padding);
        w = conv_out_size(w, cv.kernel, cv.stride, cv.padding);
        c = cv.out_channels;

        if (cfg.pool > 0)
        {
            auto r = model.add<MaxPool2D>(c, h, w, cfg.pool);
            NN_TRY_CHECK(r);
            h = (h - cfg.pool) / cfg.pool + 1;
            w = (w - cfg.pool) / cfg.pool + 1;
        }

        // 归一化（NormPlace::Conv / Both）：池化之后、下一组卷积之前，
        // 按展平后的特征图 (C*H*W, batch) 每样本归一化（与布局约定一致）。
        if (cfg.norm_place == NormPlace::Conv || cfg.norm_place == NormPlace::Both)
        {
            auto r = model.add_layer(make_norm_layer(c * h * w, cfg.norm_type));
            NN_TRY_CHECK(r);
        }
    }

    const std::size_t flattened = c * h * w;
    if (flattened == 0)
        NN_FAIL("CNN: flattened size is zero");
    // 全连接头：Linear(flatten → fc_dims[0]) → [Norm] → ReLU → ...
    // → Linear(→ num_classes)；Norm 只挂隐藏层，最后一层保持原始 logits。
    const bool norm_head = (cfg.norm_place == NormPlace::Head ||
                            cfg.norm_place == NormPlace::Both);
    {
        auto r = model.add<Linear>(flattened, cfg.fc_dims[0]);
        NN_TRY_CHECK(r);
        if (norm_head)
        {
            auto rn = model.add_layer(make_norm_layer(cfg.fc_dims[0], cfg.norm_type));
            NN_TRY_CHECK(rn);
        }
    }
    for (std::size_t i = 1; i < cfg.fc_dims.size(); ++i)
    {
        {
            auto r = model.add<ReLU>();
            NN_TRY_CHECK(r);
        }
        {
            auto r = model.add<Linear>(cfg.fc_dims[i - 1], cfg.fc_dims[i]);
            NN_TRY_CHECK(r);
        }
        // 隐藏层后再挂一份 Norm（最后一层 fc_dims.size()-1 不挂）
        if (norm_head && i + 1 < cfg.fc_dims.size())
        {
            auto rn = model.add_layer(make_norm_layer(cfg.fc_dims[i], cfg.norm_type));
            NN_TRY_CHECK(rn);
        }
    }
    return model;
}

// ── 构造 CNN ModelSpec ────────────────────────────────────────────────────
// fc_dims 复用 layer_dims 字段（type 区分，与 MLP 互斥）。
[[nodiscard]] inline ModelSpec make_cnn_spec(
    std::size_t in_channels,
    std::size_t in_size,
    std::size_t pool,
    const std::vector<CnnConvSpec>& convs,
    const std::vector<std::size_t>& fc_dims,
    NormType norm_type = NormType::BatchNorm,
    NormPlace norm_place = NormPlace::Conv)
{
    ModelSpec spec;
    spec.type            = ModelType::CNN;
    spec.norm_type       = norm_type;
    spec.norm_place      = norm_place;
    spec.cnn_in_channels = in_channels;
    spec.cnn_in_size     = in_size;
    spec.cnn_pool        = pool;
    spec.layer_dims      = fc_dims;  // 复用为 CNN 全连接头
    spec.cnn_channels.clear();
    spec.cnn_kernels.clear();
    spec.cnn_strides.clear();
    spec.cnn_paddings.clear();
    for (const auto& cv : convs)
    {
        spec.cnn_channels.push_back(cv.out_channels);
        spec.cnn_kernels.push_back(cv.kernel);
        spec.cnn_strides.push_back(cv.stride);
        spec.cnn_paddings.push_back(cv.padding);
    }
    return spec;
}

// ── 从 ModelSpec 还原 CnnConfig ──────────────────────────────────────────
[[nodiscard]] inline Result<CnnConfig> cnn_config_from_spec(const ModelSpec& spec)
{
    if (!spec.is_cnn())
        NN_FAIL("cnn_config_from_spec: not a CNN spec");
    if (spec.cnn_channels.size() != spec.cnn_kernels.size() ||
        spec.cnn_channels.size() != spec.cnn_strides.size() ||
        spec.cnn_channels.size() != spec.cnn_paddings.size())
        NN_FAIL("CNN spec: conv vectors length mismatch");
    if (spec.cnn_channels.empty())
        NN_FAIL("CNN spec: no conv layers");
    CnnConfig cfg;
    cfg.in_channels = spec.cnn_in_channels;
    cfg.in_size     = spec.cnn_in_size;
    cfg.pool        = spec.cnn_pool;
    cfg.fc_dims     = spec.layer_dims;
    cfg.norm_type   = spec.norm_type;
    cfg.norm_place  = spec.norm_place;
    if (cfg.norm_place != NormPlace::None && cfg.norm_place != NormPlace::Conv &&
        cfg.norm_place != NormPlace::Head && cfg.norm_place != NormPlace::Both)
        NN_FAIL("CNN spec: norm_place must be none/conv/head/both (final is ViT-only)");
    for (std::size_t i = 0; i < spec.cnn_channels.size(); ++i)
    {
        cfg.convs.push_back(CnnConvSpec{
            spec.cnn_channels[i],
            spec.cnn_kernels[i],
            spec.cnn_strides[i] != 0 ? spec.cnn_strides[i] : 1,
            spec.cnn_paddings[i],
        });
    }
    return cfg;
}

// ── 从 ModelSpec 构建 CNN 模型（用于从二进制文件加载时自动还原） ─────────
[[nodiscard]] inline Result<Model> build_cnn_model_from_spec(
    ComputeEngine& engine, const ModelSpec& spec)
{
    auto cfg_r = cnn_config_from_spec(spec);
    NN_TRY_CHECK(cfg_r);
    auto model = build_cnn_model(engine, *cfg_r);
    if (model)
        model->set_spec(spec);  // 记录架构规格，供 load_model 校验
    return model;
}

} // namespace nn

