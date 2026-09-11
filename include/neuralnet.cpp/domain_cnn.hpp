#pragma once

// ── domain_cnn.hpp — CNN 领域构建层（引擎化架构） ──────────────────────────
//
// 依赖：Model + ComputeEngine + ModelSpec（model_container.hpp / compute_engine.hpp）
//
// 结构（LeNet 风格，batch-major 列布局 (C*H*W, batch)）：
//   Conv2D → [MaxPool2D] × N  → 展平 (C*H*W, batch)
//   → Linear(H1) → ReLU → ... → Linear(num_classes)
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
        return std::unexpected(Error{"CNN: convs must not be empty"});
    if (cfg.fc_dims.size() < 2)
        return std::unexpected(Error{"CNN: fc_dims must have at least 2 elements"});
    if (cfg.in_channels == 0 || cfg.in_size == 0)
        return std::unexpected(Error{"CNN: in_channels/in_size must be positive"});

    Model model(engine);
    std::size_t c = cfg.in_channels;
    std::size_t h = cfg.in_size;
    std::size_t w = cfg.in_size;

    for (std::size_t i = 0; i < cfg.convs.size(); ++i)
    {
        const CnnConvSpec& cv = cfg.convs[i];
        if (cv.out_channels == 0 || cv.kernel == 0)
            return std::unexpected(Error{"CNN: conv out_channels/kernel must be positive"});
        if (cv.kernel > h + 2 * cv.padding || cv.kernel > w + 2 * cv.padding)
            return std::unexpected(Error{"CNN: kernel larger than input spatial size"});

        {
            auto r = model.add<Conv2D>(c, cv.out_channels, cv.kernel,
                                       cv.stride, cv.padding, h, w);
            if (!r) return std::unexpected(r.error());
        }
        h = conv_out_size(h, cv.kernel, cv.stride, cv.padding);
        w = conv_out_size(w, cv.kernel, cv.stride, cv.padding);
        c = cv.out_channels;

        if (cfg.pool > 0)
        {
            auto r = model.add<MaxPool2D>(c, h, w, cfg.pool);
            if (!r) return std::unexpected(r.error());
            h = (h - cfg.pool) / cfg.pool + 1;
            w = (w - cfg.pool) / cfg.pool + 1;
        }
    }

    const std::size_t flattened = c * h * w;
    if (flattened == 0)
        return std::unexpected(Error{"CNN: flattened size is zero"});

    // 全连接头：Linear(flatten → fc_dims[0]) → ReLU → ... → Linear(→ num_classes)
    {
        auto r = model.add<Linear>(flattened, cfg.fc_dims[0]);
        if (!r) return std::unexpected(r.error());
    }
    for (std::size_t i = 1; i < cfg.fc_dims.size(); ++i)
    {
        {
            auto r = model.add<ReLU>();
            if (!r) return std::unexpected(r.error());
        }
        {
            auto r = model.add<Linear>(cfg.fc_dims[i - 1], cfg.fc_dims[i]);
            if (!r) return std::unexpected(r.error());
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
    const std::vector<std::size_t>& fc_dims)
{
    ModelSpec spec;
    spec.type            = ModelType::CNN;
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
        return std::unexpected(Error{"cnn_config_from_spec: not a CNN spec"});
    if (spec.cnn_channels.size() != spec.cnn_kernels.size() ||
        spec.cnn_channels.size() != spec.cnn_strides.size() ||
        spec.cnn_channels.size() != spec.cnn_paddings.size())
        return std::unexpected(Error{"CNN spec: conv vectors length mismatch"});
    if (spec.cnn_channels.empty())
        return std::unexpected(Error{"CNN spec: no conv layers"});

    CnnConfig cfg;
    cfg.in_channels = spec.cnn_in_channels;
    cfg.in_size     = spec.cnn_in_size;
    cfg.pool        = spec.cnn_pool;
    cfg.fc_dims     = spec.layer_dims;
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
    if (!cfg_r) return std::unexpected(std::move(cfg_r).error());
    auto model = build_cnn_model(engine, *cfg_r);
    if (model)
        model->set_spec(spec);  // 记录架构规格，供 load_model 校验
    return model;
}

} // namespace nn

