#ifndef NN_MODEL_SPEC_HPP
#define NN_MODEL_SPEC_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  model_spec.hpp — 模型架构描述（纯数据结构，无 L2 依赖）
//
//  分层：L3 实现层
//  职责：定义 ModelType 枚举和 ModelSpec 结构体，供 L3/L4 使用。
//  依赖：仅标准库（<cstdint>, <vector>, <string>），不依赖 L2/L1/L0。
//
//  设计理由：
//    从 model_io.hpp 中提取，使 L4 构建层无需 include model_io.hpp
//    （model_io.hpp 依赖 layer.hpp，会导致 L4 跨层依赖 L2）。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <vector>
#include <string>

namespace nn
{

// ── 模型类型枚举 ─────────────────────────────────────────────────────────
enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
};

// ── 模型架构描述 ─────────────────────────────────────────────────────────
// 嵌入到二进制文件头部，加载时可先读取规格再据此构建模型。
struct ModelSpec
{
    ModelType type = ModelType::Unknown;

    // ── MLP ──
    std::vector<std::size_t> layer_dims;

    // ── Transformer (MNIST ViT) ──
    std::size_t d_model    = 0;
    std::size_t num_heads  = 0;
    std::size_t d_ff       = 0;
    std::size_t num_layers = 0;
    std::size_t patch_size = 0;

    // ── GPT ──
    std::size_t vocab_size = 0;
    std::size_t seq_len    = 0;

    [[nodiscard]] bool is_mlp()         const noexcept { return type == ModelType::MLP; }
    [[nodiscard]] bool is_transformer() const noexcept { return type == ModelType::Transformer; }
    [[nodiscard]] bool is_gpt()         const noexcept { return type == ModelType::GPT; }
};

} // namespace nn

#endif // NN_MODEL_SPEC_HPP
