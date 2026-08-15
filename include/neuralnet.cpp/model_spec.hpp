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

// ── 位置编码类型 ─────────────────────────────────────────────────────────
enum class PosEncodingType : uint32_t
{
    Learned    = 0,  // 可学习位置嵌入（GPT 默认）
    Sinusoidal = 1,  // 正弦波固定位置编码
    ALiBi      = 2,  // 线性偏置注意力（无位置嵌入）
};

// ── FFN 激活类型 ─────────────────────────────────────────────────────────
enum class ActivationType : uint32_t
{
    GeLU   = 0,  // QuickGeLU（GPT-2 风格，默认）
    SwiGLU = 1,  // SwiGLU（LLaMA/Mistral 风格，每参数效率更高）
};

// ── 归一化层类型 ─────────────────────────────────────────────────────────
enum class NormType : uint32_t
{
    LayerNorm = 0,  // LayerNorm（GPT-2 风格，默认）
    RMSNorm   = 1,  // RMSNorm（LLaMA/Mistral 风格，更快更稳）
};

// ── 模型类型枚举 ─────────────────────────────────────────────────────────
enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
    ALiBi_GPT   = 4,  // 使用 ALiBi 的 GPT 模型（向后兼容）
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
    PosEncodingType pos_encoding = PosEncodingType::Learned;  // 位置编码类型
    ActivationType activation = ActivationType::GeLU;         // FFN 激活类型
    NormType norm_type = NormType::LayerNorm;                 // 归一化层类型

    [[nodiscard]] bool is_mlp()         const noexcept { return type == ModelType::MLP; }
    [[nodiscard]] bool is_transformer() const noexcept { return type == ModelType::Transformer; }
    [[nodiscard]] bool is_gpt()         const noexcept { return type == ModelType::GPT; }
    [[nodiscard]] bool is_alibi_gpt()   const noexcept { return pos_encoding == PosEncodingType::ALiBi; }
};

} // namespace nn

#endif // NN_MODEL_SPEC_HPP
