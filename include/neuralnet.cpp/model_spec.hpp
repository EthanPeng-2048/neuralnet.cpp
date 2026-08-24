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
    RoPE       = 3,  // 旋转位置编码（在注意力 Q/K 上施加，无位置嵌入）
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
    BatchNorm = 2,  // BatchNorm（沿 batch 维归一化，仅用于 MLP 类模型）
};

// ── 模型类型枚举 ─────────────────────────────────────────────────────────
enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
    ALiBi_GPT   = 4,  // 使用 ALiBi 的 GPT 模型（向后兼容）
    CNN         = 5,  // 卷积神经网络（LeNet 风格，MNIST）
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

    // ── CNN ──
    std::size_t cnn_in_channels = 0;         // 输入通道数（MNIST=1）
    std::size_t cnn_in_size     = 0;         // 输入空间尺寸（方形，MNIST=28）
    std::size_t cnn_pool        = 0;         // 每个卷积后的 MaxPool 窗口（0=无池化）
    std::vector<std::size_t> cnn_channels;   // 每个卷积层输出通道数
    std::vector<std::size_t> cnn_kernels;    // 每个卷积核大小
    std::vector<std::size_t> cnn_strides;    // 每个卷积步长（默认 1）
    std::vector<std::size_t> cnn_paddings;   // 每个卷积填充（默认 0）
    // CNN 的全连接头（layer_dims 复用）：展平 → Linear(H1) → ReLU → ... → Linear(num_classes)
    //   其中 layer_dims = {H1, H2, ..., num_classes}，首元素为展平后第一隐藏层宽度。

    [[nodiscard]] bool is_mlp()         const noexcept { return type == ModelType::MLP; }
    [[nodiscard]] bool is_transformer() const noexcept { return type == ModelType::Transformer; }
    [[nodiscard]] bool is_gpt()         const noexcept { return type == ModelType::GPT; }
    [[nodiscard]] bool is_alibi_gpt()   const noexcept { return pos_encoding == PosEncodingType::ALiBi; }
    [[nodiscard]] bool is_cnn()         const noexcept { return type == ModelType::CNN; }
};

// ── 架构一致性校验 ────────────────────────────────────────────────────────
// 判断两份 ModelSpec 是否描述同一架构。用于 load_model 时把文件头部规格
// 与模型自身（Model::spec()）做一致性校验，防止把不匹配的参数加载进模型。
//
// 兼容规则：
//   * GPT 与旧格式 ALiBi_GPT 视为同一家族（统一由 GPTModel 承载，用
//     pos_encoding 区分 Learned/Sinusoidal/ALiBi/RoPE），type 不要求严格相等。
//   * 其余模型类型要求 type 严格相等，再逐字段比较该类型的关键维度。
// 纯布尔返回，保持本头文件为纯数据结构、无 L2 依赖。
[[nodiscard]] inline bool spec_matches(const ModelSpec& a, const ModelSpec& b) noexcept
{
    // GPT 家族（GPT / ALiBi_GPT）——统一 GPTModel，比较共享字段
    auto gpt_family = [](const ModelSpec& s) { return s.is_gpt() || s.is_alibi_gpt(); };
    if (gpt_family(a) && gpt_family(b))
    {
        return a.vocab_size   == b.vocab_size &&
               a.d_model      == b.d_model &&
               a.seq_len      == b.seq_len &&
               a.num_heads    == b.num_heads &&
               a.d_ff         == b.d_ff &&
               a.num_layers   == b.num_layers &&
               a.pos_encoding == b.pos_encoding &&
               a.activation   == b.activation &&
               a.norm_type    == b.norm_type;
    }

    if (a.type != b.type)
        return false;

    switch (a.type)
    {
    case ModelType::MLP:
        return a.layer_dims == b.layer_dims;
    case ModelType::Transformer:
        return a.d_model    == b.d_model &&
               a.num_heads  == b.num_heads &&
               a.d_ff       == b.d_ff &&
               a.num_layers == b.num_layers &&
               a.patch_size == b.patch_size;
    case ModelType::CNN:
        return a.cnn_in_channels == b.cnn_in_channels &&
               a.cnn_in_size     == b.cnn_in_size &&
               a.cnn_pool        == b.cnn_pool &&
               a.cnn_channels    == b.cnn_channels &&
               a.cnn_kernels     == b.cnn_kernels &&
               a.cnn_strides     == b.cnn_strides &&
               a.cnn_paddings    == b.cnn_paddings &&
               a.layer_dims      == b.layer_dims;
    default:
        return false;
    }
}

// ── 规格摘要（诊断用） ───────────────────────────────────────────────────
// 生成简洁的人类可读描述，用于 load_model 架构不匹配时的错误信息。
[[nodiscard]] inline std::string spec_summary(const ModelSpec& s)
{
    auto type_name = [](ModelType t) -> const char* {
        switch (t)
        {
        case ModelType::MLP:         return "MLP";
        case ModelType::Transformer: return "Transformer";
        case ModelType::GPT:         return "GPT";
        case ModelType::ALiBi_GPT:   return "ALiBi_GPT";
        case ModelType::CNN:         return "CNN";
        default:                     return "Unknown";
        }
    };

    if (s.is_gpt() || s.is_alibi_gpt())
    {
        return std::string(type_name(s.type)) + "(vocab=" + std::to_string(s.vocab_size) +
               ",d_model=" + std::to_string(s.d_model) +
               ",seq_len=" + std::to_string(s.seq_len) +
               ",heads=" + std::to_string(s.num_heads) +
               ",d_ff=" + std::to_string(s.d_ff) +
               ",layers=" + std::to_string(s.num_layers) + ")";
    }
    if (s.is_cnn())
    {
        std::string c = std::to_string(s.cnn_in_channels) + "x" +
                        std::to_string(s.cnn_in_size) + "->";
        for (std::size_t i = 0; i < s.cnn_channels.size(); ++i)
        {
            if (i) c += "-";
            c += std::to_string(s.cnn_channels[i]) + "@" + std::to_string(s.cnn_kernels[i]);
        }
        return std::string(type_name(s.type)) + "(" + c + ")";
    }
    return std::string(type_name(s.type)) + "(layers=" + std::to_string(s.num_layers) + ")";
}

} // namespace nn

#endif // NN_MODEL_SPEC_HPP
