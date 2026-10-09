#pragma once

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
    LayerNorm = 0,  // LayerNorm（GPT-2 风格；MLP/ViT 默认）
    RMSNorm   = 1,  // RMSNorm（LLaMA/Mistral 风格，更快更稳）
    BatchNorm = 2,  // BatchNorm（沿 batch 维归一化，CNN 默认）
};

// ── 归一化挂载位置（CNN / MNIST Transformer 专用）───────────────────────
// 归一化**类型**由 NormType 描述；本枚举描述“加不加、加在哪”。
//
// **各架构默认（builder 默认实参，2026-10-03 起）**：CNN = BatchNorm@Conv、
// ViT = LayerNorm@Final、MLP/GPT/RAPT = None（它们的 norm 由自身结构决定）。
// **旧模型文件没有 norm_place 键 → 读回 None = 不加额外归一化**——
// 缺键**不**回落架构默认，与旧实现参数布局逐位一致（旧 checkpoint 仍可加载）。
//
//   None  = 不加（MLP/GPT/RAPT 恒为 None：它们的 norm 挂载位置由自身
//           结构决定 —— MLP = Linear→Norm→激活，GPT/RAPT = pre-norm）
//   Conv  = CNN：每个卷积（+池化）之后，按 (C*H*W, batch) 每样本归一化
//   Head  = CNN：全连接头隐藏层（Linear → Norm → ReLU，与 MLP 同款）
//   Both  = CNN：Conv + Head
//   Final = ViT：编码器末端 final norm（全局平均池化之前，原版 ViT 的 ln_f）
enum class NormPlace : uint32_t
{
    None  = 0,
    Conv  = 1,
    Head  = 2,
    Both  = 3,
    Final = 4,
};

// NormPlace → CLI/日志用短名（归一化位置的唯一命名口径）
[[nodiscard]] inline const char* norm_place_name(NormPlace p) noexcept
{
    switch (p)
    {
    case NormPlace::Conv:  return "conv";
    case NormPlace::Head:  return "head";
    case NormPlace::Both:  return "both";
    case NormPlace::Final: return "final";
    default:               return "none";
    }
}

// ── 权重三值量化（docs/development/21-quantized-weights.md §4.7）──────────
// 描述"这个模型的线性层是不是 BitLinear（三值权重 1.58-bit + STE）"。
//   None  = 普通 Linear（f32/f16 权重）
//   T1_58 = BitLinear（param=T1_58；latent 仍 f32，量化缓冲是派生物、不落盘）
//
// **缺键 → None**：旧模型文件（没有 weight_quant 键）读回 None，
// 参数条数/顺序不变 → 无需升 MODEL_VERSION，与 norm_place 的缺键处理同款。
enum class WeightQuant : uint32_t
{
    None  = 0,
    T1_58 = 1,
};

// WeightQuant → CLI/日志/序列化用短名（唯一命名口径；与 Precision::T1_58 同词法）
[[nodiscard]] inline const char* weight_quant_name(WeightQuant q) noexcept
{
    return q == WeightQuant::T1_58 ? "t1_58" : "none";
}

// ── 模型类型枚举 ─────────────────────────────────────────────────────────
enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
    ALiBi_GPT   = 4,  // 使用 ALiBi 的 GPT 模型（向后兼容）
    CNN         = 5,  // 卷积神经网络（LeNet 风格，MNIST）
    // 6 = AttnZip / ZiPT —— **已于 2026-10-01 整体移除**（代码保留在 `legacy/zipt`
    //     分支，恢复前提见 docs/history.md「ZiPT 移除」条）。此枚举值仅作占位：
    //     加载旧模型文件（type=6）时据此给出明确错误，而不是含糊的 "Unknown"。
    Reserved_ZiPT = 6,
    RAPT        = 7,  // ReLU 激活线性注意力（ReLU-Linear Attention，causal LM）
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
    NormPlace norm_place = NormPlace::None;                   // 归一化挂载位置（CNN/ViT）

    // ── 线性层权重是否三值（T1_58；P1 覆盖 MLP 路径）──
    // 缺键 → None（旧文件零破坏）。见 WeightQuant 注释与 docs 21 §4.7。
    WeightQuant weight_quant = WeightQuant::None;

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
    [[nodiscard]] bool is_rapt()        const noexcept { return type == ModelType::RAPT; }
};

// ── 架构一致性校验 ────────────────────────────────────────────────────────
// 判断两份 ModelSpec 是否描述同一架构。用于 load_model 时把文件头部规格
// 与模型自身（Model::spec()）做一致性校验，防止把不匹配的参数加载进模型。
//
// 兼容规则：
//   * ALiBi_GPT 与 GPT 同族，统一由 GPTModel 承载，用
//     pos_encoding 区分 Learned/Sinusoidal/ALiBi/RoPE，type 不要求严格相等。
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
               a.norm_type    == b.norm_type &&
               a.weight_quant == b.weight_quant;
    }

    // RAPT：ReLU-Linear Attention（causal LM），共享 GPT 类似字段
    if (a.type == ModelType::RAPT && b.type == ModelType::RAPT)
    {
        return a.vocab_size   == b.vocab_size &&
               a.d_model      == b.d_model &&
               a.seq_len      == b.seq_len &&
               a.num_heads    == b.num_heads &&
               a.d_ff         == b.d_ff &&
               a.num_layers   == b.num_layers &&
               a.pos_encoding == b.pos_encoding &&
               a.activation   == b.activation &&
               a.norm_type    == b.norm_type &&
               a.weight_quant == b.weight_quant;
    }

    if (a.type != b.type)
        return false;

    switch (a.type)
    {
    case ModelType::MLP:
        // norm_type 影响参数个数（RMSNorm 无 beta、BatchNorm 另有 running 状态）
        // → 属关键维度，加载错配必须给出明确错误而非矩阵形状错位
        // weight_quant 同理：参数**形状**相同（latent 恒 f32），但层语义不同
        //（Linear vs BitLinear）→ 不比对就等于允许"把三值模型权重加载进普通 MLP"。
        return a.layer_dims == b.layer_dims &&
               a.norm_type  == b.norm_type &&
               a.weight_quant == b.weight_quant;
    case ModelType::Transformer:
        return a.d_model    == b.d_model &&
               a.num_heads  == b.num_heads &&
               a.d_ff       == b.d_ff &&
               a.num_layers == b.num_layers &&
               a.patch_size == b.patch_size &&
               a.norm_type  == b.norm_type &&
               a.norm_place == b.norm_place &&
               a.weight_quant == b.weight_quant;
    case ModelType::CNN:
        return a.cnn_in_channels == b.cnn_in_channels &&
               a.cnn_in_size     == b.cnn_in_size &&
               a.cnn_pool        == b.cnn_pool &&
               a.cnn_channels    == b.cnn_channels &&
               a.cnn_kernels     == b.cnn_kernels &&
               a.cnn_strides     == b.cnn_strides &&
               a.cnn_paddings    == b.cnn_paddings &&
               a.layer_dims      == b.layer_dims &&
               a.norm_type       == b.norm_type &&
               a.norm_place      == b.norm_place &&
               a.weight_quant    == b.weight_quant;
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
        case ModelType::Reserved_ZiPT: return "ZiPT(removed)";
        case ModelType::RAPT:        return "RAPT";
        default:                     return "Unknown";
        }
    };

    if (s.is_rapt())
    {
        return std::string(type_name(s.type)) + "(vocab=" + std::to_string(s.vocab_size) +
               ",d_model=" + std::to_string(s.d_model) +
               ",seq_len=" + std::to_string(s.seq_len) +
               ",heads=" + std::to_string(s.num_heads) +
               ",d_ff=" + std::to_string(s.d_ff) +
               ",layers=" + std::to_string(s.num_layers) +
               ",pos_enc=" + std::to_string(static_cast<unsigned>(s.pos_encoding)) + ")";
    }

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
    std::string out = std::string(type_name(s.type)) + "(layers="
                    + std::to_string(s.num_layers);
    // 三值权重是**架构级**差异（层类型 Linear → BitLinear），摘要必须显式披露
    if (s.weight_quant != WeightQuant::None)
        out += ",wq=" + std::string(weight_quant_name(s.weight_quant));
    return out + ")";
}

} // namespace nn
