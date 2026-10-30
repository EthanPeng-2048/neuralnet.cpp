#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  model_spec.hpp — 模型架构描述（纯数据结构，无 L2 依赖）
//
//  分层：L3 实现层（纯数据 + 类型化规格；不依赖 L2/L1/L0）
//  职责：定义类型化规格子类（`ModelSpecBase` 派生）与公共外壳 `ModelSpec`，
//        供 L3/L4 使用。
//
//  扩展模型（RFC 32 §6，issue #26）：`ModelSpec` 只持**域无关公共字段**
//  （多输出 schema/packing）+ `std::unique_ptr<ModelSpecBase> arch`；类型参数
//  与判别符（`type_name()`）随多态对象走。加新模型 = 写一个 `ModelSpecBase`
//  子类（+ 可选在 `ModelRegistry` 注册类型名），**不改本头的任何 switch**。
//  判别/比较/摘要/编解码四个分发点全部虚化：
//    · `spec_matches` → `arch_compatible`；`spec_summary` → `summary`；
//    · 序列化 `spec_to_kv/spec_from_kv` → `write_kv/read_kv`（在
//      model_serialization.hpp）；类型名工厂 → `make_spec_by_name`。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_errors.hpp>      // nn::Result（read_kv）
#include <neuralnet.cpp/model_keyvalue_record.hpp>  // nn::KeyValueRecord（write_kv/read_kv）

#include "tensor_schema.hpp"

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
    // 门控平方 ReLU（BitNet b1.58 2B4T 的 `hidden_act = "relu2"`）：
    //   out = relu²(gate(x)) ⊙ up(x) —— 与 SwiGLU 同接线，只换逐元素函数。
    ReLU2  = 2,
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
// **各架构默认（builder 默认实参）**：CNN = BatchNorm@Conv、
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

// ── 权重三值量化（docs/internal/21-quantized-weights.md §4.7）──────────
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

// ── 旧文件数值 type（仅作向下回落识别；规格判别符 = ModelSpecBase::type_name）──
// 规格对象不再持有本枚举：新文件的 `type` 键写**类型名字符串**（如 "GPT"），
// 读到数值（旧文件）时经 `legacy_spec_type_name` 还原为类型名再构造对应子类。
enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
    ALiBi_GPT   = 4,  // 使用 ALiBi 的 GPT 模型（向后兼容）
    CNN         = 5,  // 卷积神经网络（LeNet 风格，MNIST）
    // 6 = AttnZip / ZiPT —— **已整体移除**（代码保留在 `legacy/zipt`
    //     分支，恢复前提见 docs/history/history.md「ZiPT 移除」条）。此枚举值仅作占位：
    //     加载旧模型文件（type=6）时据此给出明确错误，而不是含糊的 "Unknown"。
    Reserved_ZiPT = 6,
    RAPT        = 7,  // ReLU 激活线性注意力（ReLU-Linear Attention，causal LM）
};

// 前置声明（`summary` 需要读外壳的公共字段）
struct ModelSpec;

// ═══════════════════════════════════════════════════════════════════════
//  ModelSpecBase：类型化规格的虚基类（判别符 + 类型参数 + 分发虚方法）
// ═══════════════════════════════════════════════════════════════════════
class ModelSpecBase
{
public:
    virtual ~ModelSpecBase() = default;

    // 判别符（字符串；同时是文件 `type` 键的值）
    [[nodiscard]] virtual const char* type_name() const noexcept = 0;
    // 深拷贝（ModelSpec 外壳按值语义的支撑）
    [[nodiscard]] virtual std::unique_ptr<ModelSpecBase> clone() const = 0;
    // 架构兼容（替换原 `spec_matches` 的类型分发；类型不同 ⇒ false）
    [[nodiscard]] virtual bool arch_compatible(const ModelSpecBase& other) const = 0;
    // 诊断摘要（替换原 `spec_summary` 的类型分发；读外壳公共字段）
    [[nodiscard]] virtual std::string summary(const ModelSpec& shell) const = 0;
    // 类型特有字段落盘（键名与既有扁平布局一致；缺失键读回默认值）
    virtual void write_kv(KeyValueRecord& kv) const = 0;
    [[nodiscard]] virtual Result<void> read_kv(const KeyValueRecord& kv) = 0;
};

// ── MLP ───────────────────────────────────────────────────────────────────
class MlpSpec : public ModelSpecBase
{
public:
    [[nodiscard]] const char* type_name() const noexcept override { return "MLP"; }
    [[nodiscard]] std::unique_ptr<ModelSpecBase> clone() const override
    { return std::make_unique<MlpSpec>(*this); }

    std::vector<std::size_t> layer_dims;          // 隐藏层宽度序列（含输出类数）
    NormType norm_type = NormType::LayerNorm;     // 影响参数个数（RMSNorm 无 beta 等）
    WeightQuant weight_quant = WeightQuant::None; // 线性层是否 BitLinear

    [[nodiscard]] bool arch_compatible(const ModelSpecBase& o) const override
    {
        const auto* p = dynamic_cast<const MlpSpec*>(&o);
        // weight_quant 同理：参数**形状**相同（latent 恒 f32），但层语义不同
        //（Linear vs BitLinear）→ 不比对等于允许"把三值模型权重加载进普通 MLP"。
        return p && layer_dims == p->layer_dims && norm_type == p->norm_type &&
               weight_quant == p->weight_quant;
    }
    [[nodiscard]] std::string summary(const ModelSpec& shell) const override;
    void write_kv(KeyValueRecord& kv) const override;
    [[nodiscard]] Result<void> read_kv(const KeyValueRecord& kv) override;
};

// ── Transformer（MNIST ViT 风格编码器）────────────────────────────────────
class TransformerSpec : public ModelSpecBase
{
public:
    [[nodiscard]] const char* type_name() const noexcept override { return "Transformer"; }
    [[nodiscard]] std::unique_ptr<ModelSpecBase> clone() const override
    { return std::make_unique<TransformerSpec>(*this); }

    std::size_t d_model = 0;
    std::size_t num_heads = 0;
    std::size_t d_ff = 0;
    std::size_t num_layers = 0;
    std::size_t patch_size = 0;
    NormType norm_type = NormType::LayerNorm;
    NormPlace norm_place = NormPlace::None;
    WeightQuant weight_quant = WeightQuant::None;

    [[nodiscard]] bool arch_compatible(const ModelSpecBase& o) const override
    {
        const auto* p = dynamic_cast<const TransformerSpec*>(&o);
        return p && d_model == p->d_model && num_heads == p->num_heads &&
               d_ff == p->d_ff && num_layers == p->num_layers &&
               patch_size == p->patch_size && norm_type == p->norm_type &&
               norm_place == p->norm_place && weight_quant == p->weight_quant;
    }
    [[nodiscard]] std::string summary(const ModelSpec& shell) const override;
    void write_kv(KeyValueRecord& kv) const override;
    [[nodiscard]] Result<void> read_kv(const KeyValueRecord& kv) override;
};

// ── GPT（含 ALiBi/RoPE 变体；旧 ALiBi_GPT 数值 type 同归本类）──────────────
class GptSpec : public ModelSpecBase
{
public:
    [[nodiscard]] const char* type_name() const noexcept override { return "GPT"; }
    [[nodiscard]] std::unique_ptr<ModelSpecBase> clone() const override
    { return std::make_unique<GptSpec>(*this); }

    std::size_t vocab_size = 0;
    std::size_t seq_len = 0;
    std::size_t d_model = 0;
    std::size_t num_heads = 0;
    std::size_t d_ff = 0;
    std::size_t num_layers = 0;
    PosEncodingType pos_encoding = PosEncodingType::Learned;
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type = NormType::LayerNorm;
    bool subln = false;              // 每 block 子层内部 norm（BitNet 2B4T）
    std::size_t n_head_kv = 0;       // GQA 的 KV 头数（0 = MHA）
    bool tie_embeddings = false;     // head 复用 token embedding
    WeightQuant weight_quant = WeightQuant::None;

    [[nodiscard]] std::size_t n_head_kv_or_heads() const noexcept
    { return n_head_kv == 0 ? num_heads : n_head_kv; }

    [[nodiscard]] bool arch_compatible(const ModelSpecBase& o) const override
    {
        const auto* p = dynamic_cast<const GptSpec*>(&o);
        // ALiBi_GPT 与 GPT 同族，统一由 GPTModel 承载，用 pos_encoding 区分
        // Learned/Sinusoidal/ALiBi/RoPE —— 同为 GptSpec 即同族。
        return p && vocab_size == p->vocab_size && d_model == p->d_model &&
               seq_len == p->seq_len && num_heads == p->num_heads &&
               d_ff == p->d_ff && num_layers == p->num_layers &&
               pos_encoding == p->pos_encoding && activation == p->activation &&
               norm_type == p->norm_type && subln == p->subln &&
               n_head_kv_or_heads() == p->n_head_kv_or_heads() &&
               tie_embeddings == p->tie_embeddings &&
               weight_quant == p->weight_quant;
    }
    [[nodiscard]] std::string summary(const ModelSpec& shell) const override;
    void write_kv(KeyValueRecord& kv) const override;
    [[nodiscard]] Result<void> read_kv(const KeyValueRecord& kv) override;
};

// ── CNN（LeNet 风格；MNIST）───────────────────────────────────────────────
class CnnSpec : public ModelSpecBase
{
public:
    [[nodiscard]] const char* type_name() const noexcept override { return "CNN"; }
    [[nodiscard]] std::unique_ptr<ModelSpecBase> clone() const override
    { return std::make_unique<CnnSpec>(*this); }

    std::size_t cnn_in_channels = 0;         // 输入通道数（MNIST=1）
    std::size_t cnn_in_size = 0;             // 输入空间尺寸（方形，MNIST=28）
    std::size_t cnn_pool = 0;                // 每个卷积后的 MaxPool 窗口（0=无）
    std::vector<std::size_t> cnn_channels;   // 每个卷积层输出通道数
    std::vector<std::size_t> cnn_kernels;    // 每个卷积核大小
    std::vector<std::size_t> cnn_strides;    // 每个卷积步长（默认 1）
    std::vector<std::size_t> cnn_paddings;   // 每个卷积填充（默认 0）
    // 全连接头（layer_dims 复用）：展平 → Linear(H1) → … → Linear(num_classes)
    std::vector<std::size_t> layer_dims;
    // 结构默认与旧扁平布局一致 = LayerNorm（缺键回落；**不**是架构 builder
    // 默认 —— make_cnn_spec 显式传 BatchNorm@Conv 并写入文件）。
    NormType norm_type = NormType::LayerNorm;
    NormPlace norm_place = NormPlace::None;
    WeightQuant weight_quant = WeightQuant::None;

    [[nodiscard]] bool arch_compatible(const ModelSpecBase& o) const override
    {
        const auto* p = dynamic_cast<const CnnSpec*>(&o);
        return p && cnn_in_channels == p->cnn_in_channels &&
               cnn_in_size == p->cnn_in_size && cnn_pool == p->cnn_pool &&
               cnn_channels == p->cnn_channels && cnn_kernels == p->cnn_kernels &&
               cnn_strides == p->cnn_strides && cnn_paddings == p->cnn_paddings &&
               layer_dims == p->layer_dims && norm_type == p->norm_type &&
               norm_place == p->norm_place && weight_quant == p->weight_quant;
    }
    [[nodiscard]] std::string summary(const ModelSpec& shell) const override;
    void write_kv(KeyValueRecord& kv) const override;
    [[nodiscard]] Result<void> read_kv(const KeyValueRecord& kv) override;
};

// ── RAPT（ReLU-Linear Attention，causal LM）───────────────────────────────
class RaptSpec : public ModelSpecBase
{
public:
    [[nodiscard]] const char* type_name() const noexcept override { return "RAPT"; }
    [[nodiscard]] std::unique_ptr<ModelSpecBase> clone() const override
    { return std::make_unique<RaptSpec>(*this); }

    std::size_t vocab_size = 0;
    std::size_t seq_len = 0;
    std::size_t d_model = 0;
    std::size_t num_heads = 0;
    std::size_t d_ff = 0;
    std::size_t num_layers = 0;
    PosEncodingType pos_encoding = PosEncodingType::Learned;
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type = NormType::LayerNorm;
    WeightQuant weight_quant = WeightQuant::None;

    [[nodiscard]] bool arch_compatible(const ModelSpecBase& o) const override
    {
        const auto* p = dynamic_cast<const RaptSpec*>(&o);
        return p && vocab_size == p->vocab_size && d_model == p->d_model &&
               seq_len == p->seq_len && num_heads == p->num_heads &&
               d_ff == p->d_ff && num_layers == p->num_layers &&
               pos_encoding == p->pos_encoding && activation == p->activation &&
               norm_type == p->norm_type && weight_quant == p->weight_quant;
    }
    [[nodiscard]] std::string summary(const ModelSpec& shell) const override;
    void write_kv(KeyValueRecord& kv) const override;
    [[nodiscard]] Result<void> read_kv(const KeyValueRecord& kv) override;
};

// ═══════════════════════════════════════════════════════════════════════
//  ModelSpec：公共字段外壳（值语义；类型参数随 arch 多态走）
// ═══════════════════════════════════════════════════════════════════════
struct ModelSpec
{
    // ── 域无关公共字段（通用多输出规范，docs/internal/31 RL-0）──────────────
    std::optional<TensorSchema> output_schema;   // nullopt = 旧单输出模型
    std::vector<std::string> output_packing;     // 空 = 不生成兼容打包表示

    // ── 类型化规格（判别符 + 类型参数；空 = Unknown）─────────────────────
    std::unique_ptr<ModelSpecBase> arch;

    ModelSpec() = default;
    explicit ModelSpec(std::unique_ptr<ModelSpecBase> a) : arch(std::move(a)) {}
    ModelSpec(const ModelSpec& o)
        : output_schema(o.output_schema), output_packing(o.output_packing),
          arch(o.arch ? o.arch->clone() : nullptr) {}
    ModelSpec& operator=(const ModelSpec& o)
    {
        if (this != &o)
        {
            output_schema = o.output_schema;
            output_packing = o.output_packing;
            arch = o.arch ? o.arch->clone() : nullptr;
        }
        return *this;
    }
    ModelSpec(ModelSpec&&) noexcept = default;
    ModelSpec& operator=(ModelSpec&&) noexcept = default;

    // 就地构造类型化规格，返回引用便于逐字段填充
    template <class T>
    T& emplace()
    {
        auto p = std::make_unique<T>();
        T& ref = *p;
        arch = std::move(p);
        return ref;
    }

    [[nodiscard]] const char* type_name() const noexcept
    { return arch ? arch->type_name() : "Unknown"; }
    [[nodiscard]] bool has_arch() const noexcept { return arch != nullptr; }

    template <class T>
    [[nodiscard]] T* as() noexcept { return dynamic_cast<T*>(arch.get()); }
    template <class T>
    [[nodiscard]] const T* as() const noexcept
    { return dynamic_cast<const T*>(arch.get()); }

    // ── 判别（调用点兼容层；实现 = 类型化 dynamic_cast）────────────────
    [[nodiscard]] bool is_mlp() const noexcept { return as<MlpSpec>() != nullptr; }
    [[nodiscard]] bool is_transformer() const noexcept
    { return as<TransformerSpec>() != nullptr; }
    [[nodiscard]] bool is_gpt() const noexcept { return as<GptSpec>() != nullptr; }
    [[nodiscard]] bool is_alibi_gpt() const noexcept
    {
        const auto* g = as<GptSpec>();
        return g && g->pos_encoding == PosEncodingType::ALiBi;
    }
    [[nodiscard]] bool is_cnn() const noexcept { return as<CnnSpec>() != nullptr; }
    [[nodiscard]] bool is_rapt() const noexcept { return as<RaptSpec>() != nullptr; }
};

// ── 架构一致性校验（公共字段 + arch_compatible；语义见子类注释）────────────
[[nodiscard]] inline bool spec_matches(const ModelSpec& a, const ModelSpec& b)
{
    if (a.output_schema.has_value() != b.output_schema.has_value())
        return false;
    if (a.output_schema.has_value() && *a.output_schema != *b.output_schema)
        return false;
    if (a.output_packing != b.output_packing)
        return false;
    if (!a.arch || !b.arch)
        return false;   // 任一侧无类型化规格（Unknown）⇒ 不匹配（与旧 default 一致）
    return a.arch->arch_compatible(*b.arch);
}

// ── 类型特有字段的诊断摘要（外壳补公共后缀）──────────────────────────────
[[nodiscard]] inline std::string spec_summary(const ModelSpec& s)
{
    if (s.arch) return s.arch->summary(s);
    // Unknown：与旧 default 分支同形（layers 恒 0）
    std::string out = "Unknown(layers=0";
    if (s.output_schema) out += ",outputs=" + std::to_string(s.output_schema->size());
    out += ")";
    return out;
}

// ── 内建类型名 → 子类（序列化读回用；未命中回落 ModelRegistry）────────────
[[nodiscard]] inline std::unique_ptr<ModelSpecBase> make_spec_by_name(const std::string& name)
{
    if (name == "MLP") return std::make_unique<MlpSpec>();
    if (name == "Transformer") return std::make_unique<TransformerSpec>();
    if (name == "GPT") return std::make_unique<GptSpec>();
    if (name == "CNN") return std::make_unique<CnnSpec>();
    if (name == "RAPT") return std::make_unique<RaptSpec>();
    return nullptr;   // 未知：调用方查 ModelRegistry 或报错
}

// ── 旧文件数值 type → 类型名（只读回落；仅向下兼容）──────────────────────
// 0=Unknown（无 arch）、4=ALiBi_GPT 归 GPT 同族、6=ZiPT（已移除，返回
// 非法哨兵由序列化层给出明确错误）。
[[nodiscard]] inline const char* legacy_spec_type_name(std::uint64_t v) noexcept
{
    switch (v)
    {
    case 0: return nullptr;      // Unknown：不构造 arch
    case 1: return "MLP";
    case 2: return "Transformer";
    case 3: return "GPT";
    case 4: return "GPT";        // ALiBi_GPT：同族，pos_encoding 从字段还原
    case 5: return "CNN";
    case 6: return "zipt";       // 哨兵：ZiPT 已移除（序列化层显式报错）
    case 7: return "RAPT";
    default: return nullptr;
    }
}

// ── 自定义模型类型注册表（#26 开放扩展点）────────────────────────────────
// 内建类型直通 `make_spec_by_name`；自定义域 `register_type("MyNet", …)` 后，
// 文件 `type="MyNet"` 即可读回 —— 加新模型不改本头任何 switch。
class ModelRegistry
{
public:
    using Factory = std::function<std::unique_ptr<ModelSpecBase>()>;

    [[nodiscard]] static ModelRegistry& instance()
    {
        static ModelRegistry reg;
        return reg;
    }

    void register_type(std::string name, Factory f)
    { types_[std::move(name)] = std::move(f); }

    [[nodiscard]] std::unique_ptr<ModelSpecBase> make(const std::string& name) const
    {
        const auto it = types_.find(name);
        return it != types_.end() && it->second ? it->second() : nullptr;
    }

private:
    std::unordered_map<std::string, Factory> types_;
};

// 统一读回入口：内建优先，其次注册表；均未命中 → nullptr
[[nodiscard]] inline std::unique_ptr<ModelSpecBase> make_spec(const std::string& name)
{
    if (auto b = make_spec_by_name(name)) return b;
    return ModelRegistry::instance().make(name);
}

// ── 子类摘要/编解码的外置定义（需完整 ModelSpec / KeyValueRecord）──────────
[[nodiscard]] inline std::string MlpSpec::summary(const ModelSpec& shell) const
{
    // 旧扁平布局的默认分支口径：layers 恒 0（MLP 用 layer_dims 表达深度）
    std::string out = std::string("MLP(layers=0");
    if (shell.output_schema)
        out += ",outputs=" + std::to_string(shell.output_schema->size());
    if (weight_quant != WeightQuant::None)
        out += ",wq=" + std::string(weight_quant_name(weight_quant));
    return out + ")";
}

[[nodiscard]] inline std::string TransformerSpec::summary(const ModelSpec& shell) const
{
    std::string out = std::string("Transformer(layers=") + std::to_string(num_layers);
    if (shell.output_schema)
        out += ",outputs=" + std::to_string(shell.output_schema->size());
    if (weight_quant != WeightQuant::None)
        out += ",wq=" + std::string(weight_quant_name(weight_quant));
    return out + ")";
}

[[nodiscard]] inline std::string GptSpec::summary(const ModelSpec&) const
{
    return std::string("GPT(vocab=") + std::to_string(vocab_size) +
           ",d_model=" + std::to_string(d_model) +
           ",seq_len=" + std::to_string(seq_len) +
           ",heads=" + std::to_string(num_heads) +
           ",d_ff=" + std::to_string(d_ff) +
           ",layers=" + std::to_string(num_layers) +
           (subln ? ",subln=1" : "") +
           (n_head_kv_or_heads() != num_heads
                ? ",kv_heads=" + std::to_string(n_head_kv_or_heads()) : "") +
           (tie_embeddings ? ",tied=1" : "") + ")";
}

[[nodiscard]] inline std::string CnnSpec::summary(const ModelSpec&) const
{
    std::string c = std::to_string(cnn_in_channels) + "x" +
                    std::to_string(cnn_in_size) + "->";
    for (std::size_t i = 0; i < cnn_channels.size(); ++i)
    {
        if (i) c += "-";
        c += std::to_string(cnn_channels[i]) + "@" + std::to_string(cnn_kernels[i]);
    }
    return std::string("CNN(") + c + ")";
}

[[nodiscard]] inline std::string RaptSpec::summary(const ModelSpec&) const
{
    return std::string("RAPT(vocab=") + std::to_string(vocab_size) +
           ",d_model=" + std::to_string(d_model) +
           ",seq_len=" + std::to_string(seq_len) +
           ",heads=" + std::to_string(num_heads) +
           ",d_ff=" + std::to_string(d_ff) +
           ",layers=" + std::to_string(num_layers) +
           ",pos_enc=" + std::to_string(static_cast<unsigned>(pos_encoding)) + ")";
}

// ── 类型特有字段编解码（键名与既有扁平布局逐键一致）──────────────────────
inline void MlpSpec::write_kv(KeyValueRecord& kv) const
{
    std::vector<std::uint64_t> dims;
    dims.reserve(layer_dims.size());
    for (auto d : layer_dims) dims.push_back(static_cast<std::uint64_t>(d));
    kv.set("layer_dims", dims);
    kv.set("norm_type", static_cast<std::uint64_t>(norm_type));
    kv.set("weight_quant", static_cast<std::uint64_t>(weight_quant));
}

inline void TransformerSpec::write_kv(KeyValueRecord& kv) const
{
    kv.set("d_model", static_cast<std::uint64_t>(d_model));
    kv.set("num_heads", static_cast<std::uint64_t>(num_heads));
    kv.set("d_ff", static_cast<std::uint64_t>(d_ff));
    kv.set("num_layers", static_cast<std::uint64_t>(num_layers));
    kv.set("patch_size", static_cast<std::uint64_t>(patch_size));
    kv.set("norm_type", static_cast<std::uint64_t>(norm_type));
    kv.set("norm_place", static_cast<std::uint64_t>(norm_place));
    kv.set("weight_quant", static_cast<std::uint64_t>(weight_quant));
}

inline void GptSpec::write_kv(KeyValueRecord& kv) const
{
    kv.set("vocab_size", static_cast<std::uint64_t>(vocab_size));
    kv.set("seq_len", static_cast<std::uint64_t>(seq_len));
    kv.set("d_model", static_cast<std::uint64_t>(d_model));
    kv.set("num_heads", static_cast<std::uint64_t>(num_heads));
    kv.set("d_ff", static_cast<std::uint64_t>(d_ff));
    kv.set("num_layers", static_cast<std::uint64_t>(num_layers));
    kv.set("pos_encoding", static_cast<std::uint64_t>(pos_encoding));
    kv.set("activation", static_cast<std::uint64_t>(activation));
    kv.set("norm_type", static_cast<std::uint64_t>(norm_type));
    kv.set("subln", static_cast<std::uint64_t>(subln ? 1 : 0));
    kv.set("n_head_kv", static_cast<std::uint64_t>(n_head_kv));
    kv.set("tie_embeddings", static_cast<std::uint64_t>(tie_embeddings ? 1 : 0));
    kv.set("weight_quant", static_cast<std::uint64_t>(weight_quant));
}

inline void CnnSpec::write_kv(KeyValueRecord& kv) const
{
    kv.set("cnn_in_channels", static_cast<std::uint64_t>(cnn_in_channels));
    kv.set("cnn_in_size", static_cast<std::uint64_t>(cnn_in_size));
    kv.set("cnn_pool", static_cast<std::uint64_t>(cnn_pool));
    auto arr = [](const std::vector<std::size_t>& v) {
        std::vector<std::uint64_t> out;
        out.reserve(v.size());
        for (auto x : v) out.push_back(static_cast<std::uint64_t>(x));
        return out;
    };
    kv.set("cnn_channels", arr(cnn_channels));
    kv.set("cnn_kernels", arr(cnn_kernels));
    kv.set("cnn_strides", arr(cnn_strides));
    kv.set("cnn_paddings", arr(cnn_paddings));
    kv.set("layer_dims", arr(layer_dims));
    kv.set("norm_type", static_cast<std::uint64_t>(norm_type));
    kv.set("norm_place", static_cast<std::uint64_t>(norm_place));
    kv.set("weight_quant", static_cast<std::uint64_t>(weight_quant));
}

inline void RaptSpec::write_kv(KeyValueRecord& kv) const
{
    kv.set("vocab_size", static_cast<std::uint64_t>(vocab_size));
    kv.set("seq_len", static_cast<std::uint64_t>(seq_len));
    kv.set("d_model", static_cast<std::uint64_t>(d_model));
    kv.set("num_heads", static_cast<std::uint64_t>(num_heads));
    kv.set("d_ff", static_cast<std::uint64_t>(d_ff));
    kv.set("num_layers", static_cast<std::uint64_t>(num_layers));
    kv.set("pos_encoding", static_cast<std::uint64_t>(pos_encoding));
    kv.set("activation", static_cast<std::uint64_t>(activation));
    kv.set("norm_type", static_cast<std::uint64_t>(norm_type));
    kv.set("weight_quant", static_cast<std::uint64_t>(weight_quant));
}

namespace spec_kv_detail
{
// 读回辅助：缺失键保持默认值（与旧"读缺省"口径一致）
inline void get_u64(const KeyValueRecord& kv, const char* key, std::uint64_t& out)
{
    std::uint64_t v = 0;
    if (kv.get(key, v)) out = v;
}
inline void get_size(const KeyValueRecord& kv, const char* key, std::size_t& out)
{
    std::uint64_t v = 0;
    if (kv.get(key, v)) out = static_cast<std::size_t>(v);
}
inline void get_dims(const KeyValueRecord& kv, const char* key,
                     std::vector<std::size_t>& out)
{
    std::vector<std::uint64_t> v;
    if (kv.get(key, v))
    {
        out.clear();
        out.reserve(v.size());
        for (auto x : v) out.push_back(static_cast<std::size_t>(x));
    }
}
} // namespace spec_kv_detail

inline Result<void> MlpSpec::read_kv(const KeyValueRecord& kv)
{
    using namespace spec_kv_detail;
    get_dims(kv, "layer_dims", layer_dims);
    std::uint64_t v = static_cast<std::uint64_t>(norm_type);
    get_u64(kv, "norm_type", v);
    norm_type = static_cast<NormType>(v);
    v = static_cast<std::uint64_t>(weight_quant);
    get_u64(kv, "weight_quant", v);
    weight_quant = static_cast<WeightQuant>(v);
    return {};
}

inline Result<void> TransformerSpec::read_kv(const KeyValueRecord& kv)
{
    using namespace spec_kv_detail;
    get_size(kv, "d_model", d_model);
    get_size(kv, "num_heads", num_heads);
    get_size(kv, "d_ff", d_ff);
    get_size(kv, "num_layers", num_layers);
    get_size(kv, "patch_size", patch_size);
    std::uint64_t v = static_cast<std::uint64_t>(norm_type);
    get_u64(kv, "norm_type", v);
    norm_type = static_cast<NormType>(v);
    v = static_cast<std::uint64_t>(norm_place);
    get_u64(kv, "norm_place", v);
    norm_place = static_cast<NormPlace>(v);
    v = static_cast<std::uint64_t>(weight_quant);
    get_u64(kv, "weight_quant", v);
    weight_quant = static_cast<WeightQuant>(v);
    return {};
}

inline Result<void> GptSpec::read_kv(const KeyValueRecord& kv)
{
    using namespace spec_kv_detail;
    get_size(kv, "vocab_size", vocab_size);
    get_size(kv, "seq_len", seq_len);
    get_size(kv, "d_model", d_model);
    get_size(kv, "num_heads", num_heads);
    get_size(kv, "d_ff", d_ff);
    get_size(kv, "num_layers", num_layers);
    std::uint64_t v = static_cast<std::uint64_t>(pos_encoding);
    get_u64(kv, "pos_encoding", v);
    pos_encoding = static_cast<PosEncodingType>(v);
    v = static_cast<std::uint64_t>(activation);
    get_u64(kv, "activation", v);
    activation = static_cast<ActivationType>(v);
    v = static_cast<std::uint64_t>(norm_type);
    get_u64(kv, "norm_type", v);
    norm_type = static_cast<NormType>(v);
    v = subln ? 1u : 0u;
    get_u64(kv, "subln", v);
    subln = (v != 0);
    get_size(kv, "n_head_kv", n_head_kv);
    v = tie_embeddings ? 1u : 0u;
    get_u64(kv, "tie_embeddings", v);
    tie_embeddings = (v != 0);
    v = static_cast<std::uint64_t>(weight_quant);
    get_u64(kv, "weight_quant", v);
    weight_quant = static_cast<WeightQuant>(v);
    return {};
}

inline Result<void> CnnSpec::read_kv(const KeyValueRecord& kv)
{
    using namespace spec_kv_detail;
    get_size(kv, "cnn_in_channels", cnn_in_channels);
    get_size(kv, "cnn_in_size", cnn_in_size);
    get_size(kv, "cnn_pool", cnn_pool);
    get_dims(kv, "cnn_channels", cnn_channels);
    get_dims(kv, "cnn_kernels", cnn_kernels);
    get_dims(kv, "cnn_strides", cnn_strides);
    get_dims(kv, "cnn_paddings", cnn_paddings);
    get_dims(kv, "layer_dims", layer_dims);
    std::uint64_t v = static_cast<std::uint64_t>(norm_type);
    get_u64(kv, "norm_type", v);
    norm_type = static_cast<NormType>(v);
    v = static_cast<std::uint64_t>(norm_place);
    get_u64(kv, "norm_place", v);
    norm_place = static_cast<NormPlace>(v);
    v = static_cast<std::uint64_t>(weight_quant);
    get_u64(kv, "weight_quant", v);
    weight_quant = static_cast<WeightQuant>(v);
    return {};
}

inline Result<void> RaptSpec::read_kv(const KeyValueRecord& kv)
{
    using namespace spec_kv_detail;
    get_size(kv, "vocab_size", vocab_size);
    get_size(kv, "seq_len", seq_len);
    get_size(kv, "d_model", d_model);
    get_size(kv, "num_heads", num_heads);
    get_size(kv, "d_ff", d_ff);
    get_size(kv, "num_layers", num_layers);
    std::uint64_t v = static_cast<std::uint64_t>(pos_encoding);
    get_u64(kv, "pos_encoding", v);
    pos_encoding = static_cast<PosEncodingType>(v);
    v = static_cast<std::uint64_t>(activation);
    get_u64(kv, "activation", v);
    activation = static_cast<ActivationType>(v);
    v = static_cast<std::uint64_t>(norm_type);
    get_u64(kv, "norm_type", v);
    norm_type = static_cast<NormType>(v);
    v = static_cast<std::uint64_t>(weight_quant);
    get_u64(kv, "weight_quant", v);
    weight_quant = static_cast<WeightQuant>(v);
    return {};
}

} // namespace nn
