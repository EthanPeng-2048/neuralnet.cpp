#ifndef NN_DOMAIN_GPT_HPP
#define NN_DOMAIN_GPT_HPP

// ── domain_gpt.hpp — GPT 领域构建层（引擎化架构） ──────────────────────────
//
// 依赖：Model + ComputeEngine + Tokenizer
//   （model_container.hpp / compute_engine.hpp / domain_tokenizer.hpp）
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>

#include "core_errors.hpp"
#include "compute_engine.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"

namespace nn {

// ── GPT 默认超参数 ────────────────────────────────────────────────────────
inline constexpr std::size_t GPT_VOCAB_SIZE    = 10000;
inline constexpr std::size_t GPT_D_MODEL       = 128;
inline constexpr std::size_t GPT_NUM_HEADS     = 4;
inline constexpr std::size_t GPT_D_FF          = 512;
inline constexpr std::size_t GPT_NUM_LAYERS    = 4;
inline constexpr std::size_t GPT_SEQ_LEN       = 256;

// ── GPT 配置结构体 ────────────────────────────────────────────────────────
// 将 build_gpt_model 的多个位置参数收拢为一个结构体，避免调用方签名过长。
struct GptConfig {
    std::size_t vocab_size  = GPT_VOCAB_SIZE;
    std::size_t d_model     = GPT_D_MODEL;
    std::size_t seq_len     = GPT_SEQ_LEN;
    std::size_t num_heads   = GPT_NUM_HEADS;
    std::size_t d_ff        = GPT_D_FF;
    std::size_t num_layers  = GPT_NUM_LAYERS;
    PosEncodingType pos_enc = PosEncodingType::Learned;
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type = NormType::LayerNorm;
};

// ── 构建 GPT 模型（GptConfig 版本，推荐使用） ────────────────────────────
// GPTModel 是一个单一 Layer（内含 TokenEmb+PosEmb+N×GPTBlock+LN+LM Head），
// 作为 Model 的唯一层。engine 决定权重张量驻留设备。
// cfg.pos_enc 控制位置编码：Learned(默认)/Sinusoidal/ALiBi
[[nodiscard]] inline Result<Model> build_gpt_model(
    ComputeEngine& engine, const GptConfig& cfg)
{
    if (cfg.d_model == 0 || cfg.num_heads == 0 || cfg.seq_len == 0 || cfg.vocab_size == 0)
        return std::unexpected(Error{"GPT model parameters must be positive"});
    if (cfg.d_ff == 0 || cfg.num_layers == 0)
        return std::unexpected(Error{"GPT d_ff and num_layers must be positive"});
    if (cfg.d_model % cfg.num_heads != 0)
        return std::unexpected(Error{"GPT d_model must be divisible by num_heads"});

    Model model(engine);
    model.add<GPTModel>(engine, cfg.vocab_size, cfg.d_model, cfg.seq_len,
                        cfg.num_heads, cfg.d_ff, cfg.num_layers, cfg.pos_enc,
                        cfg.activation, cfg.norm_type);
    return model;
}

// ── 构建 GPT 模型（位置参数版本，转发至 GptConfig 版本，向后兼容） ───────
[[nodiscard]] inline Result<Model> build_gpt_model(
    ComputeEngine& engine,
    std::size_t vocab_size  = GPT_VOCAB_SIZE,
    std::size_t d_model     = GPT_D_MODEL,
    std::size_t seq_len     = GPT_SEQ_LEN,
    std::size_t num_heads   = GPT_NUM_HEADS,
    std::size_t d_ff        = GPT_D_FF,
    std::size_t num_layers  = GPT_NUM_LAYERS,
    PosEncodingType pos_enc_type = PosEncodingType::Learned,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm)
{
    return build_gpt_model(engine, GptConfig{
        vocab_size, d_model, seq_len, num_heads, d_ff, num_layers,
        pos_enc_type, activation, norm_type});
}

// ── 从 ModelSpec 构建 GPT 模型 ──────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构。
// 统一的 GPTModel 通过 spec.pos_encoding 区分 Learned/Sinusoidal/ALiBi 模式，
// 因此无需为 ALiBi 单独提供构建函数。
[[nodiscard]] inline Result<Model> build_gpt_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec)
{
    // 接受 GPT 类型，或旧格式 ALiBi_GPT 类型（向后兼容旧模型文件）
    if (!spec.is_gpt() && !spec.is_alibi_gpt())
        return std::unexpected(Error{"Invalid ModelSpec type for GPT: expected GPT or ALiBi_GPT"});

    return build_gpt_model(
        engine,
        spec.vocab_size, spec.d_model, spec.seq_len,
        spec.num_heads, spec.d_ff, spec.num_layers,
        spec.pos_encoding, spec.activation, spec.norm_type);
}

// ── 构造 GPT ModelSpec ──────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_gpt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers,
    PosEncodingType pos_encoding = PosEncodingType::Learned,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm)
{
    ModelSpec spec;
    spec.type         = ModelType::GPT;
    spec.vocab_size   = vocab_size;
    spec.d_model      = d_model;
    spec.seq_len      = seq_len;
    spec.num_heads    = num_heads;
    spec.d_ff         = d_ff;
    spec.num_layers   = num_layers;
    spec.pos_encoding = pos_encoding;
    spec.activation   = activation;
    spec.norm_type    = norm_type;
    return spec;
}

} // namespace nn

#endif // NN_DOMAIN_GPT_HPP
