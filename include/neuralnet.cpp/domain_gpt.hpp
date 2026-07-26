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

// ── 构建 GPT 模型 ────────────────────────────────────────────────────────
// GPTModel 是一个单一 Layer（内含 TokenEmb+PosEmb+N×GPTBlock+LN+LM Head），
// 作为 Model 的唯一层。engine 决定权重张量驻留设备。
// pos_enc_type 控制位置编码：Learned(默认)/Sinusoidal/ALiBi
[[nodiscard]] inline Result<Model> build_gpt_model(
    ComputeEngine& engine,
    std::size_t vocab_size  = GPT_VOCAB_SIZE,
    std::size_t d_model     = GPT_D_MODEL,
    std::size_t seq_len     = GPT_SEQ_LEN,
    std::size_t num_heads   = GPT_NUM_HEADS,
    std::size_t d_ff        = GPT_D_FF,
    std::size_t num_layers  = GPT_NUM_LAYERS,
    PosEncodingType pos_enc_type = PosEncodingType::Learned)
{
    if (d_model == 0 || num_heads == 0 || seq_len == 0 || vocab_size == 0)
        return std::unexpected(Error{"GPT model parameters must be positive"});
    if (d_ff == 0 || num_layers == 0)
        return std::unexpected(Error{"GPT d_ff and num_layers must be positive"});
    if (d_model % num_heads != 0)
        return std::unexpected(Error{"GPT d_model must be divisible by num_heads"});

    Model model(engine);
    model.add_gpt_model(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers, pos_enc_type);
    return model;
}

// ── 从 ModelSpec 构建 GPT 模型 ──────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构
[[nodiscard]] inline Result<Model> build_gpt_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec)
{
    // 支持 GPT 和 ALiBi_GPT 两种类型
    if (!spec.is_gpt() && !spec.is_alibi_gpt())
        return std::unexpected(Error{"Invalid ModelSpec type for GPT: expected GPT or ALiBi_GPT"});

    return build_gpt_model(
        engine,
        spec.vocab_size, spec.d_model, spec.seq_len,
        spec.num_heads, spec.d_ff, spec.num_layers,
        spec.pos_encoding);
}

// ── 构造 GPT ModelSpec ──────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_gpt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers,
    PosEncodingType pos_encoding = PosEncodingType::Learned)
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
    return spec;
}

// ── 构建 ALiBi GPT 模型（向后兼容接口） ────────────────────────────────
[[nodiscard]] inline Result<Model> build_alibi_gpt_model(
    ComputeEngine& engine,
    std::size_t vocab_size  = GPT_VOCAB_SIZE,
    std::size_t d_model     = GPT_D_MODEL,
    std::size_t seq_len     = GPT_SEQ_LEN,
    std::size_t num_heads   = GPT_NUM_HEADS,
    std::size_t d_ff        = GPT_D_FF,
    std::size_t num_layers  = GPT_NUM_LAYERS)
{
    return build_gpt_model(engine, vocab_size, d_model, seq_len,
                           num_heads, d_ff, num_layers, PosEncodingType::ALiBi);
}

// ── 从 ModelSpec 构建 ALiBi GPT 模型（向后兼容接口） ──────────────────
[[nodiscard]] inline Result<Model> build_alibi_gpt_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec)
{
    return build_gpt_model_from_spec(engine, spec);
}

// ── 构造 ALiBi GPT ModelSpec（向后兼容接口） ──────────────────────────
[[nodiscard]] inline ModelSpec make_alibi_gpt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers)
{
    return make_gpt_spec(vocab_size, d_model, seq_len,
                         num_heads, d_ff, num_layers, PosEncodingType::ALiBi);
}

} // namespace nn

#endif // NN_DOMAIN_GPT_HPP
