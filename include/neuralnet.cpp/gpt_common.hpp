#ifndef GPT_COMMON_HPP
#define GPT_COMMON_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <expected>

#include "model.hpp"
#include "layer.hpp"
#include "tokenizer.hpp"

namespace nn {

// ── GPT 默认超参数 ────────────────────────────────────────────────────────
inline constexpr std::size_t GPT_VOCAB_SIZE    = 10000;
inline constexpr std::size_t GPT_D_MODEL       = 128;
inline constexpr std::size_t GPT_NUM_HEADS     = 4;
inline constexpr std::size_t GPT_D_FF          = 512;
inline constexpr std::size_t GPT_NUM_LAYERS    = 4;
inline constexpr std::size_t GPT_SEQ_LEN       = 256;

// ── 从文本文件加载语料 ────────────────────────────────────────────────────
[[nodiscard]] inline Result<std::string> load_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open text file: " + path});
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    return content;
}

// ── 构建 GPT 模型 ────────────────────────────────────────────────────────
[[nodiscard]] inline Model build_gpt_model(
    std::size_t vocab_size  = GPT_VOCAB_SIZE,
    std::size_t d_model     = GPT_D_MODEL,
    std::size_t seq_len     = GPT_SEQ_LEN,
    std::size_t num_heads   = GPT_NUM_HEADS,
    std::size_t d_ff        = GPT_D_FF,
    std::size_t num_layers  = GPT_NUM_LAYERS)
{
    // GPTModel 是一个单一 Layer，作为 Model 的唯一层
    Model model;
    model.add<GPTModel>(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers);
    return model;
}

// ── 从 ModelSpec 构建 GPT 模型 ──────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构
[[nodiscard]] inline Result<Model> build_gpt_model_from_spec(const ModelSpec &spec)
{
    if (!spec.is_gpt())
        return std::unexpected(Error{"Invalid ModelSpec type for GPT: expected GPT"});

    return build_gpt_model(
        spec.vocab_size, spec.d_model, spec.seq_len,
        spec.num_heads, spec.d_ff, spec.num_layers);
}

// ── 构造 GPT ModelSpec ──────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_gpt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers)
{
    ModelSpec spec;
    spec.type       = ModelType::GPT;
    spec.vocab_size = vocab_size;
    spec.d_model    = d_model;
    spec.seq_len    = seq_len;
    spec.num_heads  = num_heads;
    spec.d_ff       = d_ff;
    spec.num_layers = num_layers;
    return spec;
}

} // namespace nn

#endif // GPT_COMMON_HPP
