#ifndef GPT_COMMON_HPP
#define GPT_COMMON_HPP

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "model.hpp"
#include "layer.hpp"

namespace nn {

// ── GPT 默认超参数 ────────────────────────────────────────────────────────
inline constexpr std::size_t GPT_VOCAB_SIZE    = 128;  // ASCII 字符集
inline constexpr std::size_t GPT_D_MODEL       = 128;
inline constexpr std::size_t GPT_NUM_HEADS     = 4;
inline constexpr std::size_t GPT_D_FF          = 512;
inline constexpr std::size_t GPT_NUM_LAYERS    = 4;
inline constexpr std::size_t GPT_SEQ_LEN       = 256;

// ── 字符级词表 ────────────────────────────────────────────────────────────
// 简单的字符级 tokenizer：每个 ASCII 字符对应一个 token ID。
// 词表大小固定为 128（覆盖所有 ASCII 可打印字符和控制字符）。
class CharTokenizer
{
public:
    [[nodiscard]] std::size_t vocab_size() const noexcept { return GPT_VOCAB_SIZE; }

    [[nodiscard]] std::size_t encode_one(char c) const noexcept
    {
        return static_cast<std::size_t>(static_cast<unsigned char>(c)) % GPT_VOCAB_SIZE;
    }

    [[nodiscard]] std::string decode_one(std::size_t id) const
    {
        if (id < GPT_VOCAB_SIZE)
            return std::string(1, static_cast<char>(id));
        return "?";
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const
    {
        std::vector<std::size_t> tokens;
        tokens.reserve(text.size());
        for (char c : text)
            tokens.push_back(encode_one(c));
        return tokens;
    }

    [[nodiscard]] std::string decode(const std::vector<std::size_t> &tokens) const
    {
        std::string result;
        result.reserve(tokens.size());
        for (auto id : tokens)
            result += decode_one(id);
        return result;
    }
};

// ── 从文本文件加载语料 ────────────────────────────────────────────────────
[[nodiscard]] inline std::string load_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Cannot open text file: " + path);
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

} // namespace nn

#endif // GPT_COMMON_HPP
