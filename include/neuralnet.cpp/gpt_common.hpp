#ifndef GPT_COMMON_HPP
#define GPT_COMMON_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <expected>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <utility>

#include "model.hpp"
#include "layer.hpp"

namespace nn {

// ── GPT 默认超参数 ────────────────────────────────────────────────────────
inline constexpr std::size_t GPT_VOCAB_SIZE    = 10000; // 频率过滤词表
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

// ── BPE 分词器 ────────────────────────────────────────────────────────────
// 加载 Python 预处理生成的 gpt_vocab.json 词表。
// 词表构成: <unk>=0, <pad>=1, <num>=2, ASCII 3-130, 高频词 131+
//
// 编码: 先按空格分词，词表中有→直接查 ID，没有→逐字符拼 ASCII ID
// 解码: ID < 128 → 字符, ID >= 128 → 词（空格分隔）
//
class BPETokenizer
{
public:
    static constexpr std::size_t UNK_ID = 0;
    static constexpr std::size_t PAD_ID = 1;
    static constexpr std::size_t NUM_ID = 2;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> token_to_id_;
    std::string vocab_path_;

    [[nodiscard]] static bool is_numeric(const std::string &s) noexcept
    {
        if (s.empty()) return false;
        for (char c : s)
            if (c < '0' || c > '9') return false;
        return true;
    }

    // 判断 token 是否为多字节子词（需要空格分隔）
    // 单字节 token (ID 4-259) 直接拼接，多字节子词 (ID 260+) 空格分隔
    [[nodiscard]] bool is_word_token(std::size_t id) const noexcept
    {
        return id < id_to_token_.size() && id_to_token_[id].size() > 1;
    }

public:
    BPETokenizer() = default;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return id_to_token_.size(); }
    [[nodiscard]] const std::string &vocab_path() const noexcept { return vocab_path_; }

    // 加载词表 JSON（支持新格式: {"vocab": {"4": "hex", "5": "hex", ...}}）
    Result<void> load_vocab(const std::string &path)
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read vocab file: " + path});

        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());

        id_to_token_.clear();
        token_to_id_.clear();

        // 找 "vocab" 对象
        auto vocab_pos = content.find("\"vocab\"");
        if (vocab_pos == std::string::npos)
            return std::unexpected(Error{"Invalid vocab JSON: missing \"vocab\" key"});

        auto brace_pos = content.find('{', vocab_pos);
        if (brace_pos == std::string::npos)
            return std::unexpected(Error{"Invalid vocab JSON: missing opening brace"});

        // 逐对解析
        std::size_t pos = brace_pos + 1;
        while (pos < content.size())
        {
            // 跳过空白和逗号
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
                   content[pos] == '\r' || content[pos] == '\t' || content[pos] == ','))
                ++pos;
            if (pos >= content.size() || content[pos] == '}') break;

            // 解析 key（带引号的字符串）
            if (content[pos] != '"') { ++pos; continue; }
            std::string key;
            ++pos;
            while (pos < content.size() && content[pos] != '"')
            {
                if (content[pos] == '\\' && pos + 1 < content.size())
                {
                    ++pos;
                    switch (content[pos])
                    {
                        case 'n': key += '\n'; break;
                        case 't': key += '\t'; break;
                        case 'r': key += '\r'; break;
                        case '\\': key += '\\'; break;
                        case '"': key += '"'; break;
                        case 'u':
                        {
                            if (pos + 4 < content.size())
                            {
                                std::string hex = content.substr(pos + 1, 4);
                                unsigned long cp = std::stoul(hex, nullptr, 16);
                                if (cp < 0x80) key += static_cast<char>(cp);
                                else if (cp < 0x800) {
                                    key += static_cast<char>(0xC0 | (cp >> 6));
                                    key += static_cast<char>(0x80 | (cp & 0x3F));
                                } else {
                                    key += static_cast<char>(0xE0 | (cp >> 12));
                                    key += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                    key += static_cast<char>(0x80 | (cp & 0x3F));
                                }
                                pos += 4;
                            }
                            break;
                        }
                        default: key += content[pos]; break;
                    }
                }
                else
                {
                    key += content[pos];
                }
                ++pos;
            }
            ++pos; // 跳过结束引号

            // 跳过 : 和空白
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == ':'))
                ++pos;

            // 检测 value 类型：数字 → 旧格式；引号 → 新格式（hex 字符串）
            if (pos < content.size() && content[pos] == '"')
            {
                // 新格式: "id": "hex_bytes" — key 是 ID，value 是 hex 编码的字节
                std::string hex_val;
                ++pos;
                while (pos < content.size() && content[pos] != '"')
                    hex_val += content[pos++];
                ++pos;

                std::size_t id = static_cast<std::size_t>(std::stoul(key));

                // hex → bytes → string
                std::string token;
                for (std::size_t i = 0; i + 1 < hex_val.size(); i += 2)
                {
                    unsigned byte = std::stoul(hex_val.substr(i, 2), nullptr, 16);
                    token += static_cast<char>(byte);
                }

                if (id >= id_to_token_.size())
                    id_to_token_.resize(id + 1);
                id_to_token_[id] = token;
                token_to_id_[token] = id;
            }
            else
            {
                // 旧格式: "token": id — key 是 token 字符串，value 是 ID
                std::string val_str;
                while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9')
                    val_str += content[pos++];
                if (val_str.empty()) continue;

                std::size_t id = static_cast<std::size_t>(std::stoul(val_str));
                if (id >= id_to_token_.size())
                    id_to_token_.resize(id + 1);
                id_to_token_[id] = key;
                token_to_id_[key] = id;
            }
        }

        vocab_path_ = path;
        return {};
    }

    // 编码：先按空格分词，逐词查表；未命中则逐字符兜底
    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const
    {
        std::vector<std::size_t> tokens;
        std::istringstream iss(text);
        std::string word;

        while (iss >> word)
        {
            if (is_numeric(word))
            {
                tokens.push_back(NUM_ID);
            }
            else if (auto it = token_to_id_.find(word); it != token_to_id_.end())
            {
                tokens.push_back(it->second);
            }
            else
            {
                // 逐字符兜底（每个字符都一定能找到，因为 ASCII 在词表中）
                for (char c : word)
                {
                    std::string s(1, c);
                    if (auto cit = token_to_id_.find(s); cit != token_to_id_.end())
                        tokens.push_back(cit->second);
                    else
                        tokens.push_back(UNK_ID);
                }
            }
        }
        return tokens;
    }

    // 解码：词 token 空格分隔，字符 token 直接拼接
    [[nodiscard]] std::string decode(const std::vector<std::size_t> &tokens) const
    {
        std::string result;

        for (auto id : tokens)
        {
            if (id < id_to_token_.size())
            {
                if (is_word_token(id))
                {
                    // 词 token：空格分隔
                    if (!result.empty()) result += ' ';
                    result += id_to_token_[id];
                }
                else
                {
                    // 字符 token：直接拼接
                    result += id_to_token_[id];
                }
            }
            else
            {
                if (!result.empty()) result += ' ';
                result += "<unk>";
            }
        }
        return result;
    }
};

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
