#ifndef GPT_COMMON_HPP
#define GPT_COMMON_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

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

// ── 词级分词器 ────────────────────────────────────────────────────────────
// 将文本按空格分割为单词，构建词表。
// 特殊 token: <unk>=0, <pad>=1, <num>=2
// 词表大小 = min(最高频词数, max_vocab_size)
//
// 使用方式:
//   WordTokenizer tok;
//   tok.build_from_text(text, 10000);  // 训练时：从语料构建词表
//   tok.save_vocab("vocab.txt");
//   tok.load_vocab("vocab.txt");      // 推理时：加载已保存的词表
//
class WordTokenizer
{
public:
    static constexpr std::size_t UNK_ID = 0;
    static constexpr std::size_t PAD_ID = 1;
    static constexpr std::size_t NUM_ID = 2;

private:
    std::vector<std::string> id_to_word_;
    std::unordered_map<std::string, std::size_t> word_to_id_;
    std::string vocab_path_;

    [[nodiscard]] static bool is_numeric(const std::string &s) noexcept
    {
        if (s.empty()) return false;
        for (char c : s)
            if (c < '0' || c > '9') return false;
        return true;
    }

public:
    WordTokenizer() = default;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return id_to_word_.size(); }
    [[nodiscard]] const std::string &vocab_path() const noexcept { return vocab_path_; }

    // 从训练语料构建词表
    void build_from_text(const std::string &text, std::size_t max_vocab_size = 10000)
    {
        std::unordered_map<std::string, std::size_t> freq;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word)
            ++freq[word];

        // 按频率降序排列
        std::vector<std::pair<std::string, std::size_t>> sorted_freq(freq.begin(), freq.end());
        std::sort(sorted_freq.begin(), sorted_freq.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        // 保留最高频的词（留 3 个位置给特殊 token）
        const std::size_t word_slots = max_vocab_size > 3 ? max_vocab_size - 3 : 1000;
        const std::size_t actual = std::min(sorted_freq.size(), word_slots);

        id_to_word_.clear();
        word_to_id_.clear();
        id_to_word_.reserve(actual + 3);
        word_to_id_.reserve(actual + 3);

        // 特殊 token
        id_to_word_.push_back("<unk>");  word_to_id_["<unk>"] = UNK_ID;
        id_to_word_.push_back("<pad>");  word_to_id_["<pad>"] = PAD_ID;
        id_to_word_.push_back("<num>");  word_to_id_["<num>"] = NUM_ID;

        // 普通词
        for (std::size_t i = 0; i < actual; ++i)
        {
            const std::size_t id = id_to_word_.size();
            id_to_word_.push_back(sorted_freq[i].first);
            word_to_id_[sorted_freq[i].first] = id;
        }
    }

    // 保存词表到文件
    void save_vocab(const std::string &path) const
    {
        std::ofstream ofs(path);
        if (!ofs) throw std::runtime_error("Cannot write vocab file: " + path);
        for (std::size_t i = 0; i < id_to_word_.size(); ++i)
            ofs << i << '\t' << id_to_word_[i] << '\n';
    }

    // 从文件加载词表
    void load_vocab(const std::string &path)
    {
        std::ifstream ifs(path);
        if (!ifs) throw std::runtime_error("Cannot read vocab file: " + path);

        id_to_word_.clear();
        word_to_id_.clear();

        std::string line;
        while (std::getline(ifs, line))
        {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string word = line.substr(tab + 1);
            std::size_t id = id_to_word_.size();
            id_to_word_.push_back(word);
            word_to_id_[word] = id;
        }
        vocab_path_ = path;
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const
    {
        std::vector<std::size_t> tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word)
        {
            if (is_numeric(word))
                tokens.push_back(NUM_ID);
            else if (auto it = word_to_id_.find(word); it != word_to_id_.end())
                tokens.push_back(it->second);
            else
                tokens.push_back(UNK_ID);
        }
        return tokens;
    }

    [[nodiscard]] std::string decode(const std::vector<std::size_t> &tokens) const
    {
        std::string result;
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            if (i > 0) result += ' ';
            if (tokens[i] < id_to_word_.size())
                result += id_to_word_[tokens[i]];
            else
                result += "<unk>";
        }
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
