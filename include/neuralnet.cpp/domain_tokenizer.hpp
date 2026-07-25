#ifndef NN_DOMAIN_TOKENIZER_HPP
#define NN_DOMAIN_TOKENIZER_HPP

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config.hpp"

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  Tokenizer — 抽象基类
// ═══════════════════════════════════════════════════════════════════════════

class Tokenizer
{
public:
    virtual ~Tokenizer() = default;

    [[nodiscard]] virtual std::vector<std::size_t> encode(const std::string &text) const = 0;
    [[nodiscard]] virtual std::string decode(std::span<const std::size_t> ids) const = 0;
    [[nodiscard]] virtual std::size_t vocab_size() const noexcept = 0;
    [[nodiscard]] virtual const std::vector<std::string> &vocab() const noexcept = 0;
    [[nodiscard]] virtual Result<void> save(const std::string &path) const = 0;
    [[nodiscard]] virtual Result<void> load(const std::string &path) = 0;

    // ── 特殊 token ID 统一接口 ────────────────────────────────────────
    // 默认返回 npos（表示该 tokenizer 无此特殊 token）。
    // 子类根据自身词表布局重写。
    // 训练/推理入口通过这两个接口获取 BOS/EOS，无需硬编码常量。
    [[nodiscard]] virtual std::size_t bos_id() const noexcept { return npos; }
    [[nodiscard]] virtual std::size_t eos_id() const noexcept { return npos; }

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // ── 从 JSON 字符串加载（纯内存解析，零磁盘 I/O） ──────────────
    // 子类重写此方法以支持从 JSON 字符串直接加载词表。
    // 默认实现：先写临时文件再调用 load(path)，子类应直接解析字符串。
    [[nodiscard]] virtual Result<void> load_from_string(const std::string &json_content) = 0;

    // ── 从 JSON 字符串加载（公共入口，委托给虚方法） ──────────────
    [[nodiscard]] Result<void> load_json(const std::string &json_content)
    {
        return load_from_string(json_content);
    }
};


// ═══════════════════════════════════════════════════════════════════════════
//  WordZip 词级分词器
// ═══════════════════════════════════════════════════════════════════════════
//  基于词频统计的词级分词器。
//  Header-only，零外部依赖。
//
//  词表构成:
//    ID 0~3    特殊 token (<pad>, <unk>, <bos>, <eos>)
//    ID 4~259  256 个单字节（用于未知字符回退）
//    ID 260+   高频完整词
//
//  训练: 预分词 → 词频统计 → 按频次降序填充词表
//  编码: 预分词 → 逐词查表 → 未知词逐字符回退到单字节 token
//  JSON 格式与原 ByteZip 保持兼容。
// ═══════════════════════════════════════════════════════════════════════════

class WordZipTokenizer : public Tokenizer
{
public:
    // ── 特殊 token ────────────────────────────────────────────────────
    static constexpr std::size_t  BYTE_OFFSET = 4;
    static constexpr std::size_t  PAD_ID      = 0;
    static constexpr std::size_t  UNK_ID      = 1;
    static constexpr std::size_t  BOS_ID      = 2;
    static constexpr std::size_t  EOS_ID      = 3;

    // ── 训练默认参数 ──────────────────────────────────────────────────
    static constexpr std::size_t   DEFAULT_VOCAB_SIZE   = 20000;
    static constexpr std::size_t   DEFAULT_V1_MAX_LEN   = 16;
    static constexpr std::uint32_t DEFAULT_MIN_FREQ     = 2;

    // ── 日志回调（默认 cout，nullptr 静默） ────────────────────────────
    using LogFn = std::function<void(std::string_view)>;

    // ── 训练配置 ──────────────────────────────────────────────────────
    struct Config
    {
        std::size_t   vocab_size = DEFAULT_VOCAB_SIZE;
        std::uint32_t min_freq   = DEFAULT_MIN_FREQ;
        LogFn         log        = nullptr;
    };

    // ── 词缀分析结果 ─────────────────────────────────────────────────
    struct AffixInfo {
        std::string affix;          // 词缀本身
        std::size_t count;          // 包含此词缀的词数
        std::size_t stem_found;     // 词根也在词表中的词数
    };

    struct WordSplit {
        std::string word;           // 原词
        std::string stem;           // 词根
        std::string affix;          // 词缀
        bool        is_suffix;      // true=后缀, false=前缀
    };

    // ══════════════════════════════════════════════════════════════════
    //  公开接口
    // ══════════════════════════════════════════════════════════════════

    WordZipTokenizer() = default;

    [[nodiscard]] std::size_t bos_id() const noexcept override { return BOS_ID; }
    [[nodiscard]] std::size_t eos_id() const noexcept override { return EOS_ID; }

    Result<void> train(const std::string &text) { return train(text, Config{}); }

    // ── 词频训练 ──────────────────────────────────────────────────────
    Result<void> train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view) { /* 静默 */ }};

        const auto min_freq = config.min_freq;
        const auto target   = config.vocab_size;

        // ── 预分词 → 词频统计 ────────────────────────────────────────
        log("\n[训练] 预分词...");
        auto chunks = pre_tokenize(text);

        std::unordered_map<std::string, std::uint32_t> freq;
        freq.reserve(chunks.size());
        for (const auto &chunk : chunks)
            ++freq[chunk];
        log("  不同词数: " + std::to_string(freq.size()));

        // 按频次降序排序
        struct Entry { std::string word; std::uint32_t cnt; };
        std::vector<Entry> sorted;
        sorted.reserve(freq.size());
        for (const auto &[w, c] : freq)
            sorted.push_back({w, c});
        std::ranges::sort(sorted, [](const Entry &a, const Entry &b)
        { return a.cnt > b.cnt; });

        // 初始化词表：特殊 token + 256 单字节
        vocab_.clear();
        vocab_.resize(BYTE_OFFSET);
        for (std::size_t i = 0; i < 256; ++i)
            vocab_.emplace_back(1, static_cast<char>(i));

        // 按频次填充词表
        std::size_t added = 0;
        for (const auto &e : sorted)
        {
            if (vocab_.size() >= target) break;
            if (e.cnt < min_freq) break;
            if (e.word.size() <= 1) continue;   // 单字符已在单字节表中
            vocab_.push_back(e.word);
            ++added;
            if (added % 1000 == 0)
                log("  已添加 " + std::to_string(added) + " 个词条（总 "
                    + std::to_string(vocab_.size()) + "）");
        }

        // ── 最终查找表 ───────────────────────────────────────────────
        log("\n最终词表: " + std::to_string(vocab_.size()));
        max_subword_len_ = 0;
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
            if (vocab_[i].size() > max_subword_len_)
                max_subword_len_ = vocab_[i].size();
        build_lookup();

        // ── 词缀分析 + 词分解 ───────────────────────────────────────
        analyze_affixes();
        build_decomposition();

        // ── 删除可分解词 ───────────────────────────────────────────
        std::size_t removed = 0;
        std::unordered_set<std::string> to_remove;
        for (const auto &[word, parts] : decompose_map_)
            to_remove.insert(word);

        // 从 vocab_ 中移除可分解词（保留 ID 槽，置空）
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
        {
            if (to_remove.contains(vocab_[i]))
            {
                vocab_[i].clear();
                ++removed;
            }
        }

        // 重建查找表（移除的词不再出现在 lookup_ 中）
        build_lookup();

        log("  已移除可分解词: " + std::to_string(removed) + " 个");
        log("最终词表（有效）: " + std::to_string(vocab_.size() - removed));
        return {};
    }

    // ── 编码 ──────────────────────────────────────────────────────────
    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
    {
        std::vector<std::size_t> all;
        all.reserve(text.size() / 2);
        for (const auto &chunk : pre_tokenize(text))
        {
            auto ids = encode_bytes(chunk);
            all.insert(all.end(), ids.begin(), ids.end());
        }
        return all;
    }

    // ── 解码 ──────────────────────────────────────────────────────────
    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string raw;
        raw.reserve(ids.size() * 2);
        for (auto tid : ids)
            raw += tid < vocab_.size() ? vocab_[tid] : std::string{"\xef\xbf\xbd"};
        return raw;
    }

    // ── 保存 JSON ────────────────────────────────────────────────────
    [[nodiscard]] Result<void> save(const std::string &path) const override
    {
        std::ofstream ofs(path);
        if (!ofs) return std::unexpected(Error{"Cannot write: " + path});

        ofs << "{\n  \"type\": \"wordzip_tokenizer\",\n  \"vocab\": {\n";
        bool first = true;
        for (std::size_t tid = BYTE_OFFSET; tid < vocab_.size(); ++tid)
        {
            if (vocab_[tid].empty()) continue;
            if (!first) ofs << ",\n";
            first = false;
            ofs << "    \"" << std::dec << tid << "\": \"";
            for (unsigned char b : vocab_[tid])
                ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
            ofs << "\"" << std::dec;
        }
        ofs << "\n  },\n  \"vocab_size\": " << std::dec << vocab_.size()
            << ",\n  \"byte_offset\": " << BYTE_OFFSET
            << ",\n  \"max_subword_len\": " << max_subword_len_
            << ",\n  \"decompose\": {";
        {
            bool f = true;
            for (const auto &[word, parts] : decompose_map_)
            {
                ofs << (f ? "\n" : ",\n");
                f = false;
                ofs << "    \"";
                for (unsigned char b : word)
                    ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
                ofs << "\": [";
                for (std::size_t pi = 0; pi < parts.size(); ++pi)
                {
                    if (pi > 0) ofs << ", ";
                    ofs << "\"";
                    for (unsigned char b : parts[pi])
                        ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
                    ofs << "\"";
                }
                ofs << "]";
            }
        }
        ofs << "\n  },\n  \"special_tokens\": {"
            << "\n    \"<pad>\": 0,\n    \"<unk>\": 1,"
            << "\n    \"<bos>\": 2,\n    \"<eos>\": 3\n  }\n}\n";
        return {};
    }

    // ── 加载 JSON ────────────────────────────────────────────────────
    [[nodiscard]] Result<void> load(const std::string &path) override
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read: " + path});
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        return load_from_string(content);
    }

    [[nodiscard]] Result<void> load_from_string(const std::string &content) override
    {
        vocab_.clear();

        if (auto p = content.find("\"vocab_size\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::size_t v = 0;
                if (auto [ptr, ec] = std::from_chars(content.data()+c, content.data()+content.size(), v); ec == std::errc{})
                    vocab_.resize(v);
            }

        auto vp = content.find("\"vocab\"");
        if (vp == std::string::npos)
            return std::unexpected(Error{"Invalid JSON: missing \"vocab\""});
        auto br = content.find('{', vp);
        if (br == std::string::npos)
            return std::unexpected(Error{"Invalid JSON: missing '{'"});

        std::size_t pos = br + 1;
        while (pos < content.size())
        {
            while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                   content[pos]=='\r'||content[pos]=='\t'||content[pos]==',')) ++pos;
            if (pos >= content.size() || content[pos] == '}') break;
            if (content[pos] != '"') { ++pos; continue; }
            ++pos;
            std::string key;
            while (pos < content.size() && content[pos] != '"') key += content[pos++];
            ++pos;
            while (pos < content.size() && (content[pos]==' '||content[pos]==':')) ++pos;
            if (pos < content.size() && content[pos] == '"')
            {
                ++pos;
                std::string hex;
                while (pos < content.size() && content[pos] != '"') hex += content[pos++];
                ++pos;
                std::size_t id = 0;
                std::from_chars(key.data(), key.data()+key.size(), id);
                std::string tok;
                tok.reserve(hex.size()/2);
                for (std::size_t i = 0; i+1 < hex.size(); i += 2)
                { unsigned b = 0; std::from_chars(hex.data()+i, hex.data()+i+2, b, 16); tok += static_cast<char>(b); }
                if (id >= vocab_.size()) vocab_.resize(id+1);
                vocab_[id] = std::move(tok);
            }
        }

        max_subword_len_ = DEFAULT_V1_MAX_LEN;
        if (auto p = content.find("\"max_subword_len\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::size_t v = 0;
                std::from_chars(content.data()+c, content.data()+content.size(), v);
                if (v > 0) max_subword_len_ = v;
            }

        build_lookup();

        // ── 加载分解规则 ──────────────────────────────────────────────
        decompose_map_.clear();
        if (auto dp = content.find("\"decompose\""); dp != std::string::npos)
        {
            auto dbr = content.find('{', dp);
            if (dbr != std::string::npos)
            {
                std::size_t dpos = dbr + 1;
                while (dpos < content.size())
                {
                    while (dpos < content.size() && (content[dpos]==' '||content[dpos]=='\n'||
                           content[dpos]=='\r'||content[dpos]=='\t'||content[dpos]==',')) ++dpos;
                    if (dpos >= content.size() || content[dpos] == '}') break;
                    if (content[dpos] != '"') { ++dpos; continue; }
                    ++dpos;
                    std::string key_hex;
                    while (dpos < content.size() && content[dpos] != '"') key_hex += content[dpos++];
                    ++dpos;
                    // skip to '['
                    while (dpos < content.size() && content[dpos] != '[') ++dpos;
                    ++dpos;
                    // read array elements
                    std::vector<std::string> parts;
                    while (dpos < content.size())
                    {
                        while (dpos < content.size() && (content[dpos]==' '||content[dpos]=='\n'||
                               content[dpos]=='\r'||content[dpos]=='\t'||content[dpos]==',')) ++dpos;
                        if (dpos >= content.size() || content[dpos] == ']') break;
                        if (content[dpos] == '"')
                        {
                            ++dpos;
                            std::string hex;
                            while (dpos < content.size() && content[dpos] != '"') hex += content[dpos++];
                            ++dpos;
                            std::string part;
                            for (std::size_t h = 0; h+1 < hex.size(); h += 2)
                            { unsigned b = 0; std::from_chars(hex.data()+h, hex.data()+h+2, b, 16); part += static_cast<char>(b); }
                            parts.push_back(std::move(part));
                        }
                        else ++dpos;
                    }
                    ++dpos; // skip ']'
                    std::string word;
                    for (std::size_t h = 0; h+1 < key_hex.size(); h += 2)
                    { unsigned b = 0; std::from_chars(key_hex.data()+h, key_hex.data()+h+2, b, 16); word += static_cast<char>(b); }
                    if (!parts.empty())
                        decompose_map_[std::move(word)] = std::move(parts);
                }
            }
        }

        analyze_affixes();
        return {};
    }

    // ══════════════════════════════════════════════════════════════════
    //  访问器
    // ══════════════════════════════════════════════════════════════════

    [[nodiscard]] constexpr std::size_t vocab_size()      const noexcept override { return vocab_.size(); }
    [[nodiscard]] constexpr std::size_t max_subword_len() const noexcept { return max_subword_len_; }
    [[nodiscard]] constexpr std::size_t byte_offset()     const noexcept { return BYTE_OFFSET; }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept override { return vocab_; }

    [[nodiscard]] Result<std::string> token_bytes(std::size_t id) const
    {
        if (id >= vocab_.size()) return std::unexpected(Error{"Token ID out of range"});
        return vocab_[id];
    }

    [[nodiscard]] std::string try_decode_token(std::size_t id) const
    {
        if (id >= vocab_.size()) return "<invalid>";
        std::string r;
        for (char c : vocab_[id])
        {
            if      (c >= 32 && c < 127) r += c;
            else if (c == '\n')          r += "\\n";
            else if (c == '\t')          r += "\\t";
            else if (c == ' ')           r += "␣";
            else { r += "\\x"; r += "0123456789abcdef"[(static_cast<unsigned char>(c)>>4)&0xF];
                                  r += "0123456789abcdef"[ static_cast<unsigned char>(c)   &0xF]; }
        }
        return r;
    }

    // ── 词缀分析（纯数据驱动，无硬编码词缀） ────────────────────────
    void analyze_affixes()
    {
        suffixes_.clear();
        prefixes_.clear();
        splits_.clear();

        // 构建词表集合（O(1) 查找）
        std::unordered_set<std::string> vocab_set;
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
            if (!vocab_[i].empty() && vocab_[i].size() >= 2)
                vocab_set.insert(vocab_[i]);

        // ── 后缀检测 ────────────────────────────────────────────────
        // 遍历每个词的所有可能后缀(长度2~5)，统计频率与词根覆盖率
        std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> suffix_stats;
        for (const auto &word : vocab_)
        {
            if (word.size() < MIN_WORD_LEN) continue;
            if (!std::ranges::all_of(word, is_alpha)) continue;

            for (std::size_t slen = MIN_AFFIX_LEN;
                 slen <= std::min(MAX_AFFIX_LEN, word.size() - MIN_AFFIX_LEN); ++slen)
            {
                auto stem   = word.substr(0, word.size() - slen);
                auto suffix = word.substr(word.size() - slen);
                auto &[count, stem_found] = suffix_stats[suffix];
                ++count;
                if (vocab_set.contains(stem))
                    ++stem_found;
            }
        }

        for (const auto &[suffix, stats] : suffix_stats)
        {
            auto [count, stem_found] = stats;
            if (count >= MIN_AFFIX_COUNT &&
                static_cast<double>(stem_found) / count >= MIN_COVERAGE)
                suffixes_.push_back({suffix, count, stem_found});
        }
        std::ranges::sort(suffixes_, [](const AffixInfo &a, const AffixInfo &b)
        { return a.count > b.count; });

        // ── 前缀检测 ────────────────────────────────────────────────
        // 同理统计词首 n-gram 的频率与词根覆盖率
        std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> prefix_stats;
        for (const auto &word : vocab_)
        {
            if (word.size() < MIN_WORD_LEN) continue;
            if (!std::ranges::all_of(word, is_alpha)) continue;

            for (std::size_t plen = MIN_AFFIX_LEN;
                 plen <= std::min(std::size_t{4}, word.size() - MIN_AFFIX_LEN); ++plen)
            {
                auto prefix = word.substr(0, plen);
                auto stem   = word.substr(plen);
                auto &[count, stem_found] = prefix_stats[prefix];
                ++count;
                if (vocab_set.contains(stem))
                    ++stem_found;
            }
        }

        for (const auto &[prefix, stats] : prefix_stats)
        {
            auto [count, stem_found] = stats;
            if (count >= MIN_AFFIX_COUNT &&
                static_cast<double>(stem_found) / count >= MIN_COVERAGE)
                prefixes_.push_back({prefix, count, stem_found});
        }
        std::ranges::sort(prefixes_, [](const AffixInfo &a, const AffixInfo &b)
        { return a.count > b.count; });

        // ── 词拆分 ──────────────────────────────────────────────────
        // 对每个词：优先匹配最长后缀 → 其次匹配最长前缀
        for (const auto &word : vocab_)
        {
            if (word.size() < MIN_WORD_LEN) continue;
            if (!std::ranges::all_of(word, is_alpha)) continue;

            // 优先尝试后缀（从最长到最短）
            bool found = false;
            for (const auto &affix : suffixes_)
            {
                if (word.size() <= affix.affix.size()) continue;
                if (word.ends_with(affix.affix))
                {
                    auto stem = word.substr(0, word.size() - affix.affix.size());
                    if (stem.size() >= 2 && vocab_set.contains(stem))
                    {
                        splits_.push_back({word, stem, affix.affix, true});
                        found = true;
                        break;
                    }
                }
            }
            if (found) continue;

            // 尝试前缀
            for (const auto &affix : prefixes_)
            {
                if (word.size() <= affix.affix.size()) continue;
                if (word.starts_with(affix.affix))
                {
                    auto stem = word.substr(affix.affix.size());
                    if (stem.size() >= 2 && vocab_set.contains(stem))
                    {
                        splits_.push_back({word, stem, affix.affix, false});
                        break;
                    }
                }
            }
        }
    }

    // ── 词分解：检测词表中可拆分为两个词表词的组合 ──────────────────
    // 纯数据驱动：只要 A+B 都在词表中且长度合理，就记录为可分解
    void build_decomposition()
    {
        decompose_map_.clear();

        // 构建词表集合
        std::unordered_set<std::string> vocab_set;
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
            if (!vocab_[i].empty())
                vocab_set.insert(vocab_[i]);

        for (const auto &word : vocab_)
        {
            if (word.size() < 4) continue;

            // 剥离前导非字母（如空格），保留前缀字符串以便还原
            std::string prefix;
            std::string word_body;

            if (!is_alpha(word.front()))
            {
                auto first_alpha = static_cast<std::size_t>(std::ranges::distance(
                    word.begin(), std::ranges::find_if(word, is_alpha)));
                if (first_alpha > 0 && first_alpha < word.size())
                {
                    prefix    = word.substr(0, first_alpha);
                    word_body = word.substr(first_alpha);
                }
                else
                {
                    word_body = word;
                }
            }
            else
            {
                word_body = word;
            }

            if (word_body.size() < 4 || !std::ranges::all_of(word_body, is_alpha))
                continue;

            // 找最优二分：偏好更长、更均衡的拆分
            std::size_t best_min = 0;
            std::vector<std::string> best_parts;

            for (std::size_t pos = 1; pos < word_body.size(); ++pos)
            {
                auto a = word_body.substr(0, pos);
                auto b = word_body.substr(pos);

                if (vocab_set.contains(a) && vocab_set.contains(b))
                {
                    auto m = std::min(pos, word_body.size() - pos);
                    if (m >= 2 && m > best_min)
                    {
                        best_min = m;
                        // 还原：前缀附加到第一部分
                        best_parts = {prefix + a, b};
                    }
                }
            }

            if (!best_parts.empty())
                decompose_map_[word] = std::move(best_parts);
        }
    }

    // ── 打印词缀分析报告 ──────────────────────────────────────────────
    void print_affix_report(std::ostream &os = std::cout) const
    {
        os << "\n══════════════════════════════════════════════\n";
        os << "  词缀分析报告\n";
        os << "══════════════════════════════════════════════\n";

        // ── 后缀表 ──────────────────────────────────────────────────
        os << "\n后缀表 (共 " << suffixes_.size() << " 个):\n";
        for (const auto &s : suffixes_)
        {
            std::string examples;
            std::size_t n = 0;
            for (const auto &sp : splits_)
            {
                if (sp.is_suffix && sp.affix == s.affix)
                {
                    if (n > 0) examples += ", ";
                    examples += sp.stem + "+" + sp.affix;
                    if (++n >= 3) break;
                }
            }
            double cov = 100.0 * s.stem_found / s.count;
            os << "  " << s.affix << "  ×" << s.count
               << "  覆盖" << std::fixed << std::setprecision(1) << cov << "%"
               << "  例: " << examples << "\n";
        }

        // ── 前缀表 ──────────────────────────────────────────────────
        os << "\n前缀表 (共 " << prefixes_.size() << " 个):\n";
        for (const auto &p : prefixes_)
        {
            std::string examples;
            std::size_t n = 0;
            for (const auto &sp : splits_)
            {
                if (!sp.is_suffix && sp.affix == p.affix)
                {
                    if (n > 0) examples += ", ";
                    examples += p.affix + "+" + sp.stem;
                    if (++n >= 3) break;
                }
            }
            double cov = 100.0 * p.stem_found / p.count;
            os << "  " << p.affix << "  ×" << p.count
               << "  覆盖" << std::fixed << std::setprecision(1) << cov << "%"
               << "  例: " << examples << "\n";
        }

        // ── 词拆分表 ────────────────────────────────────────────────
        os << "\n词拆分表 (共 " << splits_.size() << " 个):\n";
        for (const auto &sp : splits_)
        {
            os << "  " << sp.word << " → " << sp.stem
               << " + " << (sp.is_suffix ? "" : "[前缀]") << sp.affix << "\n";
        }

        // ── 词分解规则 ────────────────────────────────────────────────
        os << "\n词分解规则 (共 " << decompose_map_.size() << " 个):\n";
        for (const auto &[word, parts] : decompose_map_)
        {
            os << "  " << word << " → ";
            for (std::size_t i = 0; i < parts.size(); ++i)
            {
                if (i > 0) os << " + ";
                os << parts[i];
            }
            os << "\n";
        }

        os << "\n══════════════════════════════════════════════\n";
    }

    // ── 访问器（词缀分析结果） ────────────────────────────────────────
    [[nodiscard]] const std::vector<AffixInfo> &suffixes() const noexcept { return suffixes_; }
    [[nodiscard]] const std::vector<AffixInfo> &prefixes() const noexcept { return prefixes_; }
    [[nodiscard]] const std::vector<WordSplit> &splits()    const noexcept { return splits_; }
    [[nodiscard]] const std::unordered_map<std::string, std::vector<std::string>> &decompose_map() const noexcept { return decompose_map_; }

    // ── 预分词（公开，供外部复用） ────────────────────────────────────
    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view text)
    {
        std::vector<std::string> chunks;
        chunks.reserve(text.size() / 4);
        std::size_t pos = 0;
        while (pos < text.size())
        {
            if (auto n = try_contraction(text, pos); n > 0)
            { chunks.emplace_back(text.substr(pos, n)); pos += n; continue; }

            const char c = text[pos];

            if (is_alpha(c) || (c==' ' && pos+1<text.size() && is_alpha(text[pos+1])))
            { auto s=pos; if(c==' ')++pos; while(pos<text.size()&&is_alpha(text[pos]))++pos;
              chunks.emplace_back(text.substr(s,pos-s)); continue; }

            if (is_digit(c) || (c==' ' && pos+1<text.size() && is_digit(text[pos+1])))
            { auto s=pos; if(c==' ')++pos; while(pos<text.size()&&is_digit(text[pos]))++pos;
              chunks.emplace_back(text.substr(s,pos-s)); continue; }

            if (!is_whitespace(c) && !is_word_char(c) && c!='\'')
            { auto s=pos; while(pos<text.size()&&!is_whitespace(text[pos])&&
                  !is_word_char(text[pos])&&text[pos]!='\'') ++pos;
              chunks.emplace_back(text.substr(s,pos-s)); continue; }

            if (is_whitespace(c))
            { auto s=pos; while(pos<text.size()&&is_whitespace(text[pos]))++pos;
              chunks.emplace_back(text.substr(s,pos-s)); continue; }

            chunks.emplace_back(text.substr(pos, 1)); ++pos;
        }
        return chunks;
    }

    // ══════════════════════════════════════════════════════════════════
    //  内部实现
    // ══════════════════════════════════════════════════════════════════
private:
    std::vector<std::string> vocab_;
    std::size_t max_subword_len_ = DEFAULT_V1_MAX_LEN;

    // 透明哈希查找表（编码热路径）
    struct TransparentHash {
        using is_transparent = void;
        [[nodiscard]] std::size_t operator()(std::string_view s)  const noexcept { return std::hash<std::string_view>{}(s); }
        [[nodiscard]] std::size_t operator()(const std::string &s) const noexcept { return std::hash<std::string>{}(s); }
    };
    struct TransparentEqual {
        using is_transparent = void;
        [[nodiscard]] bool operator()(std::string_view a, std::string_view b) const noexcept { return a==b; }
    };
    using LookupTable = std::unordered_map<std::string, std::size_t, TransparentHash, TransparentEqual>;
    LookupTable lookup_;

    // ── 词缀分析参数与结果 ────────────────────────────────────────────
    static constexpr std::size_t MIN_AFFIX_LEN   = 2;   // 最短词缀长度
    static constexpr std::size_t MAX_AFFIX_LEN   = 5;   // 最长词缀长度
    static constexpr std::size_t MIN_WORD_LEN    = 4;   // 词根至少 2 字符
    static constexpr std::size_t MIN_AFFIX_COUNT = 3;   // 最少出现词数
    static constexpr double      MIN_COVERAGE    = 0.5;  // 词根覆盖率阈值

    std::vector<AffixInfo>  suffixes_;
    std::vector<AffixInfo>  prefixes_;
    std::vector<WordSplit>  splits_;
    std::unordered_map<std::string, std::vector<std::string>> decompose_map_;

    // ── 字符分类 ──────────────────────────────────────────────────────
    [[nodiscard]] static constexpr bool is_alpha(char c)      noexcept { return (c>='a'&&c<='z')||(c>='A'&&c<='Z'); }
    [[nodiscard]] static constexpr bool is_digit(char c)      noexcept { return c>='0'&&c<='9'; }
    [[nodiscard]] static constexpr bool is_whitespace(char c) noexcept { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
    [[nodiscard]] static constexpr bool is_word_char(char c)  noexcept { return is_alpha(c)||is_digit(c)||c=='_'; }

    [[nodiscard]] static constexpr std::size_t try_contraction(std::string_view t, std::size_t p) noexcept
    {
        if (p>=t.size()||t[p]!='\'') return 0;
        auto r = t.size()-p; if (r<2) return 0;
        auto n1 = t[p+1];
        if (n1=='s'||n1=='t'||n1=='m'||n1=='d') return 2;
        if (r>=3) { auto n2=t[p+2];
            if ((n1=='r'&&n2=='e')||(n1=='v'&&n2=='e')||(n1=='l'&&n2=='l')) return 3; }
        return 0;
    }

    // ── 编码（内部，用于 encode()） ───────────────────────────────────
    // ── decompose_map → byte fallback → 整词查表
    [[nodiscard]] std::vector<std::size_t> encode_bytes(std::string_view data) const
    {
        // 1. 优先查分解规则（被移除的可分解词走这里）
        auto dec_it = decompose_map_.find(std::string(data));
        if (dec_it != decompose_map_.end())
        {
            std::vector<std::size_t> ids;
            for (const auto &part : dec_it->second)
            {
                auto part_ids = encode_bytes(part);
                ids.insert(ids.end(), part_ids.begin(), part_ids.end());
            }
            return ids;
        }

        // 2. 整词匹配
        if (auto it = lookup_.find(data); it != lookup_.end())
            return {it->second};

        // 3. 逐字符回退到单字节 token
        std::vector<std::size_t> ids;
        ids.reserve(data.size());
        for (unsigned char b : data)
            ids.push_back(static_cast<std::size_t>(b) + BYTE_OFFSET);
        return ids;
    }

    // ── 构建查找表 ────────────────────────────────────────────────────
    void build_lookup()
    {
        lookup_.clear();
        lookup_.reserve(vocab_.size() - BYTE_OFFSET);
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
            if (!vocab_[i].empty()) lookup_[vocab_[i]] = i;
    }

};

// ═══════════════════════════════════════════════════════════════════════════
//  SpaceTokenizer — 基于空格的分词器
// ═══════════════════════════════════════════════════════════════════════════
//  训练：按空格分词 → 词频统计 → 截断 top-N 填充词表
//  编码：空格分词 → 查词表 → 未命中则逐字符 ASCII 回退
//  解码：多字节 token 空格分隔，单字节 token 直接拼接
// ═══════════════════════════════════════════════════════════════════════════

class SpaceTokenizer : public Tokenizer
{
public:
    static constexpr std::size_t UNK_ID = 0;
    static constexpr std::size_t PAD_ID = 1;
    static constexpr std::size_t NUM_ID = 2;
    static constexpr std::size_t ASCII_BASE = 3;
    static constexpr std::size_t DEFAULT_VOCAB_SIZE = 10000;
    static constexpr std::uint32_t DEFAULT_MIN_FREQ = 2;

    using LogFn = std::function<void(std::string_view)>;

    struct Config
    {
        std::size_t   vocab_size = DEFAULT_VOCAB_SIZE;
        std::uint32_t min_freq   = DEFAULT_MIN_FREQ;
        LogFn         log        = nullptr;
    };

    SpaceTokenizer() = default;

    Result<void> train(const std::string &text) { return train(text, Config{}); }

    Result<void> train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view) { /* 静默 */ }};

        log("\n[Space 训练] 分词统计...");

        std::unordered_map<std::string, std::uint32_t> freq;
        {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word)
                ++freq[word];
        }
        log("  不同词数: " + std::to_string(freq.size()));

        struct Entry { std::string word; std::uint32_t cnt; };
        std::vector<Entry> sorted;
        sorted.reserve(freq.size());
        for (const auto &[w, c] : freq)
            sorted.push_back({w, c});
        std::ranges::sort(sorted, [](const Entry &a, const Entry &b)
        { return a.cnt > b.cnt; });

        id_to_token_.clear();
        id_to_token_.resize(ASCII_BASE);
        // 修复：原代码只添加 128 个 ASCII 字符，导致非 ASCII 字节（如中文 UTF-8）
        // 在 encode() 中 id >= ASCII_BASE+128 超出词表，全部回退到 UNK_ID，
        // 丢失原文信息。改为添加全部 256 个单字节，使任意 UTF-8 字节都能
        // 映射到唯一 token ID（与 WordZipTokenizer 的字节回退设计一致）。
        for (std::size_t i = 0; i < 256; ++i)
            id_to_token_.emplace_back(1, static_cast<char>(static_cast<unsigned char>(i)));

        std::size_t added = 0;
        for (const auto &e : sorted)
        {
            if (id_to_token_.size() >= config.vocab_size) break;
            if (e.cnt < config.min_freq) break;
            if (e.word.size() <= 1) continue;
            id_to_token_.push_back(e.word);
            if (++added % 1000 == 0)
                log("  已添加 " + std::to_string(added) + " 个词");
        }

        build_lookup();
        log("最终词表: " + std::to_string(id_to_token_.size()));
        return {};
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
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
                for (char c : word)
                {
                    auto id = static_cast<std::size_t>(static_cast<unsigned char>(c)) + ASCII_BASE;
                    tokens.push_back(id < id_to_token_.size() ? id : UNK_ID);
                }
            }
        }
        return tokens;
    }

    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string result;
        for (auto id : ids)
        {
            if (id < id_to_token_.size())
            {
                const auto &tok = id_to_token_[id];
                if (tok.size() > 1) { if (!result.empty()) result += ' '; result += tok; }
                else result += tok;
            }
            else { if (!result.empty()) result += ' '; result += "<unk>"; }
        }
        return result;
    }

    [[nodiscard]] Result<void> save(const std::string &path) const override
    {
        std::ofstream ofs(path);
        if (!ofs) return std::unexpected(Error{"Cannot write: " + path});
        ofs << "{\n  \"type\": \"space_tokenizer\",\n  \"vocab\": {\n";
        bool first = true;
        for (std::size_t tid = ASCII_BASE; tid < id_to_token_.size(); ++tid)
        {
            if (id_to_token_[tid].empty()) continue;
            if (!first) ofs << ",\n";
            first = false;
            ofs << "    \"" << std::dec << tid << "\": \"";
            for (unsigned char b : id_to_token_[tid])
                ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
            ofs << "\"" << std::dec;
        }
        ofs << "\n  },\n  \"vocab_size\": " << std::dec << id_to_token_.size()
            << ",\n  \"special_tokens\": {"
            << "\n    \"<unk>\": 0,\n    \"<pad>\": 1,\n    \"<num>\": 2\n  }\n}\n";
        return {};
    }

    [[nodiscard]] Result<void> load(const std::string &path) override
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read: " + path});
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        return load_from_string(content);
    }

    [[nodiscard]] Result<void> load_from_string(const std::string &content) override
    {
        id_to_token_.clear();
        token_to_id_.clear();

        if (auto p = content.find("\"vocab_size\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::size_t v = 0;
                if (auto [ptr, ec] = std::from_chars(content.data()+c, content.data()+content.size(), v); ec == std::errc{})
                    id_to_token_.resize(v);
            }

        auto vp = content.find("\"vocab\"");
        if (vp == std::string::npos)
            return std::unexpected(Error{"Invalid JSON: missing \"vocab\""});
        auto br = content.find('{', vp);
        if (br == std::string::npos)
            return std::unexpected(Error{"Invalid JSON: missing '{'"});

        std::size_t pos = br + 1;
        while (pos < content.size())
        {
            while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                   content[pos]=='\r'||content[pos]=='\t'||content[pos]==',')) ++pos;
            if (pos >= content.size() || content[pos] == '}') break;
            if (content[pos] != '"') { ++pos; continue; }
            ++pos; std::string key;
            while (pos < content.size() && content[pos] != '"') key += content[pos++];
            ++pos;
            while (pos < content.size() && (content[pos]==' '||content[pos]==':')) ++pos;
            if (pos < content.size() && content[pos] == '"')
            {
                ++pos; std::string hex;
                while (pos < content.size() && content[pos] != '"') hex += content[pos++];
                ++pos;
                std::size_t id = 0;
                std::from_chars(key.data(), key.data()+key.size(), id);
                std::string tok;
                for (std::size_t i = 0; i+1 < hex.size(); i += 2)
                { unsigned b = 0; std::from_chars(hex.data()+i, hex.data()+i+2, b, 16); tok += static_cast<char>(b); }
                if (id >= id_to_token_.size()) id_to_token_.resize(id+1);
                id_to_token_[id] = std::move(tok);
            }
        }
        build_lookup();
        return {};
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept override { return id_to_token_.size(); }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept override { return id_to_token_; }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, std::size_t> token_to_id_;

    [[nodiscard]] static bool is_numeric(const std::string &s) noexcept
    {
        if (s.empty()) return false;
        for (char c : s) if (c < '0' || c > '9') return false;
        return true;
    }

    void build_lookup()
    {
        token_to_id_.clear();
        token_to_id_.reserve(id_to_token_.size());
        for (std::size_t i = 0; i < id_to_token_.size(); ++i)
            if (!id_to_token_[i].empty())
                token_to_id_[id_to_token_[i]] = i;
    }
};


// ═══════════════════════════════════════════════════════════════════════════
//  BPETokenizer — 真正的 Byte-Pair Encoding 分词器
// ═══════════════════════════════════════════════════════════════════════════
//  训练：GPT-2 风格预分词 → 迭代 byte-pair 合并
//  编码：预分词 → 按合并优先级对字节序列进行合并
//  解码：直接拼接字节
// ═══════════════════════════════════════════════════════════════════════════

class BPETokenizer : public Tokenizer
{
public:
    static constexpr std::size_t BYTE_BASE       = 256;
    static constexpr std::size_t BOS_ID          = BYTE_BASE;       // 256
    static constexpr std::size_t EOS_ID          = BYTE_BASE + 1;   // 257
    static constexpr std::size_t DEFAULT_VOCAB_SIZE = 5000;
    static constexpr std::size_t DEFAULT_MIN_FREQ   = 2;

    using LogFn = std::function<void(std::string_view)>;

    struct Config
    {
        std::size_t   vocab_size = DEFAULT_VOCAB_SIZE;
        std::uint32_t min_freq   = DEFAULT_MIN_FREQ;
        LogFn         log        = nullptr;
    };

    static const std::regex &pre_pattern()
    {
        static const std::regex pat(
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+)",
            std::regex::optimize
        );
        return pat;
    }

    BPETokenizer() = default;

    [[nodiscard]] std::size_t bos_id() const noexcept override { return BOS_ID; }
    [[nodiscard]] std::size_t eos_id() const noexcept override { return EOS_ID; }

    Result<void> train(const std::string &text) { return train(text, Config{}); }

    Result<void> train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view) { /* 静默 */ }};

        log("\n[BPE 训练] 预分词...");
        auto chunks = pre_tokenize(text);
        log("  预分词块数: " + std::to_string(chunks.size()));

        std::vector<std::vector<std::size_t>> byte_chunks;
        byte_chunks.reserve(chunks.size());
        std::size_t total_bytes = 0;
        for (const auto &chunk : chunks)
        {
            std::vector<std::size_t> ids;
            ids.reserve(chunk.size());
            for (unsigned char b : chunk)
                ids.push_back(static_cast<std::size_t>(b));
            total_bytes += ids.size();
            byte_chunks.push_back(std::move(ids));
        }
        log("  总字节数: " + std::to_string(total_bytes));

        vocab_.clear();
        vocab_.reserve(config.vocab_size);
        for (std::size_t i = 0; i < BYTE_BASE; ++i)
            vocab_.emplace_back(1, static_cast<char>(i));
        // 预留特殊 token 槽位（BOS=256, EOS=257），BPE 合并从 258 开始
        vocab_.emplace_back("<bos>");
        vocab_.emplace_back("<eos>");

        merges_.clear();
        merges_.reserve(config.vocab_size - BYTE_BASE - 2);

        const std::size_t target_merges = config.vocab_size - BYTE_BASE - 2;
        log("  目标合并数: " + std::to_string(target_merges));

        for (std::size_t round = 0; round < target_merges; ++round)
        {
            std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> pair_freq;
            for (const auto &chunk : byte_chunks)
                for (std::size_t i = 0; i + 1 < chunk.size(); ++i)
                    ++pair_freq[{chunk[i], chunk[i + 1]}];

            if (pair_freq.empty()) break;

            auto best = std::ranges::max_element(pair_freq,
                [](const auto &a, const auto &b) { return a.second < b.second; });

            if (best->second < config.min_freq)
            {
                log("  无更多高频 pair，提前停止于第 " + std::to_string(round) + " 轮");
                break;
            }

            auto [id_a, id_b] = best->first;
            std::size_t new_id = vocab_.size();
            vocab_.push_back(vocab_[id_a] + vocab_[id_b]);
            merges_.push_back({id_a, id_b, new_id});

            for (auto &chunk : byte_chunks)
            {
                // 原地压缩：避免 vector::erase 的 O(n) 移动
                std::size_t write = 0;
                std::size_t read = 0;
                while (read < chunk.size())
                {
                    if (read + 1 < chunk.size() && chunk[read] == id_a && chunk[read + 1] == id_b)
                    {
                        chunk[write++] = new_id;
                        read += 2;
                    }
                    else
                    {
                        chunk[write++] = chunk[read++];
                    }
                }
                chunk.resize(write);
            }

            if ((round + 1) % 500 == 0)
                log("  合并第 " + std::to_string(round + 1) + " 轮，词表: " + std::to_string(vocab_.size()));
        }

        rebuild_merge_map();
        log("最终词表: " + std::to_string(vocab_.size()));
        log("合并规则: " + std::to_string(merges_.size()));
        return {};
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
    {
        auto chunks = pre_tokenize(text);
        std::vector<std::size_t> all_ids;

        for (const auto &chunk : chunks)
        {
            std::vector<std::size_t> ids;
            ids.reserve(chunk.size());
            for (unsigned char b : chunk)
                ids.push_back(static_cast<std::size_t>(b));

            bool merged = true;
            while (merged)
            {
                merged = false;
                std::size_t best_pos = 0;
                std::size_t best_prio = merges_.size() + 1;

                for (std::size_t i = 0; i + 1 < ids.size(); ++i)
                {
                    std::size_t prio = merge_priority(ids[i], ids[i + 1]);
                    if (prio < best_prio) { best_prio = prio; best_pos = i; }
                }

                if (best_prio < merges_.size())
                {
                    ids[best_pos] = merges_[best_prio].new_id;
                    // 原地压缩：将 best_pos+1 之后所有元素前移一位（替代 O(n) 的 vector::erase）
                    for (std::size_t i = best_pos + 1; i + 1 < ids.size(); ++i)
                        ids[i] = ids[i + 1];
                    ids.pop_back();
                    merged = true;
                }
            }

            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
        return all_ids;
    }

    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string result;
        result.reserve(ids.size());
        for (auto id : ids)
            result += id < vocab_.size() ? vocab_[id] : std::string{"\xef\xbf\xbd"};
        return result;
    }

    [[nodiscard]] Result<void> save(const std::string &path) const override
    {
        std::ofstream ofs(path);
        if (!ofs) return std::unexpected(Error{"Cannot write: " + path});
        ofs << "{\n  \"type\": \"bpe_tokenizer\",\n  \"vocab\": {\n";
        bool first = true;
        for (std::size_t tid = 0; tid < vocab_.size(); ++tid)
        {
            if (vocab_[tid].empty()) continue;
            if (!first) ofs << ",\n";
            first = false;
            ofs << "    \"" << std::dec << tid << "\": \"";
            for (unsigned char b : vocab_[tid])
                ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
            ofs << "\"" << std::dec;
        }
        ofs << "\n  },\n  \"vocab_size\": " << std::dec << vocab_.size()
            << ",\n  \"merges\": [";
        for (std::size_t i = 0; i < merges_.size(); ++i)
        {
            if (i > 0) ofs << ",";
            ofs << "\n    [" << merges_[i].id_a << ", " << merges_[i].id_b << ", " << merges_[i].new_id << "]";
        }
        ofs << "\n  ]\n}\n";
        return {};
    }

    [[nodiscard]] Result<void> load(const std::string &path) override
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read: " + path});
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        return load_from_string(content);
    }

    [[nodiscard]] Result<void> load_from_string(const std::string &content) override
    {
        vocab_.clear();
        merges_.clear();

        if (auto p = content.find("\"vocab_size\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::size_t v = 0;
                if (auto [ptr, ec] = std::from_chars(content.data()+c, content.data()+content.size(), v); ec == std::errc{})
                    vocab_.resize(v);
            }

        auto vp = content.find("\"vocab\"");
        if (vp != std::string::npos)
        {
            auto br = content.find('{', vp);
            if (br != std::string::npos)
            {
                std::size_t pos = br + 1;
                while (pos < content.size())
                {
                    while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                           content[pos]=='\r'||content[pos]=='\t'||content[pos]==',')) ++pos;
                    if (pos >= content.size() || content[pos] == '}') break;
                    if (content[pos] != '"') { ++pos; continue; }
                    ++pos; std::string key;
                    while (pos < content.size() && content[pos] != '"') key += content[pos++];
                    ++pos;
                    while (pos < content.size() && (content[pos]==' '||content[pos]==':')) ++pos;
                    if (pos < content.size() && content[pos] == '"')
                    {
                        ++pos; std::string hex;
                        while (pos < content.size() && content[pos] != '"') hex += content[pos++];
                        ++pos;
                        std::size_t id = 0;
                        std::from_chars(key.data(), key.data()+key.size(), id);
                        std::string tok;
                        for (std::size_t i = 0; i+1 < hex.size(); i += 2)
                        { unsigned b = 0; std::from_chars(hex.data()+i, hex.data()+i+2, b, 16); tok += static_cast<char>(b); }
                        if (id >= vocab_.size()) vocab_.resize(id+1);
                        vocab_[id] = std::move(tok);
                    }
                }
            }
        }

        auto mp = content.find("\"merges\"");
        if (mp != std::string::npos)
        {
            auto mb = content.find('[', mp);
            if (mb != std::string::npos)
            {
                std::size_t pos = mb + 1;
                while (pos < content.size())
                {
                    while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                           content[pos]=='\r'||content[pos]=='\t'||content[pos]==',')) ++pos;
                    if (pos >= content.size() || content[pos] == ']') break;
                    if (content[pos] != '[') { ++pos; continue; }
                    ++pos;
                    std::size_t nums[3]{};
                    for (int n = 0; n < 3; ++n)
                    {
                        while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                               content[pos]=='\r'||content[pos]=='\t'||content[pos]==',')) ++pos;
                        std::size_t digit_start = pos;
                        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9')
                            ++pos;
                        if (pos > digit_start)
                            std::from_chars(content.data() + digit_start,
                                            content.data() + pos,
                                            nums[n]);
                    }
                    while (pos < content.size() && content[pos] != ']') ++pos;
                    if (pos < content.size()) ++pos;
                    merges_.push_back({nums[0], nums[1], nums[2]});
                }
            }
        }
        rebuild_merge_map();
        return {};
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept override { return vocab_.size(); }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept override { return vocab_; }
    [[nodiscard]] std::size_t merge_count() const noexcept { return merges_.size(); }

    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view text)
    {
        std::vector<std::string> chunks;
        std::string s(text);
        auto begin = std::sregex_iterator(s.begin(), s.end(), pre_pattern());
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it)
            chunks.push_back(it->str());
        return chunks;
    }

private:
    struct MergeRule { std::size_t id_a, id_b, new_id; };

    struct pair_hash
    {
        [[nodiscard]] std::size_t operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept
        { return p.first * 31 + p.second; }
    };

    std::vector<std::string> vocab_;
    std::vector<MergeRule> merges_;
    // ── 合并优先级 hash 表：O(1) 查找替代线性扫描 ─────────────────────
    // 在 train()/load() 结束时由 rebuild_merge_map() 重建
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> merge_map_;

    // 重建 merge_map_，需要在 merges_ 改变后调用
    void rebuild_merge_map()
    {
        merge_map_.clear();
        merge_map_.reserve(merges_.size());
        for (std::size_t i = 0; i < merges_.size(); ++i)
            merge_map_[{merges_[i].id_a, merges_[i].id_b}] = i;
    }

    // O(1) 查找：(a, b) → 优先级（merges_ 中的索引），不存在返回 merges_.size()
    [[nodiscard]] std::size_t merge_priority(std::size_t a, std::size_t b) const noexcept
    {
        auto it = merge_map_.find({a, b});
        return it != merge_map_.end() ? it->second : merges_.size();
    }
};


// ═══════════════════════════════════════════════════════════════════════════
//  CharBPETokenizer — 字符级 BPE（支持 UTF-8/中文）
// ═══════════════════════════════════════════════════════════════════════════
//  与字节级 BPE (BPETokenizer) 不同，本分词器以 UTF-8 字符为基础单元：
//    - 中文每个汉字 = 1 个基础 token（而非 3 个字节）
//    - BPE 能学会合并常见中文词组（如"人工智能"、"深度学习"）
//    - 小模型更易学到词义
//
//  词表布局：
//    ID 0~3     特殊 token (<pad>, <unk>, <bos>, <eos>)
//    ID 4~259   256 个单字节 ASCII 兜底（保证任意字节可编码，避免小词表 crash）
//    ID 260+    非 ASCII 字符 token（中文等，训练时按首次出现分配）
//    之后       BPE 合并 token
//
//  最小词表大小 = 260（4 特殊 + 256 ASCII）。即使 vocab_size=260 也能正常工作，
//  此时无合并规则，等同于纯字符级编码（中文走 UTF-8 字符，ASCII 走单字节）。
//
//  预分词：按空白分词，每个词内部按 UTF-8 字符切分。
//          CJK 标点也作为分隔符，避免跨句合并。
// ═══════════════════════════════════════════════════════════════════════════

class CharBPETokenizer : public Tokenizer
{
public:
    // 特殊 token ID（固定布局）
    static constexpr std::size_t PAD_ID          = 0;
    static constexpr std::size_t UNK_ID          = 1;
    static constexpr std::size_t BOS_ID          = 2;
    static constexpr std::size_t EOS_ID          = 3;
    static constexpr std::size_t BYTE_BASE       = 4;   // 256 单字节兜底起始 ID
    static constexpr std::size_t CHAR_BASE       = 260; // 非 ASCII 字符 token 起始 ID
    static constexpr std::size_t MIN_VOCAB_SIZE  = CHAR_BASE; // 最小词表大小
    static constexpr std::size_t DEFAULT_VOCAB_SIZE = 5000;
    static constexpr std::uint32_t DEFAULT_MIN_FREQ = 2;

    using LogFn = std::function<void(std::string_view)>;

    struct Config
    {
        std::size_t   vocab_size = DEFAULT_VOCAB_SIZE;
        std::uint32_t min_freq   = DEFAULT_MIN_FREQ;
        LogFn         log        = nullptr;
    };

    CharBPETokenizer() = default;

    [[nodiscard]] std::size_t bos_id() const noexcept override { return BOS_ID; }
    [[nodiscard]] std::size_t eos_id() const noexcept override { return EOS_ID; }

    Result<void> train(const std::string &text) { return train(text, Config{}); }

    Result<void> train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view) { /* 静默 */ }};

        log("\n[CharBPE 训练] 预分词...");
        auto chunks = pre_tokenize(text);
        log("  预分词块数: " + std::to_string(chunks.size()));

        // ── 0. 校验最小词表大小（4 特殊 + 256 ASCII 兜底） ──────
        if (config.vocab_size < MIN_VOCAB_SIZE)
        {
            return std::unexpected(Error{
                "CharBPE vocab_size too small: minimum is " +
                std::to_string(MIN_VOCAB_SIZE) +
                " (4 special + 256 ASCII fallback), got " +
                std::to_string(config.vocab_size)});
        }

        // ── 1. 初始化词表：4 特殊 + 256 单字节 ASCII 兜底 ──────
        // 这保证任意字节序列都能编码（不会因词表太小 crash），
        // ASCII 字符直接走单字节 token，非 ASCII（如中文）走多字节字符 token。
        char_to_id_.clear();
        vocab_.clear();
        vocab_.reserve(config.vocab_size);
        vocab_.emplace_back("<pad>");
        vocab_.emplace_back("<unk>");
        vocab_.emplace_back("<bos>");
        vocab_.emplace_back("<eos>");
        for (std::size_t i = 0; i < 256; ++i)
        {
            std::string ch(1, static_cast<char>(static_cast<unsigned char>(i)));
            vocab_.push_back(ch);
            char_to_id_[ch] = BYTE_BASE + i;
        }
        // assert: vocab_.size() == CHAR_BASE == 260

        // ── 2. 收集非 ASCII 字符，建立 char_to_id ──────────────
        // ASCII 字符已在兜底表中，只收集非 ASCII（如中文汉字）。
        std::vector<std::vector<std::size_t>> char_chunks;
        char_chunks.reserve(chunks.size());

        for (const auto &chunk : chunks)
        {
            std::vector<std::size_t> ids;
            std::size_t i = 0;
            while (i < chunk.size())
            {
                auto [cp, len] = decode_utf8(chunk, i);
                std::string ch = chunk.substr(i, len);
                auto it = char_to_id_.find(ch);
                std::size_t id;
                if (it == char_to_id_.end())
                {
                    // 新字符：仅当词表未满时才添加
                    if (vocab_.size() >= config.vocab_size)
                    {
                        // 词表已满，未知字符回退到 UNK
                        id = UNK_ID;
                    }
                    else
                    {
                        id = vocab_.size();
                        vocab_.push_back(ch);
                        char_to_id_[ch] = id;
                    }
                }
                else
                    id = it->second;
                ids.push_back(id);
                i += len;
            }
            if (!ids.empty())
                char_chunks.push_back(std::move(ids));
        }

        log("  非 ASCII 字符数: " + std::to_string(vocab_.size() - CHAR_BASE));
        log("  总字符数: " + std::to_string(
            std::accumulate(char_chunks.begin(), char_chunks.end(), std::size_t{0},
                [](std::size_t s, const auto &c) { return s + c.size(); })));

        // ── 3. BPE 合并训练 ──
        merges_.clear();
        // 注意：用条件判断避免无符号下溢（vocab_size 可能 < vocab_.size()）
        if (config.vocab_size > vocab_.size())
            merges_.reserve(config.vocab_size - vocab_.size());

        const std::size_t target_merges =
            config.vocab_size > vocab_.size()
                ? config.vocab_size - vocab_.size()
                : 0;
        log("  目标合并数: " + std::to_string(target_merges));

        for (std::size_t round = 0; round < target_merges; ++round)
        {
            std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> pair_freq;
            for (const auto &chunk : char_chunks)
                for (std::size_t i = 0; i + 1 < chunk.size(); ++i)
                    ++pair_freq[{chunk[i], chunk[i + 1]}];

            if (pair_freq.empty()) break;

            auto best = std::ranges::max_element(pair_freq,
                [](const auto &a, const auto &b) { return a.second < b.second; });

            if (best->second < config.min_freq)
            {
                log("  无更多高频 pair，提前停止于第 " + std::to_string(round) + " 轮");
                break;
            }

            const std::size_t new_id = vocab_.size();
            const std::size_t id_a = best->first.first;
            const std::size_t id_b = best->first.second;
            vocab_.push_back(vocab_[id_a] + vocab_[id_b]);
            merges_.push_back({id_a, id_b, new_id});

            // 应用合并到所有 chunk
            for (auto &chunk : char_chunks)
            {
                for (std::size_t i = 0; i + 1 < chunk.size(); )
                {
                    if (chunk[i] == id_a && chunk[i + 1] == id_b)
                    {
                        chunk[i] = new_id;
                        for (std::size_t j = i + 1; j + 1 < chunk.size(); ++j)
                            chunk[j] = chunk[j + 1];
                        chunk.pop_back();
                    }
                    else
                        ++i;
                }
            }

            if ((round + 1) % 500 == 0)
                log("  合并第 " + std::to_string(round + 1) + " 轮，词表: " +
                    std::to_string(vocab_.size()));
        }

        rebuild_merge_map();
        log("最终词表: " + std::to_string(vocab_.size()));
        log("合并规则: " + std::to_string(merges_.size()));
        return {};
    }

    // ── 预分词：按空白 + CJK 标点切分，每词按 UTF-8 字符 ──
    [[nodiscard]] static std::vector<std::string> pre_tokenize(const std::string &text)
    {
        std::vector<std::string> chunks;
        std::size_t i = 0;
        while (i < text.size())
        {
            // 捕获前导空白（附加到下一个非空字符前，GPT-2 风格）
            std::string prefix;
            while (i < text.size() && is_space_byte(text[i]))
            {
                prefix += text[i++];
            }
            if (i >= text.size()) break;

            // 读取一个"词"：连续同类字符
            std::string word = prefix;
            auto [cp, len] = decode_utf8(text, i);

            if (is_cjk(cp))
            {
                // CJK 字符：连续读取，遇到非 CJK 或 CJK 标点停止
                do {
                    word.append(text, i, len);
                    i += len;
                    if (i >= text.size()) break;
                    std::tie(cp, len) = decode_utf8(text, i);
                } while (is_cjk(cp) && !is_cjk_punct(cp));
            }
            else if (is_alnum(cp))
            {
                do {
                    word.append(text, i, len);
                    i += len;
                    if (i >= text.size()) break;
                    std::tie(cp, len) = decode_utf8(text, i);
                } while (is_alnum(cp));
            }
            else
            {
                // 其他字符（标点等）：单个字符
                word.append(text, i, len);
                i += len;
            }
            if (!word.empty())
                chunks.push_back(std::move(word));
        }
        return chunks;
    }

    // ── UTF-8 解码：返回 (码点, 字节长度) ──
    [[nodiscard]] static std::pair<uint32_t, std::size_t>
    decode_utf8(const std::string &s, std::size_t pos)
    {
        if (pos >= s.size()) return {0, 1};
        unsigned char b0 = static_cast<unsigned char>(s[pos]);
        if (b0 < 0x80) return {b0, 1};
        if (pos + 1 >= s.size()) return {0xFFFD, 1};
        if ((b0 & 0xE0) == 0xC0)
        {
            uint32_t cp = ((b0 & 0x1F) << 6) |
                          (static_cast<unsigned char>(s[pos+1]) & 0x3F);
            return {cp, 2};
        }
        if (pos + 2 >= s.size()) return {0xFFFD, 1};
        if ((b0 & 0xF0) == 0xE0)
        {
            uint32_t cp = ((b0 & 0x0F) << 12) |
                          ((static_cast<unsigned char>(s[pos+1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(s[pos+2]) & 0x3F);
            return {cp, 3};
        }
        if (pos + 3 >= s.size()) return {0xFFFD, 1};
        if ((b0 & 0xF8) == 0xF0)
        {
            uint32_t cp = ((b0 & 0x07) << 18) |
                          ((static_cast<unsigned char>(s[pos+1]) & 0x3F) << 12) |
                          ((static_cast<unsigned char>(s[pos+2]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(s[pos+3]) & 0x3F);
            return {cp, 4};
        }
        return {0xFFFD, 1};
    }

    [[nodiscard]] static constexpr bool is_space_byte(char c) noexcept
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    // CJK 统一表意文字（常用汉字）
    [[nodiscard]] static constexpr bool is_cjk(uint32_t cp) noexcept
    {
        return (cp >= 0x4E00 && cp <= 0x9FFF)   // CJK 统一表意文字
            || (cp >= 0x3400 && cp <= 0x4DBF)   // CJK 扩展 A
            || (cp >= 0xF900 && cp <= 0xFAFF);  // CJK 兼容表意
    }

    // CJK 标点（作为分隔符，避免跨句合并）
    [[nodiscard]] static constexpr bool is_cjk_punct(uint32_t cp) noexcept
    {
        return (cp >= 0x3000 && cp <= 0x303F)   // CJK 标点和符号
            || (cp >= 0xFF00 && cp <= 0xFFEF);  // 半角/全角形式
    }

    [[nodiscard]] static constexpr bool is_alnum(uint32_t cp) noexcept
    {
        return (cp >= '0' && cp <= '9')
            || (cp >= 'A' && cp <= 'Z')
            || (cp >= 'a' && cp <= 'z');
    }

    // ════════════════════════════════════════════════════════════════════
    // Tokenizer 接口实现
    // ════════════════════════════════════════════════════════════════════

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
    {
        auto chunks = pre_tokenize(text);
        std::vector<std::size_t> all_ids;

        for (const auto &chunk : chunks)
        {
            std::vector<std::size_t> ids;
            std::size_t i = 0;
            while (i < chunk.size())
            {
                auto [cp, len] = decode_utf8(chunk, i);
                std::string ch = chunk.substr(i, len);
                auto it = char_to_id_.find(ch);
                ids.push_back(it != char_to_id_.end() ? it->second : UNK_ID);
                i += len;
            }

            // 按合并优先级合并
            bool merged = true;
            while (merged)
            {
                merged = false;
                std::size_t best_pos = 0;
                std::size_t best_prio = merges_.size() + 1;

                for (std::size_t k = 0; k + 1 < ids.size(); ++k)
                {
                    std::size_t prio = merge_priority(ids[k], ids[k + 1]);
                    if (prio < best_prio) { best_prio = prio; best_pos = k; }
                }

                if (best_prio < merges_.size())
                {
                    ids[best_pos] = merges_[best_prio].new_id;
                    for (std::size_t k = best_pos + 1; k + 1 < ids.size(); ++k)
                        ids[k] = ids[k + 1];
                    ids.pop_back();
                    merged = true;
                }
            }
            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
        return all_ids;
    }

    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string result;
        for (auto id : ids)
        {
            // BYTE_BASE(4) 及以上均为有效 token（含 ASCII 兜底 + 字符 + 合并）
            // 特殊 token (0-3) 解码为空
            if (id < vocab_.size() && id >= BYTE_BASE)
                result += vocab_[id];
        }
        return result;
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept override { return vocab_.size(); }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept override { return vocab_; }

    [[nodiscard]] std::size_t merge_count() const noexcept { return merges_.size(); }

    // ── 保存 JSON ────────────────────────────────────────────────────
    [[nodiscard]] Result<void> save(const std::string &path) const override
    {
        std::ofstream ofs(path);
        if (!ofs) return std::unexpected(Error{"Cannot write: " + path});
        ofs << "{\n  \"type\": \"char_bpe_tokenizer\",\n  \"vocab\": {\n";
        bool first = true;
        for (std::size_t tid = 0; tid < vocab_.size(); ++tid)
        {
            if (vocab_[tid].empty()) continue;
            if (!first) ofs << ",\n";
            first = false;
            ofs << "    \"" << std::dec << tid << "\": \"";
            // 仅 4 个特殊 token (<pad>/<unk>/<bos>/<eos>) 用纯文本，
            // 其余（含 ASCII 兜底、字符、合并）一律 hex 编码字节，避免破坏 JSON
            if (tid < 4)
                ofs << vocab_[tid];
            else
                for (unsigned char b : vocab_[tid])
                    ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
            ofs << "\"" << std::dec;
        }
        ofs << "\n  },\n  \"vocab_size\": " << std::dec << vocab_.size()
            << ",\n  \"merges\": [";
        for (std::size_t i = 0; i < merges_.size(); ++i)
        {
            if (i > 0) ofs << ",";
            ofs << "\n    [" << merges_[i].id_a << ", " << merges_[i].id_b
                << ", " << merges_[i].new_id << "]";
        }
        ofs << "\n  ]\n}\n";
        return {};
    }

    [[nodiscard]] Result<void> load(const std::string &path) override
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read: " + path});
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        return load_from_string(content);
    }

    [[nodiscard]] Result<void> load_from_string(const std::string &content) override
    {
        vocab_.clear();
        merges_.clear();
        char_to_id_.clear();

        // 读取 vocab_size
        std::size_t vs = 0;
        if (auto p = content.find("\"vocab_size\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::from_chars(content.data()+c, content.data()+content.size(), vs);
            }
        if (vs == 0) vs = CHAR_BASE;
        vocab_.resize(vs);

        // 解析 vocab 对象
        auto vp = content.find("\"vocab\"");
        if (vp == std::string::npos)
            return std::unexpected(Error{"CharBPE: missing vocab"});
        auto br = content.find('{', vp);
        if (br == std::string::npos)
            return std::unexpected(Error{"CharBPE: malformed vocab"});
        std::size_t pos = br + 1;
        while (pos < content.size())
        {
            while (pos < content.size() && (content[pos]==' '||content[pos]=='\n'||
                   content[pos]=='\r'||content[pos]=='\t')) ++pos;
            if (pos >= content.size() || content[pos] == '}') break;
            if (content[pos] == ',') { ++pos; continue; }

            // 读取 key (ID)
            if (content[pos] != '"') { ++pos; continue; }
            ++pos;
            std::size_t id = 0;
            auto [ptr, ec] = std::from_chars(content.data()+pos, content.data()+content.size(), id);
            if (ec != std::errc{}) { ++pos; continue; }
            pos = static_cast<std::size_t>(ptr - content.data());

            // 跳过空白到 ':'
            while (pos < content.size() && content[pos] != ':') ++pos;
            ++pos;
            while (pos < content.size() && content[pos]==' ') ++pos;
            if (pos >= content.size() || content[pos] != '"') continue;
            ++pos;

            // 读取 value
            std::string val;
            if (id < 4)
            {
                // 4 个特殊 token：纯文本
                while (pos < content.size() && content[pos] != '"')
                {
                    if (content[pos] == '\\' && pos+1 < content.size()) ++pos;
                    val += content[pos++];
                }
            }
            else
            {
                // hex 编码（含 ASCII 兜底、字符、合并）
                std::string hex;
                while (pos < content.size() && content[pos] != '"')
                    hex += content[pos++];
                for (std::size_t k = 0; k + 1 < hex.size(); k += 2)
                {
                    unsigned byte = 0;
                    auto [p2, e2] = std::from_chars(hex.data()+k, hex.data()+k+2, byte, 16);
                    if (e2 == std::errc{}) val += static_cast<char>(byte);
                }
            }
            if (pos < content.size() && content[pos] == '"') ++pos;

            if (id < vocab_.size())
            {
                vocab_[id] = val;
                if (id >= BYTE_BASE)
                    char_to_id_[val] = id;
            }
        }

        // 解析 merges 数组
        auto mp = content.find("\"merges\"");
        if (mp != std::string::npos)
        {
            auto mb = content.find('[', mp);
            if (mb != std::string::npos)
            {
                std::size_t mpos = mb + 1;
                while (mpos < content.size())
                {
                    auto lb = content.find('[', mpos);
                    if (lb == std::string::npos) break;
                    auto rb = content.find(']', lb);
                    if (rb == std::string::npos) break;
                    std::size_t a=0, b=0, n=0;
                    auto s1 = content.substr(lb+1, rb-lb-1);
                    std::istringstream iss(s1);
                    char sep;
                    iss >> a >> sep >> b >> sep >> n;
                    merges_.push_back({a, b, n});
                    mpos = rb + 1;
                }
            }
        }

        // 重建 char_to_id_（加载时 vocab 已填充）
        // 从 BYTE_BASE(4) 开始，含 ASCII 兜底 + 字符 + 合并 token
        for (std::size_t i = BYTE_BASE; i < vocab_.size(); ++i)
            if (!vocab_[i].empty())
                char_to_id_[vocab_[i]] = i;

        rebuild_merge_map();
        return {};
    }

private:
    struct MergeRule { std::size_t id_a, id_b, new_id; };
    struct pair_hash
    {
        std::size_t operator()(const std::pair<std::size_t, std::size_t> &p) const noexcept
        {
            return p.first * 31 + p.second;
        }
    };

    std::vector<std::string> vocab_;
    std::vector<MergeRule> merges_;
    std::unordered_map<std::string, std::size_t> char_to_id_;
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pair_hash> merge_map_;

    void rebuild_merge_map()
    {
        merge_map_.clear();
        merge_map_.reserve(merges_.size());
        for (std::size_t i = 0; i < merges_.size(); ++i)
            merge_map_[{merges_[i].id_a, merges_[i].id_b}] = i;
    }

    [[nodiscard]] std::size_t merge_priority(std::size_t a, std::size_t b) const noexcept
    {
        auto it = merge_map_.find({a, b});
        return it != merge_map_.end() ? it->second : merges_.size();
    }
};


// ═══════════════════════════════════════════════════════════════════════════
//  Tokenizer 工厂：根据 JSON 的 "type" 字段创建对应分词器实例
// ═══════════════════════════════════════════════════════════════════════════
//  自动识别 BBPE / CharBPE / WordZip / Space 四种分词器。
//  训练/推理入口统一使用此工厂，无需手动判断类型。

[[nodiscard]] inline std::string peek_tokenizer_type(const std::string &json_content)
{
    // 查找 "type" 字段值
    auto p = json_content.find("\"type\"");
    if (p == std::string::npos) return {};
    auto c = json_content.find(':', p);
    if (c == std::string::npos) return {};
    ++c;
    while (c < json_content.size() && json_content[c] == ' ') ++c;
    if (c >= json_content.size() || json_content[c] != '"') return {};
    ++c;
    std::string val;
    while (c < json_content.size() && json_content[c] != '"')
        val += json_content[c++];
    return val;
}

[[nodiscard]] inline std::unique_ptr<Tokenizer> make_tokenizer(const std::string &type_str)
{
    if (type_str == "bpe_tokenizer")
        return std::make_unique<BPETokenizer>();
    if (type_str == "char_bpe_tokenizer")
        return std::make_unique<CharBPETokenizer>();
    if (type_str == "wordzip_tokenizer")
        return std::make_unique<WordZipTokenizer>();
    if (type_str == "space_tokenizer")
        return std::make_unique<SpaceTokenizer>();
    return nullptr;
}

// 从 JSON 内容创建已加载的 tokenizer（失败返回 nullptr）
[[nodiscard]] inline std::unique_ptr<Tokenizer>
load_tokenizer_from_string(const std::string &json_content)
{
    auto type_str = peek_tokenizer_type(json_content);
    if (type_str.empty())
        return nullptr;  // 无法识别类型
    auto tok = make_tokenizer(type_str);
    if (!tok)
        return nullptr;
    auto r = tok->load_from_string(json_content);
    if (!r)
        return nullptr;
    return tok;
}

// 从文件创建已加载的 tokenizer
[[nodiscard]] inline std::unique_ptr<Tokenizer>
load_tokenizer_from_file(const std::string &path)
{
    std::ifstream ifs(path);
    if (!ifs) return nullptr;
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    return load_tokenizer_from_string(content);
}


} // namespace nn

#endif // NN_DOMAIN_TOKENIZER_HPP
