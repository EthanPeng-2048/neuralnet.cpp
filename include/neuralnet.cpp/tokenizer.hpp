#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

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
#include <ranges>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nn_config.hpp"

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

    // ── 从 JSON 字符串加载（用于从模型文件中提取嵌入词表） ──────
    [[nodiscard]] Result<void> load_json(const std::string &json_content)
    {
        // 写入临时文件后调用 load（各子类已实现 load）
        const auto tmp_path = std::filesystem::temp_directory_path() / "nn_tokenizer_tmp.json";
        {
            std::ofstream ofs(tmp_path, std::ios::binary);
            if (!ofs)
                return std::unexpected(Error{"Cannot write tmp tokenizer file"});
            ofs.write(json_content.data(), static_cast<std::streamsize>(json_content.size()));
        }
        auto result = load(tmp_path.string());
        std::filesystem::remove(tmp_path);
        return result;
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

    void train(const std::string &text) { train(text, Config{}); }

    // ── 词频训练 ──────────────────────────────────────────────────────
    void train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view m) { std::cout << m << '\n'; }};

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
    }

    // ── 编码 ──────────────────────────────────────────────────────────
    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const
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
    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const
    {
        std::string raw;
        raw.reserve(ids.size() * 2);
        for (auto tid : ids)
            raw += tid < vocab_.size() ? vocab_[tid] : std::string{"\xef\xbf\xbd"};
        return raw;
    }

    // ── 保存 JSON ────────────────────────────────────────────────────
    [[nodiscard]] Result<void> save(const std::string &path) const
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
    [[nodiscard]] Result<void> load(const std::string &path)
    {
        std::ifstream ifs(path);
        if (!ifs) return std::unexpected(Error{"Cannot read: " + path});
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());

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

    [[nodiscard]] constexpr std::size_t vocab_size()      const noexcept { return vocab_.size(); }
    [[nodiscard]] constexpr std::size_t max_subword_len() const noexcept { return max_subword_len_; }
    [[nodiscard]] constexpr std::size_t byte_offset()     const noexcept { return BYTE_OFFSET; }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept { return vocab_; }

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

    void train(const std::string &text) { train(text, Config{}); }

    void train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view m) { std::cout << m << '\n'; }};

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
        for (std::size_t i = 0; i < 128; ++i)
            id_to_token_.emplace_back(1, static_cast<char>(i));

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

    void train(const std::string &text) { train(text, Config{}); }

    void train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view m) { std::cout << m << '\n'; }};

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

        merges_.clear();
        merges_.reserve(config.vocab_size - BYTE_BASE);

        const std::size_t target_merges = config.vocab_size - BYTE_BASE;
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
                for (std::size_t i = 0; i + 1 < chunk.size(); )
                {
                    if (chunk[i] == id_a && chunk[i + 1] == id_b)
                    {
                        chunk[i] = new_id;
                        chunk.erase(chunk.begin() + static_cast<std::ptrdiff_t>(i + 1));
                    }
                    else ++i;
                }
            }

            if ((round + 1) % 500 == 0)
                log("  合并第 " + std::to_string(round + 1) + " 轮，词表: " + std::to_string(vocab_.size()));
        }

        log("最终词表: " + std::to_string(vocab_.size()));
        log("合并规则: " + std::to_string(merges_.size()));
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
                    ids.erase(ids.begin() + static_cast<std::ptrdiff_t>(best_pos + 1));
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
                        std::string num_str;
                        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9')
                            num_str += content[pos++];
                        if (!num_str.empty())
                            nums[n] = static_cast<std::size_t>(std::stoull(num_str));
                    }
                    while (pos < content.size() && content[pos] != ']') ++pos;
                    if (pos < content.size()) ++pos;
                    merges_.push_back({nums[0], nums[1], nums[2]});
                }
            }
        }
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

    [[nodiscard]] std::size_t merge_priority(std::size_t a, std::size_t b) const
    {
        for (std::size_t i = 0; i < merges_.size(); ++i)
            if (merges_[i].id_a == a && merges_[i].id_b == b)
                return i;
        return merges_.size();
    }
};


} // namespace nn

#endif // TOKENIZER_HPP
