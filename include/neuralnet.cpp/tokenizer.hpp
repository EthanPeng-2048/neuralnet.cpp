#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nn_config.hpp"

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  ByteZip v2.2 分词器
// ═══════════════════════════════════════════════════════════════════════════
//  基于频率统计 + 独立度去冗余 + 词缀保护 + V2 残差挖掘的子词分词器。
//  Header-only，零外部依赖。
//
//  词表构成:
//    ID 0~3    特殊 token (<pad>, <unk>, <bos>, <eos>)
//    ID 4~259  256 个单字节
//    ID 260+   V1 高频子词 + V2 超长短语
//
//  训练:
//    V1 — 预分词 → 子串频率+上下文统计 → 词缀保护 → 独立度过滤填充
//    V2 — 编码全文找残差 → ±8 字节上下文提取 → 长子串挖掘
//  编码: 预分词 → 逐块贪心最长匹配 → Token ID 序列
//  JSON 格式兼容 Python train_ByteZip.py。
// ═══════════════════════════════════════════════════════════════════════════

class ByteZipTokenizer
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
    static constexpr std::size_t   DEFAULT_V2_MAX_LEN   = 24;
    static constexpr std::uint32_t DEFAULT_MIN_FREQ     = 2;
    static constexpr double        DEFAULT_SKIP_RATIO   = 1.2;
    static constexpr double        DEFAULT_AFFIX_RATIO  = 0.4;
    static constexpr std::size_t   DEFAULT_MAX_V2_BYTES = 300'000;
    static constexpr std::size_t   SMALL_VOCAB_THRESHOLD = 1500;

    // ── 日志回调（默认 cout，nullptr 静默） ────────────────────────────
    using LogFn = std::function<void(std::string_view)>;

    // ── 训练配置 ──────────────────────────────────────────────────────
    struct Config
    {
        std::size_t   vocab_size          = DEFAULT_VOCAB_SIZE;
        std::size_t   v1_max_len          = DEFAULT_V1_MAX_LEN;
        std::size_t   v2_max_len          = DEFAULT_V2_MAX_LEN;
        std::uint32_t min_freq            = DEFAULT_MIN_FREQ;
        double        skip_ratio          = DEFAULT_SKIP_RATIO;
        double        affix_protect_ratio = DEFAULT_AFFIX_RATIO;
        std::size_t   v2_reserve          = 0;  // 0 = 自动分配
        std::size_t   max_v2_scan_bytes   = DEFAULT_MAX_V2_BYTES;
        LogFn         log                 = nullptr;
    };

    // ══════════════════════════════════════════════════════════════════
    //  公开接口
    // ══════════════════════════════════════════════════════════════════

    ByteZipTokenizer() = default;

    void train(const std::string &text) { train(text, Config{}); }

    // ── 两阶段训练（V1 + V2） ─────────────────────────────────────────
    void train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view m) { std::cout << m << '\n'; }};

        max_v2_scan_bytes_ = config.max_v2_scan_bytes;

        const auto v1_max_len    = config.v1_max_len;
        const auto v2_max_len    = config.v2_max_len;
        const auto min_freq      = config.min_freq;
        const auto skip_ratio    = config.skip_ratio;
        const auto affix_ratio   = config.affix_protect_ratio;

        // ── 智能槽位分配 ─────────────────────────────────────────────
        std::size_t v1_target, v2_reserve;
        if (config.vocab_size <= SMALL_VOCAB_THRESHOLD)
        {
            v2_reserve = 0;
            v1_target  = config.vocab_size;
            log("小词表模式 (<=1500): 跳过 V2，全部给 V1 (目标 "
                + std::to_string(v1_target) + ")");
        }
        else
        {
            v2_reserve = config.v2_reserve != 0
                ? config.v2_reserve
                : std::min(std::size_t{1500},
                           static_cast<std::size_t>(config.vocab_size * 0.12));
            v1_target = config.vocab_size - v2_reserve;
            if (v1_target < BYTE_OFFSET + 256)
            {
                v1_target  = BYTE_OFFSET + 256;
                v2_reserve = config.vocab_size - v1_target;
            }
            log("大词表模式: V1 目标 " + std::to_string(v1_target)
                + ", V2 预留 " + std::to_string(v2_reserve));
        }

        // ══════════════════════════════════════════════════════════════
        //  V1: 频率 + 上下文统计 → 词缀保护填充
        // ══════════════════════════════════════════════════════════════
        log("\n[V1] 统计...");
        auto chunks = pre_tokenize(text);

        std::unordered_map<std::string, std::uint32_t> freq;
        std::unordered_map<std::string, std::unordered_set<std::int32_t>> left_ctx, right_ctx;
        freq.reserve(text.size() / 2);

        for (const auto &chunk : chunks)
        {
            const auto L = chunk.size();
            if (L < 2) continue;
            for (std::size_t s = 0; s < L; ++s)
            {
                const auto max_e = std::min(s + v1_max_len, L);
                if (s + 2 > max_e) break;
                for (auto e = s + 2; e <= max_e; ++e)
                {
                    auto sub = chunk.substr(s, e - s);
                    ++freq[sub];
                    left_ctx[sub].insert(s > 0
                        ? static_cast<std::int32_t>(static_cast<unsigned char>(chunk[s - 1]))
                        : -1);
                    right_ctx[sub].insert(e < L
                        ? static_cast<std::int32_t>(static_cast<unsigned char>(chunk[e]))
                        : -1);
                }
            }
        }
        log("  不同子串数: " + std::to_string(freq.size()));

        // V1 排序
        struct Entry { std::string sub; std::uint32_t cnt; std::size_t ben; std::size_t len; };
        std::vector<Entry> sorted;
        sorted.reserve(freq.size());
        for (const auto &[s, c] : freq)
        {
            auto l = s.size();
            sorted.push_back({s, c, static_cast<std::size_t>(c) * (l - 1), l});
        }
        std::ranges::sort(sorted, [](const Entry &a, const Entry &b)
        { return a.ben != b.ben ? a.ben > b.ben : a.len > b.len; });

        // 初始化词表
        vocab_.clear();
        vocab_.resize(BYTE_OFFSET);
        for (std::size_t i = 0; i < 256; ++i)
            vocab_.emplace_back(1, static_cast<char>(i));

        // 词缀检测 + 独立度过滤
        std::unordered_set<std::string> skip;
        skip.reserve(sorted.size() / 4);
        std::size_t added = 0;

        for (const auto &e : sorted)
        {
            if (vocab_.size() >= v1_target) break;
            if (e.cnt < min_freq) break;
            if (e.len <= 1 || skip.contains(e.sub)) continue;

            // 词缀判断：使用上下文多样性
            auto lit = left_ctx.find(e.sub);
            auto rit = right_ctx.find(e.sub);
            bool is_suffix = false, is_prefix = false;
            if (lit != left_ctx.end() && rit != right_ctx.end())
            {
                auto len_l = lit->second.size();
                auto len_r = rit->second.size();
                is_suffix = (len_l >= 3 && len_l > len_r * 2.0);
                is_prefix = (len_r >= 3 && len_r > len_l * 2.0);
            }
            const auto threshold = (is_suffix || is_prefix) ? affix_ratio : skip_ratio;

            // 智能 skip
            const auto sl = e.sub.size();
            for (std::size_t i = 0; i < sl; ++i)
                for (std::size_t j = i + 2; j <= sl; ++j)
                {
                    if (i == 0 && j == sl) continue;
                    auto child = e.sub.substr(i, j - i);
                    auto it = freq.find(child);
                    if (it != freq.end() &&
                        it->second <= static_cast<std::uint32_t>(e.cnt * threshold))
                        skip.insert(std::move(child));
                }

            vocab_.push_back(e.sub);
            ++added;
            if (added % 1000 == 0)
                log("  V1: 已添加 " + std::to_string(added) + " 个词条（总 "
                    + std::to_string(vocab_.size()) + "）");
        }
        log("V1 完成: " + std::to_string(vocab_.size()) + " 词条（新增 "
            + std::to_string(added) + "）");

        // ══════════════════════════════════════════════════════════════
        //  V2: 残差挖掘（仅当预留 > 0）
        // ══════════════════════════════════════════════════════════════
        if (v2_reserve > 0)
        {
            log("\n[V2] 提取残差...");
            // 构建 V1 查找表
            LookupTable lookup_v1;
            for (std::size_t tid = BYTE_OFFSET; tid < vocab_.size(); ++tid)
                if (!vocab_[tid].empty() && vocab_[tid].size() <= v1_max_len)
                    lookup_v1[vocab_[tid]] = tid;

            // 编码全文找残差跨度
            auto data = text;  // 保持 UTF-8
            auto spans = get_residual_spans(data, lookup_v1, v1_max_len);
            std::size_t total_bytes = 0;
            for (const auto &sp : spans) total_bytes += sp.size();
            log("  残差跨度: " + std::to_string(spans.size()) + ", 总字节: "
                + std::to_string(total_bytes));

            if (!spans.empty())
            {
                log("  挖掘超长短语（最多 " + std::to_string(v2_reserve) + " 个）...");
                auto v2_words = build_v2_from_spans(spans, v2_max_len, v2_reserve);
                std::unordered_set<std::string> existing(vocab_.begin(), vocab_.end());
                for (const auto &w : v2_words)
                {
                    if (vocab_.size() >= config.vocab_size) break;
                    if (!existing.contains(w))
                    {
                        vocab_.push_back(w);
                        existing.insert(w);
                    }
                }
                log("  V2 新增 " + std::to_string(vocab_.size() - (BYTE_OFFSET + 256 + added))
                    + " 个");
            }
        }
        else
        {
            log("\n[V2] 跳过（小词表或预留为 0）");
        }

        // ── 最终查找表 ───────────────────────────────────────────────
        log("\n最终词表: " + std::to_string(vocab_.size()));
        max_subword_len_ = std::max(v1_max_len, v2_max_len);
        build_lookup();
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

        ofs << "{\n  \"type\": \"freq_based_tokenizer\",\n  \"vocab\": {\n";
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
            << ",\n  \"special_tokens\": {"
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

    // ── 编码（使用指定查找表，返回位置） ──────────────────────────────
    struct EncodedChunk { std::vector<std::size_t> ids; std::vector<std::size_t> positions; };

    [[nodiscard]] EncodedChunk encode_with_positions(
        std::string_view data, const LookupTable &lut, std::size_t max_len) const
    {
        EncodedChunk result;
        result.ids.reserve(data.size());
        result.positions.reserve(data.size());
        std::size_t pos = 0;
        while (pos < data.size())
        {
            result.positions.push_back(pos);
            auto best_id  = static_cast<std::size_t>(static_cast<unsigned char>(data[pos])) + BYTE_OFFSET;
            auto best_len = std::size_t{1};
            for (auto l = std::min(max_len, data.size() - pos); l > 1; --l)
                if (auto it = lut.find(data.substr(pos, l)); it != lut.end())
                { best_id = it->second; best_len = l; break; }
            result.ids.push_back(best_id);
            pos += best_len;
        }
        return result;
    }

    // ── 编码（内部，用于 encode()） ───────────────────────────────────
    [[nodiscard]] std::vector<std::size_t> encode_bytes(std::string_view data) const
    {
        std::vector<std::size_t> ids;
        ids.reserve(data.size());
        std::size_t pos = 0;
        while (pos < data.size())
        {
            auto best_id  = static_cast<std::size_t>(static_cast<unsigned char>(data[pos])) + BYTE_OFFSET;
            auto best_len = std::size_t{1};
            for (auto l = std::min(max_subword_len_, data.size()-pos); l > 1; --l)
                if (auto it = lookup_.find(data.substr(pos, l)); it != lookup_.end())
                { best_id = it->second; best_len = l; break; }
            ids.push_back(best_id);
            pos += best_len;
        }
        return ids;
    }

    // ── V2: 提取残差跨度 ──────────────────────────────────────────────
    // 用 V1 查找表编码全文，找到单字节运行段，提取 ±8 上下文
    [[nodiscard]] std::vector<std::string> get_residual_spans(
        const std::string &text, const LookupTable &lut, std::size_t max_len) const
    {
        auto enc = encode_with_positions(text, lut, max_len);
        const auto n_tokens = enc.ids.size();
        const auto data_size = text.size();

        std::vector<std::string> spans;
        std::size_t i = 0;
        while (i < n_tokens)
        {
            if (enc.ids[i] < BYTE_OFFSET + 256)
            {
                auto start_pos = enc.positions[i];
                auto j = i;
                while (j < n_tokens && enc.ids[j] < BYTE_OFFSET + 256)
                    ++j;
                auto end_pos = (j < n_tokens) ? enc.positions[j] : data_size;
                auto ext_start = (start_pos >= 8) ? start_pos - 8 : 0;
                auto ext_end   = std::min(data_size, end_pos + 8);
                spans.emplace_back(text.substr(ext_start, ext_end - ext_start));
                i = j;
            }
            else
            {
                ++i;
            }
        }
        return spans;
    }

    // ── V2: 从残差跨度挖掘超长短语 ────────────────────────────────────
    [[nodiscard]] std::vector<std::string> build_v2_from_spans(
        std::vector<std::string> &spans, std::size_t max_len,
        std::size_t max_items) const
    {
        std::size_t total_bytes = 0;
        for (const auto &s : spans) total_bytes += s.size();
        if (total_bytes == 0 || max_items == 0)
            return {};

        // 采样：如果超过上限，随机采样
        if (total_bytes > max_v2_scan_bytes_)
        {
            // 简单采样：打乱后截取
            std::ranges::shuffle(spans, rng_);
            std::string sampled;
            sampled.reserve(max_v2_scan_bytes_);
            for (const auto &sp : spans)
            {
                if (sampled.size() >= max_v2_scan_bytes_) break;
                auto remain = max_v2_scan_bytes_ - sampled.size();
                sampled.append(sp, 0, remain);
            }
            spans = {std::move(sampled)};
        }

        // 统计子串频率（仅长度 16..max_len）
        std::unordered_map<std::string, std::uint32_t> freq;
        for (const auto &span : spans)
        {
            const auto L = span.size();
            if (L < 16) continue;
            for (std::size_t s = 0; s < L; ++s)
            {
                const auto max_e = std::min(s + max_len, L);
                if (s + 16 > max_e) break;
                for (auto e = s + 16; e <= max_e; ++e)
                    ++freq[span.substr(s, e - s)];
            }
        }

        // 按压缩收益排序，过滤 freq < 2
        struct V2Entry { std::string sub; std::uint32_t cnt; std::size_t ben; };
        std::vector<V2Entry> sorted;
        sorted.reserve(freq.size());
        for (const auto &[s, c] : freq)
        {
            if (c < 2) continue;
            sorted.push_back({s, c, static_cast<std::size_t>(c) * (s.size() - 1)});
        }
        std::ranges::sort(sorted, [](const V2Entry &a, const V2Entry &b)
        { return a.ben > b.ben; });

        std::vector<std::string> result;
        result.reserve(std::min(max_items, sorted.size()));
        for (auto &e : sorted)
        {
            if (result.size() >= max_items) break;
            result.push_back(std::move(e.sub));
        }
        return result;
    }

    // ── 构建查找表 ────────────────────────────────────────────────────
    void build_lookup()
    {
        lookup_.clear();
        lookup_.reserve(vocab_.size() - BYTE_OFFSET);
        for (std::size_t i = BYTE_OFFSET; i < vocab_.size(); ++i)
            if (!vocab_[i].empty()) lookup_[vocab_[i]] = i;
    }

    // ── 训练用 RNG ────────────────────────────────────────────────────
    inline static thread_local std::mt19937_64 rng_{std::random_device{}()};

    // ── V2 采样上限（从 config 传入） ──────────────────────────────────
    std::size_t max_v2_scan_bytes_ = DEFAULT_MAX_V2_BYTES;
};

} // namespace nn

#endif // TOKENIZER_HPP
