#pragma once

#include <algorithm>
#include <atomic>
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
#include <queue>
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

#include "core_config.hpp"

#include "domain_tokenizer_base.hpp"

namespace nn
{

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
        bool          show_progress = false;   // 训练时显示控制台进度条
        std::uint32_t threads    = 0;   // 预分词并行: 0=自动(线程池全部核心), 1=顺序, >1=指定并行度
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
        const bool show_p = config.show_progress;

        // ── 预分词 + 字符串层去重计数（可并行，map-reduce） ──────
        // 边按 regex 切分边填 word_freq，不产生全部 chunk 列表：
        //   1) 内存：不再持有数亿个 string chunk（节省数 GB）
        //   2) 性能：经全局线程池在安全切分点并行，结果与单线程完全一致
        std::unordered_map<std::string, std::size_t> word_freq;
        {
            const std::string s(text);
            auto worker = [&](std::size_t b, std::size_t e,
                              std::unordered_map<std::string, std::size_t> &local,
                              const std::function<void(std::size_t)> &progress)
            {
                const std::string seg(s.data() + b, e - b);
                // 用保护标记的 pre_tokenize：保留特殊 token 作为独立 chunk，
                // 不与其前后文本/标点合并（避免 "?<|"、".<|" 等垃圾 token）。
                auto chunks = pre_tokenize(seg);
                std::size_t last_mb = static_cast<std::size_t>(-1);
                std::size_t off = 0;
                for (const auto &ch : chunks)
                {
                    ++local[ch];
                    off += ch.size();
                    // 绝对偏移 = 段起点 + 段内偏移
                    const auto pos = b + off;
                    const auto mb  = pos / (1u << 20);
                    if (mb != last_mb)
                    {
                        last_mb = mb;
                        progress(pos);
                    }
                }
            };
            word_freq = parallel_pretokenize(s, config.threads, "预分词", show_p, worker);
        }
        log("  去重后 chunk 数: " + std::to_string(word_freq.size()));

        // 只对 unique chunks 转字节向量 + 计数
        std::vector<std::pair<std::vector<std::size_t>, std::size_t>> byte_chunks;
        byte_chunks.reserve(word_freq.size());
        std::size_t total_bytes = 0;
        std::size_t enc_i = 0;
        std::size_t last_pct = static_cast<std::size_t>(-1);
        const std::size_t wf_size = word_freq.size();
        for (auto &[w, cnt] : word_freq)
        {
            std::vector<std::size_t> ids;
            ids.reserve(w.size());
            for (unsigned char b : w)
                ids.push_back(static_cast<std::size_t>(b));
            total_bytes += ids.size();
            byte_chunks.emplace_back(std::move(ids), cnt);
            if (show_p)
            {
                ++enc_i;
                const auto pct = enc_i * 100 / std::max<std::size_t>(wf_size, 1);
                if (pct != last_pct)
                {
                    last_pct = pct;
                    render_progress_("编码词条", enc_i, wf_size);
                }
            }
        }
        word_freq.clear();
        if (show_p)
            finish_progress_("预分词");
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

        // ── 共享 BPE 训练算法（基类 bpe_train_impl_） ──────────────
        bpe_train_impl_(vocab_, byte_chunks, target_merges, config.min_freq, log, show_p);

        rebuild_merge_map_();
        // ── 添加对话标记 token ──────────────────────────────────────
        add_dialogue_markers_to_vocab_(vocab_);
        log("最终词表: " + std::to_string(vocab_.size()));
        log("合并规则: " + std::to_string(merges_.size()));
        return {};
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
    {
        std::vector<std::size_t> all_ids;
        std::size_t pos = 0;
        while (pos < text.size())
        {
            // 对话标记优先匹配
            if (has_dialogue_markers())
            {
                auto [mid, mlen] = try_match_marker(text, pos);
                if (mid != Tokenizer::npos)
                { all_ids.push_back(mid); pos += mlen; continue; }
            }
            // 收集普通文本直到下一个标记
            auto seg_start = pos;
            while (pos < text.size())
            {
                if (has_dialogue_markers())
                { auto [mid, mlen] = try_match_marker(text, pos); if (mid != Tokenizer::npos) break; }
                ++pos;
            }
            std::string segment(text, seg_start, pos - seg_start);
            // 正常 BPE 编码
            auto chunks = pre_tokenize(segment);
            for (const auto &chunk : chunks)
            {
                std::vector<std::size_t> ids;
                ids.reserve(chunk.size());
                for (unsigned char b : chunk)
                    ids.push_back(static_cast<std::size_t>(b));

                // 共享 BPE 合并算法（基类 bpe_merge_impl_）
                auto merged = bpe_merge_impl_(std::move(ids));
                all_ids.insert(all_ids.end(), merged.begin(), merged.end());
            }
        }
        return all_ids;
    }

    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string result;
        result.reserve(ids.size());
        for (auto id : ids)
        {
            // 对话标记 → 原始文本
            if (id == markers_.system)        { result += "<|system|>";          continue; }
            if (id == markers_.system_end)    { result += "<|end_of_system|>";   continue; }
            if (id == markers_.user)          { result += "<|user|>";            continue; }
            if (id == markers_.user_end)      { result += "<|end_of_user|>";     continue; }
            if (id == markers_.assistant)     { result += "<|assistant|>";       continue; }
            if (id == markers_.assistant_end) { result += "<|end_of_assistant|>";continue; }
            if (id == markers_.start_think)   { result += "<think>";                continue; }
            if (id == markers_.end_think)     { result += "</think>";              continue; }
            result += id < vocab_.size() ? vocab_[id] : std::string{REPLACEMENT_CHAR};
        }
        return result;
    }

    [[nodiscard]] Result<void> save(const std::string &path) const override
    {
        std::ofstream ofs(path);
        if (!ofs) return std::unexpected(Error{"Cannot write: " + path});
        // 共享 JSON 头部 + vocab 段（基类 save_vocab_json_）
        save_vocab_json_(ofs, "bpe_tokenizer", 0);
        ofs << ",\n  \"merges\": [";
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

        // 共享 JSON vocab 解析（基类 parse_vocab_json_）
        vocab_ = parse_vocab_json_(content);

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
        // ── 恢复对话标记 ID（从已有词表检测或追加） ─────────────
        restore_dialogue_markers(vocab_);
        rebuild_merge_map_();
        return {};
    }

    [[nodiscard]] std::size_t vocab_size() const noexcept override { return vocab_.size(); }
    [[nodiscard]] const std::vector<std::string> &vocab() const noexcept override { return vocab_; }
    [[nodiscard]] std::size_t merge_count() const noexcept { return merges_.size(); }

    [[nodiscard]] static std::vector<std::string> pre_tokenize(std::string_view text)
    {
        // 保留特殊标记（按长度降序，最长匹配优先，如 <|tool_result|> 优先于 <|tool|>）
        static const std::vector<std::string> markers = [] {
            std::vector<std::string> m = {
                "<|system|>", "<|end_of_system|>", "<|user|>", "<|end_of_user|>",
                "<|assistant|>", "<|end_of_assistant|>", "<|end_of_turn|>", "<|sep|>",
                "<|tool|>", "<|tool_result|>", "<|image|>", "<|audio|>",
                "<|video|>", "<|file|>"
            };
            std::sort(m.begin(), m.end(),
                      [](const auto &a, const auto &b) { return a.size() > b.size(); });
            return m;
        }();

        // 保护标记：标记串作为独立 chunk（内部可合并成单 token），
        // 不与相邻文本/标点合并，避免 BPE 产出 "?<|"、".<|" 这类垃圾 token。
        // 合并只在 chunk 内部发生，因此标记与邻居被天然隔离。
        std::vector<std::string> chunks;
        chunks.reserve(text.size() / 4);
        std::string plain;  // 累积标记之间的普通文本
        auto flush_plain = [&]() {
            if (plain.empty()) return;
            std::string s = std::move(plain); plain.clear();
            // 手写状态机替代 std::sregex_iterator（预分词热路径），语义与
            // pre_pattern() 完全一致（左最先优先，byte 级）；正确性由
            // src/tokenizer_consistency_test.cpp 对拍 std::regex 仲裁。
            std::size_t p = 0;
            while (p < s.size())
            {
                const std::size_t len = pre_match_len(s, p);
                if (len == 0) { ++p; continue; }   // 无匹配字符（如 '_'）跳过，与 sregex_iterator 一致
                chunks.push_back(s.substr(p, len));
                p += len;
            }
        };

        std::size_t pos = 0;
        while (pos < text.size())
        {
            std::size_t mlen = 0;
            // 全部特殊标记以 '<' 开头：非 '<' 位置直接跳过线性扫描，
            // 避免对每个普通字符做 14 次比较 + substr 分配；compare 免中间字符串。
            if (text[pos] == '<')
            {
                for (const auto &mk : markers)
                    if (pos + mk.size() <= text.size() && text.compare(pos, mk.size(), mk) == 0)
                    { mlen = mk.size(); break; }
            }
            if (mlen > 0)
            {
                flush_plain();
                chunks.push_back(std::string(text.substr(pos, mlen)));  // 标记独立 chunk
                pos += mlen;
            }
            else
            {
                plain.push_back(text[pos]);
                ++pos;
            }
        }
        flush_plain();
        return chunks;
    }

private:
    // ── 手写 GPT-2 预分词状态机：单步返回从 pos 起的匹配长度 ───────────────
    // 对应 pre_pattern()（ECMAScript 左最先优先，byte 级）：
    //   's|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+
    // 语义要点：① 缩写依序尝试；② ' ?[a-zA-Z]+' 可选空格后须跟字母（空格后非
    // 字母则整体失败，落回 \s+）；③ 非 ASCII 字节视作 [^\s\w] 标点运行。
    static std::size_t pre_match_len(std::string_view text, std::size_t pos) noexcept
    {
        static constexpr std::string_view contractions[] = {
            "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
        };
        const char c = text[pos];
        if (c == '\'')
            for (const auto &a : contractions)
                if (pos + a.size() <= text.size() && text.compare(pos, a.size(), a) == 0)
                    return a.size();

        const auto is_letter = [](char ch) noexcept {
            return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
        };
        const auto is_digit = [](char ch) noexcept { return ch >= '0' && ch <= '9'; };
        const auto is_ws = [](char ch) noexcept {
            return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
        };
        const auto is_word = [&](char ch) noexcept { return is_letter(ch) || is_digit(ch) || ch == '_'; };

        // ' ?[a-zA-Z]+'
        if (c == ' ')
        {
            if (pos + 1 < text.size() && is_letter(text[pos + 1]))
            {
                std::size_t e = pos + 2;
                while (e < text.size() && is_letter(text[e])) ++e;
                return e - pos;
            }
        }
        else if (is_letter(c))
        {
            std::size_t e = pos + 1;
            while (e < text.size() && is_letter(text[e])) ++e;
            return e - pos;
        }
        // ' ?[0-9]+'
        if (c == ' ')
        {
            if (pos + 1 < text.size() && is_digit(text[pos + 1]))
            {
                std::size_t e = pos + 2;
                while (e < text.size() && is_digit(text[e])) ++e;
                return e - pos;
            }
        }
        else if (is_digit(c))
        {
            std::size_t e = pos + 1;
            while (e < text.size() && is_digit(text[e])) ++e;
            return e - pos;
        }
        // '[^\s\w]+'
        if (!is_ws(c) && !is_word(c))
        {
            std::size_t e = pos + 1;
            while (e < text.size() && !is_ws(text[e]) && !is_word(text[e])) ++e;
            return e - pos;
        }
        // '\s+'
        if (is_ws(c))
        {
            std::size_t e = pos + 1;
            while (e < text.size() && is_ws(text[e])) ++e;
            return e - pos;
        }
        // 无匹配：返回 0 表示该字符被跳过（如 '_' 是 \w 但非字母/数字，不属于
        // 任何分支），与 std::sregex_iterator 的"跳过无匹配字符"语义一致。
        return 0;
    }

    std::vector<std::string> vocab_;
};


} // namespace nn

