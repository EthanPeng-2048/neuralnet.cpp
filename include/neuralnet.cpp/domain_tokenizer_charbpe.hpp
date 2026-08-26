#ifndef NN_DOMAIN_TOKENIZER_CHARBPE_HPP
#define NN_DOMAIN_TOKENIZER_CHARBPE_HPP

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
#include "domain_tokenizer_bpe.hpp"   // make_tokenizer 工厂需引用 BPETokenizer

namespace nn
{

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

    static_assert(MIN_VOCAB_SIZE == CHAR_BASE, "MIN_VOCAB_SIZE must equal CHAR_BASE");
    static_assert(CHAR_BASE == BYTE_BASE + 256, "CHAR_BASE must be BYTE_BASE + 256");

    using LogFn = std::function<void(std::string_view)>;

    struct Config
    {
        std::size_t   vocab_size = DEFAULT_VOCAB_SIZE;
        std::uint32_t min_freq   = DEFAULT_MIN_FREQ;
        LogFn         log        = nullptr;
        bool          show_progress = false;   // 训练时显示控制台进度条
        std::uint32_t threads    = 0;   // 预分词并行: 0=自动(线程池全部核心), 1=顺序, >1=指定并行度
    };

    CharBPETokenizer() = default;

    [[nodiscard]] std::size_t bos_id() const noexcept override { return BOS_ID; }
    [[nodiscard]] std::size_t eos_id() const noexcept override { return EOS_ID; }

    // ── write_vocab_entry_ 钩子：前 4 个特殊 token 用纯文本，其余 hex ──
    // 基类 save_vocab_json_ 在写入每个词表条目时回调此虚方法。
    void write_vocab_entry_(std::ofstream &ofs, const std::string &tok, std::size_t tid) const override
    {
        if (tid < 4)
            ofs << tok;
        else
            for (unsigned char b : tok)
                ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
    }

    // ── decode_vocab_entry_ 钩子：前 4 个特殊 token 纯文本，其余 hex ──
    // 基类 parse_vocab_json_ 在解析每个词表条目时回调此虚方法。
    [[nodiscard]] std::string decode_vocab_entry_(const std::string &raw, std::size_t tid) const override
    {
        if (tid < 4)
            return raw;
        std::string tok;
        tok.reserve(raw.size() / 2);
        for (std::size_t i = 0; i + 1 < raw.size(); i += 2)
        {
            unsigned b = 0;
            std::from_chars(raw.data() + i, raw.data() + i + 2, b, 16);
            tok += static_cast<char>(b);
        }
        return tok;
    }

    Result<void> train(const std::string &text) { return train(text, Config{}); }

    Result<void> train(const std::string &text, const Config &config)
    {
        auto log = config.log
            ? config.log
            : LogFn{[](std::string_view) { /* 静默 */ }};

        log("\n[CharBPE 训练] 预分词...");
        const bool show_p = config.show_progress;

        // ── 预分词 + 字符串层去重计数（可并行，map-reduce） ──────
        // 边按 UTF-8 字符切分边填 word_freq，不产生全部 chunk 列表；
        // 经全局线程池在安全切分点并行，结果与单线程完全一致。
        std::unordered_map<std::string, std::size_t> word_freq;
        {
            auto worker = [&](std::size_t b, std::size_t e,
                              std::unordered_map<std::string, std::size_t> &local,
                              const std::function<void(std::size_t)> &progress)
            {
                std::size_t i = b;
                std::size_t last_mb = static_cast<std::size_t>(-1);
                while (i < e)
                {
                    // 捕获前导空白（附加到下一个非空字符前，GPT-2 风格）
                    std::string prefix;
                    while (i < e && is_space_byte(text[i]))
                    {
                        prefix += text[i++];
                    }
                    if (i >= e) break;

                    // 读取一个"词"：连续同类字符
                    std::string word = std::move(prefix);
                    auto [cp, len] = decode_utf8(text, i);

                    if (is_cjk(cp))
                    {
                        // CJK 字符：连续读取，遇到非 CJK 或 CJK 标点停止
                        do {
                            word.append(text, i, len);
                            i += len;
                            if (i >= e) break;
                            std::tie(cp, len) = decode_utf8(text, i);
                        } while (is_cjk(cp) && !is_cjk_punct(cp));
                    }
                    else if (is_alnum(cp))
                    {
                        do {
                            word.append(text, i, len);
                            i += len;
                            if (i >= e) break;
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
                        ++local[std::move(word)];

                    const auto mb = i / (1u << 20);
                    if (mb != last_mb)
                    {
                        last_mb = mb;
                        progress(i);
                    }
                }
            };
            word_freq = parallel_pretokenize(text, config.threads, "预分词", show_p, worker);
        }
        log("  去重后 chunk 数: " + std::to_string(word_freq.size()));

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

        // ── 3. 收集非 ASCII 字符，建立 char_to_id ──────────────
        // 遍历 unique chunks 收集字符。为保证并行预分词与顺序执行结果一致，
        // 先收集去重字符、再按码点排序后分配 ID——不依赖 unordered_map 的
        // 迭代顺序，结果确定且与线程数无关。
        const std::size_t wf_size = word_freq.size();
        std::unordered_set<std::string> char_set;
        {
            std::size_t cc_i = 0;
            std::size_t cc_last_pct = static_cast<std::size_t>(-1);
            for (const auto &[chunk, _] : word_freq)
            {
                std::size_t i = 0;
                while (i < chunk.size())
                {
                    auto [cp, len] = decode_utf8(chunk, i);
                    if (static_cast<unsigned char>(chunk[i]) >= 0x80) // 非 ASCII
                        char_set.insert(chunk.substr(i, len));
                    i += len;
                }
                if (show_p)
                {
                    ++cc_i;
                    const auto pct = cc_i * 100 / std::max<std::size_t>(wf_size, 1);
                    if (pct != cc_last_pct)
                    {
                        cc_last_pct = pct;
                        render_progress_("收集字符", cc_i, wf_size);
                    }
                }
            }
        }

        // 按码点排序（UTF-8 字节序 = 码点序），依次分配 ID
        std::vector<std::string> sorted_chars(char_set.begin(), char_set.end());
        std::ranges::sort(sorted_chars);
        for (const auto &ch : sorted_chars)
        {
            if (vocab_.size() >= config.vocab_size)
                break; // 词表已满，后续字符回退 UNK
            char_to_id_[ch] = vocab_.size();
            vocab_.push_back(ch);
        }

        log("  非 ASCII 字符数: " + std::to_string(vocab_.size() - CHAR_BASE));

        // ── 4. 只对 unique chunks 转字符 ID 向量 ──────────────
        std::vector<std::pair<std::vector<std::size_t>, std::size_t>> char_chunks;
        char_chunks.reserve(word_freq.size());
        std::size_t total_chars = 0;
        std::size_t enc_i = 0;
        std::size_t enc_last_pct = static_cast<std::size_t>(-1);
        for (const auto &[chunk, cnt] : word_freq)
        {
            std::vector<std::size_t> ids;
            ids.reserve(chunk.size());
            std::size_t i = 0;
            while (i < chunk.size())
            {
                auto [cp, len] = decode_utf8(chunk, i);
                std::string ch = chunk.substr(i, len);
                auto it = char_to_id_.find(ch);
                ids.push_back(it != char_to_id_.end() ? it->second : UNK_ID);
                i += len;
            }
            total_chars += ids.size();
            if (!ids.empty())
                char_chunks.emplace_back(std::move(ids), cnt);
            if (show_p)
            {
                ++enc_i;
                const auto pct = enc_i * 100 / std::max<std::size_t>(wf_size, 1);
                if (pct != enc_last_pct)
                {
                    enc_last_pct = pct;
                    render_progress_("编码词条", enc_i, wf_size);
                }
            }
        }
        word_freq.clear();
        if (show_p)
            finish_progress_("预分词");
        log("  总字符数: " + std::to_string(total_chars));

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

        // ── 共享 BPE 训练算法（基类 bpe_train_impl_） ──────────────
        bpe_train_impl_(vocab_, char_chunks, target_merges, config.min_freq, log, show_p);

        rebuild_merge_map_();
        // ── 添加对话标记 token ──────────────────────────────────────
        add_dialogue_markers_to_vocab_(vocab_);
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
            // 正常 CharBPE 编码
            auto chunks = pre_tokenize(segment);
            for (const auto &chunk : chunks)
            {
                std::vector<std::size_t> ids;
                ids.reserve(chunk.size());
                std::size_t i = 0;
                while (i < chunk.size())
                {
                    if (static_cast<unsigned char>(chunk[i]) < 0x80)
                    {
                        ids.push_back(BYTE_BASE + static_cast<unsigned char>(chunk[i]));
                        i += 1;
                    }
                    else
                    {
                        auto [cp, len] = decode_utf8(chunk, i);
                        std::string ch = chunk.substr(i, len);
                        auto it = char_to_id_.find(ch);
                        ids.push_back(it != char_to_id_.end() ? it->second : UNK_ID);
                        i += len;
                    }
                }

                // 共享 BPE 合并算法（基类 bpe_merge_impl_）
                auto merged = bpe_merge_impl_(std::move(ids));
                all_ids.insert(all_ids.end(), merged.begin(), merged.end());
            } // end for(chunk)
        } // end while(pos)
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
        // 共享 JSON 头部 + vocab 段（基类 save_vocab_json_ + write_vocab_entry_ 钩子）
        // 起始 ID = 0：保留前 4 个特殊 token + ASCII 兜底，全部写入。
        save_vocab_json_(ofs, "char_bpe_tokenizer", 0);
        ofs << ",\n  \"merges\": [";
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

        // 共享 JSON vocab 解析（基类 parse_vocab_json_ + decode_vocab_entry_ 钩子）
        vocab_ = parse_vocab_json_(content);
        if (vocab_.empty())
        {
            // parse_vocab_json_ 未能解析 vocab_size 时回退到 CHAR_BASE
            vocab_.resize(CHAR_BASE);
        }

        // 重建 char_to_id_（BYTE_BASE 及以上均为字符 token）
        for (std::size_t id = BYTE_BASE; id < vocab_.size(); ++id)
            if (!vocab_[id].empty())
                char_to_id_[vocab_[id]] = id;

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

        // ── 恢复对话标记 ID（从已有词表检测或追加） ─────────────
        restore_dialogue_markers(vocab_);
        rebuild_merge_map_();
        return {};
    }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::size_t> char_to_id_;
};


// ═══════════════════════════════════════════════════════════════════════════
//  Tokenizer 工厂：根据 JSON 的 "type" 字段创建对应分词器实例
// ═══════════════════════════════════════════════════════════════════════════
//  自动识别 BBPE / CharBPE 两种分词器（WordZip / Space 已于 2026 清理移除）。
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

#endif // NN_DOMAIN_TOKENIZER_CHARBPE_HPP
