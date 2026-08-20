#ifndef NN_DOMAIN_TOKENIZER_HPP
#define NN_DOMAIN_TOKENIZER_HPP

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
    // 训练/推理入口通过这些接口获取 BOS/EOS/PAD，无需硬编码常量。
    [[nodiscard]] virtual std::size_t bos_id() const noexcept { return npos; }
    [[nodiscard]] virtual std::size_t eos_id() const noexcept { return npos; }
    [[nodiscard]] virtual std::size_t pad_id() const noexcept { return 0; }

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // UTF-8 替换字符 U+FFFD（解码未知 ID 时的回退）
    static constexpr std::string_view REPLACEMENT_CHAR = "\xef\xbf\xbd";

    // ── 从 JSON 字符串加载（纯内存解析，零磁盘 I/O） ──────────────
    // 子类重写此方法以支持从 JSON 字符串直接加载词表。
    [[nodiscard]] virtual Result<void> load_from_string(const std::string &json_content) = 0;

    // ── 对话标记集合（基类持有，子类直接复用，无需各自重定义） ──────
    struct DialogueMarkers {
        std::size_t system = npos, user = npos, assistant = npos;
        std::size_t system_end = npos, user_end = npos, assistant_end = npos;
    };

    // ── 对话标记 token（默认从 markers_ 读取，子类一般无需重写） ─────
    [[nodiscard]] virtual std::size_t system_marker_id()        const noexcept { return markers_.system; }
    [[nodiscard]] virtual std::size_t user_marker_id()          const noexcept { return markers_.user; }
    [[nodiscard]] virtual std::size_t assistant_marker_id()    const noexcept { return markers_.assistant; }
    [[nodiscard]] virtual std::size_t end_system_marker_id()   const noexcept { return markers_.system_end; }
    [[nodiscard]] virtual std::size_t end_user_marker_id()      const noexcept { return markers_.user_end; }
    [[nodiscard]] virtual std::size_t end_assistant_marker_id() const noexcept { return markers_.assistant_end; }

    // ── 是否包含对话标记 ──────────────────────────────────────
    [[nodiscard]] bool has_dialogue_markers() const noexcept
    {
        return markers_.system != npos;
    }

    // ── 从词表中恢复对话标记 ID（加载后调用） ─────────────────
    // 扫描词表查找 <|system|> 等标记字符串，若已存在则绑定已有 ID，
    // 若不存在则追加到词表末尾。随后追加额外保留特殊 token。
    void restore_dialogue_markers(std::vector<std::string> &vocab)
    {
        static constexpr std::string_view marker_strs[] = {
            "<|system|>", "<|end_of_system|>", "<|user|>", "<|end_of_user|>",
            "<|assistant|>", "<|end_of_assistant|>"
        };
        std::size_t ids[6];
        bool all_found = true;
        for (int i = 0; i < 6; ++i)
        {
            ids[i] = npos;
            for (std::size_t j = 0; j < vocab.size(); ++j)
            {
                if (vocab[j] == marker_strs[i])
                { ids[i] = j; break; }
            }
            if (ids[i] == npos) all_found = false;
        }
        if (!all_found)
        {
            // 部分标记缺失，追加全部（保持连续 ID）
            for (int i = 0; i < 6; ++i)
            {
                if (ids[i] == npos)
                { ids[i] = vocab.size(); vocab.emplace_back(marker_strs[i]); }
            }
        }
        // 追加额外保留特殊 token（回合结束/SEP/工具/多模态/文件）
        append_reserved_extras_(vocab);
        // 通过虚方法设置 markers_（基类默认实现赋值 markers_）
        set_dialogue_marker_ids(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
    }

protected:
    // 对话标记 ID 集合（子类直接读取，无需各自定义 6 个字段）
    DialogueMarkers markers_;

    // 额外保留 token → id 查找表（按字符串长度降序，供 try_match_marker 最长匹配）
    std::vector<std::pair<std::string, std::size_t>> reserved_ids_;

    // 默认实现：将参数赋值到 markers_（子类一般无需重写）
    virtual void set_dialogue_marker_ids(
        std::size_t system, std::size_t end_system,
        std::size_t user, std::size_t end_user,
        std::size_t assistant, std::size_t end_assistant)
    {
        markers_.system        = system;
        markers_.system_end    = end_system;
        markers_.user          = user;
        markers_.user_end      = end_user;
        markers_.assistant      = assistant;
        markers_.assistant_end = end_assistant;
    }

    // 添加 6 个对话标记到词表末尾，并绑定 ID 到 markers_；随后追加额外保留 token
    void add_dialogue_markers_to_vocab_(std::vector<std::string> &vocab)
    {
        markers_.system        = vocab.size(); vocab.emplace_back("<|system|>");
        markers_.system_end    = vocab.size(); vocab.emplace_back("<|end_of_system|>");
        markers_.user          = vocab.size(); vocab.emplace_back("<|user|>");
        markers_.user_end      = vocab.size(); vocab.emplace_back("<|end_of_user|>");
        markers_.assistant      = vocab.size(); vocab.emplace_back("<|assistant|>");
        markers_.assistant_end = vocab.size(); vocab.emplace_back("<|end_of_assistant|>");
        append_reserved_extras_(vocab);
    }

    // ── 追加额外的保留特殊 token（供未来 SFT/工具/多模态/文件/SEP 使用） ──
    // 仅作为词表保留 id，并登记到 reserved_ids_ 供编码时匹配为单 token；
    // 幂等（已存在则不重复追加）。
    void append_reserved_extras_(std::vector<std::string> &vocab)
    {
        static constexpr std::string_view extras[] = {
            "<|end_of_turn|>", "<|sep|>", "<|tool|>", "<|tool_result|>",
            "<|image|>", "<|audio|>", "<|video|>", "<|file|>"
        };
        reserved_ids_.clear();
        for (const auto ex : extras)
        {
            std::size_t id = npos;
            for (std::size_t j = 0; j < vocab.size(); ++j)
                if (vocab[j] == ex) { id = j; break; }
            if (id == npos) { id = vocab.size(); vocab.emplace_back(ex); }
            reserved_ids_.emplace_back(std::string(ex), id);
        }
        // 按长度降序，保证 try_match_marker 最长匹配优先（如 <|tool|> vs <|tool_result|>）
        std::sort(reserved_ids_.begin(), reserved_ids_.end(),
                  [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
    }

    // ── 辅助：尝试匹配对话标记，返回 (token_id, 匹配字节数)，不匹配返回 (npos, 0) ──
    // 先匹配 6 个对话标记，再匹配额外保留 token（按长度降序，最长优先）。
    [[nodiscard]] std::pair<std::size_t, std::size_t>
    try_match_marker(std::string_view text, std::size_t pos) const noexcept
    {
        static constexpr std::string_view markers[] = {
            "<|system|>", "<|end_of_system|>", "<|user|>", "<|end_of_user|>",
            "<|assistant|>", "<|end_of_assistant|>"
        };
        for (const auto &m : markers)
        {
            if (pos + m.size() <= text.size() && text.substr(pos, m.size()) == m)
            {
                std::size_t id = npos;
                if      (m == "<|system|>")            id = markers_.system;
                else if (m == "<|end_of_system|>")    id = markers_.system_end;
                else if (m == "<|user|>")             id = markers_.user;
                else if (m == "<|end_of_user|>")      id = markers_.user_end;
                else if (m == "<|assistant|>")        id = markers_.assistant;
                else if (m == "<|end_of_assistant|>") id = markers_.assistant_end;
                if (id != npos) return {id, m.size()};
            }
        }
        // 额外保留 token：reserved_ids_ 已按长度降序，首个命中即最长匹配
        for (const auto &[tok, id] : reserved_ids_)
        {
            if (id != npos && pos + tok.size() <= text.size()
                && text.substr(pos, tok.size()) == tok)
                return {id, tok.size()};
        }
        return {npos, 0};
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  BPE 共享构件（供 BPETokenizer / CharBPETokenizer 复用）
    // ═══════════════════════════════════════════════════════════════════════════

    // BPE 合并规则（基类共享，子类直接复用，无需各自定义）
    struct MergeRule { std::size_t id_a, id_b, new_id; };

    // 64 位编码 pair：(a, b) → 单个 uint64_t，消除自定义 hash 开销
    // 要求 a, b < 2^32（词表大小不会超过此限制）
    [[nodiscard]] static constexpr std::uint64_t pair_key(std::size_t a, std::size_t b) noexcept
    {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    }

    // 共享 BPE 训练：在已初始化的词表和 ID 序列上执行合并循环。
    // vocab 由子类传入（子类的 vocab_），merges_ / merge_map_ 为基类成员。
    // 训练结束后由调用方负责 rebuild_merge_map() 和 add_dialogue_markers_to_vocab_()。
    //
    // 优化：
    //   1. 输入已去重（chunk + weight 对），调用方负责去重（见各 tokenizer train）
    //   2. 增量 pair 频次更新：只更新 merge 邻域 pair（最多 5 个）
    //   3. 每轮线性扫描 pair_freq 找 max + 扫描 dedup chunks 找 affected
    void bpe_train_impl_(
        std::vector<std::string> &vocab,
        std::vector<std::pair<std::vector<std::size_t>, std::size_t>> &chunks,
        std::size_t target_merges,
        std::uint32_t min_freq,
        const std::function<void(std::string_view)> &log,
        bool show_progress = false)
    {
        const std::size_t num_dedup = chunks.size();
        log("  去重后 chunk 数: " + std::to_string(num_dedup));

        // ── pair 频次表 ───────────────────────────────────────
        std::unordered_map<std::uint64_t, std::size_t> pair_freq;
        for (const auto &[chunk, w] : chunks)
            for (std::size_t i = 0; i + 1 < chunk.size(); ++i)
                pair_freq[pair_key(chunk[i], chunk[i + 1])] += w;

        // ── 合并循环 ───────────────────────────────────────────
        std::size_t last_pct = static_cast<std::size_t>(-1);
        for (std::size_t round = 0; round < target_merges; ++round)
        {
            // 进度条：每变化 1% 刷新一次
            if (show_progress)
            {
                const auto pct = round * 100 / std::max<std::size_t>(target_merges, 1);
                if (pct != last_pct)
                {
                    last_pct = pct;
                    render_progress_("合并", round, target_merges,
                        "词表 " + std::to_string(vocab.size()));
                }
            }

            // 线性扫描找最优 pair。
            // 频次相同时按 pair_key 打破平局（与 unordered_map 迭代顺序无关），
            // 保证顺序执行与并行预分词训练结果完全一致。
            std::uint64_t best_key = 0;
            std::size_t best_freq = 0;
            for (const auto &[key, freq] : pair_freq)
                if (freq > best_freq ||
                    (freq == best_freq && key < best_key))
                { best_freq = freq; best_key = key; }

            if (best_freq < min_freq)
            {
                log("  无更多高频 pair，提前停止于第 " + std::to_string(round) + " 轮");
                break;
            }

            const std::size_t id_a = static_cast<std::size_t>(best_key >> 32);
            const std::size_t id_b = static_cast<std::size_t>(best_key & 0xFFFFFFFF);
            const std::size_t new_id = vocab.size();
            vocab.push_back(vocab[id_a] + vocab[id_b]);
            merges_.push_back({id_a, id_b, new_id});

            // 扫描去重 chunk，处理包含目标 pair 的 chunk
            for (auto &[chunk, weight] : chunks)
            {
                bool has_merge = false;
                for (std::size_t i = 0; i + 1 < chunk.size(); ++i)
                    if (chunk[i] == id_a && chunk[i + 1] == id_b) { has_merge = true; break; }
                if (!has_merge) continue;

                // 增量更新：原地压缩 + 更新 pair_freq
                std::size_t write = 0;
                std::size_t read = 0;
                while (read < chunk.size())
                {
                    bool is_match = (read + 1 < chunk.size() &&
                                    chunk[read] == id_a && chunk[read + 1] == id_b);
                    if (is_match)
                    {
                        auto w = static_cast<std::ptrdiff_t>(weight);
                        auto dec = [&](std::uint64_t k) {
                            auto it = pair_freq.find(k);
                            if (it != pair_freq.end())
                            {
                                if (it->second <= static_cast<std::size_t>(w))
                                    pair_freq.erase(it);
                                else
                                    it->second -= static_cast<std::size_t>(w);
                            }
                        };
                        auto inc = [&](std::uint64_t k) { pair_freq[k] += static_cast<std::size_t>(w); };

                        dec(pair_key(chunk[read], chunk[read + 1]));
                        if (write > 0)
                        {
                            dec(pair_key(chunk[write - 1], chunk[read]));
                            inc(pair_key(chunk[write - 1], new_id));
                        }
                        if (read + 2 < chunk.size())
                        {
                            dec(pair_key(chunk[read + 1], chunk[read + 2]));
                            inc(pair_key(new_id, chunk[read + 2]));
                        }
                        chunk[write] = new_id;
                        ++write;
                        read += 2;
                    }
                    else
                    {
                        if (read != write) chunk[write] = chunk[read];
                        ++write;
                        ++read;
                    }
                }
                chunk.resize(write);
            }
        }

        if (show_progress)
            finish_progress_("合并");
    }

    // 共享 BPE 合并：对单个 ID 序列应用合并规则（优先队列驱动，O(n log n)）。
    // 使用基类的 merges_ 与 merge_priority_。
    [[nodiscard]] std::vector<std::size_t>
    bpe_merge_impl_(std::vector<std::size_t> ids) const
    {
        if (ids.size() <= 1) return ids;

        const std::size_t n = ids.size();
        std::vector<std::size_t> ll_prev(n), ll_next(n);
        for (std::size_t j = 0; j < n; ++j)
        {
            ll_prev[j] = (j == 0) ? n : j - 1;
            ll_next[j] = (j + 1 == n) ? n : j + 1;
        }
        std::vector<bool> alive(n, true);

        using HeapEntry = std::pair<std::size_t, std::size_t>;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

        auto push_pair = [&](std::size_t pos) {
            std::size_t nxt = ll_next[pos];
            if (nxt < n && alive[pos] && alive[nxt])
            {
                std::size_t prio = merge_priority_(ids[pos], ids[nxt]);
                if (prio < merges_.size())
                    heap.push({prio, pos});
            }
        };

        for (std::size_t j = 0; j + 1 < n; ++j)
            push_pair(j);

        while (!heap.empty())
        {
            auto [prio, pos] = heap.top();
            heap.pop();

            if (!alive[pos]) continue;
            std::size_t nxt = ll_next[pos];
            if (nxt >= n || !alive[nxt]) continue;
            if (merge_priority_(ids[pos], ids[nxt]) != prio) continue;

            ids[pos] = merges_[prio].new_id;
            alive[nxt] = false;

            std::size_t prev = ll_prev[pos];
            std::size_t after = ll_next[nxt];
            ll_next[pos] = after;
            if (after < n) ll_prev[after] = pos;

            if (prev < n && alive[prev])
                push_pair(prev);
            push_pair(pos);
        }

        // 按链表顺序收集结果
        std::vector<std::size_t> result;
        std::size_t cur = 0;
        do {
            result.push_back(ids[cur]);
            cur = ll_next[cur];
        } while (cur < n);
        return result;
    }

    // BPE 子类共享的合并规则表与查找表（BPE/CharBPE 使用，其他子类不访问）
    std::vector<MergeRule> merges_;
    std::unordered_map<std::uint64_t, std::size_t> merge_map_;

    // 重建 merge_map_，需要在 merges_ 改变后调用
    void rebuild_merge_map_()
    {
        merge_map_.clear();
        merge_map_.reserve(merges_.size());
        for (std::size_t i = 0; i < merges_.size(); ++i)
            merge_map_[pair_key(merges_[i].id_a, merges_[i].id_b)] = i;
    }

    // O(1) 查找：(a, b) → 优先级（merges_ 中的索引），不存在返回 merges_.size()
    [[nodiscard]] std::size_t merge_priority_(std::size_t a, std::size_t b) const noexcept
    {
        auto it = merge_map_.find(pair_key(a, b));
        return it != merge_map_.end() ? it->second : merges_.size();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  控制台进度条共享构件
    // ═══════════════════════════════════════════════════════════════════════════
    // 使用 \r 原地刷新当前行。子类在 show_progress 时调用。
    // render_progress_ 渲染进度；finish_progress_ 结束并换行。

    // 渲染进度条：\r[label] [####----] 45% tail
    static void render_progress_(std::string_view label,
                                 std::size_t cur, std::size_t total,
                                 std::string_view tail = {})
    {
        constexpr std::size_t BAR_W = 40;
        const auto denom = std::max<std::size_t>(total, 1);
        const std::size_t filled = (total == 0) ? BAR_W
            : std::min(BAR_W, cur * BAR_W / denom);
        std::cout << '\r' << label << " [";
        for (std::size_t i = 0; i < BAR_W; ++i)
            std::cout << (i < filled ? '#' : '-');
        const int pct = (total == 0) ? 100
            : static_cast<int>(std::min<std::size_t>(100, cur * 100 / denom));
        std::cout << "] " << pct << "% " << tail << std::flush;
    }

    // 结束进度条：覆盖为 "[label] 完成" 并换行
    static void finish_progress_(std::string_view label)
    {
        std::cout << '\r' << label << " 完成" << std::string(48, ' ') << '\n' << std::flush;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  JSON save/load 共享构件
    // ═══════════════════════════════════════════════════════════════════════════

    // 写入 JSON 头部 + vocab 段 + vocab_size 字段。
    // 子类继续追加差异段（merges / special_tokens / decompose 等）并以 "\n}\n" 结尾。
    // start_id: 起始写入的 token ID（用于跳过特殊 token 槽位）。
    void save_vocab_json_(std::ofstream &ofs, std::string_view type_str, std::size_t start_id) const
    {
        const auto &v = vocab();
        ofs << "{\n  \"type\": \"" << type_str << "\",\n  \"vocab\": {\n";
        bool first = true;
        for (std::size_t tid = start_id; tid < v.size(); ++tid)
        {
            if (v[tid].empty()) continue;
            if (!first) ofs << ",\n";
            first = false;
            ofs << "    \"" << std::dec << tid << "\": \"";
            write_vocab_entry_(ofs, v[tid], tid);
            ofs << "\"" << std::dec;
        }
        ofs << "\n  },\n  \"vocab_size\": " << std::dec << v.size();
    }

    // 钩子：默认 hex 编码所有字节。
    // CharBPETokenizer 重写以对前 4 个特殊 token (<pad>/<unk>/<bos>/<eos>) 用纯文本。
    virtual void write_vocab_entry_(std::ofstream &ofs, const std::string &tok, std::size_t /*tid*/) const
    {
        for (unsigned char b : tok)
            ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(b);
    }

    // 从 JSON 内容解析 vocab 段，返回按 ID 索引的词表向量。
    // 调用 decode_vocab_entry_(raw, tid) 解码每个条目（默认 hex，CharBPE 重写）。
    std::vector<std::string> parse_vocab_json_(const std::string &content) const
    {
        std::vector<std::string> vocab;
        if (auto p = content.find("\"vocab_size\""); p != std::string::npos)
            if (auto c = content.find(':', p); c != std::string::npos)
            {
                ++c; while (c < content.size() && content[c] == ' ') ++c;
                std::size_t v = 0;
                if (auto [ptr, ec] = std::from_chars(content.data() + c, content.data() + content.size(), v); ec == std::errc{})
                    vocab.resize(v);
            }

        auto vp = content.find("\"vocab\"");
        if (vp == std::string::npos) return vocab;
        auto br = content.find('{', vp);
        if (br == std::string::npos) return vocab;

        std::size_t pos = br + 1;
        while (pos < content.size())
        {
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' ||
                   content[pos] == '\r' || content[pos] == '\t' || content[pos] == ',')) ++pos;
            if (pos >= content.size() || content[pos] == '}') break;
            if (content[pos] != '"') { ++pos; continue; }
            ++pos;
            std::string key;
            while (pos < content.size() && content[pos] != '"') key += content[pos++];
            ++pos;
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == ':')) ++pos;
            if (pos < content.size() && content[pos] == '"')
            {
                ++pos;
                std::string raw;
                while (pos < content.size() && content[pos] != '"')
                {
                    if (content[pos] == '\\' && pos + 1 < content.size()) ++pos;
                    raw += content[pos++];
                }
                ++pos;
                std::size_t id = 0;
                std::from_chars(key.data(), key.data() + key.size(), id);
                std::string tok = decode_vocab_entry_(raw, id);
                if (id >= vocab.size()) vocab.resize(id + 1);
                vocab[id] = std::move(tok);
            }
        }
        return vocab;
    }

    // 钩子：默认将 hex 字符串解码为字节序列。
    // CharBPETokenizer 重写以对前 4 个特殊 token 用纯文本。
    virtual std::string decode_vocab_entry_(const std::string &raw, std::size_t /*tid*/) const
    {
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

    // ═══════════════════════════════════════════════════════════════════════════
    //  并行预分词共享构件（BPETokenizer / CharBPETokenizer 训练复用）
    // ═══════════════════════════════════════════════════════════════════════════
    //  预分词（文本 → 去重 chunk 频次表）是训练中耗时步骤之一，天然可并行：
    //    1. 在"安全切分点"处把文本分成若干段（段与段互不重叠，且不切断任何词）
    //    2. 每段通过全局线程池（global_thread_pool）在独立任务中执行预分词，
    //       各自生成局部频次表（map）
    //    3. 顺序归并为全局频次表（reduce）——结果与单线程完全一致
    //
    //  安全切分点：pos 处是空白、pos-1 处非空白（即"空白串起点"）。
    //  因为 BPE/CharBPE 都是"空白作为下一个词的前缀"（GPT-2 风格），
    //  任何词都不会以空白结尾，因此在空白串起点处切分绝不破坏任何词。

    // 与 BPE 正则 \s / CharBPE is_space_byte 一致的空白判定（ASCII 空白集）
    [[nodiscard]] static constexpr bool is_space_char(char c) noexcept
    {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    // 查找安全切分点：把 [0, text.size()) 分为约 target_segments 段。
    // 返回升序切分点数组，均满足 is_space_char(text[pos]) && !is_space_char(text[pos-1])。
    // 文本空白过少时返回较少切分点（调用方回退到更少并行任务）。
    [[nodiscard]] static std::vector<std::size_t>
    find_safe_splits(std::string_view text, std::size_t target_segments)
    {
        std::vector<std::size_t> splits;
        if (target_segments <= 1 || text.size() < 2) return splits;

        // 收集全部安全切分点（空白串起点）
        std::vector<std::size_t> safe;
        safe.reserve(text.size() / 16);
        for (std::size_t i = 1; i < text.size(); ++i)
            if (is_space_char(text[i]) && !is_space_char(text[i - 1]))
                safe.push_back(i);
        if (safe.empty()) return splits;

        // 贪心选取最接近理想均匀位置的切分点
        const std::size_t n_seg = std::min(target_segments, safe.size() + 1);
        splits.reserve(n_seg - 1);
        for (std::size_t k = 1; k < n_seg; ++k)
        {
            const std::size_t ideal = text.size() * k / n_seg;
            auto it = std::lower_bound(safe.begin(), safe.end(), ideal);
            std::size_t best = 0;
            if (it == safe.end())
                best = safe.back();
            else if (it == safe.begin())
                best = *it;
            else
            {
                auto prev = it - 1;
                best = (ideal - *prev <= *it - ideal) ? *prev : *it;
            }
            if (splits.empty() || best > splits.back())
                splits.push_back(best);
        }
        return splits;
    }

    // 并行预分词框架（map-reduce，复用项目全局线程池）：
    //   worker(begin, end, local, progress) 对 [begin,end) 执行预分词，
    //     将 chunk 频次累加到 local；progress(pos) 可周期报告已处理偏移。
    //   threads: 0=自动(线程池全部核心), 1=顺序, >1=指定并行度（上限=线程池大小）。
    //   找不到安全切分点时自动回退顺序执行，结果与并行完全一致。
    //   返回合并后的全局频次表。
    static std::unordered_map<std::string, std::size_t>
    parallel_pretokenize(
        const std::string &text,
        std::size_t threads,
        std::string_view label,
        bool show_progress,
        const std::function<void(std::size_t, std::size_t,
                                 std::unordered_map<std::string, std::size_t> &,
                                 const std::function<void(std::size_t)> &)> &worker)
    {
        std::unordered_map<std::string, std::size_t> global;

        // 顺序回退：进度直接按字节偏移渲染
        auto run_sequential = [&] {
            const std::function<void(std::size_t)> progress =
                [&](std::size_t pos) {
                    if (show_progress) render_progress_(label, pos, text.size());
                };
            worker(0, text.size(), global, progress);
        };

        if (text.size() < 2 || threads == 1) { run_sequential(); return global; }

        auto &pool = global_thread_pool();
        if (threads == 0)
            threads = pool.size();                     // 自动：线程池全部核心
        else if (threads > 1)
            threads = std::min(threads, pool.size());  // 指定并行度，收敛到池大小
        if (threads <= 1) { run_sequential(); return global; }

        const auto splits = find_safe_splits(text, threads);
        if (splits.size() + 1 < 2) { run_sequential(); return global; }

        // 段边界（互不重叠，覆盖整个文本）
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(splits.size() + 1);
        std::size_t begin = 0;
        for (auto sp : splits) { ranges.emplace_back(begin, sp); begin = sp; }
        ranges.emplace_back(begin, text.size());

        // 进度：原子字节计数 + 进度线程（并行任务运行期间轮询渲染）
        std::atomic<std::size_t> bytes_done{0};
        std::thread progress_thr;
        if (show_progress)
            progress_thr = std::thread([&] {
                std::size_t last = static_cast<std::size_t>(-1);
                while (true)
                {
                    const auto d = bytes_done.load(std::memory_order_relaxed);
                    if (d != last)
                    {
                        last = d;
                        render_progress_(label, d, text.size());
                    }
                    if (d >= text.size()) break;
                    std::this_thread::yield();
                }
            });

        // 每段一个局部频次表（无共享写，天然无竞争），通过全局线程池并行执行
        const std::size_t n = ranges.size();
        std::vector<std::unordered_map<std::string, std::size_t>> locals(n);
        pool.parallel_for_samples(n, [&](std::size_t k) {
            const std::function<void(std::size_t)> noop = [](std::size_t) {};
            worker(ranges[k].first, ranges[k].second, locals[k], noop);
            bytes_done.fetch_add(ranges[k].second - ranges[k].first,
                                 std::memory_order_relaxed);
        });

        if (progress_thr.joinable()) progress_thr.join();

        // 归并局部频次表 → 全局
        std::size_t total = 0;
        for (const auto &m : locals) total += m.size();
        global.reserve(total);
        for (const auto &m : locals)
            for (const auto &[w, c] : m)
                global[w] += c;
        return global;
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

public:
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

        // ── 添加对话标记 token ──────────────────────────────────────
        add_dialogue_markers_to_vocab_(vocab_);

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
        std::size_t pos = 0;
        while (pos < text.size())
        {
            // 对话标记优先匹配
            if (has_dialogue_markers())
            {
                auto [mid, mlen] = try_match_marker(text, pos);
                if (mid != Tokenizer::npos)
                { all.push_back(mid); pos += mlen; continue; }
            }
            // 普通文本按预分词分块
            if (auto n = WordZipTokenizer::try_contraction(text, pos); n > 0)
            { all.push_back(UNK_ID); pos += n; continue; }
            auto s = pos;
            while (pos < text.size())
            {
                if (has_dialogue_markers())
                { auto [mid, mlen] = try_match_marker(text, pos); if (mid != Tokenizer::npos) break; }
                if (WordZipTokenizer::try_contraction(text, pos) > 0 && pos > s) break;
                ++pos;
            }
            if (pos > s)
            {
                auto ids = encode_bytes(std::string_view{text.data() + s, pos - s});
                all.insert(all.end(), ids.begin(), ids.end());
            }
        }
        return all;
    }

    // ── 解码 ──────────────────────────────────────────────────────────
    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string raw;
        raw.reserve(ids.size() * 2);
        for (auto tid : ids)
        {
            // 对话标记 → 原始文本
            if (tid == markers_.system)        { raw += "<|system|>";          continue; }
            if (tid == markers_.system_end)    { raw += "<|end_of_system|>";   continue; }
            if (tid == markers_.user)          { raw += "<|user|>";            continue; }
            if (tid == markers_.user_end)      { raw += "<|end_of_user|>";     continue; }
            if (tid == markers_.assistant)     { raw += "<|assistant|>";       continue; }
            if (tid == markers_.assistant_end) { raw += "<|end_of_assistant|>";continue; }
            raw += tid < vocab_.size() ? vocab_[tid] : std::string{REPLACEMENT_CHAR};
        }
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

        // ── 恢复对话标记 ID（从已有词表检测或追加） ─────────────
        restore_dialogue_markers(vocab_);
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
//  注：未在生产线使用，仅保留作对比实现（与 WordZip/BPE 对照分词效果）。
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

    [[nodiscard]] std::size_t pad_id() const noexcept override { return PAD_ID; }

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

        // ── 添加对话标记 token ──────────────────────────────────────
        add_dialogue_markers_to_vocab_(id_to_token_);
        build_lookup();
        log("最终词表: " + std::to_string(id_to_token_.size()));
        return {};
    }

    [[nodiscard]] std::vector<std::size_t> encode(const std::string &text) const override
    {
        std::vector<std::size_t> tokens;
        std::size_t pos = 0;
        while (pos < text.size())
        {
            // 对话标记优先匹配
            if (has_dialogue_markers())
            {
                auto [mid, mlen] = try_match_marker(text, pos);
                if (mid != Tokenizer::npos)
                { tokens.push_back(mid); pos += mlen; continue; }
            }
            // 跳过空白
            if (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')
            { ++pos; continue; }
            // 提取一个词
            auto s = pos;
            while (pos < text.size() && text[pos] != ' ' && text[pos] != '\t' &&
                   text[pos] != '\n' && text[pos] != '\r')
            {
                if (has_dialogue_markers())
                { auto [mid, mlen] = try_match_marker(text, pos); if (mid != Tokenizer::npos) break; }
                ++pos;
            }
            std::string word(text, s, pos - s);
            if (word.empty()) continue;
            if (is_numeric(word))
                tokens.push_back(NUM_ID);
            else if (auto it = token_to_id_.find(word); it != token_to_id_.end())
                tokens.push_back(it->second);
            else
                for (unsigned char c : word)
                {
                    auto id = static_cast<std::size_t>(c) + ASCII_BASE;
                    tokens.push_back(id < id_to_token_.size() ? id : UNK_ID);
                }
        }
        return tokens;
    }

    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const override
    {
        std::string result;
        for (auto id : ids)
        {
            // 对话标记 → 原始文本
            if (id == markers_.system)        { result += "<|system|>";          continue; }
            if (id == markers_.system_end)    { result += "<|end_of_system|>";   continue; }
            if (id == markers_.user)          { result += "<|user|>";            continue; }
            if (id == markers_.user_end)      { result += "<|end_of_user|>";     continue; }
            if (id == markers_.assistant)     { result += "<|assistant|>";       continue; }
            if (id == markers_.assistant_end) { result += "<|end_of_assistant|>";continue; }
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
        restore_dialogue_markers(id_to_token_);
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
            auto begin = std::sregex_iterator(s.begin(), s.end(), pre_pattern());
            auto end   = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) chunks.push_back(it->str());
        };

        std::size_t pos = 0;
        while (pos < text.size())
        {
            std::size_t mlen = 0;
            for (const auto &mk : markers)
                if (pos + mk.size() <= text.size() && text.substr(pos, mk.size()) == mk)
                { mlen = mk.size(); break; }
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
    std::vector<std::string> vocab_;
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
