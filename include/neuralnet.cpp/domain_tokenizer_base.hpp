#ifndef NN_DOMAIN_TOKENIZER_BASE_HPP
#define NN_DOMAIN_TOKENIZER_BASE_HPP

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
        // 所有标记（6 对话 + 保留 token）都以 '<' 开头：非 '<' 位置直接返回，
        // 避免对每个普通字符做线性扫描与 substr 分配。
        if (pos >= text.size() || text[pos] != '<')
            return {npos, 0};

        static constexpr std::string_view markers[] = {
            "<|system|>", "<|end_of_system|>", "<|user|>", "<|end_of_user|>",
            "<|assistant|>", "<|end_of_assistant|>"
        };
        for (const auto &m : markers)
        {
            if (pos + m.size() <= text.size() && text.compare(pos, m.size(), m) == 0)
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
                && text.compare(pos, tok.size(), tok) == 0)
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
} // namespace nn

#endif // NN_DOMAIN_TOKENIZER_BASE_HPP
