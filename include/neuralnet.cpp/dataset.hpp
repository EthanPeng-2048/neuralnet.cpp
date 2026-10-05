#pragma once

// ══════════════════════════════════════════════════════════════════════════
//  dataset.hpp — 统一数据集：.nndataset 读写 + nn::Dataset 只读类
//  （设计见 docs/development/19-unified-dataset.md §4/§5）
//
//  文件布局（§4.3）：
//    [label "NNDS" + file_version=1][文件级 kvrec v2][doc 块 arena ...]
//    文件级 kvrec：
//      "kind"      := UInt   0=text（本轮）| 1=tabular（预留，§4.4）
//      "companion" := Record 配套：
//                        "vocab"   := Record 词表 kvrec（三层嵌套的最内层）
//                        "source"  := Record { "path" Str, "sha256" Str, "docs" UInt }
//                        "license" := Str  纯文本文档全文（可缺省；不是 kvrec）
//                        "gen"     := Record { "loss_scope" Str, "tool" Str, "created" UInt }
//      "train"     := Record 子集（§4.3.1）
//      "test"      := Record 子集（结构与 train 相同；可缺省）
//    子集 kvrec：
//      "num_docs"  := UInt   doc 数
//      "doc_index" := UIntArray(num_docs + 1)  各 doc 块的文件绝对偏移，末项 = 末块结尾
//    doc 块（经 doc_index 寻址）：
//      [token_count u64][token_count × u64 tokens][token_count × u8 loss_mask]
//      （loss_mask 段仅 loss_scope=assistant 时存在；u64 token 宽度 = 裁决 #14）
//
//  doc_id 不存储：= doc 在子集中的序号 + 1（§4.3.1，加载端按 doc_index 顺序重建）。
//  掩码不存储为全 1（loss_scope=all）：加载端按 gen.loss_scope 合成（裁决 #11 的体积实施）。
//
//  只读、不转换、不做窗口化——窗口是训练策略，留在训练入口（裁决 #3）。
// ══════════════════════════════════════════════════════════════════════════

#include <chrono>
#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core_errors.hpp"
#include "core_file.hpp"
#include "core_sha256.hpp"
#include "dataset_format.hpp"
#include "model_keyvalue_record.hpp"

namespace nn
{

enum class DatasetKind : std::uint64_t { Text = 0, Tabular = 1 };

struct DatasetInfo
{
    DatasetKind kind = DatasetKind::Text;
    std::uint64_t num_docs = 0;     // train 子集 doc 数
    std::uint64_t test_docs = 0;    // test 子集 doc 数（0 = 无 test 子集）
    std::string loss_scope = "all"; // "all" | "assistant"（来自配套 gen）
    std::string source_path;        // 源文本路径/名称
    std::string source_sha256;      // 源文本内容摘要（诊断/复现用）
};

// 与现 tokcache + mark_assistant_spans 产物同构（§4.3.2）：
// 下游"拼流 → 滑窗 → 训练"代码一行不改。
struct TextCorpus
{
    std::vector<std::size_t> token_flow;
    std::vector<std::size_t> doc_ids;      // 每 token 的文档 id（序号 + 1）
    std::vector<unsigned char> loss_mask;  // loss_scope=all 时为全 1
};

// ══════════════════════════════════════════════════════════════════════════
//  生成期公共件（dataset_gen 与 dataset_test 共用）
//  §4.3.2 一致性硬约束的三段逻辑逐字落在本头：
//    1. 行 trim 规则（read_text_docs）      —— 原 text_train read_file_lines
//    2. 并行 tokenize 保序（encode_docs_parallel，domain_tokenizer_base.hpp）
//    3. 掩码计算的 doc 边界重置语义（mark_assistant_span 逐 doc 独立 = 全局扫描）
// ══════════════════════════════════════════════════════════════════════════

// 行 trim 规则（逐字搬自 read_file_lines：首尾去空格/`\t`/`\r`、跳过空行）
[[nodiscard]] inline Result<std::vector<std::string>>
read_text_docs(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        NN_FAIL("无法打开文件: " + path);
    std::string buffer;
    buffer.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    std::vector<std::string> docs;
    std::size_t pos = 0;
    const std::size_t total = buffer.size();
    while (pos < total)
    {
        std::size_t line_start = pos;
        while (pos < total && buffer[pos] != '\n') ++pos;
        std::size_t line_end = pos;
        if (pos < total) ++pos;  // skip '\n'
        while (line_start < line_end &&
               (buffer[line_start] == ' ' || buffer[line_start] == '\t'
                || buffer[line_start] == '\r'))
            ++line_start;
        while (line_end > line_start &&
               (buffer[line_end - 1] == ' ' || buffer[line_end - 1] == '\t'
                || buffer[line_end - 1] == '\r'))
            --line_end;
        if (line_start < line_end)
            docs.emplace_back(buffer, line_start, line_end - line_start);
    }
    return docs;
}

// 逐 doc 的 assistant 段掩码（裁决 #11）：
// 把 <|assistant|>…<|end_of_assistant|> 段（含首尾标记）标为 1。
// 逐 doc 独立计算与原全局扫描**逐位一致**——原逻辑在 doc 边界重置段状态。
[[nodiscard]] inline std::vector<unsigned char>
mark_assistant_span(std::span<const std::size_t> tokens,
                    std::size_t asst_begin, std::size_t asst_end)
{
    std::vector<unsigned char> mask(tokens.size(), 0);
    bool in_asst = false;
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        const std::size_t id = tokens[i];
        if (id == asst_begin) in_asst = true;
        mask[i] = in_asst ? 1u : 0u;
        if (id == asst_end) in_asst = false;
    }
    return mask;
}

// ── doc 块编码（写入侧）────────────────────────────────────────────
// mask 为空 = 不写 mask 段（loss_scope=all）。
[[nodiscard]] inline std::uint64_t doc_block_size(std::size_t token_count, bool has_mask)
{
    return 8 + static_cast<std::uint64_t>(token_count) * 8
        + (has_mask ? static_cast<std::uint64_t>(token_count) : 0);
}

[[nodiscard]] inline std::string
encode_doc_block(std::span<const std::size_t> tokens,
                 std::span<const unsigned char> mask)
{
    std::string out;
    detail::append_u64(out, static_cast<std::uint64_t>(tokens.size()));
    for (std::size_t t : tokens)
        detail::append_u64(out, static_cast<std::uint64_t>(t));
    for (unsigned char m : mask)
        out.push_back(static_cast<char>(m));
    return out;
}

// ── .nndataset 写入（dataset_gen 与 dataset_test 共用）────────────────
// 两遍序列化：doc_index 的值要等文件级 kvrec 字节块定长后才能算出
// （kvrec v2 数据块连续排布，长度与取值无关 → 第二遍字节数不变）。
struct NndatasetBuild
{
    KeyValueRecord vocab_kvrec;              // companion.vocab（词表 kvrec）
    std::string source_path;                 // companion.source.path
    std::string source_sha256;               // companion.source.sha256（内容摘要）
    std::uint64_t source_docs = 0;           // companion.source.docs（源文档行数）
    std::string loss_scope = "all";          // gen.loss_scope: "all" | "assistant"
    std::string license_text;                // companion.license（可选，纯文本）
    std::string tool = "dataset_gen v1";     // gen.tool
    std::uint64_t created = 0;               // gen.created（unix 时间戳）
    std::vector<std::vector<std::size_t>> train_docs;
    std::vector<std::vector<unsigned char>> train_masks;  // assistant 模式逐 doc（可空）
    std::vector<std::vector<std::size_t>> test_docs;
    std::vector<std::vector<unsigned char>> test_masks;
};

namespace detail
{

// 组装子集 kvrec（doc_index 值由调用方给定：占位或真实偏移）
inline KeyValueRecord make_subset_record(
    std::size_t num_docs, const std::vector<std::uint64_t> &index)
{
    KeyValueRecord sub;
    sub.set("num_docs", static_cast<std::uint64_t>(num_docs));
    sub.set("doc_index", index);
    return sub;
}

inline KeyValueRecord make_companion_record(const NndatasetBuild &b)
{
    KeyValueRecord companion;
    companion.set("vocab", b.vocab_kvrec);

    KeyValueRecord source;
    source.set("path", b.source_path);
    source.set("sha256", b.source_sha256);
    source.set("docs", b.source_docs);
    companion.set("source", source);

    if (!b.license_text.empty())
        companion.set("license", b.license_text);

    KeyValueRecord gen;
    gen.set("loss_scope", b.loss_scope);
    gen.set("tool", b.tool);
    gen.set("created", b.created);
    companion.set("gen", gen);
    return companion;
}

inline KeyValueRecord make_root_record(
    const NndatasetBuild &b,
    const std::vector<std::uint64_t> &train_index,
    const std::vector<std::uint64_t> &test_index)
{
    KeyValueRecord root;
    root.set("kind", static_cast<std::uint64_t>(DatasetKind::Text));
    root.set("companion", make_companion_record(b));
    root.set("train", make_subset_record(b.train_docs.size(), train_index));
    if (!b.test_docs.empty())
        root.set("test", make_subset_record(b.test_docs.size(), test_index));
    return root;
}

}  // namespace detail

[[nodiscard]] inline Result<void>
write_nndataset(std::ostream &os, const NndatasetBuild &b)
{
    const bool has_mask = (b.loss_scope == "assistant");
    if (has_mask)
    {
        if (b.train_masks.size() != b.train_docs.size())
            NN_FAIL("write_nndataset: assistant 模式要求 train_masks 与 train_docs 一一对应");
        if (!b.test_docs.empty() && b.test_masks.size() != b.test_docs.size())
            NN_FAIL("write_nndataset: assistant 模式要求 test_masks 与 test_docs 一一对应");
    }
    else if (!b.train_masks.empty() || !b.test_masks.empty())
    {
        NN_FAIL("write_nndataset: loss_scope=all 时不写 mask 段（masks 应为空）");
    }

    // ── 第一遍：占位 doc_index（长度正确即可）→ 定出 kvrec 字节块大小 ──
    const std::vector<std::uint64_t> zeros_train(b.train_docs.size() + 1, 0);
    const std::vector<std::uint64_t> zeros_test(b.test_docs.size() + 1, 0);
    const std::string probe = detail::make_root_record(b, zeros_train, zeros_test)
        .serialize_v2(FILE_LABEL_SIZE);
    const std::uint64_t arena_begin = FILE_LABEL_SIZE + probe.size();

    // ── 算出各 doc 块真实绝对偏移（train arena 在前、test 在后）────────
    std::vector<std::uint64_t> train_index(b.train_docs.size() + 1, 0);
    std::vector<std::uint64_t> test_index(b.test_docs.size() + 1, 0);
    std::uint64_t cursor = arena_begin;
    for (std::size_t i = 0; i < b.train_docs.size(); ++i)
    {
        train_index[i] = cursor;
        cursor += doc_block_size(b.train_docs[i].size(),
                                 has_mask && !b.train_masks.empty());
    }
    train_index[b.train_docs.size()] = cursor;
    for (std::size_t i = 0; i < b.test_docs.size(); ++i)
    {
        test_index[i] = cursor;
        cursor += doc_block_size(b.test_docs[i].size(),
                                 has_mask && !b.test_masks.empty());
    }
    test_index[b.test_docs.size()] = cursor;

    // ── 第二遍：真实 doc_index → 序列化（字节数与第一遍相同）──────────
    const std::string root = detail::make_root_record(b, train_index, test_index)
        .serialize_v2(FILE_LABEL_SIZE);
    if (root.size() != probe.size())
        NN_FAIL("write_nndataset: 两遍序列化字节数不一致（内部错误）");

    os.write(root.data(), static_cast<std::streamsize>(root.size()));

    // ── doc 块 arena（顺序 = doc_index 指向）──────────────────────────
    auto emit_docs = [&](const std::vector<std::vector<std::size_t>> &docs,
                         const std::vector<std::vector<unsigned char>> &masks) -> Result<void>
    {
        for (std::size_t i = 0; i < docs.size(); ++i)
        {
            std::span<const unsigned char> mask;
            if (has_mask && !masks.empty())
            {
                if (masks[i].size() != docs[i].size())
                    NN_FAIL("write_nndataset: 第 " + std::to_string(i)
                        + " 个 doc 的掩码长度与 token 数不一致");
                mask = masks[i];
            }
            const std::string block = encode_doc_block(docs[i], mask);
            os.write(block.data(), static_cast<std::streamsize>(block.size()));
        }
        return {};
    };
    NN_TRY_CHECK(emit_docs(b.train_docs, b.train_masks));
    NN_TRY_CHECK(emit_docs(b.test_docs, b.test_masks));
    if (!os)
        NN_FAIL("write_nndataset: 写文件失败");
    return {};
}

// ══════════════════════════════════════════════════════════════════════════
//  nn::Dataset —— 统一只读类（嗅探 label magic 分派）
// ══════════════════════════════════════════════════════════════════════════

namespace detail
{

// 按绝对偏移读文件片段（KeyValueRecord v2 的 ReadAt 源）
[[nodiscard]] inline Result<KeyValueRecord>
parse_kvrec_in_file(std::ifstream &ifs, std::uint64_t offset)
{
    ifs.clear();
    ifs.seekg(0, std::ios::end);
    const auto fsize = static_cast<std::uint64_t>(ifs.tellg());
    KeyValueRecord::ReadAt read =
        [&ifs, fsize](std::uint64_t addr, std::uint64_t len) -> Result<std::string>
    {
        if (addr > fsize || len > fsize - addr)
            NN_FAIL("数据集文件读取越界（addr=" + std::to_string(addr)
                + ", len=" + std::to_string(len) + "）");
        std::string buf(static_cast<std::size_t>(len), '\0');
        ifs.clear();
        ifs.seekg(static_cast<std::streamoff>(addr), std::ios::beg);
        if (!ifs)
            NN_FAIL("数据集文件 seek 失败");
        ifs.read(buf.data(), static_cast<std::streamsize>(len));
        if (!ifs)
            NN_FAIL("数据集文件读取失败");
        return buf;
    };
    return KeyValueRecord::parse_at(read, offset);
}

// 读一个 doc 块（含可选 mask 段）
[[nodiscard]] inline Result<std::pair<std::vector<std::size_t>, std::vector<unsigned char>>>
read_doc_block(std::ifstream &ifs, std::uint64_t offset, bool has_mask)
{
    ifs.clear();
    ifs.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!ifs)
        NN_FAIL("数据集文件 seek 失败");
    std::uint64_t count = 0;
    if (!read_pod(ifs, count))
        NN_FAIL("数据集 doc 块读取失败（token_count）");
    std::vector<std::size_t> tokens(static_cast<std::size_t>(count));
    for (auto &t : tokens)
    {
        std::uint64_t v = 0;
        if (!read_pod(ifs, v))
            NN_FAIL("数据集 doc 块读取失败（tokens）");
        t = static_cast<std::size_t>(v);
    }
    std::vector<unsigned char> mask;
    if (has_mask)
    {
        mask.resize(tokens.size());
        if (!read_pod_span(ifs, std::span(mask)))
            NN_FAIL("数据集 doc 块读取失败（loss_mask）");
    }
    return std::make_pair(std::move(tokens), std::move(mask));
}

}  // namespace detail

class Dataset
{
public:
    // 嗅探 label magic 分派 reader：.nndataset / .nnvocab 都能开；
    // 其它扩展/坏文件 → 明确错误。
    [[nodiscard]] static Result<Dataset> open(const std::string &path)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            NN_FAIL("无法打开数据集文件: " + path);
        ifs.seekg(0, std::ios::end);
        const auto fsize = static_cast<std::uint64_t>(ifs.tellg());
        ifs.seekg(0, std::ios::beg);
        std::string label(FILE_LABEL_SIZE, '\0');
        if (fsize >= FILE_LABEL_SIZE)
            ifs.read(label.data(), static_cast<std::streamsize>(FILE_LABEL_SIZE));
        else
            label.clear();
        NN_TRY(label_r, read_file_label(label));

        Dataset ds;
        ds.path_ = path;
        if (label_r->magic == std::string_view(NNVOCAB_MAGIC, 4))
        {
            // .nnvocab：独立词表文件（词表独立加载场景），无数据集子集
            ifs.clear();
            ifs.seekg(0, std::ios::beg);
            std::string content(static_cast<std::size_t>(fsize), '\0');
            ifs.read(content.data(), static_cast<std::streamsize>(fsize));
            if (!ifs)
                NN_FAIL("读取词表文件失败: " + path);
            NN_TRY(rec, parse_nnvocab(content));
            ds.vocab_ = std::move(*rec);
            ds.is_vocab_only_ = true;
            return ds;
        }
        if (label_r->magic != std::string_view(NNDATASET_MAGIC, 4))
            NN_FAIL("不是 .nndataset / .nnvocab 文件: " + path);
        if (label_r->file_version != NNDATASET_FILE_VERSION)
            NN_FAIL("不支持的 .nndataset 文件版本: "
                + std::to_string(label_r->file_version));

        NN_TRY(root, detail::parse_kvrec_in_file(ifs, FILE_LABEL_SIZE));

        // companion
        KeyValueRecord companion;
        if (!root->get("companion", companion))
            NN_FAIL(".nndataset 缺少 companion 配套记录");
        if (!companion.get("vocab", ds.vocab_))
            NN_FAIL(".nndataset 配套缺少 vocab 词表记录");
        KeyValueRecord source;
        if (companion.get("source", source))
        {
            (void)source.get("path", ds.info_.source_path);
            (void)source.get("sha256", ds.info_.source_sha256);
        }
        KeyValueRecord gen;
        if (companion.get("gen", gen))
            (void)gen.get("loss_scope", ds.info_.loss_scope);
        if (ds.info_.loss_scope != "all" && ds.info_.loss_scope != "assistant")
            NN_FAIL(".nndataset gen.loss_scope 非法: " + ds.info_.loss_scope);

        std::uint64_t kind = 0;
        (void)root->get("kind", kind);
        ds.info_.kind = (kind == 1) ? DatasetKind::Tabular : DatasetKind::Text;

        // 子集
        KeyValueRecord train;
        if (!root->get("train", train))
            NN_FAIL(".nndataset 缺少 train 子集");
        NN_TRY_CHECK(ds.load_subset_(train, ds.train_index_, ds.info_.num_docs));
        KeyValueRecord test;
        if (root->get("test", test))
        {
            NN_TRY_CHECK(ds.load_subset_(test, ds.test_index_, ds.info_.test_docs));
        }
        return ds;
    }

    [[nodiscard]] const DatasetInfo &info() const noexcept { return info_; }

    // 词表 kvrec（companion.vocab；模型 v6 内嵌词表的字节来源）
    [[nodiscard]] const KeyValueRecord &vocab_kvrec() const noexcept { return vocab_; }

    // 一次性给出训练所需的全部三样（流式留待 P2-4 后半）。
    // test=false → train 子集；true → test 子集（无 test 子集则报错）。
    [[nodiscard]] Result<TextCorpus> load_text(bool test = false) const
    {
        if (is_vocab_only_)
            NN_FAIL("该文件是 .nnvocab 词表，不含数据集子集: " + path_);
        if (info_.kind != DatasetKind::Text)
            NN_FAIL("tabular 数据集尚未支持（阶段四预留，见 docs/development/19 §4.4）");
        const std::vector<std::uint64_t> &index = test ? test_index_ : train_index_;
        if (test && info_.test_docs == 0)
            NN_FAIL("该数据集没有 test 子集");
        const bool has_mask = (info_.loss_scope == "assistant");

        std::ifstream ifs(path_, std::ios::binary);
        if (!ifs)
            NN_FAIL("无法打开数据集文件: " + path_);

        TextCorpus out;
        const std::size_t n = index.empty() ? 0 : index.size() - 1;
        for (std::size_t i = 0; i < n; ++i)
        {
            NN_TRY(blk, detail::read_doc_block(ifs, index[i], has_mask));
            auto &[tokens, mask] = *blk;
            const std::size_t doc_id = i + 1;   // 文档 id（1 起，序号 + 1）
            for (std::size_t t : tokens)
            {
                out.token_flow.push_back(t);
                out.doc_ids.push_back(doc_id);
            }
            if (has_mask)
                out.loss_mask.insert(out.loss_mask.end(), mask.begin(), mask.end());
            else
                out.loss_mask.insert(out.loss_mask.end(), tokens.size(), 1u);
        }
        return out;
    }

private:
    Result<void> load_subset_(const KeyValueRecord &sub,
                              std::vector<std::uint64_t> &index,
                              std::uint64_t &num_docs)
    {
        std::uint64_t n = 0;
        if (!sub.get("num_docs", n))
            NN_FAIL("数据集子集缺少 num_docs");
        if (!sub.get("doc_index", index))
            NN_FAIL("数据集子集缺少 doc_index");
        if (index.size() != n + 1)
            NN_FAIL("数据集子集 doc_index 长度与 num_docs 不符");
        num_docs = n;
        return {};
    }

    std::string path_;
    DatasetInfo info_;
    KeyValueRecord vocab_;
    std::vector<std::uint64_t> train_index_;
    std::vector<std::uint64_t> test_index_;
    bool is_vocab_only_ = false;
};

}  // namespace nn
