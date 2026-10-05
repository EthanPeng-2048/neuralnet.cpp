#pragma once

// ══════════════════════════════════════════════════════════════════════════
//  dataset_format.hpp — .nnvocab / .nndataset 文件格式公共件
//  （设计见 docs/development/19-unified-dataset.md §4）
//
//  文件标签（两类文件通用）：
//    [label] := [magic 4B][file_version u32]
//      .nndataset: magic = 'N''N''D''S'   file_version = 1
//      .nnvocab:   magic = 'N''N''V''C'   file_version = 1
//    标签之后紧跟文件级 kvrec v2。嵌套进别的文件的 kvrec 没有标签
//    （靠哨兵判版本）。`file_version`（文件布局演进）与 kvrec 内部
//    `kvrec_version`（记录编码演进）是两条独立的演进轴。
//
//  v2 地址是文件内绝对偏移，因此 kvrec v2 必须以"它在文件中的真实起点"
//  为 base 序列化（.nnvocab = 标签后 8 字节处），解析时以同一文件映像
//  为字节源——整体搬迁的字节会被 KeyValueRecord 的地址连续性校验抓住。
// ══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core_errors.hpp"
#include "model_keyvalue_record.hpp"

namespace nn
{

// ── 文件标签 ─────────────────────────────────────────────────────────
inline constexpr char NNVOCAB_MAGIC[4]   = {'N', 'N', 'V', 'C'};
inline constexpr char NNDATASET_MAGIC[4] = {'N', 'N', 'D', 'S'};
inline constexpr std::uint32_t NNVOCAB_FILE_VERSION   = 1;
inline constexpr std::uint32_t NNDATASET_FILE_VERSION = 1;
inline constexpr std::size_t FILE_LABEL_SIZE = 8;  // magic 4B + file_version u32

[[nodiscard]] inline std::string write_file_label(const char (&magic)[4],
                                                  std::uint32_t file_version)
{
    std::string out;
    out.append(magic, 4);
    detail::append_u32(out, file_version);
    return out;
}

struct FileLabel
{
    std::string magic;              // 4 字节
    std::uint32_t file_version = 0;
};

[[nodiscard]] inline Result<FileLabel> read_file_label(std::string_view bytes)
{
    if (bytes.size() < FILE_LABEL_SIZE)
        NN_FAIL("数据集文件过短（缺少文件标签）");
    FileLabel label;
    label.magic.assign(bytes.substr(0, 4));
    std::string_view rest = bytes.substr(4, 4);
    if (!detail::take_u32(rest, label.file_version))
        NN_FAIL("数据集文件标签损坏");
    return label;
}

// 校验文件标签；magic 不符时给出针对性提示（旧 JSON 词表 → 指路转换器）
[[nodiscard]] inline Result<void>
check_file_label(std::string_view bytes, const char (&magic)[4],
                 std::string_view kind_name)
{
    NN_TRY(label, read_file_label(bytes));
    if (label->magic != std::string_view(magic, 4))
    {
        if (!bytes.empty() && bytes[0] == '{')
            NN_FAIL(std::string(kind_name) + ": 这是旧 JSON 格式，不再直读；"
                "请先用 `dataset_convert vocab <in.json> -o <out.nnvocab>` 转换");
        NN_FAIL(std::string(kind_name) + ": 文件标签不匹配（不是 "
            + std::string(kind_name) + " 文件）");
    }
    return {};
}

// ── 词表 blob（§4.2 "vocab" 字段值，Str 类型下的自描述结构）──────────
//    [count u64] [ [len u64][token raw bytes] × count ]
//    token 原样存字节（无 hex 转义）；空 token（len=0）也占位，
//    保证 ID 逐位还原。
[[nodiscard]] inline std::string
encode_vocab_blob(const std::vector<std::string> &vocab)
{
    std::string out;
    detail::append_u64(out, static_cast<std::uint64_t>(vocab.size()));
    for (const auto &tok : vocab)
    {
        detail::append_u64(out, static_cast<std::uint64_t>(tok.size()));
        out.append(tok);
    }
    return out;
}

[[nodiscard]] inline Result<std::vector<std::string>>
decode_vocab_blob(std::string_view value)
{
    std::uint64_t count = 0;
    if (!detail::take_u64(value, count))
        NN_FAIL("decode_vocab_blob: 缺少词表计数");
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i)
    {
        std::uint64_t len = 0;
        if (!detail::take_u64(value, len))
            NN_FAIL("decode_vocab_blob: 词表条目长度越界");
        if (len > value.size())
            NN_FAIL("decode_vocab_blob: 词表条目数据不足");
        vocab.emplace_back(value.substr(0, static_cast<std::size_t>(len)));
        value.remove_prefix(static_cast<std::size_t>(len));
    }
    return vocab;
}

// ── .nnvocab 文件编解码（[label "NNVC"] + 词表 kvrec v2 @ base=8）────
[[nodiscard]] inline std::string write_nnvocab(const KeyValueRecord &vocab_rec)
{
    std::string out = write_file_label(NNVOCAB_MAGIC, NNVOCAB_FILE_VERSION);
    out += vocab_rec.serialize_v2(FILE_LABEL_SIZE);
    return out;
}

[[nodiscard]] inline Result<KeyValueRecord>
parse_nnvocab(std::string_view file_bytes)
{
    NN_TRY_CHECK(check_file_label(file_bytes, NNVOCAB_MAGIC, ".nnvocab"));
    return KeyValueRecord::parse_at(kvrec_memory_source(file_bytes),
                                    FILE_LABEL_SIZE);
}

}  // namespace nn
