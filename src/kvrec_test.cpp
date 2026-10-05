// ── kvrec_test — KVRecord v1/v2 解析与序列化测试 ───────────────────────────
// 覆盖（docs/development/19-unified-dataset.md §3 / §9 阶段一验收）：
//   1. v1 round-trip（模型 spec 头路径，字节语义不变）
//   2. 版本自举：无版本字段 = v1、有 = v2（同一入口 parse 分派）
//   3. v2 round-trip：UInt/Str/UIntArray + 多层嵌套 Record
//   4. v2 未知类型按长度跳过（向前兼容）
//   5. 整体搬迁的 v2 字节地址失效 → 硬报错（不静默错值）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <neuralnet.cpp/model_keyvalue_record.hpp>

namespace
{

int g_failures = 0;

void expect(bool ok, const char *name)
{
    if (!ok)
    {
        ++g_failures;
        std::printf("  [FAIL] %s\n", name);
    }
    else
        std::printf("  [ ok ] %s\n", name);
}

using nn::KeyValueRecord;

// ── 用例 1：v1 round-trip（含空记录）────────────────────────────────
void test_v1_roundtrip()
{
    std::printf("v1 round-trip:\n");
    KeyValueRecord rec;
    rec.set("answer", std::uint64_t{42});
    rec.set("name", std::string("hello 世界"));
    rec.set("arr", std::vector<std::uint64_t>{1, 2, 3, 0xFFFFFFFFFFFFFFFFull});

    const std::string bytes = rec.serialize();
    auto parsed_r = KeyValueRecord::parse(bytes);
    expect(static_cast<bool>(parsed_r), "v1 parse 成功");
    if (!parsed_r) return;

    std::uint64_t u = 0;
    std::string s;
    std::vector<std::uint64_t> arr;
    expect(parsed_r->get("answer", u) && u == 42, "v1 UInt 逐位还原");
    expect(parsed_r->get("name", s) && s == "hello 世界", "v1 Str 逐位还原");
    expect(parsed_r->get("arr", arr) && arr.size() == 4
        && arr[3] == 0xFFFFFFFFFFFFFFFFull, "v1 UIntArray 逐位还原");
    expect(parsed_r->serialize() == bytes, "v1 serialize 字节稳定");

    // 空记录：field_count = 0，按 v1 处理
    KeyValueRecord empty;
    auto empty_r = KeyValueRecord::parse(empty.serialize());
    expect(static_cast<bool>(empty_r) && !empty_r->has("answer"), "v1 空记录");
}

// ── 用例 2：v1 未知类型按长度跳过 ─────────────────────────────────
void test_v1_unknown_type_skipped()
{
    std::printf("v1 未知类型跳过:\n");
    std::string bytes;
    nn::detail::append_u32(bytes, 2);            // field_count
    // field 0: "keep", UInt(0) = 7
    nn::detail::append_u32(bytes, 4);
    bytes.append("keep");
    bytes.push_back(0);
    nn::detail::append_u32(bytes, 8);
    nn::detail::append_u64(bytes, 7);
    // field 1: "future", type=9（未知），value_len=5
    nn::detail::append_u32(bytes, 6);
    bytes.append("future");
    bytes.push_back(9);
    nn::detail::append_u32(bytes, 5);
    bytes.append("12345");

    auto rec_r = KeyValueRecord::parse(bytes);
    expect(static_cast<bool>(rec_r), "v1 含未知类型仍解析成功");
    if (!rec_r) return;
    std::uint64_t u = 0;
    expect(rec_r->get("keep", u) && u == 7, "v1 已知字段照常读出");
    expect(!rec_r->has("future"), "v1 未知字段被跳过");
}

// ── 用例 3：v2 round-trip（多层嵌套）────────────────────────────────
void test_v2_roundtrip()
{
    std::printf("v2 round-trip:\n");
    KeyValueRecord leaf;
    leaf.set("leaf_u", std::uint64_t{9});
    leaf.set("leaf_s", std::string("嵌套字符串"));

    KeyValueRecord mid;
    mid.set("mid_arr", std::vector<std::uint64_t>{10, 20});
    mid.set("child", leaf);

    KeyValueRecord root;
    root.set("kind", std::uint64_t{1});
    root.set("blob", std::string(100, 'x'));
    root.set("sub", mid);

    // base = 0（自包含字节块）
    const std::string blob0 = root.serialize_v2(0);
    auto p0_r = KeyValueRecord::parse(blob0);
    expect(static_cast<bool>(p0_r), "v2 base=0 parse 成功（哨兵自举判 v2）");
    if (p0_r)
    {
        std::uint64_t u = 0;
        std::string s;
        KeyValueRecord sub;
        expect(p0_r->get("kind", u) && u == 1, "v2 UInt 还原");
        expect(p0_r->get("blob", s) && s == std::string(100, 'x'), "v2 Str 还原");
        expect(p0_r->get("sub", sub), "v2 Record 字段读出");
        KeyValueRecord child;
        std::vector<std::uint64_t> arr;
        expect(sub.get("mid_arr", arr) && arr.size() == 2 && arr[1] == 20,
               "v2 嵌套 UIntArray 还原");
        expect(sub.get("child", child), "v2 二层嵌套 Record 读出");
        std::string ls;
        expect(child.get("leaf_s", ls) && ls == "嵌套字符串", "v2 二层嵌套 Str 还原");
    }

    // base = 8（模拟 .nnvocab：标签 8 字节后才是记录）
    const std::string file = std::string("NNVCxxxx") + root.serialize_v2(8);
    auto p8_r = KeyValueRecord::parse_at(nn::kvrec_memory_source(file), 8);
    expect(static_cast<bool>(p8_r), "v2 base=8（带文件标签）parse 成功");
    if (p8_r)
    {
        KeyValueRecord sub;
        std::vector<std::uint64_t> arr;
        expect(p8_r->get("sub", sub) && sub.get("mid_arr", arr)
               && arr.size() == 2 && arr[1] == 20, "v2 base=8 嵌套可读");
        KeyValueRecord child;
        std::uint64_t u = 0;
        std::string ls;
        expect(sub.get("child", child) && child.get("leaf_u", u) && u == 9,
               "v2 base=8 二层嵌套还原");
    }
}

// ── 用例 4：v2 未知类型按长度跳过（地址连续性仍成立）──────────────
void test_v2_unknown_type_skipped()
{
    std::printf("v2 未知类型跳过:\n");
    // 手工拼一条 v2：哨兵 + 两字段（其一为未知类型 7），数据块连续排布
    std::string bytes;
    const std::uint32_t field_count = 3;
    // 字段表 = field_count(4) + 哨兵(30) + "keep" 头(4+4+1+8) + "future" 头(4+6+1+8)
    const std::uint64_t table = 4 + 30 + (4 + 4 + 1 + 8) + (4 + 6 + 1 + 8);
    nn::detail::append_u32(bytes, field_count);
    // 哨兵（v1 内联）
    nn::detail::append_u32(bytes, 13);
    bytes.append("kvrec_version");
    bytes.push_back(0);
    nn::detail::append_u32(bytes, 8);
    nn::detail::append_u64(bytes, 2);
    // field 1: "keep" UInt，addr = table
    nn::detail::append_u32(bytes, 4);
    bytes.append("keep");
    bytes.push_back(0);
    nn::detail::append_u64(bytes, table);
    // field 2: "future" type=7（未知），addr = table + 16
    nn::detail::append_u32(bytes, 6);
    bytes.append("future");
    bytes.push_back(7);
    nn::detail::append_u64(bytes, table + 16);
    // 数据块（连续）
    nn::detail::append_u64(bytes, 8);
    nn::detail::append_u64(bytes, 123);
    nn::detail::append_u64(bytes, 3);
    bytes.append("abc");

    auto rec_r = KeyValueRecord::parse(bytes);
    expect(static_cast<bool>(rec_r), "v2 含未知类型仍解析成功");
    if (!rec_r) return;
    std::uint64_t u = 0;
    expect(rec_r->get("keep", u) && u == 123, "v2 已知字段照常读出");
    expect(!rec_r->has("future"), "v2 未知字段被跳过");
}

// ── 用例 5：整体搬迁的 v2 字节地址失效 → 硬报错 ────────────────────
void test_v2_relocation_rejected()
{
    std::printf("v2 搬迁地址校验:\n");
    KeyValueRecord rec;
    rec.set("x", std::uint64_t{5});
    const std::string at8 = rec.serialize_v2(8);

    // 搬迁：去掉 8 字节基址后当作独立记录解析 → 地址不连续/越界，必须报错
    auto moved_r = KeyValueRecord::parse(at8);
    expect(!moved_r, "整体搬迁的 v2 字节被拒绝");
    if (!moved_r)
        std::printf("         （错误信息: %s）\n", moved_r.error().message.c_str());

    // 原地（base=8 且按文件解析）依然成功——排除"解析器本身坏了"的误报
    const std::string file = std::string("NNVCxxxx") + at8;
    auto ok_r = KeyValueRecord::parse_at(nn::kvrec_memory_source(file), 8);
    expect(static_cast<bool>(ok_r), "正确基址下同一字节解析成功");
}

}  // namespace

int main()
{
    std::printf("kvrec_test:\n");
    test_v1_roundtrip();
    test_v1_unknown_type_skipped();
    test_v2_roundtrip();
    test_v2_unknown_type_skipped();
    test_v2_relocation_rejected();
    std::printf("\nkvrec_test: %d failure(s)\n", g_failures);
    return g_failures != 0 ? 1 : 0;
}
