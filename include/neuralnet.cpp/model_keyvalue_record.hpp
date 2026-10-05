#pragma once

// ══════════════════════════════════════════════════════════════════════════
//  model_keyvalue_record.hpp — 自描述键值记录（替代 JSON 的轻量二进制格式）
//
//  设计目标：
//    1. 容易解析 —— 长度前缀 + 显式类型 + 值长度前缀，无状态机/偏移量假设。
//    2. 自描述   —— 每条记录自带 key + type + value，未知字段可按长度跳过。
//    3. 面向 C++ —— set/get 直接对应 uint64_t / std::string / vector<uint64_t>。
//    4. 版本友好 —— 缺失字段由上层按版本记录默认值；新增字段只需加一条 set/get。
//
//  两个版本（设计见 docs/development/19-unified-dataset.md §3）：
//
//  ── v1（内联布局，小端）────────────────────────────────────────────────
//    [field_count u32]
//    field := [key_len u32][key bytes][type u8][value_len u32][value bytes]
//      type 0 (UInt)     : value = 8 字节 uint64
//      type 1 (Str)      : value = 原始字符串字节
//      type 2 (UIntArray): value = count×8 字节的 uint64 数组
//    value_len 前缀保证：即使出现未知类型，解析器也能按长度安全跳过（向前兼容）。
//    模型规格头（model_serialization.hpp 的 spec header）继续用 v1，字节零变化。
//
//  ── v2（地址布局，小端；"v2 仅记录位移"）───────────────────────────────
//    [field_count u32]                      // 含哨兵在内
//    field[0] := 哨兵，按 v1 布局内联编码（v2 中唯一的内联字段）:
//                [key_len=13]["kvrec_version"][type=0][value_len=8][u64 version=2]
//    field[i>0] := [key_len u32][key bytes][type u8][addr u64]   ← 只记 key、类型、地址
//    addr       := 值数据块在**文件内的绝对偏移**
//    数据块     := [value_len u64][value bytes]                   ← 每块自带长度
//      type 0/1/2 同 v1
//      type 3 (Record) := 完整嵌套 v2 kvrec 字节（含自身哨兵；其内部 addr
//                         同样是文件绝对偏移）
//
//  版本自举（v1/v2 判别）：把 field[0] 按 v1 布局读出——
//    key == "kvrec_version" ⇒ 本记录为 v2，余下字段按 v2 布局解析；
//    否则 ⇒ 本记录为 v1（field[0] 已被正确按 v1 消费，继续 v1 解析）。
//    即：无版本字段 = v1，有 = 按该版本读。"kvrec_version" 为保留键名；
//    空记录（field_count=0）按 v1 处理。
//
//  ⚠ 哨兵是 v2 中唯一按内联编码的字段——判别版本必须先有一个两边都读得懂
//    的字段，除此之外所有值一律走地址。
//
//  ⚠ v2 数据块地址为文件绝对偏移，且写入方约定"按字段顺序、紧随字段表连续
//    排布"（serialize_v2 即此布局）。解析器逐字段校验 addr == 期望游标——
//    整体搬迁（换基址复制字节）会在此硬报错，而不是静默读到错值。
// ══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "core_errors.hpp"

namespace nn
{

namespace detail
{

// ── 小端编码/解码工具 ──────────────────────────────────────────────

inline void append_u32(std::string &out, uint32_t v)
{
    out.push_back(static_cast<char>(v & 0xFF));
    out.push_back(static_cast<char>((v >> 8) & 0xFF));
    out.push_back(static_cast<char>((v >> 16) & 0xFF));
    out.push_back(static_cast<char>((v >> 24) & 0xFF));
}

inline void append_u64(std::string &out, uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

inline bool take_u32(std::string_view &s, uint32_t &out)
{
    if (s.size() < 4) return false;
    out = static_cast<uint32_t>(static_cast<unsigned char>(s[0]))
        | (static_cast<uint32_t>(static_cast<unsigned char>(s[1])) << 8)
        | (static_cast<uint32_t>(static_cast<unsigned char>(s[2])) << 16)
        | (static_cast<uint32_t>(static_cast<unsigned char>(s[3])) << 24);
    s.remove_prefix(4);
    return true;
}

inline bool take_u64(std::string_view &s, uint64_t &out)
{
    if (s.size() < 8) return false;
    out = 0;
    for (int i = 0; i < 8; ++i)
        out |= static_cast<uint64_t>(static_cast<unsigned char>(s[i])) << (8 * i);
    s.remove_prefix(8);
    return true;
}

// ── v2 字节源与游标（内存块 / 文件皆可包装为 ReadAt）────────────────
// ReadAt 语义：addr 为文件绝对偏移，取 [addr, addr+len) 字节。
using KvReadAt = std::function<Result<std::string>(std::uint64_t addr, std::uint64_t len)>;

struct KvCursor
{
    const KvReadAt *read = nullptr;
    std::uint64_t pos = 0;

    [[nodiscard]] Result<std::string> take(std::uint64_t n)
    {
        NN_TRY(buf, (*read)(pos, n));
        pos += n;
        return buf;
    }

    [[nodiscard]] Result<std::uint32_t> take_u32()
    {
        NN_TRY(b, take(4));
        std::string_view sv = *b;
        std::uint32_t v = 0;
        if (!::nn::detail::take_u32(sv, v))
            NN_FAIL("KeyValueRecord::parse: u32 解码失败");
        return v;
    }

    [[nodiscard]] Result<std::uint64_t> take_u64()
    {
        NN_TRY(b, take(8));
        std::string_view sv = *b;
        std::uint64_t v = 0;
        if (!::nn::detail::take_u64(sv, v))
            NN_FAIL("KeyValueRecord::parse: u64 解码失败");
        return v;
    }

    [[nodiscard]] Result<std::uint8_t> take_u8()
    {
        NN_TRY(b, take(1));
        return static_cast<std::uint8_t>(static_cast<unsigned char>((*b)[0]));
    }
};

}  // namespace detail

// ── ReadAt 的内存实现：bytes 视为完整文件映像，addr 为绝对偏移 ─────────
// ⚠ 捕获 string_view，调用方须保证 bytes 生命周期覆盖解析期。
inline detail::KvReadAt kvrec_memory_source(std::string_view bytes)
{
    return [bytes](std::uint64_t addr, std::uint64_t len) -> Result<std::string>
    {
        if (addr > bytes.size() || len > bytes.size() - addr)
            NN_FAIL("KeyValueRecord::parse: 读取越界（addr=" + std::to_string(addr)
                + ", len=" + std::to_string(len)
                + ", size=" + std::to_string(bytes.size()) + "）");
        return std::string(bytes.substr(static_cast<std::size_t>(addr),
                                        static_cast<std::size_t>(len)));
    };
}

class KeyValueRecord
{
public:
    enum class Type : uint8_t
    {
        UInt      = 0,  // uint64_t
        Str       = 1,  // std::string
        UIntArray = 2,  // std::vector<uint64_t>
        Record    = 3,  // 嵌套 KeyValueRecord（仅 v2；value = 完整嵌套 v2 kvrec 字节）
    };

    // v2 版本哨兵（field[0]，v1 内联布局编码）的保留键名与版本值
    static constexpr std::string_view KVREC_VERSION_KEY = "kvrec_version";
    static constexpr std::uint64_t KVREC_VERSION_V2 = 2;

    // v2 字节源：按绝对偏移取字节（见 detail::KvReadAt）
    using ReadAt = detail::KvReadAt;

    // ── 写入 ────────────────────────────────────────────────────
    // ⚠ "kvrec_version" 为 v2 保留键名，业务字段不得使用。
    KeyValueRecord &set(const std::string &key, uint64_t v);
    KeyValueRecord &set(const std::string &key, const std::string &v);
    KeyValueRecord &set(const std::string &key, const std::vector<uint64_t> &v);
    KeyValueRecord &set(const std::string &key, const KeyValueRecord &v);

    // v1 序列化（模型 spec 头继续用；Record 字段 v1 不支持，跳过不写）
    [[nodiscard]] std::string serialize() const;

    // v2 序列化：产出自包含字节块（字段表 + 连续数据块），
    // addr = base_offset + 块内相对位移。嵌套 Record 的子块以
    // 自身数据块起点 + 8 为 base 递归序列化（addr 恒为文件绝对偏移）。
    [[nodiscard]] std::string serialize_v2(std::uint64_t base_offset = 0) const;

    // ── 解析（哨兵自举：无版本字段 = v1，有 = 按版本读）──────────
    // bytes 视为完整文件映像，v2 地址相对 bytes 起点。
    [[nodiscard]] static Result<KeyValueRecord> parse(std::string_view bytes);
    // 同上，但字节经 ReadAt 按绝对偏移取（大文件/流式场景）。
    [[nodiscard]] static Result<KeyValueRecord> parse_at(const ReadAt &read,
                                                         std::uint64_t offset);

    // ── 读取（返回 false 表示缺失或类型不符） ──────────────────
    [[nodiscard]] bool has(const std::string &key) const;
    [[nodiscard]] bool get(const std::string &key, uint64_t &out) const;
    [[nodiscard]] bool get(const std::string &key, std::string &out) const;
    [[nodiscard]] bool get(const std::string &key, std::vector<uint64_t> &out) const;
    [[nodiscard]] bool get(const std::string &key, KeyValueRecord &out) const;

private:
    struct Field
    {
        std::string key;
        Type type = Type::UInt;
        uint64_t u = 0;
        std::string s;
        std::vector<uint64_t> arr;
        std::vector<Field> rec;   // Type::Record 的子字段（递归嵌套，v2）
    };
    std::vector<Field> fields_;

    // v2 内部构件
    static constexpr std::uint64_t SENTINEL_INLINE_SIZE = 4 + 13 + 1 + 4 + 8;
    [[nodiscard]] static std::uint64_t value_size_(const Field &f);
    [[nodiscard]] static std::uint64_t fields_blob_size_(const std::vector<Field> &fs);
    static void serialize_fields_v2_(const std::vector<Field> &fs,
                                     std::uint64_t base_offset, std::string &out);
    [[nodiscard]] static Result<std::vector<KeyValueRecord::Field>>
        parse_fields_v2_(const ReadAt &read, std::uint64_t offset,
                         std::uint32_t total_count);
};

// ══════════════════════════════════════════════════════════════════════════
// 实现
// ══════════════════════════════════════════════════════════════════════════

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, uint64_t v)
{
    fields_.push_back(Field{key, Type::UInt, v, {}, {}, {}});
    return *this;
}

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, const std::string &v)
{
    fields_.push_back(Field{key, Type::Str, 0, v, {}, {}});
    return *this;
}

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, const std::vector<uint64_t> &v)
{
    fields_.push_back(Field{key, Type::UIntArray, 0, {}, v, {}});
    return *this;
}

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, const KeyValueRecord &v)
{
    fields_.push_back(Field{key, Type::Record, 0, {}, {}, v.fields_});
    return *this;
}

inline std::string KeyValueRecord::serialize() const
{
    std::string out;
    // v1 无嵌套概念：Record 字段跳过不写（count 随之修正）
    std::uint32_t count = 0;
    for (const auto &f : fields_)
        if (f.type != Type::Record) ++count;
    detail::append_u32(out, count);
    for (const auto &f : fields_)
    {
        if (f.type == Type::Record) continue;
        detail::append_u32(out, static_cast<uint32_t>(f.key.size()));
        out.append(f.key);
        out.push_back(static_cast<char>(f.type));
        switch (f.type)
        {
        case Type::UInt:
            detail::append_u32(out, 8);
            detail::append_u64(out, f.u);
            break;
        case Type::Str:
            detail::append_u32(out, static_cast<uint32_t>(f.s.size()));
            out.append(f.s);
            break;
        case Type::UIntArray:
            detail::append_u32(out, static_cast<uint32_t>(f.arr.size() * sizeof(uint64_t)));
            for (auto v : f.arr)
                detail::append_u64(out, v);
            break;
        case Type::Record:
            break;  // v1 不支持（已跳过）
        }
    }
    return out;
}

inline std::uint64_t KeyValueRecord::value_size_(const Field &f)
{
    switch (f.type)
    {
    case Type::UInt:      return 8;
    case Type::Str:       return static_cast<std::uint64_t>(f.s.size());
    case Type::UIntArray: return static_cast<std::uint64_t>(f.arr.size() * sizeof(uint64_t));
    case Type::Record:    return fields_blob_size_(f.rec);
    }
    return 0;
}

inline std::uint64_t KeyValueRecord::fields_blob_size_(const std::vector<Field> &fs)
{
    std::uint64_t sz = 4 + SENTINEL_INLINE_SIZE;
    for (const auto &f : fs)
        sz += 4 + static_cast<std::uint64_t>(f.key.size()) + 1 + 8;
    for (const auto &f : fs)
        sz += 8 + value_size_(f);
    return sz;
}

inline void KeyValueRecord::serialize_fields_v2_(const std::vector<Field> &fs,
                                                 std::uint64_t base_offset,
                                                 std::string &out)
{
    // 字段表大小与取值无关，可先算出；数据块按字段顺序紧随其后连续排布
    std::uint64_t table = 4 + SENTINEL_INLINE_SIZE;
    for (const auto &f : fs)
        table += 4 + static_cast<std::uint64_t>(f.key.size()) + 1 + 8;

    // ── 字段表（哨兵 + 字段头，头内 addr 直接算出）────────────────
    detail::append_u32(out, static_cast<uint32_t>(fs.size() + 1));  // 含哨兵
    detail::append_u32(out, 13);
    out.append(KVREC_VERSION_KEY);
    out.push_back(static_cast<char>(Type::UInt));
    detail::append_u32(out, 8);
    detail::append_u64(out, KVREC_VERSION_V2);

    std::uint64_t cursor = base_offset + table;
    std::vector<std::uint64_t> addrs;
    addrs.reserve(fs.size());
    for (const auto &f : fs)
    {
        detail::append_u32(out, static_cast<uint32_t>(f.key.size()));
        out.append(f.key);
        out.push_back(static_cast<char>(f.type));
        addrs.push_back(cursor);
        detail::append_u64(out, cursor);
        cursor += 8 + value_size_(f);
    }

    // ── 数据块（[value_len u64][value bytes] × 字段序）────────────
    cursor = base_offset + table;
    for (std::size_t i = 0; i < fs.size(); ++i)
    {
        const Field &f = fs[i];
        switch (f.type)
        {
        case Type::UInt:
            detail::append_u64(out, 8);
            detail::append_u64(out, f.u);
            break;
        case Type::Str:
            detail::append_u64(out, static_cast<std::uint64_t>(f.s.size()));
            out.append(f.s);
            break;
        case Type::UIntArray:
            detail::append_u64(out,
                static_cast<std::uint64_t>(f.arr.size() * sizeof(uint64_t)));
            for (auto v : f.arr)
                detail::append_u64(out, v);
            break;
        case Type::Record:
        {
            const std::uint64_t child_size = fields_blob_size_(f.rec);
            detail::append_u64(out, child_size);
            // 子记录字节紧随其后 = 本块 value；其内部 addr 以 value 起点为 base
            serialize_fields_v2_(f.rec, cursor + 8, out);
            break;
        }
        }
        cursor += 8 + value_size_(f);
    }
}

inline std::string KeyValueRecord::serialize_v2(std::uint64_t base_offset) const
{
    std::string out;
    out.reserve(static_cast<std::size_t>(fields_blob_size_(fields_)));
    serialize_fields_v2_(fields_, base_offset, out);
    return out;
}

// ── 解析 ─────────────────────────────────────────────────────────────

namespace detail
{
// v1 布局单字段解码：value 已取出，按类型填 Field；未知类型返回 false（跳过）
inline bool kv_decode_value_v1(KeyValueRecord::Type type, std::string_view value,
                               std::uint64_t &u, std::string &s,
                               std::vector<std::uint64_t> &arr)
{
    switch (type)
    {
    case KeyValueRecord::Type::UInt:
    {
        uint64_t v = 0;
        if (!take_u64(value, v)) return false;
        u = v;
        return true;
    }
    case KeyValueRecord::Type::Str:
        s.assign(value);
        return true;
    case KeyValueRecord::Type::UIntArray:
        if (value.size() % sizeof(uint64_t) != 0) return false;
        {
            const std::size_t count = value.size() / sizeof(uint64_t);
            arr.reserve(count);
            for (std::size_t k = 0; k < count; ++k)
            {
                uint64_t v = 0;
                take_u64(value, v);
                arr.push_back(v);
            }
        }
        return true;
    default:
        return false;
    }
}
}  // namespace detail

inline Result<std::vector<KeyValueRecord::Field>>
KeyValueRecord::parse_fields_v2_(const ReadAt &read, std::uint64_t offset,
                                 std::uint32_t total_count)
{
    // 哨兵已按 v1 消费；余下字段按 v2 布局读取（字段头 → 数据块两阶段）
    struct Head { std::string key; Type type; std::uint64_t addr; };
    detail::KvCursor c{&read, offset};
    std::vector<Head> heads;
    heads.reserve(total_count > 0 ? total_count - 1 : 0);
    for (std::uint32_t i = 1; i < total_count; ++i)
    {
        NN_TRY(key_len, c.take_u32());
        NN_TRY(key, c.take(*key_len));
        NN_TRY(t, c.take_u8());
        NN_TRY(addr, c.take_u64());
        heads.push_back(Head{std::move(*key), static_cast<Type>(*t), *addr});
    }
    // 字段表到此结束；数据块按字段顺序紧随其后连续排布
    const std::uint64_t blocks_begin = c.pos;

    std::vector<Field> fields;
    std::uint64_t expect = blocks_begin;
    for (const auto &h : heads)
    {
        if (h.addr != expect)
            NN_FAIL("KeyValueRecord::parse: 字段 '" + h.key
                + "' 数据块地址不连续（整体搬迁的 kvrec 字节地址已失效）");
        detail::KvCursor bc{&read, h.addr};
        NN_TRY(value_len, bc.take_u64());
        expect = h.addr + 8 + *value_len;

        // 未知类型：按长度跳过（向前兼容），不报错
        if (h.type != Type::UInt && h.type != Type::Str
            && h.type != Type::UIntArray && h.type != Type::Record)
            continue;

        Field f;
        f.key = h.key;
        f.type = h.type;
        if (h.type == Type::Record)
        {
            NN_TRY(sub, parse_at(read, h.addr + 8));
            f.rec = std::move(sub->fields_);
        }
        else
        {
            NN_TRY(value, bc.take(*value_len));
            std::string_view sv = *value;
            if (!detail::kv_decode_value_v1(h.type, sv, f.u, f.s, f.arr))
                NN_FAIL("KeyValueRecord::parse: 字段 '" + h.key + "' 值长度错误");
        }
        fields.push_back(std::move(f));
    }
    return fields;
}

inline Result<KeyValueRecord> KeyValueRecord::parse_at(const ReadAt &read,
                                                       std::uint64_t offset)
{
    detail::KvCursor c{&read, offset};
    KeyValueRecord rec;
    NN_TRY(count, c.take_u32());
    if (*count == 0) return rec;  // 空记录按 v1 处理

    // ── field[0] 按 v1 布局读出（版本自举的判别点）────────────────
    NN_TRY(key_len0, c.take_u32());
    NN_TRY(key0, c.take(*key_len0));
    NN_TRY(type0, c.take_u8());
    NN_TRY(value_len0, c.take_u32());
    NN_TRY(value0, c.take(*value_len0));

    if (std::string_view(*key0) == KVREC_VERSION_KEY)
    {
        // ── v2：哨兵必须是 UInt(2)；余下字段按 v2 布局解析 ─────────
        if (static_cast<Type>(*type0) != Type::UInt || *value_len0 != 8)
            NN_FAIL("KeyValueRecord::parse: kvrec 版本哨兵字段格式错误");
        std::string_view sv = *value0;
        std::uint64_t version = 0;
        if (!detail::take_u64(sv, version))
            NN_FAIL("KeyValueRecord::parse: kvrec 版本哨兵值长度错误");
        if (version != KVREC_VERSION_V2)
            NN_FAIL("KeyValueRecord::parse: 不支持的 kvrec 版本 "
                + std::to_string(version) + "（当前支持 v2）");
        NN_TRY(fields, parse_fields_v2_(read, c.pos, *count));
        rec.fields_ = std::move(*fields);
        return rec;
    }

    // ── v1：field[0] 已被正确按 v1 消费，继续 v1 解析 ────────────
    {
        Field f;
        f.key = std::move(*key0);
        f.type = static_cast<Type>(*type0);
        std::string_view sv = *value0;
        const bool unknown_v1_type = (f.type != Type::UInt && f.type != Type::Str
                                      && f.type != Type::UIntArray);
        if (unknown_v1_type)
        {
            // 未知类型：按长度安全跳过（向前兼容），不报错
        }
        else if (!detail::kv_decode_value_v1(f.type, sv, f.u, f.s, f.arr))
            NN_FAIL("KeyValueRecord::parse: 字段 '" + f.key + "' 值长度错误");
        else
            rec.fields_.push_back(std::move(f));
    }
    for (std::uint32_t i = 1; i < *count; ++i)
    {
        NN_TRY(key_len, c.take_u32());
        NN_TRY(key, c.take(*key_len));
        NN_TRY(t, c.take_u8());
        NN_TRY(value_len, c.take_u32());
        NN_TRY(value, c.take(*value_len));

        const Type type = static_cast<Type>(*t);
        // 未知类型：按长度跳过（向前兼容），不报错
        if (type != Type::UInt && type != Type::Str && type != Type::UIntArray)
            continue;

        Field f;
        f.key = std::move(*key);
        f.type = type;
        std::string_view sv = *value;
        if (!detail::kv_decode_value_v1(type, sv, f.u, f.s, f.arr))
            NN_FAIL("KeyValueRecord::parse: 字段 '" + f.key + "' 值长度错误");
        rec.fields_.push_back(std::move(f));
    }
    return rec;
}

inline Result<KeyValueRecord> KeyValueRecord::parse(std::string_view bytes)
{
    return parse_at(kvrec_memory_source(bytes), 0);
}

inline bool KeyValueRecord::has(const std::string &key) const
{
    for (const auto &f : fields_)
        if (f.key == key) return true;
    return false;
}

inline bool KeyValueRecord::get(const std::string &key, uint64_t &out) const
{
    for (const auto &f : fields_)
        if (f.key == key && f.type == Type::UInt) { out = f.u; return true; }
    return false;
}

inline bool KeyValueRecord::get(const std::string &key, std::string &out) const
{
    for (const auto &f : fields_)
        if (f.key == key && f.type == Type::Str) { out = f.s; return true; }
    return false;
}

inline bool KeyValueRecord::get(const std::string &key, std::vector<uint64_t> &out) const
{
    for (const auto &f : fields_)
        if (f.key == key && f.type == Type::UIntArray) { out = f.arr; return true; }
    return false;
}

inline bool KeyValueRecord::get(const std::string &key, KeyValueRecord &out) const
{
    for (const auto &f : fields_)
        if (f.key == key && f.type == Type::Record)
        {
            KeyValueRecord sub;
            sub.fields_ = f.rec;
            out = std::move(sub);
            return true;
        }
    return false;
}

}  // namespace nn
