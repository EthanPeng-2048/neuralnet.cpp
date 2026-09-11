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
//  字节布局（小端）：
//    [field_count u32]
//    field := [key_len u32][key bytes][type u8][value_len u32][value bytes]
//      type 0 (UInt)     : value = 8 字节 uint64
//      type 1 (Str)      : value = 原始字符串字节
//      type 2 (UIntArray): value = count×8 字节的 uint64 数组
//    value_len 前缀保证：即使出现未知类型，解析器也能按长度安全跳过（向前兼容）。
// ══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core_errors.hpp"

namespace nn
{

class KeyValueRecord
{
public:
    enum class Type : uint8_t
    {
        UInt      = 0,  // uint64_t
        Str       = 1,  // std::string
        UIntArray = 2,  // std::vector<uint64_t>
    };

    // ── 写入 ────────────────────────────────────────────────────────────
    KeyValueRecord &set(const std::string &key, uint64_t v);
    KeyValueRecord &set(const std::string &key, const std::string &v);
    KeyValueRecord &set(const std::string &key, const std::vector<uint64_t> &v);

    // 序列化为字节串（不含总长度前缀；总长度由文件格式负责）
    [[nodiscard]] std::string serialize() const;

    // ── 解析：整块字节串 → KeyValueRecord ───────────────────────────────
    [[nodiscard]] static Result<KeyValueRecord> parse(std::string_view bytes);

    // ── 读取（返回 false 表示缺失或类型不符） ──────────────────────────
    [[nodiscard]] bool has(const std::string &key) const;
    [[nodiscard]] bool get(const std::string &key, uint64_t &out) const;
    [[nodiscard]] bool get(const std::string &key, std::string &out) const;
    [[nodiscard]] bool get(const std::string &key, std::vector<uint64_t> &out) const;

private:
    struct Field
    {
        std::string key;
        Type type = Type::UInt;
        uint64_t u = 0;
        std::string s;
        std::vector<uint64_t> arr;
    };
    std::vector<Field> fields_;
};

// ══════════════════════════════════════════════════════════════════════════
// 实现
// ══════════════════════════════════════════════════════════════════════════

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, uint64_t v)
{
    fields_.push_back(Field{key, Type::UInt, v, {}, {}});
    return *this;
}

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, const std::string &v)
{
    fields_.push_back(Field{key, Type::Str, 0, v, {}});
    return *this;
}

inline KeyValueRecord &KeyValueRecord::set(const std::string &key, const std::vector<uint64_t> &v)
{
    fields_.push_back(Field{key, Type::UIntArray, 0, {}, v});
    return *this;
}

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
}  // namespace detail

inline std::string KeyValueRecord::serialize() const
{
    std::string out;
    detail::append_u32(out, static_cast<uint32_t>(fields_.size()));
    for (const auto &f : fields_)
    {
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
        }
    }
    return out;
}

inline Result<KeyValueRecord> KeyValueRecord::parse(std::string_view bytes)
{
    KeyValueRecord rec;
    uint32_t count = 0;
    if (!detail::take_u32(bytes, count))
        return std::unexpected(Error{"KeyValueRecord::parse: 缺少字段计数"});

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t key_len = 0;
        if (!detail::take_u32(bytes, key_len))
            return std::unexpected(Error{"KeyValueRecord::parse: 字段 key 长度越界"});
        if (bytes.size() < key_len)
            return std::unexpected(Error{"KeyValueRecord::parse: key 数据不足"});
        std::string key(bytes.substr(0, key_len));
        bytes.remove_prefix(key_len);

        if (bytes.empty())
            return std::unexpected(Error{"KeyValueRecord::parse: 缺少字段类型"});
        const auto type = static_cast<Type>(static_cast<unsigned char>(bytes[0]));
        bytes.remove_prefix(1);

        uint32_t value_len = 0;
        if (!detail::take_u32(bytes, value_len))
            return std::unexpected(Error{"KeyValueRecord::parse: 缺少值长度"});
        if (bytes.size() < value_len)
            return std::unexpected(Error{"KeyValueRecord::parse: 值数据不足"});
        auto value = bytes.substr(0, value_len);
        bytes.remove_prefix(value_len);

        // 未知类型：按长度跳过（向前兼容），不报错
        if (type != Type::UInt && type != Type::Str && type != Type::UIntArray)
            continue;

        Field f;
        f.key = std::move(key);
        f.type = type;
        switch (type)
        {
        case Type::UInt:
        {
            uint64_t v = 0;
            if (!detail::take_u64(value, v))
                return std::unexpected(Error{"KeyValueRecord::parse: UInt 值长度错误"});
            f.u = v;
            break;
        }
        case Type::Str:
            f.s.assign(value);
            break;
        case Type::UIntArray:
            if (value.size() % sizeof(uint64_t) != 0)
                return std::unexpected(Error{"KeyValueRecord::parse: UIntArray 值长度错误"});
            {
                // 先固定元素个数，避免 take_u64 缩短 value 影响循环边界
                const std::size_t count = value.size() / sizeof(uint64_t);
                f.arr.reserve(count);
                for (std::size_t k = 0; k < count; ++k)
                {
                    uint64_t v = 0;
                    detail::take_u64(value, v);
                    f.arr.push_back(v);
                }
            }
            break;
        }
        rec.fields_.push_back(std::move(f));
    }
    return rec;
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

}  // namespace nn

