#pragma once

// ── tensor_schema.hpp — 有序张量字段 schema（docs/todo/31 §3）──────────────
//
// 公共语义：
//   · 字段按稳定顺序保存；名称只允许在 setup/deserialize 期解析；
//   · 热路径使用 FieldHandle（schema 指纹 + 下标），不做字符串查找；
//   · schema 指纹是字段名称/顺序/约束的确定性摘要，供序列化与跨对象校验。
//
// 本头只依赖标准库与精度词汇表，不依赖 Tensor/ComputeEngine，故 ModelSpec 等
// 纯数据层可安全复用。具名张量值在 named_tensor_set.hpp。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/precision.hpp>

namespace nn
{

inline constexpr std::size_t kInvalidFieldIndex =
    static_cast<std::size_t>(-1);
inline constexpr std::uint32_t kMaxTensorSchemaFields = 65536;

// 字段所在设备约束。使用独立枚举而不是 Device，保持本头不依赖 compute_tensor。
enum class TensorPlacement : std::uint8_t
{
    Any = 0,
    CPU = 1,
    GPU = 2,
};

[[nodiscard]] inline const char* tensor_placement_name(
    TensorPlacement p) noexcept
{
    switch (p)
    {
    case TensorPlacement::CPU: return "cpu";
    case TensorPlacement::GPU: return "gpu";
    default: return "any";
    }
}

// 二维张量的可选形状约束。nullopt = 该轴不限；值 0 表示明确要求空轴。
struct TensorShapeSpec
{
    std::optional<std::size_t> rows;
    std::optional<std::size_t> cols;

    [[nodiscard]] static constexpr TensorShapeSpec any() noexcept { return {}; }
    [[nodiscard]] static constexpr TensorShapeSpec exact(
        std::size_t rows, std::size_t cols) noexcept
    {
        return TensorShapeSpec{rows, cols};
    }

    [[nodiscard]] bool accepts(std::size_t r, std::size_t c) const noexcept
    {
        return (!rows || *rows == r) && (!cols || *cols == c);
    }

    [[nodiscard]] std::string to_string() const
    {
        auto axis = [](const std::optional<std::size_t>& v) {
            return v ? std::to_string(*v) : std::string("*");
        };
        return "(" + axis(rows) + "," + axis(cols) + ")";
    }
};

struct TensorFieldSpec
{
    std::string name;
    TensorShapeSpec shape;
    bool required = true;
    std::optional<Precision> precision;
    TensorPlacement placement = TensorPlacement::Any;
};

[[nodiscard]] inline TensorFieldSpec tensor_field(
    std::string name,
    TensorShapeSpec shape = {},
    bool required = true,
    std::optional<Precision> precision = std::nullopt,
    TensorPlacement placement = TensorPlacement::Any)
{
    TensorFieldSpec field;
    field.name = std::move(name);
    field.shape = shape;
    field.required = required;
    field.precision = precision;
    field.placement = placement;
    return field;
}

// setup 期解析结果。指纹让不同 schema 中恰好同下标的 handle 不会被误用。
struct FieldHandle
{
    std::uint64_t schema_fingerprint = 0;
    std::size_t index = kInvalidFieldIndex;

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return schema_fingerprint != 0 && index != kInvalidFieldIndex;
    }
};

// FNV-1a over schema name + each field's semantic metadata.
inline constexpr std::uint64_t kTensorSchemaFnvOffset = 1469598103934665603ULL;
inline constexpr std::uint64_t kTensorSchemaFnvPrime = 1099511628211ULL;

namespace detail
{
[[nodiscard]] inline std::uint64_t schema_hash_bytes(
    std::uint64_t h, std::string_view bytes) noexcept
{
    for (unsigned char c : bytes)
    {
        h ^= c;
        h *= kTensorSchemaFnvPrime;
    }
    return h;
}

[[nodiscard]] inline std::uint64_t schema_hash_u64(
    std::uint64_t h, std::uint64_t v) noexcept
{
    for (unsigned i = 0; i < 8; ++i)
    {
        h ^= static_cast<unsigned char>((v >> (i * 8)) & 0xFFu);
        h *= kTensorSchemaFnvPrime;
    }
    return h;
}

[[nodiscard]] inline std::uint64_t schema_hash_string(std::uint64_t h,
                                                      std::string_view value) noexcept
{
    h = schema_hash_u64(h, value.size());
    return schema_hash_bytes(h, value);
}

[[nodiscard]] inline std::uint64_t schema_hash_optional_size(
    std::uint64_t h, const std::optional<std::size_t>& v) noexcept
{
    h = schema_hash_u64(h, v ? 1u : 0u);
    return v ? schema_hash_u64(h, static_cast<std::uint64_t>(*v)) : h;
}
} // namespace detail

class TensorSchema
{
private:
    std::string name_;
    std::vector<TensorFieldSpec> fields_;
    std::uint64_t fingerprint_ = 0;

    TensorSchema(std::string name, std::vector<TensorFieldSpec> fields,
                 std::uint64_t fingerprint)
        : name_(std::move(name)),
          fields_(std::move(fields)),
          fingerprint_(fingerprint)
    {
    }

    [[nodiscard]] static std::uint64_t make_fingerprint_(
        std::string_view name, const std::vector<TensorFieldSpec>& fields) noexcept
    {
        std::uint64_t h = kTensorSchemaFnvOffset;
        h = detail::schema_hash_string(h, name);
        h = detail::schema_hash_u64(h, fields.size());
        for (const TensorFieldSpec& f : fields)
        {
            h = detail::schema_hash_string(h, f.name);
            h = detail::schema_hash_optional_size(h, f.shape.rows);
            h = detail::schema_hash_optional_size(h, f.shape.cols);
            h = detail::schema_hash_u64(h, f.required ? 1u : 0u);
            h = detail::schema_hash_u64(
                h, f.precision
                       ? static_cast<std::uint64_t>(*f.precision) + 1u
                       : 0u);
            h = detail::schema_hash_u64(
                h, static_cast<std::uint64_t>(f.placement));
        }
        // 0 保留给"未初始化 handle"。
        return h == 0 ? 1u : h;
    }

    [[nodiscard]] Result<void> validate_handle_(
        const FieldHandle& handle, std::string_view operation) const
    {
        if (!handle.valid())
            NN_FAIL(std::string(operation) + ": 字段 handle 未初始化");
        if (handle.schema_fingerprint != fingerprint_)
            NN_FAIL(std::string(operation) + ": 字段 handle 属于另一个 schema（"
                    "handle=" + std::to_string(handle.schema_fingerprint) +
                    "，schema=" + std::to_string(fingerprint_) + "；schema 名=" +
                    name_ + "）");
        if (handle.index >= fields_.size())
            NN_FAIL(std::string(operation) + ": 字段 handle 下标越界（" +
                    std::to_string(handle.index) + " >= " +
                    std::to_string(fields_.size()) + "；schema 名=" + name_ + "）");
        return {};
    }

public:
    TensorSchema() = default;
    TensorSchema(const TensorSchema&) = default;
    TensorSchema& operator=(const TensorSchema&) = default;
    TensorSchema(TensorSchema&&) noexcept = default;
    TensorSchema& operator=(TensorSchema&&) noexcept = default;
    ~TensorSchema() = default;

    [[nodiscard]] static Result<TensorSchema> create(
        std::string name, std::vector<TensorFieldSpec> fields)
    {
        if (name.empty())
            NN_FAIL("TensorSchema::create: schema 名不能为空");
        if (fields.empty())
            NN_FAIL("TensorSchema::create: schema '" + name +
                    "' 至少需要一个字段");

        std::unordered_set<std::string_view> field_names;
        field_names.reserve(fields.size());
        for (std::size_t i = 0; i < fields.size(); ++i)
        {
            if (fields[i].name.empty())
                NN_FAIL("TensorSchema::create: schema '" + name + "' 的第 " +
                        std::to_string(i) + " 个字段名为空");
            if (!field_names.insert(fields[i].name).second)
                NN_FAIL("TensorSchema::create: schema '" + name + "' 含重复字段 '" +
                        fields[i].name + "'");
        }

        const std::uint64_t fp = make_fingerprint_(name, fields);
        return TensorSchema(std::move(name), std::move(fields), fp);
    }

    [[nodiscard]] bool valid() const noexcept { return fingerprint_ != 0; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] std::uint64_t fingerprint() const noexcept
    {
        return fingerprint_;
    }
    [[nodiscard]] std::size_t size() const noexcept { return fields_.size(); }
    [[nodiscard]] bool empty() const noexcept { return fields_.empty(); }
    [[nodiscard]] const std::vector<TensorFieldSpec>& fields() const noexcept
    {
        return fields_;
    }

    [[nodiscard]] Result<const TensorFieldSpec*> field(
        const FieldHandle& handle) const
    {
        NN_TRY_CHECK(validate_handle_(handle, "TensorSchema::field"));
        return &fields_[handle.index];
    }

    // setup/反序列化期使用；训练热路径应缓存返回的 handle。
    [[nodiscard]] Result<FieldHandle> field(std::string_view field_name) const
    {
        if (!valid())
            NN_FAIL("TensorSchema::field: schema 未初始化");
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            if (fields_[i].name == field_name)
                return FieldHandle{fingerprint_, i};
        }
        NN_FAIL("TensorSchema::field: schema '" + name_ + "' 不含字段 '" +
                std::string(field_name) + "'");
    }

    [[nodiscard]] Result<FieldHandle> optional_field(
        std::string_view field_name) const
    {
        if (!valid()) return FieldHandle{};
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            if (fields_[i].name == field_name)
                return FieldHandle{fingerprint_, i};
        }
        return FieldHandle{};
    }

    [[nodiscard]] Result<void> validate(const FieldHandle& handle) const
    {
        return validate_handle_(handle, "TensorSchema::validate");
    }

    [[nodiscard]] bool operator==(const TensorSchema& other) const noexcept
    {
        if (fingerprint_ != other.fingerprint_ ||
            name_ != other.name_ ||
            fields_.size() != other.fields_.size())
            return false;
        for (std::size_t i = 0; i < fields_.size(); ++i)
        {
            const TensorFieldSpec& a = fields_[i];
            const TensorFieldSpec& b = other.fields_[i];
            if (a.name != b.name ||
                a.shape.rows != b.shape.rows ||
                a.shape.cols != b.shape.cols ||
                a.required != b.required ||
                a.precision != b.precision ||
                a.placement != b.placement)
                return false;
        }
        return true;
    }
};

// 输出侧语义别名：端口、ModelSpec 与 GradientBundle 共享同一有序 schema。
using OutputSchema = TensorSchema;

// ── Schema 紧凑二进制编解码（ModelSpec / checkpoint 扩展 section 共用）────
//
// 布局（小端）：
//   "NNS1" | u32 name_len | name | u32 field_count |
//   per field: u32 name_len | name | u64 rows | u64 cols |
//              u8 required | u8 precision_plus_one | u8 placement
// 行/列无约束编码为 UINT64_MAX。解码走公开 create()，因此重复名/空名等
// 非法输入得到与程序内构造完全相同的 Result 错误。
namespace detail
{
inline void schema_put_u32(std::string& out, std::uint32_t v)
{
    for (unsigned i = 0; i < 4; ++i)
        out.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

inline void schema_put_u64(std::string& out, std::uint64_t v)
{
    for (unsigned i = 0; i < 8; ++i)
        out.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
}

[[nodiscard]] inline Result<std::uint32_t> schema_take_u32(std::string_view& in)
{
    if (in.size() < 4)
        NN_FAIL("TensorSchema::decode: 截断（u32）");
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i)
        v |= static_cast<std::uint32_t>(
                 static_cast<unsigned char>(in[i]))
             << (i * 8);
    in.remove_prefix(4);
    return v;
}

[[nodiscard]] inline Result<std::uint64_t> schema_take_u64(std::string_view& in)
{
    if (in.size() < 8)
        NN_FAIL("TensorSchema::decode: 截断（u64）");
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i)
        v |= static_cast<std::uint64_t>(
                 static_cast<unsigned char>(in[i]))
             << (i * 8);
    in.remove_prefix(8);
    return v;
}

[[nodiscard]] inline Result<std::string> schema_take_string(
    std::string_view& in)
{
    NN_TRY(n, schema_take_u32(in));
    if (*n > in.size())
        NN_FAIL("TensorSchema::decode: 字符串长度越界");
    std::string out(in.substr(0, *n));
    in.remove_prefix(*n);
    return out;
}

inline void schema_put_string(std::string& out, std::string_view s)
{
    if (s.size() > std::numeric_limits<std::uint32_t>::max())
        NN_UNWRAP(false, "TensorSchema::encode: 字符串过长");
    schema_put_u32(out, static_cast<std::uint32_t>(s.size()));
    out.append(s);
}
} // namespace detail

[[nodiscard]] inline std::string encode_tensor_schema(const TensorSchema& schema)
{
    std::string out("NNS1", 4);
    detail::schema_put_string(out, schema.name());
    if (schema.size() > std::numeric_limits<std::uint32_t>::max())
        NN_UNWRAP(false, "TensorSchema::encode: 字段数过多");
    detail::schema_put_u32(
        out, static_cast<std::uint32_t>(schema.size()));
    constexpr std::uint64_t no_axis = std::numeric_limits<std::uint64_t>::max();
    for (const TensorFieldSpec& f : schema.fields())
    {
        detail::schema_put_string(out, f.name);
        detail::schema_put_u64(
            out, f.shape.rows ? static_cast<std::uint64_t>(*f.shape.rows)
                              : no_axis);
        detail::schema_put_u64(
            out, f.shape.cols ? static_cast<std::uint64_t>(*f.shape.cols)
                              : no_axis);
        out.push_back(static_cast<char>(f.required ? 1 : 0));
        out.push_back(static_cast<char>(
            f.precision ? static_cast<std::uint8_t>(*f.precision) + 1u : 0u));
        out.push_back(static_cast<char>(
            static_cast<std::uint8_t>(f.placement)));
    }
    return out;
}

[[nodiscard]] inline Result<TensorSchema> decode_tensor_schema(
    std::string_view bytes)
{
    if (bytes.size() < 4 || bytes.substr(0, 4) != "NNS1")
        NN_FAIL("TensorSchema::decode: 魔数不匹配");
    bytes.remove_prefix(4);
    NN_TRY(name, detail::schema_take_string(bytes));
    NN_TRY(count, detail::schema_take_u32(bytes));
    if (*count > kMaxTensorSchemaFields)
        NN_FAIL("TensorSchema::decode: 字段数超过上限 " + std::to_string(kMaxTensorSchemaFields));
    // 每个字段至少包含 name_len + rows + cols + 3 个 flags。
    constexpr std::size_t min_field_bytes = 4 + 8 + 8 + 3;
    if (*count > bytes.size() / min_field_bytes)
        NN_FAIL("TensorSchema::decode: 字段数与负载长度不一致");
    std::vector<TensorFieldSpec> fields;
    fields.reserve(*count);
    constexpr std::uint64_t no_axis = std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t i = 0; i < *count; ++i)
    {
        NN_TRY(field_name, detail::schema_take_string(bytes));
        NN_TRY(rows, detail::schema_take_u64(bytes));
        NN_TRY(cols, detail::schema_take_u64(bytes));
        if (bytes.size() < 3)
            NN_FAIL("TensorSchema::decode: 截断（field flags）");
        const unsigned char required =
            static_cast<unsigned char>(bytes[0]);
        const unsigned char precision_tag =
            static_cast<unsigned char>(bytes[1]);
        const unsigned char placement =
            static_cast<unsigned char>(bytes[2]);
        bytes.remove_prefix(3);
        if (required > 1)
            NN_FAIL("TensorSchema::decode: required 标志非法");
        if (precision_tag > static_cast<unsigned char>(Precision::T1_58) + 1u)
            NN_FAIL("TensorSchema::decode: precision 标志非法");
        if (placement >
            static_cast<unsigned char>(TensorPlacement::GPU))
            NN_FAIL("TensorSchema::decode: placement 标志非法");

        TensorFieldSpec field;
        field.name = std::move(*field_name);
        if (*rows != no_axis)
        {
            if (*rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
                NN_FAIL("TensorSchema::decode: rows 超出平台范围");
            field.shape.rows = static_cast<std::size_t>(*rows);
        }
        if (*cols != no_axis)
        {
            if (*cols > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
                NN_FAIL("TensorSchema::decode: cols 超出平台范围");
            field.shape.cols = static_cast<std::size_t>(*cols);
        }
        field.required = required != 0;
        if (precision_tag != 0)
            field.precision =
                static_cast<Precision>(precision_tag - 1u);
        field.placement =
            static_cast<TensorPlacement>(placement);
        fields.push_back(std::move(field));
    }
    if (!bytes.empty())
        NN_FAIL("TensorSchema::decode: 尾部存在未消费字节");
    return TensorSchema::create(std::move(*name), std::move(fields));
}

// ── 名称列表（output packing 等纯结构元数据）──────────────────────────
[[nodiscard]] inline std::string encode_tensor_name_list(
    const std::vector<std::string>& names)
{
    std::string out("NNL1", 4);
    if (names.size() > std::numeric_limits<std::uint32_t>::max())
        NN_UNWRAP(false, "encode_tensor_name_list: 名称过多");
    detail::schema_put_u32(out, static_cast<std::uint32_t>(names.size()));
    for (const std::string& name : names)
        detail::schema_put_string(out, name);
    return out;
}

[[nodiscard]] inline Result<std::vector<std::string>>
decode_tensor_name_list(std::string_view bytes)
{
    if (bytes.size() < 4 || bytes.substr(0, 4) != "NNL1")
        NN_FAIL("decode_tensor_name_list: 魔数不匹配");
    bytes.remove_prefix(4);
    NN_TRY(count, detail::schema_take_u32(bytes));
    if (*count > kMaxTensorSchemaFields)
        NN_FAIL("decode_tensor_name_list: 名称数超过上限 " +
                std::to_string(kMaxTensorSchemaFields));
    // 每个名称至少有一个 u32 长度前缀。
    if (*count > bytes.size() / 4) NN_FAIL("decode_tensor_name_list: 名称数与负载长度不一致");
    std::vector<std::string> names;
    names.reserve(*count);
    for (std::uint32_t i = 0; i < *count; ++i)
    {
        NN_TRY(name, detail::schema_take_string(bytes));
        names.push_back(std::move(*name));
    }
    if (!bytes.empty())
        NN_FAIL("decode_tensor_name_list: 尾部存在未消费字节");
    return names;
}

} // namespace nn
