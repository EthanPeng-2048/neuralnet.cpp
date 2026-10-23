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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core_errors.hpp"
#include "precision.hpp"

namespace nn
{

inline constexpr std::size_t kInvalidFieldIndex =
    static_cast<std::size_t>(-1);

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
        h = detail::schema_hash_bytes(h, name);
        h = detail::schema_hash_u64(h, fields.size());
        for (const TensorFieldSpec& f : fields)
        {
            h = detail::schema_hash_bytes(h, f.name);
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

        for (std::size_t i = 0; i < fields.size(); ++i)
        {
            if (fields[i].name.empty())
                NN_FAIL("TensorSchema::create: schema '" + name + "' 的第 " +
                        std::to_string(i) + " 个字段名为空");
            for (std::size_t j = 0; j < i; ++j)
            {
                if (fields[i].name == fields[j].name)
                    NN_FAIL("TensorSchema::create: schema '" + name +
                            "' 含重复字段 '" + fields[i].name + "'");
            }
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
};

} // namespace nn
