#pragma once

// ── named_tensor_set.hpp — 有序具名张量集合与梯度包 ──────────────────────
//
// NamedTensorSet 在构造期完成 schema 对齐：字段必须按 schema 顺序给出；
// 必需字段缺失、未知字段、重名、形状/精度/设备不符均返回 Result。因此训练
// 热路径通过 FieldHandle 直接按下标读取。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compute_tensor.hpp"
#include "tensor_schema.hpp"

namespace nn
{

struct NamedTensor
{
    std::string name;
    Tensor value;
};

class NamedTensorSet
{
private:
    TensorSchema schema_;
    std::vector<Tensor> values_;

    NamedTensorSet(TensorSchema schema, std::vector<Tensor> values)
        : schema_(std::move(schema)), values_(std::move(values))
    {
    }

    [[nodiscard]] Result<Tensor*> value_(const FieldHandle& handle)
    {
        NN_TRY_CHECK(schema_.validate(handle));
        return values_[handle.index].valid() ? &values_[handle.index] : nullptr;
    }

    [[nodiscard]] Result<const Tensor*> value_(const FieldHandle& handle) const
    {
        NN_TRY_CHECK(schema_.validate(handle));
        return values_[handle.index].valid() ? &values_[handle.index] : nullptr;
    }

    [[nodiscard]] static Result<void> validate_value_(
        const TensorFieldSpec& field, const Tensor& value)
    {
        if (!value.valid())
            NN_FAIL("NamedTensorSet: 字段 '" + field.name + "' 是无效张量");
        if (!field.shape.accepts(value.rows(), value.cols()))
            NN_FAIL("NamedTensorSet: 字段 '" + field.name + "' 形状不匹配（期望 " +
                    field.shape.to_string() + "，实际 " + value.shape_str() + "）");
        if (field.precision && *field.precision != value.precision())
            NN_FAIL("NamedTensorSet: 字段 '" + field.name + "' 精度不匹配（期望 " +
                    precision_name(*field.precision) + "，实际 " +
                    precision_name(value.precision()) + "）");
        if (field.placement != TensorPlacement::Any)
        {
            const TensorPlacement actual = value.device() == Device::CPU
                                               ? TensorPlacement::CPU
                                               : TensorPlacement::GPU;
            if (field.placement != actual)
                NN_FAIL("NamedTensorSet: 字段 '" + field.name +
                        "' 设备不匹配（期望 " +
                        tensor_placement_name(field.placement) + "，实际 " +
                        tensor_placement_name(actual) + "）");
        }
        return {};
    }

public:
    NamedTensorSet() = default;
    NamedTensorSet(const NamedTensorSet&) = delete;
    NamedTensorSet& operator=(const NamedTensorSet&) = delete;
    NamedTensorSet(NamedTensorSet&&) noexcept = default;
    NamedTensorSet& operator=(NamedTensorSet&&) noexcept = default;
    ~NamedTensorSet() = default;

    [[nodiscard]] static Result<NamedTensorSet> create(
        TensorSchema schema, std::vector<NamedTensor> tensors)
    {
        if (!schema.valid())
            NN_FAIL("NamedTensorSet::create: schema 未初始化");

        std::vector<Tensor> ordered(schema.size());
        std::vector<bool> seen(schema.size(), false);
        std::size_t matched = 0;

        for (const NamedTensor& nt : tensors)
        {
            auto h_r = schema.optional_field(nt.name);
            NN_TRY_CHECK(h_r);
            if (!h_r->valid())
                NN_FAIL("NamedTensorSet::create: schema '" + schema.name() +
                        "' 不认识字段 '" + nt.name + "'");
            if (seen[h_r->index])
                NN_FAIL("NamedTensorSet::create: 字段 '" + nt.name +
                        "' 重复出现");
            seen[h_r->index] = true;
            NN_TRY_CHECK(validate_value_(
                schema.fields()[h_r->index], nt.value));
            ordered[h_r->index] = nt.value;
            ++matched;
        }

        for (std::size_t i = 0; i < schema.size(); ++i)
        {
            if (!seen[i] && schema.fields()[i].required)
                NN_FAIL("NamedTensorSet::create: schema '" + schema.name() +
                        "' 缺少必需字段 '" + schema.fields()[i].name + "'");
        }
        if (tensors.size() != matched)
            NN_FAIL("NamedTensorSet::create: 输入字段计数不一致");

        return NamedTensorSet(std::move(schema), std::move(ordered));
    }

    [[nodiscard]] const TensorSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] bool empty() const noexcept { return values_.empty(); }

    [[nodiscard]] Result<FieldHandle> field(std::string_view name) const
    {
        return schema_.field(name);
    }

    [[nodiscard]] Result<FieldHandle> optional_field(
        std::string_view name) const
    {
        return schema_.optional_field(name);
    }

    [[nodiscard]] Result<Tensor*> require(const FieldHandle& handle)
    {
        return value_(handle);
    }

    [[nodiscard]] Result<const Tensor*> require(
        const FieldHandle& handle) const
    {
        return value_(handle);
    }

    // setup 期便捷入口；热路径请先缓存 FieldHandle。
    [[nodiscard]] Result<const Tensor*> require(
        std::string_view name) const
    {
        NN_TRY(h, schema_.field(name));
        return require(*h);
    }

    [[nodiscard]] const Tensor* find(const FieldHandle& handle) const noexcept
    {
        if (!handle.valid() ||
            handle.schema_fingerprint != schema_.fingerprint() ||
            handle.index >= values_.size())
            return nullptr;
        return values_[handle.index].valid() ? &values_[handle.index] : nullptr;
    }

    [[nodiscard]] const Tensor* find(std::string_view name) const noexcept
    {
        for (std::size_t i = 0; i < schema_.size(); ++i)
            if (schema_.fields()[i].name == name)
                return values_[i].valid() ? &values_[i] : nullptr;
        return nullptr;
    }

    [[nodiscard]] Tensor* find(const FieldHandle& handle) noexcept
    {
        return const_cast<Tensor*>(
            static_cast<const NamedTensorSet*>(this)->find(handle));
    }
};

// 梯度与输出端口共享同一 schema；类型独立，避免把推理值误当梯度传入。
class GradientBundle
{
private:
    NamedTensorSet values_;

    explicit GradientBundle(NamedTensorSet values)
        : values_(std::move(values))
    {
    }

public:
    GradientBundle() = default;
    GradientBundle(const GradientBundle&) = delete;
    GradientBundle& operator=(const GradientBundle&) = delete;
    GradientBundle(GradientBundle&&) noexcept = default;
    GradientBundle& operator=(GradientBundle&&) noexcept = default;
    ~GradientBundle() = default;

    [[nodiscard]] static Result<GradientBundle> create(
        TensorSchema schema, std::vector<NamedTensor> gradients)
    {
        NN_TRY(values,
               NamedTensorSet::create(std::move(schema), std::move(gradients)));
        return GradientBundle(std::move(*values));
    }

    [[nodiscard]] const TensorSchema& schema() const noexcept
    {
        return values_.schema();
    }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] Result<FieldHandle> field(std::string_view name) const
    {
        return values_.field(name);
    }
    [[nodiscard]] Result<const Tensor*> require(
        const FieldHandle& handle) const
    {
        return values_.require(handle);
    }
    [[nodiscard]] const Tensor* find(const FieldHandle& handle) const noexcept
    {
        return values_.find(handle);
    }
};

} // namespace nn
