#pragma once

// ── compute_layer_bundle.hpp — 多输出分支与行带打包适配 ──────────────────
//
// 公共语义是多端口输出。RowBandPacking 只把端口映射到一张连续行带张量，
// 供仍接收单 Tensor 的旧调用方使用；named API 从不依赖行区间。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compute_layer_base.hpp"
#include "named_tensor_set.hpp"
#include "tensor_schema.hpp"

namespace nn
{

struct RowBand
{
    std::string name;
    std::size_t first_row = 0;
    std::size_t row_count = 0;
};

class RowBandPacking
{
private:
    TensorSchema schema_;
    std::vector<RowBand> bands_;
    std::size_t total_rows_ = 0;

    RowBandPacking(TensorSchema schema, std::vector<RowBand> bands,
                   std::size_t total_rows)
        : schema_(std::move(schema)),
          bands_(std::move(bands)),
          total_rows_(total_rows)
    {
    }

public:
    RowBandPacking() = default;
    RowBandPacking(const RowBandPacking&) = default;
    RowBandPacking& operator=(const RowBandPacking&) = default;
    RowBandPacking(RowBandPacking&&) noexcept = default;
    RowBandPacking& operator=(RowBandPacking&&) noexcept = default;
    ~RowBandPacking() = default;

    [[nodiscard]] static Result<RowBandPacking> create(
        const TensorSchema& schema, std::vector<std::string> order = {})
    {
        if (!schema.valid())
            NN_FAIL("RowBandPacking::create: schema 未初始化");
        if (order.empty())
        {
            order.reserve(schema.size());
            for (const TensorFieldSpec& f : schema.fields())
                order.push_back(f.name);
        }
        if (order.size() != schema.size())
            NN_FAIL("RowBandPacking::create: 打包字段必须无遗漏地覆盖 schema");

        std::vector<bool> seen(schema.size(), false);
        std::vector<RowBand> bands;
        bands.reserve(order.size());
        std::size_t cursor = 0;
        for (const std::string& name : order)
        {
            NN_TRY(handle, schema.field(name));
            if (seen[handle->index])
                NN_FAIL("RowBandPacking::create: 打包字段 '" + name +
                        "' 重复");
            seen[handle->index] = true;
            const TensorFieldSpec& field = schema.fields()[handle->index];
            if (!field.shape.rows)
                NN_FAIL("RowBandPacking::create: 字段 '" + name +
                        "' 未声明固定 rows，无法建立行带；请先给 schema 明确形状");
            const std::size_t rows = *field.shape.rows;
            if (rows > static_cast<std::size_t>(-1) - cursor)
                NN_FAIL("RowBandPacking::create: 行数溢出");
            bands.push_back(RowBand{name, cursor, rows});
            cursor += rows;
        }
        return RowBandPacking(schema, std::move(bands), cursor);
    }

    [[nodiscard]] bool valid() const noexcept { return schema_.valid(); }
    [[nodiscard]] const TensorSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] const std::vector<RowBand>& bands() const noexcept
    {
        return bands_;
    }
    [[nodiscard]] std::size_t total_rows() const noexcept { return total_rows_; }

    [[nodiscard]] Result<Tensor> pack(
        ComputeEngine& engine, const NamedTensorSet& outputs) const
    {
        if (!valid())
            NN_FAIL("RowBandPacking::pack: packing 未初始化");
        if (outputs.schema() != schema_)
            NN_FAIL("RowBandPacking::pack: 输出 schema 与 packing 不匹配");

        const Tensor* first = nullptr;
        for (const RowBand& band : bands_)
        {
            const Tensor* value = outputs.find(band.name);
            if (value == nullptr)
                NN_FAIL("RowBandPacking::pack: 缺少输出端口 '" + band.name + "'");
            if (first == nullptr) first = value;
            if (value->cols() != first->cols())
                NN_FAIL("RowBandPacking::pack: 端口列数不一致");
            if (value->precision() != first->precision())
                NN_FAIL("RowBandPacking::pack: 端口精度不一致");
            if (value->rows() != band.row_count)
                NN_FAIL("RowBandPacking::pack: 端口 '" + band.name +
                        "' 行数与 schema/packing 不一致");
        }
        if (first == nullptr)
            NN_FAIL("RowBandPacking::pack: schema 没有输出端口");

        NN_TRY(packed,
               engine.try_create_tensor(total_rows_, first->cols(),
                                        first->precision()));
        for (const RowBand& band : bands_)
        {
            const Tensor* value = outputs.find(band.name);
            NN_ASSERT_ALWAYS(value != nullptr, "RowBandPacking::pack lost port");
            NN_TRY_CHECK(engine.insert_rows(*packed, band.first_row, *value));
        }
        return packed;
    }

    [[nodiscard]] Result<NamedTensorSet> unpack(
        ComputeEngine& engine, const Tensor& packed) const
    {
        if (!valid())
            NN_FAIL("RowBandPacking::unpack: packing 未初始化");
        if (packed.rows() != total_rows_)
            NN_FAIL("RowBandPacking::unpack: 打包张量行数不匹配（期望 " +
                    std::to_string(total_rows_) + "，实际 " +
                    std::to_string(packed.rows()) + "）");

        std::vector<NamedTensor> values;
        values.reserve(bands_.size());
        for (const RowBand& band : bands_)
        {
            NN_TRY(part,
                   engine.slice_rows(packed, band.first_row, band.row_count));
            values.push_back(NamedTensor{
                .name = band.name,
                .value = std::move(*part),
            });
        }
        return NamedTensorSet::create(schema_, std::move(values));
    }

    [[nodiscard]] Result<GradientBundle> unpack_gradients(
        ComputeEngine& engine, const Tensor& packed) const
    {
        if (!valid())
            NN_FAIL("RowBandPacking::unpack_gradients: packing 未初始化");
        if (packed.rows() != total_rows_)
            NN_FAIL("RowBandPacking::unpack_gradients: 打包张量行数不匹配"
                    "（期望 " + std::to_string(total_rows_) + "，实际 " +
                    std::to_string(packed.rows()) + "）");

        std::vector<NamedTensor> values;
        values.reserve(bands_.size());
        for (const RowBand& band : bands_)
        {
            NN_TRY(part,
                   engine.slice_rows(packed, band.first_row, band.row_count));
            values.push_back(NamedTensor{
                .name = band.name,
                .value = std::move(*part),
            });
        }
        return GradientBundle::create(schema_, std::move(values));
    }
};

class OutputBundleLayer final : public Layer
{
private:
    struct Port
    {
        std::string name;
        std::unique_ptr<Layer> branch;
    };

    TensorSchema schema_;
    std::vector<Port> ports_;
    std::optional<RowBandPacking> packing_;
    std::vector<bool> produced_;
    bool forward_ready_ = false;

    [[nodiscard]] Port* find_port_(std::string_view name) noexcept
    {
        for (Port& p : ports_)
            if (p.name == name) return &p;
        return nullptr;
    }

    [[nodiscard]] const Port* find_port_(std::string_view name) const noexcept
    {
        for (const Port& p : ports_)
            if (p.name == name) return &p;
        return nullptr;
    }

    [[nodiscard]] Result<NamedTensorSet> run_forward_(
        const Tensor& input, const TensorSchema& requested_schema)
    {
        if (requested_schema != schema_)
            NN_FAIL("OutputBundleLayer: named forward schema 不匹配");

        std::vector<NamedTensor> values;
        values.reserve(schema_.size());
        produced_.assign(schema_.size(), false);
        for (std::size_t i = 0; i < schema_.size(); ++i)
        {
            const TensorFieldSpec& field = schema_.fields()[i];
            const Port* port = find_port_(field.name);
            if (port == nullptr)
            {
                if (field.required)
                    NN_FAIL("OutputBundleLayer: 必需输出端口 '" + field.name +
                            "' 没有分支");
                continue;
            }
            NN_TRY(out, port->branch->forward(input));
            produced_[i] = true;
            values.push_back(NamedTensor{
                .name = field.name,
                .value = std::move(*out),
            });
        }
        if (values.empty())
            NN_FAIL("OutputBundleLayer: 本次 forward 没有产生任何输出");
        NN_TRY(set, NamedTensorSet::create(schema_, std::move(values)));
        forward_ready_ = true;
        return set;
    }

public:
    explicit OutputBundleLayer(TensorSchema schema)
        : schema_(std::move(schema))
    {
        NN_ASSERT_ALWAYS(schema_.valid(),
                         "OutputBundleLayer: schema 必须有效");
    }

    [[nodiscard]] const char* layer_name() const noexcept override
    {
        return "OutputBundleLayer";
    }

    [[nodiscard]] Result<TensorSchema> output_schema() const override
    {
        return schema_;
    }

    [[nodiscard]] Result<void> add_port(
        std::string name, std::unique_ptr<Layer> branch)
    {
        if (binding_state() == BindingState::ready)
            NN_FAIL("OutputBundleLayer::add_port: 层已 init，不能增加端口");
        if (!branch)
            NN_FAIL("OutputBundleLayer::add_port: 分支为空");
        NN_TRY_CHECK(schema_.field(name));
        if (find_port_(name) != nullptr)
            NN_FAIL("OutputBundleLayer::add_port: 输出端口 '" + name +
                    "' 重复");

        branch->set_precision_profile(p_);
        ports_.push_back(Port{std::move(name), std::move(branch)});
        return {};
    }

    [[nodiscard]] Result<void> set_row_band_packing(
        std::vector<std::string> order = {})
    {
        if (binding_state() == BindingState::ready)
            NN_FAIL("OutputBundleLayer::set_row_band_packing: 层已 init，"
                    "packing 之后不可变");
        NN_TRY(p, RowBandPacking::create(schema_, std::move(order)));
        packing_ = std::move(*p);
        return {};
    }

    [[nodiscard]] bool has_row_band_packing() const noexcept
    {
        return packing_.has_value();
    }

    void set_precision_profile(const PrecisionProfile& profile) override
    {
        Layer::set_precision_profile(profile);
        for (Port& p : ports_)
            p.branch->set_precision_profile(profile);
    }

    [[nodiscard]] Result<Tensor> forward(const Tensor& input) override
    {
        NN_TRY(schema, output_schema());
        NN_TRY(set, run_forward_(input, *schema));
        if (!packing_)
        {
            if (set->size() != 1)
                NN_FAIL("OutputBundleLayer::forward: 多输出层未配置 RowBandPacking；"
                        "旧单 Tensor ABI 调用方必须先显式选择 packing");
            const FieldHandle h{schema->fingerprint(), 0};
            NN_TRY(value, set->require(h));
            return **value;
        }
        return packing_->pack(engine_ref(), *set);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        if (!packing_)
            NN_FAIL("OutputBundleLayer::backward: 多输出 backwards 必须使用 "
                    "backward_named()，或先配置 RowBandPacking");
        NN_TRY(grads,
               packing_->unpack_gradients(engine_ref(), grad_output));
        return backward_named(*grads);
    }

    [[nodiscard]] Result<NamedTensorSet> forward_named(
        const Tensor& input) override
    {
        return run_forward_(input, schema_);
    }

    [[nodiscard]] Result<NamedTensorSet> forward_named(
        const Tensor& input, const TensorSchema& schema) override
    {
        return run_forward_(input, schema);
    }

    [[nodiscard]] Result<Tensor> backward_named(
        const GradientBundle& gradients) override
    {
        return backward_named(gradients, schema_);
    }

    [[nodiscard]] Result<Tensor> backward_named(
        const GradientBundle& gradients,
        const TensorSchema& requested_schema) override
    {
        if (!forward_ready_)
            NN_FAIL("OutputBundleLayer::backward_named: 没有匹配的 forward");
        if (requested_schema != schema_ ||
            gradients.schema() != schema_)
            NN_FAIL("OutputBundleLayer::backward_named: schema 不匹配");

        // backward 只能消费一次；错误路径也不保留半配对状态。
        forward_ready_ = false;
        Tensor input_grad;
        bool have_input_grad = false;
        for (std::size_t i = 0; i < schema_.size(); ++i)
        {
            const TensorFieldSpec& field = schema_.fields()[i];
            const FieldHandle handle{schema_.fingerprint(), i};
            const Tensor* grad = gradients.find(handle);
            Port* port = find_port_(field.name);

            if (!produced_[i])
            {
                if (port == nullptr && grad != nullptr)
                    NN_FAIL("OutputBundleLayer::backward_named: 未激活端口 '" +
                            field.name + "' 收到了梯度");
                continue;
            }
            if (grad == nullptr)
                NN_FAIL("OutputBundleLayer::backward_named: 激活端口 '" +
                        field.name + "' 缺少梯度");
            NN_ASSERT_ALWAYS(port != nullptr,
                             "OutputBundleLayer backward lost active port");
            NN_TRY(branch_grad, port->branch->backward(*grad));
            if (!have_input_grad)
            {
                input_grad = std::move(*branch_grad);
                have_input_grad = true;
            }
            else
            {
                if (branch_grad->rows() != input_grad.rows() ||
                    branch_grad->cols() != input_grad.cols() ||
                    branch_grad->precision() != input_grad.precision())
                    NN_FAIL("OutputBundleLayer::backward_named: 分支对输入的"
                            "梯度形状/精度不一致");
                NN_TRY_CHECK(engine_ref().accumulate(input_grad, *branch_grad));
            }
        }
        if (!have_input_grad)
            NN_FAIL("OutputBundleLayer::backward_named: 没有可反向的输出端口");
        return input_grad;
    }

    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> out;
        for (Port& p : ports_)
        {
            auto ps = p.branch->parameters();
            out.insert(out.end(), ps.begin(), ps.end());
        }
        return out;
    }

    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> out;
        for (Port& p : ports_)
        {
            auto gs = p.branch->param_gradients();
            out.insert(out.end(), gs.begin(), gs.end());
        }
        return out;
    }

    [[nodiscard]] std::vector<TensorRef> extra_state() override
    {
        std::vector<TensorRef> out;
        for (Port& p : ports_)
        {
            auto xs = p.branch->extra_state();
            out.insert(out.end(), xs.begin(), xs.end());
        }
        return out;
    }

    void set_training(bool training) override
    {
        for (Port& p : ports_) p.branch->set_training(training);
    }

    void set_flush_interval(std::size_t interval) override
    {
        for (Port& p : ports_) p.branch->set_flush_interval(interval);
    }

    void set_checkpoint_every(std::size_t stride) override
    {
        for (Port& p : ports_) p.branch->set_checkpoint_every(stride);
    }

    void set_activation_offload(bool enabled) override
    {
        for (Port& p : ports_) p.branch->set_activation_offload(enabled);
    }

    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        for (Port& p : ports_) p.branch->set_doc_ids(ids);
    }

    [[nodiscard]] std::size_t offload_ram_bytes() override
    {
        std::size_t total = 0;
        for (Port& p : ports_) total += p.branch->offload_ram_bytes();
        return total;
    }

    [[nodiscard]] std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> out;
        for (Port& p : ports_)
        {
            auto caches = p.branch->activation_cache();
            out.insert(out.end(), caches.begin(), caches.end());
        }
        return out;
    }

    void clear_cache() override
    {
        for (Port& p : ports_) p.branch->clear_cache();
        forward_ready_ = false;
    }

protected:
    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        for (Port& p : ports_)
            NN_TRY_CHECK(p.branch->init(engine));
        for (const TensorFieldSpec& field : schema_.fields())
        {
            if (field.required && find_port_(field.name) == nullptr)
                NN_FAIL("OutputBundleLayer::init: 必需输出端口 '" + field.name +
                        "' 没有分支");
        }
        return {};
    }

    [[nodiscard]] std::vector<Layer*> param_children() noexcept override
    {
        std::vector<Layer*> out;
        out.reserve(ports_.size());
        for (Port& p : ports_) out.push_back(p.branch.get());
        return out;
    }
};

} // namespace nn
