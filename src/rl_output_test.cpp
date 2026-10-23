// ── rl_output_test — 具名多输出 + 行带打包兼容门禁（RL-0）───────────────
//
// 验收：
//   [1] policy (A,T) / value (1,T) 具名端口与共享输入；
//   [2] named backward 按端口分流并汇总到共享 trunk；
//   [3] named 与 RowBandPacking 路径的 forward/参数梯度一致；
//   [4] 缺失/多余梯度与非法 packing 明确报错；
//   [5] ModelSpec output_schema/output_packing 往返，旧键缺失兼容。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

nn::TensorSchema policy_value_schema()
{
    auto r = nn::TensorSchema::create(
        "policy_value",
        {nn::tensor_field(
             "policy",
             nn::TensorShapeSpec{.rows = 2, .cols = std::nullopt},
             true, nn::Precision::F32, nn::TensorPlacement::CPU),
         nn::tensor_field(
             "value",
             nn::TensorShapeSpec{.rows = 1, .cols = std::nullopt},
             true, nn::Precision::F32, nn::TensorPlacement::CPU)});
    NN_ASSERT_ALWAYS(r, "policy_value_schema");
    return std::move(*r);
}

void set_matrix(nn::ComputeEngine& engine, nn::Tensor& tensor,
                const nn::Matrix& value)
{
    NN_UNWRAP(engine.copy_from(tensor, value));
}

auto make_bundle_model(nn::CpuEngine& engine, const nn::TensorSchema& schema)
    -> nn::Result<nn::Model>
{
    nn::Model model(engine);
    NN_TRY_CHECK(model.add<nn::Linear>(2, 2));

    auto bundle = std::make_unique<nn::OutputBundleLayer>(schema);
    NN_TRY_CHECK(bundle->add_port(
        "policy", std::make_unique<nn::Linear>(2, 2)));
    NN_TRY_CHECK(bundle->add_port(
        "value", std::make_unique<nn::Linear>(2, 1)));
    NN_TRY_CHECK(bundle->set_row_band_packing({"policy", "value"}));
    NN_TRY_CHECK(model.add_layer(std::move(bundle)));

    // 固定参数，使 named 与 packed 两条路径比较的是同一函数。
    auto matrix = [](std::size_t rows, std::size_t cols,
                     std::initializer_list<float> items) {
        nn::Matrix m(rows, cols);
        NN_ASSERT_ALWAYS(items.size() == m.size(),
                         "matrix initializer size");
        auto dst = m.span();
        std::size_t i = 0;
        for (float v : items) dst[i++] = v;
        return m;
    };
    const std::vector<nn::Matrix> values{
        matrix(2, 2, {0.1f, -0.2f, 0.3f, 0.4f}),
        matrix(2, 1, {0.05f, -0.07f}),
        matrix(2, 2, {0.2f, -0.1f, -0.3f, 0.25f}),
        matrix(2, 1, {0.01f, -0.02f}),
        matrix(1, 2, {0.35f, 0.15f}),
        matrix(1, 1, {-0.03f}),
    };
    auto params = model.parameters();
    NN_ASSERT_ALWAYS(params.size() == values.size(),
                     "bundle model parameter inventory");
    for (std::size_t i = 0; i < values.size(); ++i)
        set_matrix(engine, params[i].get(), values[i]);
    return model;
}

std::vector<nn::Matrix> snapshot(nn::CpuEngine& engine,
                                 std::vector<nn::TensorRef> refs)
{
    std::vector<nn::Matrix> out;
    out.reserve(refs.size());
    for (nn::TensorRef ref : refs)
        out.push_back(download(engine, ref.get()));
    return out;
}

bool matrices_equal(const std::vector<nn::Matrix>& a,
                    const std::vector<nn::Matrix>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].rows() != b[i].rows() ||
            a[i].cols() != b[i].cols())
            return false;
        const auto as = a[i].span();
        const auto bs = b[i].span();
        for (std::size_t j = 0; j < as.size(); ++j)
        {
            if (std::fabs(as[j] - bs[j]) >
                1e-5f * (1.0f + std::fabs(as[j]) + std::fabs(bs[j])))
                return false;
        }
    }
    return true;
}

struct ObjectiveGrads
{
    nn::GradientBundle named;
    nn::Tensor packed;
};

class FailSecondForwardLayer final : public nn::Layer
{
private:
    bool first_forward_ = true;

public:
    [[nodiscard]] const char* layer_name() const noexcept override
    { return "FailSecondForwardLayer"; }

    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& input) override
    {
        if (!first_forward_) NN_FAIL("FailSecondForwardLayer: intentional failure");
        first_forward_ = false;
        return engine_ref().slice_rows(input, 0, 1);
    }

    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& grad_output) override
    { return grad_output; }

protected:
    [[nodiscard]] nn::Result<void> init_impl(nn::ComputeEngine&) override { return {}; }
};

nn::Result<ObjectiveGrads> make_objective_grads(
    nn::CpuEngine& engine, const nn::TensorSchema& schema,
    const nn::NamedTensorSet& output)
{
    NN_TRY(policy_h, output.field("policy"));
    NN_TRY(value_h, output.field("value"));
    NN_TRY(policy, output.require(*policy_h));
    NN_TRY(value, output.require(*value_h));

    nn::MSELoss policy_loss;
    nn::MSELoss value_loss;
    auto zero_policy = make_tensor(engine, (*policy)->rows(),
                                   (*policy)->cols(),
                                   0.0f, 0.0f);
    auto zero_value = make_tensor(engine, (*value)->rows(),
                                  (*value)->cols(),
                                  0.0f, 0.0f);
    NN_TRY_CHECK(policy_loss.forward(engine, **policy, zero_policy));
    NN_TRY_CHECK(value_loss.forward(engine, **value, zero_value));
    NN_TRY(policy_grad, policy_loss.backward());
    NN_TRY(value_grad, value_loss.backward());
    NN_TRY_CHECK(engine.scale_inplace(*value_grad, 0.5f));

    auto named = nn::NamedTensorSet::create(
        schema,
        {nn::NamedTensor{.name = "policy", .value = *policy_grad},
         nn::NamedTensor{.name = "value", .value = *value_grad}});
    NN_TRY_CHECK(named);
    NN_TRY(packing, nn::RowBandPacking::create(schema, {"policy", "value"}));
    NN_TRY(packed, packing->pack(engine, *named));
    NN_TRY(named_grads, nn::GradientBundle::create(
                            schema,
                            {nn::NamedTensor{.name = "policy",
                                             .value = *policy_grad},
                             nn::NamedTensor{.name = "value",
                                             .value = *value_grad}}));
    return ObjectiveGrads{.named = std::move(*named_grads),
                          .packed = std::move(*packed)};
}

void test_named_and_packed()
{
    auto owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *owner;
    const nn::TensorSchema schema = policy_value_schema();

    auto named_model_r = make_bundle_model(engine, schema);
    auto packed_model_r = make_bundle_model(engine, schema);
    CHECK(named_model_r && packed_model_r, "build bundle models");
    if (!named_model_r || !packed_model_r) return;
    nn::Model& named_model = *named_model_r;
    nn::Model& packed_model = *packed_model_r;
    CHECK(named_model.output_schema().has_value(), "bundle exposes schema");
    CHECK(named_model.forward_named(
              make_tensor(engine, 2, 3, 0.2f, 0.1f))
              .has_value(),
          "named forward");

    const nn::Tensor input = make_tensor(engine, 2, 3, 0.2f, 0.1f);
    NN_UNWRAP(named_model.zero_grad());
    NN_UNWRAP(packed_model.zero_grad());

    auto named = named_model.forward_named(input);
    CHECK(named.has_value(), "named forward succeeds");
    if (!named) return;
    auto policy_h = named->field("policy");
    auto value_h = named->field("value");
    CHECK(policy_h && value_h, "named output fields");
    if (!policy_h || !value_h) return;
    auto policy = named->require(*policy_h);
    auto value = named->require(*value_h);
    CHECK(policy && value, "named output values");
    if (!policy || !value) return;
    CHECK((*policy)->rows() == 2 && (*policy)->cols() == 3,
          "policy shape (A,T)");
    CHECK((*value)->rows() == 1 && (*value)->cols() == 3,
          "value shape (1,T)");

    auto packing = NN_UNWRAP(nn::RowBandPacking::create(
        schema, {"policy", "value"}));
    auto named_packed = packing.pack(engine, *named);
    CHECK(named_packed.has_value(), "named outputs pack");
    auto packed = packed_model.forward(input);
    CHECK(packed.has_value(), "legacy packed forward");
    if (named_packed && packed)
        check_close(engine, *packed, *named_packed,
                    "packed forward equals named ports", g_fail);

    auto objective = make_objective_grads(engine, schema, *named);
    CHECK(objective.has_value(), "build policy/value objective gradients");
    if (!objective) return;

    CHECK(named_model.backward_named(objective->named).has_value(),
          "named backward");
    CHECK(packed_model.backward(objective->packed).has_value(),
          "packed backward");

    auto named_grads = snapshot(engine, named_model.param_gradients());
    auto packed_grads = snapshot(engine, packed_model.param_gradients());
    CHECK(matrices_equal(named_grads, packed_grads),
          "named and packed parameter gradients agree");
    CHECK(named_grads.size() == 6, "all branch/trunk gradients materialized");
    if (named_grads.size() == 6)
    {
        bool trunk_moves = false;
        for (std::size_t i = 0; i < 2; ++i)
            for (float v : named_grads[i].span())
                trunk_moves = trunk_moves || std::fabs(v) > 1e-7f;
        CHECK(trunk_moves, "shared trunk receives summed port gradients");
    }
}

void test_negative_paths()
{
    auto owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *owner;
    const nn::TensorSchema schema = policy_value_schema();

    auto bad_order = nn::RowBandPacking::create(
        schema, {"policy", "policy"});
    CHECK(!bad_order, "packing rejects duplicate field");
    auto missing_field = nn::RowBandPacking::create(schema, {"policy"});
    CHECK(!missing_field, "packing requires complete coverage");

    auto model_r = make_bundle_model(engine, schema);
    CHECK(model_r.has_value(), "negative model builds");
    if (!model_r) return;
    const nn::Tensor input = make_tensor(engine, 2, 2, 0.1f, 0.1f);
    auto output = model_r->forward_named(input);
    CHECK(output.has_value(), "negative named forward");
    if (!output) return;
    auto policy_h = output->field("policy");
    if (!policy_h) return;
    auto policy = output->require(*policy_h);
    if (!policy) return;
    auto wrong_schema = nn::TensorSchema::create(
        "wrong", {nn::tensor_field("policy")});
    CHECK(wrong_schema.has_value(), "wrong schema builds");
    if (!wrong_schema) return;
    auto wrong_grads = nn::GradientBundle::create(
        *wrong_schema,
        {nn::NamedTensor{.name = "policy", .value = **policy}});
    CHECK(wrong_grads.has_value(), "wrong-schema gradient bundle builds");
    if (wrong_grads)
        CHECK(!model_r->backward_named(*wrong_grads).has_value(),
              "backward rejects gradient schema mismatch");

    nn::Model stateful_model(engine);
    auto stateful_bundle = std::make_unique<nn::OutputBundleLayer>(schema);
    NN_UNWRAP(stateful_bundle->add_port("policy", std::make_unique<nn::Linear>(2, 2)));
    NN_UNWRAP(stateful_bundle->add_port("value", std::make_unique<FailSecondForwardLayer>()));
    NN_UNWRAP(stateful_model.add_layer(std::move(stateful_bundle)));

    const nn::Tensor stateful_input = make_tensor(engine, 2, 2, 0.1f, 0.1f);
    CHECK(stateful_model.forward_named(stateful_input).has_value(),
          "stateful bundle initial forward");
    CHECK(!stateful_model.forward_named(stateful_input).has_value(),
          "stateful bundle second forward fails");
    auto stale_grads = nn::GradientBundle::create(
        schema, {nn::NamedTensor{.name = "policy", .value = make_tensor(engine, 2, 2, 0.0f, 0.0f)},
                 nn::NamedTensor{.name = "value", .value = make_tensor(engine, 1, 2, 0.0f, 0.0f)}});
    CHECK(stale_grads.has_value(), "stale-state gradient bundle");
    if (stale_grads)
        CHECK(!stateful_model.backward_named(*stale_grads).has_value(),
              "failed forward invalidates previous backward state");
}

void test_model_spec_roundtrip()
{
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = {8, 4};
    spec.output_schema = policy_value_schema();
    spec.output_packing = {"policy", "value"};

    const std::string bytes = nn::detail::spec_to_kv(spec).serialize();
    auto kv = nn::KeyValueRecord::parse(bytes);
    CHECK(kv.has_value(), "output schema spec kv parses");
    if (!kv) return;
    auto loaded = nn::detail::spec_from_kv(*kv);
    CHECK(loaded.has_value(), "output schema spec decodes");
    if (!loaded) return;
    CHECK(loaded->output_schema.has_value(), "loaded schema present");
    CHECK(loaded->output_packing == spec.output_packing,
          "loaded packing order");
    CHECK(*loaded->output_schema == *spec.output_schema,
          "schema semantic roundtrip");
    CHECK(nn::spec_matches(spec, *loaded),
          "spec_matches includes output schema");

    nn::ModelSpec duplicate_packing = spec;
    duplicate_packing.output_packing = {"policy", "policy"};
    auto malformed = nn::detail::spec_from_kv(
        *nn::KeyValueRecord::parse(nn::detail::spec_to_kv(duplicate_packing).serialize()));
    CHECK(!malformed, "spec decode rejects duplicate output packing");

    nn::ModelSpec old;
    old.type = nn::ModelType::MLP;
    old.layer_dims = {8, 4};
    auto old_loaded = nn::detail::spec_from_kv(
        *nn::KeyValueRecord::parse(
            nn::detail::spec_to_kv(old).serialize()));
    CHECK(old_loaded && !old_loaded->output_schema &&
              old_loaded->output_packing.empty(),
          "legacy single-output spec stays schema-free");
}

} // namespace

int main()
{
    std::puts("=== RL-0 named outputs / row-band packing ===");
    test_named_and_packed();
    test_negative_paths();
    test_model_spec_roundtrip();
    if (g_fail == 0) std::puts("All RL output tests passed.");
    else std::printf("%d RL output test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
