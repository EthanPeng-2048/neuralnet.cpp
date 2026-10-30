// ── tensor_schema_test — 通用输出/experience 的公共契约 ────────────────────
//
// docs/internal/31 §6 公共基础 + RL-0 的 schema 前置：
//   [1] 有序字段、名称解析、稳定指纹；
//   [2] 重复/空字段名、跨 schema handle、越界 handle；
//   [3] 必需/可选字段、未知/重名字段；
//   [4] shape / precision / placement 约束；
//   [5] NamedTensorSet / GradientBundle 共享 schema 语义。
// ─────────────────────────────────────────────────────────────────────────

#include <optional>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

nn::TensorFieldSpec simple_field(std::string name)
{
    nn::TensorFieldSpec field;
    field.name = std::move(name);
    return field;
}

nn::TensorSchema output_schema(bool value_optional = false)
{
    std::vector<nn::TensorFieldSpec> fields{
        nn::TensorFieldSpec{
            .name = "policy",
            .shape = nn::TensorShapeSpec::any(),
            .required = true,
            .precision = nn::Precision::F32,
            .placement = nn::TensorPlacement::CPU,
        },
        nn::TensorFieldSpec{
            .name = "value",
            .shape = nn::TensorShapeSpec{.rows = 1, .cols = std::nullopt},
            .required = !value_optional,
            .precision = nn::Precision::F32,
            .placement = nn::TensorPlacement::CPU,
        },
    };
    auto r = nn::TensorSchema::create("policy_value", std::move(fields));
    NN_ASSERT_ALWAYS(r, "output_schema: create");
    return std::move(*r);
}

nn::Tensor make_f16_tensor(nn::ComputeEngine& engine, std::size_t rows, std::size_t cols)
{
    nn::Matrix host(rows, cols);
    auto r = engine.from_matrix(host, nn::Precision::F16);
    NN_ASSERT_ALWAYS(r, "make_f16_tensor: from_matrix");
    return std::move(*r);
}

void test_schema_and_handles()
{
    const nn::TensorSchema schema = output_schema();
    CHECK(schema.valid(), "schema valid");
    CHECK(schema.size() == 2, "schema field count");
    CHECK(schema.fields()[0].name == "policy", "schema preserves field order");
    CHECK(schema.fingerprint() != 0, "schema fingerprint non-zero");

    auto p = schema.field("policy");
    auto v = schema.field("value");
    CHECK(p && v, "field name resolves");
    if (!p || !v) return;
    CHECK(p->index == 0 && v->index == 1, "field handle indexes");
    CHECK(p->schema_fingerprint == schema.fingerprint(), "field handle carries schema fingerprint");
    CHECK(schema.field(*p).has_value(), "handle lookup succeeds");
    CHECK(!schema.field("missing").has_value(), "unknown field rejected");
    CHECK(!schema.optional_field("missing")->valid(), "optional lookup reports missing");

    const nn::TensorSchema same = output_schema();
    CHECK(same.fingerprint() == schema.fingerprint(),
          "same semantic schema has stable fingerprint");

    std::vector<nn::TensorFieldSpec> changed{
        simple_field("value"),
        simple_field("policy"),
    };
    auto other_r = nn::TensorSchema::create("policy_value", std::move(changed));
    CHECK(other_r.has_value(), "reordered schema builds");
    if (!other_r) return;
    CHECK(other_r->fingerprint() != schema.fingerprint(), "field order changes fingerprint");
    auto foreign = other_r->field("policy");
    CHECK(foreign.has_value(), "foreign field resolves there");
    if (foreign) CHECK(!schema.field(*foreign).has_value(), "cross-schema handle rejected");

    nn::FieldHandle bad{schema.fingerprint(), 99};
    CHECK(!schema.field(bad).has_value(), "out-of-range handle rejected");
    CHECK(!schema.field(nn::FieldHandle{}).has_value(), "uninitialized handle rejected");
}

void test_schema_construction_errors()
{
    CHECK(!nn::TensorSchema::create("", {simple_field("x")}), "empty schema name rejected");
    CHECK(!nn::TensorSchema::create("empty", {}), "empty schema rejected");
    CHECK(!nn::TensorSchema::create("blank", {simple_field("")}), "empty field name rejected");
    CHECK(!nn::TensorSchema::create("dup", {simple_field("x"), simple_field("x")}),
          "duplicate field name rejected");
}

void test_decode_bounds()
{
    std::string schema_bytes("NNS1");
    nn::detail::schema_put_string(schema_bytes, "x");
    nn::detail::schema_put_u32(schema_bytes, static_cast<std::uint32_t>(-1));
    CHECK(!nn::decode_tensor_schema(schema_bytes),
          "schema decode rejects hostile field count before allocation");

    std::string names_bytes("NNL1");
    nn::detail::schema_put_u32(names_bytes, static_cast<std::uint32_t>(-1));
    CHECK(!nn::decode_tensor_name_list(names_bytes),
          "name-list decode rejects hostile count before allocation");
}

void test_named_tensor_set()
{
    auto owner = nn::CpuEngine::create();
    const nn::TensorSchema schema = output_schema();
    nn::Tensor policy = make_tensor(*owner, 4, 3, 0.1f, 0.01f);
    nn::Tensor value = make_tensor(*owner, 1, 3, 0.2f, 0.01f);

    auto set_r =
        nn::NamedTensorSet::create(schema, {nn::NamedTensor{.name = "value", .value = value},
                                            nn::NamedTensor{.name = "policy", .value = policy}});
    CHECK(set_r.has_value(), "named set accepts out-of-order input");
    if (!set_r) return;
    CHECK(set_r->size() == 2, "named set size");

    auto p_h = set_r->field("policy");
    auto v_h = set_r->field("value");
    CHECK(p_h && v_h, "named set fields resolve");
    if (!p_h || !v_h) return;
    auto p = set_r->require(*p_h);
    auto v = set_r->require(*v_h);
    CHECK(p && (*p)->rows() == policy.rows() && (*p)->cols() == policy.cols(),
          "policy handle returns original tensor");
    CHECK(v && (*v)->rows() == value.rows() && (*v)->cols() == value.cols(),
          "value handle returns original tensor");
    CHECK(set_r->find("policy") == *p, "find by name");
    CHECK(set_r->find(*v_h) == *v, "find by handle");

    CHECK(!nn::NamedTensorSet::create(schema, {nn::NamedTensor{.name = "policy", .value = policy}}),
          "missing required field rejected");
    CHECK(!nn::NamedTensorSet::create(schema, {nn::NamedTensor{.name = "policy", .value = policy},
                                               nn::NamedTensor{.name = "policy", .value = policy},
                                               nn::NamedTensor{.name = "value", .value = value}}),
          "duplicate named tensor rejected");
    CHECK(!nn::NamedTensorSet::create(schema, {nn::NamedTensor{.name = "policy", .value = policy},
                                               nn::NamedTensor{.name = "value", .value = value},
                                               nn::NamedTensor{.name = "extra", .value = value}}),
          "unknown named tensor rejected");
    CHECK(!nn::NamedTensorSet::create(
              schema, {nn::NamedTensor{.name = "policy", .value = policy},
                       nn::NamedTensor{.name = "value", .value = make_tensor(*owner, 2, 3)}}),
          "shape constraint enforced");
    CHECK(!nn::NamedTensorSet::create(
              schema, {nn::NamedTensor{.name = "policy", .value = make_f16_tensor(*owner, 4, 3)},
                       nn::NamedTensor{.name = "value", .value = value}}),
          "precision constraint enforced");
}

void test_optional_and_gradients()
{
    auto owner = nn::CpuEngine::create();
    const nn::TensorSchema schema = output_schema(/*value_optional=*/true);
    nn::Tensor policy = make_tensor(*owner, 5, 2, 0.3f, 0.01f);

    auto only_policy =
        nn::NamedTensorSet::create(schema, {nn::NamedTensor{.name = "policy", .value = policy}});
    CHECK(only_policy.has_value(), "optional field may be absent");
    if (only_policy)
        CHECK(only_policy->find("value") == nullptr, "absent optional field has no tensor");

    auto grads =
        nn::GradientBundle::create(schema, {nn::NamedTensor{.name = "policy", .value = policy}});
    CHECK(grads.has_value(), "gradient bundle uses output schema");
    if (!grads) return;
    auto gh = grads->field("policy");
    CHECK(gh.has_value(), "gradient field resolves");
    if (gh)
    {
        auto g = grads->require(*gh);
        CHECK(g && (*g)->rows() == policy.rows(), "gradient bundle exposes tensor by handle");
    }
}

} // namespace

int main()
{
    std::puts("=== tensor_schema / named tensor contract ===");
    test_schema_and_handles();
    test_schema_construction_errors();
    test_decode_bounds();
    test_named_tensor_set();
    test_optional_and_gradients();
    if (g_fail == 0) std::puts("All tensor schema tests passed.");
    else std::printf("%d tensor schema test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
