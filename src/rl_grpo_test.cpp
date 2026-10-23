// ── rl_grpo_test — group advantage + GRPO preset（RL-1d）──────────────────
//
// 验收：
//   [1] 多组 returns -> 组内总体标准化；
//   [2] 零方差组输出 0，不产生 NaN；
//   [3] GRPO preset == GroupAdvantage + ClippedPolicy + Entropy 手工组合；
//   [4] 单组、少样本组与缺少 old_log_prob 的失败路径。
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

nn::Tensor matrix_tensor(nn::ComputeEngine& engine, std::size_t rows,
                         std::size_t cols,
                         std::initializer_list<float> values)
{
    nn::Matrix m(rows, cols);
    NN_ASSERT_ALWAYS(values.size() == m.size(), "matrix_tensor size");
    auto span = m.span();
    std::size_t i = 0;
    for (float v : values) span[i++] = v;
    return upload(engine, m);
}

std::vector<float> values_of(nn::ComputeEngine& engine,
                             const nn::Tensor& tensor)
{
    auto m = download(engine, tensor);
    std::vector<float> out;
    out.reserve(m.size());
    for (float v : m.span()) out.push_back(v);
    return out;
}

bool near_values(const std::vector<float>& got,
                 const std::vector<float>& want,
                 float tol = 2e-5f)
{
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) >
            tol * (1.0f + std::fabs(want[i])))
            return false;
    return true;
}

nn::Result<nn::TensorRecord> grpo_record(
    nn::Tensor actions, nn::Tensor mask, nn::Tensor returns,
    nn::Tensor old_log_prob, nn::Tensor advantage = {})
{
    auto schema = nn::ExperienceSchema::create(
        {"action", "action_mask", "return", "old_log_prob"});
    NN_TRY_CHECK(schema);
    std::vector<nn::NamedTensor> values{
        nn::NamedTensor{.name = "action", .value = actions},
        nn::NamedTensor{.name = "action_mask", .value = mask},
        nn::NamedTensor{.name = "return", .value = returns},
        nn::NamedTensor{.name = "old_log_prob", .value = old_log_prob},
    };
    if (advantage.valid())
        values.push_back(nn::NamedTensor{.name = "advantage",
                                         .value = advantage});
    return nn::TensorRecord::create(schema->schema(), std::move(values));
}

void test_grpo_multiple_groups()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 6,
        {1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         -1, 0, 1, 0, 0, 0});
    auto actions = matrix_tensor(*owner, 1, 6, {0, 1, 2, 0, 1, 2});
    auto mask = matrix_tensor(*owner, 1, 6, {1, 1, 1, 1, 1, 1});
    auto returns = matrix_tensor(*owner, 6, 1, {1, 2, 3, 10, 10, 10});
    auto old_logp = matrix_tensor(*owner, 1, 6, {0, 0, 0, 0, 0, 0});
    auto record = grpo_record(actions, mask, returns, old_logp);
    CHECK(record.has_value(), "GRPO record");
    if (!record) return;
    auto view = nn::PolicyBatchView::create(*record);
    CHECK(view.has_value(), "GRPO view");
    if (!view) return;

    const std::vector<nn::GroupSpan> spans{
        nn::GroupSpan{.first_row = 0, .row_count = 3},
        nn::GroupSpan{.first_row = 3, .row_count = 3},
    };
    auto output_schema = nn::TensorSchema::create(
        "grpo_output",
        {nn::tensor_field("policy",
                          nn::TensorShapeSpec::exact(3, 6))});
    CHECK(output_schema.has_value(), "GRPO output schema");
    if (!output_schema) return;

    nn::GrpoObjective preset;
    auto report = preset.evaluate(*owner, *output_schema, logits, *view,
                                  spans);
    CHECK(report.has_value(), "GRPO preset");
    if (!report) return;

    nn::GroupAdvantage group;
    auto grouped = group.apply(*owner, returns, spans);
    CHECK(grouped.has_value(), "GRPO group advantage");
    if (!grouped) return;
    auto grouped_row = NN_UNWRAP(owner->reshape(*grouped, 1, 6));
    auto manual_record = grpo_record(actions, mask, returns, old_logp,
                                     grouped_row);
    CHECK(manual_record.has_value(), "manual GRPO record with advantage");
    if (!manual_record) return;
    auto manual_view = nn::PolicyBatchView::create(*manual_record);
    CHECK(manual_view.has_value(), "manual GRPO view");
    if (!manual_view) return;

    const float s = std::sqrt(2.0f / 3.0f);
    auto manual_adv = manual_view->advantage();
    CHECK(manual_adv &&
              near_values(values_of(*owner, *manual_adv),
                          {-1.0f / s, 0, 1.0f / s, 0, 0, 0}),
          "GRPO group advantages");

    nn::ClippedPolicyObjective policy;
    nn::EntropyBonusObjective entropy;
    auto p = policy.evaluate_context(*owner, logits, *manual_view);
    auto e = entropy.evaluate_context(*owner, logits, *manual_view);
    CHECK(p && e, "manual GRPO components");
    if (!p || !e) return;
    CHECK(std::fabs(report->loss - (p->loss + e->loss)) < 1e-5f,
          "GRPO preset loss equals manual composition");
    auto expected_grad = NN_UNWRAP(owner->clone(p->gradient));
    NN_UNWRAP(owner->accumulate(expected_grad, e->gradient));
    auto gh = report->gradients.field("policy");
    CHECK(gh.has_value(), "GRPO gradient field");
    if (!gh) return;
    auto got = report->gradients.require(*gh);
    CHECK(got.has_value(), "GRPO gradient present");
    if (got)
        check_close(*owner, **got, expected_grad,
                    "GRPO preset gradient == manual composition", g_fail);
}

void test_grpo_zero_variance_and_missing_old()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 2, 3, {0, 0, 0, 0, 0, 0});
    auto actions = matrix_tensor(*owner, 1, 3, {0, 0, 0});
    auto mask = matrix_tensor(*owner, 1, 3, {1, 1, 1});
    auto returns = matrix_tensor(*owner, 3, 1, {5, 5, 5});
    auto old_logp = matrix_tensor(*owner, 1, 3, {-0.7f, -0.7f, -0.7f});
    auto record = grpo_record(actions, mask, returns, old_logp);
    CHECK(record.has_value(), "zero-variance GRPO record");
    if (!record) return;
    auto view = nn::PolicyBatchView::create(*record);
    CHECK(view.has_value(), "zero-variance GRPO view");
    if (!view) return;
    const std::vector<nn::GroupSpan> one_group{
        nn::GroupSpan{.first_row = 0, .row_count = 3}};
    auto schema = nn::TensorSchema::create(
        "grpo_output", {nn::tensor_field("policy")});
    CHECK(schema.has_value(), "zero-variance GRPO schema");
    if (!schema) return;
    nn::GrpoObjective preset;
    auto report = preset.evaluate(*owner, *schema, logits, *view,
                                  one_group);
    CHECK(report.has_value(), "zero-variance GRPO evaluates");
    if (report)
    {
        auto gh = report->gradients.field("policy");
        if (gh)
        {
            auto grad = report->gradients.require(*gh);
            bool finite = grad.has_value();
            if (grad)
                for (float v : values_of(*owner, **grad))
                    finite = finite && std::isfinite(v);
            CHECK(finite, "zero-variance GRPO remains finite");
        }
    }

    auto bad_schema = nn::ExperienceSchema::create(
        {"action", "action_mask", "return"});
    CHECK(bad_schema.has_value(), "missing-old schema");
    if (!bad_schema) return;
    auto missing_record = nn::TensorRecord::create(
        bad_schema->schema(),
        {nn::NamedTensor{.name = "action", .value = actions},
         nn::NamedTensor{.name = "action_mask", .value = mask},
         nn::NamedTensor{.name = "return", .value = returns}});
    CHECK(missing_record.has_value(), "missing-old record");
    if (!missing_record) return;
    auto missing_view = nn::PolicyBatchView::create(*missing_record);
    CHECK(missing_view.has_value(), "missing-old view");
    if (missing_view)
        CHECK(!preset.evaluate(*owner, *schema, logits, *missing_view,
                               one_group),
              "GRPO rejects missing old_log_prob");
}

} // namespace

int main()
{
    std::puts("=== RL-1d GRPO group objective ===");
    test_grpo_multiple_groups();
    test_grpo_zero_variance_and_missing_old();
    if (g_fail == 0) std::puts("All GRPO tests passed.");
    else std::printf("%d GRPO test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
