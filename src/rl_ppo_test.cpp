// ── rl_ppo_test — PPO policy/value clip 与 preset 组合（RL-1c）─────────────
//
// 验收：
//   [1] ratio 低/中/高三分支，正负 advantage；
//   [2] policy clip 解析损失、指标、正/负 advantage 的有限差分梯度；
//   [3] value clip 双侧、越界梯度为 0；
//   [4] PPO preset == clipped policy + entropy + clipped value 手工组合；
//   [5] 缺少 old_log_prob / old_value 明确报错。
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

nn::Tensor matrix_tensor(nn::ComputeEngine& engine, std::size_t rows, std::size_t cols,
                         std::initializer_list<float> values)
{
    nn::Matrix m(rows, cols);
    NN_ASSERT_ALWAYS(values.size() == m.size(), "matrix_tensor size");
    auto span = m.span();
    std::size_t i = 0;
    for (float v : values)
        span[i++] = v;
    return upload(engine, m);
}

std::vector<float> values_of(nn::ComputeEngine& engine, const nn::Tensor& tensor)
{
    auto m = download(engine, tensor);
    std::vector<float> out;
    out.reserve(m.size());
    for (float v : m.span())
        out.push_back(v);
    return out;
}

bool near_values(const std::vector<float>& got, const std::vector<float>& want, float tol = 2e-5f)
{
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) > tol * (1.0f + std::fabs(want[i]))) return false;
    return true;
}

nn::TensorSchema fixture_schema()
{
    auto r = nn::ExperienceSchema::create(
        {"action", "action_mask", "advantage", "return", "old_value", "old_log_prob"});
    NN_ASSERT_ALWAYS(r, "fixture schema");
    return r->schema();
}

struct Fixture
{
    std::unique_ptr<nn::TensorRecord> record;
    nn::PolicyBatchView view;
};

nn::Result<Fixture> make_fixture(nn::Tensor actions, nn::Tensor mask, nn::Tensor advantage,
                                 nn::Tensor returns, nn::Tensor old_value, nn::Tensor old_log_prob)
{
    NN_TRY(record,
           nn::TensorRecord::create(
               fixture_schema(), {nn::NamedTensor{.name = "action", .value = actions},
                                  nn::NamedTensor{.name = "action_mask", .value = mask},
                                  nn::NamedTensor{.name = "value_mask", .value = mask},
                                  nn::NamedTensor{.name = "advantage", .value = advantage},
                                  nn::NamedTensor{.name = "return", .value = returns},
                                  nn::NamedTensor{.name = "old_value", .value = old_value},
                                  nn::NamedTensor{.name = "old_log_prob", .value = old_log_prob}}));
    Fixture out;
    out.record = std::make_unique<nn::TensorRecord>(std::move(*record));
    NN_TRY(view, nn::PolicyBatchView::create(*out.record));
    out.view = std::move(*view);
    return out;
}

nn::Tensor make_old_logp(nn::ComputeEngine& engine, const nn::Tensor& new_logp,
                         std::initializer_list<float> ratios)
{
    auto lp = download(engine, new_logp);
    nn::Matrix old(lp.rows(), lp.cols());
    auto src = lp.span();
    auto dst = old.span();
    std::size_t i = 0;
    for (float ratio : ratios)
    {
        dst[i] = src[i] - std::log(ratio);
        ++i;
    }
    return upload(engine, old);
}

void test_policy_clip_branches()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 2, {1, 0, 0, 2, -1, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {1, 1});
    auto returns = matrix_tensor(*owner, 1, 2, {0, 0});
    auto old_value = matrix_tensor(*owner, 1, 2, {0, 0});
    nn::CategoricalDistribution dist;
    auto new_logp = dist.log_prob(*owner, logits, actions);
    CHECK(new_logp.has_value(), "PPO new log-prob");
    if (!new_logp) return;
    auto old_logp = make_old_logp(*owner, *new_logp, {0.5f, 1.5f});
    auto fixture = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "positive-advantage PPO fixture");
    if (!fixture) return;

    nn::ClippedPolicyObjective objective(0.2f);
    auto out = objective.evaluate_context(*owner, logits, fixture->view);
    CHECK(out.has_value(), "clipped policy evaluates");
    if (!out) return;
    CHECK(std::fabs(out->loss - (-0.85f)) < 1e-5f, "PPO low/high clip branches: min(rA, clip(r)A)");
    CHECK(std::fabs(objective.metrics().ratio_mean - 1.0f) < 1e-5f, "PPO ratio metric");
    CHECK(std::fabs(objective.metrics().clip_fraction - 1.0f) < 1e-5f, "PPO clip fraction");

    constexpr float eps = 1e-3f;
    const auto analytic = values_of(*owner, out->gradient);
    const std::vector<float> base{1, 0, 0, 2, -1, 0};
    bool finite_ok = true;
    for (std::size_t i = 0; i < base.size(); ++i)
    {
        auto plus_v = base;
        plus_v[i] += eps;
        auto plus_t = matrix_tensor(
            *owner, 3, 2, {plus_v[0], plus_v[1], plus_v[2], plus_v[3], plus_v[4], plus_v[5]});
        auto plus_fix = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
        if (!plus_fix)
        {
            finite_ok = false;
            break;
        }
        auto plus = objective.evaluate_context(*owner, plus_t, plus_fix->view);

        auto minus_v = base;
        minus_v[i] -= eps;
        auto minus_t = matrix_tensor(
            *owner, 3, 2, {minus_v[0], minus_v[1], minus_v[2], minus_v[3], minus_v[4], minus_v[5]});
        auto minus_fix = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
        if (!minus_fix)
        {
            finite_ok = false;
            break;
        }
        auto minus = objective.evaluate_context(*owner, minus_t, minus_fix->view);
        if (!plus || !minus)
        {
            finite_ok = false;
            break;
        }
        const float numeric = (plus->loss - minus->loss) / (2 * eps);
        if (std::fabs(numeric - analytic[i]) >= 3e-3f)
            std::printf("[info] PPO finite-diff [%zu] numeric=%g analytic=%g\n", i,
                        static_cast<double>(numeric), static_cast<double>(analytic[i]));
        finite_ok = finite_ok && std::fabs(numeric - analytic[i]) < 3e-3f;
    }
    CHECK(finite_ok, "PPO policy clip finite-difference gradient");

    auto neg_adv = matrix_tensor(*owner, 1, 2, {-1, -1});
    auto neg_fixture = make_fixture(actions, mask, neg_adv, returns, old_value, old_logp);
    CHECK(neg_fixture.has_value(), "negative-advantage PPO fixture");
    if (neg_fixture)
    {
        auto neg = objective.evaluate_context(*owner, logits, neg_fixture->view);
        CHECK(neg && std::fabs(neg->loss - 1.15f) < 1e-5f, "PPO negative advantage clip branches");
    }
}

void test_negative_off_policy_gradient()
{
    auto owner = nn::CpuEngine::create();
    const std::vector<float> base{1, 0, 0, 2, -1, 0};
    auto logits =
        matrix_tensor(*owner, 3, 2, {base[0], base[1], base[2], base[3], base[4], base[5]});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {-1, -1});
    auto returns = matrix_tensor(*owner, 1, 2, {0, 0});
    auto old_value = matrix_tensor(*owner, 1, 2, {0, 0});
    nn::CategoricalDistribution dist;
    auto new_logp = dist.log_prob(*owner, logits, actions);
    CHECK(new_logp.has_value(), "negative off-policy new log-prob");
    if (!new_logp) return;
    // 第一列在容差带内，第二列位于 A<0 的裁剪侧。
    auto old_logp = make_old_logp(*owner, *new_logp, {1.1f, 0.5f});
    auto fixture = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "negative off-policy fixture");
    if (!fixture) return;

    nn::ClippedPolicyObjective objective(0.2f);
    auto out = objective.evaluate_context(*owner, logits, fixture->view);
    CHECK(out.has_value(), "negative off-policy objective evaluates");
    if (!out) return;
    const auto analytic = values_of(*owner, out->gradient);

    constexpr float eps = 1e-3f;
    bool finite_ok = true;
    for (std::size_t i = 0; i < base.size(); ++i)
    {
        auto plus_v = base;
        plus_v[i] += eps;
        auto plus_t = matrix_tensor(
            *owner, 3, 2, {plus_v[0], plus_v[1], plus_v[2], plus_v[3], plus_v[4], plus_v[5]});
        auto plus_fix = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
        if (!plus_fix)
        {
            finite_ok = false;
            break;
        }
        auto plus = objective.evaluate_context(*owner, plus_t, plus_fix->view);

        auto minus_v = base;
        minus_v[i] -= eps;
        auto minus_t = matrix_tensor(
            *owner, 3, 2, {minus_v[0], minus_v[1], minus_v[2], minus_v[3], minus_v[4], minus_v[5]});
        auto minus_fix = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
        if (!minus_fix)
        {
            finite_ok = false;
            break;
        }
        auto minus = objective.evaluate_context(*owner, minus_t, minus_fix->view);
        if (!plus || !minus)
        {
            finite_ok = false;
            break;
        }
        const float numeric = (plus->loss - minus->loss) / (2 * eps);
        finite_ok = finite_ok && std::fabs(numeric - analytic[i]) < 3e-3f;
    }
    CHECK(finite_ok, "negative-advantage PPO gradient exact off-policy");
}

void test_value_clip()
{
    auto owner = nn::CpuEngine::create();
    auto actions = matrix_tensor(*owner, 1, 2, {0, 0});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {0, 0});
    auto returns = matrix_tensor(*owner, 1, 2, {0, 1});
    auto old_value = matrix_tensor(*owner, 1, 2, {0, 1});
    auto old_logp = matrix_tensor(*owner, 1, 2, {0, 0});
    auto fixture = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "value-clip fixture");
    if (!fixture) return;
    auto value = matrix_tensor(*owner, 1, 2, {1, 1.8f});
    nn::ClippedValueObjective objective(0.2f, 1.0f);
    auto out = objective.evaluate_context(*owner, value, fixture->view);
    CHECK(out && std::fabs(out->loss - 0.82f) < 1e-5f, "value clip uses both sides");
    CHECK(out && near_values(values_of(*owner, out->gradient), {1, 0.8f}),
          "value clip selects max(unclipped, clipped) gradient");
    CHECK(std::fabs(objective.metrics().clip_fraction - 1.0f) < 1e-5f, "value clip fraction");
}

void test_preset_matches_manual()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 2, {1, 0, 0, 2, -1, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {1, -0.5f});
    auto returns = matrix_tensor(*owner, 1, 2, {1, 2});
    auto old_value = matrix_tensor(*owner, 1, 2, {0.5f, 1.5f});
    auto old_logp = matrix_tensor(*owner, 1, 2, {-0.2f, 0.1f});
    auto fixture = make_fixture(actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "PPO preset fixture");
    if (!fixture) return;
    auto value = matrix_tensor(*owner, 1, 2, {0.7f, 1.2f});

    nn::PpoConfig cfg;
    cfg.clip_epsilon = 0.2f;
    cfg.value_clip_epsilon = 0.2f;
    cfg.value_coefficient = 0.5f;
    cfg.entropy_coefficient = 0.01f;
    auto output_schema = nn::TensorSchema::create(
        "policy_value", {nn::tensor_field("policy", nn::TensorShapeSpec{.rows = 3, .cols = 2}),
                         nn::tensor_field("value", nn::TensorShapeSpec{.rows = 1, .cols = 2})});
    CHECK(output_schema.has_value(), "PPO output schema");
    if (!output_schema) return;

    nn::PpoObjective preset(cfg);
    auto report = preset.evaluate(*owner, *output_schema, logits, value, fixture->view);
    CHECK(report.has_value(), "PPO preset evaluates");
    if (!report) return;

    nn::ClippedPolicyObjective policy(cfg.clip_epsilon);
    nn::EntropyBonusObjective entropy(cfg.entropy_coefficient);
    nn::ClippedValueObjective value_obj(cfg.value_clip_epsilon, cfg.value_coefficient);
    auto p = policy.evaluate_context(*owner, logits, fixture->view);
    auto e = entropy.evaluate_context(*owner, logits, fixture->view);
    auto v = value_obj.evaluate_context(*owner, value, fixture->view);
    CHECK(p && e && v, "manual objective components evaluate");
    if (!p || !e || !v) return;
    CHECK(std::fabs(report->loss - (p->loss + e->loss + v->loss)) < 1e-5f,
          "PPO preset loss equals component composition");

    auto manual_policy_grad = NN_UNWRAP(owner->clone(p->gradient));
    NN_UNWRAP(owner->accumulate(manual_policy_grad, e->gradient));
    auto policy_h = report->gradients.field("policy");
    auto value_h = report->gradients.field("value");
    CHECK(policy_h && value_h, "PPO preset gradient bundle fields");
    if (!policy_h || !value_h) return;
    auto preset_policy = report->gradients.require(*policy_h);
    auto preset_value = report->gradients.require(*value_h);
    CHECK(preset_policy && preset_value, "PPO preset gradients present");
    if (preset_policy && preset_value)
    {
        check_close(*owner, **preset_policy, manual_policy_grad,
                    "PPO preset policy gradient == manual components", g_fail);
        check_close(*owner, **preset_value, v->gradient,
                    "PPO preset value gradient == manual components", g_fail);
    }
}

void test_missing_old_fields()
{
    auto owner = nn::CpuEngine::create();
    auto actions = matrix_tensor(*owner, 1, 1, {0});
    auto mask = matrix_tensor(*owner, 1, 1, {1});
    auto advantage = matrix_tensor(*owner, 1, 1, {1});
    auto returns = matrix_tensor(*owner, 1, 1, {1});
    auto schema = nn::ExperienceSchema::create({"action", "action_mask", "advantage", "return"});
    CHECK(schema.has_value(), "minimal PPO schema");
    if (!schema) return;
    auto record = nn::TensorRecord::create(
        schema->schema(), {nn::NamedTensor{.name = "action", .value = actions},
                           nn::NamedTensor{.name = "action_mask", .value = mask},
                           nn::NamedTensor{.name = "advantage", .value = advantage},
                           nn::NamedTensor{.name = "return", .value = returns}});
    CHECK(record.has_value(), "minimal PPO record");
    if (!record) return;
    auto view = nn::PolicyBatchView::create(*record);
    CHECK(view.has_value(), "minimal PPO view");
    if (!view) return;
    auto logits = matrix_tensor(*owner, 2, 1, {0, 0});
    auto value = matrix_tensor(*owner, 1, 1, {0});
    nn::ClippedPolicyObjective policy;
    nn::ClippedValueObjective value_obj;
    CHECK(!policy.evaluate_context(*owner, logits, *view),
          "PPO policy rejects missing old_log_prob");
    CHECK(!value_obj.evaluate_context(*owner, value, *view),
          "PPO value rejects missing old_value/value_mask");
}

} // namespace

int main()
{
    std::puts("=== RL-1c PPO clipped objectives ===");
    test_policy_clip_branches();
    test_negative_off_policy_gradient();
    test_value_clip();
    test_preset_matches_manual();
    test_missing_old_fields();
    if (g_fail == 0) std::puts("All PPO tests passed.");
    else std::printf("%d PPO test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
