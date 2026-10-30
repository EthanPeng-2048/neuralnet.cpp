// ── rl_policy_test — categorical distribution + objective 组件（RL-1b）────
//
// 验收：
//   [1] log-prob / entropy / sample 与手算参考；
//   [2] REINFORCE、baseline、全 mask；
//   [3] value regression、entropy bonus、sampled KL；
//   [4] entropy 解析梯度 vs 有限差分；
//   [5] objective 的无上下文 Loss 入口明确报错。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
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

class ForwardingDistribution final : public nn::PolicyDistribution
{
private:
    nn::CategoricalDistribution inner_;

public:
    [[nodiscard]] nn::Result<nn::Tensor> log_prob(nn::ComputeEngine& engine,
                                                  const nn::Tensor& parameters,
                                                  const nn::Tensor& actions) const override
    { return inner_.log_prob(engine, parameters, actions); }

    [[nodiscard]] nn::Result<nn::Tensor> entropy(nn::ComputeEngine& engine,
                                                 const nn::Tensor& parameters) const override
    { return inner_.entropy(engine, parameters); }

    [[nodiscard]] nn::Result<Evaluation>
    log_prob_with_gradient(nn::ComputeEngine& engine, const nn::Tensor& parameters,
                           const nn::Tensor& actions, const nn::Tensor& upstream) const override
    { return inner_.log_prob_with_gradient(engine, parameters, actions, upstream); }

    [[nodiscard]] nn::Result<Evaluation>
    entropy_with_gradient(nn::ComputeEngine& engine, const nn::Tensor& parameters,
                          const nn::Tensor& upstream) const override
    { return inner_.entropy_with_gradient(engine, parameters, upstream); }

    [[nodiscard]] nn::Result<SampleResult> sample(nn::ComputeEngine& engine,
                                                  const nn::Tensor& parameter_mesh,
                                                  std::uint64_t seed) const override
    { return inner_.sample(engine, parameter_mesh, seed); }
};

bool near_values(const std::vector<float>& got, const std::vector<float>& want, float tol = 2e-5f)
{
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) > tol * (1.0f + std::fabs(want[i]))) return false;
    return true;
}

std::vector<float> softmax_column(const std::vector<float>& logits, std::size_t rows,
                                  std::size_t col)
{
    const std::size_t cols = logits.size() / rows;
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t r = 0; r < rows; ++r)
        mx = std::max(mx, logits[r * cols + col]);
    std::vector<float> out(rows);
    float sum = 0;
    for (std::size_t r = 0; r < rows; ++r)
    {
        out[r] = std::exp(logits[r * cols + col] - mx);
        sum += out[r];
    }
    for (float& v : out)
        v /= sum;
    return out;
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
    nn::Tensor logits;
    std::unique_ptr<nn::TensorRecord> record;
    nn::PolicyBatchView view;
};

nn::Result<Fixture> make_fixture(nn::CpuEngine& engine, nn::Tensor logits, nn::Tensor actions,
                                 nn::Tensor mask, nn::Tensor advantage, nn::Tensor returns,
                                 nn::Tensor old_value, nn::Tensor old_log_prob)
{
    (void)engine;
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
    out.logits = std::move(logits);
    out.record = std::make_unique<nn::TensorRecord>(std::move(*record));
    NN_TRY(stable_view, nn::PolicyBatchView::create(*out.record));
    out.view = std::move(*stable_view);
    return out;
}

bool reinforce_loss_unsupported(nn::ComputeEngine& engine, const nn::Tensor& logits);

void test_distribution()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 2, {1, 0, 0, 2, -1, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    nn::CategoricalDistribution dist;
    auto logp = dist.log_prob(*owner, logits, actions);
    CHECK(logp.has_value(), "categorical log_prob");
    if (!logp) return;

    const std::vector<float> logits_v{1, 0, 0, 2, -1, 0};
    const auto p0 = softmax_column(logits_v, 3, 0);
    const auto p1 = softmax_column(logits_v, 3, 1);
    CHECK(near_values(values_of(*owner, *logp), {std::log(p0[0]), std::log(p1[1])}),
          "categorical log_prob hand reference");

    auto entropy = dist.entropy(*owner, logits);
    CHECK(entropy.has_value(), "categorical entropy");
    if (entropy)
    {
        float h0 = 0, h1 = 0;
        for (float p : p0)
            h0 -= p * std::log(p);
        for (float p : p1)
            h1 -= p * std::log(p);
        CHECK(near_values(values_of(*owner, *entropy), {h0, h1}),
              "categorical entropy hand reference");
    }

    auto upstream = matrix_tensor(*owner, 1, 2, {1, 1});
    auto eval = dist.log_prob_with_gradient(*owner, logits, actions, upstream);
    CHECK(eval.has_value(), "log_prob gradient");
    if (eval)
    {
        std::vector<float> want_col0 = p0;
        want_col0[0] = 1 - want_col0[0];
        want_col0[1] = -want_col0[1];
        want_col0[2] = -want_col0[2];
        std::vector<float> want_col1 = p1;
        want_col1[0] = -want_col1[0];
        want_col1[1] = 1 - want_col1[1];
        want_col1[2] = -want_col1[2];
        CHECK(near_values(values_of(*owner, eval->gradient),
                          {want_col0[0], want_col1[0], want_col0[1], want_col1[1], want_col0[2],
                           want_col1[2]}),
              "log_prob gradient is onehot - softmax");
    }

    auto s1 = dist.sample(*owner, logits, 12345);
    auto s2 = dist.sample(*owner, logits, 12345);
    CHECK(s1 && s2, "categorical sampling");
    if (s1 && s2)
    {
        CHECK(values_of(*owner, s1->action) == values_of(*owner, s2->action),
              "same categorical sample seed is reproducible");
        CHECK(near_values(values_of(*owner, s1->log_prob), values_of(*owner, *logp)),
              "sample log_prob matches distribution");
    }

    auto bad_action = matrix_tensor(*owner, 1, 2, {3, 0});
    CHECK(!dist.log_prob(*owner, logits, bad_action), "out-of-range categorical action rejected");
}

void test_reinforce()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 2, {1, 0, 0, 2, -1, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {1, 0.5f});
    auto returns = matrix_tensor(*owner, 1, 2, {2, 3});
    auto old_value = matrix_tensor(*owner, 1, 2, {1, 2.5f});
    auto old_logp = matrix_tensor(*owner, 1, 2, {0, 0});
    auto fixture =
        make_fixture(*owner, logits, actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "REINFORCE fixture");
    if (!fixture) return;

    nn::CategoricalDistribution dist;
    auto lp = dist.log_prob(*owner, fixture->logits, actions);
    CHECK(lp.has_value(), "REINFORCE log prob");
    if (!lp) return;
    const auto lpv = values_of(*owner, *lp);
    const float want_loss = -(lpv[0] + 0.5f * lpv[1]) / 2.0f;

    nn::ReinforceObjective reinforce;
    auto loss = reinforce.forward(*owner, fixture->logits, fixture->view);
    CHECK(loss && std::fabs(*loss - want_loss) < 1e-5f, "REINFORCE loss");
    auto grad = reinforce.backward();
    CHECK(grad.has_value(), "REINFORCE cached backward");

    nn::ReinforceObjective baseline_reinforce(true);
    auto baseline_loss = baseline_reinforce.forward(*owner, fixture->logits, fixture->view);
    CHECK(baseline_loss && std::fabs(*baseline_loss - want_loss) < 1e-5f,
          "REINFORCE old_value baseline equals explicit advantage");
    auto baseline_grad = baseline_reinforce.backward();
    if (grad && baseline_grad)
        check_close(*owner, *baseline_grad, *grad, "REINFORCE baseline gradient", g_fail);

    auto polymorphic = std::make_shared<ForwardingDistribution>();
    nn::ReinforceObjective injected(false, nn::Precision::F32, polymorphic);
    auto injected_loss = injected.forward(*owner, fixture->logits, fixture->view);
    auto injected_grad = injected.backward();
    CHECK(injected_loss && std::fabs(*injected_loss - want_loss) < 1e-5f,
          "objective accepts custom PolicyDistribution interface");
    if (grad && injected_grad)
        check_close(*owner, *injected_grad, *grad, "custom distribution gradient composition",
                    g_fail);

    auto zero_mask = matrix_tensor(*owner, 1, 2, {0, 0});
    auto all_masked = make_fixture(*owner, fixture->logits, actions, zero_mask, advantage, returns,
                                   old_value, old_logp);
    CHECK(all_masked.has_value(), "all-mask fixture");
    if (!all_masked) return;
    nn::ReinforceObjective masked_reinforce;
    auto masked_loss = masked_reinforce.forward(*owner, all_masked->logits, all_masked->view);
    auto masked_grad = masked_reinforce.backward();
    CHECK(masked_loss && *masked_loss == 0, "all-mask REINFORCE loss = 0");
    if (masked_grad)
    {
        bool all_zero = true;
        for (float v : values_of(*owner, *masked_grad))
            all_zero = all_zero && std::fabs(v) < 1e-8f;
        CHECK(all_zero, "all-mask REINFORCE gradient = 0");
    }
}

void test_value_and_kl()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 3, 2, {1, 0, 0, 2, -1, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {0, 0});
    auto returns = matrix_tensor(*owner, 1, 2, {1, 2});
    auto old_value = matrix_tensor(*owner, 1, 2, {0, 0});
    auto old_logp = matrix_tensor(*owner, 1, 2, {0, 0});
    auto fixture =
        make_fixture(*owner, logits, actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "value/KL fixture");
    if (!fixture) return;

    auto value = matrix_tensor(*owner, 1, 2, {0.5f, 1.0f});
    nn::ValueRegressionObjective value_obj(2.0f);
    auto value_loss = value_obj.forward(*owner, value, fixture->view);
    auto value_grad = value_obj.backward();
    CHECK(value_loss && std::fabs(*value_loss - 1.25f) < 1e-5f, "value regression loss");
    CHECK(value_grad && near_values(values_of(*owner, *value_grad), {-1, -2}),
          "value regression gradient");

    nn::SampledKlObjective kl;
    auto kl_loss = kl.forward(*owner, fixture->logits, fixture->view);
    CHECK(kl_loss, "sampled KL objective runs");

    const bool unsupported = reinforce_loss_unsupported(*owner, fixture->logits);
    CHECK(unsupported, "contextual objective rejects target-only Loss entry");
}

bool reinforce_loss_unsupported(nn::ComputeEngine& engine, const nn::Tensor& logits)
{
    nn::ReinforceObjective objective;
    auto target = logits;
    return !objective.forward(engine, logits, target).has_value();
}

void test_entropy_finite_difference()
{
    auto owner = nn::CpuEngine::create();
    const std::vector<float> base_values{1, 0, 0, 2, -1, 0};
    auto logits = matrix_tensor(*owner, 3, 2,
                                {base_values[0], base_values[1], base_values[2], base_values[3],
                                 base_values[4], base_values[5]});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto mask = matrix_tensor(*owner, 1, 2, {1, 1});
    auto advantage = matrix_tensor(*owner, 1, 2, {1, 1});
    auto returns = matrix_tensor(*owner, 1, 2, {1, 1});
    auto old_value = matrix_tensor(*owner, 1, 2, {0, 0});
    auto old_logp = matrix_tensor(*owner, 1, 2, {0, 0});
    auto fixture =
        make_fixture(*owner, logits, actions, mask, advantage, returns, old_value, old_logp);
    CHECK(fixture.has_value(), "entropy finite-difference fixture");
    if (!fixture) return;

    nn::EntropyBonusObjective objective(1.0f);
    auto loss = objective.forward(*owner, fixture->logits, fixture->view);
    auto analytic = objective.backward();
    CHECK(loss && analytic, "entropy objective forward/backward");
    if (!analytic) return;
    const auto analytic_v = values_of(*owner, *analytic);

    constexpr float eps = 1e-3f;
    bool close = true;
    for (std::size_t i = 0; i < base_values.size(); ++i)
    {
        auto values = base_values;
        values[i] += eps;
        auto plus = matrix_tensor(
            *owner, 3, 2, {values[0], values[1], values[2], values[3], values[4], values[5]});
        auto plus_loss = objective.forward(*owner, plus, fixture->view);
        values = base_values;
        values[i] -= eps;
        auto minus = matrix_tensor(
            *owner, 3, 2, {values[0], values[1], values[2], values[3], values[4], values[5]});
        auto minus_loss = objective.forward(*owner, minus, fixture->view);
        if (!plus_loss || !minus_loss)
        {
            close = false;
            break;
        }
        const float numeric = (*plus_loss - *minus_loss) / (2 * eps);
        close = close && std::fabs(numeric - analytic_v[i]) < 2e-3f;
    }
    CHECK(close, "entropy analytic gradient matches finite difference");
}

void test_policy_shape_validation()
{
    auto owner = nn::CpuEngine::create();
    auto logits = matrix_tensor(*owner, 2, 2, {0, 0, 0, 0});
    auto actions = matrix_tensor(*owner, 1, 2, {0, 1});
    auto bad_row = matrix_tensor(*owner, 1, 1, {1});
    auto fixture =
        make_fixture(*owner, logits, actions, bad_row, bad_row, bad_row, bad_row, bad_row);
    CHECK(fixture.has_value(), "shape-validation fixture");
    if (!fixture) return;

    nn::ReinforceObjective reinforce;
    nn::EntropyBonusObjective entropy;
    nn::SampledKlObjective kl;
    nn::ClippedPolicyObjective clipped;
    CHECK(!reinforce.evaluate_context(*owner, fixture->logits, fixture->view),
          "REINFORCE rejects (1,1) advantage/mask against (A,2) logits");
    CHECK(!entropy.evaluate_context(*owner, fixture->logits, fixture->view),
          "entropy rejects mask width mismatch");
    CHECK(!kl.evaluate_context(*owner, fixture->logits, fixture->view),
          "sampled KL rejects old-log-prob width mismatch");
    CHECK(!clipped.evaluate_context(*owner, fixture->logits, fixture->view),
          "PPO rejects policy-batch width mismatch");
}

} // namespace

int main()
{
    std::puts("=== RL-1b categorical policy / objectives ===");
    test_distribution();
    test_reinforce();
    test_value_and_kl();
    test_entropy_finite_difference();
    test_policy_shape_validation();
    if (g_fail == 0) std::puts("All RL policy tests passed.");
    else std::printf("%d RL policy test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
