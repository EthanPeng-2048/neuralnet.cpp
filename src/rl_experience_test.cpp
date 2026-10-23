// ── rl_experience_test — experience schema 与 advantage transforms ─────────
//
// docs/todo/31 RL-1a：
//   [1] 标准/自定义字段、必需字段、PolicyBatchView；
//   [2] GAE 手算值、done 截断、bootstrap、mask；
//   [3] MonteCarloReturn；
//   [4] overall/group normalization（总体标准差、零方差）。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <initializer_list>
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
                 const std::vector<float>& want, float tol = 1e-5f)
{
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - want[i]) >
            tol * (1.0f + std::fabs(want[i])))
            return false;
    return true;
}

void test_experience_schema_and_view()
{
    auto schema_r = nn::ExperienceSchema::create(
        {"action", "action_mask"},
        {nn::tensor_field("custom_score")});
    CHECK(schema_r.has_value(), "custom experience schema builds");
    if (!schema_r) return;
    nn::ExperienceSchema schema = std::move(*schema_r);
    CHECK(schema.action().has_value(), "action handle resolved");
    CHECK(schema.action_mask().has_value(), "action mask handle resolved");
    CHECK(schema.group_id().has_value(), "group id standard field registered");
    CHECK(schema.schema().fields().back().name == "custom_score",
          "custom field appended");

    auto owner = nn::CpuEngine::create();
    auto action = make_tensor(*owner, 2, 2, 0.0f, 1.0f);
    auto mask = matrix_tensor(*owner, 2, 2, {1, 1, 1, 0});
    auto custom = make_tensor(*owner, 2, 2, 0.5f, 0.1f);
    auto record_r = nn::TensorRecord::create(
        schema.schema(),
        {nn::NamedTensor{.name = "action", .value = action},
         nn::NamedTensor{.name = "action_mask", .value = mask},
         nn::NamedTensor{.name = "custom_score", .value = custom}});
    CHECK(record_r.has_value(), "experience record builds");
    if (!record_r)
        std::printf("[info] TensorRecord error: %s\n",
                    record_r.error().message.c_str());
    if (!record_r) return;
    auto view = nn::PolicyBatchView::create(*record_r);
    CHECK(view.has_value(), "policy batch view builds");
    if (!view) return;
    CHECK(view->action().size() == 4, "view exposes action");
    CHECK(view->action_mask().size() == 4, "view exposes explicit mask");
    CHECK(view->old_log_prob() == nullptr,
          "view distinguishes absent optional policy fields");
    auto custom_field = record_r->require("custom_score");
    CHECK(custom_field.has_value(), "custom field readable by consumer");

    auto missing = nn::TensorRecord::create(
        schema.schema(),
        {nn::NamedTensor{.name = "action_mask", .value = mask}});
    CHECK(!missing, "required action rejects incomplete record");
    CHECK(!nn::ExperienceSchema::create({"not_a_standard_field"}),
          "unknown required standard field rejected");
}

void test_gae_hand_calculation()
{
    auto owner = nn::CpuEngine::create();
    auto reward = matrix_tensor(*owner, 3, 1, {1, 2, 3});
    auto value = matrix_tensor(*owner, 3, 1, {0.5f, 0.5f, 0.5f});
    auto done = matrix_tensor(*owner, 3, 1, {0, 0, 1});
    auto bootstrap = matrix_tensor(*owner, 1, 1, {0.7f});
    nn::GaeAdvantage gae(nn::AdvantageConfig{.gamma = 0.5f,
                                             .lambda = 0.8f,
                                             .precision = nn::Precision::F32});
    auto out = gae.apply(*owner, reward, value, &done, nullptr, &bootstrap);
    CHECK(out.has_value(), "GAE forward");
    if (!out) return;
    CHECK(near_values(values_of(*owner, out->advantage),
                      {1.85f, 2.75f, 2.5f}),
          "GAE hand-computed advantages");
    CHECK(near_values(values_of(*owner, out->returns),
                      {2.35f, 3.25f, 3.0f}),
          "GAE returns");
}

void test_gae_done_mask_and_bootstrap()
{
    auto owner = nn::CpuEngine::create();
    auto reward = matrix_tensor(*owner, 3, 1, {1, 2, 3});
    auto value = matrix_tensor(*owner, 3, 1, {0, 0, 0});
    auto done = matrix_tensor(*owner, 3, 1, {0, 1, 0});
    auto mask = matrix_tensor(*owner, 3, 1, {1, 0, 1});
    nn::GaeAdvantage gae(nn::AdvantageConfig{.gamma = 1.0f,
                                             .lambda = 1.0f,
                                             .precision = nn::Precision::F32});
    auto out = gae.apply(*owner, reward, value, &done, &mask);
    CHECK(out.has_value(), "GAE done/mask forward");
    if (!out) return;
    CHECK(near_values(values_of(*owner, out->advantage),
                      {1.0f, 0.0f, 3.0f}),
          "done terminates GAE and mask outside steps contributes zero");

    auto reward2 = matrix_tensor(*owner, 2, 1, {1, 1});
    auto value2 = matrix_tensor(*owner, 2, 1, {0, 0});
    auto bootstrap = matrix_tensor(*owner, 1, 1, {2});
    nn::GaeAdvantage gae_lambda1(
        nn::AdvantageConfig{.gamma = 0.5f, .lambda = 1.0f,
                             .precision = nn::Precision::F32});
    auto boot = gae_lambda1.apply(*owner, reward2, value2, nullptr, nullptr,
                                  &bootstrap);
    CHECK(boot.has_value(), "GAE bootstrap forward");
    if (boot)
        CHECK(near_values(values_of(*owner, boot->advantage),
                          {2.0f, 2.0f}),
              "terminal bootstrap enters GAE");
}

void test_returns_and_normalization()
{
    auto owner = nn::CpuEngine::create();
    auto reward = matrix_tensor(*owner, 3, 1, {1, 2, 3});
    nn::MonteCarloReturn mc(0.5f);
    auto returns = mc.apply(*owner, reward);
    CHECK(returns.has_value(), "Monte Carlo return forward");
    if (returns)
        CHECK(near_values(values_of(*owner, *returns),
                          {2.75f, 3.5f, 3.0f}),
              "Monte Carlo discounted returns");

    auto values = matrix_tensor(*owner, 3, 1, {1, 2, 3});
    auto mask = matrix_tensor(*owner, 3, 1, {1, 1, 1});
    nn::MaskedNormalizer normalizer;
    auto norm = normalizer.apply(*owner, values, mask);
    CHECK(norm.has_value(), "advantage normalization");
    if (norm)
    {
        const float s = std::sqrt(2.0f / 3.0f);
        CHECK(near_values(values_of(*owner, *norm),
                          {-1.0f / s, 0.0f, 1.0f / s}),
              "population standard deviation normalization");
    }

    auto constant = matrix_tensor(*owner, 3, 1, {2, 2, 2});
    auto zero_var = normalizer.apply(*owner, constant, mask);
    CHECK(zero_var.has_value(), "zero-variance normalization returns");
    if (zero_var)
        CHECK(near_values(values_of(*owner, *zero_var),
                          {0, 0, 0}),
              "zero variance yields zero, not NaN");
}

void test_group_advantage()
{
    auto owner = nn::CpuEngine::create();
    auto returns = matrix_tensor(*owner, 6, 1,
                                 {1, 2, 3, 10, 10, 10});
    const std::vector<nn::GroupSpan> spans{
        nn::GroupSpan{.first_row = 0, .row_count = 3},
        nn::GroupSpan{.first_row = 3, .row_count = 3},
    };
    nn::GroupAdvantage group;
    auto out = group.apply(*owner, returns, spans);
    CHECK(out.has_value(), "GRPO group normalization");
    if (!out) return;
    const float s = std::sqrt(2.0f / 3.0f);
    CHECK(near_values(values_of(*owner, *out),
                      {-1.0f / s, 0.0f, 1.0f / s, 0, 0, 0}),
          "multiple groups normalize independently; zero-variance group -> 0");

    const std::vector<nn::GroupSpan> bad{
        nn::GroupSpan{.first_row = 0, .row_count = 2},
        nn::GroupSpan{.first_row = 3, .row_count = 1},
    };
    CHECK(!group.apply(*owner, returns, bad).has_value(),
          "non-contiguous group spans rejected");
}

} // namespace

int main()
{
    std::puts("=== RL-1a experience / advantage transforms ===");
    test_experience_schema_and_view();
    test_gae_hand_calculation();
    test_gae_done_mask_and_bootstrap();
    test_returns_and_normalization();
    test_group_advantage();
    if (g_fail == 0) std::puts("All RL experience tests passed.");
    else std::printf("%d RL experience test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
