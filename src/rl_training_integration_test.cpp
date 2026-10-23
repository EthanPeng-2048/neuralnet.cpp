// ── rl_training_integration_test — named RL objective → Session optimizer ──
//
// 验证设计硬连接：OutputBundleLayer + PpoObjective 产生 GradientBundle，经
// TrainingSession::train_named_step 完成梯度累积/optimizer step，共享 trunk
// 与 policy/value 分支参数实际移动。
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

double l1_move(const std::vector<float>& a, const std::vector<float>& b)
{
    double total = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
        total += std::fabs(static_cast<double>(a[i] - b[i]));
    return total;
}

nn::TensorSchema policy_value_schema()
{
    auto r = nn::TensorSchema::create(
        "rl_policy_value",
        {nn::tensor_field("policy",
                          nn::TensorShapeSpec{.rows = 3,
                                              .cols = std::nullopt}),
         nn::tensor_field("value",
                          nn::TensorShapeSpec{.rows = 1,
                                              .cols = std::nullopt})});
    NN_ASSERT_ALWAYS(r, "policy_value schema");
    return std::move(*r);
}

nn::ExperienceSchema experience_schema()
{
    auto r = nn::ExperienceSchema::create(
        {"action", "action_mask", "advantage", "return", "old_value",
         "old_log_prob"});
    NN_ASSERT_ALWAYS(r, "experience schema");
    return std::move(*r);
}

void test_named_rl_session_update()
{
    auto engine = nn::CpuEngine::create();
    const nn::Tensor input = make_tensor(*engine, 2, 2, 0.2f, 0.1f);
    auto output_schema = policy_value_schema();

    nn::Model model(*engine);
    NN_UNWRAP(model.add<nn::Linear>(2, 2));
    auto bundle = std::make_unique<nn::OutputBundleLayer>(output_schema);
    NN_UNWRAP(bundle->add_port(
        "policy", std::make_unique<nn::Linear>(2, 3)));
    NN_UNWRAP(bundle->add_port(
        "value", std::make_unique<nn::Linear>(2, 1)));
    NN_UNWRAP(model.add_layer(std::move(bundle)));

    auto before = model.forward_named(input);
    CHECK(before.has_value(), "pair model named forward");
    if (!before) return;
    auto before_policy_ptr = before->require("policy");
    CHECK(before_policy_ptr.has_value(), "initial policy output");
    if (!before_policy_ptr) return;
    const auto initial_policy = values_of(*engine, **before_policy_ptr);

    nn::TrainingConfig cfg;
    cfg.optimizer = "sgd";
    cfg.lr = 0.05f;
    cfg.weight_decay = 0.0f;
    cfg.grad_accum_steps = 2;
    auto session_r = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(session_r.has_value(), "create named-output TrainingSession");
    if (!session_r) return;
    nn::TrainingSession session = std::move(*session_r);

    const nn::ExperienceSchema exp = experience_schema();
    nn::PpoConfig ppo_cfg;
    ppo_cfg.value_coefficient = 0.5f;
    ppo_cfg.entropy_coefficient = 0.01f;
    bool losses_finite = true;

    for (std::size_t step = 0; step < 6; ++step)
    {
        auto rollout = session.model().forward_named(input);
        CHECK(rollout.has_value(), "rollout named forward");
        if (!rollout) return;
        auto policy_h = rollout->field("policy");
        auto value_h = rollout->field("value");
        if (!policy_h || !value_h)
        {
            CHECK(false, "rollout ports");
            return;
        }
        auto policy = rollout->require(*policy_h);
        auto value = rollout->require(*value_h);
        if (!policy || !value)
        {
            CHECK(false, "rollout values");
            return;
        }

        nn::CategoricalDistribution distribution;
        auto sample = distribution.sample(*engine, **policy, 9000 + step);
        CHECK(sample.has_value(), "rollout action sample");
        if (!sample) return;

        auto mask = matrix_tensor(*engine, 1, 2, {1, 1});
        auto advantage = matrix_tensor(*engine, 1, 2,
            {1.0f, -0.5f});
        auto returns = matrix_tensor(*engine, 1, 2,
            {1.0f, -0.3f});
        auto record = nn::TensorRecord::create(
            exp.schema(),
            {nn::NamedTensor{.name = "action", .value = sample->action},
             nn::NamedTensor{.name = "action_mask", .value = mask},
             nn::NamedTensor{.name = "value_mask", .value = mask},
             nn::NamedTensor{.name = "advantage", .value = advantage},
             nn::NamedTensor{.name = "return", .value = returns},
             nn::NamedTensor{.name = "old_value", .value = **value},
             nn::NamedTensor{.name = "old_log_prob",
                             .value = sample->log_prob}});
        CHECK(record.has_value(), "rollout experience record");
        if (!record) return;
        auto view = nn::PolicyBatchView::create(*record);
        CHECK(view.has_value(), "rollout policy view");
        if (!view) return;

        nn::TrainingStepOptions options;
        // 每个 micro-batch 有 2 个有效动作，整个 update 有 4 个。
        options.normalization_units = 4;
        auto report = session.train_named_step(
            input, options,
            [&](nn::Model& active, nn::ComputeEngine& engine_ref,
                const nn::Tensor& batch_input)
                -> nn::Result<nn::NamedTrainingStepOutput>
            {
                NN_TRY(outputs, active.forward_named(batch_input));
                NN_TRY(policy_port, outputs->require("policy"));
                NN_TRY(value_port, outputs->require("value"));
                nn::PpoObjective ppo(ppo_cfg);
                NN_TRY(ppo_report,
                       ppo.evaluate(engine_ref, output_schema,
                                    **policy_port, **value_port, *view));
                losses_finite = losses_finite &&
                                std::isfinite(ppo_report->loss);
                return nn::NamedTrainingStepOutput{
                    .gradients = std::move(ppo_report->gradients),
                    .gradient_units = 2,
                    .loss_sum = nn::Tensor{},
                    .loss_units = 0,
                };
            });
        const bool expect_update = (step % 2) == 1;
        CHECK(report && report->parameters_updated == expect_update,
              expect_update ? "named PPO accumulation boundary updates optimizer"
                            : "named PPO first micro-step accumulates only");
        if (!report) return;
    }

    CHECK(losses_finite, "PPO losses finite across named training loop");
    CHECK(session.step_count() == 3,
          "named training loop advances optimizer once per two micro-steps");
    auto after = session.model().forward_named(input);
    CHECK(after.has_value(), "post-training named forward");
    if (!after) return;
    auto after_policy_ptr = after->require("policy");
    CHECK(after_policy_ptr.has_value(), "post-training policy output");
    if (!after_policy_ptr) return;
    const auto final_policy = values_of(*engine, **after_policy_ptr);
    CHECK(l1_move(initial_policy, final_policy) > 1e-5,
          "policy/value model parameters move through named RL gradients");
}

} // namespace

int main()
{
    std::puts("=== RL named objective -> TrainingSession integration ===");
    test_named_rl_session_update();
    if (g_fail == 0) std::puts("All RL training integration tests passed.");
    else std::printf("%d RL integration test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
