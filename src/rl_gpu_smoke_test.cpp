// ── rl_gpu_smoke_test — RL 表达式在 Vulkan 闭合世界的端到端冒烟 ──────────
//
// 覆盖 GAE、categorical distribution、PPO policy/value clip 与 GRPO preset。
// 无 Vulkan/设备时返回 77（ctest SKIP）。数值正确性由 CPU 门禁负责；本用例
// 专门抓 AOT 结构漏登记与 GPU 设备域错误。
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

bool finite(nn::ComputeEngine& engine, const nn::Tensor& tensor)
{
    auto values = nn::detail::download_vector(engine, tensor);
    if (!values) return false;
    for (float v : *values)
        if (!std::isfinite(v)) return false;
    return true;
}

void run_smoke(nn::ComputeEngine& engine)
{
    auto reward = matrix_tensor(engine, 2, 2, {1, 2, 3, 4});
    auto done = matrix_tensor(engine, 2, 2, {0, 0, 1, 1});
    auto mask_rc = matrix_tensor(engine, 2, 2, {1, 1, 1, 1});
    nn::GaeAdvantage gae;
    auto gae_out = gae.apply(engine, reward, reward, &done, &mask_rc);
    CHECK(gae_out.has_value(), "GPU GAE");
    if (gae_out)
        CHECK(finite(engine, gae_out->advantage) &&
                  finite(engine, gae_out->returns),
              "GPU GAE finite");

    auto logits = matrix_tensor(engine, 3, 4,
        {1, 0, 0, 0,
         0, 1, 0, 0,
         -1, 0, 1, 0});
    auto actions = matrix_tensor(engine, 1, 4, {0, 1, 2, 0});
    auto mask = matrix_tensor(engine, 1, 4, {1, 1, 1, 1});
    auto advantage = matrix_tensor(engine, 1, 4, {1, -0.5f, 0.2f, -0.1f});
    auto returns = matrix_tensor(engine, 1, 4, {1, 2, 3, 4});
    auto old_value = matrix_tensor(engine, 1, 4, {0.5f, 1, 2, 3});
    auto old_logp = matrix_tensor(engine, 1, 4, {0, 0, 0, 0});
    auto value = matrix_tensor(engine, 1, 4, {0.7f, 1.2f, 2.5f, 3.4f});
    nn::ExperienceSchema exp = NN_UNWRAP(nn::ExperienceSchema::create(
        {"action", "action_mask", "advantage", "return", "old_value",
         "old_log_prob"}));
    nn::TensorRecord record = NN_UNWRAP(nn::TensorRecord::create(
        exp.schema(),
        {nn::NamedTensor{.name = "action", .value = actions},
         nn::NamedTensor{.name = "action_mask", .value = mask},
         nn::NamedTensor{.name = "value_mask", .value = mask},
         nn::NamedTensor{.name = "advantage", .value = advantage},
         nn::NamedTensor{.name = "return", .value = returns},
         nn::NamedTensor{.name = "old_value", .value = old_value},
         nn::NamedTensor{.name = "old_log_prob", .value = old_logp}}));
    nn::PolicyBatchView view = NN_UNWRAP(
        nn::PolicyBatchView::create(record));

    auto ppo_schema = NN_UNWRAP(nn::TensorSchema::create(
        "ppo_output",
        {nn::tensor_field("policy",
                          nn::TensorShapeSpec::exact(3, 4)),
         nn::tensor_field("value",
                          nn::TensorShapeSpec::exact(1, 4))}));
    nn::ClippedPolicyObjective ppo_policy;
    auto policy_out = ppo_policy.evaluate_context(engine, logits, view);
    CHECK(policy_out.has_value(), "GPU PPO policy component");
    if (!policy_out)
        std::printf("[info] GPU PPO policy error: %s\n",
                    policy_out.error().message.c_str());
    nn::EntropyBonusObjective entropy;
    auto entropy_out = entropy.evaluate_context(engine, logits, view);
    CHECK(entropy_out.has_value(), "GPU entropy component");
    if (!entropy_out)
        std::printf("[info] GPU entropy error: %s\n",
                    entropy_out.error().message.c_str());
    nn::ClippedValueObjective ppo_value;
    auto value_out = ppo_value.evaluate_context(engine, value, view);
    CHECK(value_out.has_value(), "GPU PPO value component");
    if (!value_out)
        std::printf("[info] GPU PPO value error: %s\n",
                    value_out.error().message.c_str());
    nn::PpoObjective ppo;
    auto ppo_out = ppo.evaluate(engine, ppo_schema, logits, value, view);
    CHECK(ppo_out.has_value(), "GPU PPO preset");
    if (!ppo_out)
        std::printf("[info] GPU PPO error: %s\n",
                    ppo_out.error().message.c_str());
    if (ppo_out)
    {
        auto ph = ppo_out->gradients.field("policy");
        auto vh = ppo_out->gradients.field("value");
        if (ph && vh)
        {
            auto pg = ppo_out->gradients.require(*ph);
            auto vg = ppo_out->gradients.require(*vh);
            CHECK(pg && vg, "GPU PPO gradients");
            if (pg && vg)
                CHECK(finite(engine, **pg) && finite(engine, **vg),
                      "GPU PPO gradients finite");
        }
    }

    auto grpo_schema = NN_UNWRAP(nn::TensorSchema::create(
        "grpo_output",
        {nn::tensor_field("policy",
                          nn::TensorShapeSpec::exact(3, 4))}));
    const std::vector<nn::GroupSpan> spans{
        nn::GroupSpan{.first_row = 0, .row_count = 2},
        nn::GroupSpan{.first_row = 2, .row_count = 2},
    };
    nn::GrpoObjective grpo;
    auto grpo_out = grpo.evaluate(engine, grpo_schema, logits, view, spans);
    CHECK(grpo_out.has_value(), "GPU GRPO preset");
    if (!grpo_out)
        std::printf("[info] GPU GRPO error: %s\n",
                    grpo_out.error().message.c_str());
    if (grpo_out)
    {
        auto gh = grpo_out->gradients.field("policy");
        if (gh)
        {
            auto grad = grpo_out->gradients.require(*gh);
            CHECK(grad && finite(engine, **grad),
                  "GPU GRPO gradient finite");
        }
    }
}

} // namespace

int main()
{
    std::puts("=== RL Vulkan smoke ===");
#ifdef NN_HAS_VULKAN
    auto backend = nn::GpuBackend::create();
    if (!backend || !backend->initialize())
    {
        std::puts("[SKIP] 无可用 Vulkan 设备");
        return 77;
    }
    auto engine = nn::GpuEngine::create(backend);
    run_smoke(*engine);
    if (g_fail == 0) std::puts("All RL GPU smoke checks passed.");
    else std::printf("%d RL GPU smoke check(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
#else
    std::puts("[SKIP] 构建未启用 Vulkan");
    return 77;
#endif
}
