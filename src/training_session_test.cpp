// ── training_session_test — TrainingSession 生命周期与训练原语 ──────────────
//
// docs/todo/29-session-architecture.md §8 P2a 验收：
//   [1] 创建 → 8 步 step 参数确实移动（优化器与参数连通）
//   [2] Session 移动后 step 正常（TensorRef 指向堆上 Layer 成员，地址稳定）
//   [3] 作用域结束析构顺序正确（optimizer 先于 Model；ASan 下验证）
//   [4] 推理：session.model() 可直接 forward；独立 Model 不依赖 Session
//   [5] 精度配方透传（f16 profile 下 step 正常）
//   [6] 未知优化器 / 未绑定引擎 → Result 错误
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

std::vector<nn::Scalar> snapshot_params(nn::CpuEngine& eng, nn::Model& model)
{
    std::vector<nn::Scalar> out;
    for (auto& t : model.parameters())
    {
        auto m = eng.to_matrix(t.get(), nn::Precision::F32);
        NN_ASSERT_ALWAYS(m, "snapshot_params: to_matrix failed");
        for (auto v : m->span()) out.push_back(v);
    }
    return out;
}

double l1_move(const std::vector<nn::Scalar>& a, const std::vector<nn::Scalar>& b)
{
    double s = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
        s += std::fabs(static_cast<double>(a[i] - b[i]));
    return s;
}

void test_create_and_train()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "add Linear(4,3)");
    CHECK(model.add<nn::ReLU>(), "add ReLU");
    CHECK(model.add<nn::Linear>(3, 2), "add Linear(3,2)");

    nn::TrainingConfig cfg;
    cfg.optimizer = "adam";
    cfg.lr = 0.02f;
    cfg.weight_decay = 0.0f;
    auto session_r = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(session_r.has_value(), "TrainingSession::create");
    if (!session_r) return;
    nn::TrainingSession session = std::move(*session_r);

    CHECK(session.step_count() == 0, "初始 step_count = 0");

    const auto x = make_tensor(*engine, 4, 3, 0.1f, 0.05f);   // (in, batch)
    auto before = snapshot_params(*engine, session.model());
    CHECK(!before.empty(), "参数快照非空");

    for (int step = 0; step < 8; ++step)
    {
        NN_UNWRAP(session.zero_grad());
        auto logits = session.model().forward(x);
        CHECK(logits.has_value(), "forward");
        if (!logits) return;

        nn::MSELoss loss;
        auto target = make_tensor(*engine, logits->rows(), logits->cols(), 0.0f, 0.0f);
        auto l = loss.forward(*engine, *logits, target);
        CHECK(l.has_value(), "loss forward");
        if (!l) return;
        auto grad = loss.backward();
        CHECK(grad.has_value(), "loss backward");
        if (!grad) return;
        auto bw = session.model().backward(*grad);
        CHECK(bw.has_value(), "model backward");
        auto st = session.step();
        CHECK(st.has_value(), "session.step");
    }

    CHECK(session.step_count() == 8, "step_count 累计 8");
    auto after = snapshot_params(*engine, session.model());
    CHECK(after.size() == before.size(), "参数数量不变");
    CHECK(l1_move(before, after) > 1e-6, "8 步后参数确实移动");

    // 推理路径：Session 内的 Model 可直接 forward，不需要 optimizer 参与
    auto infer = session.model().forward(x);
    CHECK(infer.has_value(), "session.model().forward 推理可用");
}

void test_move_session_keeps_refs_valid()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "add Linear(4,3)");
    CHECK(model.add<nn::Linear>(3, 2), "add Linear(3,2)");

    nn::TrainingConfig cfg;
    cfg.optimizer = "sgd";
    cfg.lr = 0.1f;
    auto s1 = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(s1.has_value(), "create");
    if (!s1) return;

    nn::TrainingSession moved = std::move(*s1);
    CHECK(moved.step_count() == 0, "移动后计数器保留");

    const auto x = make_tensor(*engine, 4, 2, 0.2f, 0.1f);
    NN_UNWRAP(moved.zero_grad());
    auto logits = moved.model().forward(x);
    CHECK(logits.has_value(), "移动后 forward");
    if (!logits) return;
    nn::MSELoss loss;
    auto target = make_tensor(*engine, logits->rows(), logits->cols(), 0.f, 0.f);
    auto l = loss.forward(*engine, *logits, target);
    CHECK(l.has_value(), "移动后 loss");
    if (!l) return;
    auto grad = loss.backward();
    CHECK(grad.has_value(), "移动后 loss backward");
    if (!grad) return;
    CHECK(moved.model().backward(*grad).has_value(), "移动后 model backward");
    CHECK(moved.step().has_value(), "移动后 step（TensorRef 仍有效）");
    CHECK(moved.step_count() == 1, "移动后计数器推进");
}

void test_inference_without_session()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(3, 2), "独立 Model::add");
    const auto x = make_tensor(*engine, 3, 2, 0.5f, 0.1f);
    auto y = model.forward(x);
    CHECK(y.has_value(), "独立 Model（无 Session）可 forward");
}

void test_precision_passthrough()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    model.set_default_precision_profile(nn::profile_f16());
    CHECK(model.add<nn::Linear>(4, 3), "f16 Linear");
    CHECK(model.add<nn::Linear>(3, 2), "f16 Linear");

    nn::TrainingConfig cfg;
    cfg.optimizer = "adamw";
    cfg.lr = 0.01f;
    cfg.precision = nn::profile_f16();
    auto session_r = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(session_r.has_value(), "f16 配方 create");
    if (!session_r) return;
    nn::TrainingSession session = std::move(*session_r);

    CHECK(session.config().precision.compute == nn::Precision::F16,
          "config 保留调用方精度配方");
    CHECK(session.model().parameters()[0].get().precision() == nn::Precision::F16,
          "模型参数按 f16 存储");

    const auto x = make_tensor(*engine, 4, 2, 0.1f, 0.05f);
    NN_UNWRAP(session.zero_grad());
    auto logits = session.model().forward(x);
    CHECK(logits.has_value(), "f16 forward");
    if (!logits) return;
    nn::MSELoss loss;
    auto target = make_tensor(*engine, logits->rows(), logits->cols(), 0.f, 0.f);
    auto l = loss.forward(*engine, *logits, target);
    CHECK(l.has_value(), "f16 loss");
    if (!l) return;
    auto grad = loss.backward();
    CHECK(grad.has_value(), "f16 loss backward");
    if (!grad) return;
    CHECK(session.model().backward(*grad).has_value(), "f16 model backward");
    CHECK(session.step().has_value(), "f16 优化器 step");
}

void test_errors()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(3, 2), "add");

    nn::TrainingConfig bad;
    bad.optimizer = "no_such_optimizer";
    auto r1 = nn::TrainingSession::create(std::move(model), bad);
    CHECK(!r1, "未知优化器返回 Result 错误而非 nullptr");

    nn::Model unbound;   // 未绑定引擎
    auto r2 = nn::TrainingSession::create(std::move(unbound));
    CHECK(!r2, "未绑定引擎的 Model 返回 Result 错误");

    // CPU 引擎上的 activation offload 是 no-op（会毒化 backward）→ 配置期拒绝
    nn::Model m2(*engine);
    CHECK(m2.add<nn::Linear>(3, 2), "add");
    nn::TrainingConfig offload_cfg;
    offload_cfg.activation_offload = true;
    auto r3 = nn::TrainingSession::create(std::move(m2), offload_cfg);
    CHECK(!r3, "CPU 引擎 + activation_offload 返回 Result 错误");
}

void test_destroy_order()
{
    // ASan 下验证：Session 作用域结束时 optimizer 先于 Model 析构，
    // 不会在析构/最后一步踩已销毁的 Tensor 成员。
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "add");
    {
        nn::TrainingConfig cfg;
        cfg.optimizer = "adam";
        auto s = nn::TrainingSession::create(std::move(model), cfg);
        CHECK(s.has_value(), "create");
        if (!s) return;
        CHECK(s->step().has_value(), "step");
    }   // s 析构：opt_ 先于 model_
    CHECK(true, "Session 析构无 UAF（ASan 口径）");
}

} // namespace

int main()
{
    std::printf("── TrainingSession：所有权 / 训练原语 / 精度透传 ──\n");
    test_create_and_train();
    test_move_session_keeps_refs_valid();
    test_inference_without_session();
    test_precision_passthrough();
    test_errors();
    test_destroy_order();

    if (g_fail == 0)
        std::printf("training_session_test: ALL PASS\n");
    else
        std::printf("training_session_test: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
