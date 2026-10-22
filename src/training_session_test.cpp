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

[[nodiscard]] nn::Result<nn::TrainingStepReport>
run_mse_step(nn::TrainingSession& session, const nn::Tensor& x,
             std::size_t normalization_units, bool force_update = false)
{
    nn::TrainingStepOptions opts;
    opts.normalization_units = normalization_units;
    opts.force_update = force_update;
    return session.train_step(
        x, opts,
        [&](nn::Model& model, nn::ComputeEngine& engine, const nn::Tensor& input)
            -> nn::Result<nn::TrainingStepOutput> {
            auto logits_r = model.forward(input);
            NN_TRY_CHECK(logits_r);
            nn::MSELoss loss;
            auto target = make_tensor(engine, logits_r->rows(), logits_r->cols(), 0.f, 0.f);
            auto loss_r = loss.forward(engine, *logits_r, target);
            NN_TRY_CHECK(loss_r);
            auto grad_r = loss.backward();
            NN_TRY_CHECK(grad_r);
            nn::TrainingStepOutput out;
            out.grad = std::move(*grad_r);
            out.gradient_units = logits_r->size();
            return out;
        });
}

void test_accumulation_and_session_state()
{
    // docs/todo/30：Session 内部累积；未到边界不更新，移动后状态仍有效。
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "state: add");
    nn::TrainingConfig cfg;
    cfg.optimizer = "sgd";
    cfg.lr = 0.01f;
    cfg.grad_accum_steps = 2;
    auto s = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(s.has_value(), "state: create");
    if (!s) return;
    CHECK(s->state().optimizer_step == 0, "state: 初始 optimizer_step=0");

    const auto x1 = make_tensor(*engine, 4, 2, 0.1f, 0.05f);
    const auto x2 = make_tensor(*engine, 4, 2, 0.2f, 0.05f);
    auto r1 = run_mse_step(*s, x1, 6);
    CHECK(r1 && !r1->parameters_updated, "accum: 第 1 个 micro 不更新");
    CHECK(s->state().microsteps_in_update == 1, "accum: microstep 计数");

    nn::TrainingSession moved = std::move(*s);
    CHECK(moved.state().microsteps_in_update == 1, "state: Session 移动后累积状态保留");
    auto r2 = run_mse_step(moved, x2, 6);
    CHECK(r2 && r2->parameters_updated, "accum: 第 2 个 micro 完成一次 update");
    CHECK(moved.step_count() == 1, "accum: 只推进一次 optimizer step");
    CHECK(moved.state().microsteps_in_update == 0, "accum: 更新后状态复位");
}

void test_global_batch_normalization()
{
    auto engine = nn::CpuEngine::create();

    auto make_zero_model = [&]() {
        nn::Model model(*engine);
        NN_ASSERT_ALWAYS(model.add<nn::Linear>(1, 1), "norm: add");
        for (auto& p : model.parameters())
            NN_ASSERT_ALWAYS(engine->zero(p.get()), "norm: zero parameter");
        return model;
    };
    auto make_sgd_session = [&](std::size_t accum) {
        nn::TrainingConfig cfg;
        cfg.optimizer = "sgd";
        cfg.lr = 0.1f;
        cfg.weight_decay = 0.0f;
        cfg.grad_accum_steps = accum;
        auto s = nn::TrainingSession::create(make_zero_model(), cfg);
        NN_ASSERT_ALWAYS(s, "norm: create session");
        return std::move(*s);
    };

    nn::TrainingSession global = make_sgd_session(1);
    const auto global_x = make_tensor(*engine, 1, 4, 0.1f, 0.1f);
    auto global_r = run_mse_step(global, global_x, 4);
    CHECK(global_r && global_r->parameters_updated, "norm: global batch update");

    nn::TrainingSession accumulated = make_sgd_session(2);
    const auto micro1 = make_tensor(*engine, 1, 2, 0.1f, 0.1f);
    const auto micro2 = make_tensor(*engine, 1, 2, 0.3f, 0.1f);
    auto m1 = run_mse_step(accumulated, micro1, 4);
    auto m2 = run_mse_step(accumulated, micro2, 4);
    CHECK(m1 && !m1->parameters_updated, "norm: micro1 只累积");
    CHECK(m2 && m2->parameters_updated, "norm: micro2 完成 update");

    const auto global_params = snapshot_params(*engine, global.model());
    const auto accum_params = snapshot_params(*engine, accumulated.model());
    CHECK(global_params.size() == accum_params.size(), "norm: 参数数量一致");
    CHECK(l1_move(global_params, accum_params) < 1e-6,
          "norm: global batch 与 micro 累积更新结果一致");
}

void test_force_update()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "force: add");
    nn::TrainingConfig cfg;
    cfg.optimizer = "sgd";
    cfg.lr = 0.01f;
    cfg.grad_accum_steps = 4;
    auto s = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(s, "force: create");
    if (!s) return;
    const auto x = make_tensor(*engine, 4, 2, 0.1f, 0.05f);
    auto r = run_mse_step(*s, x, 6, /*force_update=*/true);
    CHECK(r && r->parameters_updated, "force: 单个 micro 可强制提交尾部 update");
    CHECK(s->step_count() == 1, "force: step_count 推进一次");
}

void test_before_update_abort()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "abort: add");
    nn::TrainingConfig cfg;
    cfg.optimizer = "sgd";
    cfg.grad_accum_steps = 1;
    auto s = nn::TrainingSession::create(std::move(model), cfg);
    CHECK(s, "abort: create");
    if (!s) return;

    const auto x = make_tensor(*engine, 4, 2, 0.1f, 0.05f);
    nn::TrainingStepOptions opts;
    opts.normalization_units = 6;
    opts.before_update = []() -> nn::Result<bool> { return true; };
    auto r = s->train_step(
        x, std::move(opts),
        [&](nn::Model& m, nn::ComputeEngine& e, const nn::Tensor& input)
            -> nn::Result<nn::TrainingStepOutput> {
            auto logits = m.forward(input);
            NN_TRY_CHECK(logits);
            nn::MSELoss loss;
            auto target = make_tensor(e, logits->rows(), logits->cols(), 0.f, 0.f);
            auto l = loss.forward(e, *logits, target);
            NN_TRY_CHECK(l);
            auto g = loss.backward();
            NN_TRY_CHECK(g);
            nn::TrainingStepOutput out;
            out.grad = std::move(*g);
            out.gradient_units = logits->size();
            return out;
        });
    CHECK(r && r->aborted && !r->parameters_updated, "abort: 回调可在 update 前中止");
    CHECK(s->step_count() == 0, "abort: optimizer step 未推进");
    CHECK(s->state().microsteps_in_update == 0 && !s->state().accumulation_dirty,
          "abort: 累积状态已清理");
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
    test_accumulation_and_session_state();
    test_global_batch_normalization();
    test_force_update();
    test_before_update_abort();
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
