// ── run_driver_test — RFC 32 §5：RunDriver 领域无关训练骨架 ────────────────
//
// do_update 是注入钩子 ⇒ 测试记录 driver 发出的 UpdateRequest 序列，验证：
//   [1] epoch×step 循环：步数、epoch 边界、global_step 递增
//   [2] lr 接线：cosine（epoch 级）与 step_cosine（step 级）逐 lr 正确
//   [3] 断点节奏：epoch 边界 + save_interval 触发 on_checkpoint，计数器归一化
//   [4] --max-steps 提前停（is_final_step 标记 + force 收尾）
//   [5] --stop-after-epoch 存档退出
//   [6] 探针中止 → abort_code=9，停止推进
//   [7] 步进驱动（steps_per_epoch=nullopt）：跑满 max_steps
//   [8] 必选钩子缺失 / 参数校验 → create/fit 报错
//   [9] 断点续训起点：start_epoch/start_step 传入 do_update，rewind 被调用
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/batch_source.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/run_driver.hpp>
#include <neuralnet.cpp/training_session.hpp>

static int g_fail = 0;
#define CHECK(cond, msg)                                                    \
    do                                                                      \
    {                                                                       \
        if (!(cond))                                                        \
        {                                                                   \
            std::cout << "    [FAIL] " << (msg) << "  (" << #cond << ")\n"; \
            ++g_fail;                                                       \
        }                                                                   \
    } while (0)

namespace
{

// ── 最小域：一个 4→3→2 的 MLP Session ──────────────────────────────────
struct Fixture
{
    std::shared_ptr<nn::CpuEngine> engine;   // CpuEngine::create() 返回 shared_ptr
    std::unique_ptr<nn::TrainingSession> session;

    static Fixture make()
    {
        Fixture f;
        f.engine = nn::CpuEngine::create();
        nn::Model model(*f.engine);
        (void)model.add<nn::Linear>(4, 3);
        (void)model.add<nn::ReLU>();
        (void)model.add<nn::Linear>(3, 2);
        nn::TrainingConfig cfg;
        cfg.lr = 0.01f;
        auto s = nn::TrainingSession::create(std::move(model), cfg);
        if (s) f.session = std::make_unique<nn::TrainingSession>(std::move(*s));
        return f;
    }
};

// 记录 driver 发出的请求；模拟一个"域"
struct Recorder
{
    std::vector<nn::UpdateRequest> reqs;
    std::size_t ckpt_calls = 0;
    std::vector<nn::TrainCounters> ckpt_counters;   // 断点时刻的计数器快照
    std::size_t eval_calls = 0;
    std::size_t epoch_end_calls = 0;
    int abort_at = -1;              // >=0：第 N 次 do_update 返回 aborted
    bool fail_ckpt = false;         // on_checkpoint 返回错误
    std::size_t samples_per_step = 4;
    std::size_t do_update_calls = 0;
    std::vector<std::size_t> rewind_epochs;   // 记录 rewind(epoch)

    nn::RunHooks make_hooks()
    {
        nn::RunHooks h;
        h.do_update = [this](nn::TrainingSession&,
                             const nn::UpdateRequest& r) -> nn::Result<nn::UpdateOutcome>
        {
            reqs.push_back(r);
            const std::size_t call = do_update_calls++;
            nn::UpdateOutcome o;
            o.parameters_updated = true;
            o.samples = samples_per_step;
            o.loss = 1.0f;
            o.has_loss = true;
            if (abort_at >= 0 && static_cast<int>(call) == abort_at)
            {
                o.aborted = true;
                o.parameters_updated = false;
            }
            return o;
        };
        h.on_checkpoint = [this](nn::TrainingSession&, const nn::TrainState& st) -> nn::Result<void>
        {
            if (fail_ckpt) NN_FAIL("模拟断点失败");
            ++ckpt_calls;
            ckpt_counters.push_back(st.counters);
            return {};
        };
        h.evaluate = [this](nn::TrainingSession&, std::size_t) -> nn::Result<nn::Scalar>
        {
            ++eval_calls;
            return nn::Scalar{0.5f};
        };
        h.on_epoch_end = [this](const nn::RunReport&, std::size_t) -> nn::Result<void>
        {
            ++epoch_end_calls;
            return {};
        };
        return h;
    }
};

// 有固定 epoch 的源：n_samples / batch → steps_per_epoch
nn::IndexedBatchSource make_source(std::size_t n, std::size_t bs, std::uint64_t seed)
{
    auto r = nn::IndexedBatchSource::create(
        n, bs, seed,
        [](std::span<const std::size_t> ix) -> nn::Result<std::pair<std::any, std::size_t>>
        { return std::make_pair(std::any(), ix.size()); });
    // create 对合法参数必成功
    if (!r)
    {
        std::cout << "    [FATAL] 源创建失败\n";
        std::abort();
    }
    return std::move(*r);
}

// 无固定 epoch 的 Pull 源（步进驱动）
class StepOnlySource : public nn::BatchSource
{
public:
    [[nodiscard]] nn::Result<nn::BatchAt> next(std::size_t) override
    {
        nn::BatchAt b;
        b.samples = 4;
        b.units = 4;
        b.epoch_done = false;
        return b;
    }
    [[nodiscard]] nn::Result<void> rewind(std::size_t, std::size_t) override { return {}; }
};

// 空步钩子（0 步运行时不会被调用，但 fit 要求存在）
static nn::RunHooks rec_empty_hooks()
{
    nn::RunHooks h;
    h.do_update = [](nn::TrainingSession&,
                     const nn::UpdateRequest&) -> nn::Result<nn::UpdateOutcome>
    { return nn::UpdateOutcome{}; };
    return h;
}

// ── [1] epoch×step 循环 ────────────────────────────────────────────────
void test_epoch_loop()
{
    std::cout << "\n── [1] epoch×step 循环：步数 / global_step 递增 ──\n";
    auto fx = Fixture::make();
    CHECK(fx.session != nullptr, "建 Session");
    if (!fx.session) return;

    auto src = make_source(/*n=*/10, /*bs=*/4, /*seed=*/1);   // 3 步/epoch
    nn::RunDriverConfig cfg;
    cfg.epochs = 2;
    auto rec = Recorder{};

    auto drv = nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
    CHECK(drv.has_value(), "RunDriver::create");
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());

    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit 成功");
    if (!rep) return;
    CHECK(rep->steps_executed == 6, "2 epoch × 3 步 = 6 步");
    CHECK(rec.reqs.size() == 6, "do_update 被调 6 次");
    CHECK(rep->samples_executed == 6 * 4, "样本数 = 步数 × 4");
    // global_step 连续递增 0..5
    bool seq_ok = true;
    for (std::size_t i = 0; i < rec.reqs.size(); ++i)
        if (rec.reqs[i].global_step != i) seq_ok = false;
    CHECK(seq_ok, "global_step 连续 0..5");
    // epoch/step 网格
    CHECK(rec.reqs[0].epoch == 0 && rec.reqs[0].step == 0, "首步 (0,0)");
    CHECK(rec.reqs[3].epoch == 1 && rec.reqs[3].step == 0, "第 4 步 (1,0)");
    CHECK(rep->final_epoch == 2 && rep->final_step_in_epoch == 0, "正常跑完 = (epochs,0)");
    CHECK(rep->abort_code == 0, "无中止");
}

// ── [2] lr 接线 ────────────────────────────────────────────────────────
void test_lr_wiring()
{
    std::cout << "\n── [2] lr 接线：cosine（epoch 级）/ step_cosine（step 级）──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 2);   // 3 步/epoch

    // cosine：epoch 0/1/2 的 lr 单调不增（base→min）
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 4;
        cfg.lr_schedule = "cosine";
        cfg.lr = 0.1f;
        cfg.min_lr = 0.01f;
        auto rec = Recorder{};
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv)
        {
            drv->set_hooks(rec.make_hooks());
            (void)drv->fit();
            CHECK(rec.reqs.size() == 12, "4 epoch × 3 步");
            // 同一 epoch 内 lr 恒定；epoch 间递减（warmup=0）
            const bool same_within =
                rec.reqs[0].lr == rec.reqs[1].lr && rec.reqs[1].lr == rec.reqs[2].lr;
            CHECK(same_within, "epoch 0 内 lr 恒定");
            CHECK(rec.reqs[0].lr > rec.reqs[3].lr && rec.reqs[3].lr > rec.reqs[9].lr,
                  "cosine 跨 epoch 递减");
        }
    }

    // step_cosine：step 级接管，每步可变
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 2;
        cfg.lr_schedule = "step_cosine";
        cfg.lr = 0.1f;
        cfg.min_lr = 0.01f;
        auto rec = Recorder{};
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv)
        {
            drv->set_hooks(rec.make_hooks());
            (void)drv->fit();
            CHECK(rec.reqs.size() == 6, "step_cosine 6 步");
            // step 级：同一 epoch 内 lr 不再恒定（cosine 随步下降）
            CHECK(rec.reqs[0].lr > rec.reqs[1].lr, "step_cosine 逐步下降");
        }
    }
}

// ── [3] 断点节奏 ───────────────────────────────────────────────────────
void test_checkpoint_cadence()
{
    std::cout << "\n── [3] 断点节奏：epoch 边界 + save_interval + 计数器归一化 ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 3);   // 3 步/epoch

    // 仅 epoch 边界（save_interval=0）：2 epoch → 2 次
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 2;
        auto rec = Recorder{};
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv)
        {
            drv->set_hooks(rec.make_hooks());
            (void)drv->fit();
            CHECK(rec.ckpt_calls == 2, "2 epoch → 2 次 epoch 边界断点");
            CHECK(!rec.ckpt_counters.empty() && rec.ckpt_counters.back().epoch == 2,
                  "末次断点计数器 epoch = 2（下一步位置）");
            CHECK(rec.ckpt_counters.back().global_step == 6,
                  "断点 global_step = 绝对步号 2×3+0 = 6（跨续训累计）");
            CHECK(rec.epoch_end_calls == 2, "on_epoch_end 每 epoch 一次");
            CHECK(rec.eval_calls == 2, "evaluate 每 epoch 一次");
        }
    }

    // save_interval=1：每步都断点 → 6 步 + epoch 边界
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 2;
        cfg.save_interval = 1;
        auto rec = Recorder{};
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv)
        {
            drv->set_hooks(rec.make_hooks());
            (void)drv->fit();
            CHECK(rec.ckpt_calls >= 6, "save_interval=1 → 每步断点");
        }
    }

    // 断点失败计数（不中止训练）
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 2;
        auto rec = Recorder{};
        rec.fail_ckpt = true;
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv)
        {
            drv->set_hooks(rec.make_hooks());
            auto rep = drv->fit();
            CHECK(rep && rep->ckpt_failures == 2, "断点失败计 2 次且训练继续");
            CHECK(rep && rep->steps_executed == 6, "断点失败不中止训练");
        }
    }
}

// ── [4] --max-steps 提前停 ─────────────────────────────────────────────
void test_max_steps()
{
    std::cout << "\n── [4] --max-steps 提前停（is_final_step 收尾）──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 4);

    nn::RunDriverConfig cfg;
    cfg.epochs = 10;
    cfg.max_steps = 4;
    auto rec = Recorder{};
    auto drv = nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
    CHECK(drv.has_value(), "epochs>max_steps 时 create 仍合法");
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());
    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit");
    if (!rep) return;
    CHECK(rep->steps_executed == 4, "只执行 4 步");
    CHECK(rec.reqs.size() == 4, "do_update 4 次");
    CHECK(rec.reqs.back().is_final_step, "最后一步 is_final_step=true");
    CHECK(!rec.reqs[0].is_final_step, "首步 is_final_step=false");
}

// ── [5] --stop-after-epoch ─────────────────────────────────────────────
void test_stop_after_epoch()
{
    std::cout << "\n── [5] --stop-after-epoch 存档退出 ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 5);

    nn::RunDriverConfig cfg;
    cfg.epochs = 5;
    cfg.stop_after_epoch = 2;   // 训到 epoch 2（1-based，不含）→ 跑 2 个 epoch
    auto rec = Recorder{};
    auto drv = nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());
    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit");
    if (!rep) return;
    CHECK(rep->steps_executed == 6, "2 epoch × 3 步 = 6 步");
    CHECK(rep->final_epoch == 2, "停在 epoch 2");
}

// ── [6] 探针中止 ───────────────────────────────────────────────────────
void test_abort()
{
    std::cout << "\n── [6] 探针中止 → abort_code=9 ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 6);

    nn::RunDriverConfig cfg;
    cfg.epochs = 3;
    auto rec = Recorder{};
    rec.abort_at = 4;   // 第 5 次 do_update 中止
    auto drv = nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());
    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit 返回报告（非错误）");
    if (!rep) return;
    CHECK(rep->abort_code == 9, "abort_code = 9");
    CHECK(rep->steps_executed == 5, "中止后不再推进（4 成功 + 1 中止）");
}

// ── [7] 步进驱动（无固定 epoch）────────────────────────────────────────
void test_step_driven()
{
    std::cout << "\n── [7] 步进驱动：steps_per_epoch=nullopt → 跑满 max_steps ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    StepOnlySource src;

    nn::RunDriverConfig cfg;
    cfg.epochs = 0;          // 步进驱动
    cfg.max_steps = 7;
    auto rec = Recorder{};
    auto drv = nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
    CHECK(drv.has_value(), "epochs=0 + max_steps → create 合法");
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());
    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit");
    if (!rep) return;
    CHECK(rep->steps_executed == 7, "跑满 7 步");
    CHECK(rec.reqs.size() == 7, "do_update 7 次");
    bool spe0 = true;
    for (auto& r : rec.reqs)
        if (r.steps_per_epoch != 0) spe0 = false;
    CHECK(spe0, "步进驱动 steps_per_epoch=0");
    CHECK(rec.reqs.back().is_final_step, "末步 is_final_step=true");
}

// ── [8] 参数与必选钩子校验 ─────────────────────────────────────────────
void test_validation()
{
    std::cout << "\n── [8] create/fit 校验 ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 7);

    CHECK(!nn::RunDriver::create(*fx.session, src, nullptr, {}).has_value(), "state 为空 → 报错");
    {
        nn::RunDriverConfig bad;
        bad.epochs = -1;
        CHECK(!nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), bad)
                   .has_value(),
              "epochs=-1 → 报错");
    }
    {
        // epochs=0 + max_steps=0 = 合法的 0 步运行（TextTrainer --epochs 0 语义）
        nn::RunDriverConfig zero;
        zero.epochs = 0;
        zero.max_steps = 0;
        auto z =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), zero);
        CHECK(z.has_value(), "epochs=0 + max_steps=0 → 0 步运行合法");
        if (z)
        {
            z->set_hooks(rec_empty_hooks());
            auto zr = z->fit();
            CHECK(zr && zr->steps_executed == 0, "0 步运行 → steps_executed=0");
        }
    }
    {
        nn::RunDriverConfig bad;
        bad.epochs = 2;
        bad.stop_after_epoch = 5;
        CHECK(!nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), bad)
                   .has_value(),
              "stop_after_epoch > epochs → 报错");
    }
    // fit 缺 do_update → 报错
    {
        nn::RunDriverConfig cfg;
        cfg.epochs = 1;
        auto drv =
            nn::RunDriver::create(*fx.session, src, std::make_unique<nn::TextTrainState>(), cfg);
        if (drv) CHECK(!drv->fit().has_value(), "缺 do_update → fit 报错");
    }
}

// ── [9] 断点续训起点 ───────────────────────────────────────────────────
void test_resume_start()
{
    std::cout << "\n── [9] 断点续训：start_epoch/start_step 传入 + rewind ──\n";
    auto fx = Fixture::make();
    if (!fx.session) return;
    auto src = make_source(10, 4, 8);   // 3 步/epoch

    auto st = std::make_unique<nn::TextTrainState>();
    st->counters.epoch = 1;          // 续训起点 = epoch 1
    st->counters.step_in_epoch = 1;  // 本 epoch 已完成 1 步

    nn::RunDriverConfig cfg;
    cfg.epochs = 3;
    auto rec = Recorder{};
    auto drv = nn::RunDriver::create(*fx.session, src, std::move(st), cfg);
    if (!drv) return;
    drv->set_hooks(rec.make_hooks());
    auto rep = drv->fit();
    CHECK(rep.has_value(), "fit");
    if (!rep) return;
    // epoch 1 从 step 1 起（2 步）+ epoch 2（3 步）= 5 步
    CHECK(rep->steps_executed == 5, "续训从 (1,1) 起 = 5 步");
    CHECK(!rec.reqs.empty() && rec.reqs[0].epoch == 1 && rec.reqs[0].step == 1,
          "首步 = (epoch 1, step 1)");
    CHECK(rec.reqs[0].global_step == 4,
          "续训首步 global_step = 绝对步号 1×3+1 = 4（非进程本地 0）");
}

} // namespace

int main()
{
    test_epoch_loop();
    test_lr_wiring();
    test_checkpoint_cadence();
    test_max_steps();
    test_stop_after_epoch();
    test_abort();
    test_step_driven();
    test_validation();
    test_resume_start();

    if (g_fail == 0)
    {
        std::cout << "\nrun_driver_test: ALL PASS\n";
        return 0;
    }
    std::cout << "\nrun_driver_test: " << g_fail << " FAILURE(S)\n";
    return 1;
}
