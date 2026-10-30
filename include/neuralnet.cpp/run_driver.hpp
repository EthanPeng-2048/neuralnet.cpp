#pragma once

// ── run_driver.hpp — 领域无关训练骨架（RFC 32 §5；issue #39）─────────────
//
// 补上 `TrainingSession`（单步，docs/todo/30）与 `TextTrainer`（文本域驱动）
// 之间的**空洞层**：epoch/step 循环、lr 调度接线、断点节奏、报告聚合。
//
// 分层（不变 vs 可变）：
//   · **RunDriver（本头）拥有"不变"**：外层循环、`compute_epoch_lr`/`compute_step_lr`
//     接线、断点触发时机（epoch 边界 + save_interval）、计数器归一化、报告；
//   · **域拥有"可变"**：通过 `RunHooks::do_update` 注入"取一批 + micro 切分 +
//     `TrainingSession::train_step`"（batch 类型是域定的 —— `train_step` 是模板）。
//
// 必选核心 vs 可选能力（自由度不损失，RFC §5.1）：
//   · 必选：epoch/step 计数、断点时机、lr 接线、报告聚合 —— 保证库口径统一；
//   · 可选（默认开，可关/可空）：评估 `evaluate`、指标 `on_epoch_end`、
//     断点 `on_checkpoint`、相位 `on_phase` —— 空 = 该能力不启用。
//
// 用法（域只写 do_update，不碰外层循环）：
//     auto driver = NN_UNWRAP(nn::RunDriver::create(session, source, state, cfg));
//     driver.set_hooks({.do_update = [&](nn::TrainingSession& s, const nn::UpdateRequest& r) {
//         ... 取批 / micro 切分 / s.train_step(...) ...
//         return nn::UpdateOutcome{...};
//     }});
//     auto report = NN_UNWRAP(driver.fit());
// ─────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_errors.hpp>      // nn::Result / NN_FAIL / NN_TRY
#include "batch_source.hpp"                   // nn::BatchSource / BatchAt
#include "compute_engine.hpp"                 // nn::ComputeEngine
#include "lr_scheduler.hpp"                   // compute_epoch_lr / compute_step_lr
#include "train_state.hpp"                    // nn::TrainState
#include "training_session.hpp"               // nn::TrainingSession

namespace nn
{

// ── 一次 update 的请求（driver → 域；"何时 + 什么 lr"由 driver 决定）────────
struct UpdateRequest
{
    std::size_t epoch = 0;              // 当前 epoch（步进驱动 = 0）
    std::size_t step = 0;               // 本 epoch 内步号
    std::uint64_t global_step = 0;      // 绝对步号 = epoch·steps_per_epoch + step
                                        //（跨续训累计，与 step 级调度同定义）
    Scalar lr = 0;                      // 本步学习率（driver 按 schedule 算好）
    bool is_final_step = false;         // 达到 max_steps：域需 force_update 收尾
    std::size_t steps_per_epoch = 0;    // 0 = 无固定 epoch（步进驱动）
};

// ── 一次 update 的结果（域 → driver；"是否更新/是否中止/消费了多少"）────────
struct UpdateOutcome
{
    bool parameters_updated = false;
    bool aborted = false;               // 探针要求中止（exit 9 语义）
    std::size_t samples = 0;            // 本步消费样本数（末批可能不满）
    Scalar loss = 0;                    // 本步 loss（域已回读后给出）
    bool has_loss = false;              // false = 本步无 loss（不计入均值）
};

// ── 训练报告（领域无关；退出码/统计由调用方消费）──────────────────────────
struct RunReport
{
    Scalar seconds = 0;                 // 循环墙钟（秒）
    std::uint64_t elapsed_ns = 0;       // 同一区间的纳秒数
    std::size_t steps_executed = 0;     // 本进程执行的训练步数
    std::size_t samples_executed = 0;   // 本进程实际消费的样本数
    std::size_t ckpt_failures = 0;      // 断点保存失败次数
    int abort_code = 0;                 // 非 0 = 被探针中止（9）
    Scalar final_lr = 0;
    std::size_t final_epoch = 0;        // 停止时的 epoch（正常跑完 = epochs）
    std::size_t final_step_in_epoch = 0;// 停止时的 epoch 内步号
    Scalar avg_loss = 0;                // 最后一个完整 epoch 的平均 loss（域提供时）
};

// ── 领域注入点（do_update 必选；其余可空 = 该能力不启用）───────────────────
struct RunHooks
{
    // 必选：执行一步 optimizer update（取批 + micro 切分 + train_step）。
    // 域在这里拥有 batch 类型与 loss 回读时机；driver 只给 lr/步号/final 标记。
    std::function<Result<UpdateOutcome>(TrainingSession&, const UpdateRequest&)> do_update;

    // 可选：一个 epoch 结束时的评估（返回标量指标，如 test loss）。
    std::function<Result<Scalar>(TrainingSession&, std::size_t epoch)> evaluate;
    // 可选：一个 epoch 结束（指标/落盘）。返回错误 = 中止训练并传播
    // （epoch 末 drain 等失败必须能终止，与旧 TextTrainer 口径一致）。
    std::function<Result<void>(const RunReport&, std::size_t epoch)> on_epoch_end;
    // 可选：断点请求（"何时/状态是什么"由 driver 决定；路径与格式由域落地）。
    // 返回错误 = 本次断点未落盘（driver 计数，训练继续）。
    std::function<Result<void>(TrainingSession&, const TrainState&)> on_checkpoint;
    // 可选：显存/阶段标记探针。
    std::function<void(const char*)> on_phase;
};

// ── 骨架配置（领域无关的运行配方）─────────────────────────────────────────
struct RunDriverConfig
{
    int epochs = 1;                     // <=0 = 步进驱动（按 max_steps 推进；0 = 0 步）
    std::size_t max_steps = 0;          // 0 = 按 epoch 跑完
    int stop_after_epoch = 0;           // >0 = 训到该 epoch（1-based，不含）存档退出

    // lr 调度（接线 compute_epoch_lr / compute_step_lr）
    std::string lr_schedule = "fixed";  // fixed/cosine/multistep/step_cosine
    Scalar lr = 0.001f;
    Scalar min_lr = 0;
    int warmup_epochs = 0;
    std::size_t warmup_steps = 0;
    std::vector<Scalar> lr_per_epoch;   // 逃生舱：非空则覆盖自动调度

    // 断点节奏
    std::size_t save_interval = 0;      // 0 = 仅 epoch 边界断点

    // 可选能力开关（默认全开；关 = 高自由度/简单域）
    bool enable_checkpoint = true;
    bool enable_metrics = true;
    bool enable_eval = true;
};

// ═══════════════════════════════════════════════════════════════════════
//  RunDriver：领域无关训练骨架
// ═══════════════════════════════════════════════════════════════════════
class RunDriver
{
public:
    [[nodiscard]] static Result<RunDriver> create(TrainingSession& session,
                                                  BatchSource& source,
                                                  std::unique_ptr<TrainState> state,
                                                  RunDriverConfig cfg)
    {
        if (!state)
            NN_FAIL("RunDriver::create: state 不能为空（域需提供其 TrainState 子类）");
        if (cfg.epochs < 0)
            NN_FAIL("RunDriver::create: epochs 不能为负");
        if (cfg.stop_after_epoch < 0)
            NN_FAIL("RunDriver::create: stop_after_epoch 不能为负");
        if (cfg.stop_after_epoch > cfg.epochs && cfg.epochs > 0)
            NN_FAIL("RunDriver::create: stop_after_epoch 超出 epochs");
        return RunDriver(session, source, std::move(state), std::move(cfg));
    }

    // 注入钩子（生命周期：活到下一次 fit 返回；fit 结束会清空）。
    void set_hooks(RunHooks hooks) { hooks_ = std::move(hooks); }

    // 驱动：epoch×step 循环 + lr 接线 + 断点节奏 + 报告。
    [[nodiscard]] Result<RunReport> fit()
    {
        if (!hooks_.do_update)
            NN_FAIL("RunDriver::fit: do_update 钩子必选（域的单步更新逻辑）");

        const auto t0 = std::chrono::steady_clock::now();
        RunReport report;

        const auto spe = source_.steps_per_epoch();   // nullopt = 步进驱动
        const bool epoch_mode = spe.has_value() && cfg_.epochs > 0;
        const std::size_t steps_per_epoch = spe.value_or(0);

        // 起点（断点续训 = 计数器"下一步的位置"）
        const std::size_t start_epoch = static_cast<std::size_t>(state_->counters.epoch);
        const std::size_t start_step =
            static_cast<std::size_t>(state_->counters.step_in_epoch);
        if (epoch_mode && start_epoch > static_cast<std::size_t>(cfg_.epochs))
            NN_FAIL("RunDriver::fit: 断点 epoch 超出本次配方 epochs");

        // lr 调度配置（epoch 级）
        LrScheduleConfig epoch_lr_cfg;
        epoch_lr_cfg.base_lr = cfg_.lr;
        epoch_lr_cfg.min_lr = cfg_.min_lr;
        epoch_lr_cfg.warmup_epochs = cfg_.warmup_epochs;
        epoch_lr_cfg.total_epochs = (cfg_.epochs > 0) ? cfg_.epochs : 1;
        epoch_lr_cfg.schedule = cfg_.lr_schedule;
        for (Scalar v : cfg_.lr_per_epoch) epoch_lr_cfg.lr_per_epoch.push_back(v);

        // lr 调度配置（step 级；step_cosine 用）
        StepLrScheduleConfig step_lr_cfg;
        step_lr_cfg.base_lr = cfg_.lr;
        step_lr_cfg.min_lr = cfg_.min_lr;
        step_lr_cfg.warmup_steps = static_cast<int>(cfg_.warmup_steps);
        step_lr_cfg.total_steps = static_cast<int>(
            (steps_per_epoch > 0) ? steps_per_epoch * static_cast<std::size_t>(
                                       (cfg_.epochs > 0) ? cfg_.epochs : 1)
                                  : cfg_.max_steps);
        step_lr_cfg.cosine = (cfg_.lr_schedule == "step_cosine");

        Scalar current_lr = cfg_.lr;
        if (epoch_mode)
        {
            // 续训起点：先按起始 epoch 定 lr（与 TextTrainer 同口径）
            if (cfg_.lr_schedule != "step_cosine")
            {
                const Scalar elr = compute_epoch_lr(epoch_lr_cfg, static_cast<int>(start_epoch));
                if (elr != session_.current_lr()) session_.set_lr(elr);
                current_lr = elr;
            }
        }

        std::uint64_t steps_executed = 0;
        std::size_t samples_executed = 0;
        std::size_t ckpt_failures = 0;
        std::size_t last_epoch = start_epoch;
        std::size_t last_step_done = start_step;
        bool stop_training = false;
        int abort_code = 0;

        // 定位数据源（断点续训：把游标移到起始位置）
        NN_TRY(rw, source_.rewind(start_epoch, start_step));

        // 绝对步号 = epoch·steps_per_epoch + step（与 step 级调度同定义；
        // 跨续训累计，进程无关）。步进驱动退化为 step 本身。
        auto abs_global_step = [steps_per_epoch](std::size_t epoch,
                                                 std::size_t step) -> std::uint64_t {
            return static_cast<std::uint64_t>(epoch) *
                       static_cast<std::uint64_t>(steps_per_epoch) +
                   static_cast<std::uint64_t>(step);
        };

        auto do_checkpoint = [&](std::size_t epoch, std::size_t step_done) -> Result<void>
        {
            if (!cfg_.enable_checkpoint || !hooks_.on_checkpoint) return {};
            // 计数器记"下一步的位置"，epoch 边界归一化（step_done >= steps_per_epoch
            // ⇒ 下一 epoch 第 0 步）。
            if (steps_per_epoch > 0 && step_done >= steps_per_epoch)
            {
                ++epoch;
                step_done = 0;
            }
            state_->counters.epoch = epoch;
            state_->counters.step_in_epoch = step_done;
            state_->counters.global_step = abs_global_step(epoch, step_done);
            state_->counters.lr = static_cast<double>(current_lr);
            if (const auto r = hooks_.on_checkpoint(session_, *state_); !r) ++ckpt_failures;
            return {};
        };

        if (epoch_mode)
        {
            // ── epoch 驱动 ────────────────────────────────────────────────
            for (int epoch = static_cast<int>(start_epoch); epoch < cfg_.epochs; ++epoch)
            {
                // epoch 级 lr（step_cosine 由 step 级接管）
                if (cfg_.lr_schedule != "step_cosine")
                {
                    const Scalar elr = compute_epoch_lr(epoch_lr_cfg, epoch);
                    if (elr != current_lr)
                    {
                        session_.set_lr(elr);
                        current_lr = elr;
                    }
                }

                const std::size_t epoch_first_step =
                    (static_cast<std::size_t>(epoch) == start_epoch) ? start_step : 0;
                NN_TRY(rw2, source_.rewind(static_cast<std::size_t>(epoch),
                                           epoch_first_step));

                for (std::size_t step = epoch_first_step; step < steps_per_epoch; ++step)
                {
                    if (cfg_.lr_schedule == "step_cosine")
                    {
                        const int gstep = static_cast<int>(
                            static_cast<std::uint64_t>(epoch) *
                                static_cast<std::uint64_t>(steps_per_epoch) +
                            static_cast<std::uint64_t>(step));
                        const Scalar slr = compute_step_lr(step_lr_cfg, gstep);
                        if (slr != current_lr)
                        {
                            session_.set_lr(slr);
                            current_lr = slr;
                        }
                    }

                    const bool is_final =
                        (cfg_.max_steps > 0) && (steps_executed + 1 >= cfg_.max_steps);
                    ++steps_executed;

                    UpdateRequest req;
                    req.epoch = static_cast<std::size_t>(epoch);
                    req.step = step;
                    req.global_step =
                        abs_global_step(static_cast<std::size_t>(epoch), step);
                    req.lr = current_lr;
                    req.is_final_step = is_final;
                    req.steps_per_epoch = steps_per_epoch;

                    NN_TRY(outcome, hooks_.do_update(session_, req));
                    samples_executed += outcome->samples;

                    if (outcome->aborted)
                    {
                        abort_code = 9;
                        stop_training = true;
                        last_epoch = static_cast<std::size_t>(epoch);
                        last_step_done = step + 1;
                        break;
                    }

                    // save_interval 断点（以 update step 计）
                    if (cfg_.save_interval > 0 &&
                        ((step + 1) % cfg_.save_interval == 0 ||
                         step + 1 == steps_per_epoch))
                    {
                        NN_TRY(ck, do_checkpoint(static_cast<std::size_t>(epoch), step + 1));
                    }

                    last_epoch = static_cast<std::size_t>(epoch);
                    last_step_done = step + 1;

                    if (is_final)
                    {
                        stop_training = true;
                        break;
                    }
                }

                if (stop_training) break;

                // ── epoch 边界：断点 + 评估 + 指标 ────────────────────────
                // 断点：save_interval==0 时以 epoch 边界为节奏；save_interval>0 时
                // 末步（step+1==steps_per_epoch）已覆盖 epoch 末，这里不再双写。
                if (cfg_.save_interval == 0)
                {
                    NN_TRY(ck, do_checkpoint(static_cast<std::size_t>(epoch) + 1, 0));
                }

                if (cfg_.enable_eval && hooks_.evaluate)
                {
                    // 评估失败不中止训练（与 TextTrainer 同口径）
                    (void)hooks_.evaluate(session_, static_cast<std::size_t>(epoch));
                }
                if (cfg_.enable_metrics && hooks_.on_epoch_end)
                {
                    RunReport snapshot = report;
                    snapshot.steps_executed = steps_executed;
                    snapshot.samples_executed = samples_executed;
                    snapshot.ckpt_failures = ckpt_failures;
                    snapshot.final_lr = current_lr;
                    snapshot.final_epoch = static_cast<std::size_t>(epoch) + 1;
                    snapshot.final_step_in_epoch = 0;
                    NN_TRY_MSG(hk, "on_epoch_end 失败",
                               hooks_.on_epoch_end(snapshot, static_cast<std::size_t>(epoch)));
                }

                // --stop-after-epoch：训到该 epoch（1-based，不含）即退出
                if (cfg_.stop_after_epoch > 0 &&
                    epoch + 1 >= cfg_.stop_after_epoch)
                {
                    stop_training = true;
                    last_epoch = static_cast<std::size_t>(epoch) + 1;
                    last_step_done = 0;
                    break;
                }
            }

            // 正常跑完（未中止/未提前停）⇒ 计数器归一化到 (epochs, 0)
            // （"下一步位置"语义；与 epoch 边界断点一致）。
            if (!stop_training && abort_code == 0 &&
                static_cast<std::size_t>(cfg_.epochs) > start_epoch)
            {
                last_epoch = static_cast<std::size_t>(cfg_.epochs);
                last_step_done = 0;
            }
        }
        else
        {
            // ── 步进驱动（无固定 epoch；RL/在线学习）───────────────────────
            const std::size_t total = cfg_.max_steps;   // 必 >0（create 已校验）
            for (std::size_t step = start_step; step < total; ++step)
            {
                const int gstep = static_cast<int>(step);
                const Scalar slr = compute_step_lr(step_lr_cfg, gstep);
                if (slr != current_lr)
                {
                    session_.set_lr(slr);
                    current_lr = slr;
                }

                const bool is_final = (step + 1 >= total);
                ++steps_executed;

                UpdateRequest req;
                req.epoch = 0;
                req.step = step;
                req.global_step = abs_global_step(0, step);
                req.lr = current_lr;
                req.is_final_step = is_final;
                req.steps_per_epoch = 0;

                NN_TRY(outcome, hooks_.do_update(session_, req));
                samples_executed += outcome->samples;

                if (outcome->aborted)
                {
                    abort_code = 9;
                    stop_training = true;
                    last_step_done = step + 1;
                    break;
                }

                if (cfg_.save_interval > 0 && (step + 1) % cfg_.save_interval == 0)
                {
                    NN_TRY(ck, do_checkpoint(0, step + 1));
                }

                last_step_done = step + 1;
                if (is_final) break;
            }
        }

        // 非中止时停止位置按"下一步位置"归一化（step_done 越过 epoch 末 ⇒
        // 下一 epoch 第 0 步）；中止保持原始 (epoch, step+1)，与旧口径一致。
        if (abort_code == 0 && steps_per_epoch > 0 && last_step_done >= steps_per_epoch)
        {
            ++last_epoch;
            last_step_done = 0;
        }

        const auto t1 = std::chrono::steady_clock::now();
        report.seconds = std::chrono::duration<Scalar>(t1 - t0).count();
        report.elapsed_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        report.steps_executed = steps_executed;
        report.samples_executed = samples_executed;
        report.ckpt_failures = ckpt_failures;
        report.abort_code = abort_code;
        report.final_lr = current_lr;
        report.final_epoch = last_epoch;
        report.final_step_in_epoch = last_step_done;

        hooks_ = {};   // 钩子只活到本次 fit 返回
        (void)stop_training;
        return report;
    }

    // ── 诊断面 ────────────────────────────────────────────────────────
    [[nodiscard]] const TrainState& state() const noexcept { return *state_; }
    [[nodiscard]] const RunDriverConfig& config() const noexcept { return cfg_; }

private:
    RunDriver(TrainingSession& session, BatchSource& source,
              std::unique_ptr<TrainState> state, RunDriverConfig cfg)
        : session_(session), source_(source), state_(std::move(state)), cfg_(std::move(cfg))
    {
    }

    TrainingSession& session_;
    BatchSource& source_;
    std::unique_ptr<TrainState> state_;
    RunDriverConfig cfg_;
    RunHooks hooks_;
};

} // namespace nn
