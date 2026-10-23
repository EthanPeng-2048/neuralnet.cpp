#pragma once

// ── training_session.hpp — TrainingSession：单次参数更新会话（docs/todo/30）──
//
// 定位：
//   · 拥有 Model + Optimizer，结构性保证 optimizer 先于参数所有者析构；
//   · 接收一个已经装配好的 micro-batch，通过任务回调完成 forward/loss/backward；
//   · 内部维护梯度累积，在 update 边界统一归一化、裁剪、step、zero_grad；
//   · **不**拥有数据集、shuffle、epoch、lr 曲线、评估或 checkpoint 策略。
//
// 一次 update 的全局有效单位由调用方传入。Step 回调返回“已按
// gradient_units 归一化”的输出梯度；Session 再乘
// gradient_units / normalization_units，使多个 micro-batch 等价于一个
// global batch。该契约保证 mask 后有效 token 不等的 micro-batch 不会被错误地
// 按“micro 均值的均值”训练。
//
// 推荐用法：
//     auto session = NN_UNWRAP(nn::TrainingSession::create(std::move(model), cfg));
//     for (each micro_batch) {
//         auto report = NN_UNWRAP(session.train_step(
//             micro_batch, opts,
//             [&](nn::Model& model, nn::ComputeEngine& engine, const Batch& batch) {
//                 ... forward + loss + loss.backward() ...
//                 return nn::TrainingStepOutput{grad, units, loss_sum, loss_units};
//             }));
//     }
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "compute_engine.hpp"
#include "compute_optimizer.hpp"
#include "core_errors.hpp"
#include "model_container.hpp"
#include "precision.hpp"

namespace nn
{

// 创建优化器与单步执行所需的稳定配置。
// epoch/lr 曲线/数据网格不在这里；它们属于 Run Driver。
struct TrainingConfig
{
    std::string optimizer = "adamw";
    Scalar lr = 1e-4f;
    Scalar weight_decay = 0.01f;
    PrecisionProfile precision{};
    Scalar beta1 = 0.9f;
    Scalar beta2 = 0.999f;
    Scalar clip_grad_norm = 0;          // 低层 clip_grads() 的默认值
    std::size_t grad_accum_steps = 1;   // 一次 update 的 micro-step 数

    // 模型侧训练期开关（create 在构造优化器前应用）。
    bool activation_offload = false;
    std::size_t checkpoint_every = 0;
};

struct TrainingSessionState
{
    Scalar current_lr = 0;
    std::uint64_t optimizer_step = 0;
    std::size_t microsteps_in_update = 0;
    std::size_t pending_loss_units = 0;
    Tensor pending_loss_sum;
    Scalar update_lr = 0;
    std::size_t update_normalization_units = 0;
    bool accumulation_dirty = false;
};

// 任务 Step 回调的输出。
//
// `grad` 可以按本 micro-batch 的自然单位归一化（例如 CE 按本 micro 的
// 有效 token 数），但必须通过 `gradient_units` 如实报告该归一化单位数。
// `loss_sum` 是可选设备侧标量分母量；Text 稀疏 CE 用 NLL sum，使训练热路径
// 不必在每步同步下载 loss。
struct TrainingStepOutput
{
    Tensor grad;
    std::size_t gradient_units = 0;
    Tensor loss_sum;
    std::size_t loss_units = 0;
};

// 具名多输出模型的 micro-step 结果。字段必须覆盖本次 forward 产生的输出端口；
// Session 只负责按 global normalization 缩放并交给 Model::backward_named。
struct NamedTrainingStepOutput
{
    GradientBundle gradients;
    std::size_t gradient_units = 0;
    Tensor loss_sum;
    std::size_t loss_units = 0;
};

struct TrainingStepOptions
{
    std::optional<Scalar> lr;
    Scalar clip_grad_norm = 0;
    // 本次 global update 的有效单位总数。必须在本 update 的所有 micro-step
    // 中保持一致。
    std::size_t normalization_units = 0;
    bool force_update = false;

    // backward 已结束、参数尚未更新。返回 true = 中止本次 update（梯度清零，
    // report.aborted = true）；返回错误 = 传播错误并将累积标记为无效。
    std::function<Result<bool>()> before_update;
    // optimizer.step 已成功、zero_grad 之前。适合梯度统计等诊断。
    std::function<Result<void>()> after_update;
};

struct TrainingStepReport
{
    bool parameters_updated = false;
    bool aborted = false;
    std::uint64_t optimizer_step = 0;
    std::size_t microsteps_accumulated = 0;
    Tensor loss_sum;
    std::size_t loss_units = 0;
};

class TrainingSession
{
public:
    [[nodiscard]] static Result<TrainingSession> create(Model model,
                                                        TrainingConfig cfg = {})
    {
        if (!model.engine_bound())
            NN_FAIL("TrainingSession::create: Model 未绑定引擎"
                    "（请用 Model(engine) / build_*_model 构建）");
        if (cfg.grad_accum_steps == 0)
            NN_FAIL("TrainingSession::create: grad_accum_steps 必须 > 0");

        if (cfg.checkpoint_every > 0)
            model.set_checkpoint_every(cfg.checkpoint_every);
        if (cfg.activation_offload)
        {
            if (model.engine().device() != Device::GPU)
                NN_FAIL("TrainingSession::create: activation_offload 仅支持 GPU 引擎"
                        "（CPU 的 offload 原语是 no-op，请改用 checkpoint_every）");
            model.set_activation_offload(true);
        }

        ComputeEngine& engine = model.engine();
        auto opt = create_optimizer(cfg.optimizer, engine, model.parameters(),
                                    model.param_gradients(), cfg.lr,
                                    cfg.weight_decay, cfg.precision, cfg.beta1,
                                    cfg.beta2);
        if (!opt)
            NN_FAIL("TrainingSession::create: 未知优化器 '" + cfg.optimizer + "'"
                    "（可用: sgd / sgd_momentum / adam / adamw / adamwquant / muon）");

        TrainingSessionState state;
        state.current_lr = cfg.lr;
        return TrainingSession(std::move(model), std::move(opt), std::move(cfg),
                               std::move(state));
    }

    TrainingSession(TrainingSession&&) noexcept = default;
    TrainingSession& operator=(TrainingSession&&) noexcept = default;
    TrainingSession(const TrainingSession&) = delete;
    TrainingSession& operator=(const TrainingSession&) = delete;
    ~TrainingSession() = default;

    [[nodiscard]] Model& model() noexcept { return model_; }
    [[nodiscard]] const Model& model() const noexcept { return model_; }
    [[nodiscard]] Optimizer& optimizer() noexcept { return *opt_; }
    [[nodiscard]] const Optimizer& optimizer() const noexcept { return *opt_; }
    [[nodiscard]] ComputeEngine& engine() noexcept { return model_.engine(); }
    [[nodiscard]] const TrainingConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] const TrainingSessionState& state() const noexcept { return state_; }
    [[nodiscard]] Scalar current_lr() const noexcept { return state_.current_lr; }
    [[nodiscard]] std::uint64_t step_count() const noexcept
    { return state_.optimizer_step; }

    // 核心单步入口。`step_fn` 只在本次调用期间使用，不进入 Session 状态。
    template <class Batch, class StepFn>
    [[nodiscard]] Result<TrainingStepReport>
    train_step(const Batch& batch, TrainingStepOptions options, StepFn&& step_fn)
    {
        if (state_.accumulation_dirty)
            NN_FAIL("TrainingSession::train_step: 上一次失败留下了不完整梯度；"
                    "请先 reset_accumulation()");
        if (options.normalization_units == 0)
            NN_FAIL("TrainingSession::train_step: normalization_units 必须 > 0");

        const bool starts_update = state_.microsteps_in_update == 0;
        if (starts_update)
        {
            if (options.lr) set_lr(*options.lr);
            state_.update_lr = state_.current_lr;
            state_.update_normalization_units = options.normalization_units;
        }
        else
        {
            if (options.lr && *options.lr != state_.update_lr)
                NN_FAIL("TrainingSession::train_step: 同一 update 的 lr 发生变化");
            if (options.normalization_units != state_.update_normalization_units)
                NN_FAIL("TrainingSession::train_step: 同一 update 的 "
                        "normalization_units 发生变化（" +
                        std::to_string(state_.update_normalization_units) + " vs " +
                        std::to_string(options.normalization_units) + "）");
        }

        Result<void> begin_r = starts_update ? begin_and_zero_() : Result<void>{};
        NN_TRY_CHECK(begin_r);

        RecordingGuard rec{engine()};
        auto out_r = step_fn(model_, engine(), batch);
        if (!out_r)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(out_r.error());
        }
        TrainingStepOutput out = std::move(*out_r);
        if (!out.grad.valid())
        {
            state_.accumulation_dirty = true;
            NN_FAIL("TrainingSession::train_step: Step 回调返回了空梯度");
        }

        if (out.gradient_units > 0 &&
            options.normalization_units != out.gradient_units)
        {
            if (out.gradient_units > options.normalization_units)
            {
                state_.accumulation_dirty = true;
                NN_FAIL("TrainingSession::train_step: gradient_units 超出 "
                        "normalization_units（" +
                        std::to_string(out.gradient_units) + " > " +
                        std::to_string(options.normalization_units) + "）");
            }
            const Scalar scale =
                static_cast<Scalar>(out.gradient_units) /
                static_cast<Scalar>(options.normalization_units);
            auto sr = engine().scale_inplace(out.grad, scale);
            if (!sr)
            {
                state_.accumulation_dirty = true;
                NN_FAIL(sr.error());
            }
        }

        auto bwd = model_.backward(out.grad);
        if (!bwd)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(bwd.error());
        }

        if (out.loss_sum.valid())
        {
            if (!state_.pending_loss_sum.valid())
            {
                state_.pending_loss_sum = std::move(out.loss_sum);
            }
            else
            {
                auto ar = engine().accumulate(state_.pending_loss_sum, out.loss_sum);
                if (!ar)
                {
                    state_.accumulation_dirty = true;
                    NN_FAIL(ar.error());
                }
            }
            state_.pending_loss_units += out.loss_units;
        }

        auto end_r = rec.finish();
        if (!end_r)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(end_r.error());
        }
        ++state_.microsteps_in_update;

        const bool update_ready =
            state_.microsteps_in_update >= cfg_.grad_accum_steps;
        if (!update_ready && !options.force_update)
        {
            TrainingStepReport report;
            report.optimizer_step = state_.optimizer_step;
            report.microsteps_accumulated = state_.microsteps_in_update;
            return report;
        }

        if (options.before_update)
        {
            auto stop_r = options.before_update();
            if (!stop_r)
            {
                state_.accumulation_dirty = true;
                NN_FAIL(stop_r.error());
            }
            if (*stop_r)
            {
                auto rr = reset_accumulation();
                NN_TRY_CHECK(rr);
                TrainingStepReport report;
                report.aborted = true;
                report.optimizer_step = state_.optimizer_step;
                return report;
            }
        }

        auto update_r = finish_update_(options);
        NN_TRY_CHECK(update_r);

        TrainingStepReport report;
        report.parameters_updated = true;
        report.optimizer_step = state_.optimizer_step;
        report.microsteps_accumulated = state_.microsteps_in_update;
        report.loss_sum = std::move(state_.pending_loss_sum);
        report.loss_units = state_.pending_loss_units;
        clear_accumulation_state_();
        return report;
    }

    // 具名多输出版本：更新/累积/归一化/裁剪语义与 train_step 完全一致，唯一差别
    // 是 step_fn 返回 GradientBundle 并走 Model::backward_named。
    template <class Batch, class StepFn>
    [[nodiscard]] Result<TrainingStepReport>
    train_named_step(const Batch& batch, TrainingStepOptions options,
                     StepFn&& step_fn)
    {
        if (state_.accumulation_dirty)
            NN_FAIL("TrainingSession::train_named_step: 上一次失败留下了不完整"
                    "梯度；请先 reset_accumulation()");
        if (options.normalization_units == 0)
            NN_FAIL("TrainingSession::train_named_step: normalization_units "
                    "必须 > 0");

        const bool starts_update = state_.microsteps_in_update == 0;
        if (starts_update)
        {
            if (options.lr) set_lr(*options.lr);
            state_.update_lr = state_.current_lr;
            state_.update_normalization_units = options.normalization_units;
        }
        else
        {
            if (options.lr && *options.lr != state_.update_lr)
                NN_FAIL("TrainingSession::train_named_step: 同一 update 的 lr "
                        "发生变化");
            if (options.normalization_units != state_.update_normalization_units)
                NN_FAIL("TrainingSession::train_named_step: 同一 update 的 "
                        "normalization_units 发生变化（" +
                        std::to_string(state_.update_normalization_units) +
                        " vs " +
                        std::to_string(options.normalization_units) + "）");
        }

        Result<void> begin_r = starts_update ? begin_and_zero_() : Result<void>{};
        NN_TRY_CHECK(begin_r);

        RecordingGuard rec{engine()};
        auto out_r = step_fn(model_, engine(), batch);
        if (!out_r)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(out_r.error());
        }
        NamedTrainingStepOutput out = std::move(*out_r);
        if (out.gradients.size() == 0)
        {
            state_.accumulation_dirty = true;
            NN_FAIL("TrainingSession::train_named_step: Step 回调返回了空梯度包");
        }

        if (out.gradient_units > 0 &&
            options.normalization_units != out.gradient_units)
        {
            if (out.gradient_units > options.normalization_units)
            {
                state_.accumulation_dirty = true;
                NN_FAIL("TrainingSession::train_named_step: gradient_units "
                        "超出 normalization_units（" +
                        std::to_string(out.gradient_units) + " > " +
                        std::to_string(options.normalization_units) + "）");
            }
            const Scalar scale =
                static_cast<Scalar>(out.gradient_units) /
                static_cast<Scalar>(options.normalization_units);
            for (std::size_t i = 0; i < out.gradients.schema().size(); ++i)
            {
                const FieldHandle handle{
                    out.gradients.schema().fingerprint(), i};
                auto grad_r = out.gradients.require(handle);
                NN_TRY_CHECK(grad_r);
                if (*grad_r == nullptr) continue;
                auto sr = engine().scale_inplace(**grad_r, scale);
                if (!sr)
                {
                    state_.accumulation_dirty = true;
                    NN_FAIL(sr.error());
                }
            }
        }

        auto bwd = model_.backward_named(out.gradients);
        if (!bwd)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(bwd.error());
        }

        if (out.loss_sum.valid())
        {
            if (!state_.pending_loss_sum.valid())
            {
                state_.pending_loss_sum = std::move(out.loss_sum);
            }
            else
            {
                auto ar = engine().accumulate(state_.pending_loss_sum,
                                              out.loss_sum);
                if (!ar)
                {
                    state_.accumulation_dirty = true;
                    NN_FAIL(ar.error());
                }
            }
            state_.pending_loss_units += out.loss_units;
        }

        auto end_r = rec.finish();
        if (!end_r)
        {
            state_.accumulation_dirty = true;
            NN_FAIL(end_r.error());
        }
        ++state_.microsteps_in_update;

        const bool update_ready =
            state_.microsteps_in_update >= cfg_.grad_accum_steps;
        if (!update_ready && !options.force_update)
        {
            TrainingStepReport report;
            report.optimizer_step = state_.optimizer_step;
            report.microsteps_accumulated = state_.microsteps_in_update;
            return report;
        }

        if (options.before_update)
        {
            auto stop_r = options.before_update();
            if (!stop_r)
            {
                state_.accumulation_dirty = true;
                NN_FAIL(stop_r.error());
            }
            if (*stop_r)
            {
                auto rr = reset_accumulation();
                NN_TRY_CHECK(rr);
                TrainingStepReport report;
                report.aborted = true;
                report.optimizer_step = state_.optimizer_step;
                return report;
            }
        }

        auto update_r = finish_update_(options);
        NN_TRY_CHECK(update_r);

        TrainingStepReport report;
        report.parameters_updated = true;
        report.optimizer_step = state_.optimizer_step;
        report.microsteps_accumulated = state_.microsteps_in_update;
        report.loss_sum = std::move(state_.pending_loss_sum);
        report.loss_units = state_.pending_loss_units;
        clear_accumulation_state_();
        return report;
    }

    // 失败后显式丢弃不完整 update。正常情况下 train_step 在 update 边界自动清零。
    [[nodiscard]] Result<void> reset_accumulation()
    {
        clear_accumulation_state_();
        auto b = engine().begin_batch();
        if (!b) return b;
        RecordingGuard rec{engine()};
        auto z = model_.zero_grad();
        if (!z) return z;
        auto end_r = rec.finish();
        if (end_r) state_.accumulation_dirty = false;
        return end_r;
    }

    // ── 低层逃生接口（自定义训练器/诊断）────────────────────────────────
    [[nodiscard]] Result<void> zero_grad() { return model_.zero_grad(); }

    [[nodiscard]] Result<void> clip_grads()
    {
        if (cfg_.clip_grad_norm <= 0) return {};
        return opt_->clip_grad_norm(cfg_.clip_grad_norm);
    }

    [[nodiscard]] Result<void> step()
    {
        NN_TRY(r, opt_->step());
        ++state_.optimizer_step;
        return {};
    }

    void set_lr(Scalar lr)
    {
        state_.current_lr = lr;
        opt_->set_lr(lr);
    }

private:
    // GPU 录制作用域。正常路径显式 finish()；错误提前返回时析构兜底 end_batch。
    class RecordingGuard
    {
    public:
        explicit RecordingGuard(ComputeEngine& engine) noexcept : engine_(&engine) {}
        RecordingGuard(RecordingGuard&& other) noexcept
            : engine_(other.engine_) { other.engine_ = nullptr; }
        RecordingGuard& operator=(RecordingGuard&&) = delete;
        RecordingGuard(const RecordingGuard&) = delete;
        ~RecordingGuard()
        {
            if (engine_ != nullptr) (void)engine_->end_batch();
        }

        [[nodiscard]] Result<void> finish()
        {
            ComputeEngine* e = engine_;
            engine_ = nullptr;
            if (e == nullptr) return {};
            return e->end_batch();
        }

    private:
        ComputeEngine* engine_;
    };

    [[nodiscard]] Result<void> begin_and_zero_()
    {
        auto b = engine().begin_batch();
        if (!b) return b;
        RecordingGuard rec{engine()};
        auto z = model_.zero_grad();
        if (!z) return z;
        return rec.finish();
    }

    [[nodiscard]] Result<void> finish_update_(const TrainingStepOptions& options)
    {
        auto b = engine().begin_batch();
        if (!b) return b;
        RecordingGuard rec{engine()};

        const Scalar max_norm = options.clip_grad_norm;
        if (max_norm > 0)
        {
            auto c = opt_->clip_grad_norm(max_norm);
            if (!c) return c;
        }
        auto s = opt_->step();
        if (!s)
        {
            state_.accumulation_dirty = true;
            return s;
        }
        ++state_.optimizer_step;
        state_.current_lr = state_.update_lr;

        if (options.after_update)
        {
            auto h = options.after_update();
            if (!h)
            {
                state_.accumulation_dirty = true;
                return h;
            }
        }
        auto z = model_.zero_grad();
        if (!z)
        {
            state_.accumulation_dirty = true;
            return z;
        }
        auto end_r = rec.finish();
        if (!end_r) state_.accumulation_dirty = true;
        return end_r;
    }

    void clear_accumulation_state_() noexcept
    {
        state_.microsteps_in_update = 0;
        state_.pending_loss_units = 0;
        state_.pending_loss_sum = {};
        state_.update_lr = state_.current_lr;
        state_.update_normalization_units = 0;
    }

    // ⚠ 声明顺序 = 析构顺序：model 先声明、optimizer 后声明 ⇒ optimizer 先析构。
    Model model_{};
    std::unique_ptr<Optimizer> opt_{};
    TrainingConfig cfg_{};
    TrainingSessionState state_{};

    TrainingSession(Model model, std::unique_ptr<Optimizer> opt, TrainingConfig cfg,
                    TrainingSessionState state)
        : model_(std::move(model)), opt_(std::move(opt)), cfg_(std::move(cfg)),
          state_(std::move(state))
    {
    }
};

} // namespace nn
