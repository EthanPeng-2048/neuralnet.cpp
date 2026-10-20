#pragma once

// ── training_session.hpp — TrainingSession：训练生命周期容器（docs/todo/29）──
//
// 动机（D3 的结构性修复）：`Optimizer` 的 `params_/grads_` 是指向 `Model` 内
// Layer 成员 Tensor 的**非拥有引用**，因此"优化器必须在 Model 存活期内销毁"
// 是一条只能靠调用约定维持的契约。把 Model 与 Optimizer 放进同一个值对象，
// 并让成员声明顺序保证 optimizer 先析构，UAF 就结构性地不可能发生。
//
// 分层与职责：
//   · `Model` 保持纯值对象（forward/backward/generate）——推理不需要 Session；
//   · Session 只拥有 Model + Optimizer + 训练状态，**不**接管数据集 / epoch /
//     CLI / 日志 / Loss（Loss 是任务相关的，调用方按 step 组合）；
//   · 未来的 `InferenceSession` / `RLTrainingSession` 是同一模式的占位设计
//     （见 docs/todo/29 §6/§7），不在本头实现。
//
// 用法：
//     nn::Model model = NN_UNWRAP(nn::build_mnist_mlp_model(*engine));
//     nn::TrainingConfig cfg;
//     cfg.optimizer = "adamw";
//     cfg.lr = 1e-3f;
//     auto session = NN_UNWRAP(nn::TrainingSession::create(std::move(model), cfg));
//
//     // 每 step：调用方组合 forward/loss/backward（与自由工厂用法一致）
//     NN_UNWRAP(session.zero_grad());
//     auto logits = NN_UNWRAP(session.model().forward(x));
//     ... loss ...
//     NN_UNWRAP(session.model().backward(*grad));
//     NN_UNWRAP(session.clip_grads());   // cfg.clip_grad_norm > 0 时生效
//     NN_UNWRAP(session.step());
// ─────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "compute_engine.hpp"
#include "compute_optimizer.hpp"
#include "core_errors.hpp"
#include "model_container.hpp"
#include "precision.hpp"
#include "train_state.hpp"   // nn::TrainState（P2b：训练状态由 Session 持有）

namespace nn
{

// 训练配置：只包含"创建优化器 + 训练期开关"所需的字段。
// 模型结构/精度在 Model 构建时决定；本配置的 precision 只作用于**优化器状态**
// 与 Session 透传的参数（与自由工厂 `create_optimizer(..., precision)` 同义）。
struct TrainingConfig
{
    std::string optimizer = "adamw";
    Scalar lr = 1e-4f;
    Scalar weight_decay = 0.01f;
    PrecisionProfile precision{};
    Scalar beta1 = 0.9f;
    Scalar beta2 = 0.999f;
    Scalar clip_grad_norm = 0;          // 0 = 不裁剪
    // 模型侧训练期开关（若模型支持；create() 在构造优化器前应用）
    bool activation_offload = false;
    std::size_t checkpoint_every = 0;
};

class TrainingSession
{
public:
    // 唯一入口：Model 必须已绑定引擎。内部按 cfg 创建 optimizer。
    [[nodiscard]] static Result<TrainingSession> create(Model model,
                                                        TrainingConfig cfg = {})
    {
        if (!model.engine_bound())
            NN_FAIL("TrainingSession::create: Model 未绑定引擎"
                    "（请用 Model(engine) / build_*_model 构建）");

        // 模型侧训练期开关先应用（检查点/offload 都由 Model 自己实现）。
        if (cfg.checkpoint_every > 0)
            model.set_checkpoint_every(cfg.checkpoint_every);
        if (cfg.activation_offload)
        {
            // CPU 引擎的 offload 原语是 no-op（restore 返回 1×1）→ 开启会破坏
            // backward。CLI 早已忽略；Session 对编程式调用方直接拒绝。
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

        return TrainingSession(std::move(model), std::move(opt), std::move(cfg));
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
    [[nodiscard]] std::uint64_t step_count() const noexcept { return step_count_; }

    // ── 训练状态（P2b）──────────────────────────────────────────────────
    // `TrainState` = 计数器 / lr 配方 / 数据集身份 / 优化器指针。**写入方是训练
    // 驱动**（epoch × step 的位置只有循环知道）；Session 只负责持有它，保证
    // "断点状态与模型/优化器同生共死"，调用方不必再手工搬运一份状态。
    [[nodiscard]] const TrainState& train_state() const noexcept { return state_; }
    [[nodiscard]] TrainState& train_state() noexcept { return state_; }
    void set_train_state(TrainState state) { state_ = std::move(state); }

    // 训练原语（epoch/step 循环与数据集仍留在调用方/CLI）
    [[nodiscard]] Result<void> zero_grad() { return model_.zero_grad(); }

    // 全局 L2 梯度裁剪（cfg.clip_grad_norm <= 0 时为 no-op）
    [[nodiscard]] Result<void> clip_grads()
    {
        if (cfg_.clip_grad_norm <= 0)
            return {};
        return opt_->clip_grad_norm(cfg_.clip_grad_norm);
    }

    [[nodiscard]] Result<void> step()
    {
        NN_TRY(r, opt_->step());
        ++step_count_;
        return {};
    }

    void set_lr(Scalar lr) { opt_->set_lr(lr); }

private:
    // ⚠ 成员声明顺序 = 析构顺序的正确性：model_ 在前、opt_ 在后 ⇒ optimizer
    // 先于 Model 析构（`params_/grads_` 引用的 Tensor 成员始终有效）。
    Model model_{};
    std::unique_ptr<Optimizer> opt_{};
    TrainingConfig cfg_{};
    std::uint64_t step_count_ = 0;
    TrainState state_{};

    TrainingSession(Model model, std::unique_ptr<Optimizer> opt, TrainingConfig cfg)
        : model_(std::move(model)), opt_(std::move(opt)), cfg_(std::move(cfg))
    {
    }
};

} // namespace nn
