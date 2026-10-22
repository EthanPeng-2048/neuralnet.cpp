// ── text_trainer.hpp — 文本训练 Run Driver：数据集 → global batch → Session ─
//
// P2b（docs/todo/29 §8）的"训练工厂"：把原先散在 `text_train.cpp` / `cli/
// text_train_loop.hpp` 里的**数据装配 + epoch×step 循环 + 断点状态**收进库层，
// 调用方只需要：
//
//     auto dataset  = NN_UNWRAP(nn::Dataset::open(path));
//     auto session  = NN_UNWRAP(nn::TrainingSession::create(std::move(model), tcfg));
//     auto trainer  = NN_UNWRAP(nn::TextTrainer::create(std::move(session), dataset,
//                                                       nn::TextTrainConfig{}));
//     auto report   = NN_UNWRAP(trainer.fit());
//
// 分工（docs/todo/30）：
//   · `TrainingSession` 拥有 Model + Optimizer，执行 micro-step 并内置梯度累积；
//   · Step 回调承载文本 sparse-CE，Session 不认识 token/loss mask；
//   · `TextSampleSet` 拥有样本与最终 loss_mask，负责 batch 装配；
//   · **本类 = Run Driver**：global batch 切分、sample 顺序、lr 调度、epoch、
//     测试评估、TrainState 与断点时机。Session 仍不打印、不保存、不看数据。
//
// Batch 语义：`cfg.batch_size` = 一次 optimizer update 的理论 global batch；
// 实际 micro-batch = `cfg.batch_size / cfg.accum_steps`。一次 update 的梯度按
// global batch 的有效 token 总数归一化，而不是把各 micro 的均值再平均。
//
// 可观测行为锚（与 P1-1 拆分后的 `cli/text_train_loop.hpp` 逐字一致）：
//   * 进度/epoch 汇总/测试 loss 的文案与顺序；
//   * 同一 seed 语义（每 epoch shuffle；seed 由 random_device 给出，与旧行为一致）；
//   * 断点触发点与 TrainState 计数器语义（make_train_state 的 epoch 归一化）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "compute_engine.hpp"        // nn::ComputeEngine
#include "compute_loss.hpp"          // nn::CrossEntropyLoss
#include "compute_optimizer.hpp"     // nn::Optimizer
#include "core_config.hpp"           // nn::Scalar
#include "core_errors.hpp"           // nn::Result / NN_TRY / NN_FAIL
#include "core_file.hpp"             // nn::sha256_file_hex（数据集身份）
#include "core_profile.hpp"          // NN_PROF_SCOPE / nn::prof::Phase
#include "dataset.hpp"               // nn::Dataset
#include "dataset_samples.hpp"       // nn::TextSampleSet / TextSampleBatch
#include "domain_tokenizer.hpp"      // nn::load_tokenizer_from_kvrec / Tokenizer
#include "lr_scheduler.hpp"          // nn::compute_epoch_lr / compute_step_lr
#include "model_container.hpp"       // nn::Model
#include "text_train_config.hpp"     // nn::TextTrainConfig
#include "train_state.hpp"           // nn::TrainState / DatasetIdentity
#include "training_session.hpp"      // nn::TrainingSession

namespace nn
{

// ── 断点保存请求（CLI 的序列化层在 hook 里落地）──────────────────────────
// Trainer 负责"何时/状态是什么"，保存路径与文件格式留给调用方（CLI 的
// `save_ckpt_with_state` / 库外使用者自己的 save_model）。
struct TextCheckpointRequest
{
    TrainingSession& session;    // model/optimizer 的当前状态
    const TrainState& state;     // 已按计数器/配方/数据集身份装配好
};

// ── 训练期钩子（全部可空；空 = 该能力不启用）─────────────────────────────
struct TextTrainerHooks
{
    std::ostream* out = &std::cout;   // 进度输出（默认 stdout；测试可重定向）
    std::ostream* err = &std::cerr;   // 错误/警告输出（当前 Trainer 本体不写）

    // batch 装配完成后、上传前（诊断探针用；NN_NAN_TRACE 的 token/doc_ids 快照）
    std::function<void(TrainingSession&, const TextSampleBatch&, std::size_t step)> on_batch_ready;
    // backward 已提交后（返回 true = 探针要求中止训练：退出码 9 语义）
    std::function<bool(TrainingSession&, std::size_t step)> on_probe_check;
    // 参数更新完成、zero_grad 之前（--grad-log 的梯度统计）
    std::function<void(TrainingSession&, std::size_t epoch, std::size_t step, Scalar lr)> on_update;
    // 定期断点。返回错误 = 本次断点未落盘（Trainer 计数，训练继续）。
    // **契约**：诊断信息由 hook 自己打印（CLI 用 TextTrainCheckpointLog）；Trainer
    // 只计数并计入 TextTrainReport.ckpt_failures（供收尾退出码）。
    std::function<Result<void>(const TextCheckpointRequest&)> on_checkpoint;
    // 显存阶段标记（NN_MEM_STATS 探针）
    std::function<void(const char*)> on_phase;
};

// ── 训练结果（退出码/统计由调用方消费）───────────────────────────────────
struct TextTrainReport
{
    Scalar seconds = 0;                 // 循环墙钟（秒）
    std::uint64_t elapsed_ns = 0;       // 同一区间的纳秒数（NN_PROFILE 报告用）
    std::size_t steps_executed = 0;     // 本进程执行的训练步数（--max-steps 计数）
    std::size_t samples_executed = 0;   // 本进程实际消费的样本数（含末批不满）
    std::size_t ckpt_failures = 0;      // 断点保存失败次数（收尾退出码据此）
    int abort_code = 0;                 // 非 0 = 训练被探针中止（9）：跳过收尾保存
    Scalar final_lr = 0;
    // 训练停止时的"下一步位置"（与 Trainer 持有的 TrainState 同口径）：
    // 正常跑完 = (epochs, 0)；--max-steps 提前停 = 实际下一步；探针中止 = 不更新状态，
    // 这两个字段仍报告实际位置。
    std::size_t final_epoch = 0;
    std::size_t final_step_in_epoch = 0;
};

// ── 从 Session 的优化器配方派生训练配方（避免两份默认值漂移）──────────────
// `TextTrainConfig` 中 optimizer/lr/weight_decay/beta/precision/max_norm 必须与
// 构造 Session 的 `TrainingConfig` 一致（`TextTrainer::create` 会逐项校验）；
// 这个 helper 给出"从 Session 展开"的便捷起步，调用方只改数据网格/调度形状。
[[nodiscard]] inline TextTrainConfig make_text_train_config(const TrainingSession& session)
{
    TextTrainConfig c;
    const TrainingConfig& sc = session.config();
    c.optimizer_name = sc.optimizer;
    c.lr = sc.lr;
    c.weight_decay = sc.weight_decay;
    c.beta1 = sc.beta1;
    c.beta2 = sc.beta2;
    c.precision = sc.precision;
    c.max_norm = sc.clip_grad_norm;
    c.accum_steps = sc.grad_accum_steps;
    return c;
}

// ── 断点状态装配（P2b：唯一实现；CLI/测试都走这里）────────────────────────
// 计数器记的是**下一步的位置**（epoch 边界已归一化）⇒ 续训直接拿它当起点；
// `global_step` 与 step 级调度的索引同定义（= epoch·steps_per_epoch + step）；
// `lr` 只作显示/交叉校验（权威值由调度按步号重算）。
[[nodiscard]] inline TrainState
make_text_train_state(const TextTrainConfig& cfg, const DatasetIdentity& identity,
                      std::size_t steps_per_epoch, std::size_t epoch, std::size_t step_done,
                      Scalar current_lr, std::uint64_t optimizer_step)
{
    TrainState s;
    std::size_t e = epoch, st = step_done;
    if (steps_per_epoch > 0 && st >= steps_per_epoch)
    {
        ++e;
        st = 0;
    }
    s.counters.epoch = e;
    s.counters.step_in_epoch = st;
    s.counters.global_step =
        static_cast<std::uint64_t>(e) * static_cast<std::uint64_t>(steps_per_epoch) + st;
    s.counters.lr = static_cast<double>(current_lr);

    s.recipe.precision = cfg.precision;
    s.recipe.optimizer = cfg.optimizer_name;
    s.recipe.lr_schedule = cfg.lr_schedule;
    s.recipe.lr = static_cast<double>(cfg.lr);
    s.recipe.min_lr = static_cast<double>(cfg.min_lr);
    s.recipe.weight_decay = static_cast<double>(cfg.weight_decay);
    s.recipe.beta1 = static_cast<double>(cfg.beta1);
    s.recipe.beta2 = static_cast<double>(cfg.beta2);
    s.recipe.max_norm = static_cast<double>(cfg.max_norm);
    s.recipe.epochs = static_cast<std::uint64_t>(cfg.epochs);
    s.recipe.total_steps =
        static_cast<std::uint64_t>(steps_per_epoch) * static_cast<std::uint64_t>(cfg.epochs);
    s.recipe.warmup_epochs = static_cast<std::uint64_t>(cfg.warmup_epochs);
    s.recipe.warmup_steps = static_cast<std::uint64_t>(cfg.warmup_steps);
    s.recipe.batch_size = static_cast<std::uint64_t>(cfg.batch_size);
    s.recipe.micro_batch_size =
        static_cast<std::uint64_t>(cfg.batch_size / cfg.accum_steps);
    s.recipe.batch_size_is_global = true;
    s.recipe.accum_steps = static_cast<std::uint64_t>(cfg.accum_steps);
    s.recipe.stride = static_cast<std::uint64_t>(cfg.stride);
    s.recipe.lr_per_epoch.clear();
    s.recipe.lr_per_epoch.reserve(cfg.lr_per_epoch.size());
    for (Scalar v : cfg.lr_per_epoch)
        s.recipe.lr_per_epoch.push_back(static_cast<double>(v));

    s.dataset = identity;
    s.dataset.steps_per_epoch = static_cast<std::uint64_t>(steps_per_epoch);
    s.optimizer_step = optimizer_step;
    return s;
}

// ══════════════════════════════════════════════════════════════════════════
//  TextTrainer
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline bool same_precision_profile(const PrecisionProfile& a,
                                                 const PrecisionProfile& b) noexcept
{
    return a.param == b.param && a.compute == b.compute && a.stable == b.stable &&
           a.optimizer == b.optimizer;
}
class TextTrainer
{
public:
    // 便捷入口：直接把数据集交给训练工厂（内部建样本集 + 载词表）。
    // `cfg.identity` 为空字段由工厂补齐（sha256 空 = 计算整文件摘要）。
    [[nodiscard]] static Result<TextTrainer> create(TrainingSession session, const Dataset& dataset,
                                                    TextTrainConfig cfg,
                                                    TextTrainerHooks hooks = {})
    {
        NN_TRY_MSG(tokenizer_r, "加载数据集内嵌词表失败",
                   load_tokenizer_from_kvrec(dataset.vocab_kvrec()));
        std::unique_ptr<Tokenizer> tokenizer = std::move(*tokenizer_r);

        const TextWindowConfig win{cfg.seq_len, cfg.stride, tokenizer->pad_id()};
        NN_TRY_MSG(train_r, "读取训练样本集失败",
                   TextSampleSet::create(dataset, DatasetSubset::Train, win));
        std::optional<TextSampleSet> test;
        if (dataset.info().test_docs > 0)
        {
            NN_TRY_MSG(test_r, "读取测试样本集失败",
                       TextSampleSet::create(dataset, DatasetSubset::Test, win));
            // 空 test 子集（合法文件：test 全是空 doc）⇒ 视为"无 test"，跳过评估
            if (test_r->sample_count() > 0) test = std::move(*test_r);
        }
        return create_impl(std::move(session), std::move(*train_r), std::move(test),
                           std::move(tokenizer), std::move(cfg), std::move(hooks));
    }

    // 高性能入口：调用方已建好样本集（避免同一数据集重复加载/哈希），
    // CLI 走这条（样本集在打印数据集统计时就要用到）。
    [[nodiscard]] static Result<TextTrainer> create(TrainingSession session, TextSampleSet train,
                                                    std::optional<TextSampleSet> test,
                                                    TextTrainConfig cfg,
                                                    TextTrainerHooks hooks = {})
    {
        // 高性能入口的网格来自样本集：配方必须与样本集一致（否则 cfg 会被静默覆盖）
        if (train.subset() != DatasetSubset::Train)
            NN_FAIL("TextTrainer::create: 训练样本集必须是 DatasetSubset::Train");
        if (test && test->subset() != DatasetSubset::Test)
            NN_FAIL("TextTrainer::create: 测试样本集必须是 DatasetSubset::Test");
        if (cfg.seq_len != train.seq_len())
            NN_FAIL("TextTrainer::create: 配方 seq_len 与训练样本集不一致（"
                    + std::to_string(cfg.seq_len) + " vs "
                    + std::to_string(train.seq_len()) + "）");
        if (cfg.stride != 0 && cfg.stride != train.stride())
            NN_FAIL("TextTrainer::create: 配方 stride 与训练样本集不一致（"
                    + std::to_string(cfg.stride) + " vs "
                    + std::to_string(train.stride()) + "）");
        NN_TRY_MSG(tokenizer_r, "加载数据集内嵌词表失败",
                   load_tokenizer_from_kvrec(train.vocab_kvrec()));
        if ((*tokenizer_r)->pad_id() != train.pad_id())
            NN_FAIL("TextTrainer::create: 样本集 pad_id 与词表不一致（"
                    + std::to_string(train.pad_id()) + " vs "
                    + std::to_string((*tokenizer_r)->pad_id()) + "）");
        return create_impl(std::move(session), std::move(train), std::move(test),
                           std::move(*tokenizer_r), std::move(cfg), std::move(hooks));
    }

    TextTrainer(TextTrainer&&) noexcept = default;
    TextTrainer& operator=(TextTrainer&&) noexcept = default;
    TextTrainer(const TextTrainer&) = delete;
    TextTrainer& operator=(const TextTrainer&) = delete;
    ~TextTrainer() = default;

    // ── 训练 ────────────────────────────────────────────────────────────
    // 安装/替换钩子（CLI 在 fit 前装诊断/断点；库外使用者可留空）。
    // **生命周期契约**：hooks 只保证活到下一次 `fit()` 返回（fit 结束会清空），
    // 因此 lambda 捕获的局部引用必须活过这次调用；不要在 set_hooks 后移动 Trainer
    // 再依赖"捕获自身"的钩子。
    void set_hooks(TextTrainerHooks hooks) { hooks_ = std::move(hooks); }

    [[nodiscard]] Result<TextTrainReport> fit();

    // ── 访问器（打印/门禁/测试）─────────────────────────────────────────
    [[nodiscard]] TrainingSession& session() noexcept { return session_; }
    [[nodiscard]] const TrainingSession& session() const noexcept { return session_; }
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *tokenizer_; }
    [[nodiscard]] std::size_t sample_count() const noexcept { return train_.sample_count(); }
    [[nodiscard]] std::size_t steps_per_epoch() const noexcept { return steps_per_epoch_; }
    [[nodiscard]] std::size_t micro_batch_size() const noexcept { return micro_batch_size_; }
    [[nodiscard]] const TextSampleSet& train_samples() const noexcept { return train_; }
    [[nodiscard]] const TextSampleSet* test_samples() const noexcept
    { return test_ ? &*test_ : nullptr; }
    [[nodiscard]] const DatasetIdentity& dataset_identity() const noexcept { return identity_; }
    [[nodiscard]] const TrainState& train_state() const noexcept { return train_state_; }
    [[nodiscard]] const TextTrainConfig& config() const noexcept { return cfg_; }

private:
    struct PendingLoss
    {
        std::size_t slot = 0;
        std::size_t step = 0;
        std::size_t valid = 0;
        Scalar inv_num_valid = Scalar{0};
        Tensor keepalive;   // 回读命令引用的张量须活到就绪
    };

    TextTrainer(TrainingSession session, TextSampleSet train, std::optional<TextSampleSet> test,
                std::unique_ptr<Tokenizer> tokenizer, TextTrainConfig cfg, TextTrainerHooks hooks,
                DatasetIdentity identity, std::size_t steps_per_epoch,
                std::size_t micro_batch_size)
        : session_(std::move(session)), train_(std::move(train)), test_(std::move(test)),
          tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)), hooks_(std::move(hooks)),
          identity_(std::move(identity)), steps_per_epoch_(steps_per_epoch),
          micro_batch_size_(micro_batch_size)
    {
        ce_loss_.set_precision_profile(cfg_.precision);
        train_state_ = make_text_train_state(
            cfg_, identity_, steps_per_epoch_,
            static_cast<std::size_t>(cfg_.start_epoch), cfg_.start_step,
            cfg_.lr, session_.step_count());
    }

    [[nodiscard]] static Result<TextTrainer>
    create_impl(TrainingSession session, TextSampleSet train, std::optional<TextSampleSet> test,
                std::unique_ptr<Tokenizer> tokenizer, TextTrainConfig cfg, TextTrainerHooks hooks)
    {
        // 形状/网格前置条件（fail-fast：训练工厂不接受无法定义 step 的配置）
        if (cfg.batch_size == 0)
            NN_FAIL("TextTrainer::create: batch_size（global batch）必须 > 0");
        if (cfg.accum_steps == 0) NN_FAIL("TextTrainer::create: accum_steps 必须 > 0");
        if (cfg.batch_size % cfg.accum_steps != 0)
            NN_FAIL("TextTrainer::create: batch_size 必须能被 accum_steps 整除（global="
                    + std::to_string(cfg.batch_size) + ", accum="
                    + std::to_string(cfg.accum_steps) + "）");
        if (cfg.log_interval == 0)
            NN_FAIL("TextTrainer::create: log_interval 必须 > 0（0 会在进度打印时除零；"
                    "关闭输出请把 hooks.out 置空）");
        if (cfg.epochs < 0) NN_FAIL("TextTrainer::create: epochs 不能为负");
        if (cfg.start_epoch < 0) NN_FAIL("TextTrainer::create: start_epoch 不能为负");
        if (cfg.start_epoch > cfg.epochs)
            NN_FAIL("TextTrainer::create: start_epoch 超出 epochs");
        if (train.sample_count() == 0) NN_FAIL("TextTrainer::create: 训练样本集为空");
        // 模型序列长度校验（发布审查 P1-1）：窗口 seq_len 必须等于模型构建时的
        // seq_len。GPTModel::forward 用 input.rows() 解释序列长度，位置编码/
        // 因果掩码却按模型 seq_len 构建（GPU 行 gather 无越界守卫 ⇒ 越界读），
        // backward 又固定用 seq_len_ ⇒ 形状错配。spec 只在 spec 工厂/加载路径
        // 上存在；手工 build_* 调用没有 spec 时无法在此校验（该缺口记入
        // v1.10.0 release note 已知限制）。
        if (const auto& spec = session.model().spec();
            spec.has_value() && spec->seq_len > 0 && cfg.seq_len != spec->seq_len)
            NN_FAIL("TextTrainer::create: 配方 seq_len 与模型 spec.seq_len 不一致（"
                    + std::to_string(cfg.seq_len) + " vs "
                    + std::to_string(spec->seq_len)
                    + "）；请让 TextTrainConfig.seq_len 与模型构建时的 seq_len 对齐");
        if (tokenizer->pad_id() != train.pad_id())
            NN_FAIL("TextTrainer::create: 样本集 pad_id 与词表不一致");
        // 空 test 子集 = 不评估（旧 CLI 语义：样本数 0 ⇒ 跳过测试评估，不终止训练）
        if (test && test->sample_count() == 0) test.reset();
        // 训练/测试必须同一窗口网格（否则 logits 形状与掩码口径不一致）
        if (test && (test->seq_len() != train.seq_len() || test->pad_id() != train.pad_id()))
            NN_FAIL("TextTrainer::create: 训练/测试样本集的窗口网格不一致"
                    "（seq_len/pad_id 必须相同）");

        // ── 配方一致性：Session 的 TrainingConfig 是优化器侧唯一真源 ───────
        // TextTrainConfig 里的重叠字段只用于"断点状态段/调度"，必须与 Session 完全
        // 一致，否则会出现"实际用 Session 的 lr、日志/断点写配方 lr"的静默错训。
        const TrainingConfig& sc = session.config();
        if (sc.optimizer != cfg.optimizer_name)
            NN_FAIL("TextTrainer::create: 优化器不一致（Session=" + sc.optimizer
                    + ", 配方=" + cfg.optimizer_name + "）");
        if (sc.lr != cfg.lr)
            NN_FAIL("TextTrainer::create: lr 不一致（Session=" + std::to_string(sc.lr)
                    + ", 配方=" + std::to_string(cfg.lr) + "）");
        if (sc.weight_decay != cfg.weight_decay)
            NN_FAIL("TextTrainer::create: weight_decay 不一致（Session="
                    + std::to_string(sc.weight_decay) + ", 配方="
                    + std::to_string(cfg.weight_decay) + "）");
        if (sc.beta1 != cfg.beta1 || sc.beta2 != cfg.beta2)
            NN_FAIL("TextTrainer::create: beta1/beta2 不一致（Session="
                    + std::to_string(sc.beta1) + "/" + std::to_string(sc.beta2)
                    + ", 配方=" + std::to_string(cfg.beta1) + "/"
                    + std::to_string(cfg.beta2) + "）");
        if (!same_precision_profile(sc.precision, cfg.precision))
            NN_FAIL("TextTrainer::create: precision profile 不一致"
                    "（Session 与配方必须用同一份；见 make_text_train_config）");
        if (sc.clip_grad_norm != cfg.max_norm)
            NN_FAIL("TextTrainer::create: 梯度裁剪配置不一致（Session="
                    + std::to_string(sc.clip_grad_norm) + ", 配方 max_norm="
                    + std::to_string(cfg.max_norm) + "）");
        if (sc.grad_accum_steps != cfg.accum_steps)
            NN_FAIL("TextTrainer::create: 梯度累积次数不一致（Session="
                    + std::to_string(sc.grad_accum_steps) + ", 配方 accum_steps="
                    + std::to_string(cfg.accum_steps) + "）");

        const std::size_t steps = train.steps_per_epoch(cfg.batch_size);
        const std::size_t micro_batch_size = cfg.batch_size / cfg.accum_steps;
        if (steps == 0) NN_FAIL("TextTrainer::create: 训练步数为 0");
        if (micro_batch_size == 0)
            NN_FAIL("TextTrainer::create: 派生 micro_batch_size 为 0");
        if (cfg.start_step >= steps)
            NN_FAIL("TextTrainer::create: start_step 超出每 epoch 步数（"
                    + std::to_string(cfg.start_step) + " >= " + std::to_string(steps) + "）");

        // 数据身份：调用方已算好的字段优先（CLI 已为断点门禁算过一次 sha256，
        // 这里不重复哈希大文件）；未给的由样本集合补齐。
        DatasetIdentity id = cfg.identity;
        if (id.path.empty()) id.path = train.path();
        if (id.sha256.empty())
        {
            NN_TRY_MSG(sha_r, "计算数据集身份失败", sha256_file_hex(id.path));
            id.sha256 = std::move(*sha_r);
        }
        id.num_tokens = train.token_count();
        id.num_docs = train.doc_count();
        id.num_samples = train.sample_count();
        id.steps_per_epoch = steps;
        id.loss_scope = train.loss_scope();

        return TextTrainer(std::move(session), std::move(train), std::move(test),
                           std::move(tokenizer), std::move(cfg), std::move(hooks), std::move(id),
                           steps, micro_batch_size);
    }

    // 断点状态的权威装配（计数器 = "下一步的位置"，epoch 边界归一化）。
    [[nodiscard]] TrainState make_train_state(std::size_t epoch, std::size_t step_done,
                                              Scalar current_lr) const
    {
        return make_text_train_state(cfg_, identity_, steps_per_epoch_, epoch, step_done,
                                     current_lr, session_.step_count());
    }

    TrainingSession session_;
    TextSampleSet train_;
    std::optional<TextSampleSet> test_;
    std::unique_ptr<Tokenizer> tokenizer_;
    CrossEntropyLoss ce_loss_;
    TextTrainConfig cfg_;
    TextTrainerHooks hooks_;
    DatasetIdentity identity_;
    std::size_t steps_per_epoch_ = 0;
    std::size_t micro_batch_size_ = 0;
    TrainState train_state_;
};

// ══════════════════════════════════════════════════════════════════════════
//  fit —— epoch × step 训练循环
// ══════════════════════════════════════════════════════════════════════════
inline Result<TextTrainReport> TextTrainer::fit()
{
    std::ostream& out = (hooks_.out != nullptr) ? *hooks_.out : std::cout;
    std::ostream& err = (hooks_.err != nullptr) ? *hooks_.err : std::cerr;
    ComputeEngine& engine = session_.engine();
    Model& model = session_.model();
    const std::size_t vocab_size = tokenizer_->vocab_size();
    const std::size_t steps_per_epoch = steps_per_epoch_;
    const auto mem_mark = [&](const char* tag)
    {
        if (hooks_.on_phase) hooks_.on_phase(tag);
    };

    // ── 测试子集概况（旧 loop 的启动打印，文案/顺序不变）─────────────────
    if (test_)
    {
        if (test_->loss_scope() == "assistant")
        {
            std::size_t t_asst = 0;
            for (unsigned char m : test_->corpus().loss_mask)
                t_asst += m;
            out << "测试集 assistant 掩码: " << t_asst << " / " << test_->token_count()
                << " tokens 参与评估\n";
        }
        out << "测试集: " << test_->doc_count() << " docs  样本数: " << test_->sample_count()
            << "  tokens: " << test_->token_count() << std::endl;
    }

    // ── 学习率调度配置（epoch 级 + step 级）──────────────────────────────
    LrScheduleConfig lr_sched_cfg;
    lr_sched_cfg.base_lr = cfg_.lr;
    lr_sched_cfg.min_lr = cfg_.min_lr;
    lr_sched_cfg.warmup_epochs = cfg_.warmup_epochs;
    lr_sched_cfg.total_epochs = cfg_.epochs;
    lr_sched_cfg.schedule = cfg_.lr_schedule;
    lr_sched_cfg.lr_per_epoch = cfg_.lr_per_epoch;
    Scalar optimizer_current_lr = cfg_.lr;
    // 优化器侧真源是 Session；create 已校验 cfg.lr 与之一致。这里再显式设一次，
    // 保证"日志/断点配方里的 lr"和优化器实际 lr 从第一步起就同源。
    session_.set_lr(cfg_.lr);

    StepLrScheduleConfig step_lr_cfg;
    step_lr_cfg.base_lr = cfg_.lr;
    step_lr_cfg.min_lr = cfg_.min_lr;
    step_lr_cfg.warmup_steps = static_cast<int>(cfg_.warmup_steps);
    step_lr_cfg.total_steps =
        static_cast<int>(steps_per_epoch * static_cast<std::size_t>(cfg_.epochs));
    step_lr_cfg.cosine = true;

    // ── 样本顺序（每 epoch shuffle；batch 是 global batch）──────────────
    std::mt19937_64 rng{std::random_device{}()};
    std::vector<std::size_t> sample_indices(train_.sample_count());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;

    // ── 可复用的 batch 装配缓冲（每步 resize 复用）────────────────────────
    TextSampleBatch batch;

    auto t_start = std::chrono::steady_clock::now();
    std::size_t steps_executed = 0;   // update steps
    std::size_t samples_executed = 0;
    std::size_t ckpt_failures = 0;
    bool stop_training = false;
    // 真实停止位置（"下一步"）：--max-steps 提前停时用它装配 TrainState，
    // 不能无条件写 (epochs,0)（否则会话状态谎报"训练已完成"）。
    std::size_t last_epoch = static_cast<std::size_t>(cfg_.start_epoch);
    std::size_t last_step_done = cfg_.start_step;

    for (int epoch = cfg_.start_epoch; epoch < cfg_.epochs; ++epoch)
    {
        // ── 学习率：epoch 级（step_cosine 由 step 级接管）─────────────────
        if (cfg_.lr_schedule != "step_cosine")
        {
            const Scalar epoch_lr = compute_epoch_lr(lr_sched_cfg, epoch);
            if (epoch_lr != optimizer_current_lr)
            {
                session_.set_lr(epoch_lr);
                optimizer_current_lr = epoch_lr;
            }
        }

        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_weighted = 0.0;   // Σ(loss × 有效 token)
        std::size_t total_valid = 0;

        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
        const std::size_t epoch_first_step = (epoch == cfg_.start_epoch) ? cfg_.start_step : 0;

        // ── 非阻塞 loss 回读（设备侧异步槽位；host 不等 GPU）───────────────
        std::deque<PendingLoss> pending_loss;
        const std::size_t loss_slots = std::max<std::size_t>(engine.scalar_readback_slots(), 1);
        std::size_t loss_slot_next = 0;

        auto harvest_loss = [&]() -> Result<void>
        {
            while (!pending_loss.empty())
            {
                Scalar sum = Scalar{0};
                NN_TRY(pr, engine.poll_scalar_readback(pending_loss.front().slot, sum));
                if (!*pr) break;   // 未就绪：留待下次（不阻塞）
                PendingLoss pl = std::move(pending_loss.front());
                pending_loss.pop_front();
                const Scalar loss = -sum * pl.inv_num_valid;
                total_weighted += loss * static_cast<Scalar>(pl.valid);
                total_valid += pl.valid;
                if ((pl.step + 1) % cfg_.log_interval == 0 || pl.step + 1 == steps_per_epoch)
                {
                    out << "\r  Epoch " << epoch + 1 << "/" << cfg_.epochs << "  step "
                        << pl.step + 1 << "/" << steps_per_epoch << "  loss: " << std::fixed
                        << std::setprecision(4) << loss << "   " << std::flush;
                }
            }
            return {};
        };
        auto drain_loss = [&]() -> Result<void>
        {
            while (!pending_loss.empty())
            {
                NN_TRY_CHECK(harvest_loss());
                if (pending_loss.empty()) break;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            return {};
        };

        for (std::size_t step = epoch_first_step; step < steps_per_epoch; ++step)
        {
            // ── step 级学习率（step_cosine 模式）──────────────────────────
            if (cfg_.lr_schedule == "step_cosine")
            {
                const int global_step =
                    static_cast<int>(epoch) * static_cast<int>(steps_per_epoch) +
                    static_cast<int>(step);
                const Scalar step_lr = compute_step_lr(step_lr_cfg, global_step);
                if (step_lr != optimizer_current_lr)
                {
                    session_.set_lr(step_lr);
                    optimizer_current_lr = step_lr;
                }
            }

            // ── --max-steps：达到上限的那一步为最后一步，并强制完成参数更新 ──
            const bool is_final_step =
                (cfg_.max_steps > 0) && (steps_executed + 1 >= cfg_.max_steps);
            ++steps_executed;

            NN_PROF_SCOPE("step");
            nn::prof::Phase prof;
            prof.start("data");

            // ── 取一个理论 global batch（末批可能不满）────────────────────
            const std::size_t offset = step * cfg_.batch_size;
            const std::size_t global_bs =
                std::min(cfg_.batch_size, train_.sample_count() - offset);
            const auto global_indices =
                std::span<const std::size_t>(sample_indices.data() + offset, global_bs);
            NN_TRY_MSG(norm_r, "统计 global batch 有效 token 数失败",
                       train_.valid_target_count(global_indices));
            const std::size_t normalization_units = *norm_r;
            if (normalization_units == 0)
                NN_FAIL("TextTrainer::fit: 当前 global batch 没有任何有效 loss token");

            samples_executed += global_bs;
            const std::size_t micro_count =
                (global_bs + micro_batch_size_ - 1) / micro_batch_size_;

            bool updated_this_step = false;
            for (std::size_t micro = 0; micro < micro_count; ++micro)
            {
                const std::size_t micro_offset = micro * micro_batch_size_;
                const std::size_t this_micro_bs =
                    std::min(micro_batch_size_, global_bs - micro_offset);
                const auto micro_indices = global_indices.subspan(micro_offset, this_micro_bs);
                NN_TRY_MSG(
                    batch_r, "装配训练 micro-batch 失败",
                    train_.make_batch(micro_indices, batch));
                model.set_doc_ids(batch.doc_ids);
                if (hooks_.on_batch_ready) hooks_.on_batch_ready(session_, batch, step);

                // Matrix → Tensor 必须在 Session 开始录制之前完成。
                NN_TRY_MSG(x_tensor_r, "\nfrom_matrix(x_tokens) failed",
                           engine.from_matrix(batch.input));
                mem_mark("micro/ready");

                TrainingStepOptions options;
                options.lr = optimizer_current_lr;
                options.clip_grad_norm = cfg_.max_norm;
                options.normalization_units = normalization_units;
                options.force_update = (micro + 1 == micro_count) || is_final_step;
                if (hooks_.on_probe_check)
                {
                    options.before_update = [&]() -> Result<bool> {
                        return hooks_.on_probe_check(session_, step);
                    };
                }
                if (hooks_.on_update)
                {
                    options.after_update = [&]() -> Result<void> {
                        hooks_.on_update(session_, static_cast<std::size_t>(epoch), step,
                                         optimizer_current_lr);
                        return {};
                    };
                }

                prof.start(micro == 0 ? "forward" : "micro");
                NN_TRY_MSG(step_r, "TrainingSession::train_step 失败",
                           session_.train_step(
                               batch, std::move(options),
                               [&](Model& step_model, ComputeEngine& step_engine,
                                   const TextSampleBatch& micro_batch)
                                   -> Result<TrainingStepOutput> {
                                   auto logits_r = step_model.forward(*x_tensor_r);
                                   NN_TRY_CHECK(logits_r);
                                   Tensor logits = std::move(*logits_r);
                                   if (logits.rows() != vocab_size)
                                       NN_FAIL("TextTrainer::fit: 模型 logits 行数（"
                                               + std::to_string(logits.rows())
                                               + "）≠ 数据集词表大小（"
                                               + std::to_string(vocab_size)
                                               + "）；模型与数据集词表不匹配");

                                   auto mask_span =
                                       std::span<const Scalar>(micro_batch.flat_mask);
                                   std::size_t loss_units = 0;
                                   NN_TRY_MSG(
                                       loss_sum_r, "损失前向（稀疏 CE）失败",
                                       ce_loss_.forward_sparse_sum(
                                           step_engine, logits, micro_batch.flat_targets,
                                           mask_span, vocab_size, loss_units,
                                           /*grad_reuse=*/&logits));
                                   NN_TRY(grad_r, ce_loss_.backward());
                                   TrainingStepOutput out;
                                   out.grad = std::move(*grad_r);
                                   out.gradient_units = loss_units;
                                   out.loss_sum = std::move(*loss_sum_r);
                                   out.loss_units = loss_units;
                                   return out;
                               }));
                mem_mark("micro/backward");
                NN_TRY_MSG(rel_r, "\n显存回收失败", engine.release_idle_pool_blocks());

                if (step_r->aborted)
                {
                    TextTrainReport aborted;
                    aborted.steps_executed = steps_executed;
                    aborted.samples_executed = samples_executed;
                    aborted.ckpt_failures = ckpt_failures;
                    aborted.abort_code = 9;
                    aborted.final_lr = optimizer_current_lr;
                    aborted.final_epoch = static_cast<std::size_t>(epoch);
                    aborted.final_step_in_epoch = step + 1;
                    hooks_ = {};
                    return aborted;
                }

                if (!step_r->parameters_updated) continue;

                NN_TRY_MSG(hv0, "harvest_loss failed", harvest_loss());
                if (pending_loss.size() >= loss_slots)
                {
                    NN_TRY_MSG(hv1, "drain_loss failed", drain_loss());
                }
                if (!step_r->loss_sum.valid() || step_r->loss_units == 0)
                    NN_FAIL("TextTrainer::fit: update 完成但 loss_sum/loss_units 缺失");
                NN_TRY_MSG(loss_sum_f32_r, "loss 回读 cast 失败",
                           engine.cast(step_r->loss_sum, Precision::F32));

                PendingLoss pl;
                pl.slot = loss_slot_next;
                pl.step = step;
                pl.valid = step_r->loss_units;
                pl.inv_num_valid =
                    Scalar{1} / static_cast<Scalar>(step_r->loss_units);
                pl.keepalive = std::move(*loss_sum_f32_r);
                NN_TRY_MSG(sr, "submit_scalar_readback failed",
                           engine.submit_scalar_readback(pl.slot, pl.keepalive));
                loss_slot_next = (loss_slot_next + 1) % loss_slots;
                pending_loss.push_back(std::move(pl));
                updated_this_step = true;
                break;
            }
            if (!updated_this_step)
                NN_FAIL("TextTrainer::fit: global batch 结束时未完成参数更新");

            NN_TRY_MSG(hv, "harvest_loss failed", harvest_loss());

            // ── 定期断点（save_interval 以 update step 计）────────────────
            if (cfg_.save_interval > 0 &&
                ((step + 1) % cfg_.save_interval == 0 || step + 1 == steps_per_epoch))
            {
                prof.start("ckpt");
                train_state_ = make_train_state(static_cast<std::size_t>(epoch), step + 1,
                                                optimizer_current_lr);
                if (hooks_.on_checkpoint)
                {
                    const TextCheckpointRequest req{session_, train_state_};
                    if (!hooks_.on_checkpoint(req)) ++ckpt_failures;
                }
            }

            last_epoch = static_cast<std::size_t>(epoch);
            last_step_done = step + 1;

            if (is_final_step)
            {
                stop_training = true;
                break;
            }
        }

        // ── epoch 收尾：读完最后几步尚未取回的 loss ───────────────────────
        NN_TRY_MSG(ep_drain, "loss drain failed", drain_loss());

        if (stop_training)
        {
            out << "\n已达 --max-steps=" << cfg_.max_steps << "（本进程执行 " << steps_executed
                << " 步），提前结束训练\n";
            break;
        }

        auto ep_end = std::chrono::steady_clock::now();
        const Scalar ep_sec = std::chrono::duration<Scalar>(ep_end - ep_start).count();
        const Scalar avg_loss =
            (total_valid > 0) ? total_weighted / static_cast<Scalar>(total_valid) : Scalar{0};

        out << "\r  Epoch " << epoch + 1 << "/" << cfg_.epochs << "  lr=" << std::scientific
            << std::setprecision(4) << optimizer_current_lr << "  avg_loss=" << std::fixed
            << std::setprecision(4) << avg_loss << "  time=" << std::setprecision(1) << ep_sec
            << "s";

        // ── 测试集评估（可选；与训练一致的滑动窗口 + doc 感知掩码）────────
        // 语义与旧 CLI 一致：测试评估失败**不中止训练**（打印到 err 后放弃本轮
        // 剩余 test batch，训练继续并照常保存模型）。
        if (test_)
        {
            Scalar test_total_weighted = 0.0;
            std::size_t test_total_valid = 0;
            const std::size_t test_bs = std::min(micro_batch_size_, test_->sample_count());
            const std::size_t test_steps = (test_->sample_count() + test_bs - 1) / test_bs;

            for (std::size_t tstep = 0; tstep < test_steps; ++tstep)
            {
                const std::size_t toffset = tstep * test_bs;
                const std::size_t this_bs = std::min(test_bs, test_->sample_count() - toffset);
                // 测试集按样本顺序逐批评估（不沿用训练顺序）
                std::vector<std::size_t> test_indices(this_bs);
                for (std::size_t i = 0; i < this_bs; ++i)
                    test_indices[i] = toffset + i;
                if (const auto br = test_->make_batch(
                        std::span<const std::size_t>(test_indices), batch);
                    !br)
                {
                    err << "  测试装配 batch 失败: " << br.error().message << '\n';
                    break;
                }
                model.set_doc_ids(batch.doc_ids);

                auto tx_r = engine.from_matrix(batch.input);
                if (!tx_r)
                {
                    err << "  测试 from_matrix 失败: " << tx_r.error().message << '\n';
                    break;
                }
                auto tb_r = engine.begin_batch();
                if (!tb_r)
                {
                    err << "  测试 begin_batch 失败: " << tb_r.error().message << '\n';
                    break;
                }
                auto tf_r = model.forward(*tx_r);
                if (!tf_r)
                {
                    err << "  测试前向传播出错: " << tf_r.error().message << '\n';
                    (void)engine.end_batch();
                    break;
                }
                auto tlogits = std::move(*tf_r);
                auto tmask = std::span<const Scalar>(batch.flat_mask);
                auto tl_r = ce_loss_.forward_sparse(engine, tlogits, batch.flat_targets, tmask,
                                                    vocab_size);
                if (!tl_r)
                {
                    err << "  测试评估出错: " << tl_r.error().message << '\n';
                    (void)engine.end_batch();
                    break;
                }
                auto te_r = engine.end_batch();
                if (!te_r)
                {
                    err << "  测试 end_batch 失败: " << te_r.error().message << '\n';
                    break;
                }

                const Scalar batch_loss = *tl_r;
                test_total_weighted += batch_loss * static_cast<Scalar>(batch.valid_tokens);
                test_total_valid += batch.valid_tokens;
            }

            const Scalar test_avg_loss =
                (test_total_valid > 0) ? test_total_weighted / static_cast<Scalar>(test_total_valid)
                                       : Scalar{0};
            out << "  test_loss=" << std::fixed << std::setprecision(4) << test_avg_loss;
        }

        out << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    // Run state = "下一步位置"：正常跑完所有 epoch ⇒ (epochs, 0)；
    // --max-steps 提前停 ⇒ 真实停止位置（last_epoch/last_step_done），绝不谎报完成。
    if (stop_training)
        train_state_ =
            make_train_state(last_epoch, last_step_done, optimizer_current_lr);
    else
        train_state_ =
            make_train_state(static_cast<std::size_t>(cfg_.epochs), 0, optimizer_current_lr);
    TextTrainReport report;
    report.seconds = std::chrono::duration<Scalar>(t_end - t_start).count();
    report.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    report.steps_executed = steps_executed;
    report.samples_executed = samples_executed;
    report.ckpt_failures = ckpt_failures;
    report.final_lr = optimizer_current_lr;
    report.final_epoch = static_cast<std::size_t>(train_state_.counters.epoch);
    report.final_step_in_epoch =
        static_cast<std::size_t>(train_state_.counters.step_in_epoch);
    hooks_ = {};   // hooks 的契约 = 只活到 fit 返回（见 set_hooks 注释）
    return report;
}

} // namespace nn
