#pragma once

// ── train_state.hpp — 训练断点状态（`.nnckpt` 的 role 门控状态段；P1-2）────
//
// 纯数据 + KeyValueRecord 编解码（无文件 I/O、无引擎依赖）。
//
// RFC 32（issue #39）：状态域扩展点。`TrainState` 是**虚基类**，承载领域无关
// 字段（计数器 / 训练配方 / 优化器指针）与**唯一**的编解码骨架（写满、读缺省）；
// 域子类（`TextTrainState` 等）持有域特化字段并覆写三个钩子：
//   · `state_kind()`      —— 落盘键 `state_kind`；旧文件缺键 ⇒ "text"（历史
//                           断点全部是文本训练）；未知 kind ⇒ 硬报错；
//   · `write_extra_kv()`  —— 域字段落盘（在通用字段之后追加）；
//   · `read_extra_kv()`   —— 域字段读回（缺失键 → 保持默认）；
//   · `verify_extra()`    —— 域级续训交叉校验（见 `verify_resume`）。
//
// 内容分类（基类 + 文本子类合计）：
//   * **计数器**：global_step / epoch / epoch 内步号 / 断点时刻的 lr；
//   * **优化器指针**：`optimizer_step`（= Adam 的 `t`，偏置修正计数）；
//     张量本体（m/v/momentum…）紧随本段之后按参数条目落盘（数量见
//     `optimizer_blob_count`）；
//   * **训练配方**：精度 profile 四槽 / lr 调度配方与**曲线形状** /
//     batch_size、accum_steps、max_norm；
//   * **文本域**（`TextTrainState`）：滑窗 stride + 数据集身份（`.nndataset`
//     文件 sha256 + tokens/docs/**样本数**；旧 rc0 断点键 `dataset_windows`
//     兼容读 / steps_per_epoch / loss_scope）。
//
// 为什么这些必须随断点走（而不是续训时由 CLI 再给一遍）：docs/todo/18 §4.2
// P1-2、docs/todo/23 §2.1.1 —— ①精度 profile 今天完全不进文件，f16 断点忘带
// `--f16` 会被静默重建成 f32；②数据集 / `--epochs` / `--batch-size` 三者都能
// 改变 lr 曲线形状（`total_epochs`、`total_steps`、`global_step` 都从这里派生）；
// ③只存 m/v 而丢 `t` 会让偏置修正在重启后从 0 重算。
//
// 编码口径：浮点字段一律 `std::bit_cast` 成 u64 落盘（逐位无损、跨平台确定；
// Scalar 为 float 或 double 都精确）。**缺失键 = 默认值**（老文件/缺段兜底，
// 与 spec 的版本默认值同口径）。读旧文件与新文件走同一套键查找，键序变化
// （域键现在追加在通用字段之后）不影响读取。
// ─────────────────────────────────────────────────────────────────────────

#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_config.hpp>           // nn::Scalar
#include <neuralnet.cpp/core_errors.hpp>           // nn::Result / NN_FAIL / NN_TRY
#include <neuralnet.cpp/model_keyvalue_record.hpp> // nn::KeyValueRecord
#include <neuralnet.cpp/precision.hpp>             // nn::Precision / nn::PrecisionProfile

namespace nn
{
// ── 浮点字段的落盘编码（double 位模式装进 u64）──────────────────────────
[[nodiscard]] inline std::uint64_t frac_to_bits(double v) noexcept
{
    return std::bit_cast<std::uint64_t>(v);
}

[[nodiscard]] inline double bits_to_frac(std::uint64_t bits) noexcept
{
    return std::bit_cast<double>(bits);
}

// ── 计数器（续训起点）──────────────────────────────────────────────────
struct TrainCounters
{
    std::uint64_t global_step = 0;    // 跨 --resume 累计的参数更新步数
    std::uint64_t epoch = 0;          // 已完成的 epoch 数（0-based，= 下次续训的起点）
    std::uint64_t step_in_epoch = 0;  // 本 epoch 内已完成的步数（下次续训的起点）
    double lr = 0.0;                  // 断点时刻优化器的学习率（交叉校验用）
};

// ── 训练配方（形状 + 轨迹相关；断点权威；领域无关）───────────────────────
struct TrainRecipe
{
    PrecisionProfile precision{};     // 四槽：param/compute/stable/optimizer
    std::string optimizer = "adam";   // kOptimizerTable 的名字
    std::string lr_schedule = "fixed";// fixed / cosine / multistep / step_cosine
    double lr = 0.0;                  // 基准 lr（= CLI --lr）
    double min_lr = 0.0;              // 余弦退火下限
    double weight_decay = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double max_norm = 0.0;            // 0 = 不裁剪
    std::uint64_t epochs = 0;         // **曲线形状**（epoch 级调度的 total_epochs）
    std::uint64_t total_steps = 0;    // **曲线形状**（step 级调度的 total_steps）
    std::uint64_t warmup_epochs = 0;
    std::uint64_t warmup_steps = 0;
    // docs/todo/30：新状态写 global batch；旧状态缺语义键时 batch 仍按
    // micro-batch 解释。步号在新语义下均指 optimizer update。
    std::uint64_t batch_size = 0;
    std::uint64_t micro_batch_size = 0;
    bool batch_size_is_global = false;
    std::uint64_t accum_steps = 0;
    std::vector<double> lr_per_epoch;// 逐个 epoch 的手动 lr（空 = 自动调度）
};

// ── 数据集身份（内容权威；路径仅作位置回落；文本域）──────────────────────
// P2b：字段名从 `num_windows` 改为 `num_samples`（**样本**是训练单位；窗口只是
// 文本样本的落地形态），落盘键 `dataset_samples`；旧 rc0 断点的
// `dataset_windows` 仍可读（缺失时回退），写出一律用新键。
struct DatasetIdentity
{
    std::string path;                 // 位置（机器相关；resume 未给路径时的回落值）
    std::string sha256;               // 身份（`.nndataset` 文件字节的 sha256 hex）
    std::uint64_t num_tokens = 0;     // 训练子集 token 数
    std::uint64_t num_docs = 0;       // 训练子集文档数
    std::uint64_t num_samples = 0;    // 训练样本数（文本训练 = 滑动窗口数）
    std::uint64_t steps_per_epoch = 0;// = ceil(num_samples / batch_size)
    std::string loss_scope;           // all / assistant（生成期定好，随数据集）
};

// ── 数据集身份校验（文本域续训门禁；docs/todo/18 §4.2 P1-2）──────────────
// 一致判据 = **sha256 相同且网格三元组相同**。数据集一变，样本数/网格变 ⇒
// steps_per_epoch 变 ⇒ step 级调度的 total_steps、global_step 的语义、
// epoch 边界**同时**改变 ⇒ "同一断点 + 不同数据集"不存在正确的续训语义。
// 返回空串 = 一致；否则返回人类可读的差异说明（诊断面，逐项列出变了什么）。
[[nodiscard]] inline std::string dataset_mismatch_reason(
    const DatasetIdentity &saved, const DatasetIdentity &current)
{
    auto short_sha = [](const std::string &s) {
        return s.size() > 12 ? s.substr(0, 12) + "…" : s;
    };
    auto u64 = [](std::uint64_t v) { return std::to_string(v); };

    std::string why;
    auto add = [&why](const std::string &line) {
        if (!why.empty()) why += "\n";
        why += "    · " + line;
    };
    if (saved.sha256 != current.sha256)
        add("文件 sha256 : " + short_sha(saved.sha256) + " → " + short_sha(current.sha256));
    if (saved.num_tokens != current.num_tokens)
        add("token 数    : " + u64(saved.num_tokens) + " → " + u64(current.num_tokens));
    if (saved.num_docs != current.num_docs)
        add("文档数      : " + u64(saved.num_docs) + " → " + u64(current.num_docs));
    if (saved.num_samples != current.num_samples)
        add("样本数      : " + u64(saved.num_samples) + " → " + u64(current.num_samples));
    if (saved.steps_per_epoch != current.steps_per_epoch)
        add("每 epoch 步数: " + u64(saved.steps_per_epoch) + " → " +
            u64(current.steps_per_epoch));
    if (saved.loss_scope != current.loss_scope)
        add("loss 范围   : " + saved.loss_scope + " → " + current.loss_scope);
    return why;
}

// ═══════════════════════════════════════════════════════════════════════
//  TrainState：虚基类（领域无关字段 + 编解码骨架；RFC 32 §3）
// ═══════════════════════════════════════════════════════════════════════
class TextTrainState;   // 前置声明（工厂函数用）

class TrainState
{
public:
    virtual ~TrainState() = default;

    // 域状态种类（落盘键 `state_kind`；旧文件缺键 ⇒ "text"）
    [[nodiscard]] virtual const char *state_kind() const noexcept = 0;
    // 深拷贝（状态段写盘需要快照；多态值语义的支撑）
    [[nodiscard]] virtual std::unique_ptr<TrainState> clone() const = 0;

    // 域字段落盘（基类通用字段之后追加）
    virtual void write_extra_kv(KeyValueRecord & /*kv*/) const {}
    // 域字段读回（缺失键 → 保持默认）
    [[nodiscard]] virtual Result<void> read_extra_kv(const KeyValueRecord & /*kv*/)
    {
        return {};
    }
    // 域级续训交叉校验（`verify_resume` 在通用校验后调用；返回错误 = 不可续训）
    [[nodiscard]] virtual Result<void> verify_extra(const TrainState & /*current*/) const
    {
        return {};
    }

    TrainCounters counters;
    TrainRecipe recipe;
    std::uint64_t optimizer_step = 0;        // Adam `t`（偏置修正计数）
    std::uint64_t optimizer_blob_count = 0;  // 紧随本段之后的优化器状态张量条目数
};

// ── 文本域状态（现有断点的权威类型；旧文件缺 state_kind 键一律回退到这里）──
class TextTrainState : public TrainState
{
public:
    [[nodiscard]] const char *state_kind() const noexcept override { return "text"; }
    [[nodiscard]] std::unique_ptr<TrainState> clone() const override
    {
        return std::make_unique<TextTrainState>(*this);
    }

    std::uint64_t stride = 0;   // 滑动窗口步长（0 = 等于 seq_len）
    DatasetIdentity dataset;    // 数据集身份（续训门禁的数据源）

    void write_extra_kv(KeyValueRecord &kv) const override
    {
        kv.set("stride", stride);
        kv.set("dataset_path", dataset.path);
        kv.set("dataset_sha256", dataset.sha256);
        kv.set("dataset_tokens", dataset.num_tokens);
        kv.set("dataset_docs", dataset.num_docs);
        kv.set("dataset_samples", dataset.num_samples);
        kv.set("dataset_steps_per_epoch", dataset.steps_per_epoch);
        kv.set("dataset_loss_scope", dataset.loss_scope);
    }

    [[nodiscard]] Result<void> read_extra_kv(const KeyValueRecord &kv) override
    {
        std::uint64_t u = 0;
        if (kv.get("stride", u)) stride = u;
        if (!kv.get("dataset_path", dataset.path)) dataset.path.clear();
        if (!kv.get("dataset_sha256", dataset.sha256)) dataset.sha256.clear();
        if (kv.get("dataset_tokens", u)) dataset.num_tokens = u;
        if (kv.get("dataset_docs", u)) dataset.num_docs = u;
        // 新键优先；rc0 旧断点只有 dataset_windows（同一语义：窗口数 = 样本数）
        {
            std::uint64_t samples = 0;
            if (kv.get("dataset_samples", samples) || kv.get("dataset_windows", samples))
                dataset.num_samples = samples;
        }
        if (kv.get("dataset_steps_per_epoch", u)) dataset.steps_per_epoch = u;
        if (!kv.get("dataset_loss_scope", dataset.loss_scope)) dataset.loss_scope.clear();
        return {};
    }

    [[nodiscard]] Result<void> verify_extra(const TrainState &current) const override
    {
        const auto *ts = dynamic_cast<const TextTrainState *>(&current);
        if (ts == nullptr)
            NN_FAIL("续训交叉校验: 断点是文本状态（text）而本次不是文本状态（kind=" +
                    std::string(current.state_kind()) + "）");
        const std::string why = dataset_mismatch_reason(dataset, ts->dataset);
        if (!why.empty())
            NN_FAIL("续训交叉校验: 数据集身份不一致（内容身份变了 ⇒ 续训不存在正确语义）:\n" +
                    why);
        if (stride != ts->stride)
            NN_FAIL("续训交叉校验: 滑窗 stride 不一致（" + std::to_string(stride) +
                    " vs " + std::to_string(ts->stride) + "）");
        return {};
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  TrainState ⇄ KeyValueRecord（自描述编解码；写满、读缺省）
// ═══════════════════════════════════════════════════════════════════════
inline constexpr std::uint64_t TRAIN_STATE_VERSION = 1;

[[nodiscard]] inline KeyValueRecord train_state_to_kv(const TrainState &s)
{
    KeyValueRecord kv;
    kv.set("state_version", TRAIN_STATE_VERSION);
    kv.set("state_kind", std::string(s.state_kind()));

    // 计数器
    kv.set("global_step", s.counters.global_step);
    kv.set("epoch", s.counters.epoch);
    kv.set("step_in_epoch", s.counters.step_in_epoch);
    kv.set("current_lr_bits", frac_to_bits(s.counters.lr));

    // 优化器（t 是状态；名字/超参在下面的配方里）
    kv.set("optimizer", s.recipe.optimizer);
    kv.set("optimizer_step", s.optimizer_step);
    kv.set("optimizer_blob_count", s.optimizer_blob_count);

    // 配方：精度四槽（存枚举原值；读回时校验）
    kv.set("precision_param", static_cast<std::uint64_t>(s.recipe.precision.param));
    kv.set("precision_compute", static_cast<std::uint64_t>(s.recipe.precision.compute));
    kv.set("precision_stable", static_cast<std::uint64_t>(s.recipe.precision.stable));
    kv.set("precision_optimizer", static_cast<std::uint64_t>(s.recipe.precision.optimizer));
    // 配方：调度与形状
    kv.set("lr_schedule", s.recipe.lr_schedule);
    kv.set("lr_bits", frac_to_bits(s.recipe.lr));
    kv.set("min_lr_bits", frac_to_bits(s.recipe.min_lr));
    kv.set("weight_decay_bits", frac_to_bits(s.recipe.weight_decay));
    kv.set("beta1_bits", frac_to_bits(s.recipe.beta1));
    kv.set("beta2_bits", frac_to_bits(s.recipe.beta2));
    kv.set("max_norm_bits", frac_to_bits(s.recipe.max_norm));
    kv.set("epochs", s.recipe.epochs);
    kv.set("total_steps", s.recipe.total_steps);
    kv.set("warmup_epochs", s.recipe.warmup_epochs);
    kv.set("warmup_steps", s.recipe.warmup_steps);
    kv.set("batch_size", s.recipe.batch_size);
    kv.set("micro_batch_size", s.recipe.micro_batch_size);
    kv.set("batch_size_semantics",
           std::string(s.recipe.batch_size_is_global ? "global" : "micro"));
    kv.set("accum_steps", s.recipe.accum_steps);
    {
        std::vector<std::uint64_t> lrs;
        lrs.reserve(s.recipe.lr_per_epoch.size());
        for (double v : s.recipe.lr_per_epoch) lrs.push_back(frac_to_bits(v));
        kv.set("lr_per_epoch_bits", lrs);
    }

    // 域扩展（文本 = stride + 数据集身份）
    s.write_extra_kv(kv);
    return kv;
}

// 精度槽读回：只接受**存储精度**与三值（BF16/F64 是保留值，文件里出现即报错）
[[nodiscard]] inline Result<Precision> precision_from_state_kv(
    const KeyValueRecord &kv, const std::string &key, Precision fallback)
{
    std::uint64_t v = 0;
    if (!kv.get(key, v)) return fallback;
    const auto p = static_cast<Precision>(v);
    if (p != Precision::F16 && p != Precision::F32 && p != Precision::T1_58)
        NN_FAIL("训练状态段的 " + key + " 精度非法（值 " + std::to_string(v) +
                "）；BF16/F64 为保留值，T1_58/F16/F32 之外不接受");
    return p;
}

// 工厂：kind → 子类实例。旧文件缺 `state_kind` 键 ⇒ "text"（历史断点全部是
// 文本训练；零破坏）。未知 kind ⇒ 硬报错（不用错误类型的字段布局读文件）。
[[nodiscard]] inline Result<std::unique_ptr<TrainState>>
train_state_from_kv(const KeyValueRecord &kv)
{
    std::string kind = "text";
    if (!kv.get("state_kind", kind)) kind = "text";

    std::unique_ptr<TrainState> s;
    if (kind == "text")
        s = std::make_unique<TextTrainState>();
    else
        NN_FAIL("未知的训练状态 kind=\"" + kind + "\"（本构建只认识 text；"
                "自定义域请注册对应 TrainState 子类的读回路径）");

    std::uint64_t u = 0;
    if (kv.get("global_step", u)) s->counters.global_step = u;
    if (kv.get("epoch", u)) s->counters.epoch = u;
    if (kv.get("step_in_epoch", u)) s->counters.step_in_epoch = u;
    if (kv.get("current_lr_bits", u)) s->counters.lr = bits_to_frac(u);

    if (!kv.get("optimizer", s->recipe.optimizer)) s->recipe.optimizer = "adam";
    if (kv.get("optimizer_step", u)) s->optimizer_step = u;
    if (kv.get("optimizer_blob_count", u)) s->optimizer_blob_count = u;

    NN_TRY(p_param, precision_from_state_kv(
        kv, "precision_param", s->recipe.precision.param));
    s->recipe.precision.param = *p_param;
    NN_TRY(p_compute, precision_from_state_kv(
        kv, "precision_compute", s->recipe.precision.compute));
    s->recipe.precision.compute = *p_compute;
    NN_TRY(p_stable, precision_from_state_kv(
        kv, "precision_stable", s->recipe.precision.stable));
    s->recipe.precision.stable = *p_stable;
    NN_TRY(p_optimizer, precision_from_state_kv(
        kv, "precision_optimizer", s->recipe.precision.optimizer));
    s->recipe.precision.optimizer = *p_optimizer;

    if (!kv.get("lr_schedule", s->recipe.lr_schedule)) s->recipe.lr_schedule = "fixed";
    if (kv.get("lr_bits", u)) s->recipe.lr = bits_to_frac(u);
    if (kv.get("min_lr_bits", u)) s->recipe.min_lr = bits_to_frac(u);
    if (kv.get("weight_decay_bits", u)) s->recipe.weight_decay = bits_to_frac(u);
    if (kv.get("beta1_bits", u)) s->recipe.beta1 = bits_to_frac(u);
    if (kv.get("beta2_bits", u)) s->recipe.beta2 = bits_to_frac(u);
    if (kv.get("max_norm_bits", u)) s->recipe.max_norm = bits_to_frac(u);
    if (kv.get("epochs", u)) s->recipe.epochs = u;
    if (kv.get("total_steps", u)) s->recipe.total_steps = u;
    if (kv.get("warmup_epochs", u)) s->recipe.warmup_epochs = u;
    if (kv.get("warmup_steps", u)) s->recipe.warmup_steps = u;
    if (kv.get("batch_size", u)) s->recipe.batch_size = u;
    if (kv.get("micro_batch_size", u)) s->recipe.micro_batch_size = u;
    {
        std::string semantics;
        if (kv.get("batch_size_semantics", semantics))
        {
            if (semantics != "global" && semantics != "micro")
                NN_FAIL("训练状态的 batch_size_semantics 非法（只接受 global/micro）");
            s->recipe.batch_size_is_global = (semantics == "global");
        }
    }
    if (kv.get("accum_steps", u)) s->recipe.accum_steps = u;
    {
        std::vector<std::uint64_t> lrs;
        if (kv.get("lr_per_epoch_bits", lrs))
        {
            s->recipe.lr_per_epoch.clear();
            s->recipe.lr_per_epoch.reserve(lrs.size());
            for (std::uint64_t b : lrs) s->recipe.lr_per_epoch.push_back(bits_to_frac(b));
        }
    }

    NN_TRY(x, s->read_extra_kv(kv));
    return s;
}

// ── 续训交叉校验（库口径唯一；RFC 32 §3.5）──────────────────────────────
// Strict（默认）：kind 相等 + 配方关键维度相等 + 计数器不越界 + 域级
// `verify_extra()`（文本 = 数据集身份 + stride）。Relaxed：只做 kind 检查，
// 允许研究期换数据集 / 改配方（代价 = lr 曲线形状可能跨续训改变）。
enum class ResumePolicy { Strict, Relaxed };

[[nodiscard]] inline Result<void> verify_resume(const TrainState &saved,
                                                const TrainState &current,
                                                ResumePolicy policy = ResumePolicy::Strict)
{
    if (std::string(saved.state_kind()) != std::string(current.state_kind()))
        NN_FAIL("续训交叉校验: 状态 kind 不一致（" + std::string(saved.state_kind()) +
                " vs " + std::string(current.state_kind()) + "）");
    if (policy == ResumePolicy::Relaxed)
        return {};

    const TrainRecipe &a = saved.recipe;
    const TrainRecipe &b = current.recipe;
    auto mismatch = [](const char *name, const std::string &x, const std::string &y) {
        return std::string("续训交叉校验: 配方 ") + name + " 不一致（断点 \"" + x +
               "\" vs 本次 \"" + y + "\"）；lr 曲线形状必须跨续训不变";
    };
    if (a.optimizer != b.optimizer)
        NN_FAIL(mismatch("optimizer", a.optimizer, b.optimizer));
    if (a.lr_schedule != b.lr_schedule)
        NN_FAIL(mismatch("lr_schedule", a.lr_schedule, b.lr_schedule));
    if (a.epochs != b.epochs)
        NN_FAIL(mismatch("epochs", std::to_string(a.epochs), std::to_string(b.epochs)));
    if (a.total_steps != b.total_steps)
        NN_FAIL(mismatch("total_steps", std::to_string(a.total_steps),
                         std::to_string(b.total_steps)));
    if (a.warmup_epochs != b.warmup_epochs)
        NN_FAIL(mismatch("warmup_epochs", std::to_string(a.warmup_epochs),
                         std::to_string(b.warmup_epochs)));
    if (a.warmup_steps != b.warmup_steps)
        NN_FAIL(mismatch("warmup_steps", std::to_string(a.warmup_steps),
                         std::to_string(b.warmup_steps)));
    if (a.batch_size != b.batch_size)
        NN_FAIL(mismatch("batch_size", std::to_string(a.batch_size),
                         std::to_string(b.batch_size)));
    if (a.accum_steps != b.accum_steps)
        NN_FAIL(mismatch("accum_steps", std::to_string(a.accum_steps),
                         std::to_string(b.accum_steps)));
    if (a.precision.param != b.precision.param ||
        a.precision.compute != b.precision.compute ||
        a.precision.stable != b.precision.stable ||
        a.precision.optimizer != b.precision.optimizer)
        NN_FAIL(mismatch("precision", "(见断点)", "(见本次)"));

    // 计数器不越界：断点已训完的 epoch 超过本次配方的 epochs ⇒ 配方缩水
    if (b.epochs != 0 && saved.counters.epoch > b.epochs)
        NN_FAIL("续训交叉校验: 断点已训完 " + std::to_string(saved.counters.epoch) +
                " 个 epoch，超过本次配方 epochs=" + std::to_string(b.epochs));

    NN_TRY(x, saved.verify_extra(current));
    return {};
}

} // namespace nn
