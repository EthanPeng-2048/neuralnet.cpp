#pragma once

// ── train_state.hpp — 训练断点状态（`.nnckpt` 的 role 门控状态段；P1-2）────
//
// 纯数据 + KeyValueRecord 编解码（无文件 I/O、无引擎依赖）。内容四类：
//   * **计数器**：global_step / epoch / epoch 内步号 / 断点时刻的 lr；
//   * **优化器指针**：`optimizer_step`（= Adam 的 `t`，偏置修正计数）；
//     张量本体（m/v/momentum…）紧随本段之后按参数条目落盘（数量见
//     `optimizer_blob_count`）；
//   * **训练配方**：精度 profile 四槽 / lr 调度配方与**曲线形状** /
//     batch_size、accum_steps、max_norm、stride；
//   * **数据集身份**：`.nndataset` 文件 sha256 + tokens/docs/windows/
//     steps_per_epoch/loss_scope（+ 路径，仅作位置回落）。
//
// 为什么这些必须随断点走（而不是续训时由 CLI 再给一遍）：docs/todo/18 §4.2
// P1-2、docs/todo/23 §2.1.1 —— ①精度 profile 今天完全不进文件，f16 断点忘带
// `--f16` 会被静默重建成 f32；②数据集 / `--epochs` / `--batch-size` 三者都能
// 改变 lr 曲线形状（`total_epochs`、`total_steps`、`global_step` 都从这里派生）；
// ③只存 m/v 而丢 `t` 会让偏置修正在重启后从 0 重算。
//
// 编码口径：浮点字段一律 `std::bit_cast` 成 u64 落盘（逐位无损、跨平台确定；
// Scalar 为 float 或 double 都精确）。**缺失键 = 默认值**（老文件/缺段兜底，
// 与 spec 的版本默认值同口径）。
// ─────────────────────────────────────────────────────────────────────────

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "core_config.hpp"           // nn::Scalar
#include "model_keyvalue_record.hpp" // nn::KeyValueRecord
#include "precision.hpp"             // nn::Precision / nn::PrecisionProfile

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

// ── 训练配方（形状 + 轨迹相关；断点权威）───────────────────────────────
struct TrainRecipe
{
    PrecisionProfile precision{};     // 四槽：param/compute/stable/optimizer
    std::string optimizer = "adam";   // kOptimizerTable 的名字
    std::string lr_schedule = "fixed";// fixed / cosine / step_cosine
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
    std::uint64_t batch_size = 0;     // 决定 steps_per_epoch ⇒ 也决定步号语义
    std::uint64_t accum_steps = 0;
    std::uint64_t stride = 0;         // 滑动窗口步长（0 = 等于 seq_len）
    std::vector<double> lr_per_epoch;// 逐个 epoch 的手动 lr（空 = 自动调度）
};

// ── 数据集身份（内容权威；路径仅作位置回落）────────────────────────────
struct DatasetIdentity
{
    std::string path;                 // 位置（机器相关；resume 未给路径时的回落值）
    std::string sha256;               // 身份（`.nndataset` 文件字节的 sha256 hex）
    std::uint64_t num_tokens = 0;     // 训练子集 token 数
    std::uint64_t num_docs = 0;       // 训练子集文档数
    std::uint64_t num_windows = 0;    // 滑动窗口样本数
    std::uint64_t steps_per_epoch = 0;// = ceil(num_windows / batch_size)
    std::string loss_scope;           // all / assistant（生成期定好，随数据集）
};

// ── 训练状态（= 状态段的完整内容；张量本体不在本结构里）────────────────
struct TrainState
{
    TrainCounters counters;
    TrainRecipe recipe;
    DatasetIdentity dataset;
    std::uint64_t optimizer_step = 0;        // Adam `t`（偏置修正计数）
    std::uint64_t optimizer_blob_count = 0;  // 紧随本段之后的优化器状态张量条目数
};

// ═════════════════════════════════════════════════════════════════════════
//  TrainState ⇄ KeyValueRecord（自描述编解码；写满、读缺省）
// ═════════════════════════════════════════════════════════════════════════
inline constexpr std::uint64_t TRAIN_STATE_VERSION = 1;

[[nodiscard]] inline KeyValueRecord train_state_to_kv(const TrainState &s)
{
    KeyValueRecord kv;
    kv.set("state_version", TRAIN_STATE_VERSION);

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
    kv.set("accum_steps", s.recipe.accum_steps);
    kv.set("stride", s.recipe.stride);
    {
        std::vector<std::uint64_t> lrs;
        lrs.reserve(s.recipe.lr_per_epoch.size());
        for (double v : s.recipe.lr_per_epoch) lrs.push_back(frac_to_bits(v));
        kv.set("lr_per_epoch_bits", lrs);
    }

    // 数据集身份
    kv.set("dataset_path", s.dataset.path);
    kv.set("dataset_sha256", s.dataset.sha256);
    kv.set("dataset_tokens", s.dataset.num_tokens);
    kv.set("dataset_docs", s.dataset.num_docs);
    kv.set("dataset_windows", s.dataset.num_windows);
    kv.set("dataset_steps_per_epoch", s.dataset.steps_per_epoch);
    kv.set("dataset_loss_scope", s.dataset.loss_scope);
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

[[nodiscard]] inline Result<TrainState> train_state_from_kv(const KeyValueRecord &kv)
{
    TrainState s;
    std::uint64_t u = 0;

    if (kv.get("global_step", u)) s.counters.global_step = u;
    if (kv.get("epoch", u)) s.counters.epoch = u;
    if (kv.get("step_in_epoch", u)) s.counters.step_in_epoch = u;
    if (kv.get("current_lr_bits", u)) s.counters.lr = bits_to_frac(u);

    if (!kv.get("optimizer", s.recipe.optimizer)) s.recipe.optimizer = "adam";
    if (kv.get("optimizer_step", u)) s.optimizer_step = u;
    if (kv.get("optimizer_blob_count", u)) s.optimizer_blob_count = u;

    NN_TRY(p_param, precision_from_state_kv(
        kv, "precision_param", s.recipe.precision.param));
    s.recipe.precision.param = *p_param;
    NN_TRY(p_compute, precision_from_state_kv(
        kv, "precision_compute", s.recipe.precision.compute));
    s.recipe.precision.compute = *p_compute;
    NN_TRY(p_stable, precision_from_state_kv(
        kv, "precision_stable", s.recipe.precision.stable));
    s.recipe.precision.stable = *p_stable;
    NN_TRY(p_optimizer, precision_from_state_kv(
        kv, "precision_optimizer", s.recipe.precision.optimizer));
    s.recipe.precision.optimizer = *p_optimizer;

    if (!kv.get("lr_schedule", s.recipe.lr_schedule)) s.recipe.lr_schedule = "fixed";
    if (kv.get("lr_bits", u)) s.recipe.lr = bits_to_frac(u);
    if (kv.get("min_lr_bits", u)) s.recipe.min_lr = bits_to_frac(u);
    if (kv.get("weight_decay_bits", u)) s.recipe.weight_decay = bits_to_frac(u);
    if (kv.get("beta1_bits", u)) s.recipe.beta1 = bits_to_frac(u);
    if (kv.get("beta2_bits", u)) s.recipe.beta2 = bits_to_frac(u);
    if (kv.get("max_norm_bits", u)) s.recipe.max_norm = bits_to_frac(u);
    if (kv.get("epochs", u)) s.recipe.epochs = u;
    if (kv.get("total_steps", u)) s.recipe.total_steps = u;
    if (kv.get("warmup_epochs", u)) s.recipe.warmup_epochs = u;
    if (kv.get("warmup_steps", u)) s.recipe.warmup_steps = u;
    if (kv.get("batch_size", u)) s.recipe.batch_size = u;
    if (kv.get("accum_steps", u)) s.recipe.accum_steps = u;
    if (kv.get("stride", u)) s.recipe.stride = u;
    {
        std::vector<std::uint64_t> lrs;
        if (kv.get("lr_per_epoch_bits", lrs))
        {
            s.recipe.lr_per_epoch.clear();
            s.recipe.lr_per_epoch.reserve(lrs.size());
            for (std::uint64_t b : lrs) s.recipe.lr_per_epoch.push_back(bits_to_frac(b));
        }
    }

    if (!kv.get("dataset_path", s.dataset.path)) s.dataset.path.clear();
    if (!kv.get("dataset_sha256", s.dataset.sha256)) s.dataset.sha256.clear();
    if (kv.get("dataset_tokens", u)) s.dataset.num_tokens = u;
    if (kv.get("dataset_docs", u)) s.dataset.num_docs = u;
    if (kv.get("dataset_windows", u)) s.dataset.num_windows = u;
    if (kv.get("dataset_steps_per_epoch", u)) s.dataset.steps_per_epoch = u;
    if (!kv.get("dataset_loss_scope", s.dataset.loss_scope)) s.dataset.loss_scope.clear();
    return s;
}

// ═════════════════════════════════════════════════════════════════════════
//  数据集身份校验（续训门禁；docs/todo/18 §4.2 P1-2）
//
//  一致判据 = **sha256 相同且网格三元组相同**。数据集一变，窗口数变 ⇒
//  steps_per_epoch 变 ⇒ step 级调度的 total_steps、global_step 的语义、
//  epoch 边界**同时**改变 ⇒ "同一断点 + 不同数据集"不存在正确的续训语义，
//  因此续训路径必须**硬报错**（无逃生阀；出口 = 从 `.nnmodel` warm start）。
//
//  返回空串 = 一致；否则返回人类可读的差异说明（诊断面，逐项列出变了什么）。
// ═════════════════════════════════════════════════════════════════════════
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
    if (saved.num_windows != current.num_windows)
        add("窗口数      : " + u64(saved.num_windows) + " → " + u64(current.num_windows));
    if (saved.steps_per_epoch != current.steps_per_epoch)
        add("每 epoch 步数: " + u64(saved.steps_per_epoch) + " → " +
            u64(current.steps_per_epoch));
    if (saved.loss_scope != current.loss_scope)
        add("loss 范围   : " + saved.loss_scope + " → " + current.loss_scope);
    return why;
}
} // namespace nn
