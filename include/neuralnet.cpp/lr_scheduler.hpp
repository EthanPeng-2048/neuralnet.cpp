// ── lr_scheduler.hpp — 学习率调度（epoch 级 + step 级）─────────────────────
//
// 从 cli/cli_lr_scheduler.hpp 上收到库层（P2b）：训练驱动（TextTrainer，L5 库内
// 编排）与 CLI 入口（mnist_train）共用同一实现，避免"库内再抄一份调度"。
// `cli/cli_lr_scheduler.hpp` 保留为转发头（旧 include 点零改动）。
//
// 调度优先级（从高到低）：
//   1. lr_per_epoch 非空 → **直接索引，完全覆盖下述任何 schedule**
//      （见下方"逃生舱"说明；epoch 超出范围时钳到最后一个值）
//   2. schedule ∈ {cosine, multistep, step_cosine} → 线性 warmup + 对应退火
//   3. 其他（"constant"/"fixed"）→ 固定 base_lr
//
// 退火形式：
//   cosine      线性 warmup + 余弦退火到 min_lr（epoch 级）
//   multistep   线性 warmup + 每过一个 milestone 乘一次 gamma（ImageNet 风格）
//   step_cosine 线性 warmup + 分段余弦 + 每段末重启回 base_lr（SGDR 风格）
// 余弦退火使用 0.5 * (1 + cos(π * progress)) 的标准形式。
//
// ⚠ warmup 对**所有** schedule 统一生效（含 multistep / step_cosine）：
//   warmup 只占 epoch ∈ [0, warmup_epochs)，退火进度按"扣掉 warmup 后的
//   有效区间"计算，因此 warmup 结束点正好衔接退火起点。
//
// ── lr_per_epoch 逃生舱（优先级最高的覆盖语义）──────────────────────
// `lr_per_epoch` **优先级最高**：只要非空就逐 epoch 直接取值，`schedule` /
// `warmup_epochs` / `base_lr` / `min_lr` / `total_epochs` **全部被忽略**。
// 适用：手算的分段/自定义曲线（如三段余弦 + 重启）无法用内建 schedule 表达时，
// 先用任意工具算出逐 epoch 表塞进来。epoch ≥ size() 时返回最后一个值（不再变化）。
// 注意它是 **epoch 级**表，无法表达 epoch 内的 lr 变化；需要更细粒度请用
// `StepLrScheduleConfig`（step 级）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <neuralnet.cpp/core_config.hpp>   // nn::Scalar

namespace nn
{
// ── 学习率调度配置（epoch 级）────────────────────────────────────────────
struct LrScheduleConfig
{
    Scalar base_lr = 0.001f;          // 基础学习率（fixed 模式直接返回）
    Scalar min_lr = 0.0f;             // 余弦退火最低学习率
    int warmup_epochs = 0;            // 线性预热轮数（0 = 不预热；对所有 schedule 生效）
    int total_epochs = 10;            // 总训练轮数（用于 cosine/step_cosine 进度计算）
    // "constant"/"fixed"（固定）| "cosine"（余弦退火）|
    // "multistep"（milestones 逐个 × gamma）| "step_cosine"（分段余弦 + 重启）
    std::string schedule = "constant";
    // multistep：每过一个 milestone（epoch 号，1-based 语义见 compute_epoch_lr）
    // 就把 lr 乘一次 gamma。空 = 退化为"预热后恒定 base_lr"。
    std::vector<int> milestones;
    Scalar gamma = 0.1f;              // multistep 每个 milestone 的衰减因子
    // step_cosine：把扣掉 warmup 后的有效区间等分为 n_segments 段，
    // 每段内余弦退火到 min_lr，**段末重启**回 base_lr（最后一段不重启，停在 min_lr）。
    int n_segments = 1;               // 段数（1 = 普通 cosine；>=2 = SGDR 重启）
    // ⚠ 逃生舱：非空则**完全覆盖**上方所有字段（优先级最高），见文件头说明。
    std::vector<Scalar> lr_per_epoch;
};

// ── 计算第 epoch 的学习率 ────────────────────────────────────────────────
// 调度优先级：lr_per_epoch > cosine > constant
[[nodiscard]] inline Scalar compute_epoch_lr(const LrScheduleConfig& cfg, int epoch)
{
    // 手动指定优先
    if (!cfg.lr_per_epoch.empty())
    {
        if (epoch < static_cast<int>(cfg.lr_per_epoch.size())) return cfg.lr_per_epoch[epoch];
        return cfg.lr_per_epoch.back();   // 超出部分用最后一个
    }

    // ── 线性 warmup：对所有 schedule 统一生效 ───────────────────────────
    // epoch ∈ [0, warmup_epochs) 时线性爬升到 base_lr；退火进度随后按
    // "扣掉 warmup 后的有效区间"计算（warmup 结束点 == 退火起点）。
    const int warm = cfg.warmup_epochs > 0 ? cfg.warmup_epochs : 0;
    if (warm > 0 && epoch < warm)
    {
        return cfg.base_lr * static_cast<Scalar>(epoch + 1) / static_cast<Scalar>(warm);
    }

    const Scalar max_lr = cfg.base_lr;
    const Scalar lr_min = cfg.min_lr;
    // 退火有效区间（扣掉 warmup）；epoch 相对退火起点的偏移
    const int anneal_span = cfg.total_epochs - warm;
    const int rel = epoch - warm;

    if (cfg.schedule == "cosine")
    {
        if (anneal_span <= 0) return max_lr;
        const Scalar progress = static_cast<Scalar>(rel) / static_cast<Scalar>(anneal_span);
        // 钳制：epoch 超出 total_epochs（如 TDR 重启续训）时停在 min_lr，
        // 与 compute_step_lr 一致（不钳制时 cos 回升 → lr 振荡）
        const Scalar p = progress < 0 ? Scalar{0} : (progress > 1 ? Scalar{1} : progress);
        return lr_min + 0.5f * (max_lr - lr_min) * (1.0f + std::cos(3.14159265358979323846f * p));
    }

    // ── multistep：每过一个 milestone 就 × gamma 一次（ImageNet 风格）──────
    // milestones 存的是 **epoch 号**（1-based 触发点）：epoch >= m 视为已过。
    // 例 milestones={30,60,90}, gamma=0.1 → epoch<30 恒 base_lr；
    // 30<=epoch<60 为 base_lr*0.1；…。空 milestones = 预热后恒定。
    if (cfg.schedule == "multistep")
    {
        Scalar lr = max_lr;
        const Scalar g = cfg.gamma;
        for (int m : cfg.milestones)
        {
            if (epoch >= m) lr *= g;
            else break;   // milestones 升序约定；遇到首个未到即停
        }
        return lr;
    }

    // ── step_cosine：分段余弦 + 段末重启（SGDR 风格）─────────────────────
    // 把退火有效区间等分 n_segments 段：段内余弦退火到 min_lr，段末重启回
    // base_lr（最后一段不重启，停在 min_lr）。n_segments=1 时 == 普通 cosine。
    if (cfg.schedule == "step_cosine")
    {
        if (anneal_span <= 0) return max_lr;
        int segs = cfg.n_segments > 0 ? cfg.n_segments : 1;
        // 每段长度（整数除法；余数摊到最后一段，保证段内进度单调到 1）
        const int seg_len = anneal_span / segs;
        // 当前段号与段内偏移；epoch 超出末段 → 停在最后一段终点（min_lr）
        // seg_len = 每段基准长度；末段吸收余数（长度 = anneal_span - (segs-1)*seg_len），
        // 保证末段覆盖到 epoch == anneal_span（lr 精确落在 min_lr）。
        int seg, off, this_len;
        if (seg_len <= 0)
        {
            seg = 0; off = 0; this_len = anneal_span > 0 ? anneal_span : 1;
        }
        else if (rel >= anneal_span)
        {
            // 越界（如续训超出 total_epochs）：钳到末段末 → min_lr（不回升）
            seg = segs - 1;
            this_len = anneal_span - (segs - 1) * seg_len;
            if (this_len <= 0) this_len = 1;
            off = this_len;   // 段内进度 = 1 → min_lr
        }
        else
        {
            // 末段吸收余数：前 (segs-1) 段各 seg_len，末段 = 剩余全部。
            // 用"末段起点"判定段号，避免整数除法把余数挤到末段末尾。
            const int last_start = (segs - 1) * seg_len;
            if (rel >= last_start)
            {
                seg = segs - 1;
                this_len = anneal_span - last_start;
                off = rel - last_start;
            }
            else
            {
                seg = rel / seg_len;
                this_len = seg_len;
                off = rel - seg * seg_len;
            }
            if (this_len <= 0) this_len = 1;
            if (off > this_len) off = this_len;
        }
        const Scalar progress = static_cast<Scalar>(off) / static_cast<Scalar>(this_len);
        const Scalar p = progress < 0 ? Scalar{0} : (progress > 1 ? Scalar{1} : progress);
        const Scalar annealed = lr_min + 0.5f * (max_lr - lr_min) *
                                (1.0f + std::cos(3.14159265358979323846f * p));
        // 最后一段不重启（停在 min_lr）；其余段末重启回 base_lr（下一段起点）
        if (seg + 1 >= segs) return annealed;
        // 段内最后一个 epoch（off == this_len-1）即"段末"，已 annealed 到接近
        // min_lr；下一 epoch 进入新段，progress 归 0 → 自然回到 base_lr。
        // 这里在段末直接返回 annealed（≈min_lr），重启发生在跨段的第一步。
        return annealed;
    }

    return cfg.base_lr;   // fixed / constant（未匹配的 schedule 兜底）
}

// ── Step 级学习率调度配置 ────────────────────────────────────────────────
// 与 epoch 级调度互补：epoch 级在大量步数（如 TinyStories 60MB 每 epoch
// 数千步）下粒度太粗，step 级可按单个训练步做 warmup + 余弦退火。
struct StepLrScheduleConfig
{
    Scalar base_lr = 0.001f;   // 峰值学习率
    Scalar min_lr = 0.0f;      // 余弦退火最低学习率
    int warmup_steps = 0;      // 线性预热步数（0 = 不预热）
    int total_steps = 1;       // 总训练步数（用于 cosine 进度）
    bool cosine = false;       // true: 预热后余弦退火到 min_lr；
                               // false: 预热后保持 base_lr
};

// ── 计算第 step 步的学习率（step 为 0-based 全局步号）────────────────────
// 预热阶段：base_lr * (step+1)/warmup_steps（线性上升）
// 预热后：
//   cosine=true → 标准余弦退火到 min_lr
//   cosine=false → 保持 base_lr
[[nodiscard]] inline Scalar compute_step_lr(const StepLrScheduleConfig& cfg, int step)
{
    const Scalar max_lr = cfg.base_lr;

    // 线性预热
    if (cfg.warmup_steps > 0 && step < cfg.warmup_steps)
    {
        return max_lr * static_cast<Scalar>(step + 1) / static_cast<Scalar>(cfg.warmup_steps);
    }

    // 余弦退火（标准形式 0.5*(1+cos(π*progress))）
    if (cfg.cosine)
    {
        const int cosine_steps = cfg.total_steps - cfg.warmup_steps;
        if (cosine_steps <= 0) return max_lr;
        const Scalar progress =
            static_cast<Scalar>(step - cfg.warmup_steps) / static_cast<Scalar>(cosine_steps);
        const Scalar p = progress < 0 ? Scalar{0} : (progress > 1 ? Scalar{1} : progress);
        return cfg.min_lr +
               0.5f * (max_lr - cfg.min_lr) * (1.0f + std::cos(3.14159265358979323846f * p));
    }

    return max_lr;   // 预热后恒定
}
} // namespace nn
