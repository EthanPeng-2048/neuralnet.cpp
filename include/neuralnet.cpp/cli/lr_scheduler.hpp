// ── lr_scheduler.hpp — 学习率调度器 ─────────────────────────────────────────
//
// 抽取自 mnist_train/text_train 中逐字重复的 compute_epoch_lr lambda。
//
// 调度优先级：
//   1. lr_per_epoch 非空 → 直接索引；epoch 超出范围时使用最后一个值
//   2. schedule == "cosine" → 线性 warmup + 余弦退火
//   3. 其他 → 固定 base_lr
//
// 注：cosine 退火使用 0.5 * (1 + cos(π * progress)) 的标准形式。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "neuralnet.cpp/core_config.hpp"  // nn::Scalar

namespace nn::cli
{
    // ── 学习率调度配置 ─────────────────────────────────────────────────────
    struct LrScheduleConfig
    {
        nn::Scalar base_lr = 0.001f;          // 基础学习率（fixed 模式直接返回）
        nn::Scalar min_lr = 0.0f;             // 余弦退火最低学习率
        int warmup_epochs = 0;                // 线性预热轮数（0 = 不预热）
        int total_epochs = 10;                // 总训练轮数（用于 cosine 进度计算）
        std::string schedule = "constant";    // "constant" | "cosine"
        // 可选：每 epoch 手动指定 lr（优先级最高）
        std::vector<nn::Scalar> lr_per_epoch;
    };

    // ── 计算第 epoch 的学习率 ──────────────────────────────────────────────
    // 调度优先级：lr_per_epoch > cosine > constant
    [[nodiscard]] inline nn::Scalar compute_epoch_lr(const LrScheduleConfig &cfg, int epoch)
    {
        // 手动指定优先
        if (!cfg.lr_per_epoch.empty())
        {
            if (epoch < static_cast<int>(cfg.lr_per_epoch.size()))
                return cfg.lr_per_epoch[epoch];
            return cfg.lr_per_epoch.back();  // 超出部分用最后一个
        }

        if (cfg.schedule == "cosine")
        {
            const nn::Scalar max_lr = cfg.base_lr;
            const nn::Scalar lr_min = cfg.min_lr;
            if (cfg.warmup_epochs > 0 && epoch < cfg.warmup_epochs)
            {
                // 线性预热
                return max_lr * static_cast<nn::Scalar>(epoch + 1) /
                       static_cast<nn::Scalar>(cfg.warmup_epochs);
            }
            // 余弦退火
            const int cosine_epochs = cfg.total_epochs - cfg.warmup_epochs;
            if (cosine_epochs <= 0) return max_lr;
            const nn::Scalar progress =
                static_cast<nn::Scalar>(epoch - cfg.warmup_epochs) /
                static_cast<nn::Scalar>(cosine_epochs);
            return lr_min + 0.5f * (max_lr - lr_min) *
                                (1.0f + std::cos(3.14159265358979323846f * progress));
        }

        return cfg.base_lr;  // fixed / constant
    }

    // ── Step 级学习率调度配置 ────────────────────────────────────────────
    // 与 epoch 级调度互补：epoch 级在大量步数（如 TinyStories 60MB 每 epoch
    // 数千步）下粒度太粗，step 级可按单个训练步做 warmup + 余弦退火。
    struct StepLrScheduleConfig
    {
        nn::Scalar base_lr = 0.001f;   // 峰值学习率
        nn::Scalar min_lr = 0.0f;      // 余弦退火最低学习率
        int warmup_steps = 0;          // 线性预热步数（0 = 不预热）
        int total_steps = 1;           // 总训练步数（用于 cosine 进度）
        bool cosine = false;           // true: 预热后余弦退火到 min_lr；
                                       // false: 预热后保持 base_lr
    };

    // ── 计算第 step 步的学习率（step 为 0-based 全局步号） ──────────────
    // 预热阶段：base_lr * (step+1)/warmup_steps（线性上升）
    // 预热后：
    //   cosine=true → 标准余弦退火到 min_lr
    //   cosine=false → 保持 base_lr
    [[nodiscard]] inline nn::Scalar compute_step_lr(const StepLrScheduleConfig &cfg, int step)
    {
        const nn::Scalar max_lr = cfg.base_lr;

        // 线性预热
        if (cfg.warmup_steps > 0 && step < cfg.warmup_steps)
        {
            return max_lr * static_cast<nn::Scalar>(step + 1) /
                   static_cast<nn::Scalar>(cfg.warmup_steps);
        }

        // 余弦退火（标准形式 0.5*(1+cos(π*progress))）
        if (cfg.cosine)
        {
            const int cosine_steps = cfg.total_steps - cfg.warmup_steps;
            if (cosine_steps <= 0) return max_lr;
            const nn::Scalar progress =
                static_cast<nn::Scalar>(step - cfg.warmup_steps) /
                static_cast<nn::Scalar>(cosine_steps);
            const nn::Scalar p =
                progress < 0 ? nn::Scalar{0} : (progress > 1 ? nn::Scalar{1} : progress);
            return cfg.min_lr + 0.5f * (max_lr - cfg.min_lr) *
                                    (1.0f + std::cos(3.14159265358979323846f * p));
        }

        return max_lr;  // 预热后恒定
    }
} // namespace nn::cli
