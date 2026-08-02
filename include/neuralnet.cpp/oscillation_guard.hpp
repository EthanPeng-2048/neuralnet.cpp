#ifndef NN_OSCILLATION_GUARD_HPP
#define NN_OSCILLATION_GUARD_HPP

// ── oscillation_guard.hpp — 智能训练振荡抑制器 ────────────────────────────
//
// 检测训练 loss 的高频振荡，自动降低学习率以稳定训练。
//
// 检测算法：
//   维护一个滑动窗口（默认 20 步），统计 loss 差分的方向反转率：
//     reversal_ratio = |{i : Δloss_i · Δloss_{i-1} < 0}| / (window - 1)
//
//   若 reversal_ratio > threshold（默认 0.55），判定为振荡，
//   将 lr 乘以 decay_factor（默认 0.5），并进入冷却期。
//
// 设计要点：
//   - 每个 step 调用 update(loss)，O(window) 时间，零额外内存分配
//   - 冷却期内不重复降 lr，避免 lr 崩塌
//   - 支持 warmup_steps：前 N 步不检测（避免启动噪声误判）
//   - 支持 min_lr 下限，防止 lr 降至 0
//   - 纯 CPU 逻辑，与 ComputeEngine / GPU 无关
//
// 用法：
//   nn::OscillationGuard guard(/*window=*/20, /*threshold=*/0.55);
//   for (step...) {
//       Scalar loss = train_step(...);
//       guard.update(loss);
//       if (guard.should_adjust()) {
//           optimizer->set_lr(guard.current_lr());
//           std::cout << "lr reduced to " << guard.current_lr() << "\n";
//       }
//   }
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>

#include "config.hpp"  // Scalar

namespace nn
{

struct OscillationGuardConfig
{
    std::size_t window_size    = 20;     // 滑动窗口大小
    Scalar      threshold     = 0.55f;   // 反转率阈值（>此值判定振荡）
    Scalar      decay_factor  = 0.8f;    // lr 衰减因子（每次乘以此值，0.8 = 温和衰减）
    std::size_t cooldown      = 100;     // 降 lr 后的冷却步数
    Scalar      min_lr        = 1e-6f;   // lr 下限
    std::size_t warmup_steps  = 0;       // 前 N 步不检测
    bool        verbose       = true;    // 是否打印抑制信息

    // ── lr 恢复机制 ──────────────────────────────────────
    // 当振荡停止后，逐步恢复 lr，避免 lr 永久偏低
    Scalar      recovery_factor = 1.02f; // 恢复系数（每步乘以此值，1.02 = 温和恢复）
    std::size_t recovery_stable_steps = 50; // 连续多少步低反转率后开始恢复
    Scalar      recovery_threshold = 0.3f;  // 低于此反转率视为"稳定"
    std::size_t max_reductions = 5;      // 最大衰减次数，防止 lr 崩塌
};

class OscillationGuard
{
public:
    using Config = OscillationGuardConfig;

private:
    Config cfg_;

    Scalar      base_lr_;           // 初始学习率
    Scalar      current_lr_;        // 当前学习率
    std::size_t global_step_ = 0;   // 全局步计数（跨 epoch）
    std::size_t cooldown_left_ = 0; // 剩余冷却步数
    std::size_t reduction_count_ = 0; // 累计降 lr 次数

    // ── lr 恢复状态 ──────────────────────────────────────
    std::size_t stable_steps_ = 0;  // 连续稳定步数
    bool        was_reduced_ = false; // 是否曾经降过 lr（才需要恢复）

    // 滑动窗口：存储最近 window_size 个 loss 值
    std::deque<Scalar> loss_window_;

public:
    explicit OscillationGuard(Scalar initial_lr, Config cfg = {})
        : cfg_(std::move(cfg)), base_lr_(initial_lr), current_lr_(initial_lr)
    {
    }

    // ── 每 step 调用：传入当前 loss，返回是否触发了 lr 调整 ──
    // 编排：冷却/warmup → 计算反转率 → 检测+降低 → 尝试恢复
    bool update(Scalar loss)
    {
        ++global_step_;

        // 冷却期内跳过检测
        if (cooldown_left_ > 0)
        {
            --cooldown_left_;
            return false;
        }

        // warmup 期跳过检测
        if (global_step_ <= cfg_.warmup_steps)
        {
            push_loss_(loss);
            return false;
        }

        push_loss_(loss);

        // 窗口未满时不检测
        if (loss_window_.size() < cfg_.window_size)
            return false;

        // 核心：计算方向反转率
        Scalar reversal_ratio = compute_reversal_ratio_();

        // 振荡检测 + lr 降低
        if (detect_and_reduce_(reversal_ratio))
            return true;

        // lr 恢复逻辑
        try_recover_(reversal_ratio);

        return false;
    }

    // ── 访问器 ──
    [[nodiscard]] Scalar      current_lr()    const { return current_lr_; }
    [[nodiscard]] Scalar      base_lr()       const { return base_lr_; }
    [[nodiscard]] std::size_t reduction_count() const { return reduction_count_; }
    [[nodiscard]] std::size_t global_step()   const { return global_step_; }

    // ── 外部 LR 调度器设置新 base lr ──
    // 用于 per-epoch LR 调度（如 cosine annealing + warmup），
    // 重置内部状态以匹配新的 base lr，同时保留已有的冷却/衰减记忆。
    // 注意：此方法会重置状态（was_reduced_/reduction_count_ 等），并非单纯设置 lr。
    void rebase_lr(Scalar new_base_lr)
    {
        base_lr_ = new_base_lr;
        current_lr_ = new_base_lr;
        was_reduced_ = false;
        reduction_count_ = 0;
        stable_steps_ = 0;
        cooldown_left_ = 0;
    }

    // ── 诊断信息 ──
    [[nodiscard]] Scalar last_reversal_ratio() const
    {
        if (loss_window_.size() < 2) return Scalar{0};
        return compute_reversal_ratio_();
    }

    [[nodiscard]] std::string status_summary() const
    {
        std::string s = "OscillationGuard[step=" + std::to_string(global_step_)
                      + " lr=" + std::to_string(current_lr_)
                      + " reductions=" + std::to_string(reduction_count_)
                      + " window=" + std::to_string(loss_window_.size())
                      + "/" + std::to_string(cfg_.window_size) + "]";
        return s;
    }

private:
    // 振荡检测 + lr 降低，返回 true 表示触发了降 lr
    bool detect_and_reduce_(Scalar ratio)
    {
        if (!(ratio > cfg_.threshold && reduction_count_ < cfg_.max_reductions))
            return false;

        current_lr_ = std::max(current_lr_ * cfg_.decay_factor, cfg_.min_lr);
        cooldown_left_ = cfg_.cooldown;
        ++reduction_count_;
        was_reduced_ = true;
        stable_steps_ = 0;  // 重置稳定计数

        log_reduction_(ratio);

        // 缩小窗口（衰减后用更短的历史判断，避免旧数据干扰）
        shrink_window_();
        return true;
    }

    // lr 恢复逻辑：振荡停止后逐步恢复
    void try_recover_(Scalar ratio)
    {
        if (!was_reduced_ || current_lr_ >= base_lr_)
            return;

        if (ratio < cfg_.recovery_threshold)
        {
            ++stable_steps_;
            if (stable_steps_ >= cfg_.recovery_stable_steps)
            {
                Scalar old_lr = current_lr_;
                current_lr_ = std::min(current_lr_ * cfg_.recovery_factor, base_lr_);
                stable_steps_ = 0;

                if (cfg_.verbose && current_lr_ != old_lr)
                {
                    std::cout << "\n  ✓ lr 恢复 [step " << global_step_
                              << "]  " << old_lr << " → " << current_lr_
                              << std::flush;
                }

                // 恢复到 base_lr 后停止恢复
                if (current_lr_ >= base_lr_)
                {
                    current_lr_ = base_lr_;
                    was_reduced_ = false;
                }
            }
        }
        else
        {
            stable_steps_ = 0;  // 不够稳定，重置计数
        }
    }

    // verbose 打印降 lr 信息
    void log_reduction_(Scalar ratio)
    {
        if (!cfg_.verbose) return;
        std::cout << "\n  ⚠ 振荡检测 [step " << global_step_
                  << "] reversal=" << std::fixed << std::setprecision(2)
                  << ratio * 100.0 << "%"
                  << "  lr: " << current_lr_ / cfg_.decay_factor
                  << " → " << current_lr_
                  << "  (第 " << reduction_count_ << "/" << cfg_.max_reductions << " 次衰减)"
                  << std::flush;
    }

    void push_loss_(Scalar loss)
    {
        if (loss_window_.size() >= cfg_.window_size)
            loss_window_.pop_front();
        loss_window_.push_back(loss);
    }

    // 计算滑动窗口内 loss 差分的方向反转率
    //
    // 原理：正常训练中 loss 单调下降，差分 Δloss < 0；
    //       振荡时 Δloss 反复变号，反转率接近 1.0。
    //
    // reversal_ratio = |{i : Δ_i · Δ_{i-1} < 0}| / (N - 2)
    [[nodiscard]] Scalar compute_reversal_ratio_() const
    {
        const std::size_t n = loss_window_.size();
        if (n < 3) return Scalar{0};

        std::size_t reversals = 0;
        std::size_t total = 0;

        Scalar prev_delta = loss_window_[1] - loss_window_[0];

        for (std::size_t i = 2; i < n; ++i)
        {
            Scalar curr_delta = loss_window_[i] - loss_window_[i - 1];

            // 方向反转：相邻两次差分异号
            if (prev_delta * curr_delta < Scalar{0})
                ++reversals;

            // 忽略零差分（完全平坦不算反转）
            if (prev_delta != Scalar{0})
                ++total;

            prev_delta = curr_delta;
        }

        return (total > 0) ? static_cast<Scalar>(reversals) / static_cast<Scalar>(total)
                           : Scalar{0};
    }

    // 衰减后缩小窗口到一半，丢弃旧的振荡历史
    void shrink_window_()
    {
        const std::size_t keep = cfg_.window_size / 2;
        while (loss_window_.size() > keep)
            loss_window_.pop_front();
    }
};

} // namespace nn

#endif // NN_OSCILLATION_GUARD_HPP
