// ── cli_precision_warning.hpp — 危险精度组合启动期告警（仅警告，不拒绝）─────
//
// 口径见 docs/internal/24-f16-training-defects.md §2.3 / §5 F1：
//   · 触发集合 = `optimizer = F16`（四字段全 f16 必含 optimizer=F16，故被覆盖）；
//   · `stable = F16` **单独**出现不触发（小模型健康，大词表才 ~200 步 NaN，
//     是否单列见 24 §5 F1 备注）——但若与触发字段同时出现，会在同一条警告里
//     补一句 CE 链风险；
//   · 仅 stderr 文本一次：**不改变退出码、不阻断训练、不静默降级**；
//     `--f16`（profile_f16）与任何合法组合返回空串（静默）。
//
// 用法（CLI 参数解析后、进入训练循环前调用一次）：
//   if (const std::string w = nn::cli::dangerous_precision_warning(cfg.precision);
//       !w.empty())
//       std::cerr << w;
//
// 返回：应打印到 stderr 的警告文本；无危险组合 → 空串。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <string>

#include "neuralnet.cpp/precision.hpp"   // nn::PrecisionProfile / Precision / precision_name

namespace nn::cli
{
    // 危险精度组合启动期告警文本（无危险组合 → 空串）。
    inline std::string dangerous_precision_warning(const nn::PrecisionProfile& p)
    {
        using nn::Precision;

        // 唯一触发字段：优化器状态精度 = F16（Adam 的 m/v 下溢 → 更新爆炸）。
        if (p.optimizer != Precision::F16)
            return {};

        const bool all_f16 = p.param == Precision::F16 && p.compute == Precision::F16 &&
                             p.stable == Precision::F16 && p.optimizer == Precision::F16;

        std::string w;
        w += "[精度警告] 检测到危险精度组合（仅提醒，不阻断训练，详见 "
             "docs/internal/24-f16-training-defects.md §2）：\n";
        w += std::string("  · optimizer=") + nn::precision_name(p.optimizer) +
             "：Adam 二阶矩 v≈g²~1e-10 在 f16 下溢为 0 → 更新量 -lr·m̂/(√v̂+eps) "
             "爆炸 → loss 发散/NaN。\n";
        if (p.stable == Precision::F16)
            w += std::string("  · stable=") + nn::precision_name(p.stable) +
                 "：CE/softmax 归约在 f16 下范围(65504)/尾数(10 位)不足 → 大词表 "
                 "~200 步 NaN。\n";
        if (all_f16)
            w += "  · param/compute/stable/optimizer 全 f16：更新量 |delta| 可能小于"
                 "参数所在的 f16 网格步长 → 原地写回被舍入吃光（loss 恒定，数值性冻结）。\n";
        w += "  安全配方：`--f16`（param+compute=f16，stable/optimizer=f32）。"
             "如需继续实验请自行留意数值稳定性。\n";
        return w;
    }
} // namespace nn::cli
