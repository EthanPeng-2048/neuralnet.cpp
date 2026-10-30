// ── cli_lr_scheduler.hpp — 学习率调度（CLI 转发头）────────────────────────
//
// P2b：实现已上收到库层 `neuralnet.cpp/lr_scheduler.hpp`（`nn::` 命名空间），
// 训练驱动（TextTrainer）与 mnist_train 共用同一份。本头只保留旧 include 点与
// 旧名字（`nn::cli::LrScheduleConfig` 等），调用方零改动。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include "neuralnet.cpp/lr_scheduler.hpp"

namespace nn::cli
{
using nn::compute_epoch_lr;
using nn::compute_step_lr;
using nn::LrScheduleConfig;
using nn::StepLrScheduleConfig;
} // namespace nn::cli
