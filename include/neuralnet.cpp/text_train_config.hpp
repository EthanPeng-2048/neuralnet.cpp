// ── text_train_config.hpp（库层）— 文本训练配方 ────────────────────────────
//
// P2b：把"训练循环 + 断点状态段"需要的配方字段上收到库层，供训练驱动
// `TextTrainer` 直接消费；CLI 层 `cli/text_train_config.hpp` 的 `TrainConfig`
// 继承本结构，只额外保留**入口相关**字段（数据集/保存路径/恢复/GPU/日志等）。
// 这样"配方"只有一份定义，就不会出现 CLI 与训练工厂各漂各的默认值。
//
// 本头只依赖数值/精度基础类型（无引擎、无模型），可被 CLI 参数解析层轻量包含。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <neuralnet.cpp/core_config.hpp>   // nn::Scalar
#include <neuralnet.cpp/precision.hpp>     // nn::PrecisionProfile
#include "train_state.hpp"   // nn::DatasetIdentity（断点状态段的数据身份）

namespace nn
{
// 文本训练配方（epoch × step 循环、lr 调度、记录、断点状态段共用的形状）
//
// ⚠ **配方一致性**：优化器侧字段（`optimizer_name`/`lr`/`weight_decay`/`beta1`/
// `beta2`/`precision`/`max_norm`）的真源是 `TrainingSession` 的 `TrainingConfig`；
// 本结构的同名字段只是"喂给循环/断点状态"的副本，必须与 Session 完全一致
// （`TextTrainer::create` 逐项校验并报错）。起步用 `nn::make_text_train_config(session)`。
//
// **Batch 语义（docs/todo/30）**：`batch_size` 是一次 optimizer update 覆盖的
// 理论 global batch；每次 forward/backward 的 micro batch =
// `batch_size / accum_steps`。`steps_per_epoch` 与计数器均以 update step 为单位。
//
// 默认值 = 原 `text_train` CLI 默认值（2026-10 口径）。`seq_len` 的 CLI 默认
// 是 `nn::GPT_SEQ_LEN`（当前 256），CLI 构造 TrainConfig 时显式对齐一次，
// 避免本头反向依赖 domain_gpt。
struct TextTrainConfig
{
    // ── 数据网格（样本/step 的派生根）────────────────────────────────────
    std::size_t seq_len = 256;   // 每个训练样本（窗口）的 token 数
    std::size_t stride = 0;      // 窗口步长；0 = seq_len（不重叠）
    std::size_t batch_size = 32; // global batch（一次 optimizer update）
    int epochs = 10;
    int start_epoch = 0;         // resume 后从第几个 epoch 继续（0-based）
    std::size_t start_step = 0;  // resume 后本 epoch 内从第几步继续（0-based）
    std::size_t max_steps = 0;   // 本进程最多训练步数（0 = 按 epoch 跑完）
    std::size_t accum_steps = 1; // 梯度累积次数；micro = batch_size / accum_steps

    // ── 学习率调度 ───────────────────────────────────────────────────────
    std::string lr_schedule = "fixed";  // fixed / cosine / step_cosine
    Scalar lr = 0.001f;
    Scalar min_lr = 1e-6f;
    int warmup_epochs = 0;
    std::size_t warmup_steps = 0;
    std::vector<Scalar> lr_per_epoch;

    // ── 记录 / 断点 ──────────────────────────────────────────────────────
    std::size_t log_interval = 50;
    std::size_t save_interval = 100;
    bool grad_log = false;

    // ── 训练期配方（也写进断点状态段，续训时是权威）──────────────────────
    Scalar max_norm = 0.0f;              // 0 = 不裁剪
    std::string optimizer_name = "adam";
    Scalar weight_decay = 0.01f;
    Scalar beta1 = 0.9f;
    Scalar beta2 = 0.999f;
    PrecisionProfile precision{};        // 默认全 F32（零回归）

    // ── 数据身份（断点状态段；也参与 continue-training 门禁）──────────────
    // 调用方（CLI）已算好 sha256/身份时直接填入——避免大文件被重复哈希；
    // 未填的字段由训练工厂从样本集合补齐（sha256 空 = 工厂自行计算整文件摘要）。
    DatasetIdentity identity{};
};
} // namespace nn
