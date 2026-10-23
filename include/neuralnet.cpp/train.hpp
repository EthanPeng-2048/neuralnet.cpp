#pragma once

// ── train.hpp — 训练常用集合（按需加载薄转发头）───────────────────────────
//
// 与 `nn.hpp` 的关系：**`nn.hpp` 仍是唯一聚合头**（一次拿齐全部模块）；本头
// 是「按需加载」里最常用的那一档 = 引擎 + 层 + 损失 + 优化器 + 模型容器与
// 序列化，够写完整训练循环。详见 `engine.hpp` 头注释。
//
// 仍需**单独按需** include 的：
//   · 模型工厂 → `domain_*.hpp`（如 domain_gpt.hpp；`TextTrainer` 已带分词器头）
//   · 数据集 CSV 读取   → `cli/cli_mnist_io.hpp`（当前挂在 CLI 头下）
//
// 用法（完整训练循环骨架，签名细节见 AGENTS.md §8）：
//     #include <neuralnet.cpp/train.hpp>
//
//     auto engine = nn::CpuEngine::create();      // 或 nn::GpuEngine::create(backend)
//     nn::Model model(*engine);
//     NN_UNWRAP(model.add<nn::Linear>(784, 256));
//     NN_UNWRAP(model.add<nn::ReLU>());
//     NN_UNWRAP(model.add<nn::Linear>(256, 10));
//
//     nn::TrainingConfig tcfg; tcfg.optimizer = "adam"; tcfg.lr = 1e-3f;
//     auto session = NN_UNWRAP(nn::TrainingSession::create(std::move(model), tcfg));
//     nn::CrossEntropyLoss loss_fn;
//
//     // 每个 micro-batch：任务回调返回 output grad；Session 管累积/裁剪/update。
//     // 完整签名见 AGENTS.md §8 与 training_session.hpp。

#include "engine.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "model.hpp"
#include "training_session.hpp"   // TrainingSession（推荐：Model + Optimizer 同生共死）
#include "training_group_session.hpp" // 通用多模型具名角色
#include "rl_pipeline.hpp"        // 可替换 RL 编排组件
#include "dataset.hpp"            // Dataset（.nndataset 只读）
#include "dataset_samples.hpp"    // TextSampleSet：数据集 → 训练样本（窗口）
#include "lr_scheduler.hpp"       // 学习率调度（epoch / step 级）
#include "text_train_config.hpp"  // TextTrainConfig（训练配方）
#include "text_trainer.hpp"       // TextTrainer：数据集直接接入的训练工厂
