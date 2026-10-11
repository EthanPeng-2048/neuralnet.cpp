#pragma once

// ── train.hpp — 训练常用集合（按需加载薄转发头）───────────────────────────
//
// 与 `nn.hpp` 的关系：**`nn.hpp` 仍是唯一聚合头**（一次拿齐全部模块）；本头
// 是「按需加载」里最常用的那一档 = 引擎 + 层 + 损失 + 优化器 + 模型容器与
// 序列化，够写完整训练循环。详见 `engine.hpp` 头注释。
//
// 仍需**单独按需** include 的：
//   · 模型工厂 / 分词器 → `domain_*.hpp`（如 domain_gpt.hpp、domain_tokenizer.hpp）
//   · 数据集 CSV 读取   → `cli/cli_mnist_io.hpp`（当前挂在 CLI 头下）
//
// 用法（完整训练循环骨架，签名细节见 AGENTS.md §8）：
//     #include <neuralnet.cpp/train.hpp>
//
//     nn::CpuEngine engine;                       // 或 nn::GpuEngine（NN_HAS_VULKAN）
//     nn::Model model(engine);
//     NN_UNWRAP(model.add<nn::Linear>(784, 256));
//     NN_UNWRAP(model.add<nn::ReLU>());
//     NN_UNWRAP(model.add<nn::Linear>(256, 10));
//
//     auto opt = NN_UNWRAP(nn::create_optimizer("adam", engine,
//                 model.parameters(), model.param_gradients(), 1e-3f, 0.01f));
//     nn::CrossEntropyLoss loss_fn;
//
//     // 每步 = 清零 → 前向 → 损失 → 反向 → 更新（签名细节见 AGENTS.md §8）
//     NN_UNWRAP(model.zero_grad());
//     nn::Tensor logits = NN_UNWRAP(model.forward(x));
//     nn::Scalar loss   = NN_UNWRAP(loss_fn.forward(engine, logits, y));
//     nn::Tensor grad   = NN_UNWRAP(loss_fn.backward());
//     NN_UNWRAP(model.backward(*grad));
//     NN_UNWRAP(opt->step());

#include "engine.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "model.hpp"
