#pragma once

// ── loss.hpp — 损失函数（按需加载薄转发头）────────────────────────────────
//
// 与 `nn.hpp` 的关系：`nn.hpp` 是唯一聚合头，本头只转发 `compute_loss.hpp`。
// 详见 `engine.hpp` 头注释（整洁性入口、不省 parse）。
//
// 提供：`Loss` 基类 + `MSELoss` / `CrossEntropyLoss`。
// 用法：
//     #include <neuralnet.cpp/loss.hpp>
//     nn::CrossEntropyLoss ce;
//     // 大词表用稀疏路径（整数标签，不物化 one-hot，铁律 #9）：
//     Scalar l = NN_CHECK(ce.forward_sparse(engine, logits, labels, mask, vocab));
//
// 注：`Loss::forward/forward_sparse` **仍显式收 engine 形参**（与 Layer 不同，
//     Loss 无天然绑定时机，见 AGENTS.md §8）。

#include "compute_loss.hpp"
