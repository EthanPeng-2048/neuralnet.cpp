#pragma once

// ── optimizer.hpp — 优化器（按需加载薄转发头）─────────────────────────────
//
// 与 `nn.hpp` 的关系：`nn.hpp` 是唯一聚合头，本头只转发 `compute_optimizer.hpp`。
// 详见 `engine.hpp` 头注释（整洁性入口、不省 parse）。
//
// 提供：`Optimizer` 基类 + `create_optimizer(name, engine, params, grads, lr, wd)`
//       （name：sgd / sgd_momentum / adam / adamw / muon）。
// 用法：
//     #include <neuralnet.cpp/optimizer.hpp>
//     auto opt = NN_CHECK(nn::create_optimizer("adamw", engine,
//                 model.parameters(), model.param_gradients(), /*lr=*/1e-4f, /*wd=*/0.01f));

#include "compute_optimizer.hpp"
