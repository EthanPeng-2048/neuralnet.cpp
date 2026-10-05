#pragma once

// ── engine.hpp — 计算引擎（按需加载薄转发头）───────────────────────────────
//
// 与 `nn.hpp` 的关系：**`nn.hpp` 仍是唯一聚合头**（一次拿齐全部模块）；本头
// 是「按需加载」的转发，只为了 `#include <neuralnet.cpp/engine.hpp>` 这种
// 好记的写法，不必去猜 `compute_*.hpp` 的命名。
//
// ⚠ 这是**整洁性**入口，**不省 parse**：各头的依赖闭包本就相互牵连（如
//   `compute_tensor.hpp` 无条件引用 Vulkan 后端头），真正省时间要按依赖
//   重切（另案，见 docs/development/18-roadmap.md P0-4 / 编译提速）。
//
// 提供：`ComputeEngine`（原语接口）、`CpuEngine`，以及 GPU 构建下的 `GpuEngine`。
// 用法：
//     #include <neuralnet.cpp/engine.hpp>
//     nn::CpuEngine engine;

#include "compute_engine.hpp"
#include "compute_cpu_engine.hpp"
#ifdef NN_HAS_VULKAN
#include "compute_gpu_engine.hpp"
#endif
