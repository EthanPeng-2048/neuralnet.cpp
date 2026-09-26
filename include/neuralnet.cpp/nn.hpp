#pragma once

// ── neuralnet.cpp 统一入口 ────────────────────────────────────────────────
// 包含所有核心模块，用户只需 #include "nn.hpp"
//
// 包含顺序按依赖关系排列：L0 → L1 → L2 → L3 → L4
// 注意：algebra_span.hpp 由 algebra_matrix.hpp 传递包含；algebra_ops.hpp 由
//       expr_dsl.hpp 引入（旧 algebra_expr/algebra_compute 已随逐元素算子移除），
//       core_config.hpp 已传递包含 core_errors.hpp，
//       此处显式列出所有头文件是为了清晰展示模块结构。
//
// 新架构（引擎化）：
//   ComputeEngine 抽象硬件接触，Layer/Loss/Optimizer 通过组合原语表达算法，
//   forward/backward 单套实现适配 CPU/GPU。

// L0 硬件层
#include "core_errors.hpp"
#include "core_assert.hpp"
#include "core_observer_ptr.hpp"
#include "core_threadpool.hpp"
#include "core_file.hpp"
#include "core_config.hpp"

// L1 代数层（algebra_matrix.hpp 已传递包含其余代数头文件）
#include "algebra_matrix.hpp"

// L2 计算层 — 引擎化
#include "compute_tensor.hpp"
#include "expr_spec.hpp"
#include "compute_engine.hpp"
#include "expr_dsl.hpp"
#include "compute_cpu_engine.hpp"
#ifdef NN_HAS_VULKAN
#include "compute_gpu_engine.hpp"
#endif
// 多精度适配层（f16 边界 cast：把 CpuEngine/GpuEngine 包装成支持 f16 存储的引擎）
#include "compute_precision_engine.hpp"
#include "compute_layer.hpp"
#include "compute_loss.hpp"
#include "compute_optimizer.hpp"

// L3 实现层
#include "model_container.hpp"
#include "model_spec.hpp"
#include "model_serialization.hpp"

// L4 构建层
#include "domain_mnist.hpp"
#include "domain_gpt.hpp"
#include "domain_zipt.hpp"
#include "domain_rla.hpp"
#include "domain_cnn.hpp"
#include "domain_tokenizer.hpp"

