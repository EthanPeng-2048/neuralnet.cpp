#pragma once

// ── compute_layer.hpp — 引擎化计算层（聚合头） ───────────────────────────
//
// 按逻辑域拆分为多个 compute_layer_*.hpp 子文件，此文件按依赖顺序聚合，
// 保持对既有代码（nn.hpp / scan_exprs / 测试 / src）的单一入口兼容。
//
// 架构铁律：
//   1. Layer 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。不再有 forward_gpu / backward_gpu。
//   2. 算法只在 Layer（通过组合 engine 原语表达），绝不在 Engine/Shader 中。
//   3. Engine/Shader 只提供 op-level 原语（matmul, add, exp, max, reduce 等）。
//
// 拆分文件（按依赖顺序）：
//   compute_layer_base.hpp          Layer 基类 + clone_tensor 辅助
//   compute_layer_mlp.hpp           Linear/ReLU/GeLU/SwiGLU/LayerNorm/RMSNorm/FusedChainLayer
//   compute_layer_conv.hpp          Conv2D / MaxPool2D
//   compute_layer_softmax.hpp       Softmax
//   compute_layer_attention.hpp     RotaryEmbedding / AttentionBase / MultiHeadAttention / CausalSelfAttention
//   compute_layer_feedforward.hpp   FeedForward
//   compute_layer_transformer.hpp   PositionalEncoding / TransformerEncoderLayer / TransformerEncoder / PatchEmbedding
//   compute_layer_gpt.hpp           GPTBlock / PositionEncoder 家族 / GPTModel
//   compute_layer_zipt.hpp          CrossAttention / ZiPTBlock / ZiPTModel
//   compute_layer_rapt.hpp          ReLULinearAttention / RAPTBlock / RAPTModel
// ─────────────────────────────────────────────────────────────────────────

// L2 计算层（引擎化）— 各子文件已自带依赖 include，这里按拓扑序聚合。
#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_conv.hpp"
#include "compute_layer_softmax.hpp"
#include "compute_layer_attention.hpp"
#include "compute_layer_feedforward.hpp"
#include "compute_layer_transformer.hpp"
#include "compute_layer_gpt.hpp"
#include "compute_layer_zipt.hpp"
#include "compute_layer_rapt.hpp"

