#pragma once

// ── layer.hpp — 神经网络层（按需加载薄转发头）─────────────────────────────
//
// 与 `nn.hpp` 的关系：`nn.hpp` 是唯一聚合头，本头只转发 `compute_layer.hpp`。
// 详见 `engine.hpp` 头注释（整洁性入口、不省 parse）。
//
// 提供：`Layer` 基类 + Linear/ReLU/GeLU/SwiGLU/LayerNorm/RMSNorm/BatchNorm/
//       Softmax/Conv2D/MaxPool2D/FeedForward/注意力/Transformer/GPTBlock/RAPT
//       等全部内置层，以及位置编码策略族。
// 用法：
//     #include <neuralnet.cpp/layer.hpp>
//     nn::Linear fc(784, 256);          // 手工构造的层须先 fc.init(engine)
//
// 注：模型**工厂**（build_gpt_model 等）在 domain_*.hpp，不在本头。

#include "compute_layer.hpp"
