#pragma once

// ── model.hpp — 模型容器 / 规格 / 序列化（按需加载薄转发头）───────────────
//
// 与 `nn.hpp` 的关系：`nn.hpp` 是唯一聚合头，本头转发 L3 三个头。
// 详见 `engine.hpp` 头注释（整洁性入口、不省 parse）。
//
// 提供：`Model`（层容器：add/forward/backward/parameters/zero_grad/set_training）、
//       `ModelSpec`（架构规格）、`save_model` / `load_model` / `peek_model_spec`。
// 用法：
//     #include <neuralnet.cpp/model.hpp>
//     nn::Model model(engine);
//     NN_CHECK(model.add<nn::Linear>(784, 256));
//     NN_CHECK(nn::save_model("model.bin", model, spec));
//
// 注：模型**工厂**（build_mnist_mlp_model / build_gpt_model / build_cnn_model…
//     与分词器）在 `domain_*.hpp`，不在本头；本头的 `Model::spec()` 由那些
//     工厂在构建时回填，供 `load_model` 做文件头与架构一致性校验。

#include "model_spec.hpp"
#include "model_container.hpp"
#include "model_serialization.hpp"
