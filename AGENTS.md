# AGENTS.md — AI 开发速览（neuralnet.cpp）

> 本文档专为 AI 编码助手编写：用最少 token 建立项目心智模型，快速开始开发。
> 人类向的完整文档在 `docs/`（索引见 §11）。**改代码前必读 §5 铁律与 §10 高频坑。**

## 1. 项目一句话

从零实现的 C++26 神经网络库：CPU / Vulkan 双后端，支持 MLP / ViT / GPT 训练推理 + BPE 分词器。
`include/neuralnet.cpp/` 是 header-only 库（唯一入口 `nn.hpp`），`src/` 是可执行入口，`shaders/` 是 GPU 原语 shader。

## 2. 构建与测试（Linux）

```bash
# 构建（默认 Release：-O3 -march=native -fno-exceptions -Wall -Wextra -Wpedantic -Werror）
cmake -B build -G Ninja && cmake --build build

# 测试（默认关闭；开启后测试程序在 build/test/，并注册 ctest）
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build && ctest --test-dir build
```

- 编译器：Clang++ 22.1+（C++26）；CMake 3.30+。
- Vulkan 可选：CMake 自动探测 Vulkan + glslc，找到则定义 `NN_HAS_VULKAN` 启用 GPU，否则纯 CPU。
- **CUDA 后端 1.0.0 已停用**（融合原语未实现，`compute_cuda_engine.hpp` 返回"未实现"），勿依赖、勿在文档中声称支持。
- 应用入口：`build/{mnist_train,mnist_infer,text_train,text_infer,tokenizer_train,tokenizer_infer}`；`gui.py` 是 Python GUI（subprocess 调这些可执行文件）。

## 3. 目录速查（改什么任务 → 看什么文件）

| 任务 | 文件 |
|------|------|
| 加/改神经网络层（Linear/Attention/Norm/激活…） | `compute_layer.hpp`（聚合头）+ `compute_layer_{base,mlp,conv,softmax,attention,feedforward,transformer,gpt,zipt,rapt}.hpp` |
| 加/改损失函数 | `compute_loss.hpp` |
| 加/改优化器（SGD/Adam/AdamW/Muon） | `compute_optimizer.hpp` |
| 加/改引擎原语（CPU 实现） | `compute_engine.hpp`（接口）+ `compute_cpu_engine.hpp` |
| 加/改引擎原语（GPU 实现） | `compute_gpu_engine.hpp` + `backend/compute_vk_backend.hpp` + `shaders/*.comp` |
| 张量/设备抽象 | `compute_tensor.hpp` |
| 矩阵/表达式模板（CPU 代数层） | `algebra_matrix.hpp` / `algebra_expr.hpp` / `algebra_ops.hpp` / `algebra_compute.hpp` |
| 表达式 DSL / 融合 IR | `expr_dsl.hpp` / `expr_spec.hpp` / `expr_opt.hpp` / `expr_graph.hpp` / `expr_registry.hpp` |
| 模型容器/规格/序列化 | `model_container.hpp` / `model_spec.hpp` / `model_serialization.hpp` |
| MNIST / GPT / CNN / 分词器 模型工厂 | `domain_mnist.hpp` / `domain_gpt.hpp` / `domain_cnn.hpp` / `domain_tokenizer.hpp` |
| 训练/推理 CLI 入口 | `src/mnist_train.cpp` 等；公共 CLI 逻辑在 `include/neuralnet.cpp/cli/` |
| 构建期工具（AOT 融合） | `tools/scan_exprs.cpp` / `tools/gen_fused.cpp` |
| 与 PyTorch 对拍 | `compare_with_torch/`（model.py / text_train.py / text_infer.py） |

## 4. 分层架构（L0→L5，严格单向依赖，上层只依赖下层公有接口）

```
L5 入口    src/*.cpp（mnist/text/tokenizer 的 train/infer）、gui.py
L4 领域    domain_*.hpp（模型工厂：build_mnist_* / build_gpt_model / Tokenizer）
L3 实现    model_container.hpp（Model 容器）、model_spec.hpp、model_serialization.hpp
L2 计算    compute_engine.hpp（引擎抽象）、compute_cpu/gpu_engine.hpp、compute_layer/loss/optimizer.hpp
L1 代数    algebra_*.hpp（Matrix、表达式模板 AST、compute::apply）
L0 硬件    core_config.hpp（Scalar=float、BLOCK_SIZE=64、SmartPolicy）、core_threadpool/errors/assert/file.hpp
```

**核心设计：引擎化（Engine-Based）**
- Layer 的 `forward/backward` 只写一次，通过 `ComputeEngine&` 参数自动适配 CPU/GPU。没有 `forward_gpu` 这种东西。
- **算法与原语分离**：Engine 只提供 op-level 原语（`matmul`/`add`/`exp`/`reduce`…），不知道 "ReLU" 是什么；Layer 用原语组合表达算法（`ReLU = max(x,0)`）。
- 数据流：`Matrix → engine.from_matrix → Tensor[GPU] → forward/loss/optimizer 全程在 GPU → 仅 evaluate 时 to_matrix 回 CPU`。

**ComputeEngine 原语分类**（`compute_engine.hpp`）：
矩阵级 `matmul/batched_matmul/transpose/add_inplace/scale_inplace/zero/axpy_inplace`；归约级 `row/col_reduce_sum/max`；广播级 `broadcast_row/col_inplace`；逐元素 `elementwise_unary/binary/binary_scalar`；数据操作 `slice_rows/insert_rows/gather_rows/scatter_add_rows/rearrange_3d/clone`；批控制 `begin_batch/end_batch`（CPU no-op，GPU 录 command buffer）。

## 5. 铁律（违反必出 bug）

1. **禁止 throw/try/catch**：编译期 `-fno-exceptions` 强制。错误一律 `Result<T> = std::expected<T, Error>`（`core_errors.hpp`），调用方 `if (!r) return std::unexpected(...)` 传播。
2. **禁止 new/delete/裸指针所有权**：`std::vector` / `std::unique_ptr` / `std::span`。
3. **分层职责单一**：Matrix（L1）不写神经网络算法；Layer（L2）不写底层计算；原语 shader 永不含算法（ReLU/Softmax/Attention 等一律来自 Layer 或 DSL）。
4. **不穿透接口**：上层不访问下层内部数据结构（`.data()` 等），改一个模块只改一个头文件。
5. **布局全局统一 batch-major**：序列展平列序 `i = b*seq + t`（batch 在列方向）。历史上 GPT 曾用 position-major（`i = t*batch + b`）导致跨样本串扰的灾难级 bug，现已统一。**所有注意力/序列相关测试必须覆盖 batch>1**（batch=1 时两种布局重合，测不出）。
6. **GPU 命令录制生命周期**：录制期（`begin_batch`→`end_batch` 之间）引用的所有张量必须存活到 `end_batch()` 之后；GPU buffer 销毁走 `pending_destroys_` 延迟队列。
7. **AOT 闭合世界**：GPU 表达式 shader 全部构建期生成，运行时按 `expr_spec_key` 精确匹配，**未命中硬报错**，无 eager、无运行时编译。
8. **确定性**：任何"依赖容器迭代顺序"的决策点（BPE 平局打破、ID 分配等）必须显式排序/按 key 打破平局；并行化后结果必须与单线程逐字节一致。
9. **大词表禁止物化 one-hot**：用 `CrossEntropyLoss::forward_sparse`（整数标签 + loss_mask）。

## 6. 数据布局约定

```
Matrix: 行主序 (rows, cols)，data_[row*cols + col]
神经网络张量: 列主序 batch-major
  输入 (feature_dim, batch_size)   每列一个样本
  权重 (out_features, in_features)
  多头 Q/K/V: (H*d_k, batch*seq) → rearrange_3d → (batch*H*d_k, seq)
GPT 序列展平: 列序 i = b*seq + t（batch-major，全局唯一约定）
```

## 7. 表达式 DSL 与 AOT 融合管线（GPU 开发必读）

- Layer 内用 `nn::dsl`（`expr_dsl.hpp`）写普通数学表达式；CPU 编译期模板直接求值（内联+SIMD），GPU 折叠成 `ExprSpec`（扁平 IR，`expr_spec.hpp`）→ 按 key 查预编译融合 shader。
- 块式融合：`start_expr() ... end_expr()` 录制一段表达式链。
- **构建期两步**（CMake 自动编排，改 Layer 内联表达式后重跑构建即可）：
  1. `scan_exprs`：dry-run 跑 Layer forward/backward，收集折叠出的 `ExprSpec` 结构（去重）→ `build/generated/expr_specs.bin`
  2. `gen_fused`：读 bin → `glsl_gen` 生成 GLSL → glslc → 内联 SPIR-V → `build/generated/fused_registry.hpp`
- 手写原语 shader 在 `shaders/*.comp`（matmul、reduce、broadcast、bmm_*、col_softmax_* 等），构建期 glslc 编译并嵌入 C++ 头文件。
- IR 优化 pass（canonicalize/CSE/寄存器分配/图 IR 融合）见 `expr_opt.hpp` / `expr_graph.hpp`，设计文档 `docs/11-ir-optimization.md`。

## 8. 训练循环范式（写新入口时照抄）

```cpp
#include <neuralnet.cpp/nn.hpp>
nn::CpuEngine engine;                       // 或 nn::GpuEngine（NN_HAS_VULKAN）
auto model_r = nn::build_gpt_model(engine, vocab, d_model, seq, heads, d_ff, layers);
nn::Model model = std::move(*model_r);
// 优化器工厂：sgd / sgd_momentum / adam / adamw / muon
auto optimizer = nn::create_optimizer("adamw", engine,
    model.parameters(), model.param_gradients(), /*lr=*/1e-4, /*wd=*/0.01);

nn::CrossEntropyLoss ce_loss;
// 每 step（GPT 用 forward_sparse 稀疏路径，不物化 one-hot）：
model.zero_grad();
auto x = engine.from_matrix(batch);         // Matrix → Tensor
auto logits = model.forward(*x);
auto loss = ce_loss.forward_sparse(engine, *logits, labels, loss_mask, vocab);
auto grad = ce_loss.backward();             // (vocab, total) 梯度
model.backward(*grad);
optimizer.step();
```

> 注意签名细节：`Model::forward/backward/zero_grad` 与 `Loss::backward`、`Optimizer::step/zero_grad` **都不带 engine 参数**（engine 在 Model 构造时已绑定）；只有 `Loss::forward/forward_sparse` 和 `engine.from_matrix` 需要显式传 engine。

- 模型工厂：`nn::build_mnist_mlp_model(engine)` / `build_mnist_transformer_model(engine)` / `build_gpt_model(...)` / `build_gpt_model_from_spec(spec)`。
- 链式构建：`model.add_linear(784,256).add_relu().add_linear(256,10)`；模板版 `model.add<nn::Linear>(784,256)`。
- 序列化：`save_model` / `load_model` / `peek_model_spec`（`model_serialization.hpp`，v4 自描述格式）；`.nnpkg` 训练包见 `docs/07-train-package.md`。
- GPT 高级特性（`GPTModel`，`compute_layer.hpp` 尾部）：梯度检查点（`checkpoint_every_`）、activation offload、文档感知掩码（`set_doc_ids`）、batch flush 粒度。

## 9. 关键常量与配置（`core_config.hpp`）

- `Scalar = float`；`BLOCK_SIZE = 64`（matmul 分块，b_block 栈预算 64KB）；`PARALLEL_THRESHOLD = 524288`（SmartPolicy 并行阈值）。
- 不使用 `-ffast-math`（保 NaN/Inf，训练稳定性）。

## 10. 高频坑（Top 8，详见 `docs/08-pitfalls-and-lessons.md`）

1. **布局混用**：position-major vs batch-major → 跨样本串扰、loss 平台期。改序列代码先确认列序约定。
2. **GPU 双存储影子一致性**：CPU/GPU 双存储是分布式状态机问题；GPU-resident 路径已禁用，走 staging（每次算子往返 PCIe）。
3. **录制期 use-after-free**：局部张量在 `end_batch()` 前析构 → `VK_ERROR_DEVICE_LOST`。
4. **TDR/设备死亡**：`VK_ERROR_DEVICE_LOST` 不可重试（存 checkpoint 退出）；`VK_TIMEOUT` 可减半 batch 重试。
5. **缓存 key 冲突**：naive 位域/组合 key 会撞车，用强哈希（FNV-1a）或双字段。
6. **零拷贝 reshape 的 CPU/GPU 语义差异**：CPU 分支 reshape 需复制数据，GPU 保持零拷贝。
7. **内存爆炸**："所有出现位置"类索引随输入线性膨胀（BPE 曾 60-80GB）；大词表 one-hot 曾 3.2GB。先做内存预算。
8. **gradcheck 必须先 forward** 填充 `input_cache_` 再 backward，否则空缓存崩溃。

## 11. 文档索引（docs/）

| 文档 | 何时读 |
|------|--------|
| `01-architecture.md` | 需要完整分层/数据流/模块详解时 |
| `02-performance.md` | 性能优化（SmartPolicy、缓存分块、GPU） |
| `03-quickstart-model.md` | 构建模型 API 教程 |
| `04-quickstart-train-infer.md` | 训练/推理 CLI + C++ API + GUI |
| `05-algorithm-reference.md` | 每个 Layer/Loss/Optimizer 的数学与原语分解 |
| `06-cuda-backend.md` | CUDA 后端（已停用，恢复参考） |
| `07-train-package.md` | `.nnpkg` 训练包 |
| `08-pitfalls-and-lessons.md` | **踩坑警示录，改代码前读** |
| `09-operator-fusion.md` | 算子融合（M1-M6 已完成） |
| `10-memory-optimization.md` | 显存优化（L1/L2 已实施） |
| `11-ir-optimization.md` | IR 优化（IR-A/B/C/D 已实施） |
| `12-innovative-designs.md` | 创新设计全景 |
| `13-optimize-proposal-list.md` | 待做优化方案清单 |
| `14-operator-fusion-2.md` | 融合算子二期：matmul 参与 IR 融合 + 跨 kernel 自动融合（P2-05/P2-10，删 M4-M6 手写原语） |
| `DEVELOPMENT_STANDARDS.md` | C++ 编码规范全文 |
