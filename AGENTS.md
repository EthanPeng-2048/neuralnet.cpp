# AGENTS.md — AI 开发速览（neuralnet.cpp）

> 本文档专为 AI 编码助手编写：用最少 token 建立项目心智模型，快速开始开发。
> 人类向的完整文档在 `docs/`（索引见 §11）。**改代码前必读 §5 铁律与 §10 高频坑。**
> 本文只记录**当前状态**；演进记录、已修复勘误、被否决方案与性能 A/B 过程统一归档在 `docs/history.md`（索引见 §11）。当前能力清单见 §12。

## 1. 项目一句话

从零实现的 C++26 神经网络库：CPU / Vulkan 双后端，支持 MLP / ViT / GPT 训练推理 + BPE 分词器，可选 f16 混合精度。
`include/neuralnet.cpp/` 是 header-only 库（唯一入口 `nn.hpp`），`src/` 是可执行入口，`shaders/` 是 GPU 原语 shader。

## 2. 构建与测试

```bash
# 构建（默认 Release：-O3 -funroll-loops -march=native，Clang/GCC 另加 -fno-exceptions -Wall -Wextra -Wpedantic -Werror）
cmake -B build -G Ninja && cmake --build build

# 测试（默认关闭；开启后测试程序在 build/test/，并注册 ctest）
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build && ctest --test-dir build
```

- 编译器：Clang++ 22.1+（C++26）；CMake 3.30+。CMake 在 Linux 上若 PATH 中能找到 `clang++` 会优先选它（自动向量化优于 g++），可用 `CXX=g++` 或 `-DCMAKE_CXX_COMPILER` 覆盖；MSVC 走 `/std:c++latest`。
- 构建选项：`NN_ENABLE_NATIVE`（默认 ON，开启 `-march=native`，分发/CI 用 `OFF` 生成可移植基线）；`NN_ENABLE_TESTS`（默认 OFF）。
- Vulkan 可选：CMake 自动探测 Vulkan + glslc，找到则定义 `NN_HAS_VULKAN` 启用 GPU，否则纯 CPU。支持多 Vulkan 设备选择（`--gpu` 参数，见 `cli/cli_gpu_option.hpp`）。
- **本项目不支持 CUDA**：后端仅 CPU / Vulkan，CLI 无 `--cuda` 参数；文档勿声称支持 CUDA。
- 应用入口：`build/{mnist_train,mnist_infer,text_train,text_infer,tokenizer_train,tokenizer_infer}`，另有 `layer_bench`（性能）与测试类可执行文件。`gui.py` 是 Python GUI，`train_pkg.py` 打包 `.nnpkg` 训练包，`cli_controllers.py` 提供 CLI 控制逻辑。

## 3. 目录速查（改什么任务 → 看什么文件）

| 任务 | 文件 |
|------|------|
| 加/改神经网络层（Linear/Attention/Norm/激活…） | `compute_layer.hpp`（聚合头）+ `compute_layer_{base,mlp,conv,softmax,attention,feedforward,transformer,gpt,zipt,rapt}.hpp` |
| 加/改损失函数 | `compute_loss.hpp` |
| 加/改优化器（SGD/Adam/AdamW/Muon） | `compute_optimizer.hpp` |
| 加/改引擎原语（CPU 实现） | `compute_engine.hpp`（接口）+ `compute_cpu_engine.hpp` |
| 加/改引擎原语（GPU 实现） | `compute_gpu_engine.hpp` + `backend/compute_vk_backend.hpp` + `backend/compute_vk_device.hpp` + `shaders/*.comp` |
| 张量/设备抽象 | `compute_tensor.hpp` |
| 混合精度 / f16 类型系统 | `precision.hpp`（Precision 枚举、`nn::f16`、`PrecisionProfile`） |
| 矩阵/代数层（CPU 存储与手写内核） | `algebra_matrix.hpp` / `algebra_span.hpp` / `algebra_ops.hpp`（`Expression`/`BoolExpression` 概念在 `expr_dsl.hpp`） |
| 表达式 DSL / 融合 IR | `expr_dsl.hpp` / `expr_spec.hpp` / `expr_opt.hpp` / `expr_registry.hpp`（IR-C 图融合未采用、无 `expr_graph.hpp`；取舍记录见 `docs/history.md`） |
| 后端代码生成（IR-D emitter 抽象） | `expr_emitter.hpp`（注册表）+ `expr_glsl_gen.hpp`（GlslEmitter） |
| 模型容器/规格/序列化 | `model_container.hpp` / `model_spec.hpp` / `model_serialization.hpp` / `model_keyvalue_record.hpp` |
| MNIST / GPT / CNN / RLA / ZiPT / 分词器 模型工厂 | `domain_mnist.hpp` / `domain_gpt.hpp` / `domain_cnn.hpp` / `domain_rla.hpp` / `domain_zipt.hpp` / `domain_tokenizer{,_base,_bpe,_charbpe}.hpp` |
| 训练/推理 CLI 入口 | `src/mnist_train.cpp` 等；公共 CLI 逻辑在 `include/neuralnet.cpp/cli/` |
| 构建期工具（AOT 融合） | `tools/scan_exprs.cpp` / `tools/gen_fused.cpp`（另有 `tools/decode_fused.py` 调试用） |
| 批量改写 / 一致性审计（改多处时用，均带 `-DryRun`） | `tools/edit_ranges.ps1`（行区间删除：四重断言 + **花括号平衡护栏**）/ `tools/test_refactor.ps1`（删定义块 / 插 include / 正则替换）/ `tools/doc_rename.ps1`（文档词法改名）/ `bench/doc_align_audit.ps1`（文档↔代码对齐审计：文件/符号/CLI/数字/测试名） |
| 与 PyTorch 对拍 | `compare_with_torch/`（model.py / text_train.py / text_infer.py） |

## 4. 分层架构（L0→L5，严格单向依赖，上层只依赖下层公有接口）

### 4.1 架构概览图

```mermaid
graph TB
    subgraph "L5 用户入口层"
        A["mnist_train/infer"]
        B["text_train/infer"]
        C["tokenizer_train/infer"]
        D["gui.py"]
    end
    
    subgraph "L4 领域构建层"
        E["domain_mnist.hpp"]
        F["domain_gpt.hpp"]
        G["domain_tokenizer*.hpp"]
        F2["domain_cnn / domain_rla / domain_zipt.hpp"]
    end
    
    subgraph "L3 实现层"
        H["model_container.hpp"]
        I["model_spec.hpp"]
        J["model_serialization.hpp"]
    end
    
    subgraph "L2 计算层（引擎化）"
        K["compute_engine.hpp"]
        L["compute_cpu_engine.hpp"]
        M["compute_gpu_engine.hpp"]
        N["compute_layer*.hpp"]
        O["compute_loss.hpp"]
        P["compute_optimizer.hpp"]
        P2["precision.hpp"]
    end
    
    subgraph "L1 代数层"
        Q["algebra_matrix.hpp"]
        R["algebra_span.hpp"]
        S["algebra_ops.hpp"]
    end
    
    subgraph "L0 硬件层"
        V["core_config.hpp"]
        W["core_threadpool.hpp"]
        X["core_errors.hpp"]
    end
    
    D -->|subprocess| A & B & C
    A & B & C --> E & F & G & F2
    E & F & G & F2 --> H & I & J
    H & I & J --> K & N & O & P
    K --> L & M
    N & O & P --> Q
    P2 -.精度配置.-> K
    Q --> R & S
    Q --> V & W & X
```

### 4.2 核心设计原则

**引擎化（Engine-Based）**：Layer 的 `forward/backward` 只写一次，通过 `ComputeEngine&` 参数自动适配 CPU/GPU。

**算法与原语分离**：
- Engine 只提供 op-level 原语（`matmul`/`add`/`exp`/`reduce`…）
- Layer 用原语组合表达算法（`ReLU = max(x,0)`）
- Engine 不知道 "ReLU" 是什么

**数据流**：
```
Matrix → engine.from_matrix → Tensor[GPU] → forward/loss/optimizer 全程在 GPU → 仅 evaluate 时 to_matrix 回 CPU
```

**多精度**：`Precision`（F16/F32；BF16/F64 为保留值，使用即报错）贯穿张量存储与算子输出；`PrecisionProfile{param/compute/stable/optimizer}` 做模型级配置。语义锚点见 `precision.hpp` 头注释与 `docs/development/05-mixed-precision.md`。

### 4.3 ComputeEngine 原语分类

| 类别 | 原语 |
|------|------|
| 矩阵级 | `matmul/batched_matmul/matmul_with_bias/transpose/add_inplace/accumulate/scale_inplace/zero` |
| 归约级 | `row_reduce_sum`、`col_reduce_sum/max`、`grouped_reduce_sum/max`（沿行方向按固定长度 R 分组归约，池化/多头等场景免逐通道循环） |
| 数据操作 | `slice_rows/insert_rows/gather_rows/scatter_add_rows/rearrange_3d/im2col/col2im/clone/copy_from/cast`（`im2col/col2im`：卷积/池化窗口展开与伴随散射，纯数据搬运） |
| 扫描级 | `scan_prefix_outer/scan_suffix_outer/outer_col`（RLA/RAPT，dk≤64，见 docs/development/06 §扫描原语） |
| 表达式 | `eval_expr/eval_expr_into/eval_expr_reduce`（AOT 融合 shader 入口） |
| 批次/内存 | `begin_batch/end_batch/flush_batch/release_idle_pool_blocks/pool_stats` |
| offload | `create_offload_buffer/offload_save/offload_restore`（activation offload） |

> 引擎共 **49** 个 virtual（复现：`bench/doc_inventory.ps1`）；逐元素/广播等计算类原语一律经表达式
> DSL 求值，不经 Layer 直调（`dsl::row_reduce_max` 是 DSL 叶子，不是引擎算子）。CRLF/`scan_exprs` AOT 键规则见 §7。

### 4.4 理解优先级（建议学习顺序）

1. **先理解 L0-L1**（基础数据结构）：`Matrix`、`Tensor`、`Scalar`
2. **再理解 L2 核心**：`ComputeEngine` 接口 + 一个简单 Layer（如 `Linear`）
3. **然后理解 L3**：`Model` 容器如何组合 Layer
4. **最后理解 L4-L5**：具体模型实现和训练流程

## 5. 铁律（违反必出 bug）

1. **禁止 throw/try/catch**：编译期 `-fno-exceptions` 强制。错误一律 `Result<T> = std::expected<T, Error>`（`core_errors.hpp`），调用方 `if (!r) return std::unexpected(...)` 传播。
2. **禁止 new/delete/裸指针所有权**：`std::vector` / `std::unique_ptr` / `std::span`。
3. **分层职责单一**：Matrix（L1）不写神经网络算法；Layer（L2）不写底层计算；原语 shader 永不含算法（ReLU/Softmax/Attention 等一律来自 Layer 或 DSL）。
4. **不穿透接口**：上层不访问下层内部数据结构（`.data()` 等），改一个模块只改一个头文件。
5. **布局全局统一 batch-major**：序列展平列序 `i = b*seq + t`（batch 在列方向）。position-major（`i = t*batch + b`）会造成跨样本串扰，任何情况下不得混用。**所有注意力/序列相关测试必须覆盖 batch>1**（batch=1 时两种布局重合，测不出）。
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
- 主要入口：`dsl::compute(engine, expr, rows, cols)`（一行表达式，最常用——**必带输出形状**，没有 2 参重载）；把结果写进既有张量（原地更新，零分配）用 `dsl::compute_into(engine, expr, dst)`；归约语义用 `dsl::compute_reduce`。**跨表达式融合（IR-C：`start_expr/end_expr`、`begin_expr/end_expr`、`expr_graph.hpp`）未采用**，当前没有任何跨表达式录制机制——取舍记录与重新立项前提见 `docs/development/03-ir-optimization.md` §5.3。分组归约（MaxPool 窗口等）用 `dsl::grouped_reduce_sum/max(tensor, R)` 视图（R 进 key）；**alpha 缩放**写成 `dsl::matmul(..., batch) * dsl::rparam(alpha)` 尾链（rparam 值不进 key）。
- **fold 段（P-C1/C2）**：`ExprSpec.fold = FoldSpec`（分块状态归约：键域逐块 body + 行标量态跨块进位 + `vecacc` 行向量态块累加 + 向量域 finalize）——**通用折叠表达式机制，不含任何注意力专属语义**（与注意力无关的通用样例 rowmax/rowsum/softmax_denom 在 `expr_fold.hpp`）。注意力 forward 直调 `engine.eval_expr(make_fold_attn_o(...))`（**不经 DSL 钩子——scan 显式登记块是 fold spec 唯一注册来源**）；该构造与 5 掩码变体 `FoldAttnMask`（Plain/Causal/Alibi/Doc/AlibiDoc，漏登记即 GPU 闭合世界硬报错）**定义在 `compute_layer_attention.hpp`**（按 AOT 原则"表达式文本只出现在 Layer"归位）；`FoldSpec.tri_skip`（行界整块跳过）进 key（codegen 分歧点）；`EXPR_FOLD_BLOCK=128`/`EXPR_FOLD_ROWS_PER_WG=2` 为 CPU/GPU 共享常量（改则两侧同改）。详见 `expr_fold.hpp` 与 `expr_spec.hpp` 的 FoldSpec 注释。
- **构建期两步**（CMake 自动编排，改 Layer 内联表达式后重跑构建即可）：
  1. `scan_exprs`：dry-run 跑 Layer forward/backward，收集折叠出的 `ExprSpec` 结构（去重）→ `build/generated/expr_specs.bin`
  2. `gen_fused`：读 bin → 经 `emitter_registry` 选后端（默认 `"glsl"` = `GlslEmitter`）生成 GLSL → glslc → 内联 SPIR-V → `build/generated/fused_registry.hpp`
- **IR-D emitter 抽象**（`expr_emitter.hpp`）：把后端代码生成从 GLSL 专用抽象为 emitter 接口（一份 canonical IR → 多后端代码），`--list-backends` 可列出注册后端。目前仅 `glsl` 后端注册（无 `CpuEmitter`/`CudaEmitter` 实现）。
- 手写原语 shader 在 `shaders/*.comp`（matmul、matmul_tiled、matmul_gemv、batched_matmul、reduce、elementwise_v2、transpose、gather、scatter_add、rearrange_3d、im2col、col2im、group_reduce、scan_prefix_outer、scan_suffix_outer、scan_prefix_outer_gen、scan_suffix_outer_gen（gen = 通用 dk 版，dk>64 后端走它）、outer_col、cast），构建期 glslc 编译并嵌入 C++ 头文件；多数 shader 另用 `-DNN_SHADER_F16=1` 编一份 f16 变体（`NN_ETYPE=float16_t`，f32 分支留在 `#else`）。
- IR 优化 pass（canonicalize/CSE/寄存器分配）见 `expr_opt.hpp`，设计文档 `docs/development/03-ir-optimization.md`（IR-C 取舍记录在其 §5.3，归档正文在 `docs/history.md`）。

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
- 逐层构建：`model.add<nn::Linear>(784,256); model.add<nn::ReLU>(); model.add<nn::Linear>(256,10);`——`add<T>(args...)` 是模板方法（无链式 API），返回 `Result<void>`，须检查错误；层构造后自动 `init(engine)`。
- 序列化：`save_model` / `load_model` / `peek_model_spec`（`model_serialization.hpp`，v4 自描述格式）；`.nnpkg` 训练包见 `docs/usage/04-train-package.md`。
- GPT/RAPT 高级特性（`GPTModel` / `RAPTModel`，`compute_layer.hpp` 尾部）：梯度检查点（`checkpoint_every_`）、activation offload、文档感知掩码（`set_doc_ids`）、batch flush 粒度。RAPT 与 GPT 同档支持前三者（`RAPTModel::set_checkpoint_every/set_activation_offload/set_flush_interval`），offload 走共用实现 `ActivationOffloader`（`compute_layer_base.hpp`）。

## 9. 关键常量与配置（`core_config.hpp`）

- `Scalar = float`；`BLOCK_SIZE = 64`（matmul 分块，b_block 栈预算 64KB）；`PARALLEL_THRESHOLD = 524288`（自适应并行阈值，由 `nn::parallel_for_blocks/parallel_for_samples` 门控）。
- 不使用 `-ffast-math`（保 NaN/Inf，训练稳定性）。

## 10. 高频坑（Top 8，详见 `docs/development/08-pitfalls-and-lessons.md`）

1. **布局混用**：position-major vs batch-major → 跨样本串扰、loss 平台期。改序列代码先确认列序约定。
2. **GPU 双存储影子一致性**：CPU/GPU 双存储是分布式状态机问题；GPU-resident 路径已禁用，走 staging（每次算子往返 PCIe）。
3. **录制期 use-after-free**：局部张量在 `end_batch()` 前析构 → `VK_ERROR_DEVICE_LOST`。
4. **TDR/设备死亡**：`VK_ERROR_DEVICE_LOST` 不可重试（存 checkpoint 退出）；`VK_TIMEOUT` 可减半 batch 重试。
5. **缓存 key 冲突**：naive 位域/组合 key 会撞车，用强哈希（FNV-1a）或双字段。
6. **零拷贝 reshape 的 CPU/GPU 语义差异**：CPU 分支 reshape 需复制数据，GPU 保持零拷贝。
7. **内存爆炸**："所有出现位置"类索引随输入线性膨胀、大词表 one-hot 物化都会把内存吃到几十 GB 级。先做内存预算（历史量级见 `docs/development/08-pitfalls-and-lessons.md`）。
8. **gradcheck 必须先 forward** 填充 `input_cache_` 再 backward，否则空缓存崩溃。

## 11. 文档索引（docs/，按类别子目录组织）

### 总档

| 文档 | 何时读 |
|------|--------|
| `history.md` | **全仓唯一的历史状态归档**：演进记录 / 已修复勘误 / 被否决方案 / 性能 A/B 过程。代码注释与其他文档只记录当前状态；查"为什么当初这么改、旧数字是多少"只来这里 |

### 介绍类（docs/introduction/）

| 文档 | 何时读 |
|------|--------|
| `introduction/01-architecture.md` | 需要完整分层/数据流/模块详解时（**含快速理解指南和理解路线图**） |
| `introduction/02-performance.md` | 性能优化（自适应并行、缓存分块、GPU） |
| `introduction/03-algorithm-reference.md` | 每个 Layer/Loss/Optimizer 的数学与原语分解 |
| `introduction/04-innovative-designs.md` | 创新设计全景 |

### 开发类（docs/development/）

| 文档 | 何时读 |
|------|--------|
| `development/01-compute-engine-development.md` | **计算引擎开发指南：接口详解、实现模式、添加新原语** |
| `development/02-operator-fusion.md` | **算子融合全篇（当前形态与设计约束）：IR 扩展（归约语义）→ 表达式录制 → matmul 参与 IR 融合 → 跨 kernel 自动融合（一期 M1-M7 + 二期 S1-S5/S7；分期实施流水与否决方案见 `docs/history.md`）** |
| `development/03-ir-optimization.md` | IR 优化（现行 IR-A/B/D；**IR-C 图融合未采用——取舍记录正文在 `docs/history.md`，重新立项前提见其 §5.3**） |
| `development/04-memory-optimization.md` | 显存优化：L1 激活重计算（梯度检查点）/ L2 内存池整块归还；L3 远期 |
| `development/05-mixed-precision.md` | **多精度计算（f16/混合精度）：Precision 类型系统、类型化存储、硬件/兼容路径分派、显式精度推导、PrecisionProfile（param/compute/stable/optimizer）、当前实测结论与已知数值限制（§12）** |
| `development/06-rapt-algorithm.md` | **线性注意力（RLA → RAPT → RLA-2）：直观理解 + 数学定义 + 扫描原语工程落地 + 训练显存开关** |
| `development/07-zipt-algorithm.md` | AttnZip / ZiPT：记忆压缩解码器算法设计 |
| `development/08-pitfalls-and-lessons.md` | **踩坑警示录，改代码前读** |
| `development/10-development-standards.md` | C++ 编码规范全文 |
| `development/12-compute-engine-inventory.md` | **引擎接口盘点（复现：`bench/doc_inventory.ps1`）：49 个 virtual、Layer 直调 23 个（基础设施/数据搬运/状态扫描/fold 登记）、两套 CPU 求值机制（DSL 模板 + IR 解释器）、ctest 19 个目标；演进记录见 `docs/history.md`** |
| `development/13-refactor-backlog.md` | **重构与性能机会清单（2026-09-25 审查）：只记录方案不实施；误报/已修复项对照表与已执行记录在 `docs/history.md`，重复立项前先读本文件** |
| `development/14-f16-stable-gpu-loss-frozen.md` | **故障报告（2026-09-26，未修）：GPU `stable=f16` 训练 loss 打印冻结（权重不冻结）——触发矩阵、测试覆盖缺口、证据与复现** |

### 使用类（docs/usage/）

| 文档 | 何时读 |
|------|--------|
| `usage/01-quickstart-model.md` | 构建模型 API 教程 |
| `usage/02-quickstart-train-infer.md` | 训练/推理 CLI + C++ API + GUI |
| `usage/03-compute-engine-usage.md` | 计算引擎使用指南：张量操作、矩阵运算、表达式融合 |
| `usage/04-train-package.md` | `.nnpkg` 训练包 |

## 12. 当前状态

> 本节只列**当前事实**；演进记录、已修复勘误、被否决方案与性能 A/B 过程见 `docs/history.md`。

### 已知问题

- **GPU `stable=f16` 训练 loss 打印冻结（未修）**：`text_train --precision-stable f16` 时 step 恒 4.0137、跨 epoch/跨进程逐位相同；**权重照常更新**（同进程跨 epoch 模型快照 63% 参数字节不同，冻结仅在 loss 回读链）；CPU 同配置健康；ctest 19/19 全绿但**无 GPU+stable=f16 用例**。触发矩阵、覆盖缺口与证据见 `docs/development/14-f16-stable-gpu-loss-frozen.md`。推荐路径 `--f16`（stable/optimizer=f32）实测健康。
- **数值性限制**：四字段全 f16 不可训练——`optimizer=f16` 单独即令 Adam 更新爆炸、`stable=f16` 链约 200 步 NaN，见 `docs/development/05-mixed-precision.md` §12.5；常规 f32 训练健康收敛。
- **不存在的 CLI 参数**：`--tdr-retry`/`--max-tdr-retries`（勿引用）。

### 已交付能力（当前功能清单）

- **混合精度**：`precision.hpp` 的 `Precision`/`PrecisionProfile{param/compute/stable/optimizer}`；CPU/GPU f16 路径 = 边界 cast（`PrecisionEngine` 适配层）+ in-kernel f16 带类型变体 + op-level f16 GEMM（一份 .comp 用 `-DNN_SHADER_F16=1` 编第二份 SPIR-V）；`--f16` = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}，实测峰值显存低于 f32（`docs/development/05` §12.11）。
- **线性注意力**：RLA-2 / RAPT（`docs/development/06`）。
- **训练稳定性**：防 NaN 跳步。
- **训练显存开关（GPT 与 RAPT 同档）**：梯度检查点 `set_checkpoint_every`、activation offload `set_activation_offload`、文档掩码 `set_doc_ids`、batch flush 粒度 `set_flush_interval`；检查点与 offload **可混合**（checkpoint 块重算、其余块 offload）。引擎侧 API 为 `create_offload_buffer`/`offload_save`/`offload_restore`（`set_offload_enabled` 不存在）。
- **Vulkan 多设备选择**：`--gpu` 参数与 `NN_VULKAN_DEVICE` 环境变量（`cli/cli_gpu_option.hpp`、`backend/compute_vk_device.hpp`）。
- **计算类原语全量走 DSL**：Layer 直调 23 个，全部是基础设施/数据搬运/状态扫描/fold 显式登记（`docs/development/12` §2.1）；逐元素/归约/matmul 等一律经 `dsl::compute*`。
- **注意力单 fold kernel**：`FoldSpec` 分块流式求值（分数矩阵 S 不物化），5 种掩码变体（Plain/Causal/Alibi/Doc/AlibiDoc），`tri_skip` 整块跳过被屏蔽区。
- **CNN 全引擎化**：`im2col`/`col2im` 数据搬运原语 + `rearrange_3d` 布局置换，Conv2D/MaxPool2D 前反向为「引擎原语 + DSL」，无 PCIe 往返；池化反向为窗口并列最大值均分梯度。
- **评估分块**：`evaluate_mnist` 的 `eval_batch`（默认 1000）分块前向 + 每块 `release_idle_pool_blocks()`，防大 batch 评估 OOM。
- **CPU 性能**：DSL 模板路径向量化/并行、`dsl::compute_into` 零分配原地更新、`Tensor::cpu_get_ptr`、分块 GEMM 内核（BLOCK_SIZE=64）。
- **测试**：ctest 注册 19 个目标（`-DNN_ENABLE_TESTS=ON`；需 Vulkan 的用例退出码 77 = skip）。

### 融合二期状态与 IR 编码约束（改融合/IR 代码前必读）

二期范围 = S1-S5、S7（S6 自动窗口不接线，不在当前计划；详见 `docs/development/02-operator-fusion.md`，分期实施流水见 `docs/history.md`）。

1. **运行时值禁进表达式常量池**（进 `expr_spec_key` 会破坏闭合世界）：`scale` 折进 Q（forward `scale_inplace` + backward 补乘）、`inv_num_valid` 后置 `scale_inplace`。
2. **BatchCol 视图要求 `(1, BH*seq)`**（doc_ids 按 (b,h) 块重复），`(1, batch*seq)` 会越界。
3. **RowGather 主输入行数≠网格行数**（loss_vec 在 (1,N) 读 (C,N) logits），校验只查 cols。
4. `gen_fused` `emit_spec` 的 ±inf 常量必须用 `numeric_limits`。
5. **matmul + 列归约可用**：`generate_glsl_reduce` 列分支按元素分解 batch（`batch = row/m_per`，遍历全部 `rows = batch*m_per` 行）；覆盖 = `expr_cpu_test::col_max(matmul)`（batch=2 独立标量参考）+ `expr_gpu_test` col 对拍（广播/向量/batch=2）。
6. **IR 扩展点**：MatmulSpec.batch（不进 key，dispatch z）、Row/Col/Batch 操作数(6/7/8)、RowGather(9)/BatchMod(10)/BatchCol(11)；注意力 forward = 单 fold kernel（`FoldSpec`，5 掩码变体经 `fold_mask_variant_`），bwd = R/X 表达式 + 3 个 `batched_matmul`；CE 稠密 `denom = col_sum(exp(logits-cb(col_max)))`，稀疏 grad/loss_vec 用 Row+RowGather。

> 变更此文件时务必同步 git 状态：`CMakeLists.txt` 的 `project(... VERSION ...)` 可能滞后于 git tag，以 git tag 为准。
