# 计算引擎现状盘点（2026-09-18；2026-09-25 复核）

> ⚠️ **更新（2026-09-19）：本文中的 IR-C 部分已执行完毕——IR-C 被整体删除。**
> `expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`、`dsl::start_expr/end_expr`、
> `FusedChainLayer`、`Tensor::virtual_tag_` 均已从代码移除；取舍依据与重新立项前提见
> `03-ir-optimization.md` §5.3。§4.2/§4.3/§5/§8 保留原文作为**决策前的证据快照**（其中"IR-C 该接上"
> 的判断已被 §8.9 的实测结论推翻），阅读时请以本横幅与 §8 末的"决策已执行"为准。
>
> **2026-09-25 复核**：接口数字已按当前 HEAD 重测——**58 个 virtual（Phase 2 加入
> `cast_into`/`copy_into`/`supports_*`/`eval_expr_*` 等）、Layer/Loss/Optimizer 直调 35 个**。
> （该数字随后被 **2026-09-26 收敛更新为 49 个 virtual / Layer 直调 32 个**，见顶部横幅与 §2。）
> 复现命令见 `bench/doc_inventory.ps1` 与 §9。
>
> 目的：先把**当前代码里实际并存的计算 API 与遗留物**列清楚，作为收敛讨论的事实底座。
> 全部结论来自当前工作树的源码与构建配置，未做改动。

> ## 🗑️ 2026-09-26：本文档 §5 遗留物清单已**执行完毕**（收敛记录）
>
> - **旧代数 AST 整体移除**：`algebra_expr.hpp`、`algebra_compute.hpp`（`nn::compute::apply`）删除；`Expression`/`BoolExpression`
>   概念迁入 `expr_dsl.hpp`（唯一使用者）；`algebra_span.hpp`/`algebra_matrix.hpp`/`compute_cpu_engine.hpp` 的相关 include 清除；
>   `Matrix::detail::{apply,binary_apply,binary_apply_inplace}`（零调用者）一并删除。
> - **无根/测试-only operator 全部删除**：`axpy_inplace`、`broadcast_row_inplace`、`broadcast_col_inplace`（连带 `shaders/broadcast.comp`
>   + `broadcast_gpu` + `broadcast_pipeline_` + `has_broadcast_pipeline` 全链）、`elementwise_unary/binary/binary_scalar`、
>   `elementwise_select_scalar_cond`、引擎 `row_reduce_max`、offload B 组 `offload_store/offload_load`，
>   以及 `UnaryOp`/`BinaryOp`/`CompareOp` 三个枚举。
> - **数字更新**：引擎 virtual **58 → 49**；CPU 求值机制 **4 → 2**（① DSL 模板路径 ② IR 解释器；旧代数 AST 与 `eval_cpu` 串行模板分支已不复存在）。
> - **随行清理**：`src/test_common.hpp`（测试公共工具唯一副本）、死的 `activation_cache()` override（ViT/ZiPT）、
>   重复的 `offload_test` 聚合目标（ctest 20 → 19）、`gpt_offload_test` 补 batch 录制包裹。
> - 下文 §2.2/§3/§5 的行内标注即为逐项对照；**§3 表格中的"机制 1（旧代数 AST）"整体作废**。

---

## 0. 一句话总览

引擎经历过 5 代设计（纯 Matrix → forward_gpu 双实现 → eager → begin_expr/end_expr → `dsl::compute`），
**5 代里有 3 代的代码今天仍在编译、且 2 代仍在被调用**（第 4 代 IR-C 已于 2026-09-19 删除）：

| 世代 | 现状 |
|------|------|
| 纯 Matrix（无 Tensor） | `algebra_*` 仍在，是 `Tensor` 的底层存储 + CPU 手写 kernel |
| `forward_gpu` / `backward_gpu` 双实现 | ✅ **已彻底移除**（仅注释残留） |
| eager（直接调引擎算子） | ⚠️ **仍是 Layer 的一部分主力**（35 个算子被直接调用，2026-09-25 复核） |
| `begin_expr` / `end_expr`（IR-C 录制图） | 🗑️ **已移除**（2026-09-19：无收益点且无生产调用方，见 `03-ir-optimization.md` §5.3） |
| `dsl::compute` | ✅ 当前主推，被 Layer 大量使用 |

---

## 1. 引擎实现层：3 个实现，2 个活的

| 实现 | 文件 | 大小 | 状态 | 证据 |
|------|------|------|------|------|
| `CpuEngine` | `compute_cpu_engine.hpp` | 88.5 KB | ✅ 活 | 默认引擎 |
| `GpuEngine` | `compute_gpu_engine.hpp` | 58.0 KB | ✅ 活 | Vulkan（`NN_HAS_VULKAN`） |
| ~~`CudaEngine`~~ | ~~`compute_cuda_engine.hpp`~~ | ~~22.4 KB~~ | 🗑️ **已移除** | 与 `backend/compute_cuda_backend.hpp`、`cuda/` 目录一并删除；历史快照见 git 分支 `legacy/cuda` |

**CUDA 移除范围**（原先约 100 KB 死代码 + 10 处条件分支已全部清除）：

- 删除 `cuda/`（`CMakeLists.txt` / `cuda_kernels.h` / `cuda_kernels.cu`）
- 删除 `compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`
- 清除 `compute_tensor.hpp`（6 处 `#ifdef NN_HAS_CUDA`）、`nn.hpp`、
  `cli/cli_engine_factory.hpp`（`EngineConfig::use_cuda`）、`cli/cli_train_common.hpp`
  （`--cuda`）、`CMakeLists.txt` 停用段
- 清除 15 个 `src/*.cpp` 的 `--cuda` / `use_cuda` / `cuda_enabled`
- 清除 `cli_controllers.py` 的 `--cuda` 传参

---

## 2. `ComputeEngine` 接口：49 个 virtual 成员（2026-09-26 收敛后；原 58）

文件：`compute_engine.hpp`（2026-09-25 复核，`bench/doc_inventory.ps1` 可复现；
多精度 Phase 2 新增 `cast_into`/`copy_into`/`supports_native_data_move`/
`supports_expr_precision_variant` 与 `Precision P` 参数的运算类原语。）

### 2.1 被 Layer / Loss / Optimizer **直接调用**的算子：35 个

（统计口径：`compute_layer*.hpp` + `compute_loss.hpp` + `compute_optimizer.hpp` +
`model_container.hpp` 中出现的 `engine.<op>(`，排除 `engine_.reset(...)` 这类非算子调用）

```
accumulate, add_inplace, batched_matmul, begin_batch, clone,
col_reduce_max, col_reduce_sum, col2im, create_offload_buffer,
create_tensor, elementwise_binary, end_batch, eval_expr, flush_batch,
from_matrix, gather_rows, grouped_reduce_max, grouped_reduce_sum,
im2col, insert_rows, matmul, matmul_with_bias, offload_restore,
offload_save, outer_col, rearrange_3d, row_reduce_sum, scale_inplace,
scan_prefix_outer, scan_suffix_outer, scatter_add_rows, slice_rows,
to_matrix, transpose, zero
```
（`im2col`/`col2im` 于 2026-09-20 新增：卷积/池化窗口展开与伴随散射，
纯数据搬运、无算法语义；`grouped_reduce_max/sum` 同期新增：沿行方向按固定长度
分组归约，用来把"逐通道一次 dispatch"的层内循环压成单次原语调用。
Conv2D 与 MaxPool2D 已**全引擎化**，不再有 `to_matrix/from_matrix` 往返、
CPU 标量循环或逐通道循环。`eval_expr` 的 Layer 直调点是注意力 fold
——fold 显式登记是 FoldSpec 唯一注册来源，属 AOT 设计而非旁路。）

### 2.2 不被 Layer 直接调用（其余 23 个：DSL / 序列化 / CLI / 适配层 / 测试）

| 算子 | 实际调用方 |
|------|-----------|
| `eval_expr_into` / `eval_expr_reduce` | `dsl::compute_into` / `dsl::compute_reduce`（`expr_dsl.hpp`）+ `PrecisionEngine` 转发 |
| `cast` / `copy_from` | `model_serialization.hpp` + f16 测试 / gradcheck |
| `cast_into` / `copy_into` / `supports_*` | 仅 `compute_precision_engine.hpp` 内部（Phase 2 管道，非死代码） |
| `offload_store` / `offload_load` | 仅 `offload_primitive_test.cpp`（**Layer 用的是 `offload_save`/`offload_restore`**） |
| `elementwise_unary` / `elementwise_binary_scalar` | 仅 f16/gpu 测试与 `layer_bench`；`elementwise_binary` 另有 1 处生产调用（`compute_layer_rapt.hpp:866` Div） |
| `broadcast_row_inplace` | **无根调用方**（仅 `CpuEngine` 实现 + `PrecisionEngine` 转发） |
| `broadcast_col_inplace` | 仅 `f16_precision_test` / `layer_bench` |
| `axpy_inplace` | **无根调用方**（仅 `PrecisionEngine` 转发；optimizer 已迁 `dsl::compute_into`） |
| `row_reduce_max`（引擎算子） | 仅 `PrecisionEngine` 转发；Layer 走的是 `dsl::row_reduce_max`（DSL 叶子，非此算子） |
| `elementwise_select_scalar_cond` | **全仓无调用点**（接口 + 3 份引擎实现） |
| `pool_stats` | `src/text_train.cpp`（池账本统计） |
| `release_idle_pool_blocks` | `cli/cli_mnist_io.hpp`、`src/text_train.cpp`、`mem_probe.cpp` |
| `submit_scalar_readback` / `poll_scalar_readback` / `scalar_readback_slots` | 标量回读（loss / `ce_fusion_test`），Layer 不直调 |
| `device` | 引擎外部少量判断 |

---

## 3. 表达式求值：**四套机制并存**

这是遗留产物的核心。

| # | 机制 | 实现位置 | 并行度（实测） | 谁在用 |
|---|------|----------|----------------|--------|
| 1 | ~~**旧代数 AST** `compute::apply(span, expr)`~~ | 🗑️ **2026-09-26 已整体删除**（`algebra_compute.hpp`/`algebra_expr.hpp`/`Matrix::detail::*`；概念迁入 `expr_dsl.hpp`） | — | 无（原仅 `CpuEngine` elementwise，该算子亦已删） |
| 2 | **DSL 模板路径** `eval_cpu(e, rows, cols)` → `eval_into_span` | `expr_dsl.hpp:181` | **并行 + 向量化**（2026-09-19 复核：`n≥PARALLEL_THRESHOLD` 走 `nn::parallel_for_samples` 分块，块内 `NN_VECTORIZE_PRAGMA`；不再 per-element 重导 `cpu_matrix().span()`） | `dsl::compute` 的**无归约**分支 + `dsl::compute_into` 的逐元素分支 |
| 3 | **DSL IR 解释器** `eval_expr_impl` | `compute_cpu_engine.hpp:1251-1789` | **串行**（逐元素 switch 分派 + 归约前缀重放） | `dsl::compute` 的**含归约/广播/matmul/索引视图**分支（`expr_dsl.hpp:680`）+ `compute_reduce` |
| 4 | **GPU AOT 融合** `eval_expr` | `compute_gpu_engine.hpp:942` | shader 并行 | GPU 全部表达式 |

**分流判据**：`has_reduction_v<E>`（`expr_dsl.hpp:455-486`）。注意它把 **`MatmulRef` 也算作"归约"**
（`:462-464`），所以连 matmul 都被推进了解释器。DSL 在 Layer 中的用量：
`dsl::compute(` 88 处、`dsl::compute_reduce(` 20 处（其中相当一部分含 `matmul`/`broadcast`/归约，落到机制 3）。

**死/重复**（2026-09-26 更新）：
- 机制 1（旧代数 AST）**已整体删除**；机制 2（DSL 模板）与机制 3（DSL IR 解释器）是 DSL 在 CPU 上的两条路，一条内联、一条解释——**这是收敛后的全部 CPU 求值路径**。机制 3 逐元素 switch 分派 + 归约前缀重放，实测比等价原语慢 1.5–3.8 倍（2026-09-18 诊断实测；原诊断报告已删除，本行保留结论）。

---

## 4. 融合世代：**eager** 与 **DSL/IR 栈（含未接线的 IR-C）** 并存

### 4.1 eager（直接调引擎算子）
没有"被移除"，而是**仍然是 Layer 的主力**：§2.1 的 35 个算子。
GPU 侧每个都是独立的 backend kernel dispatch（`compute_gpu_engine.hpp:485` matmul、
`:843` elementwise_unary、`:615` add_inplace…），即当初"kernel 开销大"的那条路。

### 4.2 `begin_expr` / `end_expr` + `expr_graph.hpp`（IR-C 录制图）🗑️ 已移除

> **2026-09-19：本节所述代码已全部删除**（依据见 §8.9 与 `03-ir-optimization.md` §5.3）。
> 以下保留决策前的盘点原文，行号已随删除失效。
>
- 接口：`compute_engine.hpp:110-111`；`CpuEngine` 实现 `compute_cpu_engine.hpp:61/72`；`GpuEngine` 实现 `:184/194`
- 录制图：`expr_graph.hpp`（22.5 KB，含 `ExprGraph` / `recording_graph_owner`）
- **生产调用方：无**。唯二调用者是：
  - `FusedChainLayer`（`compute_layer_mlp.hpp:762-800`）——注释自述是 **"IR-C 演示"** 层，不被任何模型工厂使用
  - 构建期 `tools/scan_exprs.cpp:136-143` + 测试 `src/expr_fuse_test.cpp`
- scan 分支：`expr_dsl.hpp:660` / `:750` 在 `NN_EXPR_SCAN` 下登记录制图

#### 4.2.1 代数关系（重要）：IR-C **不是**独立系统，是 DSL 同一 IR 流水线的阶段 C

| 阶段 | 内容 | 文件 | 文档 |
|------|------|------|------|
| IR-A | canonicalization（DCE/常量折叠/重编号） | `expr_opt.hpp` | `docs/development/03-ir-optimization.md` §5.1（2026-08-23 完成） |
| IR-B | CSE + 寄存器分配 | `expr_opt.hpp` | 同上 §5.2 |
| **IR-C** | **图 IR / 跨表达式融合** | **`expr_graph.hpp`** | 同上 §5.3 |
| IR-D | 后端 emitter 抽象 | `expr_emitter.hpp` + `expr_glsl_gen.hpp` | 同上「后端 emitter 抽象」 |

三条硬证据：

1. **代码上嵌在 DSL 执行路径里**：`dsl::compute` / `compute_reduce` 检测
   `fused::recording_graph()`（`expr_dsl.hpp:660` / `:750`）；GPU `eval_expr` 同样检测
   （`compute_gpu_engine.hpp:951`）。IR-C 不是独立入口，是 DSL 求值的一个**分支**。
2. **历史（原始开发分支 `optimize/ir`，后被 squash 进 release）**：
   - `bf86af2`（2026-08-21）"AOT&简化计算写法" → **新增 `expr_dsl.hpp`（DSL）**
   - `36caba4`（2026-08-24）"**IR-C/D优化**，图执行eager编写拉爆所有ML框架" →
     **新增 `expr_graph.hpp`(304) + `expr_emitter.hpp`(103) + `cpu_emitter.hpp`(252)**
   即 DSL 先出现，3 天后 IR-C/D 作为**同一融合栈的下一阶段**一起加入，目标写明是打赢 eager。
3. **形态也是围绕 DSL 定义的**：录制段内 `dsl::compute` 把 ExprSpec 加进 `ExprGraph`，
   `end_expr` 用 `fuse_expr_graph` 贪心拼接逐元素链。

> **所以 IR-C 的定性是"最新的优化层，但从未接线"，不是"旧系统遗留"。**
> 它要解决的是"多个 `dsl::compute` 仍是多个 kernel、中间 Tensor 全部物化"这个问题——
> 正是 §7.5 里说的 DSL 缺的第二样能力。

#### 4.2.2 不是二选一：`dsl::compute` 是前端，IR-A/B/C/D 是同一流水线的阶段

> ⚠️ 本节描述的是**决策前**的流水线（含 IR-C）。2026-09-19 后流水线只剩 IR-A/B/D，见 §8.10。

数据流是**一条线**，不是两条候选路线：

```
Layer 写 dsl::compute(expr)
   │  to_expr_spec()
   ▼
ExprSpec ──► IR-A/B (expr_opt.hpp: canonicalize/CSE/寄存器分配)      ← 每次 dsl::compute 都在用
   │
   ├─ 无 begin_expr 段 ──► eval_expr ──► IR-D(GlslEmitter 产物) dispatch
   │
   └─ begin_expr 段内 ───► ExprGraph 录制 ──► IR-C fuse_expr_graph ──► 每 kernel 一次 dispatch
                                                    ↑
                                     这一格 = 唯一"未启用"的阶段
```

**IR-A/IR-B 早已被 `dsl::compute` 吸收**：CPU `eval_expr_impl` 与 GPU `eval_expr` 都先调
`canonicalize_expr_spec`（`compute_cpu_engine.hpp:1260` / `compute_gpu_engine.hpp:965`）。
所以"IRA/B/C/D 系列"里，唯独 **IR-C 没接**；IR-D 也只登记了 `GlslEmitter` 一个后端。

**时间线也不支持"三代→四代"的说法**（原始开发分支）：

| 日期 | 提交 | 内容 |
|------|------|------|
| 08-21 | `bf86af2` | `expr_dsl.hpp`（**dsl::compute**） |
| 08-22 | `662cb08` | `begin_expr/end_expr` 接口（step1 of fused kernel） |
| 08-24 | `36caba4` | `expr_graph.hpp`（**IR-C**）+ IR-D |

即 `dsl::compute` **先出现**，`begin_expr/end_expr` 和 IR-C 是随后 1~3 天在同一波里加的。
它们是同一波开发的同一条流水线，不是前后取代关系。

#### 4.2.3 关键非对称：IR-C 在 GPU 是活的，在 CPU 是 no-op（决策前结论，保留为证据）

| | `begin_expr` / `end_expr` 运行时行为 |
|---|---|
| **GPU** | 真录制 + `execute_fused_graph(g)`（`compute_gpu_engine.hpp:184/194/201`）→ **确实融合**（测试里 3 节点 → 1 kernel） |
| **CPU** | `#ifdef NN_EXPR_SCAN` 才动作；注释明写"**普通 CPU 运行：no-op（各表达式直接求值，行为不变——融合是 GPU 优化）**"（`compute_cpu_engine.hpp:41-45, 58`） |

**推论：接 IR-C 对 CPU 的收益是零。** CPU 的瓶颈不是 kernel 数量，而是含归约/matmul 的
表达式走串行标量解释（§3 机制 3）。所以在 CPU 侧讨论"IR-C vs dsl::compute"没有意义。

### 4.3 `dsl::start_expr` / `end_expr`（块式融合 API）🗑️ 已移除
- 原 `expr_dsl.hpp` 的 `ExprBlock`；**仅被测试使用**
- 2026-09-19 随 IR-C 一并删除（连同 `expr_dsl_test` 的对应用例）

### 4.4 `dsl::compute`（当前）
- `expr_dsl.hpp:646-690`，Layer 全面使用。

---

## 5. 遗留物清单（按可清理度排序）

| 项 | 规模 | 判定 |
|----|------|------|
| ~~CUDA 引擎 + backend + `cuda/` + 10 处 `#ifdef`~~ | ~100 KB | 🗑️ **已移除**（快照见 `legacy/cuda`） |
| 旧代数 AST：`algebra_expr.hpp` + `algebra_ops.hpp` + `algebra_compute.hpp` | 27.6 KB | ⚠️ 仅 CPU elementwise 用，与 DSL 模板路径重复 |
| IR-D 的 `CpuEmitter` 残留 | 注释 | ✅ **已清理**（2026-09-25）：`cpu_emitter.hpp` 已删、`expr_emitter.hpp` 头注释已更正为"仅 GlslEmitter 注册"；文档 03 对应说明同步修正 |
| ~~`FusedChainLayer`（IR-C 演示层）~~ | ~60 行 | 🗑️ **已移除**（2026-09-19） |
| ~~`begin_expr`/`end_expr` + `expr_graph.hpp` + `Tensor::virtual_tag_`~~ | ~24 KB | 🗑️ **已移除**（2026-09-19） |
| ~~`eval_cpu` 串行模板路径~~ | — | ✅ **已修复**：`eval_into_span` 已并行化 + 向量化（机制 2，见 §3） |
| `axpy_inplace` | 接口 + 3 引擎实现 | ❌ **无根调用方**（仅 `PrecisionEngine` 转发；optimizer 已迁 `dsl::compute_into`） |
| `broadcast_row_inplace` | 接口 + 3 引擎实现 | ❌ **无根调用方**（仅实现 + 转发） |
| `broadcast_col_inplace` | 接口 + 3 引擎实现 | ⚠️ 仅测试/bench 用 |
| `nn::one_hot`（`nn.hpp`） | 1 函数 | ❌ 无调用点，且与铁律 9（大词表禁物化 one-hot）冲突 |
| `Matrix::multiply_transposed_add_to`（`algebra_matrix.hpp`） | 1 方法 | ❌ 注释自述"全库无调用者（死代码）"，实测 0 调用 |
| `pool_stats()` | 接口 1 个 | ✅ 活（`src/text_train.cpp` 池账本统计） |
| `elementwise_select_scalar_cond` | 接口 + 3 引擎实现 | ❌ 无调用点 |
| `offload_store` / `offload_load` | 接口 + GPU 实现 | ⚠️ 仅测试用（Layer 用 `offload_save/restore`） |
| `row_reduce_max`（引擎算子，非 DSL 叶子） | 接口 + 3 引擎实现 | ⚠️ 无 Layer 调用点 |
| ~~`eval_cpu` 串行模板路径~~ | — | ✅ **已修复**：`eval_into_span` 已并行化 + 向量化（机制 2，见 §3） |
| `NN_EXPR_SCAN` + `expr_registry.hpp` | — | ✅ 活（AOT 管线必需） |

> **不属于遗留物**：IR-A/B（`expr_opt.hpp`）、IR-D（`expr_emitter.hpp`）是同一 IR 流水线的阶段。
> **IR-C（`expr_graph.hpp`）曾是同一流水线的阶段 C，但已于 2026-09-19 整体删除**——原因是它
> **设计完整却无处接线**，且接线代价（逃逸检测 + 自动作用域）远超收益（见 §8.9、§8.10）。

---

## 6. 与"只保留 `dsl::compute` + 纯算子"的差距

目标形态：
1. Layer 只写 `dsl::compute` / `compute_reduce`（不再直接调 35 个 eager 算子）；
2. 引擎保留一组"纯算子"，但**只服务于 DSL 的 lowering**，不再作为 Layer 的公开 API；
3. 干掉重复的代际产物。

按现状，差距在四处：

| 差距 | 现状 | 说明 |
|------|------|------|
| A. Layer 直调算子 | 35 个（2026-09-25 复核） | 要收敛到"只写表达式"，包括 `matmul`/`batched_matmul`/`transpose`/`gather_rows`/`scan_*`/`offload_*` |
| B. 四套求值机制 | 4 套 | 目标只剩 DSL + 引擎 lowering；旧代数 AST、`eval_cpu`、解释器都要重定义为 lowering |
| C. 融合世代 | eager / dsl | **只有 eager 是待收敛的旧世代**；IR-C 已于 2026-09-19 移除（设计无收益点，见 §8.9 + `03-ir-optimization.md` §5.3） |
| D. 死代码 | 无调用点接口（CUDA 全链 ~100 KB、IR-C ~24 KB 已清） | 与目标无关，已清理 |

**注意**：A 与 C 不是纯删除——`scan_prefix_outer` / `outer_col`（RLA/RAPT）、`offload_*`（activation offload）、
`begin_batch/end_batch`（GPU 命令录制）这些**不是普通逐元素/矩阵算子**，DSL 目前表达不了。
所以"只保留 dsl::compute + 纯算子"需要先给这几类定去向（DSL 内建？独立子系统？），
否则会卡在 RLA/ZiPT/offload 上。

---

## 7. 专题：把目标定为"保留 DSL + eager 算子"是否最优

### 7.1 先纠正一个前提：**DSL 并不依赖 eager 算子**

用代码检验（不是推测）：

- **GPU**：`eval_expr`（`compute_gpu_engine.hpp:942`）→ `canonicalize_expr_spec` → `expr_spec_key` →
  `backend_.run_fused_gpu(...)`（`:985`）。全程**不调用** `engine.matmul` / `engine.elementwise_*`。
- **CPU**：`eval_expr_impl`（`compute_cpu_engine.hpp:1251`）的 matmul 段直接调
  `Matrix::multiply_to_span`（`:1432`）；逐元素链用自己的循环。

即 DSL 真正依赖的是 **`algebra_matrix.hpp` 的 Matrix 级内核 + `expr_spec.hpp` 的 IR**，
不是 `ComputeEngine` 的 49 个 virtual。`dsl::matmul` 折成的是 `MatmulSpec`（IR 前置段），
不是对 `engine.matmul` 的调用。

> 所以"DSL 依赖 eager 算子"这句话，只在"eager 算子是 DSL 的 lowering 目标"这个意义上成立，
> 而在"DSL 需要它们公开存在"这个意义上**不成立**。

### 7.2 eager 算子对 DSL 其实分三类

| 类别 | 算子 | 处置 |
|------|------|------|
| **① 与 DSL 重复**（DSL 完全能表达） | `elementwise_unary/binary/binary_scalar`、`broadcast_row/col_inplace`、`add_inplace`、`scale_inplace`、`axpy_inplace`、`accumulate`、`zero` | 应降为 **internal lowering**，不再对层公开 |
| **② DSL 已有对应 IR 叶子** | `matmul`/`batched_matmul`、`row/col_reduce_sum/max`、`row_gather`、`row_mod`、`row_access`、`rotate_half`、`batch_mod`、`batch_col`、`select` | DSL 自带；engine 版本仅供 lowering |
| **③ DSL 表达不了**（真·不可约） | `scan_prefix_outer`/`scan_suffix_outer`/`outer_col`（RLA/RAPT）、`im2col`/`col2im`（CNN 卷积/池化窗口展开）、`offload_store/load/save/restore/create_offload_buffer`、`begin_batch`/`end_batch`/`flush_batch`、`slice_rows`/`insert_rows`/`rearrange_3d`/`transpose`、`create_tensor`/`from_matrix`/`to_matrix`/`clone`/`cast`/`copy_from` | **必须保留为层可见原语** |

### 7.3 最优形态：不是两层并列，而是三档

```
L1  层可见   : dsl::compute / compute_reduce  +  §7.2 ③ 的不可约原语
L2  内部     : eager 算子 = DSL 的 lowering 目标（不再对层公开）
L3  后端     : SIMD / 线程池 / GLSL kernel
```

**判据一句话：凡是能被 DSL 表达的，就不该再作为层的公开算子存在。**

### 7.4 保留 eager 作为公开 API 的代价（但有一个真实好处）

代价：
1. 每次直调 = 一次 kernel dispatch（GPU）/ 一次 Tensor 物化——**这正是 DSL 被造出来要消灭的东西**
   （对比 `eval_expr` 一次 dispatch 完成整条链）。
2. 保留与 DSL 重复的逐元素路径 → 就是今天 CPU 上"旧代数 AST / DSL 模板 / 解释器"三套并存、
   层直调 35 个算子的直接来源。

好处（必须承认）：
- **eager 是闭合世界的逃生舱**。GPU 的 DSL 未命中 AOT 就**硬报错**（`compute_gpu_engine.hpp:996`），
  因为 shader 必须构建期枚举；而 eager 走手写 shader（`shaders/*.comp`），不需要构建期枚举。
  所以 §7.2 ③ 那批结构原语**留在 eager 侧是设计上的必需**，不是历史包袱。

### 7.5 但要真做到"只保留 DSL"，DSL 现在缺两样（原为三样）

1. ~~**输出缓冲 / 原地语义**：`dsl` 没有 `compute_into`~~ → ✅ **已完成**：`dsl::compute_into(eng, expr, dst)`
   已落地（`expr_dsl.hpp`），optimizer / 残差加法 / LayerNorm 梯度累加 / CE 梯度均已迁移（CPU 走
   `eval_into_span` 模板内联，GPU 走 `run_fused_gpu` 的 `output_override`）。
2. ~~**跨表达式融合没接线**：`begin_expr`/`end_expr` + `expr_graph.hpp`（IR-C）无生产调用方~~
   → 🗑️ **已判定为无收益并移除**（2026-09-19）：需要融合的层都以归约为骨架、且要为 backward 缓存
   中间量（§8.9），能融的表达式本就可写成单个 `dsl::compute`。见 `03-ir-optimization.md` §5.3。
3. **CPU lowering**（剩下的一项）—— 逐元素路径已向量化+并行（机制 2），
   但**含归约/matmul 的表达式仍走串行解释器**（机制 3，2026-09-18 实测比等价原语慢 1.5–3.8 倍；
   原诊断报告已删除，结论保留在 §3 与本条）。

### 7.6 结论

方向对（DSL 确实是 IR 优化能力所在），但目标应表述为：

> **"DSL 作为唯一前端 + 一个封闭的不可约原语集合；eager 算子降为内部 lowering 目标。"**

- 若坚持"eager 算子继续**公开**"，等于保留两套逐元素系统，DSL 的融合收益会在层边界漏掉；
  这时的"保留 DSL + eager"只是把当前状态合法化，不是最优。
- 务实路径：先把 §7.2 ① 标记为 legacy 并逐层迁移（LayerNorm / Softmax / Linear / 激活这类纯表达式先行，
  optimizer / RLA / offload 放后面）；同时补 §7.5 的三样，否则迁不动。

---

## 8. 方案（决策前草案）：把 IR-C 吸收进 `dsl::compute`

> ⚠️ **本节是 2026-09-18 的候选方案，最终未被采用。** 2026-09-19 的决定是**直接删除 IR-C**（不做
> 自动作用域/逃逸检测，因为 §8.9 已证明当前层集合里没有可安全接入的融合点）。本节保留用于说明
> "当时考虑过哪条路、为什么放弃"。**已执行的决定见 §8.10。**

### 8.1 目标：用户入口从 3 个收敛到 1 个

| # | 当前入口（决策前） | 使用情况 | 2026-09-19 处置 |
|---|----------|----------|----------------|
| 1 | `dsl::compute` / `compute_reduce` / `compute_into` | ✅ torch 风格，Layer 全面使用 | **保留** |
| 2 | `dsl::start_expr(...) … dsl::end_expr(block)`（`ExprBlock`） | 仅 `expr_dsl_test.cpp` | 🗑️ 已删 |
| 3 | `engine.begin_expr() … engine.end_expr()` | 仅 `FusedChainLayer` + scan | 🗑️ 已删 |

结果：**只留 1**，但**不是**通过"融合由框架自动完成"（原目标），而是通过**删除跨表达式融合能力**。

### 8.2 障碍：逃逸语义现在由**用户**负责（这是必须先解决的）

文档明写："**中间量不逃逸是融合语义约束**"。反例就在 LayerNorm
（`compute_layer_mlp.hpp:499-510`）：

```cpp
auto normalized = dsl::compute(engine,
    dsl::leaf(*diff) * dsl::col_broadcast(std_inv_t), F, B);
Tensor normalized_t = std::move(*normalized);
if (!checkpoint_mode_) normalized_cache_ = normalized_t;   // ← 节点输出被成员持有
return dsl::compute(engine,
    dsl::leaf(normalized_t) * dsl::row_broadcast(gamma_)
    + dsl::row_broadcast(beta_), F, B);
```

若**自动**加作用域，`fuse_expr_graph` 会认为 `normalized` 只有一个图内消费者而把它内联掉
——于是 `normalized_cache_` 指向一个**从未被写入**的 buffer，backward **静默算错**。

> **结论：单靠"自动开作用域"不能做；必须同时把逃逸检测从用户责任变成框架不变量。**

### 8.3 逃逸检测可行：`Tensor` 是 shared_ptr

`Tensor` 内部持有 `std::shared_ptr<MatrixT<P>>`（`compute_tensor.hpp:67-68`），
所以引用计数天然可用：

1. 在 `Tensor` 拷贝构造中，若 `virtual_tag_ != 0`，通知录制图 `note_external_copy(tag)`；
2. `end_expr` 时，凡"图外拷贝数 > 0"的节点 → **pin（必须物化，禁止内联）**。

这样 9.2 的反例会自动被 pin 住，语义安全。

### 8.4 作用域放在哪

| 方案 | 覆盖 | 改动量 |
|------|------|--------|
| (a) `Layer::forward/backward` 加非虚壳（`forward` 非虚 + `forward_impl` 虚） | 所有层 + 所有直接调用方（测试/bench） | 26+26 处，机械化 |
| (b) `Model::forward/backward` 每层包 RAII | 只覆盖 Model 路径；复合层内部直调子层（6 个文件）要各自再包 | 少但易漏 |

**建议 (a)。** 规则：`scope` 构造时**先 flush 已有 pending**（使嵌套安全，得到"最内层 layer"粒度），
析构时 flush。

### 8.5 自动 flush 点（规则化）

1. 任何**非 DSL** 引擎算子（`eval_expr`/`eval_expr_reduce` 之外）——约 22 个：
   `matmul`/`batched_matmul`/`add_inplace`/`scale_inplace`/`axpy_inplace`/`zero`/`accumulate`/
   `transpose`/`slice_rows`/`insert_rows`/`gather_rows`/`scatter_add_rows`/`rearrange_3d`/
   `row/col_reduce_*`/`broadcast_*`/`elementwise_*`/`to_matrix`/`copy_from`/`clone`/`cast`；
2. 逃逸（§8.3）；
3. 作用域结束。
`create_tensor` / `device` 纯分配或查询，**不** flush。

### 8.6 移什么、留什么

**移除（用户可见面）**：
- `ComputeEngine::begin_expr` / `end_expr` → 降为内部 `EngineScope`
- `FusedChainLayer`（演示层，`compute_layer_mlp.hpp:762-810`）
- `dsl::start_expr` / `end_expr`（`ExprBlock`）

**保留（被吸收）**：
- `expr_graph.hpp` / `fuse_expr_graph` / `recording_graph_owner` → 成为引擎内部机制

净效果：用户入口 3 → 1，写法与现有 `dsl::compute` **完全一致**。

### 8.7 代价与收益（如实）

**收益**：GPU 上跨表达式融合自动生效；**且 scan 覆盖率自动提升**——今天只有 `FusedChainLayer`
会录制，这恰恰是 IR-C 一直没接线的根因。

**代价**：
- 构建期复合 spec 数量上升 → `scan_exprs` 与 `fused_registry.hpp` 变大；
- 未命中 AOT 仍是**硬报错**，风险面随 spec 数上升；
- **CPU 仍是 no-op**（`compute_cpu_engine.hpp:58`）——必须先有 CPU 图执行才有意义。

### 8.8 务实建议（决策前）：分两步，可先做第 0 步

- **第 0 步（零语义风险）**：删掉**显式** IR-C API（`begin_expr`/`end_expr`、`FusedChainLayer`、
  `dsl::start/end_expr`），保留 `expr_graph.hpp` 作为内部资产。因为**没有任何生产调用方**，
  这是纯收敛，不承担 §8.2 的逃逸风险。
- **第 1 步（有真实需求再做）**：按 §8.3–8.5 实现自动作用域 + 逃逸检测。

**为什么第 1 步可以等**：`fuse_expr_graph` 只拼接**逐元素链**，归约是硬边界。
而 LayerNorm / Attention 的结构是"归约 → 逐元素 → 归约"，**本来就融不动**；
真正能融的"长逐元素链"用户现在就能直接写成**一个** `dsl::compute` 表达式（DSL 已支持）。
所以 IR-C 在当前层集合上的边际收益 < 它的 API 成本 + 逃逸语义风险。

> 一句话：**你要的"简洁写法"不需要靠 IR-C；而 IR-C 的优化能力，只有当某个层的表达式
> 长到不能写成一个表达式时才开始回本。**

### 8.9 接线可行性实测评估（结论：当前层集合里**无处可安全接入**）

融合条件（`expr_graph.hpp:180-211`）：两节点均**无归约**、同形状、非 `vector_out`、
B 不含 matmul、且 tail **恰好一个消费者**并以 **Linear** 视图使用它。

按此条件逐层扫描，所有"≥2 个 DSL 表达式"的位置都被三类之一挡住：

| 阻挡类型 | 具体位置 | 依据 |
|---|---|---|
| **① 归约是硬边界** | LayerNorm fwd（mean/var 两次 `col_reduce`）、RMSNorm、Softmax（`row_max`/`row_sum`）、Attention 的 m/l/W | `expr_spec_reduce_axis(A/B) != -1 → 放弃`（`expr_graph.hpp:187-188`） |
| **② eager 算子交错** | RLA `rms_norm_backward_`：`term=dsl → diff=eager elementwise_binary → gx=dsl`（`compute_layer_rapt.hpp:226-232`）；LayerNorm step2/step5 的 `elementwise_binary_scalar` | 录制段内 eager 算子会读到**尚未写入**的虚拟 buffer |
| **③ 中间量逃逸到 backward 缓存** | LayerNorm `normalized_cache_`（`compute_layer_mlp.hpp:499-504`）、RMSNorm `normed_cache_`、GPTBlock `residual2_cache_`（`compute_layer_gpt.hpp:228-230`）、Attention `W_re` | 融合要求"tail 只有一个消费者"，而**缓存就是图外的第二个消费者**——`fuse_expr_graph` 目前**看不见**它 |

而**能融的地方早已被写成单个表达式**：GeLU / SwiGLU / ReLU / Softmax / RoPE / 掩码，
代码注释都写着"全部折叠为单个 GPU 融合 kernel（仅 input/output 落显存，无中间 Tensor）"。

**三点结论**：

1. **不简单**：真要接，先得有 §8.3–8.5 的 flush + 逃逸检测（~22 个 flush 点 + `Tensor` 拷贝钩子 +
   `Layer::forward` 非虚壳 26+26 处）。
2. **收益小**：当前**没有任何"安全且可融"的多表达式链**。三类阻挡里最难的是 ③——
   **需要融合的层恰恰都要为 backward 缓存中间量**，这不是巧合，而正是"中间量不逃逸"这条约束的由来。
3. **CPU 零收益**：`compute_cpu_engine.hpp:58`（CPU 的 begin/end 是 no-op）。

**因此不建议现在接 IR-C。** 让它回本的前提是：出现（或写出）一个
**纯逐元素、同形状、中间量不必为 backward 保留**的连续表达式链——那时 §8.3–8.5 的投入才值得。

### 8.10 决策已执行（2026-09-19）：IR-C 整体删除，而非"只删显式 API"

§8.8 的第 0 步建议是"删显式 API、**保留** `expr_graph.hpp` 作内部资产"。**最终决定更进一步：
连 `expr_graph.hpp` 一起删。** 理由：

- 删掉显式 API 后 `expr_graph.hpp` 就**一个调用方都没有**（`recording_graph()` 的唯一消费者是
  被删的 `begin/end_expr`），保留它就是本文档自己批判的"悬空设计"；
- §8.9 已证明**当前层集合无处可安全接入**，即"保留待接线"没有时间表，等于永久的死代码。

**实际删除清单**（代码 + 测试 + 文档，已构建验证）：

| 删除项 | 原位置 |
|--------|--------|
| `expr_graph.hpp`（`ExprGraph`/`fuse_expr_graph`/`recording_graph_owner`/图级计划缓存） | `include/neuralnet.cpp/` |
| `ComputeEngine::begin_expr` / `end_expr`（接口 + CPU/GPU 实现） | `compute_engine.hpp` / `compute_cpu_engine.hpp` / `compute_gpu_engine.hpp` |
| `GpuEngine::execute_fused_graph` | `compute_gpu_engine.hpp` |
| `dsl::start_expr` / `end_expr`（`ExprBlock`） | `expr_dsl.hpp` |
| `FusedChainLayer` | `compute_layer_mlp.hpp` |
| `Tensor::virtual_tag_` / `virtual_tag()` / `set_virtual_tag()` | `compute_tensor.hpp` |
| `src/expr_graph_test.cpp`、`src/expr_fuse_test.cpp` + 两个聚合器里的引用 | `src/` |
| `scan_exprs` 的 `FusedChainLayer` dry-run 段 | `tools/scan_exprs.cpp` |

**保留**：`run_fused_gpu` 的 `output_override`（现服务于 `dsl::compute_into` 的原地目标传递）。

**验证**：重新构建通过；`ctest` 15/15 全绿（`expr_cpu_test` / `expr_gpu_test` 在去掉 IR-C 用例后
仍覆盖 DSL 编译期融合、归约语义、matmul IR 融合、优化 pass 与 AOT 数值）。

**重新立项前提**（避免"再设计一次却仍不接线"）：出现或写出**纯逐元素、同形状、中间量不必为
backward 保留、且长到单个 `dsl::compute` 写不下**的链。届时按"逃逸检测 + 自动作用域 + 自动 flush"
实现（§8.3–8.5），而不是复活显式录制 API。

---

## 9. 盘点用到的命令

```powershell
# 一键复现本文 §2 的两个数字（virtual 数 / Layer 直调集合 / 未直调集合）：
pwsh -File bench\doc_inventory.ps1
```
