# 计算引擎现状盘点（最近复核 2026-09-27）

> 本文只记录**当前状态**：引擎接口清单、表达式求值机制、与目标形态的差距。
> 目的：把当前代码里实际并存的计算 API 与遗留物列清楚，作为事实底座；全部结论来自
> 当前工作树的源码与构建配置，复现命令见 `bench/doc_inventory.ps1` 与 §9。
>
> **当前数字**：引擎 virtual **52 个方法**（`bench/doc_inventory.ps1` 口径；M6 新增 `import_impl` 使其由 48 → 49，2026-10-06 IR-C 恢复新增 `begin_expr/end_expr` 49 → 51、P2 多输出新增 `eval_expr_multi_into_impl` 51 → 52。注意 `grep -c "\bvirtual\b"` 原始计 **54** 处 = 52 方法 + 析构 1 + 注释里的词 1，勿混用口径）；Layer/Loss/Optimizer 直调 **22 个**（含 `begin_expr`——由 `Adam::step`/`AdamW::step` 打开 IR-C 录制段；其余 30 个
> 只服务 DSL lowering / 序列化 / CLI / 适配层 / 测试，见 §2）；CPU 求值机制 **2 套**
> （DSL 模板路径 + IR 解释器，见 §3）；ctest **29** 个测试（**25** 个测试目标 + `cnn_test_gpu`
> = `cnn_test --gpu` + `batchnorm_test_gpu` = `batchnorm_test --gpu` + `fusion_custom_layer_example`
> = 库外使用者形态的 AOT 融合端到端门禁 + `expr_fuse_test` = IR-C 图融合 GPU 端到端，
> 见 `AGENTS.md` §7。其中 `fused_gpu_test` = 融合 shader **逐形态** GPU 对拍；`expr_graph_test` = IR-C 图融合分析（CPU 12 用例）；`error_macro_test`
> = 错误处理宏族（`NN_CHECK`/`NN_EXIT`/`NN_TRY_MSG`/`NN_FAIL`）；`kvrec_test`/`nnvocab_test`/`dataset_test`
> = 统一数据集（KVRecord v2 / `.nnvocab` / `.nndataset`，见 `docs/development/19`）；
> `zipt_test` 随 ZiPT 于 2026-10-01 移除，见 `docs/history.md`）。
>
> 历史演进与收敛记录（2026-09-27 计算类原语全量迁 DSL、2026-09-26 算子收敛与遗留物
> 清理、接口数字 58→49 / 直调 35→23 的过程）已移入 `docs/history.md`；**M4（2026-09-30）
> 后直调 23→21**：`from_matrix`/`to_matrix` 退出 L2（铁律 #12 / 17 §3 D11，层自算辅助
> 数据改走 `detail::upload_span/download_span`，它们是自由函数、不计入 `engine.<op>(` 口径）。
> **IR-C（2026-09-19 移除 → 2026-10-06 恢复）**：`begin_expr/end_expr` + `expr_graph.hpp` 现行，
> P1/P2 已落地，`Adam::step`/`AdamW::step` 直调 `begin_expr`（直调 21→22）。移除期记录见
> `docs/history.md`，定位修正与取舍见 `03-ir-optimization.md` §5.3。

---

## 0. 一句话总览

引擎设计历经 5 代（纯 Matrix → forward_gpu 双实现 → eager → begin_expr/end_expr → `dsl::compute`），
当前仍在编译、被调用的形态（IR-C 录制 2026-10-06 恢复，与 `dsl::compute` 并存）：

| 世代 | 现状 |
|------|------|
| 纯 Matrix（无 Tensor） | `algebra_*` 在用，是 `Tensor` 的底层存储 + CPU 手写 kernel |
| `forward_gpu` / `backward_gpu` 双实现 | 不存在（仅注释残留） |
| eager（直接调引擎算子） | Layer 仍直调 **21 个**算子，且全部是基础设施 / 数据搬运 / 状态扫描 / fold 显式登记（见 §2.1）；计算类原语一律走 DSL |
| `begin_expr` / `end_expr`（IR-C 录制图） | ✅ 现行（2026-10-06 恢复：P1 机制 + P2 写穿多输出；`expr_graph.hpp`，见 §4.2/§4.3） |
| `dsl::compute` | ✅ 当前主推，被 Layer 大量使用 |

---

## 1. 引擎实现层：2 个实现，均在用

| 实现 | 文件 | 大小 | 状态 |
|------|------|------|------|
| `CpuEngine` | `compute_cpu_engine.hpp` | 88.5 KB | 默认引擎 |
| `GpuEngine` | `compute_gpu_engine.hpp` | 58.0 KB | Vulkan（`NN_HAS_VULKAN`） |

> 不存在 CUDA 引擎：`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`、`cuda/`
> 目录与全库 `NN_HAS_CUDA` / CLI `--cuda` 均不存在，勿声称支持 CUDA；历史快照见 git 分支
> `legacy/cuda`（移除范围清单见 `docs/history.md`）。

---

## 2. `ComputeEngine` 接口：52 个 virtual 方法

文件：`compute_engine.hpp`；数字可用 `bench/doc_inventory.ps1` 复现（见 §9）。
多精度 Phase 2 加入 `cast_into`/`copy_into`/`supports_native_data_move`/
`supports_expr_precision_variant` 与 `Precision P` 参数的运算类原语。

### 2.1 被 Layer / Loss / Optimizer **直接调用**的算子：22 个

（统计口径：`compute_layer*.hpp` + `compute_loss.hpp` + `compute_optimizer.hpp` +
`model_container.hpp` 中出现的 `engine.<op>(`，排除 `engine_.reset(...)` 这类非算子调用；
复现：`bench/doc_inventory.ps1`）

```
begin_batch, begin_expr, clone, col2im, create_offload_buffer, create_tensor,
end_batch, eval_expr, flush_batch, gather_rows, im2col,
insert_rows, offload_restore, offload_save, outer_col, rearrange_3d,
scan_prefix_outer, scan_suffix_outer, scatter_add_rows, slice_rows,
transpose, zero
```
（10 个计算类原语不在此列：`matmul`/`matmul_with_bias`/`batched_matmul`/
`row_reduce_sum`/`col_reduce_sum`/`col_reduce_max`/`grouped_reduce_sum`/
`grouped_reduce_max`/`add_inplace`/`scale_inplace`/`accumulate` 一律不经 Layer 直调
——Layer 写 `dsl::compute`/`compute_into`/`compute_reduce`（归类见 §7.2）。
这 23 个按设计保留：**基础设施**（begin/end/flush_batch、create/offload_*、
clone、reshape、zero）、**数据搬运**（transpose/slice_rows/insert_rows/
rearrange_3d/gather_rows/scatter_add_rows/im2col/col2im——DSL 表达式是"每输出元素
独立计算"模型，搬运类的不规则索引/原子累加不属于它）、**状态扫描**（scan_prefix_outer/
scan_suffix_outer/outer_col——顺序状态机，DSL 无此语义）、`eval_expr`（fold 登记：
值构造经 `FoldAnchor` 自登记、注意力 5 组合由 scan 显式登记块覆盖，均属 AOT 设计
而非旁路）、`begin_expr`（IR-C
录制段入口：`Adam::step`/`AdamW::step` 打开图录制，`ExprSegment` 守卫收口）。
**`from_matrix`/`to_matrix` 自 M4 起退出本列**（铁律 #12：L2+ 禁用 Matrix 型 I/O
动词），层自算辅助数据改走 `detail::upload_span/download_span/download_vector`
（自由函数，不进 `engine.<op>(` 统计）。）

**改写 Layer 直调点 / 扩展 DSL 时的现行规则**：
1. **scan dry-run 禁止 `(void)` 吞错**：dry-run 必须按真实参数形态调用并在失败时
   abort 带栈——**dry-run 长期绿 ≠ 覆盖存在**，吞错的 dry-run 是静默失效的闭合
   世界（存量表达式 key 恰好处处有登记时不会暴露，新增结构立刻闭合世界硬报错）。
2. **带 alpha 的批量 matmul 写 `dsl::matmul(...) * dsl::rparam(alpha)` 尾链**：
   CPU（累加后乘）/GPU（写出时乘）/DSL（尾指令乘）三端同序，数值等价；alpha 必须
   rparam 不能 const（值进 key 会破坏形状无关融合，与 S7 教训 1 同族）。
3. **新增 ExprViewKind 的落点清单**（以 grouped_reduce 为参照）：`expr_spec.hpp`
   枚举+validate → `expr_dsl.hpp` SpecBuilder helper + 叶子 + 自由函数 +
   `has_reduction_v=false` 显式特化 → `expr_glsl_gen.hpp` `glsl_view_read` case
   → `compute_cpu_engine.hpp` 解释器两处（形状校验 switch + 读取 switch）。
   **param 是否进 key 取决于 `expr_view_has_runtime_param`**：进（结构参数）= case
   内按 `v.param` 编译期展开；不进（运行期视图参数）= case 内只发一个读取调用，
   另需一处生成器侧辅助函数（如分组归约的 `glsl_emit_grouped_reduce_helpers` 发
   `gr_r<i>()` 运行期循环）并在生成器 PC 之后调用。GPU 侧无 per-view 分派（只填 vp
   push constant）；`glsl_vec4_eligible` 白名单与 `glsl_view_uses_row/col` 排除集
   自动正确——无需改。

### 2.2 不被 Layer 直接调用（其余 30 个：DSL lowering / 序列化 / CLI / 适配层 / 测试）

这 30 个正是 §7.3 的"eager 算子 = DSL 的内部 lowering 目标"形态：计算类原语只被
**DSL lowering 内部**（`MatmulRef::prepare*` 调 `eng.matmul/batched_matmul`、
`ReduceViewRef::prepare` 调各 reduce）、**基类 NVI 边界 cast 入口转发**（原 `PrecisionEngine`，P-1 已下沉）与
**测试/bench**（`f16_*_test`/`gpu_f16_test`/`layer_bench`/`f16_cpu_probe` 等）使用。

| 算子 | 实际调用方 |
|------|-----------|
| `eval_expr_into` / `eval_expr_reduce` | `dsl::compute_into` / `dsl::compute_reduce`（`expr_dsl.hpp`）+ 基类 NVI 入口转发 |
| `matmul` / `matmul_with_bias` / `batched_matmul` | DSL lowering（`MatmulRef::prepare*`）+ 基类 NVI 入口转发 + 测试/bench |
| `row_reduce_sum` / `col_reduce_sum` / `col_reduce_max` / `grouped_reduce_sum` / `grouped_reduce_max` | DSL lowering（`ReduceViewRef::prepare`）+ 基类 NVI 入口转发 + 测试 |
| `add_inplace` / `scale_inplace` / `accumulate` | 基类 NVI 入口（accumulate 为非虚入口本体，无 `_impl`）+ 测试/bench/probe |
| `import` / `import_impl` | **跨设备/引擎拉取**（M6 新增，17 §4.3）：`GpuEngine::import_impl` = 原 `ensure_gpu` 的直传路径（16 §2 的 43 处调用点已全部改名 `import`）；基类默认 = 同设备别名/引擎内 cast + 跨设备宿主中转 |
| `from_matrix` / `to_matrix` | **I/O 层**（`domain_*.hpp` 数据集与初值、`model_serialization.hpp`、`cli_*`、测试）——**M4 起 L2+ 禁用**（铁律 #12 / 17 §3 D11） |
| `cast` / `copy_from` | `model_serialization.hpp` + f16 测试 / gradcheck |
| `cast_into` / `copy_into` / `supports_*` | 基类 NVI 边界 cast 的"写回原存储"路径（`compute_engine.hpp`）+ 引擎实现互调（CPU `cast_into` 同精度分支转 `copy_into`）+ `write` 的 f16 目标（经 `copy_from` → f32 上传 + `cast_into`） |
| `pool_stats` | `src/text_train.cpp`（池账本统计） |
| `release_idle_pool_blocks` | `cli/cli_mnist_io.hpp`、`src/text_train.cpp`、`mem_probe.cpp` |
| `submit_scalar_readback` / `poll_scalar_readback` / `scalar_readback_slots` | 标量回读（loss / `ce_fusion_test`），Layer 不直调 |
| `device` | 引擎外部少量判断 |

（已删除算子的历史调用方快照见 `docs/history.md`。）

---

## 3. 表达式求值：**CPU 两套 + GPU AOT**

| # | 机制 | 实现位置 | 并行度（实测） | 谁在用 |
|---|------|----------|----------------|--------|
| 1 | **DSL 模板路径** `eval_cpu(e, rows, cols)` → `eval_into_span` | `expr_dsl.hpp:181` | **并行 + 向量化**（`n≥PARALLEL_THRESHOLD` 走 `nn::parallel_for_samples` 分块，块内 `NN_VECTORIZE_PRAGMA`；不 per-element 重导 `cpu_matrix().span()`） | `dsl::compute` 的**无归约**分支 + `dsl::compute_into` 的逐元素分支 |
| 2 | **DSL IR 解释器** `eval_expr_impl` | `compute_cpu_engine.hpp:1251-1789` | **串行**（逐元素 switch 分派 + 归约前缀重放） | `dsl::compute` 的**含归约/广播/matmul/索引视图**分支（`expr_dsl.hpp:680`）+ `compute_reduce` |
| 3 | **GPU AOT 融合** `eval_expr` | `compute_gpu_engine.hpp:942` | shader 并行 | GPU 全部表达式 |

**分流判据**：`has_reduction_v<E>`（`expr_dsl.hpp:455-486`）。注意它把 **`MatmulRef` 也算作"归约"**
（`:462-464`），所以连 matmul 都被推进了解释器。DSL 用量（`include/neuralnet.cpp` 全量
计数）：`dsl::compute(` 133 处、`dsl::compute_into(` 34 处、`dsl::compute_reduce(` 30 处
（其中含 `matmul`/`broadcast`/归约的落到机制 2）。

**现状判定**：机制 1 与机制 2 是 DSL 在 CPU 上的两条路，一条内联、一条解释——**这是全部
CPU 求值路径**。机制 2 逐元素 switch 分派 + 归约前缀重放，实测比等价原语慢 1.5–3.8 倍
（2026-09-18 诊断实测，结论数字保留）。

---

## 4. 融合世代：**eager** 与 **DSL/IR 栈** 并存

### 4.1 eager（直接调引擎算子）
Layer 仍直调的 22 个算子（§2.1）走的就是这条路；GPU 侧每个都是独立的 backend kernel
dispatch，即"每算子一次 kernel 开销"的求值方式——**计算类原语不经此路径**（一律写
DSL 表达式，见 §2.1/§7.2），eager 只剩基础设施、不可约原语与 IR-C 录制段开合。

### 4.2 `begin_expr` / `end_expr` + `expr_graph.hpp`（IR-C 录制图）
**现行**（2026-10-06 恢复）。`ComputeEngine::begin_expr/end_expr` 是虚入口（引擎 virtual 52 个
方法之一），`expr_graph.hpp` 提供 `ExprGraph`/`ExprGraphNode`/`fuse_expr_graph`/
`recording_graph_owner`、per-graph `node_outputs`、全局唯一占位 tag 分配器 `next_virtual_tag()`，
以及 P2-12 图级计划缓存（`graph_cache_key`/`FusedKernelPlan`/`instantiate_plan`）。GPU 真录制
（`execute_fused_graph`：图 → 融合分析 → kernel 序列 → AOT dispatch，kernel 输出经
`output_override` 直接写进尾节点占位存储 + `run_graph_kernel_into_`）；CPU 普通运行
`begin_expr/end_expr` 是 no-op，构建期收集器 `NN_EXPR_SCAN` 下 `CpuEngine::end_expr` 跑同一套
`fuse_expr_graph` 并登记每个融合 kernel 的 spec，使扫描与运行期一致（闭合世界）。P2 还新增
引擎入口 `eval_expr_multi_into`（NVI + 虚 `eval_expr_multi_into_impl`）与 `ExprSpec.extras`
（多输出写穿，≤4）。IR 流水线现状 = IR-A/B/C/D（`expr_opt.hpp`/`expr_graph.hpp`/`expr_emitter.hpp`）。
2026-09-19 的移除记录（原 §4.2.1–§4.2.3 盘点与提交时间线）作为历史见 `docs/history.md`。

### 4.3 `dsl::start_expr` / `end_expr`（块式语法糖）
**现行**（2026-10-06 随 IR-C 恢复，`expr_dsl.hpp` 的 `ExprBlock`）：让**单个表达式**跨行书写，
`end_expr` 延迟到整块写完再求值。⚠ 与 `ComputeEngine::begin_expr/end_expr`（**跨表达式**图录制）
不是一回事（历史记录见 `docs/history.md`）。

### 4.4 `dsl::compute`（当前）
- `expr_dsl.hpp:646-690`，Layer 全面使用。

---

## 5. 遗留物清单（当前）

| 项 | 规模 | 判定 |
|----|------|------|
| DSL IR 解释器（机制 2：含归约/matmul 表达式的 CPU 串行解释） | `compute_cpu_engine.hpp` 内 | ⚠️ 当前最大遗留——比等价原语慢 1.5–3.8 倍（见 §3 与 §7.5 的 CPU lowering 缺口） |
| `pool_stats()` | 接口 1 个 | ✅ 活（`src/text_train.cpp` 池账本统计） |
| `NN_EXPR_SCAN` + `expr_registry.hpp` | — | ✅ 活（AOT 管线必需） |

> **不属于遗留物**：IR-A/B/C（`expr_opt.hpp`/`expr_graph.hpp`）、IR-D（`expr_emitter.hpp`）是同一 IR 流水线的
> 阶段；eager 算子中 §7.2 ③ 的不可约原语按设计保留为层可见。
> 其余历史遗留项（CUDA 全链、旧代数 AST、`axpy_inplace`/`broadcast_*`/
> `elementwise_*`/`offload_store`·`offload_load`/`elementwise_select_scalar_cond`/
> 引擎 `row_reduce_max` 等无根算子、`nn::one_hot`、`Matrix::multiply_transposed_add_to`、
> `CpuEmitter` 残留）**均不存在**——原清单与判定见 `docs/history.md`（IR-C 不在其列：它已被恢复为现行能力，见 §4.2）。

---

## 6. 与"只保留 `dsl::compute` + 纯算子"的差距

目标形态：
1. Layer 只写 `dsl::compute` / `compute_reduce`（不直接调计算类 eager 算子）；
2. 引擎保留一组"纯算子"，但**只服务于 DSL 的 lowering**，不再作为 Layer 的公开 API；
3. 干掉重复的代际产物。

按现状，差距在四处：

| 差距 | 现状 | 说明 |
|------|------|------|
| A. Layer 直调算子 | **22 个**（全部为基础设施/数据搬运/状态扫描/fold 显式登记/IR-C 录制段开合） | ✅ **计算类原语已全部退出 Layer 直调**：Layer 只写 DSL；保留的 22 个按 §7.2 分类属"必须保留为层可见原语"（见 §2.1）；M4 起 `from_matrix`/`to_matrix` 也已退出 L2 |
| B. 求值机制 | **CPU 2 套 + GPU AOT** | ⚠️ 目标只剩 DSL + 引擎 lowering；余量 = 含归约/matmul 表达式仍走串行解释器（机制 2），即 §7.5 的 CPU lowering 缺口 |
| C. 融合世代 | eager / dsl 并存 | eager 中待收敛的重复逐元素路径已清空，只剩 §7.2 ③ 不可约原语按设计公开；IR-C（图录制融合）现行（见 §4.2 与 `03-ir-optimization.md` §5.3） |
| D. 死代码 | 无 | 无调用点接口已清空（历史清单见 `docs/history.md`） |

**注意**：A 与 C 不是纯删除——`scan_prefix_outer` / `outer_col`（RLA/RAPT）、`offload_*`（activation offload）、
`begin_batch/end_batch`（GPU 命令录制）这些**不是普通逐元素/矩阵算子**，DSL 目前表达不了。
所以"只保留 dsl::compute + 纯算子"需要先给这几类定去向（DSL 内建？独立子系统？），
否则会卡在 RLA/offload 上。

---

## 7. 专题：把目标定为"保留 DSL + eager 算子"是否最优

### 7.1 先纠正一个前提：**DSL 并不依赖 eager 算子**

用代码检验（不是推测）：

- **GPU**：`eval_expr`（`compute_gpu_engine.hpp:942`）→ `canonicalize_expr_spec` → `expr_spec_key` →
  `backend_.run_fused_gpu(...)`（`:985`）。全程**不调用** `engine.matmul` 这类 eager 算子。
- **CPU**：`eval_expr_impl`（`compute_cpu_engine.hpp:1251`）的 matmul 段直接调
  `Matrix::multiply_to_span`（`:1432`）；逐元素链用自己的循环。

即 DSL 真正依赖的是 **`algebra_matrix.hpp` 的 Matrix 级内核 + `expr_spec.hpp` 的 IR**，
不是 `ComputeEngine` 的 52 个 virtual 方法。`dsl::matmul` 折成的是 `MatmulSpec`（IR 前置段），
不是对 `engine.matmul` 的调用。

> 所以"DSL 依赖 eager 算子"这句话，只在"eager 算子是 DSL 的 lowering 目标"这个意义上成立，
> 而在"DSL 需要它们公开存在"这个意义上**不成立**。

### 7.2 eager 算子对 DSL 其实分三类

| 类别 | 算子 | 处置 |
|------|------|------|
| **① 与 DSL 重复**（DSL 完全能表达） | `add_inplace`、`scale_inplace`、`accumulate` | Layer 不直调（一律 `dsl::compute_into` 等）；engine 版本仅作适配层转发 / 内部 lowering 与测试目标（`zero` 属基础设施，见 §2.1） |
| **② DSL 已有对应 IR 叶子** | `matmul`/`matmul_with_bias`/`batched_matmul`、`row_reduce_sum`、`col_reduce_sum`、`col_reduce_max`、`grouped_reduce_sum`/`grouped_reduce_max`、`row_gather`、`row_mod`、`row_access`、`rotate_half`、`batch_mod`、`batch_col`、`select` | DSL 自带；engine 版本仅供 lowering |
| **③ DSL 表达不了**（真·不可约） | `scan_prefix_outer`/`scan_suffix_outer`/`outer_col`（RLA/RAPT）、`im2col`/`col2im`（CNN 卷积/池化窗口展开）、`offload_save`/`offload_restore`/`create_offload_buffer`、`begin_batch`/`end_batch`/`flush_batch`、`slice_rows`/`insert_rows`/`rearrange_3d`/`transpose`、`gather_rows`/`scatter_add_rows`、`create_tensor`/`from_matrix`/`to_matrix`/`clone`/`cast`/`copy_from` | **必须保留为层可见原语** |

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
2. 与 DSL 重复的逐元素 eager 路径若保留，就会在 CPU 上造成多套求值机制并存——重复部分
   已清空：当前 CPU 求值只剩 §3 的两套，Layer 直调的 22 个全属基础设施 / 数据搬运 /
   状态扫描 / 不可约原语 / IR-C 录制段开合。

好处（必须承认）：
- **eager 是闭合世界的逃生舱**。GPU 的 DSL 未命中 AOT 就**硬报错**（`compute_gpu_engine.hpp:996`），
  因为 shader 必须构建期枚举；而 eager 走手写 shader（`shaders/*.comp`），不需要构建期枚举。
  所以 §7.2 ③ 那批结构原语**留在 eager 侧是设计上的必需**，不是历史包袱。

### 7.5 但要真做到"只保留 DSL"，DSL 还缺什么

1. **输出缓冲 / 原地语义**：已具备——`dsl::compute_into(eng, expr, dst)`（`expr_dsl.hpp`），
   optimizer / 残差加法 / LayerNorm 梯度累加 / CE 梯度均已迁移（CPU 走 `eval_into_span`
   模板内联，GPU 走 `run_fused_gpu` 的 `output_override`）。
2. **跨表达式融合**：已由 IR-C 落地——`begin_expr/end_expr` + `expr_graph.hpp` 图融合
   （写穿多输出，`Adam::step`/`AdamW::step` 已接入，见 §4.2）；不需要写穿的表达式仍可写成
   单个 `dsl::compute`（取舍与定位修正见 `03-ir-optimization.md` §5.3）。
3. **CPU lowering（当前唯一缺口）**——逐元素路径已向量化+并行（机制 1），但**含归约/matmul
   的表达式仍走串行解释器**（机制 2，实测比等价原语慢 1.5–3.8 倍，结论见 §3）。

### 7.6 结论

方向对（DSL 确实是 IR 优化能力所在），目标表述为：

> **"DSL 作为唯一前端 + 一个封闭的不可约原语集合；eager 算子降为内部 lowering 目标。"**

- 若坚持"eager 算子继续**公开**"，等于保留两套逐元素系统，DSL 的融合收益会在层边界漏掉；
  这时的"保留 DSL + eager"只是把当前状态合法化，不是最优。
- 当前状态与剩余工程项：§7.2 ① ② 的计算类原语已全部退出 Layer 直调（Layer 只写 DSL，
  见 §2.1），不可约原语按设计保留为层可见；剩余工程项是 §7.5 第 3 条的 CPU lowering。

---

## 8. 方案：把 IR-C 吸收进 `dsl::compute`（已实现）

2026-09-18 的决策前草案（原 §8.1–§8.10：自动作用域 + 逃逸检测 + 自动 flush）与其后
2026-09-19 的删除记录作为历史见 `docs/history.md`。**2026-10-06 定位修正（非否决，系当年实现
不完整）后已恢复并通用化**：现行形态 = 显式录制 API（`engine.begin_expr()/end_expr()` +
`ExprSegment` RAII 守卫）承载图级融合，P1（链融合）+ P2（写穿多输出）已落地，
`Adam::step`/`AdamW::step` 为首个生产调用方（见 §4.2 与 `03-ir-optimization.md` §5.3）。

---

## 9. 盘点用到的命令

```powershell
# 一键复现本文 §2 的两个数字（virtual 数 / Layer 直调集合 / 未直调集合），
# 并输出第 [4] 节 L2+ 分层审计（铁律 #12 门禁：L2-VIOLATIONS 必须为 0）：
pwsh -File bench\doc_inventory.ps1
```
