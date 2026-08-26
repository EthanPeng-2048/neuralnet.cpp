# 算子融合二期（Operator Fusion II）—— 手写融合算子收敛为 IR 融合

> 状态：**实施完成（S1-S5、S7 已完成；S6 自动窗口搁置）** · 关联文档：`09-operator-fusion.md`（M1-M7 一期）、`11-ir-optimization.md`（IR-A/B/C/D）、`13-optimize-proposal-list.md`（P2-05 / P2-10）
> 目标版本：在既有 AOT 闭合世界（`scan_exprs` → `gen_fused` → 内联 SPIR-V）内，让 **matmul 参与 IR 融合**，并把**跨 kernel 融合自动化**，最终**删除一期手写融合原语**（M4-M6 的 `batched_matmul_*` / `col_softmax_*`）。
> 对应提案：`docs/13` 的 **P2-05（跨 matmul 融合）** 与 **P2-10（跨 kernel 自动融合）**，并为 **P7-01（Flash-Attention 效果）** 提供融合算子承载。
>
> ## 实施状态（2026 二期落地）
>
> | 阶段 | 状态 | 交付物 |
> |------|------|--------|
> | S1 IR 地基 | ✅ | `ExprSpec.matmul`（`MatmulSpec`）+ `ExprOperandKind::Matmul`；`validate_expr_spec`/`expr_spec_key`/`expr_spec_reduce_axis` 兼容 matmul 段；key 与形状无关（`k` 不进 key，`transA/transB/a_input/b_input` 进 key） |
> | S2 CPU 正确性 | ✅ | `CpuEngine::eval_expr` 支持 matmul 段（matmul 预计算 + 逐元素链，`eval_expr_reduce` 经归约指令消费 matmul 输出）；新增 `expr_matmul_test` |
> | S3 GLSL 生成 | ✅ | `generate_glsl_matmul`（共享内存分块 + vec4）；`gen_fused`/`scan_exprs`/`run_fused_gpu` 接入；`fused_gpu_test` matmul 融合用例 |
> | S4 Layer 迁移（线性） | ✅ | `Linear::forward` 改走 `dsl::compute(matmul(W,x) + row_broadcast(b))` |
> | S5 跨归约/跨 matmul 链 | ✅ | 融合分块矩阵乘法（16×16 线程/64×64 块/4×4 寄存器分块/vec4 转置共享内存 + `eval_tail` 函数）；matmul+归约（`row_max(matmul)`、`row_sum(exp(matmul-rm))`）经 `generate_glsl_reduce` 内联点积；图融合允许 matmul 节点拼接尾逐元素节点 |
> | S6 自动窗口（P2-10） | ⏸ | 搁置（用户决策） |
> | S7 删除 M4-M6 | ✅ | **全面替换并删除**：`batched_matmul_reduce/softmax_denom/softmax_apply`（M4）、`batched_matmul_softmax_backward_q/kv`（M6）、`col_softmax_denom/col_softmax_sparse_forward`（M5）7 个原语全部删除（接口 + CPU/GPU/CUDA 实现 + 7 个手写 shader + vk pipeline + 测试改造）。IR 扩展：`MatmulSpec.batch`（批量语义，形状参数不进 key，dispatch z=batch）、`ExprOperandKind::Row/Col/Batch`（网格索引操作数）、`ExprViewKind::RowGather/BatchMod/BatchCol`（标签行收集/按批次索引/按批次列切片） |

---

## 1. 动机与方向（为什么做二期）

一期（09 文档 M1-M7）用**手写 op 级融合原语**解决了 GPT+Vulkan 训练显存/开销问题：

| 原语 | 作用 |
|------|------|
| `batched_matmul_reduce` / `..._softmax_denom` / `..._softmax_apply` | 两趟式注意力 forward（不物化 `seq²`） |
| `batched_matmul_softmax_backward_q/kv` | 两趟式注意力 backward（重算 W） |
| `col_softmax_denom` / `col_softmax_sparse_forward` | 稀疏交叉熵（大词表不物化 softmax） |

这些原语虽符合"引擎认结构不认算法名"（`docs/09` §0.3 红线），但存在长期问题：

1. **不可组合**：每个融合模式都是独立 shader + 独立 dispatch + 独立 bindings。新模式（如 `linear+bias+activation`、RoPE+attention、更多归一化变体）都要另写 kernel，工作量大、易错。
2. **与 IR 融合重复**：`IR-A/B/C/D`（`11` 文档）已能表达"逐元素 + 归约 + 广播"链，并由 `scan_exprs/gen_fused` 自动合成 shader。手写原语占用了本可交给 IR 的空间，造成两套融合机制并存、心智负担高。
3. **难以支撑 Flash-Attention 类效果**（P7-01）：注意力是 `matmul → 归约(max/denom) → 广播 → matmul` 的链。一期手写原语已实现两趟式，但更细的 SRAM tiling / online-softmax 需要跨 matmul 融合，手写每一变体成本过高。

**二期方向（本次定稿）**：
- **不再新增任何手写融合原语**。
- **让 matmul 成为 IR 的一等公民**：`ExprSpec` 表达 `matmul(A,B)` 作为逐元素链的起始段，`glsl_gen` 从 IR 结构合成"matmul + 尾逐元素链"的单 kernel。
- **跨 kernel 融合自动化（P2-10）**：消除手工 `begin_expr/end_expr`，让真实 Layer 的连续可融合表达式在 `eval_expr` 内自动并入窗口。
- **逐步迁移并删除 M4-M6 手写原语**：用 IR 融合重写注意力/交叉熵路径，最终删掉手写 shader。

> 约束重申（红线，`docs/09` §0.3 不可逾越）：
> - 引擎只提供 op-level 原语与结构融合，**绝不出现** `attention`/`softmax`/`flash_attn` 命名的接口。
> - 算法文本只在 Layer；融合/代码生成归工具与引擎内部。
> - AOT 闭合世界：`scan_exprs` dry-run 覆盖所有被融合的 Layer 路径，未命中硬报错、不静默回退。

---

## 2. 现状与差距

### 2.1 当前 IR 能力（已实现）
- `ExprSpec`：扁平 SSA 指令表（`ExprInstr`）+ 输入视图（`ExprView`：Linear/RotateHalf/RowMod/归约/广播）+ 常量池。**纯逐元素 + 行/列归约 + 广播**。
- 上限：指令 ≤64、寄存器 ≤16、输入 ≤8、常量 ≤16。
- 归约轴语义：`expr_spec_reduce_axis` 判定整条表达式是逐元素 / 行归约 / 列归约。
- 图 IR（`expr_graph.hpp`）：`begin_expr/end_expr` 录制图 → `fuse_expr_graph` 贪心拼接逐元素链 → 单 kernel。`P2-12` 已加跨 step 图级缓存。
- AOT 管线：`scan_exprs`（CPU dry-run 收集结构）→ `gen_fused` → `glsl_gen` 生成 GLSL → glslc → 内联 SPIR-V。

### 2.2 关键差距
| 差距 | 现状 | 二期目标 |
|------|------|----------|
| **matmul 不参与 IR** | matmul 是引擎原语（`matmul_gpu` dispatch），`scan_exprs` 不收集 | `ExprSpec` 表达 `matmul(A,B)` 段，`glsl_gen` 生成含 matmul 的融合 shader |
| **融合需手工标记** | 仅 `FusedChainLayer` 用 `begin_expr/end_expr`；真实 Layer 每表达式独立 dispatch | `eval_expr` 自动维护融合窗口，Layer 无感 |
| **跨归约/跨 matmul 链** | 单表达式内归约已融合；跨节点归约向量复用未做 | 融合分析允许 `elementwise → reduce → elementwise → matmul` 链 |
| **手写原语并存** | M4-M6 手写 shader + 专用 dispatch | 迁移到 IR 后删除 |

---

## 3. IR 扩展设计：matmul 参与融合

### 3.1 `MatmulSpec`（前置 matmul 段）

`ExprSpec` 增加一个**可选的 matmul 段**（位于逐元素指令之前），表达 `C = op(A, B)` 作为逐元素链的起始寄存器：

```cpp
// 前置 matmul 段：C(rows, cols) = op(A, B)，结果作为逐元素链的"寄存器 0"
struct MatmulSpec {
    std::uint8_t a_input = 0;   // A 是第几个输入（0-based，指向 views/inputs）
    std::uint8_t b_input = 0;   // B 是第几个输入
    std::uint8_t transA  = 0;   // 1 = A 存储为 (K, M)，按 A^T 使用
    std::uint8_t transB  = 0;   // 1 = B 存储为 (N, K)，按 B^T 使用
    std::uint32_t k      = 0;   // 求和维度（形状参数，运行时 push constant）
};
```

- **语义**：`C[r][c] = Σ_k opA(r,k) * opB(k,c)`，输出 `(rows, cols)`。
- **逐元素链引用**：新增操作数 kind `Matmul = 5`（引用 matmul 结果寄存器，按 `(r,c)` 读取）。等价于把 matmul 输出当作一个"虚拟输入寄存器"，供 `Add/Sub/Mul/…` 消费。
- **形状语义扩展**：matmul 的 A/B 形状 `(M,K)/(K,N)` **与逐元素输出网格 `(M,N)` 不同**。这是对"所有输入同形状"假设的**定向放宽**：
  - `MatmulSpec.a_input/b_input` 指向的两个输入按 matmul 语义解释（M/K/N 由 dispatch 提供）。
  - 其余逐元素输入的视图/形状仍要求 `(M,N)`（如 bias、残差）。
  - 运行时 dispatch 需要 A/B/bias 的实际 shape → 由 `GpuEngine::eval_expr` 从输入张量推导，与 `k`（v-p 形状参数）一起填充。

### 3.2 key 与形状无关
- `transA/transB` 是**结构** → 进 `expr_spec_key`。
- `k` 是**形状参数** → **不进 key**（同 P2-13 的 RowMod/RotateHalf 处理），作为 push constant `vp` 槽运行时填充。同一结构不同 K 共享一个融合 shader。

### 3.3 `expr_spec_reduce_axis` 扩展
- 含 matmul 段的 spec：matmul 段本身是"输出 `(M,N)` 的矩阵"，无归约轴；逐元素段若有归约（如 `row_sum(matmul(x))`）→ 按归约轴处理。matmul 段与归约段的组合（注意力 `bmm_reduce`）需支持"matmul 后沿某轴归约"。

### 3.4 CPU 正确性
- `CpuEngine::eval_expr` 遇到 matmul 段先做一次 matmul（参考实现），结果写入寄存器 0，再执行逐元素链（同现有归约预计算模式）。

### 3.5 上限
- matmul 段不额外消耗逐元素寄存器（它作为"寄存器 0"的源头），但需计入输入上限（A、B + 逐元素输入 ≤8）。
- 若 matmul + 后续链超限 → 融合分析保守放弃（各自成 kernel），与 IR-C 现状一致。

---

## 4. 跨 kernel 自动融合（P2-10）

### 4.1 目标
消除手工 `begin_expr/end_expr`。真实 Layer（如 LayerNorm 的 5 个 `dsl::compute`）连续可融合表达式自动并入窗口，遇不兼容自动关窗执行。**Layer 无感知**。

### 4.2 机制：eval_expr 自动窗口
- `GpuEngine` 维护 `std::optional<ExprGraph> auto_window_`（线程局部，与 D3/图级缓存一致）。
- `eval_expr/eval_expr_reduce` 被调用时：
  1. 若窗口为空 → 开新窗口，加入当前表达式，返回占位 Tensor。
  2. 若窗口非空 → 尝试把新表达式并入当前窗口的图（`ExprGraph::add_node`）；若节点可与窗口内已有节点融合（同形状、归约边界允许、依赖关系成立）→ 并入；否则 **flush 窗口**（`execute_fused_graph` 物化已有节点）+ 为新表达式开新窗口。
- **窗口关闭（flush）触发点**（统一集中式检查，防漏检）：
  - 下一个 `eval_expr` 不兼容时（见上）。
  - **任何"消费 Tensor"的引擎入口**（`matmul`/`batched_matmul`/`elementwise_*`/`row|col_reduce_*`/`transpose`/`rearrange_3d`/`axpy`/`to_matrix`/下载等）收到**属于当前窗口的占位 Tensor** 时，先 flush。
  - `end_batch`/`flush_batch` 边界（若窗口残留）自动 flush。
- **占位 Tensor 识别**：`virtual_tag` 已存在（IR-C 复用）。新增辅助 `bool tensor_in_auto_window(const Tensor&)`，各入口统一调用 `flush_auto_window_if_needed(span<const Tensor>)`。

### 4.3 正确性与风险缓解
- **铁律 8 确定性**：窗口融合决策基于结构（`fuse_expr_graph` 已确定），不改变数值；P2-12 图级缓存保证同结构复用。
- **主要风险**：漏检某入口导致占位 Tensor 被当作真实 buffer → 读未初始化内存 → 静默错。缓解：
  1. 所有原语入口**统一**走 `flush_auto_window_if_needed`（集中在 `GpuEngine` 一个类内，可 grep 审计）。
  2. 逐 Layer 回归：`gpt_gradcheck` / `fused_gpu_test` / `expr_fuse_test` / 各 `*_gradcheck` 全绿。
  3. 提供 `NN_AUTO_FUSION=0` 开关可禁用（回退现状逐表达式 dispatch），便于二分定位。
- **默认关闭窗口到 Layer 语义边界**：为避免误融合跨 Layer 的 matmul/归约边界，窗口默认在"遇到的第一个 matmul / 归约输出 / 形状变化"时 flush（与 `fuse_expr_graph` 的融合边界一致）。

### 4.4 与 P2-12 的关系
- P2-12（图级缓存）已消除"命中缓存时"的融合分析 CPU 开销。
- P2-10 消除"未命中/首步"时的手工窗口与重复 dispatch，并让**真实 Layer** 落入图级缓存的受益范围。

---

## 5. 分阶段实施路线

> 每阶段独立可构建、可测试、可回归；阶段间不破坏既有功能。

| 阶段 | 内容 | 交付物 / 验证 |
|------|------|----------------|
| **S1 IR 地基** | `ExprSpec` 增加 `MatmulSpec` + `ExprOperandKind::Matmul`；`validate_expr_spec`/`expr_spec_key`/`expr_spec_reduce_axis` 兼容 matmul 段 | 构建通过；`expr_spec_test`/`expr_opt_test` 回归绿 |
| **S2 CPU 正确性** | `CpuEngine::eval_expr` 支持 matmul 段（matmul 预计算 + 逐元素链） | 新增 `expr_matmul_test`：`matmul+bias` 融合 vs 参考；CPU err=0 |
| **S3 GLSL 生成** | `glsl_gen` 生成含 matmul 的融合 shader（matmul 分块 + 尾逐元素链）；`gen_fused`/`scan_exprs` 管线接入 | `fused_gpu_test` 新增 matmul 融合用例（GPU vs CPU）；AOT 命中 |
| **S4 Layer 迁移（线性）** | `Linear`/`FeedForward` 的 `matmul+bias+activation` 改走 `dsl::compute`（含 matmul 段） | `gpt_gradcheck` / MNIST/GPT 训练回归 |
| **S5 跨归约/跨 matmul 链** | 融合分析支持 `elementwise → reduce → elementwise → matmul`（注意力 forward/backward 结构） | 注意力相关 gradcheck 全绿 |
| **S6 自动窗口（P2-10）** | `GpuEngine` 自动融合窗口 + 全原语 flush 检查 | `expr_fuse_test` 扩展"真实 Layer 无标记融合"；各 Layer gradcheck 回归 |
| **S7 迁移并删除 M4-M6** | 用 IR 融合重写注意力/交叉熵路径，删除手写 `batched_matmul_*` / `col_softmax_*` shader 与接口 | 全量 ctest；训练冒烟（CPU+GPU） |

### 依赖关系
- S1→S2→S3 串行（IR → CPU → GPU 生成）。
- S4 依赖 S3；S5 依赖 S3；S6 相对独立（可在 S3 后并行推进）；S7 依赖 S5+S6。

---

## 6. 迁移与删除 M4-M6 的计划（S7 详述）

1. **保留接口稳定性**：迁移期间 `batched_matmul_*` 等接口保留但改为内部走 IR 融合路径（或逐步让 Layer 改用 `dsl::compute`），保证训练不中断。
2. **逐原语替换**：
   - 两趟式注意力 forward → S5 的 `matmul → 归约 → matmul` IR 链。
   - 稀疏交叉熵 → S5 的 `col_softmax_denom/sparse` IR 结构（`col_reduce_max` + `exp` + `row_gather`）。
3. **删除顺序**：先删 GPU shader（`bmm_*.comp` / `col_softmax_*.comp`），再删 `ComputeEngine` 虚接口，最后删 CPU/CUDA 实现与 `vk_backend` dispatch。
4. **回归保障**：每一步删除前，对应 IR 融合路径已在 `fused_gpu_test` / `ce_fusion_test` / `matmul_fusion_test` 覆盖；删除后全量 ctest + CPU/GPU 训练冒烟。

---

## 7. 验证与回归策略

- **确定性**：同一结构经 `scan_exprs` 与运行时 `eval_expr` 两端 canonicalize 后 key 一致（闭合世界两端一致）。
- **语义等价**：matmul 融合 spec 的 CPU 求值与"独立 matmul + 逐元素"参考逐位/容差一致。
- **现有回归**：`expr_*_test` / `fused_gpu_test` / `matmul_fusion_test` / `ce_fusion_test` / `attn_gradcheck` / `gpt_gradcheck` / `rmsnorm_gradcheck` / `softmax_gradcheck` / `gpt_checkpoint_test` 保持全绿。
- **形状无关**：matmul 的 `k` 不进 key，测试覆盖不同 K（含非对齐）命中同一融合 shader。
- **并发**：自动窗口线程局部（D3 一致），多线程 forward/backward 隔离。

---

## 8. 风险与开放问题

| 项 | 说明 | 应对 |
|----|------|------|
| **matmul 形状语义** | A/B 与逐元素输入形状不同，破坏"输入同形状"假设 | S1 明确定义输入顺序与 shape 推导；dispatch 从实际张量推 M/N/K |
| **自动窗口漏检** | 占位 Tensor 被未检查入口误用 → 静默错 | 集中式 `flush_auto_window_if_needed` + 逐 Layer gradcheck + `NN_AUTO_FUSION=0` 开关 |
| **GLSL matmul 生成** | 生成含 matmul 分块的 shader 工作量大 | S3 先复用 `matmul_tiled.comp` 逻辑生成（结构驱动，非手写固定 kernel） |
| **上限压力** | matmul + 尾链可能超输入/寄存器上限 | 融合分析保守放弃（回退独立 kernel）；S5 再评估上限动态化（P2-07） |
| **M4-M6 删除回归面** | 注意力/交叉熵路径广泛 | 分阶段删、每步先有 IR 覆盖再删 |

---

## 9. 里程碑小结

| 里程碑 | 内容 | 关联文档 |
|--------|------|----------|
| **M-II-1** | `ExprSpec` 支持 matmul 段（S1+S2：IR + CPU 正确性） | 本设计 §3 |
| **M-II-2** | `glsl_gen` 生成 matmul 融合 shader + AOT 接入（S3） | 本设计 §3.3/§5 |
| **M-II-3** | 跨归约/跨 matmul 链融合（S5，注意力/CE 结构） | 本设计 §3.3/§5 |
| **M-II-4** | 跨 kernel 自动窗口（S6，P2-10） | 本设计 §4 |
| **M-II-5** | 删除 M4-M6 手写原语（S7） | 本设计 §6 |
| **M-II-6** | Flash-Attention 效果经融合算子实现（P7-01） | 本设计 §5 / `13` P7-01 |

> 每里程碑增量、可独立验证，符合项目"小步演进、不推翻"原则（`11` §9）。
