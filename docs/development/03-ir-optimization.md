# IR 优化（IR Optimization）—— 为融合算子引入中间表示

> 目标：在**不推翻现有 AOT 闭合世界**的前提下，把 `ExprSpec` 从"直通轻量 IR"演进为带优化 pass 的规范 IR，分阶段提升开发便利与代码质量。
> 状态：**IR-A（canonicalize：DCE/常量折叠/代数化简/稳定重编号）+ IR-B（CSE + 寄存器分配 liveness）已实施（2026-08-23）**；**IR-D（后端 emitter 抽象）已实施（2026-08-24）**；**IR-C（图 IR + `begin_expr/end_expr` 融合分析，基础版 2026-08-24 实施）已于 2026-09-19 整体移除**——因归约边界 + backward 缓存逃逸，当前层集合里无处可安全接入（详见 §5.3）。
> 关联文档：`02-operator-fusion.md`（算子融合）、`10-development-standards.md`（分层铁律）、`08-pitfalls-and-lessons.md`。

## 目录

1. [背景与动机](#背景与动机)
2. [分层红线](#分层红线)
3. [IR 设计目标](#ir-设计目标)
4. [IR 结构（阶段 A：规范化 ExprSpec）](#ir-结构阶段-a规范化-exprspec)
5. [优化 pass 设计](#优化-pass-设计)
6. [后端 emitter 抽象（阶段 D）](#后端-emitter-抽象阶段-d)
7. [落地路径与实施记录](#落地路径与实施记录)
8. [验证策略](#验证策略)
9. [工作量小结](#工作量小结)

---

## 背景与动机

当前融合算子生成流水线为四段：

```
Layer 内联表达式 → to_expr_spec 折叠 → ExprSpec → glsl_gen → GLSL → glslc → SPIR-V → fused_registry.hpp
                    （前端）                （IR）        （后端）
运行时：fold → expr_spec_key → find_fused(key) → dispatch（闭合世界）
```

`ExprSpec` 已经是事实上的轻量 IR：SSA 式扁平指令表、内存访问视图、可序列化（NNEXP）、跨后端、确定性结构 key。**但它没有优化 pass**——指令序列是表达式模板按求值顺序直接产出的线性代码，`glsl_gen` 逐条原样展开。

由此产生实际问题：

1. **子表达式重复**导致超输入/寄存器上限（如 `grad*gamma` 出现 3 次 → 超 `EXPR_MAX_INPUTS=8`，被迫手工拆表达式）。
2. **无死代码消除、无常量折叠、无代数化简**，shader 携带冗余计算。
3. **无寄存器分配**，`num_regs` 线性增长，受 `EXPR_MAX_REGS=16` 约束。
4. **后端耦合**：`glsl_gen` 为 GLSL 专用，无法"一份 IR 多后端"（CUDA 已停用，目标为 GLSL/CPU 双 emitter）。
5. **跨表达式融合曾无落地形态**：`begin_expr/end_expr` 规划的多表达式融合经图 IR（IR-C）实现后，因归约边界与 backward 缓存逃逸而**无生产调用方**，已于 2026-09-19 移除（§5.3）。

本文给出一个**分阶段**引入 IR 优化的设计，核心约束是**不破坏闭合世界的确定性 key 匹配**。

---

## 分层红线

与算子融合文档一致：

| 红线 | 说明 |
|------|------|
| **算法文本只在 Layer** | 公式只写在 `compute_layer.hpp` / `compute_loss.hpp` |
| **引擎只提供 op-level 原语** | 引擎认"结构"（`reduce(matmul(A,B))`），绝不认"算法名" |
| **融合/优化逻辑归工具/引擎内部** | IR、pass、emitter 都在引擎/工具内部，Layer 无感知 |
| **Shader 是内部实现** | 融合 shader 只存在于 `shaders/` + 引擎 |

IR 属于"引擎/工具内部"，**完全落在红线允许区**，且强化"引擎认结构不认算法名"的哲学。

---

## IR 设计目标

1. **规范性**：把 `ExprSpec` 正式确立为 IR 规范，写清语义、上限、序列化。
2. **可优化性**：提供确定性优化 pass（DCE、常量折叠、CSE、寄存器分配、代数化简）。
3. **闭合世界兼容**：key 定义在 **canonical（优化后）IR** 上，scan 与 runtime 两端一致。
4. **跨后端**：IR → 多 emitter（GLSL / CPU），可选扩展。
5. **可扩展性**：IR 保持"单表达式"形态；跨表达式融合（图 IR）已评估并**明确不采用**（§5.3），如需重提须先制造出可安全融合的层。

---

## IR 结构（阶段 A：规范化 ExprSpec）

沿用现有 `ExprSpec` 数据结构，仅确立为 IR 并加 canonicalization 层（下为**核心四字段节选**——当前完整结构还含 `rparams` 运行时标量池、可选 `matmul` 段（S1）与可选 `fold` 段（P-C1/C2 `FoldSpec`，分块状态归约，注意力 forward 即此结构），见 `expr_spec.hpp`）：

```cpp
struct ExprSpec {                      // 节选：核心字段
    std::vector<ExprInstr> instrs;   // SSA 式指令表
    std::vector<ExprView>  views;    // 与 inputs 一一对应（索引映射）
    std::vector<Scalar>    consts;   // 常量池
    std::uint32_t          num_regs;
    // + rparams / std::optional<MatmulSpec> matmul / std::optional<FoldSpec> fold
};
```

- 指令：`ExprInstr{ op, dst, a, b, c }`，操作数 kind 为 `Reg/Input/Const/Fanout/Reduce`（S7 起增网格索引 `Row/Col/Batch` 与 `Matmul`；`VecState` 仅 fold finalize 可用）。
- 视图：`Linear/RotateHalf/RowMod/RowBroadcast/ColBroadcast` + 归约视图（S7 起增 `RowGather/BatchMod/BatchCol`，后又有 `RowAccess`）。
- 上限：`EXPR_MAX_INPUTS=16 / REGS=32 / INSTRS=64 / CONSTS=16`；fold 段另有独立上限 `FOLD_MAX_BODY/FINALIZE/STATE/VEC`。

### 4.1 canonical IR 定义

> **canonical IR = `canonicalize_expr_spec(spec)` 的输出**，是 key 计算、去重、shader 合成的唯一依据。

关键约定：

```
expr_spec_key(spec) ≡ expr_spec_key(canonicalize_expr_spec(spec))
```

即 **key 一定在 canonical IR 上计算**。scan（构建期折叠时）与 runtime（运行时折叠时）必须都先 canonicalize 再算 key，保证两端一致。

### 4.2 确定性铁律

- **pass 遍历顺序必须固定**（如始终按指令序从前到后、视图按输入序、常量按出现序）。
- **CSE 胜利者选择必须确定**（如取首次出现、同序最小寄存器号）。
- **寄存器分配算法必须确定**（固定贪心顺序，杜绝跨编译器漂移）。
- 参考既有教训：C++ 实参求值顺序未指定曾导致 key 跨编译器不稳定。任何 pass 不得引入依赖未定义求值顺序的逻辑。

---

## 优化 pass 设计

### 5.1 阶段 A：canonicalization（地基）

`canonicalize_expr_spec(spec) → spec`，含：

| pass | 说明 |
|------|------|
| **DCE（死代码消除）** | 从最后一条指令（输出）反向遍历，剔除不影响输出的指令 |
| **常量折叠** | `x+0`、`x*1`、`x*0`、`neg(neg(x))`、`max(x,x)`、比较真值等 |
| **代数化简** | 幂等、结合/交换律的确定性规范化（可选，注意浮点语义） |
| **稳定排序/重编号** | 保证 canonical 形态跨调用一致 |

**注意浮点语义**：代数化简须保守，避免改变数值结果（如 `a+b+c` 的合并顺序会影响舍入）。默认只做**不改变求值语义**的化简。

### 5.2 阶段 B：CSE + 寄存器分配

| pass | 说明 |
|------|------|
| **CSE（公共子表达式消除）** | 哈希指令 `(op,a,b,c)` → 若已存在等价指令则复用其 dst，消除重复子表达式 → 降低输入/寄存器压力，缓解超限问题 |
| **寄存器分配** | 活跃性分析（liveness）→ 寄存器复用 → 降低 `num_regs`，让更多表达式落在 `EXPR_MAX_REGS=16` 内 |
| **死寄存器回收** | 结合 DCE 回收不再活跃的寄存器 |

CSE 需处理 `Input/Const/Reduce` 操作数的等价性（视图相同 + 输入相同 + 常量相同才等价）。归约指令的 CSE 需保证归约槽语义一致。

### 5.3 阶段 C：图 IR —— 已评估并移除（2026-09-19）

阶段 C 的落地形态是：`expr_graph.hpp`（`ExprGraph` / `fuse_expr_graph` / `recording_graph_owner`
/ 图级缓存）+ `ComputeEngine::begin_expr/end_expr` 显式录制 API + `dsl::start_expr/end_expr`
（`ExprBlock`）+ 演示层 `FusedChainLayer`，基础版于 2026-08-24 实施、08-26 增补中间张量消除。

**2026-09-19 决定：不接线，整体删除。** 依据（逐条经代码核对）：

1. **归约是硬边界**。`fuse_expr_graph` 只拼接**纯逐元素链**（两节点均无归约、同形状、
   tail 恰好一个消费者且以 Linear 视图消费）。而 LayerNorm/RMSNorm 的前向是
   "`col_reduce` → 逐元素 → `col_reduce`"、Softmax 是 "`row_max`/`row_sum` → 逐元素"、
   Attention 的 m/l/W 同理——**需要融合的层恰好都以归约为骨架**，融合在这里立即中断。
2. **backward 缓存逃逸**。融合要求"tail 只有一个消费者"，但它要融的层恰恰都要为 backward
   缓存中间量（`LayerNorm::normalized_cache_`、`RMSNorm::normed_cache_`、
   `GPTBlock::residual2_cache_`、Attention `W_re`）——**缓存就是图外的第二个消费者**，
   而录制图看不见它。要让它看见，需要 Tensor 拷贝钩子做逃逸检测 + 自动作用域 +
   ~22 个 flush 点 + `Layer::forward` 非虚壳（26+26 处），成本远高于收益。
3. **能融的地方早已被写成单个表达式**。GeLU/SwiGLU/ReLU/Softmax/RoPE/掩码本就各是一个
   `dsl::compute`（单个 AOT 融合 kernel，只有 input/output 落显存）；当前层集合里
   **不存在"写不进一行"的纯逐元素链**。
4. **CPU 零收益**。`CpuEngine` 的 begin/end 本身是 no-op（融合是纯 GPU 优化）。
5. **无生产调用方 → 悬空设计**。唯一使用方是演示层 `FusedChainLayer`（任何模型工厂都不用它）
   + `scan_exprs` dry-run + 两个测试。设计与实现完整却无人使用，是负资产。

**删除项**：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr` 及其 CPU/GPU 实现、
`dsl::start_expr/end_expr`（`ExprBlock`）、`FusedChainLayer`、`Tensor::virtual_tag_`、
`GpuEngine::execute_fused_graph` 与图级计划缓存，以及 `expr_graph_test` / `expr_fuse_test`
与 scan 的 `FusedChainLayer` dry-run 段。

**保留项**：`run_fused_gpu` 的 `output_override` 参数——它现在服务于 `dsl::compute_into`
（原地目标传递，零分配/零拷贝），与图 IR 无关。

**重新立项的前提**（写清是为了避免"再设计一次却仍不接线"）：出现或写出这样一条链——
**纯逐元素、同形状、中间量不必为 backward 保留、且长到单个 `dsl::compute` 写不下**。
届时按"逃逸检测 + 自动作用域 + 自动 flush"实现，而不是复活显式录制 API。

### 实施记录（批内上传免 flush，与图 IR 无关，保留）

- **批内上传免 flush（P2）**：`GpuEngine::from_matrix` 移除 batch 模式下的强制 `end_batch → begin_batch`。安全依据：`from_matrix` 总是新建 GpuTensor，`upload_blocking` 独立提交 + 等待完成，新 buffer 未被正在录制的 batch 引用，独立上传提交先于 batch（队列 FIFO）。`to_matrix` / `copy_from` 仍须 flush（前者读 batch 中刚写的 buffer、后者写 batch 已引用的既有 dst）。

---

## 后端 emitter 抽象（阶段 D）

把 `glsl_gen.hpp` 的 GLSL 专用生成抽象为 emitter 接口（`expr_emitter.hpp`）：

```
IR → GlslEmitter（当前唯一注册后端）
```

- `ExprEmitter` 纯接口（name/generate/generate_reduce）+ `emitter_registry`（按后端名选择工厂）。
- `GlslEmitter`（`expr_glsl_gen.hpp` 的 generate_glsl/generate_glsl_reduce 封装）。
- `gen_fused` 经 emitter 注册表选择后端（默认 glsl）；`--list-backends` 展示可用后端。

> **注意**：历史上的 `CpuEmitter`（`cpu_emitter.hpp`，生成可编译 C++ 直线代码）**已删除**——
> 其产物从不参与编译（gen_fused 硬编码 glsl），缺陷全隐性，随文件一并移除。
> 现在只剩 `glsl` 一个注册后端；`expr_emitter.hpp:16` 的注释仍提到 `cpu_emitter.hpp`（已知残留）。

---

## 落地路径与实施记录

| 里程碑 | 内容 | 工作量 | 风险 | 收益 |
|--------|------|--------|------|------|
| **IR-A** | 确立 ExprSpec 为 IR 规范 + `canonicalize_expr_spec`（DCE/常量折叠/稳定排序）+ 接入 key | 小（1–2天） | 低 | DCE/常量折叠、确定性地基 |
| **IR-B** | CSE + 寄存器分配（liveness） | 中（3–5天） | 中 | 缓解超限拆表达式 |
| **IR-C** | 图 IR + `begin_expr/end_expr` 融合分析 | 大（1–2周） | 高 | **已评估并移除**（归约边界 + 缓存逃逸，无收益点，§5.3） |
| **IR-D** | 后端 emitter 抽象（GLSL/CPU，已实施） | 中（3–5天） | 中 | 一份 IR 多后端 |

### 实施记录（2026-08-23，IR-A + IR-B 已完成）

- 新增 `include/neuralnet.cpp/expr_opt.hpp`：`fold_constants_and_algebra`（常量池去重 + 保守常量折叠 + 代数化简）、`dead_code_elimination`、`renumber_registers`（寄存器连续重编号 + 常量池清理）、`common_subexpression_elimination`（Fanout 归一化 + 哈希复用）、`allocate_registers_liveness`（liveness 线性扫描，确定性贪心）、`canonicalize_expr_spec`（完整链）。
- **canonical IR 接入 key**：`ExprRegistry::add/contains`、CPU `eval_expr_impl`、GPU `eval_expr/eval_expr_reduce` 全部先 `canonicalize_expr_spec` 再算 key；scan/gen_fused 存 canonical spec → shader 按 canonical 合成。**dispatch 必须用 canonical 的 consts**（折叠可能增删常量池）。
- **关键不变量**：canonicalize 不改变 views/inputs（顺序、内容），只优化 instrs/consts/num_regs → 运行时输入绑定布局不变；输出指令（最后一条）恒为真实寄存器（glsl_gen 输出 `r<last_dst>`）。
- **glsl_gen 配套重构**：寄存器"先声明、后赋值"（`float r0, r1, ...;` + 纯赋值），兼容 liveness 复用同号寄存器（否则 GLSL redefinition）。
- **关键坑**：① regalloc 复用后归约指令 dst 与逐元素 dst 必须**区段分离**（validate 要求 reduce_dst/elem_dst 按号互斥）；② 各 pass 重映射只处理 `expr_instr_num_operands(op)` 实际使用的操作数（未用 b/c 是默认哨兵 {0,0}，不得当 Reg(0) 重编号）。
- 验证：新增 `src/expr_opt_test`；`expr_dsl_test`/`expr_reduce_test`/`tensor_expr_test`/`fused_gpu_test`/`matmul_fusion_test`/`ce_fusion_test`/gradcheck 系列/gpt_checkpoint_test 全绿；MNIST + text_train（CPU/GPU、含 checkpoint-every）端到端训练正常。
- 顺带修复预存在 bug：`text_train --save-interval 0` 触发 `(step+1) % 0` 整数除零崩溃（与 IR 无关）。

---

## 验证策略

canonicalization 与优化 pass 的验证重点：

1. **确定性**：同一 spec 多次 canonicalize 得相同结果；跨编译器（Clang/MSVC）key 稳定。
2. **语义等价**：canonical 前后 CPU 求值结果一致（容差内）；优化不改变浮点语义。
3. **现有测试全量回归**：`expr_dsl_test` / `expr_reduce_test` / `tensor_expr_test`、`fused_gpu_test`、`matmul_fusion_test` / `attn_gradcheck` / `gpt_gradcheck`、GPT / MNIST 训练冒烟（CPU + GPU）。
4. **上限压力测试**：构造超输入/超寄存器用例，验证 CSE + 寄存器分配后落入限制内。
5. **闭合世界覆盖**：新增优化后，scan 覆盖的路径仍能命中（key 定义在 canonical IR 上，两端一致）。

---

## 工作量小结

| 目标 | 状态 |
|------|------|
| 轻量 IR + 优化（IR-A + IR-B） | 已实施 |
| 图 IR 融合分析（C） | **已评估并移除**（§5.3） |
| + 后端 emitter（D） | 已实施（GLSL；CPU emitter 已删） |

最大成本集中在**确定性 key 验证**。现有 `ExprSpec` 已是合格轻量 IR，直通、简单、确定性是其最大优点；
IR-C 的教训是**"设计完整度不等于价值"**——一个没有生产调用方的优化层，即使实现了也是负资产。
后续演进建议以"小步演进、不推翻、有真实调用方"为原则推进。