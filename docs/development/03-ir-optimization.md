# IR 优化（IR Optimization）—— 为融合算子引入中间表示

> 目标：在**不推翻现有 AOT 闭合世界**的前提下，把 `ExprSpec` 从"直通轻量 IR"演进为带优化 pass 的规范 IR，分阶段提升开发便利与代码质量。
> 状态：现行优化链 = **IR-A（canonicalize：DCE/常量折叠/代数化简/稳定重编号）+ IR-B（CSE + 寄存器分配 liveness）+ IR-D（后端 emitter 抽象）**，分别由 `expr_opt.hpp` / `expr_emitter.hpp` 承载；**IR-C（图 IR + `begin_expr/end_expr` 融合分析）不存在**——当前没有任何跨表达式录制机制（取舍记录见 §5.3）。
> 关联文档：`02-operator-fusion.md`（算子融合）、`10-development-standards.md`（分层铁律）、`08-pitfalls-and-lessons.md`。

## 目录

1. [背景与动机](#背景与动机)
2. [分层红线](#分层红线)
3. [IR 设计目标](#ir-设计目标)
4. [IR 结构（阶段 A：规范化 ExprSpec）](#ir-结构阶段-a规范化-exprspec)
5. [优化 pass 设计](#优化-pass-设计)
6. [后端 emitter 抽象（阶段 D）](#后端-emitter-抽象阶段-d)
7. [阶段落地路径与当前状态](#阶段落地路径与当前状态)
8. [验证策略](#验证策略)
9. [当前状态小结](#当前状态小结)

---

## 背景与动机

当前融合算子生成流水线为四段：

```
Layer 内联表达式 → to_expr_spec 折叠 → ExprSpec → glsl_gen → GLSL → glslc → SPIR-V → fused_registry.hpp
                    （前端）                （IR）        （后端）
运行时：fold → expr_spec_key → find_fused(key) → dispatch（闭合世界）
```

`ExprSpec` 是事实上的轻量 IR：SSA 式扁平指令表、内存访问视图、可序列化（NNEXP）、跨后端、确定性结构 key。它**自带一组确定性优化 pass**（IR-A/IR-B，`expr_opt.hpp`）：折叠产出的指令序列先经 `canonicalize_expr_spec` 规范化再算 key，`glsl_gen` 只消费 canonical 形态。

这组 pass 解决的是"直通 IR"（无 pass 时）的五类问题——它们是本文 pass 设计的动机，也是当前由 pass 自动处理的事项：

1. **子表达式重复**导致超输入/寄存器上限（如 `grad*gamma` 出现 3 次 → 曾超当时的输入上限、被迫手工拆表达式）→ 现由 **CSE** 消除。
2. **无死代码消除、无常量折叠、无代数化简**，shader 携带冗余计算 → 现由 canonicalize 链处理。
3. **无寄存器分配**，`num_regs` 线性增长，受 `EXPR_MAX_REGS=32` 约束 → 现由活跃性分配复用。
4. **后端耦合**：`glsl_gen` 为 GLSL 专用，无法"一份 IR 多后端"（CUDA 后端已整体移除）→ IR-D emitter 抽象，当前注册的只有 GLSL。
5. **跨表达式融合无落地形态**：运行时没有跨表达式录制机制（图级 IR 即 IR-C 不存在，§5.3）；长逐元素链直接写成单个 `dsl::compute`（单个 AOT 融合 kernel）。

本文记录 IR 优化 pass 的设计与当前实现，核心约束是**不破坏闭合世界的确定性 key 匹配**。

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
5. **可扩展性**：IR 保持"单表达式"形态；跨表达式融合（图 IR）**不采用**（§5.3），如需重提须先制造出可安全融合的层。

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
| **寄存器分配** | 活跃性分析（liveness）→ 寄存器复用 → 降低 `num_regs`，让更多表达式落在 `EXPR_MAX_REGS=32` 内 |
| **死寄存器回收** | 结合 DCE 回收不再活跃的寄存器 |

CSE 需处理 `Input/Const/Reduce` 操作数的等价性（视图相同 + 输入相同 + 常量相同才等价）。归约指令的 CSE 需保证归约槽语义一致。

### 5.3 阶段 C：图 IR（已移除）

**IR-C（图级跨表达式融合）已移除，取舍记录见 `docs/history.md`。**（记录含：五条否证依据、删除项/保留项清单、以及"重新立项的前提"。）

### 批内上传免 flush（P2，与图 IR 无关）

- **批内上传免 flush（P2）**：`GpuEngine::from_matrix` 移除 batch 模式下的强制 `end_batch → begin_batch`。安全依据：`from_matrix` 总是新建 GpuTensor，`upload_blocking` 独立提交 + 等待完成，新 buffer 未被正在录制的 batch 引用，独立上传提交先于 batch（队列 FIFO）。`to_matrix` / `copy_from` 仍须 flush（前者读 batch 中刚写的 buffer、后者写 batch 已引用的既有 dst）。

---

## 后端 emitter 抽象（阶段 D）

把 `expr_glsl_gen.hpp` 的 GLSL 专用生成抽象为 emitter 接口（`expr_emitter.hpp`）：

```
IR → GlslEmitter（当前唯一注册后端）
```

- `ExprEmitter` 纯接口（name/generate/generate_reduce）+ `emitter_registry`（按后端名选择工厂）。
- `GlslEmitter`（`expr_glsl_gen.hpp` 的 generate_glsl/generate_glsl_reduce 封装）。
- `gen_fused` 经 emitter 注册表选择后端（默认 glsl）；`--list-backends` 展示可用后端。

> **当前只有 `glsl` 一个注册后端**（`GlslEmitter`）；`gen_fused --list-backends` 可列出。
> 接口 + 注册表即 IR-D 的全部交付：新后端实现 `ExprEmitter` 并注册即可，运行时无需包含该头（与 `glsl_gen` 一致）。

---

## 阶段落地路径与当前状态

| 阶段 | 内容 | 当前状态 |
|--------|------|--------|
| **IR-A** | 确立 ExprSpec 为 IR 规范 + `canonicalize_expr_spec`（DCE/常量折叠/稳定排序）+ 接入 key | **现行**（`expr_opt.hpp`） |
| **IR-B** | CSE + 寄存器分配（liveness） | **现行**（`expr_opt.hpp`） |
| **IR-C** | 图 IR + `begin_expr/end_expr` 融合分析 | **不采用**（归约边界 + 缓存逃逸，无收益点；§5.3，取舍记录在 `docs/history.md`） |
| **IR-D** | 后端 emitter 抽象（多后端，一份 canonical IR） | **现行**（`expr_emitter.hpp`，仅 `GlslEmitter` 注册） |

### IR-A / IR-B 实现要点（`expr_opt.hpp`）

- `fold_constants_and_algebra`（常量池去重 + 保守常量折叠 + 代数化简）、`dead_code_elimination`、`renumber_registers`（寄存器连续重编号 + 常量池清理）、`common_subexpression_elimination`（Fanout 归一化 + 哈希复用）、`allocate_registers_liveness`（liveness 线性扫描，确定性贪心）、`canonicalize_expr_spec`（完整链）。
- **canonical IR 接入 key**：`ExprRegistry::add/contains`、CPU `eval_expr_impl`、GPU `eval_expr/eval_expr_reduce` 全部先 `canonicalize_expr_spec` 再算 key；scan/gen_fused 存 canonical spec → shader 按 canonical 合成。**dispatch 必须用 canonical 的 consts**（折叠可能增删常量池）。
- **关键不变量**：canonicalize 不改变 views/inputs（顺序、内容），只优化 instrs/consts/num_regs → 运行时输入绑定布局不变；输出指令（最后一条）恒为真实寄存器（glsl_gen 输出 `r<last_dst>`）。
- **glsl_gen 配套**：寄存器"先声明、后赋值"（`float r0, r1, ...;` + 纯赋值），兼容 liveness 复用同号寄存器（否则 GLSL redefinition）。
- **关键坑**：① regalloc 复用后归约指令 dst 与逐元素 dst 必须**区段分离**（validate 要求 reduce_dst/elem_dst 按号互斥）；② 各 pass 重映射只处理 `expr_instr_num_operands(op)` 实际使用的操作数（未用 b/c 是默认哨兵 {0,0}，不得当 Reg(0) 重编号）。
- 回归范围：`expr_cpu_test`（`expr_opt_test`/`expr_reduce_test` 等子测试已并入该聚合目标）+ `expr_gpu_test` + gradcheck 系列 + MNIST / text_train 端到端训练冒烟（CPU + GPU）。

---

## 验证策略

canonicalization 与优化 pass 的验证重点：

1. **确定性**：同一 spec 多次 canonicalize 得相同结果；跨编译器（Clang/MSVC）key 稳定。
2. **语义等价**：canonical 前后 CPU 求值结果一致（容差内）；优化不改变浮点语义。
3. **现有测试全量回归**：`expr_cpu_test` / `expr_gpu_test` / gradcheck 系列 / GPT、MNIST 训练冒烟（CPU + GPU）。
4. **上限压力测试**：构造超输入/超寄存器用例，验证 CSE + 寄存器分配后落入限制内。
5. **闭合世界覆盖**：新增优化后，scan 覆盖的路径仍能命中（key 定义在 canonical IR 上，两端一致）。

---

## 当前状态小结

| 目标 | 状态 |
|------|------|
| 轻量 IR + 优化（IR-A + IR-B） | 现行 |
| 图 IR 融合分析（IR-C） | **不采用**（已移除，§5.3） |
| + 后端 emitter（IR-D） | 现行（仅 `GlslEmitter` 注册） |

最大成本集中在**确定性 key 验证**。现有 `ExprSpec` 已是合格轻量 IR，直通、简单、确定性是其最大优点；
IR-C 留下的教训是**"设计完整度不等于价值"**——一个没有生产调用方的优化层，即使实现了也是负资产。
后续演进以"小步演进、不推翻、有真实调用方"为原则推进。