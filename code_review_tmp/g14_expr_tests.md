# g14 表达式/IR 测试代码审查

## 模块概览

| 测试文件 | 被测模块 | 行数 | 执行环境 | batch>1 |
|----------|---------|------|---------|---------|
| expr_dsl_test.cpp | expr_dsl.hpp (DSL 求值 + RoPE) | 312 | CPU only | ❌ (ROWS=64, COLS=4) |
| expr_fuse_test.cpp | begin_expr/end_expr 块式融合 (IR-C GPU) | 278 | GPU (Vulkan) | ✅ (F=8, B=6) |
| expr_graph_test.cpp | expr_graph.hpp (IR-C 图融合) | 641 | CPU only | ❌ (rows≤3, cols≤5) |
| expr_matmul_test.cpp | expr_matmul (MatmulSpec + IR 融合) | 352 | CPU only | ❌ (M≤4, K≤6, N≤5) |
| expr_opt_test.cpp | expr_opt (fold+DCE+CSE+regalloc) | 524 | CPU only | ❌ (R≤8, C≤7) |
| expr_reduce_test.cpp | expr_reduce (行/列归约) | 369 | CPU only | ❌ (R≤5, C≤7) |
| ce_fusion_test.cpp | ce_fusion (交叉熵融合 IR 组合) | 363 | CPU + GPU | ✅ (C=8, N=12) |
| tensor_expr_test.cpp | tensor_expr (DSL compute CPU) | 187 | CPU only | ❌ (R=4, C=5) |
| nan_repro_test.cpp | NaN 回归 (step-474 doc 边界) | 156 | CPU + GPU | ✅ (batch=8, seq=256) |

**期望值来源总评**：9 个测试全部使用独立推导的参考实现（手写循环/独立引擎调用），无自证预言。具体：
- `expr_dsl_test`：手写 `rope_ref()` 逐元素参考 (L54-67)
- `expr_fuse_test`：CPU 引擎逐节点求值 vs GPU 融合
- `expr_graph_test`：`eng.eval_expr` 逐节点参考 + 手工数值验证
- `expr_matmul_test`：`eng.matmul` + `ref_matmul_bias_relu` 独立参考 (L84-94)
- `expr_opt_test`：原始 spec vs canonical spec 数值对比 (L317-366)
- `expr_reduce_test`：手写循环参考 (L92-100, L114-122 等)
- `ce_fusion_test`：独立 `ref_denom` / `ref_sparse_forward` 参考 (L46-96)
- `tensor_expr_test`：手写 `ref_binary` 参考 (L78-91)
- `nan_repro_test`：回归测试，检查输出无 NaN（非正确性测试）

## 发现

### P0

无

### P1

**P1-1 `nan_repro_test.cpp:130` — NaN/Inf 传播断言缺失，仅检测最终输出有限性**

证据：测试仅在 L130 检查 `!std::isfinite(sp[i])`，即最终 logits 是否有非有限值。未测试中间计算的 NaN/Inf 传播行为：
- 未验证 exp(NaN)=NaN、exp(inf)=inf、exp(-inf)=0
- 未验证 log(0)=-inf、log(NaN)=NaN
- 未验证 0/0=NaN、inf/inf=NaN
- 若某实现将 NaN 静默吞掉（如 `-ffast-math` 下 exp(NaN)→0），测试仍会通过

项目明确不使用 `-ffast-math`（保 NaN/Inf），但测试未断言这一行为。当前测试是 step-474 回归测试（"期望无 NaN"），但名称 `nan_repro_test` 暗示应覆盖 NaN 传播语义。

建议：增加单元断言验证 exp/log 在 NaN/Inf 输入下的传播行为（如 `exp(NaN) → isfinite==false`、`log(0) → isinf(-inf)`），或至少在注释中明确声明此测试仅为回归测试、非 NaN 传播单元测试。

**P1-2 `ce_fusion_test.cpp:201` — Row 操作数仅在纯逐元素上下文测试，matmul+行归约语义未覆盖**

证据：ce_fusion_test 在 L201、L238、L279 使用 `nn::dsl::row()` 做标签比较（纯逐元素上下文）。已知 IR 缺陷（g10 报告）："Row 操作数在纯归约/matmul+行归约下语义不同（row 全局行号 vs batch 内行号）"。9 个测试中无一测试 Row 在 matmul+行归约上下文的语义。

建议：增加一个测试用例验证 `row_sum(matmul(A,B))` 场景下 Row 操作数的语义（全局行号 vs batch 内行号），或在 expr_matmul_test 中增加 Row+matmul 组合测试。

**P1-3 无 BatchCol/BatchMod 视图测试覆盖**

证据：grep 确认 9 个测试文件中无一使用 `batch_col()`、`batch_mod()`、`BatchCol`、`BatchMod`。这些视图类型在 attention 层中使用（docs/14 §S7），且有已知 IR 缺陷（g10："BatchMod/BatchCol 引用未声明 batch"）。测试套件对这两个关键视图类型零覆盖。

建议：在 expr_reduce_test 或新增测试中，增加 BatchCol/BatchMod 视图的折叠+求值+validate 测试。

### P2

**P2-1 6 个纯 CPU 表达式测试均为 batch=1**

证据：expr_dsl_test (ROWS=64, COLS=4)、expr_graph_test (rows≤3, cols≤5)、expr_matmul_test (M≤4)、expr_opt_test (R≤8)、expr_reduce_test (R≤5)、tensor_expr_test (R=4) 均为 batch=1（矩阵只有一组数据）。铁律 5："batch=1 时布局 bug 不可见"。

注：表达式层 batch 仅体现为矩阵列数，batch>1 在此层面对布局 bug 的检测意义低于 attention/GPT 层面。但 Row 操作数在 batch>1 时的语义差异（P1-2）仍需覆盖。

建议：至少在 expr_reduce_test 中增加 batch>1（如 R=4, C=8 视为 2 batch×4 classes）的归约测试。

**P2-2 `expr_graph_test.cpp:73-105` — 图融合测试仅覆盖 Linear + RowBroadcast 视图**

证据：expr_graph_test 中所有手工构造的 ExprSpec 仅使用 `nn::expr::linear()` (L73, 89, 90, 105, 313, 441, 442, 497, 551, 559) 和 `nn::expr::row_broadcast()` (L292) 视图。未测试 ColBroadcast、RowReduceSum、ColReduceSum 等视图在图融合边界的行为。

建议：增加一个测试用例验证含归约视图（如 ColReduceSum）的节点在图融合中的边界行为。

**P2-3 容差不一致：绝对容差 vs 相对容差混用**

证据：
- `expr_dsl_test.cpp:98`：绝对容差 `1e-5f`
- `expr_matmul_test.cpp:71-72`：相对容差 `1e-4f * scale`（`scale = max(1.0, |ref|)`）
- `expr_reduce_test.cpp:69-70`：相对容差 `1e-4f * scale`
- `ce_fusion_test.cpp:38`：绝对容差 `1e-4f`
- `tensor_expr_test.cpp:104`：绝对容差 `1e-4f`

相对容差在大值时更宽松（避免 flaky），绝对容差在小值时更严格。混用可能导致某些测试过紧（flaky）或过松（漏 bug）。

建议：统一为 `d > tol * max(1.0f, |ref|)` 的相对容差模式，或至少在每个测试文件头部注释容差选择理由。

**P2-4 `expr_opt_test.cpp:370-441` — 压力测试未验证数值等价性**

证据：test_pressure 仅检查 canonicalize 后的 spec 通过 validate 且不超过上限（L432-439），未验证 canonical spec 与原始 spec 的数值求值结果一致。其他测试（test_semantic_equivalence）验证了特定表达式的数值等价，但压力测试的复杂表达式（8 输入 + 重复子表达式 + 链式求和）未做此验证。

建议：在 test_pressure 末尾增加 `eng.eval_expr` 对比原始 spec 与 canonical spec 的数值结果。

### P3

**P3-1 `expr_graph_test.cpp:247-248` — 独立分支测试未验证数值正确性**

证据：test_independent (L219-249) 检查两个独立节点各成一个 kernel（L235-238），并验证两个 kernel 均可求值（L247），但未验证求值结果的数值正确性（缺少与逐节点参考的比对）。

建议：增加两个 kernel 求值结果与独立参考的数值比对。

**P3-2 `expr_fuse_test.cpp:219-249` — vec4 路径仅测试 4 的倍数列宽**

证据：run_vec4_path 测试 cols=4,8,12（均为 4 的倍数）。注释 (L213-214) 声称非 4 倍数由 run_non_recording (C=6) 覆盖，但 run_non_recording 使用的是不同表达式（row_broadcast 路径），vec4 标量回退路径本身未被直接测试。

建议：在 run_vec4_path 中增加 cols=5 或 cols=7 的用例，直接验证 vec4 标量回退路径。

## 覆盖矩阵

| 测试文件 | batch>1 | 非4对齐维度 | 独立期望值 | NaN/Inf传播 | BatchCol/Mod | Row+matmul语义 | 失败→退出码 |
|----------|---------|------------|-----------|------------|-------------|---------------|------------|
| expr_dsl_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L311) |
| expr_fuse_test | ✅ (B=6) | ✅ (C=6) | ✅ (CPU参考) | N/A | ❌ | ❌ | ✅ (L276) |
| expr_graph_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L640) |
| expr_matmul_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L351) |
| expr_opt_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L523) |
| expr_reduce_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L368) |
| ce_fusion_test | ✅ (N=12) | N/A | ✅ | N/A | ❌ | ❌ (仅纯逐元素) | ✅ (L361) |
| tensor_expr_test | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ✅ (L186) |
| nan_repro_test | ✅ (batch=8) | N/A | ✅ (回归) | ❌ (仅最终输出) | ❌ | N/A | ✅ (L155) |

## 疑似被测问题

**待验证 1：BatchMod/BatchCol 无测试覆盖 + 已知 IR 缺陷**
g10 已报告 "BatchMod/BatchCol 引用未声明 batch"（ee49f5d 引入）。由于无测试覆盖，此缺陷在测试层面完全不可见。若修复被测代码后无对应测试，回归风险高。

**待验证 2：Row 操作数语义在不同上下文的差异**
g10 报告 "Row 操作数在纯归约/matmul+行归约下语义不同（row 全局行号 vs batch 内行号）"。当前测试仅覆盖纯逐元素上下文（ce_fusion_test），matmul+行归约上下文未覆盖。建议在修复被测代码时同步增加测试。
