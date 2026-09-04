# g09 表达式 DSL + AOT 工具代码审查

## 模块概览

本组 5 个文件构成 AOT 融合管线的 DSL 前端与构建期工具链：

| 文件 | 行数 | 职责 |
|------|------|------|
| `expr_dsl.hpp` | 713 | DSL 算子集（叶子/节点模板/运算符重载）、to_expr_spec 折叠、compute 统一入口、start_expr/end_expr 块式融合 |
| `expr_emitter.hpp` | 102 | 后端 emitter 抽象接口（IR-D）+ 注册表 |
| `expr_spec.hpp` | 559 | 扁平 IR 数据结构（ExprSpec/ExprInstr/ExprView/MatmulSpec）、FNV-1a key、validate、常量/视图/操作数辅助 |
| `scan_exprs.cpp` | 390 | 构建期 dry-run 收集 ExprSpec → bin（覆盖 RoPE/ReLU/SwiGLU/GeLU/Softmax/RMSNorm/LayerNorm/FusedChain/RLA/GPTBlock/TransformerEncoder/ZiPT/MSE/CE/Adam/Linear/matmul 融合/CSA 注意力/CE 稀疏） |
| `gen_fused.cpp` | 276 | 读 bin → emitter 抽象生成 GLSL → glslc → SPIR-V → 内联 fused_registry.hpp |

**关键设计**：单一事实来源 = Layer 内联表达式（编译期类型）。DSL 算子复用 `nn::ops`（`op_id()` 统一映射到 `ExprOp`）；叶子持有 Tensor（零拷贝 shared_ptr）；视图是索引映射不物化中间张量。Binary::to_spec 显式固定 `l.to_spec(b)` / `r.to_spec(b)` 求值顺序（C++ 函数实参求值顺序未指定），保证跨编译器确定。

## 发现

### P0 正确性 bug / 崩溃 / UB / 语义错误

无。

### P1 生命周期 / 内存 / 并发 / 错误传播断裂 / key 冲突

**P1-1 `tools/gen_fused.cpp:104-108` — emit_spec 的 MatmulSpec 聚合初始化遗漏 `batch` 字段**

```cpp
o << "MatmulSpec{" << static_cast<int>(spec.matmul->a_input) << ", "
  << static_cast<int>(spec.matmul->b_input) << ", "
  << static_cast<int>(spec.matmul->transA) << ", "
  << static_cast<int>(spec.matmul->transB) << ", "
  << spec.matmul->k << "u}";
// ← 缺少 batch 字段
```

`MatmulSpec` 定义（`expr_spec.hpp:220-230`）：`a_input, b_input, transA, transB, k, batch`（6 个字段）。`emit_spec` 只输出前 5 个（缺 `batch`），生成的 `fused_registry.hpp` 中 `ExprSpec` 聚合初始化的 `matmul` 字段会用 `batch` 的默认值（=1）。

**与 g10 P1-23 同根**：`expr_registry.hpp:107-112` 的 `write_registry` 也未序列化 `batch`（只写 k）。两端一致地丢失 batch 信息。

**影响**：若 `MatmulSpec.batch > 1` 的 spec 进入管线，运行时 dispatch 的 z 维 = 1（默认）而非实际 batch → 跨 batch 串扰（铁律 5：batch-major 布局）。当前 scan_exprs 所有手动构造的 matmul spec 均 `batch=1`，暂不触发。

**修复建议**：`emit_spec` 加 `<< ", " << spec.matmul->batch << "u"`；`write_registry` 加 `write_pod(f, s.matmul->batch)`；`read_registry` 加 `read_pod(f, mm.batch)`；版本 bump 到 v3。

---

**P1-2 `include/neuralnet.cpp/expr_emitter.hpp:16` — 注释引用已删除的 `cpu_emitter.hpp`**

```cpp
//    - 具体 emitter 类在各自头文件（glsl_gen.hpp / cpu_emitter.hpp）提供，
```

`cpu_emitter.hpp` 已在 docs/17 §1.3 删除（2026-08-24）。此处注释残留，会误导维护者尝试包含不存在的头文件。

**修复建议**：改为 `（glsl_gen.hpp 等）` 或直接删除该子句。

### P2 中等风险 / 设计缺陷

**P2-1 `include/neuralnet.cpp/expr_spec.hpp:431-523` — validate_expr_spec 不校验 Reduce 操作数引用的寄存器是否确为归约指令定义**

```cpp
if (opnd.kind == static_cast<uint8_t>(ExprOperandKind::Reduce))
{
    if (opnd.idx >= spec.num_regs)
        return std::unexpected(Error{"validate_expr_spec: reduce ref out of range"});
    if (opnd.idx == ins.dst)
        return std::unexpected(Error{"validate_expr_spec: reduce self-reference"});
    if (!reduce_dst[opnd.idx] || elem_dst[opnd.idx])
        return std::unexpected(Error{...});
}
```

校验 `!reduce_dst[opnd.idx] || elem_dst[opnd.idx]` 拒绝"引用非归约 dst"和"引用已被元素指令覆盖的 dst"。但未检查 `elem_dst[opnd.idx]` 为 0 的情况——若 `opnd.idx` 指向一个**尚未被任何指令定义**的寄存器（`reduce_dst=0, elem_dst=0`），条件 `!reduce_dst[opnd.idx]` 为 true → 正确拒绝。实际安全。

但存在另一盲区：若一条元素指令的 dst 与某归约指令的 dst 相同（`elem_dst[dst]=1, reduce_dst[dst]=1`），Reduce 操作数引用该 dst 时条件 `!reduce_dst || elem_dst` = `0 || 1` = true → **不拒绝**，但该寄存器已被元素指令覆盖（不再是归约结果）。在 canonical IR 中每个寄存器只被一条指令定义（寄存器分配保证），故不会发生。但 validate 接受原始 IR（canonicalize 前），理论上可达。

**修复建议**：在归约校验循环结束后，增加断言 `for (dst : num_regs) assert(!(reduce_dst[dst] && elem_dst[dst]))` 或在 DSL 层保证原始 IR 的 dst 唯一性。

---

**P2-2 `tools/gen_fused.cpp:259-263` — `find_fused` 线性搜索 O(n)，注册表增长后性能退化**

```cpp
[[nodiscard]] inline const FusedShader* find_fused(const std::string& key)
{
    for (const auto& f : kFusedShaders)
        if (key == f.key) return &f;
    return nullptr;
}
```

当前 kFusedShaderCount ≈ 30-50（覆盖所有 Layer 的 fwd+bwd）。若未来表达式种类增长到数百，线性搜索 + `std::string` 比较会成为热路径瓶颈（每 step 每次 eval_expr 调用一次）。

**修复建议**：改用 `constexpr` 编译期排序 + 二分查找，或 `std::unordered_map`（但 constexpr 限制）。优先级低，当前规模可接受。

---

**P2-3 `include/neuralnet.cpp/expr_spec.hpp:431-453` — validate_expr_spec 不校验 `num_regs` 与 `instrs` 的一致性**

```cpp
if (spec.num_regs > EXPR_MAX_REGS)
    return std::unexpected(Error{"validate_expr_spec: too many registers"});
// 未检查 num_regs == 0 && !spec.matmul 的情况（已由 :435 覆盖）
// 未检查 spec.instrs.back().dst == spec.num_regs - 1（隐含不变量）
```

`num_regs` 是"寄存器总数"（= 最后一条指令 dst + 1），由寄存器分配保证。validate 未显式校验 `instrs.back().dst == num_regs - 1`（当 instrs 非空时）。若手动构造的 spec 违反此不变量，glsl_gen 的 `last_dst`（`:276`）可能引用未声明的寄存器。

在实践中，DSL to_spec 保证 `dst` 连续递增（`num_regs++`），canonicalize 重编号后也保持连续。手动构造的 scan_exprs spec（`:256-282`）也遵守。但 validate 作为防御层应覆盖。

**修复建议**：在 validate 的 instrs 循环后加 `if (!spec.instrs.empty() && spec.instrs.back().dst != spec.num_regs - 1) return error`。

---

**P2-4 `include/neuralnet.cpp/expr_spec.hpp:448-453` — validate_expr_spec 不校验 MatmulSpec 的 `a_input != b_input`**

```cpp
if (spec.matmul->a_input >= num_inputs || spec.matmul->b_input >= num_inputs)
    return std::unexpected(Error{"validate_expr_spec: matmul input out of range"});
```

允许 `a_input == b_input`（A*A 矩阵乘）。数学上合法（如协方差矩阵），但 GPU 融合 shader 假设 A/B 是不同输入（两个不同的 Linear 视图），若 A/B 指向同一 Tensor 且 transA=0，shader 内 A/B 的读取偏移可能冲突。

**修复建议**：可选加 `if (spec.matmul->a_input == spec.matmul->b_input) return error`（保守拒绝），或在注释中明确"A/B 允许指向同一 Tensor（GPU 融合 shader 已支持）"。

### P3 风格 / 优化建议

**P3-1 `include/neuralnet.cpp/expr_spec.hpp:357` / `include/neuralnet.cpp/expr_graph.hpp:377` — 两套 FNV-1a 实现常量格式不一致**

`expr_spec.hpp:357` 用十六进制 `0xcbf29ce484222325ull` / `0x100000001b3ull`；`expr_graph.hpp`（g10 P3-3 已报）用十进制。值相同，风格不一致。建议统一用十六进制。

**P3-2 `tools/gen_fused.cpp:50` — 使用 `std::system` 执行 glslc，路径含特殊字符时有 shell 注入风险**

```cpp
return std::system(cmd.c_str()) == 0;
```

构建期工具，路径由 CMake 控制（无用户输入），实际安全。但 `std::system` 在 Windows 上走 `cmd.exe`，路径分隔符需注意。建议未来迁移到 `std::process`（C++26）或 `CreateProcess`（Windows）。

**P3-3 `include/neuralnet.cpp/expr_dsl.hpp:366-369` — Binary::to_spec 的求值顺序修复方案正确但有冗余拷贝**

```cpp
const ExprOperand lo = l.to_spec(b);
const ExprOperand ro = r.to_spec(b);
```

`ExprOperand` 是 2 字节 POD（`uint8_t kind + idx`），拷贝开销可忽略。设计正确（固定跨编译器顺序）。纯观察，无修改建议。

## S7 教训核对

| # | 教训 | 状态 | 证据 |
|---|------|------|------|
| 1 | 运行时值禁进表达式常量池 | ✅ 遵守 | `compute_layer_attention.hpp:459` scale 通过 `engine.scale_inplace(Q, scale_)` 独立执行，不进 DSL 常量池；`kNegInf_`（`:217`）是结构常量（`-inf`），非运行时值 |
| 2 | BatchCol 视图要求 `(1, BH*seq)` | ✅ 遵守 | `compute_layer_attention.hpp:208` `mask_doc_ids_` 声明为 `(1, batch*seq)`；`:242` `dsl::batch_col(*mask_doc_ids_(), seq)` 的 `per_batch_cols=seq`，视图读 `data[batch*seq+col]`，不越界 |
| 3 | RowGather 主输入行数≠网格行数 | ✅ 遵守 | `compute_loss.hpp:345-346` `dsl::row_gather(logits, *labels_t)` 中 logits=(C,total)、labels_t=(1,total)；`expr_spec.hpp:456-461` validate 只查 cols（`views[k].param >= num_inputs`），不查行数 |
| 4 | gen_fused emit_spec 的 ±inf 用 numeric_limits | ✅ 遵守 | `gen_fused.cpp:95-97`：`std::isinf(v)` → `std::numeric_limits<Scalar>::infinity()` |
| 5 | matmul + 列归约不支持（gen_fused 跳过） | ✅ 遵守 | `gen_fused.cpp:185-191`：`raxis == 1 && spec.matmul` → skip + stderr |

## 已知项核对

**g10 已报 P1-23（registry batch 序列化丢失）**：已确认同根问题存在于 `gen_fused.cpp:104-108` 的 `emit_spec`（本报告 P1-1）。两端一致丢失 batch 信息。

**AGENTS.md §12 CpuEmitter 隐性缺陷（3 条）**：已确认全部 N/A（CpuEmitter 已删）。
- ① BatchMod/BatchCol 引用未声明 batch：N/A（文件已删）
- ② Row/Col/Batch/Matmul/Reduce 操作数与 RowGather 视图走 default 错语义：N/A
- ③ 纯 matmul spec `instrs.back()` UB：N/A

`expr_emitter.hpp:16` 的 `cpu_emitter.hpp` 注释残留（本报告 P1-2）是唯一痕迹。

**g10 P0-1（matmul 分块注释与实现不符）**：不在本组文件范围。

**g10 P0-2（matmul rows/mm_batch 无守卫）**：不在本组文件范围，但本报告 P2-3 提及 validate 未校验 num_regs 一致性（同根防御缺失）。

**g10 P2-1（glsl_view_read default 分支）**：不在本组文件范围。

## 待验证

1. **`expr_dsl.hpp:516` `relu` 宏实现**：`max(e, ConstLeaf{Scalar{0}})` 用 `ConstLeaf` 而非 `expr::cst(0)`。`ConstLeaf` 的 `to_spec` 调用 `b.add_const(value)`，每次调用都往常量池追加 `0.0f`。若多个 relu 共用一个表达式树，常量池会有重复的 `0.0f`。canonicalize 的常量折叠（IR-A）是否去重？`expr_opt.hpp` 的 `add_const_op` 线性查找去重（g10 P3-2 已报 `O(n²)`），应能合并。但需确认：若两个 `ConstLeaf{0}` 在折叠后变成同一个常量索引，Binary<Max, E, ConstLeaf> 的两个实例是否共享同一 c=0？**待验证** canonicalize 是否消除冗余常量。

2. **`scan_exprs.cpp` 未扫描 `CrossEntropyLoss::backward`**：CE backward（`compute_loss.hpp:245-248`）只返回 `grad_input_`，不产生新表达式，故不需扫描。但若未来 CE backward 改为 DSL 实现，需补扫描。**待验证**当前 backward 路径是否有任何 dsl::compute/end_expr 调用。
