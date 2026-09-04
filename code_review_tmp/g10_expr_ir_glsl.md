# 表达式 IR + GLSL 生成（L2 构建期）代码审查

## 模块概览

本组 4 个文件构成 AOT 融合管线的 L2 构建期核心：`expr_glsl_gen.hpp`（1317 行）把扁平 `ExprSpec` 展开为 GLSL 计算 shader（逐元素标量/vec4、matmul 分块、归约三类 kernel，共三套代码生成路径）；`expr_opt.hpp`（605 行）实现 IR-A/IR-B 优化链（fold+DCE+renumber+CSE+liveness 寄存器分配）；`expr_graph.hpp`（480 行）实现图 IR 与贪心融合分析（IR-C）+ 图级缓存；`expr_registry.hpp`（180 行）提供 AOT 收集注册表与 bin 序列化。已核 `expr_spec.hpp`（IR 定义 + FNV-1a key + validate）、`docs/11-ir-optimization.md`（设计意图）、`tools/gen_fused.cpp`（matmul 跳过 + 混合轴跳过）、`backend/compute_vk_backend.hpp`（push constant 布局 + dispatch）。

## 发现

### P0 正确性 bug / 崩溃 / UB / 生成 GLSL 语义错误

**[P0] `expr_glsl_gen.hpp:223,398` — matmul 分块尺寸注释与实现严重不符（2×2 vs 4×4），dispatch 维度按 32 分块但代码按 16 分块**

```cpp
constexpr std::uint32_t B = EXPR_MATMUL_BLOCK;   // 输出块（32 = 2*T）   ← 注释说 32
...
L << "    float acc[4][4];\n";  // 4×4 寄存器累加器（本线程负责 16 个输出元素）
L << "    const uint rr = block_row + ty * 4u + i;\n";  // 每线程 4 行
L << "    const uint cc = block_col + tx * 4u + j;\n";  // 每线程 4 列
```

`EXPR_MATMUL_BLOCK = 64`（`expr_spec.hpp:427`，注释"64×64"），但生成器头注释（`:196-198`）说"32×32"、`constexpr B = 32` 的注释（`:223`）、`#pragma unroll` 的 4×4 累加器（`:398`）都是 32 时代的设计。`vk backend` 的 dispatch（`compute_vk_backend.hpp:2259-2263`）按 `EXPR_MATMUL_BLOCK=64` 算 wg 数（`ceil(cols/64)×ceil(rows/64)`），但生成器内部用 16×16 线程 + 4×4 寄存器分块 → **每工作组只算 16×4=64 行 × 16×4=64 列**，恰好 64×64，与 dispatch 匹配。但 `:223` 的 `B = EXPR_MATMUL_BLOCK` 实际是 64，而 `:255` 的共享内存 `AshT[BK4][B]`（B=64）与 `:398` 的 `acc[4][4]`（每线程 4 行 4 列 × 256 线程 = 1024 元素，但 16×16 线程 × 4×4 = 64×64）——**实际每工作组计算 64×64 = 4096 元素，dispatch 按 64 分块 → 匹配**。注释"32 = 2*T"是历史遗留（T=16，2*T=32 是旧 BLOCK=32 时代的值），当前 BLOCK=64 但 T 仍 16，2*T=32≠64。代码本身自洽（4×4 累加器 × 16×16 线程 = 64×64 输出块，与 dispatch 一致），但注释会误导维护者改尺寸时出错。标"疑似"——若未来有人按注释把 `acc` 改回 2×2，dispatch 与生成器失配 → 越界/数据错乱。建议：把 `:223` 注释改为 `// 输出块（64 = 4*T，4×4 寄存器分块）`，`:196-198` 头注释同步。

**[P0] `expr_glsl_gen.hpp:310,394` — matmul+逐元素 的 `grow`（全局行）在 `eval_tail` 内重复计算，与 `main` 中 `m_per` 定义重复，但 `row` 参数是 batch 内行号——若 `rows % mm_batch != 0` 则 `m_per` 截断导致跨 batch 串扰**

```cpp
// eval_tail 内（:309-310）
L << "    const uint m_per = rows / mm_batch;\n";
L << "    const uint grow = batch * m_per + row;\n";
// main 内（:394）
L << "    const uint m_per = rows / mm_batch;    // 每批输出行数\n";
```

`rows / mm_batch` 是整数除法。`expr_spec_rows_per_batch`（`expr_spec.hpp:268-273`）有守卫 `(b > 0 && rows % b == 0) ? rows / b : rows`，但生成器**直接写 `rows / mm_batch` 无守卫**——若 `rows` 不被 `mm_batch` 整除（如 batch=3, rows=100），`m_per = 33`，`grow = 3*33 + 1 = 100`（越界，实际应 99）。`validate_expr_spec` 不校验 `rows % batch == 0`（`:448-453` 只查 `batch > 0` 与 a/b 输入范围）。**建议**：在 `generate_glsl_matmul` 开头加断言或注释"调用方保证 rows % mm_batch == 0"，或在 `validate_expr_spec` 加 `rows % batch == 0` 检查（但 validate 不知 rows，需在 dispatch 侧加）。

**[P0] `expr_glsl_gen.hpp:936` — 归约 kernel 的 `Row` 操作数语义不一致：matmul+行归约时 `row % m_per`（batch 内行号），但纯归约时 `float(row)`（全局行）**

```cpp
case static_cast<uint8_t>(ExprOperandKind::Row):
    return mm ? "float(row % m_per)" : "float(row)";
```

纯归约（无 matmul）时 `row` 是全局行号（`idx` 是行下标，`:901` `const uint idx = gl_WorkGroupID.x`，`:1077` `const uint row = idx`）。matmul+行归约时 `row % m_per` 是 batch 内行号。语义不一致：同一 `Row` 操作数在不同 spec 下含义不同。Causal 掩码 `col > row` 在纯归约时用全局行（如 100 行网格中第 99 行 `row=99`），在 matmul+归约时用 batch 内行（如 batch=4, m_per=25，第 99 行 `row%25=24`）——**causal 掩码语义错误**。标"疑似"——若 Layer 只用 matmul+行归约（注意力）且 causal 掩码在 matmul 段内生成（`Row`/`Col` 操作数），则一致；但跨段混用会出错。建议：统一为 batch 内行号（与 matmul 段一致），或在注释中明确"Row 操作数 = batch 内行号（matmul 段）/ 全局行号（纯归约）"并在 Layer 侧按场景选择。

### P1 生命周期 / 内存 / 并发 / 错误传播断裂 / key 冲突

**[P1] `expr_opt.hpp:519` — 寄存器分配的输出寄存器保护用 `last_use[n-1]` 但 `n` 是**指令数**而非寄存器数，若 `num_regs > instrs.size()` 则越界**

```cpp
const std::size_t n = spec.instrs.size();
...
last_use[n - 1] = static_cast<int>(n - 1);  // 输出寄存器保护
```

`last_use` 大小为 `n`（指令数）。`n-1` 是最后一条指令下标。若 `num_regs > n`（不可能，但 `num_regs` 是 `uint32_t`，`instrs.size()` 是 `size_t`），`last_use` 仍按 `n` 分配。实际 `num_regs ≤ instrs.size()`（每指令一 dst），所以 `n-1` 合法。但注释"输出寄存器（最后一条指令 dst）的 last_use 设为 n-1 防复用"——**若最后一条指令的 dst 不是寄存器 `n-1`（如 CSE 后寄存器重编号），`last_use[n-1]` 保护的是错误的寄存器**。实际 CSE 后 `reg_of[n-1]` 是输出寄存器，`last_use[def_at[reg_of[n-1]]]` 才是正确的 last_use 位置。但代码用 `last_use[n-1]`（指令下标 n-1 的 last_use），而 `def_at[spec.instrs[n-1].dst]` 才是定义该寄存器的指令下标。**若 `spec.instrs[n-1].dst != n-1`（CSE 后可能），保护错位**。标"疑似"——CSE 后 `reg_of[i]` 是连续分配（0,1,2,...），`spec.instrs[i].dst = i`（CSE 输出 dst 就是分配顺序），所以 `n-1` 的 dst 就是 `n-1`。实际安全，但依赖 CSE 的隐含不变量（dst 连续）。建议：显式注释"CSE 后 dst 连续，最后一条指令 dst = n-1"，或用 `def_at[spec.instrs[n-1].dst]` 更稳健。

**[P1] `expr_graph.hpp:226-227` — 图融合溢出检查用 `>= 256` 但 `ExprInstr.dst` 是 `uint8_t`（0-255），`255` 是合法值**

```cpp
for (const auto& ins : Bs.instrs)
    if (static_cast<std::size_t>(ins.dst) + A.num_regs >= 256)
        return false;
```

`dst + reg_base` 若 = 255，`>= 256` 为 false → 允许。`ExprInstr.dst` 是 `uint8_t`，255 是最大值，合法。但 `validate_expr_spec` 的 `EXPR_MAX_REGS = 16`（`expr_spec.hpp:415`），实际 dst 不会超过 16。`>= 256` 的溢出检查是防御 `uint8_t` 回绕（255+1=0），正确。但注释"必须 < 256"与代码 `>= 256` 一致。无 bug。

**[P1] `expr_registry.hpp:73-116` — `write_registry` 的 `matmul->k` 和 `matmul->batch` 序列化，但 `read_registry` 只读 `k` 不读 `batch`**

```cpp
// write（:112）
if (!write_pod(f, s.matmul->k)) return false;
// read（:171）
if (!read_pod(f, mm.k)) return false;
// mm.batch 未读（默认 1）
```

`MatmulSpec` 有 `k`（u32）和 `batch`（u32）两个形状参数。`write_registry` 只写 `k`（1 u32），`read_registry` 只读 `k`。`batch` 默认 1（`expr_spec.hpp:227`）。**若 spec 的 `matmul.batch > 1`，bin 序列化丢失 batch 信息，读回后 batch=1 → 运行时 dispatch z=1 而非实际 batch → 跨 batch 串扰（铁律 3：batch-major 布局）**。标"疑似"——若 scan 阶段 batch 总是 1（注意力 batch 由 Layer 侧 dispatch 决定，不进 spec），则无 bug。但 `MatmulSpec.batch` 是结构字段（`expr_spec.hpp:227`），理论上可 > 1。**建议**：`write_registry` 加 `write_pod(f, s.matmul->batch)`，`read_registry` 加 `read_pod(f, mm.batch)`，版本 bump 到 v3。

### P2 中等风险 / 设计缺陷（≤10 条）

1. **[P2] `expr_glsl_gen.hpp:76,78` — `glsl_view_read` 的 `default` 与 `Linear` 共用，未知 view kind 静默当 Linear 处理**

```cpp
default:
case static_cast<uint8_t>(ExprViewKind::Linear):
    os << buf << "[" << idx_var << "]";
```

若 `ExprViewKind` 新增枚举值，`glsl_view_read` 未更新 → 静默当 Linear（越界读）。`glsl_vec4_eligible`（`:166-189`）会拒绝非 Linear/RowBroadcast/ColBroadcast 的 view（返回 false → 标量路径），但标量路径的 `glsl_view_read` 仍 `default` 当 Linear。**建议**：`default` 分支 `abort()` 或返回错误，不静默。

2. **[P2] `expr_opt.hpp:422` — CSE 的 `unordered_map<uint64_t, uint8_t>` 用 64 位 key，但 key 编码（op 8bit + a.kind 8bit + a.idx 8bit + ...）只有 56 位，16 位未用**

```cpp
std::uint64_t key = static_cast<std::uint64_t>(in.op);
key |= static_cast<std::uint64_t>(a.kind) << 8;
key |= static_cast<std::uint64_t>(a.idx)  << 16;
// ... 到 c.idx << 48
```

key 最高位（bit 56-63）恒 0。`unordered_map` 的 64 位 key 中 16 位浪费，hash 分布略差（碰撞概率略高）。无正确性 bug（idx 是 uint8_t，0-255，8 位足够）。**建议**：用 FNV-1a 或把未用 16 位填 0（已填 0，无问题）。纯优化。

3. **[P2] `expr_graph.hpp:467-474` — `graph_plan_cache` 的 `thread_local unordered_map` 无容量上限时的清理策略**

```cpp
inline constexpr std::size_t GRAPH_PLAN_CACHE_MAX = 512;
```

常量定义了但代码中**未使用**（无 `if (cache.size() > MAX) cache.clear()`）。若 Layer 结构种类 > 512，缓存无限增长（训练每 step 复用，不会无限增长，但防御性上限未实现）。**建议**：在 `instantiate_plan` 或 `fuse_expr_graph` 入口加 `if (cache.size() > GRAPH_PLAN_CACHE_MAX) cache.clear()`。

4. **[P2] `expr_glsl_gen.hpp:276-277` — `generate_glsl_matmul` 的 `last_dst` 在 `instrs.empty()` 时用 0，但 `eval_tail` 空链时直接 `return mm`（`:384`），`r0` 未声明（`:324-328` 只在 `num_regs > 0` 时声明）→ 若 `instrs` 非空但 `num_regs == 0`（不可能，但防御性），`r0` 未声明**

```cpp
const std::uint32_t last_dst = spec.instrs.empty() ? 0u : spec.instrs.back().dst;
...
if (!spec.instrs.empty()) {
    ...
    if (spec.num_regs > 0) { L << "    float r0"; ... }
    ...
    L << "    return r" << last_dst << ";\n";
} else { L << "    return mm;\n"; }
```

`instrs` 非空时 `num_regs` 必然 > 0（每指令一 dst），所以 `r0` 一定声明。实际安全，但无断言。

5. **[P2] `expr_glsl_gen.hpp:1059` — 归约 kernel 的 `gl_NumSubgroups` 在 subgroup 0 内用作边界，但 `gl_NumSubgroups` 是 subgroup 级变量，在 subgroup 0 内读值正确**

```cpp
L << "        float " << wn << " = (gl_SubgroupInvocationID < gl_NumSubgroups) ? "
  << s << "[gl_SubgroupInvocationID] : " << idt << ";\n";
```

`gl_NumSubgroups` 在 subgroup 0 内是"总 subgroup 数"（正确），`gl_SubgroupInvocationID` 在 subgroup 0 内是"该 subgroup 在组内的下标"（0..subgroup_size-1）。若 subgroup_size > gl_NumSubgroups（不可能，subgroup 0 的 size ≤ 总 subgroup 数），越界。实际安全（subgroup 0 的 size = min(32, 256) = 32，总 subgroup = 256/32 = 8，32 > 8 → **越界读 `s[32..31]`**）。**标"疑似"——若 subgroup size 32 > gl_NumSubgroups 8，`gl_SubgroupInvocationID` 最大 31，`s[31]` 越界（`s_red` 大小 256，`s[31]` 合法，但语义错误：读的是其他 subgroup 的部分和，未初始化）**。实际 `s_red` 大小 256，`s[0..255]` 都合法，但 `s[8..31]` 未写入（只有 subgroup 0-7 写入 `s[0..7]`）。**建议**：用 `gl_SubgroupInvocationID < 8u`（硬编码 subgroup 数 = 256/32）或 `gl_SubgroupInvocationID < gl_NumSubgroups`（正确，但 `gl_NumSubgroups` 在 subgroup 0 内的值需确认）。

6. **[P2] `expr_opt.hpp:547-548` — 寄存器分配的 `next_elem` 用 `release_pos.size()`（逐元素区段寄存器数），但 `release_pos` 只含逐元素寄存器（归约第二遍独立分配），正确**

```cpp
const std::uint32_t next_elem = static_cast<std::uint32_t>(release_pos.size());
std::uint32_t next_red = next_elem;
```

归约寄存器从 `next_elem` 开始独立分配，与逐元素区段 `[0, next_elem)` 不重叠。正确。

7. **[P2] `expr_graph.hpp:172-317` — `try_append` 的 `tail_slots` 只检查 B 对 tail 的引用是 Linear，但未检查 A 的输出形状与 B 的输入形状匹配（tail 的 rows/cols vs B 的 rows/cols 已检查 `:189`，但 tail 的输出是 (rows, cols) 网格，B 的输入也是 (rows, cols)，正确）**

无 bug，但注释未明确"tail 输出形状 = B 输入形状"的不变量。

8. **[P2] `expr_registry.hpp:44-47` — `ExprRegistry::add` 用 `canonicalize_expr_spec` 后再算 key，但 `contains` 也 canonicalize（`:49-50`），两端一致。但 `specs` 向量存的是 canonical spec，`keys` 存的是 canonical key——若两次 `add` 同一 raw spec（不同 canonical 形态但同 key），第二次 `keys.insert` 返回 false → 不重复存。正确。**

### P3 风格 / 优化建议（≤10 条）

1. **[P3] `expr_glsl_gen.hpp:29` — `glsl_binary_op` 的 `is_compare` 出参在 `Add/Sub/Mul/Div` 分支未显式设 false（`is_compare = false` 在函数开头，正确）。**

2. **[P3] `expr_opt.hpp:87-96` — `add_const_op` 的线性查找 `O(n²)` 常量池去重（n ≤ 16，无性能问题）。**

3. **[P3] `expr_graph.hpp:377-378` — `graph_cache_key` 的 FNV-1a 常量 `kFnvOffset`/`kFnvPrime` 与 `expr_spec_key`（`expr_spec.hpp:357`）的 `0xcbf29ce484222325`/`0x100000001b3` 不同——两套 FNV-1a 实现，常量值一致（`14695981039346656037` = `0xcbf29ce484222325`，`1099511628211` = `0x100000001b3`），但一个用十进制字面量一个用十六进制，风格不一致。建议统一用十六进制。**

4. **[P3] `expr_glsl_gen.hpp:1310-1314` — `kGlslEmitterRegistered` 的静态初始化在 header 中，多次包含（`#pragma once` 保证一次）无害。但 `inline const bool` 的初始化顺序跨 TU 未定义（C++ 静态初始化顺序坑）。实际无副作用（只注册，不读返回值）。**

5. **[P3] `expr_opt.hpp:184` — `def_op` 默认值 `ExprOp::Add` 是"最危险"的默认值（若某寄存器未被定义，`def_op` 是 Add，`neg(neg(x))` 化简可能误触发）。建议用 `ExprOp::Neg` 或哨兵值（如 `static_cast<ExprOp>(255)`）。**

6. **[P3] `expr_glsl_gen.hpp:228-229` — 生成 GLSL 的头部注释 `// 表达式: " << name` 含用户提供的 `name`（如 `fused_<key>`），若 `name` 含 `//` 会破坏注释。实际 `name` 是 `fused_` + 16 位 hex key，无 `//`。防御性。**

7. **[P3] `expr_registry.hpp:78` — `f.write("NNEXP", 5)` 的 magic 是 5 字节，但 `read_registry` 的 `magic[5]` 也是 5 字节，一致。但 magic 不含长度前缀（固定 5），若未来扩展需版本化。**

8. **[P3] `expr_graph.hpp:98` — `add_node` 的 `inputs[k].virtual_tag()` 假设 `k < inputs.size()`（`:86-87` 有守卫），但 `input_tensors` 用 `inputs[k]`（`:98`），若 `k >= inputs.size()` 用 `Tensor{}`（默认），正确。但 `dep_of_input` 用 `inputs[k].virtual_tag()`（`:87`），若 `k >= inputs.size()` 用 `tag = 0`（`:86-87`），正确。防御性。**

## 已知问题核对

AGENTS.md §12 标注的 CpuEmitter 隐性缺陷（3 条）：**已全部不存在（N/A）**。

证据：
1. `include/neuralnet.cpp/` 下无 `expr_cpu_emitter.hpp` 文件（`fast_locate` 扫描 153 目录 0 匹配，`grep include/**CpuEmitter*` 0 匹配）。
2. docs/17-pointer-audit.md:27（1.3 行）明确"删 `expr_cpu_emitter.hpp`（文件+注册行）"，docs/13:126/652 标注 P2-04 已实施（2026-08-24）。
3. `gen_fused.cpp:200` 硬编码 `emitter_registry::make("glsl")`（注释 `:9-10` "IR-D 落地：本工具经 nn::emitter_registry 选择后端（默认 'glsl'）"），CpuEmitter 产物从不参与编译。

因此：
- ① BatchMod/BatchCol 引用未声明 batch：N/A（CpuEmitter 已删）
- ② Row/Col/Batch/Matmul/Reduce 操作数与 RowGather 视图走 default 错语义：N/A
- ③ 纯 matmul spec `instrs.back()` UB：N/A

**注意**：`expr_glsl_gen.hpp:76-78` 的 `glsl_view_read` 仍有 `default` 分支当 Linear 处理（见 P2-1），但 GlslEmitter 产物参与编译（`gen_fused.cpp:208` 调用 `emitter->generate`），若未来新增 view kind 未同步更新 `glsl_view_read` → 静默越界。建议 `default` 分支加 `static_assert` 或 `abort()`。

## 其他观察

1. **EXPR_MATMUL_BLOCK 尺寸演进**：`expr_spec.hpp:427` 定义 `EXPR_MATMUL_BLOCK = 64`，注释"64×64"。`expr_glsl_gen.hpp:196-198` 头注释"32×32"（历史遗留），`:223` `constexpr B = 64` 注释"32 = 2*T"（历史遗留，T=16, 2*T=32 是旧 BLOCK=32 时代的值）。`vk backend:2252-2253` 注释"每工作组 32×32 输出块、16×16 线程、每线程 2×2 寄存器分块"（也是旧值）。**三处注释都停留在 BLOCK=32 时代，实际 BLOCK=64 + 4×4 寄存器分块**。建议统一更新注释。

2. **matmul+列归约跳过**：`gen_fused.cpp:185-191` 正确跳过（`raxis == 1 && spec.matmul`），与 `expr_glsl_gen.hpp:909-910` 的 `generate_glsl_reduce` 返回空一致（`if (!is_row) return {}`）。运行时闭合世界硬报错（`find_fused` 未命中 → 后端报错）。一致。

3. **混合归约轴跳过**：`gen_fused.cpp:178-184` 跳过 `raxis == -2`，与 `expr_glsl_gen.hpp:806-807` 的 `if (axis < 0) return {}` 一致（-2 是混合轴，`expr_spec.hpp:314` "−2=混合（不支持单 kernel 融合）"）。一致。

4. **key 计算在 canonical IR 上**：`expr_registry.hpp:44` `add` 先 `canonicalize_expr_spec` 再 `expr_spec_key`，`expr_spec.hpp:355` `expr_spec_key` 用 FNV-1a 64。`docs/11:80` "expr_spec_key(spec) ≡ expr_spec_key(canonicalize_expr_spec(spec))"。两端一致。`graph_cache_key`（`expr_graph.hpp:375`）也先 `expr_spec_key(n.spec)`（n.spec 是 canonical，`:46` "canonical IR（录制时已 canonicalize）"）。一致。

5. **确定性**：`expr_opt.hpp:21-22` "全部 pass 遍历顺序固定（指令序从前到后、寄存器按号、常量按出现序）"。CSE 的 `unordered_map` 迭代顺序未定义，但 CSE 只读 `cse.find(key)`（命中/未命中），不迭代 map，无顺序依赖。寄存器分配的 `release_pos` 是 `vector`（顺序分配），`free_mark` 按号扫描（`:531-533` 从 0 到 size），确定。`graph_cache_key` 的 FNV-1a 按节点序（`:397-406` 从 0 到 nodes.size()），确定。无顺序依赖。
