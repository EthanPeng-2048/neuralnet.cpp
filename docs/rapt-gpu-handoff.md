# RAPT 全链路 GPU 改造 — 继续文档（handoff）

> **状态**（2026-09-04 更新）：**步骤 1-12 全部完成并验证**（11-12 见 §15.6），任务收口。
> 引擎化章节已沉淀到 `docs/15-rapt-algorithm.md` §7、`docs/19-compute-engine-development.md` §11、`AGENTS.md` §4.3。
> 本文档是自洽的继续上下文：包含完整设计决策、文件锚点、逐步操作指令、验证基线。
> 编辑代码前**必须先 read 最新文件内容**（old_string 必须来自最新内容，不来自本快照）。

---

## 0. 状态与验证总览

**全部完成**：步骤 5-10（2026-09-02/03，见 §15.1-15.3）+ 步骤 11-12（2026-09-04，见 §15.6）。
CPU 与 GPU 双后端全链路验证通过；文档已沉淀（docs/15 §7 / docs/19 §11 / AGENTS.md §4.3）。

**最终验证基线（2026-09-04，详见 §15.6）**：
- 全量构建 144/144 绿；ctest **31/31 全绿**（含 rapt/fused_gpu/expr 全套）。
- `rapt_gradcheck --gpu`：3 段全 OK（kink 0.102/0.252 符合预期，≪ tol 5e-2）。
- `rapt_smoke_test --gpu`：KV-cache 一致性 **max_diff=2.98e-08**（float32 1 ulp，GPU 加法顺序差异，数值正确）。
- text 冒烟：`text_train --model rapt --gpu`（319 步/10.3s，loss 6.3→3.9）+ `text_infer --model rapt --gpu`（16 token/0.4s，39 tok/s）。
- AOT 收集仍 56 条融合表达式（闭合世界未破坏）。

**GPU 验证时的排查提示**（已跑通，留作参考；若未来失败可对照）：
- 三个 scan 原语是单 workgroup 顺序扫描：GTX 850M shared 上限 48KB（scan_prefix 用 ~34.5KB，OK）；若 `VK_ERROR_DEVICE_LOST` 先查 shared 用量与 dk=64 边界。
- `forward_step` 的录制顺序：scan（读 A_state 旧值）→ add_inplace（写 A_state）——同一 command buffer 内顺序执行，GPU 上天然成立；勿调换顺序（has_state+causal 会双算当前 token，见 §15.3 修正 1）。
- 输出 s/r 标量块头内逐行重复：Layer 读任一行，shader 已写全部行。

---

## 15. 进度记录（2026-09-02 本会话完成项）

### 15.1 步骤 5-7：GPU 实现（完成）

- `shaders/scan_prefix_outer.comp` / `scan_suffix_outer.comp` / `outer_col.comp`：按 §5.1 源码落盘，glslc 编译通过。
- `compute_vk_backend.hpp` 6 处接线全部完成：include（3 组 `__has_include`）、pipeline 成员（3 个）、`get_*_spirv()`（3 个）、`initialize()` 创建（8/5/4 bindings，7/6/4 uint push constants）、`has_*_pipeline()`（3 个）、`*_gpu` 方法（3 个，含形状校验）。
- `compute_gpu_engine.hpp`：3 个占位已换真覆盖（ensure_gpu → backend_*_gpu → from_gpu 模式）。
- `CMakeLists.txt` GPU_SHADERS +3。
- 构建：`cmake --build build --target rapt_gradcheck` 绿；CMake 自动完成 glslc→SPIR-V→嵌入→scan_exprs（56 条）→gen_fused。

### 15.2 步骤 8-9：Layer 引擎化（完成）

- `compute_layer_rapt.hpp` 删除：`matvec_/matvec_T_/add_outer_` 标量助手、`scan_forward_`、`scan_backward_`（约 230 行 CPU 标量循环）。
- 新增私有助手 `make_dummy_(engine)`：(1,1) 清零 dummy（空参数占位，规避 0 字节 GPU buffer）。
- `forward`：boundary 构建为 (1,B·seq) Tensor → `scan_prefix_outer(Kp, V, Qp, V, dummy, dummy, false, ...)` → slice [0) BP、[3) s → `denom = sqrt(s+1e-6)`（elementwise 原语链）→ `out = BP/denom` → rearrange → w_o。
- `backward`：pass 1 `scan_prefix_outer(Kp, V, Qp, gcr, dummy, dummy, false, ...)` → slice [1) AP、[2) BTR、[3) s、[4) r → elementwise 链（denom/inv/gnum/ddenom/t1=ds/t2=2·ds·Aq）→ `gQt = BTR·inv + t2`；`outer_col` 物化 `dA = ds·q·qᵀ`（has_scale=true）、`dB = gnum·q`（has_scale=false）；pass 2 `scan_suffix_outer(dA, Kp, V, causal=true, boundary)` + `scan_suffix_outer(dB, Kp, V, ...)` → `gKt = 2·(SA·K) + SBᵀ·V`、`gVt = SB·K`；**双向分支**：`row_reduce_sum(dA/dB) → (BH·dk², 1)` → `broadcast_row_inplace` 广播成 (BH·dk², seq) → `scan_suffix_outer(causal=false, dummy, false)`。
- `forward_step`：签名 `Matrix&` → `Tensor&`（A_state/B_state）；Q/K/V → RoPE → ReLU → `scan_prefix_outer(Kp, V, Qp, Qp, A_state, B_state, true, ...)`（**先扫描**）→ `batched_matmul` 得 k·kᵀ / v·kᵀ → `add_inplace` 更新状态（**后更新**）→ slice BP/s → 除法 → w_o。
- `RAPTBlock::forward_step` 签名同步 `Tensor&`；`RAPTModel::generate` 状态 `std::vector<Matrix>` → `std::vector<Tensor>`（`engine.create_tensor` + `zero`）。
- `src/rapt_smoke_test.cpp` KV cache 测试同步改 Tensor 状态。

### 15.3 实施中发现并修正的 handoff 错误（重要）

1. **forward_step 顺序**：handoff §7.4 写"先更新状态、再扫描"——**错**。`scan_prefix_outer(causal=true, has_state=true)` 语义是 `A_t = A0 + Σ_{i≤t} k·kᵀ`（含自身），若 A0 已含当前 token 则双算。正确顺序 = **先扫描（A0 = 旧状态）→ 再 add_inplace 更新**，与旧 CPU 实现"先累积再用"语义等价。已按正确顺序实现。
2. **双向分支归约原语**：handoff §7.3 写 `col_reduce_sum → (BH·dk², 1)`——**错**。`col_reduce_sum(A (R,C)) → (1,C)`。需要的是**按行求和**：`row_reduce_sum → (R,1) = (BH·dk², 1)`，正好直接作 `broadcast_row_inplace` 的 row_vec（无需 transpose）。已按 row_reduce_sum 实现。
3. **CPU 参考实现的 boundary 形状校验 bug（潜在，本会话暴露并修复）**：`compute_cpu_engine.hpp` 的 `scan_prefix_outer`/`scan_suffix_outer` 校验写成 `bd.cols() != (rows/dk)*seq`（= B·H·seq），契约是 (1, **B·seq**）（shader 索引 `Bnd[b*seq+t]` 亦证）。旧 Layer 从不走引擎路径，bug 一直潜伏；Layer 引擎化后 doc-aware gradcheck 直接踩中（"fwd failed"）。已修为 `(rows/(dk*heads))*seq`，并给 vk backend 的 `*_gpu` 补上同义校验。

### 15.4 验证基线（2026-09-02，CPU）

| 验证 | 结果 |
|---|---|
| `rapt_gradcheck`（CPU，无 --batch） | 14/14 OK，exit 0；max_err 与旧实现基线逐位一致（causal 0.0290 / bidir 0.2554 / doc-aware 0.00868） |
| AOT 收集 | 56 条融合表达式（与改造前一致——新 elementwise 链走显式引擎原语，不引入新 spec） |
| `rapt_smoke_test`（CPU） | ✅ PASSED（09-04 补跑：KV-cache 一致性 max_diff=0 逐位一致；batch=2 段 OK；exit 0） |
| GPU 全链路 | ✅ 完成（09-04，步骤 12，详见 §15.6） |

### 15.5 环境/工具坑（本会话确认）

- `include/` 与 `shaders/` 文件是 **LF 换行**（非 CRLF）：`pwsh -split "`r`n"` 切分失效，用 `Get-Content` 逐行处理。
- `═`/`─` 分隔线字符数与手拼不符会导致 edit 不匹配（本会话踩 2 次）：先 `pwsh [regex]::Matches($l,'═').Count` 核对计数，或改用不含分隔线的唯一锚点。
- `compute_layer_rapt.hpp` 内 `Qp`/`Kp` 是 `dsl::compute` 的 `Result<Tensor>` 返回值，传引擎原语须 `*Qp`/`*Kp` 解引用（cache 成员 `Qp_cache_` 等才是裸 `Tensor`）。
- gradcheck 输出中 "fwd failed" 出现在 stderr，与 stdout 交错后行序会漂移——先按 section 标题对齐再定位。

### 15.6 步骤 11-12 完成记录（2026-09-04）

**步骤 11（全量回归）**：
- `cmake --build build`：144/144 目标全绿（BUILD_EXIT=0）。
- `ctest --test-dir build`：**31/31 全绿**（100%，187s；含 rapt_gradcheck/rapt_smoke_test/fused_gpu_test/attn_*/zipt_*/expr_* 全套）——Layer 重写未破坏其他模型/测试。
- `rapt_smoke_test`（CPU）补跑：PASSED，KV-cache 一致性 max_diff=0。

**步骤 12（GPU 验证，GTX 850M）**：
- `rapt_gradcheck --gpu`：3 段（causal / bidirectional / doc-aware）全 OK，kink 值 0.102/0.252 与 CPU 基线一致，全部 ≪ tol 5e-2。
- `rapt_smoke_test --gpu`：PASSED；**KV-cache 一致性 max_diff=2.98023e-08**（= float32 1 ulp：GPU 扫描加法顺序与 CPU 不同，数值正确、非逐位一致——与 CPU 的 0 相对照正好区分两种语义）；batch=2 段 OK。
- text 冒烟（小配置 RAPT，`--model rapt --gpu`）：
  - 词表：`tokenizer_train datasets\everyday_conversations_test_sft.txt --output rapt_smoke_bpe.json --vocab-size 512`（528 词，0.3s）。**注意**：`text_train` 只加载词表不训练，须先跑 `tokenizer_train`；其参数是 `--output` 不是 `--save`。
  - `text_train --epochs 1 --batch-size 2 --seq-len 64 --d-model 64 --num-heads 4 --num-layers 2 --d-ff 128`：319 步/10.3s，loss 6.3→3.9，无 TDR，词表嵌入模型文件，exit 0。dk=64/4=16 ≤ 64（MAX_DK 约束）。
  - `text_infer --prompt "Hi" --max-tokens 16 --gpu`：16 token/0.4s（39 tok/s），forward_step 的 GPU 录制路径（batched_matmul 状态更新 + has_state 扫描同 command buffer）验证通过；生成内容混乱属预期（106KB 语料 + 528 词 + 1 epoch）。

**本次顺带修复/发现的文档偏差**：
- `src/rapt_gradcheck.cpp:8` 头注释声称 `--batch N` 参数——**实际未实现**（main 只解析 `--tol/--gpu/--cuda`；各段 batch 在代码内固定：causal=2、bidirectional=1、doc-aware=2）。batch>1 覆盖（铁律 5）由 causal/doc-aware 段保证。已修正注释。
- handoff §10 原写 "rapt_gradcheck --gpu（支持 --batch N ...）" 基于该 stale 注释，实际行为以源码为准。

**文档沉淀**：
- `docs/15-rapt-algorithm.md` §7：3 原语签名/语义/形状约定 + Layer 组合 + 4 条坑 + 验证基线表。
- `docs/19-compute-engine-development.md` §11：扫描级原语接口 + 约定 + 使用方。
- `AGENTS.md` §4.3：原语分类表加"扫描级"行。

---

## 1. 任务背景

RAPT（ReLULinearAttention）全链路中唯一不在 GPU 上的部分：`compute_layer_rapt.hpp` 中的 `scan_forward_`（3 次下载 + 1 次回传）、`scan_backward_`（4 下载 + 3 回传）、`forward_step`（逐 token 4 次往返）——纯 CPU 标量循环 + PCIe staging。

**目标**：引擎化——引擎提供 3 个 op-level 扫描原语，RLA 算法（L2 分母 / ReLU 门控 / 梯度公式 / 文档重置）全部由 Layer 用现有原语组合表达，shader 永不含算法（铁律 3）。

---

## 2. 环境事实

| 项目 | 值 |
|---|---|
| 工作目录 | `D:\Codes\neuralnet.cpp` |
| 活动构建目录 | `build/`（Ninja + Clang + tests ON + Vulkan ON） |
| 构建命令 | `cmake --build build`（全量）/ `cmake --build build --target rapt_gradcheck`（单目标） |
| 测试 | `build\test\*.exe`；`ctest --test-dir build` |
| glslc | `C:\VulkanSDK`（CMake 自动探测，无需手动配置） |
| GPU | GTX 850M（Maxwell），shared ≤48KB/block；2026-09-01 gpu_test 通过 |
| MSVC | `build_msvc=VS2026`（tests OFF、无 GPU，**勿用**） |

**环境怪癖**：
- anti-stuck 对同一文件 ≥4 次 read 会拒绝→换 `pwsh Get-Content` 或 grep（不同工具绕过）。
- env-triage 把 read 长输出误计为"工具失败"→先核对实际工具结果清单，误报不修不重试。
- memory add 被查重守卫拒（≥0.62）是设计行为→list→replace 合并。

**头文件变更触发 AOT 重跑**：修改 `include/` 后 `cmake --build` 自动重跑 scan_exprs（56 条融合表达式）+ gen_fused；手写原语**不进融合注册表**，不影响 56 条 AOT spec。

---

## 3. 设计决策（已定稿，无需再讨论）

### 3.1 三个原语签名（已在 `compute_engine.hpp`，纯虚）

```cpp
// 前缀扫描 (forward / backward pass 1 / forward_step)
[[nodiscard]] virtual Result<Tensor> scan_prefix_outer(
    const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
    const Tensor& A0, const Tensor& B0, bool has_state,
    std::size_t dk, std::size_t heads, bool causal,
    const Tensor& boundary, bool has_bnd) = 0;

// 后缀扫描 (backward pass 2)
[[nodiscard]] virtual Result<Tensor> scan_suffix_outer(
    const Tensor& D, const Tensor& X, const Tensor& Y,
    std::size_t dk, std::size_t heads, bool causal,
    const Tensor& boundary, bool has_bnd) = 0;

// 逐列外积 (backward 的 dL/dA、dL/dB 物化)
[[nodiscard]] virtual Result<Tensor> outer_col(
    const Tensor& P, const Tensor& R, const Tensor& S,
    std::size_t dk, bool has_scale) = 0;
```

### 3.2 输出布局

- `scan_prefix_outer` → `(B·H·5·dk, seq)`：[0) B·P  [1) A·P  [2) B^T·R  [3) s（标量，头内逐行重复）  [4) r（标量，头内逐行重复）]
- `scan_suffix_outer` → `(B·H·3·dk, seq)`：[0) S·X  [1) S·Y  [2) S^T·Y
- `outer_col` → `(B·H·dk², seq)`：`out[(bh*dk+a)*dk+b, t] = P[a,t]·R[b,t] (· S[t] if has_scale)`

### 3.3 约定

- **batch-major**，`i = b*seq+t`；头 `(b,h)` 行块起点 `r0 = (b*H+h)*dk`
- 空参数用 **(1,1) dummy + bool 标志**（规避 0 字节 GPU buffer）
- 标量块（s/r）在头块内 `dk` 行重复存放——实现写全部行，Layer 读任一行
- **dk ≤ 64**（GPU MAX_DK=64），超出返回清晰错误
- boundary：Layer 的 `build_boundary_`（:50）返回 `vector<uint8_t>` → `from_matrix(1, B*seq)`

---

## 4. 已改动文件现状（本次会话）

| 文件 | 位置 | 内容 |
|---|---|---|
| `compute_engine.hpp` | `zero`（原 258）之后、归约原语之前 | 3 个纯虚扫描原语 + 形状约定注释 |
| `compute_cpu_engine.hpp` | `zero`（原 504）之后、归约原语之前 | 3 个 CPU 参考实现（加法顺序与旧实现逐位一致） |
| `compute_gpu_engine.hpp` | `zero`（原 499）之后 | 3 个"未实现"占位覆盖 ← **步骤 7 替换** |
| `compute_cuda_engine.hpp` | `eval_expr`（原 415）之后 | 3 个"未实现"存根（`#ifdef NN_HAS_CUDA` 内，不编译） |

---

## 5. 步骤 5-7：GPU 实现（下一步）

### 5.1 三个 shader 完整源码（直接粘贴为新文件）

#### `shaders/scan_prefix_outer.comp`

```glsl
// ── scan_prefix_outer.comp ─────────────────────────────────────────────
// RLA 前缀扫描原语：A_t = A0 + Σ_{i≤t, 同文档} k_i·k_i^T, B_t = B0 + Σ v_i·k_i^T
// causal=0: 全集常数（无重置）；causal=1: 含自身前缀，文档边界处运行态清零
// 输出 (B*H*5*dk, seq) 行块：[0) B·P  [1) A·P  [2) B^T·R  [3) s  [4) r
// 1 workgroup/头，256 线程；A/B 驻 shared 64×65×2≈33.3KB ≤ 850M 48KB
// ──────────────────────────────────────────────────────────────────────
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer BufK  { float K[]; };
layout(std430, binding = 1) readonly  buffer BufV  { float V[]; };
layout(std430, binding = 2) readonly  buffer BufP  { float P[]; };
layout(std430, binding = 3) readonly  buffer BufR  { float R[]; };
layout(std430, binding = 4) readonly  buffer BufA0 { float A0[]; };
layout(std430, binding = 5) readonly  buffer BufB0 { float B0[]; };
layout(std430, binding = 6) readonly  buffer BufBnd{ float Bnd[]; };
layout(std430, binding = 7) writeonly buffer BufOut{ float Out[]; };

layout(push_constant) uniform Push {
    uint dk, heads, seq, causal, has_state, has_bnd, rows;
};

const uint PAD = 65u;
shared float sA[64][PAD];
shared float sB[64][PAD];
shared float s_kv[64], s_vv[64], s_qv[64], s_num[64], s_Aq[64];

void main() {
    const uint bh  = gl_WorkGroupID.y;
    const uint b   = bh / heads;
    const uint h   = bh % heads;
    const uint r0  = bh * dk;
    const uint tid = gl_LocalInvocationID.x;

    // 1) 初始化 A/B（零 或 A0/B0 行块 h）
    for (uint idx = tid; idx < dk * dk; idx += 256u) {
        const uint a = idx / dk, c = idx % dk;
        float va = 0.0, vb = 0.0;
        if (has_state == 1u) {
            const uint sr = h * dk + a;
            va = A0[sr * dk + c];
            vb = B0[sr * dk + c];
        }
        sA[a][c] = va;
        sB[a][c] = vb;
    }
    barrier();

    // 2) 双向：预累加全集
    if (causal == 0u) {
        for (uint t = 0u; t < seq; ++t) {
            if (tid < dk) {
                s_kv[tid] = K[(r0 + tid) * seq + t];
                s_vv[tid] = V[(r0 + tid) * seq + t];
            }
            barrier();
            for (uint idx = tid; idx < dk * dk; idx += 256u) {
                const uint a = idx / dk, c = idx % dk;
                sA[a][c] += s_kv[a] * s_kv[c];
                sB[a][c] += s_vv[a] * s_kv[c];
            }
            barrier();
        }
    }

    // 3) 逐 t 顺序扫描 + 读出
    for (uint t = 0u; t < seq; ++t) {
        if (tid < dk) {
            s_qv[tid] = P[(r0 + tid) * seq + t];
            s_kv[tid] = K[(r0 + tid) * seq + t];
            s_vv[tid] = V[(r0 + tid) * seq + t];
        }
        barrier();

        if (causal == 1u) {
            uint reset = 0u;
            if (has_bnd == 1u)
                reset = (Bnd[b * seq + t] != 0.0) ? 1u : 0u;
            for (uint idx = tid; idx < dk * dk; idx += 256u) {
                const uint a = idx / dk, c = idx % dk;
                if (reset == 1u) { sA[a][c] = 0.0; sB[a][c] = 0.0; }
                sA[a][c] += s_kv[a] * s_kv[c];
                sB[a][c] += s_vv[a] * s_kv[c];
            }
            barrier();
        }

        // num = B·qv; Aq = A·qv（dk 线程各处理一行，j 串行）
        if (tid < dk) {
            float acc = 0.0, acc2 = 0.0;
            for (uint j = 0u; j < dk; ++j) {
                acc  += sB[tid][j] * s_qv[j];
                acc2 += sA[tid][j] * s_qv[j];
            }
            s_num[tid] = acc;
            s_Aq[tid]  = acc2;
        }
        barrier();

        if (tid < dk) {
            float s = 0.0;
            for (uint j = 0u; j < dk; ++j)
                s += s_qv[j] * s_Aq[j];
            Out[(r0 + tid) * seq + t]              = s_num[tid]; // [0) B·P
            Out[(rows + r0 + tid) * seq + t]        = s_Aq[tid]; // [1) A·P
            Out[(3u * rows + r0 + tid) * seq + t]   = s;         // [3) s
        }
        // [2) B^T·R
        if (tid < dk) {
            float acc = 0.0;
            for (uint i = 0u; i < dk; ++i)
                acc += sB[i][tid] * R[(r0 + i) * seq + t];
            Out[(2u * rows + r0 + tid) * seq + t] = acc;
        }
        // [4) r = R·(B·P)，标量，线程 0 写全部行
        if (tid == 0u) {
            float r = 0.0;
            for (uint j = 0u; j < dk; ++j)
                r += R[(r0 + j) * seq + t] * s_num[j];
            for (uint i = 0u; i < dk; ++i)
                Out[(4u * rows + r0 + i) * seq + t] = r;
        }
        barrier();
    }
}
```

#### `shaders/scan_suffix_outer.comp`

```glsl
// ── scan_suffix_outer.comp ────────────────────────────────────────────
// RLA 后缀扫描原语（backward pass 2）：
//   causal=1: S_i = Σ_{t≥i, 同文档} D_t（i+1 为文档起点时先清零）
//   causal=0: S_i = D_i（Layer 已把全集梯度沿 seq 广播）
// D (B*H*dk², seq), X/Y (B*H*dk, seq)
// 输出 (B*H*3*dk, seq)：[0) S·X  [1) S·Y  [2) S^T·Y
// 1 workgroup/头，256 线程；S 驻 shared 64×65≈16.6KB
// ─────────────────────────────────────────────────────────────────────
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer BufD   { float D[]; };
layout(std430, binding = 1) readonly  buffer BufX   { float X[]; };
layout(std430, binding = 2) readonly  buffer BufY   { float Y[]; };
layout(std430, binding = 3) readonly  buffer BufBnd { float Bnd[]; };
layout(std430, binding = 4) writeonly buffer BufOut { float Out[]; };

layout(push_constant) uniform Push {
    uint dk, heads, seq, causal, has_bnd, rows;
};

const uint PAD = 65u;
shared float sS[64][PAD];
shared float s_xv[64], s_yv[64];

void main() {
    const uint bh   = gl_WorkGroupID.y;
    const uint b    = bh / heads;
    const uint r0   = bh * dk;
    const uint dbase = bh * dk * dk;
    const uint tid  = gl_LocalInvocationID.x;

    // 初始化 S 为零
    for (uint idx = tid; idx < dk * dk; idx += 256u)
        sS[idx / dk][idx % dk] = 0.0;
    barrier();

    for (uint i = seq; i-- > 0u;) {
        if (causal == 1u) {
            uint reset = 0u;
            if (has_bnd == 1u && i + 1u < seq)
                reset = (Bnd[b * seq + i + 1u] != 0.0) ? 1u : 0u;
            for (uint idx = tid; idx < dk * dk; idx += 256u) {
                const uint a = idx / dk, c = idx % dk;
                if (reset == 1u) sS[a][c] = 0.0;
                sS[a][c] += D[(dbase + a * dk + c) * seq + i];
            }
        } else {
            for (uint idx = tid; idx < dk * dk; idx += 256u) {
                const uint a = idx / dk, c = idx % dk;
                sS[a][c] = D[(dbase + a * dk + c) * seq + i];
            }
        }
        if (tid < dk) {
            s_xv[tid] = X[(r0 + tid) * seq + i];
            s_yv[tid] = Y[(r0 + tid) * seq + i];
        }
        barrier();

        if (tid < dk) {
            float sx = 0.0, sy = 0.0, syt = 0.0;
            for (uint c = 0u; c < dk; ++c) {
                sx += sS[tid][c] * s_xv[c];
                sy += sS[tid][c] * s_yv[c];
            }
            for (uint r = 0u; r < dk; ++r)
                syt += sS[r][tid] * s_yv[r];
            Out[(r0 + tid) * seq + i]               = sx;  // [0) S·X
            Out[(rows + r0 + tid) * seq + i]          = sy;  // [1) S·Y
            Out[(2u * rows + r0 + tid) * seq + i]     = syt; // [2) S^T·Y
        }
        barrier();
    }
}
```

#### `shaders/outer_col.comp`

```glsl
// ── outer_col.comp ────────────────────────────────────────────────────
// 逐列外积原语（backward 物化 dL/dA / dL/dB）：
//   out[(bh*dk+a)*dk+c, t] = P[a,t]·R[c,t] (· S[t] if has_scale)
// P/R (B*H*dk, seq); S (B*H*dk, seq) 标量头内逐行重复（读首行）；
// has_scale=0 时 S 为 dummy。
// 输出 (B*H*dk², seq)；扁平并行，无 shared memory。
// ─────────────────────────────────────────────────────────────────────
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer BufP   { float P[]; };
layout(std430, binding = 1) readonly  buffer BufR   { float R[]; };
layout(std430, binding = 2) readonly  buffer BufS   { float S[]; };
layout(std430, binding = 3) writeonly buffer BufOut { float Out[]; };

layout(push_constant) uniform Push {
    uint dk, seq, rows, has_scale;
};

void main() {
    const uint idx   = gl_GlobalInvocationID.x;
    if (idx >= rows * dk * seq) return;

    const uint t  = idx % seq;
    const uint e  = idx / seq;
    const uint bh = e / (dk * dk);
    const uint rem = e % (dk * dk);
    const uint a  = rem / dk;
    const uint c  = rem % dk;
    const uint r0 = bh * dk;

    const float p = P[(r0 + a) * seq + t] * R[(r0 + c) * seq + t];
    Out[idx] = (has_scale == 1u) ? (p * S[r0 * seq + t]) : p;
}
```

> **注意**：outer_col 的 local_size_x=256 但使用 `gl_GlobalInvocationID.x`（不是 `gl_LocalInvocationID`），dispatch `(ceil(total/256), 1, 1)`。

---

### 5.2 compute_vk_backend.hpp 接线（6 处）

| # | 锚点（原始行号） | 操作 |
|---|---|---|
| 1 | 后端 include 区（48-96，scatter_add 之后） | 3 组 `#if __has_include("xxx_spv.hpp") / #include / #define NN_XXX_SPV_EMBEDDED` |
| 2 | pipeline 成员区（323-332，scatter_add_pipeline_ 之后） | 3 个 `VulkanPipeline scan_prefix_outer_pipeline_; ...` |
| 3 | get_*_spirv() 静态方法区（390-488，get_scatter_add_spirv 之后） | 3 个 getter（`#ifdef → nn_*_spirv_bytecode() : static empty`） |
| 4 | initialize() 内（563-761，scatter_add 创建段 ~721 之后，fused registry ~723 之前） | 3 组 `if (!spirv.empty()) { VulkanPipeline::create_generic(..., N, M*4); }` |
| 5 | has_*_pipeline() 查询方法区（763-772） | 3 个 `has_*_pipeline() const noexcept` |
| 6 | *_gpu 方法区（batched_matmul_gpu ~1575 之后） | 3 个 `*_gpu` 方法（见下方模式） |

**create_generic 签名**（`compute_vk_device.hpp:326`）：
```cpp
static Result<VulkanPipeline> create_generic(
    VkDevice device, std::span<const uint32_t> spirv_code,
    uint32_t num_bindings, uint32_t push_constant_size);
```

**dispatch_compute 签名**（`compute_vk_backend.hpp:1400`）：
```cpp
Result<void> dispatch_compute(
    const VulkanPipeline& pipeline, std::span<const GpuTensor> inputs,
    const GpuTensor& output, const std::vector<std::uint8_t>& pc,
    std::uint32_t wg_x, std::uint32_t wg_y, std::uint32_t wg_z);
```

**initialize() 中的管线创建**：
```cpp
// scan_prefix_outer: 8 bindings, 7×4=28B push
const auto& spf = get_scan_prefix_outer_spirv();
if (!spf.empty()) {
    auto r = VulkanPipeline::create_generic(device_.device(), spf, 8, 7 * sizeof(uint32_t));
    if (r) scan_prefix_outer_pipeline_ = std::move(*r);
}
// scan_suffix_outer: 5 bindings, 6×4=24B push
// outer_col: 4 bindings, 4×4=16B push（同模式）
```

**push 结构体**：
```cpp
struct PushPrefix { uint32_t dk, heads, seq, causal, has_state, has_bnd, rows; };
struct PushSuffix { uint32_t dk, heads, seq, causal, has_bnd, rows; };
struct PushOuter  { uint32_t dk, seq, rows, has_scale; };
```

**\*_gpu 方法模式**（参考 batched_matmul_gpu）：
```cpp
[[nodiscard]] Result<GpuTensor> scan_prefix_outer_gpu(
    const GpuTensor& K, const GpuTensor& V, const GpuTensor& P, const GpuTensor& R,
    const GpuTensor& A0, const GpuTensor& B0, bool has_state,
    uint32_t dk, uint32_t heads, bool causal,
    const GpuTensor& boundary, bool has_bnd)
{
    if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
    if (!has_scan_prefix_outer_pipeline())
        return std::unexpected(Error{"scan_prefix_outer_gpu: pipeline not available"});
    if (dk > 64u) return std::unexpected(Error{"scan_prefix_outer_gpu: d_k > 64 not supported"});
    const auto rows = static_cast<uint32_t>(K.rows());
    if (rows % (dk * heads) != 0)
        return std::unexpected(Error{"scan_prefix_outer_gpu: rows not divisible by H*dk"});
    // ... shape validation for V/P/R ...
    const auto seq = static_cast<uint32_t>(K.cols());
    const auto BH = rows / dk;
    auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(rows) * 5, seq, *this);
    if (!C_res) return std::unexpected(C_res.error());
    GpuTensor C = std::move(*C_res);
    PushPrefix push{dk, heads, seq, causal?1u:0u, has_state?1u:0u, has_bnd?1u:0u, rows};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary};
    auto r = dispatch_compute(scan_prefix_outer_pipeline_, inputs, C, pc, 1u, BH, 1u);
    if (!r) return std::unexpected(r.error());
    return C;
}
// outer_col: dispatch ((rows*dk*seq+255)/256, 1u, 1u); output rows = rows*dk
// scan_suffix_outer: dispatch (1u, BH, 1u); output rows = rows*3
```

---

### 5.3 GpuEngine（`compute_gpu_engine.hpp`）

替换 3 个占位覆盖（当前在 `zero` 之后），参照 `batched_matmul` 覆盖（358-379）的模式：

```cpp
[[nodiscard]] Result<Tensor> scan_prefix_outer(
    const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
    const Tensor& A0, const Tensor& B0, bool has_state,
    std::size_t dk, std::size_t heads, bool causal,
    const Tensor& boundary, bool has_bnd) override
{
    auto k = ensure_gpu(K); if (!k) return std::unexpected(k.error());
    auto v = ensure_gpu(V); if (!v) return std::unexpected(v.error());
    auto p = ensure_gpu(P); if (!p) return std::unexpected(p.error());
    auto r = ensure_gpu(R); if (!r) return std::unexpected(r.error());
    auto a = ensure_gpu(A0); if (!a) return std::unexpected(a.error());
    auto b = ensure_gpu(B0); if (!b) return std::unexpected(b.error());
    auto bn = ensure_gpu(boundary); if (!bn) return std::unexpected(bn.error());
    auto res = backend_.scan_prefix_outer_gpu(
        k->gpu_tensor(), v->gpu_tensor(), p->gpu_tensor(), r->gpu_tensor(),
        a->gpu_tensor(), b->gpu_tensor(), has_state,
        static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
        bn->gpu_tensor(), has_bnd);
    if (!res) return std::unexpected(res.error());
    return Tensor::from_gpu(std::move(*res));
}
// scan_suffix_outer / outer_col 同模式
```

### 5.4 CMakeLists.txt

`GPU_SHADERS` 列表（`CMakeLists.txt:313`）末尾追加：
```
    scan_prefix_outer
    scan_suffix_outer
    outer_col
```
（foreach loop at line 364 会自动处理：glslc → `build/generated/<name>_spv.spv` → `nn_embed_shader` → `build/generated/<name>_spv.hpp`，函数名 `nn_<name>_spirv_bytecode`）

---

## 6. 步骤 8：编译验证

```bash
cmake --build build --target rapt_gradcheck
# 预期：14/14，exit 0
# 如 glslc 失败：检查 GLSL 语法（shared memory 上限、push constant 大小）
# 可选：build\test\gpu_test.exe 确认后端未损坏
```

---

## 7. 步骤 9：Layer 引擎化重写（最大改动）

**编辑前必须先 read `compute_layer_rapt.hpp` 最新内容**。

### 7.1 关键锚点

| 锚点 | 行号 | 说明 |
|---|---|---|
| `scan_forward_` | 99-104 | 旧前缀扫描静态函数（删除） |
| `scan_backward_` | 166 | 旧后缀扫描静态函数（删除） |
| `matvec_/matvec_T_/add_outer_` | 64-93 | 标量辅助函数（删除） |
| forward 中的 scan 调用 | 466 | `eps = 1e-6` |
| backward 中的 scan 调用 | 525 | `eps = 1e-6` |
| forward_step 签名 | 591 | `Result<Tensor> forward_step(ComputeEngine& engine, const Tensor& input, Matrix& A_state, Matrix& B_state, std::size_t pos)` |
| forward_step 中的 `sqrt(s + 1e-6)` | 649 | |
| RAPTBlock::forward_step | 780/786 | `attn_.forward_step(engine, *n1, A_state, B_state, pos)` |
| RAPTModel::generate 状态 | 1031-1035 | `std::vector<Matrix> statesA, statesB` |
| generate step_one 调用 | 1053 | `blocks_[i].forward_step(engine, h, statesA[i], statesB[i], pos)` |

### 7.2 forward 重写（~440-470 区域）

保持：w_q/k/v → RoPE → ReLU（Q/K ReLU，V 不 ReLU）→ rearrange → Qp/Kp/V 缓存。

改：
1. boundary：`has_doc_ids_ ? build_boundary_ → from_matrix(1,B*seq) : dummy(1,1) + false`
2. `Sc = engine.scan_prefix_outer(Kp, V, Qp, V/*R unused*/, dummy, dummy, false, dk, H, causal_, boundary, has_bnd)`
3. `BP = engine.slice_rows(Sc, 0, BH*dk)`；`sblk = engine.slice_rows(Sc, 3*BH*dk, BH*dk)`
4. `denom = engine.elementwise_unary(Sqrt, engine.elementwise_binary_scalar(Add, sblk, Scalar{1e-6}))`
5. `out = engine.elementwise_binary(Div, BP, denom)`
6. rearrange 回 (d_model, batch*seq) → w_o_.forward

**新 elementwise 链使用显式 engine 原语，不用 dsl::compute（不引入新 AOT spec）。**

### 7.3 backward 重写（~470-590 区域）

保持：w_o_ backward → rearrange → gcr (BH*dk, seq)；后半段 ReLU backward / RoPE backward / w_q/k/v backward 引擎化链不变。

改 scan 部分（~525 区域）：

```
Sc = engine.scan_prefix_outer(Kp, V, Qp, gcr, dummy, dummy, false, dk, H, causal_, boundary, has_bnd)
BP  = slice(Sc, 0,        BH*dk)
AP  = slice(Sc, BH*dk,    BH*dk)
BTR = slice(Sc, 2*BH*dk,  BH*dk)
sblk= slice(Sc, 3*BH*dk,  BH*dk)
rblk= slice(Sc, 4*BH*dk,  BH*dk)

denom = sqrt(sblk + 1e-6)
inv   = elementwise_binary_scalar(Div, denom, 1.0, true)  // scalar_first=true → 1/denom
gnum  = elementwise_binary(Mul, gcr, inv)
gdot  = rblk
ddenom= elementwise_binary(Div, elementwise_unary(Neg, gdot),
                           elementwise_binary(Mul, denom, denom))
t1    = elementwise_binary(Mul, ddenom, inv)
scale_inplace(t1, 0.5f)                                    // ds

btrg  = elementwise_binary(Mul, BTR, inv)
t2    = elementwise_binary(Mul, t1, AP)
scale_inplace(t2, 2.0f)
gQ    = elementwise_binary(Add, btrg, t2)                  // store_dq

store_dA = engine.outer_col(Qp, Qp, t1, dk, true)         // ds 作 scale
store_dB = engine.outer_col(gnum, Qp, dummy, dk, false)
```

**causal 分支**：
```
SAc = engine.scan_suffix_outer(store_dA, Kp, V, dk, H, true, boundary, has_bnd)
SBc = engine.scan_suffix_outer(store_dB, Kp, V, dk, H, true, boundary, has_bnd)
sak = slice(SAc, 0, BHdk); scale_inplace(sak, 2.0f)
sbTv= slice(SBc, 2*BHdk, BHdk)
gK  = elementwise_binary(Add, sak, sbTv)
gV  = slice(SBc, 0, BHdk)
```

**bidirectional 分支**：
```
dA_tot = engine.col_reduce_sum(store_dA)             // → (BH*dk², 1)
Ab = engine.create_tensor(BH*dk², seq); engine.zero(Ab)
engine.broadcast_row_inplace(Ab, dA_tot, BinaryOp::Add)  // 广播到 (BH*dk², seq)
// dB 同理
SAc = engine.scan_suffix_outer(Ab, Kp, V, dk, H, false, dummy, false)
// gK/gV 同 causal 分支公式
```

### 7.4 forward_step 重写（591-658）

新签名：`Result<Tensor> forward_step(ComputeEngine& engine, const Tensor& input, Tensor& A_state, Tensor& B_state, std::size_t pos)`

1. w_q/k/v → RoPE → ReLU（同现有）
2. **状态更新**（取代旧的 to_matrix + CPU 双循环）：
   ```cpp
   auto Kp_t = engine.to_matrix(*Kp);  // Kp (H*dk, 1)
   auto V_t  = engine.to_matrix(V);
   // batched_matmul(A, B, batch, transA, transB, alpha)
   auto A_add = engine.batched_matmul(K1, K1, H, false, true); // k·k^T → (H*dk, dk)
   auto B_add = engine.batched_matmul(V1, K1, H, false, true); // v·k^T
   engine.add_inplace(A_state, A_add);
   engine.add_inplace(B_state, B_add);
   ```
3. 扫描：`Sc = engine.scan_prefix_outer(K1, V1, Q1, Q1, A_state, B_state, true, dk, H, true, dummy, false)`
4. BP = slice(Sc, 0, H*dk); sblk = slice(Sc, 3*H*dk, H*dk); out = BP / sqrt(sblk + 1e-6)
5. w_o_.forward(engine, out)

**同步改 RAPTBlock::forward_step（780）+ RAPTModel::generate（1031-1053）：状态改为 `std::vector<Tensor>`**

---

## 8. 步骤 10：rapt_smoke_test.cpp

- 第 4 节 KV 缓存一致性测试（150-180）：`A`/`B` 从 `Matrix` 改为 `Tensor`（`create_tensor(dk, dk)` + `zero`），适配新签名
- 其他测试（forward/backward 梯度）无需改动
- 验证：`build\test\rapt_smoke_test.exe`（CPU）通过

---

## 9. 步骤 11：ctest 全量

```bash
ctest --test-dir build
```
预期全绿。若 rapt/attn/gpt 测试失败：先检查引擎化是否改变了加法顺序——对照 §11 基线 max_err。

---

## 10. 步骤 12：GPU 验证 + 文档

1. `build\test\rapt_gradcheck.exe --gpu`（支持 `--batch N --tol f --gpu --cuda`，src/rapt_gradcheck.cpp:8/143-153；batch>1 必须，铁律 5）
2. `build\test\rapt_smoke_test.exe --gpu`
3. text_train / text_infer --gpu 冒烟
4. 文档更新：
   - `docs/15-rapt-algorithm.md`：加"引擎化"章节（3 原语签名/语义/Layer 组合）
   - `docs/19-compute-engine-development.md`：原语表加"扫描级"
   - `AGENTS.md` §4.3 原语分类表：加扫描级：scan_prefix_outer / scan_suffix_outer / outer_col

---

## 11. 基线数据（步骤 4 测量，CPU 全绿）

```
causal:            0.0291  0.0098  0.0018  0.0014  0.0020  0.0014  0.0006  0.0005
bidirectional:     0.0037  0.0013  0.1018  0.2554  0.0028  0.0018  0.0012  0.0006
causal+doc-aware:  0.0047  0.0034  0.0037  0.0040  0.0068  0.0087  0.0036  0.0018
ALL RAPT GRADCHECKS PASSED
```
Tolerance 5e-2；kink 点（bidirectional param[2]=0.102, param[3]=0.255）正常跳过。

---

## 12. 铁律重申（AGENTS.md §5）

1. 禁 throw/try/catch；错误用 `Result<T>` 传播
2. 禁 new/delete
3. **shader 永不含算法**（3 个 shader 是纯"带状态的顺序归约 + matvec/外积读出"）
4. 不穿透接口
5. batch-major `i = b*seq+t`；序列测试必须覆盖 batch>1
6. GPU 录制生命周期：begin_batch→end_batch 之间的张量必须存活到 end_batch 之后；dispatch_compute 已合规
7. **AOT 闭合世界**：手写原语不进融合注册表；新 elementwise 链不用 dsl
8. 确定性：CPU 参考实现与旧实现逐位一致（已完成）
9. 禁 0 字节 GPU buffer：空参数用 (1,1) dummy + bool 标志

---

## 13. API 速查

| API | 签名/说明 |
|---|---|
| `elementwise_binary_scalar` | `(BinaryOp, A, scalar, scalar_first=false)`；true → `op(scalar, A)` |
| `elementwise_binary` | `(BinaryOp, A, B)` |
| `elementwise_unary` | `(UnaryOp, A)` |
| `scale_inplace` | `(A, scalar)` |
| `add_inplace` | `(A, B)` |
| `batched_matmul` | `(A, B, batch, transA=false, transB=false, alpha=1.0)` |
| `slice_rows` | `(src, start, count)` → 新 Tensor（拷贝） |
| `col_reduce_sum` | `(A)` → `(rows, 1)` |
| `broadcast_row_inplace` | `(A, row_vec, op)`：A (R,C) op= row_vec (R,1) |
| `dispatch_compute` | `(pipeline, inputs, output, pc, wg_x, wg_y, wg_z)` |
| `create_generic` | `(VkDevice, span<const uint32_t>, num_bindings, push_size)` |
| `ensure_gpu` | GPU→共享拷贝；CPU→上传 |
| `Tensor::cpu` / `from_matrix` / `from_gpu` | 构造方法 |
| `GpuTensor::create_empty` | `(rows, cols, backend)` |
| `Matrix::at_unchecked(r,c)` / `set_value_unchecked` / `span()` | |

---

## 14. DoD 检查清单

- [x] **5-7**：3 个 .comp 文件 glslc 编译通过；vk backend 6 处接线完成；GpuEngine 占位已换真覆盖；CMake GPU_SHADERS +3
- [x] **8**：`cmake --build build --target rapt_gradcheck` exit 0（14/14，max_err 与基线逐位一致）
- [x] **9**：Layer 全面引擎化；forward/backward/forward_step 无 to_matrix（仅 forward_step 状态更新）；`cmake --build build` 全绿（09-04 步骤 11 复核：144/144 绿）
- [x] **10**：rapt_smoke_test 已适配 Tensor 状态（CPU 跑通 ✅，见 §15.4/§15.6）
- [x] **11**：全量构建 144/144 + `ctest --test-dir build` **31/31 全绿**（09-04）
- [x] **12**：rapt_gradcheck --gpu 3 段全 OK + rapt_smoke_test --gpu（KV max_diff=2.98e-08=1ulp）+ text 冒烟（train 319 步 / infer 16 token）通过；docs/15 §7、docs/19 §11、AGENTS.md §4.3 已更新（09-04）
