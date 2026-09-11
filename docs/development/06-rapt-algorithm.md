# 线性注意力算法家族：RLA / RAPT / RLA-2 / LRLA 与 GPU 实现

> 本文整合项目自源起的 ReLU 线性注意力（RLA）算法共四份源文档（`15-rapt-algorithm.md`、`22-rla2.md`、`flash_attn_analysis.md`、`rapt-gpu-handoff.md`），按「动机 → RLA → RAPT → RLA-2/LRLA → 两趟式/flash 等价分析 → GPU 落地」的演进顺序组织成一篇连贯的文章，作为线性注意力家族的完整参考。
>
> 核心思想贯穿始终：**利用 ReLU 的非线性截断作为可微分的"硬注意力门控"，自动屏蔽无关噪声 Token，在保持 O(L) 时间复杂度和 O(L) 内存的前提下实现全局上下文的动态稀疏检索。**

---

## 1. 动机与目标

在研究长上下文建模时，标准 Transformer 的 Softmax 注意力具有 O(L²) 的计算与内存复杂度。线性注意力（Linear Attention）通过与核方法结合，将 softmax 核替换为可线性化的核函数，从而利用矩阵结合律把复杂度降到 O(L)。

本项目选择 **ReLU 线性注意力**路线，其核心哲学：

- **目标**：在保持 O(L) 时间复杂度和 O(1)/O(L) 内存增长（相对于上下文长度）的前提下，实现全局上下文的动态稀疏检索。
- **机理**：若某 query 与 key 在特征空间方向相反，其点积为负，ReLU 将其精确置为 0，从而"只激活语义相关的 Token"。
- **动态重选**：每预测一个新 Token 都会生成新的 q，算法强制基于全量原始键重新计算，不存在 RNN 隐状态的遗忘问题。

项目内实现的这一族注意力在架构图/模型中被称为 **RAPT（ReLULinearAttention）**。

---

## 2. RLA：原始的 ReLU-Linear Attention

### 2.1 算法数学定义（闭式解）

对于输入序列长度 L、隐藏维度 d，给定当前查询位置 t 的查询向量 `q_t ∈ R^d`，以及历史键值对 `(k_i, v_i)`（`i ∈ [1, L]`），RLA 的计算定义为严格的一行式：

```
Output_t = Σ_{i=1..L} σ(q_t)ᵀ σ(k_i) · v_i / ( sqrt( Σ_{i=1..L} ( σ(q_t)ᵀ σ(k_i) )² ) + ε )
```

其中：
- `σ(x) = ReLU(x) = max(0, x)`（关键：逐元素操作，非向量模长）；
- `ε` 为极小常数（如 1e-6）防止除零。

全局矩阵形式（训练并行视图）：令 `Q' = ReLU(QW_q)`、`K' = ReLU(KW_k)`、`V = VW_v`。

> 注意：公式中的分母是每个 query 独立计算的标量，无法简单地用一次矩阵除法解决，必须逐行归一化。

### 2.2 关键算法特性（必须遵守）

| 特性 | 描述 |
| :--- | :--- |
| **复杂度** | 计算 `Q'K'ᵀ` 若显式构建为 L×L 则是 O(L²)。**复现时必须利用结合律**：先算 `K'ᵀV`，再算 `Q'(K'ᵀV)`，复杂度降至 O(Ld²)。当 d ≪ L 时，视作线性。 |
| **稀疏激活** | 若 q_t 与 k_i 在特征空间方向相反，点积为负，ReLU 将其精确置为 0。实现"只激活语义相关的 Token"。 |
| **动态重选** | 每预测新 Token 生成新的 q_{t+1}，算法强制基于全量 L+1 个原始 k_i 重新计算，不存在 RNN 隐状态遗忘。 |
| **位置编码** | **严禁使用绝对位置编码（如 Learned PE）**。必须使用 RoPE（旋转位置编码）或 ALiBi，且 RoPE 必须施加在 Q 和 K 进入 ReLU 之前。 |

### 2.3 致命陷阱与解决方案

| 陷阱 | 现象 | 解决方案 |
| :--- | :--- | :--- |
| **直接照搬 Softmax 的 Scale** | 训练不稳定，Loss 震荡 | **禁止**使用 `1/sqrt(d_k)` 缩放。因为 ReLU 非负，点积随维度线性增长。改用法：将缩放放在 ReLU 之前，且对 Q/K 进行 LayerNorm 预处理。 |
| **分母计算导致梯度消失** | 长文本（大于 8k）时模型输出趋近 0 | 务必使用 L2 分母（平方和开根号），而非简单求和。求和分母会导致梯度爆炸/消失；L2 等价于对注意力分数做余弦相似度约束。 |
| **ReLU 导致大量维度永久死亡** | 模型失去表达能力 | 在 Q/K 投影后、ReLU 之前加一层 Shortcut（残差）：`q = ReLU(q) + 0.1*q`（Leaky ReLU 变体），保证负值有微弱梯度流过。 |

### 2.4 复杂度基准（预期结果）

单张 A100（80G），`dim=1024, batch=1`：

| 上下文长度 L | Softmax Attention (Flash) | RLA |
| :--- | :--- | :--- |
| 2K | 0.8 ms | 1.2 ms（微劣势，因显式求分母） |
| 16K | 12 ms（OOM 风险） | **3.5 ms** |
| 100K | 无法运行 | **18 ms** |

### 2.5 后续扩展建议

- **分块分母近似**：若显式构建 L×L 的 `attn_scores` 占用显存过大，可采用"分段累加"策略：将 Q/K 切成 Chunk，块内计算局部分母，再累加分子。显存从 O(L²) 降为 O(L·chunk_size)，但牺牲极少数全局精度。
- **搭配滑动窗口**：若 L 极端长（大于 500K），可先用本算法做"粗筛（全局）"，再搭配局部滑动窗口做"细筛（局部）"，构成混合专家层。

---

## 3. RAPT：引擎化扫描原语（GPU 落地）

RAPT（ReLULinearAttention）是项目内对上述 ReLU 线性注意力的命名与工程实现。其全链路中唯一不在 GPU 上的部分是 Layer 内的 `scan_forward_` / `scan_backward_`（纯 CPU 标量循环 + PCIe staging）。2026-09-04 完成的**引擎化**改造删除这些标量循环，改为 3 个 op-level 扫描原语，并由 Layer 用原语 + 逐元素原语组合表达算法（铁律 3：shader 永不含算法）。

### 3.1 三个扫描原语（`ComputeEngine`，`compute_engine.hpp`）

`compute_layer_rapt.hpp` 的扫描计算全面引擎化：RLA 算法（L2 归一分母 / ReLU 门控 / 梯度公式 / 文档重置策略）全部由 Layer 用原语 + 逐元素原语组合表达。

| 原语 | 语义 | 输出 |
|---|---|---|
| `scan_prefix_outer(K,V,P,R,A0,B0,has_state,dk,heads,causal,boundary,has_bnd)` | causal=true：`A_t = A0 + Σ_{i≤t, 同文档} k_i·k_iᵀ`、`B_t = B0 + Σ v_i·k_iᵀ`（文档边界处运行态清零，A0/B0 仅首文档生效）；causal=false：全集常数 | `(B·H·5·dk, seq)` 行块：[0) B·P  [1) A·P  [2) Bᵀ·R  [3) s  [4) r |
| `scan_suffix_outer(D,X,Y,dk,heads,causal,boundary,has_bnd)` | causal=true：`S_i = Σ_{t≥i, 同文档} D_t`（i+1 为文档起点时先清零）；causal=false：`S_i = D_i`（Layer 已把全集梯度沿 seq 广播） | `(B·H·3·dk, seq)`：[0) S·X  [1) S·Y  [2) Sᵀ·Y |
| `outer_col(P,R,S,dk,has_scale)` | 逐列外积 `out = P·Rᵀ`（has_scale 时逐列乘 S[t]） | `(B·H·dk², seq)` |

**形状约定**：batch-major（`i = b*seq+t`）；头 `(b,h)` 行块起点 `r0 = (b*H+h)*dk`；K/V/P/R（X/Y）为 `(B·H·dk, seq)`，D 为 `(B·H·dk², seq)`；A0/B0 为 `(H·dk, dk)`（B>1 按头循环）；boundary 为 `(1, B·seq)`（1 表示文档起点）；空参数用 **(1,1) dummy + bool 标志**（规避 0 字节 GPU buffer）；**dk ≤ 64**（GPU MAX_DK）；标量块 s/r 在头块内 dk 行重复存放（实现写全部行，Layer 读任一行均可）。

### 3.2 原语接口签名（`compute_engine.hpp` 纯虚）

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

### 3.3 Layer 组合（算法在 Layer）

- **forward**：`scan_prefix_outer(Kp, V, Qp, V, ·, ·, causal_, bnd, ·)` → slice 取 [0)BP、[3)s → `out = BP / sqrt(s + 1e-6)`（elementwise 原语链）→ rearrange → w_o。
- **backward pass 1**：`scan_prefix_outer(Kp, V, Qp, gcr, ·, ·, causal_, bnd, ·)` → slice 取 AP/BTR/s/r → elementwise 链（denom/inv/gnum/ddenom→ds、`t2 = 2·ds·Aq`）→ `gQt = BTR·inv + t2`；`outer_col` 物化 `dA = ds·q·qᵀ`（has_scale=true，S=ds）、`dB = gnum·q`（has_scale=false）。
- **backward pass 2**：`scan_suffix_outer(dA, Kp, V, ·, causal=true, bnd)`、`scan_suffix_outer(dB, ·)` → `gK = 2·(SA·K) + SBᵀ·V`、`gV = SB·K`。**双向分支**：`row_reduce_sum(dA/dB) → (BH·dk², 1)` → `broadcast_row_inplace` 沿 seq 广播 → `scan_suffix_outer(causal=false, dummy, false)`（shader 的 causal=0 分支即 S_i=D_i，D 已是全集梯度，等价全集后缀）。
- **forward_step（推理逐 token）**：Q/K/V → RoPE → ReLU → `scan_prefix_outer(K1, V1, Q1, Q1, A_state, B_state, true, dk, H, causal, ·, false)`（**先扫描**，A0=旧状态）→ `batched_matmul`（k·kᵀ、v·kᵀ）→ `add_inplace` 更新状态（**后更新**）。

### 3.4 引擎化中暴露的坑（改这段代码前必读）

1. **forward_step 顺序 = 先扫描、后更新状态**：`causal=true, has_state=true` 的扫描语义含自身（A_t 含 k_t·k_tᵀ）；若 A0 已含当前 token 则双算。与旧 CPU 实现"先累积再用"等价的前提是 A0 为旧状态。
2. **双向分支用 `row_reduce_sum`，不是 `col_reduce_sum`**：col 版返回 (1,C)；需要 (R,1)=`(BH·dk²,1)` 直接作 `broadcast_row_inplace` 的 row_vec（无需 transpose）。
3. **CPU 参考实现 boundary 形状校验曾潜伏 bug**：写成 `(rows/dk)*seq`（=B·H·seq），契约是 `(1, B·seq)`（shader 索引 `Bnd[b*seq+t]` 亦证）。旧 Layer 从不走引擎路径所以从未暴露；Layer 引擎化后 doc-aware gradcheck 直接踩中，已修为 `(rows/(dk*heads))*seq`，并给 vk backend 三个 `*_gpu` 方法补同义校验。
4. **标量块 s/r 头内逐行重复**（shader 写全部行），Layer 读任一行即可。

### 3.5 验证基线（2026-09-04，GTX 850M）

| 验证 | 结果 |
|---|---|
| `rapt_gradcheck`（CPU） | 14/14，max_err 与改造前基线逐位一致（causal 0.0290 / bidir kink 0.2554 / doc-aware 0.00868） |
| `rapt_gradcheck --gpu` | 3 段全 OK，kink 值 0.102/0.252 符合预期，全部 ≪ tol 5e-2；batch>1 覆盖（铁律 5）由 causal / doc-aware 段保证（段内 batch=2 固定，无 --batch 参数） |
| `rapt_smoke_test`（CPU） | KV-cache 一致性 max_diff=0（逐位一致） |
| `rapt_smoke_test --gpu` | KV-cache 一致性 max_diff=2.98e-08（= float32 1 ulp：GPU 加法顺序与 CPU 不同，数值正确） |
| `text_train --model rapt --gpu` | 106KB 语料 / 528 词表 / 319 步 / 1 epoch：loss 6.3→3.9，10.3s，无 TDR |
| `text_infer --model rapt --gpu` | 16 token 生成 0.4s（39 tok/s）；forward_step GPU 录制路径（batched_matmul 状态更新 + has_state 扫描同 command buffer）验证通过 |
| ctest 全量 | 31/31 全绿；AOT 收集仍 56 条融合表达式（手写原语不进融合注册表，闭合世界未破坏） |

> 状态注记：RLA 引擎化改造的 6 处 Vulkan 接线（backend include / pipeline 成员 / `get_*_spirv()` / `initialize()` 管线创建 / `has_*_pipeline()` / `*_gpu` 方法）、GpuEngine 占位替换、CMake `GPU_SHADERS` 追加等实现细节，连同 shader 完整源码与 handoff 步骤指令，沉淀在 `docs/development/01-compute-engine-development.md` 与原 `rapt-gpu-handoff.md`（已并入本篇）中；本合篇保留算法与约定，不再重复底层接线清单。

---

## 4. 两趟式等价 FlashAttention 的实现分析

本项目**没有传统的单一 Flash Attention kernel**，而是通过 **AOT 算子融合 + 两趟式（Two-Pass）注意力** 实现等效功能。注意力的 Flash Attention 语义分布在**多个独立融合 kernel** 中，按 S7 两趟式路径（`two_pass_active_ = true`）组合执行。

### 4.1 两趟式注意力（Two-Pass Attention）

```cpp
// compute_layer_attention.hpp — AttentionBase::forward() 两趟式路径
//
// Pass 1: m = row_max(masked(Q·K^T))          → (BH*seq, 1)
// Pass 2: l = row_sum(exp(masked(Q·K^T) - m)) → (BH*seq, 1)
// Pass 3: W = exp(masked(Q·K^T) - m) / l      → (BH*seq, seq)  ← 物化，backward 复用
// 输出:   O = W · V_t                          → (BH*seq, d_k)
```

对应 DSL 表达式（代码中 `compute_m` / `compute_l` / `compute_W` lambda）：

| 阶段 | DSL 表达式 | reduce_axis | 类型 |
|------|-----------|-------------|------|
| **m** | `compute_reduce(row_reduce_max(masked(matmul(Q,K))))` | row | 归约 kernel |
| **l** | `compute_reduce(row_reduce_sum(exp(masked(matmul(Q,K)) - bcast(m))))` | row | 归约 kernel |
| **W** | `compute(exp(masked(matmul(Q,K)) - bcast(m)) / bcast(l))` | elem | matmul+逐元素 kernel |

其中 `masked(...)` 根据掩码组合有 4 种变体：`masked_causal_`（纯因果）、`masked_alibi_`（因果 + ALiBi 线性偏置）、`masked_doc_`（因果 + 文档感知）、`masked_alibi_doc_`（因果 + ALiBi + 文档感知）。

### 4.2 Forward 融合算子清单

（每个 kernel 均据掩码组合派生多个变体，key 为 AOT 融合表达式的哈希，GLSL 文件名随 key 命名。）

**1. m kernel — `row_max(masked(matmul(Q, K^T)))`**，掩码变体 key：
- 纯因果 `dc022e892eacaa7f`；因果 + ALiBi `362af0a9b767519e`；因果 + doc_mask `9a3749c5fa6b1214`；因果 + ALiBi + doc_mask `81b741320ec17f2b`

核心逻辑（纯因果）：
```glsl
float mm = 0.0;  // matmul(Q, K^T) 逐元素
for (uint kk = 0u; kk < mm_k; ++kk)
    mm += b0[(batch*mm_k+kk)*m_per + row] * b1[(batch*mm_k+kk)*cols + col];
// causal mask: select(col > row, -inf, 0)
r0 = (float(col) > float(row)) ? 1.0 : 0.0;
r1 = (r0 != 0.0) ? c0 : c1;  // c0=-inf, c1=0
r0 = mm + r1;
acc = max(acc, r0);           // row_max 归约
```

**2. l kernel — `row_sum(exp(masked(matmul(Q, K^T)) - m))`**，掩码变体 key：
- 纯因果 `b79495b9a28cd730`；因果 + ALiBi `961fdd06c61acf14`；因果 + doc_mask `ceff1358db53ee3d`；因果 + ALiBi + doc_mask `4fc623847f4a0b2f`

核心逻辑（因果 + ALiBi）：
```glsl
r1 = float(col) - float(row);     // 距离 (col - row)
r2 = b2[batch % 2u] * r1;         // ALiBi slope × 距离
r1 = (r0 != 0.0) ? c2 : r2;      // 未来位置=-inf, 否则=ALiBi偏置
r0 = mm + r1;                     // scores + mask + ALiBi
r1 = r0 - b3[row];                // scores - m (减去行max)
r0 = exp(r1);                     // online softmax: exp(scores - m)
acc = (acc + r0);                 // row_sum 归约
```

**3. W kernel — `exp(masked(matmul(Q, K^T)) - m) / l`**，掩码变体 key：
- 纯因果 `bb2c45390b92ce9c`；因果 + ALiBi `08df5c62aa0b3e6a`；因果 + doc_mask `72ac283f65e09fb9`；因果 + ALiBi + doc_mask `a9fc1f78c9878b60`

核心逻辑（`eval_tail` 函数，因果 + ALiBi，使用 matmul 分块 16×16 thread / 64×64 block 的 tiled 架构）：
```glsl
float eval_tail(float mm, uint row, uint col, uint batch) {
    const float slope = b2[batch % 2u];     // ALiBi slope
    const float m_val = b3[row];            // 行 max
    const float l_val = b4[row];            // 行 sum
    // causal mask + ALiBi bias
    r0 = (float(col) > float(row)) ? 1.0 : 0.0;
    r1 = (r0 != 0.0) ? c0 : c1;            // -inf or 0
    r0 = (r1 != c1) ? 1.0 : 0.0;           // mask flag
    r1 = float(col) - float(row);
    r2 = slope * r1;                         // ALiBi
    r1 = (r0 != 0.0) ? c2 : r2;            // -inf or ALiBi
    r0 = mm + r1;                            // masked scores
    r1 = r0 - m_val;                         // stability subtract
    r0 = exp(r1);                            // exp(scores - m)
    r1 = r0 / l_val;                         // normalize → softmax weights W
    return r1;
}
```

### 4.3 Backward 融合算子

Backward 使用缓存的 W 矩阵和 DSL 表达式：

| 阶段 | DSL 表达式 | 类型 |
|------|-----------|------|
| `W × P` (elementwise) | `matmul(W, P)` (纯matmul) | matmul 无尾部 |
| `R = row_sum(W · P)` | `matmul(W, P)` + `row_reduce_sum` | 独立归约 kernel |
| `X = W · (P - R)` | `W * (P - R)` | matmul+逐元素 |

此外，backward 中还有 `max(matmul + W_cache, c0)` 形式（`c0 = -inf`，实质 `max(x, -inf) = x`，用于掩码兼容），等价于 `matmul(W_cache, P)`。

### 4.4 与传统 Flash Attention 的对比

| 特性 | 传统 Flash Attention (Dao et al.) | 本项目 (S7 Two-Pass) |
|------|----------------------------------|---------------------|
| **Kernel 数量** | 1 个 forward + 1 个 backward | 3+ 个 forward + 2+ 个 backward |
| **Online Softmax** | 在单 kernel 内完成 m/l 更新 | 分 3 个独立 kernel (m → l → W) |
| **避免物化 S 矩阵** | 完全避免 | m/l 不物化；W 物化供 backward |
| **IO 复杂度** | O(N²d/M) (M = SRAM 大小) | 类似，但多 kernel 间有中间写入 |
| **掩码支持** | 需要专用 kernel 变体 | 通过 DSL 表达式 + AOT 融合自动派生 |
| **融合粒度** | 手写 CUDA block/warp 逻辑 | AOT 编译期自动融合 IR → SPIR-V |

**总结**：本项目不存在单一 "flash-attn" kernel。Flash Attention 的核心思想（online softmax + 避免物化完整注意力矩阵）通过 **S7 两趟式 DSL 融合**实现：forward 3 个融合 kernel（m/l/W）乘 4 种掩码变体 = **12 个 forward shader**；backward 对应的 W×P / R / X 融合 kernel 乘掩码变体 ≈ **9 个 backward shader**。每个 kernel 都是在 `compute_layer_attention.hpp` 的 DSL 表达式基础上，经 `scan_exprs → gen_fused` 管线自动生成的 GLSL compute shader。

---

## 5. RLA-2：极简硬截断线性注意力（修正版）

**版本**：2.1　**日期**：2026-09-04
**核心哲学**：无门控、无学习衰减、仅依赖 ReLU 硬截断和 RoPE 位置信息，实现全局动态稀疏检索。

### 5.1 摘要

RLA-2 是对原始 RLA 的深度优化版本。它在保留原版 O(Ld²) 训练复杂度、O(d²) 每 token 推理复杂度的前提下，通过三项关键改进解决原版在数值稳定性、位置感知和神经元死亡方面的核心缺陷：

1. **归一化分母从 L2 改为 Sum** → 输出量级恒定，长度发散问题根治。
2. **移除强制指数衰减（γ）** → 完全依赖 RoPE 提供的位置信息，消除超参数，增强硬检索能力。
3. **RMSNorm 替代残差 Shortcut** → 保持 ReLU 硬截断纯粹性，稳定训练，降低神经元死亡。

最终算法极为简洁：**投影 → RMSNorm → RoPE → ReLU → 加权平均（Sum 归一化）**。它本质上是一个带位置感知的核方法，可解释性强，在超长上下文（大于 100K）任务中效率优势显著。

本修正版重点解决的一致性问题：① 统一推理与训练中的注意力分数定义（逐维 ReLU 后点积，而非对点积结果 ReLU）；② 补充因果训练的正确实现（分块状态传递）；③ 修正对 RoPE 远程衰减和 RMSNorm 性质的表述。

### 5.2 算法数学定义

**注意力分数**：对查询位置 t 与历史位置 i，先计算：

```
q_t' = ReLU( RoPE( RMSNorm(W_q x_t) ) )
k_i' = ReLU( RoPE( RMSNorm(W_k x_i) ) )
v_i  = W_v x_i
```

注意力分数定义为**逐维 ReLU 后向量的点积**：

```
s_ti = q_t' · k_i' = Σ_{j=1..d} q_{t,j}' · k_{i,j}'
```

> 注意：不要写成 `ReLU(q_t · k_i)`，两种形式不等价。只有逐维 ReLU 后才能利用矩阵结合律实现线性复杂度。

**因果推理（逐 token 增量形式）**：维护两个固定大小的状态：

```
S_t = Σ_{i=1..t} k_i' ⊗ v_i   ∈ R^{d×d}
z_t = Σ_{i=1..t} k_i'         ∈ R^d
```

当前输出：

```
Output_t = q_t' S_t / ( q_t' · z_t + ε )
```

状态更新规则：

```
S_t = S_{t-1} + k_t' ⊗ v_t
z_t = z_{t-1} + k_t'
```

每 token 推理复杂度为 O(d²)，与序列长度无关；初始状态 `S_0 = 0, z_0 = 0`；ε 取 1e-4 防止除零。

**因果训练（分块并行形式）**：训练时不能直接使用非因果的全矩阵形式 `Q'K'ᵀV`（会泄露未来信息），必须采用**分块因果线性注意力**。将长度为 L 的序列划分为 B 个块，每块大小 C（通常 64~256）；对每个块 b 维护来自之前所有块的累积状态 `S_prev`/`z_prev`，块内计算用标准线性注意力，同时将当前块的键值对累加入状态传给下一块。伪代码：

```python
def causal_linear_attention(Q, K, V, chunk_size):
    # Q, K, V: (B, L, d)，已经过 ReLU 和 RoPE
    B, L, d = Q.shape
    S = torch.zeros(B, d, d, device=Q.device)
    z = torch.zeros(B, d, device=Q.device)
    outputs = []
    for i in range(0, L, chunk_size):
        Q_c = Q[:, i:i+chunk_size]      # (B, C, d)
        K_c = K[:, i:i+chunk_size]
        V_c = V[:, i:i+chunk_size]

        # 当前块内部注意力（因果，但块内仍可并行）
        num_inner = torch.matmul(Q_c, torch.matmul(K_c.transpose(-2,-1), V_c))
        den_inner = torch.matmul(Q_c, K_c.sum(dim=1, keepdim=True))  # (B,C,1)

        # 与历史状态的交互
        num_hist = torch.matmul(Q_c, S)      # (B,C,d)
        den_hist = torch.matmul(Q_c, z.unsqueeze(-1))  # (B,C,1)

        output = (num_inner + num_hist) / (den_inner + den_hist + eps)
        outputs.append(output)

        # 更新状态
        S = S + torch.matmul(K_c.transpose(-2,-1), V_c)
        z = z + K_c.sum(dim=1)

    return torch.cat(outputs, dim=1)
```

复杂度 O(Ld²)，空间 O(Ld + Bd²)。块内计算仍为线性注意力，但块大小有限，可高效并行。

### 5.3 与原版 RLA 的关键差异

| 特性 | 原版 RLA | RLA-2 |
| :--- | :--- | :--- |
| 归一化分母 | `sqrt(Σ score²) + ε` | `Σ score + ε` |
| 分数定义 | 不一致（推理为 ReLU(点积)，训练为 ReLU(向量)点积） | **统一为逐维 ReLU 后点积** |
| 位置编码 | 必须 RoPE | 同左（严格强制） |
| 距离衰减 | 可加 γ^(t-i)（未强制） | **无**（仅依靠 RoPE 提供的位置信息） |
| 防神经元死亡 | 残差 shortcut `ReLU(x)+0.1*x` | RMSNorm（前置） |
| 额外超参数 | γ（若使用） | **零** |
| 硬截断纯粹性 | 被残差泄漏破坏 | **完全保留**（负数精确归零） |
| 输出量级 | 随 √L 增长 | **恒定**（加权平均） |
| 推理复杂度 | 每 token O(td)（若直接求和） | **每 token O(d²)**（状态递推） |

### 5.4 设计哲学与理论优势

- **为何抛弃 L2 归一化**：L2 等效于对注意力分数做余弦相似度约束，但分母为 sqrt(Σscore²)，输出幅度随 L 增长，长上下文数值易越界。Sum 归一化使输出变为 V 的**加权平均**，幅值完全稳定，梯度流更平滑。
- **为何无需强制衰减**：原 softmax 中 RoPE 通过内积相位差自然产生远程衰减；但在本模型中对 RoPE 输出做了逐维 ReLU，点积不再等于旋转前内积，因此**不保证严格单调远程衰减**。但 RoPE 仍为不同位置提供不同旋转角度，可使语义相近的 q、k 因位置差异导致部分维度符号翻转、改变 ReLU 激活模式，从而学习**位置相关的稀疏性**。强制 γ 衰减会弱化硬检索能力：若关键信息在开头，过远距离会使分数被压近零而被 ReLU 截断。实际实验中 RoPE+ReLU 仍能有效区分远近。
- **为何用 RMSNorm 而非残差 Shortcut**：残差 `ReLU(x)+0.1*x` 引入负值泄漏，违背 ReLU"负数彻底屏蔽"语义。RMSNorm 在 ReLU 之前将 Q/K 均方根归一化到 1，稳定数值分布、减少梯度消失/爆炸、降低死亡神经元比例，同时保持硬截断纯粹性。RMSNorm 默认不引入可学习参数，符合极简原则。
- **可解释性**：每个历史 token 权重为 `s_ti = q_t'·k_i'`，是两个非负向量的点积，仅当对应维度同时为正值时才有贡献；不经过 softmax 指数挤压，保持线性可解释性，可逐维分析激活模式。

### 5.5 潜在问题与解决方案

| 问题 | 方案 |
| :--- | :--- |
| **初期收敛慢**（无强位置先验） | 学习率预热（前 10% 步数线性增长）+ **AdamW** 优化器。 |
| **分母过小导致数值爆炸** | eps 设为 1e-4 或使用 `torch.clamp(denominator, min=eps)`。 |
| **多头之间缺乏多样性** | 每头增加独立可学习 RMSNorm 增益（极少参数），或使用不同初始化尺度。 |
| **处理极长序列（>500K）的显存** | 训练用更小块大小；推理状态固定 d×d，无额外显存增长。 |
| **与绝对位置编码的冲突** | 严格禁止绝对位置编码，仅用 RoPE。 |
| **RoPE+ReLU 不保证远程衰减** | 若需严格距离衰减，可在分数上加小的可学习位置偏置（会破坏纯线性形式，需权衡）。 |

### 5.6 与主流架构定位对比与训练建议

各架构复杂度对比：

| 架构 | 训练复杂度 | 推理复杂度/步 | 硬检索 | 可解释性 | 超参数数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RLA-2** | O(Ld²) | O(d²) | **极佳** | **高** | **零** |
| Gated DeltaNet | O(Ld²) | O(d²) | 良好 | 低 | 门控参数 |
| Mamba-2 | O(Ld²) | O(d²) | 良好 | 低 | 多个 |
| 稀疏注意力 (NSA) | O(L log L) | O(log L) | 极佳 | 中等 | 稀疏模式 |
| Flash Attention | O(L²d) | O(Ld) | 极佳 | 高 | 无 |

RLA-2 最适合**超长上下文、硬检索密集、对可解释性有要求**的场景（如法律文档分析、代码库检索、长视频字幕理解）。

训练建议：Q/K 投影权重采用标准差 0.02 的正态分布；AMP（FP16/BF16）混合精度；梯度全局范数裁剪至 1.0；学习率典型值 1e-3（AdamW）配合余弦退火；块大小 128~256。

---

## 6. LRLA：可学习 ReLU 线性注意力（RLA-2 的扩展）

**版本**：1.0　**日期**：2026-09-04
**核心哲学**：保留 ReLU 硬截断的稀疏与可解释性，通过引入可学习的多层特征映射提升表达能力，同时严格维持线性注意力的高效复杂度。

### 6.1 摘要与算法定义

RLA-2 的特征映射被固定为单层线性变换后接 ReLU，对复杂 token 交互的建模能力有限。LRLA（Learnable ReLU Linear Attention）将这一映射替换为**可学习的多层非线性网络**，最后一层仍为 ReLU 以保证分数非负。

定义两个可学习非线性映射 `φ_q, φ_k : R^d → R^{d_φ}`（通常 d_φ = d）：

```
φ(x) = ReLU( W_L · σ_{L-1}( W_{L-1} · … σ_1(W_1 x) ) )
```

其中中间层 σ_i 可为 GELU、SiLU 等平滑激活，**最后一层必须为 ReLU（或 Softplus 等非负激活）**。注意力分数定义为特征向量点积：

```
s_ti = q_t^φ · k_i^φ = Σ_{j=1..d_φ} q_{t,j}^φ k_{i,j}^φ  ≥ 0
```

其中 `q_t^φ = φ_q( RoPE(W_q x_t) )`、`k_i^φ = φ_k( RoPE(W_k x_i) )`、`v_i = W_v x_i`。

因果推理状态与更新同 RLA-2（`S_t ∈ R^{d_φ×d}`、`z_t ∈ R^{d_φ}`），每 token 复杂度 O(d_φ·d)。因果训练同样采用分块并行 `causal_linear_attention`，块合并公式与更新与 RLA-2 一致。

### 6.2 复杂度与定位

- 训练：O(Ld(d+d_φ))（d_φ=d 时为 O(Ld²)），空间 O(Ld + Bd_φd)；
- 推理（单步）：O(d_φ·d)，恒定，不随上下文增长；状态矩阵 S/z 与序列长度无关。

| 架构 | 训练复杂度 | 推理/步 | 表达能力 | 硬检索 | 可解释性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LRLA** | O(Ld²) | O(d²) | 中高（可学习 MLP） | **极佳** | 高 |
| RLA-2 | O(Ld²) | O(d²) | 中 | **极佳** | **极高** |
| Gated DeltaNet | O(Ld²) | O(d²) | 高 | 良好 | 低 |
| Mamba-2 | O(Ld²) | O(d²) | 高 | 良好 | 低 |
| Flash Attention | O(L²d) | O(Ld) | **极高** | **极佳** | 高 |

### 6.3 关键设计要点

- **位置信息处理**：RoPE 必须施加在进入多层网络**之前**（多层非线性会破坏旋转相对位置性质；MLP 会将 RoPE 位置信息作为输入特征进一步加工，学习位置与语义的复杂交互）。
- **保持稀疏性**：中间层可用 GELU 等平滑激活，但最后一层必须 ReLU 保证非负输出（非负避免分母出现负数、数值更稳定）。
- **与 softmax 关系**：标准 softmax 用指数核 exp(q·k/√d)，理论上具无限维特征映射；线性注意力用有限维特征映射近似，LRLA 通过可学习网络尽量逼近最优映射，可视为对 softmax 核的有参近似。

**位置解码决定性**（大量零分数使注意力稀疏；非负分数可逐维分析哪个特征维度贡献注意力；非负性避免分母负数）。

### 6.4 潜在问题与训练建议

常见问题：参数增多过拟合（dropout / 权重衰减 / 共享 Q/K 底层参数）；计算开销（保持隐藏维度与模型维度相同或用低秩分解 MLP）；训练不稳定（MLP 内加 LayerNorm/RMSNorm + 残差 + 适度降低学习率）；ReLU 死亡神经元（中间层 GELU/SiLU + 较小初始化标准差）；位置信息被 MLP 稀释（可加残差 `φ(x)=ReLU(x+MLP(x))` 保留原始 RoPE 信息）；d_φ 选择（通常与模型维度一致，可设 2d 但注意状态矩阵增大）；分母过小（eps + clamp）。

训练建议：Q/K/V 投影标准差 0.02；MLP 权重 Xavier/He 初始化；AdamW lr 1e-3 或稍低 + 线性预热 + 余弦退火；BF16/FP16（注意 LayerNorm/MLP 数值稳定性）；块大小 128~256；MLP dropout 0.1、权重衰减 1e-4；监控 ReLU 输出稀疏比例，若超 90% 为零则降低学习率或调整初始化。

---

## 7. 演进路线小结

本家族从 RLA（L2 归一分母 + 残差泄漏的"贪心但工程细节多"版本）起步，经 RLA-2（Sum 归一化 + 无衰减 + RMSNorm 的"极简修正版"）与 LRLA（可学习多层映射的"扩容版"），构成一条"在不牺牲线性复杂度与硬截断稀疏性前提下，逐步收敛表达与稳定训练"的路线图：

- **数值稳定**：L2 分母 → Sum 分母（输出量级恒定）。
- **位置感知**：强制 RoPE/ALiBi，移除 γ 衰减，彻底摆脱绝对位置编码。
- **防神经元死亡**：残差 Shortcut → RMSNorm →（LRLA 中为 MLP 内部归一化）。
- **工程层**：RAPT 引擎化给出 GPU 落地范式（3 个扫描原语 + Layer 组合 + shader 纯数据流）。

在项目实现中以 RAPT（结合 §3 的引擎化扫描原语与 §4 的两趟式 AOT 融合管线）作为当前工程基线。