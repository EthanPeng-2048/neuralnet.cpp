# Flash Attention 融合算子分析报告

## 概述

本项目**没有传统的 Flash Attention 单一 kernel**，而是通过 **AOT 算子融合 + 两趟式 (Two-Pass) 注意力** 实现了等效功能。

`build/generated/` 中的融合 shader 是由 `AttentionBase::forward()` 中的 DSL 表达式
经 `scan_exprs` 收集 + `gen_fused` 合成的。注意力的 Flash Attention 语义分布在
**多个独立融合 kernel** 中，按 S7 两趟式路径（`two_pass_active_ = true`）组合执行。

## 架构：两趟式注意力 (Two-Pass Attention)

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

其中 `masked(...)` 根据掩码组合有 4 种变体：
- `masked_causal_` — 纯因果掩码
- `masked_alibi_` — 因果 + ALiBi 线性偏置
- `masked_doc_` — 因果 + 文档感知
- `masked_alibi_doc_` — 因果 + ALiBi + 文档感知

## Forward Pass 融合算子清单

### 1. m kernel — `row_max(masked(matmul(Q, K^T)))`

| key | 掩码变体 | GLSL 文件 |
|-----|---------|-----------|
| `dc022e892eacaa7f` | 纯因果 | `fused_dc022e892eacaa7f.comp` |
| `362af0a9b767519e` | 因果 + ALiBi | `fused_362af0a9b767519e.comp` |
| `9a3749c5fa6b1214` | 因果 + doc_mask | `fused_9a3749c5fa6b1214.comp` |
| `81b741320ec17f2b` | 因果 + ALiBi + doc_mask | `fused_81b741320ec17f2b.comp` |

**核心逻辑**（以纯因果为例）：
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

### 2. l kernel — `row_sum(exp(masked(matmul(Q, K^T)) - m))`

| key | 掩码变体 | GLSL 文件 |
|-----|---------|-----------|
| `b79495b9a28cd730` | 纯因果 | `fused_b79495b9a28cd730.comp` |
| `961fdd06c61acf14` | 因果 + ALiBi | `fused_961fdd06c61acf14.comp` |
| `ceff1358db53ee3d` | 因果 + doc_mask | `fused_ceff1358db53ee3d.comp` |
| `4fc623847f4a0b2f` | 因果 + ALiBi + doc_mask | `fused_4fc623847f4a0b2f.comp` |

**核心逻辑**（以因果 + ALiBi 为例）：
```glsl
// ... 同上计算 masked score ...
r1 = float(col) - float(row);     // 距离 (col - row)
r2 = b2[batch % 2u] * r1;         // ALiBi slope × 距离
r1 = (r0 != 0.0) ? c2 : r2;      // 未来位置=-inf, 否则=ALiBi偏置
r0 = mm + r1;                     // scores + mask + ALiBi
r1 = r0 - b3[row];                // scores - m (减去行max)
r0 = exp(r1);                     // online softmax: exp(scores - m)
acc = (acc + r0);                 // row_sum 归约
```

### 3. W kernel — `exp(masked(matmul(Q, K^T)) - m) / l`

| key | 掩码变体 | GLSL 文件 |
|-----|---------|-----------|
| `bb2c45390b92ce9c` | 纯因果 | `fused_bb2c45390b92ce9c.comp` |
| `08df5c62aa0b3e6a` | 因果 + ALiBi | `fused_08df5c62aa0b3e6a.comp` |
| `72ac283f65e09fb9` | 因果 + doc_mask | `fused_72ac283f65e09fb9.comp` |
| `a9fc1f78c9878b60` | 因果 + ALiBi + doc_mask | `fused_a9fc1f78c9878b60.comp` |

**核心逻辑**（`eval_tail` 函数，以因果 + ALiBi 为例）：
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
此 kernel 使用 **matmul 分块 (16×16 thread, 64×64 block)** 的 tiled 架构。

## Backward Pass 融合算子

Backward 使用缓存的 W 矩阵和 DSL 表达式：

| 阶段 | DSL 表达式 | key | 类型 |
|------|-----------|-----|------|
| `W × P` (elementwise) | `matmul(W, P)` (纯matmul) | `5127e7efc376cbbc` | matmul 无尾部 |
| `R = row_sum(W · P)` | `matmul(W, P)` + `row_reduce_sum` | 独立归约 kernel | 归约 |
| `X = W · (P - R)` | `W * (P - R)` | matmul+elem | matmul+逐元素 |

**`7f1d283eb082815b`** / **`7f1d273eb0827fa8`** — backward 中 `max(matmul + W_cache, c0)` 形式，
其中 `c0 = -inf`（实质 `max(x, -inf) = x`，用于掩码兼容），等价于 `matmul(W_cache, P)`。

## 与传统 Flash Attention 的对比

| 特性 | 传统 Flash Attention (Dao et al.) | 本项目 (S7 Two-Pass) |
|------|----------------------------------|---------------------|
| **Kernel 数量** | 1 个 forward + 1 个 backward | 3+ 个 forward + 2+ 个 backward |
| **Online Softmax** | ✅ 在单 kernel 内完成 m/l 更新 | ✅ 分 3 个独立 kernel (m → l → W) |
| **避免物化 S 矩阵** | ✅ 完全避免 | ✅ m/l 不物化；W 物化供 backward |
| **IO 复杂度** | O(N²d/M) (M = SRAM 大小) | 类似，但多 kernel 间有中间写入 |
| **掩码支持** | 需要专用 kernel 变体 | 通过 DSL 表达式 + AOT 融合自动派生 |
| **融合粒度** | 手写 CUDA block/warp 逻辑 | AOT 编译期自动融合 IR → SPIR-V |

## 总结

**`build/generated/` 中不存在单一的 "flash-attn" kernel**。Flash Attention 的核心思想
（online softmax + 避免物化完整注意力矩阵）通过 **S7 两趟式 DSL 融合** 实现：

- **Forward**: 3 个融合 kernel (m/l/W) × 4 种掩码变体 = **12 个 forward shader**
- **Backward**: 对应的 W×P / R / X 融合 kernel × 掩码变体 = **~9 个 backward shader**

每个 kernel 都是在 `compute_layer_attention.hpp` 的 DSL 表达式基础上，
经 `scan_exprs` → `gen_fused` 管线自动生成的 GLSL compute shader。
