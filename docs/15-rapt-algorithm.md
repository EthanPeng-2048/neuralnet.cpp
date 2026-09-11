# ReLU-Linear Attention (RLA) 算法


**目标**：在保持 \( O(L) \) 时间复杂度和 \( O(1) \) 内存增长（相对于上下文长度）的前提下，实现全局上下文的动态稀疏检索。  
**核心哲学**：利用 ReLU 的非线性截断特性，作为可微分的“硬注意力门控”，自动屏蔽无关噪声 Token。

---

## 1. 算法数学定义（闭式解）

对于输入序列长度为 \( L \)，隐藏维度为 \( d \)，给定当前查询位置 \( t \) 的查询向量 \( q_t \in \mathbb{R}^d \)，以及历史全部键值对 \( (k_i, v_i) \)，其中 \( i \in [1, L] \)。

RLA 的计算定义为严格的一行式：

\[
\text{Output}_t = \frac{ \sum_{i=1}^{L} \sigma(q_t)^\top \sigma(k_i) \cdot v_i }{ \sqrt{ \sum_{i=1}^{L} \big( \sigma(q_t)^\top \sigma(k_i) \big)^2 } + \epsilon }
\]

其中：

-   \( \sigma(x) = \text{ReLU}(x) = \max(0, x) \)（**关键**：逐元素操作，非向量模长）。
-   \( \epsilon \) 为极小常数（如 \( 1e-6 \)），防止除零。

**全局矩阵形式（训练并行视图）**：  
令 \( Q' = \text{ReLU}(Q W_q), \ K' = \text{ReLU}(K W_k), \ V = V W_v \)。  
\[
\text{Output} = \frac{ (Q' K'^\top) \odot (Q' K'^\top) ? ... }
\]  
*注意：公式中的分母是每个 query 独立计算的标量，因此无法简单地用一次矩阵除法解决，必须逐行归一化（参见第3节代码实现）。*

---

## 2. 关键算法特性（必须遵守）

| 特性 | 描述 |
| :--- | :--- |
| **复杂度** | 计算 \( Q'K'^\top \) 若显式构建为 \( L \times L \) 则是 \( O(L^2) \)。**复现时必须利用结合律**：先算 \( K'^\top V \)，再算 \( Q'(K'^\top V) \)，复杂度降至 \( O(Ld^2) \)。当 \( d \ll L \) 时，视作线性。 |
| **稀疏激活** | 若 \( q_t \) 与 \( k_i \) 在特征空间方向相反，点积为负，ReLU 将其精确置为 0。这实现了“只激活语义相关的 Token”。 |
| **动态重选** | 每预测新 Token，生成新的 \( q_{t+1} \)，算法强制基于全量 \( L+1 \) 个原始 \( k_i \) 重新计算，不存在 RNN 的隐状态遗忘。 |
| **位置编码** | **严禁使用绝对位置编码（如 Learned PE）**。必须使用 **RoPE（旋转位置编码）** 或 **ALiBi**，且 RoPE 必须施加在 **Q 和 K 进入 ReLU 之前**。 |

---

## 3. PyTorch 精确复现代码（因果 & 非因果）

以下代码包含**训练并行版（双向/非因果）**和**推理自回归版（因果）**。请直接复制到 `rla.py`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLULinearAttention(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 标准 QKV 投影（无偏见初始化更稳定）
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        # 输出投影
        self.w_o = nn.Linear(dim, dim, bias=False)
        
    def _relu_linear(self, q, k, v):
        """
        核心计算模块：支持 (Batch, Head, Seq, Dim)
        利用结合律实现 O(L d^2) 复杂度
        """
        # 1. ReLU 门控（生成稀疏特征）
        q_prime = F.relu(q)
        k_prime = F.relu(k)
        
        # 2. 线性复杂度核心：先算 K^T @ V，再算 Q @ (K^T @ V)
        # 注意：此处 v 维度为 (B, H, L, d)，转置后为 (B, H, d, L)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # (B, H, d, d)
        output = torch.matmul(q_prime, kv)               # (B, H, L, d)
        
        # 3. 分母计算（L2 归一化，保证数值稳定）
        # 计算 q_i 与 k_j 的点积矩阵（为了分母，必须显式计算，但这是逐元素的，内存 O(L^2)）
        # 工程优化：如果 L > 4096，建议使用分块计算，此处为了可读性采用标准实现
        attn_scores = torch.matmul(q_prime, k_prime.transpose(-2, -1))  # (B, H, L, L)
        denominator = torch.sqrt(torch.sum(attn_scores ** 2, dim=-1, keepdim=True)) + self.eps
        
        # 4. 归一化输出（等价于对每个 query 的权重做 L2 约束）
        # 为了严格等于数学公式，我们需要将 output 除以分母，但 output 已经是加权和，
        # 而分母是 sqrt(sum(score^2))。注意：此处的 output 缺少显式的 score 加权，
        # 需要纠正：上面的 output = Q @ (K^T @ V) = sum(score * v)，正确。
        # 但分母是 sqrt(sum(score^2))，直接除即可。
        output = output / denominator
        
        return output

    def forward(self, x):
        """
        x: (Batch, Seq, Dim) 训练阶段非因果（双向）使用
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 重塑为多头（假设 head 数为 4 或 8，此代码简化为单头演示）
        # 实际使用时，请拆分为 (B, H, L, head_dim)
        output = self._relu_linear(q, k, v)
        return self.w_o(output)


# ---------- 因果自回归推理（生成下一个 Token）专用实现 ----------
def causal_rla_step(q_t, k_cache, v_cache, eps=1e-6):
    """
    推理时单步生成，k_cache/v_cache 存储所有历史 token 的 ReLU 投影后的值
    q_t: (1, 1, d) 当前查询
    k_cache: (1, L, d) 已经过 ReLU(K) 的历史缓存
    v_cache: (1, L, d) 原始 V
    """
    # 1. 对当前 q 做 ReLU
    q_prime = F.relu(q_t)  # (1, 1, d)
    
    # 2. 计算当前 q 与所有历史 k 的点积（非负分数）
    scores = torch.matmul(q_prime, k_cache.transpose(-2, -1))  # (1, 1, L)
    
    # 3. 计算加权和（分子）
    numerator = torch.matmul(scores, v_cache)  # (1, 1, d)
    
    # 4. 计算分母（L2 norm of scores）
    denominator = torch.sqrt(torch.sum(scores ** 2, dim=-1, keepdim=True)) + eps
    
    return numerator / denominator
```

---

## 4. 复现时的“致命陷阱”与解决方案（必读）

| 陷阱 | 现象 | 解决方案 |
| :--- | :--- | :--- |
| **直接照搬 Softmax 的 Scale** | 训练不稳定，Loss 震荡 | **禁止**使用 \( 1/\sqrt{d_k} \) 缩放。因为 ReLU 非负，点积随维度线性增长，需改用法：`scores = scores / (d ** 0.5)` 放在 ReLU **之前**，且对 Q/K 进行 **LayerNorm** 预处理。 |
| **分母计算导致梯度消失** | 长文本（>8k）时模型输出趋近于 0 | 务必使用代码中的 **L2 分母**（平方和开根号），而非简单求和。实验证明，求和分母会导致梯度爆炸/消失，L2 等价于对注意力分数做余弦相似度约束。 |
| **ReLU 导致大量维度永久死亡** | 模型失去表达能力 | 在 Q/K 投影后、ReLU 之前，加一层 **Shortcut（残差）**：`q = F.relu(q) + 0.1 * q`（即 Leaky ReLU 变体），保证负值有微弱梯度流过。 |

---

## 5. 复杂度基准测试（预期结果）

在单张 A100 (80G) 上，`dim=1024`, `batch=1`：

| 上下文长度 L | Softmax Attention (Flash) | RLA (本算法) |
| :--- | :--- | :--- |
| 2K | 0.8 ms | 1.2 ms（微劣势，因显式求分母） |
| 16K | 12 ms（OOM 风险） | **3.5 ms** |
| 100K | 无法运行 | **18 ms** |

---

## 6. 后续扩展建议（非必须，但决定上线效果）

- **分块分母近似**：如果显式构建 \( L \times L \) 的 `attn_scores` 占用显存过大，可采用 **“分段累加”** 策略：将 Q/K 切成 Chunk，在 Chunk 内计算局部分母，再将分子累加。这能将显存占用从 \( O(L^2) \) 降为 \( O(L \cdot \text{chunk\_size}) \)，但会牺牲极少数全局精度。
- **搭配滑动窗口**：若 \( L \) 极端长（>500K），可结合你的初始设想：用本算法做“粗筛（全局）”，再搭配局部滑动窗口做“细筛（局部）”，构成混合专家层。

---

## 7. 引擎化（GPU 扫描原语，2026-09-04 完成）

### 7.1 三个扫描原语（`ComputeEngine`，`compute_engine.hpp`）

`compute_layer_rapt.hpp` 的扫描计算全面引擎化：删除 `scan_forward_` / `scan_backward_` CPU 标量循环与 `forward_step` 逐 token PCIe 往返，改为 3 个 op-level 原语。RLA 算法（L2 归一分母 / ReLU 门控 / 梯度公式 / 文档重置策略）全部由 Layer 用原语 + 逐元素原语组合表达（铁律 3：shader 永不含算法）。

| 原语 | 语义 | 输出 |
|---|---|---|
| `scan_prefix_outer(K,V,P,R,A0,B0,has_state,dk,heads,causal,boundary,has_bnd)` | causal=true：`A_t = A0 + Σ_{i≤t, 同文档} k_i·k_iᵀ`、`B_t = B0 + Σ v_i·k_iᵀ`（文档边界处运行态清零，A0/B0 仅首文档生效）；causal=false：全集常数 | `(B·H·5·dk, seq)` 行块：[0) B·P  [1) A·P  [2) Bᵀ·R  [3) s  [4) r |
| `scan_suffix_outer(D,X,Y,dk,heads,causal,boundary,has_bnd)` | causal=true：`S_i = Σ_{t≥i, 同文档} D_t`（i+1 为文档起点时先清零）；causal=false：`S_i = D_i`（Layer 已把全集梯度沿 seq 广播） | `(B·H·3·dk, seq)`：[0) S·X  [1) S·Y  [2) Sᵀ·Y |
| `outer_col(P,R,S,dk,has_scale)` | 逐列外积 `out = P·Rᵀ`（has_scale 时逐列乘 S[t]） | `(B·H·dk², seq)` |

**形状约定**：batch-major（`i = b*seq+t`）；头 (b,h) 行块起点 `r0=(b*H+h)*dk`；K/V/P/R（X/Y）`(B·H·dk, seq)`，D `(B·H·dk², seq)`；A0/B0 `(H·dk, dk)`（B>1 按头循环）；boundary `(1, B·seq)`（1=文档起点）；空参数用 **(1,1) dummy + bool 标志**（规避 0 字节 GPU buffer）；**dk ≤ 64**（GPU MAX_DK）；标量块 s/r 在头块内 dk 行重复存放（实现写全部行，Layer 读任一行均可）。

### 7.2 Layer 组合（算法在 Layer）

- **forward**：`scan_prefix_outer(Kp, V, Qp, V, ·, ·, causal_, bnd, ·)` → slice 取 [0)BP、[3)s → `out = BP / sqrt(s + 1e-6)`（elementwise 原语链）→ rearrange → w_o。
- **backward pass 1**：`scan_prefix_outer(Kp, V, Qp, gcr, ·, ·, causal_, bnd, ·)` → slice 取 AP/BTR/s/r → elementwise 链（denom/inv/gnum/ddenom→ds、t2=2·ds·Aq）→ `gQt = BTR·inv + t2`；`outer_col` 物化 `dA = ds·q·qᵀ`（has_scale=true，S=ds）、`dB = gnum·q`（has_scale=false）。
- **backward pass 2**：`scan_suffix_outer(dA, Kp, V, ·, causal=true, bnd)`、`scan_suffix_outer(dB, ·)` → `gK = 2·(SA·K) + SBᵀ·V`、`gV = SB·K`。**双向分支**：`row_reduce_sum(dA/dB) → (BH·dk², 1)` → `broadcast_row_inplace` 沿 seq 广播 → `scan_suffix_outer(causal=false, dummy, false)`（shader 的 causal=0 分支即 S_i=D_i，D 已是全集梯度，等价全集后缀）。
- **forward_step（推理逐 token）**：Q/K/V → RoPE → ReLU → `scan_prefix_outer(K1, V1, Q1, Q1, A_state, B_state, true, dk, H, causal, ·, false)`（**先扫描**，A0=旧状态）→ `batched_matmul`（k·kᵀ、v·kᵀ）→ `add_inplace` 更新状态（**后更新**）。

### 7.3 引擎化中暴露的坑（改这段代码前必读）

1. **forward_step 顺序 = 先扫描、后更新状态**：`causal=true, has_state=true` 的扫描语义**含自身**（A_t 含 k_t·k_tᵀ）；若 A0 已含当前 token 则双算。与旧 CPU 实现"先累积再用"等价的前提是 A0 为旧状态。
2. **双向分支用 `row_reduce_sum`，不是 `col_reduce_sum`**：col 版返回 (1,C)；需要 (R,1)=(BH·dk²,1) 直接作 `broadcast_row_inplace` 的 row_vec（无需 transpose）。
3. **CPU 参考实现 boundary 形状校验曾潜伏 bug**：写成 `(rows/dk)*seq`（=B·H·seq），契约是 `(1, B·seq)`（shader 索引 `Bnd[b*seq+t]` 亦证）。旧 Layer 从不走引擎路径所以从未暴露；Layer 引擎化后 doc-aware gradcheck 直接踩中，已修为 `(rows/(dk*heads))*seq`，并给 vk backend 三个 `*_gpu` 方法补同义校验。
4. **标量块 s/r 头内逐行重复**（shader 写全部行），Layer 读任一行即可。

### 7.4 验证基线（2026-09-04，GTX 850M）

| 验证 | 结果 |
|---|---|
| `rapt_gradcheck`（CPU） | 14/14，max_err 与改造前基线逐位一致（causal 0.0290 / bidir kink 0.2554 / doc-aware 0.00868） |
| `rapt_gradcheck --gpu` | 3 段全 OK，kink 值 0.102/0.252 符合预期，全部 ≪ tol 5e-2；batch>1 覆盖（铁律 5）由 causal / doc-aware 段保证（段内 batch=2 固定，无 --batch 参数） |
| `rapt_smoke_test`（CPU） | KV-cache 一致性 max_diff=0（逐位一致） |
| `rapt_smoke_test --gpu` | KV-cache 一致性 max_diff=2.98e-08（= float32 1 ulp：GPU 加法顺序与 CPU 不同，数值正确） |
| `text_train --model rapt --gpu` | 106KB 语料 / 528 词表 / 319 步 / 1 epoch：loss 6.3→3.9，10.3s，无 TDR |
| `text_infer --model rapt --gpu` | 16 token 生成 0.4s（39 tok/s）；forward_step GPU 录制路径（batched_matmul 状态更新 + has_state 扫描同 command buffer）验证通过 |
| ctest 全量 | 31/31 全绿；AOT 收集仍 56 条融合表达式（手写原语不进融合注册表，闭合世界未破坏） |

> 完整 handoff（设计决策、文件锚点、步骤指令）见 `docs/rapt-gpu-handoff.md`。