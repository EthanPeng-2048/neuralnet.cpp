# 线性注意力家族：RLA / RAPT / RLA-2 与 GPU 落地

> **30 秒了解 RAPT**：标准 Transformer 注意力要算 L×L 的分数矩阵，序列翻倍、开销翻四倍。
> RAPT 走**线性注意力**路线：利用矩阵结合律，先把历史键值"汇总"成一个固定大小的
> d×d 状态矩阵，再用当前 query 去查这个摘要——**开销与序列长度线性、推理时每步恒定 O(d²)**。
> 汇总时用 ReLU 逐维门控（方向不相关的 key 直接置 0），实现"只检索语义相关 Token"的硬稀疏。
> 本篇按「直观理解 → 数学定义 → 家族演进 → 工程落地（扫描原语）→ 训练显存」组织。

**目录**：[1 直观理解](#1-直观理解为什么是线性注意力) ·
[2 RLA 数学](#2-rla原始的-relu-linear-attention) ·
[3 家族演进](#3-家族演进) ·
[4 RAPT 工程落地](#4-rapt引擎化扫描原语gpu-落地) ·
[5 历史：两趟式注意力](#5-历史两趟式注意力已删除) ·
[6 训练显存](#6-训练期显存激活重计算与-offload)

---

## 1. 直观理解：为什么是线性注意力

**Softmax 注意力在做什么**：每个位置的 query 和**所有**历史 key 两两打分，softmax 归一化后加权求和 value。打分矩阵 L×L → 时间 O(L²d)、显存 O(L²)。

**线性注意力换法**：把"逐个打分再加权"换成"先汇总历史、再查一次"——数学上靠结合律：

```
softmax 风格：  Σ_i  score(q, k_i) · v_i          ← 逐个 i 打分，O(L)
线性注意力：    q · ( Σ_i  φ(k_i) ⊗ v_i )          ← 先把历史压进状态 S = Σ φ(k)⊗v，再查
                          └── 这个和与 L 无关！ └── 每次查询只做一次 d×d 矩阵向量乘
```

状态 `S ∈ R^{d×d}`（键值二阶矩）大小**固定**，不随序列增长——训练时可分块并行累加，推理时逐步递推，这就是 O(L) 训练 / O(d²) 每步推理的来源。

**ReLU 的作用（本项目的路线选择）**：特征映射 φ 取 `ReLU`（逐维 `max(0,·)`）。若 query 与某个 key 在特征空间方向相反，点积为负 → ReLU 精确置 0 → **该 Token 完全不被检索**。相比 softmax 的"全都沾一点"，这是可微分的硬注意力门控：

- **动态重选**：每生成一个新 token，q 变了，门控重新计算——不像 RNN 隐状态会遗忘久远信息；
- **可解释**：注意力权重是非负向量点积，可逐维分析哪些特征在起作用。

**三个容易踩错的实现约束**（源自 RLA 原版的教训，RLA-2 的取舍见 §3）：

1. 不能照搬 `1/√d_k` 缩放（ReLU 非负，点积随维度线性增长 → 训练震荡）；
2. 归一化分母必须逐 query 计算——原版用 L2 分母（否则长文本梯度消失），**项目实现的 RLA-2 改用 Sum 分母**（输出=加权平均，幅值恒定，见 §3）；
3. **禁止绝对位置编码**，只能 RoPE/ALiBi，且 RoPE 要在进 ReLU 之前。

> 项目内这套实现统称 **RAPT**（`compute_layer_rapt.hpp` 的 `ReLULinearAttention` / `RAPTModel`）。

---

## 2. RLA：原始的 ReLU-Linear Attention

### 2.1 算法数学定义（闭式解）

序列长 L、隐藏维 d，查询位置 t 的输出一行式：

```
Output_t = Σ_{i=1..L} σ(q_t)ᵀ σ(k_i) · v_i / ( sqrt( Σ_{i=1..L} ( σ(q_t)ᵀ σ(k_i) )² ) + ε )
```

- `σ(x) = ReLU(x) = max(0, x)`（**逐元素**操作，不是向量模长）
- `ε` 极小常数（1e-6）防除零
- 全局并行视图：`Q' = ReLU(QW_q)`、`K' = ReLU(KW_k)`、`V = VW_v`
- 分母是**每个 query 独立的标量**，必须逐行归一化，不能一次矩阵除法解决

### 2.2 关键算法特性（必须遵守）

| 特性 | 描述 |
| :--- | :--- |
| **复杂度** | 显式构建 L×L 是 O(L²)。**必须用结合律**：先算 `K'ᵀV`，再算 `Q'(K'ᵀV)` → O(Ld²)；d ≪ L 时视作线性 |
| **稀疏激活** | q、k 方向相反 → 点积为负 → ReLU 置 0，"只激活语义相关的 Token" |
| **动态重选** | 每个新 token 生成新 q，基于全量原始 k 重算，无 RNN 遗忘问题 |
| **位置编码** | **严禁 Learned PE**；必须 RoPE 或 ALiBi，且 RoPE 施加在 Q/K 进 ReLU **之前** |

### 2.3 致命陷阱与解决方案

| 陷阱 | 现象 | 解决方案 |
| :--- | :--- | :--- |
| **照搬 Softmax 的 scale** | 训练不稳定，loss 震荡 | **禁用** `1/√d_k`；缩放放在 ReLU 之前，Q/K 先过 LayerNorm |
| **分母导致梯度消失** | 长文本（>8k）输出趋近 0 | 用 L2 分母（平方和开根）而非求和——等价于对注意力分数做余弦约束 |
| **ReLU 维度永久死亡** | 模型失去表达能力 | Q/K 投影后加残差：`q = ReLU(q) + 0.1*q`（保证负值有微弱梯度） |

---

## 3. 家族演进

三版一条主线：**在不牺牲线性复杂度与硬截断稀疏的前提下，收敛数值稳定性与超参数**。

| 维度 | RLA（原版） | RLA-2（修正版，2026-09-04） | LRLA（扩展版） |
| :--- | :--- | :--- | :--- |
| 归一化分母 | `sqrt(Σ score²) + ε`（L2） | `Σ score + ε`（Sum，输出=加权平均，幅值恒定） | 同 RLA-2 |
| 分数定义 | 推理/训练曾不一致 | **统一为逐维 ReLU 后点积** | 同左 |
| 防神经元死亡 | 残差 `ReLU(x)+0.1x`（泄漏破坏硬截断） | **RMSNorm 前置**（无泄漏） | MLP 内 LayerNorm |
| 位置衰减 | 可加 γ^(t-i)（可选） | **无 γ**，纯 RoPE 提供位置信息 | 同左 |
| 特征映射 φ | 固定 ReLU | 固定 ReLU | **可学习多层网络**（末层必须 ReLU，保证非负） |
| 额外超参数 | γ（若使用） | **零** | MLP 结构/宽度 |
| 推理每步 | O(td)（直接求和时） | **O(d²)**（状态递推） | O(d_φ·d) |

**RLA-2 核心公式**（当前研究基线）：

```
q' = ReLU( RoPE( RMSNorm(W_q x) ) ),  k' 同理
S_t = S_{t-1} + k_t' ⊗ v_t   ∈ R^{d×d}     z_t = z_{t-1} + k_t'
Output_t = q_t' S_t / ( q_t' · z_t + ε )     （ε = 1e-4）
```

训练用**分块因果**形式：块间传状态 `S/z`，块内并行——复杂度 O(Ld²)，空间 O(Ld + Bd²)，块大小取 128~256。

**RLA-2 vs 主流**：训练复杂度与 Gated DeltaNet / Mamba-2 同档 O(Ld²)，但**零超参数、可解释性最高**（无 softmax 指数挤压，权重线性可解释）；最适合超长上下文、硬检索密集场景（法律文档、代码库检索）。

**LRLA 一句话**：把固定的 ReLU 特征映射换成可学习 MLP（末层仍 ReLU 保非负），逼近 softmax 核的无限维特征映射；代价是参数增多、需防过拟合。RoPE 必须在进 MLP **之前**（非线性会破坏旋转相对位置性质）。

**工程基线**：**项目实现 = RLA-2 算法 + §4 的扫描原语工程**——`compute_layer_rapt.hpp` 的
`ReLULinearAttention` 头注释即 RLA-2（Sum 分母、无 A 状态、RMSNorm→RoPE→ReLU 顺序）。
§2 的 L2 分母细节是 RLA 原版演进来源，阅读时以 §3/§4 的现行实现为准；LRLA 为设计储备，未实现。

---

## 4. RAPT：引擎化扫描原语（GPU 落地）

RAPT 是上述算法的工程实现。2026-09-04 的**引擎化**改造删掉了 Layer 内的
`scan_forward_/scan_backward_` 纯 CPU 标量循环（PCIe 往返），改为 3 个 op-level
扫描原语，由 Layer 用原语 + 逐元素原语组合表达算法（铁律 3：shader 永不含算法）。

### 4.1 三个扫描原语（`compute_engine.hpp` 纯虚）

| 原语 | 语义 | 输出 |
|---|---|---|
| `scan_prefix_outer(K,V,P,R,A0,B0,has_state,dk,heads,causal,boundary,has_bnd)` | causal=true：`A_t = A0 + Σ_{i≤t, 同文档} k_i·k_iᵀ`、`B_t = B0 + Σ v_i·k_iᵀ`（文档边界处运行态清零，A0/B0 仅首文档生效）；causal=false：全集常数 | `(B·H·5·dk, seq)` 行块：[0) B·P  [1) A·P  [2) Bᵀ·R  [3) s  [4) r |
| `scan_suffix_outer(D,X,Y,dk,heads,causal,boundary,has_bnd)` | causal=true：`S_i = Σ_{t≥i, 同文档} D_t`（i+1 为文档起点时先清零）；causal=false：`S_i = D_i` | `(B·H·3·dk, seq)`：[0) S·X  [1) S·Y  [2) Sᵀ·Y |
| `outer_col(P,R,S,dk,has_scale)` | 逐列外积 `out = P·Rᵀ`（has_scale 时逐列乘 S[t]） | `(B·H·dk², seq)` |

**形状约定**：batch-major（`i = b*seq+t`）；头 `(b,h)` 行块起点 `r0 = (b*H+h)*dk`；K/V/P/R（X/Y）为 `(B·H·dk, seq)`，D 为 `(B·H·dk², seq)`；A0/B0 为 `(H·dk, dk)`（B>1 按头循环）；boundary 为 `(1, B·seq)`（1 = 文档起点）；空参数用 **(1,1) dummy + bool 标志**（规避 0 字节 GPU buffer）；**dk ≤ 64**（GPU MAX_DK）；标量块 s/r 在头块内 dk 行重复存放（实现写全部行，Layer 读任一行均可）。

### 4.2 Layer 组合（算法在 Layer；现行 RLA-2 路径）

前置统一：`Q/K → RMSNorm → RoPE → ReLU` 得 `Qp/Kp`（`V` 不 ReLU、不 RoPE）。

- **forward**（两条前缀扫描）：
  1. **主扫描** `scan_prefix_outer(Kp, V, Qp, V, dummy, dummy, false, causal, bnd)` → slice `[0)` = `B·q'`（分子 num）；
  2. **z-scan** `scan_prefix_outer(Kp, V_ones, Qp, V_ones, …)`（V 换成全 1 矩阵 → `B_t` 退化为 `z = Σk'`）→ slice `[0)` = `q·z`（分母标量）；
  3. 单条 DSL：`out = num / (q·z + 1e-4)` → `rearrange_3d`（batch>1）→ `w_o_`。
- **backward**（一前缀 + 两条后缀）：
  1. z-scan 取 `q·z` 与 `[2)/dk = z`；主扫描（R=梯度）取 `Bᵀ·g`、`r = g·(B·q)`；
  2. DSL 链算 `gnum = g/(q·z+ε)`、`scale = −r/(q·z+ε)²`、`gQ = Bᵀ·gnum + scale·z`（全融合）；
  3. `outer_col(gnum, Qp)` 物化 `dB` → `scan_suffix_outer(dB, Kp, V)` 得 `gV = S_B·k`、`gK_B = S_Bᵀ·v`；
     分母项 `gK += suffix(scale·q)`：causal 走 `outer_col(Qp, e_0, scale, has_scale=true)` + 后缀扫描；
     双向走 `row_reduce_sum(dB)` → 单条 `row_broadcast + rparam(0)` 表达式广播 → `scan_suffix_outer(causal=false)`；
  4. `select(Qp > 0, gQ, 0)` 反传过 ReLU → RoPE 反向 → RMSNorm 反向 → 各投影 `w_*.backward`。
- **forward_step（推理逐 token，KV cache）**：`Q/K/V → RMSNorm → RoPE → ReLU` → **先更新状态**（`batched_matmul(V, Kp, transB)` → `add_inplace(B_state)`；`add_inplace(z_state, Kp)`）→ `num = B·q'`（batched_matmul）→ `den = q·z + ε`（逐头点积，CPU round-trip 可接受）→ `out = num/den`。

### 4.3 改这段代码前必读的坑

1. **RLA-2 无 A 状态**（`Σk'k'ᵀ`）：分母只依赖 `z = Σk'`，`scan_prefix_outer` 的 A 行块在 RLA-2 路径上以 dummy 传入；**别按原版 RLA 的 `2·ds·Aq` 项去改 backward**。
2. **双向分支的广播技巧**：单视图表达式不合法（IR 规定输出=最后一条指令的 dst，空指令表被 `validate_expr_spec` 拒绝），故必须写成 `row_broadcast(x) + rparam(0)`；这是**刻意的 hack，不是冗余**。
3. **boundary 形状契约是 `(1, B·seq)`**（shader 索引 `Bnd[b*seq+t]`），不是 `(rows/dk, seq)`；写错则 doc-aware gradcheck 才会暴露（历史已踩过）。
4. **backward 缓存前置校验**：checkpoint/offload 模式下缓存为空必须**立刻报错**，不能让空张量流进 matmul（旧行为 = 静默算出垃圾梯度）。
5. 标量块 s/r 头内逐行重复，Layer 读任一行即可；空参数一律 `(1,1) dummy + bool 标志`（规避 0 字节 GPU buffer）。

### 4.4 验证基线（2026-09-04，GTX 850M）

| 验证 | 结果 |
|---|---|
| `rapt_gradcheck`（CPU / --gpu） | 14/14、3 段全 OK，max_err 与改造前基线一致（causal 0.0290 / bidir 0.2554 / doc-aware 0.00868） |
| `rapt_smoke_test`（CPU / --gpu） | KV-cache 一致性 max_diff = 0 / 2.98e-08（= 1 ulp，加法顺序不同） |
| `text_train --model rapt --gpu` | 319 步：loss 6.3→3.9，10.3s，无 TDR |
| `text_infer --model rapt --gpu` | 16 token 0.4s；forward_step GPU 录制路径验证通过 |

> 手写原语不进融合注册表（闭合世界未破坏，AOT 收集条数不变）；6 处 Vulkan 接线细节沉淀在
> `01-compute-engine-development.md`。

---

## 5. 历史：两趟式注意力（已删除）

> S7 的"两趟式多 kernel 注意力"（forward 3 个融合 kernel m→l→W × 4 种掩码变体、
> backward 3 个，W 物化供复用）**已删除**——2026-09-24 起注意力 forward 为**单 fold kernel**
> 分块流式（QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 矩阵绝不物化，
> 见 `08-pitfalls-and-lessons.md` §4 与 `compute_layer_attention.hpp` 头注释）。
>
> 保留的结论：本项目**不存在单一 "flash-attn" kernel**——FlashAttention 的核心思想
> （online softmax + 不物化 S）由 fold 流式求值实现；掩码变体（Plain/Causal/Alibi/Doc/AlibiDoc）
> 经 `FoldSpec` 登记，漏登记即 GPU 闭合世界硬报错。与传统 Flash Attention 的对照分析
> （kernel 数量、IO 复杂度、掩码扩展方式）见 git 历史中本节原文。

---

## 6. 训练期显存：激活重计算与 offload

RAPT 与 GPT 同档支持三种开关（2026-09-19 补齐；此前 RAPT 侧是**静默 no-op** 而 CLI 会打印"已启用"）：

| 能力 | 入口 | 实现要点 |
|------|------|----------|
| 梯度检查点（激活重计算） | `RAPTModel::set_checkpoint_every(N)` | 每 N 个 RAPTBlock 存一次块输入；backward 先 `forward_recompute` 再反向，用后 `clear_cache()` |
| activation offload | `RAPTModel::set_activation_offload(true)` | 通用 `ActivationOffloader`（`compute_layer_base.hpp`，与 GPTBlock 共用）把 `activation_cache()` 搬 host-visible |
| batch flush | `RAPTModel::set_flush_interval(N)` | 每 N 个块 `flush_batch()`，拆小录制防 TDR |

**为什么 RLA 的重算特别划算**：RLA 前向是 O(L·d_k²)，GPT 注意力重算含 QKᵀ 是 O(L²)——同一 `stride` 下 RLA 重算代价约为 GPT 的 1/L。因此 RAPT **优先用检查点**，offload 次之。

`ReLULinearAttention` 在 checkpoint 模式下不写任何 backward 缓存（Qp/Kp/V_re/Q_normed/K_normed/逐头 1/rms），backward 缺缓存直接报错而非拿空张量算梯度；`activation_cache()` 列出全部本层缓存 + 4 个子层（Q_normed/K_normed/Q_rms_inv/K_rms_inv 曾漏列 → offload 漏搬，已补）。

**已修的两个前置缺陷**（不修则静默毁模型）：
1. `RAPTModel::clear_cache()` 曾清空 `token_emb_`（**模型参数**）——检查点每块 backward 后都调它，等于毁掉词嵌入。
2. `RAPTModel::forward` 曾只在 `doc_ids` 非空时下发文档 id → `set_doc_ids({})` 关不掉文档感知，跨 step 残留边界重置（跨样本串扰）。

**坑**：复合层 override `forward_recompute` 必须调用**虚函数** `set_checkpoint_mode` 关闭子层——基类默认实现只改本块标志位，子层缓存不重建，表现为"stride=1 能过、stride=2 过不了"（见 `08-pitfalls-and-lessons.md` 模式 H）。

**验收**：`rapt_test` 内含 `rapt_checkpoint`（stride∈{1,2} vs 全存基线逐位一致，max_abs=0）；`rapt_offload_test`（GPU-only）同样逐位一致。
