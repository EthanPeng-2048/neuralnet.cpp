# 线性注意力家族：RLA / RAPT / RLA-2 与 GPU 落地

> **读者**：调用者（库使用者 / 扩展者）——本分册定位见 [`README.md`](README.md)。
> 项目内部的设计 / 路线图 / 盘点在 [`../todo/`](../todo/)，历史在 [`../history.md`](../history/history.md)。
> **详细研究基线**：RLA-2/RAPT 的完整公式、复杂度、实现映射和事实/假设边界见
> [`../../research/rla/01-what-is-rla.md`](../../research/rla/01-what-is-rla.md)；效果实验尚为
> [`../../research/rla/02-rla-effectiveness.md`](../../research/rla/02-rla-effectiveness.md) 的 TODO。

> **30 秒了解 RAPT**：标准 Transformer 注意力要算 L×L 的分数矩阵，序列翻倍、开销翻四倍。
> RAPT 走**线性注意力**路线：利用矩阵结合律，先把历史键值"汇总"成一个固定大小的
> d×d 状态矩阵，再用当前 query 去查这个摘要——**开销与序列长度线性、推理时每步恒定 O(d²)**。
> 汇总前 Q/K 各自逐维 ReLU；两者的正支持集不交时 score 精确为 0。训练后是否形成可解释、
> 可复用的稀疏检索，属于待测问题，不能只由公式直接推出。
> 本篇按「直观理解 → 数学定义 → 家族演进 → 工程落地（扫描原语）→ 训练显存」组织。

**目录**：[1 直观理解](#1-直观理解为什么是线性注意力) ·
[2 RLA 数学](#2-rla原始的-relu-linear-attention) ·
[3 家族演进](#3-家族演进) ·
[4 RAPT 工程落地](#4-rapt引擎化扫描原语gpu-落地) ·
[5 注意力实现形态](#5-注意力实现形态fold-流式) ·
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

**ReLU 的作用（本项目的路线选择）**：特征映射 φ 取 `ReLU`（逐维 `max(0,·)`），
query/key 各自先截断再点积，因此 score 非负。若两者的正特征支持集没有交集，score 精确为 0，
该 Token 对本次输出无贡献；若支持集有交集，则 score 为正。它是**正支持集交集门控**，不是
“负点积截断”，也不是只选一个 Token 的 Top-1 硬选择。相比 softmax 的指数归一化：

- **动态 query**：每生成一个新 token，q 会重新计算；历史状态持续累加，没有输入相关的学习型
  遗忘门。这不等于长距离信息一定能被有效利用，容量与检索效果仍需实测；
- **可检查**：score 是非负向量点积，可以拆到具体正特征维度；这是可解释性实验的入口，
  不等于已经证明训练后会出现稳定、人类可读的检索模式。

**当前实现的三个关键选择**（RLA-2 的取舍见 §3）：

1. 不使用 softmax 的 `1/√d_k` 缩放；Q/K 先做逐头 RMSNorm；
2. 归一化分母逐 query 计算——原版设计用 L2 分母，**当前 RLA-2 改用 Sum 分母**
   （输出为带 `ε` 修正的非负加权平均，见 §3）；
3. 当前公开 RAPT 模型只接受 RoPE，且 RoPE 要在进 ReLU 之前。

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

### 2.2 核心计算特性

| 特性 | 描述 |
| :--- | :--- |
| **复杂度** | 显式构建 L×L 是 O(L²)。**必须用结合律**：先算 `K'ᵀV`，再算 `Q'(K'ᵀV)` → O(Ld²)；d ≪ L 时视作线性 |
| **非负 score** | score = `Σ ReLU(q_j)·ReLU(k_j)`；正支持集不交时为 0，有交集时为正；不是 Top-1 选择 |
| **动态 query** | 每个新 token 生成新 q，并查询累计状态；causal 状态只在文档边界清零 |
| **位置编码** | 当前公开 RAPT 工厂只接受 RoPE，且 RoPE 施加在 Q/K 进 ReLU **之前** |

### 2.3 历史设计取舍与当前 RLA-2 对策

| 问题 | 旧设计描述 | 当前 RLA-2 |
| :--- | :--- | :--- |
| Softmax 式缩放 | 旧概览主张去掉 `1/√d_k` | 不使用 `1/√d_k`；Q/K 先做逐头 RMSNorm |
| 归一化分母 | 原版 RLA 使用 L2 分母 | 使用 Sum 分母 `Σscore+1e-4`，输出为非负加权平均 |
| ReLU 零维 | 旧设计提到 `ReLU(x)+0.1x` 泄漏残差 | 当前不加入泄漏项；用 Q/K RMSNorm 调整数值尺度 |

表中“为什么这样选”多为设计意图。它是否带来更好的训练稳定性或表达能力，必须由
[`../../research/rla/02-rla-effectiveness.md`](../../research/rla/02-rla-effectiveness.md)
的消融实验回答。

---

## 3. 家族演进

三版一条主线：**在不牺牲线性复杂度与硬截断稀疏的前提下，收敛数值稳定性与超参数**。

| 维度 | RLA（原版） | RLA-2（修正版） | LRLA（扩展版） |
| :--- | :--- | :--- | :--- |
| 归一化分母 | `sqrt(Σ score²) + ε`（L2） | `Σ score + ε`（Sum，输出为带 ε 修正的加权平均） | 同 RLA-2 |
| 分数定义 | 推理/训练定义不一致（原版缺陷） | **统一为逐维 ReLU 后点积** | 同左 |
| 防神经元死亡 | 残差 `ReLU(x)+0.1x`（泄漏破坏硬截断） | **RMSNorm 前置**（无泄漏） | MLP 内 LayerNorm |
| 位置衰减 | 可加 γ^(t-i)（可选） | **无 γ**，纯 RoPE 提供位置信息 | 同左 |
| 特征映射 φ | 固定 ReLU | 固定 ReLU | **可学习多层网络**（末层必须 ReLU，保证非负） |
| 额外学习参数 | γ（若使用） | 无新增可学习核参数 | MLP 结构/宽度 |
| 推理每步 | O(td)（直接求和时） | **O(d²)**（状态递推） | O(d_φ·d) |

**RLA-2 核心公式**（当前研究基线）：

```
q' = ReLU( RoPE( RMSNorm(W_q x) ) ),  k' 同理
S_t = S_{t-1} + k_t' ⊗ v_t   ∈ R^{d×d}     z_t = z_{t-1} + k_t'
Output_t = q_t' S_t / ( q_t' · z_t + ε )     （ε = 1e-4）
```

训练时可分块并行累加状态；当前实现用前缀扫描生成逐位置状态供 backward 使用，
时间随 `L` 线性增长，激活空间也随 `L` 线性增长（但不会物化 `L×L` score 矩阵）。

**与主流的复杂度关系**：RLA-2 的序列混合复杂度与常见线性注意力同档 `O(Ld²)`。
“实际更快”“长上下文更强”“可解释性最高”等效果主张当前**没有本仓实验证据**；
见 [`../../research/rla/02-rla-effectiveness.md`](../../research/rla/02-rla-effectiveness.md)。

**LRLA 一句话**：把固定的 ReLU 特征映射换成末层 ReLU 的可学习 MLP，以研究更灵活的非负
特征映射；代价是参数增多、需防过拟合。它当前只有设计描述，效果未验证。

**工程基线**：**项目实现 = RLA-2 算法 + §4 的扫描原语工程**——`compute_layer_rapt.hpp` 的
`ReLULinearAttention` 头注释即 RLA-2（Sum 分母、无 A 状态、RMSNorm→RoPE→ReLU 顺序）。
§2 的 L2 分母细节是 RLA 原版演进来源，阅读时以 §3/§4 的现行实现为准；LRLA 为设计储备，未实现。

---

## 4. RAPT：引擎化扫描原语（GPU 落地）

RAPT 是上述算法的工程实现：Layer 用 3 个 op-level 扫描原语 + DSL 表达式组合出整个算法
（铁律 3：shader 永不含算法——shader 只做"带状态的顺序归约 + matvec 读出 / 外积"）。

### 4.1 三个扫描原语（`compute_engine.hpp` 纯虚）

| 原语 | 语义 | 输出 |
|---|---|---|
| `scan_prefix_outer(K,V,P,R,A0,B0,has_state,dk,heads,causal,boundary,has_bnd)` | causal=true：`A_t = A0 + Σ_{i≤t, 同文档} k_i·k_iᵀ`、`B_t = B0 + Σ v_i·k_iᵀ`（文档边界处运行态清零，A0/B0 仅首文档生效）；causal=false：全集常数 | `(B·H·5·dk, seq)` 行块：[0) B·P  [1) A·P  [2) Bᵀ·R  [3) s  [4) r |
| `scan_suffix_outer(D,X,Y,dk,heads,causal,boundary,has_bnd)` | causal=true：`S_i = Σ_{t≥i, 同文档} D_t`（i+1 为文档起点时先清零）；causal=false：`S_i = D_i` | `(B·H·3·dk, seq)`：[0) S·X  [1) S·Y  [2) Sᵀ·Y |
| `outer_col(P,R,S,dk,has_scale)` | 逐列外积 `out = P·Rᵀ`（has_scale 时逐列乘 S[t]） | `(B·H·dk², seq)` |

**形状约定**：batch-major（`i = b*seq+t`）；头 `(b,h)` 行块起点 `r0 = (b*H+h)*dk`；K/V/P/R（X/Y）为 `(B·H·dk, seq)`，D 为 `(B·H·dk², seq)`；A0/B0 为 `(H·dk, dk)`（B>1 按头循环）；boundary 为 `(1, B·seq)`（1 = 文档起点）；空参数用 **(1,1) dummy + bool 标志**（规避 0 字节 GPU buffer）；**dk ≤ 64**（GPU MAX_DK）；标量块 s/r 在头块内 dk 行重复存放（实现写全部行，Layer 读任一行均可）。

**GPU 实现骨架（P4-2b，2026-10-15）**：1 个 workgroup/头、256 线程（派发 `wg=(1, B·H, 1)`）。
`scan_prefix_outer` 的专用路径（dk ≤ 64）是**行归属**结构——线程 a 独占 A/B 第 a 行，于是
"前缀累加 → 行读出"（`B·q`/`A·q`）由同一线程完成、每 t 只需 **3 次 barrier + 1 次全局往返**；
内层循环 4 路展开（先并行取数、再按原序累加）只提 ILP、**不改任何元素的累加次序** ⇒ GPU 与
CPU 实现**逐位一致**（门禁 `scan_bitwise_test`）。**dk ≤ 32 走窄共享变体**（同一份 `.comp` 加
`-DNN_SCAN_DK_MAX=32`；共享 33.3KB → 8.5KB，Maxwell 每 SM 驻留 6 个 WG 而不是 1 个）。
⚠ `scan_suffix_outer` **刻意保持原结构**：它的成本由 D 的 `(dk², seq)` 列式散布读主导（流量/争用
受限），提高占用率实测更慢——见 `docs/history/history.md`「P4-2 (b)」条。

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
- **forward_step（推理逐 token，KV cache）**：`Q/K/V → RMSNorm → RoPE → ReLU` → **先更新状态**（`dsl::matmul(V, Kp, transB, batch=H)` 求 `v·k'ᵀ` → `dsl::compute_into` 累加进 `B_state`；`dsl::compute_into` 累加 `z_state += k'`）→ `num = B·q'`（`dsl::matmul(batch=H)`）→ `den = q·z + ε`（逐头点积，CPU round-trip 可接受）→ `out = num/den`（与 forward 的除法表达式同构）。

### 4.3 改这段代码前必读的坑

1. **RLA-2 公式不消费 A 状态**（`Σk'k'ᵀ`）：分母只依赖 `z = Σk'`，
   `scan_prefix_outer` 的 A0 输入在 RLA-2 路径上以 dummy 传入，A·P 输出被忽略；
   **别按原版 RLA 的 `2·ds·Aq` 项去改 backward**。注意当前通用扫描原语仍会在内部计算 A，
   这是现存实现成本。
2. **双向分支的广播技巧**：单视图表达式不合法（IR 规定输出=最后一条指令的 dst，空指令表被 `validate_expr_spec` 拒绝），故必须写成 `row_broadcast(x) + rparam(0)`；这是**刻意的 hack，不是冗余**。
3. **boundary 形状契约是 `(1, B·seq)`**（shader 索引 `Bnd[b*seq+t]`），不是 `(rows/dk, seq)`；写错只有 doc-aware gradcheck 能暴露，改动后必跑。
4. **backward 缓存前置校验**：checkpoint/offload 模式下缓存为空必须**立刻报错**，不能让空张量流进 matmul（静默算出的垃圾梯度比崩溃危险得多）。
5. 标量块 s/r 头内逐行重复，Layer 读任一行即可；空参数一律 `(1,1) dummy + bool 标志`（规避 0 字节 GPU buffer）。

### 4.4 验收基线（参考硬件 GTX 850M）

| 验证 | 结果 |
|---|---|
| `rapt_gradcheck`（CPU / --gpu） | 14/14、3 段全 OK，max_err 基线：causal 0.0290 / bidir 0.2554 / doc-aware 0.00868 |
| `rapt_smoke_test`（CPU / --gpu） | KV-cache 一致性 max_diff = 0 / 2.98e-08（= 1 ulp，加法顺序不同） |
| `text_train --model rapt --gpu` | 319 步：loss 6.3→3.9，10.3s，无 TDR |
| `text_infer --model rapt --gpu` | 16 token 0.4s；forward_step GPU 录制路径验证通过 |

> 扫描原语是手写 shader，不进融合注册表（闭合世界不受影响，AOT 收集条数不变）；
> GPU 侧接线实现见 `compute_gpu_engine.hpp` 与 `backend/compute_vk_backend.hpp`。

---

## 5. 注意力实现形态（fold 流式）

本项目**不存在单一 "flash-attn" kernel**——FlashAttention 的核心思想（online softmax + 不物化 S）
由注意力 forward 的**单 fold kernel 分块流式**实现（`FoldSpec`：QKᵀ/掩码/online softmax/ΣwV 逐
`EXPR_FOLD_BLOCK=128` 块完成，S 矩阵绝不物化，见 `08-pitfalls-and-lessons.md` §4 与
`compute_layer_attention.hpp` 头注释）。掩码变体（Plain/Causal/Alibi/Doc/AlibiDoc）经 `FoldSpec`
登记，漏登记即 GPU 闭合世界硬报错。backward 为 R/X 表达式 + 3 个 `batched_matmul`。

> 与传统两趟式多 kernel 注意力（W 物化供复用）的对照分析已移入 `docs/history/history.md`。

---

## 6. 训练期显存：激活重计算与 offload

RAPT 与 GPT 同档支持三种开关：

| 能力 | 入口 | 实现要点 |
|------|------|----------|
| 梯度检查点（激活重计算） | `RAPTModel::set_checkpoint_every(N)` | 每 N 个 RAPTBlock 存一次块输入；backward 先 `forward_recompute` 再反向，用后 `clear_cache()` |
| activation offload | `RAPTModel::set_activation_offload(true)` | 通用 `ActivationOffloader`（`compute_layer_base.hpp`，与 GPTBlock 共用）把 `activation_cache()` 搬 host-visible |
| batch flush | `RAPTModel::set_flush_interval(N)` | 每 N 个块 `flush_batch()`，拆小录制防 TDR |

**为什么 RLA 的重算特别划算**：RLA 前向是 O(L·d_k²)，GPT 注意力重算含 QKᵀ 是 O(L²)——同一 `stride` 下 RLA 重算代价约为 GPT 的 1/L。因此 RAPT **优先用检查点**，offload 次之。

`ReLULinearAttention` 在 checkpoint 模式下不写任何 backward 缓存（Qp/Kp/V_re/Q_normed/K_normed/逐头 1/rms），backward 缺缓存直接报错而非拿空张量算梯度；`activation_cache()` 列出全部本层缓存 + 4 个子层缓存（offload 完整覆盖）。

**坑**：复合层 override `forward_recompute` 必须调用**虚函数** `set_checkpoint_mode` 关闭子层——基类默认实现只改本块标志位，子层缓存不重建，表现为"stride=1 能过、stride=2 过不了"（见 `08-pitfalls-and-lessons.md` 模式 H）。

**验收**：`rapt_test` 内含 `rapt_checkpoint`（stride∈{1,2} vs 全存基线逐位一致，max_abs=0）；`rapt_offload_test`（GPU-only）同样逐位一致。
