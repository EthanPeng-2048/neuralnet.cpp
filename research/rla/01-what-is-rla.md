# 什么是 RLA：从公式到本仓 RLA-2 / RAPT 实现

> 状态：算法与实现定义基线，非效果报告。
> 源码基线：`9b5e7ec`（2026-10-24）。
> 范围：本仓当前代码；不断言学术首创性，不做 GPT 质量对比。

## 0. 可引用结论

1. **RLA 是 ReLU Linear Attention。** 当前实现先对每个 head 的 Q/K 做 RMSNorm 与 RoPE，
   再分别逐元素 ReLU，最后用 `ReLU(q)·ReLU(k)` 作为相似度。
2. **它不是 O(L²) softmax 注意力。** 训练时用前缀扫描累计固定大小的键值状态，时间随序列
   长度线性增长；推理时每步只更新一个状态，不需要保存完整 KV 序列。
3. **当前算法版本是 RLA-2。** 分母是 `q'_t·Σk'_i + ε`，不是原版的 L2 分母；分母计算不
   需要 `Σk'k'ᵀ` 状态，因此本路径把通用扫描原语的 A 状态输入作为 dummy 传入。
4. **ReLU 给出的是正支持集门控，不是“负分数截断”。** 因 Q/K 已经过 ReLU，score 恒非负；
   两个 token 的正特征支持集不相交时 score 才精确为 0，并非只在“方向相反”时置零。
5. **RAPT 是模型/工程名，RLA-2 是注意力算法名。** RAPT 用 RLA-2 替换 decoder-only
   Transformer 中的 softmax 自注意力。

## 1. 名称与边界

### 1.1 RLA

RLA 指 ReLU Linear Attention：把 softmax 注意力的指数核替换成
`φ(q)ᵀφ(k)`，其中 `φ(x)=ReLU(x)`，并利用矩阵乘法的结合律避免显式构造
`L×L` 的 score 矩阵。

本仓当前实际运行的不是一句泛称“RLA”，而是项目定义的 **RLA-2** 变体。阅读旧文档时要把
“原版 RLA”和“当前 RLA-2”分开：

| 名称 | 本仓状态 | 关键语义 |
|------|----------|----------|
| 原版 RLA | 历史设计来源，源码当前不单独实现 | 旧概览称使用 L2 分母 `sqrt(Σ score²)+ε`；相关可选衰减/泄漏残差没有可执行实现可核验 |
| **RLA-2** | **当前实现** | Sum 分母、无 A 状态、Q/K 内置逐头 RMSNorm、RoPE 后 ReLU |
| LRLA | 仅设计储备 | 可学习 ReLU 特征网络；未实现 |
| RAPT | 当前模型实现 | decoder-only LM；RLA-2 注意力 + Pre-Norm 残差 + FFN + LM head |

“RLA 是自研路线”是本项目的归属口径；本文不把它扩展成“已证明学术首创”。要做优先权判断，
需要另做相关工作检索与逐项差异分析。

## 2. 记号

先看一个 batch 中的一条序列、一个 head。令：

- `L`：序列长度；
- `d`：模型宽度 `d_model`；
- `H`：head 数；
- `d_k=d/H`：单头宽度；
- `x_t∈R^d`：位置 `t` 的输入；
- `W_q,W_k,W_v,W_o∈R^{d×d}`：四个投影矩阵。

代码张量是 batch-major：batch 内位置索引为 `i=b·L+t`。数学式按单条序列书写；不同 batch
和 head 的状态彼此独立。

## 3. 当前 RLA-2 的完整计算

### 3.1 Q/K/V 投影

```text
q_t = W_q x_t
k_t = W_k x_t
v_t = W_v x_t
```

投影后按 head 切成 `H` 个 `d_k` 维向量。以下所有 RMSNorm、RoPE、ReLU 都逐 head 执行。

### 3.2 投影后 RMSNorm

RLA-2 在 Q 和 K 上各做一次无仿射参数的 RMSNorm：

```text
q̄_t = q_t / sqrt(mean(q_t²) + 1e-5)
k̄_t = k_t / sqrt(mean(k_t²) + 1e-5)
```

这里的 `mean` 在当前 head 的 `d_k` 个分量上求。源码位置：
[`rms_norm_forward_`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L143)。

这层 RMSNorm 是 RLA-2 的内部预处理，不是模型 block 外面的 LayerNorm/RMSNorm；它也没有
可学习缩放或偏置。

### 3.3 RoPE，然后 ReLU

```text
q̃_t = RoPE(q̄_t, t)
k̃_t = RoPE(k̄_t, t)
q'_t = ReLU(q̃_t)
k'_t = ReLU(k̃_t)
```

顺序是 **投影 → RMSNorm → RoPE → ReLU**。公开的 RAPT 工厂当前只接受 RoPE：
[`domain_rla.hpp`](../../include/neuralnet.cpp/domain_rla.hpp#L39)。旧概览曾写“RoPE 或
ALiBi”，但当前 `build_rapt_model()` 对非 RoPE 配置直接报错；算法层虽然可以拿到其他
`PositionEncoder`，公开模型路径没有开放 ALiBi。

值是原始投影，不做 ReLU 或 RoPE：

```text
v_t = W_v x_t
```

### 3.4 可检索集合

令 `I_t` 为位置 `t` 能读取的历史集合：

- causal：`I_t={i | i≤t}`；
- bidirectional：`I_t={0..L-1}`；
- document-aware：先限制在同一文档，再按 causal 与否限制 `i≤t`。

文档边界由 `boundary[b,t]=1` 标记；它是“位置 `t` 是新文档首位置”，不是文档 ID 张量本身。
实现见 [`build_boundary_`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L92)。

### 3.5 键值状态、分子、分母与输出

对每个 head，定义固定大小的键值记忆 `M_t` 和 key 总和 `z_t`：

```text
M_t = Σ_{i∈I_t} v_i (k'_i)ᵀ        ∈ R^{d_k×d_k}
z_t = Σ_{i∈I_t} k'_i                ∈ R^{d_k}
```

当前 query 查询状态：

```text
u_t   = M_t q'_t
den_t = (q'_t)ᵀ z_t + 1e-4
o_t   = u_t / den_t
```

逐 head 的结果拼回 `d` 维，再过输出投影：

```text
y_t = W_o concat(o_t^{(1)}, ..., o_t^{(H)})
```

当前代码把 `M_t` 对应的状态叫 `B_t`/`B_state`：
前向见 [`ReLULinearAttention::forward`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L391)，
推理见 [`forward_step`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L783)。

当分母接近零时，`1e-4` 防止直接除零；此时输出也接近零。该常数来自源码，不是可学习参数。

需要注意“无 A 状态”的精确范围：**RLA-2 的输出和反向公式不消费 A 状态**，但当前通用
`scan_prefix_outer` 仍会内部累计 `A=Σk'k'ᵀ` 并写入 `A·P` 行块，只是 Layer 忽略该输出。
这是当前扫描原语（[`scan_prefix_outer.comp`](../../shaders/scan_prefix_outer.comp)）的实现成本，
不改变 RLA-2 的算法定义。

## 4. 为什么它是线性注意力

直接展开 `u_t`：

```text
u_t = Σ_{i∈I_t} (q'_tᵀ k'_i) v_i
```

这与 softmax 注意力的“对所有 `i` 逐项打分再加权”在形式上有同一个二次型来源，但 RLA-2
把历史先压成 `M_t`，当前 query 只做一次矩阵向量乘。对每个新 token：

```text
更新状态：O(H·d_k²)
查询一次：O(H·d_k²)
```

两者都与当前上下文长度 `L` 无关。整段训练（batch 大小记为 `N`）的注意力混合成本近似：

```text
O(N·L·H·d_k²) = O(N·L·d·d_k)
```

当 `H` 固定、`d_k=d/H` 时，就是常见的 `O(L·d²)` 线性注意力口径；softmax 注意力的
score/value 混合是 `O(L²·d)`。

状态大小也线性于模型宽度：

```text
M: H·d_k² = d·d_k 个标量
z: H·d_k  = d    个标量
```

训练扫描实现会物化每位置状态供 backward 使用，空间是 `O(N·L·d·d_k)`；它仍是 `L` 的一次方，
但会随 `d²` 增长。这里的“线性”指序列长度，不表示成本与模型宽度线性。

### 4.1 与标准 KV cache 的差异

标准 softmax 推理每层保存随 `L` 增长的 K/V：每层约 `O(L·d)` 个标量。RLA-2 保存固定大小的
`M,z`，每步更新并查询，不会随上下文变长继续分配 KV 序列。代价是把历史压进有限状态，且
状态更新和查询涉及 `d_k²` 的矩阵量。

### 4.2 与 RNN 的差异

RLA-2 有递归形式，但它没有通常 RNN 那样的输入相关遗忘门。当前 causal 路径会一直累加历史，
只在文档边界清零。因此“不会遗忘旧信息”是当前状态更新规则的结构结果，不等于它一定能有效
利用长距离信息；后者需要 `02` 的实验回答。

## 5. ReLU 到底做了什么

当前 score 是：

```text
a_{t,i} = (q'_t)ᵀ k'_i
        = Σ_{j=1}^{d_k} ReLU(q̃_{t,j}) · ReLU(k̃_{i,j})
```

因此精确的机械性质是：

- 每个分量乘积非负，所以 `a_{t,i}≥0`；
- 若 `q'_t` 与 `k'_i` 的非零正支持集没有交集，则 `a_{t,i}=0`；
- 若有交集，则 score 为正；ReLU 并不保证每个 token 只匹配一个 token，也不产生
  “Top-1 硬选择”；
- causal 输出可写成
  `o_t = Σ_i a_{t,i}v_i / (Σ_i a_{t,i}+ε)`，即由非负 score 加权的 value 平均。

**旧概览中“query 与 key 方向相反 → 点积为负 → ReLU 截断”的说法不准确。** ReLU 在点积之前
已经分别作用于 Q/K，点积结果不可能为负。正确的直觉是“正特征支持集的交集决定是否开启”，
而不是把负点积截成零。

这也意味着“动态稀疏检索”需要加限定：当正支持集不交时 score 精确为零；当支持集重叠时，
并不存在由 softmax 产生的额外锐化。它是否真的学到可解释、可检索的门控，是效果 TODO，
不是公式自动保证的结论。

## 6. 训练与推理的两条等价路径

### 6.1 批量训练

同一层所有位置可以并行。代码当前执行两条 `scan_prefix_outer`：

1. 主扫描：累计 `M=Σv k'ᵀ`，并读出 `M q'` 作为分子；
2. z 扫描：把 V 换成全 1，累计 `z=Σk'`，读出 `q'·z` 作为分母的未加 ε 部分。

这样不必展开所有 token 对。扫描原语契约见
[`compute_engine.hpp`](../../include/neuralnet.cpp/compute_engine.hpp#L1230)。
当前两次扫描都会经过同一个通用原语，因此即使 RLA-2 忽略 A 输出，原语内部的 A 计算仍存在。

反向传播需要求“某个 key/value 被未来哪些 query 使用”。causal 路径按时间方向使用后缀扫描，
把梯度从后往前累计；当前实现还会用到列式外积来物化 `dL/dM`。反向公式与路径见
[`ReLULinearAttention::backward`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L545)。

### 6.2 逐 token 推理

增量路径执行相同算法：

```text
M_t = M_{t-1} + v_t (k'_t)ᵀ
z_t = z_{t-1} + k'_t
o_t = M_t q'_t / ((q'_t)ᵀz_t + 1e-4)
```

每层的每个 head 维护一份状态。代码按 `B_state=(d,d_k)`、`z_state=(d,1)` 组织整个 head 组。
`rapt_smoke_test` 用整序列 forward 的末位与逐 token 状态的末位对拍；这是路径一致性测试，
不是语言质量测试。

## 7. RAPT 如何组成一个语言模型

一个 RAPT block 是 Pre-Norm decoder block：

```text
h = x + RLA2(Norm(x))
y = h + FFN(Norm(h))
```

RAPTModel 的完整路线：

```text
token embedding
  → N × RAPTBlock
  → final Norm
  → LM head
  → logits
```

实现与默认配置：

| 组件 | 当前实现 |
|------|----------|
| 注意力 | `ReLULinearAttention`，默认 causal |
| block 外部 norm | 可配置；默认 LayerNorm |
| block 内 FFN | 可配置；默认 GeLU |
| 位置编码 | 公开 RAPT 工厂只接受 RoPE；施加在 Q/K 上、ReLU 之前 |
| 顶部 | final norm + Linear LM head |
| 掩码 | causal；可追加 document boundary |

类定义见 [`RAPTBlock`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L865) 与
[`RAPTModel`](../../include/neuralnet.cpp/compute_layer_rapt.hpp#L1060)。

## 8. 当前实现与原版 RLA 的关键差异

| 维度 | 原版 RLA（旧概览记录） | 当前 RLA-2 |
|------|-------------------------|------------|
| 分母 | `sqrt(Σ score²)+ε` | `Σ score+1e-4` |
| 第二矩状态 | 若用 L2 分母，需要 `Σk'k'ᵀ` 参与二次型 | 公式不消费 A；当前通用扫描仍内部计算 A，Layer 传 dummy A0 并忽略 A 输出 |
| 数值预处理 | 旧设计提到 LayerNorm/泄漏残差等 | Q/K 内置逐头 RMSNorm；无泄漏项 |
| ReLU | 固定逐元素 ReLU | 固定逐元素 ReLU |
| 位置衰减 | 旧设计提到可选 `γ^(t-i)` | 无 γ；RoPE 提供位置信息 |
| 推理状态 | 取决于原版分母设计 | `M` + `z` |
| 额外学习参数 | 取决于设计 | 没有新增可学习核参数 |

“无额外学习参数”不等于“零超参数”：至少还有 `1e-5` RMSNorm epsilon、`1e-4` 输出分母
epsilon、RoPE 基频 `10000`，以及 `d/H/L` 等结构超参数。

## 9. 哪些是事实，哪些仍需验证

| 命题 | 当前证据等级 |
|------|--------------|
| RLA-2 构造固定大小状态，序列维不物化 `L×L` score | **源码机械事实** |
| score 非负；正支持集不交时为 0 | **公式机械事实** |
| Sum 分母使输出形式上是带 ε 修正的非负加权平均 | **公式机械事实** |
| causal、bidirectional、document boundary 状态语义 | **源码 + gradcheck 覆盖** |
| 增量推理与整序列前向一致 | **smoke test 路径一致性** |
| RLA-2 的预测质量、长上下文能力、实际吞吐优势 | **未验证；见 02** |
| ReLU 门控一定比 softmax 更可解释或更擅长检索 | **未验证；见 02** |
| RLA 是本项目首创 | **本仓没有证据；需要单独文献检索** |

## 10. 本定义文档不能说明什么

- 它不能说明 RLA 在 TinyStories、代码、法律文本或任何真实语料上的 loss/perplexity。
- 它不能说明 RLA 与同参 GPT 的准确率、长上下文检索、吞吐、延迟或显存对比。
- 它不能证明“动态稀疏”在训练后仍保持稀疏，也不能证明人类能理解学到的正支持集。
- `rapt_test` 的 gradcheck、smoke 与 KV-cache 一致性只证明实现自洽，不证明模型值得采用。
- 旧概览中的经验性判断不能当实验证据；其与当前源码冲突处，以本文为准。

效果问题统一进入 [`02-rla-effectiveness.md`](02-rla-effectiveness.md)。
