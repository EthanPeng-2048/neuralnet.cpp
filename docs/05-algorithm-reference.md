# 📚 算法解析：Layer / Loss / Optimizer

> 本文档详细解析 neuralnet.cpp 中每个计算组件的数学原理、实现细节和引擎原语分解。

---

## 📐 约定

- **矩阵布局**：行主序 `(rows, cols)`，`data_[row * cols + col]`
- **批处理布局**：列主序 batch-major `(feature_dim, batch_size)`
- **标量类型**：`Scalar = float`
- **引擎原语**：所有算法通过 `ComputeEngine` 的 op-level 原语组合表达

---

## 🧱 Layer 篇

### 1. Linear — 全连接层

**参数：** `in_features`, `out_features`

**可学习参数：**
- `w_`: `(out_features, in_features)` — Xavier 均匀初始化
- `b_`: `(out_features, 1)` — 零初始化

**Forward：**

$$\text{out} = W \times x + b$$

引擎原语分解：
```
out = engine.matmul(W, x)                        // 矩阵乘法
engine.broadcast_row_inplace(out, b, Add)         // 按行广播加法
```

**Backward：**

$$\text{grad}_x = W^T \times \text{grad}_{\text{out}}$$
$$\text{grad}_W += \text{grad}_{\text{out}} \times x^T$$
$$\text{grad}_b += \sum_{\text{batch}} \text{grad}_{\text{out}}$$

引擎原语分解：
```
grad_input = engine.matmul(W, grad_out, transA=true)
grad_W += engine.matmul(grad_out, input, transB=true)
grad_b += engine.row_reduce_sum(grad_out)
```

---

### 2. ReLU — ReLU 激活函数

**参数：** 无

**Forward：**

$$\text{out} = \max(x, 0)$$

```
out = engine.elementwise_binary_scalar(Max, x, 0)
```

**Backward：**

$$\text{grad}_x = \begin{cases} \text{grad}_{\text{out}} & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

```
grad_input = engine.elementwise_select_scalar_cond(Gt, x, 0, grad_out, 0)
```

---

### 3. GeLU — QuickGeLU 激活函数

**参数：** `β = 1.702`（固定常量）

**Forward：**

$$\text{out} = x \cdot \sigma(\beta x) = x \cdot \frac{1}{1 + e^{-\beta x}}$$

引擎原语分解（5 次调用）：
```
t1 = engine.elementwise_binary_scalar(Mul, x, β)      // βx
t2 = engine.elementwise_unary(Neg, t1)                 // -βx
t3 = engine.elementwise_unary(Exp, t2)                 // exp(-βx)
t4 = engine.elementwise_binary_scalar(Add, t3, 1)      // 1 + exp(-βx)
s  = engine.elementwise_binary_scalar(Div, t4, 1)      // sigmoid(βx)
out = engine.elementwise_binary(Mul, x, s)
```

**Backward：**

$$\text{factor} = s \cdot (1 + \beta x \cdot (1 - s)), \quad s = \sigma(\beta x)$$
$$\text{grad}_x = \text{grad}_{\text{out}} \cdot \text{factor}$$

引擎原语分解（5 次调用）：
```
one_minus_s = engine.elementwise_binary_scalar(Sub, s, 1)
bx = engine.elementwise_binary_scalar(Mul, x, β)
inner = engine.elementwise_binary(Mul, bx, one_minus_s)
paren = engine.elementwise_binary_scalar(Add, inner, 1)
factor = engine.elementwise_binary(Mul, s, paren)
grad_input = engine.elementwise_binary(Mul, grad_out, factor)
```

---

### 4. LayerNorm — 层归一化

**参数：** `normalized_shape`, `epsilon = 1e-5`

**可学习参数：**
- `gamma_`: `(normalized_shape, 1)` — 初始化为 1
- `beta_`: `(normalized_shape, 1)` — 初始化为 0

**Forward：**

$$\mu = \frac{1}{F} \sum_f x_{f,b}$$
$$\text{diff} = x - \mu$$
$$\sigma^2 = \frac{1}{F} \sum_f \text{diff}_f^2$$
$$\text{normalized} = \frac{\text{diff}}{\sqrt{\sigma^2 + \epsilon}}$$
$$\text{out} = \gamma \cdot \text{normalized} + \beta$$

**Backward：**

$$\text{gy} = \text{grad}_{\text{out}} \cdot \gamma$$
$$\text{mean}_g = \frac{1}{F} \sum_f \text{gy}_f$$
$$\text{gy}_{\text{norm}} = \text{gy} \cdot \text{normalized}$$
$$\text{mean}_{gn} = \frac{1}{F} \sum_f \text{gy}_{\text{norm},f}$$
$$\text{grad}_x = (\text{gy} - \text{mean}_g - \text{normalized} \cdot \text{mean}_{gn}) \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}}$$
$$\text{grad}_\gamma += \sum_b \text{gy}_{\text{norm},b}$$
$$\text{grad}_\beta += \sum_b \text{grad}_{\text{out},b}$$

---

### 5. Softmax — 行级 Softmax

**参数：** 无

**Forward（数值稳定）：**

$$\text{row\_max}_r = \max_c x_{r,c}$$
$$\text{shifted}_{r,c} = x_{r,c} - \text{row\_max}_r$$
$$\text{exp\_shift}_{r,c} = e^{\text{shifted}_{r,c}}$$
$$\text{row\_sum}_r = \sum_c \text{exp\_shift}_{r,c}$$
$$\text{out}_{r,c} = \frac{\text{exp\_shift}_{r,c}}{\text{row\_sum}_r}$$

引擎原语分解：
```
row_max = engine.row_reduce_max(input)
shifted = engine.clone(input)
engine.broadcast_row_inplace(shifted, row_max, Sub)
exp_shift = engine.elementwise_unary(Exp, shifted)
row_sum = engine.row_reduce_sum(exp_shift)
output = engine.clone(exp_shift)
engine.broadcast_row_inplace(output, row_sum, Div)
```

**Backward：**

$$\text{ep} = \text{out} \odot \text{grad}_{\text{out}}$$
$$\text{dot}_r = \sum_c \text{ep}_{r,c}$$
$$\text{gmd} = \text{grad}_{\text{out}} - \text{dot}$$
$$\text{grad}_x = \text{out} \odot \text{gmd}$$

---

### 6. MultiHeadAttention — 多头注意力

**参数：** `d_model`, `num_heads`, `seq_len`

**可学习参数：** 4 个 Linear 层（`w_q`, `w_k`, `w_v`, `w_o`），每个 `(d_model, d_model)`

**Forward（批量化）：**

1. 线性投影：`Q = W_q·x`, `K = W_k·x`, `V = W_v·x` → `(H·d_k, batch·seq)`
2. 重排：`Q_re = rearrange_3d(Q, H·d_k, batch, seq)` → `(batch·H·d_k, seq)`
3. 注意力分数：`S = batched_matmul(Q_re, K_re, batch·H, transA=true)` → `(batch·H·seq, seq)`
4. 缩放：`S *= scale`，其中 `scale = 1/√d_k`
5. Softmax：`A = softmax(S)`
6. 注意力输出：`O_re = batched_matmul(V_re, A, batch·H)` → `(batch·H·d_k, seq)`
7. 反重排：`O = rearrange_3d(O_re, H·d_k, batch, seq, inverse=true)` → `(H·d_k, batch·seq)`
8. 输出投影：`out = W_o·O`

**性能关键：** `rearrange_3d` + `batched_matmul` 将 H 个 per-head matmul 融合为 1 次 batch dispatch。

---

### 7. PositionalEncoding — 正弦波位置编码

**参数：** `d_model`, `max_len`

**算法（固定，不可学习）：**

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$
$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**Forward：** `out = input + PE`

**Backward：** 梯度直接穿透（编码不可学习）

**Tiling 模式：** 当 `tile_size > 0` 时，生成 `(d_model, tile_size)` 基础编码，沿列方向 tile 为 `(d_model, batch·tile_size)`，每个样本独立使用同一份编码。

---

### 8. FeedForward — 前馈网络

**参数：** `d_model`, `d_ff`

**结构：** `FFN(x) = Linear₂(GeLU(Linear₁(x)))`

- `fc1_`: `(d_model, d_ff)` — 扩展
- `fc2_`: `(d_ff, d_model)` — 压缩

Forward/Backward 直接委托给子层的 forward/backward。

---

### 9. TransformerEncoderLayer — Pre-Norm 编码器层

**参数：** `d_model`, `num_heads`, `d_ff`, `seq_len`

**算法（Pre-Norm）：**

$$x_1 = \text{LN}_1(x)$$
$$a = \text{SelfAttn}(x_1)$$
$$r_2 = x + a$$
$$x_2 = \text{LN}_2(r_2)$$
$$f = \text{FFN}(x_2)$$
$$\text{out} = r_2 + f$$

包含子层：`MultiHeadAttention` + `LayerNorm`×2 + `FeedForward`

---

### 10. TransformerEncoder — ViT 编码器

**参数：** `d_model`, `num_heads`, `d_ff`, `num_layers`, `num_patches`

**算法：**

1. 添加 tiled 位置编码
2. 通过 N 个 `TransformerEncoderLayer`
3. 全局平均池化（按样本聚合 `num_patches` 维度）

**池化：**
$$\text{out}[:, b] = \frac{1}{P} \sum_{p=0}^{P-1} x[:, b \cdot P + p]$$

**反池化（backward）：**
$$\text{grad}_x[:, b \cdot P + p] = \frac{1}{P} \cdot \text{grad}_{\text{out}}[:, b]$$

---

### 11. PatchEmbedding — 图像 patch 嵌入

**参数：** `img_size`, `patch_size`, `d_model`

**算法：**

1. 将 `(img_size², batch)` 图像提取 `num_patches = (img_size/patch_size)²` 个不重叠 patch
2. 每个展平 patch `(patch_size²)` 经 Linear 投影到 `d_model` 维
3. 重排为 `(d_model, batch·num_patches)` — batch-major 列布局

---

### 12. CausalSelfAttention — 因果自注意力

**参数：** `d_model`, `num_heads`, `max_len`, `seq_len`

与 `MultiHeadAttention` 相同的批量化策略，额外施加因果掩码：

$$\text{mask}[i][j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

掩码在 softmax 前施加：`S += mask`

**掩码缓存：** 相同 `(batch, seq_len)` 组合只构造一次。

---

### 13. GPTBlock — Pre-Norm 解码器块

**参数：** `d_model`, `num_heads`, `d_ff`, `max_len`, `seq_len`

**算法（Pre-Norm）：**

$$x = x + \text{CausalSelfAttn}(\text{LN}_1(x))$$
$$x = x + \text{FFN}(\text{LN}_2(x))$$

包含子层：`CausalSelfAttention` + `LayerNorm`×2 + `FeedForward`

---

### 14. GPTModel — Decoder-only Transformer

**参数：** `vocab_size`, `d_model`, `seq_len`, `num_heads`, `d_ff`, `num_layers`

**可学习参数：**
- `token_emb_`: `(vocab_size, d_model)` — Token 嵌入
- `pos_emb_`: `(seq_len, d_model)` — 位置嵌入
- N 个 `GPTBlock`
- `ln_f_`: `LayerNorm` — 最终归一化
- `lm_head_`: `Linear(d_model, vocab_size)` — 语言模型头

**Forward：**

1. `all_emb = gather_rows(token_emb_, input)` — 按 token ID 查表
2. `pos_gathered = gather_rows(pos_emb_, pos_indices)` — 位置嵌入
3. `x = transpose(all_emb) + transpose(pos_gathered)` — `(d_model, seq·batch)`
4. 通过 N 个 GPTBlock
5. 最终 LayerNorm
6. LM Head → `(vocab_size, seq·batch)`

**Backward 关键：**
- Token 嵌入梯度用 `scatter_add_rows` 按 token ID 累加
- 位置嵌入梯度用 CPU 辅助聚合（数据量小）

---

## 📉 Loss 篇

### 1. MSELoss — 均方误差

**算法：**

$$\text{diff} = \text{pred} - \text{target}$$
$$\text{loss} = \frac{1}{N} \sum \text{diff}^2$$
$$\text{grad} = \frac{2}{N} \cdot \text{diff}$$

---

### 2. CrossEntropyLoss — 交叉熵损失（含 Softmax）

**算法（数值稳定）：**

$$\text{col\_max} = \text{col\_reduce\_max}(\text{logits})$$
$$\text{shifted} = \text{logits} - \text{col\_max}$$
$$\text{exp\_shift} = e^{\text{shifted}}$$
$$\text{col\_sum} = \text{col\_reduce\_sum}(\text{exp\_shift})$$
$$\text{softmax} = \text{exp\_shift} / \text{col\_sum}$$
$$\text{grad} = \text{softmax} - \text{target}$$
$$\text{log\_sm} = \text{shifted} - \log(\text{col\_sum})$$
$$\text{loss} = -\frac{1}{\text{batch}} \sum \text{target} \cdot \text{log\_sm}$$

**Backward：** `grad = softmax - target_onehot`

---

## ⚙️ Optimizer 篇

### 1. SGD — 随机梯度下降

**算法：**

$$p \leftarrow p - \eta \cdot g$$

**引擎原语：** `scale_add_(p, -lr, g)` — 单次 axpy

---

### 2. SGDWithMomentum — 带动量的 SGD

**算法：**

$$v \leftarrow \beta \cdot v + (1 - \beta) \cdot g$$
$$p \leftarrow p - \eta \cdot v$$

**引擎原语：**
```
engine.scale_inplace(v, β)           // v *= β
scale_add_(v, 1-β, g)               // v += (1-β)*g
scale_add_(p, -lr, v)               // p -= lr*v
```

**参数：** `lr`（学习率）, `beta = 0.9`（动量系数）

---

### 3. Adam — 自适应矩估计

**算法：**

$$m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g$$
$$v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$$
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$
$$p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

**引擎原语：**
```
// m = β1*m + (1-β1)*g
engine.scale_inplace(m, β1)
scale_add_(m, 1-β1, g)

// v = β2*v + (1-β2)*g²
engine.scale_inplace(v, β2)
auto g_sq = engine.elementwise_binary(Mul, g, g)
scale_add_(v, 1-β2, g_sq)

// 偏差修正 + 更新
auto m_hat = engine.scale(m, inv_bc1)
auto v_hat = engine.scale(v, inv_bc2)
auto sqrt_v = engine.elementwise_unary(Sqrt, v_hat)
auto denom = engine.elementwise_binary_scalar(Add, sqrt_v, eps)
auto step = engine.elementwise_binary(Div, m_hat, denom)
engine.scale_inplace(step, -lr)
engine.add_inplace(p, step)
```

**参数：** `lr`, `β1=0.9`, `β2=0.999`, `ε=1e-8`

**偏差修正：**
- `bc1 = 1 - β1^t`（一阶矩偏差修正）
- `bc2 = 1 - β2^t`（二阶矩偏差修正）

---

### 4. AdamW — 解耦权重衰减

**算法：**

$$p \leftarrow (1 - \eta \cdot \lambda) \cdot p \quad \text{（权重衰减，独立于梯度）}$$
$$m \leftarrow \beta_1 \cdot m + (1 - \beta_1) \cdot g$$
$$v \leftarrow \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$$
$$p \leftarrow p - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

**与 Adam 的区别：**
- Adam 的 L2 正则化：`g += λ·p` → 权重衰减被自适应学习率稀释
- AdamW 的解耦衰减：`p *= (1-η·λ)` → 权重衰减对所有参数等效

**参数：** `lr`, `β1=0.9`, `β2=0.999`, `ε=1e-8`, `weight_decay=0.01`

---

### 5. Muon — MomentUm Orthogonalized by Newton-Schulz

**算法（Keller Jordan et al., 2024）：**

对于每个 2D 参数 $p$（权重矩阵）：

1. **SGD-Momentum：** $v \leftarrow \mu \cdot v + g$
2. **Newton-Schulz 正交化：** $\text{update} = \text{NS}_5(v)$
3. **参数更新：** $p \leftarrow p - \eta \cdot \text{update}$

对于非 2D 参数（bias 等）：标准 SGD 更新。

#### Newton-Schulz 正交化

**目标：** 计算矩阵 $G$ 的最近半正交矩阵 $\text{Ortho}(G) = UV^T$（$G = USV^T$ 为 SVD）。

**算法：**

$$X = \frac{G}{\|G\|_F + \epsilon}$$

重复 5 次：
$$A = X \cdot X^T$$
$$B = b \cdot A + c \cdot A^2$$
$$X = a \cdot X + B \cdot X$$

**调优系数：** `a = 3.4445`, `b = -4.7750`, `c = 2.0315`

这些系数使 quintic 多项式 $\phi^N(x) \to 1$ for $x \in [0,1]$，5 步内收敛。

**参考：**
- Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks"
- https://kellerjordan.github.io/posts/muon/

**引擎原语（每步）：**
```
A = engine.matmul(X, X, transB=true)    // X × X^T
A_sq = engine.matmul(A, A)              // A²
B = engine.scale(A, b) + engine.scale(A_sq, c)
BX = engine.matmul(B, X)
X = engine.scale(X, a) + BX
```

**参数：** `lr`, `momentum=0.95`, `nesterov=true`, `ns_steps=5`, `ns_eps=1e-7`

---

## 🏭 优化器工厂函数

```cpp
auto optimizer = nn::create_optimizer(
    name, engine, params, grads, lr, weight_decay);
```

| `name` | 创建的优化器 | 额外参数 |
|--------|-------------|----------|
| `"sgd"` | `SGD` | — |
| `"sgd_momentum"` | `SGDWithMomentum` | — |
| `"adam"` | `Adam` | — |
| `"adamw"` | `AdamW` | `weight_decay` |
| `"muon"` | `Muon` | — |
| 默认 | `Adam` | — |

---

## 📊 各组件复杂度对比

| 组件 | 参数量 | Forward FLOPs (per sample) | Backward FLOPs |
|------|--------|---------------------------|----------------|
| `Linear(in, out)` | in×out + out | 2×in×out | 4×in×out |
| `ReLU` | 0 | N | N |
| `GeLU` | 0 | ~5N | ~5N |
| `LayerNorm(F)` | 2F | ~5F | ~8F |
| `Softmax` | 0 | ~3S | ~3S |
| `MHA(d, h, s)` | 4d² | ~8d²s + 2ds² | ~16d²s + 4ds² |
| `FeedForward(d, f)` | 2df + d + f | 4df | 8df |
| `GPTBlock(d, h, f, s)` | — | MHA + FFN + 4LN | MHA + FFN + 4LN |

其中：N = 元素数, F = 特征维度, S = 序列长度, d = d_model, h = num_heads, f = d_ff
