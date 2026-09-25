# 计算引擎使用指南

**面向想要使用计算引擎构建神经网络的开发者。**

---

## 目录

1. [快速入门](#快速入门)
2. [引擎选择](#引擎选择)
3. [张量操作](#张量操作)
4. [矩阵运算](#矩阵运算)
5. [归约与广播](#归约与广播)
6. [逐元素运算](#逐元素运算)
7. [数据操作](#数据操作)
8. [表达式融合](#表达式融合)
9. [批处理控制](#批处理控制)
10. [实际示例](#实际示例)
11. [性能建议](#性能建议)
12. [常见问题](#常见问题)

---

## 快速入门

### 1. 包含头文件

```cpp
#include <neuralnet.cpp/nn.hpp>
```

### 2. 创建引擎

```cpp
// CPU 引擎
nn::CpuEngine engine;

// GPU 引擎（需要 Vulkan 支持）
nn::GpuEngine gpu_engine;
```

### 3. 创建张量

```cpp
// 从 Matrix 创建
nn::Matrix mat(3, 4);  // 3行4列
// ... 初始化 mat ...

auto tensor = engine.from_matrix(mat);

// 直接创建空张量
auto empty = engine.create_tensor(3, 4);
```

### 4. 基本运算

```cpp
// 矩阵乘法
auto result = engine.matmul(*a, *b);

// 逐元素加法
engine.add_inplace(*result, *bias);

// 标量缩放
engine.scale_inplace(*result, 0.5f);
```

### 5. 获取结果

```cpp
auto matrix = engine.to_matrix(*result);
// matrix 现在是 CPU Matrix，可以访问数据
```

---

## 引擎选择

### CPU vs GPU

| 特性 | CPU 引擎 | GPU 引擎 |
|------|----------|----------|
| **设备** | CPU 内存 | GPU 显存 |
| **并行性** | 多核 CPU | 数千 Vulkan 计算核心 |
| **内存带宽** | ~50 GB/s | ~1 TB/s |
| **延迟** | 低 | 高（PCIe 传输） |
| **适用场景** | 小模型、调试 | 大模型、训练 |

### 选择建议

```cpp
// 根据模型大小选择
if (model_params < 10'000'000) {  // < 10M 参数
    nn::CpuEngine engine;  // CPU 足够
} else {
    nn::GpuEngine engine;  // 需要 GPU
}
```

### 条件编译

```cpp
#ifdef NN_HAS_VULKAN
    nn::GpuEngine engine;
#else
    nn::CpuEngine engine;
#endif
```

---

## 张量操作

### 创建张量

```cpp
// 从 Matrix 创建（拷贝）
nn::Matrix mat(2, 3);
auto tensor = engine.from_matrix(mat);

// 创建空张量
auto empty = engine.create_tensor(2, 3);
```

### 访问信息

```cpp
tensor->rows();    // 行数
tensor->cols();    // 列数
tensor->device();  // 设备类型 (CPU/GPU)
tensor->is_cpu();  // 是否在 CPU
```

### 设备转换

```cpp
// CPU → GPU（通过 engine.from_matrix）
auto gpu_tensor = engine.from_matrix(cpu_matrix);

// GPU → CPU（通过 engine.to_matrix）
auto cpu_matrix = engine.to_matrix(*gpu_tensor);
```

### 深拷贝

```cpp
auto copy = engine.clone(*tensor);
```

---

## 矩阵运算

### 矩阵乘法

```cpp
// C = A × B
auto c = engine.matmul(*a, *b);

// C = A^T × B
auto c = engine.matmul(*a, *b, /*transA=*/true);

// C = A × B^T
auto c = engine.matmul(*a, *b, /*transA=*/false, /*transB=*/true);
```

**形状要求**：
- A: `(M, K)`
- B: `(K, N)`
- 输出: `(M, N)`

### 批量矩阵乘法

```cpp
// 批量乘法：对每个 batch 计算 C_b = A_b × B_b
auto c = engine.batched_matmul(*a, *b, /*batch=*/4);

// 带缩放系数
auto c = engine.batched_matmul(*a, *b, /*batch=*/4, 
                               /*transA=*/false, /*transB=*/false,
                               /*alpha=*/0.5f);
```

**形状要求**：
- A: `(batch * A_rows, A_cols)`
- B: `(batch * B_rows, B_cols)`
- 输出: `(batch * M, N)`

**典型用途**：多头注意力

```cpp
// Q, K, V: (H*d_k, batch*seq)
auto attn = engine.batched_matmul(*q, *k, /*batch=*/batch * heads);
```

### 就地加法

```cpp
// A += B
engine.add_inplace(*a, *b);
```

### 就地缩放

```cpp
// A *= scalar
engine.scale_inplace(*a, 0.1f);
```

### 融合 axpy

```cpp
// A += scalar * B
engine.axpy_inplace(*a, 0.01f, *grad);
```

**优势**：比 `clone + scale + add` 更高效。**注**：项目内当前更推荐一条 DSL 表达式
（`dsl::compute_into(engine, leaf(a) + leaf(b) * rparam(s), a)`）——optimizer 即此写法，
`axpy_inplace` 目前无生产调用方（见 `development/12-compute-engine-inventory.md` §2.2）。

### 置零

```cpp
engine.zero(*tensor);
```

---

## 归约与广播

### 归约操作

```cpp
// 按行求和: (rows, cols) → (rows, 1)
auto row_sum = engine.row_reduce_sum(*tensor);

// 按列求和: (rows, cols) → (1, cols)
auto col_sum = engine.col_reduce_sum(*tensor);

// 按行求最大值: (rows, cols) → (rows, 1)
auto row_max = engine.row_reduce_max(*tensor);

// 按列求最大值: (rows, cols) → (1, cols)
auto col_max = engine.col_reduce_max(*tensor);
```

**典型用途**：

```cpp
// LayerNorm: 归一化每行
auto mean = engine.row_reduce_sum(*x);
engine.scale_inplace(*mean, 1.0f / x->cols());

// Softmax: 归一化每列
auto max_val = engine.col_reduce_max(*logits);
```

### 广播操作

```cpp
// 按行广播: A (R, C) op= row_vec (R, 1)
engine.broadcast_row_inplace(*a, *row_vec, BinaryOp::Add);

// 按列广播: A (R, C) op= col_vec (1, C)
engine.broadcast_col_inplace(*a, *col_vec, BinaryOp::Mul);
```

**典型用途**：

```cpp
// 加偏置: output += bias (每行加同一个偏置)
engine.broadcast_row_inplace(*output, *bias, BinaryOp::Add);

// 缩放: output *= scale (每列乘同一个缩放)
engine.broadcast_col_inplace(*output, *scale, BinaryOp::Mul);
```

---

## 逐元素运算

> 单步调试/小工具用下面的 eager 原语即可；**生产代码组合多个运算时优先写一条
> `dsl::compute` 表达式**（见"表达式融合"）——一次 dispatch、无中间 Tensor。

### 一元运算

```cpp
// out = -a
auto neg = engine.elementwise_unary(UnaryOp::Neg, *a);

// out = exp(a)
auto exp = engine.elementwise_unary(UnaryOp::Exp, *a);

// out = log(a)
auto log = engine.elementwise_unary(UnaryOp::Log, *a);

// out = sqrt(a)
auto sqrt = engine.elementwise_unary(UnaryOp::Sqrt, *a);

// out = 1/sqrt(a)
auto rsqrt = engine.elementwise_unary(UnaryOp::Rsqrt, *a);

// out = abs(a)
auto abs = engine.elementwise_unary(UnaryOp::Abs, *a);

// out = tanh(a)
auto tanh = engine.elementwise_unary(UnaryOp::Tanh, *a);
```

### 二元运算

```cpp
// out = a + b
auto add = engine.elementwise_binary(BinaryOp::Add, *a, *b);

// out = a - b
auto sub = engine.elementwise_binary(BinaryOp::Sub, *a, *b);

// out = a * b
auto mul = engine.elementwise_binary(BinaryOp::Mul, *a, *b);

// out = a / b
auto div = engine.elementwise_binary(BinaryOp::Div, *a, *b);

// out = max(a, b)
auto max = engine.elementwise_binary(BinaryOp::Max, *a, *b);

// out = min(a, b)
auto min = engine.elementwise_binary(BinaryOp::Min, *a, *b);
```

### 标量二元运算

```cpp
// out = a + scalar
auto add = engine.elementwise_binary_scalar(BinaryOp::Add, *a, 1.0f);

// out = scalar + a
auto add = engine.elementwise_binary_scalar(BinaryOp::Add, *a, 1.0f, /*scalar_first=*/true);

// out = a * scalar
auto mul = engine.elementwise_binary_scalar(BinaryOp::Mul, *a, 0.1f);
```

### 条件选择

```cpp
// out = (a > 0) ? then_t : 0.0f
auto relu_grad = engine.elementwise_select_scalar_cond(
    CompareOp::Gt, *x, 0.0f, *grad, 0.0f);
```

**典型用途**：ReLU 反向传播。
**注**：ReLU 层实际写法是 DSL `dsl::select(leaf(x) > 0, leaf(g), 0)`（单表达式融合）；
本原语当前无生产调用方，接口保留。

---

## 数据操作

### 行切片

```cpp
// 切片: src 的行 [start_row, start_row + count)
auto slice = engine.slice_rows(*src, /*start_row=*/10, /*count=*/5);
```

**形状**：
- 输入: `(rows, cols)`
- 输出: `(count, cols)`

### 行插入

```cpp
// 插入: src 的行写入 dst 的行 [dst_start_row, ...)
engine.insert_rows(*dst, /*dst_start_row=*/10, *src);
```

### 行收集（Embedding 查表）

```cpp
// gather_rows: 按 indices 从 table 查表
auto embedding = engine.gather_rows(*table, *indices);
```

**形状**：
- table: `(vocab_size, embedding_dim)`
- indices: `(num_indices,)` 或任意形状
- 输出: `(num_indices, embedding_dim)`

**典型用途**：Token Embedding

```cpp
// token_ids: (batch, seq)
auto embeddings = engine.gather_rows(*weight, *token_ids);
```

### 行散列累加（Embedding 梯度）

```cpp
// scatter_add_rows: 按 indices 把 grad 累加到 dst
engine.scatter_add_rows(*grad_weight, *indices, *grad_embedding);
```

**典型用途**：Embedding 反向传播

### 3D 重排

```cpp
// rearrange_3d: (M, B, N) ↔ (B, M, N)
auto rearranged = engine.rearrange_3d(*x, M, B, N, /*inverse=*/false);
```

**典型用途**：多头注意力维度重排

```cpp
// Q: (H*d_k, batch*seq) → (batch*H*d_k, seq)
auto q_rearranged = engine.rearrange_3d(*q, heads * d_k, batch, seq);
```

### 转置

```cpp
auto transposed = engine.transpose(*tensor);
```

**形状**：
- 输入: `(rows, cols)`
- 输出: `(cols, rows)`

---

## 表达式融合

> **签名**：`nn::dsl::compute(engine, expr, rows, cols[, P])` —— `expr` 由 `dsl::leaf(tensor)`
> 引用输入张量、`dsl::rparam(v)` 引用标量常量拼成普通数学表达式（没有 lambda + 输入列表的写法）。
> CPU 编译期内联求值，GPU 折叠为 `ExprSpec` 匹配预编译融合 shader。

### 基本用法

```cpp
#include <neuralnet.cpp/expr_dsl.hpp>

// out = a + b * c  →  单个融合 kernel，中间量不落显存
auto result = nn::dsl::compute(
    engine,
    nn::dsl::leaf(tensor_a) + nn::dsl::leaf(tensor_b) * nn::dsl::leaf(tensor_c),
    rows, cols
);
```

### 原地更新（compute_into）

```cpp
// 把结果写进既有张量，不分配新 Tensor：
nn::dsl::compute_into(engine, nn::dsl::leaf(dst) + nn::dsl::leaf(other) * nn::dsl::rparam(k), dst);
```

### ~~块式融合~~（已移除）

> 2026-09-19：`engine.begin_expr()` / `engine.end_expr()` 已随 IR-C 整体删除（无收益点，见
> `docs/development/03-ir-optimization.md` §5.3）。需要多步逐元素变换时，直接写成**一个**
> `dsl::compute` 表达式即可（单个 GPU 融合 kernel）。

### 典型应用

#### GeLU 激活

```cpp
using namespace nn::dsl;
auto gelu = compute(engine,
    leaf(input) / (Scalar{1} + exp(-(leaf(input) * 1.702f))),
    rows, cols);
```

#### RoPE 位置编码

```cpp
using namespace nn::dsl;
// LLaMA half-swap：out = q·cos + rotate_half(q)·sin（dk = d_k，行表按 dk 交错）
auto rope = compute(engine,
    leaf(q) * row_mod(cos_cache, dk) + rotate_half(q, dk) * row_mod(sin_cache, dk),
    rows, cols);
```

---

## 批处理控制

### CPU 引擎

```cpp
// CPU 为 no-op，操作立即执行
engine.begin_batch();
// ... 操作 ...
engine.end_batch();  // 无实际效果
```

### GPU 引擎

```cpp
// GPU 录制 command buffer
engine.begin_batch();
// ... 操作（录制到 command buffer） ...
engine.end_batch();  // 提交 + 等待完成
```

### 中间刷新（防 TDR）

```cpp
engine.begin_batch();
// ... 前向传播 ...
engine.flush_batch();  // 提交当前操作，避免超时
// ... 反向传播 ...
engine.end_batch();
```

---

## 实际示例

> 以下对应层的**真实实现**见 `include/neuralnet.cpp/compute_layer_mlp.hpp`。

### 示例 1：线性层前向传播

```cpp
// out = W × x + b —— matmul + 按行广播 bias 一条原语
auto output = engine.matmul_with_bias(*weight_, input, *bias_, /*transA=*/false, /*transB=*/false);
```

### 示例 2：ReLU 激活

```cpp
// 单表达式融合：out = max(x, 0)
auto output = nn::dsl::compute(engine, nn::dsl::max(nn::dsl::leaf(input), Scalar{0}),
                               input.rows(), input.cols());
// backward 同理：grad = select(leaf(x) > 0, leaf(grad_out), 0)
```

### 示例 3：LayerNorm

真实实现（`compute_layer_mlp.hpp`）拆成 4 条 DSL 表达式（每条一个融合 kernel）：

```cpp
// 1) mean / diff（归约视图）
// 2) var = col_reduce_sum(diff*diff) / F
// 3) normalized = diff * rsqrt(var + eps)      ← 一条 compute
// 4) out = normalized * row_broadcast(gamma) + row_broadcast(beta)
// backward：gy = grad*gamma；grad_x = (gy - mean(gy) - norm*mean(gy*norm)) * rsqrt(var+eps)
```

用旧式多原语逐步写（能跑，但每步一次 dispatch + 一个中间 Tensor，仅作对照）：

```cpp
auto mean = engine.row_reduce_sum(input);
engine.scale_inplace(*mean, 1.0f / input.cols());
auto centered = engine.clone(input);
engine.broadcast_col_inplace(*centered, *mean, BinaryOp::Sub);
// ...（后续方差/归一化/仿射变换同理）
```

### 示例 4：多头注意力

> **真实实现已演进**：注意力 forward 现为**单 fold kernel**（QKᵀ/掩码/online softmax/ΣwV
> 分块流式，S 矩阵绝不物化，见 `compute_layer_attention.hpp`），不再是下面这条
> "batched_matmul → softmax → batched_matmul" 的物化链。下例保留作**结构示意**：

```cpp
// 1. 线性投影 Q/K/V → 2. rearrange_3d 到 (batch*H*d_k, seq)
// 3. QKᵀ（transA）+ 1/√d_k 缩放 → 4. softmax → 5. ×V → 6. rearrange 回去 → 7. 输出投影
// 真实 forward 由 AttentionBase::forward 走 fold 表达式完成；
// backward = R/X 两条 DSL 表达式 + 3 个 batched_matmul。
```

---

## 性能建议

### 1. 使用表达式融合

```cpp
// ❌ 不推荐：多次 kernel launch
auto temp1 = engine.elementwise_unary(UnaryOp::Exp, *x);
auto temp2 = engine.elementwise_binary(BinaryOp::Add, *temp1, *y);
auto result = engine.elementwise_binary(BinaryOp::Mul, *temp2, *z);

// ✅ 推荐：融合为单次调用
auto result = nn::dsl::compute(engine,
    (nn::dsl::exp(nn::dsl::leaf(x)) + nn::dsl::leaf(y)) * nn::dsl::leaf(z),
    rows, cols);
```

### 2. 避免不必要的拷贝

```cpp
// ❌ 不推荐：多次拷贝
auto a_copy = engine.clone(*a);
auto b_copy = engine.clone(*b);
engine.add_inplace(*a_copy, *b_copy);

// ✅ 推荐：就地操作
engine.add_inplace(*a, *b);
```

### 3. 使用批量操作

```cpp
// ❌ 不推荐：循环调用
for (int i = 0; i < batch; ++i) {
    auto q_i = engine.slice_rows(*q, i * seq, seq);
    auto k_i = engine.slice_rows(*k, i * seq, seq);
    auto score = engine.matmul(*q_i, *k_i);
    // ...
}

// ✅ 推荐：批量操作
auto scores = engine.batched_matmul(*q, *k, batch);
```

### 4. 合理使用批处理控制

```cpp
// GPU：批量提交减少开销
engine.begin_batch();
for (int i = 0; i < num_layers; ++i) {
    // 前向传播
}
engine.end_batch();

// 大模型：中间刷新防 TDR
engine.begin_batch();
// 前向传播
engine.flush_batch();
// 反向传播
engine.end_batch();
```

### 5. 内存预分配

```cpp
// 预分配中间张量
auto temp1 = engine.create_tensor(rows, cols);
auto temp2 = engine.create_tensor(rows, cols);

for (int step = 0; step < num_steps; ++step) {
    // 复用预分配的张量
    // ...
}
```

---

## 常见问题

### Q1: 为什么 `from_matrix` 返回的是智能指针？

**A**: `from_matrix` 返回 `Result<Tensor>`，需要解引用使用：

```cpp
auto tensor = engine.from_matrix(matrix);
auto& t = *tensor;  // 解引用
```

### Q2: 如何处理 GPU 内存不足？

**A**:
1. 减小 batch size
2. 使用梯度检查点
3. 使用 activation offload
4. 检查是否有内存泄漏

### Q3: 为什么 GPU 比 CPU 慢？

**A**: 可能原因：
1. 模型太小，GPU 开销大于收益
2. 频繁的 CPU↔GPU 数据传输
3. 没有使用批处理控制
4. 没有使用表达式融合

### Q4: 如何调试数值问题？

**A**:
1. 使用 `gradcheck` 验证梯度
2. 检查 NaN/Inf
3. 使用小模型测试
4. 对比 CPU 和 GPU 结果

### Q5: 如何添加自定义操作？

**A**: 参考 `docs/development/01-compute-engine-development.md` 的"添加新原语"部分。

---

## 相关文档

- **引擎开发**：`docs/development/01-compute-engine-development.md`
- **架构设计**：`docs/introduction/01-architecture.md`
- **算法参考**：`docs/introduction/03-algorithm-reference.md`
- **性能优化**：`docs/introduction/02-performance.md`
- **快速开始（训练与推理）**：`docs/usage/02-quickstart-train-infer.md`

---

*最后更新：2026-09-25（DSL 签名与示例对齐当前代码；接口清单见 `docs/development/12-compute-engine-inventory.md`）*