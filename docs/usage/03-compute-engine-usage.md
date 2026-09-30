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

### 批量读写（I/O 分组本体，M3）

```cpp
// 批量写：覆盖张量既有存储（不替换对象）；span 元素类型必须与张量精度精确匹配
std::vector<float> data(rows * cols);   // 填好数据
engine.write(*t, std::span(data));      // F32 张量收 float；F16 张量收 nn::f16

// 批量读：张量 → span（GPU 上隐含 flush + 同步，与 to_matrix 同路）
std::vector<float> back(rows * cols);
engine.read(*t, std::span(back));

// 索引级是语法糖：宿主直读写；GPU = 整张批量往返
auto v = engine.get_index(*t, 0, 0);    // → Scalar（f16 存储精确提升）
engine.set_index(*t, 0, 0, 0.5f);       // f16 存储按 round-half-to-even 舍入
```

> **U2 精确匹配**：`span<float>` 只配 F32 张量、`span<nn::f16>` 只配 F16 张量——错配返回
> `Result` 错误，元素类型非 float/f16 是编译期错误（防 f16/f32 槽错位）。`from_matrix/to_matrix`
> 仍用于 Matrix 宿主载体的整批进出（数据集/对拍/落盘）；`copy_from` 保留 f32 Matrix 转换填充
> （序列化加载）。**热路径请攒批量**，逐元素循环不要用 index API（GPU 每次一整轮 staging 往返）。
> 录制窗口内 `read/write` 的调用约定（隐含 flush、写入张量须存活到 `end_batch` 之后）见
> `compute_engine.hpp` I/O 分组注释。

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

### 融合 axpy（`dst += scalar * B`）

引擎不提供 axpy 原语，原地融合一律走表达式目标传递（单条融合表达式 + 零分配原地写回）：

```cpp
// A += scalar * B
(void)dsl::compute_into(engine, dsl::leaf(*a) + dsl::leaf(*b) * dsl::rparam(s), *a);
```

optimizer 即此写法（见 `compute_optimizer.hpp`）。

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

// 按行求最大值: 没有对应的引擎算子，用 DSL 归约叶子
//   auto row_max = dsl::compute_reduce(engine, dsl::row_reduce_max(*tensor), rows, cols);

// 按列求最大值: (rows, cols) → (1, cols)
auto col_max = engine.col_reduce_max(*tensor);
```

**典型用途**：

```cpp
// LayerNorm: 沿特征维（行）求均值 → (1, batch)，再乘 1/F
auto mean = engine.col_reduce_sum(*x);
engine.scale_inplace(*mean, 1.0f / x->rows());

// Softmax: 沿列求最大值（数值稳定的减法基准）
auto max_val = engine.col_reduce_max(*logits);
```

### 广播操作（表达式写法）

广播没有独立原语，一律在表达式里表达：

```cpp
// 按行广播：A += row_vec（row_vec 每行一个值）
(void)dsl::compute_into(engine, dsl::leaf(*a) + dsl::row_broadcast(*row_vec), *a);

// 按列广播：A *= col_vec（col_vec 每列一个值）
(void)dsl::compute_into(engine, dsl::leaf(*a) * dsl::col_broadcast(*col_vec), *a);
```

**典型用途**：

```cpp
// 加偏置: output += bias (每行加同一个偏置)
(void)dsl::compute_into(engine, dsl::leaf(*output) + dsl::row_broadcast(*bias), *output);

// 缩放: output *= scale (每列乘同一个缩放)
(void)dsl::compute_into(engine, dsl::leaf(*output) * dsl::col_broadcast(*scale), *output);
```

---

## 逐元素运算

> 下面全部是表达式写法：单步调试可以每步各写一条 `dsl::compute`，**生产代码组合多个
> 运算时优先写成一条表达式**（见"表达式融合"）——一次 dispatch、无中间 Tensor。

### 一元运算（DSL 叶子）

```cpp
// neg / exp / log / sqrt / rsqrt / abs / tanh 都是 DSL 一元叶子
auto exp_t = dsl::compute(engine, dsl::exp(dsl::leaf(*a)), a->rows(), a->cols());
auto neg_t = dsl::compute(engine, -dsl::leaf(*a),           a->rows(), a->cols());
```

### 二元运算（表达式运算符）

```cpp
// + - * / 与 max/min 直接写在表达式里（运算符重载）
auto sum_t = dsl::compute(engine, dsl::leaf(*a) + dsl::leaf(*b), a->rows(), a->cols());
auto mx_t  = dsl::compute(engine, dsl::max(dsl::leaf(*a), dsl::leaf(*b)), a->rows(), a->cols());
```

### 标量二元运算（`dsl::rparam`）

```cpp
// 标量在右 / 在左皆可；dsl::rparam 承载运行时标量（不进 AOT key）
auto y = dsl::compute(engine, dsl::leaf(*a) * dsl::rparam(0.1f), a->rows(), a->cols());
auto z = dsl::compute(engine, dsl::rparam(1.0f) + dsl::leaf(*a), a->rows(), a->cols());
```

### 条件选择（`dsl::select`）

```cpp
// ReLU 反向等条件选择用 dsl::select(cond, then, else)
auto relu_grad = dsl::compute(engine,
    dsl::select(dsl::leaf(*x) > Scalar{0}, dsl::leaf(*g), Scalar{0}),
    x->rows(), x->cols());
```

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

### 多步变换：写成一个表达式

多步逐元素变换直接写成**一条** `dsl::compute` 表达式（单个 GPU 融合 kernel，
中间量不落显存）：跨张量引用用 `dsl::leaf`、运行时标量用 `dsl::rparam`、
结果写进既有张量用 `dsl::compute_into`。表达式之间没有"块式录制"入口，
融合粒度就是单条表达式。

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
// 真实实现（compute_layer_mlp.hpp）：matmul 段 + 按行广播 bias 融合为单 kernel
auto output = nn::dsl::compute(engine,
    nn::dsl::matmul(*weight_, input) + nn::dsl::row_broadcast(*bias_),
    weight_->rows(), input.cols());
```

（引擎级 `engine.matmul_with_bias(A, B, bias, transA, transB)` 也可用，等价语义。）

### 示例 2：ReLU 激活

```cpp
// 单表达式融合：out = max(x, 0)
auto output = nn::dsl::compute(engine, nn::dsl::max(nn::dsl::leaf(input), Scalar{0}),
                               input.rows(), input.cols());
// backward 同理：grad = select(leaf(x) > 0, leaf(grad_out), 0)
```

### 示例 3：LayerNorm

真实实现（`compute_layer_mlp.hpp`）拆成 7 条 DSL 表达式（归约输出 `(1,B)` 小向量，
逐元素步用 col/row 广播把小向量并入全尺寸张量；每条一个融合 kernel）：

```cpp
// 1) mean_raw = compute_reduce(col_reduce_sum(leaf(x)), F, B)          → (1,B)
// 2) mean     = compute(leaf(mean_raw) * rparam(1/F), 1, B)
// 3) diff     = compute(leaf(x) - col_broadcast(mean), F, B)
// 4) var_raw  = compute_reduce(col_reduce_sum(leaf(diff)*leaf(diff)), F, B)
// 5) std_inv  = compute(rsqrt(leaf(var_raw) * rparam(1/F) + rparam(eps)), 1, B)
// 6) norm     = compute(leaf(diff) * col_broadcast(std_inv), F, B)
// 7) out      = compute(leaf(norm)*row_broadcast(gamma) + row_broadcast(beta), F, B)
// backward：gy = grad*gamma；grad_x = (gy - mean(gy) - norm*mean(gy*norm)) * rsqrt(var+eps)
```

（每步的完整实参形如 `dsl::compute(engine, expr, rows, cols, p_.stable)`。）

### 示例 4：多头注意力

> 注意力 forward 为**单 fold kernel**（QKᵀ/掩码/online softmax/ΣwV 分块流式，
> S 矩阵绝不物化，见 `compute_layer_attention.hpp`）；下面这条
> "batched_matmul → softmax → batched_matmul" 的链式写法只用于说明**结构语义**：

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
// ❌ 不推荐：拆成多次调用（每个中间量落显存，每个中间量一次 dispatch）
auto t1 = nn::dsl::compute(engine, nn::dsl::exp(nn::dsl::leaf(x)), rows, cols);
auto t2 = nn::dsl::compute(engine, nn::dsl::leaf(*t1) + nn::dsl::leaf(y), rows, cols);
auto result = nn::dsl::compute(engine, nn::dsl::leaf(*t2) * nn::dsl::leaf(z), rows, cols);

// ✅ 推荐：一条表达式 = 一个融合 kernel
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

*本文示例与当前代码签名一致；引擎接口全量清单见 `docs/development/12-compute-engine-inventory.md`。*