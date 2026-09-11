# ⚡ 性能优化手段

> 本文档详细梳理 neuralnet.cpp 项目中的所有性能优化策略，涵盖 CPU 多线程、缓存优化、GPU 加速、内存管理、算法级优化等层面。

---

## 📊 优化全景图

```
┌─────────────────────────────────────────────────────────────┐
│                    性能优化层次                               │
├─────────────────────────────────────────────────────────────┤
│  L5 算法级    │ 批量化注意力、融合算子、axpy 融合             │
│  L4 GPU 加速  │ Vulkan Compute、批量提交、Command Buffer     │
│  L3 内存级    │ 缓存分块、零分配热路径、Tensor 零拷贝         │
│  L2 并行级    │ SmartPolicy 自适应并行、线程池 latch 零分配   │
│  L1 指令级    │ AVX2 SIMD、-ffast-math、-funroll-loops       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. SmartPolicy 自适应并行策略

### 原理

根据数据规模自动决定串行还是并行执行，避免小数据量时线程调度开销（约 50~200μs）大于计算收益。

```cpp
// config.hpp
inline constexpr std::size_t PARALLEL_THRESHOLD = 524288;  // 512K 元素
```

### 实测数据

| 元素数 | 串行 μs | 并行 μs | 加速比 | 建议 |
|--------|---------|---------|--------|------|
| 16384 | 9.6 | 9.8 | 0.98x | 串行 |
| 65536 | 38.7 | 121.7 | 0.32x | 串行 |
| 524288 | 308.7 | 201.5 | **1.53x** | ← 首次稳定 > 1.5x |
| 1048576 | 620.3 | 186.7 | **3.32x** | 并行 |
| 4194304 | 2657.3 | 340.6 | **7.80x** | 并行 |

### 实现方式

所有顶层并行函数自动判断：

```cpp
// config.hpp — 顶层并行算法函数
template<typename Iterator, typename Func>
inline void for_each(Iterator first, Iterator last, Func&& func)
{
    const auto n = std::distance(first, last);
    if (n >= PARALLEL_THRESHOLD)
        global_thread_pool().parallel_for_each(first, last, func);
    else
        for (auto it = first; it != last; ++it) func(*it);
}
```

---

## 2. 线程池 latch 零分配设计

### 旧版问题

每次 `submit` 需要：
1. 构造 `shared_ptr<packaged_task>` → **堆分配**
2. 获取 `future` → **同步开销**
3. 加锁入队 → **互斥开销**

### 新版优化

- **一次加锁批量入队**：N 个分块仅 1 次 `lock_guard`
- **原子计数器替代 future**：`std::atomic<int> latch` 替代 N 个 future
- **调用者参与计算**：调用者不空等，而是从队列取任务执行（work-stealing）
- **三阶段等待策略**：
  1. 短自旋（16 次 `pause` 指令）— 应对 latch 即将归零的快路径
  2. Work-stealing：尝试从队列取任务执行
  3. `condition_variable` 短超时等待（100μs），避免 CPU 空转

### 性能收益

```
旧版：N 分块 = N 次堆分配 + N 次加锁 + N 个 future 同步
新版：N 分块 = 1 次加锁 + 0 次堆分配 + 1 个原子计数器
```

---

## 3. 矩阵乘法缓存分块（Tiling）

### 原理

将矩阵乘法分解为 BLOCK_SIZE × BLOCK_SIZE 的小块，确保每块的工作集装入 CPU L1 缓存（32KB）。

```cpp
// config.hpp
inline constexpr std::size_t BLOCK_SIZE = 64;
// 64 × 64 × 8 bytes = 32KB — 安全装入大多数 CPU 的 L1 缓存
```

### 分块矩阵乘法内核

```
A (M×K) × B (K×N) = C (M×N)

for i_block in range(0, M, BLOCK_SIZE):
    for j_block in range(0, N, BLOCK_SIZE):
        for k_block in range(0, K, BLOCK_SIZE):
            // 从 B 中预取一块到本地数组 b_block（转置布局）
            // 累加 C 的一块
```

关键优化：
- **B 块预取转置**：`b_block[jj * k_len + kk]` — 消除内层循环的跨行访问
- **栈分配 b_block**：`std::array<Scalar, 64*64>` — 零堆分配
- **分块级并行**：每个 `(i_block, j_block)` 块独立，可并行调度

---

## 4. 编译器优化选项

### Release 模式编译标志

```cmake
# CMakeLists.txt
set(RELEASE_FLAGS -O3 -ffast-math -funroll-loops -march=native)
```

| 标志 | 作用 |
|------|------|
| `-O3` | 最高级别优化（内联、循环变换、向量化） |
| `-ffast-math` | 允许浮点重排序、忽略 NaN/Infinity（提升 FMA 吞吐） |
| `-funroll-loops` | 循环展开，减少分支预测开销 |
| `-march=native` | 启用本机 CPU 指令集（AVX2/AVX-512 等 SIMD） |

---

## 5. GPU 加速（Vulkan Compute）

### 架构设计

GPU 加速遵循引擎化架构铁律：
- `ComputeEngine` 抽象接口不变
- `GpuEngine` 实现所有原语的 GPU 版本
- Layer 的 forward/backward **零修改**

### 关键设计

#### 5.1 批量命令提交

```cpp
// GPU 引擎：begin_batch 开始录制，end_batch 统一提交
engine.begin_batch();           // 开始录制
engine.matmul(A, B);            // 不立即执行
engine.elementwise_binary(...); // 不立即执行
engine.end_batch();             // 一次性提交 + fence wait
```

- CPU 引擎：`begin_batch/end_batch` 为 no-op（同步执行）
- GPU 引擎：录制到 Command Buffer，`end_batch` 统一提交

#### 5.2 零层间 PCIe 传输

```
训练数据从 CPU 上传一次 → 全程 GPU 计算 → 仅 evaluate 时下载
```

#### 5.3 算子融合

- `axpy_inplace`：融合 `clone + scale + add` 三步为一次 dispatch
- `elementwise_select_scalar_cond`：融合条件选择（ReLU backward）
- `broadcast_row_inplace` / `broadcast_col_inplace`：融合广播 + 逐元素操作

---

## 6. 张量零拷贝传递

### Tensor 的 shared_ptr 语义

```cpp
class Tensor {
    std::shared_ptr<Matrix> cpu_data_;     // 引用计数
    std::shared_ptr<GpuTensor> gpu_data_;  // 引用计数
};
```

- **拷贝 Tensor = 增加引用计数**，不复制底层数据
- **修改需要显式 clone**：`engine.clone(src)` 创建深拷贝
- Layer 间传递 Tensor 全部零拷贝

### 引擎辅助方法

```cpp
// 仅在需要修改但不影响原 Tensor 时使用
auto diff = clone_tensor(engine, input);  // 深拷贝
// 原地修改 diff，input 不受影响
engine.broadcast_col_inplace(*diff, mean, BinaryOp::Sub);
```

---

## 7. 融合 Axpy（scale + add）

### 问题

旧实现中 `dst += scalar * src` 需要 3 步：

```
clone_tensor → scale_inplace → add_inplace
= 3 个 GPU 原语 + 3 个临时 buffer
```

### 优化

```cpp
// 新实现：单次 dispatch
engine.axpy_inplace(dst, scalar, src);
// = 1 个 GPU 原语 + 1 个临时 buffer
```

### 性能收益

```
100 个参数 × 每 step 调用 ~3 次 = 减少 600 次 GPU buffer 分配/step
```

---

## 8. 多头注意力批量化

### 旧版：per-head 循环

```cpp
for (size_t h = 0; h < num_heads; ++h) {   // N 次循环
    Q_h = slice(Q, h*d_k, (h+1)*d_k);
    scores = Q_h × K_h^T;
    attn = softmax(scores);
    O_h = attn × V_h;
}
output = concat(O_0, O_1, ..., O_{H-1});
```

### 新版：批量 dispatch

```
rearrange_3d → (batch*H*d_k, seq)  // 头维度在行方向
batched_matmul → 单次 dispatch 处理所有样本和所有头
rearrange_3d_back → (H*d_k, batch*seq)
```

**性能提升**：将 H 次 matmul 融合为 1 次 `batched_matmul`。

---

## 9. 因果掩码缓存

### 问题

GPT 训练中，每个 forward 都需要创建因果掩码矩阵 `(batch*H*seq, seq)`。

### 优化

```cpp
class CausalSelfAttention {
    Tensor mask_cache_;
    std::size_t mask_cached_key_ = 0;  // (batch << 16) | seq_len

    void ensure_mask(engine, batch, seq_len) {
        const std::size_t key = (batch << 16) | seq_len;
        if (mask_cached_key_ == key) return;  // 已缓存，跳过
        // ... 构造掩码 ...
        mask_cached_key_ = key;
    }
};
```

**收益**：相同 `(batch, seq_len)` 组合只构造一次掩码。

---

## 10. 位置编码缓存

```cpp
class PositionalEncoding {
    Tensor encoding_cache_;
    std::size_t cached_total_ = 0;

    void rebuild_encoding(engine, total_len) {
        if (cached_total_ == total_len) return;  // 已缓存
        // ... 构造编码 ...
        cached_total_ = total_len;
    }
};
```

---

## 11. L1 Cache 友好的数据布局

### 行主序存储

```cpp
// data_[row * cols + col] — 行方向连续
// 相邻元素在内存中连续 → SIMD 向量化友好
```

### 转置内核

```cpp
// 分块转置：每个块内行优先读、列优先写
// 块间并行 → 利用 L1 缓存的空间局部性
```

---

## 12. 编译期零开销抽象

### 表达式模板（AST）

```cpp
// algebra_expr.hpp — 编译期构造计算树
auto expr = max(x, Scalar{0});          // ReLU AST
auto expr2 = a + b * c;                 // 融合乘加 AST

// 统一执行：单次遍历，无临时矩阵
compute::apply(span, expr);
```

**收益**：
- 消除临时矩阵分配（`a + b * c` 不创建 `b*c` 的中间矩阵）
- 编译器可内联整个表达式树

---

## 13. 性能基准测试工具

### bench_thresholds

测量不同数据规模下的串行/并行性能，确定最优并行阈值：

```bash
./build/bench_thresholds
# 输出：元素数 | 串行μs | 并行μs | 加速比 | 建议
```

### compute_bench

测量矩阵乘法、逐元素运算、归约等核心原语的性能。

---

## 📋 优化效果总结

| 优化手段 | 场景 | 预期收益 |
|----------|------|----------|
| SmartPolicy | 小数据串行、大数据并行 | 避免小数据 0.3x 退化 |
| 线程池 latch | 所有并行操作 | 消除 N 次堆分配/加锁 |
| 缓存分块 | 矩阵乘法 | L1 命中率 → 3-8x 加速 |
| -ffast-math | 所有浮点运算 | FMA 重排序 + SIMD |
| -march=native | 所有计算 | AVX2/AVX-512 SIMD |
| GPU 批量提交 | GPU 推理/训练 | 减少 kernel launch 开销 |
| Tensor 零拷贝 | Layer 间传递 | 消除数据拷贝 |
| 融合 axpy | 优化器更新 | 减少 600 次 buffer 分配/step |
| 注意力批量化 | MHA | H 次 matmul → 1 次 |
| 掩码/编码缓存 | GPT 训练 | 避免重复构造 |
| 表达式模板 | 逐元素运算 | 消除临时矩阵 |
| 分块转置 | 转置操作 | L1 友好 + 并行 |
