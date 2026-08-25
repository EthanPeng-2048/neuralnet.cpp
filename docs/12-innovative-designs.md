# 🚀 neuralnet.cpp 创新设计总览

> 本文档汇总本项目中所有具有**原创性或系统性思考**的设计。它不是架构/性能/融合等专项文档的重复，而是从"为什么这么做、创新点在哪、带来什么收益"的角度，把这些设计串成一张全景图。
>
> 关联文档：[01-architecture](./01-architecture.md) · [02-performance](./02-performance.md) · [09-operator-fusion](./09-operator-fusion.md) · [10-memory-optimization](./10-memory-optimization.md) · [11-ir-optimization](./11-ir-optimization.md) · [05-algorithm-reference](./05-algorithm-reference.md)

---

## 0. 总览

一个好的深度学习框架，设计难点往往不在"算法本身"，而在于**如何把算法用干净、可维护、可并行、可扩展的方式组织起来**。本项目的创新集中在以下几个维度：

| 维度 | 核心创新 | 一句话 |
|------|---------|--------|
| **架构** | 分层引擎化 | 一次编写 Layer，CPU/GPU 零修改双跑 |
| **架构** | 铁律式职责划分 | 算法留 Layer、原语进引擎、Shader 是内部实现 |
| **编译** | AOT 闭合世界表达系统 | 表达式文本唯一、构建期扫描生成、运行时按 key 精确分发 |
| **性能** | SmartPolicy + 零分配线程池 | 自适应并行、消除每次任务的堆分配 |
| **性能** | 表达式模板零临时矩阵 | `a + b * c` 不物化中间结果 |
| **显存** | 结构融合（而非手写 kernel） | 三粒度融合：逐元素链 / matmul+归约 / 两趟注意力 |
| **显存** | 两趟式注意力 + 反向重算 | 不物化 `O(seq²)` 分数矩阵 |
| **显存** | 稀疏 CE + 梯度检查点 + 池归还 | 把显存峰值从 29GB 压到 27GB（并持续下探） |
| **IR** | 带确定性 pass 的规范 IR | DCE/常量折叠/CSE/寄存器分配，key 定义在 canonical IR 上 |
| **算法** | 纯 BPE + 兜底词表自举 | 小模型也能稳定启动、支持中文 |

下面逐条展开。

---

## 1. 核心架构创新

### 1.1 引擎化架构（Engine-Based）

**创新点**：让 `Layer::forward/backward` 只写一次，通过传入的 `ComputeEngine` 自动适配 CPU / GPU 双后端（CUDA 已停用，见 `docs/13-optimize-proposal-list.md` §5），而不是像多数教学框架那样为每后端各写一份。

```cpp
class Layer {
    virtual Result<Tensor> forward(ComputeEngine& engine, const Tensor& input) = 0;
    virtual Result<Tensor> backward(ComputeEngine& engine, const Tensor& grad) = 0;
};
```

**收益**：
- 新增一个后端 = 只实现 `ComputeEngine` 抽象，零 Layer 改动。
- 算法正确性在 CPU 上验证后可放心跑 GPU。

### 1.2 分层"铁律"（Red Lines）

这是本项目最鲜明的**原创治理原则**，贯穿所有文档：

```
引擎只提供 op-level 原语（matmul/reduce/broadcast/exp…），绝不出现 softmax/attention/layernorm 命名接口
算法文本只写在 Layer（ReLU/GeLU/Softmax/Attention 的公式只在 compute_layer.hpp / compute_loss.hpp）
Shader / 融合逻辑是引擎内部实现，用户不可见
```

**创新点**：不是把"引擎是否认识 softmax"当实现细节，而是把它当一条**不可逾越的红线**。引擎只认**结构**（如 `reduce(matmul(A,B))`），绝不认**算法名**。这使得算子融合、IR 优化、新后端都能在一个稳定的契约上展开。

配套约束：
- `ComputEngine` 原语可"专"（matmul+归约、matmul+softmax分母都是合法原语）但必须通用可复用、不叫算法名。
- 全程 `noexcept` / `Result<T>`（`std::expected`），零手动内存管理（`-fno-exceptions`）。

### 1.3 统一张量 `Tensor` + 零拷贝

`Tensor` 用 `shared_ptr` 持有 CPU/GPU 两份互斥存储，拷贝=引用计数，跨层传递零拷贝。修改需显式 `clone`。这为 GPU 训练"全程驻留显存、仅 evaluate 时下载"提供了容器基础。

---

## 2. AOT 闭合世界表达系统（Compiler-Style）

这是本项目**最具工程创新性**的设计，把"表达式"当成一门可编译的语言：

### 2.1 单一来源（Single Source of Truth）

算法表达式**直接写在 C++ 源码里**（如 `exp(x - row_max) / sum(...)`），而不是单独维护一份"计算图定义"。

### 2.2 构建期扫描 + 生成

```
Layer 内联表达式 → to_expr_spec 折叠 → ExprSpec（扁平 SSA IR）
    → expr_spec_key（规范结构 key）→ scan_exprs 收集
    → gen_fused 用 glsl_gen 合成 GLSL → glslc → 内联 SPIR-V → fused_registry.hpp
运行时：fold → key → find_fused(key) → 精确 dispatch（闭合世界）
```

**创新点**：
- **闭合世界（closed world）**：所有可能出现的融合 kernel 都在构建期预编译、内联进最终程序。运行时**没有编译器、没有 eager 分发**。
- **派生而非手写**：`scan_exprs` 通过 dry-run 执行 Layer 的 forward/backward 触达表达式，把折叠后的**结构（派生物）**采集进注册表，而不是手工维护一份 kernel 清单。
- **未命中硬报错**：GPU 上遇到未预生成的表达式结构，**硬报错**并且绝不静默回退 CPU，保持项目"不降级"哲学——据此暴露扫描覆盖缺口。

### 2.3 确定性 key 铁律

key 定义在 **canonical（优化后）IR** 上，scan 与 runtime 两端必须先 canonicalize 再算 key。且 pass 遍历顺序固定、CSE 胜利者选择固定、寄存器分配固定——避免跨编译器（Clang/MSVC）漂移。这条铁律来自一次真实事故：C++ 实参求值顺序未指定曾导致 key 跨编译器不稳定。

---

## 3. 算子融合（结构融合，而非手写 kernel）

### 3.1 设计哲学

**不手写任何命名算法 kernel**（如不手写 flash-attention、不手写 softmax kernel），而是：
1. Layer 用 `eval_expr` / `begin_expr/end_expr` 录制表达算法；
2. 引擎/工具按**结构**合成融合 kernel；
3. 所有中间 Tensor 由融合 kernel 内部消解，不落显存。

### 3.2 三粒度融合

| 粒度 | 覆盖 | 消除的中间量 |
|------|------|-------------|
| **逐元素链** | ReLU/GeLU/SwiGLU 连续逐元素 | 链上的全尺寸中间 Tensor |
| **matmul + 归约** | softmax 分子分母、norm 统计 | `A·B` 全尺寸物化 |
| **两趟注意力 / 稀疏 CE** | attention、大词表交叉熵 | `O(seq²)` / `O(vocab×seq)` 物化 |

### 3.3 三个 matmul 融合原语

为承载两趟注意力，新增了一组**通用、可复用**的原语（不是"attention"，而是"matmul 后接结构"）：

- `batched_matmul_reduce`：matmul 后沿输出维度归约，不物化中间 `A·B`。
- `batched_matmul_softmax_denom`：减行 max → exp → 按列求和（softmax 分母，数值稳定）。
- `batched_matmul_softmax_apply`：行 softmax 归一化后与 V 相乘累加，逐 tile 流式、不物化权重矩阵。

反向同样有 `..._softmax_backward_q` / `..._softmax_backward_kv`，kernel 内部重算权重矩阵。

### 3.4 形状无关融合（Key 创新）

RoPE 的 `RowMod/RotateHalf` 参数如果以**结构常量**折进 key，每个 `d_k` 就要一个新融合 shader → 闭合世界无法穷举。

**解决**：把形状相关参数（mod/block）从 key 里**拿出来变成运行时 push constant**。结构相同、形状不同共享同一融合 shader——任意 `d_k`（含非 2 的幂）都全融合，零额外显存。这是"闭合世界"与"形状多变"矛盾的一个优雅解法。

---

## 4. 两趟式注意力（内存高效 Attention）

### 4.1 问题

传统 `scores/masked/attn_cache_` 各占一份 `H·batch·seq²`，且 `attn_cache_` 永久缓存用于反向。这是 GPT+Vulkan 训练显存高的元凶之一。

### 4.2 方案

把 `scores → mask → softmax → ×V` 拆成**不物化 `seq²` 矩阵**的算法：

- **Forward**：`m = max of QᵀK`（bmm_reduce）→ `l = Σ exp(QᵀK - m)`（bmm_denom）→ `O = W·V` 逐 tile（bmm_apply）。只留下 `m/l`（`H·batch·seq`）与 `O`（`H·batch·d_k·seq`）。
- **Backward**：默认**反向重算 W**（用原语再算一次），放弃 `attn_cache_`。代价是 2× FLOPs，换整份 `BH·seq²` 缓存。

### 4.3 收益

每层从 ~3×`BH·seq²` 降到 `O(BH·seq·d_k)`。这是典型的**用计算换显存**，在训练显存吃紧、而算力相对有余的场景下是最优取舍。

---

## 5. 稀疏交叉熵（Sparse CE）

### 5.1 问题

大词表（vocab≈25k）交叉熵若物化整个 `(classes, total)` 的 softmax，显存与下载开销巨大。

### 5.2 方案

两个 op-level 原语，单 kernel 完成全部计算：
- `col_softmax_denom`：exp 后列内树形归约求分母，不物化 exp 张量。
- `col_softmax_sparse_forward`：单 kernel 同时产出**稠密梯度**（每列只对合法列施加）与**标签位置的 log_softmax loss**，不从整张 softmax gather。

**创新点**：labels 以 `(1,N)` 浮点打包上传（`vocab ≤ 2²⁴` 时 float 可精确表示类别索引），kernel 内转 `uint` 读取；非法/被屏蔽列整列置 0，与 CPU 参考完全一致。显存从 ~3-4×`(classes,total)` 降到 ~2×。

---

## 6. IR 优化（带确定性 pass 的规范 IR）

### 6.1 动机

`ExprSpec` 已是事实上的轻量 IR（SSA、扁平、可序列化、确定性 key），但没有优化 pass，直接导致真实问题：
- 子表达式 `grad*gamma` 重复 3 次 → 超 `EXPR_MAX_INPUTS=8`，被迫手工拆表达式。
- 无 DCE/常量折叠，shader 携带冗余计算。
- 无寄存器分配，`num_regs` 受 `EXPR_MAX_REGS=16` 约束。

### 6.2 方案

- **IR-A canonicalize**：DCE + 常量折叠 + 代数化简 + 稳定重编号（保守、不改变浮点语义）。
- **IR-B CSE + 寄存器分配**：哈希指令去重，liveness 线性扫描确定性贪心分配寄存器。
- **IR-C 图 IR + 融合分析**：`begin_expr/end_expr` 录制虚拟寄存器 DAG；**逐元素链拼接**——B 以 Linear 视图消费 A 输出且 A 无其他消费者、形状相同、均无归约 → 指令内联进 B，单 kernel。归约节点/归约输出作为融合边界。
- **IR-D emitter 抽象**：`ExprEmitter` 接口 + 注册表，一份 canonical IR 产出 GLSL/CPU 多后端（验证"一份 IR 多后端"）。

### 6.3 关键不变量

canonicalize 不改变 views/inputs 的顺序与内容，只优化 instrs/consts/num_regs → 运行时输入绑定布局不变。寄存器"先声明后赋值"以兼容 liveness 复用同号寄存器。

---

## 7. 显存优化体系（三级）

围绕 GPT 训练峰值显存（29GB → 27GB，持续下探）的三条独立路径：

### L1 激活重计算（梯度检查点）
- `Layer` 契约扩展 `recompute_supported / forward_recompute`；`GPTModel` 实现**块级检查点**（每 N 个 GPTBlock 存一次输入，backward 重算中间）。
- 与全存基线**逐位一致（max_abs=0）**。

### L2 内存池整块归还 + 统计
- `MemoryPool` 新增 `PoolStats` / `release_idle_blocks()`（整块 `vkFreeMemory`，带保留阈值防抖动）。
- **结构融合宣示**：中间 Tensor 的归还路径安全——每个原语录制都包裹全内存屏障（写入→`SHADER_READ` / 输出→`MEMORY_READ`），保证 batch 命令缓冲内前序写入对后续可见。

### L3（远期）
- 自动融合 + 无 m/v 优化器（Adafactor 类）降参数状态内存。

### 设计红线
全程**不引入 f16/bf16**（数值统一 fp32）——"省内存但不牺牲精度"，这本身也是一个收敛的取舍决策。

---

## 8. CPU 性能优化（L1/L2 层）

### 8.1 SmartPolicy 自适应并行
根据数据规模自动决定串行/并行，避免小数据量时线程调度开销（约 50~200μs）大于计算收益。实测在 ~512K 元素处首次稳定 >1.5x，此前串行更优。

### 8.2 线程池 latch 零分配
旧版每任务 = N 次堆分配 + N 次加锁 + N 个 future 同步；新版 = 1 次加锁 + 0 次堆分配 + 1 个原子计数器，配合三阶段等待（短自旋 → work-stealing → 条件变量短超时）与"调用者参与计算"。

### 8.3 缓存分块 + 栈分配
`BLOCK_SIZE=64`（64×64×4B=16KB 装入 L1），B 块预取转置，b_block 栈分配零堆分配。

### 8.4 算子融合（原语级）
- `axpy_inplace`：`clone+scale+add` 三步并一步。
- `elementwise_select_scalar_cond`：条件选择融合（ReLU backward）。
- 多头注意力**批量化**：`rearrange_3d → 单次 batched_matmul → 转回`，把 H 次 matmul 融为 1 次。
- 因果掩码 / 位置编码缓存：相同 `(batch,seq)` 只构造一次。

### 8.5 表达式模板零临时矩阵
`a + b * c` 在编译期构造 AST，单次遍历执行，不创建 `b*c` 中间矩阵，编译器可整体内联向量化。

---

## 9. 分词器与算法创新

### 9.1 纯 BPE（CharBPE）支持中文
默认 `bpe` 为纯 BPE（优先于 BBPE），**可支持中文**：每个汉字作为基础字符单元，学习常用字词合并——比跨语言字节切分更能捕捉词义；BBPE 作为备选。词表含 256 个 ASCII 兜底 token（ID 4~259）+ 4 个特殊符号，保证最小 `vocab_size=260` 也能稳定启动不崩溃。

### 9.2 GPT 训练约定
每行作为独立样本，`[BOS]` 开头、`[EOS]` 仅行尾；`generate()` 带 `min_new_tokens` 防止小模型过早输出 EOS；`seq_len` 降到覆盖 >99% 样本的规模以避 `n²` 显存。

---

## 10. 创新背后的工程方法

除了具体技术，项目在**方法论**上也有鲜明的、可复用的设计：

| 方法 | 体现 |
|------|------|
| **合规红线先行** | 每次演进（融合/IR/显存）都先写"不可逾越红线"表，再谈实现 |
| **结构 > 命名的哲学** | 引擎认结构不认算法名，是融合与 IR 能落地的共同前提 |
| **闭合世界 + 硬报错** | 不静默降级，宁可报错暴露覆盖缺口 |
| **恶意保守的正确性安全网** | GPU 融合未命中/失败回退原语组合；融合路径与回退路径数值一致并互相验证 |
| **数值确定性** | 全程 fp32、定点可复现，优化 pass 不改变浮点语义 |
| **可验证增量里程碑** | M1-M6 每步有独立测试验证（gradcheck / GRUD/CPU 对照），先小层数回归再全面放开 |

---

## 11. 收益一栏

| 设计 | 量化收益（示例配置） |
|------|---------------------|
| 两趟注意力 | 每层 `~3×BH·seq²` → `O(BH·seq·d_k)` |
| 稀疏 CE | 全 softmax `(vocab×seq)` 物化 → 仅标签 gather |
| 算子融合 + 显存体系 | GPT 训练峰值 ~29GB → ~27GB（并持续下探） |
| IR CSE + 寄存器分配 | 消除"手工拆表达式"，`num_regs` 受控在 16 内 |
| SmartPolicy | 小矩阵 0.3x 退化 → 串行；大矩阵最高 ~7.8x 加速 |
| 融合 axpy | 每 step 减少 ~600 次 GPU buffer 分配 |
| 注意力批量化 | H 次 matmul → 1 次 batched_matmul |

---

> 🌙 本文档为创新设计的**全景速览**，不替代各专项文档的细节。想要深入的读者请跳转文首的关联文档。