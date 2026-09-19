# CPU 多核利用率问题诊断（2026-09-18）

> 触发：用户报告"CPU 计算不能充分利用多核性能"，怀疑 ①线程池设计 ②调用方 ③分块拆得太细。
> 结论：①②③ 中 **②（调用方）是主因**，且不是"拆太细"，而是 **DSL 表达式求值路径完全没有并行化且是逐元素解释执行**。

---

## 1. 结论摘要

| 路径 | 实测平均占用核数 | 判定 |
|------|------------------|------|
| blocked matmul（`Matrix::multiply_to_span`） | 13~15 / 32 逻辑核 | ✅ 正常（16 物理核已接近跑满，HT 无收益） |
| `nn::for_each` 逐元素（add/scale/transform） | 2~4 | ⚠️ 受阈值/内存带宽限制，基本合理 |
| **`CpuEngine::eval_expr_impl`（DSL 融合）** | **1.00** | ❌ **完全串行** |
| **LayerNorm / RMSNorm / Softmax** | **0.94 ~ 1.00** | ❌ 走 DSL，单核标量解释执行 |
| **MHA / Transformer / GPTBlock** | **2.3 ~ 2.8** | ❌ 注意力主体走 DSL |

**核心事实：一个 GPT block 的前向只用到约 2.8 / 32 个核，8.5 GFLOPS。**
其中 MHA 641ms + FeedForward 131ms + 2×Norm 86ms ≈ 858ms（前向 875ms）——
**约 83% 的 Transformer 前向时间花在"单线程 + 解释执行"的 DSL 路径上。**

---

## 2. 测量方法与原始数据

机器：Intel Xeon E5-2697A v4，16 物理核 / 32 逻辑核 @2.6GHz（AVX2，无 AVX-512）。
构建：`build/`（Release，clang++ 23，`-O3 -march=native`）。
"平均占用核数" = 进程 `TotalProcessorTime / wall time`（见 §5 复现脚本）。

### 2.1 单个 Layer（`layer_bench`，默认尺寸 dmodel=768, seq=512, batch=1）

```
matmul         :   13.238 ms       162.2 GFLOPS          avg_cores=14.73
matmul_bt      :   12.074 ms       177.9 GFLOPS          avg_cores=13.25

layernorm      : fwd   42.252 ms    0.1 GFLOPS           avg_cores= 1.00
rmsnorm        : fwd   27.395 ms    0.1 GFLOPS           avg_cores= 0.94
softmax        : fwd   40.946 ms    0.1 GFLOPS           avg_cores= 0.99
swiglu         : fwd    0.144 ms    0.3 GFLOPS           avg_cores= 0.75

feedforward    : fwd  151.215 ms   32.0 GFLOPS           avg_cores= 5.68
mha            : fwd  641.701 ms    4.1 GFLOPS           avg_cores= 2.29
transformer    : fwd  875.602 ms    8.5 GFLOPS           avg_cores= 2.76
gpt_block      : fwd  875.826 ms    8.5 GFLOPS           avg_cores= 2.74
```

### 2.2 逐元素 / 归约算子（`--op`，1024×1024）

```
add_inplace      avg_cores= 1.95 |   0.220 ms   57.1 GB/s   （内存带宽受限，正常）
elementwise_exp  avg_cores= 3.99 |   2.472 ms    3.4 GB/s
transpose        avg_cores= 2.21 |   2.132 ms    3.9 GB/s
row_reduce_sum   avg_cores= 0.90 |   0.852 ms    9.9 GB/s   （串行）
col_reduce_sum   avg_cores= 2.26 |   0.477 ms   17.6 GB/s
```

### 2.3 线程池本身的扩展性（隔离探针，专用 pool）

```
matmul 512x512x512   threads 1→32 :  11.8 → 153.0 GFLOPS   （13x）
matmul 1024x1024x1024 threads 1→32 :  11.3 → 171.9 GFLOPS   （15x）
matmul 1024x4096x1024 threads 1→32 :  10.0 → 176.9 GFLOPS   （17.7x）
```

### 2.4 参考 LayerNorm 实现（同样 (F=768,B=512) 数据）

```
当前 DSL 解释执行 LayerNorm : 42.25 ms
参考实现 · 串行标量          :  4.674 ms      ← 解释器本身慢 ~9x
参考实现 · 并行 nt=16        :  0.548 ms      ← 再叠 8.5x
-----------------------------------------------
合计可达加速                 :  ~77x
```

### 2.5 CPU DSL 两条路径的并行度（探针 `build/perfprobe/bench_dsl.cpp`，R=C=1024）

```
A) 无归约  exp(x)*x+1  → dsl::compute → eval_cpu（编译期模板路径）: 40.278 ms
B) 含归约  row_sum(exp(x)) → compute_reduce → eval_expr（解释器） : 21.284 ms
avg_cores (A+B 全程)                                              = 0.92
```

**两条路径都是单线程**。`expr_dsl.hpp:642` 注释声称"CPU：编译期模板求值（SIMD 融合）"，
但 `eval_cpu`（`expr_dsl.hpp:630-638`）只是一个 `for (i) sp[i] = e.eval(i);` 的串行循环，
每元素还要重算 `i/cols`、`i%cols` 并解释分派——比朴素标量还慢。

---

## 3. 根因分析（按严重度）

### R1（P0）`CpuEngine::eval_expr_impl` 完全没有并行化，且逐元素解释执行

文件：`include/neuralnet.cpp/compute_cpu_engine.hpp:1251-1789`

- 整个函数 **没有任何 `nn::for_each` / `parallel_for_*` 调用**（`compute_cpu_engine.hpp`
  全文只有两处线程池调用：`:283` gather_rows、`:405` rearrange_3d）。
- 主求值循环是逐元素的 `eval_element(r,c)`：
  - 每个元素执行 `Scalar regs[EXPR_MAX_REGS] = {}`（**EXPR_MAX_REGS=32，即每元素清零 128B**）；
  - 每条指令对每个操作数做 `switch(op.kind)` 解释分派；
  - `ExprOperandKind::Matmul` 分支每元素调用一次 `matmul_out.span()`；
  - `read_input` lambda 带 `switch` 再访存。
- 归约还要 **重放**：每条归约指令对其源重放 `[0, ri)` 整条前缀链（`:1556-1627`），
  Softmax/Attention 有 2~3 条归约 → 同一张网格被解释 3~4 遍。

**受益面（走 DSL 的层）**：LayerNorm、RMSNorm、Softmax、SwiGLU/GELU 激活、
Attention 的 m/l/W、CrossEntropy 稀疏/稠密路径。
即 `expr_dsl.hpp:676` 注释自述的"CPU 扩展语义处理归约视图/指令，**先正确后优化**"——
M3 算子融合把原先手写向量化的 CPU 通路换成了标量解释器。

**同源问题：`eval_cpu` 也是串行的**（`expr_dsl.hpp:630-638`）。CPU 端 DSL 分为两条路：
- 无归约 → `eval_cpu`（模板求值，串行，无 SIMD 保证）；
- 含归约 → `eval_expr_impl`（解释器，串行）。
两条路 §2.5 实测均为 0.92 核。所以"CPU 引擎没跟上 GPU 设计"并非只在解释器一处，
而是 **CPU 端缺乏一个把 IR 降到"专门化 + 并行"kernel 的后端**。

### R2（P0/P1）注意力对 Q·Kᵀ 重复计算 3 次

文件：`include/neuralnet.cpp/compute_layer_attention.hpp:514-584`

`compute_m` / `compute_l` / `compute_W` 各自把 `dsl::matmul(Q, K, true, false, BH)`
重新折成表达式。`eval_expr_impl` 对每次调用都 **重新执行一遍 matmul**（`:1408-1463`）
并各自 `Matrix result(rows,cols)` 分配。

以 seq=512、BH=12 计：QKᵀ 单次 ≈ 0.4 GFLOP，白算 2 遍 ≈ 0.8 GFLOP；
再叠 R1 的三遍标量解释。MHA 前向 641ms 里约六成来自这里。

> 注：`compute_m` 只用 K 矩阵片段，`compute_W` 又只需其归约归一化结果——
> 三趟完全可合并为一趟：matmul → row_max → 行内 exp/归一化。

### R3（P1）`row_reduce` 的并行门控错误 → 实际永不并行

文件：`include/neuralnet.cpp/algebra_matrix.hpp:843`

```cpp
nn::for_each(row_indices.begin(), row_indices.end(), process_row);
```

`nn::for_each`（`core_config.hpp:68`）按**元素个数**与 `PARALLEL_THRESHOLD=524288` 比较；
这里的"元素"是**行**。所以只有 `rows >= 524288` 才并行——神经网络的任何真实形状都达不到，
`row_reduce` 实际上**恒定串行**（实测 0.90 核）。

对比：`broadcast_row_inplace`/`broadcast_col_inplace`（`:971`/`:996`）用
`nn::parallel_for_samples(R, ...)` + `R*C >= PARALLEL_THRESHOLD` 门控，才是正确写法。
`col_reduce`（`:921`）也正确地直接调 `pool.parallel_for_blocks`。

### R4（P2）`chunk_count` 的粒度语义缺陷

文件：`include/neuralnet.cpp/core_threadpool.hpp:102-110`

```cpp
constexpr std::size_t MIN_CHUNK = 1024;        // 假设"每元素=1 个标量"
if (nw <= 1 || total < MIN_CHUNK * 2) return 1;
auto n = std::min(nw, total / MIN_CHUNK);
```

- 对 `nn::for_each`：因 `PARALLEL_THRESHOLD(524288) > nw*MIN_CHUNK(32768)`，
  `MIN_CHUNK` 实际**永不生效**（真正限流的是 `min(nw, ...)`），
  注释里"降低阈值使 MNIST 小隐藏层也能触发多核并行"是误导。
- 分块数上界是 `nw`，所以 **不存在"拆太细导致块数爆炸"的问题**——
  用户假设③在 matmul/逐元素路径上不成立。
- 真正的粒度错配在 R3：把"行/样本"当成"标量元素"套用同一套 MIN_CHUNK/阈值。

### R5（P2）线程池的潜在扩展性瓶颈（当前不是主因）

文件：`include/neuralnet.cpp/core_threadpool.hpp`

- 全局**单队列 + 单 mutex**（`:38`）：所有并行区共用一把锁；`nw` 大或嵌套并行时争用放大。
- `finish_chunk` 每次分片完成 `condition_.notify_all()`（`:119`）：32 等待者 → 惊群，
  每个并行区产生 O(n_chunks × waiters) 次无效唤醒。
- 全局池线程数 = `hardware_concurrency()` = 32（16 物理核），逻辑核参与导致 HT 争用。
- 隔离探针显示 matmul 仍能近线性扩展（§2.3），故当前**不是**主要瓶颈；
  但若把 DSL 求值也并行化（修 R1），这些开销会立刻成为下一个瓶颈，建议一并处理。

---

## 4. 对用户三个假设的判定

| 假设 | 判定 | 依据 |
|------|------|------|
| ① 线程池设计有问题 | **部分成立（次要）** | 单队列/惊群/HT 超订是隐患；但隔离测量 matmul 1→32 线程近线性扩展，当前不是主因 |
| ② 调用线程池的有问题 | **成立（主因）** | `eval_expr_impl` 零并行调用；`row_reduce` 门控错；注意力三重复算 |
| ③ 分块拆太细 | **不成立（方向相反）** | 分块数被 `min(nw, total)` 封顶，不会拆太细；真正问题是 DSL 路径**根本不拆**、`row_reduce` **永不并行** |

---

## 5. 修复建议（按收益/风险排序）

1. **R3 并行化 `row_reduce`**（低风险，1 行级）
   `algebra_matrix.hpp:843` 改为 `nn::parallel_for_samples(rows_, process_row)` 并加
   `rows_*cols_ >= PARALLEL_THRESHOLD` 门控。附带修 `chunk_count` 的语义注释。

2. **R1 并行化 `eval_expr_impl` 的求值循环**（高收益，中等风险）
   - 广播输出循环（`:1775`）按行/块用 `nn::parallel_for_blocks` 分片，每元素链无依赖；
   - 归约重放循环（`:1556`）改为按 `(行, 线程)` 分片 + per-thread 局部累加器再归并
     （参考 `col_reduce` 的 `local_acc` 模式，`:909-936`）；
   - 顺带把 `regs[EXPR_MAX_REGS] = {}` 缩到 `spec.num_regs`（消掉 128B/元素的清零）。
   预计 LayerNorm/Softmax/Attention 的 DSL 部分提升 10~50x。

3. **R2 合并注意力三趟 Q·Kᵀ 为一趟**（高收益，中高风险，需与 GPU 融合路径对齐）

4. **R5 线程池改造**（在 R1 之后再做）
   任务队列按 worker 分桶或 work-stealing deque；`finish_chunk` 改 `notify_one`
   或原子计数 + 无锁等待；线程数默认取物理核数。

**不建议**：调整 `MIN_CHUNK` / 分块大小——它不是瓶颈，改动只会增加回归面。

---

## 6. 复现命令

```powershell
# 核数占用（CPU时间/墙钟）
$p = Start-Process .\build\layer_bench.exe -ArgumentList '--layer','layernorm' -NoNewWindow -PassThru
$p.WaitForExit(); $p.TotalProcessorTime.TotalSeconds

# 隔离探针（本次诊断新增，位于 build/perfprobe/，不在 git 跟踪范围）
cmake --build build --target layer_bench         # 或用已有 build/layer_bench.exe
clang++ -std=c++26 -O3 -march=native -I include -o build/perfprobe/bench_pool.exe build/perfprobe/bench_pool.cpp
.\build\perfprobe\bench_pool.exe     # 线程池扩展性
.\build\perfprobe\bench_ln.exe       # LayerNorm 参考实现对比
.\build\perfprobe\bench_dsl.exe      # CPU DSL 两条路径并行度
```

---

## 7. 专题：重写成"GPU 样式"的完整 CPU 引擎能否修复

**结论：能，这正是对症的架构修复；但要注意它与 GPU 的非对称性，不能字面照搬。**

### 7.1 现状差异（为什么 CPU 引擎"没跟上"）

| | GPU 引擎 | CPU 引擎 |
|---|---|---|
| `eval_expr` 语义 | `expr_spec_key` → AOT 预编译融合 shader，一次 dispatch | 通用标量解释器，逐元素分派 |
| 并行模型 | shader thread grid（天然大规模并行） | **无并行**（`eval_cpu` / `eval_expr_impl` 都是串行循环） |
| 降级/专门化 | 每个 ExprSpec 有专属 kernel | 一条通用代码路径服务所有表达式 |
| 归约 | shader 内 workgroup 归约 | 单线程重放前缀链 |

**版本事实核对**：`eval_expr_impl` 并非"一直停留"，它是**在 `0.2.0-Optimize`
（commit `42c0883`）随 M3 融合一起引入的**；此后文件本身改过 +728/−632（f16、
精度、校验等）。所以"停留在 v0.2.0"指的是**执行模型**自那时未变，而不是文件没动——
这个直觉在实质上是对的。

### 7.2 仓库其实预留过这条路

- `expr_emitter.hpp` 的 IR-D 抽象就是"**一份 canonical IR → 多后端**"；
- `cpu_emitter.hpp`（233 行）曾是第二个 emitter 实现，把 `ExprSpec` 展开为
  可嵌入 C++ 的**逐元素直线代码**（"独立函数，无解释器开销"），
  见 commit `36caba4`；
- 该文件在 `708c6f4`（"Layer拆分头文件…"）被删除，CpuEmitter 未落地。
- 注意其自身注释：**归约不生成 kernel，仍由解释器承担**（"先正确后优化"）——
  与今天的现状同源。

### 7.3 三个必须注意的非对称性

1. **CPU 没有 JIT**。GPU 是构建期 `gen_fused` + `glslc` 把 IR 编成 SPIR-V；
   CPU 若"运行时按 IR 生成 C++"则无法编译。因此 CPU 后端的可行形态是二者之一：
   (a) **构建期 AOT**（像 GLSL 那样预生成 kernel，仅覆盖构建期能枚举的 spec）；
   (b) **结构化专门化**：按 IR 的结构类别（纯逐元素 / 行归约 / 列归约 / 广播…）
   分派到少数几个手写、显式 SIMD 的并行 kernel，而不是字符串 codegen。
   对 header-only + 运行时形状的库，(b) 才是主路。
2. **并行原语不同**。GPU 靠线程网格天然并行；CPU 必须 `ThreadPool + 向量化`。
   非归约表达式是 embarrassingly parallel（按行/块切分即可），
   归约需要两阶段（per-thread 局部累加器 → 归并，仓库已有 `col_reduce` 的 `local_acc` 范式可抄）。
3. **"重写"不是前提**。当前 83% 时间损失里，约 9x 来自解释开销、约 8.5x 来自缺并行，
   两者都可以在**不改接口**的前提下补上：`eval_cpu` 并行化 + `eval_expr_impl`
   按 R1 分片。真正的架构收益在于把这套逻辑固化成"CPU 后端"，而不是继续堆在
   `eval_expr_impl` 里。

### 7.4 建议路线

- **第 1 步（低风险，立即见效）**：`eval_cpu` 加行块并行；`eval_expr_impl` 的
  广播输出按行块并行、归约改 per-thread 局部累加器；`regs[EXPR_MAX_REGS]` 缩到实际 `num_regs`。
- **第 2 步**：把上述逻辑抽成 `CpuExecutor`（IR → 结构类别 → 专门化并行 kernel），
  即"GPU 样式"的 CPU 后端，`eval_expr` 只做分发。
- **第 3 步（可选）**：恢复 IR-D 的 CPU emitter，但走**构建期 AOT**（与 `gen_fused`
  并列），覆盖构建期可枚举的 spec；运行时未命中回落到第 2 步的结构化路径。
- 同时按 §5 修 R3（`row_reduce`）与 R2（注意力三趟 Q·Kᵀ），否则并行化收益会被它们吃掉。

---

## 8. 设计溯源：CPU 端本来该被编译器内联，为什么会有解释器

**一句话：解释器不是为了 CPU，而是为了复用 GPU 的"数据化 IR"（`ExprSpec`）才被引入的。**

### 8.1 事实一：纯逐元素表达式**确实**走编译期内联

`dsl::compute`（`expr_dsl.hpp:646-690`）在 CPU 端有两条路：

```cpp
if (eng.device() == Device::CPU) {
    if constexpr (nn::dsl::has_reduction_v<E>) {   // 含归约/广播/matmul/索引视图
        ... to_expr_spec(e) → eng.eval_expr(spec, ...)   // ← 解释器
    }
    return eval_cpu(e, rows, cols);                // ← 模板内联
}
```

所以"CPU 该被编译器内联"这件事**设计上就是成立的**，而且已经实现。
解释器只在 `has_reduction_v<E> == true` 时启用。

### 8.2 事实二：归约在 `eval(i)` 契约下无解

模板 AST 的求值契约是 `Scalar eval(std::size_t i)`——"给我第 i 个输出元素"。
`ReduceRef::eval` 直接返回 `Scalar{0}` 占位（`:444`），`ReduceViewRef::eval` 同样（`:239`），
`MatmulRef::eval` 也是（`:268`）。代码注释给出判据：**"含归约的表达式无法按'逐元素
模板求值'（归约需要全行/全列信息）"**（`:451-453`）。

在一个纯 `eval(i)` 模型里，`row_reduce_sum(exp(x))` 要么每次重算整行（O(n²)），
要么需要一套不同的调度（先算归约向量，再算逐元素链）。

**注意**：这就要求"两阶段模板调度"——但这在 C++ 模板里是可做的，**不是非解释器不可**。

### 8.3 事实三：真正的原因是 GPU 的闭合世界

GPU 没有运行时编译器：表达式必须折叠成**数据**（`ExprSpec` IR），
构建期 `scan_exprs` 枚举 → `gen_fused` 生成融合 shader → 运行时按 key 精确匹配
（`expr_dsl.hpp:642-644`、`expr_spec.hpp` 头注释、AGENTS.md §5 铁律 7）。

一旦"每个表达式都要有数据形态"成为硬需求，最省事的做法就是**让 CPU 也消费同一份
`ExprSpec`**，而不是给 CPU 单独维护一套类型化 schedule。于是：

- 判据被从"是不是归约"放宽成"**能不能塞进纯逐元素 `eval(i)` 模型**"——
  所以 `MatmulRef`（`MatmulRef` 根本不是归约）也被标成 `has_reduction_v = true`（`:462-464`），
  连本能内联成循环的 matmul 一起被推进解释器；
- 执行一份数据 IR，**只能解释**。

即：`has_reduction_v` 这条分流不是性能决策，而是"CPU 复用 GPU 表示"的架构副作用，
注释自己承认是 **"先正确后优化"**（`:673-676`）。

### 8.4 附带问题：连内联的那条路也没内联好

`TensorRef::eval` 是 `t.cpu_matrix().span()[i]`（`:163`）：**每个元素**都重新调用
`cpu_matrix()`、重新构造 `span`；`RowModRef`/`RowAccessRef`/`RotateHalfRef`
每元素还要做 `i / cols`、`i % cols`（`:173-217`）。这些 per-element 重导出让编译器
无法把这些循环降成干净的向量化——正是 §2.5 里 1M 元素 `exp(x)*x+1` 要 40ms 的原因。

### 8.5 结论

CPU 端**不该有解释器**，缺的是"IR → 专门化/内联 kernel"的下沉层：

1. 构建期 AOT 生成 C++ kernel（与被删的 `cpu_emitter.hpp` 对称，与 `gen_fused` 并列）；
2. 运行期按结构类别分派到手写的、显式 SIMD + 并行的 kernel
   （纯逐元素 / 行归约 / 列归约 / matmul-链）。

两者都是"让编译器内联/专门化"，正是期望的形态。缺的不是技术可行性，
而是这一层在"CPU 复用 GPU IR"的选择里被顺带省掉了。
