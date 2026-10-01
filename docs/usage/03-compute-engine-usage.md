# 计算引擎（ComputeEngine）完整用法手册

> **定位**：本手册是 `nn::ComputeEngine` 的**唯一权威用法文档**。目标是"不读演示代码、不读实现代码，只读本手册就能完整使用计算引擎"。
>
> **范围**：`ComputeEngine` 全部公共入口 + `nn::dsl` 表达式层全部书写入口 + 使用它们所必需的类型契约（`Tensor` / `Precision` / `Result` / `InitSpec`）。
>
> **不覆盖**：引擎内部实现（`*_impl`、shader、内存池算法）、Layer/Loss/Optimizer 的算法语义、IR/代码生成管线细节。这些见 §23"相关文档"。
>
> **版本锚**：以 `include/neuralnet.cpp/compute_engine.hpp`（2087 行）、`expr_dsl.hpp`（1573 行）、`compute_tensor.hpp`、`precision.hpp`、`core_config.hpp`、`core_errors.hpp` 的当前实现为准。任何与本手册不符的代码行为以源码为准，并请同步修订本手册。

---

## 目录

1. [心智模型：三层职责](#1-心智模型三层职责)
2. [最小可运行骨架](#2-最小可运行骨架)
3. [基础契约（必读）](#3-基础契约必读)
4. [张量工厂与生命周期](#4-张量工厂与生命周期)
5. [I/O：宿主 ↔ 张量](#5-io宿主--张量)
6. [精度操作](#6-精度操作)
7. [激活 offload 与内存控制](#7-激活-offload-与内存控制)
8. [表达式 DSL（核心编程模型）](#8-表达式-dsl核心编程模型)
9. [矩阵级原语](#9-矩阵级原语)
10. [归约原语](#10-归约原语)
11. [数据搬运原语](#11-数据搬运原语)
12. [扫描原语（RLA/RAPT）](#12-扫描原语rlarapt)
13. [表达式直接求值（高级）](#13-表达式直接求值高级)
14. [批处理与内存控制](#14-批处理与内存控制)
15. [异步标量回读](#15-异步标量回读)
16. [L2 宿主桥](#16-l2-宿主桥)
17. [引擎能力查询](#17-引擎能力查询)
18. [怎样用引擎把 Layer 写简单](#18-怎样用引擎把-layer-写简单)
19. [方法速查总表](#19-方法速查总表)
20. [诊断与环境变量](#20-诊断与环境变量)
21. [故障速查表](#21-故障速查表)
22. [不存在 / 已废弃的 API](#22-不存在--已废弃的-api)
23. [相关文档](#23-相关文档)

---

## 1. 心智模型：三层职责

```
Layer / Loss / Optimizer   ← 只写算法（"ReLU = max(x,0)"、"Attention = softmax(QKᵀ/√d)·V"）
        ↓ 用原语与表达式组合
表达式 DSL (nn::dsl)       ← 逐元素 / 广播 / 归约 / matmul 的【唯一编程模型】
        ↓ 折叠
ComputeEngine              ← op-level 原语，永远不知道 "ReLU" / "Attention" 是什么
```

三条不可违反的规则（决定你写代码时"该调谁"）：

| 规则 | 含义 | 后果 |
|---|---|---|
| **算法只在 Layer** | Engine 与 shader 永不含算法语义，也不含算法名 | 想加 ReLU/LayerNorm/Attention → 用现有原语**组合**，不加引擎算子 |
| **逐元素一律走 DSL** | 逐元素、广播、归约、matmul 融合 → `dsl::compute*`；不要用 `engine.add_inplace` 把多步串起来 | 直调 `add_inplace` 只用于"纯累加、无表达式"的场合（如优化器 `m += g`） |
| **Layer 只用引擎的 4 类直调** | 基础设施（`begin_batch`/`zero`/`create_tensor`…）、数据搬运（`slice_rows`/`im2col`…）、状态扫描（`scan_*`）、fold 登记（`eval_expr`） | 其余计算类调用出现在 Layer 里 = 设计走偏 |

> **引擎不知道"层"**：`ComputeEngine` 没有 `add_layer`、没有"matmul + 激活"的融合算子。所有融合表达在 Layer 的表达式里。

### 1.1 两套执行机制，一套写法

| 设备 | 表达式如何执行 |
|---|---|
| CPU | 编译期模板内联求值（编译器 SIMD 融合），等价手写循环；含 matmul/归约的表达式先"预绑定"再内联 |
| GPU | 折叠成 `ExprSpec` → 按 key 匹配**构建期预生成**的融合 shader（闭合世界，未命中硬报错，无 runtime 编译） |

**你的代码在两者上完全一样**——只写表达式，不写设备分支。差异只体现在性能与 `P`（精度）上。

---

## 2. 最小可运行骨架

### 2.1 头文件与命名空间

```cpp
#include <neuralnet.cpp/nn.hpp>   // 唯一入口；已含 expr_dsl / compute_cpu_engine / compute_gpu_engine
```

- `nn::Scalar` = `float`（`core_config.hpp`）。
- `nn::Precision` / `nn::Tensor` / `nn::ComputeEngine` 均在 `nn`。
- DSL 全部在 `nn::dsl`。**建议显式限定**：`nn::dsl::exp(...)`、`nn::dsl::max(...)`、`nn::dsl::select(...)`——避免与 `<cmath>` 的 `::exp`、`std::max` 歧义。运算符（`+ - * / > < …`）靠 ADL 自动找到。

### 2.2 创建引擎

**CPU**：

```cpp
nn::CpuEngine engine;          // 无参，立即可用
```

**Vulkan GPU**（`GpuEngine` **不能**无参构造，必须绑定已初始化的 `GpuBackend`）：

```cpp
#ifdef NN_HAS_VULKAN
    nn::GpuBackend& backend = nn::GpuBackend::instance();   // 进程内单例
    if (auto r = backend.initialize(/*device_selector=*/""); !r)
        return std::unexpected(r.error());                  // 空 = 自动选卡
    nn::GpuEngine engine(backend);
    // 设备选择优先级：initialize(sel) > 环境变量 NN_VULKAN_DEVICE > 自动打分
    // sel 可以是索引 "2" 或名称子串 "40HX"/"NVIDIA"
#else
    // 未编译 Vulkan 支持
#endif
```

> ⚠ **常见错误**：写 `nn::GpuEngine engine;`（无参）——编译失败。本文件旧版本曾这样写；`docs/usage/01-quickstart-model.md:295` 的 `❌` 反例也用了这一行（那里本意是演示 `cpu_matrix` 私有化，但该行本身同样编译不过）。
>
> ⚠ **生命周期**：`engine` 必须晚于任何持有其非拥有指针的对象（`nn::Model`、`Layer`、`Tensor::engine_`）析构。`GpuBackend` 必须比 `GpuEngine` 活得更久。

**统一工厂**（CLI 入口用的写法，可直接复用）：

```cpp
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

nn::cli::EngineConfig cfg{/*use_gpu=*/true, /*gpu_device=*/""};
auto eng_r = nn::cli::create_engine(cfg);
if (!eng_r) return std::unexpected(eng_r.error());
std::unique_ptr<nn::ComputeEngine> engine = std::move(*eng_r);
```

该工厂在**显式请求 GPU 但初始化失败时返回错误、绝不静默回退 CPU**（保持"硬报错、不降级"）。

### 2.3 第一个完整程序

```cpp
#include <neuralnet.cpp/nn.hpp>

nn::Result<void> demo()
{
    nn::CpuEngine engine;

    // 1) 造输入：Matrix(3,4) → Tensor
    nn::Matrix A(3, 4);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            A.set_value(r, c, static_cast<float>(r + 1) * 0.1f);

    auto a = engine.from_matrix(A);                  // Result<Tensor>
    if (!a) return std::unexpected(a.error());

    // 2) 权重 (2,3)，声明式初始化
    nn::Tensor w = engine.create_tensor(2, 3, nn::Precision::F32,
                                        nn::InitSpec::uniform(-0.5f, 0.5f, nn::kInitSeed));
    if (!w.valid()) return std::unexpected(nn::Error{"create_tensor failed"});

    // 3) 融合表达式：y = tanh(W·a)     ← matmul 段 + 逐元素段，单个 kernel
    auto y = nn::dsl::compute(engine, nn::dsl::tanh(nn::dsl::matmul(w, *a)), 2, 4);
    if (!y) return std::unexpected(y.error());

    // 4) 取回宿主
    auto m = engine.to_matrix(*y, nn::Precision::F32);
    if (!m) return std::unexpected(m.error());
    std::printf("y(0,0)=%f  shape=%s\n", m->at(0, 0), y->shape_str().c_str());
    return {};
}
```

要点：
- **每一步都可能失败**，`Result<T>` 必须检查（`-fno-exceptions`，全库禁止 throw/try/catch）。
- 表达式**必带输出形状**：`nn::dsl::compute(engine, expr, rows, cols)`。
- `dsl::matmul` 是表达式叶子，输出形状由你给（这里是 `(2,4)`）。

### 2.4 错误处理范式

```cpp
// 顶层入口
nn::Result<Tensor> f(nn::ComputeEngine& engine)
{
    auto a = engine.create_tensor(4, 4);
    if (!a.valid())
        return std::unexpected(nn::Error{"create_tensor 失败"});

    if (auto r = engine.zero(a); !r)      // void 型：就地检查
        return std::unexpected(r.error());

    return nn::dsl::compute(engine, nn::dsl::exp(nn::dsl::leaf(a)), 4, 4);
}
```

- `nn::Error` 只有 `std::string message` 一个成员。
- 想让"丢弃 Result"编译期告警：用 `nn::check_result(...)` 包一层。
- 数值解析用 `nn::parse_number<T>(sv)`（不抛异常），不要用 `std::stoi/stod`。
- 少数不可恢复点用 `NN_ASSERT` fail-fast（如 `Tensor::engine()` 在未绑定张量上调用）。

---

## 3. 基础契约（必读）

### 3.1 `nn::Tensor` 的公共面

`Tensor` 是跨设备统一句柄（CPU 持 `Matrix<P>`，GPU 持 `GpuTensor<P>`，内部 `std::shared_ptr` → 拷贝廉价、零拷贝传递）。

**公共可用的只有这些**：

```cpp
nn::Device    device()    const noexcept;   // Device::CPU / Device::GPU
nn::Precision precision() const noexcept;   // F16 / F32
std::size_t   rows()      const noexcept;
std::size_t   cols()      const noexcept;
std::size_t   size()      const noexcept;   // rows*cols
bool          is_cpu()    const noexcept;
bool          is_gpu()    const noexcept;
bool          valid()     const noexcept;   // 存储存在且与 precision() 匹配
std::string   shape_str() const;            // "(rowsxcols)[CPU]f32"，调试用
bool          bound()     const noexcept;   // 是否已出生绑定到某引擎
nn::ComputeEngine& engine() const;          // 取绑定引擎；未绑定 → NN_ASSERT
```

**存储访问器全部私有**（`cpu_matrix()` / `cpu_shared()` / `gpu_tensor()` / `gpu_shared()`，以及静态工厂 `Tensor::from_matrix` / `Tensor::cpu` / `Tensor::cpu_uninitialized` / `Tensor::from_gpu`）：

> **铁律 #11**：一切 Tensor 的创建/读写/reshape **必须经 `ComputeEngine`**。绕过 = **编译错误**（不是运行时错误）。库内豁免仅两类：引擎 friend 与 `detail::TensorAccess`（**库外禁用**，DSL 求值器专用通道）。

所以：
- 造张量 → `engine.create_tensor` / `engine.from_matrix` / `dsl::compute` / `engine.clone` …
- 读张量 → `engine.to_matrix` / `engine.read` / `engine.get_index` / `detail::download_span`
- 写张量 → `engine.write` / `engine.copy_from` / `engine.set_index` / `engine.cast_into` / 引擎原语
- 改形状 → `engine.reshape`（**不是** `t.reshape()`，那个也已私有）

### 3.2 内存布局（必须背下来）

| 层 | 布局 |
|---|---|
| `nn::Matrix`（L1 代数层，宿主 I/O 载体） | **行主序** `(rows, cols)`，`data[row*cols + col]` |
| 神经网络 `Tensor` 逻辑约定 | **列主序 batch-major**：输入 `(feature_dim, batch_size)`，每列一个样本 |
| 权重 | `(out_features, in_features)` |
| GPT 序列展平 | **列序 `i = b*seq + t`（batch 在列方向）** |

> **铁律 #5**：`i = b*seq + t` 是全局唯一约定。position-major（`i = t*batch + b`）会造成跨样本串扰。**所有注意力/序列相关测试必须覆盖 `batch>1`**（batch=1 时两种布局重合，测不出）。

`Tensor` 本身只是"行主序的二维数字网格"——"列主序"是**语义约定**，由使用者的维度解释决定。`engine.matmul(A,B)` 按标准 `(M,K)×(K,N) → (M,N)` 计算，`dsl::row_broadcast` 沿"行"广播等，都建立在这个约定上。

### 3.3 精度系统

```cpp
enum class nn::Precision : std::uint8_t { F16 = 0, F32 = 1, BF16 = 2, F64 = 3 };
// 仅 F16 / F32 可用；BF16 / F64 是保留值，使用即报错（check_precision_supported）

struct nn::PrecisionProfile {
    Precision param     = Precision::F32;   // 权重 / 嵌入表 / 参数存储
    Precision compute   = Precision::F32;   // 常规算子（matmul / 逐元素 / 数据操作）
    Precision stable    = Precision::F32;   // 数值敏感算子（softmax / Norm / loss）
    Precision optimizer = Precision::F32;   // 优化器状态（m / v / momentum）
};
// 配方：profile_f32() / profile_master_weights() / profile_f16() / profile_all_f16()
```

**P 精度算术的正式定义**（一切精度行为的语义锚）：

> 对 `P ≠ F32`：以 **f32 参考精度计算** + 每个算子输出**舍入到 P**（round-half-to-even）；matmul / 归约类额外：**累加精度 = f32**。
> 对 `P = F32`：参考即自身，不舍入。

由此得到写代码时唯一要记的口诀：

> **f16 只改变"数据存哪儿"，不改变"算子怎么算"。** f16 的边界处理全部集中在引擎公共入口（NVI 边界 cast 层），你不需要、也不应该自己 cast 操作数。

**哪些入口带 `P`，哪些不带**（决定你传不传精度）：

| 类别 | 是否带 `P` | 语义 |
|---|---|---|
| 运算类：`matmul` / `batched_matmul` / `matmul_with_bias` / 归约 / `scan_*` / `outer_col` / `eval_expr*` | **带**（`Precision P = Precision::F32`） | `P` = 输出（及计算目标）精度 |
| 纯数据搬运：`transpose` / `slice_rows` / `insert_rows` / `gather_rows` / `scatter_add_rows` / `rearrange_3d` / `im2col` / `col2im` / `clone` | **不带** | **输出精度 = 源精度** |
| in-place：`add_inplace` / `scale_inplace` / `zero` / `accumulate` / `eval_expr_into` / `dsl::compute_into` | **不带** | **存储精度不可变**（`dst` 精度即输出精度） |
| 唯一变精度算子 | `cast(src, dst)` / `cast_into(src, dst)` | 永远显式 |

**没有 Auto 推导**：`P` 是唯一可见实参，来源可追溯。Layer 按约定显式传 `p_.compute` / `p_.stable`。

`nn::f16` 是 IEEE binary16 值类型：2 字节、与 GPU R16F 内存布局一致（**同精度 CPU↔GPU 传输 = 原始字节拷贝**）、算术为"f32 参考 + 舍入到 f16"、`static_cast<float>(x)` 升位精确无损。所有 f32↔f16 转换统一 **round-half-to-even**。

### 3.4 出生绑定与跨引擎检查

张量在**引擎入口产物处自动绑定**创建它的引擎（`Tensor::engine_` 观察者，非拥有）。绑定随拷贝/reshape 传播。

- `engine` 的每个公共入口都做 `bind_check_`：本轮调用的所有操作数中，**双方都绑定且引擎指针不同 → `Result` 硬错误**；单侧未绑定按库外豁免放行。
- 设置 `NN_BIND_DEBUG=1`（进程启动前）后，**未绑定输入进引擎**也报错，用来抓漏网。错误带 `source_location`（file:line）与张量形状，并同步打到 stderr。
- **跨设备互传只准经 `import`**（见 §5.4）。`ensure_gpu` 这个名字已不存在（改名即防线）。
- 引擎生命周期必须长于其创建的所有张量（一期不加析构探测）。

### 3.5 GPU 批处理录制生命周期（铁律 #6）

```cpp
engine.begin_batch();   // GPU：开始录制到 command buffer；CPU：no-op
engine.end_batch();     // GPU：提交 + fence wait；CPU：no-op
engine.flush_batch();   // GPU：提交当前 buffer 并等待，然后自动开新帧；CPU：no-op
```

> **录制期（`begin_batch`→`end_batch` 之间）引用的所有张量必须存活到 `end_batch()` 之后**。局部张量在 `end_batch()` 前析构 → `VK_ERROR_DEVICE_LOST`。

其它录制期规则：
- `read` / `to_matrix` 在 GPU 上**隐含 flush + 同步**（`end_batch` → `wait_in_flight` → 开新帧）。逐 step 的 loss 回读请用 `submit_scalar_readback` 快路（§15），不要用 `read`。
- `write` 覆盖既有存储、**不替换对象**；GPU 录制窗口内走 `copy_from` 的既有 drain 语义。
- 不要在录制期读一个"还没被排入命令的执行结果"。

---

## 4. 张量工厂与生命周期

### 4.1 `create_tensor` —— 纯分配（3 参）

```cpp
[[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols,
                                  Precision P = Precision::F32);
```

- 返回 `Tensor`（**不是 `Result`**）。失败返回**无效张量** → 用 `t.valid()` 检查。
- **语义 = 纯分配**：CPU 为零填充，GPU 分配**未初始化**。需要零请显式 `engine.zero(t)` 或用下面的 4 参重载。
- 读之前必须把**全部**元素写满（`Uninitialized` 同款契约）。

```cpp
nn::Tensor t = engine.create_tensor(64, 128);                       // f32
nn::Tensor h = engine.create_tensor(64, 128, nn::Precision::F16);   // f16
if (!t.valid()) return std::unexpected(nn::Error{"oom"});
```

### 4.2 `create_tensor` + `InitSpec` —— 声明式初始化（4 参，推荐）

```cpp
[[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols, Precision P,
                                  const nn::InitSpec& spec);
```

```cpp
struct nn::InitSpec {
    enum class Kind { Uninitialized, Zero, Constant, Uniform, Normal };
    Kind kind = Kind::Zero;
    Scalar value = 0;                 // Constant
    Scalar lo = 0, hi = 0;            // Uniform
    Scalar mean = 0, stddev = 0;      // Normal
    std::uint64_t seed = 42;          // 分布种子（Uninitialized/Zero/Constant 不消费）

    static InitSpec uninitialized();
    static InitSpec zero();
    static InitSpec constant(Scalar v);
    static InitSpec uniform(Scalar lo, Scalar hi, std::uint64_t seed);       // seed 必填
    static InitSpec normal(Scalar mean, Scalar stddev, std::uint64_t seed);  // seed 必填
};
inline constexpr std::uint64_t nn::kInitSeed = 42;   // 层传的显式默认 seed
```

**分工（铁律 #3）**：**层算分布参数**（limit / mean / stddev 这类层语义留在 Layer），**引擎负责填数**。初始化策略（host 生成后上传 vs 设备端原生生成）由引擎自选，调用方不可见。

**确定性保证**：引擎内部把 `seed` 与"本次创建的序号"混流（splitmix64），所以——
- 同 `seed` 的不同张量**不撞流**（同形状多层不互为镜像）；
- 创建顺序 = 模型构造顺序 → **跨进程初值逐字节确定**（铁律 #8）。

```cpp
// Linear 权重：U[-limit, limit]
const float limit = std::sqrt(6.0f / static_cast<float>(in + out));
nn::Tensor w = engine.create_tensor(out, in, p_.param,
                                    nn::InitSpec::uniform(-limit, limit, nn::kInitSeed));
// LayerNorm 的 beta
nn::Tensor beta = engine.create_tensor(features, 1, p_.param, nn::InitSpec::zero());
```

### 4.3 失败契约小结

| 入口 | 失败方式 |
|---|---|
| `create_tensor`（3/4 参） | 返回**无效张量**（`valid() == false`）；4 参的 Constant/Uniform/Normal 分支出错也返回空张量 |
| `from_matrix` / `clone` / `cast` / `reshape` / 各类搬运与运算 | 返回 `Result<T>` 错误 |
| `*_inplace` / `write` / `insert_rows` / `scatter_add_rows` | 返回 `Result<void>` |

### 4.4 `reshape`

```cpp
[[nodiscard]] Result<Tensor> reshape(const Tensor& t, std::size_t new_rows, std::size_t new_cols);
```

- 元素数不匹配 → 返回错误 `"reshape: element count mismatch"`。
- **CPU/GPU 语义不同（高频坑）**：GPU **共享底层 buffer（零拷贝视图）**；CPU `Matrix` 无零拷贝视图能力 → **按 `precision_` 复制数据到新形状**。
- 绑定随 `src` 传播。

### 4.5 `clone`

```cpp
[[nodiscard]] Result<Tensor> clone(const Tensor& src);   // 无 P：输出精度 = 源精度
```

深拷贝（CPU 矩阵拷贝 / GPU buffer 拷贝，**无 PCIe 传输**）。用途：需要修改中间结果但不影响原张量。

### 4.6 `copy_from` —— f32 Matrix → 已有张量

```cpp
[[nodiscard]] Result<void> copy_from(Tensor& dst, const nn::Matrix& src);
```

- 覆盖 `dst` 的**既有存储**（不替换对象）。
- f32 目标：直接上传。**f16 目标**：先按 f32 上传，再 `cast_into` 写进 `dst` 原存储（f32→f16 用 RHE）。
- 语义定位：这是"**f32 宿主载体 → 张量的转换填充入口**"，序列化加载用它。
- **L2+（Layer/Loss/Optimizer/Model）禁用**（铁律 #12，见 §16）。

---

## 5. I/O：宿主 ↔ 张量

引擎的 I/O 分组固定为这 7 个动词（**宿主数据进出张量只准经本组**）：

```
from_matrix / to_matrix / import / read / write / get_index / set_index
```

分层口径：
- **库外**（tests / tools / CLI / 序列化）只准出现本组动词；
- **计算路径**（Layer / Loss / Optimizer / Model）禁用 **Matrix 型三动词**（`from_matrix` / `to_matrix` / `copy_from`），批量 `read`/`write` 与宿主桥（§16）受控可用；
- `import` 是跨设备拉取入口。

### 5.1 `from_matrix` / `to_matrix`

```cpp
[[nodiscard]] Result<Tensor> from_matrix(const nn::Matrix& m, Precision P = Precision::F32);
[[nodiscard]] virtual Result<nn::Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) = 0;
```

| | 语义 |
|---|---|
| `from_matrix(m, P)` | 上传 `Matrix(rows, cols)`，按 `P` 建张量。`P=F16` 时做 f32→f16 **RHE 舍入**。返回张量形状 = `m.rows() × m.cols()` |
| `to_matrix(t, P)` | **返回的永远是 f32 `Matrix`**（行主序宿主载体）。`P` 是"要求下载精度"：`P=F32`（默认）= f16 存储升位读取（精确无损）；`P=F16` = 数据先按 f16 表示（f32 张量会先降位再升回，用于观察 f16 舍入误差） |

- `to_matrix` 在 GPU 上**隐含 flush + 同步**（与 `read` 同路）。
- `to_matrix` 是**唯一仍是公共虚函数的 I/O 入口**（其它 I/O 都在基类做 NVI）。

> ⚠ **已知边界（GPU + f16）**：`to_matrix(f16 张量, Precision::F16)` 在 GPU 上会落到 F32 存储槽位并触发 `NN_ASSERT`（`compute_gpu_engine.hpp:245-259` 的 `if (P == F32)` 分支未覆盖"f16 张量 + `P == F16`"，会 fall through 到 `t.gpu_tensor()`）。**结论：f16 张量在 GPU 上一律用默认 `P=F32`（或 `read`）**。CPU 上 `P=F16` 正常工作，见 `src/tensor_precision_test.cpp:85`。

### 5.2 `read` / `write` —— 批量（本体）

```cpp
template <class T>
[[nodiscard]] Result<void> read(const Tensor& t, std::span<T> dst);     // 张量 → span

template <class T>
[[nodiscard]] Result<void> write(Tensor& t, std::span<T> src);          // span → 张量既有存储
```

**U2 精确匹配（必须记牢）**：

| span 元素类型 | 只能配 |
|---|---|
| `float`（`nn::Scalar`） | `Precision::F32` 张量 |
| `nn::f16` | `Precision::F16` 张量 |

- 错配 → **运行期返回错误**；元素类型不是 `float`/`f16`（如 `double`）→ **编译期 `static_assert`**。这是防 f16/f32 槽位错位的关键。
- 元素数必须等于 `rows*cols`，否则返回错误。
- `read`：GPU 上**隐含 flush + 同步**（走 `to_matrix` 同路）。f16 存储先升 f32 再按需 RHE 落回 → **往返位不变**。
- `write`：覆盖既有存储，**不替换对象、不改形状/精度**。CPU 直写既有存储；GPU 走 `copy_from` 的 drain 语义。写入的张量必须存活到 `end_batch()` 之后。
- **批量是本体**：热路径请攒批量。

```cpp
std::vector<float> buf(rows * cols);
std::fill(buf.begin(), buf.end(), 1.0f);
nn::Tensor t = engine.create_tensor(rows, cols, nn::Precision::F32);
if (auto r = engine.write(t, std::span(buf)); !r) return std::unexpected(r.error());

std::vector<float> back(rows * cols);
if (auto r = engine.read(t, std::span(back)); !r) return std::unexpected(r.error());

// f16 张量必须用 span<nn::f16>
nn::Tensor h = engine.create_tensor(rows, cols, nn::Precision::F16);
std::vector<nn::f16> hbuf(rows * cols);
if (auto r = engine.write(h, std::span(hbuf)); !r) return std::unexpected(r.error());
```

> `std::span` 是必需形参：`engine.write(t, buf)`（直接传 `std::vector`）**不会**编译。

### 5.3 `get_index` / `set_index` —— 索引（语法糖）

```cpp
[[nodiscard]] Result<Scalar> get_index(const Tensor& t, std::size_t row, std::size_t col);
[[nodiscard]] Result<void>   set_index(Tensor& t, std::size_t row, std::size_t col, Scalar v);
```

- 行主序 `(row, col)`；越界 → 错误。
- `get_index`：f16 存储**精确提升**为 `Scalar`。
- `set_index`：f16 存储按 **RHE 舍入**（与 `from_matrix` / 边界 cast 同口径）。
- **不给热循环承诺**：CPU 是宿主直读写；**GPU 每次调用 = 一整轮 staging 往返**（`read → 改一格 → write`）。热路径请攒批量用 `read`/`write`。

### 5.4 `import` —— 跨设备/引擎拉取

```cpp
[[nodiscard]] Result<Tensor> import(const Tensor& src);              // 保持 src 精度
[[nodiscard]] Result<Tensor> import(const Tensor& src, Precision P);
```

| 情形 | 行为 |
|---|---|
| 同设备 + 同精度 | **零拷贝别名**（返回共享同一存储的新句柄） |
| 同设备 + 异精度 | 引擎内 `cast` |
| 跨设备 | 经宿主中转（基类默认）；`GpuEngine` override 为"CPU → 设备直传"（省一次宿主拷贝） |

- **返回新句柄，绝不改写调用方的 `src`**。需要就地重绑定必须显式写：

```cpp
auto up = engine.import(cpu_tensor);          // cpu_tensor 不变
if (!up) return std::unexpected(up.error());
nn::Tensor gpu_tensor = std::move(*up);

// 或就地重绑定（显式）
if (auto r = engine.import(t); r) t = std::move(*r);
```

- 与 `cast_into` 同族红线：**不替换调用方对象**。
- 命名即防线：跨设备互传**只准经 `import`**（`ensure_gpu` 已全部改名）。

---

## 6. 精度操作

```cpp
// 唯一"变精度"算子。永远显式。
[[nodiscard]] Result<Tensor> cast(const Tensor& src, Precision dst);
// 同精度 → 返回 src（共享所有权，零拷贝）；跨精度 → 引擎实现（CPU 逐元素转换、GPU 原生转换）
// 升 cast（f16→f32）精确无损；降 cast（f32→f16）round-half-to-even

// 把 src 转换后写入 dst 的既有存储（不替换 dst 对象、不改底层 buffer）
[[nodiscard]] virtual Result<void> cast_into(const Tensor& src, Tensor& dst);
// 要求 rows/cols 一致；默认实现：同精度走 copy_into，跨精度报错（引擎覆盖）

// 同精度同形状拷贝（dst 既有存储被完整覆盖，不替换对象）
[[nodiscard]] virtual Result<void> copy_into(Tensor& dst, const Tensor& src);
```

**`cast` 与 `cast_into` 的区别是"落点"**：

- `cast` 返回**新张量**；
- `cast_into` 保留 `dst` 的**对象身份与底层 buffer**。

为什么必须有 `cast_into`：边界 cast 做 f16 原地更新时，若替换 `dst` 对象，其它持有同一张量句柄的缓存会**静默失联**。

```cpp
// 显式降精度（把 f32 权重换成 f16 存储）
auto w16 = engine.cast(*w32, nn::Precision::F16);
if (!w16) return std::unexpected(w16.error());

// 原地写回（保留对象身份）
if (auto r = engine.cast_into(*src_like, *dst_like); !r) return std::unexpected(r.error());
```

> **不要手写 cast 来"适配" f16**：引擎公共入口已按能力自动做边界 cast。`dsl::compute` 的 `P` 已经表达"输出存储精度"，引擎会在无原生 f16 变体时自动"抬 f32 算 → 按 P 落回"。手动 cast 只会平白多出全尺寸临时量。

---

## 7. 激活 offload 与内存控制

### 7.1 激活 offload（GPU 特性，CPU 为 no-op）

```cpp
[[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes);
// GPU：持久 host-visible slab（跨 step 复用）；CPU：返回 1×1 占位张量

[[nodiscard]] Result<void> offload_save(const Tensor& buffer, std::size_t offset,
                                       const Tensor& src);
// 把 src 复制到 buffer 的 offset（float 单位）处
// 边界 cast 入口：f16 激活在写入 slab 前抬到 f32（slab 恒 f32 存）

[[nodiscard]] Result<Tensor> offload_restore(const Tensor& buffer, std::size_t offset,
                                            std::size_t rows, std::size_t cols);
// 从 buffer 的 offset 处复制 rows×cols 到新张量
```

- `set_offload_enabled()` **不存在**。开关在 Layer 侧：`GPTModel` / `RAPTModel` 的 `set_activation_offload(bool)`，底层由 `ActivationOffloader`（`compute_layer_base.hpp`）调用这三个引擎入口。
- **slab 容量必须按"本次导出的激活总量"校验**（批大小/序列长度变化、`--resume`、最后一个不满 batch 都会改变总量）；容量不足会越界写 → 缓冲区破坏 / 设备丢失。`ActivationOffloader` 已按此实现（每次导出重算 `needed`，不足则重建）。
- `restore` 出来是 f32（slab 恒 f32 存）；f16 激活按 f32 存、不享 f16 折半收益。

```cpp
auto slab = engine.create_offload_buffer(needed_floats * sizeof(float));
if (!slab) return std::unexpected(slab.error());
std::size_t offset = 0;
if (auto r = engine.offload_save(*slab, offset, act); !r) return std::unexpected(r.error());
offset += act.size();
// ... backward 前 ...
auto back = engine.offload_restore(*slab, 0, act.rows(), act.cols());
if (!back) return std::unexpected(back.error());
```

### 7.2 内存统计

```cpp
[[nodiscard]] virtual std::string pool_stats() const;
```

- **GPU**：`persist{…} transient{…} pending=…`
- **CPU（无池引擎）**：基类默认返回宿主直配账本
  `direct{ blocks=… live_bytes=… peak_bytes=… total_blocks=… total_bytes=… }`

**两引擎同一动词、都非空**——训练日志一套代码读两个引擎。

### 7.3 显存回收

```cpp
[[nodiscard]] virtual Result<void> release_idle_pool_blocks();
```

- GPU：在 `end_batch()`（提交完成、延迟销毁已 flush）之后归还完全空闲的内存池底材。
- CPU：**语义成立的 no-op**（直配模式没有可归还的整块）。
- 典型用法：评估分块前向时每块调用，防大 batch 评估 OOM。

---

## 8. 表达式 DSL（核心编程模型）

> **一句话**：DSL 表达式就是**普通 C++ 表达式树**（值语义、编译期类型），由 `nn::dsl::leaf(tensor)` 引用输入、`nn::dsl::rparam(v)` 引用运行时标量，用普通数学运算符拼出来，交给 `dsl::compute` / `dsl::compute_into` / `dsl::compute_reduce` 求值。
>
> **没有 lambda 形式的提交流水线，没有跨表达式录制**（`start_expr`/`end_expr` 不存在）。融合粒度 = **单条表达式**。

### 8.1 三个求值入口（完整签名）

```cpp
namespace nn::dsl {

// 默认实参在 compute_engine.hpp 的前置声明处给出
template <typename E>
[[nodiscard]] Result<Tensor> compute(ComputeEngine& eng, const E& e,
                                     std::size_t rows, std::size_t cols,
                                     Precision P = Precision::F32);

template <typename E>
[[nodiscard]] Result<void> compute_into(ComputeEngine& eng, const E& e, Tensor& dst);

template <typename E>
[[nodiscard]] Result<Tensor> compute_reduce(ComputeEngine& eng, const E& e,
                                            std::size_t rows, std::size_t cols,
                                            Precision P = Precision::F32);
}
```

| 入口 | 输出 | 形状参数含义 | 精度 | 限制 |
|---|---|---|---|---|
| `compute` | 新张量 | 输出网格 `(rows, cols)` | `P` = 输出存储精度（默认 F32） | **没有 3 参重载**——必须给 `rows, cols` |
| `compute_into` | 写进 `dst`（零分配） | 用 `dst.rows()/dst.cols()` | **无 `P` 形参**，恒 = `dst` 存储精度 | 仅支持**逐元素**表达式（无真归约） |
| `compute_reduce` | 新张量，**归约向量本身** | `(rows, cols)` 为输入网格 | `P` = 输出向量精度 | 归约轴须为 0/1；行归约 → `(rows,1)`，列归约 → `(1,cols)` |

**`compute` 的分派（你不需要写分支，但要理解行为差异）**：

| 条件 | 路径 |
|---|---|
| CPU 且表达式**不含**归约/matmul/索引叶子 | 编译期模板内联求值（SIMD 融合）。**无任何形状校验**（见 §8.7 陷阱 ①） |
| CPU 且含归约/matmul/索引 | 先试"预绑定 + 模板内联"快路径；不行则折叠 `ExprSpec` 走 CPU 解释器（此时**有**形状校验） |
| GPU | 折叠 `ExprSpec` → `validate_expr_spec` → `eval_expr` 匹配**构建期预生成** shader；**未命中 = 硬报错**（闭合世界，无 eager 回退） |

**`compute_into` 的等价改写**（把"原地更新"纳入 DSL，无需逐操作调用引擎原地原语）：

```cpp
dst += expr             →  dsl::compute_into(eng, dsl::leaf(dst) + expr, dst)
dst *= k                →  dsl::compute_into(eng, dsl::leaf(dst) * dsl::rparam(k), dst)
dst += k * other        →  dsl::compute_into(eng, dsl::leaf(dst) + dsl::leaf(other) * dsl::rparam(k), dst)
dst += row_broadcast(v) →  dsl::compute_into(eng, dsl::leaf(dst) + dsl::row_broadcast(v), dst)
grad_w += matmul(gy, x, false, true)
                        →  dsl::compute_into(eng, dsl::leaf(grad_w) + dsl::matmul(gy, x, false, true), grad_w)
```

- `dst` 与某个输入是同一 buffer **是安全的**（逐元素"先读完全部输入再写 `out[i]`"）。
- `dst` 不在 CPU 而引擎是 CPU → 返回错误 `"dsl::compute_into: dst not on CPU"`。
- matmul 段 / 网格索引**不算真归约** → 仍走 CPU 预绑定 + 模板内联（Linear/Conv 反向的 `grad_w += matmul(...)` 正是这一类）。

### 8.2 叶子与常量

```cpp
[[nodiscard]] inline TensorRef    leaf(Tensor t);                 // 线性叶子
[[nodiscard]] inline constexpr RParamLeaf rparam(Scalar v);       // 运行时标量（值不进 key）
[[nodiscard]] inline MatmulRef    matmul(Tensor a, Tensor b,
                                        bool transA = false, bool transB = false,
                                        std::uint32_t batch = 1);
[[nodiscard]] inline RowIdxLeaf   row();                          // 当前输出行号
[[nodiscard]] inline ColIdxLeaf   col();                          // 当前输出列号
[[nodiscard]] inline BatchIdxLeaf batch();                        // 当前批次下标
```

| 入口 | 语义 |
|---|---|
| `leaf(t)` | 按行主序扁平下标读 `t` 的全部元素。**同一表达式内所有普通叶子必须与输出网格 `(rows, cols)` 同形** |
| `rparam(v)` | 运行时标量。参与算术但**值不进 `expr_spec_key`** → 结构固定、值可变 → 同结构共享融合 shader。用于 lr / eps / β / 偏差修正 / 每步会变的 `scale` |
| `row()/col()/batch()` | 网格索引叶子，按 `Scalar` 参与算术。`col() > row()` 做因果掩码、`-slope*(col()-row())` 做 ALiBi |

**常量怎么给（最容易写错的一处）**：

```cpp
// ❌ 编译失败：Scalar 不能单独构成表达式
auto bad  = nn::dsl::compute(eng, 1.0f, r, c);
auto bad2 = nn::dsl::exp(1.0f);

// ✅ 写法 1：放在二元重载里（最常用）
auto a = nn::dsl::compute(eng, nn::dsl::leaf(x) * 0.5f, r, c);
auto b = nn::dsl::compute(eng, nn::dsl::max(nn::dsl::leaf(x), nn::Scalar{0}), r, c);

// ✅ 写法 2：显式常量节点（库里 relu 就是这么实现的）
auto c = nn::dsl::compute(eng, nn::dsl::max(nn::dsl::leaf(x), nn::dsl::ConstLeaf{0.0f}), r, c);
```

**`rparam` vs `ConstLeaf` 的选择（很重要）**：

| | 值是否进 `expr_spec_key` | 用途 |
|---|---|---|
| `dsl::ConstLeaf{v}` | **进** | 结构常量，编译期固定（ReLU 的 0、GELU 的 1.702、掩码的 `-inf`） |
| `dsl::rparam(v)` | **不进** | 运行时标量（lr / eps / β / γ / 每 step 变化的 `scale`）。**运行时值必须走 rparam**，否则破坏 AOT 闭合世界 |

### 8.3 一元函数（全部 7 个 + 一元负号）

```cpp
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto neg(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] auto abs(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] auto exp(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] auto log(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] auto sqrt(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] auto rsqrt(const E& e);   // 1/sqrt(x)
template <nn::dsl::DslExpr E> [[nodiscard]] auto tanh(const E& e);
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto operator-(const E& e);  // 等价 neg
```

**不存在**的：`sin` / `cos` / `erf` / `gelu` / `sigmoid` / `pow` / `clamp` / `where` / `sqr` / `step` / `sum`。

怎么表达缺的：

| 想要 | 写法 |
|---|---|
| GELU（tanh 近似） | `leaf(x) / (1 + exp(-(leaf(x) * 1.702f)))` |
| sigmoid | `1 / (1 + exp(-leaf(x)))` |
| cos / sin | **以张量形式预先算好**，用 `row_mod` 视图读入（RoPE 就是这么做的） |
| clamp(x, lo, hi) | `min(max(x, lo), hi)` |
| 平方 | `leaf(x) * leaf(x)` |

### 8.4 二元运算符与函数

```cpp
// 宏生成，对 OP ∈ {+,-,*,/,>,<,>=,<=,==,!=} 各生成 3 个重载：
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> constexpr auto operator OP(const L&, const R&);  // expr⊗expr
template <nn::dsl::DslExpr E>                     constexpr auto operator OP(const E&, Scalar);     // expr⊗标量
template <nn::dsl::DslExpr E>                     constexpr auto operator OP(Scalar, const E&);     // 标量⊗expr
```

| 运算符 | 折叠算子 | 结果类型 |
|---|---|---|
| `+ - * /` | `Add` / `Sub` / `Mul` / `Div` | 值表达式 |
| `> < >= <= == !=` | `Gt` / `Lt` / `Ge` / `Le` / `Eq` / `Ne` | **`BoolExpression`**（只供 `select` 用） |

```cpp
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> constexpr auto max(const L&, const R&);
template <nn::dsl::DslExpr E>                     constexpr auto max(const E&, Scalar);
template <nn::dsl::DslExpr E>                     constexpr auto max(Scalar, const E&);
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> constexpr auto min(const L&, const R&);
template <nn::dsl::DslExpr E>                     constexpr auto min(const E&, Scalar);
template <nn::dsl::DslExpr E>                     constexpr auto min(Scalar, const E&);
template <nn::dsl::DslExpr E> [[nodiscard]] auto relu(const E& e);   // = max(e, 0)
```

**没有隐式广播**：`+ - * /` 两侧网格必须一致（标量侧由 `ConstLeaf` 承担）。广播只能通过**显式视图**（§8.6）。

### 8.5 `select` —— 唯一的条件选择

引擎**不暴露**条件选择原语，由 DSL 承担：

```cpp
template <nn::BoolExpression C, nn::dsl::DslExpr T, nn::dsl::DslExpr E>
[[nodiscard]] constexpr auto select(const C& c, const T& t, const E& e);
template <nn::BoolExpression C, nn::dsl::DslExpr T>
[[nodiscard]] constexpr auto select(const C& c, const T& t, Scalar ev);
template <nn::BoolExpression C, nn::dsl::DslExpr E>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, const E& e);
template <nn::BoolExpression C>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, Scalar ev);
```

语义：`dst = cond ? then : else`。

**红线**：`cond` 必须是 `BoolExpression`（比较表达式）；而 `select(...)` 自身返回**值表达式**。要嵌套 `select`，必须把内层转回布尔：

```cpp
// ReLU 反向
auto grad = nn::dsl::compute(eng,
    nn::dsl::select(nn::dsl::leaf(x) > nn::Scalar{0}, nn::dsl::leaf(g), nn::Scalar{0}),
    rows, cols);

// 嵌套 select（因果掩码 + 已掩码判定）——注意 "!= Scalar{0}" 的转布尔写法
const auto causal = nn::dsl::select(nn::dsl::col() > nn::dsl::row(), kNegInf_, nn::Scalar{0});
const auto masked = nn::dsl::select(causal != nn::Scalar{0}, kNegInf_, nn::Scalar{0});
```

**函数式概念（可作模板约束）**：

```cpp
template <typename T> concept nn::Expression =
    requires(const T& t, std::size_t i) { { t.eval(i) } -> std::convertible_to<nn::Scalar>; };
template <typename T> concept nn::BoolExpression =
    requires(const T& t, std::size_t i) { { t.eval(i) } -> std::convertible_to<bool>; };
template <typename T> concept nn::dsl::DslExpr =
    nn::Expression<T> && requires(nn::dsl::SpecBuilder& b, const T& t)
    { { t.to_spec(b) } -> std::convertible_to<nn::ExprOperand>; };
```

`DslExpr` 是"能折叠成 `ExprSpec` 的节点"——所有 `dsl::` 自由函数的形参都用它。这就是"普通 `Span` / `ConstSpan` 不是 DSL 表达式、误用会在编译期报错"的机制。

### 8.6 视图 / 广播 / 归约叶子

#### 8.6.1 广播与索引视图

```cpp
[[nodiscard]] inline RowBroadcastRef row_broadcast(Tensor t);    // t: (rows, 1)  → 按行广播
[[nodiscard]] inline ColBroadcastRef col_broadcast(Tensor t);    // t: (1, cols)  → 按列广播

[[nodiscard]] inline RotateHalfRef rotate_half(Tensor t, std::uint32_t block);
[[nodiscard]] inline RowModRef     row_mod(Tensor t, std::uint32_t mod);
[[nodiscard]] inline RowAccessRef  row_access(Tensor t, std::uint32_t offset, std::uint32_t mod);

[[nodiscard]] inline RowGatherRef row_gather(Tensor t, Tensor labels);
[[nodiscard]] inline BatchModRef  batch_mod(Tensor t, std::uint32_t modulo);
[[nodiscard]] inline BatchColRef  batch_col(Tensor t, std::uint32_t per_batch_cols);

[[nodiscard]] inline GroupedReduceRef grouped_reduce_sum(Tensor t, std::uint32_t R);
[[nodiscard]] inline GroupedReduceRef grouped_reduce_max(Tensor t, std::uint32_t R);
```

| 入口 | 源形状契约 | 读取语义（输出网格 `(rows, cols)`） |
|---|---|---|
| `row_broadcast(t)` | `t` = `(rows, 1)` | 每行一个标量 → 按行广播。**gamma / beta / doc_col** |
| `col_broadcast(t)` | `t` = `(1, cols)` | 每列一个标量 → 按列广播。**mean / std_inv / loss 权重** |
| `rotate_half(t, block)` | `t` = `(rows, cols)`，`rows % block == 0`，`block` 偶 | 按 `block` 行分块，块内前后半行交换且前半取负（LLaMA rotate_half） |
| `row_mod(t, mod)` | `t.rows() == mod` | 行取模广播：读 `data[(r % mod)*cols + c]`。**cos/sin 频率表平铺** |
| `row_access(t, offset, mod)` | `t.cols() == cols`，`offset+mod <= t.rows()` | 读 `data[(offset + r % mod)*cols + c]`。**SwiGLU 半切分** |
| `grouped_reduce_sum/max(t, R)` | `t` = `(rows*R, cols)`，`R != 0` | `out[g][c] = Σ/max_{i<R} in[(g*R+i)][c]`。**池化窗口 / 多头分组统计** |
| `row_gather(t, labels)` | `t` 行数任意、`cols` 相符；`labels` 为 `(1, cols)` | 读 `t[uint(labels[c])][c]`。**稀疏 CE 的 `logits[label[c]][c]`** |
| `batch_mod(t, modulo)` | `t.rows() == 1` | 读 `data[batch % modulo]`。**ALiBi 按头斜率** |
| `batch_col(t, per_batch_cols)` | `t` = `(1, batch*param)`，**param 必须等于 matmul 段的 batch** | 读 `data[batch*param + col]`。**doc_ids 按 `(batch,col)` 切片** |

> ⚠ `batch_col` 要求 `(1, BH*seq)` 这类按 `(b,h)` 块重复的形状；写 `(1, batch*seq)` 会**越界**。

**`R`（分组归约长度）是运行期视图参数，不进 `expr_spec_key`**（经 push constant `vp` 槽传入，任一 shader 用运行期循环读取 → 任意池化窗口共享一个融合 shader）；它不是"归约视图"，不参与归约轴判定，走 elementwise 路径。累加顺序固定为 `i` 升序左结合（三端一致，铁律 #8）。

```cpp
// 典型：加偏置（每行同一个偏置）
if (auto r = nn::dsl::compute_into(eng,
        nn::dsl::leaf(out) + nn::dsl::row_broadcast(bias), out); !r) ...

// 典型：SwiGLU
auto gate = nn::dsl::row_access(x, 0, dff);
auto up   = nn::dsl::row_access(x, dff, dff);
auto y = nn::dsl::compute(eng,
    gate * (nn::dsl::leaf(one) / (nn::Scalar{1} + nn::dsl::exp(-up))), rows, cols);
```

#### 8.6.2 归约（两种形态，命名方向必须记准）

```cpp
// 形态 A：对 Tensor 直接归约 → 归约【视图】（GPU 融合更友好）
[[nodiscard]] inline RowReduceSumRef row_reduce_sum(Tensor t);
[[nodiscard]] inline RowReduceMaxRef row_reduce_max(Tensor t);
[[nodiscard]] inline ColReduceSumRef col_reduce_sum(Tensor t);
[[nodiscard]] inline ColReduceMaxRef col_reduce_max(Tensor t);

// 形态 B：对表达式结果归约 → 归约【指令】
template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_sum(const E& e);   // ExprOp::RowSum
template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_max(const E& e);   // ExprOp::RowMax
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_sum(const E& e);   // ExprOp::ColSum
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_max(const E& e);   // ExprOp::ColMax
```

> **命名方向是最高频误读点**：
> - `row_reduce_sum` = **每行出一个标量**（沿列方向归约）→ 结果 `(rows, 1)`
> - `col_reduce_sum` = **每列出一个标量**（沿行方向归约）→ 结果 `(1, cols)`
>
> 参与算术时**自动按行/按列广播**。

- 精度：归约**恒在 f32 参考空间累加**（f16 输入先升 f32），输出按 `P` 舍入。
- 累加顺序固定（行 → 沿列升序；列 → 沿行升序），**并行与串行逐字节一致**。
- 没有 `row_reduce_min` / `col_reduce_min`。

```cpp
// LayerNorm 语义的归约步
auto mean_raw = nn::dsl::compute_reduce(eng,
    nn::dsl::col_reduce_sum(nn::dsl::leaf(x)), F, B, p_.stable);         // → (1,B)
auto mean = nn::dsl::compute(eng,
    nn::dsl::leaf(*mean_raw) * nn::dsl::rparam(1.0f / F), 1, B, p_.stable);
auto diff = nn::dsl::compute(eng,
    nn::dsl::leaf(x) - nn::dsl::col_broadcast(*mean), F, B, p_.stable);
```

### 8.7 五个必须知道的陷阱

**① 模板路径形状零校验**

`dsl::compute` 在 CPU 且表达式不含归约/matmul/索引叶子时直接走模板求值，**不校验叶子与广播视图的形状**。把 `row_broadcast` 传给非 `(rows,1)`、或让 `leaf` 与输出网格不同形 → **静默越界 / 算错**（不是报错）。含归约或走 GPU 的路径才有校验。

> **结论：表达式里的形状由你负责**。写完后请逐个核对叶子的形状。

**② 顶层只有视图/叶子 → 空指令表**

折叠后指令表为空的 spec 会被 `validate_expr_spec` 拒绝（"empty instruction list"），而构建期扫描（`NN_EXPR_SCAN`）遇到这种是**硬失败**（`_Exit(3)`）。所以"视图/叶子单独作根"会炸。惯例是**尾接一个恒等指令**：

```cpp
// ✅ 惯用法：+ rparam(0) 提供一条恒等指令
auto pooled = nn::dsl::compute(eng,
    nn::dsl::grouped_reduce_max(col, static_cast<std::uint32_t>(kk)) + nn::dsl::rparam(0.0f),
    rows, cols);
```

（`dsl::matmul` 段不在此列：纯 matmul 无尾链是合法的。）

**③ `dsl::matmul` 没有 `alpha` 参数**

缩放写成**尾链**（值不进 key，符合 AOT 闭合世界）：

```cpp
// ✅
auto scores = nn::dsl::compute(eng,
    nn::dsl::matmul(q, k, /*transA=*/true, /*transB=*/false, /*batch=*/1) * nn::dsl::rparam(scale),
    rows, cols);

// ❌ 不存在
auto bad = nn::dsl::matmul(q, k, true, false, 1, 0.125f);
```

**④ `select` 的 cond 必须是布尔表达式**（§8.5）。

**⑤ 运行时值禁进常量池**

`scale` / `inv_num_valid` / lr / eps 等每步或每次调用会变的量，**必须走 `rparam`**；用 `ConstLeaf` 会把值编进 `expr_spec_key` → GPU 闭合世界查不到预生成 shader → 硬报错。

### 8.8 `matmul` 表达式叶子

```cpp
[[nodiscard]] inline MatmulRef matmul(Tensor a, Tensor b,
                                      bool transA = false, bool transB = false,
                                      std::uint32_t batch = 1);
```

| 参数 | `false` | `true` |
|---|---|---|
| `transA` | `A` 存储 `(batch*M, K)` | `A` 存储 `(batch*K, M)`，按 `Aᵀ` 用 |
| `transB` | `B` 存储 `(K, N)` | `B` 存储 `(batch*N, K)`，按 `Bᵀ` 用 |

- 输出 = 逐元素链的输入网格 `(rows, cols)`；`batch>1` 时 `rows = batch*M`（A/B 按 batch 垂直切分为连续行块）。
- `k` 与 `batch` 是**形状参数，不进 key**；`transA/transB` 与输入槽位是**结构，进 key**。
- 执行：**CPU** 先用引擎通用 GEMM 物化 C（f16 操作数先抬 f32，C 恒 f32），再交模板路径内联尾链；**GPU** 折叠成 `MatmulSpec` 前置段，与逐元素链**融合进一个 kernel**。

**DSL 里没有的矩阵入口**（它们是**引擎原语**，不是 DSL）：`dsl::add_inplace` / `scale_inplace` / `accumulate` / `transpose` / `batched_matmul` / `matmul_with_bias` / `slice_rows` / `gather_rows` / `zero`。DSL 里的原地写法用 `compute_into`（§8.1）。

---

## 9. 矩阵级原语

```cpp
// C = op(A) × op(B)
[[nodiscard]] Result<Tensor> matmul(const Tensor& A, const Tensor& B,
                                    bool transA = false, bool transB = false,
                                    Precision P = Precision::F32);

// 批量：对每个 batch b 计算 C_b = alpha * op(A_b) × op(B_b)，结果垂直堆叠
[[nodiscard]] Result<Tensor> batched_matmul(const Tensor& A, const Tensor& B,
                                            std::size_t batch,
                                            bool transA = false, bool transB = false,
                                            Scalar alpha = Scalar{1},
                                            Precision P = Precision::F32);

// out = A × B + bias（bias (out,1) 广播到 (out,batch)）
[[nodiscard]] Result<Tensor> matmul_with_bias(const Tensor& A, const Tensor& B,
                                              const Tensor& bias,
                                              bool transA = false, bool transB = false,
                                              Precision P = Precision::F32);

// dst += src（逐元素、同形状；dst 存储精度不可变）
[[nodiscard]] Result<void> accumulate(Tensor& dst, const Tensor& src);
[[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B);   // A += B
[[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s);        // A *= s
[[nodiscard]] Result<void> zero(Tensor& A);                           // A = 0

// A (R,C) → (C,R)
[[nodiscard]] Result<Tensor> transpose(const Tensor& A);
```

**形状要求（`matmul`）**：`A: (M,K)`、`B: (K,N)`、输出 `(M,N)`；`transA` 时 `A` 按 `(K,M)` 存储，`transB` 时 `B` 按 `(N,K)` 存储。

**`batched_matmul` 形状**：
- `A: (batch * A_rows_per_batch, A_cols)`，`B: (batch * B_rows_per_batch, B_cols)`
- 输出 `(batch * M, N)`
- `alpha` 是输出缩放（cuBLAS sgemm 语义），**GPU 在 shader 写出时一次完成** → 供上层折叠 `1/√d_k` 等系数，省一次全矩阵 scale pass

**`accumulate` vs `add_inplace`**：语义相同（`dst += src`）。`accumulate` 额外支持 `NN_F16_DEBUG=1` 的数值诊断。日常用 `add_inplace`。

```cpp
// 多头注意力：Q, K, V: (H*d_k, batch*seq)
auto attn = engine.batched_matmul(*q, *k, /*batch=*/batch * heads);

// 带 alpha 折叠缩放
auto scaled = engine.batched_matmul(*q, *k, batch * heads, false, false,
                                    1.0f / std::sqrt(static_cast<float>(d_k)), p_.compute);

// Linear 层（融合写法，推荐）
auto out = nn::dsl::compute(engine,
    nn::dsl::matmul(*weight_, input) + nn::dsl::row_broadcast(*bias_),
    weight_->rows(), input.cols(), p_.compute);
```

> **选型**：单层前向请优先用 `dsl::matmul` + 尾链（`matmul_with_bias` 与 DSL 融合二选一，语义等价）。`engine.matmul_with_bias` 用于"只需要 matmul + bias、不需要别的尾链"的场合。**`matmul` + bias 不要写成两步**（先 `matmul` 再 `add_inplace`），那会多一次全尺寸 dispatch。

### 9.1 高效融合写法（多步逐元素变换）

```cpp
// ❌ 不推荐：三次调用 = 三个 kernel、两个全尺寸中间量
auto t1 = nn::dsl::compute(eng, nn::dsl::exp(nn::dsl::leaf(x)), rows, cols);
auto t2 = nn::dsl::compute(eng, nn::dsl::leaf(*t1) + nn::dsl::leaf(y), rows, cols);
auto r  = nn::dsl::compute(eng, nn::dsl::leaf(*t2) * nn::dsl::leaf(z), rows, cols);

// ✅ 推荐：一条表达式 = 一个融合 kernel
auto r = nn::dsl::compute(eng,
    (nn::dsl::exp(nn::dsl::leaf(x)) + nn::dsl::leaf(y)) * nn::dsl::leaf(z), rows, cols);
```

---

## 10. 归约原语

```cpp
// 按行求和：A (rows, cols) → (rows, 1)；out[r] = Σ_c A[r][c]
[[nodiscard]] Result<Tensor> row_reduce_sum(const Tensor& A, Precision P = Precision::F32);

// 按列求和：A (rows, cols) → (1, cols)；out[c] = Σ_r A[r][c]
[[nodiscard]] Result<Tensor> col_reduce_sum(const Tensor& A, Precision P = Precision::F32);

// 按列求最大值：A (rows, cols) → (1, cols)；out[c] = max_r A[r][c]
[[nodiscard]] Result<Tensor> col_reduce_max(const Tensor& A, Precision P = Precision::F32);

// 分组归约：x (G*R, N) → (G, N)
[[nodiscard]] Result<Tensor> grouped_reduce_sum(const Tensor& x, std::size_t G, std::size_t R,
                                                Precision P = Precision::F32);
[[nodiscard]] Result<Tensor> grouped_reduce_max(const Tensor& x, std::size_t G, std::size_t R,
                                                Precision P = Precision::F32);
// grouped_reduce_sum: out[g][n] = Σ_{i<R} x[g*R + i][n]
// grouped_reduce_max: out[g][n] = max_i  x[g*R + i][n]
```

- **没有 `row_reduce_max` 引擎算子**——按行求最大值用 DSL：`dsl::compute_reduce(eng, dsl::row_reduce_max(dsl::leaf(*t)), rows, cols)`。同理没有 `grouped_reduce_min` / `col_reduce_min`。
- **分组归约的用途**：把"逐通道/逐头一次 dispatch"的层内循环压成**单次**原语调用（池化窗口归约、多头分组统计、分组归一化）。`R` 是语义参数，`G = rows / R`。
- **精度**：累加恒 f32 + 输出舍入到 `P`。原生 f16 归约（GPU `reduce.comp` 的 f16 变体：输入 f16 直读、f32 归约、输出 f32 向量）能力由 `supports_native_f16_reduce()` 报告；无能力时基类自动边界 cast。

```cpp
// LayerNorm 的 (1,B) 统计量
auto var_raw = nn::dsl::compute_reduce(eng,
    nn::dsl::col_reduce_sum(nn::dsl::leaf(*diff) * nn::dsl::leaf(*diff)), F, B, p_.stable);
auto std_inv = nn::dsl::compute(eng,
    nn::dsl::rsqrt(nn::dsl::leaf(*var_raw) * nn::dsl::rparam(1.0f / F) + nn::dsl::rparam(eps)),
    1, B, p_.stable);

// grad_bias：把 (F, B) 梯度归约成 (F, 1)
auto gb = engine.row_reduce_sum(grad_out, p_.compute);
```

---

## 11. 数据搬运原语

全部**无 `P`**：**输出精度 = 源精度**（in-place 类保留 `dst` 精度）。

```cpp
// 深拷贝（无 PCIe 传输）
[[nodiscard]] Result<Tensor> clone(const Tensor& src);

// 行切片：src 的行 [start_row, start_row + count) 的连续拷贝 → (count, cols)
[[nodiscard]] Result<Tensor> slice_rows(const Tensor& src, std::size_t start_row,
                                       std::size_t count);

// 行插入：src 的所有行写入 dst 的行 [dst_start_row, ...)
[[nodiscard]] Result<void> insert_rows(Tensor& dst, std::size_t dst_start_row, const Tensor& src);

// 行 gather（Embedding 查表）：table (vocab, D)、indices (num,)
[[nodiscard]] Result<Tensor> gather_rows(const Tensor& table, const Tensor& indices);
//   out (num, D)，out[i] = table[indices[i]]；越界索引 → 零行（防御性，不报错）

// 行 scatter-add（Embedding 梯度）：dst[indices[i]] += grad[i]
[[nodiscard]] Result<void> scatter_add_rows(Tensor& dst, const Tensor& indices,
                                           const Tensor& grad);
//   dst (vocab, D) 原地修改；重复 indices 会被多次累加

// 3D 维度转置：(M, B, N) ↔ (B, M, N)
[[nodiscard]] Result<Tensor> rearrange_3d(const Tensor& x, std::size_t M, std::size_t B,
                                         std::size_t N, bool inverse = false);

// 卷积/池化窗口展开与伴随散射
[[nodiscard]] Result<Tensor> im2col(const Tensor& x,
                                    std::size_t C, std::size_t H, std::size_t W,
                                    std::size_t k, std::size_t stride, std::size_t pad,
                                    std::size_t OH, std::size_t OW);
[[nodiscard]] Result<Tensor> col2im(const Tensor& col,
                                    std::size_t C, std::size_t H, std::size_t W,
                                    std::size_t k, std::size_t stride, std::size_t pad,
                                    std::size_t OH, std::size_t OW);
```

**`rearrange_3d` 语义**：

```
inverse=false: 输入 (M, B*N) → 输出 (B*M, N)
               out[b*M + m, n] = in[m, b*N + n]
inverse=true : 输入 (B*M, N) → 输出 (M, B*N)
               out[m, b*N + n] = in[b*M + m, n]
```

典型用途：MHA 批量化时把 `(H*d_k, batch*seq)` 重排为 `(batch*H*d_k, seq)`，使 `batched_matmul` 能按 `batch*H` 切分行块。

**`im2col` / `col2im` 语义**：

```
x:   (C*H*W, B)
out: (C*k*k, B*OH*OW)
out[(ci*k + kh)*k + kw, (oh*OW + ow)*B + b]
    = x[ci*H*W + (oh*stride + kh - pad)*W + (ow*stride + kw - pad), b]
```

- 越界（padding 区）取 **0**。`OH/OW` 由调用方按 `(H + 2*pad - k)/stride + 1` 给出。
- **列序为「位置优先」`(oh, ow, b)`** —— 这样 Layer 只需一次 `rearrange_3d` + 一次 `gather_rows` 即可完成 `(C, B*P) → (C*P, B)` 的 samples 布局转换。
- `col2im` 是 `im2col` 的**伴随**（反向散射）：`out[ci*H*W + ih*W + iw, b] = Σ_{kh,kw} col[...]`，其中 `oh = (ih + pad - kh)/stride` 须整除且落在 `[0, OH)`。**重叠窗口（`stride < k`）的贡献在此累加**；每个输出元素由单线程完整求和，**无需原子操作/预清零**。
- 用途：Conv2D 把卷积变成 matmul（`W × col`）；MaxPool2D 展开窗口做列归约（取 `k=pool, stride=stride, pad=0`）。

```cpp
// Embedding 前向
auto emb = engine.gather_rows(*weight, *token_ids);
// Embedding 反向
if (auto r = engine.scatter_add_rows(*grad_weight, *token_ids, *grad_emb); !r) ...

// MHA 布局转换
auto q_r  = engine.rearrange_3d(*q, heads * d_k, batch, seq);           // → (batch*H*d_k, seq)
auto back = engine.rearrange_3d(*ctx, heads * d_k, batch, seq, true);   // inverse
```

---

## 12. 扫描原语（RLA/RAPT）

引擎只提供"前缀/后缀顺序归约 + matvec 读出"；**RLA 算法（L2 归一化分母 / ReLU 门控 / 梯度公式 / 文档重置策略）全部由 Layer 组合表达**。

**形状约定（batch-major，列序 `i = b*seq + t`；头 `(b,h)` 的行块起点 `r0 = (b*H + h)*d_k`）**：

| 张量 | 形状 |
|---|---|
| `K / V / P / R (X/Y)` | `(B·H·d_k, seq)` |
| `D` | `(B·H·d_k², seq)`，`(b,h)` 的 `(a,b')` 元素在行 `(b*H*d_k + a)*d_k + b'` |
| `A0 / B0` | `(H·d_k, d_k)` 初始运行态，行块 `h` = 第 `h` 头（`B>1` 时按头循环）；`has_state=false` → 按零处理（传 `(1,1)` dummy 规避 0 字节 buffer） |
| `boundary` | `(1, B·seq)`，`1` = 文档起点（`t==0` 或与前一位置文档不同）；`has_bnd=false` → 无文档感知（传 `(1,1)` dummy） |
| 标量块 `s/r` | 每 `(b,h,t)` 一个标量，在头块内 `d_k` 行**重复存放**（避免块级广播原语）；实现写全部 `d_k` 行同一值，Layer 读任一行均可 |

```cpp
// 前缀扫描
[[nodiscard]] Result<Tensor> scan_prefix_outer(
    const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
    const Tensor& A0, const Tensor& B0, bool has_state,
    std::size_t dk, std::size_t heads, bool causal,
    const Tensor& boundary, bool has_bnd,
    Precision prec = Precision::F32);

// 后缀扫描（RLA 反向 pass 2）
[[nodiscard]] Result<Tensor> scan_suffix_outer(
    const Tensor& D, const Tensor& X, const Tensor& Y,
    std::size_t dk, std::size_t heads, bool causal,
    const Tensor& boundary, bool has_bnd,
    Precision prec = Precision::F32);

// 逐列外积（RLA 反向的 dL/dA、dL/dB 物化）
[[nodiscard]] Result<Tensor> outer_col(
    const Tensor& P, const Tensor& R, const Tensor& S,
    std::size_t dk, bool has_scale,
    Precision prec = Precision::F32);
```

**`scan_prefix_outer` 语义**：

- `causal=true`：含自身前缀（`i <= t`）：`A_t = A0 + Σ_{i≤t, 同文档} k_i·k_iᵀ`，`B_t = B0 + Σ_{i≤t, 同文档} v_i·k_iᵀ`；**文档边界处运行态清零**（`A0/B0` 仅首个文档生效）。
- `causal=false`：全集常数 `A = A0 + Σ_all k·kᵀ`，`B = B0 + Σ_all v·kᵀ`（无边界重置）。
- 输出 `(B·H·5·d_k, seq)`，行块（每块 `(B·H·d_k, seq)`）：

  ```
  [0) B·P    [1) A·P    [2) Bᵀ·R    [3) s = P·(A·P)    [4) r = R·(B·P)
  ```

  其中 `[3)/[4)` 为逐列标量（头内逐行重复）。

**`scan_suffix_outer` 语义**：

- `causal=true`：`S_i = Σ_{t≥i, 同文档} D_t`（`i+1` 为文档起点时先清零再累加 `D_i`）。
- `causal=false`：`S_i = D_i`（Layer 预先把全集梯度沿 `seq` 广播）。
- `D (B·H·d_k², seq)`，`X/Y (B·H·d_k, seq)`。
- 输出 `(B·H·3·d_k, seq)`，行块：`[0) S·X   [1) S·Y   [2) Sᵀ·Y`。

**`outer_col` 语义**：`out[(b,h): (a,b'), t] = P[a,t]·R[b',t]`（`has_scale=true` 时再乘 `S[t]`）。

- `P/R: (B·H·d_k, seq)`；`S: (B·H·d_k, seq)`（标量头内逐行重复，实现读头块首行；`has_scale=false` → 传 dummy）。
- 输出 `(B·H·d_k², seq)`。

**原生 f16 扫描（GPU）**：f16 输入直接下传，**不物化全尺寸 f32 输入副本**（scan 输出是 RLA 训练的大头临时量）。CPU 能力为 false → 边界 cast。输出按 `prec` 落回。

> **接口约定泄漏提示**：状态块偏移（`0 / BHdk / 2*BHdk / 4*BHdk`）是引擎契约的一部分，Layer 必须按上表手工切块——`compute_layer_rapt.hpp:535-785` 的复杂度很大一部分来自这里。

算法细节见 `docs/development/06-rapt-algorithm.md`。

---

## 13. 表达式直接求值（高级）

当你已经持有 `ExprSpec`（例如从 Layer 显式登记的 fold 构造）时，直接走引擎：

```cpp
[[nodiscard]] Result<Tensor> eval_expr(const ExprSpec& spec,
                                       std::span<const Tensor> inputs,
                                       std::size_t rows, std::size_t cols,
                                       Precision P = Precision::F32);
// 输出 (rows, cols)；输出 = 最后一条指令的目标寄存器

[[nodiscard]] Result<Tensor> eval_expr_reduce(const ExprSpec& spec,
                                              std::span<const Tensor> inputs,
                                              std::size_t rows, std::size_t cols,
                                              Precision P = Precision::F32);
// 输出为归约向量本身：行归约轴 → (rows,1)；列归约轴 → (1,cols)
// 要求表达式归约轴为 0/1

[[nodiscard]] Result<void> eval_expr_into(const ExprSpec& spec,
                                          std::span<const Tensor> inputs,
                                          std::size_t rows, std::size_t cols,
                                          Tensor& out);
// 目标传递：直接写进 out（不分配新张量）
// out 允许与某个输入是同一 buffer（逐元素先读后写，就地安全）
// 仅支持逐元素表达式（无归约）；归约向量输出用 eval_expr_reduce
```

**共同前置条件**：

- 所有输入张量必须**绑定到同一引擎**（`bind_check_`）。
- `inputs.size()` 必须与 spec 记录的输入数一致。
- **闭合世界**：GPU 上该 `(结构 key, 精度签名)` 没有预生成 shader → **硬报错**，无 eager、无运行时编译。

**辅助工具**：

```cpp
namespace nn::dsl {
template <typename E>
[[nodiscard]] std::pair<ExprSpec, std::vector<Tensor>> to_expr_spec(const E& e);
}
[[nodiscard]] inline Result<void> validate_expr_spec(const ExprSpec& spec, std::size_t num_inputs);
[[nodiscard]] inline std::string    expr_spec_key(const ExprSpec& s);
[[nodiscard]] inline int            expr_spec_reduce_axis(const ExprSpec& s);
```

**典型场景（Layer 里的 fold 用法）**：

```cpp
// 注意力 forward 走单 fold kernel（S 矩阵不物化）
//   入参 = 掩码种类 × 位置偏置（正交两维），共 5 个实际使用的组合
auto attn = engine.eval_expr(
    nn::expr::make_fold_attn_o(seq, dk, bh, nn::expr::AttnMaskKind::Causal,
                               /*score_bias=*/false),
    inputs, rows, cols, p_.stable);
```

> `nn::expr::make_fold_attn_o` 等 fold 构造定义在 `compute_layer_attention.hpp`（AOT 原则："表达式文本只出现在 Layer"）。入参是**掩码 `AttnMaskKind{Plain, Causal, CausalDoc}` × 位置偏置 `bool score_bias`（ALiBi）**两个正交维度，`scan_exprs` 里必须把 5 个实际组合（3 掩码 × 2 偏置 − 1，"Plain + 偏置"不存在）**全部登记**，漏登记 → GPU 闭合世界硬报错。ALiBi 的偏置在 **backward** 里是掩码之后的独立一步（`PositionEncoder::apply_score_bias`），forward 则融在同一个 fold kernel 内。

**普通业务代码不应该直接调这三个入口**——用 `dsl::compute*`。它们是给"Layer 显式登记 fold 构造"和"引擎自测"用的。

---

## 14. 批处理与内存控制

```cpp
[[nodiscard]] virtual Result<void> begin_batch() = 0;   // CPU no-op / GPU 开始录制
[[nodiscard]] virtual Result<void> end_batch() = 0;     // CPU no-op / GPU 提交 + fence wait
[[nodiscard]] virtual Result<void> flush_batch();       // 默认 {}；GPU 提交+等待并自动开新帧
[[nodiscard]] virtual Result<void> release_idle_pool_blocks();  // 默认 {}；GPU 归还空闲池底材
[[nodiscard]] virtual std::string pool_stats() const;   // 见 §7.2
```

**推荐用法**：

```cpp
// 小步数：一次录制覆盖整个 step
engine.begin_batch();
// ... forward / loss / backward ...
engine.end_batch();

// 大模型：前向与反向之间刷新，防 Windows TDR 超时
engine.begin_batch();
// ... forward ...
engine.flush_batch();
// ... backward ...
engine.end_batch();

// 大 batch 评估：分块 + 每块归还空闲显存
for (auto& chunk : chunks)
{
    // ... forward chunk ...
    if (auto r = engine.release_idle_pool_blocks(); !r) return std::unexpected(r.error());
}
```

**录制期铁律（#6）**：`begin_batch()` → `end_batch()` 之间引用的所有张量必须存活到 `end_batch()` 之后。

---

## 15. 异步标量回读

**动机**：用 `to_matrix` 取每步 loss，GPU 引擎会 `end_batch` + `wait_in_flight`（等全部在飞帧）→ 每 step 一次全流水线 drain，host/GPU 无法重叠。

```cpp
// 把 t 的头 4 字节排入一次 D2H 拷贝并提交，不等待（源统一抬到 f32）
[[nodiscard]] Result<void> submit_scalar_readback(std::size_t slot, const Tensor& t);

// 非阻塞查询：就绪写值并返回 true；未就绪返回 false（调用方稍后重试）
[[nodiscard]] virtual Result<bool> poll_scalar_readback(std::size_t slot, Scalar& out);

// 可用槽位数（调用方据此做环形复用）
[[nodiscard]] virtual std::size_t scalar_readback_slots() const;   // 默认 1
```

**调用约定（必须遵守）**：

1. **必须在产出 `t` 的主帧已提交之后调用**——同一队列 FIFO 保证拷贝执行在生产者之后。**在主帧提交前提交会读到上一轮旧值**。
2. `t` 的宿主 Tensor 必须**存活到 `poll` 返回就绪**（buffer 生命周期跨越提交）。
3. 用 `scalar_readback_slots()` 做环形 slot 复用，避免同 slot 重排。

```cpp
// 训练循环里的 loss 回读（非阻塞）
const std::size_t slots = engine.scalar_readback_slots();
engine.submit_scalar_readback(step % slots, *loss_tensor);
engine.end_batch();
// ... 后续计算 ...
Scalar loss_val = 0.0f;
if (auto ready = engine.poll_scalar_readback(step % slots, loss_val); ready && *ready)
    std::printf("step %zu loss=%f\n", step, loss_val);
```

CPU 引擎的默认实现：`submit` 立刻取值、`poll` 恒立刻就绪。

---

## 16. L2 宿主桥

**铁律 #12**：**L2+（Layer / Loss / Optimizer / Model）头文件里不得出现 `Matrix` 类型，也不得调用 `from_matrix` / `to_matrix` / `copy_from`。**

层自算的辅助数据（没有引擎侧生成原语的那类：索引 / 位置编码表 / 掩码 / 斜率 / 编码表 …）走**宿主桥**：

```cpp
namespace nn::detail {

// 标量缓冲 → 新建 (rows, cols, P) 张量
// f16 目标按 round-half-to-even 舍入，与 from_matrix(m, P) 逐位同口径
[[nodiscard]] inline Result<Tensor> upload_span(ComputeEngine& engine,
                                                std::size_t rows, std::size_t cols,
                                                Precision P,
                                                std::span<const Scalar> src);

// 张量 → 既有标量缓冲；f16 存储先升 f32，与 to_matrix(t, F32) 同值
[[nodiscard]] inline Result<void> download_span(ComputeEngine& engine,
                                                const Tensor& t, std::span<Scalar> dst);

// download_span 的取值便捷形态
[[nodiscard]] inline Result<std::vector<Scalar>> download_vector(ComputeEngine& engine,
                                                                 const Tensor& t);
}
```

- 内部实现 = 批量 `write` / `read`（GPU 自动 staging、隐含 flush + 同步）。**不新增任何引擎原语，不进引擎虚表**（49 个虚函数骨架不变）。
- **适用范围仅限"层自算的小规模辅助数据"**。数据集、预训练权重、对拍/落盘等大批量 I/O 仍走 I/O 层的 `from_matrix` / `to_matrix`。
- 零张量直接 `engine.create_tensor(r, c, P, InitSpec::zero())`，不需要宿主桥。

```cpp
// Layer 内构造正弦位置编码
std::vector<float> enc(seq * d_model);
// ... 填充 enc ...
auto pos = nn::detail::upload_span(engine, seq, d_model, p_.param, std::span(enc));
if (!pos) return std::unexpected(pos.error());

// 读取一个统计量做诊断
auto v = nn::detail::download_vector(engine, *t);
if (!v) return std::unexpected(v.error());
```

> 分层审计：`pwsh -File bench\doc_inventory.ps1` 第 `[4]` 节，验收 = `L2-VIOLATIONS: 0`（宿主桥用量只披露、不判违规）。

---

## 17. 引擎能力查询

这三个虚函数是"引擎告诉基类它有什么原生路径"的钩子。**写 Layer 时你通常不直接调它们**，但必须理解它们决定了 f16 的性能与临时量规模。

```cpp
// GPU = true / CPU = false
[[nodiscard]] virtual bool supports_native_data_move() const noexcept;

// GPU = true / CPU = false
[[nodiscard]] virtual bool supports_native_f16_reduce() const noexcept;

// 该 (结构, 输入精度, 目标输出精度) 是否有预生成的【带类型】融合 shader
[[nodiscard]] virtual bool supports_expr_precision_variant(
    const ExprSpec& spec, std::span<const Tensor> inputs,
    Precision P = Precision::F32) const;
```

| 钩子 | 返回 true 时基类做什么 |
|---|---|
| `supports_native_data_move` | 直接放行 f16 张量给数据搬运/原地算术（省掉"抬 f32 → 算 → 落回 f16"的全尺寸临时量与 3 倍流量） |
| `supports_native_f16_reduce` | f16 输入直接下传归约（输入 f16 直读、f32 归约、输出 f32 向量），不物化整份 f32 输入副本 |
| `supports_expr_precision_variant` | 把 f16 原张量直接交给带类型融合 shader（读 f16 / 写 f16，算术 f32），零边界临时量 |

**能力为 false 时基类的兜底**（这就是"边界 cast 层"）：入 = 把 f16 操作数抬到 f32（已是 f32 则零拷贝直通）；算 = 调用既有 f32 实现；出 = 结果按目标精度落回（RHE）。**正确性不变，只多花带宽。**

**全 f32 配置下这一层是纯直通**（每入口一条快速判定分支），行为与直接调用引擎实现**逐字节一致**。

---

## 18. 怎样用引擎把 Layer 写简单

把"引擎能力"翻译成"Layer 的写法纪律"。

### 18.1 每个 Layer 方法的标准骨架

```cpp
class Linear : public nn::Layer
{
    nn::Tensor weight_, bias_;
    nn::Tensor input_cache_, grad_weight_, grad_bias_;

public:
    [[nodiscard]] nn::Result<void> init_impl(nn::ComputeEngine& engine) override
    {
        // 声明式初始化：层算分布参数，引擎填数
        const float limit = std::sqrt(6.0f / static_cast<float>(in_ + out_));
        weight_ = engine.create_tensor(out_, in_, p_.param,
                                       nn::InitSpec::uniform(-limit, limit, nn::kInitSeed));
        bias_   = engine.create_tensor(out_, 1, p_.param, nn::InitSpec::zero());
        if (!weight_.valid() || !bias_.valid())
            return std::unexpected(nn::Error{"Linear: 参数分配失败"});

        // ★ 新不变量：复合层必须 init 全部子层
        if (auto r = sub_layer_.init(engine); !r) return r;
        return {};
    }

    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& input) override
    {
        nn::ComputeEngine& engine = engine_ref();          // ★ 引擎来自 init 绑定
        input_cache_ = input;                              // 持有句柄（零拷贝）
        return nn::dsl::compute(engine,
            nn::dsl::matmul(weight_, input) + nn::dsl::row_broadcast(bias_),
            weight_.rows(), input.cols(), p_.compute);     // ★ 一条表达式 = 一个 kernel
    }

    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& grad_output) override
    {
        nn::ComputeEngine& engine = engine_ref();

        // grad_w += grad_output · xᵀ  ← 原地累加，零分配
        if (auto r = nn::dsl::compute_into(engine,
                nn::dsl::leaf(grad_weight_) +
                    nn::dsl::matmul(grad_output, input_cache_, false, true),
                grad_weight_); !r)
            return std::unexpected(r.error());

        // grad_x = Wᵀ · grad_output
        return nn::dsl::compute(engine,
            nn::dsl::matmul(weight_, grad_output, true, false),
            weight_.cols(), grad_output.cols(), p_.compute);
    }
};
```

> 注意 `Layer::init` 是 **NVI**：公共入口负责把 `engine_` 绑定到层上，层只 override `init_impl`。`forward`/`backward`/`zero_grad`/`forward_recompute` **不再收 `engine` 形参**（M6 段 C）。手工构造的层必须先 `layer.init(engine)`（`Model::add` 会自动调）；未绑定即 fail-fast 并打印调用点 `file:line`。

### 18.2 Layer 可以直调引擎的 4 类（其余都应经 DSL）

| 类别 | 允许的动词 | 例子 |
|---|---|---|
| 基础设施 | `create_tensor` / `zero` / `clone` / `reshape` / `cast` / `read` / `write` / `begin_batch` / `end_batch` / `flush_batch` / `create_offload_buffer` 系列 | `zero_grad()` 里 `engine.zero(grad)` |
| 数据搬运 | `slice_rows` / `insert_rows` / `gather_rows` / `scatter_add_rows` / `rearrange_3d` / `im2col` / `col2im` / `transpose` | Embedding 前向 `gather_rows`、MHA `rearrange_3d` |
| 状态扫描 | `scan_prefix_outer` / `scan_suffix_outer` / `outer_col` | RAPT 前向/反向 |
| fold 登记 | `eval_expr`（配合 `make_fold_*`） | 注意力 forward 单 fold kernel |

> **`from_matrix` / `to_matrix` / `copy_from` 不在上表**（铁律 #12 禁用）。层内辅助数据用 `detail::upload_span` / `download_span`（§16）。
>
> 当前实测分布：Layer 共 211 次引擎直调、21 个 distinct 动词。`rearrange_3d`(50) + `transpose`(33) + `slice_rows`(22) + `create_tensor`(41) 占了其中大部分。

### 18.3 写 Layer 的 8 条纪律（每条都指向"更简洁"）

1. **一条表达式 = 一个计算单元**。多步逐元素变换不要拆成多次 `dsl::compute`；写成一个表达式。
2. **原地更新一律 `compute_into`**，不分配中间张量，也不用 `add_inplace` 串多步。
3. **广播只走视图**：`row_broadcast` / `col_broadcast`；不要手工 `clone` + 循环。
4. **归约输出小向量用 `compute_reduce`**（`(rows,1)` / `(1,cols)`），不要广播成全尺寸再处理。
5. **层内循环能用原语压掉就压掉**：逐通道/逐头循环 → `grouped_reduce_*` / `batched_matmul` / `rearrange_3d`。
6. **精度靠 `P` 表达**，不要手写 `cast` 去"适配" f16。
7. **辅助数据走宿主桥或 `InitSpec`**，不要在层里拼 `Matrix`；`create_tensor` + `zero` 两步请合并成 `InitSpec::zero()`。
8. **复合层 `init_impl` 必须 init 全部子层**（含 `ReLU` / `GeLU` / `SwiGLU` / `Softmax` 这类无参量子层）——漏了会在 forward 时 fail-fast 并打印调用点 `file:line`。

### 18.4 反面模式（会显著增加 Layer 复杂度与体积）

| 反面模式 | 为什么坏 | 正确写法 |
|---|---|---|
| 逐元素拆成多次 `dsl::compute` | N 个 kernel、N-1 个全尺寸中间量 | 写成一条表达式 |
| 手工 `clone` 造临时张量再 `add_inplace` | 多余分配 + 多余 dispatch | `compute_into` 原地 |
| 在 Layer 里出现 `Matrix` / `to_matrix` | 违反铁律 #12；GPU 上引入 PCIe 往返 | `detail::upload_span` / `download_span` |
| 手写 `cast` 适配 f16 | 全尺寸临时量，且与边界 cast 层重复 | 传对 `P` |
| 循环 per-head / per-channel 调 dispatch | dispatch 次数随通道数线性增长 | `batched_matmul` / `grouped_reduce_*` |
| 运行时标量写成 `ConstLeaf` | 值进 key → GPU 闭合世界查不到 shader | `dsl::rparam` |
| 视图 / 叶子单独作表达式根 | 空指令表 → 构建期硬失败 | 尾接 `+ dsl::rparam(0.0f)` |
| `create_tensor` + `zero` 两步 | 多一次全量写与一次 dispatch | `InitSpec::zero()` 一步 |
| 每调用点手写 `p_.stable` / `p_.compute` | 同一层内可能混用，语义漂移 | 由层类型决定，集中在少数调用点 |

---

## 19. 方法速查总表

### 19.1 张量与 I/O

| 方法 | 签名要点 | 输入 → 输出 | 精度 | 失败 |
|---|---|---|---|---|
| `create_tensor` | `(rows, cols, P=F32)` | — → `(rows,cols)` | `P` | 返回无效 `Tensor`（查 `valid()`） |
| `create_tensor` | `(rows, cols, P, const InitSpec&)` | — → `(rows,cols)` | `P` | 返回无效 `Tensor` |
| `from_matrix` | `(const Matrix&, P=F32)` | 宿主 `Matrix` → `(m.rows, m.cols)` | `P`（f32→f16 RHE） | `Result` |
| `to_matrix` | `(const Tensor&, P=F32)` | 张量 → **f32 Matrix** | `P` 为"要求下载精度" | `Result` |
| `import` | `(const Tensor&[, P])` | 跨设备/引擎拉取 | 同精度别名 / 异精度 cast | `Result`；**不改写 src** |
| `read<T>` | `(const Tensor&, std::span<T>)` | 张量 → span | `T` 须精确匹配 | `Result<void>`（错配/长度） |
| `write<T>` | `(Tensor&, std::span<T>)` | span → 张量既有存储 | `T` 须精确匹配 | `Result<void>` |
| `get_index` | `(const Tensor&, row, col)` | → `Scalar` | f16 精确提升 | `Result<Scalar>`（越界） |
| `set_index` | `(Tensor&, row, col, Scalar)` | → 既有存储 | f16 RHE | `Result<void>` |
| `reshape` | `(const Tensor&, new_rows, new_cols)` | — | 保持 | `Result`（元素数不匹配） |
| `clone` | `(const Tensor&)` | — | = 源 | `Result` |
| `copy_from` | `(Tensor&, const Matrix&)` | 宿主 f32 → 已有张量 | f16 目标经 `cast_into` | `Result<void>` |
| `cast` | `(const Tensor&, Precision)` | — | 变精度（RHE） | `Result` |
| `cast_into` | `(const Tensor&, Tensor&)` | 写 dst 既有存储 | 变精度 | `Result<void>` |
| `copy_into` | `(Tensor&, const Tensor&)` | 同精度同形拷贝 | 保持 | `Result<void>` |

### 19.2 矩阵 / 归约 / 数据搬运 / 扫描 / 表达式

| 方法 | 输入形状 → 输出形状 | `P` |
|---|---|---|
| `matmul(A,B,tA,tB,P)` | `(M,K)×(K,N)` → `(M,N)`（转置时按 `(K,M)`/`(N,K)` 存储） | 有 |
| `batched_matmul(A,B,batch,tA,tB,alpha,P)` | `(b*Mr,K)×(b*Kr,N)` → `(b*M,N)` | 有 |
| `matmul_with_bias(A,B,bias,tA,tB,P)` | `(M,K)×(K,N)+(M,1)` → `(M,N)` | 有 |
| `accumulate` / `add_inplace` | `(rows,cols)` 同形，`dst += src` | **无**（in-place） |
| `scale_inplace(A,s)` | `A *= s` | **无** |
| `zero(A)` | `A = 0` | **无** |
| `transpose(A)` | `(R,C)` → `(C,R)` | **无**（= 源） |
| `row_reduce_sum(A,P)` | `(rows,cols)` → `(rows,1)` | 有 |
| `col_reduce_sum(A,P)` | `(rows,cols)` → `(1,cols)` | 有 |
| `col_reduce_max(A,P)` | `(rows,cols)` → `(1,cols)` | 有 |
| `grouped_reduce_sum/max(x,G,R,P)` | `(G*R,N)` → `(G,N)` | 有 |
| `clone(src)` | — → 同形 | **无** |
| `slice_rows(src,start,count)` | `(rows,cols)` → `(count,cols)` | **无** |
| `insert_rows(dst,start,src)` | `src(rows,cols)` → `dst[start.., ]` | **无**（dst 精度） |
| `gather_rows(table,indices)` | `(V,D)×(num,)` → `(num,D)` | **无** |
| `scatter_add_rows(dst,indices,grad)` | `(V,D) += (num,D)` by `indices` | **无**（dst 精度） |
| `rearrange_3d(x,M,B,N,inverse)` | `(M,B*N)` ↔ `(B*M,N)` | **无** |
| `im2col(x,C,H,W,k,stride,pad,OH,OW)` | `(C*H*W,B)` → `(C*k*k, B*OH*OW)` | **无** |
| `col2im(col,C,H,W,k,stride,pad,OH,OW)` | `(C*k*k,B*OH*OW)` → `(C*H*W,B)` | **无** |
| `scan_prefix_outer(...,prec)` | 见 §12 → `(B·H·5·d_k, seq)` | `prec` |
| `scan_suffix_outer(...,prec)` | 见 §12 → `(B·H·3·d_k, seq)` | `prec` |
| `outer_col(P,R,S,dk,has_scale,prec)` | → `(B·H·d_k², seq)` | `prec` |
| `eval_expr(spec,inputs,rows,cols,P)` | → `(rows,cols)` | 有 |
| `eval_expr_reduce(spec,inputs,rows,cols,P)` | → `(rows,1)` / `(1,cols)` | 有 |
| `eval_expr_into(spec,inputs,rows,cols,out)` | → `out` 既有存储 | **无** |

### 19.3 控制 / 诊断 / 能力

| 方法 | 说明 |
|---|---|
| `device()` | `Device::CPU` / `Device::GPU` |
| `begin_batch()` / `end_batch()` / `flush_batch()` | CPU no-op；GPU 录制 / 提交 / 中点刷新 |
| `release_idle_pool_blocks()` | GPU 归还空闲池底材；CPU 语义成立 no-op |
| `pool_stats()` | 两引擎都非空（GPU 池 / CPU 直配账本） |
| `create_offload_buffer` / `offload_save` / `offload_restore` | 激活 offload（GPU 特性） |
| `submit_scalar_readback` / `poll_scalar_readback` / `scalar_readback_slots` | 异步标量回读 |
| `supports_native_data_move` / `supports_native_f16_reduce` / `supports_expr_precision_variant` | 精度原生路径能力查询 |
| `temp_stats()` / `dump_temp_stats()` / `reset_temp_stats()` | 边界 cast 临时量归因（需 `NN_PREC_TRACE=1`） |
| `nn::detail::upload_span` / `download_span` / `download_vector` | L2 宿主桥 |

---

## 20. 诊断与环境变量

| 环境变量 | 作用 |
|---|---|
| `NN_BIND_DEBUG=1` | 未绑定输入进引擎也报错（抓库内 stamp 漏网）；错误同步打 stderr，带 `file:line` 与形状。**建议作为门禁运行 `NN_BIND_DEBUG=1 ctest`** |
| `NN_PREC_TRACE=1` | 打印每个 `(结构 key, 精度签名)`；`[prec][miss]` = "请求了非零精度签名却没命中带类型变体"（静默回退边界 cast 的唯一可见信号）；`[into] branch=…` 打印 `eval_expr_into` 的分支 |
| `NN_F16_DEBUG=1` | f16 中间量数值扫描（`nn_dbg_scan`）、`accumulate` 巨值打印、`compute_into` 预绑定失败原因 |
| `NN_MEM_STATS=1` | 训练中内存采样（配合 `pool_stats()`） |
| `NN_VULKAN_DEVICE` | 强制指定 Vulkan 计算设备（索引 `"2"` 或名称子串 `"NVIDIA"`）。优先级：显式 API > 本变量 > 自动打分 |

**临时量归因**：

```cpp
nn::ComputeEngine::reset_temp_stats();
// ... 跑一段 ...
if (nn::dsl::env_flag("NN_PREC_TRACE"))
    std::fprintf(stderr, "%s", nn::ComputeEngine::dump_temp_stats().c_str());
// 输出：按字节降序的 (rows, cols) × 次数 + 调用点行号，用来定位"哪个算子在 cast"
```

---

## 21. 故障速查表

| 症状 | 原因 | 修法 |
|---|---|---|
| `nn::GpuEngine engine;` 编译失败 | `GpuEngine` 必须绑定 `GpuBackend&` | `GpuBackend::instance().initialize(...)` 后 `GpuEngine engine(backend)` |
| `NN_ASSERT: tensor has no P-precision CPU storage` | 用 `Precision::F16` 取了一个 F32 张量的 `cpu_matrix<F16>()` | 检查 `precision()`；库外不要用存储访问器 |
| `NN_ASSERT: tensor has no P-precision GPU storage`（`[gpu_tensor-fail]`） | GPU 上对 f16 张量传了 `to_matrix(t, Precision::F16)` | 用默认 `P=F32` 或 `read` |
| `read/write: span 元素类型与张量精度不匹配` | `span<float>` 配了 f16 张量（或反之） | U2 精确匹配：f16 张量用 `std::span<nn::f16>` |
| `read/write: 元素数与张量形状不匹配` | span 长度 ≠ `rows*cols` | 修正长度 |
| `bind_check_ mixed engines` | 两个不同引擎创建的张量进了同一个算子 | 用 `import` 把张量拉到目标引擎 |
| `dsl::compute_into: dst not on CPU` | CPU 引擎下 `dst` 是 GPU 张量 | 检查 `dst.device()` |
| GPU 报闭合世界未命中 | 表达式结构或精度签名没有预生成 shader | 检查是否把运行时值写成了 `ConstLeaf`（应改 `rparam`）；改 Layer 表达式后需重新构建（`scan_exprs` → 生成阶段 自动跑） |
| `validate_expr_spec: empty instruction list` | 表达式根是裸视图/叶子 | 尾接 `+ dsl::rparam(0.0f)` |
| 构建期 `[scan] validate_expr_spec 失败` + `_Exit(3)` | 新增表达式非法（最常是空指令表 / 裸分组归约视图作根） | 同上；这是**构建期闸门**，不是运行期 bug |
| `VK_ERROR_DEVICE_LOST` | 录制期张量提前析构（铁律 #6），或 TDR | 确认张量存活到 `end_batch()`；TDR 不可重试，存 checkpoint 退出；`VK_TIMEOUT` 可减半 batch 重试 |
| GPU 对、CPU 错（或反之） | 表达式里叶子形状与输出网格不符（CPU 模板路径**不校验**） | 逐个核对叶子 / 广播视图形状 |
| 序列任务 loss 平台期 / 跨样本串扰 | 布局混用（position-major vs batch-major） | 统一 `i = b*seq + t`；测试必须覆盖 `batch>1` |
| 显存爆炸 | 逐元素拆多次调用、手写 cast、每通道一次 dispatch、大词表物化 one-hot | §18.4 反面模式表；大词表用 `CrossEntropyLoss::forward_sparse`（铁律 #9） |
| CPU 与 GPU 同精度结果不字节一致 | 归约内累加顺序差异 | 这是**设计内**行为：跨设备同精度为**容差内相等，不字节一致**；同设备内逐字节一致 |

---

## 22. 不存在 / 已废弃的 API

写代码前请对照本表。

| 不存在的写法 | 事实 |
|---|---|
| `nn::GpuEngine engine;`（无参） | `GpuEngine(GpuBackend&)` 是唯一构造 |
| `engine.ensure_gpu(...)` | 已改名 `import`（符号清零，命名即防线） |
| `engine.set_offload_enabled(...)` | 不存在；开关在 `GPTModel`/`RAPTModel` 的 `set_activation_offload(bool)`，底层 `ActivationOffloader` |
| `dsl::start_expr` / `end_expr` / `begin_expr` / `record_expr` | **跨表达式录制（IR-C）未采用**；融合粒度 = 单条表达式。无 `expr_graph.hpp` |
| lambda 形式提交流水线 | 不存在；表达式就是普通 C++ 表达式树，提交靠 3 个 `dsl::compute*` |
| `dsl::pow` / `sin` / `cos` / `erf` / `gelu` / `sigmoid` / `clamp` / `where` / `sqr` / `sum` / `step` | 不存在；用 §8.3 的组合表达，cos/sin 以张量 + `row_mod` 读入 |
| `dsl::matmul(A, B, alpha)` | **没有 `alpha` 形参**；缩放写尾链 `* dsl::rparam(alpha)` |
| `dsl::add_inplace` / `scale_inplace` / `accumulate` / `transpose` / `batched_matmul` / `matmul_with_bias` / `slice_rows` / `gather_rows` / `zero` | DSL 无这些入口；它们是引擎原语。DSL 的原地写法是 `compute_into` |
| `dsl::compute(engine, expr, rows)`（3 参） | 不存在；必须给 `rows, cols` |
| `dsl::compute_into(engine, expr, dst, P)` | 不存在第 4 参；输出精度恒 = `dst` 精度 |
| `dsl::row_reduce_min` / `col_reduce_min` | 不存在 |
| `nn::Tensor` 的 `cpu_matrix()` / `from_matrix()` / `cpu()` / `reshape()` | **已私有**（铁律 #11）；经引擎访问 |
| `Tensor::reshape` | 已私有；公共入口是 `engine.reshape(t, rows, cols)` |
| `from_matrix` / `to_matrix` / `copy_from` 在 Layer / Loss / Optimizer / Model 中 | **铁律 #12 禁用**；用 `detail::upload_span` / `download_span` / `engine.read` / `engine.write` |
| 用 `char*`/裸指针接管张量存储 | 铁律 #2：禁止 new/delete/裸指针所有权 |
| 在 `//` 注释里写反斜杠 | **铁律 #10**：GCC 行拼接会吞掉下一行 |

---

## 23. 相关文档

| 文档 | 何时读 |
|---|---|
| `docs/development/01-compute-engine-development.md` | 想**扩展引擎**（加新原语）而不是使用它 |
| `docs/development/12-compute-engine-inventory.md` | 引擎 49 个 virtual 方法的盘点与 Layer 直调分类（复现：`bench/doc_inventory.ps1`） |
| `docs/development/05-mixed-precision.md` | 多精度类型系统、f16 语义锚与实测结论（§12 已知限制） |
| `docs/development/02-operator-fusion.md` + `03-ir-optimization.md` | 表达式融合与 IR 细节 |
| `docs/development/17-unified-tensor-engine.md` | 统一张量/引擎总纲：访问不变量（铁律 #11）、Matrix 降级（铁律 #12）、InitSpec、批量读写、内存契约 |
| `docs/development/06-rapt-algorithm.md` | 扫描原语的算法背景 |
| `docs/development/08-pitfalls-and-lessons.md` | 踩坑警示录（改代码前读） |
| `docs/introduction/01-architecture.md` | 完整分层架构与数据流 |
| `docs/usage/01-quickstart-model.md` | 用现成 Layer 搭模型的教程 |
| `docs/usage/02-quickstart-train-infer.md` | 训练/推理 CLI + C++ API + GUI |

---

*本手册与当前代码签名一致。修改 `compute_engine.hpp` / `expr_dsl.hpp` / `compute_tensor.hpp` / `precision.hpp` 的公共面后，请同步修订 §19 速查总表与 §22 不存在清单。*
