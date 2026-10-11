# C++ API 手册（库使用者）

> **读者**：调用者（库使用者）——要把 neuralnet.cpp 当成一个 C++ 库来用、写自己的程序。
> 本分册定位见 [`README.md`](README.md)；只想跑现成 CLI / GUI 见 [`../usage/02-quickstart-train-infer.md`](../usage/02-quickstart-train-infer.md)。
>
> **目标**：让你**不读源码**就能把库的全部能力接进自己的程序。本文所有签名都取自
> `include/neuralnet.cpp/` 的实际声明（稳定入口），每节给出可粘贴的代码片段。
> 引擎原语、张量 I/O、表达式 DSL 的**逐条细节**在
> [`../usage/03-compute-engine-usage.md`](../usage/03-compute-engine-usage.md)（下称「引擎手册」），
> 本文只做速查与交叉引用，不重复。

---

## 0. 阅读路径：我要做什么 → 看哪节

| 我要做的事 | 看 |
|---|---|
| 把库嵌进我的 CMake 工程（add_subdirectory / FetchContent） | §2 + [`cmake-integration.md`](cmake-integration.md) + [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md) |
| 创建引擎、开 GPU | §3 |
| 创建 / 读写张量、上传下载数据 | §4（细节见引擎手册 §3–§6） |
| 打开 `--f16` / 三值权重那样的精度配置 | §5 |
| 用表达式 DSL 写逐元素 / 归约数学 | §6（细节见引擎手册 §8） |
| 搭 MLP / Transformer / CNN / GPT / RAPT 模型 | §7、§8 |
| 用现成模型工厂一条命令建模型 | §8 |
| 算 loss、建优化器、写训练循环 | §9、§10、§11 |
| 保存 / 加载 `.nnmodel` / `.nnckpt`、读架构 spec | §12 |
| 读 `.nndataset` 数据集 | §13 |
| 训练 / 使用分词器、编码解码 | §14 |
| 自回归生成文本 | §15 |
| 剖析性能、看环境变量开关 | §16 |
| 查错误处理宏、断言宏 | §17 |
| 写自己的 CLI / 用底层工具（文件、哈希、线程池） | §19 |
| 写一个**自己的层** | §18 + [`write-a-custom-layer.md`](write-a-custom-layer.md) |
| 往引擎里**加一个新原语** | [`01-compute-engine-development.md`](01-compute-engine-development.md) |
| 多精度的语义契约（写层必读） | [`05-mixed-precision.md`](05-mixed-precision.md) |

> **两条最容易踩的铁律**：① 一切张量的创建 / 读写 / reshape **必须经 `ComputeEngine`**
> （绕过 = 编译错误）；② 全程 **禁止 throw/try/catch**（`-fno-exceptions`），
> 错误一律走 `Result<T>`。

---

## 1. 起步：头文件、命名空间、错误处理

### 1.1 include 方式

```cpp
#include <neuralnet.cpp/nn.hpp>   // 唯一聚合入口：一次拿齐引擎/层/损失/优化器/序列化/分词器/数据集
```

也可按用途 include **薄转发头**（名字即用途，**不省编译时间**，只为可读）：

```cpp
#include <neuralnet.cpp/engine.hpp>     // ComputeEngine / CpuEngine / GpuEngine
#include <neuralnet.cpp/layer.hpp>      // Layer 基类 + 全部内置层 + 位置编码
#include <neuralnet.cpp/loss.hpp>       // Loss / MSELoss / CrossEntropyLoss
#include <neuralnet.cpp/optimizer.hpp>  // Optimizer + create_optimizer
#include <neuralnet.cpp/model.hpp>      // Model / ModelSpec / save_model / load_model
#include <neuralnet.cpp/train.hpp>      // 上面五者的合集（写训练循环够用）
```

模型工厂（`build_gpt_model` 等）与分词器在 `domain_*.hpp`（已随 `nn.hpp` 一起引入）。
一切公开符号都在 `namespace nn` 下；所有形状 / 维度都是 `std::size_t`。

### 1.2 `nn::Scalar` 与 `nn::f16`

```cpp
nn::Scalar = float;   // core_config.hpp 里的类型别名；全库标量与宿主数值都用它
nn::f16;              // IEEE binary16 值类型（2 字节，与 GPU R16F 内存布局一致）
```

f32↔f16 的转换一律 **round-half-to-even**；`static_cast<float>(nn::f16)` 升位精确无损。
你**不需要**手动 cast 操作数——引擎公共入口的边界 cast 层会自动处理（见 §5）。

### 1.3 错误处理宏（唯一一族，无异常）

```cpp
nn::Result<T>  = std::expected<T, nn::Error>;   // Error 含 .message
```

| 宏 | 语义 | 用在哪 |
|---|---|---|
| `NN_TRY(decl, expr)` | 声明 + 检查 + 失败即**传播** `return std::unexpected(...)` | 函数返回 `Result<T>` |
| `NN_TRY_MSG(decl, "语境", expr)` | 同上，失败时带上调用点语境 | 同上 |
| `NN_TRY_CHECK(x)` | 检查**已声明**的变量（失败传播） | `Result` 已在手 |
| `NN_CHECK(expr[, "语境"])` | **解包**：失败打印 `message + file:line` 后 `abort`；成功**返回解包后的值** | `main` / `void` / 测试 |
| `NN_EASY_CHECK(x)` | `NN_CHECK` 的别名 | 同上 |
| `NN_EXIT(expr, code[, "语境"])` | 解包，失败打印后 **`exit(code)`** | CLI，保住退出码 |
| `NN_FAIL("msg")` | **源头**：返回 `std::unexpected(Error{"msg"})` | 函数返回 `Result` |
| `NN_ASSERT_ALWAYS(cond, msg)` | 前置条件断言，**与构建类型无关**（NDEBUG 下同样终止） | 断言后还要用该对象 |
| `NN_FATAL(msg)` | 无条件 fail-fast（打印后终止） | 不可恢复 |

> ⚠ **`NN_CHECK` 名字叫 CHECK，其实是"解包"**，不是断言：
> `auto x = NN_CHECK(engine.from_matrix(m));` 里 **`x` 已经是 `Tensor`**，再写 `*x` 会编译错误。
> 返回 `Result<T>` 且 `T` 带 `[[nodiscard]]` 的入口（如 `Model::backward`）丢弃返回值要写
> `(void)NN_CHECK(...)`。`NN_TRY_CHECK(x)` 对带副作用的表达式会**重复求值**，只能对已声明变量用。

```cpp
// 三种典型写法
[[nodiscard]] nn::Result<nn::Tensor> step(nn::Model& model, const nn::Tensor& x)
{
    NN_TRY(out, model.forward(x));          // 传播
    return out;
}

int main()
{
    nn::CpuEngine engine;
    nn::Tensor t = engine.create_tensor(4, 4);          // 纯分配：查 .valid()
    NN_CHECK(t.valid(), "oom");                         // bool 条件也可直接喂给 NN_CHECK
    nn::Matrix m = NN_CHECK(engine.to_matrix(t));       // 解包 + 失败终止
    (void)m;
    return 0;
}
```

> 注意 `NN_FAIL(...)` 展开是 `return std::unexpected(...)`——只能出现在**返回 `Result`
> 的函数**里。在 `main` / `void` / 测试里做检查用 `NN_CHECK(cond[, "msg"])` 或 `NN_FATAL("msg")`，
> 否则 `-Wreturn-mismatch` 直接编译失败。

---

## 2. 作为库嵌入（速查）

两行 CMake + 一个 include：

```cmake
add_subdirectory(3rd_party/neuralnet.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

```cpp
#include <neuralnet.cpp/nn.hpp>
int main() { nn::CpuEngine engine; /* ... */ }
```

门面目标 `neuralnet::nn` 传递：C++26、include 目录、线程库、GPU 宏 `NN_HAS_VULKAN`、
SPIR-V 嵌入头目录、Vulkan 链接与 shader 生成顺序。嵌入时 `NN_BUILD_APPS` 默认 OFF、
`NN_ENABLE_GPU` 默认 `AUTO`。自研 `nn::dsl::compute` 表达式要一行
`nn_enable_gpu_fusion(...)` 才进 GPU 闭合世界。

> **完整版（全部构建开关、GPU 语义、融合注册、故障排查）见 [`cmake-integration.md`](cmake-integration.md)**；
> 消费方视角的精简版见 [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md)。

---

## 3. 引擎：选择与创建

引擎是所有计算的宿主，必须**先于**任何持有其非拥有指针的对象（`nn::Model`、`nn::Tensor`）析构。

```cpp
// ① CPU（默认，无额外依赖）
nn::CpuEngine engine;

// ② GPU（需 Vulkan，编译期宏 NN_HAS_VULKAN）
#ifdef NN_HAS_VULKAN
nn::GpuBackend& backend = nn::GpuBackend::instance();      // 进程单例
auto r = backend.initialize("40HX");                       // 空串 = 自动选卡；也可传枚举索引 "2"
if (r)
{
    nn::GpuEngine gpu_engine(backend);
}
else
{
    std::fprintf(stderr, "%s\n", r.error().message.c_str());
}
#endif
```

CLI 里更常用的是**工厂函数**（与各可执行文件同一套选择逻辑；显式请求失败**不回退 CPU**）：

```cpp
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

nn::cli::EngineConfig cfg;
cfg.use_gpu    = true;      // 启用 Vulkan
cfg.gpu_device = "NVIDIA";  // 索引 "2" 或名称子串 "40HX"/"NVIDIA"；空 = 自动
std::unique_ptr<nn::ComputeEngine> engine =
    NN_CHECK(nn::cli::create_engine(cfg, std::cerr));
```

```cpp
// 引擎能力是**只读信息**，经 describe() 披露（EngineInfo）——禁止用它选路（铁律 #13）
std::cout << engine->describe().text() << "\n";
```

> 引擎原语（matmul / 归约 / 数据搬运 / 扫描 / 表达式求值 / 批处理录制）的**完整清单与签名**
> 见引擎手册 §19 的「方法速查总表」。本文不再重复。

---

## 4. 张量：创建、初始化、读写

`Tensor` 是跨设备统一句柄（CPU 持 `Matrix<P>`，GPU 持 `GpuTensor<P>`，内部 `std::shared_ptr`，
拷贝廉价）。**公共面只有**：`device / precision / rows / cols / size / is_cpu / is_gpu / valid /
shape_str / bound / engine`——**存储访问器与静态工厂已私有**（铁律 #11）。

### 4.1 创建

```cpp
// 纯分配（3 参）：CPU 零填充，GPU 未初始化；失败返回**无效张量**（查 .valid()）
nn::Tensor t = engine.create_tensor(784, 32);
NN_CHECK(t.valid(), "create_tensor failed");

// 指定精度
nn::Tensor h = engine.create_tensor(784, 32, nn::Precision::F16);

// 声明式初始化（4 参，推荐）：层算分布参数，引擎填数；分布类 seed 必填
nn::Tensor w = engine.create_tensor(32, 10, nn::Precision::F32,
                                    nn::InitSpec::uniform(-0.1f, 0.1f, nn::kInitSeed));
nn::Tensor b = engine.create_tensor(10, 1, nn::Precision::F32, nn::InitSpec::zero());

// 需要结构化错误（而非"无效张量"）时用 try_create_tensor：失败会说明原因
nn::Tensor s = NN_CHECK(engine.try_create_tensor(10, 1, nn::Precision::F32,
                                                 nn::InitSpec::normal(0.f, 0.02f, nn::kInitSeed)));
```

`nn::InitSpec` 工厂：`uninitialized()` / `zero()` / `constant(v)` / `uniform(lo, hi, seed)` /
`normal(mean, stddev, seed)`。**同 seed 的不同张量不撞流**（引擎把 seed 与创建序号混流）
→ 跨进程初值逐字节确定（铁律 #8）。

### 4.2 宿主 ↔ 张量（只准经这 7 个动词）

```
from_matrix / to_matrix / import / read / write / get_index / set_index
```

```cpp
nn::Matrix m(784, 32);                                   // 宿主 I/O 载体（行主序）
// ... 填 m ...
nn::Tensor t   = NN_CHECK(engine.from_matrix(m));        // 上传
nn::Matrix back = NN_CHECK(engine.to_matrix(t));         // 下载（GPU 会先等在飞命令）

// 大批量、免 Matrix 开销（性能敏感路径用它）：span 直读直写
std::vector<nn::Scalar> host(t.size());
NN_CHECK(engine.write(t, std::span(host)));              // 宿主 → 张量
NN_CHECK(engine.read(t, std::span(host)));               // 张量 → 宿主
```

`reshape` / `clone` / `cast` / `copy_from` 等语义见引擎手册 §4–§6。要点：
`engine.reshape(t, rows, cols)`（**不是** `t.reshape()`，那个已私有）；元素数不匹配报错。

### 4.3 布局（必须背下来）

```
Matrix（宿主）：行主序 (rows, cols)，data[row*cols + col]
神经网络张量：**列主序 batch-major**
  输入 (feature_dim, batch_size)   每列一个样本
  权重 (out_features, in_features)
GPT 序列展平：列序 i = b*seq + t（batch 在列方向，铁律 #5）
```

最典型的错误是写成 `create_tensor(batch, features)`——正确是 `(features, batch)`。

---

## 5. 精度：`PrecisionProfile` 与预设

```cpp
enum class nn::Precision : std::uint8_t { F16, F32, BF16, F64, T1_58 };
// 存储 / 舍入精度只有 F16 / F32；BF16 / F64 是保留值（使用即报错）；
// T1_58 = 三值权重**声明**（不是存储精度：precision_bytes(T1_58) == 0）。

struct nn::PrecisionProfile {
    Precision param     = Precision::F32;   // 权重 / 嵌入表 / 参数存储
    Precision compute   = Precision::F32;   // 常规算子（matmul / 逐元素 / 数据操作）
    Precision stable    = Precision::F32;   // 数值敏感算子（softmax / Norm / loss）
    Precision optimizer = Precision::F32;   // 优化器状态（m / v / momentum）
};
```

现成配方（与 CLI 预设一一对应）：

```cpp
nn::profile_f32();               // 全 F32（默认）
nn::profile_f16();               // --f16：param=F16, compute=F16, stable=F32, optimizer=F32
nn::profile_t1_58();             // --t1_58：param=T1_58（其余三槽 = profile_f16）
nn::profile_master_weights();    // f32 主权重 + f16 计算 + f32 稳定 / 优化器
nn::profile_all_f16();           // 四槽全 F16（激进实验配方，当前数值上不可训练）
```

两种用法：

```cpp
// ① 模型级：在 add() 之前设默认 profile（权重按精度创建）
nn::Model model(engine);
model.set_default_precision_profile(nn::profile_f16());
NN_CHECK(model.add<nn::Linear>(784, 256));

// ② 工厂直接传：build_* 的 precision 形参
nn::GptConfig cfg;
cfg.precision = nn::profile_t1_58();
nn::Model gpt = NN_CHECK(nn::build_gpt_model(engine, cfg));
```

**唯一要记的口诀**：*f16 只改变"数据存哪儿"，不改变"算子怎么算"*——每个算子以 f32
参考计算、输出舍入到目标 P；matmul / 归约累加恒 f32。边界 cast 全在引擎公共入口完成，
**不要自己 cast 操作数**。

层的能力校验集中在 `Layer::init()` 一处：不兼容组合（例如给普通 `Linear` 传 `param=T1_58`）
= **初始化期报错**，信息含层名 + 槽位 + 取值 + 允许集合。三值权重只被 `BitLinear` 接受
（GPT / MLP 工厂会把 `param=T1_58` 自动分流到 BitLinear；RAPT / ViT 明确拒绝）。

> 完整语义契约（P 精度算术的形式化定义、分派规则、实测数值限制）见
> [`05-mixed-precision.md`](05-mixed-precision.md)。

---

## 6. 表达式 DSL：写逐元素 / 归约数学

DSL 是库内**所有**逐元素与归约运算的写法（层不写手工循环）。三个入口：

```cpp
// 输出一个**新张量**（默认 P = F32）
nn::Tensor y = NN_CHECK(nn::dsl::compute(engine, expr, rows, cols, nn::Precision::F32));

// 就地写进已有张量（只支持逐元素表达式，无归约）
NN_CHECK(nn::dsl::compute_into(engine, expr, dst));

// 归约：输出归约向量（(rows,1) 或 (1,cols) 原生形状）
nn::Tensor r = NN_CHECK(nn::dsl::compute_reduce(engine, expr, rows, cols, P));
```

叶子与算子（全部在 `namespace nn::dsl`）：

```cpp
nn::dsl::leaf(t)                          // 张量输入
nn::dsl::rparam(v)                        // 标量常量（值不进 expr_spec_key）
nn::dsl::matmul(a, b, transA, transB)     // 一个表达式**至多一个** matmul
nn::dsl::row_broadcast(t)                 // (F,1) → 沿行广播
nn::dsl::col_broadcast(t)                 // (1,B) → 沿列广播
nn::dsl::col_reduce_sum(x), nn::dsl::row_reduce_sum(x)
nn::dsl::col_reduce_max(x), nn::dsl::row_reduce_max(x)
nn::dsl::select(cond, a, b)               // 唯一的条件选择（cond 是 BoolExpression）
t1 + t2, t1 - t2, t1 * t2, t1 / t2, -t    // 运算符重载
```

```cpp
// 例：逐列减去列最大值（数值稳定 softmax 的一步）
nn::Tensor z = NN_CHECK(nn::dsl::compute(engine,
    nn::dsl::leaf(x) - nn::dsl::col_broadcast(mx), x.rows(), x.cols(), P));
```

> **四条必须知道的约束**：① 一个表达式只允许**一个** matmul（`ExprSpec::matmul` 是单个
> optional）——两层 matmul 要拆成两次 `dsl::compute`；② 归约是独立叶子，配合
> `compute_reduce` 用；③ 条件是 `nn::dsl::BoolExpression`（比较 / 逻辑运算的结果），
> 只能喂给 `select`；④ 传 `T1_58`/`BF16`/`F64` 作输出 P 会被拒绝。
>
> 一元函数（`exp`/`log`/`sqrt`/`tanh` 等 7 个 + 一元负号）、视图 / 广播 / 归约的**完整清单**
> 与 5 个陷阱见引擎手册 §8。GPU 侧：库内层的融合 shader 构建期已生成；**自研表达式**
> 需一行 CMake 注册（见 §2 / [`cmake-integration.md`](cmake-integration.md)），否则运行期硬报错。

---

## 7. 层与 `Model` 容器

### 7.1 `Model`

```cpp
nn::Model model(engine);                          // 必须绑定引擎（默认构造后使用时断言失败）

// 添加层：add<T>(args...) 构造 + 自动 init(engine)，返回 Result<void>（不能链式调用）
NN_CHECK(model.add<nn::Linear>(784, 256));
NN_CHECK(model.add<nn::ReLU>());
NN_CHECK(model.add<nn::Linear>(256, 10));

// 添加工厂构造的层（如按 NormType 建归一化层），可显式注入 profile
NN_CHECK(model.add_layer(nn::make_norm_layer(256, nn::NormType::RMSNorm)));
NN_CHECK(model.add_layer(nn::make_linear_layer(256, 10, nn::WeightQuant::T1_58), profile));

nn::Tensor out = NN_CHECK(model.forward(input));       // (10, batch)
nn::Tensor gin = NN_CHECK(model.backward(grad_out));   // 逐层反向

std::vector<nn::TensorRef> ps = model.parameters();    // 供优化器
std::vector<nn::TensorRef> gs = model.param_gradients();
NN_CHECK(model.zero_grad());                           // 每步开头
```

其它 `Model` 方法：

| 方法 | 作用 |
|---|---|
| `num_layers()` / `layer_at(i)` | 层数与按序号取 `Layer&`（越界即 fail-fast） |
| `engine()` | 取绑定引擎 |
| `set_spec(spec)` / `spec()` | 记录 / 读取架构 spec（`load_model` 用它校验） |
| `set_default_precision_profile(p)` | **add 之前**调用，给后续层设默认 profile |
| `set_precision_profile(p)` | 给已有顶层层注入 profile（复合层会下传子层） |
| `set_flush_interval(n)` | GPU：每 n 个块 flush 一次（防 TDR；GPTModel 生效） |
| `set_checkpoint_every(n)` | 梯度检查点（激活重计算）间隔 |
| `set_activation_offload(bool)` | 激活卸载到 host 可见内存 |
| `offload_ram_bytes()` | 理论 offload 字节数（诊断） |
| `set_training(bool)` | 训练 / 推理模式（BatchNorm 等用） |
| `set_doc_ids(ids)` | 文档边界（掩码注意力用） |
| `extra_state()` | 非可学习状态（BatchNorm running 统计量，供序列化） |

### 7.2 内置层目录（构造签名）

全部层都派生自 `nn::Layer`，经 `Model::add<T>(...)` 或
`Model::add_layer(std::make_unique<T>(...))` 接入。`=` 后为默认值。

| 层 | 构造签名 | 用途 |
|---|---|---|
| `nn::Linear` | `Linear(in_features, out_features)` | 全连接 |
| `nn::BitLinear` | `BitLinear(in_features, out_features)` | 三值权重线性（param 必须 T1_58） |
| `nn::ReLU` | `ReLU()` | ReLU |
| `nn::GeLU` | `GeLU()` | QuickGeLU |
| `nn::SwiGLU` | `SwiGLU()` | 门控激活 |
| `nn::ReLU2GLU` | `ReLU2GLU()` | 门控平方 ReLU |
| `nn::LayerNorm` | `LayerNorm(normalized_shape, epsilon = EPSILON)` | 层归一化 |
| `nn::RMSNorm` | `RMSNorm(normalized_shape, epsilon = EPSILON)` | RMSNorm |
| `nn::BatchNorm` | `BatchNorm(features, epsilon = EPSILON, momentum = MOMENTUM)` | 批归一化（含 running 统计量） |
| `nn::Softmax` | `Softmax()` | 行级 softmax |
| `nn::FeedForward` | `FeedForward(d_model, d_ff, activation = GeLU, subln = false, subln_norm_type = LayerNorm, weight_quant = None)` | FFN |
| `nn::PositionalEncoding` | `PositionalEncoding(d_model, max_len = 5000, tile_size = 0)` | 正弦位置编码 |
| `nn::TransformerEncoderLayer` | `TransformerEncoderLayer(d_model, num_heads, d_ff, seq_len = 0, norm_type = LayerNorm)` | Pre-Norm 编码器层 |
| `nn::TransformerEncoder` | `TransformerEncoder(d_model, num_heads, d_ff, num_layers, num_patches, norm_type = LayerNorm, final_norm = false)` | ViT 编码器 |
| `nn::PatchEmbedding` | `PatchEmbedding(img_size, patch_size, d_model)` | 图像 patch 嵌入 |
| `nn::CausalSelfAttention` | `CausalSelfAttention(d_model, num_heads, max_len = 1024, seq_len = 0, pos_enc = Learned, subln = false, subln_norm_type = LayerNorm, n_head_kv = 0, weight_quant = None)` | 因果自注意力 |
| `nn::GPTBlock` | `GPTBlock(d_model, num_heads, d_ff, max_len = 1024, seq_len = 0, pos_enc = Learned, activation = GeLU, norm_type = LayerNorm, precision = {}, subln = false, n_head_kv = 0, weight_quant = None)` | Pre-Norm 解码块 |
| `nn::GPTModel` | `GPTModel(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers, pos_enc_type = Learned, activation = GeLU, norm_type = LayerNorm, precision = {}, subln = false, n_head_kv = 0, tie_embeddings = false, weight_quant = None)` | 完整 GPT（单层封装） |
| `nn::Conv2D` | `Conv2D(in_channels, out_channels, kernel, stride = 1, padding = 0, in_h = 0, in_w = 0)` | 卷积 |
| `nn::MaxPool2D` | `MaxPool2D(channels, in_h, in_w, pool = 2, stride = 0)` | 最大池化 |
| `nn::ReLULinearAttention` | `ReLULinearAttention(d_model, num_heads, seq_len = 0, causal = true, pos_enc = RoPE)` | RLA 注意力 |
| `nn::RAPTBlock` | `RAPTBlock(d_model, num_heads, d_ff, seq_len, pos_enc, activation = GeLU, norm_type = LayerNorm, causal = true, precision = {})` | RAPT 块 |
| `nn::RAPTModel` | `RAPTModel(vocab_size, d_model, seq_len, num_heads, d_ff, num_layers, pos_enc = RoPE, activation = GeLU, norm_type = LayerNorm, causal = true, precision = {})` | 完整 RAPT |

> `nn::MultiHeadAttention` 的构造函数是 **protected**（设计为基类），不通过 `add` 直接添加——
> 用 `CausalSelfAttention` / `TransformerEncoderLayer` 组合。
>
> 每个层的**数学公式与原语分解**见
> [`../introduction/03-algorithm-reference.md`](../introduction/03-algorithm-reference.md)。

### 7.3 归一化 / 线性工厂（按枚举建层）

```cpp
std::unique_ptr<nn::Layer> nn::make_norm_layer(std::size_t features, nn::NormType t);
std::unique_ptr<nn::Layer> nn::make_linear_layer(std::size_t in, std::size_t out,
                                                 nn::WeightQuant q = nn::WeightQuant::None);
```

`nn::NormType` = `LayerNorm` / `RMSNorm` / `BatchNorm`；`nn::WeightQuant` = `None` / `T1_58`。

### 7.4 位置编码

模型工厂内部已按 spec 选好；手动组装时用工厂（`compute_position_encoding.hpp`）：

```cpp
std::unique_ptr<nn::PositionEncoder> pe =
    nn::make_attention_position_encoder(nn::PosEncodingType::RoPE,
                                        d_model / num_heads, num_heads);   // 注意力注入点
// 嵌入注入点：nn::make_embedding_position_encoder(type, d_model, seq_len)
//   （Learned / Sinusoidal 生效；ALiBi / RoPE 返回 NoPositionEncoder）
```

`nn::PosEncodingType` = `Learned` / `Sinusoidal` / `ALiBi` / `RoPE`。

---

## 8. 模型工厂（`domain_*.hpp`）

工厂 = 一条调用建出带正确权重初始化与正确 spec 的模型，是最省心的入口。

### 8.1 MNIST（MLP / ViT）

```cpp
// MLP：784 → 512 → 256 → 128 → 64 → 10（默认层维度）
nn::Model mlp = NN_CHECK(nn::build_mnist_mlp_model(engine));

// ViT：28×28 图像，patch=7（默认含末端 final norm）
nn::Model vit = NN_CHECK(nn::build_mnist_transformer_model(engine));

// 从 spec 还原（加载 .nnmodel 时用）
nn::ModelSpec spec = NN_CHECK(nn::peek_any_model_spec("mnist_model.bin"));
nn::Model m2 = NN_CHECK(nn::build_mnist_model_from_spec(engine, spec, nn::profile_f32()));
```

签名（默认值节选）：
`build_mnist_mlp_model(engine, layer_dims = MNIST_LAYER_DIMS, norm_type = LayerNorm, precision = {})`；
`build_mnist_transformer_model(engine, img_size = 28, patch_size = 7, d_model = 64, num_heads = 4, d_ff = 128, num_layers = 2, precision = {}, norm_type = LayerNorm, norm_place = NormPlace::Final)`。

### 8.2 GPT

```cpp
nn::GptConfig cfg;                        // 每个字段都有默认值
cfg.vocab_size = 10000; cfg.d_model = 128; cfg.seq_len = 256;
cfg.num_heads  = 4;     cfg.d_ff = 512;    cfg.num_layers = 4;
cfg.pos_enc    = nn::PosEncodingType::RoPE;
cfg.activation = nn::ActivationType::SwiGLU;
cfg.norm_type  = nn::NormType::RMSNorm;
cfg.n_head_kv  = 0;                       // 0 = MHA；>0 = GQA 的 KV 头数
cfg.tie_embeddings = false;               // true = head 复用 token embedding
cfg.precision  = nn::profile_f16();
nn::Model gpt = NN_CHECK(nn::build_gpt_model(engine, cfg));

// 位置参数版（等价的简写）
nn::Model gpt2 = NN_CHECK(nn::build_gpt_model(engine, 10000, 128, 256, 4, 512, 4));
```

### 8.3 CNN 与 RAPT

```cpp
// CNN（LeNet 风格可自定义）
nn::CnnConfig c;                          // in_channels / in_size / pool / convs / fc_dims ...
nn::Model cnn = NN_CHECK(nn::build_cnn_model(engine, c));
nn::ModelSpec cspec = nn::make_cnn_spec(1, 28, 2, nn::MNIST_CNN_CONVS, nn::MNIST_CNN_FC);

// RAPT（线性注意力 causal LM；默认 pos_enc = RoPE，RLA 约束）
nn::RAPTConfig r;
nn::Model rapt = NN_CHECK(nn::build_rapt_model(engine, r));
```

### 8.4 spec 构造与还原（统一表）

| 模型族 | 从 spec 还原 | 构造 spec |
|---|---|---|
| MNIST MLP / ViT | `nn::build_mnist_model_from_spec` | `nn::make_mnist_transformer_spec` |
| GPT | `nn::build_gpt_model_from_spec` | `nn::make_gpt_spec` |
| CNN | `nn::build_cnn_model_from_spec` | `nn::make_cnn_spec` |
| RAPT | `nn::build_rapt_model_from_spec` | `nn::make_rapt_spec` |

`nn::ModelSpec`（`model_spec.hpp`）是架构的**唯一描述**：`type`（`nn::ModelType`）、
`vocab_size` / `d_model` / `seq_len` / `num_heads` / `d_ff` / `num_layers` / `patch_size`、
`layer_dims`、`pos_encoding` / `activation` / `norm_type` / `norm_place`、
`subln` / `n_head_kv` / `tie_embeddings` / `weight_quant`、CNN 的
`cnn_in_channels` / `cnn_in_size` / `cnn_pool` / `cnn_channels` 等。`nn::spec_matches(a, b)`
判定两份 spec 是否兼容（`load_model` 用它拦截架构不匹配）。

---

## 9. 损失函数（`compute_loss.hpp`）

```cpp
class nn::Loss {
public:
    void set_precision_profile(const nn::PrecisionProfile& p);
    [[nodiscard]] nn::Result<nn::Scalar> forward(nn::ComputeEngine&, const nn::Tensor& pred,
                                                 const nn::Tensor& target);
    [[nodiscard]] nn::Result<nn::Tensor> backward();   // 对 pred 的梯度
};
```

### 9.1 `nn::MSELoss`

```cpp
nn::MSELoss loss;
nn::Scalar l    = NN_CHECK(loss.forward(engine, pred, target));   // pred/target 同形状
nn::Tensor grad = NN_CHECK(loss.backward());                      // = (2/N)·(pred−target)
```

### 9.2 `nn::CrossEntropyLoss`（含 softmax）

```cpp
// 稠密版：pred/target 都是 (classes, batch)
nn::CrossEntropyLoss loss;
nn::Scalar l    = NN_CHECK(loss.forward(engine, logits, one_hot_target));
nn::Tensor grad = NN_CHECK(loss.backward());                      // (softmax − target)/batch
```

大词表**禁止物化 one-hot**（铁律 #9）——用**稀疏标签**版：

```cpp
std::vector<std::size_t> labels;   // 平坦标签，长度 = logits.cols()
std::vector<nn::Scalar>  mask;     // 可选；空 = 全参与；>0.5 参与 loss

// ① 同步版（测试 / 评估）：内部下载标量
nn::Scalar l = NN_CHECK(loss.forward_sparse(engine, logits, labels, mask, vocab_size));

// ② 热路径版：设备端求和 + 异步标量回读（避免每 step 全流水线 drain）
std::size_t num_valid = 0;
nn::Tensor loss_sum = NN_CHECK(loss.forward_sparse_sum(engine, logits, labels, mask,
                                                       vocab_size, num_valid));
// 训练热循环配合 engine.submit_scalar_readback(...) 异步取回
```

> 稀疏 CE 走**融合路径**（online-softmax 单趟 fold），不物化全 softmax；
> 前置条件是 `vocab_size <= 2^24`。稠密 / 稀疏两条路径的数值都稳定（避免 `0·log(0)=NaN`）。

---

## 10. 优化器（`compute_optimizer.hpp`）

### 10.1 工厂（名单唯一真源 = `kOptimizerTable`）

```cpp
std::unique_ptr<nn::Optimizer> nn::create_optimizer(
    std::string_view name, nn::ComputeEngine& engine,
    std::vector<nn::TensorRef> params, std::vector<nn::TensorRef> grads,
    nn::Scalar lr, nn::Scalar weight_decay = 0,
    nn::PrecisionProfile precision = {}, nn::Scalar beta1 = 0.9, nn::Scalar beta2 = 0.999);
```

```cpp
auto opt = nn::create_optimizer("adamw", engine, model.parameters(),
                                model.param_gradients(), 1e-3f, 0.01f, nn::profile_f16());
NN_CHECK(opt != nullptr, "unknown optimizer");
```

可用名字（`nn::optimizer_names(", ")` 可打印）：`sgd` / `sgd_momentum` / `adam` /
`adamw` / `adamwquant` / `muon`。未知名字返回 `nullptr`（由调用方处理）。
`beta1` / `beta2` 仅 `adam` / `adamw` 系列使用（LLM 预训练常取 `beta2=0.95`）。
`nn::find_optimizer(name)` 查元数据（状态缓冲数、是否半精度存储）。

### 10.2 训练循环里的调用

```cpp
opt->set_lr(lr);                      // 动态改学习率（调度器用）
NN_CHECK(opt->zero_grad());           // 每步开头清零（或 model.zero_grad()；均 [[nodiscard]]）
// ... forward / loss / backward ...
NN_CHECK(opt->clip_grad_norm(1.0f));  // 全局 L2 范数裁剪（在 step 之前；<=0 或空则 no-op）
NN_CHECK(opt->step());
```

其它：`opt->precision_profile()`、`opt->set_precision_profile(p)`。

---

## 11. 完整训练循环模板

```cpp
#include <neuralnet.cpp/nn.hpp>

int main()
{
    nn::CpuEngine engine;

    // 1. 模型 + 优化器 + 损失
    nn::Model model = NN_CHECK(nn::build_mnist_mlp_model(engine));
    auto opt = nn::create_optimizer("adam", engine, model.parameters(),
                                    model.param_gradients(), 1e-3f);
    nn::CrossEntropyLoss loss;

    const std::size_t batch = 64;
    for (int step = 0; step < 400; ++step)
    {
        // 2. 取一个 batch：输入 (784, batch)，标签 (10, batch)
        nn::Tensor x = engine.create_tensor(784, batch, nn::Precision::F32);
        nn::Tensor y = engine.create_tensor(10,  batch, nn::Precision::F32);
        // ... 用 engine.write(...) / engine.from_matrix(...) 填 x, y ...

        // 3. 一步
        NN_CHECK(opt->zero_grad());
        nn::Tensor out  = NN_CHECK(model.forward(x));
        nn::Scalar l    = NN_CHECK(loss.forward(engine, out, y));
        nn::Tensor grad = NN_CHECK(loss.backward());
        (void)NN_CHECK(model.backward(grad));
        NN_CHECK(opt->clip_grad_norm(1.0f));
        NN_CHECK(opt->step());

        if (step % 50 == 0)
            std::printf("[step %d] loss=%.4f\n", step, static_cast<double>(l));
    }
    return 0;
}
```

> 含真实数据加载的完整版见 `src/mnist_train.cpp` / `src/text_train.cpp`，以及
> [`../usage/02-quickstart-train-infer.md`](../usage/02-quickstart-train-infer.md) 的「C++ API 编程训练」节。
> GPU 上把 forward / backward 包进 `engine.begin_batch()` / `engine.end_batch()` 可消去
> per-primitive 提交开销（见引擎手册 §3.5）。

---

## 12. 模型文件：角色、保存、加载

**一份格式、两种角色**（`model_serialization.hpp`，v7）：`*.nnmodel` = 推理角色
（三值层写打包形态），`*.nnckpt` = 训练断点（写 f32 latent）。文件头 =
`magic | version | 精度 | role | spec`，之后是**参数级 `kind`** 条目。

```cpp
// 保存：role 默认 Model（.nnmodel）；断点传 ModelRole::Checkpoint
NN_CHECK(nn::save_model("gpt.nnmodel", model, spec));
NN_CHECK(nn::save_model("gpt_ckpt.nnckpt", model, spec, /*tokenizer_data=*/{},
                        nn::ModelRole::Checkpoint));

// 只读头：拿架构 spec / 文件角色（不读参数，用于先建对模型）
nn::ModelSpec s1    = NN_CHECK(nn::peek_any_model_spec("gpt.nnmodel"));
nn::ModelSpec s2    = NN_CHECK(nn::peek_model_spec("gpt.nnmodel"));
nn::ModelRole role  = NN_CHECK(nn::peek_model_role("gpt_ckpt.nnckpt"));
std::cout << nn::model_role_name(role) << "\n";

// 加载：先用 spec 建同构模型，再灌参数（tokenizer_out 取内嵌词表）
nn::Model model = NN_CHECK(nn::build_gpt_model_from_spec(engine, s1));
std::string embedded_tok;
NN_CHECK(nn::load_any_model("gpt.nnmodel", model, &embedded_tok));
```

相关类型：

- `nn::ModelRole`：`Model`（`.nnmodel`）/ `Checkpoint`（`.nnckpt`）；`nn::model_role_name(r)` 取名字。
- `nn::ParamKind`：`Dense` / `TernaryLatent` / `TernaryPacked2` / `TernaryPacked3` /（预留）`I8Packed4`；
  `nn::param_kind_name(k)` 取名字。形态由**拥有该参数的层**经 `Layer::export_param` /
  `import_param` 决定，不被接受即 fail-fast（铁律 #14）。

> 版本兼容：`load_model` 接受 v4 及以上自描述格式；v5/v6 老 `.bin` 仍可读
> （v6 起内嵌 `.nnvocab` 词表）。v1–v3 在头校验阶段拒绝并提示重新训练保存。

---

## 13. 数据集：`nn::Dataset` 与生成 CLI

`.nndataset` 是统一数据集：token 流、词表、loss 掩码、test 子集都在**生成期**定好并嵌入。

### 13.1 生成与转换（CLI）

```bash
# 先训词表，再打包数据集
./build/tokenizer_train corpus.txt --vocab-size 10000 --output my_bpe.nnvocab
./build/dataset_gen corpus.txt --vocab my_bpe.nnvocab -o llm.nndataset

# 对话 SFT：生成期写 assistant 掩码
./build/dataset_gen dialogs.txt --vocab my_bpe.nnvocab -o sft.nndataset --loss-scope assistant

# 旧 JSON 词表迁移
./build/dataset_convert vocab old.json -o old.nnvocab
```

### 13.2 C++ 读取

```cpp
#include <neuralnet.cpp/dataset.hpp>

nn::Dataset ds = NN_CHECK(nn::Dataset::open("llm.nndataset"));   // 也接受 .nnvocab（词表独立场景）
const nn::DatasetInfo& info = ds.info();    // kind / num_docs / test_docs / loss_scope / source_*

nn::TextCorpus train = NN_CHECK(ds.load_text(false));   // train 子集
nn::TextCorpus test  = NN_CHECK(ds.load_text(true));    // test 子集（无则报错）

const nn::KeyValueRecord& vocab = ds.vocab_kvrec();      // 词表（模型 v6 内嵌词表来源）
```

`nn::TextCorpus { token_flow, doc_ids, loss_mask }`——三者都是扁平数组，
`doc_ids` 是每 token 的文档 id（1 起），`loss_mask` 在 `loss_scope=all` 时全 1。
从 `token_flow` 到训练张量（拼流 → 滑窗 → batch）的写法见 `src/text_train.cpp`；
窗口是**训练策略**，不在数据集里（数据集只读、不做窗口化）。

---

## 14. 分词器（`domain_tokenizer*.hpp`）

```cpp
class nn::Tokenizer {
public:
    [[nodiscard]] std::vector<std::size_t> encode(const std::string& text) const;
    [[nodiscard]] std::string decode(std::span<const std::size_t> ids) const;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
    [[nodiscard]] const std::vector<std::string>& vocab() const noexcept;
    [[nodiscard]] std::size_t bos_id() const noexcept;
    [[nodiscard]] std::size_t eos_id() const noexcept;
    [[nodiscard]] std::size_t pad_id() const noexcept;
    void set_encode_threads(std::size_t n) noexcept;   // 0=自动（全部核心），1=顺序
    [[nodiscard]] nn::Result<void> save(const std::string& path) const;
    [[nodiscard]] nn::Result<void> load(const std::string& path);
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
};
```

两种实现：`nn::BPETokenizer`（字节级，适合英文）与 `nn::CharBPETokenizer`（字符级，适合中文）。

```cpp
// 训练一个 BPE
nn::BPETokenizer tok;
nn::BPETokenizer::Config cfg;
cfg.vocab_size    = 10000;
cfg.min_freq      = 2;
cfg.show_progress = true;
cfg.threads       = 0;               // 预分词并行度：0=自动 / 1=顺序 / >1=指定
NN_CHECK(tok.train(text, cfg));
NN_CHECK(tok.save("my_bpe.nnvocab"));        // .nnvocab（kvrec v2 自描述格式）

// 编码 / 解码
std::vector<std::size_t> ids = tok.encode("Hello world");
std::string s = tok.decode(ids);

// CharBPE 同款 API：nn::CharBPETokenizer（PAD=0 / UNK=1 / BOS=2 / EOS=3）
```

**从文件 / 模型内嵌词表加载**：

```cpp
std::unique_ptr<nn::Tokenizer> t1 = NN_CHECK(nn::load_tokenizer_from_file("my_bpe.nnvocab"));
std::unique_ptr<nn::Tokenizer> t2 = NN_CHECK(nn::load_tokenizer_from_model_blob(embedded_tok, model_version));
std::unique_ptr<nn::Tokenizer> t3 = NN_CHECK(nn::load_tokenizer_from_bytes(bytes));  // 按内容嗅探
std::unique_ptr<nn::Tokenizer> t4 = nn::make_tokenizer("bpe");   // 只造对象、未加载（"bpe"/"char_bpe"）
```

> 旧 JSON 词表**不直读**，用 `dataset_convert vocab` 迁移。**任意并行度下 encode 输出
> 与顺序执行逐字节一致**（铁律 #8）。

---

## 15. 自回归生成（`GPTModel` / `RAPTModel`）

`nn::GPTModel` 与 `nn::RAPTModel` 各自暴露 `generate`（带 KV cache，按 `seq_len` 滑动窗口）：

```cpp
nn::Model model = NN_CHECK(nn::build_gpt_model_from_spec(engine, spec));
NN_CHECK(nn::load_any_model("gpt.nnmodel", model));

// Model 只有一个顶层层；取出来 downcast
auto* gpt = dynamic_cast<nn::GPTModel*>(&model.layer_at(0));
if (gpt != nullptr)
{
    std::vector<std::size_t> prompt = tok->encode("Once upon a time");
    std::vector<std::size_t> out = NN_CHECK(gpt->generate(
        engine, prompt, /*max_new_tokens=*/200, /*temperature=*/0.8,
        /*eos_token_id=*/tok->eos_id(), /*min_new_tokens=*/0));
    std::cout << tok->decode(out) << "\n";
}
```

签名：`generate(engine, prompt, max_new_tokens, temperature = 1.0,
eos_token_id = npos, min_new_tokens = 0)`。`temperature = 0` 为贪心。
完整流程（加载模型含内嵌词表 → 编码 → 生成 → 解码）见 `src/text_infer.cpp`。

---

## 16. 剖析与诊断

训练 step 剖析器（`core_profile.hpp`）默认**零开销**，设 `NN_PROFILE=1` 打开：

```bash
NN_PROFILE=1 ./build/text_train llm.nndataset --epochs 1
NN_PROFILE=1 NN_PROFILE_TRACE=trace.json ./build/mnist_train --epochs 1
```

```cpp
// 自己加计时点：name 必须是字符串字面量或静态存储（聚合表按地址做 key，不拷贝）
for (std::size_t li = 0; li < model.num_layers(); ++li)
{
    NN_PROF_SCOPE(nn::prof::interned("fwd.layer", li));   // 动态名经 interned 驻留
    // ...
}
NN_PROF_OP("my_primitive");   // 原语级计时，归到最内层 stage
nn::prof::dump(stdout);       // 进程结束前打表（stage / op / GPU / per-key）
```

常用环境变量（**诊断类只在 `-DNN_ENABLE_PROBES=ON` 构建下生效**；初始化期读一次并缓存的
行为开关是例外）：

| 环境变量 | 作用 |
|---|---|
| `NN_PROFILE=1` / `NN_PROFILE_TRACE=<path>` / `NN_PROFILE_EVERY=<n>` | host + 设备侧聚合计时 / Chrome trace / 每 n 次 dump 打表 |
| `NN_F16_DEBUG` / `NN_PREC_TRACE` / `NN_NAN_TRACE` / `NN_MEM_STATS` | 精度 / NaN / 内存归因探针（探针门 `NN_ENABLE_PROBES`） |
| `NN_BIND_DEBUG=1` | 未绑定输入进引擎即报错（抓漏网绑定） |
| `NN_VULKAN_DEVICE` / `NN_VULKAN_NO_16BIT_ALU` / `NN_VULKAN_NO_TIMELINE` | GPU 设备与特性逃生阀 |
| `NN_POOL_BLOCK_MB` / `NN_POOL_LADDER_MAX_MB` / `NN_NO_EARLY_REAP` / `NN_CPU_PAUSE` / `NN_GPU_PROFILE` | 内存池 / 调度 A/B |

> 探针门语义（哪些算探针、默认为什么不编译）见 [`cmake-integration.md`](cmake-integration.md)。

---

## 17. 错误 / 断言宏速查

见 §1.3 的表；两条易混点再强调：

- `NN_CHECK` = **解包 + 失败终止**（不是断言）；`NN_TRY` = **声明 + 传播**。
- 断言后仍会**使用被校验对象**（解引用 / 索引 / 除零）的点位，一律用 `NN_ASSERT_ALWAYS` /
  `NN_FATAL`（`core_assert.hpp`，与构建类型无关）；`NN_ASSERT` 是"可选诊断"（NDEBUG 展开为空），
  只留给失败也不影响后续正确性的检查。

失败协议要记牢（与 `create_tensor` 的"无效张量"不同）：
`from_matrix` / `clone` / `cast` / `reshape` / 各类运算 / `model.forward` / `backward` 返回
`Result<T>`；`*_inplace` / `write` 返回 `Result<void>`。把字符串解析成数值用
`nn::parse_number<T>(s)`（返回 `Result<T>`）。

---

## 18. 写自己的层 / 加新原语（去哪读）

- **写自定义层**（`Layer` 子类：`init_impl` NVI、参数注册、精度槽声明、缓存契约、
  GPU checklist、梯度自检）→ [`write-a-custom-layer.md`](write-a-custom-layer.md)。
- **给引擎加新原语**（CPU / GPU 两后端实现、shader、AOT 闭合世界注册）→
  [`01-compute-engine-development.md`](01-compute-engine-development.md)。
- **多精度语义契约（写层前必读）** → [`05-mixed-precision.md`](05-mixed-precision.md)。
- **线性注意力家族算法** → [`06-rapt-algorithm.md`](06-rapt-algorithm.md)。

---

## 19. 工具与周边（CLI 辅助 / 底层工具）

### 19.1 CLI 辅助（`nn::cli`，`include/neuralnet.cpp/cli/`）

写自己的可执行文件时复用这套，与仓库自带 CLI 行为逐字一致：

```cpp
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>   // EngineConfig / create_engine
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>       // parse_gpu_option
#include <neuralnet.cpp/cli/cli_train_common.hpp>     // TrainCommonArgs / parse_train_common_args
#include <neuralnet.cpp/cli/cli_lr_scheduler.hpp>     // compute_epoch_lr / compute_step_lr
#include <neuralnet.cpp/cli/cli_mnist_io.hpp>         // load_mnist_csv / evaluate_mnist
#include <neuralnet.cpp/cli/cli_help.hpp>             // Help：统一的 --help 排版器
```

| 名字 | 作用 |
|---|---|
| `nn::cli::EngineConfig` / `nn::cli::create_engine(cfg, log)` | 引擎选择（`--gpu` 语义；显式请求失败不回退 CPU） |
| `nn::cli::parse_gpu_option(argc, argv, i, allow_name_value)` | 解析 `--gpu` / `--gpu=<设备>`（返回设备选择子或空） |
| `nn::cli::TrainCommonArgs` / `nn::cli::parse_train_common_args(argc, argv, i, cfg)` | 解析 `--epochs` / `--lr` / `--batch-size` / `--optimizer` / `--weight-decay` / `--lr-schedule` / `--warmup-epochs` / `--min-lr` / `--lr-per-epoch` / `--gpu` |
| `nn::cli::LrScheduleConfig` / `nn::cli::compute_epoch_lr(cfg, epoch)` | epoch 级调度（手动 lr 表 > cosine > 常数） |
| `nn::cli::StepLrScheduleConfig` / `nn::cli::compute_step_lr(cfg, step)` | step 级 warmup + 余弦 |
| `nn::cli::load_mnist_csv(path, max_samples)` | 读 MNIST CSV → `(feat (784,N), one-hot 标签 (10,N))` |
| `nn::cli::evaluate_mnist(model, engine, x, y_onehot, eval_samples, eval_batch)` | 全量前向 → argmax 准确率 |
| `nn::cli::Help` | 各 CLI 共用的帮助版式（宽度感知对齐；`usage/section/note/text/item/opt` 链式填表） |

### 19.2 底层工具

```cpp
nn::load_text_file(path);          // 读整个文本文件（Result<std::string>）
nn::sha256_hex(data);              // SHA-256 十六进制摘要（数据集 source.sha256 用它）
nn::parse_number<int>("123");      // 字符串 → 数值（Result<T>）
nn::KeyValueRecord rec;            // 自描述键值记录（模型 spec 头 / .nnvocab 的容器）
rec.set("key", uint64_t{1});
rec.set("name", std::string("x"));
uint64_t v = 0;
if (rec.get("key", v)) { /* ... */ }
```

并行：`nn::global_thread_pool()` 返回全局 `nn::ThreadPool`，`parallel_for_each(first, last, f)` /
`parallel_for_each_indexed(...)` 按安全切分点并行（结果与顺序执行逐字节一致，铁律 #8）。
多数场景不需要直接用它——引擎原语与分词器的并行都在内部完成。

---

## 20. 已废弃 / 不存在的 API（避免照着旧代码写）

- `Tensor::reshape()` / `Tensor::cpu_matrix()` / `Tensor::from_matrix()` —— **已私有**，
  走 `engine.reshape` / `engine.to_matrix` / `engine.from_matrix`（铁律 #11）。
- `nn::Layer::sub_layers()` —— 已删除（层树枚举没有消费者）。
- 带 engine 形参的旧 `Layer::forward(engine, input)` —— 现为 `forward(input)`，
  engine 在 `init` 时绑定。
- `ensure_gpu` —— 不存在（改名即防线），跨设备统一走 `engine.import`。
- CLI `--export-deploy` 与 ZiPT(`AttnZip`) 相关选项 / `nn::*ZiPT*` —— 已随库级清理移除。
- **不支持 CUDA**：后端只有 CPU / Vulkan，无 `--cuda`。
