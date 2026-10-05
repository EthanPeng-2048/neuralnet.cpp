# 自定义层易用性实测报告（库外使用者视角）

> **状态**：实测报告（2026-10-05）· 基准版本 v1.7.0（`f20532b`）· 环境：本机 CPU（clang++，Debug）
> **方法**：以"库外使用者"身份**只写新文件、不改库内任何代码**，从零实现一个自定义层
> 并把它训起来；记录全部编译失败、契约误解与"必须回读源码才能确定"的点。
> **证据**：`examples/custom_layer_lowrank.hpp`（层，157 行）+ `examples/custom_layer_train.cpp`
> （数值梯度检查 + MNIST 训练，277 行）+ `CMakeLists.txt` 的 `custom_layer_example` 目标。
> **关联**：`docs/development/18-roadmap.md` X6（本文登记的问题项）；X2/P2-4 是数据侧的同类"使用者视角"缺口。
> **写作纪律**：行号取自 2026-10-05 实测；坐标系为 `文件:行`，行号漂移后以符号名为准。

---

## 0. 一句话

**核心抽象成立**（一个头文件接入、训练循环与内置层零差异、数值梯度一次通过），
**但"写自定义层"这条路完全没有文档**——7 个摩擦点里 5 个的根因是"契约只存在于库内源码里"，
而不是设计缺陷。

---

## 1. 实测对象与结果

### 1.1 被测层

`LowRankAdapter`（LoRA 形态的低秩残差适配器，库内不存在此层）：

```
h   = A · x                    A: (rank, dim)
out = x + scale · (B · h)      B: (dim, rank)   B 初始化为 0 ⇒ 初始为恒等映射
```

选它的理由：**有可学习参数**（2 个张量）、**backward 需要 4 个转置 matmul**、
**需要 forward 缓存**，能同时压到参数注册、精度注入、缓存契约、DSL 折叠四条路径。

### 1.2 实测结果

| 项 | 结果 |
|---|---|
| 代码量 | 层 157 行（含注释，核心逻辑约 60 行）+ 训练/梯度检查 277 行 |
| 库内文件改动 | **0**（只新增 `examples/` 两个文件 + `CMakeLists.txt` 一个目标） |
| 编译通过轮次 | **第 3 轮**（第 1 轮 12 个错误，第 2 轮 1 个错误） |
| 数值梯度检查 | **一次通过**：`max|Δ| = 9.098e-04`（中心差分 ε=1e-3，容差 5e-3） |
| 训练 | 400 step / batch 64（Debug 构建 47.3s，118.2 ms/step）：loss 2.3802 → 0.1645 |
| 测试集准确率 | 90.60%（2000 样本；模型 784→Linear(128)→LowRankAdapter(128,16)→ReLU→Linear(10)） |

> Debug（`-O0`）下的 118 ms/step 不代表性能，仅证明路径正确。
> 复现：`cmake --build build --target custom_layer_example && ./build/custom_layer_example`

---

## 2. 问题清单

严重度口径：**高** = 会让使用者卡住、或写出"能编译但静默失效"的代码；**中** = 明显增加试错成本；
**低** = 体验问题。

| # | 问题 | 严重度 | 坐标 |
|---|---|---|---|
| 2.1 | `NN_CHECK` 名为 CHECK 实为 unwrap，丢弃返回值还需 `(void)` | 高 | `core_errors.hpp:201` |
| 2.2 | "造张量"存在两套失败协议（`Tensor`+`valid()` vs `Result<Tensor>`） | 高 | `compute_engine.hpp:298/308/335` |
| 2.3 | 无"写自定义层"文档，四条契约全靠读源码 | 高 | `docs/usage/` 无对应篇章 |
| 2.4 | 精度 profile 无自检（C7 类静默失效的成因） | 高 | `compute_layer_base.hpp:212` |
| 2.5 | 一个表达式只允许一个 matmul，且未写进使用文档 | 中 | `expr_spec.hpp:423` |
| 2.6 | 库外没有梯度检查工具 | 中 | `src/test_common.hpp:18` |
| 2.7 | GPU 路径是"额外一步 + 运行期才失败" | 中 | `CMakeLists.txt`（`nn_enable_gpu_fusion`） |

### 2.1 `NN_CHECK` 名为 CHECK 实为 unwrap（严重度：高）

**现象**：`NN_CHECK(x)` 的定义是"失败即 abort，成功返回**解包后的 `T`**"
（`core_errors.hpp:201`），不是断言宏。使用者按"断言"直觉写：

```cpp
auto x = NN_CHECK(engine.from_matrix(m));   // x 已经是 Tensor，不是 Result<Tensor>
x.rows();                                   // OK
*x;                                         // 编译错误（我实际写的形态）
```

一次触发 **12 个编译错误**，报错却指向 `core_errors.hpp` 内部
（`no matching function for call to 'check_value'`、
`member reference type 'nn::Tensor' is not a pointer`），而不是指向使用者写错的那一行。

**放大器一**：[`docs/usage/01-quickstart-model.md`](../usage/01-quickstart-model.md) 的
`NN_CHECK(model.add<nn::Linear>(784, 256));` 用的是 `Result<void>`，
两种语义（纯检查 / 检查+解值）在这里**看起来完全一样**，掩盖了差异。

**放大器二**：`Result<Tensor>` 的 `T` 带 `[[nodiscard]]`，丢弃返回值必须写
`(void)NN_CHECK(model.backward(grad));`——一个"断言宏"要求使用者 `(void)` 它。

**实测原始错误**（第 1 轮 12 条之一，第 2 轮复现）：

```
error: ignoring return value of function declared with 'nodiscard' attribute
    NN_CHECK(model.backward(grad));
```

### 2.2 "造张量"存在两套失败协议（严重度：高）

同一件事（拿到一个张量）有两套失败约定：

| 入口 | 签名（实测） | 失败如何表达 |
|---|---|---|
| `engine.create_tensor(r, c, P)` | `Tensor`（`compute_engine.hpp:298`） | 调用方查 `.valid()` |
| `engine.create_tensor(r, c, P, spec)` | `Tensor`（`:308`） | 调用方查 `.valid()` |
| `engine.from_matrix(m, P)` | `Result<Tensor>`（`:335`） | `Result` |
| `Layer::forward(input)` | `Result<Tensor>` | `Result` |

写自定义层时四个入口都要用（建权重、建梯度缓冲、上传输入、调 forward），
必须逐个记住哪个是哪个。把 `create_tensor` 包进 `NN_CHECK` 得到的错误是
`no known conversion from 'Tensor' to 'Result<void>'`——报错不告诉你正确写法是查 `valid()`。

**注**：`.valid()` 协议本身有明确理由（`:305` 注释：失败返回空 Tensor），
但该理由只在库内注释里，使用文档没有对应段落。

### 2.3 无"写自定义层"文档，四条契约全靠读源码（严重度：高）

`docs/usage/` 现有五篇（01 构建模型 / 02 训练推理 / 03 引擎用法 / 04 训练包 / 05 库内嵌消费），
**没有一篇讲"加一个自己的层"**。而写一个自定义层需要同时满足四条隐式契约：

1. **`init_impl` 而非 `init`**：`init` 是 NVI 公共入口（`compute_layer_base.hpp:212`），
   绑定 `engine_` 后转发到 `init_impl`。这个改名只在 `docs/release-notes/v1.5.0.md` 出现过一次。
2. **forward 缓存与 `checkpoint_mode_` 耦合**：`checkpoint_mode_ = true` 时 forward **不得**保留中间激活
   （`compute_layer_base.hpp:53`）。若照抄顺序写 `x_cache_ = input;` 而不加 `if (!checkpoint_mode_)`，
   编译通过、普通训练正常，**只有开梯度检查点才崩**。
3. **`activation_cache()` / `clear_cache()`**（`:232` / `:228`）：不实现不报错，
   只是 activation offload 与本层缓存释放**静默失效**。
4. **精度 profile 的用法约定**（见 2.4）。

这四条**没有一条能靠编译器兜底**：全部是"写了才对、不写也能过编译"的约定。

### 2.4 精度 profile 无自检（严重度：高）

`p_.param`（权重存储精度）与 `p_.compute`（表达式输出精度）靠使用者自觉：

```cpp
a_ = engine.create_tensor(rank_, dim_, p_.param, ...);   // 写对 → f16 下存储减半
a_ = engine.create_tensor(rank_, dim_, Precision::F32, ...);  // 写错 → 编译过、测试过、静默失去收益
```

这不是假想风险：**18 号路线图 C7 就是这个成因**
（`compute_layer_conv.hpp` 的 Conv2D 权重/偏置/梯度硬编码 `Precision::F32`，RAPT 32 处 `dsl::compute*`
未接精度实参）。本层的写法能对是因为**照抄了 Linear**，而不是因为有文档或检查。

**可实施的收口点**：`Layer::init`（`compute_layer_base.hpp:212`）是唯一的 NVI 卡口，
`init_impl` 返回后 `parameters()` 已可用，且此时还没跑 forward、成本为零。
在那里统一断言"`profile.param == F16` ⇒ `parameters()` 返回的每个张量存储精度为 F16"，
一处检查即可消灭 C7 这一整类静默失效。

### 2.5 一个表达式只允许一个 matmul，且未写进使用文档（严重度：中）

`ExprSpec::matmul` 是 `std::optional<MatmulSpec>`——**单个**（`expr_spec.hpp:423`）。
于是 `x + scale·(B·(A·x))` **不能**写成一个 `dsl::compute`，必须拆成两次：

```cpp
NN_TRY(h, dsl::compute(engine, dsl::matmul(a_, input, false, false), ...));      // ① A·x
return dsl::compute(engine,                                                      // ② + 残差尾链
    dsl::leaf(input) + dsl::rparam(scale_) * dsl::matmul(b_, *h, false, false), ...);
```

这个结论是从 IR 结构反推的，使用文档没有这句话。库内 `Linear` 恰好是"matmul + bias"（正好一个），
所以照抄能对；但任何想写"两层 matmul 融合"的人第一反应一定是写成一个表达式。
错误形态是**折叠失败**（而不是清晰的 constexpr 断言），自查成本高。

> 注：DSL 的 `dsl::matmul(A, B, transA, transB)` 四参转置语义本身**没有问题**——
> 本层的 4 处转置在数值梯度检查中一次全对（§1.2）。缺的只是"一个 spec 一个 matmul"这条约束的成文。

### 2.6 库外没有梯度检查工具（严重度：中）

`src/test_common.hpp:18-23` 明确写着"刻意不收编 `max_abs_diff` / `check_grad_tensor` / `eval_loss`"，
而库内散着 7 份 `max_abs_diff` 副本（18 号路线图 §5 已裁定"择机、非机械改写"）。

后果：**写自定义层的人必须自己实现数值梯度检查**（本样例写了约 70 行），
而 backward 的转置标志恰恰是最容易写错的地方。这是整条链路里**唯一"最容易错 × 最缺工具"**的组合。

另注：铁律 #8（gradcheck 必须先 forward 填充缓存）目前在 AGENTS.md §5，#8 条，
但**没有可复制的代码**——使用者只能自己从"缓存为空会崩"这个现象反推。

### 2.7 GPU 路径是"额外一步 + 运行期才失败"（严重度：中）

CPU 上写完即可跑；切 GPU 需要额外在 CMake 加 `nn_enable_gpu_fusion(<target>)`
（库外自定义层的表达式结构不在库内注册表里）。漏掉这一步时：

- **构建期不报错**；
- **运行期硬报错**（AOT 闭合世界，未命中即失败）。

这是设计上必然的（铁律 #7），本报告不建议改机制，只建议把"CPU 通过 → 加一行 CMake → GPU 通过"
写成使用文档里的检查清单。对使用者而言"CPU 对、GPU 报错"是最难自查的一类问题，
而现有的 `examples/fusion_custom_layer*` 样例讲的是**融合表达式**，不是**层**。

---

## 3. 设计成立的部分（同等重要）

避免只记录抱怨：以下各点经实测确认可用，是本报告结论的另一半。

1. **零库内改动**：整个层是一个 157 行头文件，`model.add<nn_example::LowRankAdapter>(128, 16)`
   与内置层用法**逐字一致**。自定义层的接入成本确实只有"写一个头文件 + 一行 CMake"。
2. **表达式即数学**：`dsl::leaf(input) + dsl::rparam(scale_) * dsl::matmul(b_, *h, false, false)`
   ——没有 IR、没有注册、没有手工 kernel。`rparam` 天然满足"值不进 `expr_spec_key`"（S7 教训）。
3. **训练循环与内置层零差异**：本样例的训练循环是从 `src/mnist_train.cpp` 直接缩写的，
   没有为自定义层增加任何特殊分支。
4. **优化器接入 = 两个 `return`**：`parameters()` / `param_gradients()` 各返回一个列表即可，
   `create_optimizer` / `zero_grad` / `clip_grad_norm` 全部自动生效。
5. **多精度免费**：因为全程使用 `p_.param` / `p_.compute`，本层在 `--f16` 配置下
   自动获得 f16 存储 + f32 计算，无需第二份代码（前提是 2.4 的自觉性）。
6. **数值梯度一次通过**：4 处转置 matmul、残差恒等路径、累加式梯度写入，全部一次对。
   这说明 DSL 的形状推导与视图语义对使用者是**自洽**的——原本以为会在这里翻车。

---

## 4. 建议的修复项

按"单位成本收益"排序。全部为**增量**改动，无破坏性 API 变更。

| # | 动作 | 成本 | 消解的问题 |
|---|---|---|---|
| R1 | 新增 `docs/usage/06-write-a-custom-layer.md`：一页模板（层骨架 + gradcheck 片段 + 四条契约 + GPU checklist） | 小 | 2.1 / 2.2 / 2.3 / 2.5 / 2.7 |
| R2 | `Layer::init` NVI 处加精度自检（`profile.param == F16` ⇒ `parameters()` 全为 F16 存储） | 小 | 2.4（同时为 C7 提供自动门禁） |
| R3 | 提供公共梯度检查工具（或至少在 R1 的文档里给可复制的 60 行） | 小–中 | 2.6 |
| R4 | `quickstart` 首屏加一句"`NN_CHECK` 是 unwrap 不是断言"；可选新增 `NN_UNWRAP` 别名 | 小 | 2.1 |
| R5 | 把 `examples/custom_layer_*` 挂进 README 与使用文档索引（现有 `examples/` 只有融合样例） | 小 | 2.3 / 2.7 |

**不建议做的**：

- 不建议为 2.2 引入统一协议（`create_tensor` 改返回 `Result`）——`.valid()` 是既有契约，
  改动会波及全部 Layer 与测试，收益只是"少记一个例外"；文档写清楚即可。
- 不建议为 2.5 放宽 IR（多 matmul 融合）——这是 AOT 闭合世界的结构性约束，
  触发条件应按 18 号路线图 §6 的 IR-C/S6 判据走，不因"写法不够顺手"立项。

---

## 5. 变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-05 | 本文建立：库外使用者视角的自定义层实测，7 个问题项 + 5 条修复建议 | `examples/custom_layer_*` 实测（v1.7.0 / `f20532b`） |
