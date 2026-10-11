# 模型格式 v7（单格式多精度 + 参数形态认定）与引擎能力内敛（修复设计 · 讨论稿）

> **状态：讨论稿（未实施）**。本文件只记录**问题评估**与**设计草案**，等设计确认后再落地。
> **触发**：2026-10-11 用户裁决的硬约束与四项要求（原文见 §0）。
> **关联**：`docs/development/21-quantized-weights.md` §4.15/§4.16（被本稿取代的部分）、
> `docs/history.md`（P2 部署格式那条记录）、`include/neuralnet.cpp/ternary_deploy.hpp`（拟删除）。

---

## 0. 用户裁决（约束与要求）

**三条硬约束（后续所有工作必须遵守）**

1. 不以任何形式绕过项目既定的设计理论/架构。
2. 不用特例方式解决问题（"这就是为什么要做 FoldAttn 而不是直接做 FlashAttention，即便工作量增加"）。
3. 复杂度必须控制在可控范围内。

**四项具体要求**

1. NNNN 模型格式**本身**要支持多精度存储，而不是新建文件格式。
2. 序列化/反序列化要能**识别**该字段并据此读取。
3. 取消 `text_train` 等处的"部署标签"，**统一 NNNN 格式**。
4. **引擎能力查询严格限制在引擎内部使用**，不得出现在任何 Layer 层 / 计算调用层。

**用户的澄清**

- 复合层（为解决"多层激活函数"等而存在）**不属于特例** —— 用它承载结构/转发是正当的。
- 要求"全链路优化，不管是不是有效"（= 不做"因为预期收益小就不做"的裁剪）。

**第二轮裁决（2026-10-11，逐条）**

1. **kind 粒度 = 参数级**（采纳本稿建议）。
2. **只读态 = 由 kind 推断**；但**质疑"可训练/不可训练"这一状态本身**：模型文件的可训练性
   只有在"优化器状态（AdamW 动量）进模型"之后才可能成为问题，**现在不应引入该状态**。
   → 本稿据此**删除**模型级 `trainable()`／层级"只读态"标志，只保留"结构性事实 + 用点报错"
   （见 §2.2.1）。
3. **引擎能力查询**：收 private 并按 `supports_*_` 命名（**尾随下划线 = private 约定**）；
   **不得出现在基类 `ComputeEngine`**；不得作为"`matmul_q` 支持与否"的标签；只作为
   **引擎内部分发计算路径**的标签；设备不支持时在引擎内**降级或报错**（见 §2.3）。

---

## 1. 问题评估（现状证据，全部可复现）

### 1.1 【主要】另起 `"NNQT"` 文件格式，绕开了既定设计（违反约束 1、要求 1/3）

**设计侧早已定的东西**（不是事后附会）：

- `docs/development/21-quantized-weights.md` §4.1 命名表 **D4**：`Precision::T1_58` 的
  "**序列化 tag 4**"是裁决项；§4.9 步骤 1 把"`precision_tag`/`precision_from_tag` 追加
  `4 = t1_58`"列入实施清单。
- `include/neuralnet.cpp/precision.hpp` 的 tag 表注释原文：
  `// 0=f32、1=f64（占位）、2=f16、3=bf16（保留）、4=t1_58（P2 打包权重用）`。
- 模型格式本身就是"自描述 + per-tensor 精度 tag"（v5 起，v6 = 内嵌词表换 `.nnvocab` 镜像）：
  每个张量条目 = `[precision tag u8][rows u64][cols u64][data]`。

**实际做法（本轮）**：新建 `include/neuralnet.cpp/ternary_deploy.hpp`，自定 magic `"NNQT"` +
自定版本号 + 整套并行读写器与入口（`is_deployment_file` / `peek_deployment_spec` /
`peek_any_model_spec` / `save_deployment` / `load_deployment` / `load_any_model`），
并在自建格式里**重新定义**了一套 tag 语义（虽然数值借用了 0/2/4）。

**后果（可复现）**

| 后果 | 证据 |
|---|---|
| 同一份权重有两套落盘/读取实现 | `model_serialization.hpp`（NNNN）与 `ternary_deploy.hpp`（NNQT）并存 |
| 两类文件**不可互读** | 既有读器 `read_matrix_v5` 对未知 tag 直接报 `Unsupported precision tag in v5: 4` |
| 类型层预留给打包权重的 tag 4 被闲置 | `precision_tag(T1_58) == 4` 已存在，但 NNNN 读写路径从不使用它 |
| 多出一套版本号 | `DEPLOY_VERSION = 1` 与 `MODEL_VERSION = 6` 并行 |

### 1.2 "部署标签"扩散到 CLI/GUI（违反要求 3）

- `src/text_train.cpp`：新增 `--export-deploy <path>`（含 help 文案与"非三值模型显式拒绝"分支）。
- `gui.py`：新增"导出部署文件 (可选)"输入行；`cli_controllers.py`：`export_deploy` 透传。
- `bench/gui_cli_audit.py`：为让审计通过，往 `CASES` 里补了一条 `export_deploy="d.nnqt"`。
- `src/text_infer.cpp` / `src/mnist_infer.cpp`：各自写一遍 `is_deployment_file` 分支 + 版本号
  伪装（部署文件按 `model_version = 6` 处理词表）。

这些都不是"格式能力"，而是"一种文件类型的标签" —— 正是要求 3 要取消的东西。

### 1.3 BitLinear 的特例 API（违反约束 2）—— 逐条清单

现状 `BitLinear` 的公开面（`compute_layer_bitlinear.hpp`）分三类：**规范接口**（Layer 要求）、
**诊断**（只读派生视图）、**本会话新增的特例**。逐条：

| API（行号） | 用途 | 谁调用 | 判定 |
|---|---|---|---|
| `set_ternary_pack_enabled(bool)`（138） | 层上强制"打包/稠密"缓冲形态（A/B、回退） | 测试、`write_epoch_test`、部署导出 | **特例**：把"引擎能力/实现选择"做成层开关 |
| `set_ternary_pack_auto()`（149） | 恢复"跟随引擎能力" | 无（保留） | **特例**：同上 |
| `uses_packed_weights()`（186） | 当前缓冲形态 | 测试、部署导出、`test_shapes` | 特例（形态单一化后无意义） |
| `quantized_weights_packed()`（187） | 打包容器视图 | 部署导出/载入、测试 | 半特例：序列化直接抓内部派生缓冲 |
| `row_scales()`（183） | τ 视图 | 部署导出、测试 | 同上（且 τ 是**权重派生物**，见 §2.1 的存储设计） |
| `quantized_weights()`（182） | 稠密 f16 三值缓冲视图 | 测试 | 形态单一化后应删 |
| `latent_weights()`（181） | latent f32 视图 | 测试、`write_epoch_test` | 保留（训练语义需要） |
| `quantize_calls()`（192） | 量化次数（缓存生效证据） | 测试 | 保留（纯诊断，无能力语义） |
| **`load_packed_weights(words, tau)`（198）** | 把层切成"只有打包字 + τ"的只读态 | 部署载入（**经 `dynamic_cast<BitLinear*>`**） | **特例**：载入协议做成了具体层的方法 |
| **`packed_only()`（226）** | 只读态查询 | 部署载入、测试 | **特例**：把"模型只读"塞进单层状态位 |
| **`ensure_quantized()`（232）** | 导出前确保派生物同步 | 部署导出 | **特例**：导出协议做成了具体层的方法 |
| `parameters()`（160）/`param_gradients()`（167）在 packed-only 下 **fail-fast**（abort） | 防止"以为在训练" | 训练入口/优化器 | **特例**：把"模型只读"的状态问题转嫁给调用方并升级成 abort |
| `in_features()/out_features()`（227/228） | 形状查询 | 部署导出/载入 | 无害，但应经统一的参数条目协议表达 |

**共性**：这些都是"**某个具体层的存储/载入协议**"，而项目既有的做法是"**复合层转发 + 叶子层实现
统一接口**"（`parameters()` / `param_gradients()` / `activation_cache()` / `extra_state()` /
`set_doc_ids()` 全是这个形状）。所以修法不是"再讨论要不要保留"，而是**把打包形态的存取
升级成 Layer 的统一参数条目协议**（§2.2）。

### 1.4 引擎能力查询外泄到 L2（违反要求 4）

**现状**：`compute_layer_bitlinear.hpp` 有 **5 处** `engine_ref().supports_ternary_gemm()`：

| 行 | 位置 | 用途 |
|---|---|---|
| 127 | `init_impl` | 选打包/稠密缓冲形态 |
| 141 | `set_ternary_pack_enabled` | 强制开关的合法化 |
| 152 | `set_ternary_pack_auto` | 恢复自动 |
| 210 | `load_packed_weights` | 部署态前置校验 |
| 265 / 354 | `forward` / `sync_buffer_mode_` | 部署态/形态一致性校验 |

**对照：既有设计本来是对的**。其它三个能力查询都**只**在引擎内部使用（可复现 grep）：

- `supports_native_data_move()`：只在 `compute_engine.hpp` 的 `fill_zero/copy_from/add_inplace/
  accumulate/...` 等 NVI 入口内部使用；
- `supports_native_f16_reduce()`：同上（归约族 NVI 内部）；
- `supports_expr_precision_variant(...)`：只在 `compute_gpu_engine.hpp` 内部自查。

也就是说：**"能力查询 = 引擎内部实现细节"是本项目的既有模式**，`supports_ternary_gemm()`
是本轮（P2/F3）新引入的、唯一外泄到 L2 的一个。另外本会话新增的
`supports_int8_dot()` / `int8_dot_accelerated()` 也是 public（当前只被测试打印）。

### 1.5 复杂度（违反约束 3）

| 维度 | 现状 |
|---|---|
| GEMM 内核 | 4 套：分块 f32 / 分块 f16 / GEMV f32 / GEMV f16（+ 打包 GEMV） |
| BitLinear 缓冲形态 | 2 套（打包容器 / f16 稠密），互相切换要 `quant_valid_ = false` |
| 文件格式 | 2 套（NNNN v6 / NNQT v1） |
| 参数存取 API | 2 套（`parameters()` 路径 + `load_packed_weights/ensure_quantized` 路径） |
| 选址逻辑 | 2 处（GPU 后端：GEMV vs 分块阈值；`BitLinear`：打包 vs 稠密） |

### 1.6 L3 依赖 L2 具体层

`ternary_deploy.hpp`（序列化层）里出现 `dynamic_cast<BitLinear*>` —— 与"上层只依赖下层公有
接口"的既有纪律不符（`model_serialization.hpp` 自身只走 `Layer` 接口）。

---

## 2. 设计草案

### 2.1 模型格式 v7：**单格式**、参数级形态认定、多精度

**目标**：一份格式同时表达"这个参数是什么（Linear/BitLinear/后期扩展）"与"它是什么精度/
什么存储形态"，读写双端据此分派；不再有第二种文件类型、不再有 CLI 标签。

**条目布局（v7 草案）**

```
[header]  magic "NNNN"(4B) | version=7(4B) | Scalar precision(1B) | spec_len(8B) | spec kvrec
[每个参数条目]
    [kind u8][precision u8][rows u64][cols u64][aux_len u64][payload_len u64]
    [aux bytes ...][payload bytes ...]
[extra state 条目]  同一套条目布局
[tokenizer]  同 v6（len u64 + .nnvocab 镜像字节）
```

**字段语义（草案取值）**

| kind | 含义 | precision | aux | payload |
|---|---|---|---|---|
| `0 Dense` | 普通浮点参数（Linear 权重/偏置、norm 参数、嵌入、BatchNorm 统计…） | f32/f16 | 空 | `rows×cols` 元素 |
| `1 TernaryLatent` | BitLinear 的 **f32 latent**（**可续训**） | f32 | 空 | `rows×cols` f32 |
| `2 TernaryPacked2` | BitLinear 的三值打包（2 bit/权重） | n/a | τ（rows × f32） | `rows × ceil(K/16) × 4B` |
| `3 TernaryPacked3` | 同上，base-3 载荷（1.6 bpw） | n/a | τ（rows × f32） | `rows × ceil(K/5) B` |
| `4…` | **预留**（下一档：`I8Packed4` = 4×int8/字，P3 用；以后 scale/zero-point 类量化） | — | 由 kind 定义 | 由 kind 定义 |

**为什么 kind 放在"参数级"而不是"层级"**

1. 与现有格式的"逐张量条目"同构（改动最小、最像"扩展"而不是"新格式"）。
2. 参数↔层 的映射本来就有唯一事实源（`Model::parameters()` 的顺序 + 复合层转发），不需要在
   文件里再抄一份层树。
3. 天然支持**异构**（同一模型里有的层 Linear、有的 BitLinear；不同层不同量化档）—— 这正是
   "其他后期扩展"要留的口子。

**"是 Linear 还是 BitLinear"的认定**：由 kind 直接给出（`Dense` ⇔ 普通浮点权重；
`Ternary*` ⇔ 三值权重）；载入时**双端校验**：

- 目标层期望的形态集合（Layer 侧声明，见 §2.2）与实际读到的 kind 不符 → fail-fast
  （含层名 + 期望集合 + 实际 kind，措辞沿用 `Layer::init` 的既有风格）；
- 这比今天"靠 spec 的全局 `weight_quant` + 参数顺序推断"更硬：文件损坏/串味会当场暴露。

**只读态怎么表达**：**不表达**。见 §2.2.1 —— 只有"某个参数是打包形态（没有 latent）"
这一**数据结构事实**，以及它在**用点**（`backward()`）自然产生的错误；**不引入**任何
"可训练/不可训练"的状态位、标志或模型级 API。

**与我已实现内容的复用**：base-3 编解码（`ternary_pack.hpp` 的 `pack_row_base3`/
`unpack_row_base3`）、打包 GEMM 内核、GEMV 内核、`BitLinear` 的量化/缓存逻辑**全部保留**；
被替换的只有"落盘/载入协议"。

### 2.2 参数条目协议：替代 `dynamic_cast` + `load_packed_weights`

**Layer 侧新增 2 个虚函数（叶子实现、复合层按 `parameters()` 顺序转发）**：

```cpp
// 导出：第 i 个参数的**存储形态**与数据（序列化用；默认 Dense + tensor 视图）
virtual Result<ParamEntry> export_param(std::size_t index);
// 导入：按文件里的形态接管第 i 个参数（默认：Dense → copy_from）
virtual Result<void> import_param(std::size_t index, const ParamEntry& entry);
```

`ParamEntry` = { kind, precision, rows, cols, aux(Tensor), payload(Tensor) }（`Tensor` 而非
`Matrix`，保持 L2 铁律；宿主下载/上传由序列化层负责）。

**这套协议取代的东西**：

- 取代 `load_packed_weights` / `ensure_quantized` / `quantized_weights_packed()` /
  `row_scales()` 的"对外特例"——它们降级为 `export_param/import_param` 的内部实现；
- 取代 §1.6 的 `dynamic_cast<BitLinear*>`：序列化层只与 `Layer` 的协议打交道；
- **取代 `Layer::sub_layers()`**（本会话新增）：有了参数级协议，就不需要"层树遍历器 + 覆盖数
  自检 + 层类型 cast"这套机制。复合层继续被使用（用户认可它是正当概念），只是它承载的是
  **参数转发**而不是**层枚举**，与 `parameters()` 的既有形状完全一致，复杂度更低。
  > 若你更希望保留 `sub_layers()` 作为正式架构能力（它对"结构遍历"类需求更通用），
  > 也可以两者并存；但按约束 3，我建议**只留参数条目协议**。

#### 2.2.1 为什么**不**需要"可训练 / 不可训练"状态（回应第二轮第 2 条）

**当时为什么会走到这一步**：打包形态的 `BitLinear` 结构上**没有 f32 latent** —— 而 STE 的
`dW += (dY·Xᵀ)∘τ` 要落到 latent、优化器也要有参数张量。我当时把这个事实**升级成了状态**
（层级 `packed_only()` + 模型级 `trainable()` + `parameters()` 里 abort），这是错的一步：
把"数据形态的事实"包装成了"模型策略"，于是到处都要先问一句"这个模型能不能训"。

**正确的形态**：只保留事实与用点错误，不引入任何状态。

| 层次 | 正确做法 | 反面（本稿删除的东西） |
|---|---|---|
| 文件 | 只记"该参数的**存储形态**"（参数级 kind） | 不记"可训练性"标志 |
| 层 | 打包形态的层没有 latent ⇒ `backward()` 返回**明确的 Result 错误**（与"必须先 forward"同类的**前置条件**检查，不是策略） | 不用 `packed_only()` 位、不在 `parameters()/param_gradients()` 里 abort |
| 模型 | 无 `trainable()` —— 模型没有"状态"，只有参数与结构 | 不聚合"只读"标志 |
| 训练入口 | 不需要提前检查；错误在真正需要 latent 的那一步报出（信息足够定位：层名 + "该文件只含打包权重"） | 不因"文件类型"提前拒绝 |

**为什么现在不做、以后可能要做的**：等优化器状态（AdamW 动量）也进模型文件时（roadmap
P1-2 的 checkpoint 项），"这个模型承载了哪些**训练态数据**"才第一次成为一个真实的模型级
概念 —— 那时再讨论它的表达；**不提前造**。

**可选（不是状态）**：载入器/CLI 可以在载入含打包形态的文件时**打一行日志**（"该文件只含
打包三值权重，无 latent → 不能继续训练"）作为提示；这只是日志，不改变任何 API 或行为。

### 2.3 引擎能力内敛（要求 4 + 第二轮第 3 条）

**规则（本稿拟写成 AGENTS 铁律）**

1. 能力谓词属于**具体引擎的私有实现细节**：命名 `supports_*_`（**尾随下划线 = private 约定**），
   **不允许任何外部调用**（Layer / 优化器 / CLI / 测试都不行）。
2. **不得出现在基类 `ComputeEngine`** —— 它不是引擎接口的一部分。
3. **不得作为"某原语支持与否"的对外标签**（典型反例：`supports_ternary_gemm()` 被 BitLinear
   用来决定缓冲形态）。能力谓词**只能**用于**引擎内部分发计算路径**。
4. 设备不支持某条路径时，由引擎内部**降级**（走等效实现）或**报错**（明确的 `Result` 错误），
   调用方只看到"成功 / 明确错误"，看不到能力位。

**本轮新增位的处置**

- **删除 `ComputeEngine::supports_ternary_gemm()`**：`BitLinear` 不查能力、缓冲形态单一化
  （只有打包容器）；`GpuEngine::matmul_q_impl` 内部按私有 `supports_ternary_gemm_()` 分发：
  有原生打包内核 → 用；没有 → **引擎内降级**（例如引擎内解量化后走既有 matmul，或直接返回
  明确错误）。这条降级路径必须**有测试**（本机 CPU-only 构建天然覆盖"无打包 pipeline"）。
- `supports_int8_dot()` / `int8_dot_accelerated()` → 私有化为 `GpuEngine::supports_int8_dot_()`
  / `int8_dot_accelerated_()`；P3 的 int8 内核只由 `GpuEngine` 内部选址使用。
- A/B 逃生阀从层 API（`set_ternary_pack_enabled`）改为**引擎级环境变量**
  （`NN_VULKAN_NO_TERNARY_GEMM` 一类，初始化期读一次并缓存，与既有 `NN_VULKAN_NO_16BIT_*` 同风格）。

**既有三处（`supports_native_data_move` / `supports_native_f16_reduce` /
`supports_expr_precision_variant`）的两种口径**

| 口径 | 范围 | 代价 / 风险 |
|---|---|---|
| **A｜窄口径（本轮做）** | 只处理"新引入且被 L2 使用"的位（ternary / int8）。既有三处保持现状 —— 它们**已经只被基类内部使用**（不违反"不得出现在 Layer/计算调用层"），把"是否也移出基类"单独立项 | 小：一处删除 + 一处分发下沉 + 逃生阀搬家 |
| **B｜严格口径** | 三处也从基类移出 → 等于把**"f16 精度边界 cast 层"从基类下沉为具体引擎复用件**：基类 NVI 只调 `*_impl`，cast/直通的分支由各引擎在 impl 内用自己的私有 `supports_*_()` 决定（`CpuEngine` 的 impl 已自带精度归一化；`GpuEngine` 需把约 15 个入口的 cast 包装搬进去） | 中-大：触及**精度体系核心**，必须逐入口迁移 + 每步跑精度族测试（`precision_test` / `f16_precision_test` / `gpu_f16_test` / `precision_profile_test` …） |

**建议**：A 立即做；B **单独立项**（把核心精度层与"三值/格式修复"混在一起会失控 —— 约束 3）。
B 是否本轮一并做，列为待裁决第 1 条。

**能力信息的对外披露**：收 private 后，能力信息只能走**探针通道**（`NN_ENABLE_PROBES` 门下的
`mem_probe` 一类目标）或**日志**，不回到任何 public API。测试改为**行为断言**，例如
"在不支持打包的设备上 `matmul_q` 仍给出与 f16 路径一致的数值"（这正是降级路径的验收），
而不是断言能力位。

### 2.4 删除/保留清单

**删除**

- `include/neuralnet.cpp/ternary_deploy.hpp`（整文件）、`DEPLOY_MAGIC` / `DEPLOY_VERSION`、
  `is_deployment_file` / `peek_deployment_spec` / `peek_any_model_spec` / `save_deployment` /
  `load_deployment` / `load_any_model`。
- `src/text_train.cpp` 的 `--export-deploy`（含 help/解析/导出分支）；
  `gui.py` 的"导出部署文件"行；`cli_controllers.py` 的 `export_deploy` 透传；
  `bench/gui_cli_audit.py` 里为它加的那条用例。
- `src/text_infer.cpp` / `src/mnist_infer.cpp` 的 `is_deployment_file` 分支（回到
  `peek_model_spec` + `load_model` 单一路径）。
- `BitLinear` 的 `set_ternary_pack_enabled` / `set_ternary_pack_auto` / `uses_packed_weights`
  / `load_packed_weights` / `packed_only` / `ensure_quantized` / `quantized_weights()`
  / `quantized_weights_packed()` / `row_scales()`（降级为协议内部实现）；
  **`parameters()` / `param_gradients()` 在 packed 形态下的 abort 一并删除**（改为
  `backward()` 的明确 Result 错误，见 §2.2.1）。
- `Layer::sub_layers()`（被参数条目协议取代；若裁决保留则改为正式能力并补契约与审计）。
- `ComputeEngine`（基类）的 `supports_ternary_gemm()` / `supports_int8_dot()` /
  `int8_dot_accelerated()`：**从基类删除**（不是收 protected —— 能力谓词不得进入基类），
  改为具体引擎私有 `supports_*_()`；`supports_native_*` / `supports_expr_precision_variant`
  的处置见 §2.3 的口径 A/B。
- 本稿草案里出现过的"模型级 `trainable()`"：**不再引入**（§2.2.1）。
- `src/ternary_deploy_test.cpp`（并入 `model_serialization`/`ternary_gemm` 的测试面）。

**保留**

- `ternary_pack.hpp`（2-bit + base-3 编解码，含往返测试）。
- `matmul_q` / `matmul_q_t` 原语与全部打包 GEMM 内核（分块 + GEMV）、CPU 行块解码。
- `BitLinear` 的量化/STE/写版本缓存、`AdamWQuant`、P3 的设备特性探测（改为内部）。
- 训练/解码/显存 A/B 的实测口径（`bench/run_with_vram.ps1` 的整卡峰值采样）。

### 2.5 迁移步骤（可独立验收，逐步替换而不是一次性大爆炸）

1. **引擎内敛（口径 A）**（独立、低风险）：从基类删除本轮新增的三个能力位；`BitLinear` 去掉
   5 处能力查询、缓冲形态单一化；`matmul_q` 在 GPU 引擎内部按私有 `supports_ternary_gemm_()`
   分发（原生内核 / 引擎内降级 / 明确错误）；A/B 改引擎环境变量。验收：ctest 全绿 +
   `layer_bench` 配对 A/B 与今天同值 + **新增"无打包内核 → 降级"的行为用例** +
   本机 CPU-only 构建行为不变。
2. **参数条目协议**（Layer 侧）：新增 `export_param/import_param`（**不带**任何只读态/Trainable
   标志）；复合层按 `parameters()` 顺序转发；`BitLinear` 实现三值三形态；删除 `sub_layers`
   （待裁决）与全部特例 API；`backward()` 在无 latent 时返回明确错误。验收：迁移序列化测试 +
   "packed 载入后 backward 报错"的用例。
3. **v7 读写**：`save_model`/`load_model` 支持 v7 条目（kind + 精度 + aux/payload），
   v5/v6 仍可读（写一律 v7）；删除 `ternary_deploy.hpp` 与 CLI/GUI 标签。验收：
   "训练检查点往返逐位一致" + "packed 载入前向与原模型逐位一致（沿用今天 11.9×/逐字节等价的
   那组实测）" + v6 老文件可读。
4. **产出 packed 文件的入口**（重开裁决，见 §3.3）：**候选 A = 模型级"冻结"动作被否决**
   （那就是被否掉的"模型状态"）。剩下：候选 B = `save_model` 的一个**显式存储形态参数**
   （调用点选择，不是模型状态、不是文件类型标签）；候选 C = 只经库 API 导入形态（把打包
   `ParamEntry` 交给 `import_param`），CLI 不提供产出入口。
5. **文档同步**：删掉 21 §4.15 的"独立格式"表述，改写为"NNNN v7 的多精度与形态认定"；
   history 记一条"修复：把 v1 独立格式并入 NNNN v7"；AGENTS 增一条铁律
   （能力查询只在引擎内部）+ 参数条目协议契约。

### 2.6 验收（沿用本项目既有口径）

- 本机（CPU-only）+ 102（含全部 GPU 用例）ctest 全绿；
- `doc_inventory`：L2-VIOLATIONS 0 / GATE PASS / virtual 计数回落（去掉 sub_layers 与能力位）；
- `doc_align_audit` E/F 节 0；`gui_cli_audit` PASS（且**不再有** `export-deploy` 这条用例）；
- 端到端：训练 60 步 2.0 s / 训练峰值 522 MiB / 解码 0.7 s / 推理峰值 76 MiB 不回退；
  packed 文件 11.9× 更小、载入后**逐字节等价**（沿用今天的实测方法）；
- v6 老文件（`data/dep_train.bin` 一类）仍可正常 `peek`/`load`。

---

## 3. 待裁决（第二轮剩余问题）

> 第一轮 6 条中已裁决：① kind 粒度 = **参数级**；② 只读态 = **不引入**（由 kind 推断 + 用点报错）；
> ④ 引擎能力查询 = **具体引擎私有 `supports_*_`、不得进基类、只作内部分发**。
> ⑤/⑥ 的下文给出我的推导与建议，请确认。

1. **既有三个能力位是否本轮一并移出基类**（§2.3 的 A/B 口径）？
   - A（我建议）：本轮只处理 ternary/int8；既有 `supports_native_*` /
     `supports_expr_precision_variant` 单独立项、逐入口迁移。
   - B：本轮一并把"f16 精度边界 cast 层"从基类下沉到具体引擎（触及精度核心，工作量与风险都在这里）。
2. **`sub_layers()` 去留**：因为 kind 已定为**参数级**、载入走**参数条目协议**，`sub_layers()`
   再无消费者（它的唯一用途是让序列化层递归找到层）。我建议**删除**；
   若你认为"层树枚举"作为通用架构能力该留，则需补全所有复合层实现 + 契约 + 审计（那是另一项工作）。
3. **产出 packed 文件的入口**：A（模型级 freeze）= 已被否决；剩下
   **B = `save_model` 的显式存储形态参数**（推荐）／**C = 只经库 API 导入形态、CLI 不提供**。
4. **能力信息披露通道**：探针目标（`NN_ENABLE_PROBES` 门内）／日志／两者。
   （测试一律改行为断言，不再断言能力位。）

## 4. 与 P3（全链路优化）的关系

- P3 的 int8 路线在本设计里**不另起容器特例**：激活 int8 作为 v7 的一档 kind
  （`I8Packed4`：4×int8/字 + 逐列 scale 走 aux）表达，序列化/载入与三值同一条协议。
- 引擎能力（`shaderIntegerDotProduct` + 硬件加速位）已在设备侧探测并启用（§21 §4.16），
  但按要求 4 它**只能在引擎内部**被使用：int8 内核的选址由引擎内部完成。
- 全链路（`quantize_act_i8` + `matmul_q_i8` + 端到端对拍）在设计确认后一并排入，
  不因"预期收益低于 BitNet 官方（无张量核）"而裁剪 —— 只按实测收益决定是否作为**默认**路径。

