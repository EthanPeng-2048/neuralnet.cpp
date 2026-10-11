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

**只读态怎么表达**：**由 kind 推断，不另设标志**。规则：某模型若含
`TernaryPacked*`（无 latent 的形态）→ 该模型为**只读推理态**；`--resume` / 训练入口在
建立模型后统一检查一次（§2.2 的模型级 `trainable()`），给出明确错误，而不是在层里 abort。

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

**只读态**：`Layer` 增加一个**只读**声明（例如 `virtual bool trainable() const noexcept`，
默认 true；BitLinear 在 packed 载入后返回 false）；`Model::trainable()` 聚合之。训练/优化器
入口统一按它 fail-fast；`parameters()`/`param_gradients()` 不再 abort。

### 2.3 引擎能力内敛（要求 4）

**规则（写进 AGENTS 铁律候选）**：能力查询只允许在引擎实现内部使用 —— 形式上全部收为
`protected`（编译期强制 L2 无法调用），并由引擎的 NVI 入口自行分派。既有三个
`supports_native_*` / `supports_expr_precision_variant` 一并收紧（现状已只被引擎内部使用，
收紧是"把既有约定变成编译期约束"）。

**BitLinear 侧**：不查任何能力，只有一种形态（打包容器），`forward` 只调
`engine.matmul_q(...)` / `matmul_q_t(...)`；"该设备有没有打包内核"由 **GPU 引擎内部**决定：

- 有打包 pipeline → 走打包 GEMM（现状）；
- 没有（或用户设了逃生阀）→ **引擎内部**用等效实现（例如引擎内解量化到 f16 稠密缓冲再走既有
  matmul），对 L2 完全透明。**这条回退在引擎内实现，意味着 `BitLinear` 不再需要 `wq_` 稠密
  缓冲、`use_packed_`/`force_pack_`/`sync_buffer_mode_`/`ensure_quant_buffer_` 的两形态逻辑**
  （§1.5 的"2 套缓冲形态"随之消失，复杂度下降）。
- A/B 逃生阀从层 API 改为**引擎级环境变量**（`NN_VULKAN_NO_TERNARY_GEMM` 等），与既有的
  `NN_VULKAN_NO_16BIT_STORAGE` / `NN_VULKAN_NO_16BIT_ALU` 同一风格（初始化期读一次并缓存）。
- `supports_int8_dot()` / `int8_dot_accelerated()`（P3 起步新增）同样收进内部；P3 的 int8 内核
  由引擎内部选址。

**能力信息的对外披露**（需要裁决，见 §3.4）：探针目标（`mem_probe` 一类，`NN_ENABLE_PROBES`
门下）或日志行，而不是 public API；测试改为**只测行为**（例如"在不支持打包的引擎上
`matmul_q` 仍给出与 f16 路径一致的数值"），不再断言能力位。

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
  / `quantized_weights_packed()` / `row_scales()`（降级为协议内部实现）。
- `Layer::sub_layers()`（被参数条目协议取代；若裁决保留则改为正式能力并补契约与审计）。
- `ComputeEngine` 的 public `supports_ternary_gemm()` / `supports_int8_dot()` /
  `int8_dot_accelerated()`（收为 protected；`supports_native_*` 一并收紧）。
- `src/ternary_deploy_test.cpp`（并入 `model_serialization`/`ternary_gemm` 的测试面）。

**保留**

- `ternary_pack.hpp`（2-bit + base-3 编解码，含往返测试）。
- `matmul_q` / `matmul_q_t` 原语与全部打包 GEMM 内核（分块 + GEMV）、CPU 行块解码。
- `BitLinear` 的量化/STE/写版本缓存、`AdamWQuant`、P3 的设备特性探测（改为内部）。
- 训练/解码/显存 A/B 的实测口径（`bench/run_with_vram.ps1` 的整卡峰值采样）。

### 2.5 迁移步骤（可独立验收，逐步替换而不是一次性大爆炸）

1. **引擎内敛**（独立、低风险）：把 `supports_*` 收 protected；`BitLinear` 去掉 5 处能力查询，
   缓冲形态单一化（引擎内部回退）；A/B 改引擎环境变量。验收：ctest 全绿 + `layer_bench`
   配对 A/B 与今天同值 + 无打包 pipeline 的构建（本机 CPU-only）行为不变。
2. **参数条目协议**（Layer 侧）：新增 `export_param/import_param` + `trainable()`；复合层转发；
   `BitLinear` 实现三值三形态；删除 `sub_layers`/特例 API。验收：新增/迁移序列化测试。
3. **v7 读写**：`save_model`/`load_model` 支持 v7 条目（kind + 精度 + aux/payload），
   v5/v6 仍可读（写一律 v7）；删除 `ternary_deploy.hpp` 与 CLI/GUI 标签。验收：
   "训练检查点往返逐位一致" + "packed 载入前向与原模型逐位一致（沿用今天 11.9×/逐字节等价的
   那组实测）" + v6 老文件可读。
4. **产出 packed 文件的动作**（需裁决，见 §3.3）：候选 A = 模型级"冻结/推理形态"动作
   （`Model::freeze_weights()`），之后 `save_model` 自然写 packed；候选 B = `save_model` 的一个
   存储形态参数；候选 C = 只由载入侧推断（今天这样）。我建议 **A**（"文件形态 = 模型形态"，
   不引入新的 CLI 标签，也符合"统一 NNNN"）。
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

## 3. 待裁决（多轮讨论的输入）

1. **kind 粒度与取值集合**：参数级（本稿建议）还是层级 + 参数级？`3 = bf16` 保留位是否继续
   不碰（本稿把 bf16 排除在 kind 之外，因为它不是本项目支持的存储精度）？
2. **只读态表达**：由 kind 推断（本稿建议）还是 spec 加一个模型级标志？
3. **产出 packed 文件的动作**：模型级 `freeze/推理形态`（本稿建议 A）／`save_model` 形态参数／
   只由载入推断（今天）？
4. **能力披露通道**：`supports_*` 全部收 protected 后，能力信息去哪（探针目标 / 日志 / 都要）？
   测试是否改为纯行为断言？
5. **`sub_layers()` 去留**：删掉（本稿建议，参数条目协议已覆盖需求）还是保留为正式架构能力
   （并补全复合层实现 + 契约 + 审计）？
6. **"打包/稠密"两条缓冲形态**：删掉层侧形态（本稿建议，回退下沉到引擎内部）还是保留？

---

## 4. 与 P3（全链路优化）的关系

- P3 的 int8 路线在本设计里**不另起容器特例**：激活 int8 作为 v7 的一档 kind
  （`I8Packed4`：4×int8/字 + 逐列 scale 走 aux）表达，序列化/载入与三值同一条协议。
- 引擎能力（`shaderIntegerDotProduct` + 硬件加速位）已在设备侧探测并启用（§21 §4.16），
  但按要求 4 它**只能在引擎内部**被使用：int8 内核的选址由引擎内部完成。
- 全链路（`quantize_act_i8` + `matmul_q_i8` + 端到端对拍）在设计确认后一并排入，
  不因"预期收益低于 BitNet 官方（无张量核）"而裁剪 —— 只按实测收益决定是否作为**默认**路径。

