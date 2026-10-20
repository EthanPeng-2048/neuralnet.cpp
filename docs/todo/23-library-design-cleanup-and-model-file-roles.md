# 库级违规设计清理：模型文件角色（`.nnmodel` / `.nnckpt`）· 参数形态协议 · 引擎能力体系

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history.md`](../history/history.md)。

> **状态：已实施（2026-10-11）**。三轮裁决已汇总于 §0；本文件记录**全库审查取证**与**设计**，
> 以及**落地进度**（§2.5 各步带 ✅ / ⏳ 标记）。**范围 = 整个库的违规设计清理**（不只三值路径；三值只是触发点）。
> **取代**：本文件替代早先的 `23-model-format-v7-and-engine-capability.md`（那版的"NNQT 独立
> 格式 + 部署标签"已被裁决取消）。

> **增补（2026-10-16，2026-10-18 收口）**：§2.1.1 = P1-2 的**训练状态段（v8）**与 role 语义（**已实施**）——
> 状态段是**唯一权威**（A/B 类参数禁止再由 CLI 给出）、数据集身份硬错、`.nnmodel` 恢复放开以支持 SFT。
> 裁定正文与验收口径在 [`18-roadmap.md`](18-roadmap.md) §4.2 P1-2 行 / §3.4 M6 / §8.2 门禁 7。

---

## 0. 裁决汇总（三轮）

**第一轮**：四条要求（NNNN 自身支持多精度存储 / 序列化识别该字段 / 取消部署标签统一格式 /
能力查询只在引擎内部）+ 澄清（复合层不是特例；"全链路优化，不管是不是有效"）
**第二轮**：① kind 粒度 = **参数级**；② 只读态 = **不引入**（不造"可训练/不可训练"状态）；
③ 能力谓词 = **具体引擎私有 `supports_*_`、不得进基类、不作对外标签、只作内部分发**；
④（本轮明确）能力**信息**走**统一对外查询通道**。
**第三轮（本稿依据）**：

1. 既有能力谓词**全部移出基类**（不是只在基类里收 protected）。
2. `Layer::sub_layers()` **不保留**（删除）。
3. "训练需要、推理不需要的参数"= **独立的文件角色**：**`.nnmodel`**（`.bin` 的接替者）与
   **`.nnckpt`**（含训练参数的模型文件）；`--save-interval` 保存的就是 `.nnckpt`。
4. 能力信息 = **统一查询通道**，通过对外接口实现（见 §2.3）。
5. 审查热路径上的能力判断（浪费计算 / 分支预测），评估**用编译期 NATIVE 标签默认选路**以
   消除热路径判断（见 §1.3 的取证与 §2.3 的政策）。
6. 本文件**不针对 t158**，而是**全库违规设计的清理**。

---

## 1. 全库审查取证（可复现）

### 1.1 文件格式：一个模型文件同时承担两种角色（违规）

- 现状：所有模型都写同一个 `*.bin`（NNNN v5/v6），**同一个文件既当训练断点又当推理模型** ——
  "哪些参数是训练才需要的"这件事只能靠 `ModelSpec.weight_quant` + 参数顺序**隐式**推断。
- 本轮新增的 `"NNQT"`（`ternary_deploy.hpp`）是**另一个格式**（新 magic + 新版本 + 并行读写器 +
  并行入口），而类型层的 `precision_tag(T1_58) == 4`（"P2 打包权重用"）预留位**从未被使用**。
- 两个后果都在生产路径上：`text_infer`/`mnist_infer` 要各自分流两类文件；`text_train --resume`
  拿到部署文件时的行为**未定义**（今天靠 magic 不符报错）。

### 1.2 能力查询：进了基类、外泄到 L2/L5、面散

| 位置 | 调用点数 | 判定 |
|---|---|---|
| **基类 `compute_engine.hpp` 的 NVI 体内** | `supports_native_data_move()` **11**、`supports_native_f16_reduce()` **5**、`supports_expr_precision_variant()` **3**、`supports_ternary_gemm()` **4** | **违规**：能力谓词成了引擎接口的一部分，且是 `virtual`（每次原语一次 vtable 调用） |
| `compute_layer_bitlinear.hpp`（L2 层） | 5（`supports_ternary_gemm()`） | **违规**：能力查询出现在计算调用层 |
| `ternary_deploy.hpp`（L3 序列化） | 2 | 违规（连同该文件一起删） |
| `src/layer_bench.cpp`（L5） | 1 | 违规（bench 也按能力位选路/报错） |
| 测试：`ternary_gemm_test` 4 / `ternary_deploy_test` 2 / `write_epoch_test` 1 | 7 | 违规（能力位当"跳过标签"用） |
| `compute_gpu_engine.hpp` / `compute_cpu_engine.hpp` 的**实现体内** | 若干 | **合规**：这正是唯一允许的位置 |

### 1.3 热路径专项（裁决第 5 条）—— 取证

**F1｜`virtual` 能力查询在每次原语调用上**
基类 19 处调用（11 + 5 + 3）分布在 `clone / slice_rows / scatter_add_rows / zero /
matmul / matmul_with_bias / add_inplace / accumulate / row_reduce_sum / col_reduce_max / …`
等 NVI 入口 —— 每个入口每次调用都要过一次 **vtable 间接 + 分支**。这些入口在训练里是
**每步每层多次**（`zero_grad` 一步就 N 次 `zero`）。

**F2｜`eval_expr` 系：查询本身在做实质工作，而且和实现重复做一遍**（最严重）
- 基类 NVI：`compute_engine.hpp:1382 / 1408 / 1441` 三处 `supports_expr_precision_variant(spec, inputs, P)`
  （**每次表达式求值都调**，代码注释亦承认"本函数只是查询（NVI 入口每次都调）"）。
- GPU 实现（`compute_gpu_engine.hpp:1602`）每次做：`expr_prec_sig_of(inputs,P)` →
  `canonicalize_expr_spec(raw_spec)`（**拷贝整份 spec**）→ `expr_spec_key(canon)`（**构造
  `std::string`**）→ `find_prec_variant_`（查表）→ `backend_.has_fused_shader`（再查一次）。
- 随后 `eval_expr_impl`（`:1478`）**又**做一遍 key 构造与查表 → **每次求值两套归因 + 一次
  string 构造（堆分配）**。表达式求值在一步训练里是千次量级 → 这是可量化的热路径浪费。

**F3｜BitLinear 的每 forward 形态判定**
`sync_buffer_mode_()` 在**每次 forward** 进入（其中 5 处查询能力）；虽然 `want == use_packed_`
时提前返回，但判定本身仍在热路径上。

**F4｜不受影响的地方（已核对）**：GPU 后端的 `has_*_pipeline()` 是**成员查询**（init 期定值）；
池/批次类原语（`begin_batch/end_batch/flush_*`）没有能力判断；DSL 的 CPU 求值路径不在热路径上
查能力。

**结论（回答裁决第 5 条）**：设备特性是**运行期**事实 → 不可能全靠编译期标签；但可以做到：
① **init 期解析成常量成员**（不是 virtual、不是每次查询）；
② **编译期标签**覆盖"由引擎类别决定"的部分：CPU 引擎永不具备原生 f16 搬运/归约，
   纯 CPU 构建（`NN_ENABLE_GPU=OFF` / 无 Vulkan）连 GPU 引擎都不存在 → 这些用
   `if constexpr` / 宏在编译期消掉分支，热路径零判断；
③ **每结构可变**的事实（有没有该表达式的融合变体）不属于"能力"，应回到**实现内部一次解析**
   （并缓存到 spec 身份上，见 §2.3 的 F2 修复）。

### 1.4 其它违规/不良设计（全库扫描）

| 项 | 现状 | 判定 |
|---|---|---|
| L3 依赖具体层 | `ternary_deploy.hpp:195/332` 的 `dynamic_cast<BitLinear*>` | 违规（随文件删除） |
| L5 依赖具体模型 | `src/text_infer.cpp:123/126` 的 `dynamic_cast<GPTModel*/RAPTModel*>`（生成/打印用） | **登记在案**：CLI 面向具体架构是可接受的，但要统一口径（见 §2.4 的"登记而非删除"） |
| 探针目标依赖具体模型 | `src/mem_probe.cpp:551` | 合规（探针，`NN_ENABLE_PROBES` 门内） |
| 层上的"只被测试用"的 API | `quantize_calls()` / `grad_buffers_allocated()` | 登记为**诊断面**（不进设计面），保留但标注 |
| 双形态缓冲 | `BitLinear` 的打包容器 / f16 稠密（`use_packed_` + `force_pack_` + `sync_buffer_mode_` + `ensure_quant_buffer_`） | 违规（形态应由引擎内部决定，层只有一种） |
| 并行读写器 | `ternary_deploy.hpp` 全套 | 违规（随文件删除） |

---

## 2. 设计

### 2.1 文件角色：`.nnmodel` 与 `.nnckpt`（单一格式 NNNN v7）

**一句话**：**一份格式，两种角色**；角色决定"写哪些参数"，扩展名即角色，**不需要任何 CLI 标签**
（裁决第 3 条）。

| 角色 | 文件 | 内容 | 谁写 | 谁读 |
|---|---|---|---|---|
| **模型**（推理） | `*.nnmodel`（`.bin` 的接替者） | 推理所需的**全部**参数：Dense 层参数 + 三值层的**打包形态** + τ。**不含**"训练才需要"的参数 | `text_train`/`mnist_train` 收尾的 `--save`（默认 `gpt_model.nnmodel`） | `text_infer` / `mnist_infer` / 任何人 |
| **训练断点** | `*.nnckpt` | 模型 + **训练参数**：三值层 latent（f32）+ 训练状态段（v8；见 §2.1.1；格式、读写与训练路径接线均已落地） | `--save-interval`（周期性）、需要续训时 | `--resume`（Checkpoint 角色；v8 起状态段为唯一权威） |

**v7 文件头**：`magic "NNNN" | version=7 | Scalar precision | role u8 | spec_len + spec kvrec`
—— `role`（Model / Checkpoint）是**文件属性**，不是"模型状态"：它只回答"这份文件是什么"，
用来做 `--resume` 的角色校验与 CLI 提示（裁决第 2 条的口径：不引入 可训练/不可训练 状态）。

**v7 参数条目（参数级 kind，裁决第一轮第 1 条）**

```
[kind u8][precision u8][rows u64][cols u64][aux_len u64][payload_len u64][aux][payload]
```

| kind | 含义 | aux | payload |
|---|---|---|---|
| `0 Dense` | 普通浮点参数（Linear 权重/偏置、norm、嵌入、BatchNorm 统计…） | 空 | `rows×cols` 元素（precision = f32/f16） |
| `1 TernaryLatent` | BitLinear 的 f32 **latent**（训练态；写进 `.nnckpt`） | 空 | `rows×cols` f32 |
| `2 TernaryPacked2` | 三值打包（2 bit/权重；运行期容器形态） | τ（rows×f32） | `rows × ceil(K/16) × 4B` |
| `3 TernaryPacked3` | 三值打包（base-3，1.6 bpw；**`.nnmodel` 的默认载荷**） | τ（rows×f32） | `rows × ceil(K/5) B` |
| `4 I8Packed4`（预留） | 激活/张量 int8（4×int8/字，P3 用） | 逐列 scale | `rows × ceil(K/4) × 4B` |
| `5…` | 其它后期扩展（同一套 aux/payload 结构） | — | — |

**载入规则（"识别该字段并据此读取"，裁决第一轮第 2 条）**

1. 读头部 → 校验 magic/版本/精度/**role**；
2. 逐条读 `[kind][precision][rows][cols][aux][payload]`，顺序 = `Model::parameters()`（复合层转发）；
3. 按 kind 分派：`Dense`/`TernaryLatent` → 写入目标参数张量（`copy_from`）；`TernaryPacked*` →
   交给**拥有该参数的那个层**（参数形态协议，§2.2）接管；`I8Packed4` 等同理；
4. **双端校验**：目标层声明的"可接受形态集合"与实际 kind 不符 → fail-fast（层名 + 期望 + 实际）；
5. `--resume` 额外要求 `role == Checkpoint`（角色不符 → 明确错误，而不是等到 backward 才炸）。

**旧文件**：v5/v6 的 `*.bin` 仍可读（零破坏，`role` 缺失按 Model 处理）；**写一律 v7 + 新扩展名**；
CLI/GUI 的默认保存名同步改为 `.nnmodel`（`.bin` 不再被写入）。

### 2.1.1 训练状态段（v8；**P1-2 已实施**，2026-10-16 裁定）

**一句话**：断点是**训练的可续点**，不是"更大的模型文件"——状态段里的一切都是**断点权威**，
CLI 不再有发言权。

**落地进度（2026-10-16）**：**第一刀已落地** —— v8 段布局 + `TrainState` 编解码
（`include/neuralnet.cpp/train_state.hpp`）+ 读写 API（`save_model` 的 `CheckpointState`、
`peek_train_state` / `load_train_state` / `verify_model_integrity`）+ 回归用例
`train_state_test`（含"剥成 v7 仍可加载"的零破坏用例）；**第二刀已落地** ——
`Optimizer::save_state` / `load_state` / `state_step`（全表覆盖，含"重启续跑 = 直跑"的逐位门禁
`optimizer_state_test`）；**第三刀已落地** —— 训练路径接线（`--save-interval` 写状态段、
`--resume` 采纳状态段：精度 profile / 曲线形状 / 网格 / 起点 / 优化器 `t`）、A/B 类**禁列**
（显式给出即报错，按旗标名判定）、数据集身份**硬错门禁**、`.nnmodel` **warm start 放行**
（第三刀时三值 `.nnmodel` 仍拒并指向 P1-7），端到端四例 + 单测见 `docs/history/history.md`
「P1-2（第三刀）」；**第四刀 / 收口已落地** —— GUI 的 `--resume-epoch/--resume-step` 随 role 禁用
（`gui_cli_audit.py` 同步）；**P1-7（2026-10-16）收掉了最后一条拒绝**：三值 `.nnmodel` 由
`WeightLoadIntent::Trainable` 反量化成可训练 latent（见 18 §4.2 P1-7），`role` 一面已无"必须
拒绝"的组合。

**段布局（v8，追加在 tokenizer 之后；老文件按版本回落）**

```
[magic][version=8][precision][role][spec_len][spec]        ← 头部（同 v7）
[参数条目 …][extra state …][tokenizer_len][tokenizer]       ← 同 v7（role 决定参数形态）
[state_len u64][state kvrec][优化器状态条目 …]               ← 新增；role=Checkpoint 非空，Model 写 0
[sha256 32B][magic "NNEND" 5B]                              ← P2-1：整文件校验和 + 尾部完整性标记
```

**状态段内容（四类）**

| 类 | 内容 | 为什么必须有 |
|---|---|---|
| 计数器 | global_step / epoch / epoch 内步号 / 当前 lr | 续训起点；`--resume-epoch/--resume-step` 从"必填"降级为"禁止再填" |
| 优化器状态 | Adam m/v **+ `t`**、SGD momentum、Muon、AdamWQuant 的 f16 状态与逐行尺度（经 `Optimizer::save_state/load_state`） | 缺 `t` 会让偏置修正在重启后从 0 重算 ⇒ 前几百步更新量偏小，"loss 轨迹连续"不成立 |
| 训练配方 | 精度 profile 四槽 + weight_quant、lr 调度配方与**曲线形状**（`total_epochs`/`total_steps`）、batch_size / accum_steps / max_norm / stride | 曲线形状**是配方**（`total_epochs = --epochs`、`total_steps = steps_per_epoch × epochs`），不是"跑多久"；profile 今天完全不进文件 ⇒ `--f16` 断点忘带旗标会被静默重建成 f32 |
| 数据集身份 | `.nndataset` 文件 sha256 + tokens/docs/**samples**/`steps_per_epoch`/loss_scope | 身份不匹配 ⇒ **硬报错、无逃生阀**（理由见下） |

**role 语义（冻结）**

| role | `--resume` 行为 |
|---|---|
| `Checkpoint`（`.nnckpt`） | 状态段是唯一权威：除"完全不改变训练"的参数（保存路径 / 保存间隔 / log 间隔 / `--gpu` / 激活卸载 / `--max-steps` / 探针环境变量 / 数据集**路径**）外，**任何显式给出的训练参数都报错**（按旗标名判定、不看值）。禁列 = 形状与配方（`--d-model`…/`--optimizer`/`--lr`/`--min-lr`/`--weight-decay`/`--beta1`/`--beta2`/`--lr-schedule`/`--warmup-epochs`/`--warmup-steps`/`--batch-size`/`--accum-steps`/`--max-norm`/`--epochs`/`--f16`/`--t1_58`/`--precision-*`）+ 计数器（`--resume-epoch`/`--resume-step`）。 |
| `Model`（`.nnmodel`） | **CLI 全权**：文件只提供权重与规格（架构仍以文件为准），优化器冷启动、step/lr 从 0 起算——这是 **SFT / 换数据集续训**的入口。三值模型的 `.nnmodel` 载荷是打包形态：载入由**意图**（`WeightLoadIntent`）分流——`Inference`（默认）进 `packed_only_` 部署只读态，`Trainable`（训练侧一律用）= 按文件 τ 反量化成 f32 latent ⇒ **三值 SFT / QAT 同样可续训**（P1-7，2026-10-16 落地，取代了原"唯一拒绝"那条门禁）。 |

**数据集身份的字段口径（P2b，2026-10-21 / v1.10.0-rc0.1）**：`DatasetIdentity.num_windows` →
**`num_samples`**（语义 = **训练样本数** = 文本窗口数；`step` 仍是 `ceil(num_samples / batch_size)`
的派生量）。状态段落盘键同步改 `dataset_samples`；旧 rc0 断点的 `dataset_windows` **兼容读**
（缺失新键时回退，同一语义）；写出一律用新键，**v8 文件版本号不变**。

**为什么数据集必须硬错、且不需要逃生阀**：数据集变 ⇒ **样本数**变 ⇒ `steps_per_epoch` 变 ⇒
① step_cosine 的 `total_steps`、② `global_step = epoch*steps_per_epoch + step`、③ epoch 边界与
`--resume-step` 的含义**同时**改变 ⇒ "同一断点 + 不同数据集"**不存在正确的续训语义**，放行等于
放行一个自欺的续训。合法出口就是上表的 `Model` 行：换数据集 = 新任务 ⇒ 从 `.nnmodel` warm start。

**为什么状态段走 role 门控而不是独立格式**：checkpoint = model 的**超集**，两者共享头部、spec、
参数条目与词表；独立格式要把参数形态协议复制一遍、让推理侧嗅探两个 magic。而 `.nnckpt` 还要能被
`text_infer` 直接读来做推理自检（裁决第 3 条的"测试推理"用途）⇒ **同一容器 + role 字节**是唯一
自然的表达。（独立格式 + 独立版本号、无差别忽略 CLI、数据集逃生阀三条均已在 2026-10-16 裁定中否决，
过程见 `docs/history/history.md`。）

**数据集路径 vs 身份**：路径是**位置**（机器相关，CLI/位置参数优先），身份是**内容**（断点权威）。
resume 时不给数据集参数可回落断点里记的路径；给了路径则照常打开、按 sha256 校验身份。

### 2.2 参数形态协议（取代 `dynamic_cast` 与 `Layer::sub_layers()`）

```cpp
// Layer 侧（叶子实现；复合层按 `parameters()` 顺序转发 —— 与既有 parameters() 同形）
virtual Result<ParamEntry> export_param(std::size_t index);
virtual Result<void>      import_param(std::size_t index, const ParamEntry& entry);
```

- `ParamEntry` = { kind, precision, rows, cols, aux(Tensor), payload(Tensor) }（`Tensor`，非 `Matrix`
  → 不触 L2 铁律；宿主上传/下载在序列化层）。
- 叶子层声明**可接受形态集合**（如 `BitLinear` = {TernaryLatent, TernaryPacked2}，
  `Linear` = {Dense}），载入时用于 §2.1 的第 4 步校验。**盘上的 base-3 载荷（kind=3）由序列化层
  在读盘时转回 2-bit 容器**（布局唯一来源 `ternary_pack.hpp`）⇒ 层只见到 kind=2，故不在集合里。
- **删除 `Layer::sub_layers()`**（裁决第 2 条）：kind 已定参数级 + 载入走参数协议后，层树枚举没有
  消费者。复合层仍然是正当的（转发参数、组织多层激活等），只是承载的是**参数转发**而不是**层枚举**。
- **不引入只读态/可训练态**（裁决第二轮第 2 条）：打包形态的层没有 latent ⇒ `backward()` 返回
  明确的前置条件错误；`parameters()`/`param_gradients()` 不再 abort；模型不聚合任何标志。

### 2.3 引擎能力体系（裁决第 4/5 条 + 第二轮第 3 条）

**规则（拟写进 AGENTS 铁律）**

1. 能力谓词 = **具体引擎的私有实现细节**：命名 `supports_*_`（尾随下划线 = private 约定），
   **不允许任何外部调用**（Layer / L4 / L5 / CLI / 测试都不行）。
2. **不得出现在基类 `ComputeEngine`**：基类 NVI 体内不做能力分支（裁决第 1 条"移除全部"）；
   分支下沉到各引擎的 `*_impl`。
3. **不得作为"某原语支持与否"的对外标签**；只用于**引擎内部分发计算路径**。
4. 设备不支持某路径时：引擎内**降级**（等效实现）或**报错**（明确 `Result` 错误）。
5. **能力信息**（给 CLI/探针/测试/日志看的那一份）走**统一的对外查询接口**：
   ```cpp
   struct EngineInfo { std::string engine; std::string device;   // 名称
                       bool native_f16_move, native_f16_reduce, fused_prec_variant,
                            ternary_gemm, int8_dot, int8_dot_accelerated;  // init 期已解析的事实
                       std::string text() const; };                 // 稳定、可机读的一行式摘要
   [[nodiscard]] EngineInfo EngineInfo_() const;   // 唯一对外通道（只读缓存，不做工作）
   ```
   —— 它**只披露**事实，**禁止**用于选路（选路必须走内部谓词）。

**热路径政策（裁决第 5 条的落地）**

| 事实类型 | 表达方式 | 热路径代价 |
|---|---|---|
| **由引擎类别决定**（CPU 引擎无原生 f16 搬运/归约；纯 CPU 构建无 GPU 引擎） | **编译期标签**：`if constexpr`（引擎类型已知）/ `#if NN_HAS_VULKAN` / 复用既有 `NN_ENABLE_GPU` 与 `NN_ENABLE_NATIVE` 口径 | **0**（分支不存在） |
| **由设备决定**（Vulkan 特性/管线是否创建成功） | **init 期解析成常量成员**（`const bool native_f16_move_;`），非 virtual | 1 次可预测的内存读 |
| **每结构/每张量可变**（有无该表达式的融合变体） | **不属于能力**：在实现内部一次解析并**按 spec 身份缓存**（而非每次重建 key） | 1 次查表 |

**F2（`eval_expr` 双份归因）的修复方向**：基类三处 `supports_expr_precision_variant(...)` 调用删除，
把"选变体还是边界 cast"整段下沉到 `eval_expr_impl` / `eval_expr_into_impl` / `eval_expr_reduce_impl`
（它们本来就要算 key）→ 每次求值**只归因一次**；并把 key/变体解析结果缓存在 `ExprSpec` 对象身份上
（spec 是构建期常量对象），消除每次求值的 `std::string` 构造。
**验收**：`NN_PREC_TRACE` 命中计数不变（语义等价）+ 单测/CI 计时对比（每 step 的 host 时间、
`layer_bench --op` 的 host 侧）。

**本轮新增位**：`supports_ternary_gemm()` 从基类删除（三值 NVI 段里那 4 处调用一并消失）；
`BitLinear` 不再查能力（形态单一化，降级/报错在 `matmul_q_impl` 内部）；`supports_int8_dot()` /
`int8_dot_accelerated()` → 引擎私有谓词 + 进 `EngineInfo`。

### 2.4 删除 / 保留 / 登记清单

**删除**

- `include/neuralnet.cpp/ternary_deploy.hpp` 整文件（magic / 版本 / 全部并行入口）；
  `src/ternary_deploy_test.cpp`（并入 `model_serialization` 侧的测试面）。
- CLI/GUI 标签：`text_train --export-deploy`、`gui.py` 的"导出部署文件"行、
  `cli_controllers.py` 的 `export_deploy`、`bench/gui_cli_audit.py` 的对应用例、
  `text_infer`/`mnist_infer` 的 `is_deployment_file` 分支。
- `Layer::sub_layers()`（含 AttentionBase / FeedForward / GPTBlock / GPTModel 的实现）。
- 基类 `ComputeEngine` 的**全部**能力谓词：`supports_native_data_move` / `supports_native_f16_reduce`
  / `supports_expr_precision_variant` / `supports_ternary_gemm` / `supports_int8_dot` /
  `int8_dot_accelerated`（→ 各引擎私有 `*_` 成员 + `EngineInfo`）。
- `BitLinear` 特例 API：`set_ternary_pack_enabled` / `set_ternary_pack_auto` / `uses_packed_weights`
  / `quantized_weights()` / `quantized_weights_packed()` / `row_scales()` / `load_packed_weights`
  / `packed_only()` / `ensure_quantized()`；`parameters()`/`param_gradients()` 的 abort；
  两形态缓冲（`use_packed_` / `force_pack_` / `sync_buffer_mode_` / `ensure_quant_buffer_` 的双支）。

**保留**

- `ternary_pack.hpp`（2-bit + base-3 编解码）、`matmul_q`/`matmul_q_t` 与全部打包内核（分块 + GEMV）、
  CPU 行块解码；`BitLinear` 的量化/STE/写版本缓存；`AdamWQuant`；P3 的设备特性探测（改为内部）。
- `.nndataset` / `.nnvocab`（数据集格式，见 docs 19）与 `.nnpkg`（训练包，`train_pkg.py`）——**与模型
  文件角色无关**，不在本次清理范围。

**登记（不删，写明边界）**

- `src/text_infer.cpp` 的 `dynamic_cast<GPTModel*/RAPTModel*>`（生成路径）与 `mem_probe` 的用法：
  CLI/探针**允许**面向具体架构；库内 L2/L3 **禁止**。写入 AGENTS 的分层说明。
- `quantize_calls()` / `grad_buffers_allocated()`：标注为**诊断面**（测试可见），不进设计面。

### 2.5 迁移步骤（每步可独立验收）

1. **能力体系**（裁决 1/4/5）：基类能力谓词全部移出 → 各引擎私有 `*_` 成员；新增 `EngineInfo`
   唯一对外通道；热路径按 §2.3 的政策重写（编译期标签 / init 期常量）；修 F2 的双份归因。
   验收：ctest 全绿 + `NN_PREC_TRACE` 语义等价 + host 侧计时对比 + 本机 CPU-only 构建
   （无 Vulkan）与 102 各跑一遍。
2. **参数形态协议**：`export_param`/`import_param` + 可接受形态集合 + 复合层转发；删
   `sub_layers()` 与全部 `BitLinear` 特例 API；两形态缓冲收敛为单形态。
3. **v7 读写 + 文件角色**：`save_model`/`load_model` 支持 v7（role + 参数级 kind）；
   `.nnmodel`（默认写 packed/base-3）与 `.nnckpt`（写 latent）两种角色；`--save` 默认名改
   `*.nnmodel`；`--save-interval` 写 `*.nnckpt`；`--resume` 只收 `.nnckpt`；v5/v6 仍可读。
   验收："训练收尾 → .nnmodel → 推理输出与原模型逐字节一致"（沿用今天 11.9×/逐字节那组口径）+
   "`.nnckpt` → resume 继续训练" + 老 `.bin` 可读。
4. **删标签**：删 `--export-deploy`、GUI 行、controller 透传、审计用例、推理 CLI 的分流分支。
5. **文档同步**：21 §4.15/§4.16 改写（去掉"独立格式/部署文件"的表述）、history 记一条
   "库级清理：文件角色 + 能力体系 + 参数协议"、AGENTS 增两条铁律（能力体系 / 参数形态协议）
   并更新 §4.2 的三值段落。
6. **P3 全链路**（裁决"不管是不是有效"）：`quantize_act_i8` + `matmul_q_i8` + 端到端对拍；
   int8 走 v7 的 `I8Packed4` 与同一套参数协议，不另起容器特例；能力走 `EngineInfo`。

### 2.6 验收口径

- 本机（CPU-only）+ 102（含全部 GPU 用例）ctest 全绿；`doc_inventory` L2-VIOLATIONS 0 / GATE PASS；
  `doc_align_audit` E/F 节 0；`gui_cli_audit` PASS（且不再有 `export-deploy` 用例）。
- 端到端不回退：训练 60 步 2.0 s / 训练峰值 522 MiB / 解码 0.7 s / 推理峰值 76 MiB；
  `.nnmodel` 文件 ≤ 今天 NNQT 的 1 126 970 B 量级、载入后**逐字节等价**。
- 热路径：`eval_expr` 每次求值的 key 构造次数从 2 → ≤1（并按 spec 身份缓存后为 0 次重建）；
  基类 NVI 体内的能力分支 0 处。

---

## 3. 待裁决（剩余）

1. **`EngineInfo` 的形态**：结构体 + `text()` 摘要（本稿建议）／只给稳定文本／其它？
   以及它的**命名**（`EngineInfo_()` 与既有 `*_` 私有约定冲突，需要另定，如 `describe()`）。
2. **`.nnmodel` 是否也保留 latent**：我建议**不保留**（那是 `.nnckpt` 的职责；否则又变成"一个文件
   两种东西"）。若你希望"仅推理但可微调"，请给出口径。
3. **旧 `.bin` 的迁移**：只"读旧写新"（本稿建议）／提供一次性转换工具／`.bin` 直接不再被识别？
4. **spec-key 缓存方案**（F2 修复的次要项）：按 `ExprSpec` 对象身份缓存（本稿建议）／把 key
   在构建期算好存进 spec／不做缓存只去掉重复。

---

## 4. 落地结果（2026-10-11，本节记录**实施后**的实际形态）

**§3 待裁决的四条，按建议落地**：① `EngineInfo` 结构体 + `text()` 一行摘要，接口名
`ComputeEngine::describe()`（避开 `*_` 私有约定）；② `.nnmodel` **不保留** latent；
③ 旧 `.bin` **只读不写**（不提供转换工具：header 无 role 字节 → 按 Model 处理）；
④ 取消"按 spec 身份缓存"，只**去掉重复归因**（选路+回退合并进 `*_impl`）——实测无回归。

**§2.5 六步**：1（能力体系）✅ / 2（参数形态协议）✅ / 3（v7 + 文件角色）✅ /
4（删标签）✅ / 5（文档同步）✅ / 6（P3 int8 全链路）**⏳ 未做**（属新功能，见 §2.5-6）。

**落地时的三处实现细节**（与 §2 设计的差异，均已写进代码注释）：

1. **盘上编码 vs 运行期形态**：`export_param(index, Model)` 返回**运行期**形态
   （`TernaryPacked2`，2 bit 容器）；序列化层（L3）在写 `.nnmodel` 时把它编成
   `TernaryPacked3`（base-3 1.6 bpw），读回时解成 2-bit 容器再交给层。理由是
   base-3 编解码需要宿主字节，而 L2 层不得碰 Matrix/宿主 I/O（铁律 12）——
   **层只认运行期形态**，盘上编码是 L3 的事。
2. **子层枚举的落点**：公开的 `Layer::sub_layers()` 删除；参数下标转发改由**受保护的**
   `param_children()` 承担（默认 `export_param`/`import_param` 按"自身参数在前 + 子层
   顺序"转发），写在 `compute_layer_base.hpp` 的参数形态协议注释里。
3. **`BitLinear` 的两形态缓冲**：打包是**默认且唯一**的训练/解码形态；`force_pack_`
   与 `set_ternary_pack_enabled/auto` 保留为 **A/B 与回退**开关（`layer_bench` 的
   `matmul_q_bw` 口径依赖它），其余特例 API（`load_packed_weights`/`packed_only()`/
   `ensure_quantized`/`quantized_weights_packed`/`row_scales`）依旧保留给
   `import_param` 与诊断使用 —— 它们不再是**对外部署格式**的一部分。

