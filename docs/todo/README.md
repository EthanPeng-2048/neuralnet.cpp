# todo/ —— 项目开发者内部工作区

> **本分册写给项目开发者**（改这个库的人），不是给调用者的。
> 调用者要看的开发文档在 [`../development/`](../development/)；历史归档在 [`../history/history.md`](../history/history.md)。

## 这里的文档怎么组织

- **一件事一篇**：设计稿（含未实施部分）、路线图与裁定、现状盘点、编码规范、踩坑记录、
  未修故障、审查/评审整改立项。
- **裁定与计划分家（2026-10-15 起）**：**裁定台账**（要不要做 / 条件触发 / 不做 + 验收口径）住
  [`18-roadmap.md`](18-roadmap.md)；**执行计划**（先做哪个 / 第一步动作 / 并行与工时）一律**单独成文**，
  18 号只留指针（当前执行排序 = [`27-execution-plan.md`](27-execution-plan.md)，专项方案 = `31` / `26` / `04` / `25` / `28` / `29`）。
  新增内容先判断属于哪一类：能二值化成"做/不做"的进 18，只描述顺序与动作的进 27。
- **允许带进度状态**（✅ 已实施 / ⏳ 待做 / ❌ 已否决），但**不写历史叙事**——
  演进过程、被否决方案的理由、旧数字一律进 `../history/history.md`。
- 编号沿用历史编号（`02`…`31`），**编号缺号是正常的**（`05` `06` `20` 等留在
  `development/`；`13` 已删除；`15` `16` 已并入 `history/history.md`）。

## 分类与索引

### 立项 / 计划 / 台账（先看这些）

| 文档 | 何时读 |
|------|--------|
| [`00-code-review-remediation.md`](00-code-review-remediation.md) | **2026-10-11 代码审查整改立项台账**：审查发现的问题逐条立项（现象/证据/根因/方案/验收/优先级） |
| [`18-roadmap.md`](18-roadmap.md) | **"某件事该不该做"的唯一入口**：未来 6–12 个月方向 + 未完成项**裁定**台账（做 / 条件触发 / 不做）+ 验收口径 + 现状实测；**§4.5 P4 = 持续优化方向**（性能利用率 / 算子性能 / 显存开销 + API 易用性与 Fold 自动化扫描）。**不含执行计划**（见下一行） |
| [`27-execution-plan.md`](27-execution-plan.md) | **执行计划（优先级象限）**：把 18 号 §3/§4/§6 与 00 号 RV 台账里当前可动手的项排成"先做哪个"（重要×简单四象限 + 第一步动作 + 串行依赖）——**想知道"下一个先做哪个"读它** |
| [`31-rl-training-primitives.md`](31-rl-training-primitives.md) | **RL 与可组合训练基础设施（2026-10-23 架构冻结、当前主线）**：具名输出/schema（RL-0）→ experience/ transforms/ objectives（RL-1）→ `TrainingGroupSession` 与可替换编排（RL-2）；需求来自 issue #23 但它是能力下界，公共 API 面向未知库消费者。**f16 与训练性能项在 RL 收口前暂缓** |
| [`26-f16-speedup-plan.md`](26-f16-speedup-plan.md) | **f16 提速 + 两套 f16 计划**：F4 存储路径提速已收口；§8 分 A/B 两条路线——**A = 标准 `--f16`（参考 PyTorch AMP，补动态 loss scaling/master weights）**；**B = 全流程真 f16（B-Mem 无 f32 持久态；capable 设备的 B-Arith 零转换）**。**§8.B.7 汇总 SR / GradientStabilizer / ECO / Unit Scaling / Peri-LN / attention δ 修复的外部证据与 R2.5 落地顺序**。A 是推荐默认，B 是实验/性能目标 |
| [`04-memory-optimization.md`](04-memory-optimization.md) | 显存优化计划：L1 激活重计算（已实施）/ L2 内存池归还（已实施）/ L3 远期 |

### 现行设计 / 总纲（改对应模块前读）

| 文档 | 何时读 |
|------|--------|
| [`17-unified-tensor-engine.md`](17-unified-tensor-engine.md) | Tensor / ComputeEngine / MemoryPool 底层总纲（M1–M6 已实施；M7 未立项）：访问不变量、InitSpec、批量读写、Matrix 降级、内存契约、`import`、Layer 删 engine 形参 |
| [`02-operator-fusion.md`](02-operator-fusion.md) | 算子融合全篇（归约语义 IR → 表达式录制 → matmul 参与融合 → 跨 kernel 自动融合）：改融合/IR 前必读 |
| [`03-ir-optimization.md`](03-ir-optimization.md) | IR-A/B/C/D 优化链设计（IR-C = 图 IR + `begin_expr/end_expr` 融合分析） |
| [`19-unified-dataset.md`](19-unified-dataset.md) | 统一数据集（`.nndataset` / `.nnvocab` / KVRecord v2）格式与 `nn::Dataset`；**P2b 起含样本层**（`dataset_samples.hpp`：存储单位 = doc、训练单位 = 样本/窗口，`step` 是派生量） |
| [`21-quantized-weights.md`](21-quantized-weights.md) | 三值（1.58-bit）量化权重与 `BitLinear`：`Precision::T1_58`、按槽能力声明、打包 GEMM |
| [`22-bitnet-2b4t-architecture.md`](22-bitnet-2b4t-architecture.md) | BitNet b1.58 2B4T 的结构对齐：SubLN / GQA / tied embedding（与 21 号正交） |
| [`23-library-design-cleanup-and-model-file-roles.md`](23-library-design-cleanup-and-model-file-roles.md) | 库级违规设计清理：模型文件角色（`.nnmodel` / `.nnckpt`）+ 参数形态协议 + 引擎能力体系 |
| [`28-lifecycle-v3.md`](28-lifecycle-v3.md) | **生命周期 v3（2026-10-20 立项）**：删 `EngineId`/`stamp_`/`adopt`/`bound`/`NN_BIND_DEBUG`，设备域从存储派生（统一 CPU/GPU 域）；P0/P1/P3/P4 + CPU 独立轨道 P-CPU。改生命周期/Tensor/引擎绑定前读 |
| [`29-session-architecture.md`](29-session-architecture.md) | Session 层历史与所有权设计：Model + Optimizer 同生共死、`InferenceSession` / 通用 `TrainingGroupSession` 占位；`RLTrainingSession` 不是核心类型。P2b 的训练状态归属已由 30 号修订 |
| [`30-training-session-step.md`](30-training-session-step.md) | **TrainingSession 单步协议（已实施）**：Session 接收 micro-batch、内置梯度累积、按 global batch 有效单位归一化并完成一次 optimizer update；`batch_size` 是理论 batch；Loss mask 固化在 Sample；epoch/lr 曲线/评估/运行态 `TrainState` 归 `TextTrainer`。改训练入口前读 |
| [`31-rl-training-primitives.md`](31-rl-training-primitives.md) | **RL 与通用训练基础设施（架构已冻结、未实施）**：具名多输出、schema 化 experience、可组合 policy objectives、通用多模型 Session 与可替换编排；「f16 暂缓、RL 先行」的 2026-10-23 裁定 |

### 现状盘点 / 规范 / 教训 / 故障

| 文档 | 何时读 |
|------|--------|
| [`12-compute-engine-inventory.md`](12-compute-engine-inventory.md) | 引擎接口现状盘点（复现：`bench/doc_inventory.ps1`）：virtual 方法数、Layer 直调分类、两套 CPU 求值机制、ctest 计数 |
| [`10-development-standards.md`](10-development-standards.md) | C++ 编码规范全文（内存管理 / 模块隔离 / 高性能 / 错误处理 / 注释） |
| [`08-pitfalls-and-lessons.md`](08-pitfalls-and-lessons.md) | **踩坑警示录，改代码前读**：分级 bug 与跨领域根因模式 |
| [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md) | **已修故障报告（根因已订正）**：GPU `--precision-stable f16` 训练 loss 打印冻结——真因 = CE 的 `(1,1)` 归约输出按 f16 落盘溢出（不是回读链） |
| [`24-f16-training-defects.md`](24-f16-training-defects.md) | **f16 训练缺陷台账**：全 f16 NaN / loss 冻结 / `--f16` 降速——含报告核对与立项 F1–F7（F5 = 可训练性已完成；**F6 = 真 f16（Plan B）**；**F7 = 标准 `--f16`（Plan A）**） |
| [`25-release-requirements.md`](25-release-requirements.md) | **大版本发布要求**（`1.x.y → 1.(x+1).0`）：端到端模型训练测试 / 发布文档 / 完整 ctest + Debug 构建 / 多编译器 × 标准库矩阵 |

## 相关（相邻分册 / 总档）

| 想去哪 | 文档 |
|--------|------|
| 全仓历史归档（演进 / 勘误 / 否决 / A-B） | [`../history/history.md`](../history/history.md) |
| 总索引与分册定位 | [`../README.md`](../README.md) |
| 性能对比基准 | [`../benchmarks/`](../benchmarks/) |
