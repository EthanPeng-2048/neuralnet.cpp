# 文档总索引与分册定位

> 本文件是 `docs/` 的**总入口**：规定每个子目录**写给谁看**，并给出全部文档的一行索引。
> 一句话总则：**除 `todo/` 与 `history/history.md` 外，`docs/` 下的文档都是写给调用者的**；
> `todo/` 与 `history/history.md` 是项目开发者的内部工作区。

## 1. 分册定位（写给谁看）

| 目录 | 读者 | 内容 | 只记录 |
|------|------|------|--------|
| `introduction/` | 读者 / 调用者 | 架构、性能、算法、创新设计的总览 | 当前状态 |
| `usage/` | 调用者 | 用现成能力：CLI、C++ API、GUI、`.nnpkg`、作为库嵌入 | 当前状态 |
| `development/` | **调用者（库使用者 / 扩展者）** | **在库之上写代码**：CMake 接入与全功能 C++ API 手册、写自定义层、扩展引擎原语、精度语义契约、算法与原语的对应 | 当前状态 |
| `todo/` | **项目开发者** | 项目自身的工作：设计稿、路线图、盘点、规范、踩坑、故障报告、审查整改立项 | 当前状态（设计+进度） |
| `history/history.md` | 项目开发者 | 全仓**唯一**历史归档：演进记录 / 已修复勘误 / 被否决方案 / 性能 A/B | 历史 |
| `benchmarks/` | 项目开发者 / 评估者 | 对外性能对比基准报告（Vulkan vs CUDA / Torch） | 采集时的实测 |
| `release-notes/` | 调用者 | 版本变更说明（破坏性变更 + 迁移指南 + 验收基线） | 版本发布时 |
| [`../research/`](../research/README.md) | **想引用实测结论的人**（含库外） | 一次研究一个子目录：结论 / 证据 / **重跑命令** / 适用边界 | 实测结论（不随代码漂移） |

### 1.1 `development/` 的定位（明确约定）

**`docs/development/` 是给调用者的**——即"拿着这个库去开发自己东西"的人：写自定义层、
扩展引擎原语、接入 GPU 融合、选精度策略。它**不是**项目开发者的工作区：

- 项目内部的设计稿、路线图、盘点、规范、踩坑、故障报告、审查整改 → `docs/todo/`；
- 已落地的演进过程、被否决方案、旧数字 → `docs/history/history.md`。

判断标准一句话：**这篇文档的读者是"库外的人"还是"改库的人"？** 库外的放
`development/`（或 `usage/`），改库的放 `todo/`。组件式设计文档即使被源码注释引用，
也仍属 `todo/`（源码注释引用内部设计稿是正常的）。

### 1.2 `todo/` 的定位

`docs/todo/` = **项目开发者的内部工作区**，按"一件事一篇"存档：设计（含未实施部分）、
路线图与裁定、现状盘点、编码规范、踩坑记录、未修故障、以及针对代码审查/评审的**整改立项**。
这里的文档允许带进度状态（✅ 已实施 / ⏳ 待做），但不承载"当前是什么"以外的历史叙事——
历史叙事一律进 `history/history.md`。

## 2. 索引

### introduction/

| 文档 | 何时读 |
|------|--------|
| [`01-architecture.md`](introduction/01-architecture.md) | 完整分层 / 数据流 / 模块详解 + 快速理解指南 |
| [`02-performance.md`](introduction/02-performance.md) | 性能优化（自适应并行、缓存分块、GPU） |
| [`03-algorithm-reference.md`](introduction/03-algorithm-reference.md) | 每个 Layer/Loss/Optimizer 的数学与原语分解 |
| [`04-innovative-designs.md`](introduction/04-innovative-designs.md) | 创新设计全景 |

### usage/

| 文档 | 何时读 |
|------|--------|
| [`01-quickstart-model.md`](usage/01-quickstart-model.md) | 构建模型 API 教程 |
| [`02-quickstart-train-infer.md`](usage/02-quickstart-train-infer.md) | 训练/推理 CLI + C++ API + GUI |
| [`03-compute-engine-usage.md`](usage/03-compute-engine-usage.md) | 计算引擎使用：张量、矩阵、表达式融合 |
| [`04-train-package.md`](usage/04-train-package.md) | `.nnpkg` 训练包 |
| [`05-consume-as-library.md`](usage/05-consume-as-library.md) | 把库嵌进自己工程（`neuralnet::nn` 门面 / FetchContent） |

### development/（给调用者）

| 文档 | 何时读 |
|------|--------|
| [`README.md`](development/README.md) | 本分册定位与索引（含入门路径建议） |
| [`cpp-api-manual.md`](development/cpp-api-manual.md) | **全功能 C++ API 手册**：错误宏 / 引擎 / 张量 / 精度 / DSL / 层与 Model / 工厂 / 损失 / 优化器 / 训练循环 / 模型文件 / 数据集 / 分词器 / 生成 / 剖析（每节带代码片段） |
| [`cmake-integration.md`](development/cmake-integration.md) | 接入与构建：`add_subdirectory`/`FetchContent`、全部构建开关、GPU 语义、`nn_enable_gpu_fusion`、跨平台、故障排查 |
| [`write-a-custom-layer.md`](development/write-a-custom-layer.md) | 写自己的层：完整示例（实测过梯度检查）+ 接入契约 + 精度槽声明 + 梯度自检 + GPU 清单 |
| [`01-compute-engine-development.md`](development/01-compute-engine-development.md) | 扩展引擎原语：接口全解 + CPU/GPU 实现模式 + 加新原语的完整步骤 |
| [`05-mixed-precision.md`](development/05-mixed-precision.md) | 多精度语义契约（写层/选 profile 必读）+ 实测限制 |
| [`06-rapt-algorithm.md`](development/06-rapt-algorithm.md) | 线性注意力（RLA → RAPT → RLA-2）算法 + 扫描原语落地 |
| [`07-vulkan-backend-map.md`](development/07-vulkan-backend-map.md) | Vulkan 后端结构地图（维护者）：调用链 / 所有权 / 依赖方向 / 扩展步骤 |
| [`08-object-lifetimes.md`](development/08-object-lifetimes.md) | 对象生命周期契约 v2：EngineId / shared_ptr 强所有权 / GPU buffer keepalive（迁移检查清单） |
| [`09-engine-ownership.md`](development/09-engine-ownership.md) | 引擎/ backend 所有权设计：不变量、改动范围、API 影响、性能口径与实施顺序 |
| [`20-custom-layer-ergonomics.md`](development/20-custom-layer-ergonomics.md) | 写自定义层的**实测报告**：接入契约与 7 个摩擦点（库外使用者视角） |

### todo/（给项目开发者）

见 [`todo/README.md`](todo/README.md) 的完整索引。要点：

- `todo/00-code-review-remediation.md` — **2026-10-11 代码审查整改立项台账**（本批新增）。
- `todo/18-roadmap.md` — **"某件事该不该做"的唯一入口**（裁定台账 + 验收口径 + 现状实测；不含执行计划）。
- `todo/27-execution-plan.md` — **执行计划（优先级象限）**："下一个先做哪个"。
- `todo/12` / `todo/17` / `todo/19` / `todo/21` / `todo/22` / `todo/23` — 现行设计/盘点。
- `todo/24` / `todo/25` — f16 训练缺陷台账 / **大版本发布要求**（`1.x.y → 1.(x+1).0`）。

### 总档

| 文档 | 何时读 |
|------|--------|
| [`history/history.md`](history/history.md) | 查"为什么当初这么改、旧数字是多少"（全仓唯一历史归档） |
| [`../research/README.md`](../research/README.md) | 引用**实测结论**（MNIST 准确率、f16 回弹根因、三值 STE / 尺度规则）：每条带证据、重跑命令与边界 |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 参与本项目（构建 / 测试 / 规范 / PR / CI 口径）——**给非作者的入口** |
| [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | 社区行为准则 |
| [`../CMakePresets.json`](../CMakePresets.json) | 预设构建（`cmake --preset dev` 等）——见 CONTRIBUTING §1 |
| [`benchmarks/2026-09-25-vulkan-vs-cuda.md`](benchmarks/2026-09-25-vulkan-vs-cuda.md) | Vulkan 后端 vs CUDA（torch）基准 |
| [`benchmarks/2026-10-06-vulkan-vs-torch.md`](benchmarks/2026-10-06-vulkan-vs-torch.md) | 第二轮复测 |
| `release-notes/v1.5.0.md` … `v1.8.2.md` | 版本变更说明（**大版本发布要求**见 [`todo/25-release-requirements.md`](todo/25-release-requirements.md)） |

## 3. 2026-10-11 文档整理记录

本仓库的文档在 2026-10-11 做过一次结构性整理，原则 =「`development/` 只留调用者文档；
项目开发者文档进 `todo/`；历史进 `history/history.md`」。逐项：

**迁移到 `docs/todo/`**（原 `docs/development/`，文件名与编号不变）：`02` `03` `04` `08`
`10` `12` `14` `17` `18` `19` `21` `22` `23`。

**整篇并入 `docs/history/history.md`**（内容已由后继文档承接，正文保留在 history 文末归档节）：
`15-computeengine-refresh.md`、`16-computeengine-p0-inventory.md`。

**删除**：`13-refactor-backlog.md`——2026-10-01 起就只是 16 行的重定向 stub，
正文早在 `history/history.md` 与 `todo/18-roadmap.md`。

**保留在 `docs/development/`（并重定义其读者为"调用者"）**：`01` `05` `06` `20`。

**新增**：`docs/README.md`（本文件）、`docs/todo/README.md`、`docs/development/README.md`、
`docs/todo/00-code-review-remediation.md`（审查整改立项）。

**同日第二轮（补"调用者能否不读源码用全库"的缺口）**：`docs/development/` 新增三篇
用户向文档——`cpp-api-manual.md`（全功能 C++ API 手册）、`cmake-integration.md`
（接入与构建全解）、`write-a-custom-layer.md`（写自定义层指南，含实测过梯度检查的示例层）。
三篇均通过符号/文件/CLI 参数三重自审（对齐 `bench/doc_align_audit.ps1` 的 [A][D][E] 口径），
其中示例层的代码片段经真实编译 + 数值梯度检查实跑验证。`development/20` 的
悬空样例引用（`examples/custom_layer_*` 已不在仓库）一并订正。

整理前 `docs/development/` 共有 20 篇文档，其中多数是"一件事只用几次"的内部设计/盘点稿——
这正是本次把它们从面向用户的分册里移出的原因。

## 4. 2026-10-11 对外入口补全（第二轮）

第一轮解决"文档写给谁看"，第二轮补"**非作者怎么进得来**"。逐项：

| 增加 | 作用 |
|------|------|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 给非作者的统一入口：构建（preset）、测试口径、代码规范、提交信息写法、PR 自查清单、CI 实际做什么（**含"CI 不跑 ctest"这一事实**）、文档/研究怎么贡献 |
| [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | 社区行为准则（Contributor Covenant 2.1 改编）+ 本仓特有的"技术讨论对事不对人"约定 |
| [`../CMakePresets.json`](../CMakePresets.json) | 预设构建：`release` / `dev` / `debug` / `portable` / `cpu` / `probes`（+ build/test/workflow preset）。**只描述口径，不改变 CMakeLists 默认值** |
| [`../.clang-format`](../.clang-format) + [`../.editorconfig`](../.editorconfig) + [`../.gitattributes`](../.gitattributes) | 把 §8 的风格规则变成机器可读；全仓 LF。⚠ 定位是**目标风格 + 增量工具**（`git clang-format`），**不是整仓重排脚本**——对全仓整体格式化会改约 41% 的行，见 history 的「代码风格可执行化」条 |
| [`../research/README.md`](../research/README.md) 重写 | 从"目录表"升级为**可引用结论索引**：每条给 结论（带数字与样本量）/ 证据 / 重跑命令 / **边界**，并写明"怎么引用、怎么新增一份研究" |

同时修掉的构建问题：**ccache 与共享 PCH 互斥**（Clang 侧缺 `sloppiness=pch_defines,time_macros`
与 `-fno-pch-timestamp`）导致 40+ 个目标的编译全部不可缓存——已修，单个 TU 从 30.2s 降到 0.11s
（命中缓存；全量重建 13m26s → 58.8s）。归因与实测见 [`history/history.md`](history/history.md) 的「ccache × 共享 PCH」条，
口径与复现见 [`development/cmake-integration.md`](development/cmake-integration.md) §8。
