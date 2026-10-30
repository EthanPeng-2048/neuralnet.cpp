# 文档总索引与分册定位

> **读者**：找任何一篇文档在哪的人（调用者与项目参与者共用入口）。
> **一句话结论**：`docs/` 按读者分册——调用者看 `introduction`/`usage`/`development`，
> 贡献者看 `contributors/`，改库的人看 `internal/`，查历史看 `history/`。
> **状态**：现行。
> 分册职责与编号制度的**规范正文**在 [`contributors/documentation-standard.md`](contributors/documentation-standard.md)；
> 本文只做索引。

## 1. 分册定位（写给谁看）

| 目录 | 读者 | 内容 | 只记录 |
|------|------|------|--------|
| `introduction/` | 读者 / 调用者 | 架构、性能、算法、创新设计的总览 | 当前状态 |
| `usage/` | 调用者 | 用现成能力：CLI、C++ API、GUI、`.nnpkg`、作为库嵌入 | 当前状态 |
| `development/` | **调用者（库使用者 / 扩展者）** | **在库之上写代码**：CMake 接入与全功能 C++ API 手册、写自定义层、扩展引擎原语、精度语义契约、算法与原语的对应 | 当前状态 |
| [`contributors/`](contributors/README.md) | **贡献者 / 维护者 / agent** | 长期规范与治理：C++ 规范、踩坑警示录、**文档规范**、发布规则 | 长期真源 |
| `internal/` | **项目开发者** | 项目自身的工作：设计稿、路线图、盘点、故障报告、审查整改立项 | 当前状态（设计+进度） |
| `history/history.md` | 项目开发者 | 全仓**唯一**历史归档：演进记录 / 已修复勘误 / 被否决方案 / 性能 A/B | 历史 |
| `benchmarks/` | 项目开发者 / 评估者 | 对外性能对比基准报告（Vulkan vs CUDA / Torch） | 采集时的实测 |
| `release-notes/` | 调用者 | 版本变更说明（破坏性变更 + 迁移指南 + 验收基线） | 版本发布时 |
| [`../research/`](../research/README.md) | **想引用实测结论或核对算法基线的人**（含库外） | 一次研究一个子目录：定义 / 结论 / 证据 / **重跑命令** / 适用边界 | 研究档案（结果与明确标注的 TODO 分开） |

> **一句话总则**：`docs/` 下按"读者是谁"分册；贡献者规则住 `contributors/`，
> 项目内部台账住 `internal/`，历史住 `history/`，其余都是写给调用者的。

### 1.1 `development/` 的定位

**给调用者**——"拿着这个库去开发自己东西"的人：写自定义层、扩展引擎原语、接入 GPU 融合、
选精度策略。编号 = 阅读顺序（01→09，连续不留洞）。

### 1.2 `contributors/` 的定位

**给向本仓贡献的人**：长期有效的规范与治理规则（编码规范 / 踩坑 / 文档规范 / 发布规则）。
`CONTRIBUTING.md` 与 `AGENTS.md` 对本分册**只留摘要与指针**，规则正文以本分册为准。

### 1.3 `internal/` 的定位

**项目开发者的内部工作区**，按"一件事一篇"存档：设计（含未实施部分）、路线图与裁定、
现状盘点、未修故障、整改立项。**编号 = 稳定台账 ID**（新增接最大号，永不重排）。
允许带进度状态（✅ 已实施 / ⏳ 待做），但不承载历史叙事——历史叙事一律进 `history/history.md`。

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
| [`06-reinforcement-learning.md`](usage/06-reinforcement-learning.md) | RL：具名多输出、experience/GAE、REINFORCE/PPO/GRPO、多模型 Session、可替换 rollout pipeline |
| [`07-thread-safety.md`](usage/07-thread-safety.md) | **线程安全契约**：每个对象的并发口径矩阵 + 服务化下游设计模板（issue #38） |

### development/（给调用者）

按阅读顺序；本分册完整索引见 [`development/README.md`](development/README.md)。

| 文档 | 何时读 |
|------|--------|
| [`01-cmake-integration.md`](development/01-cmake-integration.md) | 接入与构建：`add_subdirectory`/`FetchContent`、全部构建开关、GPU 语义、`nn_enable_gpu_fusion`、跨平台、故障排查 |
| [`02-cpp-api-manual.md`](development/02-cpp-api-manual.md) | **全功能 C++ API 手册**：错误宏 / 引擎 / 张量 / 精度 / DSL / 层与 Model / 工厂 / 损失 / 优化器 / 训练循环 / 模型文件 / 数据集 / 分词器 / 生成 / 剖析（每节带代码片段） |
| [`03-write-a-custom-layer.md`](development/03-write-a-custom-layer.md) | 写自己的层：完整示例（实测过梯度检查）+ 接入契约 + 精度槽声明 + 梯度自检 + GPU 清单 |
| [`04-compute-engine-development.md`](development/04-compute-engine-development.md) | 扩展引擎原语：接口全解 + CPU/GPU 实现模式 + 加新原语的完整步骤 |
| [`05-mixed-precision.md`](development/05-mixed-precision.md) | 多精度语义契约（写层/选 profile 必读）+ 实测限制 |
| [`06-vulkan-backend-map.md`](development/06-vulkan-backend-map.md) | Vulkan 后端结构地图：调用链 / 所有权 / 依赖方向 / 扩展步骤 |
| [`07-engine-ownership.md`](development/07-engine-ownership.md) | **生命周期 v3 唯一真源**：设备域模型、入口域检查、backend/GpuBuffer 生命周期、失败语义、迁移清单 |
| [`08-rapt-algorithm.md`](development/08-rapt-algorithm.md) | 线性注意力（RLA → RAPT → RLA-2）算法 + 扫描原语落地；定义基线见 [`../research/rla/`](../research/rla/README.md) |
| [`09-custom-layer-ergonomics.md`](development/09-custom-layer-ergonomics.md) | 写自定义层的**实测报告**：接入契约与 7 个摩擦点（库外使用者视角） |

### contributors/（给贡献者 / 维护者 / agent）

本分册完整索引见 [`contributors/README.md`](contributors/README.md)。要点：

- [`contributors/documentation-standard.md`](contributors/documentation-standard.md) — **文档规范**：分册职责、编号制度、单篇模板、代码示例标注、机器审查口径。**写任何文档前读**。
- [`contributors/cpp-standards.md`](contributors/cpp-standards.md) — **C++ 开发规范全文**（改代码前读）。
- [`contributors/pitfalls.md`](contributors/pitfalls.md) — **踩坑警示录**（改代码前读；踩新坑先补这里）。
- [`contributors/release-policy.md`](contributors/release-policy.md) — **版本号·分支·发布治理**：**发布须由人提出（agent 不得擅自发布）**、`dev` 不写版本号、发布只在 `main`。
- [`contributors/release-requirements.md`](contributors/release-requirements.md) — **大版本发布要求**（`1.x.y → 1.(x+1).0`）。

### internal/（给项目开发者）

见 [`internal/README.md`](internal/README.md) 的完整索引。要点：

- `internal/18-roadmap.md` — **"某件事该不该做"的唯一入口**（裁定台账 + 验收口径 + 现状实测；不含执行计划）。
- `internal/27-execution-plan.md` — **执行计划（优先级象限）**："下一个先做哪个"。
- `internal/31-rl-training-primitives.md` — **RL 主线与通用训练基础设施**（2026-10-23 ✅ 已随 `v1.12.0` 发布）；调用指南见 `usage/06-reinforcement-learning.md`。
- `internal/00-code-review-remediation.md` — 2026-10-11 代码审查整改立项台账。
- `internal/12` / `17` / `19` / `21` / `22` / `23` — 现行设计/盘点。
- `internal/28` / `29` / `30` — 生命周期 v3 / Session 层 / TrainingSession 单步协议（已随 `v1.10.0`/`v1.11.0` 发布）。
- `internal/02` / `03` / `04` / `26` — 融合/IR/显存/f16 提速计划。
- `internal/14` / `24` — 已修故障报告 / f16 训练缺陷台账。

### 总档

| 文档 | 何时读 |
|------|--------|
| [`history/history.md`](history/history.md) | 查"为什么当初这么改、旧数字是多少"（全仓唯一历史归档） |
| [`../research/README.md`](../research/README.md) | 引用**实测结论**或核对算法定义（MNIST、f16、三值、RLA）：结果带证据/重跑/边界，TODO 不得当结果 |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 参与本项目（构建 / 测试 / 规范 / PR / CI 口径）——**给非作者的入口** |
| [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | 社区行为准则 |
| [`../CMakePresets.json`](../CMakePresets.json) | 预设构建（`cmake --preset dev` 等）——见 CONTRIBUTING §1 |
| [`benchmarks/2026-09-25-vulkan-vs-cuda.md`](benchmarks/2026-09-25-vulkan-vs-cuda.md) | Vulkan 后端 vs CUDA（torch）基准 |
| [`benchmarks/2026-10-06-vulkan-vs-torch.md`](benchmarks/2026-10-06-vulkan-vs-torch.md) | 第二轮复测 |
| `release-notes/v1.5.0.md` … `v1.13.0.md` | 版本变更说明，覆盖 v1.5.0 → v1.13.0 的全部 tag（**大版本发布要求**见 [`contributors/release-requirements.md`](contributors/release-requirements.md)） |

> 历次文档整理的记录（2026-10-11 两轮、2026-11 第三轮）统一在
> [`history/history.md`](history/history.md)；文档规范正文见
> [`contributors/documentation-standard.md`](contributors/documentation-standard.md)。
