# development/ —— 面向调用者的开发文档

> **本分册写给调用者（库使用者 / 扩展者）**，不是项目开发者的工作区。
> 内部设计稿、路线图、盘点、规范、踩坑、故障报告、审查整改一律在 [`../todo/`](../todo/)；
> 历史叙事在 [`../history.md`](../history/history.md)。

## 这里的文档解决什么问题

"我要拿着这个库写自己的东西"——从**把库接进工程**、**用全套 C++ API 写程序**，
到**扩展它**（自定义层 / 新引擎原语 / 自研融合算子）：

- **接入与构建**：CMake 两行嵌入、全部构建开关、GPU 语义、自研算子融合注册；
- **用全库能力**：引擎 / 张量 / 精度 / DSL / 层 / 模型 / 损失 / 优化器 / 训练 /
  模型文件 / 数据集 / 分词器 / 生成 / 剖析，逐节附可粘贴代码片段；
- **写自定义层**（`Layer` 子类的接入契约、参数协议、精度槽声明、数值梯度自检）；
- **扩展引擎原语**（给 CPU/GPU 加一个新的 `op`，两后端如何各自落地、如何进 AOT 闭合世界）；
- **选精度策略**（`PrecisionProfile` 四槽语义、f16 的边界 cast 与已知数值限制）。

## 索引

**入门路径建议**：先读 [`cpp-api-manual.md`](cpp-api-manual.md) §0 的「我要做什么 → 看哪节」表，
按任务跳到对应章节；要改构建就再读 [`cmake-integration.md`](cmake-integration.md)。

| 文档 | 何时读 |
|------|--------|
| [`cpp-api-manual.md`](cpp-api-manual.md) | **全功能 C++ API 手册**（首选）：错误处理宏、引擎、张量、精度、DSL、层与 Model、模型工厂、损失、优化器、训练循环、模型文件、数据集、分词器、生成、剖析——每节带代码片段 |
| [`cmake-integration.md`](cmake-integration.md) | **接入与构建**：`add_subdirectory`/`FetchContent`、全部构建开关、GPU 语义、`nn_enable_gpu_fusion`、本仓构建/测试、跨平台、故障排查 |
| [`write-a-custom-layer.md`](write-a-custom-layer.md) | **写自己的层**：完整示例（实测通过数值梯度检查）+ 七条接入契约 + 精度槽声明 + 可复制的梯度自检 + GPU 检查清单 + 常见编译错误 |
| [`01-compute-engine-development.md`](01-compute-engine-development.md) | 想**扩展引擎**（加新原语）而不只是使用它：`ComputeEngine` 接口全解、CPU/GPU 实现模式、加原语的完整步骤 |
| [`05-mixed-precision.md`](05-mixed-precision.md) | 多精度**语义契约**（f16 的 P 精度算术、`PrecisionProfile{param/compute/stable/optimizer}`）+ 当前实测结论与已知数值限制（§12） |
| [`06-rapt-algorithm.md`](06-rapt-algorithm.md) | 线性注意力家族（RLA / RAPT / RLA-2）：直观理解 + 数学定义 + 扫描原语工程落地 |
| [`07-vulkan-backend-map.md`](07-vulkan-backend-map.md) | **维护者视角**：Vulkan 后端的调用链、所有权、依赖方向，以及"加一个算子/pipeline 要改哪几处" |
| [`08-object-lifetimes.md`](08-object-lifetimes.md) | **维护者视角**：v3 转发页（唯一真源 = `09`：设备域从存储派生 / 入口域检查 / 跨域迁移清单） |
| [`09-engine-ownership.md`](09-engine-ownership.md) | 引擎/backend 所有权设计全文：不变量、改动范围、API 影响、性能口径、实施顺序 |
| [`20-custom-layer-ergonomics.md`](20-custom-layer-ergonomics.md) | 自定义层的**实测报告**（库外使用者视角）：7 个摩擦点、逐条坐标。写层前读 [`write-a-custom-layer.md`](write-a-custom-layer.md)，本文是它的背景证据 |

> **命名约定**：带数字前缀的文件沿用历史设计稿编号（`01` / `05` / `06` / `20`）；
> 新增的**使用类文档**改用描述性文件名（`cpp-api-manual` / `cmake-integration` /
> `write-a-custom-layer`），避免与已归档的编号混淆。

## 相关（相邻分册）

| 想去哪 | 文档 |
|--------|------|
| 只是**用引擎**（张量/矩阵/表达式） | [`../usage/03-compute-engine-usage.md`](../usage/03-compute-engine-usage.md) |
| 把库**嵌进自己工程** | [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md) |
| 跑现成 CLI / GUI 训练推理 | [`../usage/02-quickstart-train-infer.md`](../usage/02-quickstart-train-infer.md) |
| 看**数据流/AOT 融合**的整体设计 | [`../introduction/01-architecture.md`](../introduction/01-architecture.md)、[`../introduction/04-innovative-designs.md`](../introduction/04-innovative-designs.md) |
| 融合/IR 的**内部设计**（改库才需要） | [`../todo/02-operator-fusion.md`](../todo/02-operator-fusion.md)、[`../todo/03-ir-optimization.md`](../todo/03-ir-optimization.md) |

> ⚠ 本分册的文档同样受仓库写作纪律约束：**只描述当前状态**；演进过程、被否决方案、
> 旧数字一律在 [`../history.md`](../history/history.md)。
