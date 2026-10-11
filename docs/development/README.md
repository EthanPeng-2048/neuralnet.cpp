# development/ —— 面向调用者的开发文档

> **本分册写给调用者（库使用者 / 扩展者）**，不是项目开发者的工作区。
> 内部设计稿、路线图、盘点、规范、踩坑、故障报告、审查整改一律在 [`../todo/`](../todo/)；
> 历史叙事在 [`../history.md`](../history.md)。

## 这里的文档解决什么问题

"我要拿着这个库写自己的东西"：

- **写自定义层**（`Layer` 子类的接入契约、参数协议、精度槽声明、数值梯度自检）；
- **扩展引擎原语**（给 CPU/GPU 加一个新的 `op`，两后端如何各自落地、如何进 AOT 闭合世界）；
- **接 GPU 融合**（`nn_enable_gpu_fusion`、`dsl::compute` 的表达式写法）；
- **选精度策略**（`PrecisionProfile` 四槽语义、f16 的边界 cast 与已知数值限制）。

## 索引

| 文档 | 何时读 |
|------|--------|
| [`01-compute-engine-development.md`](01-compute-engine-development.md) | 想**扩展引擎**（加新原语）而不只是使用它：`ComputeEngine` 接口全解、CPU/GPU 实现模式、加原语的完整步骤 |
| [`05-mixed-precision.md`](05-mixed-precision.md) | 多精度**语义契约**（f16 的 P 精度算术、`PrecisionProfile{param/compute/stable/optimizer}`）+ 当前实测结论与已知数值限制（§12） |
| [`06-rapt-algorithm.md`](06-rapt-algorithm.md) | 线性注意力家族（RLA / RAPT / RLA-2）：直观理解 + 数学定义 + 扫描原语工程落地 |
| [`20-custom-layer-ergonomics.md`](20-custom-layer-ergonomics.md) | 自己写一个层：以"库外使用者"身份实测的接入契约、7 个摩擦点与必须先读源码才能确定的点 |

## 相关（相邻分册）

| 想去哪 | 文档 |
|--------|------|
| 只是**用引擎**（张量/矩阵/表达式） | [`../usage/03-compute-engine-usage.md`](../usage/03-compute-engine-usage.md) |
| 把库**嵌进自己工程** | [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md) |
| 看**数据流/AOT 融合**的整体设计 | [`../introduction/01-architecture.md`](../introduction/01-architecture.md)、[`../introduction/04-innovative-designs.md`](../introduction/04-innovative-designs.md) |
| 融合/IR 的**内部设计**（改库才需要） | [`../todo/02-operator-fusion.md`](../todo/02-operator-fusion.md)、[`../todo/03-ir-optimization.md`](../todo/03-ir-optimization.md) |

> ⚠ 本分册的文档同样受仓库写作纪律约束：**只描述当前状态**；演进过程、被否决方案、
> 旧数字一律在 [`../history.md`](../history.md)。
