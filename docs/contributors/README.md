# contributors/ —— 贡献者与仓库作者规范

> **本分册写给向本仓贡献的人**：提 PR 的外部贡献者、仓库维护者、AI 编码助手（agent）。
> 与相邻分册的分工：**调用者**（拿库写自己东西的人）看 [`../usage/`](../usage/) 与
> [`../development/`](../development/)；**项目内部设计/台账**看 [`../internal/`](../internal/)；
> 历史归档在 [`../history/history.md`](../history/history.md)。
>
> 一句话判断：**这篇规则管的是"怎么参与本仓的开发与治理"，还是"怎么用这个库"？**
> 前者住这里，后者住 `usage/` / `development/`。
> **读者**：向本仓贡献代码的人（含维护者与 AI 编码助手）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
> **状态**：现行。

## 这里的文档怎么组织

- **长期真源**：本分册放的是**长期有效的规范与治理规则**，不是一次性台账。
  临时性的设计稿/整改立项进 [`../internal/`](../internal/)。
- **别处只留引用**：`CONTRIBUTING.md`、`AGENTS.md` 等入口文档对本分册**只留摘要与指针**，
  规则正文一律以这里为准（避免同一规则多处维护、互相漂移）。
- **无编号主题名**：本分册文档用主题名（`cpp-standards.md`），不用数字编号——
  编号制度见 [`documentation-standard.md`](documentation-standard.md) §1。

## 索引

| 文档 | 何时读 |
|------|--------|
| [`documentation-standard.md`](documentation-standard.md) | **文档规范**：分册职责、编号制度、单篇模板（读者/结论/状态）、代码示例与 API 标注、状态词表、机器审查口径。**写任何文档前读** |
| [`cpp-standards.md`](cpp-standards.md) | **C++ 开发规范**：内存管理 / 模块隔离 / 高性能 / 错误处理 / 注释——改代码前读 |
| [`pitfalls.md`](pitfalls.md) | **踩坑警示录**：分级 bug 与跨领域根因模式——改代码前读；**踩了新坑先补这里再修代码** |
| [`release-policy.md`](release-policy.md) | **版本号·分支·发布治理（三条红线）**：`v1.<API 变化>.<漏洞修复>`、**发布版本号必须由人提出（agent 不得擅自发布）**、`dev` 不写版本号、发布只在 `main`。定版本号 / 发版 / 改分支策略前读 |
| [`release-requirements.md`](release-requirements.md) | **大版本发布要求**（`1.x.y → 1.(x+1).0`）：端到端模型训练测试 / 发布文档 / 完整 ctest + Debug 构建 / 多编译器 × 标准库矩阵 |

## 相关（相邻分册 / 总档）

| 想去哪 | 文档 |
|--------|------|
| 提 PR / 构建测试口径 | [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) |
| AI 开发速览 | [`../../AGENTS.md`](../../AGENTS.md) |
| 总索引与分册定位 | [`../README.md`](../README.md) |
| 项目内部设计 / 台账 | [`../internal/README.md`](../internal/README.md) |
