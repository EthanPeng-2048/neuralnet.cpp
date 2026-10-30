# 版本号、分支与发布治理规范

> **读者**：项目维护者 / 发布者与 AI 编码助手（agent）。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。
> 关联：[`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) §4（分支模型表 / 提 PR）、
> [`../release-notes/README.md`](../release-notes/README.md) §1（版本号格式与位定义全文）、
> [`release-requirements.md`](release-requirements.md)（大版本必做清单）、
> [`18-roadmap.md`](../internal/18-roadmap.md) §2（**git tag 是版本权威**）。
>
> `AGENTS.md` 与 `CONTRIBUTING.md` 对本文**只留引用**；三条规则的正文以本文为准。
> **状态**：现行。

## 0. 三条规则（正文见 §1–§3）

1. **版本号规则**：`v1.<API 变化>.<漏洞修复>`；版本权威 = `git tag`。
2. **发布必须由人提出**：定哪个号、何时发，由人明确给出；**agent 不得擅自决定版本号或执行发布**。
3. **版本号与分支绑定**：`dev` 及功能分支**从不写版本号**；定档与发布**只发生在 `main`**。

## 1. 版本号规则

格式 `v1.<API 变化>.<漏洞修复>`（例 `v1.13.0`、`v1.8.2`）。位定义 / rc / 勘误 / 子仓同规范的
**全文**在 [`../release-notes/README.md`](../release-notes/README.md) §1；本节只复述治理相关部分：

| 位 | 名称 | 递增条件（摘要，全文见 release-notes/README §1） |
|---|---|---|
| 第 1 位 | 主版本 | 极少动：模型文件格式不兼容、项目定位级重构 |
| 第 2 位 | **API 变化** | 对外接口（头文件 API / CLI 选项 / 文件格式键 / CMake 开关）新增或语义变更；**破坏性变更只允许出现在这一位**，release note 须附迁移章节 |
| 第 3 位 | **漏洞修复** | 不改 API 的缺陷 / 数值 / 性能修复；行为只允许从"错"变"对" |

- **版本权威 = `git tag`**：`CMakeLists.txt` 的 `project(... VERSION ...)` 与 `docs/release-notes/v*.md`
  都向 tag 对齐；一个正式 tag 配一册 release note。
- **已发布版本不重打**：tag 发布即冻结；问题以勘误块追加在该册顶部并指向修复版本。
- **当前最新版本不在本文维护**（写死号必然过时），查法：
  ```bash
  git tag --list 'v1.*' | sort -V | tail -1     # 最新正式版
  ```
  历史遗留的非规范 tag（拼写错误的 `v.1.3.0`、`nightly`、`archive/*`、`0.1.0-Transformer` 等）
  **不属版本序列**，不作版本权威，也不得仿照其命名。

## 2. 发布必须由人提出（agent 不得擅自发布）

1. **版本号由人定档**：发版升哪一位、用正式号还是 rc、何时发，必须由**人**（维护者）在
   issue / 讨论 / 明确指令里写出；agent（AI 编码助手）**不得**自行推断或决定版本号。
2. **人未给出明确的版本号与发布指令前，agent 禁止执行任何"把版本号变成发布事实"的动作**：
   - `git tag` / `git push --tags` / `gh release create|edit|publish`；
   - 改 `CMakeLists.txt` 的 `project(VERSION ...)` 为新的定档值；
   - 在 `docs/release-notes/` 新建 `v*.md` **定稿**（草案可写，但必须标注"未定档"）。
3. **人发出指令后 agent 只是执行者**：可按 [`release-requirements.md`](release-requirements.md)
   清单代跑，但执行**前**须复述将用的版本号与动作清单并得到确认；号与内容不一致时**停下报告**，
   不得自行改号或"顺手"发版。
4. **清单全绿 ≠ 可发布**：必做项全部就绪也不产生发布权——没有人的定档指令就不发。

> 违反口径：agent 擅自打出的 tag / release / 定稿 release note 一律视为无效产物，由人删除后
> 按本文重走；事件记入 `docs/history/history.md`。

## 3. 开发分支与版本号的绑定

分支模型表（`main` / `dev` / `feature/<name>` / `hotfix/<name>` 的定位与流转）见
[`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) §4；本节只钉**版本号落在哪条分支**：

1. **`dev` 及其上的功能分支从不写版本号**：
   - 不 bump `CMakeLists.txt` 的 `project(VERSION ...)`；
   - 不新建 / 不定稿 `docs/release-notes/v*.md`；
   - 不打任何 tag。
   "下一版可能是几号"只允许作为**讨论**出现在 issue / PR 描述里，不落进分支内容。
2. **版本号必须在 `main` 分支发布**：定档、打 tag、发 GitHub release、同步
   `project(VERSION ...)`、release note 定稿——全部只发生在 `main`（发布基线 = `main`，tag 打在这里）。
3. **流转顺序**：`feature/*` / `hotfix/*` → 合入 `dev` → 经审核合入 `main` → **在 `main` 上**按
   §1 定格式、按 §2 由人发起发布。
4. 纯文档改动（含本文）可直接在 `main` 提 PR：文档不携带版本号，不受"dev 不写版本号"约束。

## 4. 真源表（谁管什么）

| 主题 | 真源 |
|---|---|
| 版本号格式 / 位定义 / rc / 勘误细则 | [`../release-notes/README.md`](../release-notes/README.md) §1 |
| 分支模型（分支表与流转） | [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) §4 |
| 版本号·分支·发布治理（本文三条规则） | 本文（`docs/contributors/release-policy.md`） |
| 大版本发布必做清单 | [`release-requirements.md`](release-requirements.md) |
| 版本权威与现状实测 | `git tag`；[`18-roadmap.md`](../internal/18-roadmap.md) §2 |
| 发布范围 / 主题 / 迁移指南 | [`../release-notes/`](../release-notes/) 每版一册 |
