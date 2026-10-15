# 25 — 版本发布要求（大版本必做清单）

> **读者**：项目维护者 / 发布者。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。
> 关联：[`18-roadmap.md`](18-roadmap.md)（**git tag 是版本权威**）；
> [`24-f16-training-defects.md`](24-f16-training-defects.md)（F2 的端到端全 f16 训练归口本清单第 1 项）。

## 0. 适用范围

**触发条件**：仅当**大版本**升级时执行——版本号 `1.x.y → 1.(x+1).0`（次版本号 +1、修订号归零）。
小版本 / 修订号升级（`1.x.y → 1.x.(y+1)`）不强制走本清单，按常规 PR 门禁（构建 + `ctest` + 文档审计）即可。

**版本权威**：`git tag`；`CMakeLists.txt` 的 `project(... VERSION ...)` 与之对齐
（口径见 [`18-roadmap.md`](18-roadmap.md) §2）。发布说明落在 `docs/release-notes/`。

## 1. 必做项（大版本发布前逐条完成）

| # | 项目 | 要求 | 证据（记入 release note 或 PR） |
|---|---|---|---|
| 1 | **端到端模型训练测试** | 至少一条**真实规模**训练跑通且健康（loss 收敛、无 NaN / 无冻结、权重确实移动），并覆盖关键配方（f16 / 三值 / GPU 视当期改动而定）。**这类测试不进默认 ctest**（见 §2），只在本清单实跑一次。**注（2026-10-13）**：`stable=f16` 的「loss 冻结/inf」根因已修（[`24`](24-f16-training-defects.md) §3.1 = CE 的 (1,1) 归约按 f16 落盘溢出），故本项的「无冻结」判据现对 `stable=f16` 也可测；**注（2026-10-14）**：`optimizer=f16` 的 NaN 根因（Adam 二阶矩在 f16 下下溢为 0）已修（[`24`](24-f16-training-defects.md) §4.6 = 优化器状态缩放），故**四字段全 f16 现已可训练**（实测 GPU 500 步 6.05→4.05 无 NaN），本项的「无 NaN/无冻结、权重确实移动」判据对全 f16 同样适用；启动期精度告警（F1）继续作为"显式实验档"的提示保留 | 训练命令 + 步数 + loss 起止值（必要时轨迹）+ 产物 |
| 2 | **发布文档** | 落 `docs/release-notes/v1.(x+1).0.md`：发布范围（上一 tag → 本 tag）、主题、**破坏性变更 + 迁移指南**、验收基线 | release-note 文件 |
| 3 | **完整 ctest + Debug 构建** | `Debug` 配置下构建并跑**完整 ctest**、全绿（需 GPU 的用例按退出码 77 记 skip）；Release 全套 ctest 同样绿 | `cmake --preset debug`（或 `-DCMAKE_BUILD_TYPE=Debug -DNN_ENABLE_TESTS=ON`）+ `ctest` 输出 |
| 4 | **多编译器 × 标准库编译测试** | 至少 **4 组**「编译器 × 标准库」**各自全新构建 + ctest**：clang + libc++、g++ + libstdc++、clang + MSVC STL、cl.exe + MSVC STL。MSVC 必须带 `/Zc:preprocessor`（已写进 `CMakeLists.txt` 的目录级与接口选项） | 每组 configure / build / ctest 结果（口径见 `AGENTS.md` §2） |

## 2. 为什么端到端训练不进默认 ctest

ctest 是 CI **每次 push 都跑**的默认门禁，要求**分钟级**跑完。模型端到端训练（尤其全 f16 / 大词表形态）
耗时远超该预算，塞进 ctest 会拖垮 CI 并诱发超时 flaky。故按耗时分层：

- **默认门禁（ctest）** 只放**廉价**断言——组合矩阵、数值不变量、单步 / 短链轨迹（见
  [`24`](24-f16-training-defects.md) F2）；
- **端到端训练**（含全 f16 可用性核对）归口**本清单第 1 项**，在大版本发布时实跑一次。

## 3. 与常规门禁的关系

- 常规 PR 门禁（构建 + `ctest` + `NN_BIND_DEBUG=1 ctest` + 文档审计）见
  [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) §6 与 [`18-roadmap.md`](18-roadmap.md) §8。
- 本清单是**超集**：大版本在常规门禁之上，再要求第 1 项（端到端训练）与第 4 项（多编译器矩阵）。
