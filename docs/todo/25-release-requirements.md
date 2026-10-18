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
| 1 | **端到端模型训练测试** | 至少一条**真实规模**训练跑通且健康（loss 收敛、无 NaN / 无冻结、权重确实移动），并覆盖关键配方（f16 / 三值 / GPU 视当期改动而定）。**这类测试不进默认 ctest**（见 §2），只在本清单实跑一次。**注（2026-10-13）**：`stable=f16` 的「loss 冻结/inf」根因已修（[`24`](24-f16-training-defects.md) §3.1 = CE 的 (1,1) 归约按 f16 落盘溢出），故本项的「无冻结」判据现对 `stable=f16` 也可测；**注（2026-10-14）**：`optimizer=f16` 的 NaN 根因（Adam 二阶矩在 f16 下下溢为 0）已修（[`24`](24-f16-training-defects.md) §4.6 = 优化器状态缩放），故**四字段全 f16 现已可训练**（实测 GPU 500 步 6.05→4.05 无 NaN），本项的「无 NaN/无冻结、权重确实移动」判据对全 f16 同样适用；**注（2026-10-18）**：`26` 号 §8 分两套计划——**Plan A 标准 `--f16`** 的发布门禁是与 f32/PyTorch AMP 同量级、无 NaN；**Plan B 真 f16** 的额外门禁是 capable 设备零转换 / incapable 仅引擎内转换 / 多模型/多设备/长跑。R5 通过前本项只证明“可训练”，不等于“真 f16 可转正”。启动期精度告警（F1）继续作为"显式实验档"的提示保留 | 训练命令 + 步数 + loss 起止值（必要时轨迹）+ 产物 |
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

## 4. 2026-10-18 发布前实跑记录（v1.9.0 发布态复验）

> 口径：`NN_ENABLE_NATIVE=OFF`（可移植基线）；Windows 102 = VS 2026 / cl 19.51 /
Vulkan SDK 1.4.350 / NVIDIA CMP 40HX；Linux 本机 = clang 24 + libstdc++ / GTX 850M。

### 4.1 第 1 项：真实数据集端到端训练

数据集 `datasets/tinystories_200mb.nndataset`（584,568 docs / 124,715,061 tokens），
GPT `d_model=64 / heads=4 / layers=2 / d_ff=128 / seq_len=128 / batch=8`，850M GPU，
每种配方 20 step：

| 配方 | loss 起 → 止 | 结果 |
|---|---|---|
| f32 | 8.7601 → 7.6061 | 健康下降，无 NaN |
| `--f16` | 8.7734 → 7.5770 | 健康下降，无 NaN；`param/compute=f16` |
| `--t1_58` | 8.8839 → 8.0488 | 健康下降，无 NaN；BitLinear 三值权重 |

复现（以 f32 为例）：

```bash
./build/text_train datasets/tinystories_200mb.nndataset --model gpt \
  --d-model 64 --num-heads 4 --num-layers 2 --d-ff 128 \
  --seq-len 128 --batch-size 8 --max-steps 20 --log-interval 5 \
  --gpu=850M --save /tmp/nn_e2e_f32.nnmodel
```

### 4.2 第 2 项：发布文档

- `docs/release-notes/` 已覆盖 **v1.5.0 → v1.9.0 的全部 tag**；`v1.5.1` / `v1.6.1` /
  `v1.7.1` / `v1.8.0` 的独立说明于 2026-10-18 补齐，`v1.9.0` 随本次发布新增。
- `bench/doc_align_audit.ps1` / `bench/doc_inventory.ps1` / `bench/gui_cli_audit.py`
  可行动项均为 **0**，三脚本 exit 0。

### 4.3 第 3 项：Release / Debug ctest

| 树 | 结果 |
|---|---|
| Linux Release + Vulkan（850M） | **55/55 通过**（含 `scan_bitwise_test` / `rapt_test_f16`） |
| Linux Debug + Vulkan | 54/54（`scan_bitwise_test` 因 Debug 非优化构建按 77 skip；`rapt_test_f16` 按既有 C11 口径 `-E` 排除） |
| Windows clang 23.1 + MSVC STL Release（40HX） | **55/55 通过**（`scan_bitwise_test` 按 Windows 口径 skip） |
| Windows cl.exe 19.51 Debug CPU-only | **52/52 通过**（GPU 用例 77 skip；`rapt_test_f16` 按 CI 口径排除） |

### 4.4 第 4 项：四组「编译器 × 标准库」

| 配置 | 结果 |
|---|---|
| WSL clang++ 24 + **libc++**（CPU-only） | **53/53**（18 条 GPU 用例 77 skip） |
| WSL g++ 15.2 + **libstdc++**（CPU-only） | **53/53**（18 条 GPU 用例 77 skip） |
| Windows clang 23.1 + **MSVC STL**（Vulkan / 40HX） | **55/55**（1 条 Windows 口径 skip） |
| Windows cl.exe 19.51 + **MSVC STL**（Debug / CPU-only） | **52/52**（C11 排除 / GPU skip） |

### 4.5 同批收口的缺陷

- **RV-05 / `GpuBackend` 故意泄漏**：已由引擎所有权 v2 修复（`EngineId` + `shared_ptr`
  keepalive，`GpuBuffer` 持 backend 强引用，`reduce_partial_` 改裸 `VkBuffer` 切断环）；
  GPU ctest 55/55，`layer_lifetime_test` 与 `NN_GPU_LIFETIME_FULL=1` 均通过。
- **GPU f16 回退边界 cast 的身份漏 stamp**：`gather` / `rearrange_3d` / `transpose` /
  `im2col` / `col2im` / `matmul` / `batched_matmul` 的“f32 计算 → cast 回 f16”出口在
  `NN_BIND_DEBUG=1` 下暴露未绑定输入；7 处统一改为 `cast(adopt(...), F16)`。修复后
  Release+GPU 的 `NN_BIND_DEBUG=1` ctest **55/55**，CPU-only 同口径 **53/53**。
- **Windows 可移植性**：clang++ 补 `-Wl,/STACK:8388608`；`layer_lifetime_test` 改用
  `nn::detail::env_flag_raw`；CLI 子进程测试使用绝对路径与 cmd 安全引号；三值/模型测试
  不再因 1 MB 默认栈 SegFault。
- **`scan_bitwise_test` 平台口径**：逐位锚以 Linux Release + 850M 为权威；Windows 与
  Debug 构建按 77 skip（数值正确性仍由 `gpu_test` 的容差用例覆盖）。
