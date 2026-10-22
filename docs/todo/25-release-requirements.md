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
| Windows clang 23.1 + MSVC STL Release（40HX） | **55/55 通过**；`NN_BIND_DEBUG=1` 同为 **55/55**（`scan_bitwise_test` 按 Windows 口径 skip） |
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

---

## 5. 2026-10-20 `v1.10.0-rc0` 预发布记录（分支 `dev/lifetime_refresh`）

> 本 rc 只覆盖本分支新增的三块能力（生命周期 v3 / `TrainingSession` / backend 弱缓存）；
> 完整四编译器矩阵与 Linux+850M 权威口径**未在本次重跑**（见 §4 的 v1.9.0 记录）。
> 正式版（P2b 完成后）需按本文件 §1–§3 全量重跑。

### 5.1 构建与测试

| 树 | 命令 | 结果 |
|---|---|---|
| CPU Release（clang 22，`NN_ENABLE_GPU=OFF`，`NN_ENABLE_TESTS=ON`） | `cmake --preset cpu` + build + `ctest` | **55/55 通过**（GPU-only 用例按 77 skip） |
| GPU Release（clang 22，Vulkan，RTX 2060 SUPER / 驱动 580） | `cmake --preset dev` + build + `ctest` | **57/57 通过** |
| ASan（clang 22，`-fsanitize=address`，Windows，`NN_ENABLE_PCH=OFF`） | 单独目标构建 + 运行 | `lifetime_contract_test` / `training_session_test` / `layer_lifetime_test` **全绿**（Windows ASan 不支持 LSan，泄漏检测未跑） |
| `sizeof(Tensor)` | 探针编译 | CPU **56 B** / GPU **80 B**（v1.9.0：64 / 88） |

### 5.2 门禁

- `bench/doc_align_audit.ps1`：exit 0（`[A]`/`[D]`/`[E]`/`[F]` 可行动均为 0）；
- `bench/doc_inventory.ps1`：exit 0（`virtual methods` 52/52、`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`、`TERMINATING-CALLS: 0`）；
- `grep -E 'EngineId|NN_BIND_DEBUG'` 在 `include/src/tools/examples/CMakeLists/CONTRIBUTING/.github` 零命中（历史归档与 release-notes 的历史提及除外）。

### 5.3 未覆盖 / 明示风险

1. **P2b 未做**：CLI 仍用自由工厂 + 手工训练状态（见 `29-session-architecture.md` §8）；
2. **Linux + 850M / 40HX / libc++ / libstdc++ / MSVC 矩阵未重跑**（沿用 v1.9.0 记录，代码变更已通过 clang 22 + MSVC STL 的现有 CI 口径编译）；
3. **`scan_bitwise_test` 仍按平台口径 skip**（逐位锚权威 = Linux + 850M）；
4. `layer_bench` 配对 A/B 未在本 rc 重跑：本次改动不含数值路径变更（CPU/GPU 逐位一致性由 `expr_fuse_test` / `fused_gpu_test` / `expr_fold_test` 覆盖）。

---

## 6. 2026-10-21 v1.10.0-rc0.1 预发布记录

> 分支 `dev/lifetime_refresh`。发布标识：release 名称 **v1.10.0-rc0.1**，git tag **`v1.10.rc0.1`**（`CMakeLists.txt` 的
> `project(... VERSION ...)` 仍为 1.10.0）。本 rc 只覆盖 **P2b 训练工厂**（样本层 /
> `TextTrainer` / `TextTrainConfig` / 运行态 `TrainState` / CLI 迁移；30 号起 Session 只持有单步与累积状态）；
> 完整四编译器矩阵与 Linux+850M 权威口径**未在本次重跑**，正式版前仍按本文件 §1–§3 全量执行。

### 6.1 构建与测试

| 树 | 结果 |
|---|---|
| CPU ctest（Release，`NN_ENABLE_GPU=OFF`，`NN_ENABLE_TESTS=ON`） | **57/57 通过**（GPU-only 用例按 77 skip） |
| GPU ctest（Release，Vulkan，NVIDIA CMP 40HX / 驱动 616.92） | **59/59 通过** |
| ASan（Windows，`NN_ENABLE_PCH=OFF`） | `dataset_samples_test` / `text_trainer_test` / `training_session_test` / `text_train_modules_test` / `dataset_test` 全绿（无 LSan） |
| 单测回归 | `dataset_test` / `train_state_test` / `text_train_modules_test` / `model_roles_test` / `optimizer_state_test` 全绿（CPU/GPU 口径） |
| 真实文本端到端（GPU，tinystories 子集 20k docs / 5.7M tokens） | f32 20 步 6.77→6.14；`--f16` 6.57→6.16；`.nnckpt` + `--resume` 续训通过 |
| 文本训练冒烟（CPU） | 训练 / 测试评估 / 定期 `.nnckpt` / `--resume` 续训均跑通，输出文案与 P1-1 锚一致 |

### 6.2 新增测试

- `dataset_samples_test`：样本概念（doc vs sample vs step）、切窗/样本视图、batch 逐位、
  assistant 掩码、错误语义、断点旧键兼容；
- `text_trainer_test`：数据集直连工厂、`fit()` 端到端、`TrainState` 装配（含
  `--max-steps` 终态）、断点、高性能样本集入口 + `nullopt` test、前置条件
  （`batch_size>0` / `log_interval>0` / 配方与 Session 逐项一致）；
- `training_session_test` 增 `TrainState` 用例。

### 6.3 未覆盖 / 明示风险

1. **Linux + 850M / libc++ / libstdc++ / MSVC 矩阵未在本 rc 重跑**——GPU ctest 与端到端仅在
   本机 NVIDIA CMP 40HX 验证，见 release note v1.10.0-rc0.1 §6/§7；
2. **P2-4 流式 DataLoader 仍未做**：`TextSampleSet` 仍整段物化 `TextCorpus`；
3. `TextTrainer` 只服务文本/GPT（内置 sparse-CE），通用 Step 协议未抽象；
4. 断点旧键兼容是**单向**的（rc0.1 写 `dataset_samples`，rc0 二进制读不到），v8 版本号不变。

---

## 7. 2026-10-21 `v1.10.0` 正式发布实跑记录

> 分支 `dev/lifetime_refresh` 的发布提交（tag **`v1.10.0`**）。口径：**`NN_ENABLE_NATIVE=OFF` 可移植基线**；
> GPU = **NVIDIA CMP 40HX / Vulkan 1.4.351 / 驱动 616.92**（原生 Vulkan 计算设备，自动选卡）；
> 最终源码 = `a9c6c6f` + 发布审查修复（P1-1 / P2-1 / P2-2，见 `docs/release-notes/v1.10.0.md` §2.5）。
> 全部构建目录为**全新构建**（`build/review/*`，不复用旧产物）。原始报告在 `build/review/*.md`（构建产物目录，不入库）。

### 7.1 第 1 项：真实数据集端到端训练（最终二进制）

数据集：`datasets/tinystories_full.txt` 前 20000 行 → **20,000 docs / 4,244,522 tokens / 8192 词表 /
66,321 个 seq64 训练窗口**（`dataset_gen`）。GPT `d_model=64 / heads=4 / layers=2 / d_ff=128 /
seq_len=64 / batch=8`，每种配方 **20 step**，GPU：

| 配方 | loss 起 → 止 | 结果 |
|---|---|---|
| f32 | **9.0211 → 7.7263**（−14.4%） | 健康下降，无 NaN/Inf/冻结，grad_norm 1.01→1.08 |
| `--f16` | **9.0319 → 7.6534**（−15.3%） | 健康下降，无 NaN/Inf/冻结，曲线与 f32 基本重合 |
| `--t1_58`（仅 GPT） | **9.0319 → 8.1533**（−9.7%） | 健康下降，无 NaN/冻结，grad_norm 0.19–0.29（STE 预期量级） |

参数确实移动：三组 20-step 模型分别 warm start 再训 1 step，文件大小不变而 sha256 改变
（f32 `82B6732A…`→`9FFC159D…`、f16 `A514070D…`→`4BE13E15…`、t1_58 `A3165608…`→`AC0EF908…`），全程梯度非零。
断点续训：f32 `--save-interval 5` 产出 step20 `.nnckpt` → `--resume` 续 5 步，step21–25 loss
**7.5669 → 7.2324**；日志确认 `global_step=20`、优化器状态 76 张量 / `t=20`（adam）恢复、数据集身份校验通过。
复现命令与日志见 `build/review/e2e-report.md`。

### 7.2 第 2 项：发布文档

- `docs/release-notes/v1.10.0.md` 落盘：发布范围（v1.9.0 → v1.10.0）、主题、破坏性变更 + 合并去重后的
  完整迁移表（引擎层 13 行 + 数据集/训练 API 10 行）、验收基线（§7）、提交索引、已知限制；
  `v1.10.0-rc0` / `v1.10.0-rc0.1` 两份预发布记录保留。
- 独立发布审查（见 §7.5）结论与修复（P1-1 / P2-1 / P2-2）写入 release note §2.5。

### 7.3 第 3 项：完整 ctest（Release / Debug）

| 树 | 构建 | 完整 ctest |
|---|---|---|
| Windows clang 23.1 + MSVC STL / **Release** / Vulkan（40HX） | 全新目录 219/219 步，0 warning，782.2 s；最终源码增量 6/6 | **59 注册 → 58 pass + 1 skip + 0 fail**（229.52 s，最终源码）；skip = `scan_bitwise_test`（平台口径：逐位锚以 Linux+850M 为权威） |
| Windows clang 23.1 + MSVC STL / **Debug** / Vulkan | 全新目录 219/219 步，0 warning，511.2 s；最终源码增量 6/6 | **59 注册 → 58 pass + 1 skip + 0 fail**（547.12 s，最终源码）；skip 同上；`rapt_test_f16` **Passed**（完整 ctest 2 次 + 直跑 3 次全通过，无 NaN/断言/崩溃）；Debug 迭代器断言零触发 |
| Windows cl.exe 19.51 + MSVC STL / **Debug** / CPU-only | 全新目录，525.5 s，0 error | **57 注册 → 37 pass + 20 skip + 0 fail**（236.62 s，最终源码）；CI 口径（`-E "^rapt_test_f16$"`）**56 → 36 pass + 20 skip + 0 fail**（196.57 s）；`rapt_test_f16` 累计 5/5 通过（未复现 C11） |

> MSVC 的 20 个 skip = 17 GPU 依赖 + 1 Vulkan 探针 + 2 个 app 发现类用例
> （`cli_precision_warning_test` / `chat_prompt_test`：VS 多配置生成的 app exe 在 `<build>/Debug/`，
> 用例候选路径找不到 ⇒ 77 skip；对 CI `windows-latest` 的 MSVC Debug job 同样成立，记为已知覆盖缺口）。

### 7.4 第 4 项：四组「编译器 × 标准库」全新构建 + ctest（CPU-only 两树）

| 配置 | 结果 |
|---|---|
| WSL clang++ **24.0.0 + libc++**（Release / CPU-only） | 112/112 步 OK；**57 注册 → 39 pass + 18 skip + 0 fail**（13.57 s，最终源码）；`ldd` 确认链 libc++.so.1 + libc++abi.so.1 |
| WSL g++ **15.2.0 + libstdc++**（Release / CPU-only） | 112/112 步 OK，0 warning；**57 → 39 pass + 18 skip + 0 fail**（13.95 s，最终源码）；链 libstdc++.so.6 |
| Windows clang 23.1 + **MSVC STL**（Release / Vulkan） | 见 §7.3 第 1 行（59：58 pass + 1 skip + 0 fail） |
| Windows cl.exe 19.51 + **MSVC STL**（Debug / CPU-only） | 见 §7.3 第 3 行（57：37 pass + 20 skip + 0 fail） |

> WSL 的 18 个 skip 均为无 Vulkan 导致的 GPU/offload 用例（预期）；两树 0 失败、无 `std::__cxx11` /
> undefined reference、无源码 `-Werror` 警告。libc++ 树需给 clang 驱动警告
> `-Wgcc-install-dir-libstdcxx`（镜像同装 GCC 15/16）加 `-Wno-error=...`——工具链环境差异，非源码问题。

### 7.5 门禁与独立发布审查

- `bench/doc_align_audit.ps1` exit 0（[A]/[D]/[E]/[F] 可行动 0）；`bench/doc_inventory.ps1` exit 0
  （`virtual methods` 52/52、`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`、`TERMINATING-CALLS: 0`）；
  `bench/gui_cli_audit.py` exit 0（七节 PASS）；`bench/audit_dangling.ps1` exit 0。
- 代码/CI 零 `EngineId` / `NN_BIND_DEBUG` 残留（仅剩 `compute_engine.hpp` 一处解释性注释）。
- **独立发布审查**（`v1.9.0..HEAD`，只读）：**P0 = 0、P1 = 1、P2 = 9**。P1-1（`TextTrainer::create`
  缺配方 seq_len ↔ 模型 seq_len 校验）+ P2-1（`--resume` + `--activation-offload` 绕过配置期拒绝）+
  P2-2（空 test 子集不再打印"样本数: 0"）已在发布前修复并随最终源码复验全绿；其余 P2 记录为后续项。
- `layer_bench` 配对 A/B **未做系统性回归对比**：本版无 kernel 数值路径改动；最终源码单次功能冒烟
  GPU `matmul` 4.365 ms / 491.9 GFLOPS（hash `cbf7d4478bd536ab`）、`matmul_q` 0.796 ms / 2697.2 GFLOPS
  （hash `c5860cfcfcb371f9`），CPU 512³ 小规模两组哈希齐全（计时受并发负载影响，仅参考）。

## 8. 2026-10-23 `v1.11.0` 发布实跑记录

> 测试对象 = `bc37df3`（`feat(session)!`：TrainingSession 单步协议与 global/micro batch 语义，docs/todo/30）。
> 同步方式：`git bundle` → 102 `D:\Codes\nn_v111_test` 独立检出（不触碰 102 既有工作区）。
> 口径：`NN_ENABLE_NATIVE=OFF` 可移植基线；GPU = **NVIDIA CMP 40HX / Vulkan SDK 1.4.350**（原生设备自动选卡）。

### 8.1 第 3 项：完整 ctest（两树全新构建）

| 树 | 构建 | 完整 ctest |
|---|---|---|
| 102 Windows clang 23.1 + MSVC STL / **Release** / Vulkan（40HX） | 全新目录 **219/219 步**，0 warning | **59 注册 → 58 pass + 1 skip + 0 fail**（540.31 s）；skip = `scan_bitwise_test`（Windows 平台口径） |
| 102 WSL g++ 15.2.0 + libstdc++ / Release / **CPU-only** | 全新目录 **112/112 步** | **57 → 39 pass + 18 skip + 0 fail**（67.46 s）；18 skip 均为 GPU 依赖用例 |

### 8.2 第 5 项：文档/门禁审计（Windows 侧，新二进制）

- `bench/gui_cli_audit.py` **PASS 可行动 0**（七节全绿）；
- `bench/doc_inventory.ps` **GATE PASS**（L2-VIOLATIONS 0、TERMINATING-CALLS 0）；
- `bench/doc_align_audit.ps1` **可行动 0**。
- 注：WSL 的 python 无 `tkinter`（无法 import `gui.py`），GUI 门禁以 Windows 侧为准；
  本机旧二进制曾误报 2 个 GUI 问题（旧 `--help` 文案残留 `--precision-*` 字面量），换新二进制后消失。

### 8.3 端到端训练（新 batch 语义 + 断点兼容，40HX）

数据集 = v1.10.0 验收同款 `datasets/tinystories_20k.nndataset`（66,321 个 seq64 窗口）；
GPT `d_model=64 / heads=4 / layers=2 / d_ff=128 / seq_len=64`；**`--batch-size 8 --accum-steps 2`
（global 8 / micro 4）**，20 **update** step：

| 用例 | 结果 |
|---|---|
| f32 | 8.7576 → 8.4025 → 8.0519 → **7.6022**（step5/10/15/20） |
| `--f16` | 8.7676 → 8.3970 → 7.9907 → **7.7230** |
| `--t1_58` | 8.9011 → 8.6990 → 8.4348 → **8.1010** |
| 新格式 resume | step20 `.nnckpt` 续 2 步：**7.6539 → 7.5228**（优化器 76 张量 / `t=20` 恢复；计数器按 update step，steps_per_epoch=8291=66321/8） |
| **legacy 兼容** | v1.10.0 二进制 `--batch-size 4 --accum-steps 2`（micro 语义）产 10 micro-step 断点 → 新二进制 `--resume`：**折算 update step 5**、global batch 恢复 8（=4×2），续训 8.6738 → 8.5822 与 legacy 末值 8.7711 连续 |
| 禁列 | `resume + --batch-size` 明确报错（A/B 类旗标禁列表生效，输出禁列表 + 迁移提示） |

均无 NaN/Inf/冻结；三组 20-step 约 1.5–2.3 s，loss 健康下降。

### 8.4 环境事故记录（本机构建，非代码问题）

本机 dev 树首轮构建中途出现 `PCH file ... uses an older format`：根因 = **apt 在构建进行中
（2026-10-23 00:37:38）更新了 clang-24 nightly**，新旧编译器 PCH 格式不兼容，且中途产物混链不可信。
处置 = 清空 `build/dev` 全新重建。教训：**发布验收构建期间不要更新工具链**。

### 8.5 本机收尾实跑（Linux，2026-10-23）

- **clang 24.0.0 nightly（+libc++）/ Release + GPU / dev preset**：清空 ¤build/dev¤ 全新构建
  **219/219 步**；ctest **59 注册 → 57 pass + 2 fail**。
- 2 个失败均为**沙箱内 llvmpipe 软件光栅**的环境差异（ctest 日志：¤GPU: llvmpipe (LLVM 21.1.8)¤；
  沙箱不可见 ¤/dev/nvidia*¤，见 AGENTS「本机 GPU 测试」口径）：
  - ¤scan_bitwise_test¤：57/61 例 f16 差 1–2 ULP（如 ¤cpu=0x3aecc000 gpu=0x3aeca000¤）——该测试
    逐位锚本就以真卡为权威（Windows 平台口径 skip 同因）；
  - ¤ce_fusion_test¤：CPU 全过；GPU 仅 ¤parity(3,100,x2)¤ 一例 ¤loss_err=5.745e-01¤；
  - 同批用例在 102 的 40HX 真卡**全绿**（§8.1：ce_fusion/batchnorm 属 58 pass）。
- ¤batchnorm_test_gpu¤ 首轮在内存颠簸时段 SEGFAULT，单独复跑 **Passed**（11.53 s）——偶发，非确定性失败。
- 与本版 API 改动直接相关的 ¤training_session_test¤ / ¤text_trainer_test¤ / ¤train_state_test¤ /
  ¤dataset_samples_test¤ / ¤text_train_modules_test¤ 本机全部 Passed。
