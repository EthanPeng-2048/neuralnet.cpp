# 开发路线图与未完成项裁定（2026-10-01）

> **本文是"下一步该做什么"的唯一入口**：把 `docs/development/13-refactor-backlog.md` 的未完成条目
> 与未来 6–12 个月的方向合并为一份可执行的裁定台账。
> 本轮 13 号文档改为重定向 stub；已完成项、被否决方案与历史取舍统一归档在 `docs/history.md`。
> 底层契约的已落地部分（统一总纲 M1–M6）见 `docs/development/17-unified-tensor-engine.md`。
> 写作纪律：只写事实与决定；行号取自 2026-10-01 实测，未能确证的位置只给文件名、不给行号。

## 0. 一句话

未来一年两条主线：**① 把工程门禁与生态缺口补成真的**（CI 不跑测试、文档自相矛盾、无安装导出、
推理无采样截断）；**② 模型能力按"有真实调用方 / 有对比数据"再立项**，不为可能性写代码。

## 1. 定位与用法

| 文档 | 角色 | 何时读 |
|---|---|---|
| **本文（18）** | 未来 6–12 个月方向 + 未完成项裁定台账 | 想知道"下一个该做什么"时 |
| `docs/development/13-refactor-backlog.md` | **重定向 stub**：条目原文已合并进本文，历史执行记录在 `docs/history.md` | 只在需要追溯条目来源时 |
| `docs/history.md` | 全仓唯一历史归档：演进记录 / 已修复勘误 / 被否决方案 / 性能 A/B | 查"为什么当初这么改、旧数字是多少" |
| `docs/development/17-unified-tensor-engine.md` | **已落地**的底层总纲（M1–M6 全部实施；仅 M7 未立项） | 改 Tensor/Engine/存储前 |
| `docs/development/15-computeengine-refresh.md`、`16-computeengine-p0-inventory.md` | 17 的前置设计与 P0 盘点，历史引用 | 只做证据追溯 |

### 1.1 立项三问（三问都答不上来就不立项）

1. **触发条件是否已满足**（§6 触发式立项清单给了精确判据）？
2. **是否已被否决**（§5 裁定表的 16 条 + §7 非目标）？
3. **验收口径能否复现**（§8 门禁与验收基线）？

### 1.2 与 17 的关系

17 负责"底层已经是什么样"（不变量、契约、已实施分期）；本文负责"上层接下来做什么"。
凡属 17 范围内的未决点（如 M7），本文只做**立项与否的裁定**，实施设计回 17。

## 2. 现状画像（2026-10-01 实测）

| 维度 | 实测 | 复现 |
|---|---|---|
| `include/` | **58 个 `.hpp` / 35637 行**（非空 32594） | `Get-ChildItem include -Recurse -Filter *.hpp` + 逐文件 `Get-Content` 计数 |
| `src/` | **55 个 `.cpp` / 17873 行** | 同上换 `src -Filter *.cpp` |
| `shaders/` | **19 个 `.comp`** | `Get-ChildItem shaders -Recurse -Filter *.comp` |
| `docs/` | **29 篇 `.md`**（含 `history.md` 与 `release-notes/`） | `Get-ChildItem docs -Recurse -Filter *.md` |
| 最大源文件 | `compute_vk_backend.hpp` **5161** 行、`expr_glsl_gen.hpp` **2806**、`compute_cpu_engine.hpp` **2309**、`compute_engine.hpp` **2003**、`gui.py` **1662**（非空约 1470）、`src/text_train.cpp` **1493** | 逐文件 `(Get-Content $f).Count` |
| ctest | **27 个测试** = 24 个测试目标（`list(APPEND NN_TEST_TARGETS` 计数）+ `cnn_test_gpu`（= `cnn_test --gpu`）+ `batchnorm_test_gpu`（= `batchnorm_test --gpu`，2026-10-03 随 BatchNorm 落地新增）+ `fusion_custom_layer_example`（AOT 融合端到端） | `ctest --test-dir build -N` |
| 引擎接口 | **49 个 virtual 方法**；`L2-VIOLATIONS: 0`；宿主桥 40 处（仅披露） | `pwsh -File bench/doc_inventory.ps1` |
| 版本 | git tag **v1.7.0**；`CMakeLists.txt` 的 `project(... VERSION 1.7.0)`（2026-10-06 发布时对齐；口径仍为 **git tag 是版本权威**）；`release-notes/` 有 `v1.5.0.md`、`v1.6.0.md`、`v1.7.0.md`（v1.5.1 / v1.6.1 无独立发布说明，v1.6.1 的用户可见变化补记于 v1.7.0.md §7） | `git describe` + 读 `CMakeLists.txt` |
| CI | 仅 `.github/workflows/cmake-single-platform.yml`：clang++ / Ninja / Release / `NN_ENABLE_NATIVE=OFF`，Linux + Windows；**不开 `NN_ENABLE_TESTS`、不跑 ctest** | 读该 workflow |

> 口径说明：表中行数均为**总行数**（2026-10-06 统一数据集落地后实测刷新）。
> `27` 是当前唯一正确计数（2026-10-05/06 随统一数据集落地新增 `kvrec_test`/`nnvocab_test`/
> `dataset_test`：24 → 27；此前 2026-10-05 起为 24 = 随错误处理宏族落地新增 `error_macro_test`；
> 再此前 2026-10-03 起为 23 = 随 BatchNorm 落地新增 `batchnorm_test` + `batchnorm_test_gpu`，
> 更早为 21）。历史上仓库里同时存在"ctest 计数 19"与"ctest 20 比 20"两种表述（§3 工程化第 4 条）
> ——那是**计数口径不同**（是否含 `cnn_test_gpu`、以及后来新增的目标），2026-10-01 的 A4/A5 两轮
> 把计数推到 21 并统一到本文与 `AGENTS.md`。

### 2.1 已经稳固的底座（本轮不再动）

| 底座 | 状态 | 依据 |
|---|---|---|
| 统一 Tensor / ComputeEngine 契约（M1–M6） | 全部实施 2026-09-30 | `docs/development/17-unified-tensor-engine.md` §5 |
| 访问不变量（Tensor 存储私有、一切经引擎） | 铁律 #11，编译期强制 | `AGENTS.md` §5 |
| L2+ 计算路径禁用 `Matrix` | 铁律 #12，审计口径 `L2-VIOLATIONS: 0` | `AGENTS.md` §5；`bench/doc_inventory.ps1` 第 [4] 节 |
| AOT 闭合世界 + 单 fold 注意力 kernel | 现行；掩码 × 位置偏置两正交维度 | `docs/development/02-operator-fusion.md` |
| 位置编码多态统一（按注入点分所有权） | 现行；新增 `compute_position_encoding.hpp` | `docs/history.md`「Layer 层简化轮」条 |
| BPE 保序并行 encode（任意并行度逐字节一致） | 现行 | `AGENTS.md` §12 |
| 声明式初始化 `InitSpec`（初值跨进程确定） | 现行 | `docs/development/17-unified-tensor-engine.md` §4.4 |
| CPU/Vulkan 双后端 + 多设备选择 | 现行 | `AGENTS.md` §2 |

**含义**：本文的路线图不需要再排"底层重构"——底层已完成；缺口集中在工程门禁与上层生态。

## 3. 缺口总表

每条 = 一句话现象 + 坐标 + 影响面。**坐标只写在已实测确认的位置。**

### 3.1 正确性

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| C1 | GPU `stable=f16` 训练 loss 回读冻结（权重照常更新） | `docs/development/14-f16-stable-gpu-loss-frozen.md` | GPU f16 训练的 loss 曲线不可观测；无对应 ctest 用例 |
| C2 | `optimizer=f16` 无 loss scaling 即更新爆炸、`stable=f16` 约 200 步 NaN | `docs/development/05-mixed-precision.md` §12.5；`src/text_train.cpp` 解析处只赋值不校验 | 用户可组合出必崩配置，且无任何启动期提示 |
| C3 | `NormType::BatchNorm` **静默回落**到 LayerNorm | `compute_layer_mlp.hpp:782`（`make_norm_layer`）；`README.md:248` 声称支持；`gui.py:60` 把 `batchnorm` 列为可选项；`model_spec.hpp:43` 保留枚举 | 模型规格写 batchnorm、实际跑 layernorm，无告警、无日志。**✅ 已解决（2026-10-03）**：真正实现 `BatchNorm` 层（P0-3），`make_norm_layer` 真分支，见 `docs/history.md` |
| C4 | checkpoint（`.bin`）不存 Adam m/v | `model_serialization.hpp`（`MODEL_VERSION = 5`，见该文件版本常量注释）；`compute_optimizer.hpp` 无 `save_state/load_state` | TDR 重启 / `--resume` 后动量归零，续训轨迹不连续 |
| C5 | `.bin` 无整文件校验和与尾部完整性标记 | `model_serialization.hpp` 版本常量附近的注释（明示"若引入需升 MODEL_VERSION"） | 截断/损坏的模型文件无法与合法文件区分 |
| C6 | `gpt_test` 偶发失败未定位 | 13 原条目 §9.5（62 次直跑 1 次失败、日志被覆盖） | CI 化之后会成为随机红 |
| C7 | RAPT / CNN 未接 f16：`Conv2D` 权重/偏置/梯度硬编码 `Precision::F32` | `compute_layer_conv.hpp:148`（权重）、`:151`（偏置）、`:154`/`:155`（权重/偏置梯度）；RAPT 头内 32 处 `dsl::compute*` 调用未接精度实参 | f16 只覆盖 MLP/GPT 路径，CNN/RAPT 用户得不到显存收益 |
| C8 | `host_bridge_uses` 计数与文档不一致（实测 40，13/17 记 41） | `pwsh -File bench/doc_inventory.ps1` 第 [4] 节 | 审计数字失去可比性（仅披露项，不是违规） |

### 3.2 工程化

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| E1 | CI 是唯一自动门禁，却**只编译生产目标、不编译测试、不跑 ctest** | `.github/workflows/cmake-single-platform.yml`（configure 无 `NN_ENABLE_TESTS`，无 ctest 步骤） | 所有测试结论只存在于本地；回归可在无人察觉时合入 |
| E2 | 无 `install()` / export，无 `CMakePresets.json` | `CMakeLists.txt` 实测 `install(` 0 处；根目录无 `CMakePresets.json` | **嵌入路径已完成（2026-10-04）**：门面目标 `neuralnet::nn` + `NN_ENABLE_GPU`/`NN_BUILD_APPS` 开关 + `nn_enable_gpu_fusion` 树外可用（见 `docs/usage/05-consume-as-library.md`、样例 `examples/downstream/`）；**剩余**：`install()`/`find_package` 与 `CMakePresets.json`（用户裁定本轮不做） |
| E3 | `CMakeLists.txt` 版本与 git tag 不一致，且缺 v1.5.1 release note | `CMakeLists.txt` 的 `project(...)` 行（**已对齐 v1.6.0，2026-10-04**）；`docs/release-notes/` 仍缺 v1.5.1 | 版本可读性缺失；分发产物无对应变更说明（release note 缺口仍在） |
| E4 | 文档内 ctest 计数曾自相矛盾（19 vs 20） | `AGENTS.md` §12 已统一为实测 21（19 目标 + `cnn_test_gpu` + `fusion_custom_layer_example`） | 已修；口径差异来源见 §2 口径说明 |
| E5 | `src/text_train.cpp` 1746 行、34 处 `std::exit` | 该文件（CLI 解析 + 训练循环 + TDR 恢复 + checkpoint 混杂） | 任一改动牵动全文件；恢复逻辑是 C4 的前置阻塞 |
| E6 | 单文件多职责：`compute_vk_backend.hpp` 5174 行 | 该文件（fence / staging / dispatch / 算子混装） | header-only 项目里增量编译是主要开发成本 |
| E7 | `README.md` 与实际不符：死链 `docs/development/07-zipt-algorithm.md`、mojibake、声称链式 `add<LayerType>()`、文档索引缺 13–17 | `README.md:21`、`:35`、`:63`、`:269`、索引表（`:23` 起） | 新用户第一入口即误导；`add<T>(args...)` 是返回 `Result<void>` 的模板方法，**无链式 API** |
| E8 | `bench/doc_align_audit.ps1` 的 [A] 只检查**反引号里的文件名**，markdown 链接目标不查 | 该脚本 [A] 段正则 | README 的 `07-zipt-algorithm.md` 死链因此漏检（需扩审计或在文档侧修） |

### 3.3 生态与可用性

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| X1 | 推理无 top-k / top-p 采样截断 | `src/text_infer.cpp` 只有 `--temperature` 与 `--interactive`；`compute_layer_gpt.hpp:622` 的采样入口只吃 temperature | 长文本生成质量不可控；无法做"多轮对话闭环"演示 |
| X2 | ~~无 Dataset / DataLoader 抽象，文本训练数据加载不成体系~~ **前半已解决（2026-10-05/06，19 号设计阶段一~三）**：`nn::Dataset` + `.nndataset`/`.nnvocab` 统一格式 + `dataset_gen`/`dataset_convert`，text_train 只吃数据集；**剩 DataLoader/流式** | ~~`src/text_train.cpp` 内联数据路径；`train_pkg.py` / `cli_controllers.py` 各自处理~~（已收编，见 `docs/development/19-unified-dataset.md`） | 流式大语料仍不可行（`load_text` 一次性物化，P2-4 后半解决） |
| X3 | 无 Python / C 绑定 | 仓库仅有 `gui.py` 这一 Python 入口（subprocess 调 CLI） | 生态集成只能走命令行 + 文件 |
| X4 | `gui.py` 六个 Tab、7 处手写 `collect_args` | `gui.py`（`def collect_args` 实测 7 处） | 加 CLI 选项要手改多处，漏参风险（13 §4 的"五个 Tab / 1471-1500"坐标已过期） |
| X5 | GUI 画布拖动全量降采样重绘 | `gui.py:1135`（`_on_draw_move`）；已有节流先例 `gui.py:216`（`_RENDER_MS = 250`） | 快速拖动卡顿 |

### 3.4 模型能力

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| M1 | 优化器共 **5 个**：SGD / SGDWithMomentum / Adam / **AdamW** / Muon；其中**直接派生自 `Optimizer` 的是 4 个**（SGD/SGDWithMomentum/Adam/Muon），`AdamW` 派生自 `Adam` | `compute_optimizer.hpp:200`（SGD）、`:236`（SGDWithMomentum）、`:294`（Adam）、`:434`（`AdamW : public Adam`）、`:628`（Muon）；工厂注册名 `:723`、创建分支 `:745` | 缺无 m/v 状态的低显存优化器，模型规模受优化器状态限制（P2-2） |
| M2 | LRLA 仅存在于文档对比表，未实现 | `docs/development/06-rapt-algorithm.md` §LRLA | 线性注意力家族停在本项目基线 RLA-2/RAPT |
| M3 | IR-C 跨表达式图融合不采用、S6 自动窗口不接线 | `docs/development/03-ir-optimization.md:134`；`docs/development/02-operator-fusion.md` §跨 kernel 自动融合 | 每个表达式仍各 dispatch 一次；能融的必须写进单个 `dsl::compute` |
| M4 | M7 存储多态未立项 | `docs/development/17-unified-tensor-engine.md` §5 M7 行 | 新增第三后端仍需改 `compute_tensor.hpp` |
| M5 | ZiPT / AttnZip 已整体移除，代码保留在 `legacy/zipt` 分支 | `docs/history.md`「ZiPT 移除」条 | 记忆压缩方向暂缺实现 |

## 4. 路线图分期

优先级判据：**P0 = 低成本高价值且不修就有实际风险**；P1 = 1–2 月内的能力与健壮性；
P2 = 3–6 月的结构优化；P3 = 择机（trigger 未到就不排期）。
每项固定四栏：目标 / 验收口径 / 成本量级 / 依赖。

### 4.0 排期总览

| 期 | 项数 | 项目 | 共同前置 |
|---|---|---|---|
| **P0** | 5 | CI 真跑测试、文档一致性回正、BatchNorm 禁止静默回落（**✅ 已完成 2026-10-03**）、CMake install/export + presets、推理采样 top-k/top-p | 无（P0-1 是所有人的前置） |
| **P1** | 5 | `text_train` 模块化、checkpoint 优化器状态、危险精度组合 fail-fast、`gpt_test` flaky、CI 冒烟扩 MSVC Debug | P0-1 |
| **P2** | 8 | `.bin` 校验和、Adafactor、RAPT/CNN f16、Dataset/DataLoader、Python/C 绑定、GUI 两项、header 拆分、F16C | P0-1；P2-7 另需 P1-5；P2-6 另需 P0-5 |
| **P3** | 8 | 由 §6 触发条件启动（含 CUDA、分布式） | 各自触发条件 |

**串行约束（不可并行推进的组合）**：

1. **P1-1 → P1-2**：checkpoint 优化器状态依赖 `text_train` 先模块化，否则恢复路径继续膨胀。
2. **P0-1 → P0-2**：文档改动需要 CI 才能被自动验证。
3. **P0-1 + P1-5 → P2-7**：header 拆分前必须先有编译/链接期冒烟，否则无回归网。
4. **P0-5 → P2-6**：GUI 字段表在采样开关定型后一次改完。
5. **P1-2 → P2-1**：`.bin` 校验和与优化器状态段同属版本段改动，合并做省一次 `MODEL_VERSION` 升版。

### 4.1 P0（立刻做）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P0-1 CI 真跑构建 + 测试**（最高优先） | workflow 加 `-DNN_ENABLE_TESTS=ON`，并在 Linux 与 Windows 各跑一轮 `ctest`（失败时输出完整失败日志） | GitHub Actions 绿且日志出现 23 个用例；需 Vulkan 的用例按退出码 77 记 skip 而非失败 | 小（改一个 `.yml`，约 10 行） | 无。**先于其它一切**——没有真门禁，后面所有验收都无法自动防回退 |
| **P0-2 文档一致性回正** | ① ctest 计数统一为实测 23（20 目标 + `cnn_test_gpu` + `batchnorm_test_gpu` + `fusion_custom_layer_example`；2026-10-03 起）；② `CMakeLists.txt` 版本与 tag 对齐（或明确声明"以 tag 为准"）；③ 补 `docs/release-notes/v1.5.1.md`；④ 修 `README.md` 死链/mojibake/链式 API 声明/索引缺 13–17 | `bench/doc_align_audit.ps1` 的 [A][D][E][F] 可行动项为 0；开发文档中 ctest 计数的旧写法（ASCII 斜杠数字）零命中（历史归档 `docs/history.md` 除外） | 小–中（纯文档 + 一处 `project()` 版本号） | P0-1（否则改动无法被自动验证） |
| **P0-3 BatchNorm 禁止静默回落** ✅ **已完成（2026-10-03，真实现路线）** | **已实施**：`compute_layer_mlp.hpp` 新增 `BatchNorm` 层（训练/推理双态、running 统计 EMA + `extra_state` 序列化、`forward_recompute` 抑制 EMA），`make_norm_layer` 真分支；`scan_exprs` 模型 pass 补推理态 fwd+bwd 覆盖（GPU 闭合世界）；新增断言测试 `batchnorm_test`（+`--gpu` 变体），ctest 21 → 23，scan 结构 84 → 91（库内口径；样例收集器 93） | 构造 batchnorm 规格给出真层（可观测、不再静默降级）；ctest 含 `batchnorm_test`；详见 `docs/history.md`「BatchNorm 落地」条 | 已完成 | 无 |
| **P0-4 CMake install/export + CMakePresets** ⚡ **嵌入路径已完成（2026-10-04，用户裁定拆分）** | **已实施**：门面目标 `neuralnet::nn`（`add_subdirectory` + 一行 link + 一个 include 即用；C++26/线程/`NN_HAS_VULKAN`/SPIR-V 头/Vulkan/shader 顺序全经接口传递）、`NN_ENABLE_GPU=AUTO\|ON\|OFF`、`NN_BUILD_APPS` 嵌入默认 OFF、`nn_enable_gpu_fusion` 树外可用（`NN_LIB_SOURCE_DIR/BINARY_DIR` CACHE 跨作用域 + GPU 不可用时 stub 报错）、消费方编译选项（`-Wno-pass-failed`/`-fexperimental-library`/`/utf-8`）随门面传递；样例 `examples/downstream/`（CPU/GPU 双路径实测通过）+ 指南 `docs/usage/05-consume-as-library.md`。**剩余**：`install()`/`find_package(neuralnet.cpp)`、`CMakePresets.json`（用户裁定本轮不做） | 嵌入口径 ✅：干净构建目录里 `cmake -S examples/downstream` CPU 与 GPU（`NN_ENABLE_GPU=ON`）两路径均编译、链接、运行通过（GPU 路径含 scan 收集 91 条 = 库内口径），fusion 树外调用可用；install 口径待剩余部分 | 已完成（嵌入）/ 小–中（剩余） | P0-1（preset 需进 CI 才不腐化） |
| **P0-5 推理采样 top-k / top-p + 多轮对话闭环** | 采样器支持 top-k / top-p 截断与重复惩罚；`text_infer` 暴露对应开关；GUI GPT 推理 Tab 支持多轮上下文 | 同 prompt 在 top-k / top-p 下输出可复现（同 seed）；多轮对话示例可连续交互 N 轮不丢上下文 | 中 | 无。命名冲突提醒：`--topk` 属 `mnist_infer`（`src/mnist_infer.cpp:44`），`--top` 属 `tokenizer_infer`（`src/tokenizer_infer.cpp:46`）；`text_infer` 两者都没有，新开关命名不得与它们冲突 |

> **口径说明（与本文 §4 前置清单的差异）**：任务书前置清单同时把"危险精度组合 fail-fast"
> 列在 P0；按裁定台账（§5 与 §6）该项归 **P1**，理由是它需要先定"报错还是禁用"的语义。
> 本文以裁定台账为准，不重复排期。

### 4.2 P1（1–2 月：能力与健壮性）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P1-1 `text_train` 模块化** | 抽 `train_loop` / `recovery` / `checkpoint` 模块；退出码语义修正（保存失败不得成功退出） | 现有关键路径回归逐字节一致；新增恢复路径单测；`std::exit` 只在 CLI 参数解析层出现 | 中（1746 行文件拆解，34 处 `std::exit`） | 无，但它是 P1-2 的前置 |
| **P1-2 checkpoint 优化器状态** | `.bin` 增加可选优化器状态段（Adam m/v，`MODEL_VERSION` 5 → 6）；`Optimizer` 基类加 `save_state/load_state`；恢复路径接线 | 老文件缺段 = 冷启动语义；save→load→save SHA256 逐字节一致；TDR 模拟重启后动量非零且 loss 轨迹连续 | 中 | **P1-1**（否则恢复逻辑继续膨胀） |
| **P1-3 危险精度组合 fail-fast** | 启动期拒绝 `optimizer=f16` / 四字段全 f16 的组合，或明确要求显式确认开关（拟新增，名字待定） | 危险组合直接退出并打印原因；`--f16` 与合法 `--precision-*` 组合不受影响；新增 CLI 断言测试 | 小 | 无。依据 `docs/development/05-mixed-precision.md` §12.5 的实测结论 |
| **P1-4 `gpt_test` flaky 定位** | 复现并消除偶发失败（或定位为已知资源边界并加显式 skip/重试语义） | 连续大样本直跑零失败；定位结论写入 `docs/history.md` | 小–中（复现成本不确定） | P0-1（需要 CI 反复跑才有样本） |
| **P1-5 CI 冒烟扩到 MSVC Debug + 权重变化断言** | Debug 构建冒烟；训练 N 步后断言参数**确实变化**（防"loss 不动"类静默失败） | CI 中 MSVC Debug 目标编译通过；权重变化断言可检出冻结；大 TU 若触发 C1128 再出 `/bigobj` | 中 | P0-1。可复用 `f16_writeback_probe` 的"8 步 Adam 参数必须移动"思路 |

### 4.3 P2（3–6 月：结构优化）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P2-1 `.bin` 整文件校验和 + 尾部完整性标记** | 版本段之后加校验和；按注释约定升 `MODEL_VERSION` | 截断文件明确报错；合法文件 round-trip SHA256 一致 | 小–中 | P1-2（同一版本段改动，合并做省一次升版） |
| **P2-2 Adafactor 类无 m/v 优化器** | 新增不物化二阶矩的优化器，训练更大模型 | 同任务对比 Adam 的显存下降与收敛曲线；`create_optimizer` 可创建 | 中 | 无 |
| **P2-3 RAPT / CNN f16 接线** | RAPT 的 `dsl::compute*` 调用接精度实参；`Conv2D` 权重/梯度去 `Precision::F32` 硬编码 | f16 下 CNN/RAPT 训练健康收敛；CPU/GPU 字节锚按计划内变化流程重立 | 中 | 无（C2 的 fail-fast 先落地，防用户踩坑） |
| **P2-4 Dataset / DataLoader 抽象 + 流式文本**（**前半 Dataset 已于 2026-10-05/06 落地** = 19 号设计阶段一~三；本条只剩**流式后半**：DataLoader/分块迭代 `for_each_chunk`，不再把语料整段读进内存） | 流式加载（Dataset 格式与统一只读类见 `docs/development/19-unified-dataset.md`） | 大文件训练内存曲线平稳；`dataset_test` 逐位一致口径回归 | 中 | 无 |
| **P2-5 Python / C 绑定** | 提供稳定 C ABI 或 pybind 绑定，供生态集成 | 绑定层跑通训练/推理最小示例；ABI 报错信息可读 | 中–大 | **条件触发**（§6） |
| **P2-6 GUI 声明式字段表 + 绘制节流** | dataclass 元数据自动生成 UI 行与 `collect_args`；画布 motion 按帧节流、dirty 标志跳过空帧 | 现有 Tab 参数零遗漏（`bench/gui_cli_audit.py` 退出码 0）；拖动帧率回升 | 小–中（两项同批） | P0-5（采样新开关先定型，避免二次改字段表） |
| **P2-7 超大 header 拆分** | backend 拆 fence / staging / dispatch / 算子为独立 TU；`compute_vk_backend.hpp` 瘦身 | 拆分前后 build 全绿 + ctest 全绿 + CPU/GPU 字节锚逐位一致 | 中–大 | **前置 = 先补编译/链接期冒烟**（P0-1 + P1-5），否则拆分无回归网 |
| **P2-8 F16C 半精度转换快路径** | `_cvtss_sh` 快路径 + 批量转换，替代纯软件位操作 | CPUID 探测 + `NN_ENABLE_NATIVE` ON/OFF 两态；`f16_cpu_probe` 全量对拍逐位一致 | 中 | **条件触发**（§6） |

### 4.4 P3（择机）

P3 不排期，只由 §6 的触发条件启动：IR-C 图融合、S6 自动窗口、M7 存储多态、IR-D 第二 emitter、
ZiPT 恢复、LRLA、CUDA、分布式。

## 5. 13 号清单逐条裁定表

条目编号沿用 13 号原文的章节号（13 号本身已改为 16 行重定向 stub）。
其原正文的归档位置：**已执行/已否决记录**在 `docs/history.md`「13 号清单合并归档（2026-10-01）」节；
**仍未决的条目**由下表承接。本节逐条覆盖其**全部 16 条**未决项（含 17 §5 的 M7），一条不漏。

| 条目 | 裁定 | 理由与触发条件 |
|---|---|---|
| 13 §1 线程池归约分块统一 | **不做** | 归约已是确定性分段且 1/8-worker/串行逐字节一致；剩余差异（`parallel_transform`/`parallel_for_*` 用 `chunk_count()`，`row_reduce`/`col_reduce` 各自行块）**无正确性问题，只是概念一致性**。历史上 `Tensor::cpu_get_ptr` 的 8-70× 教训说明动热路径风险远大于收益。归档见 `docs/history.md` |
| 13 §2 超大 header 拆分 | **做（P2-7）** | 唯一被确认的日常成本是增量编译。**前置**：先补编译/链接期冒烟（P0-1 + P1-5），否则拆分期间 ODR/include 次序问题无网可抓 |
| 13 §3 `text_train` 模块化 | **做（P1-1）** | `src/text_train.cpp` 1746 行、34 处 `std::exit`，CLI/训练/恢复/checkpoint 混杂；退出码语义有真实缺陷。**它同时是 §7 的前置** |
| 13 §4 gui.py 声明式字段表 | **做（P2-6）** | 实测 6 个 Tab、7 处 `collect_args` 手工罗列，漏参风险真实。与 §5.3 绘制节流同批做（同一文件、同一批次验收） |
| 13 §5.1 F16C 快路径 | **条件触发（P2-8）** | 纯软件位操作在 f16 热点链上是真实开销，但收益未量化。**触发条件（三条同时）**：① CPUID 探测可行；② `NN_ENABLE_NATIVE` ON/OFF 两态都能构建；③ `f16_cpu_probe` 全量对拍逐位一致。历史次正规移位 UB 教训（`docs/development/05-mixed-precision.md` §12.12）要求对拍先行 |
| 13 §5.2 CPU f32→f16→f32 双趟 | **不做** | 该双趟是**量化模拟语义**，删除会改变行为；收益是热路径一次全量转换，属低量级。若将来有真实调用方依赖"直接返回"语义，另立 API 区分而非删除 |
| 13 §5.3 GUI 绘制节流 | **做（P2-6）** | `_on_draw_move` 每次 motion 全量降采样重绘；已有 `_RENDER_MS` 节流先例可复用。低成本、用户可感知 |
| 13 §7 checkpoint 优化器状态 | **做（P1-2）** | TDR 重启动量归零是**真实功能缺陷**（C4）。设计已明确：可选段 + 老文件缺段 = 冷启动 + `MODEL_VERSION` 升版。**依赖 §3 先拆** |
| 13 §8 CI 冒烟 | **做（P1-5）** | MSVC Debug + 权重变化断言可防"loss 不动"类静默失败，与 C1/C3 同族。`/bigobj` 未复现前不加 |
| 13 §9.4 max_abs_diff 统一 | **不做（择机）** | **实测 7 份定义**（`attn_consistency_test.cpp`、`ce_fusion_test.cpp`、`conv2d_gradcheck.cpp`、`fused_gpu_test.cpp`、`gpu_test.cpp`、`maxpool_gradcheck.cpp`、`tensor_expr_test.cpp`），且语义有分叉（conv2d 版多形状守卫返回 `1e9`）。**计数三处不一致**：13 号原文写 9 份、`src/test_common.hpp:19` 注释仍写 9 份、中途交接口径写 8 份，实测为 7 份——P0-2 文档回正时需一并订正。纯测试卫生、无产品风险，机械改写有反例 |
| 13 §9.5 `gpt_test` 偶发失败 | **做（P1-4）** | 偶发红在 CI 化（P0-1）后必然变成噪声；先复现再定性。62 次直跑 1 次失败、日志被覆盖，需先补日志留存 |
| 13 §9b H3 解码循环骨架 | **不做** | 仅采样段已合并，三个 `generate()` 循环体因运行态不同而保留；统一会引入条件分支，收益低。理由已归档 |
| 13 §9b H4 精度/形状三元组收敛 | **不做** | 尾部 `p_.compute` 实参在各点**可能是刻意不同**（RAPT 残差相加即反例），机械改写有语义风险；收益只是少写一个参数 |
| 13 §9b H5 掩码/索引构造 | **不做** | 注意力侧已随策略类收敛；剩余（conv 置换、patch 提取）的"CPU 循环 + upload_span"声明式化无正确性问题，且新原语成本高 |
| 13 §10 句柄指针化 / 存储多态提案 | **不做（已被 17 吸收）** | 13 §10.8 的"不采用句柄指针化"由 17 §3 D2 **维持否决**；其"直接读写"诉求由访问不变量（铁律 #11）承接。存储多态部分改期 M7 |
| 17 §5 **M7** 存储多态 + `as_cpu/as_gpu` | **条件触发（P3）** | 触发条件：**第三后端出现**。在此之前收益只剩"新增后端不改 `compute_tensor.hpp`"，无调用方 |

## 6. 触发式立项清单

原则：**触发器不到，不写代码、不排期**。触发条件必须可观测、可复现。

| 项 | 精确触发条件 | 立项时先读 |
|---|---|---|
| IR-C 跨表达式图融合 | 出现真实调用方满足**三条同时**：① 同形状长逐元素链；② 中间量不需留 backward；③ 单个 `dsl::compute` 写不下。**且不得复活录制 API**（记录 API 的历史否证见 `docs/history.md`） | `docs/development/03-ir-optimization.md` §5.3 |
| S6 自动窗口（跨 kernel 自动融合） | 同 IR-C 的调用方判据；当前"能融的直接写成单个 `dsl::compute`"已覆盖生产路径 | `docs/development/02-operator-fusion.md` |
| M7 存储多态 | **第三后端出现**（CPU/Vulkan 之外的引擎要落地） | `docs/development/17-unified-tensor-engine.md` §4.2 + §5 M7 |
| IR-D 第二 emitter | 需要 glsl 之外的代码生成后端，且该后端**必须走 AOT 闭合世界**（表达式结构须被 `scan_exprs` 覆盖） | `docs/development/03-ir-optimization.md` 阶段 D；`expr_emitter.hpp` |
| ZiPT 恢复 | ① 压缩向量保证**因果**（压缩器只能看当前位置之前）；② 实现质量对齐 RAPT/GPT（`forward_recompute` 梯度检查点 + activation offload + CLI/序列化/测试齐备）；③ 有同参对比数据 | `docs/history.md`「ZiPT 移除」条；`legacy/zipt` 分支 |
| LRLA | 先有与 RLA-2 / RAPT 的**同参对比数据**（质量、显存、吞吐），再决定是否实现 | `docs/development/06-rapt-algorithm.md` |
| Adafactor 类优化器 | 已排 **P2-2**；条件 = 需要训练超出 Adam m/v 显存预算的模型 | 本文 §4.3；`compute_optimizer.hpp` |
| CUDA 后端 | **P3**。当前明确不支持（见 §7）；触发 = 目标平台出现真实需求 + 有维护者承诺长期维护 + 愿意补 IR-D 第二 emitter 的 AOT 全链 | `AGENTS.md` §2；本文 §7 |
| 分布式训练 | **P3**。触发 = 单机单卡能力被真实任务触顶，且有明确的多机部署场景 | 本文 §7 |

## 7. 非目标（明确不做）

| 非目标 | 理由 |
|---|---|
| CUDA 全链 | 项目契约是 CPU / Vulkan 双后端，CLI 无 CUDA 参数；引入 CUDA 等于重走一遍 `import` / 存储多态 / IR-D 全链。仓库中无 CUDA 代码（grep 仅有 Muon 的 URL 注释） |
| 分布式训练 | 数据并行/张量并行需要先有通信层与容错语义；当前连 CI 都不跑测试（E1），先补门禁 |
| HTTP 服务化 | 仓库无任何 HTTP/gRPC 代码；推理入口是 CLI + GUI，服务化的部署/鉴权/并发语义无真实需求方 |
| ONNX 导入导出 | 无 ONNX 依赖与实现；导出需先把动态形状/精度语义固定下来，而 M7 尚未立项 |

> 列入本表的前提是"当前不做"有证据支持（上表逐条给了证据）。触发条件成熟时经 §6 重新立项。

## 8. 门禁与验收口径

### 8.1 现有基线（沿用 17 §6）

| 基线 | 口径 | 当前值 / 复现 |
|---|---|---|
| build | 全绿、零告警（`-Werror`） | `cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build` |
| ctest | **实测 23**（20 目标 + `cnn_test_gpu` + `batchnorm_test_gpu` + `fusion_custom_layer_example`）；需 Vulkan 用例退出码 77 = skip | `ctest --test-dir build` |
| L2 分层审计 | `L2-VIOLATIONS: 0`（宿主桥用量只披露） | `pwsh -File bench/doc_inventory.ps1` 第 [4] 节（实测：virtual 49 / host_bridge 40） |
| CPU 字节锚 | `--steps 20` hash = `6f8849f14da23110` | 逐位一致 |
| GPU 字节锚 | dev2 hash = `8ef51b2927253c50` | 逐位一致（GPU 档位按 17 §6 容差规则） |
| scan 产物 | `fused_registry.hpp`（**唯一产物**；`expr_specs.bin` 已随构建期单步删除） | `tools/scan_exprs.cpp` + `tools/fused_generate.hpp` 产物 |
| 初值锚 | `gpu_stability_probe --init-hash` 各模型跨进程一致 | 见 `AGENTS.md` §12 当前锚 |
| 文档对齐 | `pwsh -File bench/doc_align_audit.ps1`：[A][B][D][E][F] 可行动项为 0 | 2026-10-01 实测：可行动 0 / 历史白名单跳过 11 |
| 行为回归 | `layer_bench` 同窗交错 + 配对 A/B，无系统性回退（噪声 ±6%） | `build/layer_bench` |

### 8.2 本轮新增门禁（P0 后生效）

1. **CI 必须执行 ctest**：日志需出现 23 个用例；77 = skip 不算失败（P0-1）。
2. **开发文档不得出现 ctest 计数的旧写法**：统一写实测 23（P0-2）。
3. **BatchNorm 行为门禁**：构造 batchnorm 规格必须有可观测结果（报错或正确层），不允许静默降级（P0-3）。**✅ 已满足（2026-10-03）**：真层 + `batchnorm_test` 工厂断言。
4. **危险精度组合门禁**：危险组合必须启动期失败（P1-3）。
5. **审计覆盖补强**：`bench/doc_align_audit.ps1` 的 [A] 需扩展到 markdown 链接目标（E8），或由 P0-2 直接修掉已知死链。

### 8.3 验收纪律

- **计划内字节变化**：按 17 §6 例外流程**先立新锚**，新锚两次独立运行之间仍须逐字节，并在该期验收中显式记录。
- **既有故障不混入验收**：C1（GPU `stable=f16` loss 冻结）与 C6（`gpt_test` flaky）在修复前按"既有状态如实记录"处理，不得当成新回归。
- **裁定不可静默偏离**：任何与本文 §5/§7 不一致的做法，先在本文改裁定并说明理由，再实施。

### 8.4 本文档的维护约定

1. **状态只写现在式**：做完一项就把该项从 §4 移到 `docs/history.md`，并在 §4 留一行"已完成"指针。
2. **数字必须带复现命令**：任何计数（行数、测试数、锚值）都要能在命令里复现；不可复现的只写"约"或删。
3. **行号会腐烂**：行号只作入口线索；引用时优先给文件名 + 符号名，行号在改动后由 `bench/doc_align_audit.ps1` 兜底
   不覆盖的，由自动化测试兜底（新增断言测试优于文档数字）。
4. **裁定要留反例**：凡"不做"必须写清触发条件或反例（如 H4 的 RAPT 残差相加），否则后人无法判断能否重新立项。
5. **与 13 的关系已终结**：13 号是 stub，新增条目只进本文，不再往 13 追加。

## 9. 仍然生效的工程教训（原 13 §9 承接）

13 号清单的"仍然生效的工程教训"中，以下 2 条在 13 改为 stub 后无处落地，按原义承接。
（其余 3 条已由别处覆盖：`dsl::compute` 无 2 参重载见 `AGENTS.md` §7；`max_abs_diff` 计数已入本文 §5 裁定表；
`gpt_test` 偶发失败已列为本文 P1-4。）

### 9.1 eager → DSL 迁移前必须确认目标表达式结构已被 `scan_exprs` 覆盖（原 13 §9-2）

否则 GPU 闭合世界运行期**硬报错**。低成本做法 = 把新表达式写成与已扫描表达式**同构**
（如 RAPT 除法的 ε 位置对齐 `forward`），即可零成本复用键。
（对应铁律 #7：AOT 闭合世界，未命中硬报错，无 eager、无运行时编译。）

### 9.2 改测试公共头后，聚合器的 `#undef CHECK` 必须同步删除（原 13 §9-3）

`pragma once` 之下 `#undef` 会让后续子文件失去宏定义。测试用「`#define main test_xxx` + `#include 子测试.cpp`」
聚合编译，聚合器里任何 `#undef` 都会影响其后被 include 的所有子测试。

## 10. 变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-01 | 本文建立：合并 13 号未完成条目 + 未来方向 + 裁定台账；13 号改重定向 stub | 维护者裁定 |
| 2026-10-01 | 订正实测口径：ctest 19（后于同日的 A4/A5 两轮由 19 → 20 → 21）；`max_abs_diff` 7 份（非 9/8）；`gui.py` 1646 总行 / 1456 非空 / `collect_args` 7 处；承接原 13 §9 的两条工程教训 | Lead 与归档 owner 双重核验 |

