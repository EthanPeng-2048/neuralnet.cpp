# 开发路线图与未完成项裁定

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。

> **本文是"下一步该做什么"的唯一入口**：把 13 号清单（原 `docs/development/13-refactor-backlog.md`，
> 现归档于 `docs/history/history.md`）的未完成条目与未来 6–12 个月的方向合并为一份可执行的裁定台账。
> 13 号文档已于 2026-10-11 删除（内容全在本文与 `docs/history/history.md`）；已完成项、被否决方案与历史取舍统一归档在 `docs/history/history.md`。
> 底层契约的已落地部分（统一总纲 M1–M6）见 `docs/todo/17-unified-tensor-engine.md`。
> 写作纪律：只写事实与决定；行号取自实测，行号漂移后以符号名为准，未能确证的位置只给文件名、不给行号。

## 0. 一句话

未来一年两条主线：**① 把工程门禁与生态缺口补成真的**（CI 不跑测试、文档自相矛盾、无安装导出、
推理无采样截断）；**② 模型能力按"有真实调用方 / 有对比数据"再立项**，不为可能性写代码。

## 1. 定位与用法

| 文档 | 角色 | 何时读 |
|---|---|---|
| **本文（18）** | 方向 + **未完成项裁定台账**（做 / 条件触发 / 不做 + 验收口径 + 现状实测）；**不含执行计划** | 想知道"某件事该不该做、做到什么算完成"时 |
| `docs/todo/27-execution-plan.md` | **执行计划（优先级象限）**：把本文 §3/§4/§6 与 00 号 RV 台账里当前可动手的项排成"先做哪个" | 想知道"**下一个先做哪个**"时（本文只给裁定，排序在那边） |
| `docs/todo/00-code-review-remediation.md` | **2026-10-11 代码审查整改立项台账**（RV-01…RV-14）：外部审查发现逐条复核 + 立项（证据/方案/验收/批次） | 接手审查整改时（本文只登记"是否要做"，执行细节在那边） |
| `docs/history/history.md`「13 号清单合并归档」节 | 13 号清单（原 13 号文档，**已删除**）的条目原文与历史执行记录 | 只在需要追溯条目来源时 |
| `docs/history/history.md` | 全仓唯一历史归档：演进记录 / 已修复勘误 / 被否决方案 / 性能 A/B | 查"为什么当初这么改、旧数字是多少" |
| `docs/todo/17-unified-tensor-engine.md` | **已落地**的底层总纲（M1–M6 全部实施；仅 M7 未立项） | 改 Tensor/Engine/存储前 |
| `docs/history/history.md`「docs/development 15 · 16」归档节 | 17 的前置设计（15）与 P0 盘点（16），历史引用 | 只做证据追溯 |

### 1.1 立项三问（三问都答不上来就不立项）

1. **触发条件是否已满足**（§6 触发式立项清单给了精确判据）？
2. **是否已被否决**（§5 裁定表的 16 条 + §7 非目标）？
3. **验收口径能否复现**（§8 门禁与验收基线）？

### 1.2 与 17 的关系

17 负责"底层已经是什么样"（不变量、契约、已实施分期）；本文负责"上层接下来做什么"。
凡属 17 范围内的未决点（如 M7），本文只做**立项与否的裁定**，实施设计回 17。

### 1.3 与 27 的关系（**计划不落本文**）

**本文不承载执行计划**：不写"先做哪个 / 第一步动作 / 工时与并行安排"。这些一律单独成文——
当前执行排序见 [`27-execution-plan.md`](27-execution-plan.md)，专项方案见
[`26-f16-speedup-plan.md`](26-f16-speedup-plan.md)、[`04`](04-memory-optimization.md)、
[`25-release-requirements.md`](25-release-requirements.md) 等。本文只保留三样东西：
**要不要做**（§5 裁定表 / §6 触发条件 / §7 非目标）、**做到什么算完成**（§4 各行的验收口径）、
**现状实测**（§2/§3）。新增内容时先判断它属于哪一类；计划类内容进 27 号，本文只加一行指针。

## 2. 现状画像（实测）

| 维度 | 实测 | 复现 |
|---|---|---|
| `include/` | **84 个 `.hpp` / 45350 行** | `Get-ChildItem include -Recurse -Filter *.hpp` + 逐文件 `Get-Content` 计数 |
| `src/` | **83 个 `.cpp` / 29281 行** | 同上换 `src -Filter *.cpp` |
| `shaders/` | **22 个 `.comp`** | `Get-ChildItem shaders -Recurse -Filter *.comp` |
| `docs/` | **52 篇 `.md`**（含 `history/history.md` 与 `release-notes/`） | `Get-ChildItem docs -Recurse -Filter *.md` |
| 最大源文件 | `expr_glsl_gen.hpp` **3435** 行、`compute_cpu_engine.hpp` **2797**、`compute_gpu_engine.hpp` **2625**、`compute_engine.hpp` **2260**、`expr_dsl.hpp` **1982**、`gui.py` **2038**、`backend/compute_vk_backend.hpp` **846**（v1.8.1 拆 4 片段后回落）。**`src/text_train.cpp` 已不在榜**——P1-1 模块化后 1830 → **396** 行（循环/恢复/断点拆到 `cli/text_train_*.hpp`） | 逐文件 `(Get-Content $f).Count` |
| ctest | **55 个测试** = 42 个测试目标（`list(APPEND NN_TEST_TARGETS` 计数）+ 13 个变体/附加（`cnn_test_gpu`、`cnn_test_f16`、`cnn_test_f16_gpu`、`rapt_test_f16`、`rapt_test_f16_gpu`、`batchnorm_test_gpu`、`t1_58_test_gpu`、`bitnet_struct_test_gpu`、`write_epoch_test_gpu`、`ternary_gemm_test_gpu`、`adam_quant_test_gpu`、`fusion_custom_layer_example`、`expr_fuse_test`）；**无 Vulkan 的树只注册 53 条**（`fusion_custom_layer_example` / `expr_fuse_test` 是 GPU 专属门禁） | `ctest --test-dir build -N` |
| 引擎接口 | **52 个 virtual 方法**；`L2-VIOLATIONS: 0`；宿主桥 39 处（仅披露，2026-10-16 实测；P4-2 拆掉 RAPT 逐头 host 往返后由 41 回落） | `pwsh -File bench/doc_inventory.ps1` |
| 版本 | git tag **v1.8.2**；`CMakeLists.txt` 的 `project(... VERSION 1.8.2)`（与 tag 对齐；口径仍为 **git tag 是版本权威**）；`release-notes/` 覆盖 **v1.5.0 → v1.8.2 的全部 tag**——`v1.5.1` / `v1.6.1` / `v1.7.1` / `v1.8.0` 的独立说明已于 2026-10-18 补齐 | `git describe` + 读 `CMakeLists.txt` + `ls docs/release-notes/` |
| CI | `.github/workflows/cmake-single-platform.yml`：clang++ / Ninja / Release / `NN_ENABLE_NATIVE=OFF`，Linux + Windows；**开 `NN_ENABLE_TESTS` 并跑 ctest**（普通 + `NN_BIND_DEBUG=1` 各一轮，P0-1）。另有 `msvc-debug` job：Windows + Visual Studio 17 2022（cl.exe）/ Debug / `NN_ENABLE_GPU=OFF` 纯 CPU / ctest（除已知 C11 外全跑；P1-5，2026-10-18） | 读该 workflow |

> 口径说明：表中行数均为**总行数**（实测刷新）。
> `51`（GPU 树）/ `49`（无 Vulkan 树）是当前唯一正确计数；`--preset` 各树同口径（GPU 是否可用决定
> 两条 GPU 专属门禁是否注册）。文档中其它 ctest 计数写法属**历史记录**：已完成行（如 P2-3/P2-8 的
> "47/47"）与 `docs/history/history.md` 的旧数字是当时的验收事实，不属计数口径冲突；**新增/修改文档
> 时写当前值**。口径差异来源见 §3 工程化第 4 条。

### 2.1 已经稳固的底座（本轮不再动）

| 底座 | 状态 | 依据 |
|---|---|---|
| 统一 Tensor / ComputeEngine 契约（M1–M6） | 全部实施 | `docs/todo/17-unified-tensor-engine.md` §5 |
| 访问不变量（Tensor 存储私有、一切经引擎） | 铁律 #11，编译期强制 | `AGENTS.md` §5 |
| L2+ 计算路径禁用 `Matrix` | 铁律 #12，审计口径 `L2-VIOLATIONS: 0` | `AGENTS.md` §5；`bench/doc_inventory.ps1` 第 [4] 节 |
| AOT 闭合世界 + 单 fold 注意力 kernel | 现行；掩码 × 位置偏置两正交维度 | `docs/todo/02-operator-fusion.md` |
| 位置编码多态统一（按注入点分所有权） | 现行；新增 `compute_position_encoding.hpp` | `docs/history/history.md`「Layer 层简化轮」条 |
| BPE 保序并行 encode（任意并行度逐字节一致） | 现行 | `AGENTS.md` §12 |
| 声明式初始化 `InitSpec`（初值跨进程确定） | 现行 | `docs/todo/17-unified-tensor-engine.md` §4.4 |
| CPU/Vulkan 双后端 + 多设备选择 | 现行 | `AGENTS.md` §2 |

**含义**：本文的路线图不需要再排"底层重构"——底层已完成；缺口集中在工程门禁与上层生态。

## 3. 缺口总表

每条 = 一句话现象 + 坐标 + 影响面。**坐标只写在已实测确认的位置。**

### 3.1 正确性

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| C1 | GPU `stable=f16` 训练 loss 回读冻结（权重照常更新） | `docs/todo/14-f16-stable-gpu-loss-frozen.md` | GPU f16 训练的 loss 曲线不可观测；无对应 ctest 用例 |
| C2 | `optimizer=f16` 无 loss scaling 即更新爆炸、`stable=f16` 约 200 步 NaN | `docs/development/05-mixed-precision.md` §12.5；`src/text_train.cpp` 解析处只赋值不校验；核对与立项见 [`24-f16-training-defects.md`](24-f16-training-defects.md) | 用户可组合出必崩配置，且无任何启动期提示 |
| C3 | `NormType::BatchNorm` **静默回落**到 LayerNorm | `compute_layer_mlp.hpp:782`（`make_norm_layer`）；`README.md:248` 声称支持；`gui.py:60` 把 `batchnorm` 列为可选项；`model_spec.hpp:43` 保留枚举 | 模型规格写 batchnorm、实际跑 layernorm，无告警、无日志。**✅ 已解决**：真正实现 `BatchNorm` 层（P0-3），`make_norm_layer` 真分支，见 `docs/history/history.md` |
| C4 | checkpoint（`.bin`）不存 Adam m/v | `model_serialization.hpp`（版本常量现为 **v7**，见该文件注释）；`compute_optimizer.hpp` 无 `save_state/load_state` | TDR 重启 / `--resume` 后动量归零，续训轨迹不连续。**2026-10-16 裁定：范围扩写为「训练状态段」（计数器 / 优化器状态 / 训练配方 / 数据集身份），见表下 P1-2 行** |
| C5 | `.bin` 无整文件校验和与尾部完整性标记 | `model_serialization.hpp` 版本常量附近的注释（明示"若引入需升 MODEL_VERSION"） | 截断/损坏的模型文件无法与合法文件区分 |
| C6 | ~~`gpt_test` 偶发失败未定位~~ **✅ 已定性（2026-10-15）：未复现 + 同机完全确定**。278 次直跑（200 串行 + 8 并发 + 40 串行 + 30 次 `taskset -c 0,1` 2 核超订）**零失败且日志逐字节一致**（单一 md5）⇒ 不存在数值随机性；历史 1/62 只能是环境差异（编译器/CPU/FMA）或当次进程级异常。已加环境指纹行（`[env] 编译器/标准库/构建/线程数/FMA`）使未来任何失败自带坐标；**不加 retry/skip**（输出确定 ⇒ 失败必属环境或代码变化，重试会掩盖回归） | 13 原条目 §9.5（62 次直跑 1 次失败、日志被覆盖）；本轮证据与复现命令见 `docs/history/history.md` | 剩余风险 = ≤1.5%（95% 上界，本机口径）的 CI 偶发红；已在失败日志自描述上闭环 |
| C7 | RAPT / CNN 未接 f16：`Conv2D` 权重/偏置/梯度硬编码 `Precision::F32` | `compute_layer_conv.hpp:148`（权重）、`:151`（偏置）、`:154`/`:155`（权重/偏置梯度）；RAPT 头内 32 处 `dsl::compute*` 调用未接精度实参 | f16 只覆盖 MLP/GPT 路径，CNN/RAPT 用户得不到显存收益 |
| C8 | `host_bridge_uses` 计数与文档不一致（曾实测 40，13/17 记 41） | `pwsh -File bench/doc_inventory.ps1` 第 [4] 节 | **✅ 已回正（2026-10-16 复测：实测 39 = §2 文档 39）**——P4-2 拆掉 RAPT 逐头 host 往返后该计数由 41 回落到 39，文档已随本轮刷新；披露项，从来不是违规 |
| C9 | f16 训练**降速**：`--f16` 同窗比 f32 慢 11–17%（118.4 vs 101.0 / 107.1 ms/step），与「f16 省带宽应更快」预期相反。**✅ 已兑现（2026-10-14）**：P2-11（编译期精度特化）+ P2-13（K 切分/细 tile 变体）后 40HX `--f16` **1013 vs f32 1027 ms（−1.4%）**、850M（无 f16 ALU）**2005 vs 2038 ms（−1.6%）**，峰值显存 818 → **606 MiB** | [`24-f16-training-defects.md`](24-f16-training-defects.md) §4；`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md`、`2026-10-06-vulkan-vs-torch.md`；`docs/history/history.md`「设备侧 kernel 表按阶段归因」 | 原缺口已关闭；剩余"墙钟 − 设备 kernel ≈4.2%"归 P4-1、单算子调参归 P4-2 |
| C10 | **RAPT GPU 训练比同配置 GPT 慢 ≈16×**（同一命令只换 `--model rapt`，40HX/d128·h4·L4·ff512）。**✅ 已归因（2026-10-15，850M 实测 + A/B 验证）**：本机 b8/seq128/d128·h4·L4·ff512 复现为 **5.5×**（RAPT step 741.81 ms vs GPT 134.84 ms / 3 步）。**原怀疑方向被证伪**（host 侧扫描派发 = 0.1–0.3% 墙钟）；真因是两项独立成本：**① `to_matrix` host 往返 = RAPT 墙钟的 75%**（exclusive 573.82 ms/3 步；来源 = `rms_norm_forward_` 的**逐头** 1/rms 收集：每 block 每 step 2 × batch×heads ≤ 64 次下载，b8/h4/L4 = **256 次/step**，每次 `download_vector` 触发整条流水线 drain ⇒ 258 次 begin/end_batch per step）。**A/B 证据**：`--checkpoint-every 1`（forward 侧不收集 ⇒ 无下载）令 RAPT **forward 659.88 → 160.77 ms（4.1×）**，与 GPT forward 103.19 ms 同量级。**② 设备侧 `scan_prefix` kernel = 530.40 ms 设备时间里的 340 ms**（≈113 ms/step，单次 8–10 ms @ dk=32/seq=128/BH=32）⇒ 扫的是 kernel **本身**耗时，不是启动/staging 常数 | GitHub **issue #15**（2026-10-01 开，唯一未进本文台账的待办，本行即其登记入口）；RAPT 路径 = `compute_layer_rapt.hpp` + 扫描原语（`docs/development/06` §扫描原语）；实测证据 `docs/history/history.md`「C10 归因」条 | 结论：**归 P4-2**，两把杠杆分开做——(a) 把逐头 1/rms 缓存改成**设备侧**拼装（`insert_rows` 进 (BH,seq) 缓冲，避免 download+再 upload；注意 f16 下 dst 精度须与被消费处一致）+ 逐头循环批量化（同一循环还有 1060 次 `slice_rows`/step）；(b) `scan_prefix` kernel 调参（P4-2 的逐 op 项）。 **✅ (a) 已修（2026-10-15，P4-2）**：逐头 1/rms 收集改设备侧拼装后同配置 **step 741.81 → 521.68 ms（−29.7%）**、`to_matrix` 768→0、begin/end_batch 774→6、忙率 69.3%→84.0%，数值逐位不变（CPU f32/f16 `rapt_test` 输出逐字节一致）；**✅ (b) 已修（2026-10-15，P4-2b）**：`scan_prefix` 专用路径改行归属 + 4 路展开 + `dk ≤ 32` 窄共享变体，单次 **7.4–7.9 → 1.64 ms**、同会话 e2e **step 888.24 → 510.24 ms（−42.6%）**，数值**逐位不变**（`rapt_test --gpu`/`--f16 --gpu` stdout md5 前后相同 + 新门禁 `scan_bitwise_test` 61 用例 CPU↔GPU 逐位一致）。**新暴露的 e2e 瓶颈**：host 侧 staging 等待（`copy_from` 355.67/510 ms ≈66% 墙钟）。见 §4.5 P4-2 行。 |
| C11 | **RAPT f16 冒烟出现 NaN（已知问题，根因未定）**。**用户已排查，怀疑是 GPU/设备后端自身问题；Release 下不触发只是概率，不是"Debug 专有"**。2026-10-18 本机补充观测：`build/p1-5-debug`（clang 24 / Debug / `NN_ENABLE_NATIVE=OFF` / CPU-only）`rapt_test --f16` 的 loss `3.90206 → 2.79626`（step 7）→ **step 8 `nan`**；Release 同输入（`build/cpu` native ON / `build-rv` native OFF）本轮 15 步通过（末步 ≈2.24–2.26）；**102 MSVC Debug CPU-only 直接跑 `rapt_test --f16` 3/3 通过**（不是必现）。CPU-only Debug 也复现过 ⇒ 至少这个构建下还存在**非 GPU** 的触发路径；Release/MSVC 通过**不能**作为"已排除"的证据 | `src/rapt_smoke_test.cpp`（经 `rapt_test --f16`）；`build/p1-5-debug/test/rapt_test --f16`；102 `nn-p1-5-msvc` | P1-5 的 MSVC Debug job 暂以 `ctest -E "^rapt_test_f16$"` 排除，避免这个已知不稳定项把编译冒烟永久染红；该排除**不代表**问题已解决，也不影响 P1-5 的门禁结论。根因定位后应删除排除项 |

### 3.2 工程化

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| E1 | CI 是唯一自动门禁，却**只编译生产目标、不编译测试、不跑 ctest** | `.github/workflows/cmake-single-platform.yml`（configure 无 `NN_ENABLE_TESTS`，无 ctest 步骤） | 所有测试结论只存在于本地；回归可在无人察觉时合入 |
| E2 | 无 `install()` / export，无 `CMakePresets.json` | `CMakeLists.txt` 实测 `install(` 0 处；根目录无 `CMakePresets.json` | **嵌入路径已完成**：门面目标 `neuralnet::nn` + `NN_ENABLE_GPU`/`NN_BUILD_APPS` 开关 + `nn_enable_gpu_fusion` 树外可用（见 `docs/usage/05-consume-as-library.md`、样例 `examples/downstream/`）；**剩余**：`install()`/`find_package`（用户裁定本轮不做）；`CMakePresets.json` **已于 2026-10-11 补上**（`release`/`dev`/`debug`/`portable`/`cpu`/`probes` + build/test/workflow preset，见 `CONTRIBUTING.md` §1） |
| E3 | `CMakeLists.txt` 版本与 git tag 不一致，且缺 v1.5.1 release note | `CMakeLists.txt` 的 `project(...)` 行（**已对齐 v1.6.0**）；`docs/release-notes/` 仍缺 v1.5.1 | 版本可读性缺失；分发产物无对应变更说明（release note 缺口仍在） |
| E4 | 文档内 ctest 计数曾自相矛盾（19 vs 20） | `AGENTS.md` §12 已统一为实测 21（19 目标 + `cnn_test_gpu` + `fusion_custom_layer_example`） | 已修；口径差异来源见 §2 口径说明 |
| E5 | ~~`src/text_train.cpp` 1746 行、34 处 `std::exit`~~ **✅ 已解决（P1-1 模块化，2026-10-16）**：1830 → **396 行**（入口只留接线），训练循环 / 恢复 / 断点三块拆到 `include/neuralnet.cpp/cli/text_train_{config,loop,recovery,checkpoint}.hpp`，三模块一律返回 `Result` 不终止进程——`std::exit` 全部集中在 **CLI 参数解析层**（`cli/text_train_config.hpp`） | 原坐标（CLI 解析 + 训练循环 + TDR 恢复 + checkpoint 混杂） | 原影响面（任一改动牵动全文件）已解除；C4 的前置阻塞解除，P1-2 可直接开工 |
| E6 | 单文件多职责：`compute_vk_backend.hpp` 5174 行 | 该文件（fence / staging / dispatch / 算子混装） | header-only 项目里增量编译是主要开发成本 |
| E7 | `README.md` 与实际不符：死链 `docs/history/history.md`、mojibake、声称链式 `add<LayerType>()`、文档索引缺 13–17 | `README.md:21`、`:35`、`:63`、`:269`、索引表（`:23` 起） | 新用户第一入口即误导；`add<T>(args...)` 是返回 `Result<void>` 的模板方法，**无链式 API** |
| E8 | `bench/doc_align_audit.ps1` 的 [A] 只检查**反引号里的文件名**，markdown 链接目标不查 | 该脚本 [A] 段正则 | **✅ 已修（2026-10-15）**：[A] 扩展为同时校验 markdown 链接的**目标可达性**（相对路径按文档所在目录解析；行内代码段先剥离，避免把示例链接误当链接）；实测当场抓到 `todo/25-release-requirements.md` 的 `../CONTRIBUTING.md` 死链并一并修掉。README 的 `07-zipt-algorithm.md` 旧死链已在 P0-2 迭代中修 |
| E9 | **归约融合 shader 不支持"归约后逐元素后处理"形态**（`reduce(...) * k + c` 类表达式）：GPU 硬报错拒绝（契约，防静默错值），CPU 已支持 | `GpuEngine::eval_expr_reduce` 的拒绝分支（错误文案即本缺口）；回归锁 `src/ce_fusion_test.cpp`「归约+后处理 (reduce*k+c)」用例（断言 GPU 必须硬报错）；机制背景 `docs/history/history.md` 同名条目 | 该形态只能拆两步 dispatch（先 `compute_reduce` 再 `dsl::compute` 后处理），多一次 kernel 与中间张量。**支持路径（通用、非单算子）**：归约生成器在归约累加收尾（epilogue）后接逐元素链发射（操作对象是归约结果的广播值），随后放开 `eval_expr_reduce` 的末指令校验；CPU 的指令下标反向切片前置校验已是同语义，可直接对拍 |

### 3.3 生态与可用性

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| X1 | ~~推理无 top-k / top-p 采样截断 / 无多轮上下文~~ **✅ 两半全完成（P0-5：采样器 2026-10-15 / 多轮闭环 2026-10-16）**：**采样** = `SamplingConfig`（temperature / top-k / top-p / repeat_penalty / repeat_window / seed）+ `text_infer` 五个开关（`--top-k`/`--top-p`/`--repeat-penalty`/`--repeat-window`/`--seed`）+ GUI GPT 推理 Tab 同步暴露 + 门禁 `sampling_test`。**多轮** = `Tokenizer::chat_prompt`（对话标记模板 + 按整轮截断）+ `--interactive`（进程内累积）/ `--turn`+`--reply-out`（历史自带、一次一问）+ GUI 多轮面板（`ChatSession`）+ 门禁 `chat_prompt_test` / `bench/gui_cli_audit.py` [7] | 采样器在 `compute_layer_base.hpp`（CPU 侧 host 采样，**不进 AOT 闭合世界**）；多轮 prompt 构建在 `domain_tokenizer_base.hpp`，CLI 在 `src/text_infer.cpp`，GUI 历史在 `cli_controllers.py`（`ChatSession`）+ `gui.py` | **闭合（2026-10-16）**：`--interactive` 此前每轮只拼当轮输入（**每轮从零起算**）、GUI 完全无历史；现两条路径都带完整历史，且"上下文不丢"落到可断言形式 = 第 N 轮 prompt 以第 N-1 轮为**逐 token 前缀**（见 §4.1 P0-5 行） |
| X2 | ~~无 Dataset / DataLoader 抽象，文本训练数据加载不成体系~~ **前半已解决（19 号设计阶段一~三）**：`nn::Dataset` + `.nndataset`/`.nnvocab` 统一格式 + `dataset_gen`/`dataset_convert`，text_train 只吃数据集；**剩 DataLoader/流式** | ~~`src/text_train.cpp` 内联数据路径；`train_pkg.py` / `cli_controllers.py` 各自处理~~（已收编，见 `docs/todo/19-unified-dataset.md`） | 流式大语料仍不可行（`load_text` 一次性物化，P2-4 后半解决） |
| X3 | 无 Python / C 绑定 | 仓库仅有 `gui.py` 这一 Python 入口（subprocess 调 CLI） | 生态集成只能走命令行 + 文件 |
| X4 | `gui.py` 六个 Tab、7 处手写 `collect_args` | `gui.py`（`def collect_args` 实测 7 处） | 加 CLI 选项要手改多处，漏参风险（13 §4 的"五个 Tab / 1471-1500"坐标已过期） |
| X5 | ~~GUI 画布拖动全量降采样重绘~~ **✅ 已修（2026-10-15，P2-6 绘制半边）**：原坐标已过期（`_on_draw_move` 当时不降采样）；实测的拖动开销真因是**逐事件新建画布项**（一笔 2000 事件 = 2000 个 item）+ 逐事件 PIL 画线。现改为一笔一个 item（原地 `coords` 刷新）+ 空帧跳过 + 视觉刷新按帧节流（`_PAINT_MS = 16`）+ PIL 光栅化延迟到边界（`_downsample_buffer` 唯一刷新点） | `gui.py` 的 `StrokeTrack` / `_paint_live` / `_rasterize_pending`；门禁 `bench/gui_draw_logic_audit.py` + `bench/gui_draw_runtime_audit.py` | 实测（同机合成 2000 事件）：16.6 ms vs 旧写法 37.9 ms（**2.3×**）；画布项 **2058 → 59**（O(事件) → O(笔数)） |
| X6 | **无"写自定义层"使用文档**（**文档部分已修，2026-10-11**）：层接入的七条契约（`init_impl` NVI / `checkpoint_mode_` 缓存 / `activation_cache` / 精度 profile 用法 …）现成文于 `docs/development/write-a-custom-layer.md`（含自包含、经数值梯度检查实测通过的层模板 + 可复制 gradcheck 片段 + GPU 清单）；`NN_UNWRAP` 是 unwrap 也已在该文与 `docs/development/cpp-api-manual.md` §1.3 说明 | 指南：`docs/development/write-a-custom-layer.md`；背景实测报告（含逐条坐标）见 `docs/development/20-custom-layer-ergonomics.md`（v1.7.0，库外视角） | 使用者写出"能编译但静默失效"的层（精度硬编码、offload 失效、gradcheck 全无）；库内 C7（Conv2D 硬编码 F32）即同一成因；剩余缺口 = R2（`Layer::init` 精度自检）仍为代码项 |

### 3.4 模型能力

| # | 现象 | 坐标 | 影响面 |
|---|---|---|---|
| M1 | 优化器共 **5 个**：SGD / SGDWithMomentum / Adam / **AdamW** / Muon；其中**直接派生自 `Optimizer` 的是 4 个**（SGD/SGDWithMomentum/Adam/Muon），`AdamW` 派生自 `Adam` | `compute_optimizer.hpp:200`（SGD）、`:236`（SGDWithMomentum）、`:294`（Adam）、`:434`（`AdamW : public Adam`）、`:628`（Muon）；工厂注册名 `:723`、创建分支 `:745` | 缺无 m/v 状态的低显存优化器，模型规模受优化器状态限制（P2-2） |
| M2 | LRLA 仅存在于文档对比表，未实现 | `docs/development/06-rapt-algorithm.md` §LRLA | 线性注意力家族停在本项目基线 RLA-2/RAPT |
| M3 | ~~IR-C 跨表达式图融合不采用~~ **已恢复并接入训练路径（P1+P2）**：`begin_expr/end_expr` + 图融合（P1 链 / P2 写穿多输出）+ Adam/AdamW 接线；S6 自动窗口仍未接线 | `docs/todo/03-ir-optimization.md`；`docs/history/history.md`「IR-C 定位修正与恢复立项」「IR-C P2 收官」 | S6 之外的"能融的必须写进单个 `dsl::compute`"已放宽（跨表达式经录制段融合）；**P4 CE online-softmax fold 已落地（通用 FoldSpec 机制，5→3 kernel/step，见 `docs/history/history.md`「CE online-softmax 单趟 fold（P4）」）**；**P3 跨链批量派发已落地（`#b` 批量变体 + BDA 实例表 + 图内同签名分组 CAP 32）**：设备表读数口径 = 批量/非批量分列（用户裁定），`fused@optimizer` 归零、派发计入 `fused_b@optimizer` 行，**合计 1330 → 57 calls/20 步（2.85/step ≤ 6/step）**；批量 vs 逐个逐字节一致（含同组混形状）、字节锚逐位不变，见 `docs/history/history.md`「P3 跨链批量派发」 |
| M4 | M7 存储多态未立项 | `docs/todo/17-unified-tensor-engine.md` §5 M7 行 | 新增第三后端仍需改 `compute_tensor.hpp` |
| M5 | ZiPT / AttnZip 已整体移除，代码保留在 `legacy/zipt` 分支 | `docs/history/history.md`「ZiPT 移除」条 | 记忆压缩方向暂缺实现 |
| M6 | ~~**三值（T1_58）模型无法续训**~~ **✅ 已解决（P1-7，2026-10-16）**：原现象 = `.nnmodel` 载荷是打包形态，BitLinear 载入后进 `packed_only_` 部署态，`backward()` 直接报错（「部署态只读」）⇒ 三值模型不能从推理模型做 SFT / QAT 续训。**现在**：载入意图 `WeightLoadIntent::Trainable` 按文件 τ 反量化成 f32 latent（量化器不动点）+ 播种量化缓存 ⇒ 首次 forward 与推理文件逐位一致、可反向、可更新；`.nnmodel` 的 role 门禁（P1-2 的"唯一拒绝"）已删除 | `ternary_pack.hpp`（`latent_row_scale`/`dequant_row`）；`model_serialization.hpp`（`param_blob_to_entry` 意图分流 + `load_model`）；`compute_layer_bitlinear.hpp`（`TernaryLatent + aux(τ)` 播种分支）；`cli/text_train_recovery.hpp` | 缺口已关闭（见 §4.2 P1-7 行）；推理侧部署只读态不变（`Inference` 为默认意图） |

## 4. 路线图分期

优先级判据：**P0 = 低成本高价值且不修就有实际风险**；P1 = 1–2 月内的能力与健壮性；
P2 = 3–6 月的结构优化；P3 = 择机（trigger 未到就不排期）；
P4 = 持续优化方向（底座已稳后继续加杠杆，2026-10-14 立项，见 §4.5）。
每项固定四栏：目标 / 验收口径 / 成本量级 / 依赖。

> **排序不在这里**：§4 只写"是什么 / 做到什么算完成 / 依赖什么"；**先做哪个**见
> [`27-execution-plan.md`](27-execution-plan.md)（见 §1.3：计划类内容一律单独成文，本文只留指针）。

### 4.0 排期总览

| 期 | 项数 | 项目 | 共同前置 |
|---|---|---|---|
| **P0** | 5 | CI 真跑测试（**✅**）、文档一致性回正（**✅ 2026-10-15**）、BatchNorm 禁止静默回落（**✅**）、CMake install/export + presets（嵌入路径 ✅ / install 用户裁定不做）、**推理采样 top-k/top-p + 多轮对话闭环（✅ 2026-10-16，两半齐，见 §4.1 P0-5 行）** | 无（P0-1 是所有人的前置） |
| **P1** | 7 | `text_train` 模块化（**✅ 2026-10-16**）、checkpoint 优化器状态（P1-2，**✅ 2026-10-16**）、危险精度组合启动期告警（细则见 [`24`](24-f16-training-defects.md)）、`gpt_test` flaky（**✅ 2026-10-15**）、CI 冒烟扩 MSVC Debug（**✅ 2026-10-18**）、**自定义层使用文档 + 精度自检**（P1-6 **✅ 2026-10-14**）、**三值模型续训入口**（P1-7，**✅ 2026-10-16**） | P0-1；P1-6 无依赖 |
| **P2** | 15 | `.bin` 校验和（**✅ 2026-10-16，随 P1-2 同批交付**）、Adafactor、RAPT/CNN f16、Dataset/DataLoader、Python/C 绑定、GUI 两项、header 拆分、F16C、step 内固定 I/O 开销（P2-9）、**原生 f16 算术用足（P2-10）**、**编译期精度特化（P2-11，已完成）**、**优化器 f16 批量派发（P2-12）**、**f16 K 切分/fine tile（P2-13）**、**全流程真 f16（P2-14）**、**标准 `--f16` / PyTorch AMP（P2-15）** | P0-1；P2-7 另需 P1-5（**✅ 2026-10-18**）；P2-6 另需 P0-5 |
| **P3** | 7 | 由 §6 触发条件启动（含 CUDA、分布式）——IR-C 已落地故已移出本档（2026-10-16 计数回正：原写 8） | 各自触发条件 |
| **P4** | 4 | 性能利用率 / 算子性能 / 显存开销的**持续提升** + API 易用性与 **Fold 表达式自动化扫描**（见 §4.5；P4-2 的 C10 派生 (a)+(b) 均已完成 2026-10-15） | 各自见 §4.5 |

**串行约束（不可并行推进的组合）**：

1. **P1-1 → P1-2**：checkpoint 优化器状态依赖 `text_train` 先模块化，否则恢复路径继续膨胀。
2. **P0-1 → P0-2**：文档改动需要 CI 才能被自动验证。
3. **P0-1 + P1-5 → P2-7**：header 拆分前必须先有编译/链接期冒烟，否则无回归网。**P1-5 已完成（2026-10-18）⇒ P2-7 的前置已解锁**。
4. **P0-5 → P2-6**：GUI 字段表在采样开关定型后一次改完。
5. **P1-2 → P2-1**：`.bin` 校验和与优化器状态段同属版本段改动，合并做省一次 `MODEL_VERSION` 升版。

### 4.1 P0（立刻做）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P0-1 CI 真跑构建 + 测试** ✅ **已完成（2026-10-12）**（最高优先） | workflow 加 `-DNN_ENABLE_TESTS=ON`，并在 Linux 与 Windows 各跑一轮 `ctest`（失败时输出完整失败日志）；顺带再跑一轮 `NN_BIND_DEBUG=1 ctest`（RV-06 绑定门禁）。**实施**：`.github/workflows/cmake-single-platform.yml` 两个 configure 步各加 `-DNN_ENABLE_TESTS=ON`，Build 之后加 4 个测试步（Linux/Windows × 普通/`NN_BIND_DEBUG=1`，`--output-on-failure`）；打包步补 `dataset_gen`/`dataset_convert`（8 个应用入口齐全）。**前置修复**：`matmul_q_impl`（CPU/GPU）在递归 impl 产物/静态工厂产物上进公共 `cast` 前未 stamp → 三值用例（`t1_58_test`/`ternary_gemm_test`/`model_roles_test`）在 `NN_BIND_DEBUG=1` 下红；补 `adopt` 后本机 Release+CPU `NN_BIND_DEBUG=1 ctest` **40/40** | GitHub Actions 绿且日志出现全部用例；需 Vulkan 的用例按退出码 77 记 skip 而非失败 | 完成 | 无 |
| **P0-2 文档一致性回正** ✅ **已完成 2026-10-15** | ① ctest 计数统一（当时实测 **48**〔GPU 树〕/ **46**〔无 Vulkan 树〕；同日 P0-5 加 `sampling_test` 后为 **49** / **47**、P4-2b 再加 `scan_bitwise_test` 后为 **50** / **48**，口径与历史数字的处理见 §2 口径说明）；② `CMakeLists.txt` 版本与 tag 对齐（v1.8.2）；③ 补 `docs/release-notes/v1.5.1.md`（2026-10-15 依据 `git log v1.5.0..v1.5.1` 与 history 两条归档**补写**）；④ `README.md` 死链/mojibake/链式 API 声明/索引（已在前一轮迭代修完）。**⑤ 门禁补强（原 E8）**：`bench/doc_align_audit.ps1` 的 [A] 扩展到 markdown 链接目标，并修掉它当场抓到的 `todo/25` → `../CONTRIBUTING.md` 死链 | `bench/doc_align_audit.ps1` 的 [A][B][D][E][F] 可行动项为 0（2026-10-15 复测：[A] 0 / [D] 0 / [E] 0 / [F] 0）；文档中 ctest 计数统一写当前值（当时 48，现 49） | 小–中（纯文档 + 审计脚本一段） | P0-1（否则改动无法被自动验证） |
| **P0-3 BatchNorm 禁止静默回落** ✅ **已完成（真实现路线）** | **已实施**：`compute_layer_mlp.hpp` 新增 `BatchNorm` 层（训练/推理双态、running 统计 EMA + `extra_state` 序列化、`forward_recompute` 抑制 EMA），`make_norm_layer` 真分支；`scan_exprs` 模型 pass 补推理态 fwd+bwd 覆盖（GPU 闭合世界）；新增断言测试 `batchnorm_test`（+`--gpu` 变体），ctest 21 → 23，scan 结构 84 → 91（库内口径；样例收集器 93） | 构造 batchnorm 规格给出真层（可观测、不再静默降级）；ctest 含 `batchnorm_test`；详见 `docs/history/history.md`「BatchNorm 落地」条 | 已完成 | 无 |
| **P0-4 CMake install/export + CMakePresets** ⚡ **嵌入路径已完成（用户裁定拆分）** | **已实施**：门面目标 `neuralnet::nn`（`add_subdirectory` + 一行 link + 一个 include 即用；C++26/线程/`NN_HAS_VULKAN`/SPIR-V 头/Vulkan/shader 顺序全经接口传递）、`NN_ENABLE_GPU=AUTO\|ON\|OFF`、`NN_BUILD_APPS` 嵌入默认 OFF、`nn_enable_gpu_fusion` 树外可用（`NN_LIB_SOURCE_DIR/BINARY_DIR` CACHE 跨作用域 + GPU 不可用时 stub 报错）、消费方编译选项（`-Wno-pass-failed`/`-fexperimental-library`/`/utf-8`）随门面传递；样例 `examples/downstream/`（CPU/GPU 双路径实测通过）+ 指南 `docs/usage/05-consume-as-library.md`。**剩余**：`install()`/`find_package(neuralnet.cpp)`（用户裁定本轮不做）；`CMakePresets.json` ✅ 已于 2026-10-11 落地（`cmake --preset dev/portable/cpu/probes`，历史记录见 `docs/history/history.md` 的 presets 条） | 嵌入口径 ✅：干净构建目录里 `cmake -S examples/downstream` CPU 与 GPU（`NN_ENABLE_GPU=ON`）两路径均编译、链接、运行通过（GPU 路径含 scan 收集 91 条 = 库内口径），fusion 树外调用可用；install 口径待剩余部分 | 已完成（嵌入）/ 小–中（剩余） | P0-1（preset 需进 CI 才不腐化） |
| **P0-5 推理采样 top-k / top-p + 多轮对话闭环** ✅ **两半全完成（采样 2026-10-15 / 多轮闭环 2026-10-16）** | **采样半边（已实施）**：`nn::SamplingConfig`（temperature / top_k / top_p / repeat_penalty / repeat_window / seed）落在 `compute_layer_base.hpp`，共用采样器按"重复惩罚 → 温度 → top-k → top-p → softmax → 采样/贪心"固定顺序；`GPTModel`/`RAPTModel::generate` 新增配置重载（旧 temperature 重载保留、语义 = 全默认）；`text_infer` 暴露 `--top-k` / `--top-p` / `--repeat-penalty` / `--repeat-window` / `--seed`，GUI GPT 推理 Tab 同步；门禁 `sampling_test`。**多轮闭环半边（2026-10-16，P0-5 尾项）**：① prompt 构建唯一真源 = `Tokenizer::chat_prompt`（`domain_tokenizer_base.hpp`）：对话标记模板（`BOS<|system|>…<|end_of_system|><|user|>…<|end_of_user|><|assistant|>`，末轮 user 补 `<|assistant|>`）+ 内容两种给法（`text` 走 encode / `tokens` **原样灌入**）；超 `seq_len` 时从**最旧的一问一答整轮丢弃**（system 块与末轮恒保留、丢弃后仍以 user 起头），无标记词表退化为换行相接 + 截尾。② CLI：`--interactive` 从"每轮只拼当轮输入"改为**进程内累积历史**（助手轮用生成的原 token 回灌，超限打印丢弃轮数）；新增 `--turn <role>:<text>`（可重复，历史自带、一次一问）与 `--reply-out <path>`（回复落盘 → 调用方不解析日志）与 `--system <text>`；形状不合法（role 写错 / 不交替 / 末条不是 user / 与 `--prompt`·`--interactive` 并用）**解析期报错**。③ GUI：「GPT 推理」页新增「对话记录」面板 + **「💬 发送（多轮）」** + **「🧹 清空对话」**（历史 = `cli_controllers.py` 的 `ChatSession`，**无状态协议**：每轮把整段历史随采样参数一起传回 CLI）；字段表新增「系统提示（多轮对话）」（→ `--system`）；`TabBase._run_args` 抽出共用运行路径。**门禁**：`chat_prompt_test`（模板结构 / ★前缀不变式 / tokens 回灌 / 整轮截断 / 纯文本回退 / 真实 `text_infer` 子进程端到端：prompt tokens 与库逐 token 一致 + `--reply-out` == 生成 token 解码 / 7 条 CLI 负例）+ `bench/gui_cli_audit.py` 第 **[7]** 节（Python 侧参数面：第 N 轮 args 以第 N-1 轮为前缀、回复闭环、清空归零、`_on_send` 真的传历史）。**实测（三组「编译器 × 标准库」，2026-10-16）**：本机 clang 24 + libstdc++（`build/cpu`，Release + CPU-only）**52/52**；`102` WSL clang 24 + **libc++**（`~/nn-verify/build`）**52/52**；`102` WSL g++ 15.2 + libstdc++（同目录 `build-gcc`）**52/52**——三棵树均零告警（`-Werror`）且都含新用例 `chat_prompt_test`。本机 GPU 树（`build-gpu`，Release + Vulkan）新用例已构建并运行通过、`ctest -N` 注册 **54** 条，但**全量 GPU ctest 未跑**（本项结论只依赖 CPU 树）。`gui_cli_audit` 退出码 0（七节） | 采样：同 prompt 在 top-k / top-p 下输出可复现（同 seed）✅。多轮：**连续 N 轮不丢上下文** ✅ —— 可验证形式 = 第 N 轮的 prompt 以第 N-1 轮为**逐 token 前缀**（`chat_prompt_test` [2] 库级 + [6] CLI 端到端；GUI 侧 `gui_cli_audit` [7] 在参数层同款断言）；GUI 侧真机链路已实测（`ChatSession` → `GptInferController` → `text_infer`：第 2 轮 prompt = 第 1 轮 prompt + 第 1 轮生成 token + `<|end_of_assistant|>` + `<|user|>` + 新输入 + `<|end_of_user|>` + `<|assistant|>`）；超 `seq_len` 时按一问一答丢弃并有可见提示 | 中 | 无。命名冲突：`--topk` 属 `mnist_infer`、`--top` 属 `tokenizer_infer`；新开关取 `--top-k`/`--top-p`，无冲突。多轮开关取 `--turn`/`--reply-out`/`--system`，亦无冲突 |

> **口径说明（与本文 §4 前置清单的差异）**：任务书前置清单把"危险精度组合 fail-fast"列在 P0；
> 该项归 **P1**，且口径已定为**仅告警**（不退出、不加确认开关）——细则与验收一律见
> [`24-f16-training-defects.md`](24-f16-training-defects.md) §2.3 / §5 F1。本文以裁定台账为准，不重复排期。

### 4.2 P1（1–2 月：能力与健壮性）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P1-1 `text_train` 模块化** ✅ **已完成（2026-10-16）**<br>（2026-10-15 第一刀 = 退出码语义） | 抽 `train_loop` / `recovery` / `checkpoint` 模块；退出码语义修正（保存失败不得成功退出） | 现有关键路径回归逐字节一致；新增恢复路径单测；`std::exit` 只在 CLI 参数解析层出现 | 中（1746 行文件拆解，34 处 `std::exit`） | 无，但它是 P1-2 的前置 —— **已完成，P1-2 已解锁**。**落地形态**：`src/text_train.cpp` 1830 → **396 行**（只留接线与唯一出口 `main`），四模块落 `include/neuralnet.cpp/cli/`：`text_train_config.hpp`（`TrainConfig` + `--help` + argv 解析 =**唯一允许 `std::exit` 的层**）、`text_train_loop.hpp`（epoch × step + 非阻塞 loss 回读 + 梯度累积/裁剪 + 测试评估 + `--max-steps` + 定期断点 + `NN_NAN_TRACE`/`NN_MEM_STATS` 探针）、`text_train_recovery.hpp`（`--resume`：角色门禁 / 按文件规格重建 / 载参）、`text_train_checkpoint.hpp`（`.nnckpt` 路径派生 + 保存记账 + 收尾退出码 + `.nnmodel` 产出）。**三模块不终止进程**：错误一律 `Result` 上抛，`main` 是唯一出口（错误 → 打印 + 退出码 1）；探针中止经 `abort_code = 9` 回报（旧 `return 9` 语义不变）。**验收证据（2026-10-16，clang 24 + Release + `NN_ENABLE_NATIVE=OFF` 的 `build-rv`）**：① **关键路径逐字节一致**——21 个场景（f32 / `--f16` / `--model rapt` / `--t1_58` / `--grad-log` / `--checkpoint-every 1` / `--activation-offload` / `--accum-steps 2` / `step_cosine` / `--max-norm` / 单窗口多 epoch）改动前后 stdout/stderr **逐行相同**（仅时间字段归一）、退出码相同、`.nnmodel`/`.nnckpt` **md5 逐位相同**；② 错误路径文案仅 2 处有意变更（`Error: X` → `推理模型保存失败: X`、数据集打开错误去掉 `[nn] … (expr) at file:line` 噪声），角色门禁与断点失败文案**逐字未变**；③ 新门禁 `text_train_modules_test`（断点路径派生 / 角色门禁 / 断点往返参数逐位一致 / 记账与退出码 / 推理模型角色）；④ `ctest` 49/49（无 Vulkan 树；GPU 用例 77 = skip）、`doc_inventory` 新增第 [6] 节门禁（三模块 + `src/text_train.cpp` 不得出现 `std::exit`/`NN_EXIT`/`NN_UNWRAP`）PASS。过程与盘点见 `docs/history/history.md`「P1-1（第二刀）」。 |
| **P1-2 `.nnckpt` 训练状态段 + role 语义**（**2026-10-16 裁定扩写**，原名「checkpoint 优化器状态」；**进度：第一刀**〔v8 段布局 + 整文件校验和，门禁 `train_state_test`〕**+ 第二刀**〔`Optimizer::save_state/load_state` 全表，门禁 `optimizer_state_test`〕**已落地；训练路径接线 = 第三刀**） | `.nnckpt`（role=Checkpoint）增加 **role 门控的训练状态段**，`MODEL_VERSION` **7 → 8**（与 P2-1 校验和/尾部标记**同批升版**）：① **计数器**（global_step / epoch / epoch 内步号 / 当前 lr）；② **优化器状态** 经 `Optimizer::save_state/load_state`（Adam m/v **+ `t`**、SGD momentum、Muon、AdamWQuant 的 f16 状态与逐行尺度）；③ **训练配方**（精度 profile 四槽 + weight_quant、lr 调度配方与**曲线形状** `total_epochs/total_steps`、batch_size/accum_steps/max_norm/stride）；④ **数据集身份**（`.nndataset` 文件 sha256 + tokens/docs/windows/`steps_per_epoch`/loss_scope）。**role 语义（冻结）**：`Checkpoint` ⇒ 除"完全不改变训练"的参数外**一律禁止输入**（显式给出即报错，按**旗标名**判定、不看值；`--resume-epoch/--resume-step` 同在禁列），`Model` ⇒ **CLI 全权**（`--resume` 放开 `.nnmodel`，供 SFT warm start：优化器冷启动、step 归零） | 老文件（v5/v6/v7）缺段 = 冷启动语义；save→load→save SHA256 逐字节一致；TDR 模拟重启后动量非零、`t` 连续、loss 轨迹连续；断点自带精度 profile ⇒ 不给 `--f16`/`--t1_58` 也能完整还原；数据集身份不匹配**硬报错、无逃生阀**；禁列旗标显式出现即报错（单测覆盖） | 中–大 | **P1-1** ✅（已解锁）；P2-1 合批。**落点** = `model_serialization.hpp`（v8 段布局 + 尾部 sha256/完整性标记）、`compute_optimizer.hpp`（`save_state/load_state`）、`cli/text_train_{loop,recovery,checkpoint}.hpp` 接线、`bench/doc_inventory.ps1` 新增门禁、GUI 的 `--resume-epoch/--resume-step` 随 role 禁用 |
| **P1-3 危险精度组合启动期告警** ✅ **已完成 2026-10-13** | **不单独定义**——需求、口径、验收一律引用 [`24-f16-training-defects.md`](24-f16-training-defects.md) §2.3 / §5 F1（口径：仅警告，不退出、不加确认开关）。**落地**：`include/neuralnet.cpp/cli/cli_precision_warning.hpp`（`nn::cli::dangerous_precision_warning`）+ `text_train`/`mnist_train` 调用点 + 门禁 `src/cli_precision_warning_test.cpp`（单测组合矩阵 + 子进程捕获 stderr 断言退出码不变） | 见 24 号台账 F1 验收口径 | 小 | 无 |
| **P1-4 `gpt_test` flaky 定位** ✅ **已完成 2026-10-15** | **未复现（278 次直跑零失败，含 2 核超订）且同机逐字节确定** ⇒ 定性为"环境差异或进程级异常"，非测试随机性。**未加 retry/skip**（输出确定 ⇒ 失败必对应环境或代码变化）；改为加环境指纹行 + 复用 P0-1 的 `--output-on-failure` 日志留存。证据/统计口径/复现命令见 `docs/history/history.md` 的同名条目与本文 §3.1 C6 行 | 连续大样本直跑零失败 ✅；结论入 history ✅ | 小–中 | P0-1 ✅ |
| **P1-5 CI 冒烟扩到 MSVC Debug + 权重变化断言** ✅ **已完成（2026-10-18）** | Debug 构建冒烟；训练 N 步后断言参数**确实变化**（防"loss 不动"类静默失败）。**落地**：① `.github/workflows/cmake-single-platform.yml` 新增 `msvc-debug` job（`windows-latest` + Visual Studio 17 2022 生成器 = **cl.exe**；`NN_ENABLE_GPU=OFF` 纯 CPU，避开 Vulkan/glslc 变量；`NN_ENABLE_TESTS=ON`，`cmake --build`（Debug 配置）后 `ctest -C Debug`（C11 除外））；② `src/optimizer_test.cpp` 新增 [4] **训练冒烟**：小 MLP + Adam 跑 8 步，断言 loss 有限且下降、全参数 L1 位移 > 1e-3；同一 `detect_movement()` 的负例（同一快照自比）必须判为"无移动"，证明断言不是恒真。该段纯 CPU f32，无 Vulkan 也**不会 77 跳过**；③ 本机 `build/cpu`（Release + CPU-only）与 `build/p1-5-debug`（Debug + CPU-only，全量构建共 98 个 ninja 步骤）均实测 `optimizer_test` ALL PASS，`loss ≈3.016 → 0.1555（8 步），参数 L1 位移 41824.5`；④ Debug 全量 ctest 触发既有的 **C11**（RAPT f16 NaN；用户已排查，怀疑设备/后端自身问题，Release 通过只是概率）⇒ CI 的 Debug 测试步暂以 `-E "^rapt_test_f16$"` 排除并在 C11 登记，其余用例全跑；⑤ **MSVC 102 实机验证（2026-10-18，`ethan@192.168.1.102`，全新目录 `nn-p1-5-msvc`）**：VS 2026 18.10.4 / **cl 19.51.36260.0** / CMake 4.3.3，Debug + CPU-only。首次构建 `text_train.cpp` 触发 **C1128**（补 `/bigobj`，目录级 + 接口选项）；ctest 又触发 **9 个 0xC00000FD 栈溢出**（`t1_58`/`bitnet_struct`/`model_roles`/`batchnorm`/`model_spec_validation`/`text_train_modules`/`train_state` + 2 个 GPU 变体，MSVC 默认 1 MB 栈；补 `/STACK:8388608`）。修复后 CI 口径 ctest **51/51 通过 / 0 失败 / 180.00 s**（`rapt_test_f16` 按 C11 排除，20 条 GPU/依赖 app 用例 skip；该用例在 102 MSVC Debug 上单独跑 **3/3 通过**，排除是为避免概率性红）；另跑**不排除的完整 ctest 52/52 通过 / 181.61 s**（`rapt_test_f16` 2.33 s 通过）；`optimizer_test` MSVC 实测 `loss 2.773293 → 0.170448（8 步），参数 L1 位移 41965.2` | CI 中 MSVC Debug 目标编译通过；权重变化断言可检出冻结（含负例）；**C1128 已实测触发 ⇒ `/bigobj` 已加；栈溢出已实测 ⇒ `/STACK:8388608` 已加** | 中 | P0-1。复用 `f16_writeback_probe` 的"8 步 Adam 参数必须移动"思路，但改跑纯 CPU f32 以覆盖无 Vulkan 的 Debug job |
| **P1-6 自定义层使用文档 + 精度自检**（文档部分 **✅ 已完成 2026-10-11**；精度自检 R2 **✅ 已完成 2026-10-14**） | ① 文档：已落 `docs/development/write-a-custom-layer.md`（层骨架模板 + 可复制 gradcheck 片段 + 七条契约 + GPU checklist；示例层经数值梯度检查实测通过 `max|num−an| = 4.7e-06`、50 步 Adam 收敛）。② 代码（**已完成**）：`Layer::init` NVI 在 `init_impl` 之后统一执行 `check_param_precision_()`——层声明的 `param` 槽是存储精度时，逐个校验 `parameters()` 的张量实际存储精度与之相同（`precision_bytes == 0` 的 T1_58 等非存储精度声明跳过）。**它当场抓到了 3 处真实静默降级**：Conv2D 的 w_/b_、RAPT 的 token_emb_、以及 `ReLULinearAttention` **从未把 profile 转发给它自持的 4 个 Linear**（RAPT 注意力投影权重在 `--f16` 下一直是 f32）——前两处在 P2-3 修掉，第三处由本自检在扫描期报错后发现并修复（现 47/47 全绿） | 文档：按模板能写出可训练的自定义层；自检：硬编码 F32 的层在 init 期被拒绝 | 小 | ✅ **已完成 2026-10-14** |

| **P1-7 三值模型的续训入口（打包 → latent 反量化载入）** ✅ **已完成（2026-10-16）** | 让 `--resume <x.nnmodel>`（role=Model，三值层为打包形态）能进入**可训练态**：载入意图 `WeightLoadIntent`（与文件角色正交）分两路——`Inference`（默认）= 部署只读态、`Trainable` = 序列化层按文件 τ 把打包权重**反量化**成 f32 latent（`ternary_pack.hpp` 的 `latent_row_scale`/`dequant_row`，宿主侧逐行解码）+ 层按文件 τ **播种**量化缓存，不再落到 `packed_only_`。**反量化取量化器的不动点 `L = trit·(τ·K/n)`**（n = 该行非零 trit 数）：字面的 `trit·τ` 会让逐行 absmean 变成 `τ·n/K` ⇒ 续训一开始就是有效权重的尺度跳变（见「为什么不是 `trit·τ`」）。**层协议不变**：反量化后用 `TernaryLatent + aux(τ)` 交回 `import_param`，dense / `.nnckpt` 路径逐位不变 | 反量化载入后首次 forward 的 wq/τ 与文件内打包权重**逐位一致**（量化确定性：`model_roles_test` [6] 位级比对打包字 + τ）；随后 backward 可跑、8 步 Adam 参数确实移动（同一用例 + `text_train_modules_test` 的三值 GPT 端到端）；dense 文件两种意图参数逐位相同；逐行 `absmean(latent) == τ`（不动点，实测最差相对偏差 5.0e-08，`ternary_gemm_test` [8]） | 小–中 | ✅ **已完成 2026-10-16**；前置 P1-2 的 role 语义已落地。落点 = `ternary_pack.hpp`（反量化 helper）、`model_serialization.hpp`（`param_blob_to_entry` 的意图分流 + `load_model` 新实参）、`compute_layer_bitlinear.hpp`（播种缓存）、`cli/text_train_recovery.hpp`（删 `check_resume_target`、改传 `Trainable`）、`src/mnist_train.cpp`。验收证据见 `docs/history/history.md`「P1-7」条 |

> **为什么不是 `trit·τ`（P1-7 落地时的实现口径，2026-10-16）**：本项目量化器的尺度定义是
> **逐行 absmean**（`τ = (1/K)·Σ|W|`），所以"把 trit 乘回 τ"得到的 latent **不在量化器的像里**：
> 它的逐行 absmean 是 `τ·n/K`（n = 该行非零 trit 数，≤ K）⇒ 第一次参数写回后重算 τ 就是
> `τ·n/K`，有效权重 `wq·τ` 随之缩 `n/K` 倍 —— 对 warm start 而言这是"载入后第一步就换了模型"。
> 取 `L = trit·(τ·K/n)` 则两条不变量同时成立：行 L1 质量 = `K·τ`（⇒ `absmean(L) == τ`，与训练时
> latent 的定义同源）且非零幅值 `τ·K/n ≥ τ > 0.5τ`（⇒ 再量化复现**同一组 trit**）。首次 forward 与
> 文件逐位一致由"层按文件 τ 播种量化缓存"保证（`τ` 与容器字都逐位来自文件，不经过重算）。

> **P1-2 状态（2026-10-16）：✅ 四刀全部落地** —— ①v8 段布局 + 整文件校验和（含 P2-1）②`Optimizer::save_state/load_state` 全表
> ③训练路径接线 + A/B 禁列 + 数据集身份硬错门禁 + role 语义（`.nnmodel` warm start ）④GUI 收口。
> 验收口径逐条见 §8.2 门禁 7；过程与实测见 `docs/history/history.md` 的四条「P1-2（第N刀）」；上面 P1-2 行内的"进度"字样以本条为准。

### 4.3 P2（3–6 月：结构优化）

| 项目 | 目标 | 验收口径 | 成本 | 依赖 |
|---|---|---|---|---|
| **P2-1 `.bin` 整文件校验和 + 尾部完整性标记** ✅ **已完成（2026-10-16，随 P1-2 同批交付）** | 尾部加 `[sha256 32B][magic "NNEND"]`（边写边算，`core_sha256.hpp` 的增量 `sha256_update` 已就绪）；`MODEL_VERSION` **7 → 8**（与 P1-2 同一次升版）。**已实施（2026-10-16）**：v8 尾部 `[sha256 32B]["NNEND" 5B]`（边写边算）+ `verify_model_integrity` 在 `load_model` / `load_train_state` 先行校验；门禁 `train_state_test` [5]（完好 / 翻一字节 / 截断三态）与 §8.2 门禁 7 的 ④ | 截断/损坏文件明确报错；合法文件 round-trip SHA256 一致 | 小–中 | P1-2（同一版本段改动，合并做省一次升版） |
| **P2-2 Adafactor 类无 m/v 优化器** | 新增不物化二阶矩的优化器，训练更大模型 | 同任务对比 Adam 的显存下降与收敛曲线；`create_optimizer` 可创建 | 中 | 无 |
| **P2-3 RAPT / CNN f16 接线** | ✅ **已完成 2026-10-14**。① RAPT：**27 处** `dsl::compute*` 补精度实参 `p_.compute`（此前全部落默认 F32）；两个残差分支从硬编码 `Precision::F32` 改为 `p_.compute`（与 GPTBlock/TransformerEncoderLayer 同口径）；`token_emb_`/`grad_token_emb_` 改 `p_.param`；`rms_norm_forward_/backward_` 的输出张量从默认 F32 改 `p_.compute`（**否则 dst/src 精度错配 → `insert_rows` 断言**）。② Conv2D：`w_`/`b_`/`grad_w_`/`grad_b_` 改 `p_.param`，4 处 `dsl::compute*` 补 `p_.compute`。③ `CnnConfig` 增 `precision` 字段，`build_cnn_model(_from_spec)` 在 add 之前注入（新增可选 profile 形参）。**门禁**：`cnn_smoke_test`/`rapt_smoke_test` 增 `--f16`；ctest 新增 4 条 `cnn_test_f16(_gpu)` / `rapt_test_f16(_gpu)`（断言 loss 有限且下降、梯度无 NaN） | f16 下 CNN/RAPT 训练健康收敛；默认 profile 的 f32 逐位不变（param/compute 默认 F32） | 中 | ✅ **已完成**（ctest 47/47） |
| **P2-4 Dataset / DataLoader 抽象 + 流式文本**（**前半 Dataset 已落地** = 19 号设计阶段一~三；本条只剩**流式后半**：DataLoader/分块迭代 `for_each_chunk`，不再把语料整段读进内存） | 流式加载（Dataset 格式与统一只读类见 `docs/todo/19-unified-dataset.md`） | 大文件训练内存曲线平稳；`dataset_test` 逐位一致口径回归 | 中 | 无 |
| **P2-5 Python / C 绑定** | 提供稳定 C ABI 或 pybind 绑定，供生态集成 | 绑定层跑通训练/推理最小示例；ABI 报错信息可读 | 中–大 | **条件触发**（§6） |
| **P2-6 GUI 声明式字段表 + 绘制节流** ✅ **两侧均已完成（2026-10-15）** | **已实施（绘制半边）**：一笔一个画布项（`_paint_live` 原地 `coords`）+ 空帧跳过 + 视觉刷新按帧节流（`_PAINT_MS = 16`，数据侧不节流）+ PIL 光栅化延迟到 `_downsample_buffer`（读缓冲唯一入口）。**实测**：合成 2000 motion 事件 16.6 ms vs 旧写法 37.9 ms（2.3×），画布项 2058 → 59。**✅ 字段表半边已完成（2026-10-15）**：`Field` 声明式表 + 纯函数 `collect_fields` + `ParamsMixin`；6 个 Tab 全部迁移（`engine→--gpu` / `precision_preset→--f16/--t1_58` / `input_method→input` 这类后处理显式保留）；**逐字段等价性验证**：同一个 fake self 驱动 HEAD 版旧 `collect_args` 与新表实现，22 个场景中 17 个逐字段一致、4 个是有意收紧（fixed 调度下不再发被隐藏的 min_lr/warmup）、1 个命令行层等价；审计新增 [5] 节（表 ↔ Controller 闭包），并当场抓到 `TabBase` 遗留 `collect_args` 桩覆盖 mixin（4 个页会一个参数都不发）等 2 个真 bug | 现有 Tab 参数零遗漏（`bench/gui_cli_audit.py` 退出码 0，五节 ✅）；拖动开销下降 ✅ | 小–中 | P0-5（采样新开关已定型 ✅） |
| **P2-7 超大 header 拆分** | backend 拆 fence / staging / dispatch / 算子为独立 TU；`compute_vk_backend.hpp` 瘦身 | 拆分前后 build 全绿 + ctest 全绿 + CPU/GPU 字节锚逐位一致 | 中–大 | **前置 = 先补编译/链接期冒烟**（P0-1 + P1-5），否则拆分无回归网 |
| **P2-8 F16C 半精度转换快路径** | ✅ **已完成 2026-10-14**。`detail::float_to_half_bits` / `half_bits_to_float` 增加 `#if defined(__F16C__)` 快路径：**运行时**走硬件单指令（`_cvtss_sh`/`_cvtsh_ss`，imm8 = `_MM_FROUND_TO_NEAREST_INT` → 与 MXCSR 无关、恒 RNE），**常量求值**与 `NN_ENABLE_NATIVE=OFF` 的可移植构建仍走软件位操作（`std::is_constant_evaluated()` 分流；两条路径语义同源 ⇒ constexpr 求值不受影响）。**新增门禁**：`precision_test` 增「快路径一致性」段——用 23 条 f32→f16（±0 / 次正规 / 2^-25 平局 / 65504/65519/65520/2^16 上溢边界 / ±inf / NaN 负载）与 15 条 f16→f32 位型，把**编译期软件路径**与**运行期硬件路径**逐位对比。**实测**：CPU GPT f16 训练 10 步 **3.6 s → 1.9 s**（f32 1.9 s，即 CPU f16 不再慢于 f32）；`NN_ENABLE_NATIVE=ON`（F16C）与 `OFF`（portable）两态均通过；ctest 47/47 | CPUID 探测 + 两态构建 + 逐位对拍 | 小 | ✅ **已完成** |
| **P2-9 训练 step 内固定 I/O 开销归因与消除** | ❌ **已评估-不做（2026-10-14，实测推翻前提）**。原判断"每步约 20% 墙钟在 `copy_from` ⇒ 可消除的固定开销"不成立：① **host 等待总量不变、只是位置可移**——把 staging ring 从 `64MB×4` 改成 `8MB×16`（消除 acquire 阻塞）后 `copy_from` **392 → 3 ms**，但 `flush_batch` 同步从 **955 → 1213 ms**，两者之和恒 ≈1347 ms；② 设备忙率 95.9%，host 本就在等 GPU，减少等待点不缩短墙钟（改后 3 轮墙钟 2200/2059/2053，与原 2044 在噪声内，故**已回退该改动**）；③ "减少每步帧数"方向相反——`--flush-interval 1 → 4`（flush 次数 105 → 20）墙钟 **2044 → 2096 ms（+2.5%）**，因为大帧降低 host 录制与 GPU 执行的overlap（默认 1 已是实测最优，见 text_train 注释）；④ 真正的天花板 = `墙钟 − 设备 kernel` ≈ **86 ms/15 步 ≈ 4.2%**，且 f32/f16 同幅、与精度无关。结论：该项**没有可回收的收益**；保留默认配置。 | 归因到具体调用点与每步 drain 次数（✅ 已完成，见左）；消除后 e2e step 下降（❌ 实测无收益） | 无 | 低 | ❌ **已评估-不做**（证据见左；[`26`](26-f16-speedup-plan.md) §4 S0） |
| **P2-10 原生 f16 算术（`shaderFloat16`）用足** | 🟡 **已评估-不再扩张（2026-10-14，附实测）**。口径：原生 f16 **允许启用、不排除**——`#a` 在运行期查找序里**优先级最高**，能力经 `shaderFloat16` 查询 + 启用（逃生阀 `NN_VULKAN_NO_16BIT_ALU`），**40HX 实测已有 9 条结构命中 `#a`**。**实测增益**（40HX，`d64/h4/L4/ff256/seq256/b64`，f16，15 步 ×3 轮，`NN_VULKAN_NO_16BIT_ALU=1/0` 配对）：ALU 开 **1005 ms** vs 关 **1021 ms** ⇒ 现有 native16 覆盖贡献 **≈1.5%**。**为什么不再把资格扩到 fold / matmul / reduce**：① 这三类的耗时主体是**归约累加循环**，而累加按数值要求必须保持 f32（本仓 §7.2 与业界 fp16 GEMM 同口径）⇒ 扩资格只能覆盖其中的**逐元素**片段，量级与已覆盖的 9 条同构（≈1%）；② 其余排除项（`rparams` / 微常量 / 行-列-批索引值）**是原则性的**——rparam 恒 f32 传递（放宽会把整链隐式提升回 f32，收益归零），微常量与索引在 f16 下会 flush/失精（放宽即静默错值）；③ 因此原判断"最重的几条链永远拿不到原生 f16 ⇒ 收益大"**不成立**：那几条链的成本在 f32 累加循环里，不在 f16 可用片段里。**保留**：`#a` 优先级、`NN_VULKAN_NO_16BIT_ALU` A/B 逃生阀、单元与端到端回归全部不变（`f16_precision_test` / 40HX 的 `#a` 命中实测） | 40HX 上 `--f16` step ≤ f32（✅ 已达成：1013 vs 1027）；原生开更快（✅ 实测 +1.5%，但扩张收益有界 → 不做） | [`26`](26-f16-speedup-plan.md) 的 **S5** | 低 | 🟡 **已评估-不再扩张** |
| **P2-11 编译期精度特化（去运行期分派三元）** | ✅ **已完成 2026-10-13**。把"精度是运行期参数"（`#x`）在热路径换回"精度是编译期常量"：新增 `key#s`（全输入 f16 + 输出 f16，无条件发射）与 `key#xxxx`（扫描期**真实观察到的签名**：`scan_exprs` 的 dry-run/模型 pass 各跑 f32+f16 两遍，`scan_register` + CPU 引擎 `eval_expr*` 按真实张量精度 + 真实 P 记入 `ExprRegistry::variants`）；运行期查找顺序 `#a` → `#xxxx`/`#s` → `#x`（未枚举到的签名照旧兜底）。**只需 16 位存储，不需要 `shaderFloat16`** | 850M：`--f16` 2260 → **2055 ms/15 步**（f32 2037，+10.6% → **+0.9%**），峰值显存 818 → **606 MiB**；`ctest` CPU 41/41 + GPU 43/43；构建期 110 条签名 → 88 条特化 | 无 | ✅ **已完成**（[`26`](26-f16-speedup-plan.md) §4 S1s、[`24`](24-f16-training-defects.md) §4.5） |
| **P2-12 优化器 f16 批量派发（`#b`）** | ✅ **已完成 2026-10-14**（两段：`#b` 变体发射 + "scan 与运行期同源"修复）。① `#b` 对**每个 extras 为空的静态签名变体**各发一份（键形如 `key#0002#b`；extras 的元素类型不在 `ExprPrecSig` 编码里 → 带 extras 的 f16 结构仍逐 kernel 降级）；运行期组键带精度身份、实例缓冲按 `t.precision()` 取槽位。② **残差收口**：Adam 复合 kernel 在 `--f16` 下逐成员降级的根因是 **scan 与运行期不同源**（不是派发器）——(a) scan 的优化器 dry-run 把当前 pass 的 precision profile 透传给 `create_optimizer`（否则 f16 pass 融合出的复合结构与运行期不同 ⇒ 命中不到 ⇒ 降级）；(b) scan 的 CPU `end_expr` 为**复合 kernel** 补记运行期精度签名（单节点有 `scan_register`/`eval_expr*` 入口，融合结构只在 GPU `end_expr` 才算出签名）。**实测（40HX，f16，`d64/h4/L4/ff256/seq256/b64`，20 步，`329f8f1` vs `5bb6a51` 配对）**：`[ir-c][degrade]` **70 → 0**/步；优化器派发 **138.7 → 8.55 calls/步（−94%）**（每步恰 3 组批量：K1 `#0002#b` / K2 `#0002#b` / 复合 `#10004#b`，各 32+32+6 实例）；优化器 host 阶段 **9.94 → 6.51 ms/步（−34%）**；f16 loss 轨迹与 f32 同步（850M 9.009→8.594 vs 9.006→8.591）。**端到端墙钟无可见收益**（1320.95 → 1324.08 ms，噪声内）——设备忙率 95.5%，host 节省被 overlap 吃掉。字节锚（cpu/40HX/AMD iGPU）逐位不变。**边界**：只覆盖 scan 真跑过的 profile（f32 + `--f16`）；其它显式组合（`stable=f16`+`optimizer=f16`）仍按设计回落降级 | f16 下优化器回落批量派发；残差收口后**每步 0 降级、派发 −94%**（e2e 收益在 GPU-bound 下不可测） | P2-11 | 低 | ✅ **已完成 2026-10-14** |
| **P2-13 f16 的 K 切分 / fine tile 变体** | ✅ **已完成 2026-10-14**。把 sig 穿进 `#f`/`#ks`/`#kr` 生成路径：每个**静态签名变体**各发一份网格粒度变体（键形如 `key#10003#ks`）；**部分和缓冲恒 f32**（后端按 f32 分配 → `#ks` 输出声明钉成 `float`，sig 只决定 A/B 读取类型），末段由 `#kr` 按 sig 落盘；后端"网格变体无需 16 位存储"的豁免收窄到**纯 V0 键**。**实测**：40HX（b64）该 wgrad **f16 16.0 ms/块 vs f32 25.1 ms/块**（此前 f16 79 ms/块）；40HX 端到端 `--f16` **1013 ms vs f32 1027 ms（−1.4%）**，850M（b16）`--f16` **2005 vs f32 2038 ms（−1.6%）** | 40HX 上 `--f16` step ≤ f32；850M 不回归 | 中 | ✅ **已完成**（[`24`](24-f16-training-defects.md) §4.5、[`26`](26-f16-speedup-plan.md) §4 S3） |
| **P2-14 全流程真 f16（恢复计划）** | 🟡 **活跃（2026-10-18 恢复立项）**。F5 后四字段全 f16 已**可训练**，但还不是真 f16（CE 标量仍是 F32 补丁、matmul/归约仍有 f32 累加、native16 未全覆盖）。本项要求：**B-Mem = 训练/推理/checkpoint/model 文件无 f32 持久张量**；capable 设备另追 **B-Arith = 零 f16↔f32 转换**（补偿 kernel 的局部 f32 逐条豁免）；incapable 设备转换只由计算引擎在算子内部隐式完成（Layer/DSL/CLI 零手写 cast）。包含一键 `--f16-all`、native16 全覆盖、f16 累加/CE 标量改造、logits/CE 融合、**R2.5 数值鲁棒性技术栈**（SR / GradientStabilizer / ECO / attention δ 修复，见 26 §8.B.7）、全模型/全设备长跑与转正。原 S1/S5 的“已评估-不做”只针对当时的**提速**目标；S1 重做前必须先让 `scan_exprs` 覆盖 CE backward | 见 [`26`](26-f16-speedup-plan.md) §8.3 R0–R6、§8.B.7 与 §8.B.4 验收矩阵；B-Mem 必达；B-Arith = capable 设备 `cast`/`cast_into` = 0、f32 临时量 = 0（补偿 kernel 豁免表）；incapable 设备转换只在引擎内；GPT/MLP/CNN/RAPT × CPU/850M/40HX 长跑无 NaN、loss 下降、权重移动 | 中–大 | F5 ✅；RAPT f16 C11（§3.1）是 R5 前置 |
| **P2-15 标准 `--f16` 训练（参考 PyTorch AMP）** | 🟡 **活跃（2026-10-18）**：A 路线，目标是把现有 `--f16` 做成可解释、可对拍的标准混合精度——`profile_f16()`（f16 参数/激活 + f32 stable/optimizer）为默认，`profile_master_weights()`（f32 主权重 + f16 计算）为更接近 AMP 的可选形态；补动态 loss scaling（GradScaler 语义）、敏感链 f32 与 PyTorch AMP 对照。**不要求零转换/native16 全覆盖** | 同数据/同 seeds：`--f16` 与 f32 轨迹在冻结容差内、无 NaN/冻结、权重移动；与 PyTorch AMP 同量级；loss scaling 的 growth/inf-skip/恢复用例通过；峰值显存低于 f32，step 不慢于 f32 | 中 | 无（可与 P2-14 并行；A1.1 的 loss scaling 可被 P2-14 复用） |

> **原生 f16 口径（2026-10-18 更新）**：`shaderFloat16` 通路**允许启用、不排除**——`#a` 变体
> 在运行期查找序里**优先级最高**，设备能力经 `shaderFloat16` 查询 + 启用，逃生阀
> `NN_VULKAN_NO_16BIT_ALU` 可关；40HX 实测已有 **9 条结构命中 `#a`**。**P2-10 的开放部分是
> "把资格扩到 fold / matmul / reduce"**。⚠ **口径反转（2026-10-18）**：旧“累加必须保持 f32”
> 只适用于 F4 提速路径；**真 f16（P2-14 R4.1）要求 capable 设备 native16 全覆盖且零 f16↔f32
> 转换**，P2-10 的“资格不再扩张”不再是当前结论。

> **2026-10-18 数值鲁棒性补充**：P2-14 的口径拆成 **B-Mem（硬门禁：无 f32 持久态）** 与
> **B-Arith（capable 设备附加目标：零转换，补偿 kernel 局部 f32 逐条豁免）**。SR、
> GradientStabilizer、ECO、Unit Scaling、Peri-LN 与 attention `δ` 修复的证据/边界/落地顺序
> 见 [`26`](26-f16-speedup-plan.md) §8.B.7；其中 SR/GradientStabilizer/ECO 进入 R2.5，
> Unit Scaling/Peri-LN 是 R5.1/R5.2 的模型级实验，不阻塞 R1–R4。

### 4.4 P3（择机）

P3 不排期，只由 §6 的触发条件启动：S6 自动窗口、M7 存储多态、IR-D 第二 emitter、
ZiPT 恢复、LRLA、CUDA、分布式（7 项）。
**IR-C 跨表达式图融合已不在本档**——它已落地为现行能力（`begin_expr/end_expr` + 图融合，
已接入 Adam/AdamW），见 §3.4 M3 与 [`03`](03-ir-optimization.md) §7。

### 4.5 P4（持续优化方向，2026-10-14 立项）

> **来源**：用户 2026-10-14 指定的四条长期方向。**不是新里程碑，而是"底座已稳后继续加杠杆"的常设项**——
> 每条都按"同窗配对 A/B + 字节锚逐位不变"验收，避免"继续提升"退化成无口径的口号。
> **与既有裁定的关系**：P2-9（step 内固定 I/O）已证**无可回收收益** ⇒ P4-1 **不再从"搬 I/O 位置"找收益**；
> P2-10（原生 f16 资格扩张）已评估不做 ⇒ P4-2 **不含"扩 native16 覆盖"**。凡与 §5/§7 冲突的做法先改裁定再实施。

| 项目 | 现状（实测锚） | 方向 | 验收口径 | 成本 | 关联 |
|---|---|---|---|---|---|
| **P4-1 性能利用率继续提升** | GPU 忙率 ~95.5%（40HX f16）/ ~95.9%（850M）；`墙钟 − 设备 kernel ≈ 4.2%`（P2-9，f32/f16 同幅、与精度无关） | ① 把剩余 4.2% 归因到 host 录制 / driver 提交 / fence 等待三处；② 提高设备端并行（帧环深度、descriptor 复用、队列提交粒度）；③ CPU 线程池利用率（小算子并行度与分块） | `NN_PROFILE=1` 同窗配对 A/B：墙钟下降 ≥3% 且 busy% 上升；字节锚逐位不变 | 中 | P2-9（**已否决**的搬 I/O 不重复）；`core_profile.hpp` |
| **P4-2 算子性能继续提升** ✅ **C10 派生的 (a)+(b) 均已完成（2026-10-15）** | `layer_bench` 已提供 op 级配对 A/B（`--op` … + `--dk`/`--hash`）。**(a)** `rms_norm_forward_` 逐头 1/rms 收集改设备侧拼装：step 741.81 → 521.68 ms（−29.7%）。**(b)** `scan_prefix_outer` 专用路径（dk ≤ 64）改**行归属 + 4 路展开 + `dk ≤ 32` 窄共享变体**：单次 **7.4–7.9 → 1.64 ms（4.7×）**、`layer_bench --op scan_prefix` 配对 A/B **5.9×**（10.9 → 1.85 ms @ dk=32）、同会话 e2e RAPT **step 888.24 → 510.24 ms（−42.6%）**；数值**逐位不变**（位级哈希 + `rapt_test` stdout md5 + 新门禁 `scan_bitwise_test`）。**新瓶颈（下一手）**：host 侧 staging 等待 `copy_from` ≈66% 墙钟（与 GPU 忙时成正比 ⇒ 与 P2-9 的「host 等待总量不变」同机制）。`scan_suffix_outer` **刻意不改**（流量/争用受限，加占用率实测更慢）。 | 按 op 逐项调参（tile / occupancy / 网格粒度）与算法（三值 GEMV、扫描原语）；**不含**扩 native16 覆盖（P2-10 已裁定不做） | `layer_bench --op <name> --hash` 配对 A/B 提效、无系统性回退（噪声 ±6%）；端到端不回归。**(b) 已达**：−42.6% 且逐位不变 | 中（shader 迭代可只重建 `layer_bench` ≈2 min：PCH + 1 TU；全量重建 ≈20 min 只在最后做一次） | P2-11/P2-13；`docs/todo/02`；C10 归因（2026-10-15）；细节/消融/否决项见 `docs/history/history.md`「P4-2 (b)」条 |
| **P4-3 显存开销继续压低** | 04 号 L1（激活重计算）/L2（池归还）已实施；三值权重已压 param 侧（4 B → <2 B）；峰值显存（850M f16）606 MiB | 立项 04 号 **L3**：① Adafactor 类无 m/v 优化器（= P2-2）；② 注意力/softmax 子图进一步融合、消除 Q/K/V 激活集物化；③ 优化器状态低精度（`AdamWQuant` 已存在，4 B → 与三值同量级） | 同任务峰值显存下降，loss 曲线与基线一致（gradcheck 全绿） | 中–大 | `docs/todo/04`（L3 待立项）；P2-2 |
| **P4-4 API 易用性 + Fold 表达式自动化扫描** | ✅ **① fold 自登记已完成 2026-10-15**：注意力 fold 组合改由**定义方**自登记（`compute_layer_attention.hpp` 的 `attn_fold_anchor::register_all()` 在静态初始化期经 `FoldAnchor` 登记「掩码清单 × {无偏置, 位置偏置} × {MHA, GQA}」，掩码清单 `NN_ATTN_MASK_KIND_LIST` 是 X 宏单一真源 → 枚举与登记表同源）；`expr_fold.hpp` 末尾同款登记它的 3 个 v1 样例；`scan_exprs.cpp` 的 fold 显式登记块**整体删除**（消除"手写登记块漏项 = GPU 闭合世界硬报错"这类风险）。**实测**：结构集合逐条（key + 指令摘要）与改动前零差异（126 条）；`-DNN_SCAN_NO_ANCHOR` 下这 13 条 fold 结构全部消失（126 → 91）⇒ 自登记是唯一来源；新回归 `expr_fold_test` §8 锁代表值契约。② 之一 `dsl::compute` 无形状重载 **✅ 已完成 2026-10-15** | 剩余 ②：GUI 声明式字段表（= P2-6）、CLI/GUI/文档一致性 | ① 验收已达成：新增掩码/组合**无需改 `scan_exprs`**（`NN_SCAN_DUMP_SOURCES=1` 逐条来源可观测）；结构数不丢（逐条对比零差异）；`bench/gui_cli_audit.py` 退出码 0（本机实测 0） | 小–中 | `docs/todo/02`（fold 登记机制）；P2-6 |

## 5. 13 号清单逐条裁定表

条目编号沿用 13 号原文的章节号（13 号文档本身已于 2026-10-11 删除）。
其原正文的归档位置：**已执行/已否决记录**在 `docs/history/history.md`「13 号清单合并归档」节；
**仍未决的条目**由下表承接。本节逐条覆盖其**全部 16 条**未决项（含 17 §5 的 M7），一条不漏。

| 条目 | 裁定 | 理由与触发条件 |
|---|---|---|
| 13 §1 线程池归约分块统一 | **不做** | 归约已是确定性分段且 1/8-worker/串行逐字节一致；剩余差异（`parallel_transform`/`parallel_for_*` 用 `chunk_count()`，`row_reduce`/`col_reduce` 各自行块）**无正确性问题，只是概念一致性**。历史上 `Tensor::cpu_get_ptr` 的 8-70× 教训说明动热路径风险远大于收益。归档见 `docs/history/history.md` |
| 13 §2 超大 header 拆分 | **做（P2-7）** | 唯一被确认的日常成本是增量编译。**前置**：先补编译/链接期冒烟（P0-1 + P1-5），否则拆分期间 ODR/include 次序问题无网可抓 |
| 13 §3 `text_train` 模块化 | **✅ 已完成（P1-1，2026-10-16）** | 原判据：`src/text_train.cpp` 1746 行、34 处 `std::exit`，CLI/训练/恢复/checkpoint 混杂；退出码语义有真实缺陷。**它同时是 P1-2 的前置** —— 已拆为四个 `cli/text_train_*.hpp` 模块 + 396 行入口，见 §4.2 P1-1 行 |
| 13 §4 gui.py 声明式字段表 | **做（P2-6）** | 实测 6 个 Tab、7 处 `collect_args` 手工罗列，漏参风险真实。与 §5.3 绘制节流同批做（同一文件、同一批次验收） |
| 13 §5.1 F16C 快路径 | **条件触发（P2-8）** | 纯软件位操作在 f16 热点链上是真实开销，但收益未量化。**触发条件（三条同时）**：① CPUID 探测可行；② `NN_ENABLE_NATIVE` ON/OFF 两态都能构建；③ `f16_cpu_probe` 全量对拍逐位一致。历史次正规移位 UB 教训（`docs/development/05-mixed-precision.md` §12.12）要求对拍先行 |
| 13 §5.2 CPU f32→f16→f32 双趟 | **不做** | 该双趟是**量化模拟语义**，删除会改变行为；收益是热路径一次全量转换，属低量级。若将来有真实调用方依赖"直接返回"语义，另立 API 区分而非删除 |
| 13 §5.3 GUI 绘制节流 | **✅ 已完成（2026-10-15，P2-6 绘制半边）** | 原表述"每次 motion 全量降采样重绘"与实测不符（当时已在 render 端节流）；真正的拖动开销 = 逐事件建画布项 + 逐事件 PIL 画线，已改为一笔一项 + 空帧跳过 + 按帧节流 + 光栅化延迟，见 §3.3 X5 与 `docs/history/history.md` |
| 13 §7 checkpoint 优化器状态 | **做（P1-2）** | TDR 重启动量归零是**真实功能缺陷**（C4）。设计已明确：可选段 + 老文件缺段 = 冷启动 + `MODEL_VERSION` 升版。**依赖 §3 先拆**。**2026-10-16 裁定扩写**：段内容 = 计数器 + 优化器状态 + 训练配方 + 数据集身份，role 门控，**v7 → v8**（与 P2-1 同批）；role 语义 = `Checkpoint` 禁 A/B 输入 / `Model` 全权（含 `.nnmodel` warm start，SFT）——细则见 §4.2 P1-2 行与 [`23`](23-library-design-cleanup-and-model-file-roles.md) §2.1.1 |
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
| ~~IR-C 跨表达式图融合~~ **已落地** | **不再是触发项**：`begin_expr/end_expr` 图融合（P1 链 / P2 写穿多输出）已接入 Adam/AdamW step，§3.4 M3 | `docs/todo/03-ir-optimization.md` §5.3、§7；`expr_graph.hpp` |
| S6 自动窗口（跨 kernel 自动融合） | 同 IR-C 的调用方判据；当前"能融的直接写成单个 `dsl::compute`"已覆盖生产路径 | `docs/todo/02-operator-fusion.md` |
| M7 存储多态 | **第三后端出现**（CPU/Vulkan 之外的引擎要落地） | `docs/todo/17-unified-tensor-engine.md` §4.2 + §5 M7 |
| IR-D 第二 emitter | 需要 glsl 之外的代码生成后端，且该后端**必须走 AOT 闭合世界**（表达式结构须被 `scan_exprs` 覆盖） | `docs/todo/03-ir-optimization.md` 阶段 D；`expr_emitter.hpp` |
| ZiPT 恢复 | ① 压缩向量保证**因果**（压缩器只能看当前位置之前）；② 实现质量对齐 RAPT/GPT（`forward_recompute` 梯度检查点 + activation offload + CLI/序列化/测试齐备）；③ 有同参对比数据 | `docs/history/history.md`「ZiPT 移除」条；`legacy/zipt` 分支 |
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
| ctest | **实测 54**（41 目标 + 13 变体/附加；无 Vulkan 树 52）；需 Vulkan 用例退出码 77 = skip | `ctest --test-dir build` |
| L2 分层审计 | `L2-VIOLATIONS: 0`（宿主桥用量只披露） | `pwsh -File bench/doc_inventory.ps1` 第 [4] 节（实测：virtual 52 / host_bridge 39） |
| CPU 字节锚 | `--steps 20` hash = `6f8849f14da23110` | 逐位一致 |
| GPU 字节锚 | dev2 hash = `8ef51b2927253c50` | 逐位一致（GPU 档位按 17 §6 容差规则） |
| scan 产物 | `fused_registry.hpp`（**唯一产物**；`expr_specs.bin` 已随构建期单步删除） | `tools/scan_exprs.cpp` + `tools/fused_generate.hpp` 产物 |
| 初值锚 | `gpu_stability_probe --init-hash` 各模型跨进程一致 | 见 `AGENTS.md` §12 当前锚 |
| 文档对齐 | `pwsh -File bench/doc_align_audit.ps1`：[A][B][D][E][F] 可行动项为 0 | 实测：[A][B][D][E][F] 可行动均为 0（历史归档提及按白名单跳过） |
| 行为回归 | `layer_bench` 同窗交错 + 配对 A/B，无系统性回退（噪声 ±6%） | `build/layer_bench` |

### 8.2 本轮新增门禁（P0 后生效）

1. **CI 必须执行 ctest**：日志需出现当前全部用例（**54**，无 Vulkan 树 52）；77 = skip 不算失败（P0-1）。
2. **开发文档不得出现 ctest 计数的旧写法**：统一写实测 **54**（GPU 树）/ **52**（无 Vulkan 树）——已完成行与 history 归档里的旧数字是当时记录，不算冲突（P0-2 ✅）。
3. **BatchNorm 行为门禁**：构造 batchnorm 规格必须有可观测结果（报错或正确层），不允许静默降级（P0-3）。**✅ 已满足**：真层 + `batchnorm_test` 工厂断言。
4. **危险精度组合**：启动期**告警**（仅警告、不失败、不改退出码）——口径与细则见 [`24-f16-training-defects.md`](24-f16-training-defects.md) §2.3 / §5 F1（P1-3）。
5. **审计覆盖补强**：`bench/doc_align_audit.ps1` 的 [A] **✅ 已扩展到 markdown 链接目标**（E8，2026-10-15）；新增死链会在审计里以 [A] 计数非 0 暴露。

6. **`text_train` 模块不得终止进程（P1-1，2026-10-16 起生效）**：`pwsh -File bench/doc_inventory.ps1` 第 [6] 节
   断言 `cli/text_train_{loop,recovery,checkpoint}.hpp` 与 `src/text_train.cpp` 里**不出现** `std::exit` /
   `NN_EXIT` / `NN_UNWRAP`（这三种写法都会直接终止进程）——三模块的错误一律经 `Result` 上抛，进程唯一出口是
   `main`（CLI 参数解析层是另一个例外，见 `cli/text_train_config.hpp`）。P1-2 的 checkpoint 状态段与
   `--resume` 接线必须保持这条不变量：新增错误检查用 `NN_TRY` / `NN_TRY_MSG` / `NN_FAIL`。

7. **`.nnckpt` 训练状态段与 role 语义（P1-2，v8，2026-10-16 起生效）**：① `role = Checkpoint` 恢复时，
   A/B 类旗标（形状/配方/计数器，清单见 §4.2 P1-2 行）**显式出现即报错**——按旗标名判定、不看值；
   `role = Model` 时 CLI 全权；② 状态段 save→load→save 逐字节一致，老版本（v5/v6/v7）缺段回落
   冷启动语义；③ 数据集身份（sha256 + 网格三元组）不匹配**硬报错且不得提供逃生阀**；④ P2-1 的校验和 /
   尾部完整性标记缺失或截断必须报错。
   **状态（2026-10-16）：① ② ③ ④ 全部落地并有门禁**——①③ 见 `text_train_modules_test` [6] 与
   端到端四例（`docs/history/history.md`「P1-2（第三刀）」）、② 见 `train_state_test` [9]、
   ④ 见 `train_state_test` [5]；优化器侧见 `optimizer_state_test`（重启续跑 = 直跑逐位一致）。
   **已收口（2026-10-16 第四刀）**：`gui.py` 按**文件头的 role 字节**识别 `.nnckpt`（`model_file_role`），
   断点续训模式下 `collect_args` 不发任何禁列键（并显示一行模式提示），`.nnmodel` warm start 仍 CLI 全权；
   `bench/gui_cli_audit.py` 新增 [6] 节（禁列集合 ↔ `kForbiddenResumeFlags` 逐项一致 + 两种 role 下的
   收集行为 + C 类不被误砍）。

8. **三值模型可续训（P1-7，2026-10-16 起生效）**：训练侧路径（`text_train` / `mnist_train` 的
   `--resume`）载入参数一律走 `WeightLoadIntent::Trainable`；三值 `.nnmodel` 的打包载荷必须被
   **反量化成可训练 latent**（不得落进 `packed_only_` 部署态），且首次 forward 的 `wq`/`τ` 与文件里的
   打包权重**逐位一致**（层用文件 τ 播种量化缓存）。**门禁**：`model_roles_test` [6]（打包字 + τ 位级比对、
   首次 forward 逐位一致、`absmean(latent) == τ`、8 步 Adam 后 latent 移动、dense 两种意图逐位相同）、
   `ternary_gemm_test` [8]（反量化 helper 的三条不变量）、`text_train_modules_test`（三值 GPT 走通
   `resume_text_train` + 8 步 Adam，首次 forward 与文件逐位一致）。推理侧默认意图 `Inference`
   不变（部署只读态 = 0.25 B/权重）。

9. **多轮对话上下文不丢（P0-5 尾项，2026-10-16 起生效）**：多轮 prompt 的构建只有一条路
   （`Tokenizer::chat_prompt`），且必须满足两条可断言的不变量：① **第 N 轮的 prompt 以第 N-1 轮
   的 prompt 为逐 token 前缀**（"上下文不丢"的可验证形式）；② 超 `seq_len` 时按**整轮（一问一答）**
   丢弃 —— system 块与末轮恒保留、丢弃后仍以 user 起头（标记结构不破），且丢弃轮数可观测。
   `--turn` 的形状不合法（role 写错 / 不交替 / 末条不是 user / 与 `--prompt`·`--interactive` 并用）
   **解析期报错**，不得静默降级；GUI/脚本侧一律经 `--turn`（整段历史随采样参数一起传回）+ `--reply-out`
   （回复落盘），不得靠解析 CLI 日志取回复。**门禁**：`chat_prompt_test`（库级 5 节 + 真实 `text_infer`
   子进程端到端 + 7 条负例；app 缺失时 77 = skip）、`bench/gui_cli_audit.py` 第 [7] 节（Python 侧
   参数面：前缀不变式 / 回复闭环 / 清空归零 / `_on_send` 真的传历史）。

10. **MSVC Debug 冒烟 + 权重变化（P1-5，2026-10-18 起生效）**：CI 的 `msvc-debug` job 必须
    用 **cl.exe + Debug** 编译全部目标（`NN_ENABLE_GPU=OFF` 纯 CPU，`NN_ENABLE_TESTS=ON`）
    并跑 ctest（除下述已知 C11 排除项外全跑）；`optimizer_test` 的 [4] 段必须断言
    **8 步 Adam 后全参数 L1 位移 > 1e-3**
    （loss 有限且下降），并用"同一快照自比 → 判为无移动"的负例证明检测器可识别冻结。
    该断言**不得**放在 `f16_writeback_probe` 之类无 Vulkan 就 77 的用例里——它必须在
    MSVC Debug CPU-only 树上真实执行。**C1128 已在 102 实测触发**（`text_train.cpp`）⇒
    `/bigobj` 已加进 MSVC 目录级与接口选项；**栈溢出也已实测**（9 个用例 `0xC00000FD`，
    MSVC 默认 1 MB 栈）⇒ `/STACK:8388608` 已加。MSVC 102（VS 2026 / cl 19.51）
    Debug CPU-only 修复后 **51/51 通过（180 s）**。
    **已知排除**：`rapt_test_f16` 是既有 C11（RAPT f16 NaN，用户已排查，怀疑设备/后端
    自身问题；Release 通过只是概率），CI 冒烟暂以 `-E "^rapt_test_f16$"` 排除；大版本发布
    清单（[`25`](25-release-requirements.md) §1）仍要求 Debug **完整 ctest 全绿**，C11
    修复后必须删掉该排除项。

### 8.3 验收纪律

- **计划内字节变化**：按 17 §6 例外流程**先立新锚**，新锚两次独立运行之间仍须逐字节，并在该期验收中显式记录。
- **既有故障不混入验收**：C1（GPU `stable=f16` loss 冻结）与 C6（`gpt_test` flaky）在修复前按"既有状态如实记录"处理，不得当成新回归。
- **裁定不可静默偏离**：任何与本文 §5/§7 不一致的做法，先在本文改裁定并说明理由，再实施。

### 8.4 本文档的维护约定

1. **状态只写现在式**：做完一项就把该项从 §4 移到 `docs/history/history.md`，并在 §4 留一行"已完成"指针。
2. **数字必须带复现命令**：任何计数（行数、测试数、锚值）都要能在命令里复现；不可复现的只写"约"或删。
3. **行号会腐烂**：行号只作入口线索；引用时优先给文件名 + 符号名，行号在改动后由 `bench/doc_align_audit.ps1` 兜底
   不覆盖的，由自动化测试兜底（新增断言测试优于文档数字）。
4. **裁定要留反例**：凡"不做"必须写清触发条件或反例（如 H4 的 RAPT 残差相加），否则后人无法判断能否重新立项。
5. **与 13 的关系已终结**：13 号文档已删除（内容全在本文与 `docs/history/history.md`），新增条目只进本文。
6. **计划不落本文**（2026-10-15 起）：本文只写**裁定 / 验收口径 / 现状实测**；"先做哪个、第一步动作、
   并行与工时安排"一律单独成文（当前 = [`27-execution-plan.md`](27-execution-plan.md)），本文只留指针（§1.3）。
   判断口径：**可被"做/不做"二值化**的进本文，**只描述顺序与动作**的进 27 号。

### 8.5 大版本发布要求

**大版本**升级（`1.x.y → 1.(x+1).0`，次版本号 +1、修订号归零）在常规门禁之上还要走一遍
[`25-release-requirements.md`](25-release-requirements.md) 的必做清单：① 端到端模型训练测试
（这类测试**不进默认 ctest**，太费时）；② 发布文档；③ 完整 ctest + **Debug** 构建；
④ 多编译器 × 标准库（≥4 组）全新构建 + ctest。小 / 修订号升级不强制。

## 9. 仍然生效的工程教训（原 13 §9 承接）

13 号清单的"仍然生效的工程教训"中，以下 2 条在 13 删除后无处落地，按原义承接。
（其余 3 条已由别处覆盖：`dsl::compute` 无形状重载（已完成）见 `AGENTS.md` §7；`max_abs_diff` 计数已入本文 §5 裁定表；
`gpt_test` 偶发失败已列为本文 P1-4。）

### 9.1 eager → DSL 迁移前必须确认目标表达式结构已被 `scan_exprs` 覆盖（原 13 §9-2）

否则 GPU 闭合世界运行期**硬报错**。低成本做法 = 把新表达式写成与已扫描表达式**同构**
（如 RAPT 除法的 ε 位置对齐 `forward`），即可零成本复用键。
（对应铁律 #7：AOT 闭合世界，未命中硬报错，无 eager、无运行时编译。）

### 9.2 改测试公共头后，聚合器的 `#undef CHECK` 必须同步删除（原 13 §9-3）

`pragma once` 之下 `#undef` 会让后续子文件失去宏定义。测试用「`#define main test_xxx` + `#include 子测试.cpp`」
聚合编译，聚合器里任何 `#undef` 都会影响其后被 include 的所有子测试。
