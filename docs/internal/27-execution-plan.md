# 27 — 执行计划（优先级象限）：未完成项的执行排序

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。
> **状态**：现行。

> **本文与 [`18-roadmap.md`](18-roadmap.md) 的分工（唯一规则）**：18 号裁定"**要不要做 / 做到什么算完成**"
> （做 / 条件触发 / 不做 + 验收口径 + 现状实测），本文只排"**先做哪个**"（象限 + 第一步动作 + 串行约束）。
> **执行计划不写进 18 号**——18 号只留裁定、状态与指向本文的指针（18 §1/§4 的指针行、§8.4 纪律 6）；
> 本文**不产生新裁定**：与 18 §5/§7 冲突时先在 18 号改裁定，再回来调本文的排序。
> 触发式项（第三后端 / 第二 IR emitter / S6 / CUDA / 分布式 …）**不进本文**，触发条件成立并由 18 §6
> 立项后再进表。来源 = 18 号 §3 缺口表 / §4 排期 / §6 触发清单、[`00-code-review-remediation.md`](00-code-review-remediation.md)
> 的 RV 台账、GitHub issue。
>
> **2026-10-23 用户裁定覆盖默认排序：当前主线 = RL 训练原语。** f16（P2-14/P2-15）与训练性能
> （P4-1/P4-2/P4-3）暂缓；本文其余象限是 RL 收口后的 backlog，不得插队。

## 1. 判据

- **重要** = 影响三者之一：**门禁**（CI 结论是否可信）、**正确性**（用户会不会静默拿到错误结果）、
  **已发布负载可用性**（训练/推理路径能不能用）。
- **简单** = ≤1 天级工作量；不改跨模块结构（IR / 引擎接口 / 存储 / 序列化格式）；不需要 GPU 长跑或远程机器。
- 归类描述**当前**判断。一项若天然分"定位"与"修复"两步，两步可落不同象限（例：C10 归因便宜、修复未定）。
- 象限只回答顺序，**不改变验收口径**：每项的"做到什么算完成"一律引 18 号对应行。

## 2. RL 主线（2026-10-23 已完成）

RL-0 → RL-1a/b/c/d → RL-2a/b 全链已落地，公共契约、组件与编排接口按
[`31-rl-training-primitives.md`](31-rl-training-primitives.md) 冻结；验收与落点见该文 §11，
裁定状态见 [`18-roadmap.md`](18-roadmap.md) §4.6。调用指南在
[`../usage/06-reinforcement-learning.md`](../usage/06-reinforcement-learning.md)。

按本文维护约定，完成项不再留在执行表。后续 RL 扩展（连续/多离散分布、领域 replay、
消费者自定义 checkpoint 格式）由真实调用方触发；f16 与性能 backlog 仍按裁定暂缓。

## 3. 象限（一）重要且简单——建议按序做

| 项 | 坐标 | 为什么重要 | 第一步动作 | 验收归属 |
|---|---|---|---|---|
| **生命周期 v3 · P0 前置修复** | [`28`](28-lifecycle-v3.md) §5 P0 | 三个已确认的生命周期缺口（offload 状态机、`ExprSegment` 裸引用、文档假承诺）+ 文档矛盾；≤1 天、不改跨模块结构 | 先修 offload fail-fast 与 `ExprSegment` 强引用，再补 ASan 负例 | 28 §5 P0 / §6 |

> **上一批（2026-10-15）已清空**——五条全部完成（`gpt_test` flaky、C10 归因、P0-5 采样+
> CLI 半边、P2-6 绘制节流半边、RV-14 审计）。按 §8 约定完成后从本表删除，状态与证据见
> 18 号对应行（§3.1 C6/C10、§3.3 X1/X5、§4.1 P0-5、§4.2 P1-4、§4.3 P2-6）与
> `docs/history/history.md` 的同名条目。**另：P2-6 的「声明式字段表」半边也已完成（2026-10-15，同上）** ——
> 6 个 Tab 全部迁到 `fields()` 表；等价性用 22 个场景逐字段比对（17 一致 / 4 有意收紧 / 1 命令行层等价），
> 并在 `bench/gui_cli_audit.py` 新增第 [5] 节做闭包门禁。
> **当前不从本象限取新主线**；完成 §2 的 RL 链后再回到这里。

## 4. 象限（二）不重要且简单——顺手做

| 项 | 坐标 | 为什么排这里 | 前置 / 成本 |
|---|---|---|---|
| **P2-2 Adafactor 类优化器** | 18 §4.3 P2-2 / §3.4 M1（= P4-3①） | 显存收益要在"训练超出 Adam m/v 预算的模型"时才兑现，当前无该场景 | RV-04 已把成本压到"`kOptimizerTable` 1 行 + `create_optimizer` 1 分支"；小 |
| **索引/口径卫生**（`release-notes` 索引、各表计数） | 18 §8.2 门禁 2 | 一致性卫生活；`python scripts/doc_audit.py` 已能抓大部分 | 小；随其他改动顺手做 |

> **2026-10-16：GUI 多轮对话闭环（P0-5 尾项）已完成**——按 §8 约定从本表删除。
> 落地形态 = `Tokenizer::chat_prompt`（唯一 prompt 构建）+ `--turn`/`--reply-out`/`--system`
> （无状态多轮协议）+ `--interactive` 进程内累积历史 + GUI 的「💬 发送（多轮）」面板
> （`ChatSession`）；门禁 `chat_prompt_test` + `bench/gui_cli_audit.py` [7]。验收口径与实测见
> 18 §4.1 P0-5 行 / §8.2 门禁 9，过程见 `docs/history/history.md`「P0-5 尾项」条。

## 5. 象限（三）重要但复杂——RL 收口后的 backlog

| 项 | 坐标 | 为什么重要 | 前置 / 成本 |
|---|---|---|---|
| **P1-8 生命周期 v3（core）** ✅ **已完成（2026-10-20）** | [`28`](28-lifecycle-v3.md) §5 P1；18 §4.2 P1-8 | 删除 `EngineId`/`stamp_`/`adopt`/`bound`/`NN_BIND_DEBUG`，统一设备域（D1/D2/D7）；破坏性 API，分支发布 `v1.10.0-rc0` | 先落 28 号设计；P1 完成后 P2/P3 才排；中–大 |
| **Session 层（P2a/P2b + 单步协议）** ✅ **P2a 2026-10-20、P2b 2026-10-21、单步边界 2026-10-22** | [`29`](29-session-architecture.md)、[`30`](30-training-session-step.md)；18 §4.2 P1-8 | `TrainingSession` 拥有 Model+Optimizer，结构性修复 D3 的 UAF。30 号起 Session 接收 micro-batch、内置累积并完成 update；global batch/lr/epoch/运行态 `TrainState` 归 `TextTrainer` | 已完成；后续只随训练入口演化 |
| **P3 GPU backend 确定性生命周期** ✅ **已完成（2026-10-20）** | [`28`](28-lifecycle-v3.md) §5 P3 | `shared()` 改 weak 缓存 + 删 `instance()`（D5）；确定性回收，16 处迁移 | 需 GPU 验证（850M/40HX）；中 |
| **P1-2（+ P2-1 合并）** —— P1-1 ✅ 已完成（2026-10-16） | 18 §4.2 P1-2、§4.3 P2-1；C4/C5，门禁 §8.2 第 7 条 | P1-2 是**真实功能缺陷**：TDR 重启 / `--resume` 后 Adam 动量归零，续训轨迹不连续。**2026-10-16 裁定扩写**为「`.nnckpt` 训练状态段 + role 语义」：状态段 = 计数器 + 优化器状态 + 训练配方 + 数据集身份（role 门控，**v7 → v8**，与 P2-1 同批）；`Checkpoint` **禁止 A/B 类 CLI 输入**（显式即报错），`Model` **CLI 全权**（放开 `.nnmodel` 恢复以支持 SFT） | 前置 **P1-1 已解锁**（`text_train` 已拆为 `cli/text_train_{config,loop,recovery,checkpoint}.hpp` + 396 行入口，恢复路径不再膨胀）；P1-2 与 P2-1 同属版本段改动，**合批做省一次 `MODEL_VERSION` 升版**。细则见 [`23`](23-library-design-cleanup-and-model-file-roles.md) §2.1.1；中–大 |
| **P1-7 三值模型续训入口（打包 → latent 反量化）** ✅ **已完成（2026-10-16）** | 18 §4.2 P1-7 / §3.4 M6 | 原现象：三值模型的 `.nnmodel` 载入后是 `packed_only_` 部署态 ⇒ `backward()` 明确报错，**三值 SFT / QAT 续训没有入口**。**已修**：载入意图 `WeightLoadIntent::Trainable` 按文件 τ 反量化成 f32 latent（量化器不动点 `L = trit·τ·K/n`）+ 播种量化缓存 ⇒ 首次 forward 与推理文件逐位一致、可续训；训练侧 CLI 一律传该意图，原"三值 `.nnmodel` 必须被拒"门禁删除 | ✅ 已完成：验收 = 反量化后 wq/τ 与文件内打包权重逐位一致（`model_roles_test` [6] 位级比对）+ 8 步 Adam 参数移动 + dense 路径逐位不变；端到端（`.nnmodel` warm start → 训练 → `.nnmodel`/`.nnckpt` → 续训）见 `docs/history/history.md`「P1-7」 |
| **P2-14 全流程真 f16（恢复计划）** ⏸ **暂缓（2026-10-23：RL 优先）** | 18 §4.3 P2-14 / [`26`](26-f16-speedup-plan.md) §8 | F5 后四字段全 f16 已可训练，但还不是真 f16：CE 标量仍是 F32 补丁、matmul/归约仍有 f32 累加、native16 未全覆盖。目标是 **B-Mem：无 f32 持久态**；capable 设备另追 B-Arith 零转换。数值鲁棒性技术栈（SR / GradientStabilizer / ECO / attention δ 修复）见 26 §8.B.7 | RL 收口后从 R0 口径统一 + R1 `--f16-all` + §8.B.7.0.1 的 P0 探针开始；中–大 |
| **P2-15 标准 `--f16` 训练（Plan A，参考 PyTorch AMP）** ⏸ **暂缓（2026-10-23：RL 优先）** | 18 §4.3 P2-15 / [`26`](26-f16-speedup-plan.md) §8.A | 把现有 `--f16` 做成可解释、可对拍的标准混合精度：`profile_f16()` 默认、`profile_master_weights()` 可选，补 GradScaler 语义的 loss scaling 与 PyTorch AMP 对照。**不要求零转换** | RL 收口后从 A1.1 动态 loss scaling + PyTorch AMP 对照脚本开始；中 |
| **P4-1 性能利用率** ⏸ **暂缓（2026-10-23：RL 优先）** | 18 §4.5 P4-1 / §3.1 C9 尾注 | 唯一还没归因的固定成本：`墙钟 − 设备 kernel ≈ 4.2%`（f32/f16 同幅） | RL 收口后用 `NN_PROFILE` 归因 host 录制 / driver 提交 / fence 等待；中 |
| **P4-2 算子性能** ⏸ **暂缓（2026-10-23：RL 优先）** | 18 §4.5 P4-2 | C10 派生的 (a)+(b) 已完成；剩余重型 kernel = fold / matmul / reduce / `matmul_q` 调参 | RL 收口后用 `layer_bench --op <name> --hash` 配对 A/B；中 |
| **P4-3 显存 L3（04 号立项）** ⏸ **暂缓（2026-10-23：RL 优先）** | 18 §4.5 P4-3 / [`04`](04-memory-optimization.md) L3 | 下一档显存杠杆 = 注意力/softmax 子图进一步融合（消 Q/K/V 物化）+ 优化器状态低精度 | RL 收口后再立项 04 号 L3；中–大 |
| **P2-7 backend 超大 header 拆分** | 18 §4.3 P2-7 / §3.2 E6 | header-only 项目里增量编译是主要开发税 | **前置 = P1-5 ✅（2026-10-18 已解锁）**；中–大 |

> **2026-10-23 当前主动推进项 = RL 主线；本表是收口后的候选。** P2-14/P2-15/P4 系列已明确暂缓；
> P2-7 backend 拆分为独立工程项，可在 RL 完成后排；已完成项的状态仍以 18 号与 history 为准。

## 6. 象限（四）不重要但复杂——不做 / 等触发

| 项 | 坐标 | 现状与触发条件 |
|---|---|---|
| P2-4 流式 Dataset / DataLoader 后半 | 18 §4.3 P2-4 / §3.3 X2 | 触发：语料大到内存放不下（当前规模不需要；格式前半已落地） |
| P2-5 Python / C 绑定 | 18 §4.3 P2-5 / §3.3 X3 | 触发：出现真实外部调用方（18 §6"条件触发"） |
| E9 归约 epilogue 融合（`reduce(...)*k+c`） | 18 §3.2 E9 | 已有降级路径（拆两步 dispatch）；触发：出现真实热路径 |
| M7 存储多态 / IR-D 第二 emitter / S6 自动窗口 | 18 §6 / §3.4 M4 | 触发：第三后端 / 第二代码生成后端 / 自动窗口的真实调用方 |
| LRLA / ZiPT 恢复 | 18 §6 / §3.4 M2、M5 | 触发：同参对比数据（ZiPT 另需"压缩器因果 + 质量对齐"） |
| CUDA / 分布式 / HTTP 服务化 / ONNX | 18 §7 非目标、§6 触发清单 | 明确不做；触发条件见 §6 |
| 21 号 P3（激活 int8 + 整数点积 + DSL `round`/`clamp`）、23 号 §2.5 第 6 步（int8 全链路） | [`21`](21-quantized-weights.md)、[`23`](23-library-design-cleanup-and-model-file-roles.md) | 能力扩张、无调用方；22 号的两项（`--model bitnet` 预设、注意力增量路径三值）已裁定不做 |

## 7. 建议执行顺序

1. **RL 当前主线**：公共基础 → RL-0 → RL-1a/b/c/d → RL-2a/b；每一步的验收在 [`31`](31-rl-training-primitives.md) §6。
2. **RL 收口后**才回到 backlog：P2-7 → P2-14/P2-15 → P4 系列；顺序可再按当时调用方重排。
3. f16 / 性能项不在 RL 期间启动，也不得被当作 RL 的前置或验收条件。
4. 已完成的生命周期 v3 / Session 层只保留维护，不重新插队。
5. 串行硬依赖：**公共基础 → RL-0 → RL-1a/b → RL-1c → RL-1d → RL-2a/b**。

## 8. 维护约定

1. **做完一项**：从本文删除该行，并把状态写回 18 号对应行 / `docs/history/history.md`（按 18 §8.4 纪律 1）。
2. **象限变化**（比如发现某项成本远超预期）：改本文归类并注明理由；若牵涉"要不要做"，先改 18 号裁定。
3. **触发式项不预先列表**：18 §6 的触发条件成立、并经 18 号立项后，才从 §6 移入本文相应象限。
4. 本文**不带长期状态**：完成后不留"已完成"行（去 18 号看状态、去 history 看过程）。
