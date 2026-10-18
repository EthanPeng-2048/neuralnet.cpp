# 24 — f16 训练缺陷台账：全 f16 NaN / loss 冻结 / 降速

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。

> 状态（2026-10-18 更新）：**F1–F4 已完成；F5 的「可训练性」已完成**（optimizer state scaling，
> 2026-10-14）。后续拆成两套计划：**F7 = 标准 `--f16`（Plan A，参考 PyTorch AMP）**；
> **F6 = 全流程真 f16（Plan B）**，二者均于 2026-10-18 立项/恢复（细则见
> [`26-f16-speedup-plan.md`](26-f16-speedup-plan.md) §8）。本文中「全 f16 不可训练 / 必崩」的
> 表述均指 **F5 之前**；F5 之后应读作「可训练，但尚未达到真 f16（零转换 / native16 全覆盖）」。
> A/B1 的 `stable=f16` 根因订正与修复仍有效：**A 全 f16 NaN 的「`stable=f16` 直接生产者」已修 /
> B1 GPU `stable=f16` loss 冻结已修（根因 = CE 的 (1,1) loss 归约按 `stable=f16` 落盘溢出，
> [`14`](14-f16-stable-gpu-loss-frozen.md) 原「回读链」定位并非真因）。**
> 落地改动：`include/neuralnet.cpp/compute_loss.hpp`（CE 稀疏/稠密两条链的 (1,1) 归约输出恒
> F32）+ `src/f16_precision_test.cpp` 回归用例 + `src/cli_precision_warning_test.cpp` 门禁。
> 来源：核对用户提供的《全 f16 训练 NaN 问题定位报告》，并按追加要求定位「f16 训练冻结」
> 与「f16 训练速度降低」两项，合并立项。
> 关联：`docs/development/05-mixed-precision.md` §12.4/§12.5/§12.11；`precision.hpp`（`profile_all_f16`
> 数值注记）；[`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)；
> [`18-roadmap.md`](18-roadmap.md) §3.1 的 C1/C2 与 §4.2 的 P1-3；
> [`25-release-requirements.md`](25-release-requirements.md)（F2 的端到端全 f16 训练归口大版本发布清单）。

## 0. 结论速览

「全 f16」= `profile_all_f16()` = **四字段全 F16**（`--f16 --precision-stable f16 --precision-optimizer f16`），
区别于健康推荐路径 `--f16` = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}。
本档把用户报告的「全 f16 不可训练」（F5 前的历史现象）与追加定位的「冻结」「降速」并列立项。

| 缺陷 | 现象 | 定位结论 | 坐标 | 状态 |
|---|---|---|---|---|
| **A 全 f16 NaN** | ① epoch2 起 loss `-nan`；② `stable=f16` CE 链 ~200 步 NaN | **NaN 直接生产者 = `stable=F16`**（CE 归约的 (1,1) 输出溢出 f16 上限 65504 → ±inf；见 §2.1 订正）；**前导放大器 = `optimizer=F16`**（Adam 二阶矩下溢 → 更新爆炸 → logits 出 f16 范围） | `compute_loss.hpp`（`softmax_cols_` / `fused_forward_sparse_`）、`compute_optimizer.hpp`（`Adam::adam_update_`） | **`stable=F16` 部分已修**（2026-10-13：(1,1) 归约输出恒 F32）；`optimizer=F16` 放大器 **F5 已修可训练性**；长跑质量/覆盖见 F6 |
| **B1 冻结（GPU）** | GPU + `stable=f16` 时 loss 打印恒 `4.0137`，权重照常更新 | **订正**：冻结 = **CE 的 (1,1) loss 归约按 `stable=f16` 落盘溢出**（大 token 数下 Σ≈ -log(vocab)·N 超 65504）；`4.0137 ≈ 65504/num_valid` = 该溢出在部分硬件上饱和为 f16 上限的表现。**并非**回读链问题（原结论） | `compute_loss.hpp`（`softmax_cols_` / `fused_forward_sparse_` 的 (1,1) reduce 输出精度） | **已修**（2026-10-13；[`14`](14-f16-stable-gpu-loss-frozen.md) 已同步订正根因） |
| **B2 冻结（数值）** | 四字段全 f16 时 loss 恒定（更新被 f16 舍入吃光） | 数值性冻结：`upd_prec=F16` 下 `p += delta` 的 `|delta|` 小于参数所在 f16 网格步长 → 原地写回无变化 | `compute_optimizer.hpp` `Adam::adam_update_` K3（`delta` + `compute_into(p + delta)`） | **F5 已修可训练性**（状态缩放后 m/v 回到 f16 正常区）；长跑/质量见 F6 |
| **C 降速** | `--f16`（健康路径）比 f32 **慢 11–17%**，与「f16 应省带宽/更快」的预期相反 | **已兑现（F4）**：运行期精度分派三元税为主，S1s/S1t + 网格粒度变体修掉；边界 cast 的 f32 临时量与 CPU 无 f16 ISA 两个来源也已分别处理/评估 | `backend/compute_vk_backend_ops.inc`、`shaders/matmul_tiled.comp`、`compute_gpu_engine.hpp`；`docs/development/05-mixed-precision.md` §12.11 | ✅ **2026-10-14 已完成**；真 f16 的零转换/native16 覆盖见 [`26`](26-f16-speedup-plan.md) §8 R4 |

## 1. 报告核对（逐条）

对用户《全 f16 训练 NaN 问题定位报告》逐条比对代码与既有文档。

| 报告条目 | 核对 | 依据 |
|---|---|---|
| 范围：全 f16 = 四字段全 F16；健康路径 `--f16` 只设 param/compute | ✅ 证实 | `precision.hpp` `profile_f16()` / `profile_all_f16()`；`src/text_train.cpp` `--f16` 与 `--precision-*` 解析 |
| 根因①代码位置 `softmax_cols_`（报告写 ~L178–215） | ✅ 符号正确（行号略偏，实为 L173 起） | `compute_loss.hpp` `CrossEntropyLoss::softmax_cols_` |
| 根因①：`col_max` / `col_sum` / `shifted` / `softmax` / `log_softmax` 全部以 `p_.stable` 落盘 | ✅ 证实（`p_.stable` 逐处可见） | `compute_loss.hpp` L179/186/193/197/203/207 |
| 根因①：loss 归约 `Σ target·log_softmax` 用 `compute_reduce(..., p_.stable)` | ✅ 证实 | `compute_loss.hpp` `forward`（稠密，L248–255）与 `fused_forward_sparse_`（稀疏，`loss_sum` L413、`grad` L446、`lse` L392） |
| 根因①推论「f16 舍入可把接近 1 的 `col_sum` 落到 <1 → `log()` 出负值」 | ⚠️ **更正**：不成立。`col_sum ≥ 1`（max 元素 shifted=0 → exp=1），且 1.0 在 f16 可精确表示 → 输出 RHE 舍入后仍 ≥ 1.0，**`log(col_sum)` 不可能为负**。真正风险是尾数不足带来的归约精度损失与 `loss_sum` 输出溢出到 ±inf | `compute_loss.hpp` `softmax_cols_` 的 `col_sum` 不变式注释；`precision.hpp` f16 相对精度 2⁻¹⁰ |
| 根因①：大 batch·seq（16K token）时 loss 归约部分和超过 f16 上限 65504 → Inf → NaN | ✅ 机制成立（归约**累加** = max(P,f32)，但**输出**按 P 落 f16；`Σ log_softmax ≈ -9 × 16384 ≈ -1.5e5` → f16 输出溢出）。具体「-Inf/batch 与 Inf+(-Inf)」的衔接属推断，未逐算子验证 | `compute_loss.hpp`；`docs/development/05-mixed-precision.md` §7.2（输出存储全在 P） |
| 根因②代码位置 `Adam::adam_update_` K2（v 用 `p_.optimizer`）与 K3（`sqrt(v)+eps`） | ✅ 证实 | `compute_optimizer.hpp` `adam_update_`（K2 L330–333、K3 L345–353） |
| 根因②：`v ≈ g²` ~ 1e-10 低于 f16 次正规最小 6e-8 → 下溢 0；`eps=1e-8 → 0` | ✅ 证实（f16 最小正规数 2⁻¹⁴≈6.1e-5、次正规最小 2⁻²⁴≈5.96e-8；默认 `eps=1e-8`） | `precision.hpp` §5.2 数值事实；`compute_optimizer.hpp` `Adam` 默认 `eps=1e-8`（L396） |
| 根因②实测「loss 7.9→3.6e4」 | ✅ 与既有实测一致（另一次配制实测 1.8e7，同为爆炸形态） | `precision.hpp` L539–541 注记；[`14`](14-f16-stable-gpu-loss-frozen.md) §2.1 case c |
| 根因③「四字段全 f16 → loss 恒定」 | ✅ 机制证实；**常数口径已订正**：原 `precision.hpp` 注记的「32.1」无 bench 背书、且已在 2026-10-13 随修复移除（该注记仍停留在 F5 前的「不可训练」口径，R0 待同步）。GPU [`14`](14-f16-stable-gpu-loss-frozen.md) 的 `4.0137` **另有根因**：= CE 的 (1,1) loss 归约按 `stable=f16` 落盘**饱和到 f16 上限**（`≈ 65504/num_valid`，见 §3.1）——与「更新被舍入吃光」的数值性冻结是两回事，报告把二者并陈需分开看 | `precision.hpp` `profile_f16()` 注；[`14`](14-f16-stable-gpu-loss-frozen.md) §2 与 §3.1 |
| 报告「14 号报告 Open / GPU 特有」 | ✅ 证实 | [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)（状态 Open） |
| 报告 §三.2「`debug_profile_matrix` 矩阵**没有**四槽全 f16 这一行」 | ✅ 证实（cases = f32 / param / compute / stable / opt / f16(CLI)，无 all=f16） | `src/f16_precision_test.cpp` `debug_profile_matrix`（cases L639–646） |
| 报告 §三.2「预期 `[dbg stable=f16]` 与 `[dbg opt=f16]` 出现非有限/爆炸」 | ⚠️ **部分更正**（本档已实测复现，见 §2.4）：`opt=f16` ✅ 爆炸（`3.4772 → 3.50e6 → … → nan`）；`stable=f16` 在该 CPU 小 GPT（vocab 32、20 步）下**健康**（`3.4785 → 3.1797`）。~200 步 NaN 需大词表 + 更多步，报告对小矩阵的预期与实测矛盾 | [`14`](14-f16-stable-gpu-loss-frozen.md) §2.3；§2.4 复现输出 |
| 报告 §三.3 `train_tiny_gpt(...).scan()` 逐阶段定位首个非有限值 | ✅ 证实（`train_tiny_gpt` + `scan` 存在） | `src/f16_precision_test.cpp` `train_tiny_gpt` |
| 报告 §四 `--f16` = 安全配方；全 f16 仅供显式实验 | ✅ 证实 | `precision.hpp` L544–545；`docs/development/05-mixed-precision.md` §9.4 |
| 报告 §四「CLI 应保持现状，全 f16 仅作显式实验并在启动期给警告」 | ✅ **已落地（2026-10-13，仅警告不拒绝）**：`nn::cli::dangerous_precision_warning`（`include/neuralnet.cpp/cli/cli_precision_warning.hpp`）在 `text_train`/`mnist_train` 解析后、进训练循环前打印一次到 stderr；触发 = `optimizer=f16`（四字段全 f16 被其覆盖）；退出码与训练行为不变（详见 §2.3） | `src/text_train.cpp`、`src/mnist_train.cpp`；门禁 `src/cli_precision_warning_test.cpp`；[`18-roadmap.md`](18-roadmap.md) P1-3 |
| 报告 §二「已排除项」（次正规移位 UB / fold 恒等元 / f16 输出缓冲 / CPU 解释器直读）均已修复 | ✅ 证实（均已修，见历史归档） | `docs/development/05-mixed-precision.md` §12.12；`docs/history/history.md` |

## 2. 缺陷 A：全 f16 NaN（已定位）

### 2.1 `stable=F16` 是 NaN 的直接生产者

`CrossEntropyLoss` 的整条数值敏感链（softmax / log_softmax / CE loss）以 `p_.stable` 落盘
（稠密 `softmax_cols_` 与稀疏 `fused_forward_sparse_` 一致）。`stable=F16` 时：

1. **归约输出按 f16 落盘**：`col_sum`、`log_softmax`、`loss_sum`（`(1,1)`）均为 f16 存储。
   归约**累加**是 `max(P,f32)`（§7.2），但**输出存储 = P** → 大 batch·seq 的 loss 和
   （≈ `-9 × 16384`）超出 f16 上限 65504 → 输出 ±inf。
2. **尾数不足**：f16 10 位尾数在 vocab 级 log-sum-exp / log_softmax 上累积相对误差，
   轨迹偏差放大（不必然 NaN，但与 §12.5「~200 步 NaN」现象一致）。

> 更正见 §1：报告「`col_sum` 落到 <1 → `log()` 出负值」不成立；真正的溢出点在
> `loss_sum` 输出与 logits 出 f16 范围，而非 `log(col_sum)`。

> **已修（2026-10-13）**：`compute_loss.hpp` 的 CE 稀疏（`fused_forward_sparse_`）与稠密
> （`forward` 步骤 4+5）两条链，其 **(1,1) loss 归约输出精度恒用 F32**（原先 = `p_.stable`）。
> 该标量是跨 `batch×seq` 的 log_prob 求和（≈ `-log(vocab)·N`），是按 f16 落盘必然溢出的
> 聚合量；归约内部本就恒 f32 累加，输出改 F32 即得正确值，且调用方随后本就 `cast` 到 F32
> 回读。**真实触发范围 = 大 token 数**（本机实测：`seq128×b8`=1024 token 健康；
> `seq256×b32`=8192 token 时 `stable=f16` 打印 `inf`，`stable=f32` 正常 9.0→8.7）。
> 回归用例见 §2.4 与 §5 F2/F3。

### 2.2 `optimizer=F16` 是前导放大器

`Adam::adam_update_` 的 K2 以 `p_.optimizer` 存 `v = β2·v + (1-β2)·g²`。小梯度 `g~1e-5` 时
`g²~1e-10` 低于 f16 次正规最小 5.96e-8 → `v` 舍入为 0；K3 `delta = -lr·m̂/(sqrt(v̂)+eps)` 分母
`sqrt(0)+1e-8` 在 f16 中进一步坍缩 → **单步更新爆炸** → logits 超出 f16 范围 → forward 侧 cast 出
inf → CE 链 `inf-inf = NaN`。这与 `precision.hpp` L539–541、§12.5 的实测（loss 7.9→3.6e4 / 1.8e7）一致；
本档 CPU 小 GPT 实测 `opt=f16` 轨迹 `3.4772 → 3.50e6 → 5.06e6 → … → nan`（同族形态）。

### 2.3 CLI 语义与危险组合口径（本轮裁定：**仅警告，不拒绝**）

`src/text_train.cpp` / `src/mnist_train.cpp` 的 `--precision-param/-compute/-stable/-optimizer`
只赋值、**不校验**——用户可无告警地组合出 `optimizer=f16` / 四字段全 f16 这类高风险配置
（F5 前是必崩；F5 后已可训练，但仍有 `|g|` 溢出与长跑质量边界）
（[`18-roadmap.md`](18-roadmap.md) P1-3 已立项）。

**裁定（本轮，取代原「fail-fast / 显式确认」口径）**：危险精度组合**不 fail-fast、不加确认开关**，
只在 CLI 启动期向 stderr 打印一次明确警告后**照常运行**（保留用户显式实验的自由）：

- **触发集合** = `optimizer=F16`、四字段全 F16（`profile_all_f16()`）。全 f16 必含 `optimizer=F16`，
  命中即合并为一条更完整的警告；`stable=F16` 单独出现**暂不纳入**（小模型健康、大词表才 ~200 步
  NaN，是否单列见 §5 F1 备注）。
- **警告内容**：命中的字段与取值 + 一句话风险（优化器状态下溢爆炸 / CE 链 NaN / 数值性冻结）+
  指回本档 §2；
- **行为**：仅 stderr 文本——**不改变退出码、不阻断训练、无静默降级**；`--f16` 与合法
  `--precision-*` 组合保持静默。
- **落点**：CLI 参数解析后、进入训练循环前统一判定一次（`src/text_train.cpp`、`src/mnist_train.cpp`）。

### 2.4 复现（CPU-only 即可复现 NaN 家族）

```bash
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON -DNN_ENABLE_PROBES=ON
cmake --build build --target f16_precision_test
NN_F16_DEBUG=1 build/test/f16_precision_test   # 单字段矩阵：opt=f16 爆炸；stable=f16 小模型健康
```

本档在一份现成 Release 构建上实跑（CPU 分支）复现，逐字段轨迹：

```text
[dbg f32        ] 3.4772 → 3.1796（健康）
[dbg param=f16  ] 3.4772 → 3.1796（健康）
[dbg compute=f16] 3.4772 → 3.1795（健康）
[dbg stable=f16 ] 3.4785 → 3.1797（健康 —— 小模型/vocab 32 不触发 CE 溢出）
[dbg opt=f16    ] 3.4772 → 3.50e6 → … → nan（爆炸）
[dbg f16(CLI)   ] 3.4772 → 3.1796（健康）
```

**覆盖缺口已补（2026-10-13，F2）**：`f16_precision_test` 的 profile 组合矩阵已补
`{"all=f16", nn::profile_all_f16()}` 行，并从「只在 CPU + `NN_F16_DEBUG`」**提为默认 ctest
断言**（`run_profile_matrix`，CPU 与 GPU 两分支都跑）：健康组合断言 loss 全程有限且不增，
危险组合只断言「能跑完」。整条矩阵 CPU 实跑约 0.1s（廉价）。
另有 `test_ce_loss_stable_f16_no_overflow` 断言大 token 数下 `stable=f16` 的 CE loss 有限且与
f32 参考一致（`C=100, N=20000`，CPU/GPU 实测 `f32 5.1876 = stable=f16 5.1876`）。

## 3. 缺陷 B：f16 训练冻结（两个不同机制，勿混）

### 3.1 B1 — GPU `stable=f16` loss 冻结（**已修 + 根因订正**）

见 [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)。症状（原报告）：
GPU + `--precision-stable f16` 时 `text_train` 打印的 loss 恒为 `4.0137`（跨 epoch/数据集/进程逐位
相同），**权重照常更新**，CPU 同配置健康。

**根因订正（2026-10-13，本档实测）**：14 号台账把该现象归到「loss 回读/报告链」（嫌疑
`cast`/异步槽位读陈旧）。实测**不是**回读问题——原「回读链」的证据（常数、与权重/数据无关、
跨进程逐位相同）同样完全由**数值溢出**解释：

- `CrossEntropyLoss` 的 (1,1) loss 和 ≈ `-log(vocab)·token 数`，**按 `p_.stable` 落盘**；
  大 token 数下超 f16 上限 65504 → 部分硬件溢出为 ±inf、部分饱和为 f16 max。
  `4.0137 ≈ 65504 / num_valid` ⇒ 正是**饱和**形态（数据无关、逐位相同）。
- 本机（850M）实测对照（`text_train`，`seq256×b32`=8192 token，仅差 `stable`）：
  `--f16`（stable=f32）→ `9.0063 → 8.7390`（健康）；`stable=f16` → **`inf`（每步）**。
  小 token 数（`seq128×b8`=1024）下 `stable=f16` 健康——因为 sum 未越 65504。

**修复**：同 §2.1（CE 两条链的 (1,1) 归约输出恒 F32）。修复后本机 GPU 复测原 `inf` 配置 →
`9.0056 → 8.6922` 正常下降；`f16_precision_test` 的 `test_ce_loss_stable_f16_no_overflow`
在 CPU/GPU 双端通过。**回读路径（`forward_sparse_sum → cast → submit_scalar_readback →
harvest_loss`）本身无误**——`ce_fusion_test` 的「异步标量回读 vs 同步下载」用例（迭代 24 轮
> 槽位 8，覆盖环形复用）持续通过。

### 3.2 B2 — 四字段全 f16 的数值性冻结（已定位）

`Adam::adam_update_` K3 的更新量 `delta` 以 `upd_prec`（此处 = `p_.param` = F16）落盘，随后
`compute_into(p + delta)` 原地写回。当 `|delta|` 小于参数 `p` 所在 f16 网格步长（≈ `|p| · 2⁻¹⁰`）时，
`p + delta` 舍入回原值 → **参数不再移动、loss 恒定**（`precision.hpp` 注记的「32.1」即此形态）。
这是**数值性**冻结（与设备无关），与 B1 的**回读链**冻结是两回事。

> 判别口诀：B1 权重在动、只有打印冻结（设备链路）；B2 权重真的不动（精度吃光更新）。

> **后续（2026-10-14）**：F5 的 Adam 状态缩放让 m/v 回到 f16 正常区，B2 的「更新被舍入吃光」
> 在 tiny GPT / 500 步记录中不再出现；但「长跑是否始终不动」仍归 F6 的健康断言（§4.6 / 26 §8）。

## 4. 缺陷 C：f16 训练降速（历史归因；已由 F4 兑现）

### 4.1 复现数据（40HX，GPT d64/h4/L4/ff256/seq256/B64/adam lr1e-3）

| 采集 | f32 | `--f16`（健康路径） | 判定 |
|---|---|---|---|
| 2026-09-25 基准 | 101.0 ms/step（162.2K tok/s） | 118.4 ms/step（138.4K tok/s） | f16 **慢 17%** |
| 2026-10-06 基准 | 107.1 ms/step（152.9K tok/s） | 118.4 ms/step（138.4K tok/s） | f16 **慢 11%** |

`--f16` 显存低于 f32、且轨迹健康（注意 `--f16` 不是全 f16），唯**速度反增**，
与「f16 省带宽 → 更快」预期相反。

### 4.2 已定位部分

`docs/history/history.md`「设备侧 kernel 表按阶段归因」（2026-10-06）实测：整步时间 **~30% 卡在 wgrad
工作组饥饿**——tiled matmul 输出块 64×64、`grid = ceil(N/64)×ceil(M/64)`，反传 wgrad 形状
`M=64..256, K=16384, N=64..256` 只派 **3–4 个工作组**（40HX ≈ 34 SM 大面积空转）。

该问题已由 **P0 细 tile 变体**部分修复（2026-10-07：网格饥饿时改派 `#f` 变体，输出块 64²→16²；
f32 e2e 106.4→98.1 ms/step，**−7.8%**，对 torch 差距 2.10×→1.94×）。但这是**把 f32 拉快**，
**不改变「f16 不省时间」**——`history` 记「⑤ f16 提速兑现（P3）」仍未做。

### 4.3 待定位 / 候选来源

1. **边界 cast 的 f32 临时量**：仍走边界 cast 的算子对每个 f16 入参抬 f32 副本（§12.11 ④ 记残余
   ~550MB cast），f16 直读直写的收益被这部分抵消；
2. **f16 GEMM 收益被 K 链/LDS 地板吃掉**：细 tile 三设计与 naive 都在 ~300–440 GFLOPS 平台、
   耗时对 K 线性（每 k 迭代 ~80–134ns 串行地板），f16 未把该地板搬走；
3. **CPU f16 无 ISA**：逐元素 decode/encode 开销使 CPU f16「省内存不省时间」（§12.4 已知限制，
   独立于 GPU 现象）。

→ 建议：以 `NN_PROFILE=1 text_train --f16` 与 f32 逐步对照（阶段表 + per-key 表），把「f16 多出来的
时间」归到具体阶段/kernel（期望落点：CE 链、无变体恢复的边界 cast、或 f16 pipeline 占用率）。

### 4.4 本机（850M）归因测量（2026-10-13，**仅供参考，不可外推到 40HX**）

命令（`build/gpu`，`NN_PROFILE=1`，`d64/h4/L4/ff256/seq256/b16`，15 步，`--gpu=850M`）：

| | f32 | `--f16` |
|---|---|---|
| step 墙钟 | 2337 ms（155.8 ms/step） | 2416 ms（161.1 ms/step，**+3.4%**） |
| forward | 1547 ms（65.0%） | 1620 ms（65.1%） |
| optimizer | 403 ms（16.9%） | 452 ms（18.1%） |
| backward | 295 ms（12.4%） | 263 ms（10.6%） |

净归属 top：两者都是 `flush_batch`（36–39%）、`copy_from`（17–18%）、`begin_batch`（9–13%）、
`zero`（7–12%）、`eval_expr`（6–8%）；**`cast` 仅 0.3%**——即 §4.3① 的「边界 cast 临时量」在本
配置下**不是**主因。**为何不可外推**：850M 是 Maxwell，**无 `shaderFloat16`** → f16 ALU 走回退，
其「f16 损耗」结构（ALU 回退 vs 40HX 的带宽/GEMM 账）与 40HX 不同；本机只看到 +3.4%（对照
40HX 的 +11–17%）。

**结论**：F4 的验收**必须在真机（40HX）上做**；本轮不改 shader/派发。**方案与口径**见
[`26-f16-speedup-plan.md`](26-f16-speedup-plan.md)（用户要求改为「**任意设备**都要提速、不依赖
`shaderFloat16`」，主线 = 带宽优先 + 先清 step 内固定开销）。
可复用的下一步：在 40HX 上跑同口径 `NN_PROFILE=1`（阶段表 + per-key 表）对照本表，把 f16 多出的
时间落到具体阶段/kernel（`flush_batch`/`copy_from`/`begin_batch` 这类**与精度无关**的固定开销是
共同分母，f16 的真实增量要看 `eval_expr`/`matmul` 侧的 per-key 差）。

> **附带发现 → 2026-10-14 已评估-不做**：上面看到的 `copy_from`（每步约 20%，**与精度无关**）经实测
> 属"host 等待总量不变、位置可移"（消除 acquire 阻塞后 `copy_from` 392→3 ms 而 `flush_batch`
> 955→1213 ms，墙钟不变；`--flush-interval 1→4` 反而 +2.5%）→ **无收益，已归档**（[`26`](26-f16-speedup-plan.md) §4 S0）。
> 原归口
> [`18-roadmap.md`](18-roadmap.md) **P2-9**（step 内固定 I/O 开销归因与消除）——它与 F4 是两件
> 事：F4 是「f16 相对 f32 的增量」，P2-9 是「两条路径共有的固定开销」。修 P2-9 会同时拉快 f16
> 与 f32，且不改数值。

### 4.5 S1s/S1t 编译期特化的实测（2026-10-13，**本轮兑现**）

**根因订正**：上一轮把"f16 慢"归到"固定开销掩盖 + 主导张量仍 f32"。**实测真正的最大单项是
运行期精度分派本身**：`#x` 是"一个 shader 覆盖任意签名"的载体，热循环里每次读写都带
`((prec >> i & 1u) ? float(bNh[…]) : bN[…])` 三元。**对照（只换 shader，不改任何数值口径）**：

| 结构 / kernel | f32 基线 | `#x`（分派） | `#s`/`#xxxx`（特化） |
|---|---|---|---|
| CE 融合注意力 `f7e6c2bd8e35af36`（850M，per block） | 117.2 ms | 146.1 ms | **120.0 ms** |
| LM head `c2aca8a29eb4617f` | 137.1 | 154.4 | **137.4** |
| wgrad `71733007a831714a` | 140.6 | 143.9 | **138.8** |
| `6996c1ccdb7656fc` | 111.5 | 117.1 | **112.2** |

**修复** = 在构建期为**扫描期真实观察到的签名**发射静态特化 shader（键 `key#xxxx`；另有无条件
的 `key#s` = 全输入 f16 + 输出 f16 兜底）。纯优化：未枚举到的签名照旧落 `#x`。只需
`storageBuffer16BitAccess`（**不需要** `shaderFloat16`）。详见
[`26-f16-speedup-plan.md`](26-f16-speedup-plan.md) §4 S1s。

**850M（GTX 850M，Maxwell，无 `shaderFloat16`）**，`d64/h4/L4/ff256/seq256/b16`，15 步，3 轮：

| | f32 | `--f16`（修前） | `--f16`（修后） |
|---|---|---|---|
| step 墙钟（平均） | 2037 ms | 2260 ms（+10.6%） | **2055 ms（+0.9%）** |
| 设备 kernel 合计 | 1949 ms | 2163 ms | **1962 ms** |
| 峰值显存（`nvidia-smi` 采样） | 818 MiB | — | **606 MiB（−26%）** |

**40HX（CMP 40HX，Turing，有 `shaderFloat16`）**，`d64/h4/L4/ff256/seq256/b64`，15 步：

| | f32 | `--f16` |
|---|---|---|
| step 墙钟 | 1027 ms | 1349 ms（**+30%**，S1s 之前） |
| step 墙钟（P2-13 之后） | 1027 ms | 1013 ms（−1.4%） |
| **step 墙钟（2026-10-14 全部改动后，3 轮）** | **1046 ms** | **1013 ms（−3.2%）** |

归因（40HX，per-key 设备时间差）：**残差全部来自一个结构**——
`71733007a831714a@fused_mm@bwd.blk#*`（wgrad）：f32 走 **K 切分 `#ks`** 25 ms/块，f16 当时没有
K 切分变体（`#ks`/`#kr`、`#f` **按设计只发 V0 全 f32**）→ 掉回 64-tile **79 ms/块**，
四块合计 +368 ms（其余结构 f16 反而更快，净 −50 ms）。配对的旁证（同机 f32 关逃生阀）：
`NN_MM_NO_KSPLIT=1` → 细 tile 84 ms/块（仍远差于 K 切分 25 ms）→ **只有 K 切分能救**，
细 tile 不够。→ 立项 [`18`](18-roadmap.md) **P2-13**。

**P2-13 已落地（2026-10-14）**：把 sig 穿进 `#f`/`#ks`/`#kr` 生成路径，为每个静态签名变体各
发一份网格粒度变体（键形如 `key#10003#ks`），部分和缓冲恒 f32。实测该 wgrad
**f16 16.0 ms/块 < f32 25.1 ms/块**；端到端 **40HX `--f16` 1013 vs f32 1027 ms（−1.4%）**、
**850M `--f16` 2005 vs f32 2038 ms（−1.6%）** —— 两台机器上 f16 都转为**更快**。
loss 对拍（40HX）：f32 8.6757/8.2646/7.8743 vs f16 8.6676/8.2569/7.8643（<0.15%）。
**S1（logits 落 f16）已实测并回退（2026-10-14）**：LM head 前向 −7.9 ms，但 CE 反向从 V0 全 f32
77.4 → **116.4 ms（掉回 `#x` 分派）** —— 因为 `scan_exprs` 只跑 `ce.forward_sparse`、从不跑
`ce.backward()` ⇒ 该结构的签名未被观察 ⇒ 无特化变体。净值 **+31 ms**，且 loss 轨迹偏离 ~3%
（f16 logits 让 CE 梯度也按 f16 落盘）。→ 当前形态下不采用（[`26`](26-f16-speedup-plan.md) §4 S1）。

次要项：优化器在 f16 下掉出批量派发 `#b`（只发 f32）→ 4 个逐 kernel 融合，+12.6 ms/15 步。
**P2-12 已完成（2026-10-14）**，分两段：
① `#b` 改对"extras 为空的静态签名变体"各发一份 → 4 个结构里 2 个（K1/K2）从 980 次/个降到 42 次/个。
② **残差收口**：另 2 个结构（K3/K4）是 `--f16` 下**逐成员降级的复合 kernel**，根因**不是派发器**，
而是 **scan 与运行期不同源**——(a) `scan_exprs` 的优化器 dry-run 没把当前 pass 的 precision profile
透传给 `create_optimizer`（`text_train` 是透传的）⇒ f16 pass 里融合出的复合结构与运行期不同 ⇒ 运行期
命中不到 ⇒ 降级；(b) 复合 kernel 的**运行期精度签名**在 scan 侧没有登记入口（单节点结构有
`scan_register`/`eval_expr*`，融合结构只在 GPU `end_expr` 执行时才算出）⇒ 只有 V0/`#s`/`#x`，拿不到
`#xxxx` 及其 `#b`。修法 = dry-run 透传 profile + scan 的 CPU `end_expr` 按同源口径补记复合签名。
**实测（40HX，f16，`d64/h4/L4/ff256/seq256/b64`，20 步，`329f8f1` vs `5bb6a51` 配对）**：
`[ir-c][degrade]` **70 → 0**/步；优化器派发 **138.7 → 8.55 calls/步（−94%）**（每步恰 3 组批量：
K1 `#0002#b` / K2 `#0002#b` / 复合 `#10004#b`，各 32+32+6 实例）；优化器 host 阶段
**9.94 → 6.51 ms/步（−34%）**；f16 loss 轨迹与 f32 同步（850M 9.009→8.594 vs 9.006→8.591）。
**端到端墙钟无可见收益**（1320.95 → 1324.08 ms，噪声内）——设备忙率 95.5%，host 节省被 overlap
吃掉；这比原来的"≈1 ms/step ≈0.5%"估计更保守。字节锚逐位不变；`--f16`/`adamw --f16`/`--t1_58`
均 0 降级；`stable=f16`+`optimizer=f16` 组合仍按设计回落降级（构建期 2^4 枚举不可取）。

**口径小结**：S1s/S1t 让「存储路径」在**任意设备**上都把 f16 从"慢 10~30%"拉回"持平或更好"
（850M 持平 + 显存 −26%；40HX 只差 K 切分这一项）。**原生 f16（`shaderFloat16`）已启用且在用**
（40HX 上 9 条结构命中 `#a`），但 `expr_prec_sig_native16` 仍排除 fold/matmul/reduce → 用足
仍是 [`18`](18-roadmap.md) **P2-10** 的开放项。

### 4.6 F5：四字段全 f16 落地实测（2026-10-14）

**根因（实测定位）**：`--precision-optimizer f16` 下第 1 步更新后 loss 即 NaN（CPU/GPU 同）。逐项 bisect：`--f16`(param/compute) 健康、`+stable=f16` 健康、**`+optimizer=f16` 复现**。机制 = Adam 二阶矩 `v ≈ (1-β2)·g² ~ 1e-10` 在 f16 下**下溢为 0**，而一阶矩 `m ~ 1e-8` 仍在次正规区
⇒ `m/(√v+eps)` 退化（分子退化值 × 1000 的 bias 修正 / 分母只剩 eps）⇒ 更新量暴涨 ⇒ 参数越界 ⇒ NaN。

**修法**：`Adam::optimizer_state_scale_()` —— 状态存储精度 < 4B 时取 **S = 4096**（2 的幂 ⇒ 与 g 相乘在 f16 下无损），把 `(1-β1)·S`、`(1-β2)·S²`、`eps·S` 作为 K1/K2/K3 的 rparam。Adam 的更新是尺度不变的 ⇒ **更新量数学不变**（f16 舍入内），但 m/v 回到正常区。S 只由精度决定 ⇒ 确定性；checkpoint 存的即缩放后的状态，可原样读回。

**门禁与实测**：
- `f16_precision_test` 新增「四字段全 f16 可训练」段（tiny GPT 20 步：loss 有限且下降）。
- profile 组合矩阵里 `all=f16`（此前为 NaN 的失败形态）现为 `ok=1`，轨迹 3.4778 → 3.1780（f32 3.4772 → 3.1796，差 ≤0.002）。
- 端到端（850M）：CPU 100 步 7.78 → 5.96、GPU 100 步 7.73 → 6.16、**GPU 500 步 6.05 → 4.05**，全程无 NaN。

**残留边界**：`|g|·S` 逼近 f16 上限 65504（即 |g| ≳ 0.0625）时 v 会溢出 ⇒ 实践配 `--max-norm`；F1 的启动期告警保持有效（`optimizer=f16` 仍属"显式实验"档）。

> **与真 f16 的差距（2026-10-18）**：本节只证明“四字段全 f16 可训练”。当前 `S` / `eps` 仍以
> f32 rparam 传入，CE loss 标量仍恒 F32，matmul/归约仍有 f32 累加，native16 也未全覆盖——这些
> 都不满足 [`26`](26-f16-speedup-plan.md) §8 的真 f16 零转换门禁，归 F6/R2–R4。

## 5. 立项（todo）

| 立项 | 事项 | 验收口径 | 依赖 | 优先级 | 状态 |
|---|---|---|---|---|---|
| **F1** 危险精度组合**启动期告警**（口径：仅警告，不拒绝） | 启动期对 `optimizer=f16` 与四字段全 f16 打印明确警告后**继续运行**；`--f16` 与合法组合静默不受影响。**不** fail-fast、**不**加确认开关（备注：`stable=f16` 单独出现是否单列待定，见 §2.3） | 危险组合在 stderr 打印含字段/风险/文档指引的警告，退出码与训练行为不变；新增 CLI 断言测试（捕获 stderr） | 无 | 对应 [`18-roadmap.md`](18-roadmap.md) **P1-3**（该处已改为引用本档、不再单独定义） | ✅ **已完成 2026-10-13**（`cli_precision_warning.hpp` + 门禁 `cli_precision_warning_test`） |
| **F2** 补 GPU + 全 f16 覆盖缺口 | profile 组合矩阵补 `{F16,F16,F16,F16}` 行（CPU、廉价）；把 stable=f16 / 全 f16 形态搬到 GPU 分支做数值断言。**端到端全 f16 模型训练不进 ctest**（太费时，见下注），改挂到大版本发布清单 [`25`](25-release-requirements.md) 第 1 项「端到端模型训练测试」 | 单字段 / 全 f16 组合有**廉价**断言进默认门禁；端到端全 f16 轨迹在大版本发布时实跑通过 | F1（或与 F1 同批） | 高（关 [`14`](14-f16-stable-gpu-loss-frozen.md) §3 缺口） | ✅ **已完成（组合矩阵 + GPU 分支 + CE 溢出回归，2026-10-13）**；端到端全 f16 仍归发布清单 |
| **F3** 修 GPU `stable=f16` loss 冻结 | 最小复现 → 定位并修。**实际结论**：不是回读链（`cast`/槽位）问题，而是 **CE 的 (1,1) loss 归约按 `stable=f16` 落盘溢出**（见 §3.1 订正） | 冻结/`inf` 消失；GPU + stable=f16 loss 曲线可观测；回归用例 | 无 | 高（既有 Open 故障，[`14`](14-f16-stable-gpu-loss-frozen.md)） | ✅ **已完成 2026-10-13**（详见 §3.1） |
| **F4** f16 提速兑现 | 归因「f16 比 f32 慢」的时间去向；方案见 [`26`](26-f16-speedup-plan.md)（**存储路径**带宽优先、**原生 f16 用足**；口径「任意设备」） | ① 存储路径在任意设备（40HX / 850M / CPU）上 `--f16` step ≤ f32 且峰值显存更低、**不依赖** `shaderFloat16`；② capable 设备额外：原生 f16 已启用（`NN_VULKAN_NO_16BIT_ALU` A/B 显示原生开更快） | F3 等 | 中 | ✅ **已完成 2026-10-14**。根因订正 = 「运行期精度分派 `#x` 的三元税」为主（非"固定开销掩盖"、非"logits 仍 f32"）→ 用 **S1s/S1t 编译期特化** + **网格粒度变体（P2-13）** 修掉。**跨设备结果**（各 15 步 ×3 轮）：**850M**（无 `shaderFloat16`）`--f16` **2005 vs f32 2038 ms（−1.6%）**、峰值显存 **606 vs 818 MiB（−26%）**；**40HX**（有原生 f16 ALU）`--f16` **1013 vs f32 1046 ms（−3.2%）**；**CPU** f16 不再慢于 f32（3.6→1.9 s，见 P2-8）。另外 **F5 全 f16 已可训练**（§4.6）。S0/S1/S2/S5 的评估见 [`26`](26-f16-speedup-plan.md)（0/1/2 为实测后不做、5 为资格不再扩张） |
| **F5** 全 f16 可用性 | ✅ **可训练性完成 2026-10-14**（产品化转 F6）。落地形态与原方案不同：不用训练循环里的 loss scaling，而是**在优化器状态更新处缩放**（`Adam::optimizer_state_scale_`）——进入 m/v 的梯度放大 **S=4096**（2 的幂 → f16 下无损）、`(1-β1)·S` / `(1-β2)·S²` / `eps·S` 同步换算。**为什么有效**：Adam 的更新是尺度不变的 `lr·(S·m̂)/(√(S²·v̂)+S·eps) = lr·m̂/(√v̂+eps)` ⇒ **更新量数学不变**，但 m~g·S、v~g²·S² 都回到 f16 正常区（根因 = v ≈ (1-β2)·g² ~ 1e-10 在 f16 下**下溢为 0** 而 m 仍在次正规区 ⇒ `m/(√v+eps)` 退化、更新暴涨 → 第 1 步即 NaN）。比 loss scaling 更内聚：不改 CLI / 不改梯度与裁剪语义、无"是否反缩放"分支；rparam 恒 f32 传递 ⇒ 原方案里的"eps flush-to-0"不成问题（eps 的**相对**比例由 `eps·S` 保持）。**边界**：|g| ≳ 0.0625 时 v 会溢出 f16 ⇒ 实践配 `--max-norm` 裁剪；启动期精度告警（F1）保持有效 | 四字段全 f16 在 CPU/GPU 上健康收敛 | F1/F2/F3 | 中 | ✅ **可训练性已完成**（见 §4.6）；但 CE 标量仍是 F32 补丁、native16 未全覆盖 ⇒ **不等于真 f16**；🟡 **全流程真 f16 产品化 = F6** |
| **F6** 全流程真 f16 产品化（**2026-10-18 恢复立项**） | 把 F5 的「可训练性」推进到「真 f16、可长期、全模型、可发布」：**B-Mem = 无 f32 持久态**；capable 设备另追 **B-Arith = 零 f16↔f32 转换**；不支持时只由计算引擎在算子内部隐式转换（Layer/DSL/CLI 零手写 cast）。包含一键 `--f16-all`、native16 全覆盖、f16 累加/归约与 CE 标量改造、logits/CE 融合、**R2.5 数值鲁棒性技术栈**（SR / GradientStabilizer / ECO / attention δ 修复）、全模型/全设备长跑与质量基线。**原 F5 的 loss scaling 不作为前置**（F5 已用 optimizer state scaling 解决下溢）；F5 的 F32 CE 标量补丁与 f32 累加器必须在 R2 消除 | [`26`](26-f16-speedup-plan.md) §8.3 R0–R6、§8.B.7 全部门禁；B-Mem 必达；B-Arith = capable 设备 `cast`/`cast_into` = 0、f32 临时量 = 0（补偿 kernel 豁免表）、native16 全覆盖；incapable 设备转换只在引擎内；至少 GPT/MLP/CNN/RAPT × CPU/850M/40HX 的长跑无 NaN、loss 下降、权重移动；`gui_cli_audit` 退出码 0 | F5 | 中–大 | 🟡 **活跃（2026-10-18）**，细则见 [`26`](26-f16-speedup-plan.md) §8 |
| **F7** 标准 `--f16` 训练（Plan A，**2026-10-18 立项**） | 参考 PyTorch AMP：`profile_f16()` 为默认（f16 参数/激活 + f32 stable/optimizer），`profile_master_weights()` 为可选 master-weights 形态；补动态 loss scaling（GradScaler 语义）、敏感链 f32 与 PyTorch AMP 对照。**不要求零转换/native16 全覆盖** | 同数据/同 seeds：`--f16` 与 f32 轨迹在冻结容差内、无 NaN/冻结、权重移动；与 PyTorch AMP 同量级；loss scaling 的 growth/inf-skip/恢复用例通过；峰值显存低于 f32，step 不慢于 f32 | F5 | 中 | 🟡 **活跃（2026-10-18）**，细则见 [`26`](26-f16-speedup-plan.md) §8.A |

> 注：26 号已把两套计划拆为 **§8.A（Plan A / F7）** 与 **§8.B（Plan B / F6）**；此前文档里的
> “§8.3 / §8.4”现分别指 §8.B.3 / §8.B.4。

> **纪律**：F1–F4 属「先止血再提速」；F5 解决「全 f16 能不能训练」；**F7（Plan A）**解决「标准
> `--f16` 能不能按 PyTorch AMP 口径稳定训练」；**F6（Plan B）**解决「真 f16 能不能零转换、全模型、
> 可发布地训练」。**当前**：F1–F5 已完成；F6/F7 于 2026-10-18 立项。CLI 仍保持
> `--f16` = 推荐安全配方，`--f16-all` 在 R5 通过前是显式实验档；启动期告警保留，但文案必须从
> 「必崩/不可训练」改为「数值边界与质量权衡」（R0）。
>
> **训练为何不进 ctest**：ctest 要能在 CI runner 上分钟级跑完，而模型端到端训练（含全 f16 形态）
> 太费时——故默认门禁只放**廉价**断言，端到端训练归口大版本发布清单
> [`25-release-requirements.md`](25-release-requirements.md)（F2 因此只把组合矩阵 / 数值断言放进 ctest）。

## 6. 证据索引

| 来源 | 内容 |
|---|---|
| `include/neuralnet.cpp/compute_loss.hpp` | `softmax_cols_` / `forward` / `fused_forward_sparse_`：整条 CE 链按 `p_.stable` 落盘 |
| `include/neuralnet.cpp/compute_optimizer.hpp` | `Adam::adam_update_`：K2 `p_.optimizer`、K3 `upd_prec`；默认 `eps=1e-8` |
| `include/neuralnet.cpp/precision.hpp` | `profile_f16()` / `profile_all_f16()`；f16 数值注记（L539–545） |
| `src/f16_precision_test.cpp` | `debug_profile_matrix`（cases 无 all=f16，仅 CPU + `NN_F16_DEBUG`）、`train_tiny_gpt` 逐阶段 `scan` |
| `src/text_train.cpp` | 每步 loss 链（`forward_sparse_sum` / `cast` / `submit_scalar_readback` / `harvest_loss`）；`--precision-*` 只赋值不校验 |
| `include/neuralnet.cpp/backend/compute_vk_backend_ops.inc` | matmul dispatch 网格（`EXPR_MATMUL_BLOCK`，`use_fine` 细 tile） |
| `docs/development/05-mixed-precision.md` | §12.4 已知限制、§12.5 全链精度接线与数值结论、§12.11 op-level f16 GEMM 与当前实测 |
| [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md) | GPU `stable=f16` loss 回读冻结：触发矩阵、证据、缺口 |
| `docs/benchmarks/2026-09-25-vulkan-vs-cuda.md` / `2026-10-06-vulkan-vs-torch.md` | f16 降速实测（118.4 vs 101.0 / 107.1 ms/step） |
| `docs/history/history.md` | 「设备侧 kernel 表按阶段归因」（wgrad 派发饥饿）、「matmul 细 tile 变体」（P0 修复） |
