# 24 — f16 训练缺陷台账：全 f16 NaN / loss 冻结 / 降速

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。

> 状态：**NaN 已定位（本档核对并确认根因）/ GPU loss 冻结未修（[`14`](14-f16-stable-gpu-loss-frozen.md) Open）/
> f16 降速未兑现（P3）**。
> 来源：核对用户提供的《全 f16 训练 NaN 问题定位报告》，并按追加要求定位「f16 训练冻结」
> 与「f16 训练速度降低」两项，合并立项。
> 关联：`docs/development/05-mixed-precision.md` §12.4/§12.5/§12.11；`precision.hpp`（`profile_all_f16`
> 数值注记）；[`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)；
> [`18-roadmap.md`](18-roadmap.md) §3.1 的 C1/C2 与 §4.2 的 P1-3。

## 0. 结论速览

「全 f16」= `profile_all_f16()` = **四字段全 F16**（`--f16 --precision-stable f16 --precision-optimizer f16`），
区别于健康推荐路径 `--f16` = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}。
本档把用户报告的「全 f16 不可训练」与追加定位的「冻结」「降速」并列立项。

| 缺陷 | 现象 | 定位结论 | 坐标 | 状态 |
|---|---|---|---|---|
| **A 全 f16 NaN** | ① epoch2 起 loss `-nan`；② `stable=f16` CE 链 ~200 步 NaN | **NaN 直接生产者 = `stable=F16`**（CE/softmax 归约溢出 65504 + 尾数不足）；**前导放大器 = `optimizer=F16`**（Adam 二阶矩下溢 → 更新爆炸 → logits 出 f16 范围） | `compute_loss.hpp`（`softmax_cols_` / `fused_forward_sparse_`）、`compute_optimizer.hpp`（`Adam::adam_update_`） | 已定位（本档） |
| **B1 冻结（GPU）** | GPU + `stable=f16` 时 loss 打印恒 `4.0137`，权重照常更新 | 冻结在 **loss 回读/报告链**，非权重 | `src/text_train.cpp`（`forward_sparse_sum`→`cast`→`submit_scalar_readback`→`harvest_loss`） | 未修（[`14`](14-f16-stable-gpu-loss-frozen.md) Open，未定根因） |
| **B2 冻结（数值）** | 四字段全 f16 时 loss 恒定（更新被 f16 舍入吃光） | 数值性冻结：`upd_prec=F16` 下 `p += delta` 的 `|delta|` 小于参数所在 f16 网格步长 → 原地写回无变化 | `compute_optimizer.hpp` `Adam::adam_update_` K3（`delta` + `compute_into(p + delta)`） | 已定位（机制清楚，与 `precision.hpp` 注记一致） |
| **C 降速** | `--f16`（健康路径）比 f32 **慢 11–17%**，与「f16 应省带宽/更快」的预期相反 | **部分定位**：step 时间 ~30% 卡在 **wgrad 工作组饥饿**（细 tile 变体 2026-10-07 已落地，f32 e2e −7.8%），**f16 提速兑现（P3）未做**；另有边界 cast 的 f32 临时量与 CPU 无 f16 ISA 两个已知来源 | `backend/compute_vk_backend_ops.inc`（matmul dispatch 网格）、`shaders/matmul_tiled.comp`；`docs/development/05-mixed-precision.md` §12.11 | 未兑现（P3） |

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
| 根因③「四字段全 f16 → loss 恒定」 | ✅ 机制证实；**常数口径需注记**：`precision.hpp` 注记为 **32.1**，GPU [`14`](14-f16-stable-gpu-loss-frozen.md) 冻结常数为 **4.0137**——两者是不同配置/链路的两个常数，报告把二者并陈，勿混为一谈。「32.1」在仓库内仅见于 `precision.hpp` 注释，无 bench 产物背书 | `precision.hpp` L543；[`14`](14-f16-stable-gpu-loss-frozen.md) §2 |
| 报告「14 号报告 Open / GPU 特有」 | ✅ 证实 | [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)（状态 Open） |
| 报告 §三.2「`debug_profile_matrix` 矩阵**没有**四槽全 f16 这一行」 | ✅ 证实（cases = f32 / param / compute / stable / opt / f16(CLI)，无 all=f16） | `src/f16_precision_test.cpp` `debug_profile_matrix`（cases L639–646） |
| 报告 §三.2「预期 `[dbg stable=f16]` 与 `[dbg opt=f16]` 出现非有限/爆炸」 | ⚠️ **部分更正**（本档已实测复现，见 §2.4）：`opt=f16` ✅ 爆炸（`3.4772 → 3.50e6 → … → nan`）；`stable=f16` 在该 CPU 小 GPT（vocab 32、20 步）下**健康**（`3.4785 → 3.1797`）。~200 步 NaN 需大词表 + 更多步，报告对小矩阵的预期与实测矛盾 | [`14`](14-f16-stable-gpu-loss-frozen.md) §2.3；§2.4 复现输出 |
| 报告 §三.3 `train_tiny_gpt(...).scan()` 逐阶段定位首个非有限值 | ✅ 证实（`train_tiny_gpt` + `scan` 存在） | `src/f16_precision_test.cpp` `train_tiny_gpt` |
| 报告 §四 `--f16` = 安全配方；全 f16 仅供显式实验 | ✅ 证实 | `precision.hpp` L544–545；`docs/development/05-mixed-precision.md` §9.4 |
| 报告 §四「CLI 应保持现状，全 f16 仅作显式实验并在启动期给警告」 | ⚠️ **待做**：当前 `--precision-stable/-optimizer f16` **只赋值不校验、无任何启动期提示**（危险组合可无告警组合出） | `src/text_train.cpp` L585–591；[`18-roadmap.md`](18-roadmap.md) P1-3 |
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

### 2.2 `optimizer=F16` 是前导放大器

`Adam::adam_update_` 的 K2 以 `p_.optimizer` 存 `v = β2·v + (1-β2)·g²`。小梯度 `g~1e-5` 时
`g²~1e-10` 低于 f16 次正规最小 5.96e-8 → `v` 舍入为 0；K3 `delta = -lr·m̂/(sqrt(v̂)+eps)` 分母
`sqrt(0)+1e-8` 在 f16 中进一步坍缩 → **单步更新爆炸** → logits 超出 f16 范围 → forward 侧 cast 出
inf → CE 链 `inf-inf = NaN`。这与 `precision.hpp` L539–541、§12.5 的实测（loss 7.9→3.6e4 / 1.8e7）一致；
本档 CPU 小 GPT 实测 `opt=f16` 轨迹 `3.4772 → 3.50e6 → 5.06e6 → … → nan`（同族形态）。

### 2.3 CLI 语义

`src/text_train.cpp` 的 `--precision-param/-compute/-stable/-optimizer` 只赋值、**不校验**——
用户可无告警地组合出 `optimizer=f16` / 四字段全 f16 这类「必崩」配置（[`18-roadmap.md`](18-roadmap.md) P1-3 已立项）。

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

**覆盖缺口**：`debug_profile_matrix` 的 cases **无 `{F16,F16,F16,F16}`**，且整条矩阵只在 CPU
且 `NN_F16_DEBUG` 下执行（不进默认 ctest）。如需钉死组合，应在 cases 补一
`{"all=f16", nn::profile_all_f16()}`，并考虑搬到 GPU 分支（见 §5 立项）。

## 3. 缺陷 B：f16 训练冻结（两个不同机制，勿混）

### 3.1 B1 — GPU `stable=f16` loss 回读冻结（未修）

见 [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md)（状态 Open）：
GPU + `--precision-stable f16` 时 `text_train` 打印的 loss 恒为 `4.0137`（跨 epoch/数据集/进程逐位相同），
**权重照常更新**（同进程跨 epoch 快照 63% 参数字节不同），CPU 同配置健康。链路坐标
（行号会腐烂，给符号入口）：

```text
text_train.cpp: forward_sparse_sum ──▶ cast(loss_sum -> F32) ──▶ submit_scalar_readback(slot)
              ...  harvest_loss: loss = -sum * inv_num_valid（打印）
```

嫌疑集中在 `forward_sparse_sum` 的 f16 归约输出 / `cast` 读陈旧内容 / 异步回读槽位读到未写入数据。
根因未定，[`14`](14-f16-stable-gpu-loss-frozen.md) §7 已给出「最小复现 / 补 GPU 用例 / 回读二分」三步。

### 3.2 B2 — 四字段全 f16 的数值性冻结（已定位）

`Adam::adam_update_` K3 的更新量 `delta` 以 `upd_prec`（此处 = `p_.param` = F16）落盘，随后
`compute_into(p + delta)` 原地写回。当 `|delta|` 小于参数 `p` 所在 f16 网格步长（≈ `|p| · 2⁻¹⁰`）时，
`p + delta` 舍入回原值 → **参数不再移动、loss 恒定**（`precision.hpp` 注记的「32.1」即此形态）。
这是**数值性**冻结（与设备无关），与 B1 的**回读链**冻结是两回事。

> 判别口诀：B1 权重在动、只有打印冻结（设备链路）；B2 权重真的不动（精度吃光更新）。

## 4. 缺陷 C：f16 训练降速（未兑现）

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

## 5. 立项（todo）

| 立项 | 事项 | 验收口径 | 依赖 | 优先级 |
|---|---|---|---|---|
| **F1** 危险精度组合 fail-fast / 告警 | 启动期拒绝或显式确认 `optimizer=f16` 与四字段全 f16；`--f16` 与合法组合不受影响 | 危险组合退出并打印原因；新增 CLI 断言测试 | 无 | 对应 [`18-roadmap.md`](18-roadmap.md) **P1-3** |
| **F2** 补 GPU + 全 f16 覆盖缺口 | `debug_profile_matrix` 补 `{F16,F16,F16,F16}` 行；把 stable=f16 / 全 f16 形态搬到 GPU 分支并纳入默认 ctest（或加端到端轨迹对拍） | GPU 下 stable=f16 与全 f16 有可观测断言；缺口用例进默认门禁 | F1（或与 F1 同批） | 高（关 [`14`](14-f16-stable-gpu-loss-frozen.md) §3 缺口） |
| **F3** 修 GPU `stable=f16` loss 回读冻结 | 最小复现（小张量 + `GpuEngine` 复现 `forward_sparse_sum` f16 输出恒定）→ 回读二分（提交前 vs 读回）→ 定位并修 | 冻结消失；GPU + stable=f16 loss 曲线可观测；回归用例 | 无 | 高（既有 Open 故障，[`14`](14-f16-stable-gpu-loss-frozen.md)） |
| **F4** f16 提速兑现 | 归因「f16 比 f32 慢」的时间去向；按 §12.11 ① `-DNN_SHADER_F16` 手法补剩余边界 cast 变体 / 评估小 N GEMV 与 CE 链 | 40HX 上 `--f16` 同窗复测 **≤ f32**（目标：显存低且不慢）；字节锚按计划内流程重立 | F3（loss 链口径一致） | 中（P3 未兑现） |
| **F5** 全 f16 可用性（长期） | loss scaling + in-kernel f32 归约（logsumexp/col_sum 内部 f32、仅输出落 f16）+ eps 随精度自适应（≥1e-4） | 四字段全 f16 在 CPU/GPU 上健康收敛若干 epoch | F1/F2/F3 | 低（Phase 2 / P3，仅当有真实需求） |

> **纪律**：F1–F4 属「先止血再提速」；F5 是「让全 f16 真正可用」的长期项。在 F5 落地前，
> CLI 保持 `--f16` = 安全配方，全 f16 仅作显式实验（**F1 落实后必须带启动期告警**）。

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
