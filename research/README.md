# research/ — 可引用的研究档案（定义 / 实测结论 / 待办）

> **读者**：想**引用本仓实测结果**、核对算法定义、或想知道"某个数字是怎么来的、该怎么重跑"的人——包括库外使用者。
> 与 `docs/` 的分工：`docs/` 写"项目**现在**是什么"（见 [`../docs/README.md`](../docs/README.md)）；
> `research/` 写"我们**实测过**什么、结论是什么、边界在哪"，也承载尚在研究中的算法定义与待办。
> 全仓的历史叙事在
> [`../docs/history/history.md`](../docs/history/history.md)，这里只放**自包含的研究报告 + 可重跑的脚本/数据**。

## 1. 怎么读 / 怎么引用

每条**实测研究**都按同一口径组织，引用效果结论时请**连口径一起引用**：

* **结论**——一句话、带数字与样本量（例：`99.34% ± 0.049, n=5`，而不是"约 99.3%"）；
* **证据**——什么数据 / 多少轮 / 什么设备；
* **重跑**——从零复现的入口命令（不需要读源码）；
* **边界**——"本实测**不能**说明什么"（每条都单列一节，请勿跨边界外推）。

算法定义基线可以没有效果数字，但必须引用源码版本并区分“机械事实 / 设计意图 / 未验证主张”；
尚未执行的研究必须标为 `TODO`，不得写成结果。

两个通用注意事项：

1. **单轮差异不要当结论**：轮间 std 常在 0.05 个百分点量级；凡未标 `n=` 的数字都是单轮。
2. **设备/口径必须一起写**：本仓的 GPU 结论多来自 CMP 40HX（Windows / clang）；换卡
   （尤其无 `shaderFloat16` 的 Maxwell 卡）可能落到不同的 f16 路径。

## 2. 结论索引

> **证据等级**：标为“实测”的条目可引用为结果；标为“定义基线”的条目只说明当前实现；
> 标为“TODO”的条目**不可**引用为效果结论。

### 2.1 [`mnist/`](mnist/README.md) — MNIST：模型尺寸 / 训练超参 → 准确率

* **结论（可引用）**：**743 KB 的 CNN（Conv16k5+Pool → Conv32k5+Pool → FC256 → FC128 → 10，
  BatchNorm@conv）在 MNIST 上 5 轮独立运行均值 `99.34% ± 0.049`（99.26~99.38）**，
  训练成本约 114 s/轮（GPU，18 epoch）。推荐超参：
  `adamw / wd=1e-3 / lr=3e-3 / cosine + warmup 3 / batch 128 / 18 epoch`。
* **证据**：41 轮确定性广度扫描（模型尺寸 × lr × optimizer × wd × batch × 精度）+ 聚焦筛选 +
  5 轮确认。汇总表 [`results/REPORT_TABLES.md`](mnist/results/REPORT_TABLES.md)、逐 run 明细
  [`results/all_runs.csv`](mnist/results/all_runs.csv)、图
  [`results/charts/*.svg`](mnist/results/charts)、聚焦实验
  [`results/focus_cnn/REPORT.md`](mnist/results/focus_cnn/REPORT.md)。
* **重跑**：`cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release` → MNIST 报告 §6「复现」
  （含 `scripts/focus_cnn.py --stage screen/confirm/report` 与广度扫描口径）。
  权重不进仓，理由与「从零重训」的命令见 [`models/README.md`](mnist/models/README.md)。
* **边界**：见该报告 **§5.1「本实测不能说明什么」**——结论**不能外推到更难的数据集**；
  CLI 无 `--seed`，单轮差异（<0.1 pp）不当结论，能站住的只有「743 KB 稳定到 99.3 级」；
  本报告只量准确率与体积，不含吞吐/显存；训练时间为单卡单次观察值。工程细节（未跑完的扫描组、
  "训练中日志别当结果"）见该报告 §7。

### 2.2 [`f16_weight_decay/`](f16_weight_decay/REPORT.md) — GPU `param=f16` 训练 loss 中途回弹的根因

* **结论（可引用）**：`--precision-param f16` 下 AdamW 的权重衰减写回 `p *= (1-lr·wd)` 走融合
  shader 的 `float16_t(...)`（SPIR-V `OpFConvert`），**在实测驱动上按"向零截断"舍入**：该差值
  远小于 1 ULP，正确舍入应为恒等，截断则**每 optimizer step 恰好掉 1 个 f16 网格步**
  （与 lr 无关，只要 wd>0）→ 权重以 ≈0.07%/步衰减，数万步后下溢为 0 → loss 回到 `ln(V)=9.01`。
* **现状**：**根因已修**——`expr_glsl_gen.hpp` 的 f16 写出改为 `glsl_f16_store()`
  （`float16_t(nn_f16_rne(v))`，整数位运算做精确 RNE）。⚠ 报告里建议的 `packHalf2x16`
  **实测同样截断**（Vulkan-Docs #1825），最终没有采用它。
* **证据**：现场模型张量解析 + 同数据 A/B 隔离 + 收缩量的定量性质 + 最小探针
  （[`decay_probe.cpp`](f16_weight_decay/decay_probe.cpp)）+ SPIR-V 反汇编 + 时间线对账。
* **重跑**：报告 §7.2「重建探针」/ §7.3「复现实验组」。
* **边界**：舍入模式是**驱动相关**事实（"规范没钉死它"才是规范侧结论）；`wd=0` 或短跑看不到该
  现象——所以"f16 轨迹与 f32 一致"这类旧结论只在 `wd=0/短跑`下成立。

### 2.3 [`ternary_ste/`](ternary_ste/REPORT.md) — 三值 / 二值权重的 STE 训练可行性

* **结论（可引用）**：**用现有公开 API 就能跑通三值 `{-1,0,+1}`（1.58 bit）与二值 `{-1,+1}`
  的 STE 训练，零库改动**：两个探针都把教师权重模式**逐项 100% 还原**
  （三值 1370/1370、二值 2048/2048；MSE 14.21→0 / 20.95→0）。
* **证据**：两个自包含探针 [`poc_ternary_ste.cpp`](ternary_ste/poc_ternary_ste.cpp) /
  [`poc_binary_ste.cpp`](ternary_ste/poc_binary_ste.cpp)（`dsl::compute` + `select` + `rparam` + `matmul`）。
* **重跑**：报告 §1「复现」（一条 `clang++` 命令；两个程序毫秒级跑完 3000 步）。
* **边界**：**只证明算法路径可行，不证明性能**——全部是 CPU、f32 存储、单层玩具问题。它是
  [`../docs/internal/21-quantized-weights.md`](../docs/internal/21-quantized-weights.md) 的**前置证据**，
  不等于三值整机能力（打包内核 / 精度枚举 / 能力声明是另一条线）。

### 2.4 [`ternary_scale/`](ternary_scale/REPORT.md) — 三值量化的「尺度规则 × 粒度」（D1 裁定依据）

* **结论（可引用）**：三值化 `w → q·τ` 的统计量**必须用 absmean（τ = 组内 mean|w|），不能沿用
  llama.cpp TQ1_0 的 absmax**。三个真实模型一致：absmean 的三值分布携带 **1.567~1.585 bit/权重**
  （≈ `log2 3` 的理论上限），absmax 只有 **0.53~1.29 bit**（GPT 上 absmax 把 77~84% 权重归零）。
  **粒度是次要变量**：absmean 下 tensor / row / blk256 / 128 / 64 的零化率与重构误差相差 ≤2 个百分点。
* **证据**：GPT d128(3.18M) / MNIST ViT d64(198K) / MNIST CNN(108K) 三个真实权重，逐矩阵明细见报告 §3.2。
* **重跑**：探针 [`probe_scale.cpp`](ternary_scale/probe_scale.cpp)（只调公开 API）+ 报告 §1。
* **边界**：只量**权重侧统计量**（零化率 / 重构误差 / 信息熵），**不含任务准确率**；结论是
  "统计量选错会浪费 30%~92% 的存储"，不是"absmean 训练效果一定更好"。

### 2.5 [`rla/`](rla/README.md) — RLA-2 / RAPT：算法定义基线 + 效果研究 TODO

* **定义基线（可引用）**：当前 `RLA-2` 是 ReLU 线性注意力变体：Q/K 经逐头 RMSNorm、RoPE、
  逐元素 ReLU 后，用 `M_t=Σv k'ᵀ`、`z_t=Σk'` 和非负 score 加权 value；训练/推理均为序列长度
  线性的固定状态路径。详见 [`rla/01-what-is-rla.md`](rla/01-what-is-rla.md)。
* **效果结论（当前不可引用）**：质量、长上下文、吞吐、显存与可解释性实验**尚未执行**。实验矩阵、
  消融、统计口径与入库门槛见 [`rla/02-rla-effectiveness.md`](rla/02-rla-effectiveness.md)。
* **边界**：定义文档不是“RLA 优于 GPT”的证据，也不证明学术首创性；正确性测试通过不等于
  模型质量或实际效率。

## 3. 目录约定

| 入库 | 不入库（可重跑，`.gitignore` 全域忽略） |
|------|------|
| 报告 `*.md`、脚本/探针 `*.py`/`*.cpp`、汇总 `*.csv`、图 `*.svg` | 权重 `*.bin`、日志 `*.log`、中间产物 |

每个子目录是**一次独立研究/排查的自包含归口**（报告 + 脚本 + 汇总数据），互不混写。脚本里的路径
常量按本目录解析（`ROOT` = 仓库根，三层 `dirname`），因此从仓库根或子目录执行都应能跑通。

## 4. 新增一份研究时

1. 建 `research/<主题>/` 并写 `REPORT.md`（或 `README.md`），**结论先行**：开头一段给出可引用的
   结论 + 数字 + 样本量，然后才是过程。
2. 单列一节 **「本实测不能说明什么」**——边界比结论更常被人误用。
3. 给出**从零复现**的入口命令（含构建口径、设备、数据集来源）；原始日志与权重不要进仓。
4. 结论若影响代码/设计决策，在 [`../docs/internal/`](../docs/internal/) 的相应设计稿里引用本报告
   （例：`ternary_scale` → `docs/internal/21-quantized-weights.md` 的 D1 裁定）。
5. 在本文件 §2 加一条索引（结论 / 证据 / 重跑 / 边界 四段式）。

尚未执行的研究可以先用 `TODO` 状态入库，但必须明确：它不提供效果结论，不可被代码/设计稿
当作已有证据引用。算法定义与实测结果应分文件，避免后续把“设计意图”误读成“实测结论”。
