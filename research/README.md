# MNIST：模型尺寸 / 训练超参 → 准确率 研究

本目录是这次研究的**唯一归口**：报告、脚本、原始日志、模型权重、汇总表都在这里。
（仓库原有的 `tests/` 是 2026-10 的「归一化 A/B」实验包，本次研究只引用其结论，不再往里写东西。）

日期：本次会话 ｜ 设备：Windows / Xeon E5-2697A（32 线程）/ **NVIDIA CMP 40HX**（`--gpu`）
数据：MNIST 全量训练 60000 / 全量测试 10000（`datasets/mnist_data`，Kaggle CSV）

---

## 1. 目标

1. 最初目标：实证「模型尺寸（CNN 通道/深度/FC 头）、训练超参（lr、optimizer、weight decay、batch size、epoch、schedule/warmup、精度）」对 MNIST 测试准确率的影响。
2. **中途按用户澄清聚焦为**：只需要在**体积可控**的前提下，试出一个效果最好的 CNN，多轮均值 **≥99.3%** 即可。
   后续实验（第 4.3 节的聚焦筛选 + 多轮确认）按这个口径执行；第 4.2 节的广度扫描作为先验证据保留。

---

## 2. 结论（推荐配置）

**743 KB 的 CNN，5 轮独立运行均值 99.34%、单轮 99.26%~99.38%，达到「体积可控 + ≥99.3%」。**

```bash
# 先构建（原因见 §7.1，仓库自带的 build/ 是过期 Debug 产物）
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_TESTS=OFF
cmake --build build_rel

./build_rel/mnist_train.exe --arch cnn --gpu \
  --epochs 18 --batch-size 128 --optimizer adamw --weight-decay 0.001 \
  --lr 0.003 --lr-schedule cosine --warmup-epochs 3 --min-lr 1e-6 \
  --cnn-channels 16,32 --cnn-fc 256,128,10 \
  --save mnist_cnn_best_v3.bin
```

| 项 | 值 |
|---|---|
| 结构 | Conv(16,k5)+Pool2 → Conv(32,k5)+Pool2 → FC 256 → FC 128 → 10 |
| 归一化 | `BatchNorm @ conv`（CNN 架构默认，`--norm-place auto`） |
| 训练 | 18 epoch / batch 128 / AdamW wd 1e-3 / cosine + warmup 3 / lr **3e-3** |
| 模型体积 | **743.2 KB**（与既有 `tests/results/models_reference/mnist_cnn_best_v2.bin` 715 KB 同级） |
| 测试准确率 | **mean 99.34% ± 0.049，n=5**（99.32 / 99.38 / 99.37 / 99.36 / 99.26） |
| 训练成本 | 单进程约 114 s（GPU，18 epoch，含每轮全量 60k+10k 评估） |
| 产出权重 | `research/models/mnist_cnn_best_v3.bin`（99.38% 那一轮） |

单轮波动来自 batch 打乱的随机性（`std::random_device` 播种，CLI 无 `--seed`），所以**结论看均值**：
5 轮中 4 轮 ≥99.30，均值 99.34 ≥ 99.30。

---

## 3. 方法

### 3.1 两套训练制度

| 制度 | 命令 | 性质 | 用途 |
|---|---|---|---|
| 确定性 | `--shuffle-steps false` | 模型初始化（InitSpec 按创建序号定种）+ 样本顺序都固定 → **逐位可复现** | 无噪声 A/B（机制比较） |
| 真实 | 默认（打乱） | 每轮独立随机样本 → 需多轮取均值 | 最终指标判定 |

确定性已验证：同一配置两次运行的 `.bin` SHA256 完全相同（新旧二进制、以及 3~4 进程并发下都成立），
见 `results/focus_cnn/verify/`（含 `det_*` 校验痕迹与独立复验日志）。

### 3.2 评估口径

每轮训练后由 `evaluate_mnist` 对**全量 60000 训练集 + 全量 10000 测试集**评估；评估时 `model.set_training(false)`，
BatchNorm 使用 running 统计（`cli_mnist_io.hpp` 的 `TrainingGuard`）。`--eval-samples` 只对 ViT 生效，CNN 不受影响。

### 3.3 并发

单 GPU 下实测吞吐：1 进程 ≈3.9 s/epoch，4 进程并发 ≈12.4 s/epoch → 总吞吐约 1.3×。
因此扫描用 `--jobs 4`，但**判定用独立单进程复验**，避免并发引入的系统误差。

---

## 4. 过程与结果

### 4.1 阶段划分

| 阶段 | 内容 | run 数 | 状态 |
|---|---|---|---|
| A 广度确定性扫描 | CNN 宽度/深度/FC 头/卷积核·池化、lr、lr×尺寸、优化器、精度 | 41 完成 | 完成部分（见 4.2，其余组因目标变更未跑） |
| B 聚焦筛选 | 14 个候选 × shuffle=true × 18/30 epoch | 14 | 完成 |
| C 多轮确认 | 筛出 top5 各 3 轮 + 1 次独立单进程复验 | 16 | 完成 |

> 未执行（目标聚焦后主动砍掉）：weight decay / batch size / schedule / optimizer 完整网格、数据量 scale、
> 学习曲线 30~60 epoch、MLP 9 例、ViT 15 例、尺寸×数据/尺寸×归一化交互、shuffle=true 稳健性组。

### 4.2 阶段 A：确定性广度扫描（41 runs）

完整表：`research/results/REPORT_TABLES.md`（由 `research/scripts/analyze.py` 生成），
逐 run 明细：`research/results/all_runs.csv`，逐 epoch：`research/results/all_epochs.csv`。

> ⚠ 本阶段使用的是仓库自带的**过期 Debug 二进制**（CNN 默认 LayerNorm，无 `--norm-place`），
> 绝对值不能与当前源码直接对比，但机制性结论（容量饱和、lr 悬崖）与阶段 B/C 一致。

| 维度 | 结果 |
|---|---|
| 卷积宽度（FC 固定 128,10） | 4,8 = 98.93；8,16 = 98.97；**16,32 = 99.04**；32,64 = 99.08；64,128 = 98.98 |
| FC 头（卷积 16,32） | 32,10 = 98.82；64,10 = 99.03；128,10 = 99.04；256,10 = 99.08；256,128,10 = 99.07；512,256,10 = 99.21；**1024,512,10 = 99.22（体积 4.2 MB）** |
| 卷积深度 | 1 层（k5, ch32）= 98.61；**2 层 = 99.04**；3 层 k5 在 28×28 上不可行（第 3 层输入只剩 4×4） |
| 卷积核 / 池化 | k3 = 99.08；**k5 = 99.04**；k7 = 98.80；pool3 = 98.97；**去池化 = 97.55（严重过拟合，1.65 M 参数全砸在 FC）** |
| 学习率（base 16,32/256,128,10） | 1e-4 = 98.77；3e-4 = 99.03；1e-3 = 99.09；**3e-3 = 99.11**；5e-3 = 99.07；1e-2 = 99.07；**3e-2 = 11.35（发散，第 1 轮后崩）** |
| lr × 尺寸 | small@1e-2 = 98.93（小模型能吃大 lr）；**large@1e-3 = 99.37**；large@1e-2 = 11.35（发散）；large@3e-2 = 11.35（发散） |
| 精度 | f32 = 99.07；f16 = 99.12（差 0.05，噪声内） |
| 优化器 | 仅 SGD@1e-3 跑完 = 85.56（欠训练）；其余优化器网格未执行 |

### 4.3 阶段 B/C：聚焦筛选与多轮确认（shuffle=true）

脚本：`research/scripts/focus_cnn.py`；自动表：`research/results/focus_cnn/REPORT.md`；
逐 run CSV：`summary_screen.csv` / `summary_confirm.csv`；图：`candidates.svg`。

| 候选 | n | mean% | std | min | max | 模型KB | ≥99.3 |
|---|---|---|---|---|---|---|---|
| **`base_lr3e3`**（推荐） | 5 | **99.34** | 0.049 | 99.26 | 99.38 | 743 | ✅ |
| `bnhead_lr5e3`（BatchNorm@head） | 4 | 99.31 | 0.030 | 99.28 | 99.35 | 705 | ✅ |
| `base_e30`（同配置 30 epoch） | 1 | 99.30 | — | 99.30 | 99.30 | 743 | ✅ |
| `bnboth_lr5e3`（@both） | 4 | 99.30 | 0.052 | 99.24 | 99.36 | 749 | — |
| `med_lr3e3`（32,64/512,256,10） | 1 | 99.29 | — | 99.29 | 99.29 | 2865 | — |
| `lnconv_lr5e3`（LayerNorm@conv） | 1 | 99.28 | — | 99.28 | 99.28 | 721 | — |
| `base_lr5e3`（既有基线参数） | 4 | 99.28 | 0.052 | 99.23 | 99.33 | 743 | — |
| `fcbig_lr3e3`（FC 1024,512,10） | 4 | 99.27 | 0.049 | 99.22 | 99.33 | 4219 | — |
| `base_wd1e2`（wd 1e-2） | 1 | 99.23 | — | 99.23 | 99.23 | 743 | — |
| `med_e30` | 1 | 99.22 | — | 99.22 | 99.22 | 2865 | — |
| `none_lr5e3`（不加 norm） | 1 | 99.21 | — | 99.21 | 99.21 | 699 | — |
| `base_lr1e2`（lr 1e-2） | 1 | 99.18 | — | 99.18 | 99.18 | 743 | — |
| `deep3k3_lr3e3`（3 层卷积 k3） | 1 | 99.06 | — | 99.06 | 99.06 | 801 | — |

---

## 5. 结论与解释

1. **不需要更大的模型**。743 KB 的配置（99.34）优于 2.9 MB（99.29）与 4.2 MB（99.27）；3 层卷积反而 99.06。MNIST 上容量早已饱和，瓶颈是泛化而非拟合。
2. **lr 是最有效的旋钮**：3e-3（99.34）> 5e-3（99.28）> 1e-2（99.18）；确定性实验里 3e-2 直接发散，大模型 1e-2 就发散——**模型越大，可承受的 lr 越低**。
3. **18 epoch 足够**：同配置 30 epoch 只有 99.30（单轮），训练更久无收益；`weight-decay 1e-2` 明显有害（99.23，且确定性实验里更低）。
4. **归一化必须用 BatchNorm**：不加 norm 99.21、LayerNorm@conv 99.28；BatchNorm 的 `conv`（默认）/`head`/`both` 三者打平（99.28~99.34），差异在噪声内。**体积最小的达标变体**是 `BatchNorm@head`（705 KB，99.31±0.03）。
5. **与既有成果的关系**：`tests/README.md` 的全尺寸基线（同结构、lr 5e-3、单轮）为 99.33；本机同配置实测 99.28±0.05（n=4）。把 lr 降到 3e-3 得 99.34±0.05（n=5）——**+0.06 pp 属于弱证据**（小于轮间 std）。可靠的部分是「**743 KB 就能稳定到 99.3 级**」，不是「lr 3e-3 显著更优」。
6. **精度 f16 ≈ f32**（差 0.05），可放心用 `--f16` 省显存。

---

## 6. 复现

```bash
# ① 构建 Release（必须，见 §7.1）
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_TESTS=OFF
cmake --build build_rel

# ② 推荐配置单跑（约 2 分钟）
./build_rel/mnist_train.exe --arch cnn --gpu --epochs 18 --batch-size 128 \
  --optimizer adamw --weight-decay 0.001 --lr 0.003 \
  --lr-schedule cosine --warmup-epochs 3 --min-lr 1e-6 \
  --cnn-channels 16,32 --cnn-fc 256,128,10 --save mnist_cnn_best_v3.bin

# ③ 聚焦实验（筛选 → 多轮确认 → 表/图）
python research/scripts/focus_cnn.py --stage screen  --jobs 4
python research/scripts/focus_cnn.py --stage confirm --rounds 3 --top 5 --jobs 4
python research/scripts/focus_cnn.py --stage report
python research/scripts/focus_chart.py

# ④ 广度确定性扫描（未跑完的组可按需续跑）
python research/scripts/sweep.py --phases scale_conv,scale_fc,scale_depth,scale_kernel_pool,hp_lr,hp_lr_size,hp_opt,hp_wd,hp_bs,hp_sched --jobs 4
python research/scripts/sweep.py --rebuild-csv
python research/scripts/analyze.py > research/results/REPORT_TABLES.md
python research/scripts/charts.py
```

推理自检（已通过）：

```bash
./build_rel/mnist_infer.exe <图片.csv> --model research/models/mnist_cnn_best_v3.bin --topk 3
# 实测：test.csv 第 1 行真实标签 2 → 预测 2 (100.0%)
```

---

## 7. 踩坑与注意事项

1. **仓库自带的 `build/` 是过期产物**：`CMAKE_BUILD_TYPE=Debug`，且早于源码 HEAD `5d19e74`（NormPlace + BatchNorm 默认）。
   用它训练时 CNN 走 LayerNorm、`--norm-place` 直接报「未知参数」，同样结构只能到 99.1~99.3，且 BatchNorm@head/both 无法测试。
   **所以本次新建了 `build_rel/`；所有阶段 B/C 数字都来自它。**
2. **R5 240 不可用**：`--gpu 0` 是不具备完整 Vulkan 支持的 R5 240（约 50 s/epoch 且行为不可预期）；`--gpu 1` 报 DXIL 校验错误。
   `--gpu` 不带索引时自动选中 CMP 40HX（等价 `--gpu 2`），本次实验只用它。
3. **不要把「训练中」的半截日志当结果**：每完成 1 个 epoch 就会写日志，未跑完的 run 会被误读成「准确率崩到 98%」。
   `focus_cnn.py` / `sweep.py` 的统计已按「出现 `训练完成`」过滤。
4. **单轮差异 <0.1 pp 不可当结论**：打乱制度下轮间 std ≈0.05 pp；报告里凡未标 n 的行都是单轮。
5. **深度 ≥3 的卷积要换小核**：28×28 上 3×(k5+pool2) 会算到第 3 层输入 4×4 < 5 而构建失败（本次 `dp_3` 就是这么失败的，已改用 k3）。
6. GPU 是矿卡（CMP 40HX），`nvidia-smi` 显示 `clocks_throttle_reasons = 0x400`（功耗墙）；持续高负载时单 run 耗时波动较大，但不影响数值结果（确定性已验证）。

---

## 8. 目录清单

```
research/
├── README.md                       本文（目标 / 方法 / 过程 / 结论 / 复现 / 踩坑）
├── models/
│   ├── README.md                   权重走 GitHub Releases 的说明
│   └── mnist_cnn_best_v3.bin       推荐配置的最佳权重（99.38% 那一轮，743 KB，**不进 git**）
├── scripts/
│   ├── focus_cnn.py                聚焦实验驱动（screen / confirm / report，可断点续跑）
│   ├── focus_chart.py              候选对比 SVG
│   ├── sweep.py                    广度确定性扫描驱动（尺寸 × 超参）
│   ├── analyze.py                  广度扫描汇总表生成
│   └── charts.py                   零依赖 SVG 图表
└── results/
    ├── REPORT_TABLES.md            阶段 A 自动汇总表
    ├── all_runs.csv / all_epochs.csv   阶段 A 逐 run / 逐 epoch 明细（提交入库）
    ├── charts/                     阶段 A 图表（宽度/FC 头/lr/学习曲线）
    ├── <phase>/logs/*.log          阶段 A 各配置原始日志（本地；`*.log` 全域 gitignore）
    ├── focus_cnn/
    │   ├── REPORT.md               阶段 B/C 自动汇总表
    │   ├── summary_screen.csv / summary_confirm.csv
    │   ├── candidates.svg          候选均值对比图
    │   ├── screen|confirm/logs/    逐 run 原始日志（本地；不入库）
    │   └── verify/                 独立单进程复验（99.26%）与端到端推理样例
    └── （原始 .log/.bin 均可由上面脚本从零重跑生成）
```

**入库范围**：只有报告、脚本、汇总 CSV/表与 SVG。仓库 `.gitignore` 全域忽略 `*.bin`（模型）与 `*.log`（日志），
因此 79 个逐 run 模型、88 个逐 run 日志都不在仓库里：

- 权重：`research/models/mnist_cnn_best_v3.bin` 通过 **GitHub Releases** 单独发布（见 `models/README.md`）；
  其余逐 run 模型已清理，需要时用 `focus_cnn.py` / `sweep.py` 任意重跑（脚本幂等、可断点续跑）。
- 日志：逐 run 训练日志保留在本地；所有关键数值已汇总进 `all_runs.csv` / `all_epochs.csv` /
  `summary_*.csv`，这些是入库的原始证据。
- 过期 Debug 二进制下的早期筛选结果（30 个 run）已删除，其教训保留在 §7.1。

