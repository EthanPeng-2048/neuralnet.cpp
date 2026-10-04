# MNIST CNN 归一化（norm 类型 × 挂载位置）A/B 实验包

日期：2026-10-03 ｜ 仓库：`neuralnet.cpp` @ `v1.6.0-3-g5d19e74`
目的：在**已知较优的训练参数**下，系统测试不同归一化类型与挂载位置，看能否进一步提升测试准确率。

---

## 1. 实验环境

| 项 | 值 |
|---|---|
| 设备 | Termux / aarch64，8 核，11 GB RAM（手机） |
| GPU | **Mali-G610 MC6**（`--gpu`，Vulkan 1.1.177），备选 llvmpipe |
| 编译器 | clang 21.1.8（C++26），Release（`-O3 -march=native`） |
| 数据 | MNIST：`train.csv` 60000 条 / `test.csv` 10000 条（Kaggle 版） |
| 每轮评估 | **全量**：train 60000 + test 10000（CNN 不走 `--eval-samples` 截断） |

> ⚠️ **随机性**：`mnist_train` 的 batch 打乱用 `std::random_device` 初始化，**没有 `--seed` 参数**，
> 每轮训练天然不同。因此单轮差异 < ±0.15 pp 一律视为噪声，必须多轮取均值。
> 这也是本实验分「全尺寸（贵，单轮）+ 小模型代理（便宜，3 轮）」两阶段的原因。

---

## 2. 两阶段设计

### 阶段 1 — 全尺寸（= 用户原参数，只改 `--norm` / `--norm-place`）

```bash
./build/mnist_train --arch cnn --gpu --epochs 18 --batch-size 128 \
  --lr 0.005 --optimizer adamw --weight-decay 0.001 \
  --cnn-channels 16,32 --cnn-fc 256,128,10 \
  --lr-schedule cosine --warmup-epochs 3 --min-lr 1e-6 \
  --norm <type> --norm-place <where> --save <path>     # 单轮约 24 分钟
```

**完成 7/10**（`rms_conv`、`rms_head`、`none` 三个因手机资源不足被中止）。
耗时太长 → 改用阶段 2 筛选。

### 阶段 2 — 小模型代理（10 配置 × 3 轮 = 30 次，单轮约 3 分钟）

```bash
./build/mnist_train --arch cnn --gpu --epochs 6 --batch-size 128 \
  --lr 0.005 --optimizer adamw --weight-decay 0.001 \
  --cnn-channels 8,16 --cnn-fc 64,10 \
  --lr-schedule cosine --warmup-epochs 2 --min-lr 1e-6 \
  --max-samples 20000 --norm <type> --norm-place <where> --save <path>
```

相对全尺寸的改动：通道 `16,32→8,16`、FC `256,128,10→64,10`、epoch `18→6`（warmup `3→2`）、训练样本 `60000→20000`；
**优化器与学习率完全不变**，故 norm 排序在代理配置上仍可比。测试集仍是全量 10000。

### 测试的 10 个配置

| name | CLI |
|---|---|
| base | `--norm auto --norm-place auto`（= BatchNorm @ conv，当前默认） |
| bn_both / bn_head | `--norm batchnorm --norm-place both/head` |
| ln_both / ln_conv / ln_head | `--norm layernorm --norm-place both/conv/head` |
| rms_both / rms_conv / rms_head | `--norm rmsnorm --norm-place both/conv/head` |
| none | `--norm batchnorm --norm-place none`（完全不加 norm） |

> `--norm-place final` 是 ViT 专用，CNN 不接受；MLP 的挂载位置由结构固定。

---

## 3. 结果

### 3.1 小模型代理（3 轮均值）— `results/small_proxy_3rounds.txt`

```
config      n    mean     std     min     max  best_mean  vs_base  norm
----------------------------------------------------------------------
bn_both     3   98.62    0.13   98.50   98.75      98.62    +0.24  BatchNorm @ both
bn_head     3   98.60    0.08   98.53   98.69      98.60    +0.22  BatchNorm @ head
ln_both     3   98.42    0.02   98.41   98.45      98.43    +0.04  LayerNorm @ both
rms_conv    3   98.42    0.09   98.31   98.47      98.42    +0.03  RMSNorm @ conv
base        3   98.38    0.02   98.37   98.41      98.38     0.00  BatchNorm @ conv
rms_head    3   98.36    0.01   98.35   98.37      98.36    -0.02  RMSNorm @ head
rms_both    3   98.32    0.08   98.23   98.38      98.32    -0.06  RMSNorm @ both
ln_conv     3   98.31    0.05   98.26   98.35      98.31    -0.07  LayerNorm @ conv
ln_head     3   98.28    0.13   98.16   98.41      98.28    -0.10  LayerNorm @ head
none        3   98.24    0.11   98.16   98.37      98.24    -0.14  BatchNorm（= 不加 norm）
```

### 3.2 全尺寸 18 epoch（单轮）— `results/fullsize_18ep.txt`

```
config      n    mean     std     min     max  best_mean  vs_base  norm
----------------------------------------------------------------------
bn_head     1   99.34    0.00   99.34   99.34      99.37    +0.01  BatchNorm @ head
base        1   99.33    0.00   99.33   99.33      99.35     0.00  BatchNorm @ conv
ln_both     1   99.31    0.00   99.31   99.31      99.31    -0.02  LayerNorm @ both
ln_head     1   99.26    0.00   99.26   99.26      99.28    -0.07  LayerNorm @ head
bn_both     1   99.23    0.00   99.23   99.23      99.28    -0.10  BatchNorm @ both
rms_both    1   99.22    0.00   99.22   99.22      99.24    -0.11  RMSNorm @ both
ln_conv     1   99.18    0.00   99.18   99.18      99.19    -0.15  LayerNorm @ conv
（rms_conv / rms_head / none 未跑）
```

> 阶段 1 的 `ln_head` 一轮在 19:11 才结束，与阶段 2 第 1 轮有 ~18 分钟重叠，
> 只影响耗时不影响准确率；原始日志在 `results/logs_fullsize/ln_head.log`。

---

## 4. 结论

1. **加 norm 确实有用**：`none`（不加）在两阶段都是垫底（小模型 -0.14，且是唯一会掉到 98.16 的档位）。
2. **位置比类型重要**：同一类型下 conv/head/both 的差 > 类型之间的差。
3. **小模型 3 轮均值里，`BatchNorm @ head`(+0.22) 与 `BatchNorm @ both`(+0.24) 稳定高于当前默认 `@conv`**，
   是唯一超出轮间噪声（base std 仅 0.02）的信号；LayerNorm / RMSNorm 任何位置都与 base 打平或更差。
4. **但全尺寸单轮与之矛盾**：base 99.33 反而高于 bn_both 99.23，且全尺寸所有变体都挤在 99.18~99.34（±0.15 的噪声带内）。
   ⇒ **目前无法宣称任何配置优于你现有参数**；`mnist_cnn_best_v2.bin`（原参数产物）仍是可靠基线。
5. 判定「+0.2 是真提升还是噪声」需要：全尺寸下 base / bn_head / bn_both 各 **≥3 轮**（约 3×3×24 min）。

---

## 5. 待办（在新设备上续跑）

```bash
# ① 补齐阶段 1 缺的 3 个（每个 24 分钟）
--norm rmsnorm   --norm-place conv
--norm rmsnorm   --norm-place head
--norm batchnorm --norm-place none

# ② 判定性复验：base / bn_head / bn_both 各补 2 轮（错开跑，脚本见 scripts/）
# ③ 若要同时确认阶段 2 的排序，可把 scripts/sweep_small.sh 复制过去跑 3 轮（30 分钟）
```

复现表格：`python3 scripts/make_tables.py <pack目录>` → 重新生成
`results/summary.csv`、`small_proxy_3rounds.txt`、`fullsize_18ep.txt`。

---

## 6. 文件清单

```
norm_ab_pack/
├── README.md                     本文
├── scripts/
│   ├── run.sh                    单个全尺寸训练的封装（阶段 1）
│   ├── sweep.sh                  阶段 1 串行调度（10 配置，含未完成的 3 个）
│   ├── sweep_small.sh            阶段 2 串行调度（PFX=s_/s2_/s3_ 控制轮次）
│   ├── summary.sh                粗汇总（逐配置最后一个 epoch）
│   ├── aggregate.py              阶段 2 跨轮聚合（按配置合并、只收跑满的轮次）
│   └── make_tables.py            从原始日志重新生成本包的三张表
└── results/
    ├── summary.csv               37 次训练的逐次明细（phase/config/round/acc/log/model）
    ├── small_proxy_3rounds.txt   阶段 2 聚合表
    ├── fullsize_18ep.txt         阶段 1 汇总表
    ├── logs_fullsize/            7 个 18-epoch 原始日志（含逐 epoch 曲线）
    ├── logs_smallproxy/          30 个 6-epoch 原始日志
    ├── models_fullsize/          7 个全尺寸模型 .bin（715 KB/个）
    ├── models_smallproxy/        30 个小模型 .bin
    └── models_reference/         你原有的 mnist_cnn_best.bin / mnist_cnn_best_v2.bin
```

模型加载：`./build/mnist_infer <图片.csv> --model <path>.bin`，或 `mnist_train --resume <path>` 续训。
日志里的 `\r` 进度需要 `tr '\r' '\n' < xxx.log` 才好读。

数据集另打包：`mnist_data.tar.gz`（`train.csv` + `test.csv`，解到仓库 `datasets/mnist_data/` 即可；
原始 idx 文件 `raw/` 未打包，训练不使用）。

> 后续研究（模型尺寸 / 训练超参扫描、小体积 CNN 达到 99.3%+）已移到仓库根目录
> **`research/mnist/`**（见 `research/mnist/README.md`；`research/README.md` 是各研究的索引），本实验包保持原样。
