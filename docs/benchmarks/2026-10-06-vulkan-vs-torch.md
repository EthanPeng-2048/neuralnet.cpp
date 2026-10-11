# neuralnet.cpp (Vulkan) vs PyTorch (CUDA) 端到端训练基准（2026-10-06）

> 日期：2026-10-06 · 硬件：NVIDIA CMP 40HX (8 GB，本机唯一计算卡) · **全部数据本会话实测**
> **数据版本**：nn 与 torch 两侧**同日、同机、同卡、同语料**采集，无跨会话漂移。
> 与上一份报告 [`2026-09-25-vulkan-vs-cuda.md`](2026-09-25-vulkan-vs-cuda.md) 的关系见 §5：那份报告的语料未入库、无法复现，本轮语料为重建；其 09-25/09-26 数值在 §5 中作为对照列。
> 本轮 nn 侧包含工作区未提交的三项 host 侧优化（staging 重复上传、`import` 快路径、内存池单扫描），见 §4.3 与 `docs/history.md`。

## 0. 结论速览

| 维度 | neuralnet.cpp (Vulkan) | PyTorch (CUDA) | 结论 |
|---|---|---|---|
| **训练速度**（f32 ↔ fp32，**主对比**） | **107.1 ms/step ≈ 152.9K tok/s** | **50.2 ms/step ≈ 326.6K tok/s** | **torch 快 2.14×** |
| 训练速度（双方各自可用的低精度） | 118.4 ms/step ≈ 138.4K tok/s（`--f16`，健康） | 30.4 ms/step ≈ 538.2K tok/s（fp16，稳定） | torch 快 3.89× |
| 训练速度（双方全 f16 ⚠） | 107.1 ms/step ≈ 152.9K tok/s ⚠ loss = `-nan` | 30.4 ms/step ≈ 538.2K tok/s | 仅 nn 侧计时有效，**训练无效** |
| **显存占用**（nvidia-smi 峰值） | **f32 3141 / `--f16` 2113 / 全 f16 1526 MiB** | fp32 3313 / fp16 1981 MiB | f32：**nn 更省（0.95×）**；低精度：nn 多 6~7% |
| 训练稳定性（lr=1e-3，3 epoch） | **f32 与 `--f16` 均健康收敛**；全 f16 epoch1 起 `-nan` | fp32 / fp16 全程稳定 | f32 打平；nn 全 f16 不可用 |
| GPU 利用率（同配置 20 步 profile） | **busy 89.8%**（设备 kernel 1972.6 ms / 墙钟 2197.8 ms） | — | nn 已接近设备饱和 |

**两个关键发现**：

1. **torch 在该模型上的 fp16"病态慢"不复现**（§3）。09-25 报告记录 fp16 比 fp32 慢 3.16×（150.3 vs 47.5 ms/step；单步 forward 口径为 3.25×）；本轮同脚本、同卡、同 torch 版本实测 fp16 = 30.4 ms/step，**比 fp32 快 1.65×**，单步 forward 从 47.96 ms 降到 8.55 ms。该差异未能二分（旧运行环境不可复现），标注为待查项。
2. **nn 与 torch 的 f32 差距由 device kernel 效率决定，而不是 host 开销**（§4）。nn 侧 GPU busy 已达 89.8%，`fused_mm` + `fused` 两个 kernel 族占每步 87% 的设备时间；继续压 host 侧固定开销的收益空间有限。

---

## 1. 环境与配置

| 项 | neuralnet.cpp | PyTorch |
|---|---|---|
| 版本 | 本工作区 Release（clang、`-O3 -march=native`、AOT 融合 shader），HEAD `652c53d` + 工作区未提交优化 | 2.14.0+cu132（`.venv/`） |
| 后端 | Vulkan GpuEngine（`--gpu=40HX`） | CUDA (cuBLAS/cuDNN) |
| 精度（**主对比**） | **f32（默认）** | **fp32（默认）** |
| 对照精度 | `--f16` = {param:f16, compute:f16, stable:f32, optimizer:f32}（健康）；全 f16 = 四字段全 f16（⚠ 本轮 `-nan`） | fp16 = `model.half()`，`adam eps=1e-4` |
| 优化器 | Adam，lr=1e-3，β=(0.9,0.999)，eps=1e-8 | 同超参 |

**模型（两侧同构）**：GPT，d_model=64，heads=4，layers=4，d_ff=256，seq_len=256，batch=64。
参数量：nn **1,275,280**（vocab 8208）/ torch **1,273,216**（vocab 8192），差 0.16%。

**数据（本轮重建）**：09-25 报告使用的 `datasets/tinystories_bench40.txt` 不在仓库内，本轮按同口径重建——取 `datasets/tinystories_200mb/tinystories_small.txt` **前 4000 条非空行**（3,381,238 字节），`dataset_gen` 记录的 `source.sha256 = 584422376434996e…`。

- nn：BBPE 训练词表 8208 → 789,828 tokens → 滑窗 3086 窗口 → **49 步/epoch**
- torch：沿用 `bench/raw/torch_bpe_8192.json`（HF BPE 8192）→ 785,820 tokens → 3070 窗口 → **48 步/epoch**

两侧分词器独立、token 流不同，**每步形状完全相同（64×256 = 16,384 位置）**，速度按 tok/s 归一。

**数据管线（与 09-25 报告同一口径）**：每行独立 tokenize 后拼接；窗口 `pos = 0,256,…`，`x[t]=flow[pos+t]`，`y[t]=flow[pos+t+1]` 且仅 `t+1 < win_len` 参与 loss；每 epoch 打乱窗口；epoch 边界同步 GPU（nn drain loss 回读 / torch `synchronize`），epoch 内 host 构 batch 与 H2D 和 GPU 重叠。

> **注**：nn 侧现已改为统一数据集格式，训练入口只接受 `.nndataset`（`tokenizer_train` → `dataset_gen` 两步生成），不再直接读 txt + vocab——09-25 报告 §6 的 nn 复现命令已失效，本轮复现命令见 §7。

---

## 2. 端到端训练（主表）

**口径**：3 epoch，**稳态 = epoch 2–3 平均**（epoch 1 含初始化与时钟爬坡）；显存统一由 `bench/run_with_vram.ps1` 以 200 ms 采样 nvidia-smi 整卡峰值；每步 16,384 个位置。

| 配置 | 稳态 s/epoch | ms/step | tok/s | 显存峰 (nvidia-smi) | loss 曲线（e1→e2→e3） |
|---|---:|---:|---:|---:|---|
| **nn Vulkan f32**（**主对比**） | 5.25 | **107.1** | **152,917** | **3141 MiB** | 7.335 → 5.868 → 5.171 健康 |
| nn Vulkan `--f16`（推荐低精度 profile） | 5.80 | 118.4 | 138,417 | 2113 MiB | 7.337 → 5.929 → 5.293 健康 |
| nn Vulkan 全 f16（⚠） | 5.25 | 107.1 | 152,917 | 1526 MiB | **`-nan` ×3（训练无效）** |
| **torch CUDA fp32**（**主对比**） | 2.41 | **50.2** | **326,593** | **3313 MiB** | 7.383 → 5.897 → 5.342 最稳 |
| torch CUDA fp16 | 1.46 | 30.4 | 538,219 | 1981 MiB | 7.532 → 6.687 → 6.303 稳定 |

**读数**：

- **主对比（f32 ↔ fp32）**：torch 吞吐 **2.14×** 于 nn（326,593 / 152,917 = 2.136）；**显存 nn 更省**（3141 vs 3313 = 0.95×）。
- **可用低精度对比**：nn `--f16` 138,417 vs torch fp16 538,219 → torch **3.89×**；显存 nn 2113 vs torch 1981 → nn 多 6.7%。
- **双方全 f16**：nn 本轮 epoch1 起即 `-nan`（09-25 为 epoch2 发散、09-26 为打印冻结——三种表现都属"不可用"，见 14 号报告）；torch fp16 正常。该行**只有 nn 的计时与显存有效**。
- 两侧 f32 loss 曲线几乎重合（nn 7.335→5.171 / torch 7.383→5.342），可交叉验证负载等价。
- 本表 5 行均来自本轮实测，**不含跨会话数据**。

---

## 3. torch 单步分解与 fp16 复核

`bench/probe_torch_step.py`（50 轮 best，CUDA event，fp16/half 路径，与 09-25 报告 §2.4 同一脚本）：

| 段 | 09-25 实测 | **2026-10-06 实测** | 变化 |
|---|---:|---:|---:|
| host 构 batch（CPU wall） | 0.12 ms | **0.125 ms** | 持平 |
| H2D（含构 batch） | 0.59 ms | **0.507 ms** | 持平 |
| forward（无 loss） | 47.64 ms | **8.554 ms** | **−82%** |
| forward + CE loss | 49.64 ms | **11.490 ms** | **−77%** |
| forward + loss + backward | 150.02 ms | **26.604 ms** | **−82%** |
| Adam step | 1.77 ms | 1.678 ms | 持平 |
| zero_grad | 0.12 ms | 0.112 ms | 持平 |
| full step（host wall / 双侧 sync） | 32.3 / 150.96 ms | **29.017 / 28.797 ms** | 见下 |

**结论：09-25 记录的"torch fp16 病态（比 fp32 慢 3.3×）"在本轮不复现。**

- 端到端口径：fp16 30.4 ms/step vs fp32 50.2 ms/step → **fp16 快 1.65×**（旧记录为 fp16 慢 3.16×）。
- 单步口径：`fwd+loss+bwd` 150.02 → 26.604 ms，forward 47.64 → 8.554 ms。
- 未变化的部分（host 构 batch、H2D、Adam、zero_grad）说明**脚本与管线未变**，差异集中在 GPU kernel 执行本身。
- torch 版本两侧均为 2.14.0+cu132、同一张 40HX，因此差异更可能来自**驱动 / 设备时钟 / 运行时环境**的变化；旧运行环境不可复现，**未做二分**，本条标注为待查项，不作为"旧报告错误"的结论。
- 影响：旧报告中"torch 该配置的自然最优是 fp32、fp16 是其最差路径"的判断**本轮不成立**；fp16 现为 torch 侧更快的路径。

---

## 4. neuralnet 训练 step 剖析（`NN_PROFILE=1`）

与 §2 完全同配置（d64 / 4 层 / heads4 / d_ff256 / seq256 / batch64，同一 `.nndataset`，`--gpu=40HX`），`--max-steps 20`。

产物：`build/prof_bench_20261006.txt`、`build/trace_bench_20261006.json`。

### 4.1 阶段构成（inclusive 墙钟）

墙钟合计 **2197.68 ms / 20 步 = 109.9 ms/step**（e2e 实测 107.1 ms/step，profile 开销 +2.6%）。

| stage | ms | %wall |
|---|---:|---:|
| forward | 1875.89 | 85.4% |
| └ fwd.blk#0 | 1191.53 | 54.2% |
| └ fwd.blk#1 | 118.46 | 5.4% |
| └ fwd.blk#2 | 118.40 | 5.4% |
| └ fwd.blk#3 | 115.79 | 5.3% |
| backward | 119.42 | 5.4% |
| optimizer | 110.33 | 5.0% |
| sync | 80.69 | 3.7% |
| data | 7.53 | 0.3% |
| submit-fwd | 3.76 | 0.2% |

### 4.2 原语净归属（exclusive）与设备侧

| op | ms | calls | avg µs | %wall |
|---|---:|---:|---:|---:|
| `copy_from` | 1720.72 | 121 | 14220.8 | 78.3% |
| `eval_expr` | 180.44 | 7800 | 23.1 | 8.2% |
| `release_idle_pool_blocks` | 76.38 | 20 | 3819.0 | 3.5% |
| `eval_expr_into` | 36.45 | 3020 | 12.1 | 1.7% |
| `zero` | 35.00 | 1664 | 21.0 | 1.6% |
| `end_batch` | 27.40 | 161 | 170.2 | 1.2% |
| `eval_expr_reduce` | 24.94 | 1720 | 14.5 | 1.1% |
| `create_tensor` | 23.87 | 421 | 56.7 | 1.1% |
| `flush_batch` | 21.93 | 140 | 156.6 | 1.0% |
| `import` | 11.54 | 68128 | 0.2 | 0.5% |

设备侧（timestamp query）：**GPU busy = 89.8%**（设备 kernel 1972.59 ms / 墙钟 2197.75 ms）

| kernel | ms | calls | avg µs |
|---|---:|---:|---:|
| `fused_mm` | 1211.18 | 1830 | 661.8 |
| `fused` | 704.15 | 10197 | 69.1 |
| `rearrange_3d` | 20.79 | 936 | 22.2 |
| `scatter_add` | 19.35 | 38 | 509.2 |
| `transpose` | 16.57 | 471 | 35.2 |
| `gather` | 0.54 | 40 | 13.6 |

### 4.3 读数与下一步方向

1. **`copy_from` 的 78.3% 不能读作"PCIe 传输占 78%"**。op 表是 host 侧 exclusive 计时，**包含阻塞等待**：GPU busy 已达 89.8%，设备接近饱和，host 调用点一旦等设备就把时间记在自己头上。证据是 `fwd.blk#0` 的 `copy_from` 57 ms/次（1139.92/20），而该处每步的输入张量（x/y/loss_mask，64×256 量级）只有 KB 级——57 ms 传输 KB 级数据不合 PCIe 带宽物理量级，实为**承接上一步的设备排队等待**。
2. **瓶颈在设备侧**：`fused_mm` + `fused` 合计 1915.3 ms / 20 步 = 95.8 ms/step，占 109.9 ms/step 的 **87%**；每步约 602 次 dispatch（`fused` 510 + `fused_mm` 91.5）。这与上一份报告 §3 的算子级结论一致：nn GEMM 达 cuBLAS 的 54–67%。
3. **host 侧优化的收益空间已收窄**：本轮已落地的三项优化把 staging 重复上传（`copy_from` 调用 201→121）、`import` NVI 开销（12.26→10.78 ms）与内存池释放扫描（39.88→34.76 ms）压掉一部分；但相对 89.8% busy 的设备墙，继续压 host 固定开销难以改变 §2 的 2.14× 差距。
4. **下一杠杆应落在 `fused_mm`/`fused` 的设备执行效率与 dispatch 次数**（减少每步 kernel 数、提升 GEMM 占用率），而非再做 host 侧微优化。`release_idle_pool_blocks` 在本配置下 76.38 ms（3.5%）仍是可观的每步固定成本，可作为次优先项。

> §4 的 profile 配置为 **d64/batch64**，与上一份报告 §2.6 使用的 **d128/batch8** profile **不是同一负载**，两组毫秒数不可直接比较；各自的 A/B 只在同配置内有效。

---

## 5. 与 2026-09-25/26 报告的差异

| 指标 | 09-25 / 09-26 | **2026-10-06** | 判定 |
|---|---:|---:|---|
| nn f32 ms/step | 101.0 | 107.1 | +6.0%，落在 ±15% 跨会话漂移内 |
| torch fp32 ms/step | 47.5 | 50.2 | +5.7%，同上 |
| **f32 吞吐比值** | **2.13×** | **2.14×** | **复现** |
| torch 显存 fp32 / fp16 | 3313 / 1981 MiB | 3313 / 1981 MiB | **逐字复现** |
| nn f32 显存 | 3069 MiB | 3141 MiB | +2.3% |
| nn f32 loss 收敛 | 7.38 → 5.93 → 5.28 | 7.335 → 5.868 → 5.171 | 复现（语料重建后仍健康） |
| torch fp32 loss 收敛 | 7.40 → 5.85 → 5.15 | 7.383 → 5.897 → 5.342 | 复现 |
| **torch fp16 ms/step** | **150.3（比 fp32 慢 3.16×）** | **30.4（比 fp32 快 1.65×）** | **不复现，见 §3** |
| torch fp16 forward | 47.64 ms | 8.55 ms | **不复现** |
| nn 全 f16 | 比 f32 慢 17%、loss 打印冻结 | 与 f32 持平、loss `-nan` | 表现变化，**两者都不可用** |

**不复现项的处理**：本报告如实记录新值，并保留旧值供追溯；未对旧结论做"勘误"，因为差异未二分（旧环境不可复现）。`docs/development/14-f16-stable-gpu-loss-frozen.md` 记录的 nn 全 f16 缺陷仍成立，且本轮新增一种表现形态（epoch1 起 `-nan`）。

---

## 6. 方法论与已知偏差

1. **同窗对照**：本轮 nn 与 torch 在同一会话、同一张卡、同一份语料上采集，规避了上一份报告 ±15% 的跨会话漂移问题；§5 中与旧值的比较仍含该漂移。
2. **计时口径**：两侧均为训练主循环 wall-clock，epoch 边界同步 GPU；epoch 内 host（构 batch / H2D）与 GPU 重叠，不做逐步 synchronize。稳态取 epoch 2–3 平均。
3. **轮次纪律**：3 epoch（与上一份报告一致）；epoch 1 只作 warmup 不进稳态；torch 侧 fp32 两个独立进程复跑得 325,911 / 326,593 tok/s（±0.2%），可复现性良好。
4. **词表差异**：C++ BBPE 8208 vs torch HF BPE 8192，两侧分词器独立、token 流不同，LM-head FLOPs 差 0.16%；速度按 tok/s 归一，不影响可比性。
5. **掩码语义**：nn 训练启用 per-line 文档掩码（fold 内生效），torch 为标准 causal——FLOPs 相同，语义不同，速度无影响。
6. **显存口径**：统一 `nvidia-smi` 200 ms 采样整卡峰值（含 CUDA context）；torch allocator 视角另记 `peak_allocated` / `peak_reserved`（fp32 2794.8 / 3355.4 MB，fp16 1406.7 / 1956.6 MB）。
7. **profile 口径**：`NN_PROFILE=1` 默认关闭零开销，开启后同配置 20 步墙钟比 e2e 高 2.6%（109.9 vs 107.1 ms/step），属观测开销；插桩的数值无关性由 `gpu_stability_probe` 剖析开/关 hash 逐位一致验证（`docs/history.md`「训练 step 剖析器」）。
8. **本机噪声**：40HX 为专用计算卡（跑测时显存 0 占用），显示输出走 AMD R5 240。

---

## 7. 产物索引与复现命令

**本轮原始数据（`build/`）**

- nn f32：`nn_short_f32_20261006.log`（+ `.vram` / `.peak`）
- nn `--f16`：`nn_short_f16rec_20261006.log`（+ `.vram` / `.peak`）
- nn 全 f16：`nn_short_f16_20261006.log`（+ `.vram` / `.peak`，⚠ loss `-nan`）
- torch：`torch_short_fp32_20261006.log` / `torch_short_fp16_20261006.log`（+ `.peak`），`torch_train_fp32_20261006.json` / `torch_train_fp16_20261006.json`
- step 剖析：`prof_bench_20261006.txt`、`trace_bench_20261006.json`
- 数据集（重建）：`datasets/tinystories_bench40.txt`、`build/bench_bpe_8192.nnvocab`、`build/tinystories_bench40.nndataset`

**复现命令**（仓库根目录）：

```powershell
# 0) 重建语料（4000 行）—— 原 09-25 语料未入库
#    取 tinystories_small.txt 前 4000 条非空行写入 datasets\tinystories_bench40.txt

# 1) nn 侧：词表 → 统一数据集 → 训练
.\build\tokenizer_train.exe datasets\tinystories_bench40.txt --vocab-size 8192 --output build\bench_bpe_8192.nnvocab
.\build\dataset_gen.exe datasets\tinystories_bench40.txt --vocab build\bench_bpe_8192.nnvocab -o build\tinystories_bench40.nndataset

#    f32 主对比（含 200ms 显存采样）
.\bench\run_with_vram.ps1 -Exe .\build\text_train.exe -Log build\nn_short_f32_20261006.log -Arguments @(
  'build\tinystories_bench40.nndataset','--d-model','64','--num-heads','4','--num-layers','4',
  '--d-ff','256','--seq-len','256','--batch-size','64','--epochs','3',
  '--optimizer','adam','--lr','0.001','--gpu=40HX',
  '--save-interval','0','--save','build\gpt_bench_nn.bin','--log-interval','100000')

#    --f16 推荐低精度：同上，追加 '--f16'
#    全 f16（⚠ loss -nan，仅计时/显存有效）：追加
#      '--precision-param','f16','--precision-compute','f16','--precision-stable','f16','--precision-optimizer','f16'

# 2) torch 侧（同一语料，沿用 09-25 的 HF BPE 词表）
$py = '.\.venv\Scripts\python.exe'
$common = @('bench\train_torch.py','datasets\tinystories_bench40.txt','--vocab','bench\raw\torch_bpe_8192.json','--vocab-size','8192','--epochs','3')
.\bench\run_with_vram.ps1 -Exe $py -Log build\torch_short_fp32_20261006.log -Arguments ($common + @('--dtype','fp32','--out','build\torch_train_fp32_20261006.json'))
.\bench\run_with_vram.ps1 -Exe $py -Log build\torch_short_fp16_20261006.log -Arguments ($common + @('--dtype','fp16','--adam-eps','1e-4','--out','build\torch_train_fp16_20261006.json'))

# 3) torch 单步分解（§3）
.\.venv\Scripts\python.exe bench\probe_torch_step.py

# 4) nn step 剖析（§4，与 e2e 同配置）
$env:NN_PROFILE='1'; $env:NN_PROFILE_TRACE='build\trace_bench_20261006.json'
.\build\text_train.exe build\tinystories_bench40.nndataset --d-model 64 --num-heads 4 --num-layers 4 `
  --d-ff 256 --seq-len 256 --batch-size 64 --epochs 1 --max-steps 20 `
  --optimizer adam --lr 0.001 --gpu=40HX --save-interval 0 `
  --save build\gpt_prof_nn.bin --log-interval 100000 *> build\prof_bench_20261006.txt
Remove-Item Env:NN_PROFILE,Env:NN_PROFILE_TRACE
```
