# neuralnet.cpp (Vulkan) vs PyTorch (CUDA) GPU 基准报告

> 日期：2026-09-25 · 硬件：NVIDIA CMP 40HX (8 GB，本机唯一计算卡) · 全部数据来自本机实测
> 复现脚本：`bench/`（训练/算子/Layer 三方对照）· 原始日志与 JSON：`bench/raw/`

## 0. 结论速览

| 维度 | neuralnet.cpp (Vulkan) | PyTorch (CUDA) | 结论 |
|---|---|---|---|
| **训练速度**（双方全 f16，主对比） | **117.3 ms/step ≈ 139.6K tok/s** | 150.3 ms/step ≈ 109.0K tok/s | **nn 快 1.28×** |
| 训练速度（各自最优精度） | 116.3 ms/step ≈ 140.8K tok/s（f32，与 f16 持平） | **47.5 ms/step ≈ 344.2K tok/s（fp32）** | **torch 快 2.44×** |
| **显存占用**（nvidia-smi 峰值） | **5743–5867 MiB** | 1981 MiB (fp16) / 3313 MiB (fp32) | **nn 多用 ~2.9×** |
| 算子（f32 GEMM，大矩阵） | cuBLAS 的 47–65% | 基准 1.5–1.8× 领先 | torch 快 |
| 算子（f32 transpose，大矩阵） | **0.94 ms vs 3.48 ms @8192²** | 基准 | **nn 快 3.7×** |
| Layer（f32，fwd+bwd，large 组） | gpt_block 90.5 ms | 44.8 ms | torch 快 2.0× |
| 训练稳定性（lr=1e-3 实测） | **~100–2000 步发散 NaN（f16/f32 均发散）** | fp16/fp32 全程稳定 | torch 更稳 |

两个反直觉发现（§2.4、§5）：① torch 在 40HX 上该模型的 **fp16 路径比 fp32 慢 3.3×（病态）**，主对比中 nn 胜出主要源于此；② nn 的 **f16 对速度零收益**（f16≈f32≈117 ms），且与精度无关的训练发散值得跟进。

---

## 1. 环境与配置

| 项 | neuralnet.cpp | PyTorch |
|---|---|---|
| 版本 | 本工作区 build（Release，`-O3 -march=native`，AOT 融合 shader） | 2.14.0+cu132（`.venv/`） |
| 后端 | Vulkan GpuEngine（`--gpu=40HX` / `NN_VULKAN_DEVICE=40HX`） | CUDA (cuBLAS/cuDNN) |
| 精度（主对比） | 全 f16：`--precision-param/compute/stable/optimizer f16` | 全 fp16：`model.half()` |
| 对照精度 | f32（默认） | fp32（默认） |
| 优化器 | Adam，lr=1e-3，β=(0.9,0.999)，eps=1e-8（fp32 数学） | Adam，同超参；**fp16 时 eps=1e-4**（见 §5.2） |

**模型（按要求配置）**：GPT，d_model=64，heads=4，layers=4，d_ff=256，seq_len=256，batch=64。
参数量：nn **1,275,280**（vocab 8208）/ torch **1,273,216**（vocab 8192）。

**数据**：`datasets/tinystories_small.txt` 子集
- 短测 `tinystories_bench40.txt`（4000 行）：nn 3097 窗口 → **49 步/epoch**；torch 3070 窗口 → **48 步/epoch**。每步工作量相同（64×256 位置），吞吐按 tok/s 归一。
- 长测 `tinystories_bench.txt`（40000 行）：521 步/epoch，仅用于 nn 交叉验证（§2.2）。

**词表 8192**：C++ `tokenizer_train --vocab-size 8192` 实际产出 **8208**（BBPE 固定含 256 个字节 token 与特殊 token 的计数口径，请求值与产出值差 0.2%）；torch 侧 HF BPE 实际 **8192**。两侧分词器各自独立（token 流不同、每步形状完全相同），速度用 tok/s 归一，不影响可比性。

**数据管线（逐位镜像）**：每行独立 tokenize 后纯拼接（无 BOS/EOS）；窗口 `pos=0,256,…`，`x[t]=flow[pos+t]`，`y[t]=flow[pos+t+1]` 且仅 `t+1 < win_len=min(seq, len-pos)` 参与 loss；每 epoch shuffle 窗口；epoch 边界同步 GPU（nn 在 epoch 末 drain loss 回读，torch 在 epoch 末 `synchronize`），epoch 内 host（构 batch + H2D）与 GPU 重叠。

---

## 2. 训练速度与显存（端到端）

### 2.1 主表（短测，稳态 = epoch 2–3 平均；epoch 1 含初始化/时钟爬坡）

| 配置 | ms/step | tok/s | 显存峰 (nvidia-smi) | torch allocator peak | loss 曲线 |
|---|---|---|---|---|---|
| **nn Vulkan 全 f16**（主对比） | **117.3** | **139,620** | **5772 MiB** | — | e1 7.40 → **e2 起 -nan** |
| nn Vulkan f32（对照） | 116.3 | 140,845 | 5743 MiB | — | e1 7.38 → **e2 起 -nan** |
| **torch CUDA 全 fp16**（主对比） | **150.3** | **109,000** | **1981 MiB** | 1406.7 MB | 7.57→6.71→6.31 稳定 |
| torch CUDA fp32（对照） | **47.5** | **344,213** | 3313 MiB | 2794.8 MB | 7.40→5.85→5.15 最稳 |

主对比（双方全 f16）：**nn 吞吐 1.28× 于 torch，显存 2.9× 于 torch**。

### 2.2 长测交叉验证（nn 全 f16，40000 行，521 步/epoch，epoch 1–4 健康）

`61.1 / 61.5 / 61.7 / 61.9 s` → **117.3–118.8 ms/step，137.7–140.1K tok/s**，与短测一致（±1%）。
显存峰 **5867 MiB**（与短测 5772 一致）。**epoch 5（第 2084 步）发散 -nan**——与短测现象一致（§2.5）。

### 2.3 显存构成观察

模型本体极小（参数 ~1.27M × f16 ≈ 2.5 MB + 梯度/Adam 状态 ~10 MB；logits 64×256×8208 f16 ≈ 269 MB）。
- **nn 峰值 ~5.7–5.9 GB 与精度无关**（f16 5772 / f32 5743）→ 由 **Vulkan 内存池分配粒度 + 驱动开销**主导，而非张量本身。
- torch 峰值随精度变化（fp16 1981 / fp32 3313 MiB），allocator 视角 1407 / 2795 MB，符合理论张量构成。

### 2.4 torch 单步分解（probe 实测，50 轮 best）

| 段 | 耗时 |
|---|---|
| host 构 batch（CPU） | 0.12 ms |
| H2D（含构 batch） | 0.59 ms |
| forward（无 loss） | **47.64 ms** |
| forward + CE loss | 49.64 ms（CE 仅 +2.0 ms） |
| forward + loss + backward | **150.02 ms**（→ backward ≈ 100 ms） |
| Adam step | 1.77 ms |
| zero_grad | 0.12 ms |
| full step（host 异步入队） | 32.3 ms enqueue ↔ GPU 实耗 150.96 ms（与 epoch 实测吻合） |

**关键反常**：同一 torch 模型、同一卡，全模型 forward **fp32 = 14.75 ms vs fp16 = 47.96 ms（fp16 慢 3.25×）**。
该尺寸（d=64 瘦 GEMM + vocab 头）在 40HX 上 torch 2.14 的 fp16 kernel 选择病态（未进一步剖析）。
因此：**torch 侧该配置的自然最优是 fp32（344K tok/s），fp16 是其最差路径**；nn 侧 f16/f32 完全持平（117 ms），f16 路径无净收益。

### 2.5 训练稳定性（如实记录）

- **nn**：bench40 上无论 f16/f32 都在 **epoch 2（约 98 步）发散 -nan**；bench40K 上 f16 撑到 2084 步。发散与精度无关（f32 同样发散），同 lr 下 torch 稳定——提示 nn 侧存在数值/训练稳定性问题（超出本次基准范围，建议后续跟进）。
- **torch fp16**：默认 `eps=1e-8` 在 fp16 中表示为 0，稀疏 embedding 零梯度行 `0/0=NaN`，**必须 `eps=1e-4`** 后稳定（实测 3 epoch 平稳下降）。
- **nn 侧一个实现事实**：`--precision-optimizer f16` 在 Adam 路径**不改变**状态精度——`create_zero_buffers_()` → `create_tensor(rows, cols)` 默认 `Precision::F32`（`compute_optimizer.hpp:65`、`compute_engine.hpp:210`），Adam m/v 恒为 f32，eps 亦以 fp32 RParam 进 kernel。`precision.hpp` 注释亦注明"f16 训练下必须 f32"。故 nn "全 f16" 的实际语义 = 参数/激活/梯度/loss 为 f16 + **Adam 状态 f32**；torch 对应语义 = 参数/状态全 fp16（更激进）+ eps 调整。

---

## 3. 算子对比（torch CUDA vs neuralnet.cpp Vulkan）

**口径**：全部 **f32**（`layer_bench` 的 Matrix=Scalar=float，torch 侧同为 `float32`）；统一尺寸；每配置 **3 轮进程 × 每轮 warmup 20 + best-of-50**，取 3 轮最优。
- nn = `layer_bench --op`，**wall-clock（含提交/等待固定开销 ≈0.13 ms/次）**——小形状被系统性抬高。
- torch = CUDA event 纯 kernel 计时。
- 记账：GFLOPS=2·m·n·k；GB/s 按读+写实际字节（归约 = N + 输出，不虚记 2N）。
- **已知不公平项均标注**；两者皆非 Python 延迟（大矩阵 + CUDA event 已稀释框架开销）。

### 3.1 GEMM（ms，nn / torch / 比值）

| 形状 | matmul nn | torch | 比值 | matmul_bt nn | torch | 比值 | matmul_at nn | torch | 比值 |
|---|---|---|---|---|---|---|---|---|---|
| 1024³ | 0.848 | 0.504 | 1.68 | 0.889 | 0.509 | 1.75 | 0.642 | 0.414 | 1.55 |
| 2048³ | 4.130 | 2.384 | 1.73 | 4.372 | 2.578 | 1.70 | 3.910 | 2.382 | 1.64 |
| 4096³ | 31.821 | 17.857 | 1.78 | 33.737 | 19.483 | 1.73 | 30.188 | 17.700 | 1.71 |
| LM-head (16384,8208)·(64) | 4.534 | 2.937 | 1.54 | 4.641 | 3.046 | 1.52 | 4.316 | 2.824 | 1.53 |

峰值吞吐（同轮）：4096³ matmul **nn 4319 GFLOPS（≈40HX FP32 理论 9.14 TF 的 47%）vs torch 7697（84%）**；1024³ 2532 vs 4262；LM-head 3796 vs 5861。

### 3.2 batched_matmul

| 形状 | nn | torch | 比值 | nn GFLOPS | torch GFLOPS |
|---|---|---|---|---|---|
| B=64, 256³ | 0.811 ms | 0.380 ms | 2.13 | 2649 | 5656 |
| B=512, 64³ | 0.229 ms | 0.141 ms | 1.62 | 1172 | 1900 |
| B=4096, 64³ | 0.857 ms | 0.547 ms | 1.57 | 2505 | 3927 |

### 3.3 逐元素 / 归约 / 转置（ms，nn / torch / 比值；比值 <1 = nn 更快）

| 算子 | 1024² | 2048² | 4096² | 8192² |
|---|---|---|---|---|
| add_inplace | 0.154/0.104/1.48 | 0.238/0.163/1.46 | 0.593/0.507/1.17 | 1.967/1.901/1.03 |
| elementwise_exp | 0.145/0.109/1.33 | 0.204/0.129/1.58 | 0.448/0.369/1.21 | 1.399/1.328/1.05 |
| broadcast_col | 0.130/0.101/1.29 | 0.197/0.127/1.55 | 0.448/0.365/1.23 | 1.387/1.307/1.06 |
| row_reduce_sum | 0.133/0.137/0.97 | 0.164/0.122/1.34 | 0.281/0.211/1.33 | 0.705/0.645/1.09 |
| col_reduce_sum | 0.133/0.160/**0.83** | 0.161/0.164/0.98 | 0.282/0.255/1.11 | 0.724/0.673/1.08 |
| transpose | 0.135/0.139/0.97 | 0.176/0.231/**0.76** | 0.342/0.692/**0.49** | 0.940/3.483/**0.27** |
| scale_inplace | 0.125/0.111/1.13 | 0.194/0.135/1.44 | 0.449/0.374/1.20 | 1.395/1.330/1.05 |

**观察**
1. **GEMM：torch 全面领先 1.5–2.1×**（cuBLAS 吃到 84% FP32 峰值 vs nn 自研 shader 47%）。
2. **8192² 大形状下带宽型算子双双收敛到 ~380–420 GB/s**（40HX 理论 448 GB/s），差距缩到 3–9%——瓶颈是显存而非实现。
3. **transpose 是 nn 的强项**：8192² 快 **3.7×**（0.94 vs 3.48 ms，571 vs 154 GB/s）、4096² 快 2×（砖块化 64×64 tile + 8×8 砖块 dispatch 的收益；torch `.t().contiguous()` 走通用拷贝路径）。
4. **1024² 小形状**：nn 被 ~0.13 ms 提交固定开销托底（因此 row/col_reduce、transpose 反而持平或略优）；2048² 起固定开销占比下降，torch 的带宽优势显现。
5. 归约记账已按 N+out 精确口径（此前 2N 口径会虚高 ~2×，见 AGENTS bench 方法论）。

---

## 4. Layer 对比（`layer_bench --layer` vs torch 同构层，f32）

**口径**：3 轮 × warmup 20 + best-of-50，取最优；形状两侧严格一致。
- model 组 = 被测模型配置（d=64/heads4/dff256/seq256/batch64，T=16384）
- large 组 = 大尺寸（d=512/heads8/dff2048/seq1024/batch8，T=8192）
- nn 为 wall-clock（含固定开销），torch 为 CUDA event；**ms 是主指标**。
- GFLOPS 公式两侧同源（`flops_attn = 6d²T + 4T²d`，T=batch·seq 会高估注意力绝对值，**比值仍公平**）；softmax 的 nn tok/s 口径无意义，不引用。

### 4.1 model 组（fwd / train，ms）

| 层 | nn fwd | torch fwd | 比值 | nn train | torch train | 比值 |
|---|---|---|---|---|---|---|
| linear (64→256, 16384 tok) | 0.312 | 0.333 | 0.94 | 2.435 | 0.725 | 3.36 |
| layernorm | 1.085 | 0.218 | 4.98 | 2.514 | 0.539 | 4.66 |
| softmax | 0.187 | 0.122 | 1.53 | 0.381 | 0.494 | **0.77** |
| mha | 7.441 | 1.972 | 3.77 | 22.541 | 4.756 | 4.74 |
| causal_attn | 6.061 | 2.310 | 2.62 | 21.594 | 5.031 | 4.29 |
| feedforward | 0.865 | 0.700 | 1.24 | 5.312 | 1.880 | 2.83 |
| gpt_block (pre-norm) | 9.513 | 3.176 | 3.00 | 32.612 | 7.391 | 4.41 |
| transformer (post-norm) | 11.196 | 2.876 | 3.89 | 33.711 | 7.045 | 4.79 |

### 4.2 large 组（fwd / train，ms）

| 层 | nn fwd | torch fwd | 比值 | nn train | torch train | 比值 |
|---|---|---|---|---|---|---|
| linear (512→2048, 8192 tok) | 4.205 | 2.623 | 1.60 | 13.475 | 5.192 | 2.60 |
| layernorm | 1.328 | 0.180 | 7.38 | 3.025 | 0.549 | 5.51 |
| softmax | 0.313 | 0.142 | 2.20 | 0.651 | 0.467 | 1.39 |
| mha | 36.629 | 8.516 | 4.30 | 66.067 | 22.371 | 2.95 |
| causal_attn | 25.280 | 10.287 | 2.46 | 55.078 | 24.070 | 2.29 |
| feedforward | 8.996 | 6.183 | 1.46 | 27.974 | 15.410 | 1.82 |
| gpt_block | 37.899 | 16.858 | 2.25 | 90.451 | 44.769 | 2.02 |
| transformer | 49.108 | 15.135 | 3.24 | 101.878 | 41.007 | 2.49 |

**观察**
1. **f32 层级 torch 全面领先 1.4–7.4×**，与 §3 的 GEMM 结论一致；尺寸越小，nn 的固定提交开销占比越高（model 组 train 比值 2.8–4.8，large 组收敛到 1.8–2.6）。
2. nn 的 softmax train 反超（0.77×）——小 kernel + torch autograd 开销所致。
3. **层微基准（f32）与端到端（f16）排序矛盾的解释**：nn 端到端 ≈ 层分解量级（4×gpt_block fp32 ≈ 130 ms → f16/流水后 118 ms 合理）；torch 端到端被 §2.4 的 **fp16 病态路径**拖到 150 ms，其 fp32 端到端（47.5 ms/step）才与层分解一致（4×7.4 + lm_head + CE + Adam ≈ 45–50 ms ✓）。

---

## 5. 方法论与已知偏差清单

1. **计时口径**：nn `layer_bench` = wall-clock，含 Vulkan 提交/等待固定开销（≈0.13 ms/iter，已由 `NN_GPU_PROFILE` 剖面确认）；torch = CUDA event 纯 kernel。小形状对 nn 系统性不利，大形状（≫0.13 ms）可忽略。训练侧双方均为 epoch 边界同步的 wall-clock，口径一致。
2. **torch fp16 eps**：fp16 下 `eps=1e-8` 下溢为 0 → 必须 `eps=1e-4`（nn 侧 eps 以 fp32 进 kernel 保持 1e-8）。仅影响极小梯度项的更新幅度，不影响速度；两侧 g=0 行均为 0 更新。
3. **"全 f16" 实际语义差**：nn 的 Adam m/v 实为 f32（`create_tensor` 默认），torch fp16 的 m/v 为 fp16——torch 状态显存减半但数值更脆（靠 eps 兜底）。
4. **词表**：C++ 8208（请求 8192 的 BBPE 计数口径）vs torch 8192，LM-head/CE 差 0.2% FLOPs，可忽略。
5. **文档掩码**：nn 训练启用 per-line doc mask（fold 内生效），torch 为标准 causal——FLOPs 相同，掩码语义不同（速度无影响）。
6. **算子/Layer 为 f32**：`layer_bench` 无精度开关（Matrix=f32），故 §3/§4 是 f32↔f32 对比；f16 仅在 §2 端到端。
7. **发散不影响计时**：NaN 不改变 kernel 执行路径；nn 发散 epoch 的耗时与健康 epoch 一致（5.7–5.9 s），且与长测交叉验证。
8. **轮次纪律**：所有点 warmup≥20、≥3 轮取最优（防时钟爬坡/跨会话漂移，见 AGENTS bench 方法论）；训练取稳态 epoch 平均并以长测交叉验证。
9. **本机噪声**：40HX 为专用计算卡（跑测时显存 0 占用），显示输出走 AMD R5 240。

---

## 6. 产物索引

**脚本（`bench/`）**
- `train_torch.py` — torch 训练 e2e（分词镜像 C++、epoch 计时、显存统计、`--adam-eps`、`--dtype`）
- `ops_torch.py` / `layers_torch.py` — torch 算子/层基准（CUDA event、多轮 best）
- `probe_torch_step.py` — torch 单步分段分解（§2.4）
- `run_with_vram.ps1` — 任意进程 + nvidia-smi 200 ms 采样显存峰值
- `run_ops.ps1` / `run_layers.ps1` — 双侧统一尺寸编排（3 轮）

**原始数据（`bench/raw/`）**
- 训练：`nn_train.log`（长测）、`nn_short.log`、`nn_short_f32.log`、`torch_short.log`、`torch_short_fp32.log`，对应 `*.vram`（显存采样）、`torch_train*.json`（逐 epoch 结构化）
- 算子：`ops_nn.log`（8 形状组 ×3 轮）、`ops_torch.log` / `ops_torch.json`（43 case）
- 层：`layers_nn.log`、`layers_torch_model.log` / `layers_torch_large.log` + `.json`
- 语料/词表：`datasets/tinystories_bench40.txt`、`datasets/bench_bpe_8192.json`（C++ 8208）、`bench/raw/torch_bpe_8192.json`（torch 8192）

**复现命令**（仓库根目录）：
```powershell
# 训练（nn / torch）
.\build\text_train.exe datasets\tinystories_bench40.txt --vocab datasets\bench_bpe_8192.json `
  --d-model 64 --num-heads 4 --num-layers 4 --d-ff 256 --seq-len 256 --batch-size 64 `
  --epochs 3 --optimizer adam --lr 0.001 --gpu=40HX `
  --precision-param f16 --precision-compute f16 --precision-stable f16 --precision-optimizer f16 `
  --save-interval 0 --log-interval 100000
.\venv\Scripts\python.exe bench\train_torch.py datasets\tinystories_bench40.txt `
  --vocab bench\raw\torch_bpe_8192.json --vocab-size 8192 --epochs 3 --dtype fp16 --adam-eps 1e-4

# 算子 / 层
.\bench\run_ops.ps1 -Side nn     # 或 -Side torch
.\bench\run_layers.ps1 -Side nn  # 或 -Side torch
```
