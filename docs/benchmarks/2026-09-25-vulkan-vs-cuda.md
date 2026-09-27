# neuralnet.cpp (Vulkan) vs PyTorch (CUDA) GPU 基准报告

> 日期：2026-09-25 · 硬件：NVIDIA CMP 40HX (8 GB，本机唯一计算卡) · 全部数据来自本机实测
> **数据版本**：nn 侧（§2 训练 / §3 算子 / §4 Layer）采集于 HEAD `4a12876`（2026-09-26 会话）；torch 侧采集于 2026-09-25。
> 两侧跨会话比值含 ±15% 系统漂移（AGENTS bench 方法论），单看 nn 侧不同采集日之间的差异时该漂移同样适用。
> 复现脚本：`bench/`（训练/算子/Layer 三方对照，日志解析 `bench/parse_bench.ps1`）· 原始日志与 JSON：`bench/raw/`

## 0. 结论速览

| 维度 | neuralnet.cpp (Vulkan) | PyTorch (CUDA) | 结论 |
|---|---|---|---|
| **训练速度**（f32 ↔ fp32，**主对比**） | **101.0 ms/step ≈ 162.2K tok/s** | 47.5 ms/step ≈ 344.2K tok/s | **torch 快 2.13×** |
| 训练速度（双方全 f16 ⚠） | 118.4 ms/step ≈ 138.4K tok/s ⚠loss 冻结 | 150.3 ms/step ≈ 109.0K tok/s | nn 吞吐快 1.27×（nn 侧训练有效性未验证） |
| **显存占用**（nvidia-smi 峰值） | **f32 3069–3076 / f16 5738–6315 MiB** | 1981 MiB (fp16) / 3313 MiB (fp32) | f32：**nn 反超（0.93×，更省）**；f16：nn 多用 ~3× |
| 算子（f32 GEMM，大矩阵） | cuBLAS 的 54–67% | 基准 1.5–1.9× 领先 | torch 快 |
| 算子（f32 transpose，大矩阵） | **1.495 ms vs 3.483 ms @8192²** | 基准 | **nn 快 2.3×**（本表口径，见 §3.3‡） |
| Layer（f32，fwd+bwd，large 组） | gpt_block 95.2 ms | 44.8 ms | torch 快 2.1× |
| 训练稳定性（lr=1e-3 实测） | **f32 5 epoch 健康收敛（09-26 采集；09-25 采集为 NaN 发散）**；f16+`stable=f16` **loss 打印冻结**（见 §2.5 与 14 号报告） | fp16/fp32 全程稳定 | f32 打平；f16 存在缺陷 |

三个反直觉发现（§2.4、§2.5、§3.3）：① torch 在 40HX 上该模型的 **fp16 路径比 fp32 慢 3.3×（病态）**；② nn 的 **f16 比 f32 慢 17%**（118.4 vs 101.0 ms；09-25 采集为 f16≈f32≈117 ms），且 `stable=f16` 下 **loss 打印冻结而权重照常更新**（2026-09-26 发现，`docs/development/14-f16-stable-gpu-loss-frozen.md`）；③ nn f32 显存 **3069 MiB**（09-25 采集 5743，差 −47%）——峰值强相关于精度，见 §2.3。

---

## 1. 环境与配置

| 项 | neuralnet.cpp | PyTorch |
|---|---|---|
| 版本 | 本工作区 build（Release，`-O3 -march=native`，AOT 融合 shader，HEAD `4a12876`） | 2.14.0+cu132（`.venv/`） |
| 后端 | Vulkan GpuEngine（`--gpu=40HX` / `NN_VULKAN_DEVICE=40HX`） | CUDA (cuBLAS/cuDNN) |
| 精度（**主对比**） | **f32（默认）** | **fp32（默认）** |
| 对照精度 | 全 f16：`--precision-param/compute/stable/optimizer f16`（⚠ 实验性，loss 冻结见 §2.5）；推荐 `--f16` = {param:f16, compute:f16, stable:f32, optimizer:f32} 实测健康 | fp16：`model.half()` |
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

### 2.1 主表（短测，稳态 = epoch 2–3 平均；epoch 1 含初始化/时钟爬坡；nn 列采集于 09-26）

| 配置 | ms/step | tok/s | 显存峰 (nvidia-smi) | torch allocator peak | loss 曲线 |
|---|---|---|---|---|---|
| **nn Vulkan f32**（**主对比**） | **101.0** | **162,185** | **3069 MiB** | — | e1 7.38 → e2 5.93 → e3 5.28 健康下降 |
| nn Vulkan 全 f16（⚠ loss 冻结） | 118.4 | 138,417 | 6315 MiB | — | 打印恒 4.0137（**不可信**，§2.5 / 14 号报告） |
| **torch CUDA fp32**（**主对比**） | **47.5** | **344,213** | 3313 MiB | 2794.8 MB | 7.40→5.85→5.15 最稳 |
| torch CUDA 全 fp16 | 150.3 | 109,000 | 1981 MiB | 1406.7 MB | 7.57→6.71→6.31 稳定 |

- **主对比（f32 ↔ fp32）**：**torch 吞吐 2.13× 于 nn**（按 09-25 采集的 nn f32 116.3 ms/step 计为 2.44×；本表 nn f32 口径 = 101.0 ms/step），**显存 nn 反而更省（3069 vs 3313 MiB，0.93×）**。
- 参考（双方全 f16）：nn 吞吐 1.27× 于 torch、显存 3.19× 于 torch——但 nn 侧 loss 冻结，**该行仅计时/显存有效**。
- 09-25 采集值（提交 `75255da` 前）：f16 117.3 / f32 116.3 ms/step，显存 5772/5743 MiB，两者 loss 均在 epoch2 发散 -nan。

### 2.2 长测交叉验证（40000 行，521 步/epoch，09-26 采集）

- **nn f32**：`52.4 / 52.6 / 52.7 / 52.7 / 52.7 s` → **100.6–101.1 ms/step，162.0–162.9K tok/s**，与短测一致（±0.5%）。显存峰 **3076 MiB**（短测 3069，±0.2%）。**avg_loss 4.62 → 2.76 五轮健康收敛，全程无 NaN**（09-25 采集的同类长测在 epoch 5 出现 -nan，见 §2.5）。
- **nn 全 f16**：`59.9 / 59.8 / 59.8 / 59.9 / 59.8 s` → **114.8–115.0 ms/step，142.5–142.7K tok/s**，显存峰 **5738 MiB**。**5 个 epoch 的 avg_loss 逐位相同（4.0182 ×5）= 冻结**——计时有效、loss 无效（§2.5 / 14 号报告）。
- 09-25 采集值（f16）：`61.1–61.9 s` → 117.3–118.8 ms/step，显存 5867 MiB，epoch 5 发散 -nan。

### 2.3 显存构成观察（09-26 采集口径）

模型本体极小（参数 ~1.27M × f16 ≈ 2.5 MB + 梯度/Adam 状态 ~10 MB；logits 64×256×8208 f16 ≈ 269 MB）。
- **峰值强相关于精度**：f32 3069/3076（短/长）、f16 6315/5738；09-25 采集值为 f32 5743 / f16 5772（两者接近）。差值：f32 −47%、f16 短测 +9%。
- f32 峰值跨短/长测稳定在 3069–3076（±0.2%），可复现性好；f16 短/长差 9%（6315 vs 5738，200 ms 采样粒度 + 分配时序敏感，见 05 文档 §12 的池粒度讨论）。
- torch 峰值随精度变化（fp16 1981 / fp32 3313 MiB），allocator 视角 1407 / 2795 MB，符合理论张量构成（torch 侧数据采集于 09-25）。

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
因此：**torch 侧该配置的自然最优是 fp32（344K tok/s），fp16 是其最差路径**；nn 侧 09-25 采集为 f16/f32 持平（均 ≈117 ms），09-26 采集为 f32 101.0 / f16 118.4 ms——f16 路径相对 f32 无收益（见 §2.1）。

### 2.5 训练稳定性（按采集日期如实记录）

- **nn f32（09-26 采集）**：短测 3 epoch 与长测 5 epoch **全程健康收敛**（短测 avg 7.38→5.93→5.28；长测 4.62→2.76），无 NaN。09-25 采集记录为「f16/f32 均 ~100–2000 步发散 NaN」；两者的差异归于 09-25 之后的提交窗口，未逐个二分。
- **nn f16 + `stable=f16`（09-26 采集到的缺陷）**：loss **打印冻结**——step 5–45 恒 4.0137、跨 epoch/跨数据集/跨进程逐位相同；同时**权重照常更新**（同进程跨 epoch 模型快照 63% 参数字节不同）。定性矩阵：触发器 = `stable=f16`（GPU 特有，CPU 同配置健康）；`--f16` 推荐 profile（stable/optimizer=f32）实测健康；`optimizer=f16` 单独即爆炸（= 下条已知数值限制）。**测试全绿与该缺陷并存**（覆盖缺口：GPU+stable=f16 无自动化用例）。详见 `docs/development/14-f16-stable-gpu-loss-frozen.md`。
- **nn 四字段全 f16 的缺陷表现（按采集日期）**：09-25 采集为「epoch1 活 → epoch2 发散 -nan」（= 05 文档 §12.5「stable=F16 时 CE 链 ~200 步 NaN」）；09-26 采集为「epoch1 起即静默冻结」——冻结是静默出错（NaN 反而会暴露），风险等级更高。
- **torch fp16**：默认 `eps=1e-8` 在 fp16 中表示为 0，稀疏 embedding 零梯度行 `0/0=NaN`，**必须 `eps=1e-4`** 后稳定（实测 3 epoch 平稳下降）。
- **nn optimizer=f16 语义（本报告口径，以代码为准）**：`compute_optimizer.hpp:51-53` 注释明确状态张量按 `p_.optimizer` 创建（默认 F32；全 f16 profile 按字面取 F16，"用户显式选择全 f16 时不予阻拦，只如实报告数值表现"）。09-25 采集条目记的「`--precision-optimizer f16` 不改变 Adam 状态精度（m/v 恒 f32）」与当前代码不符，以本条为准。定性矩阵 case c（optimizer=f16, stable=f32）loss 爆炸至 1.8e7 即为该语义的实测表现（与 05 文档 §12.5「optimizer=F16 时 Adam 的 v≈g²~1e-10 下溢→更新爆炸」一致）。故 nn `--f16`（推荐 profile）实际语义 = 参数/激活 f16 + stable/optimizer **f32**；四字段全 f16 = 含爆炸/冻结双重风险的实验性组合。

---

## 3. 算子对比（torch CUDA vs neuralnet.cpp Vulkan）

**口径**：全部 **f32**（`layer_bench` 的 Matrix=Scalar=float，torch 侧同为 `float32`）；统一尺寸；每配置 **3 轮进程 × 每轮 warmup 20 + best-of-50**，取 3 轮最优。
- nn = `layer_bench --op`，**wall-clock（含提交/等待固定开销 ≈0.13 ms/次）**——小形状被系统性抬高。**nn 列采集于 2026-09-26，torch 列采集于 09-25**（比值含跨会话漂移，见 §5.10）。
- torch = CUDA event 纯 kernel 计时。
- 记账：GFLOPS=2·m·n·k；GB/s 按读+写**精确字节**计（归约 = N + 输出）。
- **已知不公平项均标注**；两者皆非 Python 延迟（大矩阵 + CUDA event 已稀释框架开销）。

### 3.1 GEMM（ms，nn / torch / 比值）

| 形状 | matmul nn | torch | 比值 | matmul_bt nn | torch | 比值 | matmul_at nn | torch | 比值 |
|---|---|---|---|---|---|---|---|---|---|
| 1024³ | 0.853 | 0.504 | 1.69 | 0.760 | 0.509 | 1.49 | 0.661 | 0.414 | 1.60 |
| 2048³ | 4.324 | 2.384 | 1.81 | 4.565 | 2.578 | 1.77 | 4.114 | 2.382 | 1.73 |
| 4096³ | 33.316 | 17.857 | 1.87 | 35.637 | 19.483 | 1.83 | 31.563 | 17.700 | 1.78 |
| LM-head (16384,8208)·(64) | 4.722 | 2.937 | 1.61 | 4.880 | 3.046 | 1.60 | 4.526 | 2.824 | 1.60 |

峰值吞吐（同轮）：4096³ matmul **nn 4125 GFLOPS（≈40HX FP32 理论 9.14 TF 的 45%）vs torch 7697（84%）**；1024³ 2518 vs 4262；LM-head 3645 vs 5861。nn 达 torch GFLOPS 的 54–67%（12 个 GEMM 点全量核算）。

### 3.2 batched_matmul

| 形状 | nn | torch | 比值 | nn GFLOPS | torch GFLOPS |
|---|---|---|---|---|---|
| B=64, 256³ | 0.820 ms | 0.380 ms | 2.16 | 2618 | 5656 |
| B=512, 64³ | 0.222 ms | 0.141 ms | 1.57 | 1210 | 1900 |
| B=4096, 64³ | 0.853 ms | 0.547 ms | 1.56 | 2517 | 3927 |

### 3.3 逐元素 / 归约 / 转置（ms，nn / torch / 比值；比值 <1 = nn 更快）

| 算子 | 1024² | 2048² | 4096² | 8192² |
|---|---|---|---|---|
| add_inplace | 0.146/0.104/1.40 | 0.235/0.163/1.44 | 0.622/0.507/1.23 | 2.060/1.901/1.08 |
| elementwise_exp † | 0.145/0.109/1.33 | 0.204/0.129/1.58 | 0.448/0.369/1.21 | 1.399/1.328/1.05 |
| broadcast_col † | 0.130/0.101/1.29 | 0.197/0.127/1.55 | 0.448/0.365/1.23 | 1.387/1.307/1.06 |
| row_reduce_sum | 0.130/0.137/0.95 | 0.158/0.122/1.30 | 0.283/0.211/1.34 | 0.721/0.645/1.12 |
| col_reduce_sum | 0.115/0.160/0.72 | 0.160/0.164/0.98 | 0.286/0.255/1.12 | 0.760/0.673/1.13 |
| transpose | 0.141/0.139/1.01 | 0.205/0.231/0.89 | 0.481/0.692/0.70 | 1.495/3.483/0.43 |
| scale_inplace | 0.119/0.111/1.07 | 0.192/0.135/1.42 | 0.465/0.374/1.24 | 1.474/1.330/1.11 |

> † `elementwise_exp` / `broadcast_col`：本行数值为 **09-25 采集口径**；当前 `layer_bench` 算子表不含这两项（引擎不提供逐元素/广播原语，逐元素一律走 DSL `eval_expr`），故沿用 09-25 读数（当前工具无法复现该点）。
> ‡ **transpose 记账口径**：本表数值按「派发网格与 shader 行界契约同源」口径采集（缺陷记录 `200955f`）。09-25 采集的 4096² 0.342 / 8192² 0.940 ms 为契约不同源状态下的读数（派发网格与 shader 契约不同源 → 行>512 静默半写，kernel 实际少做一半工作）：当时 8192² 折算 571 GB/s 超出 40HX 理论峰值 448，本表口径为 359 GB/s ≈ 峰值 80%。按本表口径 transpose 的 nn 优势 = **2.3×**（09-25 采集口径为 3.7×）。

**观察**
1. **GEMM：torch 全面领先 1.5–1.9×**（cuBLAS 吃到 84% FP32 峰值 vs nn 自研 shader 45%）。
2. **8192² 大形状下带宽型算子收敛到 nn ~350–390 GB/s vs torch ~380–420 GB/s**（40HX 理论 448 GB/s），差距 8–13%——瓶颈是显存而非实现。
3. **transpose 仍是 nn 的强项**：8192² 快 **2.3×**（1.495 vs 3.483 ms，359 vs 154 GB/s）、4096² 快 1.4×（0.481 vs 0.692）；口径与早先采集值（3.7×/2×）的差异见 ‡。
4. **1024² 小形状**：nn 被 ~0.13 ms 提交固定开销托底（因此 row/col_reduce、transpose 反而持平或略优）；2048² 起固定开销占比下降，torch 的带宽优势显现。
5. 归约记账按 N+out **精确字节**口径（见 AGENTS bench 方法论）。

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
| linear (64→256, 16384 tok) | 0.306 | 0.333 | 0.92 | 2.544 | 0.725 | 3.51 |
| layernorm | 1.031 | 0.218 | 4.73 | 2.358 | 0.539 | 4.37 |
| softmax | 0.179 | 0.122 | 1.47 | 0.371 | 0.494 | **0.75** |
| mha | 7.740 | 1.972 | 3.92 | 23.224 | 4.756 | 4.88 |
| causal_attn | 6.094 | 2.310 | 2.64 | 22.851 | 5.031 | 4.54 |
| feedforward | 0.834 | 0.700 | 1.19 | 5.702 | 1.880 | 3.03 |
| gpt_block (pre-norm) | 9.542 | 3.176 | 3.00 | 33.989 | 7.391 | 4.60 |
| transformer (post-norm) | 11.318 | 2.876 | 3.94 | 35.063 | 7.045 | 4.98 |

### 4.2 large 组（fwd / train，ms）

| 层 | nn fwd | torch fwd | 比值 | nn train | torch train | 比值 |
|---|---|---|---|---|---|---|
| linear (512→2048, 8192 tok) | 4.456 | 2.623 | 1.70 | 14.173 | 5.192 | 2.73 |
| layernorm | 1.273 | 0.180 | 7.07 | 2.954 | 0.549 | 5.38 |
| softmax | 0.315 | 0.142 | 2.22 | 0.656 | 0.467 | 1.40 |
| mha | 38.705 | 8.516 | 4.54 | 69.919 | 22.371 | 3.13 |
| causal_attn | 26.726 | 10.287 | 2.60 | 57.666 | 24.070 | 2.40 |
| feedforward | 9.486 | 6.183 | 1.53 | 29.204 | 15.410 | 1.90 |
| gpt_block | 39.796 | 16.858 | 2.36 | 95.191 | 44.769 | 2.13 |
| transformer | 51.858 | 15.135 | 3.43 | 107.178 | 41.007 | 2.61 |

**观察**
1. **f32 层级 torch 全面领先 1.4–7.1×**，与 §3 的 GEMM 结论一致；尺寸越小，nn 的固定提交开销占比越高（model 组 train 比值 3.0–5.0，large 组除 layernorm 外收敛到 1.9–3.1）。
2. nn 的 softmax train 反超（0.75×）——小 kernel + torch autograd 开销所致。
3. **层微基准与端到端的关系**：nn 层分解 4×gpt_block train ≈ 4×34.0 = 136 ms，而 f32 端到端仅 101.0 ms/step——**端到端低于简单加和约 26%**（09-25 采集对应值 130→118 ms，差 9%）。解释：`layer_bench` 每迭代同步/分配开销 + 无跨步流水，e2e 走中点 flush + 异步 loss 回读 + host/GPU 跨步重叠（§1 数据管线）；两个采集日之间的 e2e（−13%）与层基准（gpt_block 32.6→34.0）差异均在 ±15% 会话漂移内。torch 端到端被 §2.4 的 **fp16 病态路径**拖到 150 ms，其 fp32 端到端（47.5 ms/step）与层分解一致（4×7.4 + lm_head + CE + Adam ≈ 45–50 ms ✓）。

---

## 5. 方法论与已知偏差清单

1. **计时口径**：nn `layer_bench` = wall-clock，含 Vulkan 提交/等待固定开销（≈0.13 ms/iter，已由 `NN_GPU_PROFILE` 剖面确认）；torch = CUDA event 纯 kernel。小形状对 nn 系统性不利，大形状（≫0.13 ms）可忽略。训练侧双方均为 epoch 边界同步的 wall-clock，口径一致。
2. **torch fp16 eps**：fp16 下 `eps=1e-8` 下溢为 0 → 必须 `eps=1e-4`（nn 侧 eps 以 fp32 进 kernel 保持 1e-8）。仅影响极小梯度项的更新幅度，不影响速度；两侧 g=0 行均为 0 更新。
3. **"全 f16" 语义（本报告口径，以代码为准）**：nn 的 Adam m/v 按 `p_.optimizer` 字面创建（`compute_optimizer.hpp:51-53`；`--f16` 推荐 profile 下为 f32，四字段全 f16 下为 f16 且实测爆炸）；torch fp16 的 m/v 为 fp16（靠 eps 兜底）。09-25 采集条目记的「nn m/v 恒 f32」与当前代码不符，以本条为准（详见 §2.5）。
4. **词表**：C++ 8208（请求 8192 的 BBPE 计数口径）vs torch 8192，LM-head/CE 差 0.2% FLOPs，可忽略。
5. **文档掩码**：nn 训练启用 per-line doc mask（fold 内生效），torch 为标准 causal——FLOPs 相同，掩码语义不同（速度无影响）。
6. **算子/Layer 为 f32**：`layer_bench` 无精度开关（Matrix=f32），故 §3/§4 是 f32↔f32 对比；f16 仅在 §2 端到端。
7. **异常值不影响计时**：NaN 与 loss 打印冻结都不改变 kernel 执行路径；nn 异常 epoch 的耗时与健康 epoch 一致（09-25 发散 epoch 5.7–5.9 s、09-26 冻结 epoch 5.8–5.9 s，均与长测交叉验证）。**但 loss 无效 ⇒ 该配置的"训练速度"只是计时，不代表有效训练**。
8. **轮次纪律**：所有点 warmup≥20、≥3 轮取最优（防时钟爬坡/跨会话漂移，见 AGENTS bench 方法论）；训练取稳态 epoch 平均并以长测交叉验证。
9. **本机噪声**：40HX 为专用计算卡（跑测时显存 0 占用），显示输出走 AMD R5 240。
10. **采集批次口径**：nn 侧三部分（§2/§3/§4）在 HEAD `4a12876` 上于 2026-09-26 同会话采集；torch 侧数据采集于 2026-09-25。**跨会话系统漂移 ±15%（AGENTS 方法论实测口径）直接进入所有 nn/torch 比值**；nn 侧两个采集日之间的同侧差异中，超出 ±5% 噪声地板的部分（GEMM +4~5%、transpose +40~59%、f32 e2e −13%、f32 显存 −47%）反映真实差异，归因见 §2/§3 对应条目。

---

## 6. 产物索引

**脚本（`bench/`）**
- `train_torch.py` — torch 训练 e2e（分词镜像 C++、epoch 计时、显存统计、`--adam-eps`、`--dtype`）
- `ops_torch.py` / `layers_torch.py` — torch 算子/层基准（CUDA event、多轮 best）
- `probe_torch_step.py` — torch 单步分段分解（§2.4）
- `run_with_vram.ps1` — 任意进程 + nvidia-smi 200 ms 采样显存峰值
- `run_ops.ps1` / `run_layers.ps1` — 双侧统一尺寸编排（3 轮）
- `parse_bench.ps1` — layer_bench 日志解析（best-of-3 轮 → markdown 行，§3/§4 数据来源）

**原始数据（`bench/raw/`）**
- 训练（nn，09-26 采集）：`nn_short.log`（全 f16 ⚠冻结）、`nn_short_f32.log`（f32 健康）、`nn_train.log`（全 f16 长测 ⚠冻结）、`nn_train_f32.log`（f32 长测健康，**09-26 采集**），对应 `*.vram` / `*.peak`
- 训练（torch，09-25 采集）：`torch_short.log`、`torch_short_fp32.log`、`torch_train*.json`
- 算子：`ops_nn.log`（8 形状组 ×3 轮，09-26 采集）、`ops_torch.log` / `ops_torch.json`（43 case）
- 层：`layers_nn.log`（09-26 采集）、`layers_torch_model.log` / `layers_torch_large.log` + `.json`
- f16 缺陷存证（09-26 采集，详见 14 号报告）：`f16_stable_matrix.log`（定性矩阵）、`f16_ep1.bin`/`f16_ep2.bin`（跨 epoch 权重快照）、`f16_probe_gpu.log`、`f16_precision_gpu.log`
- 语料/词表：`datasets/tinystories_bench40.txt`、`datasets/bench_bpe_8192.json`（C++ 8208）、`bench/raw/torch_bpe_8192.json`（torch 8192）

**复现命令**（仓库根目录）：
```powershell
# nn 训练 —— f32（主对比，健康收敛）
.\build\text_train.exe datasets\tinystories_bench40.txt --vocab datasets\bench_bpe_8192.json `
  --d-model 64 --num-heads 4 --num-layers 4 --d-ff 256 --seq-len 256 --batch-size 64 `
  --epochs 3 --optimizer adam --lr 0.001 --gpu=40HX `
  --save-interval 0 --log-interval 100000

# nn 训练 —— 全 f16（实验性；loss 打印冻结，仅计时/显存有效，见 docs/development/14-*.md）
#   追加: --precision-param f16 --precision-compute f16 --precision-stable f16 --precision-optimizer f16
# nn 训练 —— 推荐 f16 profile（实测健康）: 追加 --f16

# torch 训练（09-25 采集口径）
.\venv\Scripts\python.exe bench\train_torch.py datasets\tinystories_bench40.txt `
  --vocab bench\raw\torch_bpe_8192.json --vocab-size 8192 --epochs 3 --dtype fp16 --adam-eps 1e-4

# 算子 / 层（结果解析）
.\bench\run_ops.ps1 -Side nn     # 或 -Side torch
.\bench\run_layers.ps1 -Side nn  # 或 -Side torch
.\bench\parse_bench.ps1 -Kind ops      # best-of-3 表格
.\bench\parse_bench.ps1 -Kind layers
```
