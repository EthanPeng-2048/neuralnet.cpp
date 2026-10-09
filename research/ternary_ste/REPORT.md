# 三值 / 二值权重的 STE 训练：最小可行性探针

> **结论先行**：**可以**。用**现有公开 API**（`dsl::compute` + `select` + `rparam` + `matmul` +
> `compute_into`，**零库改动**）在本项目引擎上跑通了三值 `{-1,0,+1}`（1.58 bit）与二值 `{-1,+1}`
> （1 bit）的 STE（straight-through estimator）训练。两个实验都把教师权重模式**逐项 100% 还原**。
>
> | 变体 | 量化器 | MSE 首 → 末 | 教师模式还原 |
> |---|---|---|---|
> | 三值（含 0） | `select(w>0.5γ, +1, select(w<-0.5γ, -1, 0))` | 14.207462 → **0.000000** | **1370 / 1370 = 100%** |
> | 二值 | `select(w>0, +1, -1)` | 20.949087 → **0.000000** | **2048 / 2048 = 100%** |
>
> 本文是 `docs/development/21-quantized-weights.md` 的**前置证据**：它证明 P1 的算法路径
> （Layer 级量化 + STE backward）没有障碍，剩下的是架构与工程（精度枚举、能力声明、打包内核）。
> **它不证明性能**——本文全部是 CPU、f32 存储、单层玩具问题。

---

## 1. 复现

在仓库根目录执行（本机 clang 24，Release 语义；单 TU 编译约 20 s，两个程序各在毫秒级跑完 3000 步）：

```bash
clang++ -std=c++26 -O2 -march=native -fno-exceptions -Wno-pass-failed \
  -Iinclude -o /tmp/poc_ternary_ste research/ternary_ste/poc_ternary_ste.cpp -pthread
/tmp/poc_ternary_ste

clang++ -std=c++26 -O2 -march=native -fno-exceptions -Wno-pass-failed \
  -Iinclude -o /tmp/poc_binary_ste research/ternary_ste/poc_binary_ste.cpp -pthread
/tmp/poc_binary_ste
```

实测输出（逐字粘贴）：

```
# poc_ternary_ste
step    0  mse=14.207462  gamma=0.25052
step  500  mse=0.002389  gamma=0.49256
step 1000  mse=0.000000  gamma=0.49997
step 1500  mse=0.000000  gamma=0.50000
step 2000  mse=0.000000  gamma=0.50000
step 2500  mse=0.000000  gamma=0.50000
step 2999  mse=0.000000  gamma=0.50000

mse: first=14.207462  final=0.000000  (降 100.0%)
教师非零项符号/零帽吻合率: 1370 / 1370 = 100.0%

# poc_binary_ste
step    0  mse=20.949087  gamma=0.25052
step  500  mse=0.000037  gamma=0.49924
step 1000  mse=0.000000  gamma=0.50000
step 1500  mse=0.000000  gamma=0.50000
step 2000  mse=0.000000  gamma=0.50000
step 2500  mse=0.000000  gamma=0.50000
step 2999  mse=0.000000  gamma=0.50000

mse: first=20.949087  final=0.000000  (降 100.0%)
二值符号吻合率: 2048 / 2048 = 100.0%
```

---

## 2. 实验设计

**任务**：学一个**教师三元/二值线性映射** `y = W_t · x`（`IN=64 → OUT=32`，batch 512，
`x ~ N(0,1)`），教师权重 `W_t` 的元素按等概率取 `{-0.5, 0, +0.5}`（三值）/ `{-0.5, +0.5}`（二值）。

选这个任务的理由：它让"学没学到"有**逐项可判的答案**——训练后把模型权重量化，
与教师的符号/零帽逐项比对即可，不必解读 loss 曲线。

**模型**：单个量化线性层，每步：

```
γ    = mean|W|                      // 逐张量 absmean（研究口径：宿主读回算）
Wq   = select(W > 0.5γ, +1, select(W < -0.5γ, -1, 0))     // 三值化（DSL）
Ŷ    = γ · (Wq · X)                 // matmul 段 + 缩放融合
loss = MSE(Ŷ, Y)
dW_latent = γ · (dY · Xᵀ)           // STE：把 Wq 当常数求导
W   -= lr · dW_latent               // SGD（lr = 0.5）
```

**STE 的来历**：`Ŷ = γ·(Wq·X)` 对 latent `W` 的数学导数几乎处处为 0（量化器不可微），
STE 用恒等映射 `dWq/dW ≈ 1` 替代，于是 `dW_latent = γ · dY · Xᵀ`。注意这与本项目
`Linear::backward` 里 `grad_w = grad_output × inputᵀ` 是**同一个式子**，只是乘了去量化尺度 γ。

---

## 3. 实现要点（与 `docs/development/21-quantized-weights.md` 的对应）

1. **量化只用现有 DSL**：`ExprOp` 里已有 `Gt` / `Lt` / `Select` / `Abs`，`rparam` 承载"值不进 key"的
   运行期标量（γ、阈值、lr）。所以本探针**没有新增引擎原语、没有新增 dtype**。
2. **DSL 没有 `round`/`floor`**：三值化用"与阈值 0.5γ 比较的 select 链"表达——在 `[-1,1]` 区间内
   它与 BitNet 的 `RoundClip(w/γ, -1, 1)` 等价。这正是 P3（激活量化）要补算子的原因：
   权重三值化不需要 round，int8 激活量化需要。
3. **γ 在本探针里由宿主持有**（`to_matrix` 读回后算 absmean）。这只在 `src/` 层合法；
   落成 `Layer`（L2）时必须改用引擎归约（`row_reduce_sum(|W|)` + `row_broadcast`），
   否则违反铁律 #12（L2+ 禁用 Matrix，`bench/doc_inventory.ps1` 第 [4] 节会报违规）。
4. **量化缓冲用 f32**（本探针）；落成 `BitLinear` 时改用 f16——`{-1,0,+1}` 在 f16 下精确可表示，
   所以 P1 不需要新 dtype 就能拿到一半的内存收益（相对 f32）。

---

## 4. 本探针不能说明什么

- **不代表性能**：全部 f32 存储、CPU、单层、玩具规模。打包存储与专用 GEMM 是 P2 的事，
  且"打包省带宽 → 墙钟变快"必须先由 `layer_bench` 的配对 A/B 证明（本项目主线模型
  `d_model` 只有 64–256，matmul 可能落在启动延迟区而非带宽区）。
- **不代表大规模可训**：BitNet b1.58 的实战配方还包含 SubLN（量化前的额外归一化）等稳定性技巧，
  本探针不含。
- **不代表 STE 可被 gradcheck 验证**：量化器不可微，数值梯度与 STE 梯度本就不一致；
  落成层时只能验证"latent 梯度 = STE 公式"与"量化输出逐位符合 γ 规则"。
- **没有验证 GPU**：探针走 `CpuEngine`。GPU 侧的新表达式需要经 AOT 收集
  （库内 dry-run / 库外 `nn_enable_gpu_fusion`），否则闭合世界硬报错。
