# 26 — f16 提速方案（台账 24 · F4）：带宽优先、ALU 可选

> **读者**：项目开发者（本仓内部工作文档）。本册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。

> 状态：**部分实施**——**S1s/S1t（编译期特化）已落地并实测**（§4 S1s），S0/S1/S2–S5 待做。
> 来源 = [`24-f16-training-defects.md`](24-f16-training-defects.md) §4/§5 的
> **F4「f16 提速兑现」**（原口径「40HX 上 `--f16` ≤ f32」）。
> 关联：[`24`](24-f16-training-defects.md) §4（复现数据 / 候选来源）、`docs/development/05-mixed-precision.md`
> §12.4/§12.5/§12.11、[`18-roadmap.md`](18-roadmap.md) **P2-8（F16C）/ P2-9（step 内固定 I/O）**。

## 0. 结论速览

用户口径：**提速必须在「支持 / 不支持 f16 ALU」的设备上都能拿到**。本方案据此把 F4 的目标从
「让 f16 更快」改成 **「让 `--f16` 只靠省字节就更快」**（S0–S4，必须在无 ALU 设备上也成立）；
f16 原生算术（`shaderFloat16`，40HX 有、850M 无）**另立 S5 正式推进**——允许、且要把它用足，
但它**不是 S0–S4 的验收前提**。

四条关键判断（证据见 §1/§2/§4 S1s）：

0. **最大的一笔 f16 开销根本不是"f16"**：`#x` 是"一个 shader 覆盖任意精度签名"的运行期
   分派载体，热循环里每次读写都带 `((prec >> i & 1u) ? float(bNh[…]) : bN[…])` 三元。
   实测（850M，GPT `d64/h4/L4/ff256/seq256/b16`，15 步 `NN_PROFILE=1`，仅换 shader）：
   CE 融合注意力 kernel **146.1 → 120.0 ms**（−18%；同机 f32 基线 117.2 ms）。换成
   **编译期特化**（ggml/torch 的 per-type kernel 做法）即可拿到，且**不需要**
   `shaderFloat16` → 任意设备成立。这是 S1s/S1t（已实施）。

1. **当前 f16 慢，主因不是缺 f16 ALU，而是步内被「与精度无关」的框架固定开销主导**——本机 850M
   实测 `flush_batch` + `begin_batch` + `copy_from` ≈ **88% step 墙钟**；根因是 `start_frame()` 复用
   帧槽时**阻塞等 fence**，而 `PIPELINE_FRAMES = 6` < 每步 ~7 次 flush → host 每次都撞环深度。
   这部分开销 f32/f16 一模一样，**f16 的省字节收益被完全掩盖**。
2. **`--f16` 下最大的那个张量（logits）仍是 f32**：LM head 的计算精度被**强制 = stable**，而 `--f16`
   的 stable = F32（`compute_layer_gpt.hpp`）。`(vocab × batch·seq)` 是全模型最大激活，
   它不省字节 ⇒ 「f16 省带宽」在最关键的张量上落空。
3. **f16 路径相对 f32 还多了一套 `import`/`cast` 机器**（850M，15 步：`import` 调用 29198 → 54398、
   新增 `cast` 585 次）。它带来开销却不带来收益；应定位并消除。

方案主轴（对齐 ggml / torch 的通行做法）：**存储降精度、算术留 f32、融合消费者、去掉每步同步**。
验收锚改为 **「同设备、同配置，`--f16` step ≤ f32 且峰值显存更低」**，并要求该结论**在无
`shaderFloat16` 的设备（本机 850M）上同样成立**。

## 1. 复现与归因（本机实测）

### 1.1 端到端（`NN_PROFILE=1`，850M，`d64/h4/L4/ff256/seq256/b16`，15 步）

| 采集 | f32 | `--f16` | 判定 |
|---|---|---|---|
| 本轮（两次） | 134.6 ms/step | 139.3 / 152.4 ms/step | f16 **慢 3–13%**（同窗方差大） |
| 台账 [`24`](24-f16-training-defects.md) §4.1（40HX） | 101–107 ms/step | 118.4 ms/step | f16 **慢 11–17%** |

> 方向一致：**f16 不更快**；两机台「变慢的绝对量」接近（~10–15 ms/step），提示是**固定量**而非
> 「f16 计算更慢」。

### 1.2 per-key 归因（f32 → f16，同一 15 步）

| op | f32 | f16 | 变化 |
|---|---|---|---|
| `flush_batch` | 957.9 ms（46.9%） | 1062.8 ms（45.9%） | +11% |
| `begin_batch` | 464.7 ms（22.8%） | 504.2 ms（21.8%） | +8% |
| `copy_from` | 385.8 ms（18.9%） | 471.6 ms（20.4%） | **+22%** |
| `import` | 29198 calls | **54398 calls** | **×1.86** |
| `cast` | 15 calls | **585 calls** | f16 侧新增 |
| `eval_expr`（真·逐元素计算） | 58.8 ms | 63.1 ms | +7% |

**读法**：真正的「计算」项（`eval_expr`/`eval_expr_reduce`/`eval_expr_into`）两边都只占个位数百分比，
且 f16 并不更贵；f16 的额外开销集中在**边界/绑定机制**（`import` 翻倍、`cast` 出现、`copy_from` +22%）
与**帧同步**（`flush_batch`/`begin_batch`）上。**88.6% 的 step 时间是精度无关的固定开销。**

### 1.3 为什么固定开销这么大：帧槽阻塞

`begin_batch`/`flush_batch` 都经 `start_frame()`，复用帧槽时**阻塞等该槽 fence**（唯一常规等待点，
见 `backend/compute_vk_backend_frame.inc`）。环深度 `PIPELINE_FRAMES = 6`
（`backend/compute_vk_backend.hpp`），而本配置**每步 ~7 次 flush**（1 中点刷新 + 1 `end_batch` +
若干来自上传/drain 的帧切换）→ **每步都撞环深度** → host 每次换帧都在等 GPU → 流水线退化为串行。
这解释了 `begin_batch` **15.5 ms/次**、`flush_batch` **9.1 ms/次**这种「提交级操作却是毫秒级」的怪象。

> 换言之：**先把每步的帧数与同步点降下来，才可能看见任何计算/带宽优化的收益**（见 §4 S0）。

### 1.4 f16 变体确实命中（不是退回边界 cast）

`NN_PREC_TRACE=1`（探针构建）显示热路径命中 `[prec][alu-hit] <key>#x`，无 `miss`。即：本仓 f16 走的是
**storage-only** 变体 —— shader 用 `GL_EXT_shader_16bit_storage` 声明 `float16_t` 缓冲、**算术仍 f32**
（`shaders/elementwise_v2.comp` 头部注释即写「缓冲 float16_t、算术 f32」）。

**这是好消息**：本仓已有「不依赖 `shaderFloat16` 的 f16 存储变体」这一正确底座（和 ggml 同思路），
所以 §4 的方案不需要等新硬件支持。

## 2. 为什么 f16 现在省不到时间（三条根因）

### 2.1 固定开销掩盖（§1.2/§1.3）

计算侧的 `eval_expr*` 只占个位数百分比，`flush_batch`+`begin_batch`+`copy_from` 合计 ≈88% 的
**host 侧墙钟**。但 —— **订正（2026-10-13，S1s 实测）**：这 88% 主要是 **host 在等 GPU**
（`start_frame` 复用帧槽时阻塞等 fence），而**设备忙率已达 95.9%/96%**（f32/f16 同）。
即 GPU 才是瓶颈，host 侧的 `flush_batch` 时长不是"GPU 空转的浪费"，而是"host 被 GPU 拖住"。
因此：
  · **kernel 侧的优化是可测的**（S1s 把 f16 设备时间从 2163 拉到 1962 ms，墙钟同幅下降）；
  · S0（降帧数/去 drain）的天花板 = `墙钟 − 设备 kernel`（本机 ≈90 ms ≈ 4.4%），
    且**对 f32/f16 同幅**——它能拉快两条路径，但**不会**改变 f16↔f32 的相对关系。
原判断（"固定开销会把任何计算侧收益吞掉"）**不成立**，S0 的定位从"先决阻塞项"改为
**独立的通用提速项**（[`18`](18-roadmap.md) P2-9）。

### 2.2 主导张量仍是 f32（本节是「f16 本该省字节却没省」的核心）

`GPTModel` 的 LM head 计算精度**强制 = `stable`**（`compute_layer_gpt.hpp` 构造器与
`set_precision_profile` 两处；tied 路径同样按 `p_.stable`）。`--f16` 的 stable = F32 ⇒ **logits 是 f32 存储**，
而 `(vocab × batch·seq)` 是全模型最大激活（本配置 8208×4096×4B ≈ **134 MB**，CE 链还要多次读它）。

> 权衡背景（见 `compute_layer_gpt.hpp` 注释、`05-mixed-precision.md` §12.5）：让 logits 走 f32 是**为了
> 避免「边界 cast 被 k 个算子各物化一份 f32 副本」（当时更大）而做的选择**。但它把 f16 唯一的
> 收益点（最大张量省一半字节）直接放弃了。

### 2.3 f16 路径自身多出绑定/cast 工作量（§1.2 的 `import` ×1.86、`cast` 585）

同一张量在 f16 下被「按 f16 视图绑定一次 + 可能再按 f32 视图绑定/转换一次」，实测调用数近乎翻倍。
它带来时间却无收益，属于**应清零的纯开销**。

## 3. 外部实现参考（ggml / torch 的通行做法）

> ⚠ 本节是**外部实现的做法归纳**（依据其公开设计与源码的通行模式），**未在本仓复核**；
> 实施时按 §6 的「四段式（结论 / 证据 / 重跑命令 / 边界）」对当时版本源码逐条复核后再引用。

| 机制 | ggml | torch | 本仓现状 / 可行性 |
|---|---|---|---|
| **降精度只降存储，算术留 f32 + f32 累加** | 量化块格式（q4_0/q4_K…）与 f16 一律在 kernel 内**反量化到 f32** 再算，累加恒 f32 | AMP：f32 master 权重 + fp16/bf16 计算 + **f32 累加** | ✅ **已具备**（storage-only `#x` 变体，§1.4） |
| **能力只选「计算风格」，不选「存储风格」** | 有 `shaderFloat16` 就走原生 f16 算术路径，否则走 f32 算术路径，但**两条都保持 f16/量化存储** | 有 tensor core 用 fp16，无则退回 f32 计算，但激活仍是半精度 | ✅/⚠ 本仓 `#a`(native16，43 条)/`#x`(f32 算术，126 条) 已二分且 `#a` 优先，方向正确；**缺口 = `#a` 的结构谓词排除 fold/matmul/任何 reduce/rparams** → 最重的链拿不到原生 f16（见 **S5**） |
| **融合消费者，不物化中间量** | `mul_mat` 把 dequant+GEMM 融成一个 kernel（绝不物化反量化后的权重） | fused `cross_entropy` / `scaled_dot_product_attention`（**从不物化完整 logits/softmax**） | ⚠ 本仓 CE 链已部分融合（fold logsumexp + 单 kernel loss/grad），但 logits 仍被「强制 f32 + 多算子读」 |
| **向量化打包读写（f16x2）** | 用 `float16x2` 打包读，**把事务数也减半**（不只是字节） | CUDA/CPU 侧靠向量化 load/store 与 F16C | ❌ 本仓 f16 变体目前是逐元素 `float(buf[i])`，未打包 |
| **一次提交跑完一个图（无 per-op 同步）** | 整个 forward 录进**一个** command buffer，host 不中途 drain | CUDA graph / 单 kernel 融合，无 per-op sync | ❌ 本仓每步 ~7 次帧切换 + 环深度 6 → 串行化（§1.3） |

**归纳为四条对任意设备都成立的原则**：
1. 省字节靠**存储格式**；提速**不依赖** ALU 特性。
2. **不要为了省 cast 而把最大张量升回 f32**——改为**融合它的消费者**。
3. 打包读写把**事务数**也降下来（带宽之外的第二层收益）。
4. **减少每步的提交/同步点**，否则优化测不出来。

## 4. 方案（分期 + 验收）

**S0–S4** 每期都给出「为什么能在**任意设备**（含无 `shaderFloat16` 的 850M）生效」；**S5** 是
capable 设备（有原生 f16 ALU）的额外项。

### S1s / S1t（✅ **已实施 2026-10-13**）— 编译期特化：去掉热循环里的运行期精度分派

- **做什么**：把"精度是运行期参数"（`#x` 分派）在**热路径**上换回"精度是编译期常量"。
  两条新变体，与 `#x` 数值逐字等价（同一 `load→float()→f32 算术→float16_t()` 链），
  只需 `storageBuffer16BitAccess`（**不需要** `shaderFloat16`）：
  - **`key#s`（V1s）**：签名固定为"全输入 f16 + 输出 f16"，对每个可融合结构**无条件发射**
    （`expr_spec.hpp` 的 `EXPR_PREC_STORAGE_SUFFIX`）；
  - **`key#xxxx`（V1t）**：构建期扫描**真实观察到的签名**——`scan_exprs` 的 dry-run 与
    模型 pass **各跑 f32 + f16 两遍**，`dsl::compute*`（`scan_register`）与
    `eval_expr*`（CPU 引擎入口，覆盖 fold / 直接求值）按**真实张量精度 + 真实 P** 记入
    `ExprRegistry::variants`，生成阶段为每条发一份静态特化 shader。
    **纯优化**：未枚举到的签名照旧落 `#x`（语义兜底不变）。
- **为什么设备无关**：不碰数值、不碰 ALU 特性，只把"运行期三元"变成"编译期常量"。
- **实测（850M，无 `shaderFloat16`，GPT `d64/h4/L4/ff256/seq256/b16`，15 步，3 轮回合）**：
  | 口径 | f32 | `--f16`（特化前） | `--f16`（特化后） |
  |---|---|---|---|
  | step 墙钟（平均） | 2037 ms | 2260 ms（**+10.6%**） | **2055 ms（+0.9%）** |
  | 设备 kernel 合计 | 1949 ms | 2163 ms | **1962 ms** |
  | 峰值显存（`nvidia-smi` 采样） | 818 MiB | — | **606 MiB（−26%）** |
  | CE 融合注意力 kernel | 117.2 ms | 146.1 ms（`#x`） | **120.0 ms**（`#s`/`#10007`） |
  分项：`c2aca8a29eb4617f`（LM head）154.4→`#0005`137.4（f32 137.1）；`71733007a831714a`
  （wgrad）143.9→`#10001`138.8（f32 140.6）；`6996c1ccdb7656fc` 117.1→`#0001`112.2
  （f32 111.5）。**唯一残留的 f16 正差 = 优化器**（f16 退化为 4 个逐 kernel 融合
  ≈ +12.6 ms/15 步 ≈ 0.6%，因 `#b` 批量派发变体**只发 V0 全 f32**）→ 另立 **P2-12**。
- **回归**：`ctest` CPU **41/41**、GPU **43/43**（含 `f16_precision_test` / `ce_fusion_test` /
  `expr_fuse_test` / `bitwise` 类字节锚）全绿；构建期融合结构 126 条 / 精度变体 **377** 条
  （`#x` 124 + `#s` 124 + `#xxxx` 89 + `#a` 41；110 条观察签名里 22 条全 f16 与 `#s` 同 shader
  故跳过）。构建时间影响：`glslc` 并行，fused shader 由 402 增到 **617** 份（`text_train` 单目标
  Release 冷构建含 scan + glslc 实测 ≈2m35s，另有 ccache 命中路径）。
- **风险**：低——新增变体与 `#x` 同源同值；特化集合缺失时自动回退。

### S0 — 去掉 step 内固定开销 ❌ **已评估-不做（2026-10-14，实测推翻前提）**

原前提"88% 的 host 墙钟是固定开销、会把计算侧收益吞掉"**已被推翻**（见 §2.1 订正）：设备忙率 95.9%，host 本就在等 GPU，等待点之间可以互相搬家。三项实测：
  1. **消除 `copy_from` 的 acquire 阻塞**（staging ring `64MB×4` → `8MB×16`）：`copy_from`
     **392 → 3 ms**，但 `flush_batch` 同步 **955 → 1213 ms**——两者之和恒 ≈1347 ms，墙钟不变
     （3 轮 2200/2059/2053 vs 原 2044，噪声内）→ **已回退**。
  2. **减少每步帧数**（`--flush-interval 1 → 4`，flush 次数 105 → 20）：墙钟 **2044 → 2096 ms
     （+2.5%，反而更慢）**——大帧降低 host 录制/GPU 执行 overlap；默认 1 已是实测最优。
  3. 真正的天花板 = `墙钟 − 设备 kernel` ≈ **86 ms/15 步 ≈ 4.2%**，f32/f16 同幅。
结论：该项**无可回收收益**，保留默认配置（原"S0 是先决、阻塞其余所有项"的判断作废——
S1s 在 S0 未做的情况下已拿到 10.6% → 0.9% 的实测收益）。

#### （原方案，保留备查）

- **做什么**：归口 [`18-roadmap.md`](18-roadmap.md) **P2-9**——(a) CE 的 labels/mask **不再每步上传**
  （缓存为常驻张量，或按 batch 预分配复用）；(b) 去掉每步多余的帧切换，使 **帧数/步 ≤ 1–2**
  （远小于环深 6）；(c) 复核 `copy_from` 非 fresh 分支的 drain 是否被热路径触发。
- **为什么设备无关**：它不碰数值、不碰精度，纯粹是提交/同步结构。
- **验收**：`NN_PROFILE` 每步 `copy_from` ≈ 0；帧切换/步 ≤ 2；`begin_batch`/`flush_batch` 单次
  回到 **µs 量级**；同口径 e2e step 时间下降（f32 与 f16 都降）；**数值/字节锚逐位不变**。
- **风险**：低（控制流改动 + 缓存张量），但必须用 `NN_BIND_DEBUG=1 ctest` 与 CPU/GPU 字节锚守住。

### S1 — 让主导张量真正以 f16 存 ❌ **已评估-不做（2026-10-14，实测净亏）**

做法（已实现并实测后**回退**）：把 LM head 的计算精度从"强制 = stable（`--f16` 下即 F32）"
改为跟随 `compute` —— 即 `--f16` 下 logits 落 f16（全模型最大张量减半）。历史顾虑
（"边界 cast 会给 loss 链每个算子各物化一份 f32 副本 → OOM"）已由 in-kernel 变体消除，
故该做法在原理上成立。

**但实测净亏（850M，f16，15 步）**：
  · LM head 前向 `c2aca8a29eb4617f`：138.4 → **130.5 ms**（输出写量减半，−7.9）✓
  · CE 反向 `26c8f729e11e40bd`：77.4（V0 全 f32）→ **116.4 ms（`#x` 运行期分派）**（+38.9）✗
  · 净 **+31 ms**，端到端 2018 → 2042 ms。
  **根因**：CE 反向链的**签名从未被扫描观察到**——`scan_exprs` 只调 `ce.forward_sparse`，
  从不调 `ce.backward()`（见 `tools/scan_exprs.cpp`）→ 该结构没有编译期特化变体 → 掉回带
  `prec` 三元的 `#x`。**前提**：要先让扫描跑一次 CE 反向（或在模型 pass 里补一步训练），
  否则 S1 的收益会被这一项吃光。
  · 数值：loss 轨迹偏离约 3%（f32 8.6764/8.2419/7.8675 vs f16 8.7111/8.3892/8.1161）——
    f16 logits 使 CE 梯度（softmax−onehot）也按 f16 落盘，收敛变慢。

结论：**当前形态下不采用**（净亏 + 质量下降）。若将来重开，前置条件是"扫描覆盖 CE 反向"
（= 让 scan_exprs 跑一次真实的 forward→loss→backward→optimizer），并把质量影响重新评估。

#### （原方案，保留备查）

- **做什么**：logits **保持 f16 存储**（不再因 LM head 强制 stable 而升 f32），把 CE 链对 logits 的
  多次读取**融成一次读**（fold logsumexp + loss + grad 共享同一份 f16 读；kernels 内 f32 累加）。
  即把 §2.2 的「升 f32 免得 k 份副本」换成 ggml/torch 式「**f16 存 + 一次读**」。
- **为什么设备无关**：只依赖 16bit storage（已具备），算术仍是 f32。
- **验收**：`NN_PREC_TRACE` 在热路径**零 boundary-cast miss**；logits 相关字节读量≈减半
  （建议加「每步 bytes moved」计数器做锚）；f16 峰值显存 < f32；step ≤ f32（**850M 上也要成立**）。
- **风险**：中——需保证数值口径（f32 累加不变、RNE 落 f16）与既有容差一致；`ce_fusion_test` 对拍
  + `f16_precision_test` 轨迹对拍必须全绿。

### S2 — 向量化打包读写（f16x2）

- **做什么**：热 f16 变体（elementwise/reduce/matmul）改为按 `f16x2`/`u32` 打包读写，再在寄存器内
  转 f32 计算。
- **为什么设备无关**：`GL_EXT_shader_16bit_storage` 提供 `f16vec2` 存储类型，不要求 ALU 特性。
- **验收**：数值与 S1 逐位一致；`eval_expr*` 项耗时或「事务数」下降（用 S1 的计数器）。
- **风险**：中——尾部/奇偶元素与对齐需按既有 `create_f16_tensor` 奇数保护口径处理。

### S3 — f16 GEMM（storage-only，f32 累加）+ 小 N GEMV

**网格粒度部分 ✅ 已完成 2026-10-14**：`#f`（细 tile）与 `#ks`/`#kr`（K 切分）原本**只对
V0 全 f32 发射**（`generate_glsl_matmul` 对 `k_split && sig != 0` 直接拒绝）→ 网格饥饿
形状（大 batch 的 wgrad）在 f16 下没有 K 切分可用、掉回 64-tile：**40HX 实测 79 ms/块
vs f32 的 K 切分 25 ms/块**（占该机 f16 全部残差；配对旁证 `NN_MM_NO_KSPLIT=1` → 细 tile
84 ms/块 ⇒ **只有 K 切分能救**）。修法 = 把 sig 穿进三条生成路径，为每个静态签名变体各发
一份（键形如 `key#10003#ks`）：
  · **部分和缓冲恒 f32**（后端 `use_ks` 分支按 f32 分配）→ `#ks` 输出声明钉成 `float`、
    写原始部分和（不做 f16 舍入），注入的 sig **只决定 A/B 读取类型**；`#kr` 跨段求和后
    再按 sig 落盘（含 RNE），累加恒 f32 ✓。
  · 后端"`#f`/`#ks`/`#kr` 无需 16 位存储"的豁免收窄到**纯 V0 键**（后缀可叠加：
    `key#10003#ks` 里仍有 `float16_t` 缓冲）；运行期派发哨兵 `#x` 不叠加网格变体。
  · **实测**：40HX（b64）该 wgrad **f16 16.0 ms/块 < f32 25.1 ms/块**；端到端 40HX
    `--f16` **1013 vs f32 1027 ms（−1.4%）**、850M（b16）**2005 vs 2038 ms（−1.6%）**；
    loss 对拍（40HX 三点）f32 8.6757/8.2646/7.8743 vs f16 8.6676/8.2569/7.8643（<0.15%）；
    `ctest` GPU 43/43。变体数：细 tile 14→46、K 切分 14→46（fused shader 617→713）。

**仍未做**：小 N GEMV 的 f16 变体，以及"K 重的普通 matmul 形状全面补 f16 存储变体"。

- **做什么**：K 重的 matmul/wgrad 形状补 f16 存储变体（A/B 读 f16、算 f32、写 f16）；小 N 走
  `matmul_q_gemv` 同类 GEMV。
- **为什么设备无关**：同 S1/S2，只依赖 16bit storage。
- **验收**：`layer_bench --op matmul --f16` ≤ f32（**850M 上**）；40HX 上 e2e `--f16` ≤ f32。
- **风险**：中——`matmul_tiled` 的轴向/分块不变量（AGENTS §4.2 ③）须逐条守。

### S4 — CPU 侧：F16C 打包转换（= P2-8）

- **做什么**：`_cvtss_sh`/打包转换替代软件位操作（`18-roadmap` P2-8）。
- **为什么**：CPU 无 f16 ISA 时 f16 只省内存不省时间（`05-mixed-precision.md` §12.4）；F16C 让
  **转换**这一步变快，配合 S1/S2 的字节减少才可能接近平速。
- **验收**：CPUID 探测 + `NN_ENABLE_NATIVE` ON/OFF 两态；`f16_cpu_probe` 全量对拍逐位一致；
  CPU `--f16` 训练 step 不慢于 f32（内存更低）。

### S5 — 原生 f16 算术（`#a` / `shaderFloat16`）：允许 + 用足 + 验收

**现状（本仓已有，但覆盖不全）**：`#a` = V2b native16（`float16_t` 全链算术），构建期已生成
**43 条**结构（`#x` 126 条）；运行期优先级**高于** `#x`（`find_prec_variant_` 先试 `#a`），
设备能力经 `shaderFloat16` 查询 + 启用（`NN_VULKAN_NO_16BIT_ALU` 逃生阀可关）。**缺口**在于结构谓词
`expr_prec_sig_native16` **排除** `fold` / `matmul` / **任何 reduce** / 带 `rparams` 的表达式
（`expr_spec.hpp`）——于是最重的几条链（CE 的 logsumexp fold、归约、matmul 融合段）**永远走不到
原生 f16**，只能用 `#x`（f16 存储 + f32 算术）。

- **做什么**：把 native16 资格扩到 **reduce / fold / matmul**（**累加必须保持 f32**——只让逐元素段走
  `float16_t` 算术，归约仍是 f32 累加，否则数值口径变化）；逐条复核既有禁忌（索引值 >2048 精度、
  `rparam` 微值 flush-to-0、常量 <6e-8）；保留 `NN_VULKAN_NO_16BIT_ALU` 作为 A/B 与逃生阀。
- **为什么**：在 capable 设备上省掉「f16→f32→f16」的转换与寄存器压力（40HX 有原生 f16 ALU）。
- **验收**：① 40HX 上 `--f16` step ≤ f32；② **相对 storage-only 路径有可测增益**——同一 40HX 上
  用 `NN_VULKAN_NO_16BIT_ALU=1/0` 做配对 A/B（这是「原生 f16 到底值多少」的唯一干净口径）；
  ③ 数值与 storage-only 路径在容差内一致（`ce_fusion_test` / `f16_precision_test` 对拍）。
- **风险**：中——f16 累加（禁用）、`rparam`/常量微值、索引精度三类禁忌必须逐条守住；`#a` 覆盖
  扩张会增大构建期 shader 数。

## 5. 验收总口径（跨设备矩阵）

| 维度 | 取值 |
|---|---|
| 设备 | 40HX（有 f16 ALU） / 850M（无 f16 ALU） / CPU |
| 配方 | 全 f32 / `--f16`（stable=f32） |
| 指标 | step 时间、峰值显存、**每步 bytes moved**（S1 起新增计数器）、loss 轨迹（容差内） |
| 门禁（存储路径 S0–S4） | ① f32 不回归（字节锚/轨迹）；② **`--f16` step ≤ f32**（**三设备都要，含无 `shaderFloat16` 的 850M**）；③ f16 峰值显存更低；④ `ctest` 全绿（含 `*_gpu`、`ce_fusion_test`、`f16_precision_test`）；⑤ `NN_BIND_DEBUG=1 ctest` 全绿 |
| 门禁（原生 f16，S5） | 在 capable 设备（40HX）额外要求：native16 已启用且有增益 —— `NN_VULKAN_NO_16BIT_ALU=1/0` 配对 A/B 显示 `=0`（原生开）更快；数值与 storage-only 路径容差内一致 |

**进度总览（2026-10-14）**：✅ 已落地 = **S1s/S1t**（编译期特化）+ **网格粒度变体**（#f/#ks/#kr 叠加到
特化签名）+ **P2-12 部分**（批量派发）+ **P2-3**（RAPT/CNN 接线）；❌ 已评估-不做 = **S0**（前提被实测推翻）、
**S1**（净亏 + 质量下降，见 §4 S1）；⬜ 未做 = **S2**（f16x2 打包）、**S4/P2-8**（CPU F16C）、
**S5/P2-10**（原生 f16 用足）、**F5**（全 f16 可用性）。

**已达成（2026-10-13，S1s/S1t）**：850M（**无 `shaderFloat16`**）`--f16` **step 墙钟 2055 ms vs
f32 2037 ms（+0.9%，特化前 +10.6%）**、设备 kernel 1962 vs 1949 ms、**峰值显存 606 vs 818 MiB
（−26%）**；② 项"≤ f32"**尚未完全达成**（差 0.9%，且其中 ≈0.6% 是可修的优化器退化，见 P2-12）。
40HX（capable）的对照见 [`24`](24-f16-training-defects.md) §4.5。

> 口径变化（相对 [`24`](24-f16-training-defects.md) §5 F4 原文「40HX 上 ≤ f32」）：① 按用户要求改为
> **「任意设备」**（存储路径）；② **原生 f16 不被排除**——它为 **S5 正式立项**，在 capable 设备上
> 额外验收，但**不允许**用它解释存储路径（S0–S4）的收益。

## 6. 纪律与边界

1. **不动数值语义**：f32 累加、RNE 落 f16、既有容差口径保持不变；每期以对拍守住。
2. **不引入 loss scaling**：那是 [`24`](24-f16-training-defects.md) **F5**（全 f16 可用性）的范围，与本册
   （f16 存储提速）正交。
3. **允许并推进原生 f16（S5）**：不排除 `shaderFloat16`，反而要在 capable 设备上把它用足；但
   **存储路径（S0–S4）不得依赖它**——能力探测只允许切换**算术风格**，任何路径都不放弃 f16 存储。
4. **外部参考需复核**：§3 的 ggml/torch 做法按四段式对当时版本源码复核后再引用。
5. **先 S0 再谈提速**：S0 未完成前，任何 S1–S3 的收益都不可信（会被帧同步噪声吞掉）。

## 7. 证据索引

| 来源 | 内容 |
|---|---|
| [`24-f16-training-defects.md`](24-f16-training-defects.md) §4 | F4 复现数据（40HX）+ 候选来源 + §4.4 本机归因 |
| `backend/compute_vk_backend_frame.inc` | `begin_batch`/`flush_batch`/`start_frame`（帧槽阻塞等待） |
| `backend/compute_vk_backend.hpp` | `PIPELINE_FRAMES = 6`（环深度） |
| `shaders/elementwise_v2.comp` 等 | f16 变体 = storage-only（`GL_EXT_shader_16bit_storage` + f32 算术） |
| `include/neuralnet.cpp/compute_layer_gpt.hpp` | LM head 计算精度强制 = `stable`（⇒ `--f16` 下 logits 为 f32） |
| `include/neuralnet.cpp/compute_engine.hpp`（`upload_span`） / `compute_gpu_engine.hpp`（`copy_from_impl`） | `fresh_dst` 与 drain 语义（每步上传是否排空） |
| `docs/development/05-mixed-precision.md` §12.5/§12.11 | 边界 cast 代价、op-level f16 GEMM 现状 |
| [`18-roadmap.md`](18-roadmap.md) | P2-8（F16C）/ P2-9（step 内固定 I/O）/ **P2-10（原生 f16 算术用足）** |
| `include/neuralnet.cpp/expr_spec.hpp` | `expr_prec_sig_native16`：native16 结构谓词（排除 fold/matmul/reduce/rparams） |
| `build/<cfg>/generated/fused_registry.hpp` | 注册表头注释：`#x`（运行期精度分派）/ `#s`（全 f16 特化）/ `#xxxx`（扫描签名特化）/ `#a`（native16）等各变体 |
