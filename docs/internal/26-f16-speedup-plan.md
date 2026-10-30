# 26 — f16 提速与两套 f16 计划（标准 `--f16` / 全流程真 f16）

> **读者**：项目开发者（本仓内部工作文档）。本册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)。
> **状态**：现行。

> 状态（2026-10-18）：**F4 提速已收口；后续拆成 Plan A（标准 `--f16`，参考 PyTorch AMP）与
> Plan B（全流程真 f16）两条活跃路线**——S1s/S1t（编译期特化）+
> 网格粒度变体 + S4（CPU F16C）+ P2-12（批量派发）已落地并实测；验收锚「**任意设备** `--f16` ≤ f32」
> **已达成**（40HX 1013 vs 1027 ms、850M 2005 vs 2038 ms，见 §4 S3/P2-13）。S0/S1/S2/S5 的
> **已评估-不做**保留为历史证据（各附实测与上界分析），但它们只否定当时「以提速为唯一目标」的形态，
> **不否定四槽全 f16（B-Mem 无 f32 持久态；capable 另追 B-Arith 零转换）的可用性目标**。
> 恢复后的定义、当前基线与 R0–R6 分期见 **§8（当前执行口径）**；
> §8.B.7 新增 **数值鲁棒性技术栈**（SR / GradientStabilizer / ECO / Unit Scaling / Peri-LN /
> attention `δ` 修复）的外部证据与 R2.5 落地顺序。
> 逐条历史状态见 §4 进度总览。
> 来源 = [`24-f16-training-defects.md`](24-f16-training-defects.md) §4/§5 的 **F4「f16 提速兑现」**
> 与 **F6/F7 两套 f16 计划**（F4 原口径「40HX 上 `--f16` ≤ f32」）。
> 关联：[`24`](24-f16-training-defects.md) §4（复现数据 / 候选来源）、`docs/development/05-mixed-precision.md`
> §12.4/§12.5/§12.11、[`18-roadmap.md`](18-roadmap.md) **P2-8（F16C）/ P2-9（step 内固定 I/O）/
> P2-14（全流程真 f16）/ P2-15（标准 `--f16`，PyTorch AMP 参照）**。

## 0. 结论速览

用户口径：**提速必须在「支持 / 不支持 f16 ALU」的设备上都能拿到**。本方案据此把 F4 的目标从
「让 f16 更快」改成 **「让 `--f16` 只靠省字节就更快」**（S0–S4，必须在无 ALU 设备上也成立）；
f16 原生算术（`shaderFloat16`，40HX 有、850M 无）**另立 S5 正式推进**——允许、且要把它用足，
但它**不是 S0–S4 的验收前提**。

> **口径提示**：§0–§4 是 **F4 提速**口径（“存储降精度、算术留 f32”）。§8 的**真 f16** 是另一目标：
> capable 设备要求零 f16↔f32 转换与 native16 全覆盖，不能拿本节的“f32 累加”当 §8 的验收口径。

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
  **P2-12 已完成（2026-10-14）**：除"`#b` 对 extras 为空的静态签名变体各发一份"外，残差根因是
  **scan 与运行期不同源**（优化器 dry-run 没透传 profile + 复合 kernel 的运行期签名没在 scan 登记）
  → 40HX/f16 上 `[ir-c][degrade]` 70 → **0**/步、优化器派发 138.7 → **8.55 calls/步（−94%）**、
  优化器 host 阶段 −34%，但**端到端墙钟在 GPU-bound 下无可见收益**（设备忙率 95.5%）。
  详见 [`18`](18-roadmap.md) P2-12 行与 [`24`](24-f16-training-defects.md) §4.5。
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

### S2 — 向量化打包读写（f16x2）❌ **已评估-不做（2026-10-14，附实测）**

前提是"f16 热变体的读是**事务数**瓶颈，打包成 f16x2 能把事务数也减半"。**实测不支持这个前提**：
若热核受事务数支配，那么"存储减半"的 f16 变体就该显著快于 f32；而实测（850M，同一结构、只换变体）
f16 存储变体与 f32 V0 **已在 ±2.5% 内**（注意力 +2.4%、LM head +0.2%、wgrad −1.3%、`6996c1ccdb7656fc` +0.6%），
部分结构 f16 反而快 11% —— 即**字节/事务不是限制项**：注意力核是 shared-memory/barrier 主导、
matmul 核是 f32 累加循环主导（同一事实也解释了 S5 的收益上界）。
**结论**：打包读的预期收益 <3%（且只作用于地址连续的快路径，本仓热核多为跨步/广播访问），
而代价是重写逐元素 + 归约 + 矩阵三条生成路径的取址与尾部处理 —— **不做**。
（保留：`#s`/`key#xxxx` 特化已覆盖的编译期类型选择；若将来出现**真正带宽受限**的形状
（大 batch 的纯 GEMM），按四段式重新立项。）

#### （原方案，保留备查）

- **做什么**：热 f16 变体（elementwise/reduce/matmul）改为按 `f16x2`/`u32` 打包读写，再在寄存器内
  转 f32 计算。
- **为什么设备无关**：`GL_EXT_shader_16bit_storage` 提供 `f16vec2` 存储类型，不要求 ALU 特性。
- **验收**：数值与 S1 逐位一致；`eval_expr*` 项耗时或「事务数」下降（用 S1 的计数器）。
- **风险**：中——尾部/奇偶元素与对齐需按既有 `create_f16_tensor` 奇数保护口径处理。

### S3 — f16 GEMM（storage-only，f32 累加）+ 小 N GEMV ✅ **已完成 2026-10-14**

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

**小 N GEMV 与本条的关系**：f16 的小 N 路径**早已由 op-level 原语覆盖**——`matmul_gemv_f16`（`#f` 之外的"小 N GEMV f16 变体"，见 `destroy.inc` Phase C2e）在设备支持 16 位存储时即创建；本条原方案里的`matmul_q_gemv` 是三值打包 GEMM 的专属原语（T1_58，非 f16 稠密路径）。故 S3 无残余项。

**K 重的普通 matmul 形状**：由 `#f`（细 tile）与 `#ks`/`#kr`（K 切分）在**静态签名变体上各发一份**覆盖（键形如 `key#10003#ks`），40HX 实测该 wgrad **f16 16.0 ms/块 < f32 25.1 ms/块**。

- **做什么**：K 重的 matmul/wgrad 形状补 f16 存储变体（A/B 读 f16、算 f32、写 f16）；小 N 走
  `matmul_q_gemv` 同类 GEMV。
- **为什么设备无关**：同 S1/S2，只依赖 16bit storage。
- **验收**：`layer_bench --op matmul --f16` ≤ f32（**850M 上**）；40HX 上 e2e `--f16` ≤ f32。
- **风险**：中——`matmul_tiled` 的轴向/分块不变量（AGENTS §4.2 ③）须逐条守。

### S4 — CPU 侧：F16C 打包转换（= P2-8）✅ **已完成 2026-10-14**

`detail::float_to_half_bits` / `half_bits_to_float` 增 `#if defined(__F16C__)` 快路径（运行时硬件单指令；
常量求值与 `NN_ENABLE_NATIVE=OFF` 仍走软件位操作）。**实测（CPU，GPT d64/L2/seq64/b4，10 步）**：
f16 **3.6 s → 1.9 s**（f32 1.9 s）—— CPU 侧 f16 从"慢 2×"变为**不慢于 f32**。逐位一致性由
`precision_test` 新增段守（编译期软件路径 vs 运行期硬件路径，38 条边界值全等）。

#### （原方案，保留备查）

- **做什么**：`_cvtss_sh`/打包转换替代软件位操作（`18-roadmap` P2-8）。
- **为什么**：CPU 无 f16 ISA 时 f16 只省内存不省时间（`05-mixed-precision.md` §12.4）；F16C 让
  **转换**这一步变快，配合 S1/S2 的字节减少才可能接近平速。
- **验收**：CPUID 探测 + `NN_ENABLE_NATIVE` ON/OFF 两态；`f16_cpu_probe` 全量对拍逐位一致；
  CPU `--f16` 训练 step 不慢于 f32（内存更低）。

### S5 — 原生 f16 算术（`#a`）：**允许 + 已在用**；✅ **F4 增速口径下不扩张（2026-10-14）**

> **注意**：本节是 **F4 提速口径**的历史结论；§8 的真 f16 目标已把“native16 全覆盖”反转为核心
> 门禁（R4.1），下面的“不再扩张”不适用于真 f16。

**口径（用户要求）**：原生 f16 **允许启用、不排除** —— `#a` 在运行期查找序里**优先级最高**，
能力经 `shaderFloat16` 查询 + 启用，逃生阀 `NN_VULKAN_NO_16BIT_ALU`；**40HX 实测 9 条结构命中 `#a`**。
**实测增益**（40HX，`d64/h4/L4/ff256/seq256/b64`，f16，15 步 ×3 轮，`NN_VULKAN_NO_16BIT_ALU` 配对 A/B）：
ALU 开 **1005 ms** vs 关 **1021 ms** ⇒ 现有覆盖贡献 **≈1.5%**。
**为什么不再扩资格到 fold / matmul / reduce**：① 这三类的耗时主体是**归约累加循环**，而累加按数值
要求**必须保持 f32**（本仓 §7.2 / 业界 fp16 GEMM 同口径）⇒ 扩资格只能覆盖其中的**逐元素**片段，
量级与已覆盖的 9 条同构（≈1%）；② 其余排除项（`rparams`/微常量/行-列-批索引）**是原则性的**：
rparam 恒 f32 传递（放宽会把整链隐式提升回 f32 → 收益归零），微常量与索引在 f16 下 flush/失精
（放宽即静默错值）；③ 故原判断"最重的几条链拿不到原生 f16 ⇒ 收益大"**不成立**——那几条链的
成本在 f32 累加循环里。**保留**：优先级、逃生阀、`#a` 覆盖与全部回归不变。

#### （原方案，保留备查）

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

> §5 是 **F4 提速**的验收口径；「全流程真 f16」的验收矩阵见 **§8.B.4**（配方含四槽全 F16，覆盖
> 零转换、正确性、长跑质量与全模型/全设备）。

**进度总览（2026-10-18 更新）**：**F4 提速已收口；§8 的 Plan A（标准 `--f16`）与 Plan B（真 f16）
两条路线均为活跃计划**——
✅ F4 已落地 = **S1s/S1t**（编译期特化）+ **网格粒度变体**（#f/#ks/#kr 叠加到特化签名）+ **P2-12**（批量派发
＋scan/运行期同源修复）+ **P2-3**（RAPT/CNN 接线）+ **S4/P2-8**（CPU F16C 快路径，见 §4 S4）+
**F5**（全 f16 可训练性，[`24`](24-f16-training-defects.md) §4.6）；
❌ 已评估-不做（**仅按当时提速目标**）= **S0**（前提被实测推翻）、**S1**（净亏 + 质量下降）、
**S2**（f16x2 打包，前提不成立）、**S5/P2-10**（原生 f16 资格扩张，收益有界）——各附实测与上界分析；
🎯 验收锚「任意设备 `--f16` ≤ f32」**已达成**：850M **2005 vs 2038 ms**、40HX **1013 vs 1027 ms**（P2-13，
[`18`](18-roadmap.md) P2-13 行），峰值显存 818 → **606 MiB**。
🟡 **待做 = §8.A / §8.B**：A 路线做标准 `--f16`（PyTorch AMP 对齐）；B 路线做真 f16 的一键预设、
零转换普查、native16 全覆盖、logits/CE 融合、全模型/全设备长跑与转正。
其中 **S1 恢复为目标但需先补 CE backward 扫描**；**S5 反转为核心 R4.1**；S2 仅作 R4.2 的可选性能项。

**已达成（2026-10-13，S1s/S1t）**：850M（**无 `shaderFloat16`**）`--f16` **step 墙钟 2055 ms vs
f32 2037 ms（+0.9%，特化前 +10.6%）**、设备 kernel 1962 vs 1949 ms、**峰值显存 606 vs 818 MiB
（−26%）**；② 项"≤ f32"**尚未完全达成**（差 0.9%，其中 ≈0.6% 的优化器退化已由 P2-12 收口——但
40HX 实测该项的端到端收益在噪声内，见 [`18`](18-roadmap.md) P2-12 行）。
（**后续**：P2-13 的网格粒度变体把残差收掉，同日实测 850M 2005 vs 2038 ms / 40HX 1013 vs 1027 ms，
② 项已于 2026-10-14 达成——见上文进度总览。）
40HX（capable）的对照见 [`24`](24-f16-training-defects.md) §4.5。

> 口径变化（相对 [`24`](24-f16-training-defects.md) §5 F4 原文「40HX 上 ≤ f32」）：① 按用户要求改为
> **「任意设备」**（存储路径）；② **原生 f16 不被排除**——它为 **S5 正式立项**，在 capable 设备上
> 额外验收，但**不允许**用它解释存储路径（S0–S4）的收益。

## 6. 纪律与边界

1. **S0–S4 提速路径不动数值语义**：f32 累加、RNE 落 f16、既有容差口径保持不变；每期以对拍守住。
   §8 的“真 f16”是另一目标（支持设备零 f16↔f32 转换），不受本条“提速路径保持 f32 累加”的限制。
2. **S0–S4 提速路径不引入 loss scaling**：F5 已用 optimizer state scaling 解决 `optimizer=f16` 的下溢。
   loss scaling 是 §8.A A1.1 的标准项；§8.B 的真 f16 可以复用它作为数值缓解手段。
3. **S0–S4 的存储提速路径不依赖原生 f16**：能力探测只允许切换算术风格，任何路径都不放弃 f16 存储。
   但 §8 的**真 f16** 在 capable 设备上把 native16 全覆盖列为硬门禁——两条口径不要混用。
4. **外部参考需复核**：§3 的 ggml/torch 做法按四段式对当时版本源码复核后再引用。
5. **先 S0 再谈提速**：S0 未完成前，任何 S1–S3 的收益都不可信（会被帧同步噪声吞掉）。

## 7. 证据索引

| 来源 | 内容 |
|---|---|
| [`24-f16-training-defects.md`](24-f16-training-defects.md) §4 | F4 复现数据（40HX）+ 候选来源 + §4.4 本机归因 |
| [`24-f16-training-defects.md`](24-f16-training-defects.md) §4.6 / `compute_optimizer.hpp` | F5：四字段全 f16 可训练性（Adam 状态缩放 `S=4096`）与残留边界 |
| `backend/compute_vk_backend_frame.inc` | `begin_batch`/`flush_batch`/`start_frame`（帧槽阻塞等待） |
| `backend/compute_vk_backend.hpp` | `PIPELINE_FRAMES = 6`（环深度） |
| `shaders/elementwise_v2.comp` 等 | f16 变体 = storage-only（`GL_EXT_shader_16bit_storage` + f32 算术） |
| `include/neuralnet.cpp/compute_layer_gpt.hpp` | LM head 计算精度强制 = `stable`（⇒ `--f16` 下 logits 为 f32） |
| `include/neuralnet.cpp/compute_engine.hpp`（`upload_span`） / `compute_gpu_engine.hpp`（`copy_from_impl`） | `fresh_dst` 与 drain 语义（每步上传是否排空） |
| `docs/development/05-mixed-precision.md` §12.5/§12.11 | 边界 cast 代价、op-level f16 GEMM 现状 |
| [`18-roadmap.md`](18-roadmap.md) | P2-8（F16C）/ P2-9（step 内固定 I/O）/ **P2-10（原生 f16 算术用足）** |
| `include/neuralnet.cpp/expr_spec.hpp` | `expr_prec_sig_native16`：native16 结构谓词（排除 fold/matmul/reduce/rparams） |
| `build/<cfg>/generated/fused_registry.hpp` | 注册表头注释：`#x`（运行期精度分派）/ `#s`（全 f16 特化）/ `#xxxx`（扫描签名特化）/ `#a`（native16）等各变体 |

---

## 8. 全流程真 f16 计划（2026-10-18，当前执行口径）

> 本节是 **2026-10-18 恢复的活跃计划**。§4 的 S0/S1/S2/S5 仍是当时实测结论，不删除、不改写；
> 它们与本节的差别是**目标不同**：§4 回答“怎样让 `--f16` 存储路径更快”。本节把后续工作拆成
> **两套计划**：A = 参考 PyTorch 的标准 `--f16` 训练；B = 全流程真 f16（支持设备零转换，
> 不支持时只由引擎在算子内隐式转换）。

### 8.0 两条路线（不可混用）

| 路线 | 精度配方 | 目标 | 转换策略 | 定位 |
|---|---|---|---|---|
| **A 标准 `--f16`** | `profile_f16()` = `{F16,F16,F32,F32}`；可选 `profile_master_weights()` = `{F32,F16,F32,F32}` | 参考 PyTorch AMP：稳定训练/推理，允许 f32 累加与边界 cast | f32 累加；stable/optimizer 保持 f32；loss scaling / master weights 按 A1 落地 | 推荐默认 |
| **B 全流程真 f16** | `profile_all_f16()` = `{F16,F16,F16,F16}` | 支持设备全程零 f16↔f32 转换；不支持设备只由引擎在算子内隐式转换 | capable：native16 全覆盖、零转换；incapable：转换只进 `*_impl` | 实验/性能目标 |

两条路线的验收不能互相替代：A 证明“标准混合精度可用”，B 证明“真 f16 可行”。

> 兼容说明：此前文档里的“§8.3 / §8.4”现分别指 **§8.B.3 / §8.B.4**；“§8”是两套计划的总入口。

### 8.A 标准 `--f16` 训练（参考 PyTorch AMP）

#### A0 口径与参照

- **PyTorch 参照 = AMP/autocast**：f16 计算、f32 master weights（或当前 `--f16` 的 f16 参数）、
  f32 优化器状态、softmax / LayerNorm / loss 等敏感算子走 f32、matmul/归约 f32 累加，
  并用 **GradScaler**（动态 loss scaling）避免梯度下溢。它不是 `model.half()` 的纯 fp16。
- **本仓映射**：`--f16` = `profile_f16()`（f16 参数/激活 + f32 stable/optimizer）；
  更接近 AMP master-weights 的形态是 `profile_master_weights()`（f32 参数 + f16 计算）。
- **默认映射待裁定**：严格 PyTorch AMP 使用 f32 master weights；当前 `--f16` 是 f16 参数。
  A0 先冻结两种候选（保留 `--f16` 兼容形态 + 新增/显式选择 `profile_master_weights()`，
  或把 `--f16` 默认切到 master weights），A2 用 PyTorch AMP 对拍与显存/速度结果决定，避免先改语义。
- **对照脚本**：`bench/train_torch.py --dtype fp16` 当前是 `model.half()` 的**纯 fp16**，
  只作为 B 路线的对照；A 路线要另补 AMP/GradScaler 对照，不能拿纯 fp16 结果冒充 AMP。

#### A1 任务

- **A1.1 动态 loss scaling**：实现 GradScaler 语义——放大 loss、反缩放梯度、检测 inf/NaN 后
  skip step 并回退 scale、按窗口增长；与现有 `--max-norm` 的顺序固定为“反缩放 → 裁剪 → step”。
  当前 F5 的 `optimizer_state_scale_` 只解决 Adam m/v 下溢，**不能替代 loss scaling**。
- **A1.2 master weights（可选但推荐对照）**：让 `profile_master_weights()` 成为可训练路径：
  f32 主权重 + f16 计算副本，优化器更新 f32 主权重后再刷新 f16 计算副本；与 `--f16`
  （参数本身 f16）明确区分。
- **A1.3 数值敏感链固定 f32**：softmax / LayerNorm / RMSNorm / CE 的敏感中间量与 loss 标量保持
  f32；matmul/归约保持 f32 累加。这些是 A 路线的**允许项**，不是 B 路线的验收项。
- **A1.4 CLI/GUI/文档与对照脚本**：`--f16` 帮助明确“标准混合精度”；GUI 预设与 `gui_cli_audit`
  同步；不引入 `--f16-all`（那是 B 路线）；`bench/train_torch.py` 增加 `--dtype amp`
  （`torch.autocast` + `GradScaler`），保留 `--dtype fp16` 作为 `model.half()` 的纯 fp16 对照。

#### A2 验收

- 同数据/同超参/同 seeds：`--f16`（两种 A0 候选映射）与 f32 的 loss 轨迹在冻结容差内，
  无 NaN、无冻结、权重移动；与 PyTorch AMP（同结构、同数据）的 loss 轨迹同量级。
- 动态 loss scaling：构造小梯度/溢出用例，验证 scale 增长、inf 跳过与恢复；不改变正常路径的
  loss 语义。
- 峰值显存低于 f32；step 不慢于 f32（允许在 A1 完成前后分别记录基线）。
- `ctest` / `NN_BIND_DEBUG=1 ctest` 全绿；不要求零转换，也不要求 native16 全覆盖。

### 8.B 全流程真 f16（Plan B）

#### 8.B.1 定义与边界

**全流程真 f16** = `profile_all_f16()` = `{param:F16, compute:F16, stable:F16, optimizer:F16}`
覆盖从张量进入引擎到输出离开引擎的整条计算图：权重/嵌入、激活、梯度、logits、loss 链、
优化器状态与参数更新、训练与推理、`.nnmodel` / `.nnckpt` 往返。模型内部不存在 f32 驻留张量。

按设备能力分两条**都必须成立**的路径：

1. **设备支持 f16 计算（`shaderFloat16` + native16 pipeline）——“B-Arith 严格档”**：
   每个算子都走 native16 变体；kernel 的 load/store、算术、累加、常量与输出全部保持 f16，
   **不允许任何 f16↔f32 转换**。不允许 `cast` / `cast_into`、f32 镜像、f32 临时量，也不允许
   “读 f16 → `float(x)` → f32 累加 → `float16_t(x)`”这种半真 f16。若某算子做不到，它就不满足
   **B-Arith** 的验收门禁，必须补 native16 或给出分块/补偿实现。
2. **设备不支持 f16 计算（无 `shaderFloat16`）——引擎隐式转换**：
   允许 f16↔f32 转换，但只能发生在**算子实现内部**（ComputeEngine 的 NVI / `*_impl` 边界），
   由引擎自动插入；Layer、DSL、CLI、模型代码不得手写 cast，不得产生跨算子的 f32 驻留副本。
   每个算子只允许算子局部的转换临时量，能融合的必须融合。

> **2026-10-18 调研后的口径澄清（B-Mem / B-Arith）**：外部有效的低精度训练方案（随机舍入、
> ECO、Unit Scaling）在论文实现里都允许**优化器内部局部 f32**：SR 的 dithering 会临时升到
> FP32 生成噪声；ECO 用 FP32 accumulation 计算量化残差；Unit Scaling 的 FP16 实验仍把 Adam
> 二阶矩存 FP32。它们消除的是**持久态**（master weights / m / v / 激活 / 模型文件），不是
> 指令级的所有 f32 临时量。因此本计划把“真 f16”拆成两个可分别验收的层次：
>
> - **B-Mem（硬门禁，用户目标）**：训练/推理/checkpoint/model 文件里**没有 f32 持久张量**；
>   参数、梯度、激活、logits、m/v、loss 标量都由 f16（或更窄的补偿表示）承载；模型内部不存在
>   跨算子的 f32 驻留副本。
> - **B-Arith（严格档，capable 设备附加目标）**：连 kernel 内部也不出现 f16↔f32 转换；
>   每个算子登记转换点，按 §8.B.7 的优先级逐步归零。数值补偿 kernel（SR/ECO/attention δ 修正）
>   允许登记为“局部 f32 豁免”，但不得升级为持久张量。
>
> 下面的 R0–R6 默认按 **B-Mem** 验收；B-Arith 继续作为 capable 设备的纯度目标，不再作为
> SR/ECO/Unit Scaling 的入场前提。若维护者坚持把 B-Arith 作为唯一门禁，这些方案必须改写为
> 整数/分块补偿实现，不能直接照搬论文的局部 f32 实现。

**边界**：模型文件 / 张量 I/O 也保持 f16（`.nnmodel` / `.nnckpt`、KVRecord 往返）；数据集里的
token/id 是整数，不是 f16 张量。只有日志 / 调试打印这类宿主展示边界允许 f32，且不得把 f32 张量
重新注入模型内部。CE 的 loss 标量当前恒 F32 是**真 f16 的已知偏差**，R2 必须用 f16 安全归约
（per-token 平均 / 分块缩放）消除，不能把它当作永久例外。

f16x2 打包属于性能项（R4.2），不影响“零转换”的定义；**B-Mem 的硬门禁是持久态无 f32**，
**B-Arith 的门禁是 native16 覆盖与转换计数**。

#### 8.B.2 当前基线（2026-10-18）

| 能力 | 状态 | 证据 / 边界 |
|---|---|---|
| 四字段 profile 可构造 | ✅ | `profile_all_f16()`；`mem_probe --f16-all` |
| `optimizer=f16` 可训练 | ✅ **可训练性** | F5 的 Adam 状态缩放 `S=4096`；tiny GPT 20 步与 GPU 500 步记录见 [`24`](24-f16-training-defects.md) §4.6 |
| `stable=f16` 的 CE 标量 | 🟡 **当前是 F32 补丁** | CE 的 `(1,1)` loss 归约输出恒 F32（[`24`](24-f16-training-defects.md) §3.1）；这与“真 f16”冲突，R2 必须改成 f16 安全归约 |
| 一键 CLI 预设 | ❌ | `text_train` / `mnist_train` 没有 `--f16-all`，只能用 `--f16 --precision-stable f16 --precision-optimizer f16` 拼 |
| 长跑 / 质量 / 多模型 | 🟡 | 已有单模型 500 步记录；RAPT f16 仍有已知 C11 NaN（[`18`](18-roadmap.md) §3.1）；尚未建立与 f32 同窗的质量基线 |
| `--f16` 下 logits 为 f16 | ❌ | LM head 计算精度强制 = `stable`；`--f16` 的 stable=F32 ⇒ logits 仍 f32（全 f16 时才是 f16） |
| 启动期告警 | 🟡 | 仍把 `optimizer=f16` 当作危险组合；R0 改文案，R6 再决定是否保留“显存/质量权衡”提示 |
| capable 设备转换计数 | ❌ | 当前仍有 `#x`/边界 cast/`cast_into` 与 f32 累加路径；真 f16 要求 `cast` 调用 = 0、f32 临时量 = 0 |
| incapable 设备转换位置 | 🟡 | 设计上应只在引擎 `*_impl` 内；尚未建立“Layer/DSL 零 cast”的自动门禁 |
| f16 累加 / 归约 | ❌ | matmul、reduce、fold、CE 仍存在 f32 累加器；真 f16 要求 native16 全链，或用分块/补偿实现且不得回退 f32 累加器 |
| 原生 f16 ALU 覆盖 | ❌ | `#a` 已启用且优先级最高，但结构谓词仍排除 fold/matmul/reduce；真 f16 要求覆盖全部参与计算的结构 |
| 数值鲁棒性机制 | ❌ | 参数写回仍是 RNE；无 SR / GradientStabilizer / ECO；attention `rowsum(W)`/`rowsum(X)` 未测；Unit Scaling / Peri-LN 未实现（§8.B.7） |

#### 8.B.3 分期（R0–R6）

#### R0 — 口径统一（文档先行，代码注释/告警随后）

- **做什么**：把“F5 前不可训练”的旧结论统一改成“F5 后可训练、但仍未产品化”；同步
  `precision.hpp` 的 profile 注释、`precision_profile_test.cpp` 的旧注释、`cli_precision_warning.hpp`
  的风险文案与 `f16_precision_test` 的矩阵标签。**只改文案/标签，不改数值行为**。
- **验收**：非历史文档中不再出现无限定的“全 f16 不可训练”；`rg` 命中的旧表述都能看出是“F5 前”；
  告警测试仍通过，但文案不再把已知可训练的组合描述成必崩。
- **依赖**：无。**本轮已完成文档侧（26/24/18/27/05/21/25/cpp-api/AGENTS/README），代码注释与告警文案是 R0 剩余项。**

#### R1 — 一键预设与产品入口

- **做什么**：给 `text_train` / `mnist_train` 增加 `--f16-all`（语义 = `profile_all_f16()`）；
  `--f16` / `--t1_58` / `--f16-all` 的互斥或覆盖顺序必须显式定义并报错/提示；GUI 两个训练页的
  “精度预设”下拉增加对应项；帮助、`gui_cli_audit.py`、`doc_inventory` 同步。
- **验收**：`--help` 可见；解析结果四槽全 F16；`bench/gui_cli_audit.py` 退出码 0；CLI 负例覆盖
  与 `--f16` 并用的冲突；`mem_probe --f16-all` 与 CLI 预设逐字段一致。
- **边界**：`--f16` 仍是默认推荐的安全配方；`--f16-all` 在 R5 通过前保持“实验/显式选择”定位。

#### R2 — 真 f16 数值与转换收口

- **做什么**：
  1. **转换普查**：在 capable 设备上跑全 f16，统计每次 `cast` / `cast_into` / f32 临时量 / `#x`
     回退，建立“转换点清单”；清单上的每一项都必须被 R4 的 native16 或融合消掉。
  2. **f16 累加 / 归约**：matmul、reduce、fold、softmax / LayerNorm / RMSNorm、CE 链全部改为
     native16 路径；累加器、常量、归一化因子与输出都保持 f16。若直接 f16 累加的误差不可接受，
     用分块、补偿求和或归一化缩放解决，**不得用 f32 累加器冒充真 f16**。
  3. **CE loss 标量**：去掉当前“归约输出恒 F32”的补丁，改 per-token 平均 / 分块缩放，使
     `batch×seq` 级的 loss 求和仍在 f16 可表示范围内。
  4. **优化器**：m/v、参数更新、`S` / `eps` 等常量全部走 f16 可表示形式（或编译期 shader 常量），
     不再依赖 f32 rparam；补齐 `|g|·S` 逼近 65504 的溢出护栏（与 `--max-norm` 的策略需实测后定）。
  5. **数值鲁棒性技术栈**：按 §8.B.7 的优先级接入 stochastic rounding、GradientStabilizer、
     ECO、低精度 attention 的 `δ`/零行和修正；复用 §8.A A1.1 的动态 loss scaling、梯度裁剪、
     分块/补偿求和与逐行/分块缩放。这些技术必须保持 **B-Mem**（无 f32 持久态）；局部 f32
     寄存器只在登记的补偿 kernel 内允许，不能靠引入 f32 张量驻留来“缓解”。
- **验收**：GPT tiny + 真实规模各 3 seeds × ≥500 步，loss 全程有限且下降、权重确实移动、末步 loss
  与 f32 基线偏差在冻结阈值内；`f16_precision_test` 的全 f16 段升级为健康断言；checkpoint 往返
  轨迹连续；**capable 设备转换点清单归零**。
- **依赖**：R0；与 R4 的 native16 补齐并行。

#### R2.5 — 数值鲁棒性技术栈（外部方案落地闸门）

- **定位**：R2 解决“数值表示与转换”，R2.5 解决“表示变窄后训练动力学是否还健康”。它不是
  §8.A 的 loss scaling，也不是 R4 的 native16 覆盖；三者正交，可并行，但所有新机制默认关闭。
- **做什么**：按 §8.B.7 的 P0 → P1 → P1b 顺序落地：
  1. **P0 观测**（§8.B.7.0.1）：权重冻结/ULP、梯度尖峰、激活 65504、注意力零行和、持久态审计；
  2. **P1**（§8.B.7.1 / §8.B.7.2）：SR 参数写回 + GradientStabilizer；
  3. **P1b**（§8.B.7.3）：ECO 的量化误差注入（Adam/AdamW，依赖 SR）；
  4. **P1c**（§8.B.7.4）：attention `δ`/重复最大值修正，按 P0 残差条件触发。
- **验收**：P0 有可机读基线；P1 的单元/小模型/续训门禁全绿；ECO 的玩具与 tiny GPT 对照通过；
  attention 修正只在其触发条件成立时启用；所有开关关闭时 f32/f16 路径逐字节不变。
- **依赖**：P0 无代码依赖（只读探针）；P1 需 R1 的一键预设；ECO 需 SR；P1c 需 R4.1 的
  native16 fold。**Unit Scaling / Peri-LN 不在 R2.5**，见 §8.B.7.5/§8.B.7.6 与 R5.1/R5.2。

#### R3 — 主导张量与融合消费者（恢复 S1 的目标，不沿用旧实现）

- **做什么**：让 logits 在 `compute=f16` 时以 f16 存储，并把 CE 链对它的多次读取融成一次读
  （fold logsumexp + loss + grad 共享同一份 f16 读，kernel 内保持 f16 累加/分块 f16 安全归约）。**前置**：`scan_exprs`
  必须覆盖一次真实的 forward → loss → backward → optimizer，否则 CE 反向结构没有特化变体，会重演
  §4 S1 的“前向省、反向亏”净负结果。
- **验收**：`NN_PREC_TRACE` 的 CE 反向 `[ir-c][degrade]` = 0、CE 链 `cast` 计数 = 0；logits 相关读量下降；全 f16 下
  step 不劣于 `--f16` 的同设备基线（允许先记录“质量换显存”的例外，但不得倒退成旧版净亏）；
  `ce_fusion_test` / `f16_precision_test` 全绿。
- **边界**：`--f16`（stable=F32）是否也把 logits 降为 f16，由本阶段基准单独裁决；在全流程真 f16 下
  logits 本来就应是 f16。

#### R4 — native16 全覆盖（核心门禁）

- **R4.1 覆盖**：把 `#a` 从“按结构谓词少量生成”扩到**所有参与计算的生成结构**，包括
  fold / matmul / reduce / rparams / 常量与 CE 链；每个结构都必须有 native16 变体。旧 S5 的
  “资格不再扩张”在真 f16 目标下**反转**为本阶段的核心任务。
- **R4.2 打包（可选性能项）**：按 S2 评估 f16x2 打包读写；它只影响事务数/吞吐，不影响“零转换”门禁。
- **验收**：**B-Mem** 必达——全 f16 训练/推理/checkpoint 中无 f32 持久张量、无跨算子 f32
  驻留副本；**B-Arith** 按 §8.B.7 的转换点清单逐项登记并归零，`NN_PREC_TRACE` 无 `#x`/边界
  cast 回退。数值补偿 kernel 的局部 f32 若无法消除，必须列入豁免表并给出上界；不得以“豁免”
  名义保留 f32 镜像或跨算子临时量。`NN_VULKAN_NO_16BIT_ALU=1/0` 配对只用于性能 A/B，
  不用于给缺失覆盖找借口。
- **依赖**：R2/R3；f16x2 可并行，但不作为零转换的前置。

#### R5 — 全模型 / 全设备覆盖

- **做什么**：GPT、MLP、CNN、RAPT 四类模型 × CPU / 850M / 40HX × 训练 / 推理全部跑全 f16；
  覆盖 `--activation-offload`、梯度累积、checkpoint 续训、`.nnmodel` 推理、KVRecord round-trip；
  修复 RAPT f16 的 C11 NaN（[`18`](18-roadmap.md) §3.1）。**无 `shaderFloat16` 的设备**验证
  “转换只在引擎 `*_impl` 内”：Layer/DSL/CLI 无手写 cast，转换临时量按算子局部计。
- **架构级数值实验（R5.1/R5.2，不阻塞 R5 基线）**：Unit Scaling 与 Peri-LN 按 §8.B.7.5 /
  §8.B.7.6
  先做模型级 A/B（GPT/Transformer 优先），通过后再另立模型预设；它们是“减少溢出/方差”的
  架构手段，不是 f16 全流程的入场券。默认模型与既有 checkpoint 语义不变。
- **验收**：覆盖矩阵逐格有记录；每格 loss 有限、下降、权重移动；CPU/GPU 同设备轨迹在 f16 容差内；
  `ctest` 与 `NN_BIND_DEBUG=1 ctest` 全绿；RAPT f16 不再需要 CI 排除项；**capable 设备零转换、
  incapable 设备转换点全部落在引擎内部**。
- **依赖**：R2/R3/R4；R4 的 f16x2 子项可并行。

#### R6 — 转正与发布

- **做什么**：R5 通过后，把 `--f16-all` 从“实验”转成受支持配方；同步 `--help`、GUI、release notes、
  05 文档与 AGENTS 的当前状态；决定是否调整启动期警告（警告仍可保留“显存/质量权衡”提示，但不得再写
  “不可训练”）。
- **验收**：用户文档中 `--f16` 与 `--f16-all` 的定位清晰；真 f16 的设备矩阵（capable 零转换 /
  incapable 仅引擎内转换）逐条可查；默认仍是 f32，推荐仍是 `--f16`，`--f16-all` 是显式选择。

#### 8.B.4 验收矩阵

| 维度 | 门禁 |
|---|---|
| 正确性 | f32 路径逐字节不变；全 f16 loss 全程有限、下降、权重移动；checkpoint 往返连续 |
| 数值质量 | 同数据/同超参/同 seeds，与 f32 的末步 loss 偏差在冻结阈值内；f16 累加/归约的误差预算单独记录 |
| **B-Mem 持久态（硬门禁）** | 训练/推理/checkpoint/model 文件中无 f32 持久张量；参数、梯度、激活、logits、m/v、loss 标量均为 f16 或更窄补偿表示；无跨算子 f32 驻留副本 |
| **B-Arith 转换（严格档）** | capable 设备全 f16 训练/推理中 `cast` / `cast_into` 调用 = 0、f32 临时量 = 0、无 `#x` 或边界 cast 回退；native16 覆盖全部生成结构；补偿 kernel 的局部 f32 逐条豁免 |
| **incapable 设备转换位置** | f16↔f32 只发生在 `ComputeEngine` 的 `*_impl` 内；Layer/DSL/CLI/模型代码零手写 cast；无跨算子 f32 驻留副本 |
| 无静默回退 | 模型内部驻留张量均为 F16；`NN_PREC_TRACE` 的 miss 为 0（capable）或全部落在引擎内部（incapable） |
| 覆盖 | GPT/MLP/CNN/RAPT × CPU/850M/40HX × 训练/推理；RAPT f16 C11 关闭 |
| 性能 | 峰值显存低于 `--f16`（若更低精度未带来显存收益则记录原因）；step 不出现无法解释的倒退 |
| 产品面 | `--f16-all` 帮助/GUI/审计闭环；文档不再出现无限定的“不可训练” |

#### 8.B.5 与 §4 旧结论的关系

| 旧结论 | 恢复后的处理 |
|---|---|
| **S0** 去掉 step 内固定 I/O：已评估-不做 | **保持不做**。它服务于提速，且实测无收益；真 f16 不依赖它。 |
| **S1** 主导张量全 f16：实测净亏 | **恢复目标，重做前置**。旧结论的净亏来自 CE 反向未扫描；R3 先补扫描覆盖再复测。 |
| **S2** f16x2 打包：前提不成立 | **降为 R4.2 的可选性能项**。不作为零转换/正确性门禁。 |
| **S5 / P2-10** native16 资格不再扩张 | **反转为核心 R4.1**。旧结论只适用于“以提速为唯一目标”的形态；真 f16 要求 native16 覆盖全部计算结构。 |
| **F5** 四字段全 f16 可训练 | **作为基线保留，但不等于真 f16**。F5 只关闭“能不能训练”（且 CE 标量仍是 F32 补丁）；R2–R5 关闭“零转换、长期、全模型、可发布”。 |

#### 8.B.6 第一刀（建议顺序）

> **Plan A 的第一刀**（可与 B 并行）：A1.1 动态 loss scaling + PyTorch AMP 对照脚本；
> Plan A 不等待 B 的 native16 覆盖。下面是 Plan B 的顺序。

1. **R0 剩余项**：同步 `precision.hpp` / `precision_profile_test.cpp` / `cli_precision_warning.hpp`
   的旧文案与标签（不改行为）。
2. **R1**：`--f16-all` + GUI 预设 + `gui_cli_audit`，先让全 f16 有一个可复现的单一入口。
3. **R2 转换普查 + §8.B.7.0.1 的 P0 探针**：在 40HX（capable）跑全 f16，逐项列出 `cast` / f32
   临时量 / `#x` 回退；同时记录权重冻结率、梯度尖峰、激活超 65504 计数、注意力
   `rowsum(W)`/`rowsum(X)` 残差。P0 只测量、不改行为。
4. **R2.5 第一刀（可与 R4.1 并行）**：SR 参数写回 + GradientStabilizer；两者先在 opt-in
   开关下做单元/小模型 A/B，默认关闭。SR 必须先于 ECO 落地（ECO 的最强保证依赖无偏量化）。
5. **R4.1 覆盖**：按普查清单补 native16（fold / matmul / reduce / CE / rparams），直到 capable
   设备的 B-Arith 转换点归零；同时冻结 f16 累加误差预算。
6. **R2.5 第二刀**：ECO（Adam/AdamW）+ 低精度 attention 的 `δ`/零行和修正；ECO 依赖 SR，
   attention 修正依赖 native16 fold 的 f16 累加路径（存储-only/f32 累加时通常观察不到该偏差）。
7. **R3 前置**：让 `scan_exprs` 覆盖 CE backward，再复测 S1 的净收益。
8. **R5/R6**：全模型/全设备长跑与转正；Unit Scaling / Peri-LN 作为 R5.1/R5.2 的架构实验，
   通过后再决定是否进入默认配方。

### 8.B.7 数值鲁棒性技术栈（2026-10-18 外部调研）

> **结论先行**：随机舍入（SR）、GradientStabilizer、ECO 与低精度 attention 的 `δ`/零行和
> 修正都有可核查的原文与实验证据，**列入 Plan B 的 R2.5**；Unit Scaling 与 Peri-LN 的证据也
> 成立，但它们是**模型级改造**，列入 R5.1/R5.2 的架构实验，不作为 R1–R4 的前置。LuminaV
> 目前只有 HuggingFace 模型卡（无论文正文、无代码、无实验），**不作为证据**；其 SR 主张由
> Ozkara et al. 等独立文献支持。
>
> 纪律：外部结论按“结论 / 证据 / 重跑 / 边界”四段式引用；本节所有“有效”均指**外部证据**
> 支持，是否在本仓有效必须先过 §8.B.7.0.1 的 P0 探针。所有新机制默认关闭，f32 路径逐位不变。

| 方案 | 判断 | 处置 |
|---|---|---|
| 随机舍入（SR） | **有效，采纳** | R2.5.1；参数写回 opt-in，确定性 hash，先过冻结/无偏/续训测试 |
| GradientStabilizer | **有效，采纳** | R2.5.1；opt-in，复用全局 norm，记录首步 `ρ0` |
| ECO | **有效，条件采纳** | R2.5.2；依赖 SR；当前全 f16 的收益是更新保真度，不是额外省显存 |
| 低精度 attention `δ` 修复 | **有效，条件采纳** | R4.3；先测 `rowsum(W)`/`rowsum(X)`，按残差决定是否改 fold |
| Unit Scaling | **有效，但模型级** | R5.1 架构实验；不阻塞 R1–R4，不作为“去掉优化器状态”的证据 |
| Peri-LN | **有效，但模型级** | R5.2 架构实验；先测激活是否逼近 65504 |
| LuminaV | **无法核验** | 不采纳为证据；SR 部分改引 Ozkara et al. |

#### 8.B.7.0 证据分级

| 级别 | 含义 | 本节对应 |
|---|---|---|
| **A** | 同行评审 / 多规模实验，结论可直接支撑工程决策 | SR（AISTATS 2025）、GradientStabilizer（ICML 2026）、Unit Scaling（ICML 2023）、Peri-LN（ICML 2025） |
| **B** | 预印本 + 受控实验，方向明确但需本仓复现 | ECO（2026）、Qiu & Yao 的 FlashAttention 失败分析（ICLR 2026）、Delta-Matching（2026） |
| **C** | 灰色材料 / 无正文 / 无实验，不能单独作为依据 | LuminaV 模型卡 |

#### 8.B.7.0.1 P0 观测探针（先量后改）

五个探针全部走 `NN_ENABLE_PROBES` 编译门，默认构建零开销、零行为改变：

1. **权重冻结/ULP**：按参数元素统计“本步 `|Δp|` / 该元素 f16 ULP”的分布；记录
   `|Δp| < 0.5 ULP` 的比例、连续 N 步未移动的参数数、以及 RNE 与 SR 的期望漂移差。
2. **梯度尖峰**：记录全局 `||g||`、`max|g|`、相邻步的范数比；统计 `R_t / R_{t-1}` 的尾部分布，
   用于判断 GradientStabilizer 的触发频率与首步 `ρ0` 风险。
3. **激活范围**：逐层记录 top-100 `|activation|` 与 `>65504` 的计数（§8.B.7.6 的 Peri-LN
   前置证据）。
4. **注意力零行和**：记录每层/每头的 `max |rowsum(W)-1|`、`max |rowsum(X)|`、重复行最大值
   频率与 `P̄==1` 频率（§8.B.7.4 的修复触发条件）。
5. **持久态审计**：在训练/推理/checkpoint 写出点扫描所有驻留 Tensor 的 `precision()` 与
   字节数，列出任何 f32 持久张量；这是 B-Mem 的直接门禁。

每个探针输出一行可机读记录（CSV/JSON），至少覆盖 GPT tiny + 真实规模、CPU/850M/40HX；
没有基线记录的优化项不得进入默认路径。

#### 8.B.7.1 方案一：随机舍入（SR）——对抗权重冻结

**结论**：SR 是无偏量化。若 `x` 位于相邻 f16 网格点 `a < b` 之间，则以
`p = (x-a)/(b-a)` 选 `b`、以 `1-p` 选 `a`，于是 `E[SR(x)] = x`。当更新量 `|η·g|` 小于半个
ULP 时，RNE 会把它舍成 0；SR 以 `|δ|/ULP` 的概率保留一个 ULP 的跳动，期望上仍等于
`δ`，因此不会像 RNE 那样把微小更新永久吃掉。代价是方差（约 `ULP²·p(1-p)`），不是零误差。

**证据**：

- Ozkara, Yu, Park, *Stochastic Rounding for LLM Training: Theory and Practice*（AISTATS 2025，
  arXiv:2502.20566）：**全 BF16**（含优化器状态）+ 只在最终更新步做 SR，最大 6.7B 模型；
  相对 `(BF16, FP32)` 混合精度，验证困惑度更好、吞吐最高 **1.54×**、显存最高 **30%** 更少。
  论文明确说 SR 的 dithering 实现会**临时升到 FP32** 加噪声，再移位截断；临时量不在前向峰值内。
- ECO（arXiv:2601.22101）把 SR 作为最强保证的前提；其 Limitations 明确：只有 RNE 的设备上
  ECO 仍优于 naive，但噪声底更高。
- 早期理论（arXiv:2202.12276）给出低精度梯度下降因 RNE 停滞、SR 消除偏差的分析；同时指出
  SR 仍可能因学习率过小而停滞（SR 不是“任何步长都必然移动”的魔法）。

**本仓落点**：`include/neuralnet.cpp/compute_optimizer.hpp` 的 Adam/AdamW K3 参数写回；
当前是 `delta`（`upd_prec=F16`）与 `p += delta` 的融合，需要在写回处增加一个
**f16 随机舍入**入口。建议新增引擎/DSL 级 `f16_stochastic_round`（或 `compute_into` 的
`RoundMode::Stochastic`），内部用局部 f32 计算精确和与 ULP 概率，再用整数位操作落到 f16；
明确登记为 B-Arith 的局部豁免。

**实现约束**：

1. **只作用于参数写回**（以及经 A/B 证明需要的 m/v），不默认作用于所有激活；否则方差会扩散到
   整张计算图。
2. **确定性**：用 counter-based hash `hash(seed, step, param_id, element_index)` 产生随机位，
   禁止热路径 `std::random_device`/全局 RNG；CPU 与 GPU 必须产生**同一比特序列**。
3. **续训**：`seed` 与舍入模式写入 `TrainState.recipe`；步数已存在，因此 resume 不需要额外
   随机状态缓冲。换 seed 必须显式改变训练语义并写入 checkpoint。
4. **边界值**：±0、次正规、溢出（±inf）、NaN、恰好落在网格中点（tie）逐条测试；负数的
   相邻网格点按数值序而不是位模式序处理。
5. **开关**：`--stochastic-rounding`（或 `--sr`）默认关闭；`--f16-all --stochastic-rounding`
   才是“无主权重 + 无偏更新”的实验配方。

**验收**：

- 纯 CPU 逐位测试：给定 seed/step/索引，SR 序列与参考实现逐位一致；大样本均值误差在
  `3·ULP/√N` 量级内（记录 `N`）。
- 冻结测试：构造 `|δ| < 0.5 ULP` 的 f16 权重，RNE 在 N 步后不动，SR 在 N 步后必须出现移动；
  记录移动概率与期望漂移。
- 续训测试：save → load → 继续，参数与 SR 序列逐位一致；CPU/GPU 同 seed 同序列。
- 回归：默认关闭时 f32/f16 字节锚逐位不变；开启后 loss 有限、无 NaN、显存不增加。

**边界**：Ozkara 的实验是 BF16（8 位指数），不是 FP16；FP16 的动态范围更窄，SR 只解决
**舍入偏差**，不解决溢出。SR 的方差还可能让极小的学习率更抖；必须与本仓的 Adam 状态缩放
`S=4096`、GradientStabilizer / max-norm 一起做 A/B。

#### 8.B.7.2 方案二：GradientStabilizer——驯服梯度尖峰

**结论**：保留梯度方向，替换幅度。令 `R_t = ||g_t||₂`，维护
`m_R = γ1 m_R + (1-γ1)R`、`v_R = γ2 v_R + (1-γ2)R²`，取
`ρ = m_R / sqrt(v_R)`，输出 `g̃ = ρ · g/R`。论文证明尖峰步的 `||g̃||` 有一致上界，且
Adam/AMSGrad 的矩状态不会因原始尖峰无限增长；阈值不需要手调。

**证据**：Huang et al., *GradientStabilizer: Fix the Norm, Not the Gradient*（ICML 2026，
arXiv:2502.17055）。LLaMA-130M/350M、FP16/FP4、ImageNet、RL、时间序列；FP16 预训练中
AdamW 的 PPL 从 24.31→23.14（130M）、19.21→17.80（350M），FP4 也从 28.72→26.66、
21.35→18.84；固定 `γ1=0.6, γ2=0.999`。论文同时报告它降低 Adam 对 weight decay 的敏感性。

**本仓落点**：`Optimizer::clip_grad_norm` 已经实现“全局 L2 范数 + 一次下载 + 等比例缩放”，
可复用同一段 norm 计算，新增 `Optimizer::stabilize_grad_norm(γ1, γ2)`；状态 `m_R/v_R` 是
**每优化器一个标量**，随 `TrainState` 保存即可。建议顺序：
`loss unscale → GradientStabilizer → 可选 max_norm → step`。max_norm 保留为最后一道安全阀，
不再承担“尖峰抑制”的主职责。

**实现警告（必须记录）**：按论文默认值，第一步
`ρ0 = (1-γ1)/sqrt(1-γ2) = 0.4/0.03162 ≈ 12.65`，即第一步会把单位方向放大 12.65 倍。论文
没有显式 bias correction；本仓 P0 必须测这个首步幅度是否可接受。若不可接受，优先试
“首步用原始梯度 / 对 m_R、v_R 做偏置修正 / 初始化 v_R 为 R²”，并把偏离论文的改动写进
开关文档，不能静默改默认。

**验收**：

- 合成尖峰测试：在平稳梯度流中注入 `10×/100×/1000×` 的尖峰，断言 `||g̃||` 不随尖峰幅度
  增长（在记录容差内），且方向余弦接近 1。
- FP16 小 GPT：与 `--max-norm`、无稳定器三组 A/B；loss 有限、下降、无 NaN；首步参数移动
  与 `ρ0` 的记录一致。
- 续训：`m_R/v_R` 保存/恢复后轨迹连续；关闭时 f32/f16 路径逐位不变。

#### 8.B.7.3 方案三：ECO——把量化误差注入动量

**结论**：ECO 在参数更新后把“未量化的更新结果”与“量化后的 f16 参数”之间的误差
`e = p_tilde - q(p_tilde)` 注入一阶矩，形成误差反馈；不需要额外持久缓冲。对 Adam，
注入强度用该元素的**有效步长倒数**（论文 Algorithm 3）；对 SGDM 用
`m += (1/η)(1 - 1/β)e` 的等价形式。论文证明在标准假设与衰减学习率下收敛到常数半径邻域，
而 naive 去主权重会随学习率衰减出现反比误差。

**证据**：Nikdan, Zandieh, Alistarh, Mirrokni, *ECO: Quantized Training without Full-Precision
Master Weights*（arXiv:2601.22101，2026-01）。30M–800M scaling law、Gemma-3 1B、2.1B SMoE、
DeepSeek-MoE-16B INT4；FP8/INT4 下接近 master-weight 基线；静态显存最多降 **25%**
（其 SMoE 口径是 12→9 bytes/param，不是“任何模型都省 25%”）。**最强结果依赖 SR**；
纯 RNE 时仍有改善但噪声底更高。

**与本仓的关系（重要）**：当前 `profile_all_f16()` 已经是 f16 参数 + f16 m/v，**没有 f32
master 可删**；因此 ECO 在本仓的即时收益不是“再省 25% 显存”，而是**改善 f16 原地更新的
保真度**（把被舍入吃掉的更新记到动量里）。若将来重新引入 f32 master / 混合 master 形态，
ECO 才兑现“去掉 master”的显存收益。

**落点**：`Adam::adam_update_` 的 K3 需要从“`delta` 直接写回”改为：
1. 在局部更宽累加器里算 `p_tilde = p + delta`；
2. `p_hat = q(p_tilde)`（f16 RNE 或 SR）；
3. `e = p_tilde - p_hat`；
4. 把 `e` 按 Adam 有效步长倒数注入 `m_[i]`（不新增持久张量）。

这要求 K3 拆分或新增一个融合 kernel；`e` 只活在一个 kernel 内，不能物化成跨算子 f32 张量。
ECO 先只接 Adam/AdamW；SGD/Muon 后续按同一框架扩展。

**验收**：

- 玩具二次型：ECO 的驻留误差随学习率衰减保持有界；naive 去 master 的误差随 `1/η` 增长
  （复现论文的定性图）。
- tiny GPT + 真实规模：与 f16 master-weight 基线、naive f16 三组对照；loss 有限、末步偏差
  在冻结阈值内、m/v 无 NaN、显存不增加。
- checkpoint：`m/v` 与 `t` round-trip 后轨迹连续；ECO 不需要额外持久缓冲。
- 依赖：SR 先落地；ECO + RNE 只能作为对照，不作为默认组合。

#### 8.B.7.4 方案四：低精度 attention 的 `δ` 与零行和修复

**结论**：低精度 FlashAttention 的一类失败不是“整体精度不够”，而是反向的
`δ_i = Σ_j P_ij dP_ij` 与 `P`/`dP` 的量化不一致，破坏了 softmax 梯度的零行和
`Σ_j dS_ij = 0`。误差会被注意力中**涌现的低秩结构**反复放大，形成有方向的权重更新，
最终 loss 爆炸。修复不必把整个 attention 提升到高精度：只要让 `δ` 与实际参与乘法的
`P`/`dP` 一致，或避免重复行最大值导致 `P̄=1` 的偏置舍入，就能恢复稳定性。

**证据**：

- Qiu & Yao, *Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention*
  （ICLR 2026，arXiv:2510.04212）：BF16、GPT-2S 600K steps、GPT-2M 100K steps；定位到
  低精度 `O` 使 `δ` 有偏，与低秩表示共同造成 loss 爆炸。最小修复 = 当一行有多个相同最大值时
  动态调整 safe-softmax 的平移量（`β∈[2,8]`，实验中 `β=2`），保证 `P̄<1`；另一种等价诊断
  是反向重算高精度 `O`。**注意：这不是“把 O 永久升 f32”，而是针对重复最大值的条件修正。**
- Delta-Matching（arXiv:2609.37852，2026-09）把同一类问题推广到 FP8 attention，显式证明
  zero-row-sum invariant，并报告 569M/1.67B/5.29B、30B tokens 的对照；它用 FP32 概率与
  FP8 操作数的 FP32 乘积做修正，属于同一“让 δ 与乘法一致”的思路。该文是 FP8 场景，
  对本仓是**佐证与测试口径**，不是直接移植目标。

**本仓现状（先别急着改）**：

- 训练前向走 `make_fold_attn_o`（online softmax，fold 的 `vecacc` 累加器当前是 **f32**），
  反向走 `recompute_W_` + `R = rowsum(W * grad_A)`、`X = W * (grad_A - row_R)`；它**没有**
  使用 `δ = rowsum(dO * O)` 的 saved-output 捷径，因此论文里的 stale-δ 路径不是逐字相同。
- 但反向的 `W` 以 `p_.compute` 落盘；在 `--f16-all` 下 `W` 是 f16，`rowsum(W)` 未必精确为
  1。若 `rowsum(W) ≠ 1`，`X` 的行和仍会偏离 0——这是本仓真正要测的量。

**P0 探针**（不改变行为，`NN_ENABLE_PROBES` 门控）：

1. 对每层/每头记录 `max |rowsum(W) - 1|` 与 `max |rowsum(X)|`；
2. 记录每行“重复最大值”的频率、是否跨 fold block、以及 `P̄` 是否出现精确 1；
3. 记录 f16 累加路径与 f32 累加路径的上述残差比。

**修复选择**（按探针结果触发，不默认全开）：

- **重复最大值局部修正**：在 `make_fold_attn_o` 的 safe-softmax 段增加条件谓词
  `rs = rowsum(rm - S <= eps)`；`rm>0 && rs>1` 时用 `β·rm`，`rm<0 && rs>1` 时用 0。
  由于 fold 是分块的，跨块重复最大值需要预扫描或保守处理，不能直接把论文的逐块算法照搬。
- **δ 一致性修正**：若 `rowsum(W)` 偏差是主因，优先让 `δ` 用与 `W` 同一份量化概率/乘积，
  或对 `W` 做行归一化；只有在局部 f32 豁免下才考虑高精度 `P`/`O` 路径。
- **Delta-Matching 式修正**：仅当本仓未来接入 FP8/更低精度 attention 时再立项；FP16 下先用
  零行和测试判断是否必要。

**验收**：`rowsum(W)→1`、`rowsum(X)→0` 的残差降到与 f16 舍入预算一致；重复最大值回归
（构造一行多个相同最大值）不出现 loss 爆炸；GPT 长跑 3 seeds 无尖峰；`ce_fusion_test` /
`f16_precision_test` / attention CPU-GPU 对拍全绿；存储-only/f32 累加路径的字节锚不变。

#### 8.B.7.5 方案五：Unit Scaling（模型级）

**结论**：Unit Scaling 通过在计算图中按算子性质插入缩放因子，使初始化时权重、激活、梯度
都接近单位方差；矩阵乘法的 `m` 倍方差放大、残差累加、softmax/归一化等分别用对应的
`1/√m`、`1/√2` 等因子抵消。它让模型在 FP16/FP8 的“甜点区”内训练，减少对 loss scaling
的依赖。

**证据**：Blake, Orr, Luschi, *Unit Scaling: Out-of-the-Box Low-Precision Training*
（ICML 2023，arXiv:2303.11257）。BERT-Base/Large 首次在 FP16 **无 loss scaling** 下训练；
BERT-Large FP16 SQuAD v1.1 EM/F1 = 85.67/92.14，优于 loss-scaling 的 84.23/90.93；
FLOPs 仅增加 **0.2%**（BERT-Large 的理论估算）。**边界**：其 FP16 实验仍把 Adam 二阶矩
存 FP32（论文附录 J.1 明说）；Unit Scaling 降低动态范围压力，不自动消除优化器状态。

**本仓落点**：这是模型级改造，不是 CLI 开关。需要：

1. 在 GPT/Transformer 的计算图上定义每条边/每个算子的缩放因子（embedding、Q/K/V、attention
   logits、softmax、residual、FFN、LM head、loss）；
2. 初始化方差与学习率按缩放因子重标定；
3. 与现有 `PrecisionProfile` 正交：Unit Scaling 决定数值尺度，profile 决定存储/计算精度；
4. 只作为 `--unit-scaling` 的显式模型变体，默认关闭，旧 checkpoint 语义不变。

**验收**：同数据/同超参/同 seeds，unit-scaled GPT 与基线 loss 同量级；FP16 无 loss scaling
可训练；峰值显存/step 不劣化到不可接受；`model_serialization` round-trip 保留缩放配置。
先做 GPT/Transformer，CNN/MLP 后续；不要把它作为 R4.1 零转换的前置。

#### 8.B.7.6 方案六：Peri-LN（模型级）

**结论**：Pre-LN 的主残差路径不归一化，隐藏状态方差会随深度/训练增长，3.2B 实验中
Pre-LN 的 top-100 激活在 **0.5B tokens 左右就超过 FP16 上限 65504**；Peri-LN 在每个子层
输出再归一化一次（`x + Norm(Module(Norm(x)))`），把方差增长压住，从而降低 FP16 溢出和
梯度尖峰风险。

**证据**：Kim et al., *Peri-LN: Revisiting Normalization Layer in the Transformer Architecture*
（ICML 2025，arXiv:2502.02732）；400M/1.5B/3.2B、30B tokens，Pre-LN 在多种子/学习率下
出现尖峰或发散，Peri-LN 更稳。**边界**：论文主要是方差/激活幅度分析与 BF16/FP32 训练，
不是“FP16 全流程训练”的完整复现；其 FP16 结论来自对激活上限的测量与 OPT 的外部报告，
本仓仍需自己跑 FP16 A/B。

**本仓落点**：`GPTBlock` / `TransformerEncoderLayer` / `RAPTBlock` 当前是 Pre-LN 骨架；
Peri-LN 需要为 attention 与 FFN 各增加一个 **output norm**（新增 `NormLayout::{Pre, Peri}`
或等价字段，不复用只服务 CNN/ViT 的 `NormPlace`）。这会增加参数与计算，且改变参数布局，
必须新模型/新 checkpoint 语义。

**验收**：先跑 §8.B.7.0.1 的激活幅度探针，确认当前 GPT 的 top-100 激活是否逼近 65504；
只有确认存在溢出风险才进入实现。A/B 至少覆盖 400M 级 GPT、同 seeds、同 token 数；
loss 不劣于 Pre-LN，激活最大值有显著裕量，step 开销记录在案；旧模型文件不受影响。

#### 8.B.7.7 LuminaV：不作为证据

检索到的 `LuminaV-Optimizer-Paper` 只是 HuggingFace 上的模型卡/DOI
（`https://huggingface.co/cloverx-id/LuminaV-Optimizer-Paper`，
DOI `10.57967/hf/10365`）：仓库没有论文正文/代码/实验，README 只写“新版本即将发布”，
作者与机构信息也不完整。它声称的“片上按位随机舍入”无法核验。**结论**：不把 LuminaV 写入
证据链；SR 的有效性改用 Ozkara et al.（AISTATS 2025）、ECO（2026）与经典 SR 理论支撑。

#### 8.B.7.8 分期、依赖与默认值

| 优先级 | 项目 | 计划位置 | 依赖 | 默认 |
|---|---|---|---|---|
| **P0** | 权重冻结/ULP、梯度尖峰、激活 65504、注意力零行和、持久态审计 | R2.5.0 / §8.B.7.0.1 | R1；探针构建 | 仅探针开启 |
| **P1** | SR 参数写回 | R2.5.1 | P0；确定性 hash | 关闭 |
| **P1** | GradientStabilizer | R2.5.1 | P0；全局 norm 复用 | 关闭 |
| **P1b** | ECO（Adam/AdamW） | R2.5.2 | SR 已落地 | 关闭 |
| **P1c** | attention `δ`/重复最大值修正 | R4.3 | R4.1 native16 fold；P0 残差 | 条件触发 |
| **P2** | Unit Scaling | R5.1 | R2/R3/R4；模型级图缩放 | 模型变体 |
| **P2** | Peri-LN | R5.2 | 激活探针；模型级改造 | 模型变体 |

**组合顺序**：

1. 先测后改：P0 的五个探针不改变任何数值行为；没有探针证据的项不进入默认路径。
2. SR 先于 ECO：ECO 的最强理论/实验保证依赖 SR 的无偏性。
3. GradientStabilizer 与 SR 可并行：前者改梯度幅度，后者改参数写回舍入，作用点不冲突；
   组合顺序固定为 `unscale → stabilizer → max_norm → optimizer(SR/ECO)`。
4. attention 修正放在 native16 fold 之后：当前 fold 的 `vecacc` 是 f32 累加，重复最大值
   偏置被大幅掩盖；只有在 f16 累加路径上才可能成为主因。
5. Unit Scaling / Peri-LN 最后做：它们改变模型与参数布局，不能与 R2.5 的数值开关混为一谈。

**与现有验收的关系**：R2.5 不降低 R2/R4 的门禁；它只提供“在 B-Mem 下把 loss 拉回基线”的
手段。所有开关默认关闭时，f32 路径逐字节不变；`--f16-all` 默认仍是不带 SR/ECO/稳定器的
基线，只有显式组合才进入“数值鲁棒性”配方。

#### 8.B.7.9 参考（外部文献）

| 文献 | 链接 | 用于本节 |
|---|---|---|
| Ozkara et al., Stochastic Rounding for LLM Training (AISTATS 2025) | https://arxiv.org/abs/2502.20566 | SR 的全 BF16 证据、吞吐/显存数字 |
| Nikdan et al., ECO (2026) | https://arxiv.org/abs/2601.22101 | 无 master weight 的误差反馈、25% 显存口径 |
| Huang et al., GradientStabilizer (ICML 2026) | https://arxiv.org/abs/2502.17055 | 梯度范数稳定化、FP16/FP4 PPL |
| Blake et al., Unit Scaling (ICML 2023) | https://arxiv.org/abs/2303.11257 | BERT FP16/FP8 无 loss scaling、0.2% FLOPs |
| Kim et al., Peri-LN (ICML 2025) | https://arxiv.org/abs/2502.02732 | Pre-LN 方差增长、FP16 上限 |
| Qiu & Yao, Why Low-Precision Transformer Training Fails (ICLR 2026) | https://arxiv.org/abs/2510.04212 | attention `δ`、低秩放大、重复最大值修正 |
| Tang et al., Delta-Matching (2026) | https://arxiv.org/abs/2609.37852 | zero-row-sum invariant、FP8 attention 佐证 |
| LuminaV 模型卡（灰色材料） | https://huggingface.co/cloverx-id/LuminaV-Optimizer-Paper | 仅记录“无法核验”，不作为证据 |
