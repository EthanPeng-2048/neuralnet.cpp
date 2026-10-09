# 三值（1.58-bit）量化权重与 BitLinear 设计

> **状态**：**设计定稿**（2026-10-09 起草；2026-10-10 裁定尺度 / 打包 / 命名三项）。
> **P1 未实施**——本文是实现 P1 的唯一依据：§4 是设计，§7 是文件级实施清单，§5 是分期与验收。
> 仍有 3 项待批（D3 复合层 profile 下传、D5 `BitLinear` 接入位置、D6 行步长对齐），
> 本文均已给出**推荐做法与理由**（§4.3 / §4.8 / §6），实施前确认即可。
> **落地分支**：`dev/t1_58`。
> **关联**：`05-mixed-precision.md`（Precision 语义契约 §7.2）、
> `17-unified-tensor-engine.md` §5 M7（存储多态——本文给出它的第一个触发条件）、
> `18-roadmap.md` §1.1 立项三问 + §3.1 C2/C3（"可选但静默失效"缺陷类）、
> `20-custom-layer-ergonomics.md`（自定义层的四条接入契约）。
> **参考实现**：`~/codes/Lumina-Engine`（llama.cpp b10991 分支）的 `GGML_TYPE_TQ1_0`。
> 本文 §3 是该格式与内核的完整拆解；§4 记录**哪些照搬、哪些必须重选**。
> **前置证据**：`research/ternary_ste/REPORT.md`（CPU 端三值/二值 STE 训练实测；两者都能把
> 教师权重模式 100% 还原，说明 P1 的算法路径无障碍）；
> `research/ternary_scale/REPORT.md`（**D1 实测**：三个真实模型上 absmean vs absmax × 五种粒度的
> 零化率 / 重构误差 / 信息熵——absmax 只有 0.13~1.29 bit，absmean 命中 `log2 3 = 1.585 bit`）。

---

## 0. 裁决记录（2026-10-09 起草；2026-10-10 尺度 / 打包 / 命名裁定）

| # | 议题 | 裁决 |
|---|------|------|
| 1 | 三值是否进 `Precision` 枚举 | **进**。枚举是"精度词汇表"，三值应当在其中（`Precision::T1_58`；C++ 标识符不能以数字开头） |
| 2 | 其他模型能否选三值 | **能选**。效果差是使用者自己的事，不靠文档劝阻、不靠运行期拒绝 |
| 3 | 那如何避免"可选但静默失效" | **Layer 声明支持的精度集合**：不兼容组合 = **构建期报错**，而不是静默降级（这正是 18 §3.1 C2/C3 登记的两类缺陷） |
| 4 | `BitLinear` | 新增层，声明**只支持** `param=T1_58`；量化进 forward，STE 进 backward |
| 5 | 其余层 | 默认声明支持 `{f16, f32}`（= 现状），**一行代码都不用改** |
| 6 | 声明粒度 | **按槽**（param / compute / stable / optimizer）分别声明，不是扁平集合（§4.2，R1） |
| 7 | P1 是否做打包存储 | **不做**。P1 的量化缓冲用 f16（`{-1,0,1}` 在 f16 下精确可表示），只交付架构统一与功能正确 |
| 8 | P1 里 `T1_58` 能否建张量 | **不能**。`check_precision_supported(T1_58)` 在 P1 明确拒绝；此时它只是 **profile 声明**，不是存储标签 |
| 9 | 性能 | P2（打包存储 + 专用 GEMM）才有；P3 才是激活量化 |
| 10 | 本轮范围 | 只写本文 + `research/ternary_ste/`（前置证据）；不改库代码 |
| 11 | 实测机器 | D1 实测跑在 `ethan@192.168.1.102`（Windows + LLVM clang 23，比本机稳）；证据归档 `research/ternary_scale/` |
| 12 | **D1 统计量** | **absmean**（`τ = 组内 mean\|w\|`）。实测 absmax 把 64%–98% 的权重归零、信息熵掉到 0.13–1.29 bit/权重；absmean 命中 `log2 3 = 1.585` 上限 |
| 13 | **D1 粒度** | **per-row（输出通道）**。实测 absmean 下 `tensor/row/blk256/blk128/blk64` 的质量相差不到 2 个百分点 → **粒度按工程约束挑**；per-row 还避开了 `TQ1_0` 的"行宽必须整除 256"约束 |
| 14 | **D2 打包** | **照抄 `TQ1_0` 的编码与解码**（base-3 5-per-byte + `qh` 尾巴 + 定点 `(s·3^n·3)>>8`），**但块粒度 = 行、尺度 = absmean**；不采用它的 per-256 fp16 absmax 尺度 |
| 15 | **D4 命名** | 枚举 `Precision::T1_58`、`precision_name` = `"t1_58"`、CLI 词法 `t1_58`（**无别名**）、序列化 tag 4、层名保持 **`BitLinear`** |

---

## 1. 一句话

三值权重在本项目分三条腿落地：

1. `Precision::T1_58` 进枚举 —— **词汇表**统一（"这个模型用什么精度"有了唯一说法）；
2. 每个 `Layer` 声明自己支持的精度槽位，框架在 `init` 时**构建期**拒绝不兼容组合；
3. `BitLinear` 把三值量化做进 `forward`、把 STE 做进 `backward`。

P1 只交付 1/2/3 的架构与正确性，**不交付性能**。性能在 P2。

**非目标**（本文明确不做）：三值作为 `compute` / `stable` 槽的取值；量化激活（P3）；
打包存储与专用内核（P2）；改动 f16/f32 的任何既有路径；CUDA（本项目无此后端）。

---

## 2. 现状缺口：为什么需要"Layer 声明支持精度"

### 2.1 `Precision` 现在的语义是一元的舍入契约

`05-mixed-precision.md` §7.2 的形式化定义：**P 精度算术 := 以 f32 参考精度计算 + 每个算子输出舍入到 P**；
matmul / 归约类额外规定累加精度 = `max(P, f32)` = f32。全精度系统（提升序 `max_precision`、
`precision_bytes`、`cast`、`expr_prec_sig_*` 的 1 bit/操作数 ABI）都建立在这条定义上。

三值不满足这条定义：量化到 `{-1,0,+1}` **需要一个尺度 γ**，而 γ 是整张张量的统计量，
`Precision` 枚举里没有地方承载它。所以"三值精度"必须同时引入**谁能声明它**的规则——
这就是 §4.2 的 `precision_support()`。

### 2.2 现状是"任何 profile × 任何模型"都能组合

今天的 `PrecisionProfile` 四个字段对全部模型、全部层一视同仁：CLI 的
`--precision-param/compute/stable/optimizer <f16|f32>` 在 `mnist_train` 与 `text_train` 各 4 个 flag、
`gui.py` 的 `PRECISION_OPTIONS` 与 5 个控件、以及 `bench/gui_cli_audit.py` 的审计基线里一致展开。
`Precision::` 在 **43 个文件里出现 788 次**。

于是"加一个枚举值"= 让**所有模型的所有槽位**都多出一个可选项。这个可选项当且仅当层真的实现了它才有意义,
否则就是 18 §3.1 登记的缺陷类（C2："用户可组合出必崩配置，且无任何启动期提示"；C3："静默回落到另一个实现，
无告警无日志"）。

### 2.3 结论

把"能不能选"交给**层自己声明**，由框架在 `init` 时校验。这样：

- 加枚举值不再等于"到处多一个坑"——它只对声明支持的层生效；
- 读到不兼容组合时是**构建期/初始化期报错**，带层名、槽位、取值；
- 默认声明 = `{f16, f32}`，既有层的语义与代码**零变化**。

---

## 3. 参考实现：Lumina-Engine 的三值权重设计

`Lumina-Engine` 是 llama.cpp 的一个分支，为 **Bonsai 系列 1-bit 模型（GGUF `Q1_0`）** 做纯 CPU 推理优化。
它的优化只覆盖 `Q1_0`（二值）；仓库内同时保留了 llama.cpp 上游的 **`TQ1_0`（三值）** 与 `TQ2_0`。
**`TQ1_0` 就是我们说的 1.58-bit 权重格式**，且是"格式 + 量化 + 反量化 + 点积内核"全套在树内的参考实现。

### 3.1 三种格式对照

| 类型 | 取值 | 块大小 | 块布局 | 位宽 | 尺度 |
|---|---|---|---|---|---|
| `Q1_0`（二值） | `{-1,+1}` | 128 | `fp16 d` + `u8 qs[16]`（符号位） | **1.125 bpw** | per-128 absmax |
| **`TQ1_0`（三值）** | `{-1,0,+1}` | 256 | `u8 qs[48]` + `u8 qh[4]` + `fp16 d` | **1.6875 bpw** | per-256 absmax |
| `TQ2_0`（三值） | `{-1,0,+1}` | 256 | `u8 qs[64]`（2 bit/元素）+ `fp16 d` | 2.0625 bpw | per-256 absmax |

Bonsai-8B 全权重 `Q1_0`：8.19B 参数 ≈ 1.15 GB（相对 FP16 缩小 14.2×），除 RMSNorm 外**每一个张量**都是
`Q1_0`——这是"1-bit 模型"的完整形态参考。

### 3.2 `TQ1_0` 的块布局与编解码（最值得抄的一段）

```c
// ggml-common.h：1.6875 bpw
typedef struct {
    uint8_t   qs[(QK_K - 4 * QK_K / 64) / 5];  // QK_K=256 → 48 字节，每字节 5 个三进制位（3^5 = 243 < 256）
    uint8_t   qh[QK_K / 64];                   // 4 字节，每字节 4 个三进制位
    ggml_half d;                               // fp16 尺度
} block_tq1_0;                                 // 54 字节 / 256 权重
```

**编码**（`quantize_row_tq1_0_ref`）：

```
trit xi = lroundf(v / d) + 1           // {-1,0,+1} → {0,1,2}
code c  = ((((xi0*3 + xi1)*3 + xi2)*3 + xi3)*3 + xi4)     // 5 位三进制，c ∈ [0,242]
存盘 s  = (c * 256 + 242) / 243                            // 定点"预乘"到 8 位空间
```

**解码**（`dequantize_row_tq1_0` / 内核内联）：

```
q  = s * pow3[n]        // pow3 = {1,3,9,27,81}；uint8 乘法按 256 回绕，这是刻意的
xi = (q * 3) >> 8       // 取出第 n 个 trit，无除法
值 = (xi - 1) * d
```

**为什么这么绕**：朴素做法是 `c / 3^n mod 3`（除法）或查表（访存）。预乘 `256/243` 之后，
`s·3^n·3>>8` 就是"除以 3 的幂并取整"的定点等价形式——**编码期一次性付出，解码期每个权重只剩乘法与移位**。
这就是三值格式在 CPU 上能跑得动的根本原因。

**布局**：`qs` 的字节 `m`（0..31）承载权重 `{m, m+32, m+64, m+96, m+128}` 的 5 个 trit；
解码按 n 外层、m 内层展开成连续的 32 个权重——正好匹配 AVX2 一次 32 字节 load 的节奏。
剩余 240..255 这 16 个权重走 `qh`（每字节 4 个 trit），尾巴单独处理。

**⚠ 块大小不可照搬**：ggml 的块类型要求**行宽整除块大小**（`quantize_row_tq1_0_ref` 里直接
`assert(k % QK_K == 0)`，`ggml_row_size` 也按整除算行步长）。我们的 K 常见为 **64 / 128 / 784 / 10**
（GPT d_model、MNIST 输入维等），都不是 256 的倍数——照搬 256 块要么把行补齐（K=10 时浪费 26×），
要么让块跨输出行（尺度语义变糊）。所以 D2 的结论是**照抄编码与解码、块粒度改行级**（见 §3.3）。

### 3.3 尺度规则：per-256 absmax 是为"事后量化"选的

`quantize_row_tq1_0_ref` 的尺度是块内 **绝对最大**：`d = amax; xi = lroundf(v/d) + 1`。
注意这不是 BitNet 论文的 per-tensor **absmean**——llama.cpp 的量化器面对的是**任意给定的 f32 权重**，
per-block absmax 才能让逐块相对误差可控。代价是：当块内只有一个离群值时，多数权重会被舍到 0
（`|v| < 0.5·amax` → trit 0）。

**实测裁定（2026-10-10，三模型；完整数据见 `research/ternary_scale/REPORT.md`）**：

| 模型 | absmean@row 零化% / 重构% / **信息熵** | absmax@row 零化% / 重构% / **信息熵** |
|---|---|---|
| GPT d128（3.18M 权重） | 32.7% / 50.4% / **1.585 bit** | 77.4% / 61.4% / 0.997 bit |
| MNIST ViT d64（198K） | 26.1% / 42.9% / **1.567 bit** | 67.0% / 66.3% / 1.244 bit |
| MNIST CNN（108K） | 27.6% / 45.8% / **1.574 bit** | 84.1% / 80.8% / 0.791 bit |

**看"信息熵"那一列**：absmean 让三值分布恰好携带 `1.567~1.585 bit/权重`（≈ `log2 3`，三值编码的
理论上限），而 absmax 只有 `0.13~1.29 bit`——**用 1.6875 bpw 的存储装不到 1 bit 的信息**。
原因是 absmax 的阈值 `0.5·max|w|` 被单个离群值支配，绝大多数权重被归零；absmean 的阈值
`0.5·mean|w|` 自适应地把权重摊到三个码位上（三项近等概率 ⇒ 熵取满）。

粒度则是**次要变量**：absmean 下 `row / blk64 / blk128 / blk256` 几乎同分（零化率与重构误差相差
不到 2 个百分点），`tensor` 只略差。**所以粒度按工程约束挑**——本项目取 **per-row**：
没有"行宽整除块大小"的约束（§3.2 的 256 块要求 K%256==0，而我们的 K 常见 64/128/784/10），
尺度就是每行一个 fp16（O(rows) 个字节），且与 §4.4 的 `row_reduce_sum` 形状天然对齐。

`TQ1_0` 选 absmax 有其语境：它是"事后量化别人训好的 f32 权重"，没有训练去适应量化误差，
逐块 absmax 至少保证**相对**误差有界。我们做 QAT / 从头训，权重会适应量化，此时该关心的是
"三值码位的信息容量"——absmean 完胜。

### 3.4 点积的整数化：`{0,1,2}` + 每 16 个激活修正一次

`ggml_vec_dot_tq1_0_q8_K`（激活侧是 `Q8_K`：256 个 int8 + fp32 尺度 + 每 16 个 int16 的和 `bsums`）：

```
sum(int32) = Σ_j xq_j · a_j          // xq ∈ {0,1,2}（不是 {-1,0,1}）
结果        = (sum - Σ_j a_j) · d_w · d_a
           = Σ_j (xq_j - 1) · a_j · d_w · d_a
```

关键在于**三值被存成 `{0,1,2}`**：这样才能用无符号 × 有符号的乘加（`maddubs` / `vpdpbusd`），
代价是每个块多减一次"激活和"。`Σ a_j` 不需要重新算——`Q8_K` 的 `bsums` 本来就按每 16 个激活记录，
**一次减法就修正 16 个权重**，成本可忽略。

这条设计与 §3.2 的编码互为因果：`{0,1,2}` 正是"三进制码位"的自然取值。

### 3.5 AVX2 内核的两个手法

1. **乘 3 的幂用字节加/移位树**：`3x = x+x`；`9x = (x<<3 & 0xF8) + x`；`3·9x`、`9·9x` 同理——
   每个 trit 的 `s·3^n` 都是 2~3 条字节指令，不是乘法。
2. **`3x/256` 用 `avg_epu8` 逼近**：`avg(a,b) = (a+b+1)>>1`，组合两次再加 `srli 6` 得到高 2 bit；
   前置 `subs_epu8(1)` 抵消 `avg` 的 +1 舍入。于是"乘 3 再取第 8 位以上"全程无乘除。

累加在 int16 → 末尾 `madd_epi16` 一次升到 int32 → `cvtepi32_ps` 后乘 `d_w·d_a`。

### 3.6 二值能查表、三值不能（Q1LUT 给我们的警告）

Lumina 的主力优化是 **二值激活侧查表（Q1LUT）**：4 个连续二值权重只有 16 种符号组合，
为 4 个激活预计算 16 项表 → 一次 `vpshufb` 出 4 个权重的部分点积，且利用二值对称性
`T[c ^ 0xf] = -T[c]` 只算一半表项。实测 tg 约 +30%（已到单通道带宽上限），pp2048 +57%。

wiki 里对三值的判断值得原文照录（§7）：**BitNet 的三值 `{-1,0,+1}` 无法用单条 `vpsignb` 表达**，
而按 BitNet TL 的思路"每权重一张 16 字节表"会变成 **16 字节/权重（约 128 倍膨胀）**，
在带宽受限的 CPU 上必输。所以：

- 三值**不能**照搬二值的查表法，也不能照搬 TL 的每权重查表；
- 三值的正道是 §3.2~§3.5 的**打包位域 + 整数乘加**，或（GPU 侧）解包成 int8 后走整数点积。

### 3.7 照搬 / 必须重选

| 项目 | 处置 | 理由 |
|---|---|---|
| 5-per-byte base-3 编码 + `(s·3^n·3)>>8` 解码 | **照搬（D2 已定）** | 无除法；字节回绕是编码的一部分，必须逐位复刻。块粒度改行级后位宽 ≈ `log2 3 + 16/K`（1.59~1.71 bpw） |
| `{0,1,2}` 存储 + `Σa` 修正 | **照搬** | 是无符号乘加能用三值的前提；修正粒度对齐已有的按块激活和 |
| 每 32 权重成组的字节布局 | **照搬** | 与 32 字节 SIMD load / GPU workgroup 节奏对齐 |
| per-256 absmax 尺度 | **重选：absmean + 行级**（D1/D2 已定，实测见 §3.3） | absmax 把 64~98% 权重归零、信息熵只有 0.13~1.29 bit；absmean 命中 `log2 3 = 1.585` |
| Q1LUT / VNNI / 14 个 CPU 变体 | **不适用** | x86 CPU 专用；本项目 CPU 侧可参考但收益有限，主战场在 Vulkan |
| 无损容器 `.gguf.lumina` / QLoRA 训练引擎 | **不适用** | 与本设计的权重量化正交（QLoRA 训练的是 LoRA 适配器，不动量化权重） |

---

## 4. 本项目设计

### 4.1 `Precision::T1_58` 与提升序（R4）

```cpp
enum class Precision : std::uint8_t {
    F16 = 0, F32 = 1, BF16 = 2, F64 = 3,
    T1_58 = 4,          // 三值 {-1,0,+1}（1.58 bit/权重）；P2 起才有打包存储
};
```

- `precision_name` 返回 `"t1_58"`；`precision_tag` / `precision_from_tag` 追加 `4 = t1_58`
  （0..3 已被 f32/f64/f16/bf16 占满）。
- **命名口径（D4 已定，要求全局统一）**：`T` 表示**类型**（三值），不是"量化"——类型前缀
  回答"是什么"（`F`=float、`BF`=bfloat、`T`=ternary），与现有 `F16/F32/BF16/F64` 同构。
  "量化"保留为**操作名**（`quantize`：f32 latent 投影到三值确实是量化，有尺度、有舍入）；
  两件事分开，不冲突。

  | 位置 | 定名 |
  |---|---|
  | 枚举 | `Precision::T1_58` |
  | `precision_name` | `"t1_58"` |
  | CLI 词法 | `t1_58`（**无别名**——不提供 `1_58`/`t158` 等变体） |
  | CLI 帮助 | `--precision-* <f16\|f32\|t1_58>` |
  | 序列化 tag | `4` |
  | 层名 | **`BitLinear`**（保留 BitNet 专名，便于检索） |
  | 存储格式 | "T1_58 打包（编码参照 llama.cpp `TQ1_0`，尺度 = 行级 absmean）" |
  | 文档口径 | 三值（T1_58） |

  另注：`--precision-*` 接受 `t1_58` 之后，**普通模型选它会在 `init` 校验处报错**（§4.3），
  这是设计意图而非缺陷；CLI 不需要为它加模型白名单。
- **`max_precision` 必须显式处理**：它现在直接比较枚举值（`F16=0 < F32=1 < BF16=2 < F64=3`），
  加 `T1_58=4` 会变成"比 F64 还高"。做法二选一：① `max_precision` 对非"舍入精度"的值 fail-fast；
  ② 把序从枚举值里拆成独立的 `precision_rank()`。**推荐 ②**——枚举值从此只是标签，
  任何依赖"枚举值大小 = 精度高低"的代码都会被编译器/测试逼出来。
- `precision_bytes(T1_58)`：P1 返回 0（同 BF16/F64 的"非存储精度"口径）；P2 落地打包后，
  位宽由打包格式表达：行级 base-3（5 trit/字节 + 尾巴）= `log2 3 + 16/K` ≈ **1.59~1.71 bpw**
  （K 为行宽；K=128 → 1.71、K=784 → 1.61），**且必须另算对齐填充**——若为 vec4 快路径把每行
  pad 到 16 B，K=128 会从 26 字节涨到 32 字节 = **2.0 bpw，收益被填平**（§5 的 P2 待定项）。
  **不要**试图把位宽塞进"整数字节/元素"。
- `check_precision_supported(T1_58)`：**P1 拒绝**（清晰报错："t1_58 不是存储精度"）。
  这条保证 P1 里不会有任何代码无意间拿 `T1_58` 去建张量。P2 再按新的存储类型放宽。

### 4.2 `PrecisionSet` + `Layer::precision_support()`（R1）

声明**按槽**给出，而不是一个扁平集合——因为 `BitLinear` 需要的是
`param ∈ {T1_58}`、而 `compute/stable/optimizer ∈ {f16, f32}`。
扁平 `{T1_58}` 会让 `{param:T1_58, compute:T1_58}` 通过校验，然后**每个激活算子都被要求输出三值**。

```cpp
// L0：位掩码，无分配、可平凡拷贝、留足未来扩展（uint32 支持 32 种精度）
struct PrecisionSet {
    std::uint32_t mask = 0;
    constexpr bool has(Precision p) const noexcept {
        return (mask & (1u << static_cast<std::uint32_t>(p))) != 0;
    }
    static constexpr PrecisionSet of(std::initializer_list<Precision>);
};

// L2 基类：默认 = 现状（f16/f32 全槽），故既有层零改动
struct PrecisionSupport {
    PrecisionSet param, compute, stable, optimizer;
};
[[nodiscard]] virtual PrecisionSupport precision_support() const;
[[nodiscard]] virtual const char* layer_name() const noexcept { return "Layer"; }
```

- 默认实现返回 `{f16,f32}` × 4 槽 → **所有既有层一行不改**（这是"不影响其他层"的落实方式）。
- `BitLinear` override 为 `param={T1_58}`、其余三槽 `{f16,f32}`。
- 错误信息要能定位：**层名 + 槽位 + 取值 + 该槽允许的集合**。
  `layer_name()` 是新增的小设施（现在 `Layer` 没有任何名称设施，诊断只能靠 file:line）。

### 4.3 校验点与语义（R3）

- **校验点** = `Layer::init(engine)` 这个 NVI（`compute_layer_base.hpp`）。
  它是唯一咽喉：`Model::add<T>()` 先注入 profile 再 `init`；工厂路径
  （`build_gpt_model` 把 `cfg.precision` 传进 `GPTModel` 构造器）也在 `init` 前完成注入；
  复合层的 `init_impl` 会逐个 `init` 子层 → **叶子层各自校验即可，不需要任何聚合逻辑**。

#### 4.3.1 现状：三条 profile 注入路径，只有一条能到 GPT/RAPT 的子层（D3）

| 路径 | 语义 | 能否到 GPT/RAPT 子层 |
|---|---|---|
| ① 工厂构造器：`build_gpt_model(..., cfg.precision)` / `build_mnist_model_from_spec(engine, spec, precision)` | 复合层在**构造器**里对自身 + 子层逐个 `set_precision_profile`（`compute_layer_gpt.hpp:54-58/282-313`、`compute_layer_rapt.hpp:849-853/1050-1066`） | **✅** |
| ② `Model::set_default_precision_profile(p)` | 只在**之后 `add()` 的层**上注入（文档要求"须在 add 之前调用"） | 取决于被 add 的层 |
| ③ `Model::set_precision_profile(p)` | 只遍历**顶层** `layers_`（`model_container.hpp:125-129`） | **❌** 对 GPT/RAPT 只打到复合层自己 |

override 了 `set_precision_profile` 的只有 `MultiHeadAttention`（`compute_layer_attention.hpp:483`）、
`FeedForward`（`compute_layer_feedforward.hpp:74`）、`TransformerEncoderLayer/Encoder/Classifier`
（`compute_layer_transformer.hpp:145/309/494`）；**`GPTModel` / `GPTBlock` / `RAPTModel` 没有 override**，
它们只在构造器里向下传。两个 CLI 的实际用法：`text_train` 只走路径 ①；`mnist_train` 额外调了一次
路径 ③（`src/mnist_train.cpp:678`，对 MLP 是冗余——`build_mnist_mlp_model` 已用路径 ② 在 add 前注入）。

**结论**：校验放 `init` 的前提是"profile 在 `init` 前定稿"，这条前提**今天只在路径 ① 成立**；
路径 ③ 上 GPT 的子层拿不到新 profile，校验会被绕过。

#### 4.3.2 建议（待批）

**（a）+（b）组合，两条都做**：

- **（a）补 override**：给 `GPTModel` / `RAPTModel` 补 `set_precision_profile` override（照
  `TransformerEncoder` 的写法，逐子层下传），让树内两条复合路径语义一致。顺带删掉
  `src/mnist_train.cpp:678` 的冗余调用。
- **（b）把契约写死**：`set_precision_profile` 在 `engine_ != nullptr`（已 init）时 **fail-fast**——
  "profile 在 init 后不可变"。这样"profile 何时有效"从"三条约定的交集"变成一个可判定的规则，
  校验不再可能被旁路。

被否决的（c）：校验放 `init` + 让每个复合层聚合子层的 capability 声明。它把 `precision_support()`
从"每层的静态声明"变成"递归求交"，复杂度上一个量级，而收益只是省掉（a）的几行 override。
- **语义（关键）**：`param = T1_58` 表示"**该层 forward 的有效权重精度是三值**"，
  **不是**"参数张量的存储布局是三值"。因此：
  - `BitLinear` 的 `parameters()` 仍返回 **f32 latent**（优化器与 STE 的作用点）；
    `create_tensor` 永远不用 `p_.param` 当实参（P1 里 `p_.param=T1_58` 只是一句声明）；
  - 若不这样定义，就会掉进死角：优化器要 `step()` 一个三值张量，而没有 f32 latent 就没有 STE。

### 4.4 `BitLinear`（P1 形态）

```cpp
class BitLinear final : public Layer {
    // latent_w_ : f32 参数（优化器更新对象；STE 梯度落点）→ parameters()
    // wq_       : 量化缓冲，f16 存储、值 ∈ {-1,0,1}（f16 精确可表示）——不落盘，每步重算
    // tau_      : (out,1) f32，逐行 absmean —— 由 DSL 归约算出，不回读宿主（铁律 #12）
};
```

**forward**（两处 `dsl::compute`，AOT 锚点自动登记；库内还需在 `tools/scan_exprs.cpp`
的 per-layer dry-run 里补一个 `BitLinear`，理由见 `AGENTS.md` §7——dry-run 是结构的主要来源）：

```
tau  = row_reduce_sum(|latent_w_|) * (1/in_features)                       // (out,1)
wq   = select(w > 0.5·tau, +1, select(w < -0.5·tau, -1, 0))                // 逐元素
Y    = matmul(wq, X) * row_broadcast(tau) + row_broadcast(b)
```

（`select` 链就是 BitNet 的 `RoundClip(w/τ, -1, 1)`：DSL 没有 `round`/`floor`，
但在 `[-1,1]` 区间内它与阈值 `0.5` 的 select 等价。激活量化才需要新增 `round`/`floor` 算子，见 P3。）

**backward**（STE = 把 `wq` 当常数求导；与现有 `Linear::backward` 同形，只是权重换成 `wq`）：

```
dX     = matmul(wq_scaled, dY, transA=true)        // wq_scaled = wq 逐行乘 τ（与 forward 复用）
dW_lat = matmul(dY, X^T) * row_broadcast(tau)      // 恒等 STE × 去量化尺度
db     = row_reduce_sum(dY)
```

（`τ` 是逐输出通道的，所以它会分别落到 `dX` 的左乘侧与 `dW` 的右乘侧——不能只在一处补。）

**可行性的前置证据**：`research/ternary_ste/REPORT.md` 的两个 PoC 用**现有公开 API**
（`dsl::compute` + `select` + `rparam` + `matmul` + `compute_into`，零库改动）在 CPU 上跑通了三值与二值
STE 训练：三值 MSE 14.207 → 0.000000、教师模式（含零）**1370/1370 全对**；
二值 MSE 20.949 → 0.000000、**2048/2048 全对**。

### 4.5 γ 的归宿（R2）与存储演进

`(rows, cols, T1_58)` **不足以描述一个张量**——还需要尺度（同一个三值矩阵配 γ=0.5 与配 γ=2.0
是两个不同的映射）。因此"三值作为存储精度"必然要求张量带上 qparam，这正是
`17-unified-tensor-engine.md` §5 **M7 存储多态**的第一个真实触发条件（原文裁定是"第三后端出现前不立项"，
现在触发条件是**出现了需要非均匀布局的存储**，应当据此更新 18 号路线图）。

分期因此是干净的：**P1 不需要 M7**（量化缓冲是 f16，qparam 就是那张 `(out,1)` 的 τ），
**P2 才需要**（打包字节 + 逐行尺度 + `matmul_q` 原语）。

### 4.6 约束：`T1_58` 张量只进量化 GEMM

规定 **`T1_58` 存储张量只能进入量化 GEMM 原语**（`matmul_q` 及其转置），
其余一切算子入口遇到它走既有的 `unsupported precision` 错误路径。效果：

`P == Precision::F32` / `P == Precision::F16` 的**二分支不需要逐个改成三分支**——
它们本来就以 `NN_FAIL` 收尾；加上 §4.3 的构建期校验，用户永远看不到那条运行期错误。
这把"788 处分支"的爆炸半径压回到"一个原语 + 一处校验"。

### 4.7 序列化

- `save_model` 写的是 `model.parameters()` 的扁平序列 + `extra_state()`；`BitLinear` 的
  latent 是 f32，**三值缓冲是派生物，不落盘**（加载后重算），τ 同理。
- 但**加载时必须重建出 `BitLinear` 而不是 `Linear`**，所以 `ModelSpec` 需要记住"这个模型用了量化层"：
  追加 `weight_quant`（`None` / `T1_58`）字段，**缺键 = `None` → 旧文件零破坏**
  （与 `norm_place` 的缺键处理同款）。参数条数/顺序不变的模型不需要升 `MODEL_VERSION`，
  纯追加字段按现有惯例判定（见 `19-unified-dataset.md` 的模型版本分派）。

### 4.8 `BitLinear` 的接入位置（D5，建议待批）

#### 4.8.1 现状：GPT/RAPT 不是用 `model.add<Linear>()` 拼的

MLP 路径（`build_mnist_mlp_model`）确实是 `model.add<Linear>(...)` 一条条拼的 → 换成 `BitLinear`
等于换一个模板实参，**零结构改动**。但 GPT/RAPT 的内部是**硬编码的 `Linear` 成员**：

| 复合层 | 持有的线性层 | 位置 |
|---|---|---|
| `CausalSelfAttention` | `Linear w_q_ / w_k_ / w_v_ / w_o_` | `compute_layer_attention.hpp:409-412` |
| `FeedForward` | `Linear fc1_ / fc2_` | `compute_layer_feedforward.hpp:24-25` |
| `GPTModel` | `Linear lm_head_` | `compute_layer_gpt.hpp:248` |
| `RAPTModel` | 同上（`attn_` / `ff_` / `lm_head_`） | `compute_layer_rapt.hpp` |

所以"1.58-bit GPT"必须先决定这三条里的成员怎么办。

#### 4.8.2 三条路

| 方案 | 内容 | 代价 |
|---|---|---|
| ① 模板化 | 把 `CausalSelfAttention<LinearT>` / `FeedForward<LinearT>` 参数化 | 改动最大：4 个层头 + 所有实例化点；但最彻底 |
| ② 工厂 + 基类指针 | `std::unique_ptr<Layer> fc1_ = make_linear_layer(kind, ...)`，运行期选 `Linear` / `BitLinear` | 中等；改成员类型与初始化路径 |
| ③ P1 只覆盖 MLP | P1 交付 `model.add<BitLinear>` 可用 + 能力校验 + 训练闭环；GPT/RAPT 留 P1.5 | 最小；P1 不含 GPT 端到端 |

#### 4.8.3 建议（待批）

- **P1 走 ③**：P1 的目标是"架构统一 + 功能正确"（§1），把 GPT/RAPT 的接线拖进 P1 会让
  "契约先行"变成"契约 + 大重构"。`research/ternary_ste/REPORT.md` 已在单层层面证明算法路径可行，
  端到端 GPT 的三值训练属于 P1.5。
- **P1.5 走 ②，并照抄仓库自己的先例**：归一化层已经是这个模式——`make_norm_layer(NormType)`
  返回 `unique_ptr<Layer>`，`GPTBlock` 持有 `std::unique_ptr<Layer> norm1_/norm2_`
  （`compute_layer_gpt.hpp:56-58` 的 `if (norm1_) norm1_->set_precision_profile(...)` 就是证据）。
  照此引入 `make_linear_layer(...)`，把 4 处 `Linear` 成员换成 `unique_ptr<Layer>`，
  即可让 `ModelSpec` 的一个开关同时切换 GPT/RAPT 的线性层类型。
- **不建议 ①**：模板化会把"层类型是编译期决定"这件事扩散到 attention/feedforward/GPT/RAPT
  四个头与全部实例化点，收益只是省掉一次虚调用——与 `docs/development/20-custom-layer-ergonomics.md`
  里"层接入四条契约"的方向相反。

---

## 5. 分期与验收

| 阶段 | 内容 | 落点 | 验收 |
|---|---|---|---|
| **P1** 架构统一 + 功能正确<br>（**范围 = MLP 路径**，D5 建议③） | `Precision::T1_58`（含 R4 的序处理、tag 4、`check_precision_supported` 拒绝）；`PrecisionSet` + `Layer::precision_support()` + `init` 校验 + `layer_name()`；`BitLinear`（量化 forward + STE backward）；D3 的 (a)+(b)（§4.3.2）；`scan_exprs` 补 dry-run；spec 的 `weight_quant` 字段；2 个 CLI × 4 flag 的词法 + GUI + `gui_cli_audit` | `precision.hpp` / `compute_layer_base.hpp` / 新 `compute_layer_bitlinear.hpp` / `tools/scan_exprs.cpp` / `model_spec.hpp` / `src/*` / `gui.py` | ① 不兼容组合**构建期报错**（含 `param=T1_58` + 普通 `Linear` 的反例用例）；② latent 权重的 gradcheck（STE 穿不过量化器，只对 latent 做数值梯度）；③ CPU/GPU 各一个端到端小训练收敛；④ 既有 27 个 ctest 全绿、L2 审计 0。**逐文件清单见 §7** |
| **P2** 真性能 | Tensor qparam（M7）+ **行级打包**（`TQ1_0` 编码：base-3 5-per-byte + `qh` 尾巴 + 定点解码；尺度 = 行级 absmean，**无 per-256 尺度**；行步长/对齐待定）+ `matmul_q` / `matmul_q_t`（CPU 微内核 + Vulkan shader）+ `layer_bench` A/B + 序列化打包 | `compute_tensor.hpp` / `compute_engine.hpp` / `compute_cpu_engine.hpp` / `compute_gpu_engine.hpp` / 新 `.comp` | ① `layer_bench --op matmul` 配对 A/B 报告（含"当前 matmul 是带宽受限还是延迟受限"的判定）；② 打包往返逐字节；③ 数值与 P1 的 f16 缓冲路径一致（同 latent 权重下逐位）；④ 与 llama.cpp `TQ1_0` 的反量化对拍（**只对编码/解码**——尺度规则不同，块内容不互通） |
| **P1.5** GPT/RAPT 接入 | `make_linear_layer` + `unique_ptr<Layer>` 替换 4 处 `Linear` 成员（照 `make_norm_layer` 先例，§4.8.3） | `compute_layer_attention.hpp` / `compute_layer_feedforward.hpp` / `compute_layer_gpt.hpp` / `compute_layer_rapt.hpp` | GPT/RAPT 的三值端到端训练；既有全 f32 路径零回归 |
| **P3** 计算侧收益 | 激活 int8 量化 + 整数点积（GPU 需 `GL_EXT_shader_integer_dot_product` 的设备特性探测；CPU 侧参考 §3.4/§3.5）；DSL 新增 `round`/`clamp` 算子（三处同步：`ExprOp` + CPU 求值 + GLSL emitter） | `expr_spec.hpp` / `expr_dsl.hpp` / `expr_glsl_gen.hpp` / 新 `.comp` | 与 PyTorch 参考实现对拍（`compare_with_torch/`） |

> **性能预期必须先量后承诺**：本项目主线模型 `d_model` 只有 64–256（GPT d64/h4/L4/ff256），
> matmul 很可能落在 **kernel 启动延迟区而不是带宽区**——打包省下的权重带宽未必能兑现成墙钟时间。
> `layer_bench` 的配对 A/B 是 P2 的准入门槛，不是收尾动作。

---

## 6. 风险 / 已否决 / 待定

**已否决**

1. **扁平能力集合**（`BitLinear` 声明 `{T1_58}` 一个集合）：会被 `compute=T1_58` 穿透，
   导致激活全变三值。→ 改按槽声明（§4.2）。
2. **"能选但静默出垃圾"**：18 §3.1 C2/C3 已把这类"可选但无提示"登记为缺陷。
   → 校验提前到 `init`，且报错要带层名/槽位/取值。
3. **把三值当作 `compute`/`stable` 的值**：损失链（softmax / LayerNorm / CE）被三值化必然崩，
   与 `05-mixed-precision.md` §12.5 记录的"全 f16 不可训练"同一性质、更严重。
4. **照搬 BitNet TL 的每权重查表**：16 字节/权重，带宽上必输（§3.6，Lumina 已有结论）。
5. **在 P1 里用 `T1_58` 建张量**：P1 拒绝，避免"半实现的存储类型"散进引擎。

**风险**

- **梯度与量化器不可微**：STE 的正确性无法用数值 gradcheck 验证（量化器导数恒 0）。
  测试只能覆盖"latent 梯度 = STE 公式"，以及"量化输出逐位符合 τ 规则"。
- **优化器状态必须留 f32**：`research/f16_weight_decay/REPORT.md` 已证 f16 状态会被舍入吃掉；
  三值训练同理，`optimizer` 槽不允许 `T1_58`（校验会拦）。
- **闭合世界**：新层的表达式若没进 `scan_exprs` 的 dry-run，GPU 会以"闭合世界硬报错"暴露，
  且离根因很远（AGENTS §7 已有先例）。

**已裁定（2026-10-10）**

- **D1 = absmean + per-row**：实测见 §3.3 与 `research/ternary_scale/REPORT.md`。
  absmax 把 64~98% 的权重归零（信息熵 0.13~1.29 bit），absmean 命中 `log2 3 = 1.585 bit`；
  粒度在 absmean 下不敏感，取 per-row 是因为它同时满足"无 256 整除约束"与"尺度只占 O(rows)"。
- **D2 = `TQ1_0` 的编码与解码 + 行级块**：照抄 base-3 5-per-byte、`qh` 尾巴与定点
  `(s·3^n·3)>>8`；不采用它的 per-256 fp16 尺度（与 D1 冲突，且要求 K%256==0）。
  与 llama.cpp `TQ1_0` **块内容不互通**（尺度规则不同），只保证编解码逐位等价。
- **D4 = `T1_58`**：CLI 词法 `t1_58`、无别名、tag 4、层名 `BitLinear`（见 §4.1 的命名表）。

**待定**

- **D3 复合层 profile 下传** —— 本文建议 **（a）补 `GPTModel`/`RAPTModel` 的 override + （b）
  `set_precision_profile` 在 init 后 fail-fast**，见 §4.3.2；否决了"复合层递归求交"。
- **D5 `BitLinear` 接入位置** —— 本文建议 **P1 只覆盖 MLP（③），P1.5 照 `make_norm_layer` 先例
  引入 `make_linear_layer` + `unique_ptr<Layer>`（②）**，见 §4.8.3；否决了模板化（①）。
- **D6 行步长与对齐**（P2 编码前定案）—— 行连续打包（`ceil(5K/… )` 字节/行，允许非对齐标量读）
  vs 每行 pad 到 16 B（换 vec4 快路径，但 K=128 时位宽从 1.71 回到 2.0）。
  本文倾向**先按行连续、不 pad**：P2 的第一目标是拿到真实的带宽/时间数据（§5 的 `layer_bench`），
  对齐优化应在数据之后。

---

## 7. 实施清单（P1，文件级）

按依赖顺序；每项给出"改什么 / 为什么"。**P1 不含 GPT/RAPT 三值训练**（§4.8.3 ③）。

| # | 文件 | 改动 |
|---|---|---|
| 1 | `include/neuralnet.cpp/precision.hpp` | 加 `Precision::T1_58 = 4`；`precision_name` → `"t1_58"`；`precision_tag`/`precision_from_tag` 加 `4`；`precision_bytes` 返回 0 并注明"非存储精度，位宽由打包格式表达"；`check_precision_supported` 拒绝 `T1_58`；**把"序"从枚举值拆出**（`precision_rank()`，见 §4.1 R4） |
| 2 | `include/neuralnet.cpp/compute_layer_base.hpp` | 加 `PrecisionSet` / `PrecisionSupport`（§4.2 骨架）；`virtual precision_support()` 默认 `{f16,f32}×4`；`virtual const char* layer_name()`；在 `Layer::init` NVI 内做校验（层名 + 槽位 + 取值 + 允许集合）；按 D3(b) 在已 init 时拒绝 `set_precision_profile` |
| 3 | 新 `include/neuralnet.cpp/compute_layer_bitlinear.hpp` | `BitLinear`：`latent_w_`(f32) + `wq_`(f16 缓冲) + `tau_`((out,1) f32)；`forward` = `compute_reduce(row_reduce_sum(abs(W)))` → `select` 量化 → `matmul+row_broadcast(τ)`；`backward` = STE（`dX`/`dW_lat`/`db`，§4.4）；`precision_support()` = `param={T1_58}` 其余 `{f16,f32}`；`layer_name()` = `"BitLinear"` |
| 4 | `tools/scan_exprs.cpp` | per-layer dry-run 补一个 `BitLinear`（否则 GPU 闭合世界硬报错；dry-run 是结构的主要来源，见 `AGENTS.md` §7） |
| 5 | `include/neuralnet.cpp/model_spec.hpp` + `model_serialization.hpp` | `ModelSpec` 加 `weight_quant`（`None`/`T1_58`，缺键 = `None`）；`save/load` 往返；`spec_summary` 与校验同步 |
| 6 | `src/mnist_train.cpp`、`src/text_train.cpp` | `parse_precision` 接受 `t1_58`（**无别名**）；4 行 help 改成 `<f16\|f32\|t1_58>`；删掉 `mnist_train:678` 的冗余 `set_precision_profile`（D3(a) 的连带） |
| 7 | `gui.py` + `bench/gui_cli_audit.py` | `PRECISION_OPTIONS` 加 `"t1_58"`（4 行下拉）；审计脚本跑一遍确认仍退 0（只加**取值**、不加 flag，flag 面无变化） |
| 8 | 新测试 | ① 能力校验：`param=T1_58` + 普通 `Linear` → **构建期报错**（反向用例）；`BitLinear` + `param=f16` → 报错；② `BitLinear` 的 latent 梯度 = STE 公式（数值梯度只对 latent）；③ 量化输出逐位符合 τ 规则；④ MLP 端到端小训练收敛（CPU + GPU 各一） |
| 9 | 文档 | `AGENTS.md` §11（已加行）+ `docs/usage/03-compute-engine-usage.md` 的精度表（"仅 F16/F32 可用"）在 P1 落地后同步 |

**验收口径**（沿用仓库四件套 + 本设计的专属项）：

1. 构建零告警；`ctest` 既有 27 个全绿（GPU 用例按既有 `77 = skip` 口径如实记录）；
   `pwsh -File bench/doc_inventory.ps1` 第 [4] 节 `L2-VIOLATIONS: 0`；
   scan 产物 hash 的变化**必须**是"新增 T1_58 相关结构"可解释的（逐条核对来源，见 `AGENTS.md` §7）。
2. 能力校验的反例必须**在构建/初始化期**报错，且错误信息含层名 + 槽位 + 取值。
3. CPU 与 GPU 各跑通一个 `BitLinear` MLP 端到端训练，loss 收敛（不设精度门槛，P1 不比性能）。
4. `f32`/`f16` 既有路径**零回归**：全 f32 配置下 `Layer::init` 的校验是只读的、无额外分配。

---

## 8. 附录：与 f16 的正交性

| 维度 | f16/f32（现有） | 三值（本文） |
|---|---|---|
| 语义 | **舍入**：f32 参考计算 + 输出舍入到 P | **离散化**：按统计尺度投影到 `{-1,0,+1}` |
| 需要的元数据 | 无（精度即全部信息） | **尺度 γ**（统计量，随张量/行/块） |
| 适用算子 | 全部（统一 P） | 只适用于权重；激活保持 f16/f32（P3 才 int8） |
| 累加 | `acc<P> = f32` | 恒 f32（`{-1,0,1}` 累加无意义） |
| 与提升序的关系 | 有（F16 < F32） | **无**（必须排除，见 §4.1 R4） |

**一句话**：f16 是"同一算法换更窄的容器"，三值是"换一种权重表示"。二者正交，
所以三值进的是"精度词汇表"，而它的实现落在 `Layer` 与（P2 起的）专用原语上。
