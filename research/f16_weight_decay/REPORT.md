# 训练 loss 中途回弹排查报告（2026-10-04）

> 归口：`research/f16_weight_decay/`（报告 + 探针 + 对照模型 + 原始日志，自包含）。
> 设备：Windows / **NVIDIA CMP 40HX**（`--gpu`）。
> 数据：`datasets/tinystories_200mb/tinystories_small.txt` 前 30 MB 切片 + 仓库 BPE 8192 词表。

**现象**：GPU + “f16 (存储)” 预设、adamw(wd=0.1)、3 epoch，loss 先正常下降，
约在训练中点（≈ 第 2 epoch 中间）开始回弹，最终涨回 8.x（≈ 没训练过）。

**结论（一句话）**：`--precision-param f16` 下，AdamW 的权重衰减写回
`p *= (1-lr·wd)` 走的融合 shader 用 `float16_t(...)`（SPIR-V `OpFConvert`）做
f32→f16 存储，**在本机驱动上按“向零截断”舍入**——而 `1-lr·wd = 0.99997` 与 `p`
的差远小于 1 个 f16 ULP，正确舍入本应是恒等，截断则**每个 optimizer step 恰好
掉 1 个 f16 网格步（与 lr 无关、只要 wd>0 就发生）**。权重以 ≈0.07%/步的速度
指数衰减，约每 1000 步减半，数万步后全部下溢到 f16 次正规数并最终**精确变成 0**
→ 归一化 γ=0 把残差流清零 → logits 只剩 lm_head bias → 近乎均匀分布 →
loss → ln(8192) = 9.01（即用户看到的 8.x）。

---

## 1. 现场证据：最终模型已经彻底损坏

解析 `gpt_model.bin`（v5，61 个张量，3,194,496 参数，f16）：

| 张量 | init（基线 `init_f16.bin`，同目录） | 用户最终模型 |
|---|---|---|
| 全部参数 zeros 占比 | 0.465%（只有 bias 初始化为 0） | **97.768%**（无 NaN/Inf） |
| 4 个 Transformer 块的 Q/K/V/O、FFN 权重 | mean\|w\| ≈ 0.036–0.077 | **100% 精确为 0** |
| 所有 norm 的 γ（init = 1.0） | 1.0 | **0**（个别残留 9 个以内的碎值） |
| token_emb / pos_emb | mean\|w\| ≈ 0.016 | 99.3% / 99.2% 为 0 |
| lm_head bias（init = 0） | 0 | 堆在 ±0.03125（f16 网格）附近，均值 −0.0226 |

推论：`ln_f.γ = 0` ⇒ 残差流被归零 ⇒ `logits = W·0 + lm_head bias = bias`
⇒ softmax 近乎均匀 ⇒ loss ≈ ln(8192) = 9.01。
用 `text_infer --model gpt_model.bin --temperature 0` 贪心生成只会输出空白 token，
与上面的参数状态完全吻合。

## 2. 复现与隔离（同数据 30MB 切片、同超参、同 GPU，各 3 epoch / 1371 步）

| # | param / compute | weight-decay | 3 epoch avg_loss | 权重中位数 \|run\|/\|init\|（q.w / γ） |
|---|---|---|---|---|
| C | f32 / f32 | 0.1 | **3.68** | 0.90 / 0.94（正常） |
| E | f32 / **f16** | 0.1 | 5.19 | 0.96 / 0.986（无收缩） |
| D | **f16** / f32 | 0.1 | 4.83 | **0.415 / 0.416** |
| B | **f16** / **f16**（= `--f16`） | 0.1 | 5.84 | **0.40 / 0.418** |
| F | **f16** / **f16** | **0** | 5.88 | **1.00 / 1.000（无收缩）** |

- 收缩**只由 `param=f16` 引起**（E 无收缩）；
- 收缩**只由 `wd>0` 引起**（F 无收缩）。

## 3. 收缩的定量性质（决定根因）

- **与 lr 无关**：fixed lr=3e-4 与 lr=1e-4 各跑 457 步，γ 从 1.0 分别降到
  **0.7764 / 0.7769**（完全相同）——排除“decay 系数算错/ lr 没生效”。
- **每步恰好 1 个 f16 网格步**：p∈[0.5,1) 网格 = 2⁻¹¹，
  1 − 457×2⁻¹¹ = **0.7769**（实测 0.7764）；1371 步（跨 binade 后网格减半）理论
  ≈ **0.415**（实测 0.418）。理论与实测逐位对得上。
- 收缩是**乘性**的：按初值大小分箱，中位比值在各箱上都是 ≈0.40（与幅值无关）。

## 4. 根因定位（最小探针 + 变体追踪 + SPIR-V 反汇编）

`decay_probe.cpp`（同目录，重建方法见文末）把两条写回路径缩到单表达式，
500 步、f16 目标、factor = 0.99997：

| 路径 | CPU | GPU (NVIDIA CMP 40HX) |
|---|---|---|
| `p × rparam(0.99997)`（AdamW 权重衰减） | 500 步后**完全不变**（正确） | 1.0 → **0.7558594 = 恰好 500 个网格步**（每步 1 ULP） |
| `p + delta`（优化器 K3 加法） | 不变/随机游走 | 不变/随机游走（无系统收缩） |

`NN_PREC_TRACE=1` 显示两条走了**不同变体**：

- 衰减乘法 → `06db7ce897752074#x`（V1 运行期分派，f32 算术），
  存储为 `bout16[i] = float16_t(r0)`；
- `p + delta` → `79f355d6f8d1d247#a`（native16，全 f16 算术、结果直接是 f16，
  **没有 float→half 转换**）——所以加法路径不受影响。

`spirv-dis` 该 `#x` shader：存储点是 **`OpFConvert %half`**。
SPIR-V/Vulkan 并未把 float→float16 的舍入钉死为 round-to-nearest-even，实测本机
驱动按**向零截断**：`f16(0.99997×p)` 落在 `p` 正下方 → 截断到下一格 ⇒ 每步掉
1 ULP；而正确 RNE 会舍回 `p`（`|Δ| = 3e-5·p ≪ 0.5 ULP`）。CPU 路径用
`nn::f16`（`float_to_half_bits`，明确 round-half-to-even）⇒ 健康。

**为什么偏偏是权重衰减中招**：衰减的乘积结果“紧贴”原网格点，任何朝零方向的
误差都会掉满一格；其他运算（加法/梯度/m,v）结果离网格边界远，截断只损失
≈0.5 ULP（0.05% 量级），不易察觉。

## 5. 时间线对账

- 相对收缩 ≈ ULP/p ≈ 4.9e-4 ~ 9.8e-4 /步 ⇒ **每 ~1000 步权重减半**；
- 0.077 → f16 次正规（<6.1e-5）≈ 1e4 步，→ 精确 0 ≈ 2e4 步（Adam 的 ±lr
  随机游走会拖慢，量级 2–4 万步）；
- 用户全程 ≈ 87k 步（3 epoch × ~29k），**中点 ≈ 43k 步**——与“第 2 epoch 中间
  开始回弹、最终回到 ln(V)”完全吻合。

## 6. 建议

### 立即规避（不改代码）
1. **参数精度改回 f32**：GUI 精度预设选 f32，或 CLI `--precision-param f32`
   ——收缩直接消失（实验 D/E 已验证）。
2. 临时替代：`--weight-decay 0`（收缩消失，但失去正则，不推荐长期用）。
3. 注意：**`compute=f16` 单独也会明显掉点**（同配置 3.68 → 5.19，虽不塌陷）。
   所以“f16 (存储)”预设的两项都建议先退回 f32，等下面的修复落地再开。

### 代码修复建议
1. **根修**：`expr_glsl_gen.hpp` 里 f16 输出的存储由 `float16_t(x)`
   （→ `OpFConvert`，舍入不受规范保证）改为 `packHalf2x16(vec2(x))`
   （GLSL 规范明确 round-half-to-even）；其它 op-level shader 的 f16 写出同查。
2. **防御**：AdamW 的权重衰减在 f16 参数上做本身就无意义
   （lr·wd ≪ 1 ULP，正确舍入 = 无操作）——可在 f32 下算 decay 再按
   `cast.comp`（`packHalf2x16`，RNE）回写，或干脆对 f16 参数跳过 decoupled
   decay 并在文档里说明。
3. **补回归测试**：现有 `f16_writeback_probe` 只测 `p+delta`（`#a` 路径）与
   Adam(wd=0)，**正好漏掉 decay（`#x` 路径）**。建议加：
   GPU + param=f16 + AdamW(wd>0) 跑 N 步 → 断言权重相对变化 ≈ N·lr·wd
   （而不是 N·ULP），或直接与 CPU 结果逐位对拍。
4. **文档勘误**：`docs/development/05-mixed-precision.md` §12.5
   “profile_f16 与 f32 轨迹一致” 的结论只在 wd=0/短跑下成立；
   AGENTS/已交付能力里的 f16 描述与 `docs/development/14` 需要补这条已知问题。

## 7. 材料与复现

### 7.1 本目录材料

```
research/f16_weight_decay/           本研究自包含目录（与 research/mnist/ 平级）
├── REPORT.md                        本文
├── decay_probe.cpp                  最小复现探针（衰减乘法 vs 加法写回的舍入行为）
├── init_f16.bin                     初值基线（同结构、逐元素可比的确定性 init）
├── run_f16.bin                      --f16 + wd0.1 → 权重收缩到 0.40×（对照损坏形态）
├── run_f32.bin                      全 f32 → 正常（权重 0.90×、avg_loss 3.68）
└── logs/{a..f,k,l,m}.log            各组实验原始日志（*.log 全域 gitignore，仅本地）
```

### 7.2 重建探针

在 `CMakeLists.txt` 的 `gpu_stability_probe` 那行后加：

```cmake
nn_add_executable(decay_probe research/f16_weight_decay/decay_probe.cpp)
list(APPEND NN_APP_TARGETS decay_probe)
```

然后：

```bash
cmake --build build --target decay_probe
build/decay_probe.exe --gpu                    # 跑两组表达式 × CPU/GPU × f16/f32
NN_PREC_TRACE=1 build/decay_probe.exe --gpu    # 打印命中的变体（#x / #a）
spirv-dis build/generated/fused_06db7ce897752074_x.spv | grep OpFConvert
```

### 7.3 复现实验组（第 2 节表格）

```bash
# 语料：tinystories_small.txt 前 30MB 切片 + 仓库 BPE 8192 词表
#   head -c 31457280 datasets/tinystories_200mb/tinystories_small.txt > corpus_30mb.txt
COMMON="--vocab datasets/tinystories_full/bpe_vocab_tinystories_full_8192.json \
  --batch-size 64 --accum-steps 1 --seq-len 256 --optimizer adamw --gpu \
  --activation swiglu --norm rmsnorm --positional-encoding learned \
  --epochs 3 --lr 0.0003 --weight-decay 0.1 --save-interval 0"

build/text_train.exe corpus_30mb.txt $COMMON                                    # C  全 f32
build/text_train.exe corpus_30mb.txt $COMMON --precision-param f32 --precision-compute f16   # E
build/text_train.exe corpus_30mb.txt $COMMON --precision-param f16 --precision-compute f32   # D
build/text_train.exe corpus_30mb.txt $COMMON --f16                              # B
build/text_train.exe corpus_30mb.txt $COMMON --f16 --weight-decay 0            # F（无收缩）
```

权重对比（对 `init_f16.bin` 逐元素取中位 `|run|/|init|`）用 numpy 解析 v5 `.bin` 即可，
解析脚本要点：`magic+version+u8 precision+spec_len → KeyValueRecord → [u8 tag][u64 rows][u64 cols][data]*`。

---

（工作区已还原：`CMakeLists.txt` 的临时改动已撤销，仓库只剩原有的 `D out.txt`、`?? tests/` 状态。）

---

## 8. 实施记录（2026-10-04：根修 + 回归测试落地）

### 8.1 ⚠ 勘误：§6 建议 1 的 `packHalf2x16` 方案**实测不成立**

本机（NVIDIA CMP 40HX）上 `packHalf2x16` 的 f32→f16 转换**同样按向零截断**，
与宿主 `nn::f16`（RNE）对拍 8 个探针值有 **4 个不一致**（`0.99997 → 0.9995117`、
`0.077 → 0.07696533`、`0.036 → 0.03598022`、`-0.077 → -0.07696533`）；
生成的 SPIR-V 已确认是 `OpExtInst PackHalf2x16`（非 `OpFConvert` 残留）。
规范侧对应 Khronos **Vulkan-Docs issue #1825**「PackHalf2x16 rounding behavior is
ambiguously defined」——该舍入**不是**规范保证的 RNE。

推论：**边界 cast（`cast.comp`）原本也在截断**，即 `engine.cast(f32→f16)`、
`from_matrix(..., F16)` 等所有 GPU 侧 f32→f16 转换都带同向偏差，本次一并修复。

### 8.2 实际采用的根修：整数位运算精确 RNE 预舍入

新增 **`shaders/nn_f16_rne.glsl`**（`nn_f16_rne(float)`）：在 f32 域用整数位运算做
一次精确 RNE，返回值本身落在 f16 网格上（f16 ⊂ f32，可精确表示）→ 随后的
`float16_t()` 收窄 / `packHalf2x16` 打包**无论驱动取哪种舍入模式都不再产生偏差**。
正规区用「加半个 ULP + tie-to-even 再截 13 位」；`|x| < 2^-14` 与 Inf/NaN 走
`roundEven(x·2^24)/2^24` 分支（f32 内精确）。

接入点（无一处 f16 写出再依赖驱动舍入）：

| 位置 | 接入 |
|------|------|
| `expr_glsl_gen.hpp`（5 个生成器：matmul / fold v1 / fold v2 / 逐元素 / 归约） | 所有 f16 写出点 + native16 的常量/参数收窄统一走 `glsl_f16_store()`；f16 变体 shader 前导发射 `nn_f16_rne` |
| 11 个手写 shader 的 `NN_WR`（matmul_tiled / matmul_gemv / batched_matmul / elementwise_v2 / im2col / col2im / outer_col / scan_*×4） | `#include "nn_f16_rne.glsl"` + `NN_WR(w) = float16_t(nn_f16_rne(w))` |
| `cast.comp`（f32→f16 边界 cast） | pack 前逐分量 RNE |
| `elementwise_v2.comp` 的 `NN_PACK4`（vec4 快路径） | 逐 lane RNE 后再 pack |
| `scatter_add.comp`（f16 原子累加写回） | pack 前逐 lane RNE |

`CMakeLists.txt` 的 `nn_embed_shader` 增加 `-I shaders` 与对
`shaders/nn_f16_rne.glsl` 的依赖（改动该头 → 全部 shader 重编）。

### 8.3 回归测试（`src/f16_writeback_probe.cpp`，已注册 ctest）

1. **cast 舍入对拍**：40 个 f32 值（8 固定边界值 + 32 确定性伪随机）经
   `engine.cast(f32→f16)` 后与宿主 `nn::f16`（RNE）**逐位一致**；
2. **f16 表达式 decay**：`dsl::compute_into(leaf(p)·rparam(0.99997), p)` 500 步后
   逐位不变（正确 RNE = 恒等）；
3. **AdamW(wd>0) 零梯度**：`p *= (1-lr·wd)` 500 步后权重逐位不变；
   并附 f32 对照（实测收缩 1.4879% ≈ 理论 `1-0.99997^500`）证明循环真实执行。

修复前（仅生成器换 `packHalf2x16` 后）GPU 仍失败 2 项、CPU 全过；改为软件 RNE 后
CPU/GPU 全部通过。

### 8.4 验收

- build 零告警；**ctest 23/23 全绿**（含 `f16_writeback_probe` / `f16_precision_test` /
  `gpu_f16_test` / `fused_gpu_test` / `expr_gpu_test` / 两个 `*_gpu` 用例）；
- **字节锚不变**：CPU `gpu_stability_probe --steps 20` = `6f8849f14da23110`、
  GPU = `8ef51b2927253c50`（均与历史锚逐位一致 → f32 路径零影响）；
- `bench/doc_inventory.ps1`：`L2-VIOLATIONS: 0`、virtual 49；
  `bench/doc_align_audit.ps1`：A/D/E/F 可行动项 0；
- `fused_registry.hpp` 的内容 hash 改变（内嵌 SPIR-V 字节），**结构集不变**
  （scan 仍为 91 条结构 / 130 条精度变体）——即"改结构不改数"判据下属预期。

### 8.5 已知边界

- f16 **次正规/溢出**边界：`nn_f16_rne` 已按 f16 网格处理次正规区；`-0.0` 经
  次正规分支可能归一为 `+0.0`（数值等价，不影响训练）。
- 未做：§6 建议 2（对 f16 参数跳过 decoupled decay）与建议 4 之外的**文档勘误**
  （`docs/development/05` §12.5、`AGENTS.md` 已知问题、`docs/development/14`）
  仍待补——根修已使原缺陷消失，但"f16 存储下 `lr·wd < 0.5 ULP` 时 decay 为恒等"
  这一语义值得写进精度文档。
