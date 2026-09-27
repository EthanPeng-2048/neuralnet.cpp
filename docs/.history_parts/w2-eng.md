# w2-eng

## f16 次正规转换守卫的移位 UB（原位置 include/neuralnet.cpp/precision.hpp:197，整改前行号）
- 类型：bug 根因
- 内容：`float_to_half_bits` 的次正规分支曾误写守卫 `exp <= -46`，与其自身公式矛盾（公式只对 exp ∈ [-25,-15]、shift 14..24 合法）。exp ∈ [-45,-33]（|v| ∈ 2.8e-14 ~ 1.2e-10）时 shift ≥ 32 → uint32 移位 UB（x86 按 5 位掩码 → d 取满 24 位尾数 → 指数字段回绕成垃圾 half：512/8192/11776/18432，随机落 e=31 即 NaN）。梯度恰在该量级、前向激活 ~0.1 从不落入，故损坏只见于 f16 梯度写回、若干步后更新爆炸；GPU 走硬件转换故"只有 CPU 错"，单算子测试输入 ≥1e-3 故全过。修复 = 守卫改 `exp <= -26`（exp ≤ -26 时 D < 0.5 恒 flush 为 0，无需移位）。整改后注释只保留当前守卫的正确性论证（合法 shift 区间 14..24 < 32、exp ≤ -26 恒 0），删掉"曾误写成"叙事。

## `--f16` CLI 语义与 profile 混淆（原位置 include/neuralnet.cpp/precision.hpp:477、compute_optimizer.hpp:52）
- 类型：语义修正（历史误会）
- 内容：`profile_master_weights()` 曾是 `--f16` 的语义，曾造成"传了 --f16 却是 f32 主权重混合精度"的误解；后修正为 `--f16` = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}（docs 05 §9.4）。`compute_optimizer.hpp` 的状态精度注释也曾把 `profile_all_f16` 标注为"CLI --f16"。整改后两处只保留当前事实：三种 f16 相关 profile（f16 / master_weights / all_f16）各自的字段与区分，删除误会叙事。

## Adam m/v 状态更新的旧 clone 开销（原位置 include/neuralnet.cpp/compute_optimizer.hpp:308）
- 类型：性能 A/B / 删除清单
- 内容：旧实现对 m/v 各做一次 `clone_tensor`（整份模型尺寸的 vkCmdCopyBuffer，逐步 2×model_size 的 Copy）+ 多次逐元素原语 + 多个中间张量；融合改造后为三个 DSL 融合 kernel（K1/K2/K3），超参经 RParam 承载，无任何 clone、无 m_hat/v_hat/sqrt_v/denom/ratio 物化。整改后注释只描述当前三 kernel 结构与 RParam 理由，删掉与旧实现的对比。

## Newton-Schulz 母矩阵边选择（原位置 include/neuralnet.cpp/compute_optimizer.hpp:537）
- 类型：性能 A/B（OOM 根因）
- 内容：旧构造沿长边做 A/A²——对 50257×1024 词嵌入即 50257² ≈ 10GB/个 OOM（Muon 显存 > AdamW 的根因）。当前实现恒选短边：母矩阵 min(m,n)²（该例 1024² ≈ 4MB）。整改后注释保留"为何短边侧构造在数学与显存上都成立"的当前推理，删掉"旧实现…新实现降至"对比。

## Newton-Schulz 每步 dispatch 计数（原位置 include/neuralnet.cpp/compute_optimizer.hpp:549、561）
- 类型：性能 A/B
- 内容：`A = b·A + c·A²` 原为 scale(A²,c) + scale(A,b) + add(A,A²) 三次 dispatch（原实现合计 7 次/步），融合为一次原地目标传递后合计 5 次/步。整改后注释只保留当前"每步 5 次 dispatch"的构成与"单 kernel 原地目标传递"，删掉 7→5 演进叙事。

## 稠密 CE 旧列广播路径（原位置 include/neuralnet.cpp/compute_loss.hpp:180）
- 类型：性能 A/B / 删除清单
- 内容：旧路径 = clone 一份 (classes,batch) 整块 vkCmdCopyBuffer + 3 次就地广播 dispatch；现 shifted/softmax/log_softmax 全部为列广播表达式，各 1 个融合 kernel、零中间拷贝。整改后注释只陈述当前"无中间拷贝"的事实。

## CE 稀疏/稠密路径引用已删除原语（原位置 include/neuralnet.cpp/compute_loss.hpp:274、277、332）
- 类型：删除清单
- 内容：注释曾引用 `col_softmax_denom`、`col_softmax_sparse_forward`（M5 期 op-level 原语，已随 S7 删除，见 docs/development/02 §"S7 删除 M4-M6"）以及"融合原语不可用（如 CUDA 未对齐返回未实现）"的 CUDA 例子（CUDA 后端已整体移除）。当前实现为 S7 IR 组合（col_max 原语 + denom/loss_vec/grad 表达式）；labels 以 (1,total) 浮点打包，前置条件 vocab_size ≤ 2^24，失败直接透传错误。整改后注释改为当前结构 + 当前前置条件。

## CpuEngine::matmul_with_bias 覆盖前的 GPU host 往返（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:673）
- 类型：bug 根因 / 性能 A/B
- 内容：此前 GPU Linear 每步做 to_matrix/from_matrix 的 CPU 往返（走基类默认"matmul + host 加 bias"），且该 DSL 结构从未被 scan_exprs 扫描 → "Linear 结构"融合测试 AOT 未命中。现 CPU/GPU 两引擎均以 DSL 表达该结构，scan dry-run 收集 → gen_fused 生成融合 kernel。整改后注释只保留"覆盖基类默认 + scan 收集该结构（否则闭合世界缺 key）"的当前理由。

## fold 视图白名单两处清单曾失同步（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1406）
- 类型：bug 根因
- 内容：CPU 解释器（compute_cpu_engine）与 expr_spec 的 fold validate 两处白名单曾漏同步，致 RowBroadcast(doc_col) 被拒、forward 报错被 scan dry-run 的 `(void)` 吞掉、backward 拿空缓存触发 NN_ASSERT。整改后注释保留"两处清单必须同步维护"的当前契约与后果，删掉事故叙事（dry-run 吞错已另修）。

## fold 标签表只查 rows 的越界读（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1421）
- 类型：bug 根因
- 内容：标签表（BatchMod/BatchCol）此前只校验 rows，形状违约时静默越界读；现守卫 cols（BatchMod: cols ≥ param；BatchCol: cols ≥ (batch-1)*param + K）。整改后注释保留当前守卫规则与"只查 rows 不足"的理由。

## fold 状态复位与 vecd 跨行残留（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1589）
- 类型：bug 根因
- 内容：vecd 等临时/行向量态槽曾在循环外声明，跨行残留累加、err 随行数滚雪球；现每行开头统一复位状态初值 + 归约槽 + 行向量态。整改后注释保留当前"每行独立复位"的理由，删掉"曾漏在循环外声明"的修复叙事。

## CPU 解释器拒收 f16 输入（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1711）
- 类型：bug 根因
- 内容：解释器按 f32 存储读取 span，f16 输入直读 = 空指针 UB，曾是 CPU f16 训练 NaN/AV 家族成员；现为响亮报错（"CPU 解释器仅支持 f32 输入"）。整改后注释保留"直读即 UB，故响亮报错"的当前理由，删掉历史故障家族标注。

## PrecisionEngine 头部"零 shader 变体"论述已被推翻（原位置 include/neuralnet.cpp/compute_precision_engine.hpp:19-24）
- 类型：否决方案 / 性能 A/B
- 内容：头注释原主张"不做引擎内逐原语 f16 分支（§11.1 Phase 2 才做）、边界 cast 换零 shader 变体/零注册表膨胀、收益仅在带宽（显存占用由存储决定）"。后续实测（docs 05 §12.8-§12.11）证明逐算子边界 cast 使 transient 膨胀约 2.4×、f16 峰值反超 f32；带类型 in-kernel f16 变体（逐元素/归约/matmul 段，gen_fused 54 变体）落地后峰值显著下降。整改后头注释改为当前结构：**变体优先（f16 直读直写、零边界临时量）→ 无变体时回退边界 cast（任何 (结构,签名) 都有正确路径，代价是带宽 + f32 临时量）**，并保留"为何不做逐原语手写 f16 分支"的现行理由；同时 in-place 原语清单删去已不存在的 `axpy`/`broadcast_*`。

## compute_engine 接口注释中的 CUDA 残留（原位置 include/neuralnet.cpp/compute_engine.hpp:89、93）
- 类型：删除清单
- 内容：`release_idle_pool_blocks` / `pool_stats` 注释曾写"CPU/CUDA 引擎"——CUDA 后端已整体移除（见 AGENTS §2），现仅 CPU（no-op/返回空）与 GPU。已改为当前引擎清单。

## Tensor reshape 头注释与实现不符（原位置 include/neuralnet.cpp/compute_tensor.hpp:291）
- 类型：错误注释（文档失真）
- 内容：头注释原称"零拷贝 reshape（共享底层 buffer）——CPU/Vulkan 均适用"，但 CPU 分支并无零拷贝视图能力、实际复制数据（与 AGENTS §10.6 记载一致）。整改后改为如实记录当前语义：GPU 共享 buffer（零拷贝）、CPU 复制到新形状 Matrix。
