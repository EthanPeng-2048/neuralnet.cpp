# w1-attn

> 摘自注意力 / ZiPT / Transformer / FeedForward / 层基类头文件的历史状态类注释（本轮清理）。
> 每条记录原位置、类型与内容；琐碎的一句话描述直接删除、未收录。

## fold 注意力构造从 expr_fold.hpp 迁出（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:161）
- 类型：迁移叙事 / 删除清单
- 内容：`make_fold_attn_o` 与 `FoldAttnMask` 5 种掩码变体原定义在 `expr_fold.hpp`，fold 头因此带上了 attn 命名与掩码设计；2026-09-25 前按 AOT"表达式文本只出现在 Layer"原则迁入 `compute_layer_attention.hpp`，`expr_fold.hpp` 只保留 rowmax/rowsum/softmax_denom 等与注意力无关的通用样例。头文件注释改为现在式：本文件是该构造与掩码变体的定义来源。

## 掩码物化路径与两趟 forward 分支删除（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:490-504、697、394、1128、1196）
- 类型：删除清单
- 内容：P-C2-7（2026-09-23）把注意力 forward 收敛为单 fold kernel 后，`apply_mask_` 物化掩码钩子、`two_pass_mask_` 决策钩子（返回 `{use_two_pass, bias}` 组合式 `AttnBias` 描述子）与旧物化/两趟 forward 分支整体删除；掩码现恒在 fold body / `recompute_W_` 掩码树内表达，绝不物化 (BH·seq, seq) 掩码矩阵，只保留增量推理的 `apply_mask_step_`。原注释中的"与迁移前 apply_mask_ 默认 no-op 同义""原 two_pass_mask_ 的构建体原样保留、职责纯化""与旧 apply_mask_ override 语义同构"等对照论证随之删除。

## P-C2-7 / P0-5 计划项标签摘除（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:392、440、697、742、791、810）
- 类型：修复叙事
- 内容：原注释多处带"（P-C2-7 正确性修复）""（P0-5：从 Q/K 重算 W…）""P-C2-7 起 m/l 也不缓存""（P-C2-7：旧 softmax.backward 分支不保留）"等计划项/修复标签。P0-5 在 docs 中无对应条目（悬空引用）；P-C2-7 的演进叙事已完整记录于 `docs/development/02-operator-fusion.md`（§关键算法状态横幅、§注意力 M6→S7→P-C2-7）。头文件注释改为直接陈述当前行为：forward 恒单 fold 路径、W/m/l/attn 均不缓存、`recompute_W_` 内部 softmax 单表达式归一化、反向走 R/X 路径。

## recompute_W_ 裸 matmul 的 dry-run 覆盖（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:458）
- 类型：bug 根因
- 内容：MHA（Plain）分支的裸 matmul 结构历史上从未被 scan dry-run 覆盖，AOT 闭合世界下会运行期硬报错；补 MHA dry-run 后由 `scan_exprs` 登记。注释改为契约表述"该结构必须有 dry-run 覆盖"。

## ALiBi 斜率表形状约束（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:1086、1138）
- 类型：bug 根因
- 内容：斜率表原为 (1, num_heads)，而 fold 的 `batch_mod(BH)` 按网格下标 b*H+h 直读 → batch≥2 时 b≥1 越界读、ALiBi 静默错；batch=1 时下标恒 <H 恰好掩蔽，历史单样本测试未暴露该缺陷。修复为 (1, batch*H)、按 (b,h) 块重复 slopes_[h]。旧 recompute 的 `batch_mod(%H)` 与增量推理按 h∈[0,H) 读均落首块，语义不变。头文件保留"表长必须 = batch*H 及其后果"的约束理由，删除"旧…表"实现史。

## doc_ids 缓存布局演进（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:1157）
- 类型：删除清单
- 内容：`doc_ids_cache_` 布局从 M5 时代 AttnBias 的 (1, batch*seq) 改为 S7 的 (1, BH*seq)（BatchCol 视图的 batch 下标 = BH 网格下标，按 (b,h) 块重复 doc_ids[b*seq..]），旧布局 S7 起不再适用。注释只保留"为什么必须 (1, BH*seq)"。

## 精度 profile 下传的历史缺陷（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:540、compute_layer_feedforward.hpp:75）
- 类型：bug 根因
- 内容：Phase 2 精度接线时 AttentionBase/FeedForward 等复合层最初无人把 PrecisionProfile 下传给子层，f16 配置下子层静默停在 f32（参数按 F32 创建、DSL 求值退回 F32）。注释改写为契约："复合层必须把 profile 下传给子层，否则子层静默停在 f32、f16 配置下静默失效"。

## scale 原地缩放写法迁移（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:687）
- 类型：迁移叙事 / 性能 A/B
- 内容：2026-09-27 `engine.scale_inplace` 改为 `dsl::compute_into`（零分配原地写）。旧注释"DSL 表达式多分配一整块缓冲更慢"针对的是 `dsl::compute` **分配版**，与 compute_into 无关。头文件改为直接写"用 compute_into（零分配原地写）；dsl::compute 是分配版，另分配一整块缓冲"。

## 三路梯度累加的等价性论证（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:914）
- 类型：性能 A/B
- 内容：原注释以"与原实现一致（(giq+gik)+giv）→ 逐字节等价"论证 clone + 两次 `add_inplace` 改为单次 `dsl::compute_into` 的正确性。改为对当前实现的直接陈述："结合顺序固定为 (giq+gik)+giv —— 求和顺序确定，结果可复现"。

## DSL 迁移标注（"迁 dsl::…"）（原位置 attention:796/831/845/852/994/996/1015、zipt:132/142/477/484/494/555、transformer:391/397/418/423）
- 类型：迁移叙事
- 内容：计算原语全量迁 DSL（2026-09-27）后，注释普遍带"迁 dsl::matmul(batch)（原 engine.xxx）"式迁移说明，其中"与原 shader 写出时乘等价""原 alpha 写出时乘，等价"等与旧实现的等价性论证删除。保留的是当前契约：结构 key 与哪个 dry-run 同 key、rparam 值不进 `expr_spec_key`、闭合世界注册来源。

## ZiPT 联合注意力反向镜像标注（原位置 include/neuralnet.cpp/compute_layer_zipt.hpp:547）
- 类型：删除清单
- 内容：ZiPTBlock 联合注意力反向原注释标"（materialized 路径，镜像 AttentionBase 旧路径）"；AttentionBase 的旧物化反向路径已删，现 ZiPT 自成一条 materialized 路径（softmax `output_cache()` 持有 A + batched matmul）。删去镜像标注。

## ZiPT W/L 双模式命名（原位置 include/neuralnet.cpp/compute_layer_zipt.hpp:656、657、659、675、678、766、922、989、1035、1036）
- 类型：删除清单
- 内容："旧行为/新行为"命名对（W=L 无压缩 vs W<L 历史/窗口分离）改为当前态命名"兼容模式 / split（W<L）模式"；`W==0 或 W>seq_len → 回退 W=L` 是当前仍执行的兼容分支，改述为"缺省/越界时无压缩"。

## Transformer 尾部遗留的 CausalSelfAttention 类头注释（原位置 include/neuralnet.cpp/compute_layer_transformer.hpp:585-610）
- 类型：删除清单
- 内容：文件尾部遗留整段 CSA 类头注释（类本体定义在 `compute_layer_attention.hpp`，同名注释在该处保留改写版），且其中"在 scale 之后施加预计算掩码 S += mask (batch*H*seq, seq)"描述的物化掩码路径已不存在——整段删除。attention 内同段改为"掩码恒在 fold body / recompute_W_ 掩码树内逐块表达，绝不物化"。

## TransformerEncoder 池化反向旧写法（原位置 include/neuralnet.cpp/compute_layer_transformer.hpp:413）
- 类型：删除清单
- 内容：反向池化注释原列有"`scale(inv_n) → matmul(ones_row_, result_T)` … 直接：`matmul(grad_col_vec, ones_row_)`"的新旧两套写法，旧写法删除，只留现行管线（rearrange → 原地 scale(inv_n) → matmul(grad_col_vec, ones_row_) → rearrange 回）。

## ActivationOffloader slab 容量校验（原位置 include/neuralnet.cpp/compute_layer_base.hpp:196）
- 类型：bug 根因
- 内容：slab 原先只在 `!slab_.valid()` 时按当时的激活总量分配、之后永不增长；后续 step 激活总量变大（批大小/序列长度变化、--resume 后续训、最后一个不满 batch 之后的 step）时 `offload_save` 按新 offset 越界写 slab → 缓冲区破坏/设备丢失。修复为每次导出按当前总量校验、不足即重建。头文件只保留现行不变量与其后果。
