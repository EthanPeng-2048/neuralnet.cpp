# w1-layers2

> 摘自 RAPT/RLA、CNN 卷积池化、MLP、Softmax、GPT 模型层头文件的历史状态类注释（本轮清理）。
> 每条记录原位置、类型与内容；琐碎的一句话描述直接删除、未收录。

## 缓存前置校验的修复过程叙述（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:215-216、include/neuralnet.cpp/compute_layer_rapt.hpp:540-542）
- 类型：bug 根因
- 内容：Conv2D 原注释「不校验会拿空/上一 batch 的陈旧 im2col 静默算出错误梯度」、RLA-2 原注释「旧行为会把问题推到某个 matmul 里甚至算出垃圾梯度；这里立刻返回明确错误」——均为 2026-09-19 CNN 缓存契约加固与 RAPT backward 缓存校验的修复过程叙述（旧行为 = 空/陈旧缓存静默进 matmul）。改写为现行契约：backward 必须校验缓存命中且形状匹配、checkpoint/offload 模式下缓存为空必须立刻返回明确错误；错误文案（"checkpoint 模式需先 forward_recompute…"）原样保留。

## MaxPool2D 分组归约的逐通道循环与迁移标注（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:367-369）
- 类型：删除清单 / 迁移叙事
- 内容：原注释「迁 DSL grouped_reduce_max 视图（缺口 A，2026-09-27）：单表达式单 dispatch（此前引擎原语；更早版本按通道循环 C 次 dispatch）」（HEAD 版本另作「早期版本按通道循环 slice_rows + col_reduce_max + insert_rows，C 次 dispatch」）。演进链：按通道 3 类原语循环 C 次 dispatch → `engine.grouped_reduce_max` 单原语（2026-09-20，连带消除逐通道循环）→ `dsl::grouped_reduce_max` 视图（2026-09-27，R 进 expr_spec_key）。改为当前陈述：「单表达式单 dispatch 完成全通道分组归约（grouped_reduce_max 视图，R = 窗口面积 kk）」。

## MaxPool2D 并列梯度的 argmax-first 等价性论证与头部日期流水（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:265-273、418-420）
- 类型：等价性论证 / 日期流水
- 内容：原注释以「无并列最大值时与原 argmax-first 实现逐位一致」「无并列时与 argmax 散射逐位一致」论证与**已删除**的 argmax-first 散射实现等价，头部带日期「实现（全引擎化，2026-09-20）」，且 forward 一步仍写已不存在的「每通道 col_reduce_max」（现行 grouped_reduce_max）、backward 写「每通道 mask」（现为整张量单表达式）。改为当前语义直接陈述：无并列（cnt=1）时梯度全部落在唯一最大值位置、并列按 1/cnt 均分（总梯度守恒）；PyTorch 首个 argmax / TensorFlow 全量并列的对照保留为并列语义差异说明（删「历史上」）。

## 已不存在的 CPU to_matrix/from_matrix 路径描述（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:820-822、include/neuralnet.cpp/compute_layer_gpt.hpp:506-508）
- 类型：删除清单
- 内容：mlp.hpp 尾部 Conv2D 段落的说明「im2col/col2im 涉及复杂重排，沿用 PatchEmbedding 的先例在 CPU 端完成（to_matrix/from_matrix），GEMM 仍复用引擎 matmul 内核；MNIST 尺度下 CPU↔设备往返开销可忽略；GPU 融合卷积内核留作后续优化」描述的是 2026-09-20 CNN 全引擎化之前的 CPU im2col 路径（现 im2col/col2im 为设备原语、`col_cache_` 为设备张量，conv.hpp 同步注释「无 to_matrix/from_matrix 往返、无 CPU 标量循环」），整段删除（Conv2D 类段落本身留在原处，其布局/算法说明仍为当前实现）。gpt.hpp 的「token embedding 查表 + 位置 embedding 相加…此处用 to_matrix/from_matrix 在 CPU 端完成（batch 边界，PCIe 传输符合纯 GPU 架构约定）」同为已不存在的路径（现行 forward 全程 transpose/gather_rows/逐元素加，backward 用 scatter_add_rows），改为「查表与位置相加全程由引擎原语在设备端完成、无 CPU 中间拷贝」。

## Linear forward/backward 的 DSL 迁移标注与零回归论证（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:59、108-113、128-129、142、154-157）
- 类型：迁移叙事 / 删除清单
- 内容：2026-09-27 计算原语迁 DSL 时留下的标注——「迁移到 DSL（2026-09-27）：原 engine.matmul_with_bias（CPU/GPU 引擎内部实现本就是本表达式…）改为 Layer 直写」「（原 engine.matmul；scan 的 Linear backward dry-run 自动登记新结构）」「取代 matmul + accumulate 两次 dispatch」「累加步迁移为 dsl::compute_into（…取代 engine.accumulate）」「归约步迁移为 dsl::compute_reduce（原 engine.row_reduce_sum…）；累加步见下」「全 F32 配置下与迁移前逐字节一致（零回归）」全部删除/改写。保留当前契约：融合单 kernel（与 `engine.matmul_with_bias` 的 DSL 融合结构一致）、dry-run 自动登记该结构、归约与累加无法并入同一表达式的语义冲突理由、f16 经 PrecisionEngine 内部 f32 计算输出按目标精度舍入。

## SwiGLU 的 slice_rows/insert_rows 拆分描述（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:295-296、304-307、334、351）
- 类型：删除清单
- 内容：原注释写 gate/up 由 `slice_rows(x, 0, d_ff)` / `slice_rows(x, d_ff, d_ff)` 切出、「split/merge 用 slice_rows / insert_rows」「不再 slice_rows 物化半张量 → 消去 2 次 D2D 拷贝」「backward: 单 kernel 融合，消去 create+zero+2×insert_rows」——现行实现为 `dsl::row_access` 行视图（零拷贝）+ `select(Row() < d_ff, …)` 单表达式写出，不物化半张量也不经 insert_rows 拼装；改写为「行视图零拷贝、不物化半张量（无 D2D 拷贝）」「gate/up/s 经 RowAccess 行视图定位，两半由 select 一次写出」「单表达式融合，直接写出整张 grad_input (2*d_ff, batch)」。

## RLA-2 RMSNorm 反向缓存漏在 activation_cache 之外（原位置 include/neuralnet.cpp/compute_layer_rapt.hpp:352-353）
- 类型：bug 根因
- 内容：原注释「此前漏在 activation_cache 之外 → offload 无法覆盖它们；补齐后 offload 覆盖 backward 所需的全部激活」——RLA-2 RMSNorm 的 4 项反向缓存（Q/K normed、Q/K rms_inv）最初未登记进 `activation_cache()`，activation offload 因此换出不完整、backward 缺缓存。改写为现行契约：「契约：activation_cache 必须覆盖 backward 所需的全部激活，offload 才能完整换出并恢复」。

## GPT backward 逐块释放激活缓存的旧行为叙事（原位置 include/neuralnet.cpp/compute_layer_gpt.hpp:802-807）
- 类型：性能 A/B
- 内容：原注释「原先只在 checkpoint / offload 模式清理（默认路径不清理），于是每块激活一直驻留、累积到整个 backward 结束才被下一轮 forward 覆盖——探针实测这是 backward 段峰值主项（torch 的等价行为是"用完即释放"）」。删「原先…默认路径不清理」的旧行为描述，保留当前行为与理由：backward 后立即 `clear_cache()`，否则块激活驻留到整个 backward 结束（探针实测为 backward 段峰值主项），释放后内存可被后续块 backward 临时量复用、下一轮 forward 重新填充。

## 位置编码多态抽取叙事与不存在的标志位（原位置 include/neuralnet.cpp/compute_layer_gpt.hpp:262-264、36、426、941）
- 类型：删除清单
- 内容：PositionEncoder 段落原写「把 GPTModel 中**原本**按 PosEncodingType 散落的 if-else 位置编码逻辑**抽离**为多态层次…**消除 use_pos_emb_ / pos_emb_learnable_ 等标志位**分支的耦合」——两个标志位在代码中已不存在，抽取过程属于重构史，改为现在式「按 PosEncodingType 把位置编码逻辑组织为多态层次，GPTModel 通过基类指针使用，不依赖标志位分支」。同批琐碎项：「实现已抽到 compute_layer_base.hpp 的 ActivationOffloader」→「实现见 …」；「与 token_emb_ 共享同一 rng 序列（保持与旧实现完全一致的可复现性）」→「共享同一 rng 序列，保证跨 run 可复现」；「历文 K/V 不再重复投影」→「历史 K/V 复用缓存，不重复投影」（滑动窗口「丢弃最旧 token」「复用历史 K/V」等时序语义原样保留）。

## 「文档 22」悬空引用（原位置 include/neuralnet.cpp/compute_layer_rapt.hpp:35、411）
- 类型：删除清单（指向不存在文档的引用）
- 内容：注释引用「与原版 RLA 的关键差异（§3 文档 22）」「稳定数值分布…（文档 22 §4.3）」，但仓库内无"文档 22"编号；两处改指现存的 `docs/development/06-rapt-algorithm.md`（§3 家族演进即 RLA vs RLA-2 差异表，防神经元死亡 / RMSNorm 前置一并记录在该节）。RLA-2 与原版 RLA 的差异列表本身是当前算法定义，保留。

## "迁 dsl::…（原 engine.xxx）"与 dispatch 计数对照（原位置 conv.hpp:191、229、235、244、427；mlp.hpp:142；rapt.hpp:118、174、238-239、683、688-689、713-714、831-832、845、878-880）
- 类型：迁移叙事 / 性能 A/B
- 内容：2026-09-26~27 计算原语迁 DSL 后注释里残留的演进标注——「归约步迁 dsl::compute_reduce（原 engine.row_reduce_sum）」「纯 matmul 迁 DSL 直写（原 engine.matmul）」「（迁 DSL grouped_reduce_sum 视图，同 forward）」「原地累加迁 dsl::compute_into」「迁 dsl::matmul(batch)：…」「（GPU 上 2 → 1）/（3 → 1 次 dispatch）/（原 zero + broadcast_row_inplace + elementwise Add 三次 dispatch → 1 次）」「取代 matmul + accumulate 两次 dispatch」；rapt 另有「（原为 term/diff/gx 三个独立 kernel + 一个逐元素原语）」「原"乘 x² → 列归约 → 乘 1/dk → 加 eps → rsqrt"五步压成两步 DSL」「不再吞错后让…」（ensure_ones_ 的错误传播）「同一个 AOT 键…（此前的 eager elementwise_binary 已退役）」。删除后保留的当前陈述：单次 dispatch / 不物化中间张量的融合事实、该表达式与哪个 dry-run 同 `expr_spec_key`（闭合世界注册来源）、分两步执行的原因（归约融合 shader 不支持归约后逐元素后处理，GPU 侧显式拒绝）、错误传播避免根因被 "ensure_gpu: invalid tensor" 掩盖。
