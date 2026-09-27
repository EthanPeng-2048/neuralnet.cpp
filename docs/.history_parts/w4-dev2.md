# w4-dev2

`docs/development/` 三个文档（02-operator-fusion / 03-ir-optimization / 05-mixed-precision）
整改时移出的历史记录。整改原则：正文只记录当前状态、当前机制与仍然适用的结论；本文件保存
被移出的演进流水、分期验收记录、被否决方案与性能 A/B 过程，供追溯用。

## 两期主线与分期状态横幅（原位置 docs/development/02-operator-fusion.md 篇首导语）
- 类型：演进记录
- 内容：原文把两期工作串成流水：「一期（M1-M7）用手写 op 级融合原语解决 GPT+Vulkan 训练显存/开销问题；
  二期（S1-S7）把 matmul 纳入 IR 融合、实现跨 kernel 自动融合（原规划的图级缓存落地方案已随 IR-C 于
  2026-09-19 删除），最终删除一期手写融合原语。状态：一期已实施完成（M4-M6 手写原语已由二期 IR 融合取代）；
  二期 S1-S5、S7 已实施，S6 自动窗口因用户决策搁置（其替代方案 P2-12 图级缓存亦随 IR-C 删除）；
  P-C2-7 起注意力 forward 进一步改为单 fold kernel。」正文导语已改为现在式（M4-M6 不存在、S6 不接线、
  forward = 单 fold kernel）。

## 表达式录制 API（begin_expr/end_expr）的设计与否证（原位置 02 §表达式录制与融合边界（已移除））
- 类型：被否决方案 / 取舍记录
- 内容：一期曾用 `begin_expr/end_expr` 作"计算级融合"入口（`begin_batch/end_batch` 是提交级）：
  `begin_expr` 进入录制、`end_expr` 做融合分析（构建虚拟寄存器 DAG + 判定融合边界）——小中间量留寄存器、
  逐元素链并入同一 kernel、大张量 spill 成下一 kernel 输入，CPU 端为 no-op。**最终不用它**（逐条经代码核对）：
  ① 融合条件要求"两节点均无归约、同形状、tail 恰好一个消费者"，而 LayerNorm/RMSNorm/Softmax/Attention
  的骨架是"归约 → 逐元素 → 归约"，归约处即中断；② 需要融的层都要为 backward 缓存中间量
  （`normalized_cache_`/`normed_cache_`/`residual2_cache_`/`W_re`）——缓存就是图外的第二个消费者，
  录制图看不见它，补逃逸检测（Tensor 拷贝钩子 + 自动作用域 + ~22 个 flush 点）远超收益；
  ③ 能融的长逐元素链本来就能写成**一个** `dsl::compute`；④ 唯一使用方是演示层 `FusedChainLayer`，无生产调用方。
  **删除项**：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`、`dsl::start_expr/end_expr`（`ExprBlock`）、
  `FusedChainLayer`、`Tensor::virtual_tag_`；**保留** `run_fused_gpu` 的 `output_override`
  （现服务于 `dsl::compute_into` 原地目标传递，与图 IR 无关）。

## 两趟式注意力 forward 三原语与 M4/M6 实施坑（原位置 02 §关键算法「两趟式注意力（Forward）」）
- 类型：演进记录 / 已删实现
- 内容：一期用三个 op 级融合原语（M4/M6，二期 S7 后删除）实现不物化的两趟 forward：
  Pass1a `engine.batched_matmul_reduce(Q,K,BH,ReduceOp::Max,transA=true,…,reduce_cols=true)` 求 m；
  Pass1b `engine.batched_matmul_softmax_denom(Q,K,m,…)` 求 l = Σ exp(alpha·QᵀK − m)；
  Pass2 `engine.batched_matmul_softmax_apply(Q,K,V,m,l,…)` 逐 tile 累加 O。显存收益：`scores`/`masked`/
  `attn_cache_` 三份 `BH·seq×seq` 消失，只剩 `m`/`l`（`BH·seq`）与 `O`；代价是 QᵀK 算两遍（2× FLOPs），
  该代价在 fold 单 kernel 后消失。掩码描述子：`apply_mask_` 钩子与 `two_pass_mask_` 决策钩子返回
  `{use_two_pass, bias}` 组合式 `AttnBias`（已删除）。M4/M6 实施坑：
  - 掩码用共享 (M,N) 张量（`-inf` 屏蔽 + 有限偏置），不在引擎硬编码因果/ALiBi；`bmm_apply` 共享内存存
    W 行（N≤4096），Phase1 算 W、Phase2 累加 O。
  - `dispatch_bmm_generic` 输出约定：返回张量永远最后一个 binding、out 参数倒数第二
    （曾写反导致 GPU grad_K 与 grad_V 互换）。
  - kv_backward 的 `A_b[:,i]` 索引：transA 时 `A_b[k][i] = flat[b*K*M + k*M + i]`；!transA 时
    `A_b[i][k] = flat[b*M*K + i*K + k]`（CPU 参考曾把两分支写反，以 shader/前向 dot_ab 约定为准）。
  - V/G 布局转换：`transpose → rearrange_3d(seq, BH, d_k, false)` 实现按 batch 转置。
  - S7 曾用 `batched_matmul_softmax_backward_q/kv` 两融合原语做 backward（已随 S7 删除）。

## 稀疏 CE 手写原语期（原位置 02 §关键算法「稀疏交叉熵（M5）」）
- 类型：演进记录 / 已删实现
- 内容：一期用 `col_softmax_denom` + `col_softmax_sparse_forward` 两个原语，单 kernel 同时算稠密梯度与
  标签位置 loss_vec；M5 期的坑：labels 以 `(1,N)` float 上传（vocab_size ≤ 2^24 精确表示，kernel 内
  `uint(labels[i])` 读取，越界/被 mask 列整列置 0）；`dispatch_bmm_generic` 的 binding 顺序；
  融合路径与回退路径的 num_valid 判定必须一致。二期 S7 后改由 IR 表达（`col_reduce_max + exp + row_gather`）；
  labels/loss_mask 以 `(1,total)` 浮点打包这一条**仍是当前机制**（见 compute_loss.hpp）。

## 跨 kernel 自动融合：S6 自动窗口与 P2-12 图级缓存（原位置 02 §跨 kernel 自动融合）
- 类型：被否决方案
- 内容：一期曾规划用 `begin_expr/end_expr` 做跨表达式融合（仅演示 Layer `FusedChainLayer` 使用；真实
  Layer 每表达式独立 dispatch）。二期 S6（P2-10 自动窗口）方案：`GpuEngine` 维护线程局部
  `std::optional<ExprGraph>` 窗口，`eval_expr/eval_expr_reduce` 调用时尝试并入、不兼容时 flush——该方案搁置；
  随后落地的 P2-12 图级缓存（`graph_cache_key`/`plan_from_kernel`/`instantiate_plan`，跨 step 复用融合分析）
  也随 IR-C 于 2026-09-19 删除。当前结论不变：不接线，能融的写成单个 `dsl::compute`。

## 二期 S1-S7 的依赖关系与删除顺序（原位置 02 §二期 S1-S7 表尾）
- 类型：演进记录
- 内容：依赖关系：S1→S2→S3 串行（IR → CPU → GPU 生成）；S4 依赖 S3；S5 依赖 S3；S7 依赖 S5。
  删除顺序：先删 GPU shader（`bmm_*.comp` / `col_softmax_*.comp`），再删 `ComputeEngine` 虚接口，
  最后删 CPU/CUDA 实现与 `vk_backend` dispatch；每步删除前对应 IR 融合路径已在测试覆盖。
  S6 行原记「搁置（用户决策）；替代方案 P2-12 图级缓存亦随 IR-C 于 2026-09-19 删除」。
  S7 行原记删除 7 个原语（`batched_matmul_reduce/softmax_denom/softmax_apply`、
  `batched_matmul_softmax_backward_q/kv`、`col_softmax_denom/col_softmax_sparse_forward`）
  的接口 + CPU/GPU 实现 + 7 个手写 shader + vk pipeline + 测试改造。

## 一期 M1-M7 里程碑表与当时验收（原位置 02 §里程碑与实施记录 一期 M1-M7）
- 类型：演进记录
- 内容（含当时测试名，现均已并入聚合目标）：
  | 里程碑 | 内容 | 验证（当时目标名） |
  |---|---|---|
  | M1 | `ExprSpec` 加归约视图/指令 + CPU `eval_expr` 扩展 | `expr_reduce_test` + `expr_dsl_test` |
  | M2 | `begin_expr/end_expr` 录制框架 + CPU no-op | 2026-09-19 随 IR-C 删除 |
  | M3 | Softmax/LayerNorm/RMSNorm fwd/bwd 改 DSL 归约表达式 + GPU 归约融合 shader | `fused_gpu_test` + gradcheck |
  | M4 | 三个 matmul 融合原语（bmm_reduce/denom/apply） | `matmul_fusion_test`（CPU err=0 / GPU err≤1.9e-6） |
  | M5 | CrossEntropyLoss 稀疏融合 | `ce_fusion_test`（CPU err=0 / GPU err≤4.8e-7） |
  | M6 | Attention 两趟式（forward + 反向重算 W） | `matmul_fusion_test` 8 用例 + gradcheck + 训练 |
  | M7 | 文档补"原语可专、不叫算法名"约定 | 全套测试 |
  测试名归并：`matmul_fusion_test` 已无同名目标（用例在 `expr_cpu_test`/`expr_gpu_test`）；
  `expr_dsl_test`/`expr_reduce_test`/`expr_matmul_test`/`expr_opt_test` 并入 `expr_cpu_test`；
  `fused_gpu_test`/`tensor_expr_test` 并入 `expr_gpu_test`。
  形状无关融合的原始效果记录：扫描表达式 24 → 18 条（RoPE 4×dk 去重）。

## 教训条目「dispatch_compute 误删后从调用点重建」（原位置 02 §二期关键教训 7）
- 类型：演进记录（修复叙事）
- 内容：一次把 `compute_vk_backend.hpp` 的 `dispatch_compute` 误删后，按调用点重建。
  正文已把该条改写为现在式提醒（改签名/删除时按调用点逐一核对）。

## 教训 1 的旧示例写法（原位置 02 §二期关键教训 1）
- 类型：演进记录
- 内容：原文为「`scale` 折进 Q（forward `scale_inplace` + backward 补乘）、`inv_num_valid` 后置
  `scale_inplace`」。核对当前代码：**scale 折进 Q + backward 补乘仍是当前做法**，但载体已从引擎
  `scale_inplace` 改为 `dsl::compute_into(engine, leaf(Q) * dsl::rparam(scale_), Q)`
  （`compute_layer_attention.hpp`，rparam 值不进 key）；`inv_num_valid` 也改为 CE 表达式内的
  `dsl::rparam(inv_num_valid)` 尾链（`compute_loss.hpp`），不再后置 `scale_inplace`。
  正文教训 1 已按当前写法改写。另：形状无关融合的原始效果记录（扫描条目 24 → 18，RoPE 4×dk 去重）
  与教训 5 的「74 条 spec 全生成」计数已从正文移除（计数以构建输出为准）。

## IR-C（图 IR + begin_expr/end_expr 融合分析）取舍记录（原位置 docs/development/03-ir-optimization.md §5.3）
- 类型：取舍记录 / 被否决方案
- 内容：阶段 C 的落地形态 = `expr_graph.hpp`（`ExprGraph`/`fuse_expr_graph`/`recording_graph_owner`/图级缓存）
  + `ComputeEngine::begin_expr/end_expr` 显式录制 API + `dsl::start_expr/end_expr`（`ExprBlock`）
  + 演示层 `FusedChainLayer`，基础版 2026-08-24 实施、08-26 增补中间张量消除。
  **2026-09-19 决定：不接线，整体删除。** 依据（逐条经代码核对）：
  1. **归约是硬边界**。`fuse_expr_graph` 只拼接纯逐元素链（两节点均无归约、同形状、tail 恰好一个消费者
     且以 Linear 视图消费）；LayerNorm/RMSNorm 前向是 `col_reduce → 逐元素 → col_reduce`、Softmax 是
     `row_max/row_sum → 逐元素`、Attention 的 m/l/W 同理——需要融合的层恰好以归约为骨架。
  2. **backward 缓存逃逸**。融合要求"tail 只有一个消费者"，但要融的层都要为 backward 缓存中间量
     （`LayerNorm::normalized_cache_`、`RMSNorm::normed_cache_`、`GPTBlock::residual2_cache_`、
     Attention `W_re`）——缓存是图外第二个消费者，录制图看不见它；补逃逸检测需 Tensor 拷贝钩子 +
     自动作用域 + ~22 个 flush 点 + `Layer::forward` 非虚壳（26+26 处），成本远高于收益。
  3. **能融的地方早已写成单个表达式**。GeLU/SwiGLU/ReLU/Softmax/RoPE/掩码各是一个 `dsl::compute`；
     当前层集合里不存在"写不进一行"的纯逐元素链。
  4. **CPU 零收益**。`CpuEngine` 的 begin/end 本身是 no-op。
  5. **无生产调用方 → 悬空设计**。唯一使用方是演示层 `FusedChainLayer` + scan dry-run + 两个测试。
  **删除项**：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr` 及 CPU/GPU 实现、
  `dsl::start_expr/end_expr`（`ExprBlock`）、`FusedChainLayer`、`Tensor::virtual_tag_`、
  `GpuEngine::execute_fused_graph` 与图级计划缓存、`expr_graph_test`/`expr_fuse_test`
  与 scan 的 `FusedChainLayer` dry-run 段。
  **保留项**：`run_fused_gpu` 的 `output_override`（服务 `dsl::compute_into`，与图 IR 无关）。
  **重新立项的前提**：出现或写出这样一条链——纯逐元素、同形状、中间量不必为 backward 保留、
  且长到单个 `dsl::compute` 写不下；届时按"逃逸检测 + 自动作用域 + 自动 flush"实现，
  而不是复活显式录制 API。
  （原正文另有指针：本记录亦是 AGENTS/多份文档所指"03 §5.3 末段"的重新立项前提。）

## IR-A/B 实施记录的日期与当时验收（原位置 03 §实施记录（2026-08-23，IR-A + IR-B 已完成））
- 类型：演进记录
- 内容：落地日期 2026-08-23；当时验证清单：新增 `src/expr_opt_test`；`expr_dsl_test`/`expr_reduce_test`/
  `tensor_expr_test`/`fused_gpu_test`/`matmul_fusion_test`/`ce_fusion_test`/gradcheck 系列/gpt_checkpoint_test
  全绿；MNIST + text_train（CPU/GPU、含 checkpoint-every）端到端正常（这些目标名现已并入
  `expr_cpu_test`/`expr_gpu_test` 聚合目标）。顺带修复的与 IR 无关 bug：`text_train --save-interval 0`
  触发 `(step+1) % 0` 整数除零崩溃。
  （实现要点、关键不变量、关键坑 ①② 已作为当前事实保留在正文。）

## CpuEmitter 的存在与删除（原位置 03 §后端 emitter 抽象「注意」块 + §当前状态小结表行）
- 类型：演进记录 / 已删实现
- 内容：历史上存在 `cpu_emitter.hpp`（生成可编译 C++ 直线代码），其产物从不参与编译
  （gen_fused 硬编码 glsl）、缺陷全隐性，随文件一并删除；当时 `expr_emitter.hpp:16` 注释仍提到
  `cpu_emitter.hpp`（该注释残留后来也已清理）。早期"Phase 2 前必须先修 CpuEmitter 缺陷"的前置条件随之作废。
  工作量小结原表（含计划期估计）：IR-A 小（1–2 天）/低；IR-B 中（3–5 天）/中；IR-C 大（1–2 周）/高；
  IR-D 中（3–5 天）/中。

## 05 §12.5 边界 cast 期的实测 A/B 与探针数据（原位置 docs/development/05-mixed-precision.md §12.5）
- 类型：性能 A/B
- 内容（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq 256、adam lr 1e-3）：
  | 配置 | batch | 峰值 | 耗时 | avg_loss |
  |---|---|---|---|---|
  | f32 基线 | 64 | 3069 MiB | 4.7s | 7.32 |
  | `--f16`（profile_f16，边界 cast 期） | 64 | 2831 MiB 后 OOM | — | — |
  | f32 基线 | 32 | 1588 MiB | 4.7s | 6.6726 |
  | `--f16` | 32 | 4124 MiB（2.6×） | 6.2s（+32%） | 6.7086 |
  （batch32 轨迹一致：avg_loss 6.6726 vs 6.7086，偏差 0.5%，数值正确；但峰值 2.6× → 边界 cast
  在训练峰值上净亏。）探针逐阶段（transient live / pending）：
  step0/forward 1027/230 vs 1574/1100；step0/loss-fwd 1541/230 vs 2088/1100；
  step0/backward 2795/2300 vs **6687/7000**。
  根因：每个算子都要把 f16 入参抬 f32 副本——被 k 个算子读取的张量要 k 份副本；
  `(vocab, batch·seq)` 的 logits（512MB）被 loss 链读 4~6 次 → transient 桶 6×512MB（3.1GB）。
  已缓解：LM head 计算精度固定 `stable`（F32）→ loss 链零 cast（峰值 3069→2831 MiB）；
  隐藏层激活每次 cast 仍在（2.4× 膨胀）→ batch64 依旧 OOM。

## 05 §12.6 第一期计划与"下一步"（原位置 05 §12.6 尾部）
- 类型：演进记录（计划项，均已落地）
- 内容：原「下一步（in-kernel f16 第一期）」清单：`scan_exprs` 增加 f16 profile 的 dry-run pass
  （发现与运行时同源的变体）→ bin v9 增变体段 → `GlslEmitter` 按签名对每个输入/输出选 `float16_t`
  或 `float`（`GL_EXT_shader_16bit_storage` + 显式转换）→ 设备启用 `storageBuffer16BitAccess`
  （不支持则回退边界 cast）→ 适配层按签名选 pipeline。原「结论 2」：逐元素/归约占 84%（32/38），
  第一期只做逐元素 + 归约，matmul 段与 fold 留第二期。（该 38 变体空间实测表与"签名必须逐输入"
  结论作为当前事实保留在正文。）

## 05 §12.7 第一期落地的实测与"剩余缺口"（原位置 05 §12.7）
- 类型：性能 A/B / 演进记录
- 内容（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq 256、batch 32、1 epoch）：
  | 配置 | 峰值 | avg_loss | 耗时 |
  |---|---|---|---|
  | f32 基线 | 1595 MiB | 6.6969 | 4.7s |
  | f16 in-kernel（纯逐元素变体） | 3457 MiB | 6.7059（差 0.13%） | 5.9s |
  | f16 边界 cast（NN_VULKAN_NO_16BIT_STORAGE=1） | 4124 MiB | 6.7238 | 6.3s |
  收益：相对边界 cast −16%、耗时 −6%，但仍高于 f32 基线 2.2×；`f16_precision_test` GPT 轨迹对拍
  max_rel 3.2%。当时 gen_fused：54 变体 → 注册 31（其余 23 = 10 matmul 段 + 13 含归约）。
  剩余缺口（下一期）：① matmul 段（Linear 族，10 变体）；② 含归约指令的逐元素（13 变体，
  LayerNorm/RMSNorm 统计量链）；③ fold（注意力）——后由 §12.10 三类变体与 Phase D2/D3 补齐。

## 05 §12.8 探针逐阶段归因数字（原位置 05 §12.8）
- 类型：性能 A/B
- 内容：f32 vs f16(in-kernel) 同配置逐阶段：步末 released 272.5 → 288.5 MB（基本持平 → 多出来全是
  transient）；backward 峰值 1397 → 2869 MB，10–100MB 中块 16 → 52 项、<10MB 小块 234 → 587 项；
  同窗交错 3 样本 f32 1588/1588/1588（稳定）vs f16 4514/4582/3448（双峰，贵 2.2~2.9×），
  f16 transient 分配次数 119 vs 50 块 → 池回收/复用时序成为峰值决定因素。
  `supports_native_data_move()`（clone/slice_rows/insert_rows/zero 对 f16 放行）探针复核逐项相同
  → GPT 路径峰值中性（数据量小），保留但非主攻方向。

## 05 §12.9 池底材粒度 A/B（被复测否定的单点）（原位置 05 §12.9）
- 类型：性能 A/B（方法论反面教材）
- 内容：原记录「4MB / 阶梯 ≤16MB → 2705 MiB、耗时 11.3s」不可复现——同窗交错 3 轮复测三档配置
  （默认 12MB / `NN_POOL_LADDER_MAX_MB=16` / `NN_POOL_BLOCK_MB=4`）峰值 3513/3509/3141~3509 MiB、
  耗时全部 ≈5.7s，f32 同码对照也从 1588 漂到 1753（~10%）。结论：跨会话单点 + 不同时段对照组
  会被系统漂移骗；机制之争必须用形状/尺寸归因表裁决，不要用总量推理。
  仍有效副产物：`NN_PREC_TRACE=1` 的 `[prec][miss]` 打印（38 个非零签名只 6 个未命中 = eval_expr
  路径 84% 已原生）——该工具结论保留在正文。

## 05 §12.10 的轮次实测与旧"下一步"（原位置 05 §12.10 ①②④⑤）
- 类型：性能 A/B / 演进记录
- 内容：
  - ① 池粒度复测表（同窗交错 3 轮，默认 12MB 3513/3513/3513 @6.03s；ladder16 3509/3509/3141 @5.74s；
    block4 3509/3249/3457 @5.69s）→ 均值仅 −3.6%，"−21% + 2× 耗时"不成立；池账本 1 万次 allocate
    只扫 ~5 万 block / ~2 万 region、`vkAllocateMemory` 累计 ~150ms → 池记账不是瓶颈。
  - ② 形状级归因表原始输出（batch32 f16，1 step）：
    `->f32 副本 (32768,256) x36 1152.0MB ← W/grad_A`；`(64,8192) x135 270.0MB`；
    `(256,8192) x20 160.0MB`；`->按P落回 (32768,256) x8 128.0MB`；`(2048,256) x56 112.0MB`。
    `(32768,256) = (batch·H·seq, seq)` 即 AttentionBase::backward 物化的 W 与 grad_A（各 32MB f32），
    被 `batched_matmul`/`compute_reduce`/`compute_into` 反复抬 f32，每个副本活到帧末。
    教学点：机制之争必须用形状/尺寸归因表裁决。
  - ④ 实测（同窗 3 轮，batch32/steps2/no-kv）：f32 1753（3/3 一致）、f16 2025/2025/2273
    （1.15~1.30×；此前 3513 = 2.2×，cast 路径 −35%~−41%）；耗时 f32 5.33s / f16 5.40s（无回退）；
    backward transient live 2878 → 1767MB、32MB 级块 44 → 16、pending 2914 → 1674MB；
    batch64 f16 不再 OOM（4475 vs f32 3411）；gen_fused 由「74 结构 + 41 变体（14 skip）」
    → 「74 + 54 变体（0 skip）」（纯逐元素 31 + matmul 段 10 + 含归约 13）。
  - ⑤ 当时的 `[prec][miss]` 7 条清单（fold 1 / matmul 段 run-only sig 3 / 含归约 `[f32,f16]→f32` 2 /
    逐元素 2）与「下一步」：让 gen_fused 发射补集变体、op-level f16 GEMM（3 个手写 GEMM 宏参数化
    编第二份 SPIR-V）、fold 带类型变体——三者分别由 `tools/prec_backfill.txt`（Phase D3）、
    `-DNN_SHADER_F16`（§12.11）与 Phase D2 落地。

## 05 §12.11 的零回归字节数与失败叙事（原位置 05 §12.11 ①②）
- 类型：演进记录
- 内容：
  - 零回归证明（当时）：`matmul_tiled_spv.hpp` / `batched_matmul_spv.hpp` 字节数改动前后完全一致
    （68670 / 74408）→ 宏展开后的 f32 GLSL 编出同一份 SPIR-V。CMake 接线当时只有两行：
    `nn_embed_shader(batched_matmul_f16 … -DNN_SHADER_F16=1)`、`nn_embed_shader(matmul_tiled_f16 … -DNN_SHADER_F16=1)`
    （后续已扩展到 reduce/transpose/gather/rearrange_3d/scan_*/outer_col/group_reduce/scatter_add/
    im2col/col2im/elementwise_v2/matmul_gemv）。
  - 失败叙事：适配层第一版漏了「把 `P != F32` 直接下传内层」——归因表逐项不变、峰值纹丝不动
    （内层永远看不到 f16 → 原生 GEMM 永远命中不了）；`CpuEngine::matmul_with_bias` 曾 `(void)P`，
    导致 scan 的 f16 dry-run 只登记全 f32 签名 → matmul 段变体永远发现不到（补后变体数 54 → 56）。
  - 当时的剩余项（现已处理/部分过期）：仍走 cast 的 op-level 原语约 550MB
    （`(2048,256)×80`、`(64,8192)×72`、`(256,8192)×10`、`(32768,16)×24`、`(8208,64)×10`）；
    7 条运行时签名；fold 带类型变体未做。
- 备注：`gen_fused` 变体数 54→56、以及演进全景数字（f16 边界 cast 4124 → in-kernel 首期 3513 →
  +matmul/归约/目标传递 2025~2286 → +op-level f16 GEMM 1570，f32 基线 ~1588~1754）
  属分阶段 A/B 流水，最终数字以正文 §12.11 ③ 为准。
