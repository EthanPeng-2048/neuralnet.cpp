# 📋 优化方案全览（Optimization Proposal List）

> **文档版本**：2026-08-26
> **适用范围**：neuralnet.cpp 框架全部优化方向——GPU 内核、IR/编译器、CPU 计算、显存管理、架构扩展
> **关联文档**：[02-performance.md](./02-performance.md) · [09-operator-fusion.md](./09-operator-fusion.md) · [10-memory-optimization.md](./10-memory-optimization.md) · [11-ir-optimization.md](./11-ir-optimization.md) · [12-innovative-designs.md](./12-innovative-designs.md)

---

## 0. 状态图例

| 标记 | 含义 |
|------|------|
| ✅ | 已完成 |
| 🚧 | 进行中 / 部分完成 |
| 📋 | 本文档规划，待立项 |
| ❌ | 经评估后搁置（附原因） |
| 🔮 | 远期探索，当前无明确路径 |

---

## 1. GPU 内核与代码生成

### P1-01 ✅ vec4 向量化发射（已实现）
- **文件**：`glsl_gen.hpp`、`gen_fused.cpp`、`vk_backend.hpp`
- **现状**：所有视图 ∈ `{Linear, RowBroadcast, ColBroadcast}` 的表达式自动走 vec4 路径（每线程 4 元素），不满足时回退标量。
- **收益**：计算:加载比从 2:1 提升到 8:1，bank conflict 消除。
- **限制**：`RotateHalf`、`RowMod` 视图排除（索引映射逐通道变化，vec4 同行保证不成立）。

### P1-02 ✅ 按需 row/col 发射（已实现）
- **文件**：`glsl_gen.hpp` `glsl_view_uses_row()`
- **现状**：`Linear` 和 `ColBroadcast` 不发射 `row = i / cols`；其他视图按需发射。
- **收益**：消除 GPU 昂贵的整数除法/模运算（对只用 `col` 的 kernel 消除一次 `i/cols` 和 `i%cols`）。

### P1-03 ✅ GPU matmul tiled + vec4（已实现）
- **文件**：`shaders/matmul_tiled.comp`
- **现状**：16×16 workgroup、64×64 输出 tile、BK=32、vec4 共享内存布局、每线程 4×4=16 元素。
- **收益**：compute:load 比 8:1、bank conflict 消除、转置感知。

### P1-04 ✅ 逐元素 kernel 的 row/col 除法消除（已实现）
- **文件**：`glsl_gen.hpp`
- **现状**：标量路径（纯标量 kernel + vec4 回退循环）均已按 `need_row` / `need_col` 守卫发射，仅计算实际使用的 `row`/`col`。与 vec4 快速路径共用同一套 `glsl_view_uses_row` / `glsl_view_uses_col` 判定。
- **收益**：所有非 vec4 融合 kernel 消除冗余整数除法/模运算（~10-15% 内核时间）。

### P1-05 ✅ 归约指令 pass 多累加器（已实现，2026-08-26）
- **文件**：`glsl_gen.hpp`（`generate_glsl_reduce` 行/列归约指令 pass）
- **现状**：归约指令 pass（每条归约指令）已改为 4 路独立标量累加器（acc0-acc3，独立变量非数组索引）+ 4 路展开循环，与归约视图 pass 一致。打破 `acc = acc + x` 串行 FMA 依赖链，提升 ILP。
- **验证**：27/27 ctest 全绿（含 gradcheck/融合测试）。原 `reduce_instr_bench`（GPU 驻留内核级基准）已随 bench 工具移除（2026 清理），吞吐以 fused_gpu_test 覆盖。
- **A/B 实测（GTX 850M）**：`col_reduce_sum(a*b)`（LayerNorm 方差路径，**内存带宽受限**）新/旧基本持平（±5% 噪声内）。这是**预期的**——带宽受限 kernel 的累加器依赖链已被掩盖，多累加器无增益。
- **理论收益**：对**计算密集**归约（每元素 exp/sqrt/多步运算后才累加，计算成瓶颈时）可 ~1.5-2×。当前生产 DSL 归约均为内存受限简单归约，该收益为"潜伏"状态；随更多计算密集融合走 DSL 路径（P2-09/P2-14）而兑现。
- **结论**：保留（零回归 + 与视图 pass 一致性 + 计算密集归约潜在收益）。**评估教训**：A/B 需针对目标瓶颈类型选 kernel，测内存受限 kernel 得出中性不代表实现无效。

### P1-06 📋 归约 shader 公共子表达式消除
- **现状**：每条归约指令重跑 `emit_instrs(0, ri)`，逐元素链在多个归约间重复计算。复杂度 O(归约数 × 元素指令数)。
- **方案**：在 `generate_glsl_reduce` 中缓存已生成的逐元素指令块（哈希 `(op, a, b, c)` → 生成代码），跨归约复用。
- **工作量**：中（2-3 天）
- **收益**：减少生成 shader 体积，提升指令缓存命中率；对含多归约的 kernel（如 LayerNorm = mean + variance 两次归约）效果显著。
- **风险**：中（需确保 regalloc 对共享指令块的寄存器分配正确）

### P1-07 📋 行归约 tile 化 / 转置
- **现状**：行归约已是合并访问（~21-83 GB/s），但对极高列宽（cols > 8192）时单工作组处理一整行可能溢出寄存器。
- **方案**：对行归约也引入分块策略：每工作组处理行的一个 tile（如 256 列），多工作组协作完成一行归约。
- **工作量**：中（3-5 天）
- **收益**：大列宽场景 ~1.5× 提升；常规场景无损
- **风险**：中（需改 dispatch 和 Reduce 广播逻辑）

### P1-08 📋 可变 workgroup 大小
- **现状**：所有 compute shader 硬编码 `local_size_x = 256`。
- **方案**：按 kernel 类型选择 workgroup 大小：
  - 逐元素：256 或 512（取决于 ALU 密度）
  - 归约：256（匹配 warp/subgroup 大小）
  - matmul tiled：256（16×16）
- **工作量**：小（0.5-1 天）
- **收益**：对 ALU 轻量 kernel（如 broadcast）可提升 occupancy
- **风险**：低（但需逐 kernel 测评，收益可能有限）

### P1-09 ✅ warp shuffle（subgroup）归约（已实现）
- **文件**：`glsl_gen.hpp`（`emit_tree_reduce`）
- **现状**：归约已改用 subgroup 蝴蝶归约替代共享内存树归约——第 1 步 `subgroupAdd`/`subgroupMax` 得到每 warp 部分和（零共享/屏障），第 2 步首 warp 归约全部 warp 部分和（`gl_SubgroupID == 0`，屏障后全线程可见）。
- **收益**：消除共享内存树归约的屏障与共享内存访问开销。
- **注意**：启用 `GL_KHR_shader_subgroup_basic` / `GL_KHR_shader_subgroup_arithmetic` 扩展，需 Vulkan 1.3+ 或 `VK_EXT_subgroup_size_control` 支持。

### P1-10 📋 matmul 分块参数调优（对话 O1）
- **现状**：`BM=BN=64` 硬编码。历史上 tiled dispatch 曾错用 16（峰值掉到 4%），参数极敏感。
- **方案**：基准测 32/64/128，按 shape 范围选最优 tile；大 K 用 K 维分块循环。
- **工作量**：小-中（1-2 天）
- **收益**：不同矩阵形状自适应最优分块
- **风险**：低（纯调参 + 基准）

### P1-11 📋 共享内存两级归约（对话 O3）
- **现状**：reduce 每 workgroup 256 线程单级归约。
- **方案**：局部（thread→shared）→ 全局（workgroup→block）两级归约，减少 barrier 次数、提升大 N/K 归约吞吐。
- **工作量**：中（3-5 天）
- **收益**：大列宽/大行归约 ~1.5×
- **风险**：中

### P1-12 📋 scatter_add 原子优化（对话 O4）
- **现状**：CAS 循环 float atomicAdd，慢且非确定（已在 docs/08 文档化为例外）。
- **方案**：若设备支持 `VK_EXT_shader_atomic_float` 用原生 float atomic；否则冲突分桶减少 CAS 重试。
- **工作量**：小（1 天）
- **收益**：embedding 梯度累积提速
- **风险**：低（原语义不变，仍保持非确定例外）

### P1-13 📋 matmul / batched_matmul 代码去重（对话审查 P3）
- **现状**：`matmul_gpu` 与 `batched_matmul_gpu` 各有 ~200 行重复的 descriptor/cmd/barrier/dispatch 逻辑。
- **方案**：提取 `dispatch_matmul_generic` 公共 helper（仿 `dispatch_bmm_generic`）。
- **工作量**：小-中（1-2 天）
- **收益**：减少重复代码与维护风险
- **风险**：低

---

## 2. IR 与编译器优化

### P2-01 ✅ IR-A：DCE + 常量折叠 + 代数化简 + 稳定重编号
- **文件**：`expr_opt.hpp`
- **现状**：完整 pass 链，确定性遍历顺序，保守折叠（不折叠超越函数避免 ULP 差异）。

### P2-02 ✅ IR-B：CSE + liveness 寄存器分配
- **文件**：`expr_opt.hpp`
- **现状**：哈希 CSE（Fanout→Reg 归一化）+ 两区段分离（逐元素/归约）贪心分配。

### P2-03 ✅ IR-C：图 IR + 链融合分析
- **文件**：`expr_graph.hpp`
- **现状**：贪心节点序拼接——B 以 Linear 视图消费 A 输出、A 无其他消费者、同形状、均无归约 → 指令内联。拼接后 canonicalize + validate，超限保守放弃。
- **限制**：仅支持逐元素→逐元素链，不支持跨 matmul、不支持多消费者 DAG。

### P2-04 ✅ IR-D：后端 emitter 抽象
- **文件**：`expr_emitter.hpp`、`cpu_emitter.hpp`
- **现状**：`ExprEmitter` 纯接口 + emitter registry，`GlslEmitter` + `CpuEmitter` 双实现。

### P2-05 📋 跨 matmul 融合（matmul + 偏置 + 激活）
- **现状**：matmul 作硬融合边界，`linear(x) + bias + relu` 需 3 次 kernel dispatch + 2 个中间 Tensor。
- **方案**：
  - 在 `ExprSpec` 中引入 `MatmulInstr`（或 `FusedLinear` 原语），表达 `matmul(A, B) + bias + activation`。
  - 或扩展融合分析：允许 `elementwise → reduce → elementwise` 跨归约融合（将 matmul 结果直接内联进后续逐元素链）。
  - GPU 端：生成包含 matmul + bias + activation 的单 kernel（利用共享内存分块 + 尾部激活融合）。
- **工作量**：大（1-2 周）
- **收益**：GPT forward 每层减少 2+ kernel dispatch + 消除 Linear 输出中间 Tensor（~seq×d_ff × 4B）
- **风险**：高（需修改 ExprSpec 语义或融合分析核心逻辑）

### P2-06 📋 多消费者 DAG 融合
- **现状**：当前融合要求 `A` 只有一个消费者（`A.tail` 必须只有一个下游）。多消费者场景（如注意力中 Q/K/V 共享投影输出）各自成独立 kernel。
- **方案**：
  - 扩展图 IR：允许 A 有多个消费者，只要各消费者消费的是 A 的不同视图（如 A[Linear] 和 A[RotateHalf]）。
  - 或引入中间结果复用：A 的输出保存在寄存器（而非落 VRAM），后续多个 kernel 从寄存器直接读取。
- **工作量**：大（1-2 周）
- **收益**：注意力层 forward/backward 减少重复 matmul + 中间 Tensor
- **风险**：高（寄存器压力增加、融合分析复杂度大幅上升）

### P2-07 📋 ExprSpec 上限动态化
- **现状**：硬编码上限（64 指令 / 16 寄存器 / 8 输入 / 16 常量）。复杂融合（如完整 Attention + RoPE + LayerNorm）可能触及上限被迫放弃融合。
- **方案**：
  - **方案 A**：分级策略——简单融合保持当前上限，复杂融合退回 eager（`run_expr_eager`）。
  - **方案 B**：上限扩展到 128 指令 / 32 寄存器，`gen_fused` 在构建期生成更大 shader。
  - **方案 C**：自动拆分——超过上限时自动拆为多 kernel，kernel 间通过共享内存传递中间结果。
- **工作量**：方案 A 小（1 天）；方案 B 中（2-3 天）；方案 C 大（1 周）
- **收益**：更多层可享受融合收益
- **风险**：方案 A 低；方案 B 低但 shader 体积增长；方案 C 中（需修改 dispatch 逻辑）

### P2-08 📋 超越函数常量折叠开放
- **现状**：`fold_constants_and_algebra` 保守不折叠 `exp/log/sqrt/rsqrt/tanh`（避免 CPU/GPU ULP 差异导致 key 不匹配）。
- **方案**：在构建期（`scan_exprs` / `gen_fused`）允许折叠（此时在同一平台编译），运行时不折叠。双轨策略。
- **工作量**：中（2-3 天，需修改 canonicalize 语义）
- **收益**：减少 shader 中的冗余超越函数调用（如 `exp(0) → 1.0`）
- **风险**：中（key 两端一致性约束更复杂）

### P2-09 📋 归约→广播→逐元素链融合（对话 G1，最高价值）
- **现状**：归约是硬边界，LayerNorm/Softmax 的 mean/var/rms_inv 归约向量必须落显存。
- **方案**：ExprOp 已有 ColSum/ColMax/RowSum/RowMax + Reduce 操作数（单 ExprSpec 内已支持归约广播）。关键缺口是**跨节点复用归约向量**——节点 A 的归约结果直接作为节点 B 的 `Reduce` 输入（kernel 内部共享内存保留），而非落显存再 broadcast 读回。
- **工作量**：中-大（1 周）
- **收益**：LayerNorm 等"归约+广播+逐元素"模式核心收益
- **风险**：中（需扩展归约向量生命周期追踪）

### P2-10 📋 跨 kernel 自动融合（对话 G2/L3，消除手工 begin_expr/end_expr）
- **现状**：只有演示层 `FusedChainLayer` 手工标记，真实 Layer 全部独立 dispatch——IR 融合收益几乎为零。
- **方案**：`eval_expr` 内部自动维护"融合窗口"——相邻可融合表达式自动并入；遇不兼容（matmul/归约边界）自动关窗。Layer 无感知。
- **工作量**：中-大（1 周）
- **收益**：让 IR 融合真正覆盖实际 Layer，是收敛专用融合算子的前提
- **风险**：中-高（正确性敏感）

### P2-11 📋 融合成本/收益启发式（对话 G4）
- **现状**：纯按节点序贪心，超限即放弃。
- **方案**：建模每步融合收益（省 launch/bytes）与成本（寄存器/指令压力），贪心选最优。小中间量（归约向量）收益高，大张量多次消费的复制不划算。
- **工作量**：中（3-5 天）
- **收益**：融合决策更优
- **风险**：低（需保证确定性）

### P2-12 📋 图级缓存跨 step 复用（对话 G5）
- **现状**：训练每 step 表达式结构高度重复，却每 step 重跑融合分析 + canonicalize + key 计算。
- **方案**：整图 key（FNV 哈希）→ 缓存"节点序列→kernel 计划"，命中直接复用。
- **工作量**：中（3-5 天）
- **收益**：训练热路径消除重复 CPU 融合分析开销
- **风险**：中（key 须覆盖全部语义含视图 param）

### P2-13 📋 视图组合化简（对话 G6）
- **现状**：RotateHalf/RowMod 是运行时 param（不进 key），组合视图会重复物化。
- **方案**：相邻视图链（如 RotateHalf→Linear）化简/合并索引映射。
- **工作量**：小（1-2 天）
- **收益**：进一步消中间物化
- **风险**：低

### P2-14 📋 冗余 broadcast/逐元素原语收敛 DSL（对话 O5）
- **现状**：独立 broadcast_row/col、elementwise_* 原语在 IR 成熟后是冗余（DSL 内联已覆盖）。
- **方案**：Layer 逐步改用 `dsl::compute` 融合表达，独立原语仅作 fallback。
- **前置依赖**：P2-10（自动融合）先落地。
- **工作量**：中（随 Layer 逐步迁移）
- **收益**：减少算子数量（呼应"融合收敛到 IR"）
- **风险**：中

### P2-15 📋 融合分析并行化（对话 C1，低风险）
- **现状**：`fuse_expr_graph` 是纯函数（只读图、无共享状态）。
- **方案**：end_expr 时多线程并行做各 kernel 的 canonicalize / key 计算 / dispatch 参数打包。
- **工作量**：小-中（2-3 天）
- **收益**：减少融合分析 CPU 开销
- **风险**：低（每 kernel 独立计算，无顺序依赖）

### P2-16 📋 双缓冲 command buffer 流水化（对话 C2/O6）
- **现状**：batch 模式录制所有 kernel 到单 command buffer 一次性提交，CPU 录制天然快于 GPU 执行（已有隐式重叠）。
- **方案**：双缓冲——录制下一个 batch 时提交上一个，消除 end_batch 的空闲等待，让 CPU 融合分析与 GPU 执行流水化。
- **工作量**：中（3-5 天）
- **收益**：消除提交间隙空闲
- **风险**：中（生命周期管理，可复用 pending_destroys 机制）

### P2-17 📋 多线程数据并行 IR（对话 C3，高价值大工程）
- **现状**：thread-local 图已安全（每线程独立录制图），但 backend 单队列串行化提交。
- **方案**：每线程独立 command buffer / 多 compute queue，end_expr 后各自提交。需 GpuBackend 支持多队列或并发 submit 调度。
- **工作量**：大（2-3 周）
- **收益**：多 batch 数据并行收益
- **风险**：高（队列同步、内存池并发、确定性）

### P2-18 📋 内存池并发安全（对话 C4）
- **现状**：`MemoryPool` 已有 `mutex_`，多线程高并发分配成瓶颈。
- **方案**：per-thread 分配缓存或 arena 分区，减少锁竞争。与 P2-17 配套。
- **工作量**：中（3-5 天）
- **收益**：多线程分配吞吐
- **风险**：中

### P2-19 📋 并发确定性保证（对话 C5）
- **现状**：铁律 8 要求单线程逐字节一致；决策类逻辑（融合顺序/key/ID）并发时必须显式同步。
- **方案**：并发版融合分析结果与串行版逐字节比对测试；原子累加类保持已文档化例外。
- **工作量**：小（1-2 天）
- **收益**：并发安全的前提保障
- **风险**：低

---

## 3. CPU 计算优化

### P3-01 ✅ 缓存分块 matmul（BLOCK_SIZE=64）
- **文件**：`config.hpp`
- **现状**：64×64 tile，`static_assert` 确保 b_block ≤ 64KB 栈预算。

### P3-02 ✅ SmartPolicy 自适应并行
- **现状**：`PARALLEL_THRESHOLD = 524288`，基于 32 核 Haswell 调优。

### P3-03 ✅ 线程池 latch 零分配
- **现状**：原子计数器 + work-stealing + 三阶段等待（自旋→偷任务→条件变量）。

### P3-04 ❅ 8 路累加器 dot_kernel——经评估搁置
- **曾尝试**：8 路独立累加器手写内核。
- **搁置原因**：编译器已把 forward 单累加器循环自动向量化到最佳；手写内核让 forward 回退。唯一真赢点是 backward/转置变体（~2×），但整体收益不值得增加的复杂度。
- **结论**：CPU matmul 优化暂缓，保留基线。

### P3-05 📋 BLOCK_SIZE 和 PARALLEL_THRESHOLD 自适应
- **现状**：硬编码常量，针对特定硬件调优。
- **方案**：
  - 运行时探测 L1/L2 大小（Linux：`/sys/devices/system/cpu/cpu0/cache/index1/size`；跨平台：`std::thread::hardware_concurrency()` + 启发式）。
  - `BLOCK_SIZE = sqrt(L1_size / sizeof(Scalar) / 2)` 保证 A/B 块均入 L1。
  - `PARALLEL_THRESHOLD` 基于 `hardware_concurrency() × 小数据量单核时间` 推算。
  - 可设 `NN_OVERRIDE_BLOCK_SIZE` 环境变量允许用户手动覆盖。
- **工作量**：小（1 天）
- **收益**：不同硬件自动适配最优参数
- **风险**：低

### P3-06 📋 CPU SIMD 向量化路径
- **现状**：CMake 编译选项 `-march=native` 已启用（自动使用 AVX2/FMA），但 `BLOCK_SIZE` 硬编码 64 不保证与 SIMD 宽度对齐。
- **方案**：
  - matmul 内层循环显式使用 `__m256`/`__m512` intrinsic（AVX2/AVX-512）。
  - 或确保 `-march=native` + `-O3` + `-ffast-math`（可选）让编译器自动向量化。
  - 注意：当前项目**禁用** `-ffast-math`（避免数值差异），向量化需在不启用 fast-math 的前提下进行。
- **工作量**：大（1-2 周，需逐 kernel 检查自动向量化是否生效）
- **收益**：CPU matmul ~1.5-3×（取决于自动向量化质量）
- **风险**：中（手动 intrinsic 可维护性差；`-ffast-math` 数值风险）

### P3-07 📋 CPU 融合 kernel（CpuEmitter 生成代码优化）
- **现状**：`CpuEmitter` 生成可编译 C++ 直线代码，但未启用 `-O3` 编译（运行时解释/解释执行 vs AOT 编译）。
- **方案**：
  - 构建期将 CpuEmitter 生成的代码编译为共享库（`dlopen`/`LoadLibrary`），运行时按 key 调用。
  - 或在 `eval_expr` 路径中直接内联解释，减少虚函数/分支开销。
- **工作量**：大（1-2 周，涉及运行时编译基础设施）
- **收益**：CPU 融合 kernel ~2-5×（消除逐条指令解释开销）
- **风险**：高（跨平台兼容性、构建复杂度）

### P3-08 📋 CNN im2col GPU 化
- **现状**：`Conv2D` / `MaxPool2D` 的 im2col / col2im / 池化在 CPU 端完成（`to_matrix`/`from_matrix`），即使用 GPU 训练也需要 CPU↔GPU 往返。
- **方案**：将 im2col 和 max pool 实现为 Vulkan compute shader（或融合进现有 matmul 流水线）。
- **工作量**：中（3-5 天）
- **收益**：CNN 训练 GPU 路径消除 CPU↔GPU 数据传输瓶颈
- **风险**：中（需验证 im2col 输出布局与 matmul 的兼容性）

---

## 4. 显存管理与优化

### P4-01 ✅ L1 激活重计算（梯度检查点）
- **现状**：`checkpoint_mode_` / `set_checkpoint_every` 逐层/块级检查点，backward 重算激活。stride∈{1,2} 与全存基线逐位一致。

### P4-02 ✅ L2 内存池整块归还 + 统计
- **现状**：`release_idle_blocks()` + `retain_free_bytes_` 防抖动 + `PoolStats` 诊断。

### P4-03 📋 半精度（bf16/fp16）训练
- **现状**：全项目 `Scalar = float`，所有激活/梯度/优化器状态 4B。
- **方案**：
  - **阶段 1**：前向传播 bf16（激活 + 权重），反向传播 fp32（梯度 + 优化器）→ **混合精度训练**。
  - **阶段 2**：全链路 bf16（含优化器状态）→ **纯 bf16 训练**。
  - 实现：`config.hpp` 增加 `Precision` 枚举；tensor 支持 bf16 存储 + fp32 累加；matmul 内核支持 bf16→fp32 dequant。
  - 需 GPU 端：Vulkan `VK_KHR_shader_float16_int8` 或 CUDA `__nv_bfloat16`。
- **工作量**：大（2-3 周，含数值验证）
- **收益**：
  - 显存减半（27GB → ~15GB）
  - GPU 带宽吞吐翻倍
  - CUDA matmul 利用 Tensor Core（V100/T4/A100 bf16 原生支持）
- **风险**：高（数值稳定性需全面验证；某些操作如 LayerNorm/Softmax 需 fp32 中间累加）

### P4-04 📋 L3 优化器降内存（8-bit Adam / 状态分片）
- **现状**：Adam 的 `m` / `v` 各一份参数大小（~3.7GB × 2 = 7.4GB）。
- **方案**：
  - **8-bit Adam**：m/v 用 INT8 存储 + 动态指数对齐（参考 bitsandbytes 实现）。
  - **梯度累积**：小 batch + 梯度累积模拟大 batch，减少同时驻留的激活量。
  - **ZeRO-style 分片**：将优化器状态分片到不同 GPU（需多卡支持，远期）。
- **工作量**：8-bit Adam 中（1 周）；ZeRO 大（2-3 周，需分布式框架）
- **收益**：8-bit Adam 可将优化器内存从 7.4GB 降到 ~2GB
- **风险**：8-bit Adam 中（精度损失需验证）；ZeRO 高

### P4-05 📋 激活 offload 增强
- **现状**：`Layer::set_activation_offload()` / `offload_ram_bytes()` 已有接口，但仅在部分层实现。
- **方案**：
  - 将所有大激活（如注意力 Q/K/V、FFN 中间结果）支持异步 offload 到 NVMe/系统内存。
  - 利用 CUDA stream / Vulkan timeline semaphore 实现计算与传输重叠。
- **工作量**：大（1-2 周）
- **收益**：训练更大 batch/seq_len 而不增加 GPU 显存
- **风险**：高（需精确调度避免计算等待 I/O）

### P4-06 📋 内存池碎片化度量与压缩
- **现状**：`PoolStats::fragmentation` 已统计碎片率，但无主动压缩。
- **方案**：
  - 定期检查碎片率，超过阈值时执行碎片整理（分配连续新块 → 拷贝 → 释放旧块）。
  - 或改用 buddy allocator 替代 first-fit，从根源减少碎片。
- **工作量**：碎片压缩中（3-5 天）；buddy allocator 大（1-2 周）
- **收益**：减少长训练的显存浪费
- **风险**：碎片压缩中；buddy allocator 高（需完全重写内存池）

### P4-07 📋 内存池预分配策略
- **现状**：`MemoryPool` 按需申请 128MB block，首次分配可能触发较大延迟。
- **方案**：训练启动时根据模型大小预分配足够 block（如 `预分配 = 模型参数 × 10` 作为 warmup）。
- **工作量**：小（0.5 天）
- **收益**：减少训练初期的分配延迟
- **风险**：低

---

## 5. CUDA 后端

### P5-01 ✅ CUDA 后端基础架构（已实现但 v1.0.0 停用）
- **文件**：`backend/cuda_backend.hpp`、`cuda/cuda_kernels.cu`
- **现状**：完整实现（CudaBuffer/CudaTensor/CudaBackend），支持所有基础 op，但因 M4/M5/M6 融合原语未实现而停用。
- **TODO**：`CMakeLists.txt:376` — 恢复 CUDA 后端。

### P5-02 📋 CUDA 融合原语补齐
- **现状**：Vulkan 端已有 `batched_matmul_reduce/denom/apply`、`col_softmax_denom/sparse_forward`、`bmm_q/kv_backward` 等手写融合 kernel。CUDA 端缺失这些。
- **方案**：
  - **方案 A**：为 CUDA 端手写等价 cuBLAS/cuSPARSE 融合 kernel。
  - **方案 B**：通过 IR-D emitter 为 CUDA 后端生成融合 kernel（利用 `CpuEmitter` 作为参考）。
  - **方案 C**：利用 Triton / CUTLASS 模板库替代手写 CUDA kernel。
- **工作量**：方案 A 大（2-3 周）；方案 B 大（2-3 周）；方案 C 中（1 周，但引入依赖）
- **收益**：NVIDIA GPU 训练速度大幅提升（Tensor Core、高带宽 HBM）
- **风险**：方案 A/B 中；方案 C 中（CUTLASS/Triton 版本兼容性）

### P5-03 📋 CUDA cuBLAS 集成
- **现状**：`NN_HAS_CUBLAS` 开关已有，但 `matmul_gpu` 仅用 naive kernel。
- **方案**：将 `matmul_gpu` 替换为 `cublasGemmEx`（支持 bf16/fp16/fp32），利用 Tensor Core。
- **工作量**：中（3-5 天）
- **收益**：GEMM 吞吐提升 3-10×（取决于矩阵大小和 GPU 型号）
- **风险**：低（cuBLAS 是稳定 API）

---

## 6. 序列化与工程

### P6-01 ✅ 序列化 EOF 语义修复（已实现）
- **文件**：`model_serialization.hpp`
- **现状**：`load_model` 已用 `ifs.peek() == EOF` 在读取前检查文件结尾，读到 EOF 时保持默认值（向后兼容旧格式），非 EOF 错误才报错。原 `TODO(1.1, M2)` 已移除。
- **收益**：旧模型格式兼容性

### P6-02 📋 Vulkan singleton 内存泄漏修复
- **文件**：`backend/vk_backend.hpp:1017`
- **现状**：`TODO(1.1, L1)` — `GpuBackend::instance()` 故意泄漏以避免静态析构顺序问题。
- **方案**：增加 `GpuBackend::shutdown()` 显式析构 + 引用计数（`shared_ptr` + `atexit` 注册）。
- **工作量**：小（1 天）
- **收益**：Valgrind / ASan 干净退出
- **风险**：低（但需验证多线程安全）

### P6-03 📋 优化器构造器重构
- **文件**：`compute_optimizer.hpp`
- **现状**：原 `TODO(1.1, S2)` 已移除；配置校验失败时改用 `std::abort()`（非原 `NN_ASSERT`），但 Release 模式下仍会中止进程而非通过 `Result<T>` 返回可恢复错误。
- **方案**：构造器改为返回 `Result<T>`（如 `Adam::create(...)`），签名级重构。
- **工作量**：中（2-3 天，涉及所有优化器调用点）
- **收益**：Release 模式下也能正确报告配置错误而非中止进程
- **风险**：中（大量调用点需同步修改）

### P6-04 📋 测试覆盖率提升
- **现状**：17 个测试目标（`src/*test*.cpp`）覆盖核心功能，但缺少：
  - 极端形状测试（rows=1, cols=1, rows=100000）
  - 数值边界测试（全零、全最大值、NaN/Inf 传播）
  - 并发安全测试（多线程同时 forward/backward）
  - 长序列压力测试（seq_len=8192+）
- **方案**：增加 fuzzing 测试 + property-based 测试。
- **工作量**：中（持续进行）
- **收益**：发现边界 bug、提升长期可维护性
- **风险**：低

### P6-05 📋 CMake 构建系统优化
- **现状**：每次 `cmake --build build` 重新扫描所有头文件（header-only 架构）。
- **方案**：
  - 引入预编译头（PCH）加速编译。
  - 或将核心头文件拆分为编译单元（`.cpp`），减少增量编译时间。
  - 或用 `ccache` / `sccache` 缓存编译结果。
- **工作量**：PCH 小（0.5 天）；拆分单元大（1-2 周）
- **收益**：开发迭代速度提升
- **风险**：PCH 低；拆分单元高（破坏 header-only 设计哲学）

---

## 7. 算法与模型扩展

### P7-01 📋 注意力形态升级（Flash-Attention）
- **现状**：M6 两趟式注意力消除了 `seq²` 物化，但仍用多 kernel dispatch（`batched_matmul_reduce` → `denom` → `apply`）。
- **方案**：参考 FlashAttention 算法——将 matmul + softmax + matmul 融合为单 kernel，利用 SRAM tiling 消除所有中间矩阵。
- **工作量**：大（2-3 周，需重写注意力 GPU kernel）
- **收益**：
  - 显存 O(seq) vs 当前 O(seq)（M6 已消除 seq²，但多 kernel 仍有开销）
  - 速度 ~1.5-2×（减少 kernel dispatch + 重访 Q/K/V 的 HBM 读写）
- **风险**：高（正确性验证复杂、backward 更难实现）

### P7-02 📋 KV Cache 优化（推理）
- **现状**：`forward_step` 已支持增量推理（逐 token 写 cache），但 cache 复制/管理可能有冗余。
- **方案**：
  - **PagedAttention**：类似 vLLM 的虚拟内存分页，按需分配 KV cache block。
  - **GQA（Grouped Query Attention）**：多个 query head 共享 K/V head，减少 cache 大小。
  - **MQA（Multi-Query Attention）**：所有 query head 共享单个 K/V head。
- **工作量**：PagedAttention 大（2-3 周）；GQA/MQA 中（1 周）
- **收益**：推理 batch size 扩大 2-8×（GQA/MQA）或支持长序列推理（PagedAttention）
- **风险**：高

### P7-03 📋 分布式训练（数据并行 / 模型并行）
- **现状**：单卡训练，无跨设备通信。
- **方案**：
  - **阶段 1**：数据并行（DDP）——多卡各持完整模型副本，AllReduce 梯度。
  - **阶段 2**：ZeRO-style 分片——参数 / 梯度 / 优化器状态跨卡分片。
  - **阶段 3**：张量并行 / 流水线并行（超大模型）。
- **工作量**：DDP 大（2-3 周）；ZeRO 大（3-4 周）；张量/流水线并行 极大（1-2 月）
- **收益**：训练速度线性扩展（数据并行）、支持更大模型（模型并行）
- **风险**：高（通信基础设施、数值一致性、调试复杂度）

### P7-04 📋 自动算子调优（Auto-tuning）
- **现状**：matmul tile 大小、workgroup 大小、归约策略均为硬编码常量。
- **方案**：参考 TVM / Ansor 的 auto-schedule——对每个 kernel 形状 profile 多种参数组合，选择最快配置。
- **工作量**：大（2-3 周，需构建 benchmark 框架 + 参数搜索空间）
- **收益**：不同硬件 / 不同矩阵形状自动适配最优配置
- **风险**：中（搜索开销、过度拟合特定硬件）

### P7-05 📋 混合专家（MoE）支持
- **现状**：所有 Transformer 层 dense（每个 token 经过所有 FFN）。
- **方案**：引入 TopK 路由 + 专家 FFN + 负载均衡损失。
- **工作量**：大（2-3 周）
- **收益**：参数量扩展而不增加计算量
- **风险**：高（路由训练不稳定、AllToAll 通信）

### P7-06 📋 多模态支持（视觉+语言）
- **现状**：已有 `PatchEmbedding`（ViT）和 `Conv2D`/`MaxPool2D`（CNN），可搭建视觉编码器。
- **方案**：引入 Cross-Attention 层 + 多模态投影层，支持图文输入。
- **工作量**：大（2-3 周）
- **收益**：扩展模型能力
- **风险**：中

### P7-07 📋 量化训练 / 量化推理（INT8/INT4）
- **现状**：全 fp32 推理。
- **方案**：
  - **后训练量化（PTQ）**：训练完成后权重量化为 INT8/INT4，推理时 dequant。
  - **量化感知训练（QAT）**：训练中模拟量化误差，学习对量化鲁棒的权重。
- **工作量**：PTQ 中（1 周）；QAT 大（2-3 周）
- **收益**：推理显存减少 2-4×，推理速度提升 2-4×（INT8 matmul）
- **风险**：中（精度损失、需 calibration 数据集）

---

## 8. 工具链与开发者体验

### P8-01 📋 基准测试框架自动化
- **现状**：bench 工具（`compute_bench` / `mnist_bench` / `bench_thresholds` / `reduce_bench` / `reduce_instr_bench`）已移除（2026 清理，无法在 -Werror 下编译）。若需性能基准，重建统一框架。
- **方案**：
  - 统一基准框架（类似 Google Benchmark），支持 CI 集成 + 回归检测。
  - 自动生成性能报告（HTML/Markdown）。
  - 支持 A/B 对比（`git diff` 前后性能变化）。
- **工作量**：中（1 周）
- **收益**：性能回归及时发现
- **风险**：低

### P8-02 📋 训练可视化增强
- **现状**：`gui.py` 已有实时训练监控（loss/accuracy 曲线），但缺少：
  - 显存使用曲线
  - GPU 利用率 / 带宽监控
  - 学习率调度可视化
  - 多 run 对比
- **方案**：集成 TensorBoard / WandB，或扩展 GUI。
- **工作量**：中（3-5 天）
- **收益**：训练调试效率提升
- **风险**：低

### P8-03 📋 文档自动化
- **现状**：12 份设计文档手动维护。
- **方案**：从代码注释 / `DEVELOPMENT_STANDARDS.md` 自动生成 API 文档（Doxygen / mdBook）。
- **工作量**：中（3-5 天）
- **收益**：文档与代码同步
- **风险**：低

### P8-04 📋 模型 zoo / 预训练权重
- **现状**：`pretrained/` 目录存在，但无系统化的预训练模型管理。
- **方案**：
  - 提供标准模型（GPT-2 small/medium、LeNet-5）的预训练权重下载。
  - 支持 HuggingFace 格式导入/导出。
- **工作量**：中（1 周）
- **收益**：降低用户上手门槛
- **风险**：低

---

## 9. 优先级汇总

### 🔥 高影响 + 可行（建议优先实施）

| 编号 | 方案 | 工作量 | 收益 |
|------|------|--------|------|
| P1-05 | 归约多累加器 | 2-3 天 | 计算密集归约 ~2× |
| P2-07 | ExprSpec 上限分级策略 | 1 天 | 更多层享受融合 |
| P3-05 | BLOCK_SIZE 自适应 | 1 天 | 多硬件适配 |
| P5-03 | CUDA cuBLAS GEMM | 3-5 天 | NVIDIA GPU GEMM 3-10× |
| P4-07 | 内存池预分配 | 0.5 天 | 减少初期延迟 |
| P6-02 | Vulkan 内存泄漏修复 | 1 天 | ASan 干净退出 |

### ⚡ 中影响 + 架构级（需要更大投入）

| 编号 | 方案 | 工作量 | 收益 |
|------|------|--------|------|
| P2-05 | 跨 matmul 融合 | 1-2 周 | 减少 kernel dispatch + 中间 Tensor |
| P4-03 | 半精度训练 | 2-3 周 | 显存减半 + 带宽翻倍 |
| P5-02 | CUDA 融合原语补齐 | 2-3 周 | 恢复 CUDA 后端 |
| P1-06 | 归约 CSE | 2-3 天 | shader 体积 + 指令缓存 |
| P3-08 | CNN im2col GPU 化 | 3-5 天 | CNN 训练全 GPU |
| P6-03 | 优化器构造器重构 | 2-3 天 | Release 模式错误报告 |

### 🔮 远期探索（大投入 + 大收益）

| 编号 | 方案 | 工作量 | 收益 |
|------|------|--------|------|
| P7-01 | Flash-Attention | 2-3 周 | 注意力速度 + 显存 |
| P7-03 | 分布式训练 | 2-4 周 | 多卡扩展 |
| P4-04 | 8-bit Adam | 1 周 | 优化器显存 7.4→2 GB |
| P7-02 | PagedAttention / GQA | 1-3 周 | 推理扩展 |
| P7-04 | Auto-tuning | 2-3 周 | 自动最优配置 |
| P7-07 | 量化推理 | 1-3 周 | 推理速度 2-4× |

### ❌ 经评估搁置

| 编号 | 方案 | 搁置原因 |
|------|------|---------|
| P3-04 | 8 路累加器 dot_kernel | 编译器自动向量化已最优，手写反而让 forward 回退 |

---

## 9.5 本对话（2026-08-26）新增优化汇总

> 由 GPU 后端缺陷审查 + IR 中间张量消除设计 + 进阶 IR/并发/算子优化讨论整理而成。

### ✅ 本次已完成

| 编号 | 项 | 状态 |
|------|-----|------|
| D1 | GpuBuffer 延迟销毁 buffer 内存重叠（规范违规）修复 | ✅ 已修（内存归还延迟到 vkDestroyBuffer 之后） |
| D2 | scatter_add CAS 原子加非确定性 | ✅ 已文档化为例外（docs/08 坑 19，不修复） |
| D3 | node_outputs_ 跨线程共享数据竞争 | ✅ 已修（录制图改 thread_local unique_ptr） |
| P1 | IR 融合中间节点占位 buffer 显存消除 | ✅ 已实现（execute_fused_graph 释放被融合节点 buffer） |
| P2 | from_matrix 批内免强制 flush | ✅ 已实现（上传新建 buffer 独立提交先于 batch） |

### 📋 待做（按优先级）

| 优先级 | 编号 | 方案 | 工作量 |
|--------|------|------|--------|
| P0 | P2-10 | 跨 kernel 自动融合（消除手工 begin_expr） | 1 周 |
| P0 | P2-09 | 归约→广播→逐元素链融合 | 1 周 |
| P1 | P1-10 | matmul tile 调优 | 1-2 天 |
| P1 | P1-11 | 共享内存两级归约 | 3-5 天 |
| P1 | P2-15 | 融合分析并行化 | 2-3 天 |
| P1 | P2-16 | 双缓冲 command buffer 流水化 | 3-5 天 |
| P2 | P2-11 | 融合成本/收益启发式 | 3-5 天 |
| P2 | P2-12 | 图级缓存跨 step 复用 | 3-5 天 |
| P2 | P2-13 | 视图组合化简 | 1-2 天 |
| P2 | P2-14 | 冗余原语收敛 DSL | 中 |
| P2 | P1-12 | scatter_add 原子优化 | 1 天 |
| P2 | P1-13 | matmul 代码去重 | 1-2 天 |
| P3 | P2-17 | 多线程数据并行 IR | 2-3 周 |
| P3 | P2-18 | 内存池并发安全 | 3-5 天 |
| P3 | P2-19 | 并发确定性保证 | 1-2 天 |

---

## 10. 已完成优化索引

| 编号 | 方案 | 完成日期 | 关键文件 |
|------|------|---------|---------|
| P1-01 | vec4 向量化发射 | 2026-08-25 | `glsl_gen.hpp`, `gen_fused.cpp`, `vk_backend.hpp` |
| P1-02 | 按需 row/col | 2026-08-25 | `glsl_gen.hpp` |
| P1-03 | GPU matmul tiled | — | `shaders/matmul_tiled.comp` |
| P1-04 | 标量路径 row/col 除法消除 | 2026-08-25 | `glsl_gen.hpp` |
| P1-09 | warp shuffle 归约 | 2026-08-25 | `glsl_gen.hpp` |
| P1-05 | 归约指令 pass 多累加器 | 2026-08-26 | `glsl_gen.hpp`（原 `reduce_instr_bench` 已移除） |
| P2-01 | IR-A DCE/折叠/化简 | 2026-08-23 | `expr_opt.hpp` |
| P2-02 | IR-B CSE/寄存器分配 | 2026-08-23 | `expr_opt.hpp` |
| P2-03 | IR-C 图 IR/链融合 | 2026-08-24 | `expr_graph.hpp` |
| P2-04 | IR-D emitter 抽象 | 2026-08-24 | `expr_emitter.hpp`, `cpu_emitter.hpp` |
| P3-01 | 缓存分块 matmul | — | `config.hpp` |
| P3-02 | SmartPolicy 自适应并行 | — | `config.hpp` |
| P3-03 | 线程池 latch 零分配 | — | `core_threadpool.hpp` |
| P4-01 | 激活重计算 L1 | 2026-08-22 | `compute_layer.hpp`, `model_container.hpp` |
| P4-02 | 内存池归还 L2 | 2026-08-22 | `backend/memory_pool.hpp` |
| P5-01 | CUDA 后端基础 | — | `backend/cuda_backend.hpp` |
| P6-01 | 序列化 EOF 语义修复 | 2026-08-25 | `model_serialization.hpp` |
| — | 注意力 M4/M5/M6 融合 | 2026-08-21 | `compute_layer.hpp`, `shaders/` |
| — | 列归约 tile 化 | 2026-08-25 | `glsl_gen.hpp`, `vk_backend.hpp` |
| — | CNN im2col（CPU 端） | 2026-08 | `compute_layer.hpp` |
| D1 | GpuBuffer 延迟销毁 buffer 内存重叠修复 | 2026-08-26 | `backend/vk_backend.hpp` |
| D3 | node_outputs_ 线程安全修复 | 2026-08-26 | `gpu_engine.hpp`, `cpu_engine.hpp`, `expr_graph.hpp` |
| P1 | IR 融合中间节点 buffer 显存消除 | 2026-08-26 | `gpu_engine.hpp` |
| P2 | from_matrix 批内免 flush | 2026-08-26 | `gpu_engine.hpp` |
| D2 | scatter_add 非确定性文档化例外 | 2026-08-26 | `shaders/scatter_add.comp`, `docs/08` |

---

> **使用建议**：本文档按模块和优先级组织，可作为 roadmap 输入。建议每季度 review 一次，根据实际需求和技术演进调整优先级。
