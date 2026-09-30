# 仓库历史记录（唯一归档）

> 本文档是全仓**唯一**允许记录历史状态的地方：代码注释与其余文档只描述
> 「当前是什么、为什么这样实现」；被移出的演进过程、已修复勘误、被否决方案、
> 性能 A/B 过程一律归档到这里。新增历史内容请追加到对应分组，**不要**回流到代码注释。

按来源分组；每条注明**整改前的原位置**（`文件:行` 或章节），行号以整改前版本为准。
类型标注：bug 根因 / 否决方案 / 性能 A/B / 删除清单 / 演进记录 / 覆盖缺口。

## 目录

- [表达式 DSL / 融合 IR（expr_*.hpp）](#表达式 DSL / 融合 IR（expr_*.hpp）)（15 条）
- [GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）](#GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）)（17 条）
- [注意力 / ZiPT / Transformer / FeedForward 层](#注意力 / ZiPT / Transformer / FeedForward 层)（16 条）
- [RAPT / CNN / MLP / Softmax / GPT 层](#RAPT / CNN / MLP / Softmax / GPT 层)（12 条）
- [引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器](#引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器)（19 条）
- [代数层 / 基础设施 / 模型容器与序列化](#代数层 / 基础设施 / 模型容器与序列化)（12 条）
- [领域模型工厂与 CLI 公共头](#领域模型工厂与 CLI 公共头)（3 条）
- [src 应用入口（text/mnist/tokenizer/bench）](#src 应用入口（text/mnist/tokenizer/bench）)（5 条）
- [GPU 手写原语 shader（shaders/*.comp）](#GPU 手写原语 shader（shaders/*.comp）)（12 条）
- [构建期工具 / 基准脚本 / 对拍脚本](#构建期工具 / 基准脚本 / 对拍脚本)（12 条）
- [表达式与融合测试](#表达式与融合测试)（18 条）
- [层 / 模型测试](#层 / 模型测试)（14 条）
- [精度 / RAPT / ZiPT / 分词器测试与探针](#精度 / RAPT / ZiPT / 分词器测试与探针)（11 条）
- [docs/development 12 · 13 · 08](#docs/development 12 · 13 · 08)（17 条）
- [docs/development 02 · 03 · 05](#docs/development 02 · 03 · 05)（19 条）
- [docs/development 01 · 04 · 06 · 07 · 10 · 14](#docs/development 01 · 04 · 06 · 07 · 10 · 14)（25 条）
- [docs/introduction · usage · benchmarks](#docs/introduction · usage · benchmarks)（46 条）
- [构建系统 / CI（CMakeLists.txt、.github/workflows）](#构建系统 / CI（CMakeLists.txt、.github/workflows）)（1 条）
- [AGENTS.md](#AGENTS.md)（24 条）

---

# 表达式 DSL / 融合 IR（expr_*.hpp）

表达式 DSL / 融合 IR 家族头文件（expr_dsl / expr_emitter / expr_fold / expr_glsl_gen /
expr_opt / expr_registry / expr_spec）「历史状态类注释」摘录。整改原则：注释只记录当前实现的
原理、契约与理由；本文件保存被移出的历史叙事（bug 根因、被否决方案、性能 A/B 过程、
删除清单），供追溯用。

## 融合 matmul BK 取值 A/B 流水（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:262-269、303-310）
- 类型：性能 A/B / 否决方案
- 内容：40HX 变体 trade-off 记录（% = vs 单缓冲耗时，负=更快）：BK=32 单缓冲(16KB) 为深网格基线、
  浅网格无流水；BK=32 双缓冲(32KB) barrier 节奏同原版但占用率砍半，浅点收益（linear 1024³ -1.9%、
  batch512 -4.9%）深点 ±0，**曾判"最终采用"**；**2026-09-24 OP/融合统一 A/B 复测推翻上条**（同参数
  四点对照，AGENTS §12 ⑥）：融合侧 BK32→16 反超，浅 -24%、深 fwd -24%、深 train -17%
  （8.21→6.81ms）→ 最终 BK=16，与 OP 级同参。历史口径备注：早期"深网格 BK16 -2%"是对单缓冲
  旧基线的口径，非同参双缓冲对比。整改后注释只留当前结论：BK=16 双缓冲、四点全点最优。

## 融合 matmul 共享布局与 32KB 设备要求（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:370-381）
- 类型：性能 A/B / 否决方案
- 内容：vec4 沿 k + 16 dot 的旧布局 ≈5× 指令发射，是融合 matmul 落后 op 级 ~2× 的主因
  （2026-09-24 由 fused_49fa 与 op 级同形状对拍 7.2 vs 3.15ms 定位）→ 改 vec4 沿 m/n。
  旧注释按 BK=32 描述：[2][32][16]vec4×2 = 32KB（布局改向后总量不变，旧 [2][8][64]vec4 同为
  32KB；BK16 变体曾 A/B 深网格 -2% 不取）；> 单缓冲 16KB → blocks/SM 砍半、深网格靠 WG 余量补、
  浅网格靠流水补；需设备 maxComputeSharedMemorySize ≥ 32KB（Vulkan 规范下限仅 16KB，合规低端
  设备 pipeline 创建失败）。代码现为 BK=16 双缓冲（Ash/Bsh 各 [2][16][16]vec4 = 8KB，合计 16KB，
  恰在规范下限内），原注释与代码不符、已按当前实现改写。

## TARGET_CHUNK=65536 的实测记录（原位置 include/neuralnet.cpp/expr_dsl.hpp:264-272）
- 类型：性能 A/B
- 内容：probe_par_break_even / probe_chunk_at_shape（本机 32 逻辑核）一次实测记录，结论"保持不变"：
  并行区固定开销 ~100µs（32 线程）/ ~3~16µs（≤16 线程）；隔离内核上计算型 body（exp）n=393216
  时 nch=48 比 nch=6 快 1.4x、带宽型 body（x-mean）任何 nch 都不如串行；但 TARGET_CHUNK 改
  4096/16384（→96/24 块）后层级别反而变差：layernorm 2.8→3.2ms、rmsnorm 1.7→1.9ms
  （softmax 1.99→1.7 改善）、transformer/gpt_block 合计下降——隔离内核收益不传递到层。
  故维持 65536，再动必须先拿层级别证据。

## cpu_preparable_leaf_v 默认值教训（原位置 include/neuralnet.cpp/expr_dsl.hpp:1060-1061）
- 类型：bug 根因
- 内容：白名单默认值曾被设为 true，导致 SwiGLU::backward 的 select(row()<d_ff,…) 在模板路径下
  row() 恒为 0、两半梯度选错（layer_gradcheck 的 fc1.w 全红）。由此确立 fail-safe 默认：
  只有 has_reduction_v 为假才可预绑定，含归约节点必须逐个显式加白名单。

## MatmulRef::prepare_cpu 精度-存储不匹配（原位置 include/neuralnet.cpp/expr_dsl.hpp:556-561）
- 类型：bug 根因
- 内容：C 固定以 P=F32 物化，而操作数可能带 f16 存储：内层 GEMM 的 f32 路径按 f32 存储直接读，
  cpu_matrix<F32>() 对 f16 存储返回空指针 → Release 读空 UB、Debug 断言；实测 CPU f16 训练
  step0 forward 即 NaN、text_train 0xC0000005。修复 = 先把非 f32 操作数抬到 f32 再物化
  （docs/development/05-mixed-precision.md §12.12 的双根因之一）。整改后注释只留当前契约与 UB 后果。

## 归约 max 恒等元 −inf−−inf=NaN（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1285-1288、2136-2139 等 4 处）
- 类型：bug 根因
- 内容：max 恒等元取 -inf 时，全 -inf 输入（doc 掩码整块屏蔽）下 m_old=blk_m=-inf →
  dm = −inf−−inf = NaN 污染 l/O，是 2026-09 训练 -nan 的根因。全库归约（fold v1/v2、归约
  视图、归约指令、reduce 原语）统一 numeric_limits::lowest()（0xFF7FFFFF）根除该风险类。
  整改后注释只留 NaN 机制与统一 lowest 的现状。

## EXPR_FOLD_BLOCK 取值阶梯（原位置 include/neuralnet.cpp/expr_spec.hpp:825-827）
- 类型：性能 A/B
- 内容："历次调整均以交错 bench 实测裁决：32→64（每块协议减半，fwd −6%）；64→128（协议再减半 +
  链/归约活跃线程翻倍（tid<BLOCK）；shared 6.75→~8.9KB、驻留 8→7 WG 的占用代价被收益盖过——
  mha fwd 5.77→5.41，18/18 绿）"。整改后注释只留 128 的取值依据（相对 64 的结论与代价数字）。

## vecacc 4 路软件流水 A/B（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1353-1355）
- 类型：性能 A/B
- 内容：旧版单发串行读 V（GPU 无硬件预取 → 64 级 L2 延迟链完全暴露），探针实测占 fold 55%
  （5694→2567µs）。4 路流水每拍发出 j..j+3 四个读使延迟重叠；四条独立累加语句保持 j 升序求和
  与旧版逐位一致。整改后注释只留当前结构的延迟重叠原理 + 55% 占比依据，删 A/B 对数值与
  "与旧版逐位一致"表述。

## 归约蝶式替换串行扫描（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1271-1281、931-934）
- 类型：否决方案 / 性能 A/B
- 内容：旧实现 256 线程各串行扫 valid（64 级 shared 依赖链 ×256 冗余，窗口探针量化占 fold
  ~11%）；旧头注释"免 subgroup 防混邻 WG lane"的顾虑源自跨 WG 混 lane 的担心，对合规驱动不成立
  （规范保证 subgroup ⊆ workgroup，AMD wave64 经 gl_SubgroupSize 泛化）→ 采 subgroup
  shuffleDown 蝶式。整改后注释保留"串行扫描占 ~11%"的选型依据，删"旧实现/旧头注释/故此改"叙事。

## finalize 写后读跨迭代污染的实测症状（原位置 include/neuralnet.cpp/expr_spec.hpp:1093-1096）
- 类型：bug 根因
- 内容：dst 复用被读状态 l 时首列输出覆盖除数、次列起全错，实测症状值 0.4=10/25。
  静态校验（向量域多列循环下 finalize 被写寄存器不得再被读）即由此而来。

## vec_state_len 不进 key 的实证（原位置 include/neuralnet.cpp/expr_spec.hpp:645-648、expr_glsl_gen.hpp:973-974）
- 类型：bug 根因 / 否决方案
- 内容：veclen 曾进 expr_spec_key，导致每个 dk 一个 shader、闭合世界永远缺登记
  （offload/attn 测试按 dk miss 实证）→ 改为运行时经 PC 的 vector_out 槽填充（形状无关融合）。
  整改后注释改为"若进 key 则必然缺登记"的现在式警告。

## bin 格式版本流水（原位置 include/neuralnet.cpp/expr_registry.hpp:97-110）
- 类型：删除清单 / 版本流水
- 内容：原注释含逐版编年：v2 起支持 matmul 段（v1 无 matmul，读 v1 等价 has=0）；v3 rparams；
  v4 views.param2（RowAccess offset）；v5 MatmulSpec.batch；v6 FoldSpec 段；v7 FoldSpec 双域字段
  （vec_state_len/matmul/vecacc）；v8 tri_skip；v9 精度变体段。布局行原写作 version (u8=8)（陈旧，
  实际 kExprBinVersion=9）。read_registry 严格要求 ver == kExprBinVersion，旧版本一律拒绝，
  编年对当前代码无约束力 → 整改后只留当前字段布局 + 读写对称警告，version 行改指常量。

## DslExpr 概念的消歧由来（原位置 include/neuralnet.cpp/expr_dsl.hpp:748-750）
- 类型：删除清单
- 内容：原注释：DslExpr 约束偏序设计是"让 DSL 运算符严格优先于旧代数 nn::operator* 等"
  （旧代数与 DSL 共用 nn::Expression，不区分则二者对 DSL 叶子同为候选）。旧代数 AST
  （algebra_expr.hpp/algebra_compute.hpp）已于 2026-09 整体移除，该竞争者不复存在
  → 整改后改写为当前理由：DSL 运算符只接受可折叠节点（带 to_spec），非折叠的
  nn::Expression 不成为候选、误用编译期暴露。

## 带类型变体支持范围的过期表述（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1391-1392、1408-1410）
- 类型：删除清单 / 版本流水
- 内容：原注释称"目前只支持纯逐元素形态；reduce / matmul / fold 拿到 sig != 0 时返回空串"
  与"已支持：纯逐元素 + matmul 段 + fold 段"自相矛盾（in-kernel f16 分期推进后的过期残留）。
  按代码现状订正：generate_glsl 支持纯逐元素 / 前置 matmul 段 / fold 段（v1/v2）的带类型生成，
  归约 kernel 由 generate_glsl_reduce 独立支持（sign_f16 + rd/wr 读写点转换）；gen_fused 按
  raxis 路由两者，仍返回空串的只有归约混合轴（== -2）与归约槽过多（>8）。

## 其余一句话级过时描述（直接删除，未摘录）
- 类型：删除清单
- 内容：expr_emitter.hpp「历史上的 CpuEmitter / CudaEmitter 均已随各自后端删除」；expr_fold.hpp
  「对拍 engine.row_reduce_max/row_reduce_sum」（引擎 row_reduce_max 算子已删，测试现对拍独立
  标量参考）；expr_dsl.hpp「此前只能靠引擎原地原语 add_inplace/scale_inplace」「DSL 取代
  elementwise_* 时性能不倒退」「与迁移前逐字节/逐位一致」等迁移等价性表述；expr_glsl_gen.hpp
  「GLSL 与迁移前逐字节相同（零回归）」「原行为」「语义与旧版一致」「此前出现的运行时参数视图
  个数（改写为按视图序累计）」；expr_glsl_gen.hpp 断链引用 matmul-opt.md（文件不存在，已删）；
  expr_dsl.hpp「未来运行时 JIT」表述（与 AOT 闭合世界铁律冲突，改写为当前执行策略）。

---

# GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）

GPU 后端 / GPU 引擎头文件「历史状态类注释」摘录。整改原则：注释只记录当前实现的原理、契约与理由；
本文件保存被移出的历史叙事（bug 根因、被否决方案、性能 A/B 过程、删除清单），供追溯用。

## solo fence / command buffer 复用的由来（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:12）
- 类型：删除清单
- 内容：旧实现每个算子各自 vkCreateFence/vkDestroyFence + vkAllocateCommandBuffers/vkFreeCommandBuffers，
  是逐元素算子固定开销的主要构成之一；现改为 initialize() 预分配 solo fence + solo cmd，submit 前 reset 复用。

## 单帧阻塞提交模型（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:559-562）
- 类型：否决方案 / 删除清单
- 内容：旧实现是「单一 batch_cmd_ + 单 fence」，end_batch/flush_batch 每次 vkQueueSubmit 后立即
  vkWaitForFences —— GPU 执行 step N 时 host 被 fence 卡死、无法录制 step N+1。原注释引用的
  《显存 & 负载不均衡分析报告》§2.1 在仓库中不存在（断链，整改时删除）。现为 PIPELINE_FRAMES=6
  的 (cmd + fence) 帧环，提交不等待、wait_in_flight 才阻塞。环深度 3→6 的加深记录：3 槽时
  host 录制与 GPU 执行重叠度不足。

## 逐元素算子固定开销（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:617）
- 类型：性能 A/B
- 内容：改造前逐元素算子固定开销实测 ≈0.16ms/次（kernel 本身 4096² 已达 370GB/s，瓶颈全在 host 侧）；
  solo fence/cmd 复用后降至 ≈0.143ms（AGENTS §12 同一改动的口径）。整改后注释只保留当前值 ≈0.14ms。

## 帧环预分配（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1304-1305）
- 类型：删除清单
- 内容：旧实现每步（step 边界）vkAllocateCommandBuffers + vkCreateFence；现为 initialize() 预分配
  PIPELINE_FRAMES 个 cmd + fence，全程零分配。

## run_fused_gpu 固定头长度 bug（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3495-3498）
- 类型：bug 根因
- 内容：「归约但无 matmul」形态曾按 5 个 uint 计算固定头（真实头部只有 4 个）→ push constant 常量池
  整体后移一个 uint → shader 从错位处读常量，实测把 select(cond,1,0) 的常量读成垃圾，
  **GPU 上带常量的归约静默错值、CPU 正常**（表现为 col_reduce_sum(...) 恒返回 kk-1 而非真实并列数）。
  教训：这类「只有 GPU 错」的问题先打印生成的 GLSL/IR 再猜成因。

## fold PC 漏分支 → 未定义残留（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1727-1728）
- 类型：bug 根因
- 内容：pipeline 创建侧（pc_base 计算）漏 fold 分支时 range 少算，vkCmdPushConstants 超 range 部分被
  驱动丢弃 → fold_k 读未定义残留，**曾致滑窗式错值**，且残留随前序 op 漂移、表现为时对时错的假 PASS
  （对拍全绿）。整改后注释改为直接陈述该失败模式与「创建侧/写入侧必须同改」的契约。

## matmul dispatch 曾用 WORKGROUP_SIZE=16 统一计算（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2560-2562）
- 类型：bug 根因 / 性能 A/B
- 内容：曾按 WORKGROUP_SIZE=16 统一计算工作组数 → tiled 版（64×64 块）多派 16 倍冗余工作组
  （每 16×16 一个组），GPU 做 16 倍无效计算，实测 matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。

## transpose 曾误派发 16 宽砖（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:4602-4603）
- 类型：bug 根因（issue #13 P0-①）
- 内容：shader 按 8 宽砖解算 tile_c，dispatch 曾给 x=16 → tile_c 跨两砖且越界 tile 只早退一半，
  行>512 且列>512 时静默只写前 512 行。

## 物理设备选择：「第一个 DISCRETE_GPU」（原位置 include/neuralnet.cpp/backend/compute_vk_device.hpp:184）
- 类型：否决方案
- 内容：旧规则「取第一个 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU」在本机枚举顺序下会选中 [0]
  AMD Radeon R5 240（老专有驱动，maxComputeSharedMemorySize 仅 32768 < matmul 分块所需 34560，
  校验层报 VUID-RuntimeSpirv-Workgroup-06530；且对跨 submit 信号量重复 wait 直接死锁）。
  现为「设备类型权重 + apiVersion 打分、同分取先枚举者、D3D12 转译层降权」。

## 上传 region 专属 cmd 与「submit 后立即 free」（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2104-2110、compute_staging_ring.hpp:80-83）
- 类型：bug 根因 / 删除清单
- 内容：旧代码每次上传 vkAllocateCommandBuffers、submit 后立即 free/复用，违反
  VUID-vkFreeCommandBuffers-pCommandBuffers-00058（禁止释放 pending 的 command buffer）；
  实测导致驱动通道排序失效、fence 提前 signal、跨 submit 乱序执行。

## staging ring 初版预分配（原位置 include/neuralnet.cpp/backend/compute_staging_ring.hpp:9-13、47）
- 类型：性能 A/B / 删除清单
- 内容：旧默认 4×256MB=1GB 预分配过大；V100 上按「整机 host heap 1/16」规则曾达 2×2GB=4GB host 预算。
  现默认 4 个 region、每 region 取 max(用户请求, clamp(host_visible/16, 32MB, 256MB))，
  且单次 PCIe 传输量通常 << 64MB（最大 token_emb 上传约 5MB）。
  注：文件头原写「默认 2 个 region / 2 regions 共 512MB」，与 DEFAULT_NUM_REGIONS=4 不符，一并订正。

## 列归约两段式的 A/B 方法论（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3713-3717）
- 类型：性能 A/B
- 内容：同负载窗交错 A/B 定案方法：TP/SP 两二进制逐轮换序、warmup 30、best-of-10、同码对照噪声地板
  ±5%；结果 5244² TP 4/5 胜（中位 0.405 vs 0.430，−6%）、4096² SP 5/5 胜 ~2%（0.277 vs 0.283）、
  512² TP −16%（对 ~10µs kernel 第二遍+屏障净亏）→ 512K 元素尺寸护栏只排除微型形状。
  整改后注释只保留三点实测数字与护栏结论。

## eval_expr_into 的 output_override 曾写死 F32（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:1639-1645）
- 类型：bug 根因（issue #13 P0-②）
- 内容：曾写死默认 F32 的 gpu_tensor()：f16 目标 → gpu_get<F32>() 空 shared_ptr → Debug NN_ASSERT 引爆；
  Release 空解引用 = UB：MSVC 得到空 override → 结果落临时 buffer 被丢弃 → **参数静默冻结
  （loss 恒 ln V）**；clang 当时「正常」只是 UB 代码生成运气。修复 = f16 目标经 f16_view 包 f16 buffer。

## 显存底材 128MB → 12MB 调参（原位置 include/neuralnet.cpp/backend/compute_memory_pool.hpp:42-52）
- 类型：性能 A/B
- 内容：2026-09 探针实测调参：峰值对块尺寸在 8–32MB 是平台区、最优 ≈12MB 且与模型规模无关
  （跨负载表保留在注释中）；「最小 2 的幂」阶梯类（NN_POOL_LADDER_MAX_MB）实测 d128 上
  ladder16=4497 比 fixed16=4457 差 40MB（配对 3/3），故阶梯默认关闭。

## 真原地语义与 cpu_fallback 的删除（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:25、466）
- 类型：删除清单
- 内容：2026-09 起引擎内 add_inplace/scale_inplace 等全部改为真原地（此前 copy-on-write 描述已作废）；
  GpuEngine::insert_rows 的 cpu_fallback（CPU 目标回退路径）已删除，CPU 目标直接报错。
  整改后注释只陈述当前契约（真原地的 read-before-write 论证、dst 必须为 GPU Tensor）。

## 收割点提前的探针数据（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1850-1854）
- 类型：性能 A/B
- 内容：原先 reap_completed_frames 只在 step 边界调用 → 步内已析构中间张量的延迟销毁挂在
  「已完成但未被收割」的帧上不还池；bench 配置实测 transient live 3837MB 里 pending 高达 3000MB，
  步内峰值 ≈ 稳态 2.8×。现收割提前到每个帧提交点（submit_frame_no_wait）。
  整改后注释保留数字与「收割必须紧随提交点」的结论，去掉「原先/2026-09 起」的演进叙事。

## 精度归因编号（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:530、989、1272）
- 类型：性能 A/B
- 内容：曾用探针归因编号描述 f16 带类型变体的收益来源：scatter_add 的 (8208,64)+(8192,64)、
  add_inplace 的 inplace2_ 大头、reduce 的 (64,8192)×38（Linear::backward grad_bias）。
  这些编号依赖一次性探针输出，整改后改为直接陈述「免去 f32 副本物化」这一当前设计收益。

---

# 注意力 / ZiPT / Transformer / FeedForward 层

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

## 注意力变体拆分 + 位置编码多态统一（2026-10-01，原位置 include/neuralnet.cpp/compute_layer_attention.hpp 整文件结构、compute_layer_gpt.hpp PositionEncoder 家族、compute_layer_rapt.hpp:60/253/270/335/423/733/813/1143、compute_layer_zipt.hpp:806-817）
- 类型：演进记录
- 内容：原 `AttentionBase` 一个类里同时用 `use_rope_` 布尔 + `use_alibi_`/`has_doc_ids_` 布尔 + `fold_mask_variant_()` 虚钩子 + 4 个 `masked_*_` 模板承载"多种注意力"，且 `recompute_W_` 每次反向都重跑 `if (fold_mask_variant_() == Plain) … if (use_slopes && use_doc) … if (use_slopes) … if (use_doc) …` 组合链（forward 每调用也重算变体，另有三处 `if (use_rope_)`）。整改为**两个正交策略对象、构造期定型**：
  ① `AttnScoreMask` 族（`PlainScoreMask` / `CausalScoreMask` / `CausalDocScoreMask`，**只做掩码**）——每种掩码语义一个类，各自实现类级常量 `mask_kind()`、`prepare()`、`append_fold_inputs()`、`masked_scores()`（反向 DSL 文本）；由 `AttentionBase::make_score_mask_()` 在**配置期**（构造 / `set_doc_ids`）工厂化选定，forward/backward 各只做一次虚调用。文档掩码的"有/无"是运行期数据，其裁定落在 `set_doc_ids` 这一**配置调用**里（模式不变时零开销），不再是每步 forward 的判断。
  **后续同一轮内又把 ALiBi 偏置从掩码策略里摘出去**：原 `CausalAlibiScoreMask` / `CausalAlibiDocScoreMask` 两个类把"掩码"和"位置偏置"融成同一条 select，掩码工厂因此必须查询 `pos_->has_score_bias()`。现在掩码族只保留三个纯掩码类，位置偏置由 `PositionEncoder::apply_score_bias()` 在 backward 的"掩码之后、softmax 之前"独立叠加（forward 仍融在同一个 fold kernel 内，由 `make_fold_attn_o` 的正交第二入参 `bool score_bias` 表达）。`expr::FoldAttnMask`（5 值融合枚举）随之改为 `expr::AttnMaskKind{Plain,Causal,CausalDoc}` + `bool score_bias`，`scan_exprs` 登记块改为 3×2 组合循环（跳过不存在的 "Plain + 偏置"），掩码策略类由 5 个降为 3 个。
  ② `PositionEncoder` 族（新文件 `compute_position_encoding.hpp`）——`RotaryEmbedding` 从 attention 头迁出、由 `RopePositionEncoder` 持有；`use_rope_` 布尔消失，Q/K 旋转改走策略对象（非 RoPE 策略 = 恒等 no-op）。`PositionEncoder` 基类按**注入点**分三组（嵌入侧 `apply/apply_step/backward`、Q/K 侧 `apply_qk/apply_qk_step/set_position_offset`、分数侧 `has_score_bias/prepare_score_bias/apply_score_bias/apply_bias_step`），`Learned/Sinusoidal` 走嵌入侧、`RoPE` 走 Q/K 侧、`ALiBi` 走分数侧、`NoPositionEncoder` 恒等——各子类只覆写自己那组，其余继承基类 no-op。
  ③ **⚠ 拆开时踩到的坑（`row()` 的批内语义）**：ALiBi 的偏置项原来是 `batch_mod(slopes, H) * (col − row)`，`row()` 是"批内行号"、其分解来自**同一 ExprSpec 里的 matmul 段**（`MatmulSpec.batch`）——旧实现把它融在掩码表达式里，那个表达式恰好含 `dsl::matmul(..., BH)`，所以正确。拆成独立一步后 spec 里没有 matmul → batch 退化为 1 → `row()` 变**全局行号** → 偏置静默错值（探针实测 |偏置| 达 27，理论界 (seq−1)·m_0≈7），`attn_test` 的 alibi+doc gradcheck 立刻失败。修法：把 `(col − row)` 换成**`(rows,1)` 行表 + `dsl::row_broadcast`**（按全局行号直读，与网格分解无关）——`AlibiPositionEncoder::prepare_score_bias` 建两张 O(B·H·seq) 行表（每行斜率 `m_{h(row)}` 与批内位置 `row % seq`），`apply_score_bias` 用 `leaf(S) + rb(slope_row) * (col() − rb(pos_row))`。该坑已写进 `AGENTS.md` §7 的告警与 `docs/development/02` §IR 扩展。
  ④ **所有权按注入点划分（谁拥有 = 谁负责）**：`PosEncodingType` 的分发收敛为两个工厂，同一策略类型只在一个工厂里实做、在另一个里映射为恒等——
     · 模型侧 `make_embedding_position_encoder(type, d_model, seq_len)`（Learned/Sinusoidal 实做），由 `GPTModel`/`ZiPTModel`/`RAPTModel` 持有并在 `apply`/`apply_step` 里施加；
     · 注意力侧 `make_attention_position_encoder(type, d_k, num_heads)`（RoPE/ALiBi 实做），由 `CausalSelfAttention`/`ReLULinearAttention` **自持**（`AttentionBase::install_position_encoder`），在 `apply_qk`/`apply_score_bias` 上取数。
     这样层间**没有任何位置编码对象传递**：删掉了 `AttentionBase::set_position_encoder(PositionEncoder&)`、`GPTBlock::set_position_encoder`、`RAPTBlock::set_position_encoder` 与两个模型的注入循环，也一并去掉了"非拥有指针指向模型对象的生命周期约束"（原先为规避层对象移动导致成员地址失效，兜底对象还得特意放堆上——现在 `pos_` 就是层自己的 `unique_ptr` 成员）。代价是 **RoPE 的 cos/sin 表随层构建**：每层 `2·d_k·seq·4B`，本仓默认配置（`d_model=128`/`H=4`/`layers=4`/`seq=256`）≈ 64KB/层、全模型 256KB，三角函数只在 `seq` 变化时算一次；层数极多的配置才需要回头考虑共享。
     另外，`AttentionBase::init_impl` 改为**调用** `pos_->init(engine)`（三组策略的 init 都是 no-op）——符合 M6 段 C 的不变量"复合层 `init_impl` 必须 init 全部子对象"；此前"不 init"是"注入的模型级对象由模型自己 init"这一设计的副产物。`max_len` 参数保留仅为签名兼容，当前实现不使用。AOT 闭合世界不受影响：`make_fold_attn_o` 与 5 个组合的登记仍在 `scan_exprs` 的显式块中。
- 验收：build 零告警（`-Werror`）；ctest 20/20（含 `NN_BIND_DEBUG=1` 一轮）；CPU 锚 `6f8849f14da23110` 与 GPU dev2 锚 `8ef51b2927253c50` 逐位不变；`--init-hash` 六模型与 M2 锚全同；`--io-roundtrip` CPU/GPU 全过；L2 审计 `L2-VIOLATIONS: 0`（`bench/doc_inventory.ps1` 的 L2 文件清单已补入 `compute_position_encoding.hpp`，否则新文件不在门禁内）；引擎 virtual 方法仍 49、Layer 直调算子仍 21。
  **scan 双 hash 在本轮内变过两次**（`bdc3a442…`/`7a10412c…` → `74e02702…`/`bb03e5e2…` → `45edce4a…`/`15899056…`）：位置编码**所有权**调整那次（注入 → 自持）表达式文本未动、hash 逐字节不变；而把 ALiBi 偏置从掩码表达式里拆成独立一步那次**必然改变**backward 的表达式树（`scores + select(blocked,-inf,alibi)` → `(scores + select(blocked,-inf,0)) + bias`），故 hash 变化是设计结果、不是漂移。两次都做到了**数值逐位不变**（CPU/GPU 字节锚不动）——这正是"改结构不改数"的判据：hash 说明结构变了，字节锚说明值没变。

## ActivationOffloader slab 容量校验（原位置 include/neuralnet.cpp/compute_layer_base.hpp:196）
- 类型：bug 根因
- 内容：slab 原先只在 `!slab_.valid()` 时按当时的激活总量分配、之后永不增长；后续 step 激活总量变大（批大小/序列长度变化、--resume 后续训、最后一个不满 batch 之后的 step）时 `offload_save` 按新 offset 越界写 slab → 缓冲区破坏/设备丢失。修复为每次导出按当前总量校验、不足即重建。头文件只保留现行不变量与其后果。

---

# RAPT / CNN / MLP / Softmax / GPT 层

> 摘自 RAPT/RLA、CNN 卷积池化、MLP、Softmax、GPT 模型层头文件的历史状态类注释（本轮清理）。
> 每条记录原位置、类型与内容；琐碎的一句话描述直接删除、未收录。

## Layer 层四处收敛：参数收集 / Pre-Norm 残差反向 / 解码采样 / 错误传播（2026-10-01，原位置 compute_layer_{gpt,rapt,zipt,transformer,attention,feedforward,base}.hpp、core_errors.hpp）
- 类型：演进记录
- 内容：
  ① **参数收集**：`parameters()` / `param_gradients()` 里 `p.insert(p.end(), x.begin(), x.end())` 的级联在三个模型 + 四个块层共 15 处同构重复 → `Layer::collect_refs(...)`（变参拼接：实参可以是单个 `Tensor` 或子层返回的 `std::vector<TensorRef>`）与 `Layer::collect_block_refs_(blocks, &Type::fn)`（块容器逐块取参）。拼接顺序即实参顺序，不改变优化器看到的参数序。
  ② **Pre-Norm 残差**：`GPTBlock` / `TransformerEncoderLayer` / `RAPTBlock` 的两处残差分流（正/反向）逐字同构 → `Layer::prenorm_residual_forward_` / `Layer::prenorm_residual_backward_`。**残差相加的表达式输出精度显式传入**：GPT/Transformer 传 `p.compute`，RAPT 历史上未传 profile（恒 F32），故传 `Precision::F32`——"顺手统一成 p.compute"会改变 f16 路径的数值行为，因此不做，只把差异显式化。
  ③ **解码采样**：`GPTModel/RAPTModel/ZiPTModel::generate()` 里"temperature 缩放 → 数值稳定 softmax → 随机采样 / 贪心"三段逐字相同 → `sample_next_token_()`（`compute_layer_base.hpp`）。RNG 消耗序（每步恰好一次 `dist(rng)`，仅采样分支）保持不变。**解码循环骨架刻意未合并**：三者运行态本质不同（GPT = KV cache 增量 + 滑窗重建、RAPT = RLA 运行态、ZiPT = 无状态整窗前向），用标志位强行统一会得到更难读的"上帝函数"，与"保持 Layer 简洁"的目标相反。
  ④ **错误传播**：`auto r = f(); if (!r) return std::unexpected(r.error());` 在 L2 层重复 494 处、占 Layer 代码近 9% → `NN_TRY(decl, expr)` / `NN_TRY_CHECK(x)` 宏（`core_errors.hpp`）。展开与手写形态**逐字等价**（同样的错误消息、返回类型、语句数），不含控制流或所有权语义；实现用普通语句而非 `do{}while(0)`，以便在 `if/else` 里保持与原 `if` 语句一致的悬垂-else 行为。机械改写脚本 `tools/apply_nn_try.ps1`（带 `-DryRun` 计数，只改模式完全匹配的行）可复现。
- 验收：build 零告警（`-Werror`）；ctest 20/20（含 `NN_BIND_DEBUG=1` 一轮）；scan 双 hash 与 CPU/GPU 字节锚逐位不变；`--init-hash` 六模型与 M2 锚全同；`--io-roundtrip` CPU/GPU 全过；L2 审计 `L2-VIOLATIONS: 0`。

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

---

# 引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器

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
- 内容：`release_idle_pool_blocks` / `pool_stats` 注释曾写"CPU/CUDA 引擎"——CUDA 后端已整体移除（见 AGENTS §2），现仅 CPU（no-op/返回空）与 GPU。已改为当前引擎清单。（**M5 2026-09-30 后**：CPU 不再返回空串，改报宿主直配账本，见下方"统一总纲 M5"条。）

## Tensor reshape 头注释与实现不符（原位置 include/neuralnet.cpp/compute_tensor.hpp:291）
- 类型：错误注释（文档失真）
- 内容：头注释原称"零拷贝 reshape（共享底层 buffer）——CPU/Vulkan 均适用"，但 CPU 分支并无零拷贝视图能力、实际复制数据（与 AGENTS §10.6 记载一致）。整改后改为如实记录当前语义：GPU 共享 buffer（零拷贝）、CPU 复制到新形状 Matrix。

## ComputeEngine Refresh P-1：PrecisionEngine 装饰器删除与 NVI 下沉（原位置 include/neuralnet.cpp/compute_precision_engine.hpp 整文件（948 行，46 override）；compute_engine.hpp 类结构）
- 类型：演进记录 / 性能 A/B
- 内容：P-1（2026-09-29，方案见 docs/development/15 §4.1）把 f16 边界 cast 装饰器 `PrecisionEngine`（每次现问内层 `inner_` 要不要 cast、49 方法全量委托）下沉为 `ComputeEngine` 基类 NVI：公有非虚入口承载边界 cast 逻辑（变体优先 / 全 f32 快速直通 / 抬 f32→算→按 P 落回），protected 虚 `*_impl` 只管算；CPU 33 / GPU 29 处 override 机械改名 `*_impl`（基类 virtual 总数仍 49，仅公有虚变非虚 + 新增 `_impl`）；`text_train` / `mnist_train` / `mem_probe` / `f16_*` / `scan_exprs` 等 7 处使用方迁移（`optional<PrecisionEngine>` 舞步与 `dump_temp_stats` 静态调用点改指基类），`compute_precision_engine.hpp` 整文件删除。验收：dev1 ctest 20/20（含 f16_precision/gpu_f16/writeback 4 个 f16 用例）、CPU 稳定性探针与 P0 基线逐字节、scan 产物（expr_specs.bin / fused_registry.hpp）hash 不变。**性能 A/B**：layer_bench 单轮小算子 ±30% 摆动且方向不一（linear/layernorm +25% vs mha -7%）→ 交错 4 轮复测方向翻转（layernorm +25%→-7%）→ 判定手机 DVFS/调度噪声主导、未测得系统性回退；精确数字留待桌面平台复测。整改后代码/注释只保留"基类 NVI 边界 cast 入口"的当前结构，适配层叙事仅存于本文与 docs 15/05。

## ComputeEngine Refresh P1：张量出生绑定与跨引擎检查（原位置 compute_tensor.hpp 类结构、compute_engine.hpp 公共入口集、expr_dsl.hpp compute/compute_reduce 出口）
- 类型：演进记录 / 性能 A/B
- 内容：P1（2026-09-29，方案见 docs/development/15 §3.1/§4.2/§4.3/§7.2）按"只加检查、不删形参"落地：① `Tensor` 加 8B `observer_ptr<ComputeEngine> engine_`（friend ComputeEngine）+ `bound()`/`engine()`（未绑定 `NN_ASSERT`）；拷贝/`reshape` 传播绑定，`Tensor{}` 空槽与库外直构保持不绑定。② 工厂与单操作数虚入口 NVI 化——`create_tensor`/`from_matrix`/`cast`/`create_offload_buffer`/`offload_restore` 改为公共非虚包装（尾部 `stamp_`），CpuEngine/GpuEngine 各 3+5 处 override 改名 `*_impl`（基类 virtual 总数仍 49，与 P-1 口径一致）。③ 20 个返回 Tensor 的公共入口逐个包 `return stamp_(...)`（含 `to_prec` 出口统一行替换）：未绑定才补、已绑定沿用 src 绑定（传播规则）、无效空槽不 stamp。④ 跨引擎检查 `bind_check_` 按 D3 字面：操作数指针判等，**双方都 bound 且不同 → Result 硬错误**；单侧未绑定按库外豁免放行（P1 零行为变化，ctest 不红）；`NN_BIND_DEBUG=1` 时"未绑定输入进引擎"也报错（带 source_location 与张量形状）；插入约 28 个公共入口顶部。⑤ 库内 stamp 通道 `ComputeEngine::adopt(Tensor&&)`：friend 限定给 `nn::dsl::compute`/`compute_reduce`（compute_engine.hpp 前置声明 dsl 模板——**默认实参必须写在前置声明处，C++ 不允许在后续声明追加**），覆盖 `eval_cpu` 出口、扫描占位张量、`reduce_vector_tensor` 归约向量出口（15 §4.2 片段 A/B）。验收（本机 Debug+Ninja+clang）：ctest 20/20 两轮全绿；scan 产物 hash 不变（`expr_specs.bin`=bdc3a442…a58360 与 15 锚点一致、`fused_registry.hpp`=7a10412c…；后者含 SPIR-V 字节，仅本机 pre/post 可比）；`gpu_stability_probe` CPU 两轮输出与 P0 基线**逐字节**；**性能 A/B（stash 往返重建）**：layer_bench linear/layernorm/mha/transformer 全部 ±6% 内（linear 反而 -6%，其余 ±1%，噪声内无回退），`text_train` 9 步 244.7s（基线）vs 245.0s（P1）持平。`NN_BIND_DEBUG=1` 诊断：mnist MLP（CPU/GPU）/CNN/Transformer、text_train gpt/rapt、zipt_test 全部零漏网（库内产物出生即绑定成立）。顺带修复既有构建失败：`src/gpu_stability_probe.cpp` 的 `std::getenv` 在 MSVC CRT `-Wdeprecated-declarations -Werror` 下编译不过（照仓库 `_dupenv_s` 模式改写）。观察到的既有问题（基线同现、非 P1 引入，未修）：`gpu_stability_probe --gpu` 首个 `vkCreateBuffer` 即崩（16 §7-2 未决）；`text_train --model zipt` 启动即 `set_checkpoint_every(0)` 无条件 abort 且 abort 后挂死（compute_layer_zipt.hpp:850 未判 stride==0）。整改后叙事收敛为当前结构：绑定字段/`stamp_`/`bind_check_`/`adopt` 各就其位，实施过程仅存于本文与 15。

## 统一总纲 M1：Tensor 访问收口——存储私有化 + reshape 引擎化 + 库外全迁（2026-09-30，原位置 compute_tensor.hpp 公共面、compute_engine.hpp 新增 reshape、expr_dsl.hpp 求值出口、model_serialization.hpp save f16 分支、src 9 测试文件 + text_train.cpp:756 + tools/scan_exprs.cpp）
- 类型：演进记录 / 性能 A/B
- 内容：M1（方案见 docs/development/17 §4.1/§5 M1 行；commits `3d76478` 库外全迁、`c8f308f` 访问收口）分两段落地。① **私有化**：`Tensor` 的静态直构工厂（`from_matrix`×2/`cpu`×3/`cpu_uninitialized`/`from_gpu`×2）、存储构造、访问器（`cpu_matrix`×2/`cpu_shared`/`gpu_tensor`×2/`gpu_shared`）、`reshape` 全部转 private；friend 集 = `ComputeEngine/CpuEngine/GpuEngine` + 新增 `detail::TensorAccess`（前向声明于文件头、定义于尾部；`nn::dsl` 内 `using TensorAccess` 别名，eval_cpu 出口/扫描占位/求值叶子共 **19 处**改道）。公共面收敛为形状/精度/设备查询、`bound()/engine()`、`valid()`、`shape_str()`——绕过引擎 = 编译错误（铁律 #11 新立于 AGENTS §5）。② **`ComputeEngine::reshape`（D10）**：`bind_check_` + `stamp_` + 元素数不匹配返回 Result 错误（原 NN_ASSERT 升级）；`Tensor::reshape` 实现保留为私有成员供引擎调用（逻辑零搬移）。③ **库外全迁（D7，不留过渡期）**：`test_common.hpp` 新增 `upload/download` 助手（Result 失败 NN_ASSERT 兜底），`make_tensor`（48 调用点）/`check_close`（18 调用点）签名加 engine 首参；9 个测试文件读路径一律 download 值接收（防 `auto&` 悬垂）、写路径一律"本地 Matrix 写好 → upload 回传"（rng 消耗顺序逐位保持，断言/容差/打印原字节）；`fused_gpu_test` 双引擎对拍 32 个输入张量 cpu/gpu 各上传一份（出生绑定检查下跨引擎喂食 = 硬错误，逼出拆分）；`text_train.cpp:756` CPU/GPU 分支合一走 `engine.to_matrix`；`scan_exprs.cpp` 的 `scan_tensor` 收编为 dry_run lambda 内 `engine.create_tensor(rows,cols,g_scan_prof.compute)`（60 处调用点零改动，出生绑定与 dry-run 引擎一致）。④ **model_serialization save_model f16 分支**：CPU 直读 `cpu_matrix<F16>` 删除，CPU/GPU 统一经 `engine.to_matrix` 下载再转 f16 落盘（f16→f32→f16 往返精确，v5 字节不变）。⑤ `tensor_precision_test` 按引擎 I/O 重写：`cpu_shared` 别名断言随私有化取消（无公共等价物）、新增 reshape 元素数不匹配拒绝用例。**教训**：旧审计 pattern `cpu_matrix\s*\(` 漏掉**模板形态** `cpu_matrix<Precision::F16>()`——`model_serialization.hpp:550` 即此形态漏网，审计须用裸词 `\.cpu_matrix`。**验收**（本机 Debug+Ninja+clang）：build 122/122；ctest 20/20；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` 与 M1 前基线**逐位一致**；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；库外 grep 旧 API 零命中（编译期强制成立）。**性能 A/B（配对裁决）**：首轮 --all 单点对拍显 feedforward fwd/train +6~7%——但基线采集期并发了子代理语法编译且为单样本；按配对方法重建 pre-M1 对照二进制（git worktree @ HEAD~2 独立构建 layer_bench），feedforward/swiglu/linear 各 4 轮 pre/post 交错：**分布重叠、方向混合**（feedforward train 中位 3229.7ms vs 3228.3ms，swiglu train 均值 10.28 vs 10.46，linear 落在彼此散布内），无系统性回退；附带实测本机 Debug layer_bench 单点 run-to-run 摆动可达 ±25%（linear fwd 0.537~0.860ms），单点不可裁决——机制上 M1 只动冷路径与调用点层（`cpu_get_ptr` 热循环零变化）。四子代理并行迁移 + 主线统一构建的分工可行（子代理仅做 `-fsyntax-only` 与 %TEMP% 单 TU 运行，不碰 cmake）。

## 统一总纲 M2：声明式初始化——InitSpec + Layer::init 迁移 + RNG 收编（2026-09-30，原位置 compute_engine.hpp 张量工厂区、compute_layer_{mlp,conv,gpt,zipt,rapt,transformer}.hpp 的 init 函数、src/gpu_stability_probe.cpp）
- 类型：演进记录 / 性能 A/B
- 内容：M2（方案见 docs/development/17 §4.4/§5 M2 行）分三段落地（commits `8bd6bd5` 核心、`4492098` 层迁移、文档段随后）。① **`InitSpec` + 四参 `create_tensor`**（compute_engine.hpp）：`Kind{Uninitialized,Zero,Constant,Uniform,Normal}` + 工厂（分布类 **seed 必填** = U1 裁定，层传 `kInitSeed=42`）；四参 `create_tensor(rows,cols,P,spec)` 为公共非虚 NVI——Zero/Uninitialized 走既有 `create_tensor_impl`（Zero 另补 `zero()`），分布类 = 引擎内 host 生成后 `from_matrix` 上传（D6 策略调用方不可见）：`fill_host_` 用 mt19937_64，种子 = splitmix64(spec.seed × 引擎内创建序号 `init_seq_` 混流)——**同 seed 不同张量不撞流**（否则 GPT 同形状多层 Linear/注意力投影会互为镜像）；3 参 `create_tensor` 保持既有纯分配语义不动作（初值要什么显式走 InitSpec，零仍可显式 `zero`）。② **Layer::init 全迁**（6 头文件 25 处）：Linear/Conv2D 权重 `uniform(-limit,limit)`（层算 limit、引擎填数）、bias/梯度缓冲 `zero`、LayerNorm `gamma=1/beta=0`、RMSNorm `gamma=1`、GPT/ZiPT/RAPT token_emb `normal(0,0.02)`、LearnedPositionEncoder `normal(0,0.02)`（`AdditivePositionEncoder::init_` 入参 Matrix→Tensor，Sinusoidal 公式路径改走 `from_matrix`）、CrossAttention 记忆查询 P `uniform`、TransformerEncoder `ones_row_` `constant(1)`；mlp/conv 的 `thread_local rng_` 删除、init 侧 `random_device` 清零。**精度口径逐点保持**：conv/zipt/rapt 原 `from_matrix` 默认 F32 → create_tensor 显式 F32，gpt/norm 原 `p_.param` 照传——不借机改 profile。③ **裁定不迁（运行期随机性，17 §1 表已注）**：`generate()` 采样 RNG（gpt:954/zipt:1041/rapt:1461）、`text_train.cpp:1270` 数据洗牌 RNG——均非 init。④ **探针新增 `--init-hash` 模式**（gpu_stability_probe）：6 类模型（mnist_mlp/cnn/mnist_transformer/gpt/zipt/rapt）建后即哈希（不覆写），输出 `INIT1/INIT2` 固定前缀行供跨进程 diff；默认模式头注释更新——init 已确定性化后，覆写保留是为了**隔离 init 变量**、只测后端执行确定性。
- **计划内字节变化（17 §6 例外，新锚已立）**：原 random_device 播种站点与 seed-42 站点的 RNG 流（改引擎序号派生）字节全变。新锚（2026-09-30，两次独立进程逐字节）：`gpu_stability_probe --init-hash` 六模型 hash = mnist_mlp `a22e807ee05ec3ac` / cnn `4020958a14160bbd` / mnist_transformer `04a72d865624042a` / gpt `8efa936ac5c8c9b2` / zipt `f90bf8c8783c2f89` / rapt `b037632f75b7159c`；`mnist_train` 三架构（`--epochs 1 --max-samples 512 --batch-size 64 --shuffle-steps false`）双进程 loss 序列 + 模型文件逐字节一致（模型 SHA256：mlp `B596BB19…85A1`、cnn `19A29E0B…6FE7`、transformer `5D533039…2ADA`，仅墙钟行不同）。**反例定位**：`text_train` 双跑 loss 不等（9.0546 vs 9.0566）→ 根因 `text_train.cpp:1270` `mt19937_64{random_device{}()}` 数据洗牌——M2 前即存在、与 init 无关（init 已确定性后它成为 text_train 跨进程差异的唯一来源），范围外未修；要 text_train 全链路复现需另立 --seed 类开关。
- **验收**（本机 Debug+Ninja+clang）：build 122/122 零告警；ctest 20/20；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` **与 M1 锚点逐位一致**（该探针覆写 init，证明后端执行路径零变化）；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；`--init-hash` 两进程 `INIT1` 行全同 + 进程内两轮 PASS；`NN_BIND_DEBUG=1` mnist 冒烟零诊断。**性能 A/B（配对 + 逆序复测）**：pre 基线 = git worktree @ M1 HEAD（735594d）独立构建 layer_bench，同窗交错；feedforward 正序 4 轮 fwd/train POST 中位 +3.0%/+2.9%，**逆序 4 轮（POST 先跑）复测差异依旧**（+3.1%/+3.5%）→ 排除运行顺序热偏置，为 Debug(-O0) 二进制重排/内联漂移级小差（M2 未改动任何热路径语义，init 仅建模时执行一次）；swiglu fwd/train ±2% 分布重叠；linear 分布交叉重叠（已知单点 ±25% 方差，M1 记录在案）。全部 < ±6% 验收带，**无系统性回退**。

## 统一总纲 M3：批量读写 API——read/write/get_index/set_index 落地 I/O 分组（2026-09-30，原位置 compute_engine.hpp I/O 分组区、src 10 个 gradcheck/一致性测试文件、src/gpu_stability_probe.cpp）
- 类型：演进记录 / 附带缺陷修复
- 内容：M3（方案见 docs/development/17 §4.3 D3/D9、U2；commits `e0338d9` 核心、`4977a29` 探针、`3ddb13c` 测试迁移）分三段落地。① **`compute_engine.hpp` I/O 分组区**：`from_matrix/to_matrix/read/write/get_index/set_index` 六动词同组成文（D9：保留 `from_matrix/to_matrix` 原名，15 P2 的 `to_host/from_host` 改名随之取消）；新增 **基类非虚模板** `read(const Tensor&, span<T>)` / `write(Tensor&, span<T>)`（不进 49 virtual 骨架）——`read` 统一经 `to_matrix`（GPU 分支自带 `end_batch`→`wait_in_flight`→新帧，即"隐含 flush+同步"），`write` 宿主直写既有存储、GPU 走 `copy_from` 既有 drain 语义（F16 目标 = f32 上传 + `cast_into` 写回原存储），二者都**覆盖既有存储、不替换对象**（沿 `copy_into` 红线）；`get_index/set_index` 为语法糖（宿主直读写零往返，GPU = 整张批量往返，明确不给热循环承诺）。② **U2 精确匹配**：span 元素类型须与 `t.precision()` 严格一致（`float`↔F32、`nn::f16`↔F16）——元素类型非 float/f16 或 `read` 的 T 为 const = `static_assert` 编译期报错，精度错配/元素数不符/越界 = `Result` 运行期错误；`write` 形参用 `span<T>` 而非 `span<const T>`（模板实参推导中 `const T` 无法匹配非 const span，推导失败），静态断言按去 cv 后元素类型校验。`read` 对 f16 存储升 f32 后按 T 落回（RHE，往返位不变）。③ **调用约定成文**（17 §7-1/§7-4、铁律 #6）：read 隐含 flush、写入张量须存活到 `end_batch()` 之后；逐 step loss 回读继续走 `submit_scalar_readback` 快路。④ **测试迁移**（10 文件）：gradcheck 扰动-恢复、CPU→GPU 权重同步等 **f32 路径 `copy_from` → `write`**；f16 专项（`f16_cpu_probe`/`f16_precision_test`/`f16_writeback_probe`）**保留 `copy_from`**——其语义就是 f32 Matrix → f16 的转换填充（RHE），正是 U2 精确匹配下 `write` 不承担的转换入口，故两动词分工成文而非并吞。
- **附带缺陷修复（16 §7-2 根因，P0 遗留未决项收口）**：`gpu_stability_probe` GPU 模式首个 `vkCreateBuffer` 即 "Invalid device" 崩溃——根因 = `GpuBackend::instance()` 只是惰性单例，Vulkan 设备须显式 `backend.initialize()` 才建立，探针 `make_engine` 漏调（`cli_engine_factory` 与全部 GPU 测试入口都会调它，这正是 16 §7-2 当时记的"差异在探针路径"那一处），首个 buffer 拿到 VK_NULL_HANDLE 即崩。修复 = 探针建引擎前补 `initialize()`（空 selector 仍按 显式 > `NN_VULKAN_DEVICE` > 自动打分 选卡）。修复后 GPU 档位首次可测：Windows 本机 **dev2 `8ef51b2927253c50`、dev0 `47977edc71a21a20`、dev4 `01b7192bd15a59d9`** 进程内两轮 + 跨进程两次启动 loss/hash **逐字节一致** → D9 的 GPU 验收档位在本机坐实为**逐字节**；dev1（MESA/D3D12 转译）初始化即报 DXIL container 校验错误、exit=2，为设备级不可用（非探针/库问题，另记）。
- **探针新增 `--io-roundtrip` 模式**（M3 专属验收，GPU 模式即 staging 验收）：三用例 = `f32_roundtrip`（写后读回 + `read` vs `to_matrix` 交叉对拍）、`f16_roundtrip`（f16 精确匹配与 RHE 往返）、`batch_window`（铁律 #6：录制窗口内 write → read → `end_batch` → 复读值不变），另覆盖索引往返、越界、U2 错配错误路径；输出 `RESULT io_cases=3 io_failures=0 verdict=PASS`。
- **验收**（本机 Debug+Ninja+clang）：build 全绿零告警；ctest 20/20；`--io-roundtrip` CPU + GPU（dev2）三用例全 PASS；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` **与 M1/M2 锚点逐位一致**（I/O 组为纯新增、未动任何计算路径）；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）。

## 统一总纲 M4：Matrix 降级收口——L2+ 禁用 + span 宿主桥 + 分层审计（2026-09-30，原位置 compute_layer_{attention,base,conv,gpt,rapt,transformer,zipt}.hpp、compute_loss.hpp、compute_optimizer.hpp、compute_engine.hpp 尾部、bench/doc_inventory.ps1）
- 类型：演进记录 / 新增裁定（D11、U5）
- 内容：M4（方案见 docs/development/17 §4.6、§5 M4 行）分四段落地。
  ① **先立审计再迁移（编译器/grep 驱动）**：`bench/doc_inventory.ps1` 新增第 [4] 节——它本就枚举 L2 文件集（`compute_layer*.hpp` + `compute_loss.hpp` + `compute_optimizer.hpp` + `model_container.hpp`），逐行剥离 `//` 与块注释（行内状态机处理跨行注释）后匹配三类：`Matrix`/`MatrixT` 类型（**硬**）、`from_matrix/to_matrix/copy_from`（**硬**，签名即含 Matrix）、宿主桥动词（**披露**，只计数不判违规）；末行输出 `L2-VIOLATIONS: N`（验收口径 0）。**首跑违规模基线 = 77 条（32 处 `Matrix` 类型 + 45 处 Matrix 型 I/O 动词），分布在 10 个头文件**。
  ② **D11 裁定（新增到 17 §3，10 项 → 11 项）**：17 §4.3 原字面"from_matrix/to_matrix/read/write 只准出现在 I/O/测试代码"在 L2 上**不可执行**——层确实需要生成索引/位置/掩码类辅助数据，而引擎没有 arange / RoPE 表 / 斜率表这类生成原语（要严格执行得先加原语，超 M4 范围）。裁定把禁令本体收窄为**`Matrix` 类型**（数据格式统一的可审计判据）：Matrix 型三动词随之在 L2 全禁，span/标量级作为受控宿主桥保留（判据 = 不引入 Matrix 即不破坏统一），大批量 I/O 仍归 I/O 层。U5 同时裁定：审计落点 = 扩展 `doc_inventory.ps1`，不另建脚本。
  ③ **宿主桥三个自由函数**（`compute_engine.hpp` 尾部 `namespace detail`，0 虚表项、0 新原语）：`upload_span(engine, rows, cols, P, span<const Scalar>) -> Result<Tensor>`（= `from_matrix` 的 span 形态；f16 目标按 `f16{float}` RHE，与 CPU `from_matrix_impl` 的 `dst[i] = src[i]` 同一条舍入路径）、`download_span(engine, t, span<Scalar>)`（= `to_matrix(t, F32)`：f16 存储先升 f32 再落，升 cast 精确无损）、`download_vector(engine, t)`（取值便捷形态）。三者内部只调 M3 的批量 `write/read`（GPU staging、隐含 flush 照旧），不进 49 virtual 骨架。
  ④ **10 个头文件全量迁移（45 处 I/O + 32 处类型）**，逐类：RoPE `fill_pos_column_` 签名 `Matrix&`→`span<Scalar>`+cols（`compute_layer_attention.hpp`）；正弦/可学习位置索引与采样 1×1 词元 id（`compute_layer_gpt.hpp`，`pe` 公式数据改 `upload_span(..., p_.param)`——f16 param 转换与原 `from_matrix(pe, p_.param)` 逐位同）；ALiBi 斜率表/偏置、doc_ids 的 `(1,BH*seq)` 与 `(BH*seq,1)` 双布局（attention）；RAPT 的 `V_ones` 改 `InitSpec::constant(1)`、`e_0` 稀疏表、逐头 RMSNorm 的 1/rms 缓存（下载→行缓冲→上传）、文档边界、decode 步 `den = q'·z` 主机点积（`to_matrix`×2 → `download_vector`×2，算术顺序不变）；ViT patch 提取与梯度散射（`compute_layer_transformer.hpp`，`at_unchecked(row,col)` → 行主序 flat 下标，**布局等价**）；ZiPT 联合掩码/滑窗输入/logits 列读取；CE 稀疏路径 labels+mask 打包与三处 loss 标量下载；优化器两处 (1,1) 范数下载；`nn_dbg_scan` 诊断下载；conv 的置换索引与分组广播索引。零梯度返回张量（gpt/zipt/rapt 的 `grad_input`）改 `create_tensor(..., InitSpec::zero())`，省一次主机上传。
- **教训（审计工具）**：给 `compute_engine.hpp` 的**注释**里写"49 virtual 骨架"会让 `bench/doc_align_audit.ps1` 的 `virtual_in_engine` 实测数 50→51、与文档 49/50 口径打架——注释里用中文"虚函数"即可；数字型断言的复现命令只看裸 token，不看语境。
- **验收**（本机 Debug+Ninja+clang）：① **审计零违规**：`L2-VIOLATIONS: 0`，`host_bridge_uses: 41`（披露）；② build 122/122 零告警；③ ctest 20/20；④ **字节零变化**（迁移是纯载体替换：`Matrix(rows,cols)` 与 `std::vector(n)` 同为零填充、行主序 flat 下标 = `at_unchecked(r,c)`、f16 转换同一条 RHE 路径）——CPU `--steps 20` hash=`6f8849f14da23110`、GPU dev2 `8ef51b2927253c50` **均与迁移前逐位一致**；⑤ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）——`scan_exprs` 的 dry-run 走 Layer forward，表达式文本未动；⑥ **layer_bench 配对 A/B**（pre = `git worktree add` @ 48189d4 独立构建 layer_bench，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错，`--iter 4`）：feedforward fwd/train **-1.2%/-1.6%**、linear +1.4%/-4.2%、swiglu fwd -6.2%/train +0.2%——swiglu fwd 为 0.38–0.47ms 亚毫秒项、PRE 自身极差即 ±10%，全部落在 ±6% 噪声带且方向混合，**无系统性回退**；⑦ `doc_align_audit.ps1` A/D/E/F 可行动项全 0、`virtual_in_engine` 仍 50。
- **连带数字更新**：doc 12 的"Layer 直调 23 → **21**"（`from_matrix`/`to_matrix` 退出 L2；宿主桥是自由函数，不进 `engine.<op>(` 统计口径）与"其余 26 → 28"，§2.2 顺带修正 P-1 遗留的陈旧行 `cast_into/copy_into 仅 compute_precision_engine.hpp 内部`（该文件已删，实际调用方 = 基类 NVI 边界 cast + 引擎互调 + `write` 的 f16 目标）。AGENTS.md 新立**铁律 #12**（L2+ 禁用 Matrix + 宿主桥 + 审计口径），§3 审计表登记 `doc_inventory.ps1` 第 [4] 节。

## 统一总纲 M5：内存池契约统一——无池引擎 pool_stats 补齐（2026-09-30，原位置 algebra_matrix.hpp 顶部账本与 MatrixT 分配/释放路径、compute_engine.hpp 基类 pool_stats/release_idle_pool_blocks 注释、docs/development/04-memory-optimization.md）
- 类型：演进记录
- 内容：M5（方案见 17 §4.5 / §5 M5 行）按"统一的是**接口与语义**、不是实现（D5）"落地，两段：
  ① **宿主直配账本 `nn::HostAllocLedger`**（`algebra_matrix.hpp` 顶部，L1）：`live_blocks/live_bytes/peak_bytes/total_blocks/total_bytes` 五个 `std::atomic<std::uint64_t>`，全程 `memory_order_relaxed`，**只在分配/释放路径记账，元素热循环零参与**；峰值用 load + CAS 循环（只在变大时写）。全进程唯一实例由 `inline` 函数内静态对象提供（header-only，跨翻译单元同一地址）。
  ② **`MatrixT` 三处释放出口收口**：分配在 `allocate_()` 成功后 `on_alloc(n * sizeof(element))`；释放统一走新增私有 `release_storage_()`（记账 + `data_.reset()` + `size_=0`），**析构 / 移动赋值 / `resize()`** 三处调用——移动赋值与 `resize` 原本是"直接覆盖 `data_`"的隐式释放，若只挂析构会漏记（账本越走越偏）。移动构造把源 `size_` 置 0（原实现已有），不重复记账。
  ③ **`ComputeEngine::pool_stats()` 基类默认实现**：由返回空串改为返回 `direct{ blocks=<活分配数> live_bytes=<> peak_bytes=<> total_blocks=<> total_bytes=<> }`（单位字节，key 显式带 `_bytes` 避免与 GPU 的 `total=..MB` 混淆）；`GpuEngine` 的 `persist{…} transient{…} pending=…` override 不动。`release_idle_pool_blocks()` 保持 no-op，但把理由（"直配无整块可归还，语义成立"）写进 `compute_engine.hpp` 注释与 `04-memory-optimization.md`——M4 之前那句"CPU 为 no-op/返回空"的口径一并更正（history 的同名旧条目已加后注）。
- **未做（保持 U3）**：CPU 真内存池是独立优化立项，需先分配剖析量化 DSL 临时张量热点；本期只补契约不改分配策略，`Matrix` 仍是 `std::unique_ptr<element[]>` 直配。
- **验收**（本机 Debug+Ninja+clang）：① **两引擎 `pool_stats` 非空且口径已文档化**（17 §4.5 + `04-memory-optimization.md` + `01-compute-engine-development.md`）——`NN_MEM_STATS=1 text_train` 实测抓取：CPU `engine-init` 全 0 → `model-built` `direct{ blocks=44 live_bytes=9168512 peak_bytes=9168512 total_blocks=56 … }` → `optimizer-created` `blocks=88 live_bytes=18337024 …`（随生命周期单调反映），GPU dev2 `persist{blocks=5 allocs=12 total=1028MB …} transient{…} pending=0MB` 照旧；② build 122/122 零告警；③ ctest 20/20；④ CPU `--steps 20` hash=`6f8849f14da23110` **与 M1–M4 锚点逐位一致**（账本只读计数，不进任何数值路径）；⑤ scan 双 hash 不变；⑥ `--io-roundtrip` CPU + GPU(dev2) 三用例全 PASS；⑦ L2 审计仍 `L2-VIOLATIONS: 0`；⑧ **layer_bench 配对 A/B**（pre = `git worktree add` @ d28c899 独立构建，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错 `--iter 4`）：feedforward fwd/train **+0.3%/+1.1%**（分配最密集的算例，说明 relaxed 原子开销不可测）、swiglu fwd -5.7%/train +0.1%、linear fwd -3.0%/train **-13.0%**（POST 更快；linear 单点方差 M1/M2 已记录在案 ±25%，且两组分布交叉），**方向混合、无系统性回退**。

## 统一总纲 M6 段 A/B：15 D6/D7/D8 落地 + NN_BIND_DEBUG 门禁 + ensure_gpu→import（2026-09-30，原位置 model_serialization.hpp v5 加载分支、model_container.hpp set_engine、compute_engine.hpp（matmul_with_bias_impl 默认体 / to_prec / bind_error_ / 新增 import 与 import_impl）、expr_dsl.hpp 归约向量 F16 分支、compute_gpu_engine.hpp ensure_gpu）
- 类型：演进记录 / 新增引擎契约 / 门禁化
- 内容：17 §5 M6 分三段，本轮交付 **A、B**；**C（删每调用 engine 形参）未实施**，交接写在 17 §8。
  ① **段 A = 15 D6/D7/D8 裁定落地 + 门禁**（commit `8ecf1bd`）：
  - **D6-1（真缺陷，修）**：`load_model` v5 两支都白做——同精度支 `from_matrix(file_matrix, target_p)` 上传后**丢弃结果**再 `copy_from`（每参数一次全量上传白做），跨精度支"上传 → `cast` → `to_matrix` 下载 → 再 `copy_from`"（多两个 PCIe 往返）。因为 `file_matrix` 恒为 f32 宿主值（f16 文件数据读入时已精确升 f32，f16→f32→f16 位不变），四种 `file_prec × target` 组合与原路径**逐位一致**，收敛为 `engine.copy_from(p_tensor, file_matrix)` 一行。正确性实证：`mnist_train --save A` → `--resume A --lr 0 --save B` → **A/B SHA256 完全相同**（`7ad4c468…` f32 模型、`0e6c26b2…` f16-param 模型两条）。
  - **D6-2（复核为已修）**：15 §4.6-2 说 `CpuEngine::copy_from` 硬编码 F32 槽、f16 dst 在 NDEBUG 下解引用空指针——实际基类 `copy_from` 入口已拦 F16 目标（`from_matrix(src, F32)` + `cast_into`），`copy_from_impl` 的 `cpu_matrix()`（默认 P=F32）只在 F32 目标下可达。**裁定：现状即修复，无代码改动**，把该前提写进注释。
  - **D6-3（删死码）**：`matmul_with_bias_impl` 的默认实现是 `to_matrix → 主机逐列加 bias → from_matrix` 的 PCIe 兜底，CpuEngine/GpuEngine **均已 override**（全仓仅两个 `ComputeEngine` 子类）→ 改为 `= 0` 纯虚：删死码，且第三后端必须给出真现值（编译期强制），不再可能静默退化成主机往返。
  - **D7**：`Model::set_engine` 零调用方 + "绑定后换引擎"与"层已 init 绑定"冲突 → 删除；绑定只经 `Model(ComputeEngine&)`。
  - **D8**：基类 `create_offload_buffer_impl` / `offload_restore_impl` 默认体的 `Tensor::cpu(1,1)`（静态直构）→ `create_tensor(1,1)`（出生经引擎入口，P1 一致）。
  - **`NN_BIND_DEBUG=1` ctest 门禁化**（17 §5 M6 验收项）：首跑 **18/20**（`f16_precision_test` 13 处 FAIL、`f16_writeback_probe` 1 处）。**先补诊断**——`bind_error_` 原先把 file:line+形状只塞进 `Result`，测试的 `CHECK` 不打印 → 门禁失败无从定位；改为 NN_BIND_DEBUG 下同步 `fprintf(stderr, "[NN_BIND_DEBUG] …")`。定位出**两处库内中转漏网**：(a) `to_prec(t, P)` 的入参常是 `*_impl` 产物（`unary_/binary_/move_` 的 `fn(*a)` 返回值尚未走出调用方 `stamp_`），进 `cast`（内部有 `bind_check_`）前未绑定 → `to_prec` 内先 `t = stamp_(std::move(t))`；(b) `expr_dsl.hpp` 归约向量出口 `reduce_vector_tensor` 的 **F16 分支直接 `eng.cast(v, P)`**（F32 分支本就 `eng.adopt`）→ 改 `eng.cast(eng.adopt(std::move(v)), P)`。修完 **ctest 20/20（带与不带开关各一轮全绿）**，门禁成立。
  ② **段 B = `ensure_gpu` → `import`（15 P3 主体，commit 待本文档段）**：新增**引擎契约** `ComputeEngine::import(src[, P])`（公共 NVI：`bind_check_` + `stamp_`，无默认形参的单参重载委托到双参版）+ **新虚 `import_impl`（virtual 方法 48 → 49，`doc_inventory` 口径；`grep -c "\bvirtual\b"` 原始 51 处 = 49 方法 + 析构 1 + 注释里的 "Non-Virtual" 1——本轮唯一虚表变更）**。基类默认实现按 15 §3.2：同设备同精度 = 零拷贝别名、同设备异精度 = `cast`、跨设备 = `to_matrix`+`from_matrix` 宿主中转；`GpuEngine::import_impl` = 原 `ensure_gpu` 的宿主直传快路径（`GpuTensor::from_matrix(src.cpu_matrix(), backend_)`，f32→f32 零额外宿主拷贝），f16 源改走宿主中转分支——**顺带修掉 `ensure_gpu` 硬取 `cpu_matrix()`（F32 槽）对 f16 源取空指针的隐患**。调用点：`compute_gpu_engine.hpp` 全文 `ensure_gpu(` → `import(`（**48 处命中清零，符号 `ensure_gpu` 从代码中消失**），注释同步；`import` **返回新句柄、不改写 `src`**，B 类 6 处"调用方句柄重绑定"保持原形态但显式（`if (A.is_cpu()) A = std::move(*a_gpu);` 语义不变）。P3 第三项"序列化去 `model.engine()`"**裁定保留**：M1 后 save/load 的引擎来源只有"调用方显式传"与"模型回查"两条，改签名会破坏 `save_model(model, path)` 公开 API，收益为零（记入 15 §5 P3 行）。
- **连带数字更新（"49 virtual" 口径统一到 `doc_inventory` 脚本）**：AGENTS §4.3 注、doc 12（头行/§2 标题/§2.2 其余 28 + `import` 行/§2.15）、13 §10 表、17 §4.3 两条 bullet 与 import 行、`introduction/01` §L2 表 + 注、`introduction/03` 注、15 §5 P2/P3 行、16 §2 加"已迁完"横幅与失效的复现命令说明。**口径勘误（本轮发现）**：历史文档的"49 个 virtual / 共 50 处"实际把注释里的 `Non-Virtual` 也算进去了（真实方法数当时是 48）——现统一为 **virtual 方法 = `doc_inventory` 输出（现 49）**，并把三口径写明：`grep -c "\bvirtual\b"` = 51（49 方法 + 析构 1 + 注释词 1）、`doc_inventory` = 49、`doc_align_audit` 的 `virtual_in_engine` = 51。
- **验收**（本机 Debug+Ninja+clang）：① build 122/122 零告警；② ctest 20/20；③ **`NN_BIND_DEBUG=1 ctest` 20/20（门禁）**；④ CPU `--steps 20` hash=`6f8849f14da23110` 不变；⑤ **GPU dev2 hash=`8ef51b2927253c50` 不变**（import 快路径与 ensure_gpu 同一条，GPU 数值零变化）；⑥ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；⑦ `--io-roundtrip` CPU + GPU(dev2) 三用例 PASS；⑧ L2 分层审计 `L2-VIOLATIONS: 0`；⑨ 序列化 round-trip SHA256 逐字节一致（f32/f16 param 两模型）；⑩ `doc_align_audit` A/D/E/F 可行动项 0。**未做 layer_bench A/B**：段 A/B 均为控制流/API 变更（无热路径语义改动），字节锚与 GPU 锚已逐位不变——如需补测，按 §6 口径配对复跑。

## 统一总纲 M6 段 C：删 Layer 每调用 engine 形参——init 绑定 + 四虚接口瘦身（2026-09-30，原位置 compute_layer_base.hpp 类结构、10 个 compute_layer_*.hpp、model_container.hpp、tools/scan_exprs.cpp、src 25 个测试/探针文件）
- 类型：演进记录 / API 变更
- 内容：15 P2 的主体（17 §5 M6 段 C），commit `402b213`。
  ① **绑定设计**：`Layer` 新增 `ComputeEngine* engine_`；`init` 从虚函数改成 **NVI**——公共 `init(engine)` 先绑定 `engine_` 再转 `init_impl(engine)`，10 个层头里 **17 处 `init(...) override` 全部改名 `init_impl`**（编译器穷尽驱动；`PositionEncoder` 家族的 `init` 是**另一套虚表**，刻意不改——它不是 Layer）。四个虚接口去形参：`forward/backward/zero_grad/forward_recompute`。
  ② **方法体零改写技巧**：每个改签名的方法在函数首行插 `ComputeEngine& engine = engine_ref();`（**54 处**）——方法体里原有的 `dsl::compute(engine, …)`、`engine.slice_rows(…)` 全部不动；DSL 入口按 15 §3.3 保留形参。**自动化过程与两处失误**：脚本按"签名行到首个 `{`"切区域做替换，(a) 对 `= 0;` 纯虚声明会一路扫到下一个函数的 `{`，在错误位置插了别名（`PositionEncoder::backward` 三处，手工回滚并恢复其 engine 形参与调用点）；(b) 只匹配 `.` 前缀调用，漏掉 `forward(engine, saved_input)` 这类裸调用与 `*engine_` 形参的 `Model` 调用点（编译器报 "too many arguments" 后逐个修）。
  ③ **调用点**：include 内 99 处 + src/tools 约 106 处（`engine/eng/cpu/gpu/ad/raw/*engine` 等形参名按白名单替换）；**Loss 的 forward 被误伤 7 处**（`ce.forward(engine, logits, target)` 是 3 参），按调用方变量名回滚。
  ④ **新不变量（本轮最有价值的发现）**：**复合层 `init_impl` 必须 init 全部子层**——包括 ReLU/GeLU/SwiGLU/Softmax 这类"无参数"子层，因为它们以前靠 forward 形参拿引擎。补齐 5 处：`FeedForward::gelu_/swiglu_`、`AttentionBase::softmax_`、`CrossAttention::softmax_`、`ZiPTBlock::softmax_`、`TransformerEncoder::pos_encoding_`。定位靠的是把 `engine_ref()` 的 NN_ASSERT 换成 **`std::source_location` 默认参 + `fprintf` 打调用点 file:line**（裸断言只报断言行，无法反查是哪个层），该诊断保留为长期工具。
  ⑤ **库外配套**：`tools/scan_exprs` 的 dry-run 是"构造层→直接 forward"（以前不需要 init）→ 9 处补 `init(engine)`；`fused_gpu_test`/`softmax_gradcheck`/`maxpool_gradcheck` 共 6 处同理。
- **保留形参（＝15 §3.3 "Layer 之后"的两档，本轮裁定不动，见 17 §8）**：`Loss::forward/forward_sparse`（**Loss 没有天然绑定时机**：默认构造、`zipt_consistency_test` 里同一作用域并存 `ce_cpu/ce_gpu` 两个引擎——要删须先裁定构造签名 `CrossEntropyLoss(engine)` 或 `Loss::bind`）；DSL 入口（15 明确保留）；`PositionEncoder`/`RotaryEmbedding`（`apply/apply_step/backward`）、`ActivationOffloader`、`nn_dbg_scan` 与层内私有 helper（由方法体的局部 `engine` 转传，签名不动）。
- **验收**（本机 Debug+Ninja+clang）：① build 122/122 零告警（`-Werror`）；② ctest 20/20；③ **`NN_BIND_DEBUG=1 ctest` 20/20（门禁）**；④ CPU `--steps 20` hash=`6f8849f14da23110` **逐位不变**；⑤ **GPU dev2 hash=`8ef51b2927253c50` 逐位不变**（`engine_ref()` 只是绑定读取，不进数值路径）；⑥ `--init-hash` 六模型与 M2 锚全同（`a22e807ee05ec3ac`/`4020958a14160bbd`/`04a72d865624042a`/`8efa936ac5c8c9b2`/`f90bf8c8783c2f89`/`b037632f75b7159c`）；⑦ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…——**dry-run 收集的是表达式结构，与引擎绑定无关**）；⑧ `--io-roundtrip` CPU+GPU PASS；⑨ L2 分层审计 0。
- **性能 A/B（配对 + 逆序复测，M2 同款方法）**：pre = `git worktree add` @ `4ce5884` 独立构建 layer_bench，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错：feedforward fwd/train **+4.1%/+4.5%**、swiglu -0.7%/+0.9%、linear fwd **+8.5%**/train +3.8%（linear fwd 为 0.8ms 亚毫秒项）。**逆序复测**（POST 先跑，3 轮）：feedforward **-0.7%/+0.4%**、linear +5.6%/-6.9% → 正序的 +4% 系**运行窗口/机器状态**（同窗 PRE 绝对值本身较上一轮高约 18%）与 -O0 二进制重排所致，**方向混合、无系统性回退**。

---

# 代数层 / 基础设施 / 模型容器与序列化

## 线程池删除通用单任务 submit（原位置 include/neuralnet.cpp/core_threadpool.hpp:77）
- 类型：删除清单
- 内容：通用单任务提交（submit）已删除：全库无调用方（审查 P1-3），且每次调用 make_shared<packaged_task> 堆分配，违背本池"零分配 latch"设计。需要 future 语义时应在调用方分块后用 parallel_* 系列原语。

## 线程池 wait_for_latch 等待策略演进（原位置 include/neuralnet.cpp/core_threadpool.hpp:118）
- 类型：性能 A/B
- 内容：优化依据为性能审查报告。旧实现 spin 64 次 + yield，64 次 spin 中反复原子读取，CPU 占用率显示 100% 但实际有效计算比例低（调用者空转）。新实现为三阶段：1) 短自旋 16 次 pause；2) 自旋失败后 work-steal 从队列取任务执行；3) 队列为空阻塞 cv 等待，由 finish_chunk 唤醒。

## 线程池归约分段依赖 worker 数导致字节不一致（原位置 include/neuralnet.cpp/core_threadpool.hpp:541）
- 类型：bug 根因
- 内容：旧实现 n_chunks = chunk_count(total) 依赖 workers_.size()，同一输入在 1-worker 与 N-worker 下走不同折叠结构 → 浮点非结合律导致字节不一致（跨机也不一致）。修复后边界只由 total 决定，1-worker/N-worker、任何机器走完全相同分段。

## 归约 init 重复累加（原位置 include/neuralnet.cpp/core_threadpool.hpp:555）
- 类型：bug 根因
- 内容：旧实现每块都加 init、合并时再加一次，init≠0 时数学错误。现行约定：块 0 以 init 为种子、块 c>0 以块内首元素为种子，init 恰好计入一次。

## 线程池其它阈值/引用调整（原位置 include/neuralnet.cpp/core_threadpool.hpp:85、:106）
- 类型：删除清单
- 内容：MIN_CHUNK 阈值曾从 4096 降至 1024（使 MNIST 小隐藏层 64×batch 也能触发多核并行）；lost-wakeup 压测引用的 build/perfprobe/probe_pool2.cpp 已不存在（压测数据本身保留为持锁通知的实证）。

## GEMM 微内核 vs 旧标量内核 A/B（原位置 include/neuralnet.cpp/algebra_matrix.hpp:278-288）
- 类型：性能 A/B
- 内容：旧内核每个 (i,j) 一个标量累加器、每次 FMA 2 次 load（受 load port 限制），编译日志显示内层循环未被向量化（-Wpass-failed=transform-warning）。微内核（4×8 AVX2 列块）实测 GFLOPS（3072x768x512 / 768x3072x512 / 1024^3 / 512^3）：140→401 / 155→407 / 163→354 / 82→281；与旧内核逐位一致（对拍 max_abs_diff = 0）。

## row_reduce 并行门控曾恒定串行（原位置 include/neuralnet.cpp/algebra_matrix.hpp:744）
- 类型：bug 根因
- 内容：旧实现用 nn::for_each(row_indices)，把"行数"当元素数与 PARALLEL_THRESHOLD 比较 → 行数永远达不到 512K → 恒定串行（原 docs/development/11 §R3，该诊断文档已删除，结论并入 12-compute-engine-inventory/03 §5.3）。改为按元素数（R*C）门控 + parallel_for_samples 按行分片；col_reduce 行数门槛由 1024 降至 256。

## MatrixT F32 零回归验证（原位置 include/neuralnet.cpp/algebra_matrix.hpp:44）
- 类型：性能 A/B
- 内容：模板化改造时 F32 实例与旧非模板 Matrix 逐字节一致（测试项 T1 零回归）；f32 便捷别名保持 Matrix 以保证既有代码零改动。

## 旧代数 AST 移除（原位置 include/neuralnet.cpp/algebra_span.hpp:8、nn.hpp:8）
- 类型：删除清单
- 内容：旧代数 AST（自由运算符 + compute::apply 入口）已随逐元素算子移除；algebra_expr.hpp / algebra_compute.hpp 删除，Expression/BoolExpression 概念迁入 expr_dsl.hpp。Span 不再有"运算符构建 AST"的旧路径；ConstSpan 原注释引用的 expr.hpp 自由函数模板（operator+/-/* 等基于 Expression 概念统一处理，使 ConstSpan 与 Span/Val 自然组合）已不存在，现 DSL 运算符只接受 DslExpr 可折叠节点。

## 序列化审查待办 S3/M1/M2（原位置 include/neuralnet.cpp/model_serialization.hpp:54）
- 类型：bug 根因 / 删除清单
- 内容：S3（已修复）：read_spec_header / read_tokenizer 曾无长度上限，现均经 kMaxSerializedStringBytes（64 MiB）校验后才预分配，损坏/恶意文件返回 Error 而非 bad_alloc → terminate。M1（未修）：.bin 全文件无校验和（.nnpkg 有 sha256），内容损坏会被静默载入错误权重且无感知，曾建议升 MODEL_VERSION 加整文件校验和与尾部完整性标记。M2（审查记录）：extra_state 的注释称"旧文件读到 EOF 保持默认（running_mean=0 等）"但实现直接返回错误，注释与实现不符，需统一为按版本回退默认值。

## v1/v2/v3 偏移量格式移除支持（原位置 include/neuralnet.cpp/model_serialization.hpp:46、:430）
- 类型：删除清单
- 内容：v1/v2/v3 为旧的偏移量定长格式，已移除支持（无有意义的旧模型）；现仅接受自描述格式 v4+，旧格式在 read_and_validate_header 拒绝。Tokenizer JSON 读写曾标注"V3 新增"（版本流水，随 v1-v3 拒绝一并失去意义）。

## 失效文档链接清理（原位置 core_file.hpp:13/:42、model_serialization.hpp:24/:82、algebra_matrix.hpp:41/:43/:746、core_config.hpp:47）
- 类型：删除清单
- 内容：docs/17-pointer-audit.md、docs/23-mixed-precision.md、docs/development/11-cpu-performance-diagnosis.md、DEVELOPMENT_STANDARDS.md、src/bench_thresholds.cpp 均已删除或改名；对应链接改指 docs/development/10-development-standards.md、docs/development/05-mixed-precision.md，或去掉文件名只留实测结论（blocked 优于 naive、并行阈值 524288 标定表等实测数据保留）。

---

# 领域模型工厂与 CLI 公共头

## BPE encode 每 chunk 的 4 次堆分配（原位置 include/neuralnet.cpp/domain_tokenizer_base.hpp bpe_merge_impl_）
- 类型：性能 A/B
- 内容：旧实现每处理一个 chunk 分配 4 次（`ll_prev` / `ll_next` / `std::vector<bool> alive` / `std::priority_queue` 内部 vector），输入 `ids` 与返回结果各再分配一次。现改为 thread_local `BpeMergeScratch` 复用容量、结果**追加**到调用方 `out`、堆用 `std::push_heap/pop_heap` 手工维护（与 `std::priority_queue` 同算法，弹出顺序逐位相同）。实测 34MB 语料 `tokenizer_infer --encode-file` 单线程 18.4s → 9.2s（本机 32 核，2026-10-01；旧二进制基线取自本会话）。输出逐字节不变：`text_train` 的 `.tokcache` 重新生成后 SHA256 = `9E1F6DD4CC6B00F057BB04654E62E6CD4665410ED76E2BAEF578639341CC326E`（与旧实现产出的缓存完全相同）。

## BPE encode 保序并行（原位置 include/neuralnet.cpp/domain_tokenizer_{base,bpe,charbpe}.hpp 的 encode）
- 类型：演进记录 / 性能 A/B
- 内容：`encode` 从「单线程顺序处理整段文本」改为 `encode_segments_`：按空白串起点（`find_safe_splits`）分段 → 各段经全局线程池并发编码 → **按段下标升序拼接**。正确性依据（已作为当前契约写进代码注释）：安全切分点对 BPE/CharBPE 的预分词**必然是 chunk 边界**、标记不含空白故不会被切断、预分词与合并都是 chunk 局部操作，因此任意并行度与顺序执行逐字节一致（铁律 #8）。文本 < 256 KiB（`PARALLEL_ENCODE_MIN_BYTES`）或找不到切分点时回退顺序路径；新增 `Tokenizer::set_encode_threads`（0=自动/1=顺序/>1=指定）。实测 8 MiB 多样文本 `encode` 0.93s → 0.134s（32 线程，7.0x；本机 32 核，2026-10-01）。此前「BPE 不能并行」的结论只适用于训练期合并循环（见 `docs/development/08` §4.2）。

## evaluate_mnist 原始签名否决（cli/cli_mnist_io.hpp:16-18 头注释与 evaluate_mnist 函数注释，整改前行号）
- 类型：否决方案
- 内容：曾按「用户原始签名 evaluate_mnist(nn::Layer&, ...)」设计，实际不可行——nn::Model 不是 nn::Layer 的派生类，且 Model::forward 自带 engine 绑定（签名不同于 Layer::forward(engine, ...)）；最终改用 nn::Model&，原注释还附带「与原签名行为完全等价」的等价性论证。技术理由已改写为当前注释（签名为何取 nn::Model&），原始签名叙事与等价性论证移至本条。

---

# src 应用入口（text/mnist/tokenizer/bench）

## base 模式 BOS/EOS 插入的删除（原位置 src/text_train.cpp:4-5 头注释、src/text_train.cpp:98、src/text_train.cpp:808，整改前行号）
- 类型：删除清单
- 内容：早期 text_train 把每行编码为 `[BOS]+tokens+[EOS]` 再拼接，行边界靠 EOS 编码进 token 流；base 模式改为纯拼接、行间无分隔符后，文档边界改由 parallel_tokenize 产出的 doc_ids（块对角掩码）表达。随迁的等价性叙事：text_infer 曾以「训练时每行以 BOS 开头 / 与训练时每行格式一致」论证 prompt 加 BOS 的合理性（src/text_infer.cpp:171、:193、:362 整改前），该前提在纯拼接方案下不成立（`scripts/download_everyday_conversations.py` 生成的对话行也不含 BOS）。现注释只陈述两侧各自的现行行为。

## host 取 loss 的同步旧路径（原位置 src/text_train.cpp:1339，整改前行号）
- 类型：性能 A/B
- 内容：旧路径每步为取 loss 做 `end_batch + wait_in_flight`，drain 整条流水线 → GPU 在 host 录制期间空转，表现为占用率锯齿。现改为「异步回读槽位 + 非阻塞收割 harvest_loss + 槽位将满/epoch 收尾时阻塞兜底 drain_loss」；注释保留现行机制与「为何 host 不打断录制帧」的理由，删除旧路径描述。

## NaN 跳步的移除（原位置 src/text_train.cpp:1509，整改前行号）
- 类型：删除清单
- 内容：训练循环曾在 host 侧读回 loss 判定 NaN 并跳过该步；现 host 不判定 loss（loss 只在 device 侧异步回读），数值稳定性由 `--max-norm` 梯度裁剪 + 观察 loss 曲线负责。

## layer_bench transB/transA 变体与 plain 共用操作数（原位置 src/layer_bench.cpp:263-264，整改前行号）
- 类型：bug 根因
- 内容：旧版 setup 让 transB/transA 变体与 plain 共用 `(k,n)` 操作数：`k==n` 时方阵形状掩盖错配，`k≠n` 时 backend 返回 K mismatch 而 run 内 `*expected` 不查错 → UB，测出 0.000ms 垃圾时长 / 超物理 GFLOPS。已改为 per-variant 独立 setup + run 查 Result 失败即打印并中止；注释保留现行契约（每变体自建操作数、run 必须查错）与形状错配的坑。

## text_train::parallel_tokenize 换用全局线程池（原位置 src/text_train.cpp parallel_tokenize）
- 类型：演进记录
- 内容：旧实现用 `std::async(std::launch::async)` 每块起一个 future、按块顺序 `get()` 后拼接（保序但每次调用有 future 堆分配与共享状态）。现改为按块下标写 `parts[c]` + `nn::parallel_for_samples` + 顺序拼接；块边界仍是 `(n + n_threads - 1) / n_threads`，token 流与 doc_ids **逐字节不变**（`.tokcache` 重新生成 SHA256 一致）。同批 `tokenizer_infer` 新增 `--threads`（单次 encode 并行度）。

---

# GPU 手写原语 shader（shaders/*.comp）

## matmul_tiled vec4 优化策略的旧版对比（原位置 shaders/matmul_tiled.comp:10-14，整改前行号）
- 类型：性能 A/B / 差异对比
- 内容：原注释标题为「优化策略（对比旧标量 4×4 版：内层每 k 8 次标量 LDS + 16 FMA，2:1）」，并在正文写「消除旧版标量读的 bank 冲突」——即当前 vec4 配方的 8:1 计算:加载比是相对「旧标量 4×4 版每 k 8 次标量 LDS + 16 FMA（2:1）」表述的。整改后注释只陈述当前算法（vec4 LDS + 16 FMA、8:1、vec4 读沿 lane 连续消除 bank 冲突）。

## matmul_tiled v2/v3 版本标签与 barrier 演进叙事（原位置 shaders/matmul_tiled.comp:16-23、256）
- 类型：演进叙事
- 内容：原注释以「v2（双缓冲流水，BK=16）」「v3（vec4 全局加载/写出…）」标版本，v2 段写「每 tile 的 barrier 由 2 次减为 1 次」「BK=32 双缓冲需 32KB，会砍半 blocks/SM 抵消流水收益（实测净 0）」；流水循环内写「每 tile 仅 1 次 barrier（旧版 2 次）」。整改后改为现在式小节标题（双缓冲流水 / vec4 全局加载写出）与「每 tile 仅 1 次 barrier」。

## matmul_tiled BK=32 否决方案 A/B（原位置 shaders/matmul_tiled.comp:85-89）
- 类型：否决方案 / 性能 A/B
- 内容：原文「BK=16 + 双缓冲（2026-09-24 OP/融合统一 A/B 定案）：…曾试与融合侧同参 BK=32 (32KB)：OP 级四点**全回退**（512³ +9%、1024³ +26%、4096³ +29%、batched b8 +29%，3 样本 best）——32KB 每 SM 只驻 2 个 WG，浅深网格都被占用率卡死；OP 级恒取 16（下轮再统一须四点复测）」。四点实测数字与占用率结论已保留在现注释，删除的只是日期与「曾试与融合侧同参」的对比流水。

## batched_matmul 旧标量版与移植叙事（原位置 shaders/batched_matmul.comp:15-25）
- 类型：演进叙事 / 性能 A/B
- 内容：原文「与 matmul_tiled.comp 同配方（v3 移植，2026-09-24）：本 shader 长期停留在最初的标量 4×4 版（每 k 8 次标量 LDS + 每 tile 2 次 barrier + 标量全局访问带逐元素 /,% 与边界分支），而 matmul_tiled 已演进到 v3 —— 同形状下 batched 比 matmul 慢 1.85×（1024³ 实测 1.51 vs 0.81ms）。移植的三重配方（逐项与旧版的差异）：① vec4 共享（8:1，旧版标量 2:1）② BK=16 双缓冲（barrier 2→1 次）③ vec4 全局快路径（旧版每元素 1 次标量 ldg/stg + /,% + 边界判断）」。三条配方本身保留为现注释，删去旧版描述、1.85× 对比数据与移植流水。

## batched_matmul BK 注释的日期决策流水（原位置 shaders/batched_matmul.comp:93-94）
- 类型：演进叙事
- 内容：原文「分块维度常量（与 matmul_tiled 保持同步；2026-09-24 OP/融合统一 A/B：OP 级 BK=32 四点全回退 → 定案 BK=16 双缓冲 16KB）」。整改后只留当前事实（与 matmul_tiled 同步、BK=16 双缓冲 16KB 保占用率），数据指向 matmul_tiled 的 BK 常量注释。

## reduce 差异分解整段（原位置 shaders/reduce.comp:39-55）
- 类型：差异对比 / 性能 A/B
- 内容：原注释标题「2026-09-24：与 DSL 生成归约（expr_glsl_gen generate_glsl_reduce）同步」，下接「差异分解（此前两条路算法不一致，OP 级是慢的那条）」：① 旧版列归约「WG=单列、256 线程按行号跨步」——lane 间地址步进 cols×4B 完全不合并（每 warp 32 笔独立事务），4096² 实测 159 GB/s（同形状 row_reduce 482、add 412）；② 旧版 max 恒等元 -inf（0xff800000），DSL/CPU 侧 2026-09 已改；③ 旧版行归约 shared 串行树（8 次 barrier）。整改后三条只保留当前设计的现在式陈述（按列合并访存、lowest() 恒等元及 NaN 理由、subgroup 蝶式 3 次 barrier），旧版描述与 159 GB/s 对比数据移此。

## gather 旧版逐元素除模与「逐位同构」（原位置 shaders/gather.comp:21-24、96）
- 类型：演进叙事
- 内容：原头注释「vec4 + 按组求商（2026-09-24，与 DSL 侧…契约同步）：旧版每线程 1 元素、每元素一次 i=idx/D、d=idx%D 两笔整数除模 + 4 次标量表读。现：…」；回退路径原写「回退：标量（每元素求 i/d；与旧版逐位同构）」。整改后改为组级求商的现在式描述，回退路径改为「与 vec4 快路径逐位一致」（等价性指的是当前两条路径之间）。

## elementwise_v2 的旧标量 OP 级描述（原位置 shaders/elementwise_v2.comp:51-54）
- 类型：演进叙事
- 内容：原文「vec4 kernel（2026-09-24，与 DSL 融合逐元素 vec4 路径同步）：DSL 侧（expr_glsl_gen glsl_vec4_eligible + vec4 kernel）早已按『每线程 4 个相邻元素』发射…；OP 级此前是标量每线程 1 元素。本 shader 现镜像同一契约」。整改后只保留「与 DSL 侧同契约 + 本 shader 镜像同一契约」的现在式。

## group_reduce 恒等元注释的日期与旧值叙事（原位置 shaders/group_reduce.comp:59-61）
- 类型：演进叙事
- 内容：原文「max 恒等元 = numeric_limits::lowest()（0xFF7FFFFF，2026-09-24 与 reduce.comp / DSL / CPU 全库统一；旧 -inf 在全 -inf 输入下与减法组合会出 −inf−−inf=NaN）」。数值、全库统一契约与 NaN 理由保留，删除日期与「旧 -inf」叙事（改述为「-inf 不能作恒等元」）。

## transpose 消融链、轮次与 32×32 对比（原位置 shaders/transpose.comp:6-31、49）
- 类型：性能 A/B / 演进叙事
- 内容：原头注释「64×64 分块 + 8×8 WG 砖块化 dispatch（2026-09-24 二轮定稿）：一轮（32×32 tile / 线性 dispatch）5244² 实测 0.958ms；二轮经三步消融定案：消融链（5244²，wall，warmup 30 best-of-10）scale 纯拷贝 0.736 / 过 shared+barrier 恒等拷贝 0.754（shared/barrier 仅 ~2%）/ 真转置 + 线性 dispatch 0.942（+25% 来自写侧足迹散布，136 个并发 WG）/ 真转置 + 8×8 砖块化 0.798（本版本，−15%）；结论『瓶颈不是段宽（64×64 vs 32×32 仅 2%）』；其余定案（A/B 均已验证）：64×64 tile（vs 32×32）段宽 128B→256B、每线程 8→16 元素摊薄 barrier、共享落位 A/B 持平；f16 变体注释另含「（spike 验证 shared float16_t 合法）」。整改后保留：足迹聚集度结论 + 砖块化 0.942→0.798ms（−15%）+ 16×8 砖持平 8×8 + 全部设计要点；移出：轮次流水、完整四点消融链、32×32 对比数据、spike 备注。原文 dispatch 网格「(16→8, 8, n_bricks)」含 16→8 演进标记，已改为 (8, 8, n_bricks)（与文件末尾 dispatch 契约一致）。

## scan_prefix_outer_gen「不再驻 shared」措辞（原位置 shaders/scan_prefix_outer_gen.comp:3）
- 类型：措辞（历史残留）
- 内容：原句「A/B 运行态不再驻 shared（shared 是 O(dk²) 且受 48KB 预算限制），改驻全局显存 scratch」暗示曾驻 shared；整改后为「A/B 运行态驻全局显存 scratch（B*H*2*dk*dk）——shared 是 O(dk²) 且受 48KB 预算限制，无法驻留」。

## 精度变体注释的「逐字未改 / 零回归」参照措辞（原位置 shaders/matmul_tiled.comp:38、batched_matmul.comp:48、reduce.comp:69-70）
- 类型：措辞（历史残留）
- 内容：原文「f32 分支文本逐字未改（零回归）」「f32 分支展开与原文逐字同形 → SPIR-V 字节零回归」以改动前状态为参照；整改后改为当前机制陈述「f32 分支宏展开与裸写 f32 代码逐字同形 → SPIR-V 字节零差异」。

---

# 构建期工具 / 基准脚本 / 对拍脚本

> 本文件摘录 `tools/`、`bench/`、`compare_with_torch/` 注释整改中移出的历史状态记录（原位置为整改前行号）。

## push-constant 固定头长度按 5 个 uint 打包的错误（原位置 tools/scan_exprs.cpp:572）
- 类型：bug 根因
- 内容：历史 bug 曾把"归约但无 matmul"形态的 push-constant 固定头按 5 个 uint 打包（应为 4）→ 常量池整体后移一个 uint → GPU 上"带常量池的归约"静默错值而 CPU 正常（表现为 `col_reduce_sum(select(x == col_broadcast(max), 1, 0))` 恒返回 kk-1 而非真实并列数）。回归用例 `fused_gpu_test::run_reduce_consts`。教训：这类"只有 GPU 错"的问题先打印生成的 GLSL/IR 再猜成因。

## dry-run 用 (void) 吞错导致真实路径从未登记（原位置 tools/scan_exprs.cpp:254；同型 :442）
- 类型：bug 根因
- 内容：旧 dry-run 用 `(void)` 吞掉 forward 错误（ZiPTBlock 的 dry-run 调单参 `forward(x)`，实际返回"用双参"错误）→ 该 dry-run 从未执行真实路径，存量结构恰好处处有 key 才长期绿，块内新表达式迁移后立刻闭合世界硬报错（2026-09-27 教训）。已修为：双参真实签名 + 检查返回值 + 失败带栈 abort；CSA 的 `run_csa` 同型（旧行为是吞错后 backward 在 batched_matmul 读空张量上 NN_ASSERT，栈无上下文难定位）。**dry-run 长期绿 ≠ 覆盖存在。**

## MHA 裸 S 表达式此前只存在于 masked 分支（原位置 tools/scan_exprs.cpp:491）
- 类型：覆盖演进
- 内容：backward recompute 的无掩码 `dsl::compute(matmul)`（裸 S 表达式）此前只在 masked 分支里出现过；MHA（Plain 双向无掩码）dry-run 加入后才成为该结构的闭合世界注册来源。

## fold 掩码 Doc/AlibiDoc 曾漏登记（原位置 tools/scan_exprs.cpp:617）
- 类型：bug 根因
- 内容：P-C2 attention fold 的 5 个掩码变体曾漏登记 Doc/AlibiDoc → GPU doc 训练闭合世界硬报错；由 fused 对拍补充 Doc/AlibiDoc 用例后暴露。该块是 fold spec 唯一注册来源（层 forward 直调 `engine.eval_expr(make_fold_attn_o)`，不经 `dsl::compute` 的 NN_EXPR_SCAN 钩子）。

## canonicalize 曾静默丢 fold 段（原位置 tools/gen_fused.cpp:249）
- 类型：bug 根因
- 内容：canonicalize 类变换曾静默丢掉 fold 段，产出"空指令表且无 matmul/fold"的结构；逐元素生成器对其 `back()` 触发 UB（实测 0xC00000FD 栈崩溃）。现生成器对此防御性跳过 + 告警；正常管线到不了该分支。

## gen_fused 对 matmul+归约/列归约形态的跳过逻辑（原位置 tools/gen_fused.cpp:248、259）
- 类型：删除清单
- 内容：`gen_fused` 曾对 matmul+列归约（S5 列方向，如 `col_max(matmul)`）与 matmul+归约（注意力结构）整体跳过生成；随生成器补齐（列方向按元素分解 batch `batch = row/m_per`、归约遍历全部 rows；`generate_glsl_reduce` 支持 Matmul 操作数内联点积）后，跳过分支删除，两形态均正常生成。

## 精度变体支持范围的过期声明（原位置 tools/gen_fused.cpp:309）
- 类型：演进状态
- 内容：变体发射注释曾称"reduce/matmul/fold 的带类型变体尚未实现"；现状为纯逐元素、matmul 段、fold 段（Phase D2）与归约 kernel（`generate_glsl_reduce(name,spec,sig)`）均已支持带类型生成，`gen_fused` 实测 0 skip。注：`include/neuralnet.cpp/expr_glsl_gen.hpp:1391` 存在同一过期表述（"目前只支持纯逐元素形态；reduce/matmul/fold 拿到 sig != 0 时返回空串"，与同文件 :1406 矛盾），不在本次 tools 整改范围内，需另行处理。

## edit_ranges 花括号护栏的事故来源（原位置 tools/edit_ranges.ps1:61）
- 类型：bug 根因
- 内容：结构性不变量（区间内 `{}` 自平衡）源于 2026-09-26 gpu_test 事故：边界串匹配全过但删掉了 `if (…)` 的起始行、留下其闭合 `}`，文本断言全过而结构已损坏。整改后注释只保留"断言全过 ≠ 结构对"的理由，去掉事故日期叙事。

## text_train.py 样本采样口径演进（原位置 compare_with_torch/text_train.py:139、268、293）
- 类型：演进叙事
- 内容：该脚本早期每 step 独立随机采样，后改为"epoch 开头 shuffle 索引队列 + 每 step 顺序切片"，保证每个样本每 epoch 恰被访问一次；C++ `src/text_train.cpp` 现为同一口径（`std::shuffle` 索引队列 + 顺序切片）。整改后注释只记录当前口径（"构造约定"/"每个样本每 epoch 恰被访问一次"），去掉"不再/改造点（相对 C++ 版）"叙事。

## "旧行为"表述与旧实现等价性论证（原位置 tools/scan_exprs.cpp:60、84、586；compare_with_torch/model.py:286；compare_with_torch/tokenizer.py:3）
- 类型：等价性论证（措辞）
- 内容：曾用"旧行为"描述 f32 pass（sig=0）与无 mask loss 路径，并论证 f32 pass 输出"逐字节旧行为 / bin 不变"；tokenizer.py 头注释曾写"不再手写 BPE 算法"。整改后改为现在式事实：f32 pass 收集 sig=0 基础结构（结构表）；无 mask 路径 `reduction="mean"` 逐位置等权；本模块直接使用 HuggingFace tokenizers 库、不手写 BPE 算法。

## Conv/Pool 引擎化与优化器/Linear 的"迁移"叙事（原位置 tools/scan_exprs.cpp:555、308、336、338）
- 类型：演进叙事
- 内容：MaxPool2D dry-run 注释曾写"这是 Conv/Pool 引擎化后新增的结构"；优化器与 Linear 注释曾写"已迁移为 DSL 融合表达式 / dsl::compute(...)”。整改后改为当前事实表述："该表达式未被扫描覆盖时 GPU 闭合世界硬报错"、"实现为 dsl::compute(...)"，表达式形态与 dry-run 理由原样保留。

## 悬空文档引用（原位置 tools/scan_exprs.cpp:352、389、359）
- 类型：失效链接
- 内容：注释曾引用不存在的 `docs/14`（S1-S3/S5 章节）与已无对应文档的条目号 `P2-13`。前者改指现存的 `docs/development/02-operator-fusion.md`（二期 S1-S7 主线）；后者改为直接描述现行机制"同 RowMod/RotateHalf 的视图参数处理：不进 key，运行时填充"。`include/neuralnet.cpp/expr_spec.hpp:429` 亦有同款 `P2-13` 悬空引用，不在本次范围内，需另行处理。

---

# 表达式与融合测试

## ADL 作用域说明中的旧代数运算符（原位置 src/expr_dsl_test.cpp:27、src/expr_opt_test.cpp:28、src/expr_reduce_test.cpp:28、src/expr_matmul_test.cpp:30，整改前）
- 类型：演进记录
- 内容：四文件同一句「测试写在全局作用域（非 namespace nn），避免与旧代数运算符的 ADL 歧义。」——「旧代数运算符」指 algebra_expr.hpp / algebra_compute.hpp 的 Expression / BoolExpression（已于 2026-09 移除，概念迁入 expr_dsl.hpp），已不存在。整改后改为当前理由「避免 ADL 把匹配拉进 nn 命名空间」。

## matmul+列归约组合曾被 gen_fused 跳过（原位置 src/expr_matmul_test.cpp:269-270、src/fused_gpu_test.cpp:454-455，整改前）
- 类型：覆盖缺口 / 演进记录
- 内容：expr_matmul 原文「该结构曾被 gen_fused 跳过（"matmul+列归约组合暂不支持"），生成器补齐按元素 batch 分解后，此处锁死其语义：列归约遍历全部 rows（= batch*M），与独立标量参考…一致。」fused_gpu 原文「该组合曾被 gen_fused 跳过（"matmul+列归约组合暂不支持"），生成器补齐按元素 batch 分解后，此对拍锁死 GPU 与 CPU 语义一致（列归约遍历全部 rows，含所有 batch）。」整改后删除「曾被跳过/补齐」的实现进度流水，只留当前覆盖陈述（覆盖 matmul+列归约组合、列归约遍历全部 rows、GPU 与 CPU 语义一致）。

## 注意力 fold 构造的归位流水（原位置 src/expr_fold_test.cpp:33，整改前）
- 类型：演进记录
- 内容：原文「#include <neuralnet.cpp/compute_layer_attention.hpp> // 注意力 fold 构造（已从 expr_fold.hpp 归位到 Layer）」——fold 构造与 5 掩码变体于 2026-09-25 从 expr_fold.hpp 迁至 compute_layer_attention.hpp。整改后改为「（定义于 Layer：表达式文本只写在 Layer）」，只表达当前 AOT 分工。

## fold 对拍 NaN 守卫的红验证流水（原位置 src/expr_fold_test.cpp:46-47，整改前）
- 类型：bug 根因 / 演进记录
- 内容：原文「IEEE fmax(err, NaN) = err 会静默吞掉 NaN diff——2026-09 fold doc 掩码 GPU -nan 回归曾因此在对拍中漏抓（红验证实证：revert 后仍 PASS）。」整改后保留机制与当前理由（若无此守卫，被测端输出 NaN 时对拍仍可能 PASS，故直接把非有限 diff 记为 inf），删除回归日期与红验证/实证流水。

## doc 分段边界：2026-09 GPU -nan 触发形态与旧边界漏抓（原位置 src/expr_fold_test.cpp:316-321，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「…被文档掩码**全部**屏蔽——这正是 2026-09 GPU 训练 -nan 的触发形态（max init=-inf 时 m_old=blk_m=-inf → dm=−inf−−inf=NaN）。旧固定 seq/2=66<128 永远让首块留有有效项 → ctest 全绿漏抓。seq ≤ 128 时保持原二分段。」整改后保留当前覆盖理由（边界须越过 128 才能让首块被全屏蔽、触发 dm=−inf−−inf=NaN 分支；边界落在 128 之内则首块永留有效项、该分支测不到），删除 2026-09 日期与「旧固定 seq/2=66 / ctest 全绿漏抓」流水。

## GPU 侧 doc 边界的钉根因与旧边界漏抓（原位置 src/fused_gpu_test.cpp:732-734，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「让 i≥129 的行首 fold 块被 doc 掩码**全屏蔽**——钉住 2026-09 GPU -nan 根因（max init=-inf → dm=−inf−−inf=NaN）；旧固定 seq/2=66<128 首块永留有效项 → 对拍漏抓该 bug。」整改后改为当前分支触发说明（全屏蔽触发 NaN 分支；边界落在 128 之内测不到），删除「钉住…根因」与旧边界漏抓流水。

## NaN 守卫的双层漏抓出处（原位置 src/fused_gpu_test.cpp:789-790，整改前）
- 类型：演进记录
- 内容：原文「…会静默吞掉 → err 保持正常值照样 PASS（2026-09 fold doc -nan 回归双层漏抓之一，红验证实证）→ 记 inf 必超容差」。整改后保留机制与当前结论（记 inf 必超容差、NaN 输出必须硬失败），删除回归出处与红验证流水。

## gpu_test 头注中的逐元素职责（原位置 src/gpu_test.cpp:6-8，整改前）
- 类型：删除清单
- 内容：原文「验证 CpuEngine 与 GpuEngine 在 matmul / 转置 matmul / 逐元素 / 归约 / roundtrip 等原语上的一致性，并对比 GPU vs CPU matmul 性能。」——逐元素算子已随 2026-09 收敛删除，元素级表达式对拍职责转到 expr_gpu_test。整改后头注去掉「逐元素」并加指路「（元素级表达式对拍见 expr_gpu_test）」。

## 逐元素算子移除的删除清单碎片（原位置 src/gpu_test.cpp:354、466、503，整改前）
- 类型：删除清单
- 内容：原文「4b. CPU/GPU 张量准备（逐元素算子已移除；元素级 DSL 对拍见 expr_gpu_test）」「链式: matmul(A, B) → col_reduce_sum（Exp 级已随逐元素算子移除）」「── Step 2: 直接复用 matmul 输出（Exp 级已随逐元素算子移除）──」。整改后删除括号内的移除流水，只留指路与当前步骤名。另 :555 的输出字符串「（Exp 级已随逐元素算子移除）」是打印字面量，按规则未改动。

## transpose 砖块派发 bug 与旧形状漏检（原位置 src/gpu_test.cpp:271-276，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「── 3d. transpose 逐元素对拍（issue #13 P0-① 回归）──」「历史 bug：backend 派发 (16,8,n_bricks) 而 transpose.comp 按 8 宽砖解算（假定 gl_WorkGroupID.x ∈ [0,8)）→ 行>512 且 列>512 时静默只写前 512 行。旧测试形状最大 64×256，永远单边 ≤512，故漏检。」「形状表覆盖：单边 ≤512（旧代码 PASS）/ 双边 >512（旧代码 FAIL 50%）/ 奇数边界…」。整改后改为当前契约（backend 必须按 (8,8,n_bricks) 派发、与 transpose.comp 的砖解算不同源时双边 >512 静默只写前 512 行；单边 ≤512 测不出），删除 issue #13 编号、「历史 bug」叙述、「旧测试形状 64×256 漏检」与「旧代码 PASS/FAIL」对照。

## row_max/denom 与旧手写内核的等价性论证（原位置 src/fused_gpu_test.cpp:397-398，整改前）
- 类型：与旧实现的等价性论证
- 内容：原文「row_max(QK^T)（bmm_reduce Max 等价）与 denom = row_sum(exp(QK^T - rm))（bmm_denom 等价）均为"matmul 段 + 归约指令"单表达式，GPU 经 generate_glsl_reduce 的 matmul 支持单 kernel 完成。」——bmm_reduce / bmm_denom 旧实现已删除。整改后去掉两处「（…等价）」括注，保留结构归类与 GPU 单 kernel 完成的当前陈述。

## push-constant 固定头长度的既往错档 bug（原位置 src/fused_gpu_test.cpp:556-561，整改前）
- 类型：bug 根因
- 内容：原文「── 归约表达式内联常量：CPU vs GPU（push-constant 头长度回归）──…该结构带**常量池**。GPU 侧 push-constant 固定头长度必须按形态算；历史上"归约但无 matmul"曾多算 1 个 uint（5 vs 4）→ 常量池整体后移一个 uint → shader 读错常量 → **GPU 静默错值而 CPU 正常**。本用例锁死该回归。」整改后给出当前契约表（逐元素 2 / 逐元素+matmul 5 / 归约 4 / 归约+matmul 6），把「历史上曾多算」改写为「按错档则常量池整体错位 → shader 读错常量 → GPU 静默错值而 CPU 正常」，删除既往 bug 流水。

## fold K 族注释中的常量升位流水（原位置 src/fused_gpu_test.cpp:616，整改前；本轮补改）
- 类型：演进记录
- 内容：原文「K 族覆盖单列 / 非块整除 / 整除 / 尾块（EXPR_FOLD_BLOCK=128 边界两侧——256/260 补多块+尾块；BLOCK 升 128 后 K≤100 会静默退化单块）。」本轮改写为「…256/260 补多块+尾块；K≤100 全落 128 以内的单块区间，测不到多块/尾块）」，保留 K 族选值理由，去掉常量升位的历史叙述。

## 列式 softmax 手写原语的删除清单（原位置 src/ce_fusion_test.cpp:5-6，整改前）
- 类型：删除清单
- 内容：原文「- col_softmax_denom：denom[c] = Σ_r exp(logits[r][c] - col_max[c]) → 已删除（IR 融合替代）」「- col_softmax_sparse_forward：单 kernel 稠密梯度 + 标签位置 loss_vec → 已删除（IR 融合替代）（不物化全 softmax），含 mask / 越界标签处理」。整改后改为当前覆盖清单：IR denom 由 S7 组合 col_reduce_sum(exp(logits − cb(col_max))) 对照手写参考；loss_vec/grad 由 RowGather + Row 操作数组合（不物化全 softmax）；另新增一条「归约 + 后处理（reduce(...) * k + c）：CPU 正确求值、GPU 必须硬报错」。

## check_matrix 返回值曾被忽略（原位置 src/ce_fusion_test.cpp:38-39，整改前）
- 类型：bug 根因
- 内容：原文「返回是否通过：调用方必须把 false 计入失败数（此前只累加 g_fail 而不影响 run_case 返回值/进程退出码，导致 [FAIL] 行被 "ALL PASS" 掩盖 —— 修）。」整改后保留调用契约与后果（只累加 g_fail 而不影响退出码时，[FAIL] 行会被 "ALL PASS" 掩盖），删除「此前…—— 修」的修复流水。

## 归约+后处理的两段背景（原位置 src/ce_fusion_test.cpp:169-178，整改前）
- 类型：bug 根因 / 演进记录
- 内容：原文「── 回归：带"归约 + 后处理"的归约向量表达式（reduce(...) * k + c）──…背景一（CPU）：eval_expr_reduce 的"输出沿归约轴恒定"前置校验原本按**寄存器号**传播所需指令（needed[reg]），而寄存器分配器按 liveness 复用逐元素寄存器号 —— 后处理指令的 dst 与归约前的 Mul 同号时，归约前的定义被误判为输出链的一部分（Linear 访问），该类表达式被错误拒绝。现改为按**指令下标**反向切片，CPU 正确求值。背景二（GPU）：归约融合 shader 尚未正确实现"归约后仍有逐元素后处理"的形态（实测静默错值），故 GPU 必须**硬报错**而非返回错值；本用例断言这一点，防止将来退化为静默错误。」整改后改为当前实现/契约陈述（前置校验按**指令下标**反向切片；GPU 不支持该形态只会静默错值，故必须硬报错），删除「原本 / 现改为 / 尚未正确实现」的修复与实现进度流水；寄存器号复用导致误拒的机制作为当前设计理由保留。

## inv_num_valid 的旧接线方式（原位置 src/ce_fusion_test.cpp:308，整改前）
- 类型：演进记录
- 内容：原文「注：inv_num_valid 由 RParam 承载（运行时标量，不进 expr_spec_key）→ 与整条逐元素链融合为单 kernel（原先在表达式后补一次 scale_inplace）。」整改后删除括号内旧接线（原先补 scale_inplace），保留 RParam 不进 key、整链单 kernel 的当前陈述。

## 算子 f16 变体对拍的覆盖缺口与先例引用（原位置 src/gpu_f16_test.cpp:69-70，整改前）
- 类型：覆盖缺口
- 内容：原文「…的 f16 存储变体（这些路径此前只有 f32 覆盖，f16 变体一旦类型/索引出错是静默错值——对拍是唯一护栏，同 gather vec4 改写先例）」。整改后保留当前风险理由（f16 变体的类型/索引一旦出错即静默错值，对拍是唯一护栏），删除「此前只有 f32 覆盖」缺口叙述与 gather vec4 先例引用。

---

# 层 / 模型测试

## MaxPool2D 测试覆盖缺口（原位置 src/maxpool_gradcheck.cpp:3）
- 类型：覆盖缺口
- 内容：原文「目的：MaxPool2D 此前**没有任何测试覆盖**（Conv2D 由 conv2d_gradcheck 覆盖）。」整改后只保留当前覆盖陈述（本片段把层实现与独立参考逐元素比对）。

## clear_cache 后 backward 的旧行为（原位置 src/maxpool_gradcheck.cpp:8-9）
- 类型：bug 根因
- 内容：原文「① clear_cache() 后直接 backward 必须报错 —— 旧行为是越界读空 vector（UB；因 vector::clear() 保留容量，表现为"静默沿用陈旧索引"不报错）」。整改后改写为当前契约「clear_cache() 后 backward 必须报错——缓存为空时不得读取」，并把「vector::clear() 保留容量 → 越界读未必崩溃」保留为一般性陷阱警告（不再叙述「旧行为」）。

## Conv2D 测试覆盖缺口（原位置 src/conv2d_gradcheck.cpp:3）
- 类型：覆盖缺口
- 内容：原文「目的：Conv2D 此前**没有任何测试覆盖**。本片段把层实现与一份独立写法的直接卷积…」。整改后删除覆盖缺口叙述，只留「本片段把层实现与独立写的直接卷积参考逐位比对」。

## CNN 无端到端覆盖的背景（原位置 src/cnn_smoke_test.cpp:3-5）
- 类型：覆盖缺口
- 内容：原文「目的：CNN 此前只有单层 Conv2D 参考比对（conv2d_gradcheck），**没有端到端覆盖**——最大池化的反向、以及整链 forward/backward/optimizer 只能靠手工跑 `mnist_train --arch cnn` 验证。本片段补上：」。整改后改写为当前分工陈述：单层参考比对不覆盖规格序列化/层组成/整链训练，本片段验证这些。

## 合并测试里的覆盖缺口标签（原位置 src/cnn_test.cpp:10、src/layer_gradcheck_test.cpp:9）
- 类型：覆盖缺口
- 内容：两行原为「── maxpool_gradcheck（独立参考实现比对；MaxPool2D 此前无任何覆盖）──」与「── conv2d_gradcheck（独立参考实现比对；Conv2D 此前无任何覆盖）──」。整改后删除括号里的「此前无任何覆盖」，只留「独立参考实现比对」。

## doc_mask_test / build_attention_mask 删除清单（原位置 src/doc_attn_test.cpp:4-6）
- 类型：删除清单
- 内容：原文「（原 doc_mask_test 已随物化掩码构建函数 build_attention_mask 一并删除——掩码语义现由 fold body 表达：spec 级覆盖见 expr_fold_test 三掩码，doc 端到端覆盖见下方 e2e）」。整改后删除「原 doc_mask_test 已…删除」的删除清单叙述，保留当前覆盖分工（掩码语义由 fold body 表达，spec 级覆盖见 expr_fold_test，端到端见本 e2e）。

## MHA 掩码默认值的回归背景（原位置 src/attn_test.cpp:24-26）
- 类型：bug 根因
- 内容：原文「回归背景：fold 迁移曾把 fmask 默认树设为 Causal → MultiHeadAttention（本应双向无掩码，旧 apply_mask_ 默认 no-op）被静默因果遮蔽；层级此前只测 CSA，未被抓住。」整改后改写为当前约束「⚠ MHA 默认掩码必须为 Plain（双向）；默认树设为 Causal 会静默因果遮蔽双向注意力」，删除 fold 迁移流水、旧 apply_mask_ 行为与「层级此前只测 CSA」的覆盖缺口叙述。

## Doc/AlibiDoc backward 分支的零执行覆盖叙述（原位置 src/attn_test.cpp:194-196）
- 类型：覆盖缺口
- 内容：原文「Doc / AlibiDoc backward 覆盖（masked_doc_ / masked_alibi_doc_ 分支此前零执行——forward 对、梯度串文档抓不住；默认 learned+doc 与 alibi+doc 各跑一遍数值梯度）」。整改后改写为当前覆盖理由「本段专门跑 masked_doc_/masked_alibi_doc_ 分支的梯度（仅 forward 通过不足以验证——梯度跨文档串扰 forward 抓不住）」。

## GPU 融合 matmul 尾链 batch 行号 bug 的日期流水（原位置 src/attn_w_batch_test.cpp:3-5）
- 类型：bug 根因 / 日期流水
- 内容：原文「背景：GPU 融合 matmul 尾链曾按 **batch 内行号**读取 (rows,1) 全网格输入（注意力的 m/l），导致 batch>1 或多头（BH>1）时 m/l 读错 → GPU 前向错误（2026-08-27 审查发现，P0-1）。」整改后改写为当前契约（尾链必须按全局行号读取，按 batch 内行号读会读错 m/l），删除日期与 P0-1 审查流水。同文件 :107 的「复现原 batch 内行号 bug」改为「暴露 batch 内行号读错」。

## forward/forward_step 一致性与旧模型权重的结论（原位置 src/attn_consistency_test.cpp:4-5）
- 类型：历史状态
- 内容：原文「…若一致则证明 Attention 语义修复正确，生成乱码是旧模型权重的问题（用 buggy 代码训练得到），需重训。」整改后只保留当前语义（两者一致即证明两条推理路径语义等价），删除「语义修复正确 / buggy 代码训练的旧权重 / 需重训」的历史结论。

## GPT 整链 gradcheck 的覆盖缺口与用户现象（原位置 src/gpt_gradcheck.cpp:5-6）
- 类型：覆盖缺口 / bug 根因
- 内容：原文「单层 gradcheck（rmsnorm/swiglu）已验证层内数学，但多层堆叠 + learned 位置编码 + token_emb scatter 的整链路径从未验证。用户现象：浅层 GPT 正常、深层 GPT loss 卡平台 → 疑似链路级 bug。」整改后改写为当前覆盖陈述（整链路径由本测试专门验证，链路级错误只有整链梯度对拍才抓得住），删除「从未验证」缺口叙述与用户现象流水。

## attn_gradcheck 的定位流水（原位置 src/attn_gradcheck.cpp:3-5）
- 类型：调试流水 / 覆盖缺口
- 内容：原文「目的：GPT 整链 gradcheck 显示 wo OK 但 wq/wk/wv FAIL，定位 attention 内部是否真的有反向 bug（batched_matmul / softmax backward / 掩码）。batch=1 时绕过 rearrange_3d，若仍 FAIL 则是 attention 内部问题。」整改后保留分层定位的设计意图，删除「显示 … FAIL」的既往调试结果。同文件 :142-144「该分支此前零执行覆盖（forward 对、梯度串文档抓不住）」改写为「本用例专门验证该分支的梯度（仅 forward 通过不足以验证）」。

## softmax_gradcheck 的既往失败流水（原位置 src/softmax_gradcheck.cpp:3-5）
- 类型：调试流水
- 内容：原文「目的：attention gradcheck 显示 grad_Q/grad_K FAIL 但 grad_V OK，共同差异是 grad_S = softmax.backward(grad_A)。本测试单独验证 Softmax 的 forward/backward 是否与中心差分一致。」整改后改为当前意图陈述（单独隔离验证 Softmax；grad_Q/grad_K 由 grad_S 派生、grad_V 不经 Softmax，故单独跑便于分层定位），删除既往 FAIL 结果叙述。

## 跨样本隔离用例的「历史 bug」表述（原位置 src/doc_attn_e2e_test.cpp:119）
- 类型：bug 根因
- 内容：原文「其 logits 只可能因跨样本串扰而变（position-major 类历史 bug 在 batch=1 下不可见）」。整改后保留理由、删去「历史 bug」：「batch>1 是必须的：batch=1 时 position-major 与 batch-major 布局重合，串扰不可见」。

---

# 精度 / RAPT / ZiPT / 分词器测试与探针

> 本轮「精度 / RAPT / ZiPT / 分词器 / 杂项测试与探针」注释整改中摘出的实质性历史。
> 对应改动文件：src/test_common.hpp、src/precision_profile_test.cpp、
> src/precision_type_test.cpp、src/f16_precision_test.cpp、src/f16_compute_test.cpp、
> src/tensor_precision_test.cpp、src/f16_cpu_probe.cpp、src/mem_probe.cpp、
> src/rapt_offload_test.cpp、src/tokenizer_consistency_test.cpp。

## test_common.hpp 聚合结构的由来（原位置 src/test_common.hpp:3-8）
- 类型：结构演进说明
- 内容：历史上每个子测试都自带一份 CHECK / make_tensor / check_close / approx /
  dot / close_to，聚合器（`#define main test_xxx + #include 子测试.cpp` 模式）
  被迫用一长串 `#define` 逐符号重命名来规避 ODR 冲突（漏一个就是重定义或静默绑错）。
  2026-09 收敛为 test_common.hpp 单份后，聚合器只需重命名 main、失败计数器与
  各测试独有的函数名。另：「刻意不收编」决定（2026-09 审查）——max_abs_diff 因
  9 份副本语义有分叉（conv2d 版多形状守卫返回 1e9）暂不收编，另案处理。

## ref_ulp 曾误写成 2^(e-10)（原位置 src/precision_type_test.cpp:55-57）
- 类型：bug 根因
- 内容：ref_ulp 曾误写成 2^(e-10)（正确为 2^(e-25)，大 2^15 倍）→ RHE 中点
  断言的容差比被测值本身还大 → float_to_half_bits 次正规 UB 窗口
  （exp ∈ [-45,-33]）产出的垃圾 half（如 0x4000=2.0）在该容差下永远测不出来。
  修复 = ref_ulp 改回 2^(e-25)，容差恢复到真实 ulp 量级。

## 次正规移位 UB 窗口与两个垃圾代表值（原位置 src/precision_type_test.cpp:195-198, 214）
- 类型：bug 根因
- 内容：precision.hpp 的 `float_to_half_bits` 次正规分支守卫曾写 `exp <= -46`，
  与其自身公式矛盾 → exp ∈ [-45,-33]（|v| ≈ 2.8e-14 ~ 1.2e-10）走 shift ≥ 32
  的移位 UB → 指数字段回绕产出垃圾 half（0x4000=2.0、0xCCCD… 等）→ CPU f16
  训练梯度被写成 512/8192/11776/18432/NaN，若干步后更新爆炸（梯度正是这个量级、
  前向激活 ~0.1 从不落入，故单算子测试全过）。修复 = 守卫改 `exp <= -26`
  （≤-26 按 D<0.5 恒 flush 0）。定点断言的两个代表值 1.13687e-12 与 8.44011e-11
  正是修复窗口上沿，修复前分别产出 0x4000=2.0 与 0x3333。

## `--f16` 曾等价于 profile_master_weights()（原位置 src/precision_profile_test.cpp:4-7）
- 类型：语义变更记录
- 内容：`--f16` 曾等价于 profile_master_weights()（F32 主权重 + f16 计算），
  用户以为开了全 f16、实际参数仍是 F32。2026-09 修订为 profile_f16()
  = {param:F16, compute:F16, stable:F32, optimizer:F32}（本测试把该语义钉死）。
  stable/optimizer 不取 F16 的实测证据：optimizer=F16 → Adam 的 v≈g²~1e-10
  f16 下溢、更新爆炸（loss 7.9→3.6e4）；stable=F16 → CE 链 ~200 步 NaN；
  四字段全 F16 → loss 恒定（更新被 f16 舍入吃光）。

## CPU 侧 f16 训练发散的双根因（原位置 src/f16_precision_test.cpp:569-573）
- 类型：bug 根因
- 内容：该处原为「历史"已知问题"（CPU 侧 f16 训练发散）已修复并转为硬失败」的
  根因注释：根因1 = DSL 预绑定把 f16 操作数喂给 f32 GEMM（cpu_matrix<F32>()
  拿到空指针 UB）；根因2 = float_to_half_bits 次正规分支 exp<=-46 的移位 UB
  （|v| ∈ [2.8e-14, 1.2e-10] 的梯度被写成垃圾 half）。完整记录见
  docs/development/05-mixed-precision.md §12.12。修复后 CPU f16 三组 profile
  6 步轨迹与 f32 逐位一致，本处改为硬失败断言。

## 逐元素原语已整体移除（原位置 src/f16_precision_test.cpp:93-94）
- 类型：删除清单
- 内容：原注释「（逐元素原语已整体移除；逐元素路径的 f32 零回归见
  test_f32_zero_regression_dsl）」——引擎级 elementwise_unary/binary/binary_scalar
  等算子于 2026-09 收敛时删除（逐元素一律走表达式 DSL），故 f32 零回归用例
  的代表原语选归约/搬运两类，逐元素路径的对拍由 DSL 版用例承担。

## rapt_offload：RLA backward 缓存曾不在 activation_cache() 内（原位置 src/rapt_offload_test.cpp:11-13）
- 类型：覆盖缺口
- 内容：RLA 的 backward 缓存（RMSNorm 后 Q/K、逐头 1/rms）此前不在
  activation_cache() 导出集合内，activation offload 会漏导出/恢复这些缓存。
  2026-09-19 RAPT 补齐 set_activation_offload 接线与 activation_cache() 覆盖后，
  本测试对拍 offload 开/关两侧的导出恢复集合完整性（缺一项即 backward 报缓存
  缺失或数值不一致）。

## 参考实现曾称「完整旧版 pre_tokenize」（原位置 src/tokenizer_consistency_test.cpp:35-36）
- 类型：措辞修正（与已删除实现的等价性论证）
- 内容：该参考函数原注释自述「完整旧版 pre_tokenize……逐字复刻优化前
  domain_tokenizer_bpe.hpp 的实现」。它当前是手写状态机（pre_match_len）的
  对拍基准，应表述为独立朴素实现（正则切分），不称「旧版」。

## docs/23 陈旧文档引用（原位置 src/precision_type_test.cpp:1、src/f16_compute_test.cpp:2、src/tensor_precision_test.cpp:2）
- 类型：失效链接
- 内容：三处「docs/23 §13 / §12.2」指向已不存在的旧编号设计文档；同内容现行
  位置为 docs/development/05-mixed-precision.md（§13 测试计划 T1–T11、
  §12.2 Phase 1 验收标准）。precision_type_test 的 printf 输出字符串里仍保留
  字面量「docs/23 T2」（字符串字面量不改动）。

## mem_probe 默认参数的 5.7GB 峰值锚点（原位置 src/mem_probe.cpp:22-23, 281）
- 类型：过期基准值
- 内容：原注释称默认参数「与已知"峰值 5.7GB"场景对齐」。该数字是 09-25 的
  f32 旧峰值（5743 MiB）；随后的显存优化已把它降到 ~3069 MiB，且
  docs/benchmarks/2026-09-25-vulkan-vs-cuda.md §"峰值已强相关于精度" 明确
  「09-25 ~5.7GB 与精度无关」的观察不再成立（f16 现为 6315/5738）。故摘除
  5.7GB 锚点，只保留「默认参数 = 该 bench 配置（vocab 8208 / d64 / h4 / L4 /
  ff256 / seq256 / batch64）」的测量口径陈述。

## f16_cpu_probe 曾自述为「临时诊断」（原位置 src/f16_cpu_probe.cpp:1-2）
- 类型：状态措辞
- 内容：探针最初为定位 CPU f16 训练发散的临时诊断工具；现已是注册的组件级
  探针（阶段标记 + 非有限值扫描），表头改为当前用途陈述。

---

# docs/development 12 · 13 · 08

> 本文件汇集 docs/development/12-compute-engine-inventory.md、13-refactor-backlog.md、
> 08-pitfalls-and-lessons.md 整改中移出的历史叙事（演进流水 / 已修复勘误 / 否决方案 /
> 过程性验证数字）。正文只保留当前状态与仍然适用的教训。

## 2026-09-27 计算类原语全量迁 DSL 横幅（原位置 docs/development/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：迁移范围——inplace 族 `add_inplace`/`scale_inplace`/`accumulate` → `dsl::compute_into`（10 处：attention/mlp/conv/rapt/transformer/zipt）；单 matmul `matmul`/`matmul_with_bias` → `dsl::matmul`（Linear fwd/bwd、Conv gcol、Transformer 池化反传、Muon NS 6 处）；归约族 `row_reduce_sum`/`col_reduce_sum`/`col_reduce_max` → `dsl::compute_reduce`（mlp/conv/transformer/loss×4/optimizer×3/rapt）；批量 matmul `batched_matmul` 21 处 → `dsl::matmul(..., batch)`（attention backward+forward_step、CrossAttention、ZiPTBlock、rapt forward_step），alpha 经 `dsl::rparam(scale)` 尾链（值不进 key）；缺口 A：`grouped_reduce_sum/max` → 新 `ExprViewKind::GroupedReduceSum/Max` 视图（param=R 进 key），MaxPool2D 迁移。结果：Layer 直调 **34 → 23**，10 个计算类原语全部退出 Layer 直调（`bench/doc_inventory.ps1` 复测）；ctest 19/19。随迁的三条方法论教训（dry-run 禁止吞错 / alpha 用 rparam / 新 ExprViewKind 落点清单）保留在 12 §2.1 正文（现在式）。

## 2026-09-19 IR-C 删除横幅 + 2026-09-25 接口数字复核流水（原位置 docs/development/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：① 2026-09-19 横幅：IR-C 部分执行完毕——`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_` 均已从代码移除；§4.2/§4.3/§5/§8 曾保留为"决策前证据快照"。② 2026-09-25 复核：**58 个 virtual（Phase 2 加入 `cast_into`/`copy_into`/`supports_*`/`eval_expr_*` 等）、Layer/Loss/Optimizer 直调 35 个**；随后 2026-09-26 收敛更新为 **49 virtual / Layer 直调 32**，2026-09-27 迁移后为 **49 virtual / Layer 直调 23**（当前值）。

## 2026-09-26 §5 遗留物清单收敛记录横幅（原位置 docs/development/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：① 旧代数 AST 整体移除：`algebra_expr.hpp`、`algebra_compute.hpp`（`nn::compute::apply`）删除，`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`，`Matrix::detail::{apply,binary_apply,binary_apply_inplace}` 死函数删除；② 无根/测试-only 算子全部删除：`axpy_inplace`、`broadcast_row_inplace`、`broadcast_col_inplace`（连带 `shaders/broadcast.comp` 全链）、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、引擎 `row_reduce_max`、offload B 组 `offload_store/offload_load`、`UnaryOp/BinaryOp/CompareOp` 枚举；③ 数字：引擎 virtual **58 → 49**；CPU 求值机制 **4 → 2**（DSL 模板路径 + IR 解释器）；④ 随行清理：`src/test_common.hpp`、死 `activation_cache()` override、重复 `offload_test` 聚合目标（ctest 20 → 19）、`gpt_offload_test` 补 batch 录制包裹。

## §1 CUDA 移除范围清单（原位置 docs/development/12-compute-engine-inventory.md §1）
- 类型：删除清单
- 内容：原先约 100 KB 死代码 + 10 处条件分支——删除 `cuda/`（`CMakeLists.txt`/`cuda_kernels.h`/`cuda_kernels.cu`）、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`；清除 `compute_tensor.hpp`（6 处 `#ifdef NN_HAS_CUDA`）、`nn.hpp`、`cli/cli_engine_factory.hpp`（`EngineConfig::use_cuda`）、`cli/cli_train_common.hpp`（`--cuda`）、`CMakeLists.txt` 停用段、15 个 `src/*.cpp` 的 `--cuda`/`use_cuda`/`cuda_enabled`、`cli_controllers.py` 的 `--cuda` 传参。历史快照见 git 分支 `legacy/cuda`。

## §2.2 失效的 2026-09-25 调用方快照行（原位置 docs/development/12-compute-engine-inventory.md §2.2）
- 类型：已修复勘误（算子已删，行随之失效）
- 内容：`offload_store`/`offload_load`（仅 offload_primitive_test；Layer 用 `offload_save/offload_restore`）、`elementwise_unary`/`elementwise_binary_scalar`/`elementwise_binary`（含 `compute_layer_rapt.hpp:866` Div 生产调用）、`broadcast_row_inplace`（无根）、`broadcast_col_inplace`（仅 f16 测试/layer_bench）、`axpy_inplace`（无根）、`row_reduce_max` 引擎算子（仅适配层转发；Layer 走 `dsl::row_reduce_max` DSL 叶子）、`elementwise_select_scalar_cond`（全仓无调用点）——以上算子已于 2026-09-26 收敛轮从接口删除，当前 49 个 virtual 中不再存在。

## §3「四套机制」旧表与死/重复叙事（原位置 docs/development/12-compute-engine-inventory.md §3）
- 类型：演进记录
- 内容：机制 1「旧代数 AST `compute::apply(span, expr)`」（`algebra_compute.hpp`/`algebra_expr.hpp`/`Matrix::detail::*`）已于 2026-09-26 整体删除；`eval_cpu` 串行模板分支此前已被并行化+向量化的 `eval_into_span` 取代。原判定："机制 2 与机制 3 是 DSL 在 CPU 上的两条路（一条内联、一条解释）——收敛后的全部 CPU 求值路径"；机制 3（IR 解释器）逐元素 switch 分派 + 归约前缀重放，实测比等价原语慢 1.5–3.8 倍（2026-09-18 诊断实测，原诊断报告已删除，结论数字保留）。

## §4.2–§4.3 IR-C 盘点原文（原位置 docs/development/12-compute-engine-inventory.md §4.2、§4.2.1–§4.2.3、§4.3）
- 类型：否决方案 / 演进记录
- 内容：① §4.2 盘点：接口 `compute_engine.hpp:110-111`、录制图 `expr_graph.hpp`（22.5 KB，`ExprGraph`/`recording_graph_owner`）；生产调用方无——唯二调用者 `FusedChainLayer`（`compute_layer_mlp.hpp:762-800`，自述"IR-C 演示"层，不被模型工厂使用）与 `tools/scan_exprs.cpp:136-143` + `src/expr_fuse_test.cpp`。② §4.2.1 代数关系：IR-C 是 DSL 同一流水线的阶段 C（IR-A/B `expr_opt.hpp`、IR-C `expr_graph.hpp`、IR-D `expr_emitter.hpp`）；原始开发分支时间线 `bf86af2`(08-21 DSL) → `662cb08`(08-22 begin_expr/end_expr) → `36caba4`(08-24 IR-C/D)，DSL 先出现、IR-C/D 是同波加入。③ §4.2.2 数据流图：`dsl::compute → ExprSpec → IR-A/B → eval_expr → IR-D`，`begin_expr` 段走 `ExprGraph → fuse_expr_graph`；CPU `eval_expr_impl` 与 GPU `eval_expr` 都先 `canonicalize_expr_spec`。④ §4.2.3 非对称：GPU 真录制 + `execute_fused_graph`，CPU 仅 `NN_EXPR_SCAN` 下动作（普通运行 no-op）→ 接 IR-C 对 CPU 收益为零。⑤ §4.3：`dsl::start_expr/end_expr`（`ExprBlock`）仅被测试使用，随 IR-C 删除。
- 结论去向：完整取舍与重新立项前提见 `docs/development/03-ir-optimization.md` §5.3。

## §5 遗留物清单历史行（原位置 docs/development/12-compute-engine-inventory.md §5）
- 类型：演进记录（已删除/已修复条目）
- 内容：~~CUDA 引擎+backend+`cuda/`+10 处 `#ifdef`~~（~100 KB，已移除）；旧代数 AST `algebra_expr.hpp`+`algebra_ops.hpp`+`algebra_compute.hpp`（27.6 KB，已删除——注意 `algebra_ops.hpp` 仍在，判定原文含误列）；IR-D `CpuEmitter` 残留注释（2026-09-25 已清理，`cpu_emitter.hpp` 已删）；`FusedChainLayer`（~60 行，已移除）；`begin_expr`/`end_expr`+`expr_graph.hpp`+`Tensor::virtual_tag_`（~24 KB，已移除）；`eval_cpu` 串行模板路径（已由并行化 `eval_into_span` 取代）；`axpy_inplace`/`broadcast_row_inplace`（原判无根，2026-09-26 已删）；`broadcast_col_inplace`（原仅测试/bench，已删）；`nn::one_hot`（无调用点且与铁律 9 冲突，已删）；`Matrix::multiply_transposed_add_to`（自述死代码，已删）；`elementwise_select_scalar_cond`（无调用点，已删）；`offload_store`/`offload_load`（原仅测试，已删）；`row_reduce_max` 引擎算子（无 Layer 调用点，已删）。

## §6 差距表旧数字（原位置 docs/development/12-compute-engine-inventory.md §6）
- 类型：演进记录
- 内容：A. Layer 直调算子 ~~35 个~~ → 23（2026-09-27 计算类已全量迁移）；B. 求值机制原为 4 套（含旧代数 AST），2026-09-26 起 CPU 只剩 2 套；C. 融合世代：IR-C 已于 2026-09-19 移除（设计无收益点，见 §8.9 与 03 §5.3）；D. 死代码：CUDA 全链 ~100 KB、IR-C ~24 KB 已清。

## §7 演进段（原位置 docs/development/12-compute-engine-inventory.md §7.2/§7.4/§7.5/§7.6）
- 类型：演进记录
- 内容：① §7.2 类别①原列的 `elementwise_unary/binary/binary_scalar`、`broadcast_row/col_inplace`、`axpy_inplace` 已于 2026-09-26 删除（`add_inplace`/`scale_inplace`/`accumulate`/`zero` 仍在接口）。② §7.4 原文："保留与 DSL 重复的逐元素路径 → 就是今天 CPU 上'旧代数 AST / DSL 模板 / 解释器'三套并存、层直调 35 个算子的直接来源。"③ §7.5 原"缺三样"中的前两样已完成/已处置：`dsl::compute_into(eng, expr, dst)` 已落地（optimizer/残差/LayerNorm 梯度/CE 梯度已迁移）；跨表达式融合（IR-C）判定无收益并移除。④ §7.6 务实路径原文："先把 §7.2 ① 标记为 legacy 并逐层迁移（LayerNorm/Softmax/Linear/激活先行，optimizer/RLA/offload 放后面）"——该迁移已执行（2026-09-27，见顶部横幅条目）。

## §8 方案（决策前草案）：把 IR-C 吸收进 dsl::compute——全节（原位置 docs/development/12-compute-engine-inventory.md §8.1–§8.10）
- 类型：否决方案
- 内容（精简，保留关键数字与结论）：2026-09-18 候选方案，最终未采用；2026-09-19 决定直接删除 IR-C。要点：
  - §8.1 用户入口 3 → 1：`dsl::compute/compute_reduce/compute_into`（保留）、`dsl::start_expr…end_expr(ExprBlock)`（仅 expr_dsl_test 用）、`engine.begin_expr…end_expr`（仅 FusedChainLayer + scan 用）——后两者删除；收敛不是靠"自动融合"，而是靠删除跨表达式融合能力。
  - §8.2 逃逸语义由用户负责的反例：LayerNorm（`compute_layer_mlp.hpp:499-510`）`normalized_cache_` 持有节点输出，自动内联会使 backward 静默算错 → 必须把逃逸检测变成框架不变量。
  - §8.3 可行性：`Tensor` 内持 `shared_ptr<MatrixT<P>>`（`compute_tensor.hpp:67-68`），拷贝构造时 `note_external_copy(tag)`，`end_expr` 时图外拷贝数 > 0 的节点 pin（禁止内联）。
  - §8.4 作用域方案：(a) `Layer::forward/backward` 加非虚壳（26+26 处，机械）建议；(b) `Model` 层 RAII（易漏）。
  - §8.5 自动 flush 点：非 DSL 引擎算子约 22 个 + 逃逸 + 作用域结束；`create_tensor`/`device` 不 flush。
  - §8.6 移除 `begin_expr/end_expr`、`FusedChainLayer`、`dsl::start/end_expr`；保留 `expr_graph.hpp` 为内部机制。
  - §8.7 代价：构建期 spec 数上升、未命中 AOT 硬报错、CPU no-op。
  - §8.8 第 0 步（删显式 API、保留 expr_graph.hpp）/ 第 1 步（自动作用域+逃逸检测，有真实需求再做）。
  - §8.9 接线可行性实测：融合条件（`expr_graph.hpp:180-211`：两节点无归约、同形状、非 vector_out、B 不含 matmul、tail 恰一个消费者且作 Linear 视图）逐层扫描，全部被三类阻挡——① 归约是硬边界（LayerNorm mean/var 两次 col_reduce、RMSNorm、Softmax、Attention m/l/W）；② eager 算子交错（RLA `rms_norm_backward_` 等，录制段内 eager 读到未写入的虚拟 buffer）；③ 中间量逃逸到 backward 缓存（`normalized_cache_`/`normed_cache_`/`residual2_cache_`/Attention `W_re`——缓存就是图外第二个消费者）。能融的地方早已写成单个表达式。三点结论：接线需 ~22 个 flush 点 + Tensor 拷贝钩子 + 26+26 非虚壳；当前无任何"安全且可融"的多表达式链；CPU 零收益。
  - §8.10 决策已执行（2026-09-19）：连 `expr_graph.hpp` 一起删（删显式 API 后它零调用方，保留即"悬空设计"）。删除清单：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`（接口+CPU/GPU 实现）、`GpuEngine::execute_fused_graph`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_`、`src/expr_graph_test.cpp`/`src/expr_fuse_test.cpp`、`scan_exprs` 的 FusedChainLayer dry-run 段。保留 `run_fused_gpu` 的 `output_override`（现服务 `dsl::compute_into`）。验证：重建通过，ctest 15/15。重新立项前提：出现"纯逐元素、同形状、中间量不必为 backward 保留、且长到单个 dsl::compute 写不下"的链，按逃逸检测+自动作用域+自动 flush 实现，不复活显式录制 API。

## 13 文首「正确性缺陷已在同轮修复」清单（原位置 docs/development/13-refactor-backlog.md 文首）
- 类型：已修复勘误
- 内容：v1.4.2 审查同轮修复：threadpool 确定性归约、`require_same_shape` Release 守卫、mnist optimizer batch 包裹、`submit()` 死代码删除、zipt abort 诊断、python 裸 except、过时注释清理（修复详情见 git log）。

## 13 §6 已核对为误报/已修复的审查项对照表（原位置 docs/development/13-refactor-backlog.md §6）
- 类型：否决方案 / 已核对误报
- 内容（整块搬移，重复立项前先读）：

| 审查项 | 核对结论（2026-09-25 读码） |
|---|---|
| BPE vocab_size 下溢（P1） | **已修复**：`domain_tokenizer_bpe.hpp:84` 有运行时守卫（`< 258` 返回 Error），注释 :81-83 即该守卫的说明，审查误读为"未加校验" |
| 反序列化无长度上限（P1-S3） | **已修复**：`model_serialization.hpp` `kMaxSerializedStringBytes`=64MiB，read_spec_header/read_tokenizer 双处校验；同步了过时 TODO 注释 |
| registry bin 丢 batch（P1-23/25） | **已修复**：`expr_registry.hpp:157/:198/:296/:342` 读写均含 batch（bin v5+）；`gen_fused.cpp:125/:150` 显式生成 |
| GPT generate 未包 begin_batch（P1-7/29） | **已修复**：`compute_layer_gpt.hpp:984-1099` fill_cache_ 与逐步生成均已包裹 |
| forward_sparse 越界 label 静默置 0（P2） | **非缺陷**：越界列 mask 同步置 0（`compute_loss.hpp:359/:383/:394`），loss/梯度/num_valid 全部排除该列 = 忽略语义；`ce_fusion_test.cpp:122-128` 有专项测试 |
| restart_on_device_lost exit(0)（P1-33） | 当前代码无此函数（`text_train.cpp:427` 的 exit(0) 是 `--help` 正常退出） |
| checkpoint 不存 Adam m/v（P1-34） | **属实但为功能缺口非缺陷**：`.bin` 无优化器状态段；需 MODEL_VERSION 升版 + 格式设计（见 §7） |
| ZiPT stored_tokens 无门控（P1-10） | **低优先**：clone 是 token IDs（(batch,seq)，量级 KB-MB），非激活张量；GPT/RAPT 同模式，门控需训练/推理模式标志，收益小 |

## 13 §9 已执行：代码缩减轮执行表（原位置 docs/development/13-refactor-backlog.md §9）
- 类型：演进记录
- 内容：依据全仓"悬空设计"审计（A/B/C 三档）+ activation offload 只读评估，逐项 build + ctest 验证（**20 → 19 个目标，19/19 全绿**）：① A 档死码全删（`elementwise_select_scalar_cond`、`axpy_inplace`、`broadcast_row_inplace`、`nn::one_hot`、`text_train::one_hot_labels`、`Matrix::multiply_transposed_add_to`、`nn::sigmoid·relu`、`ops::Sigmoid·ReLU`）；② 新增 `src/test_common.hpp`（`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛唯一副本，保留聚合编译 `#define main`）；③ RAPT `forward_step` 的 `num/den` 除法改 `dsl::compute(engine, leaf/(leaf+Scalar{1e-4}), rows, cols)`，与 `forward` 同构复用已注册 AOT 键；④ 算子收敛（删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`+`broadcast.comp` 全链、引擎 `row_reduce_max`、`offload_store/load`、`UnaryOp/BinaryOp/CompareOp`；引擎 virtual 58 → 49）；⑤ AST 移除（`algebra_expr.hpp`+`algebra_compute.hpp` 删除，`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`，CPU 求值只剩两套）；⑥ offload 评估（保留 A 组生产链、删 B 组 `offload_store/load`+`offload_primitive_test`+重复聚合器；`gpt_offload_test` 补 `begin_batch/end_batch`；CPU/ZiPT 由静默 no-op 改显式警告；帮助文本"互斥"更正为"可混合"）；⑦ 死 override：`TransformerEncoderLayer`/`ZiPTBlock` 的 `activation_cache()`（20 行）删除。随附五条教训保留在 13 §9 正文（现在式）。

## 08 §3.5 CUDA obj 不随 Debug/Release 重编（原位置 docs/development/08-pitfalls-and-lessons.md §3.5）
- 类型：已失效条目（CUDA 后端已整体移除，坑不再可能触发）
- 内容：Debug↔Release 切换后 `cuda_kernels.obj` 的 DEPENDS 只有 .cu，不重编 → `_ITERATOR_DEBUG_LEVEL` 不匹配（clang-cl=2 vs nvcc obj=0）→ lld-link 失败。修复：`-Xcompiler /D_ITERATOR_DEBUG_LEVEL=N` 对齐；切换配置后删 `build\cuda\cuda_kernels.obj` 再重编。教训（若恢复多编译器混合构建仍适用）：调试级宏必须在各编译器侧显式对齐。

## 08 §5 nvcc 行与 §7 CUDA 清单项（原位置 docs/development/08-pitfalls-and-lessons.md §5 表格、§7 修改构建/工具链）
- 类型：已失效条目
- 内容：① §5 表行"nvcc 12.8 与 MSVC 2026 (v14.51) 不兼容，cudafe++ crash | 构建（CUDA 已停用）| 用 VS 2022 BuildTools；`-allow-unsupported-compiler`"。② §7 "修改构建/工具链"两条：`[x] Debug↔Release 切换清 build/cuda/cuda_kernels.obj`（CUDA 移除后作废）、`[ ] nvcc 编译器版本匹配（12.8 ↔ VS 2022 BuildTools）`。

## 08 各条目的过程性验证/定位叙事（原位置 docs/development/08-pitfalls-and-lessons.md §2.5/§3.7/§4.10/§4.12/§6 模式 H）
- 类型：已修复勘误 / 过程叙事（教训本身保留在正文，现在式）
- 内容：
  - §2.5 验证流水：修复前 3 类 VUID 报错 → 强制校验层跑完整 `gpu_test`（256×256/10 迭代）0 条 VUID；`--gpu=AMD` 从"10s 超时失败"变 2s 通过；时间线 / `NN_VULKAN_NO_TIMELINE=1` 回退两路径 + ctest 15/15 全绿。
  - §3.7 修复验证：60000 训练 + 全量评估一轮跑通（`train_acc=89.95% test_acc=90.45%`，7.2s）。
  - §4.10 定位过程：最初怀疑 `generate_glsl_reduce` 内联 `col_broadcast` 下标算错，写了错误注释与工作区规避，多花一轮；打印生成的 GLSL 后证明 shader 正确，真因在 `run_fused_gpu` 的 `pc_base`（`raxis>=0 || has_mm ? 5 : 2` 把"归约无 matmul"算成 5，实际固定头 4）。
  - §4.10 复发叙事（2026-09-23，P-C1 fold 形态）：pc_base 有两处——写入侧 `run_fused_gpu` 与创建侧 `VulkanPipeline::create_generic` 的 `push_constant_size`；新增 fold（5 uint 头）只改写入侧 → `vkCmdPushConstants` 超 range 被驱动丢弃 → `fold_k` 读未定义残留，表现为"时对时错"（残留恰=K 造成假 PASS）；定案方法：K=1 打印全部行 CPU/GPU 实值。
  - §4.12 修复与红验证：恒等元 `0xff800000` → `0xFF7FFFFF`（对齐 CPU `lowest()`）；fold 测试 doc 分段边界改 `seq>EXPR_FOLD_BLOCK ? BLOCK+1 : seq/2`；对拍加 NaN 守卫；红验证——revert 修复后补边界仍绿，加 NaN 守卫才转红（`[FAIL] attn-fold doc seq=133 err=inf`）。
  - §6 模式 H 案例清单（2026-09 修复，均已完成）：`RAPTModel::clear_cache()` 曾清空 `token_emb_`；`RAPTModel` 三个模式开关曾是静默 no-op；`MaxPool2D::backward()` 曾在 `clear_cache()` 后越界读空 vector；`RAPTBlock` 曾用基类默认 `forward_recompute` → stride=2 静默用陈旧缓存；`RAPTModel::forward` 曾无法关闭文档感知。

---

# docs/development 02 · 03 · 05

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

---

# docs/development 01 · 04 · 06 · 07 · 10 · 14

> 本文件汇集 docs/development/01-compute-engine-development.md、04-memory-optimization.md、
> 06-rapt-algorithm.md、10-development-standards.md、14-f16-stable-gpu-loss-frozen.md
> 整改中移出的历史叙事（演进流水 / 已修复勘误 / 删除清单 / 性能 A/B / 排查过程）。
> 正文只保留当前状态、当前机制与仍然适用的教训；07-zipt-algorithm.md 为纯算法文档，无移出条目。

## 01 §5 引擎 row_reduce_max 删除注脚（原位置 docs/development/01-compute-engine-development.md §5）
- 类型：删除清单
- 内容：原文「⚠️ 2026-09 收敛：`row_reduce_max`（引擎算子）已删除——按行求最大走 DSL 归约叶子 `dsl::row_reduce_max(...)`（详见 `expr_dsl.hpp`；AGENTS §4.3 有完整删除清单）」。当前做法（按行求最大用 DSL 叶子、归约表达式走 `dsl::compute_reduce`）已保留为正文提示。

## 01 §6 逐元素/广播算子删除清单与理由（原位置 docs/development/01-compute-engine-development.md §6）
- 类型：删除清单
- 内容：`broadcast_row_inplace`、`broadcast_col_inplace`、`elementwise_unary`、`elementwise_binary`、`elementwise_binary_scalar`、`elementwise_select_scalar_cond`（连同 `UnaryOp`/`BinaryOp`/`CompareOp` 枚举）已全部删除（2026-09 收敛轮）。理由：这些算子在 DSL 落地后已无生产调用方，且与表达式能力完全重复（详见 `development/12-compute-engine-inventory.md` 顶部收敛横幅与 `13-refactor-backlog.md` §9）。

## 01 性能优化节旧 dsl::compute 示例（原位置 docs/development/01-compute-engine-development.md 性能优化 §2）
- 类型：已修复勘误（API 形式过期）
- 内容：原文用 lambda + 张量列表形式 `dsl::compute(engine, [](auto a, auto b, auto c){...}, {tensor_a, tensor_b, tensor_c}, rows, cols)`——该重载在代码中不存在。现 API 为 `dsl::compute(eng, expr, rows, cols[, precision])`（expr_dsl.hpp:1317），已按 `dsl::leaf(*a) + dsl::leaf(*b) * dsl::leaf(*c)` 形式改写。

## 01 Q3 旧答案：通过 UnaryOp/BinaryOp 枚举加激活函数（原位置 docs/development/01-compute-engine-development.md 常见问题 Q3）
- 类型：已修复勘误
- 内容：原文步骤为「1. 在 `UnaryOp` 或 `BinaryOp` 中添加枚举值；2. 在 `CpuEngine` 中实现逐元素运算；3. 在 `GpuEngine` 中实现 shader」——枚举与逐元素引擎算子已删除，该路径不存在。现行答案：在 Layer 内用 DSL 组合表达（`ReLU::forward` = `dsl::max(dsl::leaf(x), Scalar{0})`），引擎/ shader 不认识算法名（铁律 3）。

## 04 实施进度日期标注与 RAPT 接入旧叙事（原位置 docs/development/04-memory-optimization.md 背景与目标 §实施进度）
- 类型：演进记录 / 已修复勘误
- 内容：① 原节标题「实施进度（2026-08-22）」及各条目的日期戳（L1 续：RLA/RAPT 接入 2026-09-19）。② 「此前 `set_checkpoint_every`/`set_activation_offload`/`set_flush_interval` 对 RAPT 是静默 no-op，而 CLI 会打印"已启用"」——该缺陷已于 2026-09-19 修复，当前 RAPT 三开关真实生效。③ 「`activation_cache()` 补齐此前遗漏的 4 项（否则 offload 覆盖不全）」——当前已列出全部 7 项层内缓存 + 4 个子层缓存。④ L2 条目中「（CPU no-op；CUDA 已停用）」的 CUDA 提法——CUDA 后端已整体移除（96a3675），当前只写 CPU no-op。⑤ 「新增 rapt_offload_test」「text_train 新增 --checkpoint-every」等"新增"措辞。

## 04 基准显存 27G vs 29G 的融合 A/B（原位置 docs/development/04-memory-optimization.md 背景与目标 §基准配置与峰值显存）
- 类型：性能 A/B
- 内容：原文「该配置下训练峰值显存 ≈ 27 GB（已含 M5/M6 融合收益），此前 29 GB（融合前）」；「融合（M4/M5/M6）已把非线性象限（seq² 注意力、vocab×seq 全 softmax）从 ~3× 砍到几近为 0」。现正文只保留 27GB 当前口径与"非线性象限不物化 ≈0"的当前状态。

## 04 f16 红线修订注脚（原位置 docs/development/04-memory-optimization.md 现状与根因 §数值精度）
- 类型：演进记录
- 内容：原文「（立项时口径……）（历史红线"不引入 f16"已随混合精度修订。）」。当前事实：本文档数字为 fp32 基线；v1.2.0 起 f16 混合精度见 05-mixed-precision，本文档显存账不适用于 f16 路径。

## 04 L2 旧问题叙述：内存池永不归还（原位置 docs/development/04-memory-optimization.md 现状与根因 §内存池碎片化）
- 类型：已修复勘误
- 内容：原文「但：从未将整个空 Block 归还 GPU（`blocks_.clear()` 仅在析构时触发），block 底材按需 128MB（或超尺寸单块）申请后不回收 ⇒ 峰值生命周期等于整个进程/测试生命周期」；「算子融合文档将"内存池 first-fit 碎片化 + 永不归还"列为独立跟踪项、不随融合解决」。该问题已由 L2 的 `release_idle_blocks()` / `release_idle_pool_blocks()` 解决（step 边界整块归还）；残余风险（step 内仍累积）保留为正文。

## 04 注意力两趟式旧实现描述（原位置 docs/development/04-memory-optimization.md 现状与根因 §注意力形态）
- 类型：演进记录
- 内容：原文「原 `batched_matmul_reduce/max → denom → apply` 两趟式 forward 与 `batched_matmul_softmax_backward_q/kv` backward 重算 W 方案，已被 IR 融合替代（算子融合文档 S7；2026-09-23 起 forward 进一步换单 fold kernel——FoldSpec 分块流式……）」。当前状态（单 fold kernel、S 不落显存、backward 为 R/X 表达式 + batched_matmul）已保留为正文。

## 04 「已核对（2026-08-24）」审查记录框（原位置 docs/development/04-memory-optimization.md L2 节）
- 类型：演进记录（结论仍适用，已改写为常设约定）
- 内容：原文以「已核对（2026-08-24）：中间 Tensor 的归还路径是安全的，无需修改即可维持正确性」开头的审查记录。四条安全约定（延迟销毁 + 录制期全内存屏障、复用前必写满、新增原语必须插 output barrier、期望为 0 必须显式 `zero()`）全部保留为正文「内存池复用安全约定」。

## 04 旧落地顺序与过期约束（原位置 docs/development/04-memory-optimization.md 落地顺序建议 / 与既有约束的关系）
- 类型：演进记录 / 已修复勘误
- 内容：① 原落地顺序「1. 先 L1……2. 再 L2……3. 若仍不足 → 立项 L3……」（L1/L2 已完成，现正文改为"L1/L2 已就位，下一步立项 L3"）。② 「用训练 step 耗时采样替代（bench 工具已移除）」的 bench 删除提法。③ 原约束「不引入 f16（bf16/fp16）低精度训练」——已过时（f16 混合精度已落地），改为「L1/L2 与精度设置正交，本文档账目为 fp32 口径」。

## 06 §4 引擎化改造流水（原位置 docs/development/06-rapt-algorithm.md §4 开头）
- 类型：演进记录
- 内容：原文「2026-09-04 的引擎化改造删掉了 Layer 内的 `scan_forward_/scan_backward_` 纯 CPU 标量循环（PCIe 往返），改为 3 个 op-level 扫描原语，由 Layer 用原语 + 逐元素原语组合表达算法（铁律 3：shader 永不含算法）」。当前状态保留为正文。

## 06 §4.3 坑 3/坑 4 的历史措辞（原位置 docs/development/06-rapt-algorithm.md §4.3）
- 类型：已修复勘误
- 内容：① 坑 3 原文「写错则 doc-aware gradcheck 才会暴露（历史已踩过）」。② 坑 4 原文「不能让空张量流进 matmul（旧行为 = 静默算出垃圾梯度）」——旧行为已修复，现行规则（缺缓存立刻硬报错）保留为正文。

## 06 §4.4 验证基线的演进措辞与失效交叉引用（原位置 docs/development/06-rapt-algorithm.md §4.4）
- 类型：已修复勘误 / 断链修正
- 内容：① 原标题「验证基线（2026-09-04，GTX 850M）」与表内「max_err 与改造前基线一致」的对照措辞（现直接列 max_err 基线数值）。② 原注「6 处 Vulkan 接线细节沉淀在 `01-compute-engine-development.md`」——01 号文档并无该清单，属失效交叉引用，改为直接指向 `compute_gpu_engine.hpp` 与 `backend/compute_vk_backend.hpp`。

## 06 §5 历史：两趟式注意力（已删除）——全节（原位置 docs/development/06-rapt-algorithm.md §5）
- 类型：删除清单 / 演进记录
- 内容：S7 的"两趟式多 kernel 注意力"（forward 3 个融合 kernel m→l→W × 4 种掩码变体、backward 3 个，W 物化供复用）已删除——2026-09-24 起注意力 forward 为单 fold kernel 分块流式（QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 矩阵绝不物化）。与传统 Flash Attention 的对照分析（kernel 数量、IO 复杂度、掩码扩展方式）原注「见 git 历史中本节原文」。保留的结论（当前事实）已改写为 §5 正文：本项目不存在单一 flash-attn kernel、fold 流式实现 FlashAttention 核心思想、掩码变体经 FoldSpec 登记漏登记硬报错。

## 06 §6 RAPT 开关与已修前置缺陷（原位置 docs/development/06-rapt-algorithm.md §6）
- 类型：已修复勘误
- 内容：① 「（2026-09-19 补齐；此前 RAPT 侧是静默 no-op 而 CLI 会打印"已启用"）」。② 「已修的两个前置缺陷（不修则静默毁模型）：1. `RAPTModel::clear_cache()` 曾清空 `token_emb_`（模型参数）——检查点每块 backward 后都调它，等于毁掉词嵌入。2. `RAPTModel::forward` 曾只在 `doc_ids` 非空时下发文档 id → `set_doc_ids({})` 关不掉文档感知，跨 step 残留边界重置（跨样本串扰）」。③ 「`activation_cache()` ……（Q_normed/K_normed/Q_rms_inv/K_rms_inv 曾漏列 → offload 漏搬，已补）」。④ §3 表格「分数定义：推理/训练曾不一致」措辞（改为"原版缺陷"）。

## 06 §4.2 forward_step 旧原语调用（原位置 docs/development/06-rapt-algorithm.md §4.2）
- 类型：已修复勘误（API 过期）
- 内容：原文「先更新状态（`batched_matmul(V, Kp, transB)` → `add_inplace(B_state)`；`add_inplace(z_state, Kp)`）→ num = B·q'（batched_matmul）」——2026-09-27 计算类原语迁 DSL 后，Layer 不再直调 `batched_matmul`/`add_inplace`：现为 `dsl::matmul(V, Kp, false, true, H)` + `dsl::compute_into` 原地累加（compute_layer_rapt.hpp:832-848），已按代码现状订正。

## 10 compute::apply / 代数 AST 移除注脚（原位置 docs/development/10-development-standards.md §2.6、§4.3.1、§4.3.2）
- 类型：已修复勘误
- 内容：三处注脚原文——「（旧 `compute::apply(span, expr)` 代数 AST 已于 2026-09 整体移除）」「注：旧 compute::apply 代数 AST 与广播/逐元素算子在 2026-09 已移除」「（旧 compute::apply 代数 AST 已于 2026-09 移除）」。旧规则表表述「Layer 通过 Matrix 语义 API 或 `compute::apply(span, expr)` 表达算法」「上层通过 Matrix API 或 compute::apply 间接享受并行加速」已改为表达式 DSL。

## 10 目录树中的已删文件行（原位置 docs/development/10-development-standards.md §5.1）
- 类型：删除清单
- 内容：目录树原有一行「├── （algebra_expr.hpp / algebra_compute.hpp 已于 2026-09 移除）」——文件树只列当前存在的文件，该行已删（`algebra_expr.hpp`/`algebra_compute.hpp` 确已不存在；`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`）。

## 10 §6.2.4 SmartPolicy 历史说明（原位置 docs/development/10-development-standards.md §6.2.4）
- 类型：已修复勘误
- 内容：原文「历史说明：早期文档把该策略命名为「SmartPolicy」并给出用户可替换 policy 的扩展示例，该类型在代码中从不存在（只有上述自由函数 + 阈值门控），示例已按实际 API 更正」。当前事实（无 policy 类型，只有 `parallel_for_samples`/`parallel_for_blocks` + 阈值门控）保留在代码注释形式的正文中。

## 10 过期 API 示例批量订正（原位置 docs/development/10-development-standards.md §2.5、§3.2.4、§4.2.2、§4.3.1、§4.3.4、§5.3.1、§5.3.2、§5.4、§10.2）
- 类型：已修复勘误（API 现状订正；原内容与代码现状不符，非单纯演进注脚）
- 内容：① §2.5 「遵循隔离」示例 `matrix.apply_relu()`——`Matrix` 无此方法且违反 §2.6 自身铁律（Matrix 不写算法），改为 `add_inplace`/`multiply_to`。② §3.2.4 「避免使用已废弃的 `data_ptr()`」——接口只有 `span()`（algebra_matrix.hpp:194 注明 span 替代所有 data_ptr 场景）。③ §4.2.2 正确示例原为 `compute::apply(x, max(x, Scalar{0}))` 与 `p.binary_apply_inplace(g, lambda)`——分别改为 `dsl::compute` 与 `dsl::compute_into`（与 compute_layer_mlp.hpp ReLU::forward、compute_optimizer.hpp SGD::step 一致）。④ §4.3.1 Matrix API 列表原含 `apply`/`binary_apply`/`binary_apply_inplace` 模板——三者已不存在（`Matrix::detail::apply/binary_apply/binary_apply_inplace` 死函数删除），只留 `reduce/row_reduce/col_reduce/span`。⑤ §4.3.4 「Adam::step 通过 binary_apply_inplace 表达」——现为三个 DSL kernel（K1/K2 dsl::compute + K3 compute_into，RParam 承载偏置校正，compute_optimizer.hpp:304-353）。⑥ §5.3.1 链式 fluent 示例 `model.add<Linear>(784,64).add<ReLU>()...add<CrossEntropyLoss>()`——`Model::add` 实际返回 `Result<void>`（不可链式），`CrossEntropyLoss` 不是 Layer 不可 add；§5.3.2 原工厂示例返回 `Model&`——均改为实际实现。⑦ §5.4 检查清单「是否支持链式调用？」同步改为 Result 传播检查项。⑧ §10.2 Doxygen 示例 `@throws std::invalid_argument`——与全项目禁异常冲突，改为 Error 返回说明。

## 10 §7.2 C++ 特性表过期行（原位置 docs/development/10-development-standards.md §7.2）
- 类型：已修复勘误
- 内容：删除三行未使用特性——`std::ranges`（原注「如 ranges::generate Xavier 初始化」，全仓无 ranges::generate）已改为实际用途（ranges::distance/advance 切分并行区间、ranges::sort）；`std::views::zip`（原注「优化器参数更新」，全仓无 views::zip——优化器用 dsl::compute_into 逐参数处理）；`std::execution::par_unseq`（原注「并行执行策略（矩阵运算）」，全仓无 std::execution，core_config.hpp:64 明言替代 std::execution::par）。`std::expected`/`std::span`/`std::views::iota` 三行保留（经核实仍在使用）。

## 10 不存在的头文件引用（原位置 docs/development/10-development-standards.md §8.3、§9.1）
- 类型：已修复勘误（断链修正）
- 内容：① §9.1 标题「基础类型定义（nn_core_config.hpp）」——仓库无 `nn_core_config.hpp`，`Error`/`Result` 定义在 `core_errors.hpp:15-20`。② §8.3 包含顺序示例 `#include <neuralnet.cpp/nn_core_config.hpp>`（同上）与第三方库 `#include <fmt/format.h>`（全仓无 fmt 依赖）。③ §5.2.1 注释中「layer.hpp - 层定义」——现名 `compute_layer.hpp`。

## 14 §4 回归窗口与历史脉络（原位置 docs/development/14-f16-stable-gpu-loss-frozen.md §4）
- 类型：演进记录（排查过程流水）
- 内容：① 09-24 16:54 bench 原始数据（早于 75255da）同配置行为「epoch1 loss 活（7.40）→ epoch2 发散 -nan」（该配置历史上就不可训练，§12.5 stable=F16 时 CE 链 ~200 步 NaN，但 loss 至少是活的）；② 09-26 HEAD `4a12876` 症状变为「epoch1 起即静默冻结」，「冻结比 NaN 更危险」的评述；③ 嫌疑提交清单（`git log 75255da..HEAD`：`eba744d` CPU f16 修复、`200955f` transpose 派发、`3c2b9e8` f16 MSVC 修复、`4c8c58c` 审查修复、`4a12876` 技术债清理；基线若追到 09-24 则 `b4ee2df`/`4cd0616` 也在窗口内）。与 issue #13 P0-② 的对照表为当前有效的事实区分，保留在 §4 正文。

## 14 证据行号订正（原位置 docs/development/14-f16-stable-gpu-loss-frozen.md §3、§5、§6）
- 类型：已修复勘误（数字订正）
- 内容：报告引用的源码行号整体偏移 1–7 行，已按 HEAD `4a12876` 工作区实况订正：`text_train.cpp` 1513→1514（forward_sparse_sum）、1522→1523（cast）、1571→1572（submit_scalar_readback）、1366→1367（harvest 打印）、582→583（实验性注释）；`compute_gpu_engine.hpp` 150→146（submit_scalar_readback 定义）；`f16_precision_test.cpp` 642-649→639-646（profile 单字段矩阵）、698-702→691-700（main() CPU 块 + NN_F16_DEBUG 门）。缺陷本身未修（HEAD 仍为 4a12876，无修复提交）。

---

# docs/introduction · usage · benchmarks

> 本文件收录 docs/introduction/、docs/usage/、docs/benchmarks/ 整改中移出的
> 「旧状态 / 演进过程 / 已修复勘误 / 删除清单」内容（2026-09-27 摘录）。
> 这些内容只作历史存档，不再是这些文档的正文。

---

## CUDA 后端移除备注（原位置 docs/introduction/01-architecture.md 篇末「备注：CUDA 后端已移除」整节）
- 类型：演进记录 / 删除清单
- 内容：CUDA 后端已整体移除——`cuda/` 目录（`cuda_kernels.cu` / `.h` / `CMakeLists.txt`）、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`，以及全库 `NN_HAS_CUDA` 条件分支（`compute_tensor.hpp` 6 处、`nn.hpp`、`cli/cli_engine_factory.hpp`、`cli/cli_train_common.hpp`、`CMakeLists.txt`）均已删除；移除原因是融合原语（M4/M5/M6）与 DSL 表达式在 CUDA 上未实现且无真实回退，属「文档声称支持但实际损坏」的死代码；CLI 不再接受 `--cuda`（传入报「未知参数」）；带 CUDA 的历史快照在 git 分支 `legacy/cuda`。当前事实（正文保留）：引擎只支持 `CpuEngine` 与 Vulkan `GpuEngine` 两个后端。

## 旧代数 AST 删除行（原位置 docs/introduction/01-architecture.md L1 模块表）
- 类型：删除清单
- 内容：`~~algebra_expr.hpp~~ / ~~algebra_compute.hpp~~ —— 🗑️ 2026-09 已移除：旧代数 AST 与 `compute::apply`；`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`（另见头文件依赖图括注「旧 algebra_expr / algebra_compute 已于 2026-09 移除」）。

## 引擎算子收敛横幅（原位置 docs/introduction/01-architecture.md L2 表下 blockquote）
- 类型：删除清单
- 内容：**2026-09 收敛**：`axpy_inplace`、`broadcast_row_inplace/col_inplace`、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、引擎 `row_reduce_max`、`offload_store/load` 与 `UnaryOp/BinaryOp/CompareOp` 枚举已全部删除（逐元素/广播/条件选择一律走表达式 DSL），引擎 virtual 58 → 49；逐项清单见 `development/12-compute-engine-inventory.md` 顶部收敛横幅。

## IR-C 块式融合移除括注（原位置 docs/introduction/01-architecture.md 表达式入口表 `expr_dsl.hpp` 行）
- 类型：演进记录
- 内容：「（`start_expr/end_expr` 块式融合已于 2026-09-19 随 IR-C 移除）」。

## 「不再有 forward_gpu / backward_gpu」表述（原位置 docs/introduction/01-architecture.md §1 引擎化架构）
- 类型：演进记录
- 内容：原文「Layer 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配 CPU/GPU。不再有 forward_gpu / backward_gpu。」——已改写为当前事实「每个 Layer 只有一份与后端无关的 forward/backward 实现」。

## SmartPolicy 旧称（原位置 docs/introduction/02-performance.md §1 标题）
- 类型：演进记录
- 内容：标题曾为「自适应并行策略（旧文档称「SmartPolicy」）」。

## 线程池 latch 的旧版问题叙事（原位置 docs/introduction/02-performance.md §2）
- 类型：演进记录
- 内容：「旧版问题」：每次 `submit` 需要构造 `shared_ptr<packaged_task>`（堆分配）、获取 `future`（同步开销）、加锁入队（互斥开销）；「新版优化 / 性能收益」对比：旧版 N 分块 = N 次堆分配 + N 次加锁 + N 个 future 同步，新版 N 分块 = 1 次加锁 + 0 次堆分配 + 1 个原子计数器。正文已改为「设计目标 / 当前实现 / 开销构成」三段当前事实。

## MSVC /fp:fast 早期标志记录（原位置 docs/introduction/02-performance.md §4 编译标志注）
- 类型：已修复勘误
- 内容：原文记「MSVC 分支为 `/O2 /fp:precise`——早期为 `/fp:fast`，但其禁用 NaN 语义与本铁律冲突，实测 `precision_type_test` 的 NaN 断言在 `/fp:fast` 下被编译器折叠为假失败，故对齐为 precise」；同时权威行号由 `CMakeLists.txt:74/76` 更正为 `77/79`。正文保留的理由（`/fp:fast` 与 NaN 铁律冲突）已改为当前事实陈述。

## 已删除 eager 算子的示例代码（原位置 docs/introduction/02-performance.md §5.1）
- 类型：删除清单
- 内容：批量提交示例中的 `engine.elementwise_binary(...); // 不立即执行`——`elementwise_binary` 已不在引擎接口中，示例已改为 `engine.matmul` + `engine.transpose`。

## 单算子级广播/融合原语删除条目（原位置 docs/introduction/02-performance.md §5.3 第二条）
- 类型：删除清单
- 内容：「~~`broadcast_row_inplace` / `broadcast_col_inplace`、`axpy_inplace`、`elementwise_select_scalar_cond`~~：2026-09 已全部删除——单算子级广播/融合被表达式级融合取代（`dsl::row_broadcast` / `dsl::col_broadcast` / `dsl::select` / `dsl::compute_into`）」。

## 注意力批量化的历史写法（原位置 docs/introduction/02-performance.md §8）
- 类型：演进记录
- 内容：小节标题「旧版：per-head 循环」与括注「（P-C2-7 起 forward 进一步收敛为 `eval_expr(fold)` 单 dispatch……）」——正文已改为「朴素写法（不批量化时的形态）/ 当前做法（forward 单 fold kernel + backward 批量 dispatch）」。

## 因果掩码缓存历史设计（原位置 docs/introduction/02-performance.md §9 整节）
- 类型：演进记录 / 删除清单
- 内容：「因果掩码缓存【历史：已随 fold 迁移删除】」全节——状态横幅「P-C2-7，2026-09-23：掩码物化与缓存路径整体删除，现行单 fold kernel 在 body 内以 select 链表达掩码（`tri_skip` 把被屏蔽块整块钳成空转），掩码矩阵从不存在、无需缓存；示例中 `(batch << 16) | seq_len` 位打包键正是 `08-pitfalls-and-lessons.md` §3.3 记录的溢出缺陷写法，该缺陷随整段删除一并消失」；「问题（历史）」GPT 每个 forward 需创建因果掩码矩阵 `(batch*H*seq, seq)`；「优化（历史）」曾以 `mask_cache_` + `ensure_mask(engine, batch, seq_len)` 按 `(batch, seq_len)` 键缓存掩码矩阵（完整代码见 git 历史 `compute_layer_attention.hpp`）；「收益（历史）」相同 `(batch, seq_len)` 只构造一次掩码。正文现只保留当前事实「掩码不物化、无缓存」。

## 旧代数 AST 括注（原位置 docs/introduction/02-performance.md §12 标题）
- 类型：删除清单
- 内容：标题「表达式模板（DSL；旧代数 AST 已于 2026-09 移除）」。

## 优化效果总结表中的演进条目（原位置 docs/introduction/02-performance.md 末表）
- 类型：演进记录
- 内容：「掩码/编码缓存 | GPT 训练 | 避免重复构造（掩码物化缓存已随 fold 删除，见 §9）」与「GPU 注意力 fold 流式（2026-09-24） | GPU mha/causal | fwd 7.9→5.41ms（−31.5%）、causal −40%、train −13%（40HX 交错 bench）」——正文现为「位置编码缓存（掩码不物化、无需缓存，见 §9）」与绝对值「fwd 5.41ms（mha）/ 4.75ms（causal）、train 17.5ms」。

## dsl::compute 缺 rows/cols 的示例（原位置 docs/introduction/03-algorithm-reference.md §2 ReLU）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为 `dsl::compute(engine, dsl::max(dsl::leaf(x), 0))` 与 `dsl::compute(engine, dsl::select(dsl::leaf(x) > 0, dsl::leaf(grad_out), 0))`（2 参重载不存在）。核对结果：`dsl::compute(ComputeEngine&, const E&, std::size_t rows, std::size_t cols, Precision P = F32)`（`expr_dsl.hpp:1318-1320`），必带输出形状。已补 `rows, cols` 并把字面量标量改为 `Scalar{0}`（与 `compute_layer_mlp.hpp:194/209` 的真实写法一致）。

## LayerNorm 典型用途归约方向（原位置 docs/usage/03-compute-engine-usage.md 归约节「典型用途」）
- 类型：已修复勘误（与代码不符，已订正）
- 内容：原文「LayerNorm: 归一化每行 → `row_reduce_sum(*x)` + `1.0f / x->cols()`」。核对结果：LayerNorm 沿特征维（行）求均值、每列一个均值，真实实现用 `col_reduce_sum` 且除以 `F = rows`（`compute_layer_mlp.hpp:492-511`）。已改为 `col_reduce_sum(*x)` + `1.0f / x->rows()`。

## Linear 实现的旧写法（原位置 docs/introduction/03-algorithm-reference.md §1）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：forward 曾记为「一行原语：`out = engine.matmul_with_bias(W, x, b)`」；backward 曾记为「`grad_input = engine.matmul(W, grad_out, /*transA=*/true)`；`grad_W ← dsl::compute_into(leaf(grad_W) + matmul(grad_out, x, transB))`；`grad_b += engine.accumulate(engine.row_reduce_sum(grad_out))`」。当前代码（`compute_layer_mlp.hpp:99-166`）为：forward = `dsl::compute(engine, dsl::matmul(w_, input) + dsl::row_broadcast(b_), rows, cols, p_.compute)`；backward = `dsl::compute(dsl::matmul(w_, grad_output, true, false))` + `dsl::compute_into(leaf(grad_w_) + dsl::matmul(grad_output, input_cache_, false, true))` + `dsl::compute_reduce(row_reduce_sum)` 后 `compute_into` 累加。

## MHA 历史路径括注（原位置 docs/introduction/03-algorithm-reference.md §6）
- 类型：演进记录
- 内容：「（历史路径为 `rearrange_3d` + `batched_matmul` 把 H 个 per-head matmul 融合为 1 次 batch dispatch，该结构现仅存于 backward。）」与小节标签「Forward（P-C2-7 单 fold 路径……）」「S7 起折进 Q 免独立 pass」——正文已改为当前事实（backward 使用该批量结构；缩放随表达式融合）。

## 掩码缓存删除括注（原位置 docs/introduction/03-algorithm-reference.md §12 CausalSelfAttention）
- 类型：删除清单
- 内容：「——没有 `S += mask` 矩阵，**也没有掩码缓存**（历史物化式掩码缓存已随 fold 迁移删除）」。

## 优化器篇 eager 原语删除横幅（原位置 docs/introduction/03-algorithm-reference.md Optimizer 篇开头）
- 类型：删除清单
- 内容：「不再直调 `axpy_inplace`/`elementwise_*` 这类 eager 原语（这些原语已于 2026-09 整体删除，引擎现有 49 个 virtual）」。

## Muon 旧算法描述与旧 DSL 写法（原位置 docs/introduction/03-algorithm-reference.md §5）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：原文步骤为「1. SGD-Momentum: v ← μ·v + g；2. update = NS_5(v)；3. p ← p − η·update」，DSL 用 `engine.matmul(A, A)`、`compute_into(leaf(A)*rparam(b)+...)`（缺 engine/dst 实参）。当前代码（`compute_optimizer.hpp:658-717`）含 Nesterov 方向（`update = g + μ·v`）、形状缩放 `p ← p − η·0.2·√max(m,n)·NS(update)`（NorMuon），且 NS 母矩阵取短边（m≤n 行正交化 / m>n 列正交化）；示例已按 `dsl::compute` / `dsl::compute_into` 完整签名订正。

## 总览表演进措辞（原位置 docs/introduction/04-innovative-designs.md §0）
- 类型：演进记录
- 内容：「不物化 `O(seq²)` 分数矩阵（原两趟式已演进）」「把显存峰值从 29GB 压到 27GB（并持续下探）」——正文现为「不物化 O(seq²) 分数矩阵」与「GPT 训练峰值显存压至 ~27GB 量级」。

## CUDA 停用括注（原位置 docs/introduction/04-innovative-designs.md §1.1）
- 类型：演进记录
- 内容：「自动适配 CPU / GPU 双后端（CUDA 已停用）」。

## key 确定性铁律的事故叙事（原位置 docs/introduction/04-innovative-designs.md §2.3）
- 类型：已修复勘误
- 内容：「这条铁律来自一次真实事故：C++ 实参求值顺序未指定曾导致 key 跨编译器不稳定。」——正文改为当前警告「C++ 实参求值顺序未指定是这类漂移的常见来源，属必须规避的坑」。

## 三个 matmul 融合原语整节（原位置 docs/introduction/04-innovative-designs.md §3.3）
- 类型：删除清单
- 内容：「三个 matmul 融合原语【历史：已随 S7 删除】」全节——为承载当年两趟注意力而新增的通用原语：`batched_matmul_reduce`（matmul 后沿输出维度归约，不物化 `A·B`）、`batched_matmul_softmax_denom`（减行 max → exp → 按列求和）、`batched_matmul_softmax_apply`（行 softmax 归一化后与 V 相乘累加、逐 tile 流式）；反向曾有 `..._softmax_backward_q` / `..._softmax_backward_kv`（kernel 内重算权重矩阵）。正文 §3.3 已改写为当前事实「matmul 参与表达式融合（`dsl::matmul` 折叠为 ExprSpec 前置 matmul 段）」。

## 原两趟式注意力方案整节（原位置 docs/introduction/04-innovative-designs.md §4.3 与 §4.4 前半）
- 类型：演进记录 / 删除清单
- 内容：「【历史】原两趟式方案（已删除）」——Forward：`m = max of QᵀK`（bmm_reduce）→ `l = Σ exp(QᵀK − m)`（bmm_denom）→ `O = W·V` 逐 tile（bmm_apply），三个原语已随 S7 删除；Backward：反向重算 W（`bmm_softmax_backward_q/kv`，已删）；§4.4 旧句「fold 化后连原两趟式『用计算换显存』的 2× QKᵀ 代价也消失——forward 的 QKᵀ 只算一遍」；以及 §4 标题「（fold 单遍流式；原两趟式）」与 §4.2「现行方案（P-C2-7，2026-09-23）」标签。

## 稀疏 CE 旧原语括注（原位置 docs/introduction/04-innovative-designs.md §5.2）
- 类型：删除清单
- 内容：「原为两个 op-level 原语（`col_softmax_denom` / `col_softmax_sparse_forward`，**已随 S7 删除**），现行由 IR 结构表达……」。

## IR 优化动机的「无 pass 时代」叙述（原位置 docs/introduction/04-innovative-designs.md §6.1）
- 类型：演进记录
- 内容：「`ExprSpec` 已是事实上的轻量 IR……但没有优化 pass，直接导致真实问题：子表达式 `grad*gamma` 重复 3 次 → 超 `EXPR_MAX_INPUTS=8`，**被迫**手工拆表达式」——正文改为当前事实（不做优化会带来这些问题，IR-A/IR-B pass 解决）。

## IR-C 移除条目（原位置 docs/introduction/04-innovative-designs.md §6.2）
- 类型：演进记录
- 内容：「**IR-C 图 IR + 融合分析**：~~`begin_expr/end_expr` 录制虚拟寄存器 DAG；逐元素链拼接~~ → **已评估并整体移除（2026-09-19）**……」——正文改写为「不采用」+ 指向 `docs/development/03-ir-optimization.md` §5.3 的当前表述。

## f32-only 红线修订记录（原位置 docs/introduction/04-innovative-designs.md §7「设计红线（立项时基线，v1.2.0 起修订）」）
- 类型：演进记录
- 内容：「~~全程**不引入 f16/bf16**（数值统一 fp32）~~——**已修订**：v1.2.0 起引入 f16 混合精度（`Precision` 类型系统 + `PrecisionProfile`，compute/stable 档可保 fp32；BF16 仍为『使用即报错』的保留值）。fp32-only 是立项时的取舍记录，不再是现行红线。」正文现为「精度红线（当前）」段。

## 自适应并行与线程池的旧版对比（原位置 docs/introduction/04-innovative-designs.md §8.1 / §8.2）
- 类型：演进记录
- 内容：「实测在 ~512K 元素处首次稳定 >1.5x，**此前串行更优**」；「**旧版**每任务 = N 次堆分配 + N 次加锁 + N 个 future 同步；**新版** = 1 次加锁 + 0 次堆分配 + 1 个原子计数器」。

## 算子收敛与掩码缓存删除条目（原位置 docs/introduction/04-innovative-designs.md §8.4）
- 类型：删除清单
- 内容：「单算子级融合原语（`axpy_inplace` = clone+scale+add 三步并一步、`elementwise_select_scalar_cond` = 条件选择）——**2026-09 已随算子收敛整体删除**」；「多头注意力批量化……（历史：`rearrange_3d → 单次 batched_matmul → 转回` 把 H 次融为 1 次，该结构现仅存于 backward）」；「因果掩码物化缓存**已随 fold 迁移删除**……位置编码缓存保留」；以及 §4.2 的「`tri_skip`（原名 `causal_skip`）」改名记录。

## 收益一览表的演进条目（原位置 docs/introduction/04-innovative-designs.md §12）
- 类型：演进记录
- 内容：「fold 流式注意力（**原两趟**）」「算子融合 + 显存体系 | GPT 训练峰值 **~29GB → ~27GB**（并持续下探）」「注意力批量化 | H 次 matmul → 1 次 batched_matmul（**现为** fold 单 kernel 单 dispatch）」「融合 axpy | 每 step **减少** ~600 次 GPU buffer 分配」。

## add_linear / add_relu 链式构建示例（原位置 docs/usage/01-quickstart-model.md 三件套、方式一、方式三、完整前向示例、陷阱 #2；docs/introduction/01-architecture.md 理解路线图第四步）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文用 `model.add_linear(784, 256).add_relu().add_linear(256, 128)` 链式构建，并称「方式一：链式构建（推荐）」「方式三：模板 add（自定义层）——`add<nn::Linear>` 等价于 `add_linear`」。核对结果：`Model` 中**不存在** `add_linear` / `add_relu`（全仓 grep 仅命中 AGENTS.md 与文档）；唯一构建入口是 `template <typename LayerType, typename... Args> Result<void> add(Args&&...)`（`model_container.hpp:71-80`），返回 `Result<void>`、**不返回 `*this`，不可点链调用**，构造后自动 `Layer::init(engine)`。示例已全部订正为 `model.add<nn::Linear>(784, 256);` 逐行写法。

## 自定义 Layer 旧示例签名（原位置 docs/usage/01-quickstart-model.md「自定义 Layer」）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为「构造函数 `MyLayer(ComputeEngine& engine, in_dim, out_dim)` 内初始化权重」+ `nn::Matrix w_cpu = nn::Matrix::random(out_dim, in_dim); // 假设有此方法` + `std::vector<nn::Tensor*> parameters() override { return {&weight_}; }`。核对结果：`Layer::parameters()` / `param_gradients()` 返回 `std::vector<TensorRef>`（`TensorRef = std::reference_wrapper<Tensor>`，`compute_tensor.hpp:338`）；`Matrix::random` **不存在**；权重初始化契约是 override `init(ComputeEngine&)`（`Linear` 即如此，`compute_layer_mlp.hpp:46-76`）。示例已按当前契约重写。

## 陷阱 #3「to_matrix 错误用法」（原位置 docs/usage/01-quickstart-model.md 常见陷阱）
- 类型：已修复勘误（原描述与代码不符）
- 内容：原文「❌ GPU 张量调用 `to_matrix` → 错误：tensor is not GPU；✅ 先下载到 CPU → `auto m = engine.to_matrix(t)`（错误信息会告诉你）」——实际 `to_matrix(const Tensor&, Precision)` 就是统一下载入口（`compute_engine.hpp:177`），不存在该报错；真正的跨设备错误是绕过 engine 直接取 `tensor.cpu_matrix()`（`NN_ASSERT: tensor has no CPU storage`）。示例已订正。

## 陷阱 #4 旧调用签名（原位置 docs/usage/01-quickstart-model.md 常见陷阱）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为 `model.backward(engine, grad)` 与 `model.zero_grad(engine)`、`optimizer.step()`（非指针）。核对结果：`Model::forward/backward/zero_grad` 均**不带 engine 参数**（`model_container.hpp:178/196/249`），`Optimizer::step/zero_grad` 也不带（`compute_optimizer.hpp:117/182`）；`Loss::forward` 需要 engine。

## 训练流程时序图中的 engine 实参（原位置 docs/introduction/01-architecture.md「完整训练流程」mermaid 图）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原图写 `U->>M: zero_grad(engine)`、`U->>M: backward(engine, grad)`、`U->>M: forward(engine, test_input)`、`O->>E: scale_inplace / add_inplace / ...`。核对结果：`Model::zero_grad()` / `Model::forward(const Tensor&)` / `Model::backward(const Tensor&)` 均不带 engine（`model_container.hpp:178/196/249`，engine 在 Model 构造时绑定；只有下推到 `Layer::forward(engine, tensor)` 时才带）；优化器步进实为 `dsl::compute_into(engine_, ...)`。已按当前签名订正（Mermaid 结构保留）。

## mnist_infer 的 --image / --interactive 示例（原位置 docs/usage/02-quickstart-train-infer.md MNIST 推理）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文「`./build/mnist_infer --model mnist_model.bin --image datasets/...`」「`./build/mnist_infer --interactive`」。核对结果（`src/mnist_infer.cpp:34-113`）：用法是 `mnist_infer <image.csv|目录> [选项]`（输入为**位置参数**），选项只有 `--model/--topk/--show-pixels/--gpu/--help`；`--image` 与 `--interactive` 均不存在（`--interactive` 只属于 `text_infer` / `tokenizer_infer`）。示例已订正。

## 训练 C++ 示例中的 engine 实参（原位置 docs/usage/02-quickstart-train-infer.md 两个训练示例）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：`model.backward(engine, *grad);` ×2 → `model.backward(*grad);`；`nn::save_model("gpt_model.bin", model, model.spec(), tokenizer_json)` → `*model.spec()`（`Model::spec()` 返回 `const std::optional<ModelSpec>&`，`model_container.hpp:115`）。

## GUI 步骤中的「CUDA 已停用」（原位置 docs/usage/02-quickstart-train-infer.md MNIST 训练 Tab / GPT 推理 Tab 可选项）
- 类型：演进记录
- 内容：「✅ GPU 加速 — 支持 Vulkan（下拉选择；CUDA 已停用）」两处——已改为「支持 Vulkan（下拉选择设备）」。

## 模型保存格式旧表述（原位置 docs/usage/02-quickstart-train-infer.md「模型保存格式」）
- 类型：已修复勘误（与代码不符，已订正）
- 内容：原文「V4 = 当前自描述格式 / V1-V3 = 已移除支持 / `save_model` 统一写入 v4」。核对结果：`MODEL_VERSION = 5`（v5：per-tensor precision tag，`model_serialization.hpp:53/151/500`），`save_model` 写 v5、`load_model` 接受 v≥4、v1–v3 在 header 校验即拒绝（报「旧格式…请用当前版本重新训练/保存」）。正文已改为 v5/v4/v1-v3 三行的当前事实。

## 引擎使用文档中的已删 API 段落（原位置 docs/usage/03-compute-engine-usage.md 多处）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：
  - 「### 融合 axpy（2026-09 已移除该原语）」——`engine.axpy_inplace` 已删除；
  - 归约节「按行求最大值：引擎算子已于 2026-09 删除」（引擎 `row_reduce_max` 不存在，`dsl::row_reduce_max` 是 DSL 叶子）；
  - 「### 广播操作（2026-09 已移除两个原语）」——`broadcast_row_inplace` / `broadcast_col_inplace` 已删除；
  - 逐元素四小节标题「（2026-09：原语已删除，改用表达式）」×4；
  - 「### ~~块式融合~~（已移除）」整节——`engine.begin_expr()` / `engine.end_expr()` 已随 IR-C 删除（2026-09-19）；
  - 性能建议 1 的 ❌ 示例 `engine.elementwise_unary(UnaryOp::Exp, *x)` / `engine.elementwise_binary(BinaryOp::Add, ...)`（`UnaryOp`/`BinaryOp` 枚举不存在）；
  - 示例 3 的「旧式多原语逐步写」对照块 `engine.broadcast_col_inplace(*centered, *mean, BinaryOp::Sub)`（API 不存在）；
  - 逐元素节导语「单步调试/小工具用下面的 eager 原语即可」（eager 逐元素原语不存在）；
  - 页脚「*最后更新：2026-09-25（DSL 签名与示例对齐当前代码…）*」。

## 示例 1 / 示例 3 / 示例 4 的旧实现口径（原位置 docs/usage/03-compute-engine-usage.md）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：示例 1 曾记 Linear forward = `engine.matmul_with_bias(...)`（现为 DSL 融合表达式，`compute_layer_mlp.hpp:112-114`）；示例 3 曾记 LayerNorm「拆成 4 条 DSL 表达式」（现为 7 步：mean_raw→mean→diff→var_raw→std_inv→normalized→out，`compute_layer_mlp.hpp:486-544`）；示例 4 横幅「**真实实现已演进**：注意力 forward 现为单 fold kernel……不再是物化链」（改为当前事实陈述）。

## 训练包文档中的 CUDA 状态句（原位置 docs/usage/04-train-package.md 配置模板注释与 train 命令注释）
- 类型：演进记录
- 内容：`"device": "cpu", // cpu | gpu（CUDA 已停用；train 时可用 --device 覆盖）`；`python train_pkg.py train … --device gpu # 按设备覆盖（CUDA 已停用）`。当前事实（正文新增）：`--device` 取值 `cpu`/`gpu`，CLI 同时接受 `cuda` 但引擎无 CUDA 后端、该值不启用 GPU（`train_pkg.py:525` choices、`cli_controllers.py` 只处理 `gpu`）。

## 基准报告中的「旧→新」演进叙述（原位置 docs/benchmarks/2026-09-25-vulkan-vs-cuda.md）
- 类型：演进记录（已改写为报告口径；**数据全部保留**）
- 内容（原文措辞 → 现口径措辞）：
  - 头部「**2026-09-26 更新**：nn 侧已在 HEAD `4a12876` 上全量重跑；torch 侧沿用 09-25 数据（**本轮未动 torch**）」→「数据版本：nn 侧采集于 HEAD `4a12876`（2026-09-26 会话）；torch 侧采集于 2026-09-25」。
  - §0「nn 快 2.3×（**09-25 的 3.7× 系 P0 虚报**，见 §3.3‡）」→「（本表口径，见 §3.3‡）」。
  - §0「**f32 5 epoch 健康收敛（09-25 的 NaN 发散已修复）**」→「（09-26 采集；09-25 采集为 NaN 发散，见 §2.5）」。
  - §0 发现③「f32 显存 **5743→3069 MiB（−47%）**，09-25『峰值与精度无关、由池粒度主导』的观察**已不成立**」→「3069 MiB（09-25 采集 5743，差 −47%）——峰值强相关于精度，见 §2.3」。
  - §2.1「torch 吞吐 2.13×（**09-25 为 2.44×——nn f32 提速 13%，为 09-25 之后的优化收益**）」→「（按 09-25 采集的 116.3 ms/step 计为 2.44×；本表口径 101.0）」；「09-25 **旧值**」→「09-25 采集值」。
  - §2.2「09-25 同类长测 epoch 5 发散 -nan 的问题**已不复现**」→「09-25 采集的同类长测在 epoch 5 出现 -nan（见 §2.5）」。
  - §2.3「**09-25『峰值 ~5.7 GB 与精度无关、由 Vulkan 内存池粒度主导』的观察不再成立**」→「09-25 采集值为 f32 5743 / f16 5772（两者接近）；差值 f32 −47%、f16 短测 +9%」。
  - §2.5 标题「（**09-26 重写**；如实记录）」→「（按采集日期如实记录）」；「在 f32 上**已不复现**（修复落在 09-25 之后的提交窗口，未逐个二分）」→「09-25 采集记录为……；两者的差异归于 09-25 之后的提交窗口，未逐个二分」；「nn 四字段全 f16 的**历史**：09-25 之前的行为是……09-26 **变为**……症状**由 NaN 变为冻结**，风险等级上升」→「缺陷表现（按采集日期）」；「optimizer=f16 语义（**09-26 修正，替换 09-25 旧论断**）……**现已过时**」→「（本报告口径，以代码为准）……09-25 采集条目与当前代码不符，以本条为准」。
  - §3 口径「nn 列为 2026-09-26 **重跑**」「GB/s …（归约 = N + 输出，**不虚记 2N**）」→「nn 列采集于 2026-09-26」「（归约 = N + 输出）精确字节计」。
  - §3 脚注 †「已随 2026-09-26 技术债清理……从 layer_bench 算子表移除，**本轮无法复现，保留 09-25 旧值**（nn 侧标注为历史数据）」→「本行数值为 09-25 采集口径；当前算子表不含这两项……沿用 09-25 读数、不再复现」（数值 0.145/0.204/… 全保留）。
  - §3 脚注 ‡「transpose 行为 09-26 **修复后首次正确测量**：09-25 报告的是 P0 缺陷 `200955f`……**虚报值**……优势**由 3.7× 修正为 2.3×**」→「transpose 记账口径：本表数值按『派发网格与 shader 行界契约同源』口径采集（缺陷记录 `200955f`）；09-25 采集的 4096² 0.342 / 8192² 0.940 ms 为契约不同源状态下的读数……按本表口径优势 = 2.3×（09-25 采集口径为 3.7×）」（全部数值保留）。
  - §3 观察 3「09-25 报告的 3.7×/2× 系 P0 静默半写虚报，**本轮为修复后正确值**」→「口径与早先采集值（3.7×/2×）的差异见 ‡」；观察 5「（**此前** 2N 口径会虚高 ~2×）」→「按 N+out 精确字节口径」。
  - §4 观察 3 标题「（**09-26 更新**）」与「09-26 的 e2e **较 09-25 提速 13%**」→「按两个采集日的差异（在 ±15% 会话漂移内）」。
  - §5.3「（**09-26 修正**）……09-25 版本此处记……**已随代码演进过时**」→「（本报告口径，以代码为准）……09-25 采集条目与当前代码不符，以本条为准」。
  - §5.10「**09-26 重跑口径**……为真实变化」→「**采集批次口径**……反映真实差异」。
  - §6 产物索引与复现命令注释「09-26 **新增** / **重跑**」「09-26 **起**主对比」「torch 训练（09-25 数据，**本轮未重跑**）」→「09-26 采集 / 主对比 / 09-25 采集口径」。
- 说明：报告的日期化结构、全部表格与实测数字（101.0 / 47.5 ms、3069/5743 MiB、0.342/0.940 ms、3.7×/2.3× 等）**一条未删**。

---

# 构建系统 / CI（CMakeLists.txt、.github/workflows）

## 编译器与标准库一律默认（2026-09-27，原位置 .github/workflows/cmake-single-platform.yml 原 2/53-60/71-73/182-215/281 行、AGENTS.md:22、src/expr_dsl_test.cpp:13、src/expr_opt_test.cpp:13）

- 类型：bug 根因 / 演进记录
- 内容：Linux 云端构建复现历史链接失败（`tokenizer_train` FAILED，大量 `std::__cxx11::*`/`std::__throw_*`/`std::thread` undefined reference）：链接命令带两处 `-stdlib=libc++`（分处 CMAKE_CXX_FLAGS 与 CMAKE_EXE_LINKER_FLAGS 槽位），而 .o 是 libstdc++ ABI——clang 带 `-stdlib=libc++` 但系统无 libc++ 头时**静默回退 libstdc++ 头**，链接却切到 libc++ → ABI 分叉。演进三段：① 旧 workflow 曾传 `-DCMAKE_CXX_FLAGS/-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++`，与 CMakeLists 当时的 `NN_STDLIB`（Linux+Clang 默认 libstdc++，`add_compile_options/add_link_options(-stdlib=...)`）冲突（68565b0 曾统一 libstdc++ 缓解）；② 1b415bf 从 CMakeLists 移除 NN_STDLIB 及「PATH 有 clang++ 即设为默认」块，仓库自此无任何编译器/标准库指定；③ 本轮云端**在干净源码上仍复现** → 残留载体：workflow build 目录缓存（`restore-keys` 跨 sha 恢复旧 `CMakeCache.txt`，configure 不清除未重传的 `-D` 缓存变量，历史 `-stdlib` 因此永久驻留）或调用方环境（`CXXFLAGS`/`LDFLAGS`/旧 configure 命令/旧 tag）。整改（2026-09-27，两轮）：轮1——workflow 删 build 目录缓存步骤（ccache 保留，注释记明原因）、删 `-DCMAKE_CXX_COMPILER/-DCMAKE_C_COMPILER` 与 `env CC/CXX`、删与 CMakeLists 重复的 `-DCMAKE_CXX_STANDARD*`，NN_STDLIB 陈旧注释改写；AGENTS.md:22 撤回「PATH 有 clang++ 会自动优先选」（1b415bf 已删该机制）；src/expr_{dsl,opt}_test.cpp:13 手动编译建议去掉 `-stdlib=libc++`。轮1 后云端回退默认 `/usr/bin/c++`（GCC 15）→ `-Werror=format-truncation` 失败（`expr_spec.hpp:715` 与 `gen_fused.cpp:341` 的 `snprintf` 缓冲区按 `%04x` 最坏 8 位十六进制不够；GCC 独有警告、clang 不报——为何此前 WSL g++ 15.2「全量构建 exit 0」未捕获待查，疑与该环境未开 Vulkan、不编译 tools/ 有关）；轮2——按用户要求**恢复 CI 指定 clang++**（双平台 `-DCMAKE_CXX_COMPILER=clang++`/`-DCMAKE_C_COMPILER=clang` + `env CC/CXX`，名称 `CMake (Clang, C++26)`、release 说明「Clang」）。当前事实：**CI 指定 clang++，CMakeLists 与标准库一律不指定**（标准库跟随编译器默认，编译/链接两侧必须一致）；上述两处 format-truncation 编码在 GCC 本地构建（开 Vulkan 时含 tools/）仍会触发（未修，CI 走 clang 不受影响）；云端若仍复现 -stdlib 问题，先查环境变量与删除陈旧 `build/CMakeCache.txt`。

---

# AGENTS.md

> 来源：AGENTS.md（AI 开发速览）整改摘录（2026-09-27）。AGENTS.md 自此只记录当前状态；演进记录、已修复勘误、被否决方案与性能 A/B 过程统一归档于本文档。

## 版本交付内容（原位置 AGENTS.md 头部版本行 + §12「已交付能力」v1.1.0/v1.2.0）

- 类型：演进记录
- 内容：v1.1.0 = GPT 训练错误修复、算子融合优化、ZiPT/RAPT 掩码。v1.2.0（Release）= 混合精度（`precision.hpp` + CPU/GPU f16 路径）、RLA-2/RAPT、防 NaN 跳步、CPU 优化、头文件保护更换、编译器兼容（GCC/MSVC）、测试补充。头部"当前版本 v1.2.0 + 后续提交（Vulkan 设备选择、activation offload）"表述同轮移除；当前能力清单见 AGENTS §12。

## 计算类原语全量迁 DSL（2026-09-27，原位置 AGENTS.md §12「后续提交」第 1 条）

- 类型：演进记录 + 教训
- 内容：Layer 直调 **34 → 23**（`bench/doc_inventory.ps1` 复测），10 类计算原语全部退出 Layer 直调——`add_inplace`/`scale_inplace`/`accumulate`→`dsl::compute_into`（10 处）、`matmul`/`matmul_with_bias`→`dsl::matmul`（Linear/Conv/Transformer/Muon）、`row/col_reduce_*`→`dsl::compute_reduce`（11 处）、`batched_matmul` 21 处→`dsl::matmul(..., batch)`（alpha 用 `*dsl::rparam(scale)` 尾链，累加后乘三端等价）；缺口 A：新增 `ExprViewKind::GroupedReduceSum/Max`（`dsl::grouped_reduce_sum/max(t, R)`，R 进 key、glsl_gen 编译期展开、CPU 模板路径独立归约），MaxPool2D 迁移。保留 23 个 = 基础设施/数据搬运（transpose/slice/gather/im2col…）/状态扫描（scan_*/outer_col）/fold 显式登记，与 `docs/development/12` §7.3 形态一致。ctest 19/19 全绿。
- 教训①：scan dry-run **禁止 `(void)` 吞错**——ZiPTBlock 的 dry-run 调单参 `forward(x)`（实际返回"用双参"错误）被吞 → **该 dry-run 从未执行真实路径**，存量结构恰好处处有 key 才长期绿，迁移新结构立刻闭合世界硬报错（已修为双参调用 + 失败 abort）；**dry-run 长期绿 ≠ 覆盖存在**。
- 教训②：新 `ExprViewKind` 落点清单：枚举+validate（`expr_spec.hpp`）→ SpecBuilder helper+叶子+自由函数+`has_reduction_v=false` 显式特化（`expr_dsl.hpp`）→ `glsl_view_read` case（`expr_glsl_gen.hpp`，R 编译期展开）→ 解释器两处 switch（`compute_cpu_engine.hpp` 形状校验+读取）；GPU 侧无 per-view 分派，vec4 白名单/uses_row 排除集自动正确。

## Vulkan 设备选择 / activation offload / CPU 逐元素优化的提交流水（原位置 AGENTS.md §12「后续提交」）

- 类型：演进记录
- 内容：Vulkan 设备选择（15eb731）：`--gpu` 参数指定设备。activation offload（初现 `42c0883`/v0.2.0、GPT 接线 `26d8eb4`/v1.1.0、`ActivationOffloader` 抽取 `8f2990f`）：相较梯度检查点更省时（文档记载 18s vs 26s / 5 step，**仓库内无原始计时**）；API 为 `GPTModel::set_activation_offload`（`set_offload_enabled` 不存在）+ `ComputeEngine::create_offload_buffer/offload_save/offload_restore`（`offload_store/load` 于 2026-09 删除）；checkpoint 与 offload **可混合**（此前帮助文本"互斥"是错的）。CPU 逐元素优化 + `dsl::compute_into`（c0d3298）：DSL 模板路径向量化/并行、`Tensor::cpu_get_ptr`、零分配原地目标传递（optimizer/layer/loss 已迁移）。

## 代码缩减重构：算子收敛 + 测试结构 + AST 移除（2026-09-26，原位置 AGENTS.md §12「后续提交」）

- 类型：删除清单
- 内容：① 删 A 档死码（`elementwise_select_scalar_cond` / `axpy_inplace` / `broadcast_row_inplace` / `nn::one_hot` / `one_hot_labels` / `Matrix::multiply_transposed_add_to` / `nn::sigmoid`·`relu` / `ops::Sigmoid`·`ReLU`）；② 新增测试公共头 `src/test_common.hpp`（`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛一份；`max_abs_diff` 因 9 份副本语义有分叉**暂不收编**）；③ RAPT `forward_step` 的 `num/den` 除法迁 DSL，写成与 `forward:491` 同构的 `leaf/(leaf+Scalar{1e-4})` → 复用已注册 AOT 键（无需改 scan_exprs）；④ 删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`（连带 `broadcast.comp` / `broadcast_gpu` / pipeline / `has_*` / 成员全链）、引擎 `row_reduce_max`、offload B 组 `offload_store/load`、`UnaryOp/BinaryOp/CompareOp` 枚举（**引擎 virtual 58 → 49**）；⑤ **旧代数 AST 整体移除**：`algebra_expr.hpp`/`algebra_compute.hpp` 删除，`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`，`Matrix::detail::{apply,binary_apply,binary_apply_inplace}` 死函数删除 → **CPU 求值只剩「DSL 模板路径 + IR 解释器」两套**；⑥ 删死 `activation_cache()` override（ViT/ZiPT，20 行）与重复的 `offload_test` 聚合目标（**ctest 20 → 19**），`gpt_offload_test` 补 `begin_batch/end_batch`（此前无 offload save/restore 训练真实录制路径覆盖）；`--activation-offload` 对 CPU/ZiPT 改为显式警告（原为静默 no-op 却打印"已启用"）。
- 教训：`dsl::compute` **无 2 参重载**（必带 `rows, cols`）；eager→DSL 迁移**先确认目标表达式结构已被 `scan_exprs` 覆盖**，否则 GPU 闭合世界运行期硬报错。

## CUDA 后端整体移除（96a3675，原位置 AGENTS.md §2 + §12）

- 类型：删除清单
- 内容：删除 `cuda/`、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`、全库 `NN_HAS_CUDA` 分支与 CLI `--cuda`；带 CUDA 的历史快照在 git 分支 `legacy/cuda`（c0d3298）。删除依据：40HX Vulkan 端到端 2890 ≈ torch 2640 tok/s 但算子级仅 cuBLAS 1/3、batch8 OOM。当前结论保留在 AGENTS §2：**本项目不支持 CUDA**（后端仅 CPU/Vulkan，CLI 无 `--cuda`）。

## IR-C（图 IR / 跨表达式融合）整体移除（原位置 AGENTS.md §12 + §7）

- 类型：被否决方案
- 内容：删除 `expr_graph.hpp`、`compute_engine` 的 `begin_expr/end_expr`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_`。取舍记录正文（五条否证依据、删除项/保留项清单、重新立项前提）见 `docs/development/03-ir-optimization.md` §5.3（归档正文在 `docs/history.md`）。当前结论保留在 AGENTS §7：**IR-C 未采用，当前没有任何跨表达式录制机制**。

## RLA/RAPT 激活重计算 + offload 接线（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`RAPTModel/RAPTBlock` 补齐 `set_flush_interval / set_checkpoint_every / set_activation_offload`（此前对 RAPT 是静默 no-op，而 CLI 会打印"已启用"）；`ReLULinearAttention` 响应 `checkpoint_mode_`（跳过 7 项 backward 缓存）、`activation_cache()` 补齐遗漏的 4 项、backward 缺缓存硬报错；抽出通用 `ActivationOffloader`（`compute_layer_base.hpp`，GPT/RAPT 共用）；修 `RAPTModel::clear_cache()` 清空 `token_emb_`（参数被毁）与 `doc_ids` 无法关闭两个缺陷。新增 `rapt_checkpoint_test`（并入 `rapt_test`）与 `rapt_offload_test`，此前**从未被编译**的 `gpt_offload_test` 转正（ctest 15 → 18）。

## CNN 缓存契约加固 + 测试补齐（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`MaxPool2D::backward` 补 argmax 缓存/形状校验（旧行为：`clear_cache()` 后**越界读空 vector 且不报错**——`vector::clear()` 保留容量，表现为静默用陈旧索引）；`Conv2D::backward` 补 im2col 缓存/形状校验（旧行为：checkpoint 模式或 batch 变化后**静默用陈旧 im2col**）；两层 `forward` 在 checkpoint 模式下**显式清空**缓存（只跳过赋值会因 size 相同而静默沿用旧数据）；两层补 `recompute_supported()`。新增 `cnn_test`（`maxpool_gradcheck` 独立参考比对 3 组配置 + 缓存/checkpoint 契约 + `cnn_smoke_test`），CPU 与 `--gpu` 逐位一致（此前 Conv2D/MaxPool2D **无任何 GPU 覆盖**、MaxPool2D **无任何测试**）。

## CNN 全引擎化（2026-09-20，原位置 AGENTS.md §12）

- 类型：演进记录
- 内容：新增 `im2col`/`col2im` 两个 op-level 数据搬运原语（接口 + CPU + Vulkan shader + backend + GpuEngine 包装）；`Conv2D`/`MaxPool2D` 的 forward/backward 改写为纯「引擎原语 + DSL」组合——消除 `to_matrix/from_matrix` PCIe 往返与 CPU 标量 im2col 循环，`col_cache_` 由 CPU `Matrix` 改为设备张量。布局 `(C,B*P) ↔ (C*P,B)` 用 `rearrange_3d` + `gather_rows/scatter_add_rows` + 缓存置换索引实现（`rearrange_3d` 单独做不到该三维转置）。池化反向改为「窗口内并列最大值**均分**梯度」（总梯度守恒；无并列时与 argmax-first 逐位一致）。`scan_exprs` 补 MaxPool2D dry-run（69 → 72 条融合表达式）。

## 分组归约原语 + 融合 push-constant 缺陷修复（2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：新增 `grouped_reduce_sum/max`（沿行方向按固定长度 R 分组归约），MaxPool2D 因此去掉逐通道 C 次 dispatch → 单次原语；新增 `fused_gpu_test::run_reduce_consts` 回归用例。根因：`run_fused_gpu` 的 push-constant **固定头长度必须逐形态**与生成器 PC 声明一致（逐元素 2 / 逐元素+matmul 5 / 归约 4 / 归约+matmul 6）；bug 把"归约但无 matmul"按 **5** 算 → 常量池整体后移一个 uint → **GPU 上"带常量的归约"静默错值而 CPU 正常**（表现为 `col_reduce_sum(select(x == col_broadcast(max), 1, 0))` 恒返回 kk-1 而非真实并列数）。教训：这类"只有 GPU 错"的问题要**先打印生成的 GLSL/IR 再猜成因**。详见 `docs/development/08-pitfalls-and-lessons.md` §4.10。

## 评估分块（CNN 全量评估 OOM 修复，2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`evaluate_mnist` 新增 `eval_batch`（默认 1000）分块前向 + 每块 `release_idle_pool_blocks()`；`mnist_train` 传 `cfg.batch_size`。根因：CNN/MLP 走 `N = x.cols()` 全量单次 forward，CNN 的 im2col 是 k²·C_in 倍 → 60000 样本需 ~6.4 GB → `vkAllocateMemory failed: -2`（训练步其实只多 130 MB）。见 `docs/development/08-pitfalls-and-lessons.md` §3.7。

## 复合层 checkpoint 模式坑（原位置 AGENTS.md §12「坑」）

- 类型：已修复勘误
- 内容：复合层 override `forward_recompute` 必须调用**虚函数** `set_checkpoint_mode` 关闭子层；基类默认实现只改本块标志位 → 子层缓存不重建（stride>1 时被上一轮陈旧缓存掩盖，表现为部分 stride 通过）。模式开关（checkpoint/offload/doc-mask）的缓存契约见 `docs/development/08-pitfalls-and-lessons.md` 模式 H。

## GPU 算子性能优化（2026-09-23；其 BK trade-off 表于 2026-09-24 被同参 A/B 推翻，原位置 AGENTS.md §12）

- 类型：性能 A/B（含被推翻结论）
- 内容：① `matmul_tiled.comp` 改 BK=16+双缓冲（共享保持 16KB 占用率不变），op 级 matmul +11~20%（教训：BK=32 双缓冲要 32KB → blocks/SM 砍半，流水收益被占用率损失抵消净 0）。② 融合生成器 `generate_glsl_matmul` 当时采用 BK=32+双缓冲（浅网格 linear 1024³ **-1.9%**、batch512 **-4.9%**，深网格 **±0**）——其"BK16 深网格真回退 -2%"的 trade-off 表后被 09-24 同参双缓冲 A/B **推翻**（见下条全算子压榨 ⑥，最终两侧同参 BK=16）。③ `submit_and_wait` 改 solo fence/cmd 复用 + `NN_GPU_PROFILE=1` stderr 分段计时，逐元素固定开销 0.16→0.143ms；余量 submit≈70µs + wait≈75µs 属驱动/唤醒延迟，kernel 本身 4096² 已达 370-440GB/s（峰值 448）**无优化空间**。④ bench 方法论：`layer_bench` 短跑必须 `--warmup ≥20` 等时钟爬坡（空闲 300MHz→1470→2070MHz，best-of 短跑双峰如 batched 4.57/5.9ms 即爬坡所致）；torch 用 CUDA event 只测 kernel、`layer_bench` 是 wall-clock 含提交开销；40HX 理论峰值 FP32 9.14 TFLOPS / 448 GB/s（34 SM×64×2×2.1GHz）。⑤ **batch=32 超线性异常（09-24 解决）**：train 侧残留 B² 项根因 = 融合 matmul 后端 dispatch `wg_y` 按**总行** rows 派发、z=batch 又数第二遍 → 工作量 ∝ BH²（生成器契约是 y 只覆盖批内行 `m_per = rows/mm_batch`，多派的 WG 被写回守卫整块丢弃——**结果正确，对拍永远测不出，只有计时能暴露**）。修复后 train batch=32 **2999.9→227.7ms（−92%）**、b16→b32 翻倍比 3.74→1.95 线性，batch=1 亦 −16%。教训：**dispatch 网格必须与生成器行界契约同源核对；"全绿"只证明算得对，不证明没空转**。

## fold 分块流式求值 + GPU 注意力全链优化（2026-09-24，7cdb41c+fd68573，原位置 AGENTS.md §12）

- 类型：性能 A/B
- 内容：注意力 forward 改单 fold kernel（`FoldSpec`：QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 绝不物化；`tri_skip`（当时字段名 `causal_skip`）整块跳过被屏蔽区；NR=2 每 WG 两行；subgroup 蝶式归约 + vecacc 4 路软件流水）；旧物化/两趟 forward 与 `build_attention_mask`/`mask_cache_`/`m/l/attn_cache_` 等死码删除。**实测 vs main 基线 7.9/20.3：mha fwd 5.41（−31.5%）、causal fwd 4.75（−40%）、train 17.5~17.7（−13%）**，18/18 绿。质量门禁轮：Doc/AlibiDoc fold 注册与测试矩阵全覆盖、ALiBi slopes 表 `(1, batch·H)` 越界修复、GEMV subgroup≥4 门禁、tri_skip bin v8 往返断言。

## 融合 matmul vec4 内核移植（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B + 坑
- 内容：融合 matmul 同形状比 op 级 `matmul_tiled` 慢 2.3×（768×768×8192 fwd 6.18 vs 3.15ms——**"融合 vs op 级同形状对拍"是发现内核代差的 X 光**）；三重根因：内层 `acc+=dot(a[i],b[j])` 标量链（每 k ≈5× 指令发射）、全局 4×标量 ldg+16×标量 stg、共享 vec4 沿 k 布局。移植 op 级 v3 配方（共享沿 m/n + 每 k 2 LDS+4 VFMA.128 + vec4 别名快路径；trans 进 key → 分组生成期定死；BK=32/barrier 节奏/尾链不动）后：**linear 浅/深/投影三点 −33%/−48%/−47%（投影 6.18→3.25ms，达 op 级 97%）、mha fwd 4.59→3.63、mha train b1 13.0→10.8、b32 train 227.7→173.1（vs dispatch 修复前累计 2999.9→173 = −94%）**，18/18 绿。坑（rapt_offload 抓出）：共享索引必须用 tile 内**局部 k**，全局 k（`t*BK+local`）只进缓冲地址/守卫——混用则 t≥1（mm_k>32）越界共享 = 非确定性错值，num_tiles=1 的小形状对拍全测不出（全仓唯 rapt_offload d_model=64→2 tiles 有对照）；"对拍全绿但数值每跑不同"→ 先同二进制跑两次 diff。

## 全算子压榨 + OP/DSL 同步轮（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B（含推翻 09-23 trade-off 表）+ bench 坑
- 内容：① `batched_matmul.comp` 移植 matmul_tiled v3（自 0.2.0 起未优化，同形 1024³ 比 matmul 慢 1.85×）：**b1 1.66→0.876ms（−47%）、b8 8.12→4.08ms（2116→4214 GFLOPS）**。② `transpose.comp` 重写 32×8 WG / 32×32 共享分块（tile[32][33]，读写双向 128B 合并、零 bank 冲突），backend dispatch 改 (ceil(C/32), ceil(R/32))：**4096² 1.575→0.492ms（−69%）**。③ `reduce.comp` 列归约重写（lane=列/warp=行块/32 列 tile + shared 跨块合并 + 4 路 ILP），max 恒等元 −inf→`lowest()`、行归约→subgroup 蝶式：**4096² 0.843→0.273（−68%）**；reduce 用 subgroup → CMake 需单独 `--target-env=vulkan1.2`（同 matmul_gemv）。④ DSL 列归约同结构重构（`generate_glsl_reduce`：dispatch cols/256→cols/32——CE 训练热路径旧 WG 并行度极低；合并值覆写回 s_red 与他 lane 读同址，**必须 3 barrier：partial 完成→读完→写完，否则 sum 双计**）。⑤ `elementwise_v2`/`broadcast`/`gather` → vec4（base=gid*4），三处 dispatch ÷4，broadcast 4096² −17%。⑥ **OP/融合 matmul BK 统一 A/B（推翻 09-23 结论②的 trade-off 表）**：OP 级 BK16→32 四点全回退（512³+9%、1024³+26%、4096³+29%、batched b8+29%）→ 维持 16；融合侧 BK32→16 反超 linear 浅 −24% / 深 fwd −24% / 深 train −17%（8.21→6.81ms）→ **两侧同参 BK=16**（09-23"深网格 BK16 −2%"是对单缓冲旧基线的口径，非同参双缓冲对比）。⑧ **bench 坑**：`layer_bench` transB/at 操作数曾恒建 (k,n)（应 (n,k)/(k,m)），k≠n → backend K mismatch → run 内 `*expected` 不查错 = UB → **0.000ms / 1.7e11 GFLOPS 超物理垃圾值**（k=n 正方形长期掩盖）；已改 per-variant setup + 错误检查 abort。教训：**bench 出现超物理数值先怀疑测法本身**。gather 快路径曾漏读 `indices[]`（把索引下标当表行号），zipt CPU/GPU 对拍抓出——vec4 改写后对拍必跑。最终 @1024：batched 0.871、matmul 0.853、col_reduce 0.123、transpose 0.138、broadcast 0.119；mha fwd 3.449 / causal fwd 2.819 / linear b4096 train 6.901ms。18/18 绿。

## transpose 砖块化 + reduce 两段式 + bench 记账修复（2026-09-24 二轮，原位置 AGENTS.md §12）

- 类型：性能 A/B + 方法论
- 内容：① **bench 虚高记账修复（"部分算子超出 40HX 理论"的根因）**：`bytes_reduce` 曾按 2N 记归约（输出仅向量级，实际 ≈N）→ 5244² 虚报 440 GB/s、4096² 虚报 491（超 448 理论峰值，物理不可能）；改精确 (N+out) 后真实 217-263 → **reduce 其实有一倍头寸**。② `transpose.comp` 定稿：64×64 tile（段宽 128→256B）+ 共享落位交换 + **8×8 WG 砖块化 dispatch**：5244² 0.99→0.75ms（−24%）、4096² −17~28%。消融链定根因：纯拷贝 0.556 → 过 shared+barrier 0.665 → 线性 dispatch 0.850，**真凶 = 并发 WG 写侧足迹散布整块**（转置输出行=输入列：线性 dispatch 下 136 个并发 WG 每行只写 256B、行距 21KB ≈180MB 散布）——砖块化让连续 64 WG 填满一砖、足迹收敛 ~1MB 聚集；16×8 砖与 8×8 持平。③ `reduce.comp`：行归约 vec4 快路径（cols%4==0）；列归约**两段式 partials**（ggml `rms_norm_partials` 模式：pass1 行向切 nchunk → partials(nchunk,cols)，pass2 复用整表合并；同 cmd 双 dispatch 单次提交 → 固定提交开销不翻倍；scratch 走成员 `reduce_partial_` 防录制期 UAF）：5244² 0.50→0.38-0.42ms；push 扩 5×uint=20B（`chunk_rows`），**pipeline layout push range 同步 4→5 个 uint——漏改即静默错值**；512K 元素护栏排除微型形状（512² 两段 −16% 净亏）。④ `broadcast.comp` vec 行单 vec4 装载 5244² −13~16%。⑤ **方法论（调参/判退前必读）**：同码对照两 exe 同窗交错实测 **噪声地板 ±5%、跨会话系统漂移 ±15%**（round2 全面慢 5-10%）——两段式曾被跨会话单点误判"4096² 净亏 20%"，交错 A/B 实测只亏 2%（SP 5/5 配对胜）、5244² 反赢 6-9%（TP 4/5）才定案保留；**调参/判退一律同窗交错 + 配对统计，禁止对着跨会话单点改参数**。18/18 绿。

## 多精度 Phase 2：边界 cast → in-kernel f16 → op-level f16 GEMM（2026-09-25，原位置 AGENTS.md §12 第 ⑭ 大条，docs 05 §12.5-§12.11）

- 类型：性能 A/B + 演进记录
- 内容（按阶段保留关键数字与结论）：
  - **接线**：引擎运算类原语加 `Precision P = F32`、新增 `cast_into`/`copy_into`；新增 `PrecisionEngine` 适配层（f16 边界 cast 集中一处，全 f32 = 纯直通逐字节一致）；`dsl::compute/compute_reduce` 加 P；Layer/Loss/Optimizer/工厂/CLI 全链接线。教训：**复合层必须把 profile 下传给子层**（`AttentionBase`/`FeedForward`/`TransformerEncoderLayer`——不下传则子层静默停 f32；`Model::add` 用未设默认值覆盖 GPTModel 构造器已注入的 profile 曾导致 token_emb 退回 f32）。
  - **`--f16` 语义** = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}；**四字段全 f16 不可训练**（optimizer=F16：Adam 的 v≈g²~1e-10 下溢→更新爆炸 loss 7.9→3.6e4；stable=F16：CE 链 ~200 步 NaN；两者同 f16：loss 恒定）。
  - **边界 cast 结论**（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq256）：f32 峰值 3069MiB；f16 边界 cast 峰值反而升高（2831MiB 后 backward OOM，探针 transient live 2795→6687MB）→ **边界 cast 只能拿"存储减半"，拿不到"峰值下降"；须做 in-kernel f16**。
  - **in-kernel 第一期**：`ExprPrecSig`（bit i = 第 i 个输入 f16、bit16 = 输出 f16，全 0 = 旧行为）+ `ExprRegistry::variants/add(spec,sig)` + bin v9 只存 `{sig, 基础结构下标}`；`scan_exprs` 双 pass（**f16 pass 必须经适配层——原生 CpuEngine 喂 f16 = heap corruption 0xC0000374**）；`GlslEmitter::generate(name,spec,sig)`（`RotateHalf` 取负 / `RowGather` 的 `uint()` 必须在叶子处转，否则 glslc 报错）；`run_fused_gpu` 输入 `FusedInputs{owners,bufs}` **必须同时持有 owner**（只存裸指针 = 录制中途释放上传缓冲，实测 `A+=B err=0.5`）；f16 输出按 2B/元素分配并重贴 `GpuTensorF16`（漏了 → loss=NaN）。实测 batch32：f32 1595 / f16 in-kernel 3457 / 边界 cast 4124MiB（f16 仍 2.2×）。**签名必须逐输入**：真实 GPT f16 变体空间 38 个 = 逐元素/归约 32 + matmul 段 5 + fold 1，存在混合签名。
  - **归因演进（教学点）**：⑪ 判"边界 cast 临时块数量膨胀"→ ⑫ 判"池底材粒度"（阶梯≤16MB −21% 但 2× 耗时）→ ⑬ 同窗复测**推翻 ⑫**（峰值 3513/3513/3513 持平、耗时全 ≈5.7s，"2× 时间"是跨会话单点产物，池小分配快路径工程项作废——池账本 1 万次 allocate 只扫 ~5 万 block、`vkAllocateMemory` 累计 ~150ms，本就不是瓶颈）+ 新增形状级归因表（`dump_temp_stats`，`NN_PREC_TRACE=1`）定罪 **op-level cast**：`(32768,256)` 注意力反向 W/grad_A 占 **1152MB**，其次 `(64,8192)` 270、`(256,8192)` 160、`(2048,256)` 112MB。**机制之争必须用形状/尺寸归因表裁决，别用总量推理。** 据此补三类带类型变体（matmul 段 `float16_t`+`uvec2`+`unpackHalf2x16` / 归约 kernel `rd()/wr()` 读写点转换 / `eval_expr_into|eval_expr_reduce` 加 `(key,sig)` 匹配）→ `gen_fused` 由 **41 变体/14 skip → 54 变体/0 skip**；实测同窗 3 轮：f32 1753（3/3 一致）、f16 2025/2025/2273 = 1.15~1.30×（此前 3513，即 cast 路径 **−35%~−41%**），耗时无回退，batch64 f16 不再 OOM。
  - **op-level f16 GEMM（第 ⑭ 阶段）**：手法 = 一份 .comp 用 `glslc -DNN_SHADER_F16=1` 编第二份 SPIR-V（`NN_ETYPE=float16_t`/别名槽 `NN_V4=uvec2`/`NN_LOADV4=unpackHalf2x16`，**f32 分支留在 `#else` 逐字未动 = 零回归证明：头文件字节数不变**（`matmul_tiled_spv.hpp` 68670 / `batched_matmul_spv.hpp` 74408））；后端 pipeline 只在 `has_16bit_storage()` 时创建；**适配层必须把 `P != F32` 直接下传**（第一版漏了这步 → 归因表逐项不变、峰值纹丝不动）；`CpuEngine::matmul_with_bias` 补下传 `P`（否则 scan 的 f16 dry-run 只登记全 f32 签名；变体 54→56）。实测同窗 3 轮（batch32/steps2）：f32 1754×3、f16 1570×3 = **−10.5% 显存且略快**（5.05 vs 5.11s）；batch64 亦 −10.5%；`text_train --f16` avg_loss 6.7493 vs f32 6.6979（0.77%）。**演进全景：f16 边界 cast 4124 → in-kernel 首期 3513 → +matmul段/归约/目标传递 2025~2286 → +op-level GEMM 1570**；剩余约 550MB cast（`(2048,256)×80`、`(64,8192)×72` 等）+ 7 条 run-only 签名 + fold 变体，收益已递减。

## CPU 侧 f16 训练发散双根因修复（2026-09-25，原位置 AGENTS.md §12 末条，docs 05 §12.12）

- 类型：已修复勘误（根因 + 修法 + 测试盲区）
- 内容：§12.5"已知问题 1"（CPU f16 5~7 步 NaN、未定位）实为**双缺陷叠加**。
  ① **DSL 预绑定喂错精度**：`MatmulRef::prepare_cpu` 固定以 P=F32 物化 C，而 `CpuEngine::matmul` f32 路径按 f32 存储直读 f16 张量 → `cpu_matrix<F32>()` 空指针 UB（Debug 断言 `tensor has no P-precision CPU storage`；Release 即 `text_train --f16` CPU 0xC0000005）；同族：`ReduceViewRef::prepare` 直读、适配层 `matmul/batched_matmul` 把 `P!=F32` 无条件下传、CPU 解释器 `eval_expr_impl` 无精度校验。修复 = prepare 先抬 f32 再物化 / f16 输入一次性镜像 / `CpuEngine::matmul|batched_matmul` 精度与存储不匹配时回退 f32 空间计算 + 按 P 舍入（两端同 f16 才走原生 f16 GEMM）/ 解释器加精度 `NN_REQUIRE`。
  ② **`float_to_half_bits` 次正规分支移位 UB（真正的 NaN 制造机）**：守卫 `exp <= -46` 与其自身公式矛盾（只对 exp∈[-25,-15]/shift 14..24 成立）→ **exp∈[-45,-33]（|v|∈2.8e-14~1.2e-10）shift≥32 移位 UB** → 小值被转成垃圾 half（0x4000=2.0 / 512 / 8192 / 11776 / 18432，随机落 e=31 即 NaN）；**梯度正是这个量级、前向激活 ~0.1 从不落入** → 损坏只见于 f16 梯度写回；GPU 走硬件转换故"只有 CPU 错"；单算子测试输入 ≥1e-3 故"全过"。修复 = 守卫改 `exp <= -26`（≤-26 按 D<0.5 恒 flush 0）。
  ③ **测试盲区**：`ref_ulp` 写成 `2^(e-10)`（应 `2^(e-25)`，**大 2^15 倍**）→ RHE 容差比被测值本身还大、垃圾值全放行；修复 ref_ulp + 补 exp∈[-60,-26] 全网格"必须恒 0"断言 + 两个历史垃圾代表值定点断言。
  ④ **方法论**：新组件探针 `f16_cpu_probe`（阶段 A–I）+ `NN_F16_DEBUG=1` 层内逐中间量扫描——把故障钉到 `accumulate: pre_dst=0, pre_src=8.7e-13, post_dst=18432` 单步即锁定转换函数。修复后 CPU f16 三组 profile 6 步轨迹与 f32 **逐位一致**；`f16_precision_test` 容忍分支改**硬失败**；profile 单字段矩阵仅 optimizer=F16 仍发散（§12.5 数值性限制，与本缺陷无关）。

## 基准报告更新与 f32 发散勘误（2026-09-25/09-26，原位置 AGENTS.md §12「已知问题」）

- 类型：已修复勘误 / 性能 A/B
- 内容：**f32 训练发散 NaN 已修复**——09-26 实测短测 3 epoch + 长测 5 epoch 健康收敛，09-25 基准记录的"f16/f32 均发散"在 f32 上不复现。**09-25 基准报告 nn 侧全量重跑**（`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md`）：f32 优先主对比 101.0 ms/step；transpose 3.7×→2.3×（P0 静默半写虚报修正）；`elementwise_exp`/`broadcast_col` 已删标注；f32 显存 5743→3069 MiB。

## S7 关键教训中的过程叙述（原位置 AGENTS.md §12「S7 关键教训」第 5、6 条）

- 类型：演进记录（约束本体保留在 AGENTS §12「融合二期状态与 IR 编码约束」）
- 内容：matmul + 列归约于 **2026-09-26 补齐**（`generate_glsl_reduce` 列分支按元素分解 batch、`gen_fused` 三处跳过删除）。注意力 forward 曾为 m/l/W 表达式 + bm(W,V_t) 的 S7 IR 链结构（**已删**），现为单 fold kernel（`FoldSpec`，5 掩码变体经 `fold_mask_variant_`）；bwd = R/X 表达式 + 3 个 `batched_matmul`。

## 「已过时的历史说明」整节（原位置 AGENTS.md §12 末节）

- 类型：删除清单
- 内容：五条——① **全库代码审查报告（2026-09-04）**：一次性审查文档已删除（git 历史可查）；`--tdr-retry`/`--max-tdr-retries` 选项已从代码移除（当年"被解析但从未使用"的问题不复存在——当前结论保留于 AGENTS §12）。② **CPU 性能诊断报告与路线选择建议（2026-09-18）**：临时诊断/决策文档已删除，结论并入 `12-compute-engine-inventory.md`（CPU 侧求值机制 2026-09-26 起只剩两套：DSL 模板路径 + IR 解释器）与 `03-ir-optimization.md` §5.3（IR-C 取舍）。③ **CpuEmitter**：`cpu_emitter.hpp` 与实现已删除，早期"待修 CpuEmitter 隐性缺陷"问题随之消失勿引用（当前表述：AGENTS §7 仅 `glsl` 后端注册，无 CpuEmitter/CudaEmitter）。④ **融合三期 S6**：未列入当前计划；替代方案 P2-12 图级缓存随 IR-C 于 2026-09-19 删除。⑤ **IR-C**：已评估并整体移除（删除项见上文 IR-C 条目）。

## 其余移除叙述与过期事实订正（原位置 AGENTS.md §2/§3/§4.3/§5/§7/§8/§10/§11）

- 类型：删除清单 / 已修复勘误
- 内容：
  - **§4.3「2026-09 收敛」块**：`axpy_inplace`、`broadcast_row/col_inplace`、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、`row_reduce_max`（引擎算子）、`offload_store/offload_load` 以及 `UnaryOp/BinaryOp/CompareOp` 枚举已全部删除（逐元素/广播一律走表达式 DSL；`dsl::row_reduce_max` 是 DSL 叶子）；**引擎 virtual 58 → 49**。
  - **§5 铁律 5 历史叙述**：历史上 GPT 曾用 position-major（`i = t*batch + b`）导致跨样本串扰的灾难级 bug，现已统一为 batch-major。
  - **§10 坑 7 历史数字**：BPE"所有出现位置"类索引曾膨胀到 60-80GB；大词表 one-hot 曾 3.2GB。
  - **§7 移除叙述**：跨表达式融合（`start_expr/end_expr`、`begin_expr/end_expr`、`expr_graph.hpp`）2026-09-19 移除；fold 构造 2026-09-25 前在 `expr_fold.hpp`（后按 AOT 原则归位 `compute_layer_attention.hpp`）；`FoldSpec.tri_skip` 原名 `causal_skip`；`dsl::grouped_reduce_sum/max` 2026-09-27 新增；`broadcast.comp` 随 `broadcast_*_inplace` 于 2026-09 删除；`CpuEmitter`（已删除）/`CudaEmitter`（随 CUDA 移除）勿引用。
  - **§3 目录表括注**：旧代数 AST（`algebra_expr.hpp`/`algebra_compute.hpp`）2026-09 移除、`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`；`expr_graph.hpp`/IR-C 2026-09-19 移除。
  - **§8**：RAPT "自 2026-09-19 起"与 GPT 同档（当前：RAPT 与 GPT 同档）；链式构建伪 API 见下条。
  - **§11 索引过期事实订正**：`development/12` 描述由"49 个 virtual（原 58，删 9 个）、**Layer 直调 32 个**"订正为实测 **23**（`bench/doc_inventory.ps1`：virtual 49/49、layer-called 23、not layer-called 26）；"后附 2026-09 收敛记录"→ 演进记录已移出（本文件/w4-dev1）；`development/13` "含误报/已修复对照表（§6）"→ 对照表已移入 `docs/history.md`；`development/03` "§5.3 是取舍记录"→ §5.3 现只指向 `docs/history.md`；`introduction/01` "CUDA 已移除备注见篇末"→ 该文档已无此备注；`development/04` "已实施"、`development/02` "删手写原语"等状态词与历史词改为当前形态描述；新增 §11 总档行 `docs/history.md`。

## §8 链式构建伪 API（`add_linear`/`add_relu`，原位置 AGENTS.md §8）

- 类型：已修复勘误
- 内容：原文写 `model.add_linear(784,256).add_relu().add_linear(256,10)`——**全仓不存在 `add_linear`/`add_relu`**（grep 仅命中 AGENTS.md 自身）。`Model::add` 是模板方法 `template <typename LayerType, typename... Args> Result<void> add(Args&&...)`（`include/neuralnet.cpp/model_container.hpp:71-72`），无链式 API，层构造后自动 `init(engine)`。已改为逐层写法 `model.add<nn::Linear>(784,256); model.add<nn::ReLU>(); model.add<nn::Linear>(256,10);`，与 `docs/usage/01-quickstart-model.md:19-21` 一致。
