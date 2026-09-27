# w1-expr

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
