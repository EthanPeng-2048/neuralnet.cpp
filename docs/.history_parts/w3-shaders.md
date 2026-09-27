# w3-shaders

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
