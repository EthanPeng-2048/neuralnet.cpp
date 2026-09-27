# w3-tests-a

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
