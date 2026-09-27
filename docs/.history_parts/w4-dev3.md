# w4-dev3

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
