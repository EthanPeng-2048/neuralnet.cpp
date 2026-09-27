# w4-dev1

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
