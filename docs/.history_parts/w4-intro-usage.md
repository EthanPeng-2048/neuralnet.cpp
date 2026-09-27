# w4-intro-usage

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
