# Layers 部分一（attention/softmax/feedforward/base/mlp/conv/transformer）代码审查

## 模块概览

本组 8 个文件构成 L2 计算层的"基础注意力/MLP/卷积/Transformer 块"子集：`compute_layer.hpp` 为按依赖拓扑序聚合的入口头；`compute_layer_base.hpp` 定义 `Layer` 基类（forward/backward/参数/检查点/offload 契约）；`compute_layer_mlp.hpp` 含 Linear/ReLU/GeLU/SwiGLU/LayerNorm/RMSNorm/FusedChainLayer；`compute_layer_conv.hpp` 含 Conv2D/MaxPool2D（im2col/col2im CPU 端完成）；`compute_layer_attention.hpp` 是本组核心（RotaryEmbedding/AttentionBase 两趟式注意力/MHA/CSA/ALiBi/doc_ids）；`compute_layer_softmax.hpp` 数值稳定 softmax；`compute_layer_feedforward.hpp` 含 FeedForward；`compute_layer_transformer.hpp` 含 PositionalEncoding/TransformerEncoderLayer/TransformerEncoder/PatchEmbedding。

整体质量较高：铁律 1（Result 全程检查传播）在各层 forward/backward 一致遵守；铁律 4（batch-major 列序 i=b*seq+t）在 attention 的 Q/K/V rearrange_3d、doc_ids 索引、build_attention_mask 三处均逐处核对一致；铁律 6（运行时值禁进表达式常量池）通过"scale 折进 Q（forward scale_inplace + backward scale_inplace 补乘）"与"inv_num_valid 后置 scale_inplace"两条 S7 教训正确落实。两趟式注意力的 m/l/W 数值稳定（m=行 max 先减、l=行 sum(exp)、W=exp/l）与 backward 的 R/X 表达式数学均正确。未发现 P0 正确性 bug。

## 发现

### P0
（无）

### P1
（无）

### P2

1. **[P2] compute_layer_attention.hpp:1060 — `set_doc_ids` 残留：doc_ids 旧物化路径仍在**
   `CausalSelfAttention::set_doc_ids`(1060-1070) 把 `doc_ids_` 存为 `std::vector<std::size_t>`，而 S7 IR 两趟式路径（`two_pass_mask_` 1117-1149）每步把同样的 doc_ids 重复 `from_matrix` 成 `doc_ids_cache_(1, BH*seq)` 与 `doc_col_(BH*seq,1)` 两个张量。`set_doc_ids` 本身是 GPTModel 文档感知掩码的对外入口（合法保留），但"每步重建 + 旧 apply_mask_ doc 分支仍引用 `doc_ids_`"构成重复存储与双路径（IR 表达式 + 旧 build_attention_mask 物化）。
   - 影响：每 step 两次 O(BH·seq) 的 from_matrix（PCIe 往返）；`doc_ids_` 与 `doc_ids_cache_`/`doc_col_` 三处同源数据，维护时需同步。
   - 建议：doc_ids 是逐步变化的运行时输入，无法缓存；但 `apply_mask_` 的 `has_doc_ids_` 分支（1078-1089）在 S7 两趟式恒 true 下已不可达（`two_pass_mask_` 1150 恒返回 true），属死代码，可删该分支及 `build_attention_mask` 的 doc_ids 参数（937-960）。

2. **[P2] compute_layer_attention.hpp:1040 — `ensure_mask_` 纯因果/ALiBi 掩码缓存键缺 ALiBi 标志**
   `ensure_mask_` 以 `(batch, seq)` 为缓存键（1043），但 `use_alibi_` 是构造期常量，缓存 `mask_cache_` 仅在 (batch,seq) 变化时重建。若同一 CSA 实例在运行期切换 `use_alibi_`（当前 API 不支持，但成员可变），旧掩码会被误用。当前 `use_alibi_` 构造后不变，实际无触发路径，列为低概率防御缺陷。
   - 建议：缓存键加入 `use_alibi_`，或注释明确 `use_alibi_` 运行期不可变。

3. **[P2] compute_layer_attention.hpp:585 — 旧路径（`two_pass_active_=false`）在 S7 下恒不可达**
   `two_pass_mask_` 基类与 CSA 子类均恒返回 `use_two_pass=true`（270、1150），故 forward 的 else 分支（585-615）、backward 的 else 分支（709-740）、`apply_mask_`/`apply_mask_step_` 钩子、`attn_cache_` 成员、`softmax_.output_cache()` 复用路径全部为死代码。这是 M6→S7 重构的过渡残留。
   - 影响：代码量约 60 行 + 1 成员 + 2 钩子，增加认知负担；`attn_cache_` 成员从未被填充（仅 clear）。
   - 建议：确认两趟式为唯一路径后删除旧路径分支、`attn_cache_`、`apply_mask_`/`apply_mask_step_`/`mask_backward_` 钩子及 `softmax_.output_cache()` 对外接口（仅旧路径用）。

4. **[P2] compute_layer_mlp.hpp:106 — Linear forward 缓存 `input_cache_ = input` 为浅拷贝（Tensor 别名共享）**
   `input_cache_ = input;`（107）依赖 Tensor 的共享语义（若 Tensor 为引用计数/句柄则 alias，若为值拷贝则深拷贝）。backward 中 `engine.matmul(grad_output, input_cache_, ...)`（126）消费它。若上游在 backward 前复用/释放了 input 的底层 buffer（如 TransformerEncoder 的 `x = std::move(*lr)` 链式覆盖），`input_cache_` 可能悬空。当前调用链（Model 逐层 forward 后统一 backward）保证 input 存活到 backward，故无实际触发，但语义依赖调用顺序契约。
   - 建议：注释明确"input_cache_ 要求 input 的底层 buffer 在本层 backward 完成前存活"，或改用 `engine.clone(input)` 深拷贝（代价为一次全尺寸拷贝）。

5. **[P2] compute_layer_conv.hpp:33 / 40 — `col_cache_` 用 `Matrix` 而非 `Tensor`，CPU/GPU 语义不对称**
   `col_cache_` 声明为 `Matrix`（33），forward 在 CPU 端 `im2col_` 生成后存入（214-215），backward 再 `from_matrix` 回设备（247）。这与"全程 GPU 张量流"架构约定（docs §4.3）不一致——Conv 是唯一把 im2col 结果常驻 CPU Matrix 的层。MNIST 尺度可接受，但与 PatchEmbedding（同样 CPU 端，但用 `to_matrix`/`from_matrix` 边界）风格略异。
   - 建议：与 PatchEmbedding 对齐，或注释说明 Conv 的 im2col 是"算法职责无对应原语"的合法 CPU 常驻（注释 877-879 已部分说明，可补 `col_cache_` 的 CPU 常驻理由）。

6. **[P2] compute_layer_transformer.hpp:587 / feedforward.hpp:123 — 文件尾部大段注释重复（CausalSelfAttention / TransformerEncoderLayer 算法说明）**
   `compute_layer_transformer.hpp` 560-586 重复了 attention.hpp 984-1003 的 CausalSelfAttention 算法注释（纯注释，无代码）；`compute_layer_feedforward.hpp` 123-132 重复了 transformer.hpp 110-117 的 TransformerEncoderLayer 注释。属拆分文件时的注释搬运残留。
   - 建议：删除重复注释块，仅保留一份。

### P3

1. **[P3] compute_layer_mlp.hpp:40 / conv.hpp:36 — `thread_local rng_` 跨线程初始化确定性**
   `Linear`/`Conv2D` 各用 `inline static thread_local std::mt19937_64 rng_{std::random_device{}()}`（mlp:40、conv:36）。多线程并发 `init` 时不同线程种子不同，权重初始化不可复现（跨 run 不同）。单线程 init 可复现，但训练入口若多线程构建模型则权重 seed 漂移。
   - 建议：init 前用全局固定 seed 或接收 `seed` 参数；至少文档说明"多线程构建不保证权重可复现"。

2. **[P3] compute_layer_attention.hpp:89-102 — RotaryEmbedding `d_k_` 奇数时 `half=d_k/2` 截断**
   `half = d_k_/2`（38）对奇数 d_k 截断，`fill_pos_column_` 只填 `half` 对（0..d_k-2 行），最后 1 行（d_k-1）未填 cos/sin（保持 0）。`apply`/`apply_step` 有 `d_k%2!=0` 守卫（101、128）返回错误，故奇数 d_k 不会实际执行，但 `fill_pos_column_` 本身对奇数 d_k 是未定义行为（越界写 `half+j` 到 d_k-1 合法，只是最后 1 行缺失）。守卫已覆盖，列为提示。
   - 建议：`fill_pos_column_` 加 `NN_ASSERT(d_k_ % 2 == 0)` 或在注释标注"仅偶数 d_k 有效"。

3. **[P3] compute_layer_conv.hpp:34 — `shape_invalid_` 守卫未覆盖 `in_h==0/in_w==0`（默认构造）**
   构造守卫 `in_h_ + 2*padding_ < kernel_`（153）对 `in_h_=0, in_w_=0`（默认）且 `kernel_≤1` 时为 false（0+0<1 假），会进入 else 分支算 `out_h_=(0+0-kernel_)/stride_+1`——kernel_≥1 时 `0+0-1` 无符号下溢。但 `in_h==0/in_w==0` 本就不应构造 Conv（无输入尺寸），实际入口（domain_cnn）均传真实尺寸。
   - 建议：守卫加 `in_h_==0 || in_w_==0` 判无效，防御无尺寸构造。

4. **[P3] compute_layer_mlp.hpp:50 — Linear init 未处理 `in_features_+out_features_==0`**
   `limit = sqrt(6.0/(in+out))`（50）若 in=out=0 除零。但 Linear(0,0) 无实际意义，调用方不会触发。
   - 建议：构造或 init 加 `in>0 && out>0` 守卫。

5. **[P3] compute_layer_attention.hpp:307 — `scale_` 用 `d_model/num_heads` 而非 `d_k_`（数值等价但语义绕）**
   `scale_ = 1/sqrt(d_model/num_heads)`（307）而 `d_k_ = d_model/num_heads`（305），二者数值相同。用 `d_k_` 更直接，当前写法是 d_k_ 赋值前已算（构造函数成员初始化顺序 d_k_ 在 scale_ 前，可用），改 `1/sqrt(d_k_)` 更清晰。
   - 建议：`scale_(Scalar{1}/std::sqrt(static_cast<Scalar>(d_k_)))`。

6. **[P3] compute_layer_softmax.hpp:49 — softmax forward `row_reduce_max` 出现两次（CSE 应消除但写法冗余）**
   `dsl::exp(x - row_reduce_max(x)) / row_reduce_sum(exp(x - row_reduce_max(x)))`（50-52）中 `row_reduce_max(x)` 出现两次。DSL 的 CSE（docs §7、AGENTS §12 S7 提到 βx 子表达式由 CSE 复用）理论上会合并，但源码层面冗余，若 CSE 失效则两次归约。
   - 建议：提取 `auto m = row_reduce_max(input);` 复用，显式避免对 CSE 的依赖。

7. **[P3] compute_layer_transformer.hpp:545 — PatchEmbedding forward `input_cache_` 赋值但 backward 未用**
   `input_cache_ = input;`（485）存入，但 backward（523-557）只用 `grad_output.cols()/num_patches_` 推 batch，未读 `input_cache_`（batch 可由 grad 推得，无需 input）。`input_cache_` 是死存储（每 step 存一份 (img², batch) 张量，GPU 上占显存）。
   - 建议：删除 `input_cache_` 成员及赋值，backward 已自行推 batch。

8. **[P3] compute_layer_base.hpp:119 — `clone_tensor` 辅助函数无调用点（疑似死代码）**
   `clone_tensor`（119-123）是 `engine.clone` 的包装，本组 8 文件内无调用点（grep 仅见定义）。可能在其他文件（gpt/zipt/rapt）用，属本组外的共享工具，不强制删，列为提示。
   - 建议：确认全库调用点；若仅注释引用则保留为公共工具。

9. **[P3] compute_layer_mlp.hpp:796 / transformer.hpp:560 — 多个类缺 `[[nodiscard]]`（非虚 override 方法）**
   `FusedChainLayer::parameters/param_gradients`（846-847）、`RotaryEmbedding::init`（81）、`CausalSelfAttention` 部分方法等返回 vector/Result 但未标 `[[nodiscard]]`。基类虚函数已标（base:37-73），子类 override 继承 nodiscard 语义（虚函数 override 自动继承），但非虚的 `RotaryEmbedding::init`（非 Layer 成员）等建议补标。
   - 建议：非虚返回方法补 `[[nodiscard]]`。

10. **[P3] compute_layer_attention.hpp:217 — `kNegInf_` 为 const 成员（非 static constexpr），每实例占存储**
    `const Scalar kNegInf_ = -infinity()`（217）是每 AttentionBase 实例一份的 const 成员（非 `static constexpr`）。`-infinity` 是常量，应 `static constexpr` 避免每实例存储。
    - 建议：`static constexpr Scalar kNegInf_ = -std::numeric_limits<Scalar>::infinity();`

## 已知问题核对

- **set_doc_ids 残留**：核实**仍存在**。`set_doc_ids` 定义在 base:62（no-op 虚函数）、attention:1060（CSA override）、gpt:77/686、zipt:389/790、rapt:149/641/853、model_container:143（转发）。文档感知掩码功能**仍活跃**（非纯残留）——CSA 的 `set_doc_ids`（1060）→ `two_pass_mask_` 重建 `doc_ids_cache_`/`doc_col_`（1117-1149）→ IR 掩码 `masked_doc_`/`masked_alibi_doc_`（237-257）消费。但旧物化路径 `apply_mask_` 的 `has_doc_ids_` 分支（1078-1089，调 `build_attention_mask` doc 参数）在 S7 两趟式恒 true 下**已不可达**，属死代码（见 P2-1）。`build_attention_mask` 的 doc_ids 参数（937-960）仅被该死分支引用。

## 其他观察

- **两趟式注意力数值稳定性（已核对正确）**：m = row_max(S+mask)（494-497）先减最大值 → l = row_sum(exp(S+mask−m))（518-522）→ W = exp(S+mask−m)/l（543-547）。全程无 `exp(S)` 直接溢出风险（S−m ≤ 0）。backward：R = row_sum(W·P)（680-682）、X = W·(P−R)（684-687）、grad_Q = K·X^T × scale（690-692）、grad_K = Q·X（695）、grad_V = W^T·G（698）。数学与 FlashAttention 两趟式一致（docs/flash_attn_analysis.md）。`scale` 折进 Q（459 forward `scale_inplace`）+ backward grad_Q 补乘（692 `scale_inplace`），运行时值不进表达式常量池（铁律 6 遵守）。
- **rearrange_3d shape（已核对正确）**：Q/K/V (H_dk, batch*seq) → `(H_dk, batch, seq, false)` → (batch*H_dk, seq)（427-435），与铁律 4 一致。V 转置 (seq, BH*d_k) → rearrange (seq, BH, d_k) → (BH*seq, d_k)（560-562），O 转置回 (BH*d_k, seq)（569-571），最后 (batch*H*d_k, seq) → rearrange inverse (H_dk, batch*seq)（621）。batch=1 时跳过 rearrange（439-442、625-627），正确。
- **BatchCol 视图布局（已核对正确）**：`doc_ids_cache_` 构建为 (1, batch*num_heads*seq)（1123），每 (b,h) 块重复 doc_ids（1124-1128），匹配 CPU 引擎 `BatchCol` 校验 `(1, batch*param)`=（1, BH*seq）（cpu_engine:1152-1155）。`doc_col_` 构建为 (BH*seq, 1)（1137-1146），跨 head 重复 doc_ids。masked_doc_（237-245）`row_broadcast(doc_col) != batch_col(doc_ids, seq)` 比较正确——两趟式不会越界。
- **softmax 数值稳定（已核对正确）**：`exp(x − row_max) / row_sum(exp(x − row_max))`（49-52），减 max 防溢出。backward `out ⊙ (grad − row_sum(out ⊙ grad))`（67-71）等价 p*(e−p) 展开式，正确。
- **Linear 权重布局（已核对正确）**：w_(out,in)（34），forward `matmul(w_, input)` 即 W×x（110），bias `row_broadcast(b_)` 按行广播 (out,1)→(out,batch)（110）。backward `matmul(w_, grad, true)` = W^T×grad_out（122）、`matmul(grad, input, false, true)` = grad×x^T（126）、`row_reduce_sum(grad)` = Σ_batch grad_out（132）。全部正确。
- **Conv im2col 索引（已核对正确）**：`ih = oh*stride+kh−pad`（58）、`iw = ow*stride+kw−pad`（59），越界填 0（60-65），`r = ci*k²+kh*k+kw`（66）、`c = b*OH*OW+oh*OW+ow`（67）。col2im 累加（106-107）与 im2col 互逆。构造下溢守卫（153-163）核实**仍在**：`in_h+2*pad < kernel` 判无效 → init 报错（168-169）。MaxPool 守卫（302-312）`pool==0 || in_h<pool || in_w<pool` 判无效 → forward/backward 报错（321、372）。
- **GPU 录制生命周期**：本组 forward/backward 均无 `begin_batch`/`end_batch` 调用（批控制在 GPTModel 层，base:58-59 注释），局部 Tensor（如 attention 的 Q/K/V/W 574-581）在函数内 `std::move` 到成员或返回，无 end_batch 前析构风险。`FusedChainLayer` 用 `begin_expr`/`end_expr`（820/829）配对（expr 录制，非 batch 录制），t/u 中间量不逃逸（注释 785-786 约束），正确。
- **DSL 表达式 start/end 配对**：本组仅 FusedChainLayer 用 `begin_expr`/`end_expr`（820/829），配对正确。其余层用 `dsl::compute` 单表达式（无 begin/end），`dsl::compute_reduce` 用于归约表达式（LayerNorm/RMSNorm/attention m/l/R）。无配对断裂。
- **表达式常量是否含运行时值**：已核对 attention 的 m/l/W/R/X 表达式（478-547、680-687）仅含 `mask`（-inf 常量）、`m_t`/`l_t`（归约结果，作为 leaf 输入非常量）、`slopes`（batch_mod 视图，参数化非常量池）、`doc`（row_broadcast/batch_col 视图）。`scale_` 不进表达式（折进 Q 的 scale_inplace + grad 的 scale_inplace）。`inv_features`/`epsilon_` 在 (1,B) 小向量上用 `elementwise_binary_scalar` 施加（mlp:507-525、711-714），不进融合表达式常量池。全部符合铁律 6。
