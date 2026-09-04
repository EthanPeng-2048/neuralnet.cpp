# GPT + RAPT 层（L2）代码审查

## 模块概览

本组覆盖 `compute_layer_gpt.hpp`（GPTBlock / PositionEncoder 多态层次 / GPTModel，含 concat_cols 与 CrossAttention 注释）与 `compute_layer_rapt.hpp`（ReLULinearAttention / RAPTBlock / RAPTModel）。GPTModel 提供梯度检查点（`checkpoint_every_`）、activation offload（host-visible slab 持久缓冲）、文档感知掩码（`set_doc_ids` 逐层转发）与 batch flush 粒度；训练走 `CrossEntropyLoss::forward_sparse` 稀疏 CE（本组外，已核实不物化 one-hot）。RAPT 用 `scan_prefix_outer / scan_suffix_outer / outer_col` 三扫描原语实现 O(L·d_k²) 的 ReLU 线性注意力正反向与增量生成，文档边界处重置运行态。布局约定（batch-major，i = b·seq + t）在两文件内逐处核对一致，未发现 position-major 残留。

## 发现

### P1

**[P1-1] compute_layer_gpt.hpp:1000-1016 — `GPTModel::generate()` 的 prefill 与滑动窗口重建经 `fill_cache_` 逐 token 调 `forward_step`，未包 `begin_batch/end_batch`，GPU 下每步独立提交**
```cpp
auto fill_cache_ = [&](std::size_t start) -> Result<void>
{
    for (std::size_t i = start; i < context.size(); ++i)
    {
        auto r = forward_step(engine, context[i], cur_len, k_caches, v_caches, cur_len);
```
同文件 step 阶段（:1087-1093）已按注释 "P0: begin_batch/end_batch 包裹 forward_step，单次 GPU 提交" 包裹，但 `fill_cache_`（prefill 最多 seq_len_ 次、滑动窗重建再 ~seq_len 次）未遵守，与 `RAPTModel::generate`（step_one 在 batch 内）不对称。长序列生成时 O(seq) 次提交+同步是主要瓶颈。建议：`fill_cache_` 循环体包进 `engine.begin_batch()/end_batch()`（lambda 内或调用侧），`last_logits_t` 在 batch 结束后可统一处理，正确性不变。

### P2

**[P2-1] compute_layer_rapt.hpp:85-88 — `NN_ASSERT` 在 NDEBUG 下消失，d_k 偶数约束仅调试期生效；d_k>64 硬约束（docs/15 §7，GPU 栈预算）两文件均无构造期校验**
```cpp
NN_ASSERT(d_model % num_heads == 0 && (d_model / num_heads) % 2 == 0,
          "ReLULinearAttention: RoPE requires even d_k");
```
`NN_ASSERT` Release 展开为 `((void)0)`（core_assert.hpp:13）。偶数 d_k 在 `RotaryEmbedding::apply`（attention.hpp:101）有 runtime Error 兜底，但 `fill_pos_column_` 的 `d_k_/2` 截断先于校验执行（d_k 奇数时静默错）。更实际的是 d_k>64：CPU 参考无此限制，GPU `scan_*_outer_gpu` 才报 "not supported"（vk_backend.hpp:1674/1721/1762），错误推迟到 GPU 前向。建议：构造/`init` 内用 `NN_REQUIRE`（core_assert.hpp:27，始终启用的 Error 返回）校验 `d_k` 偶数 + GPU 场景 `d_k<=64`，与 `CausalSelfAttention` 构造期 Result 校验风格对齐。

**[P2-2] compute_layer_rapt.hpp:58-59 — `build_boundary_` 直接索引 `doc_ids_[b*seq+t]`，`set_doc_ids` 不校验 size ≥ batch·seq，传短 span 越界（UB）**
```cpp
if (t == 0 || doc_ids_[b * seq + t] != doc_ids_[b * seq + t - 1])
    boundary[b * seq + t] = 1;
```
对比 `CausalSelfAttention` 经 `build_attention_mask` 的 `NN_ASSERT(doc_ids.empty() || doc_ids.size() >= batch*seq_len, ...)`（attention.hpp:937）有守卫；GPTModel::set_doc_ids（gpt.hpp:686）与 RAPT 侧均无。现有调用方（text_train.cpp:1341、rapt_gradcheck.cpp:248）都传满 size，实际未触发，但 API 契约不闭合。建议：`forward` 内（唯一知道 batch/seq 的时点）校验 `doc_ids_.size() >= batch*seq` 不足则返回 Error。

**[P2-3] compute_layer_gpt.hpp:220-221 — `GPTBlock::forward` 中途失败时 `residual2_cache_` 半填充残留，后续 step 的 offload 布局可能错位（疑似）**
```cpp
if (!checkpoint_mode_)
    residual2_cache_ = res2;
```
`residual2_cache_` 在 `r2` 成功（:215-219）后写入，其后 `norm2_->forward`/`ff_.forward`（:223-227）失败直接 `return`，不清理。常规 `GPTModel::backward` 对 checkpoint 块先 `forward_recompute`（:107-115 内部 `set_checkpoint_mode(false)` 重建缓存）再 backward，时序闭合；offload 块 `export_activations`（:147-182）按 `activation_cache()` 中非空项建偏移，`import_activations` 按同一 refs 恢复——正常路径正确。风险在错误路径：forward 失败后 `residual2_cache_` 保留半旧值，若调用方重试同块 forward（新 batch/seq）且 offload slab 已按上轮布局建好（:154-168 惰性创建后不重建），`offload_save` 按新缓存逐项写但 slab 偏移沿用旧布局假设（:169-177 每次 forward 重算 offset，实际会重算——故常规重试安全）。标注"疑似"：需"forward 失败→clear_cache 未调用→再次 forward 且 offloaded_ 残留 true"的具体序列才能触发；`offloaded_` 在失败时未复位（:179-180 仅成功路径置 true，失败路径 :149 若 `offloaded_==true` 直接 return {} 跳过导出——残留 true 会导致下轮 `backward` 的 `import_activations` 从旧 slab 恢复）。建议：forward 失败路径（或 `clear_cache`）复位 `offloaded_ = false` 与 `residual2_cache_`。

**[P2-4] compute_layer_rapt.hpp:184-185,290-291 — forward 中途失败时 `batch_cache_/seq_cache_` 残留上一步旧值，调用方忽略 forward 错误直接 backward 会用错 shape 静默产生错误梯度（错误传播断裂边缘）**
```cpp
const std::size_t seq = (seq_len_ > 0) ? seq_len_ : total;
const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;
```
`Qp_cache_/Kp_cache_/V_re_cache_` 仅在 forward 成功末尾（:287-291）写入；forward 中途失败（如 :259 scan 错误）时三成员保持旧值，`batch_cache_/seq_cache_` 也保持旧值。若调用方未检查 Result（违反铁律 1）直接 `backward`，backward 用旧缓存+新 grad_output 静默出错。建议：forward 入口先 `batch_cache_ = 0; seq_cache_ = 0;`，backward 入口 `if (batch_cache_ == 0 || seq_cache_ == 0) return std::unexpected(Error{"backward before successful forward"})`，把"forward 未成功"显式化（与 AGENTS.md §10.8 "gradcheck 必须先 forward 填充 input_cache_ 再 backward，否则空缓存崩溃"同类）。

**[P2-5] compute_layer_rapt.hpp:427-434 — 双向分支物化 (BH·dk², seq) 的 `Ab/Bb` 两个大张量 + 两次 broadcast，O(B·H·d_k²·L) 内存，与因果分支 O(L·d_k²) 运行态后缀扫描量级差 ~100×**
```cpp
Tensor Ab = engine.create_tensor(BHdk * d_k_, seq);
{ auto r = engine.zero(Ab); if (!r) return std::unexpected(r.error()); }
{ auto r = engine.broadcast_row_inplace(Ab, *dA_sum_r, BinaryOp::Add); ... }
```
数学正确（双向时 dA/dB 沿 seq 为常数，`scan_suffix_outer(causal=false)` 退化为 `S_i = D_i`，broadcast 后 `S·X` = 常数矩阵×列向量，与 CPU 参考 :719-721 一致）。d_k=32/H=8/L=2048 时单张量 ~6.7MB/层，训练多块累积可观。建议：改为"常数 (BH·dk²,1) × 逐列向量"的直接 matvec（`batched_matmul` 或逐列原语），消除 seq 维物化。正确性无问题，属性能项。

**[P2-6] compute_layer_rapt.hpp:550-553 — `forward_step` 恒传 `has_state=true` 与 A/B_state（首 token 状态全 0），每次多绑两个 (H·dk, dk) 张量**
```cpp
auto A_add_r = engine.batched_matmul(*Kp, *Kp, H, false, true);  // k·k^T → (H*dk, dk)
```
功能正确（首 token A0=B0=0，gradcheck 已覆盖）。`generate` 的 A/B_state 每块 (d_model, d_k) 且已零初始化（:931-936），首 token 前恒为零。建议：`generate` 维护 `bool has_history` 标志，首 token 传 `dummy/dummy, false`，之后传真实状态，省 GPU 端两次 buffer 绑定。

**[P2-7] compute_layer_gpt.hpp:955-957 & rapt.hpp:1015-1021 — 两处 generate 的采样累加 `cumulative += p` 存在 ~1ulp 回退 `vocab_size_-1`（浮点归一化舍入使 Σp 略小于 1 且 r 落入间隙时）**
```cpp
cumulative += last_logits[v];
if (r <= cumulative) { next_token = v; break; }
```
标准写法，概率 ~1e-7，实际无影响。两处代码完全相同（GPT 侧 :1044-1051、RAPT 侧 :1015-1021），若未来引入确定性采样需统一处理。建议：保持现状，注释说明 1ulp 回退语义。

**[P2-8] compute_layer_gpt.hpp:591-592 — `GPTBlock` 构造传 `max_len = seq_len`，ALiBi 模式下 `forward_step` 的 cur_len 若 ≥ seq_len_ 则偏置表越界（疑似，当前 generate 滑动窗 cap 保证不可达）**
```cpp
blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, seq_len,
                     pos_enc_type, activation, norm_type);
```
`GPTBlock(d_model, num_heads, d_ff, max_len=1024, seq_len=0, ...)` 中 max_len 传给 `CausalSelfAttention` 作位置偏置上界。GPT::generate 的滑动窗（:946-958）保证 cur_len < seq_len_，RoPE 模式不受影响；但外部直接调 `forward_step`（cur_len 无 cap）或未来放宽窗口时 ALiBi 越界。标记"疑似"（未读 CausalSelfAttention::init 中 max_len 的具体用途，无法 100% 确认越界路径）。建议：max_len 传 `seq_len*2` 或在 `forward_step` 文档声明 cur_len < seq_len_。

**[P2-9] compute_layer_rapt.hpp:971-975 — `RAPTModel::generate` 的 `step_one` 每步 `to_matrix` 整个 logits（(vocab,1)），vocab 大时（5 万+）每步 ~200KB CPU 拷贝，与 GPT 侧"logits 直接 (vocab,1) 无需 transpose+slice"的优化不对称**
```cpp
auto lm = engine.to_matrix(*logits);
```
正确性无问题（GPU 路径 gradcheck 已验证）。建议：保留 Tensor 仅采样列，或加 `gather_cols` 原语，与 GPT 侧对齐。属性能项。

**[P2-10] compute_layer_gpt.hpp:165,173,190 — offload 缓冲单位约定不一致：`create_offload_buffer(total)` 实参是 float 数（`ref.get().size()`），接口签名参数名 `bytes`（compute_engine.hpp:148）；`offload_save` 的 `offset` 是 float 数（内部 ×sizeof(float)），而 `offload_restore` 的 `offset` 同 float 数——各自自洽但命名误导**
```cpp
auto slab = engine.create_offload_buffer(total);   // total = float 数
```
GPU 实现 `create_offload_buffer` 把参数当字节用（`create_host_visible_empty(1, bytes, ...)`，gpu_engine.hpp:127-132），而 `offload_save/restore` 的 offset 是 float 数再乘 sizeof(float)（:141,149-156）——同一 buffer 的"总大小按字节、偏移按 float"双单位约定，极易误用。建议：统一为 float 数（参数改名 `floats`）或统一字节（offset 也 ×sizeof(float) 后传）。

### P3

**[P3-1] compute_layer_gpt.hpp:1108-1138 — 文件末尾 `CrossAttention` 大段算法注释无对应类体（死注释），该类实际在 compute_layer_zipt.hpp**
```cpp
// CrossAttention — 阶段一：全局上下文重要性压缩（AttnZip Memory Queries）
// ...（算法注释，~30 行）...
} // namespace nn
```
`concat_cols`（:1085-1106）是 live 代码（ZiPT 阶段二用），但其后 CrossAttention 注释无类定义。建议：注释段移到 compute_layer_zipt.hpp 对应类上方或删除。

**[P3-2] compute_layer_rapt.hpp:85-88 — 两条 `NN_ASSERT` 条件重叠（第一条是第二条前缀），且 even-d_k 约束在 `use_rope_==false` 时也生效（ALiBi 不需要 even d_k，v1 强制 RoPE 不可达但语义过强）**
```cpp
NN_ASSERT(d_model % num_heads == 0, ...);
NN_ASSERT(d_model % num_heads == 0 && (d_model / num_heads) % 2 == 0, "RoPE requires even d_k");
```
建议合并为 `NN_ASSERT(d_model % num_heads == 0 && (!use_rope_ || (d_model/num_heads) % 2 == 0), ...)`。

**[P3-3] compute_layer_rapt.hpp:64-70 — `make_dummy_` 每次 forward/backward 都 `create_tensor(1,1)+zero`，forward 调两次、backward 调两次；GPU 下每步 4 次小张量分配**
```cpp
[[nodiscard]] Result<Tensor> make_dummy_(ComputeEngine& engine)
{ Tensor d = engine.create_tensor(1, 1); auto r = engine.zero(d); ... }
```
建议：forward 内 boundary-dummy 与通用 dummy 合一（无 doc 时相同），或缓存 (1,1) 成员。

**[P3-4] compute_layer_gpt.hpp:1043 — `generate` 滑动窗 `keep = seq_len_ - 1` 在 `seq_len_==0` 时下溢；`GPTModel` 构造不校验 `seq_len>0`（domain_gpt.hpp:50 校验了，但 GPTModel 可独立构造）**
```cpp
const std::size_t keep = seq_len_ - 1;
```
建议：GPTModel 构造校验 `seq_len_ > 0`（与 `vocab_size_`/`d_model_` 同列）。

**[P3-5] compute_layer_rapt.hpp:1015 — 采样循环 `std::vector<Scalar> lastv = last;` 每步复制 (vocab) 向量做 temperature 缩放，vocab 大时每步 ~200KB 分配+拷贝；GPT 侧（gpt.hpp:992）in-place `v /= temperature` 无此开销**
```cpp
std::vector<Scalar> lastv = last;
if (temperature > 0.0 && temperature != 1.0)
    for (auto& x : lastv) x /= temperature;
```
建议：对齐 GPT 侧 in-place 风格（缩放后采样完即弃，或复用缓冲）。

**[P3-6] compute_layer_gpt.hpp:560 — `stored_tokens_tensor_` 注释标 "(total, 1)"，实为 `clone(*input_T)` 的 (batch·seq, 1)；`batch_size_` 是跨 forward/backward 的隐式状态（:622 写、:860 读），gradcheck 必须先 forward 填充（AGENTS.md §10.8）。建议注释显式标注"backward 依赖本 step forward 成功"。**

**[P3-7] compute_layer_rapt.hpp:550-553 — `batched_matmul(Kp,Kp,H,false,true)`/`batched_matmul(V,Kp,H,false,true)` 产出 (H·dk, dk) 与 scan 的 A0/B0 约定一致（cpu_engine.hpp:537-539 校验 `h*dk+a` 行块），正确；建议加注释标注"行块 h = [h·dk, (h+1)·dk)"对齐 scan 读取方式，降未来错位风险。**

**[P3-8] compute_layer_gpt.hpp:722-760 — 梯度检查点 forward 循环失败路径不改变块模式一致性（:735-745 每步重设 checkpoint_mode），`forward_recompute`（:107-115）失败后残留的 true 也是正确状态；逻辑闭合，建议加注释说明"失败路径不改变块的模式一致性"防未来误改。**

**[P3-9] compute_layer_rapt.hpp:712-721 — `RAPTBlock::forward_step`/`ReLULinearAttention::forward_step` 不校验 A_state/B_state 形状（(d_model, d_k)），传错时错误推迟到 scan 原语（:547 的 (H·dk,dk) 校验）。建议在 `forward_step` 入口 `NN_REQUIRE` 校验，与 forward 的 shape 校验（:180-181）对齐。**

**[P3-10] compute_layer_rapt.hpp:149-153 & gpt.hpp:686-689 — `set_doc_ids` 传空 span 清除文档感知，但 `GPTModel::set_doc_ids` 清除后不转发"清除"给已 set 过 doc_ids 的 `CausalSelfAttention`（attention 侧 `has_doc_ids_` 残留 true 直到下次 forward 重新 set）——实际每次 forward 都 `blocks_[bi].set_doc_ids(doc_ids_)`（:730）覆盖，故无残留；仅当"set 非空→set 空→不 forward 直接 backward"序列时 attention 侧 has_doc_ids_ 残留（疑似，边界场景）。建议 `set_doc_ids` 空 span 时也转发给子层（GPTModel 侧当前只清 `doc_ids_` 成员，不清子层——但子层每次 forward 重设，实际闭合）。**

## 已知问题核对

- **set_doc_ids 残留**：仍存在，核实形态——`GPTModel::set_doc_ids`（gpt.hpp:686-689）仅存 `doc_ids_` 成员，forward 循环内逐 block 转发（:730）→ `GPTBlock::set_doc_ids`（:77-79）→ `CausalSelfAttention::set_doc_ids`（attention.hpp:1060）重建 `doc_ids_cache_`（(1, BH·seq) BatchCol 布局）+ `doc_col_`。残留形态为"逐层转发 + 每 step 重建掩码"（attention.hpp:1080 注释"doc_ids 每 batch 变化，掩码不可缓存，每步重建"），与 08-28 全库审查"第三梯队 set_doc_ids 残留"记录一致，仍未修。RAPT 侧（rapt.hpp:149-153）同形态：每 step `build_boundary_` 重建 (1, batch·seq) boundary 张量（:238-248）。本组内无 set_doc_ids 新 bug，仅确认旧残留仍在。

## 其他观察

1. **布局一致性（铁律 4）逐处核对通过**：GPTModel forward `input_T` (batch, seq) flat 序 = b·seq+t（:636 注释明确）；`pos_indices_cache_` 按 b 外层/t 内层生成 position=t（:383-386）；RAPT `rearrange_3d(..., false)` 从 (H·dk, batch·seq) batch-major 到 (batch·H·dk, seq)（:200-205）；boundary (1, batch·seq) 按 b·seq+t 索引（:241-243）。text_train.cpp:1371-1374 的 position-major 源（`src_t*bs+b`）→ batch-major 目标（`b*eff_seq+t`）转换正确。未发现 position-major 串扰风险。

2. **扫描原语调用参数（dk 维度）核对**：`scan_prefix_outer(K,V,P,R,A0,B0,has_state,d_k_,num_heads_,causal,boundary,has_bnd)`——rapt.hpp:259-261（forward：P=Qp,R=V,A0/B0=dummy,has_state=false）、:346-349（backward pass1：R=gcr）、:547-548（forward_step：has_state=true,A0/B_state）均正确传 `d_k_, num_heads_`；`scan_suffix_outer(D,X,Y,d_k_,num_heads_,causal,...)`——:398-404（causal）、:435-441（双向，传 dummy,false）正确；`outer_col(P,R,S,d_k_,has_scale)`——:390（dA：S=t1,has_scale=true）、:392（dB：S=dummy,has_scale=false）正确。与 compute_cpu_engine.hpp:515-801 参考实现语义一致（prefix [0)B·P [1)A·P [2)B^T·R [3)s [4)r；suffix [0)S·X [1)S·Y [2)S^T·Y）。**双向 `row_reduce_sum` 是 row 不是 col**（:423-425 对 (BH·dk², seq) 按行求和得 (BH·dk²,1)，broadcast 回 (BH·dk², seq)）——正确，符合 docs/15 双向"全集常数沿 seq 广播"。

3. **forward_step 先扫描后更新**（铁律重点）：rapt.hpp:547-557 先 `scan_prefix_outer`（读 A/B_state 旧值，含自身累加）再 `add_inplace` 更新运行态供下一 token——顺序正确，与 docs/15 §7 及 memory 09-04 记录"forward_step 先扫描后更新"一致（gradcheck 已验证）。

4. **Qp/Kp 是 Result 需要 `*`**：rapt.hpp:259 `*Kp`、:346 `Kp_cache_`（成员，forward 末尾 :287 `std::move(*Qp)` 已解包存储）、:390 `Qp_cache_`（成员）、:547 `*Kp`/`*Qp`——本地 Result 变量（`Qp`/`Kp`）均 `*` 解包，成员缓存已解包直接用，无未解包 Result 传入引擎原语。

5. **DSL 表达式中的运行时值（铁律 7）**：两文件 `dsl::compute` 表达式全为结构常量（max/leaf/select/add），无运行时值进 ExprSpec key；RLA 的 `1e-6`（eps）与 `0.5`/`2.0`（scale）为 `Scalar` 字面量经 `elementwise_binary_scalar`/`scale_inplace` 传递，不进 key——符合 S7 处理模式（AGENTS.md §12 S7 教训 1）。RoPE 的 cos/sin 表是 Tensor 叶子（`dsl::leaf`），非运行时常量——正确。

6. **录制期局部张量生命周期（铁律 6）**：GPT generate step 阶段（:1087-1093 begin_batch→end_batch）内的 `logits_res`（:1090 局部）在 end_batch 后 `std::move` 给 `last_logits_t`（成员级，覆盖整个 generate）——存活。`fill_cache_` lambda 捕获的 `k_caches/v_caches`（引用参数）与 `last_logits_t`（this 引用）生命周期覆盖整个 generate——正确。GPTBlock forward 的 `res2`（:219 局部）在 `dsl::compute`（:229-231）消费后函数返回，无跨 end_batch 悬空。RAPT forward 的 `Qp/Kp/V`（:222-289 局部）末尾 move 进成员——正确。未发现录制期 use-after-free。

7. **GPT 训练路径（forward_sparse）**：compute_loss.hpp:267-349 已核实——`fused_forward_sparse_` 用 `col_reduce_max` + `col_reduce_sum(exp(logits - col_max))` 算 denom（不物化 softmax 矩阵），loss_vec 用 `row_gather(logits, labels_t)`（RowGather 原语，不物化 one-hot），grad 用 IR 表达式——符合铁律 9。GPTModel backward 的 `lm_head_.backward`（:835）消费 (vocab, total) 梯度——链路闭合。

8. **位置编码索引正确性**：LearnedPositionEncoder `pos_indices_cache_`（:382-386）按 batch-major 生成 position=t；`gather_rows(pos_emb_, pos_indices)` → (total, d_model) → transpose → (d_model, total) 加到 token 嵌入（:406-411）——索引 = t（0..seq-1），与 batch-major 列序 i=b·seq+t 对齐。Sinusoidal 公式（:500-502）PE(pos,2i)=sin/PE(pos,2i+1)=cos，标准。RoPE `fill_pos_column_`（attention.hpp:36-51）LLaMA 式 cos 前后半相同 + rotate_half；`apply`/`apply_step` forward=Add/backward=Sub（正交旋转逆=反角）——正确。GPT generate 滑动窗下 RoPE 位置从 0 重置是滑动窗语义（重建 cache），与 RAPT "pos 自然递增无滑动窗"设计意图不同，非 bug。
