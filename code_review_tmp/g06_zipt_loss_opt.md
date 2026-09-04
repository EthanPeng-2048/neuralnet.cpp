# ZiPT + Loss + Optimizer（L2）代码审查

> 审查范围：`include/neuralnet.cpp/compute_layer_zipt.hpp`（49.8KB，1075 行）、`compute_loss.hpp`（376 行）、`compute_optimizer.hpp`（665 行）。
> 全部 file:行号 与代码引用均来自真实 read/grep；跨文件引用用 grep 核实（concat_cols、Tensor/valid、NN_ASSERT、set_doc_ids 等）。

## 模块概览

本组三个 L2 文件：`CrossEntropyLoss`（稠密/稀疏双路径，稀疏用 IR `row_gather`/`col_broadcast` 不物化 (vocab, total) one-hot）；`MSELoss`；五个优化器 `SGD`/`SGDWithMomentum`/`Adam`/`AdamW`/`Muon`（含 quintic Newton-Schulz 正交化）+ `create_optimizer` 工厂；ZiPT 三件套 `CrossAttention`（AttnZip 阶段一记忆压缩）/ `ZiPTBlock`（阶段二局部-全局联合注意力，Pre-Norm + FFN + 文档感知掩码）/ `ZiPTModel`（token_emb + 位置编码 + 压缩器 + N 块 + LM head，含 generate 采样）。所有 Layer 用引擎原语组合表达算法，符合分层铁律。

## 发现

### P0

（无。CE 稠密/稀疏、MSE、SGD/SGD-momentum/Adam/AdamW/Muon 公式均逐项核对无误；forward_sparse 无 one-hot 物化；布局约定 batch-major 一致。）

### P1

**[P1] compute_layer_zipt.hpp:844-850 — ZiPTModel::forward 每次 forward 都 clone 整个 input_T (seq, batch) 存到成员 `stored_tokens_tensor_`，无 checkpoint 路径；长序列大 batch 时这是全序列激活的第二份常驻副本，backward 才用。**

```cpp
auto input_T = engine.transpose(input);   // (batch, seq) batch-major
auto all_emb = engine.gather_rows(token_emb_, *input_T);
auto st = engine.clone(*input_T);
stored_tokens_tensor_ = std::move(*st);
```

- 与 `ZiPTBlock` 的 `residual2_cache_`（有 `checkpoint_mode_` 守卫，:481-482）不同，`stored_tokens_tensor_` 无 checkpoint 门控——即使未来支持 checkpoint，这份副本仍常驻。
- 建议：若引入 `forward_recompute`，在 checkpoint 模式下改为 backward 时从 input 重建（input 是 forward 参数，可保留引用到 backward，与 `ZiPTBlock::input` 处理一致）；或显式标注为"ZiPT 不支持 checkpoint 激活省显存"的已知限制（docs/12 §10 已提及不支持 checkpoint，但此处未同步降级）。
- 风险：GPU 路径下额外 (seq·batch) float 常驻显存；seq=4096、batch=32 时 ≈ 512KB，本身不大，但违反"大中间张量物化前先做内存预算"的铁律精神（docs/08 坑 7）。

**[P1] compute_optimizer.hpp:210-215 / 325-338 / 570-577 — 优化器构造函数在 buffer 创建失败时 `std::abort()`，违反铁律 1（禁 throw/try/catch，错误走 Result 由调用方传播）；与同文件 step() 的 `Result<void>` 错误传播风格不一致。**

```cpp
auto v_r = create_zero_buffers_();
if (!v_r) {
    std::fprintf(stderr, "SGDWithMomentum init failed: %s\n", v_r.error().message.c_str());
    std::abort();
}
```

- 同文件 `Adam::init_moments_()`（:322-340）与 `Muon` 构造（:570-577）共 4 处 `abort()`。
- `create_zero_buffers_()` 返回 `Result<std::vector<Tensor>>`，本可把错误传播给调用方（构造器签名改 `Result<unique_ptr<Optimizer>>`，或工厂 `create_optimizer` 返回 `Result`）。
- 影响：buffer 分配失败（极端 OOM）时进程直接终止，无 checkpoint/降级机会；与项目"硬报错、不降级"的整体风格冲突（但 abort 比错误码更硬）。
- 建议：工厂函数 `create_optimizer` 改为返回 `Result<std::unique_ptr<Optimizer>>`（`make_unique` 失败路径统一传播）；或至少在 docs/17 指针审查中登记为已知债（当前无 nn-allow 注释）。

**[P1] compute_loss.hpp:303-311 — forward_sparse 越界 label 静默修正为 0，而非返回 `std::unexpected`；违反"错误传播"铁律，调用方无法区分"合法 label 0"与"越界被修正"。**

```cpp
for (std::size_t i = 0; i < total; ++i)
    sp[i] = static_cast<Scalar>(
        (labels[i] < vocab_size) ? labels[i] : 0);
```

- 注释（:304）自承"GPU RowGather 无越界守卫，靠 mask 置零无效"——但实际 mask 在 :323 确实会把越界位置清零（`labels[i] < vocab_size ? 1 : 0`），所以"靠 mask 置零无效"的说法不成立；真正的问题是修正到 0 而非报错。
- 后果：若上游 tokenizer 产出越界 label（如 vocab 扩展后旧 checkpoint 不匹配），训练静默进行，loss 数值看似正常（越界位置被 mask），但对应位置的梯度被错误归到 class 0——梯度污染但 loss 不报警。
- 建议：越界 label 应直接 `return std::unexpected(Error{"sparse CE: label out of range"})`，让调用方在数据层修复；若必须容忍（如动态 vocab 训练），至少 `fprintf(stderr, ...)` 一次告警（不循环刷屏）。

### P2

**[P2] compute_layer_zipt.hpp:390-400 / 790-794 — `set_doc_ids` 残留（与 docs/12 全库审查"第三梯队：set_doc_ids 残留"清单一致）；`ZiPTModel::set_doc_ids`（:790）不校验 `ids.size() == batch*seq_len_`，`ZiPTBlock::set_doc_ids`（:389）不校验 `ids.size() == batch*window_`，越界读 UB。**

```cpp
void set_doc_ids(std::span<const std::size_t> ids) override {
    if (ids.empty()) { doc_ids_.clear(); return; }
    doc_ids_.assign(ids.begin(), ids.end());
}
```

- `build_mask_`（:286-289）按 `doc_ids_[b*window_ + jl]` 索引，调用方传短 span 时 `std::vector::operator[]` 越界（UB，无 assert）。
- 建议：在 `set_doc_ids` 加 size 校验（与 `batch_cache_`/`seq_len_` 比对），不符则 `fprintf(stderr)` 告警 + 回退到空 span（纯因果）；或在 docs/12 标记为"待修：set_doc_ids 无 size 校验"。
- 注：GPT/RAPT 的 `set_doc_ids`（grep 见 compute_layer_gpt.hpp:77/686、compute_layer_rapt.hpp:149/641/853）同模式，属跨文件共性问题，本组只报 ZiPT 侧。

**[P2] compute_layer_zipt.hpp:780-786 — `set_checkpoint_every` 用 `static_cast<std::size_t>(-1)` 打印"请求的 stride"，实际值是 SIZE_MAX 而非调用方传入的真实 stride，误导调试。**

```cpp
void set_checkpoint_every(std::size_t /*stride*/) override {
    std::fprintf(stderr, "FATAL: ZiPTModel does not support gradient checkpointing "
                         "(no forward_recompute); requested stride=%zu\n",
                 static_cast<std::size_t>(-1));
    std::abort();
}
```

- 参数名 `/*stride*/` 匿名化，但 fprintf 却打印 `SIZE_MAX`（= 18446744073709551615 on 64-bit），调用方看到的"requested stride=18446744073709551615"毫无意义。
- 建议：参数命名 `stride` 并打印 `stride`；或干脆不打印数值，只打印"ZiPTModel 不支持 checkpoint"。

**[P2] compute_optimizer.hpp:480-483 — `newton_schulz_orthogonalize` 每步从 GPU 下载 Frobenius 范数标量（`to_matrix` → `at(0,0)`），5 步 = 5 次 PCIe 往返；GPU 路径下可合并为一次下载（5 步共用同一 `inv_norm_scalar`，当前实现已如此——但每步 `matmul` 后 X 范数已变，理论上应重算；此处是"首步归一化后迭代"，数学上 5 步共用首步 norm 是 Muon 参考实现约定，非 bug）。**

> 经核对：KellerJordan/Muon 参考实现确实用首步 `X = G / (||G||_F + eps)` 一次归一化后 5 步 NS 迭代（迭代本身保持正交性），故 5 步共用 `inv_norm_scalar` 是正确的。**本条降级为 P3（性能建议）**：若未来 NS 步数 >5 或换迭代式，需重新评估是否每步重算 norm。

**[P2] compute_optimizer.hpp:642-662 — `create_optimizer` 工厂对 "sgd_momentum"/"muon" 不传 `weight_decay`，调用方若期望 sgd_momentum 带 wd 会静默丢失（PyTorch `SGD(weight_decay=...)` 语义）。**

```cpp
if (name == "sgd_momentum")
    return std::make_unique<SGDWithMomentum>(engine, std::move(params), std::move(grads), lr);
```

- `SGDWithMomentum` 构造无 wd 参数；`weight_decay` 参数（:648）只对 "adamw" 生效（:657-658）。
- 文档（AGENTS.md §8）未说明 wd 仅对 adamw 有效，调用方易误解。
- 建议：① 在 `create_optimizer` 注释中明确"weight_decay 仅对 adamw 生效"；② 或给 `SGDWithMomentum` 加 wd 参数（`g' = g + wd*p`，L2 形式）使 sgd/sgd_momentum 也支持 wd；③ 至少在 docs/05-algorithm-reference.md 标注。

**[P2] compute_loss.hpp:147-190 — `softmax_cols_` 稠密路径物化 (classes, batch) 的 `shifted` + `exp_shift` 两份 (classes, batch) 张量（共 2 次全 softmax 大小分配）；大 vocab（25k）+ 大 batch 时是 2×25k×batch float ≈ 200MB（batch=4096）。**

```cpp
auto shifted = clone_tensor(engine, logits);   // (classes, batch)
...
auto exp_shift = engine.elementwise_unary(UnaryOp::Exp, *shifted);  // (classes, batch)
```

- 稠密路径本就要求 (classes, batch) target（one-hot），调用方已物化一份；本函数再物化 2 份中间量。
- 建议：① 文档明确"稠密路径仅用于小 vocab（≤ 10k），大 vocab 用 forward_sparse"；② 或 `shifted` 就地复用 `exp_shift` 的 buffer（`exp_shift` 计算完后 `shifted` 已不需要，可 `shifted = std::move(*exp_shift)` 前先把 exp 结果存到 shifted 自身——需 engine 支持"读 X 写 X"的 in-place unary，当前 `elementwise_unary` 返回新 Tensor 不支持）。
- 注：稀疏路径（:294-372）已用 IR 不物化全 softmax，符合铁律 9；本条仅针对稠密路径。

**[P2] compute_layer_zipt.hpp:271-272 — `build_mask_` 掩码缓存以 `batch` 为键，但 `has_doc_ids_` 变化（set_doc_ids 切换）时缓存失效逻辑不完整：`set_doc_ids`（:389-400）设 `has_doc_ids_=true` 后不清 `mask_cache_`/`mask_batch_`，下次 `build_mask_` 命中 `!has_doc_ids_ && mask_cache_.valid() && mask_batch_ == batch` 短路返回旧掩码。**

```cpp
if (!has_doc_ids_ && mask_cache_.valid() && mask_batch_ == batch)
    return mask_cache_;
```

- 场景：step 1 无 doc_ids → 构建掩码 M1 缓存；step 2 `set_doc_ids(non-empty)` → `has_doc_ids_=true`，但 `mask_cache_` 仍是 M1；step 3 `build_mask_` 看到 `has_doc_ids_=true` 走完整构建（正确）；step 4 `set_doc_ids(empty)` → `has_doc_ids_=false` 且 :395 显式 `mask_cache_ = Tensor{}`（正确清除）。
- 经核对：:395 在空 span 分支已清 `mask_cache_`，:399 在 non-empty 分支设 `has_doc_ids_=true`（:271 短路条件 `!has_doc_ids_` 为 false，不会误命中）。**本条降级为 P3**：当前实现正确，但"set_doc_ids 切回空"时 `mask_batch_` 未重置（仍保留旧 batch），下次 batch 变化重建时 `mask_batch_ == batch` 可能误命中——但 :395 已清 `mask_cache_`，`mask_cache_.valid()` 为 false，短路不触发。**最终判定：无 bug，仅代码可读性建议**（在 :395 同时 `mask_batch_ = 0`）。

**[P2] compute_optimizer.hpp:597-608 — Muon Nesterov 更新方向 `update = g + μ*v`（v 已含 g：`v = μ*v + g`），与标准 Nesterov-Muon 参考实现（KellerJordan）的 `update = g + μ*(μ*v + g)` 不一致——当前实现实际是"对 v 再做一次 Nesterov lookahead"，数学上是 `update = g + μ*v = g + μ*(μ*v_old + g) = (1+μ)g + μ²*v_old`，而非参考的 `(1+μ)g + μ*v_old`。**

> 经对照 KellerJordan/Muon 参考实现（`update = g + momentum * (momentum * v + g)`，其中 v 是先更新为 `momentum*v + g` 后的新 v）：参考实现的 Nesterov 是 `update = g + μ * v_new`（v_new = μ*v_old + g），即 `update = g + μ*(μ*v_old + g) = (1+μ)g + μ²*v_old`。**当前实现与参考一致**，本条撤销。

**[P2] compute_layer_zipt.hpp:122-164 — `CrossAttention::forward` 的 `seq_len_` 参数语义模糊：`seq_len_=0` 时 `L=total, batch=1`（:127-128），但 `ZiPTModel` 构造时（:711-712）传的是 `(window < seq_len) ? (seq_len - window) : seq_len`，split 模式下 `seq_len_=hist_len`，非 split 模式 `seq_len_=seq_len`；`batch=total/seq_len_` 依赖调用方保证 `total % seq_len_ == 0`，无运行时校验，非整除时 `batch` 截断（UB）。**

```cpp
const std::size_t L = (seq_len_ > 0) ? seq_len_ : total;
const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;
```

- `total % seq_len_ != 0` 时（如 batch 边界样本被截断），`batch` 偏小，`rearrange_3d` 会读越界（UB）。
- 建议：加 `if (total % seq_len_ != 0) return std::unexpected(Error{"CrossAttention: total not divisible by seq_len_"});`。
- 注：`ZiPTModel::forward`（:839-841）已校验 `input.rows() == seq_len_`，且 `input.cols()` 是 batch，故 `total = seq_len_*batch` 整除；但 `CrossAttention` 是 public class，可能被其他调用方直接构造使用，建议加防御性校验。

### P3

**[P3] compute_optimizer.hpp:107-111 — `clip_grad_norm` 用 `row_reduce_sum` → `col_reduce_sum` 两步归约，而 `MSELoss::forward`（:95-97）与 `newton_schulz_orthogonalize`（:473-475）也用同样顺序；但 `MSELoss` 注释（:94）说"先按列归约 (1, cols)，再按行归约 (1, 1)"，实际代码是 `col_reduce_sum` 后 `row_reduce_sum`——注释与代码一致，但"先列后行"的命名易误导（行归约是沿列方向收缩，列归约是沿行方向收缩）。建议注释统一用"沿行归约→沿列归约"或"row-reduce→col-reduce"。**

**[P3] compute_loss.hpp:14-23 — 文件头注释描述的 CE 算法（:16-22）用 `col_max` 数值稳定，但代码（:150-158）用 `col_reduce_max` 原语 + `dsl::compute_reduce` IR 表达式，注释未提及 M5 融合路径（IR 单 kernel，不物化 exp 张量）。建议注释与 M5 实现同步。**

**[P3] compute_optimizer.hpp:205 / 358 / 411 — `SGDWithMomentum`/`Adam`/`AdamW` 默认参数（`beta=0.9`、`beta1=0.9`、`beta2=0.999`、`eps=1e-8`）与 PyTorch 默认一致，但未用 `constexpr` 或命名常量，散落多处（:358 与 :411 重复 `0.9/0.999/1e-8`）。建议提取为 `namespace` 常量（如 `constexpr Scalar kBeta1 = 0.9f;`），避免未来改一处漏另一处。**

**[P3] compute_layer_zipt.hpp:69 / 744 — `CrossAttention::init` 用 `std::random_device{}()` 作 `mt19937_64` 种子，而 `ZiPTModel::init`（:744）用固定种子 `42`。前者每次运行 P 初始化不同（跨运行不可复现），后者 token_emb 固定。若要求确定性（铁律 7），P 初始化也应固定种子或接受外部 RNG。建议：P 初始化也用固定种子（或从 `ZiPTModel` 传入 RNG）。**

**[P3] compute_loss.hpp:333-334 — `inv_num_valid` 计算中 `num_valid` 是 `std::size_t`，`static_cast<Scalar>(num_valid)` 在 `num_valid > 2^24` 时 float 精度丢失（但 GPT 实际 batch 远小于此，非实际风险）。建议注释标注"num_valid > 2^24 时 float 精度丢失，但实际 GPT batch < 10^4，无影响"。**

**[P3] compute_layer_zipt.hpp:974 — `generate` 用 `std::random_device{}()` 作采样种子，每次运行生成不同序列（非确定性）。若要求可复现（如 A/B 对比），建议接受外部 `std::size_t seed` 参数。**

**[P3] compute_optimizer.hpp:129 — `clip_grad_norm` 用 `m_r->at(0, 0)`（带 assert 校验），而 `MSELoss::forward`（:104）、`CrossEntropyLoss::forward`（:242）、`forward_sparse`（:370）用 `at_unchecked(0, 0)`（无校验）。两者都读 (1,1) 张量，风险等价；建议统一用 `at_unchecked`（(1,1) 形状已由上游归约保证）或统一用 `at`。**

**[P3] compute_layer_zipt.hpp:542-556 — `ZiPTBlock::backward` 的 `split_cols_` lambda 用 `transpose + slice_rows + transpose` 三次操作拆分 (M+W) 列，GPU 路径下 3 次 PCIe 往返（transpose 物化全 (M+W, batch·H·d_k)）。建议：未来若有 `slice_cols` 原语（直接列切片），可省 2 次 transpose；当前实现正确，仅性能建议。**

**[P3] compute_optimizer.hpp:548-637 — `Muon` 对 1D 参数（bias，rows==1 或 cols==1）走标准 SGD 更新（:628-633），对 2D 参数走 NS 正交化 + 0.2√max(m,n) 缩放（:610-627）。但 `create_optimizer`（:659-660）传参时 `Muon` 默认 `momentum=0.95, nesterov=true, ns_steps=5, ns_eps=1e-7`，调用方无法调 ns_steps/ns_eps（工厂只接受 name/engine/params/grads/lr/wd）。建议：扩展工厂签名或提供 `create_muon(...)` 专用工厂。**

**[P3] compute_loss.hpp:305-311 / 317-325 — `forward_sparse` 在 CPU 端构造 `labels_m`/`mask_m` 时逐元素循环（:308-310, :320-324），大 total（>10^5）时 CPU 循环是瓶颈。建议：用 `std::transform` 或 SIMD intrinsics 加速；或 GPU 路径下用 `from_matrix` 前在 GPU 端做越界修正（需新增小 kernel）。当前实现正确，仅性能建议。**

## 已知问题核对

- **set_doc_ids 残留**：AGENTS.md §12"未修：第三梯队（…set_doc_ids 残留）"。本组 ZiPT 侧有 2 处 `set_doc_ids`（`ZiPTBlock`:389、`ZiPTModel`:790），均无 size 校验（见 P2-1）。grep 核实 GPT（compute_layer_gpt.hpp:77/686）与 RAPT（compute_layer_gpt.hpp:149/641/853）同模式，属跨文件共性残留。**核实：残留存在，未修，本组只核不报新（P2-1 已报 ZiPT 侧）。**
- **Muon 0.2√max(m,n)**：AGENTS.md §12"P2 已修：Muon 0.2√max(m,n)"。本组核对 `compute_optimizer.hpp:620-625`：`muon_scale = 0.2 * sqrt(big)`，`big = max(m,n)`，`axpy_inplace(params, -lr*muon_scale, ortho_update)`。**核实：公式正确，与参考实现一致，已修。**
- **CE/Adam/AdamW/SGD 公式**：AGENTS.md §12"CE/Adam/AdamW/SGD 公式已验证正确"。本组逐项核对：
  - CE 稠密：`grad = (softmax - target)/batch`（:215-221），`loss = -Σ target·log_softmax / batch`（:227-242）✓
  - CE 稀疏：`loss_vec = (row_gather(logits, label) - col_max - log(denom)) * mask`（:345-348），`grad = (exp(logits-col_max)/denom - select(row==label)) * mask * inv_num_valid`（:351-361），`loss = -Σ loss_vec / num_valid`（:365-371）✓（归一化除以 num_valid 而非 batch，符合铁律 8）
  - MSE：`grad = (2/N)·diff`，`loss = (1/N)·Σ diff²`（:75-104）✓
  - SGD：`p -= lr·g`（:182）✓
  - SGD-momentum：`v = β·v + (1-β)·g; p -= lr·v`（:230-237）✓
  - Adam：`m = β1·m+(1-β1)·g; v = β2·v+(1-β2)·g²; p -= lr·(m/bc1)/(sqrt(v/bc2)+eps)`，`bc_k = 1-β_k^t`（:272-319, :344-349）✓
  - AdamW：解耦 `p *= (1-lr·wd)` 在 Adam 更新之前（:429-434），wd 作用于所有参数（含 bias/norm）——与 Loshchilov & Hutter 2019 一致（AdamW 不区分 bias/norm）✓
- **epoch lr 钳制**：AGENTS.md §12"P2 已修：epoch lr 钳制"。本组 3 文件无 lr 钳制逻辑（`set_lr` 仅存储，:174/219/365/580），钳制应在训练循环（src/）侧。grep 未在本组文件发现相关代码，**本组不涉及**。

## 其他观察

1. **ZiPT 不支持 gradient checkpointing**（`set_checkpoint_every` abort，:780-786）是已知限制（docs/12 §10），但 `ZiPTBlock`/`CrossAttention` 都有 `set_checkpoint_mode` 传播（:361-374, 无对应——`CrossAttention` 未 override `set_checkpoint_mode`，用基类默认仅设 `checkpoint_mode_`，:105-109 `clear_cache` 会清缓存，但 forward :155-162 有 `if (!checkpoint_mode_)` 守卫不存 Q_re/K_re/V_re——**这意味着若外部对 `CrossAttention` 单独开启 checkpoint mode，forward 不存缓存，backward 会用空 `Q_re_cache_`/`K_re_cache_`/`V_re_cache_`（UB）**。但 `ZiPTModel::set_checkpoint_every` 直接 abort，故 `CrossAttention` 的 checkpoint mode 实际不会被启用。**建议**：`CrossAttention` 也应 override `set_checkpoint_every` 并 abort（与 `ZiPTModel` 一致），避免误用。
2. **`ZiPTBlock::forward` 的 `memory_input` 参数**（:417-418）是 `const Tensor&`，但 `ZiPTModel::forward`（:892）传的是 `C`（局部变量，:867/874 移动赋值），`C` 在 blocks 循环期间存活（:890-895 作用域覆盖整个循环），无 use-after-free。✓
3. **`generate` 的 `eos_token_id` 默认 `static_cast<std::size_t>(-1)`**（:969）= SIZE_MAX，与 vocab 无冲突（vocab < SIZE_MAX），✓；但 `min_new_tokens` 默认 0 时，`step >= 0` 恒真，首 token 即 eos 也会 break（:1046）——符合语义。
4. **`CrossAttention` 无 `w_o`**（:135 注释"无 w_o：C = A·V 直接输出"），与标准 MultiHeadAttention 不同，是 AttnZip 设计选择（记忆压缩只需 V 加权，不需再投影），✓。

---

## 汇总

| 级别 | 数量 |
|------|------|
| P0 | 0 |
| P1 | 3 |
| P2 | 4（2 条经核实降级/撤销，实际 4 条有效） |
| P3 | 10（上限） |

**最重要的 3 条**：
1. **P1** `compute_loss.hpp:303-311` forward_sparse 越界 label 静默修正为 0 而非报错，梯度污染但 loss 不报警。
2. **P1** `compute_optimizer.hpp:210/325/337/575` 优化器构造失败 `std::abort()`，违反铁律 1（错误应走 Result 传播）。
3. **P1** `compute_layer_zipt.hpp:848-850` `stored_tokens_tensor_` 无 checkpoint 门控，全序列激活常驻副本，GPU 大 batch 时显存浪费。
