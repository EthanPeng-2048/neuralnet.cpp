#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_feedforward.hpp"
#include "compute_layer_attention.hpp"
#include "compute_position_encoding.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "expr_dsl.hpp"

namespace nn
{
class GPTBlock final : public Layer
{
private:
    CausalSelfAttention self_attn_;
    std::unique_ptr<Layer> norm1_;
    FeedForward ff_;
    std::unique_ptr<Layer> norm2_;

    Tensor residual2_cache_;

    // ── activation offload（L1-offload）状态 ──
    // 实现见 compute_layer_base.hpp 的 ActivationOffloader（与 RAPTBlock 共用）
    ActivationOffloader offloader_;

public:
    GPTBlock(std::size_t d_model, std::size_t num_heads,
             std::size_t d_ff, std::size_t max_len = 1024,
             std::size_t seq_len = 0,
             PosEncodingType pos_enc = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm,
             PrecisionProfile precision = PrecisionProfile{})
        : self_attn_(d_model, num_heads, max_len, seq_len, pos_enc),
          norm1_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation),
          norm2_(make_norm_layer(d_model, norm_type))
    {
        // D7：将精度配置注入所有子层（§9.2）
        set_precision_profile(precision);
        self_attn_.set_precision_profile(precision);
        if (norm1_) norm1_->set_precision_profile(precision);
        ff_.set_precision_profile(precision);
        if (norm2_) norm2_->set_precision_profile(precision);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, self_attn_.init(engine));
        if (norm1_) { NN_TRY(r, norm1_->init(engine)); }
        NN_TRY(r2, ff_.init(engine));
        if (norm2_) { NN_TRY(r, norm2_->init(engine)); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        return collect_refs(self_attn_.parameters(),
                            norm1_->parameters(),
                            ff_.parameters(),
                            norm2_->parameters());
    }

    // 文档感知：把每样本文档 id 转发给内部自注意力（用于块对角掩码）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        self_attn_.set_doc_ids(ids);
    }

    std::vector<TensorRef> param_gradients() override
    {
        return collect_refs(self_attn_.param_gradients(),
                            norm1_->param_gradients(),
                            ff_.param_gradients(),
                            norm2_->param_gradients());
    }

    // 梯度检查点：把模式传播给内部注意力/归一化/FFN
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        self_attn_.set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
    }

    // GPTBlock 可作为“重计算单元”：从保存的块输入重算 forward 重建缓存
    [[nodiscard]] bool recompute_supported() const override { return true; }

    [[nodiscard]] Result<Tensor> forward_recompute(
        const Tensor& saved_input) override
    {
        // 临时关闭本块及其子层的 checkpoint 模式，使 forward 重建缓存
        set_checkpoint_mode(false);
        auto r = forward(saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    void clear_cache() override
    {
        self_attn_.clear_cache();
        norm1_->clear_cache();
        ff_.clear_cache();
        norm2_->clear_cache();
        residual2_cache_ = Tensor{};
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = self_attn_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto n1 = norm1_->activation_cache(); r.insert(r.end(), n1.begin(), n1.end());
        auto f = ff_.activation_cache(); r.insert(r.end(), f.begin(), f.end());
        auto n2 = norm2_->activation_cache(); r.insert(r.end(), n2.begin(), n2.end());
        if (residual2_cache_.valid()) r.emplace_back(residual2_cache_);
        return r;
    }

    // ── activation offload（L1-offload）────────────────────────────────
    void set_offload_enabled(bool enabled) { offloader_.set_enabled(enabled); }
    [[nodiscard]] bool offload_enabled() const noexcept { return offloader_.enabled(); }

    // 导出/导入：委托给通用 ActivationOffloader（见 compute_layer_base.hpp）。
    // 参与 offload 的缓存集合 = 本块 activation_cache()。
    [[nodiscard]] Result<void> export_activations(ComputeEngine& engine)
    {
        return offloader_.export_activations(engine, activation_cache());
    }

    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        return offloader_.import_activations(engine);
    }

    // 实际 slab 字节数（诊断用；未创建时 0）
    [[nodiscard]] std::size_t offload_slab_bytes() const noexcept
    {
        return offloader_.slab_bytes();
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        // 残差分支 1（Pre-Norm + 自注意力），与 TransformerEncoderLayer/RAPTBlock 同骨架
        NN_TRY(r2, prenorm_residual_forward_(engine, input, *norm1_, self_attn_, p_.compute));
        Tensor res2 = std::move(*r2);
        if (!checkpoint_mode_)
            residual2_cache_ = res2;
        // 残差分支 2（Pre-Norm + FFN）
        return prenorm_residual_forward_(engine, res2, *norm2_, ff_, p_.compute);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // activation offload：从 host 恢复激活再反向（替代重计算）
        if (offloader_.offloaded())
        {
            NN_TRY(im, import_activations(engine));
        }

        // 两处残差分流 + 子层正反向（与 TransformerEncoderLayer / RAPTBlock 同一骨架）
        return prenorm_residual_backward_(engine, grad_output,
                                          *norm1_, self_attn_, *norm2_, ff_,
                                          p_.compute);
    }

    // ── 增量推理（KV cache）──────────────────────────────────────────
    // Pre-Norm 单 token 前向：
    //   x = x_new + CausalSelfAttn(LN₁(x_new), kv_cache)
    //   x = x + FFN(LN₂(x))
    // 输入: x_new (d_model, 1)
    // 输出: (d_model, 1)
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        const Tensor& x_new,
        Tensor& k_cache,
        Tensor& v_cache,
        std::size_t cur_len)
    {
        NN_TRY(n1, norm1_->forward(x_new));

        NN_TRY(a, self_attn_.forward_step(engine, *n1, k_cache, v_cache, cur_len));

        auto r2 = dsl::compute(engine,
            dsl::leaf(x_new) + dsl::leaf(*a),
            x_new.rows(), x_new.cols(), p_.compute);
        NN_TRY_CHECK(r2);

        NN_TRY(n2, norm2_->forward(*r2));

        NN_TRY(f, ff_.forward(*n2));

        return dsl::compute(engine,
            dsl::leaf(*r2) + dsl::leaf(*f),
            r2->rows(), r2->cols(), p_.compute);
    }
};

// GPTModel — Decoder-only Transformer 语言模型
//
// 算法（只在此处，不在 Engine/Shader）：
//   组件: TokenEmb [+ PosEmb] + N × GPTBlock + LayerNorm + LM Head
//   输入: (seq_len, batch_size) — token ID 矩阵（每列为一个序列）
//   输出: (vocab_size, seq_len × batch_size) — 每个位置的 logits
//
// 通过 PosEncodingType 参数支持三种位置编码模式：
//   - Learned:    可学习位置嵌入（默认）
//   - Sinusoidal: 正弦波固定位置编码（冻结）
//   - ALiBi:      无位置嵌入，通过 CausalSelfAttention 的线性偏置注入位置信息
//
// 注意: token embedding 查表（gather_rows）与位置 embedding 相加全程由
//       引擎原语在设备端完成；反向用 scatter_add_rows 稀疏累加回嵌入表，
//       无 CPU 中间拷贝。
// ══════════════════════════════════════════════════════════════════════════
class GPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;

    // 嵌入
    Tensor token_emb_;       // (vocab_size, d_model)
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;  // 位置编码（多态：Learned/Sinusoidal/无）

    std::vector<GPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    // 反向缓存
    Tensor stored_tokens_tensor_;          // token IDs 的 Tensor 版本 (total, 1)
    std::size_t batch_size_ = 0;

    // 文档感知：当前 step 每样本文档 id（batch-major b*seq+t → doc id），
    // 由调用方在 forward 前 set_doc_ids 设置，转发给各 GPTBlock。
    std::vector<std::size_t> doc_ids_;

    // batch 录制粒度：每隔 flush_interval_ 个 Transformer block 提交一次
    // 0 = 不在 block 间 flush（默认），>0 = 每 N 个 block flush 一次
    std::size_t flush_interval_ = 0;

    // 梯度检查点（激活重计算 L1）：每隔 checkpoint_every_ 个 GPTBlock 保存一次
    // 块输入，backward 时重算以省去驻留整层激活。0 = 不启用。
    std::size_t checkpoint_every_ = 0;
    std::vector<Tensor> checkpoint_inputs_;  // 各 checkpoint 块的输入 (d_model, batch*seq)

    // activation offload（L1-offload）：把每块内部激活搬 host-visible，backward 拷回
    bool activation_offload_ = false;

public:
    GPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
             std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
             PosEncodingType pos_enc_type = PosEncodingType::Learned,
             ActivationType activation = ActivationType::GeLU,
             NormType norm_type = NormType::LayerNorm,
             PrecisionProfile precision = PrecisionProfile{})
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        // D7：将精度配置注入自身和所有子层（§9.2）
        set_precision_profile(precision);
        if (ln_f_) ln_f_->set_precision_profile(precision);
        // ── LM head（词表投影）：计算精度强制 = stable ────────────────────
        // logits 是 (vocab, batch·seq) —— 全模型最大的张量，且被 loss 链**多次**
        // 读取（col_max / denom / loss_vec / grad …）。若 head 留在 compute 精度
        // （f16），边界 cast 适配层会为 loss 的每个算子各物化一份 f32 副本：
        // 实测 vocab=8208 / batch=64 / seq=256 下 = 6×512MB → vkAllocateMemory
        // OOM（探针 transient 桶 6 项/3.1GB）。head 用 stable 后 logits 与 f32
        // 基线同构（零 cast），f16 的收益集中在隐藏层激活（体积小、生命周期短）。
        // 真正的 f16 logits 需要 in-kernel f16（typed IR，docs 05 §12.3 Phase 2）。
        {
            PrecisionProfile head_prof = precision;
            head_prof.compute = precision.stable;
            lm_head_.set_precision_profile(head_prof);
        }

        // ── 位置编码器（**嵌入侧**）：模型只负责"加到 token 嵌入上"那一半 ────
        // Learned / Sinusoidal 在此实做；RoPE / ALiBi 映射为恒等策略——它们的
        // 注入点在注意力层（Q/K 旋转 / 分数偏置），由 CausalSelfAttention
        // **自持**（经 make_attention_position_encoder），模型不再下发。
        pos_encoder_ = make_embedding_position_encoder(pos_enc_type, d_model, seq_len);

        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
        {
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, seq_len,
                                 pos_enc_type, activation, norm_type, precision);
        }

        // 位置编码器是辅助对象（非 Layer）→ profile 需单独下传：
        // 否则其内部 DSL 求值（gather→transpose→加性融合）退回 F32。
        pos_encoder_->set_precision_profile(precision);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // 初始化 token_emb_——M2 声明式：N(0, 0.02) 层算参数、引擎填数
        constexpr Scalar emb_init_std = 0.02;
        token_emb_ = engine.create_tensor(vocab_size_, d_model_, p_.param,
                                          InitSpec::normal(0, emb_init_std, kInitSeed));
        if (!token_emb_.valid())
            NN_FAIL("GPTModel: token_emb 初始化失败");
        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_, p_.param, InitSpec::zero());
        if (!grad_token_emb_.valid())
            NN_FAIL("GPTModel: token_emb 梯度缓冲初始化失败");
        // 初始化子层
        if (pos_encoder_)
        {
            NN_TRY(r, pos_encoder_->init(engine));
        }
        for (auto& block : blocks_)
        {
            NN_TRY(r, block.init(engine));
        }
        if (ln_f_)
        {
            NN_TRY(r, ln_f_->init(engine));
        }
        { NN_TRY(r, lm_head_.init(engine)); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        return collect_refs(token_emb_,
                            pos_encoder_->parameters(),
                            collect_block_refs_(blocks_, &GPTBlock::parameters),
                            ln_f_->parameters(),
                            lm_head_.parameters());
    }

    std::vector<TensorRef> param_gradients() override
    {
        return collect_refs(grad_token_emb_,
                            pos_encoder_->param_gradients(),
                            collect_block_refs_(blocks_, &GPTBlock::param_gradients),
                            ln_f_->param_gradients(),
                            lm_head_.param_gradients());
    }

    // 文档感知：设置当前 step 每样本文档 id（batch-major b*seq+t → doc id）。
    // 传入空 span 清除文档感知（退化为纯因果）。在 forward 前调用。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        const std::size_t seq_len = input.rows();
        batch_size_ = input.cols();
        // ── 1. gather 所有 token 的 embedding（统一采用 batch-major 列序） ──
        // 下方注意力（AttentionBase::forward 的 rearrange_3d + batched_matmul + 因果掩码）
        // 假定扁平列为 batch-major（b*seq + t）。因此先把输入 (seq, batch) 转置为
        // (batch, seq)，使 gather_rows 的 flat 序即为 batch-major：i = b*seq + t。
        auto input_T = engine.transpose(input);   // (batch, seq)，flat 索引 = b*seq+t
        NN_TRY_CHECK(input_T);

        // gather_rows(token_emb_, input_T) → (batch*seq, d_model)
        //   row i = token_emb[input_T[i]], i 是 batch-major 索引 b*seq+t
        NN_TRY(all_emb, engine.gather_rows(token_emb_, *input_T));

        // ── 2. 保存 token IDs 的 Tensor 拷贝（供 backward 的 scatter_add_rows） ──
        // 用 batch-major 序的 input_T，使 scatter 行号与 grad_T（transpose(grad_x)）对齐。
        // 全程 GPU：clone 在 GPU 内执行，无 PCIe 传输
        NN_TRY(st_t, engine.clone(*input_T));
        stored_tokens_tensor_ = std::move(*st_t);

        // ── 3. 构造 x: (d_model, batch*seq)（batch-major 列序） ──
        NN_TRY(all_T, engine.transpose(*all_emb));

        // ── 3. 施加位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE 为 no-op） ──
        NN_TRY(x_result, pos_encoder_->apply(engine, *all_T, batch_size_, seq_len));
        // ── 4. 通过 Transformer 块（全批量化，无 per-sample 循环） ──
        Tensor x = std::move(*x_result);
        checkpoint_inputs_.clear();
        const bool ckpt = (checkpoint_every_ > 0);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            // 文档感知：把本 step 每样本文档 id 传给各 block 的注意力
            blocks_[bi].set_doc_ids(doc_ids_);
            // 梯度检查点：每 checkpoint_every_ 个块保存一次输入；
            // 该块及其子层以 checkpoint 模式运行（不驻留中间激活）
            if (ckpt && (bi % checkpoint_every_ == 0))
            {
                NN_TRY(save, engine.clone(x));
                checkpoint_inputs_.push_back(std::move(*save));
                blocks_[bi].set_checkpoint_mode(true);
            }
            else
            {
                blocks_[bi].set_checkpoint_mode(false);
            }
            NN_TRY(r, blocks_[bi].forward(x));
            x = std::move(*r);
            // activation offload：forward 后把本块内部激活搬 host-visible，释放 GPU 显存。
            // 混合模式（offload + checkpoint 共存）：checkpoint 块 forward 不驻留激活
            // （checkpoint_mode=true），无可导出的缓存，必须跳过 export（否则会创建
            // 空 slab 而失败）；非 checkpoint 块才导出。
            if (activation_offload_ && !blocks_[bi].checkpoint_mode())
            {
                NN_TRY(ex, blocks_[bi].export_activations(engine));
            }
            // 按间隔 flush，将大录制拆分为多个小提交（防 TDR）
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < blocks_.size())
            {
                NN_TRY(fr, engine.flush_batch());
            }
        }

        // ── 5. 最终 LayerNorm/RMSNorm ──
        NN_TRY(ln, ln_f_->forward(x));
        x = std::move(*ln);

        // ── 6. LM Head → (vocab_size, seq*batch) batch-major ──
        auto lm_out = lm_head_.forward(x);
        return lm_out;
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const std::size_t seq_len = seq_len_;

        // ⚠ 注意：不用在这里 zero grad_token_emb_！
        //   grad_token_emb_ 已注册到优化器的 param_gradients() 中，
        //   由优化器的 zero_grad() 统一清零。这里如果额外清零会破坏
        //   梯度积累（accum_steps > 1 时前几轮的梯度信号全部丢失）。
        //   位置编码梯度同理（pos_encoder_->param_gradients() 已注册，勿在此清零）。
        // (void)engine.zero(grad_token_emb_);

        // ── 1. LM Head 反向 → (d_model, seq*batch) ──
        NN_TRY(b_lm, lm_head_.backward(grad_output));
        Tensor grad_x = std::move(*b_lm);

        // ── 2. LayerNorm/RMSNorm 反向 ──
        NN_TRY(b_ln, ln_f_->backward(grad_x));
        grad_x = std::move(*b_ln);

        // ── 3. 逐块反向（全批量化） ──
        {
            const std::size_t n = blocks_.size();
            for (std::size_t bi = 0; bi < n; ++bi)
            {
                const std::size_t idx = n - 1 - bi;
                // 梯度检查点：若是 checkpoint 块，先重算 forward 重建缓存再反向
                if (checkpoint_every_ > 0 && (idx % checkpoint_every_ == 0))
                {
                    const std::size_t seg = idx / checkpoint_every_;
                    NN_ASSERT(seg < checkpoint_inputs_.size(),
                              "GPTModel backward: checkpoint input missing");
                    NN_TRY(cr, blocks_[idx].forward_recompute(checkpoint_inputs_[seg]));
                }
                NN_TRY(br, blocks_[idx].backward(grad_x));
                grad_x = std::move(*br);
                // 显存：backward 后**立即释放该块已消费的激活缓存**——否则每块
                // 激活会驻留到整个 backward 结束（探针实测为 backward 段峰值
                // 主项；torch 的等价行为是"用完即释放"）。释放后该块的内存可被
                // 后续块的 backward 临时量复用；下一轮 forward 会重新填充缓存。
                blocks_[idx].clear_cache();
                if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
                {
                    NN_TRY(fr, engine.flush_batch());
                }
            }
        }

        // ── 3.5 重计算完成：释放保存的块输入（供 L2 整块归还） ──
        checkpoint_inputs_.clear();

        // ── 4. 转置 grad_x + pos_grad GPU 计算 ──
        //   grad_x: (d_model, batch*seq)（batch-major 列序）
        //   grad_T = transpose(grad_x) → (total, d_model) — 用于 scatter_add_rows
        NN_TRY(grad_T, engine.transpose(grad_x));

        // 位置编码反向（Learned 累计 grad_pos_emb_；Sinusoidal/ALiBi/RoPE no-op）
        NN_TRY(pr, pos_encoder_->backward(engine, *grad_T, batch_size_, seq_len_));

        // ── 5. scatter_add_rows: grad_token_emb_[tokens] += grad_T ──
        NN_TRY(sr, engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T));

        // grad_input: token IDs 无梯度，返回零张量（仅用于接口一致性）
        Tensor grad_input = engine.create_tensor(seq_len, batch_size_, Precision::F32,
                                                 InitSpec::zero());
        if (!grad_input.valid())
            NN_FAIL("GPT token_emb backward: 梯度张量分配失败");
        return grad_input;
    }

    // ── batch 录制粒度控制 ──
    void set_flush_interval(std::size_t interval) override { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 梯度检查点（激活重计算 L1）粒度控制 ──
    // stride：每 N 个 GPTBlock 保存一次块输入，backward 时重算该块 forward。
    // 0 = 不启用（默认）。stride=1 时显存收益最大（仅保留块输入 + 单块激活）。
    void set_checkpoint_every(std::size_t stride) override { checkpoint_every_ = stride; }
    [[nodiscard]] std::size_t checkpoint_every() const noexcept { return checkpoint_every_; }

    // ── activation offload（L1-offload）开关 ──
    // 启用后：forward 把每块内部激活搬 host-visible（释放 device-local VRAM），
    // backward 拷回再反向（不重算，FLOPs 保持 1.0×，代价是 PCIe 传输）。
    void set_activation_offload(bool enabled) override
    {
        activation_offload_ = enabled;
        for (auto& b : blocks_) b.set_offload_enabled(enabled);
    }
    [[nodiscard]] bool activation_offload() const noexcept { return activation_offload_; }

    // 实际 offload RAM 字节数：各块已创建 slab 大小之和（诊断用）
    [[nodiscard]] std::size_t offload_ram_bytes() override
    {
        std::size_t total = 0;
        for (auto& b : blocks_)
            total += b.offload_slab_bytes();
        return total;
    }

    // 梯度检查点：把模式传播给所有块与末级归一化/LM Head
    // （当本 GPTModel 整体作为 Model 的一层被置于 checkpoint 模式时生效）
    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        checkpoint_inputs_.clear();
    }

    // ── 增量推理：单 token 前向（KV cache）──────────────────────────
    // 流程: token_emb[token] [+ pos_emb[pos]] → N × GPTBlock.forward_step
    //       → ln_f → lm_head → (vocab_size, 1)
    // 每层的 KV cache 由调用方维护，forward_step 只负责写入和计算。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine,
        std::size_t token_id,
        std::size_t pos,
        std::vector<Tensor>& k_caches,
        std::vector<Tensor>& v_caches,
        std::size_t cur_len)
    {
        // 1. token embedding 查表 → (1, d_model) → transpose → (d_model, 1)
        std::vector<Scalar> id_v(1);                  // 宿主桥（17 §3 D11）
        id_v[0] = static_cast<Scalar>(token_id);
        NN_TRY(id_t, detail::upload_span(engine, 1, 1, Precision::F32, std::span(id_v)));
        NN_TRY(emb, engine.gather_rows(token_emb_, *id_t));
        NN_TRY(x_new, engine.transpose(*emb));

        // 2. 位置编码（Learned/Sinusoidal 相加；ALiBi/RoPE no-op）
        NN_TRY(x_wp, pos_encoder_->apply_step(engine, *x_new, pos));
        x_new = std::move(*x_wp);

        // 3. 逐块增量前向
        Tensor x = std::move(*x_new);
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            auto r = blocks_[i].forward_step(
                engine, x, k_caches[i], v_caches[i], cur_len);
            NN_TRY_CHECK(r);
            x = std::move(*r);
            // 按 flush_interval 拆分提交（防 TDR）
            if (flush_interval_ > 0 &&
                (i + 1) % flush_interval_ == 0 && i + 1 < blocks_.size())
            {
                NN_TRY(fr, engine.flush_batch());
            }
        }

        // 4. 最终 LayerNorm + LM Head → (vocab_size, 1)
        NN_TRY(ln, ln_f_->forward(x));
        return lm_head_.forward(*ln);
    }

    // ── 采样生成（KV cache 增量推理）────────────────────────────────
    // 性能策略（P0 + P1 + KV cache）：
    //   P0: begin_batch/end_batch 包裹 forward_step，单次 GPU 提交。
    //   P1: 每步只上传 1 个 token ID，无需重传整个 seq_len。
    //   KV cache: 每步 attention 只计算 Q×K_history（O(seq_len) 而非 O(seq_len²)），
    //             历史 K/V 复用缓存，不重复投影。
    //   logits 直接是 (vocab_size, 1)，无需 transpose+slice。
    //
    // 滑动窗口: 当 cur_len 达到 seq_len_ 时，丢弃最旧 token 重建 cache
    //           （保留最后 seq_len_-1 个 token 作为新上下文）。
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
             const std::vector<std::size_t>& prompt,
             std::size_t max_new_tokens,
             Scalar temperature = 1.0,
             std::size_t eos_token_id = static_cast<std::size_t>(-1),
             std::size_t min_new_tokens = 0)
    {
        std::vector<std::size_t> context(prompt);
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

        // ── 预分配 KV cache: 每层一对 (k, v)，形状 (seq_len_, d_model) ──
        // H*d_k = d_model（因为 d_k = d_model / num_heads）
        std::vector<Tensor> k_caches, v_caches;
        k_caches.reserve(blocks_.size());
        v_caches.reserve(blocks_.size());
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            k_caches.push_back(engine.create_tensor(seq_len_, d_model_, p_.compute));
            v_caches.push_back(engine.create_tensor(seq_len_, d_model_, p_.compute));
        }

        // ── prefill: 截断到 seq_len_ 长度（滑动窗口初始） ──────────────
        std::size_t start_init = 0;
        if (context.size() > seq_len_)
            start_init = context.size() - seq_len_;
        std::size_t cur_len = 0;
        Tensor last_logits_t;

        // ── 逐 token 填充 KV cache（prefill 与滑动窗口重建共用） ──────
        // 从 context[start..end) 逐个 forward_step，更新 cur_len 与 last_logits_t。
        // 整段包进单次 begin_batch/end_batch：GPU 下避免 O(seq) 次独立提交
        // （铁律 6）。forward_step 为纯 device-resident（不触发 host 同步），
        // 且内部按 flush_interval_ 调用 flush_batch 防 TDR。
        auto fill_cache_ = [&](std::size_t start) -> Result<void>
        {
            NN_TRY(br, engine.begin_batch());
            for (std::size_t i = start; i < context.size(); ++i)
            {
                auto r = forward_step(engine, context[i], cur_len,
                                      k_caches, v_caches, cur_len);
                if (!r)
                {
                    (void)engine.end_batch();  // 出错也收尾，避免录制状态泄漏
                    NN_TRY_CHECK(r);
                }
                last_logits_t = *r;
                ++cur_len;
            }
            NN_TRY(er, engine.end_batch());
            return {};
        };

        {
            NN_TRY(r, fill_cache_(start_init));
        }

        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            // Sliding window: rebuild cache when full (keep last seq_len_-1 tokens)
            if (cur_len >= seq_len_)
            {
                for (auto& kc : k_caches) { NN_TRY(r, engine.zero(kc)); }
                for (auto& vc : v_caches) { NN_TRY(r, engine.zero(vc)); }
                cur_len = 0;
                const std::size_t keep = seq_len_ - 1;
                const std::size_t start_new = (context.size() > keep) ? (context.size() - keep) : 0;
                NN_TRY(r, fill_cache_(start_new));
            }

            // Sample from last_logits_t (from prefill or previous step)
            auto logits_v = detail::download_vector(engine, last_logits_t);  // 宿主桥
            NN_TRY_CHECK(logits_v);

            std::vector<Scalar> last_logits(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last_logits[v] = (*logits_v)[v];

            // temperature → softmax → 采样/贪心（三模型共用的采样器）
            const std::size_t next_token =
                sample_next_token_(last_logits, temperature, rng, dist);

            context.push_back(next_token);

            if (step >= min_new_tokens && next_token == eos_token_id)
                break;

            generated.push_back(next_token);

            // Run forward_step for next_token to write KV cache and get new logits
            NN_TRY(br, engine.begin_batch());

            auto logits_res = forward_step(engine, next_token, cur_len,
                                           k_caches, v_caches, cur_len);
            NN_TRY_CHECK(logits_res);

            NN_TRY(er, engine.end_batch());

            last_logits_t = std::move(*logits_res);
            ++cur_len;
        }
        return generated;
    }
};
} // namespace nn

