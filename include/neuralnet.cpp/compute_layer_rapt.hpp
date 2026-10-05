#pragma once

#include "compute_layer_base.hpp"
#include "compute_layer_mlp.hpp"
#include "compute_layer_feedforward.hpp"
#include "compute_layer_attention.hpp"
#include "compute_layer_gpt.hpp"

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
// ══════════════════════════════════════════════════════════════════════════
// ReLULinearAttention — RLA-2：极简硬截断线性注意力（docs/development/06-rapt-algorithm.md）
//
// 算法（RLA-2，修正版；causal / bidirectional 均支持）：
//   q' = ReLU(RoPE(q)), k' = ReLU(RoPE(k)), v = W_v·x（V 不做 ReLU）
//   分子 num_t  = B_t · q'_t,   B_t = Σ_{i∈S_t} v_i k'_i^T
//   分母 den_t  = q'_t · z_t + ε,  z_t = Σ_{i∈S_t} k'_i
//   out_t = num_t / den_t
//   其中 S_t = { i<=t }（causal）或 { 全部 }（bidirectional）。
//
// 与原版 RLA 的关键差异（06-rapt-algorithm.md §3 家族演进）：
//   * 分母：Sum 归一化（加权平均）替代 L2 归一化（余弦约束）
//     → 输出量级恒定（不随 √L 增长），梯度流更平稳。
//   * 无 A 状态（Σ k'k'^T）：分母仅依赖 z = Σk'，无需二次型。
//   * 推理：O(d²) 每步（B 状态 + z 状态，替代原版 A + B）。
//
// 关键性质：
//   * 训练复杂度 O(L·d_k²)，推理每步 O(d_k²)，无 O(L²) 物化。
//   * 施加顺序：RoPE → ReLU（先旋转后截断，否则丢位置信息）。
//   * 文档感知：文档边界处重置运行态（前缀/后缀在同文档内）。
//
// 增量推理（forward_step）：
//   维护 B_state (d_model, d_k) 和 z_state (d_model, 1)：
//     B += k' ⊗ v,  z += k'
//     out = B·q' / (q'·z + ε)
//   每步 O(d_k²)，与序列长度无关。
// ══════════════════════════════════════════════════════════════════════════
class ReLULinearAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;    // 单样本序列长度（0 = 单样本，cols 即 seq）
    bool causal_;            // true=因果前缀和；false=全量双向
    // 位置编码策略（注意力层自持；非 RoPE 类型 = 恒等 → 热路径无标志位判断）
    std::unique_ptr<PositionEncoder> pos_;
    Linear w_q_, w_k_, w_v_, w_o_;

    // forward 缓存（backward 用）
    Tensor Qp_cache_;        // (BH*d_k, seq) ReLU(RoPE(Q))
    Tensor Kp_cache_;        // (BH*d_k, seq) ReLU(RoPE(K))
    Tensor V_re_cache_;      // (BH*d_k, seq) V（不 ReLU、不 RoPE）
    std::size_t batch_cache_ = 0;
    std::size_t seq_cache_   = 0;

    // z-scan 缓存：V_ones / e_0 用于计算 z = Σk'（RLA-2 分母）
    Tensor V_ones_cache_;    // (BH*dk, seq) 全1矩阵
    Tensor e_0_cache_;       // (BH*dk, seq) 每头首行=1、其余=0（单位向量）
    std::size_t ones_BH_  = 0;
    std::size_t ones_seq_ = 0;

    // RMSNorm 缓存（RLA-2：投影后 RMSNorm → RoPE → ReLU）
    Tensor Q_normed_cache_;   // (BH*dk, seq) — RMSNorm(Q) before RoPE
    Tensor K_normed_cache_;   // (BH*dk, seq) — RMSNorm(K) before RoPE
    Tensor Q_rms_inv_cache_;  // (BH, seq) — per-head 1/rms for Q
    Tensor K_rms_inv_cache_;  // (BH, seq) — per-head 1/rms for K

    // 文档感知：每位置文档 id（batch-major，size = batch*seq）；空=无文档感知
    std::vector<std::size_t> doc_ids_;
    bool has_doc_ids_ = false;

    // 由 doc_ids_ 构建文档边界向量（size = batch*seq）：boundary[b*seq+t]=1
    // 表示位置 t 是文档起点（t==0 或与前一位置文档不同）。
    [[nodiscard]] std::vector<uint8_t> build_boundary_(
        std::size_t batch, std::size_t seq) const
    {
        std::vector<uint8_t> boundary;
        if (!has_doc_ids_) return boundary;
        boundary.assign(batch * seq, 0);
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t t = 0; t < seq; ++t)
                if (t == 0 || doc_ids_[b * seq + t] != doc_ids_[b * seq + t - 1])
                    boundary[b * seq + t] = 1;
        return boundary;
    }

    // (1,1) dummy 张量（清零）：空参数占位，规避 0 字节 GPU buffer（铁律 9）
    [[nodiscard]] Result<Tensor> make_dummy_(ComputeEngine& engine)
    {
        Tensor d = engine.create_tensor(1, 1);
        // create_tensor 分配失败时返回空 Tensor（非 Result）；此处显式检查，
        // 否则 zero→import 只会报笼统的 "invalid tensor"，掩盖真实原因（显存/设备）。
        if (!d.valid())
            NN_FAIL("make_dummy_: GPU 张量分配失败（显存不足或设备异常）");
        NN_TRY(r, engine.zero(d));
        return d;
    }

    // 确保 V_ones / e_0 缓存与当前 BH·dk × seq 尺寸匹配。
    // 返回 Result：分配/upload 失败（常见为显存不足）时传播真实错误，
    // 避免后续 scan/outer_col 对空张量报 "import: invalid tensor" 掩盖根因。
    [[nodiscard]] Result<void> ensure_ones_(
        ComputeEngine& engine, std::size_t BH, std::size_t dk, std::size_t seq)
    {
        // 两个缓存都有效且尺寸匹配才命中（避免半初始化：V_ones 有效但 e_0 缺失）。
        if (ones_BH_ == BH && ones_seq_ == seq &&
            V_ones_cache_.valid() && e_0_cache_.valid())
            return {};
        const std::size_t rows = BH * dk;
        // V_ones：全1（声明式常数初始化，M2 InitSpec）
        V_ones_cache_ = engine.create_tensor(rows, seq, Precision::F32,
                                             InitSpec::constant(Scalar{1}));
        if (!V_ones_cache_.valid())
            NN_FAIL("ensure_ones_: V_ones 分配失败");
        // e_0：每头首行=1，其余=0（用于 suffix(scale·q) 的外积构造）
        std::vector<Scalar> e0(rows * seq, Scalar{0});   // 宿主桥（17 §3 D11）
        for (std::size_t bh = 0; bh < BH; ++bh)
            for (std::size_t t = 0; t < seq; ++t)
                e0[(bh * dk) * seq + t] = Scalar{1};
        auto e0t = detail::upload_span(engine, rows, seq, Precision::F32,
                                       std::span(e0));
        NN_TRY_CHECK(e0t);
        e_0_cache_ = std::move(*e0t);
        ones_BH_  = BH;
        ones_seq_ = seq;
        return {};
    }

    // ── 逐头 RMSNorm 前向（RLA-2：投影后 RMSNorm → RoPE → ReLU） ────
    // 对 (BH*dk, seq) 张量的每个 dk 块独立归一化：y = x / sqrt(mean(x²) + eps)
    // 无学习参数（固定增益=1），符合极简原则。
    [[nodiscard]] Result<Tensor> rms_norm_forward_(
        ComputeEngine& engine, const Tensor& input,
        std::size_t BH, std::size_t dk, std::size_t seq, Tensor* rms_inv_out)
    {
        const Scalar inv_dk = Scalar{1} / static_cast<Scalar>(dk);
        const Scalar eps = Scalar{1e-5};
        Tensor output = engine.create_tensor(BH * dk, seq);
        if (!output.valid())
            NN_FAIL("rms_norm_forward_: GPU 张量分配失败（显存不足或设备异常）");
        // rms_inv_out == nullptr（checkpoint 模式）：不收集逐头 1/rms 缓存，
        // backward 由 forward_recompute 重建；同时省掉 BH 次 GPU→CPU 下载。
        // 宿主桥（17 §3 D11）：标量缓冲收逐头 1/rms，不经 Matrix。
        std::vector<Scalar> rms_v;
        if (rms_inv_out) rms_v.assign(BH * seq, Scalar{0});
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            NN_TRY(x, engine.slice_rows(input, bh * dk, dk));
            // "乘 x² → 列归约 → 乘 1/dk → 加 eps → rsqrt"按两步 DSL 执行：
            //   ① 归约出 (1,seq) 向量；② 在 (1,seq) 小向量上做后处理。
            // 必须分两步：归约融合 shader 尚不支持"归约后仍有逐元素后处理"
            // 的形态（GPU 侧会静默错值，见 GpuEngine::eval_expr_reduce 的显式
            // 拒绝）；1/dk、eps 由 RParam 承载 → 值不进 expr_spec_key。
            auto s = dsl::compute_reduce(engine,
                dsl::col_reduce_sum(dsl::leaf(*x) * dsl::leaf(*x)), dk, seq);
            NN_TRY_CHECK(s);
            auto ri = dsl::compute(engine,
                dsl::rsqrt(dsl::leaf(*s) * dsl::rparam(inv_dk) + dsl::rparam(eps)),
                s->rows(), s->cols());
            NN_TRY_CHECK(ri);
            if (rms_inv_out)
            {
                NN_TRY(ri_v, detail::download_vector(engine, *ri));
                for (std::size_t t = 0; t < seq; ++t)
                    rms_v[bh * seq + t] = (*ri_v)[t];
            }
            auto n = dsl::compute(engine,
                dsl::leaf(*x) * dsl::col_broadcast(*ri), dk, seq);
            NN_TRY_CHECK(n);
            NN_TRY(ins, engine.insert_rows(output, bh * dk, *n));
        }
        if (rms_inv_out)
        {
            auto ri_t = detail::upload_span(engine, BH, seq, Precision::F32,
                                            std::span(rms_v));
            NN_TRY_CHECK(ri_t);
            *rms_inv_out = std::move(*ri_t);
        }
        return output;
    }

    // ── 逐头 RMSNorm 反向 ───────────────────────────────────────────
    // dL/dx = (1/rms) · (g - y · (g·y)/dk)
    // 其中 y = normed（缓存），rms = 1/rms_inv（缓存）。
    [[nodiscard]] Result<Tensor> rms_norm_backward_(
        ComputeEngine& engine, const Tensor& grad,
        const Tensor& normed, const Tensor& rms_inv,
        std::size_t BH, std::size_t dk, std::size_t seq)
    {
        const Scalar inv_dk = Scalar{1} / static_cast<Scalar>(dk);
        Tensor output = engine.create_tensor(BH * dk, seq);
        if (!output.valid())
            NN_FAIL("rms_norm_backward_: GPU 张量分配失败（显存不足或设备异常）");
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            NN_TRY(gy, engine.slice_rows(grad, bh * dk, dk));
            NN_TRY(y, engine.slice_rows(normed, bh * dk, dk));
            NN_TRY(ri, engine.slice_rows(rms_inv, bh, 1));
            // m = (1/dk) · col_reduce_sum(gy * y)  → (1, seq)
            // 归约与后处理分两步（原因同上：归约融合 shader 不支持归约后后处理）
            auto m_raw = dsl::compute_reduce(engine,
                dsl::col_reduce_sum(dsl::leaf(*gy) * dsl::leaf(*y)), dk, seq);
            NN_TRY_CHECK(m_raw);
            auto m = dsl::compute(engine,
                dsl::leaf(*m_raw) * dsl::rparam(inv_dk),
                m_raw->rows(), m_raw->cols());
            NN_TRY_CHECK(m);
            // grad_x = (gy − m·y) · rms_inv：三式合一，单次遍历、无中间张量
            auto gx = dsl::compute(engine,
                (dsl::leaf(*gy) - dsl::col_broadcast(*m) * dsl::leaf(*y))
                    * dsl::col_broadcast(*ri),
                dk, seq);
            NN_TRY_CHECK(gx);
            NN_TRY(ins, engine.insert_rows(output, bh * dk, *gx));
        }
        return output;
    }

public:
    ReLULinearAttention(std::size_t d_model, std::size_t num_heads,
                        std::size_t seq_len = 0,
                        bool causal = true,
                        PosEncodingType pos_enc = PosEncodingType::RoPE)
        : d_model_(d_model), num_heads_(num_heads),
          d_k_(d_model / num_heads),
          seq_len_(seq_len), causal_(causal),
          w_q_(d_model, d_model), w_k_(d_model, d_model),
          w_v_(d_model, d_model), w_o_(d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ReLULinearAttention: d_model must be divisible by num_heads");
        NN_ASSERT(d_model % num_heads == 0 && (d_model / num_heads) % 2 == 0,
                  "ReLULinearAttention: RoPE requires even d_k");
        // 注意力侧位置编码由本层自持（Learned/Sinusoidal 在 Q/K 侧是恒等）
        pos_ = make_attention_position_encoder(pos_enc, d_k_, num_heads_);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, w_q_.init(engine));
        NN_TRY(r2, w_k_.init(engine));
        NN_TRY(r3, w_v_.init(engine));
        NN_TRY(r4, w_o_.init(engine));
        // pos_ 不在此 init（注入的是模型级对象，由模型自己 init；RoPE 无需 init）
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        auto p = w_q_.parameters();
        auto k = w_k_.parameters();
        auto v = w_v_.parameters();
        auto o = w_o_.parameters();
        p.insert(p.end(), k.begin(), k.end());
        p.insert(p.end(), v.begin(), v.end());
        p.insert(p.end(), o.begin(), o.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        auto g = w_q_.param_gradients();
        auto k = w_k_.param_gradients();
        auto v = w_v_.param_gradients();
        auto o = w_o_.param_gradients();
        g.insert(g.end(), k.begin(), k.end());
        g.insert(g.end(), v.begin(), v.end());
        g.insert(g.end(), o.begin(), o.end());
        return g;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        w_q_.set_checkpoint_mode(enabled);
        w_k_.set_checkpoint_mode(enabled);
        w_v_.set_checkpoint_mode(enabled);
        w_o_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        Qp_cache_ = Tensor{};
        Kp_cache_ = Tensor{};
        V_re_cache_ = Tensor{};
        Q_normed_cache_ = Tensor{};
        K_normed_cache_ = Tensor{};
        Q_rms_inv_cache_ = Tensor{};
        K_rms_inv_cache_ = Tensor{};
        batch_cache_ = 0;
        seq_cache_   = 0;
        w_q_.clear_cache(); w_k_.clear_cache();
        w_v_.clear_cache(); w_o_.clear_cache();
    }

    // 文档感知：记录每位置文档 id（batch-major），forward/backward 据此
    // 在文档边界重置运行态（每个 token 只聚合本文档内前缀）。
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); has_doc_ids_ = false; return; }
        doc_ids_.assign(ids.begin(), ids.end());
        has_doc_ids_ = true;
    }

    // 绝对位置偏移（滑动窗生成用）：转发给位置编码策略（RoPE 生效，其余 no-op），
    // 使重计算式生成的位置从真实起点算起而非每次从 0 重置。
    void set_position_offset(std::size_t off)
    {
        pos_->set_position_offset(off);
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Qp_cache_.valid()) r.emplace_back(Qp_cache_);
        if (Kp_cache_.valid()) r.emplace_back(Kp_cache_);
        if (V_re_cache_.valid()) r.emplace_back(V_re_cache_);
        // RLA-2 的 RMSNorm 反向缓存（契约：activation_cache 必须覆盖 backward
        // 所需的全部激活，offload 才能完整换出并恢复）
        if (Q_normed_cache_.valid()) r.emplace_back(Q_normed_cache_);
        if (K_normed_cache_.valid()) r.emplace_back(K_normed_cache_);
        if (Q_rms_inv_cache_.valid()) r.emplace_back(Q_rms_inv_cache_);
        if (K_rms_inv_cache_.valid()) r.emplace_back(K_rms_inv_cache_);
        auto wq = w_q_.activation_cache(); r.insert(r.end(), wq.begin(), wq.end());
        auto wk = w_k_.activation_cache(); r.insert(r.end(), wk.begin(), wk.end());
        auto wv = w_v_.activation_cache(); r.insert(r.end(), wv.begin(), wv.end());
        auto wo = w_o_.activation_cache(); r.insert(r.end(), wo.begin(), wo.end());
        return r;
    }

    // ══════════════════════════════════════════════════════════════════════
    // RLA-2 前向（批量）：输入 X (d_model, batch·seq) → 输出 (d_model, batch·seq)
    //
    // 分母用 Sum 归一化（RLA-2 核心改动）：
    //   den_t = q'_t · z_t + ε,  z_t = Σ_{i≤t} k'_i
    //
    // 实现：两次 scan_prefix_outer（主扫描 + z-scan），z-scan 用 V=ones 获得 z。
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != d_model_)
            NN_FAIL("ReLULinearAttention forward: input shape mismatch");
        const std::size_t total = input.cols();
        const std::size_t seq = (seq_len_ > 0) ? seq_len_ : total;
        const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;
        if (total != batch * seq)
            NN_FAIL("ReLULinearAttention forward: cols not divisible by seq_len");
        const std::size_t H_dk = num_heads_ * d_k_;

        NN_TRY(q_res, w_q_.forward(input));
        NN_TRY(k_res, w_k_.forward(input));
        NN_TRY(v_res, w_v_.forward(input));

        Tensor Q, K, V;   // (BH*dk, seq) rearranged
        if (batch > 1)
        {
            NN_TRY(qr, engine.rearrange_3d(*q_res, H_dk, batch, seq, false));
            Q = std::move(*qr);
            NN_TRY(kr, engine.rearrange_3d(*k_res, H_dk, batch, seq, false));
            K = std::move(*kr);
            NN_TRY(vr, engine.rearrange_3d(*v_res, H_dk, batch, seq, false));
            V = std::move(*vr);
        }
        else
        {
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // RLA-2：RMSNorm on Q and K（per-head, dk blocks）
        // 稳定数值分布，减少神经元死亡，保持 ReLU 硬截断纯粹性（06-rapt-algorithm.md §3）。
        // checkpoint 模式：不驻留任何 backward 缓存，交由 forward_recompute 重建。
        {
            const std::size_t BHrms = batch * num_heads_;
            auto rq = rms_norm_forward_(engine, Q, BHrms, d_k_, seq,
                                        checkpoint_mode_ ? nullptr : &Q_rms_inv_cache_);
            NN_TRY_CHECK(rq);
            Q = std::move(*rq);
            if (!checkpoint_mode_) Q_normed_cache_ = Q;
            auto rk = rms_norm_forward_(engine, K, BHrms, d_k_, seq,
                                        checkpoint_mode_ ? nullptr : &K_rms_inv_cache_);
            NN_TRY_CHECK(rk);
            K = std::move(*rk);
            if (!checkpoint_mode_) K_normed_cache_ = K;
        }

        // RoPE → ReLU（顺序必须：先旋转后截断，否则丢位置信息）
        // 非 RoPE 策略在此是恒等 → 无标志位判断
        {
            NN_TRY(qr, pos_->apply_qk(engine, Q, seq, false));
            NN_TRY(kr, pos_->apply_qk(engine, K, seq, false));
        }
        auto Qp = dsl::compute(engine,
            dsl::max(dsl::leaf(Q), Scalar{0}), Q.rows(), Q.cols());
        NN_TRY_CHECK(Qp);
        auto Kp = dsl::compute(engine,
            dsl::max(dsl::leaf(K), Scalar{0}), K.rows(), K.cols());
        NN_TRY_CHECK(Kp);

        // O(L·d_k²) 运行态前缀和扫描（消除 O(L²) 得分矩阵物化）。
        const std::size_t BH = batch * num_heads_;
        const std::size_t BHdk = BH * d_k_;
        Tensor boundary_t;
        bool has_bnd = false;
        if (has_doc_ids_)
        {
            const auto boundary = build_boundary_(batch, seq);
            std::vector<Scalar> bm(batch * seq, Scalar{0});   // 宿主桥（17 §3 D11）
            for (std::size_t i = 0; i < boundary.size(); ++i)
                bm[i] = static_cast<Scalar>(boundary[i]);
            auto bt = detail::upload_span(engine, 1, batch * seq, Precision::F32,
                                          std::span(bm));
            NN_TRY_CHECK(bt);
            boundary_t = std::move(*bt);
            has_bnd = true;
        }
        else
        {
            NN_TRY(bd, make_dummy_(engine));
            boundary_t = std::move(*bd);
        }
        NN_TRY(dummy_r, make_dummy_(engine));
        const Tensor& dummy = *dummy_r;

        // 主扫描：读出 [0) B·P = num = B·q'（RLA-2 分子）
        auto Sc = engine.scan_prefix_outer(*Kp, V, *Qp, V, dummy, dummy, false,
                                           d_k_, num_heads_, causal_,
                                           boundary_t, has_bnd);
        NN_TRY_CHECK(Sc);
        NN_TRY(BP_r, engine.slice_rows(*Sc, 0, BHdk));

        // z-scan（RLA-2 核心）：V=ones → B_t = Σ 1·k'^T → B[*,c] = z[c]
        //   [0) B·P = q'·z（标量，头内逐行重复）
        //   [2) B^T·R = dk·z（向量，backward 用）
        NN_TRY(ones_r, ensure_ones_(engine, BH, d_k_, seq));
        auto Z_sc = engine.scan_prefix_outer(*Kp, V_ones_cache_, *Qp, V_ones_cache_,
                                             dummy, dummy, false,
                                             d_k_, num_heads_, causal_,
                                             boundary_t, has_bnd);
        NN_TRY_CHECK(Z_sc);
        NN_TRY(u_r, engine.slice_rows(*Z_sc, 0, BHdk));

        // den = q'·z + ε；out = num / den（逐元素链融合为单 kernel：num/(u+ε)）
        auto div_r = dsl::compute(engine,
            dsl::leaf(*BP_r) / (dsl::leaf(*u_r) + Scalar{1e-4}),
            (*BP_r).rows(), (*BP_r).cols());
        NN_TRY_CHECK(div_r);
        Tensor out_t = std::move(*div_r);

        Tensor concat;
        if (batch > 1)
        {
            NN_TRY(cb, engine.rearrange_3d(out_t, H_dk, batch, seq, true));
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(out_t);
        }

        // checkpoint 模式：backward 缓存全部不驻留（由 forward_recompute 重建）
        if (!checkpoint_mode_)
        {
            Qp_cache_ = std::move(*Qp);
            Kp_cache_ = std::move(*Kp);
            V_re_cache_ = std::move(V);
        }
        batch_cache_ = batch;
        seq_cache_   = seq;

        return w_o_.forward(concat);
    }

    // ══════════════════════════════════════════════════════════════════════
    // RLA-2 反向（批量）：grad_output (d_model, batch·seq) → 输入梯度
    //
    // 与原版 RLA 的差异：
    //   * 无 A 状态（Σ k'k'^T）：分母 den = q·z + ε，∂den/∂A = 0。
    //   * gQ = B^T·gnum + scale·z（替代原版 B^T·gnum + 2·ds·Aq）
    //   * gK = S_B^T·v + suffix(scale·q)（替代原版 2·S_A·k + S_B^T·v）
    //   * 无 dL/dA 项：A 不参与前向，无反向传播。
    //
    // 公式（causal，单头）：
    //   den = q·z + ε,  inv = 1/den,  gnum = g·inv
    //   r = g·(B·q),  scale = -r/den²
    //   gQ = B^T·gnum + scale·z
    //   dB = outer(gnum, q),  S_B = suffix(dB)
    //   gV = S_B·k,  gK = S_B^T·v + suffix(scale·q)
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // 缓存前置校验：checkpoint 模式（尚未 forward_recompute）或 offload
        // （尚未 import）下缓存为空，backward 必须立刻返回明确错误——否则空
        // 张量会流进 matmul 报出无关错误，甚至静默算出垃圾梯度。
        if (seq_cache_ == 0 || batch_cache_ == 0 ||
            !Qp_cache_.valid() || !Kp_cache_.valid() || !V_re_cache_.valid() ||
            !Q_normed_cache_.valid() || !K_normed_cache_.valid() ||
            !Q_rms_inv_cache_.valid() || !K_rms_inv_cache_.valid())
        {
            NN_FAIL("ReLULinearAttention::backward: forward 缓存缺失"                 "（checkpoint 模式需先 forward_recompute）");
        }
        const std::size_t seq = seq_cache_;
        const std::size_t batch = batch_cache_;
        const std::size_t H_dk = num_heads_ * d_k_;

        NN_TRY(gc, w_o_.backward(grad_output));
        Tensor gcr;
        if (batch > 1)
        {
            NN_TRY(g, engine.rearrange_3d(*gc, H_dk, batch, seq, false));
            gcr = std::move(*g);
        }
        else
        {
            gcr = std::move(*gc);
        }

        const std::size_t BH = batch * num_heads_;
        const std::size_t BHdk = BH * d_k_;
        Tensor boundary_t;
        bool has_bnd = false;
        if (has_doc_ids_)
        {
            const auto boundary = build_boundary_(batch, seq);
            std::vector<Scalar> bm(batch * seq, Scalar{0});   // 宿主桥（17 §3 D11）
            for (std::size_t i = 0; i < boundary.size(); ++i)
                bm[i] = static_cast<Scalar>(boundary[i]);
            auto bt = detail::upload_span(engine, 1, batch * seq, Precision::F32,
                                          std::span(bm));
            NN_TRY_CHECK(bt);
            boundary_t = std::move(*bt);
            has_bnd = true;
        }
        else
        {
            NN_TRY(bd, make_dummy_(engine));
            boundary_t = std::move(*bd);
        }
        NN_TRY(dummy_r, make_dummy_(engine));
        const Tensor& dummy = *dummy_r;

        // ── z-scan：获取 z 和 q·z（backward 需要 z 向量） ────────────
        NN_TRY(ones_r, ensure_ones_(engine, BH, d_k_, seq));
        auto Z_sc = engine.scan_prefix_outer(Kp_cache_, V_ones_cache_, Qp_cache_, V_ones_cache_,
                                             dummy, dummy, false,
                                             d_k_, num_heads_, causal_,
                                             boundary_t, has_bnd);
        NN_TRY_CHECK(Z_sc);
        auto u_r = engine.slice_rows(*Z_sc, 0, BHdk);         // q·z（标量重复 dk 次）
        NN_TRY_CHECK(u_r);
        auto z2_r = engine.slice_rows(*Z_sc, 2 * BHdk, BHdk); // dk·z（向量）
        NN_TRY_CHECK(z2_r);
        auto z_inv_r = dsl::compute(engine,
            dsl::leaf(*z2_r) * dsl::rparam(Scalar{1} / static_cast<Scalar>(d_k_)),
            z2_r->rows(), z2_r->cols());
        NN_TRY_CHECK(z_inv_r); // z = [2)/dk

        // ── pass 1：主扫描，读出 B^T·g 和 r = g·(B·q) ────────────
        auto Sc = engine.scan_prefix_outer(Kp_cache_, V_re_cache_, Qp_cache_, gcr,
                                           dummy, dummy, false,
                                           d_k_, num_heads_, causal_,
                                           boundary_t, has_bnd);
        NN_TRY_CHECK(Sc);
        auto BTR_r = engine.slice_rows(*Sc, 2 * BHdk, BHdk);  // B^T·g
        NN_TRY_CHECK(BTR_r);
        auto r_r = engine.slice_rows(*Sc, 4 * BHdk, BHdk);    // r = g·(B·q)
        NN_TRY_CHECK(r_r);

        // ── 公共中间量（逐元素链全融合为 DSL，消除 inv/den² 中间缓冲）──
        //   gnum = g/(u+ε)           （分子梯度，用于 outer→dB）
        //   scale = -r/(u+ε)²        （分母修正系数，用于 gQ/gK）
        //   gQt  = B^T·g/(u+ε) + scale·z   （gQ 完整表达式）
        const std::size_t cRows = (*u_r).rows(), cCols = (*u_r).cols();
        auto gnum_r = dsl::compute(engine,
            dsl::leaf(gcr) / (dsl::leaf(*u_r) + Scalar{1e-4}),
            cRows, cCols);
        NN_TRY_CHECK(gnum_r);
        auto scale_r = dsl::compute(engine,
            dsl::neg(dsl::leaf(*r_r))
                / ((dsl::leaf(*u_r) + Scalar{1e-4}) * (dsl::leaf(*u_r) + Scalar{1e-4})),
            cRows, cCols);
        NN_TRY_CHECK(scale_r); // -r/den²（标量重复 dk 次）

        // ── gQ = B^T·gnum + scale·z（全链融合为单 kernel） ─────────────
        auto gQt_r = dsl::compute(engine,
            dsl::leaf(*BTR_r) / (dsl::leaf(*u_r) + Scalar{1e-4})
                + dsl::leaf(*scale_r) * dsl::leaf(*z_inv_r),
            cRows, cCols);
        NN_TRY_CHECK(gQt_r);
        Tensor gQt = std::move(*gQt_r);

        // ── gV 和 gK ─────────────────────────────────────────────────
        Tensor gKt, gVt;
        // dB = outer(gnum, q) — dL/dB 矩阵
        NN_TRY(dB_r, engine.outer_col(*gnum_r, Qp_cache_, dummy, d_k_, false));

        if (causal_)
        {
            // 因果：S_B = suffix(dB)，gV = S_B·k，gK_B = S_B^T·v
            auto SBc_r = engine.scan_suffix_outer(*dB_r, Kp_cache_, V_re_cache_,
                                                  d_k_, num_heads_, true,
                                                  boundary_t, has_bnd);
            NN_TRY_CHECK(SBc_r);
            NN_TRY(gv_r, engine.slice_rows(*SBc_r, 0, BHdk));
            gVt = std::move(*gv_r);
            NN_TRY(gK_B_r, engine.slice_rows(*SBc_r, 2 * BHdk, BHdk));

            // suffix(scale·q)：D = outer(q, e_0, scale)，suffix(D)·e_0
            NN_TRY(D_zq_r, engine.outer_col(Qp_cache_, e_0_cache_, *scale_r, d_k_, true));
            auto SZ_r = engine.scan_suffix_outer(*D_zq_r, e_0_cache_, e_0_cache_,
                                                 d_k_, num_heads_, true,
                                                 boundary_t, has_bnd);
            NN_TRY_CHECK(SZ_r);
            NN_TRY(suffix_sq_r, engine.slice_rows(*SZ_r, 0, BHdk));
            auto gk_r = dsl::compute(engine,
                dsl::leaf(*gK_B_r) + dsl::leaf(*suffix_sq_r),
                gK_B_r->rows(), gK_B_r->cols());
            NN_TRY_CHECK(gk_r);
            gKt = std::move(*gk_r);
        }
        else
        {
            // 双向：A/B 为全集常数 → dB 在所有位置广播相同值
            // 归约步用 dsl::compute_reduce
            auto dB_sum_r = dsl::compute_reduce(engine,
                dsl::row_reduce_sum(dsl::leaf(*dB_r)),
                dB_r->rows(), dB_r->cols());
            NN_TRY_CHECK(dB_sum_r);
            // Bb = row_broadcast(dB_sum)：单条表达式一次 dispatch 完成，
            // 不物化中间张量。
            // 注：IR 规定"输出 = 最后一条指令的 dst"，故**单视图表达式不合法**
            // （指令表为空会被 validate_expr_spec 拒绝）。这里与一个**运行时 0**
            // （RParam，编译期无法被常量折叠掉）相加，使表达式合法且语义不变。
            auto Bb_r = dsl::compute(engine,
                dsl::row_broadcast(*dB_sum_r) + dsl::rparam(Scalar{0}),
                BHdk * d_k_, seq);
            NN_TRY_CHECK(Bb_r);
            auto SBc_r = engine.scan_suffix_outer(*Bb_r, Kp_cache_, V_re_cache_,
                                                  d_k_, num_heads_, false,
                                                  dummy, false);
            NN_TRY_CHECK(SBc_r);
            NN_TRY(gv_r, engine.slice_rows(*SBc_r, 0, BHdk));
            gVt = std::move(*gv_r);
            NN_TRY(gK_B_r, engine.slice_rows(*SBc_r, 2 * BHdk, BHdk));

            // 双向 gK 的常数项：Σ_t scale_t · q_t（逐元素链与行归约融合为单次
            // dispatch，不物化 scale_q 中间张量）
            auto tsq_r = dsl::compute_reduce(engine,
                dsl::row_reduce_sum(dsl::leaf(*scale_r) * dsl::leaf(Qp_cache_)),
                BHdk, seq);
            NN_TRY_CHECK(tsq_r);
            // gk = gK_B + row_broadcast(tsq)：单条表达式一次 dispatch 完成
            // （不物化中间张量）
            auto gk_r = dsl::compute(engine,
                dsl::leaf(*gK_B_r) + dsl::row_broadcast(*tsq_r),
                gK_B_r->rows(), gK_B_r->cols());
            NN_TRY_CHECK(gk_r);
            gKt = std::move(*gk_r);
        }

        // ── ReLU 反向：(x>0) ? g : 0 ───────────────────────────────
        auto gq_relu = dsl::compute(engine,
            dsl::select(dsl::leaf(Qp_cache_) > Scalar{0},
                        dsl::leaf(gQt), Scalar{0}),
            gQt.rows(), gQt.cols());
        NN_TRY_CHECK(gq_relu);
        auto gk_relu = dsl::compute(engine,
            dsl::select(dsl::leaf(Kp_cache_) > Scalar{0},
                        dsl::leaf(gKt), Scalar{0}),
            gKt.rows(), gKt.cols());
        NN_TRY_CHECK(gk_relu);

        // ── RoPE 反向（旋转正交，逆 = 反角；非 RoPE 策略 = 恒等）──────
        {
            NN_TRY(gq, pos_->apply_qk(engine, *gq_relu, seq, true));
            NN_TRY(gk, pos_->apply_qk(engine, *gk_relu, seq, true));
        }

        // ── RLA-2 RMSNorm 反向（RoPE 反向之后、rearrange 之前） ──────
        {
            const std::size_t BHrms = batch * num_heads_;
            auto gq_rn = rms_norm_backward_(engine, *gq_relu, Q_normed_cache_,
                                            Q_rms_inv_cache_, BHrms, d_k_, seq);
            NN_TRY_CHECK(gq_rn);
            gq_relu = std::move(*gq_rn);
            auto gk_rn = rms_norm_backward_(engine, *gk_relu, K_normed_cache_,
                                            K_rms_inv_cache_, BHrms, d_k_, seq);
            NN_TRY_CHECK(gk_rn);
            gk_relu = std::move(*gk_rn);
        }

        Tensor gq_r, gk_r, gv_r;
        if (batch > 1)
        {
            NN_TRY(a, engine.rearrange_3d(*gq_relu, H_dk, batch, seq, true));
            gq_r = std::move(*a);
            NN_TRY(b, engine.rearrange_3d(*gk_relu, H_dk, batch, seq, true));
            gk_r = std::move(*b);
            NN_TRY(c, engine.rearrange_3d(gVt, H_dk, batch, seq, true));
            gv_r = std::move(*c);
        }
        else
        {
            gq_r = std::move(*gq_relu);
            gk_r = std::move(*gk_relu);
            gv_r = std::move(gVt);
        }

        NN_TRY(giq, w_q_.backward(gq_r));
        NN_TRY(gik, w_k_.backward(gk_r));
        NN_TRY(giv, w_v_.backward(gv_r));
        // grad_input = gq + gk + gv：三路累加**原地**融合为单趟（目标传递，
        // 不额外分配）；求和按 (gq + gk) + gv 的固定顺序结合，跨 run 确定
        auto acc = dsl::compute_into(engine,
            dsl::leaf(*giq) + dsl::leaf(*gik) + dsl::leaf(*giv), *giq);
        NN_TRY_CHECK(acc);
        return giq;
    }

    // ── 增量推理（RLA-2 KV cache） ──────────────────────────────────
    // RLA-2 状态：B_state (d_model, d_k) + z_state (d_model, 1)
    //   B += k' ⊗ v,  z += k'
    //   num = B·q',  den = q'·z + ε,  out = num / den
    // 每步 O(d_k²)，与序列长度无关。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Tensor& B_state, Tensor& z_state, std::size_t pos)
    {
        auto q_res = w_q_.forward(input);   // (d_model, 1) = (H*dk, 1)
        NN_TRY_CHECK(q_res);
        NN_TRY(k_res, w_k_.forward(input));
        NN_TRY(v_res, w_v_.forward(input));
        Tensor Q = std::move(*q_res), K = std::move(*k_res), V = std::move(*v_res);
        // RLA-2：RMSNorm on Q and K（per-head, dk blocks）
        {
            Tensor dummy_ri;
            NN_TRY(rq, rms_norm_forward_(engine, Q, num_heads_, d_k_, 1, &dummy_ri));
            Q = std::move(*rq);
            NN_TRY(rk, rms_norm_forward_(engine, K, num_heads_, d_k_, 1, &dummy_ri));
            K = std::move(*rk);
        }
        {
            NN_TRY(qr, pos_->apply_qk_step(engine, Q, pos, false));
            NN_TRY(kr, pos_->apply_qk_step(engine, K, pos, false));
        }
        auto Qp = dsl::compute(engine,
            dsl::max(dsl::leaf(Q), Scalar{0}), Q.rows(), Q.cols());
        NN_TRY_CHECK(Qp);
        auto Kp = dsl::compute(engine,
            dsl::max(dsl::leaf(K), Scalar{0}), K.rows(), K.cols());
        NN_TRY_CHECK(Kp);
        const std::size_t H = num_heads_;
        const std::size_t dk = d_k_;

        // 更新运行态：B += k' ⊗ v，z += k'（原地累加，dsl::compute_into）
        // dsl::matmul(batch)：纯 {0,1} 结构（attention backward 已登记同 key）
        auto B_add_r = dsl::compute(engine,        // v·k'^T → (H*dk, dk)
            dsl::matmul(V, *Kp, false, true, H),
            V.rows(), Kp->rows() / H);
        NN_TRY_CHECK(B_add_r);
        { auto r = dsl::compute_into(engine,
              dsl::leaf(B_state) + dsl::leaf(*B_add_r), B_state);
          NN_TRY_CHECK(r); }
        { auto r = dsl::compute_into(engine,               // z += k'（(H*dk, 1)）
              dsl::leaf(z_state) + dsl::leaf(*Kp), z_state);
          NN_TRY_CHECK(r); }

        // num = B·q'：batched_matmul(B_state, Qp, H) → (H*dk, 1)
        // dsl::matmul(batch)：纯 {0,0} 结构（scan 手工登记同 key）
        auto num_r = dsl::compute(engine,
            dsl::matmul(B_state, *Qp, false, false, H),
            B_state.rows(), Qp->cols());
        NN_TRY_CHECK(num_r);

        // den = q'·z：逐头标量点积（通过宿主桥在 CPU 上计算，17 §3 D11；
        // ε 由下面的除法表达式统一添加，与 forward 的写法保持同构）
        // forward_step 是逐 token 串行的，CPU round-trip 可接受。
        NN_TRY(q_v, detail::download_vector(engine, *Qp));
        NN_TRY(z_v, detail::download_vector(engine, z_state));
        // 广播到 (H*dk, 1)：每头标量重复 dk 次
        std::vector<Scalar> den_full(H * dk);
        for (std::size_t h = 0; h < H; ++h)
        {
            Scalar dot = 0;
            for (std::size_t j = 0; j < dk; ++j)
                dot += (*q_v)[h * dk + j] * (*z_v)[h * dk + j];
            for (std::size_t j = 0; j < dk; ++j)
                den_full[h * dk + j] = dot;
        }
        auto den_t = detail::upload_span(engine, H * dk, 1, Precision::F32,
                                         std::span(den_full));
        NN_TRY_CHECK(den_t);

        // out = num / (den + ε)：与 forward 的除法（:491）**逐 token 同构**——
        // 同一个 AOT 键由 scan_exprs 的 ReLULinearAttention dry-run 覆盖，
        // GPU 闭合世界可命中。
        auto out_r = dsl::compute(engine,
            dsl::leaf(*num_r) / (dsl::leaf(*den_t) + Scalar{1e-4}),
            (*num_r).rows(), (*num_r).cols());
        NN_TRY_CHECK(out_r);
        return w_o_.forward(*out_r);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTBlock — RAPT 解码器块（GPT 风格：Norm → RLA-2 注意力 → 残差 → FFN → 残差）
// ══════════════════════════════════════════════════════════════════════════
class RAPTBlock final : public Layer
{
private:
    std::unique_ptr<Layer> norm1_;
    ReLULinearAttention attn_;
    std::unique_ptr<Layer> norm2_;
    FeedForward ff_;
    ActivationOffloader offloader_;   // L1-offload（与 GPTBlock 共用实现）

public:
    RAPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
              std::size_t seq_len, PosEncodingType pos_enc,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true,
              PrecisionProfile precision = PrecisionProfile{})
        : norm1_(make_norm_layer(d_model, norm_type)),
          attn_(d_model, num_heads, seq_len, causal, pos_enc),
          norm2_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation)
    {
        // D7：将精度配置注入所有子层（§9.2）
        set_precision_profile(precision);
        if (norm1_) norm1_->set_precision_profile(precision);
        attn_.set_precision_profile(precision);
        if (norm2_) norm2_->set_precision_profile(precision);
        ff_.set_precision_profile(precision);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        NN_TRY(r1, norm1_->init(engine));
        NN_TRY(r2, attn_.init(engine));
        NN_TRY(r3, norm2_->init(engine));
        NN_TRY(r4, ff_.init(engine));
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        return collect_refs(norm1_->parameters(),
                            attn_.parameters(),
                            norm2_->parameters(),
                            ff_.parameters());
    }
    std::vector<TensorRef> param_gradients() override
    {
        return collect_refs(norm1_->param_gradients(),
                            attn_.param_gradients(),
                            norm2_->param_gradients(),
                            ff_.param_gradients());
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        norm1_->set_checkpoint_mode(enabled);
        attn_.set_checkpoint_mode(enabled);
        norm2_->set_checkpoint_mode(enabled);
        ff_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        norm1_->clear_cache(); attn_.clear_cache();
        norm2_->clear_cache(); ff_.clear_cache();
    }

    // RAPTBlock 与 GPTBlock 一样是"重计算单元"。
    [[nodiscard]] bool recompute_supported() const override { return true; }

    // 重计算：必须走**虚函数** set_checkpoint_mode 关闭本块+子层的 checkpoint
    // 模式（基类默认实现只改本块的标志位，子层仍处于 checkpoint 模式 →
    // forward 不会重建子层缓存 → backward 要么报错、要么误用上一 step 的陈旧缓存）。
    [[nodiscard]] Result<Tensor> forward_recompute(
        const Tensor& saved_input) override
    {
        set_checkpoint_mode(false);
        auto r = forward(saved_input);
        set_checkpoint_mode(true);
        return r;
    }

    // ── activation offload（L1-offload）────────────────────────────────
    // 导出/导入本块 backward 所需的全部激活（attn + 两个 Norm + FFN）。
    void set_offload_enabled(bool enabled) { offloader_.set_enabled(enabled); }
    [[nodiscard]] bool offload_enabled() const noexcept { return offloader_.enabled(); }
    [[nodiscard]] Result<void> export_activations(ComputeEngine& engine)
    {
        return offloader_.export_activations(engine, activation_cache());
    }
    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        return offloader_.import_activations(engine);
    }
    [[nodiscard]] std::size_t offload_slab_bytes() const noexcept
    {
        return offloader_.slab_bytes();
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        auto a = attn_.activation_cache(); r.insert(r.end(), a.begin(), a.end());
        auto n1 = norm1_->activation_cache(); r.insert(r.end(), n1.begin(), n1.end());
        auto n2 = norm2_->activation_cache(); r.insert(r.end(), n2.begin(), n2.end());
        auto f = ff_.activation_cache(); r.insert(r.end(), f.begin(), f.end());
        return r;
    }

    // 文档感知：转发给内部 RLA-2 注意力（文档边界处重置运行态）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        attn_.set_doc_ids(ids);
    }

    // 绝对位置偏移：转发给内部 RLA-2 注意力（滑动窗生成时 RoPE 用绝对位置）
    void set_position_offset(std::size_t off)
    {
        attn_.set_position_offset(off);
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        // 两个残差分支（与 GPTBlock/TransformerEncoderLayer 同骨架）。
        // 残差相加历史上未传 profile（恒 F32）——显式传以保持逐位不变。
        auto r1 = prenorm_residual_forward_(engine, input, *norm1_, attn_,
                                            Precision::F32);
        NN_TRY_CHECK(r1);
        return prenorm_residual_forward_(engine, *r1, *norm2_, ff_,
                                         Precision::F32);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        // activation offload：从 host 恢复激活再反向（替代重计算）
        if (offloader_.offloaded())
        {
            NN_TRY(im, offloader_.import_activations(engine));
        }
        // 两处残差分流 + 子层反向（与 GPTBlock / TransformerEncoderLayer 同一骨架）。
        // 残差相加历史上未传 profile（恒 F32）——显式传 Precision::F32 保持逐位不变。
        return prenorm_residual_backward_(engine, grad_output,
                                          *norm1_, attn_, *norm2_, ff_,
                                          Precision::F32);
    }

    // 增量推理：单 token → norm1 → RLA-2 运行态注意力 → 残差 → norm2 → FFN → 残差
    // B_state/z_state: RLA-2 运行态 KV cache（见 ReLULinearAttention::forward_step）
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Tensor& B_state, Tensor& z_state, std::size_t pos)
    {
        NN_TRY(n1, norm1_->forward(input));
        NN_TRY(a, attn_.forward_step(engine, *n1, B_state, z_state, pos));
        auto r1 = dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(*a),
            input.rows(), input.cols());
        NN_TRY_CHECK(r1);
        NN_TRY(n2, norm2_->forward(*r1));
        NN_TRY(f, ff_.forward(*n2));
        return dsl::compute(engine,
            dsl::leaf(*r1) + dsl::leaf(*f),
            r1->rows(), r1->cols());
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTModel — RAPT 解码器（RLA-2 线性注意力语言模型）
//
//   token_emb → (+pos_enc，RoPE 在注意力内部施加) → N × RAPTBlock → LN → LM Head
//
// RLA-2 约束：位置编码必须用 RoPE（或 ALiBi），且 RoPE 施加在 Q/K 进 ReLU 之前；
// 本实现强制 RoPE（v1 不支持 ALiBi），输入侧用 NoPositionEncoder（无位置嵌入）。
// ══════════════════════════════════════════════════════════════════════════
class RAPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;
    std::size_t num_heads_;   // 用于运行态 KV cache 尺寸（d_k = d_model/num_heads）

    Tensor token_emb_;
    Tensor grad_token_emb_;
    std::unique_ptr<PositionEncoder> pos_encoder_;
    std::vector<RAPTBlock> blocks_;
    std::unique_ptr<Layer> ln_f_;
    Linear lm_head_;

    Tensor stored_tokens_tensor_;
    std::size_t batch_size_ = 0;
    std::vector<std::size_t> doc_ids_;   // 文档感知：每位置文档 id（batch-major）

    // batch 录制粒度：每隔 flush_interval_ 个块提交一次（0 = 不在块间 flush）
    std::size_t flush_interval_ = 0;

    // 梯度检查点（激活重计算 L1）：每隔 checkpoint_every_ 个块保存一次块输入，
    // backward 时重算以省去驻留整层激活。0 = 不启用。
    std::size_t checkpoint_every_ = 0;
    std::vector<Tensor> checkpoint_inputs_;   // 各 checkpoint 块的输入 (d_model, batch*seq)

    // activation offload（L1-offload）：把每块内部激活搬 host-visible，backward 拷回
    bool activation_offload_ = false;

public:
    RAPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
              std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
              PosEncodingType pos_enc = PosEncodingType::RoPE,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true,
              PrecisionProfile precision = PrecisionProfile{})
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          num_heads_(num_heads),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        // D7：将精度配置注入自身和所有子层（§9.2）
        set_precision_profile(precision);
        if (ln_f_) ln_f_->set_precision_profile(precision);
        lm_head_.set_precision_profile(precision);

        // ── 位置编码器（**嵌入侧**）：模型只负责"加到 token 嵌入上"那一半 ────
        // RLA-2 强约束：必须 RoPE（v1 不支持 ALiBi）。RoPE 的注入点在注意力层
        // （Q/K 进 ReLU 之前），由 ReLULinearAttention **自持**；嵌入侧是恒等。
        pos_encoder_ = make_embedding_position_encoder(pos_enc, d_model, seq_len);

        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
        {
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, pos_enc,
                                 activation, norm_type, causal, precision);
        }
        pos_encoder_->set_precision_profile(precision);
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // 初始化 token_emb_——M2 声明式：N(0, 0.02) 层算参数、引擎填数
        constexpr Scalar emb_init_std = 0.02;
        token_emb_ = engine.create_tensor(vocab_size_, d_model_, Precision::F32,
                                          InitSpec::normal(0, emb_init_std, kInitSeed));
        if (!token_emb_.valid())
            NN_FAIL("RAPTModel: token_emb 初始化失败");
        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_, Precision::F32,
                                               InitSpec::zero());
        if (!grad_token_emb_.valid())
            NN_FAIL("RAPTModel: token_emb 梯度缓冲初始化失败");
        if (pos_encoder_)
        {
            NN_TRY(r, pos_encoder_->init(engine));
        }
        for (auto& b : blocks_)
        {
            NN_TRY(r, b.init(engine));
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
                            collect_block_refs_(blocks_, &RAPTBlock::parameters),
                            ln_f_->parameters(),
                            lm_head_.parameters());
    }
    std::vector<TensorRef> param_gradients() override
    {
        return collect_refs(grad_token_emb_,
                            pos_encoder_->param_gradients(),
                            collect_block_refs_(blocks_, &RAPTBlock::param_gradients),
                            ln_f_->param_gradients(),
                            lm_head_.param_gradients());
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    // 只释放 backward 中间激活（Layer::clear_cache 契约）。
    // 注意：token_emb_ / grad_token_emb_ 是**模型参数**，绝不在此清理——
    // 它们由 Model/optimizer 持有，清理会导致词嵌入被销毁。
    // stored_tokens_tensor_ 是 backward 末尾 scatter_add_rows 的索引张量（GPTModel
    // 同样保留），清理会让梯度写不回词嵌入表。
    void clear_cache() override
    {
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        checkpoint_inputs_.clear();
    }

    // 文档感知：记录 doc_ids，forward 时下发给各块（文档边界处重置 RLA-2 运行态）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        if (ids.empty()) { doc_ids_.clear(); return; }
        doc_ids_.assign(ids.begin(), ids.end());
    }

    // 绝对位置偏移：下发给各块（滑动窗生成时 RoPE 用绝对位置，非 0..seq-1 重置）
    void set_position_offset(std::size_t off)
    {
        for (auto& b : blocks_) b.set_position_offset(off);
    }

    // ── batch 录制粒度（防 TDR）：每 N 个块提交一次，0 = 不启用 ──
    void set_flush_interval(std::size_t interval) override { flush_interval_ = interval; }
    [[nodiscard]] std::size_t flush_interval() const noexcept { return flush_interval_; }

    // ── 梯度检查点（激活重计算 L1）：每 N 个块保存一次块输入，0 = 不启用 ──
    void set_checkpoint_every(std::size_t stride) override { checkpoint_every_ = stride; }
    [[nodiscard]] std::size_t checkpoint_every() const noexcept { return checkpoint_every_; }

    // ── activation offload（L1-offload）：把每块激活搬 host-visible ──
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
        for (auto& b : blocks_) total += b.offload_slab_bytes();
        return total;
    }

    [[nodiscard]] Result<Tensor> forward(
        const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        const std::size_t seq = input.rows();
        const std::size_t batch = input.cols();
        batch_size_ = batch;

        auto input_T = engine.transpose(input);   // (batch, seq)
        NN_TRY_CHECK(input_T);
        NN_TRY(all_emb, engine.gather_rows(token_emb_, *input_T));
        NN_TRY(st, engine.clone(*input_T));
        stored_tokens_tensor_ = std::move(*st);
        NN_TRY(all_T, engine.transpose(*all_emb));

        NN_TRY(x_res, pos_encoder_->apply(engine, *all_T, batch, seq));
        Tensor x = std::move(*x_res);

        checkpoint_inputs_.clear();
        const bool ckpt = (checkpoint_every_ > 0);
        for (std::size_t bi = 0; bi < blocks_.size(); ++bi)
        {
            RAPTBlock& b = blocks_[bi];
            // 无条件下发：空 span 也要清掉上一 step 的文档感知（否则跨 step 串扰）
            b.set_doc_ids(doc_ids_);
            // 梯度检查点：每 checkpoint_every_ 个块保存一次输入，该块以 checkpoint
            // 模式 forward（不驻留中间激活），backward 时用保存的输入重算
            if (ckpt && (bi % checkpoint_every_ == 0))
            {
                NN_TRY(save, engine.clone(x));
                checkpoint_inputs_.push_back(std::move(*save));
                b.set_checkpoint_mode(true);
            }
            else
            {
                b.set_checkpoint_mode(false);
            }
            NN_TRY(r, b.forward(x));
            x = std::move(*r);
            // activation offload：forward 后把本块内部激活搬 host-visible（释放
            // 显存）。checkpoint 块 forward 不驻留激活 → 无可导出内容，必须跳过。
            if (activation_offload_ && !b.checkpoint_mode())
            {
                NN_TRY(ex, b.export_activations(engine));
            }
            // 按间隔 flush，将大录制拆成多个小提交（防 TDR）
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0
                && bi + 1 < blocks_.size())
            {
                NN_TRY(fr, engine.flush_batch());
            }
        }
        NN_TRY(ln, ln_f_->forward(x));
        return lm_head_.forward(*ln);
    }

    [[nodiscard]] Result<Tensor> backward(
        const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        const std::size_t seq = seq_len_;
        const std::size_t batch = batch_size_;

        NN_TRY(b_lm, lm_head_.backward(grad_output));
        NN_TRY(b_ln, ln_f_->backward(*b_lm));
        Tensor grad_x = std::move(*b_ln);

        const std::size_t n = blocks_.size();
        for (std::size_t bi = 0; bi < n; ++bi)
        {
            const std::size_t idx = n - 1 - bi;
            // 梯度检查点：checkpoint 块先用保存的输入重算 forward 重建缓存，再反向
            if (checkpoint_every_ > 0 && (idx % checkpoint_every_ == 0))
            {
                const std::size_t seg = idx / checkpoint_every_;
                NN_ASSERT(seg < checkpoint_inputs_.size(),
                          "RAPTModel backward: checkpoint input missing");
                NN_TRY(cr, blocks_[idx].forward_recompute(checkpoint_inputs_[seg]));
            }
            auto br = blocks_[idx].backward(grad_x);
            if (!br)
                NN_FAIL("RAPTModel::backward: block " + std::to_string(idx) + " failed: "                     + br.error().message);
            grad_x = std::move(*br);
            // 重算/恢复出来的激活用后即释放，避免跨块累积（否则抵消省显存收益）
            if (checkpoint_every_ > 0 || activation_offload_)
                blocks_[idx].clear_cache();
            if (flush_interval_ > 0 && (bi + 1) % flush_interval_ == 0 && bi + 1 < n)
            {
                NN_TRY(fr, engine.flush_batch());
            }
        }
        checkpoint_inputs_.clear();

        NN_TRY(grad_T, engine.transpose(grad_x));
        NN_TRY(pr, pos_encoder_->backward(engine, *grad_T, batch, seq));
        NN_TRY(sr, engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T));

        Tensor grad_input = engine.create_tensor(seq, batch, Precision::F32,
                                                 InitSpec::zero());
        if (!grad_input.valid())
            NN_FAIL("RAPT token_emb backward: 梯度张量分配失败");
        return grad_input;
    }

    // ── 采样生成（RLA-2 增量运行态，KV cache） ─────────────────────
    // 利用 RLA-2 的运行态（B_t、z_t）作为 KV cache：逐 token 增量更新，
    // 每步 O(d²)（不重算前文），使生成真正线性于上下文长度。
    [[nodiscard]] Result<std::vector<std::size_t>>
    generate(ComputeEngine& engine,
             const std::vector<std::size_t>& prompt,
             std::size_t max_new_tokens,
             Scalar temperature = 1.0,
             std::size_t eos_token_id = static_cast<std::size_t>(-1),
             std::size_t min_new_tokens = 0)
    {
        if (prompt.empty())
            NN_FAIL("RAPT generate: empty prompt");
        const std::size_t dk = d_model_ / num_heads_;
        // RLA-2 每块两个运行态：B_state (d_model, d_k) + z_state (d_model, 1)
        std::vector<Tensor> statesB, statesZ;
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            statesB.emplace_back(engine.create_tensor(d_model_, dk));
            { NN_TRY(r, engine.zero(statesB.back())); }
            statesZ.emplace_back(engine.create_tensor(d_model_, 1));
            { NN_TRY(r, engine.zero(statesZ.back())); }
        }

        // 处理单个 token：embed → 各块 forward_step → LN → LM head → 返回 logits 列
        auto step_one = [&](std::size_t tok, std::size_t pos)
            -> Result<std::vector<Scalar>>
        {
            std::vector<Scalar> idx_v(1);              // 宿主桥（17 §3 D11）
            idx_v[0] = static_cast<Scalar>(tok);
            auto idx_t = detail::upload_span(engine, 1, 1, Precision::F32,
                                             std::span(idx_v));
            NN_TRY_CHECK(idx_t);
            auto emb = engine.gather_rows(token_emb_, *idx_t);   // (1, d_model)
            NN_TRY_CHECK(emb);
            auto emb_t = engine.transpose(*emb);                  // (d_model, 1)
            NN_TRY_CHECK(emb_t);
            Tensor h = std::move(*emb_t);
            for (std::size_t i = 0; i < blocks_.size(); ++i)
            {
                NN_TRY(r, blocks_[i].forward_step(engine, h, statesB[i], statesZ[i], pos));
                h = std::move(*r);
            }
            NN_TRY(ln, ln_f_->forward(h));
            NN_TRY(logits, lm_head_.forward(*ln));
            auto lm_v = detail::download_vector(engine, *logits);   // 宿主桥 D11
            NN_TRY_CHECK(lm_v);
            const std::size_t lm_cols = logits->cols();
            std::vector<Scalar> last(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last[v] = (*lm_v)[v * lm_cols];
            return last;
        };

        // 逐 token 处理 prompt（建立运行态）；保留最后一步 logits（预测下一 token）
        std::size_t pos = 0;
        std::vector<Scalar> last;
        for (std::size_t i = 0; i < prompt.size(); ++i)
        {
            NN_TRY(r, step_one(prompt[i], pos++));
            last = std::move(*r);
        }

        // 采样生成
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            // temperature → softmax → 采样/贪心（三模型共用的采样器）
            std::vector<Scalar> lastv = last;
            const std::size_t next = sample_next_token_(lastv, temperature, rng, dist);

            if (step >= min_new_tokens && next == eos_token_id) break;
            generated.push_back(next);

            NN_TRY(r, step_one(next, pos++));
            last = std::move(*r);
        }
        return generated;
    }
};

} // namespace nn
