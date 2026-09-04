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
// ReLULinearAttention — RLA-2：极简硬截断线性注意力（docs/22-rla2.md）
//
// 算法（RLA-2，修正版；causal / bidirectional 均支持）：
//   q' = ReLU(RoPE(q)), k' = ReLU(RoPE(k)), v = W_v·x（V 不做 ReLU）
//   分子 num_t  = B_t · q'_t,   B_t = Σ_{i∈S_t} v_i k'_i^T
//   分母 den_t  = q'_t · z_t + ε,  z_t = Σ_{i∈S_t} k'_i
//   out_t = num_t / den_t
//   其中 S_t = { i<=t }（causal）或 { 全部 }（bidirectional）。
//
// 与原版 RLA 的关键差异（§3 文档 22）：
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
    bool use_rope_;          // RoPE 施加在 Q/K 进 ReLU 之前（RLA-2 强约束）
    RotaryEmbedding rope_;
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
        auto r = engine.zero(d);
        if (!r) return std::unexpected(r.error());
        return d;
    }

    // 确保 V_ones / e_0 缓存与当前 BH·dk × seq 尺寸匹配
    void ensure_ones_(ComputeEngine& engine, std::size_t BH, std::size_t dk, std::size_t seq)
    {
        if (ones_BH_ == BH && ones_seq_ == seq && V_ones_cache_.valid())
            return;
        const std::size_t rows = BH * dk;
        // V_ones：全1
        Matrix ones_mat(rows, seq);
        {
            auto sp = ones_mat.span();
            std::fill(sp.begin(), sp.end(), Scalar{1});
        }
        auto ot = engine.from_matrix(ones_mat);
        if (ot) V_ones_cache_ = std::move(*ot);
        // e_0：每头首行=1，其余=0（用于 suffix(scale·q) 的外积构造）
        Matrix e0_mat(rows, seq);
        e0_mat.zero();
        {
            auto sp = e0_mat.span();
            for (std::size_t bh = 0; bh < BH; ++bh)
                for (std::size_t t = 0; t < seq; ++t)
                    sp[(bh * dk) * seq + t] = Scalar{1};
        }
        auto e0t = engine.from_matrix(e0_mat);
        if (e0t) e_0_cache_ = std::move(*e0t);
        ones_BH_  = BH;
        ones_seq_ = seq;
    }

    // ── 逐头 RMSNorm 前向（RLA-2：投影后 RMSNorm → RoPE → ReLU） ────
    // 对 (BH*dk, seq) 张量的每个 dk 块独立归一化：y = x / sqrt(mean(x²) + eps)
    // 无学习参数（固定增益=1），符合极简原则。
    [[nodiscard]] Result<Tensor> rms_norm_forward_(
        ComputeEngine& engine, const Tensor& input,
        std::size_t BH, std::size_t dk, std::size_t seq, Tensor& rms_inv_out)
    {
        const Scalar inv_dk = Scalar{1} / static_cast<Scalar>(dk);
        const Scalar eps = Scalar{1e-5};
        Tensor output = engine.create_tensor(BH * dk, seq);
        Matrix rms_mat(BH, seq);
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            auto x = engine.slice_rows(input, bh * dk, dk);
            if (!x) return std::unexpected(x.error());
            auto x_sq = engine.elementwise_binary(BinaryOp::Mul, *x, *x);
            if (!x_sq) return std::unexpected(x_sq.error());
            auto s = engine.col_reduce_sum(*x_sq);
            if (!s) return std::unexpected(s.error());
            auto m = engine.elementwise_binary_scalar(BinaryOp::Mul, *s, inv_dk);
            if (!m) return std::unexpected(m.error());
            auto ve = engine.elementwise_binary_scalar(BinaryOp::Add, *m, eps);
            if (!ve) return std::unexpected(ve.error());
            auto ri = engine.elementwise_unary(UnaryOp::Rsqrt, *ve);
            if (!ri) return std::unexpected(ri.error());
            auto ri_m = engine.to_matrix(*ri);
            if (!ri_m) return std::unexpected(ri_m.error());
            for (std::size_t t = 0; t < seq; ++t)
                rms_mat.set_value_unchecked(bh, t, ri_m->at_unchecked(0, t));
            auto n = dsl::compute(engine,
                dsl::leaf(*x) * dsl::col_broadcast(*ri), dk, seq);
            if (!n) return std::unexpected(n.error());
            auto ins = engine.insert_rows(output, bh * dk, *n);
            if (!ins) return std::unexpected(ins.error());
        }
        auto ri_t = engine.from_matrix(rms_mat);
        if (!ri_t) return std::unexpected(ri_t.error());
        rms_inv_out = std::move(*ri_t);
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
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            auto gy = engine.slice_rows(grad, bh * dk, dk);
            if (!gy) return std::unexpected(gy.error());
            auto y = engine.slice_rows(normed, bh * dk, dk);
            if (!y) return std::unexpected(y.error());
            auto ri = engine.slice_rows(rms_inv, bh, 1);
            if (!ri) return std::unexpected(ri.error());
            // m = (1/dk) · col_reduce_sum(gy * y)  → (1, seq)
            auto gy_y = engine.elementwise_binary(BinaryOp::Mul, *gy, *y);
            if (!gy_y) return std::unexpected(gy_y.error());
            auto m_raw = engine.col_reduce_sum(*gy_y);
            if (!m_raw) return std::unexpected(m_raw.error());
            auto m = engine.elementwise_binary_scalar(BinaryOp::Mul, *m_raw, inv_dk);
            if (!m) return std::unexpected(m.error());
            // grad_x = (gy - m·y) · rms_inv
            auto term = dsl::compute(engine,
                dsl::col_broadcast(*m) * dsl::leaf(*y), dk, seq);
            if (!term) return std::unexpected(term.error());
            auto diff = engine.elementwise_binary(BinaryOp::Sub, *gy, *term);
            if (!diff) return std::unexpected(diff.error());
            auto gx = dsl::compute(engine,
                dsl::leaf(*diff) * dsl::col_broadcast(*ri), dk, seq);
            if (!gx) return std::unexpected(gx.error());
            auto ins = engine.insert_rows(output, bh * dk, *gx);
            if (!ins) return std::unexpected(ins.error());
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
          use_rope_(pos_enc == PosEncodingType::RoPE),
          rope_(d_model / num_heads),
          w_q_(d_model, d_model), w_k_(d_model, d_model),
          w_v_(d_model, d_model), w_o_(d_model, d_model)
    {
        NN_ASSERT(d_model % num_heads == 0,
                  "ReLULinearAttention: d_model must be divisible by num_heads");
        NN_ASSERT(d_model % num_heads == 0 && (d_model / num_heads) % 2 == 0,
                  "ReLULinearAttention: RoPE requires even d_k");
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = w_q_.init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = w_k_.init(engine); if (!r2) return std::unexpected(r2.error());
        auto r3 = w_v_.init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = w_o_.init(engine); if (!r4) return std::unexpected(r4.error());
        if (use_rope_)
        {
            auto r5 = rope_.init(engine); if (!r5) return std::unexpected(r5.error());
        }
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

    // 绝对位置偏移（滑动窗生成用）：转发给 RoPE，使重计算式生成的
    // 位置从真实起点算起而非每次从 0 重置。
    void set_position_offset(std::size_t off)
    {
        if (use_rope_) rope_.set_position_offset(off);
    }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (Qp_cache_.valid()) r.emplace_back(Qp_cache_);
        if (Kp_cache_.valid()) r.emplace_back(Kp_cache_);
        if (V_re_cache_.valid()) r.emplace_back(V_re_cache_);
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
        ComputeEngine& engine, const Tensor& input) override
    {
        if (input.rows() != d_model_)
            return std::unexpected(Error{"ReLULinearAttention forward: input shape mismatch"});

        const std::size_t total = input.cols();
        const std::size_t seq = (seq_len_ > 0) ? seq_len_ : total;
        const std::size_t batch = (seq_len_ > 0) ? (total / seq_len_) : 1;
        if (total != batch * seq)
            return std::unexpected(Error{"ReLULinearAttention forward: cols not divisible by seq_len"});
        const std::size_t H_dk = num_heads_ * d_k_;

        auto q_res = w_q_.forward(engine, input);
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;

        Tensor Q, K, V;   // (BH*dk, seq) rearranged
        if (batch > 1)
        {
            auto qr = engine.rearrange_3d(*q_res, H_dk, batch, seq, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = engine.rearrange_3d(*k_res, H_dk, batch, seq, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
            auto vr = engine.rearrange_3d(*v_res, H_dk, batch, seq, false); if (!vr) return std::unexpected(vr.error());
            V = std::move(*vr);
        }
        else
        {
            Q = std::move(*q_res);
            K = std::move(*k_res);
            V = std::move(*v_res);
        }

        // RLA-2：RMSNorm on Q and K（per-head, dk blocks）
        // 稳定数值分布，减少神经元死亡，保持 ReLU 硬截断纯粹性（文档 22 §4.3）。
        {
            const std::size_t BHrms = batch * num_heads_;
            Tensor Qn, Kn;
            auto rq = rms_norm_forward_(engine, Q, BHrms, d_k_, seq, Q_rms_inv_cache_);
            if (!rq) return std::unexpected(rq.error());
            Q = std::move(*rq);
            Q_normed_cache_ = Q;
            auto rk = rms_norm_forward_(engine, K, BHrms, d_k_, seq, K_rms_inv_cache_);
            if (!rk) return std::unexpected(rk.error());
            K = std::move(*rk);
            K_normed_cache_ = K;
        }

        // RoPE → ReLU（顺序必须：先旋转后截断，否则丢位置信息）
        if (use_rope_)
        {
            auto qr = rope_.apply(engine, Q, seq, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply(engine, K, seq, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = dsl::compute(engine,
            dsl::max(dsl::leaf(Q), Scalar{0}), Q.rows(), Q.cols());
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = dsl::compute(engine,
            dsl::max(dsl::leaf(K), Scalar{0}), K.rows(), K.cols());
        if (!Kp) return std::unexpected(Kp.error());

        // O(L·d_k²) 运行态前缀和扫描（消除 O(L²) 得分矩阵物化）。
        const std::size_t BH = batch * num_heads_;
        const std::size_t BHdk = BH * d_k_;
        Tensor boundary_t;
        bool has_bnd = false;
        if (has_doc_ids_)
        {
            const auto boundary = build_boundary_(batch, seq);
            Matrix bm(1, batch * seq, Scalar{0});
            for (std::size_t i = 0; i < boundary.size(); ++i)
                bm.set_value_unchecked(0, i, static_cast<Scalar>(boundary[i]));
            auto bt = engine.from_matrix(bm);
            if (!bt) return std::unexpected(bt.error());
            boundary_t = std::move(*bt);
            has_bnd = true;
        }
        else
        {
            auto bd = make_dummy_(engine);
            if (!bd) return std::unexpected(bd.error());
            boundary_t = std::move(*bd);
        }
        auto dummy_r = make_dummy_(engine);
        if (!dummy_r) return std::unexpected(dummy_r.error());
        const Tensor& dummy = *dummy_r;

        // 主扫描：读出 [0) B·P = num = B·q'（RLA-2 分子）
        auto Sc = engine.scan_prefix_outer(*Kp, V, *Qp, V, dummy, dummy, false,
                                           d_k_, num_heads_, causal_,
                                           boundary_t, has_bnd);
        if (!Sc) return std::unexpected(Sc.error());
        auto BP_r = engine.slice_rows(*Sc, 0, BHdk);
        if (!BP_r) return std::unexpected(BP_r.error());

        // z-scan（RLA-2 核心）：V=ones → B_t = Σ 1·k'^T → B[*,c] = z[c]
        //   [0) B·P = q'·z（标量，头内逐行重复）
        //   [2) B^T·R = dk·z（向量，backward 用）
        ensure_ones_(engine, BH, d_k_, seq);
        auto Z_sc = engine.scan_prefix_outer(*Kp, V_ones_cache_, *Qp, V_ones_cache_,
                                             dummy, dummy, false,
                                             d_k_, num_heads_, causal_,
                                             boundary_t, has_bnd);
        if (!Z_sc) return std::unexpected(Z_sc.error());
        auto u_r = engine.slice_rows(*Z_sc, 0, BHdk);
        if (!u_r) return std::unexpected(u_r.error());

        // den = q'·z + ε；out = num / den
        auto den_r = engine.elementwise_binary_scalar(BinaryOp::Add, *u_r, Scalar{1e-4});
        if (!den_r) return std::unexpected(den_r.error());
        auto div_r = engine.elementwise_binary(BinaryOp::Div, *BP_r, *den_r);
        if (!div_r) return std::unexpected(div_r.error());
        Tensor out_t = std::move(*div_r);

        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(out_t, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(out_t);
        }

        Qp_cache_ = std::move(*Qp);
        Kp_cache_ = std::move(*Kp);
        V_re_cache_ = std::move(V);
        batch_cache_ = batch;
        seq_cache_   = seq;

        return w_o_.forward(engine, concat);
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
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq = seq_cache_;
        const std::size_t batch = batch_cache_;
        const std::size_t H_dk = num_heads_ * d_k_;

        auto gc = w_o_.backward(engine, grad_output);
        if (!gc) return gc;
        Tensor gcr;
        if (batch > 1)
        {
            auto g = engine.rearrange_3d(*gc, H_dk, batch, seq, false);
            if (!g) return std::unexpected(g.error());
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
            Matrix bm(1, batch * seq, Scalar{0});
            for (std::size_t i = 0; i < boundary.size(); ++i)
                bm.set_value_unchecked(0, i, static_cast<Scalar>(boundary[i]));
            auto bt = engine.from_matrix(bm);
            if (!bt) return std::unexpected(bt.error());
            boundary_t = std::move(*bt);
            has_bnd = true;
        }
        else
        {
            auto bd = make_dummy_(engine);
            if (!bd) return std::unexpected(bd.error());
            boundary_t = std::move(*bd);
        }
        auto dummy_r = make_dummy_(engine);
        if (!dummy_r) return std::unexpected(dummy_r.error());
        const Tensor& dummy = *dummy_r;

        // ── z-scan：获取 z 和 q·z（backward 需要 z 向量） ────────────
        ensure_ones_(engine, BH, d_k_, seq);
        auto Z_sc = engine.scan_prefix_outer(Kp_cache_, V_ones_cache_, Qp_cache_, V_ones_cache_,
                                             dummy, dummy, false,
                                             d_k_, num_heads_, causal_,
                                             boundary_t, has_bnd);
        if (!Z_sc) return std::unexpected(Z_sc.error());
        auto u_r = engine.slice_rows(*Z_sc, 0, BHdk);         // q·z（标量重复 dk 次）
        if (!u_r) return std::unexpected(u_r.error());
        auto z2_r = engine.slice_rows(*Z_sc, 2 * BHdk, BHdk); // dk·z（向量）
        if (!z2_r) return std::unexpected(z2_r.error());
        auto z_inv_r = engine.elementwise_binary_scalar(BinaryOp::Mul, *z2_r, Scalar{1} / static_cast<Scalar>(d_k_));
        if (!z_inv_r) return std::unexpected(z_inv_r.error()); // z = [2)/dk

        // ── pass 1：主扫描，读出 B^T·g 和 r = g·(B·q) ────────────
        auto Sc = engine.scan_prefix_outer(Kp_cache_, V_re_cache_, Qp_cache_, gcr,
                                           dummy, dummy, false,
                                           d_k_, num_heads_, causal_,
                                           boundary_t, has_bnd);
        if (!Sc) return std::unexpected(Sc.error());
        auto BTR_r = engine.slice_rows(*Sc, 2 * BHdk, BHdk);  // B^T·g
        if (!BTR_r) return std::unexpected(BTR_r.error());
        auto r_r = engine.slice_rows(*Sc, 4 * BHdk, BHdk);    // r = g·(B·q)
        if (!r_r) return std::unexpected(r_r.error());

        // ── 公共中间量 ───────────────────────────────────────────────
        auto den_r = engine.elementwise_binary_scalar(BinaryOp::Add, *u_r, Scalar{1e-4});
        if (!den_r) return std::unexpected(den_r.error());
        auto inv_r = engine.elementwise_binary_scalar(BinaryOp::Div, *den_r, Scalar{1.0}, true);
        if (!inv_r) return std::unexpected(inv_r.error());
        auto gnum_r = engine.elementwise_binary(BinaryOp::Mul, gcr, *inv_r);
        if (!gnum_r) return std::unexpected(gnum_r.error());
        auto neg_r = engine.elementwise_unary(UnaryOp::Neg, *r_r);
        if (!neg_r) return std::unexpected(neg_r.error());
        auto den2_r = engine.elementwise_binary(BinaryOp::Mul, *den_r, *den_r);
        if (!den2_r) return std::unexpected(den2_r.error());
        auto scale_r = engine.elementwise_binary(BinaryOp::Div, *neg_r, *den2_r);
        if (!scale_r) return std::unexpected(scale_r.error()); // -r/den²（标量重复 dk 次）

        // ── gQ = B^T·gnum + scale·z ─────────────────────────────────
        auto btrg_r = engine.elementwise_binary(BinaryOp::Mul, *BTR_r, *inv_r);
        if (!btrg_r) return std::unexpected(btrg_r.error());
        auto gQ_den_r = engine.elementwise_binary(BinaryOp::Mul, *scale_r, *z_inv_r);
        if (!gQ_den_r) return std::unexpected(gQ_den_r.error());
        Tensor gQt;
        {
            auto r = engine.elementwise_binary(BinaryOp::Add, *btrg_r, *gQ_den_r);
            if (!r) return std::unexpected(r.error());
            gQt = std::move(*r);
        }

        // ── gV 和 gK ─────────────────────────────────────────────────
        Tensor gKt, gVt;
        // dB = outer(gnum, q) — dL/dB 矩阵
        auto dB_r = engine.outer_col(*gnum_r, Qp_cache_, dummy, d_k_, false);
        if (!dB_r) return std::unexpected(dB_r.error());

        if (causal_)
        {
            // 因果：S_B = suffix(dB)，gV = S_B·k，gK_B = S_B^T·v
            auto SBc_r = engine.scan_suffix_outer(*dB_r, Kp_cache_, V_re_cache_,
                                                  d_k_, num_heads_, true,
                                                  boundary_t, has_bnd);
            if (!SBc_r) return std::unexpected(SBc_r.error());
            auto gv_r = engine.slice_rows(*SBc_r, 0, BHdk);
            if (!gv_r) return std::unexpected(gv_r.error());
            gVt = std::move(*gv_r);
            auto gK_B_r = engine.slice_rows(*SBc_r, 2 * BHdk, BHdk);
            if (!gK_B_r) return std::unexpected(gK_B_r.error());

            // suffix(scale·q)：D = outer(q, e_0, scale)，suffix(D)·e_0
            auto D_zq_r = engine.outer_col(Qp_cache_, e_0_cache_, *scale_r, d_k_, true);
            if (!D_zq_r) return std::unexpected(D_zq_r.error());
            auto SZ_r = engine.scan_suffix_outer(*D_zq_r, e_0_cache_, e_0_cache_,
                                                 d_k_, num_heads_, true,
                                                 boundary_t, has_bnd);
            if (!SZ_r) return std::unexpected(SZ_r.error());
            auto suffix_sq_r = engine.slice_rows(*SZ_r, 0, BHdk);
            if (!suffix_sq_r) return std::unexpected(suffix_sq_r.error());
            auto gk_r = engine.elementwise_binary(BinaryOp::Add, *gK_B_r, *suffix_sq_r);
            if (!gk_r) return std::unexpected(gk_r.error());
            gKt = std::move(*gk_r);
        }
        else
        {
            // 双向：A/B 为全集常数 → dB 在所有位置广播相同值
            auto dB_sum_r = engine.row_reduce_sum(*dB_r);
            if (!dB_sum_r) return std::unexpected(dB_sum_r.error());
            Tensor Bb = engine.create_tensor(BHdk * d_k_, seq);
            { auto r = engine.zero(Bb); if (!r) return std::unexpected(r.error()); }
            { auto r = engine.broadcast_row_inplace(Bb, *dB_sum_r, BinaryOp::Add);
              if (!r) return std::unexpected(r.error()); }
            auto SBc_r = engine.scan_suffix_outer(Bb, Kp_cache_, V_re_cache_,
                                                  d_k_, num_heads_, false,
                                                  dummy, false);
            if (!SBc_r) return std::unexpected(SBc_r.error());
            auto gv_r = engine.slice_rows(*SBc_r, 0, BHdk);
            if (!gv_r) return std::unexpected(gv_r.error());
            gVt = std::move(*gv_r);
            auto gK_B_r = engine.slice_rows(*SBc_r, 2 * BHdk, BHdk);
            if (!gK_B_r) return std::unexpected(gK_B_r.error());

            // 双向 gK 的常数项：Σ_t scale_t · q_t（全集求和后广播）
            auto scale_q_r = engine.elementwise_binary(BinaryOp::Mul, *scale_r, Qp_cache_);
            if (!scale_q_r) return std::unexpected(scale_q_r.error());
            auto tsq_r = engine.row_reduce_sum(*scale_q_r);
            if (!tsq_r) return std::unexpected(tsq_r.error());
            Tensor gK_den = engine.create_tensor(BHdk, seq);
            { auto r = engine.zero(gK_den); if (!r) return std::unexpected(r.error()); }
            { auto r = engine.broadcast_row_inplace(gK_den, *tsq_r, BinaryOp::Add);
              if (!r) return std::unexpected(r.error()); }
            auto gk_r = engine.elementwise_binary(BinaryOp::Add, *gK_B_r, gK_den);
            if (!gk_r) return std::unexpected(gk_r.error());
            gKt = std::move(*gk_r);
        }

        // ── ReLU 反向：(x>0) ? g : 0 ───────────────────────────────
        auto gq_relu = dsl::compute(engine,
            dsl::select(dsl::leaf(Qp_cache_) > Scalar{0},
                        dsl::leaf(gQt), Scalar{0}),
            gQt.rows(), gQt.cols());
        if (!gq_relu) return std::unexpected(gq_relu.error());
        auto gk_relu = dsl::compute(engine,
            dsl::select(dsl::leaf(Kp_cache_) > Scalar{0},
                        dsl::leaf(gKt), Scalar{0}),
            gKt.rows(), gKt.cols());
        if (!gk_relu) return std::unexpected(gk_relu.error());

        // ── RoPE 反向（旋转正交，逆 = 反角） ────────────────────────
        if (use_rope_)
        {
            auto gq = rope_.apply(engine, *gq_relu, seq, true);
            if (!gq) return std::unexpected(gq.error());
            gq_relu = std::move(*gq);
            auto gk = rope_.apply(engine, *gk_relu, seq, true);
            if (!gk) return std::unexpected(gk.error());
            gk_relu = std::move(*gk);
        }

        // ── RLA-2 RMSNorm 反向（RoPE 反向之后、rearrange 之前） ──────
        {
            const std::size_t BHrms = batch * num_heads_;
            auto gq_rn = rms_norm_backward_(engine, *gq_relu, Q_normed_cache_,
                                            Q_rms_inv_cache_, BHrms, d_k_, seq);
            if (!gq_rn) return std::unexpected(gq_rn.error());
            gq_relu = std::move(*gq_rn);
            auto gk_rn = rms_norm_backward_(engine, *gk_relu, K_normed_cache_,
                                            K_rms_inv_cache_, BHrms, d_k_, seq);
            if (!gk_rn) return std::unexpected(gk_rn.error());
            gk_relu = std::move(*gk_rn);
        }

        Tensor gq_r, gk_r, gv_r;
        if (batch > 1)
        {
            auto a = engine.rearrange_3d(*gq_relu, H_dk, batch, seq, true); if (!a) return std::unexpected(a.error());
            gq_r = std::move(*a);
            auto b = engine.rearrange_3d(*gk_relu, H_dk, batch, seq, true); if (!b) return std::unexpected(b.error());
            gk_r = std::move(*b);
            auto c = engine.rearrange_3d(gVt, H_dk, batch, seq, true); if (!c) return std::unexpected(c.error());
            gv_r = std::move(*c);
        }
        else
        {
            gq_r = std::move(*gq_relu);
            gk_r = std::move(*gk_relu);
            gv_r = std::move(gVt);
        }

        auto giq = w_q_.backward(engine, gq_r);
        if (!giq) return giq;
        Tensor grad_input = std::move(*giq);
        auto gik = w_k_.backward(engine, gk_r);
        if (!gik) return gik;
        { auto r = engine.add_inplace(grad_input, *gik); if (!r) return std::unexpected(r.error()); }
        auto giv = w_v_.backward(engine, gv_r);
        if (!giv) return giv;
        { auto r = engine.add_inplace(grad_input, *giv); if (!r) return std::unexpected(r.error()); }
        return grad_input;
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
        auto q_res = w_q_.forward(engine, input);   // (d_model, 1) = (H*dk, 1)
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;
        Tensor Q = std::move(*q_res), K = std::move(*k_res), V = std::move(*v_res);
        // RLA-2：RMSNorm on Q and K（per-head, dk blocks）
        {
            Tensor dummy_ri;
            auto rq = rms_norm_forward_(engine, Q, num_heads_, d_k_, 1, dummy_ri);
            if (!rq) return std::unexpected(rq.error());
            Q = std::move(*rq);
            auto rk = rms_norm_forward_(engine, K, num_heads_, d_k_, 1, dummy_ri);
            if (!rk) return std::unexpected(rk.error());
            K = std::move(*rk);
        }
        if (use_rope_)
        {
            auto qr = rope_.apply_step(engine, Q, pos, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply_step(engine, K, pos, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = dsl::compute(engine,
            dsl::max(dsl::leaf(Q), Scalar{0}), Q.rows(), Q.cols());
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = dsl::compute(engine,
            dsl::max(dsl::leaf(K), Scalar{0}), K.rows(), K.cols());
        if (!Kp) return std::unexpected(Kp.error());
        const std::size_t H = num_heads_;
        const std::size_t dk = d_k_;

        // 更新运行态：B += k' ⊗ v，z += k'
        auto B_add_r = engine.batched_matmul(V, *Kp, H, false, true);  // v·k'^T → (H*dk, dk)
        if (!B_add_r) return std::unexpected(B_add_r.error());
        { auto r = engine.add_inplace(B_state, *B_add_r);
          if (!r) return std::unexpected(r.error()); }
        { auto r = engine.add_inplace(z_state, *Kp);  // z += k'（(H*dk, 1)）
          if (!r) return std::unexpected(r.error()); }

        // num = B·q'：batched_matmul(B_state, Qp, H) → (H*dk, 1)
        auto num_r = engine.batched_matmul(B_state, *Qp, H, false, false);
        if (!num_r) return std::unexpected(num_r.error());

        // den = q'·z + ε：逐头标量点积（通过 to_matrix 在 CPU 上计算）
        // forward_step 是逐 token 串行的，CPU round-trip 可接受。
        auto q_mat = engine.to_matrix(*Qp);
        if (!q_mat) return std::unexpected(q_mat.error());
        auto z_mat = engine.to_matrix(z_state);
        if (!z_mat) return std::unexpected(z_mat.error());
        Matrix den_mat(H, 1);
        for (std::size_t h = 0; h < H; ++h)
        {
            Scalar dot = 0;
            for (std::size_t j = 0; j < dk; ++j)
                dot += q_mat->at_unchecked(h * dk + j, 0) *
                       z_mat->at_unchecked(h * dk + j, 0);
            den_mat.set_value_unchecked(h, 0, dot + Scalar{1e-4});
        }
        // 广播到 (H*dk, 1)：每头标量重复 dk 次
        Matrix den_full(H * dk, 1);
        for (std::size_t h = 0; h < H; ++h)
        {
            const Scalar d = den_mat.at_unchecked(h, 0);
            for (std::size_t j = 0; j < dk; ++j)
                den_full.set_value_unchecked(h * dk + j, 0, d);
        }
        auto den_t = engine.from_matrix(den_full);
        if (!den_t) return std::unexpected(den_t.error());

        auto out_r = engine.elementwise_binary(BinaryOp::Div, *num_r, *den_t);
        if (!out_r) return std::unexpected(out_r.error());
        return w_o_.forward(engine, *out_r);
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

public:
    RAPTBlock(std::size_t d_model, std::size_t num_heads, std::size_t d_ff,
              std::size_t seq_len, PosEncodingType pos_enc,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true)
        : norm1_(make_norm_layer(d_model, norm_type)),
          attn_(d_model, num_heads, seq_len, causal, pos_enc),
          norm2_(make_norm_layer(d_model, norm_type)),
          ff_(d_model, d_ff, activation)
    {
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        auto r1 = norm1_->init(engine); if (!r1) return std::unexpected(r1.error());
        auto r2 = attn_.init(engine);   if (!r2) return std::unexpected(r2.error());
        auto r3 = norm2_->init(engine); if (!r3) return std::unexpected(r3.error());
        auto r4 = ff_.init(engine);     if (!r4) return std::unexpected(r4.error());
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        auto n1 = norm1_->parameters(); p.insert(p.end(), n1.begin(), n1.end());
        auto a = attn_.parameters();    p.insert(p.end(), a.begin(), a.end());
        auto n2 = norm2_->parameters(); p.insert(p.end(), n2.begin(), n2.end());
        auto f = ff_.parameters();      p.insert(p.end(), f.begin(), f.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        auto n1 = norm1_->param_gradients(); g.insert(g.end(), n1.begin(), n1.end());
        auto a = attn_.param_gradients();    g.insert(g.end(), a.begin(), a.end());
        auto n2 = norm2_->param_gradients(); g.insert(g.end(), n2.begin(), n2.end());
        auto f = ff_.param_gradients();      g.insert(g.end(), f.begin(), f.end());
        return g;
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
        ComputeEngine& engine, const Tensor& input) override
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;
        auto a = attn_.forward(engine, *n1);
        if (!a) return a;
        auto r1 = dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(*a),
            input.rows(), input.cols());
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
        return dsl::compute(engine,
            dsl::leaf(*r1) + dsl::leaf(*f),
            r1->rows(), r1->cols());
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;
        auto grad_r1 = dsl::compute(engine,
            dsl::leaf(grad_output) + dsl::leaf(*b_n2),
            grad_output.rows(), grad_output.cols());
        if (!grad_r1) return std::unexpected(grad_r1.error());
        auto grad_a = attn_.backward(engine, *grad_r1);
        if (!grad_a) return grad_a;
        auto b_n1 = norm1_->backward(engine, *grad_a);
        if (!b_n1) return b_n1;
        return dsl::compute(engine,
            dsl::leaf(*grad_r1) + dsl::leaf(*b_n1),
            grad_r1->rows(), grad_r1->cols());
    }

    // 增量推理：单 token → norm1 → RLA-2 运行态注意力 → 残差 → norm2 → FFN → 残差
    // B_state/z_state: RLA-2 运行态 KV cache（见 ReLULinearAttention::forward_step）
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Tensor& B_state, Tensor& z_state, std::size_t pos)
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;
        auto a = attn_.forward_step(engine, *n1, B_state, z_state, pos);
        if (!a) return a;
        auto r1 = dsl::compute(engine,
            dsl::leaf(input) + dsl::leaf(*a),
            input.rows(), input.cols());
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
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

public:
    RAPTModel(std::size_t vocab_size, std::size_t d_model, std::size_t seq_len,
              std::size_t num_heads, std::size_t d_ff, std::size_t num_layers,
              PosEncodingType pos_enc = PosEncodingType::RoPE,
              ActivationType activation = ActivationType::GeLU,
              NormType norm_type = NormType::LayerNorm,
              bool causal = true)
        : vocab_size_(vocab_size), d_model_(d_model), seq_len_(seq_len),
          num_heads_(num_heads),
          ln_f_(make_norm_layer(d_model, norm_type)),
          lm_head_(d_model, vocab_size)
    {
        blocks_.reserve(num_layers);
        for (std::size_t i = 0; i < num_layers; ++i)
            blocks_.emplace_back(d_model, num_heads, d_ff, seq_len, pos_enc,
                                 activation, norm_type, causal);
        // RLA-2 强约束：必须 RoPE（或 ALiBi）。v1 强制 RoPE，输入侧无位置嵌入。
        pos_encoder_ = std::make_unique<NoPositionEncoder>();
    }

    [[nodiscard]] Result<void> init(ComputeEngine& engine) override
    {
        Matrix te(vocab_size_, d_model_);
        constexpr Scalar emb_init_std = 0.02;
        std::mt19937_64 rng{42};
        std::normal_distribution<Scalar> dist(0.0, emb_init_std);
        auto te_s = te.span();
        for (std::size_t i = 0; i < te.size(); ++i) te_s[i] = dist(rng);
        auto te_r = engine.from_matrix(te);
        if (!te_r) return std::unexpected(te_r.error());
        token_emb_ = std::move(*te_r);

        grad_token_emb_ = engine.create_tensor(vocab_size_, d_model_);
        { auto r = engine.zero(grad_token_emb_); if (!r) return std::unexpected(r.error()); }

        if (pos_encoder_)
        {
            auto r = pos_encoder_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        for (auto& b : blocks_)
        {
            auto r = b.init(engine);
            if (!r) return std::unexpected(r.error());
        }
        if (ln_f_)
        {
            auto r = ln_f_->init(engine);
            if (!r) return std::unexpected(r.error());
        }
        { auto r = lm_head_.init(engine); if (!r) return std::unexpected(r.error()); }
        return {};
    }

    std::vector<TensorRef> parameters() override
    {
        std::vector<TensorRef> p;
        p.push_back(token_emb_);
        auto pp = pos_encoder_->parameters();
        p.insert(p.end(), pp.begin(), pp.end());
        for (auto& b : blocks_)
        {
            auto bp = b.parameters();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        auto lp = ln_f_->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
        auto hp = lm_head_.parameters();
        p.insert(p.end(), hp.begin(), hp.end());
        return p;
    }
    std::vector<TensorRef> param_gradients() override
    {
        std::vector<TensorRef> g;
        g.push_back(grad_token_emb_);
        auto gp = pos_encoder_->param_gradients();
        g.insert(g.end(), gp.begin(), gp.end());
        for (auto& b : blocks_)
        {
            auto bg = b.param_gradients();
            g.insert(g.end(), bg.begin(), bg.end());
        }
        auto lg = ln_f_->param_gradients();
        g.insert(g.end(), lg.begin(), lg.end());
        auto hg = lm_head_.param_gradients();
        g.insert(g.end(), hg.begin(), hg.end());
        return g;
    }

    void set_checkpoint_mode(bool enabled) override
    {
        Layer::set_checkpoint_mode(enabled);
        for (auto& b : blocks_) b.set_checkpoint_mode(enabled);
        ln_f_->set_checkpoint_mode(enabled);
        lm_head_.set_checkpoint_mode(enabled);
    }

    void clear_cache() override
    {
        token_emb_ = Tensor{};
        grad_token_emb_ = Tensor{};
        for (auto& b : blocks_) b.clear_cache();
        ln_f_->clear_cache();
        lm_head_.clear_cache();
        stored_tokens_tensor_ = Tensor{};
        batch_size_ = 0;
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

    [[nodiscard]] Result<Tensor> forward(
        ComputeEngine& engine, const Tensor& input) override
    {
        const std::size_t seq = input.rows();
        const std::size_t batch = input.cols();
        batch_size_ = batch;

        auto input_T = engine.transpose(input);   // (batch, seq)
        if (!input_T) return std::unexpected(input_T.error());
        auto all_emb = engine.gather_rows(token_emb_, *input_T);
        if (!all_emb) return std::unexpected(all_emb.error());
        auto st = engine.clone(*input_T);
        if (!st) return std::unexpected(st.error());
        stored_tokens_tensor_ = std::move(*st);
        auto all_T = engine.transpose(*all_emb);
        if (!all_T) return std::unexpected(all_T.error());

        auto x_res = pos_encoder_->apply(engine, *all_T, batch, seq);
        if (!x_res) return std::unexpected(x_res.error());
        Tensor x = std::move(*x_res);

        for (auto& b : blocks_)
        {
            if (!doc_ids_.empty()) b.set_doc_ids(doc_ids_);
            auto r = b.forward(engine, x);
            if (!r) return r;
            x = std::move(*r);
        }
        auto ln = ln_f_->forward(engine, x);
        if (!ln) return ln;
        return lm_head_.forward(engine, *ln);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        const std::size_t seq = seq_len_;
        const std::size_t batch = batch_size_;

        auto b_lm = lm_head_.backward(engine, grad_output);
        if (!b_lm) return b_lm;
        auto b_ln = ln_f_->backward(engine, *b_lm);
        if (!b_ln) return b_ln;
        Tensor grad_x = std::move(*b_ln);

        for (std::size_t i = blocks_.size(); i-- > 0;)
        {
            auto br = blocks_[i].backward(engine, grad_x);
            if (!br) return br;
            grad_x = std::move(*br);
        }

        auto grad_T = engine.transpose(grad_x);
        if (!grad_T) return std::unexpected(grad_T.error());
        auto pr = pos_encoder_->backward(engine, *grad_T, batch, seq);
        if (!pr) return std::unexpected(pr.error());
        auto sr = engine.scatter_add_rows(grad_token_emb_, stored_tokens_tensor_, *grad_T);
        if (!sr) return std::unexpected(sr.error());

        Matrix grad_input(seq, batch, Scalar{0});
        return engine.from_matrix(grad_input);
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
            return std::unexpected(Error{"RAPT generate: empty prompt"});
        const std::size_t dk = d_model_ / num_heads_;
        // RLA-2 每块两个运行态：B_state (d_model, d_k) + z_state (d_model, 1)
        std::vector<Tensor> statesB, statesZ;
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            statesB.emplace_back(engine.create_tensor(d_model_, dk));
            { auto r = engine.zero(statesB.back()); if (!r) return std::unexpected(r.error()); }
            statesZ.emplace_back(engine.create_tensor(d_model_, 1));
            { auto r = engine.zero(statesZ.back()); if (!r) return std::unexpected(r.error()); }
        }

        // 处理单个 token：embed → 各块 forward_step → LN → LM head → 返回 logits 列
        auto step_one = [&](std::size_t tok, std::size_t pos)
            -> Result<std::vector<Scalar>>
        {
            Matrix idx(1, 1);
            idx.set_value_unchecked(0, 0, static_cast<Scalar>(tok));
            auto idx_t = engine.from_matrix(idx);
            if (!idx_t) return std::unexpected(idx_t.error());
            auto emb = engine.gather_rows(token_emb_, *idx_t);   // (1, d_model)
            if (!emb) return std::unexpected(emb.error());
            auto emb_t = engine.transpose(*emb);                  // (d_model, 1)
            if (!emb_t) return std::unexpected(emb_t.error());
            Tensor h = std::move(*emb_t);
            for (std::size_t i = 0; i < blocks_.size(); ++i)
            {
                auto r = blocks_[i].forward_step(engine, h, statesB[i], statesZ[i], pos);
                if (!r) return std::unexpected(r.error());
                h = std::move(*r);
            }
            auto ln = ln_f_->forward(engine, h);
            if (!ln) return std::unexpected(ln.error());
            auto logits = lm_head_.forward(engine, *ln);
            if (!logits) return std::unexpected(logits.error());
            auto lm = engine.to_matrix(*logits);
            if (!lm) return std::unexpected(lm.error());
            std::vector<Scalar> last(vocab_size_);
            for (std::size_t v = 0; v < vocab_size_; ++v)
                last[v] = lm->at_unchecked(v, 0);
            return last;
        };

        // 逐 token 处理 prompt（建立运行态）；保留最后一步 logits（预测下一 token）
        std::size_t pos = 0;
        std::vector<Scalar> last;
        for (std::size_t i = 0; i < prompt.size(); ++i)
        {
            auto r = step_one(prompt[i], pos++);
            if (!r) return std::unexpected(r.error());
            last = std::move(*r);
        }

        // 采样生成
        std::vector<std::size_t> generated;
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
        for (std::size_t step = 0; step < max_new_tokens; ++step)
        {
            std::vector<Scalar> lastv = last;
            if (temperature > 0.0 && temperature != 1.0)
                for (auto& x : lastv) x /= temperature;

            Scalar max_val = lastv[0];
            for (std::size_t v = 1; v < vocab_size_; ++v)
                max_val = std::max(max_val, lastv[v]);
            Scalar sum_exp = 0;
            for (auto& x : lastv) { x = std::exp(x - max_val); sum_exp += x; }
            for (auto& x : lastv) x /= sum_exp;

            std::size_t next;
            if (temperature > 0.0)
            {
                Scalar r = dist(rng);
                Scalar cum = 0;
                next = vocab_size_ - 1;
                for (std::size_t v = 0; v < vocab_size_; ++v)
                {
                    cum += lastv[v];
                    if (r <= cum) { next = v; break; }
                }
            }
            else
            {
                next = 0;
                Scalar best = lastv[0];
                for (std::size_t v = 1; v < vocab_size_; ++v)
                    if (lastv[v] > best) { best = lastv[v]; next = v; }
            }

            if (step >= min_new_tokens && next == eos_token_id) break;
            generated.push_back(next);

            auto r = step_one(next, pos++);
            if (!r) return std::unexpected(r.error());
            last = std::move(*r);
        }
        return generated;
    }
};

} // namespace nn
