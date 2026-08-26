#ifndef NN_COMPUTE_LAYER_RAPT_HPP
#define NN_COMPUTE_LAYER_RAPT_HPP

#include "compute_layer_base.hpp"
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
class ReLULinearAttention final : public Layer
{
private:
    std::size_t d_model_;
    std::size_t num_heads_;
    std::size_t d_k_;
    std::size_t seq_len_;    // 单样本序列长度（0 = 单样本，cols 即 seq）
    bool causal_;            // true=因果前缀和；false=全量双向
    bool use_rope_;          // RoPE 施加在 Q/K 进 ReLU 之前（RLA 强约束）
    RotaryEmbedding rope_;
    Linear w_q_, w_k_, w_v_, w_o_;

    // forward 缓存（backward 用）
    Tensor Qp_cache_;        // (batch*H*d_k, seq) ReLU(RoPE(Q))
    Tensor Kp_cache_;        // (batch*H*d_k, seq) ReLU(RoPE(K))
    Tensor V_re_cache_;      // (batch*H*d_k, seq) V（不 ReLU、不 RoPE）
    std::size_t batch_cache_ = 0;
    std::size_t seq_cache_   = 0;

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

    // ── 标量线性代数助手（行主序向量化，dk×dk 矩阵） ──────────────────
    static void matvec_(const std::vector<Scalar>& M,
                        const std::vector<Scalar>& x,
                        std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t i = 0; i < dk; ++i)
        {
            Scalar acc{0};
            for (std::size_t j = 0; j < dk; ++j) acc += M[i * dk + j] * x[j];
            y[i] = acc;
        }
    }
    static void matvec_T_(const std::vector<Scalar>& M,
                          const std::vector<Scalar>& x,
                          std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t j = 0; j < dk; ++j)
        {
            Scalar acc{0};
            for (std::size_t i = 0; i < dk; ++i) acc += M[i * dk + j] * x[i];
            y[j] = acc;
        }
    }
    static void add_outer_(std::vector<Scalar>& M,
                           const std::vector<Scalar>& x,
                           const std::vector<Scalar>& y, std::size_t dk)
    {
        for (std::size_t i = 0; i < dk; ++i)
            for (std::size_t j = 0; j < dk; ++j)
                M[i * dk + j] += x[i] * y[j];
    }

    // 构造 RLA 掩码 (BH*seq, seq) 0/1 矩阵：mask[b*H+h][t, j] = 1 iff 允许 q_t 看 k_j
    //   causal:       j <= t
    //   文档感知:     j <= t 且 doc[j] == doc[t]（仅 causal 模式生效）
    //   bidirectional: 全 1（无掩码）
    [[nodiscard]] Result<Tensor> build_rla_mask_(
        ComputeEngine& engine, std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        Matrix mask(BH * seq, seq, Scalar{1});
        if (causal_)
        {
            for (std::size_t bh = 0; bh < BH; ++bh)
            {
                const std::size_t b = bh / num_heads_;
                for (std::size_t t = 0; t < seq; ++t)
                    for (std::size_t j = 0; j < seq; ++j)
                    {
                        bool ok = (j <= t);
                        if (ok && has_doc_ids_ &&
                            doc_ids_[b * seq + j] != doc_ids_[b * seq + t])
                            ok = false;
                        if (!ok) mask.set_value_unchecked(bh * seq + t, j, Scalar{0});
                    }
            }
        }
        return engine.from_matrix(mask);
    }

    // ── 原语组合版 RLA 前向（GPU 上运行，消除 PCIe 往返） ──────────────
    // 用现有引擎原语（batched_matmul / elementwise / reduce / broadcast）
    // 组合出 RLA 计算，不落 CPU。代价：物化 (BH*seq, seq) 分数矩阵（O(L²)），
    // 作为过渡实现；后续可被 IR 张量复用优化改进为 O(L·d²) 运行态。
    //
    // 输入 Qp/Kp/V: (BH*dk, seq)；返回 (BH*dk, seq)
    //   S[i,j] = q_i·k_j  (非负，Qp/Kp 经 ReLU)
    //   num = S·Vᵀ；denom = sqrt(Σ_j S² + eps)；out = num/denom
    [[nodiscard]] Result<Tensor> forward_matmul(
        ComputeEngine& engine, const Tensor& Qp, const Tensor& Kp,
        const Tensor& V, std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        const std::size_t dk = d_k_;
        auto S = engine.batched_matmul(Qp, Kp, BH, true, false, Scalar{1});   // (BH*seq, seq)
        if (!S) return std::unexpected(S.error());
        auto mask = build_rla_mask_(engine, batch, seq);                      // (BH*seq, seq)
        if (!mask) return std::unexpected(mask.error());
        auto Sm = engine.elementwise_binary(BinaryOp::Mul, *S, *mask);
        if (!Sm) return std::unexpected(Sm.error());
        auto num = engine.batched_matmul(*Sm, V, BH, false, true);            // Sm @ Vᵀ → (BH*seq, dk)
        if (!num) return std::unexpected(num.error());
        auto S2 = engine.elementwise_binary(BinaryOp::Mul, *Sm, *Sm);
        if (!S2) return std::unexpected(S2.error());
        auto denom2 = engine.row_reduce_sum(*S2);                             // (BH*seq, 1)
        if (!denom2) return std::unexpected(denom2.error());
        auto denom_e = engine.elementwise_binary_scalar(BinaryOp::Add, *denom2, Scalar{1e-6}, false);
        if (!denom_e) return std::unexpected(denom_e.error());
        auto denom = engine.elementwise_unary(UnaryOp::Sqrt, *denom_e);
        if (!denom) return std::unexpected(denom.error());
        auto out = engine.clone(*num);                                        // (BH*seq, dk)
        if (!out) return std::unexpected(out.error());
        { auto r = engine.broadcast_row_inplace(*out, *denom, BinaryOp::Div);
          if (!r) return std::unexpected(r.error()); }
        auto O_T = engine.transpose(*out);                                    // (dk, BH*seq)
        if (!O_T) return std::unexpected(O_T.error());
        return engine.rearrange_3d(*O_T, dk, BH, seq, false);                 // (BH*dk, seq)
    }

    // ── 原语组合版 RLA 反向 ──────────────────────────────────────────
    // 返回 {grad_Qp, grad_Kp, grad_V}，均 (BH*dk, seq)。与 forward_matmul 对应。
    [[nodiscard]] Result<std::array<Tensor, 3>> backward_matmul(
        ComputeEngine& engine,
        const Tensor& Qp, const Tensor& Kp, const Tensor& V,
        const Tensor& grad_out_re,   // (BH*dk, seq)
        std::size_t batch, std::size_t seq)
    {
        const std::size_t BH = batch * num_heads_;
        // ── 重算前向量 ──
        auto S = engine.batched_matmul(Qp, Kp, BH, true, false, Scalar{1});
        if (!S) return std::unexpected(S.error());
        auto mask = build_rla_mask_(engine, batch, seq);
        if (!mask) return std::unexpected(mask.error());
        auto Sm = engine.elementwise_binary(BinaryOp::Mul, *S, *mask);
        if (!Sm) return std::unexpected(Sm.error());
        auto num = engine.batched_matmul(*Sm, V, BH, false, true);            // Sm @ Vᵀ → (BH*seq, dk)
        if (!num) return std::unexpected(num.error());
        auto S2 = engine.elementwise_binary(BinaryOp::Mul, *Sm, *Sm);
        if (!S2) return std::unexpected(S2.error());
        auto denom2 = engine.row_reduce_sum(*S2);
        if (!denom2) return std::unexpected(denom2.error());
        auto denom_e = engine.elementwise_binary_scalar(BinaryOp::Add, *denom2, Scalar{1e-6}, false);
        if (!denom_e) return std::unexpected(denom_e.error());
        auto denom = engine.elementwise_unary(UnaryOp::Sqrt, *denom_e);
        if (!denom) return std::unexpected(denom.error());

        // ── dO_t: (BH*seq, dk) ──
        auto dO_T = engine.transpose(grad_out_re);
        if (!dO_T) return std::unexpected(dO_T.error());
        auto dO_t = engine.rearrange_3d(*dO_T, seq, BH, d_k_, false);
        if (!dO_t) return std::unexpected(dO_t.error());

        // dnum = dO / denom（按行广播）
        auto dnum = engine.clone(*dO_t);
        if (!dnum) return std::unexpected(dnum.error());
        { auto r = engine.broadcast_row_inplace(*dnum, *denom, BinaryOp::Div);
          if (!r) return std::unexpected(r.error()); }

        // ddenom = -rowsum(dO·num) / denom²
        auto prod = engine.elementwise_binary(BinaryOp::Mul, *dO_t, *num);
        if (!prod) return std::unexpected(prod.error());
        auto dot = engine.row_reduce_sum(*prod);
        if (!dot) return std::unexpected(dot.error());
        auto negdot = engine.elementwise_unary(UnaryOp::Neg, *dot);
        if (!negdot) return std::unexpected(negdot.error());
        auto denom_sq = engine.elementwise_binary(BinaryOp::Mul, *denom, *denom);
        if (!denom_sq) return std::unexpected(denom_sq.error());
        auto ddenom = engine.elementwise_binary(BinaryOp::Div, *negdot, *denom_sq);
        if (!ddenom) return std::unexpected(ddenom.error());
        // ddenom2 = ddenom / (2·denom)
        auto two_denom = engine.elementwise_binary_scalar(BinaryOp::Mul, *denom, Scalar{2}, false);
        if (!two_denom) return std::unexpected(two_denom.error());
        auto ddenom2 = engine.elementwise_binary(BinaryOp::Div, *ddenom, *two_denom);
        if (!ddenom2) return std::unexpected(ddenom2.error());

        // dSm = dSm_num + dSm_den
        //   dSm_num = dnum @ V；dSm_den = 2·Sm·ddenom2（按行广播）
        auto dSm_num = engine.batched_matmul(*dnum, V, BH, false, false);
        if (!dSm_num) return std::unexpected(dSm_num.error());
        auto dSm_den = engine.clone(*Sm);
        if (!dSm_den) return std::unexpected(dSm_den.error());
        { auto r = engine.broadcast_row_inplace(*dSm_den, *ddenom2, BinaryOp::Mul);
          if (!r) return std::unexpected(r.error()); }
        dSm_den = engine.elementwise_binary_scalar(BinaryOp::Mul, *dSm_den, Scalar{2}, false);
        if (!dSm_den) return std::unexpected(dSm_den.error());
        auto dSm = engine.elementwise_binary(BinaryOp::Add, *dSm_num, *dSm_den);
        if (!dSm) return std::unexpected(dSm.error());
        // dS = dSm · mask（掩码是常数 0/1 乘法）
        auto dS = engine.elementwise_binary(BinaryOp::Mul, *dSm, *mask);
        if (!dS) return std::unexpected(dS.error());

        // ── 梯度 Q/K/V ──
        auto grad_Qp = engine.batched_matmul(Kp, *dS, BH, false, true);    // Kp @ dSᵀ (BH*dk, seq)
        if (!grad_Qp) return std::unexpected(grad_Qp.error());
        auto grad_Kp = engine.batched_matmul(Qp, *dS, BH, false, false);   // Qp @ dS  (BH*dk, seq)
        if (!grad_Kp) return std::unexpected(grad_Kp.error());
        auto grad_V = engine.batched_matmul(*dnum, *Sm, BH, true, false);  // dnumᵀ @ Sm → (BH*dk, seq)
        if (!grad_V) return std::unexpected(grad_V.error());

        return std::array<Tensor, 3>{std::move(*grad_Qp), std::move(*grad_Kp), std::move(*grad_V)};
    }

    // ── 前向扫描：out[bh*dk+j][t] = RLA 输出 ──────────────────────────
    // boundary（可选，空=无文档感知）：size = batch*seq，boundary[b*seq+t]=1
    // 表示位置 t 是文档起点。causal 模式下文档边界处重置运行态 A/B，
    // 使每个 token 只聚合本文档内前缀（文档感知 = 运行态重置于边界）。
    static void scan_forward_(const Matrix& Q, const Matrix& K, const Matrix& V,
                              Matrix& out, std::size_t dk, std::size_t BH,
                              std::size_t seq, bool causal,
                              std::size_t num_heads,
                              const std::vector<uint8_t>& boundary,
                              Scalar eps)
    {
        std::vector<Scalar> A(dk * dk), B(dk * dk), qv(dk), kv(dk), vv(dk),
                            num(dk), Aq(dk);
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            const std::size_t r0 = bh * dk;
            const std::size_t batch = bh / num_heads;
            const auto doc_reset = [&](std::size_t t) {
                return !boundary.empty() && boundary[batch * seq + t] != 0;
            };
            std::fill(A.begin(), A.end(), Scalar{0});
            std::fill(B.begin(), B.end(), Scalar{0});
            if (!causal)  // 双向：先求全集 A, B
            {
                for (std::size_t t = 0; t < seq; ++t)
                {
                    for (std::size_t j = 0; j < dk; ++j)
                    {
                        kv[j] = K.at_unchecked(r0 + j, t);
                        vv[j] = V.at_unchecked(r0 + j, t);
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
            }
            for (std::size_t t = 0; t < seq; ++t)
            {
                for (std::size_t j = 0; j < dk; ++j)
                {
                    qv[j] = Q.at_unchecked(r0 + j, t);
                    kv[j] = K.at_unchecked(r0 + j, t);
                    vv[j] = V.at_unchecked(r0 + j, t);
                }
                if (causal)  // 前缀和含自身 i<=t；文档边界处重置
                {
                    if (doc_reset(t))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
                matvec_(B, qv, num, dk);
                matvec_(A, qv, Aq, dk);
                Scalar s{0};
                for (std::size_t j = 0; j < dk; ++j) s += qv[j] * Aq[j];
                const Scalar denom = std::sqrt(s + eps);
                for (std::size_t j = 0; j < dk; ++j)
                    out.set_value_unchecked(r0 + j, t, num[j] / denom);
            }
        }
    }

    // ── 反向扫描：G 为 grad_out_re，写 gQ/gK/gV（均 (BH*dk, seq)） ────
    // 前向重算运行态；dL/dA_t=dL/ds_t·q_tq_t^T，dL/dB_t=outer(dL/dnum_t,q_t)；
    // 后缀和得 dL/dk_i = 2·SA·k_i + SB^T·v_i，dL/dv_i = SB·k_i。
    static void scan_backward_(const Matrix& Q, const Matrix& K, const Matrix& V,
                               const Matrix& G, Matrix& gQ, Matrix& gK, Matrix& gV,
                               std::size_t dk, std::size_t BH, std::size_t seq,
                               bool causal, std::size_t num_heads,
                               const std::vector<uint8_t>& boundary, Scalar eps)
    {
        std::vector<Scalar> A(dk * dk), B(dk * dk), qv(dk), kv(dk), vv(dk),
                            num(dk), Aq(dk), gnum(dk), tmp(dk), tmp2(dk);
        for (std::size_t bh = 0; bh < BH; ++bh)
        {
            const std::size_t r0 = bh * dk;
            const std::size_t batch = bh / num_heads;
            const auto doc_reset = [&](std::size_t t) {
                return !boundary.empty() && boundary[batch * seq + t] != 0;
            };
            std::vector<Scalar> store_q(seq * dk), store_dq(seq * dk),
                                store_dA(seq * dk * dk), store_dB(seq * dk * dk);
            std::vector<Scalar> dA_total(dk * dk, Scalar{0}),
                                dB_total(dk * dk, Scalar{0});
            std::fill(A.begin(), A.end(), Scalar{0});
            std::fill(B.begin(), B.end(), Scalar{0});
            if (!causal)
            {
                for (std::size_t t = 0; t < seq; ++t)
                {
                    for (std::size_t j = 0; j < dk; ++j)
                    {
                        kv[j] = K.at_unchecked(r0 + j, t);
                        vv[j] = V.at_unchecked(r0 + j, t);
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
            }
            // Pass 1：重算运行态，存每位置 q_t、dL/dq_t、dL/dA_t、dL/dB_t
            for (std::size_t t = 0; t < seq; ++t)
            {
                for (std::size_t j = 0; j < dk; ++j)
                {
                    qv[j] = Q.at_unchecked(r0 + j, t);
                    kv[j] = K.at_unchecked(r0 + j, t);
                    vv[j] = V.at_unchecked(r0 + j, t);
                }
                if (causal)
                {
                    if (doc_reset(t))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    add_outer_(A, kv, kv, dk);
                    add_outer_(B, vv, kv, dk);
                }
                matvec_(B, qv, num, dk);
                matvec_(A, qv, Aq, dk);
                Scalar s{0};
                for (std::size_t j = 0; j < dk; ++j) s += qv[j] * Aq[j];
                const Scalar denom = std::sqrt(s + eps);
                const Scalar denom_inv = Scalar{1} / denom;
                Scalar gdot{0};
                for (std::size_t j = 0; j < dk; ++j)
                {
                    gnum[j] = G.at_unchecked(r0 + j, t) * denom_inv;
                    gdot   += G.at_unchecked(r0 + j, t) * num[j];
                }
                const Scalar ddenom = -gdot / (denom * denom);
                const Scalar ds = ddenom * (Scalar{0.5} * denom_inv);
                matvec_T_(B, gnum, tmp, dk);   // tmp = B^T·gnum
                for (std::size_t j = 0; j < dk; ++j)
                {
                    store_q[t * dk + j]  = qv[j];
                    store_dq[t * dk + j] = tmp[j] + Scalar{2} * ds * Aq[j];
                }
                for (std::size_t a = 0; a < dk; ++a)
                    for (std::size_t b = 0; b < dk; ++b)
                    {
                        const Scalar dA_ab = ds * qv[a] * qv[b];
                        const Scalar dB_ab = gnum[a] * qv[b];
                        if (causal)
                        {
                            store_dA[(t * dk + a) * dk + b] = dA_ab;
                            store_dB[(t * dk + a) * dk + b] = dB_ab;
                        }
                        else
                        {
                            // 双向：A、B 为全集常数，dL/dA、dL/dB 是所有 t 求和
                            dA_total[a * dk + b] += dA_ab;
                            dB_total[a * dk + b] += dB_ab;
                        }
                    }
            }
            // Pass 2：反向
            std::fill(A.begin(), A.end(), Scalar{0});  // 复用 A/B 作累积
            std::fill(B.begin(), B.end(), Scalar{0});
            for (std::size_t i = seq; i-- > 0;)
            {
                if (causal)
                {
                    // 后缀和：SA = Σ_{t>=i} dL/dA_t，SB = Σ_{t>=i} dL/dB_t
                    // 文档感知：跨入前一文档时重置后缀和（只统计本文档内 t>=i）
                    if (i + 1 < seq && doc_reset(i + 1))
                    {
                        std::fill(A.begin(), A.end(), Scalar{0});
                        std::fill(B.begin(), B.end(), Scalar{0});
                    }
                    for (std::size_t idx = 0; idx < dk * dk; ++idx)
                    {
                        A[idx] += store_dA[i * dk * dk + idx];
                        B[idx] += store_dB[i * dk * dk + idx];
                    }
                }
                else
                {
                    // 双向：A、B 即全集梯度 dL/dA、dL/dB（对每个 i 相同）
                    std::copy(dA_total.begin(), dA_total.end(), A.begin());
                    std::copy(dB_total.begin(), dB_total.end(), B.begin());
                }
                for (std::size_t j = 0; j < dk; ++j)
                {
                    kv[j] = K.at_unchecked(r0 + j, i);
                    vv[j] = V.at_unchecked(r0 + j, i);
                }
                matvec_(A, kv, tmp, dk);      // SA·k_i
                matvec_T_(B, vv, tmp2, dk);   // SB^T·v_i
                for (std::size_t j = 0; j < dk; ++j)
                    gK.set_value_unchecked(r0 + j, i, Scalar{2} * tmp[j] + tmp2[j]);
                matvec_(B, kv, tmp2, dk);     // SB·k_i
                for (std::size_t j = 0; j < dk; ++j)
                    gV.set_value_unchecked(r0 + j, i, tmp2[j]);
                for (std::size_t j = 0; j < dk; ++j)
                    gQ.set_value_unchecked(r0 + j, i, store_dq[i * dk + j]);
            }
        }
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
        batch_cache_ = 0;
        seq_cache_ = 0;
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

    // 前向：输入 X (d_model, batch·seq) → 输出 (d_model, batch·seq)
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

        // RoPE → ReLU（顺序必须：先旋转后截断，否则丢位置信息）
        if (use_rope_)
        {
            auto qr = rope_.apply(engine, Q, seq, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply(engine, K, seq, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = engine.elementwise_binary_scalar(BinaryOp::Max, Q, Scalar{0}, false);
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = engine.elementwise_binary_scalar(BinaryOp::Max, K, Scalar{0}, false);
        if (!Kp) return std::unexpected(Kp.error());

        // 核心扫描（CPU 标量；任意后端经 staging）
        // 原语组合版 RLA 前向（GPU 上运行，消除 PCIe 往返）
        auto out_t = forward_matmul(engine, *Qp, *Kp, V, batch, seq);
        if (!out_t) return std::unexpected(out_t.error());

        Tensor concat;
        if (batch > 1)
        {
            auto cb = engine.rearrange_3d(*out_t, H_dk, batch, seq, true);
            if (!cb) return std::unexpected(cb.error());
            concat = std::move(*cb);
        }
        else
        {
            concat = std::move(*out_t);
        }

        Qp_cache_ = std::move(*Qp);
        Kp_cache_ = std::move(*Kp);
        V_re_cache_ = std::move(V);
        batch_cache_ = batch;
        seq_cache_   = seq;

        return w_o_.forward(engine, concat);
    }

    // 反向：grad_output (d_model, batch·seq) → 输入梯度 (d_model, batch·seq)
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

        // 原语组合版 RLA 反向（GPU 上运行）
        auto gres = backward_matmul(engine, Qp_cache_, Kp_cache_, V_re_cache_,
                                    gcr, batch, seq);
        if (!gres) return std::unexpected(gres.error());
        auto gQt = std::move((*gres)[0]);
        auto gKt = std::move((*gres)[1]);
        auto gVt = std::move((*gres)[2]);

        // ReLU 反向：(x>0) ? g : 0（x>0 与 Qp/Kp>0 同号）
        auto gq_relu = engine.elementwise_select_scalar_cond(
            CompareOp::Gt, Qp_cache_, Scalar{0}, gQt, Scalar{0});
        if (!gq_relu) return std::unexpected(gq_relu.error());
        auto gk_relu = engine.elementwise_select_scalar_cond(
            CompareOp::Gt, Kp_cache_, Scalar{0}, gKt, Scalar{0});
        if (!gk_relu) return std::unexpected(gk_relu.error());

        // RoPE 反向（旋转正交，逆 = 反角）
        if (use_rope_)
        {
            auto gq = rope_.apply(engine, *gq_relu, seq, true);
            if (!gq) return std::unexpected(gq.error());
            gq_relu = std::move(*gq);
            auto gk = rope_.apply(engine, *gk_relu, seq, true);
            if (!gk) return std::unexpected(gk.error());
            gk_relu = std::move(*gk);
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

    // ── 增量推理（运行态 KV cache） ──────────────────────────────
    // 处理单个新 token，增量更新运行态 A/B（RLA 的"KV cache"），O(d_k²) 每步，
    // 使自回归生成真正线性于上下文长度（无需逐窗重算）。
    //   A_state/B_state: (num_heads*d_k, d_k)，行块 h 分别 = Σ k'_h k'_h^T、Σ v_h k'_h^T
    //   pos: 绝对位置（RoPE 施加于 Q/K 进 ReLU 前）
    // 输入 input: (d_model, 1) 单 token 隐藏态；返回 (d_model, 1) 经 w_o 输出。
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Matrix& A_state, Matrix& B_state, std::size_t pos)
    {
        auto q_res = w_q_.forward(engine, input);   // (d_model, 1) = (H*dk, 1)
        if (!q_res) return q_res;
        auto k_res = w_k_.forward(engine, input);
        if (!k_res) return k_res;
        auto v_res = w_v_.forward(engine, input);
        if (!v_res) return v_res;
        Tensor Q = std::move(*q_res), K = std::move(*k_res), V = std::move(*v_res);
        if (use_rope_)
        {
            auto qr = rope_.apply_step(engine, Q, pos, false); if (!qr) return std::unexpected(qr.error());
            Q = std::move(*qr);
            auto kr = rope_.apply_step(engine, K, pos, false); if (!kr) return std::unexpected(kr.error());
            K = std::move(*kr);
        }
        auto Qp = engine.elementwise_binary_scalar(BinaryOp::Max, Q, Scalar{0}, false);
        if (!Qp) return std::unexpected(Qp.error());
        auto Kp = engine.elementwise_binary_scalar(BinaryOp::Max, K, Scalar{0}, false);
        if (!Kp) return std::unexpected(Kp.error());
        auto Qm = engine.to_matrix(*Qp); if (!Qm) return std::unexpected(Qm.error());
        auto Km = engine.to_matrix(*Kp); if (!Km) return std::unexpected(Km.error());
        auto Vm = engine.to_matrix(V);   if (!Vm) return std::unexpected(Vm.error());
        const std::size_t H = num_heads_;
        const std::size_t dk = d_k_;
        Matrix out_m(H * dk, 1, Scalar{0});
        std::vector<Scalar> num(dk), Aq(dk);
        for (std::size_t h = 0; h < H; ++h)
        {
            const std::size_t r0 = h * dk;
            // 更新运行态：A[h] += k'k'^T，B[h] += v k'^T（因果：先累积再用）
            for (std::size_t a = 0; a < dk; ++a)
                for (std::size_t b = 0; b < dk; ++b)
                {
                    const Scalar ka = Km->at_unchecked(r0 + a, 0);
                    const Scalar kb = Km->at_unchecked(r0 + b, 0);
                    A_state.set_value_unchecked(r0 + a, b,
                        A_state.at_unchecked(r0 + a, b) + ka * kb);
                    B_state.set_value_unchecked(r0 + a, b,
                        B_state.at_unchecked(r0 + a, b) + Vm->at_unchecked(r0 + a, 0) * kb);
                }
            // num = B[h]·q'；Aq = A[h]·q'；s = q'·Aq；out = num / sqrt(s+eps)
            for (std::size_t i = 0; i < dk; ++i)
            {
                Scalar n{0}, aq{0};
                for (std::size_t b = 0; b < dk; ++b)
                {
                    n  += B_state.at_unchecked(r0 + i, b) * Qm->at_unchecked(r0 + b, 0);
                    aq += A_state.at_unchecked(r0 + i, b) * Qm->at_unchecked(r0 + b, 0);
                }
                num[i] = n; Aq[i] = aq;
            }
            Scalar s{0};
            for (std::size_t i = 0; i < dk; ++i) s += Qm->at_unchecked(r0 + i, 0) * Aq[i];
            const Scalar denom = std::sqrt(s + Scalar{1e-6});
            for (std::size_t i = 0; i < dk; ++i)
                out_m.set_value_unchecked(r0 + i, 0, num[i] / denom);
        }
        auto out_t = engine.from_matrix(out_m);
        if (!out_t) return std::unexpected(out_t.error());
        return w_o_.forward(engine, *out_t);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTBlock — RAPT 解码器块（GPT 风格：Norm → RLA 注意力 → 残差 → FFN → 残差）
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

    // 文档感知：转发给内部 RLA 注意力（文档边界处重置运行态）
    void set_doc_ids(std::span<const std::size_t> ids) override
    {
        attn_.set_doc_ids(ids);
    }

    // 绝对位置偏移：转发给内部 RLA 注意力（滑动窗生成时 RoPE 用绝对位置）
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
        auto r1 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
        return engine.elementwise_binary(BinaryOp::Add, *r1, *f);
    }

    [[nodiscard]] Result<Tensor> backward(
        ComputeEngine& engine, const Tensor& grad_output) override
    {
        auto grad_ff = ff_.backward(engine, grad_output);
        if (!grad_ff) return grad_ff;
        auto b_n2 = norm2_->backward(engine, *grad_ff);
        if (!b_n2) return b_n2;
        auto grad_r1 = engine.elementwise_binary(BinaryOp::Add, grad_output, *b_n2);
        if (!grad_r1) return std::unexpected(grad_r1.error());
        auto grad_a = attn_.backward(engine, *grad_r1);
        if (!grad_a) return grad_a;
        auto b_n1 = norm1_->backward(engine, *grad_a);
        if (!b_n1) return b_n1;
        return engine.elementwise_binary(BinaryOp::Add, *grad_r1, *b_n1);
    }

    // 增量推理：单 token 经 norm1 → RLA 运行态注意力 → 残差 → norm2 → FFN → 残差
    // A/B_state: RLA 运行态 KV cache（见 ReLULinearAttention::forward_step）
    [[nodiscard]] Result<Tensor> forward_step(
        ComputeEngine& engine, const Tensor& input,
        Matrix& A_state, Matrix& B_state, std::size_t pos)
    {
        auto n1 = norm1_->forward(engine, input);
        if (!n1) return n1;
        auto a = attn_.forward_step(engine, *n1, A_state, B_state, pos);
        if (!a) return a;
        auto r1 = engine.elementwise_binary(BinaryOp::Add, input, *a);
        if (!r1) return std::unexpected(r1.error());
        auto n2 = norm2_->forward(engine, *r1);
        if (!n2) return n2;
        auto f = ff_.forward(engine, *n2);
        if (!f) return f;
        return engine.elementwise_binary(BinaryOp::Add, *r1, *f);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// RAPTModel — RAPT 解码器（ReLU 激活线性注意力语言模型）
//
//   token_emb → (+pos_enc，RoPE 在注意力内部施加) → N × RAPTBlock → LN → LM Head
//
// RLA 约束：位置编码必须用 RoPE（或 ALiBi），且 RoPE 施加在 Q/K 进 ReLU 之前；
// 本实现强制 RoPE（v1 不支持 ALiBi），输入侧用 NoPositionEncoder（无位置嵌入）。
// ══════════════════════════════════════════════════════════════════════════
class RAPTModel final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::size_t seq_len_;
    std::size_t num_heads_;   // 用于运行态 KV cache 尺寸（d_k = d_model/num_heads）
    // （d_ff_/num_layers_/causal_ 仅构造时用，不存成员避免 -Wunused）

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
        // RLA 强约束：必须 RoPE（或 ALiBi）。v1 强制 RoPE，输入侧无位置嵌入。
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

    // 文档感知：记录 doc_ids，forward 时下发给各块（文档边界处重置 RLA 运行态）
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

        // 文档感知：每块 forward 前下发本步 doc_ids（边界重置运行态）
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

    // ── 采样生成（增量运行态，KV cache） ────────────────────────
    // 利用 RLA 的运行态前缀和（A_t、B_t）作为天然 KV cache：逐 token 增量更新，
    // 每步 O(d²)（不重算前文），使生成真正线性于上下文长度，且无滑动窗截断
    // （绝对位置经 pos 计数器自然递增，RoPE 位置恒正确）。
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
        // 每块一个运行态 KV cache（A/B，形状 (d_model, d_k)，行块按头划分）
        std::vector<Matrix> statesA, statesB;
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            statesA.emplace_back(d_model_, dk, Scalar{0});
            statesB.emplace_back(d_model_, dk, Scalar{0});
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
                auto r = blocks_[i].forward_step(engine, h, statesA[i], statesB[i], pos);
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

#endif // NN_COMPUTE_LAYER_RAPT_HPP
