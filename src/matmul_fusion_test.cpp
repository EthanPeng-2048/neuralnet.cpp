// ───────────────────────────────────────────────────────────────────────────
//  matmul_fusion_test.cpp — M4：matmul 融合原语（bmm_reduce/bmm_denom/bmm_apply）
//
//  验证三个 op-level 融合原语（两趟注意力的承载点）：
//    - batched_matmul_reduce：matmul 后沿输出列归约（max/sum，可选掩码）
//    - batched_matmul_softmax_denom：matmul → 减行 max → exp → 求和（分母）
//    - batched_matmul_softmax_apply：行 softmax 归一化后与 V 相乘累加（Pass 2）
//  CPU 引擎 vs 手写参考；GPU（Vulkan）融合 shader vs CPU 参考。
//
//  用法：matmul_fusion_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using nn::Scalar;

int g_fail = 0;

// ── 参考实现（手写，与 CpuEngine 语义一致）─────────────────────────────

// 批量 matmul 单点：A_b[i][k]·B_b[k][j]（尊重 transA/transB）
Scalar ref_dot(const nn::Matrix& a, const nn::Matrix& b,
               std::size_t bi, std::size_t i, std::size_t j,
               std::size_t M, std::size_t K, std::size_t N,
               bool transA, bool transB)
{
    const std::size_t abase = bi * M * K;
    const std::size_t bbase = bi * K * N;
    Scalar s = 0;
    for (std::size_t k = 0; k < K; ++k)
    {
        const Scalar av = !transA ? a.span()[abase + i * K + k]
                                  : a.span()[abase + k * M + i];
        const Scalar bv = !transB ? b.span()[bbase + k * N + j]
                                  : b.span()[bbase + j * K + k];
        s += av * bv;
    }
    return s;
}

// bmm_reduce 参考（reduce_cols=true）
nn::Matrix ref_reduce(const nn::Matrix& a, const nn::Matrix& b,
                      std::size_t batch, nn::ReduceOp op,
                      bool transA, bool transB, Scalar alpha,
                      const nn::Matrix* mask,
                      std::size_t M, std::size_t K, std::size_t N)
{
    nn::Matrix out(batch * M, 1);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        for (std::size_t i = 0; i < M; ++i)
        {
            Scalar acc = (op == nn::ReduceOp::Max)
                ? -std::numeric_limits<Scalar>::infinity()
                : (op == nn::ReduceOp::Min)
                    ? std::numeric_limits<Scalar>::infinity() : Scalar{0};
            for (std::size_t j = 0; j < N; ++j)
            {
                Scalar mv = 0;
                if (mask) mv = mask->at_unchecked(i, j);
                if ((op == nn::ReduceOp::Sum || op == nn::ReduceOp::Min) &&
                    mv == -std::numeric_limits<Scalar>::infinity())
                    continue;
                const Scalar s = alpha * ref_dot(a, b, bi, i, j, M, K, N, transA, transB) + mv;
                if (op == nn::ReduceOp::Max) acc = std::max(acc, s);
                else if (op == nn::ReduceOp::Min) acc = std::min(acc, s);
                else acc += s;
            }
            out.set_value_unchecked(bi * M + i, 0, acc);
        }
    }
    return out;
}

// bmm_denom 参考
nn::Matrix ref_denom(const nn::Matrix& a, const nn::Matrix& b, const nn::Matrix& row_max,
                     std::size_t batch, bool transA, bool transB, Scalar alpha,
                     const nn::Matrix* mask,
                     std::size_t M, std::size_t K, std::size_t N)
{
    nn::Matrix out(batch * M, 1);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        for (std::size_t i = 0; i < M; ++i)
        {
            const Scalar mval = row_max.at_unchecked(bi * M + i, 0);
            Scalar acc = 0;
            for (std::size_t j = 0; j < N; ++j)
            {
                Scalar mv = 0;
                if (mask) mv = mask->at_unchecked(i, j);
                if (mv == -std::numeric_limits<Scalar>::infinity()) continue;
                const Scalar s = alpha * ref_dot(a, b, bi, i, j, M, K, N, transA, transB) + mv - mval;
                acc += std::exp(s);
            }
            out.set_value_unchecked(bi * M + i, 0, acc);
        }
    }
    return out;
}

// bmm_apply 参考
nn::Matrix ref_apply(const nn::Matrix& a, const nn::Matrix& b, const nn::Matrix& v,
                     const nn::Matrix& row_max, const nn::Matrix& denom,
                     std::size_t batch, bool transA, bool transB, Scalar alpha,
                     const nn::Matrix* mask,
                     std::size_t M, std::size_t K, std::size_t N, std::size_t D)
{
    nn::Matrix out(batch * M, D);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        for (std::size_t i = 0; i < M; ++i)
        {
            const Scalar mval = row_max.at_unchecked(bi * M + i, 0);
            const Scalar inv_l = Scalar{1} / denom.at_unchecked(bi * M + i, 0);
            std::vector<Scalar> w(N);
            for (std::size_t j = 0; j < N; ++j)
            {
                Scalar mv = 0;
                if (mask) mv = mask->at_unchecked(i, j);
                if (mv == -std::numeric_limits<Scalar>::infinity()) { w[j] = 0; continue; }
                const Scalar s = alpha * ref_dot(a, b, bi, i, j, M, K, N, transA, transB) + mv - mval;
                w[j] = std::exp(s) * inv_l;
            }
            for (std::size_t k = 0; k < D; ++k)
            {
                Scalar acc = 0;
                for (std::size_t j = 0; j < N; ++j)
                    acc += w[j] * v.span()[(bi * N + j) * D + k];
                out.set_value_unchecked(bi * M + i, k, acc);
            }
        }
    }
    return out;
}

// bmm_q_backward 参考：R 与 grad_Q
nn::Matrix ref_backward_q(const nn::Matrix& a, const nn::Matrix& b, const nn::Matrix& p,
                          const nn::Matrix& row_max, const nn::Matrix& denom,
                          std::size_t batch, bool transA, bool transB, Scalar alpha,
                          const nn::Matrix* mask,
                          std::size_t M, std::size_t K, std::size_t N,
                          nn::Matrix& r_out)
{
    nn::Matrix r(batch * M, 1);
    nn::Matrix gq(batch * K, M);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        for (std::size_t i = 0; i < M; ++i)
        {
            const Scalar mval = row_max.at_unchecked(bi * M + i, 0);
            const Scalar inv_l = Scalar{1} / denom.at_unchecked(bi * M + i, 0);
            std::vector<Scalar> w(N);
            Scalar rv = 0;
            for (std::size_t j = 0; j < N; ++j)
            {
                Scalar mv = 0;
                if (mask) mv = mask->at_unchecked(i, j);
                if (mv == -std::numeric_limits<Scalar>::infinity()) { w[j] = 0; continue; }
                const Scalar s = alpha * ref_dot(a, b, bi, i, j, M, K, N, transA, transB) + mv - mval;
                w[j] = std::exp(s) * inv_l;
                rv += w[j] * p.at_unchecked(bi * M + i, j);
            }
            r.set_value_unchecked(bi * M + i, 0, rv);
            for (std::size_t k = 0; k < K; ++k)
            {
                Scalar acc = 0;
                for (std::size_t j = 0; j < N; ++j)
                {
                    const Scalar bv = !transB ? b.at_unchecked(bi * K + k, j)
                                              : b.at_unchecked(bi * N + j, k);
                    acc += w[j] * (p.at_unchecked(bi * M + i, j) - rv) * bv;
                }
                gq.set_value_unchecked(bi * K + k, i, alpha * acc);
            }
        }
    }
    r_out = std::move(r);
    return gq;
}

// bmm_kv_backward 参考：grad_K 与 grad_V
nn::Matrix ref_backward_kv(const nn::Matrix& a, const nn::Matrix& b, const nn::Matrix& p,
                           const nn::Matrix& g, const nn::Matrix& r,
                           const nn::Matrix& row_max, const nn::Matrix& denom,
                           std::size_t batch, bool transA, bool transB, Scalar alpha,
                           const nn::Matrix* mask,
                           std::size_t M, std::size_t K, std::size_t N, std::size_t D,
                           nn::Matrix& gv_out)
{
    (void)D;  // G 的列数由矩阵自身携带；D 仅用于语义对齐
    nn::Matrix gk(batch * K, N);
    nn::Matrix gv(batch * K, N);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            std::vector<Scalar> w(M);
            for (std::size_t i = 0; i < M; ++i)
            {
                Scalar mv = 0;
                if (mask) mv = mask->at_unchecked(i, j);
                if (mv == -std::numeric_limits<Scalar>::infinity()) { w[i] = 0; continue; }
                const std::size_t ri = bi * M + i;
                const Scalar s = alpha * ref_dot(a, b, bi, i, j, M, K, N, transA, transB)
                                 + mv - row_max.at_unchecked(ri, 0);
                w[i] = std::exp(s) / denom.at_unchecked(ri, 0);
            }
            for (std::size_t k = 0; k < K; ++k)
            {
                Scalar accK = 0, accV = 0;
                for (std::size_t i = 0; i < M; ++i)
                {
                    const std::size_t ri = bi * M + i;
                    // A_b[:,i]（第 i 个 query 向量）：!transA 取 A_b[i][k]，
                    // transA 时 A_b (K,M) 取 A_b[k][i]。
                    const Scalar av = !transA ? a.at_unchecked(bi * M + i, k)
                                              : a.at_unchecked(bi * K + k, i);
                    accK += w[i] * (p.at_unchecked(ri, j) - r.at_unchecked(ri, 0)) * av;
                    accV += w[i] * g.at_unchecked(ri, k);
                }
                gk.set_value_unchecked(bi * K + k, j, alpha * accK);
                gv.set_value_unchecked(bi * K + k, j, accV);
            }
        }
    }
    gv_out = std::move(gv);
    return gk;
}

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e30f;
    Scalar e = 0;
    for (std::size_t i = 0; i < a.span().size(); ++i)
        e = std::max(e, std::fabs(a.span()[i] - b.span()[i]));
    return e;
}void check_matrix(const nn::Matrix& got, const nn::Matrix& ref, const char* msg)
{
    const Scalar err = max_abs_diff(got, ref);
    const bool ok = err < 1e-4f;
    std::printf("[%s] %s  err=%.3e\n", ok ? "PASS" : "FAIL", msg, err);
    if (!ok) ++g_fail;
}

// ── 测试主体 ──────────────────────────────────────────────────────────────
int run_case(nn::ComputeEngine& eng, const char* tag)
{
    // 注意力式形状：batch=2, M=4(query), N=6(key), K=4(d_k), D=5
    const std::size_t batch = 2, M = 4, N = 6, K = 4, D = 5;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    // A: (batch*M, K) 或 (batch*K, M)（transA）；B: (batch*K, N)；V: (batch*N, D)
    nn::Matrix A(batch * M, K), A_t(batch * K, M);
    nn::Matrix B(batch * K, N), B_t(batch * N, K);
    nn::Matrix V(batch * N, D);
    for (auto& x : A.span()) x = dist(rng);
    for (auto& x : A_t.span()) x = dist(rng);
    for (auto& x : B.span()) x = dist(rng);
    for (auto& x : B_t.span()) x = dist(rng);
    for (auto& x : V.span()) x = dist(rng);

    // 因果掩码（-inf 屏蔽 j>i）+ 参考 row_max/denom
    nn::Matrix mask(M, N, 0.0f);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = i + 1; j < N; ++j)
            mask.set_value_unchecked(i, j, -std::numeric_limits<Scalar>::infinity());
    nn::Matrix row_max(batch * M, 1, 0.5f);
    nn::Matrix denom(batch * M, 1, 2.0f);
    for (auto& x : row_max.span()) x = dist(rng) * 0.5f;
    for (auto& x : denom.span()) x = 1.0f + dist(rng) * 0.5f;

    const Scalar alpha = 0.35f;

    int fail = 0;
    (void)fail;

    // 结果检查：先下载（GPU → CPU），再与参考比对
    const auto check = [&](const char* nm, nn::Result<nn::Tensor>&& r, const nn::Matrix& ref)
    {
        if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; return; }
        auto dm = eng.to_matrix(*r);
        if (!dm) { std::printf("[FAIL] %s: 下载失败: %s\n", nm, dm.error().message.c_str()); ++fail; return; }
        check_matrix(*dm, ref, nm);
    };

    // ── bmm_reduce：四种转置组合 × Max/Sum × 有/无掩码 ──
    {
        const nn::Tensor ta = nn::Tensor::from_matrix(nn::Matrix(A));
        const nn::Tensor tb = nn::Tensor::from_matrix(nn::Matrix(B));
        const nn::Tensor ta_t = nn::Tensor::from_matrix(nn::Matrix(A_t));
        const nn::Tensor tb_t = nn::Tensor::from_matrix(nn::Matrix(B_t));
        const nn::Tensor tm = nn::Tensor::from_matrix(nn::Matrix(mask));

        for (int combo = 0; combo < 4; ++combo)
        {
            const bool trA = combo & 1, trB = combo & 2;
            for (int om = 0; om < 2; ++om)  // Max / Sum
            {
                const nn::ReduceOp op = om ? nn::ReduceOp::Sum : nn::ReduceOp::Max;
                for (int usemask = 0; usemask < 2; ++usemask)
                {
                    char nm[128];
                    std::snprintf(nm, sizeof(nm), "%s bmm_reduce trA=%d trB=%d op=%s mask=%d",
                                  tag, trA, trB, om ? "sum" : "max", usemask);
                    const nn::Matrix ref = ref_reduce(
                        trA ? A_t : A, trB ? B_t : B, batch, op, trA, trB, alpha,
                        usemask ? &mask : nullptr, M, K, N);
                    auto r = eng.batched_matmul_reduce(
                        trA ? ta_t : ta, trB ? tb_t : tb, batch, op,
                        trA, trB, alpha, /*reduce_cols=*/true,
                        usemask ? nn::AttnBias{&tm} : nn::AttnBias{});
                    check(nm, std::move(r), ref);
                }
            }
        }
    }

    // ── bmm_denom（transA=true, transB=false 注意力布局 + 掩码） ──
    {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s bmm_denom (attn layout + mask)", tag);
        const nn::Tensor ta_t = nn::Tensor::from_matrix(nn::Matrix(A_t));
        const nn::Tensor tb = nn::Tensor::from_matrix(nn::Matrix(B));
        const nn::Tensor tm = nn::Tensor::from_matrix(nn::Matrix(mask));
        const nn::Tensor trm = nn::Tensor::from_matrix(nn::Matrix(row_max));
        const nn::Matrix ref = ref_denom(A_t, B, row_max, batch, /*trA=*/true, /*trB=*/false,
                                         alpha, &mask, M, K, N);
        auto r = eng.batched_matmul_softmax_denom(ta_t, tb, trm, batch, true, false, alpha, nn::AttnBias{&tm});
        check(nm, std::move(r), ref);

        // 无掩码
        std::snprintf(nm, sizeof(nm), "%s bmm_denom (no mask)", tag);
        const nn::Matrix ref2 = ref_denom(A_t, B, row_max, batch, true, false, alpha, nullptr, M, K, N);
        auto r2 = eng.batched_matmul_softmax_denom(ta_t, tb, trm, batch, true, false, alpha, nn::AttnBias{});
        check(nm, std::move(r2), ref2);
    }

    // ── bmm_apply（注意力布局 + 掩码） ──
    {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s bmm_apply (attn layout + mask)", tag);
        const nn::Tensor ta_t = nn::Tensor::from_matrix(nn::Matrix(A_t));
        const nn::Tensor tb = nn::Tensor::from_matrix(nn::Matrix(B));
        const nn::Tensor tm = nn::Tensor::from_matrix(nn::Matrix(mask));
        const nn::Tensor tv = nn::Tensor::from_matrix(nn::Matrix(V));
        const nn::Tensor trm = nn::Tensor::from_matrix(nn::Matrix(row_max));
        const nn::Tensor tdl = nn::Tensor::from_matrix(nn::Matrix(denom));
        const nn::Matrix ref = ref_apply(A_t, B, V, row_max, denom, batch,
                                         true, false, alpha, &mask, M, K, N, D);
        auto r = eng.batched_matmul_softmax_apply(ta_t, tb, tv, trm, tdl, batch,
                                                  true, false, alpha, nn::AttnBias{&tm});
        check(nm, std::move(r), ref);

        // 无掩码
        std::snprintf(nm, sizeof(nm), "%s bmm_apply (no mask)", tag);
        const nn::Matrix ref2 = ref_apply(A_t, B, V, row_max, denom, batch,
                                          true, false, alpha, nullptr, M, K, N, D);
        auto r2 = eng.batched_matmul_softmax_apply(ta_t, tb, tv, trm, tdl, batch,
                                                   true, false, alpha, nn::AttnBias{});
        check(nm, std::move(r2), ref2);
    }

    // ── bmm_q_backward（注意力布局 + 掩码）：R 与 grad_Q ──
    {
        char nm[128];
        const nn::Tensor ta_t = nn::Tensor::from_matrix(nn::Matrix(A_t));
        const nn::Tensor tb = nn::Tensor::from_matrix(nn::Matrix(B));
        const nn::Tensor tm = nn::Tensor::from_matrix(nn::Matrix(mask));
        const nn::Tensor trm = nn::Tensor::from_matrix(nn::Matrix(row_max));
        const nn::Tensor tdl = nn::Tensor::from_matrix(nn::Matrix(denom));
        // P: (batch*M, N) — 模拟 grad_A
        nn::Matrix P(batch * M, N);
        for (auto& x : P.span()) x = dist(rng);
        const nn::Tensor tp = nn::Tensor::from_matrix(nn::Matrix(P));

        // 有掩码：R 与 grad_Q
        nn::Matrix r_ref, gq_ref;
        gq_ref = ref_backward_q(A_t, B, P, row_max, denom, batch, true, false,
                                alpha, &mask, M, K, N, r_ref);
        {
            nn::Tensor r_out;
            std::snprintf(nm, sizeof(nm), "%s bmm_q_backward (mask)", tag);
            auto r = eng.batched_matmul_softmax_backward_q(
                ta_t, tb, tp, trm, tdl, batch, true, false, alpha, r_out, nn::AttnBias{&tm});
            if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; }
            else
            {
                auto dq = eng.to_matrix(*r);
                if (!dq) { std::printf("[FAIL] %s: 下载失败\n", nm); ++fail; }
                else check_matrix(*dq, gq_ref, nm);
                auto dr = eng.to_matrix(r_out);
                if (!dr) { std::printf("[FAIL] %s (R): 下载失败\n", nm); ++fail; }
                else check_matrix(*dr, r_ref, "  R (mask)");
            }
        }
        // 无掩码：R 与 grad_Q
        nn::Matrix r_ref2, gq_ref2;
        gq_ref2 = ref_backward_q(A_t, B, P, row_max, denom, batch, true, false,
                                 alpha, nullptr, M, K, N, r_ref2);
        {
            nn::Tensor r_out;
            std::snprintf(nm, sizeof(nm), "%s bmm_q_backward (no mask)", tag);
            auto r = eng.batched_matmul_softmax_backward_q(
                ta_t, tb, tp, trm, tdl, batch, true, false, alpha, r_out, nn::AttnBias{});
            if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; }
            else
            {
                auto dq = eng.to_matrix(*r);
                if (!dq) { std::printf("[FAIL] %s: 下载失败\n", nm); ++fail; }
                else check_matrix(*dq, gq_ref2, nm);
                auto dr = eng.to_matrix(r_out);
                if (!dr) { std::printf("[FAIL] %s (R): 下载失败\n", nm); ++fail; }
                else check_matrix(*dr, r_ref2, "  R (no mask)");
            }
        }
    }

    // ── bmm_kv_backward（注意力布局 + 掩码）：grad_K 与 grad_V ──
    {
        char nm[128];
        const nn::Tensor ta_t = nn::Tensor::from_matrix(nn::Matrix(A_t));
        const nn::Tensor tb = nn::Tensor::from_matrix(nn::Matrix(B));
        const nn::Tensor tm = nn::Tensor::from_matrix(nn::Matrix(mask));
        const nn::Tensor trm = nn::Tensor::from_matrix(nn::Matrix(row_max));
        const nn::Tensor tdl = nn::Tensor::from_matrix(nn::Matrix(denom));
        // P: (batch*M, N)，G: (batch*M, D)（模拟 grad_O^T），R: (batch*M, 1)
        nn::Matrix P(batch * M, N);
        nn::Matrix G(batch * M, D);
        nn::Matrix R(batch * M, 1);
        for (auto& x : P.span()) x = dist(rng);
        for (auto& x : G.span()) x = dist(rng);
        for (auto& x : R.span()) x = dist(rng);
        const nn::Tensor tp = nn::Tensor::from_matrix(nn::Matrix(P));
        const nn::Tensor tg = nn::Tensor::from_matrix(nn::Matrix(G));
        const nn::Tensor tr = nn::Tensor::from_matrix(nn::Matrix(R));

        nn::Matrix gk_ref, gv_ref;
        gk_ref = ref_backward_kv(A_t, B, P, G, R, row_max, denom, batch, true, false,
                                 alpha, &mask, M, K, N, D, gv_ref);
        {
            nn::Tensor gv_out;
            std::snprintf(nm, sizeof(nm), "%s bmm_kv_backward (mask)", tag);
            auto r = eng.batched_matmul_softmax_backward_kv(
                ta_t, tb, tp, tg, tr, trm, tdl, batch, true, false, alpha, gv_out, nn::AttnBias{&tm});
            if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; }
            else
            {
                auto dk = eng.to_matrix(*r);
                if (!dk) { std::printf("[FAIL] %s: 下载失败\n", nm); ++fail; }
                else check_matrix(*dk, gk_ref, nm);
                auto dv = eng.to_matrix(gv_out);
                if (!dv) { std::printf("[FAIL] %s (V): 下载失败\n", nm); ++fail; }
                else check_matrix(*dv, gv_ref, "  grad_V (mask)");
            }
        }
        nn::Matrix gk_ref2, gv_ref2;
        gk_ref2 = ref_backward_kv(A_t, B, P, G, R, row_max, denom, batch, true, false,
                                  alpha, nullptr, M, K, N, D, gv_ref2);
        {
            nn::Tensor gv_out;
            std::snprintf(nm, sizeof(nm), "%s bmm_kv_backward (no mask)", tag);
            auto r = eng.batched_matmul_softmax_backward_kv(
                ta_t, tb, tp, tg, tr, trm, tdl, batch, true, false, alpha, gv_out, nn::AttnBias{});
            if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; }
            else
            {
                auto dk = eng.to_matrix(*r);
                if (!dk) { std::printf("[FAIL] %s: 下载失败\n", nm); ++fail; }
                else check_matrix(*dk, gk_ref2, nm);
                auto dv = eng.to_matrix(gv_out);
                if (!dv) { std::printf("[FAIL] %s (V): 下载失败\n", nm); ++fail; }
                else check_matrix(*dv, gv_ref2, "  grad_V (no mask)");
            }
        }
    }

    return fail;
}

#ifdef NN_HAS_VULKAN
// ── 组合偏置（AttnBias：causal + doc_ids + ALiBi slopes）CPU vs GPU 一致性 ──
// 覆盖两趟式原语的通用偏置描述子（M7 统一位置编码路径）：5 个原语分别跑在
// CPU / GPU，比较输出（GPU shader 与 CPU attn_bias_at 参考应一致）。
// 注：仅在 NN_HAS_VULKAN 下编译，因为函数签名引用 nn::GpuEngine（Vulkan 专属）。
int run_compositional_bias(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    int fail = 0;
    const std::size_t batch = 2, heads = 2, M = 4, N = 6, K = 4, D = 5;
    std::mt19937 rng(7);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    nn::Matrix A(batch * M, K), B(batch * K, N), V(batch * N, D);
    nn::Matrix row_max(batch * M, 1), denom(batch * M, 1);
    nn::Matrix P(batch * M, N), G(batch * M, D), R0(batch * M, 1);
    for (auto* X : {&A, &B, &V, &P, &G, &R0}) for (auto& x : X->span()) x = dist(rng);
    for (auto& x : row_max.span()) x = dist(rng) + 1.0f;   // 保证 denom>0、稳定
    for (auto& x : denom.span())  x = dist(rng) + 1.0f;
    // doc_ids：(1, batch*N)，前半样本 doc=1、后半 doc=2（块对角）
    nn::Matrix doc(1, batch * N);
    for (std::size_t p = 0; p < batch * N; ++p)
        doc.set_value_unchecked(0, p, (p / N < batch / 2) ? 1.0f : 2.0f);
    // slopes：(1, heads) ALiBi 斜率 m_h = 2^(-8h/H)
    nn::Matrix slopes(1, heads);
    for (std::size_t h = 0; h < heads; ++h)
        slopes.set_value_unchecked(0, h, std::pow(2.0f, -8.0f * h / heads));

    auto to_t = [](const nn::Matrix& m) { return nn::Tensor::from_matrix(nn::Matrix(m)); };
    const nn::Tensor tA = to_t(A), tB = to_t(B), tV = to_t(V), trm = to_t(row_max),
                     tdl = to_t(denom), tP = to_t(P), tG = to_t(G), tR0 = to_t(R0),
                     tDoc = to_t(doc), tSlopes = to_t(slopes);
    nn::AttnBias bias;
    bias.causal = true;
    bias.num_heads = heads;
    bias.doc_ids = &tDoc;
    bias.slopes = &tSlopes;

    // 在指定引擎上跑 5 个原语，收集输出；失败返回 false
    auto run = [&](nn::ComputeEngine& eng, std::vector<nn::Tensor>& out) -> bool
    {
        out.clear();
        auto m = eng.batched_matmul_reduce(tA, tB, batch, nn::ReduceOp::Max,
                                           true, false, 0.7f, true, bias);
        if (!m) return false;
        auto l = eng.batched_matmul_softmax_denom(tA, tB, *m, batch, true, false, 0.7f, bias);
        if (!l) return false;
        auto O = eng.batched_matmul_softmax_apply(tA, tB, tV, *m, *l, batch, true, false, 0.7f, bias);
        if (!O) return false;
        nn::Tensor R, gv;
        auto gq = eng.batched_matmul_softmax_backward_q(tA, tB, tP, *m, *l,
                                                        batch, true, false, 0.7f, R, bias);
        if (!gq) return false;
        auto gkv = eng.batched_matmul_softmax_backward_kv(tA, tB, tP, tG, R, *m, *l,
                                                          batch, true, false, 0.7f, gv, bias);
        if (!gkv) return false;
        out = {std::move(*m), std::move(*l), std::move(*O), std::move(*gq),
               std::move(R), std::move(*gkv), std::move(gv)};
        return true;
    };

    std::vector<nn::Tensor> co, go;
    if (!run(cpu, co)) { std::printf("[FAIL] compositional: CPU 执行失败\n"); return 1; }
    if (!run(gpu, go)) { std::printf("[FAIL] compositional: GPU 执行失败\n"); return 1; }
    const char* names[] = {"m", "l", "O", "gq", "R", "gk", "gv"};
    for (std::size_t i = 0; i < co.size() && i < go.size(); ++i)
    {
        auto cm = cpu.to_matrix(co[i]);
        auto gm = gpu.to_matrix(go[i]);
        if (!cm || !gm) { ++fail; std::printf("[FAIL] compositional %s 下载失败\n", names[i]); continue; }
        std::string msg = std::string("compositional ") + names[i] + " (CPU vs GPU)";
        check_matrix(*gm, *cm, msg.c_str());
        if (max_abs_diff(*cm, *gm) >= 1e-4f) ++fail;
    }
    return fail;
}
#endif  // NN_HAS_VULKAN

int main()
{
    std::cout << "========================================\n"
              << "  M4 matmul 融合原语数值验证\n"
              << "========================================\n";

    nn::CpuEngine cpu_engine;
    int fail = run_case(cpu_engine, "CPU");

#ifndef NN_HAS_VULKAN
    std::cout << "[SKIP] 无 Vulkan，跳过 GPU 部分\n";
#else
    auto& backend = nn::GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    nn::GpuEngine gpu_engine(backend);
    fail += run_case(gpu_engine, "GPU");
    fail += run_compositional_bias(cpu_engine, gpu_engine);
#endif

    std::cout << (fail == 0 ? "\nALL PASS\n" : "\nFAILED\n");
    return fail == 0 ? 0 : 1;
}
