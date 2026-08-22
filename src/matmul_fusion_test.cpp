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

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e30f;
    Scalar e = 0;
    for (std::size_t i = 0; i < a.span().size(); ++i)
        e = std::max(e, std::fabs(a.span()[i] - b.span()[i]));
    return e;
}

void check_matrix(const nn::Matrix& got, const nn::Matrix& ref, const char* msg)
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
                        usemask ? &tm : nullptr);
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
        auto r = eng.batched_matmul_softmax_denom(ta_t, tb, trm, batch, true, false, alpha, &tm);
        check(nm, std::move(r), ref);

        // 无掩码
        std::snprintf(nm, sizeof(nm), "%s bmm_denom (no mask)", tag);
        const nn::Matrix ref2 = ref_denom(A_t, B, row_max, batch, true, false, alpha, nullptr, M, K, N);
        auto r2 = eng.batched_matmul_softmax_denom(ta_t, tb, trm, batch, true, false, alpha, nullptr);
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
                                                  true, false, alpha, &tm);
        check(nm, std::move(r), ref);

        // 无掩码
        std::snprintf(nm, sizeof(nm), "%s bmm_apply (no mask)", tag);
        const nn::Matrix ref2 = ref_apply(A_t, B, V, row_max, denom, batch,
                                          true, false, alpha, nullptr, M, K, N, D);
        auto r2 = eng.batched_matmul_softmax_apply(ta_t, tb, tv, trm, tdl, batch,
                                                   true, false, alpha, nullptr);
        check(nm, std::move(r2), ref2);
    }

    return fail;
}

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
#endif

    std::cout << (fail == 0 ? "\nALL PASS\n" : "\nFAILED\n");
    return fail == 0 ? 0 : 1;
}
