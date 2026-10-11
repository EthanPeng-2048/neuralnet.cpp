// ── ternary_gemm_test — 三值打包权重与量化 GEMM（P2 / F3）────────────────────
// 设计依据：docs/development/21-quantized-weights.md §5 的 P2 验收
//   · 打包往返逐位（编码/解码）
//   · 数值与 P1 的 f16 稠密缓冲路径一致（同 latent 权重下三值逐位）
//   · 位宽账：0.25 B/权重（f16 稠密 = 2 B/权重）
//   · 布局对齐 microsoft/BitNet `gpu/pack_weight.py`（2 bit/权重、4 权重/字节、
//     码值 = 三值 + 2），仅不采用其 CUDA interleave 置换
//
// 覆盖：
//   [1] 布局与编解码：位序、尾部填充（码 2）、尺寸账（words/bytes per row）
//   [2] 引擎 trit_pack = 宿主 pack_row 逐位一致（同一 latent + τ）
//   [3] matmul_q（transA=0）三路对拍：宿主参考 = 原语 = f16 稠密 matmul 路径
//   [4] matmul_q_t（transA=1）：同上（反向 dX 的形态）
//   [5] BitLinear 端到端：打包路径 vs 强制退回的 f16 路径（同 latent，前向
//       与梯度都在容差内一致；三值逐位一致）
//   [6] 跨形状对拍：分块内核的**形状契约**（轴向 / 尾块 / 单列 × transA × f32/f16）
//   [7] BitLinear 跨形状端到端：打包 vs 稠密（多块 + 尾块 + 两种 profile）
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备，或引擎无打包能力）
// ───────────────────────────────────────────────────────────────────────────

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_layer_bitlinear.hpp"
#include "neuralnet.cpp/ternary_pack.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

// ── [1] 布局与编解码 ──────────────────────────────────────────────────────
void test_layout()
{
    std::puts("[1] 打包布局：2 bit/权重、4 权重/字节、码值 = 三值 + 2");
    using namespace nn::trit_pack;

    CHECK(kBits == 2 && kPerByte == 4 && kPerWord == 16, "位宽/密度常量");
    CHECK(words_per_row(16) == 1 && words_per_row(17) == 2 && words_per_row(1) == 1,
          "words_per_row 向上取整");
    CHECK(bytes_per_row(512) == 128, "K=512 → 128 B/行 = 0.25 B/权重");

    // 位序：第 j 个权重的码在 (word >> 2j) & 3
    const float w[16] = {1.0f, -1.0f, 0.0f, 0.4f,  -0.4f, 1.0f, -1.0f, 0.0f,
                         0.9f, -0.9f, 0.1f, -0.1f, 0.6f, -0.6f, 0.0f, 0.0f};
    std::uint32_t word[1] = {0};
    pack_row(w, 16, /*tau=*/1.0f, word);
    // τ=1 → 阈值 0.5：1.0 → +1(码3)、-1.0 → -1(码1)、|w|≤0.5 → 0(码2)
    const std::uint32_t expect_codes[16] = {3, 1, 2, 2, 2, 3, 1, 2,
                                            3, 1, 2, 2, 3, 1, 2, 2};
    bool bits_ok = true;
    for (std::uint32_t j = 0; j < 16; ++j)
        if (((word[0] >> (2 * j)) & 3u) != expect_codes[j]) bits_ok = false;
    CHECK(bits_ok, "码值/位序 = 三值+2，低位对应行内靠前权重");

    // 解码往返（含尾部填充：K=5 时 idx 5..15 必须是"码 2 = 三值 0"）
    std::uint32_t w2[1] = {0};
    pack_row(w, 5, 1.0f, w2);
    bool roundtrip = true;
    for (std::size_t i = 0; i < 5; ++i)
        if (at(w2, i) != nn::trit_pack::value_of(expect_codes[i])) roundtrip = false;
    for (std::size_t i = 5; i < 16; ++i)
        if (at(w2, i) != 0.0f) roundtrip = false;   // 尾部 = 三值 0（不是 -2）
    CHECK(roundtrip, "解码往返（尾部位 = 码 2，不是未定义码 0）");
}

// ── 工具：宿主参考 GEMM（入参是**三值**（已量化），乘逐行 τ；transA=0/1）──
// GEMM 消费的是量化后的三值权重 —— 参考必须用 trit 而不是原始 latent。
void ref_gemm(const std::vector<float>& trit, const std::vector<float>& tau,
              std::size_t rows, std::size_t k, const nn::Matrix& B, bool transA,
              nn::Matrix& out)
{
    if (!transA)
    {
        // W (M=rows, K=k)，C = (τ∘trit)·B，B (K,N)
        for (std::size_t m = 0; m < rows; ++m)
            for (std::size_t n = 0; n < B.cols(); ++n)
            {
                double acc = 0.0;
                for (std::size_t kk = 0; kk < k; ++kk)
                    acc += static_cast<double>(tau[m] * trit[m * k + kk]) *
                           static_cast<double>(B.at(kk, n));
                out.set_value(m, n, static_cast<float>(acc));
            }
    }
    else
    {
        // A 存储 (K=rows, M=k)：C = (τ∘trit)^T·B，B (K,N) → out (M,N)
        for (std::size_t m = 0; m < k; ++m)
            for (std::size_t n = 0; n < B.cols(); ++n)
            {
                double acc = 0.0;
                for (std::size_t kk = 0; kk < rows; ++kk)
                    acc += static_cast<double>(tau[kk] * trit[kk * k + m]) *
                           static_cast<double>(B.at(kk, n));
                out.set_value(m, n, static_cast<float>(acc));
            }
    }
}

void test_primitives(nn::ComputeEngine& eng, const char* tag)
{
    constexpr std::size_t ROWS = 7, K = 40, N = 3;   // K 故意非 16 倍数
    std::printf("[2][3][4] trit_pack / matmul_q / matmul_q_t（%s）\n", tag);
    if (!eng.supports_ternary_gemm())
    {
        std::puts("    （该引擎尚未接三值打包原语 → 跳过；BitLinear 会自动走 f16 路径）");
        return;
    }

    // 造 latent 权重（含正负/微小值，覆盖三值三种码）
    nn::Matrix W(ROWS, K);
    for (std::size_t r = 0; r < ROWS; ++r)
        for (std::size_t c = 0; c < K; ++c)
            W.set_value(r, c, 0.13f * static_cast<float>((r * 7 + c * 3) % 11) - 0.6f);
    nn::Matrix X(K, N);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t n = 0; n < N; ++n)
            X.set_value(k, n, 0.21f * static_cast<float>((k + 2 * n) % 9) - 0.8f);

    nn::Tensor Wt = upload(eng, W);
    nn::Tensor Xt = upload(eng, X);

    // τ = 逐行 absmean（宿主侧，与被测的 DSL reduce 无关；那条路径 t1_58_test 覆盖）
    std::vector<float> tau(ROWS, 0.0f);
    std::vector<float> w_flat(ROWS * K, 0.0f);
    for (std::size_t r = 0; r < ROWS; ++r)
    {
        double s = 0.0;
        for (std::size_t c = 0; c < K; ++c) s += std::fabs(static_cast<double>(W.at(r, c)));
        tau[r] = static_cast<float>(s / static_cast<double>(K));
        for (std::size_t c = 0; c < K; ++c) w_flat[r * K + c] = W.at(r, c);
    }
    nn::Matrix tau_m(ROWS, 1);
    for (std::size_t r = 0; r < ROWS; ++r) tau_m.set_value(r, 0, tau[r]);
    nn::Tensor tau_t = upload(eng, tau_m);

    // 三值（宿主，作为后续参考与"逐位一致"的基准）
    std::vector<float> trit(ROWS * K, 0.0f);
    for (std::size_t r = 0; r < ROWS; ++r)
        for (std::size_t c = 0; c < K; ++c)
            trit[r * K + c] = nn::trit_pack::value_of(nn::trit_pack::code_of(W.at(r, c), tau[r]));

    // ── [2] trit_pack = 宿主 pack_row ───────────────────────────────────
    auto packed_r = eng.trit_pack(Wt, tau_t);
    CHECK(packed_r.has_value(), "trit_pack 成功");
    if (!packed_r.has_value()) return;
    nn::Tensor packed = std::move(*packed_r);
    const std::size_t words = nn::trit_pack::words_per_row(K);
    CHECK(packed.rows() == ROWS && packed.cols() == words,
          "打包容器形状 (rows, ceil(K/16))");
    {
        const nn::Matrix pm = download(eng, packed);
        bool same = true;
        std::vector<std::uint32_t> ref(words);
        for (std::size_t r = 0; r < ROWS && same; ++r)
        {
            nn::trit_pack::pack_row(w_flat.data() + r * K, K, tau[r], ref.data());
            for (std::size_t i = 0; i < words; ++i)
                if (std::bit_cast<std::uint32_t>(pm.at(r, i)) != ref[i]) same = false;
        }
        CHECK(same, "引擎 trit_pack 与宿主 pack_row 逐位一致");
        bool trit_ok = true;
        for (std::size_t r = 0; r < ROWS && trit_ok; ++r)
        {
            std::vector<std::uint32_t> row(words);
            for (std::size_t i = 0; i < words; ++i)
                row[i] = std::bit_cast<std::uint32_t>(pm.at(r, i));
            for (std::size_t c = 0; c < K; ++c)
                if (nn::trit_pack::at(row.data(), c) != trit[r * K + c]) trit_ok = false;
        }
        CHECK(trit_ok, "解码三值与 RoundClip(W/τ,-1,1) 逐位一致");
    }

    // ── [3] matmul_q（transA=0）：宿主参考 = 原语 ───────────────────────
    {
        nn::Matrix ref(ROWS, N);
        ref_gemm(trit, tau, ROWS, K, X, /*transA=*/false, ref);
        auto got = eng.matmul_q(packed, tau_t, Xt);
        CHECK(got.has_value(), "matmul_q 成功");
        if (got.has_value())
        {
            const nn::Matrix g = download(eng, *got);
            double mx = 0.0;
            for (std::size_t r = 0; r < ROWS; ++r)
                for (std::size_t n = 0; n < N; ++n)
                    mx = std::max(mx, std::fabs(static_cast<double>(g.at(r, n)) -
                                                static_cast<double>(ref.at(r, n))));
            CHECK(mx < 1e-5, "matmul_q = 宿主参考（τ∘W 去量化）");
            // 与"f16 稠密缓冲路径"（matmul(wq,·)·τ）等价
            nn::Matrix wq_m(ROWS, K);
            for (std::size_t r = 0; r < ROWS; ++r)
                for (std::size_t c = 0; c < K; ++c) wq_m.set_value(r, c, trit[r * K + c]);
            nn::Tensor wq_t = upload(eng, wq_m, nn::Precision::F16);
            auto dense = eng.matmul(wq_t, Xt, false, false, nn::Precision::F32);
            CHECK(dense.has_value(), "稠密 wq matmul 成功");
            if (dense.has_value())
            {
                const nn::Matrix d = download(eng, *dense);
                double dmax = 0.0;
                for (std::size_t r = 0; r < ROWS; ++r)
                    for (std::size_t n = 0; n < N; ++n)
                        dmax = std::max(dmax,
                            std::fabs(static_cast<double>(d.at(r, n)) * static_cast<double>(tau[r]) -
                                      static_cast<double>(ref.at(r, n))));
                CHECK(dmax < 1e-5, "f16 稠密路径（P1）同一口径（三值逐位一致）");
            }
        }
    }

    // ── [4] matmul_q_t（transA=1）：A 存为 (K,M)，输出 (M,N) ────────────
    {
        // A 存为 (K, M)：用 W 的转置形态（rows=K, M=ROWS）—— τ 按存储行取
        const std::size_t M = ROWS;
        std::vector<float> tau_k(K, 0.5f);
        std::vector<float> a_flat(K * M, 0.0f);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t m = 0; m < M; ++m)
                a_flat[k * M + m] = 0.19f * static_cast<float>((k * 5 + m * 3) % 7) - 0.5f;
        nn::Matrix A(K, M);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t m = 0; m < M; ++m) A.set_value(k, m, a_flat[k * M + m]);
        nn::Matrix tau_k_m(K, 1);
        for (std::size_t k = 0; k < K; ++k) tau_k_m.set_value(k, 0, tau_k[k]);
        nn::Tensor At = upload(eng, A);
        nn::Tensor tau_kt = upload(eng, tau_k_m);
        auto pk2_r = eng.trit_pack(At, tau_kt);
        CHECK(pk2_r.has_value(), "trit_pack（transA 形态）成功");
        if (pk2_r.has_value())
        {
            nn::Tensor pk2 = std::move(*pk2_r);
            nn::Matrix ref(M, N);
            std::vector<float> trit_a(K * M, 0.0f);
            for (std::size_t k = 0; k < K; ++k)
                for (std::size_t m = 0; m < M; ++m)
                    trit_a[k * M + m] = nn::trit_pack::value_of(
                        nn::trit_pack::code_of(a_flat[k * M + m], tau_k[k]));
            ref_gemm(trit_a, tau_k, K, M, X, /*transA=*/true, ref);
            auto got = eng.matmul_q_t(pk2, tau_kt, Xt, M);
            CHECK(got.has_value(), "matmul_q_t 成功");
            if (got.has_value())
            {
                const nn::Matrix g = download(eng, *got);
                CHECK(g.rows() == M && g.cols() == N, "输出形状 (M,N)");
                double mx = 0.0;
                for (std::size_t m = 0; m < M; ++m)
                    for (std::size_t n = 0; n < N; ++n)
                        mx = std::max(mx, std::fabs(static_cast<double>(g.at(m, n)) -
                                                    static_cast<double>(ref.at(m, n))));
                CHECK(mx < 1e-5, "matmul_q_t = 宿主参考（(τ∘A)^T·B）");
            }
        }
    }
}

// ── [5] BitLinear 端到端：打包路径 vs f16 稠密路径（同 latent）───────────
void test_bitlinear_paths(nn::ComputeEngine& eng, const char* tag)
{
    constexpr std::size_t IN = 24, OUT = 9, B = 4;   // IN 非 16 倍数：覆盖尾部
    std::printf("[5] BitLinear：打包路径 vs f16 稠密路径（%s）\n", tag);
    if (!eng.supports_ternary_gemm())
    {
        std::puts("    （引擎无打包能力 → 跳过；该引擎上 BitLinear 自动走 f16 路径）");
        return;
    }

    nn::Matrix W(OUT, IN);
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t k = 0; k < IN; ++k)
            W.set_value(o, k, 0.17f * static_cast<float>((o * 5 + k * 3) % 13) - 0.9f);
    nn::Matrix X(IN, B);
    for (std::size_t k = 0; k < IN; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, 0.23f * static_cast<float>((k + b) % 7) - 0.7f);

    nn::PrecisionProfile prof;
    prof.param = nn::Precision::T1_58;

    auto make = [&](bool packed) {
        auto bl = std::make_unique<nn::BitLinear>(IN, OUT);
        bl->set_precision_profile(prof);
        auto r = bl->init(eng);
        NN_ASSERT(r, "BitLinear init 失败");
        auto wr = eng.copy_from(bl->parameters()[0].get(), W);
        NN_ASSERT(wr, "写入 latent 失败");
        bl->set_ternary_pack_enabled(packed);
        return bl;
    };

    auto pk = make(true);
    auto dn = make(false);
    CHECK(pk->uses_packed_weights(), "打包路径已启用");
    CHECK(!dn->uses_packed_weights(), "稠密路径已强制退回");

    nn::Tensor Xt = upload(eng, X);
    auto y_pk = pk->forward(Xt);
    auto y_dn = dn->forward(Xt);
    CHECK(y_pk.has_value() && y_dn.has_value(), "两条路径 forward 都成功");
    if (y_pk.has_value() && y_dn.has_value())
    {
        const nn::Matrix a = download(eng, *y_pk), b = download(eng, *y_dn);
        double mx = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t c = 0; c < B; ++c)
                mx = std::max(mx, std::fabs(static_cast<double>(a.at(o, c)) -
                                            static_cast<double>(b.at(o, c))));
        CHECK(mx < 1e-5, "前向：打包 = 稠密（f32 累加顺序差之内）");
    }

    // 反向：dX 与 dW 两路对比（同一上游梯度）
    auto make_grad = [&] {
        nn::Matrix G(OUT, B);
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t c = 0; c < B; ++c)
                G.set_value(o, c, 0.31f * static_cast<float>((o + 2 * c) % 5) - 0.5f);
        return G;
    };
    const nn::Matrix G = make_grad();
    auto gx_pk = pk->backward(upload(eng, G));
    auto gx_dn = dn->backward(upload(eng, G));
    CHECK(gx_pk.has_value() && gx_dn.has_value(), "两条路径 backward 都成功");
    if (gx_pk.has_value() && gx_dn.has_value())
    {
        const nn::Matrix a = download(eng, *gx_pk), b = download(eng, *gx_dn);
        double mx = 0.0;
        for (std::size_t k = 0; k < IN; ++k)
            for (std::size_t c = 0; c < B; ++c)
                mx = std::max(mx, std::fabs(static_cast<double>(a.at(k, c)) -
                                            static_cast<double>(b.at(k, c))));
        CHECK(mx < 1e-5, "dX：打包 = 稠密");
    }
    {
        const nn::Matrix a = download(eng, pk->param_gradients()[0].get());
        const nn::Matrix b = download(eng, dn->param_gradients()[0].get());
        double mx = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
                mx = std::max(mx, std::fabs(static_cast<double>(a.at(o, k)) -
                                            static_cast<double>(b.at(o, k))));
        CHECK(mx < 1e-6, "dW：打包 = 稠密（同一 STE 公式，与缓冲形态无关）");
    }
}

// ── [6] 跨形状对拍：分块内核的"形状契约"（BM=BN=64 / BK=16）───────────────
// 分块内核的失效模式**只在特定形状下暴露**，方形/极小形状会掩盖它们：
//   1. **轴向**：dispatch 的 x/y 必须与 shader 的 block_col/block_row 同约定。
//      写反 → 只算到 ceil(N/64)*64 行 / ceil(M/64)*64 列；M=N≤64 的方形恰好
//      正确，于是"小形状全绿、真实形状炸"（分块版 matmul_q 第一轮尝试即如此）。
//   2. **尾块**：M/N 非 64 倍数、K 非 16 倍数（trit 越界必须判 0 = 三值 0）。
//   3. **单列**：N=1（自回归解码形态；列块大面积空转，网格与守卫都必须对）。
// 覆盖 transA=0/1 × f32/f16 存储四种组合。
void check_gemm_shape(nn::ComputeEngine& eng, std::size_t M, std::size_t K,
                      std::size_t N, bool transA, bool f16_mode)
{
    const std::size_t sr = transA ? K : M;   // A 的存储行数
    const std::size_t al = transA ? M : K;   // A 的存储行长（= 一行三值个数）

    std::vector<float> a_flat(sr * al, 0.0f);
    for (std::size_t r = 0; r < sr; ++r)
        for (std::size_t c = 0; c < al; ++c)
            a_flat[r * al + c] = 0.17f * static_cast<float>((r * 7 + c * 3) % 13) - 0.9f;
    nn::Matrix A_lat(sr, al);
    for (std::size_t r = 0; r < sr; ++r)
        for (std::size_t c = 0; c < al; ++c) A_lat.set_value(r, c, a_flat[r * al + c]);

    // τ = 逐存储行 absmean（宿主；DSL 那条归约由 t1_58_test 覆盖）
    std::vector<float> tau(sr, 0.0f);
    for (std::size_t r = 0; r < sr; ++r)
    {
        double s = 0.0;
        for (std::size_t c = 0; c < al; ++c) s += std::fabs(a_flat[r * al + c]);
        tau[r] = static_cast<float>(s / static_cast<double>(al));
    }
    nn::Matrix tau_m(sr, 1);
    for (std::size_t r = 0; r < sr; ++r) tau_m.set_value(r, 0, tau[r]);

    std::vector<float> trit(sr * al, 0.0f);
    for (std::size_t r = 0; r < sr; ++r)
        for (std::size_t c = 0; c < al; ++c)
            trit[r * al + c] = nn::trit_pack::value_of(
                nn::trit_pack::code_of(a_flat[r * al + c], tau[r]));

    // B (K,N)：f16 存储时宿主参考必须用**同一批** f16 舍入值，否则容差里混进
    // "输入表示差"（既会把真实错误判成噪声、也会把噪声判成错误）。
    nn::Matrix Bm(K, N);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t n = 0; n < N; ++n)
        {
            float v = 0.21f * static_cast<float>((k + 2 * n) % 9) - 0.8f;
            if (f16_mode) v = static_cast<float>(nn::f16(v));
            Bm.set_value(k, n, v);
        }

    nn::Matrix ref(M, N);
    ref_gemm(trit, tau, sr, al, Bm, transA, ref);
    if (f16_mode)
        for (std::size_t m = 0; m < M; ++m)
            for (std::size_t n = 0; n < N; ++n)
                ref.set_value(m, n, static_cast<float>(nn::f16(ref.at(m, n))));

    const nn::Precision P = f16_mode ? nn::Precision::F16 : nn::Precision::F32;
    const nn::Tensor At = upload(eng, A_lat);
    const nn::Tensor tau_t = upload(eng, tau_m);
    const nn::Tensor Bt = upload(eng, Bm, P);

    auto packed_r = eng.trit_pack(At, tau_t);
    CHECK(packed_r.has_value(), "跨形状：trit_pack 成功");
    if (!packed_r.has_value()) return;

    auto got_r = transA ? eng.matmul_q_t(*packed_r, tau_t, Bt, M, P)
                        : eng.matmul_q(*packed_r, tau_t, Bt, P);
    CHECK(got_r.has_value(), "跨形状：matmul_q(_t) 成功");
    if (!got_r.has_value()) return;
    CHECK(got_r->rows() == M && got_r->cols() == N, "跨形状：输出形状 (M,N)");

    const nn::Matrix got = download(eng, *got_r);
    double emax = 0.0, rmax = 0.0;
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t n = 0; n < N; ++n)
        {
            emax = std::max(emax, std::fabs(static_cast<double>(got.at(m, n)) -
                                            static_cast<double>(ref.at(m, n))));
            rmax = std::max(rmax, std::fabs(static_cast<double>(ref.at(m, n))));
        }
    // 容差按**全局幅度**定标：三值点积可正负相消，逐元素相对判据在近零点爆炸
    const double scale = std::max(1.0, rmax);
    const double tol = f16_mode ? 4e-3 : 1e-4;
    std::printf("    M=%-5zu K=%-5zu N=%-4zu transA=%u %-3s max_err=%.3e\n",
                M, K, N, transA ? 1u : 0u, f16_mode ? "f16" : "f32", emax);
    CHECK(emax <= tol * scale, "跨形状：matmul_q(_t) = 宿主参考");
}

void test_shapes(nn::ComputeEngine& eng, const char* tag, bool gpu)
{
    std::printf("[6] 跨形状对拍：分块内核形状契约（%s）\n", tag);
    if (!eng.supports_ternary_gemm())
    {
        std::puts("    （引擎无打包能力 → 跳过）");
        return;
    }
    struct Sh { std::size_t M, K, N; };
    const Sh base[] = {
        {64, 16, 64},    // 单块方形（旧极小形状 = 掩盖轴向缺陷的那一类）
        {128, 32, 64},   // 行 2 块 / 列 1 块
        {64, 32, 128},   // 行 1 块 / 列 2 块 —— 轴向对拍
        {192, 48, 320},  // 行 3 块 / 列 5 块
        {65, 17, 65},    // M/N/K 全尾块
        {129, 33, 7},    // 行尾块 + 列单块
        {256, 40, 1},    // N=1 解码形态（K 非 16 倍数）
        {9, 100, 3},     // 小 M + 长行尾块
        {130, 16, 130},  // 方形但两向都跨块边界（尾块 + 多块同时）
    };
    for (const Sh& s : base)
        for (int ta = 0; ta < 2; ++ta)
            for (int h = 0; h < 2; ++h)
                check_gemm_shape(eng, s.M, s.K, s.N, ta != 0, h != 0);

    // 真实规模一档（仅 GPU；宿主参考是 double 逐元素，CPU 上太慢）：
    // 2B4T 的行/列量级 —— M≫N 正是"轴向写反"最刺眼的形态。
    if (gpu)
    {
        const Sh big[] = {
            {2048, 2560, 64},   // lm_head / fc2 量级（M≫N）
            {2560, 2048, 64},   // 转置形态（A 存 (K,M)，M≫N）
        };
        for (const Sh& s : big)
            for (int ta = 0; ta < 2; ++ta)
                check_gemm_shape(eng, s.M, s.K, s.N, ta != 0, /*f16_mode=*/false);
    }
}

// ── [7] BitLinear 跨形状：打包 vs 稠密（多块 + 尾块 + 两种 profile）───────
// [5] 用的是 OUT=9/IN=24/B=4（两向都只有 1 块）→ 天然测不出分块/网格缺陷。
// 这里把形状撑过块边界，并**同时覆盖 f32 与 f16 compute**（后者才是训练路径：
// 2B4T 形状的 GPT 训练发散正发生在 f16 compute 的打包内核上）。
void check_bitlinear_shape(nn::ComputeEngine& eng, std::size_t IN, std::size_t OUT,
                           std::size_t B, const nn::PrecisionProfile& prof)
{
    nn::Matrix W(OUT, IN);
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t k = 0; k < IN; ++k)
            W.set_value(o, k, 0.17f * static_cast<float>((o * 5 + k * 3) % 13) - 0.9f);
    nn::Matrix X(IN, B);
    for (std::size_t k = 0; k < IN; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, 0.23f * static_cast<float>((k + b) % 7) - 0.7f);

    const nn::Precision P =
        (prof.compute == nn::Precision::F16) ? nn::Precision::F16 : nn::Precision::F32;
    auto make = [&](bool packed) {
        auto bl = std::make_unique<nn::BitLinear>(IN, OUT);
        bl->set_precision_profile(prof);
        auto r = bl->init(eng);
        NN_ASSERT(r, "BitLinear init 失败");
        auto wr = eng.copy_from(bl->parameters()[0].get(), W);
        NN_ASSERT(wr, "写入 latent 失败");
        bl->set_ternary_pack_enabled(packed);
        return bl;
    };
    auto pk = make(true);
    auto dn = make(false);
    CHECK(pk->uses_packed_weights() && !dn->uses_packed_weights(), "两条路径形态正确");

    nn::Tensor Xt = upload(eng, X, P);
    auto y_pk = pk->forward(Xt);
    auto y_dn = dn->forward(Xt);
    CHECK(y_pk.has_value() && y_dn.has_value(), "两条路径 forward 都成功");
    if (!y_pk.has_value() || !y_dn.has_value()) return;

    auto cmp = [&](const nn::Tensor& a, const nn::Tensor& b, const char* what) {
        const nn::Matrix am = download(eng, a), bm = download(eng, b);
        const auto pa = am.span();
        const auto pb = bm.span();
        double emax = 0.0, rmax = 0.0;
        for (std::size_t i = 0; i < pa.size(); ++i)
        {
            emax = std::max(emax, std::fabs(static_cast<double>(pa[i]) - pb[i]));
            rmax = std::max(rmax, std::fabs(static_cast<double>(pb[i])));
        }
        const double scale = std::max(1.0, rmax);
        const double tol = (P == nn::Precision::F16) ? 4e-3 : 1e-5;
        std::printf("    [%s] IN=%-4zu OUT=%-4zu B=%-4zu max_err=%.3e\n",
                    what, IN, OUT, B, emax);
        CHECK(emax <= tol * scale, "BitLinear 跨形状：打包 = 稠密");
    };
    cmp(*y_pk, *y_dn, "fwd");

    nn::Matrix G(OUT, B);
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t c = 0; c < B; ++c)
            G.set_value(o, c, 0.31f * static_cast<float>((o + 2 * c) % 5) - 0.5f);
    nn::Tensor Gt = upload(eng, G, P);
    auto gx_pk = pk->backward(Gt);
    auto gx_dn = dn->backward(Gt);
    CHECK(gx_pk.has_value() && gx_dn.has_value(), "两条路径 backward 都成功");
    if (gx_pk.has_value() && gx_dn.has_value()) cmp(*gx_pk, *gx_dn, "dX ");
    cmp(pk->param_gradients()[0].get(), dn->param_gradients()[0].get(), "dW ");
}

void test_bitlinear_shapes(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[7] BitLinear 跨形状：打包 vs 稠密（%s）\n", tag);
    if (!eng.supports_ternary_gemm())
    {
        std::puts("    （引擎无打包能力 → 跳过）");
        return;
    }
    nn::PrecisionProfile p32;
    p32.param = nn::Precision::T1_58;
    const nn::PrecisionProfile profs[2] = {p32, nn::profile_t1_58()};
    const char* ptags[2] = {"t1_58/f32", "t1_58/f16"};
    struct Sh { std::size_t IN, OUT, B; };
    const Sh shapes[] = {
        {70, 150, 100},   // 两向都多块 + 尾块
        {256, 64, 9},     // 窄 N（解码形态）+ K 整块
        {17, 65, 520},    // 宽 N + 行尾块
    };
    for (int p = 0; p < 2; ++p)
    {
        std::printf("  ── profile %s ──\n", ptags[p]);
        for (const Sh& s : shapes)
            check_bitlinear_shape(eng, s.IN, s.OUT, s.B, profs[p]);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    test_layout();
    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        test_primitives(**eng_r, "GPU");
        test_bitlinear_paths(**eng_r, "GPU");
        test_shapes(**eng_r, "GPU", /*gpu=*/true);
        test_bitlinear_shapes(**eng_r, "GPU");
    }
    else
    {
        nn::CpuEngine eng;
        test_primitives(eng, "CPU");
        test_bitlinear_paths(eng, "CPU");
        test_shapes(eng, "CPU", /*gpu=*/false);
        test_bitlinear_shapes(eng, "CPU");
    }

    if (g_failures == 0)
        std::puts("ternary_gemm_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
