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
    }
    else
    {
        nn::CpuEngine eng;
        test_primitives(eng, "CPU");
        test_bitlinear_paths(eng, "CPU");
    }

    if (g_failures == 0)
        std::puts("ternary_gemm_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
