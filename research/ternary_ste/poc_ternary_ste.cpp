// ── poc_ternary_ste.cpp — 三值（1.58-bit）权重的 STE 训练最小探针 ─────────────
// 配套报告：research/ternary_ste/REPORT.md；设计文档：docs/development/21-quantized-weights.md
//
// 目的：用**现有公开 API**（无库改动）证明"Layer 级三值量化 + STE backward"在本引擎上可行。
// 任务：学一个教师三元线性映射 y = W_t · x（W_t ∈ {-0.5, 0, +0.5}），训练后逐项比对教师模式。
//
// 复现（仓库根目录）：
//   clang++ -std=c++26 -O2 -march=native -fno-exceptions -Wno-pass-failed \
//     -Iinclude -o /tmp/poc_ternary_ste research/ternary_ste/poc_ternary_ste.cpp -pthread
//   /tmp/poc_ternary_ste
//
// 注意：本探针在宿主侧读回 latent 权重算 γ（absmean）。这在 src/ 层合法；
// 落成 Layer（L2）时必须改用引擎归约（铁律 #12：L2+ 禁用 Matrix）。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <random>

using nn::InitSpec;
using nn::Matrix;
using nn::Precision;
using nn::Scalar;
using nn::Tensor;

int main()
{
    nn::CpuEngine eng;

    constexpr std::size_t IN = 64, OUT = 32, NB = 512;
    constexpr Scalar lr = 0.5f;
    constexpr int STEPS = 3000;

    std::mt19937 rng(20261009u);
    std::normal_distribution<Scalar> nd(0.f, 1.f);
    std::uniform_int_distribution<int> td(0, 2);

    // ── 教师：三元权重（-0.5 / 0 / +0.5 等概率）线性映射 y = W_t · x ──
    Matrix Wt(OUT, IN);
    for (std::size_t r = 0; r < OUT; ++r)
        for (std::size_t c = 0; c < IN; ++c)
            Wt.set_value(r, c, static_cast<Scalar>(td(rng) - 1) * 0.5f);

    Matrix X(IN, NB), Y(OUT, NB);
    for (auto& v : X.span()) v = nd(rng);
    Wt.multiply_to(Y, X);

    Tensor Xt = NN_CHECK(eng.from_matrix(X));
    Tensor Yt = NN_CHECK(eng.from_matrix(Y));

    // ── 可训练对象：隐空间（latent）f32 权重 —— 优化器与 STE 的作用点 ──
    Tensor W = eng.create_tensor(OUT, IN, Precision::F32, InitSpec::uniform(-0.5, 0.5, 7));

    Scalar first_loss = 0.f, last_loss = 0.f;

    for (int step = 0; step < STEPS; ++step)
    {
        // γ = mean|W|。研究口径：宿主读回算（等价于一次全局归约）；
        // Layer 化时改为 dsl 的 row_reduce_sum + row_broadcast（逐行 absmean）。
        const Matrix Wh = NN_CHECK(eng.to_matrix(W, Precision::F32));
        Scalar gamma = 0.f;
        for (auto v : Wh.span()) gamma += std::fabs(v);
        gamma /= static_cast<Scalar>(IN * OUT);
        const Scalar thr = 0.5f * gamma;

        // Wq = select(W > thr, +1, select(W < -thr, -1, 0)) —— 纯 DSL 三值量化
        // （DSL 没有 round/floor；在 [-1,1] 区间内这条 select 链等价于 RoundClip(w/γ, -1, 1)）
        Tensor Wq = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::select(nn::dsl::leaf(W) > nn::dsl::rparam(thr),
                Scalar{1},
                nn::dsl::select(nn::dsl::leaf(W) < nn::dsl::rparam(-thr),
                    Scalar{-1}, Scalar{0})),
            OUT, IN, Precision::F32));

        // Ŷ = γ · (Wq · X)（matmul 段 + 缩放融合为单 kernel）
        Tensor Yhat = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::matmul(Wq, Xt) * nn::dsl::rparam(gamma),
            OUT, NB, Precision::F32));

        // MSE + 其雅可比 dY = 2(Ŷ - Y)/(OUT·NB)
        const Matrix Yh = NN_CHECK(eng.to_matrix(Yhat, Precision::F32));
        Scalar loss = 0.f;
        for (std::size_t i = 0; i < Yh.size(); ++i)
        {
            const Scalar d = Yh.span()[i] - Y.span()[i];
            loss += d * d;
        }
        loss /= static_cast<Scalar>(Yh.size());
        if (step == 0) first_loss = loss;
        last_loss = loss;

        Tensor dY = NN_CHECK(nn::dsl::compute(eng,
            (nn::dsl::leaf(Yhat) - nn::dsl::leaf(Yt)) * nn::dsl::rparam(Scalar{2} / Scalar(OUT * NB)),
            OUT, NB, Precision::F32));

        // STE：把 Wq 当常数，梯度直接穿回 latent W（dW = γ · dY · Xᵀ）
        Tensor dW = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::matmul(dY, Xt, false, true) * nn::dsl::rparam(gamma),
            OUT, IN, Precision::F32));

        NN_CHECK(nn::dsl::compute_into(eng,
            nn::dsl::leaf(W) - nn::dsl::leaf(dW) * nn::dsl::rparam(lr), W));

        if (step % 500 == 0 || step == STEPS - 1)
            std::printf("step %4d  mse=%.6f  gamma=%.5f\n", step, loss, gamma);
    }

    // ── 验收：量化后的符号/零帽与教师三元模式的重合度 ──
    const Matrix Wh = NN_CHECK(eng.to_matrix(W, Precision::F32));
    Scalar gamma = 0.f;
    for (auto v : Wh.span()) gamma += std::fabs(v);
    gamma /= static_cast<Scalar>(IN * OUT);

    std::size_t nz = 0, match = 0;
    for (std::size_t r = 0; r < OUT; ++r)
        for (std::size_t c = 0; c < IN; ++c)
        {
            const Scalar t = Wt.at(r, c);
            if (t == 0.f) continue;
            ++nz;
            const Scalar w = Wh.at(r, c);
            Scalar q = 0.f;
            if (w > 0.5f * gamma) q = 1.f;
            else if (w < -0.5f * gamma) q = -1.f;
            if ((q > 0.f && t > 0.f) || (q < 0.f && t < 0.f) || q == 0.f)
                ++match;
        }

    std::printf("\nmse: first=%.6f  final=%.6f  (降 %.1f%%)\n",
                first_loss, last_loss, 100.0 * (1.0 - last_loss / first_loss));
    std::printf("教师非零项符号/零帽吻合率: %zu / %zu = %.1f%%\n",
                match, nz, 100.0 * static_cast<double>(match) / static_cast<double>(nz));
    return 0;
}
