// ── poc_binary_ste.cpp — 纯 1-bit（二值）权重的 STE 训练最小探针 ──────────────
// 配套报告：research/ternary_ste/REPORT.md；设计文档：docs/development/21-quantized-weights.md
//
// 与 poc_ternary_ste.cpp 的唯一区别：量化器是 sign(W)（值域 {-1,+1}，没有 0），
// 教师也是二值 {-0.5,+0.5}。用于回答"纯 1-bit 是否同样可行"。
//
// 复现（仓库根目录）：
//   clang++ -std=c++26 -O2 -march=native -fno-exceptions -Wno-pass-failed \
//     -Iinclude -o /tmp/poc_binary_ste research/ternary_ste/poc_binary_ste.cpp -pthread
//   /tmp/poc_binary_ste
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
    std::uniform_int_distribution<int> td(0, 1);

    // ── 教师：纯二值权重 ±0.5 ──
    Matrix Wt(OUT, IN);
    for (std::size_t r = 0; r < OUT; ++r)
        for (std::size_t c = 0; c < IN; ++c)
            Wt.set_value(r, c, (td(rng) == 0 ? -0.5f : 0.5f));

    Matrix X(IN, NB), Y(OUT, NB);
    for (auto& v : X.span()) v = nd(rng);
    Wt.multiply_to(Y, X);

    Tensor Xt = NN_CHECK(eng.from_matrix(X));
    Tensor Yt = NN_CHECK(eng.from_matrix(Y));
    Tensor W = eng.create_tensor(OUT, IN, Precision::F32, InitSpec::uniform(-0.5, 0.5, 7));

    Scalar first_loss = 0.f, last_loss = 0.f;

    for (int step = 0; step < STEPS; ++step)
    {
        const Matrix Wh = NN_CHECK(eng.to_matrix(W, Precision::F32));
        Scalar gamma = 0.f;
        for (auto v : Wh.span()) gamma += std::fabs(v);
        gamma /= static_cast<Scalar>(IN * OUT);

        // Wq = sign(W) —— 纯 1-bit（二值）量化
        Tensor Wq = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::select(nn::dsl::leaf(W) > nn::dsl::rparam(0.f),
                Scalar{1}, Scalar{-1}),
            OUT, IN, Precision::F32));

        Tensor Yhat = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::matmul(Wq, Xt) * nn::dsl::rparam(gamma),
            OUT, NB, Precision::F32));

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
        Tensor dW = NN_CHECK(nn::dsl::compute(eng,
            nn::dsl::matmul(dY, Xt, false, true) * nn::dsl::rparam(gamma),
            OUT, IN, Precision::F32));
        NN_CHECK(nn::dsl::compute_into(eng,
            nn::dsl::leaf(W) - nn::dsl::leaf(dW) * nn::dsl::rparam(lr), W));

        if (step % 500 == 0 || step == STEPS - 1)
            std::printf("step %4d  mse=%.6f  gamma=%.5f\n", step, loss, gamma);
    }

    const Matrix Wh = NN_CHECK(eng.to_matrix(W, Precision::F32));
    std::size_t match = 0;
    for (std::size_t r = 0; r < OUT; ++r)
        for (std::size_t c = 0; c < IN; ++c)
            if ((Wh.at(r, c) > 0.f) == (Wt.at(r, c) > 0.f))
                ++match;

    std::printf("\nmse: first=%.6f  final=%.6f  (降 %.1f%%)\n",
                first_loss, last_loss, 100.0 * (1.0 - last_loss / first_loss));
    std::printf("二值符号吻合率: %zu / %zu = %.1f%%\n",
                match, IN * OUT, 100.0 * static_cast<double>(match) / double(IN * OUT));
    return 0;
}
