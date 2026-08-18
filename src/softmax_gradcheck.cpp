// ── Softmax 数值梯度检查（gradcheck） ─────────────────────────────────────
//
// 目的：attention gradcheck 显示 grad_Q/grad_K FAIL 但 grad_V OK，
//       共同差异是 grad_S = softmax.backward(grad_A)。本测试单独验证
//       Softmax 的 forward/backward 是否与中心差分一致。
//
// 用法：softmax_gradcheck [--cuda|--gpu] [--rows N] [--cols N] [--tol <f>]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::Softmax;

namespace {

Scalar dot(const Matrix& a, const Matrix& b)
{
    Scalar s{0};
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i) s += sa[i] * sb[i];
    return s;
}

Scalar eval_loss(ComputeEngine& engine, Softmax& sm, const Tensor& x, const Tensor& go)
{
    auto y = sm.forward(engine, x);
    auto y_m = engine.to_matrix(*y);
    auto go_m = engine.to_matrix(go);
    return dot(*y_m, *go_m);
}

bool approx(Scalar num, Scalar ana, Scalar tol)
{
    return std::fabs(num - ana) <= tol * (Scalar{1} + std::fabs(num) + std::fabs(ana));
}

bool check_input_grad(ComputeEngine& engine, Softmax& sm,
                      const Tensor& go, const Tensor& grad_x,
                      const Matrix& x_m, Scalar eps, Scalar tol)
{
    bool ok = true;
    std::size_t bad = 0;
    Scalar max_err{0};
    auto gxm = engine.to_matrix(grad_x);
    for (std::size_t r = 0; r < x_m.rows(); ++r)
        for (std::size_t c = 0; c < x_m.cols(); ++c)
        {
            const Scalar orig = x_m.at_unchecked(r, c);
            Matrix pp = x_m;
            pp.set_value_unchecked(r, c, orig + eps);
            auto xt = engine.from_matrix(pp);
            const Scalar lp = eval_loss(engine, sm, *xt, go);
            Matrix pm = x_m;
            pm.set_value_unchecked(r, c, orig - eps);
            auto xt2 = engine.from_matrix(pm);
            const Scalar lm = eval_loss(engine, sm, *xt2, go);
            const Scalar num = (lp - lm) / (Scalar{2} * eps);
            const Scalar ana = gxm->at_unchecked(r, c);
            const Scalar err = std::fabs(num - ana);
            if (err > max_err) max_err = err;
            if (!approx(num, ana, tol))
            {
                ok = false;
                if (bad < 8)
                    std::cout << "  [FAIL] (" << r << "," << c << ") num=" << num
                              << " ana=" << ana << " err=" << err << "\n";
                ++bad;
            }
        }
    std::cout << "  grad_x: " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
    return ok;
}

} // namespace

int main(int argc, char* argv[])
{
    Scalar tol = 1e-2f;
    bool use_cuda = false;
    bool use_gpu = false;
    std::size_t rows = 8;
    std::size_t cols = 8;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--rows" && i + 1 < argc) rows = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--cols" && i + 1 < argc) cols = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--cuda") use_cuda = true;
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: softmax_gradcheck [--cuda|--gpu] [--rows N] [--cols N] [--tol <f>]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_cuda = use_cuda;
    ecfg.use_gpu = use_gpu;
    auto engine = nn::cli::create_engine(ecfg, std::cout);
    if (!engine) { std::cerr << "引擎创建失败\n"; return 1; }
    ComputeEngine& eng = *engine;

    std::cout << "========================================\n";
    std::cout << "  Softmax 数值梯度检查 (gradcheck)\n";
    std::cout << "========================================\n";
    std::cout << "  rows=" << rows << " cols=" << cols << " tol=" << tol << "\n";

    Softmax sm;
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-2, 2);
    Matrix x_m(rows, cols);
    Matrix go_m(rows, cols);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = dist(rng);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    auto go = eng.from_matrix(go_m);
    auto y_fwd = sm.forward(eng, *x);
    auto gx = sm.backward(eng, *go);

    const Scalar eps = 1e-3f;
    bool all_pass = check_input_grad(eng, sm, *go, *gx, x_m, eps, tol);
    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return all_pass ? 0 : 1;
}
