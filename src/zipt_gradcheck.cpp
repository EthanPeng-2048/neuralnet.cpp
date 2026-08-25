// ── AttnZip（ZiPT）数值梯度检查（gradcheck） ──────────────────────────────
//
// 验证阶段一 CrossAttention（记忆查询 P / w_k / w_v）与阶段二 ZiPTBlock
// （局部 w_q/w_k/w_v/w_o、记忆 w_kc/w_vc、Norm、FFN）的反向梯度正确性。
//
// 用法：zipt_gradcheck [--batch N] [--tol <f>] [--gpu] [--cuda]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::TensorRef;
using nn::ComputeEngine;
using nn::CrossAttention;
using nn::ZiPTBlock;

namespace {

Scalar dot(const Matrix& a, const Matrix& b)
{
    Scalar s{0};
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
        s += sa[i] * sb[i];
    return s;
}

bool approx(Scalar num, Scalar ana, Scalar tol)
{
    return std::fabs(num - ana) <=
           tol * (Scalar{1} + std::fabs(num) + std::fabs(ana));
}

// 对单个参数张量做中心差分梯度检查。
// eval_loss：在参数被 perturb 后求标量损失的闭包；base/ana 为参数基准与解析梯度。
bool check_param(ComputeEngine& engine,
                 const std::function<Scalar()>& eval_loss,
                 Tensor& param, const Matrix& base, const Matrix& ana,
                 const std::string& name, Scalar eps, Scalar tol)
{
    bool ok = true;
    std::size_t bad = 0;
    Scalar max_err{0};
    for (std::size_t r = 0; r < base.rows(); ++r)
    {
        for (std::size_t c = 0; c < base.cols(); ++c)
        {
            const Scalar orig = base.at_unchecked(r, c);
            Matrix pp = base;
            pp.set_value_unchecked(r, c, orig + eps);
            if (!engine.copy_from(param, pp)) return false;
            const Scalar lp = eval_loss();
            Matrix pm = base;
            pm.set_value_unchecked(r, c, orig - eps);
            if (!engine.copy_from(param, pm)) return false;
            const Scalar lm = eval_loss();
            if (!engine.copy_from(param, base)) return false;
            const Scalar num = (lp - lm) / (Scalar{2} * eps);
            const Scalar ana_v = ana.at_unchecked(r, c);
            const Scalar err = std::fabs(num - ana_v);
            if (err > max_err) max_err = err;
            if (!approx(num, ana_v, tol))
            {
                ok = false;
                if (bad < 5)
                    std::cout << "  [FAIL] " << name << "(" << r << "," << c
                              << ") num=" << num << " ana=" << ana_v
                              << " err=" << err << "\n";
                ++bad;
            }
        }
    }
    std::cout << "  " << name << ": " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
    return ok;
}

// 遍历某层的全部参数做 gradcheck
bool check_all_params(ComputeEngine& engine,
                      const std::function<Scalar()>& eval_loss,
                      std::vector<TensorRef>& params,
                      std::vector<TensorRef>& grads,
                      Scalar eps, Scalar tol)
{
    bool ok = true;
    for (std::size_t i = 0; i < params.size(); ++i)
    {
        auto base = engine.to_matrix(params[i]);
        if (!base) { std::cerr << "  to_matrix failed\n"; return false; }
        auto ana = engine.to_matrix(grads[i]);
        if (!ana) { std::cerr << "  to_matrix(grad) failed\n"; return false; }
        ok &= check_param(engine, eval_loss, params[i], *base, *ana,
                          "param[" + std::to_string(i) + "]", eps, tol);
    }
    return ok;
}

} // namespace

int main(int argc, char* argv[])
{
    Scalar tol = 1e-2f;
    bool use_gpu = false;
    bool use_cuda = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--cuda") use_cuda = true;
    }

    auto engine = nn::cli::create_engine(nn::cli::EngineConfig{use_gpu, use_cuda});
    if (!engine) { std::cerr << engine.error().message << "\n"; return 1; }
    ComputeEngine& eng = **engine;

    constexpr Scalar eps = 1e-2f;
    bool all_ok = true;

    // ── 阶段一：CrossAttention ──────────────────────────────────────────
    {
        std::cout << "=== CrossAttention gradcheck ===\n";
        const std::size_t d_model = 8, memory = 4, seq = 4, batch = 2;
        CrossAttention ca(d_model, memory, seq);
        if (!ca.init(eng)) { std::cerr << "  CrossAttention init failed\n"; return 1; }

        // 输入 X (d_model, batch·seq)
        std::mt19937_64 rng{7};
        std::normal_distribution<Scalar> dist(0.0, 1.0);
        Matrix x_m(d_model, batch * seq);
        { auto sp = x_m.span(); for (auto& v : sp) v = dist(rng); }
        auto x_t = eng.from_matrix(x_m);
        if (!x_t) return 1;

        // 损失 = dot(输出 C, go)
        Matrix go_m(d_model, batch * memory);
        { auto sp = go_m.span(); for (auto& v : sp) v = dist(rng); }
        auto go_t = eng.from_matrix(go_m);
        if (!go_t) return 1;

        auto eval_loss = [&]() -> Scalar {
            auto y = ca.forward(eng, *x_t);
            auto ym = eng.to_matrix(*y);
            return dot(*ym, go_m);
        };

        // 解析梯度：zero → forward → backward
        auto grads = ca.param_gradients();
        for (auto& g : grads) { if (!eng.zero(g)) return 1; }
        auto y0 = ca.forward(eng, *x_t);
        if (!y0) { std::cerr << "  CrossAttention fwd failed\n"; return 1; }
        auto gx = ca.backward(eng, *go_t);
        if (!gx) { std::cerr << "  CrossAttention bwd failed\n"; return 1; }

        auto params = ca.parameters();
        all_ok &= check_all_params(eng, eval_loss, params, grads, eps, tol);
    }

    // ── 阶段二：ZiPTBlock ───────────────────────────────────────────────
    {
        std::cout << "=== ZiPTBlock gradcheck ===\n";
        const std::size_t d_model = 8, heads = 2, d_ff = 16;
        const std::size_t window = 4, memory = 2, batch = 2;
        ZiPTBlock blk(d_model, heads, d_ff, window, memory);
        if (!blk.init(eng)) { std::cerr << "  ZiPTBlock init failed\n"; return 1; }

        std::mt19937_64 rng{11};
        std::normal_distribution<Scalar> dist(0.0, 1.0);
        Matrix x_m(d_model, batch * window);
        { auto sp = x_m.span(); for (auto& v : sp) v = dist(rng); }
        auto x_t = eng.from_matrix(x_m);
        if (!x_t) return 1;
        Matrix c_m(d_model, batch * memory);
        { auto sp = c_m.span(); for (auto& v : sp) v = dist(rng); }
        auto c_t = eng.from_matrix(c_m);
        if (!c_t) return 1;

        Matrix go_m(d_model, batch * window);
        { auto sp = go_m.span(); for (auto& v : sp) v = dist(rng); }
        auto go_t = eng.from_matrix(go_m);
        if (!go_t) return 1;

        auto eval_loss = [&]() -> Scalar {
            auto y = blk.forward(eng, *x_t, *c_t);
            auto ym = eng.to_matrix(*y);
            return dot(*ym, go_m);
        };

        auto grads = blk.param_gradients();
        for (auto& g : grads) { if (!eng.zero(g)) return 1; }
        auto y0 = blk.forward(eng, *x_t, *c_t);
        if (!y0) { std::cerr << "  ZiPTBlock fwd failed\n"; return 1; }
        Tensor grad_C = eng.create_tensor(d_model, batch * memory);
        { auto z = eng.zero(grad_C); if (!z) return 1; }
        auto gx = blk.backward(eng, *go_t, grad_C);
        if (!gx) { std::cerr << "  ZiPTBlock bwd failed\n"; return 1; }

        auto params = blk.parameters();
        all_ok &= check_all_params(eng, eval_loss, params, grads, eps, tol);
    }

    std::cout << (all_ok ? "\nALL GRADCHECKS PASSED\n" : "\nSOME GRADCHECKS FAILED\n");
    return all_ok ? 0 : 1;
}
