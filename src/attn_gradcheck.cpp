// ── CausalSelfAttention 数值梯度检查（gradcheck） ──────────────────────────
//
// 目的：GPT 整链 gradcheck 显示 wo OK 但 wq/wk/wv FAIL，定位 attention 内部
//       是否真的有反向 bug（batched_matmul / softmax backward / 掩码）。
//       batch=1 时绕过 rearrange_3d，若仍 FAIL 则是 attention 内部问题。
//
// 用法：attn_gradcheck [--cuda|--gpu] [--batch N] [--seq N] [--tol <f>]
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
using nn::TensorRef;
using nn::ComputeEngine;
using nn::CausalSelfAttention;
using nn::PosEncodingType;

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

Scalar eval_loss(ComputeEngine& engine, CausalSelfAttention& attn,
                 const Tensor& x, const Tensor& go)
{
    auto y = attn.forward(engine, x);
    if (!y) { std::cerr << "  [eval] forward failed: " << y.error().message << "\n"; std::abort(); }
    auto y_m = engine.to_matrix(*y);
    auto go_m = engine.to_matrix(go);
    return dot(*y_m, *go_m);
}

bool approx(Scalar num, Scalar ana, Scalar tol)
{
    return std::fabs(num - ana) <=
           tol * (Scalar{1} + std::fabs(num) + std::fabs(ana));
}

bool check_grad_tensor(
    ComputeEngine& engine, CausalSelfAttention& attn,
    const Tensor& x, const Tensor& go,
    Tensor& param, const Matrix& base, const Matrix& grad_analytical,
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
            auto r_p = engine.copy_from(param, pp);
            if (!r_p) return false;
            const Scalar lp = eval_loss(engine, attn, x, go);
            Matrix pm = base;
            pm.set_value_unchecked(r, c, orig - eps);
            auto r_m = engine.copy_from(param, pm);
            if (!r_m) return false;
            const Scalar lm = eval_loss(engine, attn, x, go);
            auto r_r = engine.copy_from(param, base);
            if (!r_r) return false;
            const Scalar num = (lp - lm) / (Scalar{2} * eps);
            const Scalar ana = grad_analytical.at_unchecked(r, c);
            const Scalar err = std::fabs(num - ana);
            if (err > max_err) max_err = err;
            if (!approx(num, ana, tol))
            {
                ok = false;
                if (bad < 5)
                    std::cout << "  [FAIL] " << name << "(" << r << "," << c
                              << ") num=" << num << " ana=" << ana << " err=" << err << "\n";
                ++bad;
            }
        }
    }
    std::cout << "  " << name << ": " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
    return ok;
}

} // namespace

int main(int argc, char* argv[])
{
    Scalar tol = 5e-2f;
    bool use_cuda = false;
    bool use_gpu = false;
    std::size_t batch = 2;
    std::size_t seq = 8;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--batch" && i + 1 < argc) batch = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--seq" && i + 1 < argc) seq = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--cuda") use_cuda = true;
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: attn_gradcheck [--cuda|--gpu] [--batch N] [--seq N] [--tol <f>]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_cuda = use_cuda;
    ecfg.use_gpu = use_gpu;
    auto engine = nn::cli::create_engine(ecfg, std::cout);
    if (!engine) { std::cerr << "引擎创建失败\n"; return 1; }
    ComputeEngine& eng = *engine;

    const std::size_t d_model = 16;
    const std::size_t num_heads = 2;   // d_k = 8

    std::cout << "========================================\n";
    std::cout << "  CausalSelfAttention 数值梯度检查 (gradcheck)\n";
    std::cout << "========================================\n";
    std::cout << "  d_model=" << d_model << " heads=" << num_heads
              << " seq=" << seq << " batch=" << batch << " tol=" << tol << "\n";

    CausalSelfAttention attn(eng, d_model, num_heads, seq, seq, PosEncodingType::Learned);

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    const std::size_t total = seq * batch;
    Matrix x_m(d_model, total);
    Matrix go_m(d_model, total);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = dist(rng);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix(x) failed\n"; return 1; }
    auto go = eng.from_matrix(go_m);
    if (!go) { std::cerr << "from_matrix(go) failed\n"; return 1; }

    auto y_fwd = attn.forward(eng, *x);
    if (!y_fwd) { std::cerr << "forward failed: " << y_fwd.error().message << "\n"; return 1; }

    auto params = attn.parameters();
    auto grads  = attn.param_gradients();
    for (auto& g : grads)
    {
        auto rz = eng.zero(g.get());
        if (!rz) { std::cerr << "zero failed\n"; return 1; }
    }
    auto gx = attn.backward(eng, *go);
    if (!gx) { std::cerr << "backward failed: " << gx.error().message << "\n"; return 1; }

    const Scalar eps = 1e-3f;
    bool all_pass = true;
    const char* tags[] = {"wq.w","wq.b","wk.w","wk.b","wv.w","wv.b","wo.w","wo.b"};
    for (std::size_t p = 0; p < params.size(); ++p)
    {
        auto pm = eng.to_matrix(params[p].get());
        auto gm = eng.to_matrix(grads[p].get());
        std::string name = (p < 8) ? std::string(tags[p]) : ("param[" + std::to_string(p) + "]");
        all_pass &= check_grad_tensor(eng, attn, *x, *go, params[p].get(), *pm, *gm, name, eps, tol);
    }

    // 输入梯度
    {
        auto gxm = eng.to_matrix(*gx);
        auto xt = eng.clone(*x);
        all_pass &= check_grad_tensor(eng, attn, *xt, *go, *xt, x_m, *gxm, "grad_x", eps, tol);
    }

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return all_pass ? 0 : 1;
}
