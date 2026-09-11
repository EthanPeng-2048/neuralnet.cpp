// ── SwiGLU 数值梯度检查（gradcheck） ─────────────────────────────────────
//
// 目的：用有限差分（中心差分）验证 SwiGLU FFN 的 backward 参数梯度与
//       输入梯度是否正确。split（slice_rows）与 merge（insert_rows）是
//       SwiGLU 独有的操作，最容易引入反向传播错误，故单独验证。
//
// 原理：
//   L(θ) = Σ_ij y_ij * go_ij   （go 为固定的"输出权重"）
//   则 dL/dθ = backward 得到的梯度（因为 dL/dy = go）
//   对每个参数元素做 ±eps 扰动，重算 L，中心差分近似解析梯度并比对。
//
// 用法：swiglu_gradcheck [--tol <f>]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::TensorRef;
using nn::ComputeEngine;
using nn::CpuEngine;
using nn::ActivationType;
using nn::FeedForward;

namespace {

// L = Σ a*b（逐元素乘后求和）
Scalar dot(const Matrix &a, const Matrix &b)
{
    Scalar s{0};
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
        s += sa[i] * sb[i];
    return s;
}

// 用当前参数前向，返回 L = Σ y*go
Scalar eval_loss(ComputeEngine &engine, FeedForward &ff,
                 const Tensor &x, const Tensor &go)
{
    auto y = ff.forward(engine, x);
    auto y_m = engine.to_matrix(*y);
    auto go_m = engine.to_matrix(go);
    return dot(*y_m, *go_m);
}

bool approx(Scalar num, Scalar ana, Scalar tol)
{
    return std::fabs(num - ana) <=
           tol * (Scalar{1} + std::fabs(num) + std::fabs(ana));
}

// 验证单个张量参数/输入的梯度（逐元素中心差分）
//   fwd_input: 前向传播使用的输入。参数验证时传固定 x；输入梯度验证时
//              传被扰动的输入副本（扰动 fwd_input 本身并重算 loss）。
bool check_grad_tensor(
    ComputeEngine &engine, FeedForward &ff,
    const Tensor &fwd_input,
    const Tensor &go,
    Tensor &param,               // 被扰动的张量（参数或输入）
    const Matrix &base,          // 扰动前的值
    const Matrix &grad_analytical, // backward 得到的梯度
    const std::string &name,
    Scalar eps, Scalar tol)
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
            if (!r_p) { std::cerr << "copy_from(+) failed\n"; return false; }
            const Scalar lp = eval_loss(engine, ff, fwd_input, go);

            Matrix pm = base;
            pm.set_value_unchecked(r, c, orig - eps);
            auto r_m = engine.copy_from(param, pm);
            if (!r_m) { std::cerr << "copy_from(-) failed\n"; return false; }
            const Scalar lm = eval_loss(engine, ff, fwd_input, go);

            // 恢复原值
            auto r_r = engine.copy_from(param, base);
            if (!r_r) { std::cerr << "copy_from(restore) failed\n"; return false; }

            const Scalar num = (lp - lm) / (Scalar{2} * eps);
            const Scalar ana = grad_analytical.at_unchecked(r, c);
            const Scalar err = std::fabs(num - ana);
            if (err > max_err) max_err = err;
            if (!approx(num, ana, tol))
            {
                ok = false;
                if (bad < 10)
                    std::cout << "  [FAIL] " << name << "(" << r << "," << c
                              << ") num=" << num << " ana=" << ana << "\n";
                ++bad;
            }
        }
    }
    std::cout << "  " << name << ": "
              << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
    return ok;
}

} // namespace

int main(int argc, char *argv[])
{
    Scalar tol = 2e-2f;
    bool use_cuda = false;
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc)
            tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--cuda")
            use_cuda = true;
        else if (a == "--gpu")
            use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: swiglu_gradcheck [--cuda|--gpu] [--tol <f>]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_cuda = use_cuda;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine &eng = *engine;

    // 小规模，加速测试
    const std::size_t d_model = 8;
    const std::size_t d_ff    = 16;   // SwiGLU 时 fc1 → 2*d_ff=32
    const std::size_t batch   = 4;

    std::cout << "========================================\n";
    std::cout << "  SwiGLU 数值梯度检查 (gradcheck)\n";
    std::cout << "========================================\n";
    std::cout << "  d_model=" << d_model << " d_ff=" << d_ff
              << " batch=" << batch << " tol=" << tol << "\n";

    FeedForward ff(d_model, d_ff, ActivationType::SwiGLU);
    { auto r = ff.init(eng); if (!r) { std::cerr << "FeedForward init 失败: " << r.error().message << "\n"; return 1; } }

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    Matrix x_m(d_model, batch);
    Matrix go_m(d_model, batch);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = dist(rng);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix(x) failed\n"; return 1; }
    auto go = eng.from_matrix(go_m);
    if (!go) { std::cerr << "from_matrix(go) failed\n"; return 1; }

    // ── 前向一次：填充各层 input_cache_（backward 依赖 forward 缓存） ──
    auto y_fwd = ff.forward(eng, *x);
    if (!y_fwd) { std::cerr << "forward failed: " << y_fwd.error().message << "\n"; return 1; }

    // ── 清零梯度并 backward ──
    auto params = ff.parameters();
    auto grads  = ff.param_gradients();
    for (auto &g : grads)
    {
        auto rz = eng.zero(g.get());
        if (!rz) { std::cerr << "zero failed\n"; return 1; }
    }
    auto gx = ff.backward(eng, *go);
    if (!gx) { std::cerr << "backward failed: " << gx.error().message << "\n"; return 1; }

    // ── 验证参数梯度 ──
    const Scalar eps = 1e-3f;
    bool all_pass = true;
    const char *pnames[] = {"fc1.w", "fc1.b", "fc2.w", "fc2.b"};

    for (std::size_t pi = 0; pi < params.size(); ++pi)
    {
        auto p = eng.to_matrix(params[pi].get());
        if (!p) { std::cerr << "to_matrix(param) failed\n"; return 1; }
        auto g = eng.to_matrix(grads[pi].get());
        if (!g) { std::cerr << "to_matrix(grad) failed\n"; return 1; }
        std::string name = pi < 4 ? pnames[pi] : ("param[" + std::to_string(pi) + "]");
        all_pass &= check_grad_tensor(
            eng, ff, *x, *go, params[pi].get(), *p, *g, name, eps, tol);
    }

    // ── 验证输入梯度（扰动输入副本本身，前向用该副本） ──
    {
        auto gxm = eng.to_matrix(*gx);
        if (!gxm) { std::cerr << "to_matrix(grad_x) failed\n"; return 1; }
        auto xt = eng.clone(*x);   // 可写副本，避免扰动原始输入 Tensor
        if (!xt) { std::cerr << "clone(x) failed\n"; return 1; }
        all_pass &= check_grad_tensor(
            eng, ff, *xt, *go, *xt, x_m, *gxm, "grad_x", eps, tol);
    }

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return all_pass ? 0 : 1;
}
