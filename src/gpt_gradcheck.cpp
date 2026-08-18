// ── GPTModel 整链数值梯度检查（gradcheck） ────────────────────────────────
//
// 目的：验证 GPTModel 完整 forward/backward 链路的梯度正确性。
//       单层 gradcheck（rmsnorm/swiglu）已验证层内数学，但多层堆叠 +
//       learned 位置编码 + token_emb scatter 的整链路径从未验证。
//       用户现象：浅层 GPT 正常、深层 GPT loss 卡平台 → 疑似链路级 bug。
//
// 原理：
//   L(θ) = Σ_ij logits_ij * go_ij   （go 为固定"输出权重"）
//   则 dL/dlogits = go；对每个参数元素做 ±eps 扰动重算 L，
//   中心差分近似解析梯度（backward 所得）并比对。
//
// 用法：gpt_gradcheck [--cuda|--gpu] [--layers N] [--tol <f>]
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
using nn::GPTModel;
using nn::Layer;
using nn::PosEncodingType;
using nn::ActivationType;
using nn::NormType;

namespace {

// L = Σ a*b（逐元素乘后求和）
Scalar dot(const Matrix& a, const Matrix& b)
{
    Scalar s{0};
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
        s += sa[i] * sb[i];
    return s;
}

// 用当前参数前向，返回 L = Σ logits*go
Scalar eval_loss(ComputeEngine& engine, GPTModel& model,
                 const Tensor& input, const Tensor& go)
{
    auto y = model.forward(engine, input);
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

// 验证单个参数张量梯度（逐元素中心差分）
bool check_grad_tensor(
    ComputeEngine& engine, GPTModel& model,
    const Tensor& input, const Tensor& go,
    Tensor& param,
    const Matrix& base,
    const Matrix& grad_analytical,
    const std::string& name,
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
            const Scalar lp = eval_loss(engine, model, input, go);

            Matrix pm = base;
            pm.set_value_unchecked(r, c, orig - eps);
            auto r_m = engine.copy_from(param, pm);
            if (!r_m) { std::cerr << "copy_from(-) failed\n"; return false; }
            const Scalar lm = eval_loss(engine, model, input, go);

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
                if (bad < 6)
                    std::cout << "  [FAIL] " << name << "(" << r << "," << c
                              << ") num=" << num << " ana=" << ana
                              << " err=" << err << "\n";
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

int main(int argc, char* argv[])
{
    Scalar tol = 5e-2f;
    bool use_cuda = false;
    bool use_gpu = false;
    std::size_t num_layers = 2;
    std::size_t batch = 2;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc)
            tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--layers" && i + 1 < argc)
            num_layers = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--batch" && i + 1 < argc)
            batch = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--cuda")
            use_cuda = true;
        else if (a == "--gpu")
            use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: gpt_gradcheck [--cuda|--gpu] [--layers N] [--batch N] [--tol <f>]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_cuda = use_cuda;
    ecfg.use_gpu = use_gpu;
    auto engine = nn::cli::create_engine(ecfg, std::cout);
    if (!engine) { std::cerr << "引擎创建失败\n"; return 1; }
    ComputeEngine& eng = *engine;

    // 小规模，加速测试。配置与用户实际使用一致：swiglu + rmsnorm + learned
    const std::size_t vocab = 32;
    const std::size_t d_model = 16;
    const std::size_t seq_len = 8;
    const std::size_t num_heads = 2;
    const std::size_t d_ff = 32;
    // batch 由命令行控制，默认 2（验证 rearrange_3d batch 路径）

    std::cout << "========================================\n";
    std::cout << "  GPTModel 整链数值梯度检查 (gradcheck)\n";
    std::cout << "========================================\n";
    std::cout << "  vocab=" << vocab << " d_model=" << d_model
              << " seq=" << seq_len << " batch=" << batch
              << " heads=" << num_heads << " d_ff=" << d_ff
              << " layers=" << num_layers << "\n";
    std::cout << "  activation=SwiGLU norm=RMSNorm pos=Learned\n";
    std::cout << "  tol=" << tol << "\n";

    GPTModel model(eng, vocab, d_model, seq_len, num_heads, d_ff, num_layers,
                   PosEncodingType::Learned,
                   ActivationType::SwiGLU,
                   NormType::RMSNorm);

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    // 输入：token IDs (seq, batch)
    Matrix x_m(seq_len, batch);
    for (std::size_t i = 0; i < x_m.size(); ++i)
        x_m.span()[i] = static_cast<Scalar>(rng() % vocab);

    // 输出权重 go (vocab, seq*batch)
    Matrix go_m(vocab, seq_len * batch);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix(x) failed\n"; return 1; }
    auto go = eng.from_matrix(go_m);
    if (!go) { std::cerr << "from_matrix(go) failed\n"; return 1; }

    // ── 前向一次：填充各层 forward 缓存 ──
    auto y_fwd = model.forward(eng, *x);
    if (!y_fwd) { std::cerr << "forward failed: " << y_fwd.error().message << "\n"; return 1; }
    std::cout << "  forward OK, logits: " << y_fwd->rows() << "x" << y_fwd->cols() << "\n";

    // ── 清零梯度并 backward ──
    auto params = model.parameters();
    auto grads  = model.param_gradients();
    if (params.size() != grads.size())
    {
        std::cerr << "params/grads size mismatch: "
                  << params.size() << " vs " << grads.size() << "\n";
        return 1;
    }
    for (auto& g : grads)
    {
        auto rz = eng.zero(g.get());
        if (!rz) { std::cerr << "zero failed\n"; return 1; }
    }
    auto gx = model.backward(eng, *go);
    if (!gx) { std::cerr << "backward failed: " << gx.error().message << "\n"; return 1; }
    std::cout << "  参数总数: " << params.size() << "\n";

    // ── 验证每个参数梯度 ──
    const Scalar eps = 1e-3f;
    bool all_pass = true;
    std::size_t pi = 0;
    for (std::size_t p = 0; p < params.size(); ++p)
    {
        auto pm = eng.to_matrix(params[p].get());
        if (!pm) { std::cerr << "to_matrix(param) failed\n"; return 1; }
        auto gm = eng.to_matrix(grads[p].get());
        if (!gm) { std::cerr << "to_matrix(grad) failed\n"; return 1; }
        if (pm->rows() != gm->rows() || pm->cols() != gm->cols())
        {
            std::cerr << "shape mismatch param[" << p << "]: "
                      << pm->rows() << "x" << pm->cols()
                      << " vs grad " << gm->rows() << "x" << gm->cols() << "\n";
            return 1;
        }
        std::string name = "param[" + std::to_string(pi++) + "]";
        // 精简命名：按顺序标注
        static const char* tags[] = {"token_emb", "pos_emb", "blk0.wq", "blk0.wk",
            "blk0.wv", "blk0.wo", "blk0.norm1.g", "blk0.fc1.w", "blk0.fc1.b",
            "blk0.fc2.w", "blk0.fc2.b", "blk0.norm2.g"};
        if (pi - 1 < 12)
            name = std::string("p[") + std::to_string(pi - 1) + "]=" + tags[pi - 1];
        all_pass &= check_grad_tensor(
            eng, model, *x, *go, params[p].get(), *pm, *gm, name, eps, tol);
    }

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return all_pass ? 0 : 1;
}
