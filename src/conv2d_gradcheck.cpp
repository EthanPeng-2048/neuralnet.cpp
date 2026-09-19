// ── Conv2D 正确性检查（独立参考实现比对） ─────────────────────────────────
//
// 目的：Conv2D 此前**没有任何测试覆盖**。本片段把层实现与一份独立写法的直接
//       卷积（含 im2col 索引、grad_w/grad_b 解析式）逐元素比对，覆盖：
//         forward : Z = matmul(W, im2col(x)) + b（matmul 段融合行广播偏置）
//         backward: grad_w += matmul(gZ, col^T)（matmul 段融合原地累加）
//                   grad_b += row_sum(gZ)
//       这两条路径是 AOT 融合 shader 结构，必须两引擎都验证（--gpu）。
//
// 用法：conv2d_gradcheck [--gpu]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::Conv2D;

namespace {

// 独立参考：直接卷积 + im2col 索引（不依赖层内任何实现）
struct ConvRef
{
    std::size_t C_in, C_out, K, H, W, OH, OW, B;
    Matrix x;   // (C_in*H*W, B)
    Matrix w;   // (C_out, C_in*K*K)
    Matrix b;   // (C_out, 1)

    Scalar xv(std::size_t ci, long ih, long iw, std::size_t bi) const
    {
        if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) || iw >= static_cast<long>(W))
            return Scalar{0};
        return x.at_unchecked(ci * H * W + static_cast<std::size_t>(ih) * W +
                                  static_cast<std::size_t>(iw), bi);
    }
    // forward 输出布局 (C_out*OH*OW, B)
    Matrix fwd() const
    {
        Matrix out(C_out * OH * OW, B);
        for (std::size_t co = 0; co < C_out; ++co)
            for (std::size_t bi = 0; bi < B; ++bi)
                for (std::size_t oh = 0; oh < OH; ++oh)
                    for (std::size_t ow = 0; ow < OW; ++ow)
                    {
                        Scalar s = b.at_unchecked(co, 0);
                        for (std::size_t ci = 0; ci < C_in; ++ci)
                            for (std::size_t kh = 0; kh < K; ++kh)
                                for (std::size_t kw = 0; kw < K; ++kw)
                                    s += w.at_unchecked(co, ci * K * K + kh * K + kw)
                                       * xv(ci, static_cast<long>(oh) + static_cast<long>(kh),
                                            static_cast<long>(ow) + static_cast<long>(kw), bi);
                        out.set_value_unchecked(co * OH * OW + oh * OW + ow, bi, s);
                    }
        return out;
    }
    void bwd(const Matrix& grad_out, Matrix& gw, Matrix& gb) const
    {
        gw = Matrix(C_out, C_in * K * K, Scalar{0});
        gb = Matrix(C_out, 1, Scalar{0});
        for (std::size_t co = 0; co < C_out; ++co)
        {
            Scalar sb = 0;
            for (std::size_t bi = 0; bi < B; ++bi)
                for (std::size_t oh = 0; oh < OH; ++oh)
                    for (std::size_t ow = 0; ow < OW; ++ow)
                    {
                        const Scalar g = grad_out.at_unchecked(co * OH * OW + oh * OW + ow, bi);
                        sb += g;
                        for (std::size_t ci = 0; ci < C_in; ++ci)
                            for (std::size_t kh = 0; kh < K; ++kh)
                                for (std::size_t kw = 0; kw < K; ++kw)
                                    gw.span()[co * (C_in * K * K) + ci * K * K + kh * K + kw] +=
                                        g * xv(ci, static_cast<long>(oh) + static_cast<long>(kh),
                                               static_cast<long>(ow) + static_cast<long>(kw), bi);
                    }
            gb.set_value_unchecked(co, 0, sb);
        }
    }
};

Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return Scalar{1e9f};
    Scalar e{0};
    for (std::size_t i = 0; i < a.span().size(); ++i)
        e = std::max(e, std::fabs(a.span()[i] - b.span()[i]));
    return e;
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: conv2d_gradcheck [--gpu]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    std::cout << "========================================\n";
    std::cout << "  Conv2D 正确性检查（独立参考实现比对）\n";
    std::cout << "========================================\n";

    const std::size_t C_in = 2, C_out = 3, K = 3, H = 6, W = 6, B = 2;
    const std::size_t OH = H - K + 1, OW = W - K + 1;
    const Scalar tol = 1e-4f;

    std::mt19937_64 rng(20260918);
    std::uniform_real_distribution<Scalar> dist(-0.5f, 0.5f);
    const auto fill = [&](std::size_t r, std::size_t c) {
        Matrix m(r, c);
        for (std::size_t i = 0; i < m.size(); ++i) m.span()[i] = dist(rng);
        return m;
    };

    Conv2D conv(C_in, C_out, K, /*stride=*/1, /*padding=*/0, H, W);
    if (auto r = conv.init(eng); !r)
    {
        std::cerr << "  Conv2D init 失败: " << r.error().message << "\n";
        return 1;
    }
    // init 是随机初始化 → 用确定性权重覆盖，参考实现才可比
    Matrix wm = fill(C_out, C_in * K * K);
    Matrix bm = fill(C_out, 1);
    if (auto r = eng.copy_from(conv.parameters()[0].get(), wm); !r) return 1;
    if (auto r = eng.copy_from(conv.parameters()[1].get(), bm); !r) return 1;

    ConvRef ref{C_in, C_out, K, H, W, OH, OW, B, fill(C_in * H * W, B), wm, bm};

    int failures = 0;
    const auto report = [&](const char* what, const Matrix& got, const Matrix& want) {
        const Scalar e = max_abs_diff(got, want);
        const bool ok = e < tol;
        std::printf("  %-14s %s  max_abs_diff=%.3e\n", what, ok ? "OK" : "FAIL",
                    static_cast<double>(e));
        if (!ok) ++failures;
    };

    auto x = eng.from_matrix(ref.x);
    if (!x) return 1;
    auto out = conv.forward(eng, *x);
    if (!out) { std::cerr << "  forward 失败: " << out.error().message << "\n"; return 1; }
    auto om = eng.to_matrix(*out);
    if (!om) return 1;
    report("forward", *om, ref.fwd());

    Matrix gm = fill(C_out * OH * OW, B);
    auto go = eng.from_matrix(gm);
    if (!go) return 1;
    auto gx = conv.backward(eng, *go);
    if (!gx) { std::cerr << "  backward 失败: " << gx.error().message << "\n"; return 1; }

    Matrix gw_ref, gb_ref;
    ref.bwd(gm, gw_ref, gb_ref);
    auto gw = eng.to_matrix(conv.param_gradients()[0].get());
    auto gb = eng.to_matrix(conv.param_gradients()[1].get());
    if (gw) report("grad_w", *gw, gw_ref);
    if (gb) report("grad_b", *gb, gb_ref);

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (failures == 0 ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return failures == 0 ? 0 : 1;
}
