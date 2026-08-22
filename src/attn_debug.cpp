// ── 注意力反向逐级调试探针 ────────────────────────────────────────────────
//
// 目的：attn_gradcheck 显示 wq/wk FAIL 但 wv/wo OK。本程序用「暴力 Matrix
//       参考」完整复现 CausalSelfAttention 的 forward + backward，并逐项对比
//       引擎 batched_matmul / Softmax 原语计算出的中间量与参考，定位第一个
//       出错的中间量（grad_S / grad_A / grad_Q / grad_K / batched_matmul 原语）。
//
// 用法：attn_debug
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::CausalSelfAttention;
using nn::Softmax;
using nn::PosEncodingType;

namespace {

// ── 简单矩阵代数辅助（行主序 (rows, cols)）───────────────────────────────
Matrix mmul(const Matrix& A, const Matrix& B)
{
    Matrix C(A.rows(), B.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < B.cols(); ++j)
        {
            Scalar s{0};
            for (std::size_t k = 0; k < A.cols(); ++k)
                s += A.at_unchecked(i, k) * B.at_unchecked(k, j);
            C.set_value_unchecked(i, j, s);
        }
    return C;
}

Matrix mtrans(const Matrix& A)
{
    Matrix T(A.cols(), A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            T.set_value_unchecked(j, i, A.at_unchecked(i, j));
    return T;
}

Matrix madd(const Matrix& A, const Matrix& B)
{
    Matrix C = A;
    for (std::size_t i = 0; i < C.size(); ++i)
        C.span()[i] += B.span()[i];
    return C;
}

// 行偏置广播加：bias (rows,1) 逐行加到 A (rows, cols)
Matrix madd_bias(const Matrix& A, const Matrix& B)
{
    Matrix C = A;
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        const Scalar b = B.at_unchecked(i, 0);
        for (std::size_t j = 0; j < A.cols(); ++j)
            C.set_value_unchecked(i, j, C.at_unchecked(i, j) + b);
    }
    return C;
}

Matrix mscale(const Matrix& A, Scalar s)
{
    Matrix C = A;
    for (std::size_t i = 0; i < C.size(); ++i)
        C.span()[i] *= s;
    return C;
}

Matrix rows(const Matrix& A, std::size_t r0, std::size_t n)
{
    Matrix R(n, A.cols());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            R.set_value_unchecked(i, j, A.at_unchecked(r0 + i, j));
    return R;
}

// 逐行 softmax（含 -inf 处理）
Matrix softmax_rows(const Matrix& S)
{
    Matrix A(S.rows(), S.cols());
    for (std::size_t i = 0; i < S.rows(); ++i)
    {
        Scalar mx = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t j = 0; j < S.cols(); ++j)
            mx = std::max(mx, S.at_unchecked(i, j));
        Scalar sum{0};
        for (std::size_t j = 0; j < S.cols(); ++j)
        {
            Scalar e = std::exp(S.at_unchecked(i, j) - mx);
            A.set_value_unchecked(i, j, e);
            sum += e;
        }
        for (std::size_t j = 0; j < S.cols(); ++j)
            A.set_value_unchecked(i, j, A.at_unchecked(i, j) / sum);
    }
    return A;
}

// 逐行 softmax 反向: dS = A ⊙ (gA - row_dot)
Matrix softmax_bwd(const Matrix& A, const Matrix& gA)
{
    Matrix dS(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        Scalar dot{0};
        for (std::size_t j = 0; j < A.cols(); ++j)
            dot += A.at_unchecked(i, j) * gA.at_unchecked(i, j);
        for (std::size_t j = 0; j < A.cols(); ++j)
            dS.set_value_unchecked(i, j,
                A.at_unchecked(i, j) * (gA.at_unchecked(i, j) - dot));
    }
    return dS;
}

void cmp(const std::string& name, const Matrix& got, const Matrix& ref)
{
    bool ok = true;
    Scalar max_err{0};
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.rows(); ++i)
        for (std::size_t j = 0; j < got.cols(); ++j)
        {
            const Scalar e = std::fabs(got.at_unchecked(i, j) - ref.at_unchecked(i, j));
            if (e > max_err) max_err = e;
            if (e > Scalar{1e-4})
            {
                ok = false;
                if (bad < 6)
                    std::printf("  [%s] (%zu,%zu) got=%.6f ref=%.6f err=%.3e\n",
                                name.c_str(), i, j,
                                static_cast<double>(got.at_unchecked(i, j)),
                                static_cast<double>(ref.at_unchecked(i, j)),
                                static_cast<double>(e));
                ++bad;
            }
        }
    std::cout << "  " << name << ": " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
}

} // namespace

int main()
{
    nn::cli::EngineConfig ecfg;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    const std::size_t d_model = 16;
    const std::size_t H = 2;
    const std::size_t dk = 8;
    const std::size_t seq = 8;
    const std::size_t batch = 1;
    const Scalar scale = Scalar{1} / std::sqrt(static_cast<Scalar>(dk));

    std::cout << "========================================\n";
    std::cout << "  Attention 反向逐级调试 (d_model=16 H=2 seq=8 batch=1)\n";
    std::cout << "========================================\n";

    CausalSelfAttention attn(eng, d_model, H, seq, seq, PosEncodingType::Learned);

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-1, 1);
    const std::size_t total = seq * batch;
    Matrix x_m(d_model, total), go_m(d_model, total);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = dist(rng);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);
    auto x = eng.from_matrix(x_m);
    auto go = eng.from_matrix(go_m);

    // ── 提取权重 ──
    auto params = attn.parameters();
    Matrix Wq = *eng.to_matrix(params[0]);  // (16,16)
    Matrix bq = *eng.to_matrix(params[1]);  // (16,1)
    Matrix Wk = *eng.to_matrix(params[2]);
    Matrix bk = *eng.to_matrix(params[3]);
    Matrix Wv = *eng.to_matrix(params[4]);
    Matrix bv = *eng.to_matrix(params[5]);
    Matrix Wo = *eng.to_matrix(params[6]);
    Matrix bo = *eng.to_matrix(params[7]);

    // ── 参考 forward ──
    Matrix Q = madd_bias(mmul(Wq, x_m), bq);
    Matrix K = madd_bias(mmul(Wk, x_m), bk);
    Matrix V = madd_bias(mmul(Wv, x_m), bv);
    std::vector<Matrix> Ah(H, Matrix(seq, seq));
    for (std::size_t h = 0; h < H; ++h)
    {
        Matrix Qh = rows(Q, h * dk, dk);
        Matrix Kh = rows(K, h * dk, dk);
        Matrix Sh = mscale(mmul(mtrans(Qh), Kh), scale);
        // 因果掩码：j > i → -inf
        for (std::size_t i = 0; i < seq; ++i)
            for (std::size_t j = i + 1; j < seq; ++j)
                Sh.set_value_unchecked(i, j, -std::numeric_limits<Scalar>::infinity());
        Ah[h] = softmax_rows(Sh);
        if (h == 0)
        {
            std::cout << "  [ref] head0 Sh (masked) col0..7:\n";
            for (std::size_t i = 0; i < seq; ++i)
            {
                std::cout << "    row" << i << ":";
                for (std::size_t j = 0; j < seq; ++j)
                    std::printf(" %9.4f", static_cast<double>(Sh.at_unchecked(i, j)));
                std::cout << "\n";
            }
            std::cout << "  [ref] head0 A:\n";
            for (std::size_t i = 0; i < seq; ++i)
            {
                std::cout << "    row" << i << ":";
                for (std::size_t j = 0; j < seq; ++j)
                    std::printf(" %9.6f", static_cast<double>(Ah[h].at_unchecked(i, j)));
                std::cout << "\n";
            }
        }
    }
    Matrix concat(d_model, seq);
    for (std::size_t h = 0; h < H; ++h)
    {
        Matrix Vh = rows(V, h * dk, dk);
        Matrix Oh = mmul(Vh, mtrans(Ah[h]));  // (dk, seq)
        for (std::size_t i = 0; i < dk; ++i)
            for (std::size_t j = 0; j < seq; ++j)
                concat.set_value_unchecked(h * dk + i, j, Oh.at_unchecked(i, j));
    }
    Matrix y_ref = madd_bias(mmul(Wo, concat), bo);

    auto y_fwd = attn.forward(eng, *x);
    Matrix y_got = *eng.to_matrix(*y_fwd);
    cmp("forward out (wo concat)", y_got, y_ref);
    std::cout << "  (forward out 一致性验证 wv/wo 依赖的中间量)\n\n";

    // ── 参考 backward ──
    Matrix gc = mmul(mtrans(Wo), go_m);  // (16, seq)
    std::vector<Matrix> gQh(H, Matrix(dk, seq)), gKh(H, Matrix(dk, seq)), gVh(H, Matrix(dk, seq));
    for (std::size_t h = 0; h < H; ++h)
    {
        Matrix Vh = rows(V, h * dk, dk);
        Matrix Kh = rows(K, h * dk, dk);
        Matrix Qh = rows(Q, h * dk, dk);
        Matrix gch = rows(gc, h * dk, dk);
        Matrix gV = mmul(gch, Ah[h]);                       // (dk, seq)
        Matrix gA = mmul(mtrans(Vh), gch);                  // (seq, seq)
        Matrix gS = softmax_bwd(Ah[h], gA);                 // (seq, seq)
        gQh[h] = mscale(mmul(Kh, mtrans(gS)), scale);
        gKh[h] = mscale(mmul(Qh, gS), scale);
        gVh[h] = gV;
    }
    Matrix gQ(d_model, seq), gK(d_model, seq), gV(d_model, seq);
    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t i = 0; i < dk; ++i)
            for (std::size_t j = 0; j < seq; ++j)
            {
                gQ.set_value_unchecked(h * dk + i, j, gQh[h].at_unchecked(i, j));
                gK.set_value_unchecked(h * dk + i, j, gKh[h].at_unchecked(i, j));
                gV.set_value_unchecked(h * dk + i, j, gVh[h].at_unchecked(i, j));
            }
    Matrix ref_gwq = mmul(gQ, mtrans(x_m));
    Matrix ref_gwk = mmul(gK, mtrans(x_m));
    Matrix ref_gwv = mmul(gV, mtrans(x_m));
    Matrix ref_gwo = mmul(gc, mtrans(concat));
    Matrix ref_gbq(d_model, 1), ref_gbk(d_model, 1), ref_gbv(d_model, 1), ref_gbo(d_model, 1);
    for (std::size_t i = 0; i < d_model; ++i)
    {
        Scalar sq{0}, sk{0}, sv{0}, so{0};
        for (std::size_t j = 0; j < seq; ++j)
        {
            sq += gQ.at_unchecked(i, j);
            sk += gK.at_unchecked(i, j);
            sv += gV.at_unchecked(i, j);
            so += gc.at_unchecked(i, j);
        }
        ref_gbq.set_value_unchecked(i, 0, sq);
        ref_gbk.set_value_unchecked(i, 0, sk);
        ref_gbv.set_value_unchecked(i, 0, sv);
        ref_gbo.set_value_unchecked(i, 0, so);
    }
    Matrix ref_gx = madd(madd(mmul(mtrans(Wq), gQ), mmul(mtrans(Wk), gK)), mmul(mtrans(Wv), gV));

    // ── 引擎 backward ──
    auto grads = attn.param_gradients();
    for (auto& g : grads) { auto rz = eng.zero(g.get()); if (!rz) return 1; }
    auto gx = attn.backward(eng, *go);
    if (!gx) { std::cerr << "backward failed\n"; return 1; }

    Matrix gwq = *eng.to_matrix(grads[0]);
    Matrix gbk = *eng.to_matrix(grads[3]);
    Matrix gwv = *eng.to_matrix(grads[4]);
    Matrix gwo = *eng.to_matrix(grads[6]);
    Matrix gx_got = *eng.to_matrix(*gx);

    std::cout << "── 引擎 vs 暴力参考（仅比较重量级梯度） ──\n";
    cmp("grad_wq", gwq, ref_gwq);
    cmp("grad_wk", *eng.to_matrix(grads[2]), ref_gwk);
    cmp("grad_wv", gwv, ref_gwv);
    cmp("grad_wo", gwo, ref_gwo);
    cmp("grad_bk", gbk, ref_gbk);
    cmp("grad_x", gx_got, ref_gx);

    return 0;
}
