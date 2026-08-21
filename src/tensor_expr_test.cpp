// ── tensor_expr_test — 统一表达式 DSL 求值测试（CPU）────────────────────
//
// 验证"对真实 Tensor 用普通数学写法写表达式 + dsl::compute 统一求值"：
//   auto out = dsl::compute(engine, leaf(A) * leaf(B) + leaf(C), rows, cols);
//
// 逐元素在 CPU 走编译期模板（内联 + SIMD 融合）；matmul 是原语边界，
// 先用 engine.matmul 计算，再并入逐元素表达式。
//
// 注意（闭合世界）：GPU 只执行 fused_exprs.hpp kGenInstances 里预生成的
// AOT 融合 shader；本测试的任意表达式未预生成，故只在 CPU 上验证。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::CpuEngine;
using namespace nn::dsl;

namespace {

// 随机填充
void fill_random(Matrix& m, std::mt19937& rng,
                 std::uniform_real_distribution<Scalar>& dist)
{
    for (auto& v : m.span()) v = dist(rng);
}

// 最大绝对误差
Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    Scalar err = 0;
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
    {
        const Scalar d = std::fabs(sa[i] - sb[i]);
        if (d > err) err = d;
    }
    return err;
}

// 从引擎取回 CPU Matrix
bool to_cpu(ComputeEngine& eng, const nn::Result<Tensor>& r, Matrix& out)
{
    if (!r) { std::cerr << "  dsl::compute 失败: " << r.error().message << "\n"; return false; }
    auto m = eng.to_matrix(*r);
    if (!m) { std::cerr << "  to_matrix 失败: " << m.error().message << "\n"; return false; }
    out = std::move(*m);
    return true;
}

// 单个逐元素用例：计算并比对
template <typename Expr>
int check(ComputeEngine& eng, const std::string& name, const Expr& expr,
          std::size_t rows, std::size_t cols, const Matrix& ref, Scalar tol)
{
    auto r = nn::dsl::compute(eng, expr, rows, cols);
    Matrix got;
    if (!to_cpu(eng, r, got)) { std::cerr << "[FAIL] " << name << "\n"; return 1; }
    const Scalar err = max_abs_diff(ref, got);
    const bool ok = err <= tol;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << name
              << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
    return ok ? 0 : 1;
}

// 逐元素参考：out = op(a, b)
void ref_binary(Matrix& out, const Matrix& a, const Matrix& b, char op)
{
    const auto sa = a.span(), sb = b.span(); auto so = out.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
    {
        switch (op)
        {
        case '+': so[i] = sa[i] + sb[i]; break;
        case '-': so[i] = sa[i] - sb[i]; break;
        case '*': so[i] = sa[i] * sb[i]; break;
        case '/': so[i] = sa[i] / sb[i]; break;
        }
    }
}

} // namespace

int main()
{
    CpuEngine engine;
    ComputeEngine& eng = engine;

    std::cout << "========================================\n"
              << "  统一表达式 DSL 求值测试（CPU）\n"
              << "========================================\n";

    constexpr Scalar kTol = 1e-4f;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    int fail = 0;

    const std::size_t R = 4, C = 5;
    Matrix mA(R, C), mB(R, C), mC(R, C), mD(R, C);
    fill_random(mA, rng, dist); fill_random(mB, rng, dist);
    fill_random(mC, rng, dist); fill_random(mD, rng, dist);

    const Tensor ta = Tensor::from_matrix(Matrix(mA));
    const Tensor tb = Tensor::from_matrix(Matrix(mB));
    const Tensor tc = Tensor::from_matrix(Matrix(mC));
    const Tensor td = Tensor::from_matrix(Matrix(mD));

    // ── 1. 二元：A + B ────────────────────────────────────────────────
    {
        Matrix ref(R, C); ref_binary(ref, mA, mB, '+');
        fail += check(eng, "A + B", leaf(ta) + leaf(tb), R, C, ref, kTol);
    }
    // ── 2. 融合：A * B + C（一次求值完成乘与加，编译期融合） ──────────
    {
        Matrix ab(R, C); ref_binary(ab, mA, mB, '*');
        Matrix ref(R, C); ref_binary(ref, ab, mC, '+');
        fail += check(eng, "A * B + C (fused)", leaf(ta) * leaf(tb) + leaf(tc), R, C, ref, kTol);
    }
    // ── 3. 嵌套融合：(A + B) * (C - D) ────────────────────────────────
    {
        Matrix ab(R, C); ref_binary(ab, mA, mB, '+');
        Matrix cd(R, C); ref_binary(cd, mC, mD, '-');
        Matrix ref(R, C); ref_binary(ref, ab, cd, '*');
        fail += check(eng, "(A+B)*(C-D)", (leaf(ta) + leaf(tb)) * (leaf(tc) - leaf(td)), R, C, ref, kTol);
    }
    // ── 4. 标量广播：A * 2 + B 与 2 * A ────────────────────────────────
    {
        Matrix ref(R, C);
        for (std::size_t i = 0; i < mA.size(); ++i) ref.span()[i] = mA.span()[i] * Scalar{2} + mB.span()[i];
        fail += check(eng, "A*2 + B (scalar)", leaf(ta) * Scalar{2} + leaf(tb), R, C, ref, kTol);
    }
    {
        Matrix ref(R, C);
        for (std::size_t i = 0; i < mA.size(); ++i) ref.span()[i] = Scalar{2} * mA.span()[i];
        fail += check(eng, "2 * A (scalar-left)", Scalar{2} * leaf(ta), R, C, ref, kTol);
    }
    // ── 5. 一元融合：relu(A) + tanh(B) ─────────────────────────────────
    {
        Matrix ref(R, C);
        for (std::size_t i = 0; i < mA.size(); ++i)
            ref.span()[i] = std::max(mA.span()[i], Scalar{0}) + std::tanh(mB.span()[i]);
        fail += check(eng, "relu(A)+tanh(B)", relu(leaf(ta)) + tanh(leaf(tb)), R, C, ref, kTol);
    }
    // ── 6. 比较 + 选择：mask = (A > B) ? 1 : 0 ─────────────────────────
    {
        Matrix ref(R, C);
        for (std::size_t i = 0; i < mA.size(); ++i)
            ref.span()[i] = (mA.span()[i] > mB.span()[i]) ? Scalar{1} : Scalar{0};
        fail += check(eng, "A > B ? 1 : 0 (select)",
                      select(leaf(ta) > leaf(tb), Scalar{1}, Scalar{0}), R, C, ref, kTol);
    }
    // ── 7. matmul 边界 + 逐元素：matmul(A2x3, B3x4) + C ───────────────
    {
        Matrix a2(2, 3), b3(3, 4), c2(2, 4);
        fill_random(a2, rng, dist); fill_random(b3, rng, dist); fill_random(c2, rng, dist);
        Matrix ref(2, 4);
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 4; ++j)
            {
                Scalar s = 0;
                for (std::size_t k = 0; k < 3; ++k) s += a2.span()[i * 3 + k] * b3.span()[k * 4 + j];
                ref.span()[i * 4 + j] = s + c2.span()[i * 4 + j];
            }
        auto mm = eng.matmul(Tensor::from_matrix(Matrix(a2)), Tensor::from_matrix(Matrix(b3)), false, false);
        if (!mm) { std::cerr << "[FAIL] matmul 原语\n"; ++fail; }
        else
        {
            const Tensor mmt = std::move(*mm);
            const Tensor c2t = Tensor::from_matrix(Matrix(c2));
            fail += check(eng, "matmul(A,B)+C", leaf(mmt) + leaf(c2t), 2, 4, ref, kTol);
        }
    }

    std::cout << (fail == 0 ? "\n全部通过 ✅\n" : "\n存在失败 ❌\n");
    return fail == 0 ? 0 : 1;
}
