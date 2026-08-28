// ───────────────────────────────────────────────────────────────────────────
//  perf_smoke.cpp — 阶段1 CPU 优化 Release 抽检（手动编译，不注册 CMake）
//
//  验证对象（对应 docs/13-optimize-proposal-list.md 阶段1）：
//    1. eval_expr matmul 段：分块 span 内核 vs 旧三层标量循环（0,0)/(1,0)/(0,1)
//    2. broadcast_row/col_inplace：行并行无取模 vs 旧 flat for_each + i%C/i/C
//    3. col_reduce：R=768（新门槛 256 触发并行）vs 串行参考
//
//  编译（与 CMake Release 同旗标）：
//    clang++ -std=c++26 -fno-exceptions -Wall -Wextra -Wpedantic -Werror \
//            -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native \
//            -I include perf_smoke.cpp -o perf_smoke.exe
// ───────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <ranges>
#include <vector>

#include <neuralnet.cpp/algebra_matrix.hpp>

using nn::Scalar;

namespace
{

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void fill_random(std::vector<Scalar>& v, unsigned seed)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<Scalar> d(-1.0f, 1.0f);
    for (auto& x : v)
        x = d(g);
}

// ── 参考实现：旧 eval_expr 三层标量循环 ─────────────────────────────────────
// (0,0): r = A*B，A(M,K) B(K,N)
void naive_matmul(std::vector<Scalar>& r, const std::vector<Scalar>& a,
                  const std::vector<Scalar>& b,
                  std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            Scalar s = 0;
            for (std::size_t kk = 0; kk < K; ++kk)
                s += a[i * K + kk] * b[kk * N + j];
            r[i * N + j] = s;
        }
}

// (1,0): r = A^T*B，A 存 (K,M)，B(K,N)
void naive_transpose_matmul(std::vector<Scalar>& r, const std::vector<Scalar>& a,
                            const std::vector<Scalar>& b,
                            std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            Scalar s = 0;
            for (std::size_t kk = 0; kk < K; ++kk)
                s += a[kk * M + i] * b[kk * N + j];
            r[i * N + j] = s;
        }
}

// (0,1): r = A*B^T，A(M,K)，B 存 (N,K)
void naive_multiply_transposed(std::vector<Scalar>& r, const std::vector<Scalar>& a,
                               const std::vector<Scalar>& b,
                               std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j)
        {
            Scalar s = 0;
            for (std::size_t kk = 0; kk < K; ++kk)
                s += a[i * K + kk] * b[j * K + kk];
            r[i * N + j] = s;
        }
}

double max_abs_err(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > e)
            e = d;
    }
    return e;
}

// ── 1. matmul 内核对比 ─────────────────────────────────────────────────────
void bench_matmul()
{
    const std::size_t M = 768, N = 768, K = 768;
    std::vector<Scalar> a(M * K), b(K * N), r1(M * N), r2(M * N);
    fill_random(a, 1);
    fill_random(b, 2);

    std::span<const Scalar> asp(a.data(), a.size());
    std::span<const Scalar> bsp(b.data(), b.size());
    std::span<Scalar> rsp(r2.data(), r2.size());

    // ── (0,0)：multiply_to_span vs naive ──
    naive_matmul(r1, a, b, M, N, K);
    std::fill(r2.begin(), r2.end(), 0.0f);
    nn::Matrix::multiply_to_span(rsp, M, N, asp, M, K, bsp, K, N);
    std::printf("[matmul 00] err vs naive: %.3e\n", max_abs_err(r1, r2));

    double t_naive = 1e30, t_blocked = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        naive_matmul(r1, a, b, M, N, K);
        t_naive = std::min(t_naive, ms_since(t0));

        std::fill(r2.begin(), r2.end(), 0.0f);
        t0 = std::chrono::steady_clock::now();
        nn::Matrix::multiply_to_span(rsp, M, N, asp, M, K, bsp, K, N);
        t_blocked = std::min(t_blocked, ms_since(t0));
    }
    std::printf("[matmul 00] 768^3  naive %.2fms  blocked %.2fms  speedup %.2fx\n",
                t_naive, t_blocked, t_naive / t_blocked);

    // ── (1,0)：transpose_multiply_to_span（A 存 (K,M)）vs naive ──
    std::vector<Scalar> at(K * M);  // a 转置存储
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t kk = 0; kk < K; ++kk)
            at[kk * M + i] = a[i * K + kk];
    std::span<const Scalar> atsp(at.data(), at.size());

    naive_transpose_matmul(r1, at, b, M, N, K);
    std::fill(r2.begin(), r2.end(), 0.0f);
    nn::Matrix::transpose_multiply_to_span(rsp, M, N, atsp, K, M, bsp, K, N);
    std::printf("[matmul 10] err vs naive: %.3e\n", max_abs_err(r1, r2));

    t_naive = 1e30;
    t_blocked = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        naive_transpose_matmul(r1, at, b, M, N, K);
        t_naive = std::min(t_naive, ms_since(t0));

        std::fill(r2.begin(), r2.end(), 0.0f);
        t0 = std::chrono::steady_clock::now();
        nn::Matrix::transpose_multiply_to_span(rsp, M, N, atsp, K, M, bsp, K, N);
        t_blocked = std::min(t_blocked, ms_since(t0));
    }
    std::printf("[matmul 10] 768^3  naive %.2fms  blocked %.2fms  speedup %.2fx\n",
                t_naive, t_blocked, t_naive / t_blocked);

    // ── (0,1)：multiply_transposed_to_span（B 存 (N,K)）vs naive ──
    std::vector<Scalar> bt(N * K);
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t kk = 0; kk < K; ++kk)
            bt[j * K + kk] = b[kk * N + j];
    std::span<const Scalar> btsp(bt.data(), bt.size());

    naive_multiply_transposed(r1, a, bt, M, N, K);
    std::fill(r2.begin(), r2.end(), 0.0f);
    nn::Matrix::multiply_transposed_to_span(rsp, M, N, asp, M, K, btsp, N, K);
    std::printf("[matmul 01] err vs naive: %.3e\n", max_abs_err(r1, r2));

    t_naive = 1e30;
    t_blocked = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        naive_multiply_transposed(r1, a, bt, M, N, K);
        t_naive = std::min(t_naive, ms_since(t0));

        std::fill(r2.begin(), r2.end(), 0.0f);
        t0 = std::chrono::steady_clock::now();
        nn::Matrix::multiply_transposed_to_span(rsp, M, N, asp, M, K, btsp, N, K);
        t_blocked = std::min(t_blocked, ms_since(t0));
    }
    std::printf("[matmul 01] 768^3  naive %.2fms  blocked %.2fms  speedup %.2fx\n",
                t_naive, t_blocked, t_naive / t_blocked);
}

// ── 2. broadcast：行并行无取模 vs 旧 flat for_each + i%C / i/C ────────────
void bench_broadcast()
{
    const std::size_t R = 768, C = 1024;  // R*C = 786432 >= PARALLEL_THRESHOLD
    nn::Matrix m(R, C);
    nn::Matrix m2(R, C);
    nn::Matrix col(1, C);
    nn::Matrix row(R, 1);
    // 与 fill_random 同分布直接填充矩阵数据
    std::mt19937 g(3);
    std::uniform_real_distribution<Scalar> d(-1.0f, 1.0f);
    {
        auto sp = m.span();
        auto sp2 = m2.span();
        auto sc = col.span();
        auto sr = row.span();
        for (std::size_t i = 0; i < sp.size(); ++i)
        {
            const Scalar v = d(g);
            sp[i] = v;
            sp2[i] = v;
        }
        for (std::size_t i = 0; i < sc.size(); ++i)
            sc[i] = d(g);
        for (std::size_t i = 0; i < sr.size(); ++i)
            sr[i] = d(g);
    }

    // ── col 广播：新（行并行直接 v[c]） vs 旧（flat for_each + i%C） ──
    auto old_col = [&]() {
        const auto v = col.span();
        auto dm = m2.span();
        auto idx = std::views::iota(std::size_t{0}, dm.size());
        nn::for_each(idx.begin(), idx.end(),
            [&dm, &v](std::size_t i) noexcept { dm[i] = dm[i] + v[i % C]; });
    };
    auto new_col = [&]() {
        m.broadcast_col_inplace(col, [](Scalar x, Scalar y) noexcept { return x + y; });
    };
    old_col();
    new_col();
    double err = 0;
    {
        auto s1 = m.span();
        auto s2 = m2.span();
        for (std::size_t i = 0; i < s1.size(); ++i)
            err = std::max(err, static_cast<double>(std::abs(s1[i] - s2[i])));
    }
    std::printf("[broadcast col] err vs old: %.3e\n", err);

    double t_old = 1e30, t_new = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        old_col();
        t_old = std::min(t_old, ms_since(t0));

        t0 = std::chrono::steady_clock::now();
        new_col();
        t_new = std::min(t_new, ms_since(t0));
    }
    std::printf("[broadcast col] 768x1024  old %.2fms  new %.2fms  speedup %.2fx\n",
                t_old, t_new, t_old / t_new);

    // ── row 广播：新 vs 旧（flat for_each + i/C） ──
    auto old_row = [&]() {
        const auto v = row.span();
        auto dm = m2.span();
        auto idx = std::views::iota(std::size_t{0}, dm.size());
        nn::for_each(idx.begin(), idx.end(),
            [&dm, &v](std::size_t i) noexcept { dm[i] = dm[i] + v[i / C]; });
    };
    auto new_row = [&]() {
        m.broadcast_row_inplace(row, [](Scalar x, Scalar y) noexcept { return x + y; });
    };
    old_row();
    new_row();
    err = 0;
    {
        auto s1 = m.span();
        auto s2 = m2.span();
        for (std::size_t i = 0; i < s1.size(); ++i)
            err = std::max(err, static_cast<double>(std::abs(s1[i] - s2[i])));
    }
    std::printf("[broadcast row] err vs old: %.3e\n", err);

    t_old = 1e30;
    t_new = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        old_row();
        t_old = std::min(t_old, ms_since(t0));

        t0 = std::chrono::steady_clock::now();
        new_row();
        t_new = std::min(t_new, ms_since(t0));
    }
    std::printf("[broadcast row] 768x1024  old %.2fms  new %.2fms  speedup %.2fx\n",
                t_old, t_new, t_old / t_new);
}

// ── 3. col_reduce：R=768（新门槛 256 触发并行） vs 串行参考 ────────────────
void bench_col_reduce()
{
    const std::size_t R = 768, C = 1024;
    nn::Matrix m(R, C);
    std::mt19937 g(7);
    std::uniform_real_distribution<Scalar> d(-1.0f, 1.0f);
    {
        auto sp = m.span();
        for (std::size_t i = 0; i < sp.size(); ++i)
            sp[i] = d(g);
    }

    auto serial_ref = [&]() {
        nn::Matrix out(1, C);
        auto o = out.span();
        const auto sm = m.span();
        for (std::size_t c = 0; c < C; ++c)
        {
            Scalar acc = 0;
            for (std::size_t r = 0; r < R; ++r)
                acc += sm[r * C + c];
            o[c] = acc;
        }
        return out;
    };
    auto new_reduce = [&]() {
        return m.col_reduce(Scalar{0},
            [](Scalar a, Scalar b) noexcept { return a + b; },
            [](Scalar x) noexcept { return x; });
    };

    nn::Matrix ref = serial_ref();
    nn::Matrix got = new_reduce();
    double err = 0;
    {
        auto s1 = ref.span();
        auto s2 = got.span();
        for (std::size_t i = 0; i < s1.size(); ++i)
            err = std::max(err, static_cast<double>(std::abs(s1[i] - s2[i])));
    }
    std::printf("[col_reduce] 768x1024 err vs serial: %.3e\n", err);

    double t_serial = 1e30, t_new = 1e30;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        (void)serial_ref();
        t_serial = std::min(t_serial, ms_since(t0));

        t0 = std::chrono::steady_clock::now();
        (void)new_reduce();
        t_new = std::min(t_new, ms_since(t0));
    }
    std::printf("[col_reduce] 768x1024  serial %.2fms  new %.2fms  speedup %.2fx\n",
                t_serial, t_new, t_serial / t_new);
}

} // namespace

int main()
{
    std::printf("=== neuralnet.cpp 阶段1 优化 Release 抽检 ===\n");
    bench_matmul();
    bench_broadcast();
    bench_col_reduce();
    std::printf("done\n");
    return 0;
}
