// ── batched_matmul 原语探针 ─────────────────────────────────────────────
//
// 目的：直接对比 CpuEngine::batched_matmul 四种转置组合与手工参考计算。
//       若某组合输出不匹配，即证明底层原语有 bug。
//
// 语义（见 cpu_engine.hpp::batched_matmul）：
//   A 按行切成 batch 个纵向块（每块 a_rows_per × a.cols()），B 同理。
//   transA=true 表示使用 A 块的转置，transB=true 表示使用 B 块的转置。
//   块内维度须匹配：所有四种转置组合都要求块为方阵（a_rows_per == a.cols()）。
//
// 用法：matmul_probe [--gpu|--cuda]
//   --gpu   使用 Vulkan GpuEngine（验证 tiled batched_matmul shader）
//   --cuda  使用 CUDA CudaEngine
//   默认    CpuEngine
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <string_view>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;

namespace {

// 逐块参考：C_b = A_b^T × B_b（A_b/B_b 为原矩阵中的第 bi 个行块）
// A_b 以 (a_rows_per, a_cols) 行主序存储；transA/transB 语义与引擎一致。
Matrix ref_batched(const Matrix& A, const Matrix& B,
                   std::size_t batch, bool transA, bool transB)
{
    const std::size_t a_rows_per = A.rows() / batch;
    const std::size_t b_rows_per = B.rows() / batch;

    const std::size_t M  = transA ? A.cols() : a_rows_per;
    const std::size_t N  = transB ? b_rows_per : B.cols();

    Matrix C(batch * M, N);
    for (std::size_t bi = 0; bi < batch; ++bi)
    {
        const std::size_t a_off = bi * a_rows_per;
        const std::size_t b_off = bi * b_rows_per;
        for (std::size_t i = 0; i < M; ++i)
        {
            for (std::size_t j = 0; j < N; ++j)
            {
                Scalar s{0};
                // 块内 K 维 = 逻辑 A 的列数 = 逻辑 B 的行数
                const std::size_t K = transA ? a_rows_per : A.cols();
                for (std::size_t k = 0; k < K; ++k)
                {
                    // A_b[i][k]（考虑转置）
                    const Scalar a_val = transA
                        ? A.at_unchecked(a_off + k, i)         // A_b^T[i][k] = A_b[k][i]
                        : A.at_unchecked(a_off + i, k);
                    // B_b[k][j]（考虑转置）
                    const Scalar b_val = transB
                        ? B.at_unchecked(b_off + j, k)         // B_b^T[k][j] = B_b[j][k]
                        : B.at_unchecked(b_off + k, j);
                    s += a_val * b_val;
                }
                C.set_value_unchecked(bi * M + i, j, s);
            }
        }
    }
    return C;
}

void dump_diff(const std::string& name, const Matrix& got, const Matrix& ref)
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
                if (bad < 5)
                    std::cout << "  [FAIL] " << name << "(" << i << "," << j
                              << ") got=" << got.at_unchecked(i, j)
                              << " ref=" << ref.at_unchecked(i, j)
                              << " err=" << e << "\n";
                ++bad;
            }
        }
    std::cout << "  " << name << ": " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    nn::cli::EngineConfig ecfg;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg{argv[i]};
        if (arg == "--gpu") ecfg.use_gpu = true;
        else if (arg == "--cuda") ecfg.use_cuda = true;
        else { std::cerr << "未知参数: " << arg << "\n"; return 1; }
    }
    auto engine = nn::cli::create_engine(ecfg, std::cout);
    if (!engine) { std::cerr << "引擎创建失败\n"; return 1; }
    ComputeEngine& eng = *engine;

    std::cout << "========================================\n";
    std::cout << "  batched_matmul 原语探针\n";
    std::cout << "========================================\n";

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    // 方块尺寸 K × K，H 个块纵向堆叠 → A/B 均为 (H*K, K)
    // 该形状使四种转置组合的块内维度全部合法。
    const std::size_t K = 4;
    const std::size_t H = 2;

    Matrix Am(H * K, K), Bm(H * K, K);
    for (std::size_t i = 0; i < Am.size(); ++i) Am.span()[i] = dist(rng);
    for (std::size_t i = 0; i < Bm.size(); ++i) Bm.span()[i] = dist(rng);

    auto A = eng.from_matrix(Am);
    auto B = eng.from_matrix(Bm);
    if (!A || !B) { std::cerr << "from_matrix failed\n"; return 1; }

    // ── (F,F): C = A × B ──
    {
        auto out = eng.batched_matmul(*A, *B, H, false, false);
        if (!out) { std::cerr << "  (F,F) error: " << out.error().message << "\n"; return 1; }
        auto got = eng.to_matrix(*out);
        if (!got) { std::cerr << "  (F,F) to_matrix failed\n"; return 1; }
        const Matrix ref = ref_batched(Am, Bm, H, false, false);
        dump_diff("(F,F) A*B", *got, ref);
    }
    // ── (F,T): C = A × B^T ──
    {
        auto out = eng.batched_matmul(*A, *B, H, false, true);
        if (!out) { std::cerr << "  (F,T) error: " << out.error().message << "\n"; return 1; }
        auto got = eng.to_matrix(*out);
        if (!got) { std::cerr << "  (F,T) to_matrix failed\n"; return 1; }
        const Matrix ref = ref_batched(Am, Bm, H, false, true);
        dump_diff("(F,T) A*B^T", *got, ref);
    }
    // ── (T,F): C = A^T × B ──
    {
        auto out = eng.batched_matmul(*A, *B, H, true, false);
        if (!out) { std::cerr << "  (T,F) error: " << out.error().message << "\n"; return 1; }
        auto got = eng.to_matrix(*out);
        if (!got) { std::cerr << "  (T,F) to_matrix failed\n"; return 1; }
        const Matrix ref = ref_batched(Am, Bm, H, true, false);
        dump_diff("(T,F) A^T*B", *got, ref);
    }
    // ── (T,T): C = A^T × B^T ──
    {
        auto out = eng.batched_matmul(*A, *B, H, true, true);
        if (!out) { std::cerr << "  (T,T) error: " << out.error().message << "\n"; return 1; }
        auto got = eng.to_matrix(*out);
        if (!got) { std::cerr << "  (T,T) to_matrix failed\n"; return 1; }
        const Matrix ref = ref_batched(Am, Bm, H, true, true);
        dump_diff("(T,T) A^T*B^T", *got, ref);
    }

    // ── 大尺寸 + alpha：K=100 不被 BK=16 整除（边界填零路径），M=N=100
    //    超过 64×64 tile（多 WorkGroup 分块路径），H=8 模拟 batch*num_heads。
    //    alpha=0.176777 模拟 attention 的 1/sqrt(d_k) 折叠。
    {
        const std::size_t K2 = 100;
        const std::size_t H2 = 8;
        const Scalar alpha{0.176777f};

        Matrix Am2(H2 * K2, K2), Bm2(H2 * K2, K2);
        for (std::size_t i = 0; i < Am2.size(); ++i) Am2.span()[i] = dist(rng);
        for (std::size_t i = 0; i < Bm2.size(); ++i) Bm2.span()[i] = dist(rng);

        auto A2 = eng.from_matrix(Am2);
        auto B2 = eng.from_matrix(Bm2);
        if (!A2 || !B2) { std::cerr << "from_matrix (large) failed\n"; return 1; }

        struct Case { const char* name; bool tA; bool tB; };
        for (const Case cs : {Case{"(F,F)", false, false}, Case{"(T,F)", true, false},
                              Case{"(F,T)", false, true},  Case{"(T,T)", true, true}})
        {
            auto out = eng.batched_matmul(*A2, *B2, H2, cs.tA, cs.tB, alpha);
            if (!out) { std::cerr << "  large " << cs.name << " error: "
                                  << out.error().message << "\n"; return 1; }
            auto got = eng.to_matrix(*out);
            if (!got) { std::cerr << "  large " << cs.name << " to_matrix failed\n"; return 1; }
            const Matrix ref = ref_batched(Am2, Bm2, H2, cs.tA, cs.tB);

            // ref 乘 alpha 后对比（参考实现不含 alpha）
            Matrix ref_s = ref;
            for (std::size_t i = 0; i < ref_s.size(); ++i)
                ref_s.span()[i] *= alpha;
            std::string label = std::string("large ") + cs.name + " alpha";
            dump_diff(label.c_str(), *got, ref_s);
        }
    }

    std::cout << "----------------------------------------\n";
    return 0;
}
