// ───────────────────────────────────────────────────────────────────────────
//  expr_matmul_test.cpp — 算子融合二期（docs/14）：matmul 参与 IR 融合（S1+S2）
//
//  对应 docs/14-operator-fusion-2.md 阶段 S1（IR 地基）与 S2（CPU 正确性）：
//    1. ExprSpec 增加 MatmulSpec（前置 matmul 段）+ ExprOperandKind::Matmul：
//       C = op(A,B) 作为逐元素链的"虚拟寄存器 0"，链内按 (r,c) 读取。
//    2. CpuEngine::eval_expr 支持 matmul 段（matmul 预计算 + 逐元素链），
//       CPU err=0（与参考实现逐位/容差一致）。
//    3. DSL matmul(A,B) 叶子：折叠为 matmul 段，dsl::compute 自动分流引擎。
//    4. key 与形状无关：k（求和维度）不进 key，同结构不同 K 共享融合 shader；
//       transA/transB/a_input/b_input 是结构，进 key。
//    5. matmul 输出可被归约指令消费（row_sum(matmul(x))，CPU 参考路径）。
//
//  纯 CPU、无 GPU 依赖。编译/运行：
//    clang++ -std=c++26 -I include/neuralnet.cpp src/expr_matmul_test.cpp -o build/expr_matmul_test
//    ./build/expr_matmul_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/compute_cpu_engine.hpp>

// 测试写在全局作用域（非 namespace nn），避免与旧代数运算符的 ADL 歧义。
using namespace nn::dsl;
namespace dsl = nn::dsl;   // 别名：全局作用域可用 dsl::leaf 等限定调用
using nn::Scalar;

namespace
{

int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("[FAIL] %s\n", msg);                                 \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

nn::Tensor make_tensor(std::size_t rows, std::size_t cols, float base = 0.0f, float step = 0.01f)
{
    nn::Tensor t = nn::Tensor::cpu(rows, cols);
    auto sp = t.cpu_matrix().span();
    for (std::size_t i = 0; i < sp.size(); ++i)
        sp[i] = base + static_cast<float>(i) * step;
    return t;
}

// 逐元素比对（容差 1e-4，Scalar=float）
void check_close(const nn::Tensor& got, const nn::Tensor& ref, const char* msg)
{
    if (got.rows() != ref.rows() || got.cols() != ref.cols())
    {
        std::printf("[FAIL] %s: shape mismatch got %zux%zu ref %zux%zu\n",
                    msg, got.rows(), got.cols(), ref.rows(), ref.cols());
        ++g_fail;
        return;
    }
    const auto g = got.cpu_matrix().span();
    const auto r = ref.cpu_matrix().span();
    bool ok = true;
    for (std::size_t i = 0; i < g.size(); ++i)
    {
        const float d = std::fabs(g[i] - r[i]);
        const float scale = std::max(1.0f, std::fabs(r[i]));
        if (d > 1e-4f * scale)
        {
            std::printf("[FAIL] %s: [%zu] got %.6f ref %.6f\n", msg, i, g[i], r[i]);
            ok = false;
            break;
        }
    }
    if (ok) std::printf("[ OK ] %s\n", msg);
    else    ++g_fail;
}

// ── 参考实现：独立 matmul + bias + relu ──────────────────────────────────
nn::Tensor ref_matmul_bias_relu(const nn::Tensor& A, const nn::Tensor& B,
                                const nn::Tensor& bias)
{
    nn::CpuEngine eng;
    auto m = eng.matmul(A, B, false, false);
    auto t = eng.elementwise_binary(nn::BinaryOp::Add, *m, bias);
    auto sp = t->cpu_matrix().span();
    for (auto& v : sp)
        v = std::max(v, Scalar{0});
    return *t;
}

// ── 1) 纯 matmul（无逐元素链）：输出 = matmul 结果本身 ───────────────────
void test_pure_matmul()
{
    const std::size_t M = 4, K = 5, N = 3;
    nn::CpuEngine eng;
    const nn::Tensor A = make_tensor(M, K, 0.5f, 0.03f);
    const nn::Tensor B = make_tensor(K, N, -0.5f, 0.02f);

    // 直接构造 spec（结构 = DSL 折叠产物）
    nn::ExprSpec s;
    s.views    = {nn::expr::linear(), nn::expr::linear()};
    s.num_regs = 0;
    s.matmul   = nn::MatmulSpec{/*a_input=*/0, /*b_input=*/1, 0, 0, static_cast<std::uint32_t>(K)};
    auto v = nn::validate_expr_spec(s, 2);
    CHECK(static_cast<bool>(v), "纯 matmul spec 校验应通过（允许空指令表）");

    auto r = eng.eval_expr(s, std::vector<nn::Tensor>{A, B}, M, N);
    if (!r) { CHECK(false, "纯 matmul eval_expr 失败"); return; }
    auto ref = eng.matmul(A, B, false, false);
    check_close(*r, *ref, "纯 matmul：eval_expr == 独立 matmul");
}

// ── 2) matmul + bias（matmul 段 + 尾逐元素链 Add）───────────────────────
void test_matmul_bias()
{
    const std::size_t M = 4, K = 5, N = 3;
    nn::CpuEngine eng;
    const nn::Tensor A = make_tensor(M, K, 0.5f, 0.03f);
    const nn::Tensor B = make_tensor(K, N, -0.5f, 0.02f);
    const nn::Tensor bias = make_tensor(M, N, 0.2f, 0.01f);

    nn::ExprSpec s;
    s.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
    s.num_regs = 1;
    s.matmul   = nn::MatmulSpec{0, 1, 0, 0, static_cast<std::uint32_t>(K)};
    // Add r0 = matmul + bias（Input 2）
    s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                        nn::expr::matmul_op(), nn::expr::input(2), {}});
    auto v = nn::validate_expr_spec(s, 3);
    CHECK(static_cast<bool>(v), "matmul+bias spec 校验应通过");

    auto r = eng.eval_expr(s, std::vector<nn::Tensor>{A, B, bias}, M, N);
    if (!r) { CHECK(false, "matmul+bias eval_expr 失败"); return; }
    auto ref = eng.elementwise_binary(nn::BinaryOp::Add,
                                      *eng.matmul(A, B, false, false), bias);
    check_close(*r, *ref, "matmul+bias：eval_expr == 参考");

    // DSL 路径：dsl::matmul(A,B) + bias
    auto rd = nn::dsl::compute(eng, dsl::matmul(A, B) + dsl::leaf(bias), M, N);
    if (!rd) { CHECK(false, "DSL matmul+bias compute 失败"); return; }
    check_close(*rd, *ref, "DSL matmul+bias：dsl::compute == 参考");
}

// ── 3) matmul + bias + relu（更长的尾链，经 canonicalize）────────────────
void test_matmul_bias_relu()
{
    const std::size_t M = 3, K = 4, N = 5;
    nn::CpuEngine eng;
    const nn::Tensor A = make_tensor(M, K, -0.3f, 0.05f);
    const nn::Tensor B = make_tensor(K, N, 0.1f, -0.02f);
    const nn::Tensor bias = make_tensor(M, N, -0.4f, 0.03f);

    // DSL：max(matmul(A,B) + bias, 0) —— canonicalize 后与 scan 端一致
    auto r = nn::dsl::compute(eng,
        dsl::max(dsl::matmul(A, B) + dsl::leaf(bias), Scalar{0}), M, N);
    if (!r) { CHECK(false, "DSL matmul+bias+relu compute 失败"); return; }

    auto ref = ref_matmul_bias_relu(A, B, bias);
    check_close(*r, ref, "matmul+bias+relu：dsl::compute == 参考");

    // 折叠出的 spec 应含 matmul 段
    auto [spec, inputs] = nn::dsl::to_expr_spec(
        dsl::max(dsl::matmul(A, B) + dsl::leaf(bias), Scalar{0}));
    CHECK(nn::expr_spec_has_matmul(spec), "折叠 spec 应含 matmul 段");
    CHECK(inputs.size() == 3, "折叠 spec 应有 3 个输入（A/B/bias）");
}

// ── 4) transA / transB 变体（转置标志是结构，进 key）────────────────────
void test_transpose_variants()
{
    const std::size_t M = 4, K = 5, N = 3;
    nn::CpuEngine eng;
    // A0: (M,K)；At: A^T 存储 (K,M)；B0: (K,N)；Bt: B^T 存储 (N,K)
    const nn::Tensor A0 = make_tensor(M, K, 0.2f, 0.04f);
    const nn::Tensor At = make_tensor(K, M, 0.2f, 0.04f);   // A^T
    const nn::Tensor B0 = make_tensor(K, N, -0.1f, 0.03f);
    const nn::Tensor Bt = make_tensor(N, K, -0.1f, 0.03f);  // B^T
    const nn::Tensor bias = make_tensor(M, N, 0.1f, 0.01f);

    const auto run = [&](bool trA, bool trB) {
        nn::ExprSpec s;
        s.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
        s.num_regs = 1;
        s.matmul   = nn::MatmulSpec{0, 1,
                                    static_cast<std::uint8_t>(trA ? 1u : 0u),
                                    static_cast<std::uint8_t>(trB ? 1u : 0u),
                                    static_cast<std::uint32_t>(K)};
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                            nn::expr::matmul_op(), nn::expr::input(2), {}});
        const nn::Tensor& A = trA ? At : A0;
        const nn::Tensor& B = trB ? Bt : B0;
        auto r = eng.eval_expr(s, std::vector<nn::Tensor>{A, B, bias}, M, N);
        if (!r) { CHECK(false, "transpose variant eval_expr 失败"); return std::string{}; }
        auto ref = eng.elementwise_binary(nn::BinaryOp::Add,
                                          *eng.matmul(A, B, trA, trB), bias);
        check_close(*r, *ref, trA && trB ? "matmul(A^T,B^T)+bias == 参考"
                                        : (trA ? "matmul(A^T,B)+bias == 参考"
                                               : (trB ? "matmul(A,B^T)+bias == 参考"
                                                      : "matmul(A,B)+bias == 参考")));
        return nn::expr_spec_key(s);
    };

    const std::string k00 = run(false, false);
    const std::string k01 = run(false, true);
    const std::string k10 = run(true, false);
    const std::string k11 = run(true, true);
    CHECK(k00 != k01 && k01 != k10 && k10 != k11 && k00 != k11,
          "transA/transB 是结构：不同转置组合 key 必须不同");
}

// ── 5) key 与形状无关：k 不进 key，同结构不同 K 共享融合 shader ────────
void test_key_shape_invariance()
{
    auto make_spec = [](std::uint32_t k) {
        nn::ExprSpec s;
        s.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
        s.num_regs = 1;
        s.matmul   = nn::MatmulSpec{0, 1, 0, 0, k};
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                            nn::expr::matmul_op(), nn::expr::input(2), {}});
        return s;
    };
    // canonicalize 后 key 一致（k 不参与哈希）
    const std::string k8  = nn::expr_spec_key(nn::canonicalize_expr_spec(make_spec(8)));
    const std::string k32 = nn::expr_spec_key(nn::canonicalize_expr_spec(make_spec(32)));
    const std::string k127 = nn::expr_spec_key(nn::canonicalize_expr_spec(make_spec(127)));
    CHECK(k8 == k32 && k32 == k127,
          "k（求和维度）是形状参数：不同 K 的 key 必须相同（共享融合 shader）");

    // 无 matmul 段的同类表达式 key 必须不同（matmul 段结构进 key）
    nn::ExprSpec s_nomm;
    s_nomm.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
    s_nomm.num_regs = 1;
    s_nomm.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                             nn::expr::input(0), nn::expr::input(2), {}});
    const std::string knomm = nn::expr_spec_key(nn::canonicalize_expr_spec(s_nomm));
    CHECK(knomm != k8, "有/无 matmul 段的 spec key 必须不同");

    // 不同 k 的实际求值正确（用 k=8 的 spec 在 K=8 输入上求值）
    nn::CpuEngine eng;
    const std::size_t M = 3, K = 8, N = 2;
    const nn::Tensor A = make_tensor(M, K, 0.4f, 0.02f);
    const nn::Tensor B = make_tensor(K, N, 0.3f, -0.01f);
    const nn::Tensor bias = make_tensor(M, N, 0.0f, 0.05f);
    auto r = eng.eval_expr(make_spec(static_cast<std::uint32_t>(K)),
                           std::vector<nn::Tensor>{A, B, bias}, M, N);
    if (!r) { CHECK(false, "K=8 求值失败"); return; }
    auto ref = eng.elementwise_binary(nn::BinaryOp::Add,
                                      *eng.matmul(A, B, false, false), bias);
    check_close(*r, *ref, "K=8 求值 == 参考（k 只影响形状，不影响结构）");
}

// ── 6) matmul 输出被归约指令消费：row_sum(matmul(x))（CPU 参考路径）──────
void test_matmul_reduce_instr()
{
    const std::size_t M = 3, K = 4, N = 5;
    nn::CpuEngine eng;
    const nn::Tensor A = make_tensor(M, K, 0.2f, 0.03f);
    const nn::Tensor B = make_tensor(K, N, -0.2f, 0.04f);

    // row_sum(matmul(A,B))：归约指令 RowSum 的源是 Matmul 操作数
    nn::ExprSpec s;
    s.views    = {nn::expr::linear(), nn::expr::linear()};
    s.num_regs = 1;
    s.matmul   = nn::MatmulSpec{0, 1, 0, 0, static_cast<std::uint32_t>(K)};
    s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::RowSum), 0,
                        nn::expr::matmul_op(), {}, {}});
    auto v = nn::validate_expr_spec(s, 2);
    CHECK(static_cast<bool>(v), "row_sum(matmul) spec 校验应通过");
    CHECK(nn::expr_spec_reduce_axis(s) == 0, "row_sum(matmul) 归约轴应为行(0)");

    // eval_expr_reduce：输出 (rows, 1)
    auto r = eng.eval_expr_reduce(s, std::vector<nn::Tensor>{A, B}, M, N);
    if (!r) { CHECK(false, "row_sum(matmul) eval_expr_reduce 失败"); return; }
    CHECK(r->rows() == M && r->cols() == 1, "row_sum(matmul) 输出形状应为 (M,1)");
    auto mm = eng.matmul(A, B, false, false);
    auto ref = eng.row_reduce_sum(*mm);
    check_close(*r, *ref, "row_sum(matmul) == 参考 row_reduce_sum(matmul)");

    // 广播输出：eval_expr（非 vector_out）应广播归约结果
    auto rb = eng.eval_expr(s, std::vector<nn::Tensor>{A, B}, M, N);
    if (!rb) { CHECK(false, "row_sum(matmul) eval_expr 失败"); return; }
    nn::Tensor refb = nn::Tensor::cpu(M, N);
    {
        auto d = refb.cpu_matrix().span();
        const auto rr = ref->cpu_matrix().span();
        for (std::size_t r = 0; r < M; ++r)
            for (std::size_t c = 0; c < N; ++c)
                d[r * N + c] = rr[r];
    }
    check_close(*rb, refb, "row_sum(matmul) 广播输出 == 参考");
}

// ── 7) 校验：Matmul 操作数无 matmul 段 → 拒绝 ───────────────────────────
void test_validation()
{
    nn::ExprSpec s;
    s.views    = {nn::expr::linear(), nn::expr::linear()};
    s.num_regs = 1;
    // 无 matmul 段却使用 Matmul 操作数
    s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                        nn::expr::matmul_op(), nn::expr::input(1), {}});
    auto v = nn::validate_expr_spec(s, 2);
    CHECK(!v, "无 matmul 段的 Matmul 操作数必须被校验拒绝");

    // matmul A/B 输入越界
    nn::ExprSpec s2;
    s2.views    = {nn::expr::linear()};
    s2.num_regs = 0;
    s2.matmul   = nn::MatmulSpec{5, 1, 0, 0, 4};
    auto v2 = nn::validate_expr_spec(s2, 1);
    CHECK(!v2, "matmul 输入下标越界必须被校验拒绝");
}

// ── 8) 纯 matmul 经 DSL：dsl::matmul(A,B) 单独使用 ───────────────────────
void test_dsl_pure_matmul()
{
    const std::size_t M = 4, K = 6, N = 2;
    nn::CpuEngine eng;
    const nn::Tensor A = make_tensor(M, K, 0.3f, 0.02f);
    const nn::Tensor B = make_tensor(K, N, 0.1f, 0.03f);
    auto r = nn::dsl::compute(eng, dsl::matmul(A, B), M, N);
    if (!r) { CHECK(false, "DSL 纯 matmul compute 失败"); return; }
    auto ref = eng.matmul(A, B, false, false);
    check_close(*r, *ref, "DSL 纯 matmul == 独立 matmul");
}

} // namespace

int main()
{
    std::printf("========================================\n");
    std::printf("  expr_matmul_test（S1+S2：matmul 参与 IR 融合）\n");
    std::printf("========================================\n");

    test_pure_matmul();
    test_matmul_bias();
    test_matmul_bias_relu();
    test_transpose_variants();
    test_key_shape_invariance();
    test_matmul_reduce_instr();
    test_validation();
    test_dsl_pure_matmul();

    std::printf(g_fail == 0 ? "ALL PASS\n" : "FAILED (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
