// ───────────────────────────────────────────────────────────────────────────
//  dsl_shape_infer_test.cpp — dsl::compute / compute_reduce 的**无形状重载**
//
//  背景：docs/todo/18-roadmap.md P4-4 把"`dsl::compute` 无 2 参重载"记为 API
//  摩擦点（调用方反复手写 `x.rows(), x.cols()`）。本测试锁定新重载的口径：
//
//   [1] 推导形状 == 显式形状（一族表达式：一元/二元/广播/select/matmul 段
//       四种转置 + batch + GQA/分组归约/归约视图嵌套）
//   [2] 推导路径的结果与显式路径**逐位一致**（同一个表达式，只是形状来源不同）
//   [3] 显式 P（3 参重载）透传正确，含 F16 输出
//   [4] compute_reduce 的无形状重载（输出 (rows,1)/(1,cols)）
//   [5] 负例：形状冲突 / 无可确定网格 → 明确报错（不猜形状）
//
//  纯 CPU（推理规则与后端无关）；GPU 侧的端到端覆盖由库内已转换为无形状
//  形式的调用点 + 既有 GPU 用例（attn/ce/gpt/expr_gpu）承担。
//
//  用法：dsl_shape_infer_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <string>

#include "test_common.hpp"

using nn::Scalar;

int g_fail = 0;

namespace
{

// ── 小工具 ────────────────────────────────────────────────────────────────
template <typename R>
nn::Tensor must(R&& r, const char* what)
{
    if (!r)
    {
        std::printf("[FAIL] %s: %s\n", what, r.error().message.c_str());
        ++g_fail;
        static nn::Tensor dummy;
        return dummy;
    }
    return std::move(*r);
}

// 推导路径 (2/3 参) 与显式路径 (4/5 参) 的形状必须一致
void check_same_shape(nn::CpuEngine&, const nn::Tensor& a, const nn::Tensor& b,
                      const char* what)
{
    const bool ok = (a.rows() == b.rows() && a.cols() == b.cols());
    std::printf("[%s] %s: inferred %zux%zu vs explicit %zux%zu\n",
                ok ? " OK " : "FAIL", what, a.rows(), a.cols(), b.rows(), b.cols());
    if (!ok) ++g_fail;
}

// 两条路径的数值必须**逐位**一致（同一表达式、同一形状来源不同）
void check_same_values(nn::CpuEngine& eng, const nn::Tensor& a, const nn::Tensor& b,
                       const char* what)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        std::printf("[FAIL] %s: 形状不同（%zux%zu vs %zux%zu）\n", what,
                    a.rows(), a.cols(), b.rows(), b.cols());
        ++g_fail;
        return;
    }
    const auto ma = download(eng, a);
    const auto mb = download(eng, b);
    for (std::size_t i = 0; i < ma.span().size(); ++i)
        if (ma.span()[i] != mb.span()[i])
        {
            std::printf("[FAIL] %s: [%zu] %.9g != %.9g\n", what, i,
                        static_cast<double>(ma.span()[i]),
                        static_cast<double>(mb.span()[i]));
            ++g_fail;
            return;
        }
    std::printf("[ OK ] %s: 逐位一致\n", what);
}

// 期望失败：返回错误且消息含"无法推导输出形状"（提示改用显式形式）
template <typename R>
void check_infer_error(R&& r, const char* what)
{
    const bool failed = !r;
    const bool explained = failed
        && r.error().message.find("无法推导输出形状") != std::string::npos
        && r.error().message.find("rows/cols") != std::string::npos;
    std::printf("[%s] %s%s\n", explained ? " OK " : "FAIL", what,
                failed ? ("  ->  " + r.error().message).c_str() : "  （未报错）");
    if (!explained) ++g_fail;
}

// ── [1][2] 表达式族：推导 == 显式（形状 + 逐位值）─────────────────────────
void test_infer_matches_explicit()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;

    auto t = [&](std::size_t r, std::size_t c, float base) {
        return make_tensor(eng, r, c, base, 0.01f);
    };

    // ① 一元
    {
        nn::Tensor x = t(4, 3, 0.1f);
        auto e = nn::dsl::exp(nn::dsl::leaf(x));
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "unary inferred"),
                              must(nn::dsl::compute(eng, e, x.rows(), x.cols()),
                                   "unary explicit"), "一元 exp(leaf)");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "unary inferred"),
                               must(nn::dsl::compute(eng, e, x.rows(), x.cols()),
                                    "unary explicit"), "一元 exp(leaf)");
    }
    // ② 二元（同形状）
    {
        nn::Tensor a = t(4, 3, 0.1f), b = t(4, 3, 0.2f);
        auto e = nn::dsl::leaf(a) * nn::dsl::leaf(b) + nn::dsl::rparam(Scalar{0.5});
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "binary inferred"),
                              must(nn::dsl::compute(eng, e, 4, 3), "binary explicit"),
                         "二元 a*b+k");
    }
    // ③ 列广播 (1,C) + 全网格
    {
        nn::Tensor a = t(4, 3, 0.1f), v = t(1, 3, 0.7f);
        auto e = nn::dsl::leaf(a) * nn::dsl::col_broadcast(v);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "colbcast inferred"),
                              must(nn::dsl::compute(eng, e, 4, 3), "colbcast explicit"),
                         "列广播 (1,C)");
    }
    // ④ 行广播 (F,1) + 全网格
    {
        nn::Tensor a = t(4, 3, 0.1f), v = t(4, 1, 0.7f);
        auto e = nn::dsl::leaf(a) + nn::dsl::row_broadcast(v);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "rowbcast inferred"),
                              must(nn::dsl::compute(eng, e, 4, 3), "rowbcast explicit"),
                         "行广播 (F,1)");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "rowbcast inferred"),
                               must(nn::dsl::compute(eng, e, 4, 3), "rowbcast explicit"),
                          "行广播 (F,1)");
    }
    // ⑤ 条件选择
    {
        nn::Tensor a = t(4, 3, -0.3f);
        auto e = nn::dsl::select(nn::dsl::leaf(a) > Scalar{0}, nn::dsl::leaf(a),
                                 Scalar{0});
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "select inferred"),
                              must(nn::dsl::compute(eng, e, 4, 3), "select explicit"),
                         "select(c>0,a,0)");
    }
    // ⑥ matmul + 行广播 bias（Linear::forward 的形态）──────── transA/transB 四组合
    {
        nn::Tensor W = t(5, 7, 0.01f), x = t(7, 3, 0.1f), bias = t(1, 3, 0.5f);
        auto e = nn::dsl::matmul(W, x, false, false) + nn::dsl::row_broadcast(bias);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "mm+bias inferred"),
                              must(nn::dsl::compute(eng, e, 5, 3), "mm+bias explicit"),
                         "matmul + row_broadcast(bias)");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "mm+bias inferred"),
                               must(nn::dsl::compute(eng, e, 5, 3), "mm+bias explicit"),
                          "matmul + row_broadcast(bias)");
    }
    {   // transA：A 存 (K,M)
        nn::Tensor A = t(7, 5, 0.01f), x = t(7, 3, 0.1f);
        auto e = nn::dsl::matmul(A, x, true, false);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "mm-transA inferred"),
                              must(nn::dsl::compute(eng, e, 5, 3), "mm-transA explicit"),
                         "matmul transA");
    }
    {   // transB：B 存 (N,K)
        nn::Tensor W = t(5, 7, 0.01f), B = t(3, 7, 0.1f);
        auto e = nn::dsl::matmul(W, B, false, true);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "mm-transB inferred"),
                              must(nn::dsl::compute(eng, e, 5, 3), "mm-transB explicit"),
                         "matmul transB");
    }
    // ⑦ batch>1 的 matmul（注意力 Q·Kᵀ 形态：A/B 都按 batch 垂直切分）
    {
        const std::size_t BH = 3, seq = 5, dk = 4;
        nn::Tensor Q = t(BH * dk, seq, 0.01f), K = t(BH * dk, seq, 0.02f);
        auto e = nn::dsl::matmul_kv(Q, K, true, false,
                                    static_cast<std::uint32_t>(BH), 1, 1);
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "bmm inferred"),
                              must(nn::dsl::compute(eng, e, BH * seq, seq),
                                   "bmm explicit"), "batch=BH 的 matmul (Q·Kᵀ)");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "bmm inferred"),
                               must(nn::dsl::compute(eng, e, BH * seq, seq),
                                    "bmm explicit"), "batch=BH 的 matmul (Q·Kᵀ)");
    }
    // ⑧ GQA：KV 侧只存 batch/ratio 份行块（b_kv_ratio=2）
    {
        const std::size_t BH = 4, seq = 5, dk = 4, ratio = 2;
        nn::Tensor Q = t(BH * dk, seq, 0.01f), K = t((BH / ratio) * dk, seq, 0.02f);
        auto e = nn::dsl::matmul_kv(Q, K, true, false,
                                    static_cast<std::uint32_t>(BH), 1,
                                    static_cast<std::uint32_t>(ratio));
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "gqa inferred"),
                              must(nn::dsl::compute(eng, e, BH * seq, seq),
                                   "gqa explicit"), "GQA matmul (ratio=2)");
    }
    // ⑨ 分组归约视图根（MaxPool 形态：输出 (rows/R, cols)）
    {
        const std::size_t C = 2, R = 4, PB = 3;
        nn::Tensor col = t(C * R, PB, 0.1f);
        auto e = nn::dsl::grouped_reduce_max(col, static_cast<std::uint32_t>(R))
               + nn::dsl::rparam(Scalar{0});
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "gred inferred"),
                              must(nn::dsl::compute(eng, e, C, PB), "gred explicit"),
                         "grouped_reduce_max + 0");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "gred inferred"),
                               must(nn::dsl::compute(eng, e, C, PB), "gred explicit"),
                          "grouped_reduce_max + 0");
    }
    // ⑩ 归约视图嵌套在逐元素链里（softmax 形态）：输出仍是整网格
    {
        nn::Tensor x = t(4, 3, 0.2f);
        // 归约视图本身即"按行广播向量"，直接参与算术（无需 row_broadcast 包裹）
        auto e = nn::dsl::exp(nn::dsl::leaf(x) - nn::dsl::row_reduce_max(x));
        check_same_shape(eng, must(nn::dsl::compute(eng, e), "reduce-nest inferred"),
                              must(nn::dsl::compute(eng, e, 4, 3),
                                   "reduce-nest explicit"), "归约视图嵌套（softmax 形态）");
        check_same_values(eng, must(nn::dsl::compute(eng, e), "reduce-nest inferred"),
                               must(nn::dsl::compute(eng, e, 4, 3),
                                    "reduce-nest explicit"), "归约视图嵌套（softmax 形态）");
    }
}

// ── [3] 显式 P 的 3 参重载 ────────────────────────────────────────────────
void test_explicit_precision()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    nn::Tensor a = make_tensor(eng, 4, 3, 0.25f, 0.01f);
    nn::Tensor b = make_tensor(eng, 4, 3, 0.5f, 0.02f);
    auto e = nn::dsl::leaf(a) + nn::dsl::leaf(b);

    // F16：3 参重载必须把 P 透传（输出存储精度 f16）
    nn::Tensor inf = must(nn::dsl::compute(eng, e, nn::Precision::F16), "f16 inferred");
    nn::Tensor exp_ = must(nn::dsl::compute(eng, e, 4, 3, nn::Precision::F16),
                           "f16 explicit");
    const bool ok = (inf.precision() == nn::Precision::F16)
                 && (exp_.precision() == nn::Precision::F16);
    std::printf("[%s] 3 参重载透传 P（f16 输出）\n", ok ? " OK " : "FAIL");
    if (!ok) ++g_fail;
    check_same_shape(eng, inf, exp_, "P=F16 的 3 参重载");
    check_same_values(eng, inf, exp_, "P=F16 的 3 参重载");
}

// ── [4] compute_reduce 的无形状重载 ───────────────────────────────────────
void test_compute_reduce_infer()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    nn::Tensor x = make_tensor(eng, 4, 3, 0.2f, 0.01f);
    auto e = nn::dsl::row_reduce_sum(x);            // 归约视图根
    nn::Tensor inf = must(nn::dsl::compute_reduce(eng, e), "reduce inferred");
    nn::Tensor exp_ = must(nn::dsl::compute_reduce(eng, e, 4, 3), "reduce explicit");
    const bool ok = (inf.rows() == 4 && inf.cols() == 1);
    std::printf("[%s] compute_reduce 无形状重载 → 归约向量 (rows,1)=(4,1)\n",
                ok ? " OK " : "FAIL");
    if (!ok) ++g_fail;
    check_same_shape(eng, inf, exp_, "compute_reduce 无形状重载");
    check_same_values(eng, inf, exp_, "compute_reduce 无形状重载");
}

// ── [5] 负例：不猜形状 ────────────────────────────────────────────────────
void test_negative_cases()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    // 形状冲突（4x3 与 2x3：行维既不等、又都不为 1）
    {
        nn::Tensor a = make_tensor(eng, 4, 3, 0.1f, 0.01f);
        nn::Tensor b = make_tensor(eng, 2, 3, 0.2f, 0.01f);
        check_infer_error(nn::dsl::compute(eng, nn::dsl::leaf(a) + nn::dsl::leaf(b)),
                          "形状冲突 (4,3)+(2,3) → 报错");
    }
    // 无任何可确定网格的操作数（标量 + 网格索引叶）
    check_infer_error(
        nn::dsl::compute(eng, nn::dsl::row() + nn::dsl::rparam(Scalar{0})),
        "无网格贡献者（row()+rparam）→ 报错");
    // compute_reduce 同样不猜
    check_infer_error(
        nn::dsl::compute_reduce(eng, nn::dsl::rparam(Scalar{1})),
        "compute_reduce 无网格贡献者 → 报错");
}

} // namespace

int main()
{
    test_infer_matches_explicit();
    test_explicit_precision();
    test_compute_reduce_infer();
    test_negative_cases();

    if (g_fail == 0)
    {
        std::printf("dsl_shape_infer_test: ALL PASS\n");
        return 0;
    }
    std::printf("dsl_shape_infer_test: %d FAILED\n", g_fail);
    return 1;
}
