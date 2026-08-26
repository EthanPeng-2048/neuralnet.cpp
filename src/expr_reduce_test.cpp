// ───────────────────────────────────────────────────────────────────────────
//  expr_reduce_test.cpp — M1：ExprSpec 归约语义（归约视图 + 归约指令）CPU 验证
//
//  对应 docs/09-operator-fusion.md 阶段一（M1）：
//    1. 归约视图：对输入 Tensor 直接归约出 (rows,1)/(1,cols) 标量向量，参与
//       算术时自动按行/按列广播（ExprViewKind::RowReduceSum/Max、ColReduceSum/Max）。
//    2. 归约指令：对表达式结果归约（ExprOp::RowSum/RowMax/ColSum/ColMax），
//       dst 为隐式标量向量，经 ExprOperandKind::Reduce 操作数广播访问。
//    3. dsl::compute 对含归约的表达式自动分流到引擎 eval_expr（CPU 扩展语义）。
//    4. 校验：Reduce 操作数必须引用已出现的归约指令；寄存器不得引用归约 dst。
//    5. 规范 key：归约结构参与确定性哈希（同结构同 key）。
//
//  纯 CPU、无 GPU 依赖。编译/运行：
//    clang++ -std=c++26 -I include/neuralnet.cpp src/expr_reduce_test.cpp -o build/expr_reduce_test
//    ./build/expr_reduce_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
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

// ── 测试 1：归约视图——按行求和广播 x / row_sum ─────────────────────────
void test_row_sum_broadcast()
{
    nn::CpuEngine eng;
    const std::size_t R = 4, C = 5;
    nn::Tensor x = make_tensor(R, C, 0.5f, 0.03f);

    auto out = dsl::compute(eng, dsl::leaf(x) / dsl::row_reduce_sum(x), R, C);
    CHECK(out.has_value(), "row_sum_broadcast: compute failed");

    // 参考：out[r][c] = x[r][c] / Σ_c x[r][c]
    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t r = 0; r < R; ++r)
    {
        float sum = 0.0f;
        for (std::size_t c = 0; c < C; ++c) sum += xs[r * C + c];
        for (std::size_t c = 0; c < C; ++c) rs[r * C + c] = xs[r * C + c] / sum;
    }
    check_close(*out, ref, "row_reduce_sum 广播 (x / row_sum)");
}

// ── 测试 2：归约视图——按行 max 平移 x - row_max ────────────────────────
void test_row_max_shift()
{
    nn::CpuEngine eng;
    const std::size_t R = 4, C = 6;
    nn::Tensor x = make_tensor(R, C, -1.0f, 0.07f);

    auto out = dsl::compute(eng, dsl::leaf(x) - dsl::row_reduce_max(x), R, C);
    CHECK(out.has_value(), "row_max_shift: compute failed");

    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t r = 0; r < R; ++r)
    {
        float mx = xs[r * C];
        for (std::size_t c = 1; c < C; ++c) mx = std::max(mx, xs[r * C + c]);
        for (std::size_t c = 0; c < C; ++c) rs[r * C + c] = xs[r * C + c] - mx;
    }
    check_close(*out, ref, "row_reduce_max 广播 (x - row_max)");
}

// ── 测试 3：Softmax（归约视图 + 归约指令混合）──────────────────────────
// exp(x - row_max(x)) / row_sum(exp(x - row_max(x)))
//   row_max 用归约视图（输入直接归约）；分母 row_sum 用归约指令（对 exp 结果归约）
void test_softmax_mixed()
{
    nn::CpuEngine eng;
    const std::size_t R = 3, C = 7;
    nn::Tensor x = make_tensor(R, C, -0.8f, 0.11f);

    auto expr = dsl::exp(dsl::leaf(x) - dsl::row_reduce_max(x))
              / dsl::row_reduce_sum(dsl::exp(dsl::leaf(x) - dsl::row_reduce_max(x)));
    auto out = dsl::compute(eng, expr, R, C);
    CHECK(out.has_value(), "softmax_mixed: compute failed");

    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t r = 0; r < R; ++r)
    {
        float mx = xs[r * C];
        for (std::size_t c = 1; c < C; ++c) mx = std::max(mx, xs[r * C + c]);
        float sum = 0.0f;
        for (std::size_t c = 0; c < C; ++c) sum += std::exp(xs[r * C + c] - mx);
        for (std::size_t c = 0; c < C; ++c) rs[r * C + c] = std::exp(xs[r * C + c] - mx) / sum;
    }
    check_close(*out, ref, "Softmax（视图 row_max + 指令 row_sum）");
}

// ── 测试 4：归约指令——输出即归约结果（广播）row_sum(exp(x)) ───────────
void test_reduce_instr_output()
{
    nn::CpuEngine eng;
    const std::size_t R = 4, C = 4;
    nn::Tensor x = make_tensor(R, C, 0.1f, 0.05f);

    // 纯归约指令：row_sum(exp(x))，输出 (rows,1) 向量广播到 (rows,cols)
    auto out = dsl::compute(eng, dsl::row_reduce_sum(dsl::exp(dsl::leaf(x))), R, C);
    CHECK(out.has_value(), "reduce_instr_output: compute failed");

    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t r = 0; r < R; ++r)
    {
        float sum = 0.0f;
        for (std::size_t c = 0; c < C; ++c) sum += std::exp(xs[r * C + c]);
        for (std::size_t c = 0; c < C; ++c) rs[r * C + c] = sum;
    }
    check_close(*out, ref, "归约指令 row_sum(exp(x)) 输出广播");
}

// ── 测试 5：列归约 max 广播 x * col_max ─────────────────────────────────
void test_col_max_broadcast()
{
    nn::CpuEngine eng;
    const std::size_t R = 5, C = 3;
    nn::Tensor x = make_tensor(R, C, 0.2f, 0.09f);

    auto out = dsl::compute(eng, dsl::leaf(x) * dsl::col_reduce_max(x), R, C);
    CHECK(out.has_value(), "col_max_broadcast: compute failed");

    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t c = 0; c < C; ++c)
    {
        float mx = xs[c];
        for (std::size_t r = 1; r < R; ++r) mx = std::max(mx, xs[r * C + c]);
        for (std::size_t r = 0; r < R; ++r) rs[r * C + c] = xs[r * C + c] * mx;
    }
    check_close(*out, ref, "col_reduce_max 广播 (x * col_max)");
}

// ── 测试 6：归约指令叠加归约视图——row_sum(col_sum(x)) = 全局和 ─────────
void test_reduce_of_reduce_view()
{
    nn::CpuEngine eng;
    const std::size_t R = 3, C = 4;
    nn::Tensor x = make_tensor(R, C, 1.0f, 0.02f);

    // col_sum(x) 为归约视图 (1,C)；row_sum(该视图) 为归约指令 → 全局和
    auto out = dsl::compute(eng, dsl::row_reduce_sum(dsl::col_reduce_sum(x)), R, C);
    CHECK(out.has_value(), "reduce_of_reduce_view: compute failed");

    float total = 0.0f;
    for (auto v : x.cpu_matrix().span()) total += v;
    nn::Tensor ref = nn::Tensor::cpu(R, C);
    for (auto& v : ref.cpu_matrix().span()) v = total;

    check_close(*out, ref, "归约指令叠加归约视图 row_sum(col_sum(x)) = 全局和");
}

// ── 测试 7：归约与标量混合（均值平移）x - row_mean ─────────────────────
void test_reduce_with_scalar()
{
    nn::CpuEngine eng;
    const std::size_t R = 4, C = 5;
    nn::Tensor x = make_tensor(R, C, -2.0f, 0.13f);
    const float inv = 1.0f / static_cast<float>(C);

    // x - (row_sum(x) / C)   —— 归约视图 + 标量除法
    auto out = dsl::compute(eng,
        dsl::leaf(x) - (dsl::row_reduce_sum(x) * nn::dsl::ConstLeaf{inv}), R, C);
    CHECK(out.has_value(), "reduce_with_scalar: compute failed");

    nn::Tensor ref = nn::Tensor::cpu(R, C);
    auto xs = x.cpu_matrix().span();
    auto rs = ref.cpu_matrix().span();
    for (std::size_t r = 0; r < R; ++r)
    {
        float sum = 0.0f;
        for (std::size_t c = 0; c < C; ++c) sum += xs[r * C + c];
        const float mean = sum * inv;
        for (std::size_t c = 0; c < C; ++c) rs[r * C + c] = xs[r * C + c] - mean;
    }
    check_close(*out, ref, "归约视图 + 标量 (x - row_mean)");
}

// ── 测试 8：to_expr_spec 折叠结构正确性 ─────────────────────────────────
void test_fold_structure()
{
    nn::Tensor x = nn::Tensor::cpu(4, 5);

    // x / row_reduce_sum(x)：2 输入（Linear + RowReduceSum 视图），1 条 Div 指令
    {
        auto [spec, inputs] = to_expr_spec(dsl::leaf(x) / dsl::row_reduce_sum(x));
        CHECK(spec.instrs.size() == 1, "fold: x/row_sum 应为 1 条指令");
        CHECK(spec.instrs[0].op == static_cast<uint8_t>(nn::ExprOp::Div),
              "fold: x/row_sum 指令应为 Div");
        CHECK(spec.views.size() == 2, "fold: x/row_sum 应有 2 个视图");
        CHECK(spec.views[0].kind == static_cast<uint8_t>(nn::ExprViewKind::Linear),
              "fold: 视图0 应为 Linear");
        CHECK(spec.views[1].kind == static_cast<uint8_t>(nn::ExprViewKind::RowReduceSum),
              "fold: 视图1 应为 RowReduceSum");
        CHECK(inputs.size() == 2, "fold: x/row_sum 应有 2 个输入");
    }

    // row_reduce_sum(exp(x))：1 输入，Exp 指令 + RowSum 归约指令
    {
        auto [spec, inputs] = to_expr_spec(dsl::row_reduce_sum(dsl::exp(dsl::leaf(x))));
        CHECK(spec.instrs.size() == 2, "fold: row_sum(exp(x)) 应为 2 条指令");
        CHECK(static_cast<nn::ExprOp>(spec.instrs[0].op) == nn::ExprOp::Exp,
              "fold: 指令0 应为 Exp");
        CHECK(static_cast<nn::ExprOp>(spec.instrs[1].op) == nn::ExprOp::RowSum,
              "fold: 指令1 应为 RowSum 归约指令");
        CHECK(spec.instrs[1].a.kind == static_cast<uint8_t>(nn::ExprOperandKind::Reg),
              "fold: RowSum 源应为寄存器");
        CHECK(spec.views.size() == 1 && spec.views[0].kind == static_cast<uint8_t>(nn::ExprViewKind::Linear),
              "fold: row_sum(exp(x)) 视图应为 Linear");
    }

    // 校验应通过
    {
        auto [spec, inputs] = to_expr_spec(dsl::leaf(x) / dsl::row_reduce_sum(x));
        auto v = nn::validate_expr_spec(spec, inputs.size());
        CHECK(v.has_value(), "validate: x/row_sum 应通过");
    }
}

// ── 测试 9：校验拒绝非法归约引用 ────────────────────────────────────────
void test_validation()
{
    // 构造：Reg 引用归约 dst（非法：必须用 Reduce 操作数）
    {
        nn::ExprSpec spec;
        spec.num_regs = 2;
        spec.instrs.push_back({static_cast<uint8_t>(nn::ExprOp::RowSum), 0,
                               nn::expr::input(0), {}, {}});   // r0 = RowSum(input0)
        spec.instrs.push_back({static_cast<uint8_t>(nn::ExprOp::Neg), 1,
                               nn::expr::reg(0), {}, {}});     // r1 = -r0  ← 非法：Reg 引用归约 dst
        spec.views.push_back(nn::expr::linear());
        auto v = nn::validate_expr_spec(spec, 1);
        CHECK(!v.has_value(), "validate: Reg 引用归约 dst 应被拒绝");
    }

    // 构造：Reduce 操作数引用非归约 dst（非法）
    {
        nn::ExprSpec spec;
        spec.num_regs = 2;
        spec.instrs.push_back({static_cast<uint8_t>(nn::ExprOp::Neg), 0,
                               nn::expr::input(0), {}, {}});   // r0 = -input0（逐元素）
        spec.instrs.push_back({static_cast<uint8_t>(nn::ExprOp::Mul), 1,
                               nn::expr::reduce(0), nn::expr::reg(0), {}});  // Reduce(0) 但 r0 非归约
        spec.views.push_back(nn::expr::linear());
        auto v = nn::validate_expr_spec(spec, 1);
        CHECK(!v.has_value(), "validate: Reduce 引用非归约 dst 应被拒绝");
    }

    // 构造：Reduce 操作数自引用（非法）
    {
        nn::ExprSpec spec;
        spec.num_regs = 1;
        spec.instrs.push_back({static_cast<uint8_t>(nn::ExprOp::RowSum), 0,
                               nn::expr::reduce(0), {}, {}});  // RowSum 源引用自己
        spec.views.push_back(nn::expr::linear());
        auto v = nn::validate_expr_spec(spec, 1);
        CHECK(!v.has_value(), "validate: Reduce 自引用应被拒绝");
    }
}

// ── 测试 10：规范 key 确定性 ────────────────────────────────────────────
void test_key_determinism()
{
    nn::Tensor x = nn::Tensor::cpu(4, 5);

    auto key_row = nn::expr_spec_key(to_expr_spec(
        dsl::leaf(x) / dsl::row_reduce_sum(x)).first);
    auto key_row2 = nn::expr_spec_key(to_expr_spec(
        dsl::leaf(x) / dsl::row_reduce_sum(x)).first);
    auto key_col = nn::expr_spec_key(to_expr_spec(
        dsl::leaf(x) / dsl::col_reduce_sum(x)).first);

    CHECK(key_row == key_row2, "key: 同结构同 key");
    CHECK(key_row != key_col, "key: 行/列归约 key 不同");

    // 归约指令 vs 归约视图：row_sum(exp(x)) 与 exp(x) 的 key 不同
    auto key_instr = nn::expr_spec_key(to_expr_spec(
        dsl::row_reduce_sum(dsl::exp(dsl::leaf(x)))).first);
    auto key_exp = nn::expr_spec_key(to_expr_spec(
        dsl::exp(dsl::leaf(x))).first);
    CHECK(key_instr != key_exp, "key: 归约指令参与哈希");
}

} // namespace

int main()
{
    test_row_sum_broadcast();
    test_row_max_shift();
    test_softmax_mixed();
    test_reduce_instr_output();
    test_col_max_broadcast();
    test_reduce_of_reduce_view();
    test_reduce_with_scalar();
    test_fold_structure();
    test_validation();
    test_key_determinism();

    if (g_fail == 0)
        std::printf("\n[expr_reduce_test] 全部通过 ✅\n");
    else
        std::printf("\n[expr_reduce_test] %d 项失败 ❌\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
