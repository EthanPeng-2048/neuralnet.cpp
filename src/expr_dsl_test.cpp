// ───────────────────────────────────────────────────────────────────────────
//  expr_dsl_test.cpp — 统一表达式 DSL 阶段 1 轻量验证（纯 CPU、无 GPU）
//
//  验证内容：
//   1. CPU 编译期模板求值正确性（eval_cpu 与手写参考逐元素对照）
//   2. to_expr_spec 折叠结构正确（指令/寄存器/视图/输入数量与算子序列）
//   3. forward / backward spec 可区分（闭合世界 AOT 匹配的前提）
//   4. 标量广播、比较 + select（relu）正确性
//
//  编译（本机性能差，用单 TU 快速验证）：
//    clang++ -std=c++26 -stdlib=libc++ -fno-exceptions \
//            -I include/neuralnet.cpp src/expr_dsl_test.cpp -o build/expr_dsl_test
//  运行：./build/expr_dsl_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include <neuralnet.cpp/tensor.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/fused_exprs.hpp>

// 测试写在全局作用域（非 namespace nn），避免与旧代数运算符的 ADL 歧义。
using namespace nn::dsl;
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

// 手写 RoPE forward 参考（与 RotateHalfRef/RowModRef 语义一致）
float rope_ref(const nn::Tensor& q, const nn::Tensor& cos, const nn::Tensor& sin,
               std::size_t r, std::size_t c, std::size_t dk, bool backward)
{
    const std::size_t cols = q.cols();
    const auto qs = q.cpu_matrix().span();
    const auto cs = cos.cpu_matrix().span();
    const auto ss = sin.cpu_matrix().span();
    const std::size_t rl = r % dk;
    const std::size_t rr = (r / dk) * dk + ((rl < dk / 2) ? rl + dk / 2 : rl - dk / 2);
    const float qc = qs[r * cols + c] * cs[(r % dk) * cols + c];
    const float rot = (rl < dk / 2) ? -qs[rr * cols + c] : qs[rr * cols + c];
    const float qs2 = rot * ss[(r % dk) * cols + c];
    return backward ? (qc - qs2) : (qc + qs2);
}

void test_rope()
{
    constexpr std::size_t DK = 32, ROWS = 64, COLS = 4;
    nn::Tensor q = make_tensor(ROWS, COLS, 0.5f, 0.001f);
    nn::Tensor cos = make_tensor(DK, COLS, 0.9f, 0.0002f);
    nn::Tensor sin = make_tensor(DK, COLS, 0.1f, 0.0001f);

    // forward
    {
        auto expr = make_rope<false>(q, cos, sin, DK);
        nn::Tensor out = eval_cpu(expr, ROWS, COLS);
        const auto os = out.cpu_matrix().span();
        bool ok = true;
        for (std::size_t r = 0; r < ROWS && ok; ++r)
            for (std::size_t c = 0; c < COLS; ++c)
            {
                const float want = rope_ref(q, cos, sin, r, c, DK, false);
                const float got = os[r * COLS + c];
                if (std::fabs(want - got) > 1e-5f) { ok = false; break; }
            }
        CHECK(ok, "RoPE forward CPU 求值与参考一致");

        // spec 结构
        auto [spec, inputs] = to_expr_spec(expr);
        CHECK(inputs.size() == 4, "RoPE forward: 4 个输入");
        CHECK(spec.views.size() == 4, "RoPE forward: 4 个视图");
        CHECK(spec.instrs.size() == 3, "RoPE forward: 3 条指令");
        CHECK(spec.num_regs == 3, "RoPE forward: 3 个寄存器");
        CHECK(spec.instrs[0].op == static_cast<uint8_t>(nn::ExprOp::Mul), "instr0=Mul(q*cos)");
        CHECK(spec.instrs[1].op == static_cast<uint8_t>(nn::ExprOp::Mul), "instr1=Mul(rot*sin)");
        CHECK(spec.instrs[2].op == static_cast<uint8_t>(nn::ExprOp::Add), "instr2=Add(r0+r1)");
        CHECK(spec.instrs[2].dst == 2, "输出寄存器 = 2");
        CHECK(nn::expr_spec_equal(spec, nn::fused::make_rope(DK, false)),
              "dsl 折叠 == fused::make_rope（AOT 匹配防漂移）");
        CHECK(nn::expr_spec_equal(spec, to_expr_spec(make_rope<false>(q, cos, sin, DK)).first),
              "同一表达式两次折叠 spec 相同（AOT 匹配前提）");
    }

    // backward：spec 应与 forward 不同（末条为 Sub）
    {
        auto expr = make_rope<true>(q, cos, sin, DK);
        nn::Tensor out = eval_cpu(expr, ROWS, COLS);
        const auto os = out.cpu_matrix().span();
        bool ok = true;
        for (std::size_t r = 0; r < ROWS && ok; ++r)
            for (std::size_t c = 0; c < COLS; ++c)
            {
                const float want = rope_ref(q, cos, sin, r, c, DK, true);
                if (std::fabs(want - os[r * COLS + c]) > 1e-5f) { ok = false; break; }
            }
        CHECK(ok, "RoPE backward CPU 求值与参考一致");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(spec.instrs.back().op == static_cast<uint8_t>(nn::ExprOp::Sub), "backward 末条=Sub");
        CHECK(!nn::expr_spec_equal(spec, to_expr_spec(make_rope<false>(q, cos, sin, DK)).first),
              "forward/backward spec 可区分");
    }
}

void test_elementwise()
{
    constexpr std::size_t R = 8, C = 5;
    nn::Tensor a = make_tensor(R, C, 1.0f, 0.1f);
    nn::Tensor b = make_tensor(R, C, 0.5f, 0.05f);

    // a * 2 + b（标量广播）
    {
        auto expr = leaf(a) * Scalar{2} + leaf(b);
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto as = a.cpu_matrix().span();
        const auto bs = b.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
            if (std::fabs((as[i] * 2.0f + bs[i]) - os[i]) > 1e-5f) { ok = false; break; }
        CHECK(ok, "a*2+b 标量广播正确");
        auto [spec, inputs] = to_expr_spec(expr);
        CHECK(spec.instrs.size() == 2 && spec.num_regs == 2, "a*2+b: 2 指令 2 寄存器");
        CHECK(inputs.size() == 2 && spec.consts.size() == 1, "a*2+b: 2 输入 1 常量");
    }

    // relu(a) = max(a, 0)（单条 Max 指令；编译期融合）
    {
        auto expr = relu(leaf(a));
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto as = a.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
            if (std::fabs((as[i] > 0.0f ? as[i] : 0.0f) - os[i]) > 1e-5f) { ok = false; break; }
        CHECK(ok, "relu(a) 正确");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(spec.instrs.size() == 1 && spec.num_regs == 1, "relu: 单条指令");
        CHECK(spec.instrs[0].op == static_cast<uint8_t>(nn::ExprOp::Max), "relu: Max 指令");
    }

    // select(a>0, a, 0)（比较 + Select 两条指令）
    {
        auto expr = select(leaf(a) > Scalar{0}, leaf(a), Scalar{0});
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto as = a.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
            if (std::fabs((as[i] > 0.0f ? as[i] : 0.0f) - os[i]) > 1e-5f) { ok = false; break; }
        CHECK(ok, "select(a>0,a,0) 正确");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(spec.instrs.size() == 2, "select: 比较+Select 共 2 指令");
        CHECK(spec.instrs[0].op == static_cast<uint8_t>(nn::ExprOp::Gt), "select: 先 Gt 比较");
        CHECK(spec.instrs[1].op == static_cast<uint8_t>(nn::ExprOp::Select), "select: 后 Select");
    }

    // exp + tanh
    {
        auto expr = tanh(leaf(a)) + exp(leaf(b));
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto as = a.cpu_matrix().span();
        const auto bs = b.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
            if (std::fabs((std::tanh(as[i]) + std::exp(bs[i])) - os[i]) > 1e-4f) { ok = false; break; }
        CHECK(ok, "tanh(a)+exp(b) 正确");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(spec.instrs.size() == 3, "tanh+exp+: 3 指令");
    }
}

void test_swiglu()
{
    constexpr std::size_t R = 8, C = 5;
    nn::Tensor g = make_tensor(R, C, 0.3f, 0.01f);
    nn::Tensor s = make_tensor(R, C, 0.7f, 0.008f);
    nn::Tensor gate = make_tensor(R, C, 0.5f, 0.006f);
    nn::Tensor up = make_tensor(R, C, 1.2f, 0.004f);

    // grad_gate = g*up*s*(1 + gate*(1-s))
    {
        auto expr = make_swiglu_grad_gate(g, s, gate, up);
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto gs = g.cpu_matrix().span();
        const auto ss = s.cpu_matrix().span();
        const auto gts = gate.cpu_matrix().span();
        const auto us = up.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
        {
            const float want = gs[i] * us[i] * ss[i] * (1.0f + gts[i] * (1.0f - ss[i]));
            if (std::fabs(want - os[i]) > 1e-4f) { ok = false; break; }
        }
        CHECK(ok, "swiglu grad_gate CPU 正确");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(nn::expr_spec_equal(spec, nn::fused::make_swiglu_grad_gate()),
              "dsl swiglu grad_gate == fused（AOT 防漂移）");
    }
    // grad_up = g*gate*s
    {
        auto expr = make_swiglu_grad_up(g, s, gate);
        nn::Tensor out = eval_cpu(expr, R, C);
        const auto os = out.cpu_matrix().span();
        const auto gs = g.cpu_matrix().span();
        const auto ss = s.cpu_matrix().span();
        const auto gts = gate.cpu_matrix().span();
        bool ok = true;
        for (std::size_t i = 0; i < R * C; ++i)
            if (std::fabs((gs[i] * gts[i] * ss[i]) - os[i]) > 1e-4f) { ok = false; break; }
        CHECK(ok, "swiglu grad_up CPU 正确");
        auto [spec, _] = to_expr_spec(expr);
        CHECK(nn::expr_spec_equal(spec, nn::fused::make_swiglu_grad_up()),
              "dsl swiglu grad_up == fused（AOT 防漂移）");
    }
}

} // namespace

int main()
{
    test_rope();
    test_elementwise();
    test_swiglu();

    if (g_fail == 0)
        std::printf("ALL PASS\n");
    else
        std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
