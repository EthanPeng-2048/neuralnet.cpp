// ───────────────────────────────────────────────────────────────────────────
//  expr_opt_test.cpp — IR 优化 pass（IR-A + IR-B）单元验证（纯 CPU、无 GPU）
//
//  验证内容（对应 docs/11-ir-optimization.md 的验证策略）：
//   1. canonicalize 各 pass 正确性：DCE / 常量折叠 / 代数化简 / CSE / 寄存器分配
//   2. 确定性：同一 spec 多次 canonicalize 结果完全相同（含 key）
//   3. 语义等价：canonical 前后 CPU 求值结果一致（含归约表达式）
//   4. 不变量：views 不变；输出恒为真实寄存器；canonical 通过 validate
//   5. 上限压力：CSE + 寄存器分配后落入 EXPR_MAX_INPUTS/REGS/INSTRS/CONSTS
//   6. 常量折叠保守性：不折叠超越函数（exp/log/sqrt/rsqrt/tanh）
//
//  编译（单 TU 快速验证，与 expr_dsl_test 相同）：
//    clang++ -std=c++26 -stdlib=libc++ -fno-exceptions \
//            -I include/neuralnet.cpp src/expr_opt_test.cpp -o build/expr_opt_test
//  运行：./build/expr_opt_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/compute_cpu_engine.hpp>

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

nn::Tensor make_tensor(std::size_t rows, std::size_t cols, float base = 0.0f,
                       float step = 0.01f)
{
    nn::Tensor t = nn::Tensor::cpu(rows, cols);
    auto sp = t.cpu_matrix().span();
    for (std::size_t i = 0; i < sp.size(); ++i)
        sp[i] = base + static_cast<float>(i) * step;
    return t;
}

// 辅助：直接构造 ExprSpec（控制性测试用）；num_inputs = views.size()
nn::ExprSpec build_spec(std::vector<nn::ExprInstr> instrs,
                        std::vector<nn::ExprView> views,
                        std::vector<Scalar> consts)
{
    nn::ExprSpec s;
    s.instrs = std::move(instrs);
    s.views = std::move(views);
    s.consts = std::move(consts);
    std::uint32_t max_reg = 0;
    for (const auto& in : s.instrs)
        max_reg = std::max(max_reg, static_cast<std::uint32_t>(in.dst) + 1);
    s.num_regs = max_reg;
    return s;
}

// ── 测试 1：常量折叠 + 代数化简 ─────────────────────────────────────────
void test_fold()
{
    // x*0 → 0（中间指令折叠为常量）；Add(Const0, input1) 为输出 → 保留
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::cst(0), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 1, nn::expr::reg(0),
              nn::expr::input(1), {}}},
            {nn::expr::linear(), nn::expr::linear()}, {Scalar{0}});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.instrs.size() == 1, "fold: x*0 → 常量 0 后仅剩输出 Add");
        CHECK(static_cast<nn::ExprOp>(c.instrs[0].op) == nn::ExprOp::Add,
              "fold: 输出指令仍为 Add");
        CHECK(c.instrs[0].a.kind ==
                  static_cast<uint8_t>(nn::ExprOperandKind::Const) &&
              c.consts[c.instrs[0].a.idx] == Scalar{0},
              "fold: Add 的 a 操作数被折叠成常量 0");
        auto v = nn::validate_expr_spec(c, 2);
        CHECK(v.has_value(), "fold: canonical 通过 validate");
    }

    // x + 0 → x（中间指令消失）
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Add), 0, nn::expr::input(0),
              nn::expr::cst(0), {}},
             {static_cast<uint8_t>(nn::ExprOp::Neg), 1, nn::expr::reg(0), {}, {}}},
            {nn::expr::linear()}, {Scalar{0}});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.instrs.size() == 1, "fold: x+0 → x 后仅剩输出 Neg");
        CHECK(static_cast<nn::ExprOp>(c.instrs[0].op) == nn::ExprOp::Neg,
              "fold: 输出指令仍为 Neg");
        CHECK(c.instrs[0].a.kind ==
                  static_cast<uint8_t>(nn::ExprOperandKind::Input),
              "fold: Neg 的操作数直接引用输入（x+0 消除）");
    }

    // neg(neg(x)) → x
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Neg), 0, nn::expr::input(0), {}, {}},
             {static_cast<uint8_t>(nn::ExprOp::Neg), 1, nn::expr::reg(0), {}, {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 2, nn::expr::reg(1),
              nn::expr::input(1), {}}},
            {nn::expr::linear(), nn::expr::linear()}, {});
        auto c = nn::canonicalize_expr_spec(s);
        // neg(neg(x))→x；输出 Add(x, input1)
        CHECK(c.instrs.size() == 1, "fold: neg(neg(x))→x 消除两条 Neg");
        CHECK(c.instrs[0].a.kind ==
                  static_cast<uint8_t>(nn::ExprOperandKind::Input),
              "fold: Add 的 a 直接引用输入（neg(neg) 消除）");
    }

    // max(x, x) → x；纯常量算术 Add(2,3) → 5
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Max), 0, nn::expr::input(0),
              nn::expr::input(0), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 1, nn::expr::cst(0),
              nn::expr::cst(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 2, nn::expr::reg(0),
              nn::expr::reg(1), {}}},
            {nn::expr::linear()}, {Scalar{2}, Scalar{3}});
        auto c = nn::canonicalize_expr_spec(s);
        // max(x,x)→input0（reg0 别名）；Add(2,3)→Const(5)（reg1 别名）；
        // 输出 Mul(input0, Const(5))——但 Mul 是输出，只 remap 不化简
        CHECK(c.instrs.size() == 1, "fold: max(x,x) 与常量折叠后仅剩输出 Mul");
        CHECK(static_cast<nn::ExprOp>(c.instrs[0].op) == nn::ExprOp::Mul,
              "fold: 输出指令为 Mul");
        CHECK(c.instrs[0].a.kind ==
                  static_cast<uint8_t>(nn::ExprOperandKind::Input),
              "fold: Mul 的 a 直接引用输入（max(x,x) 消除）");
        CHECK(c.instrs[0].b.kind ==
                  static_cast<uint8_t>(nn::ExprOperandKind::Const),
              "fold: Mul 的 b 为常量（Add(2,3) 折叠）");
        bool found5 = false;
        for (Scalar v : c.consts) if (v == Scalar{5}) found5 = true;
        CHECK(found5, "fold: 常量池含折叠结果 5");
    }

    // 保守性：不折叠超越函数 exp(2.0)（保留指令）
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Exp), 0, nn::expr::cst(0), {}, {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 1, nn::expr::reg(0),
              nn::expr::input(0), {}}},
            {nn::expr::linear()}, {Scalar{2.0f}});
        auto c = nn::canonicalize_expr_spec(s);
        // exp 不折叠 → 指令保留；Add 输出保留
        CHECK(c.instrs.size() == 2, "fold: 超越函数 exp 不被折叠（保守）");
        CHECK(static_cast<nn::ExprOp>(c.instrs[0].op) == nn::ExprOp::Exp,
              "fold: exp 指令保留");
    }
}

// ── 测试 2：DCE（死代码消除） ────────────────────────────────────────────
void test_dce()
{
    // r0 = Mul 结果未被引用 → 死代码
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 1, nn::expr::input(0),
              nn::expr::input(2), {}}},
            {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()}, {});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.instrs.size() == 1, "dce: 未引用指令被消除");
        CHECK(static_cast<nn::ExprOp>(c.instrs[0].op) == nn::ExprOp::Add,
              "dce: 保留输出 Add");
        auto v = nn::validate_expr_spec(c, 3);
        CHECK(v.has_value(), "dce: canonical 通过 validate");
    }
}

// ── 测试 3：CSE（公共子表达式消除） ─────────────────────────────────────
void test_cse()
{
    // 重复子表达式 Mul(input0, input1) 出现两次 → 合并
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 1, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 2, nn::expr::reg(0),
              nn::expr::reg(1), {}}},
            {nn::expr::linear(), nn::expr::linear()}, {});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.instrs.size() == 2, "cse: 两条相同 Mul 合并为一条");
        // Add 的两个操作数应引用同一寄存器
        CHECK(c.instrs[1].a == c.instrs[1].b,
              "cse: Add 两个操作数引用同一子表达式");
        auto v = nn::validate_expr_spec(c, 2);
        CHECK(v.has_value(), "cse: canonical 通过 validate");
    }

    // CSE 与 DCE 交互：死代码的重复子表达式也应被消除
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 1, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 2, nn::expr::input(2),
              nn::expr::input(3), {}}},
            {nn::expr::linear(), nn::expr::linear(), nn::expr::linear(),
             nn::expr::linear()},
            {});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.instrs.size() == 1, "cse+dce: 死代码 Mul 全部消除");
    }
}

// ── 测试 4：寄存器分配（liveness 复用） ─────────────────────────────────
void test_regalloc()
{
    // 链式依赖，中间寄存器生命周期短 → 可复用
    // r0 = in0*in1; r1 = in2*in3; r2 = r0+r1; r3 = in0*in4; r4 = r2+r3 (out)
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 1, nn::expr::input(2),
              nn::expr::input(3), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 2, nn::expr::reg(0),
              nn::expr::reg(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 3, nn::expr::input(0),
              nn::expr::input(4), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 4, nn::expr::reg(2),
              nn::expr::reg(3), {}}},
            {nn::expr::linear(), nn::expr::linear(), nn::expr::linear(),
             nn::expr::linear(), nn::expr::linear()},
            {});
        const std::uint32_t orig_regs = s.num_regs;
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(c.num_regs < orig_regs, "regalloc: 寄存器复用降低 num_regs");
        CHECK(c.num_regs > 0 && c.num_regs <= nn::EXPR_MAX_REGS,
              "regalloc: num_regs 合法");
        auto v = nn::validate_expr_spec(c, 5);
        CHECK(v.has_value(), "regalloc: canonical 通过 validate");
    }
}

// ── 测试 5：确定性（多次 canonicalize 结果一致） ────────────────────────
void test_determinism()
{
    std::vector<nn::ExprSpec> specs;

    // 1) RoPE forward（真实 DSL 折叠路径）
    {
        const std::size_t DK = 32;
        nn::Tensor q = make_tensor(2 * DK, 4);
        nn::Tensor cos = make_tensor(DK, 4);
        nn::Tensor sin = make_tensor(DK, 4);
        auto expr = leaf(q) * row_mod(cos, DK) + rotate_half(q, DK) * row_mod(sin, DK);
        specs.push_back(to_expr_spec(expr).first);
    }
    // 2) softmax 混合（归约视图 + 归约指令）
    {
        nn::Tensor x = make_tensor(4, 6);
        auto expr = exp(leaf(x) - row_reduce_max(x))
                  / row_reduce_sum(exp(leaf(x) - row_reduce_max(x)));
        specs.push_back(to_expr_spec(expr).first);
    }
    // 3) RMSNorm 风格（列归约 + 广播 + 标量）
    {
        nn::Tensor x = make_tensor(8, 5);
        nn::Tensor inv = make_tensor(1, 5);
        auto expr = leaf(x) * col_reduce_sum(leaf(x) * leaf(x)) + leaf(inv);
        specs.push_back(to_expr_spec(expr).first);
    }
    // 4) 手工构造：常量折叠 + CSE + 复用机会
    {
        specs.push_back(build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::cst(0), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 1, nn::expr::input(0),
              nn::expr::input(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Mul), 2, nn::expr::reg(1),
              nn::expr::cst(1), {}},
             {static_cast<uint8_t>(nn::ExprOp::Add), 3, nn::expr::reg(2),
              nn::expr::input(0), {}}},
            {nn::expr::linear(), nn::expr::linear()}, {Scalar{0}, Scalar{1}}));
    }

    for (std::size_t i = 0; i < specs.size(); ++i)
    {
        const nn::ExprSpec c1 = nn::canonicalize_expr_spec(specs[i]);
        const nn::ExprSpec c2 = nn::canonicalize_expr_spec(specs[i]);
        CHECK(nn::expr_spec_equal(c1, c2), "determinism: 两次 canonicalize 结果一致");
        CHECK(nn::expr_spec_key(c1) == nn::expr_spec_key(c2),
              "determinism: canonical key 稳定");
        // views 不变量：canonical 不改变输入视图
        CHECK(c1.views == specs[i].views, "determinism: views 保持不变");
        // canonical 输出合法
        auto v = nn::validate_expr_spec(c1, c1.views.size());
        CHECK(v.has_value(), "determinism: canonical 通过 validate");
        // 幂等：canonicalize(canonicalize(spec)) == canonicalize(spec)
        const nn::ExprSpec c3 = nn::canonicalize_expr_spec(c1);
        CHECK(nn::expr_spec_equal(c1, c3), "determinism: canonicalize 幂等");
    }
}

// ── 测试 6：语义等价（canonical 前后 CPU 求值一致） ─────────────────────
void test_semantic_equivalence()
{
    nn::CpuEngine eng;

    // 含归约的 softmax 表达式（走 eval_expr 路径）
    {
        const std::size_t R = 3, C = 7;
        nn::Tensor x = make_tensor(R, C, -0.8f, 0.11f);
        auto expr = exp(leaf(x) - row_reduce_max(x))
                  / row_reduce_sum(exp(leaf(x) - row_reduce_max(x)));
        auto [spec, inputs] = to_expr_spec(expr);
        const nn::ExprSpec canon = nn::canonicalize_expr_spec(spec);

        auto r_orig = eng.eval_expr(spec, inputs, R, C);
        auto r_canon = eng.eval_expr(canon, inputs, R, C);
        CHECK(r_orig.has_value() && r_canon.has_value(),
              "semantic: softmax 原始/canonical 求值均成功");
        if (r_orig && r_canon)
        {
            const auto a = r_orig->cpu_matrix().span();
            const auto b = r_canon->cpu_matrix().span();
            bool ok = a.size() == b.size();
            for (std::size_t i = 0; ok && i < a.size(); ++i)
                if (std::fabs(a[i] - b[i]) > 1e-5f) ok = false;
            CHECK(ok, "semantic: softmax 原始/canonical 数值一致");
        }
    }

    // 无归约表达式（eval_cpu 模板路径 vs canonical eval_expr 一致性）
    {
        const std::size_t R = 4, C = 5;
        nn::Tensor a = make_tensor(R, C, 1.0f, 0.1f);
        nn::Tensor b = make_tensor(R, C, 0.5f, 0.05f);
        auto expr = leaf(a) * Scalar{2} + leaf(b) * Scalar{3};
        auto [spec, inputs] = to_expr_spec(expr);
        const nn::ExprSpec canon = nn::canonicalize_expr_spec(spec);

        auto r_canon = eng.eval_expr(canon, inputs, R, C);
        nn::Tensor r_ref = eval_cpu(expr, R, C);
        CHECK(r_canon.has_value(), "semantic: 无归约 canonical 求值成功");
        if (r_canon)
        {
            const auto a_ = r_canon->cpu_matrix().span();
            const auto b_ = r_ref.cpu_matrix().span();
            bool ok = a_.size() == b_.size();
            for (std::size_t i = 0; ok && i < a_.size(); ++i)
                if (std::fabs(a_[i] - b_[i]) > 1e-5f) ok = false;
            CHECK(ok, "semantic: canonical 求值与模板求值一致");
        }
    }
}

// ── 测试 7：上限压力测试（超输入/超寄存器） ────────────────────────────
void test_pressure()
{
    // 8 输入、重复子表达式多、寄存器压力大的表达式
    // 构造: out = Σ_i in_i * c_i（每个乘法唯一）+ 若干重复子表达式
    {
        nn::ExprSpec s;
        for (std::uint8_t i = 0; i < 8; ++i)
            s.views.push_back(nn::expr::linear());
        std::vector<nn::ExprInstr> instrs;
        // 每输入乘一个常量（8 条），再加两轮重复的乘加
        for (std::uint8_t i = 0; i < 8; ++i)
        {
            nn::ExprInstr in;
            in.op = static_cast<uint8_t>(nn::ExprOp::Mul);
            in.dst = i;
            in.a = nn::expr::input(i);
            in.b = nn::expr::cst(static_cast<uint8_t>(i));
            instrs.push_back(in);
        }
        // 两轮完全相同的 (r0+r1) → CSE 合并
        for (int k = 0; k < 2; ++k)
        {
            nn::ExprInstr in;
            in.op = static_cast<uint8_t>(nn::ExprOp::Add);
            in.dst = static_cast<uint8_t>(8 + k * 2);
            in.a = nn::expr::reg(0);
            in.b = nn::expr::reg(1);
            instrs.push_back(in);
        }
        // 汇总求和（链式）
        nn::ExprInstr acc;
        acc.op = static_cast<uint8_t>(nn::ExprOp::Add);
        acc.dst = static_cast<uint8_t>(10);
        acc.a = nn::expr::reg(8);
        acc.b = nn::expr::reg(2);
        instrs.push_back(acc);
        for (std::uint8_t i = 3; i < 8; ++i)
        {
            nn::ExprInstr in;
            in.op = static_cast<uint8_t>(nn::ExprOp::Add);
            in.dst = static_cast<uint8_t>(instrs.size());
            in.a = nn::expr::reg(static_cast<uint8_t>(instrs.size() - 1));
            in.b = nn::expr::reg(i);
            instrs.push_back(in);
        }
        // 输出 = 最后一次加法（使用已存在的重复 Add 结果避免死代码）
        // 调整：输出指令 = acc（最后一个 Add 的结果）
        s.instrs = std::move(instrs);
        std::uint32_t max_reg = 0;
        for (const auto& in : s.instrs)
            max_reg = std::max(max_reg, static_cast<std::uint32_t>(in.dst) + 1);
        s.num_regs = max_reg;
        for (std::uint8_t i = 0; i < 8; ++i)
            s.consts.push_back(static_cast<Scalar>(i + 1));

        auto v0 = nn::validate_expr_spec(s, 8);
        CHECK(v0.has_value(), "pressure: 原始 spec 通过 validate");
        if (!v0) return;

        const nn::ExprSpec c = nn::canonicalize_expr_spec(s);
        auto v = nn::validate_expr_spec(c, 8);
        CHECK(v.has_value(), "pressure: canonical 通过 validate");
        CHECK(c.instrs.size() <= nn::EXPR_MAX_INSTRS,
              "pressure: 指令数 ≤ EXPR_MAX_INSTRS");
        CHECK(c.num_regs <= nn::EXPR_MAX_REGS,
              "pressure: 寄存器数 ≤ EXPR_MAX_REGS");
        CHECK(c.views.size() <= nn::EXPR_MAX_INPUTS,
              "pressure: 输入数 ≤ EXPR_MAX_INPUTS");
        CHECK(c.consts.size() <= nn::EXPR_MAX_CONSTS,
              "pressure: 常量数 ≤ EXPR_MAX_CONSTS");
    }
}

// ── 测试 8：输出保护（输出恒为真实寄存器） ──────────────────────────────
void test_output_invariant()
{
    // 输出 = x*0：即使可化简为常量 0，输出指令必须保留（glsl_gen 输出 r<last>）
    {
        auto s = build_spec(
            {{static_cast<uint8_t>(nn::ExprOp::Mul), 0, nn::expr::input(0),
              nn::expr::cst(0), {}}},
            {nn::expr::linear()}, {Scalar{0}});
        auto c = nn::canonicalize_expr_spec(s);
        CHECK(!c.instrs.empty(), "output: 输出指令保留（不为空）");
        CHECK(static_cast<nn::ExprOp>(c.instrs.back().op) == nn::ExprOp::Mul,
              "output: 输出指令不被折叠");
        auto v = nn::validate_expr_spec(c, 1);
        CHECK(v.has_value(), "output: canonical 通过 validate");
    }
}

// ── 测试 9：归约语义保持（Reduce 引用不因重编号破坏） ──────────────────
void test_reduce_preserved()
{
    // softmax 表达式：归约指令 dst 与逐元素 dst 区段分离，Reduce 引用正确
    nn::Tensor x = make_tensor(3, 7, -0.8f, 0.11f);
    auto expr = exp(leaf(x) - row_reduce_max(x))
              / row_reduce_sum(exp(leaf(x) - row_reduce_max(x)));
    auto [spec, inputs] = to_expr_spec(expr);
    const nn::ExprSpec c = nn::canonicalize_expr_spec(spec);
    auto v = nn::validate_expr_spec(c, inputs.size());
    CHECK(v.has_value(), "reduce: 归约语义在 canonical 后保持合法");

    // 归约指令数量不变（fold/CSE/regalloc 不得合并/删除归约指令本身）
    std::size_t n_red_orig = 0, n_red_canon = 0;
    for (const auto& in : spec.instrs)
        if (nn::expr_op_is_reduce(static_cast<nn::ExprOp>(in.op))) ++n_red_orig;
    for (const auto& in : c.instrs)
        if (nn::expr_op_is_reduce(static_cast<nn::ExprOp>(in.op))) ++n_red_canon;
    CHECK(n_red_orig == n_red_canon, "reduce: 归约指令数量不变");

    // 每个归约指令的 dst 与所有逐元素指令 dst 互斥（区段分离）
    bool disjoint = true;
    for (const auto& r : c.instrs)
        if (nn::expr_op_is_reduce(static_cast<nn::ExprOp>(r.op)))
            for (const auto& e : c.instrs)
                if (!nn::expr_op_is_reduce(static_cast<nn::ExprOp>(e.op)) &&
                    r.dst == e.dst)
                    disjoint = false;
    CHECK(disjoint, "reduce: 归约 dst 与逐元素 dst 互斥（区段分离）");
}

} // namespace

int main()
{
    std::printf("========================================\n");
    std::printf("  IR 优化 pass 单元测试（IR-A + IR-B）\n");
    std::printf("========================================\n");

    test_fold();
    std::printf("[run] test_fold done\n"); std::fflush(stdout);
    test_dce();
    std::printf("[run] test_dce done\n"); std::fflush(stdout);
    test_cse();
    std::printf("[run] test_cse done\n"); std::fflush(stdout);
    test_regalloc();
    std::printf("[run] test_regalloc done\n"); std::fflush(stdout);
    test_determinism();
    std::printf("[run] test_determinism done\n"); std::fflush(stdout);
    test_semantic_equivalence();
    std::printf("[run] test_semantic_equivalence done\n"); std::fflush(stdout);
    test_pressure();
    std::printf("[run] test_pressure done\n"); std::fflush(stdout);
    test_output_invariant();
    std::printf("[run] test_output_invariant done\n"); std::fflush(stdout);
    test_reduce_preserved();
    std::printf("[run] test_reduce_preserved done\n"); std::fflush(stdout);

    if (g_fail == 0)
        std::printf("\n[expr_opt_test] 全部通过 ✅\n");
    else
        std::printf("\n[expr_opt_test] %d 项失败 ❌\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
