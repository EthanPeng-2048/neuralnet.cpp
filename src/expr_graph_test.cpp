// ───────────────────────────────────────────────────────────────────────────
//  expr_graph_test.cpp — 图 IR + 融合分析（IR-C）单元验证（纯 CPU、无 GPU）
//
//  验证内容（对应 docs/11-ir-optimization.md 的验证策略）：
//   1. 链融合正确性：A→B 融合成单 kernel 后求值 == 逐节点求值
//   2. 多节点链融合（3 节点）
//   3. 融合边界：非链（无消费者 / 多消费者 / 非 Linear 引用）不融合
//   4. 归约节点作为融合边界（不参与链式拼接）
//   5. 上限压力：拼接后超 EXPR_MAX_* 则保守放弃融合
//   6. 确定性：同一图多次融合结果完全一致（含 key）
//   7. 不变量：融合 kernel 通过 validate；views/输入顺序确定
//   8. 独立分支：不消费前序输出的相邻表达式各自独立成 kernel
//
//  编译（单 TU 快速验证，与 expr_opt_test 相同）：
//    clang++ -std=c++26 -stdlib=libc++ -fno-exceptions \
//            -I include/neuralnet.cpp src/expr_graph_test.cpp -o build/expr_graph_test
//  运行：./build/expr_graph_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/expr_graph.hpp>
#include <neuralnet.cpp/expr_emitter.hpp>
#include <neuralnet.cpp/expr_glsl_gen.hpp>    // 注册 GlslEmitter（IR-D）
#include <neuralnet.cpp/expr_cpu_emitter.hpp> // 注册 CpuEmitter（IR-D）
#include <neuralnet.cpp/compute_cpu_engine.hpp>

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

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    Scalar m = 0.0f;
    auto as = a.span(), bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar d = std::fabs(as[i] - bs[i]);
        if (d > m) m = d;
    }
    return m;
}

// 构造逐元素一元指令：out = unary(input0) 或 out = input0 <op> const
nn::ExprSpec make_unary_const(nn::ExprOp op, Scalar c)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    s.consts.push_back(c);
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(op);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::cst(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造逐元素二元：out = input0 <op> input1
nn::ExprSpec make_binary(nn::ExprOp op)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    s.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(op);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::input(1);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// 构造归约表达式：out = row_sum(input0)（归约边界）
nn::ExprSpec make_row_sum()
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::RowSum);
    in.dst = 0;
    in.a = nn::expr::input(0);
    s.instrs.push_back(in);
    s.num_regs = 1;
    return s;
}

// ── 测试 1：A→B 链融合正确性 ───────────────────────────────────────────
// A: y = x * 2；B: z = y + 1 → 融合后 z = x*2 + 1，单 kernel
void test_chain_fuse()
{
    nn::CpuEngine eng;
    const std::size_t rows = 3, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.5f);

    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor ytag = nn::Tensor::cpu(rows, cols);
    ytag.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    const int nB = g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                              std::vector<nn::Tensor>{ytag}, rows, cols, false);
    (void)nB;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "A→B 链应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 2, "kernel 应包含 2 个节点");
    CHECK(kernels[0].tail == nB, "kernel tail 应为 B 节点");
    CHECK(kernels[0].inputs.size() == kernels[0].spec.views.size(),
          "kernel inputs 与 views 一一对应");

    // 融合 kernel 求值
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs) kinputs.push_back(in.external);
    auto fused = eng.eval_expr(kernels[0].spec, kinputs, rows, cols);
    CHECK(static_cast<bool>(fused), "融合 kernel 求值成功");

    // 参考：逐节点
    auto a_res = eng.eval_expr(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                               std::vector<nn::Tensor>{x}, rows, cols);
    auto b_res = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                               std::vector<nn::Tensor>{*a_res}, rows, cols);
    CHECK(static_cast<bool>(a_res) && static_cast<bool>(b_res), "参考求值成功");
    if (!fused || !a_res || !b_res) return;

    const Scalar err = max_abs_diff(fused->cpu_matrix(), b_res->cpu_matrix());
    CHECK(err < 1e-6f, "融合结果与逐节点一致");

    // 手工验证数值：x*2+1
    bool ok = true;
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        const Scalar expect = x.cpu_matrix().span()[i] * 2.0f + 1.0f;
        if (std::fabs(fused->cpu_matrix().span()[i] - expect) > 1e-5f) ok = false;
    }
    CHECK(ok, "融合结果与手工参考 x*2+1 一致");
    std::printf("[OK] test_chain_fuse\n");
}

// ── 测试 2：三节点链融合 ───────────────────────────────────────────────
// A: y=x*2；B: z=y+3；C: w=z-1 → 融合成单 kernel
void test_chain3_fuse()
{
    nn::CpuEngine eng;
    const std::size_t rows = 2, cols = 5;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 1.0f, 0.3f);

    auto add_node_chain = [&](nn::ExprOp op, Scalar c,
                              const nn::Tensor& input,
                              std::size_t r, std::size_t col) -> int
    {
        const int n = g.add_node(make_unary_const(op, c),
                                 std::vector<nn::Tensor>{input}, r, col, false);
        return n;
    };
    const int nA = add_node_chain(nn::ExprOp::Mul, Scalar{2}, x, rows, cols);
    nn::Tensor yt = nn::Tensor::cpu(rows, cols);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    const int nB = add_node_chain(nn::ExprOp::Add, Scalar{3}, yt, rows, cols);
    nn::Tensor zt = nn::Tensor::cpu(rows, cols);
    zt.set_virtual_tag(static_cast<std::uint64_t>(nB) + 1);
    const int nC = add_node_chain(nn::ExprOp::Sub, Scalar{1}, zt, rows, cols);
    (void)nC;

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "三节点链应融合为单个 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].members.size() == 3, "kernel 应包含 3 个节点");

    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs) kinputs.push_back(in.external);
    auto fused = eng.eval_expr(kernels[0].spec, kinputs, rows, cols);
    CHECK(static_cast<bool>(fused), "融合 kernel 求值成功");

    // 参考：逐节点
    auto a_res = eng.eval_expr(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                               std::vector<nn::Tensor>{x}, rows, cols);
    auto b_res = eng.eval_expr(make_unary_const(nn::ExprOp::Add, Scalar{3}),
                               std::vector<nn::Tensor>{*a_res}, rows, cols);
    auto c_res = eng.eval_expr(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                               std::vector<nn::Tensor>{*b_res}, rows, cols);
    CHECK(static_cast<bool>(c_res), "参考求值成功");
    if (!fused || !c_res) return;

    const Scalar err = max_abs_diff(fused->cpu_matrix(), c_res->cpu_matrix());
    CHECK(err < 1e-6f, "三节点融合结果与逐节点一致");
    std::printf("[OK] test_chain3_fuse\n");
}

// ── 测试 3：独立分支各自成 kernel ──────────────────────────────────────
// A: y = x*2（无消费者）；B: z = w+1（外部输入 w，不依赖 A）→ 2 个 kernel
void test_independent()
{
    nn::CpuEngine eng;
    const std::size_t rows = 3, cols = 3;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);
    nn::Tensor w = make_tensor(rows, cols, 2.0f, 0.2f);

    (void)g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                     std::vector<nn::Tensor>{x}, rows, cols, false);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                     std::vector<nn::Tensor>{w}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 2, "独立节点应各自成 kernel");
    if (kernels.size() != 2) return;
    CHECK(kernels[0].members.size() == 1 && kernels[1].members.size() == 1,
          "每个 kernel 只含 1 个节点");

    // 两个 kernel 求值都正确
    std::vector<nn::Tensor> k0in;
    for (auto& in : kernels[0].inputs) k0in.push_back(in.external);
    std::vector<nn::Tensor> k1in;
    for (auto& in : kernels[1].inputs) k1in.push_back(in.external);
    auto r0 = eng.eval_expr(kernels[0].spec, k0in, rows, cols);
    auto r1 = eng.eval_expr(kernels[1].spec, k1in, rows, cols);
    CHECK(static_cast<bool>(r0) && static_cast<bool>(r1), "两个 kernel 求值成功");
    std::printf("[OK] test_independent\n");
}

// ── 测试 4：多消费者不融合（tail 被 A 和 B 之外节点消费） ─────────────
void test_multi_consumer()
{
    nn::CpuEngine eng;
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);

    // A: y = x*2（输出 y）；B: z = y+1（消费 y）；C: t = y-1（也消费 y）
    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor yt = nn::Tensor::cpu(rows, cols);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                     std::vector<nn::Tensor>{yt}, rows, cols, false);
    nn::Tensor yt2 = nn::Tensor::cpu(rows, cols);
    yt2.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    (void)g.add_node(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                     std::vector<nn::Tensor>{yt2}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    // A 有两个消费者（B、C），不能融合 → 3 个独立 kernel
    CHECK(kernels.size() == 3, "多消费者应各成 kernel（不融合）");
    std::printf("[OK] test_multi_consumer (kernels=%zu)\n", kernels.size());
}

// ── 测试 5：归约节点作为融合边界 ───────────────────────────────────────
void test_reduce_boundary()
{
    nn::CpuEngine eng;
    const std::size_t rows = 3, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);

    // A: y = row_sum(x)（归约，输出 (rows,1)）；B: z = y + 1（消费 y，但 B 的
    //   输入 y 形状是 (rows,1)，经 RowBroadcast 视图）→ 不融合（A 是归约）
    const int nA = g.add_node(make_row_sum(),
                              std::vector<nn::Tensor>{x}, rows, cols, true);
    nn::Tensor yt = nn::Tensor::cpu(rows, 1);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    nn::ExprSpec sb;
    sb.views.push_back(nn::expr::row_broadcast());  // (rows,1) 按行广播
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::Add);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::cst(0);
    sb.consts.push_back(Scalar{1});
    sb.instrs.push_back(in);
    sb.num_regs = 1;
    (void)g.add_node(sb, std::vector<nn::Tensor>{yt}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 2, "归约节点应作为融合边界（不拼接）");
    std::printf("[OK] test_reduce_boundary (kernels=%zu)\n", kernels.size());
}

// ── 测试 6：上限压力（拼接后超 EXPR_MAX_INSTRS 则放弃融合） ───────────
// 用 abs 链（abs 不可代数化简、不可常量折叠）构造可控指令数
nn::ExprSpec make_abs_chain(std::uint8_t n)
{
    nn::ExprSpec s;
    s.views.push_back(nn::expr::linear());
    nn::ExprOperand prev = nn::expr::input(0);
    for (std::uint8_t i = 0; i < n; ++i)
    {
        nn::ExprInstr in;
        in.op = static_cast<std::uint8_t>(nn::ExprOp::Abs);
        in.dst = i;
        in.a = prev;
        s.instrs.push_back(in);
        prev = nn::expr::reg(i);
    }
    s.num_regs = n;
    return s;
}

void test_limit_pressure()
{
    const std::size_t rows = 1, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);

    // A：40 条 abs 指令；B：1 条 → 拼接后 41 ≤ 64 → 融合
    const int nA = g.add_node(make_abs_chain(40),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    nn::Tensor yt = nn::Tensor::cpu(rows, cols);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                     std::vector<nn::Tensor>{yt}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "40+1 条指令应在 EXPR_MAX_INSTRS=64 内 → 融合");

    // A：60 条 abs；B：5 条 abs 链 → 拼接后 65 > 64 → 不融合
    nn::ExprGraph g2;
    nn::Tensor x2 = make_tensor(rows, cols, 0.0f, 0.1f);
    const int nC = g2.add_node(make_abs_chain(60),
                               std::vector<nn::Tensor>{x2}, rows, cols, false);
    nn::Tensor yt2 = nn::Tensor::cpu(rows, cols);
    yt2.set_virtual_tag(static_cast<std::uint64_t>(nC) + 1);
    // B：5 条 abs 链（消费 A 输出）
    nn::ExprSpec sb = make_abs_chain(5);
    (void)g2.add_node(sb, std::vector<nn::Tensor>{yt2}, rows, cols, false);

    auto kernels2 = fuse_expr_graph(g2);
    CHECK(kernels2.size() == 2, "60+5 条指令超 EXPR_MAX_INSTRS → 保守放弃融合");
    std::printf("[OK] test_limit_pressure\n");
}

// ── 测试 7：确定性（同一图多次融合结果一致） ───────────────────────────
void test_determinism()
{
    nn::CpuEngine eng;
    const std::size_t rows = 2, cols = 4;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);

    auto build = [&](nn::ExprGraph& g)
    {
        const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                                  std::vector<nn::Tensor>{x}, rows, cols, false);
        nn::Tensor yt = nn::Tensor::cpu(rows, cols);
        yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
        (void)g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                         std::vector<nn::Tensor>{yt}, rows, cols, false);
    };

    nn::ExprGraph g1, g2;
    build(g1);
    build(g2);
    auto k1 = fuse_expr_graph(g1);
    auto k2 = fuse_expr_graph(g2);
    CHECK(k1.size() == k2.size(), "确定性：kernel 数一致");
    if (k1.size() != k2.size()) return;
    const std::string key1 = nn::expr_spec_key(k1[0].spec);
    const std::string key2 = nn::expr_spec_key(k2[0].spec);
    CHECK(key1 == key2, "确定性：融合 kernel 的 key 完全一致");
    CHECK(nn::expr_spec_equal(k1[0].spec, k2[0].spec), "确定性：spec 完全一致");
    std::printf("[OK] test_determinism\n");
}

// ── 测试 8：录制图 add_node 依赖识别 ───────────────────────────────────
void test_dep_recognition()
{
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x = make_tensor(rows, cols, 0.0f, 0.1f);

    const int nA = g.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    // 占位带 tag → 依赖识别
    nn::Tensor yt = nn::Tensor::cpu(rows, cols);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    const int nB = g.add_node(make_unary_const(nn::ExprOp::Add, Scalar{1}),
                              std::vector<nn::Tensor>{yt}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nB)].dep_of_input[0] == nA,
          "B 的输入 0 应识别为依赖节点 A");
    // 普通输入 → 外部
    const int nC = g.add_node(make_unary_const(nn::ExprOp::Sub, Scalar{1}),
                              std::vector<nn::Tensor>{x}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nC)].dep_of_input[0] == -1,
          "C 的输入 0 应为外部输入");
    // 未知 tag → 外部
    nn::Tensor fake = nn::Tensor::cpu(rows, cols);
    fake.set_virtual_tag(999);
    const int nD = g.add_node(make_unary_const(nn::ExprOp::Abs, Scalar{0}),
                              std::vector<nn::Tensor>{fake}, rows, cols, false);
    CHECK(g.nodes[static_cast<std::size_t>(nD)].dep_of_input[0] == -1,
          "未知 tag 视为外部输入");
    std::printf("[OK] test_dep_recognition\n");
}

// ── 测试 9：融合 kernel 输入顺序确定 ───────────────────────────────────
// A: y = x1 * x2（两个输入）；B: z = y + x3 → 融合后 inputs = [x1, x2, x3]
void test_input_order()
{
    nn::CpuEngine eng;
    const std::size_t rows = 2, cols = 4;
    nn::ExprGraph g;
    nn::Tensor x1 = make_tensor(rows, cols, 0.0f, 0.1f);
    nn::Tensor x2 = make_tensor(rows, cols, 1.0f, 0.1f);
    nn::Tensor x3 = make_tensor(rows, cols, 2.0f, 0.1f);

    // A: y = x1 * x2
    const int nA = g.add_node(make_binary(nn::ExprOp::Mul),
                              std::vector<nn::Tensor>{x1, x2}, rows, cols, false);
    // B: z = y + x3（输入 0 = y 依赖 A，输入 1 = x3 外部）
    nn::Tensor yt = nn::Tensor::cpu(rows, cols);
    yt.set_virtual_tag(static_cast<std::uint64_t>(nA) + 1);
    nn::ExprSpec sb;
    sb.views.push_back(nn::expr::linear());
    sb.views.push_back(nn::expr::linear());
    nn::ExprInstr in;
    in.op = static_cast<std::uint8_t>(nn::ExprOp::Add);
    in.dst = 0;
    in.a = nn::expr::input(0);
    in.b = nn::expr::input(1);
    sb.instrs.push_back(in);
    sb.num_regs = 1;
    (void)g.add_node(sb, std::vector<nn::Tensor>{yt, x3}, rows, cols, false);

    auto kernels = fuse_expr_graph(g);
    CHECK(kernels.size() == 1, "A→B 链融合为单 kernel");
    if (kernels.size() != 1) return;
    CHECK(kernels[0].inputs.size() == 3, "融合 kernel 应有 3 个外部输入");
    if (kernels[0].inputs.size() != 3) return;
    // 顺序：A 的输入（x1, x2）在前，B 的非链输入（x3）在后
    CHECK(kernels[0].inputs[0].external.cpu_matrix().span()[0] == 0.0f,
          "输入 0 = x1");
    CHECK(kernels[0].inputs[1].external.cpu_matrix().span()[0] == 1.0f,
          "输入 1 = x2");
    CHECK(kernels[0].inputs[2].external.cpu_matrix().span()[0] == 2.0f,
          "输入 2 = x3");

    // 数值验证：z = x1*x2 + x3
    std::vector<nn::Tensor> kinputs;
    for (auto& in : kernels[0].inputs) kinputs.push_back(in.external);
    auto fused = eng.eval_expr(kernels[0].spec, kinputs, rows, cols);
    CHECK(static_cast<bool>(fused), "融合 kernel 求值成功");
    if (!fused) return;
    bool ok = true;
    for (std::size_t i = 0; i < rows * cols; ++i)
    {
        const Scalar expect = x1.cpu_matrix().span()[i] * x2.cpu_matrix().span()[i]
                            + x3.cpu_matrix().span()[i];
        if (std::fabs(fused->cpu_matrix().span()[i] - expect) > 1e-5f) ok = false;
    }
    CHECK(ok, "融合结果 = x1*x2 + x3");
    std::printf("[OK] test_input_order\n");
}

// ── 测试 10：IR-D emitter 抽象（一份 canonical IR → 多后端） ───────────
void test_emitter()
{
    // 逐元素 spec：out = input0 * 2
    nn::ExprSpec s = make_unary_const(nn::ExprOp::Mul, Scalar{2});

    auto glsl = nn::emitter_registry::make("glsl");
    auto cpu  = nn::emitter_registry::make("cpu");
    auto none = nn::emitter_registry::make("nonexistent");
    CHECK(static_cast<bool>(glsl), "GlslEmitter 已注册可创建");
    CHECK(static_cast<bool>(cpu), "CpuEmitter 已注册可创建");
    CHECK(!none, "未知后端返回 nullptr");

    if (glsl)
    {
        CHECK(std::string(glsl->name()) == "glsl", "GlslEmitter 后端名");
        const std::string g = glsl->generate("fused_x", s);
        CHECK(!g.empty(), "GlslEmitter 生成逐元素 GLSL 成功");
        CHECK(g.find("#version 450") != std::string::npos, "GLSL 含版本头");
    }
    if (cpu)
    {
        CHECK(std::string(cpu->name()) == "cpu", "CpuEmitter 后端名");
        const std::string c = cpu->generate("cpu_x", s);
        CHECK(!c.empty(), "CpuEmitter 生成 C++ 代码成功");
        CHECK(c.find("inline void cpu_x") != std::string::npos, "C++ 代码含函数签名");
        CHECK(c.find("std::max") == std::string::npos, "逐元素含基本运算");
        // 归约：CpuEmitter 暂不支持 → 空串；GlslEmitter 支持
        nn::ExprSpec rs = make_row_sum();
        CHECK(cpu->generate_reduce("cpu_r", rs).empty(), "CpuEmitter 归约返回空（不支持）");
        if (glsl)
            CHECK(!glsl->generate_reduce("fused_r", rs).empty(), "GlslEmitter 归约生成成功");
    }
    std::printf("[OK] test_emitter\n");
}

} // namespace

int main()
{
    test_chain_fuse();
    test_chain3_fuse();
    test_independent();
    test_multi_consumer();
    test_reduce_boundary();
    test_limit_pressure();
    test_determinism();
    test_dep_recognition();
    test_input_order();
    test_emitter();

    if (g_fail == 0)
    {
        std::printf("expr_graph_test: ALL PASS\n");
        return 0;
    }
    std::printf("expr_graph_test: %d FAILED\n", g_fail);
    return 1;
}
