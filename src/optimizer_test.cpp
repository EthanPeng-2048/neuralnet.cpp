// ── optimizer_test — 优化器元数据表（RV-04）与同目标写序（RV-03）回归 ────────
//
// 两件事都属于"新增/改动优化器时最容易静默错"的那一类：
//
//  [1] RV-04 元数据表 ↔ 工厂一致：nn::optimizer_table() 是名单的**唯一真源**
//      （CLI 校验/帮助、scan_exprs dry-run、mem_probe 状态账、GUI 选项全部查表）。
//      断言：表内每个 name 都能被 create_optimizer 构造；未知 name → nullptr；
//      名单文本与表一致（帮助文本正是由它拼出来的）。
//
//  [2] RV-03 同目标写序：AdamW 的 `p = (1−lr·wd)·p` 与 `p += delta` 是**同一目标
//      的两次写、无数据依赖边**，正确性依赖执行序 = 录制序。这里给三件：
//      ① 单步数值 vs 手写参考（CPU、wd≠0）；
//      ② 顺序敏感性的**负例**：把两次写互换 → 结果必须明显不同（证明"顺序"这件
//         事真的影响数值，故 IR-C 的显式顺序边不是形式主义）；
//      ③ 图分析的回归：同目标两节点必须建出顺序边、融合不得把它们合组、
//         故意反序时 validate_dst_order 必须 fail-fast。
//
//  [4] P1-5 训练冒烟：小 MLP + Adam 跑 8 步，断言 loss 有限且下降、参数**确实
//      移动**；并用"同一快照自比"的负例证明检测器能识别"完全没动"的冻结。
//      本项不依赖 GPU，MSVC Debug CPU-only CI 会直接跑到。
//
//  构建/运行：cmake -B build -DNN_ENABLE_TESTS=ON && cmake --build build
//            ctest --test-dir build -R optimizer_test
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

using nn::Scalar;

namespace
{

int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg));    \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

// ── 引擎侧工具（M1：Tensor 一律经引擎创建/读写）────────────────────────────
nn::Tensor make_tensor(nn::CpuEngine& eng, std::size_t rows, std::size_t cols,
                       float base, float step)
{
    nn::Tensor t = eng.create_tensor(rows, cols);
    std::vector<Scalar> buf(rows * cols);
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = base + static_cast<float>(i) * step;
    auto w = eng.write(t, std::span<Scalar>(buf));
    if (!w)
    {
        std::printf("[FAIL] write 失败: %s\n", w.error().message.c_str());
        ++g_fail;
    }
    return t;
}

std::vector<Scalar> read_back(nn::CpuEngine& eng, const nn::Tensor& t)
{
    auto m = eng.to_matrix(t);
    if (!m)
    {
        std::printf("[FAIL] to_matrix 失败: %s\n", m.error().message.c_str());
        ++g_fail;
        return {};
    }
    const auto sp = m->span();
    return std::vector<Scalar>(sp.begin(), sp.end());
}

// ── 逐元素 spec（与 expr_graph_test 同构的最小构造器）──────────────────────
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

// ── [1] RV-04：元数据表 ↔ 工厂 ─────────────────────────────────────────────
void test_optimizer_table()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const auto table = nn::optimizer_table();
    CHECK(!table.empty(), "优化器表非空");

    std::string expect_slash;
    for (const auto& m : table)
    {
        if (!expect_slash.empty()) expect_slash += '/';
        expect_slash += m.name;

        // ① 表内每个 name 都能被工厂构造（名单与构造器不得漂移）
        nn::Tensor p = make_tensor(eng, 2, 3, 0.1f, 0.01f);
        nn::Tensor g = make_tensor(eng, 2, 3, 0.2f, 0.01f);
        auto opt = nn::create_optimizer(m.name, eng,
                                        std::vector<nn::TensorRef>{p},
                                        std::vector<nn::TensorRef>{g},
                                        /*lr=*/Scalar{1e-3f}, /*wd=*/Scalar{0.01f});
        CHECK(opt != nullptr, (std::string("工厂可构造: ") + std::string(m.name)).c_str());
        // ② 查表与工厂口径一致
        const nn::OptimizerMeta* hit = nn::find_optimizer(m.name);
        CHECK(hit == &m, (std::string("find_optimizer 命中: ") + std::string(m.name)).c_str());
        // ③ 状态缓冲账自洽（mun g 判据：有状态 ⇒ state_bufs ≥ 1；半精度状态 ⇒ 有状态）
        if (m.half_state) CHECK(m.state_bufs >= 1, "half_state 必伴随状态缓冲");
    }
    CHECK(nn::find_optimizer("no_such_optimizer") == nullptr, "未知优化器 → nullptr");
    CHECK(nn::optimizer_names("/") == expect_slash, "optimizer_names('/') 与表一致");

    std::string expect_comma;
    for (const auto& m : table)
    {
        if (!expect_comma.empty()) expect_comma += ", ";
        expect_comma += m.name;
    }
    CHECK(nn::optimizer_names(", ") == expect_comma, "optimizer_names(', ') 与表一致");
    std::printf("[OK] test_optimizer_table（%zu 项）\n", table.size());
}

// ── AdamW 单步参考（与 compute_optimizer.hpp::AdamW::step 同一公式）────────
// decay_first = true  → 先衰减后更新（AdamW 语义，正确）
// decay_first = false → 先更新后衰减（错误顺序，仅用于证明顺序敏感）
void adamw_reference(std::vector<Scalar>& p, const std::vector<Scalar>& g,
                     std::vector<Scalar>& m, std::vector<Scalar>& v,
                     Scalar lr, Scalar b1, Scalar b2, Scalar eps, Scalar wd,
                     std::size_t t, bool decay_first)
{
    const Scalar bc1 = Scalar{1} - std::pow(b1, static_cast<Scalar>(t));
    const Scalar bc2 = Scalar{1} - std::pow(b2, static_cast<Scalar>(t));
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        m[i] = b1 * m[i] + (Scalar{1} - b1) * g[i];
        v[i] = b2 * v[i] + (Scalar{1} - b2) * g[i] * g[i];
    }
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        if (decay_first) p[i] *= (Scalar{1} - lr * wd);
        p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
        if (!decay_first) p[i] *= (Scalar{1} - lr * wd);
    }
}

// ── [2] RV-03：AdamW（wd≠0）单步 vs 参考 + 顺序敏感性负例 ─────────────────
void test_adamw_order()
{
    const Scalar lr = 0.05f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
    const std::size_t rows = 3, cols = 4;

    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    nn::Tensor p = make_tensor(eng, rows, cols, 0.5f, 0.03f);
    nn::Tensor g = make_tensor(eng, rows, cols, -0.2f, 0.02f);
    const std::vector<Scalar> p0 = read_back(eng, p);
    const std::vector<Scalar> g0 = read_back(eng, g);

    nn::AdamW opt(eng, {p}, {g}, lr, b1, b2, eps, wd);
    auto st = opt.step();
    CHECK(static_cast<bool>(st), "AdamW 单步成功");
    const std::vector<Scalar> got = read_back(eng, p);
    CHECK(got.size() == p0.size(), "参数字数不变");
    if (got.size() != p0.size()) return;

    std::vector<Scalar> ref = p0;
    std::vector<Scalar> m(p0.size(), Scalar{0}), v(p0.size(), Scalar{0});
    adamw_reference(ref, g0, m, v, lr, b1, b2, eps, wd, /*t=*/1, /*decay_first=*/true);

    Scalar max_abs = 0, max_rel = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        const Scalar d = std::fabs(got[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        const Scalar denom = std::max(std::fabs(ref[i]), Scalar{1e-6f});
        max_rel = std::max(max_rel, d / denom);
    }
    std::printf("    AdamW 单步 vs 参考: max_abs=%.3e max_rel=%.3e\n",
                static_cast<double>(max_abs), static_cast<double>(max_rel));
    CHECK(max_rel < 1e-5f, "AdamW 单步与参考实现一致（wd≠0，先衰减后更新）");

    // 负例：同样的超参、只把两次写的顺序对调 → 结果必须明显不同
    std::vector<Scalar> ref_swapped = p0;
    std::vector<Scalar> m2(p0.size(), Scalar{0}), v2(p0.size(), Scalar{0});
    adamw_reference(ref_swapped, g0, m2, v2, lr, b1, b2, eps, wd, /*t=*/1,
                    /*decay_first=*/false);
    Scalar swap_diff = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
        swap_diff = std::max(swap_diff, std::fabs(ref[i] - ref_swapped[i]));
    std::printf("    顺序对调后的差值: %.3e（应当显著非零）\n",
                static_cast<double>(swap_diff));
    CHECK(swap_diff > 1e-6f, "decay→update 的顺序确实影响数值（顺序边非形式主义）");
    std::printf("[OK] test_adamw_order\n");
}

// ── [3] RV-03：同目标写序的图分析（建边 / 不合组 / 反序 fail-fast）────────
void test_dst_order_hazard()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const std::size_t rows = 2, cols = 3;
    nn::Tensor p = make_tensor(eng, rows, cols, 0.5f, 0.1f);
    nn::Tensor x = make_tensor(eng, rows, cols, 1.0f, 0.0f);

    nn::ExprGraph graph;
    // 节点 0：p *= 2（写目标 = p）
    const int n0 = graph.add_node(make_unary_const(nn::ExprOp::Mul, Scalar{2}),
                                  std::vector<nn::Tensor>{p}, rows, cols, false,
                                  nn::Precision::F32, p.storage_id());
    // 节点 1：p = p + x（读 p、写目标同样 = p）
    const int n1 = graph.add_node(make_binary(nn::ExprOp::Add),
                                  std::vector<nn::Tensor>{p, x}, rows, cols, false,
                                  nn::Precision::F32, p.storage_id());

    // ① 顺序隐患边 (n0, n1)
    const auto edges = nn::dst_order_constraints(graph);
    bool has_edge = false;
    for (const auto& e : edges)
        if (e.first == n0 && e.second == n1) has_edge = true;
    CHECK(has_edge, "同目标两次写 → 建出显式顺序边 (n0, n1)");

    // ② 融合不得把隐患两端合组
    auto kernels = nn::fuse_expr_graph(graph);
    CHECK(kernels.size() >= 2, "同目标写序的节点不得合进同一 kernel");
    bool same_kernel = false;
    for (const auto& k : kernels)
    {
        bool a = false, b = false;
        for (const int m : k.members) { a = a || (m == n0); b = b || (m == n1); }
        if (a && b) same_kernel = true;
    }
    CHECK(!same_kernel, "两节点未出现在同一 kernel 的成员表里");

    // ③ 正序通过；故意反序 → 必须 fail-fast（负例：证明这道闸抓得住重排）
    CHECK(static_cast<bool>(nn::validate_dst_order(graph, kernels)),
          "执行序 = 融合序（正序）时顺序校验通过");
    std::vector<nn::FusedKernel> reversed(kernels.rbegin(), kernels.rend());
    CHECK(!static_cast<bool>(nn::validate_dst_order(graph, reversed)),
          "故意反序 → 顺序校验必须报错（不静默算错）");
    std::printf("[OK] test_dst_order_hazard\n");
}

// ── [4] P1-5：训练冒烟 + 权重变化检测（防"loss 不动"类静默冻结）──────────
// 与 f16_writeback_probe 的"8 步 Adam 参数必须移动"同一思想，但跑纯 CPU f32，
// 因而在没有 Vulkan 的 MSVC Debug job 里也会执行（不会 77 跳过）。
double l1_move(const std::vector<Scalar>& before, const std::vector<Scalar>& after)
{
    if (before.size() != after.size()) return -1.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i)
        sum += std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i]));
    return sum;
}

bool detect_movement(const std::vector<Scalar>& before, const std::vector<Scalar>& after)
{ return l1_move(before, after) > 1e-3; }

std::vector<Scalar> snapshot_params(nn::CpuEngine& eng, nn::Model& model)
{
    std::vector<Scalar> out;
    for (auto& t : model.parameters())
    {
        auto m = eng.to_matrix(t.get(), nn::Precision::F32);
        if (!m)
        {
            std::printf("[FAIL] 参数快照: %s\n", m.error().message.c_str());
            ++g_fail;
            return {};
        }
        out.insert(out.end(), m->span().begin(), m->span().end());
    }
    return out;
}

void test_weight_movement_smoke()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    nn::PrecisionProfile prof;   // 默认全 f32
    auto m_r = nn::build_mnist_mlp_model(eng, nn::MNIST_LAYER_DIMS, nn::NormType::LayerNorm, prof);
    CHECK(m_r.has_value(), "训练冒烟：build_mnist_mlp_model");
    if (!m_r) return;
    nn::Model& model = *m_r;

    auto opt = nn::create_optimizer("adam", eng, model.parameters(), model.param_gradients(),
                                    Scalar{0.02f}, Scalar{0.0f}, prof);
    CHECK(opt != nullptr, "训练冒烟：create_optimizer(adam)");
    if (!opt) return;

    const std::size_t in_dim = nn::MNIST_LAYER_DIMS.front();
    constexpr std::size_t B = 8, C = 10;
    nn::Matrix X(in_dim, B), Y(C, B);
    for (std::size_t k = 0; k < in_dim; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, ((k * 7 + b * 3) % 11) * 0.09f - 0.5f);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t b = 0; b < B; ++b)
            Y.set_value(c, b, (c == (b % C)) ? 1.0f : 0.0f);

    auto xt = eng.from_matrix(X);
    auto yt = eng.from_matrix(Y);
    CHECK(xt.has_value() && yt.has_value(), "训练冒烟：输入/目标上传");
    if (!xt || !yt) return;

    const std::vector<Scalar> before = snapshot_params(eng, model);
    CHECK(!before.empty(), "训练冒烟：参数快照非空");
    if (before.empty()) return;

    nn::CrossEntropyLoss loss;
    Scalar first = 0.0f, last = 0.0f;
    bool all_finite = true;
    for (int s = 0; s < 8; ++s)
    {
        if (auto r = model.zero_grad(); !r)
        {
            CHECK(false, "训练冒烟：zero_grad");
            return;
        }
        auto logits = model.forward(*xt);
        if (!logits)
        {
            CHECK(false, "训练冒烟：forward");
            return;
        }
        std::vector<std::size_t> labels(B);
        std::vector<Scalar> mask(B, Scalar{1});
        for (std::size_t b = 0; b < B; ++b)
            labels[b] = b % C;
        auto ls = loss.forward_sparse(eng, *logits, labels, mask, C);
        if (!ls)
        {
            CHECK(false, "训练冒烟：loss");
            return;
        }
        if (s == 0) first = *ls;
        last = *ls;
        all_finite = all_finite && std::isfinite(*ls);
        auto gl = loss.backward();
        if (!gl)
        {
            CHECK(false, "训练冒烟：loss.backward");
            return;
        }
        auto bw = model.backward(*gl);
        if (!bw)
        {
            CHECK(false, "训练冒烟：model.backward");
            return;
        }
        auto st = opt->step();
        if (!st)
        {
            CHECK(false, "训练冒烟：optimizer.step");
            return;
        }
    }

    const std::vector<Scalar> after = snapshot_params(eng, model);
    CHECK(after.size() == before.size(), "训练冒烟：训练前后快照大小一致");
    const double moved = l1_move(before, after);
    std::printf("    loss %.6f → %.6f（8 步），参数 L1 位移 %.6g\n", static_cast<double>(first),
                static_cast<double>(last), moved);
    CHECK(all_finite, "8 步训练 loss 全部有限");
    CHECK(last < first, "8 步训练 loss 下降");
    CHECK(detect_movement(before, after), "8 步 Adam 后参数确实移动（防静默冻结）");
    // 负例：同一快照自比必须判为"没有移动"——证明上面的检测器不是恒真断言
    CHECK(!detect_movement(before, before), "冻结检测器：未更新的参数必须判为无移动");
    std::printf("[OK] test_weight_movement_smoke\n");
}

} // namespace

int main()
{
    test_optimizer_table();
    test_adamw_order();
    test_dst_order_hazard();
    test_weight_movement_smoke();

    if (g_fail == 0)
    {
        std::printf("optimizer_test: ALL PASS\n");
        return 0;
    }
    std::printf("optimizer_test: %d FAILED\n", g_fail);
    return 1;
}
