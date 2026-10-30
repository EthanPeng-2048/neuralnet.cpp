// ── cross_attention_test — CrossAttention（issue #33）──────────────────────
//
// 覆盖：
//   [1] 一致性：同 seed 下投影权重与 MultiHeadAttention 相同 →
//       forward_pair(x, x) 与 MHA.forward(x) 逐位一致（batch>1）；
//       backward_pair 的 grad_query+grad_memory 与 MHA.backward(grad) 一致
//   [2] 矩形形状：seq_q ≠ seq_kv、batch>1 的输出形状与数值健全性
//   [3] 双输入语义：调换 query/memory（不同形状）不崩溃、输出形状正确
//   [4] 单输入 forward/backward fail-fast 指路
//   [5] 数值梯度检查（grad_query 与 grad_memory 各一轮，CPU 小规模）
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace {

// 逐位比对（容差 1e-4；两侧同引擎同精度）
void compare(nn::ComputeEngine& e, const nn::Tensor& got, const nn::Tensor& ref,
             const char* msg)
{
    CHECK(got.rows() == ref.rows() && got.cols() == ref.cols(), msg);
    if (got.rows() != ref.rows() || got.cols() != ref.cols()) return;
    const auto gm = download(e, got);
    const auto rm = download(e, ref);
    const auto g = gm.span();
    const auto r = rm.span();
    for (std::size_t i = 0; i < g.size(); ++i)
    {
        if (std::fabs(g[i] - r[i]) > 1e-4f)
        {
            std::printf("[FAIL] %s: idx=%zu got=%g ref=%g\n", msg, i, g[i], r[i]);
            ++g_fail;
            return;
        }
    }
}

// [1] 与 MultiHeadAttention 的一致性（query == memory == x）
void test_mha_consistency()
{
    // 两个独立引擎：创建序号各自从 0 起 → 同 seed 混流后投影权重逐位一致
    //（同引擎内两层的创建序号不同 → 权重不同，故不能共引擎对拍）
    auto engine = nn::CpuEngine::create();
    auto engine2 = nn::CpuEngine::create();
    const std::size_t d_model = 16, heads = 2, seq = 4, batch = 2;
    nn::MultiHeadAttention mha(d_model, heads, seq);
    nn::CrossAttention xattn(d_model, heads, seq, /*seq_kv_len=*/seq);
    CHECK(mha.init(*engine).has_value(), "[1] MHA init");
    CHECK(xattn.init(*engine2).has_value(), "[1] CrossAttention init");

    // 投影权重逐位一致（同构造序 + 同 seed → 引擎混流后同值）
    auto mha_ps = mha.parameters();
    auto xa_ps = xattn.parameters();
    CHECK(mha_ps.size() == xa_ps.size(), "[1] 参数条数一致");
    if (mha_ps.size() != xa_ps.size()) return;
    for (std::size_t i = 0; i < mha_ps.size(); ++i)
    {
        const auto a = download(*engine, mha_ps[i].get());
        const auto b = download(*engine2, xa_ps[i].get());
        const auto as = a.span();
        const auto bs = b.span();
        bool same = as.size() == bs.size();
        for (std::size_t j = 0; same && j < as.size(); ++j)
            same = as[j] == bs[j];
        CHECK(same, "[1] 投影权重逐位一致（同 seed）");
    }

    auto x = make_tensor(*engine, d_model, batch * seq, 0.05f, 0.03f);
    auto x2 = make_tensor(*engine2, d_model, batch * seq, 0.05f, 0.03f);
    auto yo = mha.forward(x);
    CHECK(yo.has_value(), "[1] MHA forward");
    auto xr = xattn.forward_pair(x2, x2);
    CHECK(xr.has_value(), "[1] forward_pair(x, x)");
    if (!yo || !xr) return;
    compare(*engine2, *xr, *yo, "[1] forward_pair(x,x) == MHA.forward(x)");

    auto grad = make_tensor(*engine, d_model, batch * seq, 0.02f, 0.01f);
    auto grad2 = make_tensor(*engine2, d_model, batch * seq, 0.02f, 0.01f);
    auto gmha = mha.backward(grad);
    CHECK(gmha.has_value(), "[1] MHA backward");
    auto gpair = xattn.backward_pair(grad2);
    CHECK(gpair.has_value(), "[1] backward_pair");
    if (!gmha || !gpair) return;
    // grad_query + grad_memory == MHA 的 grad_input（同一投影拆分的两半）
    CHECK(gpair->grad_query.rows() == d_model &&
          gpair->grad_query.cols() == batch * seq, "[1] grad_query 形状");
    CHECK(gpair->grad_memory.rows() == d_model &&
          gpair->grad_memory.cols() == batch * seq, "[1] grad_memory 形状");
    auto gsum = download(*engine2, gpair->grad_query);
    auto gmem = download(*engine2, gpair->grad_memory);
    auto gs = gsum.span();
    auto gms = gmem.span();
    auto refm = download(*engine, *gmha);
    auto rs = refm.span();
    for (std::size_t i = 0; i < gs.size(); ++i)
    {
        if (std::fabs(gs[i] + gms[i] - rs[i]) > 1e-4f)
        {
            std::printf("[FAIL] [1] grad_query+grad_memory == MHA grad_input: "
                        "idx=%zu got=%g ref=%g\n", i, gs[i] + gms[i], rs[i]);
            ++g_fail;
            break;
        }
    }
}

// [2] 矩形形状（seq_q=3, seq_kv=5, batch=2）：形状 + 有限性
void test_rectangular()
{
    auto engine = nn::CpuEngine::create();
    const std::size_t d_model = 8, heads = 2, seq_q = 3, seq_kv = 5, batch = 2;
    nn::CrossAttention xattn(d_model, heads, seq_q, seq_kv);
    CHECK(xattn.init(*engine).has_value(), "[2] init");
    auto q = make_tensor(*engine, d_model, batch * seq_q, 0.05f, 0.03f);
    auto m = make_tensor(*engine, d_model, batch * seq_kv, 0.01f, 0.02f);
    auto out = xattn.forward_pair(q, m);
    CHECK(out.has_value(), "[2] forward_pair（矩形）");
    if (!out) return;
    CHECK(out->rows() == d_model && out->cols() == batch * seq_q,
          "[2] 输出形状 (d_model, batch*seq_q)");
    const auto om = download(*engine, *out);
    bool finite = true;
    for (auto v : om.span()) if (!std::isfinite(v)) finite = false;
    CHECK(finite, "[2] 输出有限");

    auto grad = make_tensor(*engine, d_model, batch * seq_q, 0.02f, 0.01f);
    auto g = xattn.backward_pair(grad);
    CHECK(g.has_value(), "[2] backward_pair（矩形）");
    if (!g) return;
    CHECK(g->grad_query.rows() == d_model && g->grad_query.cols() == batch * seq_q,
          "[2] grad_query 形状");
    CHECK(g->grad_memory.rows() == d_model &&
          g->grad_memory.cols() == batch * seq_kv, "[2] grad_memory 形状");
    const auto grm = download(*engine, g->grad_memory);
    finite = true;
    for (auto v : grm.span()) if (!std::isfinite(v)) finite = false;
    CHECK(finite, "[2] grad_memory 有限");
}

// [3] 双输入独立性：batch 一致但两侧序列长不同 → 与 [2] 同构但内存不同值
void test_dual_input_independence()
{
    auto engine = nn::CpuEngine::create();
    const std::size_t d_model = 8, heads = 2, seq_q = 2, seq_kv = 4, batch = 3;
    nn::CrossAttention xattn(d_model, heads, seq_q, seq_kv);
    CHECK(xattn.init(*engine).has_value(), "[3] init");
    auto q = make_tensor(*engine, d_model, batch * seq_q, 0.1f, 0.05f);
    auto m = make_tensor(*engine, d_model, batch * seq_kv, -0.2f, 0.01f);
    auto out = xattn.forward_pair(q, m);
    CHECK(out.has_value(), "[3] forward_pair");
    if (!out) return;
    CHECK(out->cols() == batch * seq_q, "[3] 输出列 = batch*seq_q（镜像 query 侧）");

    // batch 不一致 → fail-fast
    auto m_bad = make_tensor(*engine, d_model, (batch + 1) * seq_kv, 0.01f, 0.01f);
    auto r = xattn.forward_pair(q, m_bad);
    CHECK(!r.has_value(), "[3] batch 不一致 fail-fast");
}

// [4] 单输入协议 fail-fast
void test_single_input_guard()
{
    auto engine = nn::CpuEngine::create();
    nn::CrossAttention xattn(8, 2, 3, 4);
    CHECK(xattn.init(*engine).has_value(), "[4] init");
    auto x = make_tensor(*engine, 8, 3, 0.01f, 0.01f);
    auto fr = xattn.forward(x);
    CHECK(!fr.has_value(), "[4] 单输入 forward fail-fast");
    auto br = xattn.backward(x);
    CHECK(!br.has_value(), "[4] 单输入 backward fail-fast");
}

// [5] 数值梯度检查（CPU 小规模）：loss = <forward_pair(q,m), W>，中心差分
//     分别扰动 q / m 的条目，对照 backward_pair 的两路解析梯度。
void test_gradcheck()
{
    auto engine = nn::CpuEngine::create();
    const std::size_t d_model = 4, heads = 2, seq_q = 2, seq_kv = 3, batch = 1;
    const nn::Scalar eps = 1e-2f, tol = 2e-2f;
    nn::CrossAttention xattn(d_model, heads, seq_q, seq_kv);
    CHECK(xattn.init(*engine).has_value(), "[5] init");

    auto q = make_tensor(*engine, d_model, batch * seq_q, 0.05f, 0.04f);
    auto m = make_tensor(*engine, d_model, batch * seq_kv, 0.03f, 0.02f);
    // 固定投影权重（确定性小值）：扰动只作用于输入
    {
        std::size_t k = 0;
        for (auto& p : xattn.parameters())
        {
            auto pm = download(*engine, p.get());
            auto sp = pm.span();
            for (std::size_t i = 0; i < sp.size(); ++i, ++k)
                sp[i] = 0.05f * std::sin(0.7f * static_cast<nn::Scalar>(
                    (k * 17) % 23 + 1));
            auto cp = engine->copy_from(p.get(), pm);
            CHECK(cp.has_value(), "[5] 权重回写");
        }
    }

    // loss = <Y, W>；W 是确定性固定权重（Y 同形状）
    const std::size_t y_n = d_model * batch * seq_q;
    std::vector<nn::Scalar> W(y_n);
    for (std::size_t i = 0; i < y_n; ++i)
        W[i] = 0.01f * static_cast<nn::Scalar>(i % 7 + 1);

    auto eval = [&]() -> nn::Scalar {
        auto y = xattn.forward_pair(q, m);
        NN_ASSERT_ALWAYS(y, "gradcheck eval: forward_pair failed");
        auto ym = download(*engine, *y);
        const auto sp = ym.span();
        nn::Scalar loss = 0;
        for (std::size_t i = 0; i < sp.size(); ++i) loss += sp[i] * W[i];
        return loss;
    };

    // 解析梯度：go 第 i 列 = W 第 i 列 → backward_pair 的两路梯度即
    // d<Y,W>/dq 与 d<Y,W>/dm（W 由 go 的列主序承载）
    nn::Matrix go_m(d_model, batch * seq_q);
    { auto sp = go_m.span(); for (std::size_t i = 0; i < y_n; ++i) sp[i] = W[i]; }
    auto go = upload(*engine, go_m);
    (void)eval();   // backward_pair 需要 forward_pair 的缓存
    auto g = xattn.backward_pair(go);
    CHECK(g.has_value(), "[5] backward_pair");
    if (!g)
    {
        std::printf("[FAIL] [5] backward_pair: %s\n", g.error().message.c_str());
        return;
    }
    auto ga_q = download(*engine, g->grad_query);
    auto ga_m = download(*engine, g->grad_memory);

    // in-place 扰动辅助：改 t 的第 idx 个标量（经 copy_from 写回）
    auto perturb = [&](nn::Tensor& t, std::size_t idx, nn::Scalar v) {
        auto tm = download(*engine, t);
        tm.span()[idx] = v;
        auto cp = engine->copy_from(t, tm);
        NN_ASSERT_ALWAYS(cp, "gradcheck perturb: copy_from failed");
    };

    const nn::Scalar base_q = q.cols() > 0 ? nn::Scalar{0} : nn::Scalar{0};
    (void)base_q;
    const nn::Scalar base_loss = eval();

    std::size_t bad_q = 0, bad_m = 0;
    // query 侧抽 4 个条目
    const std::size_t nq = d_model * batch * seq_q;
    auto qm0 = download(*engine, q);
    for (std::size_t j = 0; j < 4; ++j)
    {
        const std::size_t idx = (j * 5) % nq;
        const nn::Scalar orig = qm0.span()[idx];
        perturb(q, idx, orig + eps);
        const nn::Scalar lp = eval();
        perturb(q, idx, orig - eps);
        const nn::Scalar lm = eval();
        perturb(q, idx, orig);
        const nn::Scalar num = (lp - lm) / (2 * eps);
        const nn::Scalar ana = ga_q.span()[idx];
        if (std::fabs(num - ana) > tol)
        {
            std::printf("[FAIL] [5] grad_query idx=%zu num=%g ana=%g\n",
                        idx, num, ana);
            ++bad_q;
        }
    }
    // memory 侧抽 4 个条目
    const std::size_t nm = d_model * batch * seq_kv;
    auto mm0 = download(*engine, m);
    for (std::size_t j = 0; j < 4; ++j)
    {
        const std::size_t idx = (j * 7) % nm;
        const nn::Scalar orig = mm0.span()[idx];
        perturb(m, idx, orig + eps);
        const nn::Scalar lp = eval();
        perturb(m, idx, orig - eps);
        const nn::Scalar lm = eval();
        perturb(m, idx, orig);
        const nn::Scalar num = (lp - lm) / (2 * eps);
        const nn::Scalar ana = ga_m.span()[idx];
        if (std::fabs(num - ana) > tol)
        {
            std::printf("[FAIL] [5] grad_memory idx=%zu num=%g ana=%g\n",
                        idx, num, ana);
            ++bad_m;
        }
    }
    (void)base_loss;
    CHECK(bad_q == 0, "[5] grad_query 数值梯度一致");
    CHECK(bad_m == 0, "[5] grad_memory 数值梯度一致");
}

} // namespace

int main()
{
    test_mha_consistency();
    test_rectangular();
    test_dual_input_independence();
    test_single_input_guard();
    test_gradcheck();
    if (g_fail == 0) std::printf("cross_attention_test: ALL PASS\n");
    else std::printf("cross_attention_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
