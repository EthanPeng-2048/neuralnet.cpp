// ── f16_cpu_probe — 临时诊断：CPU f16 训练发散的组件级定位 ─────────────────
// 目的：把"profile_f16 下 step0 forward 就 NaN"拆到具体 Layer/原语。
// 编译（独立于 CMake/AOT，纯 CPU）：
//   clang++ -std=c++26 -O1 -fno-exceptions -fexperimental-library \
//           -Wno-pass-failed -Iinclude src/f16_cpu_probe.cpp -o build/f16_cpu_probe.exe
// 每个阶段执行前打印标记（flush），崩溃时能直接看到断点位置。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "neuralnet.cpp/nn.hpp"

namespace
{
int g_bad = 0;

void banner(const char* tag)
{
    std::printf("── %s\n", tag);
    std::fflush(stdout);
}

// 扫描张量是否含非有限值；打印结论
void scan(nn::ComputeEngine& eng, const nn::Tensor& t, const char* tag)
{
    auto m = eng.to_matrix(t, nn::Precision::F32);
    if (!m)
    {
        std::printf("    [%s] to_matrix FAIL: %s\n", tag, m.error().message.c_str());
        ++g_bad;
        std::fflush(stdout);
        return;
    }
    for (auto v : m->span())
    {
        if (!std::isfinite(v))
        {
            std::printf("    [%s] *** NON-FINITE *** (%s)\n", tag, t.shape_str().c_str());
            ++g_bad;
            std::fflush(stdout);
            return;
        }
    }
    double mx = 0.0;
    for (auto v : m->span())
        mx = std::max(mx, std::fabs(static_cast<double>(v)));
    std::printf("    [%s] finite ok (%s) max|v|=%.6g%s\n", tag,
                t.shape_str().c_str(), mx, mx > 100.0 ? "  <<< 超大" : "");
    std::fflush(stdout);
}

nn::Matrix rnd(std::size_t rows, std::size_t cols, unsigned seed,
               float lo = -0.5f, float hi = 0.5f)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : m.span()) v = d(rng);
    return m;
}
} // namespace

int main()
{
    std::printf("f16_cpu_probe（profile_f16 组件级定位）\n");
    nn::CpuEngine raw;
    nn::PrecisionEngine ad(raw);
    const auto prof = nn::profile_f16();

    constexpr std::size_t D = 16, OUT = 32, SEQ = 4, BATCH = 2, TOTAL = SEQ * BATCH;

    // ── A. Linear forward：输入 f16 vs f32 ───────────────────────────────
    {
        banner("A1 Linear ctor+init (W/b/grad=f16)");
        nn::Linear ln(D, OUT);
        ln.set_precision_profile(prof);
        if (auto r = ln.init(ad); !r)
        {
            std::printf("    init FAIL: %s\n", r.error().message.c_str());
            return 1;
        }
        banner("A1b from_matrix(f16)");
        auto x16 = ad.from_matrix(rnd(D, TOTAL, 1), nn::Precision::F16);
        if (!x16) { std::printf("    from_matrix FAIL\n"); return 1; }
        banner("A1c Linear.forward(x=f16)");
        auto y = ln.forward(ad, *x16);
        if (!y) { std::printf("    forward ERR: %s\n", y.error().message.c_str()); ++g_bad; }
        else { std::printf("    y precision=%s\n", y->precision() == nn::Precision::F16 ? "f16" : "f32");
               scan(ad, *y, "A1 out"); }

        banner("A2 Linear.forward  (W=f16, x=f32)");
        auto x32 = ad.from_matrix(rnd(D, TOTAL, 2));
        if (!x32) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto y2 = ln.forward(ad, *x32);
        if (!y2) { std::printf("    forward ERR: %s\n", y2.error().message.c_str()); ++g_bad; }
        else scan(ad, *y2, "A2 out");

        banner("A3 Linear.backward (grad=f16)");
        auto g = ad.from_matrix(rnd(OUT, TOTAL, 3), nn::Precision::F16);
        if (!g) { std::printf("    from_matrix FAIL\n"); return 1; }
        if (auto z = ad.zero(ln.param_gradients()[0].get()); !z)
            std::printf("    zero grad FAIL\n");
        if (auto z = ad.zero(ln.param_gradients()[1].get()); !z)
            std::printf("    zero grad FAIL\n");
        auto gi = ln.backward(ad, *g);
        if (!gi) { std::printf("    backward ERR: %s\n", gi.error().message.c_str()); ++g_bad; }
        else scan(ad, *gi, "A3 grad_in");
        for (auto& gt : ln.param_gradients()) scan(ad, gt.get(), "A3 param_grad");
    }

    // ── B. LayerNorm forward（stable=F32，输入 f16）───────────────────────
    {
        banner("B LayerNorm.forward (gamma/beta=f16, x=f16, stable=f32)");
        nn::LayerNorm ln(D);
        ln.set_precision_profile(prof);
        if (auto r = ln.init(ad); !r) { std::printf("    init FAIL\n"); return 1; }
        auto x16 = ad.from_matrix(rnd(D, TOTAL, 4), nn::Precision::F16);
        if (!x16) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto y = ln.forward(ad, *x16);
        if (!y) { std::printf("    forward ERR: %s\n", y.error().message.c_str()); ++g_bad; }
        else scan(ad, *y, "B out");
    }

    // ── C. FeedForward forward（输入 f16）─────────────────────────────────
    {
        banner("C FeedForward.forward (x=f16, compute=f16)");
        nn::FeedForward ff(D, 2 * D);
        ff.set_precision_profile(prof);
        if (auto r = ff.init(ad); !r) { std::printf("    init FAIL\n"); return 1; }
        auto x16 = ad.from_matrix(rnd(D, TOTAL, 5), nn::Precision::F16);
        if (!x16) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto y = ff.forward(ad, *x16);
        if (!y) { std::printf("    forward ERR: %s\n", y.error().message.c_str()); ++g_bad; }
        else scan(ad, *y, "C out");
    }

    // ── D. CausalSelfAttention forward（输入 f16）─────────────────────────
    {
        banner("D CausalSelfAttention.forward (x=f16, compute=f16)");
        nn::CausalSelfAttention attn(D, 2, 1024, SEQ, nn::PosEncodingType::Learned);
        attn.set_precision_profile(prof);
        if (auto r = attn.init(ad); !r) { std::printf("    init FAIL\n"); return 1; }
        auto x16 = ad.from_matrix(rnd(D, TOTAL, 6), nn::Precision::F16);
        if (!x16) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto y = attn.forward(ad, *x16);
        if (!y) { std::printf("    forward ERR: %s\n", y.error().message.c_str()); ++g_bad; }
        else scan(ad, *y, "D out");
    }

    // ── E. CrossEntropyLoss（stable=f32，全 f32 输入）────────────────────
    {
        banner("E CrossEntropyLoss fwd/bwd (logits=f32, stable=f32)");
        nn::GptConfig cfg{};
        cfg.vocab_size = 32; cfg.d_model = D; cfg.seq_len = SEQ;
        cfg.num_heads = 2; cfg.d_ff = 2 * D; cfg.num_layers = 1;
        cfg.precision = nn::profile_f32();
        auto mr = nn::build_gpt_model(ad, cfg);
        if (!mr) { std::printf("    build FAIL\n"); return 1; }
        nn::Matrix x(SEQ, BATCH);
        for (std::size_t i = 0; i < SEQ * BATCH; ++i) x.span()[i] = static_cast<float>(i % 32);
        nn::Matrix tgt(32, TOTAL, 0.0f);
        for (std::size_t i = 0; i < TOTAL; ++i) tgt.set_value(i % 32, i, 1.0f);
        auto xt = ad.from_matrix(x);
        auto tt = ad.from_matrix(tgt);
        if (!xt || !tt) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto logits = mr->forward(*xt);
        if (!logits) { std::printf("    forward ERR\n"); ++g_bad; }
        else
        {
            scan(ad, *logits, "E logits(f32 profile)");
            nn::CrossEntropyLoss ce;
            ce.set_precision_profile(prof);
            auto loss = ce.forward(ad, *logits, *tt);
            if (!loss) { std::printf("    ce fwd ERR: %s\n", loss.error().message.c_str()); ++g_bad; }
            else std::printf("    loss=%.6f\n", *loss);
            auto grad = ce.backward();
            if (!grad) { std::printf("    ce bwd ERR\n"); ++g_bad; }
            else scan(ad, *grad, "E ce_grad");
        }
    }

    // ── F. optimizer=F16（param/compute/stable 全 f32）────────────────────
    {
        banner("F optimizer.step (opt=f16, 其余 f32)");
        const auto prof_f = nn::PrecisionProfile{nn::Precision::F32, nn::Precision::F32,
                                                 nn::Precision::F32, nn::Precision::F16};
        nn::Linear ln(D, OUT);
        ln.set_precision_profile(prof_f);
        if (auto r = ln.init(ad); !r) { std::printf("    init FAIL\n"); return 1; }
        auto opt = nn::create_optimizer("adam", ad, ln.parameters(), ln.param_gradients(),
                                        1e-3f, 0.0f, prof_f);
        if (!opt) { std::printf("    create_optimizer FAIL\n"); return 1; }
        for (int s = 0; s < 3; ++s)
        {
            auto x16 = ad.from_matrix(rnd(D, TOTAL, 100 + (unsigned)s));
            auto g = ad.from_matrix(rnd(OUT, TOTAL, 200 + (unsigned)s));
            if (!x16 || !g) { std::printf("    from_matrix FAIL\n"); return 1; }
            auto y = ln.forward(ad, *x16);
            if (!y) { std::printf("    fwd ERR\n"); ++g_bad; break; }
            auto gi = ln.backward(ad, *g);
            if (!gi) { std::printf("    bwd ERR\n"); ++g_bad; break; }
            if (auto r = opt->step(); !r)
            { std::printf("    opt step ERR: %s\n", r.error().message.c_str()); ++g_bad; break; }
            std::printf("    step %d: ", s);
            std::fflush(stdout);
            for (auto& p : ln.parameters()) scan(ad, p.get(), "F param");
        }
    }

    // ── H. 完整训练步：对每个"已知会 NaN"的 profile 跑 6 步 ──────────────
    // （debug 构建下 NN_ASSERT 会把残余的精度错配点直接钉到行号）
    {
        using P = nn::Precision;
        struct Case { const char* name; nn::PrecisionProfile prof; };
        const Case cases[] = {
            {"param=f16", {P::F16, P::F32, P::F32, P::F32}},
            {"compute=f16", {P::F32, P::F16, P::F32, P::F32}},
            {"f16(CLI)", nn::profile_f16()},
        };
        for (const auto& cs : cases)
        {
            banner(cs.name);
            nn::GptConfig cfg{};
            cfg.vocab_size = 32; cfg.d_model = D; cfg.seq_len = SEQ;
            cfg.num_heads = 2; cfg.d_ff = 2 * D; cfg.num_layers = 2;
            cfg.precision = cs.prof;
            auto mr = nn::build_gpt_model(ad, cfg);
            if (!mr) { std::printf("    build FAIL\n"); ++g_bad; continue; }
            // 与 f16_precision_test::train_tiny_gpt 同构：固定初值（σ=0.05）+ adamw
            {
                std::mt19937 rng(12345);
                std::normal_distribution<float> nd(0.0f, 0.05f);
                for (auto& p : mr->parameters())
                {
                    nn::Matrix m(p.get().rows(), p.get().cols());
                    for (auto& v : m.span()) v = nd(rng);
                    if (auto r = ad.copy_from(p.get(), m); !r)
                    { std::printf("    copy_from FAIL\n"); ++g_bad; }
                }
            }
            nn::Matrix x(SEQ, BATCH);
            for (std::size_t i = 0; i < SEQ * BATCH; ++i) x.span()[i] = static_cast<float>(i % 32);
            nn::Matrix tgt(32, TOTAL, 0.0f);
            for (std::size_t i = 0; i < TOTAL; ++i) tgt.set_value(i % 32, i, 1.0f);
            auto xt = ad.from_matrix(x);
            auto tt = ad.from_matrix(tgt);
            if (!xt || !tt) { std::printf("    from_matrix FAIL\n"); return 1; }
            auto opt = nn::create_optimizer("adamw", ad, mr->parameters(),
                                            mr->param_gradients(), 3e-3f, 0.0f, cs.prof);
            if (!opt) { std::printf("    optimizer FAIL\n"); ++g_bad; continue; }
            nn::CrossEntropyLoss ce;
            ce.set_precision_profile(cs.prof);
            for (int s = 0; s < 6; ++s)
            {
                if (auto r = mr->zero_grad(); !r) { std::printf("    zero_grad FAIL\n"); ++g_bad; break; }
                // step0：zero_grad 后 grad 应全 0 —— 若已非 0/非有限则 zero 或存量有问题
                if (s == 0)
                {
                    double pre_max = 0.0; bool pre_bad = false;
                    for (auto& p : mr->param_gradients())
                    {
                        auto m = ad.to_matrix(p.get(), nn::Precision::F32);
                        if (!m) continue;
                        for (auto v : m->span())
                        {
                            if (!std::isfinite(v)) pre_bad = true;
                            pre_max = std::max(pre_max, std::fabs(static_cast<double>(v)));
                        }
                    }
                    std::printf("    [%s] zero 后: max|g|=%.6g bad=%d\n", cs.name, pre_max,
                                pre_bad ? 1 : 0);
                    std::fflush(stdout);
                }
                auto logits = mr->forward(*xt);
                if (!logits) { std::printf("    fwd ERR: %s\n", logits.error().message.c_str()); ++g_bad; break; }
                char tag[64];
                std::snprintf(tag, sizeof(tag), "%s step%d logits", cs.name, s);
                scan(ad, *logits, tag);
                auto loss = ce.forward(ad, *logits, *tt);
                if (!loss) { std::printf("    ce ERR\n"); ++g_bad; break; }
                auto grad = ce.backward();
                if (!grad) { std::printf("    ce bwd ERR\n"); ++g_bad; break; }
                if (auto r = mr->backward(*grad); !r)
                { std::printf("    bwd ERR: %s\n", r.error().message.c_str()); ++g_bad; break; }
                bool bad = false;
                for (auto& p : mr->param_gradients())
                {
                    auto m = ad.to_matrix(p.get(), nn::Precision::F32);
                    if (!m) continue;
                    double mx = 0.0; bool this_bad = false; std::size_t first = 0;
                    for (std::size_t i = 0; i < m->span().size(); ++i)
                    {
                        const float v = m->span()[i];
                        if (!std::isfinite(v))
                        {
                            if (!this_bad) { first = i; this_bad = true; }
                        }
                        else mx = std::max(mx, std::fabs(static_cast<double>(v)));
                    }
                    if (this_bad)
                    {
                        bad = true;
                        std::printf("    [%s] step %d BAD grad %s [%s] first_bad_idx=%zu "
                                    "(%zu x %zu) max|finite|=%.6g\n",
                                    cs.name, s, p.get().shape_str().c_str(),
                                    p.get().precision() == nn::Precision::F16 ? "f16" : "f32",
                                    first, p.get().rows(), p.get().cols(), mx);
                        std::fflush(stdout);
                        break;
                    }
                    if (mx > 50.0)
                    {
                        std::printf("    [%s] step %d OUTLIER grad %s max|g|=%.6g\n",
                                    cs.name, s, p.get().shape_str().c_str(), mx);
                        std::fflush(stdout);
                    }
                }
                if (bad)
                {
                    std::printf("    [%s] step %d param-grad NON-FINITE\n", cs.name, s);
                    ++g_bad;
                }
                if (auto r = opt->step(); !r)
                { std::printf("    opt ERR: %s\n", r.error().message.c_str()); ++g_bad; break; }
                std::printf("    [%s] step %d loss=%.4f grads=%s\n", cs.name, s, *loss,
                            bad ? "BAD" : "ok");
                std::fflush(stdout);
                if (!std::isfinite(*loss) || bad) break;
            }
        }
    }

    // ── H0. GPTBlock 级：fwd+bwd 是否复现坏梯度 ──────────────────────────
    {
        using P = nn::Precision;
        struct Case { const char* name; nn::PrecisionProfile prof; };
        const Case cases[] = {
            {"f32", nn::profile_f32()},
            {"param=f16", {P::F16, P::F32, P::F32, P::F32}},
            {"compute=f16", {P::F32, P::F16, P::F32, P::F32}},
            {"f16(CLI)", nn::profile_f16()},
        };
        for (const auto& cs : cases)
        {
            banner((std::string("H0 GPTBlock ") + cs.name).c_str());
            nn::GPTBlock blk(D, 2, 2 * D, 1024, SEQ, nn::PosEncodingType::Learned,
                             nn::ActivationType::GeLU, nn::NormType::LayerNorm, cs.prof);
            if (auto r = blk.init(ad); !r) { std::printf("    init FAIL\n"); ++g_bad; continue; }
            // 固定初值（同 train_tiny_gpt：σ=0.05, seed12345）
            {
                std::mt19937 rng(12345);
                std::normal_distribution<float> nd(0.0f, 0.05f);
                for (auto& p : blk.parameters())
                {
                    nn::Matrix m(p.get().rows(), p.get().cols());
                    for (auto& v : m.span()) v = nd(rng);
                    if (auto r = ad.copy_from(p.get(), m); !r)
                    { std::printf("    copy_from FAIL\n"); ++g_bad; }
                }
            }
            const nn::Precision in_p = (cs.prof.compute == nn::Precision::F16)
                                           ? nn::Precision::F16 : nn::Precision::F32;
            auto x = ad.from_matrix(rnd(D, TOTAL, 42, -0.1f, 0.1f), in_p);
            if (!x) { std::printf("    from_matrix FAIL\n"); return 1; }
            auto y = blk.forward(ad, *x);
            if (!y) { std::printf("    fwd ERR: %s\n", y.error().message.c_str()); ++g_bad; continue; }
            scan(ad, *y, (std::string(cs.name) + " block-y").c_str());
            auto g = ad.from_matrix(rnd(D, TOTAL, 43, -0.1f, 0.1f), in_p);
            if (!g) { std::printf("    from_matrix FAIL\n"); return 1; }
            if (auto r = blk.backward(ad, *g); !r)
            { std::printf("    bwd ERR: %s\n", r.error().message.c_str()); ++g_bad; continue; }
            int gi = 0;
            for (auto& p : blk.param_gradients())
            {
                auto m = ad.to_matrix(p.get(), nn::Precision::F32);
                if (!m) { ++gi; continue; }
                double mx = 0.0; bool this_bad = false; std::size_t first = 0;
                for (std::size_t i = 0; i < m->span().size(); ++i)
                {
                    const float v = m->span()[i];
                    if (!std::isfinite(v)) { if (!this_bad) { first = i; this_bad = true; } }
                    else mx = std::max(mx, std::fabs(static_cast<double>(v)));
                }
                std::printf("    [%s] grad#%d %s [%s] %s idx=%zu max=%.6g\n",
                            cs.name, gi++, p.get().shape_str().c_str(),
                            p.get().precision() == nn::Precision::F16 ? "f16" : "f32",
                            this_bad ? "BAD" : (mx > 20.0 ? "outlier" : "ok"), first, mx);
                if (this_bad) ++g_bad;
            }
            std::fflush(stdout);
        }
    }

    // ── I. Linear::backward 手工拆解（param=f16, compute=f32, x/g=f32）─────
    {
        using P = nn::Precision;
        banner("I Linear.backward 拆解 (param=f16 compute=f32, x/g=f32)");
        nn::Linear ln2(D, OUT);
        ln2.set_precision_profile({P::F16, P::F32, P::F32, P::F32});
        if (auto r = ln2.init(ad); !r) { std::printf("    init FAIL\n"); return 1; }
        auto x = ad.from_matrix(rnd(D, TOTAL, 77, -0.1f, 0.1f));          // f32
        auto g = ad.from_matrix(rnd(OUT, TOTAL, 78, -0.1f, 0.1f));        // f32
        if (!x || !g) { std::printf("    from_matrix FAIL\n"); return 1; }
        auto y = ln2.forward(ad, *x);
        if (!y) { std::printf("    fwd ERR\n"); ++g_bad; }

        const auto& w = ln2.parameters()[0].get();
        const auto& cache = ln2.activation_cache()[0].get();
        std::printf("    w=%s %s | cache=%s %s\n", w.shape_str().c_str(),
                    w.precision() == nn::Precision::F16 ? "f16" : "f32",
                    cache.shape_str().c_str(),
                    cache.precision() == nn::Precision::F16 ? "f16" : "f32");

        // step1: grad_input = matmul(w, g, transA, P=compute=F32)
        auto gi = ad.matmul(w, *g, true, false, P::F32);
        if (!gi) { std::printf("    matmul ERR: %s\n", gi.error().message.c_str()); ++g_bad; }
        else scan(ad, *gi, "I grad_input");

        // step2: grad_w += matmul(g, cache^T) （compute_into, dst=f16）
        auto& gw = ln2.param_gradients()[0].get();
        auto r2 = nn::dsl::compute_into(ad,
            nn::dsl::leaf(gw) + nn::dsl::matmul(*g, cache, false, true), gw);
        if (!r2) { std::printf("    compute_into ERR: %s\n", r2.error().message.c_str()); ++g_bad; }
        else scan(ad, gw, "I grad_w (compute_into 后)");

        // step3: grad_b += row_reduce_sum(g)
        auto gb = ad.row_reduce_sum(*g, P::F32);
        if (!gb) { std::printf("    row_reduce ERR\n"); ++g_bad; }
        else
        {
            scan(ad, *gb, "I row_reduce(g)");
            auto& gbb = ln2.param_gradients()[1].get();
            if (auto r3 = ad.accumulate(gbb, *gb); !r3)
            { std::printf("    accumulate ERR\n"); ++g_bad; }
            else scan(ad, gbb, "I grad_b (accumulate 后)");
        }
    }

    std::printf("\nf16_cpu_probe: bad=%d\n", g_bad);
    return g_bad == 0 ? 0 : 1;
}
