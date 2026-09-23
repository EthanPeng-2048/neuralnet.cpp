// ── attn_test — Attention 合并测试 ──────────────────────────────────────────
// 合并：attn_gradcheck + attn_consistency_test + attn_w_batch_test
// 覆盖：CausalSelfAttention 梯度检查 / forward vs forward_step 一致性 /
//       多头多 batch W 表达式行号回归 /
//       MHA 双向语义探针（P-C2-7 fold 掩码回归钉子）+ CPU/GPU 一致性
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_attn_gradcheck
#include "attn_gradcheck.cpp"
#undef main

#define main test_attn_consistency
#include "attn_consistency_test.cpp"
#undef main

#define main test_attn_w_batch
#include "attn_w_batch_test.cpp"
#undef main

// ── MHA 双向语义探针（P-C2-7 回归钉子）──────────────────────────────────
// 回归背景：fold 迁移曾把 fmask 默认树设为 Causal → MultiHeadAttention
// （本应双向无掩码，旧 apply_mask_ 默认 no-op）被静默因果遮蔽；层级此前
// 只测 CSA，未被抓住。
// 探针：仅扰动输入的"未来"列（同 batch 内 t=seq-1），位置 0 的输出必须
//   变化（双向可见）；因果掩码下位置 0 只见自己 → 输出逐位不变 → 失败。
// CSA 阴性对照：同一扰动下因果层位置 0 输出必须**不变**（证明探针本身
//   具备区分力，而非恒真）。附带 CPU/GPU forward+backward 一致性——GPU
//   侧同时覆盖 Plain fold key 与 MHA 裸 S 重算表达式的闭合世界注册
//   （miss 会硬报错）。
static int mha_bidirectional_probe()
{
    int failures = 0;
    auto cpu_res = nn::cli::create_engine(nn::cli::EngineConfig{}, std::cout);
    if (!cpu_res) { std::cerr << "MHA probe: CPU 引擎创建失败\n"; return 1; }
    nn::ComputeEngine& cpu = **cpu_res;

    const std::size_t d_model = 16, heads = 2, seq = 4, batch = 2;
    const std::size_t total = batch * seq;
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<nn::Scalar> dist(-1, 1);

    nn::Matrix x1(d_model, total);
    for (std::size_t i = 0; i < x1.size(); ++i) x1.span()[i] = dist(rng);
    nn::Matrix x2(d_model, total);          // 仅扰动 (b0, t=seq-1) 一列
    for (std::size_t i = 0; i < x2.size(); ++i) x2.span()[i] = x1.span()[i];
    for (std::size_t r = 0; r < d_model; ++r)
        x2.span()[r * total + (seq - 1)] = dist(rng);

    // 位置 0 输出的扰动敏感度（d_model 行 × 第 0 列）
    const auto col0_delta = [](const nn::Matrix& a, const nn::Matrix& b) {
        nn::Scalar m = 0;
        for (std::size_t r = 0; r < a.rows(); ++r)
        {
            const nn::Scalar d = std::fabs(
                a.span()[r * a.cols()] - b.span()[r * b.cols()]);
            if (d > m) m = d;
        }
        return m;
    };

    // ── CPU 半场：MHA 必须敏感（双向），CSA 必须不敏感（因果阴性对照）──
    nn::MultiHeadAttention attn_c(d_model, heads, seq);
    { auto r = attn_c.init(cpu); if (!r) { std::cerr << "MHA cpu init: " << r.error().message << "\n"; return 1; } }
    auto t1 = cpu.from_matrix(x1);
    auto t2 = cpu.from_matrix(x2);
    if (!t1 || !t2) { std::cerr << "MHA probe: from_matrix failed\n"; return 1; }
    auto y1 = attn_c.forward(cpu, *t1);
    auto y2 = attn_c.forward(cpu, *t2);
    if (!y1 || !y2) { std::cerr << "MHA probe forward: "
                                << (!y1 ? y1.error().message : y2.error().message) << "\n"; return 1; }
    auto m1 = cpu.to_matrix(*y1);
    auto m2 = cpu.to_matrix(*y2);
    if (!m1 || !m2) { std::cerr << "MHA probe: to_matrix failed\n"; return 1; }
    const nn::Scalar d_mha = col0_delta(*m1, *m2);
    const bool mha_ok = d_mha > 1e-5f;
    std::cout << "  MHA 双向探针(改未来列→位置0须变): delta=" << d_mha
              << (mha_ok ? " ✅" : " ❌ 疑似因果掩码回归") << "\n";
    if (!mha_ok) ++failures;

    nn::CausalSelfAttention attn_cs_c(d_model, heads, seq, seq, nn::PosEncodingType::Learned);
    { auto r = attn_cs_c.init(cpu); if (!r) { std::cerr << "CSA cpu init: " << r.error().message << "\n"; return 1; } }
    auto cy1 = attn_cs_c.forward(cpu, *t1);
    auto cy2 = attn_cs_c.forward(cpu, *t2);
    if (!cy1 || !cy2) { std::cerr << "CSA probe forward failed\n"; return 1; }
    auto cm1 = cpu.to_matrix(*cy1);
    auto cm2 = cpu.to_matrix(*cy2);
    if (!cm1 || !cm2) { std::cerr << "CSA probe: to_matrix failed\n"; return 1; }
    const nn::Scalar d_csa = col0_delta(*cm1, *cm2);
    const bool csa_ok = d_csa < 1e-5f;
    std::cout << "  CSA 因果阴性对照(位置0须不变): delta=" << d_csa
              << (csa_ok ? " ✅" : " ❌ 探针失效(扰动列未命中)") << "\n";
    if (!csa_ok) ++failures;

    // ── GPU 半场（可用时）：探针 + 一致性 + backward 闭合世界 ──────────
    nn::cli::EngineConfig gcfg;
    gcfg.use_gpu = true;
    auto gpu_res = nn::cli::create_engine(gcfg, std::cout);
    if (!gpu_res)
    {
        std::cout << "  GPU 不可用，跳过 GPU 半场: " << gpu_res.error().message << "\n";
        return failures;
    }
    nn::ComputeEngine& gpu = **gpu_res;
    nn::MultiHeadAttention attn_g(d_model, heads, seq);
    { auto r = attn_g.init(gpu); if (!r) { std::cerr << "MHA gpu init: " << r.error().message << "\n"; return 1; } }
    {   // 权重同步：CPU 随机 → GPU
        auto pc = attn_c.parameters();
        auto pg = attn_g.parameters();
        for (std::size_t i = 0; i < pc.size(); ++i)
        {
            auto cm = cpu.to_matrix(pc[i].get());
            if (!cm) { std::cerr << "to_matrix cpu failed\n"; return 1; }
            auto cp = gpu.copy_from(pg[i].get(), *cm);
            if (!cp) { std::cerr << "copy_from gpu failed: " << cp.error().message << "\n"; return 1; }
        }
    }
    auto g1 = gpu.from_matrix(x1);
    auto g2 = gpu.from_matrix(x2);
    if (!g1 || !g2) { std::cerr << "MHA probe gpu from_matrix failed\n"; return 1; }
    auto gy1 = attn_g.forward(gpu, *g1);     // 覆盖 GPU Plain fold key
    auto gy2 = attn_g.forward(gpu, *g2);     // GPU 双向探针
    if (!gy1 || !gy2)
    {
        std::cerr << "MHA GPU forward FAILED: "
                  << (!gy1 ? gy1.error().message : gy2.error().message) << "\n";
        return 1;
    }
    auto gm1 = gpu.to_matrix(*gy1);
    auto gm2 = gpu.to_matrix(*gy2);
    if (!gm1 || !gm2) { std::cerr << "MHA gpu to_matrix failed\n"; return 1; }
    const nn::Scalar d_gpu = col0_delta(*gm1, *gm2);
    const bool gpu_ok = d_gpu > 1e-5f;
    std::cout << "  MHA GPU 双向探针: delta=" << d_gpu
              << (gpu_ok ? " ✅" : " ❌ GPU 疑似因果掩码回归") << "\n";
    if (!gpu_ok) ++failures;

    nn::Scalar max_abs = 0;
    for (std::size_t i = 0; i < m1->size(); ++i)
    {
        const nn::Scalar d = std::fabs(m1->span()[i] - gm1->span()[i]);
        if (d > max_abs) max_abs = d;
    }
    const bool cons_ok = max_abs <= 1e-3f;
    std::cout << "  MHA CPU/GPU forward 一致性: max_abs=" << max_abs
              << (cons_ok ? " ✅" : " ❌") << "\n";
    if (!cons_ok) ++failures;

    // backward 一致性：两侧最后 forward 对齐为同一输入后反传同一梯度
    // （GPU 侧触发 MHA 裸 S 重算表达式——scan 闭合世界注册的验针）
    auto b1c = attn_c.forward(cpu, *t1);
    auto b1g = attn_g.forward(gpu, *g1);
    if (!b1c || !b1g) { std::cerr << "MHA re-forward failed\n"; return 1; }
    nn::Matrix gm(d_model, total);
    for (std::size_t i = 0; i < gm.size(); ++i) gm.span()[i] = dist(rng);
    auto tg_c = cpu.from_matrix(gm);
    auto tg_g = gpu.from_matrix(gm);
    if (!tg_c || !tg_g) { std::cerr << "grad from_matrix failed\n"; return 1; }
    auto bc = attn_c.backward(cpu, *tg_c);
    auto bg = attn_g.backward(gpu, *tg_g);
    if (!bc || !bg)
    {
        std::cerr << "MHA backward FAILED: "
                  << (!bc ? bc.error().message : bg.error().message) << "\n";
        return 1;
    }
    auto bcm = cpu.to_matrix(*bc);
    auto bgm = gpu.to_matrix(*bg);
    if (!bcm || !bgm) { std::cerr << "backward to_matrix failed\n"; return 1; }
    nn::Scalar bmax = 0;
    for (std::size_t i = 0; i < bcm->size(); ++i)
    {
        const nn::Scalar d = std::fabs(bcm->span()[i] - bgm->span()[i]);
        if (d > bmax) bmax = d;
    }
    const bool bok = bmax <= 1e-3f;
    std::cout << "  MHA CPU/GPU backward 一致性: max_abs=" << bmax
              << (bok ? " ✅" : " ❌") << "\n";
    if (!bok) ++failures;

    return failures;
}

int main(int argc, char* argv[])
{
    int failures = 0;
    // 77 = 子测试因 GPU 不可用而跳过（ctest SKIP 约定），不计入失败
    const auto add = [&failures](int r) { if (r != 77) failures += r; };

    std::puts("=== attn_gradcheck (CausalSelfAttention numerical) ===");
    add(test_attn_gradcheck(argc, argv));

    std::puts("=== attn_consistency (forward vs forward_step) ===");
    add(test_attn_consistency(argc, argv));

    std::puts("=== attn_w_batch (multi-head multi-batch regression) ===");
    add(test_attn_w_batch(argc, argv));

    std::puts("=== mha_bidirectional (P-C2-7 fold mask regression pin) ===");
    failures += mha_bidirectional_probe();

    std::printf("\nattn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
