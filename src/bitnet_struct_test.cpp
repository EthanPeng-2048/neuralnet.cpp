// ── bitnet_struct_test — BitNet b1.58 2B4T 的结构开关（docs/development/22）──
// 设计依据：docs/development/22-bitnet-2b4t-architecture.md（§3 设计、§5 验收）
//
// 覆盖：
//   [1] SubLN 参数条数：on − off = layers × (d_model + d_ff)（RMSNorm）；
//       LayerNorm 下 = layers × 2×(d_model + d_ff)（证明类型跟随 norm_type）
//   [2] SubLN 在计算图里：子层 norm 参数拿到**非零梯度**（前向影响 + 反向参与）
//   [3] SubLN 改变前向：同一份权重下 on ≠ off（消融对照）
//   [4] 规格往返：subln 被 spec 记住 → 重建 → 前向一致
//   [5] spec_matches：subln 不同 → 不一致（防错载）
//   [6] 端到端小训练收敛（CPU；`--gpu` 时在 GPU 上跑）
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/cli/cli_engine_factory.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

// ── 参数元素总数 ──────────────────────────────────────────────────────────
std::size_t count_params(const std::vector<nn::TensorRef>& ps)
{
    std::size_t n = 0;
    for (const auto& p : ps) n += p.get().rows() * p.get().cols();
    return n;
}

double max_abs(const nn::Matrix& m)
{
    double r = 0.0;
    for (auto v : m.span()) r = std::max(r, std::fabs(static_cast<double>(v)));
    return r;
}

double max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e30;
    const auto sa = a.span();
    const auto sb = b.span();
    double r = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        r = std::max(r, std::fabs(static_cast<double>(sa[i]) - static_cast<double>(sb[i])));
    return r;
}

// ── [1] SubLN 参数条数 ────────────────────────────────────────────────────
void test_subln_param_count()
{
    std::puts("[1] SubLN 参数条数：on − off = layers × (d_model + d_ff)（RMSNorm）");

    nn::CpuEngine eng;
    const std::size_t V = 64, D = 32, S = 8, H = 4, FF = 64, L = 3;
    const auto P = nn::PosEncodingType::Learned;
    const auto A = nn::ActivationType::SwiGLU;

    auto build = [&](nn::NormType nt, bool subln) {
        auto m = std::make_unique<nn::GPTModel>(
            V, D, S, H, FF, L, P, A, nt, nn::PrecisionProfile{}, subln);
        NN_EXIT(m->init(eng), 1, "GPTModel init 失败: ");
        return m;
    };

    auto off = build(nn::NormType::RMSNorm, false);
    auto on  = build(nn::NormType::RMSNorm, true);
    const std::size_t n_off = count_params(off->parameters());
    const std::size_t n_on  = count_params(on->parameters());
    std::printf("    RMSNorm: off=%zu on=%zu Δ=%zu（期望 %zu）\n",
                n_off, n_on, n_on - n_off, L * (D + FF));
    CHECK(n_on == n_off + L * (D + FF), "RMSNorm 下 SubLN 增量 = layers×(d_model+d_ff)");

    // LayerNorm（gamma + beta 两个张量）→ 增量翻倍：证明类型跟随 norm_type
    auto off_ln = build(nn::NormType::LayerNorm, false);
    auto on_ln  = build(nn::NormType::LayerNorm, true);
    const std::size_t d_ln = count_params(on_ln->parameters()) -
                             count_params(off_ln->parameters());
    std::printf("    LayerNorm: Δ=%zu（期望 %zu）\n", d_ln, L * 2 * (D + FF));
    CHECK(d_ln == L * 2 * (D + FF), "LayerNorm 下 SubLN 增量 = layers×2×(d_model+d_ff)");

    // 默认关：不传 subln 与显式 false 完全同构
    nn::GPTModel def(V, D, S, H, FF, L, P, A, nn::NormType::RMSNorm);
    NN_EXIT(def.init(eng), 1, "GPTModel(default) init 失败: ");
    CHECK(count_params(def.parameters()) == n_off, "缺省 subln=false（旧布局逐位一致）");
}

// ── [2] SubLN 梯度接线：子层 norm 参数必须拿到非零梯度 ────────────────────
void test_subln_grad_wiring()
{
    std::puts("[2] SubLN 在计算图里：子层 norm 参数拿到非零梯度");

    nn::CpuEngine eng;
    const std::size_t D = 16, FF = 32, H = 4, S = 4, B = 2;

    // FeedForward：参数序 = [fc1.w, fc1.b, fc2.w, fc2.b, sub_norm.gamma]
    {
        nn::FeedForward ff(D, FF, nn::ActivationType::GeLU, /*subln=*/true,
                           nn::NormType::RMSNorm);
        NN_EXIT(ff.init(eng), 1, "FeedForward init 失败: ");
        auto ps = ff.parameters();
        CHECK(ps.size() == 5, "FF 参数条数 = 5（含 sub_norm.gamma）");
        CHECK(ps.size() == 5 && ps[4].get().rows() == FF && ps[4].get().cols() == 1,
              "FF sub_norm 宽度 = d_ff");

        auto x = make_tensor(eng, D, B, 0.1f, 0.03f);
        auto y = ff.forward(x);
        NN_EXIT(y, 1, "FeedForward forward 失败: ");
        CHECK(y->rows() == D && y->cols() == B, "FF 输出形状 (d_model, batch)");

        auto g = make_tensor(eng, D, B, 0.5f, -0.01f);
        auto gi = ff.backward(g);
        NN_EXIT(gi, 1, "FeedForward backward 失败: ");
        auto gs = ff.param_gradients();
        const double sg = max_abs(download(eng, gs[4].get()));
        std::printf("    FF sub_norm.gamma |grad|max = %.6g\n", sg);
        CHECK(sg > 0.0, "FF 的 SubLN 参与反向（梯度非零）");
    }

    // CausalSelfAttention：参数序 = [w_q, w_k, w_v, w_o, attn_sub_norm.gamma]
    {
        nn::CausalSelfAttention at(D, H, /*max_len=*/8, /*seq_len=*/S,
                                   nn::PosEncodingType::RoPE, /*subln=*/true,
                                   nn::NormType::RMSNorm);
        NN_EXIT(at.init(eng), 1, "CausalSelfAttention init 失败: ");
        auto ps = at.parameters();
        // 参数序 = [w_q.w, w_q.b, w_k.w, w_k.b, w_v.w, w_v.b, w_o.w, w_o.b,
        //           attn_sub_norm.gamma] —— SubLN 追加在**末位**
        const std::size_t k_sub = ps.size() - 1;
        CHECK(ps.size() == 9, "Attention 参数条数 = 9（4 投影 ×(w,b) + sub_norm.gamma）");
        CHECK(ps[k_sub].get().rows() == D && ps[k_sub].get().cols() == 1,
              "attn_sub_norm 宽度 = d_model");

        auto x = make_tensor(eng, D, S * B, 0.1f, 0.01f);
        auto y = at.forward(x);
        NN_EXIT(y, 1, "CausalSelfAttention forward 失败: ");
        CHECK(y->rows() == D && y->cols() == S * B, "Attention 输出形状");

        auto g = make_tensor(eng, D, S * B, 0.3f, -0.005f);
        auto gi = at.backward(g);
        NN_EXIT(gi, 1, "CausalSelfAttention backward 失败: ");
        auto gs = at.param_gradients();
        const double sg = max_abs(download(eng, gs[k_sub].get()));
        std::printf("    attn sub_norm.gamma |grad|max = %.6g\n", sg);
        CHECK(sg > 0.0, "Attention 的 SubLN 参与反向（梯度非零）");
    }
}

// ── [3] 消融：同一份权重下 SubLN 改变前向 ────────────────────────────────
void test_subln_changes_forward()
{
    std::puts("[3] SubLN 影响前向：同权重下 on ≠ off");

    nn::CpuEngine eng;
    const std::size_t D = 16, FF = 32, B = 3;

    nn::FeedForward off(D, FF, nn::ActivationType::GeLU, /*subln=*/false);
    NN_EXIT(off.init(eng), 1, "FeedForward(off) init 失败: ");
    nn::FeedForward on(D, FF, nn::ActivationType::GeLU, /*subln=*/true,
                       nn::NormType::RMSNorm);
    NN_EXIT(on.init(eng), 1, "FeedForward(on) init 失败: ");

    auto po = off.parameters();   // 4 张
    auto pn = on.parameters();    // 5 张（末位 sub_norm.gamma）
    CHECK(po.size() == 4 && pn.size() == 5, "参数条数 on/off");
    // 把 off 的 fc1/fc2 权重与偏置逐张量拷进 on（消除初始化差异 → 纯消融）
    for (std::size_t i = 0; i < 4; ++i)
    {
        const nn::Matrix w = download(eng, po[i].get());
        NN_EXIT(eng.copy_from(pn[i].get(), w), 1, "拷贝权重失败: ");
    }

    auto x = make_tensor(eng, D, B, 0.2f, 0.05f);
    auto yo = off.forward(x);
    NN_EXIT(yo, 1, "forward(off) 失败: ");
    auto yn = on.forward(x);
    NN_EXIT(yn, 1, "forward(on) 失败: ");
    const double d = max_abs_diff(download(eng, *yo), download(eng, *yn));
    std::printf("    同权重下 on/off 最大差 = %.6g\n", d);
    CHECK(d > 1e-3, "SubLN 确实改变前向输出（非静默失效）");

    // 反向消融：把 on 的 sub_norm.gamma 改成 0 → 中间激活整片归零 → fc2 偏置存活
    {
        nn::Matrix zero(FF, 1);
        for (auto& v : zero.span()) v = 0.0f;
        NN_EXIT(eng.copy_from(pn[4].get(), zero), 1, "清零 sub_norm.gamma 失败: ");
        auto y0 = on.forward(x);
        NN_EXIT(y0, 1, "forward(gamma=0) 失败: ");
        CHECK(max_abs_diff(download(eng, *yn), download(eng, *y0)) > 1e-3,
              "sub_norm.gamma 真的被前向读取（清零后输出改变）");
    }
}

// ── [4] 规格往返 ─────────────────────────────────────────────────────────
void test_subln_spec_roundtrip()
{
    std::puts("[4] 规格往返：subln 记忆 + 重建 + 前向一致");
    const std::string file = "bitnet_struct_subln_roundtrip.bin";

    const std::size_t V = 64, D = 32, S = 8, H = 4, FF = 64, L = 2, B = 2;
    nn::CpuEngine eng;

    nn::GptConfig cfg;
    cfg.vocab_size = V; cfg.d_model = D; cfg.seq_len = S;
    cfg.num_heads = H;  cfg.d_ff = FF;   cfg.num_layers = L;
    cfg.pos_enc = nn::PosEncodingType::Learned;
    cfg.activation = nn::ActivationType::SwiGLU;
    cfg.norm_type = nn::NormType::RMSNorm;
    cfg.subln = true;

    const nn::ModelSpec spec = nn::make_gpt_spec(
        V, D, S, H, FF, L, cfg.pos_enc, cfg.activation, cfg.norm_type, cfg.subln);
    CHECK(spec.subln, "make_gpt_spec 记录 subln");

    nn::Matrix x_m(S, B);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = static_cast<float>(i % V);

    nn::Matrix y_ref;
    {
        auto m = nn::build_gpt_model(eng, cfg);
        NN_EXIT(m, 1, "build_gpt_model 失败: ");
        auto y = m->forward(upload(eng, x_m));
        NN_EXIT(y, 1, "forward 失败: ");
        y_ref = download(eng, *y);
        NN_EXIT(nn::save_model(file, *m, spec), 1, "save_model 失败: ");
    }

    {
        auto peeked = nn::peek_model_spec(file);
        CHECK(peeked.has_value(), "peek_model_spec");
        if (!peeked.has_value())
        {
            std::printf("    peek 失败: %s\n", peeked.error().message.c_str());
            std::remove(file.c_str());
            return;
        }
        CHECK(peeked->subln, "规格记住 subln=true");

        auto m2 = nn::build_gpt_model_from_spec(eng, *peeked);
        NN_EXIT(m2, 1, "build_gpt_model_from_spec 失败: ");
        NN_EXIT(nn::load_model(file, *m2), 1, "load_model 失败: ");
        auto y = m2->forward(upload(eng, x_m));
        NN_EXIT(y, 1, "重建模型 forward 失败: ");
        const double d = max_abs_diff(y_ref, download(eng, *y));
        std::printf("    往返后前向最大差 = %.6g\n", d);
        CHECK(d == 0.0, "往返后前向逐位一致");
    }
    std::remove(file.c_str());
}

// ── [5] spec_matches：subln 是关键维度 ──────────────────────────────────
void test_subln_spec_matches()
{
    std::puts("[5] spec_matches：subln 不同 → 不一致");
    const auto base = [](bool subln) {
        return nn::make_gpt_spec(64, 32, 8, 4, 64, 2, nn::PosEncodingType::Learned,
                                 nn::ActivationType::SwiGLU, nn::NormType::RMSNorm, subln);
    };
    CHECK(!nn::spec_matches(base(false), base(true)), "subln 不同 → 不匹配");
    CHECK(nn::spec_matches(base(true), base(true)), "同配置 → 匹配");
    CHECK(!base(false).subln, "缺省 = false（旧文件语义）");
}

// ── S2/GQA 辅助：把 (n_kv·d_k, C) 的 KV 权重按**块映射**复制成 (n_head·d_k, C) ──
// 与实现同一约定：MHA 的第 h 头 = GQA 的第 (h / n_rep) 头。
nn::Matrix repeat_kv_rows(const nn::Matrix& src, std::size_t d_k,
                          std::size_t n_head, std::size_t n_kv)
{
    const std::size_t n_rep = n_head / n_kv;
    nn::Matrix out(n_head * d_k, src.cols());
    for (std::size_t h = 0; h < n_head; ++h)
        for (std::size_t r = 0; r < d_k; ++r)
            for (std::size_t c = 0; c < src.cols(); ++c)
                out.set_value(h * d_k + r, c, src.at((h / n_rep) * d_k + r, c));
    return out;
}

// ── [7] GQA 退化：n_head_kv == num_heads 与 MHA 逐位一致 ────────────────
void test_gqa_mha_equivalence()
{
    std::puts("[7] GQA 退化：n_head_kv == num_heads 与 MHA 逐位一致");
    nn::CpuEngine eng;
    const std::size_t D = 16, H = 4, S = 4, B = 2;

    // 声明式初始化是**确定性**的（kInitSeed）→ 同构两层的权重逐位相同
    nn::CausalSelfAttention mha(D, H, 8, S, nn::PosEncodingType::RoPE);
    NN_EXIT(mha.init(eng), 1, "MHA init 失败: ");
    nn::CausalSelfAttention gqa(D, H, 8, S, nn::PosEncodingType::RoPE,
                                /*subln=*/false, nn::NormType::LayerNorm,
                                /*n_head_kv=*/H);
    NN_EXIT(gqa.init(eng), 1, "GQA(=H) init 失败: ");

    auto pm = mha.parameters();
    auto pg = gqa.parameters();
    CHECK(pm.size() == pg.size(), "同构两层参数条数相同");
    if (pm.size() != pg.size()) return;
    // 初始化的种子按**创建序号**混流（同形状多层不互为镜像）→ 两次独立构造
    // 的权重并不相同；这里显式把 MHA 的 8 张参数拷进 GQA 侧，做成纯 A/B。
    for (std::size_t i = 0; i < pm.size(); ++i)
    {
        const nn::Matrix w = download(eng, pm[i].get());
        NN_EXIT(eng.copy_from(pg[i].get(), w), 1, "拷贝权重失败: ");
    }
    double wdiff = 0.0;
    for (std::size_t i = 0; i < pm.size(); ++i)
        wdiff = std::max(wdiff, max_abs_diff(download(eng, pm[i].get()),
                                             download(eng, pg[i].get())));
    CHECK(wdiff == 0.0, "显式拷贝后两层权重逐位相同");

    auto x = make_tensor(eng, D, S * B, 0.15f, 0.02f);
    auto ym = mha.forward(x);
    NN_EXIT(ym, 1, "MHA forward 失败: ");
    auto yg = gqa.forward(x);
    NN_EXIT(yg, 1, "GQA forward 失败: ");
    const double fd = max_abs_diff(download(eng, *ym), download(eng, *yg));
    std::printf("    forward 最大差 = %.6g\n", fd);
    CHECK(fd == 0.0, "n_head_kv=H 的前向与 MHA 逐位一致");

    auto g = make_tensor(eng, D, S * B, 0.25f, -0.004f);
    auto bm = mha.backward(g);
    NN_EXIT(bm, 1, "MHA backward 失败: ");
    auto bg = gqa.backward(g);
    NN_EXIT(bg, 1, "GQA backward 失败: ");
    const double bd = max_abs_diff(download(eng, *bm), download(eng, *bg));
    CHECK(bd == 0.0, "n_head_kv=H 的输入梯度与 MHA 逐位一致");
}

// ── [8] GQA 头映射：等价于"把 KV 权重按块映射复制成 MHA" ─────────────────
// 这条同时**钉住约定**：块映射 h/(H/n_kv)（若实现误用 h%n_kv，本用例必挂）
void test_gqa_head_mapping()
{
    std::puts("[8] GQA 头映射 = 块映射（与复制成 MHA 对拍）");
    nn::CpuEngine eng;
    const std::size_t D = 16, H = 4, S = 4, B = 2, n_kv = 2;
    const std::size_t d_k = D / H;

    nn::CausalSelfAttention gqa(D, H, 8, S, nn::PosEncodingType::RoPE,
                                /*subln=*/false, nn::NormType::LayerNorm, n_kv);
    NN_EXIT(gqa.init(eng), 1, "GQA init 失败: ");
    nn::CausalSelfAttention mha(D, H, 8, S, nn::PosEncodingType::RoPE);
    NN_EXIT(mha.init(eng), 1, "MHA init 失败: ");

    // 参数序（两者一致）：0 w_q.w / 1 w_q.b / 2 w_k.w / 3 w_k.b /
    //                     4 w_v.w / 5 w_v.b / 6 w_o.w / 7 w_o.b
    auto pg = gqa.parameters();
    auto pm = mha.parameters();
    CHECK(pg.size() == 8 && pm.size() == 8, "参数条数 8");
    if (pg.size() != 8 || pm.size() != 8) return;
    for (std::size_t i = 0; i < 8; ++i)
        std::printf("      param[%zu] gqa=%zux%zu mha=%zux%zu\n", i,
                    pg[i].get().rows(), pg[i].get().cols(),
                    pm[i].get().rows(), pm[i].get().cols());

    // Q/O 直接拷；K/V（w 与 b）按块映射复制行
    for (std::size_t i : {std::size_t{0}, std::size_t{1}, std::size_t{6}, std::size_t{7}})
        NN_EXIT(eng.copy_from(pm[i].get(), download(eng, pg[i].get())), 1, "拷贝 Q/O 失败: ");
    for (std::size_t i : {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5}})
    {
        const nn::Matrix Wg = download(eng, pg[i].get());
        NN_EXIT(eng.copy_from(pm[i].get(), repeat_kv_rows(Wg, d_k, H, n_kv)),
                1, "拷贝 K/V 失败: ");
    }

    auto x = make_tensor(eng, D, S * B, 0.15f, 0.02f);
    auto yg = gqa.forward(x);
    NN_EXIT(yg, 1, "GQA forward 失败: ");
    auto ym = mha.forward(x);
    NN_EXIT(ym, 1, "MHA(复制) forward 失败: ");
    const double fd = max_abs_diff(download(eng, *yg), download(eng, *ym));
    std::printf("    GQA vs 复制式 MHA：forward 最大差 = %.6g\n", fd);
    CHECK(fd < 1e-5, "前向等价（块映射）");

    auto g = make_tensor(eng, D, S * B, 0.2f, -0.003f);
    auto bg = gqa.backward(g);
    NN_EXIT(bg, 1, "GQA backward 失败: ");
    auto bm = mha.backward(g);
    NN_EXIT(bm, 1, "MHA(复制) backward 失败: ");
    const double bd = max_abs_diff(download(eng, *bg), download(eng, *bm));
    std::printf("    输入梯度最大差 = %.6g\n", bd);
    CHECK(bd < 1e-5, "输入梯度等价（scatter_add_rows 折叠正确）");

    // K/V 的权重梯度也必须等价（GQA 侧 = 复制式 MHA 侧对应行之和）
    for (std::size_t i : {std::size_t{2}, std::size_t{4}})
    {
        const nn::Matrix gg = download(eng, gqa.param_gradients()[i].get());
        const nn::Matrix gm_full = download(eng, mha.param_gradients()[i].get());
        const nn::Matrix gm_fold = [&] {
            nn::Matrix o(gg.rows(), gg.cols());
            const std::size_t n_rep = H / n_kv;
            for (std::size_t r = 0; r < gg.rows(); ++r)
                for (std::size_t c = 0; c < gg.cols(); ++c)
                {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < n_rep; ++j)
                        acc += gm_full.at((r / d_k * n_rep + j) * d_k + r % d_k, c);
                    o.set_value(r, c, static_cast<float>(acc));
                }
            return o;
        }();
        CHECK(max_abs_diff(gg, gm_fold) < 1e-5, "K/V 权重梯度 = 复制式 MHA 的组内和");
    }
}

// ── [9] GQA 规格往返 + spec_matches ────────────────────────────────────
void test_gqa_spec()
{
    std::puts("[9] GQA 规格往返 + spec_matches");
    const auto mk = [](std::size_t n_kv) {
        return nn::make_gpt_spec(64, 32, 8, 4, 64, 2, nn::PosEncodingType::Learned,
                                 nn::ActivationType::SwiGLU, nn::NormType::RMSNorm,
                                 false, n_kv);
    };
    CHECK(!mk(0).n_head_kv && mk(0).n_head_kv_or_heads() == 4, "0 = num_heads（旧语义）");
    CHECK(nn::spec_matches(mk(0), mk(4)), "0 与显式 num_heads 语义等价");
    CHECK(!nn::spec_matches(mk(2), mk(4)), "n_head_kv 不同 → 不匹配");
    CHECK(nn::spec_matches(mk(2), mk(2)), "同配置 → 匹配");

    const std::string file = "bitnet_struct_gqa_roundtrip.bin";
    const std::size_t V = 64, D = 32, S = 8, H = 4, FF = 64, L = 2, B = 2;
    nn::CpuEngine eng;
    nn::GptConfig cfg;
    cfg.vocab_size = V; cfg.d_model = D; cfg.seq_len = S;
    cfg.num_heads = H;  cfg.d_ff = FF;   cfg.num_layers = L;
    cfg.pos_enc = nn::PosEncodingType::Learned;
    cfg.activation = nn::ActivationType::SwiGLU;
    cfg.norm_type = nn::NormType::RMSNorm;
    cfg.subln = true;
    cfg.n_head_kv = 2;
    const nn::ModelSpec spec = nn::make_gpt_spec(
        V, D, S, H, FF, L, cfg.pos_enc, cfg.activation, cfg.norm_type,
        cfg.subln, cfg.n_head_kv);

    nn::Matrix x_m(S, B);
    for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = static_cast<float>(i % V);

    nn::Matrix y_ref;
    {
        auto m = nn::build_gpt_model(eng, cfg);
        NN_EXIT(m, 1, "build_gpt_model(GQA) 失败: ");
        auto y = m->forward(upload(eng, x_m));
        NN_EXIT(y, 1, "GQA forward 失败: ");
        y_ref = download(eng, *y);
        NN_EXIT(nn::save_model(file, *m, spec), 1, "save_model 失败: ");
    }
    {
        auto peeked = nn::peek_model_spec(file);
        CHECK(peeked.has_value() && peeked->n_head_kv == 2, "规格记住 n_head_kv=2");
        if (!peeked.has_value()) { std::remove(file.c_str()); return; }
        auto m2 = nn::build_gpt_model_from_spec(eng, *peeked);
        NN_EXIT(m2, 1, "build_gpt_model_from_spec(GQA) 失败: ");
        NN_EXIT(nn::load_model(file, *m2), 1, "load_model 失败: ");
        auto y = m2->forward(upload(eng, x_m));
        NN_EXIT(y, 1, "重建模型 forward 失败: ");
        const double d = max_abs_diff(y_ref, download(eng, *y));
        std::printf("    往返后前向最大差 = %.6g\n", d);
        CHECK(d == 0.0, "GQA 往返后前向逐位一致");
    }
    std::remove(file.c_str());
}

// ── [6] 端到端小训练（L2 损失：直接给 logits 梯度，避开 CE 依赖）──────────
bool test_training(nn::ComputeEngine& eng, const char* tag)
{
    const std::size_t V = 32, D = 32, S = 8, H = 4, FF = 64, L = 2, B = 2;
    const std::size_t N = S * B;
    std::printf("[6] SubLN 端到端小训练（%s，L2 on logits）\n", tag);

    nn::GPTModel model(V, D, S, H, FF, L, nn::PosEncodingType::Learned,
                       nn::ActivationType::SwiGLU, nn::NormType::RMSNorm,
                       nn::PrecisionProfile{}, /*subln=*/true);
    if (auto r = model.init(eng); !r)
    {
        std::printf("    init 失败: %s\n", r.error().message.c_str());
        return false;
    }

    auto opt_r = nn::create_optimizer("adamw", eng, model.parameters(),
                                      model.param_gradients(),
                                      nn::Scalar{5e-3}, nn::Scalar{0.0});
    if (!opt_r) { std::printf("    create_optimizer 失败\n"); return false; }
    nn::Optimizer& opt = *opt_r;

    // 固定输入 token ids（seq, batch）与固定目标 logits
    nn::Matrix x_m(S, B);
    for (std::size_t i = 0; i < x_m.size(); ++i)
        x_m.span()[i] = static_cast<float>((i * 7 + 3) % V);
    nn::Matrix tgt(V, N);
    {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& v : tgt.span()) v = d(rng);
    }
    const nn::Tensor x = upload(eng, x_m);

    constexpr int STEPS = 200;
    double first = 0.0, last = 0.0;
    for (int step = 0; step < STEPS; ++step)
    {
        if (auto r = opt.zero_grad(); !r)
        {
            std::printf("    zero_grad 失败: %s\n", r.error().message.c_str());
            return false;
        }
        auto y = model.forward(x);
        if (!y) { std::printf("    forward 失败: %s\n", y.error().message.c_str()); return false; }
        const nn::Matrix ym = download(eng, *y);

        nn::Matrix g(V, N);
        double loss = 0.0;
        for (std::size_t i = 0; i < V * N; ++i)
        {
            const double d = static_cast<double>(ym.span()[i]) -
                             static_cast<double>(tgt.span()[i]);
            loss += d * d;
            g.span()[i] = static_cast<float>(2.0 * d / static_cast<double>(V * N));
        }
        if (step == 0) first = loss;
        last = loss;

        auto gt = upload(eng, g);
        if (auto r = model.backward(gt); !r)
        {
            std::printf("    backward 失败: %s\n", r.error().message.c_str());
            return false;
        }
        if (auto r = opt.step(); !r)
        {
            std::printf("    step 失败: %s\n", r.error().message.c_str());
            return false;
        }
    }
    std::printf("    loss: %.6f → %.6f（%d 步）\n", first, last, STEPS);
    CHECK(last < first * 0.5f, "SubLN 模型端到端可训练（loss 显著下降）");
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    test_subln_param_count();
    test_subln_grad_wiring();
    test_subln_changes_forward();
    test_subln_spec_roundtrip();
    test_subln_spec_matches();
    test_gqa_mha_equivalence();
    test_gqa_head_mapping();
    test_gqa_spec();

    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        if (!test_training(**eng_r, "GPU")) g_failures++;
    }
    else
    {
        nn::CpuEngine eng;
        if (!test_training(eng, "CPU")) g_failures++;
    }

    if (g_failures == 0)
    {
        std::printf("\nbitnet_struct_test: ALL PASSED\n");
        return 0;
    }
    std::printf("\nbitnet_struct_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
