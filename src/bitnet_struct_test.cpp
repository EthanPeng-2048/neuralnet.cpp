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
