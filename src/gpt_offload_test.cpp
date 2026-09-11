// ── gpt_offload_test.cpp — activation offload（L1-offload）一致性验证 ─────
//
// 目的：验证 GPTModel 开启 activation offload（set_activation_offload）后，
//       forward 输出与全部参数梯度与“全存基线”（offload 关闭）一致。
//
// 做法（GPU 引擎；同一模型实例，避免 Linear 的 thread_local rng 权重漂移）：
//   1. 构建一个小型 GPTModel（SwiGLU + RMSNorm），固定输入/输出梯度。
//   2. 基线（offload 关闭）：forward → 记 logits → zero → backward → 记梯度。
//   3. 开启 offload：重跑 forward（记 logits）→ zero → backward，与基线逐张量对比。
//
// 用法：gpt_offload_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Matrix;
using nn::Scalar;
using nn::Tensor;

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持（NN_HAS_VULKAN）。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::ActivationType;
using nn::GPTModel;
using nn::GpuBackend;
using nn::GpuEngine;
using nn::NormType;
using nn::PosEncodingType;

namespace
{

bool close_to(const Matrix& a, const Matrix& b, Scalar tol,
              const std::string& name, std::size_t idx)
{
    Scalar max_abs = 0;
    Scalar max_rel = 0;
    const auto& sa = a.span();
    const auto& sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const Scalar diff = std::fabs(sa[i] - sb[i]);
        if (diff > max_abs) max_abs = diff;
        const Scalar denom = std::fabs(sb[i]) > 1e-30f ? std::fabs(sb[i]) : 1.0f;
        const Scalar rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
    }
    const bool pass = (max_abs <= tol) || (max_rel <= tol);
    std::cout << "    [" << idx << "] " << name
              << "  max_abs=" << max_abs << "  max_rel=" << max_rel
              << (pass ? "  ✅" : "  ❌") << "\n";
    return pass;
}

int run_test()
{
    const std::size_t vocab    = 257;
    const std::size_t d_model  = 64;
    const std::size_t seq_len  = 16;
    const std::size_t batch    = 2;
    const std::size_t num_heads = 4;
    const std::size_t d_ff     = 128;
    const std::size_t num_layers = 6;

    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    GpuEngine eng(backend);

    std::cout << "========================================\n"
              << "  GPT activation offload 一致性测试\n"
              << "========================================\n";
    std::cout << "  vocab=" << vocab << " d_model=" << d_model
              << " seq=" << seq_len << " batch=" << batch
              << " heads=" << num_heads << " d_ff=" << d_ff
              << " layers=" << num_layers << "\n";

    GPTModel model(eng, vocab, d_model, seq_len, num_heads, d_ff, num_layers,
                   PosEncodingType::Learned,
                   ActivationType::SwiGLU,
                   NormType::RMSNorm);

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    Matrix x_m(seq_len, batch);
    for (std::size_t i = 0; i < x_m.size(); ++i)
        x_m.span()[i] = static_cast<Scalar>(rng() % vocab);
    Matrix go_m(vocab, seq_len * batch);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix(x) failed\n"; return 1; }
    auto go = eng.from_matrix(go_m);
    if (!go) { std::cerr << "from_matrix(go) failed\n"; return 1; }

    // ── 基线（offload 关闭） ──
    Matrix baseline_logits;
    std::vector<Matrix> baseline_grads;
    {
        auto r = model.forward(eng, *x);
        if (!r) { std::cerr << "baseline forward failed: " << r.error().message << "\n"; return 1; }
        auto lm = eng.to_matrix(*r);
        if (!lm) { std::cerr << "to_matrix(logits) failed\n"; return 1; }
        baseline_logits = std::move(*lm);

        for (auto& g : model.param_gradients())
        {
            auto rz = eng.zero(g.get());
            if (!rz) { std::cerr << "baseline zero failed\n"; return 1; }
        }
        auto b = model.backward(eng, *go);
        if (!b) { std::cerr << "baseline backward failed: " << b.error().message << "\n"; return 1; }
        for (auto& g : model.param_gradients())
        {
            auto gm = eng.to_matrix(g.get());
            if (!gm) { std::cerr << "to_matrix(grad) failed\n"; return 1; }
            baseline_grads.push_back(std::move(*gm));
        }
    }

    const Scalar tol = 5e-4f;
    bool all_pass = true;

    // ── 开启 offload，重跑并对比 ──
    model.set_activation_offload(true);
    auto r = model.forward(eng, *x);
    if (!r) { std::cerr << "offload forward failed: " << r.error().message << "\n"; return 1; }
    auto lm = eng.to_matrix(*r);
    if (!lm) { std::cerr << "to_matrix(logits) failed\n"; return 1; }

    bool fwd_pass = close_to(*lm, baseline_logits, tol, "logits", 0);
    all_pass &= fwd_pass;

    for (auto& g : model.param_gradients())
    {
        auto rz = eng.zero(g.get());
        if (!rz) { std::cerr << "zero failed\n"; return 1; }
    }
    auto b = model.backward(eng, *go);
    if (!b) { std::cerr << "offload backward failed: " << b.error().message << "\n"; return 1; }

    const auto& grads = model.param_gradients();
    for (std::size_t p = 0; p < grads.size(); ++p)
    {
        auto gm = eng.to_matrix(grads[p].get());
        if (!gm) { std::cerr << "to_matrix(grad) failed\n"; return 1; }
        std::string name = "grad[" + std::to_string(p) + "]";
        bool ok = close_to(*gm, baseline_grads[p], tol, name, p);
        all_pass &= ok;
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部一致" : "❌ 存在不一致") << "\n";
    return all_pass ? 0 : 1;
}

} // namespace

int main()
{
    return run_test();
}
#endif
