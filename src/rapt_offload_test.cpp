// ── rapt_offload_test.cpp — RAPT activation offload（L1-offload）一致性验证 ──
//
// 目的：验证 RAPTModel 开启 activation offload（set_activation_offload）后，
//       forward 输出与全部参数梯度与"全存基线"（offload 关闭）一致。
//
// 做法（GPU 引擎；同一模型实例，避免 Linear 的 thread_local rng 权重漂移）：
//   1. 构建一个小型 RAPTModel（SwiGLU + RMSNorm + RoPE），固定输入/输出梯度。
//   2. 基线（offload 关闭）：forward → 记 logits → zero → backward → 记梯度。
//   3. 开启 offload：重跑 forward（记 logits）→ zero → backward，与基线逐张量对比。
//
// 与 gpt_offload_test 的差异：RLA 的 backward 缓存比 GPT 多（RMSNorm 后 Q/K、
// 逐头 1/rms），且此前不在 activation_cache() 内——本测试覆盖补齐后的
// 导出/恢复集合是否完整（缺一项就会在 backward 报缓存缺失或数值不一致）。
//
// 用法：rapt_offload_test（无 Vulkan / 无可用设备 → 返回 77 = ctest SKIP）
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

using nn::Matrix;
using nn::Scalar;
using nn::Tensor;

#ifndef NN_HAS_VULKAN
int main()
{
    // 返回 77 = ctest SKIP：纯 CPU 构建无法执行 GPU 测试，不得计为 "Passed"
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持（NN_HAS_VULKAN）。\n";
    return 77;
}
#else
#include <neuralnet.cpp/backend/compute_vk_backend.hpp>
#include "test_common.hpp"

using nn::ActivationType;
using nn::GpuBackend;
using nn::GpuEngine;
using nn::NormType;
using nn::PosEncodingType;
using nn::RAPTModel;

namespace
{


int off_run_test()
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
        // 设备初始化失败属环境问题（非本测试断言失败）→ 跳过而非报红
        std::cout << "[SKIP] GPU 初始化失败: " << init_r.error().message << "\n";
        return 77;
    }
    GpuEngine eng(backend);

    std::cout << "========================================\n"
              << "  RAPT activation offload 一致性测试\n"
              << "========================================\n";
    std::cout << "  vocab=" << vocab << " d_model=" << d_model
              << " seq=" << seq_len << " batch=" << batch
              << " heads=" << num_heads << " d_ff=" << d_ff
              << " layers=" << num_layers << "\n";

    RAPTModel model(vocab, d_model, seq_len, num_heads, d_ff, num_layers,
                    PosEncodingType::RoPE,
                    ActivationType::SwiGLU,
                    NormType::RMSNorm);
    { auto r = model.init(eng); if (!r) { std::cerr << "RAPTModel init 失败: " << r.error().message << "\n"; return 1; } }

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
    std::cout << "  offload slab: " << (model.offload_ram_bytes() / 1024) << " KB\n";

    all_pass &= close_to(*lm, baseline_logits, tol, "logits", 0);

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
        all_pass &= close_to(*gm, baseline_grads[p], tol, name, p);
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部一致" : "❌ 存在不一致") << "\n";
    return all_pass ? 0 : 1;
}

} // namespace

int main()
{
    return off_run_test();
}
#endif