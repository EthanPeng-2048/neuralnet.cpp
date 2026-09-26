// ── rapt_checkpoint_test.cpp — RAPT 梯度检查点（激活重计算 L1）一致性测试 ──
//
// 目的：验证 RAPTModel 开启激活重计算（set_checkpoint_every）后，forward 输出
//       与全部参数梯度与"全存激活"基线（checkpoint 关闭）一致。
//
// 做法（同一模型实例，避免 Linear 的 thread_local rng 导致两次构建权重不同）：
//   1. 构建一个小型 RAPTModel（SwiGLU + RMSNorm + RoPE），固定输入。
//   2. 基线（stride=0）：forward → 记 logits → zero → backward → 记梯度快照。
//   3. 分别以 stride ∈ {1, 2}：重跑 forward（记 logits）→ zero → backward，
//      与基线逐张量对比。
//   4. 容差 tol：重计算路径与基线应在浮点重算误差范围内一致。
//
// 与 gpt_checkpoint_test 的差异：RLA 的 backward 缓存（Qp/Kp/V/Q_normed/
// K_normed/rms_inv）也必须由 forward_recompute 重建——本测试正是覆盖
// ReLULinearAttention 对 checkpoint_mode_ 的响应是否完整。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

using nn::ActivationType;
using nn::ComputeEngine;
using nn::NormType;
using nn::PosEncodingType;
using nn::RAPTModel;
using nn::Scalar;
using nn::Tensor;

namespace
{


int ckpt_run_test()
{
    const std::size_t vocab    = 257;
    const std::size_t d_model  = 64;
    const std::size_t seq_len  = 16;
    const std::size_t batch    = 2;
    const std::size_t num_heads = 4;
    const std::size_t d_ff     = 128;
    const std::size_t num_layers = 6;

    nn::CpuEngine eng;

    std::cout << "========================================\n";
    std::cout << "  RAPT 梯度检查点一致性测试 (L1)\n";
    std::cout << "========================================\n";
    std::cout << "  vocab=" << vocab << " d_model=" << d_model
              << " seq=" << seq_len << " batch=" << batch
              << " heads=" << num_heads << " d_ff=" << d_ff
              << " layers=" << num_layers << "\n";

    // RLA 约束：位置编码必须 RoPE
    RAPTModel model(vocab, d_model, seq_len, num_heads, d_ff, num_layers,
                    PosEncodingType::RoPE,
                    ActivationType::SwiGLU,
                    NormType::RMSNorm);
    { auto r = model.init(eng); if (!r) { std::cerr << "RAPTModel init 失败: " << r.error().message << "\n"; return 1; } }

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    // 固定输入：token IDs (seq, batch)
    nn::Matrix x_m(seq_len, batch);
    for (std::size_t i = 0; i < x_m.size(); ++i)
        x_m.span()[i] = static_cast<Scalar>(rng() % vocab);

    // 固定输出梯度 (vocab, seq*batch)
    nn::Matrix go_m(vocab, seq_len * batch);
    for (std::size_t i = 0; i < go_m.size(); ++i) go_m.span()[i] = dist(rng);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix(x) failed\n"; return 1; }
    auto go = eng.from_matrix(go_m);
    if (!go) { std::cerr << "from_matrix(go) failed\n"; return 1; }

    // ── 基线（stride=0，全存激活） ──
    nn::Matrix baseline_logits;
    std::vector<nn::Matrix> baseline_grads;
    {
        model.set_checkpoint_every(0);
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

    // 对每个 stride 做一致性验证（同一模型，切换 checkpoint 粒度）
    for (std::size_t stride : {std::size_t{1}, std::size_t{2}})
    {
        std::cout << "\n── 检查点 stride=" << stride << " ──\n";
        model.set_checkpoint_every(stride);
        auto r = model.forward(eng, *x);
        if (!r) { std::cerr << "checkpoint forward failed: " << r.error().message << "\n"; return 1; }
        auto lm = eng.to_matrix(*r);
        if (!lm) { std::cerr << "to_matrix(logits) failed\n"; return 1; }

        // 1) forward 输出对比
        all_pass &= close_to(*lm, baseline_logits, tol, "logits", 0);

        // 2) 梯度对比
        for (auto& g : model.param_gradients())
        {
            auto rz = eng.zero(g.get());
            if (!rz) { std::cerr << "zero failed\n"; return 1; }
        }
        auto b = model.backward(eng, *go);
        if (!b) { std::cerr << "checkpoint backward failed: " << b.error().message << "\n"; return 1; }

        const auto& grads = model.param_gradients();
        NN_ASSERT(grads.size() == baseline_grads.size(),
                  "rapt_checkpoint_test: grad count mismatch");
        for (std::size_t p = 0; p < grads.size(); ++p)
        {
            auto gm = eng.to_matrix(grads[p].get());
            if (!gm) { std::cerr << "to_matrix(grad) failed\n"; return 1; }
            std::string name = "grad[" + std::to_string(p) + "]";
            all_pass &= close_to(*gm, baseline_grads[p], tol, name, p);
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "  结果: " << (all_pass ? "✅ 全部一致" : "❌ 存在不一致") << "\n";
    return all_pass ? 0 : 1;
}

} // namespace

int main()
{
    return ckpt_run_test();
}