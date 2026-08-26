// ── ZiPT（AttnZip）CPU/GPU 一致性测试 ────────────────────────────────────
//
// 构建 CPU 与 GPU 两个 ZiPT 模型，把 CPU 权重拷贝到 GPU（保证同构），
// 对同一输入比较：
//   1. forward logits 是否一致（最大绝对差）
//   2. 单步 backward 后各参数梯度的 L2 范数是否一致
//
// 用于定位 ZiPT 全模型在 GPU 上的布局/算子 bug（对比纯 fp32 数值噪声）。
// 用法：zipt_consistency_test [--steps N] [--tol <f>]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::size_t steps = 3;
    nn::Scalar tol = 1e-3f;        // forward logits 容差
    nn::Scalar grad_tol = 0.1f;    // 梯度 L2 范数容差（fp32 融合 kernel 积累噪声放宽）
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc) steps = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--tol" && i + 1 < argc) tol = static_cast<nn::Scalar>(std::atof(argv[++i]));
        else if (a == "--grad-tol" && i + 1 < argc) grad_tol = static_cast<nn::Scalar>(std::atof(argv[++i]));
    }

    // CPU 引擎（必选）
    auto cpu_engine = nn::cli::create_engine(nn::cli::EngineConfig{false, false});
    if (!cpu_engine) { std::cerr << cpu_engine.error().message << "\n"; return 1; }
    nn::ComputeEngine& cpu = **cpu_engine;

    // GPU 引擎（若不可用则跳过 GPU 比较）
    auto gpu_engine = nn::cli::create_engine(nn::cli::EngineConfig{true, false});
    if (!gpu_engine)
    {
        std::cout << "GPU unavailable, skipping GPU consistency check\n";
        return 0;
    }
    nn::ComputeEngine& gpu = **gpu_engine;

    nn::ZiPTConfig cfg;
    cfg.vocab_size    = 32;
    cfg.d_model       = 16;
    cfg.seq_len       = 8;
    cfg.num_heads     = 2;
    cfg.d_ff          = 32;
    cfg.num_layers    = 2;
    cfg.memory_tokens = 4;  // 偶数，规避 GPU matmul 奇数列宽度的既有后端 bug

    auto model_cpu = nn::build_zipt_model(cpu, cfg);
    if (!model_cpu) { std::cerr << "build CPU model failed\n"; return 1; }
    auto model_gpu = nn::build_zipt_model(gpu, cfg);
    if (!model_gpu) { std::cerr << "build GPU model failed\n"; return 1; }

    // 把 CPU 权重拷贝到 GPU，保证同构
    auto cpu_params = model_cpu->parameters();
    auto gpu_params = model_gpu->parameters();
    if (cpu_params.size() != gpu_params.size())
    {
        std::cerr << "param count mismatch\n";
        return 1;
    }
    for (std::size_t i = 0; i < cpu_params.size(); ++i)
    {
        auto m = cpu.to_matrix(cpu_params[i]);
        if (!m) return 1;
        if (!gpu.copy_from(gpu_params[i], *m)) return 1;
    }

    // 相同输入与标签
    std::mt19937_64 rng{42};
    std::uniform_int_distribution<std::size_t> tok(0, cfg.vocab_size - 1);
    nn::Matrix in(cfg.seq_len, 1);
    std::vector<std::size_t> labels(cfg.seq_len);
    for (std::size_t t = 0; t < cfg.seq_len; ++t)
    {
        const std::size_t id = tok(rng);
        in.set_value_unchecked(t, 0, static_cast<nn::Scalar>(id));
        labels[t] = tok(rng);
    }
    auto x_cpu = cpu.from_matrix(in);
    auto x_gpu = gpu.from_matrix(in);
    if (!x_cpu || !x_gpu) return 1;

    nn::CrossEntropyLoss ce_cpu, ce_gpu;
    auto opt_cpu = nn::create_optimizer("sgd", cpu, model_cpu->parameters(),
                                        model_cpu->param_gradients(), nn::Scalar{0});
    auto opt_gpu = nn::create_optimizer("sgd", gpu, model_gpu->parameters(),
                                        model_gpu->param_gradients(), nn::Scalar{0});

    bool ok = true;
    for (std::size_t step = 0; step < steps; ++step)
    {
        auto lc = model_cpu->forward(*x_cpu);
        auto lg = model_gpu->forward(*x_gpu);
        if (!lc || !lg) { std::cerr << "forward failed\n"; return 1; }
        auto lc_m = cpu.to_matrix(*lc);
        auto lg_m = gpu.to_matrix(*lg);
        if (!lc_m || !lg_m) return 1;
        nn::Scalar max_diff{0};
        for (std::size_t i = 0; i < lc_m->size(); ++i)
        {
            const nn::Scalar d = std::fabs(lc_m->span()[i] - lg_m->span()[i]);
            if (d > max_diff) max_diff = d;
        }
        const bool ok_logits = (max_diff <= tol);
        ok &= ok_logits;
        std::cout << "step " << step << "  logits_max_abs_diff=" << max_diff
                  << (ok_logits ? "  OK" : "  FAIL") << "\n";

        auto loss_r_c = ce_cpu.forward_sparse(cpu, *lc, labels, {}, cfg.vocab_size);
        auto loss_r_g = ce_gpu.forward_sparse(gpu, *lg, labels, {}, cfg.vocab_size);
        if (!loss_r_c || !loss_r_g) { std::cerr << "loss failed\n"; return 1; }
        std::cout << "  loss cpu=" << *loss_r_c << " gpu=" << *loss_r_g
                  << "  diff=" << std::fabs(*loss_r_c - *loss_r_g) << "\n";

        auto gc = ce_cpu.backward();
        auto gg = ce_gpu.backward();
        if (!gc || !gg) { std::cerr << "loss backward failed\n"; return 1; }
        if (!opt_cpu->zero_grad()) return 1;
        if (!opt_gpu->zero_grad()) return 1;
        auto bc = model_cpu->backward(*gc);
        auto bg = model_gpu->backward(*gg);
        if (!bc || !bg) { std::cerr << "model backward failed\n"; return 1; }

        // 比较各参数梯度：用「最大绝对差 / 全局最大梯度幅值」避免近零梯度范数放大
        auto cpu_grads = model_cpu->param_gradients();
        auto gpu_grads = model_gpu->param_gradients();
        nn::Scalar max_abs_diff{0};
        nn::Scalar global_max{0};
        for (std::size_t i = 0; i < cpu_grads.size(); ++i)
        {
            auto gm_c = cpu.to_matrix(cpu_grads[i]);
            auto gm_g = gpu.to_matrix(gpu_grads[i]);
            if (!gm_c || !gm_g) return 1;
            for (std::size_t k = 0; k < gm_c->size(); ++k)
            {
                const nn::Scalar d = std::fabs(gm_c->span()[k] - gm_g->span()[k]);
                if (d > max_abs_diff) max_abs_diff = d;
                if (std::fabs(gm_c->span()[k]) > global_max) global_max = std::fabs(gm_c->span()[k]);
            }
        }
        const nn::Scalar grad_rel = max_abs_diff / (global_max + nn::Scalar{1e-12f});
        const bool ok_grad = (grad_rel <= grad_tol);
        ok &= ok_grad;
        std::cout << "  grad_max_abs_diff=" << max_abs_diff
                  << "  grad_rel=" << grad_rel
                  << (ok_grad ? "  OK" : "  FAIL") << "\n";
    }

    std::cout << (ok ? "\nZIPT CONSISTENCY PASSED\n" : "\nZIPT CONSISTENCY FAILED\n");
    return ok ? 0 : 1;
}
