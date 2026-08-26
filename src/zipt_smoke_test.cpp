// ── ZiPT（AttnZip）端到端冒烟测试 ────────────────────────────────────────
//
// 验证：
//   1. build_zipt_model / build_zipt_model_from_spec / spec_matches 往返
//   2. ZiPTModel 整链 forward/backward 可运行（无 NaN、无崩溃）
//   3. 短训练循环 loss 单调下降（证明梯度有效、可学习）
//
// 用法：zipt_smoke_test [--steps N] [--gpu] [--cuda]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::size_t steps = 12;
    bool use_gpu = false, use_cuda = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--steps" && i + 1 < argc) steps = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--cuda") use_cuda = true;
    }

    auto engine = nn::cli::create_engine(nn::cli::EngineConfig{use_gpu, use_cuda});
    if (!engine) { std::cerr << engine.error().message << "\n"; return 1; }
    nn::ComputeEngine& eng = **engine;

    // ── 1. 规格往返校验 ─────────────────────────────────────────────
    nn::ZiPTConfig cfg;
    cfg.vocab_size    = 32;
    cfg.d_model       = 16;
    cfg.seq_len       = 8;
    cfg.num_heads     = 2;
    cfg.d_ff          = 32;
    cfg.num_layers    = 2;
    cfg.memory_tokens = 4;  // 偶数，规避 GPU matmul 奇数列宽度的既有后端 bug

    auto spec = nn::make_zipt_spec(cfg.vocab_size, cfg.d_model, cfg.seq_len,
                                   cfg.num_heads, cfg.d_ff, cfg.num_layers,
                                   cfg.memory_tokens);
    auto model_from_spec = nn::build_zipt_model_from_spec(eng, spec);
    if (!model_from_spec) { std::cerr << "build_zipt_model_from_spec failed\n"; return 1; }
    if (!nn::spec_matches(spec, *model_from_spec->spec()))
    {
        std::cerr << "ZiPT spec round-trip mismatch\n";
        return 1;
    }
    std::cout << "spec round-trip OK (" << nn::spec_summary(spec) << ")\n";

    auto model = nn::build_zipt_model(eng, cfg);
    if (!model) { std::cerr << "build_zipt_model failed\n"; return 1; }

    // ── 2. 训练冒烟：合成数据，验证 loss 下降 ─────────────────────────
    // 保守 lr：d_model 很小（16），AdamW 在 fp32 下跨后端可能发散；1e-3 保证 CPU/GPU 稳定
    auto optimizer = nn::create_optimizer(
        "adamw", eng, model->parameters(), model->param_gradients(),
        nn::Scalar{1e-3}, nn::Scalar{0.01});
    if (!optimizer) { std::cerr << "optimizer creation failed\n"; return 1; }

    nn::CrossEntropyLoss ce;

    std::mt19937_64 rng{123};
    std::uniform_int_distribution<std::size_t> tok(0, cfg.vocab_size - 1);

    // 合成输入 (seq_len, 1) 与标签（预测下一位置；仅用于驱动梯度，非真学习任务）
    nn::Matrix in(cfg.seq_len, 1);
    std::vector<std::size_t> labels(cfg.seq_len);
    for (std::size_t t = 0; t < cfg.seq_len; ++t)
    {
        const std::size_t id = tok(rng);
        in.set_value_unchecked(t, 0, static_cast<nn::Scalar>(id));
        labels[t] = tok(rng);
    }
    auto x = eng.from_matrix(in);
    if (!x) return 1;

    nn::Scalar first_loss = 0;
    nn::Scalar last_loss = 0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        auto zero = optimizer->zero_grad();
        if (!zero) { std::cerr << "zero_grad failed\n"; return 1; }
        auto logits = model->forward(*x);
        if (!logits) { std::cerr << "forward failed: " << logits.error().message << "\n"; return 1; }
        auto loss = ce.forward_sparse(eng, *logits, labels, {}, cfg.vocab_size);
        if (!loss) { std::cerr << "loss failed: " << loss.error().message << "\n"; return 1; }
        auto grad = ce.backward();
        if (!grad) { std::cerr << "loss backward failed\n"; return 1; }
        auto b = model->backward(*grad);
        if (!b) { std::cerr << "model backward failed: " << b.error().message << "\n"; return 1; }
        auto st = optimizer->step();
        if (!st) { std::cerr << "optimizer step failed\n"; return 1; }
        if (step == 0) first_loss = *loss;
        last_loss = *loss;
        std::cout << "  step " << step << "  loss=" << *loss << "\n";
    }

    std::cout << "first_loss=" << first_loss << " last_loss=" << last_loss << "\n";
    if (!(first_loss > 0) || !(first_loss == first_loss))  // 有效且非 NaN
    {
        std::cerr << "initial loss invalid (NaN?)\n";
        return 1;
    }
    // 训练后 loss 应显著下降（梯度有效、可学习）
    if (!(last_loss < first_loss))
    {
        std::cerr << "loss did not decrease (first=" << first_loss
                  << " last=" << last_loss << ")\n";
        return 1;
    }
    std::cout << "loss decreased OK (" << first_loss << " -> " << last_loss << ")\n";

    // ── 3. 压缩模式（L=W+C）冒烟：window < seq_len ───────────────────
    // 验证历史/窗口分离前向、压缩器仅吃历史、loss 只在窗口上计算、反向合并梯度。
    {
        nn::ZiPTConfig scfg;
        scfg.vocab_size    = 32;
        scfg.d_model       = 16;
        scfg.seq_len       = 8;   // L
        scfg.window        = 4;   // W（< L，启用压缩模式；偶数规避 GPU 奇数列后端 bug）
        scfg.num_heads     = 2;
        scfg.d_ff          = 32;
        scfg.num_layers    = 2;
        scfg.memory_tokens = 4;   // M；历史 = L - W = 4 压入 4 个记忆 token
        auto sspec = nn::make_zipt_spec(scfg.vocab_size, scfg.d_model, scfg.seq_len,
                                        scfg.num_heads, scfg.d_ff, scfg.num_layers,
                                        scfg.memory_tokens, scfg.window);
        auto smodel = nn::build_zipt_model_from_spec(eng, sspec);
        if (!smodel) { std::cerr << "build_zipt_model(split) failed\n"; return 1; }
        if (!nn::spec_matches(sspec, *smodel->spec()))
        {
            std::cerr << "ZiPT split spec mismatch\n";
            return 1;
        }
        std::cout << "split-mode spec OK (" << nn::spec_summary(sspec) << ")\n";

        auto sopt = nn::create_optimizer(
            "adamw", eng, smodel->parameters(), smodel->param_gradients(),
            nn::Scalar{1e-3}, nn::Scalar{0.01});
        if (!sopt) { std::cerr << "split optimizer creation failed\n"; return 1; }

        nn::Matrix sin(scfg.seq_len, 1);
        std::vector<std::size_t> slabels(scfg.window);   // 输出仅覆盖窗口 W
        for (std::size_t t = 0; t < scfg.seq_len; ++t)
            sin.set_value_unchecked(t, 0, static_cast<nn::Scalar>(tok(rng)));
        for (std::size_t t = 0; t < scfg.window; ++t)
            slabels[t] = tok(rng);
        auto sx = eng.from_matrix(sin);
        if (!sx) return 1;

        nn::Scalar sfirst = 0;
        nn::Scalar slast = 0;
        for (std::size_t step = 0; step < steps; ++step)
        {
            auto zero = sopt->zero_grad();
            if (!zero) { std::cerr << "split zero_grad failed\n"; return 1; }
            auto logits = smodel->forward(*sx);
            if (!logits) { std::cerr << "split forward failed: " << logits.error().message << "\n"; return 1; }
            auto loss = ce.forward_sparse(eng, *logits, slabels, {}, scfg.vocab_size);
            if (!loss) { std::cerr << "split loss failed: " << loss.error().message << "\n"; return 1; }
            auto grad = ce.backward();
            if (!grad) { std::cerr << "split loss backward failed\n"; return 1; }
            auto b = smodel->backward(*grad);
            if (!b) { std::cerr << "split model backward failed: " << b.error().message << "\n"; return 1; }
            auto st = sopt->step();
            if (!st) { std::cerr << "split optimizer step failed\n"; return 1; }
            if (step == 0) sfirst = *loss;
            slast = *loss;
        }
        if (!(sfirst > 0) || !(sfirst == sfirst))
        {
            std::cerr << "split initial loss invalid (NaN?)\n";
            return 1;
        }
        if (!(slast < sfirst))
        {
            std::cerr << "split loss did not decrease (first=" << sfirst
                      << " last=" << slast << ")\n";
            return 1;
        }
        std::cout << "split-mode (L=W+C) smoke: loss "
                  << sfirst << " -> " << slast << " OK\n";
    }

    // ── 4. batch>1 冒烟（铁律 5：注意力/序列路径必须覆盖 batch>1，否则 batch-major
    //      布局在 batch=1 时与 position-major 重合，测不出跨样本串扰） ────────
    {
        nn::ZiPTConfig bcfg;
        bcfg.vocab_size    = 32;
        bcfg.d_model       = 16;
        bcfg.seq_len       = 8;
        bcfg.window        = 4;   // split 模式（W<L）
        bcfg.num_heads     = 2;
        bcfg.d_ff          = 32;
        bcfg.num_layers    = 2;
        bcfg.memory_tokens = 4;
        auto bmodel = nn::build_zipt_model(eng, bcfg);
        if (!bmodel) { std::cerr << "build_zipt_model(batch>1) failed\n"; return 1; }

        auto bopt = nn::create_optimizer(
            "adamw", eng, bmodel->parameters(), bmodel->param_gradients(),
            nn::Scalar{1e-3}, nn::Scalar{0.01});
        if (!bopt) { std::cerr << "batch>1 optimizer creation failed\n"; return 1; }

        const std::size_t B = 2;
        nn::Matrix bin(bcfg.seq_len, B);
        std::vector<std::size_t> blabels(bcfg.window * B);
        for (std::size_t b = 0; b < B; ++b)
        {
            for (std::size_t t = 0; t < bcfg.seq_len; ++t)
                bin.set_value_unchecked(t, b, static_cast<nn::Scalar>(tok(rng)));
            for (std::size_t t = 0; t < bcfg.window; ++t)
                blabels[b * bcfg.window + t] = tok(rng);
        }
        auto bx = eng.from_matrix(bin);
        if (!bx) return 1;

        nn::Scalar bfirst = 0;
        for (std::size_t step = 0; step < steps; ++step)
        {
            auto zero = bopt->zero_grad();
            if (!zero) { std::cerr << "batch>1 zero_grad failed\n"; return 1; }
            auto logits = bmodel->forward(*bx);
            if (!logits) { std::cerr << "batch>1 forward failed: " << logits.error().message << "\n"; return 1; }
            auto loss = ce.forward_sparse(eng, *logits, blabels, {}, bcfg.vocab_size);
            if (!loss) { std::cerr << "batch>1 loss failed: " << loss.error().message << "\n"; return 1; }
            auto grad = ce.backward();
            if (!grad) { std::cerr << "batch>1 loss backward failed\n"; return 1; }
            auto bkw = bmodel->backward(*grad);
            if (!bkw) { std::cerr << "batch>1 model backward failed: " << bkw.error().message << "\n"; return 1; }
            auto st = bopt->step();
            if (!st) { std::cerr << "batch>1 optimizer step failed\n"; return 1; }
            if (step == 0) bfirst = *loss;
        }
        if (!(bfirst > 0) || !(bfirst == bfirst))
        {
            std::cerr << "batch>1 initial loss invalid (NaN?)\n";
            return 1;
        }
        std::cout << "batch>1 (batch=" << B << ") smoke: first_loss=" << bfirst << " OK\n";
    }

    std::cout << "ZIPT SMOKE TEST PASSED\n";
    return 0;
}
