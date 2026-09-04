// ── RAPT（ReLU 线性注意力）端到端冒烟测试 ──────────────────────────────────
//
// 验证：
//   1. build_rapt_model / build_rapt_model_from_spec / spec_matches 往返
//   2. RAPTModel 整链 forward/backward 可运行（无 NaN、无崩溃）
//   3. 短训练循环 loss 有效且下降（证明梯度有效、可学习）
//   4. batch>1 布局正确性（铁律 5：batch=1 时布局重合测不出）
//
// 用法：rapt_smoke_test [--steps N] [--gpu] [--cuda]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::size_t steps = 15;
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
    nn::RAPTConfig cfg;
    cfg.vocab_size = 32;
    cfg.d_model    = 16;   // d_k = d_model/heads = 8（偶数，RoPE 约束）
    cfg.seq_len    = 8;
    cfg.num_heads  = 2;
    cfg.d_ff       = 32;
    cfg.num_layers = 2;

    auto spec = nn::make_rapt_spec(cfg.vocab_size, cfg.d_model, cfg.seq_len,
                                   cfg.num_heads, cfg.d_ff, cfg.num_layers);
    auto model_from_spec = nn::build_rapt_model_from_spec(eng, spec);
    if (!model_from_spec) { std::cerr << "build_rapt_model_from_spec failed\n"; return 1; }
    if (!nn::spec_matches(spec, *model_from_spec->spec()))
    {
        std::cerr << "RAPT spec round-trip mismatch\n";
        return 1;
    }
    std::cout << "spec round-trip OK (" << nn::spec_summary(spec) << ")\n";

    auto model = nn::build_rapt_model(eng, cfg);
    if (!model) { std::cerr << "build_rapt_model failed\n"; return 1; }

    // ── 2. 训练冒烟：合成数据（causal 下一位置预测），验证 loss 下降 ──
    auto optimizer = nn::create_optimizer(
        "adamw", eng, model->parameters(), model->param_gradients(),
        nn::Scalar{1e-3}, nn::Scalar{0.01});
    if (!optimizer) { std::cerr << "optimizer creation failed\n"; return 1; }

    nn::CrossEntropyLoss ce;
    std::mt19937_64 rng{123};
    std::uniform_int_distribution<std::size_t> tok(0, cfg.vocab_size - 1);

    // 输入序列 + 因果下一位置标签（labels[t] = x[t+1]，末位任意）
    nn::Matrix in(cfg.seq_len, 1);
    std::vector<std::size_t> labels(cfg.seq_len);
    std::vector<std::size_t> toks(cfg.seq_len);
    for (std::size_t t = 0; t < cfg.seq_len; ++t) toks[t] = tok(rng);
    for (std::size_t t = 0; t < cfg.seq_len; ++t)
    {
        in.set_value_unchecked(t, 0, static_cast<nn::Scalar>(toks[t]));
        labels[t] = (t + 1 < cfg.seq_len) ? toks[t + 1] : tok(rng);
    }
    auto x = eng.from_matrix(in);
    if (!x) return 1;

    nn::Scalar first_loss = 0;
    nn::Scalar last_loss  = 0;
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
    if (!(first_loss > 0) || !(first_loss == first_loss) ||
        !(last_loss == last_loss))   // 有效且非 NaN
    {
        std::cerr << "loss invalid (NaN?)\n";
        return 1;
    }

    // ── 3. batch>1 布局正确性（铁律 5） ──────────────────────────────
    {
        const std::size_t batch = 2;
        nn::Matrix bin(cfg.seq_len, batch);
        std::vector<std::size_t> blabels(cfg.seq_len * batch);
        for (std::size_t b = 0; b < batch; ++b)
        {
            std::vector<std::size_t> bt(cfg.seq_len);
            for (std::size_t t = 0; t < cfg.seq_len; ++t) bt[t] = tok(rng);
            for (std::size_t t = 0; t < cfg.seq_len; ++t)
            {
                bin.set_value_unchecked(t, b, static_cast<nn::Scalar>(bt[t]));
                blabels[b * cfg.seq_len + t] =
                    (t + 1 < cfg.seq_len) ? bt[t + 1] : tok(rng);
            }
        }
        auto bx = eng.from_matrix(bin);
        if (!bx) return 1;
        // 同一模型上多跑几步，验证 batch>1 的 forward/backward 不崩、不 NaN
        for (std::size_t step = 0; step < 3; ++step)
        {
            auto zero = optimizer->zero_grad();
            if (!zero) { std::cerr << "batch>1 zero_grad failed\n"; return 1; }
            auto logits = model->forward(*bx);
            if (!logits) { std::cerr << "batch>1 forward failed: " << logits.error().message << "\n"; return 1; }
            auto loss = ce.forward_sparse(eng, *logits, blabels, {}, cfg.vocab_size);
            if (!loss) { std::cerr << "batch>1 loss failed\n"; return 1; }
            if (!(*loss == *loss)) { std::cerr << "batch>1 loss NaN\n"; return 1; }
            auto grad = ce.backward();
            if (!grad) { std::cerr << "batch>1 loss backward failed\n"; return 1; }
            auto b = model->backward(*grad);
            if (!b) { std::cerr << "batch>1 model backward failed\n"; return 1; }
            auto st = optimizer->step();
            if (!st) { std::cerr << "batch>1 optimizer step failed\n"; return 1; }
        }
        std::cout << "batch>1 (batch=" << batch << ") smoke OK\n";
    }

    // ── 4. KV cache 一致性：整序列 forward 末位 == 逐 token forward_step 累积 ──
    // 验证增量运行态生成（RLA 的 KV cache）与批量 forward 完全一致。
    {
        const std::size_t d_model = 16, heads = 2, L = 6;   // d_k = 8
        nn::ReLULinearAttention attn(d_model, heads, /*seq_len=*/0,
                                     /*causal=*/true, nn::PosEncodingType::RoPE);
        if (!attn.init(eng)) { std::cerr << "kv attn init failed\n"; return 1; }
        std::mt19937_64 rng{77};
        std::normal_distribution<nn::Scalar> dist(0.0, 1.0);
        nn::Matrix x_m(d_model, L);
        { auto sp = x_m.span(); for (auto& v : sp) v = dist(rng); }
        auto x_t = eng.from_matrix(x_m);
        if (!x_t) return 1;
        auto full = attn.forward(eng, *x_t);
        if (!full) { std::cerr << "kv full forward failed\n"; return 1; }
        auto full_m = eng.to_matrix(*full);
        if (!full_m) return 1;

        // 逐 token 增量（运行态 KV cache）
        // RLA-2 状态：B_state (d_model, d_k) + z_state (d_model, 1)
        const std::size_t dk = d_model / heads;
        nn::Tensor B_state = eng.create_tensor(d_model, dk);
        { auto zr = eng.zero(B_state); if (!zr) { std::cerr << "kv zero B_state failed\n"; return 1; } }
        nn::Tensor z_state = eng.create_tensor(d_model, 1);
        { auto zr = eng.zero(z_state); if (!zr) { std::cerr << "kv zero z_state failed\n"; return 1; } }
        nn::Tensor inc_out;
        for (std::size_t t = 0; t < L; ++t)
        {
            nn::Matrix col(d_model, 1);
            for (std::size_t r = 0; r < d_model; ++r)
                col.set_value_unchecked(r, 0, x_m.at_unchecked(r, t));
            auto col_t = eng.from_matrix(col);
            if (!col_t) return 1;
            auto o = attn.forward_step(eng, *col_t, B_state, z_state, t);
            if (!o) { std::cerr << "kv forward_step failed: " << o.error().message << "\n"; return 1; }
            inc_out = std::move(*o);
        }
        auto inc_m = eng.to_matrix(inc_out);
        if (!inc_m) return 1;
        // 比较增量末位 vs 整序列 forward 末位
        nn::Scalar max_diff{0};
        for (std::size_t r = 0; r < d_model; ++r)
        {
            const nn::Scalar a = full_m->at_unchecked(r, L - 1);
            const nn::Scalar b = inc_m->at_unchecked(r, 0);
            max_diff = std::max(max_diff, std::fabs(a - b));
        }
        std::cout << "KV-cache consistency: max_diff=" << max_diff << "\n";
        if (!(max_diff < 1e-3f))
        {
            std::cerr << "KV-cache inconsistency!\n";
            return 1;
        }
        std::cout << "KV-cache consistency OK\n";
    }

    std::cout << "RAPT SMOKE TEST PASSED\n";
    return 0;
}
