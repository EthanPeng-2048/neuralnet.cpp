// ── CNN 端到端冒烟 + 规格往返（LeNet 风格，Conv2D + MaxPool2D + Linear）────
//
// 目的：CNN 此前只有单层 Conv2D 参考比对（conv2d_gradcheck），**没有端到端覆盖**
//       ——最大池化的反向、以及整链 forward/backward/optimizer 只能靠手工跑
//       `mnist_train --arch cnn` 验证。本片段补上：
//         1. make_cnn_spec / build_cnn_model_from_spec / spec_matches /
//            cnn_config_from_spec 规格往返
//         2. 层组成核对（Conv2D / MaxPool2D / Linear 都真的在模型里）
//         3. 固定小批上训练若干步：loss 有限且下降、参数梯度有限非 NaN
//
// 用法：cnn_smoke_test [--gpu] [--steps N]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool cnn_all_finite(const nn::Matrix& m)
{
    for (std::size_t i = 0; i < m.span().size(); ++i)
        if (!std::isfinite(m.span()[i])) return false;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_gpu = false;
    std::size_t steps = 20;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--steps" && i + 1 < argc) steps = static_cast<std::size_t>(std::atoi(argv[++i]));
        else if (a == "--help")
        {
            std::cout << "用法: cnn_smoke_test [--gpu] [--steps N]\n";
            return 0;
        }
    }

    auto engine = nn::cli::create_engine(nn::cli::EngineConfig{use_gpu});
    if (!engine) { std::cerr << engine.error().message << "\n"; return 1; }
    nn::ComputeEngine& eng = **engine;

    std::cout << "========================================\n";
    std::cout << "  CNN 端到端冒烟（Conv2D + MaxPool2D + Linear）\n";
    std::cout << "========================================\n";

    // ── 1. 规格往返 ────────────────────────────────────────────────
    // 输入 12×12：conv3 → 10 → pool2 → 5 → conv3 → 3 → pool2 → 1（每层卷积后都池化）
    nn::CnnConfig cfg;
    cfg.in_channels = 1;
    cfg.in_size     = 12;
    cfg.pool        = 2;
    cfg.convs       = {{4, 3, 1, 0}, {8, 3, 1, 0}};   // {out_channels, kernel, stride, padding}
    cfg.fc_dims     = {16, 10};                        // flatten(8) → 16 → 10

    auto spec = nn::make_cnn_spec(cfg.in_channels, cfg.in_size, cfg.pool, cfg.convs, cfg.fc_dims);
    auto model_r = nn::build_cnn_model_from_spec(eng, spec);
    if (!model_r) { std::cerr << "build_cnn_model_from_spec 失败: " << model_r.error().message << "\n"; return 1; }
    nn::Model& model = *model_r;

    if (!model.spec() || !nn::spec_matches(spec, *model.spec()))
    {
        std::cerr << "CNN spec round-trip mismatch\n";
        return 1;
    }
    auto cfg2 = nn::cnn_config_from_spec(spec);
    if (!cfg2 ||
        cfg2->in_channels != cfg.in_channels || cfg2->in_size != cfg.in_size ||
        cfg2->pool != cfg.pool || cfg2->convs.size() != cfg.convs.size() ||
        cfg2->convs[1].out_channels != cfg.convs[1].out_channels ||
        cfg2->fc_dims != cfg.fc_dims)
    {
        std::cerr << "cnn_config_from_spec 往返不一致\n";
        return 1;
    }
    std::cout << "spec round-trip OK (" << nn::spec_summary(spec) << ")\n";

    // ── 2. 层组成核对 ──────────────────────────────────────────────
    std::size_t n_conv = 0, n_pool = 0, n_linear = 0;
    for (std::size_t i = 0; i < model.num_layers(); ++i)
    {
        nn::Layer& l = model.layer_at(i);
        if (dynamic_cast<nn::Conv2D*>(&l))    ++n_conv;
        if (dynamic_cast<nn::MaxPool2D*>(&l)) ++n_pool;
        if (dynamic_cast<nn::Linear*>(&l))    ++n_linear;
    }
    std::printf("layers=%zu  Conv2D=%zu  MaxPool2D=%zu  Linear=%zu\n",
                model.num_layers(), n_conv, n_pool, n_linear);
    if (n_conv != cfg.convs.size() || n_pool != cfg.convs.size() || n_linear < 2)
    {
        std::cerr << "层组成与配置不符\n";
        return 1;
    }

    // ── 3. 固定小批训练：loss 下降、梯度有限 ────────────────────────
    const std::size_t batch = 4;
    const std::size_t classes = cfg.fc_dims.back();
    std::mt19937_64 rng{20260919};
    std::uniform_real_distribution<nn::Scalar> dist(-1.0f, 1.0f);

    nn::Matrix xm(cfg.in_channels * cfg.in_size * cfg.in_size, batch);
    for (std::size_t i = 0; i < xm.size(); ++i) xm.span()[i] = dist(rng);
    std::vector<std::size_t> labels(batch);
    for (std::size_t i = 0; i < batch; ++i) labels[i] = static_cast<std::size_t>(rng() % classes);

    auto x = eng.from_matrix(xm);
    if (!x) { std::cerr << "from_matrix 失败\n"; return 1; }

    auto optimizer = nn::create_optimizer(
        "sgd", eng, model.parameters(), model.param_gradients(), nn::Scalar{0.1});
    if (!optimizer) { std::cerr << "optimizer 创建失败\n"; return 1; }

    nn::CrossEntropyLoss ce;
    nn::Scalar first_loss = 0, last_loss = 0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        auto zero = optimizer->zero_grad();
        if (!zero) { std::cerr << "zero_grad 失败\n"; return 1; }
        auto logits = model.forward(*x);
        if (!logits) { std::cerr << "forward 失败: " << logits.error().message << "\n"; return 1; }
        auto loss = ce.forward_sparse(eng, *logits, labels, {}, classes);
        if (!loss) { std::cerr << "loss 失败: " << loss.error().message << "\n"; return 1; }
        auto grad = ce.backward();
        if (!grad) { std::cerr << "CE backward 失败\n"; return 1; }
        auto b = model.backward(*grad);
        if (!b) { std::cerr << "model backward 失败: " << b.error().message << "\n"; return 1; }
        auto st = optimizer->step();
        if (!st) { std::cerr << "optimizer step 失败\n"; return 1; }
        if (step == 0) first_loss = *loss;
        last_loss = *loss;
        std::cout << "  step " << step << "  loss=" << *loss << "\n";
    }

    // 梯度有限性（任一参数出现 NaN/Inf 立即失败）
    auto g0 = eng.to_matrix(model.param_gradients()[0].get());
    if (!g0 || !cnn_all_finite(*g0))
    {
        std::cerr << "参数梯度含 NaN/Inf\n";
        return 1;
    }

    int failures = 0;
    if (!(first_loss > 0) || !std::isfinite(first_loss) || !std::isfinite(last_loss))
    {
        std::cerr << "loss 非法（NaN/Inf）\n";
        ++failures;
    }
    if (steps >= 2 && !(last_loss < first_loss))
    {
        std::cerr << "loss 未下降: first=" << first_loss << " last=" << last_loss << "\n";
        ++failures;
    }

    std::cout << "first_loss=" << first_loss << " last_loss=" << last_loss << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (failures == 0 ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return failures == 0 ? 0 : 1;
}
