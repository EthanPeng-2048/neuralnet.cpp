// ── CNN 端到端冒烟 + 规格往返（LeNet 风格，Conv2D + MaxPool2D + Linear）────
//
// 目的：CNN 端到端验证。单层参考比对（conv2d_gradcheck / maxpool_gradcheck）
//       不覆盖规格序列化、层组成与整链 forward/backward/optimizer，本片段验证：
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
    NN_EXIT(engine, 1, "\n");
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
    NN_EXIT(model_r, 1, "build_cnn_model_from_spec 失败: ");
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

    // ── 2. 层组成核对（cfg 为默认构造 → CNN 规范形态 BatchNorm@conv）───
    std::size_t n_conv = 0, n_pool = 0, n_linear = 0, n_norm = 0;
    for (std::size_t i = 0; i < model.num_layers(); ++i)
    {
        nn::Layer& l = model.layer_at(i);
        if (dynamic_cast<nn::Conv2D*>(&l))    ++n_conv;
        if (dynamic_cast<nn::MaxPool2D*>(&l)) ++n_pool;
        if (dynamic_cast<nn::Linear*>(&l))    ++n_linear;
        if (dynamic_cast<nn::LayerNorm*>(&l) || dynamic_cast<nn::RMSNorm*>(&l) ||
            dynamic_cast<nn::BatchNorm*>(&l)) ++n_norm;
    }
    std::printf("layers=%zu  Conv2D=%zu  MaxPool2D=%zu  Linear=%zu  Norm=%zu\n",
                model.num_layers(), n_conv, n_pool, n_linear, n_norm);
    if (n_conv != cfg.convs.size() || n_pool != cfg.convs.size() || n_linear < 2)
    {
        std::cerr << "层组成与配置不符\n";
        return 1;
    }
    // 默认挂载回归锁：BatchNorm@conv → 每个卷积(+池化)后 1 个，FC 头不挂
    if (n_norm != cfg.convs.size() || cfg.norm_type != nn::NormType::BatchNorm ||
        cfg.norm_place != nn::NormPlace::Conv)
    {
        std::cerr << "CNN 默认应为 BatchNorm@conv（归一化层数不符）\n";
        return 1;
    }

    // ── 2b. 归一化挂载（NormPlace）：both → 卷积后各 1 + FC 隐藏层 1 ───
    {
        nn::CnnConfig ncfg = cfg;
        ncfg.norm_type  = nn::NormType::RMSNorm;
        ncfg.norm_place = nn::NormPlace::Both;

        auto nspec = nn::make_cnn_spec(ncfg.in_channels, ncfg.in_size, ncfg.pool,
                                       ncfg.convs, ncfg.fc_dims,
                                       ncfg.norm_type, ncfg.norm_place);
        auto nmodel_r = nn::build_cnn_model_from_spec(eng, nspec);
        NN_EXIT(nmodel_r, 1, "norm CNN 构建失败: ");
        nn::Model& nmodel = *nmodel_r;

        std::size_t n_norm = 0;
        for (std::size_t i = 0; i < nmodel.num_layers(); ++i)
            if (dynamic_cast<nn::RMSNorm*>(&nmodel.layer_at(i))) ++n_norm;
        const std::size_t want_norm = ncfg.convs.size() + 1;  // conv 后各 1 + FC 首隐藏层 1
        std::printf("norm_layers(RMSNorm)=%zu (期望 %zu)\n", n_norm, want_norm);
        if (n_norm != want_norm)
        {
            std::cerr << "norm_place=both 的归一化层数不符\n";
            return 1;
        }
        auto ncfg_back = nn::cnn_config_from_spec(nspec);
        if (!ncfg_back || ncfg_back->norm_place != nn::NormPlace::Both ||
            ncfg_back->norm_type != nn::NormType::RMSNorm)
        {
            std::cerr << "norm spec 往返不一致\n";
            return 1;
        }

        // 挂 norm 后跑一轮 fwd/bwd：梯度必须全有限
        const std::size_t nbatch = 4;
        nn::Matrix nxm(ncfg.in_channels * ncfg.in_size * ncfg.in_size, nbatch);
        std::mt19937_64 nrng{20261003};
        std::uniform_real_distribution<nn::Scalar> ndist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < nxm.size(); ++i) nxm.span()[i] = ndist(nrng);
        std::vector<std::size_t> nlabels(nbatch);
        for (std::size_t i = 0; i < nbatch; ++i)
            nlabels[i] = static_cast<std::size_t>(nrng() % ncfg.fc_dims.back());

        auto nx = eng.from_matrix(nxm);
        NN_EXIT(nx, 1, "norm from_matrix 失败\n");
        auto nlogits = nmodel.forward(*nx);
        NN_EXIT(nlogits, 1, "norm forward 失败: ");
        nn::CrossEntropyLoss nce;
        auto nloss = nce.forward_sparse(eng, *nlogits, nlabels, {},
                                        ncfg.fc_dims.back());
        NN_EXIT(nloss, 1, "norm loss 失败\n");
        auto ngrad = nce.backward();
        NN_EXIT(ngrad, 1, "norm CE backward 失败\n");
        auto nbwd = nmodel.backward(*ngrad);
        NN_EXIT(nbwd, 1, "norm model backward 失败: ");
        bool finite = true;
        for (auto& g : nmodel.param_gradients())
        {
            auto gm = eng.to_matrix(g.get());
            if (!gm) { finite = false; break; }
            for (auto v : gm->span())
                if (!std::isfinite(v)) { finite = false; break; }
            if (!finite) break;
        }
        NN_EXIT(finite, 1, "norm CNN 梯度含 NaN/Inf\n");
        std::cout << "norm_place=both OK (loss=" << *nloss << ")\n";
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
    NN_EXIT(x, 1, "from_matrix 失败\n");

    auto optimizer = nn::create_optimizer(
        "sgd", eng, model.parameters(), model.param_gradients(), nn::Scalar{0.1});
    NN_EXIT(optimizer, 1, "optimizer 创建失败\n");

    nn::CrossEntropyLoss ce;
    nn::Scalar first_loss = 0, last_loss = 0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        auto zero = optimizer->zero_grad();
        NN_EXIT(zero, 1, "zero_grad 失败\n");
        auto logits = model.forward(*x);
        NN_EXIT(logits, 1, "forward 失败: ");
        auto loss = ce.forward_sparse(eng, *logits, labels, {}, classes);
        NN_EXIT(loss, 1, "loss 失败: ");
        auto grad = ce.backward();
        NN_EXIT(grad, 1, "CE backward 失败\n");
        auto b = model.backward(*grad);
        NN_EXIT(b, 1, "model backward 失败: ");
        auto st = optimizer->step();
        NN_EXIT(st, 1, "optimizer step 失败\n");
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
