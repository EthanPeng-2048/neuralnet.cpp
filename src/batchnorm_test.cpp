// ── BatchNorm 综合测试 ─────────────────────────────────────────────────────
//
// 覆盖（对应 roadmap P0-3「BatchNorm 禁止静默回落」的验收面）：
//   1. 工厂：make_norm_layer(NormType::BatchNorm) 返回**真** BatchNorm
//      （LayerNorm/RMSNorm 分支不受影响）
//   2. 训练态 forward vs 宿主参考（batch 统计归一化 + γ/β）
//   3. 推理态 forward vs 宿主参考（running 统计；含 batch=1）
//   4. running 统计 EMA 口径（多批逐元素对拍；推理态不更新）
//   5. gradcheck：训练态 γ / β / 输入 的中心差分数值梯度
//   6. 推理态 backward vs 宿主参考（grad_x = gy·γ·inv_std，无均值修正项）
//   7. Model 级：extra_state 收集（= 工厂没回落的证据）+ Model::set_training
//      转发 + save/load 往返参数与 running 统计逐位一致
//
// 用法: batchnorm_test [--gpu] [--tol <f>]   （--gpu 不可用 → 77 = skip）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "test_common.hpp"

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::TensorRef;
using nn::ComputeEngine;

int g_fail = 0;   // CHECK 计数器（test_common.hpp 约定）

namespace {

constexpr Scalar EPS = 1e-5f;
constexpr Scalar MOMENTUM = 0.1f;   // 与 BatchNorm 默认一致（新 batch 权重）

// ── 宿主参考：单次 forward（训练态用 batch 统计，推理态用 running 统计）──
Matrix ref_forward(bool training, const Matrix &x, const Matrix &gamma,
                   const Matrix &beta, const Matrix &run_mean,
                   const Matrix &run_var)
{
    const std::size_t F = x.rows(), B = x.cols();
    Matrix out(F, B);
    for (std::size_t f = 0; f < F; ++f)
    {
        Scalar mean = 0, var = 0;
        if (training)
        {
            for (std::size_t b = 0; b < B; ++b) mean += x.at_unchecked(f, b);
            mean /= static_cast<Scalar>(B);
            for (std::size_t b = 0; b < B; ++b)
            {
                const Scalar d = x.at_unchecked(f, b) - mean;
                var += d * d;
            }
            var /= static_cast<Scalar>(B);
        }
        else
        {
            mean = run_mean.at_unchecked(f, 0);
            var  = run_var.at_unchecked(f, 0);
        }
        const Scalar inv = 1.0f / std::sqrt(var + EPS);
        for (std::size_t b = 0; b < B; ++b)
            out.set_value_unchecked(f, b,
                gamma.at_unchecked(f, 0) * (x.at_unchecked(f, b) - mean) * inv
                + beta.at_unchecked(f, 0));
    }
    return out;
}

// 随机 (F,1) 列向量（宿主）
Matrix rand_col(std::size_t F, std::mt19937_64 &rng, Scalar lo, Scalar hi)
{
    Matrix m(F, 1);
    std::uniform_real_distribution<Scalar> dist(lo, hi);
    for (std::size_t i = 0; i < F; ++i) m.span()[i] = dist(rng);
    return m;
}

// 随机 (F,B)
Matrix rand_mat(std::size_t F, std::size_t B, std::mt19937_64 &rng)
{
    Matrix m(F, B);
    std::uniform_real_distribution<Scalar> dist(-1, 1);
    for (std::size_t i = 0; i < m.size(); ++i) m.span()[i] = dist(rng);
    return m;
}

bool matrices_equal(const Matrix &a, const Matrix &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const auto sa = a.span(), sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (sa[i] != sb[i]) return false;
    return true;
}

// ── gradcheck 辅助（rmsnorm_gradcheck 同款：中心差分）────────────────────
Scalar eval_loss(ComputeEngine &engine, nn::Layer &norm,
                 const Tensor &x, const Tensor &go)
{
    auto y = norm.forward(x);
    NN_ASSERT(y, "eval_loss: forward failed");
    auto y_m = engine.to_matrix(*y);
    auto go_m = engine.to_matrix(go);
    NN_ASSERT(y_m && go_m, "eval_loss: to_matrix failed");
    return dot(*y_m, *go_m);
}

bool check_grad_tensor(
    ComputeEngine &engine, nn::Layer &norm,
    const Tensor &fwd_input, const Tensor &go,
    Tensor &param, const Matrix &base, const Matrix &grad_analytical,
    const std::string &name, Scalar eps, Scalar tol)
{
    bool ok = true;
    std::size_t bad = 0;
    Scalar max_err{0};
    for (std::size_t r = 0; r < base.rows(); ++r)
    {
        for (std::size_t c = 0; c < base.cols(); ++c)
        {
            const Scalar orig = base.at_unchecked(r, c);

            Matrix pp = base;
            pp.set_value_unchecked(r, c, orig + eps);
            auto r_p = engine.write(param, pp.span());
            if (!r_p) { std::cerr << "write(+) failed\n"; return false; }
            const Scalar lp = eval_loss(engine, norm, fwd_input, go);

            Matrix pm = base;
            pm.set_value_unchecked(r, c, orig - eps);
            auto r_m = engine.write(param, pm.span());
            if (!r_m) { std::cerr << "write(-) failed\n"; return false; }
            const Scalar lm = eval_loss(engine, norm, fwd_input, go);

            auto r_r = engine.write(param, base.span());
            if (!r_r) { std::cerr << "write(restore) failed\n"; return false; }

            const Scalar num = (lp - lm) / (Scalar{2} * eps);
            const Scalar ana = grad_analytical.at_unchecked(r, c);
            const Scalar err = std::fabs(num - ana);
            if (err > max_err) max_err = err;
            if (!approx(num, ana, tol))
            {
                ok = false;
                if (bad < 10)
                    std::cout << "  [FAIL] " << name << "(" << r << "," << c
                              << ") num=" << num << " ana=" << ana << "\n";
                ++bad;
            }
        }
    }
    std::cout << "  " << name << ": " << (ok ? "OK" : "FAIL")
              << "  max_err=" << max_err << "  bad=" << bad << "\n";
    return ok;
}

// ── 1. 工厂分支 ──────────────────────────────────────────────────────────
void test_factory()
{
    std::cout << "\n── 1. 工厂分支（禁止静默回落）──\n";
    {
        auto l = nn::make_norm_layer(8, nn::NormType::BatchNorm);
        CHECK(l && dynamic_cast<nn::BatchNorm *>(l.get()) != nullptr,
              "make_norm_layer(BatchNorm) 应返回真 BatchNorm");
    }
    {
        auto l = nn::make_norm_layer(8, nn::NormType::LayerNorm);
        CHECK(l && dynamic_cast<nn::LayerNorm *>(l.get()) != nullptr,
              "make_norm_layer(LayerNorm) 应返回 LayerNorm");
    }
    {
        auto l = nn::make_norm_layer(8, nn::NormType::RMSNorm);
        CHECK(l && dynamic_cast<nn::RMSNorm *>(l.get()) != nullptr,
              "make_norm_layer(RMSNorm) 应返回 RMSNorm");
    }
}

// ── 2/3. 训练态 & 推理态 forward 对拍 ────────────────────────────────────
void test_forward(ComputeEngine &eng)
{
    std::cout << "\n── 2/3. 训练态 / 推理态 forward vs 宿主参考 ──\n";
    const std::size_t F = 8, B = 5;
    std::mt19937_64 rng(42);

    nn::BatchNorm bn(F);
    { auto r = bn.init(eng); CHECK(r, "BatchNorm init"); }

    // 随机 γ/β（覆盖参数广播路径）
    const Matrix g_m = rand_col(F, rng, 0.5f, 1.5f);
    const Matrix b_m = rand_col(F, rng, -0.5f, 0.5f);
    {
        auto params = bn.parameters();
        CHECK(eng.write(params[0].get(), g_m.span()), "write gamma");
        CHECK(eng.write(params[1].get(), b_m.span()), "write beta");
    }
    const Matrix x_m = rand_mat(F, B, rng);
    auto x = upload(eng, x_m);

    // ── 训练态：batch 统计归一化 ──
    bn.set_training(true);
    auto y = bn.forward(x);
    CHECK(y, "训练态 forward");
    if (y)
    {
        Matrix zeros(F, 1), ones(F, 1);
        for (std::size_t i = 0; i < F; ++i) ones.span()[i] = 1;
        const Matrix ref = ref_forward(true, x_m, g_m, b_m, zeros, ones);
        CHECK(close_to(download(eng, *y), ref, 2e-4f, "训练态 forward", 0),
              "训练态 forward 与宿主参考一致");
    }

    // ── 推理态：running 已被上面的训练态 forward 更新过 → 参考取实际值 ──
    bn.set_training(false);
    auto y_eval = bn.forward(x);
    CHECK(y_eval, "推理态 forward");
    if (y_eval)
    {
        auto extras = bn.extra_state();
        CHECK(extras.size() == 2, "extra_state 应为 {running_mean, running_var}");
        if (extras.size() == 2)
        {
            const Matrix rm = download(eng, extras[0].get());
            const Matrix rv = download(eng, extras[1].get());
            const Matrix ref = ref_forward(false, x_m, g_m, b_m, rm, rv);
            CHECK(close_to(download(eng, *y_eval), ref, 2e-4f,
                           "推理态 forward(running 更新后)", 1),
                  "推理态 forward 与宿主参考一致");
        }
    }

    // ── 推理态 batch=1（逐样本评估场景；训练态统计在此无意义）──
    {
        const Matrix x1 = rand_mat(F, 1, rng);
        auto t1 = upload(eng, x1);
        auto ye = bn.forward(t1);
        CHECK(ye, "推理态 forward batch=1");
        if (ye)
        {
            auto extras = bn.extra_state();
            CHECK(extras.size() == 2, "extra_state 应为 {running_mean, running_var}");
            if (extras.size() == 2)
            {
                const Matrix rm = download(eng, extras[0].get());
                const Matrix rv = download(eng, extras[1].get());
                const Matrix ref = ref_forward(false, x1, g_m, b_m, rm, rv);
                CHECK(close_to(download(eng, *ye), ref, 2e-4f,
                               "推理态 forward batch=1", 2),
                      "推理态 batch=1 与宿主参考一致");
            }
        }
    }
}

// ── 4. running 统计 EMA ──────────────────────────────────────────────────
void test_running_stats(ComputeEngine &eng)
{
    std::cout << "\n── 4. running 统计 EMA（多批）──\n";
    const std::size_t F = 6, B = 7;
    std::mt19937_64 rng(7);

    nn::BatchNorm bn(F);
    { auto r = bn.init(eng); CHECK(r, "BatchNorm init"); }

    Matrix run_mean(F, 1), run_var(F, 1);   // 宿主 EMA：初值 mean=0 / var=1
    for (std::size_t i = 0; i < F; ++i) run_var.span()[i] = 1;
    Matrix zeros(F, 1), ones(F, 1);
    for (std::size_t i = 0; i < F; ++i) ones.span()[i] = 1;

    bn.set_training(true);
    for (int step = 0; step < 4; ++step)
    {
        const Matrix x_m = rand_mat(F, B, rng);
        auto x = upload(eng, x_m);
        auto y = bn.forward(x);
        CHECK(y, "训练态 forward");
        // 宿主 EMA：running = 0.9*running + 0.1*batch 统计（有偏 1/B 口径）
        for (std::size_t f = 0; f < F; ++f)
        {
            Scalar mean = 0, var = 0;
            for (std::size_t b = 0; b < B; ++b) mean += x_m.at_unchecked(f, b);
            mean /= static_cast<Scalar>(B);
            for (std::size_t b = 0; b < B; ++b)
            {
                const Scalar d = x_m.at_unchecked(f, b) - mean;
                var += d * d;
            }
            var /= static_cast<Scalar>(B);
            run_mean.set_value_unchecked(f, 0,
                (1 - MOMENTUM) * run_mean.at_unchecked(f, 0) + MOMENTUM * mean);
            run_var.set_value_unchecked(f, 0,
                (1 - MOMENTUM) * run_var.at_unchecked(f, 0) + MOMENTUM * var);
        }
    }

    auto extras = bn.extra_state();
    CHECK(extras.size() == 2, "extra_state 应为 2 个张量");
    if (extras.size() == 2)
    {
        CHECK(close_to(download(eng, extras[0].get()), run_mean, 2e-4f,
                       "running_mean EMA", 3),
              "running_mean 与宿主 EMA 一致");
        CHECK(close_to(download(eng, extras[1].get()), run_var, 2e-4f,
                       "running_var EMA", 4),
              "running_var 与宿主 EMA 一致");
    }

    // ── 推理态 forward 不得更新 running 统计（逐位比较）──
    {
        const Matrix rm_before = download(eng, extras[0].get());
        const Matrix rv_before = download(eng, extras[1].get());
        bn.set_training(false);
        const Matrix x_m = rand_mat(F, 3, rng);
        auto x = upload(eng, x_m);
        auto y = bn.forward(x);
        CHECK(y, "推理态 forward");
        CHECK(matrices_equal(download(eng, extras[0].get()), rm_before),
              "推理态 forward 不更新 running_mean");
        CHECK(matrices_equal(download(eng, extras[1].get()), rv_before),
              "推理态 forward 不更新 running_var");
    }
}

// ── 5. gradcheck（训练态：γ / β / 输入）──────────────────────────────────
void test_gradcheck(ComputeEngine &eng, Scalar tol)
{
    std::cout << "\n── 5. gradcheck（训练态中心差分）──\n";
    const std::size_t F = 8, B = 6;
    std::mt19937_64 rng(123);

    nn::BatchNorm bn(F);
    { auto r = bn.init(eng); CHECK(r, "BatchNorm init"); }
    bn.set_training(true);

    const Matrix x_m = rand_mat(F, B, rng);
    const Matrix go_m = rand_mat(F, B, rng);
    auto x  = upload(eng, x_m);
    auto go = upload(eng, go_m);

    // 先 forward 一次填缓存，再 backward（铁律 #8 前置：缓存非空）
    auto y_fwd = bn.forward(x);
    CHECK(y_fwd, "gradcheck forward");
    for (auto &g : bn.param_gradients())
        CHECK(eng.zero(g.get()), "zero_grad");
    auto gx = bn.backward(go);
    CHECK(gx, "gradcheck backward");

    auto params = bn.parameters();        // [gamma, beta]
    auto grads  = bn.param_gradients();   // [grad_gamma, grad_beta]
    bool all_pass = true;
    const Scalar eps = 1e-3f;
    for (std::size_t pi = 0; pi < params.size(); ++pi)
    {
        auto p = eng.to_matrix(params[pi].get());
        auto g = eng.to_matrix(grads[pi].get());
        CHECK(p && g, "to_matrix(param/grad)");
        if (!p || !g) return;
        const std::string name = (pi == 0) ? "gamma" : "beta";
        all_pass &= check_grad_tensor(eng, bn, x, go, params[pi].get(),
                                      *p, *g, name, eps, tol);
    }
    // 输入梯度（扰动输入副本，前向用副本）
    {
        auto gxm = eng.to_matrix(*gx);
        CHECK(gxm, "to_matrix(grad_x)");
        if (!gxm) return;
        auto xt = eng.clone(x);
        CHECK(xt, "clone(x)");
        if (!xt) return;
        all_pass &= check_grad_tensor(eng, bn, *xt, go, *xt, x_m, *gxm,
                                      "grad_x", eps, tol);
    }
    CHECK(all_pass, "训练态 gradcheck 全部通过");
}

// ── 6. 推理态 backward vs 宿主参考 ───────────────────────────────────────
void test_eval_backward(ComputeEngine &eng)
{
    std::cout << "\n── 6. 推理态 backward vs 宿主参考 ──\n";
    const std::size_t F = 8, B = 5;
    std::mt19937_64 rng(99);

    nn::BatchNorm bn(F);
    { auto r = bn.init(eng); CHECK(r, "BatchNorm init"); }

    // 非平凡 running 统计：先跑两次训练态 forward（EMA 从 0/1 起步）
    bn.set_training(true);
    for (int i = 0; i < 2; ++i)
    {
        const Matrix xw = rand_mat(F, B, rng);
        auto tw = upload(eng, xw);
        auto yw = bn.forward(tw);
        CHECK(yw, "预热训练态 forward");
    }

    // 推理态 forward → backward
    bn.set_training(false);
    const Matrix x_m = rand_mat(F, B, rng);
    const Matrix go_m = rand_mat(F, B, rng);
    auto x  = upload(eng, x_m);
    auto go = upload(eng, go_m);
    auto y = bn.forward(x);
    CHECK(y, "推理态 forward");
    for (auto &g : bn.param_gradients())
        CHECK(eng.zero(g.get()), "zero_grad");
    auto gx = bn.backward(go);
    CHECK(gx, "推理态 backward");
    if (!y || !gx) return;

    // 宿主参考
    auto params = bn.parameters();
    const Matrix g_m = download(eng, params[0].get());
    const Matrix b_m = download(eng, params[1].get());
    auto extras = bn.extra_state();
    const Matrix rm = download(eng, extras[0].get());
    const Matrix rv = download(eng, extras[1].get());

    Matrix ref_out = ref_forward(false, x_m, g_m, b_m, rm, rv);
    CHECK(close_to(download(eng, *y), ref_out, 2e-4f, "推理态 forward", 5),
          "推理态 forward 与宿主参考一致");

    // grad_x = gy ⊙ γ ⊙ 1/√(running_var+ε)
    Matrix ref_gx(F, B);
    for (std::size_t f = 0; f < F; ++f)
    {
        const Scalar inv = 1.0f / std::sqrt(rv.at_unchecked(f, 0) + EPS);
        for (std::size_t b = 0; b < B; ++b)
            ref_gx.set_value_unchecked(f, b,
                go_m.at_unchecked(f, b) * g_m.at_unchecked(f, 0) * inv);
    }
    CHECK(close_to(download(eng, *gx), ref_gx, 2e-4f, "推理态 grad_x", 6),
          "推理态 grad_x 与宿主参考一致");

    // grad_gamma = Σ_b go⊙normalized, grad_beta = Σ_b go
    //（normalized = (x - running_mean)·inv_std）
    auto grads = bn.param_gradients();
    Matrix ref_gg(F, 1), ref_gb(F, 1);
    for (std::size_t f = 0; f < F; ++f)
    {
        const Scalar inv = 1.0f / std::sqrt(rv.at_unchecked(f, 0) + EPS);
        for (std::size_t b = 0; b < B; ++b)
        {
            const Scalar n = (x_m.at_unchecked(f, b) - rm.at_unchecked(f, 0)) * inv;
            ref_gg.set_value_unchecked(f, 0,
                ref_gg.at_unchecked(f, 0) + go_m.at_unchecked(f, b) * n);
            ref_gb.set_value_unchecked(f, 0,
                ref_gb.at_unchecked(f, 0) + go_m.at_unchecked(f, b));
        }
    }
    CHECK(close_to(download(eng, grads[0].get()), ref_gg, 2e-4f,
                   "推理态 grad_gamma", 7),
          "推理态 grad_gamma 与宿主参考一致");
    CHECK(close_to(download(eng, grads[1].get()), ref_gb, 2e-4f,
                   "推理态 grad_beta", 8),
          "推理态 grad_beta 与宿主参考一致");
}

// ── 7. Model 级：extra_state / set_training 转发 / save-load 往返 ────────
void test_model_level(ComputeEngine &eng)
{
    std::cout << "\n── 7. Model 级（extra_state / set_training / 序列化）──\n";
    const std::vector<std::size_t> dims = {12, 8, 4};
    const std::size_t n_norms = dims.size() - 2;   // 隐藏层归一化个数

    auto model_r = nn::build_mnist_mlp_model(eng, dims, nn::NormType::BatchNorm);
    CHECK(model_r, "build_mnist_mlp_model(BatchNorm)");
    if (!model_r) return;
    nn::Model &model = *model_r;

    // extra_state 非空 = 工厂真的造了 BatchNorm（LayerNorm/RMSNorm 为空）
    CHECK(model.extra_state().size() == 2 * n_norms,
          "Model::extra_state = 每个 BatchNorm 2 个 running 张量");

    // set_training 转发：定位 BatchNorm 层（层序 = Linear, BN, GeLU, Linear…）
    nn::BatchNorm *bn_layer = nullptr;
    for (std::size_t i = 0; i < model.num_layers(); ++i)
        if (auto *p = dynamic_cast<nn::BatchNorm *>(&model.layer_at(i)))
            bn_layer = p;
    CHECK(bn_layer != nullptr, "模型中应含 BatchNorm 层");
    if (!bn_layer) return;

    // 推理态（running 初值 0/1）：out = x·1/√(1+ε)·γ+β（γ=1、β=0 → 恒等缩放）
    // 训练态：每行（特征）均值 ≈ 0 —— 两态可观测差异即转发成功的证据。
    const std::size_t F = dims[1], B = 5;
    std::mt19937_64 rng(5);
    const Matrix x_m = rand_mat(F, B, rng);
    auto x = upload(eng, x_m);

    model.set_training(false);
    auto y_eval = bn_layer->forward(x);
    CHECK(y_eval, "转发后（推理态）BatchNorm forward");
    if (y_eval)
    {
        const Matrix got = download(eng, *y_eval);
        bool ok = true;
        for (std::size_t f = 0; f < F && ok; ++f)
            for (std::size_t b = 0; b < B && ok; ++b)
            {
                const Scalar ref = x_m.at_unchecked(f, b) / std::sqrt(1 + EPS);
                if (std::fabs(got.at_unchecked(f, b) - ref) > 2e-4f) ok = false;
            }
        CHECK(ok, "Model::set_training(false) 已转发（推理态 = 恒等缩放）");
    }

    model.set_training(true);
    auto y_train = bn_layer->forward(x);
    CHECK(y_train, "转发后（训练态）BatchNorm forward");
    if (y_train)
    {
        const Matrix got = download(eng, *y_train);
        bool ok = true;
        for (std::size_t f = 0; f < F && ok; ++f)
        {
            Scalar mean = 0;
            for (std::size_t b = 0; b < B; ++b) mean += got.at_unchecked(f, b);
            mean /= static_cast<Scalar>(B);
            if (std::fabs(mean) > 2e-4f) ok = false;
        }
        CHECK(ok, "Model::set_training(true) 已转发（训练态行均值≈0）");
    }

    // ── save/load 往返：参数 + running 统计逐位一致 ──
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = dims;
    spec.norm_type = nn::NormType::BatchNorm;

    const std::string file = "batchnorm_test_roundtrip.tmp";
    {
        auto r = nn::save_model(file, model, spec);
        CHECK(r, "save_model");
        if (!r) return;
    }
    auto model2_r = nn::build_mnist_mlp_model(eng, dims, nn::NormType::BatchNorm);
    CHECK(model2_r, "重建模型用于加载");
    if (!model2_r) return;
    auto lr = nn::load_model(file, *model2_r);
    CHECK(lr, "load_model");
    if (!lr) { std::remove(file.c_str()); return; }

    // 参数逐位一致
    {
        auto a = model.parameters(), b = model2_r->parameters();
        CHECK(a.size() == b.size(), "参数个数一致");
        bool same = a.size() == b.size();
        for (std::size_t i = 0; i < a.size() && same; ++i)
            same = matrices_equal(download(eng, a[i].get()),
                                  download(eng, b[i].get()));
        CHECK(same, "参数 save/load 逐位一致");
    }
    // running 统计逐位一致（extra_state 往返）
    {
        auto a = model.extra_state(), b = model2_r->extra_state();
        CHECK(a.size() == b.size() && a.size() == 2 * n_norms,
              "extra_state 个数一致");
        bool same = a.size() == b.size();
        for (std::size_t i = 0; i < a.size() && same; ++i)
            same = matrices_equal(download(eng, a[i].get()),
                                  download(eng, b[i].get()));
        CHECK(same, "running 统计 save/load 逐位一致");
    }
    std::remove(file.c_str());
}

} // namespace

int main(int argc, char *argv[])
{
    // --gpu 预检不可用 → 77（ctest SKIP 约定，与 cnn_test 同款）
    bool use_gpu = false;
    Scalar tol = 2e-2f;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--tol" && i + 1 < argc)
            tol = static_cast<Scalar>(std::atof(argv[++i]));
        else if (a == "--help")
        {
            std::cout << "用法: batchnorm_test [--gpu] [--tol <f>]\n";
            return 0;
        }
    }
    if (use_gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto probe = nn::cli::create_engine(ec);
        if (!probe)
        {
            std::cout << "GPU 不可用，跳过: " << probe.error().message << "\n";
            return 77;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res)
    {
        std::cerr << "引擎创建失败: " << engine_res.error().message << "\n";
        return 1;
    }
    auto engine = std::move(*engine_res);
    ComputeEngine &eng = *engine;

    std::cout << "========================================\n";
    std::cout << "  BatchNorm 测试 " << (use_gpu ? "(GPU)" : "(CPU)") << "\n";
    std::cout << "========================================\n";

    test_factory();
    test_forward(eng);
    test_running_stats(eng);
    test_gradcheck(eng, tol);
    test_eval_backward(eng);
    test_model_level(eng);

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (g_fail == 0 ? "✅ 全部通过" : "❌ 存在失败")
              << "  (fails=" << g_fail << ")\n";
    return g_fail == 0 ? 0 : 1;
}
