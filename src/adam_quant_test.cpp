// ── adam_quant_test — AdamWQuant（量化状态 AdamW，F5 第一步）───────────────
// 设计依据：docs/development/21-quantized-weights.md §4.13
//
// 覆盖：
//   [1] 状态账：状态字节 = 4 B/权重（+ 每行两个 f32 尺度），对比 f32 Adam 的 8 B/权重
//   [2] 尺度不变式：状态归一后在 [-1,1] / [0,1] 量级；尺度只前进（上界性质），
//       且**小量级行**（g 比大行小 1e3 倍）不被下溢吃掉（这正是"直接存 f16
//       状态"被 research/f16_weight_decay 否决的失效模式）
//   [3] 数值对齐：与 f32 AdamW 在同一问题上的 loss 曲线同量级（不是逐位相等：
//       状态量化必然带来差异，验收是"能训 + 同一量级"）
//   [4] 端到端：小 MLP 训练收敛（CPU + `--gpu`）+ 参数真的移动
//   [5] 边界：首步 g 全零不产生 NaN/Inf（尺度下限保护）
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_loss.hpp"
#include "neuralnet.cpp/compute_optimizer.hpp"
#include "neuralnet.cpp/domain_mnist.hpp"
#include "neuralnet.cpp/model_container.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

// ── [1] 状态字节账 ────────────────────────────────────────────────────────
void test_state_bytes(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[1] 状态账：4 B/权重 vs f32 Adam 的 8 B/权重（%s）\n", tag);
    nn::Tensor p1 = eng.create_tensor(64, 32, nn::Precision::F32, nn::InitSpec::zero());
    nn::Tensor g1 = eng.create_tensor(64, 32, nn::Precision::F32, nn::InitSpec::zero());
    nn::Tensor p2 = eng.create_tensor(64, 32, nn::Precision::F32, nn::InitSpec::zero());
    nn::Tensor g2 = eng.create_tensor(64, 32, nn::Precision::F32, nn::InitSpec::zero());
    CHECK(p1.valid() && g1.valid() && p2.valid() && g2.valid(), "建张量成功");

    nn::AdamWQuant q(eng, {p1}, {g1}, 0.01f);
    nn::AdamW a(eng, {p2}, {g2}, 0.01f);
    const std::size_t elems = 64u * 32u;
    const std::size_t q_state = q.state_bytes();
    const std::size_t a_state = elems * 4u * 2u;             // m/v 各 f32
    std::printf("    AdamWQuant 状态 = %zu B（%.2f B/权重）; f32 AdamW = %zu B（8.00 B/权重）\n",
                q_state, static_cast<double>(q_state) / static_cast<double>(elems),
                a_state);
    CHECK(q_state < a_state, "量化状态比 f32 状态小");
    // 上界：2 B/权重（f16 m）+ 2 B/权重（f16 v）+ 尺度（4 B/行 × 2 个）
    const std::size_t expect_max = elems * 4u + 64u * 4u * 2u;
    CHECK(q_state <= expect_max, "状态字节不超过 4 B/权重 + 尺度");
    // 比例 = 0.5 + 尺度占比（本例 cols=32 → 尺度占 0.25 B/权重 → 0.53；
    // cols=512 时尺度只占 0.016 B/权重 → 0.502）
    CHECK(static_cast<double>(q_state) <= 0.55 * static_cast<double>(a_state),
          "状态显存 ≤ f32 AdamW 的 55%");
}

// ── [2] 尺度不变式 + 小量级行不下溢 ──────────────────────────────────────
void test_scales(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[2] 尺度不变式（行归一 + 小量级行不下溢）（%s）\n", tag);
    constexpr std::size_t ROWS = 2, COLS = 128;
    nn::Tensor p = eng.create_tensor(ROWS, COLS, nn::Precision::F32, nn::InitSpec::zero());
    // 行 0 的梯度大 1e3 倍，行 1 小 → 直接 f16 存状态会把行 1 吃掉
    nn::Matrix gm(ROWS, COLS);
    for (std::size_t c = 0; c < COLS; ++c)
    {
        gm.set_value(0, c, 0.5f * static_cast<float>((c % 5) + 1));
        gm.set_value(1, c, 5e-4f * static_cast<float>((c % 3) + 1));
    }
    nn::Tensor g = upload(eng, gm);
    nn::AdamWQuant q(eng, {p}, {g}, 0.01f);

    std::vector<nn::Scalar> grad_buf(ROWS * COLS);
    auto dl = eng.read(g, std::span<nn::Scalar>(grad_buf));
    CHECK(dl.has_value(), "读回梯度");
    for (int step = 0; step < 20; ++step)
    {
        auto r = q.step();
        CHECK(r.has_value(), "step 成功");
        if (!r.has_value()) return;
    }
    // 状态有效（无 NaN/Inf）
    const nn::Matrix m16 = download(eng, q.moment_state()[0]);
    const nn::Matrix sm = download(eng, q.moment_scales()[0]);
    const nn::Matrix sv = download(eng, q.variance_scales()[0]);
    bool finite = true, normalized = true;
    for (std::size_t r = 0; r < ROWS; ++r)
    {
        if (!std::isfinite(sm.at(r, 0)) || !std::isfinite(sv.at(r, 0))) finite = false;
        if (!(sm.at(r, 0) > 0.0f) || !(sv.at(r, 0) > 0.0f)) finite = false;
        for (std::size_t c = 0; c < COLS; ++c)
        {
            const float mv = static_cast<float>(m16.at(r, c));
            if (!std::isfinite(mv)) finite = false;
            if (std::fabs(mv) > 8.0f) normalized = false;   // 量级应 ≈1
        }
    }
    CHECK(finite, "状态/尺度有限且为正");
    CHECK(normalized, "一阶矩状态量级 ≈ [-1,1]（行归一）");
    // 关键：小量级行（行 1，g≈5e-4）的状态**不是全零** → 没有下溢吃掉
    bool row1_nonzero = false;
    for (std::size_t c = 0; c < COLS; ++c)
        if (std::fabs(static_cast<float>(m16.at(1, c))) > 0.1f) row1_nonzero = true;
    CHECK(row1_nonzero, "小量级行的状态未被下溢吃掉（动态尺度生效）");
    // 尺度的量级应随行的梯度量级走（行 0 比行 1 大约 1e3 倍）
    CHECK(sm.at(0, 0) > 100.0f * sm.at(1, 0), "行尺度随梯度量级（行0 ≫ 行1）");
    std::printf("    scale_m: row0=%.3g row1=%.3g（比值 %.0f）\n",
                static_cast<double>(sm.at(0, 0)), static_cast<double>(sm.at(1, 0)),
                static_cast<double>(sm.at(0, 0) / sm.at(1, 0)));
}

// ── [3][4] 端到端：与 f32 AdamW 同量级收敛 + 参数移动 ────────────────────
bool train_mlp(nn::ComputeEngine& eng, const char* opt_name, int steps,
               nn::Scalar& loss_out, std::vector<nn::Scalar>& w_before_out,
               std::vector<nn::Scalar>& w_after_out)
{
    nn::PrecisionProfile prof;   // 默认 f32
    nn::Result<nn::Model> m_r = nn::build_mnist_mlp_model(
        eng, nn::MNIST_LAYER_DIMS, nn::NormType::LayerNorm, prof);
    if (!m_r) return false;
    nn::Model& model = *m_r;

    auto os = nn::create_optimizer(opt_name, eng, model.parameters(),
                                   model.param_gradients(), 0.02f, 0.0f, prof);
    if (!os) return false;
    std::unique_ptr<nn::Optimizer> opt = std::move(os);

    // 固定小数据集（合成）
    const std::size_t in_dim = nn::MNIST_LAYER_DIMS.front();
    const std::size_t B = 8, C = 10;
    nn::Matrix X(in_dim, B), Y(C, B);
    for (std::size_t k = 0; k < in_dim; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, ((k * 7 + b * 3) % 11) * 0.09f - 0.5f);
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t b = 0; b < B; ++b)
            Y.set_value(c, b, (c == (b % C)) ? 1.0f : 0.0f);
    nn::Tensor Xt = upload(eng, X), Yt = upload(eng, Y);

    // 记录首层权重的初值
    {
        auto w0 = eng.to_matrix(model.parameters()[0].get(), nn::Precision::F32);
        if (!w0) return false;
        w_before_out.assign(w0->span().begin(), w0->span().end());
    }

    nn::CrossEntropyLoss loss;
    nn::Scalar last = 0.0f;
    for (int s = 0; s < steps; ++s)
    {
        if (auto r = model.zero_grad(); !r) return false;
        auto pred = model.forward(Xt);
        if (!pred) { std::printf("    forward 失败: %s\n", pred.error().message.c_str()); return false; }
        std::vector<std::size_t> labels(B);
        for (std::size_t b = 0; b < B; ++b) labels[b] = b % C;
        std::vector<nn::Scalar> mask(B, 1.0f);
        auto ls = loss.forward_sparse(eng, *pred, labels, mask, C);
        if (!ls) { std::printf("    loss 失败: %s\n", ls.error().message.c_str()); return false; }
        last = *ls;
        if (!std::isfinite(last)) return false;
        auto gl = loss.backward();
        if (!gl) return false;
        auto bw = model.backward(*gl);
        if (!bw) return false;
        if (auto st = opt->step(); !st) return false;
    }
    loss_out = last;
    auto w1 = eng.to_matrix(model.parameters()[0].get(), nn::Precision::F32);
    if (!w1) return false;
    w_after_out.assign(w1->span().begin(), w1->span().end());
    return true;
}

void test_convergence(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[3][4] 端到端：AdamWQuant vs AdamW（同问题同超参）（%s）\n", tag);
    nn::Scalar loss_adam = 0.0f, loss_quant = 0.0f;
    std::vector<nn::Scalar> wb_a, wa_a, wb_q, wa_q;
    const bool ok_a = train_mlp(eng, "adamw", 120, loss_adam, wb_a, wa_a);
    const bool ok_q = train_mlp(eng, "adamwquant", 120, loss_quant, wb_q, wa_q);
    CHECK(ok_a, "f32 AdamW 训练跑通");
    CHECK(ok_q, "AdamWQuant 训练跑通");
    if (!ok_a || !ok_q) return;
    std::printf("    最终 loss: AdamW=%.6f  AdamWQuant=%.6f\n",
                static_cast<double>(loss_adam), static_cast<double>(loss_quant));
    CHECK(std::isfinite(loss_quant), "AdamWQuant 的 loss 有限");
    // 收敛量级：能显著下降，且与 f32 AdamW 同量级（不要求逐位：状态量化必然有差）
    CHECK(loss_quant < 0.5f, "AdamWQuant 收敛到小 loss");
    CHECK(std::fabs(static_cast<double>(loss_quant) - static_cast<double>(loss_adam)) < 0.5,
          "与 f32 AdamW 同量级");
    // 参数真的移动了（防"静默冻结"）
    double moved = 0.0;
    for (std::size_t i = 0; i < wa_q.size(); ++i)
        moved += std::fabs(static_cast<double>(wa_q[i]) - static_cast<double>(wb_q[i]));
    CHECK(moved > 1e-3, "AdamWQuant 下参数确实移动（非静默冻结）");
    std::printf("    首层权重 L1 位移: AdamW=%.4f  AdamWQuant=%.4f\n", [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < wa_a.size(); ++i)
            s += std::fabs(static_cast<double>(wa_a[i]) - static_cast<double>(wb_a[i]));
        return s;
    }(), moved);
}

// ── [5] 首步零梯度不产生 NaN ──────────────────────────────────────────────
void test_zero_grad_first_step(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[5] 首步全零梯度（尺度下限保护）（%s）\n", tag);
    nn::Tensor p = eng.create_tensor(4, 4, nn::Precision::F32,
                                     nn::InitSpec::uniform(-0.1f, 0.1f, nn::kInitSeed));
    nn::Tensor g = eng.create_tensor(4, 4, nn::Precision::F32, nn::InitSpec::zero());
    nn::AdamWQuant q(eng, {p}, {g}, 0.01f);
    auto r = q.step();
    CHECK(r.has_value(), "零梯度 step 成功");
    const nn::Matrix w = download(eng, p);
    const nn::Matrix sm = download(eng, q.moment_scales()[0]);
    bool finite = true;
    for (auto v : w.span())
        if (!std::isfinite(v)) finite = false;
    for (auto v : sm.span())
        if (!std::isfinite(v)) finite = false;
    CHECK(finite, "零梯度不产生 NaN/Inf（尺度下限）");
}

} // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        test_state_bytes(**eng_r, "GPU");
        test_scales(**eng_r, "GPU");
        test_zero_grad_first_step(**eng_r, "GPU");
        test_convergence(**eng_r, "GPU");
    }
    else
    {
        nn::CpuEngine eng;
        test_state_bytes(eng, "CPU");
        test_scales(eng, "CPU");
        test_zero_grad_first_step(eng, "CPU");
        test_convergence(eng, "CPU");
    }

    if (g_failures == 0)
        std::puts("adam_quant_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
