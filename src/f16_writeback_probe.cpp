// ── f16_writeback_probe：GPU f16 参数写回最小探针（issue #13 P0-②）────────
// 症状：MSVC 编译的 binary 上，GPU + param=f16 训练 loss 恒 = ln(V)（权重
//       冻结）；clang 同源码正常。CPU 引擎 f16 两边都正常。
// 定位：f16_precision_test 的 GPU 轨迹对拍 MSVC 打印 3.4776 → 3.4776（冻结）
//       而 clang 是 3.1822；AOT 产物逐字节相同 → 差异在 MSVC 编译的 host 侧。
// 本探针把「参数更新」缩到单表达式：
//   K3 模式 = dsl::compute_into(engine, leaf(p) + leaf(delta), p)
//   p 为 f16 存储（基类边界 cast 入口，原 PrecisionEngine 已下沉），逐步打印写入前后值。
// 期望：写后 p == p_before + delta。若不变 → 写回路径坏（host 侧）；
//       若变化 → 上层（optimizer 接线）问题。
// 用法：f16_writeback_probe [--gpu[=名称]]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Matrix;
using nn::Precision;
using nn::Scalar;

namespace {

int g_failures = 0;

void check(bool ok, const char* what)
{
    std::printf("  %s %s\n", ok ? "✅" : "❌", what);
    if (!ok) ++g_failures;
}

// f32→f16 cast 的舍入对拍：GPU 结果必须与宿主 nn::f16（float_to_half_bits，
// 明确 RNE）**逐位一致**。该路径经 cast.comp 的 f32→f16 转换，因此这条断言
// 同时锁死驱动 f32→f16 转换的舍入（OpFConvert 与 packHalf2x16 实测都向零
// 截断——后者见 Khronos Vulkan-Docs #1825）。
void probe_cast_rounding(nn::ComputeEngine& eng, const char* label)
{
    // 固定覆盖：正好在网格点下方/上方、RNE 与截断会分道扬镳的取值；
    // 再加 32 个确定性伪随机值（任意值都落在两个网格点之间，都能区分两种舍入）
    constexpr int N_FIXED = 8, N_RAND = 32, N = N_FIXED + N_RAND;
    std::vector<Scalar> vals(N);
    const Scalar fixed[N_FIXED] = {1.0f, 0.99997f, 1.00001f, 0.5f,
                                   0.077f, -0.077f, 0.036f, 2.0f};
    for (int i = 0; i < N_FIXED; ++i) vals[static_cast<std::size_t>(i)] = fixed[i];
    {
        std::mt19937 rng(20261004);
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (int i = N_FIXED; i < N; ++i) vals[static_cast<std::size_t>(i)] = d(rng);
    }
    Matrix m(1, static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) m.span()[static_cast<std::size_t>(i)] = vals[static_cast<std::size_t>(i)];

    std::printf("[%s] cast f32→f16 舍入对拍（cast.comp，%d 值）:\n", label, N);

    auto t32 = eng.from_matrix(m, Precision::F32);
    check(t32.has_value(), "cast 对拍：from_matrix(f32)");
    if (!t32) return;
    auto t16 = eng.cast(*t32, Precision::F16);
    check(t16.has_value(), "cast 对拍：engine.cast(f32→f16)");
    if (!t16) return;
    auto back = eng.to_matrix(*t16, Precision::F32);
    check(back.has_value(), "cast 对拍：to_matrix");
    if (!back) return;

    std::size_t n_diff = 0;
    for (int i = 0; i < N; ++i)
    {
        const Scalar v = vals[static_cast<std::size_t>(i)];
        const Scalar expect = static_cast<float>(nn::f16(v));   // 宿主 RNE
        const Scalar got = back->at(0, static_cast<std::size_t>(i));
        if (got != expect)
        {
            if (n_diff < 8)
                std::printf("    #%d f32=%.9g 期望(RNE)=%.9g 实际=%.9g\n",
                            i, v, expect, got);
            ++n_diff;
        }
    }
    std::printf("    cast f32→f16 与宿主 RNE 不一致 %zu/%d\n", n_diff, N);
    check(n_diff == 0, "cast f32→f16 与宿主 RNE 逐位一致（驱动转换舍入被锁死）");
}

// K3 模式：p += delta（p = f16 存储，delta 同精度物化）——text_train 冻结路径
// 的最小复刻。返回写入后 p 的下载值。
nn::Result<Matrix> inplace_add_writeback(nn::ComputeEngine& eng, Precision target_p)
{
    // 2×2 起始值全 1
    Matrix m(2, 2);
    for (auto& v : m.span()) v = 1.0f;

    auto p_r = eng.from_matrix(m, target_p);
    if (!p_r) return std::unexpected(p_r.error());
    auto p = std::move(*p_r);

    // delta 全 0.5，按 target_p 物化（对齐 optimizer 里 delta 精度 = p_.param）
    Matrix dm(2, 2);
    for (auto& v : dm.span()) v = 0.5f;
    auto d_r = eng.from_matrix(dm, target_p);
    if (!d_r) return std::unexpected(d_r.error());

    // K3：compute_into(leaf(p) + leaf(delta), p) —— 目标传递 in-place
    auto r = nn::dsl::compute_into(eng,
        nn::dsl::leaf(p) + nn::dsl::leaf(*d_r), p);
    if (!r) return std::unexpected(r.error());

    return eng.to_matrix(p);
}

// 对照：同样的 in-place 加，但目标 f32
void probe_inplace(nn::ComputeEngine& eng, const char* label, bool gpu)
{
    std::printf("[%s] in-place 写回（K3 模式 compute_into）:\n", label);

    auto r32 = inplace_add_writeback(eng, Precision::F32);
    check(r32.has_value(), "f32 目标：执行成功");
    if (r32)
        check(r32->at(0, 0) == 1.5f,
              ("f32 目标：值 1.0+0.5=1.5，实际 " +
               std::to_string(r32->at(0, 0))).c_str());

    auto r16 = inplace_add_writeback(eng, Precision::F16);
    check(r16.has_value(), "f16 目标：执行成功");
    if (r16)
    {
        const Scalar v = r16->at(0, 0);
        // f16 下 1.5 可精确表示 → 必须恰好 1.5
        check(v == 1.5f,
              ("f16 目标：值 1.0+0.5=1.5，实际 " + std::to_string(v)).c_str());
    }

    (void)gpu;
}

// optimizer 全链：小模型 + Adam + f16 profile，跑 N 步看参数是否变化。
// （复刻 text_train 冻结场景的上半段：梯度是否产生、参数是否移动）
void probe_optimizer_steps(nn::ComputeEngine& eng, const char* label, bool gpu)
{
    (void)gpu;  // 仅用于调用方标签区分，签名保持与 probe_inplace 对称
    std::printf("[%s] Adam + profile_f16 参数移动（8 步）:\n", label);

    nn::GptConfig cfg{};
    cfg.vocab_size = 32;
    cfg.d_model = 16;
    cfg.seq_len = 4;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_layers = 2;
    cfg.precision = nn::profile_f16();

    auto model_r = nn::build_gpt_model(eng, cfg);
    check(model_r.has_value(), "build_gpt_model(profile_f16)");
    if (!model_r) return;
    nn::Model& model = *model_r;

    // 固定初值（与 f16_precision_test 同法）：随机初值会让 f16/f32 对拍不可比
    {
        std::mt19937 rng(12345);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (auto& p : model.parameters())
        {
            nn::Matrix pm(p.get().rows(), p.get().cols());
            for (auto& v : pm.span()) v = nd(rng);
            if (auto r = eng.copy_from(p.get(), pm); !r)
            {
                check(false, "copy_from(固定初值)");
                return;
            }
        }
    }

    auto opt = nn::create_optimizer("adam", eng, model.parameters(),
        model.param_gradients(), /*lr=*/3e-3f, /*wd=*/0.0, cfg.precision);
    check(opt != nullptr, "create_optimizer(adam)");
    if (!opt) return;

    // 参数快照（f16 下载往返后比较）
    auto snapshot = [&]() -> std::vector<Scalar> {
        std::vector<Scalar> out;
        for (auto& t : model.parameters())
        {
            auto m = eng.to_matrix(t.get(), nn::Precision::F32);
            if (!m) { out.push_back(-1e30f); continue; }
            for (auto v : m->span()) out.push_back(v);
        }
        return out;
    };

    auto before = snapshot();
    check(!before.empty() && before[0] != -1e30f, "参数快照可下载");

    // 稠密 CE（与 train_tiny_gpt 相同的输入/目标构造）
    const std::size_t batch = 2;
    const std::size_t total = batch * cfg.seq_len;
    nn::Matrix x(cfg.seq_len, batch);
    for (std::size_t i = 0; i < cfg.seq_len * batch; ++i)
        x.span()[i] = static_cast<float>(i % cfg.vocab_size);
    nn::Matrix tgt(cfg.vocab_size, total, nn::Scalar{0});
    for (std::size_t i = 0; i < total; ++i)
        tgt.set_value(i % cfg.vocab_size, i, 1.0f);
    auto xt = eng.from_matrix(x);
    auto tt = eng.from_matrix(tgt);
    if (!xt || !tt) { check(false, "from_matrix(输入/目标)"); return; }

    nn::CrossEntropyLoss ce;
    ce.set_precision_profile(cfg.precision);

    Scalar first_loss = -1e30f, last_loss = -1e30f;
    for (int step = 0; step < 8; ++step)
    {
        if (auto r = model.zero_grad(); !r) { check(false, "zero_grad"); return; }
        auto logits = model.forward(*xt);
        if (!logits) { check(false, "model.forward"); return; }
        auto loss_r = ce.forward(eng, *logits, *tt);
        if (!loss_r) { check(false, "ce.forward"); return; }
        if (step == 0) first_loss = *loss_r;
        last_loss = *loss_r;
        auto grad = ce.backward();
        if (!grad) { check(false, "ce.backward"); return; }
        auto bwd = model.backward(*grad);
        if (!bwd) { check(false, "model.backward"); return; }
        auto st = opt->step();
        if (!st) { check(false, "optimizer.step"); return; }
    }

    auto after = snapshot();
    check(after.size() == before.size(), "快照大小一致");

    std::size_t n_changed = 0, n_total = 0;
    Scalar max_delta = 0.0f;
    for (std::size_t i = 0; i < before.size() && i < after.size(); ++i)
    {
        if (before[i] == -1e30f || after[i] == -1e30f) continue;
        ++n_total;
        const Scalar d = std::fabs(after[i] - before[i]);
        if (d > 0.0f) ++n_changed;
        if (d > max_delta) max_delta = d;
    }
    std::printf("    loss %.4f → %.4f（8 步）\n", first_loss, last_loss);
    std::printf("    参数变化: %zu/%zu，最大位移 %.6g\n",
                n_changed, n_total, max_delta);
    check(n_changed > 0, "8 步 Adam 后参数有移动（f16 GPU 写回生效）");
}

// ── AdamW 权重衰减写回（#x 路径）回归 ───────────────────────────────────
// 症状（research/f16_weight_decay/REPORT.md）：f16 参数下 AdamW 的
//   p *= (1 - lr·wd)
// 由 float16_t() 收窄（SPIR-V OpFConvert，舍入模式未被规范钉死；实测 NVIDIA
// 驱动按**向零截断**）。而 1-lr·wd 与 1 的差（lr·wd = 3e-5）远小于 0.5 个
// f16 ULP（≈2.4e-4）→ 正确舍入应是恒等，截断则每步恰掉 1 个网格步，数万步后
// 权重全部下溢为 0、loss 回到 ln(V)。
// 本探针把该路径缩小到可判定（三段，缺一不可）：
//   (a) f32 对照：500 步后应实测到 ≈N·lr·wd 的真实收缩——证明循环确实在跑；
//   (b) f16 表达式：500 步 decay 后必须**逐位不变**（RNE 舍回原网格点）；
//   (c) f16 + AdamW(wd>0) + **零梯度**：Adam 更新量恰为 0，唯一作用在权重上
//       的是 decay 写回 → 500 步后权重必须逐位不变（贴近报告的真实场景）。
void probe_decay_writeback(nn::ComputeEngine& eng, const char* label)
{
    constexpr int STEPS = 500;
    constexpr Scalar LR = 3e-4f, WD = 0.1f;
    const Scalar factor = Scalar{1} - LR * WD;   // 0.99997

    const Scalar init[6] = {1.0f, 0.5f, 0.077f, -0.077f, 0.036f, 2.0f};
    const auto make_init = [&]() {
        Matrix m(1, 6);
        for (int i = 0; i < 6; ++i) m.span()[i] = init[i];
        return m;
    };
    // 下载 f32 值序列（f16 自动升 cast）
    const auto snapshot = [&](const nn::Tensor& t) {
        std::vector<Scalar> out;
        auto m = eng.to_matrix(t, Precision::F32);
        if (!m) { out.assign(1, -1e30f); return out; }
        out.assign(m->span().begin(), m->span().end());
        return out;
    };
    // 逐位比较，打印最大相对变化；返回发生变化的元素数
    const auto compare_bits = [&](const std::vector<Scalar>& before,
                                  const std::vector<Scalar>& after,
                                  const char* what) {
        std::size_t n_diff = 0;
        Scalar max_rel = 0.0f;
        for (std::size_t i = 0; i < before.size() && i < after.size(); ++i)
        {
            if (before[i] == after[i]) continue;
            ++n_diff;
            const Scalar denom = std::fabs(before[i]) > 1e-30f
                               ? std::fabs(before[i]) : 1e-30f;
            const Scalar rel = std::fabs(after[i] - before[i]) / denom;
            if (rel > max_rel) max_rel = rel;
        }
        if (n_diff == 0)
            check(true, what);
        else
        {
            // 打印失败样例（首个变化元素 + 最大相对变化）便于定位
            for (std::size_t i = 0; i < before.size() && i < after.size(); ++i)
                if (before[i] != after[i])
                {
                    std::printf("    首个变化元素 #%zu: %.7g → %.7g\n",
                                i, before[i], after[i]);
                    break;
                }
            std::printf("    变化 %zu/%zu 元素，最大相对变化 %.4g\n",
                        n_diff, before.size(), max_rel);
            check(false, what);
        }
        return n_diff;
    };

    std::printf("[%s] AdamW 权重衰减写回（#x 路径，%d 步）:\n", label, STEPS);

    // ── (a) f32 对照：衰减必须真实生效（否则 (b)/(c) 的"不变"是假阳性）──
    {
        auto p_r = eng.from_matrix(make_init(), Precision::F32);
        check(p_r.has_value(), "f32 对照：from_matrix");
        if (!p_r) return;
        auto p = std::move(*p_r);
        for (int s = 0; s < STEPS; ++s)
        {
            auto r = nn::dsl::compute_into(eng,
                nn::dsl::leaf(p) * nn::dsl::rparam(factor), p);
            if (!r) { check(false, "f32 对照：decay compute_into"); return; }
        }
        auto m_after = eng.to_matrix(p);
        if (!m_after) { check(false, "f32 对照：to_matrix"); return; }
        // 理论收缩 = 1 - factor^N ≈ N·lr·wd = 1.49%（N=500）
        const Scalar rel = 1.0f - m_after->at(0, 0) / init[0];
        const bool ok = (rel > 0.005f && rel < 0.03f);
        std::printf("    f32 实际相对收缩 %.4f%%（理论 %.4f%%）\n",
                    static_cast<double>(rel) * 100.0,
                    (1.0 - std::pow(static_cast<double>(factor), STEPS)) * 100.0);
        check(ok, "f32 对照：500 步 decay 收缩 ≈ N·lr·wd（证明循环在跑）");
    }

    // ── (b) f16 表达式级：500 步后必须逐位不变 ──────────────────────────
    {
        auto p_r = eng.from_matrix(make_init(), Precision::F16);
        check(p_r.has_value(), "f16 表达式：from_matrix");
        if (!p_r) return;
        auto p = std::move(*p_r);
        const auto before = snapshot(p);
        check(before.size() == 6 && before[0] != -1e30f, "f16 表达式：快照可下载");
        for (int s = 0; s < STEPS; ++s)
        {
            auto r = nn::dsl::compute_into(eng,
                nn::dsl::leaf(p) * nn::dsl::rparam(factor), p);
            if (!r) { check(false, "f16 表达式：decay compute_into"); return; }
        }
        const auto after = snapshot(p);
        compare_bits(before, after,
            "f16 表达式：500 步 decay 写回逐位不变（RNE 舍回原网格点）");
    }

    // ── (c) AdamW(wd>0) + 零梯度：唯一作用在权重上的是 decay 写回 ───────
    {
        auto p_r = eng.from_matrix(make_init(), Precision::F16);
        check(p_r.has_value(), "AdamW decay：from_matrix(p)");
        if (!p_r) return;
        Matrix gz(1, 6);
        for (auto& v : gz.span()) v = 0.0f;   // 零梯度 → Adam 更新量恰为 0
        auto g_r = eng.from_matrix(gz, Precision::F16);
        check(g_r.has_value(), "AdamW decay：from_matrix(g=0)");
        if (!g_r) return;

        std::vector<nn::TensorRef> params{std::ref(*p_r)};
        std::vector<nn::TensorRef> grads{std::ref(*g_r)};
        const nn::PrecisionProfile prof{Precision::F16, Precision::F32,
                                        Precision::F32, Precision::F32};
        auto opt = nn::create_optimizer("adamw", eng, params, grads, LR, WD, prof);
        check(opt != nullptr, "AdamW decay：create_optimizer(adamw, wd=0.1)");
        if (!opt) return;

        const auto before = snapshot(*p_r);
        check(before.size() == 6 && before[0] != -1e30f, "AdamW decay：快照可下载");
        for (int s = 0; s < STEPS; ++s)
        {
            auto r = opt->step();
            if (!r) { check(false, "AdamW decay：optimizer.step"); return; }
        }
        const auto after = snapshot(*p_r);
        compare_bits(before, after,
            "f16 + AdamW(wd>0) 零梯度：500 步后权重逐位不变（decay 写回 RNE）");
    }
}

} // namespace

int main(int argc, char* argv[])
{
    // stdout 无缓冲：与 stderr（trace 打印）在合并捕获下保持正确时序，
    // 否则管道捕获时 stdout 块缓冲 → 段标题与 [into] 分支行错位无法归因。
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string gpu_device;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (auto dev = nn::cli::parse_gpu_option(argc, argv, i, true))
            gpu_device = *dev;
        else if (arg.rfind("--gpu", 0) == 0)
        {
            // --gpu 无值 = 自动选卡
            if (arg == "--gpu") gpu_device.clear();
        }
        else { std::fprintf(stderr, "未知参数: %s\n", arg.c_str()); return 2; }
    }

    std::printf("f16_writeback_probe（issue #13 P0-②：GPU f16 权重冻结定位）\n");

    // CPU 对照（始终跑）
    {
        nn::CpuEngine cpu;
        nn::ComputeEngine& padapter = cpu;   // P-1：原 PrecisionEngine 已下沉
        std::printf("── CPU ──\n");
        probe_cast_rounding(padapter, "cpu");
        probe_inplace(padapter, "cpu", false);
        probe_optimizer_steps(padapter, "cpu", false);
        probe_decay_writeback(padapter, "cpu");
    }

#ifdef NN_HAS_VULKAN
    {
        auto& backend = nn::GpuBackend::instance();
        auto init_r = backend.initialize(gpu_device);
        if (!init_r)
        {
            std::fprintf(stderr, "GPU 初始化失败: %s\n", init_r.error().message.c_str());
            return 77;  // ctest SKIP
        }
        nn::GpuEngine gpu(backend);
        nn::ComputeEngine& adapter = gpu;   // P-1：原 PrecisionEngine 已下沉
        std::printf("── GPU（%s）──\n", backend.device().device_name().c_str());
        probe_cast_rounding(adapter, "gpu");
        probe_inplace(adapter, "gpu", true);
        probe_optimizer_steps(adapter, "gpu", true);
        probe_decay_writeback(adapter, "gpu");
    }
#else
    // 纯 CPU 构建：本探针的核心价值是 GPU f16 写回（issue #13 P0-②），
    // 无 Vulkan 时无可测对象 → ctest SKIP（同 gpu_test 约定）
    (void)argc; (void)argv;
    std::printf("[SKIP] 此程序需要 Vulkan SDK 支持（GPU f16 写回探针）。\n");
    return 77;
#endif

    if (g_failures == 0) { std::printf("ALL PASSED\n"); return 0; }
    std::printf("%d FAILURES\n", g_failures);
    return 1;
}
