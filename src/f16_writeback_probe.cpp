// ── f16_writeback_probe：GPU f16 参数写回最小探针（issue #13 P0-②）────────
// 症状：MSVC 编译的 binary 上，GPU + param=f16 训练 loss 恒 = ln(V)（权重
//       冻结）；clang 同源码正常。CPU 引擎 f16 两边都正常。
// 定位：f16_precision_test 的 GPU 轨迹对拍 MSVC 打印 3.4776 → 3.4776（冻结）
//       而 clang 是 3.1822；AOT 产物逐字节相同 → 差异在 MSVC 编译的 host 侧。
// 本探针把「参数更新」缩到单表达式：
//   K3 模式 = dsl::compute_into(engine, leaf(p) + leaf(delta), p)
//   p 为 f16 存储（PrecisionEngine 适配层路径），逐步打印写入前后值。
// 期望：写后 p == p_before + delta。若不变 → 写回路径坏（host 侧）；
//       若变化 → 上层（optimizer 接线）问题。
// 用法：f16_writeback_probe [--gpu[=名称]]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
        nn::PrecisionEngine padapter(cpu);
        std::printf("── CPU ──\n");
        probe_inplace(padapter, "cpu", false);
        probe_optimizer_steps(padapter, "cpu", false);
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
        nn::PrecisionEngine adapter(gpu);
        std::printf("── GPU（%s）──\n", backend.device().device_name().c_str());
        probe_inplace(adapter, "gpu", true);
        probe_optimizer_steps(adapter, "gpu", true);
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
