// ── decay_probe：f16 参数下 AdamW 权重衰减乘法的舍入行为探针 ─────────────
// 症状（用户实测）：GPU + --precision-param f16 + AdamW(wd>0) 训练时，
//   所有权重每 step 恰好掉 1 个 f16 ULP（与 lr 无关、与 wd>0 门控），
//   数万步后全部下溢为 0 → 模型退化、loss 回到 ln(V)。
// 本探针把该乘法缩到单表达式：
//   decay = dsl::compute_into(engine, leaf(p) * rparam(1 - lr*wd), p)
// 对照：
//   add   = dsl::compute_into(engine, leaf(p) + leaf(delta), p)   (优化器 K3)
// 分别在 CPU / GPU、f16 / f32 目标精度上各跑 N 步，打印值与 ULP 位移。
// 用法：decay_probe [--gpu[=名称]]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using nn::Matrix;
using nn::Precision;
using nn::Scalar;

namespace {

// f16 网格：给定 f32 值与相邻 f16 网格点，打印“掉了几个 ULP”
void report(const char* label, const std::vector<Scalar>& before,
            const std::vector<Scalar>& after, int steps)
{
    std::printf("  [%s] %d 步：\n", label, steps);
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        const Scalar b = before[i], a = after[i];
        if (b == a)
        {
            std::printf("    p0=%-12.7g → %-12.7g (不变)\n", b, a);
            continue;
        }
        // ULP 位移 = |1/p - 1/a| 不便；直接用 half 网格步长推：按 f16 尾数
        auto ulp = [](Scalar v) -> Scalar {
            v = std::fabs(v);
            if (v == 0) return std::numeric_limits<Scalar>::min();
            int e; std::frexp(v, &e);           // v = f * 2^e, 0.5<=f<1
            return std::ldexp(1.0f, e - 11);    // f16 尾数 10 bit + 符号位对齐
        };
        const Scalar u = ulp(b);
        const Scalar steps_down = (b - a) / u;  // 正 = 掉网格
        std::printf("    p0=%-12.7g → %-12.7g  Δ/ULP=%-8.3f  相对=%-10.4g\n",
                    b, a, steps_down, (a - b) / b);
    }
}

std::vector<Scalar> download(nn::ComputeEngine& eng, const nn::Tensor& t)
{
    auto m = eng.to_matrix(t, Precision::F32);
    if (!m) return {};
    std::vector<Scalar> out(m->span().begin(), m->span().end());
    return out;
}

// 权重衰减乘法：p *= (1 - lr*wd)，目标精度 P
void probe_decay(nn::ComputeEngine& eng, const char* label, Precision P)
{
    const Scalar lr = 3e-4f, wd = 0.1f;
    const Scalar factor = Scalar{1} - lr * wd;   // 0.99997（host 侧正确值）
    const int steps = 500;

    Matrix m(1, 6);
    const Scalar init[6] = {1.0f, 0.5f, 0.077f, -0.077f, 0.036f, 2.0f};
    for (std::size_t i = 0; i < 6; ++i) m.span()[i] = init[i];

    auto p_r = eng.from_matrix(m, P);
    if (!p_r) { std::printf("  [%s] from_matrix 失败\n", label); return; }
    auto p = std::move(*p_r);

    const auto before = download(eng, p);
    for (int s = 0; s < steps; ++s)
    {
        auto r = nn::dsl::compute_into(eng,
            nn::dsl::leaf(p) * nn::dsl::rparam(factor), p);
        if (!r) { std::printf("  [%s] decay compute_into 失败: %s\n",
                              label, r.error().message.c_str()); return; }
    }
    const auto after = download(eng, p);

    std::printf("[decay] %s target=%s factor=%.7f（应为“不变”：|Δ| 远小于 1 ULP）\n",
                label, P == Precision::F16 ? "f16" : "f32", factor);
    report("decay", before, after, steps);
}

// 优化器 K3：p += delta（delta 与 p 同精度），delta = ±3e-4 交替
void probe_add(nn::ComputeEngine& eng, const char* label, Precision P)
{
    const int steps = 500;
    const Scalar d = 3e-4f;

    Matrix m(1, 6);
    const Scalar init[6] = {1.0f, 0.5f, 0.077f, -0.077f, 0.036f, 2.0f};
    for (std::size_t i = 0; i < 6; ++i) m.span()[i] = init[i];

    auto p_r = eng.from_matrix(m, P);
    if (!p_r) { std::printf("  [%s] from_matrix 失败\n", label); return; }
    auto p = std::move(*p_r);

    const auto before = download(eng, p);
    for (int s = 0; s < steps; ++s)
    {
        Matrix dm(1, 6);
        // 交替 ±d：偶数步 +d，奇数步 -d（模拟 Adam 双向更新）
        const Scalar sign = (s % 2 == 0) ? d : -d;
        for (std::size_t i = 0; i < 6; ++i) dm.span()[i] = sign;
        auto d_r = eng.from_matrix(dm, P);
        if (!d_r) { std::printf("  [%s] delta from_matrix 失败\n", label); return; }
        auto r = nn::dsl::compute_into(eng,
            nn::dsl::leaf(p) + nn::dsl::leaf(*d_r), p);
        if (!r) { std::printf("  [%s] add compute_into 失败: %s\n",
                              label, r.error().message.c_str()); return; }
    }
    const auto after = download(eng, p);

    std::printf("[p+delta] %s target=%s（±%.0e 交替：应为小幅随机游走、无系统性收缩）\n",
                label, P == Precision::F16 ? "f16" : "f32", d);
    report("add", before, after, steps);
}

void run_all(nn::ComputeEngine& eng, const char* label)
{
    probe_decay(eng, label, Precision::F32);
    probe_decay(eng, label, Precision::F16);
    probe_add(eng, label, Precision::F32);
    probe_add(eng, label, Precision::F16);
}

} // namespace

int main(int argc, char* argv[])
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string gpu_device;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (auto dev = nn::cli::parse_gpu_option(argc, argv, i, true))
            gpu_device = *dev;
        else if (arg == "--gpu") gpu_device.clear();
        else { std::fprintf(stderr, "未知参数: %s\n", arg.c_str()); return 2; }
    }

    std::printf("decay_probe：f16 权重衰减乘法 / 加法写回的舍入行为\n");
    {
        nn::CpuEngine cpu;
        nn::ComputeEngine& padapter = cpu;
        std::printf("── CPU ──\n");
        run_all(padapter, "cpu");
    }

#ifdef NN_HAS_VULKAN
    {
        auto& backend = nn::GpuBackend::instance();
        auto init_r = backend.initialize(gpu_device);
        if (!init_r)
        {
            std::fprintf(stderr, "GPU 初始化失败: %s\n", init_r.error().message.c_str());
            return 77;
        }
        nn::GpuEngine gpu(backend);
        nn::ComputeEngine& adapter = gpu;
        std::printf("── GPU（%s）──\n", backend.device().device_name().c_str());
        run_all(adapter, "gpu");
    }
#else
    (void)argc; (void)argv;
    std::printf("[SKIP] 需要 Vulkan SDK。\n");
    return 77;
#endif
    return 0;
}
