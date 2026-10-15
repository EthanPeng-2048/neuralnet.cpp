// ── scan_bitwise_test — RLA 扫描原语的**逐位**回归（CPU 参考 ↔ GPU）─────────
//
// 为什么需要（P4-2b）：`scan_prefix_outer` 的专用路径（dk ≤ 64）本轮改了结构
// （行归属 + 内层 4 路展开 + `dk ≤ 32` 窄共享变体）。改写的全部前提是**逐位不变**：
// GPU kernel 与 CPU 实现（`compute_cpu_engine.hpp` 的 `scan_*_outer_f32_`）对每个
// 元素的累加次序逐一相同（每元素每 t 恰好加一次、各行 j/c/i 均按升序）⇒ f32 下
// 输出必须**逐位**相等。
//
// 现有覆盖缺口：`gpu_test` 的扫描段只测 dk=128（通用路径）且容差 2e-3，
// **专用路径（dk≤64）与窄/宽变体选择完全没进回归网**。本测试补齐：
//   · dk ∈ {16, 31, 32, 33, 64}（16/31/32 走窄变体、33/64 走宽变体；31 走尾循环）
//   · causal × has_state × has_bnd 全组合；batch>1（铁律 5）
//   · seq 取奇数（7）让逐 t 循环不整齐
//   · f32 逐位比较 + f16 存储路径（f16→f32 精确升位后逐位比较）
//
// 用法：scan_bitwise_test（无参数）
// GPU 不可用（未编译 Vulkan / 初始化失败）→ 打印 77 = ctest SKIP 约定
// （与 gpu_test 同约定：程序自己判断可用性，不靠命令行开关）。
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cstdio>

#if !defined(NN_HAS_VULKAN)

int main()
{
    std::printf("[SKIP] scan_bitwise_test 需要 Vulkan（本构建未编译 GPU 后端）。\n");
    return 77;
}

#else

#include <cstdint>
#include <cstring>
#include <random>
#include <string>

namespace
{

int g_failures = 0;
int g_cases = 0;

std::uint32_t float_as_bits(float f)
{
    std::uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

// 逐位比较（同 u32 位型才算相等；-0.0 与 0.0 视为**不同**——它们位型不同，
// 而本测试的语义正是"位型不动"）
void compare_bits(const char* label, const nn::Matrix& a, const nn::Matrix& b,
                  int& mismatches, float& max_abs)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        std::printf("  [FAIL] %s: 形状不同 (%zu,%zu) vs (%zu,%zu)\n", label, a.rows(),
                    a.cols(), b.rows(), b.cols());
        ++mismatches;
        return;
    }
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
    {
        const std::uint32_t ba = float_as_bits(sa[i]);
        const std::uint32_t bb = float_as_bits(sb[i]);
        const float d = sa[i] - sb[i];
        const float ad = d < 0.0f ? -d : d;
        if (ad > max_abs) max_abs = ad;
        if (ba != bb)
        {
            if (mismatches < 3)
                std::printf("    [bit] %s idx=%zu cpu=%.9g(0x%08x) gpu=%.9g(0x%08x)\n",
                            label, i, static_cast<double>(sa[i]), ba,
                            static_cast<double>(sb[i]), bb);
            ++mismatches;
        }
    }
}

nn::Matrix make_matrix(std::mt19937& rng, std::size_t rows, std::size_t cols)
{
    nn::Matrix m(rows, cols);
    std::normal_distribution<nn::Scalar> d(0.0f, 0.6f);
    for (auto& v : m.span()) v = d(rng);
    return m;
}

// 边界掩码：稀疏置 1（覆盖"文档起点"的真实分布），首位恒 0
nn::Matrix make_boundary(std::mt19937& rng, std::size_t cols)
{
    nn::Matrix m(1, cols);
    std::uniform_int_distribution<int> d(0, 3);
    for (std::size_t i = 0; i < cols; ++i)
        m.set_value(0, i, (i > 0 && d(rng) == 0) ? nn::Scalar{1} : nn::Scalar{0});
    return m;
}

// 一次 prefix 比对（两个引擎跑同一份输入）
void run_prefix_case(nn::ComputeEngine& cpu, nn::ComputeEngine& gpu, std::size_t dk,
                     bool causal, bool has_state, bool has_bnd, bool f16)
{
    const std::size_t B = 2, H = 2, seq = 7;
    const std::size_t rows = B * H * dk;
    const nn::Precision prec = f16 ? nn::Precision::F16 : nn::Precision::F32;

    std::mt19937 rng(20261015u + dk * 17u + (causal ? 1u : 0u) * 3u +
                     (has_state ? 1u : 0u) * 5u + (has_bnd ? 1u : 0u) * 7u +
                     (f16 ? 1u : 0u) * 11u);
    const nn::Matrix Km = make_matrix(rng, rows, seq);
    const nn::Matrix Vm = make_matrix(rng, rows, seq);
    const nn::Matrix Pm = make_matrix(rng, rows, seq);
    const nn::Matrix Rm = make_matrix(rng, rows, seq);
    const nn::Matrix A0m = make_matrix(rng, H * dk, dk);
    const nn::Matrix B0m = make_matrix(rng, H * dk, dk);
    const nn::Matrix bnd = make_boundary(rng, B * seq);

    auto run = [&](nn::ComputeEngine& e) -> nn::Result<nn::Matrix> {
        NN_TRY(k, e.from_matrix(Km, prec));
        NN_TRY(v, e.from_matrix(Vm, prec));
        NN_TRY(p, e.from_matrix(Pm, prec));
        NN_TRY(r, e.from_matrix(Rm, prec));
        NN_TRY(a0, e.from_matrix(A0m, prec));
        NN_TRY(b0, e.from_matrix(B0m, prec));
        NN_TRY(bd, e.from_matrix(bnd, prec));
        auto out = e.scan_prefix_outer(*k, *v, *p, *r, *a0, *b0, has_state, dk, H, causal,
                                       *bd, has_bnd, prec);
        NN_TRY_CHECK(out);
        return e.to_matrix(*out);
    };

    ++g_cases;
    char label[160];
    std::snprintf(label, sizeof(label), "prefix dk=%zu causal=%d state=%d bnd=%d %s",
                  dk, causal ? 1 : 0, has_state ? 1 : 0, has_bnd ? 1 : 0,
                  f16 ? "f16" : "f32");
    auto cm = run(cpu);
    auto gm = run(gpu);
    if (!cm || !gm)
    {
        std::printf("  [FAIL] %s: 运行失败 (%s%s)\n", label,
                    cm ? "" : cm.error().message.c_str(),
                    gm ? "" : gm.error().message.c_str());
        ++g_failures;
        return;
    }
    int mism = 0;
    float max_abs = 0.0f;
    compare_bits(label, *cm, *gm, mism, max_abs);
    if (mism != 0)
    {
        std::printf("  [FAIL] %s: %d/%zu 个元素位型不同（max|Δ| = %.3g）\n", label, mism,
                    cm->rows() * cm->cols(), static_cast<double>(max_abs));
        ++g_failures;
    }
    else
    {
        std::printf("  [PASS] %s: %zu 个元素逐位一致\n", label, cm->rows() * cm->cols());
    }
}

void run_suffix_case(nn::ComputeEngine& cpu, nn::ComputeEngine& gpu, std::size_t dk,
                     bool causal, bool has_bnd, bool f16)
{
    const std::size_t B = 2, H = 2, seq = 7;
    const std::size_t rows = B * H * dk;
    const nn::Precision prec = f16 ? nn::Precision::F16 : nn::Precision::F32;

    std::mt19937 rng(777000u + dk * 13u + (causal ? 1u : 0u) * 3u +
                     (has_bnd ? 1u : 0u) * 7u + (f16 ? 1u : 0u) * 11u);
    const nn::Matrix Dm = make_matrix(rng, rows * dk, seq);
    const nn::Matrix Xm = make_matrix(rng, rows, seq);
    const nn::Matrix Ym = make_matrix(rng, rows, seq);
    const nn::Matrix bnd = make_boundary(rng, B * seq);

    auto run = [&](nn::ComputeEngine& e) -> nn::Result<nn::Matrix> {
        NN_TRY(d, e.from_matrix(Dm, prec));
        NN_TRY(x, e.from_matrix(Xm, prec));
        NN_TRY(y, e.from_matrix(Ym, prec));
        NN_TRY(bd, e.from_matrix(bnd, prec));
        auto out = e.scan_suffix_outer(*d, *x, *y, dk, H, causal, *bd, has_bnd, prec);
        NN_TRY_CHECK(out);
        return e.to_matrix(*out);
    };

    ++g_cases;
    char label[160];
    std::snprintf(label, sizeof(label), "suffix dk=%zu causal=%d bnd=%d %s", dk,
                  causal ? 1 : 0, has_bnd ? 1 : 0, f16 ? "f16" : "f32");
    auto cm = run(cpu);
    auto gm = run(gpu);
    if (!cm || !gm)
    {
        std::printf("  [FAIL] %s: 运行失败\n", label);
        ++g_failures;
        return;
    }
    int mism = 0;
    float max_abs = 0.0f;
    compare_bits(label, *cm, *gm, mism, max_abs);
    if (mism != 0)
    {
        std::printf("  [FAIL] %s: %d/%zu 个元素位型不同（max|Δ| = %.3g）\n", label, mism,
                    cm->rows() * cm->cols(), static_cast<double>(max_abs));
        ++g_failures;
    }
    else
    {
        std::printf("  [PASS] %s: %zu 个元素逐位一致\n", label, cm->rows() * cm->cols());
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    auto cpu_r = nn::cli::create_engine(nn::cli::EngineConfig{false});
    NN_EXIT(cpu_r, 1, "CPU 引擎创建失败: ");
    auto gpu_r = nn::cli::create_engine(nn::cli::EngineConfig{true});
    NN_EXIT(gpu_r, 77, "[SKIP] GPU 不可用: ");
    nn::ComputeEngine& cpu = **cpu_r;
    nn::ComputeEngine& gpu = **gpu_r;

    std::puts("=== scan_bitwise（CPU 参考 ↔ GPU kernel，逐位）===");
    std::puts("-- 前缀扫描：dk 16/31/32（窄变体）· 33/64（宽变体）--");
    for (std::size_t dk : {16u, 31u, 32u, 33u, 64u})
    {
        for (bool causal : {true, false})
        {
            for (bool has_state : {false, true})
            {
                for (bool has_bnd : {false, true})
                {
                    run_prefix_case(cpu, gpu, dk, causal, has_state, has_bnd, false);
                }
            }
        }
    }
    std::puts("-- 前缀扫描（f16 存储）--");
    for (std::size_t dk : {16u, 32u, 64u})
        for (bool has_bnd : {false, true})
            run_prefix_case(cpu, gpu, dk, /*causal=*/true, /*has_state=*/true, has_bnd, true);

    std::puts("-- 后缀扫描 --");
    for (std::size_t dk : {16u, 32u, 64u})
        for (bool causal : {true, false})
            for (bool has_bnd : {false, true})
                run_suffix_case(cpu, gpu, dk, causal, has_bnd, false);
    std::puts("-- 后缀扫描（f16 存储）--");
    for (std::size_t dk : {16u, 32u, 64u})
        run_suffix_case(cpu, gpu, dk, /*causal=*/true, /*has_bnd=*/true, true);

    std::printf("\nscan_bitwise_test: %d case(s), %d failure(s)\n", g_cases, g_failures);
    return g_failures != 0 ? 1 : 0;
}

#endif  // NN_HAS_VULKAN
