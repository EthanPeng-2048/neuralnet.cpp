// ═══════════════════════════════════════════════════════════════════════════
//  examples/fusion_custom_layer_main.cpp — A2 验收样例的可执行入口
//
//  这是"写 dsl::compute → 只跑构建 → **GPU 能跑**"的端到端验收：
//
//  ① 结构收集：本目标**只**依赖自己的 `fused_registry.hpp`（由
//     `nn_enable_gpu_fusion` 用本应用的 TU + 库内收集逻辑生成）。该注册表里
//     必须出现自研表达式的结构键 `key` 与 `key#x`（运行期精度分派）——
//     库内文件零改动。
//  ② **f16 分派路径**：下面用 f16 张量 + f16 输出跑同一个自研表达式。本表达式
//     含 rparams ⇒ `expr_prec_sig_native16` 谓词不通过 ⇒ 命中的必然是
//     `key#x`（运行期精度分派 shader，双视图 + PC `prec`）。注册表若缺它，
//     运行期会**闭合世界硬报错**——所以"跑通"本身就是断言。
//  ③ f32 路径（V0）与 CPU 参考逐位/紧容差对拍。
// ═══════════════════════════════════════════════════════════════════════════

#include "fusion_custom_layer.hpp"

#include <cstdio>
#include <span>
#include <vector>

namespace
{

// exp(0.5*0.7) / (sqrt(2.0) + 0.7)
constexpr float kExpect = 1.419067549f / (1.414213562f + 0.7f);

float abs_diff(float a, float b) { return a > b ? a - b : b - a; }

} // namespace

int main()
{
    // ── CPU 参考 ─────────────────────────────────────────────────────────
    nn::CpuEngine cpu;
    nn::Tensor x = cpu.create_tensor(4, 3, nn::Precision::F32,
                                     nn::InitSpec::constant(0.5f));
    nn::Tensor y = cpu.create_tensor(4, 3, nn::Precision::F32,
                                     nn::InitSpec::constant(2.0f));
    auto cout_r = nn_example::fused_custom_op(cpu, x, y, 0.7f);
    if (!cout_r)
    {
        std::fprintf(stderr, "[example] CPU 求值失败: %s\n",
                     cout_r.error().message.c_str());
        return 1;
    }
    std::vector<float> cbuf(12, 0.0f);
    if (auto r = cpu.read(*cout_r, std::span<float>(cbuf)); !r)
    {
        std::fprintf(stderr, "[example] CPU read 失败\n");
        return 1;
    }
    const float cpu_val = cbuf[0];
    std::printf("[example] CPU f32      = %.7f（期望 %.7f，差 %.3e）\n",
                static_cast<double>(cpu_val), static_cast<double>(kExpect),
                static_cast<double>(abs_diff(cpu_val, kExpect)));
    if (abs_diff(cpu_val, kExpect) > 1e-5f)
    {
        std::fprintf(stderr, "[example] CPU 数值不符\n");
        return 1;
    }

#ifdef NN_HAS_VULKAN
    // ── GPU ──────────────────────────────────────────────────────────────
    nn::GpuBackend& backend = nn::GpuBackend::instance();
    if (auto ir = backend.initialize(); !ir)
    {
        std::fprintf(stderr, "[example] GPU 初始化失败（跳过）：%s\n",
                     ir.error().message.c_str());
        return 77;   // ctest 约定：77 = skip
    }
    nn::GpuEngine gpu(backend);

    // ③ f32 路径（注册表命中 V0）
    {
        nn::Tensor gx = gpu.create_tensor(4, 3, nn::Precision::F32,
                                          nn::InitSpec::constant(0.5f));
        nn::Tensor gy = gpu.create_tensor(4, 3, nn::Precision::F32,
                                          nn::InitSpec::constant(2.0f));
        auto g_r = nn_example::fused_custom_op(gpu, gx, gy, 0.7f);
        if (!g_r)
        {
            std::fprintf(stderr, "[example] GPU f32 融合未命中（闭合世界）: %s\n",
                         g_r.error().message.c_str());
            return 1;
        }
        std::vector<float> gb(12, 0.0f);
        if (auto r = gpu.read(*g_r, std::span<float>(gb)); !r)
        {
            std::fprintf(stderr, "[example] GPU f32 read 失败\n");
            return 1;
        }
        std::printf("[example] GPU f32 (V0) = %.7f（与 CPU 差 %.3e）\n",
                    static_cast<double>(gb[0]),
                    static_cast<double>(abs_diff(gb[0], cpu_val)));
        if (abs_diff(gb[0], cpu_val) > 1e-5f)
        {
            std::fprintf(stderr, "[example] GPU f32 与 CPU 不符\n");
            return 1;
        }
    }

    // ② f16 路径（注册表命中 V1 运行期精度分派：双视图 + PC `prec`）
    {
        nn::Tensor gx = gpu.create_tensor(4, 3, nn::Precision::F16,
                                          nn::InitSpec::constant(0.5f));
        nn::Tensor gy = gpu.create_tensor(4, 3, nn::Precision::F16,
                                          nn::InitSpec::constant(2.0f));
        auto g_r = nn_example::fused_custom_op(gpu, gx, gy, 0.7f,
                                               nn::Precision::F16);
        if (!g_r)
        {
            std::fprintf(stderr,
                         "[example] GPU f16 分派变体未命中（闭合世界）: %s\n",
                         g_r.error().message.c_str());
            return 1;
        }
        std::vector<nn::f16> hb(12, nn::f16(0.0f));
        if (auto r = gpu.read(*g_r, std::span<nn::f16>(hb)); !r)
        {
            std::fprintf(stderr, "[example] GPU f16 read 失败\n");
            return 1;
        }
        const float gv = static_cast<float>(hb[0]);
        std::printf("[example] GPU f16 (V1) = %.7f（与 CPU 差 %.3e，f16 容差内）\n",
                    static_cast<double>(gv),
                    static_cast<double>(abs_diff(gv, cpu_val)));
        if (abs_diff(gv, cpu_val) > 2e-3f)
        {
            std::fprintf(stderr, "[example] GPU f16 与 CPU 不符\n");
            return 1;
        }
    }
    std::puts("[example] GPU OK");
#endif

    std::puts("[example] OK");
    return 0;
}
