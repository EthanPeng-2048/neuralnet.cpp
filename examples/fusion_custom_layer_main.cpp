// ═══════════════════════════════════════════════════════════════════════════
//  examples/fusion_custom_layer_main.cpp — A2 验收样例的可执行入口
//
//  验收断言（构建期即完成，见 CMake 的 nn_enable_gpu_fusion）：
//    本目标**只**依赖自己的 `fused_registry.hpp`（由 nn_enable_gpu_fusion
//    用本应用的 TU + 库内收集逻辑生成），该注册表里必须出现自研表达式
//    的结构键（以及 `#x` 运行期精度分派变体）——**库内文件零改动**。
//
//  运行时这里只验证 CPU 路径可用（模板求值，与注册表无关）；GPU 路径的
//  结构登记与分派命中由 A1 的端到端实测覆盖（见 docs/history.md）。
// ═══════════════════════════════════════════════════════════════════════════

#include "fusion_custom_layer.hpp"

#include <cstdio>
#include <span>
#include <vector>

int main()
{
    nn::CpuEngine cpu;
    nn::Tensor x = cpu.create_tensor(4, 3, nn::Precision::F32,
                                     nn::InitSpec::constant(0.5f));
    nn::Tensor y = cpu.create_tensor(4, 3, nn::Precision::F32,
                                     nn::InitSpec::constant(2.0f));
    auto out = nn_example::fused_custom_op(cpu, x, y, 0.7f);
    if (!out)
    {
        std::fprintf(stderr, "[example] fused_custom_op 失败: %s\n",
                     out.error().message.c_str());
        return 1;
    }
    std::vector<float> buf(12, 0.0f);
    if (auto r = cpu.read(*out, std::span<float>(buf)); !r)
    {
        std::fprintf(stderr, "[example] read 失败\n");
        return 1;
    }
    // exp(0.5*0.7) / (sqrt(2.0) + 0.7)
    const float expect = 1.419067549f / (1.414213562f + 0.7f);
    const float got = buf[0];
    const float diff = got > expect ? got - expect : expect - got;
    std::printf("[example] 自研融合层 CPU 结果 %.7f（期望 %.7f，差 %.3e）\n",
                static_cast<double>(got), static_cast<double>(expect),
                static_cast<double>(diff));
    if (diff > 1e-5f)
    {
        std::fprintf(stderr, "[example] 数值不符\n");
        return 1;
    }
    std::puts("[example] OK");
    return 0;
}
