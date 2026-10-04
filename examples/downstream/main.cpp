// ═══════════════════════════════════════════════════════════════════════════
//  examples/downstream/main.cpp — 最小下游消费样例（P0-4 验收）
//
//  形态 = "把库丢进 3rd_party/、CMake 两行、源码一个 include"：
//
//      add_subdirectory(3rd_party/neuralnet.cpp)
//      target_link_libraries(downstream_demo PRIVATE neuralnet::nn)
//
//      #include <neuralnet.cpp/nn.hpp>
//
//  断言：Linear forward 输出形状正确且数值有限；若编译时带了 NN_HAS_VULKAN
//  （由 neuralnet::nn 自动传递），再跑一遍 GPU 前向对拍（无设备 = 77 skip）。
//  退出码：0 = 通过，1 = 失败，77 = 跳过（ctest 约定）。
// ═══════════════════════════════════════════════════════════════════════════

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

namespace
{

// 构造一个可复现的 (rows, cols) 输入矩阵
nn::Matrix make_input(std::size_t rows, std::size_t cols)
{
    nn::Matrix m(rows, cols);
    float v = 0.01f;
    for (std::size_t c = 0; c < cols; ++c)
        for (std::size_t r = 0; r < rows; ++r)
            m.set_value(r, c, v += 0.017f);
    return m;
}

// 跑一次 Linear(D→OUT) 前向，返回输出首元素（失败返回 false）
bool run_linear(nn::ComputeEngine& engine, const char* tag, float& first_out)
{
    constexpr std::size_t D = 8, OUT = 4, BATCH = 2;

    nn::Linear layer(D, OUT);
    if (auto r = layer.init(engine); !r)
    {
        std::fprintf(stderr, "[%s] Linear::init 失败: %s\n", tag, r.error().message.c_str());
        return false;
    }

    auto x = engine.from_matrix(make_input(D, BATCH));
    if (!x)
    {
        std::fprintf(stderr, "[%s] from_matrix 失败: %s\n", tag, x.error().message.c_str());
        return false;
    }

    auto y = layer.forward(*x);
    if (!y)
    {
        std::fprintf(stderr, "[%s] forward 失败: %s\n", tag, y.error().message.c_str());
        return false;
    }
    if (y->rows() != OUT || y->cols() != BATCH)
    {
        std::fprintf(stderr, "[%s] 输出形状错误: 期望 (%zu,%zu)，实际 (%zu,%zu)\n",
                     tag, OUT, BATCH, y->rows(), y->cols());
        return false;
    }

    std::vector<float> buf(OUT * BATCH);
    if (auto r = engine.read(*y, std::span<float>(buf)); !r)
    {
        std::fprintf(stderr, "[%s] read 失败: %s\n", tag, r.error().message.c_str());
        return false;
    }
    for (float f : buf)
    {
        if (!std::isfinite(f))
        {
            std::fprintf(stderr, "[%s] 输出含非有限值\n", tag);
            return false;
        }
    }
    first_out = buf[0];
    return true;
}

} // namespace

int main()
{
    std::printf("[downstream] neuralnet.cpp 最小消费样例\n");

    // ── CPU 路径（始终执行）─────────────────────────────────────────────
    nn::CpuEngine cpu;
    float cpu_out = 0.0f;
    if (!run_linear(cpu, "cpu", cpu_out)) return 1;
    std::printf("[downstream] CPU Linear forward OK（y[0]=%.6f）\n",
                static_cast<double>(cpu_out));

#ifdef NN_HAS_VULKAN
    // ── GPU 路径（neuralnet::nn 自动传入 NN_HAS_VULKAN 才编译进来）────────
    // 无 Vulkan 设备/初始化失败 → 按 ctest 约定返回 77（skip），不算失败。
    auto& backend = nn::GpuBackend::instance();
    if (auto ir = backend.initialize(); !ir)
    {
        std::printf("[downstream] GPU 初始化失败，跳过 GPU 校验：%s\n",
                    ir.error().message.c_str());
        return 77;
    }
    nn::GpuEngine gpu(backend);
    float gpu_out = 0.0f;
    if (!run_linear(gpu, "gpu", gpu_out)) return 1;

    // GPU 与 CPU 用不同随机权重初始化，无法直接对拍数值；
    // 这里只断言两者都跑通（真正的跨引擎对拍在库内测试覆盖）。
    std::printf("[downstream] GPU Linear forward OK（y[0]=%.6f）\n",
                static_cast<double>(gpu_out));
    // GPU 与 CPU 权重不同 → 仅形状/有限性断言已由 run_linear 完成。
    (void)cpu_out;
#else
    std::printf("[downstream] 纯 CPU 构建（未定义 NN_HAS_VULKAN）\n");
#endif

    std::puts("[downstream] OK");
    return 0;
}
