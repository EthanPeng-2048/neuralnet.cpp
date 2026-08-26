// ───────────────────────────────────────────────────────────────────────────
//  reduce_instr_bench.cpp — 归约【指令 pass】多累加器收益基准
//
//  归约视图 pass（row_reduce_sum(leaf) 等）早已用 4 路独立累加器（acc0-acc3）。
//  归约【指令】pass（reduce 作用在中间寄存器，如 col_reduce_sum(a*b)、
//  row_reduce_sum(exp(x))）曾长期是单累加器串行——acc 依赖链限制 ILP。
//  P1-05 已将其改为 4 路独立标量累加器 + 4 路展开（见 glsl_gen.hpp）。
//  本基准量化该改动收益：同样走 AOT 融合归约 shader，仅累加器结构不同。
//
//  指标：GB/s = 读入整矩阵字节数 / 耗时。对比方式：
//    新代码（多累加器）直接跑；旧代码用 git stash 暂时回退 glsl_gen.hpp 后重跑。
//
//  用法：reduce_instr_bench [shapes...]  （默认若干大形状）
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;
namespace dsl = nn::dsl;

#ifndef NN_HAS_VULKAN
int main()
{
    std::printf("[SKIP] 需要 Vulkan SDK（-DNN_HAS_VULKAN）\n");
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::GpuBackend;
using nn::GpuEngine;
using nn::Tensor;

namespace
{

template <typename F>
double time_call(F&& f, int iters)
{
    f();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto r = f();
        if (!r) { std::printf("[ERR] 调用失败: %s\n", r.error().message.c_str()); return -1.0; }
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

// 归约【指令】pass 内核吞吐（数据驻留显存，仅测内核，排除 PCIe 上传/下载）。
// 归约指令 pass = reduce 作用在中间寄存器（如 col_reduce_sum(a*b)，LayerNorm 方差路径）。
// 每个 shape 上传一次，之后反复调 dsl::compute_reduce 计时内核；GPU 张量作为 leaf 保持驻留。
void bench_shape(GpuEngine& gpu, std::size_t rows, std::size_t cols, int iters)
{
    const double bytes = static_cast<double>(rows) * cols * sizeof(Scalar);
    const double mbytes = bytes / 1e6;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    nn::Matrix ma(rows, cols), mb(rows, cols);
    for (auto& v : ma.span()) v = dist(rng);
    for (auto& v : mb.span()) v = dist(rng);
    auto ar = gpu.from_matrix(ma);
    auto br = gpu.from_matrix(mb);
    if (!ar || !br) { std::printf("[ERR] 上传失败\n"); return; }
    const Tensor at = std::move(*ar);
    const Tensor bt = std::move(*br);

    // 列归约指令 pass（LayerNorm 方差模式）: col_reduce_sum(a*b)
    const double col = time_call([&]{
        return dsl::compute_reduce(gpu, dsl::col_reduce_sum(dsl::leaf(at) * dsl::leaf(bt)),
                                   rows, cols); }, iters);

    const double col_g = col > 0 ? bytes / col / 1e9 : 0.0;
    std::printf("shape %zux%-8zu (%.0f MB)\n", rows, cols, mbytes);
    std::printf("  归约指令pass·列(col_reduce_sum a*b): %7.1f GB/s  (%.0f us)\n", col_g, col * 1e6);
}

} // namespace

int main(int argc, char** argv)
{
    std::printf("reduce_instr_bench — 归约指令 pass（P1-05 多累加器）吞吐\n");
    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::printf("[ERR] GPU 初始化失败: %s\n", init_r.error().message.c_str());
        return 1;
    }
    GpuEngine gpu(backend);

    std::vector<std::pair<std::size_t, std::size_t>> shapes;
    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            std::size_t r = 0, c = 0;
            if (std::sscanf(argv[i], "%zu,%zu", &r, &c) == 2)
                shapes.push_back({r, c});
        }
    }
    if (shapes.empty())
        shapes = {{4096, 4096}, {8192, 8192}, {1024, 65536}, {65536, 1024}};

    for (auto& [r, c] : shapes)
    {
        const int iters = (r * c >= 256ull * 1024 * 1024) ? 5 : 20;
        bench_shape(gpu, r, c, iters);
    }
    return 0;
}
#endif // NN_HAS_VULKAN
