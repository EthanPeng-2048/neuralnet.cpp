// ───────────────────────────────────────────────────────────────────────────
//  reduce_bench.cpp — 归约 kernel 性能对比（旧共享内存树 vs 新 warp shuffle）
//
//  旧算法：GpuEngine::row_reduce_sum / col_reduce_sum
//          → 独立 reduce.comp（单累加器 + 共享内存树形归约）
//  新算法：dsl::compute 触发 AOT 融合归约 shader（fused_*）
//          → 单累加器 + warp shuffle(subgroup) 归约
//  两者均为"单累加器 + 逐线程跨步累加"，唯一差异是最终树形归约实现
//  （共享内存树 vs subgroup 蝴蝶归约）→ 干净隔离 warp shuffle 收益。
//
//  指标：GB/s = 读入整矩阵字节数 / 耗时。归约 kernel 主要受内存带宽限制。
//
//  用法：reduce_bench [shapes...]  （默认若干大形状）
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

// 计时一个可空返回的 GPU 调用，返回单次平均耗时（秒）
template <typename F>
double time_call(F&& f, int iters)
{
    // warmup
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

// 扣除启动/同步开销后的内核吞吐（GB/s）。overhead 用同算子小矩阵（内核≈0）测得。
// 目的：去除逐调用 submit+wait 的固定开销，还原真实内核带宽，判断跨步访问是否真瓶颈。
double kernel_gbps(double bytes, double shape_s, double overhead_s)
{
    const double kt = shape_s - overhead_s;
    if (kt <= 0.0) return 0.0;
    return bytes / kt / 1e9;
}

void bench_shape(GpuEngine& gpu, std::size_t rows, std::size_t cols, int iters)
{
    const double bytes = static_cast<double>(rows) * cols * sizeof(Scalar);
    const double mbytes = bytes / 1e6;   // MB

    std::mt19937 rng(12345);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Tensor x = Tensor::cpu(rows, cols);
    for (auto& v : x.cpu_matrix().span()) v = dist(rng);
    const Tensor xt = x;

    // ── 启动/同步开销：小矩阵上的归约（内核时间≈0），同算子测一遍 ──
    Tensor xoh = Tensor::cpu(256, 256);
    for (auto& v : xoh.cpu_matrix().span()) v = 1.0f;
    const Tensor xoht = xoh;
    const double overhead = time_call([&]{ return gpu.col_reduce_sum(xoht); }, iters);

    // ── 行求和：旧（reduce_gpu） vs 新（融合 row_reduce_sum）──
    // 新路径用 compute_reduce（输出归约向量 (rows,1)），与旧路径输出对齐，
    // 避免 compute() 把归约结果广播写到整个 (rows,cols)（多一次整矩阵写）。
    const double old_rs = time_call([&]{ return gpu.row_reduce_sum(xt); }, iters);
    const double new_rs = time_call([&]{
        return dsl::compute_reduce(gpu, dsl::row_reduce_sum(dsl::leaf(xt)), rows, cols); }, iters);

    // ── 列求和：旧（reduce_gpu） vs 新（融合 col_reduce_sum）──
    const double old_cs = time_call([&]{ return gpu.col_reduce_sum(xt); }, iters);
    const double new_cs = time_call([&]{
        return dsl::compute_reduce(gpu, dsl::col_reduce_sum(dsl::leaf(xt)), rows, cols); }, iters);

    std::printf("shape %zux%-8zu  (%.0f MB)  启动开销≈%.0fus\n",
                rows, cols, mbytes, overhead * 1e6);
    const double old_rs_g = kernel_gbps(bytes, old_rs, overhead);
    const double new_rs_g = kernel_gbps(bytes, new_rs, overhead);
    const double old_cs_g = kernel_gbps(bytes, old_cs, overhead);
    const double new_cs_g = kernel_gbps(bytes, new_cs, overhead);
    std::printf("  行求和(内核): 旧 %7.1f GB/s  新 %7.1f GB/s  加速 %s\n",
        old_rs_g, new_rs_g,
        (old_rs_g > 0 && new_rs_g > 0) ? (std::to_string(old_rs / new_rs) + "x").c_str() : "-");
    std::printf("  列求和(内核): 旧 %7.1f GB/s  新 %7.1f GB/s  加速 %s\n",
        old_cs_g, new_cs_g,
        (old_cs_g > 0 && new_cs_g > 0) ? (std::to_string(old_cs / new_cs) + "x").c_str() : "-");
    std::printf("  跨步惩罚: 列/行(新) = %s   (1.0=无惩罚；<1=列跨步访问更慢)\n",
        (new_rs_g > 0 && new_cs_g > 0) ? (std::to_string(new_cs_g / new_rs_g) + "x").c_str() : "-");
}

// 纯内核基准：数据一次性上传到显存后驻留，反复调 reduce_gpu（不再上传）。
// 目的：去掉 CPU→GPU 上传（PRIME/PCIe）对测量的污染，还原真实内核带宽，
// 判断列归约跨步访问的纯内核惩罚到底多大（决定 tile 化是否值得）。
void bench_pure_kernel(GpuBackend& backend, std::size_t rows, std::size_t cols, int iters)
{
    std::mt19937 rng(777);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    nn::Matrix m(rows, cols);
    for (auto& v : m.span()) v = dist(rng);
    auto gt_r = nn::GpuTensor::from_matrix(m, backend);
    if (!gt_r) { std::printf("[ERR] 上传到显存失败: %s\n", gt_r.error().message.c_str()); return; }
    nn::GpuTensor gt = std::move(*gt_r);
    const double bytes = static_cast<double>(rows) * cols * sizeof(Scalar);

    const double t_row = time_call([&]{ return backend.reduce_gpu(gt, 0u, 0u); }, iters); // 行,sum
    const double t_col = time_call([&]{ return backend.reduce_gpu(gt, 1u, 0u); }, iters); // 列,sum
    const double row_g = t_row > 0 ? bytes / t_row / 1e9 : 0.0;
    const double col_g = t_col > 0 ? bytes / t_col / 1e9 : 0.0;
    std::printf("  纯内核(数据驻留): 行归约 %7.1f GB/s  列归约 %7.1f GB/s  列/行 = %s\n",
        row_g, col_g,
        (row_g > 0 && col_g > 0) ? (std::to_string(col_g / row_g) + "x").c_str() : "-");
}

} // namespace

int main(int argc, char** argv)
{
    std::printf("reduce_bench — 旧(共享内存树) vs 新(warp shuffle) 归约吞吐\n");
    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::printf("[ERR] GPU 初始化失败: %s\n", init_r.error().message.c_str());
        return 1;
    }
    GpuEngine gpu(backend);

    // 默认形状；可用命令行覆盖：reduce_bench 4096,4096 8192,8192 ...
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
        // 迭代次数随规模缩放，避免总耗时过长
        const int iters = (r * c >= 256ull * 1024 * 1024) ? 3 : 10;
        bench_shape(gpu, r, c, iters);
        bench_pure_kernel(backend, r, c, iters);
    }
    return 0;
}
#endif // NN_HAS_VULKAN
