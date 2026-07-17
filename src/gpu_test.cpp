// ── GPU 矩阵乘法正确性与性能测试 ──────────────────────────────────────────
// 用法：gpu_test [--size N] [--iters N]
//   --size N   矩阵维度（默认 256，即 256×256 方阵）
//   --iters N  性能测试迭代次数（默认 10）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

using nn::Scalar;
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::Scalar;

void print_usage(const char* prog)
{
    std::cout << "GPU 矩阵乘法正确性与性能测试\n\n"
              << "用法: " << prog << " [选项]\n\n"
              << "选项:\n"
              << "  --size N   矩阵维度 (默认: 256)\n"
              << "  --iters N  性能迭代次数 (默认: 10)\n"
              << "  --help     显示此帮助信息\n";
}

int main(int argc, char* argv[])
{
    std::size_t N = 256;
    int iters = 10;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--size" && i + 1 < argc) N = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else { std::cerr << "未知参数: " << arg << "\n"; return 1; }
    }

    std::cout << "========================================\n"
              << "  GPU 矩阵乘法测试\n"
              << "========================================\n"
              << "  矩阵维度: " << N << "×" << N << "\n"
              << "  迭代次数: " << iters << "\n"
              << "========================================\n\n";

    // ── 1. 初始化 GPU ──────────────────────────────────────────────
    std::cout << "[1/4] 初始化 GPU 后端..." << std::flush;
    auto& backend = nn::GpuBackend::instance();
    auto init_result = backend.initialize();
    if (!init_result)
    {
        std::cout << " 失败\n";
        std::cerr << "错误: " << init_result.error().message << "\n";
        return 1;
    }
    std::cout << " 成功\n";

    // ── 2. 生成随机测试数据 ────────────────────────────────────────
    std::cout << "[2/4] 生成 " << N << "×" << N << " 随机矩阵..." << std::flush;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

    nn::Matrix A(N, N), B(N, N);
    for (auto& v : A.data()) v = dist(rng);
    for (auto& v : B.data()) v = dist(rng);
    std::cout << " 完成\n";

    // ── 3. 正确性验证 ─────────────────────────────────────────────
    std::cout << "[3/4] 正确性验证 (CPU vs GPU)..." << std::flush;

    // CPU 参考结果
    nn::Matrix C_cpu(N, N);
    A.multiply_to(C_cpu, B);

    // GPU 结果
    nn::Matrix C_gpu(N, N);
    nn::SmartPolicy::gpu_enabled = true;
    A.multiply_to(C_gpu, B);
    nn::SmartPolicy::gpu_enabled = false;

    // 计算最大绝对误差和相对误差
    Scalar max_abs_err = 0.0;
    Scalar max_rel_err = 0.0;
    Scalar sum_sq_err = 0.0;
    Scalar sum_sq_ref = 0.0;

    for (std::size_t i = 0; i < N * N; ++i)
    {
        Scalar ref = C_cpu.span()[i];
        Scalar gpu = C_gpu.span()[i];
        Scalar abs_err = std::abs(ref - gpu);
        Scalar rel_err = (std::abs(ref) > 1e-12) ? abs_err / std::abs(ref) : 0.0;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_sq_err += abs_err * abs_err;
        sum_sq_ref += ref * ref;
    }

    Scalar rmse = std::sqrt(sum_sq_err / static_cast<Scalar>(N * N));
    Scalar nrmse = (sum_sq_ref > 0) ? rmse / std::sqrt(sum_sq_ref / static_cast<Scalar>(N * N)) : 0.0;

    std::cout << "\n\n  正确性结果:\n"
              << "  ─────────────────────────────────\n"
              << "  最大绝对误差: " << std::scientific << std::setprecision(4) << max_abs_err << "\n"
              << "  最大相对误差: " << max_rel_err << "\n"
              << "  RMSE:         " << rmse << "\n"
              << "  NRMSE:        " << nrmse << "\n";

    // 判定：float32 精度下 NRMSE < 1e-5 为通过
    const bool passed = nrmse < 1e-5;
    std::cout << "  判定: " << (passed ? "✅ 通过" : "❌ 失败") << " (NRMSE < 1e-5)\n\n";

    if (!passed)
    {
        std::cerr << "正确性验证失败，跳过性能测试。\n";
        return 1;
    }

    // ── 4. 性能对比 ───────────────────────────────────────────────
    std::cout << "[4/4] 性能对比 (" << iters << " 次迭代)...\n\n";

    // CPU 计时
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        A.multiply_to(C_cpu, B);
    auto t1 = std::chrono::steady_clock::now();
    Scalar cpu_sec = std::chrono::duration<Scalar>(t1 - t0).count() / iters;

    // GPU 计时（含上传/下载）
    nn::SmartPolicy::gpu_enabled = true;
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        A.multiply_to(C_gpu, B);
    auto t3 = std::chrono::steady_clock::now();
    Scalar gpu_sec = std::chrono::duration<Scalar>(t3 - t2).count() / iters;
    nn::SmartPolicy::gpu_enabled = false;

    // GFLOPS（矩阵乘法：2×N³ FLOPs）
    Scalar flops = 2.0 * static_cast<Scalar>(N) * N * N;
    Scalar cpu_gflops = flops / cpu_sec / 1e9;
    Scalar gpu_gflops = flops / gpu_sec / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  性能结果:\n"
              << "  ─────────────────────────────────────────────\n"
              << "             时间(ms)    吞吐(GFLOPS)    加速比\n"
              << "  ─────────────────────────────────────────────\n"
              << "  CPU:     " << std::setw(10) << cpu_sec * 1000.0
              << "    " << std::setw(10) << cpu_gflops
              << "       1.00x\n"
              << "  GPU:     " << std::setw(10) << gpu_sec * 1000.0
              << "    " << std::setw(10) << gpu_gflops
              << "    " << std::setw(6) << cpu_sec / gpu_sec << "x\n"
              << "  ─────────────────────────────────────────────\n\n";

    std::cout << "测试完成。\n";
    return passed ? 0 : 1;
}
#endif // NN_HAS_VULKAN
