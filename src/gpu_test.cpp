// ── GPU 矩阵乘法正确性与性能测试 ──────────────────────────────────────────
// 用法：gpu_test [--size N] [--iters N]
//   --size N   矩阵维度（默认 256，即 256×256 方阵）
//   --iters N  性能测试迭代次数（默认 10）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::Matrix;
using nn::GpuBackend;
using nn::GpuTensor;
using nn::SmartPolicy;

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
    auto& backend = GpuBackend::instance();
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
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    Matrix A(N, N), B(N, N);
    for (auto& v : A.span()) v = dist(rng);
    for (auto& v : B.span()) v = dist(rng);
    std::cout << " 完成\n";

    // ── 3. 正确性验证 ─────────────────────────────────────────────
    std::cout << "[3/4] 正确性验证 (CPU vs GPU)..." << std::flush;

    // CPU 参考结果
    Matrix C_cpu(N, N);
    A.multiply_to(C_cpu, B);

    // GPU 结果
    Matrix C_gpu(N, N);
    SmartPolicy::gpu_enabled = true;
    A.multiply_to(C_gpu, B);
    SmartPolicy::gpu_enabled = false;

    // 计算最大绝对误差和相对误差
    Scalar max_abs_err = 0.0f;
    Scalar max_rel_err = 0.0f;
    Scalar sum_sq_err = 0.0f;
    Scalar sum_sq_ref = 0.0f;

    auto cpu_span = C_cpu.span();
    auto gpu_span = C_gpu.span();

    for (std::size_t i = 0; i < N * N; ++i)
    {
        Scalar diff = std::abs(cpu_span[i] - gpu_span[i]);
        Scalar ref = std::abs(cpu_span[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        if (ref > 1e-6f)
        {
            Scalar rel = diff / ref;
            if (rel > max_rel_err) max_rel_err = rel;
        }
        sum_sq_err += diff * diff;
        sum_sq_ref += ref * ref;
    }

    Scalar rmse = std::sqrt(sum_sq_err / (N * N));
    Scalar nrmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0f;

    std::cout << "\n";
    std::cout << "  最大绝对误差: " << std::scientific << std::setprecision(4) << max_abs_err << "\n";
    std::cout << "  最大相对误差: " << std::scientific << std::setprecision(4) << max_rel_err << "\n";
    std::cout << "  RMSE:         " << std::scientific << std::setprecision(4) << rmse << "\n";
    std::cout << "  NRMSE:        " << std::scientific << std::setprecision(4) << nrmse << "\n";

    const bool passed = (max_abs_err < 1e-2f) && (nrmse < 1e-3f);
    std::cout << "  结果: " << (passed ? "✅ 通过" : "❌ 失败") << "\n\n";

    if (!passed)
    {
        std::cerr << "正确性验证失败，跳过性能测试。\n";
        return 1;
    }

    // ── 4. 性能测试 ───────────────────────────────────────────────
    std::cout << "[4/4] 性能测试 (" << iters << " 次迭代)...\n";

    // CPU 性能
    Matrix C_tmp(N, N);
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        A.multiply_to(C_tmp, B);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count() / iters;

    // GPU 性能（staging path）
    SmartPolicy::gpu_enabled = true;
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        A.multiply_to(C_tmp, B);
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count() / iters;
    SmartPolicy::gpu_enabled = false;

    double speedup = cpu_ms / gpu_ms;
    double gflops = (2.0 * N * N * N) / (gpu_ms * 1e6);  // 2*N^3 FLOPs

    std::cout << "\n";
    std::cout << "  CPU 时间:  " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU 时间:  " << std::fixed << std::setprecision(2) << gpu_ms << " ms\n";
    std::cout << "  加速比:    " << std::fixed << std::setprecision(2) << speedup << "×\n";
    std::cout << "  GPU GFLOPS: " << std::fixed << std::setprecision(1) << gflops << "\n";
    std::cout << "\n";

    // ── 5. GPU-resident 路径测试 ──────────────────────────────────
    std::cout << "[5/5] GPU-resident 路径测试...\n";
    {
        // 上传 A, B 到 GPU
        auto gpu_a = GpuTensor::from_matrix(A, backend);
        auto gpu_b = GpuTensor::from_matrix(B, backend);

        if (gpu_a && gpu_b)
        {
            // 在 GPU 上计算
            auto gpu_c = backend.matmul_gpu(*gpu_a, *gpu_b);
            if (gpu_c)
            {
                // 下载结果
                auto c_res = gpu_c->to_matrix(backend);
                if (c_res)
                {
                    // 验证
                    Scalar max_err = 0.0f;
                    auto c_span = c_res->span();
                    for (std::size_t i = 0; i < N * N; ++i)
                    {
                        Scalar diff = std::abs(cpu_span[i] - c_span[i]);
                        if (diff > max_err) max_err = diff;
                    }
                    std::cout << "  最大绝对误差: " << std::scientific << std::setprecision(4) << max_err << "\n";
                    std::cout << "  结果: " << (max_err < 1e-2f ? "✅ 通过" : "❌ 失败") << "\n";
                }
                else
                    std::cout << "  ❌ 下载失败: " << c_res.error().message << "\n";
            }
            else
                std::cout << "  ❌ matmul_gpu 失败: " << gpu_c.error().message << "\n";
        }
        else
            std::cout << "  ❌ 上传失败\n";
    }

    // ── 6. 多层链式测试 ──────────────────────────────────────────
    std::cout << "\n[6/6] 多层链式 GPU-resident 测试...\n";
    {
        // 构建 2 层网络: Linear(256→128) → ReLU → Linear(128→64)
        nn::Model model;
        model.add<nn::Linear>(256, 128);
        model.add<nn::ReLU>();
        model.add<nn::Linear>(128, 64);

        // 随机输入
        Matrix input(256, 32);  // 256 features, 32 batch
        for (auto& v : input.span()) v = dist(rng);

        // CPU 前向
        SmartPolicy::gpu_enabled = false;
        auto cpu_out = model.forward(input);

        // GPU-resident 前向
        SmartPolicy::gpu_enabled = true;
        auto gpu_out = model.forward(input);
        SmartPolicy::gpu_enabled = false;

        if (cpu_out && gpu_out)
        {
            Scalar max_err = 0.0f;
            auto cs = cpu_out->span();
            auto gs = gpu_out->span();
            for (std::size_t i = 0; i < cs.size(); ++i)
            {
                Scalar diff = std::abs(cs[i] - gs[i]);
                if (diff > max_err) max_err = diff;
            }
            std::cout << "  输出维度: " << gpu_out->rows() << "×" << gpu_out->cols() << "\n";
            std::cout << "  最大绝对误差: " << std::scientific << std::setprecision(4) << max_err << "\n";
            std::cout << "  结果: " << (max_err < 1e-1f ? "✅ 通过" : "❌ 失败") << "\n";
        }
        else
        {
            if (!cpu_out) std::cout << "  ❌ CPU 前向失败: " << cpu_out.error().message << "\n";
            if (!gpu_out) std::cout << "  ❌ GPU 前向失败: " << gpu_out.error().message << "\n";
        }
    }

    std::cout << "\n========================================\n"
              << "  测试完成\n"
              << "========================================\n";

    return 0;
}
#endif
