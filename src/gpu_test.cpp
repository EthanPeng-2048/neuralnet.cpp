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

    // ── 3b. 转置 matmul 测试 ──────────────────────────────────────
    std::cout << "[3b] 转置 matmul 测试 (A^T * B)..." << std::flush;
    {
        // A^T * B：A 存储为 (N,N)，按 A^T 使用 → C = A^T * B
        // CPU 参考：C_ref = A^T * B
        Matrix A_T(N, N);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                A_T.set_value_unchecked(j, i, A.at_unchecked(i, j));
        Matrix C_ref(N, N);
        A_T.multiply_to(C_ref, B);

        // GPU: matmul_gpu(A, B, transA=1)
        auto gpu_a = GpuTensor::from_matrix(A, backend);
        auto gpu_b = GpuTensor::from_matrix(B, backend);
        if (gpu_a && gpu_b)
        {
            auto gpu_c = backend.matmul_gpu(*gpu_a, *gpu_b, 1u, 0u);
            if (gpu_c)
            {
                auto c_res = gpu_c->to_matrix(backend);
                if (c_res)
                {
                    Scalar max_err = 0.0f;
                    auto rs = c_res->span();
                    auto cs = C_ref.span();
                    for (std::size_t i = 0; i < N * N; ++i)
                    {
                        Scalar diff = std::abs(rs[i] - cs[i]);
                        if (diff > max_err) max_err = diff;
                    }
                    std::cout << "\n  A^T * B 最大误差: " << std::scientific << std::setprecision(4) << max_err
                              << (max_err < 1e-2f ? " ✅" : " ❌") << "\n";
                }
                else std::cout << "\n  ❌ download failed\n";
            }
            else std::cout << "\n  ❌ matmul_gpu failed\n";
        }
        else std::cout << "\n  ❌ upload failed\n";

        // A * B^T
        Matrix B_T(N, N);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                B_T.set_value_unchecked(j, i, B.at_unchecked(i, j));
        Matrix C_ref2(N, N);
        A.multiply_to(C_ref2, B_T);

        auto gpu_b2 = GpuTensor::from_matrix(B, backend);
        if (gpu_a && gpu_b2)
        {
            auto gpu_c2 = backend.matmul_gpu(*gpu_a, *gpu_b2, 0u, 1u);
            if (gpu_c2)
            {
                auto c_res2 = gpu_c2->to_matrix(backend);
                if (c_res2)
                {
                    Scalar max_err = 0.0f;
                    auto rs = c_res2->span();
                    auto cs = C_ref2.span();
                    for (std::size_t i = 0; i < N * N; ++i)
                    {
                        Scalar diff = std::abs(rs[i] - cs[i]);
                        if (diff > max_err) max_err = diff;
                    }
                    std::cout << "  A * B^T 最大误差: " << std::scientific << std::setprecision(4) << max_err
                              << (max_err < 1e-2f ? " ✅" : " ❌") << "\n";
                }
                else std::cout << "  ❌ download failed\n";
            }
            else std::cout << "  ❌ matmul_gpu failed\n";
        }
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
        // 5a. 纯 upload→download roundtrip 测试
        auto gpu_a = GpuTensor::from_matrix(A, backend);
        if (gpu_a)
        {
            auto a_back = gpu_a->to_matrix(backend);
            if (a_back)
            {
                Scalar max_err = 0.0f;
                auto as = A.span();
                auto bs = a_back->span();
                for (std::size_t i = 0; i < N * N; ++i)
                {
                    Scalar diff = std::abs(as[i] - bs[i]);
                    if (diff > max_err) max_err = diff;
                }
                std::cout << "  Upload→Download roundtrip 最大误差: "
                          << std::scientific << std::setprecision(4) << max_err
                          << (max_err == 0.0f ? " ✅" : " ❌") << "\n";
            }
            else
                std::cout << "  ❌ roundtrip download 失败: " << a_back.error().message << "\n";
        }
        else
            std::cout << "  ❌ roundtrip upload 失败\n";

        // 5b. 单次 matmul GPU-resident
        auto gpu_b = GpuTensor::from_matrix(B, backend);
        if (gpu_a && gpu_b)
        {
            auto gpu_c = backend.matmul_gpu(*gpu_a, *gpu_b);
            if (gpu_c)
            {
                auto c_res = gpu_c->to_matrix(backend);
                if (c_res)
                {
                    Scalar max_err = 0.0f;
                    auto c_span = c_res->span();
                    for (std::size_t i = 0; i < N * N; ++i)
                    {
                        Scalar diff = std::abs(cpu_span[i] - c_span[i]);
                        if (diff > max_err) max_err = diff;
                    }
                    std::cout << "  单次 matmul 最大误差: " << std::scientific << std::setprecision(4) << max_err
                              << (max_err < 1e-2f ? " ✅" : " ❌") << "\n";
                }
                else
                    std::cout << "  ❌ 下载失败: " << c_res.error().message << "\n";
            }
            else
                std::cout << "  ❌ matmul_gpu 失败: " << gpu_c.error().message << "\n";
        }
    }

    // ── 6. 多层链式测试 ──────────────────────────────────────────
    std::cout << "\n[6/6] 多层链式 GPU-resident 测试...\n";
    {
        // 6a. 测试 batch mode + 真实 GPU 操作（非 fallback）
        {
            auto gpu_a = GpuTensor::from_matrix(A, backend);
            auto gpu_b = GpuTensor::from_matrix(B, backend);
            if (gpu_a && gpu_b)
            {
                auto batch_r = backend.begin_batch();
                if (batch_r)
                {
                    auto gpu_c = backend.matmul_gpu(*gpu_a, *gpu_b);
                    if (gpu_c)
                    {
                        auto end_r = backend.end_batch();
                        if (end_r)
                        {
                            auto c_res = gpu_c->to_matrix(backend);
                            if (c_res)
                            {
                                Scalar max_err = 0.0f;
                                auto cs = c_res->span();
                                for (std::size_t i = 0; i < N * N; ++i)
                                {
                                    Scalar diff = std::abs(cpu_span[i] - cs[i]);
                                    if (diff > max_err) max_err = diff;
                                }
                                std::cout << "  Batch+matmul_gpu 最大误差: "
                                          << std::scientific << std::setprecision(4) << max_err
                                          << (max_err < 1e-2f ? " ✅" : " ❌") << "\n";
                            }
                            else std::cout << "  ❌ download failed: " << c_res.error().message << "\n";
                        }
                        else std::cout << "  ❌ end_batch failed: " << end_r.error().message << "\n";
                    }
                    else std::cout << "  ❌ matmul_gpu failed: " << gpu_c.error().message << "\n";
                }
                else std::cout << "  ❌ begin_batch failed: " << batch_r.error().message << "\n";
            }
        }

        // 6b. 测试 batch mode + 链式 matmul + elementwise
        {
            // A(256×256) * B(256×128) = C(256×128), then ReLU(C)
            Matrix B2(N, N/2);
            for (auto& v : B2.span()) v = dist(rng);
            Matrix C_cpu_ref(N, N/2);
            A.multiply_to(C_cpu_ref, B2);
            // ReLU
            for (auto& v : C_cpu_ref.span()) v = std::max(v, 0.0f);

            auto gpu_a = GpuTensor::from_matrix(A, backend);
            auto gpu_b2 = GpuTensor::from_matrix(B2, backend);
            if (gpu_a && gpu_b2)
            {
                auto batch_r = backend.begin_batch();
                if (batch_r)
                {
                    auto gpu_c = backend.matmul_gpu(*gpu_a, *gpu_b2);
                    if (gpu_c)
                    {
                        auto gpu_relu = backend.elementwise_gpu(*gpu_c, nullptr, 0u);
                        if (gpu_relu)
                        {
                            auto end_r = backend.end_batch();
                            if (end_r)
                            {
                                auto r_res = gpu_relu->to_matrix(backend);
                                if (r_res)
                                {
                                    Scalar max_err = 0.0f;
                                    auto cs = C_cpu_ref.span();
                                    auto rs = r_res->span();
                                    for (std::size_t i = 0; i < cs.size(); ++i)
                                    {
                                        Scalar diff = std::abs(cs[i] - rs[i]);
                                        if (diff > max_err) max_err = diff;
                                    }
                                    std::cout << "  Batch matmul+ReLU 最大误差: "
                                              << std::scientific << std::setprecision(4) << max_err
                                              << (max_err < 1e-2f ? " ✅" : " ❌") << "\n";
                                }
                                else std::cout << "  ❌ download failed\n";
                            }
                            else std::cout << "  ❌ end_batch failed\n";
                        }
                        else std::cout << "  ❌ elementwise failed\n";
                    }
                    else std::cout << "  ❌ matmul_gpu failed\n";
                }
                else std::cout << "  ❌ begin_batch failed\n";
            }
        }

        // 6c. Model::forward 多层链式
        nn::Model model;
        model.add<nn::Linear>(256, 128);
        model.add<nn::ReLU>();
        model.add<nn::Linear>(128, 64);

        Matrix input(256, 32);
        for (auto& v : input.span()) v = dist(rng);

        SmartPolicy::gpu_enabled = false;
        auto cpu_out = model.forward(input);

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
            std::cout << "  Model::forward 多层 最大误差: "
                      << std::scientific << std::setprecision(4) << max_err
                      << (max_err < 1e-1f ? " ✅" : " ❌") << "\n";
        }
    }

    std::cout << "\n========================================\n"
              << "  测试完成\n"
              << "========================================\n";

    return 0;
}
#endif
