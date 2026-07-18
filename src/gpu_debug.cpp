// GPU 调试：对比 GPU-resident matmul 与 CPU matmul 结果
#include <neuralnet.cpp/nn.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
#ifdef NN_HAS_VULKAN
    auto& backend = nn::GpuBackend::instance();
    auto init = backend.initialize();
    if (!init) {
        std::cerr << "GPU init failed: " << init.error().message << '\n';
        return 1;
    }
    nn::SmartPolicy::gpu_enabled = true;

    // 测试小矩阵乘法
    const std::size_t M = 4, K = 3, N = 2;
    nn::Matrix A(M, K);
    nn::Matrix B(K, N);

    // 填充已知值
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < K; ++j)
            A.set_value_unchecked(i, j, static_cast<nn::Scalar>(i * K + j + 1));
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < N; ++j)
            B.set_value_unchecked(i, j, static_cast<nn::Scalar>(i * N + j + 1));

    // CPU 参考结果
    nn::Matrix cpu_result(M, N);
    A.multiply_to(cpu_result, B);
    std::cout << "CPU result:\n";
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::cout << std::setw(10) << cpu_result.at_unchecked(i, j) << " ";
        std::cout << "\n";
    }

    // GPU 路径（确保影子后 multiply_to）
    A.ensure_gpu();
    B.ensure_gpu();
    nn::Matrix gpu_result(M, N);
    // 直接测试 matmul_gpu_resident
    if (A.has_gpu_shadow() && B.has_gpu_shadow()) {
        auto gpu_t = backend.matmul_gpu_resident(A.gpu_tensor(), B.gpu_tensor());
        if (gpu_t) {
            auto mat = gpu_t->to_matrix(backend);
            if (mat) {
                std::cout << "\nGPU result (direct):\n";
                for (std::size_t i = 0; i < M; ++i) {
                    for (std::size_t j = 0; j < N; ++j)
                        std::cout << std::setw(10) << mat->at_unchecked(i, j) << " ";
                    std::cout << "\n";
                }
                // 对比
                bool match = true;
                for (std::size_t i = 0; i < M && match; ++i)
                    for (std::size_t j = 0; j < N; ++j)
                        if (std::abs(mat->at_unchecked(i, j) - cpu_result.at_unchecked(i, j)) > 0.01f) {
                            std::cout << "MISMATCH at (" << i << "," << j << "): GPU="
                                      << mat->at_unchecked(i, j) << " CPU=" << cpu_result.at_unchecked(i, j) << "\n";
                            match = false;
                        }
                if (match) std::cout << "\n✓ GPU matches CPU!\n";
            }
        } else {
            std::cout << "GPU matmul failed: " << gpu_t.error().message << "\n";
        }
    }

    // 通过 multiply_to 走 GPU-resident 路径
    A.ensure_gpu();
    B.ensure_gpu();
    nn::Matrix gpu_result2(M, N);
    A.multiply_to(gpu_result2, B);  // 应该自动走 GPU-resident
    gpu_result2.flush_gpu_to_cpu();
    std::cout << "\nGPU result (multiply_to):\n";
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j)
            std::cout << std::setw(10) << gpu_result2.at_unchecked(i, j) << " ";
        std::cout << "\n";
    }
    bool match2 = true;
    for (std::size_t i = 0; i < M && match2; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (std::abs(gpu_result2.at_unchecked(i, j) - cpu_result.at_unchecked(i, j)) > 0.01f) {
                std::cout << "MISMATCH at (" << i << "," << j << "): GPU="
                          << gpu_result2.at_unchecked(i, j) << " CPU=" << cpu_result.at_unchecked(i, j) << "\n";
                match2 = false;
            }
    if (match2) std::cout << "\n✓ GPU multiply_to matches CPU!\n";

    nn::SmartPolicy::print_matmul_stats();
#else
    std::cout << "Vulkan not available\n";
#endif
    return 0;
}
