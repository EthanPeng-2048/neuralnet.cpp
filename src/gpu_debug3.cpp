// GPU 调试 v3：测试训练尺寸的完整 GPU 链路
#include <neuralnet.cpp/nn.hpp>
using nn::Scalar;
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

#ifndef NN_HAS_VULKAN
int main() { std::cout << "[SKIP] Vulkan not available\n"; return 0; }
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

int main() {
    auto& backend = nn::GpuBackend::instance();
    if (!backend.initialize()) { std::cerr << "GPU init failed\n"; return 1; }
    nn::SmartPolicy::gpu_enabled = true;

    const std::size_t in_f = 784, out_f = 512, batch = 600;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-0.1f, 0.1f);
    
    nn::Matrix W(out_f, in_f);
    nn::Matrix X(in_f, batch);
    for (std::size_t i = 0; i < W.size(); ++i) W.data()[i] = dist(rng);
    for (std::size_t i = 0; i < X.size(); ++i) X.data()[i] = dist(rng);

    // CPU 参考
    nn::Matrix cpu_ref(out_f, batch);
    W.multiply_to(cpu_ref, X);  // CPU (M*N = 307200 > 1024, BUT gpu_enabled was just set...)
    // Actually multiply_to will try GPU but ensure_gpu not called, so staging will be used
    // Wait, ensure_gpu IS called now (my fix). So GPU might be used.
    // Let's compute pure CPU by temporarily disabling GPU:
    nn::SmartPolicy::gpu_enabled = false;
    nn::Matrix cpu_ref2(out_f, batch);
    W.multiply_to(cpu_ref2, X);
    nn::SmartPolicy::gpu_enabled = true;
    
    // Compare: cpu_ref (GPU staging) vs cpu_ref2 (pure CPU)
    Scalar max_err = 0;
    for (std::size_t i = 0; i < cpu_ref.size(); ++i)
        max_err = std::max(max_err, std::abs(cpu_ref.data()[i] - cpu_ref2.data()[i]));
    std::cout << "Staging vs CPU max error: " << max_err << "\n";
    if (max_err < 0.01f) std::cout << "✓ Staging matches CPU\n";
    else std::cout << "✗ Staging MISMATCH!\n";

    // Now test GPU-resident path
    W.ensure_gpu();
    X.ensure_gpu();
    std::cout << "W has GPU: " << W.has_gpu_shadow() << " X has GPU: " << X.has_gpu_shadow() << "\n";
    
    if (W.has_gpu_shadow() && X.has_gpu_shadow()) {
        // Direct matmul_gpu_resident
        auto gpu_res = backend.matmul_gpu_resident(W.gpu_tensor(), X.gpu_tensor());
        if (gpu_res) {
            auto mat = gpu_res->to_matrix(backend);
            if (mat) {
                Scalar err = 0;
                for (std::size_t i = 0; i < mat->size(); ++i)
                    err = std::max(err, std::abs(mat->data()[i] - cpu_ref2.data()[i]));
                std::cout << "matmul_gpu_resident vs CPU max error: " << err << "\n";
                if (err < 0.01f) std::cout << "✓ matmul_gpu_resident correct!\n";
                else {
                    std::cout << "✗ matmul_gpu_resident WRONG!\n";
                    // Show first few mismatches
                    for (std::size_t i = 0; i < 10 && i < mat->size(); ++i)
                        if (std::abs(mat->data()[i] - cpu_ref2.data()[i]) > 0.01f)
                            std::cout << "  [" << i << "] GPU=" << mat->data()[i] << " CPU=" << cpu_ref2.data()[i] << "\n";
                }
            }
        }
        
        // Through multiply_to
        nn::Matrix gpu_via_mul(out_f, batch);
        W.multiply_to(gpu_via_mul, X);
        gpu_via_mul.flush_gpu_to_cpu();
        Scalar err2 = 0;
        for (std::size_t i = 0; i < gpu_via_mul.size(); ++i)
            err2 = std::max(err2, std::abs(gpu_via_mul.data()[i] - cpu_ref2.data()[i]));
        std::cout << "multiply_to (GPU-resident) vs CPU max error: " << err2 << "\n";
        if (err2 < 0.01f) std::cout << "✓ multiply_to GPU-resident correct!\n";
        else std::cout << "✗ multiply_to GPU-resident WRONG!\n";
    }
    
    nn::SmartPolicy::print_matmul_stats();
    return 0;
}
#endif
