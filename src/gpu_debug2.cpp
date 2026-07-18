// GPU 调试 v2：测试训练尺寸的 matmul + bias + elementwise 链
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

    // 模拟一层: W(512,784) × input(784,600) + bias(512,1) → GeLU → result
    const std::size_t in_f = 784, out_f = 512, batch = 600;
    
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-0.1f, 0.1f);
    
    nn::Matrix W(out_f, in_f);
    nn::Matrix input_mat(in_f, batch);
    nn::Matrix bias(out_f, 1);
    
    for (std::size_t i = 0; i < W.size(); ++i) W.data()[i] = dist(rng);
    for (std::size_t i = 0; i < input_mat.size(); ++i) input_mat.data()[i] = dist(rng);
    for (std::size_t i = 0; i < bias.size(); ++i) bias.data()[i] = dist(rng);

    // ── CPU 参考 ──
    nn::Matrix cpu_prod(out_f, batch);
    W.multiply_to(cpu_prod, input_mat);
    nn::Matrix cpu_biased = cpu_prod;
    cpu_biased.add_bias_broadcast_inplace(bias);
    // GeLU on CPU
    for (std::size_t i = 0; i < cpu_biased.size(); ++i) {
        Scalar x = cpu_biased.data()[i];
        cpu_biased.data()[i] = x / (1.0f + std::exp(-1.702f * x));
    }

    // ── GPU 完整链路 ──
    W.ensure_gpu();
    input_mat.ensure_gpu();
    bias.ensure_gpu();
    
    nn::Matrix gpu_prod(out_f, batch);
    W.multiply_to(gpu_prod, input_mat);  // GPU-resident matmul
    gpu_prod.add_bias_broadcast_inplace(bias);  // GPU-resident bias
    gpu_prod.elementwise_inplace(1);  // GPU-resident GeLU
    
    // 检查中间结果
    gpu_prod.flush_gpu_to_cpu();
    
    Scalar max_err = 0;
    for (std::size_t i = 0; i < cpu_biased.size(); ++i) {
        Scalar err = std::abs(gpu_prod.data()[i] - cpu_biased.data()[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "Max error: " << max_err << "\n";
    if (max_err < 0.1f)
        std::cout << "✓ GPU chain matches CPU!\n";
    else
        std::cout << "✗ GPU chain MISMATCH!\n";

    // 详细检查前几个元素
    std::cout << "First 5 elements:\n";
    for (int i = 0; i < 5; ++i)
        std::cout << "  CPU=" << cpu_biased.data()[i] << " GPU=" << gpu_prod.data()[i] << "\n";

    return 0;
}
#endif
