// GPU-resident 调试 v6: 测试完整前向链路
#include <neuralnet.cpp/nn.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#ifndef NN_HAS_VULKAN
int main() { std::cout << "[SKIP]\n"; return 0; }
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>
using namespace nn;

void cmp(const char* label, const Matrix& a, const Matrix& b, float tol = 0.01f) {
    float max_err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        max_err = std::max(max_err, std::abs(a.data()[i] - b.data()[i]));
    std::cout << label << ": max_err=" << max_err << (max_err < tol ? " ✓" : " ✗") << "\n";
}

int main() {
    auto& backend = GpuBackend::instance();
    if (!backend.initialize()) { std::cerr << "GPU init fail\n"; return 1; }
    SmartPolicy::gpu_enabled = true;

    // 模拟单层 Linear(784→512) + bias + GeLU
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    
    Matrix W(512, 784), X(784, 600), b(512, 1);
    for (auto& v : W.data()) v = dist(rng);
    for (auto& v : X.data()) v = dist(rng);
    for (auto& v : b.data()) v = dist(rng);

    // GPU-resident 链路
    W.ensure_gpu(); X.ensure_gpu();
    Matrix gpu_out(512, 600);
    W.multiply_to(gpu_out, X);  // GPU matmul
    gpu_out.add_bias_broadcast_inplace(b);  // GPU bias
    gpu_out.elementwise_inplace(1);  // GPU GeLU
    gpu_out.flush_gpu_to_cpu();

    // CPU 参考（禁用 GPU）
    SmartPolicy::gpu_enabled = false;
    Matrix cpu_out(512, 600);
    W.multiply_to(cpu_out, X);
    cpu_out.add_bias_broadcast_inplace(b);
    // GeLU on CPU
    for (auto& v : cpu_out.data()) {
        float x = v;
        v = x / (1.f + std::exp(-1.702f * x));
    }
    SmartPolicy::gpu_enabled = true;

    cmp("Single layer: GPU-resident vs CPU", gpu_out, cpu_out);

    // 测试两层链路
    std::cout << "\n═══ Two-layer chain ═══\n";
    Matrix W2(256, 512), b2(256, 1);
    for (auto& v : W2.data()) v = dist(rng);
    for (auto& v : b2.data()) v = dist(rng);

    // GPU chain
    W2.ensure_gpu();
    Matrix gpu_out2(256, 600);
    W2.multiply_to(gpu_out2, gpu_out);  // gpu_out has GPU shadow from GeLU
    gpu_out2.add_bias_broadcast_inplace(b2);
    gpu_out2.elementwise_inplace(1);  // GeLU
    gpu_out2.flush_gpu_to_cpu();

    // CPU chain
    SmartPolicy::gpu_enabled = false;
    Matrix cpu_out2(256, 600);
    W2.multiply_to(cpu_out2, cpu_out);
    cpu_out2.add_bias_broadcast_inplace(b2);
    for (auto& v : cpu_out2.data()) { float x = v; v = x / (1.f + std::exp(-1.702f * x)); }
    SmartPolicy::gpu_enabled = true;

    cmp("Two-layer: GPU-resident vs CPU", gpu_out2, cpu_out2);

    // 测试 backward
    std::cout << "\n═══ Backward pass ═══\n";
    Matrix grad(256, 600);
    for (auto& v : grad.data()) v = dist(rng);

    // GPU backward (flush → CPU computation)
    gpu_out2.flush_gpu_to_cpu();
    gpu_out.flush_gpu_to_cpu();
    // Manual backward for layer 2: grad_input = W2^T * grad
    Matrix W2T(512, 256);
    W2.transpose_to(W2T);
    Matrix gpu_grad_in2(512, 600);
    W2T.multiply_to(gpu_grad_in2, grad);

    // CPU backward
    Matrix cpu_grad_in2(512, 600);
    W2.transpose_to(W2T);
    W2T.multiply_to(cpu_grad_in2, grad);
    cmp("Backward grad: GPU vs CPU", gpu_grad_in2, cpu_grad_in2);

    // 打印前几个元素
    for (int i = 0; i < 8; ++i)
        std::cout << "  [" << i << "] GPU=" << gpu_out.data()[i] << " CPU=" << cpu_out.data()[i] << "\n";

    return 0;
}
#endif
