// ── offload_test.cpp — activation offload 原语往返验证 ────────────────────
//
// 目的：验证 GpuEngine::offload_store / offload_load 的往返正确性：
//   GPU 激活 → host-visible 存储 → 拷回 GPU → 与原始数据一致。
// 覆盖两条路径：
//   1. 非 batch（每次独立提交等待）
//   2. batch 录制（begin_batch 内 store/load，end_batch 统一提交）——训练真实路径
//
// 用法：offload_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>

using nn::Matrix;
using nn::Scalar;
using nn::Tensor;

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持（NN_HAS_VULKAN）。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::GpuBackend;
using nn::GpuEngine;

namespace
{

Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    Scalar max_err = 0.0f;
    const auto& as = a.span();
    const auto& bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar diff = std::fabs(as[i] - bs[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int run_roundtrip(GpuEngine& gpu, std::size_t rows, std::size_t cols,
                  bool use_batch, const std::string& tag)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix m(rows, cols);
    for (auto& v : m.span()) v = dist(rng);

    auto up = gpu.from_matrix(m);
    if (!up) { std::cerr << "  from_matrix 失败: " << up.error().message << "\n"; return 1; }

    Tensor handle, restored;
    if (use_batch)
    {
        auto bb = gpu.begin_batch();
        if (!bb) { std::cerr << "  begin_batch 失败\n"; return 1; }
        auto st = gpu.offload_store(*up);
        if (!st) { std::cerr << "  offload_store 失败: " << st.error().message << "\n"; return 1; }
        handle = std::move(*st);
        auto ld = gpu.offload_load(handle, rows, cols);
        if (!ld) { std::cerr << "  offload_load 失败: " << ld.error().message << "\n"; return 1; }
        restored = std::move(*ld);
        auto eb = gpu.end_batch();
        if (!eb) { std::cerr << "  end_batch 失败: " << eb.error().message << "\n"; return 1; }
    }
    else
    {
        auto st = gpu.offload_store(*up);
        if (!st) { std::cerr << "  offload_store 失败: " << st.error().message << "\n"; return 1; }
        handle = std::move(*st);
        auto ld = gpu.offload_load(handle, rows, cols);
        if (!ld) { std::cerr << "  offload_load 失败: " << ld.error().message << "\n"; return 1; }
        restored = std::move(*ld);
    }

    auto dl = gpu.to_matrix(restored);
    if (!dl) { std::cerr << "  to_matrix 失败\n"; return 1; }
    const Scalar err = max_abs_diff(m, *dl);
    const bool ok = err < 1e-6f;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] offload roundtrip "
              << tag << " " << rows << "x" << cols
              << "  max_err=" << err << "\n";
    return ok ? 0 : 1;
}

} // namespace

int main()
{
    std::cout << "========================================\n"
              << "  activation offload 原语往返验证\n"
              << "========================================\n";
    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    GpuEngine gpu(backend);

    int fail = 0;
    // 非 batch 路径
    fail += run_roundtrip(gpu, 128, 128, false, "non-batch");
    fail += run_roundtrip(gpu, 1024, 512, false, "non-batch");
    // batch 录制路径（训练真实路径）
    fail += run_roundtrip(gpu, 128, 128, true, "batch");
    fail += run_roundtrip(gpu, 1024, 512, true, "batch");

    std::cout << (fail == 0 ? "ALL PASS\n" : "FAILED\n");
    return fail == 0 ? 0 : 1;
}
#endif
