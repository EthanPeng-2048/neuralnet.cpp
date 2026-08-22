// ── engine_factory.hpp — 计算引擎选择工厂 ──────────────────────────────────
//
// 抽取自 mnist_train/mnist_infer/text_train/text_infer/mnist_bench 中重复约 50 行的
// `#ifdef NN_HAS_VULKAN / #ifdef NN_HAS_CUDA` 三段式样板。
//
// 选择优先级：--cuda > --gpu > CPU
//   - CUDA 路径：要求编译期 NN_HAS_CUDA，运行时 backend.initialize() 成功
//   - Vulkan 路径：要求编译期 NN_HAS_VULKAN，运行时 backend.initialize() 成功
//   - **显式请求后端（use_gpu/use_cuda）失败时直接返回错误**，绝不静默回退
//     CPU（保持"硬报错、不降级"；调用方决定是否终止）。
//   - 未显式请求时默认 CpuEngine。
//
// 调用方约束：
//   返回的 engine 必须早于任何持有其非拥有指针的对象（如 nn::Model）析构。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <iosfwd>  // std::ostream 前向声明
#include <memory>

#include "neuralnet.cpp/compute_engine.hpp"
#include "neuralnet.cpp/core_errors.hpp"
#include "neuralnet.cpp/cpu_engine.hpp"
#ifdef NN_HAS_VULKAN
#include "neuralnet.cpp/gpu_engine.hpp"
#endif
#ifdef NN_HAS_CUDA
#include "neuralnet.cpp/cuda_engine.hpp"
#endif

namespace nn::cli
{
    // ── 引擎选择配置 ───────────────────────────────────────────────────────
    struct EngineConfig
    {
        bool use_gpu = false;   // 启用 Vulkan GpuEngine
        bool use_cuda = false;  // 启用 CUDA CudaEngine（优先级高于 use_gpu）
    };

    // ── 创建计算引擎 ───────────────────────────────────────────────────────
    // 根据 use_gpu/use_cuda 选择 GpuEngine/CudaEngine/CpuEngine。
    // 返回 Result：显式请求的后端不可用（初始化失败/未编译）时返回错误，
    // **不回退 CPU**；成功返回引擎并打印选择信息到 log。
    [[nodiscard]] inline nn::Result<std::unique_ptr<nn::ComputeEngine>> create_engine(
        const EngineConfig &cfg, std::ostream &log = std::cerr)
    {
        // 优先级：CUDA > Vulkan > CPU
        if (cfg.use_cuda)
        {
#ifdef NN_HAS_CUDA
            auto &backend = nn::CudaBackend::instance();
            auto init_r = backend.initialize();
            if (init_r)
            {
                log << "CUDA GPU 加速已启用 (" << backend.device_props().name << ")\n\n";
                return std::make_unique<nn::CudaEngine>(backend);
            }
            return std::unexpected(Error{
                "请求 --cuda 但 CUDA 初始化失败: " + init_r.error().message
                + "（不回退 CPU；请移除 --cuda 或改用 CPU 模式）"});
#else
            return std::unexpected(Error{
                "请求 --cuda 但未编译 CUDA 支持（NN_HAS_CUDA）；不回退 CPU"});
#endif
        }

        if (cfg.use_gpu)
        {
#ifdef NN_HAS_VULKAN
            auto &backend = nn::GpuBackend::instance();
            auto init_r = backend.initialize();
            if (init_r)
            {
                log << "GPU 加速已启用 (Vulkan GpuEngine)\n\n";
                return std::make_unique<nn::GpuEngine>(backend);
            }
            return std::unexpected(Error{
                "请求 --gpu 但 GPU 初始化失败: " + init_r.error().message
                + "（不回退 CPU；请移除 --gpu 或改用 CPU 模式）"});
#else
            return std::unexpected(Error{
                "请求 --gpu 但未编译 Vulkan 支持（NN_HAS_VULKAN）；不回退 CPU"});
#endif
        }

        return std::make_unique<nn::CpuEngine>();
    }
} // namespace nn::cli
