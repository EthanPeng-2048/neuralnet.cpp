#ifndef LAYER_BASE_HPP
#define LAYER_BASE_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <execution>
#include <functional>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "../algebra/matrix.hpp"
#include "../nn_config.hpp"

namespace nn
{
    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual Result<Matrix> forward(const Matrix &input) = 0;
        virtual Result<Matrix> backward(const Matrix &grad_output) = 0;
        virtual std::vector<std::reference_wrapper<Matrix>> parameters() { return {}; }
        virtual std::vector<std::reference_wrapper<Matrix>> param_gradients() { return {}; }

#ifdef NN_HAS_VULKAN
        // ── GPU 路径：默认实现 Fallback 到 CPU ───────────────────────
        // 子类可覆盖以提供纯 GPU 实现（如 Linear）
        virtual Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend)
        {
            // Fallback：Download → CPU 计算 → Upload
            auto cpu_in_res = input.to_matrix(backend);
            if (!cpu_in_res) return std::unexpected(cpu_in_res.error());

            auto cpu_out_res = forward(*cpu_in_res);
            if (!cpu_out_res) return std::unexpected(cpu_out_res.error());

            return GpuTensor::from_matrix(*cpu_out_res, backend);
        }
#endif
    };

} // namespace nn

#endif // LAYER_BASE_HPP
