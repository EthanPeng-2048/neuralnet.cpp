#ifndef NN_HPP
#define NN_HPP

// ── neuralnet.cpp 统一入口 ────────────────────────────────────────────────
// 包含所有核心模块，用户只需 #include "nn.hpp"

// Core utilities
#include "core/errors.hpp"
#include "core/assert.hpp"
#include "core/thread_pool.hpp"

// Algebra layer (L1)
#include "algebra/matrix.hpp"

// Backend (L0) - optional Vulkan support
#ifdef NN_HAS_VULKAN
#include "backend/vk_backend.hpp"
#endif

// Computation layer (L2)
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

// Model layer (L3)
#include "model.hpp"
#include "model_io.hpp"
#include "model_spec.hpp"

// Domain-specific
#include "mnist_common.hpp"
#include "gpt_common.hpp"
#include "tokenizer.hpp"

namespace nn
{
    [[nodiscard]] inline Result<Matrix> one_hot(const std::vector<std::size_t> &true_i, std::size_t mat_size)
    {
        const std::size_t batch_size = true_i.size();
        Matrix result(mat_size, batch_size);

        for (std::size_t i = 0; i < batch_size; ++i)
        {
            if (true_i[i] >= mat_size)
            {
                return std::unexpected(Error{"one_hot index out of range"});
            }
            result.set_value_unchecked(true_i[i], i, 1.0);
        }

        return result;
    }
} // namespace nn

#endif // NN_HPP