#ifndef NN_HPP
#define NN_HPP

// ── neuralnet.cpp 统一入口 ────────────────────────────────────────────────
// 包含所有核心模块，用户只需 #include "nn.hpp"
//
// 包含顺序按依赖关系排列：L0 → L1 → L2 → L3 → L4
// 注意：algebra_matrix.hpp 已传递包含 algebra_span/ops/expr/compute.hpp，
//       config.hpp 已传递包含 core_errors.hpp，
//       此处显式列出所有头文件是为了清晰展示模块结构。

// L0 硬件层
#include "core_errors.hpp"
#include "core_assert.hpp"
#include "core_threadpool.hpp"
#include "config.hpp"

// L1 代数层（algebra_matrix.hpp 已传递包含其余代数头文件）
#include "algebra_matrix.hpp"

// L2 计算层
#include "compute_layer.hpp"
#include "compute_loss.hpp"
#include "compute_optimizer.hpp"

// L3 实现层
#include "model_container.hpp"
#include "model_spec.hpp"
#include "model_serialization.hpp"

// L4 构建层
#include "domain_mnist.hpp"
#include "domain_gpt.hpp"
#include "domain_tokenizer.hpp"

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