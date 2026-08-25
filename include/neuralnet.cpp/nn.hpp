#ifndef NN_HPP
#define NN_HPP

// ── neuralnet.cpp 统一入口 ────────────────────────────────────────────────
// 包含所有核心模块，用户只需 #include "nn.hpp"
//
// 包含顺序按依赖关系排列：L0 → L1 → L2 → L3 → L4
// 注意：algebra_matrix.hpp 已传递包含 algebra_span/ops/expr/compute.hpp，
//       config.hpp 已传递包含 core_errors.hpp，
//       此处显式列出所有头文件是为了清晰展示模块结构。
//
// 新架构（引擎化）：
//   ComputeEngine 抽象硬件接触，Layer/Loss/Optimizer 通过组合原语表达算法，
//   forward/backward 单套实现适配 CPU/GPU。

// L0 硬件层
#include "core_errors.hpp"
#include "core_assert.hpp"
#include "core_observer_ptr.hpp"
#include "core_threadpool.hpp"
#include "core_file.hpp"
#include "config.hpp"

// L1 代数层（algebra_matrix.hpp 已传递包含其余代数头文件）
#include "algebra_matrix.hpp"

// L2 计算层 — 引擎化
#include "tensor.hpp"
#include "expr_spec.hpp"
#include "compute_engine.hpp"
#include "expr_dsl.hpp"
#include "cpu_engine.hpp"
#ifdef NN_HAS_VULKAN
#include "gpu_engine.hpp"
#endif
#ifdef NN_HAS_CUDA
#include "cuda_engine.hpp"
#endif
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
#include "domain_zipt.hpp"
#include "domain_cnn.hpp"
#include "domain_tokenizer.hpp"

namespace nn
{
    // one_hot 工具：将类别索引向量转为 one-hot 矩阵（列主序，每列一个样本）
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
