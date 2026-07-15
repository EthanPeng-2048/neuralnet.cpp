#ifndef NN_HPP
#define NN_HPP

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "nn_config.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "model.hpp"
#include "optimizer.hpp"
#include "mnist_common.hpp"

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