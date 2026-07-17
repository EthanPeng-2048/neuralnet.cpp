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
    };

} // namespace nn

#endif // LAYER_BASE_HPP
