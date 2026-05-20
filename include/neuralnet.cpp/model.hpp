#ifndef MODEL_HPP
#define MODEL_HPP

#include "matrix.hpp"
#include <fstream>
#include <iostream>

namespace nn
{
    class Model
    {
    private:
    
    public : Model() = default;
        Model(const Model &) = default;
        Model(Model &&) noexcept = default;
        Model &operator=(const Model &) = default;
        Model &operator=(Model &&) noexcept = default;
    };
} // namespace nn

#endif // MODEL_HPP