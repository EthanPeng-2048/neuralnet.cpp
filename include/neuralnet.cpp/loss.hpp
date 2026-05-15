#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

namespace nn {

class MSELoss final {
private:
    Matrix grad_input_;

public:
    MSELoss() = default;

    [[nodiscard]] double forward(const Matrix &pred, const Matrix &target) {
        if (pred.rows() != target.rows() || pred.cols() != target.cols()) {
            throw std::invalid_argument("mse loss shape mismatch");
        }
        if (pred.empty()) {
            throw std::invalid_argument("mse loss cannot be computed on an empty matrix");
        }

        grad_input_ = Matrix(pred.rows(), pred.cols());
        const auto total = static_cast<double>(pred.size());

        const double sum_sq = std::transform_reduce(
            NN_EXEC_POLICY,
            pred.data().begin(), pred.data().end(),
            target.data().begin(),
            0.0,
            std::plus<>{},
            [](double prediction, double actual) {
                const double diff = prediction - actual;
                return diff * diff;
            });

        const double loss = sum_sq / total;
        const double factor = 2.0 / total;

        std::transform(NN_EXEC_POLICY,
                       pred.data().begin(), pred.data().end(),
                       target.data().begin(),
                       grad_input_.data().begin(),
                       [factor](double prediction, double actual) {
                           return factor * (prediction - actual);
                       });

        return loss;
    }

    [[nodiscard]] Matrix backward() const { return grad_input_; }
};

} // namespace nn

#endif // LOSS_HPP
