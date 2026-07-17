#ifndef LOSS_HPP
#define LOSS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "nn_config.hpp"
#include "matrix.hpp"

namespace nn
{
    class Loss
    {
    public:
        virtual ~Loss() = default;
        virtual std::expected<Scalar, Error> forward(const Matrix &pred, const Matrix &target) = 0;
        virtual const Matrix &backward() const = 0;
    };

    class MSELoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        MSELoss() = default;

        [[nodiscard]] std::expected<Scalar, Error> forward(const Matrix &pred, const Matrix &target)
        {
            if (pred.rows() != target.rows() || pred.cols() != target.cols())
                return std::unexpected(Error{"mse loss shape mismatch"});
            if (pred.empty())
                return std::unexpected(Error{"mse loss cannot be computed on an empty matrix"});

            const auto total = static_cast<Scalar>(pred.size());
            const Scalar factor = 2.0 / total;

            // 梯度：通过 Matrix 语义接口计算，不穿透 .data()
            grad_input_ = pred.binary_apply(target, [factor](Scalar p, Scalar t) noexcept {
                return factor * (p - t);
            });

            // 损失：对梯度平方求和还原 MSE
            // diff = grad * total/2, diff² = grad² * total²/4, loss = Σdiff² / total
            const Scalar sum_gsq = grad_input_.reduce(Scalar{0}, std::plus<>{},
                [](Scalar g) noexcept { return g * g; });
            const Scalar loss = sum_gsq * total / Scalar{4};

            return loss;
        }

        [[nodiscard]] const Matrix &backward() const noexcept { return grad_input_; }
    };

    class CrossEntropyLoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        CrossEntropyLoss() = default;

        [[nodiscard]] std::expected<Scalar, Error> forward(const Matrix &logits, const Matrix &target_onehot)
        {
            const std::size_t classes = logits.rows();
            const std::size_t batch = logits.cols();

            if (target_onehot.rows() != classes || target_onehot.cols() != batch)
                return std::unexpected(Error{"cross_entropy loss shape mismatch"});

            grad_input_ = Matrix(classes, batch);

            // 委托 Matrix 语义方法完成交叉熵前向传播
            Scalar loss = Matrix::cross_entropy_forward(
                grad_input_, logits, target_onehot);

            return loss;
        }

        [[nodiscard]] const Matrix &backward() const noexcept { return grad_input_; }
    };

} // namespace nn

#endif // LOSS_HPP