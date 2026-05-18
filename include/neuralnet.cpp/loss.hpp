#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn
{
    class Loss
    {
    public:
        virtual ~Loss() = default;
        virtual double forward(const Matrix &pred, const Matrix &target) = 0;
        virtual Matrix backward() const = 0;
    };


    class MSELoss: public Loss
    {
    private:
        Matrix grad_input_;

    public:
        MSELoss() = default;

        [[nodiscard]] double forward(const Matrix &pred, const Matrix &target)
        {
            if (pred.rows() != target.rows() || pred.cols() != target.cols())
            {
                throw std::invalid_argument("mse loss shape mismatch");
            }
            if (pred.empty())
            {
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
                [](double prediction, double actual)
                {
                    const double diff = prediction - actual;
                    return diff * diff;
                });

            const double loss = sum_sq / total;
            const double factor = 2.0 / total;

            std::transform(NN_EXEC_POLICY,
                           pred.data().begin(), pred.data().end(),
                           target.data().begin(),
                           grad_input_.data().begin(),
                           [factor](double prediction, double actual)
                           {
                               return factor * (prediction - actual);
                           });

            return loss;
        }

        [[nodiscard]] Matrix backward() const { return grad_input_; }
    };

    class CrossEntropyLoss: public Loss
    {
    private:
        Matrix grad_input_;

    public:
        CrossEntropyLoss() = default;

        [[nodiscard]] double forward(const Matrix &logits, const Matrix &target_onehot)
        {
            const std::size_t classes = logits.rows();
            const std::size_t batch = logits.cols();
            grad_input_ = Matrix(classes, batch);

            double total_loss = 0.0;

            for (std::size_t i = 0; i < batch; ++i)
            {
                // 数值稳定的 softmax
                double max_val = logits.at_unchecked(0, i);
                for (std::size_t c = 1; c < classes; ++c)
                {
                    max_val = std::max(max_val, logits.at_unchecked(c, i));
                }
                double sum_exp = 0.0;
                std::vector<double> exp_vals(classes);
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const double e = std::exp(logits.at_unchecked(c, i) - max_val);
                    exp_vals[c] = e;
                    sum_exp += e;
                }

                // 找到真实类别
                std::size_t true_class = 0;
                for (std::size_t c = 0; c < classes; ++c)
                {
                    if (target_onehot.at_unchecked(c, i) > 0.5)
                    {
                        true_class = c;
                        break;
                    }
                }

                // loss = -log(softmax[true_class])
                const double prob_true = exp_vals[true_class] / sum_exp;
                total_loss += -std::log(prob_true);

                // 梯度：softmax - target
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const double softmax_c = exp_vals[c] / sum_exp;
                    grad_input_.set_value_unchecked(c, i,
                                                    softmax_c - target_onehot.at_unchecked(c, i));
                }
            }

            return total_loss / static_cast<double>(batch);
        }

        [[nodiscard]] const Matrix &backward() const { return grad_input_; }
    };

} // namespace nn

#endif // LOSS_HPP
