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
            grad_input_ = Matrix(classes, batch);

            std::atomic<Scalar> total_loss{0.0};

            // 逐样本并行：每个样本的 softmax 完全独立，无数据竞争
            SmartPolicy::parallel_for_samples(batch, [&](std::size_t i) {
                // 数值稳定的 softmax
                Scalar max_val = logits.at_unchecked(0, i);
                for (std::size_t c = 1; c < classes; ++c)
                {
                    const Scalar val = logits.at_unchecked(c, i);
                    if (val > max_val) max_val = val;
                }

                // 栈上分配，支持最多 128 类
                std::array<Scalar, 128> exp_vals_fixed{};
                std::vector<Scalar> exp_vals_heap;
                std::span<Scalar> exp_vals;
                if (classes <= 128) {
                    exp_vals = exp_vals_fixed;
                } else {
                    exp_vals_heap.resize(classes);
                    exp_vals = exp_vals_heap;
                }

                Scalar sum_exp = 0.0;
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const Scalar e = std::exp(logits.at_unchecked(c, i) - max_val);
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
                const Scalar prob_true = exp_vals[true_class] / sum_exp;
                Scalar expected = total_loss.load(std::memory_order_relaxed);
                while (!total_loss.compare_exchange_weak(expected, expected - std::log(prob_true),
                                                          std::memory_order_relaxed)) {}

                // 梯度：softmax - target
                for (std::size_t c = 0; c < classes; ++c)
                {
                    const Scalar softmax_c = exp_vals[c] / sum_exp;
                    grad_input_.set_value_unchecked(c, i,
                        softmax_c - target_onehot.at_unchecked(c, i));
                }
            });

            return total_loss.load() / static_cast<Scalar>(batch);
        }

        [[nodiscard]] const Matrix &backward() const noexcept { return grad_input_; }
    };

} // namespace nn

#endif // LOSS_HPP