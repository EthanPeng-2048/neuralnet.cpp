#ifndef LOSS_HPP
#define LOSS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "nn_config.hpp"
#include "algebra/matrix.hpp"

namespace nn
{
    class Loss
    {
    public:
        virtual ~Loss() = default;
        virtual Result<Scalar> forward(const Matrix &pred, const Matrix &target) = 0;
        virtual const Matrix &backward() const = 0;
    };

    class MSELoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        MSELoss() = default;

        [[nodiscard]] Result<Scalar> forward(const Matrix &pred, const Matrix &target) override
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

        [[nodiscard]] const Matrix &backward() const noexcept override { return grad_input_; }
    };

    // ── CrossEntropyLoss（带 softmax） ───────────────────────────────────
    // 输入: logits (classes, batch)，target_onehot (classes, batch)
    // 算法：对每列（每个 batch 样本）独立做 softmax + cross entropy
    //   loss = -(1/batch) * Σ_i log(softmax(logits[:, i])[true_class_i])
    //   grad_input = softmax - target_onehot
    // 仅使用 Matrix 通用接口（apply/binary_apply/reduce/col_reduce/broadcast_col_inplace）
    class CrossEntropyLoss : public Loss
    {
    private:
        Matrix grad_input_;

    public:
        CrossEntropyLoss() = default;

        [[nodiscard]] Result<Scalar> forward(const Matrix &logits, const Matrix &target_onehot) override
        {
            const std::size_t classes = logits.rows();
            const std::size_t batch = logits.cols();

            if (target_onehot.rows() != classes || target_onehot.cols() != batch)
                return std::unexpected(Error{"cross_entropy loss shape mismatch"});

            // 1. 对每列求最大值（数值稳定）—— 按列归约
            Matrix col_max = logits.col_reduce(
                std::numeric_limits<Scalar>::lowest(),
                [](Scalar a, Scalar b) noexcept { return std::max(a, b); },
                [](Scalar x) noexcept { return x; });  // shape: (1, batch)

            // 2. shifted = logits - col_max（按列广播减法）
            Matrix shifted = logits;
            shifted.broadcast_col_inplace(col_max,
                [](Scalar x, Scalar m) noexcept { return x - m; });

            // 3. exp_shifted = exp(shifted)
            Matrix exp_shifted = shifted.apply(
                [](Scalar x) noexcept { return std::exp(x); });

            // 4. col_sum[i] = Σ_c exp_shifted[c][i]  —— 按列归约
            Matrix col_sum = exp_shifted.col_reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });  // shape: (1, batch)

            // 5. softmax[c][i] = exp_shifted[c][i] / col_sum[i]  —— 按列广播除法
            Matrix softmax = exp_shifted;
            softmax.broadcast_col_inplace(col_sum,
                [](Scalar e, Scalar s) noexcept { return e / s; });

            // 6. 梯度 = softmax - target_onehot
            grad_input_ = softmax.binary_apply(target_onehot,
                [](Scalar p, Scalar t) noexcept { return p - t; });

            // 7. 损失 = -(1/batch) * Σ_i log(softmax[true_class_i, i])
            //    等价实现：loss = -(1/batch) * Σ_{c,i} target[c][i] * log(softmax[c][i])
            //    （因为 target 是 one-hot，求和只对 true_class 非零）
            //    数值稳定：log(softmax) = log(exp_shifted) - log(col_sum)
            //                              = shifted - log(col_sum)
            Matrix log_softmax = shifted;
            Matrix log_col_sum = col_sum.apply(
                [](Scalar s) noexcept { return std::log(s); });
            log_softmax.broadcast_col_inplace(log_col_sum,
                [](Scalar a, Scalar b) noexcept { return a - b; });

            // dot = Σ target * log_softmax
            Matrix target_dot_log = target_onehot.binary_apply(log_softmax,
                [](Scalar t, Scalar l) noexcept { return t * l; });
            const Scalar sum_nll = target_dot_log.reduce(Scalar{0},
                [](Scalar a, Scalar b) noexcept { return a + b; },
                [](Scalar x) noexcept { return x; });

            const Scalar loss = -sum_nll / static_cast<Scalar>(batch);
            return loss;
        }

        [[nodiscard]] const Matrix &backward() const noexcept override { return grad_input_; }
    };
} // namespace nn

#endif // LOSS_HPP