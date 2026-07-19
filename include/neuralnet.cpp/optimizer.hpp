#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "algebra/matrix.hpp"

namespace nn
{

    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;
        virtual Result<void> step() = 0;
        virtual void zero_grad() = 0;
    };

    // ── SGD: p -= lr * g ───────────────────────────────────────────────
    // 用 Matrix::binary_apply_inplace 通用接口表达更新公式
    class SGD : public Optimizer
    {
    private:
        Scalar lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;

    public:
        SGD(std::vector<std::reference_wrapper<Matrix>> params,
            std::vector<std::reference_wrapper<Matrix>> grads,
            Scalar lr)
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads)) {}

        Result<void> step() override
        {
            if (params_.size() != grads_.size())
                return std::unexpected(Error{"params and grads must have same size"});

            const Scalar lr = lr_;
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                Matrix &p = params_[i].get();
                const Matrix &g = grads_[i].get();
                p.binary_apply_inplace(g,
                    [lr](Scalar pv, Scalar gv) noexcept { return pv - lr * gv; });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
                g_ref.get().zero();
        }
    };

    // ── SGD+Momentum ──────────────────────────────────────────────────
    //   v = β * v + (1 - β) * g
    //   p -= lr * v
    class SGDWithMomentum : public Optimizer
    {
    private:
        Scalar lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> velocities_;
        Scalar beta_ = 0.9;

    public:
        SGDWithMomentum(std::vector<std::reference_wrapper<Matrix>> params,
                        std::vector<std::reference_wrapper<Matrix>> grads,
                        Scalar lr)
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads))
        {
            velocities_.reserve(params_.size());
            for (const auto &p_ref : params_)
            {
                const Matrix &p = p_ref.get();
                velocities_.emplace_back(p.rows(), p.cols(), 0.0);
            }
        }

        Result<void> step() override
        {
            if (params_.size() != grads_.size())
                return std::unexpected(Error{"params and grads must have same size"});

            const Scalar lr = lr_;
            const Scalar beta = beta_;
            const Scalar one_minus_beta = Scalar{1} - beta;
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                Matrix &p = params_[i].get();
                const Matrix &g = grads_[i].get();
                Matrix &v = velocities_[i];
                // v = β * v + (1 - β) * g
                v.binary_apply_inplace(g,
                    [beta, one_minus_beta](Scalar vv, Scalar gv) noexcept {
                        return beta * vv + one_minus_beta * gv;
                    });
                // p -= lr * v
                p.binary_apply_inplace(v,
                    [lr](Scalar pv, Scalar vv) noexcept { return pv - lr * vv; });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
                g_ref.get().zero();
        }
    };

    // ── Adam ──────────────────────────────────────────────────────────
    //   m = β₁ * m + (1 - β₁) * g
    //   v = β₂ * v + (1 - β₂) * g²
    //   p -= lr * (m / (1 - β₁ᵗ)) / (sqrt(v / (1 - β₂ᵗ)) + ε)
    //
    // 仅使用 Matrix 通用接口（apply / binary_apply_inplace）
    class Adam : public Optimizer
    {
    private:
        Scalar lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> m_;                      // 一阶矩
        std::vector<Matrix> v_;                      // 二阶矩
        Scalar beta1_ = 0.9;
        Scalar beta2_ = 0.999;
        Scalar eps_ = 1e-8;
        std::size_t t_ = 0;

    public:
        Adam(std::vector<std::reference_wrapper<Matrix>> params,
             std::vector<std::reference_wrapper<Matrix>> grads,
             Scalar lr,
             Scalar beta1 = 0.9,
             Scalar beta2 = 0.999,
             Scalar eps = 1e-8)
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads)),
              beta1_(beta1), beta2_(beta2), eps_(eps), t_(0)
        {
            m_.reserve(params_.size());
            v_.reserve(params_.size());
            for (const auto &p_ref : params_)
            {
                const Matrix &p = p_ref.get();
                m_.emplace_back(p.rows(), p.cols(), 0.0);
                v_.emplace_back(p.rows(), p.cols(), 0.0);
            }
        }

        Result<void> step() override
        {
            if (params_.size() != grads_.size())
                return std::unexpected(Error{"params and grads must have same size"});

            ++t_;
            const Scalar bc1 = Scalar{1} - std::pow(beta1_, static_cast<Scalar>(t_));
            const Scalar bc2 = Scalar{1} - std::pow(beta2_, static_cast<Scalar>(t_));
            const Scalar lr = lr_;
            const Scalar beta1 = beta1_;
            const Scalar beta2 = beta2_;
            const Scalar eps = eps_;
            const Scalar one_minus_beta1 = Scalar{1} - beta1;
            const Scalar one_minus_beta2 = Scalar{1} - beta2;

            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                Matrix &p = params_[i].get();
                const Matrix &g = grads_[i].get();
                Matrix &m = m_[i];
                Matrix &v = v_[i];

                // m = β₁ * m + (1 - β₁) * g
                m.binary_apply_inplace(g,
                    [beta1, one_minus_beta1](Scalar mv, Scalar gv) noexcept {
                        return beta1 * mv + one_minus_beta1 * gv;
                    });

                // v = β₂ * v + (1 - β₂) * g²
                v.binary_apply_inplace(g,
                    [beta2, one_minus_beta2](Scalar vv, Scalar gv) noexcept {
                        return beta2 * vv + one_minus_beta2 * gv * gv;
                    });

                // p -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
                //   = lr * m_hat / (sqrt(v_hat) + eps)
                // 拆成三步用 Matrix 通用接口表达（binary_apply_inplace 只能同时访问两个矩阵）：
                //   step1: m_hat = m / bc1
                //   step2: v_hat = v / bc2; denom = sqrt(v_hat) + eps
                //   step3: update = lr * m_hat / denom;  p -= update
                Matrix m_hat = m.apply([bc1](Scalar mv) noexcept { return mv / bc1; });
                Matrix denom = v.apply([bc2, eps](Scalar vv) noexcept {
                    return std::sqrt(vv / bc2) + eps;
                });
                Matrix update = m_hat.binary_apply(denom,
                    [lr](Scalar m_val, Scalar d) noexcept { return lr * m_val / d; });
                p.binary_apply_inplace(update,
                    [](Scalar pv, Scalar u) noexcept { return pv - u; });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
                g_ref.get().zero();
        }
    };
} // namespace nn

#endif // OPTIMIZER_HPP
