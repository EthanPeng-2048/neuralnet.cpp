#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <expected>
#include <functional>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace nn
{

    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;
        virtual Result<void> step() = 0;
        virtual void zero_grad() = 0;
    };

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

            Matrix::batch_sgd_update(
                std::span{params_}, std::span{grads_}, lr_);
            return {};
        }

        void zero_grad() override
        {
            Matrix::batch_zero_grad(std::span{grads_});
        }
    };

    class SGD_w_Momentum : public Optimizer
    {
    private:
        Scalar lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> velocities_;
        Scalar beta_ = 0.9;

    public:
        SGD_w_Momentum(std::vector<std::reference_wrapper<Matrix>> params,
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

            const auto n = params_.size();
            // 逐参数矩阵并行：各矩阵独立，无数据竞争
            SmartPolicy::parallel_for_samples(n, [this](std::size_t i) {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                auto &v = velocities_[i];

                // v = beta * v + (1-beta) * g
                v.binary_apply_inplace(g, [beta = beta_](Scalar vv, Scalar gv) noexcept {
                    return beta * vv + (1.0 - beta) * gv;
                });
                // p = p - lr * v
                p.binary_apply_inplace(v, [lr = lr_](Scalar pv, Scalar vv) noexcept {
                    return pv - lr * vv;
                });
            });
            return {};
        }

        void zero_grad() override
        {
            Matrix::batch_zero_grad(std::span{grads_});
        }
    };

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
            const Scalar bc1 = 1.0 - std::pow(beta1_, static_cast<Scalar>(t_));
            const Scalar bc2 = 1.0 - std::pow(beta2_, static_cast<Scalar>(t_));

            // 构建 span 引用包装（batch_adam_update 需要 span<ref<Matrix>>）
            std::vector<std::reference_wrapper<Matrix>> m_refs(m_.begin(), m_.end());
            std::vector<std::reference_wrapper<Matrix>> v_refs(v_.begin(), v_.end());

            Matrix::batch_adam_update(
                std::span{params_}, std::span{grads_},
                std::span{m_refs}, std::span{v_refs},
                lr_, beta1_, beta2_, eps_, bc1, bc2);
            return {};
        }

        void zero_grad() override
        {
            Matrix::batch_zero_grad(std::span{grads_});
        }
    };
} // namespace nn

#endif // OPTIMIZER_HPP
