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
        double lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;

    public:
        SGD(std::vector<std::reference_wrapper<Matrix>> params,
            std::vector<std::reference_wrapper<Matrix>> grads,
            double lr)
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads)) {}

        Result<void> step() override
        {
            if (params_.size() != grads_.size())
            {
                return std::unexpected(Error{"params and grads must have same size"});
            }
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                SmartPolicy::transform(p.data().begin(), p.data().end(),
                               g.data().begin(),
                               p.data().begin(),
                               [this](double p_val, double g_val)
                               {
                                   return p_val - lr_ * g_val;
                               });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };

    class SGD_w_Momentum : public Optimizer
    {
    private:
        double lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> velocities_;
        double beta_ = 0.9;

    public:
        SGD_w_Momentum(std::vector<std::reference_wrapper<Matrix>> params,
                       std::vector<std::reference_wrapper<Matrix>> grads,
                       double lr)
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
            {
                return std::unexpected(Error{"params and grads must have same size"});
            }

            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                auto &v = velocities_[i];

                // 获取vector类型的
                auto &p_vec = p.data();
                auto &g_vec = g.data();
                auto &v_vec = v.data();

                // 构造zip
                auto zip_view = std::views::zip(p_vec, g_vec, v_vec);

                // 并行处理：std::views::zip 将参数、梯度、速度打包为 tuple
                // std::get<N> 按索引解包，顺序与 zip 传入顺序一致
                SmartPolicy::for_each(zip_view.begin(),
                              zip_view.end(),
                              [this](auto &&tuple)
                              {
                                  auto &p_val = std::get<0>(tuple);  // 参数
                                  auto &g_val = std::get<1>(tuple);  // 梯度
                                  auto &v_val = std::get<2>(tuple);  // 速度（动量）
                                  v_val = beta_ * v_val + (1 - beta_) * g_val;
                                  p_val = p_val - lr_ * v_val;
                              });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };

    class Adam : public Optimizer
    {
    private:
        double lr_;
        std::vector<std::reference_wrapper<Matrix>> params_;
        std::vector<std::reference_wrapper<Matrix>> grads_;
        std::vector<Matrix> m_;                      // 一阶矩（梯度均值）
        std::vector<Matrix> v_;                      // 二阶矩（梯度平方均值）
        double beta1_ = 0.9;                         // 一阶矩衰减率
        double beta2_ = 0.999;                       // 二阶矩衰减率
        double eps_ = 1e-8;                          // 防止除零
        std::size_t t_ = 0;                          // 时间步计数器

    public:
        Adam(std::vector<std::reference_wrapper<Matrix>> params,
             std::vector<std::reference_wrapper<Matrix>> grads,
             double lr,
             double beta1 = 0.9,
             double beta2 = 0.999,
             double eps = 1e-8)
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
            {
                return std::unexpected(Error{"params and grads must have same size"});
            }

            ++t_;
            const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(t_)); // 偏差修正分母 (一阶)
            const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(t_)); // 偏差修正分母 (二阶)

            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                auto &m = m_[i];
                auto &v = v_[i];

                auto &p_vec = p.data();
                auto &g_vec = g.data();
                auto &m_vec = m.data();
                auto &v_vec = v.data();

                auto zip_view = std::views::zip(p_vec, g_vec, m_vec, v_vec);

                SmartPolicy::for_each(zip_view.begin(),
                              zip_view.end(),
                              [this, bc1, bc2](auto &&tuple)
                              {
                                  auto &p_val = std::get<0>(tuple);
                                  auto &g_val = std::get<1>(tuple);
                                  auto &m_val = std::get<2>(tuple);
                                  auto &v_val = std::get<3>(tuple);

                                  // 更新一阶矩: m = β₁·m + (1-β₁)·g
                                  m_val = beta1_ * m_val + (1.0 - beta1_) * g_val;
                                  // 更新二阶矩: v = β₂·v + (1-β₂)·g²
                                  v_val = beta2_ * v_val + (1.0 - beta2_) * g_val * g_val;

                                  // 偏差修正
                                  const double m_hat = m_val / bc1;
                                  const double v_hat = v_val / bc2;

                                  // 更新参数: θ = θ - lr · m̂ / (√v̂ + ε)
                                  p_val = p_val - lr_ * m_hat / (std::sqrt(v_hat) + eps_);
                              });
            }
            return {};
        }

        void zero_grad() override
        {
            for (auto &g_ref : grads_)
            {
                auto &g = g_ref.get();
                std::fill(g.data().begin(), g.data().end(), 0.0);
            }
        }
    };
} // namespace nn

#endif // OPTIMIZER_HPP
