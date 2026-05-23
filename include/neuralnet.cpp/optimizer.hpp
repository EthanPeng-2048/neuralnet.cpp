#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <functional>
#include <vector>

#include <neuralnet.cpp/matrix.hpp>

namespace nn
{

    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;
        virtual void step() = 0;
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
            : lr_(lr), params_(std::move(params)), grads_(std::move(grads))
        {
            if (params_.size() != grads_.size())
            {
                throw std::invalid_argument("params and grads must have same size");
            }
        }

        void step() override
        {
            for (std::size_t i = 0; i < params_.size(); ++i)
            {
                auto &p = params_[i].get();
                auto &g = grads_[i].get();
                std::transform(NN_EXEC_POLICY,
                               p.data().begin(), p.data().end(),
                               g.data().begin(),
                               p.data().begin(),
                               [this](double p_val, double g_val)
                               {
                                   return p_val - lr_ * g_val;
                               });
            }
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
            if (params_.size() != grads_.size())
            {
                throw std::invalid_argument("params and grads must have same size");
            }
            velocities_.reserve(params_.size());
            for (const auto &p_ref : params_)
            {
                const Matrix &p = p_ref.get();
                velocities_.emplace_back(p.rows(), p.cols(), 0.0);
            }
        }

        void step() override
        {
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

                // 并行处理，注意这里NN_EXEC_POLICY可能是par_unseq，要考虑乱序问题
                std::for_each(NN_EXEC_POLICY,
                              zip_view.begin(),
                              zip_view.end(),
                              [this](auto &&tuple)
                              {
                                  auto &p_val = std::get<0>(tuple);  // tuple是什么原理？
                                  auto &g_val = std::get<1>(tuple);
                                  auto &v_val = std::get<2>(tuple);
                                  v_val = beta_ * v_val + (1 - beta_) * g_val;
                                  p_val = p_val - lr_ * v_val;
                              });
            }
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
