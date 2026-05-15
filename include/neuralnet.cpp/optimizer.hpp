#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "matrix.hpp"

#include <functional>
#include <vector>

namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad() = 0;
};

class SGD : public Optimizer {
private:
    double lr_;
    std::vector<std::reference_wrapper<Matrix>> params_;
    std::vector<std::reference_wrapper<Matrix>> grads_;

public:
    SGD(std::vector<std::reference_wrapper<Matrix>> params,
        std::vector<std::reference_wrapper<Matrix>> grads,
        double lr)
        : lr_(lr), params_(std::move(params)), grads_(std::move(grads)) {
        if (params_.size() != grads_.size()) {
            throw std::invalid_argument("params and grads must have same size");
        }
    }

    void step() override {
        for (std::size_t i = 0; i < params_.size(); ++i) {
            auto &p = params_[i].get();
            auto &g = grads_[i].get();
            std::transform(NN_EXEC_POLICY,
                           p.data().begin(), p.data().end(),
                           g.data().begin(),
                           p.data().begin(),
                           [this](double p_val, double g_val) {
                               return p_val - lr_ * g_val;
                           });
        }
    }

    void zero_grad() override {
        for (auto &g_ref : grads_) {
            auto &g = g_ref.get();
            std::fill(g.data().begin(), g.data().end(), 0.0);
        }
    }
};

} // namespace nn

#endif // OPTIMIZER_HPP
