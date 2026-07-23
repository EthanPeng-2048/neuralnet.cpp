#ifndef NN_COMPUTE_OPTIMIZER_HPP
#define NN_COMPUTE_OPTIMIZER_HPP

// ── compute_optimizer.hpp — 引擎化优化器 ───────────────────────────────────
//
// 架构铁律：
//   1. Optimizer 的 step/zero_grad 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。
//   2. 优化器算法只通过 engine 原语（scale_inplace / add_inplace /
//      elementwise_binary / elementwise_unary 等）表达，不直接操作 Matrix。
//   3. 参数与梯度均为 Tensor*（指向 Layer 持有的参数/梯度张量）。
//
// 算法表达示例（原语组合）：
//   SGD:        p -= lr * g
//               temp = clone(g); scale_inplace(temp, -lr); add_inplace(p, temp)
//   Momentum:   v = β*v + (1-β)*g;  p -= lr*v
//   Adam:       m = β1*m + (1-β1)*g;  v = β2*v + (1-β2)*g²;
//               p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "compute_engine.hpp"
#include "compute_layer.hpp"  // clone_tensor
#include "tensor.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Optimizer — 引擎化优化器基类
// ══════════════════════════════════════════════════════════════════════════
class Optimizer
{
public:
    virtual ~Optimizer() = default;
    [[nodiscard]] virtual Result<void> step() = 0;
    [[nodiscard]] virtual Result<void> zero_grad() = 0;
};

// ══════════════════════════════════════════════════════════════════════════
// SGD — 随机梯度下降
//
// 算法：p -= lr * g
// 原语分解：
//   temp = clone(g)
//   scale_inplace(temp, -lr)
//   add_inplace(p, temp)
// ══════════════════════════════════════════════════════════════════════════
class SGD : public Optimizer
{
private:
    ComputeEngine* engine_;
    Scalar lr_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;

public:
    SGD(ComputeEngine& engine,
        std::vector<Tensor*> params,
        std::vector<Tensor*> grads,
        Scalar lr)
        : engine_(&engine), lr_(lr),
          params_(std::move(params)), grads_(std::move(grads)) {}

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"SGD: params/grads size mismatch"});

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // temp = g * (-lr)
            auto temp = clone_tensor(*engine_, *grads_[i]);
            if (!temp) return std::unexpected(temp.error());
            auto r = engine_->scale_inplace(*temp, -lr_);
            if (!r) return std::unexpected(r.error());
            // p += temp  (== p -= lr * g)
            r = engine_->add_inplace(*params_[i], *temp);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    [[nodiscard]] Result<void> zero_grad() override
    {
        for (auto* g : grads_)
        {
            auto r = engine_->zero(*g);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SGDWithMomentum — 带动量的 SGD
//
// 算法：v = β*v + (1-β)*g;  p -= lr*v
// 原语分解：
//   scale_inplace(v, β)
//   temp = clone(g); scale_inplace(temp, 1-β); add_inplace(v, temp)
//   temp2 = clone(v); scale_inplace(temp2, -lr); add_inplace(p, temp2)
// ══════════════════════════════════════════════════════════════════════════
class SGDWithMomentum : public Optimizer
{
private:
    ComputeEngine* engine_;
    Scalar lr_;
    Scalar beta_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    std::vector<Tensor> velocities_;  // 与参数同设备的速度张量

public:
    SGDWithMomentum(ComputeEngine& engine,
                     std::vector<Tensor*> params,
                     std::vector<Tensor*> grads,
                     Scalar lr, Scalar beta = 0.9)
        : engine_(&engine), lr_(lr), beta_(beta),
          params_(std::move(params)), grads_(std::move(grads))
    {
        velocities_.reserve(params_.size());
        for (auto* p : params_)
        {
            auto v = engine_->create_tensor(p->rows(), p->cols());
            (void)engine_->zero(v);
            velocities_.push_back(std::move(v));
        }
    }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Momentum: params/grads size mismatch"});

        const Scalar one_minus_beta = Scalar{1} - beta_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // v = β * v
            auto r = engine_->scale_inplace(velocities_[i], beta_);
            if (!r) return std::unexpected(r.error());

            // v += (1-β) * g
            auto temp = clone_tensor(*engine_, *grads_[i]);
            if (!temp) return std::unexpected(temp.error());
            r = engine_->scale_inplace(*temp, one_minus_beta);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(velocities_[i], *temp);
            if (!r) return std::unexpected(r.error());

            // p += (-lr) * v
            auto step = clone_tensor(*engine_, velocities_[i]);
            if (!step) return std::unexpected(step.error());
            r = engine_->scale_inplace(*step, -lr_);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(*params_[i], *step);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    [[nodiscard]] Result<void> zero_grad() override
    {
        for (auto* g : grads_)
        {
            auto r = engine_->zero(*g);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Adam — 自适应矩估计优化器
//
// 算法：
//   m = β1*m + (1-β1)*g
//   v = β2*v + (1-β2)*g²
//   p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
//
// 原语分解：
//   scale_inplace(m, β1); temp_g = clone(g); scale_inplace(temp_g, 1-β1); add_inplace(m, temp_g)
//   scale_inplace(v, β2); g_sq = elementwise_binary(Mul, g, g);
//                        scale_inplace(g_sq, 1-β2); add_inplace(v, g_sq)
//   m_hat = clone(m); scale_inplace(m_hat, 1/bc1)
//   v_hat = clone(v); scale_inplace(v_hat, 1/bc2)
//   sqrt_v = elementwise_unary(Sqrt, v_hat)
//   denom = elementwise_binary_scalar(Add, sqrt_v, eps)
//   ratio = elementwise_binary(Div, m_hat, denom)
//   scale_inplace(ratio, -lr); add_inplace(p, ratio)
// ══════════════════════════════════════════════════════════════════════════
class Adam : public Optimizer
{
private:
    ComputeEngine* engine_;
    Scalar lr_;
    Scalar beta1_;
    Scalar beta2_;
    Scalar eps_;
    std::size_t t_;  // 时间步
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;
    std::vector<Tensor> m_;  // 一阶矩
    std::vector<Tensor> v_;  // 二阶矩

public:
    Adam(ComputeEngine& engine,
         std::vector<Tensor*> params,
         std::vector<Tensor*> grads,
         Scalar lr,
         Scalar beta1 = 0.9,
         Scalar beta2 = 0.999,
         Scalar eps = 1e-8)
        : engine_(&engine), lr_(lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), t_(0),
          params_(std::move(params)), grads_(std::move(grads))
    {
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (auto* p : params_)
        {
            auto mt = engine_->create_tensor(p->rows(), p->cols());
            auto vt = engine_->create_tensor(p->rows(), p->cols());
            (void)engine_->zero(mt);
            (void)engine_->zero(vt);
            m_.push_back(std::move(mt));
            v_.push_back(std::move(vt));
        }
    }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Adam: params/grads size mismatch"});

        ++t_;
        const Scalar bc1 = Scalar{1} - std::pow(beta1_, static_cast<Scalar>(t_));
        const Scalar bc2 = Scalar{1} - std::pow(beta2_, static_cast<Scalar>(t_));
        const Scalar inv_bc1 = Scalar{1} / bc1;
        const Scalar inv_bc2 = Scalar{1} / bc2;
        const Scalar one_minus_beta1 = Scalar{1} - beta1_;
        const Scalar one_minus_beta2 = Scalar{1} - beta2_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            const Tensor& g = *grads_[i];

            // m = β1 * m + (1-β1) * g
            auto r = engine_->scale_inplace(m_[i], beta1_);
            if (!r) return std::unexpected(r.error());
            auto g_temp = clone_tensor(*engine_, g);
            if (!g_temp) return std::unexpected(g_temp.error());
            r = engine_->scale_inplace(*g_temp, one_minus_beta1);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(m_[i], *g_temp);
            if (!r) return std::unexpected(r.error());

            // v = β2 * v + (1-β2) * g²
            r = engine_->scale_inplace(v_[i], beta2_);
            if (!r) return std::unexpected(r.error());
            auto g_sq = engine_->elementwise_binary(BinaryOp::Mul, g, g);
            if (!g_sq) return std::unexpected(g_sq.error());
            r = engine_->scale_inplace(*g_sq, one_minus_beta2);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(v_[i], *g_sq);
            if (!r) return std::unexpected(r.error());

            // m_hat = m / bc1
            auto m_hat = clone_tensor(*engine_, m_[i]);
            if (!m_hat) return std::unexpected(m_hat.error());
            r = engine_->scale_inplace(*m_hat, inv_bc1);
            if (!r) return std::unexpected(r.error());

            // v_hat = v / bc2
            auto v_hat = clone_tensor(*engine_, v_[i]);
            if (!v_hat) return std::unexpected(v_hat.error());
            r = engine_->scale_inplace(*v_hat, inv_bc2);
            if (!r) return std::unexpected(r.error());

            // sqrt_v = sqrt(v_hat)
            auto sqrt_v = engine_->elementwise_unary(UnaryOp::Sqrt, *v_hat);
            if (!sqrt_v) return std::unexpected(sqrt_v.error());

            // denom = sqrt_v + eps
            auto denom = engine_->elementwise_binary_scalar(BinaryOp::Add, *sqrt_v, eps_);
            if (!denom) return std::unexpected(denom.error());

            // ratio = m_hat / denom
            auto ratio = engine_->elementwise_binary(BinaryOp::Div, *m_hat, *denom);
            if (!ratio) return std::unexpected(ratio.error());

            // p += (-lr) * ratio
            r = engine_->scale_inplace(*ratio, -lr_);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(*params_[i], *ratio);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    [[nodiscard]] Result<void> zero_grad() override
    {
        for (auto* g : grads_)
        {
            auto r = engine_->zero(*g);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ── 优化器工厂函数 ────────────────────────────────────────────────────────
// 根据名称创建对应优化器，支持: "sgd", "sgd_momentum", "adam"（默认 Adam）。
[[nodiscard]] inline std::unique_ptr<Optimizer> create_optimizer(
    std::string_view name,
    ComputeEngine& engine,
    std::vector<Tensor*> params,
    std::vector<Tensor*> grads,
    Scalar lr)
{
    if (name == "sgd")
        return std::make_unique<SGD>(engine, std::move(params), std::move(grads), lr);
    if (name == "sgd_momentum")
        return std::make_unique<SGDWithMomentum>(engine, std::move(params), std::move(grads), lr);
    return std::make_unique<Adam>(engine, std::move(params), std::move(grads), lr);
}

} // namespace nn

#endif // NN_COMPUTE_OPTIMIZER_HPP
