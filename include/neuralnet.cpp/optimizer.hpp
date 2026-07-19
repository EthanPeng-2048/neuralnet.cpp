#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <cmath>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "algebra/matrix.hpp"
#include "algebra/span.hpp"
#include "algebra/expr.hpp"
#include "algebra/compute_dispatch.hpp"

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
    // 通过 AST 入口 compute::apply 表达更新公式，底层自动并行
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
                // p -= lr * g  (AST 逐元素表达式)
                Span p_s = p.span();
                ConstSpan g_s = g.span();
                compute::apply(p_s, p_s - lr * g_s);
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
    // 通过 AST 入口 compute::apply 表达更新公式
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

                // v = β * v + (1 - β) * g  (AST 逐元素表达式)
                Span v_s = v.span();
                ConstSpan g_s = g.span();
                compute::apply(v_s, beta * v_s + one_minus_beta * g_s);

                // p -= lr * v  (AST 逐元素表达式)
                // 注：v_s 在上一步 compute::apply 后已写入最新值，
                //     在本表达式中作为只读 AST 叶子参与求值（就地操作语义安全）。
                Span p_s = p.span();
                compute::apply(p_s, p_s - lr * v_s);
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
    // 通过 AST 入口 compute::apply 表达更新公式：
    //   - m/v 用 AST 就地更新（两个 compute::apply）
    //   - 参数更新将 m_hat/v_hat/denom/update 四步合一为单一 AST 表达式
    //     （消除 3 次中间 Matrix 临时对象分配）
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

                // m = β₁ * m + (1 - β₁) * g  (AST 逐元素表达式)
                Span m_s = m.span();
                ConstSpan g_s = g.span();
                compute::apply(m_s, beta1 * m_s + one_minus_beta1 * g_s);

                // v = β₂ * v + (1 - β₂) * g²  (AST 逐元素表达式)
                Span v_s = v.span();
                compute::apply(v_s, beta2 * v_s + one_minus_beta2 * g_s * g_s);

                // p -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
                // 通过单一 AST 表达式合并 m_hat / v_hat / denom / update 四步，
                // 消除中间 Matrix 临时对象分配。
                // 注：m_s/v_s 在上一步 compute::apply 后已写入最新值，
                //     在本表达式中作为只读 AST 叶子参与求值（就地操作语义安全）。
                Span p_s = p.span();
                compute::apply(p_s,
                    p_s - lr * (m_s / bc1) / (sqrt(v_s / bc2) + eps));
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
