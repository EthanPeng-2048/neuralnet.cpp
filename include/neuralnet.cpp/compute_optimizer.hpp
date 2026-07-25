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
//   Momentum:   v = β*v + (1-β)*g;  p -= lr*v
//   Adam:       m = β1*m + (1-β1)*g;  v = β2*v + (1-β2)*g²;
//               p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
//   AdamW:      同 Adam，但权重衰减解耦：p *= (1 - lr*wd) 独立于梯度更新
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <optional>
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
//
// 职责：
//   - 持有 engine / params / grads 三元组（所有子类共享）
//   - 提供 zero_grad() 默认实现
//   - 提供 scale_add_() 辅助方法（axpy: dst += scalar * src），消除子类重复模板
// ══════════════════════════════════════════════════════════════════════════
class Optimizer
{
protected:
    ComputeEngine* engine_;
    std::vector<Tensor*> params_;
    std::vector<Tensor*> grads_;

    // 辅助方法：dst += scalar * src（clone + scale + add 三步原语组合）
    // 用于 SGD / Momentum / Adam 等优化器中常见的 axpy 操作
    [[nodiscard]] Result<void> scale_add_(Tensor& dst, Scalar scalar, const Tensor& src)
    {
        auto tmp = clone_tensor(*engine_, src);
        if (!tmp) return std::unexpected(tmp.error());
        auto r = engine_->scale_inplace(*tmp, scalar);
        if (!r) return std::unexpected(r.error());
        return engine_->add_inplace(dst, *tmp);
    }

public:
    Optimizer(ComputeEngine& engine,
              std::vector<Tensor*> params,
              std::vector<Tensor*> grads)
        : engine_(&engine),
          params_(std::move(params)),
          grads_(std::move(grads)) {}

    virtual ~Optimizer() = default;

    [[nodiscard]] virtual Result<void> step() = 0;

    // 默认实现：将所有梯度清零（所有子类行为一致）
    [[nodiscard]] virtual Result<void> zero_grad()
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
// SGD — 随机梯度下降
//
// 算法：p -= lr * g
// 原语：scale_add_(p, -lr, g)
// ══════════════════════════════════════════════════════════════════════════
class SGD : public Optimizer
{
    Scalar lr_;

public:
    SGD(ComputeEngine& engine,
        std::vector<Tensor*> params,
        std::vector<Tensor*> grads,
        Scalar lr)
        : Optimizer(engine, std::move(params), std::move(grads)), lr_(lr) {}

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"SGD: params/grads size mismatch"});

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            auto r = scale_add_(*params_[i], -lr_, *grads_[i]);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SGDWithMomentum — 带动量的 SGD
//
// 算法：v = β*v + (1-β)*g;  p -= lr*v
// 原语：scale_add_(v, 1-β, g)  →  scale_add_(p, -lr, v)
// ══════════════════════════════════════════════════════════════════════════
class SGDWithMomentum : public Optimizer
{
    Scalar lr_;
    Scalar beta_;
    std::vector<Tensor> velocities_;

public:
    SGDWithMomentum(ComputeEngine& engine,
                     std::vector<Tensor*> params,
                     std::vector<Tensor*> grads,
                     Scalar lr, Scalar beta = 0.9)
        : Optimizer(engine, std::move(params), std::move(grads)),
          lr_(lr), beta_(beta)
    {
        velocities_.reserve(params_.size());
        for (auto* p : params_)
        {
            auto v = engine_->create_tensor(p->rows(), p->cols());
            { auto r = engine_->zero(v); NN_ASSERT(r, r ? "" : r.error().message.c_str()); }
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
            // v = β*v + (1-β)*g
            auto r = engine_->scale_inplace(velocities_[i], beta_);
            if (!r) return std::unexpected(r.error());
            r = scale_add_(velocities_[i], one_minus_beta, *grads_[i]);
            if (!r) return std::unexpected(r.error());

            // p -= lr * v
            r = scale_add_(*params_[i], -lr_, velocities_[i]);
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
// ══════════════════════════════════════════════════════════════════════════
class Adam : public Optimizer
{
protected:
    Scalar lr_;
    Scalar beta1_;
    Scalar beta2_;
    Scalar eps_;
    std::size_t t_;
    std::vector<Tensor> m_;  // 一阶矩
    std::vector<Tensor> v_;  // 二阶矩

    // Adam 核心更新（提取为 protected，AdamW 复用）
    [[nodiscard]] Result<void> adam_update_(std::size_t i)
    {
        const Scalar bc1 = Scalar{1} - std::pow(beta1_, static_cast<Scalar>(t_));
        const Scalar bc2 = Scalar{1} - std::pow(beta2_, static_cast<Scalar>(t_));
        const Scalar inv_bc1 = Scalar{1} / bc1;
        const Scalar inv_bc2 = Scalar{1} / bc2;
        const Scalar one_minus_beta1 = Scalar{1} - beta1_;
        const Scalar one_minus_beta2 = Scalar{1} - beta2_;

        const Tensor& g = *grads_[i];

        // m = β1*m + (1-β1)*g
        auto r = engine_->scale_inplace(m_[i], beta1_);
        if (!r) return std::unexpected(r.error());
        r = scale_add_(m_[i], one_minus_beta1, g);
        if (!r) return std::unexpected(r.error());

        // v = β2*v + (1-β2)*g²
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

        // p -= lr * ratio
        r = engine_->scale_inplace(*ratio, -lr_);
        if (!r) return std::unexpected(r.error());
        return engine_->add_inplace(*params_[i], *ratio);
    }

    void init_moments_()
    {
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (auto* p : params_)
        {
            auto mt = engine_->create_tensor(p->rows(), p->cols());
            auto vt = engine_->create_tensor(p->rows(), p->cols());
            { auto r1 = engine_->zero(mt); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }
            { auto r2 = engine_->zero(vt); NN_ASSERT(r2, r2 ? "" : r2.error().message.c_str()); }
            m_.push_back(std::move(mt));
            v_.push_back(std::move(vt));
        }
    }

public:
    Adam(ComputeEngine& engine,
         std::vector<Tensor*> params,
         std::vector<Tensor*> grads,
         Scalar lr,
         Scalar beta1 = 0.9,
         Scalar beta2 = 0.999,
         Scalar eps = 1e-8)
        : Optimizer(engine, std::move(params), std::move(grads)),
          lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0)
    {
        init_moments_();
    }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Adam: params/grads size mismatch"});

        ++t_;
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            auto r = adam_update_(i);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// AdamW — 解耦权重衰减的 Adam
//
// 算法（Loshchilov & Hutter, 2019）：
//   m = β1*m + (1-β1)*g
//   v = β2*v + (1-β2)*g²
//   p = (1 - lr*wd) * p          ← 权重衰减解耦，独立于梯度更新
//   p -= lr * (m/bc1) / (sqrt(v/bc2) + eps)
//
// 与 Adam+L2正则化的区别：
//   - L2: g' = g + wd*p，然后用 g' 做 Adam 更新 → wd 受自适应学习率缩放
//   - AdamW: 直接衰减权重 p *= (1-lr*wd)，梯度更新不受 wd 影响
//   → AdamW 的权重衰减对所有参数等效，不因自适应学习率而被稀释
// ══════════════════════════════════════════════════════════════════════════
class AdamW : public Adam
{
    Scalar wd_;  // 权重衰减系数

public:
    AdamW(ComputeEngine& engine,
          std::vector<Tensor*> params,
          std::vector<Tensor*> grads,
          Scalar lr,
          Scalar beta1 = 0.9,
          Scalar beta2 = 0.999,
          Scalar eps = 1e-8,
          Scalar weight_decay = 0.01)
        : Adam(engine, std::move(params), std::move(grads),
               lr, beta1, beta2, eps),
          wd_(weight_decay) {}

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"AdamW: params/grads size mismatch"});

        ++t_;
        const Scalar decay_factor = Scalar{1} - lr_ * wd_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // 权重衰减解耦：p = (1 - lr*wd) * p
            if (wd_ != Scalar{0})
            {
                auto r = engine_->scale_inplace(*params_[i], decay_factor);
                if (!r) return std::unexpected(r.error());
            }

            // Adam 更新
            auto r = adam_update_(i);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// Newton-Schulz 正交化 — Muon 的核心操作
//
// 给定矩阵 G，计算其最近的半正交矩阵 Ortho(G) = U V^T（G = U S V^T 为 SVD）。
// 使用 quintic Newton-Schulz 迭代高效逼近，系数经调优可在 5 步内收敛。
//
// 算法：
//   X = G / (||G||_F + eps)
//   for _ in range(steps):
//       A = X @ X^T
//       B = b*A + c*A^2
//       X = a*X + B @ X
//
// 参考：Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks"
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline Result<Tensor> newton_schulz_orthogonalize(
    ComputeEngine& engine, const Tensor& G, std::size_t steps = 5, Scalar eps = 1e-7f)
{
    // 调优后的 quintic 多项式系数（使 φ^N(x) → 1 for x ∈ [0,1]）
    constexpr Scalar a = 3.4445f;
    constexpr Scalar b = -4.7750f;
    constexpr Scalar c = 2.0315f;

    // 计算 Frobenius 范数：||G||_F = sqrt(Σ g_ij²)
    auto g_sq = engine.elementwise_binary(BinaryOp::Mul, G, G);
    if (!g_sq) return std::unexpected(g_sq.error());
    auto row_sums = engine.row_reduce_sum(*g_sq);
    if (!row_sums) return std::unexpected(row_sums.error());
    auto total_sum = engine.col_reduce_sum(*row_sums);
    if (!total_sum) return std::unexpected(total_sum.error());
    auto frob_norm = engine.elementwise_unary(UnaryOp::Sqrt, *total_sum);
    if (!frob_norm) return std::unexpected(frob_norm.error());

    // X = G / (||G||_F + eps)  →  先计算 1/(||G||_F + eps)，再 scale
    auto inv_norm_plus_eps = engine.elementwise_binary_scalar(BinaryOp::Add, *frob_norm, eps);
    if (!inv_norm_plus_eps) return std::unexpected(inv_norm_plus_eps.error());
    // inv_norm_plus_eps 是 (1,1) 张量，需要提取标量值
    // 用 Rsqrt: 1/sqrt(x) → 对 total_sum 直接用 Rsqrt
    auto inv_frob = engine.elementwise_unary(UnaryOp::Rsqrt,
        *engine.elementwise_binary_scalar(BinaryOp::Add, *total_sum, eps * eps));
    // 更简洁的方式：直接用标量运算
    // ||G||_F² = total_sum[0,0]，所以 1/||G||_F = 1/sqrt(total_sum)

    // 重新设计：避免从 Tensor 提取标量的复杂性
    // 策略：先 scale G 使 Frobenius 范数 ≤ 1，再做 NS 迭代

    // Step 1: 归一化 G 使其 Frobenius 范数 ≈ 1
    //   X = G * (1 / (||G||_F + eps))
    //   用 Rsqrt(||G||_F²) = 1/||G||_F 实现
    auto norm_sq = engine.elementwise_binary(BinaryOp::Mul, G, G);
    if (!norm_sq) return std::unexpected(norm_sq.error());
    auto row_sum_norm = engine.row_reduce_sum(*norm_sq);
    if (!row_sum_norm) return std::unexpected(row_sum_norm.error());
    auto total_norm_sq = engine.col_reduce_sum(*row_sum_norm);
    if (!total_norm_sq) return std::unexpected(total_norm_sq.error());
    // total_norm_sq 是 (1,1) 张量，值为 ||G||_F²
    // 1/sqrt(||G||_F² + eps²) 可以通过 Rsqrt 实现
    auto inv_norm_sq = engine.elementwise_binary_scalar(BinaryOp::Add, *total_norm_sq, eps * eps);
    if (!inv_norm_sq) return std::unexpected(inv_norm_sq.error());
    auto inv_norm = engine.elementwise_unary(UnaryOp::Rsqrt, *inv_norm_sq);
    if (!inv_norm) return std::unexpected(inv_norm.error());

    // X = G * inv_norm  (inv_norm 是 (1,1) 张量，需要广播)
    auto X = clone_tensor(engine, G);
    if (!X) return std::unexpected(X.error());
    // 用 broadcast_col_inplace 将 (1,1) 标量广播到整个矩阵
    // 但 broadcast_col_inplace 需要 (1, cols) 的向量
    // 这里 inv_norm 是 (1,1)，可以用 scale_inplace 的标量版本
    // 但我们需要先提取标量值... 这在纯引擎接口下不可行

    // 备选方案：用 matmul 实现标量乘法
    // X = G * s  等价于  X = G @ (s * I)，但太复杂

    // 最简方案：利用 elementwise_binary_scalar 逐元素操作
    // 但 inv_norm 不是标量，是 (1,1) 张量

    // 最终方案：重新设计归一化流程
    // 1. 计算 G 的 Frobenius 范数的平方（标量，存在 (1,1) 张量中）
    // 2. 用 elementwise_unary(Rsqrt) 得到 1/||G||_F
    // 3. 用 broadcast 乘以 G

    // 实际上，我们可以用一个更聪明的方法：
    // 先计算 G² 的行和，得到每行的平方和向量
    // 再用 broadcast_row_inplace 将每行除以该行的范数
    // 但这不是 Frobenius 范数归一化，而是行范数归一化

    // 最简洁方案：直接用 engine 的标量操作
    // 问题：我们需要从 (1,1) Tensor 中提取标量值
    // 解决：用 engine->to_matrix() 提取

    // 重新实现，使用 to_matrix 提取标量
    auto total_norm_sq_mat = engine.to_matrix(*total_norm_sq);
    if (!total_norm_sq_mat) return std::unexpected(total_norm_sq_mat.error());
    Scalar norm_sq_val = total_norm_sq_mat->at(0, 0);
    Scalar inv_norm_scalar = Scalar{1} / std::sqrt(norm_sq_val + eps * eps);

    // X = G * inv_norm_scalar
    auto X2 = clone_tensor(engine, G);
    if (!X2) return std::unexpected(X2.error());
    auto r = engine.scale_inplace(*X2, inv_norm_scalar);
    if (!r) return std::unexpected(r.error());

    // 如果 rows < cols，转置以确保 X 是"瘦"矩阵（rows ≥ cols）
    // 这样 X @ X^T 是较小的方阵，减少计算量
    bool transposed = (G.rows() < G.cols());
    if (transposed)
    {
        // X = X^T  →  用 matmul 实现转置太浪费，用 engine 的 clone + reshape
        // 但 ComputeEngine 没有直接的 transpose 操作
        // 解决：用 matmul(X^T, I) 或直接在 NS 迭代中处理
        // 实际上，对于 NS 迭代，我们只需要 X @ X^T
        // 如果 rows < cols，我们可以计算 X^T @ X（更小的方阵）
        // 然后在最后转置回来

        // 简化：假设参数是 2D 且 rows ≥ cols（大多数权重矩阵如此）
        // 如果 rows < cols，我们仍然可以工作，只是 X @ X^T 更大
        // 在实际神经网络中，权重矩阵通常是 (out_features, in_features)
        // 对于线性层，通常 out_features ≤ in_features 或 vice versa
        // 我们不做转置优化，直接计算
    }

    // Newton-Schulz 迭代
    for (std::size_t t = 0; t < steps; ++t)
    {
        // A = X @ X^T
        auto A = engine.matmul(*X2, *X2, false, true);
        if (!A) return std::unexpected(A.error());

        // A_sq = A @ A
        auto A_sq = engine.matmul(*A, *A);
        if (!A_sq) return std::unexpected(A_sq.error());

        // B = b*A + c*A_sq
        auto B = clone_tensor(engine, *A);
        if (!B) return std::unexpected(B.error());
        r = engine.scale_inplace(*B, b);
        if (!r) return std::unexpected(r.error());
        auto c_A_sq = clone_tensor(engine, *A_sq);
        if (!c_A_sq) return std::unexpected(c_A_sq.error());
        r = engine.scale_inplace(*c_A_sq, c);
        if (!r) return std::unexpected(r.error());
        r = engine.add_inplace(*B, *c_A_sq);
        if (!r) return std::unexpected(r.error());

        // X_new = a*X + B @ X
        auto aX = clone_tensor(engine, *X2);
        if (!aX) return std::unexpected(aX.error());
        r = engine.scale_inplace(*aX, a);
        if (!r) return std::unexpected(r.error());

        auto BX = engine.matmul(*B, *X2);
        if (!BX) return std::unexpected(BX.error());

        r = engine.add_inplace(*aX, *BX);
        if (!r) return std::unexpected(r.error());

        // X = X_new
        X2 = std::move(*aX);
    }

    return *X2;
}

// ══════════════════════════════════════════════════════════════════════════
// Muon — MomentUm Orthogonalized by Newton-Schulz
//
// 算法（Keller Jordan et al., 2024）：
//   对于每个 2D 参数 p：
//     1. SGD-Momentum: v = μ*v + g
//     2. Newton-Schulz 正交化: update = NS5(v)  // 将更新矩阵正交化
//     3. 参数更新: p -= lr * update
//
// 设计要点：
//   - 仅对 ≥2D 的参数（权重矩阵）应用 Muon
//   - 嵌入层、分类头、bias/gain 应使用 AdamW
//   - Nesterov 动量可略微提升性能
//
// 参考：
//   - https://kellerjordan.github.io/posts/muon/
//   - https://github.com/KellerJordan/Muon
// ══════════════════════════════════════════════════════════════════════════
class Muon : public Optimizer
{
    Scalar lr_;
    Scalar momentum_;
    bool nesterov_;
    std::size_t ns_steps_;
    Scalar ns_eps_;
    std::vector<Tensor> velocities_;

public:
    Muon(ComputeEngine& engine,
         std::vector<Tensor*> params,
         std::vector<Tensor*> grads,
         Scalar lr,
         Scalar momentum = 0.95f,
         bool nesterov = true,
         std::size_t ns_steps = 5,
         Scalar ns_eps = 1e-7f)
        : Optimizer(engine, std::move(params), std::move(grads)),
          lr_(lr), momentum_(momentum), nesterov_(nesterov),
          ns_steps_(ns_steps), ns_eps_(ns_eps)
    {
        velocities_.reserve(params_.size());
        for (auto* p : params_)
        {
            auto v = engine_->create_tensor(p->rows(), p->cols());
            { auto r = engine_->zero(v); NN_ASSERT(r, r ? "" : r.error().message.c_str()); }
            velocities_.push_back(std::move(v));
        }
    }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Muon: params/grads size mismatch"});

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            const Tensor& g = *grads_[i];

            // 1. SGD-Momentum: v = μ*v + g
            auto r = engine_->scale_inplace(velocities_[i], momentum_);
            if (!r) return std::unexpected(r.error());
            r = engine_->add_inplace(velocities_[i], g);
            if (!r) return std::unexpected(r.error());

            // 确定用于正交化的更新方向
            Tensor* update_ptr = nullptr;
            std::optional<Tensor> nesterov_buf;
            if (nesterov_)
            {
                // Nesterov: update = g + μ*v
                auto nesterov_result = clone_tensor(*engine_, g);
                if (!nesterov_result) return std::unexpected(nesterov_result.error());
                nesterov_buf = std::move(*nesterov_result);
                auto mu_v = clone_tensor(*engine_, velocities_[i]);
                if (!mu_v) return std::unexpected(mu_v.error());
                r = engine_->scale_inplace(*mu_v, momentum_);
                if (!r) return std::unexpected(r.error());
                r = engine_->add_inplace(*nesterov_buf, *mu_v);
                if (!r) return std::unexpected(r.error());
                update_ptr = &*nesterov_buf;
            }
            else
            {
                update_ptr = &velocities_[i];
            }

            // 2. Newton-Schulz 正交化（仅对 ≥2D 参数，即 rows > 1 且 cols > 1）
            if (params_[i]->rows() > 1 && params_[i]->cols() > 1)
            {
                auto ortho_update = newton_schulz_orthogonalize(
                    *engine_, *update_ptr, ns_steps_, ns_eps_);
                if (!ortho_update) return std::unexpected(ortho_update.error());

                // 3. 参数更新: p -= lr * ortho_update
                r = engine_->scale_inplace(*ortho_update, -lr_);
                if (!r) return std::unexpected(r.error());
                r = engine_->add_inplace(*params_[i], *ortho_update);
                if (!r) return std::unexpected(r.error());
            }
            else
            {
                // 非 2D 参数（bias 等）：标准 SGD 更新
                auto step = clone_tensor(*engine_, *update_ptr);
                if (!step) return std::unexpected(step.error());
                r = engine_->scale_inplace(*step, -lr_);
                if (!r) return std::unexpected(r.error());
                r = engine_->add_inplace(*params_[i], *step);
                if (!r) return std::unexpected(r.error());
            }
        }
        return {};
    }
};

// ── 优化器工厂函数 ────────────────────────────────────────────────────────
// 根据名称创建对应优化器，支持: "sgd", "sgd_momentum", "adam", "adamw", "muon"（默认 Adam）。
[[nodiscard]] inline std::unique_ptr<Optimizer> create_optimizer(
    std::string_view name,
    ComputeEngine& engine,
    std::vector<Tensor*> params,
    std::vector<Tensor*> grads,
    Scalar lr,
    Scalar weight_decay = 0)
{
    if (name == "sgd")
        return std::make_unique<SGD>(engine, std::move(params), std::move(grads), lr);
    if (name == "sgd_momentum")
        return std::make_unique<SGDWithMomentum>(engine, std::move(params), std::move(grads), lr);
    if (name == "adamw")
        return std::make_unique<AdamW>(engine, std::move(params), std::move(grads), lr,
                                       0.9, 0.999, 1e-8, weight_decay);
    if (name == "muon")
        return std::make_unique<Muon>(engine, std::move(params), std::move(grads), lr);
    return std::make_unique<Adam>(engine, std::move(params), std::move(grads), lr);
}

} // namespace nn

#endif // NN_COMPUTE_OPTIMIZER_HPP
