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
    ComputeEngine& engine_;              // 非拥有引用（永不为空）
    std::vector<TensorRef> params_;      // 非拥有引用（永不为空）
    std::vector<TensorRef> grads_;       // 非拥有引用（永不为空）

    // 辅助方法：dst += scalar * src（融合 axpy：单次 dispatch 替代 clone+scale+add 三步）
    // 用于 SGD / Momentum / Adam 等优化器中常见的 axpy 操作
    //
    // 性能改进（依据性能审查报告）：
    //   - 旧实现：clone_tensor + scale_inplace + add_inplace = 3 个 GPU 原语 + 3 个临时 buffer
    //   - 新实现：axpy_inplace = 1 个 GPU 原语 + 1 个临时 buffer
    //   - 100 个参数 × 每 step 调用 ~3 次 = 减少 600 次 GPU buffer 分配/step
    [[nodiscard]] Result<void> scale_add_(Tensor& dst, Scalar scalar, const Tensor& src)
    {
        return engine_.axpy_inplace(dst, scalar, src);
    }

public:
    Optimizer(ComputeEngine& engine,
              std::vector<TensorRef> params,
              std::vector<TensorRef> grads)
        : engine_(engine),
          params_(std::move(params)),
          grads_(std::move(grads)) {}

    virtual ~Optimizer() = default;

    // 动态调整学习率（供 OscillationGuard 等自适应调度器使用）
    virtual void set_lr(Scalar lr) = 0;

    [[nodiscard]] virtual Result<void> step() = 0;

    // 默认实现：将所有梯度清零（所有子类行为一致）
    [[nodiscard]] virtual Result<void> zero_grad()
    {
        for (auto& g : grads_)
        {
            auto r = engine_.zero(g);
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
        std::vector<TensorRef> params,
        std::vector<TensorRef> grads,
        Scalar lr)
        : Optimizer(engine, std::move(params), std::move(grads)), lr_(lr) {}

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"SGD: params/grads size mismatch"});

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            auto r = scale_add_(params_[i], -lr_, grads_[i]);
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
                     std::vector<TensorRef> params,
                     std::vector<TensorRef> grads,
                     Scalar lr, Scalar beta = 0.9)
        : Optimizer(engine, std::move(params), std::move(grads)),
          lr_(lr), beta_(beta)
    {
        velocities_.reserve(params_.size());
        for (auto& p : params_)
        {
            auto v = engine_.create_tensor(p.get().rows(), p.get().cols());
            { auto r = engine_.zero(v); NN_ASSERT(r, r ? "" : r.error().message.c_str()); }
            velocities_.push_back(std::move(v));
        }
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Momentum: params/grads size mismatch"});

        const Scalar one_minus_beta = Scalar{1} - beta_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // v = β*v + (1-β)*g
            auto r = engine_.scale_inplace(velocities_[i], beta_);
            if (!r) return std::unexpected(r.error());
            r = scale_add_(velocities_[i], one_minus_beta, grads_[i]);
            if (!r) return std::unexpected(r.error());

            // p -= lr * v
            r = scale_add_(params_[i], -lr_, velocities_[i]);
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

        const Tensor& g = grads_[i];

        // m = β1*m + (1-β1)*g
        auto r = engine_.scale_inplace(m_[i], beta1_);
        if (!r) return std::unexpected(r.error());
        r = scale_add_(m_[i], one_minus_beta1, g);
        if (!r) return std::unexpected(r.error());

        // v = β2*v + (1-β2)*g²
        r = engine_.scale_inplace(v_[i], beta2_);
        if (!r) return std::unexpected(r.error());
        auto g_sq = engine_.elementwise_binary(BinaryOp::Mul, g, g);
        if (!g_sq) return std::unexpected(g_sq.error());
        r = engine_.scale_inplace(*g_sq, one_minus_beta2);
        if (!r) return std::unexpected(r.error());
        r = engine_.add_inplace(v_[i], *g_sq);
        if (!r) return std::unexpected(r.error());

        // m_hat = m / bc1
        auto m_hat = clone_tensor(engine_, m_[i]);
        if (!m_hat) return std::unexpected(m_hat.error());
        r = engine_.scale_inplace(*m_hat, inv_bc1);
        if (!r) return std::unexpected(r.error());

        // v_hat = v / bc2
        auto v_hat = clone_tensor(engine_, v_[i]);
        if (!v_hat) return std::unexpected(v_hat.error());
        r = engine_.scale_inplace(*v_hat, inv_bc2);
        if (!r) return std::unexpected(r.error());

        // sqrt_v = sqrt(v_hat)
        auto sqrt_v = engine_.elementwise_unary(UnaryOp::Sqrt, *v_hat);
        if (!sqrt_v) return std::unexpected(sqrt_v.error());

        // denom = sqrt_v + eps
        auto denom = engine_.elementwise_binary_scalar(BinaryOp::Add, *sqrt_v, eps_);
        if (!denom) return std::unexpected(denom.error());

        // ratio = m_hat / denom
        auto ratio = engine_.elementwise_binary(BinaryOp::Div, *m_hat, *denom);
        if (!ratio) return std::unexpected(ratio.error());

        // p -= lr * ratio
        r = engine_.scale_inplace(*ratio, -lr_);
        if (!r) return std::unexpected(r.error());
        return engine_.add_inplace(params_[i], *ratio);
    }

    void init_moments_()
    {
        m_.reserve(params_.size());
        v_.reserve(params_.size());
        for (auto& p : params_)
        {
            auto mt = engine_.create_tensor(p.get().rows(), p.get().cols());
            auto vt = engine_.create_tensor(p.get().rows(), p.get().cols());
            { auto r1 = engine_.zero(mt); NN_ASSERT(r1, r1 ? "" : r1.error().message.c_str()); }
            { auto r2 = engine_.zero(vt); NN_ASSERT(r2, r2 ? "" : r2.error().message.c_str()); }
            m_.push_back(std::move(mt));
            v_.push_back(std::move(vt));
        }
    }

public:
    Adam(ComputeEngine& engine,
         std::vector<TensorRef> params,
         std::vector<TensorRef> grads,
         Scalar lr,
         Scalar beta1 = 0.9,
         Scalar beta2 = 0.999,
         Scalar eps = 1e-8)
        : Optimizer(engine, std::move(params), std::move(grads)),
          lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0)
    {
        init_moments_();
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

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
          std::vector<TensorRef> params,
          std::vector<TensorRef> grads,
          Scalar lr,
          Scalar beta1 = 0.9,
          Scalar beta2 = 0.999,
          Scalar eps = 1e-8,
          Scalar weight_decay = 0.01)
        : Adam(engine, std::move(params), std::move(grads),
               lr, beta1, beta2, eps),
          wd_(weight_decay) {}

    void set_lr(Scalar lr) override { lr_ = lr; }

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
                auto r = engine_.scale_inplace(params_[i], decay_factor);
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

    // 计算 Frobenius 范数的平方：||G||_F² = Σ g_ij²
    // 通过 elementwise(Mul) → row_reduce_sum → col_reduce_sum 三步原语得到 (1,1) 张量
    auto norm_sq = engine.elementwise_binary(BinaryOp::Mul, G, G);
    if (!norm_sq) return std::unexpected(norm_sq.error());
    auto row_sum_norm = engine.row_reduce_sum(*norm_sq);
    if (!row_sum_norm) return std::unexpected(row_sum_norm.error());
    auto total_norm_sq = engine.col_reduce_sum(*row_sum_norm);
    if (!total_norm_sq) return std::unexpected(total_norm_sq.error());

    // 从 (1,1) Tensor 提取标量值
    // 使用 to_matrix 拉到 CPU 后取值（一次 PCIe 下载，可接受）
    auto total_norm_sq_mat = engine.to_matrix(*total_norm_sq);
    if (!total_norm_sq_mat) return std::unexpected(total_norm_sq_mat.error());
    Scalar norm_sq_val = total_norm_sq_mat->at(0, 0);
    Scalar inv_norm_scalar = Scalar{1} / std::sqrt(norm_sq_val + eps * eps);

    // X = G * inv_norm_scalar（一次标量乘法归一化）
    auto X = clone_tensor(engine, G);
    if (!X) return std::unexpected(X.error());
    auto r = engine.scale_inplace(*X, inv_norm_scalar);
    if (!r) return std::unexpected(r.error());

    // Newton-Schulz 迭代：X_new = a*X + (b*A + c*A²) @ X，其中 A = X @ X^T
    //
    // 性能优化：通过就地复用 matmul 输出缓冲区，消除每步 3 个 clone_tensor 调用：
    //   - A 缓冲区就地修改为 B = b*A + c*A²（A 在计算 B 后不再需要）
    //   - BX 缓冲区就地添加 a*X 得到 X_new（BX 在 aX 加法后不再需要）
    //   - 原实现每步 6 个临时 buffer → 优化后 0 个额外 clone
    for (std::size_t t = 0; t < steps; ++t)
    {
        // A = X @ X^T（matmul 返回新 tensor，无分配开销）
        auto A = engine.matmul(*X, *X, false, true);
        if (!A) return std::unexpected(A.error());

        // A_sq = A @ A（matmul 返回新 tensor）
        auto A_sq = engine.matmul(*A, *A);
        if (!A_sq) return std::unexpected(A_sq.error());

        // B = b*A + c*A²
        // 就地复用：A_sq 不再需要后修改为 c*A²，A 不再需要后修改为 b*A
        r = engine.scale_inplace(*A_sq, c);
        if (!r) return std::unexpected(r.error());
        r = engine.scale_inplace(*A, b);
        if (!r) return std::unexpected(r.error());
        r = engine.add_inplace(*A, *A_sq);  // A 现在是 B = b*A + c*A²
        if (!r) return std::unexpected(r.error());

        // X_new = a*X + B @ X
        // 就地复用：BX 是 matmul 新输出，axpy_inplace 就地添加 a*X
        auto BX = engine.matmul(*A, *X);
        if (!BX) return std::unexpected(BX.error());
        r = engine.axpy_inplace(*BX, a, *X);  // BX += a*X → (B + aI)*X
        if (!r) return std::unexpected(r.error());

        X = std::move(*BX);
    }

    return *X;
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
         std::vector<TensorRef> params,
         std::vector<TensorRef> grads,
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
        for (auto& p : params_)
        {
            auto v = engine_.create_tensor(p.get().rows(), p.get().cols());
            { auto r = engine_.zero(v); NN_ASSERT(r, r ? "" : r.error().message.c_str()); }
            velocities_.push_back(std::move(v));
        }
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Muon: params/grads size mismatch"});

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            const Tensor& g = grads_[i];

            // 1. SGD-Momentum: v = μ*v + g
            auto r = engine_.scale_inplace(velocities_[i], momentum_);
            if (!r) return std::unexpected(r.error());
            r = engine_.add_inplace(velocities_[i], g);
            if (!r) return std::unexpected(r.error());

            // 确定用于正交化的更新方向
            Tensor* update_ptr = nullptr;
            std::optional<Tensor> nesterov_buf;
            if (nesterov_)
            {
                // Nesterov: update = g + μ*v
                // 优化：clone g 一次，用 axpy_inplace 就地添加 μ*v（原实现需要 2 次 clone + scale + add）
                auto buf = clone_tensor(engine_, g);
                if (!buf) return std::unexpected(buf.error());
                r = engine_.axpy_inplace(*buf, momentum_, velocities_[i]);
                if (!r) return std::unexpected(r.error());
                nesterov_buf = std::move(*buf);
                update_ptr = &*nesterov_buf;
            }
            else
            {
                update_ptr = &velocities_[i];
            }

            // 2. Newton-Schulz 正交化（仅对 ≥2D 参数，即 rows > 1 且 cols > 1）
            if (params_[i].get().rows() > 1 && params_[i].get().cols() > 1)
            {
                auto ortho_update = newton_schulz_orthogonalize(
                    engine_, *update_ptr, ns_steps_, ns_eps_);
                if (!ortho_update) return std::unexpected(ortho_update.error());

                // 3. 参数更新: p -= lr * ortho_update（用 axpy_inplace 融合 scale+add）
                r = engine_.axpy_inplace(params_[i], -lr_, *ortho_update);
                if (!r) return std::unexpected(r.error());
            }
            else
            {
                // 非 2D 参数（bias 等）：标准 SGD 更新（用 axpy_inplace 融合 scale+add）
                r = engine_.axpy_inplace(params_[i], -lr_, *update_ptr);
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
    std::vector<TensorRef> params,
    std::vector<TensorRef> grads,
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
