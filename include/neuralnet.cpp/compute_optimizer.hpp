#pragma once

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
#include <utility>
#include <vector>

#include "compute_engine.hpp"
#include "compute_layer_base.hpp"  // Layer 基类 / clone_tensor 等工具
#include "compute_tensor.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Optimizer — 引擎化优化器基类
//
// 职责：
//   - 持有 engine / params / grads 三元组（所有子类共享）
//   - 提供 zero_grad() 默认实现
//   - 提供 validate_sizes_() / create_zero_buffers_() 公共辅助方法
// ══════════════════════════════════════════════════════════════════════════
class Optimizer
{
protected:
    ComputeEngine& engine_;              // 非拥有引用（永不为空）
    std::vector<TensorRef> params_;      // 非拥有引用（永不为空）
    std::vector<TensorRef> grads_;       // 非拥有引用（永不为空）

    // 多精度（§9.1 / D8）：优化器**状态**（m / v / momentum）的存储精度。
    // 默认全 F32 = 现状（零回归）。f16 训练下状态留在 f32 是更稳的配方
    // （f16 舍入会污染二阶矩），但 profile_all_f16（CLI --f16）按字面语义
    // 取 F16 —— 用户显式选择全 f16 时不予阻拦，只如实报告数值表现。
    // 参数更新（p -= ...）走 in-place，存储精度不可变（§8.3）。
    PrecisionProfile p_;

    // 校验 params/grads 数量一致（各子类 step() 开头调用）
    [[nodiscard]] Result<void> validate_sizes_() const
    {
        if (params_.size() != grads_.size())
            return std::unexpected(Error{"Optimizer: params/grads size mismatch"});
        return {};
    }

    // 为每个参数创建同形状的零初始化 Tensor（供 Momentum/Adam/Muon 复用）
    // 精度 = p_.optimizer（状态存储精度，§9.1）
    [[nodiscard]] Result<std::vector<Tensor>> create_zero_buffers_() const
    {
        std::vector<Tensor> buffers;
        buffers.reserve(params_.size());
        for (auto& p : params_)
        {
            auto buf = engine_.create_tensor(p.get().rows(), p.get().cols(),
                                             p_.optimizer);
            auto r = engine_.zero(buf);
            if (!r) return std::unexpected(r.error());
            buffers.push_back(std::move(buf));
        }
        return buffers;
    }

    // ── 构造期资源分配失败记录（替代 std::abort）─────────────────────────
    // 构造函数无法返回 Result，故把首次失败原因存下，由 step() 开头经
    // check_ready_() 上抛（铁律 1：禁止 throw/abort，错误一律走 Result）。
    std::optional<Error> init_error_;

    void record_init_error_(Error e)
    {
        if (!init_error_) init_error_ = std::move(e);
    }

    [[nodiscard]] Result<void> check_ready_() const
    {
        if (init_error_) return std::unexpected(*init_error_);
        return {};
    }

public:
    Optimizer(ComputeEngine& engine,
              std::vector<TensorRef> params,
              std::vector<TensorRef> grads,
              PrecisionProfile precision = PrecisionProfile{})
        : engine_(engine),
          params_(std::move(params)),
          grads_(std::move(grads)),
          p_(precision) {}

    virtual ~Optimizer() = default;

    // ── D7：精度配置注入（工厂 create_optimizer 在构造时透传）──────────
    void set_precision_profile(const PrecisionProfile& p) { p_ = p; }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

    // 动态调整学习率（供 OscillationGuard 等自适应调度器使用）
    virtual void set_lr(Scalar lr) = 0;

    [[nodiscard]] virtual Result<void> step() = 0;

    // ── 梯度裁剪（max_norm）：全局 L2 范数裁剪 ─────────────────────
    // 计算所有梯度张量的全局 L2 范数，若超过 max_norm 则等比例缩放。
    // 在 step() 之前调用： backward() → clip_grad_norm(max_norm) → step().
    [[nodiscard]] Result<void> clip_grad_norm(Scalar max_norm)
    {
        if (max_norm <= 0 || grads_.empty())
            return {};

        // 计算全局梯度平方和：每个梯度张量归约到 (1,1) 标量后在设备端累加，
        // 最后仅下载一次（而非每张量一次 PCIe 往返）。
        Tensor acc;
        for (auto& g_ref : grads_)
        {
            auto& g = g_ref.get();
            // g² = g * g（逐元素乘法）
            auto g_sq_r = dsl::compute(engine_, dsl::leaf(g) * dsl::leaf(g),
                                       g.rows(), g.cols());
            if (!g_sq_r) return std::unexpected(g_sq_r.error());
            // 按行求和 → (rows, 1)
            auto row_sums_r = engine_.row_reduce_sum(*g_sq_r);
            if (!row_sums_r) return std::unexpected(row_sums_r.error());
            // 按列求和 → (1, 1) 标量张量
            auto col_sum_r = engine_.col_reduce_sum(*row_sums_r);
            if (!col_sum_r) return std::unexpected(col_sum_r.error());

            if (acc.valid())
            {
                auto sum_r = dsl::compute(engine_, dsl::leaf(acc) + dsl::leaf(*col_sum_r),
                                          acc.rows(), acc.cols());
                if (!sum_r) return std::unexpected(sum_r.error());
                acc = std::move(*sum_r);
            }
            else
            {
                acc = std::move(*col_sum_r);
            }
        }

        // 仅一次下载获取全局平方和
        auto m_r = engine_.to_matrix(acc);
        if (!m_r) return std::unexpected(m_r.error());
        const Scalar total_sq = m_r->at(0, 0);

        Scalar norm = std::sqrt(total_sq);
        if (norm <= max_norm)
            return {};

        // 等比例缩放所有梯度（目标传递：原地、单 dispatch）
        Scalar scale = max_norm / norm;
        for (auto& g_ref : grads_)
        {
            auto r = dsl::compute_into(engine_,
                dsl::leaf(g_ref.get()) * dsl::rparam(scale), g_ref.get());
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

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
// 原语：axpy_inplace(p, -lr, g)
// ══════════════════════════════════════════════════════════════════════════
class SGD : public Optimizer
{
    Scalar lr_;

public:
    SGD(ComputeEngine& engine,
        std::vector<TensorRef> params,
        std::vector<TensorRef> grads,
        Scalar lr,
        PrecisionProfile precision = PrecisionProfile{})
        : Optimizer(engine, std::move(params), std::move(grads), precision), lr_(lr) {}

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (auto r = validate_sizes_(); !r) return std::unexpected(r.error());

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // p -= lr * g（目标传递：原地、单 dispatch）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(params_[i]) + dsl::leaf(grads_[i]) * dsl::rparam(-lr_),
                params_[i]);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SGDWithMomentum — 带动量的 SGD
//
// 算法：v = β*v + (1-β)*g;  p -= lr*v
// 原语：axpy_inplace(v, 1-β, g)  →  axpy_inplace(p, -lr, v)
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
                     Scalar lr, Scalar beta = 0.9,
                     PrecisionProfile precision = PrecisionProfile{})
        : Optimizer(engine, std::move(params), std::move(grads), precision),
          lr_(lr), beta_(beta)
    {
        auto v_r = create_zero_buffers_();
        if (!v_r)
            record_init_error_(v_r.error());  // 不 abort：由 step() 经 Result 上抛
        else
            velocities_ = std::move(*v_r);
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (auto r = check_ready_(); !r) return std::unexpected(r.error());
        if (auto r = validate_sizes_(); !r) return std::unexpected(r.error());

        const Scalar one_minus_beta = Scalar{1} - beta_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // v = β*v + (1-β)*g（两趟原语融合为单 dispatch 的原地目标传递）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(velocities_[i]) * dsl::rparam(beta_)
                    + dsl::leaf(grads_[i]) * dsl::rparam(one_minus_beta),
                velocities_[i]);
            if (!r) return std::unexpected(r.error());

            // p -= lr * v（原地、单 dispatch）
            r = dsl::compute_into(engine_,
                dsl::leaf(params_[i]) + dsl::leaf(velocities_[i]) * dsl::rparam(-lr_),
                params_[i]);
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
    // inv_bc1/inv_bc2 由 step() 提前计算（每步仅一次 pow），失败时不推进 t。
    //
    // 融合改造（消除 GPU Copy / 中间缓冲）：
    //   旧实现对 m/v 各做一次 clone_tensor（整份模型尺寸 vkCmdCopyBuffer，逐
    //   步 2×model_size 的 Copy）+ 多次逐元素原语 + 多个中间张量。现改为三个
    //   DSL 融合 kernel，全部超参（β1/β2/eps/lr/inv_bc1/inv_bc2）经 RParam
    //   （运行时标量）承载——值不进 expr_spec_key，同结构共享 fusion shader，
    //   引擎适应计算。无任何 clone、无 m_hat/v_hat/sqrt_v/denom/ratio 物化：
    //     K1  m = β1*m + (1-β1)*g            （m_ owned，直接重赋值）
    //     K2  v = β2*v + (1-β2)*g²           （v_ owned，直接重赋值）
    //     K3  p -= lr * (inv_bc1*m) / (sqrt(inv_bc2*v)+eps)（p 为模型张量，add_inplace 就地）
    [[nodiscard]] Result<void> adam_update_(
        std::size_t i, Scalar inv_bc1, Scalar inv_bc2)
    {
        const Scalar one_minus_beta1 = Scalar{1} - beta1_;
        const Scalar one_minus_beta2 = Scalar{1} - beta2_;
        const Tensor& g = grads_[i];
        const std::size_t rows = g.rows(), cols = g.cols();

        // K1: m = β1*m + (1-β1)*g（单 kernel 融合；状态精度 = p_.optimizer）
        auto m_new = dsl::compute(engine_,
            dsl::leaf(m_[i]) * dsl::rparam(beta1_) +
                dsl::leaf(g) * dsl::rparam(one_minus_beta1),
            rows, cols, p_.optimizer);
        if (!m_new) return std::unexpected(m_new.error());
        m_[i] = std::move(*m_new);

        // K2: v = β2*v + (1-β2)*g²（单 kernel 融合；状态精度 = p_.optimizer）
        auto v_new = dsl::compute(engine_,
            dsl::leaf(v_[i]) * dsl::rparam(beta2_) +
                dsl::leaf(g) * dsl::leaf(g) * dsl::rparam(one_minus_beta2),
            rows, cols, p_.optimizer);
        if (!v_new) return std::unexpected(v_new.error());
        v_[i] = std::move(*v_new);

        // K3: p -= lr * (inv_bc1*m) / (sqrt(inv_bc2*v)+eps)（全链单 kernel 融合；
        //     依赖刚更新的 m_[i]/v_[i]，偏置修正系数 inv_bc1/inv_bc2 逐步变化
        //     由 RParam 承载，不进 key → 共享 shader）
        //     更新量精度 = p_.param（与目标参数存储精度一致，省一次 cast）
        auto delta = dsl::compute(engine_,
              -dsl::rparam(lr_)
              * ((dsl::leaf(m_[i]) * dsl::rparam(inv_bc1))
                 / (dsl::sqrt(dsl::leaf(v_[i]) * dsl::rparam(inv_bc2))
                    + dsl::rparam(eps_))),
            rows, cols, p_.param);
        if (!delta) return std::unexpected(delta.error());
        // p += delta（目标传递：原地、单 dispatch，不额外分配）
        auto r = dsl::compute_into(engine_,
            dsl::leaf(params_[i]) + dsl::leaf(*delta), params_[i]);
        if (!r) return std::unexpected(r.error());
        return {};
    }

    void init_moments_()
    {
        auto m_r = create_zero_buffers_();
        if (!m_r)
        {
            record_init_error_(m_r.error());  // 不 abort：由 step() 经 Result 上抛
            return;
        }
        m_ = std::move(*m_r);
        auto v_r = create_zero_buffers_();
        if (!v_r)
        {
            record_init_error_(v_r.error());
            return;
        }
        v_ = std::move(*v_r);
    }

    // 偏差修正系数（Adam/AdamW 共用）：给定下一步步数 t_next，
    // 返回 {inv_bc1, inv_bc2}，其中 bc = 1 - β^t_next。
    [[nodiscard]] std::pair<Scalar, Scalar> bias_correction_(std::size_t t_next) const
    {
        const Scalar bc1 = Scalar{1} - std::pow(beta1_, static_cast<Scalar>(t_next));
        const Scalar bc2 = Scalar{1} - std::pow(beta2_, static_cast<Scalar>(t_next));
        return {Scalar{1} / bc1, Scalar{1} / bc2};
    }

public:
    Adam(ComputeEngine& engine,
         std::vector<TensorRef> params,
         std::vector<TensorRef> grads,
         Scalar lr,
         Scalar beta1 = 0.9,
         Scalar beta2 = 0.999,
         Scalar eps = 1e-8,
         PrecisionProfile precision = PrecisionProfile{})
        : Optimizer(engine, std::move(params), std::move(grads), precision),
          lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0)
    {
        init_moments_();
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (auto r = check_ready_(); !r) return std::unexpected(r.error());
        if (auto r = validate_sizes_(); !r) return std::unexpected(r.error());

        // 偏差修正：每步只计算一次（而非每参数重复 pow）
        const std::size_t t_next = t_ + 1;
        const auto [inv_bc1, inv_bc2] = bias_correction_(t_next);

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            auto r = adam_update_(i, inv_bc1, inv_bc2);
            if (!r) return std::unexpected(r.error());
        }
        t_ = t_next;  // 全部参数成功才推进步数（失败时不推进）
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
          Scalar weight_decay = 0.01,
          PrecisionProfile precision = PrecisionProfile{})
        : Adam(engine, std::move(params), std::move(grads),
               lr, beta1, beta2, eps, precision),
          wd_(weight_decay) {}

    // set_lr 复用 Adam::set_lr（lr_ 为 protected，无需 override）

    [[nodiscard]] Result<void> step() override
    {
        if (auto r = check_ready_(); !r) return std::unexpected(r.error());
        if (auto r = validate_sizes_(); !r) return std::unexpected(r.error());

        // 偏差修正：每步只计算一次（而非每参数重复 pow）
        const std::size_t t_next = t_ + 1;
        const auto [inv_bc1, inv_bc2] = bias_correction_(t_next);
        const Scalar decay_factor = Scalar{1} - lr_ * wd_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // 权重衰减解耦：p = (1 - lr*wd) * p（目标传递：原地、单 dispatch）
            if (wd_ != Scalar{0})
            {
                auto r = dsl::compute_into(engine_,
                    dsl::leaf(params_[i]) * dsl::rparam(decay_factor), params_[i]);
                if (!r) return std::unexpected(r.error());
            }

            // Adam 更新
            auto r = adam_update_(i, inv_bc1, inv_bc2);
            if (!r) return std::unexpected(r.error());
        }
        t_ = t_next;  // 全部参数成功才推进步数（失败时不推进）
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
    ComputeEngine& engine, const Tensor& G, std::size_t steps = 5, Scalar eps = 1e-7f,
    Precision prec = Precision::F32)
{
    // 调优后的 quintic 多项式系数（使 φ^N(x) → 1 for x ∈ [0,1]）
    constexpr Scalar a = 3.4445f;
    constexpr Scalar b = -4.7750f;
    constexpr Scalar c = 2.0315f;

    // 计算 Frobenius 范数的平方：||G||_F² = Σ g_ij²
    // 逐元素链与行归约**融合为单 dispatch**（不物化 norm_sq 中间张量），
    // 再对 (rows,1) 小向量做列归约 → (1,1)
    auto row_sum_norm = dsl::compute_reduce(engine,
        dsl::row_reduce_sum(dsl::leaf(G) * dsl::leaf(G)), G.rows(), G.cols(), prec);
    if (!row_sum_norm) return std::unexpected(row_sum_norm.error());
    auto total_norm_sq = engine.col_reduce_sum(*row_sum_norm, prec);
    if (!total_norm_sq) return std::unexpected(total_norm_sq.error());

    // 从 (1,1) Tensor 提取标量值
    // 使用 to_matrix 拉到 CPU 后取值（一次 PCIe 下载，可接受）
    auto total_norm_sq_mat = engine.to_matrix(*total_norm_sq);
    if (!total_norm_sq_mat) return std::unexpected(total_norm_sq_mat.error());
    Scalar norm_sq_val = total_norm_sq_mat->at(0, 0);
    Scalar inv_norm_scalar = Scalar{1} / std::sqrt(norm_sq_val + eps * eps);

    // X = G * inv_norm_scalar（归一化）：单表达式（取代 clone 整块拷贝 + scale
    // 两次 dispatch；GPU 上 1 个融合 kernel + 1 次分配）
    auto X = dsl::compute(engine,
        dsl::leaf(G) * dsl::rparam(inv_norm_scalar), G.rows(), G.cols(), prec);
    if (!X) return std::unexpected(X.error());

    // 选更小一侧构造母矩阵，避免显存爆炸（Muon 显存 > AdamW 的根因）：
    //   - 短宽/方阵（m ≤ n）：行正交化，母矩阵 A = X·X^T（m×m，m 为短边）
    //   - 高窄矩阵（m > n）：列正交化，母矩阵 G = X^T·X（n×n，n 为短边）
    // 故母矩阵恒为 min(m,n)² 而非 max(m,n)²。对高窄大参数（如 50257×1024 的
    // 词嵌入），旧实现构造 50257² 的 A/A²（≈10GB/个）会 OOM；新实现降至
    // 1024²（≈4MB）。且列正交化要求 n ≤ m，高窄时"列"是唯一可达的近正交目标，
    // 数学上与参考实现（对短边一侧正交化）一致。
    const std::size_t m = G.rows();
    const std::size_t n = G.cols();
    const bool tall = m > n;

    if (!tall)
    {
        // 行正交化：X ← (a + bA + cA²)·X，A = X·X^T（m×m）
        // 每步 5 次 dispatch：matmul(A) + matmul(A²) + 融合原地 A=bA+cA²
        //                  + matmul(BX) + 融合原地 BX+=aX
        // （原实现 7 次：多出 scale(A²,c)/scale(A,b)/add(A,A²) 三次，现合并为
        //   一次原地目标传递）
        for (std::size_t t = 0; t < steps; ++t)
        {
            auto A = engine.matmul(*X, *X, false, true, prec);     // A = X·X^T
            if (!A) return std::unexpected(A.error());
            auto A_sq = engine.matmul(*A, *A, false, false, prec); // A²
            if (!A_sq) return std::unexpected(A_sq.error());

            // A = b·A + c·A²：原为 scale(A²,c) + scale(A,b) + add(A,A²) 三次
            // dispatch，融合为一次原地目标传递（GPU 上 1 个 kernel）
            auto accA = dsl::compute_into(engine,
                dsl::leaf(*A) * dsl::rparam(b) + dsl::leaf(*A_sq) * dsl::rparam(c), *A);
            if (!accA) return std::unexpected(accA.error());

            auto BX = engine.matmul(*A, *X, false, false, prec);   // B·X
            if (!BX) return std::unexpected(BX.error());
            // BX += a·X（原地目标传递：单 dispatch，不额外分配）
            auto accBX = dsl::compute_into(engine,
                dsl::leaf(*BX) + dsl::leaf(*X) * dsl::rparam(a), *BX);
            if (!accBX) return std::unexpected(accBX.error());

            X = std::move(*BX);
        }
    }
    else
    {
        // 列正交化：X ← X·(a + bG + cG²)，G = X^T·X（n×n，n < m）
        // 由 X_new^T·X_new = (aI+bG+cG²)·G·(...)=φ(G)²·G，驱动 G → I（列正交）。
        // 母矩阵恒为 n×n，高窄时远小于 m×m。
        for (std::size_t t = 0; t < steps; ++t)
        {
            auto Gr = engine.matmul(*X, *X, true, false, prec);    // G = X^T·X（n×n）
            if (!Gr) return std::unexpected(Gr.error());
            auto Gr_sq = engine.matmul(*Gr, *Gr, false, false, prec); // G²
            if (!Gr_sq) return std::unexpected(Gr_sq.error());

            // G = b·G + c·G²（同上：三次 dispatch 融合为一次原地目标传递）
            auto accG = dsl::compute_into(engine,
                dsl::leaf(*Gr) * dsl::rparam(b) + dsl::leaf(*Gr_sq) * dsl::rparam(c), *Gr);
            if (!accG) return std::unexpected(accG.error());

            auto XM = engine.matmul(*X, *Gr, false, false, prec);  // X·G
            if (!XM) return std::unexpected(XM.error());
            // XM += a·X（原地目标传递：单 dispatch）
            auto accXM = dsl::compute_into(engine,
                dsl::leaf(*XM) + dsl::leaf(*X) * dsl::rparam(a), *XM);
            if (!accXM) return std::unexpected(accXM.error());

            X = std::move(*XM);
        }
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
         Scalar ns_eps = 1e-7f,
         PrecisionProfile precision = PrecisionProfile{})
        : Optimizer(engine, std::move(params), std::move(grads), precision),
          lr_(lr), momentum_(momentum), nesterov_(nesterov),
          ns_steps_(ns_steps), ns_eps_(ns_eps)
    {
        auto v_r = create_zero_buffers_();
        if (!v_r)
            record_init_error_(v_r.error());  // 不 abort：由 step() 经 Result 上抛
        else
            velocities_ = std::move(*v_r);
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        if (auto r = check_ready_(); !r) return std::unexpected(r.error());
        if (auto r = validate_sizes_(); !r) return std::unexpected(r.error());

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            const Tensor& g = grads_[i];

            // 1. SGD-Momentum: v = μ*v + g（两趟原语融合为单 dispatch）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(velocities_[i]) * dsl::rparam(momentum_) + dsl::leaf(g),
                velocities_[i]);
            if (!r) return std::unexpected(r.error());

            // 确定用于正交化的更新方向（Nesterov 时需临时缓冲，否则直接用 v）
            std::optional<Tensor> nesterov_buf;
            if (nesterov_)
            {
                // Nesterov: update = g + μ*v（单表达式：一次 dispatch + 一次分配，
                // 取代 clone(整块拷贝) + axpy 两次 dispatch）
                auto buf = dsl::compute(engine_,
                    dsl::leaf(g) + dsl::leaf(velocities_[i]) * dsl::rparam(momentum_),
                    g.rows(), g.cols(), p_.optimizer);
                if (!buf) return std::unexpected(buf.error());
                nesterov_buf = std::move(*buf);
            }
            const Tensor& update = nesterov_buf ? *nesterov_buf : velocities_[i];

            // 2. Newton-Schulz 正交化（仅对 ≥2D 参数，即 rows > 1 且 cols > 1）
            if (params_[i].get().rows() > 1 && params_[i].get().cols() > 1)
            {
                auto ortho_update = newton_schulz_orthogonalize(
                    engine_, update, ns_steps_, ns_eps_, p_.optimizer);
                if (!ortho_update) return std::unexpected(ortho_update.error());

                // 3. 参数更新: p -= lr * 0.2 * sqrt(max(m,n)) * NS(update)
                //    NorMuon 论文 / KellerJordan 参考实现的形状缩放：NS 输出谱范数为 1，
                //    不缩放则等效学习率偏差 0.2*sqrt(max(m,n)) 倍（如 256×768 权重 → 5.5×）
                const std::size_t m = params_[i].get().rows();
                const std::size_t n = params_[i].get().cols();
                const std::size_t big = m > n ? m : n;
                const Scalar muon_scale =
                    Scalar{0.2} * std::sqrt(static_cast<Scalar>(big));
                r = dsl::compute_into(engine_,
                    dsl::leaf(params_[i])
                        + dsl::leaf(*ortho_update) * dsl::rparam(-lr_ * muon_scale),
                    params_[i]);
                if (!r) return std::unexpected(r.error());
            }
            else
            {
                // 非 2D 参数（bias 等）：标准 SGD 更新（目标传递：原地、单 dispatch）
                r = dsl::compute_into(engine_,
                    dsl::leaf(params_[i]) + dsl::leaf(update) * dsl::rparam(-lr_),
                    params_[i]);
                if (!r) return std::unexpected(r.error());
            }
        }
        return {};
    }
};

// ── 优化器工厂函数 ────────────────────────────────────────────────────────
// 根据名称创建对应优化器，支持: "sgd", "sgd_momentum", "adam", "adamw", "muon"。
// 未知名称返回 nullptr，由调用方处理。
// precision：模型级精度配置（§9.1）——状态张量按 p.optimizer 创建，参数更新
// 走 in-place（存储精度不可变）。默认全 F32 = 现状（零回归）。
[[nodiscard]] inline std::unique_ptr<Optimizer> create_optimizer(
    std::string_view name,
    ComputeEngine& engine,
    std::vector<TensorRef> params,
    std::vector<TensorRef> grads,
    Scalar lr,
    Scalar weight_decay = 0,
    PrecisionProfile precision = PrecisionProfile{})
{
    if (name == "sgd")
        return std::make_unique<SGD>(engine, std::move(params), std::move(grads), lr,
                                     precision);
    if (name == "sgd_momentum")
        return std::make_unique<SGDWithMomentum>(engine, std::move(params),
                                                 std::move(grads), lr, 0.9, precision);
    if (name == "adam")
        return std::make_unique<Adam>(engine, std::move(params), std::move(grads), lr,
                                      0.9, 0.999, 1e-8, precision);
    if (name == "adamw")
        return std::make_unique<AdamW>(engine, std::move(params), std::move(grads), lr,
                                       0.9, 0.999, 1e-8, weight_decay, precision);
    if (name == "muon")
        return std::make_unique<Muon>(engine, std::move(params), std::move(grads), lr,
                                      0.95f, true, 5, 1e-7f, precision);
    return nullptr;  // 未知名称，调用方应处理
}

} // namespace nn

