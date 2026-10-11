#pragma once

// ── compute_optimizer.hpp — 引擎化优化器 ───────────────────────────────────
//
// 架构铁律：
//   1. Optimizer 的 step/zero_grad 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。
//   2. 优化器算法只通过 engine 原语 + 表达式 DSL（dsl::compute_into /
//      scale_inplace / add_inplace 等）表达，不直接操作 Matrix。
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
#include <cstdint>   // RV-04：OptimizerMeta::state_bufs
#include <optional>
#include <span>      // RV-04：optimizer_table()
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
    // 默认全 F32。f16 训练下状态留在 f32 是更稳的配方
    // （f16 舍入会污染二阶矩），但 profile_all_f16（四字段全 F16 的实验配方）按字面语义
    // 取 F16 —— 用户显式选择全 f16 时不予阻拦，只如实报告数值表现。
    // 参数更新（p -= ...）走 in-place，存储精度不可变（§8.3）。
    PrecisionProfile p_;

    // 校验 params/grads 数量一致（各子类 step() 开头调用）
    [[nodiscard]] Result<void> validate_sizes_() const
    {
        if (params_.size() != grads_.size())
            NN_FAIL("Optimizer: params/grads size mismatch");
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
            NN_TRY_CHECK(r);
            buffers.push_back(std::move(buf));
        }
        return buffers;
    }

    // ── 构造期资源分配失败记录 ────────────────────────────────────────
    // 构造函数无法返回 Result，故把首次失败原因存下，由 step() 开头经
    // check_ready_() 上抛（铁律 1：禁止 throw/abort，错误一律走 Result）。
    std::optional<Error> init_error_;

    void record_init_error_(Error e)
    {
        if (!init_error_) init_error_ = std::move(e);
    }

    [[nodiscard]] Result<void> check_ready_() const
    {
        if (init_error_) NN_FAIL(*init_error_);
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
            NN_TRY_CHECK(g_sq_r);
            // 按行求和 → (rows, 1)（dsl::compute_reduce）
            auto row_sums_r = dsl::compute_reduce(engine_,
                dsl::row_reduce_sum(dsl::leaf(*g_sq_r)),
                g_sq_r->rows(), g_sq_r->cols());
            NN_TRY_CHECK(row_sums_r);
            // 按列求和 → (1, 1) 标量张量（dsl::compute_reduce）
            auto col_sum_r = dsl::compute_reduce(engine_,
                dsl::col_reduce_sum(dsl::leaf(*row_sums_r)),
                row_sums_r->rows(), row_sums_r->cols());
            NN_TRY_CHECK(col_sum_r);
            if (acc.valid())
            {
                auto sum_r = dsl::compute(engine_, dsl::leaf(acc) + dsl::leaf(*col_sum_r),
                                          acc.rows(), acc.cols());
                NN_TRY_CHECK(sum_r);
                acc = std::move(*sum_r);
            }
            else
            {
                acc = std::move(*col_sum_r);
            }
        }

        // 仅一次下载获取全局平方和（宿主桥，17 §3 D11）
        std::vector<Scalar> total_buf(acc.rows() * acc.cols());
        auto dl_r = detail::download_span(engine_, acc, std::span(total_buf));
        NN_TRY_CHECK(dl_r);
        const Scalar total_sq = total_buf[0];

        Scalar norm = std::sqrt(total_sq);
        if (norm <= max_norm)
            return {};

        // 等比例缩放所有梯度（目标传递：原地、单 dispatch）
        Scalar scale = max_norm / norm;
        for (auto& g_ref : grads_)
        {
            auto r = dsl::compute_into(engine_,
                dsl::leaf(g_ref.get()) * dsl::rparam(scale), g_ref.get());
            NN_TRY_CHECK(r);
        }
        return {};
    }

    // 默认实现：将所有梯度清零（所有子类行为一致）
    [[nodiscard]] virtual Result<void> zero_grad()
    {
        for (auto& g : grads_)
        {
            auto r = engine_.zero(g);
            NN_TRY_CHECK(r);
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SGD — 随机梯度下降
//
// 算法：p -= lr * g
// 实现：dsl::compute_into(leaf(p) + leaf(g) * rparam(-lr), p)
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
        NN_TRY(r, validate_sizes_());
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // p -= lr * g（目标传递：原地、单 dispatch）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(params_[i]) + dsl::leaf(grads_[i]) * dsl::rparam(-lr_),
                params_[i]);
            NN_TRY_CHECK(r);
        }
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// SGDWithMomentum — 带动量的 SGD
//
// 算法：v = β*v + (1-β)*g;  p -= lr*v
// 实现：dsl::compute_into 两步（v = β·v + (1-β)·g；p -= lr·v）
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
        NN_TRY(r, check_ready_());
        NN_TRY(r2, validate_sizes_());
        const Scalar one_minus_beta = Scalar{1} - beta_;

        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // v = β*v + (1-β)*g（两趟原语融合为单 dispatch 的原地目标传递）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(velocities_[i]) * dsl::rparam(beta_)
                    + dsl::leaf(grads_[i]) * dsl::rparam(one_minus_beta),
                velocities_[i]);
            NN_TRY_CHECK(r);
            // p -= lr * v（原地、单 dispatch）
            r = dsl::compute_into(engine_,
                dsl::leaf(params_[i]) + dsl::leaf(velocities_[i]) * dsl::rparam(-lr_),
                params_[i]);
            NN_TRY_CHECK(r);
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

    // Adam 核心更新（protected，AdamW 复用）
    // inv_bc1/inv_bc2 由 step() 提前计算（每步仅一次 pow），失败时不推进 t。
    //
    // 实现（三个 DSL 融合 kernel，零中间缓冲物化）：全部超参
    // （β1/β2/eps/lr/inv_bc1/inv_bc2）经 RParam（运行时标量）承载——值不进
    // expr_spec_key，同结构共享 fusion shader，引擎适应计算。不物化任何
    // m_hat/v_hat/sqrt_v/denom/ratio 中间张量：
    //     K1  m = β1*m + (1-β1)*g            （m_ owned，直接重赋值）
    //     K2  v = β2*v + (1-β2)*g²           （v_ owned，直接重赋值）
    //     K3  p -= lr * (inv_bc1*m) / (sqrt(inv_bc2*v)+eps)（p 为模型张量，目标传递原地更新）
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
        NN_TRY_CHECK(m_new);
        m_[i] = std::move(*m_new);

        // K2: v = β2*v + (1-β2)*g²（单 kernel 融合；状态精度 = p_.optimizer）
        auto v_new = dsl::compute(engine_,
            dsl::leaf(v_[i]) * dsl::rparam(beta2_) +
                dsl::leaf(g) * dsl::leaf(g) * dsl::rparam(one_minus_beta2),
            rows, cols, p_.optimizer);
        NN_TRY_CHECK(v_new);
        v_[i] = std::move(*v_new);

        // K3: p -= lr * (inv_bc1*m) / (sqrt(inv_bc2*v)+eps)（全链单 kernel 融合；
        //     依赖刚更新的 m_[i]/v_[i]，偏置修正系数 inv_bc1/inv_bc2 逐步变化
        //     由 RParam 承载，不进 key → 共享 shader）
        //     更新量精度 = **目标参数的存储精度**：通常就是 p_.param（省一次 cast），
        //     但当 param 槽不是"存储精度"时必须以张量自身为准 —— 例如 BitLinear 的
        //     T1_58（latent 恒 f32）：照搬 p_.param 会拿非存储精度去建张量
        //     （CPU 走 f32 兜底、GPU 在 check_precision_supported 处硬报错 =
        //      两端行为不一致）。见 docs/todo/21-quantized-weights.md §4.3.2。
        const Precision upd_prec = (precision_bytes(p_.param) != 0)
                                       ? p_.param
                                       : params_[i].get().precision();
        auto delta = dsl::compute(engine_,
              -dsl::rparam(lr_)
              * ((dsl::leaf(m_[i]) * dsl::rparam(inv_bc1))
                 / (dsl::sqrt(dsl::leaf(v_[i]) * dsl::rparam(inv_bc2))
                    + dsl::rparam(eps_))),
            rows, cols, upd_prec);
        NN_TRY_CHECK(delta);
        // p += delta（目标传递：原地、单 dispatch，不额外分配）
        auto r = dsl::compute_into(engine_,
            dsl::leaf(params_[i]) + dsl::leaf(*delta), params_[i]);
        NN_TRY_CHECK(r);
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
        NN_TRY(r, check_ready_());
        NN_TRY(r2, validate_sizes_());
        // 偏差修正：每步只计算一次（而非每参数重复 pow）
        const std::size_t t_next = t_ + 1;
        const auto [inv_bc1, inv_bc2] = bias_correction_(t_next);

        // IR-C（P2）：整个步进录制为一张图——每参数的 K1/K2 状态
        // 写穿 + K3 中间量 + K4 目标传递融合为**每张量一个多输出 kernel**（f32
        // 下 4 节点金刚石单 kernel 3 extras；混精度按门禁自动分裂），替代逐表达式
        // 数百个小 dispatch（P2 前实测 266 设备 kernel/步、host 提交 4.5ms/步）。
        // CPU begin/end 为 no-op（行为不变）；NN_EXPR_SCAN 下本段把复合 spec
        // 登记进闭合世界（scan_exprs 的优化器 dry-run 正是经此登记）。
        NN_TRY(b, engine_.begin_expr());
        ExprSegment seg(engine_);
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            NN_TRY(ur, adam_update_(i, inv_bc1, inv_bc2));
        }
        NN_TRY_CHECK(seg.end());
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
        NN_TRY(r, check_ready_());
        NN_TRY(r2, validate_sizes_());
        // 偏差修正：每步只计算一次（而非每参数重复 pow）
        const std::size_t t_next = t_ + 1;
        const auto [inv_bc1, inv_bc2] = bias_correction_(t_next);
        const Scalar decay_factor = Scalar{1} - lr_ * wd_;

        // IR-C（P2）：同 Adam——整个步进录制为一张图。⚠ decay 与 adam_update_
        // 是**同目标（p）的两次 compute_into、无数据依赖边**：执行序 = 节点序
        // （稳定拓扑序保持列表序），故 decay 必须写在 adam_update_ 之前
        // （先衰减后更新 = AdamW 语义）；两者不连通 → 各自独立 kernel，天然
        // 顺序执行，不会同 kernel 双写 p。
        NN_TRY(b, engine_.begin_expr());
        ExprSegment seg(engine_);
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            // 权重衰减解耦：p = (1 - lr*wd) * p（目标传递：原地、单 dispatch）
            if (wd_ != Scalar{0})
            {
                NN_TRY(dr, dsl::compute_into(engine_,
                    dsl::leaf(params_[i]) * dsl::rparam(decay_factor), params_[i]));
            }

            // Adam 更新
            NN_TRY(ur, adam_update_(i, inv_bc1, inv_bc2));
        }
        NN_TRY_CHECK(seg.end());
        t_ = t_next;  // 全部参数成功才推进步数（失败时不推进）
        return {};
    }
};

// ══════════════════════════════════════════════════════════════════════════
// AdamWQuant — 量化状态的 AdamW（训练显存优化，docs 21 §4.13 的 F5 第一步）
//
// **动机**：优化器状态（m/v 各 f32 = 8 B/权重）是训练显存的最大单项 ——
// 三值层训练账（latent 4 + grad 4 + m/v 8 + 打包权重 0.25 = 16.25 B/权重）里
// 占 49%。把它砍半，比继续压权重划算得多。
//
// **做法（有界尺度 + f16 payload）**：状态只存 f16，但**按行动态缩放**：
//   m16 = m_real / sm，v16 = v_real / sv（sm/sv 是 (rows,1) f32 尺度）
// 每步先推进尺度（上界估计，见下），再把新状态一趟融合写回 f16（写回本身
// 就是 RNE 舍入 → 不需要 DSL 的 round 算子、不需要新的引擎原语）。
//   · 状态显存：8 B → **4 B/权重**（+ 每行两个 f32 尺度，占 4/cols·2 B/权重，
//     cols=512 时 0.016 B —— 可忽略）；
//   · 精度：f16 的 11 bit **相对**精度（处处一致，缩放只为避免"小量级行掉进
//     非规格化区间"，即 research/f16_weight_decay 否决"直接把状态存 f16"的那
//     个失效模式）；比 bitsandbytes 的 int8 payload（8 bit 相对精度）更好。
//
// **上界估计（为什么不多付一遍遍历）**：m 是凸组合 ⇒
//     |m_new| ≤ β1·|m_old| + (1-β1)·|g| ⇒ sm_new = β1·sm + (1-β1)·row_max(|g|)·s
//     v_new   =  β2·v_old + (1-β2)·g²  ⇒ sv_new = β2·sv + (1-β2)·row_max(g²)·s
// （s = 1.02 安全系数）。尺度**无需先算出新状态**（那需要一个全尺寸 f32 临时量
// 或一个融合按块内核），只需 g 的行归约（每参数 2 次小 reduce）。
//
// **与 bitsandbytes 8-bit Adam 的差别（诚实记录）**：它是 int8 payload + 每块
// 紧凑 absmax（1 B/状态，需融合按块内核）；本版是 f16 payload + 有界尺度
// （2 B/状态，零新原语）。再往下的增量是"int8 payload + 打包"：需要 DSL 的
// `round` 算子（docs 21 §5 的 P3 前置）+ 打包容器（1 字节存储 = M7，或用 f16
// 容器位打包）。
//
// 超参与 AdamW 一致（β1/β2/eps/weight decay 解耦）；`--optimizer adamwquant`。
// ══════════════════════════════════════════════════════════════════════════
class AdamWQuant : public Optimizer
{
    Scalar lr_;
    Scalar beta1_;
    Scalar beta2_;
    Scalar eps_;
    Scalar wd_;
    std::size_t t_ = 0;
    std::vector<Tensor> m16_;   // f16 一阶矩（按 sm 归一，值域 ≈ [-1,1]）
    std::vector<Tensor> v16_;   // f16 二阶矩（按 sv 归一，值域 ≈ [0,1]）
    std::vector<Tensor> sm_;    // (rows,1) f32：|m| 的上界
    std::vector<Tensor> sv_;    // (rows,1) f32：v 的上界
    std::size_t state_bytes_ = 0;   // 诊断/实测：状态实际占用（含尺度）

    // 上界安全系数（上界估计的富余；代价 ≤1 bit 精度）与尺度下限（首步 g 全零时
    // 防 0 除）
    static constexpr Scalar kSafety = Scalar{1.02};
    static constexpr Scalar kScaleFloor = Scalar{1e-24};

    [[nodiscard]] Result<void> init_state_()
    {
        m16_.reserve(params_.size());
        v16_.reserve(params_.size());
        sm_.reserve(params_.size());
        sv_.reserve(params_.size());
        for (auto& p : params_)
        {
            const std::size_t rows = p.get().rows(), cols = p.get().cols();
            auto m = engine_.create_tensor(rows, cols, Precision::F16, InitSpec::zero());
            if (!m.valid()) NN_FAIL("AdamWQuant: 一阶矩状态分配失败");
            auto v = engine_.create_tensor(rows, cols, Precision::F16, InitSpec::zero());
            if (!v.valid()) NN_FAIL("AdamWQuant: 二阶矩状态分配失败");
            auto sm = engine_.create_tensor(rows, 1, Precision::F32, InitSpec::zero());
            if (!sm.valid()) NN_FAIL("AdamWQuant: 一阶矩尺度分配失败");
            auto sv = engine_.create_tensor(rows, 1, Precision::F32, InitSpec::zero());
            if (!sv.valid()) NN_FAIL("AdamWQuant: 二阶矩尺度分配失败");
            state_bytes_ += rows * cols * 2u * 2u + rows * 4u * 2u;
            m16_.push_back(std::move(m));
            v16_.push_back(std::move(v));
            sm_.push_back(std::move(sm));
            sv_.push_back(std::move(sv));
        }
        return {};
    }

    // 单个参数的 AdamW 更新（三 K 的结构与 Adam 同族：K1 一阶矩、K2 二阶矩、
    // K3 参数更新；此处每 K 都多一趟"尺度前进 + f16 归一写回"）
    [[nodiscard]] Result<void> update_one_(std::size_t i, Scalar inv_bc1, Scalar inv_bc2)
    {
        const Tensor& g = grads_[i];
        const std::size_t rows = g.rows(), cols = g.cols();
        const Scalar one_m_b1 = Scalar{1} - beta1_;
        const Scalar one_m_b2 = Scalar{1} - beta2_;

        // ── 尺度前进（上界；只依赖 g，不需要新状态的 f32 副本）──────────────
        NN_TRY(gabs, dsl::compute_reduce(engine_,
            dsl::row_reduce_max(dsl::abs(dsl::leaf(g))), rows, cols, Precision::F32));
        NN_TRY(gmax, dsl::compute_reduce(engine_,
            dsl::row_reduce_max(dsl::leaf(g) * dsl::leaf(g)), rows, cols, Precision::F32));
        auto sm_new = dsl::compute(engine_,
            dsl::max(dsl::leaf(sm_[i]) * dsl::rparam(beta1_)
                         + dsl::leaf(*gabs) * dsl::rparam(one_m_b1 * kSafety),
                     kScaleFloor),
            rows, 1, Precision::F32);
        NN_TRY_CHECK(sm_new);
        auto sv_new = dsl::compute(engine_,
            dsl::max(dsl::leaf(sv_[i]) * dsl::rparam(beta2_)
                         + dsl::leaf(*gmax) * dsl::rparam(one_m_b2 * kSafety),
                     kScaleFloor),
            rows, 1, Precision::F32);
        NN_TRY_CHECK(sv_new);

        // ── 状态写回（去量化 → 新状态 → ÷新尺度 → f16 RNE 写回，单 dispatch）──
        auto mw = dsl::compute_into(engine_,
            (dsl::leaf(m16_[i]) * dsl::row_broadcast(sm_[i]) * dsl::rparam(beta1_)
             + dsl::leaf(g) * dsl::rparam(one_m_b1))
                / dsl::row_broadcast(*sm_new),
            m16_[i]);
        NN_TRY_CHECK(mw);
        auto vw = dsl::compute_into(engine_,
            (dsl::leaf(v16_[i]) * dsl::row_broadcast(sv_[i]) * dsl::rparam(beta2_)
             + dsl::leaf(g) * dsl::leaf(g) * dsl::rparam(one_m_b2))
                / dsl::row_broadcast(*sv_new),
            v16_[i]);
        NN_TRY_CHECK(vw);

        // ── 尺度提交（写回用的是新尺度，故先换再用）────────────────────────
        sm_[i] = std::move(*sm_new);
        sv_[i] = std::move(*sv_new);

        // ── 参数更新（AdamW：weight decay 解耦到 p；偏置修正走 rparam）──────
        auto upd = dsl::compute_into(engine_,
            dsl::leaf(params_[i])
                - dsl::rparam(lr_)
                      * (dsl::rparam(inv_bc1)
                             * (dsl::leaf(m16_[i]) * dsl::row_broadcast(sm_[i]))
                         / (dsl::sqrt(dsl::rparam(inv_bc2)
                                      * (dsl::leaf(v16_[i]) * dsl::row_broadcast(sv_[i])))
                            + dsl::rparam(eps_))
                         + dsl::rparam(wd_) * dsl::leaf(params_[i])),
            params_[i]);
        NN_TRY_CHECK(upd);
        return {};
    }

public:
    AdamWQuant(ComputeEngine& engine,
               std::vector<TensorRef> params,
               std::vector<TensorRef> grads,
               Scalar lr,
               Scalar beta1 = 0.9,
               Scalar beta2 = 0.999,
               Scalar eps = 1e-8,
               Scalar weight_decay = 0.01,
               PrecisionProfile precision = PrecisionProfile{})
        : Optimizer(engine, std::move(params), std::move(grads), precision),
          lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(weight_decay)
    {
        auto r = init_state_();
        if (!r) record_init_error_(r.error());
    }

    void set_lr(Scalar lr) override { lr_ = lr; }

    [[nodiscard]] Result<void> step() override
    {
        NN_TRY(r, check_ready_());
        NN_TRY(r2, validate_sizes_());
        const Scalar n = static_cast<Scalar>(t_ + 1);
        const Scalar inv_bc1 = Scalar{1} / (Scalar{1} - std::pow(beta1_, n));
        const Scalar inv_bc2 = Scalar{1} / (Scalar{1} - std::pow(beta2_, n));
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            NN_TRY(u, update_one_(i, inv_bc1, inv_bc2));
        }
        ++t_;   // 全部参数成功才推进步数（与 Adam 同款）
        return {};
    }

    // ── 诊断 / 测试视图 ───────────────────────────────────────────────────
    // 状态实际字节数（用于与 f32 Adam 的状态账对拍：后者 = Σ 2·rows·cols·4）
    [[nodiscard]] std::size_t state_bytes() const noexcept { return state_bytes_; }
    [[nodiscard]] std::size_t steps() const noexcept { return t_; }
    [[nodiscard]] const std::vector<Tensor>& moment_state() const noexcept { return m16_; }
    [[nodiscard]] const std::vector<Tensor>& variance_state() const noexcept { return v16_; }
    [[nodiscard]] const std::vector<Tensor>& moment_scales() const noexcept { return sm_; }
    [[nodiscard]] const std::vector<Tensor>& variance_scales() const noexcept { return sv_; }
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
    NN_TRY_CHECK(row_sum_norm);
    // 二级归约（(rows,1) → (1,1)）用 dsl::compute_reduce
    auto total_norm_sq = dsl::compute_reduce(engine,
        dsl::col_reduce_sum(dsl::leaf(*row_sum_norm)),
        row_sum_norm->rows(), row_sum_norm->cols(), prec);
    NN_TRY_CHECK(total_norm_sq);
    // 从 (1,1) Tensor 提取标量值
    // 使用宿主桥下载后取值（一次 PCIe 下载，可接受；17 §3 D11）
    std::vector<Scalar> total_buf(total_norm_sq->rows() * total_norm_sq->cols());
    auto dl_r = detail::download_span(engine, *total_norm_sq, std::span(total_buf));
    NN_TRY_CHECK(dl_r);
    Scalar norm_sq_val = total_buf[0];
    Scalar inv_norm_scalar = Scalar{1} / std::sqrt(norm_sq_val + eps * eps);

    // X = G * inv_norm_scalar（归一化）：单表达式（GPU 上 1 个融合 kernel +
    // 1 次分配）
    auto X = dsl::compute(engine,
        dsl::leaf(G) * dsl::rparam(inv_norm_scalar), G.rows(), G.cols(), prec);
    NN_TRY_CHECK(X);
    // 选更小一侧构造母矩阵，避免显存爆炸（Muon 显存 > AdamW 的根因）：
    //   - 短宽/方阵（m ≤ n）：行正交化，母矩阵 A = X·X^T（m×m，m 为短边）
    //   - 高窄矩阵（m > n）：列正交化，母矩阵 G = X^T·X（n×n，n 为短边）
    // 故母矩阵恒为 min(m,n)² 而非 max(m,n)²：对高窄大参数（如 50257×1024 的
    // 词嵌入），沿长边构造 50257² 的 A/A²（≈10GB/个）必 OOM，沿短边只需
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
        for (std::size_t t = 0; t < steps; ++t)
        {
            // 纯 matmul 经 DSL 直写（scan 的 optimizer dry-run 自动登记结构）
            auto A = dsl::compute(engine,           // A = X·X^T
                dsl::matmul(*X, *X, false, true), m, m, prec);
            NN_TRY_CHECK(A);
            auto A_sq = dsl::compute(engine,         // A²
                dsl::matmul(*A, *A, false, false), m, m, prec);
            NN_TRY_CHECK(A_sq);
            // A = b·A + c·A²：一次原地目标传递（GPU 上 1 个 kernel）
            auto accA = dsl::compute_into(engine,
                dsl::leaf(*A) * dsl::rparam(b) + dsl::leaf(*A_sq) * dsl::rparam(c), *A);
            NN_TRY_CHECK(accA);
            auto BX = dsl::compute(engine,           // B·X
                dsl::matmul(*A, *X, false, false), m, n, prec);
            NN_TRY_CHECK(BX);
            // BX += a·X（原地目标传递：单 dispatch，不额外分配）
            auto accBX = dsl::compute_into(engine,
                dsl::leaf(*BX) + dsl::leaf(*X) * dsl::rparam(a), *BX);
            NN_TRY_CHECK(accBX);
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
            // 纯 matmul 经 DSL 直写（同上，scan optimizer dry-run 登记）
            auto Gr = dsl::compute(engine,           // G = X^T·X（n×n）
                dsl::matmul(*X, *X, true, false), n, n, prec);
            NN_TRY_CHECK(Gr);
            auto Gr_sq = dsl::compute(engine,        // G²
                dsl::matmul(*Gr, *Gr, false, false), n, n, prec);
            NN_TRY_CHECK(Gr_sq);
            // G = b·G + c·G²（同上：一次原地目标传递，单 kernel）
            auto accG = dsl::compute_into(engine,
                dsl::leaf(*Gr) * dsl::rparam(b) + dsl::leaf(*Gr_sq) * dsl::rparam(c), *Gr);
            NN_TRY_CHECK(accG);
            auto XM = dsl::compute(engine,           // X·G
                dsl::matmul(*X, *Gr, false, false), m, n, prec);
            NN_TRY_CHECK(XM);
            // XM += a·X（原地目标传递：单 dispatch）
            auto accXM = dsl::compute_into(engine,
                dsl::leaf(*XM) + dsl::leaf(*X) * dsl::rparam(a), *XM);
            NN_TRY_CHECK(accXM);
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
        NN_TRY(r, check_ready_());
        NN_TRY(r2, validate_sizes_());
        for (std::size_t i = 0; i < params_.size(); ++i)
        {
            const Tensor& g = grads_[i];

            // 1. SGD-Momentum: v = μ*v + g（两趟原语融合为单 dispatch）
            auto r = dsl::compute_into(engine_,
                dsl::leaf(velocities_[i]) * dsl::rparam(momentum_) + dsl::leaf(g),
                velocities_[i]);
            NN_TRY_CHECK(r);
            // 确定用于正交化的更新方向（Nesterov 时需临时缓冲，否则直接用 v）
            std::optional<Tensor> nesterov_buf;
            if (nesterov_)
            {
                // Nesterov: update = g + μ*v（单表达式：一次 dispatch + 一次分配）
                auto buf = dsl::compute(engine_,
                    dsl::leaf(g) + dsl::leaf(velocities_[i]) * dsl::rparam(momentum_),
                    g.rows(), g.cols(), p_.optimizer);
                NN_TRY_CHECK(buf);
                nesterov_buf = std::move(*buf);
            }
            const Tensor& update = nesterov_buf ? *nesterov_buf : velocities_[i];

            // 2. Newton-Schulz 正交化（仅对 ≥2D 参数，即 rows > 1 且 cols > 1）
            if (params_[i].get().rows() > 1 && params_[i].get().cols() > 1)
            {
                auto ortho_update = newton_schulz_orthogonalize(
                    engine_, update, ns_steps_, ns_eps_, p_.optimizer);
                NN_TRY_CHECK(ortho_update);
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
                NN_TRY_CHECK(r);
            }
            else
            {
                // 非 2D 参数（bias 等）：标准 SGD 更新（目标传递：原地、单 dispatch）
                r = dsl::compute_into(engine_,
                    dsl::leaf(params_[i]) + dsl::leaf(update) * dsl::rparam(-lr_),
                    params_[i]);
                NN_TRY_CHECK(r);
            }
        }
        return {};
    }
};

// ── 优化器元数据表（RV-04：新增一个优化器不再要改 5~6 处）─────────────────
// **名单的唯一真源**。此前同一份名单散落六处：本文件工厂、text_train 的硬编码
// 白名单、mnist_train 帮助文本、gui.py 的 OPTIMIZER_OPTIONS、scan_exprs 的
// dry-run 列表、mem_probe 的状态缓冲分支（AGENTS §3 把"别忘了改这几处"写成提醒
// ——提醒本身就说明设计有问题）。现在全部查表：
//   · CLI 校验 / 帮助文本  → find_optimizer / optimizer_names
//   · scan_exprs dry-run   → optimizer_table() 逐项建优化器
//   · mem_probe 状态预算   → state_bufs / half_state
//   · GUI 选项             → CLI --help（由本表生成）+ bench/gui_cli_audit.py 断言
// 新增优化器只需：① 在本表加一行；② 在下方 create_optimizer 加一个分支
//（构造器必须手写，无法表驱动）。
// 字段：
//   name       CLI/GUI 的稳定标识（与 create_optimizer 的 name 严格同串）
//   display    人读名（帮助/审计/未来 GUI 显示用）
//   beta       是否消费 beta1/beta2（sgd 家族与 muon 忽略这两个参数）
//   state_bufs 每权重**全尺寸**状态缓冲数（显存账：0 = 无状态）
//   half_state 状态是否减半存储（AdamWQuant：f16 payload + 逐行有界尺度）
struct OptimizerMeta
{
    std::string_view name;
    std::string_view display;
    bool             beta;
    std::uint8_t     state_bufs;
    bool             half_state;
};

inline constexpr OptimizerMeta kOptimizerTable[] = {
    {"sgd",          "SGD",                false, 0, false},
    {"sgd_momentum", "SGD + Momentum",     false, 1, false},
    {"adam",         "Adam",               true,  2, false},
    {"adamw",        "AdamW",              true,  2, false},
    {"adamwquant",   "AdamW（量化状态）",   true,  2, true },
    {"muon",         "Muon",               false, 1, false},
};

[[nodiscard]] constexpr std::span<const OptimizerMeta> optimizer_table() noexcept
{ return kOptimizerTable; }

// 按 CLI 名查元数据；未知名 → nullptr（调用方负责报错，不静默降级）
[[nodiscard]] inline const OptimizerMeta *find_optimizer(std::string_view name) noexcept
{
    for (const auto &m : kOptimizerTable)
        if (m.name == name) return &m;
    return nullptr;
}

// 名单文本（帮助文本 / 报错提示 / 审计脚本解析共用同一口径）：
// sep = '/' → "sgd/sgd_momentum/..."；sep = ", " → "sgd, sgd_momentum, ..."
[[nodiscard]] inline std::string optimizer_names(std::string_view sep)
{
    std::string s;
    for (const auto &m : kOptimizerTable)
    {
        if (!s.empty()) s += sep;
        s += m.name;
    }
    return s;
}

// ── 优化器工厂函数 ────────────────────────────────────────────────────────
// 根据名称创建对应优化器（名单与校验口径见上方 kOptimizerTable）。
// 未知名称返回 nullptr，由调用方处理。
// precision：模型级精度配置（§9.1）——状态张量按 p.optimizer 创建，参数更新
// 走 in-place（存储精度不可变）。默认全 F32。
// beta1/beta2：仅 adam/adamw 使用（sgd 家族与 muon 忽略）。默认 0.9/0.999；
// LLM 预训练惯例常取 beta2=0.95（如 GPT-3 / LLaMA 配方）。
[[nodiscard]] inline std::unique_ptr<Optimizer> create_optimizer(
    std::string_view name,
    ComputeEngine& engine,
    std::vector<TensorRef> params,
    std::vector<TensorRef> grads,
    Scalar lr,
    Scalar weight_decay = 0,
    PrecisionProfile precision = PrecisionProfile{},
    Scalar beta1 = 0.9,
    Scalar beta2 = 0.999)
{
    if (name == "sgd")
        return std::make_unique<SGD>(engine, std::move(params), std::move(grads), lr,
                                     precision);
    if (name == "sgd_momentum")
        return std::make_unique<SGDWithMomentum>(engine, std::move(params),
                                                 std::move(grads), lr, 0.9, precision);
    if (name == "adam")
        return std::make_unique<Adam>(engine, std::move(params), std::move(grads), lr,
                                      beta1, beta2, 1e-8, precision);
    if (name == "adamw")
        return std::make_unique<AdamW>(engine, std::move(params), std::move(grads), lr,
                                       beta1, beta2, 1e-8, weight_decay, precision);
    if (name == "adamwquant")
        return std::make_unique<AdamWQuant>(engine, std::move(params), std::move(grads),
                                            lr, beta1, beta2, 1e-8, weight_decay,
                                            precision);
    if (name == "muon")
        return std::make_unique<Muon>(engine, std::move(params), std::move(grads), lr,
                                      0.95f, true, 5, 1e-7f, precision);
    return nullptr;  // 未知名称，调用方应处理
}

} // namespace nn
