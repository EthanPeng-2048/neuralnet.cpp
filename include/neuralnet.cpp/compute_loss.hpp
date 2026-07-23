#ifndef NN_COMPUTE_LOSS_HPP
#define NN_COMPUTE_LOSS_HPP

// ── compute_loss.hpp — 引擎化损失函数 ──────────────────────────────────────
//
// 架构铁律：
//   1. Loss 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。
//   2. 算法只在 Loss（通过组合 engine 原语表达），绝不在 Engine/Shader 中。
//   3. Softmax/CrossEntropy 算法由本文件通过 col_reduce_max + exp + col_reduce_sum
//      + broadcast_col + elementwise 组合表达，Engine 不知道 "softmax" 是什么。
//
// 算法表达示例：
//   MSELoss forward:  grad = (2/N) * (pred - target);  loss = (1/N) * Σ diff²
//   CrossEntropy forward (with softmax):
//     col_max   = col_reduce_max(logits)             // 数值稳定
//     shifted   = logits - col_max                   (broadcast_col Sub)
//     exp_shift = exp(shifted)                       (unary Exp)
//     col_sum   = col_reduce_sum(exp_shift)
//     softmax   = exp_shift / col_sum                (broadcast_col Div)
//     grad      = softmax - target                   (elementwise Sub)
//     log_sm    = shifted - log(col_sum)             (broadcast_col Sub)
//     loss      = -(1/batch) * Σ target * log_sm
//   CrossEntropy backward: grad = softmax - target_onehot
// ─────────────────────────────────────────────────────────────────────────

#include "compute_engine.hpp"
#include "compute_layer.hpp"  // clone_tensor
#include "tensor.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Loss — 引擎化损失函数基类
// ══════════════════════════════════════════════════════════════════════════
class Loss
{
public:
    virtual ~Loss() = default;

    // forward 计算损失标量，并缓存 backward 所需中间结果
    [[nodiscard]] virtual Result<Scalar> forward(
        ComputeEngine& engine, const Tensor& pred, const Tensor& target) = 0;

    // backward 返回对 pred 的梯度
    [[nodiscard]] virtual Result<Tensor> backward() = 0;
};

// ══════════════════════════════════════════════════════════════════════════
// MSELoss — 均方误差损失
//
// 算法（只在此处，不在 Engine/Shader）：
//   diff = pred - target
//   grad = (2/N) * diff
//   loss = (1/N) * Σ diff²
// ══════════════════════════════════════════════════════════════════════════
class MSELoss final : public Loss
{
private:
    Tensor grad_input_;

public:
    MSELoss() = default;

    [[nodiscard]] Result<Scalar> forward(
        ComputeEngine& engine, const Tensor& pred, const Tensor& target) override
    {
        if (pred.rows() != target.rows() || pred.cols() != target.cols())
            return std::unexpected(Error{"mse loss: shape mismatch"});
        if (pred.size() == 0)
            return std::unexpected(Error{"mse loss: empty input"});

        const Scalar total = static_cast<Scalar>(pred.size());
        const Scalar scale = Scalar{2} / total;

        // diff = pred - target
        auto diff = engine.elementwise_binary(BinaryOp::Sub, pred, target);
        if (!diff) return std::unexpected(diff.error());

        // grad = diff * (2/N) — 深拷贝后 scale
        grad_input_ = *diff;
        auto r = engine.scale_inplace(grad_input_, scale);
        if (!r) return std::unexpected(r.error());

        // diff_sq = diff * diff
        auto diff_sq = engine.elementwise_binary(BinaryOp::Mul, *diff, *diff);
        if (!diff_sq) return std::unexpected(diff_sq.error());

        // 全局求和：先按列归约 (1, cols)，再按行归约 (1, 1)
        auto col_sum = engine.col_reduce_sum(*diff_sq);
        if (!col_sum) return std::unexpected(col_sum.error());
        auto total_t = engine.row_reduce_sum(*col_sum);
        if (!total_t) return std::unexpected(total_t.error());

        // 下载标量
        auto m = engine.to_matrix(*total_t);
        if (!m) return std::unexpected(m.error());

        return m->at_unchecked(0, 0) / total;
    }

    [[nodiscard]] Result<Tensor> backward() override
    {
        return grad_input_;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// CrossEntropyLoss — 带 softmax 的交叉熵损失
//
// 算法（只在此处，不在 Engine/Shader）：
//   对每列（batch 样本）独立做 softmax + cross entropy：
//     col_max   = max_c logits[c][i]                  (col_reduce_max)
//     shifted   = logits - col_max                    (broadcast_col Sub)
//     exp_shift = exp(shifted)                        (unary Exp)
//     col_sum   = Σ_c exp_shift[c][i]                 (col_reduce_sum)
//     softmax   = exp_shift / col_sum                 (broadcast_col Div)
//     grad      = softmax - target_onehot             (elementwise Sub)
//     log_sm    = shifted - log(col_sum)              (broadcast_col Sub)
//     loss      = -(1/batch) * Σ target * log_sm
// ══════════════════════════════════════════════════════════════════════════
class CrossEntropyLoss final : public Loss
{
private:
    Tensor grad_input_;

public:
    CrossEntropyLoss() = default;

    [[nodiscard]] Result<Scalar> forward(
        ComputeEngine& engine, const Tensor& logits, const Tensor& target) override
    {
        const std::size_t classes = logits.rows();
        const std::size_t batch   = logits.cols();

        if (target.rows() != classes || target.cols() != batch)
            return std::unexpected(Error{"cross_entropy loss: shape mismatch"});
        if (classes == 0 || batch == 0)
            return std::unexpected(Error{"cross_entropy loss: empty input"});

        // 1. col_max = max per column (数值稳定)
        auto col_max = engine.col_reduce_max(logits);
        if (!col_max) return std::unexpected(col_max.error());

        // 2. shifted = logits - col_max (broadcast_col Sub) — 深拷贝 logits
        auto shifted = clone_tensor(engine, logits);
        if (!shifted) return std::unexpected(shifted.error());
        auto r = engine.broadcast_col_inplace(*shifted, *col_max, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // 3. exp_shift = exp(shifted)
        auto exp_shift = engine.elementwise_unary(UnaryOp::Exp, *shifted);
        if (!exp_shift) return std::unexpected(exp_shift.error());

        // 4. col_sum = Σ_c exp_shift[c][i]
        auto col_sum = engine.col_reduce_sum(*exp_shift);
        if (!col_sum) return std::unexpected(col_sum.error());

        // 5. softmax = exp_shift / col_sum (broadcast_col Div) — 深拷贝 exp_shift
        auto softmax = clone_tensor(engine, *exp_shift);
        if (!softmax) return std::unexpected(softmax.error());
        r = engine.broadcast_col_inplace(*softmax, *col_sum, BinaryOp::Div);
        if (!r) return std::unexpected(r.error());

        // 6. grad = softmax - target (elementwise Sub)
        auto grad = engine.elementwise_binary(BinaryOp::Sub, *softmax, target);
        if (!grad) return std::unexpected(grad.error());
        grad_input_ = *grad;

        // 7. log_softmax = shifted - log(col_sum)
        auto log_col_sum = engine.elementwise_unary(UnaryOp::Log, *col_sum);
        if (!log_col_sum) return std::unexpected(log_col_sum.error());

        auto log_softmax = clone_tensor(engine, *shifted);
        if (!log_softmax) return std::unexpected(log_softmax.error());
        r = engine.broadcast_col_inplace(*log_softmax, *log_col_sum, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        // 8. target_dot_log = target * log_softmax
        auto target_dot_log = engine.elementwise_binary(BinaryOp::Mul, target, *log_softmax);
        if (!target_dot_log) return std::unexpected(target_dot_log.error());

        // 9. Σ target * log_softmax (先列归约再行归约 → (1,1))
        auto col_s = engine.col_reduce_sum(*target_dot_log);
        if (!col_s) return std::unexpected(col_s.error());
        auto total_t = engine.row_reduce_sum(*col_s);
        if (!total_t) return std::unexpected(total_t.error());

        // 10. loss = -total / batch — 下载标量
        auto m = engine.to_matrix(*total_t);
        if (!m) return std::unexpected(m.error());

        return -m->at_unchecked(0, 0) / static_cast<Scalar>(batch);
    }

    [[nodiscard]] Result<Tensor> backward() override
    {
        return grad_input_;
    }
};

} // namespace nn

#endif // NN_COMPUTE_LOSS_HPP
