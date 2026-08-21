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

    // ── 按列 softmax（dense/sparse forward 共用）──────────────────────
    // col_max → shifted = logits - col_max → exp → col_sum → softmax
    // 返回 (classes, batch) 的 softmax 张量
    [[nodiscard]] static Result<Tensor> softmax_cols_(
        ComputeEngine& engine, const Tensor& logits)
    {
        auto col_max = engine.col_reduce_max(logits);
        if (!col_max) return std::unexpected(col_max.error());

        auto shifted = clone_tensor(engine, logits);
        if (!shifted) return std::unexpected(shifted.error());
        auto r = engine.broadcast_col_inplace(*shifted, *col_max, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        auto exp_shift = engine.elementwise_unary(UnaryOp::Exp, *shifted);
        if (!exp_shift) return std::unexpected(exp_shift.error());

        auto col_sum = engine.col_reduce_sum(*exp_shift);
        if (!col_sum) return std::unexpected(col_sum.error());

        // softmax 就地于 exp_shift 上，省一次 (classes, batch) 分配
        r = engine.broadcast_col_inplace(*exp_shift, *col_sum, BinaryOp::Div);
        if (!r) return std::unexpected(r.error());
        return exp_shift;
    }

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

        // 1. softmax = softmax_cols(logits)（与 sparse 路径共用）
        auto softmax = softmax_cols_(engine, logits);
        if (!softmax) return std::unexpected(softmax.error());

        // 2. grad = (softmax - target) / batch (elementwise Sub + scale)
        //    loss = -(1/batch) * Σ target * log_softmax，故
        //    d(loss)/d(logits) = (softmax - one_hot) / batch。
        //    缺少 1/batch 缩放会使 SGD/动量、梯度裁剪与 PyTorch 不一致
        //    （Adam 的二阶矩会抵消常数缩放，但其他优化器不会）。
        auto grad = engine.elementwise_binary(BinaryOp::Sub, *softmax, target);
        if (!grad) return std::unexpected(grad.error());
        auto rg = engine.scale_inplace(*grad, Scalar{1} / static_cast<Scalar>(batch));
        if (!rg) return std::unexpected(rg.error());
        grad_input_ = *grad;

        // 3. log_softmax = log(softmax)
        //    数学上等价于 shifted - log(col_sum)，少一次 clone + broadcast
        auto log_softmax = engine.elementwise_unary(UnaryOp::Log, *softmax);
        if (!log_softmax) return std::unexpected(log_softmax.error());

        // 4. target_dot_log = target * log_softmax
        auto target_dot_log = engine.elementwise_binary(BinaryOp::Mul, target, *log_softmax);
        if (!target_dot_log) return std::unexpected(target_dot_log.error());

        // 5. Σ target * log_softmax (先列归约再行归约 → (1,1))
        auto col_s = engine.col_reduce_sum(*target_dot_log);
        if (!col_s) return std::unexpected(col_s.error());
        auto total_t = engine.row_reduce_sum(*col_s);
        if (!total_t) return std::unexpected(total_t.error());

        // 6. loss = -total / batch — 下载标量
        auto m = engine.to_matrix(*total_t);
        if (!m) return std::unexpected(m.error());

        return -m->at_unchecked(0, 0) / static_cast<Scalar>(batch);
    }

    [[nodiscard]] Result<Tensor> backward() override
    {
        return grad_input_;
    }

    // ── 稀疏标签版 forward：接受整数标签而非 one-hot 矩阵 ──────────
    //
    // 解决大词表（vocab_size≈25k）+ 大 batch 时 one-hot 矩阵
    // (vocab_size, total_tokens) 爆显存的问题。
    //
    // 算法：
    //   1. 在 GPU 上计算 softmax（与 dense 版相同）
    //   2. 下载 softmax 到 CPU（唯一的大数据传输）
    //   3. 在 CPU 上用整数标签计算 loss 和 gradient
    //      loss = -(1/N) * Σ log(softmax[labels[i], i])
    //      grad = softmax;  grad[labels[i]][i] -= 1.0（对有效位置）
    //   4. 将 gradient 上传回 GPU
    //
    // 参数：
    //   labels     — 平坦标签数组，大小 = logits.cols()，值域 [0, vocab_size)
    //   loss_mask  — 可选，平坦 mask 数组，>0.5 表示参与 loss，否则清零梯度
    //   vocab_size — 词表大小，用于越界检查
    [[nodiscard]] Result<Scalar> forward_sparse(
        ComputeEngine& engine, const Tensor& logits,
        std::span<const std::size_t> labels,
        std::span<const Scalar> loss_mask = {},
        std::size_t vocab_size = 0)
    {
        const std::size_t classes = logits.rows();
        const std::size_t total   = logits.cols();

        if (vocab_size == 0) vocab_size = classes;
        if (labels.size() != total)
            return std::unexpected(Error{"sparse CE: labels size mismatch"});
        if (!loss_mask.empty() && loss_mask.size() != total)
            return std::unexpected(Error{"sparse CE: mask size mismatch"});
        if (classes == 0 || total == 0)
            return std::unexpected(Error{"sparse CE: empty input"});

        // ── 1. GPU 上计算 softmax（与 dense forward 共用 softmax_cols_） ──
        auto softmax_t = softmax_cols_(engine, logits);
        if (!softmax_t) return std::unexpected(softmax_t.error());

        // ── 2. 下载 softmax 到 CPU ────────────────────────────────
        auto softmax_cpu = engine.to_matrix(*softmax_t);
        if (!softmax_cpu) return std::unexpected(softmax_cpu.error());

        // softmax_t 现在可以释放
        softmax_t = Tensor();

        // ── 3. CPU 上计算 loss 和 gradient ────────────────────────
        const std::size_t rows = softmax_cpu->rows();
        const std::size_t cols = softmax_cpu->cols();

        // gradient 拷贝 softmax（后面只修改稀疏位置）
        Matrix grad_cpu(*softmax_cpu);

        Scalar loss_sum = 0.0;
        std::size_t num_valid = 0;
        const bool use_mask = !loss_mask.empty();

        for (std::size_t i = 0; i < cols; ++i)
        {
            // mask 检查
            if (use_mask && loss_mask[i] < Scalar{0.5})
            {
                // 被 mask 的位置：gradient 整列清零
                for (std::size_t c = 0; c < rows; ++c)
                    grad_cpu.set_value_unchecked(c, i, 0.0);
                continue;
            }

            const std::size_t lbl = labels[i];
            if (lbl >= vocab_size)
            {
                // 越界标签：跳过 loss，gradient 整列清零
                for (std::size_t c = 0; c < rows; ++c)
                    grad_cpu.set_value_unchecked(c, i, 0.0);
                continue;
            }

            // loss += log(softmax[lbl, i])
            const Scalar sm_val = softmax_cpu->at_unchecked(lbl, i);
            loss_sum += std::log(std::max(sm_val, Scalar{1e-20}));
            ++num_valid;

            // gradient: softmax[lbl, i] -= 1.0（softmax - one_hot 的稀疏等价）
            grad_cpu.set_value_unchecked(lbl, i, sm_val - Scalar{1});
        }

        // 除以有效 token 数（而非 total），避免 padding 拉低 loss 报告值
        const Scalar loss = (num_valid > 0)
            ? -loss_sum / static_cast<Scalar>(num_valid)
            : Scalar{0};

        // ── 3.5 梯度归一化：与 PyTorch 一致，grad = (softmax - one_hot) / num_valid ──
        // 之前缺少 1/num_valid 缩放，导致梯度被整体放大 num_valid 倍。
        // 由于 loss = -Σ log_softmax / num_valid，故
        //   d(loss)/d(logits) = (softmax - one_hot) / num_valid（有效位置），
        //   mask 位置/越界标签位置梯度为 0（缩放 0 仍为 0，安全）。
        // 对 Adam 而言，梯度常数缩放会被二阶矩归一化抵消（曲线不变），
        // 但对 SGD/动量、梯度裁剪 --max-norm 以及 --grad-log 统计，
        // 缺少该缩放会导致与 PyTorch 行为不一致。
        if (num_valid > 0)
        {
            const Scalar inv_num_valid = Scalar{1} / static_cast<Scalar>(num_valid);
            auto grad_span = grad_cpu.span();
            for (auto& val : grad_span) val *= inv_num_valid;
        }

        // ── 4. 上传 gradient 到 GPU ──────────────────────────────
        auto grad_tensor_r = engine.from_matrix(grad_cpu);
        if (!grad_tensor_r) return std::unexpected(grad_tensor_r.error());

        grad_input_ = std::move(*grad_tensor_r);
        return loss;
    }
};

} // namespace nn

#endif // NN_COMPUTE_LOSS_HPP
