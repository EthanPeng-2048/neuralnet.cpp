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
#include "compute_tensor.hpp"
#include "expr_dsl.hpp"

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
        auto diff = dsl::compute(engine,
            dsl::leaf(pred) - dsl::leaf(target),
            pred.rows(), pred.cols());
        if (!diff) return std::unexpected(diff.error());

        // grad = diff * (2/N) — 深拷贝后 scale
        grad_input_ = *diff;
        auto r = engine.scale_inplace(grad_input_, scale);
        if (!r) return std::unexpected(r.error());

        // diff_sq = diff * diff
        auto diff_sq = dsl::compute(engine,
            dsl::leaf(*diff) * dsl::leaf(*diff),
            diff->rows(), diff->cols());
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

    // ── 按列 softmax 的稠密结果：既给 grad（softmax），也给数值稳定的
    //    log_softmax（= shifted - log(col_sum)，避免 0*log(0)=NaN，见 M3）。
    struct DenseSoftmax
    {
        Tensor softmax;     // (classes, batch)
        Tensor log_softmax; // (classes, batch) = shifted - log(col_sum)
    };

    // ── 按列 softmax（稠密 forward 使用）──────────────────────────────
    // col_max → col_sum = col_softmax_denom（M5 融合，不物化 exp 中间张量）
    //         → softmax = exp(logits - col_max) / col_sum
    // 同时计算数值稳定的 log_softmax = (logits - col_max) - log(col_sum)：
    //   直接 log(softmax) 在极负 logits/大词表下 softmax→0 → log→-inf，
    //   与 target=0 相乘得 0*(-inf)=NaN；稳定形式中 shifted 与 log(col_sum)
    //   均有限，可避免该 NaN（稠密/软标签路径，稀疏 kernel 已用稳定形式）。
    [[nodiscard]] static Result<DenseSoftmax> softmax_cols_(
        ComputeEngine& engine, const Tensor& logits)
    {
        auto col_max = engine.col_reduce_max(logits);
        if (!col_max) return std::unexpected(col_max.error());

        // S7：denom = Σ_r exp(logits[r][c] - col_max[c]) 用 IR 表达式
        // （列归约 + ColBroadcast 视图 + exp，单 kernel，不物化 exp 张量）
        auto col_sum = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))),
            logits.rows(), logits.cols());
        if (!col_sum) return std::unexpected(col_sum.error());

        auto shifted = clone_tensor(engine, logits);
        if (!shifted) return std::unexpected(shifted.error());
        auto r = engine.broadcast_col_inplace(*shifted, *col_max, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());

        auto exp_shift = engine.elementwise_unary(UnaryOp::Exp, *shifted);
        if (!exp_shift) return std::unexpected(exp_shift.error());

        // softmax 就地于 exp_shift 上，省一次 (classes, batch) 分配
        r = engine.broadcast_col_inplace(*exp_shift, *col_sum, BinaryOp::Div);
        if (!r) return std::unexpected(r.error());

        // 稳定 log_softmax = shifted - log(col_sum)
        // col_sum ≥ 1（因 max 元素 shifted=0 → exp=1），故 log(col_sum) 有限
        auto log_col_sum = dsl::compute(engine,
            dsl::log(dsl::leaf(*col_sum)),
            col_sum->rows(), col_sum->cols());
        if (!log_col_sum) return std::unexpected(log_col_sum.error());

        // log_softmax = shifted - log(col_sum)：这是**列广播**（每个元素减去
        // 对应列的 log(col_sum)），不能用 elementwise_binary —— log_col_sum
        // 形状为 (1, batch) 而 shifted 为 (classes, batch)，两引擎均要求形状
        // 完全一致会报 shape mismatch。此处 shifted 已不再使用，可就地广播。
        r = engine.broadcast_col_inplace(*shifted, *log_col_sum, BinaryOp::Sub);
        if (!r) return std::unexpected(r.error());
        auto log_softmax = std::move(shifted);

        return DenseSoftmax{/*softmax=*/std::move(*exp_shift),
                            /*log_softmax=*/std::move(*log_softmax)};
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

        // 1. softmax / log_softmax = softmax_cols(logits)（稳定形式，见 M3）
        auto sm = softmax_cols_(engine, logits);
        if (!sm) return std::unexpected(sm.error());

        // 2. grad = (softmax - target) / batch (elementwise Sub + scale)
        //    loss = -(1/batch) * Σ target * log_softmax，故
        //    d(loss)/d(logits) = (softmax - one_hot) / batch。
        //    缺少 1/batch 缩放会使 SGD/动量、梯度裁剪与 PyTorch 不一致
        //    （Adam 的二阶矩会抵消常数缩放，但其他优化器不会）。
        auto grad = dsl::compute(engine,
            dsl::leaf(sm->softmax) - dsl::leaf(target),
            sm->softmax.rows(), sm->softmax.cols());
        if (!grad) return std::unexpected(grad.error());
        auto rg = engine.scale_inplace(*grad, Scalar{1} / static_cast<Scalar>(batch));
        if (!rg) return std::unexpected(rg.error());
        grad_input_ = *grad;

        // 3. log_softmax 已在 softmax_cols_ 内以数值稳定形式算出
        //    （= shifted - log(col_sum)，避免 0*log(0)=NaN，见 M3）

        // 4. target_dot_log = target * log_softmax
        auto target_dot_log = dsl::compute(engine,
            dsl::leaf(target) * dsl::leaf(sm->log_softmax),
            target.rows(), target.cols());
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
    // M5 融合路径（唯一路径，不做旧路径回退）：
    //   1. 上传 labels / loss_mask 为 (1, total) 浮点张量（小传输）
    //   2. col_softmax_sparse_forward 单 kernel：列内 max + denom + 稠密梯度
    //      + 标签位置 log_softmax（loss_vec）——不物化 (classes, total) 全 softmax
    //   3. loss = -(1/num_valid)·Σ loss_vec（下载标量）
    // 融合原语不可用（如 CUDA 未对齐返回"未实现"、vocab_size>2^24 浮点标签
    // 不可精确表示）时**直接报错**，绝不静默回退旧路径（保持"硬报错、不降级"）。
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

        // ── M5 融合路径（不物化全 softmax；失败直接透传错误，不回退） ──
        return fused_forward_sparse_(engine, logits, labels, loss_mask, vocab_size);
    }

    // ── M5→S7 融合路径实现（IR 组合：col_max 原语 + denom/loss_vec/grad 表达式）──
    // 不物化 (classes, total) 全 softmax；与旧 col_softmax_sparse_forward 语义一致：
    //   col_max  = 列内 max（原语）
    //   denom    = col_sum(exp(logits - cb(col_max)))          （IR 表达式）
    //   loss_vec = (rg(logits) - cb(col_max) - log(denom)) * cb(mask)  （IR，(1,total)）
    //   grad     = (exp/logits 链 - select(Row==cb(labels),1,0)) * cb(mask) * inv（IR）
    [[nodiscard]] Result<Scalar> fused_forward_sparse_(
        ComputeEngine& engine, const Tensor& logits,
        std::span<const std::size_t> labels,
        std::span<const Scalar> loss_mask,
        std::size_t vocab_size)
    {
        const std::size_t classes = logits.rows();
        const std::size_t total = logits.cols();

        // 1. 上传 labels（(1, total) 浮点打包；vocab_size ≤ 2^24 时可精确表示）。
        //    越界 label 修正为 0（GPU RowGather 无越界守卫，靠 mask 置零无效）
        Matrix labels_m(1, total);
        {
            auto sp = labels_m.span();
            for (std::size_t i = 0; i < total; ++i)
                sp[i] = static_cast<Scalar>(
                    (labels[i] < vocab_size) ? labels[i] : 0);
        }
        auto labels_t = engine.from_matrix(labels_m);
        if (!labels_t) return std::unexpected(labels_t.error());

        // 2. 有效 mask（总是构造，(1,total) 0/1；含 mask 缺失与 label 越界修正）：
        //    valid = (loss_mask 空 || mask[i]>=0.5) && labels[i] < vocab_size
        Matrix mask_m(1, total);
        {
            auto sp = mask_m.span();
            for (std::size_t i = 0; i < total; ++i)
            {
                const bool masked = !loss_mask.empty() && loss_mask[i] < Scalar{0.5};
                sp[i] = (!masked && labels[i] < vocab_size) ? Scalar{1} : Scalar{0};
            }
        }
        auto mask_t = engine.from_matrix(mask_m);
        if (!mask_t) return std::unexpected(mask_t.error());

        // 3. num_valid（与 mask 判定一致）
        std::size_t num_valid = 0;
        for (std::size_t i = 0; i < total; ++i)
            if (mask_m.span()[i] >= Scalar{0.5}) ++num_valid;
        const Scalar inv_num_valid = (num_valid > 0)
            ? Scalar{1} / static_cast<Scalar>(num_valid) : Scalar{0};

        // 4. col_max → denom（IR）→ loss_vec / grad（IR）
        auto col_max = engine.col_reduce_max(logits);
        if (!col_max) return std::unexpected(col_max.error());
        auto denom = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))),
            classes, total);
        if (!denom) return std::unexpected(denom.error());
        // loss_vec[c] = (logits[label[c]][c] - col_max[c] - log(denom[c])) * mask[c]
        auto loss_vec = dsl::compute(engine,
            (dsl::row_gather(logits, *labels_t) - dsl::col_broadcast(*col_max)
             - dsl::log(dsl::leaf(*denom))) * dsl::col_broadcast(*mask_t),
            1, total);
        if (!loss_vec) return std::unexpected(loss_vec.error());
        // grad[r][c] = (exp(logits-col_max)/denom - [r==label[c]]) * mask[c]
        auto grad = dsl::compute(engine,
            (dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))
                / dsl::col_broadcast(*denom)
             - dsl::select(dsl::row() == dsl::col_broadcast(*labels_t),
                           Scalar{1}, Scalar{0}))
            * dsl::col_broadcast(*mask_t),
            classes, total);
        if (!grad) return std::unexpected(grad.error());
        // inv_num_valid 是运行时值（依赖 batch 内容），不进表达式（进 key 会
        // 破坏闭合世界匹配），后置 scale_inplace（语义等价）
        { auto gs = engine.scale_inplace(*grad, inv_num_valid); if (!gs) return std::unexpected(gs.error()); }
        grad_input_ = std::move(*grad);

        // 5. loss = -(1/num_valid)·Σ loss_vec（无效列已乘 0）
        auto total_t = engine.row_reduce_sum(*loss_vec);   // (1, total) → (1, 1)
        if (!total_t) return std::unexpected(total_t.error());
        auto m = engine.to_matrix(*total_t);
        if (!m) return std::unexpected(m.error());
        return (num_valid > 0)
            ? -m->at_unchecked(0, 0) / static_cast<Scalar>(num_valid)
            : Scalar{0};
    }
};

} // namespace nn

#endif // NN_COMPUTE_LOSS_HPP
