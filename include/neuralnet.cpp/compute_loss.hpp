#pragma once

// ── compute_loss.hpp — 引擎化损失函数 ──────────────────────────────────────
//
// 架构铁律：
//   1. Loss 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配
//      CPU/GPU 设备。
//   2. 算法只在 Loss（通过组合 engine 原语表达），绝不在 Engine/Shader 中。
//   3. Softmax/CrossEntropy 算法由本文件用 DSL 表达式（dsl::compute /
//      dsl::compute_reduce）+ 归约/IO 引擎原语组合表达，Engine 不知道
//      "softmax" 是什么。
//
// 算法表达示例（均以 DSL 表达式书写，逐元素链与相邻归约融合为单 kernel）：
//   MSELoss forward:  diff = pred-target; grad = diff*(2/N);
//                     Σdiff² = compute_reduce(col_reduce_sum(diff*diff))
//   CrossEntropy forward (with softmax):
//     col_max   = col_reduce_max(logits)             // 数值稳定（归约原语）
//     col_sum   = compute_reduce(col_reduce_sum(exp(logits - cb(col_max))))
//     shifted   = logits - col_max                   // 列广播表达式
//     softmax   = exp(shifted) / cb(col_sum)         // 列广播表达式
//     log_sm    = shifted - cb(log(col_sum))         // 列广播表达式
//     grad      = (softmax - target) * rparam(1/batch)
//     loss      = -(1/batch) * Σ target * log_sm
//   CrossEntropy backward: grad = (softmax - target_onehot) / batch
// ─────────────────────────────────────────────────────────────────────────

#include "compute_engine.hpp"
#include "compute_layer_base.hpp"  // clone_tensor
#include "compute_tensor.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Loss — 引擎化损失函数基类
// ══════════════════════════════════════════════════════════════════════════
class Loss
{
protected:
    // 多精度（§9.1 / D9）：loss 前向 + backward 输出**同一精度**（默认 F32）。
    // 用 stable 字段而非 compute：f16 范围溢出（65504）风险集中在
    // softmax / log / 大词表归约，故 loss 链默认留在参考精度。
    PrecisionProfile p_;

public:
    virtual ~Loss() = default;

    // ── D7/D9：精度配置注入（工厂/CLI 在构造后调用）────────────────────
    void set_precision_profile(const PrecisionProfile& p) { p_ = p; }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

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

        // diff = pred - target（逐元素融合）
        auto diff = dsl::compute(engine,
            dsl::leaf(pred) - dsl::leaf(target),
            pred.rows(), pred.cols(), p_.stable);
        if (!diff) return std::unexpected(diff.error());

        // Σ diff²：逐元素链与列归约融合为**单次** dispatch（GPU 上 1 个融合
        // kernel，不物化 diff_sq 中间张量；CPU 该路径更慢，但 GPU 是主战场）。
        // 必须在下面"就地缩放 diff"之前完成（缩放与 diff 共享缓冲）。
        auto col_sum = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(*diff) * dsl::leaf(*diff)),
            diff->rows(), diff->cols(), p_.stable);
        if (!col_sum) return std::unexpected(col_sum.error());
        // 归约步用 dsl::compute_reduce（col_sum (1,B) → 行归约 (1,1)）
        auto total_t = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(*col_sum)),
            col_sum->rows(), col_sum->cols(), p_.stable);
        if (!total_t) return std::unexpected(total_t.error());
        auto total_v = detail::download_vector(engine, *total_t);   // 宿主桥（17 §3 D11）
        if (!total_v) return std::unexpected(total_v.error());
        const Scalar loss = (*total_v)[0] / total;

        // grad = diff * (2/N)：2/N 由 RParam 承载（值不进 expr_spec_key，同一
        // 结构跨形状共享 AOT shader）。**就地**缩放在 diff 自己的缓冲上完成
        // （零额外分配；guard 已用完 diff²，且 grad_input_ 与 diff 共享缓冲
        // ——但此时 diff² 已取用，故不产生"平方被缩放值"的污染）。
        auto grad = dsl::compute_into(engine,
            dsl::leaf(*diff) * dsl::rparam(scale), *diff);
        if (!grad) return std::unexpected(grad.error());
        grad_input_ = std::move(*diff);

        return loss;
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
//     shifted   = logits - col_max                    (列广播表达式)
//     exp_shift = exp(shifted)                        (exp 表达式)
//     col_sum   = Σ_c exp_shift[c][i]                 (col_reduce_sum)
//     softmax   = exp_shift / col_sum                 (列广播表达式)
//     grad      = softmax - target_onehot             (逐元素表达式)
//     log_sm    = shifted - log(col_sum)              (列广播表达式)
//     loss      = -(1/batch) * Σ target * log_sm
// ══════════════════════════════════════════════════════════════════════════
class CrossEntropyLoss final : public Loss
{
private:
    Tensor grad_input_;

    // ── 按列 softmax 的稠密结果：既给 grad（softmax），也给数值稳定的
    //    log_softmax（= shifted - log(col_sum)，避免 0*log(0)=NaN）。
    struct DenseSoftmax
    {
        Tensor softmax;     // (classes, batch)
        Tensor log_softmax; // (classes, batch) = shifted - log(col_sum)
    };

    // ── 按列 softmax（稠密 forward 使用）──────────────────────────────
    // col_max → col_sum = 列内 Σ exp(logits - col_max)（denom 融合结构：
    //         列归约 + exp 单 kernel，不物化 exp 中间张量）
    //         → softmax = exp(logits - col_max) / col_sum
    // 同时计算数值稳定的 log_softmax = (logits - col_max) - log(col_sum)：
    //   直接 log(softmax) 在极负 logits/大词表下 softmax→0 → log→-inf，
    //   与 target=0 相乘得 0*(-inf)=NaN；稳定形式中 shifted 与 log(col_sum)
    //   均有限，可避免该 NaN（稠密/软标签路径，稀疏 kernel 已用稳定形式）。
    [[nodiscard]] Result<DenseSoftmax> softmax_cols_(
        ComputeEngine& engine, const Tensor& logits) const
    {
        // col_max 用 dsl::compute_reduce（归约向量 (1,cols)，供下面 col_broadcast）
        auto col_max = dsl::compute_reduce(engine,
            dsl::col_reduce_max(dsl::leaf(logits)),
            logits.rows(), logits.cols(), p_.stable);
        if (!col_max) return std::unexpected(col_max.error());

        // denom = Σ_r exp(logits[r][c] - col_max[c]) 用 IR 表达式
        // （列归约 + ColBroadcast 视图 + exp，单 kernel，不物化 exp 张量）
        auto col_sum = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))),
            logits.rows(), logits.cols(), p_.stable);
        if (!col_sum) return std::unexpected(col_sum.error());

        // shifted / softmax / log_softmax 全部用**列广播表达式**：GPU 上各为 1 个
        // 融合 kernel，不产生 (classes,batch) 的中间拷贝（列广播是表达式内的一次
        // 读取，不额外分配、不额外 dispatch）。
        auto shifted = dsl::compute(engine,
            dsl::leaf(logits) - dsl::col_broadcast(*col_max),
            logits.rows(), logits.cols(), p_.stable);
        if (!shifted) return std::unexpected(shifted.error());

        auto softmax = dsl::compute(engine,
            dsl::exp(dsl::leaf(*shifted)) / dsl::col_broadcast(*col_sum),
            shifted->rows(), shifted->cols(), p_.stable);
        if (!softmax) return std::unexpected(softmax.error());

        // 稳定 log_softmax = shifted - log(col_sum)
        // col_sum ≥ 1（因 max 元素 shifted=0 → exp=1），故 log(col_sum) 有限
        auto log_col_sum = dsl::compute(engine,
            dsl::log(dsl::leaf(*col_sum)),
            col_sum->rows(), col_sum->cols(), p_.stable);
        if (!log_col_sum) return std::unexpected(log_col_sum.error());

        auto log_softmax = dsl::compute(engine,
            dsl::leaf(*shifted) - dsl::col_broadcast(*log_col_sum),
            shifted->rows(), shifted->cols(), p_.stable);
        if (!log_softmax) return std::unexpected(log_softmax.error());

        return DenseSoftmax{/*softmax=*/std::move(*softmax),
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

        // 1. softmax / log_softmax = softmax_cols(logits)（稳定形式）
        auto sm = softmax_cols_(engine, logits);
        if (!sm) return std::unexpected(sm.error());

        // 2. grad = (softmax - target) / batch
        //    loss = -(1/batch) * Σ target * log_softmax，故
        //    d(loss)/d(logits) = (softmax - one_hot) / batch。
        //    缺少 1/batch 缩放会使 SGD/动量、梯度裁剪与 PyTorch 不一致
        //    （Adam 的二阶矩会抵消常数缩放，但其他优化器不会）。
        //    1/batch 由 RParam 承载（值不进 expr_spec_key）→ 与整个逐元素链
        //    融合为单 kernel。
        auto grad = dsl::compute(engine,
            (dsl::leaf(sm->softmax) - dsl::leaf(target))
                * dsl::rparam(Scalar{1} / static_cast<Scalar>(batch)),
            sm->softmax.rows(), sm->softmax.cols(), p_.stable);
        if (!grad) return std::unexpected(grad.error());
        grad_input_ = std::move(*grad);

        // 3. log_softmax 已在 softmax_cols_ 内以数值稳定形式算出
        //    （= shifted - log(col_sum)，避免 0*log(0)=NaN）

        // 4+5. Σ target ⊙ log_softmax（先列归约再行归约 → (1,1)）：逐元素链与
        //      列归约融合为单次 dispatch（GPU 上 1 个融合 kernel）
        auto col_s = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(dsl::leaf(target) * dsl::leaf(sm->log_softmax)),
            target.rows(), target.cols(), p_.stable);
        if (!col_s) return std::unexpected(col_s.error());
        // 归约步用 dsl::compute_reduce（(1,B) → (1,1)）
        auto total_t = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(*col_s)),
            col_s->rows(), col_s->cols(), p_.stable);
        if (!total_t) return std::unexpected(total_t.error());

        // 6. loss = -total / batch — 下载标量
        auto total_v = detail::download_vector(engine, *total_t);   // 宿主桥（17 §3 D11）
        if (!total_v) return std::unexpected(total_v.error());

        return -(*total_v)[0] / static_cast<Scalar>(batch);
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
    // 融合路径（唯一路径）：
    //   1. 上传 labels / loss_mask 为 (1, total) 浮点张量（小传输）
    //   2. 融合表达式链：列内 max + denom + 稠密梯度 + 标签位置
    //      log_softmax（loss_vec）——不物化 (classes, total) 全 softmax
    //   3. loss = -(1/num_valid)·Σ loss_vec（下载标量）
    // 前置条件：vocab_size ≤ 2^24（labels 按 (1,total) 浮点打包，超出则标签
    // 不可精确表示）。路径内任何失败直接透传错误、不降级（"硬报错、不降级"）。
    //
    // 参数：
    //   labels     — 平坦标签数组，大小 = logits.cols()，值域 [0, vocab_size)
    //   loss_mask  — 可选，平坦 mask 数组，>0.5 表示参与 loss，否则清零梯度
    //   vocab_size — 词表大小，用于越界检查
    // ── 稀疏 CE：设备端 loss 和（非阻塞热路径入口）─────────────────
    // 返回 (1,1) 设备张量 = Σ loss_vec（**未归一化，不下载**）。
    // num_valid_out 回传有效 token 数；loss = -sum / num_valid。
    // 训练热循环用本接口 + engine.submit_scalar_readback 异步取 loss，
    // 避免每 step to_matrix 触发 end_batch + wait_in_flight 全流水线 drain。
    [[nodiscard]] Result<Tensor> forward_sparse_sum(
        ComputeEngine& engine, const Tensor& logits,
        std::span<const std::size_t> labels,
        std::span<const Scalar> loss_mask,
        std::size_t vocab_size,
        std::size_t& num_valid_out,
        Tensor* grad_reuse = nullptr)
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

        // ── 融合路径（不物化全 softmax；失败直接透传错误，不降级） ──
        return fused_forward_sparse_(engine, logits, labels, loss_mask, vocab_size,
                                     num_valid_out, grad_reuse);
    }

    // 同步版稀疏 CE（测试/评估等非热路径）：内部下载标量（宿主桥，17 §3 D11）。
    [[nodiscard]] Result<Scalar> forward_sparse(
        ComputeEngine& engine, const Tensor& logits,
        std::span<const std::size_t> labels,
        std::span<const Scalar> loss_mask = {},
        std::size_t vocab_size = 0)
    {
        std::size_t num_valid = 0;
        auto sum_t = forward_sparse_sum(engine, logits, labels, loss_mask,
                                        vocab_size, num_valid);
        if (!sum_t) return std::unexpected(sum_t.error());
        auto sum_v = detail::download_vector(engine, *sum_t);
        if (!sum_v) return std::unexpected(sum_v.error());
        return (num_valid > 0)
            ? -(*sum_v)[0] / static_cast<Scalar>(num_valid)
            : Scalar{0};
    }

    // ── 融合路径实现（IR 组合：col_max 原语 + denom/loss_vec/grad 表达式）──
    // 不物化 (classes, total) 全 softmax：
    //   col_max  = 列内 max（原语）
    //   denom    = col_sum(exp(logits - cb(col_max)))          （IR 表达式）
    //   loss_vec = (rg(logits) - cb(col_max) - log(denom)) * cb(mask)  （IR，(1,total)）
    //   grad     = (exp/logits 链 - select(Row==cb(labels),1,0)) * cb(mask) * inv（IR）
    [[nodiscard]] Result<Tensor> fused_forward_sparse_(
        ComputeEngine& engine, const Tensor& logits,
        std::span<const std::size_t> labels,
        std::span<const Scalar> loss_mask,
        std::size_t vocab_size,
        std::size_t& num_valid_out,
        Tensor* grad_reuse = nullptr)
    {
        const std::size_t classes = logits.rows();
        const std::size_t total = logits.cols();

        // 1. 上传 labels（(1, total) 浮点打包；vocab_size ≤ 2^24 时可精确表示）。
        //    越界 label 修正为 0（GPU RowGather 无越界守卫，靠 mask 置零无效）
        //    宿主桥（17 §3 D11）：L2 不出现 Matrix，标量缓冲直传引擎。
        std::vector<Scalar> labels_v(total);
        for (std::size_t i = 0; i < total; ++i)
            labels_v[i] = static_cast<Scalar>(
                (labels[i] < vocab_size) ? labels[i] : 0);
        auto labels_t = detail::upload_span(engine, 1, total, Precision::F32,
                                            std::span(labels_v));
        if (!labels_t) return std::unexpected(labels_t.error());

        // 2. 有效 mask（总是构造，(1,total) 0/1；含 mask 缺失与 label 越界修正）：
        //    valid = (loss_mask 空 || mask[i]>=0.5) && labels[i] < vocab_size
        std::vector<Scalar> mask_v(total);
        for (std::size_t i = 0; i < total; ++i)
        {
            const bool masked = !loss_mask.empty() && loss_mask[i] < Scalar{0.5};
            mask_v[i] = (!masked && labels[i] < vocab_size) ? Scalar{1} : Scalar{0};
        }
        auto mask_t = detail::upload_span(engine, 1, total, Precision::F32,
                                          std::span(mask_v));
        if (!mask_t) return std::unexpected(mask_t.error());

        // 3. num_valid（与 mask 判定一致）
        std::size_t num_valid = 0;
        for (std::size_t i = 0; i < total; ++i)
            if (mask_v[i] >= Scalar{0.5}) ++num_valid;
        const Scalar inv_num_valid = (num_valid > 0)
            ? Scalar{1} / static_cast<Scalar>(num_valid) : Scalar{0};

        // 4. col_max → denom（IR）→ loss_vec / grad（IR）
        //    col_max 用 dsl::compute_reduce（归约向量，输出 (1,total) 同形）
        auto col_max = dsl::compute_reduce(engine,
            dsl::col_reduce_max(dsl::leaf(logits)),
            classes, total, p_.stable);
        if (!col_max) return std::unexpected(col_max.error());
        auto denom = dsl::compute_reduce(engine,
            dsl::col_reduce_sum(
                dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))),
            classes, total, p_.stable);
        if (!denom) return std::unexpected(denom.error());
        // loss_vec[c] = (logits[label[c]][c] - col_max[c] - log(denom[c])) * mask[c]
        auto loss_vec = dsl::compute(engine,
            (dsl::row_gather(logits, *labels_t) - dsl::col_broadcast(*col_max)
             - dsl::log(dsl::leaf(*denom))) * dsl::col_broadcast(*mask_t),
            1, total, p_.stable);
        if (!loss_vec) return std::unexpected(loss_vec.error());
        // grad[r][c] = (exp(logits-col_max)/denom - [r==label[c]]) * mask[c] / num_valid
        // 1/num_valid 由 RParam 承载（运行时值、不进 expr_spec_key）→ 与整个
        // 逐元素链融合为单 kernel
        auto grad_expr =
            (dsl::exp(dsl::leaf(logits) - dsl::col_broadcast(*col_max))
                / dsl::col_broadcast(*denom)
             - dsl::select(dsl::row() == dsl::col_broadcast(*labels_t),
                           Scalar{1}, Scalar{0}))
            * dsl::col_broadcast(*mask_t)
            * dsl::rparam(inv_num_valid);

        // 显存优化：grad_reuse 与 logits 同形同精度时把梯度**原地**写进该
        // 缓冲（典型用法：传 &logits）——省一份 (vocab×total) 显存，训练峰值
        // 第二大单项（bench 配置实测 513MB）。安全性：本表达式对输入只有
        // 逐元素读取 + 列/标量广播（row() 是隐式行下标，不是跨行 gather），
        // 每个元素读后写同址；唯一的跨行 row_gather(logits) 在上方 loss_vec
        // 中，且 loss_vec 已算入独立缓冲。
        if (grad_reuse != nullptr && grad_reuse->valid()
            && grad_reuse->rows() == classes && grad_reuse->cols() == total
            && grad_reuse->precision() == logits.precision())
        {
            auto w = dsl::compute_into(engine, grad_expr, *grad_reuse);
            if (!w) return std::unexpected(w.error());
            grad_input_ = *grad_reuse;
        }
        else
        {
            auto grad = dsl::compute(engine, grad_expr, classes, total, p_.stable);
            if (!grad) return std::unexpected(grad.error());
            grad_input_ = std::move(*grad);
        }

        // 5. loss_sum = Σ loss_vec（无效列已乘 0）——**不下载**：
        //    热路径由调用方经 engine.submit_scalar_readback 异步取回；
        //    同步版 forward_sparse 在此之后 to_matrix。
        // 归约步用 dsl::compute_reduce（(1,total) → (1,1)）
        auto total_t = dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(*loss_vec)),
            loss_vec->rows(), loss_vec->cols(), p_.stable);
        if (!total_t) return std::unexpected(total_t.error());
        num_valid_out = num_valid;
        return total_t;
    }
};

} // namespace nn

