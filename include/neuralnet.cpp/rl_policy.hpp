#pragma once

// ── rl_policy.hpp — categorical policy distribution 与 objective 组件 ──────
//
// docs/todo/31 RL-1b：
//   · logits: (actions, T)，action/advantage/log-prob/mask: (1,T) 或同形；
//   · policy objective 的上下文入口强类型接收 PolicyBatchView；
//   · 继续适配现有 Loss：forward(pred,target) 在没有 RL 上下文时明确报错；
//   · 所有梯度由 softmax/DSL 组合表达，不新增引擎原语或 shader。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "compute_loss.hpp"
#include "rl_experience.hpp"

namespace nn
{

class CategoricalDistribution
{
public:
    struct Probabilities
    {
        Tensor softmax;
        Tensor log_softmax;
    };

    struct Evaluation
    {
        Tensor value;
        Tensor gradient;
    };

    struct SampleResult
    {
        Tensor action;
        Tensor log_prob;
    };

private:
    Precision precision_ = Precision::F32;

    [[nodiscard]] Result<void> validate_actions(
        ComputeEngine& engine, const Tensor& logits,
        const Tensor& actions) const
    {
        if (logits.rows() == 0 || logits.cols() == 0)
            NN_FAIL("CategoricalDistribution: logits 不能为空");
        if (actions.rows() != 1 || actions.cols() != logits.cols())
            NN_FAIL("CategoricalDistribution: action 形状必须为 (1,T)");
        NN_TRY(v, detail::download_vector(engine, actions));
        for (std::size_t i = 0; i < v->size(); ++i)
        {
            const Scalar a = (*v)[i];
            if (!std::isfinite(a) || a < Scalar{0} ||
                a >= static_cast<Scalar>(logits.rows()) ||
                std::fabs(a - std::round(a)) > Scalar{1e-4})
                NN_FAIL("CategoricalDistribution: action[" +
                        std::to_string(i) + "]=" + std::to_string(a) +
                        " 越界或不是整数（动作数=" +
                        std::to_string(logits.rows()) + "）");
        }
        return {};
    }

    [[nodiscard]] Result<Probabilities> probabilities(
        ComputeEngine& engine, const Tensor& logits) const
    {
        if (logits.rows() == 0 || logits.cols() == 0)
            NN_FAIL("CategoricalDistribution: logits 不能为空");
        NN_TRY(col_max,
               dsl::compute_reduce(engine,
                                   dsl::col_reduce_max(dsl::leaf(logits)),
                                   logits.rows(), logits.cols(),
                                   precision_));
        NN_TRY(col_sum,
               dsl::compute_reduce(
                   engine,
                   dsl::col_reduce_sum(
                       dsl::exp(dsl::leaf(logits) -
                                dsl::col_broadcast(*col_max))),
                   logits.rows(), logits.cols(), precision_));
        NN_TRY(shifted,
               dsl::compute(engine,
                            dsl::leaf(logits) -
                                dsl::col_broadcast(*col_max),
                            logits.rows(), logits.cols(), precision_));
        NN_TRY(softmax,
               dsl::compute(engine,
                            dsl::exp(dsl::leaf(*shifted)) /
                                dsl::col_broadcast(*col_sum),
                            logits.rows(), logits.cols(), precision_));
        NN_TRY(log_col_sum,
               dsl::compute(engine, dsl::log(dsl::leaf(*col_sum)),
                            col_sum->rows(), col_sum->cols(), precision_));
        NN_TRY(log_softmax,
               dsl::compute(engine,
                            dsl::leaf(*shifted) -
                                dsl::col_broadcast(*log_col_sum),
                            logits.rows(), logits.cols(), precision_));
        return Probabilities{.softmax = std::move(*softmax),
                             .log_softmax = std::move(*log_softmax)};
    }

public:
    explicit CategoricalDistribution(Precision precision = Precision::F32)
        : precision_(precision)
    {
    }

    [[nodiscard]] Result<Tensor> log_prob(
        ComputeEngine& engine, const Tensor& logits,
        const Tensor& actions) const
    {
        NN_TRY_CHECK(validate_actions(engine, logits, actions));
        NN_TRY(probs, probabilities(engine, logits));
        return dsl::compute_reduce(
            engine,
            dsl::col_reduce_sum(
                dsl::select(dsl::row() ==
                                dsl::col_broadcast(actions),
                            dsl::leaf(probs->log_softmax),
                            Scalar{0})),
            logits.rows(), logits.cols(), precision_);
    }

    [[nodiscard]] Result<Tensor> entropy(
        ComputeEngine& engine, const Tensor& logits) const
    {
        NN_TRY(probs, probabilities(engine, logits));
        NN_TRY(term,
               dsl::compute(engine,
                            dsl::leaf(probs->softmax) *
                                dsl::leaf(probs->log_softmax),
                            logits.rows(), logits.cols(), precision_));
        NN_TRY(sum,
               dsl::compute_reduce(
                   engine, dsl::col_reduce_sum(dsl::leaf(*term)),
                   logits.rows(), logits.cols(), precision_));
        return dsl::compute(engine, dsl::leaf(*sum) * dsl::rparam(-1),
                            sum->rows(), sum->cols(), precision_);
    }

    [[nodiscard]] Result<Evaluation> log_prob_with_gradient(
        ComputeEngine& engine, const Tensor& logits, const Tensor& actions,
        const Tensor& upstream) const
    {
        NN_TRY_CHECK(validate_actions(engine, logits, actions));
        if (upstream.rows() != 1 || upstream.cols() != logits.cols())
            NN_FAIL("CategoricalDistribution: log_prob upstream 必须为 (1,T)");
        NN_TRY(probs, probabilities(engine, logits));
        NN_TRY(value,
               dsl::compute_reduce(
                   engine,
                   dsl::col_reduce_sum(
                       dsl::select(
                           dsl::row() == dsl::col_broadcast(actions),
                           dsl::leaf(probs->log_softmax), Scalar{0})),
                   logits.rows(), logits.cols(), precision_));
        NN_TRY(grad,
               dsl::compute(
                   engine,
                   (dsl::leaf(probs->softmax) -
                    dsl::select(
                        dsl::row() == dsl::col_broadcast(actions),
                        Scalar{1}, Scalar{0})) *
                       dsl::col_broadcast(upstream),
                   logits.rows(), logits.cols(), precision_));
        return Evaluation{.value = std::move(*value),
                          .gradient = std::move(*grad)};
    }

    [[nodiscard]] Result<Evaluation> entropy_with_gradient(
        ComputeEngine& engine, const Tensor& logits,
        const Tensor& upstream) const
    {
        if (upstream.rows() != 1 || upstream.cols() != logits.cols())
            NN_FAIL("CategoricalDistribution: entropy upstream 必须为 (1,T)");
        NN_TRY(probs, probabilities(engine, logits));
        NN_TRY(term,
               dsl::compute(engine,
                            dsl::leaf(probs->softmax) *
                                dsl::leaf(probs->log_softmax),
                            logits.rows(), logits.cols(), precision_));
        NN_TRY(sum,
               dsl::compute_reduce(
                   engine, dsl::col_reduce_sum(dsl::leaf(*term)),
                   logits.rows(), logits.cols(), precision_));
        NN_TRY(value,
               dsl::compute(engine, dsl::leaf(*sum) * dsl::rparam(-1),
                            sum->rows(), sum->cols(), precision_));
        NN_TRY(grad,
               dsl::compute(
                   engine,
                   (dsl::leaf(probs->softmax) *
                    (dsl::leaf(probs->log_softmax) +
                     dsl::col_broadcast(*value))) *
                       dsl::col_broadcast(upstream) * dsl::rparam(-1),
                   logits.rows(), logits.cols(), precision_));
        return Evaluation{.value = std::move(*value),
                          .gradient = std::move(*grad)};
    }

    [[nodiscard]] Result<SampleResult> sample(
        ComputeEngine& engine, const Tensor& logits,
        std::uint64_t seed) const
    {
        if (logits.rows() == 0 || logits.cols() == 0)
            NN_FAIL("CategoricalDistribution::sample: logits 不能为空");
        NN_TRY(values,
               detail::download_vector(engine, logits));
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::vector<Scalar> actions(logits.cols(), Scalar{0});
        for (std::size_t c = 0; c < logits.cols(); ++c)
        {
            Scalar mx = -std::numeric_limits<Scalar>::infinity();
            for (std::size_t r = 0; r < logits.rows(); ++r)
                mx = std::max(mx, (*values)[r * logits.cols() + c]);
            double total = 0.0;
            std::vector<double> probs(logits.rows());
            for (std::size_t r = 0; r < logits.rows(); ++r)
            {
                probs[r] = std::exp(static_cast<double>(
                    (*values)[r * logits.cols() + c] - mx));
                total += probs[r];
            }
            const double u = uniform(rng) * total;
            double running = 0.0;
            std::size_t chosen = logits.rows() - 1;
            for (std::size_t r = 0; r < logits.rows(); ++r)
            {
                running += probs[r];
                if (u <= running)
                {
                    chosen = r;
                    break;
                }
            }
            actions[c] = static_cast<Scalar>(chosen);
        }
        NN_TRY(action,
               detail::upload_span(engine, 1, logits.cols(), precision_,
                                   std::span<const Scalar>(actions)));
        NN_TRY_CHECK(validate_actions(engine, logits, *action));
        NN_TRY(lp, log_prob(engine, logits, *action));
        return SampleResult{.action = std::move(*action),
                            .log_prob = std::move(*lp)};
    }
};

struct ObjectiveOutput
{
    Scalar loss = 0;
    Tensor gradient;
};

class ContextualObjective : public Loss
{
private:
    Tensor cached_grad_;

protected:
    [[nodiscard]] virtual Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& prediction,
        const PolicyBatchView& batch) = 0;

    [[nodiscard]] Precision precision() const noexcept { return p_.stable; }

public:
    [[nodiscard]] Result<Scalar> forward(
        ComputeEngine&, const Tensor&, const Tensor&) override
    {
        NN_FAIL("ContextualObjective::forward: policy/value objective 需要强类型"
                " PolicyBatchView；请调用 forward(engine, prediction, batch)");
    }

    [[nodiscard]] Result<Scalar> forward(
        ComputeEngine& engine, const Tensor& prediction,
        const PolicyBatchView& batch)
    {
        auto result = evaluate(engine, prediction, batch);
        if (!result)
        {
            cached_grad_ = Tensor{};
            return std::unexpected(result.error());
        }
        cached_grad_ = std::move(result->gradient);
        return result->loss;
    }

    [[nodiscard]] Result<Tensor> backward() override
    {
        if (!cached_grad_.valid())
            NN_FAIL("ContextualObjective::backward: 尚未成功执行上下文 forward");
        return cached_grad_;
    }
};

namespace detail
{
[[nodiscard]] inline Result<Scalar> rl_masked_units(
    ComputeEngine& engine, const Tensor& mask, Precision p)
{
    NN_TRY(sum, dsl::compute_reduce(
                    engine, dsl::col_reduce_sum(dsl::leaf(mask)),
                    mask.rows(), mask.cols(), p));
    NN_TRY(total, dsl::compute_reduce(
                      engine, dsl::row_reduce_sum(dsl::leaf(*sum)),
                      1, mask.cols(), p));
    NN_TRY(v, download_vector(engine, *total));
    return (*v)[0];
}

[[nodiscard]] inline Result<Scalar> rl_masked_sum(
    ComputeEngine& engine, const Tensor& values, const Tensor& mask,
    Precision p)
{
    NN_TRY(weighted, dsl::compute(engine,
                                  dsl::leaf(values) * dsl::leaf(mask),
                                  values.rows(), values.cols(), p));
    NN_TRY(col, dsl::compute_reduce(
                    engine, dsl::col_reduce_sum(dsl::leaf(*weighted)),
                    values.rows(), values.cols(), p));
    NN_TRY(total, dsl::compute_reduce(
                      engine, dsl::row_reduce_sum(dsl::leaf(*col)),
                      1, values.cols(), p));
    NN_TRY(v, download_vector(engine, *total));
    return (*v)[0];
}
} // namespace detail

class ReinforceObjective final : public ContextualObjective
{
private:
    bool use_old_value_baseline_ = false;
    CategoricalDistribution distribution_;

public:
    explicit ReinforceObjective(
        bool use_old_value_baseline = false,
        Precision precision = Precision::F32)
        : use_old_value_baseline_(use_old_value_baseline),
          distribution_(precision)
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& logits,
        const PolicyBatchView& batch) override
    {
        const Tensor* advantage = batch.advantage();
        Tensor baseline_adv;
        if (advantage == nullptr && use_old_value_baseline_)
        {
            const Tensor* returns = batch.returns();
            const Tensor* old_value = batch.old_value();
            if (returns == nullptr || old_value == nullptr)
                NN_FAIL("ReinforceObjective: baseline 需要 return 与 old_value");
            NN_TRY(adv, dsl::compute(engine,
                                     dsl::leaf(*returns) -
                                         dsl::leaf(*old_value),
                                     returns->rows(), returns->cols(),
                                     precision()));
            baseline_adv = std::move(*adv);
            advantage = &baseline_adv;
        }
        if (advantage == nullptr)
            NN_FAIL("ReinforceObjective: 缺少 advantage（或启用 old_value baseline）");
        const Tensor& mask = batch.action_mask();
        if (advantage->rows() != mask.rows() ||
            advantage->cols() != mask.cols())
            NN_FAIL("ReinforceObjective: advantage/action_mask 形状不一致");

        NN_TRY(units_r,
               detail::rl_masked_units(engine, mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, logits.rows(), logits.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }

        NN_TRY(logp,
               distribution_.log_prob(engine, logits, batch.action()));
        NN_TRY(weighted_adv,
               dsl::compute(engine,
                            dsl::leaf(*advantage) * dsl::leaf(mask) *
                                dsl::rparam(-Scalar{1} / units),
                            1, logits.cols(), precision()));
        NN_TRY(eval,
               distribution_.log_prob_with_gradient(
                   engine, logits, batch.action(), *weighted_adv));
        NN_TRY(loss_terms,
               dsl::compute(engine,
                            dsl::leaf(eval->value) * dsl::leaf(*advantage) *
                                dsl::leaf(mask),
                            1, logits.cols(), precision()));
        NN_TRY(loss_sum,
               detail::rl_masked_sum(engine, *loss_terms, mask,
                                     precision()));
        return ObjectiveOutput{
            .loss = -*loss_sum / units,
            .gradient = std::move(eval->gradient),
        };
    }
};

class ValueRegressionObjective final : public ContextualObjective
{
private:
    Scalar coefficient_ = 1;

public:
    explicit ValueRegressionObjective(Scalar coefficient = 1,
                                      Precision precision = Precision::F32)
        : coefficient_(coefficient)
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& value,
        const PolicyBatchView& batch) override
    {
        const Tensor* returns = batch.returns();
        const Tensor* mask = batch.value_mask();
        if (returns == nullptr)
            NN_FAIL("ValueRegressionObjective: 缺少 return");
        if (mask == nullptr)
            NN_FAIL("ValueRegressionObjective: 缺少显式 value_mask");
        if (returns->rows() != value.rows() ||
            returns->cols() != value.cols() ||
            mask->rows() != value.rows() ||
            mask->cols() != value.cols())
            NN_FAIL("ValueRegressionObjective: value/return/mask 形状不一致");
        NN_TRY(units_r, detail::rl_masked_units(engine, *mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, value.rows(), value.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }
        NN_TRY(diff, dsl::compute(engine,
                                  dsl::leaf(value) - dsl::leaf(*returns),
                                  value.rows(), value.cols(), precision()));
        NN_TRY(square, dsl::compute(engine,
                                    dsl::leaf(*diff) * dsl::leaf(*diff),
                                    value.rows(), value.cols(), precision()));
        NN_TRY(loss_sum,
               detail::rl_masked_sum(engine, *square, *mask, precision()));
        NN_TRY(grad, dsl::compute(engine,
                                  dsl::leaf(*diff) *
                                      dsl::leaf(*mask) *
                                      dsl::rparam(Scalar{2} *
                                                  coefficient_ / units),
                                  value.rows(), value.cols(), precision()));
        return ObjectiveOutput{
            .loss = coefficient_ * *loss_sum / units,
            .gradient = std::move(*grad),
        };
    }
};

class EntropyBonusObjective final : public ContextualObjective
{
private:
    Scalar coefficient_ = 0.01f;
    CategoricalDistribution distribution_;

public:
    explicit EntropyBonusObjective(
        Scalar coefficient = 0.01f,
        Precision precision = Precision::F32)
        : coefficient_(coefficient), distribution_(precision)
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& logits,
        const PolicyBatchView& batch) override
    {
        const Tensor& mask = batch.action_mask();
        NN_TRY(units_r,
               detail::rl_masked_units(engine, mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, logits.rows(), logits.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }
        NN_TRY(upstream,
               dsl::compute(engine,
                            dsl::leaf(mask) *
                                dsl::rparam(-coefficient_ / units),
                            1, logits.cols(), precision()));
        NN_TRY(eval,
               distribution_.entropy_with_gradient(
                   engine, logits, *upstream));
        NN_TRY(total_h,
               detail::rl_masked_sum(engine, eval->value, mask,
                                     precision()));
        return ObjectiveOutput{
            .loss = -coefficient_ * *total_h / units,
            .gradient = std::move(eval->gradient),
        };
    }
};

class SampledKlObjective final : public ContextualObjective
{
private:
    Scalar coefficient_ = 1;
    CategoricalDistribution distribution_;

public:
    explicit SampledKlObjective(
        Scalar coefficient = 1,
        Precision precision = Precision::F32)
        : coefficient_(coefficient), distribution_(precision)
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& logits,
        const PolicyBatchView& batch) override
    {
        const Tensor* old_logp = batch.old_log_prob();
        if (old_logp == nullptr)
            NN_FAIL("SampledKlObjective: 缺少 old_log_prob");
        const Tensor& mask = batch.action_mask();
        NN_TRY(units_r,
               detail::rl_masked_units(engine, mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, logits.rows(), logits.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }
        NN_TRY(upstream,
               dsl::compute(engine,
                            dsl::leaf(mask) *
                                dsl::rparam(coefficient_ / units),
                            1, logits.cols(), precision()));
        NN_TRY(eval,
               distribution_.log_prob_with_gradient(
                   engine, logits, batch.action(), *upstream));
        NN_TRY(diff,
               dsl::compute(engine,
                            dsl::leaf(eval->value) - dsl::leaf(*old_logp),
                            1, logits.cols(), precision()));
        NN_TRY(total,
               detail::rl_masked_sum(engine, *diff, mask, precision()));
        return ObjectiveOutput{
            .loss = coefficient_ * *total / units,
            .gradient = std::move(eval->gradient),
        };
    }
};

} // namespace nn
