#pragma once

// ── rl_policy.hpp — categorical policy distribution 与 objective 组件 ──────
//
// docs/internal/31 RL-1b：
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

class PolicyDistribution
{
public:
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

    virtual ~PolicyDistribution() = default;
    virtual Result<Tensor> log_prob(
        ComputeEngine& engine, const Tensor& parameters,
        const Tensor& actions) const = 0;
    virtual Result<Tensor> entropy(
        ComputeEngine& engine, const Tensor& parameters) const = 0;
    virtual Result<Evaluation> log_prob_with_gradient(
        ComputeEngine& engine, const Tensor& parameters,
        const Tensor& actions, const Tensor& upstream) const = 0;
    virtual Result<Evaluation> entropy_with_gradient(
        ComputeEngine& engine, const Tensor& parameters,
        const Tensor& upstream) const = 0;
    virtual Result<SampleResult> sample(
        ComputeEngine& engine, const Tensor& parameters,
        std::uint64_t seed) const = 0;
};

class CategoricalDistribution final : public PolicyDistribution
{
private:
    struct Probabilities
    {
        Tensor softmax;
        Tensor log_softmax;
    };

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
        const Tensor& actions) const override
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
        ComputeEngine& engine, const Tensor& logits) const override
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
        const Tensor& upstream) const override
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
                   (dsl::select(
                        dsl::row() == dsl::col_broadcast(actions),
                        Scalar{1}, Scalar{0}) -
                    dsl::leaf(probs->softmax)) *
                       dsl::col_broadcast(upstream),
                   logits.rows(), logits.cols(), precision_));
        return Evaluation{.value = std::move(*value),
                          .gradient = std::move(*grad)};
    }

    [[nodiscard]] Result<Evaluation> entropy_with_gradient(
        ComputeEngine& engine, const Tensor& logits,
        const Tensor& upstream) const override
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
        std::uint64_t seed) const override
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
            NN_FAIL(result.error());
        }
        cached_grad_ = std::move(result->gradient);
        return result->loss;
    }

    // 供组合目标复用组件，而不依赖其隐式缓存。
    [[nodiscard]] Result<ObjectiveOutput> evaluate_context(
        ComputeEngine& engine, const Tensor& prediction,
        const PolicyBatchView& batch)
    {
        return evaluate(engine, prediction, batch);
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
[[nodiscard]] inline Result<void> rl_check_policy_row(const Tensor& logits, const Tensor* value,
                                                      const char* name)
{
    if (value == nullptr) return {};
    if (value->rows() != 1 || value->cols() != logits.cols())
        NN_FAIL(std::string("RL objective: ") + name + " 形状必须为 (1,T)，T=" +
                std::to_string(logits.cols()) + "（实际 " + value->shape_str() + "）");
    return {};
}

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
    std::shared_ptr<const PolicyDistribution> distribution_;

public:
    explicit ReinforceObjective(
        bool use_old_value_baseline = false,
        Precision precision = Precision::F32,
        std::shared_ptr<const PolicyDistribution> distribution = {})
        : use_old_value_baseline_(use_old_value_baseline),
          distribution_(distribution
                            ? std::move(distribution)
                            : std::make_shared<CategoricalDistribution>(
                                  precision))
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
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &batch.action(), "action"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &batch.action_mask(), "action_mask"));
        const Tensor* advantage = batch.advantage();
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, advantage, "advantage"));
        Tensor baseline_adv;
        if (advantage == nullptr && use_old_value_baseline_)
        {
            const Tensor* returns = batch.returns();
            const Tensor* old_value = batch.old_value();
            if (returns == nullptr || old_value == nullptr)
                NN_FAIL("ReinforceObjective: baseline 需要 return 与 old_value");
            NN_TRY_CHECK(detail::rl_check_policy_row(logits, returns, "return"));
            NN_TRY_CHECK(detail::rl_check_policy_row(logits, old_value, "old_value"));
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
               distribution_->log_prob(engine, logits, batch.action()));
        NN_TRY(weighted_adv,
               dsl::compute(engine,
                            dsl::leaf(*advantage) * dsl::leaf(mask) *
                                dsl::rparam(-Scalar{1} / units),
                            1, logits.cols(), precision()));
        NN_TRY(eval,
               distribution_->log_prob_with_gradient(
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
    std::shared_ptr<const PolicyDistribution> distribution_;

public:
    explicit EntropyBonusObjective(
        Scalar coefficient = 0.01f,
        Precision precision = Precision::F32,
        std::shared_ptr<const PolicyDistribution> distribution = {})
        : coefficient_(coefficient),
          distribution_(distribution
                            ? std::move(distribution)
                            : std::make_shared<CategoricalDistribution>(
                                  precision))
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
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &batch.action(), "action"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &mask, "action_mask"));
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
               distribution_->entropy_with_gradient(
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
    std::shared_ptr<const PolicyDistribution> distribution_;

public:
    explicit SampledKlObjective(
        Scalar coefficient = 1,
        Precision precision = Precision::F32,
        std::shared_ptr<const PolicyDistribution> distribution = {})
        : coefficient_(coefficient),
          distribution_(distribution
                            ? std::move(distribution)
                            : std::make_shared<CategoricalDistribution>(
                                  precision))
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
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &batch.action(), "action"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, old_logp, "old_log_prob"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &mask, "action_mask"));
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
               distribution_->log_prob_with_gradient(
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

struct PpoPolicyMetrics
{
    Scalar policy_loss = 0;
    Scalar ratio_mean = 1;
    Scalar approx_kl = 0;
    Scalar clip_fraction = 0;
};

class ClippedPolicyObjective final : public ContextualObjective
{
private:
    Scalar clip_epsilon_ = 0.2f;
    std::shared_ptr<const PolicyDistribution> distribution_;
    PpoPolicyMetrics metrics_;

public:
    explicit ClippedPolicyObjective(
        Scalar clip_epsilon = 0.2f,
        Precision precision = Precision::F32,
        std::shared_ptr<const PolicyDistribution> distribution = {})
        : clip_epsilon_(clip_epsilon),
          distribution_(distribution
                            ? std::move(distribution)
                            : std::make_shared<CategoricalDistribution>(
                                  precision))
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

    [[nodiscard]] const PpoPolicyMetrics& metrics() const noexcept
    {
        return metrics_;
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& logits,
        const PolicyBatchView& batch) override
    {
        const Tensor* old_logp = batch.old_log_prob();
        const Tensor* advantage = batch.advantage();
        if (old_logp == nullptr)
            NN_FAIL("ClippedPolicyObjective: 缺少 old_log_prob");
        if (advantage == nullptr)
            NN_FAIL("ClippedPolicyObjective: 缺少 advantage");
        const Tensor& mask = batch.action_mask();
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &batch.action(), "action"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, old_logp, "old_log_prob"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, advantage, "advantage"));
        NN_TRY_CHECK(detail::rl_check_policy_row(logits, &mask, "action_mask"));

        NN_TRY(units_r,
               detail::rl_masked_units(engine, mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            metrics_ = {};
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, logits.rows(), logits.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }

        NN_TRY(new_logp,
               distribution_->log_prob(engine, logits, batch.action()));
        NN_TRY_MSG(diff, "PPO policy log-ratio",
                   dsl::compute(engine,
                                dsl::leaf(*new_logp) -
                                    dsl::leaf(*old_logp),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(ratio, "PPO policy ratio",
                   dsl::compute(engine, dsl::exp(dsl::leaf(*diff)),
                                1, logits.cols(), precision()));
        const Scalar lower = Scalar{1} - clip_epsilon_;
        const Scalar upper = Scalar{1} + clip_epsilon_;
        NN_TRY_MSG(clipped_ratio, "PPO policy clipped ratio",
                   dsl::compute(
                   engine,
                   dsl::select(
                       dsl::leaf(*ratio) < dsl::rparam(lower),
                       dsl::rparam(lower),
                       dsl::select(
                           dsl::leaf(*ratio) > dsl::rparam(upper),
                           dsl::rparam(upper), dsl::leaf(*ratio))),
                   1, logits.cols(), precision()));
        NN_TRY_MSG(objective, "PPO policy ratio*advantage",
                   dsl::compute(engine,
                                dsl::leaf(*ratio) *
                                    dsl::leaf(*advantage),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(clipped_objective, "PPO clipped ratio*advantage",
                   dsl::compute(engine,
                                dsl::leaf(*clipped_ratio) *
                                    dsl::leaf(*advantage),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(surrogate, "PPO policy min",
                   dsl::compute(
                       engine,
                       dsl::min(dsl::leaf(*objective),
                                dsl::leaf(*clipped_objective)),
                       1, logits.cols(), precision()));
        NN_TRY_MSG(loss_terms, "PPO policy masked surrogate",
                   dsl::compute(engine,
                                dsl::leaf(*surrogate) * dsl::leaf(mask),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(loss_sum, "PPO policy reduce",
                   detail::rl_masked_sum(engine, *loss_terms, mask,
                                         precision()));

        const auto ratio_v = dsl::leaf(*ratio);
        const auto adv_v = dsl::leaf(*advantage);
        // d surrogate / d log_prob。未被裁剪的一侧始终是 ratio*advantage；
        // A>0 时只裁上界，A<0 时只裁下界。
        const auto positive = dsl::select(ratio_v > dsl::rparam(upper), Scalar{0}, ratio_v * adv_v);
        const auto negative = dsl::select(ratio_v < dsl::rparam(lower), Scalar{0}, ratio_v * adv_v);
        const auto factor = dsl::select(
            adv_v > Scalar{0}, positive, negative);
        NN_TRY_MSG(factor_tensor, "PPO policy factor",
                   dsl::compute(engine, factor, 1, logits.cols(),
                                precision()));
        auto upstream_expr =
            dsl::leaf(*factor_tensor) * dsl::leaf(mask) *
            dsl::rparam(-Scalar{1} / units);
        NN_TRY_MSG(upstream, "PPO policy upstream",
                   dsl::compute(engine, upstream_expr,
                                1, logits.cols(), precision()));
        NN_TRY_MSG(eval, "PPO policy log-prob gradient",
               distribution_->log_prob_with_gradient(
                       engine, logits, batch.action(), *upstream));

        NN_TRY_MSG(ratio_weighted, "PPO ratio weighted",
                   dsl::compute(engine,
                                ratio_v * dsl::leaf(mask),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(ratio_sum, "PPO ratio reduce",
                   detail::rl_masked_sum(engine, *ratio_weighted, mask,
                                         precision()));
        NN_TRY_MSG(kl_terms, "PPO KL terms",
                   dsl::compute(engine,
                                dsl::leaf(*diff) * dsl::leaf(*diff) *
                                    dsl::leaf(mask),
                                1, logits.cols(), precision()));
        NN_TRY_MSG(kl_sum, "PPO KL reduce",
                   detail::rl_masked_sum(engine, *kl_terms, mask,
                                         precision()));
        NN_TRY_MSG(outside, "PPO clip indicator",
                   dsl::compute(
                   engine,
                   dsl::select(
                       ratio_v < dsl::rparam(lower), Scalar{1},
                       dsl::select(ratio_v > dsl::rparam(upper),
                                   Scalar{1}, Scalar{0})) *
                       dsl::leaf(mask),
                   1, logits.cols(), precision()));
        NN_TRY_MSG(outside_sum, "PPO clip reduce",
                   detail::rl_masked_sum(engine, *outside, mask,
                                         precision()));

        metrics_.policy_loss = -*loss_sum / units;
        metrics_.ratio_mean = *ratio_sum / units;
        metrics_.approx_kl = Scalar{0.5} * *kl_sum / units;
        metrics_.clip_fraction = *outside_sum / units;
        return ObjectiveOutput{
            .loss = metrics_.policy_loss,
            .gradient = std::move(eval->gradient),
        };
    }
};

struct PpoValueMetrics
{
    Scalar value_loss = 0;
    Scalar clip_fraction = 0;
};

class ClippedValueObjective final : public ContextualObjective
{
private:
    Scalar clip_epsilon_ = 0.2f;
    Scalar coefficient_ = 0.5f;
    PpoValueMetrics metrics_;

public:
    explicit ClippedValueObjective(
        Scalar clip_epsilon = 0.2f, Scalar coefficient = 0.5f,
        Precision precision = Precision::F32)
        : clip_epsilon_(clip_epsilon), coefficient_(coefficient)
    {
        PrecisionProfile p;
        p.stable = precision;
        set_precision_profile(p);
    }

    [[nodiscard]] const PpoValueMetrics& metrics() const noexcept
    {
        return metrics_;
    }

protected:
    [[nodiscard]] Result<ObjectiveOutput> evaluate(
        ComputeEngine& engine, const Tensor& value,
        const PolicyBatchView& batch) override
    {
        const Tensor* returns = batch.returns();
        const Tensor* old_value = batch.old_value();
        const Tensor* mask = batch.value_mask();
        if (returns == nullptr)
            NN_FAIL("ClippedValueObjective: 缺少 return");
        if (old_value == nullptr)
            NN_FAIL("ClippedValueObjective: 缺少 old_value");
        if (mask == nullptr)
            NN_FAIL("ClippedValueObjective: 缺少 value_mask");
        if (returns->rows() != value.rows() ||
            returns->cols() != value.cols() ||
            old_value->rows() != value.rows() ||
            old_value->cols() != value.cols() ||
            mask->rows() != value.rows() ||
            mask->cols() != value.cols())
            NN_FAIL("ClippedValueObjective: value/return/old_value/mask 形状不一致");

        NN_TRY(units_r,
               detail::rl_masked_units(engine, *mask, precision()));
        const Scalar units = *units_r;
        if (units <= Scalar{0})
        {
            metrics_ = {};
            NN_TRY(zero_grad,
                   detail::rl_zero(engine, value.rows(), value.cols(),
                                   precision()));
            return ObjectiveOutput{.loss = 0,
                                   .gradient = std::move(*zero_grad)};
        }

        NN_TRY(clip_low,
               dsl::compute(engine,
                            dsl::leaf(*old_value) -
                                dsl::rparam(clip_epsilon_),
                            value.rows(), value.cols(), precision()));
        NN_TRY(clip_high,
               dsl::compute(engine,
                            dsl::leaf(*old_value) +
                                dsl::rparam(clip_epsilon_),
                            value.rows(), value.cols(), precision()));
        NN_TRY_MSG(clipped, "PPO value clip",
                   dsl::compute(
                   engine,
                   dsl::select(
                       dsl::leaf(value) < dsl::leaf(*clip_low),
                       dsl::leaf(*clip_low),
                       dsl::select(
                           dsl::leaf(value) > dsl::leaf(*clip_high),
                           dsl::leaf(*clip_high), dsl::leaf(value))),
                   value.rows(), value.cols(), precision()));
        NN_TRY_MSG(diff, "PPO value diff",
                   dsl::compute(engine,
                                dsl::leaf(value) - dsl::leaf(*returns),
                                value.rows(), value.cols(), precision()));
        NN_TRY_MSG(clipped_diff, "PPO value clipped diff",
                   dsl::compute(engine,
                                dsl::leaf(*clipped) - dsl::leaf(*returns),
                                value.rows(), value.cols(), precision()));
        NN_TRY_MSG(diff_sq, "PPO value diff square",
                   dsl::compute(engine,
                                dsl::leaf(*diff) * dsl::leaf(*diff),
                                value.rows(), value.cols(), precision()));
        NN_TRY_MSG(clipped_sq, "PPO value clipped square",
                   dsl::compute(engine,
                                dsl::leaf(*clipped_diff) *
                                    dsl::leaf(*clipped_diff),
                                value.rows(), value.cols(), precision()));
        auto loss_expr = dsl::max(
            dsl::leaf(*diff_sq), dsl::leaf(*clipped_sq)) *
            dsl::leaf(*mask);
        NN_TRY_MSG(loss_terms, "PPO value max loss",
                   dsl::compute(engine, loss_expr, value.rows(),
                                value.cols(), precision()));
        NN_TRY_MSG(loss_sum, "PPO value reduce",
                   detail::rl_masked_sum(engine, *loss_terms, *mask,
                                         precision()));

        const auto outside_clip = dsl::select(
            dsl::leaf(value) < dsl::leaf(*clip_low), Scalar{1},
            dsl::select(dsl::leaf(value) > dsl::leaf(*clip_high),
                        Scalar{1}, Scalar{0}));
        NN_TRY_MSG(outside_tensor, "PPO value outside tensor",
                   dsl::compute(engine, outside_clip, value.rows(),
                                value.cols(), precision()));
        NN_TRY_MSG(clipped_grad_tensor, "PPO value clipped grad",
                   dsl::compute(engine,
                                dsl::select(
                                    dsl::leaf(*outside_tensor) > Scalar{0},
                                    Scalar{0},
                                    dsl::leaf(*diff) * dsl::rparam(2)),
                                value.rows(), value.cols(), precision()));
        NN_TRY_MSG(unclipped_grad_tensor, "PPO value unclipped grad",
                   dsl::compute(engine,
                                dsl::leaf(*diff) * dsl::rparam(2),
                                value.rows(), value.cols(), precision()));
        auto choose_clipped =
            dsl::leaf(*clipped_sq) > dsl::leaf(*diff_sq);
        auto grad_expr = dsl::select(
            choose_clipped, dsl::leaf(*clipped_grad_tensor),
            dsl::leaf(*unclipped_grad_tensor)) *
            dsl::leaf(*mask) *
            dsl::rparam(coefficient_ / units);
        NN_TRY_MSG(grad, "PPO value gradient",
                   dsl::compute(engine, grad_expr, value.rows(),
                                value.cols(), precision()));
        NN_TRY_MSG(clip_terms, "PPO value clip indicator",
                   dsl::compute(engine,
                                outside_clip * dsl::leaf(*mask),
                                value.rows(), value.cols(), precision()));
        NN_TRY_MSG(clip_sum, "PPO value clip reduce",
                   detail::rl_masked_sum(engine, *clip_terms, *mask,
                                         precision()));
        metrics_.value_loss = coefficient_ * *loss_sum / units;
        metrics_.clip_fraction = *clip_sum / units;
        return ObjectiveOutput{
            .loss = metrics_.value_loss,
            .gradient = std::move(*grad),
        };
    }
};

struct PpoConfig
{
    Scalar clip_epsilon = 0.2f;
    Scalar value_clip_epsilon = 0.2f;
    Scalar value_coefficient = 0.5f;
    Scalar entropy_coefficient = 0.01f;
    Precision precision = Precision::F32;
    std::string policy_field = "policy";
    std::string value_field = "value";
    std::shared_ptr<const PolicyDistribution> distribution;
};

struct PpoReport
{
    Scalar loss = 0;
    Scalar policy_loss = 0;
    Scalar value_loss = 0;
    Scalar entropy = 0;
    Scalar approx_kl = 0;
    Scalar policy_clip_fraction = 0;
    Scalar value_clip_fraction = 0;
    GradientBundle gradients;
};

class PpoObjective
{
private:
    PpoConfig cfg_;

public:
    explicit PpoObjective(PpoConfig cfg = {})
        : cfg_(std::move(cfg))
    {
    }

    [[nodiscard]] Result<PpoReport> evaluate(
        ComputeEngine& engine, const TensorSchema& output_schema,
        const Tensor& policy_logits, const Tensor& value,
        const PolicyBatchView& batch) const
    {
        NN_TRY_CHECK(output_schema.field(cfg_.policy_field));
        NN_TRY_CHECK(output_schema.field(cfg_.value_field));

        ClippedPolicyObjective policy(cfg_.clip_epsilon, cfg_.precision,
                                      cfg_.distribution);
        EntropyBonusObjective entropy(cfg_.entropy_coefficient,
                                      cfg_.precision, cfg_.distribution);
        ClippedValueObjective value_loss(
            cfg_.value_clip_epsilon, cfg_.value_coefficient,
            cfg_.precision);
        NN_TRY(policy_out,
               policy.evaluate_context(engine, policy_logits, batch));
        NN_TRY(entropy_out,
               entropy.evaluate_context(engine, policy_logits, batch));
        NN_TRY(value_out,
               value_loss.evaluate_context(engine, value, batch));

        NN_TRY(policy_grad,
               engine.clone(policy_out->gradient));
        NN_TRY_CHECK(engine.accumulate(*policy_grad,
                                       entropy_out->gradient));
        NN_TRY(gradients,
               GradientBundle::create(
                   output_schema,
                   {NamedTensor{.name = cfg_.policy_field,
                                .value = *policy_grad},
                    NamedTensor{.name = cfg_.value_field,
                                .value = value_out->gradient}}));
        const PpoPolicyMetrics& pm = policy.metrics();
        const PpoValueMetrics& vm = value_loss.metrics();
        const Scalar entropy_value =
            cfg_.entropy_coefficient == Scalar{0}
                ? Scalar{0}
                : -entropy_out->loss / cfg_.entropy_coefficient;
        return PpoReport{
            .loss = policy_out->loss + entropy_out->loss + value_out->loss,
            .policy_loss = policy_out->loss + entropy_out->loss,
            .value_loss = value_out->loss,
            .entropy = entropy_value,
            .approx_kl = pm.approx_kl,
            .policy_clip_fraction = pm.clip_fraction,
            .value_clip_fraction = vm.clip_fraction,
            .gradients = std::move(*gradients),
        };
    }
};

struct GrpoConfig
{
    Scalar clip_epsilon = 0.2f;
    Scalar entropy_coefficient = 0.01f;
    Scalar group_epsilon = 1e-8f;
    Precision precision = Precision::F32;
    std::string policy_field = "policy";
    std::shared_ptr<const PolicyDistribution> distribution;
};

struct GrpoReport
{
    Scalar loss = 0;
    Scalar policy_loss = 0;
    Scalar entropy = 0;
    GradientBundle gradients;
};

class GrpoObjective
{
private:
    GrpoConfig cfg_;

    [[nodiscard]] static Result<TensorRecord> with_advantage(
        const TensorRecord& source, const Tensor& advantage)
    {
        std::vector<NamedTensor> values;
        values.reserve(source.size());
        for (const TensorFieldSpec& field : source.schema().fields())
        {
            if (field.name == "advantage")
            {
                values.push_back(NamedTensor{
                    .name = field.name, .value = advantage});
                continue;
            }
            const Tensor* value = source.find(field.name);
            if (value != nullptr)
                values.push_back(NamedTensor{.name = field.name,
                                             .value = *value});
        }
        return TensorRecord::create(source.schema(), std::move(values));
    }

public:
    explicit GrpoObjective(GrpoConfig cfg = {})
        : cfg_(std::move(cfg))
    {
    }

    // advantage 已存在时直接使用；否则按 spans 对 returns 做组内标准化。
    [[nodiscard]] Result<GrpoReport> evaluate(
        ComputeEngine& engine, const TensorSchema& output_schema,
        const Tensor& policy_logits, const PolicyBatchView& batch,
        std::span<const GroupSpan> spans) const
    {
        NN_TRY_CHECK(output_schema.field(cfg_.policy_field));
        TensorRecord derived;
        const TensorRecord* active_record = &batch.record();
        if (batch.advantage() == nullptr)
        {
            const Tensor* returns = batch.returns();
            if (returns == nullptr)
                NN_FAIL("GrpoObjective: 缺少 advantage 或 return");

            Tensor flat_returns;
            const Tensor* group_input = returns;
            if (returns->cols() != 1)
            {
                NN_TRY(flat, engine.reshape(
                                *returns, returns->rows() * returns->cols(), 1));
                flat_returns = std::move(*flat);
                group_input = &flat_returns;
            }
            GroupAdvantage group(cfg_.group_epsilon, cfg_.precision);
            NN_TRY(grouped, group.apply(engine, *group_input, spans));
            Tensor advantage_flat = std::move(*grouped);
            Tensor advantage;
            if (advantage_flat.cols() == 1 && batch.action_mask().rows() == 1)
            {
                NN_TRY(a,
                       engine.reshape(advantage_flat, 1,
                                      advantage_flat.rows()));
                advantage = std::move(*a);
            }
            else
            {
                advantage = std::move(advantage_flat);
            }

            NN_TRY(record, with_advantage(batch.record(), advantage));
            derived = std::move(*record);
            active_record = &derived;
        }
        NN_TRY(active_view, PolicyBatchView::create(*active_record));

        ClippedPolicyObjective policy(cfg_.clip_epsilon, cfg_.precision,
                                      cfg_.distribution);
        EntropyBonusObjective entropy(cfg_.entropy_coefficient,
                                      cfg_.precision, cfg_.distribution);
        NN_TRY(policy_out,
               policy.evaluate_context(engine, policy_logits, *active_view));
        NN_TRY(entropy_out,
               entropy.evaluate_context(engine, policy_logits, *active_view));
        NN_TRY(policy_grad, engine.clone(policy_out->gradient));
        NN_TRY_CHECK(engine.accumulate(*policy_grad,
                                       entropy_out->gradient));
        NN_TRY(gradients,
               GradientBundle::create(
                   output_schema,
                   {NamedTensor{.name = cfg_.policy_field,
                                .value = *policy_grad}}));
        const Scalar entropy_value =
            cfg_.entropy_coefficient == Scalar{0}
                ? Scalar{0}
                : -entropy_out->loss / cfg_.entropy_coefficient;
        return GrpoReport{
            .loss = policy_out->loss + entropy_out->loss,
            .policy_loss = policy_out->loss,
            .entropy = entropy_value,
            .gradients = std::move(*gradients),
        };
    }
};

} // namespace nn
