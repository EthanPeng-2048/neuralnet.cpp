#pragma once

// ── rl_pipeline.hpp — 可替换 RL 编排组件（docs/internal/31 RL-2b）─────────────
//
// 固定顺序：rollout source → experience buffer → batch sampler → task update。
// 环境、自对局、规则、算法细节与 replay policy 均留给消费者。pipeline 只
// 保证 seed/update 顺序、batch 索引校验、metrics/checkpoint 扩展点。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <neuralnet.cpp/model_keyvalue_record.hpp>
#include "rl_experience.hpp"

namespace nn
{

class DeterministicRng
{
private:
    std::uint64_t seed_ = 0;
    std::mt19937_64 engine_;

public:
    explicit DeterministicRng(std::uint64_t seed = 0) : seed_(seed), engine_(seed) {}

    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }
    [[nodiscard]] std::uint64_t next_u64() { return engine_(); }
    [[nodiscard]] std::size_t uniform_index(std::size_t upper_exclusive)
    {
        if (upper_exclusive == 0) NN_UNWRAP(false, "DeterministicRng: uniform_index 上界必须 > 0");
        std::uniform_int_distribution<std::size_t> dist(0, upper_exclusive - 1);
        return dist(engine_);
    }

    [[nodiscard]] std::string save_state() const
    {
        std::ostringstream out;
        out << "RNGS1 " << seed_ << ' ' << engine_;
        return out.str();
    }

    [[nodiscard]] Result<void> restore_state(std::string_view state)
    {
        std::istringstream in{std::string(state)};
        std::string tag;
        std::uint64_t saved_seed = 0;
        in >> tag >> saved_seed;
        if (!in || tag != "RNGS1") NN_FAIL("DeterministicRng::restore_state: 状态头非法");
        in >> engine_;
        if (!in) NN_FAIL("DeterministicRng::restore_state: 状态不完整");
        seed_ = saved_seed;
        return {};
    }
};

class RolloutSource
{
public:
    virtual ~RolloutSource() = default;
    virtual Result<TensorRecord> rollout(DeterministicRng& rng) = 0;
};

class ExperienceBuffer
{
private:
    std::vector<TensorRecord> records_;
    std::size_t capacity_ = 0;

public:
    explicit ExperienceBuffer(std::size_t capacity = 0) : capacity_(capacity) {}

    [[nodiscard]] Result<void> add(TensorRecord record)
    {
        if (capacity_ > 0 && records_.size() >= capacity_) { records_.erase(records_.begin()); }
        records_.push_back(std::move(record));
        return {};
    }

    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    void clear() noexcept { records_.clear(); }

    [[nodiscard]] Result<TensorRecord*> at(std::size_t index)
    {
        if (index >= records_.size()) NN_FAIL("ExperienceBuffer::at: 下标越界");
        return &records_[index];
    }

    [[nodiscard]] Result<const TensorRecord*> at(std::size_t index) const
    {
        if (index >= records_.size()) NN_FAIL("ExperienceBuffer::at: 下标越界");
        return &records_[index];
    }
};

class BatchSampler
{
public:
    virtual ~BatchSampler() = default;
    virtual Result<std::vector<std::size_t>> sample(std::size_t available, std::size_t batch_size,
                                                    DeterministicRng& rng) = 0;
};

class UniformBatchSampler final : public BatchSampler
{
private:
    bool replacement_ = false;

public:
    explicit UniformBatchSampler(bool replacement = false) : replacement_(replacement) {}

    [[nodiscard]] Result<std::vector<std::size_t>>
    sample(std::size_t available, std::size_t batch_size, DeterministicRng& rng) override
    {
        if (batch_size == 0) NN_FAIL("UniformBatchSampler: batch_size 必须 > 0");
        if (available == 0) NN_FAIL("UniformBatchSampler: buffer 为空");
        if (!replacement_ && batch_size > available)
            NN_FAIL("UniformBatchSampler: 无放回采样时 batch_size > buffer size");

        std::vector<std::size_t> pool(available);
        for (std::size_t i = 0; i < available; ++i)
            pool[i] = i;
        std::vector<std::size_t> out;
        out.reserve(batch_size);
        if (replacement_)
        {
            for (std::size_t i = 0; i < batch_size; ++i)
                out.push_back(rng.uniform_index(available));
            return out;
        }
        for (std::size_t i = 0; i < batch_size; ++i)
        {
            const std::size_t pick = i + rng.uniform_index(available - i);
            std::swap(pool[i], pool[pick]);
            out.push_back(pool[i]);
        }
        return out;
    }
};

struct MetricRecord
{
    std::string name;
    double value = 0;
};

class MetricSink
{
public:
    virtual ~MetricSink() = default;
    virtual Result<void> emit(const MetricRecord& metric) = 0;
};

class NullMetricSink final : public MetricSink
{
public:
    [[nodiscard]] Result<void> emit(const MetricRecord&) override { return {}; }
};

class CollectingMetricSink final : public MetricSink
{
private:
    std::vector<MetricRecord> records_;

public:
    [[nodiscard]] Result<void> emit(const MetricRecord& metric) override
    {
        records_.push_back(metric);
        return {};
    }

    [[nodiscard]] const std::vector<MetricRecord>& records() const noexcept { return records_; }
    void clear() noexcept { records_.clear(); }
};

class CheckpointProvider
{
public:
    virtual ~CheckpointProvider() = default;
    virtual Result<void> save_section(std::string_view name, std::string_view bytes) = 0;
    virtual Result<std::string> load_section(std::string_view name) const = 0;
};

class InMemoryCheckpointProvider final : public CheckpointProvider
{
private:
    std::vector<std::pair<std::string, std::string>> sections_;

public:
    [[nodiscard]] Result<void> save_section(std::string_view name, std::string_view bytes) override
    {
        if (name.empty()) NN_FAIL("InMemoryCheckpointProvider: section 名不能为空");
        for (auto& section : sections_)
        {
            if (section.first == name)
            {
                section.second.assign(bytes);
                return {};
            }
        }
        sections_.emplace_back(std::string(name), std::string(bytes));
        return {};
    }

    [[nodiscard]] Result<std::string> load_section(std::string_view name) const override
    {
        for (const auto& section : sections_)
            if (section.first == name) return section.second;
        NN_FAIL("InMemoryCheckpointProvider: 缺少 section '" + std::string(name) + "'");
    }
};

struct TrainerHooks
{
    std::function<Result<void>(std::span<const TensorRecord* const>)> before_update;
    std::function<Result<void>(std::span<const TensorRecord* const>)> after_update;
};

struct RLPipelineConfig
{
    std::size_t batch_size = 1;
    std::size_t collect_per_update = 1;
    std::size_t buffer_capacity = 0;
    std::size_t min_buffer_size = 1;
    bool clear_buffer_after_update = false;
    std::size_t checkpoint_every = 0;
    std::string checkpoint_section = "rl_pipeline";
};

class RLTrainingPipeline
{
private:
    RLPipelineConfig cfg_;
    std::unique_ptr<RolloutSource> source_;
    std::unique_ptr<BatchSampler> sampler_;
    std::shared_ptr<MetricSink> metrics_;
    std::shared_ptr<CheckpointProvider> checkpoints_;
    ExperienceBuffer buffer_;
    DeterministicRng rng_;
    std::uint64_t update_count_ = 0;
    std::uint64_t rollout_cursor_ = 0;

    RLTrainingPipeline(RLPipelineConfig cfg, std::unique_ptr<RolloutSource> source,
                       std::unique_ptr<BatchSampler> sampler, std::shared_ptr<MetricSink> metrics,
                       std::shared_ptr<CheckpointProvider> checkpoints, std::uint64_t seed)
        : cfg_(std::move(cfg)), source_(std::move(source)), sampler_(std::move(sampler)),
          metrics_(std::move(metrics)), checkpoints_(std::move(checkpoints)),
          buffer_(cfg_.buffer_capacity), rng_(seed)
    {
    }

    [[nodiscard]] Result<void> save_pipeline_state_()
    {
        KeyValueRecord state;
        state.set("pipeline_version", 1u);
        state.set("update_count", update_count_);
        state.set("rollout_cursor", rollout_cursor_);
        state.set("rng_state", rng_.save_state());
        return checkpoints_->save_section(cfg_.checkpoint_section, state.serialize());
    }

public:
    [[nodiscard]] static Result<RLTrainingPipeline>
    create(RLPipelineConfig cfg, std::unique_ptr<RolloutSource> source,
           std::unique_ptr<BatchSampler> sampler, std::shared_ptr<MetricSink> metrics = {},
           std::shared_ptr<CheckpointProvider> checkpoints = {}, std::uint64_t seed = 0)
    {
        if (!source || !sampler) NN_FAIL("RLTrainingPipeline::create: source/sampler 不能为空");
        if (cfg.batch_size == 0 || cfg.collect_per_update == 0 || cfg.min_buffer_size == 0)
            NN_FAIL("RLTrainingPipeline::create: batch/collect/min_buffer 必须 > 0");
        if (cfg.min_buffer_size < cfg.batch_size)
            NN_FAIL("RLTrainingPipeline::create: min_buffer_size 不能小于 batch_size");
        if (cfg.buffer_capacity > 0 && cfg.min_buffer_size > cfg.buffer_capacity)
            NN_FAIL("RLTrainingPipeline::create: min_buffer_size 不能超过"
                    " buffer_capacity");
        if (cfg.checkpoint_section.empty())
            NN_FAIL("RLTrainingPipeline::create: checkpoint_section 不能为空");
        if (!metrics) metrics = std::make_shared<NullMetricSink>();
        if (!checkpoints) checkpoints = std::make_shared<InMemoryCheckpointProvider>();
        return RLTrainingPipeline(std::move(cfg), std::move(source), std::move(sampler),
                                  std::move(metrics), std::move(checkpoints), seed);
    }

    RLTrainingPipeline(RLTrainingPipeline&&) noexcept = default;
    RLTrainingPipeline& operator=(RLTrainingPipeline&&) noexcept = default;
    RLTrainingPipeline(const RLTrainingPipeline&) = delete;
    RLTrainingPipeline& operator=(const RLTrainingPipeline&) = delete;
    ~RLTrainingPipeline() = default;

    [[nodiscard]] const RLPipelineConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] const ExperienceBuffer& buffer() const noexcept { return buffer_; }
    [[nodiscard]] std::uint64_t update_count() const noexcept { return update_count_; }
    [[nodiscard]] std::uint64_t rollout_cursor() const noexcept { return rollout_cursor_; }
    void set_rollout_cursor(std::uint64_t cursor) noexcept { rollout_cursor_ = cursor; }
    [[nodiscard]] DeterministicRng& rng() noexcept { return rng_; }
    [[nodiscard]] MetricSink& metrics() noexcept { return *metrics_; }
    [[nodiscard]] CheckpointProvider& checkpoints() noexcept { return *checkpoints_; }

    template <class UpdateFn>
    [[nodiscard]] Result<void> run(std::size_t updates, UpdateFn&& update, TrainerHooks hooks = {})
    {
        for (std::size_t i = 0; i < updates; ++i)
        {
            for (std::size_t c = 0; c < cfg_.collect_per_update; ++c)
            {
                NN_TRY(record, source_->rollout(rng_));
                NN_TRY_CHECK(buffer_.add(std::move(*record)));
                ++rollout_cursor_;
            }
            if (buffer_.size() < cfg_.min_buffer_size)
                NN_FAIL("RLTrainingPipeline: buffer 未达到 min_buffer_size");
            NN_TRY(indices, sampler_->sample(buffer_.size(), cfg_.batch_size, rng_));
            if (indices->size() != cfg_.batch_size)
                NN_FAIL("RLTrainingPipeline: sampler 返回 batch size 不符");

            std::vector<const TensorRecord*> selected;
            selected.reserve(indices->size());
            for (std::size_t index : *indices)
            {
                NN_TRY(record, buffer_.at(index));
                selected.push_back(*record);
            }
            if (hooks.before_update) NN_TRY_CHECK(hooks.before_update(selected));
            NN_TRY_CHECK(update(selected, rng_, *metrics_));

            // update 已成功修改外部模型；即使后置 hook 失败，也必须提交
            // pipeline 自身计数，避免调用方重试时重复执行同一次 update。
            ++update_count_;
            Result<void> after_r{};
            if (hooks.after_update) after_r = hooks.after_update(selected);
            if (cfg_.clear_buffer_after_update) buffer_.clear();
            NN_TRY_CHECK(after_r);
            if (cfg_.checkpoint_every > 0 && update_count_ % cfg_.checkpoint_every == 0)
                NN_TRY_CHECK(save_pipeline_state_());
        }
        return {};
    }

    [[nodiscard]] Result<void> save_checkpoint() { return save_pipeline_state_(); }

    [[nodiscard]] Result<void> restore_checkpoint()
    {
        NN_TRY(bytes, checkpoints_->load_section(cfg_.checkpoint_section));
        NN_TRY(kv, KeyValueRecord::parse(*bytes));
        std::uint64_t version = 0;
        if (!kv->get("pipeline_version", version) || version != 1)
            NN_FAIL("RLTrainingPipeline::restore_checkpoint: 版本不兼容");
        std::uint64_t updates = 0, cursor = 0;
        std::string rng_state;
        if (!kv->get("update_count", updates) || !kv->get("rollout_cursor", cursor) ||
            !kv->get("rng_state", rng_state))
            NN_FAIL("RLTrainingPipeline::restore_checkpoint: 状态字段缺失");
        NN_TRY_CHECK(rng_.restore_state(rng_state));
        update_count_ = updates;
        rollout_cursor_ = cursor;
        return {};
    }
};

} // namespace nn
