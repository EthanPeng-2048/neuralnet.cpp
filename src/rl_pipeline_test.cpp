// ── rl_pipeline_test — rollout/buffer/sampler/metrics/checkpoint 编排 ─────
//
// 验收：
//   [1] collect → buffer → sample → update 顺序与 batch 数一致；
//   [2] source、sampler、metric sink、checkpoint provider 可替换；
//   [3] 固定 seed 与 checkpoint 恢复后的 rollout/sample 序列一致；
//   [4] hooks 顺序、buffer clear 策略、非法 sampler/config 错误。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

nn::TensorSchema rollout_schema()
{
    auto r = nn::TensorSchema::create(
        "rollout",
        {nn::tensor_field("x",
                          nn::TensorShapeSpec::exact(1, 1),
                          true, nn::Precision::F32,
                          nn::TensorPlacement::CPU)});
    NN_ASSERT_ALWAYS(r, "rollout schema");
    return std::move(*r);
}

class CountingSource final : public nn::RolloutSource
{
private:
    nn::CpuEngine& engine_;
    nn::TensorSchema schema_;

public:
    std::size_t calls = 0;

    explicit CountingSource(nn::CpuEngine& engine)
        : engine_(engine), schema_(rollout_schema())
    {
    }

    [[nodiscard]] nn::Result<nn::TensorRecord> rollout(
        nn::DeterministicRng& rng) override
    {
        ++calls;
        const nn::Scalar value =
            static_cast<nn::Scalar>(rng.next_u64() % 100000u);
        NN_TRY(tensor, nn::detail::upload_span(
                           engine_, 1, 1, nn::Precision::F32,
                           std::span<const nn::Scalar>(&value, 1)));
        return nn::TensorRecord::create(
            schema_,
            {nn::NamedTensor{.name = "x", .value = *tensor}});
    }
};

class BadSizeSampler final : public nn::BatchSampler
{
public:
    [[nodiscard]] nn::Result<std::vector<std::size_t>> sample(
        std::size_t, std::size_t, nn::DeterministicRng&) override
    {
        return std::vector<std::size_t>{};
    }
};

class OutOfRangeSampler final : public nn::BatchSampler
{
public:
    [[nodiscard]] nn::Result<std::vector<std::size_t>> sample(
        std::size_t, std::size_t batch_size,
        nn::DeterministicRng&) override
    {
        std::vector<std::size_t> indices(batch_size, 9999);
        return indices;
    }
};

struct RunTrace
{
    std::vector<float> batches;
    std::size_t before_hooks = 0;
    std::size_t after_hooks = 0;
};

nn::Result<void> run_pipeline(
    nn::RLTrainingPipeline& pipeline, std::size_t updates,
    nn::ComputeEngine& engine, RunTrace& trace)
{
    nn::TrainerHooks hooks;
    hooks.before_update =
        [&](std::span<const nn::TensorRecord* const> batch)
            -> nn::Result<void>
        {
            ++trace.before_hooks;
            if (batch.empty()) NN_FAIL("empty hook batch");
            return {};
        };
    hooks.after_update =
        [&](std::span<const nn::TensorRecord* const> batch)
            -> nn::Result<void>
        {
            ++trace.after_hooks;
            if (batch.empty()) NN_FAIL("empty hook batch");
            return {};
        };
    return pipeline.run(
        updates,
        [&](std::span<const nn::TensorRecord* const> batch,
            nn::DeterministicRng&, nn::MetricSink& metrics)
            -> nn::Result<void>
        {
            double sum = 0;
            for (const nn::TensorRecord* record : batch)
            {
                NN_TRY(value, record->require("x"));
                NN_TRY(v, nn::detail::download_vector(engine, **value));
                trace.batches.push_back((*v)[0]);
                sum += (*v)[0];
            }
            NN_TRY_CHECK(metrics.emit(nn::MetricRecord{
                .name = "batch_mean",
                .value = sum / static_cast<double>(batch.size()),
            }));
            return {};
        },
        std::move(hooks));
}

struct PipelineBundle
{
    CountingSource* source = nullptr;
    nn::RLTrainingPipeline pipeline;
};

nn::Result<PipelineBundle> make_pipeline(
    nn::CpuEngine& engine, std::shared_ptr<nn::MetricSink> metrics,
    std::shared_ptr<nn::CheckpointProvider> checkpoints = {},
    std::uint64_t seed = 177,
    bool clear_buffer = false)
{
    nn::RLPipelineConfig cfg;
    cfg.batch_size = 2;
    cfg.collect_per_update = 2;
    cfg.buffer_capacity = 32;
    cfg.min_buffer_size = 2;
    cfg.clear_buffer_after_update = clear_buffer;
    cfg.checkpoint_every = 3;
    auto source = std::make_unique<CountingSource>(engine);
    CountingSource* source_ptr = source.get();
    NN_TRY(pipeline,
           nn::RLTrainingPipeline::create(
               std::move(cfg), std::move(source),
               std::make_unique<nn::UniformBatchSampler>(),
               std::move(metrics), std::move(checkpoints), seed));
    return PipelineBundle{
        .source = source_ptr,
        .pipeline = std::move(*pipeline),
    };
}

void test_pipeline_and_metrics()
{
    auto engine = nn::CpuEngine::create();
    auto metrics = std::make_shared<nn::CollectingMetricSink>();
    auto bundle = make_pipeline(*engine, metrics, {}, 177, true);
    CHECK(bundle.has_value(), "pipeline composition");
    if (!bundle) return;
    RunTrace trace;
    CHECK(run_pipeline(bundle->pipeline, 4, *engine, trace).has_value(),
          "pipeline run");
    CHECK(bundle->source->calls == 8,
          "collect_per_update drives source call count");
    CHECK(trace.batches.size() == 8,
          "four updates x batch_size records reach task update");
    CHECK(trace.before_hooks == 4 && trace.after_hooks == 4,
          "hooks run once around each update");
    CHECK(metrics->records().size() == 4,
          "metrics sink receives one record per update");
    CHECK(bundle->pipeline.buffer().empty(),
          "clear_buffer_after_update policy");
    CHECK(bundle->pipeline.update_count() == 4 &&
              bundle->pipeline.rollout_cursor() == 8,
          "pipeline counters");
}

void test_checkpoint_reproducibility()
{
    auto engine = nn::CpuEngine::create();
    auto checkpoints = std::make_shared<nn::InMemoryCheckpointProvider>();
    auto base = make_pipeline(*engine, {}, checkpoints, 4242, false);
    CHECK(base.has_value(), "checkpoint base pipeline");
    if (!base) return;
    RunTrace first;
    CHECK(run_pipeline(base->pipeline, 3, *engine, first).has_value(),
          "base pipeline first segment");
    CHECK(base->pipeline.save_checkpoint().has_value(),
          "pipeline checkpoint save");

    auto resumed = make_pipeline(*engine, {}, checkpoints, 4242, false);
    CHECK(resumed.has_value(), "resumed pipeline composition");
    if (!resumed) return;
    RunTrace resume_prefix;
    CHECK(run_pipeline(resumed->pipeline, 3, *engine,
                       resume_prefix).has_value(),
          "resumed pipeline prefix");
    CHECK(resumed->pipeline.restore_checkpoint().has_value(),
          "pipeline checkpoint restore");
    RunTrace resumed_tail;
    CHECK(run_pipeline(resumed->pipeline, 2, *engine,
                       resumed_tail).has_value(),
          "resumed pipeline continuation");

    auto uninterrupted = make_pipeline(*engine, {}, {}, 4242, false);
    CHECK(uninterrupted.has_value(), "uninterrupted pipeline");
    if (!uninterrupted) return;
    RunTrace reference;
    CHECK(run_pipeline(uninterrupted->pipeline, 5, *engine,
                       reference).has_value(),
          "uninterrupted pipeline run");

    const std::size_t offset = reference.batches.size() - 4;
    bool same = resumed_tail.batches.size() == 4;
    for (std::size_t i = 0; same && i < 4; ++i)
        same = resumed_tail.batches[i] == reference.batches[offset + i];
    CHECK(same, "checkpoint restores RNG/rollout/sampler sequence");
    CHECK(resumed->pipeline.update_count() == 5 &&
              resumed->pipeline.rollout_cursor() == 10,
          "restored pipeline counters continue");

    NN_UNWRAP(checkpoints->save_section("custom", "opaque"));
    auto custom = checkpoints->load_section("custom");
    CHECK(custom && *custom == "opaque",
          "checkpoint provider supports custom sections");
}

void test_invalid_components()
{
    auto engine = nn::CpuEngine::create();
    nn::RLPipelineConfig cfg;
    cfg.batch_size = 2;
    cfg.min_buffer_size = 2;
    cfg.collect_per_update = 2;
    auto source = std::make_unique<CountingSource>(*engine);
    CHECK(!nn::RLTrainingPipeline::create(
               cfg, nullptr,
               std::make_unique<nn::UniformBatchSampler>()),
          "null source rejected");
    CHECK(!nn::RLTrainingPipeline::create(
               cfg, std::move(source), nullptr),
          "null sampler rejected");

    auto bad_size = nn::RLTrainingPipeline::create(
        cfg, std::make_unique<CountingSource>(*engine),
        std::make_unique<BadSizeSampler>());
    CHECK(bad_size.has_value(), "bad-size sampler pipeline builds");
    if (bad_size)
    {
        RunTrace trace;
        CHECK(!run_pipeline(*bad_size, 1, *engine, trace),
              "sampler returning wrong batch size rejected");
    }

    auto out_of_range = nn::RLTrainingPipeline::create(
        cfg, std::make_unique<CountingSource>(*engine),
        std::make_unique<OutOfRangeSampler>());
    CHECK(out_of_range.has_value(), "out-of-range sampler pipeline builds");
    if (out_of_range)
    {
        RunTrace trace;
        CHECK(!run_pipeline(*out_of_range, 1, *engine, trace),
              "sampler out-of-range index rejected");
    }
}

} // namespace

int main()
{
    std::puts("=== RL-2b composable pipeline ===");
    test_pipeline_and_metrics();
    test_checkpoint_reproducibility();
    test_invalid_components();
    if (g_fail == 0) std::puts("All RL pipeline tests passed.");
    else std::printf("%d RL pipeline test(s) failed.\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
