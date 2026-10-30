#pragma once

// ── batch_source.hpp — 领域无关数据供给契约（RFC 32 §4；issue #39）────────
//
// 训练骨架的"可变"侧：域只回答"第 step_hint 个 batch 是什么"，epoch 网格 /
// lr 接线 / 断点节奏由 RunDriver 保证（"不变"侧）。
//
// 两层设计（自由 + 复用）：
//   · **Pull 契约（`BatchSource`）**：分层最深、最自由。RL/在线学习等无固定
//     epoch 的域直接实现它（`steps_per_epoch()` 返回 nullopt ⇒ 骨架退化为
//     步进驱动，lr 走 step 级，域自行决定何时算一轮）；
//   · **预置适配器（`IndexedBatchSource`）**：样本表 + 每 epoch shuffle + 按
//     下标装配。文本 / 视觉等"固定 epoch 网格"域复用它，不重写顺序逻辑。
//
// 载荷：`BatchAt::payload` 是 `std::any`（域自解释；文本 = `TextSampleBatch`）。
// 只用**指针形式** `std::any_cast<T>(&payload)` 读取（返回 nullptr，不抛异常；
// -fno-exceptions 下安全）。骨架不解释载荷内容，只透传给域自己的 StepFn。
//
// 与 TrainingSession（docs/todo/30）的分工：本头只管"取样顺序 + 装配"，
// 不碰梯度 / 归一化 / 裁剪 —— 后者仍是 Session 的职责。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_errors.hpp>   // nn::Result / NN_FAIL / NN_TRY

namespace nn
{

// ── 一个训练步的产物（领域无关；元数据 + 域自解释载荷）────────────────────
struct BatchAt
{
    std::size_t samples = 0;   // 本 batch 有效样本数（末批可能不满）
    std::size_t units = 0;     // 归一化单位（文本 = 有效 token；视觉 = 样本数）
    bool epoch_done = true;    // 本 batch 后是否结束当前 epoch
    std::any payload;          // 域自解释载荷（文本 = TextSampleBatch）
};

// ── Pull 契约（最自由；无固定 epoch 的域直接实现）─────────────────────────
class BatchSource
{
public:
    virtual ~BatchSource() = default;

    // 供给第 step_hint 个 batch（域自己决定来源：数据表 / 在线采集 / rollout）。
    // step_hint 从 0 开始，在一个 epoch（或一轮）内单调推进。
    [[nodiscard]] virtual Result<BatchAt> next(std::size_t step_hint) = 0;

    // 每 epoch 步数；nullopt = 无固定 epoch（骨架退化为步进驱动）。
    [[nodiscard]] virtual std::optional<std::size_t> steps_per_epoch() const
    {
        return std::nullopt;
    }

    // 断点定位：把内部游标移到 (epoch, start_step)。无固定 epoch 的域用空实现。
    // 有固定 epoch 的域在此重排样本顺序（每 epoch shuffle）。
    [[nodiscard]] virtual Result<void> rewind(std::size_t /*epoch*/,
                                              std::size_t /*start_step*/)
    {
        return {};
    }
};

// ── 预置适配器：样本表 + 每 epoch shuffle + 按下标装配（文本/视觉复用）────
//
// 语义与 TextTrainer::fit 的取样顺序一致（docs/todo/30 可观测行为锚）：
//   · 步网格 = ceil(sample_count / batch_size)；
//   · 每 epoch 重排样本顺序（epoch 作种子 ⇒ 同 seed 跨续训可复现）；
//   · 末批不满（samples < batch_size）；
//   · 归一化单位由域装配回调给出（文本 = 有效 token 总数）。
class IndexedBatchSource : public BatchSource
{
public:
    // 域装配回调：给定本 batch 的样本下标 → 载荷 + 归一化单位。
    // 返回错误 = 装配失败（骨架向上冒泡，训练中止）。
    using AssembleFn = std::function<Result<std::pair<std::any, std::size_t>>(
        std::span<const std::size_t> indices)>;

    // 工厂（参数校验；构造函数私有 —— 与 TrainingSession/TextTrainer 同口径）
    [[nodiscard]] static Result<IndexedBatchSource> create(std::size_t sample_count,
                                                           std::size_t batch_size,
                                                           std::uint64_t seed,
                                                           AssembleFn assemble)
    {
        if (batch_size == 0)
            NN_FAIL("IndexedBatchSource::create: batch_size 必须 > 0");
        if (sample_count == 0)
            NN_FAIL("IndexedBatchSource::create: sample_count 必须 > 0（训练集为空）");
        if (!assemble)
            NN_FAIL("IndexedBatchSource::create: 装配回调不能为空");
        return IndexedBatchSource(sample_count, batch_size, seed, std::move(assemble));
    }

    // ── BatchSource ────────────────────────────────────────────────────
    [[nodiscard]] Result<BatchAt> next(std::size_t step_hint) override
    {
        const std::size_t steps = *steps_per_epoch();
        if (step_hint >= steps)
            NN_FAIL("IndexedBatchSource::next: step_hint " + std::to_string(step_hint) +
                    " 超出本 epoch 步数 " + std::to_string(steps));

        const std::size_t offset = step_hint * batch_size_;
        const std::size_t n = std::min(batch_size_, sample_count_ - offset);
        const std::span<const std::size_t> indices(order_.data() + offset, n);

        NN_TRY(assembled, assemble_(indices));
        BatchAt out;
        out.samples = n;
        out.units = assembled->second;
        out.epoch_done = (step_hint + 1 >= steps);
        out.payload = std::move((*assembled).first);
        return out;
    }

    [[nodiscard]] std::optional<std::size_t> steps_per_epoch() const override
    {
        return (sample_count_ + batch_size_ - 1) / batch_size_;
    }

    [[nodiscard]] Result<void> rewind(std::size_t epoch, std::size_t start_step) override
    {
        const std::size_t steps = *steps_per_epoch();
        if (start_step > steps)
            NN_FAIL("IndexedBatchSource::rewind: start_step " + std::to_string(start_step) +
                    " 超出每 epoch 步数 " + std::to_string(steps));
        // 单 RNG 流语义：从恒等序起，用同一 RNG 依次重放 0..epoch 共 epoch+1 次
        // shuffle —— 与"每 epoch 顺序推进同一 RNG 流"的取样语义一致，同时
        // (seed, epoch) 唯一决定顺序 ⇒ 幂等（重复 rewind 同一 epoch 结果不变）。
        for (std::size_t i = 0; i < sample_count_; ++i) order_[i] = i;
        std::mt19937_64 rng(seed_);
        for (std::size_t e = 0; e <= epoch; ++e)
            std::shuffle(order_.begin(), order_.end(), rng);
        epoch_ = epoch;
        start_step_ = start_step;
        return {};
    }

    // ── 诊断面 ────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t sample_count() const noexcept { return sample_count_; }
    [[nodiscard]] std::size_t batch_size() const noexcept { return batch_size_; }
    [[nodiscard]] std::size_t current_epoch() const noexcept { return epoch_; }
    // 当前 epoch 的样本顺序（测试 / 诊断用）
    [[nodiscard]] std::span<const std::size_t> order() const noexcept { return order_; }

private:
    IndexedBatchSource(std::size_t sample_count, std::size_t batch_size,
                       std::uint64_t seed, AssembleFn assemble)
        : sample_count_(sample_count), batch_size_(batch_size), seed_(seed),
          assemble_(std::move(assemble))
    {
        order_.resize(sample_count_);
        for (std::size_t i = 0; i < sample_count_; ++i) order_[i] = i;
    }

    std::size_t sample_count_ = 0;
    std::size_t batch_size_ = 0;
    std::uint64_t seed_ = 0;
    AssembleFn assemble_;
    std::vector<std::size_t> order_;   // 当前 epoch 的样本顺序
    std::size_t epoch_ = 0;
    std::size_t start_step_ = 0;
};

} // namespace nn
