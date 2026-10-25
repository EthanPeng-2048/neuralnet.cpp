// ── dataset_samples.hpp — 训练样本视图：Dataset → 样本（窗口）→ batch ───────
//
// P2b 的"样本概念"（docs/todo/29 §8、docs/todo/19 §0 裁决 #2/#3）：
//   · **存储单位 = doc**（`.nndataset` 里的一行文本）；
//   · **训练单位 = sample**（一个 seq_len 的滑动窗口，可能跨 doc 边界）；
//   · **step = ceil(sample_count / batch_size)** —— 派生量，不是数据集概念。
//
// 本头把"拼流 → 切窗 → 填充/PAD → 移位目标 → 掩码 → doc_ids → batch-major
// 扁平化"全部收在一处，训练驱动（TextTrainer）与测试共用同一实现，避免各入口
// 各抄一遍数据装配（旧 text_train 的 window_offsets + 手工填充块）。
//
// 逐位约定（与旧 text_train 逐字节一致，不得改）：
//   · 窗口起点 = 0, stride, 2·stride, …（stride = 0 时等于 seq_len，不重叠）；
//   · 样本内 x[t] = token_flow[offset+t]（t < valid_len），否则 pad_id；
//     目标 y[t] = token_flow[offset+t+1]（t+1 < valid_len），否则 pad_id；
//   · 位置参与 loss ⇔ t+1 < valid_len 且生成期 loss_mask[offset+t+1] ≠ 0；
//   · doc_ids 的 PAD 位置填 0（0 = "未知/填充"，块对角掩码据此隔离）；
//   · batch 内列序 = 样本下标顺序；扁平数组是 **batch-major**：i = b·seq_len + t
//     （与 GPT logits 列序一致；position-major 只用于宿主矩阵的行列寻址）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "algebra_matrix.hpp"   // nn::Matrix（I/O 层允许；窗口装配是数据 I/O）
#include <neuralnet.cpp/core_assert.hpp>      // NN_ASSERT_ALWAYS（样本下标前置条件）
#include <neuralnet.cpp/core_config.hpp>      // nn::Scalar
#include <neuralnet.cpp/core_errors.hpp>      // nn::Result / NN_FAIL
#include <neuralnet.cpp/dataset.hpp>          // nn::Dataset / DatasetSubset / TextCorpus

namespace nn
{

// 窗口化策略（训练策略，留在数据装配层；不进入 `.nndataset` 文件格式）。
struct TextWindowConfig
{
    std::size_t seq_len = 0;   // 每个样本的 token 数（= 模型序列长度）
    std::size_t stride = 0;    // 窗口步长；0 = seq_len（不重叠）
    std::size_t pad_id = 0;    // 末窗不足 seq_len 时的填充 token id
};

// 一个训练样本 = 一个窗口（**视图**，不拥有语料；TextSampleSet 存活期内有效）。
// `loss_mask[i]` 是该 token 作为**目标位置**是否参与 loss 的最终样本语义：
// TextSampleSet::make_batch 把它固化到 batch.flat_mask；Session 不解释 mask。
struct TextSample
{
    std::size_t index = 0;      // 样本序号（0-based）
    std::size_t offset = 0;     // 在语料 token_flow 中的起点
    std::size_t valid_len = 0;  // 有效 token 数（<= seq_len；末窗可能更短）
    std::span<const std::size_t> tokens;         // 长度 = valid_len
    std::span<const std::size_t> doc_ids;        // 长度 = valid_len
    std::span<const unsigned char> loss_mask;    // 长度 = valid_len
};

// 一个 batch 的宿主侧装配结果（`engine.from_matrix(input)` 由训练驱动完成）。
// 复用同一对象做逐 step 填充，避免每步分配。
struct TextSampleBatch
{
    Matrix input;                       // (seq_len, batch)：每列一个样本的输入
    std::vector<std::size_t> flat_targets;  // batch-major：i = b·seq_len + t
    std::vector<Scalar> flat_mask;      // batch-major：1 = 该位置参与 loss
    std::vector<std::size_t> doc_ids;   // batch-major：0 = PAD/未知
    std::size_t batch = 0;
    std::size_t seq_len = 0;
    std::size_t valid_tokens = 0;       // Σ 参与 loss 的位置数（梯度归一化用）
};

// ══════════════════════════════════════════════════════════════════════════
//  TextSampleSet — 一个子集的窗口样本集合（只读；数据集样本概念的载体）
// ══════════════════════════════════════════════════════════════════════════
class TextSampleSet
{
public:
    // 从数据集子集切窗。失败语义：seq_len=0 / 无 test 子集 → Result 错误；
    // 空子集允许存在（sample_count=0，训练集为空由 TextTrainer fail-fast，
    // 测试集为空 = 跳过评估，与旧 CLI 语义一致）。
    [[nodiscard]] static Result<TextSampleSet> create(const Dataset& dataset, DatasetSubset subset,
                                                      TextWindowConfig cfg)
    {
        if (cfg.seq_len == 0) NN_FAIL("TextSampleSet::create: seq_len 必须 > 0（窗口 = 训练样本）");
        if (dataset.info().kind != DatasetKind::Text)
            NN_FAIL("TextSampleSet::create: 只支持 text 数据集");

        TextSampleSet out;
        out.cfg_ = cfg;
        out.info_ = dataset.info();
        out.subset_ = subset;
        out.path_ = dataset.path();
        out.vocab_ = dataset.vocab_kvrec();
        NN_TRY(corpus_r, dataset.load_text(subset));
        out.corpus_ = std::move(*corpus_r);
        if (out.corpus_.doc_ids.size() != out.corpus_.token_flow.size() ||
            out.corpus_.loss_mask.size() != out.corpus_.token_flow.size())
            NN_FAIL("TextSampleSet::create: 语料三元组长度不一致（内部错误）");

        out.stride_ = (cfg.stride == 0) ? cfg.seq_len : cfg.stride;
        out.offsets_.reserve(out.corpus_.token_flow.size() / out.stride_ + 1);
        for (std::size_t pos = 0; pos < out.corpus_.token_flow.size(); pos += out.stride_)
            out.offsets_.push_back(pos);
        return std::move(out);   // TextSampleSet 不可拷贝（只移动）
    }

    // 持有整段语料（P2-4 流式之前）：禁止拷贝，只允许移动（一次误拷贝 = 整份语料复制）。
    TextSampleSet() = default;
    TextSampleSet(TextSampleSet&&) noexcept = default;
    TextSampleSet& operator=(TextSampleSet&&) noexcept = default;
    TextSampleSet(const TextSampleSet&) = delete;
    TextSampleSet& operator=(const TextSampleSet&) = delete;
    ~TextSampleSet() = default;

    // ── 样本概念 ────────────────────────────────────────────────────────
    [[nodiscard]] std::size_t sample_count() const noexcept { return offsets_.size(); }

    // optimizer update 数 = ceil(样本数 / global_batch_size)（派生量）。
    [[nodiscard]] std::size_t steps_per_epoch(std::size_t global_batch_size) const noexcept
    {
        if (global_batch_size == 0) return 0;
        return (offsets_.size() + global_batch_size - 1) / global_batch_size;
    }

    [[nodiscard]] TextSample sample(std::size_t index) const
    {
        NN_ASSERT_ALWAYS(index < offsets_.size(), "TextSampleSet::sample: 样本下标越界");
        const std::size_t offset = offsets_[index];
        const std::size_t valid_len = std::min(cfg_.seq_len, corpus_.token_flow.size() - offset);
        TextSample s;
        s.index = index;
        s.offset = offset;
        s.valid_len = valid_len;
        s.tokens = std::span<const std::size_t>(corpus_.token_flow.data() + offset, valid_len);
        s.doc_ids = std::span<const std::size_t>(corpus_.doc_ids.data() + offset, valid_len);
        s.loss_mask = std::span<const unsigned char>(corpus_.loss_mask.data() + offset, valid_len);
        return s;
    }

    // 按样本下标装配一个 batch（indices 长度 = 本批实际 batch；末批可不满）。
    // 输出矩阵/数组按需 resize（复用内存）；返回本批参与 loss 的位置数。
    [[nodiscard]] Result<std::size_t> make_batch(std::span<const std::size_t> indices,
                                                 TextSampleBatch& out) const
    {
        const std::size_t seq = cfg_.seq_len;
        const std::size_t bs = indices.size();
        if (out.input.rows() != seq || out.input.cols() != bs)
        {
            out.input.resize(seq, bs);
            out.seq_len = seq;
            out.batch = bs;
        }
        // 扁平数组按 seq*batch 需求 resize（与矩阵实际形状同源，防"缓存 flags 与
        // 矩阵实际形状不一致时跳过 resize"的越界写）
        if (out.flat_targets.size() != seq * bs)
        {
            out.flat_targets.resize(seq * bs);
            out.flat_mask.resize(seq * bs);
            out.doc_ids.resize(seq * bs);
        }

        std::size_t valid = 0;
        for (std::size_t b = 0; b < bs; ++b)
        {
            const std::size_t idx = indices[b];
            if (idx >= offsets_.size())
                NN_FAIL("TextSampleSet::make_batch: 样本下标越界（" + std::to_string(idx) +
                        " >= " + std::to_string(offsets_.size()) + "）");
            const std::size_t offset = offsets_[idx];
            const std::size_t valid_len = std::min(seq, corpus_.token_flow.size() - offset);
            const std::size_t* toks = corpus_.token_flow.data() + offset;
            const std::size_t* docs = corpus_.doc_ids.data() + offset;
            const unsigned char* mask = corpus_.loss_mask.data() + offset;

            for (std::size_t t = 0; t < seq; ++t)
            {
                const bool has_x = (t < valid_len);
                const bool has_y = (t + 1 < valid_len);
                const std::size_t x = has_x ? toks[t] : cfg_.pad_id;
                const std::size_t y = has_y ? toks[t + 1] : cfg_.pad_id;
                const bool participate = has_y && (mask[t + 1] != 0);

                out.input.set_value_unchecked(t, b, static_cast<Scalar>(x));
                const std::size_t flat = b * seq + t;   // batch-major
                out.flat_targets[flat] = y;
                out.flat_mask[flat] = participate ? Scalar{1} : Scalar{0};
                out.doc_ids[flat] = has_x ? docs[t] : 0;
                if (participate) ++valid;
            }
        }
        out.valid_tokens = valid;
        return valid;
    }

    // 只统计目标位置的 loss 有效数，不物化 batch。
    // Global-batch 训练需要先知道整个 update 的归一化单位，再拆分 micro-batch。
    [[nodiscard]] Result<std::size_t>
    valid_target_count(std::span<const std::size_t> indices) const
    {
        std::size_t valid = 0;
        for (std::size_t idx : indices)
        {
            if (idx >= offsets_.size())
                NN_FAIL("TextSampleSet::valid_target_count: 样本下标越界（" +
                        std::to_string(idx) + " >= " +
                        std::to_string(offsets_.size()) + "）");
            const std::size_t offset = offsets_[idx];
            const std::size_t valid_len =
                std::min(cfg_.seq_len, corpus_.token_flow.size() - offset);
            for (std::size_t t = 0; t + 1 < valid_len; ++t)
                if (corpus_.loss_mask[offset + t + 1] != 0) ++valid;
        }
        return valid;
    }

    // ── 元数据访问（打印/身份/诊断）──────────────────────────────────────
    [[nodiscard]] DatasetSubset subset() const noexcept { return subset_; }
    [[nodiscard]] const TextWindowConfig& window() const noexcept { return cfg_; }
    [[nodiscard]] std::size_t seq_len() const noexcept { return cfg_.seq_len; }
    [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
    [[nodiscard]] std::size_t pad_id() const noexcept { return cfg_.pad_id; }
    [[nodiscard]] std::size_t token_count() const noexcept { return corpus_.token_flow.size(); }
    [[nodiscard]] std::size_t doc_count() const noexcept
    { return subset_ == DatasetSubset::Test ? info_.test_docs : info_.num_docs; }
    [[nodiscard]] const std::string& loss_scope() const noexcept { return info_.loss_scope; }
    [[nodiscard]] const TextCorpus& corpus() const noexcept { return corpus_; }
    [[nodiscard]] const DatasetInfo& info() const noexcept { return info_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] const KeyValueRecord& vocab_kvrec() const noexcept { return vocab_; }

private:
    TextCorpus corpus_;
    DatasetInfo info_{};
    TextWindowConfig cfg_{};
    DatasetSubset subset_ = DatasetSubset::Train;
    std::size_t stride_ = 0;
    std::vector<std::size_t> offsets_;
    std::string path_;
    KeyValueRecord vocab_;
};

} // namespace nn
