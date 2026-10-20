// ── dataset_samples_test — 训练样本概念（TextSampleSet / TextSample / batch）──
//
// P2b 验收（docs/todo/29 §8）：数据集只存 doc；**训练单位 = 样本（窗口）**；
// step = ceil(样本数 / batch_size) 是派生量。本测试锁住：
//   [1] 切窗：偏移序列、样本数、steps_per_epoch（派生）
//   [2] sample(i) 的视图（tokens/doc_ids/loss_mask/valid_len）
//   [3] make_batch 的逐位约定：PAD、目标 = 下一 token、掩码按**目标**位置、
//       doc_ids PAD=0、扁平数组 batch-major（i = b·seq+t）
//   [4] assistant 掩码（loss_scope=assistant）与 all 全 1 合成
//   [5] 错误语义：test 子集缺失 / seq_len=0 / batch 下标越界
//   [6] 断点状态段的旧键 `dataset_windows` 兼容读（rc0 → rc0.1）
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

int g_failures = 0;

namespace
{
void write_file(const std::string& path, const std::string& bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// 组装一个最小 .nndataset（train 3 docs / test 2 docs；vocab 只需可往返）。
std::string make_dataset_file(const std::string& path,
                              const std::vector<std::vector<std::size_t>>& train,
                              const std::vector<std::vector<std::size_t>>& test,
                              const std::string& loss_scope,
                              const std::vector<std::vector<unsigned char>>& train_masks = {})
{
    nn::NndatasetBuild b;
    b.vocab_kvrec.set("size", static_cast<std::uint64_t>(16));
    b.source_path = "synthetic";
    b.source_sha256 = std::string(64, 'a');
    b.source_docs = train.size();
    b.loss_scope = loss_scope;
    b.tool = "dataset_samples_test";
    b.created = 1;
    b.train_docs = train;
    b.train_masks = train_masks;
    b.test_docs = test;
    if (loss_scope == "assistant") b.test_masks = train_masks;   // 测试子集本用例不校验掩码内容

    std::ostringstream oss;
    const auto wr = nn::write_nndataset(oss, b);
    if (!wr)
    {
        std::printf("[debug] write_nndataset: %s\n", wr.error().message.c_str());
        NN_ASSERT_ALWAYS(wr, "dataset_samples_test: write_nndataset 失败");
    }
    write_file(path,
               nn::write_file_label(nn::NNDATASET_MAGIC, nn::NNDATASET_FILE_VERSION) + oss.str());
    return path;
}

void test_windowing_and_samples()
{
    const std::string path = "dataset_samples_test.nndataset";
    make_dataset_file(path, {{1, 2, 3, 4, 5}, {6, 7, 8}, {9}}, {{1, 2, 3}, {4, 5}}, "all");
    auto ds_r = nn::Dataset::open(path);
    CHECK(ds_r, "Dataset::open");
    if (!ds_r) return;
    const nn::Dataset& ds = *ds_r;
    CHECK(ds.doc_count(nn::DatasetSubset::Train) == 3, "doc_count(train) = 存储单位数");
    CHECK(ds.doc_count(nn::DatasetSubset::Test) == 2, "doc_count(test)");

    // 单 doc 访问（存储单位 = doc）
    {
        auto doc = ds.text_doc(nn::DatasetSubset::Train, 1);
        CHECK(doc, "text_doc(train, 1)");
        if (doc)
        {
            CHECK(doc->doc_id == 2, "doc_id = 子集内序号 + 1");
            CHECK(doc->tokens == std::vector<std::size_t>({6, 7, 8}), "doc tokens");
            CHECK(doc->loss_mask == std::vector<unsigned char>({1, 1, 1}),
                  "loss_scope=all 的 doc 掩码合成全 1");
        }
        auto bad = ds.text_doc(nn::DatasetSubset::Train, 3);
        CHECK(!bad, "doc 下标越界 → Result 错误");
    }

    const nn::TextWindowConfig win{/*seq_len=*/4, /*stride=*/2, /*pad_id=*/99};
    auto set_r = nn::TextSampleSet::create(ds, nn::DatasetSubset::Train, win);
    CHECK(set_r, "TextSampleSet::create(train)");
    if (!set_r) return;
    const nn::TextSampleSet& set = *set_r;

    // [1] 切窗：偏移 = 0,2,4,6,8（9 token / stride 2）⇒ 5 个样本
    CHECK(set.sample_count() == 5, "样本数 = ceil(token 数 / stride)");
    CHECK(set.steps_per_epoch(2) == 3, "step = ceil(样本数 / batch_size)（派生）");
    CHECK(set.steps_per_epoch(5) == 1, "batch = 样本数 ⇒ 1 步");
    CHECK(set.steps_per_epoch(0) == 0, "batch_size=0 ⇒ 0（调用方负责拒绝）");
    CHECK(set.stride() == 2 && set.seq_len() == 4, "窗口参数回显");
    CHECK(set.token_count() == 9 && set.doc_count() == 3, "元数据回显");

    // [2] 样本视图
    {
        const nn::TextSample s2 = set.sample(2);   // offset 4: [5,6,7,8]
        CHECK(s2.offset == 4 && s2.valid_len == 4, "sample(2) 偏移/有效长度");
        CHECK(std::vector<std::size_t>(s2.tokens.begin(), s2.tokens.end()) ==
                  std::vector<std::size_t>({5, 6, 7, 8}),
              "sample(2) tokens");
        CHECK(std::vector<std::size_t>(s2.doc_ids.begin(), s2.doc_ids.end()) ==
                  std::vector<std::size_t>({1, 2, 2, 2}),
              "sample(2) doc_ids（跨 doc 窗口保留逐 token 文档号）");
        const nn::TextSample s3 = set.sample(3);   // offset 6: [7,8,9]，末窗
        CHECK(s3.valid_len == 3, "末窗 valid_len < seq_len");
    }

    // [3] batch 装配（逐位约定）
    {
        const std::vector<std::size_t> idx{3, 4};
        nn::TextSampleBatch batch;
        auto vr = set.make_batch(idx, batch);
        CHECK(vr && *vr == 2, "make_batch 返回本批参与 loss 的位置数");
        CHECK(batch.batch == 2 && batch.seq_len == 4, "batch 形状");
        CHECK(batch.valid_tokens == 2, "valid_tokens");
        // input: (seq_len, batch) 列 = 样本
        const auto in = batch.input.span();
        // b=0: [7,8,9,99]，b=1: [9,99,99,99]
        CHECK(in[0 * 2 + 0] == 7 && in[1 * 2 + 0] == 8 && in[2 * 2 + 0] == 9 && in[3 * 2 + 0] == 99,
              "input 列为样本、行 = 窗口内位置、末尾 PAD");
        CHECK(in[0 * 2 + 1] == 9 && in[1 * 2 + 1] == 99, "末窗 PAD");
        // 扁平目标/掩码：batch-major i = b*seq+t，目标 = 下一 token
        const std::vector<std::size_t> exp_targets{8, 9, 99, 99, 99, 99, 99, 99};
        const std::vector<nn::Scalar> exp_mask{1, 1, 0, 0, 0, 0, 0, 0};
        CHECK(batch.flat_targets == exp_targets, "flat_targets = 目标下一 token（batch-major）");
        CHECK(batch.flat_mask == exp_mask, "flat_mask 仅真实目标位置参与 loss");
        const std::vector<std::size_t> exp_docs{2, 2, 3, 0, 3, 0, 0, 0};
        CHECK(batch.doc_ids == exp_docs, "doc_ids（batch-major；PAD=0）");
    }

    // 末批（样本数 5、batch 2 ⇒ 最后一批 1 个）可正常装配
    {
        const std::vector<std::size_t> idx{4};
        nn::TextSampleBatch batch;
        const auto vr = set.make_batch(idx, batch);
        CHECK(vr && *vr == 0, "单 token 末窗：无目标 ⇒ 0 个有效位置");
        CHECK(batch.batch == 1 && batch.flat_targets.size() == 4, "末批可不满");
    }

    // [5] 错误语义
    {
        std::vector<std::size_t> oob{set.sample_count()};
        nn::TextSampleBatch batch;
        CHECK(!set.make_batch(oob, batch), "batch 下标越界 → Result 错误");
    }
    {
        auto bad =
            nn::TextSampleSet::create(ds, nn::DatasetSubset::Train, nn::TextWindowConfig{0, 0, 0});
        CHECK(!bad, "seq_len=0 → Result 错误");
    }
    {
        auto bad =
            nn::TextSampleSet::create(ds, nn::DatasetSubset::Test, nn::TextWindowConfig{4, 0, 99});
        CHECK(bad, "有 test 子集 ⇒ create(Test) 成功");
        if (bad) CHECK(bad->sample_count() == 2, "test 样本数 = ceil(5/4)");
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void test_assistant_mask_target_position()
{
    const std::string path = "dataset_samples_test_asst.nndataset";
    // 5 token，掩码 [1,0,1,1,0]：只有**目标位置**的掩码决定是否参与 loss
    make_dataset_file(path, {{1, 2, 3, 4, 5}}, {}, "assistant", {{{1, 0, 1, 1, 0}}});
    auto ds_r = nn::Dataset::open(path);
    CHECK(ds_r, "assistant: Dataset::open");
    if (!ds_r) return;
    auto set_r =
        nn::TextSampleSet::create(*ds_r, nn::DatasetSubset::Train, nn::TextWindowConfig{4, 0, 99});
    CHECK(set_r, "assistant: create");
    if (!set_r) return;
    nn::TextSampleBatch batch;
    const std::vector<std::size_t> idx{0};
    const auto vr = set_r->make_batch(idx, batch);
    // 目标位置 t+1 = 1,2,3 的掩码 = 0,1,1 ⇒ 2 个有效位置
    CHECK(vr && *vr == 2, "assistant: 有效位置数按目标位置掩码");
    CHECK(batch.flat_mask == std::vector<nn::Scalar>({0, 1, 1, 0}),
          "assistant: flat_mask 取 offset+t+1 的掩码");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void test_missing_test_subset()
{
    const std::string path = "dataset_samples_test_notest.nndataset";
    make_dataset_file(path, {{1, 2, 3, 4}}, {}, "all");
    auto ds_r = nn::Dataset::open(path);
    CHECK(ds_r, "无 test 子集：open");
    if (!ds_r) return;
    auto bad =
        nn::TextSampleSet::create(*ds_r, nn::DatasetSubset::Test, nn::TextWindowConfig{4, 0, 0});
    CHECK(!bad, "无 test 子集 ⇒ create(Test) 明确报错");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void test_legacy_train_state_key()
{
    // rc0 断点只有 dataset_windows（同一语义：窗口数 = 样本数）→ 仍可读
    nn::KeyValueRecord kv;
    kv.set("dataset_tokens", static_cast<std::uint64_t>(100));
    kv.set("dataset_docs", static_cast<std::uint64_t>(4));
    kv.set("dataset_windows", static_cast<std::uint64_t>(9));
    kv.set("dataset_steps_per_epoch", static_cast<std::uint64_t>(3));
    auto st = nn::train_state_from_kv(kv);
    CHECK(st, "legacy: train_state_from_kv");
    if (st)
    {
        CHECK(st->dataset.num_samples == 9, "旧键 dataset_windows 读入 num_samples");
        CHECK(st->dataset.steps_per_epoch == 3, "steps_per_epoch 兼容");
    }
    // 新键优先
    kv.set("dataset_samples", static_cast<std::uint64_t>(11));
    auto st2 = nn::train_state_from_kv(kv);
    CHECK(st2 && st2->dataset.num_samples == 11, "新键 dataset_samples 优先");
}
} // namespace

int main()
{
    std::printf("dataset_samples_test（样本概念 / batch 装配 / 兼容）\n");
    test_windowing_and_samples();
    test_assistant_mask_target_position();
    test_missing_test_subset();
    test_legacy_train_state_key();

    if (g_failures == 0) std::printf("\ndataset_samples_test: ALL PASS\n");
    else std::printf("\ndataset_samples_test: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
