// ── text_trainer_test — 训练工厂（TextTrainer）端到端回归 ───────────────────
//
// P2b 验收（docs/internal/29 §8）：数据集**直接接入**训练工厂；样本概念归一；
// 训练状态由工厂装配、Session 持有。本测试锁住：
//   [1] create(dataset)：内部建样本集 / 载词表 / 数据身份（num_samples、steps）
//   [2] fit()：步数 = ceil(样本数/batch)、参数移动、epoch 汇总 + 测试评估输出
//   [3] max-steps / start_step 语义（本进程步数、真实停止位置写回 TrainState）
//   [3b] 高性能入口 create(session, train_set, nullopt, cfg)（CLI 实际路径）+ 网格校验
//   [4] 断点钩子：TrainState（计数器/配方/数据集身份/优化器 t）随 .nnckpt 落盘
//   [5] 前置条件：batch=0 / log_interval=0 / 配方与 Session 不一致 → Result 错误
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/text_trainer.hpp"   // 训练工厂（L5；不在 nn.hpp 聚合头内）

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

int g_failures = 0;

namespace
{
constexpr const char* kCorpus = "the quick brown fox jumps over the lazy dog\n"
                                "a small model learns to predict the next token\n"
                                "training data is split into documents and windows\n"
                                "the lazy dog sleeps while the fox runs away\n"
                                "practical tests keep the training pipeline honest\n"
                                "windows are samples and steps are derived from batches\n"
                                "the quick brown fox returns in the final line\n"
                                "a short document ends here\n";

void write_file(const std::string& path, const std::string& bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// 生成小数据集（BPE 词表内嵌；train 8 docs / test 2 docs）
std::string make_dataset(const nn::BPETokenizer& tok, const std::string& path,
                         std::size_t vocab_size)
{
    std::vector<std::string> docs;
    std::size_t pos = 0;
    const std::string corpus = kCorpus;
    while (pos < corpus.size())
    {
        const std::size_t end = corpus.find('\n', pos);
        docs.push_back(corpus.substr(pos, end == std::string::npos ? end : end - pos));
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    nn::NndatasetBuild b;
    b.vocab_kvrec = tok.to_kvrec();
    b.source_path = "text_trainer_test";
    b.source_sha256 = std::string(64, 'b');
    b.source_docs = docs.size();
    b.loss_scope = "all";
    b.tool = "text_trainer_test";
    b.created = 1;
    for (std::size_t i = 0; i < docs.size(); ++i)
    {
        auto toks = tok.encode(docs[i]);
        if (toks.empty()) continue;
        if (i < 2) b.test_docs.push_back(std::move(toks));
        else b.train_docs.push_back(std::move(toks));
    }
    std::ostringstream oss;
    const auto wr = nn::write_nndataset(oss, b);
    NN_ASSERT_ALWAYS(wr, "text_trainer_test: write_nndataset 失败");
    write_file(path,
               nn::write_file_label(nn::NNDATASET_MAGIC, nn::NNDATASET_FILE_VERSION) + oss.str());
    (void)vocab_size;
    return path;
}

[[nodiscard]] std::vector<nn::Scalar> params_snapshot(nn::ComputeEngine& eng, nn::Model& model)
{
    std::vector<nn::Scalar> out;
    for (auto& t : model.parameters())
    {
        auto m = eng.to_matrix(t.get(), nn::Precision::F32);
        NN_ASSERT_ALWAYS(m, "params_snapshot: to_matrix failed");
        for (auto v : m->span())
            out.push_back(v);
    }
    return out;
}

[[nodiscard]] double l1_move(const std::vector<nn::Scalar>& a, const std::vector<nn::Scalar>& b)
{
    double s = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
        s += std::fabs(a[i] - b[i]);
    return s;
}

struct Fixture
{
    std::shared_ptr<nn::CpuEngine> engine;
    nn::ModelSpec spec;
    std::string dataset_path;
};

[[nodiscard]] Fixture make_fixture(const std::string& dir)
{
    Fixture f;
    f.engine = nn::CpuEngine::create();
    nn::BPETokenizer::Config tcfg;
    tcfg.vocab_size = 320;
    tcfg.min_freq = 1;
    tcfg.log = [](std::string_view) {};
    nn::BPETokenizer tok;
    const auto tr = tok.train(kCorpus, tcfg);
    NN_ASSERT_ALWAYS(tr, "text_trainer_test: tokenizer 训练失败");
    f.dataset_path = make_dataset(tok, dir + "/trainer.nndataset", tcfg.vocab_size);
    f.spec = nn::make_gpt_spec(tok.vocab_size(), /*d_model=*/16, /*seq_len=*/8,
                               /*heads=*/2, /*d_ff=*/32, /*layers=*/1);
    return f;
}

[[nodiscard]] nn::TrainingSession make_session(Fixture& f, nn::Scalar lr = 1e-2f,
                                               nn::Scalar clip = 0, std::size_t accum_steps = 1)
{
    auto built = nn::build_gpt_model_from_spec(*f.engine, f.spec);
    NN_ASSERT_ALWAYS(built, "text_trainer_test: 构建小 GPT 失败");
    nn::TrainingConfig tcfg;
    tcfg.optimizer = "adam";
    tcfg.lr = lr;
    tcfg.weight_decay = 0.0f;
    tcfg.clip_grad_norm = clip;
    tcfg.grad_accum_steps = accum_steps;
    auto s = nn::TrainingSession::create(std::move(*built), tcfg);
    NN_ASSERT_ALWAYS(s, "text_trainer_test: TrainingSession::create 失败");
    return std::move(*s);
}

// 配方从 Session 派生（优化器侧唯一真源），只覆盖数据网格/记录字段。
[[nodiscard]] nn::TextTrainConfig default_cfg(const nn::TrainingSession& session)
{
    nn::TextTrainConfig cfg = nn::make_text_train_config(session);
    cfg.seq_len = 8;
    cfg.stride = 8;
    cfg.batch_size = 2;
    cfg.epochs = 1;
    cfg.log_interval = 1;
    cfg.save_interval = 0;
    return cfg;
}

// ── [1][2] 数据集直接接入 + fit 端到端 ────────────────────────────────────
void test_fit_end_to_end(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "Dataset::open");
    if (!ds) return;
    const std::size_t docs = ds->doc_count(nn::DatasetSubset::Train);
    const std::size_t tokens = [&]
    {
        auto c = ds->load_text(nn::DatasetSubset::Train);
        return c ? c->token_flow.size() : 0;
    }();

    nn::TrainingSession session = make_session(f);
    auto before = params_snapshot(*f.engine, session.model());

    nn::TextTrainConfig cfg = default_cfg(session);
    cfg.identity.path = f.dataset_path;   // 调用方已算好身份：检查工厂不重算 sha
    cfg.identity.sha256 = std::string(64, 'e');

    std::ostringstream out;
    nn::TextTrainerHooks hooks;
    hooks.out = &out;
    auto trainer_r = nn::TextTrainer::create(std::move(session), *ds, cfg, hooks);
    CHECK(trainer_r, "TextTrainer::create(session, dataset)");
    if (!trainer_r) return;
    nn::TextTrainer& trainer = *trainer_r;

    const std::size_t samples = trainer.sample_count();
    const std::size_t steps = trainer.steps_per_epoch();
    CHECK(samples == (tokens + 7) / 8, "样本数 = ceil(tokens / stride)");
    CHECK(steps == (samples + 1) / 2, "step = ceil(样本数 / batch_size)");
    CHECK(trainer.dataset_identity().num_samples == samples, "身份 num_samples");
    CHECK(trainer.dataset_identity().steps_per_epoch == steps, "身份 steps_per_epoch");
    CHECK(trainer.dataset_identity().num_docs == docs, "身份 num_docs");
    CHECK(trainer.dataset_identity().num_tokens == tokens, "身份 num_tokens");
    CHECK(trainer.dataset_identity().loss_scope == "all", "身份 loss_scope");
    CHECK(trainer.dataset_identity().sha256 == std::string(64, 'e'),
          "调用方给的身份优先（不重算大文件 sha）");
    CHECK(trainer.test_samples() != nullptr, "test 子集被接进训练工厂");

    const auto report_r = trainer.fit();
    CHECK(report_r, "TextTrainer::fit");
    if (!report_r) return;
    const nn::TextTrainReport& report = *report_r;
    CHECK(report.steps_executed == steps, "fit 执行步数 = steps_per_epoch");
    CHECK(report.samples_executed == samples, "fit 消费样本数 = 样本数");
    CHECK(report.ckpt_failures == 0 && report.abort_code == 0, "无断点失败/探针中止");
    CHECK(trainer.session().step_count() == steps, "Session 的 step_count 与循环一致");
    CHECK(trainer.train_state().dataset.num_samples == samples,
          "Trainer 持有的运行状态含数据集身份");

    // 输出锚：进度行 + epoch 汇总 + 测试 loss（文案由 Trainer 默认打印）
    const std::string text = out.str();
    CHECK(text.find("Epoch 1/1") != std::string::npos, "打印 epoch 进度");
    CHECK(text.find("avg_loss=") != std::string::npos, "打印 epoch 平均 loss");
    CHECK(text.find("test_loss=") != std::string::npos, "有 test 子集时打印测试 loss");
    CHECK(text.find("样本数") != std::string::npos, "打印样本数（不是 step 数）");

    const auto after = params_snapshot(*f.engine, trainer.session().model());
    CHECK(l1_move(before, after) > 1e-6, "fit 后参数确实移动");
}

// ── [3] max-steps / start_step 语义 ──────────────────────────────────────
void test_step_range(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "step-range: Dataset::open");
    if (!ds) return;

    nn::TrainingSession s1 = make_session(f);
    nn::TextTrainConfig cfg = default_cfg(s1);
    cfg.max_steps = 1;
    std::ostringstream out1;
    nn::TextTrainerHooks h1;
    h1.out = &out1;
    auto t1 = nn::TextTrainer::create(std::move(s1), *ds, cfg, h1);
    CHECK(t1, "step-range: create(max_steps=1)");
    if (t1)
    {
        auto r = t1->fit();
        CHECK(r && r->steps_executed == 1, "max_steps=1 只执行 1 步");
        CHECK(out1.str().find("已达 --max-steps=1") != std::string::npos, "max_steps 提前结束文案");
        // P1：--max-steps 提前停时状态必须是"真实下一步"，不能谎报 (epochs,0)
        CHECK(r && r->final_epoch == 0 && r->final_step_in_epoch == 1, "report 记录真实停止位置");
        CHECK(t1->train_state().counters.epoch == 0 &&
                  t1->train_state().counters.step_in_epoch == 1 &&
                  t1->train_state().counters.global_step == 1,
              "Trainer.TrainState 在 max_steps 下 = 下一步位置（非 epochs,0）");
    }

    // start_step = 1 ⇒ 本进程只跑 [1, steps)
    nn::TrainingSession s2 = make_session(f);
    nn::TextTrainConfig cfg2 = default_cfg(s2);
    cfg2.start_step = 1;
    auto t2 = nn::TextTrainer::create(std::move(s2), *ds, cfg2);
    CHECK(t2, "step-range: create(start_step=1)");
    if (t2)
    {
        auto r = t2->fit();
        CHECK(r && r->steps_executed == t2->steps_per_epoch() - 1,
              "start_step 之后只剩 steps-1 步");
        CHECK(r && r->samples_executed == t2->sample_count() - cfg2.batch_size,
              "start_step 跳过的样本不再消费");
    }
}

// ── [3a] global batch / micro batch / Session 内置累积 ────────────────────
void test_global_batch_accumulation(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "accum: Dataset::open");
    if (!ds) return;

    nn::TrainingSession session = make_session(f, 1e-2f, 0.0f, /*accum_steps=*/2);
    nn::TextTrainConfig cfg = default_cfg(session);
    cfg.batch_size = 4;
    cfg.accum_steps = 2;
    cfg.log_interval = 100;

    std::size_t micro_batches = 0;
    std::size_t updates = 0;
    nn::TextTrainerHooks hooks;
    hooks.out = nullptr;
    hooks.on_batch_ready = [&](nn::TrainingSession&, const nn::TextSampleBatch& batch, std::size_t)
    { micro_batches += batch.batch; };
    hooks.on_update = [&](nn::TrainingSession&, std::size_t, std::size_t, nn::Scalar)
    { ++updates; };

    auto trainer_r = nn::TextTrainer::create(std::move(session), *ds, cfg, hooks);
    CHECK(trainer_r, "accum: create");
    if (!trainer_r) return;
    CHECK(trainer_r->micro_batch_size() == 2, "accum: micro = global / accum");
    const std::size_t samples = trainer_r->sample_count();
    const std::size_t expected_steps = (samples + 3) / 4;
    CHECK(trainer_r->steps_per_epoch() == expected_steps, "accum: steps_per_epoch 按 global batch");

    auto report = trainer_r->fit();
    CHECK(report && report->steps_executed == expected_steps, "accum: update step 数正确");
    CHECK(report && report->samples_executed == samples, "accum: 样本数正确");
    CHECK(trainer_r->session().step_count() == expected_steps,
          "accum: Session step_count 按 update 计");
    CHECK(updates == expected_steps, "accum: on_update 每 update 一次");
    CHECK(micro_batches == samples, "accum: micro-batch 样本列数总和");
    CHECK(trainer_r->train_state().recipe.batch_size_is_global,
          "accum: 状态标记 global batch 语义");
    CHECK(trainer_r->train_state().recipe.micro_batch_size == 2, "accum: 状态记录派生 micro-batch");
}

// ── [3b] 高性能入口（CLI 实际路径）：样本集 + nullopt test ─────────────────
void test_fast_path_sets(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "fast-path: Dataset::open");
    if (!ds) return;
    auto tok_r = nn::load_tokenizer_from_kvrec(ds->vocab_kvrec());
    CHECK(tok_r, "fast-path: 载词表");
    if (!tok_r) return;

    const nn::TextWindowConfig win{8, 8, (*tok_r)->pad_id()};
    auto train_r = nn::TextSampleSet::create(*ds, nn::DatasetSubset::Train, win);
    CHECK(train_r, "fast-path: create(train)");
    if (!train_r) return;
    const std::size_t samples = train_r->sample_count();

    nn::TrainingSession session = make_session(f);
    nn::TextTrainConfig cfg = default_cfg(session);
    cfg.identity.path = f.dataset_path;
    cfg.identity.sha256 = std::string(64, 'a');

    std::ostringstream out;
    nn::TextTrainerHooks hooks;
    hooks.out = &out;
    auto trainer_r =
        nn::TextTrainer::create(std::move(session), std::move(*train_r), std::nullopt, cfg, hooks);
    CHECK(trainer_r, "fast-path: create(session, train_set, nullopt, cfg)");
    if (!trainer_r) return;
    CHECK(trainer_r->test_samples() == nullptr, "fast-path: nullopt test ⇒ 不评估");
    const auto report = trainer_r->fit();
    CHECK(report && report->steps_executed == (samples + 1) / 2,
          "fast-path: fit 跑完 steps = ceil(样本数/batch)");
    CHECK(out.str().find("test_loss=") == std::string::npos,
          "fast-path: 无 test 子集不打印 test_loss");
    CHECK(out.str().find("avg_loss=") != std::string::npos, "fast-path: 打印 epoch 汇总");

    // 窗口网格/子集校验：Test 子集当训练集、配方 seq_len 不一致都必须拒绝
    {
        auto t2 = nn::TextSampleSet::create(*ds, nn::DatasetSubset::Test, win);
        if (t2)
        {
            nn::TrainingSession s = make_session(f);
            nn::TextTrainConfig bad = default_cfg(s);
            auto r = nn::TextTrainer::create(std::move(s), std::move(*t2), std::nullopt, bad);
            CHECK(!r, "fast-path: Test 子集当训练集 → Result 错误");
        }
    }
    {
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig bad = default_cfg(s);
        bad.seq_len = 16;   // 与样本集窗口不一致
        auto t = nn::TextSampleSet::create(*ds, nn::DatasetSubset::Train,
                                           nn::TextWindowConfig{8, 8, (*tok_r)->pad_id()});
        if (t)
        {
            auto r = nn::TextTrainer::create(std::move(s), std::move(*t), std::nullopt, bad);
            CHECK(!r, "fast-path: 配方 seq_len 与样本集不一致 → Result 错误");
        }
    }
}

// ── [4] 断点钩子：状态落盘与身份/配方 ────────────────────────────────────
void test_checkpoint_hook(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "ckpt: Dataset::open");
    if (!ds) return;

    nn::TrainingSession session = make_session(f);
    nn::TextTrainConfig cfg = default_cfg(session);
    cfg.epochs = 2;
    cfg.save_interval = 1;   // 每步都存
    cfg.identity.path = f.dataset_path;
    cfg.identity.sha256 = std::string(64, 'f');

    const std::string ckpt = dir + "/trainer_ckpt.nnckpt";
    std::vector<nn::TextTrainState> states;
    nn::TextTrainerHooks hooks;
    std::ostringstream out;
    hooks.out = &out;
    hooks.on_checkpoint = [&states, &ckpt,
                           &f](const nn::TextCheckpointRequest& req) -> nn::Result<void>
    {
        states.push_back(req.state);
        std::vector<nn::Tensor> blobs;
        const auto sr = req.session.optimizer().save_state(blobs);
        if (!sr) NN_FAIL("保存优化器状态失败");
        const nn::CheckpointState ck{&req.state, blobs};
        return nn::save_model(ckpt, req.session.model(), f.spec, "tok-bytes",
                              nn::ModelRole::Checkpoint, ck);
    };

    auto trainer_r = nn::TextTrainer::create(std::move(session), *ds, cfg, hooks);
    CHECK(trainer_r, "ckpt: create");
    if (!trainer_r) return;
    const auto report = trainer_r->fit();
    CHECK(report && report->ckpt_failures == 0, "断点钩子全部成功");
    CHECK(!states.empty(), "断点钩子被调用");
    if (states.empty()) return;

    const nn::TextTrainState& st = states.back();
    CHECK(st.counters.global_step > 0, "状态含 global_step");
    CHECK(st.dataset.sha256 == std::string(64, 'f'), "状态含数据集身份");
    CHECK(st.dataset.num_samples == trainer_r->sample_count(), "状态含样本数");
    CHECK(st.dataset.steps_per_epoch == trainer_r->steps_per_epoch(), "状态含 steps");
    CHECK(st.recipe.batch_size == cfg.batch_size && st.recipe.epochs == 2,
          "状态含训练配方（batch/epochs）");
    CHECK(st.recipe.optimizer == "adam", "状态含优化器名");
    CHECK(st.optimizer_step == trainer_r->session().optimizer().state_step(),
          "状态含 Adam 的 t（优化器权威 t，issue #42）");
    CHECK(st.stride == cfg.stride, "状态含 stride（文本域扩展）");

    // 落盘状态可被 peek/load 读回（与 CLI 断点同一文件格式）
    auto peeked = nn::peek_train_state(ckpt);
    CHECK(peeked, "peek_train_state");
    if (peeked)
    {
        auto* pts = dynamic_cast<nn::TextTrainState*>(peeked->get());
        CHECK(pts != nullptr, "peek 得到文本域状态");
        if (pts)
        {
            CHECK(pts->counters.global_step == st.counters.global_step, "状态计数往返");
            CHECK(pts->dataset.num_samples == st.dataset.num_samples, "样本数往返");
            CHECK(pts->optimizer_step == st.optimizer_step, "优化器 t 往返");
            CHECK(pts->stride == st.stride, "stride 往返");
        }
    }
}

// ── [6] 前置条件 ─────────────────────────────────────────────────────────
void test_preconditions(const std::string& dir)
{
    Fixture f = make_fixture(dir);
    auto ds = nn::Dataset::open(f.dataset_path);
    CHECK(ds, "precond: Dataset::open");
    if (!ds) return;

    {
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.batch_size = 0;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "batch_size=0 → Result 错误");
    }
    {
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.batch_size = 3;
        cfg.accum_steps = 2;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "global batch 不被 accum_steps 整除 → Result 错误");
    }
    {
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.accum_steps = 2;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "配方 accum_steps 与 Session 不一致 → Result 错误");
    }
    {
        // 梯度裁剪真源是 Session 的 TrainingConfig；配方不一致必须拒绝
        nn::TrainingSession s = make_session(f, 1e-2f, /*clip=*/0.0f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.max_norm = 1.0f;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "max_norm 与 Session.clip_grad_norm 不一致 → Result 错误");
    }
    {
        // 配方重叠字段（lr）与 Session 不一致 → 拒绝（P0 静默错训防护）
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.lr += 0.5f;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "lr 与 Session 不一致 → Result 错误");
    }
    {
        // log_interval=0 是进度打印的除零前置 → fail-fast
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.log_interval = 0;
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "log_interval=0 → Result 错误");
    }
    {
        // 发布审查 P1-1：配方 seq_len 必须等于模型 spec.seq_len（模型用
        // build_gpt_model_from_spec 建 ⇒ spec 存在）。不一致时 GPT 的 forward 用
        // input.rows()、位置编码/掩码却按模型 seq_len 构建 ⇒ GPU 越界读。
        nn::TrainingSession s = make_session(f);
        nn::TextTrainConfig cfg = default_cfg(s);
        cfg.seq_len = 16;   // fixture 模型 spec.seq_len = 8
        auto t = nn::TextTrainer::create(std::move(s), *ds, cfg);
        CHECK(!t, "配方 seq_len 与模型 spec.seq_len 不一致 → Result 错误");
    }
}
} // namespace

int main()
{
    std::printf("text_trainer_test（训练工厂：数据集接入 / 样本 / 状态 / 断点）\n");
    const auto dir = std::filesystem::temp_directory_path() / "nn_text_trainer_test";
    std::filesystem::create_directories(dir);
    const std::string d = dir.string();

    test_fit_end_to_end(d);
    test_step_range(d);
    test_global_batch_accumulation(d);
    test_fast_path_sets(d);
    test_checkpoint_hook(d);
    test_preconditions(d);

    std::filesystem::remove_all(dir);
    if (g_failures == 0) std::printf("\ntext_trainer_test: ALL PASS\n");
    else std::printf("\ntext_trainer_test: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
