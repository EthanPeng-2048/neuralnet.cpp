// ── session_checkpoint_test — TrainingSession 断点编排（issue #37）────────
//
// 覆盖：
//   [1] save_checkpoint → resume 往返：权重逐位一致、优化器 t 恢复、
//       会话计数器（global_step/lr）恢复、加载态返回
//   [2] 编排自动同步：blob 计数由 save_model 回填（peek 验证）、
//       fill_recipe 派生配方字段（optimizer/lr/wd/beta/max_norm/accum/精度）
//   [3] 交叉校验：配方不一致 Strict fail-fast、Relaxed 放行
//   [4] 续训连通：resume 后继续 train_step，参数确实移动（优化器状态接上）
//   [5] 域字段不被覆盖：save_checkpoint 不清掉域预填的 dataset 身份
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace {

constexpr std::size_t VOCAB = 32, D_MODEL = 16, SEQ_LEN = 8, HEADS = 2,
                      D_FF = 32, LAYERS = 1;

std::filesystem::path scratch_dir()
{
    const auto dir = std::filesystem::temp_directory_path() /
                     "nn_session_ckpt_test";
    std::filesystem::create_directories(dir);
    return dir;
}

nn::ModelSpec tiny_spec() { return nn::make_gpt_spec(VOCAB, D_MODEL, SEQ_LEN, HEADS, D_FF, LAYERS); }

std::vector<nn::Scalar> snapshot(nn::CpuEngine& e, nn::Model& m)
{
    std::vector<nn::Scalar> out;
    for (auto& t : m.parameters())
    {
        auto m2 = e.to_matrix(t.get(), nn::Precision::F32);
        NN_ASSERT_ALWAYS(m2, "snapshot: to_matrix");
        for (auto v : m2->span()) out.push_back(v);
    }
    return out;
}

// 走一遍 train_step（固定输入/标签，确定性）
nn::Result<nn::TrainingStepReport> one_step(nn::TrainingSession& s,
                                            const nn::Tensor& x, nn::Scalar lr)
{
    nn::TrainingStepOptions opts;
    opts.lr = lr;
    opts.normalization_units = 1;
    opts.force_update = true;
    return s.train_step(0, opts, [&](nn::Model& model, nn::ComputeEngine& eng,
                                     int) -> nn::Result<nn::TrainingStepOutput> {
        NN_TRY(logits, model.forward(x));
        // step_fn 契约：返回**对模型输出**的梯度；model.backward 由 session 执行
        //（本测试焦点是断点编排，直接给常数输出梯度，不经 loss 层）
        auto go = eng.create_tensor(logits->rows(), logits->cols(),
                                    nn::Precision::F32, nn::InitSpec::constant(0.01f));
        if (!go.valid()) NN_FAIL("one_step: go 分配失败");
        return nn::TrainingStepOutput{std::move(go), 1, {}, 0};
    });
}

nn::TextTrainState sample_domain_state()
{
    nn::TextTrainState st;
    st.stride = 8;
    st.dataset.path = "/data/train.nndataset";
    st.dataset.sha256 = "deadbeefcafe0123456789abcdef";
    st.dataset.num_tokens = 1000;
    st.dataset.num_docs = 10;
    st.dataset.num_samples = 100;
    st.dataset.steps_per_epoch = 25;
    st.dataset.loss_scope = "all";
    return st;
}

// ── [1][2][4][5] 保存 → 恢复 → 续训 ──────────────────────────────────────
void test_roundtrip()
{
    auto engine_r = nn::CpuEngine::create();
    auto& engine = *engine_r;
    const auto path = (scratch_dir() / "roundtrip.nnckpt").string();

    // ── 训 4 步（lr 固定），第 2 步后存断点，再训 2 步 ──
    auto model_r = nn::build_gpt_model_from_spec(engine, tiny_spec(), nn::PrecisionProfile{});
    CHECK(model_r.has_value(), "建模型 A");
    if (!model_r) return;
    nn::TrainingConfig cfg;
    cfg.optimizer = "adam";
    cfg.lr = 0.001f;
    cfg.weight_decay = 0.01f;
    cfg.clip_grad_norm = 1.0f;
    cfg.grad_accum_steps = 1;
    auto ses_r = nn::TrainingSession::create(std::move(*model_r), cfg);
    CHECK(ses_r.has_value(), "create session A");
    if (!ses_r) return;
    auto session = std::move(*ses_r);

    // GPT 输入 = token ids (seq, batch)：小整数 id（f32 存储）
    auto x = make_tensor(engine, SEQ_LEN, 1, 0.0f, 1.0f);
    auto s1 = one_step(session, x, 0.001f);
    CHECK(s1.has_value(), "step 1");
    if (!s1) std::printf("     step1: %s\n", s1.error().message.c_str());
    CHECK(one_step(session, x, 0.001f).has_value(), "step 2");
    CHECK(session.step_count() == 2, "step_count = 2");

    auto mid = snapshot(engine, session.model());

    // ── [1][2][5] 保存（域字段预填）──
    auto domain = sample_domain_state();
    CHECK(session.save_checkpoint(path, tiny_spec(), domain).has_value(),
          "[1] save_checkpoint");
    auto loaded_state = snapshot(engine, session.model());   // 保存不应改动参数
    CHECK(loaded_state == mid, "[1] save_checkpoint 不改动参数");

    // [2] blob 计数：peek 回读 = Adam 2×参数条数；fill_recipe 字段已派生
    auto peek_r = nn::peek_train_state(path);
    CHECK(peek_r.has_value(), "[2] peek_train_state");
    if (peek_r)
    {
        const nn::TrainState& ps = **peek_r;
        auto params = session.model().parameters();
        CHECK(ps.optimizer_blob_count == 2 * params.size(),
              "[2] blob 计数由 save_model 回填（2×参数条数）");
        CHECK(ps.recipe.optimizer == "adam", "[2] fill_recipe: optimizer");
        CHECK(std::fabs(ps.recipe.lr - 0.001) < 1e-9, "[2] fill_recipe: lr");
        CHECK(std::fabs(ps.recipe.weight_decay - 0.01) < 1e-9, "[2] fill_recipe: wd");
        CHECK(std::fabs(ps.recipe.max_norm - 1.0) < 1e-9, "[2] fill_recipe: max_norm");
        CHECK(ps.recipe.accum_steps == 1, "[2] fill_recipe: accum_steps");
        // [5] 域字段保留
        auto* ts = dynamic_cast<const nn::TextTrainState*>(&ps);
        CHECK(ts != nullptr, "[5] 状态为文本域子类");
        if (ts)
        {
            CHECK(ts->dataset.sha256 == "deadbeefcafe0123456789abcdef",
                  "[5] save_checkpoint 不覆盖域数据集身份");
            CHECK(ts->stride == 8, "[5] 域 stride 保留");
        }
        CHECK(ps.counters.global_step == 2, "[2] global_step 同步 = 2");
        CHECK(ps.optimizer_step == 1 || ps.optimizer_step > 0,
              "[2] 优化器 t 已同步（Adam 每 update 一步）");
    }

    // ── 继续训 2 步（会话内）──
    CHECK(one_step(session, x, 0.001f).has_value(), "step 3");
    CHECK(one_step(session, x, 0.001f).has_value(), "step 4");
    CHECK(session.step_count() == 4, "step_count = 4");

    // ── [1] 新会话 resume：权重应与断点时刻（mid）逐位一致 ──
    auto model2_r = nn::build_gpt_model_from_spec(engine, tiny_spec(), nn::PrecisionProfile{});
    CHECK(model2_r.has_value(), "建模型 B");
    if (!model2_r) return;
    auto ses2_r = nn::TrainingSession::create(std::move(*model2_r), cfg);
    CHECK(ses2_r.has_value(), "create session B");
    if (!ses2_r) return;
    auto session2 = std::move(*ses2_r);

    // 交叉校验：本次状态与断点一致 → 通过（配方经 fill_recipe 派生 = 真实用法）
    auto cur = sample_domain_state();
    session2.fill_recipe(cur.recipe);
    auto st_r = session2.resume(path, &cur, nn::ResumePolicy::Strict);
    CHECK(st_r.has_value(), "[1] resume（交叉校验通过）");
    if (!st_r)
    {
        std::printf("     %s\n", st_r.error().message.c_str());
        return;
    }
    auto resumed = snapshot(engine, session2.model());
    CHECK(resumed == mid, "[1] resume 后权重与断点时刻逐位一致");
    CHECK(session2.step_count() == 2, "[1] 会话计数器恢复 = 2");

    // [3] 交叉校验：数据集身份变了 → Strict fail-fast；Relaxed 放行
    auto bad = sample_domain_state();
    session2.fill_recipe(bad.recipe);
    bad.dataset.sha256 = "0000000000000000000000000000";
    auto strict_r = session2.resume(path, &bad, nn::ResumePolicy::Strict);
    CHECK(!strict_r.has_value(), "[3] Strict：数据集身份不一致 fail-fast");
    auto relaxed_r = session2.resume(path, &bad, nn::ResumePolicy::Relaxed);
    CHECK(relaxed_r.has_value(), "[3] Relaxed：放行");
    if (!relaxed_r)
        std::printf("     %s\n", relaxed_r.error().message.c_str());

    // ── [4] resume 后继续训练：参数确实移动（优化器状态已接上）──
    auto before = snapshot(engine, session2.model());
    auto rep = one_step(session2, x, 0.001f);
    CHECK(rep.has_value(), "[4] resume 后 train_step");
    if (rep)
    {
        CHECK(rep->parameters_updated, "[4] update 发生");
        auto after = snapshot(engine, session2.model());
        CHECK(before != after, "[4] resume 后参数确实移动");
        CHECK(session2.step_count() == 3, "[4] 计数器继续累计 = 3");
    }
}

// ── [3] 配方不一致（非域字段）也应 fail-fast ──────────────────────────────
void test_recipe_mismatch()
{
    auto engine_r = nn::CpuEngine::create();
    auto& engine = *engine_r;
    const auto path = (scratch_dir() / "recipe.nnckpt").string();

    auto model_r = nn::build_gpt_model_from_spec(engine, tiny_spec(), nn::PrecisionProfile{});
    CHECK(model_r.has_value(), "建模型");
    if (!model_r) return;
    nn::TrainingConfig cfg;
    cfg.optimizer = "adam";
    auto ses_r = nn::TrainingSession::create(std::move(*model_r), cfg);
    CHECK(ses_r.has_value(), "create");
    if (!ses_r) return;
    auto session = std::move(*ses_r);
    // GPT 输入 = token ids (seq, batch)：小整数 id（f32 存储）
    auto x = make_tensor(engine, SEQ_LEN, 1, 0.0f, 1.0f);
    CHECK(one_step(session, x, 0.001f).has_value(), "step");
    auto domain = sample_domain_state();
    CHECK(session.save_checkpoint(path, tiny_spec(), domain).has_value(), "save");

    // 新会话用不同优化器名 → Strict 下配方不一致 fail-fast
    auto model2_r = nn::build_gpt_model_from_spec(engine, tiny_spec(), nn::PrecisionProfile{});
    if (!model2_r) { CHECK(false, "建模型 B"); return; }
    nn::TrainingConfig cfg2;
    cfg2.optimizer = "sgd";   // 与断点的 adam 不一致
    auto ses2_r = nn::TrainingSession::create(std::move(*model2_r), cfg2);
    if (!ses2_r) { CHECK(false, "create B"); return; }
    auto session2 = std::move(*ses2_r);
    auto cur = sample_domain_state();
    session2.fill_recipe(cur.recipe);   // cfg2.optimizer = "sgd" ≠ 断点 "adam"
    auto r = session2.resume(path, &cur, nn::ResumePolicy::Strict);
    CHECK(!r.has_value(), "[3] Strict：optimizer 配方不一致 fail-fast");
    // 注意：resume 内 load_model 已载入权重（断点参数），此处会话仍可用
}

} // namespace

int main()
{
    test_roundtrip();
    test_recipe_mismatch();
    if (g_fail == 0) std::printf("session_checkpoint_test: ALL PASS\n");
    else std::printf("session_checkpoint_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
