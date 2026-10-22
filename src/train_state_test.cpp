// ── train_state_test — v8 训练状态段 + 尾部校验和（P1-2 / P2-1）回归 ─────────
//
// 验收（docs/todo/18-roadmap.md §4.2 P1-2 + §4.3 P2-1、§8.2 门禁 7）：
//   老文件缺段 = 冷启动语义；save→load→save 逐字节一致；截断/损坏文件明确报错。
//
// 覆盖：
//   [1] `TrainState` ⇄ kvrec 往返：计数器 / 配方（含精度四槽与 lr 曲线形状）/
//       数据集身份 / 优化器指针全字段一致（含 lr_per_epoch 与 f16 精度槽）。
//   [2] Checkpoint 往返：`peek_train_state` 字段一致；`load_train_state` 张量
//       **逐位一致**且记录的文件精度正确（f16 张量物化为 f32 后值不变）。
//   [3] role 门禁：Model 角色写状态段 → 报错；有张量无状态 → 报错。
//   [4] Model 角色文件：状态段长度 0（`peek_train_state` 明确报错）。
//   [5] 尾部校验和（P2-1）：完好文件通过；翻一字节 / 截断 → 明确报错，
//       `load_model` 同步失败（不再静默载入错误权重）。
//   [6] `load_model` 跳过状态段后**参数逐位一致**（读权重不受新段影响）。
//   [7] 数据集身份不一致 → 差异说明列出变化的项（硬报错口径的诊断面）。
//   [8] 老格式回落：把 v8 文件剥成 v7（去掉状态段与尾部、版本改 7）后仍能
//       `load_model`（零破坏），`peek_train_state` 给"自 v8 起才有"的明确错误。
//
// 退出码：0 = 通过，1 = 失败。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/model_serialization.hpp"
#include "neuralnet.cpp/train_state.hpp"

#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

namespace
{
int g_fail = 0;

constexpr std::size_t VOCAB = 64;
constexpr std::size_t D_MODEL = 16;
constexpr std::size_t SEQ_LEN = 8;
constexpr std::size_t HEADS = 2;
constexpr std::size_t D_FF = 32;
constexpr std::size_t LAYERS = 1;

std::filesystem::path scratch_dir()
{
    const auto dir = std::filesystem::temp_directory_path() / "nn_train_state_test";
    std::filesystem::create_directories(dir);
    return dir;
}

[[nodiscard]] nn::ModelSpec tiny_spec()
{
    return nn::make_gpt_spec(VOCAB, D_MODEL, SEQ_LEN, HEADS, D_FF, LAYERS);
}

[[nodiscard]] nn::Result<nn::Model> make_model(nn::ComputeEngine &engine)
{
    return nn::build_gpt_model_from_spec(engine, tiny_spec(), nn::PrecisionProfile{});
}

// ── 参数快照 / 逐位比较（[6] 用；顺序 = parameters()，逐元素 ==）─────────
[[nodiscard]] std::vector<nn::Matrix> snapshot_params(nn::ComputeEngine &engine, nn::Model &m)
{
    std::vector<nn::Matrix> out;
    for (auto &t : m.parameters()) out.push_back(download(engine, t.get()));
    return out;
}

[[nodiscard]] bool same_matrices(const std::vector<nn::Matrix> &a,
                                 const std::vector<nn::Matrix> &b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].rows() != b[i].rows() || a[i].cols() != b[i].cols()) return false;
        const auto sa = a[i].span();
        const auto sb = b[i].span();
        for (std::size_t k = 0; k < sa.size(); ++k)
            if (sa[k] != sb[k]) return false;
    }
    return true;
}

// ── 测试用状态（覆盖每个字段；值刻意非默认，避免"没写也相等"的假通过）──
[[nodiscard]] nn::TrainState sample_state()
{
    nn::TrainState s;
    s.counters.global_step = 1234;
    s.counters.epoch = 3;
    s.counters.step_in_epoch = 42;
    s.counters.lr = 3.5e-4;
    s.recipe.precision.param = nn::Precision::T1_58;
    s.recipe.precision.compute = nn::Precision::F16;
    s.recipe.precision.stable = nn::Precision::F32;
    s.recipe.precision.optimizer = nn::Precision::F16;
    s.recipe.optimizer = "adamw";
    s.recipe.lr_schedule = "step_cosine";
    s.recipe.lr = 1e-3;
    s.recipe.min_lr = 1e-6;
    s.recipe.weight_decay = 0.01;
    s.recipe.beta1 = 0.9;
    s.recipe.beta2 = 0.95;
    s.recipe.max_norm = 1.0;
    s.recipe.epochs = 10;
    s.recipe.total_steps = 4096;
    s.recipe.warmup_epochs = 1;
    s.recipe.warmup_steps = 200;
    s.recipe.batch_size = 8;
    s.recipe.micro_batch_size = 2;
    s.recipe.batch_size_is_global = true;
    s.recipe.accum_steps = 4;
    s.recipe.stride = 128;
    s.recipe.lr_per_epoch = {1e-3, 5e-4, 2.5e-4};
    s.dataset.path = "/data/tiny.nndataset";
    s.dataset.sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    s.dataset.num_tokens = 40960;
    s.dataset.num_docs = 128;
    s.dataset.num_samples = 320;
    s.dataset.steps_per_epoch = 40;
    s.dataset.loss_scope = "assistant";
    s.optimizer_step = 987;
    return s;
}

[[nodiscard]] bool same_state(const nn::TrainState &a, const nn::TrainState &b)
{
    return a.counters.global_step == b.counters.global_step &&
           a.counters.epoch == b.counters.epoch &&
           a.counters.step_in_epoch == b.counters.step_in_epoch &&
           a.counters.lr == b.counters.lr &&
           a.optimizer_step == b.optimizer_step &&
           a.optimizer_blob_count == b.optimizer_blob_count &&
           a.recipe.precision.param == b.recipe.precision.param &&
           a.recipe.precision.compute == b.recipe.precision.compute &&
           a.recipe.precision.stable == b.recipe.precision.stable &&
           a.recipe.precision.optimizer == b.recipe.precision.optimizer &&
           a.recipe.optimizer == b.recipe.optimizer &&
           a.recipe.lr_schedule == b.recipe.lr_schedule &&
           a.recipe.lr == b.recipe.lr && a.recipe.min_lr == b.recipe.min_lr &&
           a.recipe.weight_decay == b.recipe.weight_decay &&
           a.recipe.beta1 == b.recipe.beta1 && a.recipe.beta2 == b.recipe.beta2 &&
           a.recipe.max_norm == b.recipe.max_norm &&
           a.recipe.epochs == b.recipe.epochs &&
           a.recipe.total_steps == b.recipe.total_steps &&
           a.recipe.warmup_epochs == b.recipe.warmup_epochs &&
           a.recipe.warmup_steps == b.recipe.warmup_steps &&
           a.recipe.batch_size == b.recipe.batch_size &&
           a.recipe.micro_batch_size == b.recipe.micro_batch_size &&
           a.recipe.batch_size_is_global == b.recipe.batch_size_is_global &&
           a.recipe.accum_steps == b.recipe.accum_steps &&
           a.recipe.stride == b.recipe.stride &&
           a.recipe.lr_per_epoch == b.recipe.lr_per_epoch &&
           a.dataset.path == b.dataset.path && a.dataset.sha256 == b.dataset.sha256 &&
           a.dataset.num_tokens == b.dataset.num_tokens &&
           a.dataset.num_docs == b.dataset.num_docs &&
           a.dataset.num_samples == b.dataset.num_samples &&
           a.dataset.steps_per_epoch == b.dataset.steps_per_epoch &&
           a.dataset.loss_scope == b.dataset.loss_scope;
}

// ── 宿主文件辅助（测试自用；库内字节 cast 收敛在 core_file.hpp）─────────
[[nodiscard]] std::string read_file(const std::filesystem::path &p)
{
    std::ifstream ifs(p, std::ios::binary);
    return std::string{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path &p, const std::string &bytes)
{
    std::ofstream ofs(p, std::ios::binary);
    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

[[nodiscard]] std::uint64_t rd_u64(const std::string &b, std::size_t off)
{
    std::uint64_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    return v;
}

// ── [1] kvrec 往返（纯编解码，不落盘）──────────────────────────────────
void test_kvrec_roundtrip()
{
    std::cout << "\n── [1] TrainState ⇄ kvrec 往返 ──\n";
    const nn::TrainState s = sample_state();
    const std::string bytes = nn::train_state_to_kv(s).serialize();
    auto kv = nn::KeyValueRecord::parse(bytes);
    CHECK(kv.has_value(), "kvrec 解析成功");
    if (!kv) return;
    auto back = nn::train_state_from_kv(*kv);
    CHECK(back.has_value(), "TrainState 读回成功");
    if (!back) return;
    CHECK(same_state(s, *back), "全字段逐项一致（含精度四槽 / lr_per_epoch / 数据集身份）");

    // 缺键 → 默认值（老文件兜底口径）
    nn::KeyValueRecord empty;
    auto def = nn::train_state_from_kv(empty);
    CHECK(def.has_value() && def->counters.global_step == 0 &&
              def->recipe.precision.param == nn::Precision::F32 &&
              def->recipe.optimizer == "adam",
          "空 kvrec → 默认值（冷启动语义的兜底）");
    // 非法精度值（BF16 保留位）→ 明确报错
    nn::KeyValueRecord bad;
    bad.set("precision_param", static_cast<std::uint64_t>(nn::Precision::BF16));
    auto bad_r = nn::train_state_from_kv(bad);
    CHECK(!bad_r.has_value(), "精度槽为保留值（BF16）→ 报错而不是静默接受");
}

// ── [2][3][4][6][8] 落盘往返与 role 门禁 ───────────────────────────────
void test_checkpoint_roundtrip(const std::filesystem::path &dir)
{
    std::cout << "\n── [2] Checkpoint 往返（状态段 + 优化器张量）──\n";
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    auto model_r = make_model(engine);
    CHECK(model_r.has_value(), "建模型");
    if (!model_r) return;
    nn::Model &model = *model_r;
    const nn::ModelSpec spec = tiny_spec();
    const auto params_before = snapshot_params(engine, model);

    // 优化器状态张量：一个 f32（值可精确比较）+ 一个 f16（验证精度记录）
    nn::Matrix m32(3, 2);
    m32.span()[0] = 0.5f;  m32.span()[1] = 1.5f;  m32.span()[2] = -2.0f;
    m32.span()[3] = 0.25f; m32.span()[4] = 8.0f;  m32.span()[5] = -0.125f;
    nn::Matrix m16(2, 2);
    m16.span()[0] = 0.5f; m16.span()[1] = 1.0f; m16.span()[2] = 2.0f; m16.span()[3] = -3.0f;
    std::vector<nn::Tensor> blobs{
        upload(engine, m32, nn::Precision::F32),
        upload(engine, m16, nn::Precision::F16),
    };

    const std::string tok = "nnvocab-train-state-test";
    nn::TrainState s = sample_state();
    s.optimizer_blob_count = blobs.size();   // 写侧按实参回填，读回应等于它
    nn::CheckpointState ckpt{&s, blobs};
    const auto ckpt_path = dir / "state.nnckpt";
    auto sr = nn::save_model(ckpt_path.string(), model, spec, tok,
                             nn::ModelRole::Checkpoint, ckpt);
    CHECK(sr.has_value(), "写 Checkpoint（含状态段）");
    if (!sr) { std::cout << "     " << sr.error().message << "\n"; return; }

    auto pr = nn::peek_train_state(ckpt_path.string());
    CHECK(pr.has_value(), "peek_train_state 成功");
    if (pr)
    {
        CHECK(same_state(s, *pr), "peek 状态与写入前逐项一致");
        CHECK(pr->optimizer_blob_count == 2, "blob 计数 = 2（写侧按实参回填）");
    }

    auto lr = nn::load_train_state(engine, ckpt_path.string());
    CHECK(lr.has_value(), "load_train_state 成功");
    if (lr)
    {
        CHECK(lr->optimizer_blobs.size() == 2, "读回 2 个张量");
        if (lr->optimizer_blobs.size() == 2)
        {
            const auto b0 = download(engine, lr->optimizer_blobs[0]);
            const auto b1 = download(engine, lr->optimizer_blobs[1]);
            const auto s0 = b0.span();
            const auto s1 = b1.span();
            CHECK(b0.rows() == 3 && b0.cols() == 2 && b0.span()[4] == 8.0f,
                  "f32 张量逐值一致（形状 + 抽样值）");
            CHECK(s1[0] == 0.5f && s1[1] == 1.0f && s1[2] == 2.0f && s1[3] == -3.0f,
                  "f16 张量物化回 f32 后值不变（精确可表示）");
            CHECK(s0.size() == 6, "f32 张量 6 元素");
        }
        CHECK(lr->optimizer_blob_precisions.size() == 2 &&
                  lr->optimizer_blob_precisions[0] == nn::Precision::F32 &&
                  lr->optimizer_blob_precisions[1] == nn::Precision::F16,
              "文件里的每张量精度被如实记录（f32 / f16）");
        CHECK(same_state(s, lr->state), "load_train_state 的状态与写入前一致");
    }

    std::cout << "\n── [6] load_model 跳过状态段后参数逐位一致 ──\n";
    {
        auto back_r = make_model(engine);
        CHECK(back_r.has_value(), "建第二个模型");
        if (back_r)
        {
            auto load_r = nn::load_model(ckpt_path.string(), *back_r);
            CHECK(load_r.has_value(), "load_model 能跳过状态段读权重");
            if (load_r)
            {
                CHECK(*load_r == tok, "内嵌词表字节一致");
                CHECK(same_matrices(params_before, snapshot_params(engine, *back_r)),
                      "参数逐位一致（状态段不影响权重读取）");
            }
        }
    }

    std::cout << "\n── [4] Model 角色：状态段长度 0 ──\n";
    {
        const auto model_path = dir / "state.nnmodel";
        auto mr = nn::save_model(model_path.string(), model, spec, tok, nn::ModelRole::Model);
        CHECK(mr.has_value(), "写 .nnmodel（无状态段）");
        auto pr2 = nn::peek_train_state(model_path.string());
        CHECK(!pr2.has_value(), "peek_train_state 对推理模型明确报错");
        if (!pr2)
            CHECK(pr2.error().message.find("没有训练状态段") != std::string::npos,
                  "错误文案点明'没有训练状态段'");
        // 状态段长度字段为 0（字节级：紧随 spec 之后）
        const std::string bytes = read_file(model_path);
        const std::size_t state_len_off = 10 + 8 + static_cast<std::size_t>(rd_u64(bytes, 10));
        CHECK(rd_u64(bytes, state_len_off) == 0, "字节级：state_len == 0");
    }

    std::cout << "\n── [3] role 门禁 ──\n";
    {
        auto bad_role = nn::save_model((dir / "bad_role.nnmodel").string(), model, spec, tok,
                                       nn::ModelRole::Model, ckpt);
        CHECK(!bad_role.has_value(), "Model 角色携带训练状态段 → 报错");
        if (!bad_role)
            CHECK(bad_role.error().message.find("Checkpoint") != std::string::npos,
                  "错误文案点明只有 Checkpoint 可写状态段");

        nn::CheckpointState no_state{nullptr, blobs};
        auto bad_blobs = nn::save_model((dir / "bad_blobs.nnckpt").string(), model, spec, tok,
                                        nn::ModelRole::Checkpoint, no_state);
        CHECK(!bad_blobs.has_value(), "只有张量没有状态 → 报错");
    }

    std::cout << "\n── [8] 老格式回落（剥成 v7）──\n";
    {
        const std::string bytes = read_file(ckpt_path);
        const std::size_t spec_len_off = 10;
        const std::size_t state_off =
            spec_len_off + 8 + static_cast<std::size_t>(rd_u64(bytes, spec_len_off));
        const std::size_t kv_len = static_cast<std::size_t>(rd_u64(bytes, state_off));
        std::size_t off = state_off + 8 + kv_len;
        for (std::size_t i = 0; i < 2; ++i)   // 两个张量条目：头 34B + aux + payload
        {
            const auto aux_len = static_cast<std::size_t>(rd_u64(bytes, off + 18));
            const auto pay_len = static_cast<std::size_t>(rd_u64(bytes, off + 26));
            off += 34 + aux_len + pay_len;
        }
        std::string v7 = bytes.substr(0, state_off) +
                         bytes.substr(off, bytes.size() - off - nn::MODEL_TAIL_BYTES);
        const std::uint32_t v7_version = 7;
        std::memcpy(&v7[4], &v7_version, sizeof(v7_version));
        const auto v7_path = dir / "legacy_v7.nnckpt";
        write_file(v7_path, v7);

        auto pr7 = nn::peek_train_state(v7_path.string());
        CHECK(!pr7.has_value(), "v7 文件没有状态段 → 明确报错（冷启动语义）");
        if (!pr7)
            CHECK(pr7.error().message.find("v8") != std::string::npos,
                  "错误文案点明'自 v8 起才有'");
        auto v = nn::verify_model_integrity(v7_path.string());
        CHECK(!v.has_value(), "v7 文件（无尾部标记）不参与 v8 校验和校验");

        auto back_r = make_model(engine);
        if (back_r)
        {
            auto load_r = nn::load_model(v7_path.string(), *back_r);
            CHECK(load_r.has_value(), "v7 老文件仍能 load_model（零破坏）");
            if (load_r)
                CHECK(same_matrices(params_before, snapshot_params(engine, *back_r)),
                      "v7 文件读出的参数与原始逐位一致");
        }
    }
}

// ── [5] 尾部校验和（P2-1）──────────────────────────────────────────────
void test_integrity(const std::filesystem::path &dir)
{
    std::cout << "\n── [5] 整文件校验和 / 尾部完整性标记 ──\n";
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    auto model_r = make_model(engine);
    if (!model_r) { CHECK(false, "建模型"); return; }
    const auto path = dir / "integrity.nnmodel";
    auto sr = nn::save_model(path.string(), *model_r, tiny_spec(), "tok", nn::ModelRole::Model);
    CHECK(sr.has_value(), "写文件");
    if (!sr) return;
    CHECK(nn::verify_model_integrity(path.string()).has_value(), "完好文件校验通过");

    const std::string good = read_file(path);
    CHECK(good.size() > nn::MODEL_TAIL_BYTES + 32, "文件尺寸合理");
    CHECK(std::string_view(good.data() + good.size() - nn::MODEL_TAIL_MAGIC.size(),
                           nn::MODEL_TAIL_MAGIC.size()) == nn::MODEL_TAIL_MAGIC,
          "尾部 5 字节 = \"NNEND\"");

    // 翻一字节（中段载荷）
    std::string flipped = good;
    flipped[good.size() / 2] = static_cast<char>(flipped[good.size() / 2] ^ 0xFF);
    const auto flip_path = dir / "integrity_flip.nnmodel";
    write_file(flip_path, flipped);
    auto fv = nn::verify_model_integrity(flip_path.string());
    CHECK(!fv.has_value(), "翻一字节 → 校验和不匹配，明确报错");
    if (!fv)
        CHECK(fv.error().message.find("校验和不匹配") != std::string::npos, "文案点明校验和不匹配");
    {
        auto back = make_model(engine);
        if (back)
            CHECK(!nn::load_model(flip_path.string(), *back).has_value(),
                  "损坏文件 load_model 失败（不再静默载入错误权重）");
    }

    // 截断
    const auto trunc_path = dir / "integrity_trunc.nnmodel";
    write_file(trunc_path, good.substr(0, good.size() - 100));
    auto tv = nn::verify_model_integrity(trunc_path.string());
    CHECK(!tv.has_value(), "截断 → 明确报错");
    if (!tv)
        CHECK(tv.error().message.find("NNEND") != std::string::npos,
              "文案点明缺少尾部完整性标记");
    {
        auto back = make_model(engine);
        if (back)
            CHECK(!nn::load_model(trunc_path.string(), *back).has_value(), "截断文件 load_model 失败");
    }
}

// ── [9] save → load → save 逐字节一致（P1-2 验收口径）───────────────────
// 全部状态张量为 f32 时（默认 profile 的常态）重写必须**逐字节一致**：
// kvrec 字段顺序固定、摘要覆盖确定、参数载荷同精度回写。
// （f16 状态张量经 load 物化为 f32 ⇒ 再 save 会被提升为 f32：值逐位不变、
//  编码变化；生产路径不存在这种再写——save 一律从活状态导出，见 23 §2.1.1。）
void test_resave_byte_identity(const std::filesystem::path &dir)
{
    std::cout << "\n── [9] save → load → save 逐字节一致（f32 状态）──\n";
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    auto model_r = make_model(engine);
    CHECK(model_r.has_value(), "建模型");
    if (!model_r) return;
    nn::Model &model = *model_r;

    nn::Matrix m(2, 3);
    m.span()[0] = 0.125f; m.span()[1] = -1.5f; m.span()[2] = 3.0f;
    m.span()[3] = -0.25f; m.span()[4] = 7.5f;  m.span()[5] = 0.5f;
    std::vector<nn::Tensor> blobs{upload(engine, m, nn::Precision::F32)};

    nn::TrainState st = sample_state();
    st.optimizer_blob_count = blobs.size();
    const std::string tok = "nnvocab-resave";
    const auto path1 = dir / "resave1.nnckpt";
    const auto path2 = dir / "resave2.nnckpt";
    CHECK(nn::save_model(path1.string(), model, tiny_spec(), tok, nn::ModelRole::Checkpoint,
                         nn::CheckpointState{&st, blobs}).has_value(),
          "写第一份断点");

    auto loaded = nn::load_train_state(engine, path1.string());
    CHECK(loaded.has_value(), "读回状态 + 状态张量");
    if (loaded)
    {
        nn::TrainState back = loaded->state;
        CHECK(nn::save_model(path2.string(), model, tiny_spec(), tok,
                             nn::ModelRole::Checkpoint,
                             nn::CheckpointState{&back, loaded->optimizer_blobs}).has_value(),
              "用读回的内容重写第二份断点");
        const std::string b1 = read_file(path1);
        const std::string b2 = read_file(path2);
        CHECK(b1.size() == b2.size() && b1 == b2,
              "两份文件逐字节一致（save → load → save）");
        CHECK(b1.size() > nn::MODEL_TAIL_BYTES, "文件含尾部标记");
    }
}

// ── [7] 数据集身份差异说明（诊断面）────────────────────────────────────
void test_dataset_identity()
{
    std::cout << "\n── [7] 数据集身份校验 ──\n";
    const auto saved = sample_state().dataset;
    CHECK(nn::dataset_mismatch_reason(saved, saved).empty(), "同一身份 → 无差异");

    auto changed = saved;
    changed.sha256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    changed.num_samples = saved.num_samples + 1;
    changed.steps_per_epoch = saved.steps_per_epoch + 1;
    const std::string why = nn::dataset_mismatch_reason(saved, changed);
    CHECK(!why.empty(), "身份变化 → 有差异说明");
    CHECK(why.find("sha256") != std::string::npos, "列出 sha256 变化");
    CHECK(why.find("样本数") != std::string::npos, "列出样本数变化");
    CHECK(why.find("每 epoch 步数") != std::string::npos, "列出每 epoch 步数变化");
    CHECK(why.find("token 数") == std::string::npos, "未变化的项不列出");
}
} // namespace

int main()
{
    std::printf("train_state_test（v8 训练状态段 + 尾部校验和；P1-2 / P2-1）\n");
    const auto dir = scratch_dir();

    test_kvrec_roundtrip();
    test_checkpoint_roundtrip(dir);
    test_integrity(dir);
    test_resave_byte_identity(dir);
    test_dataset_identity();

    std::filesystem::remove_all(dir);   // 只清自己造的临时目录
    if (g_fail == 0)
    {
        std::printf("[结果] 全部通过\n");
        return 0;
    }
    std::printf("[结果] %d 项失败\n", g_fail);
    return 1;
}
