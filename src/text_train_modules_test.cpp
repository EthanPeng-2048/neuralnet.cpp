// ── text_train_modules_test — P1-1 拆出的模块（recovery / checkpoint）回归 ──
//
// 验收（docs/internal/18-roadmap.md §4.2 P1-1）："新增恢复路径单测"。
// 覆盖：
//   [1] 断点路径派生：`--save` 的 .nnmodel → 同名 .nnckpt；非 .nnmodel 追加。
//   [2] 恢复路径的角色语义：Checkpoint = 断点续训、Model = warm start（都放行）；
//       文件不存在/不可读放行（交给 peek_model_spec 的"从头训练"语义）。
//       **P1-7 起没有需要拒绝的角色组合**：三值（T1_58）`.nnmodel` 的打包载荷
//       在载入时按文件 τ 反量化成 f32 latent ⇒ 首次 forward 与推理文件逐位一致、
//       随后 8 步 Adam 参数确实移动（本节直接覆盖这条）。
//   [3] 断点往返（恢复路径主用例）：保存 Checkpoint → `resume_text_train`
//       按文件规格重建模型并载参 → 参数与保存前**逐位一致**、内嵌词表字节数一致。
//   [4] 断点记账与退出码语义（P1-1 第一刀的回归锁）：写失败只记账不终止
//       （failures=1、错误行带"第 N 次"），收尾退出码 = 1；成功路径不计账、
//       退出码 0 且不打印。
//   [5] 推理模型产出（Model 角色）：`save_text_train_model` 写出的文件可被
//       `peek_model_role` 认成 Model（text_infer 侧的前提）。
//
// 退出码：0 = 通过，1 = 失败。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/cli/text_train_checkpoint.hpp"
#include "neuralnet.cpp/cli/text_train_config.hpp"
#include "neuralnet.cpp/cli/text_train_loop.hpp"   // save_ckpt_with_state（P1-2 接线）
#include "neuralnet.cpp/cli/text_train_recovery.hpp"

#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

namespace
{
int g_fail = 0;

// 小 GPT（CPU，秒级）：恢复路径的往返只要"能建 → 能存 → 能读回逐位相同"。
constexpr std::size_t VOCAB = 64;
constexpr std::size_t D_MODEL = 16;
constexpr std::size_t SEQ_LEN = 8;
constexpr std::size_t HEADS = 2;
constexpr std::size_t D_FF = 32;
constexpr std::size_t LAYERS = 1;
constexpr const char* kTokenizerBytes = "nnvocab-test-bytes";   // 原样嵌出/读回

std::filesystem::path scratch_dir()
{
    const auto dir = std::filesystem::temp_directory_path() / "nn_text_train_modules_test";
    std::filesystem::create_directories(dir);
    return dir;
}

[[nodiscard]] nn::ModelSpec tiny_spec()
{
    return nn::make_gpt_spec(VOCAB, D_MODEL, SEQ_LEN, HEADS, D_FF, LAYERS);
}

// 按参数顺序拍参数快照（宿主侧 f32；用于"参数确实移动"的断言）
[[nodiscard]] std::vector<nn::Scalar> params_snapshot(nn::ComputeEngine& eng,
                                                      nn::Model& m)
{
    std::vector<nn::Scalar> out;
    for (auto& t : m.parameters())
    {
        auto mat = eng.to_matrix(t.get());
        if (!mat) { out.push_back(-1e30f); continue; }
        for (auto v : mat->span()) out.push_back(v);
    }
    return out;
}

// 按参数顺序逐元素比较两个模型（同引擎、逐位相等；顺序由 parameters() 决定）
[[nodiscard]] bool params_identical(nn::ComputeEngine& eng, nn::Model& a, nn::Model& b)
{
    auto pa = a.parameters();
    auto pb = b.parameters();
    if (pa.size() != pb.size()) return false;
    for (std::size_t i = 0; i < pa.size(); ++i)
    {
        auto ma = eng.to_matrix(pa[i].get());
        auto mb = eng.to_matrix(pb[i].get());
        if (!ma || !mb) return false;
        auto sa = ma->span();
        auto sb = mb->span();
        if (sa.size() != sb.size()) return false;
        for (std::size_t k = 0; k < sa.size(); ++k)
            if (sa[k] != sb[k]) return false;
    }
    return true;
}

// ── [1] 断点路径派生 ───────────────────────────────────────────────────
void test_ckpt_path()
{
    CHECK(nn::cli::text_train_ckpt_path("gpt_model.nnmodel") == "gpt_model.nnckpt",
          ".nnmodel → 同名 .nnckpt");
    CHECK(nn::cli::text_train_ckpt_path("out/dir/m.nnmodel") == "out/dir/m.nnckpt",
          "带目录的 .nnmodel → 同名 .nnckpt");
    CHECK(nn::cli::text_train_ckpt_path("m.bin") == "m.bin.nnckpt",
          "非 .nnmodel 路径 → 追加 .nnckpt");
    CHECK(nn::cli::text_train_ckpt_path("x.nnckpt") == "x.nnckpt.nnckpt",
          "已是 .nnckpt 也追加（换扩展名只认 .nnmodel）");
    CHECK(nn::cli::text_train_ckpt_path(".nnmodel") == ".nnmodel.nnckpt",
          "裸扩展名路径不切割（退化输入不产生空名）");
}

// ── [2] 角色门禁 + [3] 断点往返 ────────────────────────────────────────
void test_role_gate_and_roundtrip(const std::filesystem::path& dir)
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const nn::ModelSpec spec = tiny_spec();
    auto built = nn::build_gpt_model_from_spec(eng, spec);
    NN_ASSERT_ALWAYS(built, "text_train_modules_test: 构建小 GPT 失败");
    nn::Model original = std::move(*built);

    const std::string model_file = (dir / "gate.nnmodel").string();
    const std::string ckpt_file = (dir / "gate.nnckpt").string();
    NN_ASSERT_ALWAYS(nn::cli::save_text_train_model(model_file, original, spec,
                                                    kTokenizerBytes),
                     "text_train_modules_test: 写推理模型失败");
    nn::cli::TextTrainCheckpointLog log;
    nn::TextTrainState ckpt_state;   // P1-2：断点带训练状态段（本用例不挂优化器状态张量）
    ckpt_state.counters.epoch = 1;
    ckpt_state.recipe.optimizer = "adam";
    ckpt_state.recipe.batch_size = 4;
    ckpt_state.recipe.micro_batch_size = 4;
    ckpt_state.recipe.batch_size_is_global = true;
    ckpt_state.recipe.accum_steps = 1;
    const std::vector<nn::Tensor> no_blobs;
    log.save(ckpt_file, original, spec, kTokenizerBytes, ckpt_state, no_blobs, std::cerr);
    CHECK(log.failures() == 0, "断点写到可写路径不应记账失败");

    // 角色语义（v8，2026-10-16 裁定；docs/internal/23 §2.1.1）：
    //   Model（.nnmodel）= warm start（SFT）⇒ **放行**（三值同样放行，见下面的
    //     P1-7 反量化块）；Checkpoint / 老文件 / 不存在 ⇒ 放行。
    //   **P1-7 已删除**"三值 .nnmodel 必须被拒"那条门禁（`check_resume_target`
    //   随之删除）：打包载荷现在按文件 τ 反量化成可训练 latent。

    // ── P1-2 第三刀 B：禁列（按旗标名）+ 状态采纳 + 数据集身份门禁 ────────
    {
        // 禁列谓词：A/B 类在列，C 类不在列（解析层据此登记）
        CHECK(nn::cli::is_forbidden_resume_flag("--lr") &&
                  nn::cli::is_forbidden_resume_flag("--epochs") &&
                  nn::cli::is_forbidden_resume_flag("--batch-size") &&
                  nn::cli::is_forbidden_resume_flag("--f16") &&
                  nn::cli::is_forbidden_resume_flag("--t1_58") &&
                  nn::cli::is_forbidden_resume_flag("--resume-step"),
              "A/B 类旗标必须在禁列（lr / epochs / batch-size / f16 / t1_58 / resume-step）");
        CHECK(!nn::cli::is_forbidden_resume_flag("--save") &&
                  !nn::cli::is_forbidden_resume_flag("--save-interval") &&
                  !nn::cli::is_forbidden_resume_flag("--log-interval") &&
                  !nn::cli::is_forbidden_resume_flag("--max-steps") &&
                  !nn::cli::is_forbidden_resume_flag("--gpu") &&
                  !nn::cli::is_forbidden_resume_flag("--grad-log") &&
                  !nn::cli::is_forbidden_resume_flag("--checkpoint-every") &&
                  !nn::cli::is_forbidden_resume_flag("--activation-offload") &&
                  !nn::cli::is_forbidden_resume_flag("--flush-interval"),
              "C 类旗标不得进禁列（只允许完全不改变训练的输入）");

        // 解析层登记（不看值：给了 --lr 且值与断点相同也登记）
        {
            std::vector<std::string> args{"text_train", "/tmp/ds.nndataset", "--lr", "0.001",
                                          "--epochs", "3", "--save", "/tmp/m.nnmodel"};
            std::vector<char*> argv;
            argv.reserve(args.size());
            for (std::string& a : args) argv.push_back(a.data());
            auto cfg_p = nn::cli::parse_text_train_args(static_cast<int>(argv.size()),
                                                        argv.data());
            CHECK(cfg_p.explicit_train_flags.size() == 2 &&
                      cfg_p.explicit_train_flags[0] == "--lr" &&
                      cfg_p.explicit_train_flags[1] == "--epochs",
                  "解析层按旗标名登记 A/B 类输入（--save 不在列）");
        }

        // 状态采纳：带状态段的断点 ⇒ adopted，且 cfg 被改写
        nn::cli::TrainConfig cfg2;
        cfg2.resume_path = ckpt_file;
        cfg2.load_existing = true;
        std::ostringstream out2, err2;
        auto ar = nn::cli::adopt_resume_state(cfg2, out2, err2);
        CHECK(ar.has_value() && ar->adopted, "带状态段的断点 ⇒ 采纳状态段");
        if (ar)
            CHECK(ar->state.counters.epoch == 1 && cfg2.start_epoch == 1 &&
                      cfg2.optimizer_name == "adam" && ar->shape_adopted,
                  "采纳后 cfg 的起点/配方/形状由断点决定");

        // 显式给出 A/B 类旗标 ⇒ 报错（禁列）
        nn::cli::TrainConfig cfg3;
        cfg3.resume_path = ckpt_file;
        cfg3.load_existing = true;
        cfg3.explicit_train_flags.push_back("--lr");
        std::ostringstream out3, err3;
        auto ar3 = nn::cli::adopt_resume_state(cfg3, out3, err3);
        CHECK(!ar3.has_value(), "断点续训时显式给出 --lr 必须报错");
        if (!ar3)
            CHECK(ar3.error().message.find("--lr") != std::string::npos &&
                      ar3.error().message.find("禁止") != std::string::npos,
                  "报错文案应列出被禁旗标与规则");

        // docs/internal/30：旧断点缺 batch 语义键 ⇒ micro-batch 解释，
        // 恢复时换成等效 global，并把 micro step 计数折成 update step。
        nn::TextTrainState legacy_state;
        legacy_state.counters.epoch = 1;
        legacy_state.counters.step_in_epoch = 3;
        legacy_state.recipe.optimizer = "adam";
        legacy_state.recipe.batch_size = 2;
        legacy_state.recipe.accum_steps = 2;
        legacy_state.recipe.epochs = 3;
        legacy_state.dataset.num_samples = 10;
        legacy_state.dataset.steps_per_epoch = 5;
        const std::string legacy_ckpt = (dir / "legacy.nnckpt").string();
        nn::cli::TextTrainCheckpointLog legacy_log;
        legacy_log.save(legacy_ckpt, original, spec, kTokenizerBytes, legacy_state,
                        no_blobs, std::cerr);
        nn::cli::TrainConfig legacy_cfg;
        legacy_cfg.resume_path = legacy_ckpt;
        legacy_cfg.load_existing = true;
        std::ostringstream legacy_out, legacy_err;
        auto legacy_r = nn::cli::adopt_resume_state(legacy_cfg, legacy_out, legacy_err);
        CHECK(legacy_r && legacy_cfg.batch_size == 4 &&
                  legacy_cfg.accum_steps == 2 && legacy_cfg.start_step == 1,
              "旧 micro-batch 断点 → global=4 / accum=2 / update step=1");
        if (legacy_r)
            CHECK(legacy_r->state.recipe.batch_size_is_global &&
                      legacy_r->state.recipe.micro_batch_size == 2 &&
                      legacy_r->state.dataset.steps_per_epoch == 3,
                  "旧断点状态已转换为 global batch 语义");

        // 数据集身份门禁（硬错、无逃生阀）
        nn::DatasetIdentity a;
        a.path = "/data/a.nndataset";
        a.sha256 = std::string(64, 'a');
        a.num_tokens = 100;
        a.num_docs = 2;
        a.num_samples = 4;
        a.steps_per_epoch = 2;
        a.loss_scope = "all";
        CHECK(nn::cli::verify_dataset_identity(a, a, "x").has_value(), "同一身份放行");
        nn::DatasetIdentity b = a;
        b.steps_per_epoch = 3;
        b.sha256 = std::string(64, 'b');
        auto dv = nn::cli::verify_dataset_identity(a, b, "x");
        CHECK(!dv.has_value(), "网格/身份变化 ⇒ 硬报错");
        if (!dv)
            CHECK(dv.error().message.find("每 epoch 步数") != std::string::npos &&
                      dv.error().message.find(".nnmodel") != std::string::npos,
                  "报错文案列出差异项并指明出口（.nnmodel 开新一轮）");
    }

    // 断点往返：空模型 → 由文件规格重建 + 载参
    nn::cli::TrainConfig cfg;
    cfg.resume_path = ckpt_file;
    cfg.load_existing = true;
    nn::ModelSpec spec_out;   // 由恢复流程覆盖
    nn::Model restored;       // 默认构造（规格来自文件）
    std::ostringstream out;
    std::ostringstream err;
    auto r = nn::cli::resume_text_train(eng, cfg, restored, spec_out, out, err);
    CHECK(r.has_value(), "从 Checkpoint 断点恢复应成功");
    CHECK(err.str().empty(), "成功恢复不应打印错误");
    if (r)
    {
        CHECK(r->spec_restored, "恢复流程应从文件规格重建模型");
        CHECK(r->weights_loaded, "恢复流程应载入参数");
        CHECK(r->vocab_bytes == std::string(kTokenizerBytes).size(),
              "内嵌词表字节数应与保存时一致");
    }
    CHECK(out.str().find("已加载模型: " + ckpt_file) != std::string::npos,
          "恢复成功应打印已加载模型路径");
    CHECK(params_identical(eng, original, restored),
          "恢复后的参数应与保存前逐位一致");

    // 角色语义变更后（2026-10-16 裁定）的反例形态：dense `.nnmodel` ⇒ **warm start**
    // 走通整条恢复流程（SFT 入口）；三值 `.nnmodel` 见下面的 P1-7 反量化块。
    nn::cli::TrainConfig warm_cfg;
    warm_cfg.resume_path = model_file;
    warm_cfg.load_existing = true;
    nn::ModelSpec spec_out2;
    nn::Model m2;
    std::ostringstream out2, err2;
    auto warm = nn::cli::resume_text_train(eng, warm_cfg, m2, spec_out2, out2, err2);
    CHECK(warm.has_value(), "dense 推理模型应能走通恢复流程（warm start）");
    CHECK(err2.str().empty(), "warm start 不应在 stderr 上留痕（错误才经 Result/stderr）");
    if (warm)
        CHECK(params_identical(eng, original, m2),
              "warm start 载入的参数应与写出前逐位一致");

    // ── P1-7：三值（T1_58）`.nnmodel` 的续训入口（打包 → latent 反量化）──────
    // 三值推理文件的载荷是打包形态（没有 latent，部署态 `backward()` 会报错）。
    // `resume_text_train` 以**可训练意图**载入：序列化层按文件 τ 反量化成 f32
    // latent（量化器不动点，见 `ternary_pack.hpp` 的 `latent_row_scale`），层再按
    // 文件 τ 播种量化缓存 ⇒ 首次 forward 与推理文件里的模型**逐位一致**。
    {
        nn::ModelSpec t_spec = tiny_spec();
        t_spec.as<nn::GptSpec>()->weight_quant = nn::WeightQuant::T1_58;
        const nn::PrecisionProfile t1 = nn::profile_t1_58();
        auto src_r = nn::build_gpt_model_from_spec(eng, t_spec, t1);
        NN_ASSERT_ALWAYS(src_r, "text_train_modules_test: 构建三值小 GPT 失败");
        nn::Model src = std::move(*src_r);

        const std::string t_file = (dir / "ternary.nnmodel").string();
        NN_ASSERT_ALWAYS(nn::cli::save_text_train_model(t_file, src, t_spec,
                                                       kTokenizerBytes),
                         "text_train_modules_test: 写三值推理模型失败");
        {
            auto role = nn::peek_model_role(t_file);
            CHECK(role.has_value() && *role == nn::ModelRole::Model,
                  "三值文件的角色仍是 Model（.nnmodel）");
            auto peeked = nn::peek_model_spec(t_file);
            CHECK(peeked.has_value() &&
                      peeked->as<nn::GptSpec>()->weight_quant == nn::WeightQuant::T1_58,
                  "文件规格记住 weight_quant=t1_58");
        }

        nn::cli::TrainConfig t_cfg;
        t_cfg.resume_path = t_file;
        t_cfg.load_existing = true;
        // 精度 profile 由命令行给出（Model 角色 = CLI 全权，文件只记 weight_quant）：
        // 用它跟文件里那个模型对齐，才能要求"首次 forward 逐位一致"。
        t_cfg.precision = t1;
        nn::ModelSpec t_spec_out;
        nn::Model t_model;
        std::ostringstream t_out, t_err;
        auto t_res =
            nn::cli::resume_text_train(eng, t_cfg, t_model, t_spec_out, t_out, t_err);
        CHECK(t_res.has_value(), "三值 .nnmodel 应能按 warm start 载入（P1-7：不再被拒）");
        if (t_res.has_value())
        {
            CHECK(t_res->weights_loaded, "三值 warm start 应载入参数");
            CHECK(t_res->spec_restored, "三值 warm start 应按文件规格重建（BitLinear）");
            CHECK(t_out.str().find("反量化") != std::string::npos,
                  "提示应说明打包权重按文件 τ 反量化成 latent");
            CHECK(t_err.str().empty(), "三值 warm start 顺利时不应在 stderr 留痕");
        }

        // 固定输入：token IDs (seq_len, batch)
        constexpr std::size_t BATCH = 2;
        nn::Matrix x_m(SEQ_LEN, BATCH);
        for (std::size_t i = 0; i < x_m.size(); ++i)
            x_m.span()[i] = static_cast<nn::Scalar>(i % VOCAB);
        const nn::Tensor x_t = upload(eng, x_m);

        // 首次 forward：与写出前的模型逐位一致（wq/τ 播种自文件，不经重算）
        auto y_src = src.forward(x_t);
        auto y_warm = t_model.forward(x_t);
        CHECK(y_src.has_value() && y_warm.has_value(), "三值两侧 forward 成功");
        if (y_src.has_value() && y_warm.has_value())
        {
            const nn::Matrix a = download(eng, *y_src);
            const nn::Matrix b = download(eng, *y_warm);
            bool same = (a.rows() == b.rows() && a.cols() == b.cols());
            for (std::size_t i = 0; same && i < a.size(); ++i)
                same = (a.span()[i] == b.span()[i]);
            CHECK(same, "同精度口径下三值 warm start 首次 forward 与推理文件**逐位一致**");
        }

        // 不给精度旗标也能载入（BitLinear 由文件规格确定；其余槽走 CLI 默认）
        {
            nn::cli::TrainConfig d_cfg;
            d_cfg.resume_path = t_file;
            d_cfg.load_existing = true;
            nn::ModelSpec d_spec_out;
            nn::Model d_model;
            std::ostringstream d_out, d_err;
            auto d_res = nn::cli::resume_text_train(eng, d_cfg, d_model, d_spec_out, d_out,
                                                    d_err);
            CHECK(d_res.has_value(), "不给精度旗标也应能载入三值 warm start（规格定 BitLinear）");
            if (d_res.has_value())
            {
                auto y_d = d_model.forward(x_t);
                CHECK(y_d.has_value(), "默认精度口径下三值 warm start 也能前向");
                auto g_d = d_model.zero_grad();
                CHECK(g_d.has_value(), "默认精度口径下 zero_grad 成功");
            }
        }

        // 8 步 Adam：backward 可跑（不是只读部署态）、参数确实移动（P1-7 验收）
        auto opt = nn::create_optimizer("adam", eng, t_model.parameters(),
                                       t_model.param_gradients(), 1e-2f, 0.0f, t1);
        NN_ASSERT_ALWAYS(opt != nullptr, "text_train_modules_test: 建三值优化器失败");
        const std::vector<nn::Scalar> before = params_snapshot(eng, t_model);

        nn::CrossEntropyLoss ce;
        ce.set_precision_profile(t1);
        nn::Matrix tgt_m(VOCAB, SEQ_LEN * BATCH, nn::Scalar{0});
        for (std::size_t i = 0; i < SEQ_LEN * BATCH; ++i)
            tgt_m.set_value(i % VOCAB, i, 1.0f);
        const nn::Tensor tgt_t = upload(eng, tgt_m);

        bool steps_ok = true;
        for (int step = 0; step < 8 && steps_ok; ++step)
        {
            steps_ok = t_model.zero_grad().has_value();
            auto logits = t_model.forward(x_t);
            if (!logits) { steps_ok = false; break; }
            auto loss = ce.forward(eng, *logits, tgt_t);
            if (!loss) { steps_ok = false; break; }
            auto grad = ce.backward();
            if (!grad) { steps_ok = false; break; }
            if (!t_model.backward(*grad)) { steps_ok = false; break; }
            if (!opt->step()) { steps_ok = false; break; }
        }
        CHECK(steps_ok,
              "三值 warm start 后 8 步 Adam（forward/loss/backward/step）全部成功");
        const std::vector<nn::Scalar> after = params_snapshot(eng, t_model);
        std::size_t moved = 0, total = 0;
        for (std::size_t i = 0; i < before.size() && i < after.size(); ++i)
        {
            ++total;
            if (before[i] != after[i]) ++moved;
        }
        std::printf("    三值 warm start 8 步：移动参数 %zu/%zu\n", moved, total);
        CHECK(moved > 0, "三值 warm start 后 8 步 Adam 参数确实移动");
    }
}

// ── [4] 断点记账与退出码语义（P1-1 第一刀回归锁）────────────────────────
void test_ckpt_accounting(const std::filesystem::path& dir)
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const nn::ModelSpec spec = tiny_spec();
    auto built = nn::build_gpt_model_from_spec(eng, spec);
    NN_ASSERT_ALWAYS(built, "text_train_modules_test: 构建小 GPT 失败");
    nn::Model model = std::move(*built);

    // 失败路径：断点路径被目录占用 ⇒ 写入必然失败（只记账，不终止）
    const auto occupied = dir / "occupied.nnckpt";
    std::filesystem::create_directories(occupied);
    nn::cli::TextTrainCheckpointLog log;
    std::ostringstream err;
    nn::TextTrainState ckpt_state;
    const std::vector<nn::Tensor> no_blobs;
    log.save(occupied.string(), model, spec, kTokenizerBytes, ckpt_state, no_blobs, err);
    CHECK(log.failures() == 1, "写失败的断点必须计入 1 次");
    CHECK(err.str().find("[ckpt] 保存失败") != std::string::npos, "错误行应带 [ckpt] 标记");
    CHECK(err.str().find("第 1 次") != std::string::npos, "错误行应带第几次（P1-1 文案）");

    std::ostringstream err_code;
    CHECK(nn::cli::text_train_ckpt_exit_code(log.failures(), err_code) == 1,
          "有断点写失败时收尾退出码必须为 1");
    CHECK(err_code.str().find("退出码 1") != std::string::npos,
          "收尾错误行应说明退出码");

    // 成功路径：不增计数、退出码 0、不打印
    log.save((dir / "ok.nnckpt").string(), model, spec, kTokenizerBytes, ckpt_state,
             no_blobs, err);
    CHECK(log.failures() == 1, "成功保存不应增加失败计数");
    std::ostringstream err_ok;
    CHECK(nn::cli::text_train_ckpt_exit_code(0, err_ok) == 0, "无失败时退出码必须为 0");
    CHECK(err_ok.str().empty(), "无失败时不应打印");
}

// ── [6] P1-2 断点状态接线：save_ckpt_with_state → peek → load_optimizer_state ──
void test_train_state_wiring(const std::filesystem::path& dir)
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const nn::ModelSpec spec = tiny_spec();
    auto built = nn::build_gpt_model_from_spec(eng, spec, nn::PrecisionProfile{});
    NN_ASSERT_ALWAYS(built, "test_train_state_wiring: 构建小 GPT 失败");
    nn::Model model = std::move(*built);

    auto opt = nn::create_optimizer("adam", eng, model.parameters(),
                                    model.param_gradients(), 1e-3f, 0.0f,
                                    nn::PrecisionProfile{}, 0.9f, 0.999f);
    NN_ASSERT_ALWAYS(opt != nullptr, "test_train_state_wiring: 建优化器失败");
    NN_ASSERT_ALWAYS(opt->step(), "test_train_state_wiring: 一步失败");

    nn::cli::TrainConfig cfg;
    cfg.save_path = (dir / "wire.nnmodel").string();
    cfg.optimizer_name = "adam";
    cfg.lr = 1e-3f;
    cfg.epochs = 3;
    cfg.batch_size = 2;
    cfg.accum_steps = 1;
    cfg.seq_len = SEQ_LEN;
    nn::DatasetIdentity id;
    id.path = "/tmp/wire.nndataset";
    id.sha256 = std::string(64, 'c');
    id.num_tokens = 64;
    id.num_docs = 2;
    id.num_samples = 4;
    id.loss_scope = "all";

    const auto path = dir / "wire.nnckpt";
    std::ostringstream err;
    nn::cli::TextTrainCheckpointLog log;
    // P2b：状态装配唯一实现 = nn::make_text_train_state（TextTrainer 内部同一函数）
    const nn::TextTrainState state = nn::make_text_train_state(
        cfg, id, /*steps_per_epoch=*/2, /*epoch=*/1, /*step_done=*/2, 1e-3,
        opt->state_step());
    nn::cli::save_ckpt_with_state(path.string(), model, spec, kTokenizerBytes, state, *opt,
                                  log, err);
    CHECK(log.failures() == 0 && err.str().empty(), "带状态的断点应一次写成功");

    auto st = nn::peek_train_state(path.string());
    CHECK(st.has_value(), "peek_train_state 应成功");
    if (st)
    {
        auto* ts = dynamic_cast<nn::TextTrainState*>(st->get());
        CHECK(ts != nullptr, "peek 得到文本域状态（TextTrainState）");
        if (ts)
        {
            // step_done == steps_per_epoch ⇒ epoch 边界归一化（下一步 = epoch 2 / step 0）
            CHECK(ts->counters.epoch == 2 && ts->counters.step_in_epoch == 0 &&
                      ts->counters.global_step == 4,
                  "计数器记'下一步的位置'且 epoch 边界归一化");
            CHECK(ts->optimizer_step == 1, "Adam 的 t 随步推进（1 步后 = 1）");
            CHECK(ts->optimizer_blob_count == 2 * model.parameters().size(),
                  "优化器状态张量数 = 2 × 参数数（m/v 交替）");
            CHECK(ts->recipe.batch_size == 2 && ts->recipe.epochs == 3 &&
                      ts->recipe.total_steps == 6 && ts->recipe.optimizer == "adam",
                  "配方随断点落盘（batch / epochs / 曲线形状 / 优化器名）");
            CHECK(ts->dataset.sha256 == id.sha256 && ts->dataset.steps_per_epoch == 2 &&
                      ts->dataset.num_samples == 4,
                  "数据集身份 + 网格随断点落盘");

            // 数据集身份门禁走的就是断点里的这份身份
            nn::DatasetIdentity changed = id;
            changed.num_samples = 5;
            CHECK(!nn::cli::verify_dataset_identity(ts->dataset, changed, path.string())
                      .has_value(),
                  "断点记录的身份 vs 变化的网格 ⇒ 硬报错（接线一致）");
        }
    }

    // 重启：新优化器（状态零初始化）→ 载入断点状态 → t 与张量恢复
    auto opt2 = nn::create_optimizer("adam", eng, model.parameters(),
                                     model.param_gradients(), 1e-3f, 0.0f,
                                     nn::PrecisionProfile{}, 0.9f, 0.999f);
    NN_ASSERT_ALWAYS(opt2 != nullptr, "test_train_state_wiring: 重建优化器失败");
    CHECK(opt2->state_step() == 0, "新优化器的步计数器从 0 起（前置条件）");
    std::ostringstream out2;
    auto lr = nn::cli::load_optimizer_state(eng, path.string(), *opt2, out2);
    CHECK(lr.has_value(), "load_optimizer_state 应成功");
    CHECK(opt2->state_step() == 1, "载入后 Adam 的 t 恢复（偏置修正计数不丢）");
    CHECK(out2.str().find("已恢复优化器状态") != std::string::npos,
          "载入成功应打印一行进度");
}

// ── [5] 推理模型角色（text_infer 侧前提）───────────────────────────────
void test_inference_model_role(const std::filesystem::path& dir)
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const nn::ModelSpec spec = tiny_spec();
    auto built = nn::build_gpt_model_from_spec(eng, spec);
    NN_ASSERT_ALWAYS(built, "text_train_modules_test: 构建小 GPT 失败");
    nn::Model model = std::move(*built);

    const std::string path = (dir / "infer.nnmodel").string();
    auto saved = nn::cli::save_text_train_model(path, model, spec, kTokenizerBytes);
    CHECK(saved.has_value(), "推理模型应写出成功");
    auto role = nn::peek_model_role(path);
    CHECK(role.has_value() && *role == nn::ModelRole::Model,
          "写出的文件角色必须是 Model（.nnmodel）");
}
} // namespace

int main()
{
    std::printf("text_train_modules_test（P1-1：recovery / checkpoint 模块）\n");
    const auto dir = scratch_dir();

    test_ckpt_path();
    test_role_gate_and_roundtrip(dir);
    test_train_state_wiring(dir);
    test_ckpt_accounting(dir);
    test_inference_model_role(dir);

    std::filesystem::remove_all(dir);   // 只清自己造的临时目录
    if (g_fail == 0)
    {
        std::printf("[结果] 全部通过\n");
        return 0;
    }
    std::printf("[结果] %d 项失败\n", g_fail);
    return 1;
}
