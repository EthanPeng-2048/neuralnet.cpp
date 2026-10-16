// ── text_train_modules_test — P1-1 拆出的模块（recovery / checkpoint）回归 ──
//
// 验收（docs/todo/18-roadmap.md §4.2 P1-1）："新增恢复路径单测"。
// 覆盖：
//   [1] 断点路径派生：`--save` 的 .nnmodel → 同名 .nnckpt；非 .nnmodel 追加。
//   [2] 恢复路径的角色门禁：Model 角色（.nnmodel）明确拒绝；Checkpoint 放行；
//       文件不存在/不可读放行（交给 peek_model_spec 的"从头训练"语义）。
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
    nn::CpuEngine eng;
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
    nn::TrainState ckpt_state;   // P1-2：断点带训练状态段（本用例不挂优化器状态张量）
    ckpt_state.counters.epoch = 1;
    ckpt_state.recipe.optimizer = "adam";
    const std::vector<nn::Tensor> no_blobs;
    log.save(ckpt_file, original, spec, kTokenizerBytes, ckpt_state, no_blobs, std::cerr);
    CHECK(log.failures() == 0, "断点写到可写路径不应记账失败");

    // 角色语义（v8，2026-10-16 裁定；docs/todo/23 §2.1.1）：
    //   Model（.nnmodel）= warm start（SFT）⇒ **放行**；
    //   Checkpoint / 老文件 / 不存在 ⇒ 放行；
    //   唯一拒绝 = 三值（T1_58）的 .nnmodel（打包形态无 latent ⇒ 指向 P1-7）。
    CHECK(nn::cli::check_resume_target(model_file, spec),
          "dense 推理模型（.nnmodel）必须放行 —— SFT warm start 入口");
    CHECK(nn::cli::check_resume_target(ckpt_file, spec), "Checkpoint 角色必须放行");
    CHECK(nn::cli::check_resume_target((dir / "missing.nnckpt").string(), spec),
          "不存在的断点应放行（由载入提示走'从头训练'语义）");
    {
        nn::ModelSpec ternary_spec = spec;
        ternary_spec.weight_quant = nn::WeightQuant::T1_58;
        auto gate_t158 = nn::cli::check_resume_target(model_file, ternary_spec);
        CHECK(!gate_t158, "三值（T1_58）的 .nnmodel 必须被拒绝（打包形态无 latent）");
        CHECK(!gate_t158 &&
                  gate_t158.error().message.find("P1-7") != std::string::npos &&
                  gate_t158.error().message.find(".nnckpt") != std::string::npos,
              "拒绝文案应指向 P1-7 与 .nnckpt");
    }

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

        // 数据集身份门禁（硬错、无逃生阀）
        nn::DatasetIdentity a;
        a.path = "/data/a.nndataset";
        a.sha256 = std::string(64, 'a');
        a.num_tokens = 100;
        a.num_docs = 2;
        a.num_windows = 4;
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
    // 走通整条恢复流程（SFT 入口）；三值 `.nnmodel` 的拒绝由 `check_resume_target`
    // 覆盖（上面），错误经 Result 回报而不在 stderr 上留痕。
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
}

// ── [4] 断点记账与退出码语义（P1-1 第一刀回归锁）────────────────────────
void test_ckpt_accounting(const std::filesystem::path& dir)
{
    nn::CpuEngine eng;
    const nn::ModelSpec spec = tiny_spec();
    auto built = nn::build_gpt_model_from_spec(eng, spec);
    NN_ASSERT_ALWAYS(built, "text_train_modules_test: 构建小 GPT 失败");
    nn::Model model = std::move(*built);

    // 失败路径：断点路径被目录占用 ⇒ 写入必然失败（只记账，不终止）
    const auto occupied = dir / "occupied.nnckpt";
    std::filesystem::create_directories(occupied);
    nn::cli::TextTrainCheckpointLog log;
    std::ostringstream err;
    nn::TrainState ckpt_state;
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
    nn::CpuEngine eng;
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
    id.num_windows = 4;
    id.loss_scope = "all";

    const auto path = dir / "wire.nnckpt";
    std::ostringstream err;
    nn::cli::TextTrainCheckpointLog log;
    nn::cli::save_ckpt_with_state(path.string(), model, spec, kTokenizerBytes, cfg, id, log,
                                  1, 2, 2, 1e-3, *opt, err);
    CHECK(log.failures() == 0 && err.str().empty(), "带状态的断点应一次写成功");

    auto st = nn::peek_train_state(path.string());
    CHECK(st.has_value(), "peek_train_state 应成功");
    if (st)
    {
        // step_done == steps_per_epoch ⇒ epoch 边界归一化（下一步 = epoch 2 / step 0）
        CHECK(st->counters.epoch == 2 && st->counters.step_in_epoch == 0 &&
                  st->counters.global_step == 4,
              "计数器记'下一步的位置'且 epoch 边界归一化");
        CHECK(st->optimizer_step == 1, "Adam 的 t 随步推进（1 步后 = 1）");
        CHECK(st->optimizer_blob_count == 2 * model.parameters().size(),
              "优化器状态张量数 = 2 × 参数数（m/v 交替）");
        CHECK(st->recipe.batch_size == 2 && st->recipe.epochs == 3 &&
                  st->recipe.total_steps == 6 && st->recipe.optimizer == "adam",
              "配方随断点落盘（batch / epochs / 曲线形状 / 优化器名）");
        CHECK(st->dataset.sha256 == id.sha256 && st->dataset.steps_per_epoch == 2 &&
                  st->dataset.num_windows == 4,
              "数据集身份 + 网格随断点落盘");

        // 数据集身份门禁走的就是断点里的这份身份
        nn::DatasetIdentity changed = id;
        changed.num_windows = 5;
        CHECK(!nn::cli::verify_dataset_identity(st->dataset, changed, path.string()).has_value(),
              "断点记录的身份 vs 变化的网格 ⇒ 硬报错（接线一致）");
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
    nn::CpuEngine eng;
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
