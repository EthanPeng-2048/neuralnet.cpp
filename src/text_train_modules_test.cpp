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

    // 门禁：推理模型被拒绝（文案 = 用户可见契约，docs/todo/23 §2.1）
    auto gate_bad = nn::cli::check_resume_role(model_file);
    CHECK(!gate_bad, ".nnmodel 必须被 --resume 的角色门禁拒绝");
    CHECK(!gate_bad &&
              gate_bad.error().message.find("--resume 需要训练断点文件") !=
                  std::string::npos &&
              gate_bad.error().message.find("Checkpoint") != std::string::npos,
          "拒绝文案应指明需要 Checkpoint 角色断点");
    CHECK(nn::cli::check_resume_role(ckpt_file), "Checkpoint 角色必须放行");
    CHECK(nn::cli::check_resume_role((dir / "missing.nnckpt").string()),
          "不存在的断点应放行（由载入提示走'从头训练'语义）");

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

    // 反例：用推理模型走完整恢复流程 → 报错（不是"从头训练"）
    nn::cli::TrainConfig bad_cfg;
    bad_cfg.resume_path = model_file;
    bad_cfg.load_existing = true;
    nn::ModelSpec spec_out2;
    nn::Model m2;
    std::ostringstream out2;
    std::ostringstream err2;
    auto bad = nn::cli::resume_text_train(eng, bad_cfg, m2, spec_out2, out2, err2);
    CHECK(!bad, "用 .nnmodel 恢复必须失败（不得静默从头训练）");
    CHECK(!bad && bad.error().message.find("--resume 需要训练断点文件") != std::string::npos,
          "完整恢复流程的失败文案应与门禁一致（经 Result 回报，不在 stderr 上）");
    CHECK(err2.str().empty() && out2.str().empty(),
          "角色门禁失败时不得先打印进度/提示（错误由调用方统一透出）");
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
