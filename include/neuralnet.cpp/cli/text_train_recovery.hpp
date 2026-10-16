// ── text_train_recovery.hpp — `--resume` 恢复路径（TDR 重启 / 分段续训）───────
//
// P1-1 拆出的 recovery 模块（docs/todo/18-roadmap.md §4.2）：
//   ① **角色门禁**：`--resume` 只认训练断点（`.nnckpt` / Checkpoint 角色）。
//      推理模型（`.nnmodel`）没有训练参数（三值层只有打包权重）→ 明确拒绝，
//      不静默降级成"从头训练"（docs/todo/23 §2.1 文件角色）。
//      v5/v6 老 `.bin` 无 role 字段 ⇒ 按训练文件处理（零破坏）。
//   ② **按文件规格重建模型**：规格是加载时的权威来源（缺键 = None → 旧文件
//      零破坏）；GPT / ALiBi-GPT 走同一条 `build_gpt_model_from_spec`，RAPT
//      独立路径。命令行给出的 `spec` 会在成功时被文件规格覆盖。
//   ③ **载入参数**：`load_model`（回报内嵌词表的字节数）。
//   非致命失败（读不到规格 / 载入参数失败）与旧行为一致：打印提示后**继续从头
//   训练**，不算错误；只有角色门禁失败是错误（调用方按 CLI 错误处理）。
//
// 本模块**不终止进程**（P1-1 口径：`std::exit` 只允许出现在 CLI 参数解析层）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/core_errors.hpp"           // nn::Result / NN_TRY
#include "neuralnet.cpp/compute_engine.hpp"        // nn::ComputeEngine
#include "neuralnet.cpp/compute_layer_base.hpp"    // nn::ModelRole / model_role_name
#include "neuralnet.cpp/domain_gpt.hpp"            // build_gpt_model_from_spec
#include "neuralnet.cpp/domain_rla.hpp"            // build_rapt_model_from_spec
#include "neuralnet.cpp/model_container.hpp"       // nn::Model
#include "neuralnet.cpp/model_serialization.hpp"   // peek_model_role / spec / load_model
#include "neuralnet.cpp/model_spec.hpp"            // nn::ModelSpec
#include "neuralnet.cpp/cli/text_train_config.hpp" // nn::cli::TrainConfig

namespace nn::cli
{
// ── 角色门禁（v7 起；docs/todo/23 §2.1）─────────────────────────────────
// 失败条件只有一条：**v7+ 且角色不是 Checkpoint**。文件打不开 / 头部不合法
// 一律放行——由后续 `peek_model_spec` 给出既有的"加载模型失败: …，将从头训练"
// 提示（旧行为对不存在/损坏的断点就是"从头训练"）。
[[nodiscard]] inline nn::Result<void> check_resume_role(const std::string& path)
{
    uint32_t resume_version = 0;
    auto role_r = nn::peek_model_role(path, &resume_version);
    if (role_r && resume_version >= 7 && *role_r != nn::ModelRole::Checkpoint)
    {
        NN_FAIL("错误: --resume 需要训练断点文件（.nnckpt，Checkpoint 角色），但 " + path +
                " 是 " + std::string(nn::model_role_name(*role_r)) +
                "；它不含训练参数（三值层只有打包权重）。\n"
                "  请用训练时 --save-interval 产出的 .nnckpt 续训。");
    }
    return {};
}

// ── 恢复结果（供调用方与测试断言）──────────────────────────────────────
struct TextTrainResume
{
    bool spec_restored = false;    // 从文件读到规格并按它重建了模型
    bool weights_loaded = false;   // 参数已载入
    std::size_t vocab_bytes = 0;   // 文件内嵌词表字节数（0 = 未嵌入）
};

// ── 恢复流程：角色门禁 → 读规格 → 按规格重建模型 → 载入参数 ──────────────
// out/err 语义与旧 main 一致（进度走 out，提示与警告走 err）。
[[nodiscard]] inline nn::Result<TextTrainResume> resume_text_train(
    nn::ComputeEngine& engine, const TrainConfig& cfg, nn::Model& model,
    nn::ModelSpec& spec, std::ostream& out, std::ostream& err)
{
    NN_TRY_CHECK(check_resume_role(cfg.resume_path));

    TextTrainResume result;
    auto spec_result = nn::peek_model_spec(cfg.resume_path);
    if (!spec_result)
    {
        err << "加载模型失败: " << spec_result.error().message << "，将从头训练。\n"
            << std::endl;
        return result;
    }

    auto file_spec = std::move(*spec_result);
    // RAPT 独立构建路径
    if (file_spec.is_rapt())
    {
        out << "从模型文件读取 RAPT 规格\n";
        NN_TRY_MSG(build_result, "从断点重建 RAPT 模型失败",
                   nn::build_rapt_model_from_spec(engine, file_spec, cfg.precision));
        model = std::move(*build_result);
        spec = file_spec;
        result.spec_restored = true;
    }
    // 统一的 GPTModel 通过 pos_encoding 区分 Learned/Sinusoidal/ALiBi，
    // GPT 与 ALiBi_GPT 走同一条构建路径。
    else if (file_spec.is_gpt() || file_spec.is_alibi_gpt())
    {
        if (file_spec.pos_encoding == nn::PosEncodingType::ALiBi)
            out << "从模型文件读取 ALiBi GPT 规格\n";
        else if (file_spec.pos_encoding == nn::PosEncodingType::RoPE)
            out << "从模型文件读取 RoPE GPT 规格\n";
        else
            out << "从模型文件读取 GPT 规格\n";
        NN_TRY_MSG(build_result, "从断点重建 GPT 模型失败",
                   nn::build_gpt_model_from_spec(engine, file_spec, cfg.precision));
        model = std::move(*build_result);
        spec = file_spec;
        result.spec_restored = true;
    }
    else
        out << "旧格式模型文件，使用命令行参数\n";

    auto load_result = nn::load_model(cfg.resume_path, model);
    if (!load_result)
    {
        err << "加载模型失败: " << load_result.error().message << "，将从头训练。\n"
            << std::endl;
        return result;
    }
    result.weights_loaded = true;
    result.vocab_bytes = load_result->size();
    out << "已加载模型: " << cfg.resume_path;
    if (!load_result->empty())
        out << " (含嵌入词表 " << load_result->size() << " 字节)";
    out << "\n" << std::endl;
    return result;
}
} // namespace nn::cli
