// ── text_train_recovery.hpp — `--resume` 恢复路径（TDR 重启 / 分段续训）───────
//
// P1-1 拆出的 recovery 模块（docs/todo/18-roadmap.md §4.2）：
//   ① **断点状态采纳**（P1-2 第三刀 B）：`--resume` 指向带训练状态段的断点时，
//      形状/配方/起点一律由状态段决定，显式给出 A/B 类旗标即报错；数据集身份
//      不一致硬报错（无逃生阀）；优化器状态（含 `t`）随之恢复。
//      角色语义（v8，docs/todo/23 §2.1.1）：`.nnckpt` = 断点续训、
//      `.nnmodel` = warm start（SFT，CLI 全权）。**P1-7 起没有需要拒绝的组合**：
//      三值 `.nnmodel` 的打包载荷在载入时按文件 τ 反量化成 f32 latent
//      （`WeightLoadIntent::Trainable`）⇒ 同样可续训。v5/v6/v7 老文件放行（零破坏）。
//   ② **按文件规格重建模型**：规格是加载时的权威来源（缺键 = None → 旧文件
//      零破坏）；GPT / ALiBi-GPT 走同一条 `build_gpt_model_from_spec`，RAPT
//      独立路径。命令行给出的 `spec` 会在成功时被文件规格覆盖。
//   ③ **载入参数**：`load_model`（回报内嵌词表的字节数）。
//   非致命失败（读不到规格 / 载入参数失败）与旧行为一致：打印提示后**继续从头
//   训练**，不算错误；禁列旗标 / 数据集身份不符才是错误
//   （调用方按 CLI 错误处理）。
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
#include "neuralnet.cpp/train_state.hpp"           // nn::TrainState / DatasetIdentity（P1-2）
#include "neuralnet.cpp/compute_optimizer.hpp"     // nn::Optimizer（状态载入）
#include "neuralnet.cpp/cli/text_train_config.hpp" // nn::cli::TrainConfig

namespace nn::cli
{
// ── 角色语义（v8；2026-10-16 裁定，docs/todo/23 §2.1.1）──────────────────
//   · Checkpoint（`.nnckpt`）⇒ **断点续训**：状态段是唯一权威（A/B 类参数禁列）。
//   · Model（`.nnmodel`）⇒ **warm start**（CLI 全权）：SFT / 换数据集续训的入口，
//     优化器冷启动、step/lr 从 0 起算。三值模型同样走这里 —— 打包载荷在载入时
//     按文件 τ 反量化成 f32 latent（P1-7），故不存在"只能从头训练"的组合。
//   · v7 及以下 / 头部不合法 ⇒ 放行（旧断点零破坏；不存在或损坏的断点由后续
//     `peek_model_spec` 给出既有的"加载模型失败: …，将从头训练"提示）。

// ── 断点状态采纳（P1-2 第三刀 B）────────────────────────────────────────
// 读断点的训练状态段，并按"状态段权威"改写 cfg：断点自带精度 profile / 曲线形状 /
// 网格 / 起点，CLI 再给一遍会把轨迹掰弯 ⇒ **显式给出 A/B 类旗标即报错**（按旗标名
// 判定、不看值；解析层已在 `TrainConfig::explicit_train_flags` 登记）。
// 覆盖的 cfg 字段 = A/B 类全量（形状 + 配方 + 起点）；C 类（保存路径 / 间隔 / 设备 /
// 探针等）留给 CLI。
struct ResumeStateAdoption
{
    bool adopted = false;      // 断点带训练状态段 ⇒ cfg 已被状态改写
    nn::TextTrainState state;  // 已采纳的状态（adopted 时有效；文本域）
    bool shape_adopted = false;// 形状/架构已按文件 spec 改写（两种角色都成立）
};

[[nodiscard]] inline nn::Result<ResumeStateAdoption> adopt_resume_state(
    TrainConfig& cfg, std::ostream& out, std::ostream& err)
{
    ResumeStateAdoption result;
    if (!cfg.load_existing) return result;

    // ── 形状/架构：**以文件 spec 为准**（对两种角色都成立——文件就是那个模型，
    //    不可能把一个 16 维的断点 warm start 成 128 维）。顺带让下方的配置横幅
    //    与实际训练的模型一致（否则 src 会用 CLI 默认形状先建一个丢掉）。
    auto spec_r = nn::peek_model_spec(cfg.resume_path);
    if (spec_r)
    {
        const nn::ModelSpec& fs = *spec_r;
        if (fs.d_model > 0) cfg.d_model = fs.d_model;
        if (fs.num_heads > 0) cfg.num_heads = fs.num_heads;
        if (fs.num_layers > 0) cfg.num_layers = fs.num_layers;
        if (fs.d_ff > 0) cfg.d_ff = fs.d_ff;
        if (fs.seq_len > 0) cfg.seq_len = fs.seq_len;
        cfg.pos_encoding = fs.pos_encoding;
        cfg.activation = fs.activation;
        cfg.norm_type = fs.norm_type;
        cfg.model_type = fs.is_rapt() ? "rapt" : "gpt";
        result.shape_adopted = true;
    }

    uint32_t version = 0;
    auto role_r = nn::peek_model_role(cfg.resume_path, &version);
    const bool is_checkpoint =
        role_r && version >= 7 && *role_r == nn::ModelRole::Checkpoint;
    if (role_r && version >= 7 && *role_r == nn::ModelRole::Model)
    {
        out << "从推理模型 warm start（.nnmodel）：配方与步计数由命令行给出，优化器冷启动"
               "（三值模型的权重按文件 τ 反量化成 f32 latent，P1-7）\n";
        return result;
    }
    if (!is_checkpoint) return result;   // 老文件 / 打不开：交给 resume_text_train 的老语义

    auto state_r = nn::peek_train_state(cfg.resume_path);
    if (!state_r)
    {
        // 无状态段的断点（v7 老断点 / 早期 v8 文件）= 冷启动语义（18 §4.2 验收）
        err << "[提示] 断点 " << cfg.resume_path
            << " 没有训练状态段 ⇒ 冷启动续训（优化器动量与步计数从 0 起；"
               "配方仍取命令行）\n";
        return result;
    }
    auto s_up = std::move(*state_r);
    auto* s_ptr = dynamic_cast<nn::TextTrainState*>(s_up.get());
    if (s_ptr == nullptr)
        NN_FAIL("断点 " + cfg.resume_path + " 的训练状态不是文本域（kind=" +
                std::string(s_up->state_kind()) + "），--resume 文本训练不接受");
    nn::TextTrainState s = std::move(*s_ptr);

    // ── 禁列校验（显式给出 A/B 类旗标即报错）────────────────────────────
    if (!cfg.explicit_train_flags.empty())
    {
        std::string given;
        for (const std::string& f : cfg.explicit_train_flags)
        {
            if (!given.empty()) given += " ";
            given += f;
        }
        NN_FAIL("错误: --resume " + cfg.resume_path +
                " 是训练断点（.nnckpt，带训练状态段）⇒ 形状/配方/计数器一律由断点决定，"
                "不得再由命令行给出。\n  检测到显式输入: " + given +
                "\n  · 禁止（A/B 类：形状 / 配方 / 计数器）: " + forbidden_resume_flags_text() +
                "\n  · 仍允许（不改变训练）: --save / --save-interval / --log-interval / "
                "--max-steps / --gpu / --grad-log / --checkpoint-every / --activation-offload / "
                "--flush-interval\n"
                "  · 要改配方 / 换数据集 ⇒ 从推理模型（.nnmodel）开新一轮（CLI 全权、优化器冷启动）。");
    }

    // ── 按状态段改写 cfg（A/B 类全量）────────────────────────────────────
    cfg.precision = s.recipe.precision;
    cfg.optimizer_name = s.recipe.optimizer;
    cfg.lr_schedule = s.recipe.lr_schedule;
    cfg.lr = static_cast<nn::Scalar>(s.recipe.lr);
    cfg.min_lr = static_cast<nn::Scalar>(s.recipe.min_lr);
    cfg.weight_decay = static_cast<nn::Scalar>(s.recipe.weight_decay);
    cfg.beta1 = static_cast<nn::Scalar>(s.recipe.beta1);
    cfg.beta2 = static_cast<nn::Scalar>(s.recipe.beta2);
    cfg.max_norm = static_cast<nn::Scalar>(s.recipe.max_norm);
    cfg.warmup_epochs = static_cast<int>(s.recipe.warmup_epochs);
    cfg.warmup_steps = static_cast<std::size_t>(s.recipe.warmup_steps);
    cfg.accum_steps = static_cast<std::size_t>(s.recipe.accum_steps);
    if (s.recipe.batch_size_is_global)
    {
        cfg.batch_size = static_cast<std::size_t>(s.recipe.batch_size);
    }
    else
    {
        // docs/todo/30：旧状态的 batch_size 是 micro-batch；还原成等效 global。
        // 旧断点若落在累积中间，pending gradients 不在文件里，只能从最近的
        // 已完成 update 边界继续。
        if (s.recipe.accum_steps == 0)
            NN_FAIL("旧断点的 accum_steps=0，无法转换 batch_size 语义");
        cfg.batch_size = static_cast<std::size_t>(
            s.recipe.batch_size * s.recipe.accum_steps);
        if (s.recipe.accum_steps > 0)
            s.counters.step_in_epoch /= s.recipe.accum_steps;
        s.recipe.batch_size = static_cast<std::uint64_t>(cfg.batch_size);
        s.recipe.micro_batch_size = static_cast<std::uint64_t>(s.recipe.batch_size /
                                                               s.recipe.accum_steps);
        s.recipe.batch_size_is_global = true;
        if (s.dataset.num_samples > 0)
            s.dataset.steps_per_epoch =
                (s.dataset.num_samples + s.recipe.batch_size - 1) /
                s.recipe.batch_size;
    }
    cfg.stride = static_cast<std::size_t>(s.stride);
    cfg.lr_per_epoch.clear();
    cfg.lr_per_epoch.reserve(s.recipe.lr_per_epoch.size());
    for (double v : s.recipe.lr_per_epoch)
        cfg.lr_per_epoch.push_back(static_cast<nn::Scalar>(v));
    if (s.recipe.epochs > 0)                       // 0 = 未记录 ⇒ 保留 CLI（老/手写断点兜底）
        cfg.epochs = static_cast<int>(s.recipe.epochs);
    // 断点续训的起点（计数器 = 下一步的位置；`--resume-epoch/--resume-step` 已禁）
    cfg.start_epoch = static_cast<int>(s.counters.epoch);
    cfg.start_step = static_cast<std::size_t>(s.counters.step_in_epoch);

    result.adopted = true;
    result.state = s;
    out << "从断点状态段恢复: epoch=" << s.counters.epoch
        << " step_in_epoch=" << s.counters.step_in_epoch
        << " global_step=" << s.counters.global_step
        << "  精度=" << nn::precision_name(s.recipe.precision.param) << "/"
        << nn::precision_name(s.recipe.precision.compute) << "/"
        << nn::precision_name(s.recipe.precision.stable) << "/"
        << nn::precision_name(s.recipe.precision.optimizer)
        << "  优化器=" << s.recipe.optimizer
        << "  global_batch=" << s.recipe.batch_size
        << " micro_batch=" << s.recipe.micro_batch_size
        << " accum=" << s.recipe.accum_steps
        << "  epochs=" << s.recipe.epochs << "\n";
    return result;
}

// ── 数据集身份门禁（硬错、无逃生阀；docs/todo/23 §2.1.1）────────────────
// 数据集一变 ⇒ 窗口数变 ⇒ steps_per_epoch 变 ⇒ step 级调度的 total_steps、
// `global_step` 的语义、epoch 边界**同时**改变 ⇒ "同一断点 + 不同数据集"
// 不存在正确的续训语义。合法出口 = 从 `.nnmodel` 开新一轮。
[[nodiscard]] inline nn::Result<void> verify_dataset_identity(
    const nn::DatasetIdentity& saved, const nn::DatasetIdentity& current,
    const std::string& resume_path)
{
    const std::string why = nn::dataset_mismatch_reason(saved, current);
    if (why.empty()) return {};
    NN_FAIL("错误: --resume 的数据集与断点不一致（内容身份变了 ⇒ 窗口网格 / step 号语义 / "
            "lr 曲线形状全变，续训不存在正确语义）:\n" + why +
            "\n  断点记录的路径: " + saved.path + "\n  本次使用的路径: " + current.path +
            "\n  断点文件: " + resume_path +
            "\n  · 换数据集 = 新任务 ⇒ 请从推理模型（.nnmodel）重开一轮（CLI 全权、优化器冷启动）；"
            "\n  · 仍要用这个断点 ⇒ 请提供与断点**内容完全一致**的 .nndataset。");
    return {};
}

// ── 载入优化器状态（续训路径；建优化器之后调用）──────────────────────────
// 优化器身份（种类/超参）已由状态段写回 cfg ⇒ 张量数量与形状必须对上，
// 对不上由 `Optimizer::load_state` 明确报错（不静默冷启动）。
[[nodiscard]] inline nn::Result<void> load_optimizer_state(
    nn::ComputeEngine& engine, const std::string& path, nn::Optimizer& optimizer,
    std::ostream& out)
{
    NN_TRY(loaded, nn::load_train_state(engine, path));
    optimizer.set_state_step(loaded->state->optimizer_step);
    NN_TRY_CHECK(optimizer.load_state(loaded->optimizer_blobs));
    out << "已恢复优化器状态: " << loaded->optimizer_blobs.size() << " 个张量"
        << "，步计数 " << loaded->state->optimizer_step
        << "（" << loaded->state->recipe.optimizer << "）\n";
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
    TextTrainResume result;
    auto spec_result = nn::peek_model_spec(cfg.resume_path);
    if (!spec_result)
    {
        err << "加载模型失败: " << spec_result.error().message << "，将从头训练。\n"
            << std::endl;
        return result;
    }

    auto file_spec = std::move(*spec_result);
    // 角色（v8）：Checkpoint = 断点续训；Model = warm start（SFT）。两者都按
    // **可训练**意图载入：三值 Model 角色的打包载荷按文件 τ 反量化成 f32 latent
    // （P1-7），dense / `.nnckpt` 路径与逐位不变（意图只影响打包载荷那一条分支）。
    uint32_t file_version = 0;
    const auto role_r = nn::peek_model_role(cfg.resume_path, &file_version);
    const bool model_role =
        role_r && file_version >= 7 && *role_r == nn::ModelRole::Model;
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

    auto load_result = nn::load_model(cfg.resume_path, model, nullptr, nullptr,
                                      nn::WeightLoadIntent::Trainable);
    if (!load_result)
    {
        err << "加载模型失败: " << load_result.error().message << "，将从头训练。\n"
            << std::endl;
        return result;
    }
    result.weights_loaded = true;
    result.vocab_bytes = load_result->size();
    if (model_role && file_spec.weight_quant == nn::WeightQuant::T1_58)
        out << "三值（T1_58）推理模型：按文件 τ 反量化打包权重为 f32 latent ⇒ 可续训"
               "（首次 forward 与推理文件逐位一致）\n";
    out << "已加载模型: " << cfg.resume_path;
    if (!load_result->empty())
        out << " (含嵌入词表 " << load_result->size() << " 字节)";
    out << "\n" << std::endl;
    return result;
}
} // namespace nn::cli
