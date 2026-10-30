// ── text_train_checkpoint.hpp — text_train 的两类模型产出（.nnckpt / .nnmodel）─
//
// P1-1 拆出的 checkpoint 模块（docs/internal/18-roadmap.md §4.2）。职责：
//   * **断点路径派生**：`--save` 的 .nnmodel 路径 → 同名 .nnckpt；
//   * **断点保存的记账语义**：一次写失败**不终止训练**（磁盘抖动不该毁掉整轮），
//     但收尾必须让进程非零退出——否则"断点没落盘"会被脚本/CI/训练包当成"这轮
//     训练完好"，`--resume` 会静默拿到旧断点（P1-1 第一刀的缺口，docs/history
//     「P1-1（第一刀）」）；
//   * **推理模型产出**（Model 角色，.nnmodel）：失败即失败（没有产出）。
//
// 本模块**不终止进程**：结论一律经 Result / 返回值交给调用方（P1-1 口径：
// `std::exit` 只允许出现在 CLI 参数解析层，见 cli/text_train_config.hpp）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "neuralnet.cpp/core_errors.hpp"          // nn::Result
#include "neuralnet.cpp/model_container.hpp"      // nn::Model
#include "neuralnet.cpp/model_serialization.hpp"  // nn::save_model / ModelRole
#include "neuralnet.cpp/model_spec.hpp"           // nn::ModelSpec
#include "neuralnet.cpp/train_state.hpp"          // nn::TrainState（P1-2）

namespace nn::cli
{
// ── 训练断点路径派生（docs/internal/23 §2.1 文件角色）────────────────────────
// `--save` 的路径换扩展名为 .nnckpt（`gpt_model.nnmodel` → `gpt_model.nnckpt`）；
// 非 .nnmodel 路径一律追加 .nnckpt。
[[nodiscard]] inline std::string text_train_ckpt_path(const std::string& save_path)
{
    static const std::string kExt = ".nnmodel";
    if (save_path.size() > kExt.size() &&
        save_path.compare(save_path.size() - kExt.size(), kExt.size(), kExt) == 0)
        return save_path.substr(0, save_path.size() - kExt.size()) + ".nnckpt";
    return save_path + ".nnckpt";
}

// ── 断点保存记账（P1-1 退出码语义 + P1-2 训练状态）──────────────────────
// 用法：循环里 `--save-interval` 到点就 `log.save(...)`；收尾用
// `text_train_ckpt_exit_code(log.failures(), std::cerr)` 取退出码。
class TextTrainCheckpointLog
{
public:
    // 保存一次训练断点（Checkpoint 角色：含训练参数，三值层写 f32 latent）。
    // P1-2 起同时写**训练状态段**（`TrainState` + 优化器状态张量）：断点是
    // "训练的可续点"，续训时以状态段为唯一权威（docs/internal/23 §2.1.1）。
    // 失败不终止——打印一行（带"第 N 次"）并计数；成功静默（与旧行为逐字一致）。
    void save(const std::string& path, nn::Model& model, const nn::ModelSpec& spec,
              const std::string& tokenizer_bytes, const nn::TrainState& state,
              std::span<const nn::Tensor> optimizer_blobs, std::ostream& err)
    {
        const nn::CheckpointState ckpt{&state, optimizer_blobs};
        auto r = nn::save_model(path, model, spec, tokenizer_bytes,
                                nn::ModelRole::Checkpoint, ckpt);
        if (r) return;
        ++failures_;
        err << "\n  [ckpt] 保存失败（第 " << failures_ << " 次）: " << r.error().message
            << "\n";
    }

    // 状态导出失败等"这一份断点没写出去"的事件：与写失败同一记账口径
    // （P1-1：不终止训练，但收尾必须非零退出）。
    void note_failure(const std::string& what, std::ostream& err)
    {
        ++failures_;
        err << "\n  [ckpt] 保存失败（第 " << failures_ << " 次）: " << what << "\n";
    }

    [[nodiscard]] std::size_t failures() const noexcept { return failures_; }

private:
    std::size_t failures_ = 0;
};

// 收尾退出码：有断点写失败 ⇒ 打印明确错误行并返回 1（推理模型文件已写出，但
// "可续训"这一性质不成立，上层不得把它当成功）；否则静默返回 0。
[[nodiscard]] inline int text_train_ckpt_exit_code(std::size_t failures, std::ostream& err)
{
    if (failures == 0) return 0;
    err << "[错误] 本次训练有 " << failures
        << " 次 checkpoint(.nnckpt) 保存失败（推理模型已写出）⇒ 退出码 1\n";
    return 1;
}

// ── 推理模型产出（Model 角色）──────────────────────────────────────────
// 只含推理所需参数（三值层写打包形态 + τ）；失败即失败：写不出去 = 这一轮训练
// 对用户没有产出，调用方据此非零退出。
[[nodiscard]] inline nn::Result<void> save_text_train_model(
    const std::string& path, nn::Model& model, const nn::ModelSpec& spec,
    const std::string& tokenizer_bytes)
{
    return nn::save_model(path, model, spec, tokenizer_bytes, nn::ModelRole::Model);
}
} // namespace nn::cli
