// ── text_train_loop.hpp — text_train 的训练驱动适配层（P2b 收窄）────────────
//
// P2b 起训练循环本身已上收到库层 `nn::TextTrainer`（text_trainer.hpp）：
//   · 数据装配 = `nn::TextSampleSet`（dataset_samples.hpp）；
//   · 生命周期 = `nn::TrainingSession`（Model + Optimizer + update state）；
//   · epoch×step / lr 调度 / global→micro 切分 / 非阻塞 loss 回读 / 测试评估 /
//     运行态 TrainState 与断点状态装配 = TextTrainer::fit()；
//   · micro-step 梯度累积 / 裁剪 / optimizer update = TrainingSession。
//
// 本模块只保留 **CLI 侧策略**（都属于"入口/诊断"，不属于训练语义）：
//   * 进度输出/EPOCH 汇总由 Trainer 默认打印（文案锚：与 P1-1 逐字一致）；
//   * `--grad-log` 的梯度统计（需要宿主侧下载张量）；
//   * `NN_NAN_TRACE` / `NN_MEM_STATS` 探针（仅 `NN_ENABLE_PROBES=ON` 构建存在）；
//   * 定期断点写盘（.nnckpt 路径派生 + 记账 + 状态段序列化）。
//
// 本模块**不终止进程**：错误经 `Result` 返回（`std::exit` 只在参数解析层）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "neuralnet.cpp/algebra_matrix.hpp"        // nn::Matrix（梯度统计）
#include "neuralnet.cpp/compute_engine.hpp"        // nn::ComputeEngine
#include "neuralnet.cpp/compute_optimizer.hpp"     // nn::Optimizer
#include "neuralnet.cpp/core_config.hpp"           // nn::Scalar / NN_PROBES_ENABLED
#include "neuralnet.cpp/core_errors.hpp"           // nn::Result / NN_TRY
#include "neuralnet.cpp/model_container.hpp"       // nn::Model
#include "neuralnet.cpp/model_spec.hpp"            // nn::ModelSpec
#include "neuralnet.cpp/text_trainer.hpp"          // nn::TextTrainer（训练驱动）
#include "neuralnet.cpp/training_session.hpp"      // nn::TrainingSession
#include "neuralnet.cpp/cli/text_train_checkpoint.hpp"
#include "neuralnet.cpp/cli/text_train_config.hpp"

namespace nn::cli
{
// ── 断点落盘（CLI 序列化策略；何时保存由 TextTrainer 决定）─────────────────
// 状态段（计数器/配方/数据集身份）由 Trainer 装配好后经 TextCheckpointRequest
// 传入；本函数只做"导出优化器张量 + 写 .nnckpt + 失败记账"。
inline void save_ckpt_with_state(const std::string& path, nn::Model& model,
                                 const nn::ModelSpec& spec, const std::string& tokenizer_bytes,
                                 const nn::TrainState& state, nn::Optimizer& optimizer,
                                 TextTrainCheckpointLog& log, std::ostream& err)
{
    std::vector<nn::Tensor> blobs;
    auto sr = optimizer.save_state(blobs);
    if (!sr)
    {
        log.note_failure("优化器状态导出失败: " + sr.error().message, err);
        return;
    }
    log.save(path, model, spec, tokenizer_bytes, state, blobs, err);
}

// ══ 非有限值追踪探针（NN_NAN_TRACE=1；探针门见 core_config.hpp）═══════════
// 探针门：默认构建（未 -DNN_ENABLE_PROBES=ON）里**整块不编译**——训练循环里
// 连判定分支都不存在（hook 同样被 #if 包住），环境变量 NN_NAN_TRACE 无效。
#if NN_PROBES_ENABLED
// 逐步扫描"参数 + 参数梯度"里的 NaN/Inf，命中第一个即打印
//（张量序号/形状/非有限元素数/最大有限值）并让训练立即停止。
// 用途：定位"训着训着 loss 变 nan"这类只在特定设备/数据顺序下复现的缺陷——
// 打印的序号即该步参数/梯度数组的下标，可与参数张量顺序对照定位到层。
inline bool nn_nan_trace_on()
{
    // MSVC CRT 弃用 getenv（-Werror 下必须走 _dupenv_s），与 NN_MEM_STATS 同款。
    static const bool on = []
    {
#if defined(_MSC_VER)
        char* buf = nullptr;
        std::size_t len = 0;
        _dupenv_s(&buf, &len, "NN_NAN_TRACE");
        const bool v = (buf != nullptr && buf[0] != '\0');
        std::free(buf);
        return v;
#else
        const char* v = std::getenv("NN_NAN_TRACE");
        return v != nullptr && v[0] != '\0';
#endif
    }();
    return on;
}

// 最近一步喂进模型的 doc_ids（仅 NN_NAN_TRACE 时维护；命中时打印首样本的
// 文档 id 序列与"每行可见列数"统计——文档感知掩码类缺陷靠它定位）。
inline std::vector<std::size_t>& nn_nan_trace_doc_ids()
{
    static std::vector<std::size_t> ids;
    return ids;
}

// 最近一步喂进模型的 token id（同上，仅 NN_NAN_TRACE 时维护；用于把
// "embedding 梯度的非有限行"映射回 batch 内的 (t, b) 位置）。
inline std::vector<std::size_t>& nn_nan_trace_tokens()
{
    static std::vector<std::size_t> toks;
    return toks;
}

inline bool nn_nan_trace_step(nn::ComputeEngine& engine, nn::Model& model, std::size_t step,
                              std::size_t seq_len)
{
    for (int pass = 0; pass < 2; ++pass)
    {
        const char* what = (pass == 0) ? "param" : "grad";
        std::vector<nn::TensorRef> refs =
            (pass == 0) ? model.parameters() : model.param_gradients();
        std::size_t idx = 0;
        for (auto& ref : refs)
        {
            auto m_r = engine.to_matrix(ref.get());
            if (!m_r)
            {
                ++idx;
                continue;
            }
            const nn::Matrix& m = *m_r;
            std::size_t bad = 0;
            double amax = 0.0;
            std::vector<std::size_t> bad_rows;   // 行级定位（只看前若干行）
            for (std::size_t r = 0; r < m.rows(); ++r)
            {
                bool row_bad = false;
                for (std::size_t c = 0; c < m.cols(); ++c)
                {
                    const float v = m.at(r, c);
                    if (std::isnan(v) || std::isinf(v))
                    {
                        ++bad;
                        row_bad = true;
                    }
                    else amax = std::max(amax, std::fabs(static_cast<double>(v)));
                }
                if (row_bad && bad_rows.size() < 32) bad_rows.push_back(r);
            }
            if (bad != 0)
            {
                std::printf("[NAN-TRACE] step %zu: %s[%zu] %zux%zu 非有限 %zu/%zu，"
                            "max|有限|=%.4g\n",
                            step, what, idx, m.rows(), m.cols(), bad, m.rows() * m.cols(), amax);
                // 同一 batch 的 doc_ids 快照（首样本）：文档感知掩码是
                // "逐行可见列数"的源头，命中时把序列与统计一起打出来。
                const auto& ids = nn_nan_trace_doc_ids();
                if (!ids.empty() && seq_len > 0 && ids.size() >= seq_len)
                {
                    std::printf("[NAN-TRACE] doc_ids[0..%zu): ", seq_len);
                    for (std::size_t t = 0; t < seq_len; ++t)
                        std::printf("%zu%s", ids[t], t + 1 < seq_len ? "," : "");
                    std::printf("\n");
                    std::size_t zero_visible = 0, min_visible = ids.size();
                    for (std::size_t t = 0; t < seq_len; ++t)
                    {
                        std::size_t vis = 0;
                        for (std::size_t j = 0; j <= t; ++j)
                            if (ids[j] == ids[t]) ++vis;
                        min_visible = std::min(min_visible, vis);
                        if (vis == 0) ++zero_visible;
                    }
                    std::printf(
                        "[NAN-TRACE] doc_ids 统计: 零可见列的行=%zu，最小可见列=%zu，"
                        "递减次数=%zu，唯一 id 数=%zu\n",
                        zero_visible, min_visible,
                        [&]
                        {
                            std::size_t d = 0;
                            for (std::size_t t = 1; t < seq_len; ++t)
                                if (ids[t] < ids[t - 1]) ++d;
                            return d;
                        }(),
                        [&]
                        {
                            auto v = ids;
                            std::sort(v.begin(), v.begin() + seq_len);
                            return static_cast<std::size_t>(
                                std::unique(v.begin(), v.begin() + seq_len) - v.begin());
                        }());
                }
                // 行级定位：`grad[0]` 是 token embedding 的梯度，行号 = token id；
                // 把它映射回 batch 内的 (t,b) 位置，可区分"某些 id 坏"（数据/查表）
                // 与"某些位置坏"（掩码/位置相关算子）。
                if (pass == 1 && idx == 0 && !bad_rows.empty())
                {
                    std::printf("[NAN-TRACE] 非有限行(token id, 前 %zu): ", bad_rows.size());
                    for (std::size_t k = 0; k < bad_rows.size(); ++k)
                        std::printf("%zu%s", bad_rows[k], k + 1 < bad_rows.size() ? "," : "");
                    std::printf("\n");
                    const auto& toks = nn_nan_trace_tokens();
                    const std::size_t bs =
                        (seq_len > 0 && !toks.empty()) ? toks.size() / seq_len : 0;
                    if (bs > 0)
                    {
                        std::printf("[NAN-TRACE] 命中位置 (t,b) 前 24: ");
                        std::size_t shown = 0;
                        for (std::size_t b = 0; b < bs && shown < 24; ++b)
                            for (std::size_t t = 0; t < seq_len && shown < 24; ++t)
                            {
                                const std::size_t id = toks[b * seq_len + t];
                                if (std::find(bad_rows.begin(), bad_rows.end(), id) !=
                                    bad_rows.end())
                                {
                                    std::printf("(%zu,%zu)=%zu ", t, b, id);
                                    ++shown;
                                }
                            }
                        std::printf("\n");
                    }
                }
                std::fflush(stdout);
                return false;
            }
            ++idx;
        }
    }
    return true;
}
#endif   // NN_PROBES_ENABLED（非有限值追踪探针）

// 计算并打印全局梯度统计：L2 范数、绝对值最大值、均值。
// GPU 模式下自动通过 engine.to_matrix() 下载张量到 CPU。
inline void log_text_train_gradient_stats(nn::ComputeEngine& engine,
                                          const std::vector<nn::TensorRef>& grads)
{
    Scalar global_sum_sq = 0.0;
    Scalar global_abs_max = 0.0;
    Scalar global_abs_sum = 0.0;
    std::size_t global_count = 0;
    std::size_t tensor_idx = 0;

    for (auto& grad_ref : grads)
    {
        const auto& grad = grad_ref.get();
        // M1（docs/internal/17 §4.1）：张量读取一律经引擎下载到宿主
        // （GPU 直接下载；CPU 同样走 to_matrix，f16 存储也安全）
        nn::Matrix mat;
        {
            auto m = engine.to_matrix(grad);
            if (!m) continue;
            mat = std::move(*m);
        }

        const Scalar sum_sq = mat.reduce(Scalar{0}, std::plus<>{}, [](Scalar x) { return x * x; });
        const Scalar abs_max = mat.reduce(
            Scalar{0}, [](Scalar a, Scalar b) { return std::max(a, b); },
            [](Scalar x) { return std::abs(x); });
        const Scalar abs_sum =
            mat.reduce(Scalar{0}, std::plus<>{}, [](Scalar x) { return std::abs(x); });
        const std::size_t n = mat.size();

        global_sum_sq += sum_sq;
        global_abs_max = std::max(global_abs_max, abs_max);
        global_abs_sum += abs_sum;
        global_count += n;

        // 逐张量输出（仅在张量数量 <= 30 时显示详情，避免刷屏）
        if (grads.size() <= 30)
        {
            std::cout << "    [" << tensor_idx << "] "
                      << "(" << mat.rows() << "x" << mat.cols() << ") "
                      << "norm=" << std::scientific << std::setprecision(4) << std::sqrt(sum_sq)
                      << "  max|g|=" << abs_max << "  mean|g|=" << (n > 0 ? abs_sum / n : Scalar{0})
                      << std::endl;
        }
        ++tensor_idx;
    }

    const Scalar global_norm = std::sqrt(global_sum_sq);
    const Scalar global_mean = global_count > 0 ? global_abs_sum / global_count : Scalar{0};
    std::cout << "    >> grad_norm=" << std::scientific << std::setprecision(4) << global_norm
              << "  max|g|=" << global_abs_max << "  mean|g|=" << global_mean
              << "  params=" << global_count << "  tensors=" << grads.size() << std::endl;
}

// ── 显存阶段采样（NN_MEM_STATS=1；探针门关时恒 false 且不读环境变量）─────────
// 与 src/mem_probe.cpp 同口径：把峰值归因到具体生命周期阶段。默认构建里
// `on == false` ⇒ 训练热路径与探针无关（判定是常量）。
struct TextTrainMemProbe
{
    bool on = false;

    void mark(nn::ComputeEngine& engine, const char* tag) const
    {
        if (!on) return;
        std::cout << "[mem] " << std::left << std::setw(16) << tag << std::right << " "
                  << engine.pool_stats() << std::endl;
    }
};

[[nodiscard]] inline TextTrainMemProbe make_text_train_mem_probe()
{
#if !NN_PROBES_ENABLED
    return {};
#else
    // MSVC CRT 弃用 getenv（-Werror）：按平台用 _dupenv_s / getenv。
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    _dupenv_s(&buf, &len, "NN_MEM_STATS");
    const bool on = (buf != nullptr && buf[0] != '\0' && buf[0] != '0');
    std::free(buf);
    return TextTrainMemProbe{on};
#else
    const char* v = std::getenv("NN_MEM_STATS");
    return TextTrainMemProbe{v != nullptr && v[0] != '\0' && v[0] != '0'};
#endif
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  训练入口适配：给 TextTrainer 装 CLI 侧 hooks（诊断/断点），然后跑 fit()
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline nn::Result<nn::TextTrainReport>
run_text_train_loop(nn::TextTrainer& trainer, const TrainConfig& cfg, const nn::ModelSpec& spec,
                    const std::string& tokenizer_bytes, const TextTrainMemProbe& mem_probe)
{
    nn::TextTrainerHooks hooks;
    hooks.on_phase = [&mem_probe, &trainer](const char* tag)
    { mem_probe.mark(trainer.session().engine(), tag); };

    // 梯度统计（--grad-log；step 后、zero_grad 前）
    hooks.on_update =
        [&cfg, &trainer](nn::TrainingSession& session, std::size_t, std::size_t step, Scalar)
    {
        if (!cfg.grad_log) return;
        const std::size_t steps_per_epoch = trainer.steps_per_epoch();
        if ((step + 1) % cfg.log_interval != 0 && step + 1 != steps_per_epoch) return;
        std::cout << "\n  [grad] step " << step + 1 << ":" << std::endl;
        log_text_train_gradient_stats(session.engine(), session.model().param_gradients());
    };

#if NN_PROBES_ENABLED
    // NN_NAN_TRACE：batch 装配后快照 doc_ids/token（batch-major），backward 后扫描
    hooks.on_batch_ready = [](nn::TrainingSession&, const nn::TextSampleBatch& batch, std::size_t)
    {
        if (!nn_nan_trace_on()) return;
        nn_nan_trace_doc_ids() = batch.doc_ids;
        auto& toks = nn_nan_trace_tokens();
        toks.assign(batch.seq_len * batch.batch, 0);
        const auto in = batch.input.span();
        for (std::size_t b = 0; b < batch.batch; ++b)
            for (std::size_t t = 0; t < batch.seq_len; ++t)
                toks[b * batch.seq_len + t] = static_cast<std::size_t>(in[t * batch.batch + b]);
    };
    hooks.on_probe_check = [&cfg](nn::TrainingSession& session, std::size_t step)
    {
        if (!nn_nan_trace_on()) return false;
        if (nn_nan_trace_step(session.engine(), session.model(), step, cfg.seq_len)) return false;
        std::printf("[NAN-TRACE] 命中即停止（NN_NAN_TRACE=1）\n");
        return true;
    };
#endif

    // 定期断点（.nnckpt）：路径派生/记账/状态段序列化都在 CLI 侧。
    auto ckpt_log = std::make_shared<TextTrainCheckpointLog>();
    hooks.on_checkpoint = [&cfg, &spec, &tokenizer_bytes,
                           ckpt_log](const nn::TextCheckpointRequest& req) -> nn::Result<void>
    {
        const std::size_t before = ckpt_log->failures();
        save_ckpt_with_state(text_train_ckpt_path(cfg.save_path), req.session.model(), spec,
                             tokenizer_bytes, req.state, req.session.optimizer(), *ckpt_log,
                             std::cerr);
        if (ckpt_log->failures() != before) NN_FAIL("checkpoint 未落盘（记账见 stderr）");
        return {};
    };

    trainer.set_hooks(std::move(hooks));
    return trainer.fit();
}
} // namespace nn::cli
