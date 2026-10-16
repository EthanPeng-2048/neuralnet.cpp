// ── text_train_loop.hpp — text_train 的训练循环（epoch × step）────────────────
//
// P1-1 拆出的 train_loop 模块（docs/todo/18-roadmap.md §4.2）。本模块负责：
//   * 测试子集的窗口化与逐 epoch 评估（可选，数据集配套里带）；
//   * 每 step：采样 batch（滑动窗口）→ forward → 稀疏 CE → backward →
//     梯度累积 → 裁剪 → optimizer step → zero_grad；
//   * **非阻塞 loss 回读**（设备侧异步槽位，host 全程不等 GPU）与进度打印；
//   * `--max-steps` / `--save-interval` 的步数记账与定期断点保存
//     （断点记账语义见 cli/text_train_checkpoint.hpp）。
//
// 本模块**不终止进程**：全部错误经 `Result` 返回（P1-1 口径：`std::exit` 只允许
// 出现在 CLI 参数解析层）。唯一的例外是探针中止（`NN_NAN_TRACE` 命中）——那不是
// 错误而是"诊断完成"，经 `TextTrainLoopOutcome::abort_code = 9` 回报，调用方据此
// 跳过收尾保存（与旧 `return 9` 的行为逐字一致）。
//
// 循环内的**输出文案与顺序**属于回归锚：改动前后必须逐字节一致（见
// docs/todo/18 §4.2 P1-1 的验收口径）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "neuralnet.cpp/algebra_matrix.hpp"        // nn::Matrix
#include "neuralnet.cpp/compute_engine.hpp"        // nn::ComputeEngine
#include "neuralnet.cpp/compute_loss.hpp"          // nn::CrossEntropyLoss
#include "neuralnet.cpp/compute_optimizer.hpp"     // nn::Optimizer
#include "neuralnet.cpp/core_config.hpp"           // nn::Scalar / NN_PROBES_ENABLED
#include "neuralnet.cpp/core_errors.hpp"           // nn::Result / NN_TRY
#include "neuralnet.cpp/core_profile.hpp"          // NN_PROF_SCOPE / nn::prof::Phase
#include "neuralnet.cpp/dataset.hpp"               // nn::Dataset
#include "neuralnet.cpp/model_container.hpp"       // nn::Model
#include "neuralnet.cpp/model_spec.hpp"            // nn::ModelSpec
#include "neuralnet.cpp/cli/cli_lr_scheduler.hpp"  // compute_epoch_lr / compute_step_lr
#include "neuralnet.cpp/cli/text_train_checkpoint.hpp"
#include "neuralnet.cpp/cli/text_train_config.hpp"

namespace nn::cli
{
// ══ 非有限值追踪探针（NN_NAN_TRACE=1；探针门见 core_config.hpp）═══════════
// 探针门：默认构建（未 -DNN_ENABLE_PROBES=ON）里**整块不编译**——训练循环里
// 连判定分支都不存在（调用点同样被 #if 包住），环境变量 NN_NAN_TRACE 无效。
#if NN_PROBES_ENABLED
// 逐步扫描"参数 + 参数梯度"里的 NaN/Inf，命中第一个即打印
//（张量序号/形状/非有限元素数/最大有限值）并让训练立即停止。
// 用途：定位"训着训着 loss 变 nan"这类只在特定设备/数据顺序下复现的缺陷——
// 打印的序号即该步参数/梯度数组的下标，可与参数张量顺序对照定位到层。
inline bool nn_nan_trace_on()
{
    // MSVC CRT 弃用 getenv（-Werror 下必须走 _dupenv_s），与 NN_MEM_STATS 同款。
    static const bool on = [] {
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

inline bool nn_nan_trace_step(nn::ComputeEngine& engine, nn::Model& model,
                              std::size_t step, std::size_t seq_len)
{
    for (int pass = 0; pass < 2; ++pass)
    {
        const char* what = (pass == 0) ? "param" : "grad";
        std::vector<nn::TensorRef> refs = (pass == 0) ? model.parameters()
                                                      : model.param_gradients();
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
                    else
                        amax = std::max(amax, std::fabs(static_cast<double>(v)));
                }
                if (row_bad && bad_rows.size() < 32) bad_rows.push_back(r);
            }
            if (bad != 0)
            {
                std::printf("[NAN-TRACE] step %zu: %s[%zu] %zux%zu 非有限 %zu/%zu，"
                            "max|有限|=%.4g\n",
                            step, what, idx, m.rows(), m.cols(), bad,
                            m.rows() * m.cols(), amax);
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
                    std::printf("[NAN-TRACE] doc_ids 统计: 零可见列的行=%zu，最小可见列=%zu，"
                                "递减次数=%zu，唯一 id 数=%zu\n",
                                zero_visible, min_visible,
                                [&] { std::size_t d = 0;
                                      for (std::size_t t = 1; t < seq_len; ++t)
                                          if (ids[t] < ids[t - 1]) ++d;
                                      return d; }(),
                                [&] { auto v = ids;
                                      std::sort(v.begin(), v.begin() + seq_len);
                                      return static_cast<std::size_t>(
                                          std::unique(v.begin(), v.begin() + seq_len)
                                          - v.begin()); }());
                }
                // 行级定位：`grad[0]` 是 token embedding 的梯度，行号 = token id；
                // 把它映射回 batch 内的 (t,b) 位置，可区分"某些 id 坏"（数据/查表）
                // 与"某些位置坏"（掩码/位置相关算子）。
                if (pass == 1 && idx == 0 && !bad_rows.empty())
                {
                    std::printf("[NAN-TRACE] 非有限行(token id, 前 %zu): ",
                                bad_rows.size());
                    for (std::size_t k = 0; k < bad_rows.size(); ++k)
                        std::printf("%zu%s", bad_rows[k],
                                    k + 1 < bad_rows.size() ? "," : "");
                    std::printf("\n");
                    const auto& toks = nn_nan_trace_tokens();
                    const std::size_t bs = (seq_len > 0 && !toks.empty())
                                               ? toks.size() / seq_len : 0;
                    if (bs > 0)
                    {
                        std::printf("[NAN-TRACE] 命中位置 (t,b) 前 24: ");
                        std::size_t shown = 0;
                        for (std::size_t b = 0; b < bs && shown < 24; ++b)
                            for (std::size_t t = 0; t < seq_len && shown < 24; ++t)
                            {
                                const std::size_t id = toks[b * seq_len + t];
                                if (std::find(bad_rows.begin(), bad_rows.end(), id)
                                    != bad_rows.end())
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
void log_text_train_gradient_stats(nn::ComputeEngine &engine, const std::vector<nn::TensorRef> &grads)
{
    Scalar global_sum_sq = 0.0;
    Scalar global_abs_max = 0.0;
    Scalar global_abs_sum = 0.0;
    std::size_t global_count = 0;
    std::size_t tensor_idx = 0;

    for (auto& grad_ref : grads)
    {
        const auto& grad = grad_ref.get();
        // M1（docs/todo/17 §4.1）：张量读取一律经引擎下载到宿主
        // （GPU 直接下载；CPU 同样走 to_matrix，f16 存储也安全）
        nn::Matrix mat;
        {
            auto m = engine.to_matrix(grad);
            if (!m) continue;
            mat = std::move(*m);
        }

        const Scalar sum_sq = mat.reduce(Scalar{0}, std::plus<>{},
            [](Scalar x) { return x * x; });
        const Scalar abs_max = mat.reduce(Scalar{0},
            [](Scalar a, Scalar b) { return std::max(a, b); },
            [](Scalar x) { return std::abs(x); });
        const Scalar abs_sum = mat.reduce(Scalar{0}, std::plus<>{},
            [](Scalar x) { return std::abs(x); });
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
                      << "norm=" << std::scientific << std::setprecision(4)
                      << std::sqrt(sum_sq) << "  max|g|=" << abs_max
                      << "  mean|g|=" << (n > 0 ? abs_sum / n : Scalar{0})
                      << std::endl;
        }
        ++tensor_idx;
    }

    const Scalar global_norm = std::sqrt(global_sum_sq);
    const Scalar global_mean = global_count > 0 ? global_abs_sum / global_count : Scalar{0};
    std::cout << "    >> grad_norm=" << std::scientific << std::setprecision(4) << global_norm
              << "  max|g|=" << global_abs_max
              << "  mean|g|=" << global_mean
              << "  params=" << global_count
              << "  tensors=" << grads.size()
              << std::endl;
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
        std::cout << "[mem] " << std::left << std::setw(16) << tag << std::right
                  << " " << engine.pool_stats() << std::endl;
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

// ── 训练循环输入：调用方（text_train 主流程）把已算好的数据集视图与组件传进来 ──
struct TextTrainLoopInput
{
    const TrainConfig& cfg;
    nn::ComputeEngine& engine;
    nn::Model& model;
    nn::CrossEntropyLoss& ce_loss;
    nn::Optimizer& optimizer;
    const nn::Dataset& dataset;
    const std::vector<std::size_t>& window_offsets;
    const std::vector<std::size_t>& token_flow;
    const std::vector<std::size_t>& flow_doc_ids;
    const std::vector<unsigned char>& flow_assistant;
    const nn::ModelSpec& spec;
    const std::string& tokenizer_bytes;
    std::size_t pad_id = 0;
    std::size_t vocab_size = 0;
    std::size_t stride = 0;
    TextTrainMemProbe mem_probe{};
};

// ── 训练循环结果 ───────────────────────────────────────────────────────
struct TextTrainLoopOutcome
{
    Scalar seconds = 0;              // 循环墙钟（秒；与旧 t_end - t_start 同口径）
    std::uint64_t elapsed_ns = 0;    // 同一区间的纳秒数（NN_PROFILE 报告用）
    std::size_t steps_executed = 0;  // 本进程执行的训练步数（--max-steps 计数）
    std::size_t ckpt_failures = 0;   // 断点保存失败次数（收尾退出码据此）
    int abort_code = 0;              // 非 0 = 训练被探针中止（9）：跳过收尾保存
};

// ── 训练循环（epoch × step）────────────────────────────────────────────
// 每样本 = 一个 seq_len 滑动窗口（可能跨行），目标 = 输入左移一位。
[[nodiscard]] inline nn::Result<TextTrainLoopOutcome> run_text_train_loop(
    const TextTrainLoopInput& in)
{
    const TrainConfig& cfg = in.cfg;
    nn::ComputeEngine* engine = &in.engine;
    nn::Model& model = in.model;
    nn::CrossEntropyLoss& ce_loss = in.ce_loss;
    nn::Optimizer& optimizer = in.optimizer;
    const nn::Dataset& dataset = in.dataset;
    const std::vector<std::size_t>& window_offsets = in.window_offsets;
    const std::vector<std::size_t>& token_flow = in.token_flow;
    const std::vector<std::size_t>& flow_doc_ids = in.flow_doc_ids;
    const std::vector<unsigned char>& flow_assistant = in.flow_assistant;
    const std::size_t pad_id = in.pad_id;
    const std::size_t vocab_size = in.vocab_size;
    const std::size_t stride = in.stride;
    const std::string& loss_scope = dataset.info().loss_scope;
    const auto mem_mark = [&](const char* tag) { in.mem_probe.mark(*engine, tag); };
    TextTrainCheckpointLog ckpt_log;

    // ── 学习率调度配置（委托给 nn::cli::compute_epoch_lr / compute_step_lr）──
    // 全部字段派生自 cfg：epoch 级（cosine）与 step 级（step_cosine）各一份。
    nn::cli::LrScheduleConfig lr_sched_cfg;
    lr_sched_cfg.base_lr = cfg.lr;
    lr_sched_cfg.min_lr = cfg.min_lr;
    lr_sched_cfg.warmup_epochs = cfg.warmup_epochs;
    lr_sched_cfg.total_epochs = cfg.epochs;
    lr_sched_cfg.schedule = cfg.lr_schedule;
    lr_sched_cfg.lr_per_epoch = cfg.lr_per_epoch;
    // 当前生效的 lr（初始 = --lr；set_lr 只在变化时调用，与旧行为一致）
    Scalar optimizer_current_lr = cfg.lr;
    // ── 测试子集（可选，数据集配套里带）────────────────────────
    std::vector<std::size_t> test_window_offsets;
    std::vector<std::size_t> test_flow;
    std::vector<std::size_t> test_flow_doc_ids;
    std::vector<unsigned char> test_flow_assistant;  // 与 test_flow 等长（all 模式全 1）
    if (dataset.info().test_docs > 0)
    {
        NN_TRY_MSG(test_corpus_r, "读取测试子集失败", dataset.load_text(true));
        test_flow = std::move(test_corpus_r->token_flow);
        test_flow_doc_ids = std::move(test_corpus_r->doc_ids);
        test_flow_assistant = std::move(test_corpus_r->loss_mask);

        if (loss_scope == "assistant")
        {
            std::size_t t_asst = 0;
            for (unsigned char m : test_flow_assistant) t_asst += m;
            std::cout << "测试集 assistant 掩码: " << t_asst << " / "
                      << test_flow.size() << " tokens 参与评估\n";
        }

        // 测试窗口：保留全部窗口（与训练一致，含跨文档；文档感知掩码 kernel 内生效）
        for (std::size_t pos = 0; pos < test_flow.size(); pos += stride)
        {
            test_window_offsets.push_back(pos);
        }
        std::cout << "测试集: " << dataset.info().test_docs << " docs  样本数: "
                  << test_window_offsets.size()
                  << "  tokens: " << test_flow.size() << std::endl;
    }
    // ── 步数与采样：每 epoch 每样本恰好访问一次 ──────────────
    // steps_per_epoch = ceil(样本数 / batch_size)：最后一个 batch 不满时
    // 以实际 this_bs 参与训练。由于 loss 已按有效 token 归一化，不满 batch
    // 的梯度与满 batch 同尺度（每 epoch 仅一个不满 batch，影响可忽略）。
    std::size_t steps_per_epoch =
        (window_offsets.size() + cfg.batch_size - 1) / cfg.batch_size;
    if (steps_per_epoch == 0)
        NN_FAIL("样本数 (" + std::to_string(window_offsets.size()) + ") 小于 batch_size (" +
                std::to_string(cfg.batch_size) + ")，请减小 --batch-size 或增大训练语料");

    // ── Step 级学习率配置（step_cosine 模式） ──
    nn::cli::StepLrScheduleConfig step_lr_cfg;
    step_lr_cfg.base_lr = cfg.lr;
    step_lr_cfg.min_lr = cfg.min_lr;
    step_lr_cfg.warmup_steps = static_cast<int>(cfg.warmup_steps);
    step_lr_cfg.total_steps =
        static_cast<int>(steps_per_epoch * static_cast<std::size_t>(cfg.epochs));
    step_lr_cfg.cosine = true;

    // 随机种子：每次训练/每次 --resume 续训后的样本顺序都不同，避免跨进程
    // 数据顺序完全一致导致 GUI 图表上出现"loss 片段重复"（与 mnist_train 一致）。
    std::mt19937_64 rng{std::random_device{}()};
    std::vector<std::size_t> sample_indices(window_offsets.size());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;

    // ── 损失覆盖范围 ─────────────────────────────────────────────
    // 全部 seq_len 位置参与 loss。（ZiPT「压缩模式仅窗口位置参与 loss」
    // 分支已整体移除，见 docs/history/history.md。）
    const std::size_t eff_seq = cfg.seq_len;

    // ── 预分配 batch 缓冲区（末批不满时 resize 到实际 this_bs） ──
    // x_tokens: (seq_len, batch) 输入 token IDs（每窗口一列）
    // y_tokens: (seq_len, batch) 目标 token IDs（x 左移一位）
    // 最后一个窗口不足 seq_len 时用 pad_id 填充，目标对应位置也用 pad_id
    nn::Matrix x_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix y_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix loss_mask(cfg.seq_len, cfg.batch_size);
    std::vector<std::size_t> flat_targets(eff_seq * cfg.batch_size);

    auto t_start = std::chrono::steady_clock::now();

    // --max-steps：按本进程执行的步数计数（与 --resume 组合可分段跑）
    std::size_t steps_executed = 0;
    bool stop_training = false;
    // 断点保存记账（P1-1 退出码语义，docs/todo/18 §4.2）：只记账不中断训练
    // （磁盘抖动不该毁掉整轮），收尾由调用方据此给非零退出码。
    // 记账与文案在 cli/text_train_checkpoint.hpp（本模块只调用）。

    for (int epoch = cfg.start_epoch; epoch < cfg.epochs; ++epoch)
    {
        // ── 学习率调度：每 epoch 开始时调整（step_cosine 由 step 级调度接管） ──
        if (cfg.lr_schedule != "step_cosine")
        {
            Scalar epoch_lr = nn::cli::compute_epoch_lr(lr_sched_cfg, epoch);
            if (epoch_lr != optimizer_current_lr)
            {
                optimizer.set_lr(epoch_lr);
                optimizer_current_lr = epoch_lr;
            }
        }

        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_weighted = 0.0;  // Σ(loss × 有效token)，用于按 token 加权平均
        std::size_t total_valid = 0;  // 累计有效 token 数

        // 每个 epoch 开始前 shuffle 样本索引队列
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        // TDR 重启续训：resume 时本 epoch 内跳过已完成的前 start_step 步，
        // 避免重启后从 epoch 开头重跑、GUI 图表出现重复 loss 片段。
        const std::size_t epoch_first_step =
            (epoch == cfg.start_epoch) ? cfg.start_step : 0;

        // 梯度累积：距上次参数更新的步数（每 accum_steps 步更新一次）
        std::size_t steps_since_update = 0;

        // ── 非阻塞 loss 回读 ──────────────────────────────────────
        // 每步把设备端 loss_sum 排入一个异步回读槽位（**不等待**），稍后就绪
        // 即取。取 loss 不做 end_batch + wait_in_flight：loss 走 device 侧异步
        // 回读，host 录制的帧不被打断——host 一旦在录制中途等待，就会 drain
        // 整条流水线，GPU 在 host 录制期间空转 → 占用率锯齿。
        // 稳定态下第 N 步即可取到第 N-1 步的 loss，打印仍是每步一条。
        struct PendingLoss
        {
            std::size_t slot = 0;
            std::size_t step = 0;
            std::size_t valid = 0;
            Scalar inv_num_valid = Scalar{0};
            nn::Tensor keepalive;   // 回读命令引用的张量须活到就绪
        };
        std::deque<PendingLoss> pending_loss;
        const std::size_t loss_slots =
            std::max<std::size_t>(engine->scalar_readback_slots(), 1);
        std::size_t loss_slot_next = 0;

        // 非阻塞收割：消费所有已就绪的回读（累计统计 + 按 log_interval 打印）
        auto harvest_loss = [&]() -> nn::Result<void>
        {
            while (!pending_loss.empty())
            {
                Scalar sum = Scalar{0};
                auto pr = engine->poll_scalar_readback(pending_loss.front().slot, sum);
                NN_TRY_CHECK(pr);
                if (!*pr) break;   // 未就绪：留待下次（不阻塞）
                PendingLoss pl = std::move(pending_loss.front());
                pending_loss.pop_front();
                const Scalar loss = -sum * pl.inv_num_valid;
                total_weighted += loss * static_cast<Scalar>(pl.valid);
                total_valid += pl.valid;
                if ((pl.step + 1) % cfg.log_interval == 0 || pl.step + 1 == steps_per_epoch)
                {
                    std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                              << "  step " << pl.step + 1 << "/" << steps_per_epoch
                              << "  loss: " << std::fixed << std::setprecision(4) << loss
                              << "   " << std::flush;
                }
            }
            return {};
        };
        // 阻塞兜底收割（槽位将满 / epoch 收尾）：等的是"已在 GPU 上跑的旧帧"，
        // GPU 不会因此空转；正常路径不触发。
        auto drain_loss = [&]() -> nn::Result<void>
        {
            while (!pending_loss.empty())
            {
                auto r0 = harvest_loss();
                if (!r0) return r0;
                if (pending_loss.empty()) break;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            return {};
        };

        for (std::size_t step = epoch_first_step; step < steps_per_epoch; ++step)
        {
            // ── step 级学习率：每个训练步更新（step_cosine 模式） ──
            if (cfg.lr_schedule == "step_cosine")
            {
                const int global_step =
                    static_cast<int>(epoch) * static_cast<int>(steps_per_epoch) +
                    static_cast<int>(step);
                Scalar step_lr = nn::cli::compute_step_lr(step_lr_cfg, global_step);
                if (step_lr != optimizer_current_lr)
                {
                    optimizer.set_lr(step_lr);
                    optimizer_current_lr = step_lr;
                }
            }

            // ── --max-steps：达到上限的那一步为最后一步，同时强制完成一次
            // 参数更新（把梯度累积中未满的累积刷进参数，不丢半截累积）──
            const bool is_final_step =
                (cfg.max_steps > 0) && (steps_executed + 1 >= cfg.max_steps);
            ++steps_executed;

            // ── step 剖析（NN_PROFILE=1 时生效）────────────────────────
            // "step" = 每步伞形 scope；prof 划分连续阶段（各阶段共享局部
            // 变量，不能用花括号块隔开，故用显式 start/stop）。
            NN_PROF_SCOPE("step");
            nn::prof::Phase prof;
            prof.start("data");

            // ── 采样 batch：每样本 = 一个滑动窗口 ─────────────
            // 按 shuffle 后的顺序切片取 this_bs 个窗口（末批可能不满）。
            // 每 epoch 每样本恰好访问一次。
            const std::size_t offset = step * cfg.batch_size;
            const std::size_t this_bs = std::min(cfg.batch_size, window_offsets.size() - offset);
            if (x_tokens.cols() != this_bs)
            {
                x_tokens.resize(cfg.seq_len, this_bs);
                y_tokens.resize(cfg.seq_len, this_bs);
                loss_mask.resize(cfg.seq_len, this_bs);
                flat_targets.resize(eff_seq * this_bs);
            }
            std::size_t step_valid = 0;  // 本 step 参与 loss 的有效 token 数

            for (std::size_t b = 0; b < this_bs; ++b)
            {
                const std::size_t win_pos = window_offsets[sample_indices[offset + b]];
                // 窗口有效长度：不超过 token_flow 末尾时为 seq_len，
                // 否则为剩余 token 数（最后一个窗口，PAD 位置被屏蔽）
                const std::size_t win_len = std::min(cfg.seq_len, token_flow.size() - win_pos);

                // 填充 x/y/mask：x[t] = flow[win_pos+t], y[t] = flow[win_pos+t+1]
                for (std::size_t t = 0; t < cfg.seq_len; ++t)
                {
                    const std::size_t x_id = (t < win_len) ? token_flow[win_pos + t] : pad_id;
                    const std::size_t y_id = (t + 1 < win_len) ? token_flow[win_pos + t + 1] : pad_id;
                    x_tokens.set_value_unchecked(t, b, static_cast<Scalar>(x_id));
                    y_tokens.set_value_unchecked(t, b, static_cast<Scalar>(y_id));
                    // 仅真实位置（非 padding）参与 loss；
                    // 且 loss_mask=1（生成期定好；assistant 段外为 0）
                    const bool participate = (t + 1 < win_len) &&
                        (flow_assistant.empty() || flow_assistant[win_pos + t + 1]);
                    loss_mask.set_value_unchecked(t, b, participate ? 1.0f : 0.0f);
                    if (participate) ++step_valid;
                }
            }

            // ── 文档感知 ────────────────────────────────────────
            // 行边界即文档边界（每行 = 一篇文档，见 build_flow_doc_ids）。
            // 为每窗口设 doc_ids 施加块对角掩码；fold 掩码变体（Doc/AlibiDoc）
            // 使该掩码在 kernel 内生效（不物化 (BH·seq,seq)），无需再丢弃跨文档窗口。
            std::vector<std::size_t> doc_ids;
            if (!flow_doc_ids.empty())
            {
                doc_ids.assign(cfg.seq_len * this_bs, 0);
                for (std::size_t b = 0; b < this_bs; ++b)
                {
                    const std::size_t win_pos = window_offsets[sample_indices[offset + b]];
                    const std::size_t win_len =
                        std::min(cfg.seq_len, token_flow.size() - win_pos);
                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                        if (t < win_len)
                            doc_ids[b * cfg.seq_len + t] = flow_doc_ids[win_pos + t];
                }
            }
            model.set_doc_ids(doc_ids);
#if NN_PROBES_ENABLED
            if (nn_nan_trace_on())
            {
                nn_nan_trace_doc_ids() = doc_ids;   // 诊断探针用（默认不维护）
                auto& tr_toks = nn_nan_trace_tokens();
                tr_toks.assign(cfg.seq_len * this_bs, 0);
                for (std::size_t b = 0; b < this_bs; ++b)
                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                        tr_toks[b * cfg.seq_len + t] =
                            static_cast<std::size_t>(x_tokens.at(t, b));
            }
#endif

            // ── Matrix → Tensor（上传到引擎设备） ──────────────
            NN_TRY_MSG(x_tensor_r, "\nfrom_matrix(x_tokens) failed",
                       engine->from_matrix(x_tokens));

            // ── 构造平坦标签（与 logits 列序一致：batch-major，i = b*eff_seq + t） ──
            // 模型 forward 对输入 transpose 后按 batch-major 列序输出 logits，
            // 因此 flat_targets / flat_mask 必须与之一一对应（b 外层、t 内层）。
            auto y_span = y_tokens.span();
            auto m_span = loss_mask.span();
            std::vector<Scalar> flat_mask(eff_seq * this_bs);
            for (std::size_t b = 0; b < this_bs; ++b)
                for (std::size_t t = 0; t < eff_seq; ++t)
                {
                    const std::size_t src_t = t;                 // 窗口内真实位置
                    const std::size_t pm = src_t * this_bs + b;  // 源的 position-major 索引
                    const std::size_t bm = b * eff_seq + t;     // 目标的 batch-major 索引
                    flat_targets[bm] = static_cast<std::size_t>(y_span[pm]);
                    flat_mask[bm] = m_span[pm];
                }

            // ── 启用 GPU batch 录制 ─────────────────────────
            // 整个 forward + backward + optimizer step 录制到一个 command buffer，
            // end_batch 时一次 vkQueueSubmit + vkWaitForFences，消除 per-primitive 同步开销。
            // CPU 引擎 begin_batch/end_batch 为 no-op，所以两套引擎都安全。
            prof.start("forward");
            NN_TRY_MSG(begin_r, "begin_batch failed", engine->begin_batch());

            // ── 前向传播 ─────────────────────────────────────
            NN_TRY_MSG(fwd_result, "前向传播失败", model.forward(*x_tensor_r));
            auto logits = std::move(*fwd_result);
            mem_mark("step/forward");
            // logits: (vocab_size, seq_len × batch_size)

            // ── 损失（稀疏标签，避免 one-hot 爆显存）────────
            // 此处不做同步读回 loss：forward_sparse_sum 只算出设备端
            // (1,1) loss 和（Σ loss_vec，未归一化）；flush 提交 forward 帧后
            // 把它排入异步回读槽位，host 立刻继续录制 backward——全程不 drain。
            // loss 只在 device 侧回读、不在 host 侧判定；数值稳定性由
            // --max-norm 梯度裁剪 + 观察 loss 曲线负责。
            auto mask_span = std::span<const Scalar>(flat_mask);
            std::size_t loss_num_valid = 0;
            NN_TRY_MSG(loss_sum_t, "损失前向（稀疏 CE）失败",
                       ce_loss.forward_sparse_sum(*engine, logits, flat_targets, mask_span,
                                                  vocab_size, loss_num_valid, /*grad_reuse=*/&logits));
            // 回读固定按 F32 取 4 字节：非 F32 时在 batch 内先 cast
            //（录制态，不提交、不 drain）
            NN_TRY_MSG(loss_sum_f32, "loss 回读 cast 失败",
                       engine->cast(*loss_sum_t, nn::Precision::F32));

            // ── 显存优化：logits 已消费完毕，立即释放（必须在 flush 之前）──
            //   延迟销毁按"当前录制帧"打标签：flush 前释放 → 标签为 forward
            //   帧，其 fence 在 backward 录制期间即完成，可被中途 reap 回收；
            //   若放在 flush 之后释放，标签落到 backward 帧，513MB
            //   （vocab×seq×batch）要等整个 backward 提交完成才还池（探针实测
            //   pending 主项）。此时 loss 已算出，LM Head 的 backward 只需
            //   input + grad_output，且 forward 帧已录完对 logits 的引用。
            logits = {};
            mem_mark("step/logits-free");

            // ── 中点刷新：提交 forward+loss，拆分为两次 GPU 提交 ──
            // 大词表 + 长序列时 forward+backward 单次提交可能触发 TDR 超时。
            // 在 forward 与 backward 之间 flush，将一次大提交拆为两次小提交。
            prof.start("submit-fwd");
            NN_TRY_MSG(flush_r, "\nflush_batch (forward) failed", engine->flush_batch());
            mem_mark("step/loss-fwd");

            // forward 帧已提交 → 排队异步回读（同队列 FIFO：拷贝在生产者之后）
            {
                // 槽位将满：先非阻塞收割，必要时阻塞兜底（保证不覆盖未取走的值）
                NN_TRY_MSG(hv0, "harvest_loss failed", harvest_loss());
                if (pending_loss.size() >= loss_slots)
                {
                    NN_TRY_MSG(hv1, "drain_loss failed", drain_loss());
                }
                PendingLoss pl;
                pl.slot = loss_slot_next;
                pl.step = step;
                pl.valid = step_valid;
                pl.inv_num_valid = (loss_num_valid > 0)
                    ? Scalar{1} / static_cast<Scalar>(loss_num_valid) : Scalar{0};
                pl.keepalive = std::move(*loss_sum_f32);
                NN_TRY_MSG(sr, "submit_scalar_readback failed",
                           engine->submit_scalar_readback(pl.slot, pl.keepalive));
                loss_slot_next = (loss_slot_next + 1) % loss_slots;
                pending_loss.push_back(std::move(pl));
            }

            // ── 反向传播（梯度已含 mask，无需额外处理） ────────
            prof.start("backward");
            NN_TRY_MSG(grad_result, "\nLoss backward failed", ce_loss.backward());

            // 梯度积累缩放：每步梯度除以 accum_steps
            //   forward_sparse 的梯度为 (softmax-one_hot)/num_valid（单步平均），
            //   积累后需除以 accum_steps 才能等价于有效 batch 的平均梯度。
            //   否则梯度是正确值的 accum_steps 倍，Adam 虽近似抵消但非精确。
            if (cfg.accum_steps > 1)
            {
                NN_TRY_MSG(scale_r, "\n梯度缩放失败",
                           engine->scale_inplace(*grad_result,
                               Scalar{1} / static_cast<Scalar>(cfg.accum_steps)));
            }

            NN_TRY_MSG(bwd_result, "反向传播失败", model.backward(*grad_result));
            mem_mark("step/backward");

            // ── 提交 backward batch（单独一次提交，已与 forward 拆分） ──
            prof.start("sync");
            NN_TRY_MSG(bwd_end, "\nend_batch (backward) failed", engine->end_batch());
            mem_mark("step/end-batch");

            // ── 非有限值追踪探针（NN_NAN_TRACE=1；默认零开销）────────────
            //   在"梯度已定稿、参数尚未更新"的时点扫描两者，命中即打印张量
            //   序号并停止训练（见 nn_nan_trace_step 头注释）。
#if NN_PROBES_ENABLED
            if (nn_nan_trace_on() &&
                !nn_nan_trace_step(*engine, model, step, cfg.seq_len))
            {
                std::printf("[NAN-TRACE] 命中即停止（NN_NAN_TRACE=1）\n");
                TextTrainLoopOutcome aborted;
                aborted.steps_executed = steps_executed;
                aborted.ckpt_failures = ckpt_log.failures();
                aborted.abort_code = 9;   // 探针中止：调用方据此跳过收尾保存
                return aborted;
            }
#endif

            // ── 收割已就绪的 loss 回读（非阻塞；打印/统计在此推进）─────────
            // 稳定态：此刻第 N-1 步的 loss 早已写回（GPU 一直在跑），
            // poll 立即命中 → 每步一条打印，且 host 一秒都不等 GPU。
            {
                NN_TRY_MSG(hv, "harvest_loss failed", harvest_loss());
            }

            // ── 显存回收（L2）：end_batch 提交完成、延迟销毁已 flush，
            //    归还完全空闲的内存池底材（GPU 引擎有效，CPU 引擎 no-op） ──
            NN_TRY_MSG(rel_r, "\n显存回收失败", engine->release_idle_pool_blocks());
            mem_mark("step/released");

            // ── 显存优化：logits 梯度已消费完毕，立即释放 ──
            //   model.backward 已把 grad 传播到各参数梯度，grad_result（1.6GB）
            //   已消费完毕。end_batch 之后 backward 已提交执行完，释放安全。
            grad_result = {};

            // ── 梯度累积：每 accum_steps 步才更新一次参数 ──
            // forward+backward 每步都执行（梯度累加到参数梯度），
            // 梯度裁剪/step/zero_grad 仅在累积到 accum_steps 或 epoch 末尾执行。
            ++steps_since_update;
            const bool do_update =
                (steps_since_update >= cfg.accum_steps) ||
                (step + 1 == steps_per_epoch) || is_final_step;

            if (do_update)
            {
                prof.start("optimizer");
                // ── 梯度裁剪（在 step() 之前，backward() 之后） ──
                // clip_grad_norm 逐原语在 batch **外**会各自
                // submit_and_wait（约 200 次 host↔GPU 往返/次裁剪）；包进
                // 一个 batch 后只剩一次范数下载同步（每 accum_steps 步一次，
                // 而非每步）。这是范数的数据依赖，无法异步化。
                if (cfg.max_norm > 0)
                {
                    NN_TRY_MSG(clip_begin, "begin_batch (clip) failed",
                               engine->begin_batch());
                    NN_TRY_MSG(clip_r, "\n梯度裁剪失败",
                               optimizer.clip_grad_norm(cfg.max_norm));
                    NN_TRY_MSG(clip_end, "end_batch (clip) failed", engine->end_batch());
                }

                NN_TRY_MSG(opt_begin, "begin_batch (optimizer) failed",
                           engine->begin_batch());

                // ── 优化器 step + 梯度清零 ──
                NN_TRY_MSG(step_result, "优化器 step 失败", optimizer.step());

                // ── 梯度统计（step 后、zero_grad 前） ──
                if (cfg.grad_log && ((step + 1) % cfg.log_interval == 0 || step + 1 == steps_per_epoch))
                {
                    std::cout << "\n  [grad] step " << step + 1 << ":" << std::endl;
                    log_text_train_gradient_stats(*engine, model.param_gradients());
                }

                NN_TRY_MSG(zero_result, "\n优化器 zero_grad 失败", optimizer.zero_grad());

                // ── 提交 batch：一次 vkQueueSubmit + vkWaitForFences ──
                NN_TRY_MSG(end_r, "end_batch (optimizer) failed", engine->end_batch());

                steps_since_update = 0;
            }

            // ── 定期保存 checkpoint（独立于 log_interval；save_interval=0 禁用） ──
            if (cfg.save_interval > 0 &&
                ((step + 1) % cfg.save_interval == 0 || step + 1 == steps_per_epoch))
            {
                prof.start("ckpt");
                // 训练断点（Checkpoint 角色）：含训练参数（三值层 = f32 latent）。
                // 路径派生 / 失败记账与文案都在 checkpoint 模块里。
                ckpt_log.save(text_train_ckpt_path(cfg.save_path), model, in.spec,
                              in.tokenizer_bytes, std::cerr);
            }

            // ── --max-steps 到达：本步已完成更新与保存，退出步循环 ──
            if (is_final_step)
            {
                stop_training = true;
                break;
            }

            // 进度显示在 harvest_loss 内完成（loss 走异步回读）：按"值就绪即打印"，
            // 稳定态仍是每步一条，且 host 全程不等 GPU。
        }

        // ── epoch 收尾：读完最后几步尚未取回的 loss（各 epoch 一次）──────
        // 等的是已经在 GPU 上执行的旧帧，不影响 GPU 占空比。
        {
            NN_TRY_MSG(ep_drain, "loss drain failed", drain_loss());
        }

        // ── --max-steps 到达：跳过本 epoch 的统计与测试评估，直接收尾 ──
        if (stop_training)
        {
            std::cout << "\n已达 --max-steps=" << cfg.max_steps << "（本进程执行 "
                      << steps_executed << " 步），提前结束训练\n";
            break;
        }

        auto ep_end = std::chrono::steady_clock::now();
        Scalar ep_sec = std::chrono::duration<Scalar>(ep_end - ep_start).count();
        // 按有效 token 加权平均（等价于全局 per-token 平均），
        // 避免"有效 token 少的 batch"拉偏 epoch loss 报告
        Scalar avg_loss = (total_valid > 0)
            ? total_weighted / static_cast<Scalar>(total_valid)
            : Scalar{0};

        std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                  << "  lr=" << std::scientific << std::setprecision(4) << optimizer_current_lr
                  << "  avg_loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  time=" << std::setprecision(1) << ep_sec << "s";

        // ── 测试集评估（可选，与训练一致的滑动窗口） ─────────────
        if (!test_window_offsets.empty())
        {
            Scalar test_total_weighted = 0.0;
            std::size_t test_total_valid = 0;
            const std::size_t test_bs = std::min(cfg.batch_size, test_window_offsets.size());
            const std::size_t test_steps_per_epoch =
                (test_window_offsets.size() + test_bs - 1) / test_bs;

            for (std::size_t tstep = 0; tstep < test_steps_per_epoch; ++tstep)
            {
                const std::size_t toffset = tstep * test_bs;
                const std::size_t this_bs = std::min(test_bs, test_window_offsets.size() - toffset);
                if (x_tokens.cols() != this_bs)
                {
                    x_tokens.resize(cfg.seq_len, this_bs);
                    y_tokens.resize(cfg.seq_len, this_bs);
                    loss_mask.resize(cfg.seq_len, this_bs);
                    flat_targets.resize(eff_seq * this_bs);
                }

                // 填充 batch（与训练一致：末窗不足时 PAD 并屏蔽）
                std::size_t test_valid = 0;
                for (std::size_t b = 0; b < this_bs; ++b)
                {
                    const std::size_t win_pos = test_window_offsets[toffset + b];
                    const std::size_t win_len = std::min(cfg.seq_len, test_flow.size() - win_pos);
                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                    {
                        const std::size_t x_id = (t < win_len) ? test_flow[win_pos + t] : pad_id;
                        const std::size_t y_id = (t + 1 < win_len) ? test_flow[win_pos + t + 1] : pad_id;
                        x_tokens.set_value(t, b, static_cast<Scalar>(x_id));
                        y_tokens.set_value(t, b, static_cast<Scalar>(y_id));
                        // 仅真实位置（非 padding）参与评估；
                        // 且 loss_mask=1（生成期定好；assistant 段外为 0）
                        const bool participate = (t + 1 < win_len) &&
                            (test_flow_assistant.empty() ||
                             test_flow_assistant[win_pos + t + 1]);
                        loss_mask.set_value(t, b, participate ? 1.0f : 0.0f);
                        if (participate) ++test_valid;
                    }
                }

                // Forward pass only（不 backward）
                auto x_tensor_r = engine->from_matrix(x_tokens);
                if (!x_tensor_r) { std::cerr << "  测试 from_matrix 失败: " << x_tensor_r.error().message << '\n'; break; }

                // 文档感知：与训练一致；为每窗口设 doc_ids（fold 块对角掩码）
                {
                    std::vector<std::size_t> test_doc_ids;
                    if (!test_flow_doc_ids.empty())
                    {
                        test_doc_ids.assign(cfg.seq_len * this_bs, 0);
                        for (std::size_t b = 0; b < this_bs; ++b)
                        {
                            const std::size_t win_pos = test_window_offsets[toffset + b];
                            const std::size_t win_len =
                                std::min(cfg.seq_len, test_flow.size() - win_pos);
                            for (std::size_t t = 0; t < cfg.seq_len; ++t)
                                if (t < win_len)
                                    test_doc_ids[b * cfg.seq_len + t] =
                                        test_flow_doc_ids[win_pos + t];
                        }
                    }
                    model.set_doc_ids(test_doc_ids);
                }

                // 构建 flat_targets / flat_mask（与 logits 列序一致：batch-major，i = b*eff_seq + t）
                auto y_span = y_tokens.span();
                auto mm_span = loss_mask.span();
                std::vector<Scalar> tmask(eff_seq * this_bs);
                for (std::size_t b = 0; b < this_bs; ++b)
                    for (std::size_t t = 0; t < eff_seq; ++t)
                    {
                        const std::size_t src_t = t;                  // 窗口内真实位置
                        const std::size_t pm = src_t * this_bs + b;   // 源 position-major 索引
                        const std::size_t bm = b * eff_seq + t;  // 目标 batch-major 索引
                        flat_targets[bm] = static_cast<std::size_t>(y_span[pm]);
                        tmask[bm] = mm_span[pm];
                    }

                auto begin_r = engine->begin_batch();
                if (!begin_r) { std::cerr << "  测试 begin_batch 失败: " << begin_r.error().message << '\n'; break; }

                auto fwd_result = model.forward(*x_tensor_r);
                if (!fwd_result) { std::cerr << "  测试前向传播出错: " << fwd_result.error().message << '\n'; break; }
                auto test_mask_span = std::span<const Scalar>(tmask);
                auto loss_result = ce_loss.forward_sparse(
                    *engine, *fwd_result, flat_targets, test_mask_span, vocab_size);

                auto end_r = engine->end_batch();
                if (!end_r) { std::cerr << "  测试 end_batch 失败: " << end_r.error().message << '\n'; break; }

                if (!loss_result) { std::cerr << "  测试评估出错: " << loss_result.error().message << '\n'; break; }
                Scalar batch_loss = *loss_result;

                // 按有效 token 加权（与训练端 per-token 平均一致）
                test_total_weighted += batch_loss * static_cast<Scalar>(test_valid);
                test_total_valid += test_valid;
            }

            Scalar test_avg_loss = (test_total_valid > 0)
                ? test_total_weighted / static_cast<Scalar>(test_total_valid)
                : Scalar{0};
            std::cout << "  test_loss=" << std::fixed << std::setprecision(4) << test_avg_loss;
        }

        std::cout << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    TextTrainLoopOutcome outcome;
    outcome.seconds = std::chrono::duration<Scalar>(t_end - t_start).count();
    outcome.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    outcome.steps_executed = steps_executed;
    outcome.ckpt_failures = ckpt_log.failures();
    return outcome;
}
} // namespace nn::cli
