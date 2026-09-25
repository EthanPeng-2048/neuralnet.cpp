// ── mem_probe — 运行时显存分项探针 ─────────────────────────────────────────
//
// 目的：回答"显存都花在哪了"。在训练/推理的每个生命周期阶段采样：
//   1. Vulkan 双内存池（persist / transient）：块数、已提交底材、device/host、
//      live（实际在用）、free（空闲）、碎片比；
//   2. 整卡显存（nvidia-smi，含驱动开销）；
//   3. 逻辑分项：参数 / 梯度 / 优化器状态 / 激活 / logits / CE梯度 / KV cache。
//
// 阶段序列（训练，镜像 text_train 的真实顺序）：
//   pre-init → engine-init → model-built → optimizer-created
//   → step{i}/forward → step{i}/loss-fwd → step{i}/logits-freed
//   → step{i}/backward → step{i}/end-batch → step{i}/released
//   → step{i}/optimizer-step
// 推理（KV）：kv/pre → kv/caches-allocated → kv/after-steps → kv/end-batch → kv/freed
//
// 用法（需 Vulkan；无 Vulkan 构建返回 77 = SKIP）：
//   mem_probe [--gpu=<dev>] [--f16] [--checkpoint-every N] [--activation-offload]
//             [--batch N] [--seq N] [--steps N] [--vocab N] [--d-model N]
//             [--num-heads N] [--num-layers N] [--d-ff N] [--optimizer name]
//             [--no-kv] [--kv-steps N]
//
// 默认参数 = docs/benchmarks/2026-09-25 的 bench 配置（vocab 8208 / d64 /
// h4 / L4 / ff256 / seq256 / batch64），与已知"峰值 5.7GB"场景对齐。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持（NN_HAS_VULKAN）。\n";
    return 77;
}
#else

#include <neuralnet.cpp/backend/compute_vk_backend.hpp>

using nn::Scalar;

namespace
{

// ── nvidia-smi 整卡采样（含驱动开销；多卡取最大值）────────────────────────
[[nodiscard]] long long smi_used_mb()
{
#ifdef _WIN32
    FILE* f = _popen(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>nul", "r");
#else
    FILE* f = popen(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (!f) return -1;
    long long best = -1;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f) != nullptr)
    {
        char* end = nullptr;
        const long long v = std::strtoll(buf, &end, 10);
        if (end != buf && v >= 0) best = std::max(best, v);
    }
#ifdef _WIN32
    _pclose(f);
#else
    pclose(f);
#endif
    return best;
}

[[nodiscard]] double to_mb(std::size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// ── 采样点 ────────────────────────────────────────────────────────────────
struct Sample
{
    std::string label;
    nn::MemoryPool::PoolStats persist{};
    nn::MemoryPool::PoolStats transient{};
    bool pools_valid = false;
    long long smi = -1;
    std::size_t pending_bytes = 0;   // 延迟销毁未归还
    std::vector<VkDeviceSize> p_allocs;   // persist 活跃分配（降序）
    std::vector<VkDeviceSize> t_allocs;   // transient 活跃分配（降序）
};

std::vector<Sample> g_samples;

namespace
{
// 打印单池 top-N 活跃分配（≥ 阈值），用于把池增量落到具体张量
void print_top_allocs(const char* name, const std::vector<VkDeviceSize>& sizes,
                      std::size_t threshold_mb)
{
    std::vector<VkDeviceSize> big;
    for (auto sz : sizes)
        if (sz >= threshold_mb * 1024ull * 1024ull) big.push_back(sz);
    if (big.empty()) return;
    std::cout << std::fixed << std::setprecision(0);
    // 按**精确字节数**聚合（同一尺寸的块数 + 合计），比 top-N 更能指出"谁在占"
    std::vector<std::pair<VkDeviceSize, std::size_t>> agg;
    for (auto sz : big)
    {
        bool found = false;
        for (auto& a : agg)
            if (a.first == sz) { ++a.second; found = true; break; }
        if (!found) agg.push_back({sz, 1});
    }
    std::sort(agg.begin(), agg.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::cout << "      " << name << " 尺寸分布(≥" << threshold_mb << "MB):";
    const std::size_t show = std::min<std::size_t>(agg.size(), 14);
    for (std::size_t i = 0; i < show; ++i)
        std::cout << " " << (agg[i].first / (1024.0 * 1024.0)) << "MB×" << agg[i].second;
    if (agg.size() > show) std::cout << " (+" << (agg.size() - show) << " 种)";
    // 桶统计：≥100MB / 10-100MB / <100MB 的总和
    std::size_t b1 = 0, b2 = 0, b3 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (auto sz : sizes)
    {
        if (sz >= 100ull * 1024 * 1024) { b1 += sz; ++c1; }
        else if (sz >= 10ull * 1024 * 1024) { b2 += sz; ++c2; }
        else { b3 += sz; ++c3; }
    }
    std::cout << "\n      " << name << " 桶: ≥100MB " << c1 << "项/"
              << (b1 / (1024.0 * 1024.0)) << "MB  10-100MB " << c2 << "项/"
              << (b2 / (1024.0 * 1024.0)) << "MB  <10MB " << c3 << "项/"
              << (b3 / (1024.0 * 1024.0)) << "MB\n";
    std::cout << std::setprecision(1);
}
} // namespace

void take_sample(const std::string& label)
{
    Sample s;
    s.label = label;
    auto& backend = nn::GpuBackend::instance();
    if (backend.is_initialized())
    {
        s.persist = backend.memory_pool().pool_debug_stats();
        s.transient = backend.transient_pool().pool_debug_stats();
        s.pending_bytes = static_cast<std::size_t>(backend.pending_destroy_bytes());
        s.p_allocs = backend.memory_pool().live_alloc_sizes();
        s.t_allocs = backend.transient_pool().live_alloc_sizes();
        s.pools_valid = true;
    }
    s.smi = smi_used_mb();

    // 与上一个有池数据的采样点比较（live / committed 增量）
    long long d_live = 0;
    long long d_commit = 0;
    bool has_prev = false;
    for (auto it = g_samples.rbegin(); it != g_samples.rend(); ++it)
    {
        if (!it->pools_valid) break;
        const long long live_now =
            static_cast<long long>(s.persist.allocated_bytes + s.transient.allocated_bytes);
        const long long live_prev =
            static_cast<long long>(it->persist.allocated_bytes + it->transient.allocated_bytes);
        const long long com_now =
            static_cast<long long>(s.persist.total_bytes + s.transient.total_bytes);
        const long long com_prev =
            static_cast<long long>(it->persist.total_bytes + it->transient.total_bytes);
        d_live = live_now - live_prev;
        d_commit = com_now - com_prev;
        has_prev = true;
        break;
    }

    std::cout << "[sample] " << label << "\n";
    if (s.pools_valid)
    {
        const auto pool_line = [](const char* name, const nn::MemoryPool::PoolStats& ps,
                                  long long d_live, long long d_commit, bool show_delta)
        {
            std::cout << "    " << std::left << std::setw(9) << name << std::right
                      << " committed=" << std::fixed << std::setprecision(1)
                      << to_mb(static_cast<std::size_t>(ps.total_bytes))
                      << "MB (dev " << to_mb(static_cast<std::size_t>(ps.device_bytes))
                      << " + host " << to_mb(static_cast<std::size_t>(ps.host_bytes)) << ")"
                      << "  live=" << to_mb(static_cast<std::size_t>(ps.allocated_bytes))
                      << "MB  free=" << to_mb(static_cast<std::size_t>(ps.free_bytes))
                      << "MB  frag=" << std::setprecision(2) << ps.fragmentation
                      << "  blocks=" << ps.block_count;
            if (show_delta)
            {
                std::cout << "  Δlive=" << std::showpos << std::setprecision(1)
                          << (d_live / (1024.0 * 1024.0))
                          << "MB Δcommit=" << (d_commit / (1024.0 * 1024.0))
                          << "MB" << std::noshowpos;
            }
            std::cout << "\n";
            // 池账本计数器（累计值；相邻采样点相减得该阶段的开销）
            std::cout << "              calls=" << ps.c_alloc_calls
                      << "  blk_new/free=" << ps.c_blocks_created << "/"
                      << ps.c_blocks_released
                      << "  scans=" << ps.c_block_scans << "blk/"
                      << ps.c_region_scans << "reg"
                      << "  vkalloc=" << std::setprecision(1) << ps.vkalloc_ms << "ms\n";
        };
        pool_line("persist", s.persist, d_live, d_commit, has_prev);
        pool_line("transient", s.transient, d_live, d_commit, false);
        std::cout << std::resetiosflags(std::ios_base::floatfield);
        if (s.pending_bytes > 0)
            std::cout << std::fixed << std::setprecision(0)
                      << "    延迟销毁未归还(pending) = "
                      << (s.pending_bytes / (1024.0 * 1024.0)) << " MB\n";
        print_top_allocs("persist", s.p_allocs, 16);
        print_top_allocs("transient", s.t_allocs, 16);
    }
    if (s.smi >= 0)
        std::cout << "    nvidia-smi used = " << s.smi << " MiB\n";

    g_samples.push_back(std::move(s));
}

// ── 逻辑字节统计 ──────────────────────────────────────────────────────────
[[nodiscard]] std::size_t tensor_bytes(const nn::Tensor& t)
{
    const std::size_t elem = (t.precision() == nn::Precision::F16) ? 2u : 4u;
    return t.rows() * t.cols() * elem;
}

[[nodiscard]] std::size_t sum_tensor_bytes(std::vector<nn::TensorRef>& refs)
{
    std::size_t total = 0;
    for (auto& r : refs) total += tensor_bytes(r.get());
    return total;
}

// 解析命令行数字
template <typename T>
[[nodiscard]] T parse_num(const char* s, const char* opt)
{
    auto v = nn::parse_number<T>(s);
    if (!v)
    {
        std::cerr << "无效 " << opt << ": " << v.error().message << "\n";
        std::exit(1);
    }
    return *v;
}

} // namespace

void print_usage(const char* prog)
{
    std::cout
        << "运行时显存分项探针\n\n用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --gpu=<设备>            指定 Vulkan 设备（索引或名称子串，如 40HX）\n"
        << "  --f16                   全 f16（param/compute/stable/optimizer 全 F16）\n"
        << "  --checkpoint-every <n>  梯度检查点（每 n 个 block 重算）\n"
        << "  --activation-offload    激活搬 host-visible\n"
        << "  --doc-mask              启用文档感知掩码（复刻 text_train doc_ids 路径）\n"
        << "  --batch/--seq/--vocab/--d-model/--num-heads/--num-layers/--d-ff <n>\n"
        << "  --optimizer <name>      sgd/sgd_momentum/adam/adamw/muon (默认 adam)\n"
        << "  --steps <n>             训练步数采样 (默认 2)\n"
        << "  --no-kv                 跳过 KV cache 推理阶段\n"
        << "  --kv-steps <n>          KV 阶段 forward_step 次数 (默认 32)\n"
        << "  --help                  显示帮助\n";
}

int main(int argc, char* argv[])
{
    // 默认 = bench 配置（docs/benchmarks/2026-09-25，已知峰值 5.7GB 场景）
    std::size_t vocab = 8208;
    std::size_t d_model = 64;
    std::size_t seq = 256;
    std::size_t heads = 4;
    std::size_t layers = 4;
    std::size_t d_ff = 256;
    std::size_t batch = 64;
    std::size_t steps = 2;
    std::size_t kv_steps = 32;
    std::size_t checkpoint_every = 0;
    std::size_t flush_interval = 2;
    bool activation_offload = false;
    bool f16 = false;
    bool f16_all = false;
    bool do_kv = true;
    bool doc_mask = false;
    std::string opt_name = "adam";
    std::string gpu_dev;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc)
            {
                std::cerr << "--" << name << " 缺少值\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--f16") f16 = true;
        else if (arg == "--f16-all") { f16 = true; f16_all = true; }
        else if (arg == "--activation-offload") activation_offload = true;
        else if (arg == "--no-kv") do_kv = false;
        else if (arg == "--doc-mask") doc_mask = true;
        else if (arg == "--batch") batch = parse_num<std::size_t>(need("batch"), "--batch");
        else if (arg == "--seq") seq = parse_num<std::size_t>(need("seq"), "--seq");
        else if (arg == "--vocab") vocab = parse_num<std::size_t>(need("vocab"), "--vocab");
        else if (arg == "--d-model") d_model = parse_num<std::size_t>(need("d-model"), "--d-model");
        else if (arg == "--num-heads") heads = parse_num<std::size_t>(need("num-heads"), "--num-heads");
        else if (arg == "--num-layers") layers = parse_num<std::size_t>(need("num-layers"), "--num-layers");
        else if (arg == "--d-ff") d_ff = parse_num<std::size_t>(need("d-ff"), "--d-ff");
        else if (arg == "--steps") steps = std::max<std::size_t>(1, parse_num<std::size_t>(need("steps"), "--steps"));
        else if (arg == "--kv-steps") kv_steps = parse_num<std::size_t>(need("kv-steps"), "--kv-steps");
        else if (arg == "--checkpoint-every") checkpoint_every = parse_num<std::size_t>(need("checkpoint-every"), "--checkpoint-every");
        else if (arg == "--flush-interval") flush_interval = parse_num<std::size_t>(need("flush-interval"), "--flush-interval");
        else if (arg == "--optimizer") opt_name = need("optimizer");
        else if (auto g = nn::cli::parse_gpu_option(argc, argv, i, /*allow_name_value=*/true))
        {
            gpu_dev = *g;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "========================================\n"
              << "  mem_probe — 运行时显存分项探针\n"
              << "========================================\n"
              << "  配置: vocab=" << vocab << " d_model=" << d_model << " heads=" << heads
              << " layers=" << layers << " d_ff=" << d_ff << " seq=" << seq
              << " batch=" << batch << "\n"
              << "  优化器: " << opt_name << "  精度: " << (f16 ? "全f16" : "f32")
              << "  checkpoint-every=" << checkpoint_every
              << "  offload=" << (activation_offload ? "on" : "off") << "\n"
              << "========================================\n\n";

    // ── 基线（引擎初始化前）：整卡 + 池尚未创建 ──────────────────────────
    take_sample("pre-init");

    // ── 创建 GPU 引擎（探针必须 GPU）────────────────────────────────────
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = true;
    eng_cfg.gpu_device = gpu_dev;
    auto engine_r = nn::cli::create_engine(eng_cfg, std::cout);
    if (!engine_r)
    {
        std::cerr << "引擎创建失败: " << engine_r.error().message << "\n";
        return 1;
    }
    auto raw_engine = std::move(*engine_r);
    if (raw_engine->device() != nn::Device::GPU)
    {
        std::cerr << "mem_probe 需要 GPU 引擎（探针测量对象是 Vulkan 显存）\n";
        return 1;
    }
    take_sample("engine-init");

    // ── 精度配置 ────────────────────────────────────────────────────────
    // f16 存储必须经 PrecisionEngine 适配层（边界 cast）—— 原生引擎只实现
    // f32 存储（见 compute_engine.hpp 的多精度说明）。
    nn::PrecisionProfile prof;
    if (f16)
    {
        prof = nn::profile_f16();   // 与 CLI --f16 同义（param+compute f16）
        if (f16_all)                // --f16-all：四字段全 f16（实验，见 precision.hpp）
        {
            prof.stable = nn::Precision::F16;
            prof.optimizer = nn::Precision::F16;
        }
    }
    std::optional<nn::PrecisionEngine> precision_adapter;
    if (!nn::is_profile_f32(prof))
        precision_adapter.emplace(*raw_engine);
    nn::ComputeEngine* engine = precision_adapter
        ? static_cast<nn::ComputeEngine*>(&*precision_adapter)
        : raw_engine.get();

    // ── 构建模型 ────────────────────────────────────────────────────────
    auto model_r = nn::build_gpt_model(*engine, nn::GptConfig{
        vocab, d_model, seq, heads, d_ff, layers,
        nn::PosEncodingType::Learned, nn::ActivationType::GeLU,
        nn::NormType::LayerNorm, prof});
    if (!model_r)
    {
        std::cerr << "构建模型失败: " << model_r.error().message << "\n";
        return 1;
    }
    nn::Model model = std::move(*model_r);
    model.set_flush_interval(flush_interval);  // 与 text_train 默认一致
    if (checkpoint_every > 0) model.set_checkpoint_every(checkpoint_every);
    if (activation_offload) model.set_activation_offload(true);
    take_sample("model-built");

    // ── 逻辑分项：参数 / 梯度 ───────────────────────────────────────────
    auto params = model.parameters();
    auto grads = model.param_gradients();
    const std::size_t param_bytes = sum_tensor_bytes(params);
    const std::size_t grad_bytes = sum_tensor_bytes(grads);
    std::size_t param_elems = 0;
    for (auto& r : params) param_elems += r.get().rows() * r.get().cols();

    // ── 优化器（状态在构造期分配）──────────────────────────────────────
    auto optimizer = nn::create_optimizer(opt_name, *engine, params, grads,
                                          /*lr=*/1e-3, /*wd=*/0.01);
    if (!optimizer)
    {
        std::cerr << "未知优化器: " << opt_name << "\n";
        return 1;
    }
    take_sample("optimizer-created");

    // 优化器状态实测增量（optimizer-created 与 model-built 的 live 差）
    std::size_t opt_measured_bytes = 0;
    {
        const auto& a = g_samples[g_samples.size() - 2];  // model-built
        const auto& b = g_samples.back();                 // optimizer-created
        if (a.pools_valid && b.pools_valid)
        {
            const long long d = static_cast<long long>(
                (b.persist.allocated_bytes + b.transient.allocated_bytes)
                - (a.persist.allocated_bytes + a.transient.allocated_bytes));
            if (d > 0) opt_measured_bytes = static_cast<std::size_t>(d);
        }
    }
    // 逻辑估算：adam/adamw = m+v（f32 恒定，create_zero_buffers_ 默认 F32）
    std::size_t opt_buffers = 0;
    if (opt_name == "adam" || opt_name == "adamw") opt_buffers = 2;
    else if (opt_name == "sgd_momentum" || opt_name == "muon") opt_buffers = 1;
    const std::size_t opt_logical_bytes = param_elems * 4 * opt_buffers;

    // ── 训练阶段采样（镜像 text_train 顺序）────────────────────────────
    nn::Precision elem_prof = f16 ? nn::Precision::F16 : nn::Precision::F32;
    const std::size_t elem = (elem_prof == nn::Precision::F16) ? 2u : 4u;
    const std::size_t logits_bytes = vocab * seq * batch * elem;   // (vocab, seq*batch)
    std::size_t act_bytes_fwd = 0;
    std::size_t logits_seen = 0;

    nn::Matrix x_tokens(seq, batch);
    nn::Matrix y_tokens(seq, batch);
    nn::Matrix loss_mask(seq, batch);
    {
        std::mt19937_64 rng{12345};
        std::uniform_int_distribution<std::size_t> uid(0, vocab - 1);
        auto xs = x_tokens.span();
        auto ys = y_tokens.span();
        auto ms = loss_mask.span();
        for (std::size_t i = 0; i < seq * batch; ++i)
        {
            xs[i] = static_cast<Scalar>(uid(rng));
            ys[i] = static_cast<Scalar>(uid(rng));
            ms[i] = Scalar{1};
        }
    }
    std::vector<std::size_t> flat_targets(seq * batch);
    std::vector<Scalar> flat_mask(seq * batch, Scalar{1});
    {
        auto ys = y_tokens.span();
        for (std::size_t i = 0; i < seq * batch; ++i)
            flat_targets[i] = static_cast<std::size_t>(ys[i]);
    }

    nn::CrossEntropyLoss ce;
    const long long smi_engine_init = g_samples[1].smi;

    for (std::size_t step = 0; step < steps; ++step)
    {
        const std::string tag = "step" + std::to_string(step) + "/";
        auto xt_r = engine->from_matrix(x_tokens);
        if (!xt_r) { std::cerr << "from_matrix 失败: " << xt_r.error().message << "\n"; return 1; }

        // 文档感知掩码（复刻 text_train 的 doc_ids 路径）：每窗口内模拟多个
        // 文档边界（batch-major b*seq+t 布局）。
        if (doc_mask)
        {
            std::vector<std::size_t> doc_ids(seq * batch, 1);
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t t = 0; t < seq; ++t)
                    doc_ids[b * seq + t] = 1 + ((t / 16 + b) % 64);
            model.set_doc_ids(doc_ids);
        }

        auto br = engine->begin_batch();
        if (!br) { std::cerr << "begin_batch 失败: " << br.error().message << "\n"; return 1; }

        auto fwd = model.forward(*xt_r);
        if (!fwd) { std::cerr << "forward 失败: " << fwd.error().message << "\n"; return 1; }
        nn::Tensor logits = std::move(*fwd);
        logits_seen = tensor_bytes(logits);
        take_sample(tag + "forward");

        // forward 后 transient live ≈ 激活缓存 + logits + 前向临时
        if (g_samples.back().pools_valid)
            act_bytes_fwd = static_cast<std::size_t>(g_samples.back().transient.allocated_bytes);

        std::size_t num_valid = 0;
        auto loss_r = ce.forward_sparse_sum(*engine, logits,
            std::span<const std::size_t>(flat_targets),
            std::span<const Scalar>(flat_mask), vocab, num_valid);
        if (!loss_r) { std::cerr << "loss 失败: " << loss_r.error().message << "\n"; return 1; }
        take_sample(tag + "loss-fwd");

        auto fl = engine->flush_batch();
        if (!fl) { std::cerr << "flush 失败: " << fl.error().message << "\n"; return 1; }

        logits = {};  // text_train 同款：logits 消费完立即释放
        take_sample(tag + "logits-freed");

        auto g = ce.backward();
        if (!g) { std::cerr << "loss backward 失败: " << g.error().message << "\n"; return 1; }
        auto bw = model.backward(*g);
        if (!bw) { std::cerr << "backward 失败: " << bw.error().message << "\n"; return 1; }
        take_sample(tag + "backward");

        auto eb = engine->end_batch();
        if (!eb) { std::cerr << "end_batch 失败: " << eb.error().message << "\n"; return 1; }
        take_sample(tag + "end-batch");

        auto rel = engine->release_idle_pool_blocks();
        if (!rel) { std::cerr << "release 失败: " << rel.error().message << "\n"; return 1; }
        take_sample(tag + "released");
        g = {};

        auto ob = engine->begin_batch();
        if (!ob) { std::cerr << "begin_batch(opt) 失败: " << ob.error().message << "\n"; return 1; }
        auto st = optimizer->step();
        if (!st) { std::cerr << "optimizer step 失败: " << st.error().message << "\n"; return 1; }
        auto zg = optimizer->zero_grad();
        if (!zg) { std::cerr << "zero_grad 失败: " << zg.error().message << "\n"; return 1; }
        auto oe = engine->end_batch();
        if (!oe) { std::cerr << "end_batch(opt) 失败: " << oe.error().message << "\n"; return 1; }
        take_sample(tag + "optimizer-step");
    }

    // ── KV cache 推理阶段 ───────────────────────────────────────────────
    std::size_t kv_logical_bytes = 0;
    if (do_kv)
    {
        auto* gpt = dynamic_cast<nn::GPTModel*>(&model.layer_at(0));
        if (!gpt)
        {
            std::cout << "[KV] layer_at(0) 不是 GPTModel，跳过\n";
        }
        else
        {
            take_sample("kv/pre");
            // generate() 同款预分配：每层一对 (seq, d_model)，在 batch 外 → persist 池
            std::vector<nn::Tensor> kc, vc;
            kc.reserve(layers);
            vc.reserve(layers);
            for (std::size_t i = 0; i < layers; ++i)
            {
                kc.push_back(engine->create_tensor(seq, d_model));
                vc.push_back(engine->create_tensor(seq, d_model));
            }
            kv_logical_bytes = layers * 2 * seq * d_model * 4;  // 创建时恒 f32
            take_sample("kv/caches-allocated");

            auto kbr = engine->begin_batch();
            if (!kbr) { std::cerr << "kv begin_batch 失败: " << kbr.error().message << "\n"; return 1; }
            nn::Tensor last_logits;
            std::mt19937_64 rng{777};
            std::uniform_int_distribution<std::size_t> uid(0, vocab - 1);
            std::size_t cur = 0;
            for (std::size_t t = 0; t < kv_steps && cur < seq; ++t, ++cur)
            {
                auto r = gpt->forward_step(*engine, uid(rng), cur, kc, vc, cur);
                if (!r) { std::cerr << "forward_step 失败: " << r.error().message << "\n"; break; }
                last_logits = std::move(*r);
            }
            take_sample("kv/after-steps(batch内)");
            auto ker = engine->end_batch();
            if (!ker) { std::cerr << "kv end_batch 失败: " << ker.error().message << "\n"; return 1; }
            take_sample("kv/end-batch");
            kc.clear();
            vc.clear();
            last_logits = {};
            take_sample("kv/freed");
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // 汇总
    // ══════════════════════════════════════════════════════════════════
    long long peak_smi = -1;
    std::size_t peak_p_commit = 0, peak_t_commit = 0, peak_p_dev = 0, peak_t_dev = 0;
    std::size_t peak_live = 0;
    for (const auto& s : g_samples)
    {
        if (s.smi >= 0) peak_smi = std::max(peak_smi, s.smi);
        if (!s.pools_valid) continue;
        peak_p_commit = std::max(peak_p_commit, static_cast<std::size_t>(s.persist.total_bytes));
        peak_t_commit = std::max(peak_t_commit, static_cast<std::size_t>(s.transient.total_bytes));
        peak_p_dev = std::max(peak_p_dev, static_cast<std::size_t>(s.persist.device_bytes));
        peak_t_dev = std::max(peak_t_dev, static_cast<std::size_t>(s.transient.device_bytes));
        peak_live = std::max(peak_live, static_cast<std::size_t>(
            s.persist.allocated_bytes + s.transient.allocated_bytes));
    }
    // 稳态（最后一个采样点）
    const Sample& last = g_samples.back();
    const std::size_t slack = (last.pools_valid)
        ? static_cast<std::size_t>(
              (last.persist.total_bytes - last.persist.allocated_bytes)
              + (last.transient.total_bytes - last.transient.allocated_bytes))
        : 0;
    const std::size_t dev_commit = peak_p_dev + peak_t_dev;

    const double MB = 1.0 / (1024.0 * 1024.0);
    std::cout << "\n══════════ 显存分项汇总 ══════════\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  参数 params            " << std::setw(9) << to_mb(param_bytes)
              << " MB   (逻辑，" << params.size() << " 张量, " << param_elems << " 元素)\n";
    std::cout << "  梯度 grads             " << std::setw(9) << to_mb(grad_bytes)
              << " MB   (逻辑，与参数同形)\n";
    std::cout << "  优化器状态 " << std::setw(6) << opt_name << "  " << std::setw(9)
              << to_mb(opt_logical_bytes) << " MB   (逻辑估计"
              << (opt_buffers ? " = " + std::to_string(opt_buffers) + "×N×4B" : "，无状态")
              << "; 实测 Δlive=" << to_mb(opt_measured_bytes) << " MB)\n";
    std::cout << "  激活 activations      " << std::setw(9) << to_mb(act_bytes_fwd)
              << " MB   (实测 transient.live @forward)\n";
    std::cout << "  logits                " << std::setw(9) << to_mb(logits_seen ? logits_seen : logits_bytes)
              << " MB   (vocab×seq×batch×elem" << (f16 ? ", f16" : ", f32") << ")\n";
    std::cout << "  CE 梯度 (同 logits 形) " << std::setw(9) << to_mb(logits_bytes)
              << " MB   (loss-fwd 期与 logits 短暂共存)\n";
    if (do_kv)
        std::cout << "  KV cache              " << std::setw(9) << to_mb(kv_logical_bytes)
                  << " MB   (layers×2×seq×d_model×4B)\n";
    std::cout << "  ── 池/卡口径 ──\n";
    std::cout << "  persist 峰值提交        " << std::setw(9) << to_mb(peak_p_commit)
              << " MB   (其中 device " << to_mb(peak_p_dev) << ")\n";
    std::cout << "  transient 峰值提交      " << std::setw(9) << to_mb(peak_t_commit)
              << " MB   (其中 device " << to_mb(peak_t_dev) << ")\n";
    std::cout << "  池 live 峰值           " << std::setw(9) << to_mb(peak_live) << " MB\n";
    std::cout << "  稳态池空闲(高水位/碎片) " << std::setw(9) << to_mb(slack) << " MB\n";
    if (peak_smi >= 0)
    {
        std::cout << "  nvidia-smi 整卡峰值     " << std::setw(9) << peak_smi
                  << " MB\n";
        std::cout << "  驱动/池外开销          " << std::setw(9)
                  << (peak_smi - static_cast<long long>(dev_commit * MB))
                  << " MB   (= 整卡峰值 − 池 device 提交)\n";
        if (smi_engine_init >= 0)
            std::cout << "  引擎初始化即占整卡      " << std::setw(9)
                      << (smi_engine_init - g_samples[0].smi)
                      << " MB   (staging+readback+驱动上下文, Δ vs pre-init)\n";
    }
    std::cout << "════════════════════════════════\n";

    // 边界 cast 临时量归因（NN_PREC_TRACE=1 时才有数据；f16 路径专用）
    std::cout << nn::PrecisionEngine::dump_temp_stats();

    return 0;
}

#endif // NN_HAS_VULKAN
