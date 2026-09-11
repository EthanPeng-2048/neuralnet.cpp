// ── MNIST MLP CPU vs GPU 对比基准测试（新引擎化架构） ──────────────────────
//
// 在相同数据集、相同模型上分别用 CpuEngine 和 GpuEngine 跑训练和推理，
// 输出逐 epoch 时间、总时间、准确率及推理吞吐量对比。
//
// 新架构下每个 Model 绑定一个 ComputeEngine；CPU/GPU 对比需要分别构建
// 独立的 engine + model 实例。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/cli/mnist_io.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using nn::Scalar;
namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
//  计时工具
// ═══════════════════════════════════════════════════════════════════════════
using Clock = std::chrono::high_resolution_clock;

struct TimingStats {
    double total_ms = 0;
    double min_ms   = 1e18;
    double max_ms   = 0;
    std::vector<double> per_epoch;
};

// ═══════════════════════════════════════════════════════════════════════════
//  命令行参数
// ═══════════════════════════════════════════════════════════════════════════
struct BenchConfig {
    std::string dataset_path = "datasets/mnist_data";
    int epochs       = 5;
    std::size_t batch_size = 64;
    int warmup       = 2;     // 推理预热轮数
    int infer_iters  = 100;   // 推理计时迭代次数
    Scalar lr        = 0.001;
};

void print_usage(const char* prog)
{
    std::cout
        << "MNIST MLP CPU vs GPU 对比基准测试 (引擎化架构)\n\n"
        << "用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --epochs <n>        训练轮数 (默认: 5)\n"
        << "  --batch-size <n>    批大小 (默认: 64)\n"
        << "  --dataset <path>    数据集目录 (默认: datasets/mnist_data)\n"
        << "  --warmup <n>        推理预热轮数 (默认: 2)\n"
        << "  --infer-iters <n>   推理计时迭代次数 (默认: 100)\n"
        << "  --lr <lr>           学习率 (默认: 0.001)\n"
        << "  --help              显示此帮助信息\n";
}

// 数字解析辅助：解析失败时打印错误并退出（替代会抛异常的 std::stoi/stod）
template <typename T>
T parse_num_or_die(const char* s, const char* opt)
{
    auto v = nn::parse_number<T>(s);
    if (!v)
    {
        std::cerr << "无效的 " << opt << " 值: " << v.error().message << "\n";
        std::exit(1);
    }
    return *v;
}

BenchConfig parse_args(int argc, char* argv[])
{
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "--epochs" && i + 1 < argc)
        {
            cfg.epochs = parse_num_or_die<int>(argv[++i], "--epochs");
            if (cfg.epochs <= 0) { std::cerr << "--epochs 必须为正整数\n"; std::exit(1); }
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            int bs = parse_num_or_die<int>(argv[++i], "--batch-size");
            if (bs <= 0) { std::cerr << "--batch-size 必须为正整数\n"; std::exit(1); }
            cfg.batch_size = static_cast<std::size_t>(bs);
        }
        else if (arg == "--dataset" && i + 1 < argc)
            cfg.dataset_path = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc)
        {
            cfg.warmup = parse_num_or_die<int>(argv[++i], "--warmup");
            if (cfg.warmup < 0) { std::cerr << "--warmup 必须为非负整数\n"; std::exit(1); }
        }
        else if (arg == "--infer-iters" && i + 1 < argc)
        {
            cfg.infer_iters = parse_num_or_die<int>(argv[++i], "--infer-iters");
            if (cfg.infer_iters <= 0) { std::cerr << "--infer-iters 必须为正整数\n"; std::exit(1); }
        }
        else if (arg == "--lr" && i + 1 < argc)
            cfg.lr = parse_num_or_die<Scalar>(argv[++i], "--lr");
        else { std::cerr << "未知参数: " << arg << "\n"; std::exit(1); }
    }
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
//  单轮训练（返回 epoch 耗时毫秒 + 平均 loss + 训练/测试准确率）
// ═══════════════════════════════════════════════════════════════════════════
struct EpochResult {
    double time_ms;
    Scalar avg_loss;
    Scalar train_acc;
    Scalar test_acc;
};

nn::Result<EpochResult> train_one_epoch(
    nn::Model& model, nn::ComputeEngine& engine,
    nn::Optimizer& optimizer, nn::CrossEntropyLoss& ce_loss,
    const nn::Matrix& train_x, const nn::Matrix& train_y,
    const nn::Matrix& test_x, const nn::Matrix& test_y,
    std::size_t batch_size, std::mt19937_64& rng)
{
    const std::size_t N = train_x.cols();
    const std::size_t num_batches = N / batch_size;
    if (num_batches == 0)
        return std::unexpected(nn::Error{
            "train_one_epoch: 样本数小于 batch_size，无法构成训练批次"});

    nn::Matrix x_batch(train_x.rows(), batch_size);
    nn::Matrix y_batch(train_y.rows(), batch_size);

    std::vector<std::size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    Scalar total_loss = 0.0;

    auto t0 = Clock::now();
    for (std::size_t b = 0; b < num_batches; ++b)
    {
        const std::size_t start = b * batch_size;
        const std::size_t R_x = train_x.rows();
        const std::size_t R_y = train_y.rows();
        const auto x_src = train_x.span().data();
        const auto y_src = train_y.span().data();
        auto x_dst = x_batch.span().data();
        auto y_dst = y_batch.span().data();
        for (std::size_t k = 0; k < batch_size; ++k)
        {
            const std::size_t src_col = indices[start + k];
            for (std::size_t r = 0; r < R_x; ++r)
                x_dst[r * batch_size + k] = x_src[r * N + src_col];
            for (std::size_t r = 0; r < R_y; ++r)
                y_dst[r * batch_size + k] = y_src[r * N + src_col];
        }

        // Matrix → Tensor（上传到引擎设备）
        auto x_tensor_r = engine.from_matrix(x_batch);
        if (!x_tensor_r) return std::unexpected(std::move(x_tensor_r).error());
        auto y_tensor_r = engine.from_matrix(y_batch);
        if (!y_tensor_r) return std::unexpected(std::move(y_tensor_r).error());

        // 前向 + 损失
        auto out_r = model.forward(*x_tensor_r);
        if (!out_r) return std::unexpected(std::move(out_r).error());
        auto loss_r = ce_loss.forward(engine, *out_r, *y_tensor_r);
        if (!loss_r) return std::unexpected(std::move(loss_r).error());
        total_loss += *loss_r;

        // 反向 + 优化
        auto grad_r = ce_loss.backward();
        if (!grad_r) return std::unexpected(std::move(grad_r).error());
        auto bwd_r = model.backward(*grad_r);
        if (!bwd_r) return std::unexpected(std::move(bwd_r).error());
        (void)optimizer.step();
        (void)optimizer.zero_grad();
    }
    auto t1 = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    Scalar avg_loss = total_loss / static_cast<Scalar>(num_batches);
    auto train_acc_r = nn::cli::evaluate_mnist(model, engine, train_x, train_y);
    auto test_acc_r  = nn::cli::evaluate_mnist(model, engine, test_x, test_y);
    if (!train_acc_r || !test_acc_r)
        return std::unexpected(!train_acc_r ? train_acc_r.error() : test_acc_r.error());

    return EpochResult{ms, avg_loss, *train_acc_r, *test_acc_r};
}

// ═══════════════════════════════════════════════════════════════════════════
//  推理基准测试（对单张图片反复 forward，测吞吐量）
// ═══════════════════════════════════════════════════════════════════════════
struct InferResult {
    double avg_ms;
    double throughput;  // images/sec
};

nn::Result<InferResult> bench_inference(
    nn::Model& model, nn::ComputeEngine& engine,
    const nn::Matrix& single_image, int warmup, int iters)
{
    if (iters <= 0)
        return std::unexpected(nn::Error{"bench_inference: iters 必须为正整数"});
    // 预热
    for (int i = 0; i < warmup; ++i)
    {
        auto x_r = engine.from_matrix(single_image);
        if (!x_r) return std::unexpected(std::move(x_r).error());
        auto out_r = model.forward(*x_r);
        if (!out_r) return std::unexpected(std::move(out_r).error());
    }

    // 计时
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto x_r = engine.from_matrix(single_image);
        if (!x_r) return std::unexpected(std::move(x_r).error());
        auto out_r = model.forward(*x_r);
        if (!out_r) return std::unexpected(std::move(out_r).error());
    }
    auto t1 = Clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;
    double throughput = (avg_ms > 0) ? 1000.0 / avg_ms : 0;
    return InferResult{avg_ms, throughput};
}

// ═══════════════════════════════════════════════════════════════════════════
//  分隔线 + 表头
// ═══════════════════════════════════════════════════════════════════════════
void print_separator()
{
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

void print_epoch_header(const char* title)
{
    std::cout << "\n  " << title << "\n\n";
    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(12) << "Loss"
              << std::setw(12) << "Train Acc"
              << std::setw(12) << "Test Acc"
              << std::setw(12) << "Time(ms)"
              << "\n";
    std::cout << std::string(56, '-') << "\n";
}

void print_epoch_row(const EpochResult& ep, int epoch)
{
    std::cout << std::left
              << std::setw(8)  << (epoch + 1)
              << std::setw(12) << std::fixed << std::setprecision(4) << ep.avg_loss
              << std::setw(12) << std::setprecision(2) << (ep.train_acc * 100.0) << "%"
              << std::setw(12) << (ep.test_acc * 100.0) << "%"
              << std::setw(12) << std::setprecision(1) << ep.time_ms
              << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  单设备训练 + 推理基准（返回 TimingStats + InferResult）
// ═══════════════════════════════════════════════════════════════════════════
struct DeviceBenchResult {
    TimingStats train_stats;
    InferResult infer;
};

nn::Result<DeviceBenchResult> bench_device(
    nn::ComputeEngine& engine,
    const BenchConfig& cfg,
    const nn::Matrix& train_x, const nn::Matrix& train_y,
    const nn::Matrix& test_x,  const nn::Matrix& test_y,
    const nn::Matrix& single_img,
    std::uint64_t rng_seed, const char* device_name)
{
    auto model_r = nn::build_mnist_mlp_model(engine);
    if (!model_r) return std::unexpected(std::move(model_r).error());
    auto model = std::move(*model_r);

    auto optimizer = nn::create_optimizer(
        "adam", engine, model.parameters(), model.param_gradients(), cfg.lr);
    if (!optimizer)
        return std::unexpected(nn::Error{"bench_device: failed to create optimizer"});
    nn::CrossEntropyLoss ce_loss;

    std::mt19937_64 rng{rng_seed};
    TimingStats stats;

    print_epoch_header(device_name);

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        auto ep_r = train_one_epoch(model, engine, *optimizer, ce_loss,
                                    train_x, train_y, test_x, test_y,
                                    cfg.batch_size, rng);
        if (!ep_r) return std::unexpected(std::move(ep_r).error());
        auto ep = *ep_r;

        stats.total_ms += ep.time_ms;
        stats.min_ms = std::min(stats.min_ms, ep.time_ms);
        stats.max_ms = std::max(stats.max_ms, ep.time_ms);
        stats.per_epoch.push_back(ep.time_ms);

        print_epoch_row(ep, epoch);
    }

    // 推理基准
    std::cout << "\n  " << device_name << " 推理基准 ("
              << cfg.infer_iters << " 次迭代, 预热 " << cfg.warmup << " 轮)...\n";
    auto infer_r = bench_inference(model, engine, single_img, cfg.warmup, cfg.infer_iters);
    if (!infer_r) return std::unexpected(std::move(infer_r).error());
    std::cout << "  平均延迟: " << std::fixed << std::setprecision(3) << infer_r->avg_ms << " ms\n";
    std::cout << "  吞吐量:   " << std::setprecision(1) << infer_r->throughput << " images/s\n";

    return DeviceBenchResult{stats, *infer_r};
}

// ═══════════════════════════════════════════════════════════════════════════
//  主函数
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    BenchConfig cfg = parse_args(argc, argv);

    print_separator();
    std::cout << "  MNIST MLP  CPU vs GPU 对比基准测试 (引擎化架构)\n";
    print_separator();
    std::cout << "  模型:     784→512→256→128→64→10 (LayerNorm+GeLU)\n";
    std::cout << "  训练轮数: " << cfg.epochs << "\n";
    std::cout << "  批大小:   " << cfg.batch_size << "\n";
    std::cout << "  学习率:   " << cfg.lr << "\n";
    std::cout << "  推理预热: " << cfg.warmup << " 轮\n";
    std::cout << "  推理迭代: " << cfg.infer_iters << " 次\n";
    print_separator();

    // ── 加载数据 ─────────────────────────────────────────────
    std::cout << "\n加载数据集...\n";
    auto train_result = nn::cli::load_mnist_csv(cfg.dataset_path + "/train.csv");
    if (!train_result) { std::cerr << "加载训练集失败: " << train_result.error().message << "\n"; return 1; }
    auto [train_x, train_y] = std::move(*train_result);

    auto test_result = nn::cli::load_mnist_csv(cfg.dataset_path + "/test.csv");
    if (!test_result) { std::cerr << "加载测试集失败: " << test_result.error().message << "\n"; return 1; }
    auto [test_x, test_y] = std::move(*test_result);

    std::cout << "  训练集: " << train_x.cols() << " 样本\n";
    std::cout << "  测试集: " << test_x.cols() << " 样本\n";
    std::cout << "  特征维度: " << train_x.rows() << "\n";

    // 取一张测试图片用于推理基准
    nn::Matrix single_img(train_x.rows(), 1);
    for (std::size_t r = 0; r < train_x.rows(); ++r)
        single_img.set_value_unchecked(r, 0, test_x.at_unchecked(r, 0));

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 1: CPU 训练 + 推理基准
    // ════════════════════════════════════════════════════════════════════════
    print_separator();
    std::cout << "  Phase 1: CPU 训练 (CpuEngine)\n";
    print_separator();

    auto cpu_engine = std::make_unique<nn::CpuEngine>();
    auto cpu_bench_r = bench_device(*cpu_engine, cfg,
                                    train_x, train_y, test_x, test_y,
                                    single_img, 42, "CPU");
    if (!cpu_bench_r)
    {
        std::cerr << "CPU 基准测试失败: " << cpu_bench_r.error().message << "\n";
        return 1;
    }
    auto cpu_bench = *cpu_bench_r;
    const auto& cpu_train_stats = cpu_bench.train_stats;
    const auto& cpu_infer = cpu_bench.infer;

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 2: GPU 训练 + 推理基准（需要 Vulkan）
    // ════════════════════════════════════════════════════════════════════════
#ifdef NN_HAS_VULKAN
    print_separator();
    std::cout << "  Phase 2: GPU 训练 (GpuEngine / Vulkan)\n";
    print_separator();

    auto& backend = nn::GpuBackend::instance();
    auto init_result = backend.initialize();
    if (!init_result)
    {
        std::cerr << "\n  GPU 初始化失败: " << init_result.error().message << "\n";
        std::cerr << "  跳过 GPU 测试，仅报告 CPU 结果。\n\n";
    }
    else
    {
        auto gpu_engine = std::make_unique<nn::GpuEngine>(backend);
        auto gpu_bench_r = bench_device(*gpu_engine, cfg,
                                        train_x, train_y, test_x, test_y,
                                        single_img, 42, "GPU");
        if (!gpu_bench_r)
        {
            std::cerr << "GPU 基准测试失败: " << gpu_bench_r.error().message << "\n";
        }
        else
        {
            auto gpu_bench = *gpu_bench_r;
            const auto& gpu_train_stats = gpu_bench.train_stats;
            const auto& gpu_infer = gpu_bench.infer;

            // ── 汇总对比 ──────────────────────────────────────────────
            print_separator();
            std::cout << "  汇总对比\n";
            print_separator();

            double avg_cpu_epoch = cpu_train_stats.total_ms / cfg.epochs;
            double avg_gpu_epoch = gpu_train_stats.total_ms / cfg.epochs;
            double train_speedup = avg_cpu_epoch / avg_gpu_epoch;
            double infer_speedup = cpu_infer.avg_ms / gpu_infer.avg_ms;

            std::cout << std::left
                      << std::setw(24) << "指标"
                      << std::setw(16) << "CPU"
                      << std::setw(16) << "GPU"
                      << std::setw(12) << "加速比"
                      << "\n";
            std::cout << std::string(68, '-') << "\n";

            std::cout << std::left
                      << std::setw(24) << "训练总时间 (ms)"
                      << std::setw(16) << std::fixed << std::setprecision(1) << cpu_train_stats.total_ms
                      << std::setw(16) << gpu_train_stats.total_ms
                      << std::setw(12) << std::setprecision(2) << (cpu_train_stats.total_ms / gpu_train_stats.total_ms) << "x"
                      << "\n";

            std::cout << std::left
                      << std::setw(24) << "平均 Epoch 时间 (ms)"
                      << std::setw(16) << std::fixed << std::setprecision(1) << avg_cpu_epoch
                      << std::setw(16) << avg_gpu_epoch
                      << std::setw(12) << std::setprecision(2) << train_speedup << "x"
                      << "\n";

            std::cout << std::left
                      << std::setw(24) << "推理延迟 (ms/img)"
                      << std::setw(16) << std::fixed << std::setprecision(3) << cpu_infer.avg_ms
                      << std::setw(16) << gpu_infer.avg_ms
                      << std::setw(12) << std::setprecision(2) << infer_speedup << "x"
                      << "\n";

            std::cout << std::left
                      << std::setw(24) << "推理吞吐量 (img/s)"
                      << std::setw(16) << std::fixed << std::setprecision(1) << cpu_infer.throughput
                      << std::setw(16) << gpu_infer.throughput
                      << std::setw(12) << std::setprecision(2) << infer_speedup << "x"
                      << "\n\n";

            print_separator();
            std::cout << "  完成\n";
            print_separator();
            return 0;
        }
    }
#endif

    // 无 Vulkan 时的 CPU-only 汇总
    print_separator();
    std::cout << "  CPU Only 汇总\n";
    print_separator();
    double avg_cpu_epoch = cpu_train_stats.total_ms / cfg.epochs;
    std::cout << "  训练总时间:   " << std::fixed << std::setprecision(1) << cpu_train_stats.total_ms << " ms\n";
    std::cout << "  平均 Epoch:   " << avg_cpu_epoch << " ms\n";
    std::cout << "  推理延迟:     " << std::setprecision(3) << cpu_infer.avg_ms << " ms/img\n";
    std::cout << "  推理吞吐量:   " << std::setprecision(1) << cpu_infer.throughput << " images/s\n\n";
    std::cout << "  提示: 编译时启用 Vulkan SDK 以启用 GPU 对比测试。\n";
    print_separator();

    return 0;
}
