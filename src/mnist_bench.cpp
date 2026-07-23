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

BenchConfig parse_args(int argc, char* argv[])
{
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "--epochs" && i + 1 < argc)
            cfg.epochs = std::stoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc)
            cfg.batch_size = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--dataset" && i + 1 < argc)
            cfg.dataset_path = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc)
            cfg.warmup = std::stoi(argv[++i]);
        else if (arg == "--infer-iters" && i + 1 < argc)
            cfg.infer_iters = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc)
            cfg.lr = static_cast<Scalar>(std::stod(argv[++i]));
        else { std::cerr << "未知参数: " << arg << "\n"; std::exit(1); }
    }
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
//  数据加载
// ═══════════════════════════════════════════════════════════════════════════
nn::Result<std::pair<nn::Matrix, nn::Matrix>> load_csv(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return std::unexpected(nn::Error{"Cannot open: " + filename});

    const auto file_size = file.tellg();
    file.seekg(0);
    std::string buffer(static_cast<std::size_t>(file_size), '\0');
    file.read(buffer.data(), file_size);
    file.close();

    std::size_t row_count = 0;
    for (char c : buffer)
        if (c == '\n') ++row_count;
    if (row_count == 0)
        return std::unexpected(nn::Error{"CSV empty: " + filename});

    const char* ptr = buffer.data();
    const char* end = buffer.data() + buffer.size();

    int first_label = 0;
    std::size_t feat_dim = 0;
    {
        const char* p = ptr;
        auto [p1, ec1] = std::from_chars(p, end, first_label);
        p = p1;
        std::size_t cnt = 0;
        while (p < end && *p != '\n' && *p != '\r')
        {
            if (*p == ',') { ++cnt; Scalar tmp; auto [p2, ec2] = std::from_chars(p + 1, end, tmp); p = p2; }
            else ++p;
        }
        feat_dim = cnt;
    }

    std::vector<Scalar> features(row_count * feat_dim);
    std::vector<int>    labels(row_count);
    std::size_t row = 0;
    ptr = buffer.data();

    while (ptr < end && row < row_count)
    {
        auto [p_label, ec_label] = std::from_chars(ptr, end, labels[row]);
        if (ec_label != std::errc{}) break;
        ptr = p_label;
        for (std::size_t j = 0; j < feat_dim; ++j)
        {
            if (ptr < end && *ptr == ',') ++ptr;
            Scalar val;
            auto [p_feat, ec_feat] = std::from_chars(ptr, end, val);
            if (ec_feat != std::errc{}) break;
            features[row * feat_dim + j] = val;
            ptr = p_feat;
        }
        while (ptr < end && (*ptr == '\r' || *ptr == '\n')) ++ptr;
        ++row;
    }

    std::size_t N = row;
    nn::Matrix feat_mat(feat_dim, N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < feat_dim; ++j)
            feat_mat.set_value_unchecked(j, i, features[i * feat_dim + j]);

    nn::Matrix label_mat(10, N);
    for (std::size_t i = 0; i < N; ++i)
        if (labels[i] >= 0 && labels[i] < 10)
            label_mat.set_value_unchecked(labels[i], i, 1.0);

    return std::pair{std::move(feat_mat), std::move(label_mat)};
}

// ═══════════════════════════════════════════════════════════════════════════
//  评估准确率（全量前向后下载到 CPU 做 argmax）
// ═══════════════════════════════════════════════════════════════════════════
nn::Result<Scalar> evaluate(nn::Model& model, nn::ComputeEngine& engine,
                             const nn::Matrix& x, const nn::Matrix& y_onehot)
{
    std::size_t N = x.cols();

    auto x_tensor_r = engine.from_matrix(x);
    if (!x_tensor_r) return std::unexpected(std::move(x_tensor_r).error());

    auto out_tensor_r = model.forward(*x_tensor_r);
    if (!out_tensor_r) return std::unexpected(std::move(out_tensor_r).error());

    auto out_r = engine.to_matrix(*out_tensor_r);
    if (!out_r) return std::unexpected(std::move(out_r).error());
    const auto& out = *out_r;

    int correct = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        Scalar max_val = out.at_unchecked(0, i);
        int pred = 0;
        for (int j = 1; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
        {
            Scalar val = out.at_unchecked(j, i);
            if (val > max_val) { max_val = val; pred = j; }
        }
        int true_label = -1;
        for (int j = 0; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
            if (y_onehot.at_unchecked(j, i) == 1.0) { true_label = j; break; }
        if (pred == true_label) ++correct;
    }
    return static_cast<Scalar>(correct) / static_cast<Scalar>(N);
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
    auto train_acc_r = evaluate(model, engine, train_x, train_y);
    auto test_acc_r  = evaluate(model, engine, test_x, test_y);
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
    auto train_result = load_csv(cfg.dataset_path + "/train.csv");
    if (!train_result) { std::cerr << "加载训练集失败: " << train_result.error().message << "\n"; return 1; }
    auto [train_x, train_y] = std::move(*train_result);

    auto test_result = load_csv(cfg.dataset_path + "/test.csv");
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
