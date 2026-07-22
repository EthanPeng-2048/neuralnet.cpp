// ── MNIST MLP CPU vs GPU 对比基准测试 ──────────────────────────────────────
// 用法：mnist_bench [--epochs N] [--batch-size N] [--dataset <path>]
//                   [--warmup N] [--infer-iters N] [--help]
//
// 在相同数据集、相同模型上分别用 CPU 和 GPU 跑训练和推理，
// 输出逐 epoch 时间、总时间、准确率及推理吞吐量对比。
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
        << "MNIST MLP CPU vs GPU 对比基准测试\n\n"
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
//  数据加载（复用 mnist_train.cpp 优化版）
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

    // 预扫描列数
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
//  评估准确率
// ═══════════════════════════════════════════════════════════════════════════
Scalar evaluate(nn::Model& model, const nn::Matrix& x, const nn::Matrix& y_onehot)
{
    auto out_result = model.forward(x);
    if (!out_result) return -1.0;
    auto out = std::move(*out_result);
    std::size_t N = x.cols();
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
//  单轮训练（返回 epoch 耗时毫秒 + 平均 loss）
// ═══════════════════════════════════════════════════════════════════════════
struct EpochResult {
    double time_ms;
    Scalar avg_loss;
    Scalar train_acc;
    Scalar test_acc;
};

EpochResult train_one_epoch(
    nn::Model& model, nn::Optimizer& optimizer, nn::CrossEntropyLoss& ce_loss,
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

        auto out_fwd = model.forward(x_batch);
        if (!out_fwd) { std::cerr << "Forward failed\n"; std::exit(1); }
        auto out = std::move(*out_fwd);
        auto loss_result = ce_loss.forward(out, y_batch);
        if (!loss_result) { std::cerr << "Loss failed\n"; std::exit(1); }
        total_loss += *loss_result;

        auto grad_result = ce_loss.backward();
        if (!grad_result) { std::cerr << "Loss bw failed\n"; std::exit(1); }
        auto bwd_result = model.backward(*grad_result);
        if (!bwd_result) { std::cerr << "Backward failed\n"; std::exit(1); }
        (void)optimizer.step();
        (void)optimizer.zero_grad();
    }
    auto t1 = Clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    Scalar avg_loss = total_loss / static_cast<Scalar>(num_batches);
    Scalar train_acc = evaluate(model, train_x, train_y);
    Scalar test_acc  = evaluate(model, test_x, test_y);

    return {ms, avg_loss, train_acc, test_acc};
}

// ═══════════════════════════════════════════════════════════════════════════
//  推理基准测试（对单张图片反复 forward，测吞吐量）
// ═══════════════════════════════════════════════════════════════════════════
struct InferResult {
    double avg_ms;
    double throughput;  // images/sec
};

InferResult bench_inference(
    nn::Model& model, const nn::Matrix& single_image,
    int warmup, int iters)
{
    // 预热
    for (int i = 0; i < warmup; ++i)
        (void)model.forward(single_image);

    // 计时
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        (void)model.forward(single_image);
    auto t1 = Clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;
    double throughput = (avg_ms > 0) ? 1000.0 / avg_ms : 0;
    return {avg_ms, throughput};
}

// ═══════════════════════════════════════════════════════════════════════════
//  分隔线
// ═══════════════════════════════════════════════════════════════════════════
void print_separator()
{
    std::cout << "══════════════════════════════════════════════════════════════\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  主函数
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[])
{
    BenchConfig cfg = parse_args(argc, argv);

    print_separator();
    std::cout << "  MNIST MLP  CPU vs GPU 对比基准测试\n";
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
    std::cout << "  特征维度: " << train_x.rows() << "\n\n";

    // 取一张测试图片用于推理基准
    nn::Matrix single_img(train_x.rows(), 1);
    for (std::size_t r = 0; r < train_x.rows(); ++r)
        single_img.set_value_unchecked(r, 0, test_x.at_unchecked(r, 0));

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 1: CPU 训练
    // ════════════════════════════════════════════════════════════════════════
    print_separator();
    std::cout << "  Phase 1: CPU 训练\n";
    print_separator();

    nn::SmartPolicy::gpu_enabled = false;

    auto model_cpu_result = nn::build_mnist_mlp_model();
    if (!model_cpu_result) { std::cerr << "构建模型失败\n"; return 1; }
    auto model_cpu = std::move(*model_cpu_result);

    auto optimizer_cpu = nn::create_optimizer(
        "adam", model_cpu.parameters(), model_cpu.param_gradients(), cfg.lr);
    nn::CrossEntropyLoss ce_loss_cpu;

    std::mt19937_64 rng_cpu{42};
    TimingStats cpu_train_stats;

    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(12) << "Loss"
              << std::setw(12) << "Train Acc"
              << std::setw(12) << "Test Acc"
              << std::setw(12) << "Time(ms)"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        auto ep = train_one_epoch(model_cpu, *optimizer_cpu, ce_loss_cpu,
                                  train_x, train_y, test_x, test_y,
                                  cfg.batch_size, rng_cpu);
        cpu_train_stats.total_ms += ep.time_ms;
        cpu_train_stats.min_ms = std::min(cpu_train_stats.min_ms, ep.time_ms);
        cpu_train_stats.max_ms = std::max(cpu_train_stats.max_ms, ep.time_ms);
        cpu_train_stats.per_epoch.push_back(ep.time_ms);

        std::cout << std::left
                  << std::setw(8)  << (epoch + 1)
                  << std::setw(12) << std::fixed << std::setprecision(4) << ep.avg_loss
                  << std::setw(12) << std::setprecision(2) << (ep.train_acc * 100.0) << "%"
                  << std::setw(12) << (ep.test_acc * 100.0) << "%"
                  << std::setw(12) << std::setprecision(1) << ep.time_ms
                  << "\n";
    }

    // CPU 推理基准
    std::cout << "\n  CPU 推理基准 (" << cfg.infer_iters << " 次迭代, 预热 " << cfg.warmup << " 轮)...\n";
    auto cpu_infer = bench_inference(model_cpu, single_img, cfg.warmup, cfg.infer_iters);
    std::cout << "  平均延迟: " << std::fixed << std::setprecision(3) << cpu_infer.avg_ms << " ms\n";
    std::cout << "  吞吐量:   " << std::setprecision(1) << cpu_infer.throughput << " images/s\n\n";

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 2: GPU 训练
    // ════════════════════════════════════════════════════════════════════════
#ifdef NN_HAS_VULKAN
    print_separator();
    std::cout << "  Phase 2: GPU 训练 (Vulkan)\n";
    print_separator();

    auto& backend = nn::GpuBackend::instance();
    auto init_result = backend.initialize();
    if (!init_result)
    {
        std::cerr << "\n  GPU 初始化失败: " << init_result.error().message << "\n";
        std::cerr << "  跳过 GPU 测试，仅报告 CPU 结果。\n\n";
        goto report;
    }
    nn::SmartPolicy::gpu_enabled = true;
    std::cout << "  GPU: Vulkan 加速已启用\n\n";

    {
        auto model_gpu_result = nn::build_mnist_mlp_model();
        if (!model_gpu_result) { std::cerr << "构建模型失败\n"; return 1; }
        auto model_gpu = std::move(*model_gpu_result);

        auto optimizer_gpu = nn::create_optimizer(
            "adam", model_gpu.parameters(), model_gpu.param_gradients(), cfg.lr);
        nn::CrossEntropyLoss ce_loss_gpu;

        std::mt19937_64 rng_gpu{42};
        TimingStats gpu_train_stats;

        std::cout << std::left
                  << std::setw(8)  << "Epoch"
                  << std::setw(12) << "Loss"
                  << std::setw(12) << "Train Acc"
                  << std::setw(12) << "Test Acc"
                  << std::setw(12) << "Time(ms)"
                  << "\n";
        std::cout << std::string(56, '-') << "\n";

        for (int epoch = 0; epoch < cfg.epochs; ++epoch)
        {
            auto ep = train_one_epoch(model_gpu, *optimizer_gpu, ce_loss_gpu,
                                      train_x, train_y, test_x, test_y,
                                      cfg.batch_size, rng_gpu);
            gpu_train_stats.total_ms += ep.time_ms;
            gpu_train_stats.min_ms = std::min(gpu_train_stats.min_ms, ep.time_ms);
            gpu_train_stats.max_ms = std::max(gpu_train_stats.max_ms, ep.time_ms);
            gpu_train_stats.per_epoch.push_back(ep.time_ms);

            std::cout << std::left
                      << std::setw(8)  << (epoch + 1)
                      << std::setw(12) << std::fixed << std::setprecision(4) << ep.avg_loss
                      << std::setw(12) << std::setprecision(2) << (ep.train_acc * 100.0) << "%"
                      << std::setw(12) << (ep.test_acc * 100.0) << "%"
                      << std::setw(12) << std::setprecision(1) << ep.time_ms
                      << "\n";
        }

        // GPU 推理基准
        nn::SmartPolicy::gpu_enabled = true;
        std::cout << "\n  GPU 推理基准 (" << cfg.infer_iters << " 次迭代, 预热 " << cfg.warmup << " 轮)...\n";
        auto gpu_infer = bench_inference(model_gpu, single_img, cfg.warmup, cfg.infer_iters);
        std::cout << "  平均延迟: " << std::fixed << std::setprecision(3) << gpu_infer.avg_ms << " ms\n";
        std::cout << "  吞吐量:   " << std::setprecision(1) << gpu_infer.throughput << " images/s\n\n";

        // ── 汇总对比 ──────────────────────────────────────────────
        nn::SmartPolicy::gpu_enabled = false;

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
                  << "\n";

        // GPU 统计
        uint64_t gpu_ops = nn::SmartPolicy::gpu_matmul_count.load();
        uint64_t cpu_ops = nn::SmartPolicy::cpu_matmul_count.load();
        std::cout << "\n  GPU 矩阵乘法次数: " << gpu_ops << "\n";
        std::cout << "  CPU 矩阵乘法次数: " << cpu_ops << "\n";
        std::cout << "  GPU 利用率: " << std::fixed << std::setprecision(1)
                  << (gpu_ops + cpu_ops > 0 ? 100.0 * gpu_ops / (gpu_ops + cpu_ops) : 0.0) << "%\n\n";

        print_separator();
        std::cout << "  完成\n";
        print_separator();
    }
    return 0;

report:
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
