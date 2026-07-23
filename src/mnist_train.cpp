// ── MNIST 手写数字训练程序（新引擎化架构） ─────────────────────────────────
//
// 数据流：
//   CSV → Matrix(feat_dim, N) + Matrix(10, N)
//   每 batch：Matrix → engine.from_matrix → Tensor[device]
//     → model.forward(Tensor) → Tensor
//     → ce_loss.forward(engine, out, target) → Scalar
//     → ce_loss.backward() → Tensor
//     → model.backward(Tensor) → (丢弃)
//     → optimizer.step() / model.zero_grad()
//   evaluate：forward → engine.to_matrix → CPU argmax
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字训练程序 (引擎化架构)\n\n"
        << "用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --resume <path>    从已有模型恢复训练 (自动读取模型规格)\n"
        << "  --save <path>      模型保存路径 (默认: mnist_model.bin)\n"
        << "  --dataset <path>   数据集目录 (默认: datasets/mnist_data)\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 64)\n"
        << "  --optimizer <name> 优化器: sgd/sgd_momentum/adam (默认: adam)\n"
        << "  --gpu              启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --layer-dims <d1,d2,...>  MLP 各层维度，逗号分隔 (默认: 784,512,256,128,64,10)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string save_path = "mnist_model.bin";
    std::string dataset_path = "datasets/mnist_data";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    Scalar lr = 0.001;
    std::size_t batch_size = 64;
    bool load_existing = false;
    std::vector<std::size_t> layer_dims;
    bool gpu_enabled = false;
};

TrainConfig parse_args(int argc, char *argv[])
{
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--resume" && i + 1 < argc)
        {
            cfg.resume_path = argv[++i];
            cfg.load_existing = true;
        }
        else if (arg == "--save" && i + 1 < argc)
        {
            cfg.save_path = argv[++i];
        }
        else if (arg == "--dataset" && i + 1 < argc)
        {
            cfg.dataset_path = argv[++i];
        }
        else if (arg == "--epochs" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --epochs: " << v.error().message << "\n"; std::exit(1); }
            if (*v <= 0) { std::cerr << "--epochs 必须为正整数\n"; std::exit(1); }
            cfg.epochs = *v;
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            auto v = nn::parse_number<double>(argv[++i]);
            if (!v) { std::cerr << "无效 --lr: " << v.error().message << "\n"; std::exit(1); }
            cfg.lr = *v;
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --batch-size: " << v.error().message << "\n"; std::exit(1); }
            if (*v <= 0) { std::cerr << "--batch-size 必须为正整数\n"; std::exit(1); }
            cfg.batch_size = static_cast<std::size_t>(*v);
        }
        else if (arg == "--optimizer" && i + 1 < argc)
        {
            cfg.optimizer_name = argv[++i];
            if (cfg.optimizer_name != "sgd" && cfg.optimizer_name != "sgd_momentum" &&
                cfg.optimizer_name != "adam")
            {
                std::cerr << "未知优化器: " << cfg.optimizer_name
                          << "，可选: sgd, sgd_momentum, adam\n";
                std::exit(1);
            }
        }
        else if (arg == "--layer-dims" && i + 1 < argc)
        {
            std::string dims_str = argv[++i];
            std::stringstream ss(dims_str);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                auto v = nn::parse_number<std::size_t>(token);
                if (!v) { std::cerr << "无效层维度: " << v.error().message << "\n"; std::exit(1); }
                if (*v == 0) { std::cerr << "层维度不能为 0\n"; std::exit(1); }
                cfg.layer_dims.push_back(*v);
            }
            if (cfg.layer_dims.size() < 2)
            {
                std::cerr << "--layer-dims 至少需要 2 个维度\n";
                std::exit(1);
            }
        }
        else if (arg == "--gpu")
        {
            cfg.gpu_enabled = true;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ==================== 数据加载（优化版） ====================
nn::Result<std::pair<nn::Matrix, nn::Matrix>> load_csv(const std::string &filename, int max_samples = -1)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return std::unexpected(nn::Error{"Cannot open file: " + filename});

    const auto file_size = file.tellg();
    file.seekg(0);

    std::string buffer(static_cast<std::size_t>(file_size), '\0');
    file.read(buffer.data(), file_size);
    file.close();

    std::size_t row_count = 0;
    for (char c : buffer)
        if (c == '\n') ++row_count;
    if (row_count == 0)
        return std::unexpected(nn::Error{"CSV file is empty or malformed: " + filename});

    if (max_samples > 0 && static_cast<std::size_t>(max_samples) < row_count)
        row_count = static_cast<std::size_t>(max_samples);

    const char *ptr = buffer.data();
    const char *end = buffer.data() + buffer.size();

    int first_label = 0;
    std::size_t feat_dim = 0;
    {
        const char *p = ptr;
        auto [p1, ec1] = std::from_chars(p, end, first_label);
        p = p1;
        std::size_t cnt = 0;
        while (p < end && *p != '\n' && *p != '\r')
        {
            if (*p == ',')
            {
                ++cnt;
                Scalar tmp;
                auto [p2, ec2] = std::from_chars(p + 1, end, tmp);
                p = p2;
            }
            else
                ++p;
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
        if (ec_label != std::errc{})
            return std::unexpected(nn::Error{"Failed to parse label at row " + std::to_string(row)});
        ptr = p_label;

        for (std::size_t j = 0; j < feat_dim; ++j)
        {
            if (ptr < end && *ptr == ',') ++ptr;
            Scalar val;
            auto [p_feat, ec_feat] = std::from_chars(ptr, end, val);
            if (ec_feat != std::errc{})
                return std::unexpected(nn::Error{"Failed to parse feature at row " + std::to_string(row)});
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
    {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= 10)
            return std::unexpected(nn::Error{"Label out of range: " + std::to_string(lbl)});
        label_mat.set_value_unchecked(lbl, i, 1.0);
    }

    return std::pair{std::move(feat_mat), std::move(label_mat)};
}

// -------------------- 评估函数 --------------------
// 全量前向后下载到 CPU 做 argmax，计算准确率
nn::Result<Scalar> evaluate(nn::Model &model, nn::ComputeEngine &engine,
                             const nn::Matrix &x, const nn::Matrix &y_onehot)
{
    std::size_t N = x.cols();

    // Matrix → Tensor（一次性上传全量数据）
    auto x_tensor_r = engine.from_matrix(x);
    if (!x_tensor_r) return std::unexpected(std::move(x_tensor_r).error());

    auto out_tensor_r = model.forward(*x_tensor_r);
    if (!out_tensor_r) return std::unexpected(std::move(out_tensor_r).error());

    auto out_r = engine.to_matrix(*out_tensor_r);
    if (!out_r) return std::unexpected(std::move(out_r).error());
    const auto &out = *out_r;

    int correct = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        Scalar max_val = out.at_unchecked(0, i);
        int pred = 0;
        for (int j = 1; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
        {
            Scalar val = out.at_unchecked(j, i);
            if (val > max_val)
            {
                max_val = val;
                pred = j;
            }
        }
        int true_label = -1;
        for (int j = 0; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
        {
            if (y_onehot.at_unchecked(j, i) == 1.0)
            {
                true_label = j;
                break;
            }
        }
        if (pred == true_label)
            ++correct;
    }
    return static_cast<Scalar>(correct) / static_cast<Scalar>(N);
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    TrainConfig cfg = parse_args(argc, argv);

    // ── 构建规格 ─────────────────────────────────────────────
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = cfg.layer_dims.empty() ? nn::MNIST_LAYER_DIMS : cfg.layer_dims;

    // ── 如果 --resume，从文件读取规格覆盖 CLI 参数 ──────────
    if (cfg.load_existing)
    {
        auto spec_result = nn::peek_model_spec(cfg.resume_path);
        if (spec_result)
        {
            if (spec_result->type == nn::ModelType::MLP)
            {
                std::cout << "从模型文件读取规格\n";
                spec = std::move(*spec_result);
            }
            else if (spec_result->type != nn::ModelType::Unknown)
            {
                std::cerr << "模型文件类型不是 MLP (type="
                          << static_cast<uint32_t>(spec_result->type)
                          << ")，新架构仅支持 MLP。\n";
                return 1;
            }
            else
            {
                std::cout << "旧格式模型文件 (V1)，使用命令行参数\n";
            }
        }
        else
        {
            std::cerr << "读取模型规格失败: " << spec_result.error().message
                      << "，使用命令行参数。\n";
        }
    }

    // ── 打印配置 ─────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "  MNIST 手写数字训练 (引擎化架构)\n";
    std::cout << "========================================\n";
    std::cout << "  网络: ";
    for (std::size_t i = 0; i < spec.layer_dims.size(); ++i)
    {
        std::cout << spec.layer_dims[i];
        if (i < spec.layer_dims.size() - 2)
            std::cout << "(LayerNorm+GeLU)";
        if (i < spec.layer_dims.size() - 1)
            std::cout << " -> ";
    }
    std::cout << "\n";
    std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
    std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
    std::cout << "  GPU: " << (cfg.gpu_enabled ? "启用" : "禁用") << "\n";
    std::cout << "  模型: " << (cfg.load_existing ? cfg.resume_path : "(从头训练)")
              << " -> " << cfg.save_path << "\n";
    std::cout << "========================================\n\n";

    // ── 创建计算引擎 ─────────────────────────────────────────
    // 引擎必须先于 model 构造并晚于 model 析构（model 持有 engine 的非拥有指针）
    std::unique_ptr<nn::ComputeEngine> engine;

#ifdef NN_HAS_VULKAN
    nn::GpuBackend *gpu_backend = nullptr;
#endif
    if (cfg.gpu_enabled)
    {
#ifdef NN_HAS_VULKAN
        auto &backend = nn::GpuBackend::instance();
        auto init_result = backend.initialize();
        if (init_result)
        {
            gpu_backend = &backend;
            engine = std::make_unique<nn::GpuEngine>(*gpu_backend);
            std::cout << "GPU 加速已启用 (Vulkan GpuEngine)\n\n";
        }
        else
        {
            std::cerr << "GPU 初始化失败: " << init_result.error().message << "\n";
            std::cerr << "回退到 CPU 模式\n\n";
            engine = std::make_unique<nn::CpuEngine>();
        }
#else
        std::cerr << "未编译 Vulkan 支持，使用 CPU 模式\n\n";
        engine = std::make_unique<nn::CpuEngine>();
#endif
    }
    else
    {
        engine = std::make_unique<nn::CpuEngine>();
    }

    // ── 加载数据 ─────────────────────────────────────────────
    std::cout << "加载数据: " << cfg.dataset_path << " ..." << std::endl;
    auto csv_train_result = load_csv(cfg.dataset_path + "/train.csv");
    if (!csv_train_result) { std::cerr << "Error: " << csv_train_result.error().message << '\n'; return 1; }
    auto [train_x, train_y] = std::move(*csv_train_result);

    auto csv_test_result = load_csv(cfg.dataset_path + "/test.csv");
    if (!csv_test_result) { std::cerr << "Error: " << csv_test_result.error().message << '\n'; return 1; }
    auto [test_x, test_y] = std::move(*csv_test_result);
    std::cout << "训练集: " << train_x.cols() << " 样本, 测试集: " << test_x.cols() << " 样本\n" << std::endl;

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    auto model_result = nn::build_mnist_model_from_spec(*engine, spec);
    if (!model_result)
    {
        std::cerr << "构建模型失败: " << model_result.error().message << '\n';
        return 1;
    }
    auto model = std::move(*model_result);

    if (cfg.load_existing)
    {
        auto load_result = nn::load_model(cfg.resume_path, model);
        if (load_result)
        {
            std::cout << "已加载模型: " << cfg.resume_path << "\n" << std::endl;
        }
        else
        {
            std::cerr << "加载模型失败: " << load_result.error().message
                      << "，将从头训练。\n" << std::endl;
        }
    }

    // ── 训练 ─────────────────────────────────────────────────
    auto optimizer = nn::create_optimizer(
        cfg.optimizer_name, *engine,
        model.parameters(), model.param_gradients(), cfg.lr);

    nn::CrossEntropyLoss ce_loss;
    const std::size_t num_batches = train_x.cols() / cfg.batch_size;

    // ── 预分配 batch 缓冲区 ──────────────────────────────────
    nn::Matrix x_batch(train_x.rows(), cfg.batch_size);
    nn::Matrix y_batch(train_y.rows(), cfg.batch_size);

    std::vector<std::size_t> sample_indices(train_x.cols());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;
    std::mt19937_64 shuffle_rng{std::random_device{}()};

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_loss = 0.0;

        std::shuffle(sample_indices.begin(), sample_indices.end(), shuffle_rng);

        for (std::size_t batch = 0; batch < num_batches; ++batch)
        {
            const std::size_t start = batch * cfg.batch_size;

            // ── 提取 batch（按 shuffle 顺序逐列拷贝） ──
            const std::size_t R_x = train_x.rows();
            const std::size_t R_y = train_y.rows();
            const std::size_t C_train = train_x.cols();
            const auto x_src = train_x.span().data();
            const auto y_src = train_y.span().data();
            auto x_dst = x_batch.span().data();
            auto y_dst = y_batch.span().data();
            for (std::size_t b = 0; b < cfg.batch_size; ++b)
            {
                const std::size_t src_col = sample_indices[start + b];
                for (std::size_t r = 0; r < R_x; ++r)
                    x_dst[r * cfg.batch_size + b] = x_src[r * C_train + src_col];
                for (std::size_t r = 0; r < R_y; ++r)
                    y_dst[r * cfg.batch_size + b] = y_src[r * C_train + src_col];
            }

            // ── Matrix → Tensor（上传到引擎设备） ──
            auto x_tensor_r = engine->from_matrix(x_batch);
            if (!x_tensor_r) {
                std::cerr << "\nfrom_matrix(x_batch) failed: " << x_tensor_r.error().message << '\n';
                return 1;
            }
            auto y_tensor_r = engine->from_matrix(y_batch);
            if (!y_tensor_r) {
                std::cerr << "\nfrom_matrix(y_batch) failed: " << y_tensor_r.error().message << '\n';
                return 1;
            }

            // ── 前向 ──
            auto out_r = model.forward(*x_tensor_r);
            if (!out_r) {
                std::cerr << "\nForward pass failed: " << out_r.error().message << '\n';
                return 1;
            }

            // ── 计算损失 ──
            auto loss_result = ce_loss.forward(*engine, *out_r, *y_tensor_r);
            if (!loss_result) {
                std::cerr << "\nLoss computation failed: " << loss_result.error().message << '\n';
                return 1;
            }
            Scalar loss = *loss_result;
            total_loss += loss;

            // ── 反向 ──
            auto grad_r = ce_loss.backward();
            if (!grad_r) { std::cerr << "\nLoss backward failed: " << grad_r.error().message << '\n'; return 1; }

            auto bwd_r = model.backward(*grad_r);
            if (!bwd_r) { std::cerr << "\nModel backward failed: " << bwd_r.error().message << '\n'; return 1; }

            // ── 优化器 step + 梯度清零 ──
            auto step_result = optimizer->step();
            if (!step_result) {
                std::cerr << "\n优化器 step 失败: " << step_result.error().message << '\n';
                return 1;
            }

            auto zero_result = optimizer->zero_grad();
            if (!zero_result) {
                std::cerr << "\n优化器 zero_grad 失败: " << zero_result.error().message << '\n';
                return 1;
            }

            // ── 进度显示 ──
            if ((batch + 1) % 10 == 0 || batch + 1 == num_batches)
            {
                auto batch_now = std::chrono::steady_clock::now();
                Scalar batch_ms = std::chrono::duration<Scalar, std::milli>(batch_now - ep_start).count();
                std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                          << "  batch " << batch + 1 << "/" << num_batches
                          << "  loss: " << std::fixed << std::setprecision(4) << loss
                          << "  time: " << std::setprecision(0) << batch_ms << "ms"
                          << "   " << std::flush;
            }
        }

        auto ep_end = std::chrono::steady_clock::now();
        Scalar ep_sec = std::chrono::duration<Scalar>(ep_end - ep_start).count();

        Scalar avg_loss = total_loss / num_batches;
        auto train_acc_r = evaluate(model, *engine, train_x, train_y);
        auto test_acc_r  = evaluate(model, *engine, test_x, test_y);
        if (!train_acc_r || !test_acc_r)
        {
            const auto &err = !train_acc_r ? train_acc_r.error() : test_acc_r.error();
            std::cerr << "\n  评估失败: " << err.message << std::endl;
            return 1;
        }
        Scalar train_acc = *train_acc_r;
        Scalar test_acc  = *test_acc_r;

        std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                  << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  train_acc=" << std::setprecision(2) << train_acc * 100.0 << "%"
                  << "  test_acc=" << test_acc * 100.0 << "%"
                  << "  time=" << std::setprecision(1) << ep_sec << "s"
                  << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    Scalar total_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    // ── 保存模型 ─────────────────────────────────────────────
    auto save_result = nn::save_model(cfg.save_path, model, spec);
    if (!save_result)
    {
        std::cerr << "保存模型失败: " << save_result.error().message << '\n';
        return 1;
    }
    std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1) << total_sec << "s" << std::endl;

    return 0;
}
