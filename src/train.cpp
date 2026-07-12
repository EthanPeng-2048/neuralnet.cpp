#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_io.hpp>
#include <cstring>     // for std::memcpy
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>

// ==================== 常量 ====================
static constexpr std::size_t INPUT_DIM = 784;
static constexpr std::size_t HIDDEN_DIM = 64;
static constexpr std::size_t NUM_CLASSES = 10;
static constexpr std::size_t NUM_HIDDEN_LAYERS = 3;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字训练程序\n\n"
        << "用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --resume <path>    从已有模型恢复训练\n"
        << "  --save <path>      模型保存路径 (默认: mnist_model.bin)\n"
        << "  --dataset <path>   数据集目录 (默认: datasets/mnist_data)\n"
        << "  --epochs <n>       训练轮数 (默认: 5)\n"
        << "  --lr <lr>          学习率 (默认: 0.01)\n"
        << "  --batch-size <n>   批大小 (默认: 64)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string save_path = "mnist_model.bin";
    std::string dataset_path = "datasets/mnist_data";
    std::string resume_path;
    int epochs = 5;
    double lr = 0.01;
    std::size_t batch_size = 64;
    bool load_existing = false;
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
            cfg.epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            cfg.lr = std::stod(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            cfg.batch_size = std::stoi(argv[++i]);
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ==================== 数据加载 ====================
std::pair<nn::Matrix, nn::Matrix> load_csv(const std::string &filename, int max_samples = -1)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::vector<double> features;
    std::vector<int> labels;
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        int label;
        std::getline(ss, token, ',');
        label = std::stoi(token);
        labels.push_back(label);

        while (std::getline(ss, token, ','))
        {
            features.push_back(std::stod(token)); // 数据已由 ToTensor() 归一化到 [0,1]
        }
    }

    std::size_t N = labels.size();
    std::size_t feat_dim = features.size() / N; // 784

    if (max_samples > 0 && static_cast<std::size_t>(max_samples) < N)
    {
        N = max_samples;
        features.resize(N * feat_dim);
        labels.resize(N);
    }

    // 特征矩阵：形状 (feat_dim, N)
    nn::Matrix feat_mat(feat_dim, N);
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < feat_dim; ++j)
        {
            feat_mat.set_value_unchecked(j, i, features[i * feat_dim + j]);
        }
    }

    // One‑hot 标签矩阵：形状 (10, N)
    nn::Matrix label_mat(10, N);
    for (std::size_t i = 0; i < N; ++i)
    {
        int lbl = labels[i];
        if (lbl < 0 || lbl >= 10)
            throw std::out_of_range("Label out of range");
        label_mat.set_value_unchecked(lbl, i, 1.0);
    }

    return {feat_mat, label_mat};
}

// -------------------- 评估函数 --------------------
double evaluate(nn::Model &model, const nn::Matrix &x, const nn::Matrix &y_onehot)
{
    std::size_t N = x.cols();
    auto out = model.forward(x);
    int correct = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        double max_val = out.at_unchecked(0, i);
        int pred = 0;
        for (int j = 1; j < static_cast<int>(NUM_CLASSES); ++j)
        {
            double val = out.at_unchecked(j, i);
            if (val > max_val)
            {
                max_val = val;
                pred = j;
            }
        }
        int true_label = -1;
        for (int j = 0; j < static_cast<int>(NUM_CLASSES); ++j)
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
    return static_cast<double>(correct) / N;
}

// -------------------- 构建网络 --------------------
nn::Model build_model()
{
    nn::Model model;
    model.add<nn::Linear>(INPUT_DIM, HIDDEN_DIM)
         .add<nn::ReLU>();
    for (std::size_t i = 0; i < NUM_HIDDEN_LAYERS - 1; ++i)
    {
        model.add<nn::Linear>(HIDDEN_DIM, HIDDEN_DIM)
             .add<nn::ReLU>();
    }
    model.add<nn::Linear>(HIDDEN_DIM, NUM_CLASSES);
    return model;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    try
    {
        TrainConfig cfg = parse_args(argc, argv);

        // ── 打印配置 ─────────────────────────────────────────────
        std::cout << "========================================\n";
        std::cout << "  MNIST 手写数字训练\n";
        std::cout << "========================================\n";
        std::cout << "  网络: " << INPUT_DIM;
        for (std::size_t i = 0; i < NUM_HIDDEN_LAYERS; ++i)
            std::cout << " -> " << HIDDEN_DIM << "(ReLU)";
        std::cout << " -> " << NUM_CLASSES << "\n";
        std::cout << "  优化器: SGD+Momentum  学习率: " << cfg.lr << "\n";
        std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
        std::cout << "  模型: " << (cfg.load_existing ? cfg.resume_path : "(从头训练)")
                  << " -> " << cfg.save_path << "\n";
        std::cout << "========================================\n\n";

        // ── 加载数据 ─────────────────────────────────────────────
        std::cout << "加载数据: " << cfg.dataset_path << " ..." << std::endl;
        auto [train_x, train_y] = load_csv(cfg.dataset_path + "/train.csv");
        auto [test_x, test_y] = load_csv(cfg.dataset_path + "/test.csv");
        std::cout << "训练集: " << train_x.cols() << " 样本, 测试集: " << test_x.cols() << " 样本\n" << std::endl;

        // ── 构建模型 ─────────────────────────────────────────────
        auto model = build_model();

        if (cfg.load_existing)
        {
            try
            {
                nn::load_model(cfg.resume_path, model);
                std::cout << "已加载模型: " << cfg.resume_path << "\n" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "加载模型失败: " << e.what() << "，将从头训练。\n" << std::endl;
            }
        }

        // ── 训练 ─────────────────────────────────────────────────
        nn::SGD_w_Momentum optimizer(model.parameters(), model.param_gradients(), cfg.lr);
        nn::CrossEntropyLoss ce_loss;
        const std::size_t num_batches = train_x.cols() / cfg.batch_size;

        // ── 预分配 batch 缓冲区（避免每次迭代重复分配） ──────────
        nn::Matrix x_batch(train_x.rows(), cfg.batch_size);
        nn::Matrix y_batch(train_y.rows(), cfg.batch_size);

        auto t_start = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < cfg.epochs; ++epoch)
        {
            auto ep_start = std::chrono::steady_clock::now();
            double total_loss = 0.0;

            for (std::size_t batch = 0; batch < num_batches; ++batch)
            {
                const std::size_t start = batch * cfg.batch_size;

                // ── 行优先 memcpy 提取 batch（比逐列复制更缓存友好） ─────
                for (std::size_t r = 0; r < train_x.rows(); ++r)
                    std::memcpy(x_batch.data_ptr() + r * cfg.batch_size,
                                train_x.data_ptr() + r * train_x.cols() + start,
                                cfg.batch_size * sizeof(double));
                for (std::size_t r = 0; r < train_y.rows(); ++r)
                    std::memcpy(y_batch.data_ptr() + r * cfg.batch_size,
                                train_y.data_ptr() + r * train_y.cols() + start,
                                cfg.batch_size * sizeof(double));

                auto out = model.forward(x_batch);
                double loss = ce_loss.forward(out, y_batch);
                total_loss += loss;

                auto grad = ce_loss.backward();
                model.backward(grad);

                optimizer.step();
                optimizer.zero_grad();

                // 进度显示
                if ((batch + 1) % 100 == 0 || batch + 1 == num_batches)
                {
                    std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                              << "  batch " << batch + 1 << "/" << num_batches
                              << "  loss: " << std::fixed << std::setprecision(4) << loss
                              << "   " << std::flush;
                }
            }

            auto ep_end = std::chrono::steady_clock::now();
            double ep_sec = std::chrono::duration<double>(ep_end - ep_start).count();

            double avg_loss = total_loss / num_batches;
            double train_acc = evaluate(model, train_x, train_y);
            double test_acc = evaluate(model, test_x, test_y);

            std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                      << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                      << "  train_acc=" << std::setprecision(2) << train_acc * 100.0 << "%"
                      << "  test_acc=" << test_acc * 100.0 << "%"
                      << "  time=" << std::setprecision(1) << ep_sec << "s"
                      << std::endl;
        }

        auto t_end = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();

        // ── 保存模型 ─────────────────────────────────────────────
        nn::save_model(cfg.save_path, model);
        std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1) << total_sec << "s" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
}
