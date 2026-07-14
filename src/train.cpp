#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_io.hpp>
#include <neuralnet.cpp/mnist_common.hpp>
#include <cstring>     // for std::memcpy
#include <memory>     // for std::unique_ptr
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
// 现在使用 nn::MNIST_INPUT_DIM, nn::MNIST_NUM_CLASSES, nn::MNIST_LAYER_DIMS

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字训练程序\n\n"
        << "用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --model-type <t>  模型类型: mlp/transformer (默认: mlp)\n"
        << "  --resume <path>    从已有模型恢复训练\n"
        << "  --save <path>      模型保存路径 (默认: mnist_model.bin)\n"
        << "  --dataset <path>   数据集目录 (默认: datasets/mnist_data)\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 64)\n"
        << "  --optimizer <name> 优化器: sgd/momentum/adam (默认: adam)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string save_path = "mnist_model.bin";
    std::string dataset_path = "datasets/mnist_data";
    std::string resume_path;
    std::string optimizer_name = "adam";
    std::string model_type = "mlp";
    int epochs = 10;  // 增加训练轮数
    double lr = 0.001;  // 调整学习率
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
            int val = std::stoi(argv[++i]);
            if (val <= 0) { std::cerr << "--epochs 必须为正整数\n"; std::exit(1); }
            cfg.epochs = val;
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            cfg.lr = std::stod(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            int val = std::stoi(argv[++i]);
            if (val <= 0) { std::cerr << "--batch-size 必须为正整数\n"; std::exit(1); }
            cfg.batch_size = static_cast<std::size_t>(val);
        }
        else if (arg == "--optimizer" && i + 1 < argc)
        {
            cfg.optimizer_name = argv[++i];
            if (cfg.optimizer_name != "sgd" && cfg.optimizer_name != "sgd_w_momentum" &&
                cfg.optimizer_name != "adam")
            {
                std::cerr << "未知优化器: " << cfg.optimizer_name
                          << "，可选: sgd, sgd_w_momentum, adam\n";
                std::exit(1);
            }
        }
        else if (arg == "--model-type" && i + 1 < argc)
        {
            cfg.model_type = argv[++i];
            if (cfg.model_type != "mlp" && cfg.model_type != "transformer")
            {
                std::cerr << "未知模型类型: " << cfg.model_type
                          << "，可选: mlp, transformer\n";
                std::exit(1);
            }
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
// 优化策略：
//   1. 整文件一次性读入内存，避免逐行 I/O 开销
//   2. 使用 std::from_chars (C++17) 替代 std::stod，无临时 string 分配
//   3. 向量预分配 reserve()，避免多次扩容拷贝
//   4. 直接指针遍历 buffer，省去 std::stringstream 开销
std::pair<nn::Matrix, nn::Matrix> load_csv(const std::string &filename, int max_samples = -1)
{
    // ── 1. 整文件读入 ──────────────────────────────────────────────
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    const auto file_size = file.tellg();
    file.seekg(0);

    std::string buffer(static_cast<std::size_t>(file_size), '\0');
    file.read(buffer.data(), file_size);
    file.close();

    // ── 2. 预扫描行数以预分配向量 ──────────────────────────────────
    std::size_t row_count = 0;
    for (char c : buffer)
        if (c == '\n') ++row_count;
    if (row_count == 0)
        throw std::runtime_error("CSV file is empty or malformed: " + filename);

    if (max_samples > 0 && static_cast<std::size_t>(max_samples) < row_count)
        row_count = static_cast<std::size_t>(max_samples);

    // ── 3. 解析：使用 from_chars 直接从 buffer 读取数值 ───────────
    const char *ptr = buffer.data();
    const char *end = buffer.data() + buffer.size();

    // 先解析第一行确定 feat_dim
    int first_label = 0;
    std::size_t feat_dim = 0;
    {
        // 跳过首行用于计数列数
        const char *p = ptr;
        auto [p1, ec1] = std::from_chars(p, end, first_label);
        p = p1;
        std::size_t cnt = 0;
        while (p < end && *p != '\n' && *p != '\r')
        {
            if (*p == ',')
            {
                ++cnt;
                double tmp;
                auto [p2, ec2] = std::from_chars(p + 1, end, tmp);
                p = p2;
            }
            else
                ++p;
        }
        feat_dim = cnt;  // 标签后的逗号数 = 特征数
    }

    std::vector<double> features(row_count * feat_dim);
    std::vector<int>    labels(row_count);

    std::size_t row = 0;
    ptr = buffer.data();  // 重置指针

    while (ptr < end && row < row_count)
    {
        // 解析标签
        auto [p_label, ec_label] = std::from_chars(ptr, end, labels[row]);
        if (ec_label != std::errc{})
            throw std::runtime_error("Failed to parse label at row " + std::to_string(row));
        ptr = p_label;

        // 解析特征
        for (std::size_t j = 0; j < feat_dim; ++j)
        {
            // 跳过逗号
            if (ptr < end && *ptr == ',') ++ptr;

            double val;
            auto [p_feat, ec_feat] = std::from_chars(ptr, end, val);
            if (ec_feat != std::errc{})
                throw std::runtime_error("Failed to parse feature at row " + std::to_string(row));
            features[row * feat_dim + j] = val;
            ptr = p_feat;
        }

        // 跳过行尾 (\r\n 或 \n)
        while (ptr < end && (*ptr == '\r' || *ptr == '\n')) ++ptr;
        ++row;
    }

    // 实际读取的行数可能少于预扫描（例如文件末尾空行）
    std::size_t N = row;

    // ── 4. 直接写入矩阵（行主序存储，逐列填充） ──────────────────
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
            throw std::out_of_range("Label out of range: " + std::to_string(lbl));
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
        for (int j = 1; j < static_cast<int>(nn::MNIST_NUM_CLASSES); ++j)
        {
            double val = out.at_unchecked(j, i);
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
    return static_cast<double>(correct) / N;
}

// -------------------- 构建网络 --------------------
// build_model 函数已移至 neuralnet.cpp/mnist_common.hpp 中的 nn::build_mnist_model()

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
        std::cout << "  模型类型: " << cfg.model_type << "\n";
        if (cfg.model_type == "mlp")
        {
            std::cout << "  网络: ";
            for (std::size_t i = 0; i < nn::MNIST_LAYER_DIMS.size(); ++i)
            {
                std::cout << nn::MNIST_LAYER_DIMS[i];
                if (i < nn::MNIST_LAYER_DIMS.size() - 2)
                    std::cout << "(LayerNorm+GeLU)";
                if (i < nn::MNIST_LAYER_DIMS.size() - 1)
                    std::cout << " -> ";
            }
            std::cout << "\n";
        }
        else
        {
            std::cout << "  网络: PatchEmbedding(7×7) -> TransformerEncoder("
                      << nn::TRANSFORMER_NUM_LAYERS << "层, d="
                      << nn::TRANSFORMER_D_MODEL << ", heads="
                      << nn::TRANSFORMER_NUM_HEADS << ") -> Linear("
                      << nn::TRANSFORMER_D_MODEL << "→"
                      << nn::MNIST_NUM_CLASSES << ")\n";
        }
        std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
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
        auto model = nn::build_mnist_model(cfg.model_type);

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
        std::unique_ptr<nn::Optimizer> optimizer;
        if (cfg.optimizer_name == "sgd")
            optimizer = std::make_unique<nn::SGD>(model.parameters(), model.param_gradients(), cfg.lr);
        else if (cfg.optimizer_name == "sgd_w_momentum")
            optimizer = std::make_unique<nn::SGD_w_Momentum>(model.parameters(), model.param_gradients(), cfg.lr);
        else
            optimizer = std::make_unique<nn::Adam>(model.parameters(), model.param_gradients(), cfg.lr);

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
                    std::memcpy(x_batch.span().data() + r * cfg.batch_size,
                                train_x.span().data() + r * train_x.cols() + start,
                                cfg.batch_size * sizeof(double));
                for (std::size_t r = 0; r < train_y.rows(); ++r)
                    std::memcpy(y_batch.span().data() + r * cfg.batch_size,
                                train_y.span().data() + r * train_y.cols() + start,
                                cfg.batch_size * sizeof(double));

                auto out = model.forward(x_batch);
                double loss = ce_loss.forward(out, y_batch);
                total_loss += loss;

                auto grad = ce_loss.backward();
                model.backward(grad);

                optimizer->step();
                optimizer->zero_grad();

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
