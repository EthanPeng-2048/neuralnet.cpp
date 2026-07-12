#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_io.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// ==================== 常量 ====================
static constexpr std::size_t INPUT_DIM = 784;
static constexpr std::size_t NUM_CLASSES = 10;
// 网络架构：必须与训练时一致
static const std::vector<std::size_t> LAYER_DIMS = {INPUT_DIM, 256, 128, 64, NUM_CLASSES};

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字推理程序\n\n"
        << "用法:\n"
        << "  " << prog << " <image.csv> [选项]     推理单张图片\n"
        << "  " << prog << " <目录>   [选项]     批量推理目录下所有 CSV\n\n"
        << "选项:\n"
        << "  --model <path>    模型文件路径 (默认: pretrained/model.bin)\n"
        << "  --topk <n>        显示前 n 个预测结果 (默认: 3)\n"
        << "  --show-pixels     显示像素矩阵 (调试用)\n"
        << "  --help            显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct InferConfig
{
    std::string model_path = "pretrained/model.bin";
    std::string input_path;
    int topk = 3;
    bool show_pixels = false;
};

InferConfig parse_args(int argc, char *argv[])
{
    InferConfig cfg;
    bool has_input = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            cfg.model_path = argv[++i];
        }
        else if (arg == "--topk" && i + 1 < argc)
        {
            cfg.topk = std::stoi(argv[++i]);
        }
        else if (arg == "--show-pixels")
        {
            cfg.show_pixels = true;
        }
        else if (!arg.starts_with("--"))
        {
            cfg.input_path = arg;
            has_input = true;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    if (!has_input)
    {
        std::cerr << "请指定图片文件或目录\n使用 --help 查看用法\n";
        std::exit(1);
    }

    return cfg;
}

// ==================== 构建网络 ====================
nn::Model build_model()
{
    nn::Model model;
    // 根据 LAYER_DIMS 自动构建网络（与训练时完全一致）
    for (std::size_t i = 0; i < LAYER_DIMS.size() - 1; ++i)
    {
        std::size_t in_dim = LAYER_DIMS[i];
        std::size_t out_dim = LAYER_DIMS[i + 1];
        
        // 添加线性层
        model.add<nn::Linear>(in_dim, out_dim);
        
        // 如果不是最后一层（输出层），则添加 LayerNorm 和 GeLU 激活函数
        if (i < LAYER_DIMS.size() - 2)
        {
            model.add<nn::LayerNorm>(out_dim)
                 .add<nn::GeLU>();
        }
    }
    return model;
}

// ==================== 数据读取 ====================
nn::Matrix load_image_from_csv(const std::string &csv_line)
{
    std::vector<double> pixels;
    std::stringstream ss(csv_line);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        pixels.push_back(std::stod(token));
    }
    if (pixels.size() != INPUT_DIM)
        throw std::runtime_error("CSV 必须包含恰好 " + std::to_string(INPUT_DIM) + " 个值，实际: " + std::to_string(pixels.size()));

    nn::Matrix img(INPUT_DIM, 1);
    for (std::size_t i = 0; i < INPUT_DIM; ++i)
        img.set_value_unchecked(i, 0, pixels[i]);
    return img;
}

// ==================== 推理 + 置信度 ====================
struct Prediction
{
    int digit;
    double confidence;
};

std::vector<Prediction> predict_with_confidence(nn::Model &model, const nn::Matrix &img, int topk)
{
    auto logits = model.forward(img);

    // Softmax 计算概率
    double max_val = logits.at_unchecked(0, 0);
    for (int c = 1; c < static_cast<int>(NUM_CLASSES); ++c)
        max_val = std::max(max_val, logits.at_unchecked(c, 0));

    double sum_exp = 0.0;
    std::vector<double> probs(NUM_CLASSES);
    for (std::size_t c = 0; c < NUM_CLASSES; ++c)
    {
        probs[c] = std::exp(logits.at_unchecked(c, 0) - max_val);
        sum_exp += probs[c];
    }
    for (auto &p : probs)
        p /= sum_exp;

    // 按概率排序取 top-k
    std::vector<int> indices(NUM_CLASSES);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<Prediction> results;
    for (int k = 0; k < topk; ++k)
        results.push_back({indices[k], probs[indices[k]]});

    return results;
}

// ==================== 显示像素矩阵 ====================
void show_pixels(const nn::Matrix &img)
{
    for (int r = 0; r < 28; ++r)
    {
        std::cout << "  ";
        for (int c = 0; c < 28; ++c)
        {
            double val = img.at_unchecked(r * 28 + c, 0);
            if (val > 0.5)
                std::cout << "##";
            else if (val > 0.1)
                std::cout << "..";
            else
                std::cout << "  ";
        }
        std::cout << "\n";
    }
}

// ==================== 推理单张图片 ====================
void infer_single(nn::Model &model, const std::string &filepath, const InferConfig &cfg)
{
    std::ifstream file(filepath);
    if (!file)
        throw std::runtime_error("无法打开文件: " + filepath);

    std::string line;
    std::getline(file, line);
    auto img = load_image_from_csv(line);

    if (cfg.show_pixels)
    {
        std::cout << "像素预览:\n";
        show_pixels(img);
        std::cout << "\n";
    }

    auto results = predict_with_confidence(model, img, cfg.topk);

    std::cout << fs::path(filepath).filename().string() << " -> ";
    for (std::size_t k = 0; k < results.size(); ++k)
    {
        if (k > 0) std::cout << "  ";
        std::cout << results[k].digit
                  << " (" << std::fixed << std::setprecision(1) << results[k].confidence * 100.0 << "%)";
    }
    std::cout << std::endl;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    try
    {
        InferConfig cfg = parse_args(argc, argv);

        // 构建并加载模型
        auto model = build_model();
        nn::load_model(cfg.model_path, model);
        std::cout << "模型已加载: " << cfg.model_path << "\n" << std::endl;

        fs::path input(cfg.input_path);

        if (fs::is_directory(input))
        {
            // 批量推理
            std::vector<fs::path> csv_files;
            for (auto &entry : fs::directory_iterator(input))
            {
                if (entry.path().extension() == ".csv")
                    csv_files.push_back(entry.path());
            }
            std::sort(csv_files.begin(), csv_files.end());

            if (csv_files.empty())
            {
                std::cerr << "目录中没有 CSV 文件: " << cfg.input_path << std::endl;
                return 1;
            }

            std::cout << "找到 " << csv_files.size() << " 个文件\n" << std::endl;

            for (auto &f : csv_files)
            {
                infer_single(model, f.string(), cfg);
            }

            std::cout << "\n共推理 " << csv_files.size() << " 张图片" << std::endl;
        }
        else if (fs::is_regular_file(input))
        {
            // 单张推理
            infer_single(model, input.string(), cfg);
        }
        else
        {
            std::cerr << "输入路径不存在: " << cfg.input_path << std::endl;
            return 1;
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}