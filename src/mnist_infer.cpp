#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字推理程序\n\n"
        << "用法:\n"
        << "  " << prog << " <image.csv> [选项]     推理单张图片\n"
        << "  " << prog << " <目录>   [选项]     批量推理目录下所有 CSV\n\n"
        << "选项:\n"
        << "  --model <path>     模型文件路径 (默认: pretrained/model.bin)\n"
        << "                       V2 格式模型文件自动读取模型规格，无需指定 --model-type\n"
        << "  --model-type <t>   模型类型: mlp/transformer (默认: mlp)\n"
        << "                       仅在 V1 旧格式模型文件时需要手动指定\n"
        << "  --topk <n>         显示前 n 个预测结果 (默认: 3)\n"
        << "  --show-pixels      显示像素矩阵 (调试用)\n"
        << "  --gpu              启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct InferConfig
{
    std::string model_path = "pretrained/model.bin";
    std::string model_type = "mlp";   // 仅 V1 旧格式使用
    std::string input_path;
    int topk = 3;
    bool show_pixels = false;
    bool gpu_enabled = false;
};

nn::Result<InferConfig> parse_args(int argc, char *argv[])
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
        else if (arg == "--model-type" && i + 1 < argc)
        {
            cfg.model_type = argv[++i];
            if (cfg.model_type != "mlp" && cfg.model_type != "transformer")
                return std::unexpected(nn::Error{"未知模型类型: " + cfg.model_type +
                                                 "，可选: mlp, transformer"});
        }
        else if (arg == "--topk" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) return std::unexpected(std::move(v).error());
            if (*v <= 0)
                return std::unexpected(nn::Error{"--topk 必须为正整数"});
            cfg.topk = *v;
        }
        else if (arg == "--show-pixels")
        {
            cfg.show_pixels = true;
        }
        else if (arg == "--gpu")
        {
            cfg.gpu_enabled = true;
        }
        else if (!arg.starts_with("--"))
        {
            cfg.input_path = arg;
            has_input = true;
        }
        else
        {
            return std::unexpected(nn::Error{"未知参数: " + arg + "，使用 --help 查看用法"});
        }
    }

    if (!has_input)
        return std::unexpected(nn::Error{"请指定图片文件或目录，使用 --help 查看用法"});

    return cfg;
}

// ==================== 数据读取 ====================
// 委托给 mnist_common.hpp 的共享工具，消除 mnist_train / mnist_infer 重复实现
nn::Result<nn::Matrix> load_image_from_csv(const std::string &csv_line)
{
    return nn::load_image_from_csv_line(csv_line);
}

// ==================== 推理 + 置信度 ====================
struct Prediction
{
    int digit;
    Scalar confidence;
};

nn::Result<std::vector<Prediction>> predict_with_confidence(nn::Model &model, const nn::Matrix &img, int topk)
{
    auto logits_result = model.forward(img);
    if (!logits_result)
        return std::unexpected(std::move(logits_result).error());
    auto logits = std::move(*logits_result);

    // Softmax 计算概率
    Scalar max_val = logits.at_unchecked(0, 0);
    for (int c = 1; c < static_cast<int>(nn::MNIST_NUM_CLASSES); ++c)
        max_val = std::max(max_val, logits.at_unchecked(c, 0));

    Scalar sum_exp = 0.0;
    std::vector<Scalar> probs(nn::MNIST_NUM_CLASSES);
    for (std::size_t c = 0; c < nn::MNIST_NUM_CLASSES; ++c)
    {
        probs[c] = std::exp(logits.at_unchecked(c, 0) - max_val);
        sum_exp += probs[c];
    }
    for (auto &p : probs)
        p /= sum_exp;

    // 按概率排序取 top-k
    std::vector<int> indices(nn::MNIST_NUM_CLASSES);
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
            Scalar val = img.at_unchecked(r * 28 + c, 0);
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
nn::Result<void> infer_single(nn::Model &model, const std::string &filepath, const InferConfig &cfg)
{
    std::ifstream file(filepath);
    if (!file)
        return std::unexpected(nn::Error{"无法打开文件: " + filepath});

    std::string line;
    std::getline(file, line);
    auto img_result = load_image_from_csv(line);
    if (!img_result)
        return std::unexpected(std::move(img_result).error());
    auto img = std::move(*img_result);

    if (cfg.show_pixels)
    {
        std::cout << "像素预览:\n";
        show_pixels(img);
        std::cout << "\n";
    }

    auto results = predict_with_confidence(model, img, cfg.topk);
    if (!results)
        return std::unexpected(std::move(results).error());

    std::cout << fs::path(filepath).filename().string() << " -> ";
    for (std::size_t k = 0; k < results->size(); ++k)
    {
        if (k > 0) std::cout << "  ";
        std::cout << (*results)[k].digit
                  << " (" << std::fixed << std::setprecision(1) << (*results)[k].confidence * 100.0 << "%)";
    }
    std::cout << std::endl;
    return {};
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    auto cfg_result = parse_args(argc, argv);
    if (!cfg_result)
    {
        std::cerr << "参数错误: " << cfg_result.error().message << std::endl;
        return 1;
    }
    InferConfig cfg = std::move(*cfg_result);

    // ── 从模型文件读取规格 ─────────────────────────────────
    auto spec_result = nn::peek_model_spec(cfg.model_path);
    if (!spec_result)
    {
        std::cerr << "读取模型文件失败: " << spec_result.error().message << std::endl;
        return 1;
    }
    nn::ModelSpec spec = spec_result.value();

    nn::Model model;
    if (spec.type != nn::ModelType::Unknown)
    {
        // V2 格式：自动从规格构建模型
        std::cout << "从模型文件读取规格 (V2 格式)\n";
        auto build_result = nn::build_mnist_model_from_spec(spec);
        if (!build_result)
        {
            std::cerr << "构建模型失败: " << build_result.error().message << std::endl;
            return 1;
        }
        model = std::move(*build_result);
    }
    else
    {
        // V1 旧格式：使用 --model-type 参数
        std::cout << "旧格式模型文件 (V1)，使用 --model-type 参数: " << cfg.model_type << "\n";
        auto build_result = nn::build_mnist_model(cfg.model_type);
        if (!build_result)
        {
            std::cerr << "构建模型失败: " << build_result.error().message << std::endl;
            return 1;
        }
        model = std::move(*build_result);
    }

    auto load_result = nn::load_model(cfg.model_path, model);
    if (!load_result)
    {
        std::cerr << "加载模型失败: " << load_result.error().message << std::endl;
        return 1;
    }
    std::cout << "模型已加载: " << cfg.model_path << "\n" << std::endl;

#ifdef NN_HAS_VULKAN
    if (cfg.gpu_enabled)
    {
        auto &backend = nn::GpuBackend::instance();
        auto init_result = backend.initialize();
        if (init_result)
        {
            nn::SmartPolicy::gpu_enabled = true;
            std::cout << "GPU 加速已启用 (Vulkan)\n\n";
        }
        else
        {
            std::cerr << "GPU 初始化失败: " << init_result.error().message << "\n";
            std::cerr << "回退到 CPU 模式\n\n";
        }
    }
#endif

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
            auto result = infer_single(model, f.string(), cfg);
            if (!result) { std::cerr << "推理失败: " << f.filename().string() << ": " << result.error().message << '\n'; }
        }

        std::cout << "\n共推理 " << csv_files.size() << " 张图片" << std::endl;
    }
    else if (fs::is_regular_file(input))
    {
        // 单张推理
        auto result = infer_single(model, input.string(), cfg);
        if (!result) { std::cerr << "推理失败: " << result.error().message << '\n'; return 1; }
    }
    else
    {
        std::cerr << "输入路径不存在: " << cfg.input_path << std::endl;
        return 1;
    }

    return 0;
}
