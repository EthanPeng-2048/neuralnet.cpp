#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_io.hpp>
#include <neuralnet.cpp/gpt_common.hpp>
#include <cstring>
#include <memory>
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
#include <random>
#include <numeric>

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "GPT 文本生成训练程序\n\n"
        << "用法: " << prog << " <text-file> [选项]\n\n"
        << "参数:\n"
        << "  <text-file>        训练文本文件路径 (必需)\n\n"
        << "选项:\n"
        << "  --save <path>      模型保存路径 (默认: gpt_model.bin)\n"
        << "  --resume <path>    从已有模型恢复训练\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 32)\n"
        << "  --seq-len <n>      序列长度 (默认: 256)\n"
        << "  --optimizer <name> 优化器: sgd/sgd_w_momentum/adam (默认: adam)\n"
        << "  --d-model <n>      模型维度 (默认: 128)\n"
        << "  --num-heads <n>    注意力头数 (默认: 4)\n"
        << "  --num-layers <n>   Transformer 层数 (默认: 4)\n"
        << "  --d-ff <n>         FFN 中间维度 (默认: 512)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string text_path;
    std::string save_path = "gpt_model.bin";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    double lr = 0.001;
    std::size_t batch_size = 32;
    std::size_t seq_len = 256;
    std::size_t d_model = nn::GPT_D_MODEL;
    std::size_t num_heads = nn::GPT_NUM_HEADS;
    std::size_t num_layers = nn::GPT_NUM_LAYERS;
    std::size_t d_ff = nn::GPT_D_FF;
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
        else if (arg == "--save" && i + 1 < argc)
            cfg.save_path = argv[++i];
        else if (arg == "--resume" && i + 1 < argc)
        {
            cfg.resume_path = argv[++i];
            cfg.load_existing = true;
        }
        else if (arg == "--epochs" && i + 1 < argc)
            cfg.epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc)
            cfg.lr = std::stod(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc)
            cfg.batch_size = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--seq-len" && i + 1 < argc)
            cfg.seq_len = static_cast<std::size_t>(std::stoi(argv[++i]));
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
        else if (arg == "--d-model" && i + 1 < argc)
            cfg.d_model = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--num-heads" && i + 1 < argc)
            cfg.num_heads = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--num-layers" && i + 1 < argc)
            cfg.num_layers = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--d-ff" && i + 1 < argc)
            cfg.d_ff = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (!arg.starts_with("--"))
            cfg.text_path = arg;
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    if (cfg.text_path.empty())
    {
        std::cerr << "请指定训练文本文件\n使用 --help 查看用法\n";
        std::exit(1);
    }

    return cfg;
}

// ==================== One-Hot 编码 ====================
nn::Matrix one_hot_labels(const std::vector<std::size_t> &tokens, std::size_t vocab_size)
{
    const std::size_t n = tokens.size();
    nn::Matrix result(vocab_size, n);
    result.zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (tokens[i] < vocab_size)
            result.set_value_unchecked(tokens[i], i, 1.0);
    }
    return result;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    try
    {
        TrainConfig cfg = parse_args(argc, argv);

        // ── 加载文本 ─────────────────────────────────────────────
        std::cout << "加载文本: " << cfg.text_path << " ..." << std::endl;
        auto text_result = nn::load_text_file(cfg.text_path);
        if (!text_result) {
            std::cerr << "Error: " << text_result.error().message << '\n';
            return 1;
        }
        std::string text = std::move(*text_result);
        if (text.empty())
        {
            std::cerr << "文本文件为空\n";
            return 1;
        }

        nn::BPETokenizer tokenizer;
        {
            auto vocab_result = tokenizer.load_vocab("gpt_bpe.json");
            if (!vocab_result) {
                std::cerr << "Error: " << vocab_result.error().message << '\n';
                return 1;
            }
        }
        auto all_tokens = tokenizer.encode(text);
        std::cout << "文本长度: " << text.size() << " 字符, "
                  << all_tokens.size() << " tokens, "
                  << tokenizer.vocab_size() << " 词表\n" << std::endl;

        // ── 打印配置 ─────────────────────────────────────────────
        std::cout << "========================================\n";
        std::cout << "  GPT 文本生成训练\n";
        std::cout << "========================================\n";
        std::cout << "  词表大小: " << tokenizer.vocab_size() << "\n";
        std::cout << "  模型维度: " << cfg.d_model << "\n";
        std::cout << "  注意力头: " << cfg.num_heads << "\n";
        std::cout << "  Transformer 层数: " << cfg.num_layers << "\n";
        std::cout << "  FFN 维度: " << cfg.d_ff << "\n";
        std::cout << "  序列长度: " << cfg.seq_len << "\n";
        std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
        std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
        std::cout << "========================================\n\n";

        // ── 构建模型 ─────────────────────────────────────────────
        auto model = nn::build_gpt_model(
            tokenizer.vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers);

        // ── 构建规格（用于保存） ─────────────────────────────────
        auto spec = nn::make_gpt_spec(
            tokenizer.vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers);

        if (cfg.load_existing)
        {
            // 尝试从文件读取规格（V2 格式）
            auto spec_result = nn::peek_model_spec(cfg.resume_path);
            if (!spec_result)
            {
                std::cerr << "加载模型失败: " << spec_result.error().message << "，将从头训练。\n" << std::endl;
            }
            else
            {
                auto file_spec = std::move(*spec_result);
                if (file_spec.is_gpt())
                {
                    std::cout << "从模型文件读取 GPT 规格 (V2 格式)\n";
                    auto build_result = nn::build_gpt_model_from_spec(file_spec);
                    if (!build_result)
                    {
                        std::cerr << "Error: " << build_result.error().message << '\n';
                        return 1;
                    }
                    model = std::move(*build_result);
                    spec = file_spec;
                }
                else
                {
                    std::cout << "旧格式模型文件，使用命令行参数\n";
                }

                auto load_result = nn::load_model(cfg.resume_path, model);
                if (!load_result)
                {
                    std::cerr << "加载模型失败: " << load_result.error().message << "，将从头训练。\n" << std::endl;
                }
                else
                {
                    std::cout << "已加载模型: " << cfg.resume_path << "\n" << std::endl;
                }
            }
        }

        // ── 优化器 ─────────────────────────────────────────────
        std::unique_ptr<nn::Optimizer> optimizer;
        if (cfg.optimizer_name == "sgd")
            optimizer = std::make_unique<nn::SGD>(model.parameters(), model.param_gradients(), cfg.lr);
        else if (cfg.optimizer_name == "sgd_w_momentum")
            optimizer = std::make_unique<nn::SGD_w_Momentum>(model.parameters(), model.param_gradients(), cfg.lr);
        else
            optimizer = std::make_unique<nn::Adam>(model.parameters(), model.param_gradients(), cfg.lr);

        nn::CrossEntropyLoss ce_loss;

        // ── 训练循环 ─────────────────────────────────────────────
        // 可用的 token 数量（需要 seq_len + 1 作为 input + target）
        const std::size_t max_start = all_tokens.size() - cfg.seq_len - 1;
        if (max_start == 0)
        {
            std::cerr << "文本太短，至少需要 " << (cfg.seq_len + 2) << " 个 token\n";
            return 1;
        }

        const std::size_t steps_per_epoch = std::min(max_start / cfg.batch_size,
                                                      std::size_t{1000});

        std::mt19937_64 rng{42};

        auto t_start = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < cfg.epochs; ++epoch)
        {
            auto ep_start = std::chrono::steady_clock::now();
            double total_loss = 0.0;

            for (std::size_t step = 0; step < steps_per_epoch; ++step)
            {
                // ── 采样 batch ───────────────────────────────────
                nn::Matrix x_tokens(cfg.seq_len, cfg.batch_size);
                nn::Matrix y_tokens(cfg.seq_len, cfg.batch_size);

                for (std::size_t b = 0; b < cfg.batch_size; ++b)
                {
                    std::uniform_int_distribution<std::size_t> dist(0, max_start);
                    std::size_t start = dist(rng);

                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                    {
                        x_tokens.set_value_unchecked(t, b,
                            static_cast<double>(all_tokens[start + t]));
                        y_tokens.set_value_unchecked(t, b,
                            static_cast<double>(all_tokens[start + t + 1]));
                    }
                }

                // ── 前向传播 ─────────────────────────────────────
                auto fwd_result = model.forward(x_tokens);
                if (!fwd_result) { std::cerr << "Error: " << fwd_result.error().message << '\n'; return 1; }
                auto logits = std::move(*fwd_result);
                // logits: (vocab_size, seq_len × batch_size)

                // ── 构造 one-hot 目标 ────────────────────────────
                // 展平 y_tokens: seq_len × batch_size 个 token ID
                const std::size_t total_tokens = cfg.seq_len * cfg.batch_size;
                std::vector<std::size_t> flat_targets(total_tokens);
                auto y_span = y_tokens.span();
                for (std::size_t t = 0; t < cfg.seq_len; ++t)
                    for (std::size_t b = 0; b < cfg.batch_size; ++b)
                        flat_targets[t * cfg.batch_size + b] =
                            static_cast<std::size_t>(y_span[t * cfg.batch_size + b]);

                auto y_onehot = one_hot_labels(flat_targets, tokenizer.vocab_size());

                // ── 损失 ─────────────────────────────────────────
                auto loss_result = ce_loss.forward(logits, y_onehot);
                if (!loss_result) { std::cerr << "Error: " << loss_result.error().message << '\n'; return 1; }
                double loss = *loss_result;
                total_loss += loss;

                // ── 反向传播 ─────────────────────────────────────
                auto grad = ce_loss.backward();
                auto bwd_result = model.backward(grad);
                if (!bwd_result) { std::cerr << "Error: " << bwd_result.error().message << '\n'; return 1; }

                {
                    auto step_result = optimizer->step();
                    if (!step_result) {
                        std::cerr << "Error: " << step_result.error().message << '\n';
                        return 1;
                    }
                }
                optimizer->zero_grad();

                // 进度显示
                if ((step + 1) % 50 == 0 || step + 1 == steps_per_epoch)
                {
                    std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                              << "  step " << step + 1 << "/" << steps_per_epoch
                              << "  loss: " << std::fixed << std::setprecision(4) << loss
                              << "   " << std::flush;
                }
            }

            auto ep_end = std::chrono::steady_clock::now();
            double ep_sec = std::chrono::duration<double>(ep_end - ep_start).count();
            double avg_loss = total_loss / steps_per_epoch;

            std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                      << "  avg_loss=" << std::fixed << std::setprecision(4) << avg_loss
                      << "  time=" << std::setprecision(1) << ep_sec << "s"
                      << std::endl;
        }

        auto t_end = std::chrono::steady_clock::now();
        double total_sec = std::chrono::duration<double>(t_end - t_start).count();

        // ── 保存模型（含规格） ─────────────────────────────────────
        {
            auto save_result = nn::save_model(cfg.save_path, model, spec);
            if (!save_result) {
                std::cerr << "Error: " << save_result.error().message << '\n';
                return 1;
            }
        }
        std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1)
                  << total_sec << "s" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
}
