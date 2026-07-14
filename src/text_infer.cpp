#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_io.hpp>
#include <neuralnet.cpp/gpt_common.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "GPT 文本生成推理程序\n\n"
        << "用法:\n"
        << "  " << prog << " --prompt <text> [选项]\n"
        << "  " << prog << " --interactive          交互模式\n\n"
        << "选项:\n"
        << "  --model <path>     模型文件路径 (默认: gpt_model.bin)\n"
        << "  --prompt <text>    输入提示文本\n"
        << "  --interactive      交互式生成模式\n"
        << "  --max-tokens <n>   最大生成 token 数 (默认: 200)\n"
        << "  --temperature <t>  温度参数 (默认: 1.0)\n"
        << "  --d-model <n>      模型维度 (默认: 128)\n"
        << "  --num-heads <n>    注意力头数 (默认: 4)\n"
        << "  --num-layers <n>   Transformer 层数 (默认: 4)\n"
        << "  --d-ff <n>         FFN 维度 (默认: 512)\n"
        << "  --seq-len <n>      序列长度 (默认: 256)\n"
        << "  --show-tokens      显示 token ID (调试用)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct InferConfig
{
    std::string model_path = "gpt_model.bin";
    std::string prompt = "Hello";
    int max_tokens = 200;
    double temperature = 1.0;
    bool interactive = false;
    bool show_tokens = false;
    std::size_t d_model = nn::GPT_D_MODEL;
    std::size_t num_heads = nn::GPT_NUM_HEADS;
    std::size_t num_layers = nn::GPT_NUM_LAYERS;
    std::size_t d_ff = nn::GPT_D_FF;
    std::size_t seq_len = nn::GPT_SEQ_LEN;
};

InferConfig parse_args(int argc, char *argv[])
{
    InferConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--model" && i + 1 < argc)
            cfg.model_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)
            cfg.prompt = argv[++i];
        else if (arg == "--interactive")
            cfg.interactive = true;
        else if (arg == "--max-tokens" && i + 1 < argc)
            cfg.max_tokens = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc)
            cfg.temperature = std::stod(argv[++i]);
        else if (arg == "--show-tokens")
            cfg.show_tokens = true;
        else if (arg == "--d-model" && i + 1 < argc)
            cfg.d_model = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--num-heads" && i + 1 < argc)
            cfg.num_heads = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--num-layers" && i + 1 < argc)
            cfg.num_layers = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--d-ff" && i + 1 < argc)
            cfg.d_ff = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (arg == "--seq-len" && i + 1 < argc)
            cfg.seq_len = static_cast<std::size_t>(std::stoi(argv[++i]));
        else if (!arg.starts_with("--"))
            cfg.prompt = arg;
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ==================== 从 GPTModel 提取采样结果 ====================
// 使用 GPTModel 内置的 generate 方法（支持温度采样 + 贪心）
std::vector<std::size_t> generate_text(
    nn::Model &model, const std::vector<std::size_t> &prompt_tokens,
    std::size_t max_new_tokens, double temperature,
    std::size_t /*seq_len — 由模型内部管理*/)
{
    // Model 容器唯一层即为 GPTModel，直接调用其 generate()
    auto &gpt = dynamic_cast<nn::GPTModel &>(model.layer_at(0));
    return gpt.generate(prompt_tokens, max_new_tokens, temperature);
}

// ==================== 交互模式 ====================
void interactive_mode(nn::Model &model, const nn::WordTokenizer &tokenizer,
                     const InferConfig &cfg)
{
    std::cout << "GPT 交互式生成 (输入 'quit' 退出)\n\n";

    while (true)
    {
        std::cout << ">>> ";
        std::string prompt;
        if (!std::getline(std::cin, prompt))
            break;

        if (prompt == "quit" || prompt == "exit")
            break;

        if (prompt.empty())
            prompt = " ";

        auto prompt_tokens = tokenizer.encode(prompt);

        std::cout << "生成中...\n";
        auto generated = generate_text(model, prompt_tokens,
                                       cfg.max_tokens, cfg.temperature,
                                       cfg.seq_len);

        std::cout << "\n" << prompt << tokenizer.decode(generated) << "\n\n";

        if (cfg.show_tokens)
        {
            std::cout << "Tokens: [";
            for (std::size_t i = 0; i < prompt_tokens.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << prompt_tokens[i];
            }
            std::cout << " -> ";
            for (std::size_t i = 0; i < generated.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << generated[i];
            }
            std::cout << "]\n\n";
        }
    }
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    try
    {
        InferConfig cfg = parse_args(argc, argv);

        // ── 加载分词器与模型 ─────────────────────────────────────
        nn::WordTokenizer tokenizer;
        tokenizer.load_vocab("gpt_vocab.txt");
        std::cout << "词表: " << tokenizer.vocab_size() << " 词" << std::endl;

        std::cout << "加载模型: " << cfg.model_path << " ..." << std::endl;
        auto model = nn::build_gpt_model(
            tokenizer.vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers);
        nn::load_model(cfg.model_path, model);
        std::cout << "模型已加载\n" << std::endl;

        if (cfg.interactive)
        {
            interactive_mode(model, tokenizer, cfg);
            return 0;
        }

        // ── 单次生成 ─────────────────────────────────────────────
        auto prompt_tokens = tokenizer.encode(cfg.prompt);

        std::cout << "提示: \"" << cfg.prompt << "\"\n";
        std::cout << "生成 " << cfg.max_tokens << " 个 token"
                  << " (temperature=" << cfg.temperature << ")\n";
        std::cout << "----------------------------------------\n";

        auto generated = generate_text(model, prompt_tokens,
                                       cfg.max_tokens, cfg.temperature,
                                       cfg.seq_len);

        std::cout << cfg.prompt << tokenizer.decode(generated) << std::endl;

        if (cfg.show_tokens)
        {
            std::cout << "\n[Tokens: ";
            for (std::size_t i = 0; i < generated.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << generated[i];
            }
            std::cout << "]\n";
        }

        std::cout << "\n----------------------------------------" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
