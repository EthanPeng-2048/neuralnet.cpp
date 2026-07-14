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
// 使用 GPTModel 内置的 generate 方法
std::vector<std::size_t> generate_text(
    nn::Model &model, const std::vector<std::size_t> &prompt_tokens,
    std::size_t max_new_tokens, double temperature)
{
    // 获取 GPTModel 指针
    // 由于 Model 封装了 unique_ptr<Layer>，我们需要通过 forward + 手动采样来实现
    // 这里直接使用逐 token 生成的方式

    std::vector<std::size_t> context(prompt_tokens);
    std::vector<std::size_t> generated;

    // 获取 seq_len（从模型参数推断）
    // GPTModel 的 forward 期望输入 (seq_len, batch)，这里 batch=1
    // 我们需要知道 seq_len，但 Model 接口不暴露这个信息
    // 所以我们用一个足够大的 seq_len 来处理

    for (int step = 0; step < static_cast<int>(max_new_tokens); ++step)
    {
        // 取最后 256 个 token 作为上下文（默认 seq_len）
        std::size_t ctx_len = std::min(context.size(), std::size_t{256});
        std::size_t start = context.size() - ctx_len;

        nn::Matrix input(ctx_len, 1);
        for (std::size_t t = 0; t < ctx_len; ++t)
            input.set_value_unchecked(t, 0,
                static_cast<double>(context[start + t]));

        auto logits = model.forward(input); // (vocab_size, ctx_len)

        // 取最后一个位置的 logits
        std::size_t vocab_size = logits.rows();
        std::vector<double> last_logits(vocab_size);
        for (std::size_t v = 0; v < vocab_size; ++v)
            last_logits[v] = logits.at_unchecked(v, ctx_len - 1);

        // temperature
        if (temperature != 1.0 && temperature > 0.0)
        {
            for (auto &v : last_logits)
                v /= temperature;
        }

        // softmax
        double max_val = last_logits[0];
        for (std::size_t v = 1; v < vocab_size; ++v)
            max_val = std::max(max_val, last_logits[v]);
        double sum_exp = 0.0;
        for (auto &v : last_logits)
        {
            v = std::exp(v - max_val);
            sum_exp += v;
        }
        for (auto &v : last_logits)
            v /= sum_exp;

        // 采样：使用温度 < 1 时的 softmax 概率采样
        std::size_t next_token;
        if (temperature > 0.0 && temperature != 1.0)
        {
            // 概率采样
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double r = dist(rng);
            double cumulative = 0.0;
            next_token = vocab_size - 1;
            for (std::size_t v = 0; v < vocab_size; ++v)
            {
                cumulative += last_logits[v];
                if (r <= cumulative)
                {
                    next_token = v;
                    break;
                }
            }
        }
        else
        {
            // 贪心
            next_token = 0;
            double best = last_logits[0];
            for (std::size_t v = 1; v < vocab_size; ++v)
            {
                if (last_logits[v] > best)
                {
                    best = last_logits[v];
                    next_token = v;
                }
            }
        }

        context.push_back(next_token);
        generated.push_back(next_token);
    }

    return generated;
}

// ==================== 交互模式 ====================
void interactive_mode(nn::Model &model, const InferConfig &cfg)
{
    nn::CharTokenizer tokenizer;

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
                                       cfg.max_tokens, cfg.temperature);

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
        nn::CharTokenizer tokenizer;

        // ── 加载模型 ─────────────────────────────────────────────
        std::cout << "加载模型: " << cfg.model_path << " ..." << std::endl;
        auto model = nn::build_gpt_model();
        nn::load_model(cfg.model_path, model);
        std::cout << "模型已加载\n" << std::endl;

        if (cfg.interactive)
        {
            interactive_mode(model, cfg);
            return 0;
        }

        // ── 单次生成 ─────────────────────────────────────────────
        auto prompt_tokens = tokenizer.encode(cfg.prompt);

        std::cout << "提示: \"" << cfg.prompt << "\"\n";
        std::cout << "生成 " << cfg.max_tokens << " 个 token"
                  << " (temperature=" << cfg.temperature << ")\n";
        std::cout << "----------------------------------------\n";

        auto generated = generate_text(model, prompt_tokens,
                                       cfg.max_tokens, cfg.temperature);

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
