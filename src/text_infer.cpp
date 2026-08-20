// ── GPT 文本生成推理程序（引擎化架构） ───────────────────────────────────
//
// 数据流：
//   加载模型 (含嵌入 tokenizer) → tokenizer.encode(prompt) → GPTModel.generate
//   → tokenizer.decode(generated) → 输出文本
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "GPT 文本生成推理程序 (引擎化架构)\n\n"
        << "用法:\n"
        << "  " << prog << " --prompt \"text\" [选项]\n"
        << "  " << prog << " --interactive          交互模式\n\n"
        << "选项:\n"
        << "  --model <path>       模型文件路径 (默认: gpt_model.bin)\n"
        << "                       V3 格式模型自动读取规格和嵌入 tokenizer\n"
        << "  --vocab <path>       词表 JSON 路径 (默认: bpe_vocab.json)\n"
        << "                       仅当模型未嵌入 tokenizer 时使用\n"
        << "                       自动识别分词器类型（bpe / charbpe / wordzip / space）\n"
        << "  --prompt <text>      输入提示文本\n"
        << "  --interactive        交互式生成模式\n"
        << "  --max-tokens <n>     最大生成 token 数 (默认: 200)\n"
        << "  --temperature <t>    温度参数 (默认: 1.0, 0=贪心)\n"
        << "  --gpu                启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --cuda               启用 CUDA GPU 加速 (需要 CUDA Toolkit)\n"
        << "  --show-tokens        显示 token ID (调试用)\n"
        << "  --help               显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct InferConfig
{
    std::string model_path = "gpt_model.bin";
    std::string vocab_path = "bpe_vocab.json";
    std::string prompt = "Hello";
    int max_tokens = 200;
    double temperature = 1.0;
    bool interactive = false;
    bool show_tokens = false;
    bool gpu_enabled = false;
    bool cuda_enabled = false;
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
        else if (arg == "--vocab" && i + 1 < argc)
            cfg.vocab_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)
            cfg.prompt = argv[++i];
        else if (arg == "--interactive")
            cfg.interactive = true;
        else if (arg == "--max-tokens" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-tokens\n"; std::exit(1); }
            cfg.max_tokens = *v;
        }
        else if (arg == "--temperature" && i + 1 < argc)
        {
            auto v = nn::parse_number<double>(argv[++i]);
            if (!v) { std::cerr << "无效 --temperature\n"; std::exit(1); }
            cfg.temperature = *v;
        }
        else if (arg == "--gpu")
            cfg.gpu_enabled = true;
        else if (arg == "--cuda")
            cfg.cuda_enabled = true;
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

// ==================== 从 Model 提取 GPTModel 并生成 ====================
nn::Result<std::vector<std::size_t>> generate_text(
    nn::Model &model, nn::ComputeEngine &engine,
    const std::vector<std::size_t> &prompt_tokens,
    std::size_t max_new_tokens, double temperature,
    std::size_t eos_token_id)
{
    auto &layer_ref = model.layer_at(0);
    auto *gpt_ptr = dynamic_cast<nn::GPTModel *>(&layer_ref);
    if (!gpt_ptr)
        return std::unexpected(nn::Error{"Model does not contain a GPTModel layer"});

    // 传入 EOS_ID，生成遇到 EOS 自动停止
    // min_new_tokens = max_new_tokens/2，至少生成一半 token 才允许 EOS 停止，
    // 避免模型因训练偏置一上来就输出 EOS 导致无输出。
    const std::size_t min_new = max_new_tokens / 2;
    return gpt_ptr->generate(engine, prompt_tokens, max_new_tokens, temperature,
                             eos_token_id, min_new);
}

// ==================== 交互模式 ====================
void interactive_mode(nn::Model &model, nn::ComputeEngine &engine,
                     const nn::Tokenizer &tokenizer, const InferConfig &cfg)
{
    const std::size_t bos_id = tokenizer.bos_id();
    const std::size_t eos_id = tokenizer.eos_id();
    const bool has_dialogue = tokenizer.has_dialogue_markers();

    if (has_dialogue)
        std::cout << "GPT 交互式对话 (输入 'quit' 退出)\n\n";
    else
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

        // 构建 prompt tokens
        std::vector<std::size_t> prompt_tokens;
        if (bos_id != nn::Tokenizer::npos)
            prompt_tokens.push_back(bos_id);

        if (has_dialogue)
        {
            // 对话模式：直接用 tokenizer 暴露的标记 ID 构建（不硬编码字符串，
            // 以后改标记只需改 domain_tokenizer.hpp）。
            // 训练格式:
            //   [BOS]<|system|>...</|end_of_system|><|user|>...</|end_of_user|><|assistant|>...
            const auto push_marker = [&](std::size_t id) {
                if (id != nn::Tokenizer::npos) prompt_tokens.push_back(id);
            };

            const std::string system_prompt = "你是一个有用的AI助手。";

            push_marker(tokenizer.system_marker_id());
            auto sys_tokens  = tokenizer.encode(system_prompt);
            prompt_tokens.insert(prompt_tokens.end(), sys_tokens.begin(), sys_tokens.end());
            push_marker(tokenizer.end_system_marker_id());

            push_marker(tokenizer.user_marker_id());
            auto user_tokens = tokenizer.encode(prompt);
            prompt_tokens.insert(prompt_tokens.end(), user_tokens.begin(), user_tokens.end());
            push_marker(tokenizer.end_user_marker_id());

            push_marker(tokenizer.assistant_marker_id());
        }
        else
        {
            // 普通模式：仅 BOS + 编码文本（与训练时每行格式一致）
            auto text_tokens = tokenizer.encode(prompt);
            prompt_tokens.insert(prompt_tokens.end(), text_tokens.begin(), text_tokens.end());
        }

        std::cout << "生成中...\n";
        auto gen_result = generate_text(model, engine, prompt_tokens,
                                       cfg.max_tokens, cfg.temperature, eos_id);
        if (!gen_result) { std::cerr << "Error: " << gen_result.error().message << '\n'; continue; }
        auto generated = std::move(*gen_result);

        // 对话模式：按 assistant 结束标记 ID 截断（避免显示在输出中）
        if (has_dialogue)
        {
            const std::size_t end_asst = tokenizer.end_assistant_marker_id();
            std::vector<std::size_t> gen_trunc = generated;
            if (end_asst != nn::Tokenizer::npos)
                for (std::size_t i = 0; i < generated.size(); ++i)
                    if (generated[i] == end_asst) { gen_trunc.resize(i); break; }
            std::cout << "\n" << tokenizer.decode(gen_trunc) << "\n\n";
        }
        else
        {
            std::cout << "\n" << tokenizer.decode(generated) << "\n\n";
        }

        if (cfg.show_tokens)
        {
            std::cout << "Prompt tokens: [";
            for (std::size_t i = 0; i < prompt_tokens.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << prompt_tokens[i];
            }
            std::cout << "]\nGenerated tokens: [";
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
    InferConfig cfg = parse_args(argc, argv);

    // ── 从模型文件读取规格 ─────────────────────────────────────
    auto spec_result = nn::peek_model_spec(cfg.model_path);
    if (!spec_result)
    {
        std::cerr << "读取模型文件失败: " << spec_result.error().message << std::endl;
        return 1;
    }
    nn::ModelSpec spec = spec_result.value();
    if (!spec.is_gpt())
    {
        std::cerr << "模型文件不是 GPT 类型 (type="
                  << static_cast<uint32_t>(spec.type) << ")\n";
        return 1;
    }

    // ── 创建计算引擎 ─────────────────────────────────────────
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = cfg.gpu_enabled;
    eng_cfg.use_cuda = cfg.cuda_enabled;
    auto engine = nn::cli::create_engine(eng_cfg, std::cout);

    // ── 构建 GPT 模型 ────────────────────────────────────────
    std::cout << "模型规格: vocab=" << spec.vocab_size
              << " d_model=" << spec.d_model
              << " heads=" << spec.num_heads
              << " layers=" << spec.num_layers
              << " d_ff=" << spec.d_ff
              << " seq_len=" << spec.seq_len;
    if (spec.is_alibi_gpt() || spec.pos_encoding == nn::PosEncodingType::ALiBi)
        std::cout << " [ALiBi]";
    else if (spec.pos_encoding == nn::PosEncodingType::Sinusoidal)
        std::cout << " [Sinusoidal]";
    else if (spec.pos_encoding == nn::PosEncodingType::RoPE)
        std::cout << " [RoPE]";
    else
        std::cout << " [Learned]";
    std::cout << "\n";

    nn::Result<nn::Model> model_result;
    // 统一的 GPTModel 通过 pos_encoding 区分 Learned/Sinusoidal/ALiBi，
    // GPT 和旧格式 ALiBi_GPT 文件都走同一条构建路径。
    model_result = nn::build_gpt_model_from_spec(*engine, spec);
    if (!model_result)
    {
        std::cerr << "构建模型失败: " << model_result.error().message << std::endl;
        return 1;
    }
    auto model = std::move(*model_result);

    // ── 加载模型参数 + tokenizer ────────────────────────────
    std::cout << "加载模型: " << cfg.model_path << " ..." << std::endl;
    auto load_result = nn::load_model(cfg.model_path, model);
    if (!load_result)
    {
        std::cerr << "加载模型失败: " << load_result.error().message << std::endl;
        return 1;
    }
    std::cout << "模型已加载" << std::endl;

    // ── 加载 tokenizer（自动识别类型） ─────────────────────────
    std::unique_ptr<nn::Tokenizer> tokenizer;
    const std::string &embedded_json = *load_result;
    if (!embedded_json.empty())
    {
        // V3 格式：模型文件嵌入了 tokenizer JSON
        tokenizer = nn::load_tokenizer_from_string(embedded_json);
        if (!tokenizer)
        {
            std::cerr << "解析嵌入 tokenizer 失败或无法识别分词器类型" << std::endl;
            return 1;
        }
        std::cout << "已从模型文件加载嵌入 tokenizer" << std::endl;
    }
    else
    {
        // 回退：从外部 JSON 文件加载
        tokenizer = nn::load_tokenizer_from_file(cfg.vocab_path);
        if (!tokenizer)
        {
            std::cerr << "加载词表失败或无法识别分词器类型: " << cfg.vocab_path
                      << "\n请使用 --vocab 指定词表路径，或使用 V3 格式模型" << std::endl;
            return 1;
        }
        std::cout << "已从外部文件加载 tokenizer: " << cfg.vocab_path << std::endl;
    }
    std::cout << "词表大小: " << tokenizer->vocab_size() << "\n" << std::endl;

    // ── 交互模式 ─────────────────────────────────────────────
    if (cfg.interactive)
    {
        interactive_mode(model, *engine, *tokenizer, cfg);
        return 0;
    }

    // ── 单次生成 ─────────────────────────────────────────────
    const std::size_t bos_id = tokenizer->bos_id();
    const std::size_t eos_id = tokenizer->eos_id();
    auto prompt_tokens = tokenizer->encode(cfg.prompt);
    // 添加 BOS 前缀，使推理输入格式与训练时一致（训练时每行以 BOS 开头）
    if (bos_id != nn::Tokenizer::npos)
        prompt_tokens.insert(prompt_tokens.begin(), bos_id);
    std::cout << "提示: \"" << cfg.prompt << "\"\n";
    std::cout << "Prompt tokens: " << prompt_tokens.size() << " 个\n";
    std::cout << "生成 " << cfg.max_tokens << " 个 token"
              << " (temperature=" << cfg.temperature << ")\n";
    std::cout << "----------------------------------------" << std::endl;

    auto t_start = std::chrono::steady_clock::now();
    auto gen_result = generate_text(model, *engine, prompt_tokens,
                                   cfg.max_tokens, cfg.temperature, eos_id);
    auto t_end = std::chrono::steady_clock::now();
    Scalar gen_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    if (!gen_result)
    {
        std::cerr << "生成失败: " << gen_result.error().message << std::endl;
        return 1;
    }
    auto generated = std::move(*gen_result);

    std::cout << cfg.prompt << tokenizer->decode(generated) << std::endl;

    if (cfg.show_tokens)
    {
        std::cout << "\n[Generated tokens: ";
        for (std::size_t i = 0; i < generated.size(); ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << generated[i];
        }
        std::cout << "]\n";
    }

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "生成 " << generated.size() << " tokens, 耗时 "
              << std::fixed << std::setprecision(1) << gen_sec << "s"
              << " (" << std::setprecision(0)
              << (gen_sec > 0 ? generated.size() / gen_sec : 0) << " tokens/s)"
              << std::endl;

    return 0;
}
