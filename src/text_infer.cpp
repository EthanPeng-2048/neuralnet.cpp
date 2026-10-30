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
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>
#include <neuralnet.cpp/cli/cli_help.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    nn::cli::Help help(std::cout, prog, "GPT 文本生成推理程序");

    help.usage("--prompt \"text\" [选项]", "单次生成");
    help.usage("--interactive [选项]", "交互模式");
    help.usage("--turn user:\"hi\" --turn assistant:\"hello\" --turn user:\"bye\" [选项]",
               "多轮对话（历史由调用方自带，一次一问）");

    help.section("选项");
    help.opt("--model <path>", "模型文件路径 (默认: gpt_model.bin)\nV3 格式模型自动读取规格和嵌入 tokenizer");
    help.opt("--vocab <path>", "词表 .nnvocab 路径 (默认: bpe_vocab.nnvocab)\n仅当模型未嵌入 tokenizer 时使用；自动识别分词器类型 (bpe / charbpe)");
    help.opt("--prompt <text>", "输入提示文本");
    help.opt("--turn <role>:<text>", "追加一轮对话历史；role = user|assistant，可重复。\n给了 --turn 即进入多轮对话模式：按词表的对话标记拼 prompt\n（末条必须是 user），并从最旧的一问一答开始截断到模型 seq_len");
    help.opt("--reply-out <path>", "多轮模式：把本轮回复（纯文本）另写一份到该文件\n（GUI 靠它取回复，无需解析日志）");
    help.opt("--system <text>", "对话模式的系统提示（默认: 你是一个有用的AI助手。）");
    help.opt("--interactive", "交互式生成模式（多轮，上下文累积保留）");
    help.opt("--max-tokens <n>", "最大生成 token 数 (默认: 200)");
    help.opt("--temperature <t>", "温度参数 (默认: 1.0；0=贪心)");
    help.opt("--top-k <n>", "top-k 截断：只保留概率最高的 n 个候选 (默认: 0=关闭)");
    help.opt("--top-p <f>", "top-p 核采样阈值 (0<f<=1；默认: 1.0=关闭)");
    help.opt("--repeat-penalty <f>", "重复惩罚 (>1 抑制重复；默认: 1.0=关闭)");
    help.opt("--repeat-window <n>", "重复惩罚只看最近 n 个 token (默认: 0=全部上下文)");
    help.opt("--seed <n>", "随机种子 (给了就固定种子 ⇒ 同 prompt 输出可复现)");
    help.opt("--show-tokens", "显示 token ID (调试用)");
    help.opt("--gpu [索引|名称]", "启用 GPU 加速 (需要 Vulkan SDK)\n空格形式只收枚举索引；名称子串用 --gpu=<名称>，如 --gpu=NVIDIA / --gpu=40HX");
    help.opt("--help, -h", "显示此帮助信息");
}

// ==================== 命令行参数 ====================
struct InferConfig
{
    std::string model_path = "gpt_model.bin";
    std::string vocab_path = "bpe_vocab.nnvocab";
    std::string prompt = "Hello";
    std::string system_prompt = "你是一个有用的AI助手。";
    std::string reply_out;                       // --reply-out（多轮模式）
    std::vector<nn::Tokenizer::ChatTurn> turns;  // --turn（多轮模式，按序累积）
    int max_tokens = 200;
    double temperature = 1.0;
    std::size_t top_k = 0;
    double top_p = 1.0;
    double repeat_penalty = 1.0;
    std::size_t repeat_window = 0;
    std::uint64_t seed = 0;
    bool has_seed = false;
    bool prompt_given = false;   // --prompt / 位置参数显式出现
    bool interactive = false;
    bool show_tokens = false;
    bool gpu_enabled = false;
    std::string gpu_device;      // --gpu 的可选设备选择子（空 = 自动选卡）
};

InferConfig parse_args(int argc, char *argv[])
{
    InferConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--model" && i + 1 < argc)
            cfg.model_path = argv[++i];
        else if (arg == "--vocab" && i + 1 < argc)
            cfg.vocab_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc)
        {
            cfg.prompt = argv[++i];
            cfg.prompt_given = true;
        }
        else if (arg == "--system" && i + 1 < argc)
            cfg.system_prompt = argv[++i];
        else if (arg == "--reply-out" && i + 1 < argc)
            cfg.reply_out = argv[++i];
        else if (arg == "--turn" && i + 1 < argc)
        {
            // <role>:<text>：取**第一个**冒号分段（正文里的冒号不转义）
            const std::string val = argv[++i];
            const auto sep = val.find(':');
            const std::string role = (sep == std::string::npos) ? val : val.substr(0, sep);
            if (sep == std::string::npos || (role != "user" && role != "assistant"))
            {
                std::cerr << "--turn 需要 <role>:<text> 形式，role = user|assistant\n"
                          << "  收到: " << val << "\n";
                std::exit(1);
            }
            nn::Tokenizer::ChatTurn t;
            t.user = (role == "user");
            t.text = val.substr(sep + 1);
            cfg.turns.push_back(std::move(t));
        }
        else if (arg == "--interactive")
            cfg.interactive = true;
        else if (arg == "--max-tokens" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-tokens\n"; std::exit(1); }
            if (*v < 1) { std::cerr << "--max-tokens 必须 >= 1\n"; std::exit(1); }
            cfg.max_tokens = *v;
        }
        else if (arg == "--temperature" && i + 1 < argc)
        {
            auto v = nn::parse_number<double>(argv[++i]);
            if (!v) { std::cerr << "无效 --temperature\n"; std::exit(1); }
            cfg.temperature = *v;
        }
        else if (arg == "--top-k" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v || *v < 0)
            {
                std::cerr << "--top-k 必须 >= 0\n";
                std::exit(1);
            }
            cfg.top_k = static_cast<std::size_t>(*v);
        }
        else if (arg == "--top-p" && i + 1 < argc)
        {
            auto v = nn::parse_number<double>(argv[++i]);
            if (!v || *v <= 0.0 || *v > 1.0)
            {
                std::cerr << "--top-p 必须在 (0, 1] 区间\n";
                std::exit(1);
            }
            cfg.top_p = *v;
        }
        else if (arg == "--repeat-penalty" && i + 1 < argc)
        {
            auto v = nn::parse_number<double>(argv[++i]);
            if (!v || *v <= 0.0)
            {
                std::cerr << "--repeat-penalty 必须 > 0（1.0 = 关闭）\n";
                std::exit(1);
            }
            cfg.repeat_penalty = *v;
        }
        else if (arg == "--repeat-window" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v || *v < 0)
            {
                std::cerr << "--repeat-window 必须 >= 0（0 = 全部上下文）\n";
                std::exit(1);
            }
            cfg.repeat_window = static_cast<std::size_t>(*v);
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            auto v = nn::parse_number<long long>(argv[++i]);
            if (!v || *v < 0)
            {
                std::cerr << "--seed 必须 >= 0\n";
                std::exit(1);
            }
            cfg.seed = static_cast<std::uint64_t>(*v);
            cfg.has_seed = true;
        }
        else if (auto gpu_dev = nn::cli::parse_gpu_option(argc, argv, i))
        {
            cfg.gpu_enabled = true;
            if (!gpu_dev->empty()) cfg.gpu_device = *gpu_dev;
        }
        else if (arg == "--show-tokens")
            cfg.show_tokens = true;
        else if (!arg.starts_with("--"))
        {
            cfg.prompt = arg;
            cfg.prompt_given = true;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    // ── 多轮模式（--turn）的自洽性校验（不合法即报错，不静默降级）──────
    if (!cfg.turns.empty())
    {
        if (cfg.prompt_given)
        {
            std::cerr << "--turn 与 --prompt 互斥：多轮模式下末条 --turn user:<text> "
                         "就是要回答的那一轮\n";
            std::exit(1);
        }
        if (cfg.interactive)
        {
            std::cerr << "--turn 与 --interactive 互斥（交互模式自己在进程内累积历史）\n";
            std::exit(1);
        }
        if (!cfg.turns.front().user)
        {
            std::cerr << "--turn 第一条必须是 user:<text>（历史以用户轮起头）\n";
            std::exit(1);
        }
        for (std::size_t i = 0; i < cfg.turns.size(); ++i)
        {
            const bool expect_user = (i % 2 == 0);
            if (cfg.turns[i].user != expect_user)
            {
                std::cerr << "--turn 角色必须 user/assistant 交替（第 " << (i + 1)
                          << " 条应为 " << (expect_user ? "user" : "assistant") << "）\n";
                std::exit(1);
            }
        }
        if (!cfg.turns.back().user)
        {
            std::cerr << "--turn 最后一条必须是 user:<text>（模型回答的就是这一轮）\n";
            std::exit(1);
        }
    }
    else if (!cfg.reply_out.empty())
    {
        std::cerr << "--reply-out 只在多轮模式（--turn）下有效\n";
        std::exit(1);
    }
    return cfg;
}

// ==================== 从 Model 提取 GPTModel 并生成 ====================
// 把 CLI 字段打包成库侧采样配置（P0-5）
nn::SamplingConfig sampling_config(const InferConfig& cfg)
{
    nn::SamplingConfig sc;
    sc.temperature = static_cast<Scalar>(cfg.temperature);
    sc.top_k = cfg.top_k;
    sc.top_p = static_cast<Scalar>(cfg.top_p);
    sc.repeat_penalty = static_cast<Scalar>(cfg.repeat_penalty);
    sc.repeat_window = cfg.repeat_window;
    sc.seed = cfg.seed;
    sc.has_seed = cfg.has_seed;
    return sc;
}

nn::Result<std::vector<std::size_t>> generate_text(nn::Model& model, nn::ComputeEngine& engine,
                                                   const std::vector<std::size_t>& prompt_tokens,
                                                   std::size_t max_new_tokens,
                                                   const nn::SamplingConfig& sc,
                                                   std::size_t eos_token_id)
{
    auto &layer_ref = model.layer_at(0);

    // 传入 EOS_ID，生成遇到 EOS 自动停止
    // min_new_tokens = max_new_tokens/2，至少生成一半 token 才允许 EOS 停止，
    // 避免模型因训练偏置一上来就输出 EOS 导致无输出。
    const std::size_t min_new = max_new_tokens / 2;
    if (auto *gpt_ptr = dynamic_cast<nn::GPTModel *>(&layer_ref))
        return gpt_ptr->generate(engine, prompt_tokens, max_new_tokens, sc, eos_token_id, min_new);
    if (auto *rapt_ptr = dynamic_cast<nn::RAPTModel *>(&layer_ref))
        return rapt_ptr->generate(engine, prompt_tokens, max_new_tokens, sc, eos_token_id, min_new);
    NN_FAIL("Model does not contain a GPTModel or RAPTModel layer");
}

// ==================== 交互模式 ====================
// 多轮（P0-5 尾项）：每轮把**本轮及之前的所有轮次**拼进 prompt（`chat_prompt`），
// 因此连续 N 轮不丢上下文；超 `seq_len` 时从最旧的一问一答开始整轮丢弃。
// 助手轮用**生成的原 token** 回灌（不经 decode → encode 往返）。
void interactive_mode(nn::Model &model, nn::ComputeEngine &engine,
                     const nn::Tokenizer &tokenizer, const InferConfig &cfg,
                     std::size_t seq_len)
{
    const std::size_t eos_id = tokenizer.eos_id();
    const bool has_dialogue = tokenizer.has_dialogue_markers();
    std::vector<nn::Tokenizer::ChatTurn> history;   // 交替：user → assistant → …

    if (has_dialogue)
        std::cout << "GPT 交互式对话（多轮，上下文保留；输入 'quit' 退出）\n\n";
    else
        std::cout << "GPT 交互式生成（多轮，上下文保留；输入 'quit' 退出）\n\n";

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

        // 本轮用户输入入列 → 整个历史（含之前 N-1 轮）一起进 prompt
        history.push_back(nn::Tokenizer::ChatTurn{true, prompt, {}});

        std::size_t dropped = 0;
        auto prompt_tokens = tokenizer.chat_prompt(history, cfg.system_prompt, seq_len,
                                                  &dropped);
        if (dropped > 0)
        {
            // 历史按"一问一答"整轮丢弃（丢弃后仍以 user 起头）
            history.erase(history.begin(),
                          history.begin() + static_cast<std::ptrdiff_t>(dropped));
            std::cout << "[上下文] 已达 seq_len=" << seq_len << " 上限：丢弃最旧 "
                      << dropped << " 轮，当前保留 " << history.size() << " 轮\n";
        }

        std::cout << "生成中...\n";
        auto gen_result = generate_text(model, engine, prompt_tokens, cfg.max_tokens,
                                        sampling_config(cfg), eos_id);
        if (!gen_result)
        {
            std::cerr << "Error: " << gen_result.error().message << '\n';
            history.pop_back();   // 失败的这一轮不进历史（下次重来）
            continue;
        }
        auto generated = std::move(*gen_result);

        // 对话模式：按 assistant 结束标记 ID 截断（不显示标记本身）
        std::vector<std::size_t> shown = generated;
        if (has_dialogue)
        {
            const std::size_t end_asst = tokenizer.end_assistant_marker_id();
            if (end_asst != nn::Tokenizer::npos)
                for (std::size_t i = 0; i < shown.size(); ++i)
                    if (shown[i] == end_asst) { shown.resize(i); break; }
        }
        std::cout << "\n" << tokenizer.decode(shown) << "\n\n";

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

        // 助手轮入列（用生成的原 token，避免 decode → encode 往返漂移）
        history.push_back(nn::Tokenizer::ChatTurn{false, {}, std::move(shown)});
    }
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    InferConfig cfg = parse_args(argc, argv);

    // ── 从模型文件读取规格（v5/v6/v7 同一 magic，自动按版本分流）──────────
    auto spec_result = nn::peek_any_model_spec(cfg.model_path);
    NN_EXIT(spec_result, 1, "读取模型文件失败: ");
    nn::ModelSpec spec = spec_result.value();
    auto role_r = nn::peek_model_role(cfg.model_path);
    NN_EXIT(role_r, 1, "读取模型文件失败: ");
    if (!spec.is_gpt() && !spec.is_rapt())
    {
        std::cerr << "模型文件不是 GPT/RAPT 类型 (type="
                  << spec.type_name() << ")\n";
        return 1;
    }

    // GPT/RAPT 共享的规格视图（两者类型参数同名，按实际类型取值）
    std::size_t v_vocab = 0, v_d_model = 0, v_heads = 0, v_layers = 0, v_d_ff = 0,
                v_seq_len = 0;
    nn::PosEncodingType v_pos = nn::PosEncodingType::Learned;
    if (const auto *g = spec.as<nn::GptSpec>())
    {
        v_vocab = g->vocab_size; v_d_model = g->d_model; v_heads = g->num_heads;
        v_layers = g->num_layers; v_d_ff = g->d_ff; v_seq_len = g->seq_len;
        v_pos = g->pos_encoding;
    }
    else if (const auto *r = spec.as<nn::RaptSpec>())
    {
        v_vocab = r->vocab_size; v_d_model = r->d_model; v_heads = r->num_heads;
        v_layers = r->num_layers; v_d_ff = r->d_ff; v_seq_len = r->seq_len;
        v_pos = r->pos_encoding;
    }

    // ── 创建计算引擎 ─────────────────────────────────────────
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = cfg.gpu_enabled;
    eng_cfg.gpu_device = cfg.gpu_device;
    auto engine_res = nn::cli::create_engine(eng_cfg, std::cout);
    NN_EXIT(engine_res, 1, "引擎创建失败: ");
    auto engine = std::move(*engine_res);

    // ── 构建 GPT 模型 ────────────────────────────────────────
    std::cout << "模型规格: vocab=" << v_vocab
              << " d_model=" << v_d_model
              << " heads=" << v_heads
              << " layers=" << v_layers
              << " d_ff=" << v_d_ff
              << " seq_len=" << v_seq_len;
    if (spec.is_rapt())
    {
        std::cout << " [RAPT (ReLU 线性注意力)]";
    }
    if (spec.is_alibi_gpt() || v_pos == nn::PosEncodingType::ALiBi)
        std::cout << " [ALiBi]";
    else if (v_pos == nn::PosEncodingType::Sinusoidal)
        std::cout << " [Sinusoidal]";
    else if (v_pos == nn::PosEncodingType::RoPE)
        std::cout << " [RoPE]";
    else
        std::cout << " [Learned]";
    std::cout << "\n";

    nn::Result<nn::Model> model_result;
    if (spec.is_rapt())
        model_result = nn::build_rapt_model_from_spec(*engine, spec);
    else  // 统一的 GPTModel 通过 pos_encoding 区分 Learned/Sinusoidal/ALiBi
        model_result = nn::build_gpt_model_from_spec(*engine, spec);
    NN_EXIT(model_result, 1, "构建模型失败: ");
    auto model = std::move(*model_result);

    // ── 加载模型参数 + tokenizer（.nnmodel 的打包三值层 → 只读推理态）──────
    std::cout << "加载模型: " << cfg.model_path << " ..." << std::endl;
    std::uint32_t model_version = 0;
    std::string embedded_vocab;
    {
        auto load_result = nn::load_model(cfg.model_path, model, &model_version);
        NN_EXIT(load_result, 1, "加载模型失败: ");
        embedded_vocab = std::move(*load_result);
        std::cout << "模型已加载（角色: " << nn::model_role_name(*role_r) << "）"
                  << std::endl;
    }

    // ── 加载 tokenizer（按模型版本分派：v5=JSON / v6=.nnvocab kvrec）──
    std::unique_ptr<nn::Tokenizer> tokenizer;
    if (!embedded_vocab.empty())
    {
        auto tok_r = nn::load_tokenizer_from_model_blob(embedded_vocab, model_version);
        NN_EXIT(tok_r, 1, "解析嵌入 tokenizer 失败: ");
        tokenizer = std::move(*tok_r);
        std::cout << "已从模型文件加载嵌入 tokenizer" << std::endl;
    }
    else
    {
        // 回退：从外部 .nnvocab 词表文件加载
        auto tok_r = nn::load_tokenizer_from_file(cfg.vocab_path);
        if (!tok_r)
        {
            std::cerr << "加载词表失败: " << cfg.vocab_path << '\n'
                      << tok_r.error().message
                      << "\n请使用 --vocab 指定 .nnvocab 词表路径，"
                         "或使用嵌入 tokenizer 的模型" << std::endl;
            return 1;
        }
        tokenizer = std::move(*tok_r);
        std::cout << "已从外部文件加载 tokenizer: " << cfg.vocab_path << std::endl;
    }
    std::cout << "词表大小: " << tokenizer->vocab_size() << "\n" << std::endl;

    const std::size_t seq_len = v_seq_len;
    const std::size_t eos_id = tokenizer->eos_id();

    // ── 交互模式 ─────────────────────────────────────────────
    if (cfg.interactive)
    {
        interactive_mode(model, *engine, *tokenizer, cfg, seq_len);
        return 0;
    }

    // ── 多轮对话（--turn）：历史由调用方（GUI / 脚本）自带 ───────────
    // 每轮一次进程：无跨轮状态 ⇒ 回复经 --reply-out 落文件交回调用方。
    if (!cfg.turns.empty())
    {
        std::size_t dropped = 0;
        auto prompt_tokens = tokenizer->chat_prompt(cfg.turns, cfg.system_prompt, seq_len,
                                                   &dropped);
        std::cout << "多轮对话: " << cfg.turns.size() << " 轮 ("
                  << (tokenizer->has_dialogue_markers() ? "对话标记模板" : "纯文本拼接") << ")"
                  << " → 进入上下文 " << (cfg.turns.size() - dropped) << " 轮";
        if (dropped > 0)
            std::cout << "（seq_len=" << seq_len << " 上限，丢弃最旧 " << dropped << " 轮）";
        std::cout << "\n本轮输入: \"" << cfg.turns.back().text << "\"\n";
        std::cout << "Prompt tokens: " << prompt_tokens.size() << " 个\n";
        std::cout << "生成 " << cfg.max_tokens << " 个 token"
                  << " (temperature=" << cfg.temperature << ", top-k=" << cfg.top_k
                  << ", top-p=" << cfg.top_p << ", repeat-penalty=" << cfg.repeat_penalty;
        if (cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0)
            std::cout << " (window=" << cfg.repeat_window << ")";
        if (cfg.has_seed) std::cout << ", seed=" << cfg.seed;
        std::cout << ")\n";
        std::cout << "----------------------------------------" << std::endl;

        auto t_start = std::chrono::steady_clock::now();
        auto gen_result = generate_text(model, *engine, prompt_tokens, cfg.max_tokens,
                                        sampling_config(cfg), eos_id);
        auto t_end = std::chrono::steady_clock::now();
        Scalar gen_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

        NN_EXIT(gen_result, 1, "生成失败: ");
        auto generated = std::move(*gen_result);

        // 对话模板：去掉末尾的 assistant 结束标记（不把标记写进回复）
        std::vector<std::size_t> shown = generated;
        if (tokenizer->has_dialogue_markers())
        {
            const std::size_t end_asst = tokenizer->end_assistant_marker_id();
            if (end_asst != nn::Tokenizer::npos)
                for (std::size_t i = 0; i < shown.size(); ++i)
                    if (shown[i] == end_asst) { shown.resize(i); break; }
        }
        const std::string reply = tokenizer->decode(shown);
        std::cout << reply << std::endl;

        if (cfg.show_tokens)
        {
            std::cout << "\n[Prompt tokens: ";
            for (std::size_t i = 0; i < prompt_tokens.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << prompt_tokens[i];
            }
            std::cout << "]\n[Generated tokens: ";
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

        // GUI 取回复的唯一通道（日志可能被其他行插花，不做解析）
        if (!cfg.reply_out.empty())
        {
            std::ofstream ofs(cfg.reply_out, std::ios::binary | std::ios::trunc);
            if (!ofs)
            {
                std::cerr << "无法写入 --reply-out 文件: " << cfg.reply_out << '\n';
                return 1;
            }
            ofs << reply << '\n';
            if (!ofs)
            {
                std::cerr << "写入 --reply-out 文件失败: " << cfg.reply_out << '\n';
                return 1;
            }
        }
        return 0;
    }

    // ── 单次生成 ─────────────────────────────────────────────
    const std::size_t bos_id = tokenizer->bos_id();
    auto prompt_tokens = tokenizer->encode(cfg.prompt);
    // 添加 BOS 前缀：tokenizer 提供 bos_id 时，用作 prompt 的起始标记
    if (bos_id != nn::Tokenizer::npos)
        prompt_tokens.insert(prompt_tokens.begin(), bos_id);
    std::cout << "提示: \"" << cfg.prompt << "\"\n";
    std::cout << "Prompt tokens: " << prompt_tokens.size() << " 个\n";
    std::cout << "生成 " << cfg.max_tokens << " 个 token"
              << " (temperature=" << cfg.temperature << ", top-k=" << cfg.top_k
              << ", top-p=" << cfg.top_p << ", repeat-penalty=" << cfg.repeat_penalty;
    if (cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0)
        std::cout << " (window=" << cfg.repeat_window << ")";
    if (cfg.has_seed) std::cout << ", seed=" << cfg.seed;
    std::cout << ")\n";
    std::cout << "----------------------------------------" << std::endl;

    auto t_start = std::chrono::steady_clock::now();
    auto gen_result =
        generate_text(model, *engine, prompt_tokens, cfg.max_tokens, sampling_config(cfg), eos_id);
    auto t_end = std::chrono::steady_clock::now();
    Scalar gen_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    NN_EXIT(gen_result, 1, "生成失败: ");
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
