// ── BPE 分词器训练程序 ──────────────────────────────────────────────────
//
// 读取文本文件 → 训练分词器 → 保存词表 JSON
// 词表 JSON 可被 text_train / text_infer 加载。
//
// 支持两种 BPE：
//   --tokenizer bpe      字节级 BPE（BBPE），天然兼容 UTF-8，但中文按字节切分
//   --tokenizer charbpe  字符级 BPE（纯 BPE），中文每个汉字为 1 token，
//                        BPE 能学会合并常见中文词组（更适合小模型）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "BPE 分词器训练程序\n\n"
        << "用法: " << prog << " <text-file> [选项]\n\n"
        << "参数:\n"
        << "  <text-file>          训练文本文件路径 (UTF-8 编码)\n\n"
        << "选项:\n"
        << "  --tokenizer <type>   分词器类型: bpe (字节级 BBPE, 默认) | charbpe (字符级, 支持中文)\n"
        << "  --output <path>      词表输出路径 (默认: bpe_vocab.json)\n"
        << "  --vocab-size <n>     目标词表大小 (默认: 5000)\n"
        << "  --min-freq <n>       最小合并频率 (默认: 2)\n"
        << "  --threads <n>        预分词并行线程数 (默认: 0=自动使用全部核心, 1=顺序)\n"
        << "  --help               显示此帮助信息\n\n"
        << "说明:\n"
        << "  bpe     字节级 BPE（BBPE）：兼容性最好，任意字节均可编码；\n"
        << "          但中文按 3 字节切分，小模型难学到词义。\n"
        << "  charbpe 字符级 BPE：每个汉字为 1 个基础 token，BPE 学会合并常见词组；\n"
        << "          中文场景推荐使用。\n";
}

// ==================== 命令行参数 ====================
struct Config
{
    std::string text_path;
    std::string output_path = "bpe_vocab.json";
    std::string tokenizer_type = "bpe";  // "bpe" 或 "charbpe"
    std::size_t vocab_size = nn::BPETokenizer::DEFAULT_VOCAB_SIZE;
    std::uint32_t min_freq = nn::BPETokenizer::DEFAULT_MIN_FREQ;
    std::uint32_t threads = 0;   // 预分词并行: 0=自动, 1=顺序, >1=指定
};

Config parse_args(int argc, char *argv[])
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--tokenizer" && i + 1 < argc)
        {
            cfg.tokenizer_type = argv[++i];
            if (cfg.tokenizer_type != "bpe" && cfg.tokenizer_type != "charbpe")
            {
                std::cerr << "未知分词器类型: " << cfg.tokenizer_type
                          << "，可选: bpe, charbpe\n";
                std::exit(1);
            }
        }
        else if (arg == "--output" && i + 1 < argc)
            cfg.output_path = argv[++i];
        else if (arg == "--vocab-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --vocab-size: " << v.error().message << "\n"; std::exit(1); }
            if (*v < 258)
            {
                // BPE 词表 = 256 字节 + BOS/EOS；更小值会在
                // merges_.reserve(vocab_size - 258) 无符号下溢 → abort
                std::cerr << "--vocab-size 必须 >= 258（256 字节 + BOS/EOS）\n";
                std::exit(1);
            }
            cfg.vocab_size = *v;
        }
        else if (arg == "--min-freq" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::uint32_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --min-freq: " << v.error().message << "\n"; std::exit(1); }
            cfg.min_freq = *v;
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::uint32_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --threads: " << v.error().message << "\n"; std::exit(1); }
            cfg.threads = *v;
        }
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

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    Config cfg = parse_args(argc, argv);

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
    std::cout << "文本大小: " << text.size() << " 字节" << std::endl;

    // ── 创建分词器实例（根据 --tokenizer 选择类型） ────────────
    std::unique_ptr<nn::Tokenizer> tokenizer;
    if (cfg.tokenizer_type == "charbpe")
    {
        tokenizer = std::make_unique<nn::CharBPETokenizer>();
        std::cout << "使用字符级 BPE (CharBPE)，支持中文分词" << std::endl;
    }
    else
    {
        tokenizer = std::make_unique<nn::BPETokenizer>();
        std::cout << "使用字节级 BPE (BBPE)" << std::endl;
    }

    // ── 训练 ──────────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "  " << (cfg.tokenizer_type == "charbpe" ? "CharBPE" : "BPE")
              << " 分词器训练\n";
    std::cout << "========================================\n";
    std::cout << "  目标词表: " << cfg.vocab_size << "\n";
    std::cout << "  最小频率: " << cfg.min_freq << "\n";
    std::cout << "  预分词线程: " << cfg.threads
              << (cfg.threads == 0 ? " (自动)" : "") << "\n";
    std::cout << "========================================\n" << std::endl;

    // 根据分词器类型调用对应训练接口（BPE 和 CharBPE 接口签名一致）
    // 使用 static_cast 向下转型以调用各子类特有的 train(text, Config) 重载，
    // 避免在基类 Tokenizer 中暴露子类专属的 Config 参数。
    auto t_start = std::chrono::steady_clock::now();
    nn::Result<void> train_result;
    if (cfg.tokenizer_type == "charbpe")
    {
        auto& tok = static_cast<nn::CharBPETokenizer&>(*tokenizer);
        nn::CharBPETokenizer::Config tcfg;
        tcfg.vocab_size = cfg.vocab_size;
        tcfg.min_freq = cfg.min_freq;
        tcfg.threads = cfg.threads;
        tcfg.show_progress = true;
        tcfg.log = [](std::string_view msg) { std::cout << msg << '\n'; };
        train_result = tok.train(text, tcfg);
    }
    else
    {
        auto& tok = static_cast<nn::BPETokenizer&>(*tokenizer);
        nn::BPETokenizer::Config tcfg;
        tcfg.vocab_size = cfg.vocab_size;
        tcfg.min_freq = cfg.min_freq;
        tcfg.threads = cfg.threads;
        tcfg.show_progress = true;
        tcfg.log = [](std::string_view msg) { std::cout << msg << '\n'; };
        train_result = tok.train(text, tcfg);
    }

    if (!train_result)
    {
        std::cerr << "训练失败: " << train_result.error().message << '\n';
        return 1;
    }
    auto t_end = std::chrono::steady_clock::now();
    Scalar train_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    std::cout << "\n训练完成! 耗时: " << std::fixed << std::setprecision(1)
              << train_sec << "s" << std::endl;
    std::cout << "词表大小: " << tokenizer->vocab_size() << std::endl;

    // ── 保存词表 ─────────────────────────────────────────────
    auto save_result = tokenizer->save(cfg.output_path);
    if (!save_result)
    {
        std::cerr << "保存词表失败: " << save_result.error().message << '\n';
        return 1;
    }
    std::cout << "词表已保存: " << cfg.output_path << std::endl;

    // ── 打印示例（验证分词效果） ─────────────────────────────
    {
        // 取文本第一行作为示例（避免截断 UTF-8 序列）
        std::string sample;
        {
            std::size_t nl = text.find('\n');
            sample = nl == std::string::npos ? text : text.substr(0, nl);
        }
        auto ids = tokenizer->encode(sample);
        std::cout << "\n示例编码 (" << sample.size() << " 字节 → "
                  << ids.size() << " tokens, 压缩率 "
                  << std::fixed << std::setprecision(2)
                  << static_cast<double>(sample.size()) / std::max<std::size_t>(ids.size(), 1)
                  << " bytes/token):\n";
        std::cout << "  原文: \"" << sample << "\"\n";
        std::cout << "  Tokens: [";
        for (std::size_t i = 0; i < ids.size() && i < 30; ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        if (ids.size() > 30) std::cout << ", ...";
        std::cout << "]\n";
        auto decoded = tokenizer->decode(ids);
        std::cout << "  解码: \"" << decoded << "\"\n";
    }

    return 0;
}
