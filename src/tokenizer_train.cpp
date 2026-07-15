/**
 * ByteZip 分词器训练程序
 *
 * 用法:
 *   tokenizer_train <text_file> [选项]
 *
 * 示例:
 *   tokenizer_train dataset.txt --vocab-size 10000 --output tokenizer.json
 */

#include <neuralnet.cpp/tokenizer.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

// ── 帮助信息 ────────────────────────────────────────────────────────────
void print_usage(const char *prog)
{
    std::cout
        << "ByteZip v2.2 分词器训练程序\n\n"
        << "用法: " << prog << " <text-file> [选项]\n\n"
        << "参数:\n"
        << "  <text-file>              训练文本文件路径 (必需)\n\n"
        << "选项:\n"
        << "  --vocab-size <n>         目标词表大小 (默认: 20000)\n"
        << "  --output <path>          输出 JSON 路径 (默认: tokenizer.json)\n"
        << "  --v1-len <n>             V1 最大子词长度 (默认: 16)\n"
        << "  --v2-len <n>             V2 最大子词长度 (默认: 24)\n"
        << "  --v2-reserve <n>         V2 预留槽位 (默认: 0=自动)\n"
        << "  --skip-ratio <f>         独立度阈值 α (默认: 1.2)\n"
        << "  --affix-ratio <f>        词缀保护阈值 (默认: 0.4)\n"
        << "  --help                   显示此帮助信息\n";
}

// ── 命令行参数 ──────────────────────────────────────────────────────────
struct TrainArgs
{
    std::string text_file;
    std::string output = "tokenizer.json";
    std::size_t vocab_size = nn::ByteZipTokenizer::DEFAULT_VOCAB_SIZE;
    std::size_t v1_len = nn::ByteZipTokenizer::DEFAULT_V1_MAX_LEN;
    std::size_t v2_len = nn::ByteZipTokenizer::DEFAULT_V2_MAX_LEN;
    std::size_t v2_reserve = 0;
    double skip_ratio = nn::ByteZipTokenizer::DEFAULT_SKIP_RATIO;
    double affix_ratio = nn::ByteZipTokenizer::DEFAULT_AFFIX_RATIO;
};

TrainArgs parse_args(int argc, char *argv[])
{
    TrainArgs args;

    if (argc < 2)
    {
        print_usage(argv[0]);
        std::exit(1);
    }

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--vocab-size" && i + 1 < argc)
            args.vocab_size = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (arg == "--output" && i + 1 < argc)
            args.output = argv[++i];
        else if (arg == "--v1-len" && i + 1 < argc)
            args.v1_len = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (arg == "--v2-len" && i + 1 < argc)
            args.v2_len = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (arg == "--v2-reserve" && i + 1 < argc)
            args.v2_reserve = static_cast<std::size_t>(std::stoul(argv[++i]));
        else if (arg == "--skip-ratio" && i + 1 < argc)
            args.skip_ratio = std::stod(argv[++i]);
        else if (arg == "--affix-ratio" && i + 1 < argc)
            args.affix_ratio = std::stod(argv[++i]);
        else if (!arg.starts_with("--"))
            args.text_file = arg;
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    if (args.text_file.empty())
    {
        std::cerr << "错误：必须指定训练文本文件\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    return args;
}

// ── 读取文本文件（UTF-8） ──────────────────────────────────────────────
std::string read_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "错误：无法打开文件 " << path << "\n";
        std::exit(1);
    }

    // 获取文件大小
    ifs.seekg(0, std::ios::end);
    const auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::string content(static_cast<std::size_t>(size), '\0');
    ifs.read(content.data(), size);

    return content;
}

// ── 主函数 ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    auto args = parse_args(argc, argv);

    // 读取文件
    std::cout << "读取文件: " << args.text_file << "\n";
    auto text = read_text_file(args.text_file);
    std::cout << "文本字符数: " << text.size() << "\n";

    // 训练
    nn::ByteZipTokenizer tokenizer;
    nn::ByteZipTokenizer::Config config;
    config.vocab_size          = args.vocab_size;
    config.v1_max_len          = args.v1_len;
    config.v2_max_len          = args.v2_len;
    config.v2_reserve          = args.v2_reserve;
    config.skip_ratio          = args.skip_ratio;
    config.affix_protect_ratio = args.affix_ratio;

    auto t0 = std::chrono::steady_clock::now();
    tokenizer.train(text, config);
    auto t1 = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "总耗时: " << std::fixed << std::setprecision(1)
              << elapsed << " 秒\n";

    // 展示前 20 个高频子词
    std::cout << "\n✅ 高频子词展示（前 20 个）:\n";
    std::size_t shown = 0;
    const auto &vocab = tokenizer.vocab();
    for (std::size_t tid = nn::ByteZipTokenizer::BYTE_OFFSET;
         tid < vocab.size() && shown < 20; ++tid)
    {
        const auto &bytes = vocab[tid];
        if (bytes.size() >= 2)
        {
            // 尝试作为 UTF-8 解码显示
            bool printable = true;
            for (unsigned char c : bytes)
            {
                if (c < 32 && c != '\n' && c != '\t')
                {
                    printable = false;
                    break;
                }
            }

            if (printable)
            {
                std::cout << "  ID " << tid << ": \""
                          << tokenizer.try_decode_token(tid) << "\"\n";
                ++shown;
            }
        }
    }

    // 编码/解码冒烟测试
    const std::string test_sent = "Alice was a very good girl, she said hello!";
    auto ids = tokenizer.encode(test_sent);
    auto decoded = tokenizer.decode(ids);

    std::cout << "\n🧪 验证:\n";
    std::cout << "  原文: " << test_sent << "\n";
    std::cout << "  Token数: " << ids.size() << "\n";
    std::cout << "  解码: " << decoded << "\n";
    std::cout << "  完美还原: " << (decoded == test_sent ? "true" : "false") << "\n";

    // 保存词表
    if (auto save_result = tokenizer.save(args.output); !save_result)
    {
        std::cerr << "保存失败: " << save_result.error().message << '\n';
        return 1;
    }
    std::cout << "词表已保存至: " << args.output
              << " (" << tokenizer.vocab_size() << " tokens)\n";

    return 0;
}
