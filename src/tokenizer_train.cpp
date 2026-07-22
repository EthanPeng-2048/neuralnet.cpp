/**
 * 分词器训练程序
 *
 * 支持三种分词器：
 *   wordzip — WordZip 词级分词器（基于词频统计）
 *   space   — Space  空格分词器（空格切分 + 词表查表）
 *   bpe     — BPE    Byte-Pair Encoding（迭代字节对合并）
 *
 * 用法:
 *   tokenizer_train <text_file> [选项]
 *
 * 示例:
 *   tokenizer_train dataset.txt --type bpe --vocab-size 5000 --output bpe.json
 *   tokenizer_train dataset.txt --type wordzip --vocab-size 20000
 *   tokenizer_train dataset.txt --type space --vocab-size 10000
 */

#include <neuralnet.cpp/domain_tokenizer.hpp>
#include <neuralnet.cpp/core_file.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

// ── 帮助信息 ────────────────────────────────────────────────────────────
void print_usage(const char *prog)
{
    std::cout
        << "分词器训练程序 v3.0\n\n"
        << "用法: " << prog << " <text-file> [选项]\n\n"
        << "参数:\n"
        << "  <text-file>              训练文本文件路径 (必需)\n\n"
        << "选项:\n"
        << "  --type <name>            分词器类型 (默认: wordzip)\n"
        << "                             wordzip - WordZip 词级分词器\n"
        << "                             space   - 空格分词器\n"
        << "                             bpe     - Byte-Pair Encoding\n"
        << "  --vocab-size <n>         目标词表大小\n"
        << "                             (wordzip 默认: 20000, space 默认: 10000, bpe 默认: 5000)\n"
        << "  --output <path>          输出 JSON 路径 (默认: tokenizer.json)\n"
        << "  --help                   显示此帮助信息\n";
}

// ── 命令行参数 ──────────────────────────────────────────────────────────
enum class TokenizerType { WordZip, Space, BPE };

struct TrainArgs
{
    std::string    text_file;
    std::string    output     = "tokenizer.json";
    TokenizerType  type       = TokenizerType::WordZip;
    std::size_t    vocab_size = 0;  // 0 = use type default
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
        else if (arg == "--type" && i + 1 < argc)
        {
            std::string t = argv[++i];
            if (t == "wordzip")      args.type = TokenizerType::WordZip;
            else if (t == "space")   args.type = TokenizerType::Space;
            else if (t == "bpe")     args.type = TokenizerType::BPE;
            else {
                std::cerr << "未知分词器类型: " << t << "\n可选: wordzip, space, bpe\n";
                std::exit(1);
            }
        }
        else if (arg == "--vocab-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --vocab-size: " << v.error().message << "\n"; std::exit(1); }
            args.vocab_size = *v;
        }
        else if (arg == "--output" && i + 1 < argc)
            args.output = argv[++i];
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

// ── 公共验证函数 ────────────────────────────────────────────────────────
template<typename T>
void verify_tokenizer(const T &tokenizer, const std::string &test_sent)
{
    auto ids     = tokenizer.encode(test_sent);
    auto decoded = tokenizer.decode(ids);

    std::cout << "\n🧪 验证:\n";
    std::cout << "  原文: " << test_sent << "\n";
    std::cout << "  Token数: " << ids.size() << "\n";
    std::cout << "  解码: " << decoded << "\n";
    std::cout << "  完美还原: " << (decoded == test_sent ? "true" : "false") << "\n";
}

// ── WordZip 训练 ────────────────────────────────────────────────────────
void train_wordzip(const std::string &text, const TrainArgs &args)
{
    nn::WordZipTokenizer tokenizer;
    nn::WordZipTokenizer::Config config;
    config.vocab_size = args.vocab_size > 0 ? args.vocab_size : nn::WordZipTokenizer::DEFAULT_VOCAB_SIZE;

    std::cout << "分词器类型: WordZip (词级分词)\n";
    std::cout << "目标词表大小: " << config.vocab_size << "\n";

    auto t0 = std::chrono::steady_clock::now();
    auto train_result = tokenizer.train(text, config);
    auto t1 = std::chrono::steady_clock::now();

    if (!train_result) {
        std::cerr << "训练失败: " << train_result.error().message << '\n';
        return;
    }

    std::cout << "总耗时: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << " 秒\n";

    // 展示前 20 个高频词
    std::cout << "\n✅ 高频词展示（前 20 个）:\n";
    std::size_t shown = 0;
    const auto &vocab = tokenizer.vocab();
    for (std::size_t tid = nn::WordZipTokenizer::BYTE_OFFSET;
         tid < vocab.size() && shown < 20; ++tid)
    {
        const auto &bytes = vocab[tid];
        if (bytes.size() >= 2)
        {
            bool printable = true;
            for (unsigned char c : bytes)
                if (c < 32 && c != '\n' && c != '\t') { printable = false; break; }
            if (printable)
            {
                std::cout << "  ID " << tid << ": \"" << tokenizer.try_decode_token(tid) << "\"\n";
                ++shown;
            }
        }
    }

    verify_tokenizer(tokenizer, "Alice was a very good girl, she said hello!");

    tokenizer.print_affix_report();

    if (auto save_result = tokenizer.save(args.output); !save_result)
    {
        std::cerr << "保存失败: " << save_result.error().message << '\n';
        std::exit(1);
    }
    std::cout << "词表已保存至: " << args.output
              << " (" << tokenizer.vocab_size() << " tokens)\n";
}

// ── Space 训练 ──────────────────────────────────────────────────────────
void train_space(const std::string &text, const TrainArgs &args)
{
    nn::SpaceTokenizer tokenizer;
    nn::SpaceTokenizer::Config config;
    config.vocab_size = args.vocab_size > 0 ? args.vocab_size : nn::SpaceTokenizer::DEFAULT_VOCAB_SIZE;

    std::cout << "分词器类型: Space (空格分词)\n";
    std::cout << "目标词表大小: " << config.vocab_size << "\n";

    auto t0 = std::chrono::steady_clock::now();
    auto train_result = tokenizer.train(text, config);
    auto t1 = std::chrono::steady_clock::now();

    if (!train_result) {
        std::cerr << "训练失败: " << train_result.error().message << '\n';
        return;
    }

    std::cout << "总耗时: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << " 秒\n";

    // 展示前 20 个高频词
    std::cout << "\n✅ 高频词展示（前 20 个）:\n";
    std::size_t shown = 0;
    const auto &vocab = tokenizer.vocab();
    for (std::size_t tid = nn::SpaceTokenizer::ASCII_BASE;
         tid < vocab.size() && shown < 20; ++tid)
    {
        const auto &word = vocab[tid];
        if (word.size() >= 2)
        {
            std::cout << "  ID " << tid << ": \"" << word << "\"\n";
            ++shown;
        }
    }

    verify_tokenizer(tokenizer, "Alice was a very good girl, she said hello!");

    if (auto save_result = tokenizer.save(args.output); !save_result)
    {
        std::cerr << "保存失败: " << save_result.error().message << '\n';
        std::exit(1);
    }
    std::cout << "词表已保存至: " << args.output
              << " (" << tokenizer.vocab_size() << " tokens)\n";
}

// ── BPE 训练 ────────────────────────────────────────────────────────────
void train_bpe(const std::string &text, const TrainArgs &args)
{
    nn::BPETokenizer tokenizer;
    nn::BPETokenizer::Config config;
    config.vocab_size = args.vocab_size > 0 ? args.vocab_size : nn::BPETokenizer::DEFAULT_VOCAB_SIZE;

    std::cout << "分词器类型: BPE (Byte-Pair Encoding)\n";
    std::cout << "目标词表大小: " << config.vocab_size << "\n";

    auto t0 = std::chrono::steady_clock::now();
    auto train_result = tokenizer.train(text, config);
    auto t1 = std::chrono::steady_clock::now();

    if (!train_result) {
        std::cerr << "训练失败: " << train_result.error().message << '\n';
        return;
    }

    std::cout << "总耗时: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double>(t1 - t0).count() << " 秒\n";

    // 展示前 20 个 token
    std::cout << "\n✅ Token 展示（前 20 个）:\n";
    std::size_t shown = 0;
    const auto &vocab = tokenizer.vocab();
    for (std::size_t tid = 0; tid < vocab.size() && shown < 20; ++tid)
    {
        const auto &bytes = vocab[tid];
        if (bytes.size() >= 2)
        {
            bool printable = true;
            for (unsigned char c : bytes)
                if (c < 32 && c != '\n' && c != '\t') { printable = false; break; }
            if (printable)
            {
                std::cout << "  ID " << tid << ": \"" << bytes << "\"\n";
                ++shown;
            }
        }
    }

    std::cout << "\n合并规则数: " << tokenizer.merge_count() << "\n";

    verify_tokenizer(tokenizer, "Alice was a very good girl, she said hello!");

    if (auto save_result = tokenizer.save(args.output); !save_result)
    {
        std::cerr << "保存失败: " << save_result.error().message << '\n';
        std::exit(1);
    }
    std::cout << "词表已保存至: " << args.output
              << " (" << tokenizer.vocab_size() << " tokens)\n";
}

// ── 主函数 ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    auto args = parse_args(argc, argv);

    // 读取文件
    std::cout << "读取文件: " << args.text_file << "\n";
    auto text_result = nn::load_text_file(args.text_file);
    if (!text_result)
    {
        std::cerr << "错误: " << text_result.error().message << "\n";
        return 1;
    }
    auto text = std::move(*text_result);
    std::cout << "文本字符数: " << text.size() << "\n";

    switch (args.type)
    {
        case TokenizerType::WordZip: train_wordzip(text, args); break;
        case TokenizerType::Space:   train_space(text, args);   break;
        case TokenizerType::BPE:     train_bpe(text, args);     break;
    }

    return 0;
}
