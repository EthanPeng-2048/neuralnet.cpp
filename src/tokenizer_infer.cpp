// ── BPE 分词器推理程序（编码/解码） ──────────────────────────────────────
//
// 加载词表 JSON → 编码文本为 token IDs / 解码 token IDs 为文本
//
// 自动根据 JSON 的 "type" 字段识别分词器类型：
//   - bpe_tokenizer      → 字节级 BPE（BBPE）
//   - char_bpe_tokenizer  → 字符级 BPE（支持中文）
//   - wordzip_tokenizer   → WordZip 词级分词器
//   - space_tokenizer     → 空白分词器
//
// 支持交互模式和命令行模式。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "BPE 分词器推理程序 (编码/解码)\n\n"
        << "用法:\n"
        << "  " << prog << " --vocab <path> --encode \"text\"    编码文本\n"
        << "  " << prog << " --vocab <path> --decode \"ids\"      解码 token IDs\n"
        << "  " << prog << " --vocab <path> --interactive         交互模式\n"
        << "  " << prog << " --vocab <path> --encode-file <file>  编码文件\n\n"
        << "选项:\n"
        << "  --vocab <path>       词表 JSON 路径 (默认: bpe_vocab.json)\n"
        << "                       自动识别分词器类型（bpe / charbpe / wordzip / space）\n"
        << "  --encode <text>      编码文本为 token IDs\n"
        << "  --decode <ids>       解码 token IDs (逗号分隔) 为文本\n"
        << "  --encode-file <path> 编码整个文件\n"
        << "  --interactive        交互模式 (输入 'quit' 退出)\n"
        << "  --show-bytes         显示原始字节 (调试用)\n"
        << "  --help               显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct Config
{
    std::string vocab_path = "bpe_vocab.json";
    std::string encode_text;
    std::string decode_ids;
    std::string encode_file;
    bool interactive = false;
    bool show_bytes = false;
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
        else if (arg == "--vocab" && i + 1 < argc)
            cfg.vocab_path = argv[++i];
        else if (arg == "--encode" && i + 1 < argc)
            cfg.encode_text = argv[++i];
        else if (arg == "--decode" && i + 1 < argc)
            cfg.decode_ids = argv[++i];
        else if (arg == "--encode-file" && i + 1 < argc)
            cfg.encode_file = argv[++i];
        else if (arg == "--interactive")
            cfg.interactive = true;
        else if (arg == "--show-bytes")
            cfg.show_bytes = true;
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ==================== 解析 token IDs 字符串 ====================
std::vector<std::size_t> parse_ids(const std::string &s)
{
    std::vector<std::size_t> ids;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        // 去除空白
        while (!token.empty() && (token.front() == ' ' || token.front() == '['))
            token.erase(0, 1);
        while (!token.empty() && (token.back() == ' ' || token.back() == ']'))
            token.pop_back();
        if (token.empty()) continue;
        auto v = nn::parse_number<std::size_t>(token);
        if (v) ids.push_back(*v);
    }
    return ids;
}

// ==================== 交互模式 ====================
void interactive_mode(const nn::Tokenizer &tokenizer, bool show_bytes)
{
    std::cout << "Tokenizer 交互模式 (输入 'quit' 退出)\n";
    std::cout << "  encode <text>  → 编码文本\n";
    std::cout << "  decode <ids>   → 解码 token IDs (逗号分隔)\n\n";

    while (true)
    {
        std::cout << ">>> ";
        std::string line;
        if (!std::getline(std::cin, line))
            break;
        if (line == "quit" || line == "exit")
            break;
        if (line.empty())
            continue;

        if (line.starts_with("encode "))
        {
            std::string text = line.substr(7);
            auto ids = tokenizer.encode(text);
            std::cout << "  Tokens (" << ids.size() << "): [";
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << ids[i];
            }
            std::cout << "]\n";
            if (show_bytes)
            {
                const auto &v = tokenizer.vocab();
                std::cout << "  Bytes: ";
                for (auto id : ids)
                {
                    if (id < v.size())
                    {
                        std::cout << "{";
                        for (unsigned char b : v[id])
                            std::cout << std::hex << static_cast<unsigned>(b) << " ";
                        std::cout << std::dec << "}";
                    }
                }
                std::cout << "\n";
            }
        }
        else if (line.starts_with("decode "))
        {
            auto ids = parse_ids(line.substr(7));
            auto text = tokenizer.decode(ids);
            std::cout << "  Text: \"" << text << "\"\n";
        }
        else
        {
            // 默认编码
            auto ids = tokenizer.encode(line);
            std::cout << "  Tokens (" << ids.size() << "): [";
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                if (i > 0) std::cout << ", ";
                std::cout << ids[i];
            }
            std::cout << "]\n";
            auto decoded = tokenizer.decode(ids);
            std::cout << "  Decode: \"" << decoded << "\"\n";
        }
    }
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    Config cfg = parse_args(argc, argv);

    // ── 加载词表（自动识别分词器类型） ─────────────────────
    auto tokenizer = nn::load_tokenizer_from_file(cfg.vocab_path);
    if (!tokenizer)
    {
        std::cerr << "加载词表失败或无法识别分词器类型: " << cfg.vocab_path << '\n'
                  << "请检查 JSON 文件是否包含有效的 \"type\" 字段" << std::endl;
        return 1;
    }
    std::cout << "词表已加载: " << tokenizer->vocab_size() << " 词" << std::endl;

    // ── 交互模式 ─────────────────────────────────────────────
    if (cfg.interactive)
    {
        interactive_mode(*tokenizer, cfg.show_bytes);
        return 0;
    }

    // ── 编码文本 ─────────────────────────────────────────────
    if (!cfg.encode_text.empty())
    {
        auto ids = tokenizer->encode(cfg.encode_text);
        std::cout << "Tokens (" << ids.size() << "): [";
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]\n";
        auto decoded = tokenizer->decode(ids);
        std::cout << "Decode: \"" << decoded << "\"\n";
        return 0;
    }

    // ── 解码 IDs ─────────────────────────────────────────────
    if (!cfg.decode_ids.empty())
    {
        auto ids = parse_ids(cfg.decode_ids);
        auto text = tokenizer->decode(ids);
        std::cout << "Text: \"" << text << "\"\n";
        return 0;
    }

    // ── 编码文件 ─────────────────────────────────────────────
    if (!cfg.encode_file.empty())
    {
        auto text_result = nn::load_text_file(cfg.encode_file);
        if (!text_result)
        {
            std::cerr << "读取文件失败: " << text_result.error().message << '\n';
            return 1;
        }
        auto ids = tokenizer->encode(*text_result);
        std::cout << "文件大小: " << text_result->size() << " 字节\n";
        std::cout << "Token 数: " << ids.size() << "\n";
        std::cout << "压缩率: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(text_result->size()) / ids.size()
                  << " bytes/token\n";
        return 0;
    }

    // 无参数 → 交互模式
    interactive_mode(*tokenizer, cfg.show_bytes);
    return 0;
}
