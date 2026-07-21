/**
 * 分词器推理程序
 *
 * 支持 WordZip / Space / BPE 三种分词器，根据 JSON 文件中的 "type" 字段自动选择。
 *
 * 用法:
 *   tokenizer_infer --model <json> --text <string>      编码+解码
 *   tokenizer_infer --model <json> --encode <string>     仅编码
 *   tokenizer_infer --model <json> --decode <id,id,...>   仅解码
 *   tokenizer_infer --model <json> --benchmark <file>    性能测试
 *   tokenizer_infer --model <json> --interactive         交互模式
 */

#include <neuralnet.cpp/domain_tokenizer.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// ── 帮助信息 ────────────────────────────────────────────────────────────
void print_usage(const char *prog)
{
    std::cout
        << "分词器推理程序 (WordZip / Space / BPE 自动识别)\n\n"
        << "用法:\n"
        << "  " << prog << " --model <json> --text <string>        编码并解码\n"
        << "  " << prog << " --model <json> --encode <string>      仅编码\n"
        << "  " << prog << " --model <json> --decode <id,id,...>    仅解码\n"
        << "  " << prog << " --model <json> --benchmark <file>     性能测试\n"
        << "  " << prog << " --model <json> --interactive           交互模式\n\n"
        << "选项:\n"
        << "  --model <path>       词表 JSON 路径 (必需)\n"
        << "  --text <string>      输入文本（编码+解码并验证）\n"
        << "  --encode <string>    仅编码\n"
        << "  --decode <ids>       仅解码（逗号分隔的 ID 列表）\n"
        << "  --benchmark <file>   对文件进行编码性能测试\n"
        << "  --interactive        交互式编码/解码\n"
        << "  --show-tokens        显示每个 token 的详细信息\n"
        << "  --help               显示此帮助信息\n";
}

// ── 命令行参数 ──────────────────────────────────────────────────────────
enum class Mode
{
    Text,        // 编码 + 解码验证
    Encode,      // 仅编码
    Decode,      // 仅解码
    Benchmark,   // 性能测试
    Interactive  // 交互模式
};

struct InferArgs
{
    std::string model_path;
    std::string input;
    Mode mode = Mode::Text;
    bool show_tokens = false;
};

InferArgs parse_args(int argc, char *argv[])
{
    InferArgs args;

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
        else if (arg == "--model" && i + 1 < argc)
        {
            args.model_path = argv[++i];
        }
        else if (arg == "--text" && i + 1 < argc)
        {
            args.input = argv[++i];
            args.mode = Mode::Text;
        }
        else if (arg == "--encode" && i + 1 < argc)
        {
            args.input = argv[++i];
            args.mode = Mode::Encode;
        }
        else if (arg == "--decode" && i + 1 < argc)
        {
            args.input = argv[++i];
            args.mode = Mode::Decode;
        }
        else if (arg == "--benchmark" && i + 1 < argc)
        {
            args.input = argv[++i];
            args.mode = Mode::Benchmark;
        }
        else if (arg == "--interactive")
        {
            args.mode = Mode::Interactive;
        }
        else if (arg == "--show-tokens")
        {
            args.show_tokens = true;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    if (args.model_path.empty())
    {
        std::cerr << "错误：必须指定 --model 路径\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    return args;
}

// ── 解析逗号分隔的 ID 列表 ─────────────────────────────────────────────
// 修复：原代码使用 std::stoul，违反"禁止异常"规范。改用 nn::parse_number。
nn::Result<std::vector<std::size_t>> parse_ids(const std::string &s)
{
    std::vector<std::size_t> ids;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ','))
    {
        // 去除空白
        auto start = token.find_first_not_of(" \t");
        if (start == std::string::npos)
            continue;
        auto v = nn::parse_number<std::size_t>(
            std::string_view{token}.substr(start));
        if (!v)
            return std::unexpected(std::move(v.error()));
        ids.push_back(*v);
    }
    return ids;
}

// ── 读取文本文件 ────────────────────────────────────────────────────────
// 修复：原代码 ifs.tellg() 失败时返回 -1，转为 size_t 后变为巨大值，
// 随后 std::string content(size, '\0') 会触发 std::bad_alloc。
// 改为使用 istreambuf_iterator 读取，避免 tellg 风险。
nn::Result<std::string> read_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return std::unexpected(nn::Error{"无法打开文件: " + path});
    return std::string{std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>()};
}

// ── 通用 token 调试输出（替代原 WordZipTokenizer::try_decode_token） ──
// 通过 Tokenizer 基类的 vocab() 虚函数获取，对三种分词器都适用。
std::string format_token(const nn::Tokenizer &tokenizer, std::size_t id)
{
    const auto &vocab = tokenizer.vocab();
    if (id >= vocab.size()) return "<invalid>";
    std::string r;
    for (char c : vocab[id])
    {
        if      (c >= 32 && c < 127) r += c;
        else if (c == '\n')          r += "\\n";
        else if (c == '\t')          r += "\\t";
        else if (c == ' ')           r += "␣";
        else { r += "\\x"; r += "0123456789abcdef"[(static_cast<unsigned char>(c)>>4)&0xF];
                              r += "0123456789abcdef"[ static_cast<unsigned char>(c)   &0xF]; }
    }
    return r;
}

// ── 展示编码结果 ────────────────────────────────────────────────────────
void show_encoding(const nn::Tokenizer &tokenizer,
                   const std::string &text,
                   const std::vector<std::size_t> &ids,
                   bool show_detail)
{
    std::cout << "  原文: " << text << "\n";
    std::cout << "  Token数: " << ids.size() << "\n";

    if (show_detail)
    {
        std::cout << "  Token序列: [";
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]\n";

        std::cout << "  Token详情:\n";
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            std::cout << "    [" << ids[i] << "] \""
                      << format_token(tokenizer, ids[i]) << "\"\n";
        }
    }
}

// ── 模式：编码+解码验证 ─────────────────────────────────────────────────
void mode_text(const nn::Tokenizer &tokenizer,
               const std::string &text, bool show_detail)
{
    auto ids = tokenizer.encode(text);
    auto decoded = tokenizer.decode(ids);

    show_encoding(tokenizer, text, ids, show_detail);

    std::cout << "  解码: " << decoded << "\n";
    std::cout << "  完美还原: " << (decoded == text ? "true" : "false") << "\n";
}

// ── 模式：仅编码 ────────────────────────────────────────────────────────
void mode_encode(const nn::Tokenizer &tokenizer,
                 const std::string &text, bool show_detail)
{
    auto ids = tokenizer.encode(text);
    show_encoding(tokenizer, text, ids, show_detail);

    std::cout << "  IDs: [";
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0)
            std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]\n";
}

// ── 模式：仅解码 ────────────────────────────────────────────────────────
void mode_decode(const nn::Tokenizer &tokenizer,
                 const std::string &id_str, bool show_detail)
{
    auto ids_result = parse_ids(id_str);
    if (!ids_result)
    {
        std::cerr << "解析 ID 列表失败: " << ids_result.error().message << "\n";
        return;
    }
    const auto &ids = *ids_result;

    if (show_detail)
    {
        std::cout << "  输入 IDs: [";
        for (std::size_t i = 0; i < ids.size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]\n";
    }

    auto decoded = tokenizer.decode(ids);
    std::cout << "  解码: " << decoded << "\n";
}

// ── 模式：性能测试 ──────────────────────────────────────────────────────
void mode_benchmark(const nn::Tokenizer &tokenizer,
                    const std::string &file_path, bool show_detail)
{
    std::cout << "读取文件: " << file_path << "\n";
    auto text_result = read_text_file(file_path);
    if (!text_result)
    {
        std::cerr << "错误: " << text_result.error().message << "\n";
        return;
    }
    const auto &text = *text_result;
    std::cout << "文件大小: " << text.size() << " 字节\n";

    // 预热
    std::cout << "预热中...\n";
    {
        auto warmup = text.substr(0, std::min(text.size(), static_cast<std::size_t>(1000)));
        (void)tokenizer.encode(warmup);
    }

    // 正式测试
    const int rounds = 5;
    std::cout << "编码测试 (" << rounds << " 轮)...\n";

    std::vector<std::size_t> ids;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i)
    {
        ids = tokenizer.encode(text);
    }
    auto t1 = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    const auto avg_ms = elapsed / rounds * 1000.0;
    const auto throughput = static_cast<double>(text.size() * rounds) / elapsed / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n📊 性能结果:\n";
    std::cout << "  总耗时: " << elapsed << " 秒\n";
    std::cout << "  平均: " << avg_ms << " 毫秒/轮\n";
    std::cout << "  吞吐量: " << throughput << " MB/s\n";
    std::cout << "  Token数: " << ids.size() << "\n";
    std::cout << "  压缩率: " << std::setprecision(1)
              << (static_cast<double>(text.size()) / static_cast<double>(ids.size()))
              << " 字节/token\n";

    if (show_detail)
    {
        // 解码验证
        auto t2 = std::chrono::steady_clock::now();
        auto decoded = tokenizer.decode(ids);
        auto t3 = std::chrono::steady_clock::now();

        std::cout << "  解码耗时: " << std::setprecision(2)
                  << std::chrono::duration<double>(t3 - t2).count() * 1000.0
                  << " 毫秒\n";
        std::cout << "  完美还原: " << (decoded == text ? "true" : "false") << "\n";
    }
}

// ── 模式：交互 ──────────────────────────────────────────────────────────
void mode_interactive(const nn::Tokenizer &tokenizer, bool show_detail)
{
    std::cout << "分词器交互模式（输入 q 退出）\n\n";

    std::string line;
    while (true)
    {
        std::cout << "> ";
        if (!std::getline(std::cin, line))
            break;
        if (line == "q" || line == "quit" || line == "exit")
            break;
        if (line.empty())
            continue;

        auto ids = tokenizer.encode(line);
        auto decoded = tokenizer.decode(ids);

        std::cout << "  Token数: " << ids.size() << "\n";
        if (show_detail)
        {
            std::cout << "  IDs: [";
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                if (i > 0)
                    std::cout << ", ";
                std::cout << ids[i];
            }
            std::cout << "]\n";
        }
        std::cout << "  解码: " << decoded << "\n";
        std::cout << "  还原: " << (decoded == line ? "✓" : "✗") << "\n\n";
    }
}

// ── 分词器工厂：根据 JSON 文件的 "type" 字段构造对应的分词器 ──────────
// 修复：原代码硬编码 nn::WordZipTokenizer，无法加载 Space/BPE 词表。
// 现读取文件头部 "type" 字段后选择对应类型构造。
nn::Result<std::unique_ptr<nn::Tokenizer>> load_tokenizer_by_type(const std::string &path)
{
    // 先读 JSON 内容以检测 "type" 字段
    std::ifstream ifs(path);
    if (!ifs)
        return std::unexpected(nn::Error{"无法打开文件: " + path});
    std::string content{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};

    // 检测 "type" 字段
    auto detect_type = [&]() -> std::string {
        auto p = content.find("\"type\"");
        if (p == std::string::npos) return "wordzip"; // 兼容旧文件
        auto c = content.find(':', p);
        if (c == std::string::npos) return "wordzip";
        ++c;
        while (c < content.size() && (content[c] == ' ' || content[c] == '\t')) ++c;
        if (c >= content.size() || content[c] != '"') return "wordzip";
        ++c;
        std::string t;
        while (c < content.size() && content[c] != '"') t += content[c++];
        return t;
    };

    const auto type = detect_type();
    std::unique_ptr<nn::Tokenizer> tk;

    if (type == "space_tokenizer" || type == "space")
        tk = std::make_unique<nn::SpaceTokenizer>();
    else if (type == "bpe_tokenizer" || type == "bpe")
        tk = std::make_unique<nn::BPETokenizer>();
    else
        tk = std::make_unique<nn::WordZipTokenizer>(); // 默认/兼容

    auto load_result = tk->load(path);
    if (!load_result)
        return std::unexpected(std::move(load_result.error()));

    return tk;
}

// ── 主函数 ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    auto args = parse_args(argc, argv);

    // 通过工厂模式加载分词器（根据 JSON "type" 字段自动选择类型）
    std::cout << "加载词表: " << args.model_path << "\n";
    auto tk_result = load_tokenizer_by_type(args.model_path);
    if (!tk_result)
    {
        std::cerr << "加载失败: " << tk_result.error().message << '\n';
        return 1;
    }
    // tk_result 是 Result<unique_ptr<Tokenizer>>，需要双重解引用得到 Tokenizer&
    const auto &tokenizer = **tk_result;
    std::cout << "词表大小: " << tokenizer.vocab_size() << " tokens\n\n";

    switch (args.mode)
    {
        case Mode::Text:
            mode_text(tokenizer, args.input, args.show_tokens);
            break;
        case Mode::Encode:
            mode_encode(tokenizer, args.input, args.show_tokens);
            break;
        case Mode::Decode:
            mode_decode(tokenizer, args.input, args.show_tokens);
            break;
        case Mode::Benchmark:
            mode_benchmark(tokenizer, args.input, args.show_tokens);
            break;
        case Mode::Interactive:
            mode_interactive(tokenizer, args.show_tokens);
            break;
    }

    return 0;
}
