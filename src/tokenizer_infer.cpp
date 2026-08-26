// ── BPE 分词器推理程序（编码/解码） ──────────────────────────────────────
//
// 加载词表 JSON → 编码文本为 token IDs / 解码 token IDs 为文本
//
// 自动根据 JSON 的 "type" 字段识别分词器类型：
//   - bpe_tokenizer      → 字节级 BPE（BBPE）
//   - char_bpe_tokenizer  → 字符级 BPE（支持中文）
//
// 支持交互模式和命令行模式。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
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
        << "  " << prog << " --vocab <path> --encode-file <file>  编码文件\n"
        << "  " << prog << " --vocab <path> <text-file>           统计文本文件每行 token 数\n\n"
        << "选项:\n"
        << "  --vocab <path>       词表 JSON 路径 (默认: bpe_vocab.json)\n"
        << "                       自动识别分词器类型（bpe / charbpe）\n"
        << "  --encode <text>      编码文本为 token IDs\n"
        << "  --decode <ids>       解码 token IDs (逗号分隔) 为文本\n"
        << "  --encode-file <path> 编码整个文件（同时自动统计最长行 token 数）\n"
        << "  <text-file>          位置参数：直接给文本文件即自动统计最长行 token 数\n"
        << "  --top <n>            最长行排行榜行数 (默认: 10)\n"
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
    std::string text_path;   // 位置参数：文本文件 → 自动统计最长行
    std::size_t top = 10;    // 最长行排行榜行数
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
        else if (arg == "--top" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v || *v == 0)
            {
                std::cerr << "无效 --top: " << argv[i] << "\n";
                std::exit(1);
            }
            cfg.top = *v;
        }
        else if (!arg.starts_with("--"))
            cfg.text_path = arg;
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

// ==================== 统一输出 token IDs ====================
// 辅助函数：以 "Tokens (<n>): [id0, id1, ...]" 格式输出 ID 列表，
// 消除交互模式 / 命令行模式 / 默认编码路径中的重复打印代码。
void print_token_ids(const std::vector<std::size_t> &ids, std::ostream &os = std::cout)
{
    os << "  Tokens (" << ids.size() << "): [";
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0) os << ", ";
        os << ids[i];
    }
    os << "]\n";
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
            print_token_ids(ids);
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
            print_token_ids(ids);
            auto decoded = tokenizer.decode(ids);
            std::cout << "  Decode: \"" << decoded << "\"\n";
        }
    }
}

// ==================== 最长行 token 统计（趣味功能，给文件即自动激活） ====================
// 逐行编码文本，找出 token 数最多的行，并输出 Top-N 排行榜。
// 无需额外开关：位置参数传入文本文件、或 --encode-file 都会自动执行。

namespace {

// 统计 UTF-8 字符串的字符数（跳过连续字节）
std::size_t utf8_char_count(std::string_view s) noexcept
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80)  // 非连续字节 → 一个字符
            ++n;
    return n;
}

} // namespace

void longest_lines_mode(const nn::Tokenizer &tokenizer, const std::string &path, std::size_t top_n)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "打开文件失败: " << path << '\n';
        std::exit(1);
    }

    struct LineStat {
        std::size_t line_no = 0;
        std::size_t tokens = 0;
        std::size_t bytes = 0;
        std::size_t chars = 0;
        std::string text;
    };

    // 小顶堆：始终保留 token 数最大的 top_n 行（堆顶 = 当前第 top_n 大）。
    // 每行只需 O(log top_n) 的堆操作，而不是对整表 std::sort（O(top_n log top_n)）。
    struct MinHeap {
        bool operator()(const LineStat &a, const LineStat &b) const noexcept
        {
            return a.tokens > b.tokens;  // 小顶堆：tokens 越小越靠堆顶
        }
    };
    std::priority_queue<LineStat, std::vector<LineStat>, MinHeap> best;

    std::size_t line_no = 0;
    std::size_t total_tokens = 0;
    std::size_t nonempty_lines = 0;

    std::string line;
    while (std::getline(in, line))
    {
        ++line_no;
        if (!line.empty() && line.back() == '\r')  // 兼容 Windows CRLF
            line.pop_back();

        auto ids = tokenizer.encode(line);
        total_tokens += ids.size();
        if (!line.empty())
            ++nonempty_lines;
        if (ids.empty() && line.empty())
            continue;  // 空行不参与排行榜

        LineStat st{line_no, ids.size(), line.size(), utf8_char_count(line), line};

        if (best.size() < top_n)
        {
            best.push(std::move(st));
        }
        else if (st.tokens > best.top().tokens)
        {
            best.pop();               // 踢掉当前第 top_n 大
            best.push(std::move(st));
        }
    }

    if (line_no == 0)
    {
        std::cout << "文件为空: " << path << '\n';
        return;
    }

    // 堆弹出顺序是升序 → 反转成降序排行榜
    std::vector<LineStat> top;
    top.reserve(best.size());
    while (!best.empty())
    {
        top.push_back(best.top());
        best.pop();
    }
    std::reverse(top.begin(), top.end());

    // ── 汇总 ──
    std::cout << "══════ 最长行 Token 统计 ══════\n";
    std::cout << "文本文件 : " << path << '\n';
    std::cout << "总行数   : " << line_no << " (非空: " << nonempty_lines << ")\n";
    std::cout << "总 Token : " << total_tokens << '\n';
    if (nonempty_lines > 0)
        std::cout << "平均     : " << std::fixed << std::setprecision(2)
                  << static_cast<double>(total_tokens) / nonempty_lines << " tokens/行\n";
    if (!top.empty())
    {
        std::cout << "最长行   : " << top.front().tokens << " tokens (第 "
                  << top.front().line_no << " 行, "
                  << top.front().bytes << " 字节, "
                  << top.front().chars << " 字符)\n";
    }
    std::cout << "\n── 最长行 Top " << top.size() << " ──\n";
    for (std::size_t i = 0; i < top.size(); ++i)
    {
        const auto &s = top[i];
        std::cout << "#" << (i + 1)
                  << "  第 " << s.line_no << " 行 | " << s.tokens << " tokens | "
                  << s.bytes << " 字节 | " << s.chars << " 字符\n";
        const std::size_t preview_limit = 80;
        std::string_view preview = s.text;
        const bool truncated = preview.size() > preview_limit;
        if (truncated)
        {
            // 截断到完整 UTF-8 字符边界，避免把多字节字符切碎
            std::size_t i = 0, end = 0;
            while (i < s.text.size() && i < preview_limit)
            {
                const auto c = static_cast<unsigned char>(s.text[i]);
                std::size_t width = 1;
                if ((c & 0xE0) == 0xC0)      width = 2;   // 2 字节字符
                else if ((c & 0xF0) == 0xE0) width = 3;   // 3 字节字符（中文等）
                else if ((c & 0xF8) == 0xF0) width = 4;   // 4 字节字符
                if (i + width > preview_limit)
                    break;  // 放不下一个完整字符
                i += width;
                end = i;
            }
            preview = preview.substr(0, end);
        }
        std::cout << "   文本: \"" << preview << (truncated ? "...\"" : "\"") << '\n';
    }
    std::cout << std::flush;
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
        print_token_ids(ids);
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

    // ── 文本文件（位置参数）→ 自动统计最长行 token 数 ──────────
    if (!cfg.text_path.empty())
    {
        longest_lines_mode(*tokenizer, cfg.text_path, cfg.top);
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
        if (ids.empty())
        {
            std::cout << "压缩率: N/A (空输入)\n";
            return 0;
        }
        std::cout << "压缩率: " << std::fixed << std::setprecision(2)
                  << static_cast<double>(text_result->size()) / ids.size()
                  << " bytes/token\n";
        std::cout << '\n';
        // 自动附加最长行 token 统计
        longest_lines_mode(*tokenizer, cfg.encode_file, cfg.top);
        return 0;
    }

    // 无参数 → 交互模式
    interactive_mode(*tokenizer, cfg.show_bytes);
    return 0;
}
