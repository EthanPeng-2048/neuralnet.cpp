// ───────────────────────────────────────────────────────────────────────────
//  tokenizer_consistency_test.cpp — 分词器优化正确性仲裁
//
//  阶段2 优化对象与验证：
//    1. pre_tokenize 手写状态机（pre_match_len）↔ std::regex 版逐字节对拍
//       （覆盖缩写/数字/空格/标点/空白/非 ASCII 字节/特殊标记边界）
//    2. try_match_marker '<' 首字符门控：训练含标记的 tokenizer 后
//       编码时标记仍被识别为单 token，encode∘decode 字节级往返一致
//
//  纯 CPU、无 GPU。编译/运行：
//    clang++ -std=c++26 -fno-exceptions -I include/neuralnet.cpp src/tokenizer_consistency_test.cpp -o tokenizer_consistency_test
//  （Release 下注册 ctest：build-tests 内 cmake --build --target tokenizer_consistency_test && ./test/tokenizer_consistency_test）
// ───────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include <neuralnet.cpp/domain_tokenizer.hpp>

using nn::BPETokenizer;

namespace
{

int g_fail = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) { std::printf("[FAIL] %s\n", msg); ++g_fail; }               \
    } while (0)

// ── 参考实现：完整旧版 pre_tokenize（标记切分 + 纯文本 sregex_iterator） ──
// 逐字复刻优化前 domain_tokenizer_bpe.hpp 的实现，作为手写状态机的对拍基准。
std::vector<std::string> old_pre_tokenize(std::string_view text)
{
    static const std::vector<std::string> markers = [] {
        std::vector<std::string> m = {
            "<|system|>", "<|end_of_system|>", "<|user|>", "<|end_of_user|>",
            "<|assistant|>", "<|end_of_assistant|>", "<|end_of_turn|>", "<|sep|>",
            "<|tool|>", "<|tool_result|>", "<|image|>", "<|audio|>",
            "<|video|>", "<|file|>"
        };
        std::sort(m.begin(), m.end(),
                  [](const auto &a, const auto &b) { return a.size() > b.size(); });
        return m;
    }();
    static const std::regex pat(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+|[^\s\w]+|\s+)",
        std::regex::optimize);

    std::vector<std::string> chunks;
    std::string plain;
    auto flush_plain = [&]() {
        if (plain.empty()) return;
        std::string s = std::move(plain); plain.clear();
        auto begin = std::sregex_iterator(s.begin(), s.end(), pat);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) chunks.push_back(it->str());
    };

    std::size_t pos = 0;
    while (pos < text.size())
    {
        std::size_t mlen = 0;
        for (const auto &mk : markers)
            if (pos + mk.size() <= text.size() && text.substr(pos, mk.size()) == mk)
            { mlen = mk.size(); break; }
        if (mlen > 0)
        {
            flush_plain();
            chunks.push_back(std::string(text.substr(pos, mlen)));
            pos += mlen;
        }
        else
        {
            plain.push_back(text[pos]);
            ++pos;
        }
    }
    flush_plain();
    return chunks;
}

bool eq_chunks(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

void check_one(std::string_view s, const char* label)
{
    const auto manual = BPETokenizer::pre_tokenize(s);
    const auto regex  = old_pre_tokenize(s);
    if (!eq_chunks(manual, regex))
    {
        ++g_fail;
        std::printf("[FAIL] pre_tokenize mismatch (%s) input=[%s]\n",
                    label, std::string(s).c_str());
        std::printf("  manual: "); for (auto& c : manual) std::printf("[%s]", c.c_str()); std::printf("\n");
        std::printf("  old   : "); for (auto& c : regex)  std::printf("[%s]", c.c_str()); std::printf("\n");
    }
}

void test_fixed_cases()
{
    static const char* fixed[] = {
        "hello", " hello", "  hello", "hello world", "don't", "can't", "I'm", "you're",
        "they've", "it'll", "we'd", "she's", "'t", "'re", "'", "'x", "it's a test",
        "123", " 123", "abc123", "123abc", "foo bar 42 baz", "0", " 0x1F",
        "!!!", "a,b.c;d:e_f", "_", "word_word", "a b\tc\nd\re\ff\v",
        "\t\n\r\f\v", "  ", "   ", " \t", "\t ", "a  b", "a\tb",
        "<|system|>", "hi<|user|>there", "<|tool_result|>", "<|sep|>",
        "x<|end_of_assistant|>y", "<|<|<|", "a< b",
        "naïve", "\xc3\xa9", "\xe4\xbd\xa0\xe5\xa5\xbd", "\xff\xfe\x00",
        "one two\nthree\tfour", "  leading  spaces  ", "trailing  \n\n",
        "'tis", "rock 'n' roll", "it''s", "A'B", "don't stop",
    };
    for (const char* s : fixed)
        check_one(s, "fixed");
    std::printf("fixed cases: %zu\n", sizeof(fixed) / sizeof(fixed[0]));
}

void test_random_ascii()
{
    std::mt19937 g(12345);
    std::uniform_int_distribution<int> dch(32, 126);   // 可打印 ASCII
    std::string buf;
    for (int iter = 0; iter < 20000; ++iter)
    {
        const std::size_t len = g() % 80;
        buf.clear();
        for (std::size_t i = 0; i < len; ++i)
            buf.push_back(static_cast<char>(dch(g)));
        check_one(buf, "rand-ascii");
    }
    std::printf("random ascii: 20000\n");
}

void test_random_bytes()
{
    std::mt19937 g(777);
    std::string buf;
    for (int iter = 0; iter < 10000; ++iter)
    {
        const std::size_t len = g() % 64;
        buf.clear();
        for (std::size_t i = 0; i < len; ++i)
            buf.push_back(static_cast<char>(g() & 0xFF));
        check_one(buf, "rand-bytes");
    }
    std::printf("random bytes: 10000\n");
}

void test_contraction_whitespace_grid()
{
    // 缩写 × 前后缀 × 空白 组合
    static const char* cont[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    static const char* pre[] = {"", "a", " ", "  ", "\t", "a ", "'"};
    static const char* post[] = {"", "b", " bb", " ", "123", "!", "_x", "'"};
    std::mt19937 g(999);
    std::string buf;
    for (auto c : cont)
        for (auto p : pre)
            for (auto q : post)
            {
                buf.assign(p);
                buf += c;
                buf += q;
                check_one(buf, "cont-grid");
            }
    // 空格/数字/字母边界随机
    for (int iter = 0; iter < 20000; ++iter)
    {
        buf.clear();
        const std::size_t len = g() % 40;
        for (std::size_t i = 0; i < len; ++i)
        {
            const int r = g() % 4;
            if (r == 0) buf.push_back(' ');
            else if (r == 1) buf.push_back(static_cast<char>('a' + g() % 26));
            else if (r == 2) buf.push_back(static_cast<char>('0' + g() % 10));
            else buf.push_back(".,!?-_\"'/"[g() % 9]);
        }
        check_one(buf, "space-digit-letter");
    }
    std::printf("contraction/whitespace grids + 20000\n");
}

void test_marker_roundtrip()
{
    // 训练一个小词表含标记的 tokenizer，验证编码识别标记 + 往返无损
    nn::BPETokenizer tok;
    nn::BPETokenizer::Config cfg;
    cfg.vocab_size = 320;
    cfg.min_freq = 1;
    std::string corpus =
        "hello world hello world this is a tokenizer test 123 456 789\n"
        "don't can't I'm you're we'll it'd\n"
        "<|system|> system text <|user|> user text <|assistant|> assistant text "
        "<|tool_result|> 42 <|sep|> end\n";
    const auto tr = tok.train(corpus, cfg);
    CHECK(tr.has_value(), "train ok");
    if (!tr) return;

    CHECK(tok.has_dialogue_markers(), "has dialogue markers");
    CHECK(tok.system_marker_id() != nn::Tokenizer::npos, "system id");
    CHECK(tok.user_marker_id() != nn::Tokenizer::npos, "user id");
    CHECK(tok.assistant_marker_id() != nn::Tokenizer::npos, "assistant id");

    // 编码含标记文本：标记须被识别为独立单 token
    const std::string text =
        "Hello <|system|> please be nice <|user|> who are you <|assistant|> I'm a bot 42";
    const auto ids = tok.encode(text);
    CHECK(!ids.empty(), "encode non-empty");
    bool saw_system = false, saw_user = false, saw_assistant = false;
    for (auto id : ids)
    {
        if (id == tok.system_marker_id()) saw_system = true;
        if (id == tok.user_marker_id()) saw_user = true;
        if (id == tok.assistant_marker_id()) saw_assistant = true;
    }
    CHECK(saw_system, "system marker encoded as single id");
    CHECK(saw_user, "user marker encoded as single id");
    CHECK(saw_assistant, "assistant marker encoded as single id");

    // 往返：byte 级 BPE + 标记解码应逐字节还原
    const std::string decoded = tok.decode(ids);
    CHECK(decoded == text, "decode(encode(text)) == text (round-trip)");
    if (decoded != text)
    {
        std::printf("  orig=[%s]\n  dec =[%s]\n", text.c_str(), decoded.c_str());
    }
}

} // namespace

int main()
{
    std::printf("=== tokenizer_consistency_test ===\n");
    test_fixed_cases();
    test_random_ascii();
    test_random_bytes();
    test_contraction_whitespace_grid();
    test_marker_roundtrip();

    if (g_fail == 0)
    {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", g_fail);
    return 1;
}
