// ───────────────────────────────────────────────────────────────────────────
//  perf_tokenizer.cpp — 预分词性能对比（阶段2）
//  对比 BPETokenizer::pre_tokenize（手写状态机）vs 旧 std::regex 版
//  （reconstructed，同 tokenizer_consistency_test.cpp 的 old_pre_tokenize）。
//
//  编译：
//    clang++ -std=c++26 -fno-exceptions -Wall -Wextra -Wpedantic -Werror \
//            -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native \
//            -I include perf_tokenizer.cpp -o perf_tokenizer.exe
// ───────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include <neuralnet.cpp/domain_tokenizer.hpp>

namespace
{

double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// 旧版 pre_tokenize（regex），对拍基准
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
        if (mlen > 0) { flush_plain(); chunks.push_back(std::string(text.substr(pos, mlen))); pos += mlen; }
        else { plain.push_back(text[pos]); ++pos; }
    }
    flush_plain();
    return chunks;
}

// 构造贴近真实英文文本的语料
std::string make_corpus(std::size_t approx_bytes)
{
    std::mt19937 g(2026);
    static const char* words[] = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
        "tokenizer", "byte", "pair", "encoding", "model", "training", "inference", "attention",
        "neural", "network", "gradient", "backprop", "loss", "optimizer", "adam", "weight"
    };
    static const char* punct[] = {" ", ", ", ". ", "! ", "? ", " - ", " (", ") ", ": ", "; "};
    static const char* cont[] = {"'s", "n't", "'re", "'ve", "'ll", "'d", "'m"};
    std::string out;
    out.reserve(approx_bytes + 1024);
    while (out.size() < approx_bytes)
    {
        const int r = static_cast<int>(g() % 8);
        if (r < 5)
            out += words[g() % (sizeof(words) / sizeof(words[0]))];
        else if (r == 5)
            out += punct[g() % (sizeof(punct) / sizeof(punct[0]))];
        else if (r == 6)
            out += std::to_string(g() % 100000);
        else
            out += cont[g() % (sizeof(cont) / sizeof(cont[0]))];
        // 偶尔插入标记 / 非 ASCII
        if ((g() % 1000) == 0) out += "<|tool_result|>";
        if ((g() % 500) == 0) out += "\xc3\xa9";  // é
    }
    return out;
}

} // namespace

int main()
{
    const std::string corpus = make_corpus(20u << 20);  // ~20MB
    std::printf("corpus size: %zu bytes\n", corpus.size());

    // 正确性抽查：两者应产出相同 chunk
    const auto old_chunks = old_pre_tokenize(corpus);
    const auto new_chunks = nn::BPETokenizer::pre_tokenize(corpus);
    bool same = (old_chunks.size() == new_chunks.size());
    if (same)
        for (std::size_t i = 0; i < old_chunks.size(); ++i)
            if (old_chunks[i] != new_chunks[i]) { same = false; break; }
    std::printf("chunk count old=%zu new=%zu identical=%s\n",
                old_chunks.size(), new_chunks.size(), same ? "yes" : "NO");

    // 计时（多次取 min）
    double t_old = 1e30, t_new = 1e30;
    std::size_t check_sum = 0;
    for (int it = 0; it < 5; ++it)
    {
        auto t0 = std::chrono::steady_clock::now();
        const auto c = old_pre_tokenize(corpus);
        t_old = std::min(t_old, ms_since(t0));
        check_sum += c.size();

        t0 = std::chrono::steady_clock::now();
        const auto c2 = nn::BPETokenizer::pre_tokenize(corpus);
        t_new = std::min(t_new, ms_since(t0));
        check_sum += c2.size();
    }
    (void)check_sum;
    std::printf("pre_tokenize %zu bytes: regex %.2fms  manual %.2fms  speedup %.2fx\n",
                corpus.size(), t_old, t_new, t_old / t_new);
    return 0;
}
