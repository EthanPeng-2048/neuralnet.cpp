// ── nnvocab_test — .nnvocab 词表文件（kvrec v2）序列化测试 ─────────────────
// 覆盖（docs/development/19-unified-dataset.md §9 阶段一验收）：
//   1. 旧 JSON 词表 → .nnvocab → 加载 语义一致（vocab/merges/markers/编解码）
//   2. 训练态 tokenizer save/load 往返逐位一致（BPE + CharBPE）
//   3. 旧 .json 词表文件明确拒绝（错误信息指路 dataset_convert vocab）
//   4. 词表文件镜像字节直载（模型内嵌词表路径：v5 JSON / v6 kvrec）
// ───────────────────────────────────────────────────────────────────────────

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <neuralnet.cpp/domain_tokenizer.hpp>

namespace
{

int g_failures = 0;

void expect(bool ok, const std::string &name)
{
    if (!ok)
    {
        ++g_failures;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
    else
        std::printf("  [ ok ] %s\n", name.c_str());
}

using MergeList = std::vector<std::array<std::size_t, 3>>;

std::string hex_encode(const std::string &raw)
{
    static const char *digits = "0123456789abcdef";
    std::string out;
    for (unsigned char b : raw)
    {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 0xF]);
    }
    return out;
}

// 旧 save_vocab_json_ 同形的 JSON 词表（bpe：全条目 hex）
std::string make_bpe_json(const std::vector<std::string> &vocab,
                          const MergeList &merges)
{
    std::ostringstream oss;
    oss << "{\n  \"type\": \"bpe_tokenizer\",\n  \"vocab\": {\n";
    bool first = true;
    for (std::size_t id = 0; id < vocab.size(); ++id)
    {
        if (vocab[id].empty()) continue;
        if (!first) oss << ",\n";
        first = false;
        oss << "    \"" << id << "\": \"" << hex_encode(vocab[id]) << "\"";
    }
    oss << "\n  },\n  \"vocab_size\": " << vocab.size() << ",\n  \"merges\": [";
    for (std::size_t i = 0; i < merges.size(); ++i)
    {
        if (i > 0) oss << ",";
        oss << "\n    [" << merges[i][0] << ", " << merges[i][1]
            << ", " << merges[i][2] << "]";
    }
    oss << "\n  ]\n}\n";
    return oss.str();
}

// 旧 save_vocab_json_ 同形的 JSON 词表（char_bpe：前 4 个特殊 token 纯文本，其余 hex）
std::string make_charbpe_json(const std::vector<std::string> &vocab,
                              const MergeList &merges)
{
    std::ostringstream oss;
    oss << "{\n  \"type\": \"char_bpe_tokenizer\",\n  \"vocab\": {\n";
    bool first = true;
    for (std::size_t id = 0; id < vocab.size(); ++id)
    {
        if (vocab[id].empty()) continue;
        if (!first) oss << ",\n";
        first = false;
        oss << "    \"" << id << "\": \""
            << (id < 4 ? vocab[id] : hex_encode(vocab[id])) << "\"";
    }
    oss << "\n  },\n  \"vocab_size\": " << vocab.size() << ",\n  \"merges\": [";
    for (std::size_t i = 0; i < merges.size(); ++i)
    {
        if (i > 0) oss << ",";
        oss << "\n    [" << merges[i][0] << ", " << merges[i][1]
            << ", " << merges[i][2] << "]";
    }
    oss << "\n  ]\n}\n";
    return oss.str();
}

std::vector<std::string> bpe_fixture_vocab()
{
    std::vector<std::string> vocab;
    for (int i = 0; i < 256; ++i)
        vocab.emplace_back(1, static_cast<char>(i));
    vocab.push_back("<|system|>");
    vocab.push_back("<|end_of_system|>");
    vocab.push_back("<|user|>");
    vocab.push_back("<|end_of_user|>");
    vocab.push_back("<|assistant|>");
    vocab.push_back("<|end_of_assistant|>");
    vocab.push_back("<think>");
    vocab.push_back("</think>");
    vocab.push_back("he");    // 264
    vocab.push_back("hel");   // 265
    return vocab;
}

std::vector<std::string> charbpe_fixture_vocab()
{
    std::vector<std::string> vocab = {"<pad>", "<unk>", "<bos>", "<eos>"};
    for (int i = 0; i < 256; ++i)
        vocab.emplace_back(1, static_cast<char>(i));
    vocab.push_back("人");    // 260
    vocab.push_back("工");    // 261
    vocab.push_back("人工");  // 262
    vocab.push_back("<|system|>");        // 263
    vocab.push_back("<|end_of_system|>"); // 264
    vocab.push_back("<|user|>");          // 265
    vocab.push_back("<|end_of_user|>");   // 266
    vocab.push_back("<|assistant|>");     // 267
    vocab.push_back("<|end_of_assistant|>"); // 268
    vocab.push_back("<think>");         // 269
    vocab.push_back("</think>");       // 270
    return vocab;
}

// 两实例语义一致：词表逐项 + merges + 对话标记 + 编解码
void expect_same_semantics(const nn::Tokenizer &a, const nn::Tokenizer &b,
                           const std::string &name)
{
    expect(a.vocab() == b.vocab(), name + ": vocab 逐项一致");
    auto ka = a.to_kvrec();
    auto kb = b.to_kvrec();
    std::vector<std::uint64_t> ma, mb;
    (void)ka.get("merges", ma);
    (void)kb.get("merges", mb);
    expect(ma == mb, name + ": merges 三元组一致");

    expect(a.system_marker_id() == b.system_marker_id()
        && a.end_system_marker_id() == b.end_system_marker_id()
        && a.user_marker_id() == b.user_marker_id()
        && a.end_user_marker_id() == b.end_user_marker_id()
        && a.assistant_marker_id() == b.assistant_marker_id()
        && a.end_assistant_marker_id() == b.end_assistant_marker_id()
        && a.start_think_marker_id() == b.start_think_marker_id()
        && a.end_think_marker_id() == b.end_think_marker_id(),
        name + ": 对话标记 ID 一致");

    // 编解码语义一致（含对话标记与普通文本）
    const std::string sample =
        "<|user|>he<|end_of_user|><|assistant|>hel<|end_of_assistant|>";
    const auto ia = a.encode(sample);
    const auto ib = b.encode(sample);
    expect(ia == ib, name + ": encode 一致");
    expect(a.decode(ia) == b.decode(ib), name + ": decode 一致");
}

void write_file(const std::string &path, const std::string &content)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
}

// ── 用例 1/2：json → nnvocab → 加载 语义一致（BPE + CharBPE）────────
void test_json_to_nnvocab_roundtrip()
{
    std::printf("json → nnvocab → 加载 语义一致:\n");
    {
        const auto vocab = bpe_fixture_vocab();
        const MergeList merges = {{104, 101, 264}, {264, 108, 265}};
        auto a = nn::load_tokenizer_from_string(make_bpe_json(vocab, merges));
        expect(static_cast<bool>(a), "bpe: JSON 词表载入（A）");
        if (!a) return;

        auto saved = a->save("nnvocab_test_bpe.nnvocab");
        expect(static_cast<bool>(saved), "bpe: 写出 .nnvocab");
        auto b_r = nn::load_tokenizer_from_file("nnvocab_test_bpe.nnvocab");
        expect(static_cast<bool>(b_r), "bpe: 读回 .nnvocab");
        if (!b_r) return;
        expect_same_semantics(*a, **b_r, "bpe");
    }
    {
        const auto vocab = charbpe_fixture_vocab();
        const MergeList merges = {{260, 261, 262}};
        auto a = nn::load_tokenizer_from_string(make_charbpe_json(vocab, merges));
        expect(static_cast<bool>(a), "charbpe: JSON 词表载入（A）");
        if (!a) return;

        auto saved = a->save("nnvocab_test_charbpe.nnvocab");
        expect(static_cast<bool>(saved), "charbpe: 写出 .nnvocab");
        auto b_r = nn::load_tokenizer_from_file("nnvocab_test_charbpe.nnvocab");
        expect(static_cast<bool>(b_r), "charbpe: 读回 .nnvocab");
        if (!b_r) return;
        expect_same_semantics(*a, **b_r, "charbpe");
    }
}

// ── 用例 3：训练态 tokenizer save/load 往返逐位一致 ────────────────
void test_trained_roundtrip()
{
    std::printf("训练态 save/load 往返:\n");
    const std::string corpus =
        "hello world, hello neural net. the quick brown fox.\n"
        "人工神经网络很有趣。深度学习训练中。\n"
        "<|user|>hi<|end_of_user|><|assistant|>hello<|end_of_assistant|>\n";

    {
        nn::BPETokenizer tok;
        nn::BPETokenizer::Config cfg;
        cfg.vocab_size = 300;
        cfg.min_freq = 1;
        cfg.log = [](std::string_view) {};
        auto tr = tok.train(corpus, cfg);
        expect(static_cast<bool>(tr), "bpe: 训练成功");
        if (!tr) return;
        auto saved = tok.save("nnvocab_test_trained_bpe.nnvocab");
        expect(static_cast<bool>(saved), "bpe: 训练态写出 .nnvocab");
        auto b_r = nn::load_tokenizer_from_file("nnvocab_test_trained_bpe.nnvocab");
        expect(static_cast<bool>(b_r), "bpe: 训练态读回");
        if (!b_r) return;
        expect_same_semantics(tok, **b_r, "bpe 训练态");
    }
    {
        nn::CharBPETokenizer tok;
        nn::CharBPETokenizer::Config cfg;
        cfg.vocab_size = 300;
        cfg.min_freq = 1;
        cfg.log = [](std::string_view) {};
        auto tr = tok.train(corpus, cfg);
        expect(static_cast<bool>(tr), "charbpe: 训练成功");
        if (!tr) return;
        auto saved = tok.save("nnvocab_test_trained_charbpe.nnvocab");
        expect(static_cast<bool>(saved), "charbpe: 训练态写出 .nnvocab");
        auto b_r = nn::load_tokenizer_from_file("nnvocab_test_trained_charbpe.nnvocab");
        expect(static_cast<bool>(b_r), "charbpe: 训练态读回");
        if (!b_r) return;
        expect_same_semantics(tok, **b_r, "charbpe 训练态");
    }
}

// ── 用例 4：旧 .json 词表文件明确拒绝 ──────────────────────────────
void test_old_json_rejected()
{
    std::printf("旧 JSON 词表拒绝:\n");
    const auto vocab = bpe_fixture_vocab();
    const MergeList merges = {{104, 101, 264}};
    write_file("nnvocab_test_old.json", make_bpe_json(vocab, merges));

    auto r = nn::load_tokenizer_from_file("nnvocab_test_old.json");
    expect(!r, "旧 .json 词表被拒绝");
    if (!r)
    {
        const std::string &msg = r.error().message;
        expect(msg.find("dataset_convert") != std::string::npos,
               "错误信息指路 dataset_convert");
        std::printf("         （错误信息: %s）\n", msg.c_str());
    }
}

// ── 用例 5：词表文件镜像字节直载（模型内嵌词表路径）────────────────
void test_load_from_bytes()
{
    std::printf("词表镜像字节直载:\n");
    const auto vocab = bpe_fixture_vocab();
    const MergeList merges = {{104, 101, 264}};

    // v5 模型内嵌 = 旧 JSON 字符串
    auto a = nn::load_tokenizer_from_bytes(make_bpe_json(vocab, merges));
    expect(static_cast<bool>(a), "JSON 字节直载（模型 v5 内嵌）");

    // v6 模型内嵌 = .nnvocab 文件镜像字节
    auto b = nn::load_tokenizer_from_string(make_bpe_json(vocab, merges));
    if (!b) return;
    const std::string file = nn::write_nnvocab(b->to_kvrec());
    auto c = nn::load_tokenizer_from_bytes(file);
    expect(static_cast<bool>(c), ".nnvocab 字节直载（模型 v6 内嵌）");
    if (a && c)
        expect_same_semantics(**a, **c, "字节直载");
}

}  // namespace

int main()
{
    std::printf("nnvocab_test:\n");
    test_json_to_nnvocab_roundtrip();
    test_trained_roundtrip();
    test_old_json_rejected();
    test_load_from_bytes();

    std::error_code ec;
    for (const char *p : {"nnvocab_test_bpe.nnvocab", "nnvocab_test_charbpe.nnvocab",
                          "nnvocab_test_trained_bpe.nnvocab",
                          "nnvocab_test_trained_charbpe.nnvocab",
                          "nnvocab_test_old.json"})
        std::filesystem::remove(p, ec);

    std::printf("\nnnvocab_test: %d failure(s)\n", g_failures);
    return g_failures != 0 ? 1 : 0;
}
