// ── chat_prompt_test — 多轮对话 prompt 构建 + CLI 接线（P0-5 尾项）──────────
//
// 背景（docs/todo/18 §3.3 X1 / §4.1 P0-5、docs/todo/27 §3 象限（二））：
// P0-5 的采样器 + CLI 半边早已交付，**"多轮对话闭环（连续 N 轮不丢上下文）"
// 一直没落地** —— CLI 的 `--interactive` 每轮只把当轮输入拼进 prompt，GUI 也
// 没有任何历史处理（每轮从零起算）。本用例锁这一项的验收：
//
//   [1] 对话模板结构：BOS + system 块 + 逐轮标记（user/assistant 交替）
//   [2] ★ 上下文不丢：第 N 轮的 prompt 以第 N-1 轮的 prompt 为**前缀**
//   [3] 助手轮 tokens 原样回灌（不经 decode → encode 漂移）
//   [4] 整轮截断：从最旧的一问一答开始丢，system 块与末轮恒保留
//   [5] 纯文本词表（无对话标记）的回退：轮次按 \n 拼接 + BOS，超限截尾
//   [6] CLI 端到端（真实子进程）：--turn 多轮 → prompt tokens 与库逐 token 一致、
//       逐轮累积（单轮是三轮的前缀）、--reply-out 落盘 == 生成 token 的解码
//   [7] CLI 负例：--turn 的形状（起头 user / 交替 / 末条 user）与互斥项
//       不合法即非零退出（不静默降级）
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（未构建 text_infer 时的 CLI 半边）
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "neuralnet.cpp/model_serialization.hpp"

#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

#ifndef _WIN32
#include <sys/wait.h>   // WEXITSTATUS
#endif

namespace
{

int g_fail = 0;

constexpr int kSkip = 77;   // ctest 约定：跳过（CLI 缺失时不阻塞门禁）

std::string g_self;         // argv[0]：定位同构建的 app 二进制

#ifdef _WIN32
// cmd.exe 把正斜杠当开关分隔符：argv[0] 是 D:/... 时直接拼进 system() 会报
// "The filename, directory name, or volume label syntax is incorrect"。
std::string native_path(std::string p)
{
    std::replace(p.begin(), p.end(), '/', '\\');
    return p;
}
#endif

// ── 小工具 ───────────────────────────────────────────────────────────────
std::string dirname_of(const std::string &path)
{
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

bool file_exists(const std::string &p)
{
    std::ifstream f(p);
    return f.good();
}

std::string read_file(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

std::string rstrip_newline(std::string s)
{
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

// 在测试二进制附近寻找同构建的 app（与 cli_precision_warning_test 同款）
std::string find_app(const std::string &name)
{
    const std::string dir = dirname_of(g_self);
    const std::string candidates[] = {dir + "/../" + name, dir + "/" + name, name};
    for (const auto &c : candidates)
        if (file_exists(c))
        {
#ifdef _WIN32
            return native_path(c);
#else
            return c;
#endif
        }
#ifdef _WIN32
    for (const auto &c : candidates)
        if (file_exists(c + ".exe")) return native_path(c + ".exe");
#endif
    return {};
}

int exit_code_of(int rc)
{
#ifdef _WIN32
    return rc;
#else
    return (rc == -1) ? -1 : WEXITSTATUS(rc);
#endif
}

std::filesystem::path scratch_dir()
{
    const auto dir = std::filesystem::temp_directory_path() / "nn_chat_prompt_test";
    std::filesystem::create_directories(dir);
    return dir;
}

// ── 训练一个小 BPE 词表（train 内部已 add_dialogue_markers_to_vocab_）─────
std::string corpus_text()
{
    return "hello world, hello neural net. the quick brown fox jumps.\n"
           "hi there, how are you? bye now, see you later.\n"
           "<|user|>hi<|end_of_user|><|assistant|>hello<|end_of_assistant|>\n"
           "人工神经网络很有趣。深度学习训练中。\n";
}

[[nodiscard]] nn::Result<nn::BPETokenizer> train_tokenizer(std::size_t vocab_size)
{
    nn::BPETokenizer tok;
    nn::BPETokenizer::Config cfg;
    cfg.vocab_size = vocab_size;
    cfg.min_freq = 1;
    cfg.log = [](std::string_view) {};
    NN_TRY_CHECK(tok.train(corpus_text(), cfg));
    return tok;
}

// 无对话标记的词表（纯文本回退路径的探针）：清掉 markers_ 即可
struct PlainTokenizer : nn::BPETokenizer
{
    void clear_markers() { markers_.system = nn::Tokenizer::npos; }
};

[[nodiscard]] std::string join_ids(const std::vector<std::size_t> &ids)
{
    std::string s;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (i > 0) s += ", ";
        s += std::to_string(ids[i]);
    }
    return s;
}

[[nodiscard]] bool is_prefix(const std::vector<std::size_t> &pre,
                             const std::vector<std::size_t> &full)
{
    if (pre.size() > full.size()) return false;
    for (std::size_t i = 0; i < pre.size(); ++i)
        if (pre[i] != full[i]) return false;
    return true;
}

[[nodiscard]] bool contains_id(const std::vector<std::size_t> &ids, std::size_t id)
{
    for (const auto v : ids)
        if (v == id) return true;
    return false;
}

// 从 CLI 输出里抽 "[Prompt tokens: a, b, c]" / "[Generated tokens: ...]"
[[nodiscard]] std::vector<std::size_t> parse_id_list(const std::string &text,
                                                     const std::string &tag)
{
    const std::string key = "[" + tag + ": ";
    const auto pos = text.find(key);
    if (pos == std::string::npos) return {};
    const auto end = text.find(']', pos);
    if (end == std::string::npos) return {};
    std::vector<std::size_t> out;
    const std::string body = text.substr(pos + key.size(), end - pos - key.size());
    std::stringstream ss(body);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        const auto first = item.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        out.push_back(static_cast<std::size_t>(std::stoull(item.substr(first))));
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
// [1] 对话模板结构
// ══════════════════════════════════════════════════════════════════════════
void test_dialogue_structure()
{
    std::printf("── [1] 对话模板结构（BOS + system 块 + 逐轮标记）──\n");
    auto tok_r = train_tokenizer(300);
    CHECK(tok_r.has_value(), "训练小 BPE 词表");
    if (!tok_r) return;
    nn::BPETokenizer &tok = *tok_r;
    CHECK(tok.has_dialogue_markers(), "训练出的词表带对话标记");

    std::vector<nn::Tokenizer::ChatTurn> turns{
        {true, "hi", {}},
        {false, "yo", {}},
        {true, "bye", {}},
    };
    const auto ids = tok.chat_prompt(turns, "SYS");

    // 期望：[BOS]<|system|>SYS<|end_of_system|><|user|>hi<|end_of_user|>
    //       <|assistant|>yo<|end_of_assistant|><|user|>bye<|end_of_user|><|assistant|>
    std::vector<std::size_t> want;
    const auto push_all = [&want](const std::vector<std::size_t> &v) {
        want.insert(want.end(), v.begin(), v.end());
    };
    want.push_back(tok.bos_id());
    want.push_back(tok.system_marker_id());
    push_all(tok.encode("SYS"));
    want.push_back(tok.end_system_marker_id());
    want.push_back(tok.user_marker_id());
    push_all(tok.encode("hi"));
    want.push_back(tok.end_user_marker_id());
    want.push_back(tok.assistant_marker_id());
    push_all(tok.encode("yo"));
    want.push_back(tok.end_assistant_marker_id());
    want.push_back(tok.user_marker_id());
    push_all(tok.encode("bye"));
    want.push_back(tok.end_user_marker_id());
    want.push_back(tok.assistant_marker_id());   // ★ 末轮 user → 开标记待生成

    CHECK(ids == want, ("模板逐 token 一致（实际: " + join_ids(ids) +
                        "；期望: " + join_ids(want) + "）").c_str());
    CHECK(!ids.empty() && ids.back() == tok.assistant_marker_id(),
          "末轮是 user 时以 <|assistant|> 收尾（提示模型作答）");

    // system_prompt 为空 → 不产生 system 块（BOS 与各轮次逐 token 不变）
    std::vector<std::size_t> want_no_sys{tok.bos_id()};
    // 去掉的正是 BOS 之后的 <|system|> + SYS + <|end_of_system|> 三段
    want_no_sys.insert(want_no_sys.end(), want.begin() + 3 + tok.encode("SYS").size(),
                       want.end());
    const auto no_sys = tok.chat_prompt(turns, "");
    CHECK(no_sys == want_no_sys,
          ("空 system_prompt ⇒ 无 system 块（实际 " + join_ids(no_sys) + "；期望 " +
           join_ids(want_no_sys) + "）").c_str());

    // 末轮是 assistant（完整记录）→ 以 <|end_of_assistant|> 收尾
    std::vector<nn::Tokenizer::ChatTurn> closed(turns.begin(), turns.begin() + 2);
    const auto closed_ids = tok.chat_prompt(closed, "");
    CHECK(!closed_ids.empty() && closed_ids.back() == tok.end_assistant_marker_id(),
          "末轮是 assistant ⇒ 以 <|end_of_assistant|> 收尾");
}

// ══════════════════════════════════════════════════════════════════════════
// [2] ★ 上下文不丢：第 N 轮的 prompt 以第 N-1 轮为前缀
// ══════════════════════════════════════════════════════════════════════════
void test_context_accumulates()
{
    std::printf("── [2] 上下文不丢（前缀不变式）──\n");
    auto tok_r = train_tokenizer(300);
    CHECK(tok_r.has_value(), "训练小 BPE 词表");
    if (!tok_r) return;
    nn::BPETokenizer &tok = *tok_r;

    // 模拟连续 5 轮的闭环（GUI / --interactive 都是这个形状）：
    // 每轮 append 一个 user 轮 → 生成 → append 一个 assistant 轮。
    std::vector<nn::Tokenizer::ChatTurn> history;
    std::vector<std::size_t> prev_prompt;
    for (int round = 1; round <= 5; ++round)
    {
        history.push_back({true, "第" + std::to_string(round) + "问", {}});
        const auto prompt = tok.chat_prompt(history, "SYS");
        if (round > 1)
        {
            CHECK(is_prefix(prev_prompt, prompt),
                  ("第 " + std::to_string(round) +
                   " 轮的 prompt 必须以上一轮为前缀（上下文丢了）").c_str());
            CHECK(prompt.size() > prev_prompt.size(),
                  ("第 " + std::to_string(round) + " 轮的 prompt 必须严格更长").c_str());
        }
        // 模型回复：内容无所谓，形状要对（assistant 轮：tokens 原样回灌）
        const std::vector<std::size_t> reply{tok.encode("y").front()};
        history.push_back({false, {}, reply});
        prev_prompt = prompt;
    }

    // 5 轮之后：第 1 轮的输入仍在 prompt 里（没有任何轮次被悄悄丢掉）
    const auto final_prompt = tok.chat_prompt(history, "SYS");
    for (const auto id : tok.encode("第1问"))
        CHECK(contains_id(final_prompt, id), "第 1 轮的输入仍在 5 轮后的 prompt 里");
    CHECK(final_prompt.size() > prev_prompt.size(),
          "带回复的历史仍严格长于提问时（回复确实进了上下文）");
}

// ══════════════════════════════════════════════════════════════════════════
// [3] 助手轮 tokens 原样回灌
// ══════════════════════════════════════════════════════════════════════════
void test_tokens_passthrough()
{
    std::printf("── [3] 助手轮 tokens 原样回灌（不经 decode → encode）──\n");
    auto tok_r = train_tokenizer(300);
    CHECK(tok_r.has_value(), "训练小 BPE 词表");
    if (!tok_r) return;
    nn::BPETokenizer &tok = *tok_r;

    // 上一轮真实生成的 token（这里用一段合法的 id 序列代表）
    const std::vector<std::size_t> generated{tok.encode("yo").front(),
                                             tok.encode(" bye").front()};
    std::vector<nn::Tokenizer::ChatTurn> turns{
        {true, "hi", {}},
        {false, "", generated},   // ★ tokens 字段优先于 text
        {true, "ok", {}},
    };
    const auto ids = tok.chat_prompt(turns, "");
    bool passthrough = false;
    for (std::size_t i = 0; i + generated.size() < ids.size(); ++i)
        if (ids[i] == tok.assistant_marker_id() &&
            std::equal(generated.begin(), generated.end(), ids.begin() + i + 1))
            passthrough = true;
    CHECK(passthrough, "tokens 字段的 id 序列原样灌入（不重新编码）");
}

// ══════════════════════════════════════════════════════════════════════════
// [4] 整轮截断（一问一答一起丢；system 块与末轮恒保留）
// ══════════════════════════════════════════════════════════════════════════
void test_truncation()
{
    std::printf("── [4] 整轮截断（最旧的一问一答先丢）──\n");
    auto tok_r = train_tokenizer(300);
    CHECK(tok_r.has_value(), "训练小 BPE 词表");
    if (!tok_r) return;
    nn::BPETokenizer &tok = *tok_r;

    std::vector<nn::Tokenizer::ChatTurn> turns;
    for (int i = 1; i <= 6; ++i)
    {
        turns.push_back({true, "q" + std::to_string(i), {}});
        if (i < 6) turns.push_back({false, "a" + std::to_string(i), {}});
    }
    const auto full = tok.chat_prompt(turns, "SYS", 0);
    CHECK(!full.empty(), "max_tokens=0 时不截断（完整 prompt）");

    // 逐级收紧上限：丢弃轮数单调不减，且恒保留 BOS + system 块 + 末轮
    const std::vector<std::size_t> limits{
        std::max<std::size_t>(1, full.size() - 1),
        std::max<std::size_t>(1, full.size() / 2),
        std::max<std::size_t>(1, full.size() / 4),
        std::size_t{1}};
    std::size_t prev_dropped = 0;
    for (const auto limit : limits)
    {
        std::size_t dropped = 0;
        const auto ids = tok.chat_prompt(turns, "SYS", limit, &dropped);
        CHECK(dropped % 2 == 0, "丢弃的是整轮对（user+assistant）");
        CHECK(dropped >= prev_dropped, "上限越小丢得越多（单调）");
        prev_dropped = dropped;
        CHECK(!is_prefix(full, ids) || ids.size() <= full.size(), "截断只会变短");
        // 恒保留：BOS + system 块
        std::vector<std::size_t> head{tok.bos_id(), tok.system_marker_id()};
        CHECK(ids.size() >= head.size() &&
                  std::equal(head.begin(), head.end(), ids.begin()),
              "BOS 与 system 块恒保留");
        // 恒保留：末轮 ⇒ 仍以 <|assistant|> 收尾
        CHECK(!ids.empty() && ids.back() == tok.assistant_marker_id(),
              "末轮恒保留（以 <|assistant|> 收尾）");
        // 丢弃后仍以 user 轮起头（交替结构不破）
        // 恒保留前缀 = BOS + <|system|> + SYS + <|end_of_system|>
        const std::size_t after_head = 3 + tok.encode("SYS").size();
        CHECK(after_head < ids.size() && ids[after_head] == tok.user_marker_id(),
              "丢弃后仍以 user 轮起头（标记交替不破）");
    }

    // 上限 1：最多丢 10 轮（11 轮里末轮恒保留），结果宁可超限也不切标记结构
    {
        std::size_t dropped = 0;
        const auto ids = tok.chat_prompt(turns, "SYS", 1, &dropped);
        CHECK(dropped == 10, "6 轮（11 条轮次）时最多丢 10 条 = 全部非末轮");
        CHECK(!ids.empty() && ids.back() == tok.assistant_marker_id(),
              "极限截断下末轮结构完整");
    }
    // 只有一轮 ⇒ 无可丢
    {
        const std::vector<nn::Tokenizer::ChatTurn> one{{true, "only", {}}};
        std::size_t dropped = 99;
        const auto ids = tok.chat_prompt(one, "SYS", 1, &dropped);
        CHECK(dropped == 0, "只有一轮时无可丢（末轮恒保留）");
        CHECK(!ids.empty() && ids.back() == tok.assistant_marker_id(),
              "单轮极端截断下结构仍完整");
    }
}

// ══════════════════════════════════════════════════════════════════════════
// [5] 纯文本词表回退（无对话标记）
// ══════════════════════════════════════════════════════════════════════════
void test_plain_text_fallback()
{
    std::printf("── [5] 纯文本词表回退（无标记 ⇒ \\n 拼接 + BOS）──\n");
    PlainTokenizer plain;
    {
        nn::BPETokenizer::Config cfg;
        cfg.vocab_size = 300;
        cfg.min_freq = 1;
        cfg.log = [](std::string_view) {};
        auto tr = plain.train(corpus_text(), cfg);
        CHECK(tr.has_value(), "训练词表");
        if (!tr) return;
    }
    plain.clear_markers();
    CHECK(!plain.has_dialogue_markers(), "探针词表确实没有对话标记");

    const std::vector<nn::Tokenizer::ChatTurn> turns{
        {true, "hi there", {}},
        {false, "hello", {}},
        {true, "bye", {}},
    };
    const auto ids = plain.chat_prompt(turns, "SYS");
    // 期望 = BOS + encode("hi there") + encode("\n") + encode("hello") + …
    std::vector<std::size_t> want{plain.bos_id()};
    const auto sep = plain.encode("\n");
    for (std::size_t i = 0; i < turns.size(); ++i)
    {
        if (i > 0) want.insert(want.end(), sep.begin(), sep.end());
        const auto t = plain.encode(turns[i].text);
        want.insert(want.end(), t.begin(), t.end());
    }
    CHECK(ids == want, ("纯文本：BOS + 轮次文本按 \\n 拼接（实际 " + join_ids(ids) +
                        "；期望 " + join_ids(want) + "）").c_str());

    // 超限：截尾（保留最近 max_tokens 个 token）
    const auto cut = plain.chat_prompt(turns, "", 3);
    CHECK(cut.size() == 3, "纯文本超限截尾到 max_tokens");
    CHECK(std::equal(cut.begin(), cut.end(), ids.end() - 3),
          "截尾保留的是最近的 token（尾部）");
}

// ══════════════════════════════════════════════════════════════════════════
// [6][7] CLI 端到端 + 负例
// ══════════════════════════════════════════════════════════════════════════
struct CliFixture
{
    nn::BPETokenizer tok;
    std::string model_path;
    std::string vocab_path;
};

std::string shell_quote(const std::string &s)
{
#ifdef _WIN32
    // cmd 会剥掉"整条命令以引号开头"时的首尾引号；无空格路径直接裸写。
    if (s.find_first_of(" \t") == std::string::npos) return s;
#endif
    std::string out = "\"";
    for (const char c : s)
    {
        if (c == '"') out += '\\';
        out += c;
    }
    out += "\"";
    return out;
}

// 造一个可推理的 tiny GPT（词表 = 训练出的小 BPE）+ 词表文件；
// 返回 false = 环境不具备（建模型失败等），调用方据此跳过。
bool make_cli_fixture(const std::filesystem::path &dir, CliFixture &fx)
{
    auto tok_r = train_tokenizer(300);
    CHECK(tok_r.has_value(), "训练小 BPE 词表");
    if (!tok_r) return false;
    fx.tok = std::move(*tok_r);
    fx.vocab_path = (dir / "chat_vocab.nnvocab").string();
    auto saved = fx.tok.save(fx.vocab_path);
    CHECK(saved.has_value(), "写出 .nnvocab");
    if (!saved) return false;

    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    const auto spec = nn::make_gpt_spec(fx.tok.vocab_size(), 16, 64, 2, 32, 1);
    auto model_r = nn::build_gpt_model_from_spec(engine, spec, nn::PrecisionProfile{});
    CHECK(model_r.has_value(), "建 tiny GPT");
    if (!model_r) return false;
    fx.model_path = (dir / "chat_model.nnmodel").string();
    // 空嵌入词表 ⇒ CLI 走 --vocab（与 GUI 的"模型未嵌入词表"路径一致）
    auto sr = nn::save_model(fx.model_path, *model_r, spec, std::string{},
                             nn::ModelRole::Model);
    CHECK(sr.has_value(), "写出 .nnmodel");
    return static_cast<bool>(sr);
}

void test_cli_multi_turn(const std::string &exe, const std::filesystem::path &dir,
                         const CliFixture &fx)
{
    std::printf("── [6] CLI 端到端：--turn 多轮 + --reply-out ──\n");
    const std::string out1 = (dir / "turn1.out").string();
    const std::string err1 = (dir / "turn1.err").string();
    const std::string reply1 = (dir / "turn1.reply").string();
    const std::string out3 = (dir / "turn3.out").string();
    const std::string err3 = (dir / "turn3.err").string();
    const std::string reply3 = (dir / "turn3.reply").string();

    const std::string sys = "SYS";
    const std::string base = shell_quote(exe) + " --model " + shell_quote(fx.model_path) +
                             " --vocab " + shell_quote(fx.vocab_path) +
                             " --system " + shell_quote(sys) +
                             " --max-tokens 6 --seed 11 --show-tokens";

    // ── 单轮（无历史）──
    const int rc1 = exit_code_of(std::system(
        (base + " --turn user:\"hi there\" --reply-out " + shell_quote(reply1) +
         " > " + shell_quote(out1) + " 2> " + shell_quote(err1)).c_str()));
    CHECK(rc1 == 0, "单轮 --turn 正常退出");
    const std::string log1 = read_file(out1);
    const auto prompt1 = parse_id_list(log1, "Prompt tokens");
    CHECK(!prompt1.empty(), "输出里有 Prompt tokens 列表（--show-tokens）");
    const std::string reply1_text = read_file(reply1);
    CHECK(!reply1_text.empty(), "单轮回复文件非空（--reply-out）");

    // ── 三轮（带历史；历史里的 assistant 文本 = 上一轮的回复）──
    // Windows cmd 不能用换行/控制字符/非 ASCII 字节做参数；把助手回复规范成
    // 可打印 ASCII 单行文本。三轮用例只验证"历史确实进入 prompt"，库侧比较
    // 使用同一份文本，故不会削弱前缀不变式。
    std::string r1 = rstrip_newline(reply1_text);
    for (char& c : r1)
    {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x20 || u >= 0x7F || c == '"' || c == '&' || c == '|' ||
            c == '<' || c == '>')
            c = ' ';
    }
    if (r1.find_first_not_of(' ') == std::string::npos) r1 = "ok";
    const int rc3 = exit_code_of(std::system(
        (base + " --turn user:\"hi there\" --turn assistant:" + shell_quote(r1) +
         " --turn user:\"bye now\" --reply-out " + shell_quote(reply3) +
         " > " + shell_quote(out3) + " 2> " + shell_quote(err3)).c_str()));
    CHECK(rc3 == 0, "三轮 --turn 正常退出");
    const std::string log3 = read_file(out3);
    const auto prompt3 = parse_id_list(log3, "Prompt tokens");

    // ★ 上下文不丢（端到端）：单轮 prompt 是三轮 prompt 的前缀，且更短
    CHECK(is_prefix(prompt1, prompt3),
          ("CLI 端到端：单轮 prompt 应为三轮 prompt 的前缀（单轮 " + join_ids(prompt1) +
           " / 三轮 " + join_ids(prompt3) + "）").c_str());
    CHECK(prompt3.size() > prompt1.size(), "三轮 prompt 严格更长（历史真的进了模型输入）");

    // 与库侧同一份 turns 逐 token 一致（接线一致，而不只是"更长"）
    const std::vector<nn::Tokenizer::ChatTurn> turns{
        {true, "hi there", {}},
        {false, r1, {}},
        {true, "bye now", {}},
    };
    const auto lib_prompt = fx.tok.chat_prompt(turns, sys, 64);
    CHECK(lib_prompt == prompt3,
          ("CLI 的 prompt tokens 与库侧 chat_prompt 逐 token 一致（CLI " +
           join_ids(prompt3) + " / 库 " + join_ids(lib_prompt) + "）").c_str());

    // ── --reply-out 落盘 = 生成 token 的解码（截到 <|end_of_assistant|>）──
    const auto gen3 = parse_id_list(log3, "Generated tokens");
    CHECK(!gen3.empty(), "三轮输出里有 Generated tokens 列表");
    std::vector<std::size_t> shown = gen3;
    for (std::size_t i = 0; i < shown.size(); ++i)
        if (shown[i] == fx.tok.end_assistant_marker_id()) { shown.resize(i); break; }
    const std::string expect_reply = fx.tok.decode(shown);
    const std::string got_reply = rstrip_newline(read_file(reply3));
    CHECK(got_reply == expect_reply,
          ("--reply-out 内容 = 生成 token 解码：实际 '" + got_reply + "' 期望 '" +
           expect_reply + "'").c_str());
}

void test_cli_rejections(const std::string &exe, const std::filesystem::path &dir,
                         const CliFixture &fx)
{
    std::printf("── [7] CLI 负例：--turn 形状 / 互斥项不合法即报错 ──\n");
    const std::string out = (dir / "neg.out").string();
    const std::string err = (dir / "neg.err").string();
    const std::string base = shell_quote(exe) + " --model " + shell_quote(fx.model_path) +
                             " --vocab " + shell_quote(fx.vocab_path) + " --max-tokens 2";
    struct Case
    {
        const char *args;
        const char *label;
        const char *needle;
    };
    const Case cases[] = {
        {"--turn user:\"hi\" --prompt \"x\"", "--turn 与 --prompt 互斥", "互斥"},
        {"--turn assistant:\"hi\" --turn user:\"x\"", "首条不是 user", "必须是 user"},
        {"--turn user:\"a\" --turn user:\"b\"", "角色不交替", "交替"},
        {"--turn user:\"a\" --turn assistant:\"b\"", "末条不是 user", "最后一条"},
        {"--prompt \"x\" --reply-out \"x.txt\"", "--reply-out 缺 --turn", "--reply-out"},
        {"--turn user", "缺少冒号", "<role>:<text>"},
        {"--turn bot:x", "role 非法", "<role>:<text>"},
    };
    for (const auto &c : cases)
    {
        const std::string cmd = base + " " + c.args + " > " + shell_quote(out) + " 2> " +
                                shell_quote(err);
        const int rc = exit_code_of(std::system(cmd.c_str()));
        CHECK(rc != 0, (std::string("应非零退出: ") + c.label).c_str());
        const std::string errtxt = read_file(err);
        CHECK(errtxt.find(c.needle) != std::string::npos,
              (std::string("错误信息应说明原因（含 '") + c.needle + "'）: " + c.label)
                  .c_str());
    }
}

} // namespace

int main(int argc, char **argv)
{
    g_self = (argc > 0 && argv[0] != nullptr)
        ? std::filesystem::absolute(argv[0]).string()
        : "";
    std::printf("chat_prompt_test（P0-5 尾项：多轮对话闭环）\n");

    test_dialogue_structure();
    test_context_accumulates();
    test_tokens_passthrough();
    test_truncation();
    test_plain_text_fallback();

    const std::string exe = find_app("text_infer");
    if (exe.empty())
    {
        std::printf("\n[skip] 未找到 text_infer（未构建 app / NN_BUILD_APPS=OFF）："
                    "CLI 半边跳过\n");
        std::printf("\n%s  失败=%d（CLI 半边跳过 = 77）\n",
                    g_fail == 0 ? "PASS" : "FAIL", g_fail);
        return g_fail == 0 ? kSkip : 1;
    }
    const auto dir = scratch_dir();
    CliFixture fx;
    if (make_cli_fixture(dir, fx))
    {
        test_cli_multi_turn(exe, dir, fx);
        test_cli_rejections(exe, dir, fx);
    }

    std::printf("\n%s  失败=%d\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
