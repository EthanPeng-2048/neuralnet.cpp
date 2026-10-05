// ── 文本数据集生成器（dataset_gen）────────────────────────────────────────
//
// 文本 + 词表 → .nndataset（docs/development/19-unified-dataset.md §6.1）：
//   读文本 → trim/过滤 → 并行 tokenize（保序）→ 掩码（--loss-scope assistant）
//   → 打包写 .nndataset（词表/来源/生成参数作配套层层嵌套 kvrec）。
//
// 职责单一：不做训练相关任何事（窗口化是训练期策略，仍归 text_train）。
//   dataset_gen <text-file> --vocab <v.nnvocab> -o <out.nndataset>
//              [--test <test-file>] [--loss-scope all|assistant]
//              [--license <file>] [--source <name>]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_help.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    nn::cli::Help help(std::cout, prog, "文本数据集生成器（文本 + 词表 → .nndataset）");

    help.usage("<text-file> --vocab <v.nnvocab> -o <out.nndataset> [选项]");

    help.section("参数");
    help.item("<text-file>", "训练文本文件路径 (UTF-8，每行 = 一篇 doc)");

    help.section("选项");
    help.opt("--vocab <v.nnvocab>", "词表文件路径 (.nnvocab，必需)\n词表从配套嵌入数据集，故必须先有词表 (tokenizer_train 先行)");
    help.opt("-o, --output <path>", "输出 .nndataset 路径 (必需)");
    help.opt("--test <test-file>", "测试集文本 (可选第二输入 → test 子集，结构与 train 相同)");
    help.opt("--loss-scope <scope>", "loss 计算范围: all / assistant (默认: all)\nall: 全部有效 token 参与 loss (预训练，不写 mask 段)\nassistant: 仅 <|assistant|>…<|end_of_assistant|> 段参与 (对话 SFT；\n逐 doc 写入 mask 段，语义在生成期决定，训练端不再选)");
    help.opt("--license <file>", "许可文本文件 (可选；全文原样写入配套 license 字段)");
    help.opt("--source <name>", "来源名称 (默认: 训练文本文件路径)");
    help.opt("--help, -h", "显示此帮助信息");
}

// ==================== 命令行参数 ====================
struct Config
{
    std::string text_path;
    std::string test_path;
    std::string vocab_path;
    std::string output_path;
    std::string loss_scope = "all";
    std::string license_path;
    std::string source_name;
};

Config parse_args(int argc, char *argv[])
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--vocab" && i + 1 < argc)
            cfg.vocab_path = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
            cfg.output_path = argv[++i];
        else if (arg == "--test" && i + 1 < argc)
            cfg.test_path = argv[++i];
        else if (arg == "--loss-scope" && i + 1 < argc)
        {
            cfg.loss_scope = argv[++i];
            if (cfg.loss_scope != "all" && cfg.loss_scope != "assistant")
            {
                std::cerr << "未知 --loss-scope: " << cfg.loss_scope
                          << "，可选: all, assistant\n";
                std::exit(1);
            }
        }
        else if (arg == "--license" && i + 1 < argc)
            cfg.license_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc)
            cfg.source_name = argv[++i];
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
    if (cfg.vocab_path.empty())
    {
        std::cerr << "请用 --vocab 指定词表文件 (.nnvocab)\n使用 --help 查看用法\n";
        std::exit(1);
    }
    if (cfg.output_path.empty())
    {
        std::cerr << "请用 -o/--output 指定输出路径\n使用 --help 查看用法\n";
        std::exit(1);
    }
    return cfg;
}

// ── 一个子集的生成：docs → encode → （assistant）掩码 ─────────────────
struct SubsetBuild
{
    std::vector<std::vector<std::size_t>> docs;
    std::vector<std::vector<unsigned char>> masks;
    std::size_t asst_tokens = 0;
};

nn::Result<SubsetBuild> build_subset(const nn::Tokenizer &tokenizer,
                                     const std::vector<std::string> &docs,
                                     bool need_mask, std::string_view name)
{
    SubsetBuild out;
    out.docs = nn::encode_docs_parallel(tokenizer, docs);
    if (!need_mask)
        return out;
    for (const auto &toks : out.docs)
    {
        out.masks.push_back(nn::mark_assistant_span(
            toks, tokenizer.assistant_marker_id(), tokenizer.end_assistant_marker_id()));
        for (unsigned char m : out.masks.back())
            out.asst_tokens += m;
    }
    if (out.asst_tokens == 0)
        NN_FAIL(std::string(name) + "中未找到 <|assistant|> 段（--loss-scope assistant "
            "要求语料含对话标记）");
    return out;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    Config cfg = parse_args(argc, argv);
    const bool need_mask = (cfg.loss_scope == "assistant");

    // ── 词表（必需：词表从配套嵌入，tokenizer_train 先行）──────────
    auto tok_r = nn::load_tokenizer_from_file(cfg.vocab_path);
    NN_EXIT(tok_r, 1, "加载词表失败: ");
    std::unique_ptr<nn::Tokenizer> tokenizer = std::move(*tok_r);
    if (need_mask && !tokenizer->has_dialogue_markers())
    {
        std::cerr << "--loss-scope assistant 需要带对话标记的词表（<|assistant|> 等）\n";
        return 1;
    }

    // ── 读文本（trim/过滤规则与原 text_train read_file_lines 逐字一致）──
    auto train_lines_r = nn::read_text_docs(cfg.text_path);
    NN_EXIT(train_lines_r, 1, "读取训练文本失败: ");
    const std::vector<std::string> &train_lines = *train_lines_r;
    if (train_lines.empty())
    {
        std::cerr << "训练文本为空: " << cfg.text_path << "\n";
        return 1;
    }
    std::vector<std::string> test_lines;
    if (!cfg.test_path.empty())
    {
        auto test_lines_r = nn::read_text_docs(cfg.test_path);
        NN_EXIT(test_lines_r, 1, "读取测试文本失败: ");
        test_lines = std::move(*test_lines_r);
    }

    // ── tokenize（保序并行）+ 掩码 ────────────────────────────────
    std::cout << "Tokenize: " << train_lines.size() << " docs ..." << std::endl;
    auto train_r = build_subset(*tokenizer, train_lines, need_mask, "训练集");
    NN_EXIT(train_r, 1, "生成训练子集失败: ");
    std::optional<SubsetBuild> test_build;
    if (!test_lines.empty())
    {
        auto test_r = build_subset(*tokenizer, test_lines, need_mask, "测试集");
        NN_EXIT(test_r, 1, "生成测试子集失败: ");
        test_build = std::move(*test_r);
    }

    std::size_t train_tokens = 0;
    for (const auto &d : train_r->docs) train_tokens += d.size();
    std::size_t test_tokens = 0;
    if (test_build)
        for (const auto &d : test_build->docs) test_tokens += d.size();

    // ── 配套：源摘要（内容 sha256，不是文件大小）/ license / 生成参数 ──
    auto text_bytes_r = nn::load_text_file(cfg.text_path);
    NN_EXIT(text_bytes_r, 1, "读取训练文本失败: ");

    nn::NndatasetBuild build;
    build.vocab_kvrec = tokenizer->to_kvrec();
    build.source_path = cfg.source_name.empty() ? cfg.text_path : cfg.source_name;
    build.source_sha256 = nn::sha256_hex(*text_bytes_r);
    build.source_docs = train_lines.size();
    build.loss_scope = cfg.loss_scope;
    build.tool = "dataset_gen v1";
    build.created = static_cast<std::uint64_t>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    build.train_docs = std::move(train_r->docs);
    build.train_masks = std::move(train_r->masks);
    if (test_build)
    {
        build.test_docs = std::move(test_build->docs);
        build.test_masks = std::move(test_build->masks);
    }
    if (!cfg.license_path.empty())
    {
        auto lic_r = nn::load_text_file(cfg.license_path);
        NN_EXIT(lic_r, 1, "读取许可文件失败: ");
        build.license_text = std::move(*lic_r);
    }

    // ── 写 .nndataset ────────────────────────────────────────────
    std::ofstream ofs(cfg.output_path, std::ios::binary);
    if (!ofs)
    {
        std::cerr << "无法写输出文件: " << cfg.output_path << "\n";
        return 1;
    }
    const std::string label = nn::write_file_label(nn::NNDATASET_MAGIC,
                                                   nn::NNDATASET_FILE_VERSION);
    ofs.write(label.data(), static_cast<std::streamsize>(label.size()));
    auto wr = nn::write_nndataset(ofs, build);
    NN_EXIT(wr, 1, "写 .nndataset 失败: ");
    ofs.close();

    std::cout << "数据集已生成: " << cfg.output_path << "\n"
              << "  kind: text  loss_scope: " << cfg.loss_scope << "\n"
              << "  train: " << train_lines.size() << " docs / " << train_tokens << " tokens";
    if (need_mask)
        std::cout << "（assistant 目标 " << train_r->asst_tokens << " tokens）";
    std::cout << "\n";
    if (test_build)
        std::cout << "  test:  " << test_lines.size() << " docs / " << test_tokens
                  << " tokens\n";
    std::cout << "  词表: " << tokenizer->vocab_size() << " 词（来自 "
              << cfg.vocab_path << "）\n"
              << "  source.sha256: " << build.source_sha256.substr(0, 16) << "...\n";
    return 0;
}
