// ── 旧格式转换器（dataset_convert）────────────────────────────────────────
//
// 把旧资产迁移到统一数据集格式（docs/development/19-unified-dataset.md）：
//   dataset_convert vocab <in.json> -o <out.nnvocab>
//       旧 JSON 词表 → .nnvocab（kvrec v2）。旧 JSON 词表不再被
//       tokenizer/text_* 直读，迁移后所有入口只吃 .nnvocab。
//   dataset_convert csv <train.csv> [<test.csv>] -o <out.nndataset>
//       CSV → .nndataset（tabular 形态，阶段四预留，尚未实现）。
//
// 本工具只做格式转换，不做训练相关任何事。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_help.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    nn::cli::Help help(std::cout, prog, "旧格式转换器（词表/数据集迁移）");

    help.usage("vocab <in.json> -o <out.nnvocab> | csv <train.csv> [<test.csv>] -o <out.nndataset>");

    help.section("子命令");
    help.item("vocab", "旧 JSON 词表 → .nnvocab（kvrec v2）");
    help.item("csv", "CSV → .nndataset（tabular 形态；阶段四预留，尚未实现）");

    help.section("选项");
    help.opt("-o, --output <path>", "输出路径 (必需)");
    help.opt("--help, -h", "显示此帮助信息");
}

// ==================== 命令行参数 ====================
struct Config
{
    std::string subcommand;
    std::string input1;
    std::string input2;
    std::string output_path;
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
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc)
            cfg.output_path = argv[++i];
        else if (!arg.starts_with("--"))
        {
            if (arg == "vocab" || arg == "csv")
                cfg.subcommand = arg;
            else if (cfg.input1.empty())
                cfg.input1 = arg;
            else if (cfg.input2.empty())
                cfg.input2 = arg;
            else
            {
                std::cerr << "多余的位置参数: " << arg << "\n使用 --help 查看用法\n";
                std::exit(1);
            }
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }
    if (cfg.subcommand.empty())
    {
        std::cerr << "请指定子命令: vocab / csv\n使用 --help 查看用法\n";
        std::exit(1);
    }
    if (cfg.input1.empty())
    {
        std::cerr << "请指定输入文件\n使用 --help 查看用法\n";
        std::exit(1);
    }
    if (cfg.output_path.empty())
    {
        std::cerr << "请用 -o/--output 指定输出路径\n使用 --help 查看用法\n";
        std::exit(1);
    }
    return cfg;
}

// ==================== vocab：旧 JSON 词表 → .nnvocab ====================
int convert_vocab(const Config &cfg)
{
    std::ifstream ifs(cfg.input1, std::ios::binary);
    if (!ifs)
    {
        std::cerr << "无法打开输入文件: " << cfg.input1 << "\n";
        return 1;
    }
    const std::string json((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    if (json.empty() || json[0] != '{')
    {
        std::cerr << "输入不是 JSON 词表（应以 '{' 开头）: " << cfg.input1 << "\n";
        return 1;
    }

    // JSON 读取路径仅此处与模型 v5 内嵌词表使用（旧 JSON 不再被直读）
    auto tok = nn::load_tokenizer_from_string(json);
    if (!tok)
    {
        std::cerr << "无法解析 JSON 词表（缺少有效 \"type\" 字段）: "
                  << cfg.input1 << "\n";
        return 1;
    }

    auto save_r = tok->save(cfg.output_path);
    NN_EXIT(save_r, 1, "写入 .nnvocab 失败: ");
    std::cout << "词表已转换: " << cfg.input1 << " -> " << cfg.output_path
              << "（" << tok->vocab_size() << " 词）" << std::endl;
    return 0;
}

// ==================== csv：预留（阶段四） ====================
int convert_csv(const Config &)
{
    std::cerr << "dataset_convert csv 尚未实现（tabular 形态为阶段四预留，"
                 "见 docs/development/19-unified-dataset.md §4.4）\n";
    return 1;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    Config cfg = parse_args(argc, argv);
    if (cfg.subcommand == "vocab")
        return convert_vocab(cfg);
    return convert_csv(cfg);
}
