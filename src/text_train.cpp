// ── GPT 文本生成训练程序（引擎化架构） ──────────────────────────────────────
//
// 数据流：
//   文本 → Tokenizer.encode → token IDs
//   每 batch：token IDs → Matrix(seq_len, batch) → engine.from_matrix → Tensor
//     → GPTModel.forward(Tensor) → Tensor(vocab_size, seq_len×batch)
//     → ce_loss.forward(engine, logits, onehot) → Scalar
//     → ce_loss.backward() → Tensor
//     → model.backward(Tensor) → (丢弃)
//     → optimizer.step() / zero_grad()
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using nn::Scalar;

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "GPT 文本生成训练程序 (引擎化架构)\n\n"
        << "用法: " << prog << " <text-file> [选项]\n\n"
        << "参数:\n"
        << "  <text-file>        训练文本文件路径 (必需)\n\n"
        << "选项:\n"
        << "  --save <path>      模型保存路径 (默认: gpt_model.bin)\n"
        << "  --resume <path>    从已有模型恢复训练\n"
        << "  --vocab <path>     词表 JSON 路径 (默认: gpt_bpe.json)\n"
        << "                     自动识别分词器类型（bpe / charbpe / wordzip / space）\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 32)\n"
        << "  --seq-len <n>      序列长度 (默认: 256)\n"
        << "  --optimizer <name> 优化器: sgd/sgd_momentum/adam/adamw/muon (默认: adam)\n"
        << "  --weight-decay <w> AdamW 权重衰减系数 (默认: 0.01)\n"
        << "  --d-model <n>      模型维度 (默认: 128)\n"
        << "  --num-heads <n>    注意力头数 (默认: 4)\n"
        << "  --num-layers <n>   Transformer 层数 (默认: 4)\n"
        << "  --d-ff <n>         FFN 中间维度 (默认: 512)\n"
        << "  --gpu              启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --positional-encoding <type>\n"
        << "                     位置编码类型: learned(默认)/sinusoidal/alibi\n"
        << "                     learned: 可学习位置嵌入（默认，GPT 原版）\n"
        << "                     sinusoidal: 正弦波固定位置编码（不参与训练）\n"
        << "                     alibi: 线性偏置注意力（无位置嵌入，支持长度外推）\n"
        << "  --log-interval <n> 每隔多少 step 显示进度 (默认: 50)\n"
        << "  --grad-log         显示梯度统计（范数/最大值/均值）\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string text_path;
    std::string save_path = "gpt_model.bin";
    std::string vocab_path = "gpt_bpe.json";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    Scalar lr = 0.001;
    Scalar weight_decay = 0.01f;  // AdamW 权重衰减系数
    std::size_t batch_size = 32;
    std::size_t seq_len = nn::GPT_SEQ_LEN;
    std::size_t d_model = nn::GPT_D_MODEL;
    std::size_t num_heads = nn::GPT_NUM_HEADS;
    std::size_t num_layers = nn::GPT_NUM_LAYERS;
    std::size_t d_ff = nn::GPT_D_FF;
    std::size_t log_interval = 50;
    bool load_existing = false;
    bool gpu_enabled = false;
    bool grad_log = false;          // 显示梯度统计
    nn::PosEncodingType pos_encoding = nn::PosEncodingType::Learned;
};

TrainConfig parse_args(int argc, char *argv[])
{
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--save" && i + 1 < argc)
            cfg.save_path = argv[++i];
        else if (arg == "--resume" && i + 1 < argc)
        {
            cfg.resume_path = argv[++i];
            cfg.load_existing = true;
        }
        else if (arg == "--vocab" && i + 1 < argc)
            cfg.vocab_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --epochs: " << v.error().message << "\n"; std::exit(1); }
            cfg.epochs = *v;
        }
        else if (arg == "--lr" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --lr: " << v.error().message << "\n"; std::exit(1); }
            cfg.lr = *v;
        }
        else if (arg == "--batch-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --batch-size: " << v.error().message << "\n"; std::exit(1); }
            cfg.batch_size = *v;
        }
        else if (arg == "--seq-len" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --seq-len: " << v.error().message << "\n"; std::exit(1); }
            cfg.seq_len = *v;
        }
        else if (arg == "--optimizer" && i + 1 < argc)
        {
            cfg.optimizer_name = argv[++i];
            if (cfg.optimizer_name != "sgd" && cfg.optimizer_name != "sgd_momentum" &&
                cfg.optimizer_name != "adam" && cfg.optimizer_name != "adamw" &&
                cfg.optimizer_name != "muon")
            {
                std::cerr << "未知优化器: " << cfg.optimizer_name
                          << "，可选: sgd, sgd_momentum, adam, adamw, muon\n";
                std::exit(1);
            }
        }
        else if (arg == "--weight-decay" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --weight-decay: " << v.error().message << "\n"; std::exit(1); }
            cfg.weight_decay = *v;
        }
        else if (arg == "--d-model" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --d-model: " << v.error().message << "\n"; std::exit(1); }
            cfg.d_model = *v;
        }
        else if (arg == "--num-heads" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --num-heads: " << v.error().message << "\n"; std::exit(1); }
            cfg.num_heads = *v;
        }
        else if (arg == "--num-layers" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --num-layers: " << v.error().message << "\n"; std::exit(1); }
            cfg.num_layers = *v;
        }
        else if (arg == "--d-ff" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --d-ff: " << v.error().message << "\n"; std::exit(1); }
            cfg.d_ff = *v;
        }
        else if (arg == "--log-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --log-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.log_interval = *v;
        }
        else if (arg == "--gpu")
            cfg.gpu_enabled = true;
        else if (arg == "--grad-log")
            cfg.grad_log = true;
        else if (arg == "--positional-encoding" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "learned")
                cfg.pos_encoding = nn::PosEncodingType::Learned;
            else if (v == "sinusoidal")
                cfg.pos_encoding = nn::PosEncodingType::Sinusoidal;
            else if (v == "alibi")
                cfg.pos_encoding = nn::PosEncodingType::ALiBi;
            else
            {
                std::cerr << "未知位置编码类型: " << v
                          << "，可选: learned, sinusoidal, alibi\n";
                std::exit(1);
            }
        }
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

    return cfg;
}

// ==================== One-Hot 编码 ====================
nn::Matrix one_hot_labels(const std::vector<std::size_t> &tokens, std::size_t vocab_size)
{
    const std::size_t n = tokens.size();
    nn::Matrix result(vocab_size, n);
    result.zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (tokens[i] < vocab_size)
            result.set_value_unchecked(tokens[i], i, 1.0);
    }
    return result;
}

// ==================== 梯度统计 ====================
// 计算并打印全局梯度统计：L2 范数、绝对值最大值、均值。
// GPU 模式下自动通过 engine.to_matrix() 下载张量到 CPU。
void log_gradient_stats(nn::ComputeEngine &engine, const std::vector<nn::Tensor *> &grads)
{
    Scalar global_sum_sq = 0.0;
    Scalar global_abs_max = 0.0;
    Scalar global_abs_sum = 0.0;
    std::size_t global_count = 0;
    std::size_t tensor_idx = 0;

    for (auto *grad : grads)
    {
        // GPU 张量需要下载到 CPU
        nn::Matrix mat;
        if (grad->device() == nn::Device::GPU)
        {
            auto m = engine.to_matrix(*grad);
            if (!m) continue;
            mat = std::move(*m);
        }
        else
        {
            mat = grad->cpu_matrix();
        }

        const Scalar sum_sq = mat.reduce(Scalar{0}, std::plus<>{},
            [](Scalar x) { return x * x; });
        const Scalar abs_max = mat.reduce(Scalar{0},
            [](Scalar a, Scalar b) { return std::max(a, b); },
            [](Scalar x) { return std::abs(x); });
        const Scalar abs_sum = mat.reduce(Scalar{0}, std::plus<>{},
            [](Scalar x) { return std::abs(x); });
        const std::size_t n = mat.size();

        global_sum_sq += sum_sq;
        global_abs_max = std::max(global_abs_max, abs_max);
        global_abs_sum += abs_sum;
        global_count += n;

        // 逐张量输出（仅在张量数量 <= 30 时显示详情，避免刷屏）
        if (grads.size() <= 30)
        {
            std::cout << "    [" << tensor_idx << "] "
                      << "(" << mat.rows() << "x" << mat.cols() << ") "
                      << "norm=" << std::scientific << std::setprecision(4)
                      << std::sqrt(sum_sq) << "  max|g|=" << abs_max
                      << "  mean|g|=" << (n > 0 ? abs_sum / n : Scalar{0})
                      << std::endl;
        }
        ++tensor_idx;
    }

    const Scalar global_norm = std::sqrt(global_sum_sq);
    const Scalar global_mean = global_count > 0 ? global_abs_sum / global_count : Scalar{0};
    std::cout << "    >> grad_norm=" << std::scientific << std::setprecision(4) << global_norm
              << "  max|g|=" << global_abs_max
              << "  mean|g|=" << global_mean
              << "  params=" << global_count
              << "  tensors=" << grads.size()
              << std::endl;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    TrainConfig cfg = parse_args(argc, argv);

    // ── 加载文本 ─────────────────────────────────────────────
    std::cout << "加载文本: " << cfg.text_path << " ..." << std::endl;
    auto text_result = nn::load_text_file(cfg.text_path);
    if (!text_result) {
        std::cerr << "Error: " << text_result.error().message << '\n';
        return 1;
    }
    std::string text = std::move(*text_result);
    if (text.empty())
    {
        std::cerr << "文本文件为空\n";
        return 1;
    }

    // ── 加载分词器（自动识别类型：BPE/CharBPE/WordZip/Space） ───
    auto tokenizer = nn::load_tokenizer_from_file(cfg.vocab_path);
    if (!tokenizer)
    {
        std::cerr << "加载词表失败或无法识别分词器类型: " << cfg.vocab_path << '\n'
                  << "请检查 JSON 文件是否包含有效的 \"type\" 字段" << std::endl;
        return 1;
    }
    const std::size_t bos_id = tokenizer->bos_id();
    const std::size_t eos_id = tokenizer->eos_id();

    // ── 按行编码：每行独立存储为 [BOS] + tokens + [EOS] ──────────
    // 关键：一行 = 一个训练样本。batch 内包含多行，每行独立 PAD/截断到 seq_len。
    // 这样模型在行尾学到 EOS（结束信号），行内只看到正常 token，
    // 不会因序列中间出现 EOS 而学到"随时输出 EOS"的错误偏置。
    // 所有分词器的 PAD_ID 都是 0（特殊 token 布局统一：0=pad,1=unk,2=bos,3=eos）
    const std::size_t pad_id = 0;
    std::vector<std::vector<std::size_t>> line_samples;
    std::size_t total_token_count = 0;
    {
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line))
        {
            // 去除行首尾空白
            auto lpos = line.find_first_not_of(" \t\r\n");
            if (lpos == std::string::npos) continue;  // 空行跳过
            line = line.substr(lpos, line.find_last_not_of(" \t\r\n") - lpos + 1);
            if (line.empty()) continue;

            auto line_tokens = tokenizer->encode(line);
            // 构造样本：[BOS] + tokens + [EOS]
            std::vector<std::size_t> sample;
            sample.reserve(line_tokens.size() + 2);
            if (bos_id != nn::Tokenizer::npos)
                sample.push_back(bos_id);
            sample.insert(sample.end(), line_tokens.begin(), line_tokens.end());
            if (eos_id != nn::Tokenizer::npos)
                sample.push_back(eos_id);
            total_token_count += sample.size();
            line_samples.push_back(std::move(sample));
        }
    }
    std::cout << "文本行数: " << line_samples.size() << ", "
              << total_token_count << " tokens (含 BOS/EOS), "
              << tokenizer->vocab_size() << " 词表\n" << std::endl;

    // ── 打印配置 ─────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "  GPT 文本生成训练 (引擎化架构)\n";
    std::cout << "========================================\n";
    std::cout << "  词表大小: " << tokenizer->vocab_size() << "\n";
    std::cout << "  模型维度: " << cfg.d_model << "\n";
    std::cout << "  注意力头: " << cfg.num_heads << "\n";
    std::cout << "  Transformer 层数: " << cfg.num_layers << "\n";
    std::cout << "  FFN 维度: " << cfg.d_ff << "\n";
    std::cout << "  序列长度: " << cfg.seq_len << "\n";
    std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
    std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
    std::cout << "  GPU: " << (cfg.gpu_enabled ? "启用" : "禁用") << "\n";
    std::cout << "  梯度日志: " << (cfg.grad_log ? "启用" : "禁用") << "\n";
    std::cout << "========================================\n\n";

    // ── 创建计算引擎 ─────────────────────────────────────────
    std::unique_ptr<nn::ComputeEngine> engine;
#ifdef NN_HAS_VULKAN
    nn::GpuBackend *gpu_backend = nullptr;
#endif
    if (cfg.gpu_enabled)
    {
#ifdef NN_HAS_VULKAN
        auto &backend = nn::GpuBackend::instance();
        auto init_result = backend.initialize();
        if (init_result)
        {
            gpu_backend = &backend;
            engine = std::make_unique<nn::GpuEngine>(*gpu_backend);
            std::cout << "GPU 加速已启用 (Vulkan GpuEngine)\n\n";
        }
        else
        {
            std::cerr << "GPU 初始化失败: " << init_result.error().message << "\n";
            std::cerr << "回退到 CPU 模式\n\n";
            engine = std::make_unique<nn::CpuEngine>();
        }
#else
        std::cerr << "未编译 Vulkan 支持，使用 CPU 模式\n\n";
        engine = std::make_unique<nn::CpuEngine>();
#endif
    }
    else
        engine = std::make_unique<nn::CpuEngine>();

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    auto model_build = nn::build_gpt_model(
        *engine,
        tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
        cfg.num_heads, cfg.d_ff, cfg.num_layers,
        cfg.pos_encoding);
    if (!model_build) {
        std::cerr << "构建模型失败: " << model_build.error().message << '\n';
        return 1;
    }
    auto model = std::move(*model_build);

    // ── 构建规格（用于保存） ─────────────────────────────────
    auto spec = nn::make_gpt_spec(
        tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
        cfg.num_heads, cfg.d_ff, cfg.num_layers,
        cfg.pos_encoding);

    if (cfg.load_existing)
    {
        auto spec_result = nn::peek_model_spec(cfg.resume_path);
        if (!spec_result)
        {
            std::cerr << "加载模型失败: " << spec_result.error().message << "，将从头训练。\n" << std::endl;
        }
        else
        {
            auto file_spec = std::move(*spec_result);
            if (file_spec.is_gpt())
            {
                std::cout << "从模型文件读取 GPT 规格\n";
                auto build_result = nn::build_gpt_model_from_spec(*engine, file_spec);
                if (!build_result)
                {
                    std::cerr << "Error: " << build_result.error().message << '\n';
                    return 1;
                }
                model = std::move(*build_result);
                spec = file_spec;
            }
            else if (file_spec.is_alibi_gpt())
            {
                std::cout << "从模型文件读取 ALiBi GPT 规格\n";
                auto build_result = nn::build_alibi_gpt_model_from_spec(*engine, file_spec);
                if (!build_result)
                {
                    std::cerr << "Error: " << build_result.error().message << '\n';
                    return 1;
                }
                model = std::move(*build_result);
                spec = file_spec;
            }
            else
                std::cout << "旧格式模型文件，使用命令行参数\n";

            auto load_result = nn::load_model(cfg.resume_path, model);
            if (!load_result)
                std::cerr << "加载模型失败: " << load_result.error().message << "，将从头训练。\n" << std::endl;
            else
            {
                std::cout << "已加载模型: " << cfg.resume_path;
                if (!load_result->empty())
                    std::cout << " (含嵌入词表 " << load_result->size() << " 字节)";
                std::cout << "\n" << std::endl;
            }
        }
    }

    // ── 优化器 ─────────────────────────────────────────────
    auto optimizer = nn::create_optimizer(
        cfg.optimizer_name, *engine,
        model.parameters(), model.param_gradients(), cfg.lr,
        cfg.weight_decay);

    nn::CrossEntropyLoss ce_loss;

    // ── 训练循环 ─────────────────────────────────────────────
    // 每行 = 一个样本。batch 内 batch_size 个样本独立采样。
    // 每样本截断/PAD 到 seq_len，目标 = 输入左移一位（next-token prediction）。
    if (line_samples.empty())
    {
        std::cerr << "无有效训练样本\n";
        return 1;
    }
    // 过滤掉长度 < 2 的样本（无法构造 x/y 对）
    std::vector<const std::vector<std::size_t> *> valid_samples;
    valid_samples.reserve(line_samples.size());
    for (const auto &s : line_samples)
        if (s.size() >= 2) valid_samples.push_back(&s);
    if (valid_samples.empty())
    {
        std::cerr << "无长度 >= 2 的训练样本\n";
        return 1;
    }
    std::cout << "有效样本数: " << valid_samples.size()
              << " (长度 >= 2)\n" << std::endl;

    const std::size_t steps_per_epoch = std::min(
        valid_samples.size() / cfg.batch_size, std::size_t{1000});

    // 改造：每 epoch 开始前 shuffle 样本索引队列，每 step 顺序切片。
    // 不再每 step 独立随机采样，保证每个样本每 epoch 被访问一次。
    std::mt19937_64 rng{42};
    std::vector<std::size_t> sample_indices(valid_samples.size());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;

    // ── 预分配 batch 缓冲区 ───────────────────────────────────
    // x_tokens: (seq_len, batch_size) 输入 token IDs
    // y_tokens: (seq_len, batch_size) 目标 token IDs（x 左移一位）
    // 短行用 pad_id 填充，目标对应位置也用 pad_id（loss 对 PAD 位置无意义但无害）
    nn::Matrix x_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix y_tokens(cfg.seq_len, cfg.batch_size);
    const std::size_t total_tokens = cfg.seq_len * cfg.batch_size;
    std::vector<std::size_t> flat_targets(total_tokens);
    nn::Matrix y_onehot = one_hot_labels(flat_targets, tokenizer->vocab_size());

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_loss = 0.0;

        // 每个 epoch 开始前 shuffle 样本索引队列
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        for (std::size_t step = 0; step < steps_per_epoch; ++step)
        {
            // ── 采样 batch：每样本是一行 ─────────────────────
            // 改造：按 shuffle 后的顺序切片取 batch_size 个样本，
            // 不再独立随机采样，保证每 epoch 样本不重复访问
            for (std::size_t b = 0; b < cfg.batch_size; ++b)
            {
                const auto &sample = *valid_samples[sample_indices[step * cfg.batch_size + b]];
                const std::size_t sample_len = sample.size();

                // 填充 x_tokens 和 y_tokens
                // x[t] = sample[t], y[t] = sample[t+1]（next-token prediction）
                // 样本长度 > seq_len+1: 截断到 seq_len+1（取前 seq_len+1 个 token）
                // 样本长度 <= seq_len:  不足部分用 pad_id 填充
                for (std::size_t t = 0; t < cfg.seq_len; ++t)
                {
                    const std::size_t x_id = (t < sample_len) ? sample[t] : pad_id;
                    const std::size_t y_id = (t + 1 < sample_len) ? sample[t + 1] : pad_id;
                    x_tokens.set_value_unchecked(t, b, static_cast<Scalar>(x_id));
                    y_tokens.set_value_unchecked(t, b, static_cast<Scalar>(y_id));
                }
            }

            // ── Matrix → Tensor（上传到引擎设备） ──────────────
            auto x_tensor_r = engine->from_matrix(x_tokens);
            if (!x_tensor_r) {
                std::cerr << "\nfrom_matrix(x_tokens) failed: " << x_tensor_r.error().message << '\n';
                return 1;
            }

            // ── 构造 one-hot 目标 ────────────────────────────
            auto y_span = y_tokens.span();
            for (std::size_t t = 0; t < cfg.seq_len; ++t)
                for (std::size_t b = 0; b < cfg.batch_size; ++b)
                    flat_targets[t * cfg.batch_size + b] =
                        static_cast<std::size_t>(y_span[t * cfg.batch_size + b]);

            y_onehot.zero();
            for (std::size_t i = 0; i < total_tokens; ++i)
            {
                if (flat_targets[i] < tokenizer->vocab_size())
                    y_onehot.set_value_unchecked(flat_targets[i], i, 1.0);
            }

            auto y_tensor_r = engine->from_matrix(y_onehot);
            if (!y_tensor_r) {
                std::cerr << "\nfrom_matrix(y_onehot) failed: " << y_tensor_r.error().message << '\n';
                return 1;
            }

            // ── 启用 GPU batch 录制 ─────────────────────────
            // 整个 forward + backward + optimizer step 录制到一个 command buffer，
            // end_batch 时一次 vkQueueSubmit + vkWaitForFences，消除 per-primitive 同步开销。
            // CPU 引擎 begin_batch/end_batch 为 no-op，所以两套引擎都安全。
            auto begin_r = engine->begin_batch();
            if (!begin_r) {
                std::cerr << "begin_batch failed: " << begin_r.error().message << '\n';
                return 1;
            }

            // ── 前向传播 ─────────────────────────────────────
            auto fwd_result = model.forward(*x_tensor_r);
            if (!fwd_result) { std::cerr << "Error: " << fwd_result.error().message << '\n'; return 1; }
            auto logits = std::move(*fwd_result);
            // logits: (vocab_size, seq_len × batch_size)

            // ── 损失 ─────────────────────────────────────────
            auto loss_result = ce_loss.forward(*engine, logits, *y_tensor_r);
            if (!loss_result) { std::cerr << "Error: " << loss_result.error().message << '\n'; return 1; }
            Scalar loss = *loss_result;
            total_loss += loss;

            // ── 反向传播 ─────────────────────────────────────
            auto grad_result = ce_loss.backward();
            if (!grad_result) { std::cerr << "\nLoss backward failed: " << grad_result.error().message << '\n'; return 1; }
            auto bwd_result = model.backward(*grad_result);
            if (!bwd_result) { std::cerr << "Error: " << bwd_result.error().message << '\n'; return 1; }

            // ── 优化器 step + 梯度清零 ──
            auto step_result = optimizer->step();
            if (!step_result) {
                std::cerr << "Error: " << step_result.error().message << '\n';
                return 1;
            }

            // ── 梯度统计（step 后、zero_grad 前） ──
            if (cfg.grad_log && ((step + 1) % cfg.log_interval == 0 || step + 1 == steps_per_epoch))
            {
                std::cout << "\n  [grad] step " << step + 1 << ":" << std::endl;
                log_gradient_stats(*engine, model.param_gradients());
            }

            auto zero_result = optimizer->zero_grad();
            if (!zero_result) {
                std::cerr << "\n优化器 zero_grad 失败: " << zero_result.error().message << '\n';
                return 1;
            }

            // ── 提交 batch：一次 vkQueueSubmit + vkWaitForFences ──
            auto end_r = engine->end_batch();
            if (!end_r) {
                std::cerr << "end_batch failed: " << end_r.error().message << '\n';
                return 1;
            }

            // 进度显示
            if ((step + 1) % cfg.log_interval == 0 || step + 1 == steps_per_epoch)
            {
                std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                          << "  step " << step + 1 << "/" << steps_per_epoch
                          << "  loss: " << std::fixed << std::setprecision(4) << loss
                          << "   " << std::flush;
            }
        }

        auto ep_end = std::chrono::steady_clock::now();
        Scalar ep_sec = std::chrono::duration<Scalar>(ep_end - ep_start).count();
        Scalar avg_loss = total_loss / steps_per_epoch;

        std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                  << "  avg_loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  time=" << std::setprecision(1) << ep_sec << "s"
                  << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    Scalar total_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    // ── 读取 tokenizer JSON 以便嵌入模型 ─────────────────────
    std::string tokenizer_json;
    {
        std::ifstream tfs(cfg.vocab_path, std::ios::binary);
        if (tfs)
        {
            std::ostringstream oss;
            oss << tfs.rdbuf();
            tokenizer_json = oss.str();
        }
    }

    // ── 保存模型（含规格 + 嵌入 tokenizer） ──────────────────
    {
        auto save_result = nn::save_model(cfg.save_path, model, spec, tokenizer_json);
        if (!save_result) {
            std::cerr << "Error: " << save_result.error().message << '\n';
            return 1;
        }
    }
    std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1)
              << total_sec << "s"
              << "  词表已嵌入模型文件" << std::endl;

    return 0;
}
