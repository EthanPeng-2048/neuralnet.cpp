// ── GPT 文本生成训练程序（引擎化架构） ──────────────────────────────────────
//
// 数据流（滑动窗口，GPT 预训练标准做法）：
//   文本 → 逐行 Tokenizer.encode → 每行编码为 [BOS]+tokens+[EOS] 后拼接成连续
//     token 流（行边界通过 EOS 编码进流，窗口可跨行，上下文连续）
//   按 stride 对 token 流滑动切 seq_len 窗口 → 每窗口 = 一个训练样本
//   每 batch：窗口 → Matrix(seq_len, batch) → engine.from_matrix → Tensor
//     → GPTModel.forward(Tensor) → Tensor(vocab_size, seq_len×batch)
//     → ce_loss.forward_sparse(engine, logits, flat_labels, loss_mask) → Scalar
//     → ce_loss.backward() → Tensor
//     → model.backward(Tensor) → (丢弃)
//     → optimizer.step() / zero_grad()
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>
#include <neuralnet.cpp/cli/lr_scheduler.hpp>

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
#include <thread>
#include <utility>
#include <vector>

using nn::Scalar;

// ── 设备丢失自动重启：保存 checkpoint → 等待 GPU 驱动恢复 → 重新启动进程 ──
// Windows TDR 重置 GPU 驱动后，需要重新创建 VkDevice 才能继续使用 GPU。
// 由于 VkDevice 已死且无法在同进程内恢复，最可靠的方式是保存 checkpoint 后
// 自动重启进程（带 --resume 和增大 flush_interval 来拆分 GPU 提交）。
[[noreturn]] void restart_on_device_lost(
    const std::string& program_name,
    nn::Model& model,
    const nn::ModelSpec& spec,
    const std::string& tokenizer_json,
    const std::string& save_path,
    const std::string& text_path,
    std::size_t new_flush_interval,
    bool gpu_enabled,
    bool cuda_enabled)
{
    std::cerr << "\n  [TDR] GPU 设备已丢失 (VK_ERROR_DEVICE_LOST)\n";
    std::cerr << "  [TDR] 尝试保存 checkpoint...\n";
    auto save_r = nn::save_model(save_path, model, spec, tokenizer_json);
    if (save_r)
        std::cerr << "  [TDR] 模型已保存到: " << save_path << "\n";
    else
        std::cerr << "  [TDR] 保存失败: " << save_r.error().message << "\n";

    // 重建命令行：加入 --resume 和增大 flush_interval（不降 batch_size）
    std::ostringstream oss;
    oss << "\"" << program_name << "\" \"" << text_path
        << "\" --resume \"" << save_path
        << "\" --flush-interval " << new_flush_interval;
    if (gpu_enabled) oss << " --gpu";
    if (cuda_enabled) oss << " --cuda";

    std::string cmd = oss.str();
    std::cerr << "\n  [TDR] 等待 GPU 驱动恢复 (5 秒)...\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cerr << "  [TDR] 自动重启: " << cmd << "\n\n";
    std::cerr.flush();

    std::system(cmd.c_str());
    std::exit(0);
}

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
        << "  --test-file <path> 测试集文件路径（可选，每 epoch 结束后评估 test loss）\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 32)\n"
        << "  --accum-steps <n>  梯度累积步数 (默认: 1)。每 n 步 forward/backward\n"
        << "                     累加梯度后再更新参数，等效放大 batch_size×n\n"
        << "  --seq-len <n>      序列长度 (默认: 256)\n"
        << "  --stride <n>       滑动窗口步长 (默认: 等于 --seq-len，即不重叠)\n"
        << "                     设小可产生重叠窗口，增加训练样本数\n"
        << "  --optimizer <name> 优化器: sgd/sgd_momentum/adam/adamw/muon (默认: adam)\n"
        << "  --weight-decay <w> AdamW 权重衰减系数 (默认: 0.01)\n"
        << "  --d-model <n>      模型维度 (默认: 128)\n"
        << "  --num-heads <n>    注意力头数 (默认: 4)\n"
        << "  --num-layers <n>   Transformer 层数 (默认: 4)\n"
        << "  --d-ff <n>         FFN 中间维度 (默认: 512)\n"
        << "  --gpu              启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --cuda             启用 CUDA GPU 加速 (需要 CUDA Toolkit)\n"
        << "  --positional-encoding <type>\n"
        << "                     位置编码类型: learned(默认)/sinusoidal/alibi\n"
        << "                     learned: 可学习位置嵌入（默认，GPT 原版）\n"
        << "                     sinusoidal: 正弦波固定位置编码（不参与训练）\n"
        << "                     alibi: 线性偏置注意力（无位置嵌入，支持长度外推）\n"
        << "  --activation <type>  FFN 激活: gelu(默认)/swiglu\n"
        << "                     gelu: QuickGeLU（GPT-2 风格）\n"
        << "                     swiglu: SwiGLU（LLaMA/Mistral 风格，每参数效率更高）\n"
        << "  --norm <type>      归一化层: layernorm(默认)/rmsnorm\n"
        << "                     layernorm: LayerNorm（GPT-2 风格）\n"
        << "                     rmsnorm: RMSNorm（LLaMA/Mistral 风格，更快更稳）\n"
        << "  --log-interval <n> 每隔多少 step 显示进度 (默认: 50)\n"
        << "  --save-interval <n> 每隔多少 step 保存 checkpoint (默认: 100)\n"
        << "  --grad-log         显示梯度统计（范数/最大值/均值）\n"
        << "\n"
        << "TDR 防护:\n"
        << "  --tdr-retry <on|off>  GPU 超时自动减小 batch 重试 (默认: on)\n"
        << "  --max-tdr-retries <n> 最大重试次数，每次 batch 减半 (默认: 4)\n"
        << "\n"
        << "Batch 录制粒度:\n"
        << "  --flush-interval <n>  每 N 个 Transformer block flush 一次 (默认: 0=不间断)\n"
        << "  增大此值可拆分大提交防 TDR，不影响 batch_size 和训练质量\n"
        << "  设备丢失 (VK_ERROR_DEVICE_LOST) 时自动保存 checkpoint 并退出，\n"
        << "  可用 --resume 恢复训练。\n"
        << "\n"
        << "学习率调度:\n"
        << "  --lr-schedule <type> 学习率调度: fixed/cosine/step_cosine (默认: fixed)\n"
        << "                   cosine: 余弦退火（epoch 级），lr 从初始值衰减到 min-lr\n"
        << "                   step_cosine: 余弦退火（step 级），按单个训练步预热+退火，\n"
        << "                   适合每 epoch 步数很多的场景（如大语料）\n"
        << "  --warmup-epochs <n> 线性预热轮数 (默认: 0, 仅 cosine)\n"
        << "  --warmup-steps <n>  线性预热步数 (默认: 0, 仅 step_cosine)\n"
        << "  --min-lr <lr>     余弦退火最低学习率 (默认: 1e-6)\n"
        << "  --lr-per-epoch <v1,v2,...>  手动指定每轮学习率 (逗号分隔，优先级最高)\n"
        << "  --max-norm <f>    梯度裁剪最大全局 L2 范数 (默认: 0=不裁剪)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string text_path;
    std::string test_path;   // 测试集文件路径（可选，用于每 epoch 结束后评估）
    std::string save_path = "gpt_model.bin";
    std::string vocab_path = "gpt_bpe.json";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    Scalar lr = 0.001;
    Scalar weight_decay = 0.01f;  // AdamW 权重衰减系数
    std::size_t batch_size = 32;
    std::size_t accum_steps = 1;    // 梯度累积步数（1 = 不累积）
    std::size_t seq_len = nn::GPT_SEQ_LEN;
    std::size_t stride = 0;   // 滑动窗口步长，0 = 默认等于 seq_len（不重叠）
    std::size_t d_model = nn::GPT_D_MODEL;
    std::size_t num_heads = nn::GPT_NUM_HEADS;
    std::size_t num_layers = nn::GPT_NUM_LAYERS;
    std::size_t d_ff = nn::GPT_D_FF;
    std::size_t log_interval = 50;
    std::size_t save_interval = 100;  // checkpoint 保存间隔（独立于 log_interval）
    bool load_existing = false;
    bool gpu_enabled = false;
    bool cuda_enabled = false;
    bool grad_log = false;          // 显示梯度统计
    nn::PosEncodingType pos_encoding = nn::PosEncodingType::Learned;
    nn::ActivationType activation = nn::ActivationType::GeLU;  // FFN 激活
    nn::NormType norm_type = nn::NormType::LayerNorm;           // 归一化层类型

    // TDR 自动重试
    bool auto_tdr_retry = true;              // 遇到 TDR 超时自动减小 batch 重试
    std::size_t max_tdr_retries = 4;         // 最大重试次数（每次 batch 减半）

    // batch 录制粒度：在 Transformer block 间按间隔 flush，拆分大提交
    std::size_t flush_interval = 0;          // 0=不间断（默认），>0=每 N 个 block flush

    // 学习率调度
    std::string lr_schedule = "fixed";  // fixed / cosine / step_cosine
    int warmup_epochs = 0;              // 线性预热轮数（epoch 级）
    std::size_t warmup_steps = 0;       // 线性预热步数（step 级）
    Scalar min_lr = 1e-6f;              // 余弦退火最低 lr
    std::vector<Scalar> lr_per_epoch;   // 手动指定每轮 lr（为空则自动计算）

    // 梯度裁剪
    Scalar max_norm = 0.0f;             // 0 = 不裁剪
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
        else if (arg == "--accum-steps" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --accum-steps: " << v.error().message << "\n"; std::exit(1); }
            if (*v == 0) { std::cerr << "--accum-steps 必须 >= 1\n"; std::exit(1); }
            cfg.accum_steps = *v;
        }
        else if (arg == "--seq-len" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --seq-len: " << v.error().message << "\n"; std::exit(1); }
            cfg.seq_len = *v;
        }
        else if (arg == "--stride" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --stride: " << v.error().message << "\n"; std::exit(1); }
            cfg.stride = *v;
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
        else if (arg == "--save-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --save-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.save_interval = *v;
        }
        else if (arg == "--gpu")
            cfg.gpu_enabled = true;
        else if (arg == "--cuda")
            cfg.cuda_enabled = true;
        else if (arg == "--grad-log")
            cfg.grad_log = true;
        else if (arg == "--lr-schedule" && i + 1 < argc)
        {
            cfg.lr_schedule = argv[++i];
            if (cfg.lr_schedule != "fixed" && cfg.lr_schedule != "cosine" &&
                cfg.lr_schedule != "step_cosine")
            {
                std::cerr << "未知 --lr-schedule: " << cfg.lr_schedule
                          << "，可选: fixed, cosine, step_cosine\n";
                std::exit(1);
            }
        }
        else if (arg == "--warmup-epochs" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --warmup-epochs: " << v.error().message << "\n"; std::exit(1); }
            cfg.warmup_epochs = *v;
        }
        else if (arg == "--warmup-steps" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --warmup-steps: " << v.error().message << "\n"; std::exit(1); }
            cfg.warmup_steps = *v;
        }
        else if (arg == "--min-lr" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --min-lr: " << v.error().message << "\n"; std::exit(1); }
            cfg.min_lr = *v;
        }
        else if (arg == "--lr-per-epoch" && i + 1 < argc)
        {
            std::string dims_str = argv[++i];
            std::stringstream ss(dims_str);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                auto v = nn::parse_number<Scalar>(token);
                if (!v) { std::cerr << "无效 --lr-per-epoch 值: " << v.error().message << "\n"; std::exit(1); }
                cfg.lr_per_epoch.push_back(*v);
            }
        }
        else if (arg == "--max-norm" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-norm: " << v.error().message << "\n"; std::exit(1); }
            cfg.max_norm = *v;
        }
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
        else if (arg == "--activation" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "gelu")
                cfg.activation = nn::ActivationType::GeLU;
            else if (v == "swiglu")
                cfg.activation = nn::ActivationType::SwiGLU;
            else
            {
                std::cerr << "未知激活类型: " << v
                          << "，可选: gelu, swiglu\n";
                std::exit(1);
            }
        }
        else if (arg == "--norm" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "layernorm")
                cfg.norm_type = nn::NormType::LayerNorm;
            else if (v == "rmsnorm")
                cfg.norm_type = nn::NormType::RMSNorm;
            else
            {
                std::cerr << "未知归一化层类型: " << v
                          << "，可选: layernorm, rmsnorm\n";
                std::exit(1);
            }
        }
        else if (arg == "--tdr-retry" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "on" || v == "1" || v == "true")
                cfg.auto_tdr_retry = true;
            else if (v == "off" || v == "0" || v == "false")
                cfg.auto_tdr_retry = false;
            else
            {
                std::cerr << "无效 --tdr-retry: " << v << "，可选: on, off\n";
                std::exit(1);
            }
        }
        else if (arg == "--max-tdr-retries" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-tdr-retries: " << v.error().message << "\n"; std::exit(1); }
            cfg.max_tdr_retries = *v;
        }
        else if (arg == "--flush-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --flush-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.flush_interval = *v;
        }
        else if (arg == "--test-file" && i + 1 < argc)
            cfg.test_path = argv[++i];
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
void log_gradient_stats(nn::ComputeEngine &engine, const std::vector<nn::TensorRef> &grads)
{
    Scalar global_sum_sq = 0.0;
    Scalar global_abs_max = 0.0;
    Scalar global_abs_sum = 0.0;
    std::size_t global_count = 0;
    std::size_t tensor_idx = 0;

    for (auto& grad_ref : grads)
    {
        const auto& grad = grad_ref.get();
        // GPU 张量需要下载到 CPU
        nn::Matrix mat;
        if (grad.device() == nn::Device::GPU)
        {
            auto m = engine.to_matrix(grad);
            if (!m) continue;
            mat = std::move(*m);
        }
        else
        {
            mat = grad.cpu_matrix();
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
    const std::string program_name = argv[0];
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

    // ── 滑动窗口数据组织（GPT 预训练标准做法）────────────────────────
    // 每行 = 一个文档，编码为 [BOS] + tokens + [EOS]，然后拼接成一条连续
    // token 流（行边界通过 EOS 编码进流，窗口可跨行，上下文连续）。
    // 训练样本 = 对 token 流按 stride 滑动切出的 seq_len 窗口：
    //   - 不再"每行一个样本"：长行不再被截断丢弃、短行不再大量 PAD，
    //     语料被 100% 利用；
    //   - 窗口可跨行，模型能学到跨行/跨段依赖；
    //   - 仅最后一个窗口不足 seq_len 时 PAD，且 loss 屏蔽 PAD 位置。
    const std::size_t pad_id = tokenizer->pad_id();
    const std::size_t stride = (cfg.stride == 0) ? cfg.seq_len : cfg.stride;
    std::vector<std::size_t> token_flow;      // 全量 token 流（含 BOS/EOS）
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
            if (line_tokens.empty()) continue;
            if (bos_id != nn::Tokenizer::npos)
                token_flow.push_back(bos_id);
            token_flow.insert(token_flow.end(), line_tokens.begin(), line_tokens.end());
            if (eos_id != nn::Tokenizer::npos)
                token_flow.push_back(eos_id);
        }
    }
    // 切窗口：每个窗口 = 一个训练样本（长度 seq_len，末窗不足则 PAD）
    std::vector<std::size_t> window_offsets;  // 每个窗口在 token_flow 中的起始偏移
    window_offsets.reserve(token_flow.size() / stride + 1);
    for (std::size_t pos = 0; pos < token_flow.size(); pos += stride)
        window_offsets.push_back(pos);
    std::cout << "文本 token 流: " << token_flow.size() << " tokens (含 BOS/EOS), "
              << tokenizer->vocab_size() << " 词表\n"
              << "滑动窗口: seq_len=" << cfg.seq_len << " stride=" << stride
              << " 样本数=" << window_offsets.size() << "\n" << std::endl;

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

    // ── 创建计算引擎 ─────────────────────────────────────────
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = cfg.gpu_enabled;
    eng_cfg.use_cuda = cfg.cuda_enabled;
    auto engine = nn::cli::create_engine(eng_cfg, std::cout);

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    auto model_build = nn::build_gpt_model(
        *engine,
        tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
        cfg.num_heads, cfg.d_ff, cfg.num_layers,
        cfg.pos_encoding, cfg.activation, cfg.norm_type);
    if (!model_build) {
        std::cerr << "构建模型失败: " << model_build.error().message << '\n';
        return 1;
    }
    auto model = std::move(*model_build);

    // ── 设置 batch 录制粒度 ──
    model.set_flush_interval(cfg.flush_interval);

    // ── 构建规格（用于保存） ─────────────────────────────────
    auto spec = nn::make_gpt_spec(
        tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
        cfg.num_heads, cfg.d_ff, cfg.num_layers,
        cfg.pos_encoding, cfg.activation, cfg.norm_type);

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
            // 统一的 GPTModel 通过 pos_encoding 区分 Learned/Sinusoidal/ALiBi，
            // 因此 GPT 和旧格式 ALiBi_GPT 文件都走同一条构建路径。
            if (file_spec.is_gpt() || file_spec.is_alibi_gpt())
            {
                if (file_spec.pos_encoding == nn::PosEncodingType::ALiBi)
                    std::cout << "从模型文件读取 ALiBi GPT 规格\n";
                else
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
    if (!optimizer)
    {
        std::cerr << "错误：未知优化器名称: " << cfg.optimizer_name << "\n";
        return 1;
    }

    Scalar optimizer_current_lr = cfg.lr;

    // ── 学习率调度配置（委托给 nn::cli::compute_epoch_lr） ──
    nn::cli::LrScheduleConfig lr_sched_cfg;
    lr_sched_cfg.base_lr = cfg.lr;
    lr_sched_cfg.min_lr = cfg.min_lr;
    lr_sched_cfg.warmup_epochs = cfg.warmup_epochs;
    lr_sched_cfg.total_epochs = cfg.epochs;
    lr_sched_cfg.schedule = cfg.lr_schedule;
    lr_sched_cfg.lr_per_epoch = cfg.lr_per_epoch;

    nn::CrossEntropyLoss ce_loss;

    // ── 训练循环 ─────────────────────────────────────────────
    // 每样本 = 一个 seq_len 滑动窗口（可能跨行），目标 = 输入左移一位。
    if (window_offsets.empty())
    {
        std::cerr << "无有效训练样本（文本 token 流为空）\n";
        return 1;
    }
    std::cout << "训练窗口样本数: " << window_offsets.size() << "\n" << std::endl;

    // ── 加载测试集（可选，与训练一致的滑动窗口处理） ─────────────
    std::vector<std::size_t> test_window_offsets;
    std::vector<std::size_t> test_flow;
    if (!cfg.test_path.empty())
    {
        auto test_text_result = nn::load_text_file(cfg.test_path);
        if (!test_text_result)
        {
            std::cerr << "加载测试集失败: " << test_text_result.error().message << '\n';
            return 1;
        }
        std::string test_text = std::move(*test_text_result);
        std::size_t test_token_count = 0;
        {
            std::istringstream tiss(test_text);
            std::string tline;
            while (std::getline(tiss, tline))
            {
                auto tlpos = tline.find_first_not_of(" \t\r\n");
                if (tlpos == std::string::npos) continue;
                tline = tline.substr(tlpos, tline.find_last_not_of(" \t\r\n") - tlpos + 1);
                if (tline.empty()) continue;

                auto tl_tokens = tokenizer->encode(tline);
                if (tl_tokens.empty()) continue;
                if (bos_id != nn::Tokenizer::npos)
                    test_flow.push_back(bos_id);
                test_flow.insert(test_flow.end(), tl_tokens.begin(), tl_tokens.end());
                if (eos_id != nn::Tokenizer::npos)
                    test_flow.push_back(eos_id);
                test_token_count += tl_tokens.size() + 2;
            }
        }
        for (std::size_t pos = 0; pos < test_flow.size(); pos += stride)
            test_window_offsets.push_back(pos);
        std::cout << "测试集: " << cfg.test_path << "  样本数: " << test_window_offsets.size()
                  << "  tokens: " << test_token_count << std::endl;
    }

    // ── 步数与采样：每 epoch 每样本恰好访问一次 ──────────────
    // steps_per_epoch = ceil(样本数 / batch_size)：最后一个 batch 不满时
    // 以实际 this_bs 参与训练。由于 loss 已按有效 token 归一化，不满 batch
    // 的梯度与满 batch 同尺度（每 epoch 仅一个不满 batch，影响可忽略）。
    std::size_t steps_per_epoch =
        (window_offsets.size() + cfg.batch_size - 1) / cfg.batch_size;
    if (steps_per_epoch == 0)
    {
        std::cerr << "样本数 (" << window_offsets.size() << ") 小于 batch_size ("
                  << cfg.batch_size << ")，请减小 --batch-size 或增大训练语料\n";
        return 1;
    }

    // ── Step 级学习率配置（step_cosine 模式） ──
    nn::cli::StepLrScheduleConfig step_lr_cfg;
    step_lr_cfg.base_lr = cfg.lr;
    step_lr_cfg.min_lr = cfg.min_lr;
    step_lr_cfg.warmup_steps = static_cast<int>(cfg.warmup_steps);
    step_lr_cfg.total_steps =
        static_cast<int>(steps_per_epoch * static_cast<std::size_t>(cfg.epochs));
    step_lr_cfg.cosine = true;

    std::mt19937_64 rng{42};
    std::vector<std::size_t> sample_indices(window_offsets.size());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;

    // ── 预分配 batch 缓冲区（末批不满时 resize 到实际 this_bs） ──
    // x_tokens: (seq_len, batch) 输入 token IDs（每窗口一列）
    // y_tokens: (seq_len, batch) 目标 token IDs（x 左移一位）
    // 最后一个窗口不足 seq_len 时用 pad_id 填充，目标对应位置也用 pad_id
    nn::Matrix x_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix y_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix loss_mask(cfg.seq_len, cfg.batch_size);
    std::vector<std::size_t> flat_targets(cfg.seq_len * cfg.batch_size);

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        // ── 学习率调度：每 epoch 开始时调整（step_cosine 由 step 级调度接管） ──
        if (cfg.lr_schedule != "step_cosine")
        {
            Scalar epoch_lr = nn::cli::compute_epoch_lr(lr_sched_cfg, epoch);
            if (epoch_lr != optimizer_current_lr)
            {
                optimizer->set_lr(epoch_lr);
                optimizer_current_lr = epoch_lr;
            }
        }

        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_weighted = 0.0;  // Σ(loss × 有效token)，用于按 token 加权平均
        std::size_t total_valid = 0;  // 累计有效 token 数

        // 每个 epoch 开始前 shuffle 样本索引队列
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);

        // 梯度累积：距上次参数更新的步数（每 accum_steps 步更新一次）
        std::size_t steps_since_update = 0;

        for (std::size_t step = 0; step < steps_per_epoch; ++step)
        {
            // ── step 级学习率：每个训练步更新（step_cosine 模式） ──
            if (cfg.lr_schedule == "step_cosine")
            {
                const int global_step =
                    static_cast<int>(epoch) * static_cast<int>(steps_per_epoch) +
                    static_cast<int>(step);
                Scalar step_lr = nn::cli::compute_step_lr(step_lr_cfg, global_step);
                if (step_lr != optimizer_current_lr)
                {
                    optimizer->set_lr(step_lr);
                    optimizer_current_lr = step_lr;
                }
            }

            // ── 采样 batch：每样本 = 一个滑动窗口 ─────────────
            // 按 shuffle 后的顺序切片取 this_bs 个窗口（末批可能不满）。
            // 每 epoch 每样本恰好访问一次。
            const std::size_t offset = step * cfg.batch_size;
            const std::size_t this_bs = std::min(cfg.batch_size, window_offsets.size() - offset);
            if (x_tokens.cols() != this_bs)
            {
                x_tokens.resize(cfg.seq_len, this_bs);
                y_tokens.resize(cfg.seq_len, this_bs);
                loss_mask.resize(cfg.seq_len, this_bs);
                flat_targets.resize(cfg.seq_len * this_bs);
            }
            std::size_t step_valid = 0;  // 本 step 参与 loss 的有效 token 数

            for (std::size_t b = 0; b < this_bs; ++b)
            {
                const std::size_t win_pos = window_offsets[sample_indices[offset + b]];
                // 窗口有效长度：不超过 token_flow 末尾时为 seq_len，
                // 否则为剩余 token 数（最后一个窗口，PAD 位置被屏蔽）
                const std::size_t win_len = std::min(cfg.seq_len, token_flow.size() - win_pos);

                // 填充 x/y/mask：x[t] = flow[win_pos+t], y[t] = flow[win_pos+t+1]
                for (std::size_t t = 0; t < cfg.seq_len; ++t)
                {
                    const std::size_t x_id = (t < win_len) ? token_flow[win_pos + t] : pad_id;
                    const std::size_t y_id = (t + 1 < win_len) ? token_flow[win_pos + t + 1] : pad_id;
                    x_tokens.set_value_unchecked(t, b, static_cast<Scalar>(x_id));
                    y_tokens.set_value_unchecked(t, b, static_cast<Scalar>(y_id));
                    const bool participate = (t + 1 < win_len);
                    loss_mask.set_value_unchecked(t, b, participate ? 1.0f : 0.0f);
                    if (participate) ++step_valid;
                }
            }

            // ── Matrix → Tensor（上传到引擎设备） ──────────────
            auto x_tensor_r = engine->from_matrix(x_tokens);
            if (!x_tensor_r) {
                std::cerr << "\nfrom_matrix(x_tokens) failed: " << x_tensor_r.error().message << '\n';
                return 1;
            }

            // ── 构造平坦标签（与 logits 列序一致：t×batch + b） ──
            auto y_span = y_tokens.span();
            for (std::size_t t = 0; t < cfg.seq_len; ++t)
                for (std::size_t b = 0; b < this_bs; ++b)
                    flat_targets[t * this_bs + b] =
                        static_cast<std::size_t>(y_span[t * this_bs + b]);

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

            // ── 损失（稀疏标签，避免 one-hot 爆显存） ────────
            auto mask_span = std::span<const Scalar>(loss_mask.span());
            auto loss_result = ce_loss.forward_sparse(
                *engine, logits, flat_targets, mask_span, tokenizer->vocab_size());
            if (!loss_result) { std::cerr << "Error: " << loss_result.error().message << '\n'; return 1; }
            Scalar loss = *loss_result;
            total_weighted += loss * step_valid;        // 按有效 token 加权
            total_valid += step_valid;

            // ── 中点刷新：提交 forward+loss，拆分为两次 GPU 提交 ──
            // 大词表 + 长序列时 forward+backward 单次提交可能触发 TDR 超时。
            // 在 forward 与 backward 之间 flush，将一次大提交拆为两次小提交。
            auto flush_r = engine->flush_batch();
            if (!flush_r) {
                std::string err_msg = flush_r.error().message;
                bool is_device_lost = err_msg.find("VK_ERROR_DEVICE_LOST") != std::string::npos;
                std::cerr << "\nflush_batch (forward) failed: " << err_msg;
                if (is_device_lost)
                {
                    restart_on_device_lost(program_name, model, spec, tokenizer_json,
                        cfg.save_path, cfg.text_path,
                        cfg.flush_interval == 0 ? std::size_t{1} : cfg.flush_interval * 2,
                        cfg.gpu_enabled, cfg.cuda_enabled);
                }
                return 1;
            }
            // ── 反向传播（梯度已含 mask，无需额外处理） ────────
            auto grad_result = ce_loss.backward();
            if (!grad_result) { std::cerr << "\nLoss backward failed: " << grad_result.error().message << '\n'; return 1; }

            auto bwd_result = model.backward(*grad_result);
            if (!bwd_result) { std::cerr << "Error: " << bwd_result.error().message << '\n'; return 1; }

            // ── 提交 backward batch（单独一次提交，已与 forward 拆分） ──
            auto bwd_end = engine->end_batch();
            if (!bwd_end) {
                std::string err_msg = bwd_end.error().message;
                bool is_device_lost = err_msg.find("VK_ERROR_DEVICE_LOST") != std::string::npos;
                std::cerr << "\nend_batch (backward) failed: " << err_msg;
                if (is_device_lost)
                {
                    restart_on_device_lost(program_name, model, spec, tokenizer_json,
                        cfg.save_path, cfg.text_path,
                        cfg.flush_interval == 0 ? std::size_t{1} : cfg.flush_interval * 2,
                        cfg.gpu_enabled, cfg.cuda_enabled);
                }
                std::cerr << '\n';
                return 1;
            }

            // ── 梯度累积：每 accum_steps 步才更新一次参数 ──
            // forward+backward 每步都执行（梯度累加到参数梯度），
            // 梯度裁剪/step/zero_grad 仅在累积到 accum_steps 或 epoch 末尾执行。
            ++steps_since_update;
            const bool do_update =
                (steps_since_update >= cfg.accum_steps) || (step + 1 == steps_per_epoch);

            if (do_update)
            {
                // ── 梯度裁剪（在 step() 之前，backward() 之后） ──
                if (cfg.max_norm > 0)
                {
                    auto clip_r = optimizer->clip_grad_norm(cfg.max_norm);
                    if (!clip_r) {
                        std::cerr << "\n梯度裁剪失败: " << clip_r.error().message << '\n';
                        return 1;
                    }
                }

                auto opt_begin = engine->begin_batch();
                if (!opt_begin) {
                    std::cerr << "begin_batch (optimizer) failed: " << opt_begin.error().message << '\n';
                    return 1;
                }

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
                    std::cerr << "end_batch (optimizer) failed: " << end_r.error().message << '\n';
                    return 1;
                }

                steps_since_update = 0;
            }

            // ── 定期保存 checkpoint（独立于 log_interval） ──
            if ((step + 1) % cfg.save_interval == 0 || step + 1 == steps_per_epoch)
            {
                auto save_r = nn::save_model(cfg.save_path, model, spec, tokenizer_json);
                if (!save_r)
                    std::cerr << "\n  [ckpt] 保存失败: " << save_r.error().message << "\n";
            }

            // ── 进度显示
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
        // 按有效 token 加权平均（等价于全局 per-token 平均），
        // 避免"有效 token 少的 batch"拉偏 epoch loss 报告
        Scalar avg_loss = (total_valid > 0)
            ? total_weighted / static_cast<Scalar>(total_valid)
            : Scalar{0};

        std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                  << "  lr=" << std::scientific << std::setprecision(4) << optimizer_current_lr
                  << "  avg_loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  time=" << std::setprecision(1) << ep_sec << "s";

        // ── 测试集评估（可选，与训练一致的滑动窗口） ─────────────
        if (!test_window_offsets.empty())
        {
            Scalar test_total_weighted = 0.0;
            std::size_t test_total_valid = 0;
            const std::size_t test_bs = std::min(cfg.batch_size, test_window_offsets.size());
            const std::size_t test_steps_per_epoch =
                (test_window_offsets.size() + test_bs - 1) / test_bs;

            for (std::size_t tstep = 0; tstep < test_steps_per_epoch; ++tstep)
            {
                const std::size_t toffset = tstep * test_bs;
                const std::size_t this_bs = std::min(test_bs, test_window_offsets.size() - toffset);
                if (x_tokens.cols() != this_bs)
                {
                    x_tokens.resize(cfg.seq_len, this_bs);
                    y_tokens.resize(cfg.seq_len, this_bs);
                    loss_mask.resize(cfg.seq_len, this_bs);
                    flat_targets.resize(cfg.seq_len * this_bs);
                }

                // 填充 batch（与训练一致：末窗不足时 PAD 并屏蔽）
                std::size_t test_valid = 0;
                for (std::size_t b = 0; b < this_bs; ++b)
                {
                    const std::size_t win_pos = test_window_offsets[toffset + b];
                    const std::size_t win_len = std::min(cfg.seq_len, test_flow.size() - win_pos);
                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                    {
                        const std::size_t x_id = (t < win_len) ? test_flow[win_pos + t] : pad_id;
                        const std::size_t y_id = (t + 1 < win_len) ? test_flow[win_pos + t + 1] : pad_id;
                        x_tokens.set_value(t, b, static_cast<Scalar>(x_id));
                        y_tokens.set_value(t, b, static_cast<Scalar>(y_id));
                        const bool participate = (t + 1 < win_len);
                        loss_mask.set_value(t, b, participate ? 1.0f : 0.0f);
                        if (participate) ++test_valid;
                    }
                }

                // Forward pass only（不 backward）
                auto x_tensor_r = engine->from_matrix(x_tokens);
                if (!x_tensor_r) { std::cerr << "  测试 from_matrix 失败: " << x_tensor_r.error().message << '\n'; break; }

                // 构建 flat_targets（与 logits 列序一致：t×batch + b）
                auto y_span = y_tokens.span();
                for (std::size_t t = 0; t < cfg.seq_len; ++t)
                    for (std::size_t b = 0; b < this_bs; ++b)
                        flat_targets[t * this_bs + b] =
                            static_cast<std::size_t>(y_span[t * this_bs + b]);

                auto begin_r = engine->begin_batch();
                if (!begin_r) { std::cerr << "  测试 begin_batch 失败: " << begin_r.error().message << '\n'; break; }

                auto fwd_result = model.forward(*x_tensor_r);
                if (!fwd_result) { std::cerr << "  测试前向传播出错: " << fwd_result.error().message << '\n'; break; }
                auto test_mask_span = std::span<const Scalar>(loss_mask.span());
                auto loss_result = ce_loss.forward_sparse(
                    *engine, *fwd_result, flat_targets, test_mask_span, tokenizer->vocab_size());

                auto end_r = engine->end_batch();
                if (!end_r) { std::cerr << "  测试 end_batch 失败: " << end_r.error().message << '\n'; break; }

                if (!loss_result) { std::cerr << "  测试评估出错: " << loss_result.error().message << '\n'; break; }
                Scalar batch_loss = *loss_result;

                // 按有效 token 加权（与训练端 per-token 平均一致）
                test_total_weighted += batch_loss * static_cast<Scalar>(test_valid);
                test_total_valid += test_valid;
            }

            Scalar test_avg_loss = (test_total_valid > 0)
                ? test_total_weighted / static_cast<Scalar>(test_total_valid)
                : Scalar{0};
            std::cout << "  test_loss=" << std::fixed << std::setprecision(4) << test_avg_loss;
        }

        std::cout << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    Scalar total_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

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
