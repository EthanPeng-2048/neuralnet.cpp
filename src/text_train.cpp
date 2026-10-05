// ── GPT 文本生成训练程序（引擎化架构） ──────────────────────────────────────
//
// 输入：.nndataset 数据集（docs/development/19-unified-dataset.md）——
//   tokenize、loss 掩码、词表都在生成期（dataset_gen）定好并嵌入数据集。
//
// 数据流（滑动窗口，GPT 预训练标准做法）：
//   nn::Dataset::load_text → token 流 + doc_ids + loss_mask（与旧 tokcache
//     + mark_assistant_spans 产物逐位同构）→ 各 doc token 纯拼接成连续 token 流
//     （行间无分隔符、不插入 BOS/EOS；行边界由 doc_ids 记录，窗口可跨行，
//       上下文连续）
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
#include <neuralnet.cpp/precision.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/cli/cli_lr_scheduler.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>
#include <neuralnet.cpp/cli/cli_help.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
#include <filesystem>
#include <future>
#include <optional>

using nn::Scalar;

namespace fs = std::filesystem;

// ── 数据来源：.nndataset（token 流/掩码在生成期已定好）─────────────────
// 19 号设计（docs/development/19-unified-dataset.md）落地后，文本读取/trim、
// 并行 tokenize、assistant 掩码扫描、.tokcache 全部移出本入口：
//   生成侧 = dataset_gen（nn::read_text_docs / nn::encode_docs_parallel /
//             nn::mark_assistant_span），读取侧 = nn::Dataset（只读）。
// 本入口只剩：拼流 → 滑窗 → 训练（窗口化是训练策略，裁决 #3）。

// ── 精度解析辅助 ─────────────────────────────────────────────────────────
nn::Precision parse_precision(const std::string& name, const char* flag)
{
    if (name == "f16" || name == "half") return nn::Precision::F16;
    if (name == "f32" || name == "float") return nn::Precision::F32;
    std::cerr << "无效 --" << flag << ": " << name << "，可选: f16, f32\n";
    std::exit(1);
}

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    nn::cli::Help help(std::cout, prog, "GPT 文本生成训练程序");

    help.usage("<dataset.nndataset> [选项]");

    help.section("参数");
    help.item("<dataset.nndataset>", "数据集文件路径 (.nndataset，由 dataset_gen 生成；\n词表/loss 掩码/test 子集都在生成期定好)");

    help.section("选项");
    help.opt("--save <path>", "模型保存路径 (默认: gpt_model.bin)");
    help.opt("--resume <path>", "从已有模型恢复训练");
    help.opt("--resume-epoch <n>", "从第 n 个 epoch 继续 (0-based，需配合 --resume；默认 0)");
    help.opt("--resume-step <n>", "从本 epoch 内第 n 步继续 (0-based，需配合 --resume；默认 0)");
    help.opt("--epochs <n>", "训练轮数 (默认: 10)");
    help.opt("--lr <lr>", "学习率 (默认: 0.001)");
    help.opt("--batch-size <n>", "批大小 (默认: 32)");
    help.opt("--accum-steps <n>", "梯度累积步数 (默认: 1)\n每 n 步 forward/backward 累加梯度后再更新参数，等效放大 batch_size×n");
    help.opt("--seq-len <n>", "序列长度 (默认: 256)");
    help.opt("--stride <n>", "滑动窗口步长 (默认: 等于 --seq-len，即不重叠)\n设小可产生重叠窗口，增加训练样本数");
    help.opt("--optimizer <name>", "优化器: sgd/sgd_momentum/adam/adamw/muon (默认: adam)");
    help.opt("--weight-decay <w>", "AdamW 权重衰减系数 (默认: 0.01)");
    help.opt("--beta1 <b>", "Adam/AdamW 一阶动量衰减 β1 (默认: 0.9，sgd/muon 忽略)");
    help.opt("--beta2 <b>", "Adam/AdamW 二阶动量衰减 β2 (默认: 0.999，sgd/muon 忽略)\nLLM 预训练惯例常取 0.95 (GPT-3/LLaMA 配方)");
    help.opt("--max-norm <f>", "梯度裁剪最大全局 L2 范数 (默认: 0=不裁剪)");
    help.opt("--log-interval <n>", "每隔多少 step 显示进度 (默认: 50)");
    help.opt("--save-interval <n>", "每隔多少 step 保存 checkpoint (默认: 100)");
    help.opt("--max-steps <n>", "本次运行最多训练多少 step 后停止并保存 (默认: 0=不限)\n按本进程执行的步数计，可与 --resume 组合分段跑；用于吞吐实测/分段训练");
    help.opt("--grad-log", "显示梯度统计 (范数/最大值/均值)");
    help.opt("--gpu [索引|名称]", "启用 GPU 加速 (需要 Vulkan SDK)\n空格形式只收枚举索引；名称子串用 --gpu=<名称>，如 --gpu=NVIDIA / --gpu=40HX");
    help.opt("--help, -h", "显示此帮助信息");

    help.section("模型结构");
    help.opt("--model <type>", "模型架构: gpt/rapt (默认: gpt)\ngpt: 标准 Transformer 语言模型\nrapt: ReLU 线性注意力 (RLA) 语言模型，动态稀疏检索，\nO(L·d²) 复杂度；强制 RoPE (ReLU 前施加)");
    help.opt("--positional-encoding <type>", "位置编码类型: learned/sinusoidal/alibi/rope (默认: learned)\nlearned: 可学习位置嵌入 (GPT 原版)\nsinusoidal: 正弦波固定位置编码 (不参与训练)\nalibi: 线性偏置注意力 (无位置嵌入，支持长度外推)\nrope: 旋转位置编码 (现代方案，在注意力 Q/K 上施加)");
    help.opt("--activation <type>", "FFN 激活: gelu/swiglu (默认: gelu)\ngelu: QuickGeLU (GPT-2 风格)\nswiglu: SwiGLU (LLaMA/Mistral 风格，每参数效率更高)");
    help.opt("--norm <type>", "归一化层: layernorm/rmsnorm (默认: layernorm)\nlayernorm: LayerNorm (GPT-2 风格)\nrmsnorm: RMSNorm (LLaMA/Mistral 风格，更快更稳)");
    help.opt("--d-model <n>", "模型维度 (默认: 128)");
    help.opt("--num-heads <n>", "注意力头数 (默认: 4)");
    help.opt("--num-layers <n>", "Transformer 层数 (默认: 4)");
    help.opt("--d-ff <n>", "FFN 中间维度 (默认: 512)");

    help.section("学习率调度");
    help.opt("--lr-schedule <type>", "学习率调度: fixed/cosine/step_cosine (默认: fixed)\ncosine: 余弦退火 (epoch 级)，lr 从初始值衰减到 min-lr\nstep_cosine: 余弦退火 (step 级)，按单个训练步预热+退火，\n适合每 epoch 步数很多的场景 (如大语料)");
    help.opt("--warmup-epochs <n>", "线性预热轮数 (默认: 0，仅 cosine)");
    help.opt("--warmup-steps <n>", "线性预热步数 (默认: 0，仅 step_cosine)");
    help.opt("--min-lr <lr>", "余弦退火最低学习率 (默认: 1e-6)");
    help.opt("--lr-per-epoch <v1,v2,...>", "手动指定每轮学习率 (逗号分隔，优先级最高)");

    help.section("显存优化");
    help.opt("--checkpoint-every <n>", "每 N 个 Transformer block 重算一次 forward (激活重计算，默认: 0=不启用)\n1=每块都重算，显存收益最大，以约 1 次额外前向\nFLOPs 为代价省去整层激活驻留");
    help.opt("--activation-offload", "把每块激活搬 host-visible，backward 拷回\n不重算，FLOPs 保持 1.0×，代价是 PCIe 传输；可与\n--checkpoint-every 混合使用：checkpoint 块重算、其余块 offload；仅 GPU 有效");
    help.note("GPU 训练每 step 末尾自动归还完全空闲的内存池底材 (L2 整块释放)。");

    help.section("Batch 录制粒度");
    help.opt("--flush-interval <n>", "每 N 个 Transformer block flush 一次 (默认: 1)");
    help.note("按层切 batch 缩短 D1 延迟销毁锁窗、拆分大提交防 TDR；\n非阻塞提交后细粒度 flush 的额外 submit 代价不在关键路径，\n不影响 batch_size 和训练质量");

    help.section("混合精度 (docs/development/05-mixed-precision.md)");
    help.opt("--f16", "快捷方式: f16 存储 (param/compute=F16，stable/optimizer=F32)");
    help.opt("--precision-param <f16|f32>", "权重/参数存储精度 (默认: f32)");
    help.opt("--precision-compute <f16|f32>", "常规算子计算精度 (matmul/逐元素/gather，默认: f32)");
    help.opt("--precision-stable <f16|f32>", "数值敏感算子精度 (softmax/LayerNorm/loss，默认: f32)");
    help.opt("--precision-optimizer <f16|f32>", "优化器状态精度 (Adam m/v，默认: f32)");
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    std::string text_path;
    std::string save_path = "gpt_model.bin";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    int start_epoch = 0;          // resume 后从第几个 epoch 继续（0-based，默认从头）
    std::size_t start_step = 0;   // resume 后本 epoch 内从第几步继续（0-based）
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
    std::string model_type = "gpt";   // gpt / rapt（zipt 已于 2026-10-01 移除）
    std::size_t log_interval = 50;
    std::size_t save_interval = 100;  // checkpoint 保存间隔（独立于 log_interval）
    std::size_t max_steps = 0;        // 本进程最多训练步数（0 = 不限，按 epoch 跑完）
    Scalar beta1 = 0.9f;              // Adam/AdamW 一阶动量衰减（sgd/muon 忽略）
    Scalar beta2 = 0.999f;            // Adam/AdamW 二阶动量衰减（LLM 预训练常取 0.95）
    bool load_existing = false;
    bool gpu_enabled = false;
    std::string gpu_device;         // --gpu 的可选设备选择子（空 = 自动选卡）
    bool grad_log = false;          // 显示梯度统计
    nn::PosEncodingType pos_encoding = nn::PosEncodingType::Learned;
    nn::ActivationType activation = nn::ActivationType::GeLU;  // FFN 激活
    nn::NormType norm_type = nn::NormType::LayerNorm;           // 归一化层类型

    // batch 录制粒度：在 Transformer block 间按间隔 flush，拆分大提交。
    // 探针实测：flush 粒度是显存与速度的双重杠杆——
    // 帧越细，"已析构但等帧 reap"的死内存越少（中途 reap 更早生效）。
    //   flush1 = 3914MiB/4.6s，flush2 = 4170MiB/4.8s，flush4 = 5488MiB/6.2s
    // 故默认 flush_interval = 1（更省显存且更快；TDR 拆分粒度也更细）。
    std::size_t flush_interval = 1;          // 0=不间断，>0=每 N 个 block flush

    // 梯度检查点（激活重计算 L1）：每 N 个 GPTBlock 重算一次
    std::size_t checkpoint_every = 0;        // 0=不启用（默认），>0=每 N 个 block 重算

    // activation offload（L1-offload）：把每块内部激活搬 host-visible，backward 拷回
    bool activation_offload = false;         // false=不启用（默认）

    // 学习率调度
    std::string lr_schedule = "fixed";  // fixed / cosine / step_cosine
    int warmup_epochs = 0;              // 线性预热轮数（epoch 级）
    std::size_t warmup_steps = 0;       // 线性预热步数（step 级）
    Scalar min_lr = 1e-6f;              // 余弦退火最低 lr
    std::vector<Scalar> lr_per_epoch;   // 手动指定每轮 lr（为空则自动计算）

    // 梯度裁剪
    Scalar max_norm = 0.0f;             // 0 = 不裁剪

    // 混合精度控制（docs/development/05-mixed-precision.md §9.1）
    nn::PrecisionProfile precision;     // 默认全 F32（零回归）
};

TrainConfig parse_args(int argc, char *argv[])
{
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
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
        else if (arg == "--resume-epoch" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v || *v < 0) { std::cerr << "无效 --resume-epoch: " << argv[i] << "\n"; std::exit(1); }
            cfg.start_epoch = *v;
        }
        else if (arg == "--resume-step" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --resume-step: " << argv[i] << "\n"; std::exit(1); }
            cfg.start_step = *v;
        }
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
            if (*v == 0) { std::cerr << "--batch-size 必须 >= 1\n"; std::exit(1); }
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
        else if (arg == "--beta1" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v || *v <= 0 || *v >= 1) { std::cerr << "--beta1 须在 (0, 1) 区间内\n"; std::exit(1); }
            cfg.beta1 = *v;
        }
        else if (arg == "--beta2" && i + 1 < argc)
        {
            auto v = nn::parse_number<Scalar>(argv[++i]);
            if (!v || *v <= 0 || *v >= 1) { std::cerr << "--beta2 须在 (0, 1) 区间内\n"; std::exit(1); }
            cfg.beta2 = *v;
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
        else if (arg == "--model" && i + 1 < argc)
        {
            cfg.model_type = argv[++i];
            if (cfg.model_type != "gpt" && cfg.model_type != "rapt")
            {
                std::cerr << "未知模型架构: " << cfg.model_type
                          << ", 可选: gpt, rapt\n";
                if (cfg.model_type == "zipt")
                    std::cerr << "（ZiPT 已于 2026-10-01 移除，代码保留在 legacy/zipt 分支；"
                                 "恢复前提见 docs/history.md）\n";
                std::exit(1);
            }
        }
        else if (arg == "--log-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --log-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.log_interval = std::max<std::size_t>(1, *v);  // 0 → 1，避免步进 % 0 除零
        }
        else if (arg == "--save-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --save-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.save_interval = *v;  // 0 = 禁用保存（下方 % 前有 >0 保护）
        }
        else if (arg == "--max-steps" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-steps: " << v.error().message << "\n"; std::exit(1); }
            cfg.max_steps = *v;  // 0 = 不限步数
        }
        else if (auto gpu_dev = nn::cli::parse_gpu_option(argc, argv, i))
        {
            cfg.gpu_enabled = true;
            if (!gpu_dev->empty()) cfg.gpu_device = *gpu_dev;
        }
        else if (arg == "--grad-log")
            cfg.grad_log = true;
        else if (arg == "--f16")
        {
            // 快捷方式：**f16 存储**（param=F16 + compute=F16；stable/optimizer
            // 留 F32）。与 profile_master_weights()（f32 主权重混合）的区别正是
            // param 由 F32 变 F16。stable/optimizer 留 f32 有实测依据（f16 的
            // softmax/loss 链 ~200 步 NaN、Adam 的 m/v f16 下溢发散）：见
            // precision.hpp profile_f16() 注释。
            // 想要四字段全 f16（实验性）：--f16 --precision-stable f16
            //                                  --precision-optimizer f16
            cfg.precision = nn::profile_f16();
        }
        else if (arg == "--precision-param" && i + 1 < argc)
        {
            cfg.precision.param = parse_precision(argv[++i], "precision-param");
        }
        else if (arg == "--precision-compute" && i + 1 < argc)
        {
            cfg.precision.compute = parse_precision(argv[++i], "precision-compute");
        }
        else if (arg == "--precision-stable" && i + 1 < argc)
        {
            cfg.precision.stable = parse_precision(argv[++i], "precision-stable");
        }
        else if (arg == "--precision-optimizer" && i + 1 < argc)
        {
            cfg.precision.optimizer = parse_precision(argv[++i], "precision-optimizer");
        }
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
            else if (v == "rope")
                cfg.pos_encoding = nn::PosEncodingType::RoPE;
            else
            {
                std::cerr << "未知位置编码类型: " << v
                          << "，可选: learned, sinusoidal, alibi, rope\n";
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
        else if (arg == "--flush-interval" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --flush-interval: " << v.error().message << "\n"; std::exit(1); }
            cfg.flush_interval = *v;
        }
        else if (arg == "--checkpoint-every" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --checkpoint-every: " << v.error().message << "\n"; std::exit(1); }
            cfg.checkpoint_every = *v;
        }
        else if (arg == "--activation-offload")
        {
            cfg.activation_offload = true;
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
        std::cerr << "请指定数据集文件 (.nndataset)\n使用 --help 查看用法\n";
        std::exit(1);
    }

    return cfg;
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
        // M1（docs/development/17 §4.1）：张量读取一律经引擎下载到宿主
        // （GPU 直接下载；CPU 同样走 to_matrix，f16 存储也安全）
        nn::Matrix mat;
        {
            auto m = engine.to_matrix(grad);
            if (!m) continue;
            mat = std::move(*m);
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

    // ── 打开数据集（.nndataset；token 流/掩码/词表都在生成期定好）──────
    std::cout << "加载数据集: " << cfg.text_path << " ..." << std::endl;
    auto ds_r = nn::Dataset::open(cfg.text_path);
    NN_EXIT(ds_r, 1, "打开数据集失败: ");
    const nn::Dataset &dataset = *ds_r;
    const std::string &loss_scope = dataset.info().loss_scope;

    // 词表来自数据集配套（内嵌三层 kvrec）；保存模型时原样嵌出（模型 v6）。
    auto tokenizer_r = nn::load_tokenizer_from_kvrec(dataset.vocab_kvrec());
    NN_EXIT(tokenizer_r, 1, "加载数据集内嵌词表失败: ");
    std::unique_ptr<nn::Tokenizer> tokenizer = std::move(*tokenizer_r);
    const std::size_t pad_id = tokenizer->pad_id();
    const std::string tokenizer_bytes = nn::write_nnvocab(dataset.vocab_kvrec());

    // ── 训练子集（与旧 tokcache + mark_assistant_spans 产物逐位同构）──
    // base 模式纯拼接、不插入 BOS/EOS（行间无分隔符），文档边界由
    // doc_ids 表示（每行 = 一篇文档）。
    const std::size_t stride = (cfg.stride == 0) ? cfg.seq_len : cfg.stride;
    auto corpus_r = dataset.load_text(false);
    NN_EXIT(corpus_r, 1, "读取训练子集失败: ");
    std::vector<std::size_t> token_flow = std::move(corpus_r->token_flow);
    std::vector<std::size_t> flow_doc_ids = std::move(corpus_r->doc_ids);
    std::vector<unsigned char> flow_assistant = std::move(corpus_r->loss_mask);
    if (token_flow.empty())
    {
        std::cerr << "数据集 train 子集为空: " << cfg.text_path << "\n";
        return 1;
    }
    std::cout << "训练子集: " << dataset.info().num_docs << " docs / "
              << token_flow.size() << " tokens  loss_scope=" << loss_scope << "\n";

    // 文档感知：doc_ids 非空（每 doc = 一篇文档）即启用块对角掩码。
    if (!flow_doc_ids.empty())
        std::cout << "文档感知掩码已启用（每行 = 一篇文档，"
                  << flow_doc_ids.back() << " 篇文档）\n";

    // loss 掩码在生成期已定（loss_scope=all 为全 1）：仅掩码内的预测目标
    // 参与 loss（对话 SFT 时 = <|assistant|>…<|end_of_assistant|> 段）。
    if (loss_scope == "assistant")
    {
        std::size_t asst_tokens = 0;
        for (unsigned char m : flow_assistant) asst_tokens += m;
        std::cout << "assistant loss 掩码已启用（" << asst_tokens << " / "
                  << token_flow.size() << " tokens 作为预测目标参与 loss）\n";
    }

    // 切窗口：每个窗口 = 一个训练样本（长度 seq_len，末窗不足则 PAD）。
    // 保留全部窗口（含跨文档）；文档感知掩码在单 fold 融合 kernel 内生效
    // （fold body 逐块 select 屏蔽，不物化 (BH·seq,seq)），无需再丢弃跨文档窗口。
    std::vector<std::size_t> window_offsets;  // 每个窗口在 token_flow 中的起始偏移
    window_offsets.reserve(token_flow.size() / stride + 1);
    for (std::size_t pos = 0; pos < token_flow.size(); pos += stride)
    {
        window_offsets.push_back(pos);
    }
    std::cout << "滑动窗口: seq_len=" << cfg.seq_len << " stride=" << stride
              << " 样本数=" << window_offsets.size() << "\n" << std::endl;

    // ── 打印配置 ─────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "  GPT 文本生成训练\n";
    std::cout << "========================================\n";
    std::cout << "  词表大小: " << tokenizer->vocab_size() << "\n";
    std::cout << "  模型维度: " << cfg.d_model << "\n";
    std::cout << "  注意力头: " << cfg.num_heads << "\n";
    std::cout << "  Transformer 层数: " << cfg.num_layers << "\n";
    std::cout << "  FFN 维度: " << cfg.d_ff << "\n";
    std::cout << "  序列长度: " << cfg.seq_len << "\n";
    std::cout << "  模型架构: "
              << (cfg.model_type == "rapt" ? "RAPT (ReLU 线性注意力)"
                  : "GPT") << "\n";
    if (cfg.model_type == "rapt")
    {
        std::cout << "  位置编码: RoPE（RLA 强制，施加在 ReLU 之前）\n";
    }
    std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
    if (cfg.optimizer_name == "adam" || cfg.optimizer_name == "adamw")
    {
        // 前面的计时打印固定过 precision(1)，这里恢复足够位数（0.95 才不会印成 0.9）
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "  Adam β1/β2: " << cfg.beta1 << " / " << cfg.beta2 << "\n";
    }
    std::cout << "  loss 范围: " << loss_scope << "（生成期定好，随数据集）\n";
    if (cfg.max_steps > 0)
        std::cout << "  最大步数: " << cfg.max_steps << "\n";
    std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
    std::cout << "  GPU: " << (cfg.gpu_enabled ? "启用" : "禁用") << "\n";
    std::cout << "  梯度日志: " << (cfg.grad_log ? "启用" : "禁用") << "\n";
    std::cout << "========================================\n\n";

    // ── 创建计算引擎 ─────────────────────────────────────────
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = cfg.gpu_enabled;
    eng_cfg.gpu_device = cfg.gpu_device;
    auto engine_res = nn::cli::create_engine(eng_cfg, std::cout);
    NN_EXIT(engine_res, 1, "引擎创建失败: ");
    auto raw_engine = std::move(*engine_res);

    // ── 多精度：边界 cast 已下沉基类（NVI，原 PrecisionEngine，f16 存储）────────────────────────────────
    // f16 存储（--f16 / --precision-*）直接由引擎入口处理：有 in-kernel f16
    // 变体/f16 GEMM 的算子直读写 f16，其余算子在入口抬 f32 计算 → 按目标精度
    // 落回（expr_spec_key 本身不含精度维度，f16 变体按 (key, 精度签名) 另行注册）。
    // 全 f32 时入口为快速直通分支，与原生引擎逐字节一致。
    // 层内的 p_.param/compute/stable/optimizer 是唯一精度来源（§8.5 G4）。
    if (!nn::is_profile_f32(cfg.precision))
    {
        std::cout << "[精度] f16 存储已启用（基类边界 cast + in-kernel f16 变体 + f16 GEMM）\n"
                     "  [提示] 池底材粒度可用 NN_POOL_BLOCK_MB / NN_POOL_LADDER_MAX_MB 调参"
                     "（探针/实验用），\n"
                     "         详见 docs/development/05-mixed-precision.md §12.10。\n";
    }
    nn::ComputeEngine* engine = raw_engine.get();

    // ── 显存阶段采样（NN_MEM_STATS=1，诊断用，默认关闭）─────────────────
    // 在真实训练负载的各生命周期阶段打印池统计 + 延迟销毁字节，用于把
    // 峰值归因到具体阶段（与 src/mem_probe.cpp 的口径一致）。
    const bool mem_stats = [] {
        // MSVC CRT 弃用 getenv（-Werror）：按平台用 _dupenv_s / getenv。
#if defined(_MSC_VER)
        char* buf = nullptr;
        std::size_t len = 0;
        _dupenv_s(&buf, &len, "NN_MEM_STATS");
        const bool on = (buf != nullptr && buf[0] != '\0' && buf[0] != '0');
        std::free(buf);
        return on;
#else
        const char* v = std::getenv("NN_MEM_STATS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
    }();
    const auto mem_mark = [&](const char* tag) {
        if (!mem_stats) return;
        std::cout << "[mem] " << std::left << std::setw(16) << tag << std::right
                  << " " << engine->pool_stats() << std::endl;
    };
    mem_mark("engine-init");

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    nn::Result<nn::Model> model_build;
    if (cfg.model_type == "rapt")
    {
        // RLA 强约束：位置编码必须 RoPE（施加在 Q/K 进 ReLU 之前），强制覆盖
        cfg.pos_encoding = nn::PosEncodingType::RoPE;
        model_build = nn::build_rapt_model(*engine, nn::RAPTConfig{
            tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers,
            cfg.pos_encoding, cfg.activation, cfg.norm_type,
            /*causal=*/true, cfg.precision});
    }
    else
    {
        model_build = nn::build_gpt_model(
            *engine,
            tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers,
            cfg.pos_encoding, cfg.activation, cfg.norm_type,
            cfg.precision);
    }
    NN_EXIT(model_build, 1, "构建模型失败: ");
    auto model = std::move(*model_build);
    mem_mark("model-built");

    // ── 打印精度配置 ──
    {
        const auto& pp = cfg.precision;
        if (pp.param != nn::Precision::F32 || pp.compute != nn::Precision::F32 ||
            pp.stable != nn::Precision::F32 || pp.optimizer != nn::Precision::F32)
        {
            std::cout << "精度配置: param=" << nn::precision_name(pp.param)
                      << " compute=" << nn::precision_name(pp.compute)
                      << " stable=" << nn::precision_name(pp.stable)
                      << " optimizer=" << nn::precision_name(pp.optimizer) << "\n";
            // 消费方说明：
            //   · param/compute 由 Layer 的 p_.param/p_.compute 传给引擎原语与
            //     dsl::compute，经基类边界 cast 入口落成 f16 **存储**
            //     （引擎提供 f16 变体的算子直读写，其余走 f32 边界 cast；见
            //     compute_engine.hpp 的 NVI 边界 cast 层）；
            //   · stable 用于 softmax/LayerNorm/loss 链；optimizer 用于 Adam m/v。
            // 局限：走边界 cast 时，"被多个算子读取的大张量"（如 vocab 级 logits）
            // 会各算子各物化一份 f32 副本 → LM head 固定走 stable（见 GPTModel
            // 注释）。
        }
    }

    // ── 设置 batch 录制粒度 ──
    model.set_flush_interval(cfg.flush_interval);

    // ── 设置梯度检查点（激活重计算 L1） ──
    model.set_checkpoint_every(cfg.checkpoint_every);
    if (cfg.checkpoint_every > 0)
        std::cout << "梯度检查点已启用: 每 " << cfg.checkpoint_every << " 个 block 重算一次\n";

    // ── 设置 activation offload（L1-offload） ──
    // 与 checkpoint 可共存（混合模式）：每 checkpoint_every_ 个 block 走重算
    // （不驻留激活），其余 block 走 offload（激活搬 host-visible 不重算）。
    // 兼顾：checkpoint 块省显存但 FLOPs 2×；offload 块省显存但 PCIe 开销。
    if (cfg.activation_offload)
    {
        if (!cfg.gpu_enabled)
        {
            // CPU 引擎的 offload 原语是 no-op（restore 返回 1×1），启用只会毒化 backward
            std::cout << "[警告] --activation-offload 在 CPU 引擎上无效（no-op），已忽略。\n";
        }
        else
        {
            model.set_activation_offload(true);
            std::cout << "activation offload 已启用: 每块激活搬 host-visible，backward 拷回（不重算）\n";
            if (cfg.checkpoint_every > 0)
                std::cout << "  （混合模式）与 checkpoint 共存：checkpoint 块重算，其余块 offload\n";
            std::cout << "  offload slab 在首次 forward 时按激活形状分配（RAM ≈ 激活实际体积）\n";
        }
    }

    // ── 构建规格（用于保存） ─────────────────────────────────
    nn::ModelSpec spec;
    if (cfg.model_type == "rapt")
        spec = nn::make_rapt_spec(
            tokenizer->vocab_size(), cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers,
            cfg.pos_encoding, cfg.activation, cfg.norm_type);
    else
        spec = nn::make_gpt_spec(
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
            // RAPT 独立构建路径
            if (file_spec.is_rapt())
            {
                std::cout << "从模型文件读取 RAPT 规格\n";
                auto build_result = nn::build_rapt_model_from_spec(*engine, file_spec, cfg.precision);
                NN_EXIT(build_result, 1, "Error: ");
                model = std::move(*build_result);
                spec = file_spec;
            }
            // 统一的 GPTModel 通过 pos_encoding 区分 Learned/Sinusoidal/ALiBi，
            // GPT 与 ALiBi_GPT 走同一条构建路径。
            else if (file_spec.is_gpt() || file_spec.is_alibi_gpt())
            {
                if (file_spec.pos_encoding == nn::PosEncodingType::ALiBi)
                    std::cout << "从模型文件读取 ALiBi GPT 规格\n";
                else if (file_spec.pos_encoding == nn::PosEncodingType::RoPE)
                    std::cout << "从模型文件读取 RoPE GPT 规格\n";
                else
                    std::cout << "从模型文件读取 GPT 规格\n";
                auto build_result = nn::build_gpt_model_from_spec(*engine, file_spec, cfg.precision);
                NN_EXIT(build_result, 1, "Error: ");
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
    // 精度配置透传：状态张量（m/v/momentum）按 profile.optimizer 创建，
    // 参数更新走 in-place（存储精度不可变，§8.3）
    auto optimizer = nn::create_optimizer(
        cfg.optimizer_name, *engine,
        model.parameters(), model.param_gradients(), cfg.lr,
        cfg.weight_decay, cfg.precision, cfg.beta1, cfg.beta2);
    NN_EXIT(optimizer, 1, "错误：未知优化器名称: ");

    Scalar optimizer_current_lr = cfg.lr;
    mem_mark("optimizer-created");

    // ── 学习率调度配置（委托给 nn::cli::compute_epoch_lr） ──
    nn::cli::LrScheduleConfig lr_sched_cfg;
    lr_sched_cfg.base_lr = cfg.lr;
    lr_sched_cfg.min_lr = cfg.min_lr;
    lr_sched_cfg.warmup_epochs = cfg.warmup_epochs;
    lr_sched_cfg.total_epochs = cfg.epochs;
    lr_sched_cfg.schedule = cfg.lr_schedule;
    lr_sched_cfg.lr_per_epoch = cfg.lr_per_epoch;

    nn::CrossEntropyLoss ce_loss;
    // loss 链精度 = profile.stable（§9.1 / D9：softmax/log/大词表归约的溢出防线；
    // 默认 F32 = loss 链全程 f32）
    ce_loss.set_precision_profile(cfg.precision);

    // ── 训练循环 ─────────────────────────────────────────────
    // 每样本 = 一个 seq_len 滑动窗口（可能跨行），目标 = 输入左移一位。
    if (window_offsets.empty())
    {
        std::cerr << "无有效训练样本（文本 token 流为空）\n";
        return 1;
    }
    std::cout << "训练窗口样本数: " << window_offsets.size() << "\n" << std::endl;

    // ── 测试子集（可选，数据集配套里带）────────────────────────
    std::vector<std::size_t> test_window_offsets;
    std::vector<std::size_t> test_flow;
    std::vector<std::size_t> test_flow_doc_ids;
    std::vector<unsigned char> test_flow_assistant;  // 与 test_flow 等长（all 模式全 1）
    if (dataset.info().test_docs > 0)
    {
        auto test_corpus_r = dataset.load_text(true);
        NN_EXIT(test_corpus_r, 1, "读取测试子集失败: ");
        test_flow = std::move(test_corpus_r->token_flow);
        test_flow_doc_ids = std::move(test_corpus_r->doc_ids);
        test_flow_assistant = std::move(test_corpus_r->loss_mask);

        if (loss_scope == "assistant")
        {
            std::size_t t_asst = 0;
            for (unsigned char m : test_flow_assistant) t_asst += m;
            std::cout << "测试集 assistant 掩码: " << t_asst << " / "
                      << test_flow.size() << " tokens 参与评估\n";
        }

        // 测试窗口：保留全部窗口（与训练一致，含跨文档；文档感知掩码 kernel 内生效）
        for (std::size_t pos = 0; pos < test_flow.size(); pos += stride)
        {
            test_window_offsets.push_back(pos);
        }
        std::cout << "测试集: " << dataset.info().test_docs << " docs  样本数: "
                  << test_window_offsets.size()
                  << "  tokens: " << test_flow.size() << std::endl;
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

    // 随机种子：每次训练/每次 --resume 续训后的样本顺序都不同，避免跨进程
    // 数据顺序完全一致导致 GUI 图表上出现"loss 片段重复"（与 mnist_train 一致）。
    std::mt19937_64 rng{std::random_device{}()};
    std::vector<std::size_t> sample_indices(window_offsets.size());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;

    // ── 损失覆盖范围 ─────────────────────────────────────────────
    // 全部 seq_len 位置参与 loss。（原 ZiPT「压缩模式仅窗口位置参与 loss」
    // 分支已随 ZiPT 于 2026-10-01 移除，见 docs/history.md。）
    const std::size_t eff_seq = cfg.seq_len;

    // ── 预分配 batch 缓冲区（末批不满时 resize 到实际 this_bs） ──
    // x_tokens: (seq_len, batch) 输入 token IDs（每窗口一列）
    // y_tokens: (seq_len, batch) 目标 token IDs（x 左移一位）
    // 最后一个窗口不足 seq_len 时用 pad_id 填充，目标对应位置也用 pad_id
    nn::Matrix x_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix y_tokens(cfg.seq_len, cfg.batch_size);
    nn::Matrix loss_mask(cfg.seq_len, cfg.batch_size);
    std::vector<std::size_t> flat_targets(eff_seq * cfg.batch_size);

    auto t_start = std::chrono::steady_clock::now();

    // --max-steps：按本进程执行的步数计数（与 --resume 组合可分段跑）
    std::size_t steps_executed = 0;
    bool stop_training = false;

    for (int epoch = cfg.start_epoch; epoch < cfg.epochs; ++epoch)
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

        // TDR 重启续训：resume 时本 epoch 内跳过已完成的前 start_step 步，
        // 避免重启后从 epoch 开头重跑、GUI 图表出现重复 loss 片段。
        const std::size_t epoch_first_step =
            (epoch == cfg.start_epoch) ? cfg.start_step : 0;

        // 梯度累积：距上次参数更新的步数（每 accum_steps 步更新一次）
        std::size_t steps_since_update = 0;

        // ── 非阻塞 loss 回读 ──────────────────────────────────────
        // 每步把设备端 loss_sum 排入一个异步回读槽位（**不等待**），稍后就绪
        // 即取。取 loss 不做 end_batch + wait_in_flight：loss 走 device 侧异步
        // 回读，host 录制的帧不被打断——host 一旦在录制中途等待，就会 drain
        // 整条流水线，GPU 在 host 录制期间空转 → 占用率锯齿。
        // 稳定态下第 N 步即可取到第 N-1 步的 loss，打印仍是每步一条。
        struct PendingLoss
        {
            std::size_t slot = 0;
            std::size_t step = 0;
            std::size_t valid = 0;
            Scalar inv_num_valid = Scalar{0};
            nn::Tensor keepalive;   // 回读命令引用的张量须活到就绪
        };
        std::deque<PendingLoss> pending_loss;
        const std::size_t loss_slots =
            std::max<std::size_t>(engine->scalar_readback_slots(), 1);
        std::size_t loss_slot_next = 0;

        // 非阻塞收割：消费所有已就绪的回读（累计统计 + 按 log_interval 打印）
        auto harvest_loss = [&]() -> nn::Result<void>
        {
            while (!pending_loss.empty())
            {
                Scalar sum = Scalar{0};
                auto pr = engine->poll_scalar_readback(pending_loss.front().slot, sum);
                NN_TRY_CHECK(pr);
                if (!*pr) break;   // 未就绪：留待下次（不阻塞）
                PendingLoss pl = std::move(pending_loss.front());
                pending_loss.pop_front();
                const Scalar loss = -sum * pl.inv_num_valid;
                total_weighted += loss * static_cast<Scalar>(pl.valid);
                total_valid += pl.valid;
                if ((pl.step + 1) % cfg.log_interval == 0 || pl.step + 1 == steps_per_epoch)
                {
                    std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                              << "  step " << pl.step + 1 << "/" << steps_per_epoch
                              << "  loss: " << std::fixed << std::setprecision(4) << loss
                              << "   " << std::flush;
                }
            }
            return {};
        };
        // 阻塞兜底收割（槽位将满 / epoch 收尾）：等的是"已在 GPU 上跑的旧帧"，
        // GPU 不会因此空转；正常路径不触发。
        auto drain_loss = [&]() -> nn::Result<void>
        {
            while (!pending_loss.empty())
            {
                auto r0 = harvest_loss();
                if (!r0) return r0;
                if (pending_loss.empty()) break;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            return {};
        };

        for (std::size_t step = epoch_first_step; step < steps_per_epoch; ++step)
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

            // ── --max-steps：达到上限的那一步为最后一步，同时强制完成一次
            // 参数更新（把梯度累积中未满的累积刷进参数，不丢半截累积）──
            const bool is_final_step =
                (cfg.max_steps > 0) && (steps_executed + 1 >= cfg.max_steps);
            ++steps_executed;

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
                flat_targets.resize(eff_seq * this_bs);
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
                    // 仅真实位置（非 padding）参与 loss；
                    // 且 loss_mask=1（生成期定好；assistant 段外为 0）
                    const bool participate = (t + 1 < win_len) &&
                        (flow_assistant.empty() || flow_assistant[win_pos + t + 1]);
                    loss_mask.set_value_unchecked(t, b, participate ? 1.0f : 0.0f);
                    if (participate) ++step_valid;
                }
            }

            // ── 文档感知 ────────────────────────────────────────
            // 行边界即文档边界（每行 = 一篇文档，见 build_flow_doc_ids）。
            // 为每窗口设 doc_ids 施加块对角掩码；fold 掩码变体（Doc/AlibiDoc）
            // 使该掩码在 kernel 内生效（不物化 (BH·seq,seq)），无需再丢弃跨文档窗口。
            std::vector<std::size_t> doc_ids;
            if (!flow_doc_ids.empty())
            {
                doc_ids.assign(cfg.seq_len * this_bs, 0);
                for (std::size_t b = 0; b < this_bs; ++b)
                {
                    const std::size_t win_pos = window_offsets[sample_indices[offset + b]];
                    const std::size_t win_len =
                        std::min(cfg.seq_len, token_flow.size() - win_pos);
                    for (std::size_t t = 0; t < cfg.seq_len; ++t)
                        if (t < win_len)
                            doc_ids[b * cfg.seq_len + t] = flow_doc_ids[win_pos + t];
                }
            }
            model.set_doc_ids(doc_ids);

            // ── Matrix → Tensor（上传到引擎设备） ──────────────
            auto x_tensor_r = engine->from_matrix(x_tokens);
            NN_EXIT(x_tensor_r, 1, "\nfrom_matrix(x_tokens) failed: ");

            // ── 构造平坦标签（与 logits 列序一致：batch-major，i = b*eff_seq + t） ──
            // 模型 forward 对输入 transpose 后按 batch-major 列序输出 logits，
            // 因此 flat_targets / flat_mask 必须与之一一对应（b 外层、t 内层）。
            auto y_span = y_tokens.span();
            auto m_span = loss_mask.span();
            std::vector<Scalar> flat_mask(eff_seq * this_bs);
            for (std::size_t b = 0; b < this_bs; ++b)
                for (std::size_t t = 0; t < eff_seq; ++t)
                {
                    const std::size_t src_t = t;                 // 窗口内真实位置
                    const std::size_t pm = src_t * this_bs + b;  // 源的 position-major 索引
                    const std::size_t bm = b * eff_seq + t;     // 目标的 batch-major 索引
                    flat_targets[bm] = static_cast<std::size_t>(y_span[pm]);
                    flat_mask[bm] = m_span[pm];
                }

            // ── 启用 GPU batch 录制 ─────────────────────────
            // 整个 forward + backward + optimizer step 录制到一个 command buffer，
            // end_batch 时一次 vkQueueSubmit + vkWaitForFences，消除 per-primitive 同步开销。
            // CPU 引擎 begin_batch/end_batch 为 no-op，所以两套引擎都安全。
            auto begin_r = engine->begin_batch();
            NN_EXIT(begin_r, 1, "begin_batch failed: ");

            // ── 前向传播 ─────────────────────────────────────
            auto fwd_result = model.forward(*x_tensor_r);
            NN_EXIT(fwd_result, 1, "Error: ");
            auto logits = std::move(*fwd_result);
            mem_mark("step/forward");
            // logits: (vocab_size, seq_len × batch_size)

            // ── 损失（稀疏标签，避免 one-hot 爆显存）────────
            // 此处不做同步读回 loss：forward_sparse_sum 只算出设备端
            // (1,1) loss 和（Σ loss_vec，未归一化）；flush 提交 forward 帧后
            // 把它排入异步回读槽位，host 立刻继续录制 backward——全程不 drain。
            // loss 只在 device 侧回读、不在 host 侧判定；数值稳定性由
            // --max-norm 梯度裁剪 + 观察 loss 曲线负责。
            auto mask_span = std::span<const Scalar>(flat_mask);
            std::size_t loss_num_valid = 0;
            auto loss_sum_t = ce_loss.forward_sparse_sum(
                *engine, logits, flat_targets, mask_span,
                tokenizer->vocab_size(), loss_num_valid, /*grad_reuse=*/&logits);
            NN_EXIT(loss_sum_t, 1, "Error: ");
            // 回读固定按 F32 取 4 字节：非 F32 时在 batch 内先 cast
            //（录制态，不提交、不 drain）
            auto loss_sum_f32 = engine->cast(*loss_sum_t, nn::Precision::F32);
            NN_EXIT(loss_sum_f32, 1, "Error: ");

            // ── 显存优化：logits 已消费完毕，立即释放（必须在 flush 之前）──
            //   延迟销毁按"当前录制帧"打标签：flush 前释放 → 标签为 forward
            //   帧，其 fence 在 backward 录制期间即完成，可被中途 reap 回收；
            //   若放在 flush 之后释放，标签落到 backward 帧，513MB
            //   （vocab×seq×batch）要等整个 backward 提交完成才还池（探针实测
            //   pending 主项）。此时 loss 已算出，LM Head 的 backward 只需
            //   input + grad_output，且 forward 帧已录完对 logits 的引用。
            logits = {};
            mem_mark("step/logits-free");

            // ── 中点刷新：提交 forward+loss，拆分为两次 GPU 提交 ──
            // 大词表 + 长序列时 forward+backward 单次提交可能触发 TDR 超时。
            // 在 forward 与 backward 之间 flush，将一次大提交拆为两次小提交。
            auto flush_r = engine->flush_batch();
            mem_mark("step/loss-fwd");
            NN_EXIT(flush_r, 1, "\nflush_batch (forward) failed: ");

            // forward 帧已提交 → 排队异步回读（同队列 FIFO：拷贝在生产者之后）
            {
                // 槽位将满：先非阻塞收割，必要时阻塞兜底（保证不覆盖未取走的值）
                auto hv0 = harvest_loss();
                NN_EXIT(hv0, 1, "harvest_loss failed: ");
                if (pending_loss.size() >= loss_slots)
                {
                    auto hv1 = drain_loss();
                    NN_EXIT(hv1, 1, "drain_loss failed: ");
                }
                PendingLoss pl;
                pl.slot = loss_slot_next;
                pl.step = step;
                pl.valid = step_valid;
                pl.inv_num_valid = (loss_num_valid > 0)
                    ? Scalar{1} / static_cast<Scalar>(loss_num_valid) : Scalar{0};
                pl.keepalive = std::move(*loss_sum_f32);
                auto sr = engine->submit_scalar_readback(pl.slot, pl.keepalive);
                if (!sr) {
                    std::cerr << "submit_scalar_readback failed: "
                              << sr.error().message << '\n';
                    return 1;
                }
                loss_slot_next = (loss_slot_next + 1) % loss_slots;
                pending_loss.push_back(std::move(pl));
            }

            // ── 反向传播（梯度已含 mask，无需额外处理） ────────
            auto grad_result = ce_loss.backward();
            NN_EXIT(grad_result, 1, "\nLoss backward failed: ");

            // 梯度积累缩放：每步梯度除以 accum_steps
            //   forward_sparse 的梯度为 (softmax-one_hot)/num_valid（单步平均），
            //   积累后需除以 accum_steps 才能等价于有效 batch 的平均梯度。
            //   否则梯度是正确值的 accum_steps 倍，Adam 虽近似抵消但非精确。
            if (cfg.accum_steps > 1)
            {
                auto scale_r = engine->scale_inplace(*grad_result,
                    Scalar{1} / static_cast<Scalar>(cfg.accum_steps));
                NN_EXIT(scale_r, 1, "\n梯度缩放失败: ");
            }

            auto bwd_result = model.backward(*grad_result);
            mem_mark("step/backward");
            NN_EXIT(bwd_result, 1, "Error: ");

            // ── 提交 backward batch（单独一次提交，已与 forward 拆分） ──
            auto bwd_end = engine->end_batch();
            mem_mark("step/end-batch");
            NN_EXIT(bwd_end, 1, "\nend_batch (backward) failed: ");

            // ── 收割已就绪的 loss 回读（非阻塞；打印/统计在此推进）─────────
            // 稳定态：此刻第 N-1 步的 loss 早已写回（GPU 一直在跑），
            // poll 立即命中 → 每步一条打印，且 host 一秒都不等 GPU。
            {
                auto hv = harvest_loss();
                NN_EXIT(hv, 1, "harvest_loss failed: ");
            }

            // ── 显存回收（L2）：end_batch 提交完成、延迟销毁已 flush，
            //    归还完全空闲的内存池底材（GPU 引擎有效，CPU 引擎 no-op） ──
            auto rel_r = engine->release_idle_pool_blocks();
            mem_mark("step/released");
            NN_EXIT(rel_r, 1, "\n显存回收失败: ");

            // ── 显存优化：logits 梯度已消费完毕，立即释放 ──
            //   model.backward 已把 grad 传播到各参数梯度，grad_result（1.6GB）
            //   已消费完毕。end_batch 之后 backward 已提交执行完，释放安全。
            grad_result = {};

            // ── 梯度累积：每 accum_steps 步才更新一次参数 ──
            // forward+backward 每步都执行（梯度累加到参数梯度），
            // 梯度裁剪/step/zero_grad 仅在累积到 accum_steps 或 epoch 末尾执行。
            ++steps_since_update;
            const bool do_update =
                (steps_since_update >= cfg.accum_steps) ||
                (step + 1 == steps_per_epoch) || is_final_step;

            if (do_update)
            {
                // ── 梯度裁剪（在 step() 之前，backward() 之后） ──
                // clip_grad_norm 逐原语在 batch **外**会各自
                // submit_and_wait（约 200 次 host↔GPU 往返/次裁剪）；包进
                // 一个 batch 后只剩一次范数下载同步（每 accum_steps 步一次，
                // 而非每步）。这是范数的数据依赖，无法异步化。
                if (cfg.max_norm > 0)
                {
                    auto clip_begin = engine->begin_batch();
                    NN_EXIT(clip_begin, 1, "begin_batch (clip) failed: ");
                    auto clip_r = optimizer->clip_grad_norm(cfg.max_norm);
                    NN_EXIT(clip_r, 1, "\n梯度裁剪失败: ");
                    auto clip_end = engine->end_batch();
                    NN_EXIT(clip_end, 1, "end_batch (clip) failed: ");
                }

                auto opt_begin = engine->begin_batch();
                NN_EXIT(opt_begin, 1, "begin_batch (optimizer) failed: ");

                // ── 优化器 step + 梯度清零 ──
                auto step_result = optimizer->step();
                NN_EXIT(step_result, 1, "Error: ");

                // ── 梯度统计（step 后、zero_grad 前） ──
                if (cfg.grad_log && ((step + 1) % cfg.log_interval == 0 || step + 1 == steps_per_epoch))
                {
                    std::cout << "\n  [grad] step " << step + 1 << ":" << std::endl;
                    log_gradient_stats(*engine, model.param_gradients());
                }

                auto zero_result = optimizer->zero_grad();
                NN_EXIT(zero_result, 1, "\n优化器 zero_grad 失败: ");

                // ── 提交 batch：一次 vkQueueSubmit + vkWaitForFences ──
                auto end_r = engine->end_batch();
                NN_EXIT(end_r, 1, "end_batch (optimizer) failed: ");

                steps_since_update = 0;
            }

            // ── 定期保存 checkpoint（独立于 log_interval；save_interval=0 禁用） ──
            if (cfg.save_interval > 0 &&
                ((step + 1) % cfg.save_interval == 0 || step + 1 == steps_per_epoch))
            {
                auto save_r = nn::save_model(cfg.save_path, model, spec, tokenizer_bytes);
                if (!save_r)
                    std::cerr << "\n  [ckpt] 保存失败: " << save_r.error().message << "\n";
            }

            // ── --max-steps 到达：本步已完成更新与保存，退出步循环 ──
            if (is_final_step)
            {
                stop_training = true;
                break;
            }

            // 进度显示在 harvest_loss 内完成（loss 走异步回读）：按"值就绪即打印"，
            // 稳定态仍是每步一条，且 host 全程不等 GPU。
        }

        // ── epoch 收尾：读完最后几步尚未取回的 loss（各 epoch 一次）──────
        // 等的是已经在 GPU 上执行的旧帧，不影响 GPU 占空比。
        {
            auto ep_drain = drain_loss();
            NN_EXIT(ep_drain, 1, "loss drain failed: ");
        }

        // ── --max-steps 到达：跳过本 epoch 的统计与测试评估，直接收尾 ──
        if (stop_training)
        {
            std::cout << "\n已达 --max-steps=" << cfg.max_steps << "（本进程执行 "
                      << steps_executed << " 步），提前结束训练\n";
            break;
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
                    flat_targets.resize(eff_seq * this_bs);
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
                        // 仅真实位置（非 padding）参与评估；
                        // 且 loss_mask=1（生成期定好；assistant 段外为 0）
                        const bool participate = (t + 1 < win_len) &&
                            (test_flow_assistant.empty() ||
                             test_flow_assistant[win_pos + t + 1]);
                        loss_mask.set_value(t, b, participate ? 1.0f : 0.0f);
                        if (participate) ++test_valid;
                    }
                }

                // Forward pass only（不 backward）
                auto x_tensor_r = engine->from_matrix(x_tokens);
                if (!x_tensor_r) { std::cerr << "  测试 from_matrix 失败: " << x_tensor_r.error().message << '\n'; break; }

                // 文档感知：与训练一致；为每窗口设 doc_ids（fold 块对角掩码）
                {
                    std::vector<std::size_t> test_doc_ids;
                    if (!test_flow_doc_ids.empty())
                    {
                        test_doc_ids.assign(cfg.seq_len * this_bs, 0);
                        for (std::size_t b = 0; b < this_bs; ++b)
                        {
                            const std::size_t win_pos = test_window_offsets[toffset + b];
                            const std::size_t win_len =
                                std::min(cfg.seq_len, test_flow.size() - win_pos);
                            for (std::size_t t = 0; t < cfg.seq_len; ++t)
                                if (t < win_len)
                                    test_doc_ids[b * cfg.seq_len + t] =
                                        test_flow_doc_ids[win_pos + t];
                        }
                    }
                    model.set_doc_ids(test_doc_ids);
                }

                // 构建 flat_targets / flat_mask（与 logits 列序一致：batch-major，i = b*eff_seq + t）
                auto y_span = y_tokens.span();
                auto mm_span = loss_mask.span();
                std::vector<Scalar> tmask(eff_seq * this_bs);
                for (std::size_t b = 0; b < this_bs; ++b)
                    for (std::size_t t = 0; t < eff_seq; ++t)
                    {
                        const std::size_t src_t = t;                  // 窗口内真实位置
                        const std::size_t pm = src_t * this_bs + b;   // 源 position-major 索引
                        const std::size_t bm = b * eff_seq + t;  // 目标 batch-major 索引
                        flat_targets[bm] = static_cast<std::size_t>(y_span[pm]);
                        tmask[bm] = mm_span[pm];
                    }

                auto begin_r = engine->begin_batch();
                if (!begin_r) { std::cerr << "  测试 begin_batch 失败: " << begin_r.error().message << '\n'; break; }

                auto fwd_result = model.forward(*x_tensor_r);
                if (!fwd_result) { std::cerr << "  测试前向传播出错: " << fwd_result.error().message << '\n'; break; }
                auto test_mask_span = std::span<const Scalar>(tmask);
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
        auto save_result = nn::save_model(cfg.save_path, model, spec, tokenizer_bytes);
        NN_EXIT(save_result, 1, "Error: ");
    }
    std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1)
              << total_sec << "s"
              << "  词表已嵌入模型文件" << std::endl;

    return 0;
}
