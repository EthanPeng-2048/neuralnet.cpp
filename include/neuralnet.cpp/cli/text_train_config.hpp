// ── text_train_config.hpp — text_train 的配置与 CLI 参数解析层 ───────────────
//
// 本头是 text_train 唯一允许**直接终止进程**的地方（`std::exit`）：参数错误属于
// 用户输入错误，用退出码表达最直接（与 mnist_train 的 parse_args 同口径）。
// 训练循环 / 恢复 / 断点保存三个模块一律返回 `Result`，不终止进程——见
// docs/todo/18-roadmap.md §4.2 P1-1 的"std::exit 只在 CLI 参数解析层出现"。
//
// 内容：
//   * `TrainConfig`（train_loop / recovery / checkpoint 三模块的输入）
//   * `print_text_train_usage`（--help 排版，走 nn::cli::Help）
//   * `parse_text_train_args`（argv → TrainConfig，错误即 exit(1)/exit(0)）
//
// 注：`--help` 的**文案与排版**属于对外契约（gui_cli_audit / 文档审计会比对选项
// 名单），改动前先跑 `python bench/gui_cli_audit.py`。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "neuralnet.cpp/core_config.hpp"        // nn::Scalar
#include "neuralnet.cpp/core_errors.hpp"        // nn::parse_number
#include "neuralnet.cpp/compute_optimizer.hpp"  // nn::find_optimizer / optimizer_names
#include "neuralnet.cpp/domain_gpt.hpp"         // GPT_* 默认形状
#include "neuralnet.cpp/model_spec.hpp"         // PosEncodingType / ActivationType / NormType
#include "neuralnet.cpp/precision.hpp"          // nn::Precision / PrecisionProfile
#include "neuralnet.cpp/cli/cli_gpu_option.hpp"    // nn::cli::parse_gpu_option
#include "neuralnet.cpp/cli/cli_help.hpp"          // nn::cli::Help

namespace nn::cli
{
using nn::Scalar;

// ── 精度解析辅助 ─────────────────────────────────────────────────────────
// allow_t1_58：三值（T1_58）只适用于 param 槽（"该层权重是三值"）；本入口的
// GPT 架构在 P1 **还没有**三值线性层（docs 21 §4.8 P1.5），但词法上照样接受：
// 不兼容组合由 `Layer::init` 的能力校验给出"层 X 不支持 param=t1_58"的报错，
// 这正是设计意图（§4.1）—— CLI 不加模型白名单。
nn::Precision parse_precision_arg(const std::string& name, const char* flag,
                              bool allow_t1_58 = false)
{
    if (name == "f16" || name == "half") return nn::Precision::F16;
    if (name == "f32" || name == "float") return nn::Precision::F32;
    if (name == "t1_58")
    {
        if (allow_t1_58) return nn::Precision::T1_58;
        std::cerr << "无效 --" << flag << ": t1_58（三值权重只适用于 --precision-param）"
                  << "，可选: f16, f32\n";
        std::exit(1);
    }
    std::cerr << "无效 --" << flag << ": " << name << "，可选: "
              << (allow_t1_58 ? "f16, f32, t1_58" : "f16, f32") << "\n";
    std::exit(1);
}

// ==================== 帮助信息 ====================
void print_text_train_usage(const char *prog)
{
    nn::cli::Help help(std::cout, prog, "GPT 文本生成训练程序");

    help.usage("<dataset.nndataset> [选项]");

    help.section("参数");
    help.item("<dataset.nndataset>", "数据集文件路径 (.nndataset，由 dataset_gen 生成；\n词表/loss 掩码/test 子集都在生成期定好)");

    help.section("选项");
    help.opt("--save <path>", "推理模型保存路径 (默认: gpt_model.nnmodel；\n只含推理所需参数——三值层写打包形态，不含训练参数)");
    help.opt("--resume <path>", "从训练断点 (.nnckpt) 恢复训练；\n推理模型文件 (.nnmodel) 没有训练参数，会被明确拒绝");
    help.opt("--resume-epoch <n>", "从第 n 个 epoch 继续 (0-based，需配合 --resume；默认 0)");
    help.opt("--resume-step <n>", "从本 epoch 内第 n 步继续 (0-based，需配合 --resume；默认 0)");
    help.opt("--epochs <n>", "训练轮数 (默认: 10)");
    help.opt("--lr <lr>", "学习率 (默认: 0.001)");
    help.opt("--batch-size <n>", "批大小 (默认: 32)");
    help.opt("--accum-steps <n>", "梯度累积步数 (默认: 1)\n每 n 步 forward/backward 累加梯度后再更新参数，等效放大 batch_size×n");
    help.opt("--seq-len <n>", "序列长度 (默认: 256)");
    help.opt("--stride <n>", "滑动窗口步长 (默认: 等于 --seq-len，即不重叠)\n设小可产生重叠窗口，增加训练样本数");
    // RV-04：名单文本由 nn::optimizer_table()（唯一真源）生成，不再手抄
    help.opt("--optimizer <name>",
             "优化器: " + nn::optimizer_names("/") +
                 " (默认: adam)\nadamwquant: 量化状态的 AdamW（状态 4 B/权重 vs 8 B；见 docs/todo/21 §4.13）");
    help.opt("--weight-decay <w>", "AdamW 权重衰减系数 (默认: 0.01)");
    help.opt("--beta1 <b>", "Adam/AdamW 一阶动量衰减 β1 (默认: 0.9，sgd/muon 忽略)");
    help.opt("--beta2 <b>", "Adam/AdamW 二阶动量衰减 β2 (默认: 0.999，sgd/muon 忽略)\nLLM 预训练惯例常取 0.95 (GPT-3/LLaMA 配方)");
    help.opt("--max-norm <f>", "梯度裁剪最大全局 L2 范数 (默认: 0=不裁剪)");
    help.opt("--log-interval <n>", "每隔多少 step 显示进度 (默认: 50)");
    help.opt("--save-interval <n>", "每隔多少 step 保存训练断点 .nnckpt (默认: 100；\n文件名 = --save 的路径换 .nnckpt 扩展名)");
    help.opt("--max-steps <n>", "本次运行最多训练多少 step 后停止并保存 (默认: 0=不限)\n按本进程执行的步数计，可与 --resume 组合分段跑；用于吞吐实测/分段训练");
    help.opt("--grad-log", "显示梯度统计 (范数/最大值/均值)");
    help.opt("--gpu [索引|名称]", "启用 GPU 加速 (需要 Vulkan SDK)\n空格形式只收枚举索引；名称子串用 --gpu=<名称>，如 --gpu=NVIDIA / --gpu=40HX");
    help.opt("--help, -h", "显示此帮助信息");

    help.section("模型结构");
    help.opt("--model <type>", "模型架构: gpt/rapt (默认: gpt)\ngpt: 标准 Transformer 语言模型\nrapt: ReLU 线性注意力 (RLA) 语言模型，动态稀疏检索，\nO(L·d²) 复杂度；强制 RoPE (ReLU 前施加)");
    help.opt("--positional-encoding <type>", "位置编码类型: learned/sinusoidal/alibi/rope (默认: learned)\nlearned: 可学习位置嵌入 (GPT 原版)\nsinusoidal: 正弦波固定位置编码 (不参与训练)\nalibi: 线性偏置注意力 (无位置嵌入，支持长度外推)\nrope: 旋转位置编码 (现代方案，在注意力 Q/K 上施加)");
    help.opt("--activation <type>", "FFN 激活: gelu/swiglu/relu2 (默认: gelu)\ngelu: QuickGeLU (GPT-2 风格)\nswiglu: SwiGLU (LLaMA/Mistral 风格，每参数效率更高)\nrelu2: 门控平方 ReLU (BitNet b1.58 2B4T 的 hidden_act)");
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
    help.opt("--t1_58", "快捷方式: 精度预设 t1_58 = f16 存储 + 三值权重\n(param=T1_58，compute=F16，stable/optimizer=F32；等价于 --f16 再把 param 换成 t1_58)。\n范围与前置拒绝同 --precision-param t1_58（docs/todo/21-quantized-weights.md §4.10）");
    help.opt("--precision-param <f16|f32|t1_58>",
             "权重/参数精度 (默认: f32)；t1_58 = 三值权重（1.58-bit）——\n"
             "GPT 已支持（注意力 4 投影 + fc1/fc2 + 未 tie 的 LM head 全部换成\n"
             "BitLinear；P1.5 落地）；--model rapt 选它会被前置拒绝\n"
             "(docs/todo/21-quantized-weights.md §4.8.3)");
    help.opt("--precision-compute <f16|f32>", "常规算子计算精度 (matmul/逐元素/gather，默认: f32)");
    help.opt("--precision-stable <f16|f32>", "数值敏感算子精度 (softmax/LayerNorm/loss，默认: f32)");
    help.opt("--precision-optimizer <f16|f32>", "优化器状态精度 (Adam m/v，默认: f32)");
}
struct TrainConfig
{
    std::string text_path;
    // 推理模型（.nnmodel，Model 角色）：只含推理所需参数（三值层写打包形态）。
    std::string save_path = "gpt_model.nnmodel";
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
    std::string model_type = "gpt";   // gpt / rapt（zipt 已移除）
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

    // ── 本进程**显式给出**的 A/B 类旗标（P1-2；按旗标名登记，不含值）──────
    // 断点续训（role=Checkpoint 且带训练状态段）时，这些旗标一律禁止输入
    // （docs/todo/23 §2.1.1），续训路径据此报错；`role=Model`（SFT warm start）
    // 时它们才是合法的配方来源。
    std::vector<std::string> explicit_train_flags;
};

// ── 断点续训禁列（P1-2，docs/todo/18 §4.2 / docs/todo/23 §2.1.1）──────────
// A 类（计数器/状态）+ B 类（形状/配方）：断点是**训练的可续点**，这些值由
// 断点自带的状态段提供，续训时再显式给 = 把轨迹掰弯（lr 曲线形状、step 网格、
// 精度路径都会跟着变），故一律报错。
// C 类（允许，见下）只留"完全不改变训练"的参数：`--save` / `--save-interval` /
// `--log-interval` / `--max-steps` / `--gpu`（+ 设备）/ `--grad-log` /
// `--checkpoint-every` / `--activation-offload` / `--flush-interval` / `--resume`
// 本身（含数据集路径 = 位置）、以及探针环境变量。
inline constexpr std::string_view kForbiddenResumeFlags[] = {
    "--resume-epoch", "--resume-step",
    "--epochs", "--lr", "--batch-size", "--accum-steps", "--seq-len", "--stride",
    "--optimizer", "--weight-decay", "--beta1", "--beta2",
    "--d-model", "--num-heads", "--num-layers", "--d-ff", "--model",
    "--f16", "--t1_58",
    "--precision-param", "--precision-compute", "--precision-stable", "--precision-optimizer",
    "--lr-schedule", "--warmup-epochs", "--warmup-steps", "--min-lr", "--lr-per-epoch",
    "--max-norm", "--positional-encoding", "--activation", "--norm",
};

[[nodiscard]] constexpr std::span<const std::string_view> forbidden_resume_flags() noexcept
{
    return kForbiddenResumeFlags;
}

[[nodiscard]] inline bool is_forbidden_resume_flag(std::string_view arg) noexcept
{
    for (std::string_view f : forbidden_resume_flags())
        if (arg == f) return true;
    return false;
}

// 禁列清单（一行式，报错文案/帮助/审计脚本共用同一口径）
[[nodiscard]] inline std::string forbidden_resume_flags_text()
{
    std::string s;
    for (std::string_view f : forbidden_resume_flags())
    {
        if (!s.empty()) s += " ";
        s += f;
    }
    return s;
}

TrainConfig parse_text_train_args(int argc, char *argv[])
{
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        // ── A/B 类旗标登记（P1-2 断点续训禁列；按**旗标名**判定、不看值）──
        // 解析层的默认值与用户输入无法区分，故必须在解析期记录"本进程真出现过"
        // 的旗标；续训路径在断点带训练状态段时发现列表非空即报错。
        if (nn::cli::is_forbidden_resume_flag(arg))
            cfg.explicit_train_flags.push_back(arg);
        if (arg == "--help" || arg == "-h")
        {
            print_text_train_usage(argv[0]);
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
            // RV-04：校验查表（唯一真源）——不再维护第二份硬编码白名单
            if (nn::find_optimizer(cfg.optimizer_name) == nullptr)
            {
                std::cerr << "未知优化器: " << cfg.optimizer_name
                          << "，可选: " << nn::optimizer_names(", ") << "\n";
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
                    std::cerr << "（ZiPT 已移除，代码保留在 legacy/zipt 分支；"
                                 "恢复前提见 docs/history/history.md）\n";
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
        else if (arg == "--t1_58")
        {
            // 精度预设 t1_58 = **三值权重 + f16 存储**（profile_t1_58()：
            // param=T1_58, compute=F16, stable/optimizer=F32）——就是 --f16 再
            // 把 param 换成 t1_58，故三条命令等价：
            //   --t1_58 / --f16 --precision-param t1_58 / --precision-param t1_58
            //   --precision-compute f16
            // 与 --f16 同理，想要字面四槽全 f16 再追加 --precision-stable f16
            // --precision-optimizer f16（数值上不可用于训练，见 precision.hpp）。
            cfg.precision = nn::profile_t1_58();
        }
        else if (arg == "--precision-param" && i + 1 < argc)
        {
            cfg.precision.param = parse_precision_arg(argv[++i], "precision-param",
                                                  /*allow_t1_58=*/true);
        }
        else if (arg == "--precision-compute" && i + 1 < argc)
        {
            cfg.precision.compute = parse_precision_arg(argv[++i], "precision-compute");
        }
        else if (arg == "--precision-stable" && i + 1 < argc)
        {
            cfg.precision.stable = parse_precision_arg(argv[++i], "precision-stable");
        }
        else if (arg == "--precision-optimizer" && i + 1 < argc)
        {
            cfg.precision.optimizer = parse_precision_arg(argv[++i], "precision-optimizer");
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
            else if (v == "relu2")
                cfg.activation = nn::ActivationType::ReLU2;
            else
            {
                std::cerr << "未知激活类型: " << v
                          << "，可选: gelu, swiglu, relu2\n";
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
} // namespace nn::cli
