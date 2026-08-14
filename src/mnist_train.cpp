// ── MNIST 手写数字训练程序（引擎化架构，支持 MLP 与 Transformer） ────────
//
// 数据流：
//   CSV → Matrix(feat_dim, N) + Matrix(10, N)
//   每 batch：Matrix → engine.from_matrix → Tensor[device]
//     → model.forward(Tensor) → Tensor
//     → ce_loss.forward(engine, out, target) → Scalar
//     → ce_loss.backward() → Tensor
//     → model.backward(Tensor) → (丢弃)
//     → optimizer.step() / model.zero_grad()
//   evaluate：forward → engine.to_matrix → CPU argmax
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
// 架构选择：--arch mlp|transformer（默认 mlp，从已保存模型 resume 时自动识别）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>
#include <neuralnet.cpp/cli/lr_scheduler.hpp>
#include <neuralnet.cpp/cli/mnist_io.hpp>
#include <neuralnet.cpp/cli/train_common.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using nn::Scalar;

enum class ArchType { MLP, Transformer };

// ==================== 帮助信息 ====================
void print_usage(const char *prog)
{
    std::cout
        << "MNIST 手写数字训练程序 (引擎化架构，支持 MLP/Transformer)\n\n"
        << "用法: " << prog << " [选项]\n\n"
        << "选项:\n"
        << "  --arch <name>      模型架构: mlp/transformer (默认: mlp)\n"
        << "  --resume <path>    从已有模型恢复训练 (自动读取模型规格与架构)\n"
        << "  --save <path>      模型保存路径 (默认: mnist_model.bin)\n"
        << "  --dataset <path>   数据集目录 (默认: datasets/mnist_data)\n"
        << "  --epochs <n>       训练轮数 (默认: 10)\n"
        << "  --lr <lr>          学习率 (默认: 0.001)\n"
        << "  --batch-size <n>   批大小 (默认: 64)\n"
        << "  --optimizer <name> 优化器: sgd/sgd_momentum/adam/adamw/muon (默认: adam)\n"
        << "  --weight-decay <w> AdamW 权重衰减系数 (默认: 0.01)\n"
        << "  --gpu              启用 GPU 加速 (需要 Vulkan SDK)\n"
        << "  --cuda             启用 CUDA GPU 加速 (需要 CUDA Toolkit)\n"
        << "  --max-samples <n>  限制训练样本数 (用于快速测试, 默认: 全部)\n"
        << "  --shuffle-steps <true|false>  每 epoch 打乱 batch 顺序 (默认: true)\n"
        << "\n"
        << "MLP 专用:\n"
        << "  --layer-dims <d1,d2,...>  各层维度，逗号分隔 (默认: 784,512,256,128,64,10)\n"
        << "\n"
        << "Transformer 专用:\n"
        << "  --d-model <n>      模型维度 (默认: 64)\n"
        << "  --num-heads <n>    注意力头数 (默认: 4)\n"
        << "  --num-layers <n>   Transformer 层数 (默认: 2)\n"
        << "  --d-ff <n>         FFN 中间维度 (默认: 128)\n"
        << "  --patch-size <n>   patch 大小 (默认: 7, 28/7=4 → 16 patches)\n"
        << "  --eval-samples <n> 评估样本数 (默认: 200，避免评估过慢)\n"
        << "  --osc-guard <on|off>  振荡检测自动降 lr (默认: on)\n"
        << "  --osc-window <n>   振荡检测窗口大小 (默认: 20)\n"
        << "  --osc-threshold <f> 振荡反转率阈值 (默认: 0.55)\n"
        << "\n"
        << "学习率调度:\n"
        << "  --lr-schedule <type> 学习率调度: fixed/cosine (默认: fixed)\n"
        << "                   cosine: 余弦退火，lr 从初始值衰减到 min-lr\n"
        << "  --warmup-epochs <n> 线性预热轮数 (默认: 0, 即不预热)\n"
        << "  --min-lr <lr>     余弦退火最低学习率 (默认: 1e-6)\n"
        << "  --lr-per-epoch <v1,v2,...>  手动指定每轮学习率 (逗号分隔，优先级最高)\n"
        << "  --help             显示此帮助信息\n";
}

// ==================== 命令行参数 ====================
struct TrainConfig
{
    ArchType arch = ArchType::MLP;
    std::string save_path = "mnist_model.bin";
    std::string dataset_path = "datasets/mnist_data";
    std::string resume_path;
    std::string optimizer_name = "adam";
    int epochs = 10;
    Scalar lr = 0.001;
    Scalar weight_decay = 0.01f;  // AdamW 权重衰减系数
    std::size_t batch_size = 64;
    bool load_existing = false;
    bool gpu_enabled = false;
    bool cuda_enabled = false;
    int max_train_samples = -1;  // -1 表示使用全部
    bool shuffle_steps = true;   // 每 epoch 打乱 batch 顺序

    // MLP 参数
    std::vector<std::size_t> layer_dims;

    // Transformer 参数
    std::size_t d_model = nn::MNIST_TF_D_MODEL;
    std::size_t num_heads = nn::MNIST_TF_NUM_HEADS;
    std::size_t d_ff = nn::MNIST_TF_D_FF;
    std::size_t num_layers = nn::MNIST_TF_NUM_LAYERS;
    std::size_t patch_size = nn::MNIST_PATCH_SIZE;
    std::size_t eval_samples = 200;  // Transformer 评估样本上限

    // 学习率调度
    std::string lr_schedule = "fixed";  // fixed / cosine
    int warmup_epochs = 0;              // 线性预热轮数
    Scalar min_lr = 1e-6f;              // 余弦退火最低 lr
    std::vector<Scalar> lr_per_epoch;   // 手动指定每轮 lr（为空则自动计算）
};

TrainConfig parse_args(int argc, char *argv[])
{
    TrainConfig cfg;
    nn::cli::TrainCommonArgs common;
    // 程序专属默认值（覆盖 TrainCommonArgs 的默认值）
    common.batch_size = 64;
    common.weight_decay = 0.01f;
    common.min_lr = 1e-6f;
    common.lr_schedule = "fixed";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        // 通用训练参数委托给 nn::cli::parse_train_common_args
        else if (nn::cli::parse_train_common_args(argc, argv, i, common))
        {
            continue;
        }
        else if (arg == "--arch" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "mlp")
                cfg.arch = ArchType::MLP;
            else if (v == "transformer" || v == "tf")
                cfg.arch = ArchType::Transformer;
            else
            {
                std::cerr << "未知 --arch: " << v << "，可选: mlp, transformer\n";
                std::exit(1);
            }
        }
        else if (arg == "--resume" && i + 1 < argc)
        {
            cfg.resume_path = argv[++i];
            cfg.load_existing = true;
        }
        else if (arg == "--save" && i + 1 < argc)
        {
            cfg.save_path = argv[++i];
        }
        else if (arg == "--dataset" && i + 1 < argc)
        {
            cfg.dataset_path = argv[++i];
        }
        else if (arg == "--layer-dims" && i + 1 < argc)
        {
            std::string dims_str = argv[++i];
            std::stringstream ss(dims_str);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                auto v = nn::parse_number<std::size_t>(token);
                if (!v) { std::cerr << "无效层维度: " << v.error().message << "\n"; std::exit(1); }
                if (*v == 0) { std::cerr << "层维度不能为 0\n"; std::exit(1); }
                cfg.layer_dims.push_back(*v);
            }
            if (cfg.layer_dims.size() < 2)
            {
                std::cerr << "--layer-dims 至少需要 2 个维度\n";
                std::exit(1);
            }
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
        else if (arg == "--patch-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --patch-size: " << v.error().message << "\n"; std::exit(1); }
            cfg.patch_size = *v;
        }
        else if (arg == "--max-samples" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --max-samples: " << v.error().message << "\n"; std::exit(1); }
            cfg.max_train_samples = *v;
        }
        else if (arg == "--eval-samples" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --eval-samples: " << v.error().message << "\n"; std::exit(1); }
            cfg.eval_samples = *v;
        }
        else if (arg == "--shuffle-steps" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "true" || v == "1")
                cfg.shuffle_steps = true;
            else if (v == "false" || v == "0")
                cfg.shuffle_steps = false;
            else
            {
                std::cerr << "无效 --shuffle-steps: " << v << "，可选: true, false\n";
                std::exit(1);
            }
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n使用 --help 查看用法\n";
            std::exit(1);
        }
    }

    // 从通用参数回填到程序专属配置
    cfg.epochs = common.epochs;
    cfg.lr = common.lr;
    cfg.batch_size = common.batch_size;
    cfg.optimizer_name = common.optimizer;
    cfg.weight_decay = common.weight_decay;
    cfg.gpu_enabled = common.use_gpu;
    cfg.cuda_enabled = common.use_cuda;
    cfg.lr_schedule = common.lr_schedule;
    cfg.warmup_epochs = common.warmup_epochs;
    cfg.min_lr = common.min_lr;
    cfg.lr_per_epoch = std::move(common.lr_per_epoch);

    return cfg;
}

// ==================== 构建模型规格 ====================
nn::ModelSpec build_spec(const TrainConfig &cfg)
{
    if (cfg.arch == ArchType::Transformer)
    {
        return nn::make_mnist_transformer_spec(
            cfg.patch_size,
            cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers);
    }
    // MLP
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = cfg.layer_dims.empty() ? nn::MNIST_LAYER_DIMS : cfg.layer_dims;
    return spec;
}

// ==================== 主函数 ====================
int main(int argc, char *argv[])
{
    TrainConfig cfg = parse_args(argc, argv);

    // ── 构建规格 ─────────────────────────────────────────────
    nn::ModelSpec spec = build_spec(cfg);

    // ── 如果 --resume，从文件读取规格覆盖 CLI 参数 ──────────
    if (cfg.load_existing)
    {
        auto spec_result = nn::peek_model_spec(cfg.resume_path);
        if (spec_result)
        {
            if (spec_result->type == nn::ModelType::MLP)
            {
                std::cout << "从模型文件读取 MLP 规格\n";
                spec = std::move(*spec_result);
                cfg.arch = ArchType::MLP;
            }
            else if (spec_result->is_transformer())
            {
                std::cout << "从模型文件读取 Transformer 规格\n";
                spec = std::move(*spec_result);
                cfg.arch = ArchType::Transformer;
                cfg.patch_size = spec.patch_size != 0 ? spec.patch_size : nn::MNIST_PATCH_SIZE;
                cfg.d_model = spec.d_model;
                cfg.num_heads = spec.num_heads;
                cfg.d_ff = spec.d_ff;
                cfg.num_layers = spec.num_layers;
            }
            else if (spec_result->type == nn::ModelType::Unknown)
            {
                std::cout << "旧格式模型文件 (V1)，使用命令行参数 (--arch)\n";
            }
            else
            {
                std::cerr << "模型文件类型不支持 (type="
                          << static_cast<uint32_t>(spec_result->type) << ")\n";
                return 1;
            }
        }
        else
        {
            std::cerr << "读取模型规格失败: " << spec_result.error().message
                      << "，使用命令行参数。\n";
        }
    }

    // ── 打印配置 ─────────────────────────────────────────────
    std::cout << "========================================\n";
    std::cout << "  MNIST 手写数字训练 (引擎化架构)\n";
    std::cout << "========================================\n";
    std::cout << "  架构: " << (cfg.arch == ArchType::Transformer ? "Transformer (ViT)" : "MLP") << "\n";

    if (cfg.arch == ArchType::Transformer)
    {
        const std::size_t grid = nn::MNIST_IMG_SIZE / cfg.patch_size;
        const std::size_t num_patches = grid * grid;
        std::cout << "  Patch: " << nn::MNIST_IMG_SIZE << "/" << cfg.patch_size
                  << " → " << grid << "×" << grid << "=" << num_patches << " patches\n";
        std::cout << "  d_model: " << cfg.d_model
                  << "  heads: " << cfg.num_heads
                  << "  layers: " << cfg.num_layers
                  << "  d_ff: " << cfg.d_ff << "\n";
    }
    else
    {
        const auto &dims = spec.is_mlp() ? spec.layer_dims : nn::MNIST_LAYER_DIMS;
        std::cout << "  网络: ";
        for (std::size_t i = 0; i < dims.size(); ++i)
        {
            std::cout << dims[i];
            if (i < dims.size() - 2)
                std::cout << "(LayerNorm+GeLU)";
            if (i < dims.size() - 1)
                std::cout << " -> ";
        }
        std::cout << "\n";
    }

    std::cout << "  优化器: " << cfg.optimizer_name << "  学习率: " << cfg.lr << "\n";
    std::cout << "  轮数: " << cfg.epochs << "  批大小: " << cfg.batch_size << "\n";
    std::cout << "  GPU: " << (cfg.gpu_enabled ? "启用" : "禁用")
              << "  打乱: " << (cfg.shuffle_steps ? "启用" : "禁用") << "\n";
    std::cout << "  模型: " << (cfg.load_existing ? cfg.resume_path : "(从头训练)")
              << " -> " << cfg.save_path << "\n";
    std::cout << "========================================\n\n";

    // ── 创建计算引擎 ─────────────────────────────────────────
    // 引擎必须先于 model 构造并晚于 model 析构（model 持有 engine 的非拥有指针）
    nn::cli::EngineConfig eng_cfg;
    eng_cfg.use_gpu = cfg.gpu_enabled;
    eng_cfg.use_cuda = cfg.cuda_enabled;
    auto engine = nn::cli::create_engine(eng_cfg, std::cout);

    // ── 加载数据 ─────────────────────────────────────────────
    std::cout << "加载数据: " << cfg.dataset_path << " ..." << std::endl;
    const std::size_t train_max =
        (cfg.max_train_samples > 0) ? static_cast<std::size_t>(cfg.max_train_samples) : 0;
    auto csv_train_result = nn::cli::load_mnist_csv(cfg.dataset_path + "/train.csv", train_max);
    if (!csv_train_result) { std::cerr << "Error: " << csv_train_result.error().message << '\n'; return 1; }
    auto [train_x, train_y] = std::move(*csv_train_result);

    auto csv_test_result = nn::cli::load_mnist_csv(cfg.dataset_path + "/test.csv");
    if (!csv_test_result) { std::cerr << "Error: " << csv_test_result.error().message << '\n'; return 1; }
    auto [test_x, test_y] = std::move(*csv_test_result);
    std::cout << "训练集: " << train_x.cols() << " 样本, 测试集: " << test_x.cols() << " 样本\n" << std::endl;

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    auto model_result = nn::build_mnist_model_from_spec(*engine, spec);
    if (!model_result)
    {
        std::cerr << "构建模型失败: " << model_result.error().message << '\n';
        return 1;
    }
    auto model = std::move(*model_result);

    if (cfg.load_existing)
    {
        auto load_result = nn::load_model(cfg.resume_path, model);
        if (load_result)
        {
            std::cout << "已加载模型: " << cfg.resume_path << "\n" << std::endl;
        }
        else
        {
            std::cerr << "加载模型失败: " << load_result.error().message
                      << "，将从头训练。\n" << std::endl;
        }
    }

    // ── 训练 ─────────────────────────────────────────────────
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
    const std::size_t num_batches = train_x.cols() / cfg.batch_size;
    if (num_batches == 0)
    {
        std::cerr << "训练样本数 (" << train_x.cols() << ") 小于 batch_size ("
                  << cfg.batch_size << ")，请减小 --batch-size 或增大 --max-samples\n";
        return 1;
    }

    // ── 预分配 batch 缓冲区 ──────────────────────────────────
    nn::Matrix x_batch(train_x.rows(), cfg.batch_size);
    nn::Matrix y_batch(train_y.rows(), cfg.batch_size);

    std::vector<std::size_t> sample_indices(train_x.cols());
    for (std::size_t i = 0; i < sample_indices.size(); ++i)
        sample_indices[i] = i;
    std::mt19937_64 shuffle_rng{std::random_device{}()};

    auto t_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < cfg.epochs; ++epoch)
    {
        // ── 学习率调度：每 epoch 开始时调整 ──
        Scalar epoch_lr = nn::cli::compute_epoch_lr(lr_sched_cfg, epoch);
        if (epoch_lr != optimizer_current_lr)
        {
            optimizer->set_lr(epoch_lr);
            optimizer_current_lr = epoch_lr;
        }

        auto ep_start = std::chrono::steady_clock::now();
        Scalar total_loss = 0.0;

        if (cfg.shuffle_steps)
            std::shuffle(sample_indices.begin(), sample_indices.end(), shuffle_rng);

        for (std::size_t batch = 0; batch < num_batches; ++batch)
        {
            const std::size_t start = batch * cfg.batch_size;

            // ── 提取 batch（按 shuffle 顺序逐列拷贝） ──
            const std::size_t R_x = train_x.rows();
            const std::size_t R_y = train_y.rows();
            const std::size_t C_train = train_x.cols();
            const auto x_src = train_x.span().data();
            const auto y_src = train_y.span().data();
            auto x_dst = x_batch.span().data();
            auto y_dst = y_batch.span().data();
            for (std::size_t b = 0; b < cfg.batch_size; ++b)
            {
                const std::size_t src_col = sample_indices[start + b];
                for (std::size_t r = 0; r < R_x; ++r)
                    x_dst[r * cfg.batch_size + b] = x_src[r * C_train + src_col];
                for (std::size_t r = 0; r < R_y; ++r)
                    y_dst[r * cfg.batch_size + b] = y_src[r * C_train + src_col];
            }

            // ── Matrix → Tensor（上传到引擎设备） ──
            auto x_tensor_r = engine->from_matrix(x_batch);
            if (!x_tensor_r) {
                std::cerr << "\nfrom_matrix(x_batch) failed: " << x_tensor_r.error().message << '\n';
                return 1;
            }
            auto y_tensor_r = engine->from_matrix(y_batch);
            if (!y_tensor_r) {
                std::cerr << "\nfrom_matrix(y_batch) failed: " << y_tensor_r.error().message << '\n';
                return 1;
            }

            // ── 批量录制：forward + loss + backward 录制到单一 command buffer ──
            auto bb = engine->begin_batch();
            if (!bb) { std::cerr << "\nbegin_batch failed: " << bb.error().message << '\n'; return 1; }

            // ── 前向 ──
            auto out_r = model.forward(*x_tensor_r);
            if (!out_r) {
                std::cerr << "\nForward pass failed: " << out_r.error().message << '\n';
                return 1;
            }

            // ── 计算损失 ──
            auto loss_result = ce_loss.forward(*engine, *out_r, *y_tensor_r);
            if (!loss_result) {
                std::cerr << "\nLoss computation failed: " << loss_result.error().message << '\n';
                return 1;
            }
            Scalar loss = *loss_result;
            total_loss += loss;

            // ── 反向 ──
            auto grad_r = ce_loss.backward();
            if (!grad_r) { std::cerr << "\nLoss backward failed: " << grad_r.error().message << '\n'; return 1; }

            auto bwd_r = model.backward(*grad_r);
            if (!bwd_r) { std::cerr << "\nModel backward failed: " << bwd_r.error().message << '\n'; return 1; }

            // ── 提交并等待所有录制的命令 ──
            auto eb = engine->end_batch();
            if (!eb) { std::cerr << "\nend_batch failed: " << eb.error().message << '\n'; return 1; }

            // ── 优化器 step + 梯度清零 ──
            auto step_result = optimizer->step();
            if (!step_result) {
                std::cerr << "\n优化器 step 失败: " << step_result.error().message << '\n';
                return 1;
            }

            auto zero_result = optimizer->zero_grad();
            if (!zero_result) {
                std::cerr << "\n优化器 zero_grad 失败: " << zero_result.error().message << '\n';
                return 1;
            }

            // ── 进度显示 ──
            if ((batch + 1) % 10 == 0 || batch + 1 == num_batches)
            {
                auto batch_now = std::chrono::steady_clock::now();
                Scalar batch_ms = std::chrono::duration<Scalar, std::milli>(batch_now - ep_start).count();
                std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                          << "  batch " << batch + 1 << "/" << num_batches
                          << "  loss: " << std::fixed << std::setprecision(4) << loss
                          << "  time: " << std::setprecision(0) << batch_ms << "ms"
                          << "   " << std::flush;
            }
        }

        auto ep_end = std::chrono::steady_clock::now();
        Scalar ep_sec = std::chrono::duration<Scalar>(ep_end - ep_start).count();

        Scalar avg_loss = total_loss / num_batches;
        // MLP 全量评估，Transformer 截取前 eval_samples 个样本评估
        const std::size_t eval_n = (cfg.arch == ArchType::Transformer) ? cfg.eval_samples : 0;
        auto train_acc_r = nn::cli::evaluate_mnist(model, *engine, train_x, train_y, eval_n);
        auto test_acc_r  = nn::cli::evaluate_mnist(model, *engine, test_x, test_y, eval_n);
        if (!train_acc_r || !test_acc_r)
        {
            const auto &err = !train_acc_r ? train_acc_r.error() : test_acc_r.error();
            std::cerr << "\n  评估失败: " << err.message << std::endl;
            return 1;
        }
        Scalar train_acc = *train_acc_r;
        Scalar test_acc  = *test_acc_r;

        std::cout << "\r  Epoch " << epoch + 1 << "/" << cfg.epochs
                  << "  lr=" << std::scientific << std::setprecision(4) << optimizer_current_lr
                  << "  loss=" << std::fixed << std::setprecision(4) << avg_loss
                  << "  train_acc=" << std::setprecision(2) << train_acc * 100.0 << "%"
                  << "  test_acc=" << test_acc * 100.0 << "%"
                  << "  time=" << std::setprecision(1) << ep_sec << "s"
                  << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    Scalar total_sec = std::chrono::duration<Scalar>(t_end - t_start).count();

    // ── 保存模型 ─────────────────────────────────────────────
    auto save_result = nn::save_model(cfg.save_path, model, spec);
    if (!save_result)
    {
        std::cerr << "保存模型失败: " << save_result.error().message << '\n';
        return 1;
    }
    std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1) << total_sec << "s" << std::endl;

    return 0;
}
