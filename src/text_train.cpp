// ── GPT 文本生成训练程序（引擎化架构） ──────────────────────────────────────
//
// 输入：.nndataset 数据集（docs/todo/19-unified-dataset.md）——
//   tokenize、loss 掩码、词表都在生成期（dataset_gen）定好并嵌入数据集。
//
// 数据流（P2b：数据集 → **样本** → 训练工厂）：
//   nn::Dataset::open → nn::TextSampleSet（每样本 = 一个 seq_len 滑动窗口，
//     token/目标/掩码/doc_ids 由样本层统一装配）→ nn::TextTrainer（训练驱动）
//     → TrainingSession（Model + Optimizer + micro-step 累积）→ GPTModel → 稀疏 CE
//     → backward → optimizer step。
//   本文件只负责：数据集/词表加载、样本集切分、模型构建、恢复、训练配方映射、
//   收尾产物（.nnmodel）与退出码。训练循环在 `nn::TextTrainer::fit()`。
//
// 引擎选择：--gpu 启用 GpuEngine（需要 Vulkan），否则 CpuEngine。
//
// 模块结构（P1-1 模块化 + P2b 训练工厂）：
//   * cli/text_train_config.hpp  ：TrainConfig + --help + argv 解析
//                                  （唯一允许 std::exit 的层）
//   * cli/text_train_recovery.hpp：--resume（角色门禁 / 按规格重建 / 载参）
//   * cli/text_train_loop.hpp    ：CLI hooks（梯度日志/探针/断点）+ 调 fit()
//   * cli/text_train_checkpoint.hpp：断点与推理模型产出（记账 + 退出码语义）
//   * text_trainer.hpp / dataset_samples.hpp / training_session.hpp：库层训练工厂
// 本文件不直接终止进程：`main` 是唯一出口（Result → 退出码）。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>
#include <neuralnet.cpp/precision.hpp>
#include <neuralnet.cpp/text_trainer.hpp>   // 训练工厂（数据集 + 样本 + 循环）
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>
#include <neuralnet.cpp/cli/cli_precision_warning.hpp>
#include <neuralnet.cpp/cli/text_train_checkpoint.hpp>
#include <neuralnet.cpp/cli/text_train_config.hpp>
#include <neuralnet.cpp/cli/text_train_loop.hpp>
#include <neuralnet.cpp/cli/text_train_recovery.hpp>
#include <neuralnet.cpp/core_profile.hpp>

namespace
{
// ── 训练主流程（除 CLI 参数解析外**不终止进程**；见文件头）──────────────
// 返回值 = 进程退出码：0 = 正常；1 = 断点保存失败（详见 checkpoint 模块）。
// 错误经 Result 交给 main 打印 + 退出码 1（P1-1 口径）。
[[nodiscard]] nn::Result<int> run_text_train(nn::cli::TrainConfig cfg)
{
    // ── 断点状态采纳（P1-2 第三刀 B）────────────────────────────────────
    // 必须**早于**建模型/建优化器/切窗口：状态段是断点的权威来源（精度 profile、
    // 曲线形状、网格、起点都在里面），CLI 再给一遍即报错（禁列见 recovery 模块）。
    nn::cli::ResumeStateAdoption adoption;
    if (cfg.load_existing)
    {
        NN_TRY(adopt_r, nn::cli::adopt_resume_state(cfg, std::cout, std::cerr));
        adoption = std::move(*adopt_r);
        // adopt 会用断点状态段覆盖 cfg.precision：必须在覆盖之后再校验一次
        // "精度 × activation offload" 组合，否则 rc0 承诺的配置期拒绝会被绕开
        // （残余路径只能到首个 block 才由库层 fail-fast）。这里经 Result 上抛
        // （不用 std::exit），保持三模块不终止进程的不变量。
        if (auto combo_err = nn::cli::precision_offload_combo_error(cfg))
            NN_FAIL(*combo_err);
    }

    // ── 打开数据集（.nndataset；token 流/掩码/词表都在生成期定好）──────
    std::cout << "加载数据集: " << cfg.text_path << " ..." << std::endl;
    NN_TRY_MSG(ds_r, "打开数据集失败", nn::Dataset::open(cfg.text_path));
    const nn::Dataset &dataset = *ds_r;
    const std::string &loss_scope = dataset.info().loss_scope;

    // 词表来自数据集配套（内嵌三层 kvrec）；保存模型时原样嵌出（模型 v6）。
    NN_TRY_MSG(tokenizer_r, "加载数据集内嵌词表失败",
               nn::load_tokenizer_from_kvrec(dataset.vocab_kvrec()));
    std::unique_ptr<nn::Tokenizer> tokenizer = std::move(*tokenizer_r);
    const std::string tokenizer_bytes = nn::write_nnvocab(dataset.vocab_kvrec());

    // ── 训练样本集（P2b：样本概念的唯一载体）──────────────────────────
    // 每样本 = 一个 seq_len 滑动窗口（可能跨 doc 边界）；目标 = 输入左移一位。
    // 窗口化策略留在训练侧（TextWindowConfig），`.nndataset` 只存 docs。
    const nn::TextWindowConfig window{cfg.seq_len, cfg.stride, tokenizer->pad_id()};
    NN_TRY_MSG(train_samples_r, "读取训练样本集失败",
               nn::TextSampleSet::create(dataset, nn::DatasetSubset::Train, window));
    nn::TextSampleSet train_samples = std::move(*train_samples_r);
    if (train_samples.sample_count() == 0)
        NN_FAIL("数据集 train 子集为空（无 token）: " + cfg.text_path);

    std::optional<nn::TextSampleSet> test_samples;
    // 空 test 子集（合法文件：test 全是空 doc）⇒ 视为无 test、跳过评估；
    // 但启动信息行保持 v1.9.0 锚（打印"样本数: 0"），不静默吞掉测试集概况。
    std::optional<std::size_t> empty_test_tokens;
    if (dataset.info().test_docs > 0)
    {
        NN_TRY_MSG(test_samples_r, "读取测试样本集失败",
                   nn::TextSampleSet::create(dataset, nn::DatasetSubset::Test, window));
        if (test_samples_r->sample_count() > 0)
            test_samples = std::move(*test_samples_r);
        else
            empty_test_tokens = test_samples_r->token_count();
    }

    std::cout << "训练子集: " << train_samples.doc_count() << " docs / "
              << train_samples.token_count() << " tokens  loss_scope=" << loss_scope
              << "\n";
    // 文档感知：doc_ids 非空（每 doc = 一篇文档）即启用块对角掩码。
    if (!train_samples.corpus().doc_ids.empty())
        std::cout << "文档感知掩码已启用（每行 = 一篇文档，"
                  << train_samples.corpus().doc_ids.back() << " 篇文档）\n";

    // loss 掩码在生成期已定（loss_scope=all 为全 1）：仅掩码内的预测目标
    // 参与 loss（对话 SFT 时 = <|assistant|>…<|end_of_assistant|> 段）。
    if (loss_scope == "assistant")
    {
        std::size_t asst_tokens = 0;
        for (unsigned char m : train_samples.corpus().loss_mask) asst_tokens += m;
        std::cout << "assistant loss 掩码已启用（" << asst_tokens << " / "
                  << train_samples.token_count()
                  << " tokens 作为预测目标参与 loss）\n";
    }

    std::cout << "滑动窗口: seq_len=" << cfg.seq_len
              << " stride=" << train_samples.stride()
              << " 样本数=" << train_samples.sample_count() << "\n" << std::endl;

    // 空 test 子集的启动信息（v1.9.0 锚：仍打印"样本数: 0"；评估跳过由
    // TextTrainer 的 nullopt test 语义保证）。
    if (empty_test_tokens)
    {
        if (loss_scope == "assistant")
            std::cout << "测试集 assistant 掩码: 0 / " << *empty_test_tokens
                      << " tokens 参与评估\n";
        std::cout << "测试集: " << dataset.info().test_docs << " docs  样本数: 0  tokens: "
                  << *empty_test_tokens << std::endl;
    }

    // ── 数据集身份（P1-2 断点门禁的"内容"侧）────────────────────────────
    // 路径是**位置**（机器相关、CLI 优先），sha256 + 网格是**身份**（断点权威）。
    // 身份不同 ⇒ 样本网格、step 号语义、lr 曲线形状全变 ⇒ 续训路径硬报错
    // （出口 = 从 .nnmodel 开新一轮，docs/todo/23 §2.1.1）。
    nn::DatasetIdentity dataset_identity;
    dataset_identity.path = cfg.text_path;
    NN_TRY_MSG(sha_r, "计算数据集身份失败", nn::sha256_file_hex(cfg.text_path));
    dataset_identity.sha256 = std::move(*sha_r);
    dataset_identity.num_tokens = train_samples.token_count();
    dataset_identity.num_docs = train_samples.doc_count();
    dataset_identity.num_samples = train_samples.sample_count();
    dataset_identity.loss_scope = loss_scope;
    // steps_per_epoch 是**派生量**（样本数 ÷ batch），与训练工厂同一公式。
    dataset_identity.steps_per_epoch =
        train_samples.steps_per_epoch(cfg.batch_size);

    // ── 数据集身份门禁（硬错、无逃生阀；docs/todo/23 §2.1.1）────────────
    // 数据集变 ⇒ 样本数变 ⇒ steps_per_epoch 变 ⇒ step 级调度的 total_steps、
    // global_step 语义、epoch 边界同时改变 ⇒ "同一断点 + 不同数据集"没有正确
    // 的续训语义。合法出口 = 从 .nnmodel 开新一轮（下面的 warm start 分支）。
    if (adoption.adopted)
    {
        NN_TRY_CHECK(nn::cli::verify_dataset_identity(
            adoption.state.dataset, dataset_identity, cfg.resume_path));
    }

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
    // β1/β2 的消费方 = 表中 beta = true 的优化器（sgd 家族/muon 忽略）
    if (const auto *om = nn::find_optimizer(cfg.optimizer_name); om != nullptr && om->beta)
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
    NN_TRY_MSG(engine_res, "引擎创建失败", nn::cli::create_engine(eng_cfg, std::cout));
    auto raw_engine = std::move(*engine_res);

    // ── 多精度：边界 cast 已下沉基类（NVI，原 PrecisionEngine，f16 存储）────────────────────────────────
    // f16 存储（--f16 / --precision-*）直接由引擎入口处理：有 in-kernel f16
    // 变体/f16 GEMM 的算子直读写 f16，其余算子在入口抬 f32 计算 → 按目标精度
    // 落回（expr_spec_key 本身不含精度维度，f16 变体按 (key, 精度签名) 另行注册）。
    // 全 f32 时入口为快速直通分支，与原生引擎逐字节一致。
    // 层内的 p_.param/compute/stable/optimizer 是唯一精度来源（§8.5 G4）。
    // 触发条件 = **真的有 f16 存储**（四个槽里任一为 f16）。不能用
    // `!is_profile_f32()`：`param = t1_58` 时三值并不产生 f16 存储（latent 恒 f32、
    // 只有量化缓冲是 f16），那样会打出误导性的"f16 存储已启用"。
    const bool f16_storage = (cfg.precision.param == nn::Precision::F16 ||
                              cfg.precision.compute == nn::Precision::F16 ||
                              cfg.precision.stable == nn::Precision::F16 ||
                              cfg.precision.optimizer == nn::Precision::F16);
    if (f16_storage)
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
    // 探针门：默认构建里 on 恒 false（且不求值 getenv），mark 是空判定——
    // 训练热路径与探针无关（实现见 cli/text_train_loop.hpp）。
    const nn::cli::TextTrainMemProbe mem_probe = nn::cli::make_text_train_mem_probe();
    mem_probe.mark(*engine, "engine-init");

    // ── 构建模型（绑定引擎） ─────────────────────────────────
    nn::Result<nn::Model> model_build;
    if (cfg.model_type == "rapt")
    {
        // RLA 强约束：位置编码必须 RoPE（施加在 Q/K 进 ReLU 之前），强制覆盖
        cfg.pos_encoding = nn::PosEncodingType::RoPE;
        // ── RAPT 的三值接线尚未落地（P1.5 范围，docs 21 §4.8.3）────────────
        // RLA 注意力自持 4 个 Linear 且当前**不转发**精度 profile（与 roadmap
        // P2「RAPT/CNN f16」同源）→ 留成"静默半三值"是最坏的选项，前置拒绝。
        if (cfg.precision.param == nn::Precision::T1_58)
            NN_FAIL("RAPT 尚不支持三值权重（--precision-param t1_58）：RLA 注意力的"
                    " 4 个 Linear 还没接 make_linear_layer，\n"
                    "见 docs/todo/21-quantized-weights.md §4.8.3。\n"
                    "  · 三值 GPT：改用 --model gpt\n"
                    "  · 三值 MLP：mnist_train --arch mlp --precision-param t1_58");
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
    NN_TRY_MSG(model_build_r, "构建模型失败", std::move(model_build));
    auto model = std::move(*model_build_r);
    mem_probe.mark(*engine, "model-built");

    // ── 打印精度配置 ──
    {
        const auto& pp = cfg.precision;
        // 三值（T1_58）与"f16 存储"是两件事：前者 latent 恒 f32、只有量化缓冲是
        // f16（P1.5-W2 起覆盖 GPT 的 4 个投影 + fc1/fc2 + LM head）。
        if (pp.param == nn::Precision::T1_58)
            std::cout << "[精度] 三值权重已启用（T1_58 / BitLinear：量化 forward + "
                         "STE backward，latent 恒 f32）\n"
                         "  [范围] GPT（注意力 4 投影 + FFN fc1/fc2 + 未 tie 的 LM head）；"
                         "RAPT/Transformer 见 docs/todo/21-quantized-weights.md §4.8.3\n";
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

    // 三值权重（P1.5，docs 21 §4.7/§4.8.3）：param=t1_58 → 线性层用 BitLinear。
    // 规格是**加载时的权威来源**（缺键 = None → 旧文件零破坏）；--resume 时下面
    // 会用文件里的规格覆盖本行。
    spec.weight_quant = (cfg.precision.param == nn::Precision::T1_58)
                            ? nn::WeightQuant::T1_58 : nn::WeightQuant::None;


    // ── 断点恢复（--resume / --resume-epoch / --resume-step）──────────────
    // 角色门禁 / 按文件规格重建模型 / 载入参数都在 recovery 模块里；
    // 非致命失败（读不到规格、载参失败）打印提示后继续从头训练（旧行为）。
    if (cfg.load_existing)
    {
        // 不加语境前缀：角色门禁的报错文案是用户可见契约（docs/todo/23 §2.1），
        // 原样透出；返回的 TextTrainResume（spec_restored / weights_loaded）
        // 只有测试消费，主流程按进度打印即可。
        NN_TRY_CHECK(nn::cli::resume_text_train(*engine, cfg, model, spec, std::cout, std::cerr));
    }

    // ── 训练会话（P2b）：Model + Optimizer 同生共死 ───────────────────────
    // 精度配置透传：状态张量（m/v/momentum）按 profile.optimizer 创建，
    // 参数更新走 in-place（存储精度不可变，§8.3）。
    // 注意：模型侧的 flush/checkpoint/offload 已在上面直接设置（CPU offload 是
    // 警告后忽略，TrainingSession 会拒绝），这里不重复传。
    nn::TrainingConfig session_cfg;
    session_cfg.optimizer = cfg.optimizer_name;
    session_cfg.lr = cfg.lr;
    session_cfg.weight_decay = cfg.weight_decay;
    session_cfg.precision = cfg.precision;
    session_cfg.beta1 = cfg.beta1;
    session_cfg.beta2 = cfg.beta2;
    session_cfg.clip_grad_norm = cfg.max_norm;
    session_cfg.grad_accum_steps = cfg.accum_steps;
    NN_TRY_MSG(session_r, "创建训练会话失败",
               nn::TrainingSession::create(std::move(model), session_cfg));
    nn::TrainingSession session = std::move(*session_r);
    mem_probe.mark(*engine, "optimizer-created");

    // ── 优化器状态恢复（P1-2；含 Adam 的 `t`）────────────────────────────
    // 张量数量/形状由 `Optimizer::load_state` 校验（对不上明确报错，不静默冷启动）。
    // 位置必须在优化器构造之后：状态张量按优化器自身的缓冲形状回写。
    if (adoption.adopted)
    {
        NN_TRY_CHECK(nn::cli::load_optimizer_state(*engine, cfg.resume_path,
                                                   session.optimizer(), std::cout));
    }

    // ── 训练工厂：数据集（样本集）直接接入 ───────────────────────────────
    // 配方字段就是 TrainConfig 的基类（nn::TextTrainConfig）；数据集身份复用上面
    // 已算好的 sha256/网格（避免大文件二次哈希）。
    cfg.identity = dataset_identity;
    NN_TRY_MSG(trainer_r, "创建训练工厂失败",
               nn::TextTrainer::create(std::move(session), std::move(train_samples),
                                       std::move(test_samples), cfg));
    nn::TextTrainer trainer = std::move(*trainer_r);

    std::cout << "训练窗口样本数: " << trainer.sample_count() << "\n" << std::endl;

    // ── 训练循环（库层 TextTrainer；CLI 只装诊断/断点 hooks）──────────────
    NN_TRY_MSG(loop_r, "训练循环失败",
               nn::cli::run_text_train_loop(trainer, cfg, spec, tokenizer_bytes, mem_probe));
    const nn::TextTrainReport report = std::move(*loop_r);

    // 探针中止（NN_NAN_TRACE 命中，退出码 9）：跳过剖析报告与收尾保存，直接以该
    // 退出码结束（与旧 `return 9` 一致）。
    if (report.abort_code != 0) return report.abort_code;

    // ── step 剖析报告（NN_PROFILE=1 / NN_PROFILE_TRACE 时生效）────────────
    if (nn::prof::enabled())
        nn::prof::dump(stderr, report.elapsed_ns);

    // ── 保存模型（含规格 + 嵌入 tokenizer） ──────────────────
    // 推理模型（Model 角色）：三值层写打包形态 + τ（不含训练参数）。
    // 写不出去 = 这一轮训练对用户没有产出 ⇒ 失败退出（错误经 Result 回报）。
    NN_TRY_MSG(save_result, "推理模型保存失败",
               nn::cli::save_text_train_model(cfg.save_path, trainer.session().model(), spec,
                                              tokenizer_bytes));
    (void)save_result;
    std::cout << "\n训练完成! 总耗时: " << std::fixed << std::setprecision(1)
              << report.seconds << "s"
              << "  词表已嵌入模型文件" << std::endl;
    // 退出码语义（P1-1）：有断点没写出去 ⇒ 非零退出。推理模型文件已落盘，
    // 但"可续训"这一性质不成立，上层不得把它当成功（文案在 checkpoint 模块）。
    return nn::cli::text_train_ckpt_exit_code(report.ckpt_failures, std::cerr);
}
} // namespace

// ==================== 入口 ====================
// 本函数是唯一的进程出口（`main` 返回退出码）：解析层内部按 CLI 语义
// exit(0)/exit(1)（--help / 参数错误），其余一切经 `run_text_train` 的 Result
// 回报后在此统一打印并返回 1。
int main(int argc, char *argv[])
{
    nn::cli::TrainConfig cfg = nn::cli::parse_text_train_args(argc, argv);

    // ── 危险精度组合启动期告警（仅警告，不拒绝；docs/todo/24 §2.3 F1）────
    // optimizer=f16（含四字段全 f16）只打印一次到 stderr：不改变退出码、不阻断
    // 训练、不静默降级；`--f16` 与合法组合返回空串（静默）。
    if (const std::string warn = nn::cli::dangerous_precision_warning(cfg.precision);
        !warn.empty())
        std::cerr << warn;

    auto run = run_text_train(cfg);
    if (!run)
    {
        std::cerr << run.error().message << "\n";
        return 1;
    }
    return *run;
}
