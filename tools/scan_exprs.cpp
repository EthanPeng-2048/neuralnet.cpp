// ───────────────────────────────────────────────────────────────────────────
//  scan_exprs.cpp — AOT 算子融合：构建期**单步**工具（收集 + 合成）
//
//  收集：CPU 引擎 + 假张量跑一遍相关 Layer 的 forward/backward，使每个
//  dsl::compute / compute_reduce / compute_into 在 NN_EXPR_SCAN 记录模式下把
//  折叠出的 ExprSpec **结构**登记进全局注册表（按 expr_spec_key 去重）。
//  脚本里还有两条互补来源：`FusedAnchor<Expr>` / `FoldAnchor` 在静态初始化期
//  自登记（编译期可达、新增表达式无需改本文件；fold 的通用样例与注意力组合
//  分别由 `expr_fold.hpp` / `compute_layer_attention.hpp` 挂锚），以及本文件
//  里的 matmul-trans 显式登记块（运行期配置决定的结构）。
//  合成：同进程内直接调 `nn::tool::generate_fused_registry`
//  （tools/fused_generate.hpp，原独立工具 gen_fused）→
//  <out_dir>/fused_registry.hpp。**不经任何中间序列化**。
//
//  表达式**文本只出现在 Layer**；这里只是"执行 Layer 代码路径"以触达它们，
//  产物是派生物（结构 + SPIR-V），不是手写定义。
//
//  用法： scan_exprs <out_dir> <glslc_path> [--list-backends]
//
//  ⚠ 闭合世界安全网：若某条 Layer 路径未被覆盖，其内联表达式在 GPU 运行时
//    将硬报错（错误信息带 key），提醒把该路径补进扫描。新增融合表达式时优先
//    依赖锚点自登记；只有"运行期配置决定的结构"才需要在这里补 dry-run 块。
// ───────────────────────────────────────────────────────────────────────────

// NN_EXPR_SCAN 由本文件自行开启；`nn_enable_gpu_fusion` 还会用同名宏编译
// 使用者的 TU（让它们的 `dsl::compute` 调用点自登记）——此处必须防重定义。
#ifndef NN_EXPR_SCAN
#define NN_EXPR_SCAN
#endif

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "compute_cpu_engine.hpp"
#include "compute_layer.hpp"
#include "compute_loss.hpp"
#include "compute_optimizer.hpp"
#include "domain_cnn.hpp"
#include "domain_gpt.hpp"
#include "domain_mnist.hpp"
#include "domain_rla.hpp"
#include "expr_registry.hpp"
#include "fused_generate.hpp"   // 生成阶段（原 gen_fused.cpp；不经 .bin 中间文件）
#include "neuralnet.cpp/expr_fold.hpp"
#include "neuralnet.cpp/rl_experience.hpp"
#include "neuralnet.cpp/rl_policy.hpp"

#include <csignal>
#if __has_include(<stacktrace>)
  #include <stacktrace>
  #define NN_SCAN_HAS_STACKTRACE 1
#endif

// abort（NN_ASSERT）时打印调用栈：NN_ASSERT 只有断言点行号，缺"谁调的"——
// MSVC 侧也只见 abort 无栈。scan 是本机构建工具（clang + MSVC STL、带 -g），
// 单文件装钩、不动全库 core_assert（GCC/CI 的 <stacktrace> 链接风险规避）；
// 老编译器无 <stacktrace> 时降级为无栈消息（__has_include 守卫）。注：
// handler 内 to_string(stacktrace) 会分配、严格说非异步信号安全——abort
// 路径本已终止进程，这里只求尽力打印，不保证死锁免疫。
namespace {
void on_abort(int)
{
#ifdef NN_SCAN_HAS_STACKTRACE
    std::fprintf(stderr, "[scan] abort captured — call stack:\n%s\n",
                 std::to_string(std::stacktrace::current()).c_str());
#else
    std::fprintf(stderr, "[scan] abort captured（本机无 <stacktrace>，无调用栈）\n");
#endif
    std::fflush(stderr);
    std::_Exit(3);   // 不回 abort（避免二次 abort 丢输出）
}
} // namespace

// ── 扫描期精度配置 ────────────────────────────────────────────────────────
// dry-run 收集的是**结构**（与形状/精度无关），故只跑 profile_f32 一遍。
// 历史上曾跑第二遍 profile_f16 以枚举"精度签名"；签名已降级为运行期参数
// （生成阶段对每个结构发运行期分派 shader，键 `key#x`），该遍已删除。
// scan_prof 仍用于：占位张量按当前 pass 的 compute 精度创建（与真实运行同源）。

int main(int argc, char* argv[])
{
    // --list-backends：列出可用 emitter 后端（IR-D 多后端验证）
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--list-backends")
        {
            std::printf("[scan] 可用 emitter 后端（IR-D）:\n");
            // 排序：names() 走 unordered_map，顺序不定；工具输出必须确定性（铁律 8）
            std::vector<std::string> backends = nn::emitter_registry::names();
            std::sort(backends.begin(), backends.end());
            for (const auto& n : backends)
                std::printf("      - %s\n", n.c_str());
            return 0;
        }
    if (argc < 3)
    {
        std::fprintf(stderr,
                     "用法: scan_exprs <out_dir> <glslc_path> [--list-backends]\n"
                     "  收集 Layer 内联表达式的**结构**（锚点自登记 + dry-run + 模型 pass），\n"
                     "  原地合成 <out_dir>/fused_registry.hpp（不经 .bin 中间文件）。\n"
                     "  精度签名不再是构建期集合：生成阶段对每个结构发一份运行期\n"
                     "  精度分派 shader（键 key#x），运行时按真实精度填 PC `prec`。\n");
        return 2;
    }
    const std::string out_dir  = argv[1];
    const std::string glslc    = argv[2];
    std::signal(SIGABRT, &on_abort);   // NN_ASSERT → abort 带栈（见文件头）

    auto raw_engine = nn::CpuEngine::create();
    // 扫描期精度配置（占位张量按当前 pass 的 compute 精度创建，与真实运行同源）
    nn::PrecisionProfile scan_prof{};

    // ── dry-run 主体：跑一遍 Layer 的 forward/backward（只收集**结构**）──
    // engine 是**形参**（遮蔽外层 raw_engine）：f16 边界 cast 由基类 NVI 入口
    // 统一处理（原 PrecisionEngine 适配层已删除，见 docs/history/history.md「15 号归档」 §4.1）。
    [[maybe_unused]] const auto dry_run = [&](nn::ComputeEngine& engine, const nn::PrecisionProfile& prof)
    {
        scan_prof = prof;
        // M1（docs/internal/17 §4.1）：库外不再直构 Tensor——scan 输入经
        // 引擎创建（出生绑定与本段 dry-run 使用的 engine 一致）。
        const auto scan_tensor = [&](std::size_t rows, std::size_t cols)
        {
            return engine.create_tensor(rows, cols, scan_prof.compute);
        };
    // ── RoPE：forward + backward ─────────────────────────────────────────
    // （apply 与 apply_step 折叠出的结构相同，会自动去重）
    //
    // 形状无关融合：RowMod/RotateHalf 的周期/块大小是**运行时视图参数**
    // （不进 expr_spec_key），故 RoPE 结构不依赖 d_k——这里任意 d_k 收集到的
    // 结构完全相同（去重成一个），**任何 d_k（含非 2 的幂）都能命中该融合
    // shader**。下面保留多个 d_k 仅作覆盖验证；若未来表达式重新引入形状
    // 常量，需再补对应 dry-run。
    for (const std::size_t dk : {std::size_t{16}, std::size_t{32},
                                 std::size_t{64}, std::size_t{128}})
    {
        nn::RotaryEmbedding rope(dk);
        rope.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        (void)rope.init(engine);   // M6 段 C：层由 init 绑定引擎
        nn::Tensor q = scan_tensor(2 * dk, 8);   // rows 为 dk 的整数倍
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/false);
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/true);
        nn::Tensor q1 = scan_tensor(dk, 1);      // 增量推理（单位置）
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/false);
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/true);
    }

    // ── MSELoss forward（diff = pred-target；diff_sq = diff*diff）────────
    {
        const std::size_t R = 8, C = 5;
        nn::MSELoss mse;
        mse.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        nn::Tensor pred = scan_tensor(R, C);
        nn::Tensor target = scan_tensor(R, C);
        (void)mse.forward(engine, pred, target);
    }

    // ── CrossEntropyLoss 稠密 forward（grad=softmax-target；target*log_sm；
    //    log_col_sum = log(col_sum)）─────────────────────────────────────
    {
        const std::size_t C = 8, B = 5;
        nn::CrossEntropyLoss ce;
        ce.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        nn::Tensor logits = scan_tensor(C, B);
        nn::Tensor target = scan_tensor(C, B);
        (void)ce.forward(engine, logits, target);
    }

    // ── 优化器 step + 梯度裁剪（全部变体）────────────────────────────────
    // 各优化器均为 DSL 融合表达式 / 目标传递（dsl::compute_into）：
    //   sgd          : p += -lr*g
    //   sgd_momentum : v = β*v + (1-β)*g ; p += -lr*v
    //   adam         : m/v 更新 + p += delta
    //   adamw        : p *= (1-lr*wd) + adam
    //   muon         : v = μ*v + g；Nesterov；Newton–Schulz（含 A=bA+cA² 等）
    //   clip_grad_norm: g *= scale
    // 必须逐个 dry-run：GPU 运行时同一结构才能命中 AOT 融合 shader（闭合世界）。
    // 结构不依赖形状，任取小 R×C 即可（Muon 的 NS 只对 ≥2D 参数生效）。
    {
        const std::size_t R = 8, C = 5;
        // RV-04：名单来自 nn::optimizer_table()（唯一真源）——新增优化器自动覆盖
        for (const auto &meta : nn::optimizer_table())
        {
            nn::Tensor p = scan_tensor(R, C);
            nn::Tensor g = scan_tensor(R, C);
            // 精度配置与运行期同源（`text_train` 把 `cfg.precision` 透传给
            // create_optimizer）：优化器的状态/参数槽精度决定 K3 更新量
            // （`upd_prec = p_.param`）与写目标的存储精度 → 直接决定该结构的
            // 运行期精度签名。漏传 = scan 只按默认 f32 profile 登记优化器，
            // `--f16` 下 K3 的 `#10000`（F32 输入 + F16 输出）签名不被观察
            // ⇒ 无 `key#10000` 特化变体 ⇒ 逐成员派发回退 `#x` 且拿不到
            // `key#10000#b` 批量变体（P2-12 残差的结构性根因之一）。
            const nn::PrecisionProfile opt_prof = scan_prof;
            auto opt = nn::create_optimizer(meta.name, engine,
                                            std::vector<nn::TensorRef>{p},
                                            std::vector<nn::TensorRef>{g},
                                            nn::Scalar{1e-3f},
                                            /*weight_decay=*/nn::Scalar{0},
                                            /*precision=*/opt_prof);
            if (opt)
            {
                (void)opt->step();
                (void)opt->clip_grad_norm(nn::Scalar{1e3f});
            }
        }
    }

    // ── 算子融合二期（docs/internal/02-operator-fusion.md）：matmul 参与 IR 融合 ──
    // 结构 = Layer 内 dsl::matmul(A,B)+bias+relu 折叠后的派生物：
    //   前置 matmul 段（MatmulSpec）+ 尾逐元素链（Add + Max）。
    // 这里直接构造折叠后的结构并登记（与 dsl::compute 登记 spec 是同一机制），
    // 保证 GPU 运行时同一结构命中 AOT 融合 shader（闭合世界）。
    //   - transA/transB 是**运行期 operand layout**（PC `mm_trans`）→ **不进 key**：
    //     4 种转置组合折叠成同一个 key，登记一次即可（历史上此处登记 4 份，
    //     key 去掉转置后它们全被去重合并）。
    //   - k（求和维度）是形状参数 → 不进 key：任取一个 K 登记，运行时任何 K
    //     都命中同一融合 shader（同 RowMod/RotateHalf 的视图参数处理：不进 key，
    //     运行时填充）。
    {
        nn::ExprSpec s;
        s.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
        s.num_regs = 2;
        // transA/transB 取 (0,0) 作代表值：它们不进 key
        s.matmul   = nn::MatmulSpec{0, 1, /*transA=*/0, /*transB=*/0, /*k=*/8};
        // Add r0 = matmul(A,B) + bias（Input 2）
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                            nn::expr::matmul_op(), nn::expr::input(2), {}});
        // Max r1 = max(r0, 0)（relu）
        s.consts.push_back(nn::Scalar{0});
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Max), 1,
                            nn::expr::reg(0), nn::expr::cst(0), {}});
        nn::fused::global_registry().add(s);
        // 纯 matmul（无逐元素链）：输出 = matmul 结果
        {
            nn::ExprSpec pure;
            pure.views    = {nn::expr::linear(), nn::expr::linear()};
            pure.num_regs = 0;
            pure.matmul   = nn::MatmulSpec{0, 1, 0, 0, /*k=*/8};
            nn::fused::global_registry().add(pure);
        }
    }

    // ── 算子融合二期（docs/internal/02-operator-fusion.md S5）：matmul+归约组合（注意力结构）
    // bmm_reduce / bmm_denom 的 IR 等价物：matmul 段被归约指令消费，
    // 不物化 (M,N) 得分矩阵（kernel 内联点积重算）。结构不依赖形状。
    //   Q (M,K)，K 存储 (N,K)（transB=1）：QK^T = matmul(Q, K, transB)
    {
        // row_max(QK^T)：行归约 max（bmm_reduce ReduceOp::Max 等价）
        nn::ExprSpec s;
        s.views    = {nn::expr::linear(), nn::expr::linear()};
        s.num_regs = 1;
        s.matmul   = nn::MatmulSpec{0, 1, 0, 1, /*k=*/8};
        s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::RowMax), 0,
                            nn::expr::matmul_op(), {}, {}});
        nn::fused::global_registry().add(s);
        // row_sum(matmul)：行归约 sum（bmm_reduce ReduceOp::Sum 等价）
        nn::ExprSpec s2;
        s2.views    = {nn::expr::linear(), nn::expr::linear()};
        s2.num_regs = 1;
        s2.matmul   = nn::MatmulSpec{0, 1, 0, 1, /*k=*/8};
        s2.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::RowSum), 0,
                             nn::expr::matmul_op(), {}, {}});
        nn::fused::global_registry().add(s2);
        // 列方向：col_max(matmul)（bmm_reduce reduce_cols=false 等价）
        nn::ExprSpec s3;
        s3.views    = {nn::expr::linear(), nn::expr::linear()};
        s3.num_regs = 1;
        s3.matmul   = nn::MatmulSpec{0, 1, 0, 1, /*k=*/8};
        s3.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::ColMax), 0,
                             nn::expr::matmul_op(), {}, {}});
        nn::fused::global_registry().add(s3);
        // denom：row_sum(exp(QK^T - rb(row_max)))（bmm_denom 等价；row_max
        // 经 RowBroadcast 视图 (M,1)，不物化得分矩阵）
        nn::ExprSpec sd;
        sd.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::row_broadcast()};
        sd.num_regs = 3;
        sd.matmul   = nn::MatmulSpec{0, 1, 0, 1, /*k=*/8};
        sd.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Sub), 0,
                             nn::expr::matmul_op(), nn::expr::input(2), {}});
        sd.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Exp), 1,
                             nn::expr::reg(0), {}, {}});
        sd.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::RowSum), 2,
                             nn::expr::reg(1), {}, {}});
        nn::fused::global_registry().add(sd);
    }

    // ── CausalSelfAttention（IR 掩码组合 forward/backward）──────────
    // 掩码表达式 4 配置（causal / causal+alibi / causal+doc / causal+alibi+doc）
    // 各产生确定结构：m/l/W（matmul 段 + Row/Col/Batch 操作数 + 视图）与
    // backward 的 R/X（纯逐元素）。必须全部 dry-run 覆盖（闭合世界）。
    {
        const std::size_t d_model = 16, heads = 2, seq = 4, batch = 2;
        const auto run_csa = [&](nn::CausalSelfAttention& attn) {
            (void)attn.init(engine);
            nn::Tensor x = scan_tensor(d_model, batch * seq);
            // 禁止 (void) 吞错：forward 失败会让缓存为空，backward 直接
            //   在 batched_matmul 读空张量上 NN_ASSERT（栈无上下文难定位）
            auto fr = attn.forward(x);
            if (!fr)
            {
                std::fprintf(stderr, "[scan] CSA forward FAILED: %s\n",
                             fr.error().message.c_str());
                std::fflush(stderr);
                std::abort();   // 带栈停在真凶处
            }
            nn::Tensor grad = scan_tensor(d_model, batch * seq);
            auto br = attn.backward(grad);
            if (!br)
            {
                std::fprintf(stderr, "[scan] CSA backward FAILED: %s\n",
                             br.error().message.c_str());
                std::fflush(stderr);
                std::abort();
            }
        };
        for (const auto enc : {nn::PosEncodingType::Learned,
                               nn::PosEncodingType::ALiBi})
        {
            nn::CausalSelfAttention attn(d_model, heads, /*max_len=*/1024,
                                         /*seq_len=*/seq, enc);
            attn.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
            (void)attn.init(engine);   // M6 段 C：层由 init 绑定引擎
            run_csa(attn);
        }
        // doc 变体（doc_ids 每位置文档 id，长度 = batch*seq）
        {
            const std::size_t doc_ids[8] = {0, 0, 1, 1, 0, 0, 1, 1};
            nn::CausalSelfAttention attn_d(d_model, heads, 1024, seq,
                                           nn::PosEncodingType::Learned);
            attn_d.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
            (void)attn_d.init(engine);   // M6 段 C：层由 init 绑定引擎
            attn_d.set_doc_ids(doc_ids);
            run_csa(attn_d);
        }
        {
            const std::size_t doc_ids[8] = {0, 0, 1, 1, 0, 0, 1, 1};
            nn::CausalSelfAttention attn_ad(d_model, heads, 1024, seq,
                                            nn::PosEncodingType::ALiBi);
            attn_ad.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
            (void)attn_ad.init(engine);   // M6 段 C：层由 init 绑定引擎
            attn_ad.set_doc_ids(doc_ids);
            run_csa(attn_ad);
        }
        // GQA（docs/internal/22 §3.2 的 P2）：K/V_t 走 **HeadGroup 视图**
        //   （头分组比是运行期 vp 参数、**不进 key** → 登记一次覆盖任意 n_rep，
        //   但"视图种类"本身是结构 → 必须是独立一条，否则 GPU 闭合世界对
        //   GQA 的 fold 直接硬报错）。heads=2 / n_head_kv=1 → ratio 2。
        {
            nn::CausalSelfAttention attn_gqa(d_model, heads, 1024, seq,
                                             nn::PosEncodingType::RoPE,
                                             /*subln=*/false, nn::NormType::LayerNorm,
                                             /*n_head_kv=*/1);
            attn_gqa.set_precision_profile(scan_prof);
            (void)attn_gqa.init(engine);
            run_csa(attn_gqa);
        }
    }

    // ── MultiHeadAttention（MHA=Plain 双向无掩码）───────────────
    //   forward fold 的 Plain 变体 + **backward recompute 的裸 S 表达式**
    //   （无掩码 dsl::compute(matmul)，MHA dry-run 是该结构唯一的
    //   闭合世界注册来源）；R/X 与 grad 累加与
    //   CSA 同构同 key。
    {
        const std::size_t d_model = 16, heads = 2, seq = 4, batch = 2;
        nn::MultiHeadAttention attn(d_model, heads, /*seq_len=*/seq);
        attn.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        (void)attn.init(engine);
        nn::Tensor x = scan_tensor(d_model, batch * seq);
        auto fr = attn.forward(x);
        if (!fr)
        {
            std::fprintf(stderr, "[scan] MHA forward FAILED: %s\n",
                         fr.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
        nn::Tensor grad = scan_tensor(d_model, batch * seq);
        auto br = attn.backward(grad);
        if (!br)
        {
            std::fprintf(stderr, "[scan] MHA backward FAILED: %s\n",
                         br.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
    }

    // ── CrossAttention（issue #33：矩形 cross 注意力）──────────────
    //   表达式结构与 MHA 同 key（Plain fold / 裸 S / R / X 均同构；seq_q/
    //   seq_kv 是形状参数不进 key）——本 dry-run 的职责是**显式登记**
    //   seq_q ≠ seq_kv 的矩形路径（闭合世界按"注册过才存在"口径），并在
    //   构建期验证 forward_pair/backward_pair 在矩形形状下可跑通。
    {
        const std::size_t d_model = 16, heads = 2, seq_q = 4, seq_kv = 6,
                          batch = 2;
        nn::CrossAttention attn(d_model, heads, /*seq_len=*/seq_q,
                                /*seq_kv_len=*/seq_kv);
        attn.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        (void)attn.init(engine);
        nn::Tensor q = scan_tensor(d_model, batch * seq_q);
        nn::Tensor mem = scan_tensor(d_model, batch * seq_kv);
        auto fr = attn.forward_pair(q, mem);
        if (!fr)
        {
            std::fprintf(stderr, "[scan] CrossAttention forward FAILED: %s\n",
                         fr.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
        nn::Tensor grad = scan_tensor(d_model, batch * seq_q);
        auto br = attn.backward_pair(grad);
        if (!br)
        {
            std::fprintf(stderr, "[scan] CrossAttention backward FAILED: %s\n",
                         br.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
    }

    // ── CrossEntropyLoss 稀疏 forward（IR 组合 denom/loss_vec/grad）──
    // 结构不依赖形状；含 mask 与 label 越界修正的统一 mask 构造。
    {
        const std::size_t C = 8, B = 5;
        nn::CrossEntropyLoss ce;
        ce.set_precision_profile(scan_prof);   // 扫描期精度（f16 pass 收集变体）
        nn::Tensor logits = scan_tensor(C, B);
        std::vector<std::size_t> labels(B, 1);
        std::vector<nn::Scalar> mask(B, 1.0f);
        (void)ce.forward_sparse(engine, logits, labels, mask, C);
        // 无 mask（全 1 修正）与带越界 label 的变体
        std::vector<std::size_t> labels2(B, 0);
        labels2[0] = 99;  // 越界 → mask 修正为 0
        (void)ce.forward_sparse(engine, logits, labels2, {}, C);
    }

    // ── Softmax forward/backward（独立层；各模型主干走 fold 不经过它）──
    // 输入槽去重（SpecBuilder::add_input）后 leaf(S)/row_reduce_max(S) 各由
    // 4 槽并为 2 槽 → 结构 key 变化；锚点符号实例（占位张量）**不参与去重**
    // 只登记旧结构，故此类"锚点覆盖但 dry-run 未覆盖"的表达式必须在此以
    // 真实张量 dry-run 登记（否则运行期去重后 key 未命中 → 闭合世界硬报错）。
    {
        const std::size_t R = 6, C = 9;   // 与 fused_gpu_test run_softmax 同形
        nn::Softmax sm;
        sm.set_precision_profile(scan_prof);
        (void)sm.init(engine);
        nn::Tensor x = scan_tensor(R, C);
        (void)sm.forward(x);
        nn::Tensor g = scan_tensor(R, C);
        (void)sm.backward(g);
    }

    // ── 归约表达式内联常量（push-constant 头长度回归）────────────────────
    // `col_reduce_sum(select(cond, 1, 0))` 是**带常量池的归约**结构：GPU 侧
    // push-constant 固定头长度必须按形态算（归约且无 matmul = 4 个 uint）。
    // ⚠ 固定头长度必须与生成器 PC 声明**逐形态一致**：错一个 uint 即常量池
    // 整体后移 → GPU 静默错值而 CPU 正常。fused_gpu_test 的 run_reduce_consts
    // 做 CPU/GPU 对比覆盖。
    {
        const std::size_t kk = 4, cols = 3;
        nn::Tensor x = scan_tensor(kk, cols);
        nn::Tensor mx = scan_tensor(1, cols);
        (void)nn::dsl::compute_reduce(engine,
            nn::dsl::col_reduce_sum(nn::dsl::select(
                nn::dsl::leaf(x) == nn::dsl::col_broadcast(mx),
                nn::Scalar{1}, nn::Scalar{0})),
            kk, cols);
    }

    // ── BitLinear 的**两条量化缓冲路径**（F3 / P2）────────────────────────
    // 模型 pass 只产出"该引擎默认走的那条"（CPU 引擎 supports_ternary_gemm()
    // = true → 打包路径），于是稠密路径的 `select → wq_` 与 matmul 尾链**不再
    // 被任何用例登记**；而未接打包的后端（GPU 未接时）运行期走稠密路径 →
    // 闭合世界硬报错（t1_58_test_gpu 就是这么暴露的）。两条路径的结构都必须
    // 登记，故这里各 dry-run 一份（结构不依赖精度）。
    for (const bool packed : {true, false})
    {
        nn::BitLinear bl(4, 3);
        nn::PrecisionProfile t1 = scan_prof;
        t1.param = nn::Precision::T1_58;
        bl.set_precision_profile(t1);
        (void)bl.init(engine);
        bl.set_ternary_pack_enabled(packed);
        nn::Tensor x = scan_tensor(4, 2);
        auto fr = bl.forward(x);
        if (!fr)
        {
            std::fprintf(stderr, "[scan] BitLinear(%s) forward FAILED: %s\n",
                         packed ? "packed" : "dense", fr.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
        nn::Tensor g = scan_tensor(3, 2);
        auto br = bl.backward(g);
        if (!br)
        {
            std::fprintf(stderr, "[scan] BitLinear(%s) backward FAILED: %s\n",
                         packed ? "packed" : "dense", br.error().message.c_str());
            std::fflush(stderr);
            std::abort();
        }
    }

    // ── RL：experience transforms / categorical policy / objectives ────────
    // RL 头不经过 Layer 模型 pass；这里显式触达每个 public evaluate 路径，
    // 让 dsl::compute* 的 FusedAnchor 在构建期登记结构。scan 模式的算子是
    // 占位返回，具体数值分支不重要；所有表达式调用点都会被实例化。
    {
        constexpr std::size_t R = 3, C = 2;
        nn::Tensor logits = scan_tensor(R, C);
        nn::Tensor action = scan_tensor(1, C);
        nn::Tensor one = scan_tensor(1, C);
        nn::Tensor value = scan_tensor(1, C);
        nn::Tensor reward = scan_tensor(R, C);
        nn::Tensor done = scan_tensor(R, C);
        nn::Tensor mask_rc = scan_tensor(R, C);
        nn::Tensor old_log_prob = scan_tensor(1, C);

        nn::CategoricalDistribution dist(scan_prof.stable);
        (void)dist.log_prob(engine, logits, action);
        (void)dist.entropy(engine, logits);
        (void)dist.log_prob_with_gradient(engine, logits, action, one);
        (void)dist.entropy_with_gradient(engine, logits, one);

        // PPO/value clip 的 select 结构：目标方法在 dry-run 中可能因全零
        // mask 提前返回；这里直接触达同一表达式类型，避免闭合世界漏项。
        nn::Tensor ratio = scan_tensor(1, C);
        nn::Tensor old_value_scan = scan_tensor(1, C);
        nn::Tensor value_scan = scan_tensor(1, C);
        nn::Tensor returns_scan = scan_tensor(1, C);
        const nn::Scalar lower = 0.8f;
        const nn::Scalar upper = 1.2f;
        auto clipped_ratio_expr = nn::dsl::select(
            nn::dsl::leaf(ratio) < nn::dsl::rparam(lower),
            nn::dsl::rparam(lower),
            nn::dsl::select(
                nn::dsl::leaf(ratio) > nn::dsl::rparam(upper),
                nn::dsl::rparam(upper), nn::dsl::leaf(ratio)));
        nn::Tensor clipped_ratio = NN_UNWRAP(nn::dsl::compute(
            engine, clipped_ratio_expr, 1, C, scan_prof.stable));
        nn::Tensor clip_low = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(old_value_scan) - nn::dsl::rparam(0.2f),
            1, C, scan_prof.stable));
        nn::Tensor clip_high = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(old_value_scan) + nn::dsl::rparam(0.2f),
            1, C, scan_prof.stable));
        auto clipped_value_expr = nn::dsl::select(
            nn::dsl::leaf(value_scan) < nn::dsl::leaf(clip_low),
            nn::dsl::leaf(clip_low),
            nn::dsl::select(
                nn::dsl::leaf(value_scan) > nn::dsl::leaf(clip_high),
                nn::dsl::leaf(clip_high), nn::dsl::leaf(value_scan)));
        nn::Tensor clipped_value = NN_UNWRAP(nn::dsl::compute(
            engine, clipped_value_expr, 1, C, scan_prof.stable));
        nn::Tensor clipped_diff = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(clipped_value) - nn::dsl::leaf(returns_scan),
            1, C, scan_prof.stable));
        nn::Tensor value_diff = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(value_scan) - nn::dsl::leaf(returns_scan),
            1, C, scan_prof.stable));
        nn::Tensor value_diff_sq = NN_UNWRAP(nn::dsl::compute(
            engine, nn::dsl::leaf(value_diff) * nn::dsl::leaf(value_diff),
            1, C, scan_prof.stable));
        nn::Tensor value_clipped_sq = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(clipped_diff) * nn::dsl::leaf(clipped_diff),
            1, C, scan_prof.stable));
        auto value_max_expr = nn::dsl::max(
            nn::dsl::leaf(value_diff_sq),
            nn::dsl::leaf(value_clipped_sq));
        (void)nn::dsl::compute(engine, value_max_expr, 1, C,
                               scan_prof.stable);

        nn::Tensor ppo_objective = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(ratio) * nn::dsl::leaf(one),
            1, C, scan_prof.stable));
        nn::Tensor ppo_clipped_objective = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::leaf(clipped_ratio) * nn::dsl::leaf(one),
            1, C, scan_prof.stable));
        auto ppo_min_expr = nn::dsl::min(
            nn::dsl::leaf(ppo_objective),
            nn::dsl::leaf(ppo_clipped_objective));
        (void)nn::dsl::compute(engine, ppo_min_expr, 1, C,
                               scan_prof.stable);
        const auto ratio_leaf = nn::dsl::leaf(ratio);
        const auto adv_leaf = nn::dsl::leaf(one);
        const auto positive_factor = nn::dsl::select(ratio_leaf > nn::dsl::rparam(upper),
                                                     nn::Scalar{0}, ratio_leaf * adv_leaf);
        const auto negative_factor = nn::dsl::select(ratio_leaf < nn::dsl::rparam(lower),
                                                     nn::Scalar{0}, ratio_leaf * adv_leaf);
        auto ppo_factor_expr = nn::dsl::select(
            adv_leaf > nn::Scalar{0}, positive_factor, negative_factor);
        nn::Tensor ppo_factor_scan = NN_UNWRAP(nn::dsl::compute(
            engine, ppo_factor_expr, 1, C, scan_prof.stable));
        (void)nn::dsl::compute(engine,
                               nn::dsl::leaf(ppo_factor_scan) *
                                   nn::dsl::leaf(one) *
                                   nn::dsl::rparam(-1.0f),
                               1, C, scan_prof.stable);
        auto ppo_outside_expr = nn::dsl::select(
            ratio_leaf < nn::dsl::rparam(lower), nn::Scalar{1},
            nn::dsl::select(ratio_leaf > nn::dsl::rparam(upper),
                            nn::Scalar{1}, nn::Scalar{0})) *
            nn::dsl::leaf(one);
        (void)nn::dsl::compute(engine, ppo_outside_expr, 1, C,
                               scan_prof.stable);
        const auto value_outside = nn::dsl::select(
            nn::dsl::leaf(value_scan) < nn::dsl::leaf(clip_low),
            nn::Scalar{1},
            nn::dsl::select(
                nn::dsl::leaf(value_scan) > nn::dsl::leaf(clip_high),
                nn::Scalar{1}, nn::Scalar{0}));
        auto value_outside_mask_expr =
            value_outside * nn::dsl::leaf(one);
        (void)nn::dsl::compute(engine, value_outside_mask_expr, 1, C,
                               scan_prof.stable);
        nn::Tensor value_outside_tensor = NN_UNWRAP(nn::dsl::compute(
            engine, value_outside, 1, C, scan_prof.stable));
        nn::Tensor value_clipped_grad = NN_UNWRAP(nn::dsl::compute(
            engine,
            nn::dsl::select(
                nn::dsl::leaf(value_outside_tensor) > nn::Scalar{0},
                nn::Scalar{0},
                nn::dsl::leaf(value_diff) * nn::dsl::rparam(2.0f)),
            1, C, scan_prof.stable));
        nn::Tensor value_unclipped_grad = NN_UNWRAP(nn::dsl::compute(
            engine, nn::dsl::leaf(value_diff) * nn::dsl::rparam(2.0f),
            1, C, scan_prof.stable));
        auto value_choose_clipped =
            nn::dsl::leaf(value_clipped_sq) >
            nn::dsl::leaf(value_diff_sq);
        auto value_grad_expr = nn::dsl::select(
            value_choose_clipped, nn::dsl::leaf(value_clipped_grad),
            nn::dsl::leaf(value_unclipped_grad)) *
            nn::dsl::leaf(one) * nn::dsl::rparam(1.0f);
        (void)nn::dsl::compute(engine, value_grad_expr, 1, C,
                               scan_prof.stable);

        nn::GaeAdvantage gae;
        (void)gae.apply(engine, reward, reward, &done, &mask_rc, &one);
        nn::MonteCarloReturn returns;
        (void)returns.apply(engine, reward, &done, &mask_rc);
        nn::MaskedNormalizer normalizer;
        (void)normalizer.apply(engine, reward, mask_rc);
        const nn::GroupSpan groups[1]{{0, R}};
        nn::GroupAdvantage group;
        (void)group.apply(engine, reward, groups);

        nn::ExperienceSchema exp = NN_UNWRAP(nn::ExperienceSchema::create(
            {"action", "action_mask", "advantage", "return", "old_value",
             "old_log_prob"}));
        nn::TensorRecord record = NN_UNWRAP(nn::TensorRecord::create(
            exp.schema(),
            {nn::NamedTensor{.name = "action", .value = action},
             nn::NamedTensor{.name = "action_mask", .value = one},
             nn::NamedTensor{.name = "value_mask", .value = one},
             nn::NamedTensor{.name = "advantage", .value = one},
             nn::NamedTensor{.name = "return", .value = one},
             nn::NamedTensor{.name = "old_value", .value = one},
             nn::NamedTensor{.name = "old_log_prob",
                             .value = old_log_prob}}));
        nn::PolicyBatchView view = NN_UNWRAP(
            nn::PolicyBatchView::create(record));

        nn::ReinforceObjective reinforce;
        (void)reinforce.evaluate_context(engine, logits, view);
        nn::ValueRegressionObjective value_reg;
        (void)value_reg.evaluate_context(engine, value, view);
        nn::EntropyBonusObjective entropy;
        (void)entropy.evaluate_context(engine, logits, view);
        nn::SampledKlObjective kl;
        (void)kl.evaluate_context(engine, logits, view);
        nn::ClippedPolicyObjective ppo_policy;
        (void)ppo_policy.evaluate_context(engine, logits, view);
        nn::ClippedValueObjective ppo_value;
        (void)ppo_value.evaluate_context(engine, value, view);

        nn::TensorSchema output = NN_UNWRAP(nn::TensorSchema::create(
            "rl_output",
            {nn::tensor_field("policy",
                              nn::TensorShapeSpec::exact(R, C)),
             nn::tensor_field("value",
                              nn::TensorShapeSpec::exact(1, C))}));
        nn::PpoObjective ppo;
        (void)ppo.evaluate(engine, output, logits, value, view);
    }

    };   // dry_run 结束
#if !defined(NN_SCAN_NO_DRYRUN)
    // 跑两遍：f32 定**结构**，f16 定**精度签名**。
    //   · 结构（与精度无关）：f16 遍对结构贡献为 0（实测），保留 f32 一遍即可。
    //   · 签名（构建期特化变体 `key#xxxx` 的输入）：只有真跑 f16 profile 才
    //     观察得到。签名不再是"身份"（运行期 `key#x` 仍覆盖任意签名），但
    //     编译期特化能去掉热循环里的 `prec` 三元——实测把 CE 融合注意力从
    //     146 ms 降到 120 ms（f32 基线 117 ms，850M/f16）。故 f16 遍重新启用，
    //     只为**收集签名**（收集侧零成本：dry-run 不真算）。
    dry_run(*raw_engine, nn::profile_f32());   // 结构（与精度无关）
    dry_run(*raw_engine, nn::profile_f16());   // 精度签名（供 key#xxxx 特化）
#endif
    // 分项台账快照（末尾合并时打印；见"结构来源"行）
    const std::size_t n_after_dry = nn::fused::global_registry().specs.size();
    // f16 安全由基类边界 cast 入口保证（原 PrecisionEngine 适配层已下沉删除）

    // ── 模型级 pass：用 shipped 工厂建模型跑 fwd/bwd（补"整模型路径"的结构）──
    // per-layer dry-run 块只覆盖"层被单独造出来"的路径；有些调用点只在**完整
    // 模型**里才会执行到（GPT/RAPT 的 LM head、PatchEmbedding、模型级位置编码
    // 等）。它们是"运行期配置决定结构"的主要来源——实测 84 条结构里 25 条只由
    // 本 pass 产出，故**失败即构建失败**：静默降级会产出不完整的注册表，只在
    // GPU 运行期以闭合世界硬报错暴露（离根因很远）。与 dry-run 的 fail-fast
    // （见 run_csa 的 abort）一致。
    const auto model_pass = [&](nn::ComputeEngine& engine,
                                const nn::PrecisionProfile& prof) -> bool
    {
        scan_prof = prof;
        std::size_t n_fail = 0;
        const std::size_t B = 3;
        // 扫描期占位张量（精度 = 当前 pass 的 compute 精度，与真实运行同源）
        const auto scan_tensor = [&](std::size_t rows, std::size_t cols)
        { return engine.create_tensor(rows, cols, scan_prof.compute); };
        // 跑一个模型：forward → 取其输出形状造梯度 → backward
        const auto run = [&](nn::Model& model, const nn::Tensor& input, const char* name)
        {
            auto out = model.forward(input);
            if (!out)
            {
                std::fprintf(stderr, "[scan][FAIL] %s forward 失败：%s\n", name,
                             out.error().message.c_str());
                ++n_fail;
                return;
            }
            nn::Tensor grad = scan_tensor(out->rows(), out->cols());
            auto br = model.backward(grad);
            if (!br)
            {
                std::fprintf(stderr, "[scan][FAIL] %s backward 失败：%s\n", name,
                             br.error().message.c_str());
                ++n_fail;
                return;
            }

            // ── 推理态再跑一轮 forward + backward：训练/推理双态层
            //（BatchNorm）的推理路径是**不同的表达式结构**——无归约链、
            // grad_x 无均值修正项——只跑训练态会在 GPU 推理时闭合世界
            // 硬报错。对无双态语义的层 set_training 是 no-op，多跑一轮
            // 只是重复登记已有结构（key 去重）。
            model.set_training(false);
            auto eout = model.forward(input);
            if (!eout)
            {
                std::fprintf(stderr, "[scan][FAIL] %s 推理态 forward 失败：%s\n",
                             name, eout.error().message.c_str());
                ++n_fail;
            }
            else
            {
                nn::Tensor egrad = scan_tensor(eout->rows(), eout->cols());
                auto ebr = model.backward(egrad);
                if (!ebr)
                {
                    std::fprintf(stderr, "[scan][FAIL] %s 推理态 backward 失败：%s\n",
                                 name, ebr.error().message.c_str());
                    ++n_fail;
                }
            }
            model.set_training(true);   // 恢复训练态（后续模型默认口径）
        };
        const auto try_build = [&](const char* name, auto&& build, const nn::Tensor& input)
        {
            auto m = build();
            if (!m)
            {
                std::fprintf(stderr, "[scan][FAIL] %s 构建失败：%s\n", name,
                             m.error().message.c_str());
                ++n_fail;
                return;
            }
            run(*m, input, name);
        };

        // 小配置（覆盖各"结构分支"：位置编码 / 激活 / 归一化 / 池化 / causal）
        constexpr std::size_t V = 64, S = 8, D = 16, H = 2, F = 32, L = 2;
        // id 输入：概念上是整数索引 → 恒 f32（text_train 同口径）；
        //   gather_rows 按值取行，必须填**合法** id（否则越界）
        const auto id_input = [&](std::size_t seq, std::size_t vocab)
        {
            nn::Tensor t = engine.create_tensor(seq, B, nn::Precision::F32);
            std::vector<nn::Scalar> v(seq * B);
            for (std::size_t i = 0; i < v.size(); ++i)
                v[i] = static_cast<nn::Scalar>(i % vocab);
            (void)engine.write(t, std::span<nn::Scalar>(v));
            return t;
        };
        const nn::Tensor ids = id_input(S, V);

        // ── 1) MNIST MLP：归一化维（LayerNorm/RMSNorm/BatchNorm 是三条不同
        //        融合表达式：归约链形状与原地更新都不同）──
        for (const auto nrm : {nn::NormType::LayerNorm, nn::NormType::RMSNorm,
                               nn::NormType::BatchNorm})
        {
            const char* nm = (nrm == nn::NormType::LayerNorm) ? "mnist_mlp_ln"
                           : (nrm == nn::NormType::RMSNorm)   ? "mnist_mlp_rms"
                                                              : "mnist_mlp_bn";
            try_build(nm,
                      [&] { return nn::build_mnist_mlp_model(
                                engine, nn::MNIST_LAYER_DIMS, nrm, prof); },
                      scan_tensor(nn::MNIST_LAYER_DIMS.front(), B));
        }
        // ── 1b) 三值（T1_58）MLP：BitLinear 的量化 / 去量化 / STE 三段结构 ──
        //   BitLinear 的表达式（τ 的 absmean 归约、wq 的阈值 select、
        //   matmul×row_broadcast(τ) 尾链、dW∘τ 的原地累加）不在任何其它层里
        //   出现 → **只由这一条用例产出**；漏掉就是 GPU 闭合世界硬报错
        //   （AGENTS §7：dry-run/模型 pass 是结构的主要来源）。
        //   结构不依赖精度（prof 只决定占位张量精度）→ 与其它用例同处一跑。
        //   见 docs/internal/21-quantized-weights.md §7 第 4 项。
        {
            nn::PrecisionProfile t1 = prof;
            t1.param = nn::Precision::T1_58;
            try_build("mnist_mlp_t1_58",
                      [&] { return nn::build_mnist_mlp_model(
                                engine, nn::MNIST_LAYER_DIMS,
                                nn::NormType::LayerNorm, t1); },
                      scan_tensor(nn::MNIST_LAYER_DIMS.front(), B));
        }
        // ── 2) MNIST Transformer（ViT 风格；输入 img²）──
        try_build("mnist_transformer",
                  [&] { return nn::build_mnist_transformer_model(
                            engine, nn::MNIST_IMG_SIZE, nn::MNIST_PATCH_SIZE,
                            nn::MNIST_TF_D_MODEL, nn::MNIST_TF_NUM_HEADS,
                            nn::MNIST_TF_D_FF, nn::MNIST_TF_NUM_LAYERS, prof); },
                  scan_tensor(nn::MNIST_IMG_SIZE * nn::MNIST_IMG_SIZE, B));
        // ── 2b) 归一化维（NormType / NormPlace 是运行期配置决定的结构，类型
        //        层面推不出 → 与 GPT 的 pe/act/norm 同理，必须在这里补 dry-run）──
        try_build("mnist_transformer_final_ln",
                  [&] { return nn::build_mnist_transformer_model(
                            engine, nn::MNIST_IMG_SIZE, nn::MNIST_PATCH_SIZE,
                            nn::MNIST_TF_D_MODEL, nn::MNIST_TF_NUM_HEADS,
                            nn::MNIST_TF_D_FF, nn::MNIST_TF_NUM_LAYERS, prof,
                            nn::NormType::LayerNorm, nn::NormPlace::Final); },
                  scan_tensor(nn::MNIST_IMG_SIZE * nn::MNIST_IMG_SIZE, B));
        try_build("mnist_transformer_final_rms",
                  [&] { return nn::build_mnist_transformer_model(
                            engine, nn::MNIST_IMG_SIZE, nn::MNIST_PATCH_SIZE,
                            nn::MNIST_TF_D_MODEL, nn::MNIST_TF_NUM_HEADS,
                            nn::MNIST_TF_D_FF, nn::MNIST_TF_NUM_LAYERS, prof,
                            nn::NormType::RMSNorm, nn::NormPlace::Final); },
                  scan_tensor(nn::MNIST_IMG_SIZE * nn::MNIST_IMG_SIZE, B));
        try_build("mnist_transformer_prenorm_bn",
                  [&] { return nn::build_mnist_transformer_model(
                            engine, nn::MNIST_IMG_SIZE, nn::MNIST_PATCH_SIZE,
                            nn::MNIST_TF_D_MODEL, nn::MNIST_TF_NUM_HEADS,
                            nn::MNIST_TF_D_FF, nn::MNIST_TF_NUM_LAYERS, prof,
                            nn::NormType::BatchNorm, nn::NormPlace::None); },
                  scan_tensor(nn::MNIST_IMG_SIZE * nn::MNIST_IMG_SIZE, B));
        // ── 3) CNN：池化窗口维（R 是运行期视图参数，pool=3 → R=9）──
        for (const std::size_t pool : {std::size_t{2}, std::size_t{3}})
        {
            nn::CnnConfig cfg;
            cfg.convs = nn::MNIST_CNN_CONVS;
            cfg.fc_dims = nn::MNIST_CNN_FC;
            cfg.pool = pool;
            // 注：CnnConfig 无精制度字段（CNN 走默认 f32 profile）
            try_build(pool == 2 ? "cnn_pool2" : "cnn_pool3",
                      [&] { return nn::build_cnn_model(engine, cfg); },
                      scan_tensor(1 * 28 * 28, B));
        }
        // ── 3b) CNN 归一化维（NormPlace × NormType 同上：运行期配置决定）──
        {
            struct CnnNormCase { const char* name; nn::NormType nt; nn::NormPlace np; };
            for (const CnnNormCase c : {
                     CnnNormCase{"cnn_norm_both_ln", nn::NormType::LayerNorm,
                                 nn::NormPlace::Both},
                     CnnNormCase{"cnn_norm_conv_rms", nn::NormType::RMSNorm,
                                 nn::NormPlace::Conv},
                     CnnNormCase{"cnn_norm_head_bn",  nn::NormType::BatchNorm,
                                 nn::NormPlace::Head},
                 })
            {
                nn::CnnConfig cfg;
                cfg.convs = nn::MNIST_CNN_CONVS;
                cfg.fc_dims = nn::MNIST_CNN_FC;
                cfg.pool = 2;
                cfg.norm_type = c.nt;
                cfg.norm_place = c.np;
                try_build(c.name,
                          [&] { return nn::build_cnn_model(engine, cfg); },
                          scan_tensor(1 * 28 * 28, B));
            }
        }
        // ── 4) GPT：位置编码 × 激活 × 归一化（每一维都换一组融合表达式：
        //        ALiBi 走分数侧偏置、Sinusoidal 走嵌入侧、RoPE 走 Q/K 侧、
        //        SwiGLU ≠ GeLU、RMSNorm ≠ LayerNorm）──
        struct GptCase
        {
            const char*          name;
            nn::PosEncodingType  pe;
            nn::ActivationType   act;
            nn::NormType         norm;
            std::size_t          n_head_kv = 0;   // 0 = MHA；>0 = GQA（头映射结构）
            bool                 subln     = false;
            bool                 tie       = false;
            bool                 ternary   = false;  // P1.5：param=T1_58 → BitLinear
        };
        for (const GptCase& c : {
                 GptCase{"gpt_learned_gelu_ln", nn::PosEncodingType::Learned,
                         nn::ActivationType::GeLU, nn::NormType::LayerNorm},
                 GptCase{"gpt_alibi_gelu_ln", nn::PosEncodingType::ALiBi,
                         nn::ActivationType::GeLU, nn::NormType::LayerNorm},
                 GptCase{"gpt_sinusoidal_gelu_ln", nn::PosEncodingType::Sinusoidal,
                         nn::ActivationType::GeLU, nn::NormType::LayerNorm},
                 GptCase{"gpt_rope_gelu_ln", nn::PosEncodingType::RoPE,
                         nn::ActivationType::GeLU, nn::NormType::LayerNorm},
                 GptCase{"gpt_learned_swiglu_rms", nn::PosEncodingType::Learned,
                         nn::ActivationType::SwiGLU, nn::NormType::RMSNorm},
                 // BitNet 2B4T 形态：门控平方 ReLU（relu2 ≠ swiglu ≠ gelu）
                 GptCase{"gpt_rope_relu2_rms", nn::PosEncodingType::RoPE,
                         nn::ActivationType::ReLU2, nn::NormType::RMSNorm},
                 // 2B4T 完整组合：GQA（HeadGroup 视图）+ SubLN + tied head。
                 // GQA 的 fold / masked_scores / grad_A / grad_Q 是**新结构**
                 //   （K/V 走 HeadGroup 视图）→ 必须在此登记（闭合世界）。
                 GptCase{"gpt_rope_relu2_rms_gqa_subln_tied", nn::PosEncodingType::RoPE,
                         nn::ActivationType::ReLU2, nn::NormType::RMSNorm,
                         // 模型 pass 的小配置 H=2 → n_head_kv=1 才得到 ratio>1
                         // （比率是**视图参数**、不进 key → 任意 n_rep 共用一份 shader）
                         /*n_head_kv=*/1, /*subln=*/true, /*tie=*/true},
                 // ── 三值 GPT（P1.5-W2）：权重侧接线后，注意力 4 个投影、
                 //    fc1/fc2、LM head 全部是 BitLinear。BitLinear 的三段结构
                 //    （τ 的 absmean 归约 / wq 的阈值 select / 去量化点积与 STE
                 //    累加）已由 1b) 的 MLP 用例登记；这里再按 GPT 组合登记一遍，
                 //    把"三值 + GQA/SubLN/tied + ReLU²/RoPE"这一整条链路钉进
                 //    闭合世界（漏登记 = GPU 硬报错，见 AGENTS §7）。
                 GptCase{"gpt_learned_gelu_ln_t1_58", nn::PosEncodingType::Learned,
                         nn::ActivationType::GeLU, nn::NormType::LayerNorm,
                         /*n_head_kv=*/0, /*subln=*/false, /*tie=*/false,
                         /*ternary=*/true},
                 GptCase{"gpt_rope_relu2_rms_gqa_subln_tied_t1_58",
                         nn::PosEncodingType::RoPE, nn::ActivationType::ReLU2,
                         nn::NormType::RMSNorm,
                         /*n_head_kv=*/1, /*subln=*/true, /*tie=*/true,
                         /*ternary=*/true}})
        {
            nn::GptConfig cfg;
            cfg.vocab_size = V; cfg.d_model = D; cfg.seq_len = S;
            cfg.num_heads = H; cfg.d_ff = F; cfg.num_layers = L;
            cfg.pos_enc = c.pe; cfg.activation = c.act; cfg.norm_type = c.norm;
            cfg.n_head_kv = c.n_head_kv;
            cfg.subln = c.subln;
            cfg.tie_embeddings = c.tie;
            cfg.precision = prof;
            if (c.ternary) cfg.precision.param = nn::Precision::T1_58;
            try_build(c.name, [&] { return nn::build_gpt_model(engine, cfg); }, ids);
        }
        // ── 5) RAPT：causal 维（RLA 的因果/双向是两组不同表达式）──
        for (const bool causal : {true, false})
        {
            nn::RAPTConfig cfg;
            cfg.vocab_size = V; cfg.d_model = D; cfg.seq_len = S;
            cfg.num_heads = H; cfg.d_ff = F; cfg.num_layers = L;
            cfg.pos_enc = nn::PosEncodingType::RoPE;   // RAPT v1 仅支持 RoPE
            cfg.causal = causal;
            cfg.precision = prof;
            try_build(causal ? "rapt_causal" : "rapt_bidir",
                      [&] { return nn::build_rapt_model(engine, cfg); }, ids);
        }
        return n_fail == 0;
    };
    // 模型 pass：**结构**来源（运行期配置相关的路径），与精度无关 → 只跑 f32。
    // 与 per-layer dry-run 同受 `-DNN_SCAN_NO_DRYRUN` 门控（该宏的文档语义就是
    // "关掉 dry-run/模型 pass"，此前只关了前者）。
#if !defined(NN_SCAN_NO_DRYRUN)
    if (!model_pass(*raw_engine, nn::profile_f32()))
    {
        std::fprintf(stderr,
                     "[scan][FAIL] 模型 pass 失败 → 注册表不完整，中止构建"
                     "（闭合世界：缺结构只在 GPU 运行期报错）\n");
        return 1;
    }
    // f16 模型 pass：同为**签名**来源（整模型路径的精度组合与 per-layer
    // dry-run 不同——例如 LM head 的"f16 输入 + f32 输出"只在这里出现）。
    // 结构对结构是第一遍的重复（注册表按 key 去重），只有签名是新信息。
    if (!model_pass(*raw_engine, nn::profile_f16()))
    {
        std::fprintf(stderr,
                     "[scan][FAIL] f16 模型 pass 失败 → 特化签名不完整，中止构建\n");
        return 1;
    }
#endif
    const std::size_t n_after_model = nn::fused::global_registry().specs.size();
    // ── 注：fold 结构（通用 v1 样例 + 注意力 5 组合 × {MHA,GQA}）不在本文件
    //    登记 ── 它们由**定义方**在静态初始化期经 `FoldAnchor` 自登记
    //    （`expr_fold.hpp` 末尾；`compute_layer_attention.hpp` 的
    //    `attn_fold_anchor`），随下方锚点注册表一并合并。此前这里有一份
    //    手写登记块，漏项即 GPU 闭合世界硬报错（P4-4 ①，见
    //    docs/history/history.md）。新增掩码/偏置组合只改定义方，不改这里。

    const std::size_t n_before_anchor = nn::fused::global_registry().specs.size();
    auto& reg = nn::fused::global_registry();
    // ── 合并自登记锚点的结构（"结构 = 表达式类型"，见 expr_dsl.hpp）─────────
    // 锚点在静态初始化期按**类型**登记，覆盖"编译进来但 dry-run 未执行到"的
    // 调用点（配置分支、未被 dry-run 造过的层路径）。两者合并后写盘。
    {
        auto& anchor = nn::dsl::anchor_registry();
        // NN_SCAN_DUMP_SOURCES=1：逐条打印"只有执行 Layer/模型才拿得到"的结构。
        // 这是"能否进一步删掉 dry-run"的唯一依据——锚点按表达式类型登记，但
        // **符号实例（`Expr{}`）与真实实例可能折叠出不同结构**（运行期配置
        // 维度：视图种类、输入个数、掩码组合…）；这些结构锚点看不到。
        const bool dump_src = nn::tool::tool_env_flag("NN_SCAN_DUMP_SOURCES");
        // 两侧的 key 集合直接取注册表已有的 keys（add() 已算过 key），不再重建。
        const std::unordered_set<std::string>& dry_keys = reg.keys;
        const std::unordered_set<std::string>& anchor_keys = anchor.keys;
        const auto summarize = [](const nn::ExprSpec& s)
        {
            const int raxis = nn::expr_spec_reduce_axis(s);
            std::string v;
            for (const auto& x : s.views)
            {
                if (!v.empty()) v += ",";
                v += std::to_string(static_cast<int>(x.kind));
            }
            std::string ins;
            for (const auto& i : s.instrs)
            {
                if (!ins.empty()) ins += ";";
                ins += std::to_string(static_cast<int>(i.op)) + "("
                     + std::to_string(static_cast<int>(i.a.kind)) + "/"
                     + std::to_string(static_cast<int>(i.a.idx)) + ","
                     + std::to_string(static_cast<int>(i.b.kind)) + "/"
                     + std::to_string(static_cast<int>(i.b.idx)) + ","
                     + std::to_string(static_cast<int>(i.c.kind)) + "/"
                     + std::to_string(static_cast<int>(i.c.idx)) + ")";
            }
            return "raxis=" + std::to_string(raxis)
                 + " mm=" + (s.matmul ? "1" : "0")
                 + " fold=" + (s.fold ? "1" : "0")
                 + " nreg=" + std::to_string(s.num_regs)
                 + " nconst=" + std::to_string(s.consts.size())
                 + " nrparam=" + std::to_string(s.rparams.size())
                 + " nview=" + std::to_string(s.views.size())
                 + " viewkinds=[" + v + "]"
                 + " instrs=[" + ins + "]";
        };
        std::size_t n_dry_only = 0;
        for (const auto& s : reg.specs)
        {
            const std::string k = nn::expr_spec_key(s);
            if (anchor_keys.count(k))
                continue;
            ++n_dry_only;
            if (dump_src)
                std::printf("[scan][dry-only] %s  %s\n", k.c_str(), summarize(s).c_str());
        }
        for (const auto& s : anchor.specs)
        {
            const std::string k = nn::expr_spec_key(s);
            if (dry_keys.count(k))
                continue;
            if (dump_src)
                std::printf("[scan][anchor-only] %s  %s\n", k.c_str(), summarize(s).c_str());
        }
        // 交集（锚点与 dry-run 都产出的结构）也打印：否则"某结构由哪个生成器
        // 负责、运行时是否按 V1 分派"这类问题无法从 dump 里回答——V1 覆盖统计
        // 的 20 条里有 18 条落在交集里。
        if (dump_src)
            for (const auto& s : reg.specs)
                if (anchor_keys.count(nn::expr_spec_key(s)))
                    std::printf("[scan][both] %s  %s\n",
                                nn::expr_spec_key(s).c_str(), summarize(s).c_str());
        for (const auto& s : anchor.specs)
            reg.add(s);
        // 分段口径（**互不重叠**）：dry-run 侧 = ①per-layer dry-run + matmul/bmm
        // 显式登记（n_after_dry）+ ②模型 pass（+）。fold 结构（通用 v1 样例与
        // 注意力组合）与其余表达式同走自登记锚点（由定义方挂锚，见上方注释）。
        std::printf("[scan] 结构来源：dry-run 侧 %zu（per-layer+显式 %zu、模型 pass +%zu）"
                    " + 锚点 %zu（新增 %zu）= 合计 %zu"
                    "（锚点独有 %zu、dry-run 独有 %zu）\n",
                    n_before_anchor,
                    n_after_dry,
                    n_after_model - n_after_dry,
                    anchor.specs.size(),
                    reg.specs.size() - n_before_anchor,
                    reg.specs.size(),
                    reg.specs.size() - n_before_anchor, n_dry_only);
        // 锚点独有 = 手写 dry-run 覆盖不到的调用点（该数突然变大 → 有新层路径
        // 没纳入 dry-run）。
        if (anchor.specs.empty() || n_before_anchor == reg.specs.size())
            std::fprintf(stderr,
                "[scan][warn] 锚点未贡献任何结构：可能被 -DNN_SCAN_NO_ANCHOR 关闭，"
                "或所有调用点都已被 dry-run 覆盖\n");
    }
    std::printf("[scan] 收集到 %zu 条融合表达式\n", reg.specs.size());
    // ── 签名覆盖：**构建期不再需要任何背书** ──────────────────────────────
    // 生成阶段对每个结构无条件发一份运行期精度分派 shader（键 `key#x`）——
    // 输入/输出双视图 + PC `prec`，一份覆盖任意 (输入精度位图, 输出精度)。
    // 因此：
    //   · 结构 = 唯一需要"收集"的东西（锚点 + dry-run/模型 pass + 显式登记）；
    //   · 签名**不再是构建期集合**，无需 dry-run 的 f16 遍、无需回填清单；
    //   · 注册表里没有"精度变体"这一层（`ExprRegistry::variants` 已删除）。
    // ⚠ 精度路径的落点仍有三档，不是"永不 miss"：`#a`（native16）→ `#x`
    //   （运行期分派）→ 基类边界 cast。`#x` 在生成被跳过（[skip]）、
    //   NN_SCAN_NO_DISPATCH=1、或设备无 16bit 存储时不注册 → 落到边界 cast。
    // ── 生成阶段（原 gen_fused）：同一进程内直接消费注册表 → fused_registry.hpp
    //    构建期因此只有一步；不再经 expr_specs.bin 中间序列化。
    if (!nn::tool::generate_fused_registry(out_dir, glslc, reg))
        return 1;
    return 0;
}
