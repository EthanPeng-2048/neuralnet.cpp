// ───────────────────────────────────────────────────────────────────────────
//  scan_exprs.cpp — AOT 算子融合：构建期表达式收集（dry-run）
//
//  用 CPU 引擎 + 假张量跑一遍相关 Layer 的 forward/backward，使每个
//  dsl::compute / end_expr 在 NN_EXPR_SCAN 记录模式下把折叠出的 ExprSpec
//  **结构**登记进全局注册表（按 expr_spec_key 去重），跑完 dump 成 bin。
//
//  表达式**文本只出现在 Layer**；这里只是"执行 Layer 代码路径"以触达它们，
//  dump 出来的是派生物（结构），不是手写定义。
//
//  用法： scan_exprs <out.bin>
//  之后 gen_fused 读 <out.bin> 合成融合 shader（→ fused_registry.hpp）。
//
//  ⚠ 闭合世界安全网：若某条 Layer 路径未被本工具覆盖，其内联表达式在
//    GPU 运行时将硬报错，提醒把该路径补进扫描。新增融合表达式 / 新 d_k
//    时，把对应的 dry-run 调用加到下面。
// ───────────────────────────────────────────────────────────────────────────

#define NN_EXPR_SCAN

#include <cstdio>
#include <cstddef>

#include "compute_cpu_engine.hpp"
#include "compute_layer.hpp"
#include "compute_loss.hpp"
#include "compute_optimizer.hpp"
#include "expr_registry.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: scan_exprs <out.bin>\n");
        return 2;
    }
    const std::string out_path = argv[1];

    nn::CpuEngine engine;

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
        nn::Tensor q = nn::Tensor::cpu(2 * dk, 8);   // rows 为 dk 的整数倍
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/false);
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/true);
        nn::Tensor q1 = nn::Tensor::cpu(dk, 1);      // 增量推理（单位置）
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/false);
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/true);
    }

    // ── ReLU forward + backward（DSL 融合，CNN/RAPT 模型使用）────────────────
    // forward:  max(x, 0)；backward: select(x>0, grad, 0)。
    // 结构不依赖形状，任取一个 R×C 即可；先 forward 填 input_cache_ 再 backward。
    {
        const std::size_t R = 6, C = 9;
        nn::ReLU relu;
        nn::Tensor input = nn::Tensor::cpu(R, C);
        (void)relu.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(R, C);
        (void)relu.backward(engine, grad);
    }

    // ── SwiGLU backward（grad_gate / grad_up 两条内联表达式）──────────────
    // 结构不依赖 d_ff，任取一个即可；先 forward 填充 cache 再 backward。
    {
        const std::size_t d_ff = 8;
        nn::SwiGLU swiglu(d_ff);
        nn::Tensor input = nn::Tensor::cpu(2 * d_ff, 5);
        (void)swiglu.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(d_ff, 5);
        (void)swiglu.backward(engine, grad);
    }

    // ── GeLU forward + backward（QuickGeLU 单表达式 DSL 融合）─────────────
    // forward:  x / (1 + exp(-βx))；backward: grad_out * s*(1 + βx*(1-s))，
    //           s = sigmoid(βx)。sigmoid 在 backward 用 input_cache_ 重算。
    // 结构不依赖形状，任取一个 R×C 即可；先 forward 填 input_cache_ 再 backward。
    {
        const std::size_t R = 6, C = 9;
        nn::GeLU gelu;
        nn::Tensor input = nn::Tensor::cpu(R, C);
        (void)gelu.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(R, C);
        (void)gelu.backward(engine, grad);
    }

    // ── Softmax forward + backward（M3 行归约融合）────────────────────────
    // forward:  exp(x - row_max) / row_sum(exp(x - row_max))
    // backward: out * (grad - row_dot(out * grad))
    // 结构不依赖形状，任取一个 R×C 即可。
    {
        const std::size_t R = 6, C = 9;
        nn::Softmax softmax;
        nn::Tensor input = nn::Tensor::cpu(R, C);
        (void)softmax.forward(engine, input);   // 填充 output_cache_ + 登记 fwd 结构
        nn::Tensor grad = nn::Tensor::cpu(R, C);
        (void)softmax.backward(engine, grad);   // 登记 bwd 结构
    }

    // ── RMSNorm forward + backward（M3 列/行归约融合）────────────────────
    // forward:  s=col_sum(x²)*invF+eps → rms_inv=rsqrt(s) → normed=x*rms_inv → out=normed*gamma
    // backward: grad_x 列归约表达式 + grad_gamma 行归约表达式
    {
        const std::size_t F = 8, B = 5;
        nn::RMSNorm rms(F);
        (void)rms.init(engine);
        nn::Tensor input = nn::Tensor::cpu(F, B);
        (void)rms.forward(engine, input);       // 填充 normed/rms_inv 缓存 + 登记 fwd 结构
        nn::Tensor grad = nn::Tensor::cpu(F, B);
        (void)rms.backward(engine, grad);       // 登记 bwd 结构
    }

    // ── LayerNorm forward + backward（M3 列/行归约融合）──────────────────
    // forward: mean → diff → var → std_inv → normalized → out=normalized*gamma+beta
    // backward: grad_x 列归约表达式 + grad_gamma/grad_beta 行归约表达式
    {
        const std::size_t F = 8, B = 5;
        nn::LayerNorm ln(F);
        (void)ln.init(engine);
        nn::Tensor input = nn::Tensor::cpu(F, B);
        (void)ln.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(F, B);
        (void)ln.backward(engine, grad);
    }

    // ── FusedChainLayer（IR-C：begin_expr/end_expr 图 IR 链式融合）────────
    // forward 的三个逐元素表达式在 end_expr 时融合成单个 kernel（t/u 内联为
    // 寄存器）。必须 dry-run 覆盖本路径：GPU 运行时 begin_expr/end_expr 融合
    // 出的复合 spec 才能在 fused_registry 中命中（闭合世界两端一致）。
    // backward 为独立表达式（非录制段），单独登记。
    {
        const std::size_t F = 8, B = 5;
        nn::FusedChainLayer chain(F);
        (void)chain.init(engine);
        nn::Tensor input = nn::Tensor::cpu(F, B);
        (void)chain.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(F, B);
        (void)chain.backward(engine, grad);
    }

    // ── ReLULinearAttention forward + backward（RLA 原语组合版逐元素链）──
    // forward:  Sm=S·mask, S2=Sm·Sm, denom=sqrt(rowsum(S2)+1e-6), out=num/denom
    // backward: dnum=dO/denom, prod=dO·num, ddenom=-dot/(denom²),
    //           ddenom2=ddenom/(2·denom), dSm_den=2·Sm·ddenom2, dSm=dSm_num+dSm_den,
    //           dS=dSm·mask
    // 结构不依赖形状，任取一个小 d_model/seq 即可。
    {
        const std::size_t d_model = 8, heads = 2, seq = 4, batch = 2;
        nn::ReLULinearAttention attn(d_model, heads, seq, /*causal=*/true,
                                     nn::PosEncodingType::RoPE);
        (void)attn.init(engine);
        nn::Tensor x = nn::Tensor::cpu(d_model, batch * seq);
        (void)attn.forward(engine, x);                       // 填 cache + 登记 fwd 结构
        nn::Tensor grad = nn::Tensor::cpu(d_model, batch * seq);
        (void)attn.backward(engine, grad);                   // 登记 bwd 结构
    }

    // ── GPTBlock forward + backward（残差相加 A+B）────────────────────────
    {
        const std::size_t d_model = 16, heads = 2, d_ff = 32, seq = 4, batch = 2;
        nn::GPTBlock block(d_model, heads, d_ff, /*max_len=*/1024, /*seq_len=*/seq,
                           nn::PosEncodingType::Learned, nn::ActivationType::GeLU,
                           nn::NormType::LayerNorm);
        (void)block.init(engine);
        nn::Tensor x = nn::Tensor::cpu(d_model, batch * seq);
        (void)block.forward(engine, x);
        nn::Tensor grad = nn::Tensor::cpu(d_model, batch * seq);
        (void)block.backward(engine, grad);
    }

    // ── TransformerEncoderLayer forward + backward（残差相加 + 位置编码）──
    {
        const std::size_t d_model = 16, heads = 2, d_ff = 32, seq = 4, batch = 2;
        nn::TransformerEncoderLayer enc(d_model, heads, d_ff, seq);
        (void)enc.init(engine);
        nn::Tensor x = nn::Tensor::cpu(d_model, batch * seq);
        (void)enc.forward(engine, x);
        nn::Tensor grad = nn::Tensor::cpu(d_model, batch * seq);
        (void)enc.backward(engine, grad);
    }

    // ── ZiPTBlock forward + backward（残差相加 + 注意力梯度累加）─────────
    {
        const std::size_t d_model = 16, heads = 2, d_ff = 32, win = 4, mem = 2;
        nn::ZiPTBlock zipt(d_model, heads, d_ff, win, mem,
                           nn::NormType::LayerNorm, nn::ActivationType::GeLU);
        (void)zipt.init(engine);
        nn::Tensor x = nn::Tensor::cpu(d_model, win);
        (void)zipt.forward(engine, x);
        nn::Tensor grad = nn::Tensor::cpu(d_model, win);
        (void)zipt.backward(engine, grad);
    }

    // ── MSELoss forward（diff = pred-target；diff_sq = diff*diff）────────
    {
        const std::size_t R = 8, C = 5;
        nn::MSELoss mse;
        nn::Tensor pred = nn::Tensor::cpu(R, C);
        nn::Tensor target = nn::Tensor::cpu(R, C);
        (void)mse.forward(engine, pred, target);
    }

    // ── CrossEntropyLoss 稠密 forward（grad=softmax-target；target*log_sm；
    //    log_col_sum = log(col_sum)）─────────────────────────────────────
    {
        const std::size_t C = 8, B = 5;
        nn::CrossEntropyLoss ce;
        nn::Tensor logits = nn::Tensor::cpu(C, B);
        nn::Tensor target = nn::Tensor::cpu(C, B);
        (void)ce.forward(engine, logits, target);
    }

    // ── Adam 优化器 step（g*g、m_hat/denom）+ 梯度裁剪（g*g、acc+col_sum）─
    {
        const std::size_t R = 8, C = 5;
        nn::Tensor p = nn::Tensor::cpu(R, C);
        nn::Tensor g = nn::Tensor::cpu(R, C);
        auto opt = nn::create_optimizer("adam", engine,
                                        std::vector<nn::TensorRef>{p},
                                        std::vector<nn::TensorRef>{g},
                                        nn::Scalar{1e-3f});
        if (opt) { (void)opt->step(); (void)opt->clip_grad_norm(nn::Scalar{1e3f}); }
    }

    // ── Linear forward（算子融合二期 S4：matmul+bias 融合路径）───────────
    // Linear::forward 已迁移为 dsl::compute(matmul(W,x) + row_broadcast(b))：
    // 折叠出前置 matmul 段 + 尾逐元素链（Add + RowBroadcast 视图）。必须
    // dry-run 覆盖本路径：GPU 运行时同一结构命中 AOT 融合 shader（闭合世界
    // 两端一致）。结构不依赖形状，任取一个 in/out/batch 即可。
    {
        const std::size_t in_f = 8, out_f = 5, B = 4;
        nn::Linear linear(in_f, out_f);
        (void)linear.init(engine);
        nn::Tensor input = nn::Tensor::cpu(in_f, B);
        (void)linear.forward(engine, input);
    }

    // ── 算子融合二期（docs/14 S1-S3）：matmul 参与 IR 融合 ───────────────
    // 结构 = Layer 内 dsl::matmul(A,B)+bias+relu 折叠后的派生物：
    //   前置 matmul 段（MatmulSpec）+ 尾逐元素链（Add + Max）。
    // 这里直接构造折叠后的结构并登记（等价于 scan 模式下 end_expr 登记复合
    // spec 的机制），保证 GPU 运行时同一结构命中 AOT 融合 shader（闭合世界）。
    //   - transA/transB 是结构 → 分别登记（4 种组合各一个 shader）
    //   - k（求和维度）是形状参数 → 不进 key：任取一个 K 登记，运行时任何 K
    //     都命中同一融合 shader（同 P2-13 的 RowMod/RotateHalf 处理）
    {
        const auto make_spec = [](std::uint8_t trA, std::uint8_t trB) {
            nn::ExprSpec s;
            s.views    = {nn::expr::linear(), nn::expr::linear(), nn::expr::linear()};
            s.num_regs = 2;
            s.matmul   = nn::MatmulSpec{0, 1, trA, trB, /*k=*/8};
            // Add r0 = matmul(A,B) + bias（Input 2）
            s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                                nn::expr::matmul_op(), nn::expr::input(2), {}});
            // Max r1 = max(r0, 0)（relu）
            s.consts.push_back(nn::Scalar{0});
            s.instrs.push_back({static_cast<std::uint8_t>(nn::ExprOp::Max), 1,
                                nn::expr::reg(0), nn::expr::cst(0), {}});
            return s;
        };
        nn::fused::global_registry().add(make_spec(0, 0));
        nn::fused::global_registry().add(make_spec(0, 1));
        nn::fused::global_registry().add(make_spec(1, 0));
        nn::fused::global_registry().add(make_spec(1, 1));
        // 纯 matmul（无逐元素链）：输出 = matmul 结果
        {
            nn::ExprSpec s;
            s.views    = {nn::expr::linear(), nn::expr::linear()};
            s.num_regs = 0;
            s.matmul   = nn::MatmulSpec{0, 1, 0, 0, /*k=*/8};
            nn::fused::global_registry().add(s);
        }
    }

    // ── 算子融合二期（docs/14 S5）：matmul+归约组合（注意力结构）──────
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

    // ── CausalSelfAttention（S7：IR 掩码组合 forward/backward）──────────
    // 掩码表达式 4 配置（causal / causal+alibi / causal+doc / causal+alibi+doc）
    // 各产生确定结构：m/l/W（matmul 段 + Row/Col/Batch 操作数 + 视图）与
    // backward 的 R/X（纯逐元素）。必须全部 dry-run 覆盖（闭合世界）。
    {
        const std::size_t d_model = 16, heads = 2, seq = 4, batch = 2;
        const auto run_csa = [&](nn::CausalSelfAttention& attn) {
            (void)attn.init(engine);
            nn::Tensor x = nn::Tensor::cpu(d_model, batch * seq);
            (void)attn.forward(engine, x);   // 填 Q/K/V/W 缓存 + 登记 m/l/W
            nn::Tensor grad = nn::Tensor::cpu(d_model, batch * seq);
            (void)attn.backward(engine, grad);  // 登记 R/X
        };
        for (const auto enc : {nn::PosEncodingType::Learned,
                               nn::PosEncodingType::ALiBi})
        {
            nn::CausalSelfAttention attn(d_model, heads, /*max_len=*/1024,
                                         /*seq_len=*/seq, enc);
            run_csa(attn);
        }
        // doc 变体（doc_ids 每位置文档 id，长度 = batch*seq）
        {
            const std::size_t doc_ids[8] = {0, 0, 1, 1, 0, 0, 1, 1};
            nn::CausalSelfAttention attn_d(d_model, heads, 1024, seq,
                                           nn::PosEncodingType::Learned);
            attn_d.set_doc_ids(doc_ids);
            run_csa(attn_d);
        }
        {
            const std::size_t doc_ids[8] = {0, 0, 1, 1, 0, 0, 1, 1};
            nn::CausalSelfAttention attn_ad(d_model, heads, 1024, seq,
                                            nn::PosEncodingType::ALiBi);
            attn_ad.set_doc_ids(doc_ids);
            run_csa(attn_ad);
        }
    }

    // ── CrossEntropyLoss 稀疏 forward（S7：IR 组合 denom/loss_vec/grad）──
    // 结构不依赖形状；含 mask 与 label 越界修正的统一 mask 构造。
    {
        const std::size_t C = 8, B = 5;
        nn::CrossEntropyLoss ce;
        nn::Tensor logits = nn::Tensor::cpu(C, B);
        std::vector<std::size_t> labels(B, 1);
        std::vector<nn::Scalar> mask(B, 1.0f);
        (void)ce.forward_sparse(engine, logits, labels, mask, C);
        // 无 mask（全 1 修正）与带越界 label 的变体
        std::vector<std::size_t> labels2(B, 0);
        labels2[0] = 99;  // 越界 → mask 修正为 0
        (void)ce.forward_sparse(engine, logits, labels2, {}, C);
    }

    auto& reg = nn::fused::global_registry();
    if (!nn::fused::write_registry(out_path, reg))
    {
        std::fprintf(stderr, "[FAIL] 无法写入 %s\n", out_path.c_str());
        return 1;
    }
    std::printf("[scan] 收集到 %zu 条融合表达式 -> %s\n",
                reg.specs.size(), out_path.c_str());
    return 0;
}
