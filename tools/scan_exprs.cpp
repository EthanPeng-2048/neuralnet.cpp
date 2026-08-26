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
