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

#include "cpu_engine.hpp"
#include "compute_layer.hpp"
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

    // ── RoPE：forward + backward，覆盖代码库实际用到的 d_k 集合 ──────────
    // （apply 与 apply_step 折叠出的结构相同，会自动去重）
    for (const std::size_t dk : {std::size_t{32}, std::size_t{64}, std::size_t{128}})
    {
        nn::RotaryEmbedding rope(engine, dk);
        nn::Tensor q = nn::Tensor::cpu(2 * dk, 8);   // rows 为 dk 的整数倍
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/false);
        (void)rope.apply(engine, q, /*seq=*/8, /*backward=*/true);
        nn::Tensor q1 = nn::Tensor::cpu(dk, 1);      // 增量推理（单位置）
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/false);
        (void)rope.apply_step(engine, q1, /*pos=*/3, /*backward=*/true);
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
        nn::RMSNorm rms(engine, F);
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
        nn::LayerNorm ln(engine, F);
        nn::Tensor input = nn::Tensor::cpu(F, B);
        (void)ln.forward(engine, input);
        nn::Tensor grad = nn::Tensor::cpu(F, B);
        (void)ln.backward(engine, grad);
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
