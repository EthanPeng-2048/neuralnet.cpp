// ───────────────────────────────────────────────────────────────────────────
//  fused_gpu_test — AOT 融合 shader GPU 数值验证（端到端）
//
//  架构：表达式**只内联写在 Layer**；构建期 scan_exprs dry-run 收集折叠后的
//  ExprSpec → gen_fused 合成融合 shader（key = expr_spec_key）；运行时
//  dsl::compute → GpuEngine::eval_expr 按 key 匹配 dispatch。
//
//  本测试验证真实路径：
//    1. 内联表达式在 GPU 端命中预编译融合 shader（未命中 eval_expr 硬报错）
//    2. GPU 融合结果 vs CPU eval_cpu 参考一致（RoPE / SwiGLU backward）
//
//  做法：用同一份输入分别驱动 CPU 引擎与 GPU 引擎上的同一批 Layer
//  （RotaryEmbedding / SwiGLU），对比输出。GPU 走融合 shader，CPU 走
//  eval_cpu，二者应一致。
//
//  用法：fused_gpu_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>

using nn::Matrix;
using nn::Tensor;
using nn::CpuEngine;
using nn::GpuEngine;
using nn::GpuBackend;

// ── 辅助：最大绝对误差 ────────────────────────────────────────────────────
Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    Scalar max_err = 0.0f;
    auto as = a.span();
    auto bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar diff = std::fabs(as[i] - bs[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

// ── RoPE：GPU（融合 shader）vs CPU（eval_cpu）───────────────────────────
// 形状无关融合：RowMod/RotateHalf 周期/块大小是运行时视图参数（不进 key），
// 任意 d_k（含非 2 的幂）都命中同一个融合 shader。
int run_rope(CpuEngine& cpu, GpuEngine& gpu, std::size_t dk, bool backward)
{
    std::mt19937 rng(1000 + static_cast<unsigned>(dk));
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    const std::size_t seq = 9;                 // 非 2 的幂，检验索引通用性
    const std::size_t rows = 2 * dk;           // d_k 的整数倍
    Matrix q(rows, seq);
    for (auto& v : q.span()) v = dist(rng);

    nn::RotaryEmbedding rope_cpu(dk);
    nn::RotaryEmbedding rope_gpu(dk);
    const Tensor qc = Tensor::from_matrix(Matrix(q));

    auto cr = rope_cpu.apply(cpu, qc, seq, backward);
    auto gr = rope_gpu.apply(gpu, qc, seq, backward);
    if (!cr) { std::cerr << "  CPU apply 失败: " << cr.error().message << "\n"; return 1; }
    if (!gr)
    {
        std::cerr << "  GPU apply 失败（未命中融合 shader？）: " << gr.error().message << "\n";
        return 1;
    }
    auto gm = gpu.to_matrix(*gr);
    if (!gm) { std::cerr << "  GPU 结果下载失败\n"; return 1; }

    const Scalar err = max_abs_diff(cr->cpu_matrix(), *gm);
    const bool ok = err < 1e-4f;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] rope "
              << (backward ? "backward" : "forward") << " dk" << dk
              << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
    return ok ? 0 : 1;
}

// ── SwiGLU backward：GPU（融合 grad_gate + grad_up）vs CPU ──────────────
int run_swiglu(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t d_ff = 8;
    const std::size_t batch = 6;
    std::mt19937 rng(555);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix input(2 * d_ff, batch), grad(d_ff, batch);
    for (auto& v : input.span()) v = dist(rng);
    for (auto& v : grad.span()) v = dist(rng);

    nn::SwiGLU sg_cpu(d_ff), sg_gpu(d_ff);
    const Tensor in = Tensor::from_matrix(Matrix(input));
    auto fc = sg_cpu.forward(cpu, in);
    auto fg = sg_gpu.forward(gpu, in);
    if (!fc) { std::cerr << "  CPU forward 失败: " << fc.error().message << "\n"; return 1; }
    if (!fg) { std::cerr << "  GPU forward 失败: " << fg.error().message << "\n"; return 1; }

    const Tensor gd = Tensor::from_matrix(Matrix(grad));
    auto bc = sg_cpu.backward(cpu, gd);
    auto bg = sg_gpu.backward(gpu, gd);
    if (!bc) { std::cerr << "  CPU backward 失败: " << bc.error().message << "\n"; return 1; }
    if (!bg)
    {
        std::cerr << "  GPU backward 失败（未命中融合 shader？）: "
                  << bg.error().message << "\n";
        return 1;
    }
    auto gm = gpu.to_matrix(*bg);
    if (!gm) { std::cerr << "  GPU 结果下载失败\n"; return 1; }

    const Scalar err = max_abs_diff(bc->cpu_matrix(), *gm);
    const bool ok = err < 1e-4f;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] swiglu backward"
              << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
    return ok ? 0 : 1;
}

// ── Softmax forward/backward：GPU（M3 行归约融合 shader）vs CPU ───────────
// forward:  exp(x - row_max) / row_sum(exp(x - row_max))
// backward: out * (grad - row_dot(out * grad))
// 归约融合 shader 未命中（未扫描）时 GPU 端 eval_expr 硬报错。
int run_softmax(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t R = 6, C = 9;   // 与 scan_exprs 结构一致（结构不依赖形状）
    std::mt19937 rng(777);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(R, C), grad(R, C);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : grad.span()) v = dist(rng);

    nn::Softmax sm_cpu, sm_gpu;
    const Tensor in = Tensor::from_matrix(Matrix(x));
    auto fc = sm_cpu.forward(cpu, in);
    auto fg = sm_gpu.forward(gpu, in);
    if (!fc) { std::cerr << "  CPU softmax forward 失败: " << fc.error().message << "\n"; return 1; }
    if (!fg)
    {
        std::cerr << "  GPU softmax forward 失败（未命中归约融合 shader？）: "
                  << fg.error().message << "\n";
        return 1;
    }
    auto fgm = gpu.to_matrix(*fg);
    if (!fgm) { std::cerr << "  GPU softmax forward 结果下载失败\n"; return 1; }
    const Scalar err_f = max_abs_diff(fc->cpu_matrix(), *fgm);
    const bool ok_f = err_f < 1e-4f;
    std::cout << "[" << (ok_f ? "PASS" : "FAIL") << "] softmax forward"
              << "  err=" << std::scientific << std::setprecision(2) << err_f << "\n";

    const Tensor gd = Tensor::from_matrix(Matrix(grad));
    auto bc = sm_cpu.backward(cpu, gd);
    auto bg = sm_gpu.backward(gpu, gd);
    if (!bc) { std::cerr << "  CPU softmax backward 失败: " << bc.error().message << "\n"; return 1; }
    if (!bg)
    {
        std::cerr << "  GPU softmax backward 失败（未命中归约融合 shader？）: "
                  << bg.error().message << "\n";
        return 1;
    }
    auto bgm = gpu.to_matrix(*bg);
    if (!bgm) { std::cerr << "  GPU softmax backward 结果下载失败\n"; return 1; }
    const Scalar err_b = max_abs_diff(bc->cpu_matrix(), *bgm);
    const bool ok_b = err_b < 1e-4f;
    std::cout << "[" << (ok_b ? "PASS" : "FAIL") << "] softmax backward"
              << "  err=" << std::scientific << std::setprecision(2) << err_b << "\n";
    return (ok_f && ok_b) ? 0 : 1;
}

// ── 归一化层通用：GPU（M3 归约融合 shader）vs CPU ────────────────────────
// 驱动同一个 Layer（RMSNorm/LayerNorm）在 CPU 与 GPU 引擎上，对比
// forward/backward 输出。未命中融合 shader 时 GPU 端 eval_expr 硬报错。
template <typename NormT>
int run_norm(const char* name, CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t F = 8, B = 5;
    std::mt19937 rng(888);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(F, B), grad(F, B);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : grad.span()) v = dist(rng);

    NormT n_cpu(F), n_gpu(F);
    { auto r = n_cpu.init(cpu); if (!r) { std::cerr << "  CPU " << name << " init 失败: " << r.error().message << "\n"; return 1; } }
    { auto r = n_gpu.init(gpu); if (!r) { std::cerr << "  GPU " << name << " init 失败: " << r.error().message << "\n"; return 1; } }
    const Tensor in = Tensor::from_matrix(Matrix(x));
    auto fc = n_cpu.forward(cpu, in);
    auto fg = n_gpu.forward(gpu, in);
    if (!fc) { std::cerr << "  CPU " << name << " forward 失败: " << fc.error().message << "\n"; return 1; }
    if (!fg)
    {
        std::cerr << "  GPU " << name << " forward 失败（未命中归约融合 shader？）: "
                  << fg.error().message << "\n";
        return 1;
    }
    auto fgm = gpu.to_matrix(*fg);
    if (!fgm) { std::cerr << "  GPU " << name << " forward 结果下载失败\n"; return 1; }
    const Scalar err_f = max_abs_diff(fc->cpu_matrix(), *fgm);
    const bool ok_f = err_f < 1e-4f;
    std::cout << "[" << (ok_f ? "PASS" : "FAIL") << "] " << name << " forward"
              << "  err=" << std::scientific << std::setprecision(2) << err_f << "\n";

    const Tensor gd = Tensor::from_matrix(Matrix(grad));
    auto bc = n_cpu.backward(cpu, gd);
    auto bg = n_gpu.backward(gpu, gd);
    if (!bc) { std::cerr << "  CPU " << name << " backward 失败: " << bc.error().message << "\n"; return 1; }
    if (!bg)
    {
        std::cerr << "  GPU " << name << " backward 失败（未命中归约融合 shader？）: "
                  << bg.error().message << "\n";
        return 1;
    }
    auto bgm = gpu.to_matrix(*bg);
    if (!bgm) { std::cerr << "  GPU " << name << " backward 结果下载失败\n"; return 1; }
    const Scalar err_b = max_abs_diff(bc->cpu_matrix(), *bgm);
    const bool ok_b = err_b < 1e-4f;
    std::cout << "[" << (ok_b ? "PASS" : "FAIL") << "] " << name << " backward"
              << "  err=" << std::scientific << std::setprecision(2) << err_b << "\n";
    return (ok_f && ok_b) ? 0 : 1;
}

// ── 闭合世界：未扫描表达式 → GPU 硬报错（绝不静默回退） ─────────────────
// 构造一个任何 Layer 都不使用的表达式（x*y + 3），其 key 不在融合注册表。
// 项目哲学：GPU eval_expr 未命中**硬报错**，不静默回退 CPU（expr_dsl.hpp
// "GPU 直接硬报错，绝不静默回退"）。本用例断言 GPU 侧必须返回错误。
int run_fallback(CpuEngine& cpu, GpuEngine& gpu)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<Scalar> dist(-2.0f, 2.0f);
    const std::size_t R = 5, C = 7;
    Matrix x(R, C), y(R, C);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : y.span()) v = dist(rng);
    const Tensor xt = Tensor::from_matrix(Matrix(x));
    const Tensor yt = Tensor::from_matrix(Matrix(y));
    const nn::Scalar three{3};

    // CPU 求值正常（模板路径）
    auto cr = nn::dsl::compute(cpu,
        nn::dsl::leaf(xt) * nn::dsl::leaf(yt) + three, R, C);
    if (!cr) { std::cerr << "  CPU 求值失败: " << cr.error().message << "\n"; return 1; }

    // 未扫描表达式在 GPU 上必须**硬报错**（闭合世界，绝不静默回退）
    auto gr = nn::dsl::compute(gpu,
        nn::dsl::leaf(xt) * nn::dsl::leaf(yt) + three, R, C);
    const bool ok = !gr;
    if (!ok)
        std::cerr << "  未扫描表达式在 GPU 上未报错（应硬报错而非静默回退/成功）\n";
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] 未扫描表达式 → GPU 硬报错（不静默回退）\n";
    return ok ? 0 : 1;
}

int main()
{
    std::cout << "========================================\n"
              << "  AOT 融合 shader GPU 数值验证（端到端）\n"
              << "========================================\n";

    auto cpu_engine = std::make_unique<CpuEngine>();
    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    auto gpu_engine = std::make_unique<GpuEngine>(backend);
    std::cout << "[init] CpuEngine + GpuEngine 就绪\n";

    int fail = 0;
    // 形状无关融合：任意 d_k（含非 2 的幂 40/96）都命中同一个融合 shader
    for (const std::size_t dk : {std::size_t{16}, std::size_t{40},
                                 std::size_t{64}, std::size_t{96},
                                 std::size_t{128}})
    {
        fail += run_rope(*cpu_engine, *gpu_engine, dk, /*backward=*/false);
        fail += run_rope(*cpu_engine, *gpu_engine, dk, /*backward=*/true);
    }
    fail += run_swiglu(*cpu_engine, *gpu_engine);
    fail += run_softmax(*cpu_engine, *gpu_engine);
    fail += run_norm<nn::RMSNorm>("rmsnorm", *cpu_engine, *gpu_engine);
    fail += run_norm<nn::LayerNorm>("layernorm", *cpu_engine, *gpu_engine);
    fail += run_fallback(*cpu_engine, *gpu_engine);

    std::cout << (fail == 0 ? "ALL PASS\n" : "FAILED\n");
    return fail == 0 ? 0 : 1;
}
#endif
