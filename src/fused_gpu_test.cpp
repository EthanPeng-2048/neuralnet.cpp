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
#include <neuralnet.cpp/backend/compute_vk_backend.hpp>

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

// ── GeLU forward/backward：GPU（单表达式 DSL 融合 shader）vs CPU ─────────
// forward:  x / (1 + exp(-βx))；backward: grad_out * s*(1 + βx*(1-s))。
// GPU 走融合 shader（scan 已收集 GeLU 结构），CPU 走 eval_cpu，二者应一致。
int run_gelu(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t R = 6, C = 9;   // 与 scan_exprs 结构一致（结构不依赖形状）
    std::mt19937 rng(999);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(R, C), grad(R, C);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : grad.span()) v = dist(rng);

    nn::GeLU gl_cpu, gl_gpu;
    const Tensor in = Tensor::from_matrix(Matrix(x));
    auto fc = gl_cpu.forward(cpu, in);
    auto fg = gl_gpu.forward(gpu, in);
    if (!fc) { std::cerr << "  CPU gelu forward 失败: " << fc.error().message << "\n"; return 1; }
    if (!fg)
    {
        std::cerr << "  GPU gelu forward 失败（未命中融合 shader？）: "
                  << fg.error().message << "\n";
        return 1;
    }
    const Scalar err_f = max_abs_diff(fc->cpu_matrix(), *gpu.to_matrix(*fg));

    const Tensor gd = Tensor::from_matrix(Matrix(grad));
    auto bc = gl_cpu.backward(cpu, gd);
    auto bg = gl_gpu.backward(gpu, gd);
    if (!bc) { std::cerr << "  CPU gelu backward 失败: " << bc.error().message << "\n"; return 1; }
    if (!bg)
    {
        std::cerr << "  GPU gelu backward 失败（未命中融合 shader？）: "
                  << bg.error().message << "\n";
        return 1;
    }
    const Scalar err_b = max_abs_diff(bc->cpu_matrix(), *gpu.to_matrix(*bg));

    const bool ok_f = err_f < 1e-4f;
    const bool ok_b = err_b < 1e-4f;
    std::cout << "[" << (ok_f ? "PASS" : "FAIL") << "] gelu forward"
              << "  err=" << std::scientific << std::setprecision(2) << err_f << "\n";
    std::cout << "[" << (ok_b ? "PASS" : "FAIL") << "] gelu backward"
              << "  err=" << std::scientific << std::setprecision(2) << err_b << "\n";
    return (ok_f && ok_b) ? 0 : 1;
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

// ── matmul 融合（二期 S3）：matmul(A,B)+bias+relu 单 kernel vs CPU ────────
// DSL 折叠出前置 matmul 段（MatmulSpec）+ 尾逐元素链，GPU 命中 AOT 合成
// 的 matmul 融合 shader（scan_exprs 已登记该结构）；k（求和维度）是形状
// 参数不进 key：不同 K 共享同一融合 shader（mm_k push constant 运行时填充）。
int run_matmul(CpuEngine& cpu, GpuEngine& gpu)
{
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    const std::size_t M = 5, K = 7, N = 4;
    Matrix A(M, K), B(K, N), bias(M, N);
    for (auto& v : A.span()) v = dist(rng);
    for (auto& v : B.span()) v = dist(rng);
    for (auto& v : bias.span()) v = dist(rng);
    const Tensor At = Tensor::from_matrix(Matrix(A));
    const Tensor Bt = Tensor::from_matrix(Matrix(B));
    const Tensor biast = Tensor::from_matrix(Matrix(bias));

    // matmul(A,B) + bias + relu（DSL 折叠 → matmul 段 + 尾链）
    auto cr = nn::dsl::compute(cpu,
        nn::dsl::max(nn::dsl::matmul(At, Bt) + nn::dsl::leaf(biast), Scalar{0}),
        M, N);
    auto gr = nn::dsl::compute(gpu,
        nn::dsl::max(nn::dsl::matmul(At, Bt) + nn::dsl::leaf(biast), Scalar{0}),
        M, N);
    if (!cr) { std::cerr << "  CPU matmul 融合求值失败: " << cr.error().message << "\n"; return 1; }
    if (!gr)
    {
        std::cerr << "  GPU matmul 融合求值失败（未命中 matmul 融合 shader？）: "
                  << gr.error().message << "\n";
        return 1;
    }
    auto gm = gpu.to_matrix(*gr);
    if (!gm) { std::cerr << "  GPU matmul 融合结果下载失败\n"; return 1; }
    const Scalar err1 = max_abs_diff(cr->cpu_matrix(), *gm);
    const bool ok1 = err1 < 1e-4f;
    std::cout << "[" << (ok1 ? "PASS" : "FAIL") << "] matmul+bias+relu 融合 (5x7x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err1 << "\n";

    // 纯 matmul（无逐元素链）：输出 = matmul 结果
    auto cp = nn::dsl::compute(cpu, nn::dsl::matmul(At, Bt), M, N);
    auto gp = nn::dsl::compute(gpu, nn::dsl::matmul(At, Bt), M, N);
    if (!cp || !gp)
    {
        std::cerr << "  纯 matmul 求值失败（"
                  << (!cp ? ("CPU: " + cp.error().message) : "")
                  << (!cp && !gp ? "；" : "")
                  << (!gp ? ("GPU: " + gp.error().message) : "") << "）\n";
        return 1;
    }
    auto gpm = gpu.to_matrix(*gp);
    if (!gpm) { std::cerr << "  GPU 纯 matmul 结果下载失败\n"; return 1; }
    const Scalar err2 = max_abs_diff(cp->cpu_matrix(), *gpm);
    const bool ok2 = err2 < 1e-4f;
    std::cout << "[" << (ok2 ? "PASS" : "FAIL") << "] 纯 matmul 融合 (5x7x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err2 << "\n";

    // 形状无关：不同 K（含非对齐）命中同一融合 shader（mm_k 运行时填充）
    const std::size_t M3 = 3, K3 = 6, N3 = 2;
    Matrix A3(M3, K3), B3(K3, N3), bias3(M3, N3);
    for (auto& v : A3.span()) v = dist(rng);
    for (auto& v : B3.span()) v = dist(rng);
    for (auto& v : bias3.span()) v = dist(rng);
    const Tensor A3t = Tensor::from_matrix(Matrix(A3));
    const Tensor B3t = Tensor::from_matrix(Matrix(B3));
    const Tensor b3t = Tensor::from_matrix(Matrix(bias3));
    auto c3 = nn::dsl::compute(cpu,
        nn::dsl::max(nn::dsl::matmul(A3t, B3t) + nn::dsl::leaf(b3t), Scalar{0}),
        M3, N3);
    auto g3 = nn::dsl::compute(gpu,
        nn::dsl::max(nn::dsl::matmul(A3t, B3t) + nn::dsl::leaf(b3t), Scalar{0}),
        M3, N3);
    if (!c3 || !g3)
    {
        std::cerr << "  K=6 matmul 融合求值失败（CPU/GPU）\n";
        return 1;
    }
    auto g3m = gpu.to_matrix(*g3);
    if (!g3m) { std::cerr << "  GPU K=6 结果下载失败\n"; return 1; }
    const Scalar err3 = max_abs_diff(c3->cpu_matrix(), *g3m);
    const bool ok3 = err3 < 1e-4f;
    std::cout << "[" << (ok3 ? "PASS" : "FAIL") << "] matmul 融合 形状无关 K=6 (3x6x2)"
              << "  err=" << std::scientific << std::setprecision(2) << err3 << "\n";

    // Linear 结构（S4）：matmul(W,x) + row_broadcast(b)（W (out,in)、x (in,batch)、
    // b (out,1) 行广播 → 输出 (out,batch)）。与 Linear::forward 完全相同的表达式。
    const std::size_t in_f = 6, out_f = 4, batch = 3;
    Matrix Wm(out_f, in_f), Xm(in_f, batch), bm(out_f, 1);
    for (auto& v : Wm.span()) v = dist(rng);
    for (auto& v : Xm.span()) v = dist(rng);
    for (auto& v : bm.span()) v = dist(rng);
    const Tensor Wt = Tensor::from_matrix(Matrix(Wm));
    const Tensor Xt = Tensor::from_matrix(Matrix(Xm));
    const Tensor bt = Tensor::from_matrix(Matrix(bm));
    auto cl = nn::dsl::compute(cpu,
        nn::dsl::matmul(Wt, Xt) + nn::dsl::row_broadcast(bt), out_f, batch);
    auto gl = nn::dsl::compute(gpu,
        nn::dsl::matmul(Wt, Xt) + nn::dsl::row_broadcast(bt), out_f, batch);
    if (!cl || !gl)
    {
        std::cerr << "  Linear 结构 matmul+row_broadcast 求值失败（CPU/GPU）\n";
        return 1;
    }
    auto glm = gpu.to_matrix(*gl);
    if (!glm) { std::cerr << "  GPU Linear 结构结果下载失败\n"; return 1; }
    const Scalar err4 = max_abs_diff(cl->cpu_matrix(), *glm);
    const bool ok4 = err4 < 1e-4f;
    std::cout << "[" << (ok4 ? "PASS" : "FAIL") << "] Linear 结构 matmul+row_broadcast (4x6x3)"
              << "  err=" << std::scientific << std::setprecision(2) << err4 << "\n";

    return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}

// ── matmul+归约（S5）：注意力 forward 结构，GPU（归约融合 shader 内联
//    点积，不物化 (M,N) 得分矩阵）vs CPU ─────────────────────────────────
//   row_max(QK^T)（bmm_reduce Max 等价）与 denom = row_sum(exp(QK^T - rm))
//   （bmm_denom 等价）均为"matmul 段 + 归约指令"单表达式，GPU 经
//   generate_glsl_reduce 的 matmul 支持单 kernel 完成。
int run_matmul_reduce(CpuEngine& cpu, GpuEngine& gpu)
{
    std::mt19937 rng(31415);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    const std::size_t M = 5, K = 7, N = 4;   // 非 16 倍数：分块/归约边界覆盖
    Matrix Qm(M, K), Km(N, K), rm_m(M, 1);   // K 存储 (N,K)（transB=1）
    for (auto& v : Qm.span()) v = dist(rng);
    for (auto& v : Km.span()) v = dist(rng);
    for (auto& v : rm_m.span()) v = dist(rng);
    const Tensor Qt = Tensor::from_matrix(Matrix(Qm));
    const Tensor Kt = Tensor::from_matrix(Matrix(Km));
    const Tensor rmt = Tensor::from_matrix(Matrix(rm_m));

    // row_max(matmul(Q, K^T))：行归约 max（输出 (M,1)）
    auto cr = nn::dsl::compute(cpu,
        nn::dsl::row_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    auto gr = nn::dsl::compute(gpu,
        nn::dsl::row_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    if (!cr || !gr)
    {
        std::cerr << "  row_max(matmul) 求值失败（CPU/GPU）: "
                  << (!cr ? cr.error().message : gr.error().message) << "\n";
        return 1;
    }
    auto grm = gpu.to_matrix(*gr);
    if (!grm) { std::cerr << "  GPU row_max(matmul) 下载失败\n"; return 1; }
    const Scalar err1 = max_abs_diff(cr->cpu_matrix(), *grm);
    const bool ok1 = err1 < 1e-4f;
    std::cout << "[" << (ok1 ? "PASS" : "FAIL") << "] row_max(matmul(Q,K^T)) 融合 (5x7x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err1 << "\n";

    // denom = row_sum(exp(matmul(Q,K^T) - rb(row_max)))：输出 (M,1)
    auto cd = nn::dsl::compute(cpu,
        nn::dsl::row_reduce_sum(nn::dsl::exp(
            nn::dsl::matmul(Qt, Kt, false, true) - nn::dsl::row_broadcast(rmt))),
        M, N);
    auto gd = nn::dsl::compute(gpu,
        nn::dsl::row_reduce_sum(nn::dsl::exp(
            nn::dsl::matmul(Qt, Kt, false, true) - nn::dsl::row_broadcast(rmt))),
        M, N);
    if (!cd || !gd)
    {
        std::cerr << "  denom(matmul) 求值失败（CPU/GPU）: "
                  << (!cd ? cd.error().message : gd.error().message) << "\n";
        return 1;
    }
    auto gdm = gpu.to_matrix(*gd);
    if (!gdm) { std::cerr << "  GPU denom(matmul) 下载失败\n"; return 1; }
    const Scalar err2 = max_abs_diff(cd->cpu_matrix(), *gdm);
    const bool ok2 = err2 < 1e-4f;
    std::cout << "[" << (ok2 ? "PASS" : "FAIL") << "] denom row_sum(exp(matmul-rm)) 融合 (5x7x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err2 << "\n";

    return (ok1 && ok2) ? 0 : 1;
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
    fail += run_gelu(*cpu_engine, *gpu_engine);
    fail += run_softmax(*cpu_engine, *gpu_engine);
    fail += run_matmul(*cpu_engine, *gpu_engine);
    fail += run_matmul_reduce(*cpu_engine, *gpu_engine);
    fail += run_norm<nn::RMSNorm>("rmsnorm", *cpu_engine, *gpu_engine);
    fail += run_norm<nn::LayerNorm>("layernorm", *cpu_engine, *gpu_engine);
    fail += run_fallback(*cpu_engine, *gpu_engine);

    std::cout << (fail == 0 ? "ALL PASS\n" : "FAILED\n");
    return fail == 0 ? 0 : 1;
}
#endif
