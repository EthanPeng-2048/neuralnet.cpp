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
#include <neuralnet.cpp/expr_fold.hpp>   // P-C1 fold 样例（与 scan_exprs 同源 → key 一致）

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
    // 返回 77 = ctest SKIP：纯 CPU 构建无法执行 GPU 测试，不得计为 "Passed"
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n";
    return 77;
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

    // col_max(matmul(Q,K^T))：列归约 + matmul 组合（S5 列方向）。
    // 该组合曾被 gen_fused 跳过（"matmul+列归约组合暂不支持"），生成器补齐
    // 按元素 batch 分解后，此对拍锁死 GPU 与 CPU 语义一致（列归约遍历全部
    // rows，含所有 batch）。广播输出（dsl::compute → vector_out=0）：
    auto ccol = nn::dsl::compute(cpu,
        nn::dsl::col_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    auto gcol = nn::dsl::compute(gpu,
        nn::dsl::col_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    if (!ccol || !gcol)
    {
        std::cerr << "  col_max(matmul) 求值失败（CPU/GPU）: "
                  << (!ccol ? ccol.error().message : gcol.error().message) << "\n";
        return 1;
    }
    auto gcolm = gpu.to_matrix(*gcol);
    if (!gcolm) { std::cerr << "  GPU col_max(matmul) 下载失败\n"; return 1; }
    const Scalar err3 = max_abs_diff(ccol->cpu_matrix(), *gcolm);
    const bool ok3 = err3 < 1e-4f;
    std::cout << "[" << (ok3 ? "PASS" : "FAIL") << "] col_max(matmul(Q,K^T)) 广播融合 (5x7x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err3 << "\n";

    // 归约向量输出（dsl::compute_reduce → vector_out=1，输出 (1,N)）
    auto ccvr = nn::dsl::compute_reduce(cpu,
        nn::dsl::col_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    auto gcvr = nn::dsl::compute_reduce(gpu,
        nn::dsl::col_reduce_max(nn::dsl::matmul(Qt, Kt, false, true)), M, N);
    if (!ccvr || !gcvr)
    {
        std::cerr << "  col_max(matmul) 归约向量求值失败（CPU/GPU）: "
                  << (!ccvr ? ccvr.error().message : gcvr.error().message) << "\n";
        return 1;
    }
    auto gcvm = gpu.to_matrix(*gcvr);
    if (!gcvm) { std::cerr << "  GPU col_max(matmul) 向量下载失败\n"; return 1; }
    const Scalar err4 = max_abs_diff(ccvr->cpu_matrix(), *gcvm);
    const bool ok4 = err4 < 1e-4f;
    std::cout << "[" << (ok4 ? "PASS" : "FAIL") << "] col_max(matmul) 归约向量 (1x4)"
              << "  err=" << std::scientific << std::setprecision(2) << err4 << "\n";

    // batch=2（S7 batch 分解）：A 堆叠 (2M,K)、K 存储 (2N,K) —— 列归约须
    // 遍历全部 2M 行，每行按 row/m_per 就地分解 batch
    {
        Matrix Q2m(M * 2, K), K2m(N * 2, K);
        for (auto& v : Q2m.span()) v = dist(rng);
        for (auto& v : K2m.span()) v = dist(rng);
        const Tensor Q2t = Tensor::from_matrix(Matrix(Q2m));
        const Tensor K2t = Tensor::from_matrix(Matrix(K2m));
        auto cb2 = nn::dsl::compute_reduce(cpu,
            nn::dsl::col_reduce_max(
                nn::dsl::matmul(Q2t, K2t, false, true, /*batch=*/2)), M * 2, N);
        auto gb2 = nn::dsl::compute_reduce(gpu,
            nn::dsl::col_reduce_max(
                nn::dsl::matmul(Q2t, K2t, false, true, /*batch=*/2)), M * 2, N);
        if (!cb2 || !gb2)
        {
            std::cerr << "  col_max(matmul) batch=2 求值失败（CPU/GPU）: "
                      << (!cb2 ? cb2.error().message : gb2.error().message) << "\n";
            return 1;
        }
        auto gb2m = gpu.to_matrix(*gb2);
        if (!gb2m) { std::cerr << "  GPU col_max(matmul) batch=2 下载失败\n"; return 1; }
        const Scalar err5 = max_abs_diff(cb2->cpu_matrix(), *gb2m);
        const bool ok5 = err5 < 1e-4f;
        std::cout << "[" << (ok5 ? "PASS" : "FAIL")
                  << "] col_max(matmul) batch=2 归约向量 (1x4)"
                  << "  err=" << std::scientific << std::setprecision(2) << err5 << "\n";
        if (!ok5) return 1;
    }

    return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
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

// ── 归约表达式内联常量：CPU vs GPU（push-constant 头长度回归）─────────────
// 结构（与 tools/scan_exprs.cpp 的对应 dry-run 完全一致）：
//   col_reduce_sum(select(x == col_broadcast(mx), 1, 0))
// 该结构带**常量池**。GPU 侧 push-constant 固定头长度必须按形态算；历史上
// "归约但无 matmul" 曾多算 1 个 uint（5 vs 4）→ 常量池整体后移一个 uint →
// shader 读错常量 → **GPU 静默错值而 CPU 正常**。本用例锁死该回归。
int run_reduce_consts(nn::ComputeEngine& cpu, nn::ComputeEngine& gpu)
{
    const std::size_t kk = 4, cols = 3;
    // 列 0：两路并列最大（max=1 出现 2 次）
    // 列 1：无并列（max=5 出现 1 次）
    // 列 2：全相等（max=7 出现 4 次）
    Matrix xm(kk, cols);
    xm.span()[0] = 1; xm.span()[1] = 5; xm.span()[2] = 7;
    xm.span()[3] = 1; xm.span()[4] = 4; xm.span()[5] = 7;
    xm.span()[6] = 0; xm.span()[7] = 3; xm.span()[8] = 7;
    xm.span()[9] = 0; xm.span()[10] = 2; xm.span()[11] = 7;
    Matrix mxm(1, cols);
    mxm.span()[0] = 1; mxm.span()[1] = 5; mxm.span()[2] = 7;
    const Scalar expect[3] = {2, 1, 4};

    const auto eval = [&](nn::ComputeEngine& e, const char* tag, Matrix& out) -> bool {
        auto x = e.from_matrix(xm);
        auto mx = e.from_matrix(mxm);
        if (!x || !mx) { std::cerr << "    [" << tag << "] 上传失败\n"; return false; }
        auto r = nn::dsl::compute_reduce(e,
            nn::dsl::col_reduce_sum(nn::dsl::select(
                nn::dsl::leaf(*x) == nn::dsl::col_broadcast(*mx),
                nn::Scalar{1}, nn::Scalar{0})),
            kk, cols);
        if (!r) { std::cerr << "    [" << tag << "] compute_reduce 失败: "
                            << r.error().message << "\n"; return false; }
        auto m = e.to_matrix(*r);
        if (!m) { std::cerr << "    [" << tag << "] to_matrix 失败\n"; return false; }
        out = *m;
        return true;
    };

    Matrix mc, mg;
    if (!eval(cpu, "CPU", mc)) { std::cout << "[FAIL] reduce_consts: CPU 求值失败\n"; return 1; }
    if (!eval(gpu, "GPU", mg)) { std::cout << "[FAIL] reduce_consts: GPU 求值失败\n"; return 1; }

    const Scalar diff = max_abs_diff(mc, mg);
    bool ok = (diff <= 1e-6f) && mc.rows() == 1 && mc.cols() == cols;
    for (std::size_t i = 0; ok && i < cols; ++i)
        ok = std::fabs(mc.span()[i] - expect[i]) < 1e-6f;
    if (!ok)
        std::cerr << "    CPU vs GPU max_diff=" << diff
                  << "（GPU 常量池错位的历史 bug 回归）\n";
    std::cout << "[" << (ok ? "PASS" : "FAIL")
              << "] 归约内联常量（push-constant 固定头长度）CPU/GPU 一致\n";
    return ok ? 0 : 1;
}

// ── P-C1 fold 分块状态归约：CPU vs GPU 对拍 ──────────────────────────────
// 同 spec → 同 key → 命中 scan 收集的 fold shader；同分块(EXPR_FOLD_BLOCK)、
// 同指令序、每线程一行与 CPU 执行器逐指令同构 → rowmax 期望**逐位一致**；
// rowsum/denom 给小容差（fp 加法结合序驱动差异 / exp 软硬件实现差异）。
// K 族覆盖单列 / 非块整除 / 整除 / 尾块（EXPR_FOLD_BLOCK=128 边界两侧——
// 256/260 补多块+尾块；BLOCK 升 128 后 K≤100 会静默退化单块）。
int run_fold_gpu(CpuEngine& cpu, GpuEngine& gpu)
{
    int fail = 0;
    std::mt19937 rng(9013);
    std::uniform_real_distribution<Scalar> dist(-2.0f, 2.0f);
    for (const std::size_t K : {std::size_t{1}, std::size_t{7}, std::size_t{32},
                                std::size_t{33}, std::size_t{100},
                                std::size_t{256}, std::size_t{260}})
    {
        const std::size_t rows = 6;
        Tensor x = Tensor::cpu(rows, K);
        {
            auto sp = x.cpu_matrix().span();
            for (auto& v : sp) v = dist(rng);
        }
        const std::vector<Tensor> ins{x};

        const auto one = [&](const char* tag, const nn::ExprSpec& spec, Scalar tol)
        {
            auto c = cpu.eval_expr(spec, ins, rows, 1);
            auto g = gpu.eval_expr(spec, ins, rows, 1);
            if (!c)
            {
                std::cout << "  [FAIL] fold " << tag << " K=" << K
                          << " CPU: " << c.error().message << "\n";
                ++fail; return;
            }
            if (!g)
            {
                std::cout << "  [FAIL] fold " << tag << " K=" << K
                          << " GPU: " << g.error().message << "\n";
                ++fail; return;
            }
            auto gm = gpu.to_matrix(*g);
            if (!gm)
            {
                std::cout << "  [FAIL] fold " << tag << " K=" << K
                          << " GPU 下载失败\n";
                ++fail; return;
            }
            const auto cs = c->cpu_matrix().span();
            const auto gs = gm->span();
            Scalar err = 0;
            for (std::size_t i = 0; i < rows; ++i)
                err = std::fmax(err, std::fabs(cs[i] - gs[i]) /
                                     std::fmax(Scalar{1}, std::fabs(cs[i])));
            const bool ok = err <= tol;
            std::cout << "[" << (ok ? "PASS" : "FAIL") << "] fold " << tag
                      << " K=" << K << "  err=" << std::scientific
                      << std::setprecision(2) << err << std::defaultfloat
                      << (tol == 0 ? " (exact)" : "") << "\n";
            if (!ok)
            {
                // 诊断：打印全部行的 CPU/GPU 实值（行同值/巨值形态定位）
                std::cout << "      cpu:";
                for (std::size_t i = 0; i < rows; ++i)
                    std::cout << " " << cs[i];
                std::cout << "\n      gpu:";
                for (std::size_t i = 0; i < rows; ++i)
                    std::cout << " " << gs[i];
                std::cout << "\n";
                ++fail;
            }
        };

        one("rowmax", nn::expr::make_fold_rowmax(
                static_cast<std::uint32_t>(K)), Scalar{0});
        one("rowsum", nn::expr::make_fold_rowsum(
                static_cast<std::uint32_t>(K)), Scalar{1e-6});
        one("softmax_denom", nn::expr::make_fold_softmax_denom(
                static_cast<std::uint32_t>(K)), Scalar{1e-5});
    }
    return fail;
}

// ── P-C2 attention fold（v2 双域）GPU 对拍 ──────────────────────────────
// 与 scan 登记的 spec 同源（vec_state_len 不进 key——形状参数经 PC
// vector_out 槽填充，dk 族登记是同 key 去重）；v2 与 CPU 执行器同分块/
// 同指令序——max 类逐位；sum/vecacc 类 GPU 蝶形结合序异于 CPU 串行、exp
// 软硬件有差 → 走容差 1e-4（非全逐位）。
// 覆盖 mm 段/5 掩码/vecacc/向量域 finalize/dispatch(ceil(rows/NR))、
// seq>EXPR_FOLD_BLOCK 多块流式 + causal 整块跳过、GPU rows=1。
int run_fold_attn_gpu(CpuEngine& cpu, GpuEngine& gpu)
{
    int fail = 0;
    std::mt19937 rng(9024);
    std::uniform_real_distribution<Scalar> dist(-1.5f, 1.5f);
    struct Sh { std::uint32_t bh, seq, dk; };
    for (const auto mk : {nn::expr::FoldAttnMask::Plain,
                          nn::expr::FoldAttnMask::Causal,
                          nn::expr::FoldAttnMask::Alibi,
                          nn::expr::FoldAttnMask::Doc,
                          nn::expr::FoldAttnMask::AlibiDoc})
    {
        const char* mname = mk == nn::expr::FoldAttnMask::Plain    ? "plain"
                         : mk == nn::expr::FoldAttnMask::Causal    ? "causal"
                         : mk == nn::expr::FoldAttnMask::Alibi     ? "alibi"
                         : mk == nn::expr::FoldAttnMask::Doc       ? "doc"
                                                                 : "alibidoc";
        // {2,133,4}：seq > EXPR_FOLD_BLOCK(128) → GPU 多块流式 + causal
        //   整块跳过分支（k0>qt）；{1,1,4}：GPU 最小 rows=1（clamp+row_ok）
        for (const Sh sh : {Sh{2, 9, 4}, Sh{1, 33, 8}, Sh{3, 5, 2},
                            Sh{2, 133, 4}, Sh{1, 1, 4}})
        {
            const std::size_t rows_out = static_cast<std::size_t>(sh.bh) * sh.seq;
            Tensor Q = Tensor::cpu(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            Tensor K = Tensor::cpu(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            Tensor Vt = Tensor::cpu(rows_out, sh.dk);
            Tensor slopes = Tensor::cpu(1, sh.bh);
            // doc 输入（与 expr_fold_test / Layer 同构）：doc_col (rows,1) 行
            //   文档 id；doc_ids (1, bh*seq) 按 (b,h) 块重复（BatchCol(seq)）
            const std::uint32_t sseq = sh.seq;
            Tensor doc_col = Tensor::cpu(rows_out, 1);
            Tensor doc_ids_t = Tensor::cpu(
                1, static_cast<std::size_t>(sh.bh) * sh.seq);
            // doc 分段边界：seq > EXPR_FOLD_BLOCK(128) 时越过 128（133→129），
            //   让 i≥129 的行首 fold 块被 doc 掩码**全屏蔽**——钉住 2026-09
            //   GPU -nan 根因（max init=-inf → dm=−inf−−inf=NaN）；
            //   旧固定 seq/2=66<128 首块永留有效项 → 对拍漏抓该 bug。
            const std::uint32_t doc_boundary =
                sseq > nn::EXPR_FOLD_BLOCK ? nn::EXPR_FOLD_BLOCK + 1u
                                           : sseq / 2u;
            const auto doc_of = [doc_boundary](std::uint32_t pos) -> Scalar
            { return pos < doc_boundary ? Scalar{1} : Scalar{2}; };
            for (std::size_t r = 0; r < rows_out; ++r)
                doc_col.cpu_matrix().span()[r] =
                    doc_of(static_cast<std::uint32_t>(r % sh.seq));
            for (std::uint32_t blk = 0; blk < sh.bh; ++blk)
                for (std::uint32_t j = 0; j < sh.seq; ++j)
                    doc_ids_t.cpu_matrix().span()
                        [static_cast<std::size_t>(blk) * sh.seq + j] = doc_of(j);
            for (auto& v : Q.cpu_matrix().span()) v = dist(rng);
            for (auto& v : K.cpu_matrix().span()) v = dist(rng);
            for (auto& v : Vt.cpu_matrix().span()) v = dist(rng);
            for (auto& v : slopes.cpu_matrix().span()) v = dist(rng) * Scalar{0.1};
            const bool alibi = (mk == nn::expr::FoldAttnMask::Alibi ||
                                mk == nn::expr::FoldAttnMask::AlibiDoc);
            const bool docm  = (mk == nn::expr::FoldAttnMask::Doc ||
                                mk == nn::expr::FoldAttnMask::AlibiDoc);
            std::vector<Tensor> ins{Q, K, Vt};
            if (alibi) ins.push_back(slopes);
            if (docm) { ins.push_back(doc_col); ins.push_back(doc_ids_t); }

            nn::ExprSpec spec = nn::expr::make_fold_attn_o(sh.seq, sh.dk, sh.bh, mk);
            auto c = cpu.eval_expr(spec, ins, rows_out, sh.dk);
            auto g = gpu.eval_expr(spec, ins, rows_out, sh.dk);
            const std::string tag = std::string("attn-fold ") + mname +
                " bh=" + std::to_string(sh.bh) + " seq=" + std::to_string(sh.seq) +
                " dk=" + std::to_string(sh.dk);
            if (!c)
            {
                std::cout << "  [FAIL] " << tag << " CPU: "
                          << c.error().message << "\n";
                ++fail; continue;
            }
            if (!g)
            {
                std::cout << "  [FAIL] " << tag << " GPU: "
                          << g.error().message << "\n";
                ++fail; continue;
            }
            auto gm = gpu.to_matrix(*g);
            if (!gm)
            {
                std::cout << "  [FAIL] " << tag << " GPU 下载失败\n";
                ++fail; continue;
            }
            const auto cs = c->cpu_matrix().span();
            const auto gs = gm->span();
            Scalar err = 0;
            for (std::size_t i = 0; i < rows_out * sh.dk; ++i)
            {
                // NaN 守卫：GPU 输出 NaN → diff=NaN → IEEE fmax(err,NaN)=err
                //   会静默吞掉 → err 保持正常值照样 PASS（2026-09 fold doc
                //   -nan 回归双层漏抓之一，红验证实证）→ 记 inf 必超容差
                const Scalar diff = std::fabs(cs[i] - gs[i]) /
                                    std::fmax(Scalar{1}, std::fabs(cs[i]));
                err = std::isfinite(diff) ? std::fmax(err, diff)
                                          : std::numeric_limits<Scalar>::infinity();
                if (!std::isfinite(err)) break;
            }
            const bool ok = err <= Scalar{1e-4};
            std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << tag
                      << "  err=" << std::scientific << std::setprecision(2)
                      << err << std::defaultfloat << "\n";
            if (!ok)
            {
                std::cout << "      cpu[0..3]:";
                for (std::size_t i = 0; i < 4; ++i) std::cout << " " << cs[i];
                std::cout << "\n      gpu[0..3]:";
                for (std::size_t i = 0; i < 4; ++i) std::cout << " " << gs[i];
                std::cout << "\n";
                ++fail;
            }
        }
    }
    return fail;
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
    fail += run_reduce_consts(*cpu_engine, *gpu_engine);
    fail += run_fold_gpu(*cpu_engine, *gpu_engine);
    fail += run_fold_attn_gpu(*cpu_engine, *gpu_engine);
    fail += run_fallback(*cpu_engine, *gpu_engine);

    std::cout << (fail == 0 ? "ALL PASS\n" : "FAILED\n");
    return fail == 0 ? 0 : 1;
}
#endif
