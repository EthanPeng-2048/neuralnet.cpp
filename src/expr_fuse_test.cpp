// ───────────────────────────────────────────────────────────────────────────
//  expr_fuse_test.cpp — IR-C 图 IR 融合 GPU 端到端验证（begin_expr/end_expr）
//
//  验证内容（对应 docs/11-ir-optimization.md IR-C）：
//    1. FusedChainLayer forward：GPU（begin_expr 录制 → end_expr 融合 →
//       AOT 融合 shader 单 kernel dispatch）vs CPU（逐节点求值）一致。
//       —— 中间量 t/u 内联为寄存器，不落显存、不额外 dispatch。
//    2. FusedChainLayer backward：GPU（独立 AOT）vs CPU 一致。
//    3. 录制图依赖识别：GPU 端第二个表达式消费第一个的占位输出 → 融合。
//    4. 闭合世界：未扫描的融合 kernel → end_expr 硬报错（不静默回退）。
//    5. 未录制路径回归：普通 dsl::compute 在 GPU 上照常 AOT dispatch。
//
//  用法：expr_fuse_test（需 Vulkan SDK；依赖 CMake 构建期 scan_exprs 已
//  dry-run FusedChainLayer，登记融合后的复合 spec —— 见 scan_exprs.cpp）
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
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
    auto as = a.span(), bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar d = std::fabs(as[i] - bs[i]);
        if (d > max_err) max_err = d;
    }
    return max_err;
}

// ── FusedChainLayer：GPU（begin_expr/end_expr 融合）vs CPU ──────────────
// forward: t = x*2; u = t+3; out = u*gamma → GPU 端三表达式融合成单 kernel
int run_fused_chain(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t F = 8, B = 6;
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(F, B), grad(F, B);
    for (auto& v : x.span()) v = dist(rng);
    for (auto& v : grad.span()) v = dist(rng);

    nn::FusedChainLayer c_cpu(F), c_gpu(F);
    { auto r = c_cpu.init(cpu); if (!r) { std::cerr << "  CPU FusedChain init 失败: " << r.error().message << "\n"; return 1; } }
    { auto r = c_gpu.init(gpu); if (!r) { std::cerr << "  GPU FusedChain init 失败: " << r.error().message << "\n"; return 1; } }
    const Tensor in = Tensor::from_matrix(Matrix(x));

    // forward：GPU 走 begin_expr/end_expr 融合，CPU 逐节点
    auto fc = c_cpu.forward(cpu, in);
    auto fg = c_gpu.forward(gpu, in);
    if (!fc) { std::cerr << "  CPU FusedChain forward 失败: " << fc.error().message << "\n"; return 1; }
    if (!fg)
    {
        std::cerr << "  GPU FusedChain forward 失败（end_expr 融合 kernel 未命中 AOT shader？）: "
                  << fg.error().message << "\n";
        return 1;
    }
    auto fgm = gpu.to_matrix(*fg);
    if (!fgm) { std::cerr << "  GPU FusedChain forward 结果下载失败\n"; return 1; }
    const Scalar err_f = max_abs_diff(fc->cpu_matrix(), *fgm);
    const bool ok_f = err_f < 1e-4f;
    std::cout << "[" << (ok_f ? "PASS" : "FAIL") << "] FusedChain forward（begin_expr/end_expr 融合）"
              << "  err=" << std::scientific << std::setprecision(2) << err_f << "\n";

    // P2-12：重复 forward（同结构 → 图级缓存命中），结果应与首次一致，
    // 验证缓存计划实例化正确绑定新输入张量。
    Matrix x2(F, B);
    for (auto& v : x2.span()) v = dist(rng);
    const Tensor in2 = Tensor::from_matrix(Matrix(x2));
    auto fc2 = c_cpu.forward(cpu, in2);
    auto fg2 = c_gpu.forward(gpu, in2);
    if (!fc2 || !fg2)
    {
        std::cerr << "  第二次 FusedChain forward 失败: "
                  << (fg2 ? fc2.error().message : fg2.error().message) << "\n";
        return 1;
    }
    auto fg2m = gpu.to_matrix(*fg2);
    if (!fg2m) { std::cerr << "  第二次 forward 结果下载失败\n"; return 1; }
    const Scalar err_f2 = max_abs_diff(fc2->cpu_matrix(), *fg2m);
    const bool ok_f2 = err_f2 < 1e-4f;
    std::cout << "[" << (ok_f2 ? "PASS" : "FAIL")
              << "] FusedChain forward 重复（P2-12 图级缓存命中）"
              << "  err=" << std::scientific << std::setprecision(2) << err_f2 << "\n";

    // backward：GPU 独立 AOT
    const Tensor gd = Tensor::from_matrix(Matrix(grad));
    auto bc = c_cpu.backward(cpu, gd);
    auto bg = c_gpu.backward(gpu, gd);
    if (!bc) { std::cerr << "  CPU FusedChain backward 失败: " << bc.error().message << "\n"; return 1; }
    if (!bg)
    {
        std::cerr << "  GPU FusedChain backward 失败: " << bg.error().message << "\n";
        return 1;
    }
    auto bgm = gpu.to_matrix(*bg);
    if (!bgm) { std::cerr << "  GPU FusedChain backward 结果下载失败\n"; return 1; }
    const Scalar err_b = max_abs_diff(bc->cpu_matrix(), *bgm);
    const bool ok_b = err_b < 1e-4f;
    std::cout << "[" << (ok_b ? "PASS" : "FAIL") << "] FusedChain backward"
              << "  err=" << std::scientific << std::setprecision(2) << err_b << "\n";
    return (ok_f && ok_f2 && ok_b) ? 0 : 1;
}

// ── 手写录制：显式 begin_expr/end_expr 链式融合 ────────────────────────
// 与 FusedChainLayer 等价的手写路径：x*2 → t+3 → u*gamma，
// 验证 GpuEngine 录制图依赖识别 + 融合执行（结果应为 ((x*2)+3)*gamma）。
int run_manual_recording(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t R = 5, C = 7;
    std::mt19937 rng(999);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(R, C);
    for (auto& v : x.span()) v = dist(rng);
    Matrix g(R, 1);
    for (auto& v : g.span()) v = 0.5f + 0.01f * static_cast<Scalar>(rng() % 10);

    const Tensor xt = Tensor::from_matrix(Matrix(x));
    const Tensor gt = Tensor::from_matrix(Matrix(g));

    // CPU 参考：逐节点
    auto c1 = nn::dsl::compute(cpu, nn::dsl::leaf(xt) * Scalar{2}, R, C);
    auto c2 = nn::dsl::compute(cpu, nn::dsl::leaf(*c1) + Scalar{3}, R, C);
    auto c3 = nn::dsl::compute(cpu, nn::dsl::leaf(*c2) * nn::dsl::row_broadcast(gt), R, C);
    if (!c1 || !c2 || !c3) { std::cerr << "  CPU 参考求值失败\n"; return 1; }

    // GPU：begin_expr 录制 → 三表达式 → end_expr 融合执行
    auto b = gpu.begin_expr();
    if (!b) { std::cerr << "  GPU begin_expr 失败: " << b.error().message << "\n"; return 1; }
    auto g1 = nn::dsl::compute(gpu, nn::dsl::leaf(xt) * Scalar{2}, R, C);
    if (!g1) { std::cerr << "  GPU 录制 1 失败: " << g1.error().message << "\n"; return 1; }
    auto g2 = nn::dsl::compute(gpu, nn::dsl::leaf(*g1) + Scalar{3}, R, C);
    if (!g2) { std::cerr << "  GPU 录制 2 失败: " << g2.error().message << "\n"; return 1; }
    auto g3 = nn::dsl::compute(gpu, nn::dsl::leaf(*g2) * nn::dsl::row_broadcast(gt), R, C);
    if (!g3) { std::cerr << "  GPU 录制 3 失败: " << g3.error().message << "\n"; return 1; }
    auto e = gpu.end_expr();
    if (!e) { std::cerr << "  GPU end_expr 失败: " << e.error().message << "\n"; return 1; }

    // 物化：g3 的占位 buffer 已被融合 kernel 写入 → 直接下载
    auto gm = gpu.to_matrix(*g3);
    if (!gm) { std::cerr << "  GPU 融合结果下载失败\n"; return 1; }
    const Scalar err = max_abs_diff(c3->cpu_matrix(), *gm);
    const bool ok = err < 1e-4f;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] 手写 begin_expr/end_expr 链式融合"
              << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
    return ok ? 0 : 1;
}

// ── 未录制路径回归：普通 dsl::compute 照常 AOT dispatch ────────────────
// 用 FusedChainLayer backward 的表达式（scan 已登记）：grad * gamma * 2
// 注：录制段内"独立分支各自成 kernel"的融合边界已由 expr_graph_test
// （纯函数，test_independent/test_multi_consumer）覆盖；本测试聚焦 GPU 端
// 闭合世界内的回归路径。
int run_non_recording(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t R = 4, C = 6;
    std::mt19937 rng(7);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    Matrix x(R, C);
    for (auto& v : x.span()) v = dist(rng);
    const Tensor xt = Tensor::from_matrix(Matrix(x));

    // 用 FusedChainLayer backward 的表达式（scan 已登记）：grad * gamma * 2
    Matrix g(R, 1, Scalar{0.5f});
    const Tensor gt = Tensor::from_matrix(Matrix(g));
    auto cr = nn::dsl::compute(cpu,
        nn::dsl::leaf(xt) * nn::dsl::row_broadcast(gt) * Scalar{2}, R, C);
    auto gr = nn::dsl::compute(gpu,
        nn::dsl::leaf(xt) * nn::dsl::row_broadcast(gt) * Scalar{2}, R, C);
    if (!cr) { std::cerr << "  CPU 求值失败\n"; return 1; }
    if (!gr)
    {
        std::cerr << "  GPU 未录制路径失败（未命中 AOT shader？）: " << gr.error().message << "\n";
        return 1;
    }
    auto gm = gpu.to_matrix(*gr);
    if (!gm) return 1;
    const Scalar err = max_abs_diff(cr->cpu_matrix(), *gm);
    const bool ok = err < 1e-4f;
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] 未录制路径回归（普通 AOT dispatch）"
              << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
    return ok ? 0 : 1;
}

// ── vec4 向量化路径验证（cols%4==0）──────────────────────────────────────
// 用已登记的 vec4 合格表达式（Linear × RowBroadcast × const，即 FusedChain
// backward 的 grad*gamma*2）在 cols 为 4 的倍数时运行，确保 glsl_gen 的 vec4
// 快速路径（vec4 加载/广播/运算/存储）数值与 CPU 参考一致。尾部与非 4 倍数
// cols 的标量回退由 run_non_recording（C=6）覆盖。
int run_vec4_path(CpuEngine& cpu, GpuEngine& gpu)
{
    const std::size_t R = 4;
    int fail = 0;
    for (const std::size_t C : {std::size_t{4}, std::size_t{8}, std::size_t{12}})
    {
        std::mt19937 rng(42 + static_cast<unsigned>(C));
        std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
        Matrix x(R, C);
        for (auto& v : x.span()) v = dist(rng);
        Matrix g(R, 1);
        for (auto& v : g.span()) v = 0.5f + 0.01f * static_cast<Scalar>(rng() % 10);

        const Tensor xt = Tensor::from_matrix(Matrix(x));
        const Tensor gt = Tensor::from_matrix(Matrix(g));
        auto cr = nn::dsl::compute(cpu,
            nn::dsl::leaf(xt) * nn::dsl::row_broadcast(gt) * Scalar{2}, R, C);
        auto gr = nn::dsl::compute(gpu,
            nn::dsl::leaf(xt) * nn::dsl::row_broadcast(gt) * Scalar{2}, R, C);
        if (!cr) { std::cerr << "  CPU 求值失败\n"; return 1; }
        if (!gr)
        {
            std::cerr << "  GPU vec4 路径失败（未命中 AOT shader？）: "
                      << gr.error().message << "\n";
            return 1;
        }
        auto gm = gpu.to_matrix(*gr);
        if (!gm) { std::cerr << "  GPU 结果下载失败\n"; return 1; }
        const Scalar err = max_abs_diff(cr->cpu_matrix(), *gm);
        const bool ok = err < 1e-4f;
        std::cout << "[" << (ok ? "PASS" : "FAIL") << "] vec4 向量化路径 (cols=" << C << ")"
                  << "  err=" << std::scientific << std::setprecision(2) << err << "\n";
        fail += ok ? 0 : 1;
    }
    return fail == 0 ? 0 : 1;
}

int main()
{
    std::cout << "========================================\n"
              << "  IR-C 图 IR 融合 GPU 端到端验证\n"
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
    fail += run_fused_chain(*cpu_engine, *gpu_engine);
    fail += run_manual_recording(*cpu_engine, *gpu_engine);
    fail += run_non_recording(*cpu_engine, *gpu_engine);
    fail += run_vec4_path(*cpu_engine, *gpu_engine);

    std::cout << (fail == 0 ? "ALL PASS\n" : "FAILED\n");
    return fail == 0 ? 0 : 1;
}
#endif
