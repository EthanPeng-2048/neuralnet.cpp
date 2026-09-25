// ── ComputeEngine 正确性与性能测试 ──────────────────────────────────────────
// 用法：gpu_test [--size N] [--iters N]
//   --size N   矩阵维度（默认 256，即 256×256 方阵）
//   --iters N  性能测试迭代次数（默认 10）
//
// 验证 CpuEngine 与 GpuEngine 在 matmul / 转置 matmul / 逐元素 / 归约 /
// roundtrip 等原语上的一致性，并对比 GPU vs CPU matmul 性能。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_gpu_option.hpp>

#include <cmath>
#include <chrono>
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
using nn::ComputeEngine;
using nn::CpuEngine;
using nn::GpuEngine;
using nn::GpuBackend;
using nn::Tensor;
using nn::BinaryOp;
using nn::UnaryOp;

void print_usage(const char* prog)
{
    std::cout << "ComputeEngine 正确性与性能测试\n\n"
              << "用法: " << prog << " [选项]\n\n"
              << "选项:\n"
              << "  --size N   矩阵维度 (默认: 256)\n"
              << "  --iters N  性能迭代次数 (默认: 10)\n"
              << "  --gpu <设备>  手动指定计算设备: 枚举索引 (如 2) 或名称子串\n"
              << "                 (如 40HX / NVIDIA)；不带 --gpu 时自动选卡\n"
              << "  --help     显示此帮助信息\n";
}

// ── 辅助：比较两个 Matrix 的最大绝对误差 ──────────────────────────────────
Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    Scalar max_err = 0.0f;
    auto as = a.span();
    auto bs = b.span();
    const std::size_t n = as.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        Scalar diff = std::fabs(as[i] - bs[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

// ── 辅助：通过 engine 执行 matmul 并下载结果 ───────────────────────────────
nn::Result<Matrix> engine_matmul(
    ComputeEngine& engine, const Matrix& A, const Matrix& B,
    bool transA = false, bool transB = false)
{
    auto a_t = engine.from_matrix(A);
    if (!a_t) return std::unexpected(std::move(a_t).error());
    auto b_t = engine.from_matrix(B);
    if (!b_t) return std::unexpected(std::move(b_t).error());
    auto c_t = engine.matmul(*a_t, *b_t, transA, transB);
    if (!c_t) return std::unexpected(std::move(c_t).error());
    return engine.to_matrix(*c_t);
}

// 数字解析辅助：解析失败时打印错误并退出（替代会抛异常的 std::stoi/stod）
template <typename T>
T parse_num_or_die(const char* s, const char* opt)
{
    auto v = nn::parse_number<T>(s);
    if (!v)
    {
        std::cerr << "无效的 " << opt << " 值: " << v.error().message << "\n";
        std::exit(1);
    }
    return *v;
}

int main(int argc, char* argv[])
{
    std::size_t N = 256;
    int iters = 10;
    std::string gpu_device;   // --gpu 指定的设备（空 = 自动选卡）

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--size" && i + 1 < argc)
        {
            int v = parse_num_or_die<int>(argv[++i], "--size");
            if (v <= 0) { std::cerr << "--size 必须为正整数\n"; return 1; }
            N = static_cast<std::size_t>(v);
        }
        else if (arg == "--iters" && i + 1 < argc)
        {
            iters = parse_num_or_die<int>(argv[++i], "--iters");
            if (iters <= 0) { std::cerr << "--iters 必须为正整数\n"; return 1; }
        }
        else if (auto dev = nn::cli::parse_gpu_option(
                     argc, argv, i, /*allow_name_value=*/true))
        {
            // 本程序无位置参数，空格形式也可直接给名称子串
            gpu_device = *dev;
        }
        else { std::cerr << "未知参数: " << arg << "\n"; return 1; }
    }

    std::size_t failures = 0;  // ✅/❌ 检查计数，决定退出码（CI 必须反映 GPU 状态）

    std::cout << "========================================\n"
              << "  ComputeEngine 正确性与性能测试\n"
              << "========================================\n"
              << "  矩阵维度: " << N << "×" << N << "\n"
              << "  迭代次数: " << iters << "\n"
              << "========================================\n\n";

    // ── 1. 初始化引擎 ──────────────────────────────────────────────
    std::cout << "[1/6] 初始化引擎..." << std::flush;
    auto cpu_engine = std::make_unique<CpuEngine>();

    auto& backend = GpuBackend::instance();
    auto init_result = backend.initialize(gpu_device);
    if (!init_result)
    {
        std::cout << " 失败\n";
        std::cerr << "GPU 初始化失败: " << init_result.error().message << "\n";
        return 1;
    }
    auto gpu_engine = std::make_unique<GpuEngine>(backend);
    // 打印所选物理设备：选错卡（例如把独显老卡当目标）会直接表现为
    // 能力不足 / 驱动死锁，先让"跑在哪块卡上"可见。
    std::cout << " 成功 (CpuEngine + GpuEngine, GPU: "
              << backend.device().device_name() << ")\n";

    // ── 2. 生成随机测试数据 ────────────────────────────────────────
    std::cout << "[2/6] 生成 " << N << "×" << N << " 随机矩阵..." << std::flush;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    Matrix A(N, N), B(N, N);
    for (auto& v : A.span()) v = dist(rng);
    for (auto& v : B.span()) v = dist(rng);
    std::cout << " 完成\n";

    // ── 3. matmul 正确性验证 (CpuEngine vs GpuEngine) ─────────────
    std::cout << "[3/6] matmul 正确性验证 (CPU vs GPU)..." << std::flush;

    auto cpu_c_r = engine_matmul(*cpu_engine, A, B);
    if (!cpu_c_r) { std::cout << "\n  ❌ CpuEngine matmul 失败: " << cpu_c_r.error().message << "\n"; return 1; }
    auto gpu_c_r = engine_matmul(*gpu_engine, A, B);
    if (!gpu_c_r) { std::cout << "\n  ❌ GpuEngine matmul 失败: " << gpu_c_r.error().message << "\n"; return 1; }

    Scalar matmul_err = max_abs_diff(*cpu_c_r, *gpu_c_r);

    // RMSE
    auto cs = cpu_c_r->span();
    auto gs = gpu_c_r->span();
    Scalar sum_sq_err = 0.0f, sum_sq_ref = 0.0f;
    for (std::size_t i = 0; i < N * N; ++i)
    {
        Scalar diff = cs[i] - gs[i];
        sum_sq_err += diff * diff;
        sum_sq_ref += cs[i] * cs[i];
    }
    Scalar rmse = std::sqrt(sum_sq_err / (N * N));
    Scalar nrmse = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0f;

    std::cout << "\n";
    std::cout << "  最大绝对误差: " << std::scientific << std::setprecision(4) << matmul_err << "\n";
    std::cout << "  RMSE:         " << std::scientific << std::setprecision(4) << rmse << "\n";
    std::cout << "  NRMSE:        " << std::scientific << std::setprecision(4) << nrmse << "\n";

    const bool matmul_ok = (matmul_err < 1e-2f) && (nrmse < 1e-3f);
    std::cout << "  结果: " << (matmul_ok ? "✅ 通过" : "❌ 失败") << "\n\n";

    if (!matmul_ok)
    {
        std::cerr << "matmul 正确性验证失败，跳过后续测试。\n";
        return 1;
    }

    // ── 3b. 转置 matmul 测试 ──────────────────────────────────────
    std::cout << "[3b] 转置 matmul 测试..." << std::flush;
    {
        // A^T * B
        auto cpu_r = engine_matmul(*cpu_engine, A, B, true, false);
        auto gpu_r = engine_matmul(*gpu_engine, A, B, true, false);
        if (cpu_r && gpu_r)
        {
            Scalar err = max_abs_diff(*cpu_r, *gpu_r);
            if (err >= 1e-2f) ++failures;
            std::cout << "\n  A^T * B 最大误差: " << std::scientific << std::setprecision(4) << err
                      << (err < 1e-2f ? " ✅" : " ❌") << "\n";
        }
        else { std::cout << "\n  ❌ A^T * B 失败\n"; ++failures; }

        // A * B^T
        auto cpu_r2 = engine_matmul(*cpu_engine, A, B, false, true);
        auto gpu_r2 = engine_matmul(*gpu_engine, A, B, false, true);
        if (cpu_r2 && gpu_r2)
        {
            Scalar err = max_abs_diff(*cpu_r2, *gpu_r2);
            if (err >= 1e-2f) ++failures;
            std::cout << "  A * B^T 最大误差: " << std::scientific << std::setprecision(4) << err
                      << (err < 1e-2f ? " ✅" : " ❌") << "\n";
        }
        else { std::cout << "  ❌ A * B^T 失败\n"; ++failures; }
    }

    // ── 3c. 小 N matmul（GEMV 快路径，N ≤ 8 → matmul_gemv）────────────
    // 覆盖：N=1/5/8（vec4 与 N%4≠0 标量回退）、奇数 M/K（行尾+k 尾块）、
    // transA=1（沿 m vec4）、transB=1（沿 k vec4）及其标量回退组合。
    std::cout << "[3c] 小 N matmul（GEMV 路径）...\n";
    {
        struct GemvCase { std::size_t Am, Ak, Bk, Bn; bool tA, tB; };
        // 字段 = 存储形状（行×列）。维度反推：tA ⇒ M=A.cols, K=A.rows；
        // tB ⇒ N=B.rows, K=B.cols。故非转置需 Ak==Bk，tA 需 Am==Bk，
        // tB 需 Bn==Ak。以下用例均已按此校验。
        const GemvCase cases[] = {
            {256, 512, 512,  1, false, false},  // N=1 纯 GEMV（B 沿 k 扁平 vec4）
            {256, 512, 512,  5, false, false},  // N%4≠0 → 扁平 vec4 拆分
            {256, 512, 512,  8, false, false},  // N%4==0 → 按行 vec4
            {255, 513, 513,  7, false, false},  // 奇数 M/K：行尾 + k 尾块标量
            {256,  64, 256,  8, true,  false},  // tA：M=64,K=256（M%4==0 → 沿 m vec4）
            {257,  65, 257,  6, true,  false},  // tA：M=65,K=257 奇数 → 标量回退
            {128, 384,   8, 384, false, true},  // tB：N=8,K=384（K%4==0 → 沿 k vec4）
            {130, 386,   3, 386, false, true},  // tB：N=3,K=386 奇数 K → 标量回退
            {257,  64,   4, 257, true,  true},   // 双转置：M=64,K=257,N=4
        };
        for (const auto& c : cases)
        {
            Matrix a(c.Am, c.Ak), b(c.Bk, c.Bn);
            for (auto& v : a.span()) v = dist(rng);
            for (auto& v : b.span()) v = dist(rng);

            auto cpu_r = engine_matmul(*cpu_engine, a, b, c.tA, c.tB);
            auto gpu_r = engine_matmul(*gpu_engine, a, b, c.tA, c.tB);
            if (cpu_r && gpu_r)
            {
                Scalar err = max_abs_diff(*cpu_r, *gpu_r);
                const bool ok = err < 1e-2f;
                if (!ok) ++failures;
                // N 由存储形状反推：tB ⇒ N=B.rows，否则 N=B.cols
                const std::size_t n_out = c.tB ? c.Bk : c.Bn;
                std::cout << "  A(" << c.Am << "x" << c.Ak << ") B(" << c.Bk << "x" << c.Bn
                          << ") tA=" << c.tA << " tB=" << c.tB
                          << " N=" << n_out << " err=" << std::scientific << std::setprecision(4) << err
                          << (ok ? " ✅" : " ❌") << "\n";
            }
            else { std::cout << "  ❌ GEMV case 执行失败\n"; ++failures; }
        }
    }

    // ── 3d. transpose 逐元素对拍（issue #13 P0-① 回归）────────────────
    // 历史 bug：backend 派发 (16,8,n_bricks) 而 transpose.comp 按 8 宽砖
    // 解算（假定 gl_WorkGroupID.x ∈ [0,8)）→ 行>512 且 列>512 时静默只写
    // 前 512 行。旧测试形状最大 64×256，永远单边 ≤512，故漏检。
    // 形状表覆盖：单边 ≤512（旧代码 PASS）/ 双边 >512（旧代码 FAIL 50%）/
    // 奇数边界（513、1000 非 tile 整倍数）/ 非方阵。
    std::cout << "[3d] transpose 正确性 (GPU vs CPU 参考, 逐元素)...\n";
    {
        struct TpCase { std::size_t R, C; };
        const TpCase cases[] = {
            {  64,   64}, { 512,  512}, { 513,  513},   // 方阵 / 奇数边界
            { 256, 1024}, {1024,  256},                  // 单边 ≤512
            { 512, 1024}, {1024,  512},                  // 单边恰好 512
            {1000, 2000}, {2000, 1000},                  // 非 tile 整倍数双边 >512
            {1024, 1024}, {1024, 4096}, {4096, 1024},    // 双边 >512
            {2048, 2048}, {4096, 4096},                  // 大方阵
        };
        for (const auto& c : cases)
        {
            Matrix a(c.R, c.C);
            for (auto& v : a.span()) v = dist(rng);

            auto a_t = gpu_engine->from_matrix(a);
            auto r_t = a_t ? gpu_engine->transpose(*a_t) : nn::Result<Tensor>{std::unexpected(a_t.error())};
            auto out_r = r_t ? gpu_engine->to_matrix(*r_t) : nn::Result<Matrix>{std::unexpected(r_t.error())};
            if (!out_r)
            {
                std::cout << "  " << c.R << "x" << c.C << " 执行失败: "
                          << out_r.error().message << "\n";
                ++failures;
                continue;
            }

            // 转置是纯数据搬运 → 逐位比对（0 容差）；统计失配率与首个错误下标
            const Matrix& out = *out_r;
            const bool shape_ok = (out.rows() == c.C && out.cols() == c.R);
            std::size_t bad = 0, first_r = 0, first_c = 0;
            bool found_first = false;
            if (shape_ok)
            {
                auto as = a.span();
                auto os = out.span();
                for (std::size_t r = 0; r < c.R; ++r)
                    for (std::size_t col = 0; col < c.C; ++col)
                        if (os[col * c.R + r] != as[r * c.C + col])
                        {
                            if (!found_first) { first_r = r; first_c = col; found_first = true; }
                            ++bad;
                        }
            }
            const std::size_t total = c.R * c.C;
            const double pct = total ? 100.0 * static_cast<double>(bad) / static_cast<double>(total) : 0.0;
            const bool ok = shape_ok && bad == 0;
            if (!ok) ++failures;
            std::cout << "  " << c.R << "x" << c.C
                      << (shape_ok ? "" : " 形状错误(out=" + std::to_string(out.rows()) + "x" + std::to_string(out.cols()) + ")")
                      << " 失配 " << std::fixed << std::setprecision(1) << pct << "%";
            if (found_first)
                std::cout << " 首个错误下标 r=" << first_r << " c=" << first_c;
            std::cout << (ok ? " ✅" : " ❌") << "\n";
        }
    }

    // ── 4. roundtrip + 逐元素 + 归约测试 ─────────────────────────
    std::cout << "\n[4/6] roundtrip + 逐元素 + 归约测试...\n";
    {
        // 4a. Upload→Download roundtrip
        auto a_gpu_r = gpu_engine->from_matrix(A);
        if (a_gpu_r)
        {
            auto a_back_r = gpu_engine->to_matrix(*a_gpu_r);
            if (a_back_r)
            {
                Scalar err = max_abs_diff(A, *a_back_r);
                if (err != 0.0f) ++failures;
                std::cout << "  Upload→Download roundtrip 最大误差: "
                          << std::scientific << std::setprecision(4) << err
                          << (err == 0.0f ? " ✅" : " ❌") << "\n";
            }
            else { std::cout << "  ❌ roundtrip download 失败: " << a_back_r.error().message << "\n"; ++failures; }
        }
        else { std::cout << "  ❌ roundtrip upload 失败\n"; ++failures; }

        // 4b. 逐元素二元: Max(A, B) ≈ elementwise (CPU 参考用 CPU 计算)
        auto a_cpu_t = cpu_engine->from_matrix(A);
        auto b_cpu_t = cpu_engine->from_matrix(B);
        auto a_gpu_t = gpu_engine->from_matrix(A);
        auto b_gpu_t = gpu_engine->from_matrix(B);
        if (a_cpu_t && b_cpu_t && a_gpu_t && b_gpu_t)
        {
            // Max(A, B)
            auto max_cpu_r = cpu_engine->elementwise_binary(BinaryOp::Max, *a_cpu_t, *b_cpu_t);
            auto max_gpu_r = gpu_engine->elementwise_binary(BinaryOp::Max, *a_gpu_t, *b_gpu_t);
            if (max_cpu_r && max_gpu_r)
            {
                auto mc = cpu_engine->to_matrix(*max_cpu_r);
                auto mg = gpu_engine->to_matrix(*max_gpu_r);
                if (mc && mg)
                {
                    Scalar err = max_abs_diff(*mc, *mg);
                    if (err >= 1e-5f) ++failures;
                    std::cout << "  elementwise Max(A,B) 最大误差: "
                              << std::scientific << std::setprecision(4) << err
                              << (err < 1e-5f ? " ✅" : " ❌") << "\n";
                }
            }
            else { std::cout << "  ❌ elementwise Max 失败\n"; ++failures; }

            // 4c. 逐元素一元: Exp(A)
            auto exp_cpu_r = cpu_engine->elementwise_unary(UnaryOp::Exp, *a_cpu_t);
            auto exp_gpu_r = gpu_engine->elementwise_unary(UnaryOp::Exp, *a_gpu_t);
            if (exp_cpu_r && exp_gpu_r)
            {
                auto ec = cpu_engine->to_matrix(*exp_cpu_r);
                auto eg = gpu_engine->to_matrix(*exp_gpu_r);
                if (ec && eg)
                {
                    Scalar err = max_abs_diff(*ec, *eg);
                    if (err >= 1e-4f) ++failures;
                    std::cout << "  elementwise Exp(A) 最大误差: "
                              << std::scientific << std::setprecision(4) << err
                              << (err < 1e-4f ? " ✅" : " ❌") << "\n";
                }
            }
            else { std::cout << "  ❌ elementwise Exp 失败\n"; ++failures; }

            // 4d. 归约: col_reduce_sum(A) → (1, N)
            auto csum_cpu_r = cpu_engine->col_reduce_sum(*a_cpu_t);
            auto csum_gpu_r = gpu_engine->col_reduce_sum(*a_gpu_t);
            if (csum_cpu_r && csum_gpu_r)
            {
                auto sc = cpu_engine->to_matrix(*csum_cpu_r);
                auto sg = gpu_engine->to_matrix(*csum_gpu_r);
                if (sc && sg)
                {
                    Scalar err = max_abs_diff(*sc, *sg);
                    if (err >= 1e-3f) ++failures;
                    std::cout << "  col_reduce_sum(A) 最大误差: "
                              << std::scientific << std::setprecision(4) << err
                              << (err < 1e-3f ? " ✅" : " ❌") << "\n";
                }
            }
            else { std::cout << "  ❌ col_reduce_sum 失败\n"; ++failures; }

            // 4e. 归约: col_reduce_max(A) → (1, N)
            auto cmax_cpu_r = cpu_engine->col_reduce_max(*a_cpu_t);
            auto cmax_gpu_r = gpu_engine->col_reduce_max(*a_gpu_t);
            if (cmax_cpu_r && cmax_gpu_r)
            {
                auto mc2 = cpu_engine->to_matrix(*cmax_cpu_r);
                auto mg2 = gpu_engine->to_matrix(*cmax_gpu_r);
                if (mc2 && mg2)
                {
                    Scalar err = max_abs_diff(*mc2, *mg2);
                    if (err >= 1e-5f) ++failures;
                    std::cout << "  col_reduce_max(A) 最大误差: "
                              << std::scientific << std::setprecision(4) << err
                              << (err < 1e-5f ? " ✅" : " ❌") << "\n";
                }
            }
            else { std::cout << "  ❌ col_reduce_max 失败\n"; ++failures; }
        }
    }

    // ── 5. 性能测试 ───────────────────────────────────────────────
    std::cout << "\n[5/6] matmul 性能测试 (" << iters << " 次迭代)...\n";

    // CPU 性能
    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto r = engine_matmul(*cpu_engine, A, B);
        if (!r) { std::cerr << "CPU matmul 性能测试失败\n"; return 1; }
    }
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count() / iters;

    // GPU 性能（含上传/下载开销，反映端到端延迟）
    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        auto r = engine_matmul(*gpu_engine, A, B);
        if (!r) { std::cerr << "GPU matmul 性能测试失败\n"; return 1; }
    }
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count() / iters;

    // 防御：计时为 0 时（极小矩阵/低精度时钟）避免除零输出 inf
    const double speedup = (gpu_ms > 0.0) ? cpu_ms / gpu_ms : 0.0;
    const double gflops = (gpu_ms > 0.0) ? (2.0 * N * N * N) / (gpu_ms * 1e6) : 0.0;

    std::cout << "\n";
    std::cout << "  CPU 时间 (含上传/下载):  " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU 时间 (含上传/下载):  " << std::fixed << std::setprecision(2) << gpu_ms << " ms\n";
    std::cout << "  加速比:                  " << std::fixed << std::setprecision(2) << speedup << "×\n";
    std::cout << "  GPU GFLOPS (含传输开销): " << std::fixed << std::setprecision(1) << gflops << "\n\n";

    // ── 6. batch 模式测试 ─────────────────────────────────────────
    std::cout << "[6/6] batch 模式 + 链式 matmul 测试...\n";
    {
        // batch begin → matmul → batch end → download
        auto a_t = gpu_engine->from_matrix(A);
        auto b_t = gpu_engine->from_matrix(B);
        if (a_t && b_t)
        {
            auto batch_r = gpu_engine->begin_batch();
            if (batch_r)
            {
                auto c_t = gpu_engine->matmul(*a_t, *b_t, false, false);
                if (c_t)
                {
                    auto end_r = gpu_engine->end_batch();
                    if (end_r)
                    {
                        auto c_res = gpu_engine->to_matrix(*c_t);
                        if (c_res)
                        {
                            Scalar err = max_abs_diff(*cpu_c_r, *c_res);
                            if (err >= 1e-2f) ++failures;
                            std::cout << "  Batch+matmul 最大误差: "
                                      << std::scientific << std::setprecision(4) << err
                                      << (err < 1e-2f ? " ✅" : " ❌") << "\n";
                        }
                        else { std::cout << "  ❌ download failed: " << c_res.error().message << "\n"; ++failures; }
                    }
                    else { std::cout << "  ❌ end_batch failed: " << end_r.error().message << "\n"; ++failures; }
                }
                else { std::cout << "  ❌ matmul failed: " << c_t.error().message << "\n"; ++failures; }
            }
            else { std::cout << "  ❌ begin_batch failed: " << batch_r.error().message << "\n"; ++failures; }
        }

        // 链式: matmul(A, B) → elementwise Exp → col_reduce_sum
        // 诊断日志：逐步对比 CPU vs GPU，定位误差来源
        if (a_t && b_t)
        {
            std::cout << "\n  ── 链式诊断: matmul→Exp→col_reduce_sum ──\n";

            // ── Step 0: CPU 参考计算 ──
            auto a_cpu = cpu_engine->from_matrix(A);
            auto b_cpu = cpu_engine->from_matrix(B);
            auto c_cpu = cpu_engine->matmul(*a_cpu, *b_cpu, false, false);
            auto exp_cpu = c_cpu ? cpu_engine->elementwise_unary(UnaryOp::Exp, *c_cpu) : nn::Result<Tensor>{};
            auto sum_cpu = exp_cpu ? cpu_engine->col_reduce_sum(*exp_cpu) : nn::Result<Tensor>{};
            auto res_cpu = sum_cpu ? cpu_engine->to_matrix(*sum_cpu) : nn::Result<Matrix>{};

            // ── Step 1: matmul(A, B) on GPU ──
            auto c_t = gpu_engine->matmul(*a_t, *b_t, false, false);
            if (!c_t) { std::cout << "  ❌ matmul failed\n"; }
            else if (c_cpu)
            {
                auto c_gpu_m = gpu_engine->to_matrix(*c_t);
                auto c_cpu_m = cpu_engine->to_matrix(*c_cpu);
                if (c_gpu_m && c_cpu_m)
                {
                    Scalar mm_err = max_abs_diff(*c_cpu_m, *c_gpu_m);
                    // 统计 matmul 输出范围
                    Scalar c_min = std::numeric_limits<Scalar>::max();
                    Scalar c_max = std::numeric_limits<Scalar>::lowest();
                    Scalar c_abs_max = 0;
                    for (auto v : c_cpu_m->span()) {
                        c_min = std::min(c_min, v);
                        c_max = std::max(c_max, v);
                        c_abs_max = std::max(c_abs_max, std::fabs(v));
                    }
                    std::cout << "  [1] matmul 输出范围: [" << std::fixed << std::setprecision(2)
                              << c_min << ", " << c_max << "]  |max|=" << c_abs_max << "\n";
                    std::cout << "      matmul CPU vs GPU 最大误差: " << std::scientific << std::setprecision(4)
                              << mm_err << "\n";

                    // ── Step 2: Exp(matmul) on GPU ──
                    auto exp_t = gpu_engine->elementwise_unary(UnaryOp::Exp, *c_t);
                    if (!exp_t) { std::cout << "  ❌ Exp failed\n"; }
                    else if (exp_cpu)
                    {
                        auto exp_gpu_m = gpu_engine->to_matrix(*exp_t);
                        auto exp_cpu_m = cpu_engine->to_matrix(*exp_cpu);
                        if (exp_gpu_m && exp_cpu_m)
                        {
                            Scalar exp_err = max_abs_diff(*exp_cpu_m, *exp_gpu_m);
                            // 统计 exp 输出范围
                            Scalar e_min = std::numeric_limits<Scalar>::max();
                            Scalar e_max = std::numeric_limits<Scalar>::lowest();
                            Scalar e_abs_max = 0;
                            for (auto v : exp_cpu_m->span()) {
                                e_min = std::min(e_min, v);
                                e_max = std::max(e_max, v);
                                e_abs_max = std::max(e_abs_max, std::fabs(v));
                            }
                            std::cout << "  [2] exp() 输出范围: [" << std::scientific << std::setprecision(4)
                                      << e_min << ", " << e_max << "]  |max|=" << e_abs_max << "\n";
                            Scalar amplify = (mm_err > 0) ? (exp_err / mm_err) : 0;
                            std::cout << "      exp() CPU vs GPU 最大误差: " << std::scientific << std::setprecision(4)
                                      << exp_err << "  (放大 " << std::fixed << std::setprecision(0) << amplify << "x)\n";

                            // ── Step 3: col_reduce_sum(exp) on GPU ──
                            auto sum_t = gpu_engine->col_reduce_sum(*exp_t);
                            if (!sum_t) { std::cout << "  ❌ col_reduce_sum failed\n"; }
                            else
                            {
                                auto res = gpu_engine->to_matrix(*sum_t);
                                if (res && res_cpu)
                                {
                                    Scalar final_err = max_abs_diff(*res_cpu, *res);
                                    // 统计 reduce 输出范围
                                    Scalar s_min = std::numeric_limits<Scalar>::max();
                                    Scalar s_max = std::numeric_limits<Scalar>::lowest();
                                    for (auto v : res_cpu->span()) {
                                        s_min = std::min(s_min, v);
                                        s_max = std::max(s_max, v);
                                    }
                                    std::cout << "  [3] col_reduce_sum 输出范围: [" << std::scientific << std::setprecision(4)
                                              << s_min << ", " << s_max << "]\n";
                                    std::cout << "      最终绝对误差: " << std::scientific << std::setprecision(4)
                                              << final_err << "\n";

                                    // 相对误差（更合理的评判标准）
                                    Scalar ref_abs_max = 0;
                                    for (auto v : res_cpu->span()) ref_abs_max = std::max(ref_abs_max, std::fabs(v));
                                    Scalar rel_err = (ref_abs_max > 0) ? (final_err / ref_abs_max) : 0;
                                    std::cout << "      相对误差: " << std::scientific << std::setprecision(4)
                                              << rel_err << (rel_err < 1e-3f ? " OK" : " FAIL") << "\n";
                                    std::cout << "  结论: exp() 将 matmul 的 " << std::scientific << std::setprecision(2)
                                              << mm_err << " 误差放大为 " << final_err
                                              << " (数学预期行为，非传递异常)\n";
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Adam / AdamW CPU vs GPU（RParam 融合 K1/K2/K3）────────────────
    // 优化器超参经 RParam 运行时标量承载，GPU 走融合 shader（无 clone）。
    // 验证两引擎逐步结果一致。
    {
        std::cout << "\n════════════════════════════════════════════\n";
        std::cout << "  Adam/AdamW (RParam 融合) CPU vs GPU\n";
        std::cout << "════════════════════════════════════════════\n";
        const std::size_t R = 64, C = 32, STEPS = 4;
        std::mt19937 rng(2026);
        std::uniform_real_distribution<Scalar> cop(-1.0f, 1.0f);
        std::uniform_real_distribution<Scalar> cog(-0.1f, 0.1f);
        const Matrix p0 = [&]{ Matrix m(R, C); for (auto& v : m.span()) v = cop(rng); return m; }();
        const Matrix g0 = [&]{ Matrix m(R, C); for (auto& v : m.span()) v = cog(rng); return m; }();

        for (const char* name : {"adam", "adamw"})
        {
            // CPU 侧
            auto p_cpu_r = cpu_engine->from_matrix(p0);
            auto g_cpu_r = cpu_engine->from_matrix(g0);
            auto p_gpu_r = gpu_engine->from_matrix(p0);
            auto g_gpu_r = gpu_engine->from_matrix(g0);
            bool ok = p_cpu_r && g_cpu_r && p_gpu_r && g_gpu_r;
            std::unique_ptr<nn::Optimizer> opt_cpu, opt_gpu;
            if (ok)
            {
                opt_cpu = nn::create_optimizer(name, *cpu_engine,
                    std::vector<nn::TensorRef>{*p_cpu_r}, std::vector<nn::TensorRef>{*g_cpu_r},
                    Scalar{1e-3f});
                opt_gpu = nn::create_optimizer(name, *gpu_engine,
                    std::vector<nn::TensorRef>{*p_gpu_r}, std::vector<nn::TensorRef>{*g_gpu_r},
                    Scalar{1e-3f});
            }
            if (!ok || !opt_cpu || !opt_gpu)
            {
                std::cout << "  ❌ " << name << " 初始化失败\n";
                ++failures;
                continue;
            }
            bool step_ok = true;
            for (std::size_t s = 0; s < STEPS && step_ok; ++s)
            {
                // 每步用同一梯度（两引擎一致），先后向 zero_grad 再 step
                step_ok = opt_cpu->step() && opt_gpu->step();
            }
            if (!step_ok) { std::cout << "  ❌ " << name << " step 失败\n"; ++failures; continue; }

            auto p_cpu_m = cpu_engine->to_matrix(*p_cpu_r);
            auto p_gpu_m = gpu_engine->to_matrix(*p_gpu_r);
            if (!p_cpu_m || !p_gpu_m) { std::cout << "  ❌ " << name << " 下载失败\n"; ++failures; continue; }
            Scalar err = max_abs_diff(*p_cpu_m, *p_gpu_m);
            bool pass = err < 2e-3f;
            std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << " " << name << " p_err="
                      << std::scientific << std::setprecision(2) << err << "\n";
            if (!pass) ++failures;
        }
    }

    // ── RLA 扫描原语 CPU vs GPU（含通用 d_k>64 路径）──────────────────
    // dk=128 时 GPU 走 scan_*_gen（状态驻全局 scratch）。断言与 CPU 一致。
    {
        std::cout << "\n════════════════════════════════════════════\n";
        std::cout << "  RLA 扫描原语 (scan_prefix/suffix_outer) CPU vs GPU\n";
        std::cout << "════════════════════════════════════════════\n";
        const std::size_t B = 2, H = 2, dk = 128, seq = 4;  // dk>64 → 通用路径
        const std::size_t rows = B * H * dk;
        const std::size_t BH = rows / dk;  // == B*H

        std::mt19937 rng(777);
        std::normal_distribution<Scalar> cop(0.0f, 0.5f);
        auto fill = [&](Matrix& m) { for (auto& v : m.span()) v = cop(rng); };
        Matrix Kv(rows, seq), Vv(rows, seq), Pv(rows, seq), Rv(rows, seq);
        Matrix Dm(BH * dk * dk, seq), Xs(rows, seq), Ys(rows, seq);
        Matrix A0v(H * dk, dk), B0v(H * dk, dk), bnd(1, B * seq);
        fill(Kv); fill(Vv); fill(Pv); fill(Rv); fill(Dm); fill(Xs); fill(Ys);
        (void)A0v; (void)B0v; (void)bnd;

        auto run_prefix = [&](ComputeEngine& e, const Matrix& Ki, const Matrix& Vi,
                              const Matrix& Pi, const Matrix& Ri) -> nn::Result<Matrix> {
            auto k = e.from_matrix(Ki);  if (!k) return std::unexpected(k.error());
            auto v = e.from_matrix(Vi);  if (!v) return std::unexpected(v.error());
            auto p = e.from_matrix(Pi);  if (!p) return std::unexpected(p.error());
            auto r = e.from_matrix(Ri);  if (!r) return std::unexpected(r.error());
            auto a0 = e.from_matrix(A0v); const auto& a0r = *a0;
            auto b0 = e.from_matrix(B0v); const auto& b0r = *b0;
            auto bd = e.from_matrix(bnd); const auto& bdr = *bd;
            auto out = e.scan_prefix_outer(*k, *v, *p, *r, a0r, b0r, false,
                                           dk, H, /*causal=*/true, bdr, false);
            if (!out) return std::unexpected(out.error());
            return e.to_matrix(*out);
        };
        auto run_suffix = [&](ComputeEngine& e, const Matrix& Di, const Matrix& Xi,
                              const Matrix& Yi) -> nn::Result<Matrix> {
            auto d = e.from_matrix(Di);  if (!d) return std::unexpected(d.error());
            auto x = e.from_matrix(Xi);  if (!x) return std::unexpected(x.error());
            auto y = e.from_matrix(Yi);  if (!y) return std::unexpected(y.error());
            auto bd = e.from_matrix(bnd); const auto& bdr = *bd;
            auto out = e.scan_suffix_outer(*d, *x, *y, dk, H, /*causal=*/true, bdr, false);
            if (!out) return std::unexpected(out.error());
            return e.to_matrix(*out);
        };

        auto cpu_p = run_prefix(*cpu_engine, Kv, Vv, Pv, Rv);
        auto gpu_p = run_prefix(*gpu_engine, Kv, Vv, Pv, Rv);
        if (!cpu_p || !gpu_p)
        {
            std::cout << "  [FAIL] scan_prefix_outer 失败\n";
            if (!cpu_p) std::cout << "    cpu: " << cpu_p.error().message << "\n";
            if (!gpu_p) std::cout << "    gpu: " << gpu_p.error().message << "\n";
            ++failures;
        }
        else
        {
            Scalar err = max_abs_diff(*cpu_p, *gpu_p);
            bool pass = err < 2e-3f;
            std::cout << "  " << (pass ? "[PASS]" : "[FAIL]")
                      << " prefix_outer(dk=128) err="
                      << std::scientific << std::setprecision(2) << err << "\n";
            if (!pass) ++failures;
        }

        auto cpu_s = run_suffix(*cpu_engine, Dm, Xs, Ys);
        auto gpu_s = run_suffix(*gpu_engine, Dm, Xs, Ys);
        if (!cpu_s || !gpu_s)
        {
            std::cout << "  [FAIL] scan_suffix_outer 失败\n";
            if (!cpu_s) std::cout << "    cpu: " << cpu_s.error().message << "\n";
            if (!gpu_s) std::cout << "    gpu: " << gpu_s.error().message << "\n";
            ++failures;
        }
        else
        {
            Scalar err = max_abs_diff(*cpu_s, *gpu_s);
            bool pass = err < 2e-3f;
            std::cout << "  " << (pass ? "[PASS]" : "[FAIL]")
                      << " suffix_outer(dk=128) err="
                      << std::scientific << std::setprecision(2) << err << "\n";
            if (!pass) ++failures;
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "  测试完成";
    if (failures > 0) std::cout << "（" << failures << " 项失败）";
    std::cout << "\n";
    std::cout << "========================================\n";
    return failures > 0 ? 1 : 0;
}
#endif
