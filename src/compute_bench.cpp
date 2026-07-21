// ── 计算库性能基准测试（f32 GFLOPS）────────────────────────────────────
// 测试 Matrix 类在 float32 下的算力：矩阵乘法、逐元素运算、转置等。
// 用法：compute_bench [--size N] [--iters N] [--warmup N] [--help]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using nn::Matrix;
using nn::Scalar;

// ── 计时工具 ──────────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

struct TimingResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double std_ms;
};

template <typename Func>
TimingResult bench(Func&& fn, int warmup, int iters)
{
    // 预热
    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / iters;
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());

    double sq_sum = 0.0;
    for (double t : times)
        sq_sum += (t - avg) * (t - avg);
    double std_dev = std::sqrt(sq_sum / iters);

    return {avg, min_t, max_t, std_dev};
}

// ── GFLOPS 计算辅助 ───────────────────────────────────────────────────────
// 矩阵乘法 C[M×N] = A[M×K] × B[K×N]  →  2·M·N·K FLOPs
static constexpr double matmul_flops(std::size_t M, std::size_t N, std::size_t K)
{
    return 2.0 * static_cast<double>(M) * N * K;
}

// 逐元素二元运算（加/减/乘）：每元素 1 FLOP
// 逐元素一元运算（ReLU/GeLU 近似）：每元素 ~若干 FLOP
// 转置：纯内存搬运，0 算术 FLOP（仅报告 GB/s）

static constexpr double elem_flops(std::size_t n) { return static_cast<double>(n); }

// ── 打印表头 ──────────────────────────────────────────────────────────────
void print_header()
{
    std::cout << "\n"
              << "┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┐\n"
              << "│ 运算             │  规格    │  FLOPs   │ avg(ms)  │ min(ms)  │ GFLOPS   │ 带宽(GB/s)   │\n"
              << "├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤\n";
}

void print_row(const char* op, const char* spec, double flops,
               double avg_ms, double min_ms, double bytes_transferred = 0.0)
{
    double gflops = (avg_ms > 0.0) ? flops / (avg_ms * 1e6) : 0.0;
    double gbps = (min_ms > 0.0 && bytes_transferred > 0.0)
                      ? bytes_transferred / (min_ms * 1e6)
                      : 0.0;

    std::cout << "│ " << std::left << std::setw(16) << op
              << "│ " << std::right << std::setw(8) << spec
              << " │ " << std::setw(8) << std::fixed << std::setprecision(0) << flops / 1e3 << "k"
              << " │ " << std::setw(7) << std::setprecision(2) << avg_ms
              << "  │ " << std::setw(7) << min_ms
              << "  │ " << std::setw(7) << std::setprecision(1) << gflops
              << "  │ " << std::setw(11) << std::setprecision(1) << gbps
              << "  │\n";
}

void print_separator()
{
    std::cout << "├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤\n";
}

void print_footer()
{
    std::cout << "└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘\n";
}

// ── 主测试 ────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    std::size_t base_size = 512;
    int iters = 20;
    int warmup = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "计算库性能基准测试 (f32 GFLOPS)\n\n"
                      << "用法: compute_bench [选项]\n\n"
                      << "选项:\n"
                      << "  --size N    基准矩阵维度 (默认: 512)\n"
                      << "  --iters N   每项测试迭代次数 (默认: 20)\n"
                      << "  --warmup N  预热次数 (默认: 5)\n"
                      << "  --help      显示此帮助信息\n";
            return 0;
        } else if (arg == "--size" && i + 1 < argc) {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --size: " << v.error().message << "\n"; return 1; }
            base_size = *v;
        } else if (arg == "--iters" && i + 1 < argc) {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --iters: " << v.error().message << "\n"; return 1; }
            iters = *v;
        } else if (arg == "--warmup" && i + 1 < argc) {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --warmup: " << v.error().message << "\n"; return 1; }
            warmup = *v;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            return 1;
        }
    }

    std::cout << "========================================\n"
              << "  计算库 f32 性能基准测试\n"
              << "========================================\n"
              << "  基准维度:   " << base_size << "×" << base_size << "\n"
              << "  迭代次数:   " << iters << "\n"
              << "  预热次数:   " << warmup << "\n"
              << "  标量类型:   float (f32)\n"
              << "  线程数:     " << std::thread::hardware_concurrency() << "\n"
              << "  编译器优化: -O3 -ffast-math\n"
              << "========================================\n";

    // ── 随机数据生成器 ─────────────────────────────────────────────
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    auto rand_matrix = [&](std::size_t rows, std::size_t cols) {
        Matrix m(rows, cols);
        auto s = m.span();
        for (auto& v : s)
            v = dist(rng);
        return m;
    };

    print_header();

    // ─═════════════════════════════════════════════════════════════════
    //  1. 矩阵乘法（核心算力指标）
    // ─═════════════════════════════════════════════════════════════════
    {
        std::vector<std::size_t> sizes = {
            64, 128, 256, 512, 768, 1024, 1536, 2048};

        for (std::size_t N : sizes) {
            if (N > base_size * 4) break;
            auto A = rand_matrix(N, N);
            auto B = rand_matrix(N, N);
            Matrix C(N, N);

            double flops = matmul_flops(N, N, N);
            auto res = bench([&] { A.multiply_to(C, B); }, warmup, iters);
            char spec[32];
            std::snprintf(spec, sizeof(spec), "%zux%zu", N, N);
            print_row("matmul (f32)", spec, flops, res.avg_ms, res.min_ms);
        }
        print_separator();
    }

    // ── 矩阵乘法（矩形矩阵，模拟 Transformer 维度）────────────────
    {
        struct RectCase { std::size_t M, K, N; const char* label; };
        std::vector<RectCase> cases = {
            {128, 768, 768,   "128×768²"},   // 注意力投影
            {128, 768, 3072,  "128×768×3K"},  // FFN 第一层
            {128, 3072, 768,  "128×3K×768"},  // FFN 第二层
            {32,  512,  512,  "32×512²"},     // 小 batch
            {1,   768,  768,  "1×768²"},      // 单样本推理
        };
        for (auto [M, K, N, label] : cases) {
            auto A = rand_matrix(M, K);
            auto B = rand_matrix(K, N);
            Matrix C(M, N);
            double flops = matmul_flops(M, N, K);
            auto res = bench([&] { A.multiply_to(C, B); }, warmup, iters);
            print_row("matmul rect", label, flops, res.avg_ms, res.min_ms);
        }
        print_separator();
    }

    // ─═════════════════════════════════════════════════════════════════
    //  2. 转置（内存带宽瓶颈）
    // ─═════════════════════════════════════════════════════════════════
    {
        std::vector<std::size_t> sizes = {256, 512, 1024, 2048};
        for (std::size_t N : sizes) {
            if (N > base_size * 4) break;
            auto A = rand_matrix(N, N);
            Matrix T(N, N);
            double bytes = 2.0 * N * N * sizeof(Scalar);  // 读+写
            auto res = bench([&] { A.transpose_to(T); }, warmup, iters);
            char spec[32];
            std::snprintf(spec, sizeof(spec), "%zux%zu", N, N);
            print_row("transpose", spec, 0, res.avg_ms, res.min_ms, bytes);
        }
        print_separator();
    }

    // ─═════════════════════════════════════════════════════════════════
    //  3. 逐元素运算
    // ─═════════════════════════════════════════════════════════════════
    {
        std::size_t N = base_size;
        std::size_t total = N * N;
        auto A = rand_matrix(N, N);
        auto B = rand_matrix(N, N);
        Matrix C(N, N);

        // 加法
        auto res_add = bench([&] { C = A + B; }, warmup, iters);
        print_row("elem add", std::to_string(N).c_str(), elem_flops(total),
                  res_add.avg_ms, res_add.min_ms, 3.0 * total * sizeof(Scalar));

        // 减法
        auto res_sub = bench([&] { C = A - B; }, warmup, iters);
        print_row("elem sub", std::to_string(N).c_str(), elem_flops(total),
                  res_sub.avg_ms, res_sub.min_ms, 3.0 * total * sizeof(Scalar));

        // 标量乘法
        Scalar s = 2.5f;
        auto res_mul = bench([&] { C = A * s; }, warmup, iters);
        print_row("scalar mul", std::to_string(N).c_str(), elem_flops(total),
                  res_mul.avg_ms, res_mul.min_ms, 2.0 * total * sizeof(Scalar));

        // scale_inplace
        auto res_scl = bench([&] { A.scale_inplace(1.01f); }, warmup, iters);
        print_row("scale inplace", std::to_string(N).c_str(), elem_flops(total),
                  res_scl.avg_ms, res_scl.min_ms, 2.0 * total * sizeof(Scalar));

        // add_inplace
        auto res_adi = bench([&] { A.add_inplace(B); }, warmup, iters);
        print_row("add inplace", std::to_string(N).c_str(), elem_flops(total),
                  res_adi.avg_ms, res_adi.min_ms, 3.0 * total * sizeof(Scalar));

        print_separator();
    }

    // ─═════════════════════════════════════════════════════════════════
    //  4. 逐元素函数（激活函数近似）
    // ─═════════════════════════════════════════════════════════════════
    {
        std::size_t N = base_size;
        std::size_t total = N * N;
        auto A = rand_matrix(N, N);
        auto B = rand_matrix(N, N);
        Matrix C(N, N);

        // ReLU: max(0, x)  — 约 1 FLOP/element
        auto res_relu = bench([&] { C = A.apply([](Scalar v) { return std::max(Scalar{0}, v); }); },
                              warmup, iters);
        print_row("ReLU (apply)", std::to_string(N).c_str(), elem_flops(total),
                  res_relu.avg_ms, res_relu.min_ms, 2.0 * total * sizeof(Scalar));

        // GeLU 近似: x * σ(1.702x)  — 约 5 FLOPs/element
        auto res_gelu = bench([&] {
            C = A.apply([](Scalar v) {
                Scalar sig = 1.0f / (1.0f + std::exp(-1.702f * v));
                return v * sig;
            });
        }, warmup, iters);
        print_row("GeLU (apply)", std::to_string(N).c_str(), 5.0 * total,
                  res_gelu.avg_ms, res_gelu.min_ms, 2.0 * total * sizeof(Scalar));

        // 逐元素乘法: A ⊙ B  — 1 FLOP/element
        auto res_emul = bench([&] {
            C = A.binary_apply(B, [](Scalar a, Scalar b) { return a * b; });
        }, warmup, iters);
        print_row("elem multiply", std::to_string(N).c_str(), elem_flops(total),
                  res_emul.avg_ms, res_emul.min_ms, 3.0 * total * sizeof(Scalar));

        print_separator();
    }

    // ─═════════════════════════════════════════════════════════════════
    //  5. 线性层（Linear forward / backward）
    // ─═════════════════════════════════════════════════════════════════
    {
        struct LinearCase { std::size_t in, out, batch; };
        std::vector<LinearCase> cases = {
            {768, 768, 128},    // Transformer 标准隐藏层
            {768, 3072, 128},   // FFN 扩展
            {3072, 768, 128},   // FFN 压缩
            {768, 768, 32},     // 小 batch
            {256, 256, 128},    // 中等规模
        };

        for (auto [in, out, batch] : cases) {
            nn::Linear layer(in, out);
            auto input = rand_matrix(in, batch);

            // Forward: C = W * input + b → FLOPs = 2·out·in·batch + out·batch
            double fwd_flops = matmul_flops(out, batch, in) + elem_flops(out * batch);
            // 首次调用校验结果，后续 benchmark 内部重复调用假设成功
            auto first_fwd = layer.forward(input);
            if (!first_fwd) {
                std::cerr << "Linear forward 失败: " << first_fwd.error().message << "\n";
                return 1;
            }
            auto res_fwd = bench([&] { auto r = layer.forward(input); (void)r; }, warmup, iters);

            char spec[64];
            std::snprintf(spec, sizeof(spec), "W=%zux%zu b=%zu", out, in, batch);
            print_row("Linear fwd", spec, fwd_flops, res_fwd.avg_ms, res_fwd.min_ms);

            // Backward: grad_input = W^T * grad_out, grad_W += grad_out * input^T, grad_b += sum
            // FLOPs ≈ 2·out·in·batch (W^T * grad_out) + out·in·batch (outer product) + out·batch (bias)
            double bwd_flops = matmul_flops(in, batch, out) + elem_flops(out * in * batch) + elem_flops(out * batch);
            auto grad_out = rand_matrix(out, batch);
            auto first_bwd = layer.backward(grad_out);
            if (!first_bwd) {
                std::cerr << "Linear backward 失败: " << first_bwd.error().message << "\n";
                return 1;
            }
            auto res_bwd = bench([&] { auto r = layer.backward(grad_out); (void)r; }, warmup, iters);
            print_row("Linear bwd", spec, bwd_flops, res_bwd.avg_ms, res_bwd.min_ms);
        }
        print_separator();
    }

    // ─═════════════════════════════════════════════════════════════════
    //  6. 归约（Sum / Mean）
    // ─═════════════════════════════════════════════════════════════════
    {
        std::size_t N = base_size;
        std::size_t total = N * N;
        auto A = rand_matrix(N, N);

        // reduce: 求和 — 1 FLOP/element
        auto res_sum = bench([&] {
            volatile Scalar s = A.reduce(Scalar{0}, std::plus<>{},
                                         [](Scalar v) { return v; });
            (void)s;
        }, warmup, iters);
        print_row("reduce (sum)", std::to_string(N).c_str(), elem_flops(total),
                  res_sum.avg_ms, res_sum.min_ms, total * sizeof(Scalar));

        print_separator();
    }

    print_footer();

    // ── 汇总：纯矩阵乘法峰值 GFLOPS ───────────────────────────────
    std::cout << "\n  📊 峰值算力摘要 (float32 矩阵乘法):\n\n";

    {
        // 自动跑一次大矩阵以获取峰值
        std::size_t peak_N = 2048;
        auto A = rand_matrix(peak_N, peak_N);
        auto B = rand_matrix(peak_N, peak_N);
        Matrix C(peak_N, peak_N);
        double flops = matmul_flops(peak_N, peak_N, peak_N);
        auto res = bench([&] { A.multiply_to(C, B); }, warmup, iters);
        double gflops = flops / (res.min_ms * 1e6);
        std::cout << "    " << peak_N << "×" << peak_N << " matmul: "
                  << std::fixed << std::setprecision(1) << gflops << " GFLOPS"
                  << "  (avg " << std::setprecision(2) << res.avg_ms << " ms, "
                  << "min " << res.min_ms << " ms)\n\n";
    }

    // 汇报硬件信息
    std::cout << "  ℹ️  线程数: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "  ℹ️  Scalar = float (sizeof = " << sizeof(Scalar) << " bytes)\n";
    std::cout << "  ℹ️  BLOCK_SIZE = " << nn::BLOCK_SIZE << "\n";
    std::cout << "\n  ✅ 测试完成。\n";

    return 0;
}
