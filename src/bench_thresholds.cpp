// ── 阈值基准测试程序 ─────────────────────────────────────────────────────
// 目标：
//   1. 测试不同元素数量下串行 vs 并行的耗时，找出 PARALLEL_THRESHOLD 最佳值
//   2. 测试不同矩阵形状下 col_reduce naive（列主序跨行访问）
//      vs blocked（行主序扫描 + 累加到列缓冲）的耗时，找出最佳切换阈值
//   3. 测试 row_reduce 串行 vs 并行切换点，验证现有阈值合理性
//
// 用法：bench_thresholds [--iters N] [--warmup N]
// 输出：表格 + 末尾给出建议阈值
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using nn::Matrix;
using nn::Scalar;
using Clock = std::chrono::high_resolution_clock;

// ── 计时工具 ──────────────────────────────────────────────────────────────
template <typename Func>
static double bench_us(Func&& fn, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) fn();
    double best = 1e18;
    for (int i = 0; i < iters; ++i)
    {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        best = std::min(best, us);
    }
    return best;
}

// ══════════════════════════════════════════════════════════════════════════
// 测试 1：PARALLEL_THRESHOLD 探测
//
// 对简单 transform 操作（y[i] = a*x[i] + b）测试不同元素数量下：
//   - 串行 for 循环
//   - nn::transform（带并行阈值检查）
//   - 强制并行（直接调用 global_thread_pool().parallel_transform）
// 找出 nn::transform 何时应该启用并行
// ══════════════════════════════════════════════════════════════════════════
static void bench_parallel_threshold(int warmup, int iters)
{
    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n"
              << "│ 测试 1: PARALLEL_THRESHOLD 探测 (y[i] = 2*x[i] + 1)            │\n"
              << "├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤\n"
              << "│ 元素数   │ 串行(μs) │ 并行(μs) │ 加速比   │ 调度(μs) │ 建议     │\n"
              << "├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n";

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    // 测试元素数（覆盖 1K ~ 4M）
    const std::vector<std::size_t> sizes = {
        1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304
    };

    std::cout << std::fixed << std::setprecision(2);

    for (std::size_t n : sizes)
    {
        std::vector<Scalar> x(n), y(n);
        for (auto& v : x) v = dist(rng);

        // 串行
        auto t_serial = bench_us([&] {
            for (std::size_t i = 0; i < n; ++i)
                y[i] = Scalar{2} * x[i] + Scalar{1};
        }, warmup, iters);

        // 并行（强制走线程池，绕过阈值检查）
        auto& pool = nn::global_thread_pool();
        auto t_parallel = bench_us([&] {
            pool.parallel_transform(x.begin(), x.end(), y.begin(),
                [](Scalar v) { return Scalar{2} * v + Scalar{1}; });
        }, warmup, iters);

        // 调度开销估算（用最小元素数 1024 测一次，作为常数）
        // 对每个 n 直接给出加速比
        double speedup = t_serial / t_parallel;
        const char* advice;
        if (speedup < 0.9)       advice = "串行优";
        else if (speedup < 1.1)  advice = "无差别";
        else                     advice = "并行优";

        std::cout << "│ " << std::setw(8) << n
                  << " │ " << std::setw(8) << t_serial
                  << " │ " << std::setw(8) << t_parallel
                  << " │ " << std::setw(7) << speedup << "x"
                  << " │ " << std::setw(8) << (t_parallel - t_serial / std::max(1.0, speedup))
                  << " │ " << std::setw(8) << advice
                  << "  │\n";
    }
    std::cout << "└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n";
}

// ══════════════════════════════════════════════════════════════════════════
// 测试 2：col_reduce naive vs blocked
//
// naive（当前实现）：按列遍历，每列内跨行 stride 访问 → cache 不友好
// blocked（分块实现）：按行主序扫描，每 L1 个元素累加到 col_acc[c]
//   → cache 友好，但需要 cols 个累加器
// ══════════════════════════════════════════════════════════════════════════
static void bench_col_reduce(int warmup, int iters)
{
    std::cout << "\n┌──────────────────────────────────────────────────────────────────────────┐\n"
              << "│ 测试 2: col_reduce naive(跨行) vs blocked(行主序扫描+列累加器)            │\n"
              << "├─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┤\n"
              << "│ 矩阵(rows×cols) │ naive(μs)│ block(μs)│ 加速比   │ winner   │ 建议     │\n"
              << "├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n";

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    // 模拟实际使用场景的矩阵形状
    struct Shape { std::size_t R, C; const char* label; };
    const std::vector<Shape> shapes = {
        // GPT / LayerNorm 场景：列数为 d_model 或 vocab_size
        {32,   128,   "32×128"},      // GPT small per-sample
        {128,  128,   "128×128"},     // GPT medium
        {1024, 128,   "1024×128"},    // GPT large batch
        {32,   1024,  "32×1K"},       // wide matrix
        {128,  1024,  "128×1K"},
        {1024, 1024,  "1K×1K"},
        {32,   10000, "32×10K"},      // logits col_reduce
        {128,  10000, "128×10K"},
        {1024, 10000, "1K×10K"},      // large logits
        {4096, 4096,  "4K×4K"},      // large square
        // 行数远大于列数（行主序分块更受益）
        {8192, 64,    "8K×64"},
        {65536, 64,   "64K×64"},
        // 列数远大于行数（naive 可能受益于 SIMD）
        {64,   8192,  "64×8K"},
        {64,   65536, "64×64K"},
    };

    std::cout << std::fixed << std::setprecision(2);

    for (auto [R, C, label] : shapes)
    {
        Matrix m(R, C);
        auto sp = m.span();
        for (auto& v : sp) v = dist(rng);

        // ---- naive 实现（当前）：每列独立跨行扫描 ----
        Matrix naive_out(1, C);
        auto naive_out_sp = naive_out.span();
        auto t_naive = bench_us([&] {
            const auto self = m.span();
            auto out = naive_out_sp;
            for (std::size_t c = 0; c < C; ++c)
            {
                Scalar acc = Scalar{0};
                for (std::size_t r = 0; r < R; ++r)
                    acc += self[r * C + c];
                out[c] = acc;
            }
        }, warmup, iters);

        // ---- blocked 实现：按行主序扫描，每 BLOCK 行累加到列累加器 ----
        // BLOCK 选择：使 cols*BLOCK*sizeof(Scalar) 装入 L1（32KB）
        // BLOCK = 32KB / (C * 4) = 8192 / C
        Matrix block_out(1, C);
        auto block_out_sp = block_out.span();
        auto t_blocked = bench_us([&] {
            const auto self = m.span();
            auto out = block_out_sp;
            // 先清零
            for (std::size_t c = 0; c < C; ++c) out[c] = Scalar{0};
            // 计算 block 大小（确保 col_acc 数组 ≤ 16KB L1）
            constexpr std::size_t L1_BUDGET = 16384;  // 16KB
            std::size_t block = std::max<std::size_t>(1, L1_BUDGET / (C * sizeof(Scalar)));
            if (block > R) block = R;
            // 按行块扫描
            for (std::size_t r0 = 0; r0 < R; r0 += block)
            {
                std::size_t r_end = std::min(r0 + block, R);
                for (std::size_t r = r0; r < r_end; ++r)
                {
                    const Scalar* row = self.data() + r * C;
                    for (std::size_t c = 0; c < C; ++c)
                        out[c] += row[c];
                }
            }
        }, warmup, iters);

        double speedup = t_naive / t_blocked;
        const char* winner = (speedup < 1.0) ? "naive" : "blocked";
        const char* advice;
        if (speedup > 1.5)      advice = "用blocked";
        else if (speedup < 0.7) advice = "用naive";
        else                    advice = "可混合";

        std::cout << "│ " << std::left << std::setw(15) << label
                  << " │ " << std::right << std::setw(8) << t_naive
                  << " │ " << std::setw(8) << t_blocked
                  << " │ " << std::setw(7) << speedup << "x"
                  << " │ " << std::setw(8) << winner
                  << " │ " << std::setw(8) << advice
                  << "  │\n";
    }
    std::cout << "└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n";
}

// ══════════════════════════════════════════════════════════════════════════
// 测试 3：col_reduce 行块并行 vs 按列并行 vs 串行 blocked
//
// 对比三种方案：
//   a) 单线程行主序 blocked（当前默认）
//   b) 按列并行（每列一个任务，跨行 stride=C 访问，cache miss 严重）
//   c) 行块并行（按行分块，每线程本地累加器 + 归并阶段）
// ══════════════════════════════════════════════════════════════════════════
static void bench_col_reduce_parallel(int warmup, int iters)
{
    std::cout << "\n┌──────────────────────────────────────────────────────────────────────────────────────────┐\n"
              << "│ 测试 3: col_reduce 单线程 blocked vs 按列并行 vs 行块并行（行主序+本地累加器+归并）        │\n"
              << "├─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤\n"
              << "│ 矩阵(rows×cols) │ 串行μs   │ 按列并μs │ 行块并μs │ 列加速   │ 行块加速 │ 建议     │\n"
              << "├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n";

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    struct Shape { std::size_t R, C; const char* label; };
    const std::vector<Shape> shapes = {
        {128,  128,   "128×128"},
        {1024, 128,   "1024×128"},
        {128,  1024,  "128×1K"},
        {1024, 1024,  "1K×1K"},
        {128,  10000, "128×10K"},
        {1024, 10000, "1K×10K"},
        {4096, 4096,  "4K×4K"},
        {8192, 64,    "8K×64"},
        {64,   8192,  "64×8K"},
        {8192, 128,   "8K×128"},
        {65536, 64,   "64K×64"},
        {4096, 512,   "4K×512"},
    };

    std::cout << std::fixed << std::setprecision(2);
    auto& pool = nn::global_thread_pool();

    for (auto [R, C, label] : shapes)
    {
        Matrix m(R, C);
        auto sp = m.span();
        for (auto& v : sp) v = dist(rng);

        Matrix out_serial(1, C), out_par_col(1, C), out_par_row(1, C);

        // (a) 串行行主序 blocked
        auto t_serial = bench_us([&] {
            const auto self = m.span();
            auto out = out_serial.span();
            for (std::size_t c = 0; c < C; ++c) out[c] = Scalar{0};
            for (std::size_t r = 0; r < R; ++r)
            {
                const Scalar* row = self.data() + r * C;
                for (std::size_t c = 0; c < C; ++c)
                    out[c] += row[c];
            }
        }, warmup, iters);

        // (b) 按列并行（cache miss 严重，作为对照）
        auto t_par_col = bench_us([&] {
            const auto self = m.span();
            auto out = out_par_col.span();
            auto col_indices = std::views::iota(std::size_t{0}, C);
            pool.parallel_for_each(col_indices.begin(), col_indices.end(),
                [self, out, R, C](std::size_t c) {
                    Scalar acc = Scalar{0};
                    for (std::size_t r = 0; r < R; ++r)
                        acc += self[r * C + c];
                    out[c] = acc;
                });
        }, warmup, iters);

        // (c) 行块并行：按行分块，每线程本地累加器 + 归并
        auto t_par_row = bench_us([&] {
            const auto self = m.span();
            auto out = out_par_row.span();

            const std::size_t n_threads = std::thread::hardware_concurrency();
            std::vector<Scalar> local_acc(n_threads * C, Scalar{0});
            const std::size_t base = R / n_threads;
            const std::size_t rem = R % n_threads;
            auto row_blocks = std::views::iota(std::size_t{0}, n_threads);
            pool.parallel_for_blocks(row_blocks.begin(), row_blocks.end(),
                [self, &local_acc, C, base, rem](std::size_t t) noexcept {
                    const std::size_t r0 = t * base + std::min(t, rem);
                    const std::size_t r_end = (t + 1) * base + std::min(t + 1, rem);
                    auto* acc = local_acc.data() + t * C;
                    for (std::size_t r = r0; r < r_end; ++r)
                    {
                        const Scalar* row = self.data() + r * C;
                        for (std::size_t c = 0; c < C; ++c)
                            acc[c] += row[c];
                    }
                });

            // 归并
            for (std::size_t c = 0; c < C; ++c)
                out[c] = local_acc[c];
            for (std::size_t t = 1; t < n_threads; ++t)
            {
                const auto* acc = local_acc.data() + t * C;
                for (std::size_t c = 0; c < C; ++c)
                    out[c] += acc[c];
            }
        }, warmup, iters);

        double sp_col = t_serial / t_par_col;
        double sp_row = t_serial / t_par_row;
        const char* advice;
        if (sp_row > 1.5)      advice = "行块并优";
        else if (sp_col > 1.5) advice = "按列并优";
        else                  advice = "串行优";

        std::cout << "│ " << std::left << std::setw(15) << label
                  << " │ " << std::right << std::setw(8) << t_serial
                  << " │ " << std::setw(8) << t_par_col
                  << " │ " << std::setw(8) << t_par_row
                  << " │ " << std::setw(7) << sp_col << "x"
                  << " │ " << std::setw(7) << sp_row << "x"
                  << " │ " << std::setw(8) << advice
                  << "  │\n";
    }
    std::cout << "└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n";
}

// ══════════════════════════════════════════════════════════════════════════
// 测试 4：Matrix::multiply_to GPU 切换点探测
// 仅当 NN_HAS_VULKAN 编译时启用
// ══════════════════════════════════════════════════════════════════════════
#ifdef NN_HAS_VULKAN
static void bench_gpu_threshold(int warmup, int iters)
{
    auto& backend = nn::GpuBackend::instance();
    if (!backend.is_initialized() && !backend.initialize())
    {
        std::cout << "\n[GPU] Vulkan 未初始化，跳过 GPU 阈值测试\n";
        return;
    }

    std::cout << "\n┌──────────────────────────────────────────────────────────────────┐\n"
              << "│ 测试 4: Matrix::multiply_to GPU 切换点 (CPU matmul vs GPU)       │\n"
              << "├─────────────┬──────────┬──────────┬──────────┬──────────┬────────┤\n"
              << "│ M×N×K       │ CPU(μs)  │ GPU(μs)  │ 加速比   │ winner   │ 建议   │\n"
              << "├─────────────┼──────────┼──────────┼──────────┼──────────┼────────┤\n";

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    auto rand_matrix = [&](std::size_t r, std::size_t c) {
        Matrix m(r, c);
        for (auto& v : m.span()) v = dist(rng);
        return m;
    };

    struct Case { std::size_t M, K, N; const char* label; };
    const std::vector<Case> cases = {
        {8,    8,    8,    "8×8×8"},
        {16,   16,   16,   "16×16×16"},
        {32,   32,   32,   "32×32×32"},
        {64,   64,   64,   "64×64×64"},
        {64,   128,  64,   "64×128×64"},
        {128,  128,  128,  "128×128×128"},
        {128,  512,  128,  "128×512×128"},
        {128,  1024, 128,  "128×1K×128"},
        {256,  256,  256,  "256×256×256"},
        {512,  512,  512,  "512×512×512"},
        {1024, 1024, 1024, "1K×1K×1K"},
    };

    std::cout << std::fixed << std::setprecision(2);
    bool old_gpu_enabled = nn::SmartPolicy::gpu_enabled;
    nn::SmartPolicy::gpu_enabled = false;  // 先测 CPU 路径

    for (auto [M, K, N, label] : cases)
    {
        auto A = rand_matrix(M, K);
        auto B = rand_matrix(K, N);
        Matrix C(M, N);

        // CPU 路径
        nn::SmartPolicy::gpu_enabled = false;
        auto t_cpu = bench_us([&] { A.multiply_to(C, B); }, warmup, iters);

        // GPU 路径（通过 matmul_direct）
        nn::SmartPolicy::gpu_enabled = true;
        auto t_gpu = bench_us([&] {
            // 直接调用 backend.matmul_direct（绕过 SmartPolicy 阈值检查）
            auto r = backend.matmul_direct(A.span(), B.span(), C.span(), M, N, K);
            (void)r;
        }, warmup, iters);

        double speedup = (t_gpu > 0) ? t_cpu / t_gpu : 0.0;
        const char* winner = (speedup > 1.0) ? "GPU" : "CPU";
        const char* advice;
        if (speedup > 1.5)       advice = "用GPU";
        else if (speedup < 0.7)  advice = "用CPU";
        else                     advice = "临界";

        std::cout << "│ " << std::left << std::setw(11) << label
                  << " │ " << std::right << std::setw(8) << t_cpu
                  << " │ " << std::setw(8) << t_gpu
                  << " │ " << std::setw(7) << speedup << "x"
                  << " │ " << std::setw(8) << winner
                  << " │ " << std::setw(6) << advice
                  << "  │\n";
    }
    std::cout << "└─────────────┴──────────┴──────────┴──────────┴──────────┴────────┘\n";
    nn::SmartPolicy::gpu_enabled = old_gpu_enabled;
}
#endif  // NN_HAS_VULKAN

// ── 主入口 ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    int iters = 30;
    int warmup = 8;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            std::cout << "阈值基准测试程序\n\n"
                      << "用法: bench_thresholds [选项]\n\n"
                      << "选项:\n"
                      << "  --iters N    每项测试迭代次数 (默认: 30)\n"
                      << "  --warmup N   预热次数 (默认: 8)\n"
                      << "  --help       显示此帮助信息\n";
            return 0;
        }
        else if (arg == "--iters" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --iters\n"; return 1; }
            iters = *v;
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --warmup\n"; return 1; }
            warmup = *v;
        }
    }

    std::cout << "========================================\n"
              << "  neuralnet.cpp 阈值基准测试\n"
              << "========================================\n"
              << "  CPU 核心数:   " << std::thread::hardware_concurrency() << "\n"
              << "  标量类型:     float (4 bytes)\n"
              << "  迭代次数:     " << iters << "\n"
              << "  预热次数:     " << warmup << "\n"
              << "========================================\n";

    bench_parallel_threshold(warmup, iters);
    bench_col_reduce(warmup, iters);
    bench_col_reduce_parallel(warmup, iters);
#ifdef NN_HAS_VULKAN
    bench_gpu_threshold(warmup, iters);
#endif

    std::cout << "\n=== 阈值建议（基于上表数据） ===\n"
              << "1. PARALLEL_THRESHOLD:\n"
              << "   取加速比首次稳定 > 1.5x 的元素数作为阈值。\n"
              << "   观察测试 1 表格，找到加速比列首次 >= 1.50 的行。\n\n"
              << "2. col_reduce blocked 阈值:\n"
              << "   观察测试 2，对加速比 > 1.5 的形状用 blocked；\n"
              << "   否则用 naive（即当 cols 极大但 rows 很小时 naive SIMD 反而更快）。\n\n"
              << "3. col_reduce 并行阈值:\n"
              << "   观察测试 3，找出并行 vs 串行 blocked 的交叉点（加速比 >= 1.5）。\n\n"
              << "4. GPU_THRESHOLD:\n"
              << "   观察测试 4，找 GPU 加速比稳定 > 1.5x 的最小 M*N*K。\n"
              << "========================================\n";
    return 0;
}
