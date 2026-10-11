// ───────────────────────────────────────────────────────────────────────────
//  expr_fuse_test.cpp — IR-C 图融合 GPU 端到端验证（2026-10-06 随 IR-C 恢复）
//
//  验证 begin_expr/end_expr 的完整闭环：录制（NVI 拦截入图）→ 融合分析
//  （fuse_expr_graph）→ 复合 kernel 写回 tail 占位（run_graph_kernel_into_）
//  → 数值与 CPU 逐节点参考一致。用例：
//   1. 3 节点链融合（x*2 → +3 → *row_broadcast(g)）+ 重复录制（P2-12 图级缓存命中）
//   2. 独立分支（融合边界 → 2 kernel，各自物化）
//   3. S5 matmul 链（matmul+bias → relu，matmul 段保留 + 尾链拼接）
//   4. 未录制路径回归（普通 AOT dispatch 不受影响）
//   5. vec4 向量化路径（cols%4==0）
//   6. 契约守卫：compute_into 在录制段内必须报错
//   7. Adam 金刚石（P2 多输出写穿，融合 vs 未融合逐位一致）
//   8. P3 跨链批量派发（`#b`）：批量 vs 逐个逐字节一致 + CPU 对照
//
//  链定义在 expr_fuse_shapes.hpp（单一事实源）；闭合世界由
//  expr_fuse_shapes.cpp 的 NN_EXPR_SCAN 静态初始化 dry-run 自登记（收集器
//  构建时把录制段的复合 spec 登记进本目标专属 registry）。
//
//  构建/运行：cmake --build build && ctest -R expr_fuse_test
//            （无 Vulkan 构建退出 77 = skip）
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "expr_fuse_shapes.hpp"

#ifndef NN_HAS_VULKAN

int main()
{
    std::printf("[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n");
    return 77;
}

#else

using nn::Scalar;

namespace
{

int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("[FAIL] %s\n", msg);                                 \
            ++g_fail;                                                        \
        }                                                                    \
    } while (0)

nn::Tensor make_cpu_tensor(nn::CpuEngine& cpu, std::size_t rows, std::size_t cols,
                           std::mt19937& rng)
{
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);
    nn::Matrix m(rows, cols);
    for (auto& v : m.span()) v = dist(rng);
    auto t = cpu.from_matrix(m);
    if (!t)
    {
        std::printf("[FAIL] from_matrix 失败\n");
        ++g_fail;
        return nn::Tensor{};
    }
    return *t;
}

nn::Tensor to_gpu(nn::GpuEngine& gpu, const nn::Tensor& t)
{
    auto r = gpu.import(t);
    if (!r)
    {
        std::printf("[FAIL] import 到 GPU 失败: %s\n", r.error().message.c_str());
        ++g_fail;
        return nn::Tensor{};
    }
    return *r;
}

nn::Matrix matrix_of(const nn::Tensor& t)
{
    if (!t.bound())
    {
        std::printf("[FAIL] to_matrix: 张量未绑定引擎\n");
        ++g_fail;
        return nn::Matrix(0, 0);
    }
    auto m = t.engine().to_matrix(t);
    if (!m)
    {
        std::printf("[FAIL] to_matrix 失败: %s\n", m.error().message.c_str());
        ++g_fail;
        return nn::Matrix(0, 0);
    }
    return std::move(*m);
}

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    Scalar m = 0.0f;
    auto as = a.span(), bs = b.span();
    if (as.size() != bs.size()) return 1e9f;
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar d = std::fabs(as[i] - bs[i]);
        if (d > m) m = d;
    }
    return m;
}

// ── 用例 1：3 节点链融合 + P2-12 图级缓存重复命中 ───────────────────────
int run_chain3(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t R = 5, C = 7;
    std::mt19937 rng(2026);
    const nn::Tensor x = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor g = make_cpu_tensor(cpu, R, 1, rng);
    // 录制段内同一次求值的输入必须同引擎（bind_check_ 硬规则）：
    // 外部输入先 import 到 GPU（GPU 常驻训练的真实形态）
    const nn::Tensor xg = to_gpu(gpu, x);
    const nn::Tensor gg = to_gpu(gpu, g);

    // CPU 参考：逐节点
    auto c1 = nn::test_fuse::chain3_a(cpu, x, R, C);
    auto c2 = nn::test_fuse::chain3_b(cpu, *c1, R, C);
    auto c3 = nn::test_fuse::chain3_c(cpu, *c2, g, R, C);
    if (!c1 || !c2 || !c3) { std::printf("[FAIL] CPU 参考求值失败\n"); return 1; }

    // GPU：begin_expr 录制 → 三表达式 → end_expr 融合执行
    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr 成功");
    auto g1 = nn::test_fuse::chain3_a(gpu, xg, R, C);
    auto g2 = nn::test_fuse::chain3_b(gpu, *g1, R, C);
    auto g3 = nn::test_fuse::chain3_c(gpu, *g2, gg, R, C);
    if (!g1) std::printf("  [diag] g1: %s\n", g1.error().message.c_str());
    if (!g2) std::printf("  [diag] g2: %s\n", g2.error().message.c_str());
    if (!g3) std::printf("  [diag] g3: %s\n", g3.error().message.c_str());
    CHECK(static_cast<bool>(g1) && static_cast<bool>(g2) && static_cast<bool>(g3),
          "录制期 dsl::compute 返回占位");
    auto eend = gpu.end_expr();
    if (!eend) std::printf("  [diag] end_expr: %s\n", eend.error().message.c_str());
    CHECK(static_cast<bool>(eend), "end_expr 融合执行成功");
    if (!g3) return 1;

    const Scalar err = max_abs_diff(matrix_of(*c3), matrix_of(*g3));
    const bool ok = err < 1e-4f;
    std::printf("[%s] 3 节点链融合（begin_expr/end_expr）  err=%.2e\n",
                ok ? "PASS" : "FAIL", static_cast<double>(err));
    if (!ok) return 1;

    // P2-12：重复录制（同结构 → 图级缓存命中），新输入仍正确
    std::mt19937 rng2(777);
    const nn::Tensor x2 = make_cpu_tensor(cpu, R, C, rng2);
    const nn::Tensor x2g = to_gpu(gpu, x2);
    auto d1 = nn::test_fuse::chain3_a(cpu, x2, R, C);
    auto d2 = nn::test_fuse::chain3_b(cpu, *d1, R, C);
    auto d3 = nn::test_fuse::chain3_c(cpu, *d2, g, R, C);
    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr(2) 成功");
    auto h1 = nn::test_fuse::chain3_a(gpu, x2g, R, C);
    auto h2 = nn::test_fuse::chain3_b(gpu, *h1, R, C);
    auto h3 = nn::test_fuse::chain3_c(gpu, *h2, gg, R, C);
    CHECK(static_cast<bool>(gpu.end_expr()), "end_expr(2) 融合执行成功");
    if (!d3 || !h3) return 1;
    const Scalar err2 = max_abs_diff(matrix_of(*d3), matrix_of(*h3));
    const bool ok2 = err2 < 1e-4f;
    std::printf("[%s] 重复录制（P2-12 图级缓存命中）  err=%.2e\n",
                ok2 ? "PASS" : "FAIL", static_cast<double>(err2));
    return (ok && ok2) ? 0 : 1;
}

// ── 用例 2：独立分支（融合边界 → 2 kernel，各自物化）────────────────────
int run_independent(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t R = 5, C = 7;
    std::mt19937 rng(11);
    const nn::Tensor x = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor w = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor xg = to_gpu(gpu, x);
    const nn::Tensor wg = to_gpu(gpu, w);

    auto c1 = nn::test_fuse::chain3_a(cpu, x, R, C);
    auto c2 = nn::test_fuse::indep_b(cpu, w, R, C);
    if (!c1 || !c2) { std::printf("[FAIL] CPU 参考求值失败\n"); return 1; }

    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr 成功");
    auto g1 = nn::test_fuse::chain3_a(gpu, xg, R, C);
    auto g2 = nn::test_fuse::indep_b(gpu, wg, R, C);
    CHECK(static_cast<bool>(gpu.end_expr()), "end_expr 融合执行成功");
    if (!g1 || !g2) return 1;

    const Scalar e1 = max_abs_diff(matrix_of(*c1), matrix_of(*g1));
    const Scalar e2 = max_abs_diff(matrix_of(*c2), matrix_of(*g2));
    const bool ok = e1 < 1e-4f && e2 < 1e-4f;
    std::printf("[%s] 独立分支各自物化（2 kernel）  err=%.2e/%.2e\n",
                ok ? "PASS" : "FAIL", static_cast<double>(e1),
                static_cast<double>(e2));
    return ok ? 0 : 1;
}

// ── 用例 3：S5 matmul 链（matmul+bias → relu）──────────────────────────
int run_matmul_chain(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t M = 3, K = 4, N = 2;
    std::mt19937 rng(31);
    const nn::Tensor a0 = make_cpu_tensor(cpu, M, K, rng);
    const nn::Tensor b0 = make_cpu_tensor(cpu, K, N, rng);
    const nn::Tensor bias = make_cpu_tensor(cpu, M, N, rng);
    const nn::Tensor a0g = to_gpu(gpu, a0);
    const nn::Tensor b0g = to_gpu(gpu, b0);
    const nn::Tensor biasg = to_gpu(gpu, bias);

    auto c1 = nn::test_fuse::mm_a(cpu, a0, b0, bias, M, N);
    auto c2 = nn::test_fuse::mm_b(cpu, *c1, M, N);
    if (!c1 || !c2) { std::printf("[FAIL] CPU 参考求值失败\n"); return 1; }

    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr 成功");
    auto g1 = nn::test_fuse::mm_a(gpu, a0g, b0g, biasg, M, N);
    auto g2 = nn::test_fuse::mm_b(gpu, *g1, M, N);
    CHECK(static_cast<bool>(gpu.end_expr()), "end_expr 融合执行成功");
    if (!g2) return 1;

    const Scalar err = max_abs_diff(matrix_of(*c2), matrix_of(*g2));
    const bool ok = err < 1e-4f;
    std::printf("[%s] S5 matmul 链融合（matmul+bias → relu）  err=%.2e\n",
                ok ? "PASS" : "FAIL", static_cast<double>(err));
    return ok ? 0 : 1;
}

// ── 用例 4/5：未录制路径回归 + vec4 向量化路径 ─────────────────────────
int run_non_recording(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t R = 4;
    int fail = 0;
    for (const std::size_t C : {std::size_t{6}, std::size_t{4},
                                std::size_t{8}, std::size_t{12}})
    {
        std::mt19937 rng(42 + static_cast<unsigned>(C));
        const nn::Tensor x = make_cpu_tensor(cpu, R, C, rng);
        const nn::Tensor g = make_cpu_tensor(cpu, R, 1, rng);
        auto cr = nn::test_fuse::one_shot_rb2(cpu, x, g, R, C);
        auto gr = nn::test_fuse::one_shot_rb2(gpu, x, g, R, C);
        if (!cr || !gr)
        {
            std::printf("[FAIL] 未录制路径求值失败（未命中 AOT shader？）: cols=%zu\n", C);
            return 1;
        }
        const Scalar err = max_abs_diff(matrix_of(*cr), matrix_of(*gr));
        const bool ok = err < 1e-4f;
        std::printf("[%s] 未录制路径 cols=%zu%s  err=%.2e\n", ok ? "PASS" : "FAIL",
                    C, (C % 4 == 0) ? "（vec4）" : "（标量）",
                    static_cast<double>(err));
        if (!ok) ++fail;
    }
    return fail == 0 ? 0 : 1;
}

// ── 用例 6：Adam 金刚石（P2 多输出：状态写穿 + 中间量 + 目标传递）────────
//   K1/K2 的 m/v 状态写穿回张量、K3 纯中间量、K4 = compute_into 目标传递
//   → 融合为单 kernel 4 输出，逐位对拍 CPU 逐节点参考。
int run_adam_diamond(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t R = 5, C = 7;
    std::mt19937 rng(55);
    const nn::Tensor m0 = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor v0 = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor g  = make_cpu_tensor(cpu, R, C, rng);
    nn::Tensor p0 = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor m0g = to_gpu(gpu, m0);
    const nn::Tensor v0g = to_gpu(gpu, v0);
    const nn::Tensor gg  = to_gpu(gpu, g);
    nn::Tensor pg = to_gpu(gpu, p0);
    nn::Tensor pu = to_gpu(gpu, p0);   // GPU 未融合参考的目标（与融合同源）

    // GPU 未融合参考（逐条 dispatch，同设备同 shader 族）——融合 vs 未融合
    // 的**逐位**对比才是融合等价性的硬验收（CPU 对比只测 CPU/GPU 舍入差）
    auto u1 = nn::test_fuse::adam_a(gpu, m0g, gg, R, C);
    auto u2 = nn::test_fuse::adam_b(gpu, v0g, gg, R, C);
    if (!u1 || !u2)
    {
        std::printf("[FAIL] GPU 未融合参考求值失败\n");
        if (!u1) std::printf("  [diag] u1: %s\n", u1.error().message.c_str());
        if (!u2) std::printf("  [diag] u2: %s\n", u2.error().message.c_str());
        return 1;
    }
    auto u3 = nn::test_fuse::adam_c(gpu, *u1, *u2, R, C);
    if (!u3)
    {
        std::printf("[FAIL] GPU 未融合参考求值失败\n");
        std::printf("  [diag] u3: %s\n", u3.error().message.c_str());
        return 1;
    }
    auto ud = nn::test_fuse::adam_d_into(gpu, pu, *u3);
    if (!ud)
    {
        std::printf("[FAIL] GPU 未融合参考求值失败\n");
        std::printf("  [diag] ud: %s\n", ud.error().message.c_str());
        return 1;
    }

    // CPU 参考：逐节点（K4 用 compute_into 原地写 p0）
    auto c1 = nn::test_fuse::adam_a(cpu, m0, g, R, C);
    auto c2 = nn::test_fuse::adam_b(cpu, v0, g, R, C);
    auto c3 = nn::test_fuse::adam_c(cpu, *c1, *c2, R, C);
    auto cd = nn::test_fuse::adam_d_into(cpu, p0, *c3);
    if (!c1 || !c2 || !c3 || !cd)
    {
        std::printf("[FAIL] CPU 金刚石参考求值失败\n");
        return 1;
    }

    // GPU：录制 → 金刚石（K3 依赖 K1/K2 合流）→ 单 kernel 4 输出
    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr 成功");
    auto g1 = nn::test_fuse::adam_a(gpu, m0g, gg, R, C);
    auto g2 = nn::test_fuse::adam_b(gpu, v0g, gg, R, C);
    auto g3 = nn::test_fuse::adam_c(gpu, *g1, *g2, R, C);
    auto gd = nn::test_fuse::adam_d_into(gpu, pg, *g3);
    CHECK(static_cast<bool>(g1) && static_cast<bool>(g2) && static_cast<bool>(g3)
              && static_cast<bool>(gd),
          "录制期金刚石求值返回占位");
    CHECK(static_cast<bool>(gpu.end_expr()), "end_expr 融合执行成功");
    if (!g1 || !g2 || !g3) return 1;

    const Scalar e1 = max_abs_diff(matrix_of(*c1), matrix_of(*g1));   // m 写穿
    const Scalar e2 = max_abs_diff(matrix_of(*c2), matrix_of(*g2));   // v 写穿
    const Scalar e3 = max_abs_diff(matrix_of(*c3), matrix_of(*g3));   // d 写穿
    const Scalar ep = max_abs_diff(matrix_of(p0), matrix_of(pg));     // p 目标传递
    const bool ok = e1 < 1e-4f && e2 < 1e-4f && e3 < 1e-4f && ep < 1e-4f;
    std::printf("[%s] Adam 金刚石单 kernel 4 输出（m/v/d 写穿 + p 目标）"
                "  err=%.2e/%.2e/%.2e/%.2e\n", ok ? "PASS" : "FAIL",
                static_cast<double>(e1), static_cast<double>(e2),
                static_cast<double>(e3), static_cast<double>(ep));
    if (!ok) return 1;

    // 融合 vs 未融合（同设备）：**必须逐位一致**（字节锚的 GPU 口径同源）
    const Scalar b1 = max_abs_diff(matrix_of(*u1), matrix_of(*g1));
    const Scalar b2 = max_abs_diff(matrix_of(*u2), matrix_of(*g2));
    const Scalar b3 = max_abs_diff(matrix_of(*u3), matrix_of(*g3));
    const Scalar bp = max_abs_diff(matrix_of(pu), matrix_of(pg));
    const bool okb = b1 == 0.0f && b2 == 0.0f && b3 == 0.0f && bp == 0.0f;
    std::printf("[%s] 融合 vs 未融合 GPU 逐位一致  diff=%.2e/%.2e/%.2e/%.2e\n",
                okb ? "PASS" : "FAIL", static_cast<double>(b1),
                static_cast<double>(b2), static_cast<double>(b3),
                static_cast<double>(bp));
    return okb ? 0 : 1;
}

// ── 用例 7：目标传递（P2）——compute_into 录制段内可用，dst 被写穿 ──────
int run_contract_guard(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    const std::size_t R = 3, C = 3;
    std::mt19937 rng(5);
    const nn::Tensor x = make_cpu_tensor(cpu, R, C, rng);
    const nn::Tensor xg = to_gpu(gpu, x);
    auto dst_r = gpu.create_tensor(R, C);
    CHECK(static_cast<bool>(gpu.begin_expr()), "begin_expr 成功");
    auto r = nn::dsl::compute_into(gpu, nn::dsl::leaf(xg) * Scalar{2}, dst_r);
    const bool ok = static_cast<bool>(r);   // P2：目标传递节点可入图
    CHECK(static_cast<bool>(gpu.end_expr()), "end_expr（目标传递节点）成功");
    std::printf("[%s] compute_into 录制段内目标传递（P2）\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;
    auto ref = nn::dsl::compute(cpu, nn::dsl::leaf(x) * Scalar{2}, R, C);
    if (!ref) return 1;
    const Scalar err = max_abs_diff(matrix_of(*ref), matrix_of(dst_r));
    const bool ok2 = err < 1e-4f;
    std::printf("[%s] 目标传递 dst 写穿 = x*2  err=%.2e\n",
                ok2 ? "PASS" : "FAIL", static_cast<double>(err));
    return ok2 ? 0 : 1;
}

// ── 进程内环境开关（NN_IRC_NO_BATCH 的 A/B 逃生阀在同进程切换）───────────
void set_env_flag(const char* name, bool on)
{
#ifdef _MSC_VER
    _putenv_s(name, on ? "1" : "");
#else
    if (on) ::setenv(name, "1", 1); else ::unsetenv(name);
#endif
}

// 逐字节比较（批量 vs 逐个必须 memcmp 全等，不是容差）
bool byte_equal(const nn::Matrix& a, const nn::Matrix& b)
{
    const auto as = a.span(), bs = b.span();
    return as.size() == bs.size() &&
           std::memcmp(as.data(), bs.data(), as.size_bytes()) == 0;
}

// ── 用例 8（P3 跨链批量派发）：批量 vs 逐个逐字节一致 ─────────────────────
// 同一录制图含三类互不依赖的节点：
//   · 3× 同签名单节点（one_shot_rb2：行广播 → row/cols 参与推导）→ 批量组 1
//   · 3× Adam 金刚石（P2 extras 多输出复合）→ 批量组 2
//   · 1 条 3 节点 P1 链（chain3_a/b/c，不同签名 → 逐 kernel 原路径）
// **两组都是同组混形状**：三个实例 (R,C) = (3,5)/(4,7)/(2,11) 各不同（count 与
// cols 都逐实例不同）仍进同一组一次派发——实例表尾槽 (count,cols) 的回归锁。
// NN_IRC_NO_BATCH=1 与默认各跑一遍：全部输出张量**逐字节相等**（memcmp）；
// CPU 逐节点参考按既有容差断言；并断言批量派发恰 2 组（否则 A/B 自比恒真）。
int run_batch_dispatch(nn::CpuEngine& cpu, nn::GpuEngine& gpu)
{
    struct Shape { std::size_t R, C; };
    const Shape sh[3] = {{3, 5}, {4, 7}, {2, 11}};
    const std::size_t CR = 3, CC = 5;   // P1 链形状
    std::mt19937 rng(20261006);

    // ── 源数据（CPU，一次性生成；p 目标每轮都要全新副本——会被原地写）──
    std::vector<nn::Tensor> xs(3), gs(3), m0(3), v0(3), gg(3), p0(3);
    for (int i = 0; i < 3; ++i)
    {
        xs[i] = make_cpu_tensor(cpu, sh[i].R, sh[i].C, rng);
        gs[i] = make_cpu_tensor(cpu, sh[i].R, 1, rng);
        m0[i] = make_cpu_tensor(cpu, sh[i].R, sh[i].C, rng);
        v0[i] = make_cpu_tensor(cpu, sh[i].R, sh[i].C, rng);
        gg[i] = make_cpu_tensor(cpu, sh[i].R, sh[i].C, rng);
        p0[i] = make_cpu_tensor(cpu, sh[i].R, sh[i].C, rng);
    }
    const nn::Tensor xc = make_cpu_tensor(cpu, CR, CC, rng);
    const nn::Tensor gc = make_cpu_tensor(cpu, CR, 1, rng);

    // ── CPU 逐节点参考（容差口径同既有用例）──────────────────────────
    std::vector<nn::Matrix> cpu_out;
    for (int i = 0; i < 3; ++i)
    {
        auto s = nn::test_fuse::one_shot_rb2(cpu, xs[i], gs[i], sh[i].R, sh[i].C);
        auto a1 = nn::test_fuse::adam_a(cpu, m0[i], gg[i], sh[i].R, sh[i].C);
        auto a2 = nn::test_fuse::adam_b(cpu, v0[i], gg[i], sh[i].R, sh[i].C);
        auto a3 = nn::test_fuse::adam_c(cpu, *a1, *a2, sh[i].R, sh[i].C);
        auto pr = cpu.from_matrix(matrix_of(p0[i]));   // p 目标的新副本
        if (!s || !a1 || !a2 || !a3 || !pr)
        {
            std::printf("[FAIL] CPU 参考求值失败\n");
            return 1;
        }
        auto ad = nn::test_fuse::adam_d_into(cpu, *pr, *a3);
        if (!ad)
        {
            std::printf("[FAIL] CPU 参考求值失败（目标传递）\n");
            return 1;
        }
        cpu_out.push_back(matrix_of(*s));
        cpu_out.push_back(matrix_of(*a1));   // m 写穿
        cpu_out.push_back(matrix_of(*a2));   // v 写穿
        cpu_out.push_back(matrix_of(*a3));   // d 写穿
        cpu_out.push_back(matrix_of(*pr));   // p 目标
    }
    {
        auto c1 = nn::test_fuse::chain3_a(cpu, xc, CR, CC);
        auto c2 = nn::test_fuse::chain3_b(cpu, *c1, CR, CC);
        auto c3 = nn::test_fuse::chain3_c(cpu, *c2, gc, CR, CC);
        if (!c1 || !c2 || !c3)
        {
            std::printf("[FAIL] CPU 参考求值失败（链）\n");
            return 1;
        }
        cpu_out.push_back(matrix_of(*c3));
    }

    // ── GPU：批量（默认）与逐个（NN_IRC_NO_BATCH=1）各跑一遍 ────────────
    auto& backend = nn::GpuBackend::instance();
    std::vector<nn::Matrix> outs[2];
    const auto run_gpu = [&](bool no_batch, std::vector<nn::Matrix>& out) -> bool
    {
        set_env_flag("NN_IRC_NO_BATCH", no_batch);
        std::vector<nn::Tensor> xg(3), gsg(3), m0g(3), v0g(3), ggg(3), pg(3);
        for (int i = 0; i < 3; ++i)
        {
            xg[i] = to_gpu(gpu, xs[i]);
            gsg[i] = to_gpu(gpu, gs[i]);
            m0g[i] = to_gpu(gpu, m0[i]);
            v0g[i] = to_gpu(gpu, v0[i]);
            ggg[i] = to_gpu(gpu, gg[i]);
            pg[i] = to_gpu(gpu, p0[i]);
        }
        const nn::Tensor xcg = to_gpu(gpu, xc);
        const nn::Tensor gcg = to_gpu(gpu, gc);

        std::vector<nn::Result<nn::Tensor>> s1(3), ma(3), va(3), da(3);
        std::vector<nn::Result<void>> pd(3);
        CHECK(static_cast<bool>(gpu.begin_expr()), "批量用例 begin_expr 成功");
        for (int i = 0; i < 3; ++i)
            s1[i] = nn::test_fuse::one_shot_rb2(gpu, xg[i], gsg[i],
                                                sh[i].R, sh[i].C);
        for (int i = 0; i < 3; ++i)
        {
            ma[i] = nn::test_fuse::adam_a(gpu, m0g[i], ggg[i], sh[i].R, sh[i].C);
            va[i] = nn::test_fuse::adam_b(gpu, v0g[i], ggg[i], sh[i].R, sh[i].C);
            da[i] = nn::test_fuse::adam_c(gpu, *ma[i], *va[i], sh[i].R, sh[i].C);
            pd[i] = nn::test_fuse::adam_d_into(gpu, pg[i], *da[i]);
        }
        auto c1 = nn::test_fuse::chain3_a(gpu, xcg, CR, CC);
        auto c2 = nn::test_fuse::chain3_b(gpu, *c1, CR, CC);
        auto c3 = nn::test_fuse::chain3_c(gpu, *c2, gcg, CR, CC);
        CHECK(static_cast<bool>(gpu.end_expr()), "批量用例 end_expr 成功");
        set_env_flag("NN_IRC_NO_BATCH", false);
        for (int i = 0; i < 3; ++i)
            if (!s1[i] || !ma[i] || !va[i] || !da[i] || !pd[i])
            {
                std::printf("[FAIL] 录制段求值失败（实例 %d）\n", i);
                return false;
            }
        if (!c1 || !c2 || !c3)
        {
            std::printf("[FAIL] 录制段求值失败（链）\n");
            return false;
        }
        out.clear();
        for (int i = 0; i < 3; ++i)
        {
            out.push_back(matrix_of(*s1[i]));
            out.push_back(matrix_of(*ma[i]));
            out.push_back(matrix_of(*va[i]));
            out.push_back(matrix_of(*da[i]));
            out.push_back(matrix_of(pg[i]));   // p 目标（compute_into 写穿）
        }
        out.push_back(matrix_of(*c3));
        return true;
    };

    const std::uint64_t b0 = backend.fused_batch_dispatches();
    if (!run_gpu(false, outs[0]))
        return 1;
    const std::uint64_t b1 = backend.fused_batch_dispatches();
    if (!run_gpu(true, outs[1]))
        return 1;

    // ① 批量 vs 逐个：逐字节相等（memcmp/哈希口径）
    bool same = outs[0].size() == outs[1].size();
    for (std::size_t i = 0; same && i < outs[0].size(); ++i)
        same = byte_equal(outs[0][i], outs[1][i]);
    std::printf("[%s] 批量 vs 逐个逐字节一致（%zu 张量，memcmp）\n",
                same ? "PASS" : "FAIL", outs[0].size());
    if (!same)
        return 1;

    // ② CPU 对照（容差口径同既有用例）
    Scalar worst = 0.0f;
    for (std::size_t i = 0; i < cpu_out.size() && i < outs[0].size(); ++i)
        worst = std::max(worst, max_abs_diff(cpu_out[i], outs[0][i]));
    const bool cpu_ok = worst < 1e-4f;
    std::printf("[%s] CPU 逐节点对照  max_err=%.2e\n",
                cpu_ok ? "PASS" : "FAIL", static_cast<double>(worst));
    if (!cpu_ok)
        return 1;

    // ③ 批量路径确实走到（否则 ① 是静默降级后的自比恒真）：两个批量组
    //    = **2 次派发**（3× 同签名单节点 + 3× 金刚石各自成组）——断言恰为 2
    //    同时是"同组混形状"回归锁：三个实例 rows/cols 各不同（(3,5)/(4,7)/
    //    (2,11)），若分组按形状碎开（或尾槽 per-instance cols 失效）会多于 2
    //    次派发或①字节比较失配。设备无 BDA 时 `#b` 不注册、合法逐 kernel
    //    降级 → 跳过该断言（A/B 一致性仍成立）。
    const std::uint64_t batched = b1 - b0;
    std::printf("[info] 批量派发次数 = %llu%s\n",
                static_cast<unsigned long long>(batched),
                backend.device().bda_available() ? "" : "（无 BDA：预期 0）");
    if (backend.device().bda_available())
        CHECK(batched == 2, "批量派发恰 2 组（同组混形状：3+3）");
    return 0;
}

} // namespace

int main()
{
    std::printf("========================================\n");
    std::printf("  IR-C 图 IR 融合 GPU 端到端验证\n");
    std::printf("========================================\n");

    auto cpu_engine = std::make_unique<nn::CpuEngine>();
    auto& backend = nn::GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::printf("GPU 初始化失败: %s\n", init_r.error().message.c_str());
        return 77;
    }
    auto gpu_engine = std::make_unique<nn::GpuEngine>(backend);
    std::printf("[init] CpuEngine + GpuEngine 就绪\n");

    int fail = 0;
    fail += run_chain3(*cpu_engine, *gpu_engine);
    fail += run_independent(*cpu_engine, *gpu_engine);
    fail += run_matmul_chain(*cpu_engine, *gpu_engine);
    fail += run_non_recording(*cpu_engine, *gpu_engine);
    fail += run_adam_diamond(*cpu_engine, *gpu_engine);
    fail += run_batch_dispatch(*cpu_engine, *gpu_engine);
    fail += run_contract_guard(*cpu_engine, *gpu_engine);

    if (fail == 0 && g_fail == 0)
    {
        std::printf("expr_fuse_test: ALL PASS\n");
        return 0;
    }
    std::printf("expr_fuse_test: FAILED\n");
    return 1;
}

#endif   // NN_HAS_VULKAN
