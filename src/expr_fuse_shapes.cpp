// ───────────────────────────────────────────────────────────────────────────
//  expr_fuse_shapes.cpp — 构建期自登记（收集器 TU；NN_EXPR_SCAN 生效）
//
//  本文件是 expr_fuse_test 的非 MAIN TU，被 expr_fuse_test_nnfusion_collect
//  以 -DNN_EXPR_SCAN 编译（nn_enable_gpu_fusion 收集目标的全部非 MAIN 源）。
//  静态初始化期把 expr_fuse_test 的全部录制段 dry-run 一遍：
//    · 录制段：CPU 引擎的 scan 行为 = dsl 入图 + end_expr 融合分析并把
//      **复合 spec** 登记进 global_registry → 本目标专属 registry 覆盖
//      运行期融合产物（闭合世界两端一致：两端同跑 fuse_expr_graph，
//      确定性保证同 key）；
//    · 非录制单表达式：scan 分支直接登记。
//  链定义在 expr_fuse_shapes.hpp（与运行期用例共用同一组函数）。
// ───────────────────────────────────────────────────────────────────────────

#include "expr_fuse_shapes.hpp"

#ifdef NN_EXPR_SCAN
namespace
{
const int g_scan_self_registered = [] {
    nn::CpuEngine eng;
    constexpr std::size_t R = 5, C = 7, M = 3, K = 4, N = 2;
    auto x    = eng.create_tensor(R, C);
    auto g    = eng.create_tensor(R, 1);
    auto w    = eng.create_tensor(R, C);
    auto a0   = eng.create_tensor(M, K);
    auto b0   = eng.create_tensor(K, N);
    auto bias = eng.create_tensor(M, N);

    // 录制段 1：3 节点链（用例 1）
    (void)eng.begin_expr();
    {
        auto t1 = nn::test_fuse::chain3_a(eng, x, R, C);
        auto t2 = nn::test_fuse::chain3_b(eng, *t1, R, C);
        auto t3 = nn::test_fuse::chain3_c(eng, *t2, g, R, C);
        (void)t3;
    }
    (void)eng.end_expr();

    // 录制段 2：独立双分支（用例 2：融合边界 → 2 kernel）
    (void)eng.begin_expr();
    {
        auto t1 = nn::test_fuse::chain3_a(eng, x, R, C);
        auto t2 = nn::test_fuse::indep_b(eng, w, R, C);
        (void)t1;
        (void)t2;
    }
    (void)eng.end_expr();

    // 录制段 3：S5 matmul 链（用例 3）
    (void)eng.begin_expr();
    {
        auto t1 = nn::test_fuse::mm_a(eng, a0, b0, bias, M, N);
        auto t2 = nn::test_fuse::mm_b(eng, *t1, M, N);
        (void)t2;
    }
    (void)eng.end_expr();

    // 录制段 4：Adam 金刚石（用例 6，P2 多输出：m/v 写穿 + d 中间量 + p 目标传递）
    {
        auto m0 = eng.create_tensor(R, C);
        auto v0 = eng.create_tensor(R, C);
        auto gg = eng.create_tensor(R, C);
        auto pp = eng.create_tensor(R, C);
        (void)eng.begin_expr();
        auto t1 = nn::test_fuse::adam_a(eng, m0, gg, R, C);
        auto t2 = nn::test_fuse::adam_b(eng, v0, gg, R, C);
        auto t3 = nn::test_fuse::adam_c(eng, *t1, *t2, R, C);
        (void)nn::test_fuse::adam_d_into(eng, pp, *t3);
        (void)t3;
        (void)eng.end_expr();
    }

    // 非录制单表达式（用例 4/5）
    (void)nn::test_fuse::one_shot_rb2(eng, x, g, R, C);
    return 0;
}();
} // namespace
#endif
