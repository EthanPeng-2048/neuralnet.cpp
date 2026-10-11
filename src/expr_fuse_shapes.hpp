#pragma once

// ───────────────────────────────────────────────────────────────────────────
//  expr_fuse_shapes.hpp — expr_fuse_test 的链定义（单一事实源）
//
//  每条录制链的每一步一个小函数：运行期用例（expr_fuse_test.cpp）与构建期
//  自登记（expr_fuse_shapes.cpp 的 NN_EXPR_SCAN 静态初始化 dry-run）**调用
//  同一组函数**——融合结构由表达式类型决定，共用函数即保证两端逐表达式
//  一致（不会漂移）。改链 = 改这里，两端自动同步。
// ───────────────────────────────────────────────────────────────────────────

#include <cstddef>

#include <neuralnet.cpp/nn.hpp>

namespace nn::test_fuse
{

// 链 1（3 节点）：t1 = x*2；t2 = t1+3；t3 = t2*row_broadcast(g)
template <typename Eng>
[[nodiscard]] auto chain3_a(Eng& eng, const Tensor& x,
                            std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(x) * Scalar{2}, R, C); }

template <typename Eng>
[[nodiscard]] auto chain3_b(Eng& eng, const Tensor& t,
                            std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(t) + Scalar{3}, R, C); }

template <typename Eng>
[[nodiscard]] auto chain3_c(Eng& eng, const Tensor& t, const Tensor& g,
                            std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(t) * dsl::row_broadcast(g), R, C); }

// 链 2（独立分支 B）：u = w+1（分支 A 复用 chain3_a）
template <typename Eng>
[[nodiscard]] auto indep_b(Eng& eng, const Tensor& w,
                           std::size_t R, std::size_t C)
{ return dsl::compute(eng, dsl::leaf(w) + Scalar{1}, R, C); }

// 链 3（S5 matmul）：t1 = matmul(a0,b0)+bias；t2 = max(t1, 0)（relu）
template <typename Eng>
[[nodiscard]] auto mm_a(Eng& eng, const Tensor& a0, const Tensor& b0,
                        const Tensor& bias, std::size_t M, std::size_t N)
{
    return dsl::compute(eng,
        dsl::matmul(a0, b0, false, false, 1) + dsl::leaf(bias), M, N);
}

template <typename Eng>
[[nodiscard]] auto mm_b(Eng& eng, const Tensor& t,
                        std::size_t M, std::size_t N)
{ return dsl::compute(eng, dsl::max(dsl::leaf(t), Scalar{0}), M, N); }

// 单表达式（未录制回归 / vec4 路径）：out = x*row_broadcast(g)*2
template <typename Eng>
[[nodiscard]] auto one_shot_rb2(Eng& eng, const Tensor& x, const Tensor& g,
                                std::size_t R, std::size_t C)
{
    return dsl::compute(eng,
        dsl::leaf(x) * dsl::row_broadcast(g) * Scalar{2}, R, C);
}

// ── 链 4（Adam 金刚石，P2 多输出）────────────────────────────────────────
//   K1: m = 0.9·m0 + 0.1·g        （状态写穿 → m 张量）
//   K2: v = 0.99·v0 + 0.01·g²     （状态写穿 → v 张量）
//   K3: d = 2·m − 3·v             （纯中间量，依赖 K1/K2 —— 金刚石合流）
//   K4: p += d（compute_into 目标传递 → p 张量）
// 目标：fuse 成**单 kernel 4 输出**（tail=p，extras=[m, v, d]）。
template <typename Eng>
[[nodiscard]] auto adam_a(Eng& eng, const Tensor& m0, const Tensor& g,
                          std::size_t R, std::size_t C)
{
    // 超参走 RParam（同真实 adam_update_ 的 lr/β/ε 形态）——拼接器的
    // RParam 槽偏移曾漏过（GPU 锚位偏移真凶），本链即其回归锁
    return dsl::compute(eng,
        dsl::leaf(m0) * dsl::rparam(0.9f) + dsl::leaf(g) * dsl::rparam(0.1f), R, C);
}

template <typename Eng>
[[nodiscard]] auto adam_b(Eng& eng, const Tensor& v0, const Tensor& g,
                          std::size_t R, std::size_t C)
{
    return dsl::compute(eng,
        dsl::leaf(v0) * dsl::rparam(0.99f)
            + dsl::leaf(g) * dsl::leaf(g) * dsl::rparam(0.01f),
        R, C);
}

template <typename Eng>
[[nodiscard]] auto adam_c(Eng& eng, const Tensor& m, const Tensor& v,
                          std::size_t R, std::size_t C)
{
    return dsl::compute(eng,
        dsl::leaf(m) * dsl::rparam(2.0f) - dsl::leaf(v) * dsl::rparam(3.0f), R, C);
}

template <typename Eng>
[[nodiscard]] auto adam_d_into(Eng& eng, Tensor& p, const Tensor& d)
{
    return dsl::compute_into(eng, dsl::leaf(p) + dsl::leaf(d), p);
}

} // namespace nn::test_fuse
