// ── test_common.hpp — 测试公共工具（唯一副本）─────────────────────────────
//
// 背景：本仓测试用「#define main test_xxx + #include 子测试.cpp」的聚合编译
// 模式（见 expr_cpu_test.cpp / precision_test.cpp / layer_gradcheck_test.cpp）。
// 历史上每个子测试都自带一份 CHECK / make_tensor / check_close / approx /
// dot / close_to，聚合器被迫用一长串 #define 逐符号重命名来规避 ODR 冲突
// （漏一个就是重定义或静默绑错）。本头把**同构可复用**的部分收敛为一份；
// 聚合器只需重命名 main、失败计数器与各测试独有的函数名。
//
// 约定：
//   1. 失败计数器仍由各子测试自带（`int g_fail = 0;` 或 `int g_failures = 0;`），
//      聚合器照旧 `#define g_fail g_fail_xxx` 重命名——计数器是"文件私有"的，
//      这样每个子测试的 main 返回值只统计自己的失败（聚合器 += 各自返回值）。
//   2. CHECK 经 NN_TEST_COUNTER 间接引用计数器（默认 g_fail）；计数器叫
//      g_failures 的文件在 include 前加一行 `#define NN_TEST_COUNTER g_failures`。
//      （宏体在**使用点**展开，聚合器对 g_fail/g_failures 的重命名照样生效。）
//   3. 本头的函数均为 inline 单份定义；子测试**不得**再定义同名函数。
//
// 刻意不收编（2026-09 审查）：
//   · max_abs_diff —— 9 份副本语义有分叉（conv2d 版多形状守卫返回 1e9，
//     其余无守卫），统一前需逐点确认语义，另案处理；
//   · check_grad_tensor / eval_loss —— 按被测层类型特化，本就不可共享。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "neuralnet.cpp/algebra_matrix.hpp"  // Matrix / Scalar
#include "neuralnet.cpp/compute_tensor.hpp"  // Tensor

// ── 失败断言宏 ─────────────────────────────────────────────────────────────
#ifndef NN_TEST_COUNTER
#define NN_TEST_COUNTER g_fail
#endif

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg));    \
            ++NN_TEST_COUNTER;                                               \
        }                                                                    \
    } while (0)

// ── 构造确定性测试张量（base + i*step 填充）───────────────────────────────
inline nn::Tensor make_tensor(std::size_t rows, std::size_t cols,
                              float base = 0.0f, float step = 0.01f)
{
    nn::Tensor t = nn::Tensor::cpu(rows, cols);
    auto sp = t.cpu_matrix().span();
    for (std::size_t i = 0; i < sp.size(); ++i)
        sp[i] = base + static_cast<float>(i) * step;
    return t;
}

// ── 逐元素比对 Tensor（容差 1e-4 相对；失败计数进 fails = 调用方的 g_fail）──
inline void check_close(const nn::Tensor& got, const nn::Tensor& ref,
                        const char* msg, int& fails)
{
    if (got.rows() != ref.rows() || got.cols() != ref.cols())
    {
        std::printf("[FAIL] %s: shape mismatch got %zux%zu ref %zux%zu\n",
                    msg, got.rows(), got.cols(), ref.rows(), ref.cols());
        ++fails;
        return;
    }
    const auto g = got.cpu_matrix().span();
    const auto r = ref.cpu_matrix().span();
    bool ok = true;
    for (std::size_t i = 0; i < g.size(); ++i)
    {
        const float d = std::fabs(g[i] - r[i]);
        const float scale = std::max(1.0f, std::fabs(r[i]));
        if (d > 1e-4f * scale)
        {
            std::printf("[FAIL] %s: [%zu] got %.6f ref %.6f\n", msg, i, g[i], r[i]);
            ok = false;
            break;
        }
    }
    if (ok) std::printf("[ OK ] %s\n", msg);
    else    ++fails;
}

// ── 数值 vs 解析梯度的相对容差判据（gradcheck 通用）───────────────────────
inline bool approx(nn::Scalar num, nn::Scalar ana, nn::Scalar tol)
{
    return std::fabs(num - ana) <=
           tol * (nn::Scalar{1} + std::fabs(num) + std::fabs(ana));
}

// ── 矩阵点积（gradcheck 计算参考损失用）───────────────────────────────────
inline nn::Scalar dot(const nn::Matrix& a, const nn::Matrix& b)
{
    nn::Scalar s{0};
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
        s += sa[i] * sb[i];
    return s;
}

// ── 比对两矩阵（abs-or-rel 容差；打印 max_abs/max_rel 与判定）──────────────
inline bool close_to(const nn::Matrix& a, const nn::Matrix& b, nn::Scalar tol,
                     const std::string& name, std::size_t idx)
{
    NN_ASSERT(a.rows() == b.rows() && a.cols() == b.cols(),
              "close_to: shape mismatch");
    nn::Scalar max_abs = 0;
    nn::Scalar max_rel = 0;
    const auto& sa = a.span();
    const auto& sb = b.span();
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const nn::Scalar diff = std::fabs(sa[i] - sb[i]);
        if (diff > max_abs) max_abs = diff;
        const nn::Scalar denom = std::fabs(sb[i]) > 1e-30f ? std::fabs(sb[i]) : 1.0f;
        const nn::Scalar rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
    }
    const bool pass = (max_abs <= tol) || (max_rel <= tol);
    std::cout << "    [" << idx << "] " << name
              << "  max_abs=" << max_abs << "  max_rel=" << max_rel
              << (pass ? "  OK" : "  FAIL") << "\n";
    return pass;
}
