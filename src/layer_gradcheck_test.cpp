// ── layer_gradcheck_test — 基础层数值梯度检查（合并） ───────────────────────
// 合并：swiglu_gradcheck + rmsnorm_gradcheck + softmax_gradcheck + conv2d_gradcheck
// 覆盖：SwiGLU / RMSNorm / Softmax 的数值梯度验证 + Conv2D（独立参考实现比对）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── conv2d_gradcheck（独立参考实现比对；Conv2D 此前无任何覆盖）─────────────
#define main test_conv2d_gradcheck
#define max_abs_diff max_abs_diff_conv2d
#include "conv2d_gradcheck.cpp"
#undef max_abs_diff
#undef main

// ── swiglu_gradcheck ───────────────────────────────────────────────────────
#define main test_swiglu_gradcheck
#define dot dot_swiglu
#define approx approx_swiglu
#include "swiglu_gradcheck.cpp"
#undef approx
#undef dot
#undef main

// ── rmsnorm_gradcheck ──────────────────────────────────────────────────────
#define main test_rmsnorm_gradcheck
#define dot dot_rmsnorm
#define approx approx_rmsnorm
#include "rmsnorm_gradcheck.cpp"
#undef approx
#undef dot
#undef main

// ── softmax_gradcheck ──────────────────────────────────────────────────────
#define main test_softmax_gradcheck
#define dot dot_softmax
#define approx approx_softmax
#include "softmax_gradcheck.cpp"
#undef approx
#undef dot
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;

    std::puts("=== swiglu_gradcheck ===");
    failures += test_swiglu_gradcheck(argc, argv);

    std::puts("=== rmsnorm_gradcheck ===");
    failures += test_rmsnorm_gradcheck(argc, argv);

    std::puts("=== softmax_gradcheck ===");
    failures += test_softmax_gradcheck(argc, argv);

    std::puts("=== conv2d_gradcheck ===");
    failures += test_conv2d_gradcheck(argc, argv);

    std::printf("\nlayer_gradcheck_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
