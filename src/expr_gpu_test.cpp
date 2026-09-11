// ── expr_gpu_test — 表达式/IR GPU 合并测试 ─────────────────────────────────
// 合并：expr_fuse_test + fused_gpu_test + tensor_expr_test
// 覆盖：图 IR 融合 GPU 端到端 / AOT 融合 shader 数值验证 / 表达式求值
// 注意：需要 Vulkan SDK；纯 CPU 构建返回 77（ctest SKIP）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持。\n";
    return 77;
}
#else

// ── expr_fuse_test ─────────────────────────────────────────────────────────
#define main test_expr_fuse
#define max_abs_diff max_abs_diff_fuse
#include "expr_fuse_test.cpp"
#undef max_abs_diff
#undef main

// ── fused_gpu_test ─────────────────────────────────────────────────────────
#define main test_fused_gpu
#define max_abs_diff max_abs_diff_fused
#include "fused_gpu_test.cpp"
#undef max_abs_diff
#undef main

// ── tensor_expr_test ───────────────────────────────────────────────────────
#define main test_tensor_expr
#include "tensor_expr_test.cpp"
#undef main

int main()
{
    int failures = 0;

    std::puts("=== expr_fuse (graph IR fusion GPU e2e) ===");
    int r = test_expr_fuse();
    if (r != 77) failures += r;

    std::puts("=== fused_gpu (AOT shader numerical) ===");
    r = test_fused_gpu();
    if (r != 77) failures += r;

    std::puts("=== tensor_expr (expression evaluation) ===");
    r = test_tensor_expr();
    if (r != 77) failures += r;

    std::printf("\nexpr_gpu_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}

#endif // NN_HAS_VULKAN
