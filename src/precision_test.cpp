// ── precision_test — 多精度类型系统合并测试 ─────────────────────────────────
// 合并：precision_type_test + tensor_precision_test + f16_compute_test
// 覆盖：f16 转换往返 / round-half-to-even / 张量精度属性 / f16 计算验证
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── precision_type_test (f16 类型系统) ─────────────────────────────────────
#define main test_precision_type
#define g_failures g_fail_precision_type
#include "precision_type_test.cpp"
#undef g_failures
#undef main

// ── tensor_precision_test (variant 存储) ───────────────────────────────────
#define main test_tensor_precision
#define g_failures g_fail_tensor_precision
#include "tensor_precision_test.cpp"
#undef g_failures
#undef main

// ── f16_compute_test (f16 计算验证) ────────────────────────────────────────
#define main test_f16_compute
#define g_failures g_fail_f16_compute
#define test_error_paths test_f16_error_paths
#include "f16_compute_test.cpp"
#undef test_error_paths
#undef g_failures
#undef main

// ── precision_profile_test (PrecisionProfile 配方语义：--f16 = 全 f16) ─────
#define main test_precision_profile
#define g_failures g_fail_precision_profile
#include "precision_profile_test.cpp"
#undef g_failures
#undef main

int main()
{
    int failures = 0;

    std::puts("=== precision_type (f16 type system) ===");
    failures += test_precision_type();

    std::puts("=== tensor_precision (variant storage) ===");
    failures += test_tensor_precision();

    std::puts("=== f16_compute (f16 matmul + cast + training) ===");
    failures += test_f16_compute();

    std::puts("=== precision_profile (--f16 = 全 f16 语义) ===");
    failures += test_precision_profile();

    std::printf("\nprecision_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}