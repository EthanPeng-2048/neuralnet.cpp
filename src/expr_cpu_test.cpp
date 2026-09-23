// ── expr_cpu_test — 表达式/IR 纯 CPU 合并测试 ──────────────────────────────
// 合并：expr_dsl_test + expr_reduce_test + expr_matmul_test + expr_opt_test
// 覆盖：DSL 编译期融合 / 归约语义 / matmul IR 融合 / 优化 pass
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── expr_dsl_test ──────────────────────────────────────────────────────────
#define main test_expr_dsl
#define g_fail g_fail_dsl
#define make_tensor make_tensor_dsl
#undef CHECK
#include "expr_dsl_test.cpp"
#undef make_tensor
#undef g_fail
#undef main

// ── expr_reduce_test ───────────────────────────────────────────────────────
#define main test_expr_reduce
#define g_fail g_fail_reduce
#define make_tensor make_tensor_reduce
#define check_close check_close_reduce
#define test_validation test_validation_reduce
#undef CHECK
#include "expr_reduce_test.cpp"
#undef test_validation
#undef check_close
#undef make_tensor
#undef g_fail
#undef main

// ── expr_matmul_test ───────────────────────────────────────────────────────
#define main test_expr_matmul
#define g_fail g_fail_matmul
#define make_tensor make_tensor_matmul
#define check_close check_close_matmul
#define test_validation test_validation_matmul
#undef CHECK
#include "expr_matmul_test.cpp"
#undef test_validation
#undef check_close
#undef make_tensor
#undef g_fail
#undef main

// ── expr_opt_test ──────────────────────────────────────────────────────────
#define main test_expr_opt
#define g_fail g_fail_opt
#define make_tensor make_tensor_opt
#define test_determinism test_determinism_opt
#undef CHECK
#include "expr_opt_test.cpp"
#undef test_determinism
#undef make_tensor
#undef g_fail
#undef main

// ── expr_fold_test（P-C1 分块状态归约；独立函数，无宏重命名）───────────────
#include "expr_fold_test.cpp"

int main()
{
    int failures = 0;

    std::puts("=== expr_dsl (compile-time fusion + AOT) ===");
    failures += test_expr_dsl();

    std::puts("=== expr_reduce (reduction semantics) ===");
    failures += test_expr_reduce();

    std::puts("=== expr_matmul (matmul in IR fusion) ===");
    failures += test_expr_matmul();

    std::puts("=== expr_opt (canonicalize + CSE + regalloc) ===");
    failures += test_expr_opt();

    std::puts("=== expr_fold (P-C1 block-state reduction) ===");
    failures += test_expr_fold();

    std::printf("\nexpr_cpu_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
