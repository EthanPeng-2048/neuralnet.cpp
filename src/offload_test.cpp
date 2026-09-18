// ── offload_test — Activation Offload 合并测试 ──────────────────────────────
// 合并：offload_primitive_test + gpt_offload_test
// 覆盖：offload 原语往返 / GPT 级 offload 一致性
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_offload_primitive
#include "offload_primitive_test.cpp"
#undef main

#define main test_gpt_offload
#include "gpt_offload_test.cpp"
#undef main

int main()
{
    int failures = 0;
    // 77 = 子测试因 GPU 不可用而跳过（ctest SKIP 约定），不计入失败
    const auto add = [&failures](int r) { if (r != 77) failures += r; };

    std::puts("=== offload_primitive (GPU activation roundtrip) ===");
    add(test_offload_primitive());

    std::puts("=== gpt_offload (GPT offload vs full-store) ===");
    add(test_gpt_offload());

    std::printf("\noffload_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
