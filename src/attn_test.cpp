// ── attn_test — Attention 合并测试 ──────────────────────────────────────────
// 合并：attn_gradcheck + attn_consistency_test + attn_w_batch_test
// 覆盖：CausalSelfAttention 梯度检查 / forward vs forward_step 一致性 /
//       多头多 batch W 表达式行号回归
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_attn_gradcheck
#include "attn_gradcheck.cpp"
#undef main

#define main test_attn_consistency
#include "attn_consistency_test.cpp"
#undef main

#define main test_attn_w_batch
#include "attn_w_batch_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;
    // 77 = 子测试因 GPU 不可用而跳过（ctest SKIP 约定），不计入失败
    const auto add = [&failures](int r) { if (r != 77) failures += r; };

    std::puts("=== attn_gradcheck (CausalSelfAttention numerical) ===");
    add(test_attn_gradcheck(argc, argv));

    std::puts("=== attn_consistency (forward vs forward_step) ===");
    add(test_attn_consistency(argc, argv));

    std::puts("=== attn_w_batch (multi-head multi-batch regression) ===");
    add(test_attn_w_batch(argc, argv));

    std::printf("\nattn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
