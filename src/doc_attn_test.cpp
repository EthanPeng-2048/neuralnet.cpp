// ── doc_attn_test — 文档感知注意力合并测试 ──────────────────────────────────
// 合并：doc_mask_test + doc_attn_e2e_test
// 覆盖：掩码构建单测 / 端到端跨文档屏蔽不变性
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_doc_mask
#define failures doc_mask_failures
#include "doc_mask_test.cpp"
#undef failures
#undef main

#define main test_doc_attn_e2e
#include "doc_attn_e2e_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;

    std::puts("=== doc_mask (mask construction unit test) ===");
    failures += test_doc_mask();

    std::puts("=== doc_attn_e2e (cross-doc invariance) ===");
    failures += test_doc_attn_e2e(argc, argv);

    std::printf("\ndoc_attn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
