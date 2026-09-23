// ── doc_attn_test — 文档感知注意力合并测试 ──────────────────────────────────
// 合并：doc_attn_e2e_test
// 覆盖：端到端跨文档屏蔽不变性
// （原 doc_mask_test 已随物化掩码构建函数 build_attention_mask 一并删除——
//   掩码语义现由 fold body 表达：spec 级覆盖见 expr_fold_test 三掩码，
//   doc 端到端覆盖见下方 e2e）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_doc_attn_e2e
#include "doc_attn_e2e_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    std::puts("=== doc_attn_e2e (cross-doc invariance) ===");
    const int failures = test_doc_attn_e2e(argc, argv);
    std::printf("\ndoc_attn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
