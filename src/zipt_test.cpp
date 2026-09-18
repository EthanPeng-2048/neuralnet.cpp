// ── zipt_test — ZiPT (AttnZip) 合并测试 ─────────────────────────────────────
// 合并：zipt_gradcheck + zipt_smoke_test + zipt_consistency_test + zipt_doc_test
// 覆盖：梯度检查 / 端到端冒烟 / CPU-GPU 一致性 / 块级文档感知掩码
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_zipt_gradcheck
#include "zipt_gradcheck.cpp"
#undef main

#define main test_zipt_smoke
#include "zipt_smoke_test.cpp"
#undef main

#define main test_zipt_consistency
#include "zipt_consistency_test.cpp"
#undef main

#define main test_zipt_doc
#include "zipt_doc_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;
    // 77 = 子测试因 GPU 不可用而跳过（ctest SKIP 约定），不计入失败
    const auto add = [&failures](int r) { if (r != 77) failures += r; };

    std::puts("=== zipt_gradcheck (numerical) ===");
    add(test_zipt_gradcheck(argc, argv));

    std::puts("=== zipt_smoke (build + spec roundtrip + loss) ===");
    add(test_zipt_smoke(argc, argv));

    std::puts("=== zipt_consistency (CPU/GPU) ===");
    add(test_zipt_consistency(argc, argv));

    std::puts("=== zipt_doc (block-level doc mask) ===");
    add(test_zipt_doc());

    std::printf("\nzipt_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
