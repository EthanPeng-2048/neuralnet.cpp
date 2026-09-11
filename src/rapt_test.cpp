// ── rapt_test — RAPT (ReLU 线性注意力) 合并测试 ─────────────────────────────
// 合并：rapt_gradcheck + rapt_smoke_test
// 覆盖：数值梯度检查 / 端到端冒烟
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_rapt_gradcheck
#include "rapt_gradcheck.cpp"
#undef main

#define main test_rapt_smoke
#include "rapt_smoke_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;

    std::puts("=== rapt_gradcheck (RLA core + block) ===");
    failures += test_rapt_gradcheck(argc, argv);

    std::puts("=== rapt_smoke (build + spec roundtrip + loss) ===");
    failures += test_rapt_smoke(argc, argv);

    std::printf("\nrapt_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
