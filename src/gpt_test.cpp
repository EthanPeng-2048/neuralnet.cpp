// ── gpt_test — GPT 模型合并测试 ─────────────────────────────────────────────
// 合并：gpt_gradcheck + gpt_checkpoint_test
// 覆盖：GPTModel 整链数值梯度检查 / 梯度检查点一致性
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_gpt_gradcheck
#include "gpt_gradcheck.cpp"
#undef main

#define main test_gpt_checkpoint
#include "gpt_checkpoint_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    int failures = 0;

    std::puts("=== gpt_gradcheck (full-chain numerical) ===");
    failures += test_gpt_gradcheck(argc, argv);

    std::puts("=== gpt_checkpoint (activation checkpointing) ===");
    failures += test_gpt_checkpoint();

    std::printf("\ngpt_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
