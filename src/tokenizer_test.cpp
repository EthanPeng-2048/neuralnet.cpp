// ── tokenizer_test — 分词器正确性测试 ───────────────────────────────────────
// 原 tokenizer_consistency_test：BPE pre_tokenize 对拍 + 标记往返
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

#define main test_tokenizer
#include "tokenizer_consistency_test.cpp"
#undef main

int main()
{
    int failures = test_tokenizer();
    std::printf("\ntokenizer_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}