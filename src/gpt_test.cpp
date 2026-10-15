// ── gpt_test — GPT 模型合并测试 ─────────────────────────────────────────────
// 合并：gpt_gradcheck + gpt_checkpoint_test
// 覆盖：GPTModel 整链数值梯度检查 / 梯度检查点一致性
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>
#include <thread>

#define main test_gpt_gradcheck
#include "gpt_gradcheck.cpp"
#undef main

#define main test_gpt_checkpoint
#include "gpt_checkpoint_test.cpp"
#undef main

// 环境指纹（C6/P1-4）：偶发失败曾因"日志被覆盖"无法归因；本行把"编译器 /
// 标准库 / 构建类型 / 线程数 / FMA"一次性写进日志，使任何一次失败都自带
// 环境坐标（本测试的两半输入全部固定种子 ⇒ 同环境输出逐字节一致，
// 200 次直跑实测单一 md5）。
static void print_env_fingerprint()
{
#if defined(__clang__)
    const char* compiler = "clang " __clang_version__;
#elif defined(_MSC_VER)
    const char* compiler = "msvc";
#elif defined(__GNUC__)
    const char* compiler = "gcc " __VERSION__;
#else
    const char* compiler = "unknown";
#endif
#if defined(_LIBCPP_VERSION) && defined(__GLIBCXX__)
    const char* stdlib = "libc++(with libstdc++ headers)";
#elif defined(_LIBCPP_VERSION)
    const char* stdlib = "libc++";
#elif defined(__GLIBCXX__)
    const char* stdlib = "libstdc++";
#else
    const char* stdlib = "unknown";
#endif
#if defined(NDEBUG)
    const char* build = "Release(NDEBUG)";
#else
    const char* build = "Debug/RelWithDebInfo";
#endif
#if defined(__FMA__)
    const char* fma = "on";
#else
    const char* fma = "off";
#endif
    std::printf("[env] %s / %s / %s / hw_threads=%u / FMA=%s\n", compiler, stdlib, build,
                static_cast<unsigned>(std::thread::hardware_concurrency()), fma);
}

int main(int argc, char* argv[])
{
    int failures = 0;

    print_env_fingerprint();

    std::puts("=== gpt_gradcheck (full-chain numerical) ===");
    failures += test_gpt_gradcheck(argc, argv);

    std::puts("=== gpt_checkpoint (activation checkpointing) ===");
    failures += test_gpt_checkpoint();

    std::printf("\ngpt_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
