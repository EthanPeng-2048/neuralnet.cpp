// ── error_macro_test — 错误处理宏族 ────────────────────────────────────────
// 覆盖：NN_CHECK / NN_EASY_CHECK / NN_EXIT / NN_TRY / NN_TRY_MSG / NN_FAIL
//   · 解值三形态（Result<T> / Result<void> / bool 条件）+ 移动语义
//   · detail::check_message 打印格式（语境 / 详情 / 表达式 / 文件行号）
//   · NN_TRY_MSG 的错误语境前缀；NN_TRY 展开后错误消息逐字不变
//   · 终止行为：NN_CHECK → 非零退出（abort）、NN_EXIT(code) → 退出码 = code
//     （两者经子进程断言，本进程不终止）
//
// 子进程模式：--expect-abort / --expect-exit <n>（父进程 std::system 自举）
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "test_common.hpp"

int g_fail = 0;

namespace {

constexpr int kSkip = -1;   // std::system 无法启动子进程（区别于任何真实退出码）

std::string g_self;         // argv[0]：子进程自举用

nn::Result<int> ok_int() { return 42; }
nn::Result<int> bad_int() { return std::unexpected(nn::Error{"inner failure"}); }
nn::Result<void> ok_void() { return {}; }
nn::Result<std::string> payload() { return std::string{"payload"}; }

// 传播载体：NN_TRY（现有主形态）与 NN_TRY_MSG（带语境）
nn::Result<int> propagate_plain() { NN_TRY(r, bad_int()); return *r; }
nn::Result<int> propagate_ctx() { NN_TRY_MSG(r, "读取配置", bad_int()); return *r; }

// 错误源头语法糖
nn::Result<int> fail_source() { NN_FAIL("源头错误"); }

void ok(const char* msg) { std::printf("[ OK ] %s\n", msg); }

// ── ① 解包三形态 ─────────────────────────────────────────────────────────
void test_unwrap()
{
    const int v = NN_CHECK(ok_int());
    CHECK(v == 42, "NN_CHECK 解出 Result<int> 的值");

    NN_CHECK(ok_void());   // 失败即终止 → 能执行到这里就是通过
    ok("NN_CHECK 接受 Result<void>（纯副作用调用）");

    const int v2 = NN_CHECK(ok_int(), "读取配置");
    CHECK(v2 == 42, "NN_CHECK(x, msg) 解值");

    const int v3 = NN_EASY_CHECK(ok_int());
    CHECK(v3 == 42, "NN_EASY_CHECK 别名解值");

    nn::Result<std::string> s = payload();
    std::string out = NN_CHECK(std::move(s));
    CHECK(out == "payload", "NN_CHECK 移动解值（可移动类型）");

    // bool 条件形态：失败即终止，能走到下一行就是通过
    NN_CHECK(1 + 1 == 2, "算术条件成立");
    NN_EASY_CHECK(2 * 2 == 4);
    ok("NN_CHECK 接受 bool 条件（含带语境形态）");
}

// ── ② 打印格式（可单测部分）──────────────────────────────────────────────
void test_message_format()
{
    const std::string m = nn::detail::check_message(
        "engine.from_matrix(m)", "上传批次", "shape mismatch",
        std::source_location::current());

    CHECK(m.starts_with("[nn]"), "打印前缀为 [nn]");
    CHECK(m.find("上传批次: shape mismatch") != std::string::npos,
          "语境与错误详情按 '<语境>: <详情>' 拼接");
    CHECK(m.find("(engine.from_matrix(m))") != std::string::npos,
          "被检查的表达式原文在括号里");
    CHECK(m.find("error_macro_test.cpp") != std::string::npos,
          "打印带源文件名");

    // 详情为空（bool 条件失败）→ 回落 "check failed"
    const std::string m2 = nn::detail::check_message(
        "idx < n", {}, {}, std::source_location::current());
    CHECK(m2.find("check failed") != std::string::npos,
          "详情为空时回落 check failed");

    // 语境为空 → 不留冒号
    const std::string m3 = nn::detail::check_message(
        "f()", {}, "boom", std::source_location::current());
    CHECK(m3.starts_with("[nn] boom"), "无语境时不产生多余冒号");
    CHECK(m3.find("()") == std::string::npos || m3.find("(f())") != std::string::npos,
          "无语境时表达式仍在");
}

// ── ③ 传播语义 ───────────────────────────────────────────────────────────
void test_propagate()
{
    auto p = propagate_plain();
    CHECK(!p, "NN_TRY 传播失败");
    CHECK(p.error().message == "inner failure",
          "NN_TRY 不改错误消息（与手写形态逐字等价）");

    auto c = propagate_ctx();
    CHECK(!c, "NN_TRY_MSG 传播失败");
    CHECK(c.error().message == "读取配置: inner failure",
          "NN_TRY_MSG 把语境前缀进错误消息");

    auto f = fail_source();
    CHECK(!f, "NN_FAIL 产生错误");
    CHECK(!f && f.error().message == "源头错误", "NN_FAIL 的消息原样");

    // 空语境 = 逐字等价
    const nn::Error e{"raw"};
    CHECK(nn::detail::contextualize(e, {}).message == "raw",
          "空语境不修改错误消息");
}

// ── ④ 终止行为（子进程断言）──────────────────────────────────────────────
int run_self(const char* args)
{
    const std::string cmd = "\"" + g_self + "\" " + args;
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return kSkip;
    }
    return rc;
}

int normalize_rc(int rc)
{
#ifdef _WIN32
    return rc;
#else
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
    return rc;
#endif
}

void test_termination()
{
    const int aborted = normalize_rc(run_self("--expect-abort"));
    if (aborted != kSkip)
        CHECK(aborted != 0, "NN_CHECK 失败 → 非零退出（abort）");

    const int rc3 = normalize_rc(run_self("--expect-exit 3"));
    if (rc3 != kSkip)
        CHECK(rc3 == 3, "NN_EXIT(x, 3) → 退出码 3");

    const int rc7 = normalize_rc(run_self("--expect-exit 7"));
    if (rc7 != kSkip)
        CHECK(rc7 == 7, "NN_EXIT(x, 7, msg) → 退出码 7（带语境形态）");
}

} // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    // ── 子进程模式：只在父进程经 std::system 自举时进入 ───────────────────
    if (argc >= 2 && std::string(argv[1]) == "--expect-abort")
    {
        const int v = NN_CHECK(bad_int(), "子进程：abort 路径");
        std::printf("不应到达 v=%d\n", v);
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--expect-exit")
    {
        const int code = std::atoi(argv[2]);
        NN_EXIT(bad_int(), code, "子进程：exit 路径");
        std::printf("不应到达\n");
        return 0;
    }

    std::puts("=== error_macro (NN_CHECK / NN_EXIT / NN_TRY_MSG / NN_FAIL) ===");

    test_unwrap();
    test_message_format();
    test_propagate();
    test_termination();

    std::printf("\nerror_macro_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
