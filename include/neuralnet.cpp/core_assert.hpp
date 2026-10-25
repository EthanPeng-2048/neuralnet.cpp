#pragma once

#include <cstdio>
#include <string>
#include <expected>
#include <neuralnet.cpp/core_errors.hpp>

namespace nn {

namespace detail {

// ── 无条件 fail-fast 的公共出口（NN_ASSERT_ALWAYS / NN_FATAL 的实现支撑）────
// 打印 "<tag> failed at <file>:<line>: <msg>" 后经 fail_abort 终止
// （Windows 下 fail_abort 已把 CRT 的 abort 弹窗/ WER 上报静音，见 core_errors.hpp）。
// ⚠ 与 NN_ASSERT 的区别是**语义**，不是开关：NN_ASSERT 是"可选诊断"（NDEBUG 下
// 为空），只允许用在"失败也不影响后续正确性"的检查上；本函数的两个宏是
// "前置条件校验"（失败后继续执行 = UB），NDEBUG 下同样终止。
[[noreturn]] inline void fatal_at(const char *tag, const char *file, int line,
                                  const char *msg)
{
    std::string s(tag);
    s += " failed at ";
    s += (file != nullptr) ? file : "?";
    s += ':';
    s += std::to_string(line);
    s += ": ";
    s += (msg != nullptr) ? msg : "?";
    fail_abort(s);
}

} // namespace detail

// ── 调试断言宏（带位置信息）────────────────────────────────────────────
// **可选诊断**：用于内部不变量检查，生产环境（NDEBUG）为空展开。
// ⚠ 只在"失败后不再使用被校验的前提"时用它；否则请用 NN_ASSERT_ALWAYS。
#ifdef NDEBUG
    #define NN_ASSERT(cond, msg) ((void)0)
#else
    #define NN_ASSERT(cond, msg) \
        do { \
            if (!(cond)) { \
                std::fprintf(stderr, "NN_ASSERT failed at %s:%d: %s\n", \
                             __FILE__, __LINE__, (msg)); \
                std::abort(); \
            } \
        } while(0)
#endif

// ── 前置条件校验宏（**Release 也终止**，RV-01）────────────────────────────
// 语义 = "断言后继续用该条件" 的正确形态：NDEBUG 下与 Debug 下**同形**
// （都打印 file:line 并终止），因此不存在"Release 静默 UB"的窗口。
// 判定口径：凡是断言点之后**仍会**解引用/索引被校验对象的，一律用这两个宏。
// 代价 = 一次整型比较，与 algebra_matrix.hpp::require_same_shape 的既有修法同价。
#define NN_ASSERT_ALWAYS(cond, msg) \
    do { \
        if (!(cond)) \
            ::nn::detail::fatal_at("NN_ASSERT_ALWAYS", __FILE__, __LINE__, (msg)); \
    } while(0)

// 无条件终止（无条件的编程错误：走到这里本身就是 bug）。
// msg 必须是**可打印的字符串**（char* / std::string::c_str()）；格式化场景
// 先 snprintf 到局部缓冲再传（避免把 std::string_view::data() 当 NUL 结尾串）。
#define NN_FATAL(msg) \
    ::nn::detail::fatal_at("NN_FATAL", __FILE__, __LINE__, (msg))

// ── 运行时检查宏（始终启用）───────────────────────────────────────────
// 用于前置条件校验，返回 Error 而非 abort
#define NN_REQUIRE(cond, msg) \
    do { \
        if (!(cond)) { \
            return std::unexpected(nn::Error{(msg)}); \
        } \
    } while(0)

// ── 热点函数标记（提示编译器优化）─────────────────────────────────────
// 用于高频调用的前向/反向传播函数
#if defined(__GNUC__) || defined(__clang__)
    #define NN_HOT_FN [[gnu::hot]] [[gnu::always_inline]]
#elif defined(_MSC_VER)
    #define NN_HOT_FN __forceinline
#else
    #define NN_HOT_FN inline
#endif

// ── 分支预测提示 ─────────────────────────────────────────────────────
// 用于罕见错误路径，帮助编译器优化指令流水线
#if defined(__GNUC__) || defined(__clang__)
    #define NN_LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define NN_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    #define NN_LIKELY(x)   (x)
    #define NN_UNLIKELY(x) (x)
#endif

} // namespace nn

