#ifndef NN_CORE_ASSERT_HPP
#define NN_CORE_ASSERT_HPP

#include <cstdio>
#include <string>
#include <expected>
#include "../core/errors.hpp"

namespace nn {

// ── 调试断言宏（带位置信息）────────────────────────────────────────────
// 用于内部不变量检查，生产环境可通过 NDEBUG 禁用
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

#endif // NN_CORE_ASSERT_HPP
