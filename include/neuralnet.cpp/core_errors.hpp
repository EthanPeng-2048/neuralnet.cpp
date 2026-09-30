#pragma once

#include <charconv>
#include <cstddef>
#include <cstdlib>  // for strtof/strtod/strtold（浮点解析回退）
#include <expected>
#include <string>
#include <string_view>
#include <type_traits>

namespace nn {

// ── 错误类型（C++23 std::expected）─────────────────────────────────────
// 所有公共 API 使用 Result<T> 返回错误，不抛异常。
struct Error {
    std::string message;
};

template <typename T>
using Result = std::expected<T, Error>;

// ── [[nodiscard]] 包装：让丢弃 Result 的调用在编译期告警 ──────────────
// 用法：auto r = nn::check_result(foo());  // 若丢弃 r 会告警
// 注：Result<T> 本身是 std::expected 的别名，无法在别名声明上加属性，
// 此处提供显式 nodiscard 包装用于关键路径。
template <typename T>
[[nodiscard]] constexpr Result<T> check_result(Result<T> r) noexcept
{
    return r;
}

// ── 错误传播（铁律 #1 的写法收敛）──────────────────────────────────────
// `auto r = f(); if (!r) return std::unexpected(r.error());` 是本项目实施
// 铁律 #1 的主形态，在 L2 层重复约 500 次、占 Layer 代码近 9%。收敛为：
//
//     NN_TRY(r, f());        // 声明 r 并检查（r 是 Result<T>）
//     NN_TRY_CHECK(r);       // 检查已声明的 r
//
// 两者都只是**语法压缩**：展开后与手写形态逐字等价（错误消息、返回类型、
// 语句结构都相同），不含任何控制流或所有权语义。展开为普通语句（不用
// do{}while(0)），因此在 if/else 里的悬垂 else 行为与原 if 语句完全一致。
//
// 注意（铁律 #10）：续行反斜杠后面紧跟的是代码，绝不能出现 // 注释。
#define NN_TRY(decl, ...) \
    auto decl = (__VA_ARGS__); \
    if (!decl) return std::unexpected(decl.error())

#define NN_TRY_CHECK(x) \
    if (!(x)) return std::unexpected((x).error())

// ── 数字解析（std::from_chars，不抛异常） ────────────────────────────
// 替代 std::stoi/stod/stoul/stoull 等会抛异常的函数。
// 用法：auto v = nn::parse_number<int>("123");
//       if (!v) return std::unexpected(v.error());
template <typename T>
    requires std::is_arithmetic_v<T>
[[nodiscard]] inline Result<T> parse_number(std::string_view s) noexcept
{
    // 去除首尾空白（from_chars 不容忍前后空白）
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\n' || s.front() == '\r'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\n' || s.back() == '\r'))
        s.remove_suffix(1);

    if (s.empty())
        return std::unexpected(Error{"empty numeric string"});

    T value{};
    if constexpr (std::is_floating_point_v<T>)
    {
        // 浮点：libc++ 部分版本缺少浮点 std::from_chars（__charconv/from_chars_integral.h
        // 只有整型版），会选中被删除的 bool 重载而编译失败。故用 C 的 strtoX
        // （全平台/全标准库都有、不抛异常）。s 已去除首尾空白，要求整串被解析。
        std::string tmp(s);  // strtoX 需要 null 结尾
        char* p = nullptr;
        if constexpr (std::is_same_v<T, float>)
            value = std::strtof(tmp.c_str(), &p);
        else if constexpr (std::is_same_v<T, double>)
            value = std::strtod(tmp.c_str(), &p);
        else
            value = std::strtold(tmp.c_str(), &p);
        if (p != tmp.c_str() + tmp.size())
            return std::unexpected(Error{"invalid number: " + std::string(s)});
    }
    else
    {
        // string_view::end() 即指向末尾后一位置的 const char*，无需手动指针算术
        auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ec != std::errc{} || ptr != s.data() + s.size())
            return std::unexpected(Error{"invalid number: " + std::string(s)});
    }
    return value;
}

} // namespace nn

