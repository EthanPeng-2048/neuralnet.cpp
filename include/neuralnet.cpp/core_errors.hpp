#ifndef NN_CORE_ERRORS_HPP
#define NN_CORE_ERRORS_HPP

#include <charconv>
#include <cstddef>
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
    const auto *end = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(s.data(), end, value);
    if (ec != std::errc{} || ptr != end)
        return std::unexpected(Error{"invalid number: " + std::string(s)});
    return value;
}

} // namespace nn

#endif // NN_CORE_ERRORS_HPP