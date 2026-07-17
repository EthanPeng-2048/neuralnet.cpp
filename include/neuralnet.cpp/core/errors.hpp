#ifndef NN_CORE_ERRORS_HPP
#define NN_CORE_ERRORS_HPP

#include <string>
#include <expected>

namespace nn {

// ── 错误类型（C++23 std::expected）─────────────────────────────────────
// 所有公共 API 使用 Result<T> 返回错误，不抛异常。
struct Error {
    std::string message;
};

template <typename T>
using Result = std::expected<T, Error>;

} // namespace nn

#endif // NN_CORE_ERRORS_HPP
