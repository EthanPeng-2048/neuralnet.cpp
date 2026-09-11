#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  core_file.hpp — 文件 I/O 工具
//
//  职责：提供通用的文件读取函数与二进制 POD 读写原语，消除各模块间的重复代码。
//  设计：
//    - 文本读取函数返回 Result<T>，失败时返回 Error 错误信息
//    - 使用 std::istreambuf_iterator 读取，避免 tellg 失败风险
//    - 二进制 I/O 的字节级转换（std::as_bytes + 唯一一处 reinterpret_cast）
//      收敛在本头的 write_pod / read_pod / write_pod_span / read_pod_span
//      四个函数内，库内其它位置不再出现 reinterpret_cast 字节 cast
//      （docs/17 §2.1 指针审查结论：二进制 I/O 是合法边界，收敛而非消灭）
// ═══════════════════════════════════════════════════════════════════════════

#include <concepts>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include "core_errors.hpp"

namespace nn
{

// ── 读取文本文件（UTF-8） ──────────────────────────────────────────────
// 使用 istreambuf_iterator 读取全部内容，无需 tellg，避免失败场景。
[[nodiscard]] inline Result<std::string> load_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open text file: " + path});
    return std::string{std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>()};
}

// ── 二进制 POD 读写（全库 reinterpret_cast 字节边界的唯一收敛点）───────
// std::basic_istream/ostream 的 char 接口与 C++ 对象字节表示之间的转换
// 走 std::as_bytes / std::as_writable_bytes + 本段内唯一的
// reinterpret_cast；其它模块一律调用这四个函数（docs/17 §2.1）。
// 返回值：流的 good 状态（false = 读/写失败，调用方决定错误语义）。

// 写单个平凡可拷贝对象（定长 POD / 整数 / float 等）
template <typename T>
    requires std::is_trivially_copyable_v<T>
inline bool write_pod(std::ostream &os, const T &v)
{
    auto bytes = std::as_bytes(std::span(&v, 1));
    os.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size_bytes()));
    return static_cast<bool>(os);
}

// 读单个平凡可拷贝对象
template <typename T>
    requires std::is_trivially_copyable_v<T>
inline bool read_pod(std::istream &is, T &v)
{
    auto bytes = std::as_writable_bytes(std::span(&v, 1));
    is.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size_bytes()));
    return static_cast<bool>(is);
}

// 写连续字节序列（std::span 视图，如 matrix 数据、packed 结构体数组）
template <typename T, std::size_t Ext = std::dynamic_extent>
inline bool write_pod_span(std::ostream &os, const std::span<T, Ext> &s)
{
    auto bytes = std::as_bytes(s);
    os.write(reinterpret_cast<const char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size_bytes()));
    return static_cast<bool>(os);
}

// 读连续字节序列
template <typename T, std::size_t Ext = std::dynamic_extent>
inline bool read_pod_span(std::istream &is, const std::span<T, Ext> &s)
{
    auto bytes = std::as_writable_bytes(s);
    is.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size_bytes()));
    return static_cast<bool>(is);
}

} // namespace nn

