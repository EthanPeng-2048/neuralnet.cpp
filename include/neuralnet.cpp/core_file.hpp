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
//      四个函数内，库内其它位置不出现 reinterpret_cast 字节 cast
//      （设计取向：二进制 I/O 是合法边界，收敛到本头而非消灭）
// ═══════════════════════════════════════════════════════════════════════════

#include <concepts>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_sha256.hpp>

namespace nn
{

// ── 读取文本文件（UTF-8） ──────────────────────────────────────────────
// 使用 istreambuf_iterator 读取全部内容，无需 tellg，避免失败场景。
[[nodiscard]] inline Result<std::string> load_text_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open text file: " + path);
    return std::string{std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>()};
}

// ── 流式文件摘要（P2-1：整文件校验和 / 数据集身份）─────────────────────
// 分块累加 sha256（1 MiB 块），不把文件读进内存；`limit_bytes` 限定只摘要
// 文件前缀（整文件校验和要排除尾部的摘要本身）。文件不足 `limit_bytes` 时报错。
[[nodiscard]] inline Result<std::array<std::uint8_t, 32>> sha256_file_prefix(
    const std::string &path, std::uint64_t limit_bytes = ~std::uint64_t{0})
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for hashing: " + path);

    detail::Sha256State st;
    std::array<char, 1u << 20> buf{};
    std::uint64_t remain = limit_bytes;
    while (remain > 0)
    {
        const auto want = static_cast<std::streamsize>(
            remain < buf.size() ? remain : buf.size());
        ifs.read(buf.data(), want);
        const auto got = ifs.gcount();
        if (got <= 0)
        {
            if (limit_bytes == ~std::uint64_t{0}) break;   // 正常读到文件尾
            NN_FAIL("文件短于摘要范围（" + path + "）：期望前 " +
                    std::to_string(limit_bytes) + " 字节");
        }
        detail::sha256_update(st, std::string_view(buf.data(),
                                                  static_cast<std::size_t>(got)));
        remain -= static_cast<std::uint64_t>(got);
    }
    return detail::sha256_finalize(st);
}

// 整文件 sha256（hex）——数据集身份（`TrainState.dataset.sha256`）用
[[nodiscard]] inline Result<std::string> sha256_file_hex(const std::string &path)
{
    NN_TRY(d, sha256_file_prefix(path));
    return sha256_hex(*d);
}

// ── 二进制 POD 读写（全库 reinterpret_cast 字节边界的唯一收敛点）───────
// std::basic_istream/ostream 的 char 接口与 C++ 对象字节表示之间的转换
// 走 std::as_bytes / std::as_writable_bytes + 本段内唯一的
// reinterpret_cast；其它模块一律调用这四个函数。
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

