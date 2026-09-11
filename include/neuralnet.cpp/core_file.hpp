#ifndef NN_CORE_FILE_HPP
#define NN_CORE_FILE_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  core_file.hpp — 文件读取工具
//
//  职责：提供通用的文件读取函数，消除各模块间的重复代码。
//  设计：
//    - 所有函数返回 Result<T>，失败时返回 Error 错误信息
//    - 使用 std::istreambuf_iterator 读取，避免 tellg 失败风险
// ═══════════════════════════════════════════════════════════════════════════

#include <fstream>
#include <iterator>
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

} // namespace nn

#endif // NN_CORE_FILE_HPP
