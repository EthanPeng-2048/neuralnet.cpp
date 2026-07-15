# ── embed_spirv.cmake ────────────────────────────────────────────────────
# 将 SPIR-V 二进制文件转换为 C++ 头文件（constexpr 数组）。
#
# 用法（由 CMakeLists.txt 调用）：
#   cmake -DSPV_INPUT=matmul.spv -DHPP_OUTPUT=matmul_spv.hpp -P embed_spirv.cmake
#
# 输出的头文件定义了：
#   nn::nn_matmul_spirv_bytecode() → const std::vector<uint32_t>&
# ─────────────────────────────────────────────────────────────────────────

file(READ "${SPV_INPUT}" HEX_CONTENT HEX)
string(LENGTH "${HEX_CONTENT}" HEX_LENGTH)
math(EXPR BYTE_COUNT "${HEX_LENGTH} / 2")
math(EXPR UINT32_COUNT "${BYTE_COUNT} / 4")

if(UINT32_COUNT LESS 1)
    message(FATAL_ERROR "SPIR-V file is empty or invalid: ${SPV_INPUT}")
endif()

set(ARRAY_INIT "")
math(EXPR LAST_INDEX "${UINT32_COUNT} - 1")

foreach(I RANGE ${LAST_INDEX})
    math(EXPR OFFSET "${I} * 8")
    string(SUBSTRING "${HEX_CONTENT}" ${OFFSET} 8 HEX_WORD)

    # 提取 4 个字节（SPIR-V 是小端序）
    string(SUBSTRING "${HEX_WORD}" 0 2 B0)
    string(SUBSTRING "${HEX_WORD}" 2 2 B1)
    string(SUBSTRING "${HEX_WORD}" 4 2 B2)
    string(SUBSTRING "${HEX_WORD}" 6 2 B3)

    # 反转字节序得到 uint32_t 的十六进制表示
    if(NOT "${ARRAY_INIT}" STREQUAL "")
        string(APPEND ARRAY_INIT ", ")
    endif()
    math(EXPR REM "${I} % 8")
    if(${REM} EQUAL 0 AND NOT ${I} EQUAL 0)
        string(APPEND ARRAY_INIT "\n        ")
    endif()
    string(APPEND ARRAY_INIT "0x${B3}${B2}${B1}${B0}")
endforeach()

# 写入生成的头文件
file(WRITE "${HPP_OUTPUT}"
"// ── 自动生成，请勿手动编辑 ──\n"
"// 由 cmake/embed_spirv.cmake 从 SPIR-V 字节码生成\n"
"#pragma once\n"
"\n"
"#include <cstdint>\n"
"#include <vector>\n"
"\n"
"namespace nn {\n"
"\n"
"[[nodiscard]] inline const std::vector<uint32_t>& nn_matmul_spirv_bytecode()\n"
"{\n"
"    static const std::vector<uint32_t> spirv = {\n"
"        ${ARRAY_INIT}\n"
"    };\n"
"    return spirv;\n"
"}\n"
"\n"
"} // namespace nn\n"
)

message(STATUS "Embedded SPIR-V: ${UINT32_COUNT} words (${BYTE_COUNT} bytes) → ${HPP_OUTPUT}")
