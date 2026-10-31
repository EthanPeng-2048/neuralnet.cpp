#pragma once

// ── core_env.hpp — 环境变量读取的**唯一**入口（RV-09）─────────────────────────
// 库内一切环境变量读取（行为开关 + 诊断探针）都经本头，不再各自写
// getenv / _dupenv_s 样板——原先散落在 expr_dsl / core_profile / compute_engine /
// backend::VulkanDevice 的 ≥6 份 helper（MSVC 的 _dupenv_s 分支处处重复）
// 全部收敛到这里，`getenv` 此后只在本文件出现。
//
// 两层语义（AGENTS §2）——本头把两层都收敛到一处：
//   1) **探针门**（probe_flag_raw / probe_string_raw）：`NN_ENABLE_PROBES=OFF`
//      （→ `NN_PROBES_ENABLED==0`）时**恒 false / 空串，且不求值 getenv**——
//      默认构建里诊断开关不产生任何分支/字符串/系统调用。
//   2) **行为开关**（env_flag_raw / env_string_raw / env_int_raw）：初始化期
//      读一次并缓存的开关（NN_VULKAN_DEVICE / NN_POOL_* …）
//      用原始读取；调用点自行写 `static const bool on = ...` 缓存（进程内恒定）。
//
// 取值约定：raw flag = "存在、非空、首字符 != '0'"（"0" = 关）。这是全仓统一
// 口径——原先 `dsl::env_flag` 用"任意非空即真"，与其余 helper 不一致；RV-09
// 一并对齐到"0 = 关"（二者都是探针开关，默认不设，行为不受影响）。
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdlib>
#include <string>

#include <neuralnet.cpp/core_config.hpp>   // NN_PROBES_ENABLED（探针编译门）

namespace nn::detail
{

// ── 原始读取（**不经**探针门；行为开关用，调用点自行缓存）───────────────────

// 环境变量开关：存在、非空、首字符 != '0' → true（"0" = 关）。
// MSVC CRT 把 getenv 标记 deprecated（-Werror 下报错）→ 走 _dupenv_s。
[[nodiscard]] inline bool env_flag_raw(const char* name) noexcept
{
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    const bool ok = (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr && buf[0] != '\0');
    const bool on = ok && buf[0] != '0';
    std::free(buf);
    return on;
#else
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
}

// 环境变量字符串：未设置 / 读取失败 → 空串。
[[nodiscard]] inline std::string env_string_raw(const char* name)
{
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || buf == nullptr) return {};
    std::string s(buf);
    std::free(buf);
    return s;
#else
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string{};
#endif
}

// 环境变量整数：strtol 十进制解析（跳过前导空白、接受正负号），要求整串合法
// （无尾随字符）；缺失 / 非数字 / 溢出 int → fallback。
[[nodiscard]] inline int env_int_raw(const char* name, int fallback) noexcept
{
    const std::string s = env_string_raw(name);
    if (s.empty()) return fallback;
    const char* p = s.c_str();
    char* end = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p || *end != '\0') return fallback;
    if (v < -2147483647L - 1L || v > 2147483647L) return fallback;
    return static_cast<int>(v);
}

// ── 探针门下的读取（`NN_ENABLE_PROBES=OFF` 时恒 false / 空串，**不求值 getenv**）──

[[nodiscard]] inline bool probe_flag_raw(const char* name) noexcept
{
#if NN_PROBES_ENABLED
    return env_flag_raw(name);
#else
    (void)name;
    return false;
#endif
}

[[nodiscard]] inline std::string probe_string_raw(const char* name)
{
#if NN_PROBES_ENABLED
    return env_string_raw(name);
#else
    (void)name;
    return {};
#endif
}

} // namespace nn::detail
