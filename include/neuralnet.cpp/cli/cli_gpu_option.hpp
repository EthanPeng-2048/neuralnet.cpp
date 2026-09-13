// ── cli_gpu_option.hpp — `--gpu` 选项解析（可带设备选择子）──────────────────
//
// `--gpu` 原本只是"启用 GPU"的布尔开关；这里把它扩展为**可选带设备**：
//   --gpu               启用 GPU 加速，自动选卡（设备类型 + apiVersion 打分）
//   --gpu <索引>        启用并指定设备（空格形式只接受纯数字枚举索引）
//   --gpu=<设备>        启用并指定设备，设备可为索引或名称子串（无歧义写法）
//
// 设备 = 枚举索引（"2"）或名称子串（"40HX" / "NVIDIA"）。
//
// 为什么空格形式只收数字：多个入口把"不以 -- 开头的参数"当位置参数
// （如 text_infer 的 prompt、text_train 的输入路径），若 `--gpu` 吞掉任意
// 下一个 token，`text_infer --gpu "hello"` 会被误解析成选设备。名称子串
// 一律用 `--gpu=<名称>`；无位置参数的程序可传 allow_name_value=true 放开。
//
// 背景：多卡机器上"第一张独显"未必是想用的卡（实测本机枚举 [0] 是
// AMD R5 240、[2] 才是 NVIDIA CMP 40HX），必须能手动作选择。
// ────────────────────────────────────────────────────────────────────────────

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace nn::cli
{
    // 纯数字 token（枚举索引）判定
    [[nodiscard]] inline bool is_device_index_token(const char *token) noexcept
    {
        if (token == nullptr || *token == '\0')
            return false;
        for (const char *p = token; *p != '\0'; ++p)
        {
            if (*p < '0' || *p > '9')
                return false;
        }
        return true;
    }

    // 在 argv[i] 处尝试解析 `--gpu` 选项。
    //   返回 std::nullopt → 不是 --gpu（i 不变，调用方继续判断其他 flag）
    //   返回 空 string    → 是 --gpu，未指定设备（i 不变）
    //   返回 非空 string  → 是 --gpu 且指定了设备（i 递增到本次消费的索引）
    //
    // allow_name_value：空格形式是否连名称也收（仅无位置参数的程序可开）。
    [[nodiscard]] inline std::optional<std::string> parse_gpu_option(
        int argc, char *argv[], int &i, bool allow_name_value = false)
    {
        const std::string_view arg = argv[i];
        if (arg == "--gpu")
        {
            if (i + 1 < argc && argv[i + 1] != nullptr)
            {
                const bool is_index = is_device_index_token(argv[i + 1]);
                if (is_index || (allow_name_value && argv[i + 1][0] != '-'))
                    return std::string(argv[++i]);
            }
            return std::string{};
        }

        constexpr std::string_view kPrefix = "--gpu=";
        if (arg.rfind(kPrefix, 0) == 0)
            return std::string(arg.substr(kPrefix.size()));

        return std::nullopt;
    }
} // namespace nn::cli
