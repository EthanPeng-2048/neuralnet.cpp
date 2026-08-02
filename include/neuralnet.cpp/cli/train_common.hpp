// ── train_common.hpp — 通用训练参数解析 ─────────────────────────────────────
//
// 抽取自 mnist_train/text_train 中重复的 flag 解析逻辑。
// 覆盖以下通用 flag：
//   --epochs <n>            训练轮数（正整数）
//   --lr <lr>               学习率
//   --batch-size <n>        批大小（正整数）
//   --optimizer <name>      优化器：sgd/sgd_momentum/adam/adamw/muon
//   --weight-decay <w>      AdamW 权重衰减系数
//   --min-lr <lr>           余弦退火最低学习率
//   --warmup-epochs <n>     线性预热轮数
//   --lr-per-epoch <v1,...> 手动指定每轮学习率（逗号分隔）
//   --lr-schedule <type>    学习率调度：fixed/constant/cosine
//   --osc-guard <on|off>    振荡检测自动降 lr
//   --osc-window <n>        振荡检测窗口大小
//   --osc-threshold <f>     振荡反转率阈值
//   --gpu                   启用 Vulkan GPU 加速
//   --cuda                  启用 CUDA GPU 加速
//
// 用法：在调用方的 parse_args 循环中优先委托：
//   for (int i = 1; i < argc; ++i) {
//       if (nn::cli::parse_train_common_args(argc, argv, i, common)) continue;
//       // 程序专属 flag...
//   }
//
// 返回值约定：
//   true  = 已处理该 flag（i 已递增到本次消费的最后一个 argv 索引）
//   false = 不是通用 flag（i 未变，调用方自行处理）
//
// 注：--help 由调用方自行处理（各程序 print_usage 不同）。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "neuralnet.cpp/config.hpp"      // nn::Scalar
#include "neuralnet.cpp/core_errors.hpp"  // nn::parse_number

namespace nn::cli
{
    // ── 通用训练参数 ───────────────────────────────────────────────────────
    // 默认值仅作为结构体占位；调用方应根据程序语义在解析前覆盖。
    struct TrainCommonArgs
    {
        int epochs = 10;
        nn::Scalar lr = 0.001f;
        std::size_t batch_size = 32;
        std::string optimizer = "adam";
        nn::Scalar weight_decay = 0.0f;
        nn::Scalar min_lr = 0.0f;
        int warmup_epochs = 0;
        std::string lr_schedule = "constant";  // "fixed"/"constant" 等价，"cosine" 退火
        std::vector<nn::Scalar> lr_per_epoch;
        bool osc_guard = false;
        int osc_window = 20;
        nn::Scalar osc_threshold = 2.0f;
        bool use_gpu = false;
        bool use_cuda = false;
    };

    // ── 解析通用训练参数 ───────────────────────────────────────────────────
    // 在 argv[i] 处尝试解析一个通用 flag。
    // 成功消费时返回 true 并将 i 递增到本次消费的最后一个 argv 索引；
    // 不是通用 flag 时返回 false（i 不变），由调用方继续判断。
    // 解析失败时打印错误并 std::exit(1)（与现有 parse_args 行为一致）。
    inline bool parse_train_common_args(
        int argc, char *argv[], int &i, TrainCommonArgs &cfg)
    {
        std::string arg = argv[i];

        if (arg == "--epochs" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --epochs: " << v.error().message << "\n"; std::exit(1); }
            if (*v <= 0) { std::cerr << "--epochs 必须为正整数\n"; std::exit(1); }
            cfg.epochs = *v;
            return true;
        }
        if (arg == "--lr" && i + 1 < argc)
        {
            auto v = nn::parse_number<nn::Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --lr: " << v.error().message << "\n"; std::exit(1); }
            cfg.lr = *v;
            return true;
        }
        if (arg == "--batch-size" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --batch-size: " << v.error().message << "\n"; std::exit(1); }
            if (*v == 0) { std::cerr << "--batch-size 必须为正整数\n"; std::exit(1); }
            cfg.batch_size = *v;
            return true;
        }
        if (arg == "--optimizer" && i + 1 < argc)
        {
            cfg.optimizer = argv[++i];
            if (cfg.optimizer != "sgd" && cfg.optimizer != "sgd_momentum" &&
                cfg.optimizer != "adam" && cfg.optimizer != "adamw" &&
                cfg.optimizer != "muon")
            {
                std::cerr << "未知优化器: " << cfg.optimizer
                          << "，可选: sgd, sgd_momentum, adam, adamw, muon\n";
                std::exit(1);
            }
            return true;
        }
        if (arg == "--weight-decay" && i + 1 < argc)
        {
            auto v = nn::parse_number<nn::Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --weight-decay: " << v.error().message << "\n"; std::exit(1); }
            cfg.weight_decay = *v;
            return true;
        }
        if (arg == "--min-lr" && i + 1 < argc)
        {
            auto v = nn::parse_number<nn::Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --min-lr: " << v.error().message << "\n"; std::exit(1); }
            cfg.min_lr = *v;
            return true;
        }
        if (arg == "--warmup-epochs" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --warmup-epochs: " << v.error().message << "\n"; std::exit(1); }
            cfg.warmup_epochs = *v;
            return true;
        }
        if (arg == "--lr-per-epoch" && i + 1 < argc)
        {
            std::string dims_str = argv[++i];
            std::stringstream ss(dims_str);
            std::string token;
            cfg.lr_per_epoch.clear();
            while (std::getline(ss, token, ','))
            {
                auto v = nn::parse_number<nn::Scalar>(token);
                if (!v) { std::cerr << "无效 --lr-per-epoch 值: " << v.error().message << "\n"; std::exit(1); }
                cfg.lr_per_epoch.push_back(*v);
            }
            return true;
        }
        if (arg == "--lr-schedule" && i + 1 < argc)
        {
            cfg.lr_schedule = argv[++i];
            // "fixed" 与 "constant" 等价（历史兼容）；"cosine" 为余弦退火
            if (cfg.lr_schedule != "fixed" && cfg.lr_schedule != "constant" &&
                cfg.lr_schedule != "cosine")
            {
                std::cerr << "未知 --lr-schedule: " << cfg.lr_schedule
                          << "，可选: fixed, constant, cosine\n";
                std::exit(1);
            }
            return true;
        }
        if (arg == "--osc-guard" && i + 1 < argc)
        {
            std::string v = argv[++i];
            if (v == "on" || v == "1" || v == "true")
                cfg.osc_guard = true;
            else if (v == "off" || v == "0" || v == "false")
                cfg.osc_guard = false;
            else
            {
                std::cerr << "无效 --osc-guard: " << v << "，可选: on, off\n";
                std::exit(1);
            }
            return true;
        }
        if (arg == "--osc-window" && i + 1 < argc)
        {
            auto v = nn::parse_number<int>(argv[++i]);
            if (!v) { std::cerr << "无效 --osc-window: " << v.error().message << "\n"; std::exit(1); }
            cfg.osc_window = *v;
            return true;
        }
        if (arg == "--osc-threshold" && i + 1 < argc)
        {
            auto v = nn::parse_number<nn::Scalar>(argv[++i]);
            if (!v) { std::cerr << "无效 --osc-threshold: " << v.error().message << "\n"; std::exit(1); }
            cfg.osc_threshold = *v;
            return true;
        }
        if (arg == "--gpu")
        {
            cfg.use_gpu = true;
            return true;
        }
        if (arg == "--cuda")
        {
            cfg.use_cuda = true;
            return true;
        }

        return false;  // 不是通用 flag
    }
} // namespace nn::cli
