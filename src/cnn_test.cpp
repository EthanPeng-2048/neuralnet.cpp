// ── cnn_test — CNN 合并测试 ────────────────────────────────────────────────
// 合并：maxpool_gradcheck + cnn_smoke_test
// 覆盖：MaxPool2D 独立参考实现比对 + 缓存/checkpoint 契约 + CNN 端到端冒烟
//       （Conv2D 的单层参考比对仍在 layer_gradcheck_test 的 conv2d 段）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>
#include <string>

// ── maxpool_gradcheck（独立参考实现比对）──────────────────────────────────
#define main test_maxpool_gradcheck
#define max_abs_diff max_abs_diff_maxpool
#include "maxpool_gradcheck.cpp"
#undef max_abs_diff
#undef main

// ── cnn_smoke_test（规格往返 + 层组成 + 端到端训练）───────────────────────
#define main test_cnn_smoke
#include "cnn_smoke_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    // --gpu 用例的可用性预检：GPU 不可用（未编译 Vulkan / 初始化失败）→ 77，
    // 依 ctest SKIP 约定记 Skipped 而非假红（与 gpu_test 等同一约定；
    // 子测试本身在引擎创建失败时返回 1，没有 77 语义，故在这里统一预检）。
    //
    // 为什么必须有 GPU 用例：DSL 表达式的 ExprSpec 校验只发生在 GPU / scan
    // 路径，CPU 走编译期模板求值、**不校验**——结构非法的表达式（如裸视图根
    // → 空指令表）只会在 GPU 运行期暴露，纯 CPU 跑一遍测不出来。
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) != "--gpu") continue;
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto probe = nn::cli::create_engine(ec);
        NN_EXIT(probe, 77, "GPU 不可用，跳过: ");
        break;
    }

    int failures = 0;

    std::puts("=== maxpool_gradcheck (independent reference) ===");
    failures += test_maxpool_gradcheck(argc, argv);

    std::puts("=== cnn_smoke (build + spec roundtrip + train) ===");
    failures += test_cnn_smoke(argc, argv);

    std::printf("\ncnn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
