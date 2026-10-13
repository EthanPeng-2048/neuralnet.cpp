// ── rapt_test — RAPT (ReLU 线性注意力) 合并测试 ─────────────────────────────
// 合并：rapt_gradcheck + rapt_smoke_test + rapt_checkpoint_test
// 覆盖：数值梯度检查 / 端到端冒烟 / 梯度检查点一致性
// （GPU 侧的 activation offload 见独立目标 rapt_offload_test）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>
#include <string>

#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#define main test_rapt_gradcheck
#include "rapt_gradcheck.cpp"
#undef main

#define main test_rapt_smoke
#include "rapt_smoke_test.cpp"
#undef main

#define main test_rapt_checkpoint
#include "rapt_checkpoint_test.cpp"
#undef main

int main(int argc, char* argv[])
{
    // --gpu 用例的可用性预检（与 cnn_test 同款约定）：GPU 不可用（未编译
    // Vulkan / 初始化失败）→ 77，依 ctest SKIP 约定记 Skipped 而非假红。
    // 子测试本身在引擎创建失败时返回 1，没有 77 语义，故在这里统一预检。
    // （本次因新增 rapt_test_f16_gpu 用例才暴露此前缺这层预检。）
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

    std::puts("=== rapt_gradcheck (RLA core + block) ===");
    failures += test_rapt_gradcheck(argc, argv);

    std::puts("=== rapt_smoke (build + spec roundtrip + loss) ===");
    failures += test_rapt_smoke(argc, argv);

    std::puts("=== rapt_checkpoint (activation checkpointing) ===");
    failures += test_rapt_checkpoint();

    std::printf("\nrapt_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
