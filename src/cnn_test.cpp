// ── cnn_test — CNN 合并测试 ────────────────────────────────────────────────
// 合并：maxpool_gradcheck + cnn_smoke_test
// 覆盖：MaxPool2D 独立参考实现比对 + 缓存/checkpoint 契约 + CNN 端到端冒烟
//       （Conv2D 的单层参考比对仍在 layer_gradcheck_test 的 conv2d 段）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── maxpool_gradcheck（独立参考实现比对；MaxPool2D 此前无任何覆盖）─────────
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
    int failures = 0;

    std::puts("=== maxpool_gradcheck (independent reference) ===");
    failures += test_maxpool_gradcheck(argc, argv);

    std::puts("=== cnn_smoke (build + spec roundtrip + train) ===");
    failures += test_cnn_smoke(argc, argv);

    std::printf("\ncnn_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
