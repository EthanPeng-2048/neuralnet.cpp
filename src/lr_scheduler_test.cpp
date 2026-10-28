// ── lr_scheduler_test — epoch 级学习率调度回归（issue #31）────────────────
//
// 覆盖 `LrScheduleConfig` / `compute_epoch_lr` 的四种退火形式与优先级：
//   [1] constant/fixed       → 恒 base_lr。
//   [2] cosine               → warmup + 余弦退火到 min_lr；越界钳 min_lr 不回升。
//   [3] multistep            → 每过一个 milestone × gamma 一次；空 milestones = 恒 base_lr。
//   [4] step_cosine (SGDR)   → 分段余弦，段末重启回 base_lr；末段吸收余数、
//                               精确落 min_lr；n_segments=1 == 普通 cosine。
//   [5] warmup 对**所有** schedule 统一生效（含 multistep / step_cosine）。
//   [6] lr_per_epoch 逃生舱  → 优先级最高，覆盖任何 schedule；越界取末值。
//   [7] compute_step_lr      → step 级 warmup + 可选余弦（回归锁定）。
//
// 数值口径：余弦退火 0.5*(1+cos(π*progress))；float 比较用相对容差。
// 退出码：0 = 通过，1 = 失败。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <vector>

#include "neuralnet.cpp/lr_scheduler.hpp"

#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

namespace
{
int g_fail = 0;

using nn::Scalar;
using nn::compute_epoch_lr;
using nn::compute_step_lr;
using nn::LrScheduleConfig;
using nn::StepLrScheduleConfig;

// 相对容差比较（float 调度曲线）
[[nodiscard]] bool near(Scalar a, Scalar b, Scalar tol = Scalar(1e-5))
{
    const Scalar d = std::fabs(a - b);
    const Scalar m = std::fabs(a) + std::fabs(b);
    return d <= tol * (m > Scalar(1) ? m : Scalar(1));
}

void test_constant()
{
    LrScheduleConfig c;
    c.base_lr = Scalar(3.14e-3);
    c.schedule = "constant";
    for (int e = 0; e < 5; ++e)
        CHECK(near(compute_epoch_lr(c, e), c.base_lr), "constant 恒 base_lr");
    // 未识别的 schedule 兜底 = constant
    c.schedule = "bogus";
    CHECK(near(compute_epoch_lr(c, 2), c.base_lr), "未知 schedule 兜底 constant");
}

void test_cosine()
{
    LrScheduleConfig c;
    c.base_lr = Scalar(1.0);
    c.min_lr = Scalar(0.0);
    c.warmup_epochs = 2;
    c.total_epochs = 10;
    c.schedule = "cosine";
    // warmup：e=0 → base*(1/2)；e=1 → base*(2/2)=base
    CHECK(near(compute_epoch_lr(c, 0), Scalar(0.5)), "cosine warmup e=0 = base/2");
    CHECK(near(compute_epoch_lr(c, 1), Scalar(1.0)), "cosine warmup e=1 = base");
    // 退火：有效区间 = 10-2 = 8；e=2 → progress 0 → base；e=10 → progress 1 → min
    CHECK(near(compute_epoch_lr(c, 2), Scalar(1.0)), "cosine 退火起点 = base_lr");
    CHECK(near(compute_epoch_lr(c, 10), Scalar(0.0)), "cosine 终点 = min_lr");
    // 中点 e=6 → progress 0.5 → 0.5*(base+min)
    CHECK(near(compute_epoch_lr(c, 6), Scalar(0.5)), "cosine 中点 = 0.5*(base+min)");
    // 越界（续训超出 total_epochs）钳到 min_lr，不回升
    CHECK(near(compute_epoch_lr(c, 20), Scalar(0.0)), "cosine 越界钳 min_lr 不回升");
}

void test_multistep()
{
    LrScheduleConfig c;
    c.base_lr = Scalar(1.0);
    c.warmup_epochs = 0;
    c.schedule = "multistep";
    c.milestones = {30, 60, 90};
    c.gamma = Scalar(0.1);
    CHECK(near(compute_epoch_lr(c, 0), Scalar(1.0)), "multistep e=0 = base");
    CHECK(near(compute_epoch_lr(c, 29), Scalar(1.0)), "multistep e=29 = base（未到首个）");
    CHECK(near(compute_epoch_lr(c, 30), Scalar(0.1)), "multistep e=30 过首个 → ×gamma");
    CHECK(near(compute_epoch_lr(c, 60), Scalar(0.01)), "multistep e=60 过两 → ×gamma²");
    CHECK(near(compute_epoch_lr(c, 90), Scalar(1e-3)), "multistep e=90 过三 → ×gamma³");
    // 空 milestones = 预热后恒 base
    c.milestones.clear();
    CHECK(near(compute_epoch_lr(c, 500), Scalar(1.0)), "multistep 空 milestones 恒 base");
}

void test_step_cosine()
{
    LrScheduleConfig c;
    c.base_lr = Scalar(1.0);
    c.min_lr = Scalar(0.0);
    c.warmup_epochs = 0;
    c.total_epochs = 12;
    c.schedule = "step_cosine";
    c.n_segments = 3;
    // anneal_span=12, segs=3 → 每段 4 epoch：段1 rel∈[0,4)、段2 [4,8)、段3 [8,12]
    CHECK(near(compute_epoch_lr(c, 0), Scalar(1.0)), "step_cosine 段1 起点重启 base");
    CHECK(near(compute_epoch_lr(c, 4), Scalar(1.0)), "step_cosine 段2 起点重启 base");
    CHECK(near(compute_epoch_lr(c, 8), Scalar(1.0)), "step_cosine 段3 起点重启 base");
    CHECK(near(compute_epoch_lr(c, 2), Scalar(0.5)), "step_cosine 段内中点 = 0.5");
    CHECK(near(compute_epoch_lr(c, 12), Scalar(0.0)), "step_cosine 末段末 = min_lr");
    CHECK(near(compute_epoch_lr(c, 99), Scalar(0.0)), "step_cosine 越界钳 min_lr 不回升");
    // n_segments=1 退化为普通 cosine
    c.n_segments = 1;
    CHECK(near(compute_epoch_lr(c, 12), Scalar(0.0)), "step_cosine n=1 末 = min_lr");
    CHECK(near(compute_epoch_lr(c, 6), Scalar(0.5)), "step_cosine n=1 中点 = 0.5");
}

void test_warmup_uniform()
{
    // warmup 对 multistep / step_cosine 同样生效
    LrScheduleConfig ms;
    ms.base_lr = Scalar(1.0);
    ms.warmup_epochs = 2;
    ms.schedule = "multistep";
    ms.milestones = {5};
    ms.gamma = Scalar(0.1);
    CHECK(near(compute_epoch_lr(ms, 0), Scalar(0.5)), "warmup 生效于 multistep e=0");
    CHECK(near(compute_epoch_lr(ms, 1), Scalar(1.0)), "warmup 生效于 multistep e=1");
    CHECK(near(compute_epoch_lr(ms, 5), Scalar(0.1)), "multistep milestone 在 warmup 后仍触发");

    LrScheduleConfig sc;
    sc.base_lr = Scalar(1.0);
    sc.warmup_epochs = 2;
    sc.total_epochs = 6;
    sc.schedule = "step_cosine";
    sc.n_segments = 2;
    CHECK(near(compute_epoch_lr(sc, 0), Scalar(0.5)), "warmup 生效于 step_cosine e=0");
    CHECK(near(compute_epoch_lr(sc, 2), Scalar(1.0)), "step_cosine 退火起点接 warmup 末");
}

void test_lr_per_epoch_escape()
{
    LrScheduleConfig c;
    c.base_lr = Scalar(1.0);
    c.schedule = "cosine";          // 应被 lr_per_epoch 完全覆盖
    c.lr_per_epoch = {Scalar(1e-3), Scalar(5e-4), Scalar(2.5e-4)};
    CHECK(near(compute_epoch_lr(c, 0), Scalar(1e-3)), "lr_per_epoch[0] 覆盖 schedule");
    CHECK(near(compute_epoch_lr(c, 1), Scalar(5e-4)), "lr_per_epoch[1]");
    CHECK(near(compute_epoch_lr(c, 2), Scalar(2.5e-4)), "lr_per_epoch[2]");
    // 越界取末值（不再变化）
    CHECK(near(compute_epoch_lr(c, 99), Scalar(2.5e-4)), "lr_per_epoch 越界取末值");
}

void test_step_lr()
{
    StepLrScheduleConfig s;
    s.base_lr = Scalar(1.0);
    s.min_lr = Scalar(0.0);
    s.warmup_steps = 4;
    s.total_steps = 10;
    s.cosine = true;
    CHECK(near(compute_step_lr(s, 0), Scalar(0.25)), "step warmup s=0 = base/4");
    CHECK(near(compute_step_lr(s, 3), Scalar(1.0)), "step warmup 末 = base");
    CHECK(near(compute_step_lr(s, 4), Scalar(1.0)), "step 退火起点 = base");
    CHECK(near(compute_step_lr(s, 10), Scalar(0.0)), "step 终点 = min_lr");
    // 无余弦：预热后恒定
    s.cosine = false;
    CHECK(near(compute_step_lr(s, 9), Scalar(1.0)), "step 无余弦预热后恒 base");
}
} // namespace

int main()
{
    std::printf("lr_scheduler_test（epoch/step 学习率调度；issue #31）\n");
    test_constant();
    test_cosine();
    test_multistep();
    test_step_cosine();
    test_warmup_uniform();
    test_lr_per_epoch_escape();
    test_step_lr();
    if (g_fail == 0)
    {
        std::printf("[结果] 全部通过\n");
        return 0;
    }
    std::printf("[结果] %d 项失败\n", g_fail);
    return 1;
}
