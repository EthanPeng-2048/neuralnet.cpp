// ── precision_profile_test — PrecisionProfile 配方语义 ──────────────────────
// 验收：CLI `--f16` 必须映射到 **f16 存储**（param + compute 全 F16），而不是
// master-weights 混合配方（F32 主权重 + f16 计算）。
// 历史问题：`--f16` 曾等价于 profile_master_weights()，用户以为开了全 f16，
// 实际**参数**仍是 F32——本测试把该语义钉死。
//
// 2026-09 修订：`--f16` 的语义为 profile_f16() = {F16, F16, F32, F32}。
// 为何 stable/optimizer 不取 F16（实测证据，见 precision.hpp 注释）：
//   · optimizer=F16：Adam 的 v ≈ g² ~ 1e-10 在 f16 下溢到 0 → 更新爆炸
//     （loss 7.9 → 3.6e4）；
//   · stable=F16：CE 链 f16 下 ~200 步 NaN；
//   · 四字段全 F16（profile_all_f16）：loss 恒定（更新被 f16 舍入吃光）。
// profile_all_f16() 保留为显式实验配方（需要 loss scaling + in-kernel f16）。
#include <cstdio>
#include <string>

#include "neuralnet.cpp/precision.hpp"
#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{
int g_failures = 0;


bool is_all(nn::PrecisionProfile p, nn::Precision v)
{
    return p.param == v && p.compute == v && p.stable == v && p.optimizer == v;
}
} // namespace

int main()
{
    using nn::Precision;
    using nn::PrecisionProfile;

    const PrecisionProfile f32 = nn::profile_f32();
    CHECK(is_all(f32, Precision::F32), "profile_f32 应四字段全 F32");
    CHECK(nn::is_profile_f32(f32), "is_profile_f32(profile_f32) 应为 true");

    // ── CLI `--f16` 语义：param + compute = F16（f16 存储）──────────────
    const PrecisionProfile f16 = nn::profile_f16();
    CHECK(f16.param == Precision::F16 && f16.compute == Precision::F16,
          "--f16 语义：param 与 compute 必须都是 F16（f16 存储，非 f32 主权重）");
    CHECK(f16.stable == Precision::F32 && f16.optimizer == Precision::F32,
          "--f16 语义：stable / optimizer 留 F32（数值敏感链，见 precision.hpp）");
    CHECK(!nn::is_profile_f32(f16), "profile_f16 不是全 f32（适配层必须启用）");

    const PrecisionProfile all16 = nn::profile_all_f16();
    CHECK(is_all(all16, Precision::F16), "profile_all_f16 应四字段全 F16（实验配方）");

    const PrecisionProfile mw = nn::profile_master_weights();
    CHECK(mw.param == Precision::F32 && mw.compute == Precision::F16 &&
          mw.stable == Precision::F32 && mw.optimizer == Precision::F32,
          "profile_master_weights 应为 F32/F16/F32/F32（混合，参数仍 f32）");

    // 三个配方两两可区分：防止日后有人把 --f16 又改回混合/全 f16
    const auto same = [](const PrecisionProfile& a, const PrecisionProfile& b)
    {
        return a.param == b.param && a.compute == b.compute &&
               a.stable == b.stable && a.optimizer == b.optimizer;
    };
    CHECK(!same(f16, mw), "profile_f16 与 profile_master_weights 必须可区分");
    CHECK(!same(f16, all16), "profile_f16 与 profile_all_f16 必须可区分");
    CHECK(!same(mw, all16), "profile_master_weights 与 profile_all_f16 必须可区分");

    CHECK(std::string(nn::precision_name(Precision::F16)) == "f16",
          "precision_name(F16) 应为 \"f16\"");
    CHECK(std::string(nn::precision_name(Precision::F32)) == "f32",
          "precision_name(F32) 应为 \"f32\"");

    if (g_failures == 0)
        std::puts("  [PASS] PrecisionProfile 配方语义（--f16 = param/compute f16）");
    return g_failures;
}