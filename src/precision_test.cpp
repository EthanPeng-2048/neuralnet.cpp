// ── precision_test — 多精度合并测试（门面侧）───────────────────────────────
// 合并：tensor_precision_test + f16_compute_test + precision_profile_test
// 覆盖：张量精度属性 / f16 计算验证 / PrecisionProfile 配方语义
// （f16 类型系统的位级转换/舍入测试已下沉 base 子仓 base_precision_type_test）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>

// ── tensor_precision_test (variant 存储) ───────────────────────────────────
#define main test_tensor_precision
#define g_failures g_fail_tensor_precision
#include "tensor_precision_test.cpp"
#undef g_failures
#undef main

// ── f16_compute_test (f16 计算验证) ────────────────────────────────────────
#define main test_f16_compute
#define g_failures g_fail_f16_compute
#define test_error_paths test_f16_error_paths
#include "f16_compute_test.cpp"
#undef test_error_paths
#undef g_failures
#undef main

// ── precision_profile_test (PrecisionProfile 配方语义：--f16 = param/compute f16) ─
#define main test_precision_profile
#define g_failures g_fail_precision_profile
#include "precision_profile_test.cpp"
#undef g_failures
#undef main

// ── f16 转换快路径一致性（P2-8）────────────────────────────────────────────
// 目标含 F16C 时 `float_to_half_bits` / `half_bits_to_float` **运行时**走硬件
// 单指令；**常量求值**仍走软件位操作（`std::is_constant_evaluated()` 分流）。
// 本段用同一组边界值把两条路径逐一对比：constexpr 表项在编译期求值（软件），
// 循环内的调用在运行期求值（硬件）→ 断言**逐位一致**。
// 覆盖：±0、次正规、2^-25 平局（RNE）、2^-14 进位、65504/65519/65520/2^16
// 上溢边界、±inf、NaN（含负载）、以及一半值的往返。
namespace
{
constexpr std::uint16_t soft_h(float v)
{ return nn::detail::float_to_half_bits(v); }
constexpr float soft_f(std::uint16_t h)
{ return nn::detail::half_bits_to_float(h); }

struct HalfCase
{
    float v;
    std::uint16_t soft;   // 编译期（软件路径）结果
};
constexpr HalfCase kHalfCases[] = {
    {0.0f, soft_h(0.0f)},
    {-0.0f, soft_h(-0.0f)},
    {1.0f, soft_h(1.0f)},
    {-1.5f, soft_h(-1.5f)},
    {0.1f, soft_h(0.1f)},
    {2.0f / 3.0f, soft_h(2.0f / 3.0f)},          // 非精确表示
    {1.0f / 3.0f, soft_h(1.0f / 3.0f)},          // 非精确表示（RNE）
    {6.103515625e-05f, soft_h(6.103515625e-05f)},   // 2^-14 最小正规
    {5.960464477539063e-08f, soft_h(5.960464477539063e-08f)},  // 2^-24 最小次正规
    {2.9802322387695312e-08f, soft_h(2.9802322387695312e-08f)}, // 2^-25 = 平局 → 0
    {65504.0f, soft_h(65504.0f)},                 // f16 最大有限值
    {65519.0f, soft_h(65519.0f)},                 // 仍舍入到 65504
    {65520.0f, soft_h(65520.0f)},                 // 舍入到 inf 的边界
    {65536.0f, soft_h(65536.0f)},                 // 2^16 → inf
    {1e30f, soft_h(1e30f)},
    {-1e30f, soft_h(-1e30f)},
    {std::numeric_limits<float>::infinity(), soft_h(std::numeric_limits<float>::infinity())},
    {-std::numeric_limits<float>::infinity(), soft_h(-std::numeric_limits<float>::infinity())},
    {std::bit_cast<float>(0x7FC00000u), soft_h(std::bit_cast<float>(0x7FC00000u))},
    {std::bit_cast<float>(0xFFC00000u), soft_h(std::bit_cast<float>(0xFFC00000u))},
    {std::bit_cast<float>(0x7F800001u), soft_h(std::bit_cast<float>(0x7F800001u))},
    {std::bit_cast<float>(0x00000001u), soft_h(std::bit_cast<float>(0x00000001u))},
    {std::bit_cast<float>(0x807FFFFFu), soft_h(std::bit_cast<float>(0x807FFFFFu))},
};

// f16 → f32 侧：软件路径恒精确，硬件同；用一组位型对比
constexpr std::uint16_t kHalfBits[] = {0x0000u, 0x8000u, 0x0001u, 0x03FFu, 0x0400u,
                                       0x3555u, 0x3C00u, 0x3C01u, 0x7BFFu, 0x7C00u,
                                       0xFC00u, 0x7E00u, 0xFE00u, 0x7C01u, 0xFFFFu};
} // namespace

namespace
{
int test_f16_fastpath_parity()
{
    int bad = 0;
    std::puts("=== f16 转换快路径一致性（P2-8：硬件路径 vs 软件参考）===");
    for (const HalfCase& c : kHalfCases)
    {
        const std::uint16_t hw = nn::detail::float_to_half_bits(c.v);   // 运行期
        if (hw != c.soft)
        {
            std::printf("  [FAIL] f32→f16 位型不一致: v=%.9g hw=0x%04x soft=0x%04x\n",
                        static_cast<double>(c.v), hw, c.soft);
            ++bad;
        }
    }
    for (const std::uint16_t h : kHalfBits)
    {
        const float hw = nn::detail::half_bits_to_float(h);             // 运行期
        if (std::bit_cast<std::uint32_t>(hw) != std::bit_cast<std::uint32_t>(soft_f(h)))
        {
            std::printf("  [FAIL] f16→f32 位型不一致: h=0x%04x\n", h);
            ++bad;
        }
    }
    std::printf("  %s（%zu 条 f32→f16 + %zu 条 f16→f32，逐位一致）\n",
                bad == 0 ? "[PASS]" : "[FAIL]", sizeof(kHalfCases) / sizeof(kHalfCases[0]),
                sizeof(kHalfBits) / sizeof(kHalfBits[0]));
    return bad;
}
} // namespace

int main()
{
    int failures = 0;

    std::puts("=== tensor_precision (variant storage) ===");
    failures += test_tensor_precision();

    std::puts("=== f16_compute (f16 matmul + cast + training) ===");
    failures += test_f16_compute();

    std::puts("=== precision_profile (--f16 = param/compute f16) ===");
    failures += test_precision_profile();

    failures += test_f16_fastpath_parity();

    std::printf("\nprecision_test: %d failure(s)\n", failures);
    return failures != 0 ? 1 : 0;
}
