// ── precision_test.cpp — T2：f16 转换往返 / round-half-to-even 舍入（docs/23 §13）──
// 验收标准（§12.2-2）：
//   1. 全 65536 个 f16 位模式：f16 → f32 → f16 往返位型不变（NaN 例外：
//      往返后必须仍是 quiet NaN——载荷可归一化）
//   2. 密集 f32 采样（均匀 + 对数网格 + 随机）：f16(float) 满足
//      round-half-to-even 保证 |float(f16(v)) - v| <= 0.5 * ulp(f16(v))
//   3. 边界值：±65504 / ±65536 / 65505（舍入进位溢出）/ 2^-14 / 2^-15 /
//      2^-24 / 2^-25（tie-to-even → 0）/ ±0 / inf / NaN 传播
//   4. RHE 中点值（normal + denormal 区）
//   5. f16 算术 = f32 参考 + 舍入（§7.2 逐元素形式）
//   6. 错误路径：BF16 / F64 → 清晰 Result 报错（§12.2-10 前置）
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "neuralnet.cpp/precision.hpp"

namespace
{

int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, (msg)); \
            ++g_failures; \
        } \
    } while (0)

// f16 位型 → f32（测试内独立引用实现，不复用被测代码）
float ref_half_to_float(std::uint16_t h)
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t e = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    const std::uint32_t m = static_cast<std::uint32_t>(h) & 0x3FFu;
    if (e == 0)
    {
        if (m == 0)
            return sign ? -0.0f : 0.0f;
        // value = m * 2^-24
        return static_cast<float>(static_cast<double>(m) * std::ldexp(1.0, -24));
    }
    if (e == 31)
    {
        if (m == 0)
            return sign ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
        return std::numeric_limits<float>::quiet_NaN();
    }
    // value = (1 + m/1024) * 2^(e-15)
    return static_cast<float>((1.0 + static_cast<double>(m) / 1024.0) * std::ldexp(1.0, static_cast<int>(e) - 15))
         * (sign ? -1.0f : 1.0f);
}

// f16 位型的 ulp（normal: 2^((e-15)-10) = 2^(e-25)；denormal: 2^-24）
// ⚠ 曾误写成 2^(e-10)（大 2^15 倍）→ RHE 容差比被测值本身还大，
//   float_to_half_bits 的次正规 UB 窗口（exp ∈ [-45,-33]，已在 precision.hpp
//   修复）产出的垃圾 half（如 0x4000=2.0）在该容差下**永远测不出来**。
double ref_ulp(std::uint16_t h)
{
    const std::uint32_t e = (h >> 10) & 0x1Fu;
    if (e == 0)
        return std::ldexp(1.0, -24);
    if (e == 31)
        return 0.0;
    return std::ldexp(1.0, static_cast<int>(e) - 25);
}

constexpr bool is_f16_nan(std::uint16_t h)
{
    return ((h >> 10) & 0x1Fu) == 31 && (h & 0x3FFu) != 0;
}

constexpr bool is_f16_quiet_nan(std::uint16_t h)
{
    return is_f16_nan(h) && (h & 0x400u) != 0;
}

void test_full_bitroundtrip()
{
    std::printf("  [1] 全位域往返 h → f32 → f16 (65536 模式)");
    for (std::uint32_t h = 0; h < 65536; ++h)
    {
        const float f = nn::detail::half_bits_to_float(static_cast<std::uint16_t>(h));
        const std::uint16_t h2 = nn::detail::float_to_half_bits(f);
        if (h2 == static_cast<std::uint16_t>(h))
            continue;
        // 唯一允许的差异：NaN 载荷归一化为 quiet NaN
        if (is_f16_nan(static_cast<std::uint16_t>(h)) && is_f16_quiet_nan(h2))
            continue;
        CHECK(false, "f16→f32→f16 往返位型改变");
        if (g_failures > 20) break;
    }
    std::printf(" done\n");
}

void test_dense_sample_rhe()
{
    std::printf("  [2] 密集 f32 采样 RHE 保证 |err| <= 0.5 ulp");
    auto check_rhe = [](float v)
    {
        if (std::isnan(v) || std::isinf(v))
            return;
        const nn::f16 h{v};
        const float f = static_cast<float>(h);
        const double ulp = ref_ulp(h.bits());
        const double err = std::fabs(static_cast<double>(f) - static_cast<double>(v));
        if (h.bits() == 0)
        {
            // 舍入到 0：|v| 必须 <= 2^-25（tie-to-even 边界内）
            if (std::fabs(v) > std::ldexp(1.0f, -25))
            {
                std::fprintf(stderr, "    v=%g 舍入到 0 但 |v| > 2^-25\n", v);
                ++g_failures;
            }
            return;
        }
        if (std::isinf(f))
        {
            // 溢出：v 必须 >= 65504 + 0.5*ulp(65504) = 65504 + 16（ulp = 32；
            // 中点 tie → 尾数进位 → 溢出 inf；numpy 对拍确认 65519→65504 / 65520→inf）
            if (std::fabs(v) < 65504.0f + 16.0f)
            {
                std::fprintf(stderr, "    v=%g 溢出为 inf 但未越过舍入边界\n", v);
                ++g_failures;
            }
            return;
        }
        if (err > 0.5 * ulp + 1e-30)
        {
            std::fprintf(stderr, "    v=%g → f16 bits=0x%04X (f=%g) err=%g > 0.5ulp=%g\n",
                         v, h.bits(), f, err, 0.5 * ulp);
            ++g_failures;
        }
    };

    // 均匀网格（含 0 附近）
    for (float v = -1024.0f; v <= 1024.0f; v += 0.25f)
        check_rhe(v);
    // 对数网格（跨全部指数）
    for (int e = -30; e <= 30; ++e)
    {
        const double base = std::ldexp(1.0, e);
        for (double frac = 0.0; frac < 1.0; frac += 0.01)
        {
            check_rhe(static_cast<float>(base * (1.0 + frac)));
            check_rhe(static_cast<float>(-base * (1.0 + frac)));
        }
    }
    // 随机（固定种子，确定性）
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (int i = 0; i < 200000; ++i)
        check_rhe(static_cast<float>(dist(rng)));
    // 超小值（denormal / underflow 区）
    for (int e = -60; e >= -20; --e)
        for (double frac = 0.0; frac < 1.0; frac += 0.05)
            check_rhe(static_cast<float>(std::ldexp(1.0, e) * (0.5 + frac)));
    std::printf(" done\n");
}

void test_edge_values()
{
    std::printf("  [3] 边界值（±65504 / ±65536 / 65505 / 2^-14..2^-25 / ±0 / inf / NaN）");
    auto bits_of = [](float v) { return nn::detail::float_to_half_bits(v); };
    auto f_of = [](std::uint16_t h) { return nn::detail::half_bits_to_float(h); };

    // 最大有限值（精确表示）
    CHECK(bits_of(65504.0f) == 0x7BFF, "65504 → 0x7BFF");
    CHECK(bits_of(-65504.0f) == 0xFBFF, "-65504 → 0xFBFF");
    // 顶区（e=30）f16 ulp = 2^(15-10) = 32（numpy 对拍确认）：
    // 65504 + 15（< 中点 65520）→ 65504
    CHECK(bits_of(65519.0f) == 0x7BFF, "65519 → 65504");
    // 65520 = 65504 + 16（中点，tie-to-even → 尾数进位 → 溢出）→ +inf
    CHECK(bits_of(65520.0f) == 0x7C00, "65520 → +inf (tie 进位溢出)");
    CHECK(bits_of(65521.0f) == 0x7C00, "65521 → +inf");
    // 65504.5（= 65504 + 2^-1，未越中点）→ 65504
    CHECK(bits_of(65504.5f) == 0x7BFF, "65504.5 → 65504");
    // 2^16 及上溢
    CHECK(bits_of(65536.0f) == 0x7C00, "65536 → +inf");
    CHECK(bits_of(1e9f) == 0x7C00, "1e9 → +inf");
    CHECK(bits_of(-1e9f) == 0xFC00, "-1e9 → -inf");
    // 最小正规数 2^-14（精确，e=1, m=0）
    CHECK(bits_of(std::ldexp(1.0f, -14)) == 0x400, "2^-14 → 0x400");
    // 2^-15（denormal 区，精确：D = 2^-15 * 2^24 = 512）
    CHECK(bits_of(std::ldexp(1.0f, -15)) == 512, "2^-15 → 512");
    // 最小 denormal 2^-24（精确：D = 1）
    CHECK(bits_of(std::ldexp(1.0f, -24)) == 1, "2^-24 → 1");
    // 2^-25：0 与 2^-24 的中点 → tie-to-even → 0
    CHECK(bits_of(std::ldexp(1.0f, -25)) == 0, "2^-25 → 0 (tie-to-even)");
    // 2^-25 + 最小扰动 → 2^-24
    CHECK(bits_of(std::ldexp(1.0f, -25) + std::ldexp(1.0f, -48)) == 1,
          "2^-25 + 2^-48 → 2^-24");
    // 2^-26（远小于中点）→ 0
    CHECK(bits_of(std::ldexp(1.0f, -26)) == 0, "2^-26 → 0");
    // ── 次正规 UB 窗口回归（precision.hpp 的 exp<=-46 → exp<=-26 修复）──────
    // exp ∈ [-45,-33]（|v| ≈ 2.8e-14 ~ 1.2e-10）曾走 shift ≥ 32 的移位 UB →
    // 指数字段回绕成垃圾 half（0x4000=2.0、0xCCCD…）→ CPU f16 训练梯度被写成
    // 512/8192/11776/18432/NaN。这个区间**必须恒 flush 到 0**。
    for (int e = -60; e <= -26; ++e)
    {
        for (double frac = 0.0; frac < 1.0; frac += 0.05)
        {
            const float v = static_cast<float>(std::ldexp(1.0, e) * (0.5 + frac));
            const std::uint16_t b = bits_of(v);
            if (b != 0)
            {
                std::fprintf(stderr, "    v=%g (2^%d 区) → bits=0x%04X，应为 0\n", v, e, b);
                ++g_failures;
            }
            if (bits_of(-v) != 0x8000)
                ++g_failures;
        }
    }
    // 修复窗口上沿的两个代表值（曾分别产出 0x4000=2.0 与 0x3333）
    CHECK(bits_of(1.13687e-12f) == 0, "1.14e-12 → 0");
    CHECK(bits_of(8.44011e-11f) == 0, "8.44e-11 → 0");
    // 次正规区仍在窗口上方：2^-24 精确、2^-25 tie→0（上面已断言），再补 2^-26±
    CHECK(bits_of(std::ldexp(1.0f, -26) * 1.5f) == 0, "1.5×2^-26 → 0");
    // ±0
    CHECK(bits_of(0.0f) == 0, "+0 → 0");
    CHECK(bits_of(-0.0f) == 0x8000, "-0 → 0x8000");
    CHECK(f_of(0) == 0.0f && std::signbit(f_of(0)) == false, "0x0000 → +0.0f");
    CHECK(std::signbit(f_of(0x8000)) == true, "0x8000 → -0.0f");
    // inf
    CHECK(bits_of(std::numeric_limits<float>::infinity()) == 0x7C00, "+inf → 0x7C00");
    CHECK(bits_of(-std::numeric_limits<float>::infinity()) == 0xFC00, "-inf → 0xFC00");
    CHECK(std::isinf(f_of(0x7C00)) && f_of(0x7C00) > 0, "0x7C00 → +inf");
    // NaN 传播（载荷归一化为 quiet NaN 0x7E00 / 0xFE00）
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    CHECK(is_f16_nan(bits_of(qnan)), "NaN → NaN");
    CHECK(bits_of(qnan) == 0x7E00, "quiet NaN → 0x7E00");
    CHECK(bits_of(-qnan) == 0xFE00, "-quiet NaN → 0xFE00");
    CHECK(std::isnan(f_of(0x7E00)), "0x7E00 → NaN");
    // 特征常量
    CHECK(nn::f16::max_finite().bits() == 0x7BFF, "max_finite");
    CHECK(nn::f16::min_normal().bits() == 0x0400, "min_normal");
    CHECK(nn::f16::min_denormal().bits() == 0x0001, "min_denormal");
    CHECK(static_cast<float>(nn::f16::max_finite()) == 65504.0f, "float(max_finite) == 65504");
    CHECK(static_cast<float>(nn::f16::min_normal()) == std::ldexp(1.0f, -14), "float(min_normal) == 2^-14");
    std::printf(" done\n");
}

void test_rhe_midpoints()
{
    std::printf("  [4] RHE 中点值（normal / denormal 区）");
    auto bits_of = [](float v) { return nn::detail::float_to_half_bits(v); };
    // [1,2) 区间 ulp = 2^-10：
    // 1 + 2^-11（1.0 与 1+2^-10 的中点）→ tie-to-even → m=0（偶）→ 1.0
    CHECK(bits_of(1.0f + std::ldexp(1.0f, -11)) == 0x3C00, "1+2^-11 → 1.0 (偶数尾数)");
    // 1 + 2^-10 + 2^-11（m=1 与 m=2 的中点）→ m=2（偶）
    CHECK(bits_of(1.0f + std::ldexp(1.0f, -10) + std::ldexp(1.0f, -11)) == 0x3C02,
          "1+2^-10+2^-11 → m=2 (偶数尾数)");
    // 中点 ± 最小扰动方向
    CHECK(bits_of(1.0f + std::ldexp(1.0f, -11) - std::ldexp(1.0f, -20)) == 0x3C00,
          "中点下方 → 1.0");
    CHECK(bits_of(1.0f + std::ldexp(1.0f, -11) + std::ldexp(1.0f, -20)) == 0x3C01,
          "中点上方 → m=1");
    // denormal 区：2^-24 与 2*2^-24 的中点 = 3*2^-25 → tie-to-even → 2*2^-24 (D=2, 偶)
    CHECK(bits_of(std::ldexp(3.0f, -25)) == 2, "3*2^-25 → D=2 (偶数)");
    // 2^-24 与 0 的中点 = 2^-25 → 0（已在 [3] 覆盖，再断言一次）
    CHECK(bits_of(std::ldexp(1.0f, -25)) == 0, "2^-25 → 0");
    // denormal 进位到正规区：(1023.5) * 2^-24 的中点 = 1024 * 2^-24 = 2^-14 → 0x400
    CHECK(bits_of(static_cast<float>(1023.5 * std::ldexp(1.0, -24))) == 0x400,
          "denormal 舍入进位 → 最小正规数");
    std::printf(" done\n");
}

void test_arithmetic_semantics()
{
    std::printf("  [5] f16 算术 = f32 参考 + 舍入（§7.2）");
    using nn::f16;
    // RHE 误差上界断言：|float(r) - f32参考| <= 0.5 * ulp(r)（r 为 f16 结果）
    auto ulp_of = [](std::uint16_t bits) -> double
    {
        const std::uint32_t e = (bits >> 10) & 0x1Fu;
        if (e == 0) return std::ldexp(1.0, -24);       // denormal ulp
        if (e == 31) return 0.0;                        // inf/NaN
        return std::ldexp(1.0, static_cast<int>(e) - 25); // 2^(e-10-15)
    };
    auto rhe = [ulp_of](const f16& r, float ref, const char* what)
    {
        const float f = static_cast<float>(r);
        if (std::isinf(ref))
        {
            if (!std::isinf(f))
            {
                std::fprintf(stderr, "    %s: r=0x%04X 非 inf 但参考 inf\n", what, r.bits());
                ++g_failures;
            }
            return;
        }
        const double ulp = ulp_of(r.bits());
        if (r.bits() == 0)
        {
            if (std::fabs(ref) > std::ldexp(1.0f, -25))
            {
                std::fprintf(stderr, "    %s: r=0 但 |ref| > 2^-25\n", what);
                ++g_failures;
            }
            return;
        }
        if (std::fabs(static_cast<double>(f) - static_cast<double>(ref)) > 0.5 * ulp + 1e-30)
        {
            std::fprintf(stderr, "    %s: r=0x%04X (f=%g) ref=%g 超 0.5ulp=%g\n",
                         what, r.bits(), f, ref, 0.5 * ulp);
            ++g_failures;
        }
    };

    const f16 a{0.1f};
    const f16 b{0.2f};
    // §7.2 形式化定义：f16(a) op f16(b) := f16(f32(a) op f32(b))
    rhe(a + b, static_cast<float>(a) + static_cast<float>(b), "a+b");
    rhe(a - b, static_cast<float>(a) - static_cast<float>(b), "a-b");
    rhe(a * b, static_cast<float>(a) * static_cast<float>(b), "a*b");
    rhe(a / b, static_cast<float>(a) / static_cast<float>(b), "a/b");
    // 可精确表示的值（f16 网格上）→ 无舍入
    CHECK(static_cast<float>(f16{0.5f} + f16{0.25f}) == 0.75f, "0.5+0.25 = 0.75 (精确)");
    CHECK(static_cast<float>(f16{1.0f} + f16{2.0f}) == 3.0f, "1+2 = 3 (精确)");
    CHECK(static_cast<float>(f16{3.0f} - f16{1.5f}) == 1.5f, "3-1.5 = 1.5 (精确)");
    CHECK(static_cast<float>(f16{0.5f} * f16{0.5f}) == 0.25f, "0.5*0.5 = 0.25 (精确)");
    // 算术溢出：65504 + 16 = 65520 = tie → +inf（numpy 对拍确认）
    const f16 big{65504.0f};
    const f16 step{16.0f};
    CHECK(std::isinf(static_cast<float>(big + step)), "65504+16 → +inf (算术 tie 溢出)");
    CHECK(std::isinf(static_cast<float>(big + big)), "65504 + 65504 = +inf");
    // 就地运算
    f16 x{1.0f};
    x += 0.5f;
    CHECK(static_cast<float>(x) == 1.5f, "x += 0.5f");
    x *= 0.5f;
    CHECK(static_cast<float>(x) == 0.75f, "x *= 0.5f");
    // 标量混合（标量 f32 在算子边界参与参考计算，§8.2）
    const f16 y{2.0f};
    CHECK(static_cast<float>(y * 0.25f) == 0.5f, "f16 * float");
    CHECK(static_cast<float>(0.25f * y) == 0.5f, "float * f16");
    // 比较
    CHECK(f16{1.0f} < f16{2.0f}, "1 < 2");
    CHECK(f16{2.0f} >= f16{2.0f}, "2 >= 2");
    const f16 n1{std::numeric_limits<float>::quiet_NaN()};
    CHECK(n1 != n1, "NaN != NaN");
    CHECK(!(n1 < f16{0.0f}) && !(f16{0.0f} < n1), "NaN 比较均为 false");
    std::printf(" done\n");
}

void test_error_paths()
{
    std::printf("  [6] 错误路径：BF16 / F64 → 清晰 Result 报错");
    CHECK(!nn::check_precision_supported(nn::Precision::F16).has_value() == false, "F16 通过");
    CHECK(nn::check_precision_supported(nn::Precision::F32).has_value(), "F32 通过");
    {
        auto r = nn::check_precision_supported(nn::Precision::BF16);
        CHECK(!r.has_value(), "BF16 被拒");
        CHECK(r.error().message.find("精度未实现") != std::string::npos, "BF16 错误信息");
        CHECK(r.error().message.find("bf16") != std::string::npos, "BF16 错误含精度名");
    }
    {
        auto r = nn::check_precision_supported(nn::Precision::F64);
        CHECK(!r.has_value(), "F64 被拒");
        CHECK(r.error().message.find("f64") != std::string::npos, "F64 错误含精度名");
    }
    // 序列化 tag 互锁（v5，§11.3）
    CHECK(nn::precision_tag(nn::Precision::F32) == 0, "tag(F32)=0");
    CHECK(nn::precision_tag(nn::Precision::F64) == 1, "tag(F64)=1 (占位)");
    CHECK(nn::precision_tag(nn::Precision::F16) == 2, "tag(F16)=2");
    CHECK(nn::precision_tag(nn::Precision::BF16) == 3, "tag(BF16)=3 (保留)");
    CHECK(nn::precision_from_tag(nn::precision_tag(nn::Precision::F16)) == nn::Precision::F16, "tag 往返 F16");
    CHECK(nn::precision_from_tag(nn::precision_tag(nn::Precision::F32)) == nn::Precision::F32, "tag 往返 F32");
    // elem / acc
    CHECK((std::is_same_v<nn::elem<nn::Precision::F16>, nn::f16>), "elem<F16> = f16");
    CHECK((std::is_same_v<nn::elem<nn::Precision::F32>, float>), "elem<F32> = float");
    CHECK(sizeof(nn::elem<nn::Precision::F16>) == 2, "f16 2 字节");
    CHECK(sizeof(nn::elem<nn::Precision::F32>) == 4, "f32 4 字节");
    std::printf(" done\n");
}

void test_reference_consistency()
{
    std::printf("  [7] 与被测代码独立的参考实现对拍（随机 100k）");
    std::mt19937 rng(0xBEEF);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < 100000; ++i)
    {
        const float v = static_cast<float>(dist(rng));
        const nn::f16 h{v};
        const float f = static_cast<float>(h);
        // 参考：f16 值集 = {m*2^-24, (1+m/1024)*2^(e-15), inf/nan}；
        // 断言：f 必须是 f16 可表示值（往返一致）且误差 <= 0.5 ulp
        const nn::f16 h2{f};
        CHECK(h.bits() == h2.bits(), "f32 往返位型一致");
        if (!std::isnan(v) && !std::isinf(v) && !std::isinf(f) && h.bits() != 0)
        {
            const double ulp = ref_ulp(h.bits());
            if (std::fabs(static_cast<double>(f) - static_cast<double>(v)) > 0.5 * ulp + 1e-30)
                CHECK(false, "RHE 误差超限");
        }
        // 独立参考实现的对拍（仅 f16→f32 方向；方向已由 [1][2] 覆盖）
        const float ref = ref_half_to_float(h.bits());
        if (std::isnan(ref) || std::isnan(f))
            CHECK(std::isnan(f) == std::isnan(ref), "NaN 一致");
        else
            CHECK(f == ref, "f16→f32 与独立参考一致");
        if (g_failures > 50)
        {
            std::fprintf(stderr, "  失败过多，提前终止 [7]\n");
            break;
        }
    }
    std::printf(" done\n");
}

} // namespace

int main()
{
    std::printf("precision_test (docs/23 T2：f16 转换 / 舍入)\n");
    test_full_bitroundtrip();
    test_dense_sample_rhe();
    test_edge_values();
    test_rhe_midpoints();
    test_arithmetic_semantics();
    test_error_paths();
    test_reference_consistency();
    if (g_failures == 0)
    {
        std::printf("precision_test: ALL PASSED\n");
        return 0;
    }
    std::printf("precision_test: %d FAILURES\n", g_failures);
    return 1;
}
