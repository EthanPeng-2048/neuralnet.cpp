#pragma once

// ── precision.hpp — 多精度类型系统（docs/development/05-mixed-precision.md）──────────────
// L1 语义层（设备无关）：
//   - Precision 枚举：F16 / F32（BF16 / F64 为保留值，Phase 1 使用 → 清晰报错）
//   - nn::f16：IEEE binary16 值类型（uint16 位布局，与 GPU R16F 内存布局一致，
//     little-endian → 同精度 CPU↔GPU 传输 = 原始字节拷贝）
//   - elem<P>：P → 元素类型（f16 / float）
//   - PrecisionProfile：模型级精度配置（param / compute / stable / optimizer）
//
// P 精度算术的形式化定义（docs/23 §7.2，本文最核心的语义锚）：
//   对 P ≠ F32：P 精度算术 := 以 f32 参考精度计算 + 每个算子输出舍入到 P
//                （round-half-to-even）；matmul / 归约类算子额外：累加精度 = max(P, f32) = f32
//   对 P = F32：参考即自身，不舍入（= 现状行为）
//
// 该定义使硬件路径与兼容路径**语义等价**（Q4）：硬件 f16 运算的输出本来就
// 舍入到 f16；tensor core 点积用 f32 累加。差异仅在归约内累加顺序 →
// 跨设备同精度**容差内相等，不字节一致**（§11.2）。
//
// Scalar（float）**保留**，含义收窄为：参考精度（Q4 的 f32 参考）+ 宿主
// 标量类型（lr / wd / epsilon / alpha 等 f32 常量）。
// ─────────────────────────────────────────────────────────────────────────

#include <bit>       // std::bit_cast
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core_errors.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// Precision — 精度枚举（D11：F16 < F32 提升序；BF16 / F64 保留）
// ══════════════════════════════════════════════════════════════════════════
enum class Precision : std::uint8_t
{
    F16  = 0,   // IEEE half（binary16），2 字节
    F32  = 1,   // 参考精度（reference）
    BF16 = 2,   // 保留（Phase 2；使用 → 清晰报错）
    F64  = 3,   // 保留（使用 → 清晰报错）
};

// ── 精度工具 ────────────────────────────────────────────────────────────────

// 提升序 F16 < F32（D11）：Auto 推导 = max(操作数精度)（§8.1）
[[nodiscard]] constexpr Precision max_precision(Precision a, Precision b) noexcept
{
    return a > b ? a : b;
}

// 元素字节数（f16=2 / f32=4；保留值返回 0，仅供能力/诊断，不用于存储）
[[nodiscard]] constexpr std::size_t precision_bytes(Precision p) noexcept
{
    switch (p)
    {
        case Precision::F16:  return 2;
        case Precision::F32:  return 4;
        default:              return 0;  // BF16 / F64：Phase 1 未实现
    }
}

[[nodiscard]] constexpr const char* precision_name(Precision p) noexcept
{
    switch (p)
    {
        case Precision::F16:  return "f16";
        case Precision::F32:  return "f32";
        case Precision::BF16: return "bf16";
        case Precision::F64:  return "f64";
    }
    return "?";
}

// Phase 1 支持检查（D11）：BF16 / F64 → 清晰 Result 报错"精度未实现"
// 用于一切"精度作为运行期输入"的入口（张量创建、cast、序列化 tag 回读…）
[[nodiscard]] inline Result<void> check_precision_supported(Precision p)
{
    if (p == Precision::BF16 || p == Precision::F64)
    {
        return std::unexpected(Error{
            std::string("精度未实现（Phase 1 仅支持 f16 / f32）: ") + precision_name(p)});
    }
    return {};
}

// 序列化 tag（v5，§11.3）：沿用 v4 数值，避免 v4 文件被误读：
//   0=f32、1=f64（占位）、2=f16、3=bf16（保留）
[[nodiscard]] constexpr std::uint8_t precision_tag(Precision p) noexcept
{
    switch (p)
    {
        case Precision::F32:  return 0;
        case Precision::F64:  return 1;
        case Precision::F16:  return 2;
        case Precision::BF16: return 3;
    }
    return 0;
}

[[nodiscard]] constexpr Precision precision_from_tag(std::uint8_t tag) noexcept
{
    switch (tag)
    {
        case 0: return Precision::F32;
        case 1: return Precision::F64;
        case 2: return Precision::F16;
        default: return Precision::BF16;
    }
}

// ══════════════════════════════════════════════════════════════════════════
// CPU f16 硬件能力（编译期常量，§7.1）
//
// 检测 CPU 是否具有原生 f16 算术指令：
//   - x86_64：AVX512-FP16（__AVX512FP16__）或 AVX10（包含 FP16）
//   - ARM：ARMv8.2-A 半精度浮点（__ARM_FP16_FORMAT_IEEE + __ARM_NEON）
//   - 兼容路径：无原生 f16 ISA → f16 算术用 f32 参考 + 逐元素舍入（§7.2）
//
// F16C（__F16C__）：x86 硬件 f16↔f32 转换指令（_cvtph2ps/_cvtps2ph），
//   提供快速转换但不提供 f16 算术。Phase 1 的边界 cast 路径可用 F16C 加速。
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] constexpr bool cpu_has_f16_isa() noexcept
{
#if defined(__AVX512FP16__)
    return true;
#elif defined(__AVX10__)
    return true;  // AVX10 包含 FP16 指令
#elif defined(__ARM_FP16_FORMAT_IEEE) && defined(__ARM_NEON)
    return true;  // ARMv8.2-A 半精度浮点
#else
    return false;  // 兼容路径：f32 参考 + 舍入（§7.2）
#endif
}

// F16C：硬件 f16↔f32 转换（非算术，但加速边界 cast）
[[nodiscard]] constexpr bool cpu_has_f16c() noexcept
{
#if defined(__F16C__)
    return true;
#else
    return false;
#endif
}

// ══════════════════════════════════════════════════════════════════════════
// f16 — IEEE binary16 值类型
//
// 语义（§5.2）：
//   - trivially copyable、无异常
//   - 位布局与 GPU 内存中的 R16F 一致（little-endian）
//   - 一切 f16↔f32 转换与 f16 输出舍入统一 **round-half-to-even**
//     （与 IEEE / GPU 硬件一致，Q4 依赖此一致性）
//   - 算术：f32 参考计算 + 结果舍入到 f16（§7.2 形式化定义的逐元素形式）
//
// 数值事实：最大 65504，最小正规数 2^-14，最小规格化前数 2^-24，
// 尾数精度 10 位（相对精度 2^-10）。
// ══════════════════════════════════════════════════════════════════════════

// ── 位级转换（round-half-to-even，constexpr 可求值）──────────────────────
namespace detail
{

// f32 位型 → f16 位型（round-half-to-even）。
// 溢出 → ±inf；underflow 边界 2^-25 按 tie-to-even 舍入到 0。
[[nodiscard]] constexpr std::uint16_t float_to_half_bits(float f) noexcept
{
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127;  // 非偏置
    const std::uint32_t mant = bits & 0x7FFFFFu;

    if (exp == -128)
    {
        // ±0 与 f32 规格化前数（< 2^-126）→ f16 下均为 0
        return static_cast<std::uint16_t>(sign);
    }
    if (exp == 128)
    {
        // f32 inf（mant==0）→ ±inf；f32 NaN（mant!=0）→ quiet NaN
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant != 0 ? 0x200u : 0u));
    }
    if (exp > 15)
    {
        // 溢出（|v| >= 2^16 > 65504）→ ±inf
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (exp < -14)
    {
        // 结果 < 2^-14：规格化前区或 0。
        // D = round(value * 2^24) = round((2^23 + M) * 2^(exp+1))
        // 通用公式（shift = -exp-1）覆盖 exp ∈ [-46, -15]：
        //   exp = -25：shift = 24，2^-25（M=0）为 0 与 2^-24 的中点 → tie-to-even → 0；
        //              M>0（v > 2^-25）→ D ∈ (0.5, 1) → 1
        //   exp ≤ -26：D < 0.5（无 tie 可能）→ 恒 0
        if (exp <= -46)
        {
            // |v| ≤ 2^-46 → D < 2^-22 → 0；同时避免 shift ≥ 32 的 UB
            return static_cast<std::uint16_t>(sign);
        }
        const std::uint32_t full = 0x800000u | mant;          // 24 位尾数（隐含 1）
        const int shift = -exp - 1;                            // 14..24
        std::uint32_t d = full >> shift;                       // 候选规格化前尾数
        const std::uint32_t lsb = (full >> (shift - 1)) & 1u;  // 被舍弃的最高位
        const std::uint32_t rem = full & ((1u << (shift - 1)) - 1u);  // 更低位
        if (lsb != 0 && (rem != 0 || (d & 1u) != 0))
            ++d;                                               // round-half-to-even
        if (d == 1024)
        {
            // 舍入进位到最小正规数 2^-14
            return static_cast<std::uint16_t>(sign | 0x400u);  // e=1, m=0
        }
        return static_cast<std::uint16_t>(sign | d);
    }
    // 正规区：-14 <= exp <= 15。keep = 11 位尾数（隐含 1 + 10 位）
    const std::uint32_t full = 0x800000u | mant;
    std::uint32_t keep = full >> 13;                           // 1024..2047
    const std::uint32_t lsb = (full >> 12) & 1u;
    const std::uint32_t rem = full & 0xFFFu;
    if (lsb != 0 && (rem != 0 || (keep & 1u) != 0))
        ++keep;
    if (keep == 2048)
    {
        // 舍入进位：尾数变为 0，指数 +1
        if (exp == 15)
        {
            return static_cast<std::uint16_t>(sign | 0x7C00u); // 进位溢出 → ±inf
        }
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(exp + 1 + 15) << 10));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp + 15) << 10) | (keep & 0x3FFu));
}

// f16 位型 → f32（无损、精确）
[[nodiscard]] constexpr float half_bits_to_float(std::uint16_t h) noexcept
{
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t e = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    const std::uint32_t m = static_cast<std::uint32_t>(h) & 0x3FFu;

    std::uint32_t bits;
    if (e == 0)
    {
        if (m == 0)
        {
            bits = sign;  // ±0
        }
        else
        {
            // 规格化前：value = m * 2^-24。找最高置位位 p ∈ [0,9]：
            // m = 2^p + f（f = m mod 2^p）→ value = (1 + f/2^p) * 2^(p-24)
            // f32 偏置指数 = (p-24) + 127 = p + 103；尾数 M = f * 2^(23-p)
            std::uint32_t mm = m;
            int p = 0;
            while (mm > 1)
            {
                mm >>= 1;
                ++p;
            }
            const std::uint32_t f = (p > 0) ? (m & ((1u << p) - 1u)) : 0u;
            bits = sign
                 | (static_cast<std::uint32_t>(p + 103) << 23)
                 | ((p > 0) ? (f << (23 - p)) : 0u);
        }
    }
    else if (e == 31)
    {
        // inf（m=0）或 NaN（m≠0 → 转 quiet NaN，保留尾数信息）
        bits = sign | 0x7F800000u | (m << 13) | (m != 0 ? 0x00400000u : 0u);
    }
    else
    {
        // 正规：f32 偏置指数 = (e-15) + 127 = e + 112；尾数左移 13 位
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    return std::bit_cast<float>(bits);
}

} // namespace detail

class f16
{
public:
    using storage = std::uint16_t;

private:
    storage bits_ = 0;

public:
    // ── 构造 / 转换 ────────────────────────────────────────────────────────
    constexpr f16() noexcept = default;

    // 位模式构造（无舍入）
    constexpr explicit f16(storage bits) noexcept : bits_(bits) {}

    // f32 → f16（round-half-to-even）。隐式：f16 的存储语义天然含舍入
    // （"存 f16" = 舍入到 f16），代数层 `x = 3.0f` 等写法依赖此行为。
    constexpr f16(float v) noexcept : bits_(detail::float_to_half_bits(v)) {}

    // f16 → f32（无损；显式，避免意外提升）
    [[nodiscard]] constexpr explicit operator float() const noexcept
    {
        return detail::half_bits_to_float(bits_);
    }

    [[nodiscard]] constexpr storage bits() const noexcept { return bits_; }

    // ── 特征常量 ───────────────────────────────────────────────────────────
    [[nodiscard]] static constexpr f16 zero() noexcept { return f16{storage{0}}; }
    [[nodiscard]] static constexpr f16 one() noexcept { return f16{storage{0x3C00}}; }
    [[nodiscard]] static constexpr f16 inf() noexcept { return f16{storage{0x7C00}}; }
    [[nodiscard]] static constexpr f16 nan() noexcept { return f16{storage{0x7E00}}; }
    [[nodiscard]] static constexpr f16 max_finite() noexcept { return f16{storage{0x7BFF}}; }   // 65504
    [[nodiscard]] static constexpr f16 min_normal() noexcept { return f16{storage{0x0400}}; }    // 2^-14 (e=1, m=0)
    [[nodiscard]] static constexpr f16 min_denormal() noexcept { return f16{storage{0x0001}}; }  // 2^-24

    // ── 比较（经 f32 转换；NaN 语义与 f32 一致：NaN != 一切，NaN < 比较为 false）
    [[nodiscard]] constexpr bool operator==(const f16& o) const noexcept
    { return static_cast<float>(*this) == static_cast<float>(o); }
    [[nodiscard]] constexpr bool operator!=(const f16& o) const noexcept
    { return !(*this == o); }
    [[nodiscard]] constexpr bool operator<(const f16& o) const noexcept
    { return static_cast<float>(*this) < static_cast<float>(o); }
    [[nodiscard]] constexpr bool operator<=(const f16& o) const noexcept
    { return static_cast<float>(*this) <= static_cast<float>(o); }
    [[nodiscard]] constexpr bool operator>(const f16& o) const noexcept
    { return static_cast<float>(*this) > static_cast<float>(o); }
    [[nodiscard]] constexpr bool operator>=(const f16& o) const noexcept
    { return static_cast<float>(*this) >= static_cast<float>(o); }

    // ── 算术（§7.2：f32 参考计算 + 结果舍入到 f16）──────────────────────
    [[nodiscard]] constexpr f16 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr f16 operator-() const noexcept
    { return f16{-static_cast<float>(*this)}; }

    [[nodiscard]] constexpr f16 operator+(const f16& o) const noexcept
    { return f16{static_cast<float>(*this) + static_cast<float>(o)}; }
    [[nodiscard]] constexpr f16 operator-(const f16& o) const noexcept
    { return f16{static_cast<float>(*this) - static_cast<float>(o)}; }
    [[nodiscard]] constexpr f16 operator*(const f16& o) const noexcept
    { return f16{static_cast<float>(*this) * static_cast<float>(o)}; }
    [[nodiscard]] constexpr f16 operator/(const f16& o) const noexcept
    { return f16{static_cast<float>(*this) / static_cast<float>(o)}; }

    // 宿主标量混合（标量是 f32，§8.2：标量在算子边界参与 f32 参考计算）
    [[nodiscard]] constexpr f16 operator+(float s) const noexcept
    { return f16{static_cast<float>(*this) + s}; }
    [[nodiscard]] constexpr f16 operator-(float s) const noexcept
    { return f16{static_cast<float>(*this) - s}; }
    [[nodiscard]] constexpr f16 operator*(float s) const noexcept
    { return f16{static_cast<float>(*this) * s}; }
    [[nodiscard]] constexpr f16 operator/(float s) const noexcept
    { return f16{static_cast<float>(*this) / s}; }

    friend constexpr f16 operator+(float s, const f16& a) noexcept
    { return f16{s + static_cast<float>(a)}; }
    friend constexpr f16 operator-(float s, const f16& a) noexcept
    { return f16{s - static_cast<float>(a)}; }
    friend constexpr f16 operator*(float s, const f16& a) noexcept
    { return f16{s * static_cast<float>(a)}; }
    friend constexpr f16 operator/(float s, const f16& a) noexcept
    { return f16{s / static_cast<float>(a)}; }

    // ── 就地运算 ───────────────────────────────────────────────────────────
    f16& operator+=(const f16& o) { *this = *this + o; return *this; }
    f16& operator-=(const f16& o) { *this = *this - o; return *this; }
    f16& operator*=(const f16& o) { *this = *this * o; return *this; }
    f16& operator/=(const f16& o) { *this = *this / o; return *this; }
    f16& operator+=(float s) { *this = *this + s; return *this; }
    f16& operator-=(float s) { *this = *this - s; return *this; }
    f16& operator*=(float s) { *this = *this * s; return *this; }
    f16& operator/=(float s) { *this = *this / s; return *this; }

    // 从浮点值赋值 = 舍入到 f16 后存储（f16 存储的固有语义）
    f16& operator=(float v) { *this = f16{v}; return *this; }

    // ── std::plus<> / std::multiplies<> 等通过上述 operator 自然可用 ──────
};

static_assert(std::is_trivially_copyable_v<f16>, "f16 must be trivially copyable");
static_assert(sizeof(f16) == 2, "f16 must be 2 bytes (R16F 布局)");

// ══════════════════════════════════════════════════════════════════════════
// elem<P> — P → 元素类型（§5.2；Phase 1 仅实例化 F16 / F32）
// 保留值（BF16 / F64）无特化 → 实例化 elem<BF16> 是编译期错误，
// 运行期使用先被 check_precision_supported 拦截（清晰 Result 报错）。
// ══════════════════════════════════════════════════════════════════════════

using f32 = float;  // 参考精度别名（§5.2）

template <Precision P>
struct Elem;  // 无通用定义（保留值不可实例化）

template <>
struct Elem<Precision::F16> { using type = f16; };

template <>
struct Elem<Precision::F32> { using type = float; };

template <Precision P>
using elem = typename Elem<P>::type;

// 累加类型（§7.2：matmul / 归约累加 = max(P, f32)）：
// F32 → float（现状）；F16 → float（f32 累加）
template <Precision P>
using acc = float;

// ══════════════════════════════════════════════════════════════════════════
// PrecisionProfile — 模型级精度配置（§9.1，D10：默认全 F32 = 现状，零回归）
//
// 参数映射（D8）：
//   param     — 权重 / 嵌入表 / 参数存储精度（工厂创建 / 上传时指定）
//   compute   — 常规算子的显式 P（matmul / 逐元素 / gather / scan / 数据操作…）
//   stable    — 数值敏感算子的显式 P（softmax / LayerNorm / RMSNorm / loss，
//               含 loss backward 输出，D9；f16 溢出风险防线）
//   optimizer — 优化器状态（m / v / momentum）创建精度（f16 训练下必须 f32）
//
// 激活 / 梯度**不设独立参数**：按 Q3-A，算子输出存储精度 = 该算子的 P，
// 已被 compute / stable 完全决定（D8）。
// ══════════════════════════════════════════════════════════════════════════
struct PrecisionProfile
{
    Precision param     = Precision::F32;
    Precision compute   = Precision::F32;
    Precision stable    = Precision::F32;
    Precision optimizer = Precision::F32;
};

// 典型配方（§9.4）
inline constexpr PrecisionProfile profile_f32() noexcept
{
    return PrecisionProfile{};  // 全 F32 = 现状
}

// master-weights（经典混合精度）：
// f32 主权重 + f16 计算 + f32 稳定算子 + f32 优化器状态
inline constexpr PrecisionProfile profile_master_weights() noexcept
{
    return PrecisionProfile{
        /*param=*/     Precision::F32,
        /*compute=*/   Precision::F16,
        /*stable=*/    Precision::F32,
        /*optimizer=*/ Precision::F32};
}

// ── f16 训练配方（CLI `--f16` 的语义）─────────────────────────────────────
// {param=F16, compute=F16, stable=F32, optimizer=F32} = docs 05 §9.4 的
// "全 f16（激进）"行：**参数与激活全部 f16 存储**（这是"全 f16"的本意——
// 与 profile_master_weights 的区别正是 param 由 F32 变 F16），而数值敏感链
// （softmax / LayerNorm / loss）与优化器状态留在 f32。
//
// 为什么不把 stable / optimizer 也设成 F16 —— 有实测证据（本机 40HX，
// GPT d64/h4/L4/ff256、vocab 8208、seq 256、batch 8、adam lr 1e-3）：
//   · optimizer=F16（Adam 的 m/v 存 f16）：v ≈ g² ~ 1e-10 **下溢到 0** →
//     delta = lr·m/(√0+eps) 爆炸 → loss 从 7.9 发散到 3.6e4；
//   · stable=F16：CE 链在 f16 下 ~200 步后出现 NaN（f16 范围 65504 / 10 位
//     尾数不足以承载 vocab 级 log-sum-exp 与 log_softmax）；
//   · 两者同时 F16（profile_all_f16）：loss 恒定在 32.1（更新被 f16 舍入吃光）。
// 全四字段 f16 需要 loss scaling + in-kernel f16 归约（docs 05 §12.3），
// 当前不具备 → 保留 profile_all_f16() 供显式实验，不作 CLI 默认。
inline constexpr PrecisionProfile profile_f16() noexcept
{
    return PrecisionProfile{
        /*param=*/     Precision::F16,
        /*compute=*/   Precision::F16,
        /*stable=*/    Precision::F32,
        /*optimizer=*/ Precision::F32};
}

// 全 f16：四个字段全部 F16（激进实验配方；见上：当前数值上不可用于训练）。
// 注意：与 profile_master_weights（f32 主权重混合）区分——后者是旧的
// `--f16` 语义，曾造成"传了 --f16 却是混合精度"的误解。
inline constexpr PrecisionProfile profile_all_f16() noexcept
{
    return PrecisionProfile{
        /*param=*/     Precision::F16,
        /*compute=*/   Precision::F16,
        /*stable=*/    Precision::F16,
        /*optimizer=*/ Precision::F16};
}

// 便捷判定：全 F32 配置（= 迁移前行为）。CLI 用它决定是否启用
// PrecisionEngine 适配层（全 f32 时直通原生引擎，零开销零回归）。
[[nodiscard]] constexpr bool is_profile_f32(const PrecisionProfile& p) noexcept
{
    return p.param == Precision::F32 && p.compute == Precision::F32 &&
           p.stable == Precision::F32 && p.optimizer == Precision::F32;
}

} // namespace nn
