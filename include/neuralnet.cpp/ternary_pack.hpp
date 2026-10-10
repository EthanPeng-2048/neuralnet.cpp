#pragma once

// ── ternary_pack.hpp — 三值权重的打包布局与编解码（P2 / F3 共用件）──────────
//
// 布局（**码值映射与位宽对齐 microsoft/BitNet** 的 `gpu/pack_weight.py`）：
//   · 2 bit / 权重；4 权重 / 字节；码值 ∈ {1,2,3} = 三值 {-1,0,+1} **+ 2**
//     （0 码留空 —— 与官方 `weight + 2` 同款：三值没有"零码"，不留浪费码位）；
//   · 容器 = **f32 张量**（每元素 4 字节 = 16 个权重），行内小端：
//     低 2 bit 对应行内更靠前的权重（第 j 个权重的码在 `(word >> 2j) & 3`）；
//   · 形状 (rows, ceil(K/16)) → **0.25 B/权重**（f16 稠密缓冲是 2 B/权重，
//     打包后权重带宽/显存降 8 倍）。
//
// 与 BitNet 的**唯一差别**：不采用其 CUDA 侧的 32-bit interleave 置换
//   （`interleave_weight_int8`，那是为 coalesced 访问做的重排，解码等价），
//   本项目按自然顺序存放 —— 位宽、码值映射、每字节权重数一致，故"编解码"
//   可以逐位对拍（docs/development/21 §5 的 P2 验收第 4 条）。
//
// 为什么容器是 f32 张量而不是新的存储类型：张量存储多态（M7）是独立立项
//   （docs/development/17 §5）；打包缓冲只在引擎原语内部被当作**原始字**读写
//   （`std::bit_cast`），从不参与浮点运算 —— 契约由本头 + 原语共同保证。
//
// 为什么不用 base-3（1.59 bpw，docs 21 §0 的 D2 裁决）：本步对齐的是官方 GPU
//   内核实际使用的 2-bit 定长编码（解码只有移位/掩码，无除法），先行拿到可量
//   的带宽/显存数据；base-3 编码留作部署格式的后续精修（无损从 2 bpw 降到
//   1.59 bpw，收益 20%，代价是逐字节除 3 的解码）。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>

namespace nn
{
namespace trit_pack
{

inline constexpr std::uint32_t kBits = 2;                 // 每权重 2 bit
inline constexpr std::uint32_t kPerByte = 8 / kBits;      // 4 权重/字节
inline constexpr std::uint32_t kPerWord = 4 * kPerByte;   // 16 权重/f32 元素
inline constexpr std::uint32_t kMask = (1u << kBits) - 1u;

// 每行需要的字（f32 元素）数：(K + 15) / 16
[[nodiscard]] inline constexpr std::size_t words_per_row(std::size_t k) noexcept
{
    return (k + kPerWord - 1) / kPerWord;
}

// 每行打包后的字节数：4 字节/字 × 字数 = 0.25·K（向上取到 4B 对齐）
[[nodiscard]] inline constexpr std::size_t bytes_per_row(std::size_t k) noexcept
{
    return words_per_row(k) * sizeof(std::uint32_t);
}

// 量化码：码 = 三值 + 2（∈{1,2,3}）；判据 = |w| > 0.5·τ（BitNet 定义形式，
// 与 BitLinear 的 select 阈值逐点一致）。
[[nodiscard]] inline std::uint32_t code_of(float w, float tau) noexcept
{
    const float half = 0.5f * tau;
    if (w > half) return 3u;    // +1
    if (w < -half) return 1u;   // -1
    return 2u;                  //  0
}

// 码 → 三值（float，精确可表示）
[[nodiscard]] inline float value_of(std::uint32_t code) noexcept
{
    return static_cast<float>(static_cast<int>(code & kMask) - 2);
}

// 打包一行：w[0..K)（宿主连续）+ 逐行尺度 τ → words_per_row(K) 个 32-bit 字。
// 尾部（idx ≥ K）填"码 2 = 三值 0"而不是 0：码 0 在映射里无定义，填 0 会让
// 误读尾部的内核拿到 -2（静默错值），填 2 最多是多算一个 0。
inline void pack_row(const float* w, std::size_t k, float tau,
                     std::uint32_t* out) noexcept
{
    const std::size_t words = words_per_row(k);
    for (std::size_t wi = 0; wi < words; ++wi)
    {
        std::uint32_t word = 0;
        for (std::uint32_t j = 0; j < kPerWord; ++j)
        {
            const std::size_t idx = wi * kPerWord + j;
            const std::uint32_t code = (idx < k) ? code_of(w[idx], tau) : 2u;
            word |= code << (kBits * j);
        }
        out[wi] = word;
    }
}

// 单点解码：words 指向该行首字
[[nodiscard]] inline float at(const std::uint32_t* words, std::size_t idx) noexcept
{
    const std::uint32_t word = words[idx / kPerWord];
    return value_of((word >> (kBits * (idx % kPerWord))) & kMask);
}

} // namespace trit_pack
} // namespace nn
