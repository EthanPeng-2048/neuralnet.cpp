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
//   可以逐位对拍（docs/todo/21 §5 的 P2 验收第 4 条）。
//
// 为什么容器是 f32 张量而不是新的存储类型：张量存储多态（M7）是独立立项
//   （docs/todo/17 §5）；打包缓冲只在引擎原语内部被当作**原始字**读写
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

// ═══════════════════════════════════════════════════════════════════════════
//  base-3（部署编码，1.6 bpw）
//
//  2-bit 定长编码是**运行期**容器（解码只有移位/掩码，喂 `matmul_q`）；部署
//  文件用 base-3 再省 20%（2.0 → 1.6 bpw）：5 个三值/字节，字节 = Σ digit_i·3^i，
//  `digit = 码 - 1 ∈ {0,1,2}`（即 trit + 1；与 2-bit 的 `码 = trit + 2` 差一个
//  偏移，故两者的映射都在本头里唯一）。
//
//  为什么 5/字节（243 个状态，1.6 bpw）：`log2(3) = 1.585`，5 位一组是"每字节
//  不浪费码位"的最大分组（3^5 = 243 ≤ 256、3^6 = 729 > 256）。这条与
//  microsoft/BitNet 无关，与 llama.cpp `TQ1_0` 的 base-3 分组同源 —— 差别是我们
//  的块粒度是**行**、尺度是逐行 absmean（`TQ1_0` 是 per-256 + qh 尾巴 + 定点
//  `(s·3^n·3)>>8`）；部署格式不需要与 GGUF 互读，故不引入 qh/interleave，
//  尾部直接补零 digit（解码按 k 截断）。
//
//  何时用它：**只在部署文件的落盘/读盘路径**（`model_serialization` 的 v6）。
//  运行期一律转回 2-bit 容器（loosen 一次宿主侧逐位转换，成本一次性）。
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr std::uint32_t kBase3PerByte = 5;   // 5 个三值/字节

// 每行 base-3 字节数：ceil(K/5)
[[nodiscard]] inline constexpr std::size_t base3_bytes_per_row(std::size_t k) noexcept
{
    return (k + kBase3PerByte - 1) / kBase3PerByte;
}

// 2-bit 容器行 → base-3 字节（部署落盘方向）。尾部不足 5 个的位置补 digit 0
// （解码按 k 截断，补位不承载信息）。
inline void pack_row_base3(const std::uint32_t* words, std::size_t k,
                           std::uint8_t* out) noexcept
{
    const std::size_t bytes = base3_bytes_per_row(k);
    for (std::size_t b = 0; b < bytes; ++b)
    {
        std::uint32_t byte = 0;
        std::uint32_t pow3 = 1;
        for (std::uint32_t i = 0; i < kBase3PerByte; ++i)
        {
            const std::size_t idx = b * kBase3PerByte + i;
            const std::uint32_t code = (idx < k)
                ? ((words[idx / kPerWord] >> (kBits * (idx % kPerWord))) & kMask)
                : 2u;   // 补位：码 2 → digit 1（无信息，解码截断）
            byte += (code - 1u) * pow3;
            pow3 *= 3u;
        }
        out[b] = static_cast<std::uint8_t>(byte);
    }
}

// base-3 字节行 → 2-bit 容器行（部署读盘方向）。尾部字按既有约定填"码 2 = 三值 0"。
inline void unpack_row_base3(const std::uint8_t* in, std::size_t k,
                             std::uint32_t* words) noexcept
{
    const std::size_t nwords = words_per_row(k);
    for (std::size_t wi = 0; wi < nwords; ++wi) words[wi] = 0;
    for (std::size_t idx = 0; idx < k; ++idx)
    {
        std::uint32_t byte = in[idx / kBase3PerByte];
        for (std::uint32_t i = 0; i < idx % kBase3PerByte; ++i) byte /= 3u;
        const std::uint32_t code = (byte % 3u) + 1u;   // digit → 码
        words[idx / kPerWord] |= code << (kBits * (idx % kPerWord));
    }
    // 尾部：显式写"码 2"（= 三值 0），与 pack_row 的约定一致（不做则留下码 0 = -2）
    for (std::size_t idx = k; idx < nwords * kPerWord; ++idx)
        words[idx / kPerWord] |= 2u << (kBits * (idx % kPerWord));
}

// 单点解码（参考/测试用）：in 指向该行首字节
[[nodiscard]] inline float at_base3(const std::uint8_t* in, std::size_t idx) noexcept
{
    std::uint32_t byte = in[idx / kBase3PerByte];
    for (std::uint32_t i = 0; i < idx % kBase3PerByte; ++i) byte /= 3u;
    return value_of((byte % 3u) + 1u);
}

} // namespace trit_pack
} // namespace nn
