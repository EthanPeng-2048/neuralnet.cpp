#pragma once

// ══════════════════════════════════════════════════════════════════════════
//  core_sha256.hpp — SHA-256 摘要（FIPS 180-4 标准实现）
//
//  用途：数据集配套的源文本内容摘要（`source.sha256`，19 号设计 §4.3）——
//  失效判断基于内容摘要而非文件大小（同大小不同内容不会骗过）。
//  单文件 header-only，输入一次性给足（语料摘要在生成期算，流式留待 P2-4）。
// ══════════════════════════════════════════════════════════════════════════

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace nn
{

namespace detail
{

struct Sha256State
{
    std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::uint64_t total = 0;      // 已吸收的字节数
    std::uint8_t block[64] = {};
    std::size_t block_len = 0;
};

inline constexpr std::uint32_t sha256_rotr(std::uint32_t x, unsigned n) noexcept
{
    return (x >> n) | (x << (32 - n));
}

inline void sha256_compress(Sha256State &st, const std::uint8_t *p)
{
    static constexpr std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24)
             | (static_cast<std::uint32_t>(p[4 * i + 1]) << 16)
             | (static_cast<std::uint32_t>(p[4 * i + 2]) << 8)
             | static_cast<std::uint32_t>(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i)
    {
        const std::uint32_t s0 = sha256_rotr(w[i - 15], 7)
            ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const std::uint32_t s1 = sha256_rotr(w[i - 2], 17)
            ^ sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    std::uint32_t a = st.h[0], b = st.h[1], c = st.h[2], d = st.h[3];
    std::uint32_t e = st.h[4], f = st.h[5], g = st.h[6], hh = st.h[7];
    for (int i = 0; i < 64; ++i)
    {
        const std::uint32_t S1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        const std::uint32_t ch = (e & f) ^ (~e & g);
        const std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
        const std::uint32_t S0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    st.h[0] += a; st.h[1] += b; st.h[2] += c; st.h[3] += d;
    st.h[4] += e; st.h[5] += f; st.h[6] += g; st.h[7] += hh;
}

inline void sha256_update(Sha256State &st, std::string_view data)
{
    st.total += data.size();
    std::size_t pos = 0;
    if (st.block_len > 0)
    {
        const std::size_t need = 64 - st.block_len;
        const std::size_t take = data.size() < need ? data.size() : need;
        for (std::size_t i = 0; i < take; ++i)
            st.block[st.block_len + i] = static_cast<std::uint8_t>(data[pos + i]);
        st.block_len += take;
        pos += take;
        if (st.block_len == 64)
        {
            sha256_compress(st, st.block);
            st.block_len = 0;
        }
    }
    while (pos + 64 <= data.size())
    {
        sha256_compress(st, reinterpret_cast<const std::uint8_t *>(data.data() + pos));
        pos += 64;
    }
    for (; pos < data.size(); ++pos)
        st.block[st.block_len++] = static_cast<std::uint8_t>(data[pos]);
}

inline std::array<std::uint8_t, 32> sha256_digest(std::string_view data)
{
    Sha256State st;
    sha256_update(st, data);

    // 收尾：填充 0x80 + 零 + 64 位大端比特长度
    const std::uint64_t bits = st.total * 8;
    st.block[st.block_len++] = 0x80;
    if (st.block_len > 56)
    {
        while (st.block_len < 64) st.block[st.block_len++] = 0;
        sha256_compress(st, st.block);
        st.block_len = 0;
    }
    while (st.block_len < 56) st.block[st.block_len++] = 0;
    for (int i = 7; i >= 0; --i)
        st.block[st.block_len++] = static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFF);
    sha256_compress(st, st.block);

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i)
    {
        out[4 * i]     = static_cast<std::uint8_t>((st.h[i] >> 24) & 0xFF);
        out[4 * i + 1] = static_cast<std::uint8_t>((st.h[i] >> 16) & 0xFF);
        out[4 * i + 2] = static_cast<std::uint8_t>((st.h[i] >> 8) & 0xFF);
        out[4 * i + 3] = static_cast<std::uint8_t>(st.h[i] & 0xFF);
    }
    return out;
}

}  // namespace detail

// SHA-256 十六进制摘要（64 个小写 hex 字符）
[[nodiscard]] inline std::string sha256_hex(std::string_view data)
{
    const auto d = detail::sha256_digest(data);
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint8_t b : d)
    {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 0xF]);
    }
    return out;
}

}  // namespace nn
