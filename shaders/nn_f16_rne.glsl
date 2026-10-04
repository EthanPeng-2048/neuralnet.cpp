// ── nn_f16_rne.glsl — f32 → f16 存储的 RNE（round-half-to-even）预舍入 ──────
// 供所有带 -DNN_SHADER_F16=1 变体的手写 shader #include（glslc 原生支持）。
//
// 为什么需要：GLSL 的 float16_t(x) 编译为 SPIR-V OpFConvert，其舍入模式**未被
// Vulkan/SPIR-V 规范钉死**；实测 NVIDIA 驱动按“向零截断”。对 AdamW 权重衰减
// p *= (1-lr·wd) 这类结果紧贴原 f16 网格点的写回，截断恰好每步掉 1 个网格步
//（与 lr 无关）→ 数万步后权重下溢为 0、loss 回到 ln(V)。
// ⚠ packHalf2x16 不能作为替代：同一驱动实测它**同样截断**（Khronos
// Vulkan-Docs #1825「PackHalf2x16 rounding behavior is ambiguously defined」），
// 与宿主 nn::f16（RNE）对拍 8 个探针值有 4 个不一致。
//
// 做法：在 f32 域用整数位运算做一次精确 RNE，返回值本身落在 f16 网格上
// （f16 ⊂ f32，可精确表示）→ 随后的 float16_t() 收窄 / packHalf2x16 打包
// 无论驱动取哪种舍入模式都不再产生偏差。净效果 = 一次 RNE 的 f16 存储。
//
// 与 include 方约定：宏 NN_F16_RNE_FN 为函数名（默认 nn_f16_rne）。
#ifndef NN_F16_RNE_FN
#define NN_F16_RNE_FN nn_f16_rne
#endif

float NN_F16_RNE_FN(float x)
{
    uint u = floatBitsToUint(x);
    const uint a = u & 0x7fffffffu;
    if (a >= 0x7f800000u || a < 0x38800000u)   // Inf/NaN，或 |x| < 2^-14（次正规）
        return roundEven(x * 16777216.0) / 16777216.0;
    u += 0x0fffu + ((u >> 13u) & 1u);          // 半个 ULP + tie-to-even
    u &= 0xffffe000u;                          // 截到 f16 的 10 位尾数
    return uintBitsToFloat(u);
}
