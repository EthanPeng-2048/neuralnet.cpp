#ifndef NN_FUSED_EXPRS_HPP
#define NN_FUSED_EXPRS_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  fused_exprs.hpp — 可融合表达式：单一事实来源（single source of truth）
//
//  所有可融合的表达式**直接写在本头（C++ 源码）**，两个消费方共用同一份
//  定义，绝不漂移：
//    1. 运行时：Layer 调用工厂构建 ExprSpec → engine.eval_expr
//                （CPU 融合解释器 / GPU eager lowering）
//    2. 构建期：tools/gen_fused 遍历 kGenInstances，用同一工厂生成
//                单个融合 .comp（AOT 融合加速，见 glsl_gen.hpp）
//
//  JSON/外部描述不可接受：类型安全、可重构、可搜索全部丢失。
//  本头是唯一的事实来源。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstddef>

#include "expr_spec.hpp"

namespace nn::fused
{

// ── 融合表达式类型 ─────────────────────────────────────────────────────────
enum class FusedKind : uint8_t
{
    RoPE = 0,  // 旋转位置编码（Q/K 上施加）
};

// ── 表达式工厂（单一事实来源）───────────────────────────────────────────
// RoPE（LLaMA half-swap 约定）：
//   forward:  q_rot = q·cos + rotate_half(q)·sin
//   backward: g_rot = g·cos − rotate_half(g)·sin（旋转正交，逆=反角）
//   inputs: [q, q, cos, sin]；views: [Linear, RotateHalf(d_k,前半取负),
//                                     RowMod(d_k), RowMod(d_k)]
[[nodiscard]] inline ExprSpec make_rope(std::uint32_t d_k, bool backward)
{
    ExprSpec spec;
    spec.views = {
        expr::linear(),
        expr::rotate_half(d_k, /*negate_first_half=*/true),
        expr::row_mod(d_k),
        expr::row_mod(d_k),
    };
    spec.num_regs = 3;

    ExprInstr i0, i1, i2;
    i0.op = static_cast<uint8_t>(ExprOp::Mul);
    i0.dst = 0; i0.a = expr::input(0); i0.b = expr::input(2);   // r0 = q*cos
    i1.op = static_cast<uint8_t>(ExprOp::Mul);
    i1.dst = 1; i1.a = expr::input(1); i1.b = expr::input(3);   // r1 = rot(q)*sin
    i2.op = static_cast<uint8_t>(backward ? ExprOp::Sub : ExprOp::Add);
    i2.dst = 2; i2.a = expr::reg(0); i2.b = expr::reg(1);       // r2 = r0 ± r1
    spec.instrs = {i0, i1, i2};
    return spec;
}

// ── 构建期生成器实例表 ───────────────────────────────────────────────────
// 构建期 tools/gen_fused 遍历此表，为每个实例生成一个融合 shader。
// 参数（如 d_k）是 shader 的编译期常量（决定视图 block/mod）。
struct GenInstance
{
    const char*  name;      // 融合 shader 名（也是产物文件名前缀）
    FusedKind    kind;
    std::uint32_t d_k;      // RoPE 维度块大小
    bool         backward;  // true = 反角（backward 用）
};

inline constexpr GenInstance kGenInstances[] = {
    {"rope_forward_dk32",  FusedKind::RoPE, 32, false},
    {"rope_forward_dk64",  FusedKind::RoPE, 64, false},
    {"rope_forward_dk128", FusedKind::RoPE, 128, false},
    {"rope_backward_dk32", FusedKind::RoPE, 32, true},
    {"rope_backward_dk64", FusedKind::RoPE, 64, true},
    {"rope_backward_dk128", FusedKind::RoPE, 128, true},
};
inline constexpr std::size_t kGenInstanceCount =
    sizeof(kGenInstances) / sizeof(kGenInstances[0]);

// ── 按实例构建 ExprSpec（供生成器与运行时复用） ──────────────────────────
[[nodiscard]] inline ExprSpec make_fused(const GenInstance& inst)
{
    switch (inst.kind)
    {
    case FusedKind::RoPE:
        return make_rope(inst.d_k, inst.backward);
    }
    return {};  // 未识别：返回空（validate 会拒绝）
}

} // namespace nn::fused

#endif // NN_FUSED_EXPRS_HPP
