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
    RoPE = 0,          // 旋转位置编码（Q/K 上施加）
    SwiGLUGradGate = 1, // SwiGLU backward: grad_gate
    SwiGLUGradUp   = 2, // SwiGLU backward: grad_up
};

// ── 表达式工厂（单一事实来源）───────────────────────────────────────────
// RoPE（LLaMA half-swap 约定）：
//   forward:  q_rot = q·cos + rotate_half(q)·sin
//   backward: g_rot = g·cos − rotate_half(g)·sin（旋转正交，逆=反角）
//   inputs: [q, q, cos, sin]；views: [Linear, RotateHalf(d_k,前半取负),
//                                     RowMod(d_k), RowMod(d_k)]
[[nodiscard]] inline ExprSpec make_rope(std::uint32_t d_k, bool backward)
{
    // 必须与 dsl::make_rope<B>(q,cos,sin,dk) 的 to_expr_spec 折叠结果**完全一致**
    // （drift 由 expr_dsl_test 用 expr_spec_equal 断言保证；这是运行时 AOT 匹配
    //  的前提，本头只依赖 expr_spec.hpp 以避免 tensor→vk_backend 循环包含）。
    // 折叠顺序：term1 = q·cos → [q(Linear), cos(RowMod)]；
    //           term2 = rot(q)·sin → [q(RotateHalf), sin(RowMod)]；
    //           out = term1 ± term2。
    ExprSpec spec;
    spec.views = {
        expr::linear(),                          // input0 = q（线性）
        expr::row_mod(d_k),                      // input1 = cos（RowMod 平铺）
        expr::rotate_half(d_k, /*negate_first_half=*/true), // input2 = rot(q)
        expr::row_mod(d_k),                      // input3 = sin
    };
    spec.num_regs = 3;

    ExprInstr i0, i1, i2;
    i0.op = static_cast<uint8_t>(ExprOp::Mul);
    i0.dst = 0; i0.a = expr::input(0); i0.b = expr::input(1);   // r0 = q*cos
    i1.op = static_cast<uint8_t>(ExprOp::Mul);
    i1.dst = 1; i1.a = expr::input(2); i1.b = expr::input(3);   // r1 = rot(q)*sin
    i2.op = static_cast<uint8_t>(backward ? ExprOp::Sub : ExprOp::Add);
    i2.dst = 2; i2.a = expr::reg(0); i2.b = expr::reg(1);       // r2 = r0 ± r1
    spec.instrs = {i0, i1, i2};
    return spec;
}

// SwiGLU backward：grad_gate = grad_out ⊙ up ⊙ s ⊙ (1 + gate ⊙ (1 − s))
// 必须与 dsl::make_swiglu_grad_gate 的 to_expr_spec 折叠结果**完全一致**（防漂移）。
// 折叠顺序：term1 = g·up；term2 = s·(1 + gate·(1−s))；out = term1·term2。
// 输入序 [g, up, s, gate, s]（s 出现两次），视图全 Linear，常量池 [1, 1]。
[[nodiscard]] inline ExprSpec make_swiglu_grad_gate()
{
    ExprSpec spec;
    spec.views = {expr::linear(), expr::linear(), expr::linear(),
                  expr::linear(), expr::linear()};
    spec.num_regs = 6;
    spec.consts = {Scalar{1}, Scalar{1}};

    ExprInstr i[6];
    i[0].op = static_cast<uint8_t>(ExprOp::Mul);
    i[0].dst = 0; i[0].a = expr::input(0); i[0].b = expr::input(1);   // r0 = g*up
    i[1].op = static_cast<uint8_t>(ExprOp::Sub);
    i[1].dst = 1; i[1].a = expr::cst(1); i[1].b = expr::input(4);     // r1 = 1−s
    i[2].op = static_cast<uint8_t>(ExprOp::Mul);
    i[2].dst = 2; i[2].a = expr::input(3); i[2].b = expr::reg(1);     // r2 = gate*(1−s)
    i[3].op = static_cast<uint8_t>(ExprOp::Add);
    i[3].dst = 3; i[3].a = expr::cst(0); i[3].b = expr::reg(2);       // r3 = 1 + gate*(1−s)
    i[4].op = static_cast<uint8_t>(ExprOp::Mul);
    i[4].dst = 4; i[4].a = expr::input(2); i[4].b = expr::reg(3);     // r4 = s*factor
    i[5].op = static_cast<uint8_t>(ExprOp::Mul);
    i[5].dst = 5; i[5].a = expr::reg(0); i[5].b = expr::reg(4);       // r5 = g*up*factor
    spec.instrs = {i[0], i[1], i[2], i[3], i[4], i[5]};
    return spec;
}

// SwiGLU backward：grad_up = grad_out ⊙ gate ⊙ s
// 必须与 dsl::make_swiglu_grad_up 折叠一致；输入序 [g, gate, s]，无常量。
[[nodiscard]] inline ExprSpec make_swiglu_grad_up()
{
    ExprSpec spec;
    spec.views = {expr::linear(), expr::linear(), expr::linear()};
    spec.num_regs = 2;

    ExprInstr i[2];
    i[0].op = static_cast<uint8_t>(ExprOp::Mul);
    i[0].dst = 0; i[0].a = expr::input(0); i[0].b = expr::input(1);   // r0 = g*gate
    i[1].op = static_cast<uint8_t>(ExprOp::Mul);
    i[1].dst = 1; i[1].a = expr::reg(0); i[1].b = expr::input(2);     // r1 = r0*s
    spec.instrs = {i[0], i[1]};
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
    {"swiglu_grad_gate",   FusedKind::SwiGLUGradGate, 0, false},
    {"swiglu_grad_up",     FusedKind::SwiGLUGradUp,   0, false},
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
    case FusedKind::SwiGLUGradGate:
        return make_swiglu_grad_gate();
    case FusedKind::SwiGLUGradUp:
        return make_swiglu_grad_up();
    }
    return {};  // 未识别：返回空（validate 会拒绝）
}

} // namespace nn::fused

#endif // NN_FUSED_EXPRS_HPP
