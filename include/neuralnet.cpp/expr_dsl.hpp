#ifndef NN_EXPR_DSL_HPP
#define NN_EXPR_DSL_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_dsl.hpp — 统一表达式 DSL（编译期模板，唯一前端）
//
//  目标：所有逐元素计算都用"普通数学写法"写出，编译期融合：
//      auto e = leaf(q) * row_mod(cos, dk) + rotate_half(leaf(q), dk) * row_mod(sin, dk);
//      engine.compute(e, rows, cols);
//
//  单一事实来源 = 这一整段表达式本身（编译期类型）。两个消费方共用：
//    1. CPU：直接把表达式当作编译期 AST，逐元素求值 → 编译器内联 + SIMD 融合，
//       等价手写 for 循环（零开销抽象）。
//    2. GPU（AOT，无运行时生成）：to_expr_spec(expr) 在编译期把同一表达式折叠
//       成扁平 ExprSpec → 与预生成融合 shader 比对 → dispatch。
//
//  闭合世界（closed-world）约定：代码里用到的表达式集合 = AOT 生成的 shader 集合。
//  运行时若遇到未预生成 shader 的表达式，GPU 直接**硬报错**，绝不静默回退
//  （即彻底删除 eager 路径，把"难以察觉的漂移"变成"立即暴露的错误"）。
//
//  设计要点：
//    - 算子复用 nn::ops（唯一的算子来源，op_id() 统一映射到 ExprOp）。
//    - 叶子持有 Tensor（零拷贝 shared_ptr）；视图（RotateHalf/RowMod）是索引
//      映射叶子，不物化中间张量。
//    - 运算符重载/概念约束：普通 + - * / 与比较、select 均照常书写。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <utility>
#include <vector>

#include "tensor.hpp"
#include "compute_engine.hpp"
#include "expr_spec.hpp"
#include "algebra_expr.hpp"   // nn::Expression / nn::BoolExpression 概念
#include "algebra_ops.hpp"    // nn::ops（唯一算子来源，含 op_id()）
#include "algebra_matrix.hpp" // Matrix

namespace nn::dsl
{

// ══════════════════════════════════════════════════════════════════════════
// 内部求值/折叠构建器：把编译期表达式折叠成扁平 ExprSpec + 输入张量列表
// ══════════════════════════════════════════════════════════════════════════
struct SpecBuilder
{
    ExprSpec spec;
    std::vector<Tensor> inputs;

    ExprOperand add_const(Scalar v)
    {
        spec.consts.push_back(v);
        return expr::cst(static_cast<std::uint8_t>(spec.consts.size() - 1));
    }
    ExprOperand add_input_linear(const Tensor& t)
    {
        spec.views.push_back(expr::linear());
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_rotate(const Tensor& t, std::uint32_t block)
    {
        spec.views.push_back(expr::rotate_half(block, /*negate_first_half=*/true));
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_rowmod(const Tensor& t, std::uint32_t mod)
    {
        spec.views.push_back(expr::row_mod(mod));
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_instr(ExprOp op, ExprOperand a, ExprOperand b = {}, ExprOperand c = {})
    {
        ExprInstr in;
        in.op = static_cast<std::uint8_t>(op);
        in.dst = static_cast<std::uint8_t>(spec.num_regs++);
        in.a = a; in.b = b; in.c = c;
        spec.instrs.push_back(in);
        return expr::reg(in.dst);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 叶子节点（同时满足 nn::Expression：可 eval(i)；以及可 to_spec(SpecBuilder&)）
// ══════════════════════════════════════════════════════════════════════════

// 标量常量
struct ConstLeaf
{
    Scalar value;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return value; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_const(value); }
};

// 线性叶子：直接读取 Tensor 数据（row-major 扁平）
struct TensorRef
{
    Tensor t;

    [[nodiscard]] Scalar eval(std::size_t i) const { return t.cpu_matrix().span()[i]; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_linear(t); }
};

// 视图：RotateHalf —— 按 block 分块，块内前后半行交换 + 前半取负（LLaMA rotate_half）
struct RotateHalfRef
{
    Tensor t;
    std::uint32_t block;

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        const std::size_t cols = t.cols();
        const std::size_t r = i / cols, c = i % cols;
        const std::size_t blk = block;
        const std::size_t rl = r % blk;
        const std::size_t rr = (r / blk) * blk + ((rl < blk / 2) ? rl + blk / 2 : rl - blk / 2);
        Scalar v = t.cpu_matrix().span()[rr * cols + c];
        return (rl < blk / 2) ? -v : v;
    }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_rotate(t, block); }
};

// 视图：RowMod —— 行取模广播（频率表平铺到多行块）
struct RowModRef
{
    Tensor t;
    std::uint32_t mod;

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        const std::size_t cols = t.cols();
        const std::size_t r = i / cols, c = i % cols;
        return t.cpu_matrix().span()[(r % mod) * cols + c];
    }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_rowmod(t, mod); }
};

// ══════════════════════════════════════════════════════════════════════════
// 节点模板（每个都同时支持 CPU 求值 eval(i) 与 GPU 折叠 to_spec）
// ══════════════════════════════════════════════════════════════════════════

template <typename Op, nn::Expression C>
struct Unary
{
    C child;
    [[nodiscard]] auto eval(std::size_t i) const { return Op::apply(child.eval(i)); }
    ExprOperand to_spec(SpecBuilder& b) const
    { return b.add_instr(Op::op_id(), child.to_spec(b)); }
};

// 比较算子（ops::Gt 等）使 eval 返回 bool → 该 Binary 是 BoolExpression，
// 供 select() 使用；to_spec 仍折叠成单条比较指令（输出 0.0/1.0 到寄存器）。
template <typename Op, nn::Expression L, nn::Expression R>
struct Binary
{
    L l; R r;
    [[nodiscard]] auto eval(std::size_t i) const { return Op::apply(l.eval(i), r.eval(i)); }
    ExprOperand to_spec(SpecBuilder& b) const
    { return b.add_instr(Op::op_id(), l.to_spec(b), r.to_spec(b)); }
};

// select(cond, then, else) —— cond 为 BoolExpression，then/else 为值表达式
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
struct Select
{
    C cond; T then_e; E else_e;
    [[nodiscard]] Scalar eval(std::size_t i) const
    { return cond.eval(i) ? then_e.eval(i) : else_e.eval(i); }
    ExprOperand to_spec(SpecBuilder& b) const
    { return b.add_instr(ExprOp::Select, cond.to_spec(b), then_e.to_spec(b), else_e.to_spec(b)); }
};

// ══════════════════════════════════════════════════════════════════════════
// 叶子构造（普通写法入口）
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline TensorRef leaf(Tensor t) { return TensorRef{std::move(t)}; }

[[nodiscard]] inline RotateHalfRef rotate_half(Tensor t, std::uint32_t block)
{ return RotateHalfRef{std::move(t), block}; }

[[nodiscard]] inline RowModRef row_mod(Tensor t, std::uint32_t mod)
{ return RowModRef{std::move(t), mod}; }

// ══════════════════════════════════════════════════════════════════════════
// 一元函数（普通数学写法）
// ══════════════════════════════════════════════════════════════════════════
template <nn::Expression E> [[nodiscard]] constexpr auto neg(const E& e) { return Unary<ops::Neg, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto abs(const E& e)   { return Unary<ops::Abs, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto exp(const E& e)   { return Unary<ops::Exp, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto log(const E& e)   { return Unary<ops::Log, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto sqrt(const E& e)  { return Unary<ops::Sqrt, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto rsqrt(const E& e) { return Unary<ops::Rsqrt, E>{e}; }
template <nn::Expression E> [[nodiscard]] auto tanh(const E& e)  { return Unary<ops::Tanh, E>{e}; }

// ══════════════════════════════════════════════════════════════════════════
// 二元函数：max / min / select
// ══════════════════════════════════════════════════════════════════════════
template <nn::Expression L, nn::Expression R> [[nodiscard]] constexpr auto max(const L& l, const R& r) { return Binary<ops::Max, L, R>{l, r}; }
template <nn::Expression E> [[nodiscard]] constexpr auto max(const E& e, Scalar s) { return Binary<ops::Max, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::Expression E> [[nodiscard]] constexpr auto max(Scalar s, const E& e) { return Binary<ops::Max, ConstLeaf, E>{ConstLeaf{s}, e}; }
template <nn::Expression L, nn::Expression R> [[nodiscard]] constexpr auto min(const L& l, const R& r) { return Binary<ops::Min, L, R>{l, r}; }
template <nn::Expression E> [[nodiscard]] constexpr auto min(const E& e, Scalar s) { return Binary<ops::Min, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::Expression E> [[nodiscard]] constexpr auto min(Scalar s, const E& e) { return Binary<ops::Min, ConstLeaf, E>{ConstLeaf{s}, e}; }

/// relu(x) = max(x, 0)
template <nn::Expression E> [[nodiscard]] auto relu(const E& e) { return nn::dsl::max(e, ConstLeaf{Scalar{0}}); }

template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
[[nodiscard]] constexpr auto select(const C& c, const T& t, const E& e)
{ return Select<C, T, E>{c, t, e}; }
template <nn::BoolExpression C, nn::Expression T>
[[nodiscard]] constexpr auto select(const C& c, const T& t, Scalar ev)
{ return Select<C, T, ConstLeaf>{c, t, ConstLeaf{ev}}; }
template <nn::BoolExpression C, nn::Expression E>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, const E& e)
{ return Select<C, ConstLeaf, E>{c, ConstLeaf{tv}, e}; }
template <nn::BoolExpression C>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, Scalar ev)
{ return Select<C, ConstLeaf, ConstLeaf>{c, ConstLeaf{tv}, ConstLeaf{ev}}; }

// ══════════════════════════════════════════════════════════════════════════
// 运算符（Sym 生成，覆盖 Expr×Expr / Expr×Scalar / Scalar×Expr）
// 值运算（+ - * /）返回值表达式；比较（> < >= <= == !=）返回布尔表达式。
// ══════════════════════════════════════════════════════════════════════════
#define NN_DSL_BINARY(OP, OpT)                                                            \
    template <nn::Expression L, nn::Expression R>                                         \
    [[nodiscard]] constexpr auto operator OP(const L& l, const R& r)                      \
    { return Binary<OpT, L, R>{l, r}; }                                                   \
    template <nn::Expression E>                                                           \
    [[nodiscard]] constexpr auto operator OP(const E& e, Scalar s)                        \
    { return Binary<OpT, E, ConstLeaf>{e, ConstLeaf{s}}; }                                \
    template <nn::Expression E>                                                           \
    [[nodiscard]] constexpr auto operator OP(Scalar s, const E& e)                        \
    { return Binary<OpT, ConstLeaf, E>{ConstLeaf{s}, e}; }

NN_DSL_BINARY(+, ops::Add)
NN_DSL_BINARY(-, ops::Sub)
NN_DSL_BINARY(*, ops::Mul)
NN_DSL_BINARY(/, ops::Div)
NN_DSL_BINARY(>, ops::Gt)
NN_DSL_BINARY(<, ops::Lt)
NN_DSL_BINARY(>=, ops::Ge)
NN_DSL_BINARY(<=, ops::Le)
NN_DSL_BINARY(==, ops::Eq)
NN_DSL_BINARY(!=, ops::Ne)
#undef NN_DSL_BINARY

template <nn::Expression E> [[nodiscard]] constexpr auto operator-(const E& e)
{ return Unary<ops::Neg, E>{e}; }

// ══════════════════════════════════════════════════════════════════════════
// GPU AOT：编译期表达式 → 扁平 ExprSpec + 输入张量（闭合世界分发用）
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] std::pair<ExprSpec, std::vector<Tensor>> to_expr_spec(const E& e)
{
    SpecBuilder b;
    (void)e.to_spec(b);
    return {std::move(b.spec), std::move(b.inputs)};
}

// ══════════════════════════════════════════════════════════════════════════
// CPU：编译期模板直接求值（编译器内联 + SIMD 融合，等价手写循环）
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Tensor eval_cpu(const E& e, std::size_t rows, std::size_t cols)
{
    Matrix out(rows, cols);
    auto sp = out.span();
    const std::size_t n = sp.size();
    for (std::size_t i = 0; i < n; ++i)
        sp[i] = static_cast<Scalar>(e.eval(i));
    return Tensor::from_matrix(std::move(out));
}

// ══════════════════════════════════════════════════════════════════════════
// 统一入口：engine.compute(expr, rows, cols)
//   CPU：编译期模板求值（SIMD 融合）。
//   GPU：闭合世界 AOT —— to_expr_spec 折叠 → engine.eval_expr 匹配预生成
//        shader；未命中由 eval_expr 硬报错（无 eager）。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Result<Tensor> compute(ComputeEngine& eng, const E& e,
                                     std::size_t rows, std::size_t cols)
{
    if (eng.device() == Device::CPU)
        return eval_cpu(e, rows, cols);

    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr(spec, inputs, rows, cols);  // 闭合世界：GPU 未命中即报错
}

// ══════════════════════════════════════════════════════════════════════════
// 常用表达式工厂（单一事实来源；CPU 与 AOT 生成器共用同一段定义）
// ══════════════════════════════════════════════════════════════════════════
// RoPE（LLaMA half-swap）：
//   forward:  q_rot = q·cos + rotate_half(q)·sin
//   backward: g_rot = q·cos − rotate_half(q)·sin（旋转正交，逆 = 反角）
// q 为 (batch*H*d_k, seq)，cos/sin 为 (d_k, seq) 短表（RowMod 平铺）。
// dk 为**运行时**索引映射参数（CPU 支持任意 d_k；GPU AOT 只对生成的
// {32,64,128} 命中，未命中即闭合世界报错）；backward 为编译期参数，
// 决定算子序列（mul/mul/add|sub），故融合在编译期完成。
template <bool Backward>
[[nodiscard]] auto make_rope(const Tensor& q, const Tensor& cos, const Tensor& sin,
                             std::uint32_t dk)
{
    auto term1 = leaf(q) * row_mod(cos, dk);
    auto term2 = rotate_half(q, dk) * row_mod(sin, dk);
    if constexpr (Backward) return term1 - term2;
    else                    return term1 + term2;
}

// SwiGLU backward：
//   grad_gate = grad_out ⊙ up ⊙ s ⊙ (1 + gate ⊙ (1 − s))
//   grad_up   = grad_out ⊙ gate ⊙ s
[[nodiscard]] inline auto make_swiglu_grad_gate(const Tensor& g, const Tensor& s,
                                                const Tensor& gate, const Tensor& up)
{
    const auto one = ConstLeaf{Scalar{1}};
    return leaf(g) * leaf(up) * (leaf(s) * (one + leaf(gate) * (one - leaf(s))));
}
[[nodiscard]] inline auto make_swiglu_grad_up(const Tensor& g, const Tensor& s,
                                              const Tensor& gate)
{ return leaf(g) * leaf(gate) * leaf(s); }

} // namespace nn::dsl

#endif // NN_EXPR_DSL_HPP
