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
#include "expr_registry.hpp"
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
    // 归约视图输入：该输入被归约为每行/每列一个标量（(rows,1)/(1,cols)），
    // 求值期自动广播——参与算术时按当前 (r,c) 读取对应标量。
    ExprOperand add_reduce_input(const Tensor& t, ExprViewKind kind)
    {
        ExprView v;
        v.kind = static_cast<std::uint8_t>(kind);
        spec.views.push_back(v);
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    // 归约指令：对某表达式结果（操作数 a）归约 → 隐式"每行/每列一个标量"
    // 的归约向量，后续指令经 Reduce 操作数按行/列广播访问。
    ExprOperand add_reduce_instr(ExprOp op, ExprOperand a)
    {
        ExprInstr in;
        in.op = static_cast<std::uint8_t>(op);
        in.dst = static_cast<std::uint8_t>(spec.num_regs++);
        in.a = a;
        spec.instrs.push_back(in);
        return expr::reduce(in.dst);
    }
    // 广播视图：输入本身已是 (rows,1)/(1,cols) 小向量，按行/列广播
    ExprOperand add_input_rowbroadcast(const Tensor& t)
    {
        spec.views.push_back(expr::row_broadcast());
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_colbroadcast(const Tensor& t)
    {
        spec.views.push_back(expr::col_broadcast());
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
// 归约叶子：把"按行/按列归约出标量向量"接入表达式
//
// 两类（对应 expr_spec.hpp 的归约视图与归约指令两种机制）：
//   1. ReduceViewRef<Kind>：对输入 Tensor 直接归约 → 归约**视图**。
//      求值期代表一个 (rows,1)/(1,cols) 的广播向量，参与算术时自动广播。
//   2. ReduceRef<E, Rop>：对子表达式结果归约 → 归约**指令**，
//      用于 exp(x) 求和这类"表达式内部归约"（见 DslExpr 概念定义之后）。
//
// 含归约的表达式**不实例化**模板 CPU 求值路径（has_reduction_v 把 dsl::compute
// 分流到引擎 eval_expr，由引擎实现归约视图/指令语义），故 eval 仅需满足
// nn::Expression 概念约束，返回占位值。
// ══════════════════════════════════════════════════════════════════════════
template <ExprViewKind Kind>
struct ReduceViewRef
{
    Tensor t;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_reduce_input(t, Kind); }
};

using RowReduceSumRef = ReduceViewRef<ExprViewKind::RowReduceSum>;
using RowReduceMaxRef = ReduceViewRef<ExprViewKind::RowReduceMax>;
using ColReduceSumRef = ReduceViewRef<ExprViewKind::ColReduceSum>;
using ColReduceMaxRef = ReduceViewRef<ExprViewKind::ColReduceMax>;

// 广播视图叶子：输入 (rows,1)/(1,cols) 小向量，按行/列广播参与算术。
// 与归约叶子同理：eval 仅满足概念约束（模板求值路径不实例化，has_reduction_v 分流
// 到 eval_expr，由引擎按 (r,c) 索引广播值）。
template <ExprViewKind Kind>
struct BroadcastRef
{
    Tensor t;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& b) const
    {
        if constexpr (Kind == ExprViewKind::RowBroadcast)
            return b.add_input_rowbroadcast(t);
        else
            return b.add_input_colbroadcast(t);
    }
};

using RowBroadcastRef = BroadcastRef<ExprViewKind::RowBroadcast>;
using ColBroadcastRef = BroadcastRef<ExprViewKind::ColBroadcast>;

// ══════════════════════════════════════════════════════════════════════════
// DSL 表达式概念（比 nn::Expression 更严格：额外要求可折叠成 ExprSpec）
//
// 用于让 DSL 运算符在约束偏序上严格优先于旧代数 nn::operator* 等，从而
// 在 namespace nn（Layer）里**直接写内联数学表达式**时消除重载歧义（旧
// 代数与 DSL 共用 nn::Expression；若不区分二者，二者对 DSL 叶子同为候选）。
// ══════════════════════════════════════════════════════════════════════════
template <typename T>
concept DslExpr = nn::Expression<T> && requires(SpecBuilder& b, const T& t)
{
    { t.to_spec(b) } -> std::convertible_to<ExprOperand>;
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
    {
        // 显式固定操作数折叠顺序（l 先 r 后）：C++ 函数实参求值顺序未指定，
        // 直接 b.add_instr(.., l.to_spec(b), r.to_spec(b)) 会让 views/inputs 的
        // 登记顺序随编译器（MSVC 左到右 / Clang 右到左）漂移 → expr_spec_key
        // 跨编译器不稳定，破坏 AOT 匹配。先求值到局部变量以固定顺序。
        const ExprOperand lo = l.to_spec(b);
        const ExprOperand ro = r.to_spec(b);
        return b.add_instr(Op::op_id(), lo, ro);
    }
};

// select(cond, then, else) —— cond 为 BoolExpression，then/else 为值表达式
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
struct Select
{
    C cond; T then_e; E else_e;
    [[nodiscard]] Scalar eval(std::size_t i) const
    { return cond.eval(i) ? then_e.eval(i) : else_e.eval(i); }
    ExprOperand to_spec(SpecBuilder& b) const
    {
        // 同 Binary：显式固定折叠顺序（cond → then → else），保证跨编译器确定。
        const ExprOperand co = cond.to_spec(b);
        const ExprOperand to = then_e.to_spec(b);
        const ExprOperand eo = else_e.to_spec(b);
        return b.add_instr(ExprOp::Select, co, to, eo);
    }
};

// 归约节点：对子表达式结果做归约（归约**指令**）→ 隐式标量向量（Reduce 操作数）。
//   row_reduce_sum(exp(x)) 这类"表达式内部归约"；对输入 Tensor 直接归约请用
//   ReduceViewRef（归约视图，GPU 融合更友好）。
template <nn::dsl::DslExpr E, ExprOp Rop>
struct ReduceRef
{
    E child;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_reduce_instr(Rop, child.to_spec(b)); }
};

// ══════════════════════════════════════════════════════════════════════════
// has_reduction_v：表达式树是否含归约（视图叶子或归约指令节点）
//
// 用于 dsl::compute 的 CPU 路径分流：含归约的表达式无法按"逐元素模板求值"
// （归约需要全行/全列信息），折叠成 ExprSpec 走引擎 eval_expr（CPU 扩展语义
// 处理归约视图/指令）；不含归约的表达式保持原编译期模板求值（零开销）。
// ══════════════════════════════════════════════════════════════════════════
template <typename T> inline constexpr bool has_reduction_v = false;

// 归约/广播叶子均需走 eval_expr（模板求值无法表达全行/全列归约或跨网格广播）
template <ExprViewKind K>
inline constexpr bool has_reduction_v<ReduceViewRef<K>> = true;
template <ExprViewKind K>
inline constexpr bool has_reduction_v<BroadcastRef<K>> = true;
template <nn::dsl::DslExpr E, ExprOp Rop>
inline constexpr bool has_reduction_v<ReduceRef<E, Rop>> = true;
template <typename Op, nn::Expression C>
inline constexpr bool has_reduction_v<Unary<Op, C>> = has_reduction_v<C>;
template <typename Op, nn::Expression L, nn::Expression R>
inline constexpr bool has_reduction_v<Binary<Op, L, R>> = has_reduction_v<L> || has_reduction_v<R>;
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
inline constexpr bool has_reduction_v<Select<C, T, E>>
    = has_reduction_v<C> || has_reduction_v<T> || has_reduction_v<E>;

// ══════════════════════════════════════════════════════════════════════════
// 叶子构造（普通写法入口）
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline TensorRef leaf(Tensor t) { return TensorRef{std::move(t)}; }

[[nodiscard]] inline RotateHalfRef rotate_half(Tensor t, std::uint32_t block)
{ return RotateHalfRef{std::move(t), block}; }

[[nodiscard]] inline RowModRef row_mod(Tensor t, std::uint32_t mod)
{ return RowModRef{std::move(t), mod}; }

// ══════════════════════════════════════════════════════════════════════════
// 归约自由函数：对输入 Tensor 直接归约 → 归约**视图**（GPU 融合更友好）；
// 对表达式结果归约 → 归约**指令**（重载按参数类型自动选择，见 ReduceRef）。
// 返回的归约叶子参与算术时自动按行/按列广播。
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline RowReduceSumRef row_reduce_sum(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline RowReduceMaxRef row_reduce_max(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColReduceSumRef col_reduce_sum(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColReduceMaxRef col_reduce_max(Tensor t) { return {std::move(t)}; }

template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_sum(const E& e) { return ReduceRef<E, ExprOp::RowSum>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_max(const E& e) { return ReduceRef<E, ExprOp::RowMax>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_sum(const E& e) { return ReduceRef<E, ExprOp::ColSum>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_max(const E& e) { return ReduceRef<E, ExprOp::ColMax>{e}; }

// 广播视图自由函数：gamma/beta (F,1) 按行广播、std_inv (1,B) 按列广播
[[nodiscard]] inline RowBroadcastRef row_broadcast(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColBroadcastRef col_broadcast(Tensor t) { return {std::move(t)}; }

// ══════════════════════════════════════════════════════════════════════════
// 一元函数（普通数学写法）
// ══════════════════════════════════════════════════════════════════════════
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto neg(const E& e) { return Unary<ops::Neg, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto abs(const E& e)   { return Unary<ops::Abs, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto exp(const E& e)   { return Unary<ops::Exp, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto log(const E& e)   { return Unary<ops::Log, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto sqrt(const E& e)  { return Unary<ops::Sqrt, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto rsqrt(const E& e) { return Unary<ops::Rsqrt, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto tanh(const E& e)  { return Unary<ops::Tanh, E>{e}; }

// ══════════════════════════════════════════════════════════════════════════
// 二元函数：max / min / select
// ══════════════════════════════════════════════════════════════════════════
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> [[nodiscard]] constexpr auto max(const L& l, const R& r) { return Binary<ops::Max, L, R>{l, r}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto max(const E& e, Scalar s) { return Binary<ops::Max, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto max(Scalar s, const E& e) { return Binary<ops::Max, ConstLeaf, E>{ConstLeaf{s}, e}; }
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> [[nodiscard]] constexpr auto min(const L& l, const R& r) { return Binary<ops::Min, L, R>{l, r}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto min(const E& e, Scalar s) { return Binary<ops::Min, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto min(Scalar s, const E& e) { return Binary<ops::Min, ConstLeaf, E>{ConstLeaf{s}, e}; }

/// relu(x) = max(x, 0)
template <nn::dsl::DslExpr E> [[nodiscard]] auto relu(const E& e) { return nn::dsl::max(e, ConstLeaf{Scalar{0}}); }

template <nn::BoolExpression C, nn::dsl::DslExpr T, nn::dsl::DslExpr E>
[[nodiscard]] constexpr auto select(const C& c, const T& t, const E& e)
{ return Select<C, T, E>{c, t, e}; }
template <nn::BoolExpression C, nn::dsl::DslExpr T>
[[nodiscard]] constexpr auto select(const C& c, const T& t, Scalar ev)
{ return Select<C, T, ConstLeaf>{c, t, ConstLeaf{ev}}; }
template <nn::BoolExpression C, nn::dsl::DslExpr E>
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
    template <nn::dsl::DslExpr L, nn::dsl::DslExpr R>                                     \
    [[nodiscard]] constexpr auto operator OP(const L& l, const R& r)                      \
    { return Binary<OpT, L, R>{l, r}; }                                                   \
    template <nn::dsl::DslExpr E>                                                         \
    [[nodiscard]] constexpr auto operator OP(const E& e, Scalar s)                        \
    { return Binary<OpT, E, ConstLeaf>{e, ConstLeaf{s}}; }                                \
    template <nn::dsl::DslExpr E>                                                         \
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

template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto operator-(const E& e)
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
#ifdef NN_EXPR_SCAN
    // 构建期扫描模式：折叠内联表达式的**结构**并登记进全局注册表，
    // 返回占位张量让 dry-run 流程继续（scan 只关心表达式集合，不真算）。
    // 表达式文本仍只出现在 Layer；这里登记的是派生物 ExprSpec。
    (void)eng;
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    fused::global_registry().add(spec);
    return Tensor::cpu(rows, cols);
#else
    if (eng.device() == Device::CPU)
    {
        if constexpr (nn::dsl::has_reduction_v<E>)
        {
            // 含归约：模板求值无法表达"全行/全列归约"，折叠成 ExprSpec 走
            // 引擎 eval_expr（CPU 扩展语义处理归约视图/指令，先正确后优化）
            auto [spec, inputs] = to_expr_spec(e);
            if (auto v = validate_expr_spec(spec, inputs.size()); !v)
                return std::unexpected(v.error());
            return eng.eval_expr(spec, inputs, rows, cols);
        }
        return eval_cpu(e, rows, cols);
    }

    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr(spec, inputs, rows, cols);  // 闭合世界：GPU 未命中即报错
#endif
}

// ══════════════════════════════════════════════════════════════════════════
// start_expr / end_expr — 把"无法写进一行"的表达式按块融合
//
// 与 compute() 等价：内联数学表达式只出现在这里（Layer），end_expr 时
// 整块一次融合（AOT 收集 / CPU 融合 / GPU 匹配预编译 shader）。跨多行
// 书写同一表达式，框内即一个融合单元。
//   auto out = dsl::end_expr(dsl::start_expr(engine, rows, cols,
//       leaf(a) * leaf(b)
//       + leaf(c) * Scalar{2}
//       - leaf(d)));
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
struct ExprBlock
{
    ComputeEngine* eng;
    std::size_t    rows, cols;
    E              expr;
};

template <typename E>
[[nodiscard]] ExprBlock<E> start_expr(ComputeEngine& eng, std::size_t rows,
                                      std::size_t cols, const E& e)
{ return {&eng, rows, cols, e}; }

template <typename E>
[[nodiscard]] Result<Tensor> end_expr(const ExprBlock<E>& b)
{ return compute(*b.eng, b.expr, b.rows, b.cols); }

// ══════════════════════════════════════════════════════════════════════════
// 归约向量原生形状输出：compute_reduce(engine, expr, rows, cols)
//
// 与 compute() 等价，但输出为归约向量本身（(rows,1)/(1,cols)），而非广播到
// (rows,cols)。用于 LayerNorm/RMSNorm 的 (1,B) 统计量缓存（mean/var/rms_inv）
// 与 (F,1) 梯度归约（grad_gamma/grad_beta）——只产出小向量，避免写全尺寸广播。
// 要求表达式归约轴为 0/1（否则引擎报错）。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Result<Tensor> compute_reduce(ComputeEngine& eng, const E& e,
                                            std::size_t rows, std::size_t cols)
{
#ifdef NN_EXPR_SCAN
    // 构建期扫描：同 compute()，登记结构（归约轴由 gen_fused 判定）。
    // 占位张量按归约轴取向量形状 (rows,1)/(1,cols)，使 Layer 后续
    // add_inplace 等形状相关操作在 dry-run 中不因形状失配而中断。
    (void)eng;
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    fused::global_registry().add(spec);
    const int raxis = expr_spec_reduce_axis(spec);
    if (raxis == 0) return Tensor::cpu(rows, 1);
    if (raxis == 1) return Tensor::cpu(1, cols);
    return Tensor::cpu(rows, cols);
#else
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr_reduce(spec, inputs, rows, cols);
#endif
}

} // namespace nn::dsl

#endif // NN_EXPR_DSL_HPP
