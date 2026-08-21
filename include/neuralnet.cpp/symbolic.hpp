#ifndef NN_SYMBOLIC_HPP
#define NN_SYMBOLIC_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  symbolic.hpp — 符号表达式层（懒构建，单一写法，CPU/GPU 统一）
//
// 在 ExprSpec 之上提供一层直观的 PyTorch 风格符号写法：
//    - matmul(A, B)        矩阵乘（对应 PyTorch 的 A @ B；C++ 无法重载 @）
//    - A * B               逐元素乘法（照抄 PyTorch 的 *）
//    - A + B / A - B / A / B  逐元素（照抄 PyTorch）
//    - mul(A,B) / max(A,B) / relu(A) / ...  逐元素函数式（懒建节点，lambda 风格）
//
//  关键性质：符号树是**懒构建**的——每个操作只建节点、不物化张量。
//  eval() 一次性降级：
//     - 连续逐元素区域 → 单个 ExprSpec → engine.eval_expr
//       （CPU 融合解释器 / GPU AOT 融合 shader 命中）
//     - matmul 节点 → 原语边界（矩阵乘不可逐元素合并）
//  因此"逐元素乘法用函数式 mul()"与"用运算符 *"建出的树完全相同，
//  融合结果一致，写法不影响性能。
//
// 约定（有歧义处照抄 PyTorch）：
//    - * 始终逐元素乘法（PyTorch 的 *）；矩阵乘用 matmul(A,B)（PyTorch 的 @）
//    - + - / 始终逐元素；标量操作数逐元素广播
//    - 逐元素操作数形状必须一致（v1 不支持向量广播，见 eval 的报错）
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "compute_engine.hpp"
#include "expr_spec.hpp"
#include "tensor.hpp"

namespace nn::sym
{

// ── 表达式节点（懒构建、不可变、共享所有权）──────────────────────────────
struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

enum class UnaryKind : uint8_t
{
    Neg = 0, Exp = 1, Log = 2, Sqrt = 3, Rsqrt = 4, Abs = 5, Tanh = 6,
};

enum class BinaryKind : uint8_t
{
    Add = 0, Sub = 1, Mul = 2, Div = 3, Max = 4, Min = 5,
};

struct Expr
{
    enum class Tag : uint8_t { Const, Input, Unary, Binary, Matmul };

    Tag tag = Tag::Input;
    Scalar value = Scalar{0};    // Const
    std::uint32_t input_idx = 0; // Input（eval 时引用 inputs[idx]）
    UnaryKind uop = UnaryKind::Neg;
    BinaryKind bop = BinaryKind::Add;
    ExprPtr lhs;                 // Unary/Binary/Matmul 的左（或唯一）子
    ExprPtr rhs;                 // Binary/Matmul 的右子
};

namespace detail
{
    [[nodiscard]] inline ExprPtr make_const(Scalar v)
    {
        auto e = std::make_shared<Expr>();
        e->tag = Expr::Tag::Const;
        e->value = v;
        return e;
    }

    [[nodiscard]] inline ExprPtr make_input(std::uint32_t idx)
    {
        auto e = std::make_shared<Expr>();
        e->tag = Expr::Tag::Input;
        e->input_idx = idx;
        return e;
    }

    [[nodiscard]] inline ExprPtr make_unary(UnaryKind k, ExprPtr x)
    {
        auto e = std::make_shared<Expr>();
        e->tag = Expr::Tag::Unary;
        e->uop = k;
        e->lhs = std::move(x);
        return e;
    }

    [[nodiscard]] inline ExprPtr make_binary(BinaryKind k, ExprPtr a, ExprPtr b)
    {
        auto e = std::make_shared<Expr>();
        e->tag = Expr::Tag::Binary;
        e->bop = k;
        e->lhs = std::move(a);
        e->rhs = std::move(b);
        return e;
    }

    [[nodiscard]] inline ExprPtr make_matmul(ExprPtr a, ExprPtr b)
    {
        auto e = std::make_shared<Expr>();
        e->tag = Expr::Tag::Matmul;
        e->lhs = std::move(a);
        e->rhs = std::move(b);
        return e;
    }
} // namespace detail

// ── 逐元素函数式 API（懒建节点）──────────────────────────────────────────
[[nodiscard]] inline ExprPtr input(std::uint32_t idx) { return detail::make_input(idx); }

[[nodiscard]] inline ExprPtr matmul(ExprPtr a, ExprPtr b)
{ return detail::make_matmul(std::move(a), std::move(b)); }

[[nodiscard]] inline ExprPtr mul(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Mul, std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr add(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Add, std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr sub(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Sub, std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr div(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Div, std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr max(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Max, std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr min(ExprPtr a, ExprPtr b)
{ return detail::make_binary(BinaryKind::Min, std::move(a), std::move(b)); }

[[nodiscard]] inline ExprPtr neg(ExprPtr a)  { return detail::make_unary(UnaryKind::Neg, std::move(a)); }
[[nodiscard]] inline ExprPtr exp(ExprPtr a)  { return detail::make_unary(UnaryKind::Exp, std::move(a)); }
[[nodiscard]] inline ExprPtr log(ExprPtr a)  { return detail::make_unary(UnaryKind::Log, std::move(a)); }
[[nodiscard]] inline ExprPtr sqrt(ExprPtr a) { return detail::make_unary(UnaryKind::Sqrt, std::move(a)); }
[[nodiscard]] inline ExprPtr tanh(ExprPtr a) { return detail::make_unary(UnaryKind::Tanh, std::move(a)); }
[[nodiscard]] inline ExprPtr abs(ExprPtr a)  { return detail::make_unary(UnaryKind::Abs, std::move(a)); }

// relu(x) = max(x, 0)（PyTorch: torch.relu / F.relu）
[[nodiscard]] inline ExprPtr relu(ExprPtr a) { return max(std::move(a), detail::make_const(Scalar{0})); }

//  ── 运算符重载 ────────────────────────────────────────────────────────────
// * ：始终逐元素乘法（照抄 PyTorch 的 *）；矩阵乘请用 matmul(A, B)（PyTorch 的 @）
// + - / ：始终逐元素；标量操作数逐元素广播
[[nodiscard]] inline ExprPtr operator*(ExprPtr a, ExprPtr b)
{ return mul(std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr operator*(ExprPtr a, Scalar s)
{ return mul(std::move(a), detail::make_const(s)); }
[[nodiscard]] inline ExprPtr operator*(Scalar s, ExprPtr a)
{ return mul(detail::make_const(s), std::move(a)); }

[[nodiscard]] inline ExprPtr operator+(ExprPtr a, ExprPtr b)
{ return add(std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr operator+(ExprPtr a, Scalar s)
{ return add(std::move(a), detail::make_const(s)); }
[[nodiscard]] inline ExprPtr operator+(Scalar s, ExprPtr a)
{ return add(detail::make_const(s), std::move(a)); }

[[nodiscard]] inline ExprPtr operator-(ExprPtr a, ExprPtr b)
{ return sub(std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr operator-(ExprPtr a, Scalar s)
{ return sub(std::move(a), detail::make_const(s)); }
[[nodiscard]] inline ExprPtr operator-(Scalar s, ExprPtr a)
{ return sub(detail::make_const(s), std::move(a)); }
[[nodiscard]] inline ExprPtr operator-(ExprPtr a)
{ return detail::make_unary(UnaryKind::Neg, std::move(a)); }

[[nodiscard]] inline ExprPtr operator/(ExprPtr a, ExprPtr b)
{ return div(std::move(a), std::move(b)); }
[[nodiscard]] inline ExprPtr operator/(ExprPtr a, Scalar s)
{ return div(std::move(a), detail::make_const(s)); }
[[nodiscard]] inline ExprPtr operator/(Scalar s, ExprPtr a)
{ return div(detail::make_const(s), std::move(a)); }

// ═══════════════════════════════════════════════════════════════════════════
// 求值：符号树 → 引擎执行
// ═══════════════════════════════════════════════════════════════════════════
namespace detail
{
    // 符号操作 → ExprOp（融合 spec 指令）
    [[nodiscard]] inline constexpr ExprOp to_expr_op(UnaryKind k)
    {
        switch (k)
        {
        case UnaryKind::Neg:   return ExprOp::Neg;
        case UnaryKind::Exp:   return ExprOp::Exp;
        case UnaryKind::Log:   return ExprOp::Log;
        case UnaryKind::Sqrt:  return ExprOp::Sqrt;
        case UnaryKind::Rsqrt: return ExprOp::Rsqrt;
        case UnaryKind::Abs:   return ExprOp::Abs;
        case UnaryKind::Tanh:  return ExprOp::Tanh;
        }
        return ExprOp::Neg;
    }

    [[nodiscard]] inline constexpr ExprOp to_expr_op(BinaryKind k)
    {
        switch (k)
        {
        case BinaryKind::Add: return ExprOp::Add;
        case BinaryKind::Sub: return ExprOp::Sub;
        case BinaryKind::Mul: return ExprOp::Mul;
        case BinaryKind::Div: return ExprOp::Div;
        case BinaryKind::Max: return ExprOp::Max;
        case BinaryKind::Min: return ExprOp::Min;
        }
        return ExprOp::Add;
    }

    // 符号操作 → ComputeEngine 原语算子（慢路径 / matmul 边界用）
    [[nodiscard]] inline constexpr nn::UnaryOp to_engine_unary(UnaryKind k)
    {
        switch (k)
        {
        case UnaryKind::Neg:   return nn::UnaryOp::Neg;
        case UnaryKind::Exp:   return nn::UnaryOp::Exp;
        case UnaryKind::Log:   return nn::UnaryOp::Log;
        case UnaryKind::Sqrt:  return nn::UnaryOp::Sqrt;
        case UnaryKind::Rsqrt: return nn::UnaryOp::Rsqrt;
        case UnaryKind::Abs:   return nn::UnaryOp::Abs;
        case UnaryKind::Tanh:  return nn::UnaryOp::Tanh;
        }
        return nn::UnaryOp::Neg;
    }

    [[nodiscard]] inline constexpr nn::BinaryOp to_engine_binary(BinaryKind k)
    {
        switch (k)
        {
        case BinaryKind::Add: return nn::BinaryOp::Add;
        case BinaryKind::Sub: return nn::BinaryOp::Sub;
        case BinaryKind::Mul: return nn::BinaryOp::Mul;
        case BinaryKind::Div: return nn::BinaryOp::Div;
        case BinaryKind::Max: return nn::BinaryOp::Max;
        case BinaryKind::Min: return nn::BinaryOp::Min;
        }
        return nn::BinaryOp::Add;
    }

    // 慢路径：逐节点 eager（区域过大 / 校验失败时的回退）
    [[nodiscard]] inline Result<Tensor> eval_slow(
        ComputeEngine& eng, const ExprPtr& n, std::span<const Tensor> inputs)
    {
        switch (n->tag)
        {
        case Expr::Tag::Input:
            if (n->input_idx >= inputs.size())
                return std::unexpected(Error{"sym::eval: input index out of range"});
            return inputs[n->input_idx];
        case Expr::Tag::Matmul:
        {
            auto a = eval_slow(eng, n->lhs, inputs);
            if (!a) return a;
            auto b = eval_slow(eng, n->rhs, inputs);
            if (!b) return b;
            return eng.matmul(*a, *b, false, false);
        }
        case Expr::Tag::Unary:
        {
            auto a = eval_slow(eng, n->lhs, inputs);
            if (!a) return a;
            return eng.elementwise_unary(to_engine_unary(n->uop), *a);
        }
        case Expr::Tag::Binary:
        {
            if (n->lhs->tag == Expr::Tag::Const)
            {
                auto a = eval_slow(eng, n->rhs, inputs);
                if (!a) return a;
                return eng.elementwise_binary_scalar(
                    to_engine_binary(n->bop), *a, n->lhs->value, /*scalar_first=*/true);
            }
            if (n->rhs->tag == Expr::Tag::Const)
            {
                auto a = eval_slow(eng, n->lhs, inputs);
                if (!a) return a;
                return eng.elementwise_binary_scalar(
                    to_engine_binary(n->bop), *a, n->rhs->value, /*scalar_first=*/false);
            }
            auto a = eval_slow(eng, n->lhs, inputs);
            if (!a) return a;
            auto b = eval_slow(eng, n->rhs, inputs);
            if (!b) return b;
            return eng.elementwise_binary(to_engine_binary(n->bop), *a, *b);
        }
        default:
            return std::unexpected(Error{"sym::eval: constant root has no shape"});
        }
    }
} // namespace detail

// ── 统一求值入口 ─────────────────────────────────────────────────────────
// root 中 Input 叶子的下标对应 inputs[i]。返回输出 Tensor（行数/列数由
// matmul 结果与逐元素输入共同决定；逐元素操作数必须同形状）。
[[nodiscard]] inline Result<Tensor> eval(
    ComputeEngine& eng, const ExprPtr& root, std::span<const Tensor> inputs)
{
    std::function<Result<Tensor>(const ExprPtr&)> eval_t;
    eval_t = [&](const ExprPtr& n) -> Result<Tensor>
    {
        switch (n->tag)
        {
        case Expr::Tag::Input:
            if (n->input_idx >= inputs.size())
                return std::unexpected(Error{"sym::eval: input index out of range"});
            return inputs[n->input_idx];

        case Expr::Tag::Matmul: // 原语边界
        {
            auto a = eval_t(n->lhs);
            if (!a) return a;
            auto b = eval_t(n->rhs);
            if (!b) return b;
            return eng.matmul(*a, *b, false, false);
        }

        case Expr::Tag::Const:
            return std::unexpected(Error{"sym::eval: constant root has no shape"});

        case Expr::Tag::Unary:
        case Expr::Tag::Binary: // 整棵逐元素区域 → 单个 ExprSpec（融合）
        {
            ExprSpec spec;
            std::vector<Tensor> leaves;              // 边界张量（Input / matmul 结果）
            std::unordered_map<const Expr*, ExprOperand> op_of; // 节点 → 已分配操作数

            const auto add_leaf = [&](Tensor t) -> std::uint8_t
            {
                spec.views.push_back(expr::linear());
                leaves.push_back(std::move(t));
                return static_cast<std::uint8_t>(leaves.size() - 1);
            };

            // 节点 → 区域操作数：叶子物化，逐元素节点分配寄存器（后序，父引用子）
            std::function<std::optional<ExprOperand>(const ExprPtr&)> operand_of;
            operand_of = [&](const ExprPtr& m) -> std::optional<ExprOperand>
            {
                if (const auto it = op_of.find(m.get()); it != op_of.end())
                    return it->second;
                ExprOperand op;
                switch (m->tag)
                {
                case Expr::Tag::Input:
                    if (m->input_idx >= inputs.size())
                        return std::nullopt;
                    op = expr::input(add_leaf(inputs[m->input_idx]));
                    break;
                case Expr::Tag::Const:
                    spec.consts.push_back(m->value);
                    op = expr::cst(static_cast<std::uint8_t>(spec.consts.size() - 1));
                    break;
                case Expr::Tag::Matmul:
                {
                    auto t = eval_t(m); // 物化 matmul（其内部可再含区域融合）
                    if (!t) return std::nullopt;
                    op = expr::input(add_leaf(std::move(*t)));
                    break;
                }
                case Expr::Tag::Unary:
                {
                    auto a = operand_of(m->lhs);
                    if (!a) return std::nullopt;
                    ExprInstr ins;
                    ins.op = static_cast<std::uint8_t>(to_expr_op(m->uop));
                    ins.dst = static_cast<std::uint8_t>(spec.num_regs);
                    ins.a = *a;
                    ++spec.num_regs;
                    spec.instrs.push_back(ins);
                    op = expr::reg(ins.dst);
                    break;
                }
                case Expr::Tag::Binary:
                {
                    auto a = operand_of(m->lhs);
                    if (!a) return std::nullopt;
                    auto b = operand_of(m->rhs);
                    if (!b) return std::nullopt;
                    ExprInstr ins;
                    ins.op = static_cast<std::uint8_t>(to_expr_op(m->bop));
                    ins.dst = static_cast<std::uint8_t>(spec.num_regs);
                    ins.a = *a;
                    ins.b = *b;
                    ++spec.num_regs;
                    spec.instrs.push_back(ins);
                    op = expr::reg(ins.dst);
                    break;
                }
                }
                op_of.emplace(m.get(), op);
                return op;
            };

            if (!operand_of(n) || leaves.empty()) // 建区域失败 / 全常量无形状
                return detail::eval_slow(eng, n, inputs);
            if (const auto v = validate_expr_spec(spec, leaves.size()); !v)
                return detail::eval_slow(eng, n, inputs);

            const std::size_t rows = leaves[0].rows();
            const std::size_t cols = leaves[0].cols();
            for (const auto& t : leaves)
                if (t.rows() != rows || t.cols() != cols)
                    return std::unexpected(Error{
                        "sym::eval: elementwise operands must share shape (broadcast is v2)"});

            return eng.eval_expr(spec, leaves, rows, cols);
        }
        }
        return std::unexpected(Error{"sym::eval: unreachable"});
    };
    return eval_t(root);
}

} // namespace nn::sym

#endif // NN_SYMBOLIC_HPP
