#ifndef NN_EXPR_EVAL_HPP
#define NN_EXPR_EVAL_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_eval.hpp — 逐元素表达式 DSL：跨后端共享执行器
//
//  数据结构见 expr_spec.hpp（ExprSpec/ExprOp/ExprView...）。
//
//  执行策略分层：
//    - CPU：CpuEngine::eval_expr 使用融合解释器（一次遍历，真融合）
//    - Vulkan/CUDA v1：CpuEngine 之外的 Gpu/CudaEngine::eval_expr 复用本文件
//      的 run_expr_eager —— 把表达式降级（lowering）为 ComputeEngine 现有
//      原语调用，中间结果物化为临时 Tensor。**正确优先，性能后续由统一
//      VM / JIT 提升**。未来替换点在各后端 eval_expr 内部实现，DSL 与
//      Layer 均不变。
//
//  数值/覆盖说明（eager 路径）：
//    - 张量 vs 张量的比较（比较指令两操作数均非常量）v1 不支持，报错；
//      常量 vs 常量应折叠到构建侧。未来 VM 消除此限制。
//    - Select 的 else 为张量时用 `b*cnd + c*(1-cnd)` 组合实现（cnd 为
//      0/1 掩码），要求 b/c 均为有限值（无 ±inf 冲突）——注意掩码常含
//      -inf 的场景请用 Add 表达，勿走 Select。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "config.hpp"
#include "core_errors.hpp"
#include "tensor.hpp"
#include "compute_engine.hpp"
#include "expr_spec.hpp"

namespace nn
{

namespace detail
{

// 构造全 1 / 全 0 张量（eager 路径用于比较/选择）。v1 用 from_matrix
// （GPU 上一次上传）；未来 VM 中由常量折叠消除。
[[nodiscard]] inline Result<Tensor> make_filled(
    ComputeEngine& engine, std::size_t rows, std::size_t cols, Scalar v)
{
    Matrix m(rows, cols);
    auto s = m.span();
    std::fill(s.begin(), s.end(), v);
    return engine.from_matrix(m);
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
// run_expr_eager — 跨后端共享 eager 执行器（v1 lowering）
//
// 逐指令把表达式降级为 ComputeEngine 现有原语调用。视图（RotateHalf /
// RowMod）通过 slice_rows/insert_rows 物化。常量操作数用标量原语。
// 输出 = 最后一条指令的目标寄存器张量。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline Result<Tensor> run_expr_eager(
    ComputeEngine& engine,
    const ExprSpec& spec,
    std::span<const Tensor> inputs,
    std::size_t rows, std::size_t cols)
{
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    // 列数必须一致；行数由视图语义决定（RowMod 允许短表、RotateHalf 允许长表）
    for (std::size_t k = 0; k < inputs.size(); ++k)
    {
        const Tensor& t = inputs[k];
        if (t.cols() != cols)
            return std::unexpected(Error{"run_expr_eager: input cols mismatch"});
        const ExprView& v = spec.views[k];
        switch (v.kind)
        {
        default:
        case static_cast<uint8_t>(ExprViewKind::Linear):
            if (t.rows() != rows)
                return std::unexpected(Error{"run_expr_eager: Linear input rows mismatch"});
            break;
        case static_cast<uint8_t>(ExprViewKind::RotateHalf):
            if (t.rows() != rows || v.param == 0 || rows % v.param != 0 || v.param % 2 != 0)
                return std::unexpected(Error{"run_expr_eager: RotateHalf shape/param invalid"});
            break;
        case static_cast<uint8_t>(ExprViewKind::RowMod):
            if (t.rows() != v.param || v.param == 0 || rows % v.param != 0)
                return std::unexpected(Error{"run_expr_eager: RowMod shape/param invalid"});
            break;
        }
    }

    // 寄存器 → 物化 Tensor（空 = 未赋值）
    std::vector<Tensor> reg_tensors(spec.num_regs);

    // ── 视图物化 ──────────────────────────────────────────────────────────
    auto materialize_input = [&](std::size_t k) -> Result<Tensor>
    {
        const ExprView& v  = spec.views[k];
        const Tensor&   src = inputs[k];
        const std::size_t src_rows = src.rows();

        if (v.kind == static_cast<uint8_t>(ExprViewKind::Linear))
            return Tensor(src);  // 共享所有权，零拷贝

        if (v.kind == static_cast<uint8_t>(ExprViewKind::RotateHalf))
        {
            const std::size_t block = v.param;
            if (block == 0 || src_rows % block != 0 || block % 2 != 0)
                return std::unexpected(Error{"run_expr_eager: bad RotateHalf param"});
            const std::size_t half = block / 2;
            const std::size_t nblk = src_rows / block;
            auto out = engine.create_tensor(rows, cols);
            for (std::size_t b = 0; b < nblk; ++b)
            {
                auto src_hi = engine.slice_rows(src, b * block + half, half);  // 源后半
                if (!src_hi) return std::unexpected(src_hi.error());
                if (v.negate_first_half)
                {
                    auto neg = engine.elementwise_unary(UnaryOp::Neg, *src_hi);
                    if (!neg) return std::unexpected(neg.error());
                    auto ir = engine.insert_rows(out, b * block, *neg);
                    if (!ir) return std::unexpected(ir.error());
                }
                else
                {
                    auto ir = engine.insert_rows(out, b * block, *src_hi);
                    if (!ir) return std::unexpected(ir.error());
                }
                auto src_lo = engine.slice_rows(src, b * block, half);         // 源前半
                if (!src_lo) return std::unexpected(src_lo.error());
                auto ir2 = engine.insert_rows(out, b * block + half, *src_lo);
                if (!ir2) return std::unexpected(ir2.error());
            }
            return out;
        }

        if (v.kind == static_cast<uint8_t>(ExprViewKind::RowMod))
        {
            const std::size_t mod = v.param;
            if (mod == 0 || src_rows != mod || rows % mod != 0)
                return std::unexpected(Error{"run_expr_eager: RowMod requires rows%mod==0"});
            auto out = engine.create_tensor(rows, cols);
            const std::size_t nblk = rows / mod;
            for (std::size_t b = 0; b < nblk; ++b)
            {
                auto ir = engine.insert_rows(out, b * mod, src);
                if (!ir) return std::unexpected(ir.error());
            }
            return out;
        }

        return std::unexpected(Error{"run_expr_eager: unknown view kind"});
    };

    // ── 操作数 → Tensor（Reg/Input；Const 由各 op 分支用标量原语处理） ──
    auto eval_tensor_operand = [&](const ExprOperand& op) -> Result<Tensor>
    {
        if (op.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
            op.kind == static_cast<uint8_t>(ExprOperandKind::Fanout))
            return reg_tensors[op.idx];
        if (op.kind == static_cast<uint8_t>(ExprOperandKind::Input))
            return materialize_input(op.idx);
        return std::unexpected(Error{"run_expr_eager: const operand where tensor required"});
    };

    const auto is_const = [](const ExprOperand& op)
    { return op.kind == static_cast<uint8_t>(ExprOperandKind::Const); };

    for (const auto& ins : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);

        // ── 一元 ──────────────────────────────────────────────────────────
        if (op == ExprOp::Neg || op == ExprOp::Exp || op == ExprOp::Log ||
            op == ExprOp::Sqrt || op == ExprOp::Rsqrt || op == ExprOp::Abs ||
            op == ExprOp::Tanh)
        {
            if (is_const(ins.a))
                return std::unexpected(Error{"run_expr_eager: unary on const unsupported (fold at build)"});
            auto src = eval_tensor_operand(ins.a);
            if (!src) return src;
            UnaryOp u = UnaryOp::Neg;
            switch (op)
            {
                case ExprOp::Neg:   u = UnaryOp::Neg;   break;
                case ExprOp::Exp:   u = UnaryOp::Exp;   break;
                case ExprOp::Log:   u = UnaryOp::Log;   break;
                case ExprOp::Sqrt:  u = UnaryOp::Sqrt;  break;
                case ExprOp::Rsqrt: u = UnaryOp::Rsqrt; break;
                case ExprOp::Abs:   u = UnaryOp::Abs;   break;
                case ExprOp::Tanh:  u = UnaryOp::Tanh;  break;
                default: break;
            }
            auto r = engine.elementwise_unary(u, *src);
            if (!r) return r;
            reg_tensors[ins.dst] = std::move(*r);
            continue;
        }

        // ── 二元（支持常量操作数 → 标量原语） ────────────────────────────
        if (op == ExprOp::Add || op == ExprOp::Sub || op == ExprOp::Mul ||
            op == ExprOp::Div || op == ExprOp::Max || op == ExprOp::Min)
        {
            BinaryOp b = BinaryOp::Add;
            switch (op)
            {
                case ExprOp::Add: b = BinaryOp::Add; break;
                case ExprOp::Sub: b = BinaryOp::Sub; break;
                case ExprOp::Mul: b = BinaryOp::Mul; break;
                case ExprOp::Div: b = BinaryOp::Div; break;
                case ExprOp::Max: b = BinaryOp::Max; break;
                case ExprOp::Min: b = BinaryOp::Min; break;
                default: break;
            }
            const bool a_c = is_const(ins.a);
            const bool b_c = is_const(ins.b);
            if (a_c && b_c)
                return std::unexpected(Error{"run_expr_eager: const-op-const unsupported (fold at build)"});
            if (a_c)
            {
                auto src = eval_tensor_operand(ins.b);
                if (!src) return src;
                auto r = engine.elementwise_binary_scalar(b, *src,
                            spec.consts[ins.a.idx], /*scalar_first=*/true);
                if (!r) return r;
                reg_tensors[ins.dst] = std::move(*r);
            }
            else if (b_c)
            {
                auto src = eval_tensor_operand(ins.a);
                if (!src) return src;
                auto r = engine.elementwise_binary_scalar(b, *src,
                            spec.consts[ins.b.idx], /*scalar_first=*/false);
                if (!r) return r;
                reg_tensors[ins.dst] = std::move(*r);
            }
            else
            {
                auto sa = eval_tensor_operand(ins.a);
                if (!sa) return sa;
                auto sb = eval_tensor_operand(ins.b);
                if (!sb) return sb;
                auto r = engine.elementwise_binary(b, *sa, *sb);
                if (!r) return r;
                reg_tensors[ins.dst] = std::move(*r);
            }
            continue;
        }

        // ── 比较：dst = (a cmp b) ? 1 : 0 ─────────────────────────────────
        if (op == ExprOp::Lt || op == ExprOp::Le || op == ExprOp::Gt ||
            op == ExprOp::Ge || op == ExprOp::Eq || op == ExprOp::Ne)
        {
            // v1：仅支持"张量 vs 常量"（现有 elementwise_select_scalar_cond
            // 的条件操作数 2 为标量）。张量 vs 张量由未来 VM 支持。
            if (!is_const(ins.b))
                return std::unexpected(Error{
                    "run_expr_eager: tensor-vs-tensor compare not supported in eager (use VM later)"});
            auto A = eval_tensor_operand(ins.a);
            if (!A) return A;
            auto ones = detail::make_filled(engine, rows, cols, Scalar{1});
            if (!ones) return ones;
            CompareOp cmp = CompareOp::Lt;
            switch (op)
            {
                case ExprOp::Lt: cmp = CompareOp::Lt; break;
                case ExprOp::Le: cmp = CompareOp::Le; break;
                case ExprOp::Gt: cmp = CompareOp::Gt; break;
                case ExprOp::Ge: cmp = CompareOp::Ge; break;
                case ExprOp::Eq: cmp = CompareOp::Eq; break;
                case ExprOp::Ne: cmp = CompareOp::Ne; break;
                default: break;
            }
            auto r = engine.elementwise_select_scalar_cond(cmp, *A,
                        spec.consts[ins.b.idx], *ones, Scalar{0});
            if (!r) return r;
            reg_tensors[ins.dst] = std::move(*r);
            continue;
        }

        // ── Select：dst = (a != 0) ? b : c ────────────────────────────────
        if (op == ExprOp::Select)
        {
            auto cond = eval_tensor_operand(ins.a);
            if (!cond) return cond;
            auto then_t = eval_tensor_operand(ins.b);
            if (!then_t) return then_t;

            if (is_const(ins.c))
            {
                // else 为标量 → 直接映射现有原语
                auto r = engine.elementwise_select_scalar_cond(
                    CompareOp::Ne, *cond, Scalar{0}, *then_t, spec.consts[ins.c.idx]);
                if (!r) return r;
                reg_tensors[ins.dst] = std::move(*r);
            }
            else
            {
                // else 为张量 → cnd*b + (1-cnd)*c（要求 b/c 有限值）
                auto else_t = eval_tensor_operand(ins.c);
                if (!else_t) return else_t;
                auto ones = detail::make_filled(engine, rows, cols, Scalar{1});
                if (!ones) return ones;
                auto cnd = engine.elementwise_select_scalar_cond(
                    CompareOp::Ne, *cond, Scalar{0}, *ones, Scalar{0});
                if (!cnd) return cnd;
                // 1-cnd
                auto not_cnd = engine.elementwise_binary_scalar(
                    BinaryOp::Sub, *cnd, Scalar{1}, /*scalar_first=*/true);
                if (!not_cnd) return not_cnd;
                auto p1 = engine.elementwise_binary(BinaryOp::Mul, *then_t, *cnd);
                if (!p1) return p1;
                auto p2 = engine.elementwise_binary(BinaryOp::Mul, *else_t, *not_cnd);
                if (!p2) return p2;
                auto sum = engine.elementwise_binary(BinaryOp::Add, *p1, *p2);
                if (!sum) return sum;
                reg_tensors[ins.dst] = std::move(*sum);
            }
            continue;
        }

        return std::unexpected(Error{"run_expr_eager: unknown op"});
    }

    return reg_tensors[spec.instrs.back().dst];
}

} // namespace nn

#endif // NN_EXPR_EVAL_HPP
