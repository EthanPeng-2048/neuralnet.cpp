#ifndef NN_CPU_ENGINE_HPP
#define NN_CPU_ENGINE_HPP

// ── cpu_engine.hpp — CPU 计算引擎实现 ─────────────────────────────────────
// CpuEngine 封装现有 Matrix 方法和 AST（compute::apply），实现
// ComputeEngine 接口。所有操作同步执行，begin_batch/end_batch 为 no-op。
//
// 逐元素运算通过 switch 分发到编译期 AST，保持零开销抽象。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <functional>
#include <limits>

#include "compute_engine.hpp"
#include "algebra_compute.hpp"
#include "algebra_expr.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// CpuEngine — CPU 计算引擎
// ══════════════════════════════════════════════════════════════════════════
class CpuEngine final : public ComputeEngine
{
public:
    [[nodiscard]] Device device() const noexcept override { return Device::CPU; }

    // ── 批处理：CPU 为 no-op（同步执行） ──────────────────────────────────
    [[nodiscard]] Result<void> begin_batch() override { return {}; }
    [[nodiscard]] Result<void> end_batch() override { return {}; }

    // ── 张量工厂 ──────────────────────────────────────────────────────────
    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols) override
    {
        return Tensor::cpu(rows, cols);
    }

    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m) override
    {
        return Tensor::from_matrix(Matrix(m));  // 拷贝，避免外部修改影响
    }

    [[nodiscard]] Result<Matrix> to_matrix(const Tensor& t) override
    {
        if (!t.is_cpu())
            return std::unexpected(Error{"to_matrix: tensor is not CPU"});
        return Matrix(t.cpu_matrix());  // 返回拷贝
    }

    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src) override
    {
        if (!dst.is_cpu())
            return std::unexpected(Error{"copy_from: dst tensor is not CPU"});
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"copy_from: shape mismatch"});
        dst.cpu_matrix() = src;  // 拷贝赋值
        return {};
    }

    [[nodiscard]] Result<Tensor> clone(const Tensor& src) override
    {
        if (!src.is_cpu())
            return std::unexpected(Error{"clone: src tensor is not CPU"});
        return Tensor::from_matrix(Matrix(src.cpu_matrix()));  // 深拷贝
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB) override
    {
        if (A.is_gpu() || B.is_gpu())
            return std::unexpected(Error{"CpuEngine: GPU tensor on CPU engine"});

        const Matrix& a = A.cpu_matrix();
        const Matrix& b = B.cpu_matrix();

        // 维度校验（基于逻辑维度，避免不必要的转置拷贝）
        // A_eff = transA ? A^T : A，B_eff = transB ? B^T : B
        const std::size_t M  = transA ? a.cols() : a.rows();
        const std::size_t K  = transA ? a.rows() : a.cols();
        const std::size_t K2 = transB ? b.cols() : b.rows();
        const std::size_t N  = transB ? b.rows() : b.cols();
        if (K != K2)
            return std::unexpected(Error{"matmul: dimension mismatch"});

        Matrix result(M, N);
        // 使用 Matrix 原生转置 matmul 方法，零额外拷贝
        if (!transA && !transB) {
            a.multiply_to(result, b);
        } else if (!transA && transB) {
            a.multiply_transposed_to(result, b);  // C = A × B^T
        } else if (transA && !transB) {
            a.transpose_multiply_to(result, b);   // C = A^T × B
        } else {
            // 双转置 C = A^T × B^T，罕见路径
            Matrix a_t = a.transpose();
            a_t.multiply_transposed_to(result, b);
        }
        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"add_inplace: shape mismatch"});
        A.cpu_matrix().add_inplace(B.cpu_matrix());
        return {};
    }

    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s) override
    {
        A.cpu_matrix().scale_inplace(s);
        return {};
    }

    [[nodiscard]] Result<void> zero(Tensor& A) override
    {
        A.cpu_matrix().zero();
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> row_reduce_sum(const Tensor& A) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result = m.row_reduce(Scalar{0},
            [](Scalar a, Scalar b) noexcept { return a + b; },
            [](Scalar x) noexcept { return x; });
        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum(const Tensor& A) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result = m.col_reduce(Scalar{0},
            [](Scalar a, Scalar b) noexcept { return a + b; },
            [](Scalar x) noexcept { return x; });
        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<Tensor> row_reduce_max(const Tensor& A) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result = m.row_reduce(
            std::numeric_limits<Scalar>::lowest(),
            [](Scalar a, Scalar b) noexcept { return std::max(a, b); },
            [](Scalar x) noexcept { return x; });
        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<Tensor> col_reduce_max(const Tensor& A) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result = m.col_reduce(
            std::numeric_limits<Scalar>::lowest(),
            [](Scalar a, Scalar b) noexcept { return std::max(a, b); },
            [](Scalar x) noexcept { return x; });
        return Tensor::from_matrix(std::move(result));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 广播原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) override
    {
        // 使用具体 lambda（非 std::function）以允许内联和向量化
        auto& a = A.cpu_matrix();
        const auto& rv = row_vec.cpu_matrix();
        switch (op)
        {
        case BinaryOp::Add: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return x + y; }); break;
        case BinaryOp::Sub: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return x - y; }); break;
        case BinaryOp::Mul: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return x * y; }); break;
        case BinaryOp::Div: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return x / y; }); break;
        case BinaryOp::Max: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return std::max(x, y); }); break;
        case BinaryOp::Min: a.broadcast_row_inplace(rv, [](Scalar x, Scalar y) noexcept { return std::min(x, y); }); break;
        }
        return {};
    }

    [[nodiscard]] Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) override
    {
        auto& a = A.cpu_matrix();
        const auto& cv = col_vec.cpu_matrix();
        switch (op)
        {
        case BinaryOp::Add: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return x + y; }); break;
        case BinaryOp::Sub: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return x - y; }); break;
        case BinaryOp::Mul: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return x * y; }); break;
        case BinaryOp::Div: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return x / y; }); break;
        case BinaryOp::Max: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return std::max(x, y); }); break;
        case BinaryOp::Min: a.broadcast_col_inplace(cv, [](Scalar x, Scalar y) noexcept { return std::min(x, y); }); break;
        }
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 逐元素原语（通过 AST switch 分发）
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> elementwise_unary(
        UnaryOp op, const Tensor& A) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result(m.rows(), m.cols());
        ConstSpan in = m.span();
        Span out = result.span();

        switch (op)
        {
        case UnaryOp::Neg:   compute::apply(out, neg(in));   break;
        case UnaryOp::Exp:   compute::apply(out, exp(in));   break;
        case UnaryOp::Log:   compute::apply(out, log(in));   break;
        case UnaryOp::Sqrt:  compute::apply(out, sqrt(in));  break;
        case UnaryOp::Rsqrt: compute::apply(out, rsqrt(in)); break;
        case UnaryOp::Abs:   compute::apply(out, abs(in));   break;
        case UnaryOp::Tanh:  compute::apply(out, tanh(in));  break;
        }

        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<Tensor> elementwise_binary(
        BinaryOp op, const Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"elementwise_binary: shape mismatch"});

        const Matrix& ma = A.cpu_matrix();
        const Matrix& mb = B.cpu_matrix();
        Matrix result(ma.rows(), ma.cols());
        ConstSpan a = ma.span();
        ConstSpan b = mb.span();
        Span out = result.span();

        switch (op)
        {
        case BinaryOp::Add: compute::apply(out, a + b); break;
        case BinaryOp::Sub: compute::apply(out, a - b); break;
        case BinaryOp::Mul: compute::apply(out, a * b); break;
        case BinaryOp::Div: compute::apply(out, a / b); break;
        case BinaryOp::Max: compute::apply(out, max(a, b)); break;
        case BinaryOp::Min: compute::apply(out, min(a, b)); break;
        }

        return Tensor::from_matrix(std::move(result));
    }

    [[nodiscard]] Result<Tensor> elementwise_binary_scalar(
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first) override
    {
        const Matrix& m = A.cpu_matrix();
        Matrix result(m.rows(), m.cols());
        ConstSpan a = m.span();
        Span out = result.span();

        if (scalar_first)
        {
            // out = op(scalar, A)
            switch (op)
            {
            case BinaryOp::Add: compute::apply(out, s + a); break;
            case BinaryOp::Sub: compute::apply(out, s - a); break;
            case BinaryOp::Mul: compute::apply(out, s * a); break;
            case BinaryOp::Div: compute::apply(out, s / a); break;
            case BinaryOp::Max: compute::apply(out, max(Val{s}, a)); break;
            case BinaryOp::Min: compute::apply(out, min(Val{s}, a)); break;
            }
        }
        else
        {
            // out = op(A, scalar)
            switch (op)
            {
            case BinaryOp::Add: compute::apply(out, a + s); break;
            case BinaryOp::Sub: compute::apply(out, a - s); break;
            case BinaryOp::Mul: compute::apply(out, a * s); break;
            case BinaryOp::Div: compute::apply(out, a / s); break;
            case BinaryOp::Max: compute::apply(out, max(a, s)); break;
            case BinaryOp::Min: compute::apply(out, min(a, s)); break;
            }
        }

        return Tensor::from_matrix(std::move(result));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 条件选择原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> elementwise_select_scalar_cond(
        CompareOp cmp, const Tensor& A, Scalar scalar_b,
        const Tensor& then_t, Scalar scalar_else) override
    {
        if (A.rows() != then_t.rows() || A.cols() != then_t.cols())
            return std::unexpected(Error{"elementwise_select: A and then shape mismatch"});

        const Matrix& ma = A.cpu_matrix();
        const Matrix& mt = then_t.cpu_matrix();
        Matrix result(ma.rows(), ma.cols());
        auto a = ma.span();
        auto t = mt.span();
        auto out = result.span();
        const std::size_t n = a.size();

        // 手动循环避免 AST 对 == / != 的 ConstSpan 支持缺失
        switch (cmp)
        {
        case CompareOp::Lt:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] < scalar_b) ? t[i] : scalar_else;
            break;
        case CompareOp::Le:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] <= scalar_b) ? t[i] : scalar_else;
            break;
        case CompareOp::Gt:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] > scalar_b) ? t[i] : scalar_else;
            break;
        case CompareOp::Ge:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] >= scalar_b) ? t[i] : scalar_else;
            break;
        case CompareOp::Eq:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] == scalar_b) ? t[i] : scalar_else;
            break;
        case CompareOp::Ne:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (a[i] != scalar_b) ? t[i] : scalar_else;
            break;
        }

        return Tensor::from_matrix(std::move(result));
    }

};

} // namespace nn

#endif // NN_CPU_ENGINE_HPP
