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

    // ── 行切片：拷贝 src 的行 [start_row, start_row + count) 到新 Tensor ──
    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count) override
    {
        if (!src.is_cpu())
            return std::unexpected(Error{"slice_rows: src tensor is not CPU"});
        const Matrix& m = src.cpu_matrix();
        if (start_row + count > m.rows())
            return std::unexpected(Error{"slice_rows: range out of bounds"});

        Matrix result(count, m.cols());
        const auto src_span = m.span();
        auto dst_span = result.span();
        const std::size_t cols = m.cols();
        for (std::size_t r = 0; r < count; ++r)
            std::copy_n(src_span.begin() + (start_row + r) * cols, cols,
                        dst_span.begin() + r * cols);
        return Tensor::from_matrix(std::move(result));
    }

    // ── 行插入：将 src 的所有行写入 dst 的行 [dst_start_row, ...) ──
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (!dst.is_cpu() || !src.is_cpu())
            return std::unexpected(Error{"insert_rows: tensors are not CPU"});
        if (dst.cols() != src.cols())
            return std::unexpected(Error{"insert_rows: column count mismatch"});
        if (dst_start_row + src.rows() > dst.rows())
            return std::unexpected(Error{"insert_rows: range out of bounds"});

        Matrix& d = dst.cpu_matrix();
        const Matrix& s = src.cpu_matrix();
        const auto dst_span = d.span();
        const auto src_span = s.span();
        const std::size_t cols = d.cols();
        for (std::size_t r = 0; r < s.rows(); ++r)
            std::copy_n(src_span.begin() + r * cols, cols,
                        dst_span.begin() + (dst_start_row + r) * cols);
        return {};
    }

    // ── gather_rows: 按 indices 从 table 中按行查表 ──
    //   table: (vocab, D), indices: (num_indices,), 输出: (num_indices, D)
    //   越界索引返回零行（防御性，不抛错）
    //
    // 并行策略：每个输出行独立，按行切分到线程。
    //   - num * D >= PARALLEL_THRESHOLD 时启用行块并行
    //   - 否则串行（小批量查表的 D 通常较小，并行调度开销不划算）
    //   - 行块并行避免跨行数据竞争（每行独立写入）
    [[nodiscard]] Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices) override
    {
        if (!table.is_cpu() || !indices.is_cpu())
            return std::unexpected(Error{"gather_rows: tensors are not CPU"});

        const Matrix& tbl = table.cpu_matrix();
        const Matrix& idx = indices.cpu_matrix();
        const std::size_t vocab = tbl.rows();
        const std::size_t D = tbl.cols();
        const std::size_t num = idx.rows();  // indices 视为 (num, 1) 形状

        Matrix result(num, D);
        auto dst_span = result.span();
        const auto tbl_span = tbl.span();
        const auto idx_span = idx.span();

        const std::size_t total = num * D;
        if (total >= PARALLEL_THRESHOLD && num > 1)
        {
            // 行块并行：每行独立查表，无数据竞争
            auto row_indices = std::views::iota(std::size_t{0}, num);
            nn::for_each(row_indices.begin(), row_indices.end(),
                [dst_span, tbl_span, idx_span, vocab, D](std::size_t i) noexcept {
                    const auto row_idx = static_cast<std::size_t>(idx_span[i]);
                    if (row_idx < vocab)
                    {
                        std::copy_n(tbl_span.begin() + row_idx * D, D,
                                    dst_span.begin() + i * D);
                    }
                    else
                    {
                        std::fill_n(dst_span.begin() + i * D, D, Scalar{0});
                    }
                });
        }
        else
        {
            // 串行路径
            for (std::size_t i = 0; i < num; ++i)
            {
                const auto row_idx = static_cast<std::size_t>(idx_span[i]);
                if (row_idx < vocab)
                {
                    std::copy_n(tbl_span.begin() + row_idx * D, D,
                                dst_span.begin() + i * D);
                }
                else
                {
                    std::fill_n(dst_span.begin() + i * D, D, Scalar{0});
                }
            }
        }
        return Tensor::from_matrix(std::move(result));
    }

    // ── scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst ──
    //   dst: (vocab, D) 原地修改, indices: (num_indices,), grad: (num_indices, D)
    //   语义: dst[indices[i]] += grad[i]
    [[nodiscard]] Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad) override
    {
        if (!dst.is_cpu() || !indices.is_cpu() || !grad.is_cpu())
            return std::unexpected(Error{"scatter_add_rows: tensors are not CPU"});
        if (dst.cols() != grad.cols())
            return std::unexpected(Error{"scatter_add_rows: column count mismatch"});

        Matrix& d = dst.cpu_matrix();
        const Matrix& idx = indices.cpu_matrix();
        const Matrix& g = grad.cpu_matrix();
        const std::size_t vocab = d.rows();
        const std::size_t D = d.cols();
        const std::size_t num = idx.rows();

        auto dst_span = d.span();
        const auto idx_span = idx.span();
        const auto grad_span = g.span();

        for (std::size_t i = 0; i < num; ++i)
        {
            const auto row_idx = static_cast<std::size_t>(idx_span[i]);
            if (row_idx < vocab)
            {
                Scalar* dst_row = dst_span.data() + row_idx * D;
                const Scalar* grad_row = grad_span.data() + i * D;
                for (std::size_t c = 0; c < D; ++c)
                    dst_row[c] += grad_row[c];
            }
            // 越界索引忽略
        }
        return {};
    }

    // ── 3D 维度转置：(M, B, N) ↔ (B, M, N) ──
    //   inverse=false: (M, B*N) → (B*M, N), out[b*M+m, n] = in[m, b*N+n]
    //   inverse=true:  (B*M, N) → (M, B*N), out[m, b*N+n] = in[b*M+m, n]
    //
    // 并行策略：按 (b, m) 二维块切分，每块独立 N 元素拷贝，无数据竞争。
    //   - total >= PARALLEL_THRESHOLD 时启用块级并行
    //   - 每块仅一次 std::copy_n(N)，cache 友好
    //   - 典型场景：MHA 中 d_model=128, batch=32, seq=256 → 1M 元素，明显受益
    [[nodiscard]] Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse) override
    {
        if (x.is_gpu())
            return std::unexpected(Error{"CpuEngine: GPU tensor on CPU engine"});

        const Matrix& m = x.cpu_matrix();
        const std::size_t total = M * B * N;
        if (m.size() != total)
            return std::unexpected(Error{"rearrange_3d: element count mismatch"});

        Matrix out(inverse ? M : (B * M), inverse ? (B * N) : N);
        const auto src = m.span();
        const auto dst = out.span();

        // (b, m) 组合索引：bm_idx = b * M + m，总块数 = B * M
        // 每块独立处理 N 个元素
        const std::size_t n_blocks = B * M;
        auto block_kernel = [src, dst, M, B, N, inverse](std::size_t bm_idx) noexcept {
            const std::size_t b = bm_idx / M;
            const std::size_t mi = bm_idx % M;
            if (!inverse)
            {
                // out[b*M + m, n] = in[m, b*N + n]
                const std::size_t dst_off = (b * M + mi) * N;
                const std::size_t src_off = mi * (B * N) + b * N;
                std::copy_n(src.begin() + src_off, N,
                            dst.begin() + dst_off);
            }
            else
            {
                // out[m, b*N + n] = in[b*M + m, n]
                const std::size_t src_off = (b * M + mi) * N;
                const std::size_t dst_off = mi * (B * N) + b * N;
                std::copy_n(src.begin() + src_off, N,
                            dst.begin() + dst_off);
            }
        };

        if (total >= PARALLEL_THRESHOLD && n_blocks > 1)
        {
            auto block_indices = std::views::iota(std::size_t{0}, n_blocks);
            nn::parallel_for_blocks(block_indices.begin(), block_indices.end(),
                                       block_kernel);
        }
        else
        {
            for (std::size_t bm = 0; bm < n_blocks; ++bm)
                block_kernel(bm);
        }

        return Tensor::from_matrix(std::move(out));
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
            return std::unexpected(Error{"matmul: dimension mismatch A=" +
                std::to_string(a.rows()) + "x" + std::to_string(a.cols()) +
                " transA=" + (transA ? "1" : "0") +
                " B=" + std::to_string(b.rows()) + "x" + std::to_string(b.cols()) +
                " transB=" + (transB ? "1" : "0") +
                " K=" + std::to_string(K) + " K2=" + std::to_string(K2)});

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

    // ── 批量矩阵乘法：按 batch 切分行块，逐 batch 矩阵乘 ──
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB) override
    {
        if (A.is_gpu() || B.is_gpu())
            return std::unexpected(Error{"CpuEngine: GPU tensor on CPU engine"});
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul: batch must be > 0"});

        const Matrix& a = A.cpu_matrix();
        const Matrix& b = B.cpu_matrix();

        // 每个 batch 的行数
        if (a.rows() % batch != 0 || b.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul: rows not divisible by batch"});

        const std::size_t a_rows_per = a.rows() / batch;
        const std::size_t b_rows_per = b.rows() / batch;

        // 逻辑维度（基于转置标志）
        //   transA=false: A_b 视为 (a_rows_per, a.cols())，M=a_rows_per, K=a.cols()
        //   transA=true:  A_b 视为 (a.cols(), a_rows_per)（按 A_b^T 使用），M=a.cols(), K=a_rows_per
        //   transB=false: B_b 视为 (b_rows_per, b.cols())，K2=b_rows_per, N=b.cols()
        //   transB=true:  B_b 视为 (b.cols(), b_rows_per)（按 B_b^T 使用），K2=b.cols(), N=b_rows_per
        const std::size_t M  = transA ? a.cols() : a_rows_per;
        const std::size_t K  = transA ? a_rows_per : a.cols();
        const std::size_t K2 = transB ? b.cols() : b_rows_per;
        const std::size_t N  = transB ? b_rows_per : b.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul: K dimension mismatch"});

        Matrix result(batch * M, N);
        auto result_span = result.span();
        const auto a_span = a.span();
        const auto b_span = b.span();

        for (std::size_t bi = 0; bi < batch; ++bi)
        {
            // 零拷贝：通过 span 子区间直接引用原始数据，无需临时 Matrix 拷贝
            const std::size_t a_off = bi * a_rows_per * a.cols();
            const std::size_t b_off = bi * b_rows_per * b.cols();
            auto a_sub = std::span<const Scalar>(a_span.data() + a_off, a_rows_per * a.cols());
            auto b_sub = std::span<const Scalar>(b_span.data() + b_off, b_rows_per * b.cols());
            auto c_sub = std::span<Scalar>(result_span.data() + bi * M * N, M * N);

            if (!transA && !transB) {
                Matrix::multiply_to_span(c_sub, M, N,
                    a_sub, a_rows_per, a.cols(),
                    b_sub, b_rows_per, b.cols());
            } else if (!transA && transB) {
                Matrix::multiply_transposed_to_span(c_sub, M, N,
                    a_sub, a_rows_per, a.cols(),
                    b_sub, b_rows_per, b.cols());
            } else if (transA && !transB) {
                Matrix::transpose_multiply_to_span(c_sub, M, N,
                    a_sub, a_rows_per, a.cols(),
                    b_sub, b_rows_per, b.cols());
            } else {
                // 双转置：先转置 A 子块，再做 A^T × B^T
                // 等价于 (B × A)^T，用 span 暂不支持，fallback 到临时矩阵
                Matrix a_view(a_rows_per, a.cols());
                auto dst_a = a_view.span();
                std::copy_n(a_sub.begin(), a_sub.size(), dst_a.begin());
                Matrix a_t = a_view.transpose();
                Matrix b_view(b_rows_per, b.cols());
                auto dst_b = b_view.span();
                std::copy_n(b_sub.begin(), b_sub.size(), dst_b.begin());
                Matrix c_part(M, N);
                a_t.multiply_transposed_to(c_part, b_view);
                std::copy_n(c_part.span().begin(), M * N, c_sub.begin());
            }
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

    // 融合 axpy：A += scalar * B（单次循环，避免 clone+scale+add 三步）
    // 使用 nn::transform 二元并行版本，与 add_inplace/scale_inplace 一致：
    //   - n >= PARALLEL_THRESHOLD 时自动并行，否则串行
    //   - 单次循环 + 内联 lambda，零开销抽象
    [[nodiscard]] Result<void> axpy_inplace(Tensor& A, Scalar scalar, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"axpy_inplace: shape mismatch"});
        auto& a = A.cpu_matrix();
        const auto& b = B.cpu_matrix();
        auto a_span = a.span();
        const auto b_span = b.span();
        nn::transform(a_span.begin(), a_span.end(), b_span.begin(),
                       a_span.begin(),
                       [scalar](Scalar x, Scalar y) noexcept { return x + scalar * y; });
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
