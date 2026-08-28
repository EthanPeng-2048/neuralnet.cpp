#ifndef NN_COMPUTE_CPU_ENGINE_HPP
#define NN_COMPUTE_CPU_ENGINE_HPP

// ── compute_cpu_engine.hpp — CPU 计算引擎实现 ─────────────────────────────────────
// CpuEngine 封装现有 Matrix 方法和 AST（compute::apply），实现
// ComputeEngine 接口。所有操作同步执行，begin_batch/end_batch 为 no-op。
//
// 逐元素运算通过 switch 分发到编译期 AST，保持零开销抽象。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>

#include "compute_engine.hpp"
#include "algebra_compute.hpp"
#include "algebra_expr.hpp"
#include "expr_opt.hpp"
#include "expr_graph.hpp"    // IR-C：图 IR + 融合分析（scan 模式登记用）
#include "expr_registry.hpp" // scan 模式融合 kernel 登记（NN_EXPR_SCAN）

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// AttnBias 组合偏置求值（CPU 参考实现，与 GPU shader 语义一致）
//
// 返回 {偏置值 mv, 是否屏蔽}。语义见 compute_engine.hpp 的 AttnBias 注释：
//   mv(bb,i,j) = dense(i,j) + (causal && j>i ? -inf) + (doc ? -inf)
//              + (slopes ? -slope[h]*(i-j))
// bb = 两趟式原语的 batch 索引 = b*num_heads + h；i = query 行，j = key 列。
// ══════════════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════════════
// CpuEngine — CPU 计算引擎
// ══════════════════════════════════════════════════════════════════════════
class CpuEngine final : public ComputeEngine
{
private:
    // IR-C 图 IR 录制（仅构建期 scan 模式启用；普通 CPU 运行保持 no-op）。
    // scan 模式下 begin_expr 开启录制，期间 dsl::compute/compute_reduce 把
    // 表达式加入录制图（见 expr_dsl.hpp 的 scan 分支），end_expr 时融合分析
    // 并把每个融合 kernel 的复合 spec 登记进 global_registry —— 保证 GPU
    // 运行时 begin_expr/end_expr 融合出的复合 spec 命中 AOT shader（闭合世界）。
    // D3：录制图由 fused::recording_graph_owner()（thread_local unique_ptr）
    // 持有，不再用引擎成员（消除跨线程共享）。

public:
    [[nodiscard]] Device device() const noexcept override { return Device::CPU; }

    // ── 批处理：CPU 为 no-op（同步执行） ──────────────────────────────────
    [[nodiscard]] Result<void> begin_batch() override { return {}; }
    [[nodiscard]] Result<void> end_batch() override { return {}; }
    [[nodiscard]] Result<void> flush_batch() override { return {}; }

    // ── 表达式录制（IR-C）──────────────────────────────────────────────
    // 普通 CPU 运行：no-op（各表达式直接求值，行为不变——融合是 GPU 优化）。
    // 构建期 scan（NN_EXPR_SCAN）：begin_expr 开启录制图；end_expr 融合分析
    // 并登记全部融合 kernel 的复合 spec（闭合世界两端一致）。
    [[nodiscard]] Result<void> begin_expr() override
    {
#ifdef NN_EXPR_SCAN
        auto& owner = fused::recording_graph_owner();
        if (owner)
            return std::unexpected(Error{"begin_expr: 嵌套录制（已有未结束的 begin_expr）"});
        owner = std::make_unique<ExprGraph>();
#endif
        return {};
    }

    [[nodiscard]] Result<void> end_expr() override
    {
#ifdef NN_EXPR_SCAN
        auto& owner = fused::recording_graph_owner();
        if (!owner)
            return {};
        ExprGraph g = std::move(*owner);
        owner.reset();
        auto kernels = fuse_expr_graph(g);
        for (auto& k : kernels)
            fused::global_registry().add(k.spec);
#endif
        return {};
    }

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
    //   table: (vocab, D), indices: 任意形状，按 flat 遍历所有元素
    //   输出: (num_indices, D)，out[i] = table[flat_indices[i]]
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
        const std::size_t num = idx.size();  // 遍历 indices 所有元素（支持任意形状）

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
    //   dst: (vocab, D) 原地修改, indices: 任意形状（按 flat 遍历）, grad: (num_indices, D)
    //   语义: dst[flat_indices[i]] += grad[i]
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
        const std::size_t num = idx.size();  // 遍历 indices 所有元素（支持任意形状）

        auto dst_span = d.span();
        const auto idx_span = idx.span();
        const auto grad_span = g.span();

        for (std::size_t i = 0; i < num; ++i)
        {
            const auto row_idx = static_cast<std::size_t>(idx_span[i]);
            if (row_idx < vocab)
            {
                auto dst_row = dst_span.subspan(row_idx * D, D);
                const auto grad_row = grad_span.subspan(i * D, D);
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

    // ── 矩阵转置：A (R, C) → out (C, R) ──
    [[nodiscard]] Result<Tensor> transpose(const Tensor& A) override
    {
        if (A.is_gpu())
            return std::unexpected(Error{"CpuEngine: GPU tensor on CPU engine"});
        return Tensor::from_matrix(A.cpu_matrix().transpose());
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
    // C_b = alpha * op(A_b, B_b)（alpha 为 cuBLAS sgemm 语义的输出缩放系数）
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha) override
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

        // 输出缩放：C_b = alpha * op(A_b, B_b)（alpha==1 时零开销跳过）
        if (alpha != Scalar{1})
        {
            auto dst = result.span();
            for (std::size_t i = 0; i < dst.size(); ++i)
                dst[i] *= alpha;
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

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（融合解释器）
    //
    // 对 ExprSpec 一次遍历求值：每个输出元素按指令序列计算，中间结果存
    // 寄存器数组，**不产生任何临时 Tensor**。视图（RotateHalf/RowMod）为
    // 纯索引映射，无物化。性能与等价手写循环一致。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        return eval_expr_impl(spec, inputs, rows, cols, /*vector_out=*/false);
    }

    // ── 归约向量原生形状输出（M3：LayerNorm/RMSNorm 小向量缓存用） ──────
    // 语义：表达式在 (rows,cols) 网格上求值，但输出为归约向量本身：
    //   行归约轴 → (rows,1)，列归约轴 → (1,cols)（而非广播到 (rows,cols)）。
    // 要求：表达式归约轴为 0/1；末指令为归约时直接取归约向量，否则按代表
    //   元素求值（所有 Input 须经归约/广播视图访问，保证沿归约轴恒定）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        return eval_expr_impl(spec, inputs, rows, cols, /*vector_out=*/true);
    }

    // ── eval_expr / eval_expr_reduce 共用实现 ────────────────────────────
    [[nodiscard]] Result<Tensor> eval_expr_impl(
        const ExprSpec& raw_spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, bool vector_out)
    {
        // canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），Layer 无感知；
        // 结构统一在 canonical 形态上，与 scan/gen_fused 两端一致。
        if (auto v = validate_expr_spec(raw_spec, inputs.size()); !v)
            return std::unexpected(v.error());
        const ExprSpec spec = canonicalize_expr_spec(raw_spec);
        if (auto v = validate_expr_spec(spec, inputs.size()); !v)
            return std::unexpected(v.error());
        // 空指令表仅允许"纯 matmul"（输出 = matmul 结果本身）
        if (spec.instrs.empty() && !spec.matmul)
            return std::unexpected(Error{"eval_expr: empty instruction list"});

        // vector_out：要求表达式为纯行/列归约（归约轴 0/1），输出取向量形状
        const int raxis = vector_out ? expr_spec_reduce_axis(spec) : -1;
        if (vector_out && raxis != 0 && raxis != 1)
            return std::unexpected(Error{
                "eval_expr_reduce: 表达式需为行/列归约（归约轴 0/1）"});
        const bool vec_is_row = (raxis == 0);

        std::vector<ConstSpan> spans;
        spans.reserve(inputs.size());
        // matmul 段（S1/S2，S7 加 batch）：A/B 输入按 matmul 形状解释（与
        // 逐元素网格 (rows,cols) 不同），其余输入仍按视图语义校验。
        //   A 存储：transA=0 → (batch*M, K)；transA=1 → (batch*K, M)
        //   B 存储：transB=0 → (batch*K, N)；transB=1 → (batch*N, K)
        //   其中 M = rows/batch（每批输出行），N = cols
        const MatmulSpec* mm = spec.matmul ? &*spec.matmul : nullptr;
        const std::size_t mm_k = mm ? mm->k : 0;
        const std::size_t mm_batch = mm ? mm->batch : 1;
        const std::size_t m_per = (mm && mm->batch > 0 && rows % mm->batch == 0)
            ? rows / mm->batch : rows;
        for (std::size_t k = 0; k < inputs.size(); ++k)
        {
            const Tensor& t = inputs[k];
            if (!t.is_cpu())
                return std::unexpected(Error{"eval_expr: input not CPU"});
            const ExprView& v = spec.views[k];
            if (mm && (k == static_cast<std::size_t>(mm->a_input) ||
                       k == static_cast<std::size_t>(mm->b_input)))
            {
                // matmul 输入：形状按 trans 标志、K 与 batch 推导（M=rows/batch）
                const bool is_a = (k == static_cast<std::size_t>(mm->a_input));
                const bool tr = is_a ? (mm->transA != 0) : (mm->transB != 0);
                const std::size_t er = is_a
                    ? (tr ? mm_batch * mm_k : rows)          // A 存储行数期望
                    : (tr ? mm_batch * cols : mm_batch * mm_k);
                const std::size_t ec = is_a
                    ? (tr ? m_per : mm_k)                    // A 存储列数期望
                    : (tr ? mm_k : cols);
                if (t.rows() != er || t.cols() != ec)
                    return std::unexpected(Error{"eval_expr: matmul input shape mismatch"});
                spans.push_back(t.cpu_matrix().span());
                continue;
            }
            // 列数/行数由视图语义决定：Linear/归约视图与输出一致；RowMod 允许
            // 短表；RotateHalf 允许长表；RowBroadcast 为 (rows,1)、ColBroadcast 为 (1,cols)。
            switch (v.kind)
            {
            default:
            case static_cast<uint8_t>(ExprViewKind::Linear):
            case static_cast<uint8_t>(ExprViewKind::ColReduceSum):
            case static_cast<uint8_t>(ExprViewKind::ColReduceMax):
            case static_cast<uint8_t>(ExprViewKind::RowReduceSum):
            case static_cast<uint8_t>(ExprViewKind::RowReduceMax):
            case static_cast<uint8_t>(ExprViewKind::RowGather):
                // 按标签行收集：主输入行数任意（读取经 label 索引），列数须一致
                if (t.cols() != cols)
                    return std::unexpected(Error{"eval_expr: RowGather input cols mismatch"});
                break;
            case static_cast<uint8_t>(ExprViewKind::RotateHalf):
                if (t.rows() != rows || t.cols() != cols || v.param == 0 ||
                    rows % v.param != 0 || v.param % 2 != 0)
                    return std::unexpected(Error{"eval_expr: RotateHalf shape/param invalid"});
                break;
            case static_cast<uint8_t>(ExprViewKind::RowMod):
                if (t.rows() != v.param || t.cols() != cols || v.param == 0 || rows % v.param != 0)
                    return std::unexpected(Error{"eval_expr: RowMod shape/param invalid"});
                break;
            case static_cast<uint8_t>(ExprViewKind::RowBroadcast):
                // 输入 (rows,1)：每行一个值，按行广播
                if (t.rows() != rows || t.cols() != 1)
                    return std::unexpected(Error{"eval_expr: RowBroadcast input must be (rows,1)"});
                break;
            case static_cast<uint8_t>(ExprViewKind::ColBroadcast):
                // 输入 (1,cols)：每列一个值，按列广播
                if (t.rows() != 1 || t.cols() != cols)
                    return std::unexpected(Error{"eval_expr: ColBroadcast input must be (1,cols)"});
                break;
            case static_cast<uint8_t>(ExprViewKind::BatchMod):
                // 输入 (1,modulo 相关)：按批次取模索引的小向量
                if (t.rows() != 1)
                    return std::unexpected(Error{"eval_expr: BatchMod input must be (1,cols)"});
                break;
            case static_cast<uint8_t>(ExprViewKind::BatchCol):
                // 输入 (1, batch*param)：按 (batch, col) 切片（doc_ids 等）
                if (t.rows() != 1 || t.cols() != mm_batch * v.param)
                    return std::unexpected(Error{"eval_expr: BatchCol input must be (1,batch*param)"});
                break;
            }
            spans.push_back(t.cpu_matrix().span());
        }

        Matrix result(rows, cols);
        Span out = result.span();
        const std::size_t n = out.size();
        if (n == 0)
            return Tensor::from_matrix(std::move(result));

        // ── 归约视图预计算：每行/每列一个标量，供广播读取 ──────────────
        // 仅在 views[k] 为归约视图时填充 view_reduce[k]（长度 rows 或 cols）。
        std::vector<std::vector<Scalar>> view_reduce(inputs.size());
        for (std::size_t k = 0; k < inputs.size(); ++k)
        {
            const ExprView& v = spec.views[k];
            if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
                continue;
            const bool is_row = expr_view_reduces_rows(static_cast<ExprViewKind>(v.kind));
            const bool is_max =
                (v.kind == static_cast<uint8_t>(ExprViewKind::RowReduceMax) ||
                 v.kind == static_cast<uint8_t>(ExprViewKind::ColReduceMax));
            const ConstSpan& s = spans[k];
            const std::size_t len = is_row ? rows : cols;
            std::vector<Scalar>& acc = view_reduce[k];
            acc.assign(len, is_max ? std::numeric_limits<Scalar>::lowest() : Scalar{0});
            if (is_row)
            {
                for (std::size_t r = 0; r < rows; ++r)
                    for (std::size_t c = 0; c < cols; ++c)
                        acc[r] = is_max ? std::max(acc[r], s[r * cols + c])
                                        : acc[r] + s[r * cols + c];
            }
            else
            {
                for (std::size_t c = 0; c < cols; ++c)
                    for (std::size_t r = 0; r < rows; ++r)
                        acc[c] = is_max ? std::max(acc[c], s[r * cols + c])
                                        : acc[c] + s[r * cols + c];
            }
        }

        // ── matmul 段预计算（S2 + S7 batch）：C = op(A,B) → (batch*M, N) ──
        // matmul 输出作为"虚拟寄存器 0"，逐元素链经 Matmul 操作数按 (r,c) 读取。
        // 与 batched_matmul 原语语义一致（A/B 按 batch 垂直切分，转置标志按
        // MatmulSpec 解释），再执行逐元素链（同现有归约预计算模式）。
        // 实现：复用 Matrix 分块 matmul span 内核（分块 + 并行 + 向量化），替代
        // 原三层标量循环；DSL 实际只产生 (0,0)/(1,0) 组合（MLP / 注意力 QK^T），
        // (0,1) 免费覆盖，(1,1) 无调用点，保留标量兜底保证语义一致。
        Matrix matmul_out;
        if (mm)
        {
            const std::size_t mm_batch = mm->batch;
            const bool trA = (mm->transA != 0);
            const bool trB = (mm->transB != 0);
            const std::size_t M = m_per;          // 每批输出行
            const std::size_t N = cols;
            const std::size_t K = mm_k;
            const ConstSpan& as = spans[mm->a_input];
            const ConstSpan& bs = spans[mm->b_input];
            matmul_out = Matrix(rows, N);   // 零初始化：span 内核为累加语义（r += A*B）
            auto out = matmul_out.span();
            for (std::size_t b = 0; b < mm_batch; ++b)
            {
                // 每批切片：A 存储 (M,K) 或 (K,M)，元素数恒为 M*K；B 同理恒为 K*N。
                // ConstSpan/Span 是 AST 视图（无到 std::span 的隐式转换），
                // 与 batched_matmul 一致，用 data()+偏移显式构造 std::span 子区间。
                const std::size_t a_off = b * M * K;
                const std::size_t b_off = b * K * N;
                const auto a_sub = std::span<const Scalar>(as.data() + a_off, M * K);
                const auto b_sub = std::span<const Scalar>(bs.data() + b_off, K * N);
                auto c_sub = std::span<Scalar>(out.data() + b * M * N, M * N);
                if (!trA && !trB)
                {
                    Matrix::multiply_to_span(c_sub, M, N,
                        a_sub, M, K,
                        b_sub, K, N);
                }
                else if (!trA && trB)
                {
                    Matrix::multiply_transposed_to_span(c_sub, M, N,
                        a_sub, M, K,
                        b_sub, N, K);
                }
                else if (trA && !trB)
                {
                    Matrix::transpose_multiply_to_span(c_sub, M, N,
                        a_sub, K, M,
                        b_sub, K, N);
                }
                else
                {
                    // 双转置（DSL 无此组合）：保留原标量循环兜底
                    const std::size_t a_base = b * K * M;
                    const std::size_t b_base = b * N * K;
                    for (std::size_t i = 0; i < M; ++i)
                        for (std::size_t j = 0; j < N; ++j)
                        {
                            Scalar s = Scalar{0};
                            for (std::size_t kk = 0; kk < K; ++kk)
                                s += as[a_base + kk * M + i] * bs[b_base + j * K + kk];
                            out[(b * M + i) * N + j] = s;
                        }
                }
            }
        }

        // ── 纯 matmul（无逐元素链）：输出 = matmul 结果本身 ─────────────
        if (spec.instrs.empty())
        {
            if (!mm)
                return std::unexpected(Error{"eval_expr: empty instruction list without matmul"});
            Matrix r(rows, cols);
            Span o = r.span();
            const auto ms = matmul_out.span();
            for (std::size_t i = 0; i < n; ++i)
                o[i] = ms[i];
            return Tensor::from_matrix(std::move(r));
        }

        // 视图求值：按 (row, col) 映射到输入 span（归约视图读取预计算标量向量）
        auto read_input = [&](std::size_t k, std::size_t r, std::size_t c) -> Scalar
        {
            const ExprView& v = spec.views[k];
            const ConstSpan& s = spans[k];
            const std::size_t b_idx = (mm && m_per > 0) ? r / m_per : 0;
            switch (v.kind)
            {
            default:
            case static_cast<uint8_t>(ExprViewKind::Linear):
                return s[r * cols + c];
            case static_cast<uint8_t>(ExprViewKind::RotateHalf):
            {
                const std::size_t block = v.param;
                const std::size_t rl = r % block;
                const std::size_t rr = (r / block) * block
                    + ((rl < block / 2) ? (rl + block / 2) : (rl - block / 2));
                Scalar val = s[rr * cols + c];
                if (v.negate_first_half && rl < block / 2)
                    return -val;
                return val;
            }
            case static_cast<uint8_t>(ExprViewKind::RowMod):
                return s[(r % v.param) * cols + c];
            case static_cast<uint8_t>(ExprViewKind::RowReduceSum):
            case static_cast<uint8_t>(ExprViewKind::RowReduceMax):
                return view_reduce[k][r];
            case static_cast<uint8_t>(ExprViewKind::ColReduceSum):
            case static_cast<uint8_t>(ExprViewKind::ColReduceMax):
                return view_reduce[k][c];
            case static_cast<uint8_t>(ExprViewKind::RowBroadcast):
                return s[r];           // 输入 (rows,1)
            case static_cast<uint8_t>(ExprViewKind::ColBroadcast):
                return s[c];           // 输入 (1,cols)
            case static_cast<uint8_t>(ExprViewKind::RowGather):
            {
                // 按标签行收集：data[label[c] * cols + c]（标签槽 = v.param）；
                // 越界守卫按主输入行数（s.size()/cols），label 越界返回 0
                const ConstSpan& ls = spans[v.param];
                const std::size_t lbl = (ls[c] >= Scalar{0})
                    ? static_cast<std::size_t>(ls[c]) : 0;
                return (lbl < (s.size() / cols)) ? s[lbl * cols + c] : Scalar{0};
            }
            case static_cast<uint8_t>(ExprViewKind::BatchMod):
                return s[b_idx % v.param];  // 按批次取模索引（ALiBi 斜率等）
            case static_cast<uint8_t>(ExprViewKind::BatchCol):
                return s[b_idx * v.param + c];  // 按 (batch, col) 切片（doc_ids 等）
            }
        };

        // ── 归约指令预计算 ──────────────────────────────────────────────
        // 归约指令 dst 是隐式"每行/每列一个标量"的向量，存于 reduce_vec[dst]；
        // reduce_axis[dst]：1=按列归约 (1,cols)，0=按行归约 (rows,1)。
        // 按指令序处理：处理第 ri 条归约指令时，其源操作数（含转递依赖）引用的
        // 更早归约指令向量已完整——指令序保证拓扑依赖成立。
        std::vector<std::vector<Scalar>> reduce_vec(EXPR_MAX_REGS);
        std::vector<std::uint8_t> reduce_axis(EXPR_MAX_REGS, 0);

        for (std::size_t ri = 0; ri < spec.instrs.size(); ++ri)
        {
            const ExprInstr& R = spec.instrs[ri];
            const ExprOp rop = static_cast<ExprOp>(R.op);
            if (!expr_op_is_reduce(rop))
                continue;

            const bool is_col = expr_op_reduces_cols(rop);
            const bool is_max = (rop == ExprOp::ColMax || rop == ExprOp::RowMax);
            reduce_axis[R.dst] = is_col ? std::uint8_t{1} : std::uint8_t{0};

            std::vector<Scalar>& acc = reduce_vec[R.dst];
            acc.assign(is_col ? cols : rows,
                       is_max ? std::numeric_limits<Scalar>::lowest() : Scalar{0});

            // 逐元素重放 [0, ri) 前缀（跳过归约指令，Reduce 操作数读已完整向量），
            // 求 R.a 的值并累加——归约指令的源允许引用更早归约结果。
            for (std::size_t i = 0; i < n; ++i)
            {
                const std::size_t r = i / cols;
                const std::size_t c = i % cols;

                Scalar regs[EXPR_MAX_REGS] = {};
                const auto eval_op = [&](const ExprOperand& op) -> Scalar
                {
                    switch (op.kind)
                    {
                    default:
                    case static_cast<uint8_t>(ExprOperandKind::Reg):
                    case static_cast<uint8_t>(ExprOperandKind::Fanout): return regs[op.idx];
                    case static_cast<uint8_t>(ExprOperandKind::Const): return spec.consts[op.idx];
                    case static_cast<uint8_t>(ExprOperandKind::Input): return read_input(op.idx, r, c);
                    case static_cast<uint8_t>(ExprOperandKind::Reduce):
                        return reduce_vec[op.idx][reduce_axis[op.idx] ? c : r];
                    case static_cast<uint8_t>(ExprOperandKind::Matmul):
                        return matmul_out.span()[r * cols + c];  // matmul 段输出（虚拟寄存器 0）
                    case static_cast<uint8_t>(ExprOperandKind::Row):
                        return static_cast<Scalar>(r % m_per);   // batch 内行号
                    case static_cast<uint8_t>(ExprOperandKind::Col):
                        return static_cast<Scalar>(c);
                    case static_cast<uint8_t>(ExprOperandKind::Batch):
                        return static_cast<Scalar>((mm && m_per > 0) ? r / m_per : 0);
                    }
                };

                for (std::size_t j = 0; j < ri; ++j)
                {
                    const ExprInstr& J = spec.instrs[j];
                    if (expr_op_is_reduce(static_cast<ExprOp>(J.op)))
                        continue;  // 归约指令已在上方处理中完整计算
                    const Scalar va = eval_op(J.a);
                    const ExprOp op = static_cast<ExprOp>(J.op);
                    switch (op)
                    {
                    // 一元
                    case ExprOp::Neg:   regs[J.dst] = -va; break;
                    case ExprOp::Exp:   regs[J.dst] = std::exp(va); break;
                    case ExprOp::Log:   regs[J.dst] = std::log(va); break;
                    case ExprOp::Sqrt:  regs[J.dst] = std::sqrt(va); break;
                    case ExprOp::Rsqrt: regs[J.dst] = Scalar{1} / std::sqrt(va); break;
                    case ExprOp::Abs:   regs[J.dst] = std::abs(va); break;
                    case ExprOp::Tanh:  regs[J.dst] = std::tanh(va); break;
                    // 二元
                    case ExprOp::Add:   regs[J.dst] = va + eval_op(J.b); break;
                    case ExprOp::Sub:   regs[J.dst] = va - eval_op(J.b); break;
                    case ExprOp::Mul:   regs[J.dst] = va * eval_op(J.b); break;
                    case ExprOp::Div:   regs[J.dst] = va / eval_op(J.b); break;
                    case ExprOp::Max:   regs[J.dst] = std::max(va, eval_op(J.b)); break;
                    case ExprOp::Min:   regs[J.dst] = std::min(va, eval_op(J.b)); break;
                    // 比较（输出 1.0 / 0.0）
                    case ExprOp::Lt:    regs[J.dst] = (va <  eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    case ExprOp::Le:    regs[J.dst] = (va <= eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    case ExprOp::Gt:    regs[J.dst] = (va >  eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    case ExprOp::Ge:    regs[J.dst] = (va >= eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    case ExprOp::Eq:    regs[J.dst] = (va == eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    case ExprOp::Ne:    regs[J.dst] = (va != eval_op(J.b)) ? Scalar{1} : Scalar{0}; break;
                    // 选择
                    case ExprOp::Select: regs[J.dst] = (va != Scalar{0}) ? eval_op(J.b) : eval_op(J.c); break;
                    default: break;  // 归约指令已在 continue 中跳过
                    }
                }

                const Scalar v = eval_op(R.a);
                if (is_col)
                    acc[c] = is_max ? std::max(acc[c], v) : acc[c] + v;
                else
                    acc[r] = is_max ? std::max(acc[r], v) : acc[r] + v;
            }
        }

        // ── 输出 ────────────────────────────────────────────────────────
        const ExprInstr& last = spec.instrs.back();
        const bool last_is_reduce = expr_op_is_reduce(static_cast<ExprOp>(last.op));

        // 求单元素 (r,c) 的逐元素链结果（跳过归约指令，归约经 reduce_vec 读取）
        auto eval_element = [&](std::size_t r, std::size_t c) -> Scalar
        {
            Scalar regs[EXPR_MAX_REGS] = {};
            const auto eval_op = [&](const ExprOperand& op) -> Scalar
            {
                switch (op.kind)
                {
                default:
                case static_cast<uint8_t>(ExprOperandKind::Reg):
                case static_cast<uint8_t>(ExprOperandKind::Fanout): return regs[op.idx];
                case static_cast<uint8_t>(ExprOperandKind::Const): return spec.consts[op.idx];
                case static_cast<uint8_t>(ExprOperandKind::Input): return read_input(op.idx, r, c);
                case static_cast<uint8_t>(ExprOperandKind::Reduce):
                    return reduce_vec[op.idx][reduce_axis[op.idx] ? c : r];
                case static_cast<uint8_t>(ExprOperandKind::Matmul):
                    return matmul_out.span()[r * cols + c];  // matmul 段输出（虚拟寄存器 0）
                case static_cast<uint8_t>(ExprOperandKind::Row):
                    return static_cast<Scalar>(r % m_per);   // batch 内行号
                case static_cast<uint8_t>(ExprOperandKind::Col):
                    return static_cast<Scalar>(c);
                case static_cast<uint8_t>(ExprOperandKind::Batch):
                    return static_cast<Scalar>((mm && m_per > 0) ? r / m_per : 0);
                }
            };
            for (const auto& ins : spec.instrs)
            {
                if (expr_op_is_reduce(static_cast<ExprOp>(ins.op)))
                    continue;  // 归约指令已预计算（隐式向量，不经寄存器）
                const Scalar va = eval_op(ins.a);
                const ExprOp op = static_cast<ExprOp>(ins.op);
                switch (op)
                {
                // 一元
                case ExprOp::Neg:   regs[ins.dst] = -va; break;
                case ExprOp::Exp:   regs[ins.dst] = std::exp(va); break;
                case ExprOp::Log:   regs[ins.dst] = std::log(va); break;
                case ExprOp::Sqrt:  regs[ins.dst] = std::sqrt(va); break;
                case ExprOp::Rsqrt: regs[ins.dst] = Scalar{1} / std::sqrt(va); break;
                case ExprOp::Abs:   regs[ins.dst] = std::abs(va); break;
                case ExprOp::Tanh:  regs[ins.dst] = std::tanh(va); break;
                // 二元
                case ExprOp::Add:   regs[ins.dst] = va + eval_op(ins.b); break;
                case ExprOp::Sub:   regs[ins.dst] = va - eval_op(ins.b); break;
                case ExprOp::Mul:   regs[ins.dst] = va * eval_op(ins.b); break;
                case ExprOp::Div:   regs[ins.dst] = va / eval_op(ins.b); break;
                case ExprOp::Max:   regs[ins.dst] = std::max(va, eval_op(ins.b)); break;
                case ExprOp::Min:   regs[ins.dst] = std::min(va, eval_op(ins.b)); break;
                // 比较（输出 1.0 / 0.0）
                case ExprOp::Lt:    regs[ins.dst] = (va <  eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                case ExprOp::Le:    regs[ins.dst] = (va <= eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                case ExprOp::Gt:    regs[ins.dst] = (va >  eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                case ExprOp::Ge:    regs[ins.dst] = (va >= eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                case ExprOp::Eq:    regs[ins.dst] = (va == eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                case ExprOp::Ne:    regs[ins.dst] = (va != eval_op(ins.b)) ? Scalar{1} : Scalar{0}; break;
                // 选择
                case ExprOp::Select: regs[ins.dst] = (va != Scalar{0}) ? eval_op(ins.b) : eval_op(ins.c); break;
                default: break;  // 归约指令已在上方 continue 跳过
                }
            }
            return regs[last.dst];
        };

        if (vector_out)
        {
            // 归约向量原生形状输出：(rows,1)（行归约轴）或 (1,cols)（列归约轴）
            const std::size_t len = vec_is_row ? rows : cols;
            Matrix result(vec_is_row ? rows : 1, vec_is_row ? 1 : cols);
            Span o = result.span();
            if (last_is_reduce)
            {
                // 末指令为归约 → 直接取归约向量
                for (std::size_t k = 0; k < len; ++k)
                    o[k] = reduce_vec[last.dst][k];
                return Tensor::from_matrix(std::move(result));
            }
            // 否则按代表元素求值：行归约 → 每行 (k, 0)；列归约 → 每列 (0, k)。
            // 前置校验（反向数据流）：从末指令收集"影响输出"的非归约指令链，
            // 其 Input 操作数必须经归约/广播视图访问（保证沿归约轴恒定）。
            // 归约指令的源在全网格求值（可自由读 Linear），不参与本分析。
            {
                std::vector<uint8_t> needed(EXPR_MAX_REGS, 0);
                needed[last.dst] = 1;
                for (std::size_t ii = spec.instrs.size(); ii-- > 0;)
                {
                    const ExprInstr& ins = spec.instrs[ii];
                    if (!needed[ins.dst])
                        continue;
                    if (expr_op_is_reduce(static_cast<ExprOp>(ins.op)))
                        continue;  // 归约指令：源全网格求值，dst 已标记
                    // 只遍历该算子实际使用的操作数（c 默认 {0,0} 会被误当作 Reg(0)）
                    const std::size_t nops =
                        expr_instr_num_operands(static_cast<ExprOp>(ins.op));
                    const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
                    for (std::size_t oi = 0; oi < nops; ++oi)
                    {
                        const ExprOperand& op = *ops[oi];
                        if (op.kind == static_cast<uint8_t>(ExprOperandKind::Input))
                        {
                            const ExprViewKind vk =
                                static_cast<ExprViewKind>(spec.views[op.idx].kind);
                            if (!expr_view_is_reduce(vk) && !expr_view_is_broadcast(vk))
                                return std::unexpected(Error{
                                    "eval_expr_reduce: 输出链经 Linear/视图直接访问输入，"
                                    "输出不沿归约轴恒定"});
                        }
                        else if (op.kind == static_cast<uint8_t>(ExprOperandKind::Matmul))
                        {
                            // matmul 输出沿两个轴都变化，不能作为归约向量输出
                            return std::unexpected(Error{
                                "eval_expr_reduce: 输出链经 Matmul 访问，输出不沿归约轴恒定"});
                        }
                        else if (op.kind == static_cast<uint8_t>(ExprOperandKind::Row) ||
                                 op.kind == static_cast<uint8_t>(ExprOperandKind::Col) ||
                                 op.kind == static_cast<uint8_t>(ExprOperandKind::Batch))
                        {
                            // 索引操作数沿归约轴的恒定性由生成器/引擎按轴判定；
                            // 这里保守拒绝（S7 实际表达式不触发此路径）
                            return std::unexpected(Error{
                                "eval_expr_reduce: 输出链经 Row/Col/Batch 索引，输出不沿归约轴恒定"});
                        }
                        else if (op.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                                 op.kind == static_cast<uint8_t>(ExprOperandKind::Fanout) ||
                                 op.kind == static_cast<uint8_t>(ExprOperandKind::Reduce))
                        {
                            needed[op.idx] = 1;
                        }
                    }
                }
            }
            for (std::size_t k = 0; k < len; ++k)
            {
                const std::size_t r = vec_is_row ? k : 0;
                const std::size_t c = vec_is_row ? 0 : k;
                o[k] = eval_element(r, c);
            }
            return Tensor::from_matrix(std::move(result));
        }

        // ── 广播输出（常规 eval_expr）：每元素求值，输出 (rows, cols) ──
        for (std::size_t i = 0; i < n; ++i)
        {
            const std::size_t r = i / cols;
            const std::size_t c = i % cols;
            if (last_is_reduce)
            {
                // 输出本身就是归约向量 → 广播到 (rows, cols)
                out[i] = reduce_vec[last.dst][reduce_axis[last.dst] ? c : r];
                continue;
            }
            out[i] = eval_element(r, c);
        }

        return Tensor::from_matrix(std::move(result));
    }

};

} // namespace nn

#endif // NN_COMPUTE_CPU_ENGINE_HPP
