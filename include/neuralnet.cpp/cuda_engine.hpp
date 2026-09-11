// ── cuda_engine.hpp — CUDA 计算引擎实现 ────────────────────────────────────
// CudaEngine 封装 CudaBackend，实现 ComputeEngine 接口。
// 与 GpuEngine（Vulkan）架构对称，所有原语在 GPU 上原生执行。
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_CUDA_ENGINE_HPP
#define NN_CUDA_ENGINE_HPP

#ifdef NN_HAS_CUDA

#include "compute_engine.hpp"
#include "backend/cuda_backend.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// CudaEngine — CUDA 计算引擎
// ══════════════════════════════════════════════════════════════════════════

class CudaEngine final : public ComputeEngine
{
private:
    CudaBackend& backend_;

public:
    explicit CudaEngine(CudaBackend& backend) : backend_(backend) {}

    [[nodiscard]] Device device() const noexcept override { return Device::GPU; }

    // ── 批处理：no-op（当前使用同步默认流）──────────────────────────────
    [[nodiscard]] Result<void> begin_batch() override { return backend_.begin_batch(); }
    [[nodiscard]] Result<void> end_batch() override { return backend_.end_batch(); }
    [[nodiscard]] Result<void> flush_batch() override { return {}; }

    // ── 表达式录制（M2 框架）：当前为 no-op（各原语独立 dispatch） ──────
    [[nodiscard]] Result<void> begin_expr() override { return {}; }
    [[nodiscard]] Result<void> end_expr() override { return {}; }

    // ══════════════════════════════════════════════════════════════════════
    // 张量工厂
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols) override
    {
        auto r = CudaTensor::create_empty(rows, cols, backend_);
        if (!r) return Tensor();
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m) override
    {
        auto r = CudaTensor::from_matrix(m, backend_);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Matrix> to_matrix(const Tensor& t) override
    {
        if (t.is_cpu())
            return Matrix(t.cpu_matrix());
        return t.cuda_tensor().to_matrix(backend_);
    }

    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"copy_from: shape mismatch"});
        if (dst.is_cpu())
        {
            dst = Tensor::from_matrix(Matrix(src));
            return {};
        }
        return backend_.upload_blocking(dst.cuda_tensor(), src.span());
    }

    [[nodiscard]] Result<Tensor> clone(const Tensor& src) override
    {
        if (src.is_cpu())
            return Tensor::from_matrix(Matrix(src.cpu_matrix()));
        auto r = backend_.clone_gpu(src.cuda_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ── 行切片 ────────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count) override
    {
        auto src_gpu = ensure_cuda(src);
        if (!src_gpu) return std::unexpected(src_gpu.error());
        auto r = backend_.slice_rows_gpu(src_gpu->cuda_tensor(), start_row, count);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ── 行插入 ────────────────────────────────────────────────────────────
    // 纯 CUDA 架构：dst 必须为 CUDA Tensor，不接受 CPU 目标（不做 CPU 回退，
    // 保持"硬报错、不降级"）。CPU 目标说明调用方混用了设备，属调用错误。
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (dst.is_cpu())
            return std::unexpected(Error{
                "CudaEngine::insert_rows: dst must be a CUDA tensor"
                "（不做 CPU 回退；请保证目标张量在 CUDA 上）"});
        auto src_gpu = ensure_cuda(src);
        if (!src_gpu) return std::unexpected(src_gpu.error());
        return backend_.insert_rows_gpu(dst.cuda_tensor(), dst_start_row, src_gpu->cuda_tensor());
    }

    // ── Gather ────────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices) override
    {
        auto tbl_gpu = ensure_cuda(table);
        if (!tbl_gpu) return std::unexpected(tbl_gpu.error());
        auto idx_gpu = ensure_cuda(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());
        auto r = backend_.gather_gpu(tbl_gpu->cuda_tensor(), idx_gpu->cuda_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ── Scatter Add ───────────────────────────────────────────────────────
    [[nodiscard]] Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad) override
    {
        if (!dst.is_gpu())
            return std::unexpected(Error{"scatter_add_rows: dst must be GPU tensor"});
        auto idx_gpu = ensure_cuda(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());
        auto grad_gpu = ensure_cuda(grad);
        if (!grad_gpu) return std::unexpected(grad_gpu.error());
        return backend_.scatter_add_gpu(
            dst.cuda_tensor(), idx_gpu->cuda_tensor(), grad_gpu->cuda_tensor());
    }

private:

public:

    // ── 3D 重排 ───────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse) override
    {
        auto x_gpu = ensure_cuda(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        auto r = backend_.rearrange_3d_gpu(
            x_gpu->cuda_tensor(),
            static_cast<uint32_t>(M), static_cast<uint32_t>(B),
            static_cast<uint32_t>(N), inverse ? 1u : 0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ── 转置 ──────────────────────────────────────────────────────────────
    [[nodiscard]] Result<Tensor> transpose(const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.transpose_gpu(a_gpu->cuda_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_cuda(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto r = backend_.matmul_gpu(
            a_gpu->cuda_tensor(), b_gpu->cuda_tensor(),
            transA ? 1u : 0u, transB ? 1u : 0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_cuda(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto r = backend_.batched_matmul_gpu(
            a_gpu->cuda_tensor(), b_gpu->cuda_tensor(),
            static_cast<uint32_t>(batch),
            transA ? 1u : 0u, transB ? 1u : 0u);
        if (!r) return std::unexpected(r.error());
        Tensor out = Tensor::from_cuda(std::move(*r));
        // CUDA kernel 暂无 alpha 参数：alpha != 1 时原地后缩放
        if (alpha != Scalar{1})
        {
            auto s = scale_inplace(out, alpha);
            if (!s) return std::unexpected(s.error());
        }
        return out;
    }

    // ── matmul 融合原语（M4）：CUDA 尚未实现，返回错误（调用方回退 CPU/组合路径） ──
    [[nodiscard]] Result<Tensor> batched_matmul_reduce(
        const Tensor&, const Tensor&, std::size_t,
        ReduceOp, bool, bool, Scalar, bool, const AttnBias&) override
    {
        return std::unexpected(Error{"CudaEngine: batched_matmul_reduce 未实现"});
    }
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_denom(
        const Tensor&, const Tensor&, const Tensor&,
        std::size_t, bool, bool, Scalar, const AttnBias&) override
    {
        return std::unexpected(Error{"CudaEngine: batched_matmul_softmax_denom 未实现"});
    }
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_apply(
        const Tensor&, const Tensor&, const Tensor&,
        const Tensor&, const Tensor&,
        std::size_t, bool, bool, Scalar, const AttnBias&) override
    {
        return std::unexpected(Error{"CudaEngine: batched_matmul_softmax_apply 未实现"});
    }

    // ── 两趟式注意力反向原语（M6）：CUDA 尚未实现，返回错误（调用方回退） ──
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_backward_q(
        const Tensor&, const Tensor&, const Tensor&,
        const Tensor&, const Tensor&,
        std::size_t, bool, bool, Scalar, Tensor&, const AttnBias&) override
    {
        return std::unexpected(Error{"CudaEngine: batched_matmul_softmax_backward_q 未实现"});
    }
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_backward_kv(
        const Tensor&, const Tensor&, const Tensor&,
        const Tensor&, const Tensor&,
        const Tensor&, const Tensor&,
        std::size_t, bool, bool, Scalar, Tensor&, const AttnBias&) override
    {
        return std::unexpected(Error{"CudaEngine: batched_matmul_softmax_backward_kv 未实现"});
    }

    // ── 列式 softmax 融合原语（M5）：CUDA 尚未实现，返回错误（调用方回退） ──
    [[nodiscard]] Result<Tensor> col_softmax_denom(
        const Tensor&, const Tensor&) override
    {
        return std::unexpected(Error{"CudaEngine: col_softmax_denom 未实现"});
    }
    [[nodiscard]] Result<Tensor> col_softmax_sparse_forward(
        const Tensor&, const Tensor&, const Tensor*,
        std::size_t, Scalar, Tensor&) override
    {
        return std::unexpected(Error{"CudaEngine: col_softmax_sparse_forward 未实现"});
    }

    // A += B
    [[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"add_inplace: shape mismatch"});
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_cuda(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto r = backend_.elementwise_binary_gpu(a_gpu->cuda_tensor(), b_gpu->cuda_tensor(), 0);
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_cuda(std::move(*r));
        return {};
    }

    // A *= scalar
    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.elementwise_binary_scalar_gpu(
            a_gpu->cuda_tensor(), 2, static_cast<float>(s), false);
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_cuda(std::move(*r));
        return {};
    }

    // A += scalar * B（融合 axpy）
    [[nodiscard]] Result<void> axpy_inplace(Tensor& A, Scalar scalar, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"axpy_inplace: shape mismatch"});
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_cuda(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto r = backend_.axpy_gpu(a_gpu->cuda_tensor(), b_gpu->cuda_tensor(), static_cast<float>(scalar));
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_cuda(std::move(*r));
        return {};
    }

    // A = 0
    [[nodiscard]] Result<void> zero(Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.fill_zero_gpu(a_gpu->cuda_tensor());
        if (!r) return std::unexpected(r.error());
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> row_reduce_sum(const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->cuda_tensor(), 0u, 0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum(const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->cuda_tensor(), 1u, 0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> row_reduce_max(const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->cuda_tensor(), 0u, 1u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_max(const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->cuda_tensor(), 1u, 1u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 广播原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto rv_gpu = ensure_cuda(row_vec);
        if (!rv_gpu) return std::unexpected(rv_gpu.error());
        auto r = backend_.broadcast_gpu(
            a_gpu->cuda_tensor(), rv_gpu->cuda_tensor(),
            0u, static_cast<uint32_t>(op));
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_cuda(std::move(*r));
        return {};
    }

    [[nodiscard]] Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto cv_gpu = ensure_cuda(col_vec);
        if (!cv_gpu) return std::unexpected(cv_gpu.error());
        auto r = backend_.broadcast_gpu(
            a_gpu->cuda_tensor(), cv_gpu->cuda_tensor(),
            1u, static_cast<uint32_t>(op));
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_cuda(std::move(*r));
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 逐元素原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> elementwise_unary(UnaryOp op, const Tensor& A) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.elementwise_unary_gpu(a_gpu->cuda_tensor(), static_cast<uint32_t>(op));
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> elementwise_binary(
        BinaryOp op, const Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"elementwise_binary: shape mismatch"});
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_cuda(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto r = backend_.elementwise_binary_gpu(
            a_gpu->cuda_tensor(), b_gpu->cuda_tensor(), static_cast<uint32_t>(op));
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> elementwise_binary_scalar(
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first) override
    {
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.elementwise_binary_scalar_gpu(
            a_gpu->cuda_tensor(), static_cast<uint32_t>(op),
            static_cast<float>(s), scalar_first);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 条件选择原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> elementwise_select_scalar_cond(
        CompareOp cmp, const Tensor& A, Scalar scalar_b,
        const Tensor& then_t, Scalar scalar_else) override
    {
        if (A.rows() != then_t.rows() || A.cols() != then_t.cols())
            return std::unexpected(Error{"select: A and then shape mismatch"});
        auto a_gpu = ensure_cuda(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto t_gpu = ensure_cuda(then_t);
        if (!t_gpu) return std::unexpected(t_gpu.error());
        auto r = backend_.elementwise_select_scalar_cond_gpu(
            a_gpu->cuda_tensor(), t_gpu->cuda_tensor(),
            static_cast<uint32_t>(cmp),
            static_cast<float>(scalar_b), static_cast<float>(scalar_else));
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（闭合世界 AOT）
    //
    // CUDA 目前没有 AOT 融合 shader（闭合世界约束下无 eager、无运行时生成），
    // 因此任何表达式都直接报错。如需 GPU 融合请用 Vulkan 后端，或为 CUDA
    // 走与 Vulkan 相同的构建期 scan_exprs + gen_fused 合成流水线。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        (void)spec; (void)inputs; (void)rows; (void)cols;
        return std::unexpected(Error{
            "CudaEngine::eval_expr: 闭合世界无 CUDA AOT 融合 shader；请用 Vulkan"});
    }

private:
    // ── 辅助：确保 Tensor 在 CUDA 上 ─────────────────────────────────────
    [[nodiscard]] Result<Tensor> ensure_cuda(const Tensor& t)
    {
        if (t.is_gpu())
            return t;
        if (!t.valid())
            return std::unexpected(Error{"ensure_cuda: invalid tensor"});
        auto r = CudaTensor::from_matrix(t.cpu_matrix(), backend_);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_cuda(std::move(*r));
    }
};

} // namespace nn

#endif // NN_HAS_CUDA
#endif // NN_CUDA_ENGINE_HPP
