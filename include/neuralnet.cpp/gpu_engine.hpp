#ifndef NN_GPU_ENGINE_HPP
#define NN_GPU_ENGINE_HPP

// ── gpu_engine.hpp — GPU 计算引擎实现（纯 GPU 架构）─────────────────────────
// GpuEngine 封装 GpuBackend，实现 ComputeEngine 接口。
//
// 纯 GPU 架构策略：
//   - 所有原语（matmul、elementwise、reduce、broadcast、zero、clone）均在
//     GPU 上原生执行，数据全程驻留 GPU 显存。
//   - from_matrix：上传 CPU Matrix → GPU Tensor（唯一的 PCIe 上传点）
//   - to_matrix：下载 GPU Tensor → CPU Matrix（唯一的 PCIe 下载点）
//   - create_tensor：分配 GPU buffer（用于参数/梯度）
//   - clone：GPU 内 buffer 拷贝（无 PCIe 传输）
//   - 所有原语返回 GPU Tensor，不回退到 CPU
//
//   前向/反向链路全程 GPU：
//     from_matrix(上传) → matmul(GPU) → broadcast(GPU) → elementwise(GPU)
//     → reduce(GPU) → ... → to_matrix(下载)
//   除 batch 边界外，无 PCIe 传输。
//
// 同步模型：
//   当前不使用 batch 录制模式（每次操作同步完成），以保证正确性。
//   begin_batch/end_batch 为 no-op。
//
// 原地操作语义：
//   add_inplace / scale_inplace / broadcast_*_inplace 采用"分配新 buffer +
//   替换 Tensor"策略（copy-on-write 语义）。zero 使用 vkCmdFillBuffer 真原地
//   清零（避免每步分配）。
// ─────────────────────────────────────────────────────────────────────────

#ifdef NN_HAS_VULKAN

#include "compute_engine.hpp"
#include "backend/vk_backend.hpp"

namespace nn
{

// ══════════════════════════════════════════════════════════════════════════
// GpuEngine — GPU 计算引擎（纯 GPU 架构）
// ══════════════════════════════════════════════════════════════════════════
class GpuEngine final : public ComputeEngine
{
private:
    GpuBackend& backend_;

public:
    explicit GpuEngine(GpuBackend& backend) : backend_(backend) {}

    [[nodiscard]] Device device() const noexcept override { return Device::GPU; }

    // ── 批处理：激活 GpuBackend 的 command buffer 录制模式 ──────────────
    // begin_batch 后，所有原语录制到共享 batch_cmd_，直到 end_batch 一次提交+等待。
    // 这消除了 per-primitive 的 vkQueueSubmit+vkWaitForFences 开销。
    [[nodiscard]] Result<void> begin_batch() override
    {
        return backend_.begin_batch();
    }

    [[nodiscard]] Result<void> end_batch() override
    {
        return backend_.end_batch();
    }

    // ── 中点刷新：提交当前 command buffer 并开始新的录制 ──
    // 用于拆分大 batch（如 forward 与 backward 之间），防 TDR。
    [[nodiscard]] Result<void> flush_batch() override
    {
        return backend_.flush_batch();
    }

    // batch 模式查询（内部使用，to_matrix/from_matrix 需检查）
    [[nodiscard]] bool in_batch() const noexcept { return backend_.in_batch(); }

    // ══════════════════════════════════════════════════════════════════════
    // 张量工厂（纯 GPU：全部创建/上传为 GPU Tensor）
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols) override
    {
        // 分配 GPU buffer（用于参数/梯度，后续通过 copy_from 填充）
        auto r = GpuTensor::create_empty(rows, cols, backend_);
        if (!r)
        {
            // 分配失败时返回空 Tensor（调用方应检查 valid()）
            return Tensor();
        }
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m) override
    {
        // 上传 CPU Matrix → GPU Tensor（PCIe 上传，唯一的上传点）
        auto r = GpuTensor::from_matrix(m, backend_);
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Matrix> to_matrix(const Tensor& t) override
    {
        if (t.is_cpu())
            return Matrix(t.cpu_matrix());
        // batch 模式下必须先 flush（提交并等待），否则 GPU 计算未执行，读到旧数据
        // flush 后自动重新 begin_batch，保持 batch 上下文不断裂
        if (in_batch())
        {
            auto r = end_batch();
            if (!r) return std::unexpected(r.error());
            auto rb = begin_batch();
            if (!rb) return std::unexpected(rb.error());
        }
        return t.gpu_tensor().to_matrix(backend_);
    }

    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"copy_from: shape mismatch"});
        if (dst.is_cpu())
        {
            // 防御性：dst 应为 GPU Tensor，但若为 CPU 则直接拷贝
            dst = Tensor::from_matrix(Matrix(src));
            return {};
        }
        return backend_.upload_blocking(dst.gpu_tensor(), src.span());
    }

    [[nodiscard]] Result<Tensor> clone(const Tensor& src) override
    {
        if (src.is_cpu())
        {
            // 防御性：CPU Tensor 深拷贝
            return Tensor::from_matrix(Matrix(src.cpu_matrix()));
        }
        auto r = backend_.clone_gpu(src.gpu_tensor());
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 行切片：GPU 内拷贝连续行区间 ──
    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count) override
    {
        auto src_gpu = ensure_gpu(src);
        if (!src_gpu) return std::unexpected(src_gpu.error());
        auto r = backend_.slice_rows_gpu(src_gpu->gpu_tensor(), start_row, count);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 行插入：GPU 内就地写入连续行区间 ──
    // 注意：若 dst 为 CPU（防御性路径），回退到上传-拷贝-替换流程。
    // 正常路径下 dst 已是 GPU Tensor，直接就地修改 buffer。
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (dst.is_cpu())
        {
            // 防御性路径：dst 为 CPU，先确保 src 也在 CPU（下载）
            if (!src.is_cpu())
            {
                auto sm = to_matrix(src);
                if (!sm) return std::unexpected(sm.error());
                Tensor src_cpu = Tensor::from_matrix(Matrix(*sm));
                return insert_rows_cpu_fallback(dst, dst_start_row, src_cpu);
            }
            return insert_rows_cpu_fallback(dst, dst_start_row, src);
        }
        auto src_gpu = ensure_gpu(src);
        if (!src_gpu) return std::unexpected(src_gpu.error());
        return backend_.insert_rows_gpu(dst.gpu_tensor(), dst_start_row, src_gpu->gpu_tensor());
    }

    // ── gather_rows: 按 indices 从 table 中按行查表 ──
    // GPU-native 实现：全程在 GPU 执行，无 PCIe 传输。
    // indices 支持任意形状，按 flat 遍历所有元素。
    [[nodiscard]] Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices) override
    {
        auto tbl_gpu = ensure_gpu(table);
        if (!tbl_gpu) return std::unexpected(tbl_gpu.error());
        auto idx_gpu = ensure_gpu(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());

        auto r = backend_.gather_gpu(tbl_gpu->gpu_tensor(), idx_gpu->gpu_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst ──
    // GPU-native 实现：使用 CAS 循环实现 float atomicAdd，无 PCIe 传输。
    [[nodiscard]] Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad) override
    {
        if (!dst.is_gpu())
            return std::unexpected(Error{"scatter_add_rows: dst must be GPU tensor"});

        auto idx_gpu = ensure_gpu(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());
        auto grad_gpu = ensure_gpu(grad);
        if (!grad_gpu) return std::unexpected(grad_gpu.error());

        return backend_.scatter_add_gpu(
            dst.gpu_tensor(), idx_gpu->gpu_tensor(), grad_gpu->gpu_tensor());
    }

private:
    // CPU 路径回退（dst 为 CPU Tensor 时使用）
    [[nodiscard]] Result<void> insert_rows_cpu_fallback(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src)
    {
        Matrix& d = dst.cpu_matrix();
        const Matrix& s = src.cpu_matrix();
        if (d.cols() != s.cols())
            return std::unexpected(Error{"insert_rows: column count mismatch"});
        if (dst_start_row + s.rows() > d.rows())
            return std::unexpected(Error{"insert_rows: range out of bounds"});
        const auto dst_span = d.span();
        const auto src_span = s.span();
        const std::size_t cols = d.cols();
        for (std::size_t r = 0; r < s.rows(); ++r)
            std::copy_n(src_span.begin() + r * cols, cols,
                        dst_span.begin() + (dst_start_row + r) * cols);
        return {};
    }

public:

    // ── 3D 维度转置：(M, B, N) ↔ (B, M, N) ──
    [[nodiscard]] Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse) override
    {
        auto x_gpu = ensure_gpu(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());

        auto r = backend_.rearrange_3d_gpu(
            x_gpu->gpu_tensor(),
            static_cast<uint32_t>(M),
            static_cast<uint32_t>(B),
            static_cast<uint32_t>(N),
            inverse ? 1u : 0u);
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 矩阵转置：A (R, C) → out (C, R) ──
    [[nodiscard]] Result<Tensor> transpose(const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.transpose_gpu(a_gpu->gpu_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB) override
    {
        // 确保 A、B 在 GPU 上
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        auto r = backend_.matmul_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
            transA ? 1u : 0u, transB ? 1u : 0u);
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 批量矩阵乘法：按 batch 切分行块，单次 dispatch 处理所有 batch ──
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        auto r = backend_.batched_matmul_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
            static_cast<uint32_t>(batch),
            transA ? 1u : 0u, transB ? 1u : 0u);
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // A += B：分配新 buffer 计算 A+B，替换 A
    [[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"add_inplace: shape mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
            count, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f);  // BINARY, Add
        if (!r)
            return std::unexpected(r.error());
        A = Tensor::from_gpu(std::move(*r));
        return {};
    }

    // A *= s：分配新 buffer 计算 A*s，替换 A
    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // BINARY, Mul, flags=1 (B is scalar), scalar_b = s
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), nullptr, nullptr,
            count, 1u, 2u, 0u, 1u, static_cast<float>(s), 0.0f, 0.0f);
        if (!r)
            return std::unexpected(r.error());
        A = Tensor::from_gpu(std::move(*r));
        return {};
    }

    // 融合 axpy：A += scalar * B（mode=3，单次 dispatch 替代 clone+scale+add 三步）
    // 分配新 buffer 计算 A + scalar*B，替换 A（遵循 GpuEngine copy-on-write 语义）
    [[nodiscard]] Result<void> axpy_inplace(Tensor& A, Scalar scalar, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"axpy_inplace: shape mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // AXPY 模式 (mode=3): out = A + scalar_b * B
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
            count, 3u, 0u, 0u, 0u, static_cast<float>(scalar), 0.0f, 0.0f);
        if (!r)
            return std::unexpected(r.error());
        A = Tensor::from_gpu(std::move(*r));
        return {};
    }

    // A = 0：使用 vkCmdFillBuffer 真原地清零（不分配新 buffer）
    [[nodiscard]] Result<void> zero(Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.fill_zero_gpu(a_gpu->gpu_tensor());
        if (!r)
            return std::unexpected(r.error());
        // fill_zero 修改的是 GPU buffer 本身，Tensor 的 shared_ptr 不变
        // 但若 ensure_gpu 上传了新 Tensor，需要替换 A
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> row_reduce_sum(const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 0u, 0u);  // row, sum
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum(const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 1u, 0u);  // col, sum
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> row_reduce_max(const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 0u, 1u);  // row, max
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_max(const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 1u, 1u);  // col, max
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 广播原语
    // ══════════════════════════════════════════════════════════════════════

    // A[r][c] = op(A[r][c], row_vec[r])：分配新 buffer，替换 A
    [[nodiscard]] Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto rv_gpu = ensure_gpu(row_vec);
        if (!rv_gpu) return std::unexpected(rv_gpu.error());

        auto r = backend_.broadcast_gpu(
            a_gpu->gpu_tensor(), rv_gpu->gpu_tensor(),
            0u, static_cast<uint32_t>(op));  // row_broadcast
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_gpu(std::move(*r));
        return {};
    }

    // A[r][c] = op(A[r][c], col_vec[c])：分配新 buffer，替换 A
    [[nodiscard]] Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto cv_gpu = ensure_gpu(col_vec);
        if (!cv_gpu) return std::unexpected(cv_gpu.error());

        auto r = backend_.broadcast_gpu(
            a_gpu->gpu_tensor(), cv_gpu->gpu_tensor(),
            1u, static_cast<uint32_t>(op));  // col_broadcast
        if (!r) return std::unexpected(r.error());
        A = Tensor::from_gpu(std::move(*r));
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 逐元素原语
    // ══════════════════════════════════════════════════════════════════════

    // out = unary_op(A)
    [[nodiscard]] Result<Tensor> elementwise_unary(
        UnaryOp op, const Tensor& A) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), nullptr, nullptr,
            count, 0u, static_cast<uint32_t>(op), 0u, 0u,
            0.0f, 0.0f, 0.0f);  // UNARY
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // out = binary_op(A, B)
    [[nodiscard]] Result<Tensor> elementwise_binary(
        BinaryOp op, const Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"elementwise_binary: shape mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
            count, 1u, static_cast<uint32_t>(op), 0u, 0u,
            0.0f, 0.0f, 0.0f);  // BINARY
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // out = binary_op(A, s) 或 binary_op(s, A)
    [[nodiscard]] Result<Tensor> elementwise_binary_scalar(
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // flags: bit0 = B is scalar, bit3 = scalar first
        const uint32_t flags = 1u | (scalar_first ? 8u : 0u);
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), nullptr, nullptr,
            count, 1u, static_cast<uint32_t>(op), 0u, flags,
            static_cast<float>(s), 0.0f, 0.0f);  // BINARY with scalar
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 条件选择原语
    // ══════════════════════════════════════════════════════════════════════

    // out = compare_op(A, scalar_b) ? then_t : scalar_else
    [[nodiscard]] Result<Tensor> elementwise_select_scalar_cond(
        CompareOp cmp, const Tensor& A, Scalar scalar_b,
        const Tensor& then_t, Scalar scalar_else) override
    {
        if (A.rows() != then_t.rows() || A.cols() != then_t.cols())
            return std::unexpected(Error{"elementwise_select: A and then shape mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto t_gpu = ensure_gpu(then_t);
        if (!t_gpu) return std::unexpected(t_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // SELECT mode:
        //   b = scalar_b (flags bit0 = 1)
        //   then_v = B[idx] = then_t (flags bit1 = 0, uses binding 1)
        //   else_v = scalar_else (flags bit2 = 1)
        const uint32_t flags = 1u | 4u;  // bit0: b is scalar, bit2: else is scalar
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &t_gpu->gpu_tensor(), nullptr,
            count, 2u, 0u, static_cast<uint32_t>(cmp), flags,
            static_cast<float>(scalar_b), 0.0f,
            static_cast<float>(scalar_else));  // SELECT
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

private:
    // ── 辅助：确保 Tensor 在 GPU 上 ──────────────────────────────────────
    // 若已是 GPU，返回共享拷贝（零开销）；若为 CPU，上传到 GPU。
    // 纯 GPU 架构下，所有 Tensor 应已是 GPU，此方法为防御性兜底。
    [[nodiscard]] Result<Tensor> ensure_gpu(const Tensor& t)
    {
        if (t.is_gpu())
            return t;  // 共享拷贝
        if (!t.valid())
            return std::unexpected(Error{"ensure_gpu: invalid tensor"});
        auto r = GpuTensor::from_matrix(t.cpu_matrix(), backend_);
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN

#endif // NN_GPU_ENGINE_HPP
