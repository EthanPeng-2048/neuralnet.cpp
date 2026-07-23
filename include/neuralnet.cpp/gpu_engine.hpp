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

    // ── 批处理：当前为 no-op（同步模式） ──────────────────────────────────
    [[nodiscard]] Result<void> begin_batch() override { return {}; }
    [[nodiscard]] Result<void> end_batch() override { return {}; }

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
