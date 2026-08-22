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
//   batch 录制模式已启用：begin_batch 后所有原语录制到共享 command buffer，
//   end_batch 一次 vkQueueSubmit + vkWaitForFences，消除 per-primitive 同步开销。
//   to_matrix/from_matrix 会打断 batch（flush 后自动重新 begin_batch）。
//
// 原地操作语义：
//   add_inplace / scale_inplace / broadcast_*_inplace 采用"分配新 buffer +
//   替换 Tensor"策略（copy-on-write 语义）。zero 使用 vkCmdFillBuffer 真原地
//   清零（避免每步分配）。
// ─────────────────────────────────────────────────────────────────────────

#ifdef NN_HAS_VULKAN

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

#include "compute_engine.hpp"
#include "cpu_engine.hpp"   // 优雅回退：未命中 AOT 融合 shader 时 CPU 求值
#if __has_include("fused_registry.hpp")
#include "fused_registry.hpp"
#endif
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

    // ── 表达式录制（M2 框架）：当前为 no-op ─────────────────────────────
    // 各原语照常独立 dispatch；录制融合分析（虚拟寄存器 DAG + 融合边界判定）
    // 在 M3 落地。begin/end 保持幂等，Layer 可先行包住算法段落。
    [[nodiscard]] Result<void> begin_expr() override { return {}; }
    [[nodiscard]] Result<void> end_expr() override { return {}; }

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
        // batch 模式下必须先 flush（提交并等待），否则 upload_blocking 会
        // 独立提交 command buffer 到队列，与正在录制的 batch 产生竞争。
        // flush 后自动重新 begin_batch，保持 batch 上下文不断裂。
        if (in_batch())
        {
            auto r = end_batch();
            if (!r) return std::unexpected(r.error());
            auto rb = begin_batch();
            if (!rb) return std::unexpected(rb.error());
        }
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
        // batch 模式下必须先 flush（与 from_matrix 同理，upload_blocking 独立提交）
        if (in_batch())
        {
            auto r = end_batch();
            if (!r) return std::unexpected(r.error());
            auto rb = begin_batch();
            if (!rb) return std::unexpected(rb.error());
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
    // 纯 GPU 架构：dst 必须为 GPU Tensor，不再回退 CPU 路径（原 cpu_fallback 已删除）。
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (dst.is_cpu())
            return std::unexpected(Error{"insert_rows: dst must be GPU tensor in pure-GPU architecture"});
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
    // batch 模式查询（内部使用，to_matrix/from_matrix 需检查）
    [[nodiscard]] bool in_batch() const noexcept { return backend_.in_batch(); }

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
    // alpha 在 shader 写出时一次完成（如注意力 1/sqrt(d_k) 缩放）
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        auto r = backend_.batched_matmul_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
            static_cast<uint32_t>(batch),
            transA ? 1u : 0u, transB ? 1u : 0u,
            static_cast<float>(alpha));
        if (!r)
            return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // matmul 融合原语（M4；GPU 融合 shader 不物化中间得分矩阵）
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> batched_matmul_reduce(
        const Tensor& A, const Tensor& B, std::size_t batch,
        ReduceOp op, bool transA, bool transB,
        Scalar alpha, bool reduce_cols, const Tensor* mask) override
    {
        if (!reduce_cols)
            return std::unexpected(Error{
                "GpuEngine::batched_matmul_reduce: reduce_cols=false 未在 GPU 实现"
                "（调用方回退 CPU/组合路径）"});
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_reduce: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_reduce: rows not divisible by batch"});
        const std::size_t a_rpb = A.rows() / batch;
        const std::size_t b_rpb = B.rows() / batch;
        const std::size_t M = transA ? A.cols() : a_rpb;
        const std::size_t K = transA ? a_rpb : A.cols();
        const std::size_t K2 = transB ? B.cols() : b_rpb;
        const std::size_t N = transB ? b_rpb : B.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul_reduce: K dimension mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        auto r = backend_.bmm_reduce_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(), mask_ptr,
            M, N, K, batch, static_cast<std::size_t>(op),
            transA, transB, alpha);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> batched_matmul_softmax_denom(
        const Tensor& A, const Tensor& B, const Tensor& row_max,
        std::size_t batch, bool transA, bool transB, Scalar alpha,
        const Tensor* mask) override
    {
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_softmax_denom: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_softmax_denom: rows not divisible by batch"});
        const std::size_t a_rpb = A.rows() / batch;
        const std::size_t b_rpb = B.rows() / batch;
        const std::size_t M = transA ? A.cols() : a_rpb;
        const std::size_t K = transA ? a_rpb : A.cols();
        const std::size_t K2 = transB ? B.cols() : b_rpb;
        const std::size_t N = transB ? b_rpb : B.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul_softmax_denom: K dimension mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto m_gpu = ensure_gpu(row_max);
        if (!m_gpu) return std::unexpected(m_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        auto r = backend_.bmm_denom_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(), mask_ptr,
            m_gpu->gpu_tensor(), M, N, K, batch, transA, transB, alpha);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> batched_matmul_softmax_apply(
        const Tensor& A, const Tensor& B, const Tensor& V,
        const Tensor& row_max, const Tensor& denom,
        std::size_t batch, bool transA, bool transB, Scalar alpha,
        const Tensor* mask) override
    {
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_softmax_apply: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0 || V.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_softmax_apply: rows not divisible by batch"});
        const std::size_t a_rpb = A.rows() / batch;
        const std::size_t b_rpb = B.rows() / batch;
        const std::size_t M = transA ? A.cols() : a_rpb;
        const std::size_t K = transA ? a_rpb : A.cols();
        const std::size_t K2 = transB ? B.cols() : b_rpb;
        const std::size_t N = transB ? b_rpb : B.cols();
        const std::size_t Nv = V.rows() / batch;
        const std::size_t D = V.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul_softmax_apply: K dimension mismatch"});
        if (Nv != N)
            return std::unexpected(Error{"batched_matmul_softmax_apply: V rows per batch != N"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto v_gpu = ensure_gpu(V);
        if (!v_gpu) return std::unexpected(v_gpu.error());
        auto m_gpu = ensure_gpu(row_max);
        if (!m_gpu) return std::unexpected(m_gpu.error());
        auto l_gpu = ensure_gpu(denom);
        if (!l_gpu) return std::unexpected(l_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        auto r = backend_.bmm_apply_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(), mask_ptr,
            m_gpu->gpu_tensor(), l_gpu->gpu_tensor(), v_gpu->gpu_tensor(),
            M, N, K, D, batch, transA, transB, alpha);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // 两趟式注意力反向 Pass 1：R 与 grad_Q（GPU 融合 kernel，不物化得分矩阵）
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_backward_q(
        const Tensor& A, const Tensor& B, const Tensor& P,
        const Tensor& row_max, const Tensor& denom,
        std::size_t batch, bool transA, bool transB, Scalar alpha,
        Tensor& r_out, const Tensor* mask) override
    {
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_softmax_backward_q: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0 || P.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_softmax_backward_q: rows not divisible by batch"});
        const std::size_t a_rpb = A.rows() / batch;
        const std::size_t b_rpb = B.rows() / batch;
        const std::size_t M = transA ? A.cols() : a_rpb;
        const std::size_t K = transA ? a_rpb : A.cols();
        const std::size_t K2 = transB ? B.cols() : b_rpb;
        const std::size_t N = transB ? b_rpb : B.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul_softmax_backward_q: K dimension mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto p_gpu = ensure_gpu(P);
        if (!p_gpu) return std::unexpected(p_gpu.error());
        auto m_gpu = ensure_gpu(row_max);
        if (!m_gpu) return std::unexpected(m_gpu.error());
        auto l_gpu = ensure_gpu(denom);
        if (!l_gpu) return std::unexpected(l_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        GpuTensor r_gpu;
        auto r = backend_.bmm_q_backward_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(), mask_ptr,
            m_gpu->gpu_tensor(), l_gpu->gpu_tensor(), p_gpu->gpu_tensor(),
            M, N, K, batch, transA, transB, alpha, &r_gpu);
        if (!r) return std::unexpected(r.error());
        r_out = Tensor::from_gpu(std::move(r_gpu));
        return Tensor::from_gpu(std::move(*r));
    }

    // 两趟式注意力反向 Pass 2：grad_K 与 grad_V（GPU 融合 kernel）
    [[nodiscard]] Result<Tensor> batched_matmul_softmax_backward_kv(
        const Tensor& A, const Tensor& B, const Tensor& P,
        const Tensor& G, const Tensor& R,
        const Tensor& row_max, const Tensor& denom,
        std::size_t batch, bool transA, bool transB, Scalar alpha,
        Tensor& grad_v_out, const Tensor* mask) override
    {
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_softmax_backward_kv: batch must be > 0"});
        if (A.rows() % batch != 0 || B.rows() % batch != 0 || P.rows() % batch != 0
            || G.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_softmax_backward_kv: rows not divisible by batch"});
        const std::size_t a_rpb = A.rows() / batch;
        const std::size_t b_rpb = B.rows() / batch;
        const std::size_t M = transA ? A.cols() : a_rpb;
        const std::size_t K = transA ? a_rpb : A.cols();
        const std::size_t K2 = transB ? B.cols() : b_rpb;
        const std::size_t N = transB ? b_rpb : B.cols();
        const std::size_t D = G.cols();
        if (K != K2)
            return std::unexpected(Error{"batched_matmul_softmax_backward_kv: K dimension mismatch"});
        if (G.rows() != batch * M)
            return std::unexpected(Error{"batched_matmul_softmax_backward_kv: G rows != batch*M"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());
        auto p_gpu = ensure_gpu(P);
        if (!p_gpu) return std::unexpected(p_gpu.error());
        auto g_gpu = ensure_gpu(G);
        if (!g_gpu) return std::unexpected(g_gpu.error());
        auto r_gpu = ensure_gpu(R);
        if (!r_gpu) return std::unexpected(r_gpu.error());
        auto m_gpu = ensure_gpu(row_max);
        if (!m_gpu) return std::unexpected(m_gpu.error());
        auto l_gpu = ensure_gpu(denom);
        if (!l_gpu) return std::unexpected(l_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        GpuTensor gv_gpu;
        auto r = backend_.bmm_kv_backward_gpu(
            a_gpu->gpu_tensor(), b_gpu->gpu_tensor(), mask_ptr,
            m_gpu->gpu_tensor(), l_gpu->gpu_tensor(), p_gpu->gpu_tensor(),
            g_gpu->gpu_tensor(), r_gpu->gpu_tensor(),
            M, N, K, D, batch, transA, transB, alpha, &gv_gpu);
        if (!r) return std::unexpected(r.error());
        grad_v_out = Tensor::from_gpu(std::move(gv_gpu));
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 列式 softmax 融合原语（M5；GPU 融合 shader 不物化 (C, N) exp/softmax）
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> col_softmax_denom(
        const Tensor& logits, const Tensor& col_max) override
    {
        const std::size_t C = logits.rows(), N = logits.cols();
        auto l_gpu = ensure_gpu(logits);
        if (!l_gpu) return std::unexpected(l_gpu.error());
        auto m_gpu = ensure_gpu(col_max);
        if (!m_gpu) return std::unexpected(m_gpu.error());
        auto r = backend_.col_softmax_denom_gpu(
            l_gpu->gpu_tensor(), m_gpu->gpu_tensor(), C, N);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_softmax_sparse_forward(
        const Tensor& logits, const Tensor& labels, const Tensor* mask,
        std::size_t vocab_size, Scalar inv_num_valid,
        Tensor& loss_vec_out) override
    {
        if (vocab_size > (std::size_t{1} << 24))
            return std::unexpected(Error{
                "col_softmax_sparse_forward: vocab_size exceeds 2^24 "
                "(float labels not exactly representable)"});
        const std::size_t C = logits.rows(), N = logits.cols();
        if (vocab_size > C)
            return std::unexpected(Error{"col_softmax_sparse_forward: vocab_size > classes"});
        auto l_gpu = ensure_gpu(logits);
        if (!l_gpu) return std::unexpected(l_gpu.error());
        auto lb_gpu = ensure_gpu(labels);
        if (!lb_gpu) return std::unexpected(lb_gpu.error());
        std::optional<GpuTensor> mask_gpu;
        GpuTensor* mask_ptr = nullptr;
        if (mask)
        {
            auto mg = ensure_gpu(*mask);
            if (!mg) return std::unexpected(mg.error());
            mask_gpu = mg->gpu_tensor();
            mask_ptr = &*mask_gpu;
        }
        GpuTensor lv_gpu;
        auto r = backend_.col_softmax_sparse_forward_gpu(
            l_gpu->gpu_tensor(), lb_gpu->gpu_tensor(), mask_ptr,
            C, N, vocab_size, inv_num_valid, &lv_gpu);
        if (!r) return std::unexpected(r.error());
        loss_vec_out = Tensor::from_gpu(std::move(lv_gpu));
        return Tensor::from_gpu(std::move(*r));
    }

    // A += B：真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    // 逐元素 kernel 每线程只读写自己下标一次，read-before-write 天然成立。
    // 免去新 buffer 分配 + 全量写出，消除优化器/梯度累积路径的分配风暴。
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
            count, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
            &a_gpu->gpu_tensor());  // BINARY, Add, 原地写回 A
        if (!r)
            return std::unexpected(r.error());
        // 原地模式下 A 的 buffer 已被更新；若 ensure_gpu 上传了新 Tensor（防御路径），替换 A
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // A *= s：真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // BINARY, Mul, flags=1 (B is scalar), scalar_b = s
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), nullptr, nullptr,
            count, 1u, 2u, 0u, 1u, static_cast<float>(s), 0.0f, 0.0f,
            &a_gpu->gpu_tensor());  // 原地写回 A
        if (!r)
            return std::unexpected(r.error());
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // 融合 axpy：A += scalar * B（mode=3，单次 dispatch 替代 clone+scale+add 三步）
    // 真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    [[nodiscard]] Result<void> axpy_inplace(Tensor& A, Scalar scalar, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"axpy_inplace: shape mismatch"});

        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        // AXPY 模式 (mode=3): out = A + scalar_b * B，原地写回 A
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
            count, 3u, 0u, 0u, 0u, static_cast<float>(scalar), 0.0f, 0.0f,
            &a_gpu->gpu_tensor());
        if (!r)
            return std::unexpected(r.error());
        if (A.is_cpu())
            A = std::move(*a_gpu);
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

    // A[r][c] = op(A[r][c], row_vec[r])：真原地，直接写回 A 的 buffer
    [[nodiscard]] Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto rv_gpu = ensure_gpu(row_vec);
        if (!rv_gpu) return std::unexpected(rv_gpu.error());

        auto r = backend_.broadcast_gpu(
            a_gpu->gpu_tensor(), rv_gpu->gpu_tensor(),
            0u, static_cast<uint32_t>(op),
            &a_gpu->gpu_tensor());  // row_broadcast，原地写回 A
        if (!r) return std::unexpected(r.error());
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // A[r][c] = op(A[r][c], col_vec[c])：真原地，直接写回 A 的 buffer
    [[nodiscard]] Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto cv_gpu = ensure_gpu(col_vec);
        if (!cv_gpu) return std::unexpected(cv_gpu.error());

        auto r = backend_.broadcast_gpu(
            a_gpu->gpu_tensor(), cv_gpu->gpu_tensor(),
            1u, static_cast<uint32_t>(op),
            &a_gpu->gpu_tensor());  // col_broadcast，原地写回 A
        if (!r) return std::unexpected(r.error());
        if (A.is_cpu())
            A = std::move(*a_gpu);
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

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（AOT 融合优先，未命中优雅回退 CPU）
    //
    // 折叠内联表达式 → expr_spec_key → 查构建期合成的融合 shader 注册表
    // （fused_registry.hpp，由 scan_exprs 收集 + gen_fused 合成）；命中则
    // dispatch 单个融合 shader；未命中（如形状未纳入闭合世界）**优雅回退**
    // CPU 求值（通用参考实现，正确性有保证，但慢）并打印一次性警告。
    //
    // 形状无关融合：RowMod/RotateHalf 的周期/块大小（如 RoPE 的 d_k）是
    // 运行时视图参数（不进 key），dispatch 时按实际 spec 填充 push constant
    // vp 槽 → 同结构不同形状共享一个融合 shader，任意 d_k 都全融合。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        // ── AOT 匹配：按规范结构 key 查预编译融合 shader ──────────────
        const std::string key = nn::expr_spec_key(spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs = nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(key))
        {
            // 命中：收集 GPU 输入（同一 buffer 可重复绑定，如 RoPE 的 q×2）
            // GpuTensor 内部为 shared_ptr<GpuBuffer>，拷贝即共享，零成本
            std::vector<GpuTensor> gpu_inputs;
            gpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs)
            {
                auto g = ensure_gpu(t);
                if (!g) return std::unexpected(g.error());
                gpu_inputs.push_back(g->gpu_tensor());
            }
            // 运行时视图参数（RowMod 周期 / RotateHalf 块大小）：
            // 同结构不同形状共享一个融合 shader，按实际 spec 填充 vp 槽
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_.run_fused_gpu(
                key, gpu_inputs, spec.consts, rows, cols,
                /*vector_out=*/false, vp);
            if (!out) return std::unexpected(out.error());
            return Tensor::from_gpu(std::move(*out));
        }
#endif

        // ── 优雅回退：未命中 AOT 融合 shader → CPU 求值（正确性兜底） ──
        warn_miss_once_(key);
        return eval_expr_cpu_fallback_(spec, inputs, rows, cols, /*reduce=*/false);
    }

    // ── 归约向量原生形状输出（M3：LayerNorm/RMSNorm 小向量缓存） ────────
    // 与 eval_expr 相同，但以 vector_out=1 调度归约融合 shader（thread 0 写
    // (rows,1)/(1,cols) 归约向量，不写全尺寸广播）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        const std::string key = nn::expr_spec_key(spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs = nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(key))
        {
            std::vector<GpuTensor> gpu_inputs;
            gpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs)
            {
                auto g = ensure_gpu(t);
                if (!g) return std::unexpected(g.error());
                gpu_inputs.push_back(g->gpu_tensor());
            }
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_.run_fused_gpu(
                key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/true, vp);
            if (!out) return std::unexpected(out.error());
            return Tensor::from_gpu(std::move(*out));
        }
#endif
        warn_miss_once_(key);
        return eval_expr_cpu_fallback_(spec, inputs, rows, cols, /*reduce=*/true);
    }

private:
    // ── 优雅回退：未命中 AOT 融合 shader → 下载输入 → CPU 求值 → 上传 ──
    // 仅未覆盖形状走此慢路径（正确性兜底）；命中仍走融合 shader。
    [[nodiscard]] Result<Tensor> eval_expr_cpu_fallback_(
        const ExprSpec& spec, std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, bool reduce)
    {
        CpuEngine cpu;
        std::vector<Tensor> cpu_inputs;
        cpu_inputs.reserve(inputs.size());
        for (const auto& t : inputs)
        {
            auto m = to_matrix(t);
            if (!m) return std::unexpected(m.error());
            cpu_inputs.push_back(Tensor::from_matrix(std::move(*m)));
        }
        auto r = reduce
            ? cpu.eval_expr_reduce(spec, cpu_inputs, rows, cols)
            : cpu.eval_expr(spec, cpu_inputs, rows, cols);
        if (!r) return std::unexpected(r.error());
        auto g = ensure_gpu(*r);
        if (!g) return std::unexpected(g.error());
        return g;  // ensure_gpu 返回共享 GPU 拷贝或上传
    }

    // 一次性警告：覆盖缺口仍可见（避免静默降级），但不再硬报错
    static void warn_miss_once_(const std::string& key)
    {
        static std::mutex m;
        static std::unordered_set<std::string> warned;
        std::lock_guard<std::mutex> lk(m);
        if (warned.insert(key).second)
            std::fprintf(stderr,
                "[GpuEngine] 警告：内联表达式未命中 AOT 融合 shader（key=%s），"
                "已回退 CPU 求值（正确但较慢）。建议将该路径补入 scan_exprs 覆盖。\n",
                key.c_str());
    }
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
