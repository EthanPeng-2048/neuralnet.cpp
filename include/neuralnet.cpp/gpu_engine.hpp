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

#include "compute_engine.hpp"
#include "fused_exprs.hpp"
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
    // 表达式求值（闭合世界 AOT）
    //
    // 与 fused_exprs.hpp 预生成实例表逐一比对（expr_spec_equal），命中则
    // dispatch 单个融合 shader；未命中**硬报错**（无 eager、无运行时生成）。
    // 新增 GPU 表达式必须加入 kGenInstances 并生成 AOT shader。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols) override
    {
        // ── AOT 匹配：查找与 spec 完全一致的预生成融合 shader ──────────
        for (const auto& inst : nn::fused::kGenInstances)
        {
            if (!backend_.has_fused_shader(inst.name))
                continue;  // 未嵌入/未注册：跳过
            if (!expr_spec_equal(spec, nn::fused::make_fused(inst)))
                continue;

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
            auto out = backend_.run_fused_gpu(
                inst.name, gpu_inputs, spec.consts, rows, cols);
            if (!out) return std::unexpected(out.error());
            return Tensor::from_gpu(std::move(*out));
        }

        // ── 闭合世界：未命中任何 AOT 融合 shader → 硬报错 ──────────────
        return std::unexpected(Error{
            "GpuEngine::eval_expr: 未找到匹配的 AOT 融合 shader（闭合世界）；"
            "请将表达式加入 fused_exprs.hpp kGenInstances"});
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
