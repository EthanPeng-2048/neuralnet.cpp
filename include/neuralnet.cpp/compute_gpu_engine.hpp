#pragma once

// ── compute_gpu_engine.hpp — GPU 计算引擎实现（纯 GPU 架构）─────────────────────────
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
// 原地操作语义（2026-09 起已全部改为真原地，此前的 copy-on-write 描述已作废）：
//   add_inplace / scale_inplace / axpy_inplace / broadcast_*_inplace 均直接写回
//   A 自己的 buffer（逐元素 kernel 每线程只读写自己下标一次，read-before-write
//   天然成立；同一 command buffer 内按录制顺序执行）。zero 用 vkCmdFillBuffer。
//   目的：消除「分配新 buffer + 全量写出」在优化器/梯度累积路径上的分配风暴。
//   注意：真原地要求调用方保证 A 不被同一录制窗口内已录制的命令读引用，
//   也不能与别的 Tensor 共享 buffer 且语义上需要保持独立（否则请传副本）。
// ─────────────────────────────────────────────────────────────────────────

#ifdef NN_HAS_VULKAN

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "compute_engine.hpp"
#include "expr_opt.hpp"
#include "expr_dsl.hpp"    // P3-3：matmul_with_bias 经 DSL 融合（单一事实源）

#ifdef NN_FUSED_REGISTRY_EMBEDDED
#include "fused_registry.hpp"
#endif
#include "backend/compute_vk_backend.hpp"

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

    // ── 显存回收（L2）：end_batch 之后归还完全空闲的内存池底材 ──────
    [[nodiscard]] Result<void> release_idle_pool_blocks() override
    {
        return backend_.release_idle_pool_blocks();
    }

    // ── 显存池统计（L2 仪器化） ──────────────────────────────────────
    // 同时上报持久池（参数/权重）与瞬态池（激活/临时），否则只报持久池会
    // 掩盖瞬态池的占用，导致"拆分后显存不降反升"的误判。
    [[nodiscard]] std::string pool_stats() const override
    {
        const std::string p = backend_.memory_pool().pool_debug_stats().to_string();
        const std::string t = backend_.transient_pool().pool_debug_stats().to_string();
        // pending：已析构但因帧未 reap 而尚未归还池的字节（显存峰值归因的
        // 关键缺口——"Tensor 置空但 live 不降"的那部分账）。
        const std::string pd =
            " pending=" + std::to_string(backend_.pending_destroy_bytes() / (1024 * 1024)) + "MB";
        return "persist{" + p + "} transient{" + t + "}" + pd;
    }

    // ── 激活 offload（L1-offload）─────────────────────────────────────
    // GPU 激活 → host-visible 存储（录制式，batch 内不提交；数据由 GPU 写、
    // 仅作中转，主机不读）。返回封装 host-visible GpuBuffer 的 Tensor 句柄。
    [[nodiscard]] Result<Tensor> offload_store(const Tensor& src) override
    {
        if (src.is_cpu())
            return src;
        const auto& g = src.gpu_tensor();
        auto host = GpuTensor::create_host_visible_empty(g.rows(), g.cols(), backend_);
        if (!host) return std::unexpected(host.error());
        const VkDeviceSize size =
            static_cast<VkDeviceSize>(g.rows() * g.cols() * sizeof(float));
        auto r = backend_.copy_buffer_gpu(g.buffer().impl(), host->buffer().impl(), size);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*host));
    }

    // 从 host-visible 句柄复制回 GPU，恢复为 (rows, cols)（录制式）
    [[nodiscard]] Result<Tensor> offload_load(
        const Tensor& handle, std::size_t rows, std::size_t cols) override
    {
        if (handle.is_cpu())
            return handle.reshape(rows, cols);
        auto dst = GpuTensor::create_empty(rows, cols, backend_);
        if (!dst) return std::unexpected(dst.error());
        const VkDeviceSize size =
            static_cast<VkDeviceSize>(rows * cols * sizeof(float));
        auto r = backend_.copy_buffer_gpu(
            handle.gpu_tensor().buffer().impl(), dst->buffer().impl(), size);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*dst));
    }

    // ── activation offload slab（持久复用缓冲） ───────────────────────
    [[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes) override
    {
        auto g = GpuTensor::create_host_visible_empty(1, bytes, backend_);
        if (!g) return std::unexpected(g.error());
        return Tensor::from_gpu(std::move(*g));
    }

    // 把 src 复制到 buffer 的 offset（float 单位）处（录制式）
    [[nodiscard]] Result<void> offload_save(
        const Tensor& buffer, std::size_t offset, const Tensor& src) override
    {
        if (src.is_cpu())
            return {};
        const VkDeviceSize size =
            static_cast<VkDeviceSize>(src.size() * sizeof(float));
        return backend_.copy_buffer_region_gpu(
            src.gpu_tensor().buffer().impl(), 0,
            buffer.gpu_tensor().buffer().impl(),
            static_cast<VkDeviceSize>(offset * sizeof(float)), size);
    }

    // 从 buffer 的 offset（float 单位）处复制 rows×cols 到新 GPU tensor（录制式）
    [[nodiscard]] Result<Tensor> offload_restore(
        const Tensor& buffer, std::size_t offset,
        std::size_t rows, std::size_t cols) override
    {
        auto dst = GpuTensor::create_empty(rows, cols, backend_);
        if (!dst) return std::unexpected(dst.error());
        const VkDeviceSize size =
            static_cast<VkDeviceSize>(rows * cols * sizeof(float));
        auto r = backend_.copy_buffer_region_gpu(
            buffer.gpu_tensor().buffer().impl(),
            static_cast<VkDeviceSize>(offset * sizeof(float)),
            dst->buffer().impl(), 0, size);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*dst));
    }

    // ── 异步标量回读（P0-2）───────────────────────────────────────────
    // 把 (1,1) F32 标量排入一次 D2H 拷贝并提交，不等待。调用时机：产出该
    // 标量的主帧（flush_batch/end_batch）已提交之后——同队列 FIFO 保证拷贝
    // 执行在生产命令之后。poll 非阻塞，就绪即取（见 compute_engine.hpp）。
    [[nodiscard]] Result<void> submit_scalar_readback(
        std::size_t slot, const Tensor& t) override
    {
        if (t.is_cpu())
            return std::unexpected(Error{
                "submit_scalar_readback: 期望 GPU 张量"});
        if (t.precision() != Precision::F32)
            return std::unexpected(Error{
                "submit_scalar_readback: 仅支持 F32 标量（调用方先 cast）"});
        return backend_.submit_scalar_readback(
            slot, t.gpu_tensor().buffer().impl());
    }

    [[nodiscard]] Result<bool> poll_scalar_readback(
        std::size_t slot, Scalar& out) override
    {
        float v = 0.0f;
        auto r = backend_.poll_scalar_readback(slot, v);
        if (!r) return std::unexpected(r.error());
        out = static_cast<Scalar>(v);
        return *r;
    }

    [[nodiscard]] std::size_t scalar_readback_slots() const override
    {
        return GpuBackend::scalar_readback_slot_count();
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
    // 统一接口：create_tensor / from_matrix / to_matrix（§6.4, §6.5）
    // P 由调用方显式指定（§8.5）：无隐式推导，无 Auto
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols, Precision P = Precision::F32) override
    {
        if (P == Precision::F16)
        {
            auto r = GpuTensorF16::create_empty(rows, cols, backend_);
            if (!r) return Tensor();
            return Tensor::from_gpu(std::move(*r));
        }
        auto r = GpuTensor::create_empty(rows, cols, backend_);
        if (!r) return Tensor();
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m, Precision P = Precision::F32) override
    {
        if (P == Precision::F16)
        {
            MatrixT<Precision::F16> m16(m.rows(), m.cols());
            const auto src = m.span();
            auto dst = m16.span();
            for (std::size_t i = 0; i < src.size(); ++i)
                dst[i] = src[i];
            auto r = GpuTensorF16::from_matrix(m16, backend_);
            if (!r) return std::unexpected(r.error());
            return Tensor::from_gpu(std::move(*r));
        }
        auto r = GpuTensor::from_matrix(m, backend_);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) override
    {
        if (t.is_cpu())
        {
            if (t.precision() == Precision::F16 && P == Precision::F32)
            {
                const auto& m16 = t.cpu_matrix<Precision::F16>();
                Matrix m32(m16.rows(), m16.cols());
                const auto src = m16.span();
                auto dst = m32.span();
                for (std::size_t i = 0; i < src.size(); ++i)
                    dst[i] = static_cast<float>(src[i]);
                return m32;
            }
            return Matrix(t.cpu_matrix());
        }
        // GPU tensor → drain + download
        if (in_batch())
        {
            auto r = end_batch();
            if (!r) return std::unexpected(r.error());
            auto rw = backend_.wait_in_flight();
            if (!rw) return std::unexpected(rw.error());
            auto rb = begin_batch();
            if (!rb) return std::unexpected(rb.error());
        }
        if (t.precision() == Precision::F16)
        {
            auto m16_r = t.gpu_tensor<Precision::F16>().to_matrix(backend_);
            if (!m16_r) return std::unexpected(m16_r.error());
            if (P == Precision::F32)
            {
                Matrix m32(m16_r->rows(), m16_r->cols());
                const auto src = m16_r->span();
                auto dst = m32.span();
                for (std::size_t i = 0; i < src.size(); ++i)
                    dst[i] = static_cast<float>(src[i]);
                return m32;
            }
        }
        return t.gpu_tensor().to_matrix(backend_);
    }

    // ── cast 原语（§7.5，统一接口；GPU 原生转换，无 PCIe 往返）─────────────
    [[nodiscard]] Result<Tensor> cast(const Tensor& src, Precision dst) override
    {
        if (src.precision() == dst)
            return src;

        // 保留精度 Pair（BF16/F64 Phase 1 未实现）清晰报错
        if (auto pcheck = check_precision_supported(src.precision()); !pcheck)
            return std::unexpected(pcheck.error());
        if (auto pcheck = check_precision_supported(dst); !pcheck)
            return std::unexpected(pcheck.error());

        if (src.is_gpu())
        {
            const std::size_t count = src.rows() * src.cols();
            if (src.precision() == Precision::F16 && dst == Precision::F32)
            {
                // f16 GPU → f32 GPU（升 cast，精确无损，GPU kernel）
                auto dst_gpu = GpuTensor::create_empty(src.rows(), src.cols(), backend_);
                if (!dst_gpu) return std::unexpected(dst_gpu.error());
                auto r = backend_.cast_gpu(
                    src.gpu_tensor<Precision::F16>().buffer(), dst_gpu->buffer(),
                    count, /*kind=0*/ 0u);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(std::move(*dst_gpu));
            }
            if (src.precision() == Precision::F32 && dst == Precision::F16)
            {
                // f32 GPU → f16 GPU（降 cast，round-half-to-even，GPU kernel）
                // create_f16_tensor 保证奇数元素 count 时 word 写入不越界。
                auto dst_gpu = backend_.create_f16_tensor(src.rows(), src.cols());
                if (!dst_gpu) return std::unexpected(dst_gpu.error());
                auto r = backend_.cast_gpu(
                    src.gpu_tensor().buffer(), dst_gpu->buffer(),
                    count, /*kind=1*/ 1u);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(std::move(*dst_gpu));
            }
            return std::unexpected(Error{"cast: unsupported precision conversion"});
        }
        // CPU path
        if (src.precision() == Precision::F16 && dst == Precision::F32)
        {
            const auto& m16 = src.cpu_matrix<Precision::F16>();
            Matrix m32(m16.rows(), m16.cols());
            const auto s = m16.span();
            auto d = m32.span();
            for (std::size_t i = 0; i < s.size(); ++i)
                d[i] = static_cast<float>(s[i]);
            return from_matrix(m32, Precision::F32);
        }
        if (src.precision() == Precision::F32 && dst == Precision::F16)
        {
            const auto& m32 = src.cpu_matrix();
            MatrixT<Precision::F16> m16(m32.rows(), m32.cols());
            const auto s = m32.span();
            auto d = m16.span();
            for (std::size_t i = 0; i < s.size(); ++i)
                d[i] = s[i];
            // CPU f16 tensor
            return Tensor::from_matrix(std::move(m16));
        }
        return std::unexpected(Error{"cast: unsupported precision conversion"});
    }

    // ── cast_into / copy_into（多精度适配层的"写回原存储"路径，§6.5）──────
    // 与 cast 的区别是**落点**：保留 dst 的对象身份与底层 buffer（in-place
    // 语义必需——若替换 dst 对象，其它持有同一张量句柄的缓存会静默失联）。
    // GPU 路径用 vkCmdCopyBuffer / cast 原语直接写 dst 的既有 buffer，无分配。
    [[nodiscard]] Result<void> copy_into(Tensor& dst, const Tensor& src) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"copy_into: shape mismatch"});
        if (dst.precision() != src.precision())
            return std::unexpected(Error{"copy_into: precision mismatch"});
        if (dst.is_cpu() != src.is_cpu())
            return std::unexpected(Error{"copy_into: device mismatch"});
        const std::size_t count = dst.rows() * dst.cols();
        if (dst.is_gpu())
        {
            if (dst.precision() == Precision::F16)
                return backend_.copy_buffer_gpu(
                    src.gpu_tensor<Precision::F16>().buffer().impl(),
                    dst.gpu_tensor<Precision::F16>().buffer().impl(),
                    static_cast<VkDeviceSize>(count * 2u));
            return backend_.copy_buffer_gpu(
                src.gpu_tensor().buffer().impl(),
                dst.gpu_tensor().buffer().impl(),
                static_cast<VkDeviceSize>(count * sizeof(float)));
        }
        if (dst.precision() == Precision::F16)
        {
            const auto s = src.cpu_matrix<Precision::F16>().span();
            auto d = dst.cpu_matrix<Precision::F16>().span();
            for (std::size_t i = 0; i < s.size(); ++i) d[i] = s[i];
        }
        else
        {
            const auto s = src.cpu_matrix().span();
            auto d = dst.cpu_matrix().span();
            for (std::size_t i = 0; i < s.size(); ++i) d[i] = s[i];
        }
        return {};
    }

    [[nodiscard]] Result<void> cast_into(const Tensor& src, Tensor& dst) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"cast_into: shape mismatch"});
        if (src.precision() == dst.precision())
            return copy_into(dst, src);
        if (dst.is_cpu() != src.is_cpu())
            return std::unexpected(Error{"cast_into: device mismatch"});
        const std::size_t count = dst.rows() * dst.cols();
        if (dst.is_gpu())
        {
            if (src.precision() == Precision::F16 && dst.precision() == Precision::F32)
                return backend_.cast_gpu(
                    src.gpu_tensor<Precision::F16>().buffer(),
                    dst.gpu_tensor().buffer(), count, /*kind=*/0u);
            if (src.precision() == Precision::F32 && dst.precision() == Precision::F16)
                return backend_.cast_gpu(
                    src.gpu_tensor().buffer(),
                    dst.gpu_tensor<Precision::F16>().buffer(), count, /*kind=*/1u);
            return std::unexpected(Error{"cast_into: unsupported precision conversion"});
        }
        if (src.precision() == Precision::F16 && dst.precision() == Precision::F32)
        {
            const auto s = src.cpu_matrix<Precision::F16>().span();
            auto d = dst.cpu_matrix().span();
            for (std::size_t i = 0; i < s.size(); ++i) d[i] = static_cast<float>(s[i]);
            return {};
        }
        if (src.precision() == Precision::F32 && dst.precision() == Precision::F16)
        {
            const auto s = src.cpu_matrix().span();
            auto d = dst.cpu_matrix<Precision::F16>().span();
            for (std::size_t i = 0; i < s.size(); ++i) d[i] = s[i];
            return {};
        }
        return std::unexpected(Error{"cast_into: unsupported precision conversion"});
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
        // batch 模式下必须先 drain（P0-1）：dst 是当前帧已引用的既有
        // buffer——当前帧尚未提交，若先把上传提交到队列，GPU 会先执行
        // 上传、后执行当前帧命令 → 本帧后续对 dst 的写入覆盖上传数据。
        // 故：end_batch 提交当前帧（不等待）→ wait_in_flight 等完 →
        // 新帧开始录制 → 上传（独立提交，排在新帧之后提交、GPU 先执行）。
        if (in_batch())
        {
            auto r = end_batch();
            if (!r) return std::unexpected(r.error());
            auto rw = backend_.wait_in_flight();
            if (!rw) return std::unexpected(rw.error());
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
        if (src.precision() == Precision::F16)   // 原生 f16 字节拷贝（后端模板化实现）
        {
            auto r16 = backend_.clone_gpu(src.gpu_tensor<Precision::F16>());
            if (!r16) return std::unexpected(r16.error());
            return Tensor::from_gpu(std::move(*r16));
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
        if (src_gpu->precision() == Precision::F16)   // 原生 f16 行切片（纯字节拷贝）
        {
            auto r16 = backend_.slice_rows_gpu(
                src_gpu->gpu_tensor<Precision::F16>(), start_row, count);
            if (!r16) return std::unexpected(r16.error());
            return Tensor::from_gpu(std::move(*r16));
        }
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
        if (dst.precision() == Precision::F16)   // 原生 f16 行插入（纯字节拷贝）
            return backend_.insert_rows_gpu(dst.gpu_tensor<Precision::F16>(),
                                            dst_start_row,
                                            src_gpu->gpu_tensor<Precision::F16>());
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

    // ── 卷积/池化窗口展开（纯数据搬运，无算法；契约见 compute_engine.hpp）──
    [[nodiscard]] Result<Tensor> im2col(
        const Tensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        auto x_gpu = ensure_gpu(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        auto r = backend_.im2col_gpu(x_gpu->gpu_tensor(),
                                     C, H, W, k, stride, pad, OH, OW);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col2im(
        const Tensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        auto c_gpu = ensure_gpu(col);
        if (!c_gpu) return std::unexpected(c_gpu.error());
        auto r = backend_.col2im_gpu(c_gpu->gpu_tensor(),
                                     C, H, W, k, stride, pad, OH, OW);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 分组归约（契约见 compute_engine.hpp）──────────────────────────
    [[nodiscard]] Result<Tensor> grouped_reduce_sum(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision = Precision::F32) override
    {
        auto x_gpu = ensure_gpu(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        auto r = backend_.grouped_reduce_gpu(x_gpu->gpu_tensor(), G, R, /*is_max=*/false);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> grouped_reduce_max(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision = Precision::F32) override
    {
        auto x_gpu = ensure_gpu(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        auto r = backend_.grouped_reduce_gpu(x_gpu->gpu_tensor(), G, R, /*is_max=*/true);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB,
        Precision P = Precision::F32) override
    {
        // 确保 A、B 在 GPU 上
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        // ── F32 路径（现状，零改动）─────────────────────────────────────
        if (P == Precision::F32)
        {
            auto r = backend_.matmul_gpu(
                a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
                transA ? 1u : 0u, transB ? 1u : 0u);
            if (!r)
                return std::unexpected(r.error());
            return Tensor::from_gpu(std::move(*r));
        }

        // ── F16 路径（Phase 2 in-kernel f16）──────────────────────────────
        // 两个操作数都是 f16 且设备支持 SSBO 16 位存储 → 走 **f16 存储版 GEMM**
        // （f16 直读 + f32 累加 + f16 写出）：不再为每个操作数物化整份 f32 副本
        // （那正是训练 transient 膨胀的主因）。小 N 走 GEMV 的场景仍用边界 cast
        // 回退——张量本来就小，收益为零却要吃 64×64 块空转。
        if (P == Precision::F16)
        {
            const std::size_t n_out = transB ? B.rows() : B.cols();
            if (a_gpu->precision() == Precision::F16 &&
                b_gpu->precision() == Precision::F16 && n_out > 8 &&
                backend_.has_matmul_tiled_f16_pipeline())
            {
                auto r = backend_.matmul_gpu(
                    f16_view(*a_gpu), f16_view(*b_gpu),
                    transA ? 1u : 0u, transB ? 1u : 0u, /*f16_io=*/true);
                if (!r)
                    return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
        }

        // ── F16 边界 cast 回退（无原生 f16 GEMM 时）──────────────────────
        // 边界 cast：f16 → f32（§7.5 cast 原语）
        if (P == Precision::F16)
        {
            // 边界 cast：f16 → f32（§7.5 cast 原语）
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            auto b32 = cast(*b_gpu, Precision::F32);
            if (!b32) return std::unexpected(b32.error());

            auto a32_gpu = ensure_gpu(*a32);
            if (!a32_gpu) return std::unexpected(a32_gpu.error());
            auto b32_gpu = ensure_gpu(*b32);
            if (!b32_gpu) return std::unexpected(b32_gpu.error());

            // f32 matmul
            auto r = backend_.matmul_gpu(
                a32_gpu->gpu_tensor(), b32_gpu->gpu_tensor(),
                transA ? 1u : 0u, transB ? 1u : 0u);
            if (!r)
                return std::unexpected(r.error());

            // cast 回 f16
            Tensor result_f32 = Tensor::from_gpu(std::move(*r));
            return cast(result_f32, Precision::F16);
        }

        return std::unexpected(Error{"matmul: unsupported precision"});
    }

    // ── matmul + 行广播 bias（P3-3）：经 DSL 融合（单一事实源）──────────────
    // 走 eval_expr → AOT fusion shader 精确匹配（闭合世界）。同时让
    // scan_exprs dry-run 收集该"Linear 结构"spec，gen_fused 生成融合 kernel，
    // 使 GPU Linear::forward 免去 to_matrix/from_matrix CPU 往返（此前走基类
    // matmul_with_bias 默认的 CPU 往返，且该 DSL 结构从未被扫描）。
    //
    // 多精度（Phase 2 in-kernel f16）：P 必须下传——f16 时**先试带类型变体**
    // （f16 直读 / f32 累加 / f16 写回，零边界 cast：Linear::forward 是 cast
    // 临时量最大的单一来源），无变体才在本函数内回退"抬 f32 → f32 融合 →
    // 落回"（**绝不把缺变体变成硬报错**，闭合世界原则只在 f32 结构缺失时适用）。
    [[nodiscard]] Result<Tensor> matmul_with_bias(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32) override
    {
        const std::size_t rows = transA ? A.cols() : A.rows();
        const std::size_t cols = transB ? B.rows() : B.cols();
        if (P == Precision::F32)
            return nn::dsl::compute(*this,
                nn::dsl::matmul(A, B, transA, transB, 1)
                    + nn::dsl::row_broadcast(bias),
                rows, cols);

        auto [spec, inputs] = nn::dsl::to_expr_spec(
            nn::dsl::matmul(A, B, transA, transB, 1) + nn::dsl::row_broadcast(bias));
        if (auto v = nn::validate_expr_spec(spec, inputs.size()); !v)
            return std::unexpected(v.error());
        if (supports_expr_precision_variant(spec, inputs, P))
            return eval_expr(spec, inputs, rows, cols, P);

        // ── 边界 cast 回退（带类型变体未预生成）──────────────────────────
        const Tensor* ops[3] = {&A, &B, &bias};
        Tensor c32[3];
        for (int i = 0; i < 3; ++i)
        {
            if (ops[i]->precision() == Precision::F32)
            {
                c32[i] = *ops[i];
                continue;
            }
            auto c = cast(*ops[i], Precision::F32);
            if (!c) return std::unexpected(c.error());
            c32[i] = std::move(*c);
        }
        auto r = nn::dsl::compute(*this,
            nn::dsl::matmul(c32[0], c32[1], transA, transB, 1)
                + nn::dsl::row_broadcast(c32[2]),
            rows, cols);
        if (!r) return std::unexpected(r.error());
        return cast(*r, P);
    }

    // ── 批量矩阵乘法：按 batch 切分行块，单次 dispatch 处理所有 batch ──
    // alpha 在 shader 写出时一次完成（如注意力 1/sqrt(d_k) 缩放）
    // P: 计算精度（D5 §8.1，同 matmul）
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha,
        Precision P = Precision::F32) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = ensure_gpu(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        // ── F32 路径（现状，零改动）─────────────────────────────────────
        if (P == Precision::F32)
        {
            auto r = backend_.batched_matmul_gpu(
                a_gpu->gpu_tensor(), b_gpu->gpu_tensor(),
                static_cast<uint32_t>(batch),
                transA ? 1u : 0u, transB ? 1u : 0u,
                static_cast<float>(alpha));
            if (!r)
                return std::unexpected(r.error());
            return Tensor::from_gpu(std::move(*r));
        }

        // ── F16 路径（Phase 2 in-kernel f16）──────────────────────────────
        // 同 matmul：两个操作数都 f16 → f16 存储版 batch GEMM（零 f32 副本）。
        if (P == Precision::F16 && a_gpu->precision() == Precision::F16 &&
            b_gpu->precision() == Precision::F16 &&
            backend_.has_batched_matmul_f16_pipeline())
        {
            auto r = backend_.batched_matmul_gpu(
                f16_view(*a_gpu), f16_view(*b_gpu),
                static_cast<uint32_t>(batch), transA ? 1u : 0u, transB ? 1u : 0u,
                static_cast<float>(alpha), /*f16_io=*/true);
            if (!r)
                return std::unexpected(r.error());
            return Tensor::from_gpu(
                GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
        }

        // ── F16 边界 cast 回退（无原生 f16 GEMM 时）──────────────────────
        if (P == Precision::F16)
        {
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            auto b32 = cast(*b_gpu, Precision::F32);
            if (!b32) return std::unexpected(b32.error());

            auto a32_gpu = ensure_gpu(*a32);
            if (!a32_gpu) return std::unexpected(a32_gpu.error());
            auto b32_gpu = ensure_gpu(*b32);
            if (!b32_gpu) return std::unexpected(b32_gpu.error());

            auto r = backend_.batched_matmul_gpu(
                a32_gpu->gpu_tensor(), b32_gpu->gpu_tensor(),
                static_cast<uint32_t>(batch),
                transA ? 1u : 0u, transB ? 1u : 0u,
                static_cast<float>(alpha));
            if (!r)
                return std::unexpected(r.error());

            Tensor result_f32 = Tensor::from_gpu(std::move(*r));
            return cast(result_f32, Precision::F16);
        }

        return std::unexpected(Error{"batched_matmul: unsupported precision"});
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
        if (a_gpu->precision() == Precision::F16)   // vkCmdFillBuffer 字节级 → f16 原生可用
        {
            auto r16 = backend_.fill_zero_gpu(a_gpu->gpu_tensor<Precision::F16>());
            if (!r16) return std::unexpected(r16.error());
            if (A.is_cpu()) A = std::move(*a_gpu);
            return {};
        }
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
    // 扫描级原语（RLA）——手写原语 shader + GpuBackend 接线
    // 形状契约见 compute_engine.hpp 扫描级原语注释；纯 GPU 架构：
    // 无 pipeline 时硬报错，不做 CPU 回退。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> scan_prefix_outer(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision = Precision::F32) override
    {
        auto k = ensure_gpu(K); if (!k) return std::unexpected(k.error());
        auto v = ensure_gpu(V); if (!v) return std::unexpected(v.error());
        auto p = ensure_gpu(P); if (!p) return std::unexpected(p.error());
        auto r = ensure_gpu(R); if (!r) return std::unexpected(r.error());
        auto a = ensure_gpu(A0); if (!a) return std::unexpected(a.error());
        auto b = ensure_gpu(B0); if (!b) return std::unexpected(b.error());
        auto bn = ensure_gpu(boundary); if (!bn) return std::unexpected(bn.error());
        auto res = backend_.scan_prefix_outer_gpu(
            k->gpu_tensor(), v->gpu_tensor(), p->gpu_tensor(), r->gpu_tensor(),
            a->gpu_tensor(), b->gpu_tensor(), has_state,
            static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
            bn->gpu_tensor(), has_bnd);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    [[nodiscard]] Result<Tensor> scan_suffix_outer(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision = Precision::F32) override
    {
        auto d = ensure_gpu(D); if (!d) return std::unexpected(d.error());
        auto x = ensure_gpu(X); if (!x) return std::unexpected(x.error());
        auto y = ensure_gpu(Y); if (!y) return std::unexpected(y.error());
        auto bn = ensure_gpu(boundary); if (!bn) return std::unexpected(bn.error());
        auto res = backend_.scan_suffix_outer_gpu(
            d->gpu_tensor(), x->gpu_tensor(), y->gpu_tensor(),
            static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
            bn->gpu_tensor(), has_bnd);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    [[nodiscard]] Result<Tensor> outer_col(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale,
        Precision = Precision::F32) override
    {
        auto p = ensure_gpu(P); if (!p) return std::unexpected(p.error());
        auto r = ensure_gpu(R); if (!r) return std::unexpected(r.error());
        auto s = ensure_gpu(S); if (!s) return std::unexpected(s.error());
        auto res = backend_.outer_col_gpu(
            p->gpu_tensor(), r->gpu_tensor(), s->gpu_tensor(),
            static_cast<uint32_t>(dk), has_scale);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> row_reduce_sum(const Tensor& A, Precision = Precision::F32) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 0u, 0u);  // row, sum
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum(const Tensor& A, Precision = Precision::F32) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 1u, 0u);  // col, sum
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> row_reduce_max(const Tensor& A, Precision = Precision::F32) override
    {
        auto a_gpu = ensure_gpu(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto r = backend_.reduce_gpu(a_gpu->gpu_tensor(), 0u, 1u);  // row, max
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_max(const Tensor& A, Precision = Precision::F32) override
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
        UnaryOp op, const Tensor& A, Precision = Precision::F32) override
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
        BinaryOp op, const Tensor& A, const Tensor& B,
        Precision = Precision::F32) override
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
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first,
        Precision = Precision::F32) override
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
        const Tensor& then_t, Scalar scalar_else,
        Precision = Precision::F32) override
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
    // 表达式求值（闭合世界 AOT；未命中硬报错，绝不静默回退）
    //
    // 折叠内联表达式 → expr_spec_key → 查构建期合成的融合 shader 注册表
    // （fused_registry.hpp，由 scan_exprs 收集 + gen_fused 合成）；命中则
    // dispatch 单个融合 shader；未命中**硬报错**（无 eager、无运行时生成、
    // 不回退 CPU），提示把该内联表达式纳入构建期扫描。
    //
    // 形状无关融合：RowMod/RotateHalf 的周期/块大小（如 RoPE 的 d_k）是
    // 运行时视图参数（不进 key），dispatch 时按实际 spec 填充 push constant
    // vp 槽 → 同结构不同形状共享一个融合 shader，任意 d_k 都全融合。

    // ── 融合输入的类型擦除收集：绑定只关心底层 buffer（元素类型由 shader 声明
    //    决定）→ 同一结构可在 f32 / f16 两种存储上跑（Phase 2 in-kernel f16）。
    //    owners 必须与 bufs 同寿命：nsure_gpu 上传出的 Tensor 若不被持有，
    //    其 GpuBuffer（此刻唯一 owner）会在录制中途析构 → 绑定悬空（铁律 6：
    //    录制期生命周期；实测表现为偶发错值，如 A+=B err=0.5）。
    struct FusedInputs
    {
        std::vector<Tensor>           owners;
        std::vector<const GpuBuffer*> bufs;
    };

    [[nodiscard]] Result<FusedInputs> fused_buffers_(std::span<const Tensor> ts)
    {
        FusedInputs out;
        out.owners.reserve(ts.size());
        out.bufs.reserve(ts.size());
        for (const auto& t : ts)
        {
            auto g = ensure_gpu(t);
            if (!g) return std::unexpected(g.error());
            if (g->precision() == Precision::F16)
                out.bufs.push_back(&g->gpu_tensor<Precision::F16>().buffer());
            else
                out.bufs.push_back(&g->gpu_tensor().buffer());
            out.owners.push_back(std::move(*g));   // 保活（见上）
        }
        return out;
    }
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& raw_spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32) override
    {
        // ── fold 段（P-C1）：canonicalize 对 fold 恒等（expr_opt 入口
        //    early-return），scan/runtime 两端 key 同源 → 直接用 raw_spec 查表。
        //    PC/分派形态由 run_fused_gpu 的 fold_k 参数走（见其形态校验）。
        if (raw_spec.fold)
        {
            const std::string fkey = nn::expr_spec_key(raw_spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
            const nn::fused::FusedShader* ffs = nn::fused::find_fused(fkey);
            if (ffs && backend_.has_fused_shader(fkey))
            {
                auto fi_r = fused_buffers_(inputs);
                if (!fi_r) return std::unexpected(fi_r.error());
                std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;
                const auto fvp = nn::expr_spec_runtime_view_params(raw_spec);
                auto out = backend_.run_fused_gpu(
                    fkey, gpu_inputs, raw_spec.consts, rows, cols,
                    /*vector_out=*/false, fvp, raw_spec.rparams,
                    /*output_override=*/nullptr,
                    // P-C2：fold 自带 matmul 段的 k/batch（7 槽 PC 的 slot5/6）
                    nn::expr_spec_runtime_matmul_k(raw_spec),
                    nn::expr_spec_runtime_matmul_batch(raw_spec),
                    nn::expr_spec_runtime_fold_k(raw_spec));
                if (!out) return std::unexpected(out.error());
                return Tensor::from_gpu(std::move(*out));
            }
#endif
            return std::unexpected(Error{
                "GpuEngine::eval_expr: fold 表达式未命中 AOT 融合 shader"
                "（闭合世界）；请将该 fold 结构纳入 scan_exprs"});
        }
        // ── canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        //    key 与 shader 合成两端一致；dispatch 用 canonical 的 consts ──
        const ExprSpec spec = nn::canonicalize_expr_spec(raw_spec);
        // ── AOT 匹配：按规范结构 key 查预编译融合 shader ──────────────
        const std::string key = nn::expr_spec_key(spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 精度变体优先（Phase 2 in-kernel f16）：按**真实输入精度** + 目标输出
        // 精度取 (key, sig)；命中带类型变体 → shader 直接半精度读/写，无需边界
        // cast（专为消除"每算子 f32 副本"的 transient 膨胀）。未命中则退回全 f32
        // key —— 此时输入必须已全 f32（由 PrecisionEngine 适配层 cast）。
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
        const std::string vkey = nn::expr_prec_sig_key(key, psig);
        const nn::fused::FusedShader* fs = (psig != 0) ? nn::fused::find_fused(vkey)
                                                       : nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(fs->key))
        {
            // 命中：收集 GPU 输入（同一 buffer 可重复绑定，如 RoPE 的 q×2）
            // GpuTensor 内部为 shared_ptr<GpuBuffer>，拷贝即共享，零成本
            auto fi_r = fused_buffers_(inputs);
            if (!fi_r) return std::unexpected(fi_r.error());
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            // 运行时视图参数（RowMod 周期 / RotateHalf 块大小）：
            // 同结构不同形状共享一个融合 shader，按实际 spec 填充 vp 槽
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            // 输出存储精度由变体签名决定（bit16）：f16 输出要按 2B/元素分配缓冲，
            // 否则 shader 只写前半、返回的却是 f32 标签 → 下游全是垃圾（实测
            // batch32 训练 loss=NaN）。缓冲布局由 out_f16 决定，返回的 GpuTensor
            // 只是占位标签，这里按实际精度重贴 GpuTensorF16。
            const bool out_f16 = nn::expr_prec_sig_out_f16(fs->prec_sig);
            auto out = backend_.run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols,
                /*vector_out=*/false, vp, spec.rparams, /*output_override=*/nullptr,
                nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec),
                /*fold_k=*/std::nullopt, out_f16);
            if (!out) return std::unexpected(out.error());
            if (out_f16)
                return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), rows, cols));
            return Tensor::from_gpu(std::move(*out));
        }
#endif

        // f16 存储进了原生引擎却没命中带类型变体：**不能**把 f16 buffer 绑到
        // f32 shader 上（静默错值）→ 明确报错，引导调用方走 PrecisionEngine
        // 适配层（边界 cast 回退）。
        if (psig != 0)
            return std::unexpected(Error{
                "GpuEngine::eval_expr: 该 (结构,精度) 变体未预生成（in-kernel f16 覆盖不足）"
                "；请经 PrecisionEngine 适配层调用（边界 cast 回退）"});
        // ── 闭合世界：未命中任何 AOT 融合 shader → 硬报错（绝不静默回退） ──
        return std::unexpected(Error{
            "GpuEngine::eval_expr: 未找到该内联表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）"});
    }

    // ── 归约向量原生形状输出（M3：LayerNorm/RMSNorm 小向量缓存） ────────
    // ── 精度变体能力查询（Phase 2 in-kernel f16）──────────────────────────
    // 只有 (结构, 真实输入精度, 目标输出精度) 的带类型 shader 已注册时才为 true：
    // 适配层据此决定"直接吃 f16"还是"边界 cast 回退"。
    [[nodiscard]] bool supports_native_data_move() const noexcept override { return true; }

    [[nodiscard]] bool supports_expr_precision_variant(
        const ExprSpec& raw_spec, std::span<const Tensor> inputs,
        Precision P = Precision::F32) const override
    {
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        if (raw_spec.fold)
            return false;   // fold 段的带类型变体尚未实现（matmul 段已支持）
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
        if (psig == 0)
            return false;
        const std::string k = nn::expr_prec_sig_key(
            nn::expr_spec_key(nn::canonicalize_expr_spec(raw_spec)), psig);
        return nn::fused::find_fused(k) != nullptr && backend_.has_fused_shader(k);
#else
        (void)raw_spec; (void)inputs; (void)P;
        return false;
#endif
    }
    // 与 eval_expr 相同，但以 vector_out=1 调度归约融合 shader（thread 0 写
    // (rows,1)/(1,cols) 归约向量，不写全尺寸广播）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce(
        const ExprSpec& raw_spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32) override
    {
        // fold 形态只经 eval_expr（PC 需 fold_k，本入口不传）——显式拒绝，
        //   否则落到 run_fused_gpu 的通用缺参错误，误导排查方向
        if (raw_spec.fold)
            return std::unexpected(Error{
                "GpuEngine::eval_expr_reduce: fold 表达式请走 eval_expr（需 fold_k 形态参数）"});
        // canonical IR：与 eval_expr 同（canonicalize 为引擎内部优化）
        const ExprSpec spec = nn::canonicalize_expr_spec(raw_spec);

        // ── 限制：归约向量输出的末指令必须是归约指令 ──────────────────────
        // 归约融合 shader（generate_glsl_reduce）对"归约**之后**还有逐元素
        // 后处理"的形态（如 rsqrt(col_sum(x²)*k + c)）尚未正确实现：列归约
        // 分支会把后处理链在错误的索引上求值，实测给出**静默错值**
        // （ce_fusion_test 的"归约+后处理"用例：GPU err≈1.9，CPU err≈1e-7）。
        // 这里显式硬报错，把"静默错值"变成"立即暴露的错误"（闭合世界原则）。
        // 改用两步写法即可：先 compute_reduce 取归约向量，再对 (rows,1)/(1,cols)
        // 小向量用 dsl::compute 施加后处理（LayerNorm/RMSNorm 即此写法）。
        // 注意：CPU 端已支持该形态（eval_expr_impl 的按指令下标反向切片），
        // 修复生成器后可在此放开。
        if (!spec.instrs.empty() &&
            !expr_op_is_reduce(static_cast<ExprOp>(spec.instrs.back().op)))
        {
            return std::unexpected(Error{
                "GpuEngine::eval_expr_reduce: 归约向量输出的表达式末指令必须是归约指令"
                "（归约后逐元素后处理在归约融合 shader 中尚未正确实现）；"
                "请拆成两步：compute_reduce 取归约向量，再用 dsl::compute 做后处理"});
        }

        const std::string key = nn::expr_spec_key(spec);
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 精度变体优先（Phase 2 in-kernel f16）：与 eval_expr 同款 (key,sig) 匹配
        const std::string vkey = nn::expr_prec_sig_key(key, psig);
        const nn::fused::FusedShader* fs = (psig != 0) ? nn::fused::find_fused(vkey)
                                                      : nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            if (!fi_r) return std::unexpected(fi_r.error());
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            const bool out_f16 = nn::expr_prec_sig_out_f16(fs->prec_sig);
            auto out = backend_.run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/true, vp,
                spec.rparams,
                /*output_override=*/nullptr, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec), std::nullopt, out_f16);
            if (!out) return std::unexpected(out.error());
            if (out_f16)
            {
                // 归约向量原生形状：(rows,1)（行）/ (1,cols)（列）
                const int raxis = nn::expr_spec_reduce_axis(spec);
                const std::size_t orows = (raxis == 1) ? 1 : rows;
                const std::size_t ocols = (raxis == 1) ? cols : 1;
                return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), orows, ocols));
            }
            return Tensor::from_gpu(std::move(*out));
        }
#endif
        if (psig != 0)
            return std::unexpected(Error{
                "GpuEngine::eval_expr_reduce: 该 (结构,精度) 变体未预生成（in-kernel f16 覆盖不足）"
                "；请经 PrecisionEngine 适配层调用（边界 cast 回退）"});
        // ── 闭合世界：未命中归约融合 shader → 硬报错（绝不静默回退） ──
        return std::unexpected(Error{
            "GpuEngine::eval_expr_reduce: 未找到该归约表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）"});
    }

    // ── 目标传递（destination-passing）：结果直接写回已有张量 ─────────────
    // 走 run_fused_gpu 的 output_override —— dst 的 buffer 作为写only输出绑定，
    // 若表达式引用了 leaf(dst) 则同一 buffer 同时作为 readonly 输入绑定：
    // 逐元素同索引"先读后写"，无跨调用危害（与 GPU 原地原语的既有做法一致）。
    // 语义与限制同 CpuEngine::eval_expr_into（仅逐元素表达式；无分配）。
    [[nodiscard]] Result<void> eval_expr_into(
        const ExprSpec& raw_spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& dst) override
    {
        if (dst.rows() != rows || dst.cols() != cols)
            return std::unexpected(Error{"eval_expr_into: dst shape mismatch"});

        // fold 形态只经 eval_expr（PC 需 fold_k，本入口不传）——显式拒绝
        if (raw_spec.fold)
            return std::unexpected(Error{
                "GpuEngine::eval_expr_into: fold 表达式请走 eval_expr（需 fold_k 形态参数）"});
        const ExprSpec spec = nn::canonicalize_expr_spec(raw_spec);
        if (nn::expr_spec_reduce_axis(spec) != -1)
            return std::unexpected(Error{
                "eval_expr_into: 仅支持逐元素表达式（无归约）"});
        const std::string key = nn::expr_spec_key(spec);
        // 精度变体优先（Phase 2 in-kernel f16）：目标精度 = dst.precision()
        const nn::ExprPrecSig psig =
            nn::expr_prec_sig_of(inputs, dst.precision());
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const std::string vkey = nn::expr_prec_sig_key(key, psig);
        const nn::fused::FusedShader* fs = (psig != 0) ? nn::fused::find_fused(vkey)
                                                      : nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            if (!fi_r) return std::unexpected(fi_r.error());
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            auto dst_gpu = ensure_gpu(dst);
            if (!dst_gpu) return std::unexpected(dst_gpu.error());
            // ── output_override 必须按目标存储精度取视图（issue #13 P0-②）────
            // 曾写死默认 F32 的 gpu_tensor()：f16 目标 → gpu_get<F32>() 空
            // shared_ptr → Debug NN_ASSERT 引爆；Release 空解引用 = UB：
            // MSVC 得到空 override → 结果落临时 buffer 被丢弃 → **参数静默
            // 冻结（loss 恒 ln V）**；clang 早先"正常"只是 UB 代码生成运气。
            // 修复：f16 目标经 f16_view 包 f16 buffer（同 run_fused_gpu 的
            // out_f16 做法）；存储/标签不一致时显式报错，绝不静默。
            std::optional<GpuTensor> dst_view;
            GpuTensor* dst_override = nullptr;
            if (dst_gpu->precision() == Precision::F16)
            {
                if (!dst_gpu->gpu_shared<Precision::F16>())
                    return std::unexpected(Error{
                        "eval_expr_into: f16 目标缺少 f16 GPU 存储"
                        "（precision 标签与 variant 存储不一致）"});
                dst_view = f16_view(*dst_gpu);
                dst_override = &*dst_view;
            }
            else
            {
                if (!dst_gpu->gpu_shared<Precision::F32>())
                    return std::unexpected(Error{
                        "eval_expr_into: f32 目标缺少 f32 GPU 存储"
                        "（precision 标签与 variant 存储不一致）"});
                dst_override = &dst_gpu->gpu_tensor();
            }
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            auto out = backend_.run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/false, vp,
                spec.rparams, dst_override,
                nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec));
            if (!out) return std::unexpected(out.error());
            // dst 原为 CPU staging 时，ensure_gpu 上传了新 buffer（结果在它上面）
            // → 用 upload 后的张量替换 dst，保证调用方看到更新后的数据
            if (dst.is_cpu())
                dst = std::move(*dst_gpu);
            return {};
        }
#endif
        if (psig != 0)
            return std::unexpected(Error{
                "GpuEngine::eval_expr_into: 该 (结构,精度) 变体未预生成（in-kernel f16 覆盖不足）"
                "；请经 PrecisionEngine 适配层调用（边界 cast 回退）"});
        // ── 闭合世界：未命中 AOT 融合 shader → 硬报错（绝不静默回退） ──
        return std::unexpected(Error{
            "GpuEngine::eval_expr_into: 未找到该表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）"});
    }

private:
    // ── 辅助：f16 Tensor → GpuTensor **绑定视图** ─────────────────────────
    // 只借用底层 buffer + 形状；字节布局由 f16 存储版 pipeline 决定（调用方
    // 必须同时保证 f16_io 语义）。与 run_fused_gpu 的 out_f16 同一套做法
    // （GpuTensorT<F32> 包裹 f16 buffer），不做任何精度转换。
    [[nodiscard]] static GpuTensor f16_view(const Tensor& t)
    {
        const auto& g = t.gpu_tensor<Precision::F16>();
        return GpuTensor(g.shared_buffer(), g.rows(), g.cols());
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

