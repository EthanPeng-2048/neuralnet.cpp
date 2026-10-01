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
//   batch 录制模式：begin_batch 后所有原语录制到每帧共享的 command buffer，
//   end_batch 提交当前帧**不等待**（消除 per-primitive 的提交/阻塞开销，
//   host 继续录制、GPU 在队列上先行执行，同队列 FIFO 保证执行顺序）；
//   真正要读 GPU 结果（to_matrix / copy_from）前由 wait_in_flight 等待
//   所有在飞帧。to_matrix/from_matrix 会打断 batch（flush 后自动重新
//   begin_batch）。
//
// 原地操作语义：
//   add_inplace / scale_inplace 均直接写回
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
#include "expr_dsl.hpp"    // matmul_with_bias 经 DSL 融合（单一事实源）

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
    // begin_batch 后所有原语录制到当前帧的 batch_cmd_，end_batch 提交该帧
    // **不等待**（阻塞点在 wait_in_flight，见文件头「同步模型」）。
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

    // ── activation offload slab（持久复用缓冲） ───────────────────────
    [[nodiscard]] Result<Tensor> create_offload_buffer_impl(std::size_t bytes) override
    {
        auto g = GpuTensor::create_host_visible_empty(1, bytes, backend_);
        if (!g) return std::unexpected(g.error());
        return Tensor::from_gpu(std::move(*g));
    }

    // 把 src 复制到 buffer 的 offset（float 单位）处（录制式）
    [[nodiscard]] Result<void> offload_save_impl(
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
    [[nodiscard]] Result<Tensor> offload_restore_impl(
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

    // ── 异步标量回读───────────────────────────────────────────
    // 把 (1,1) F32 标量排入一次 D2H 拷贝并提交，不等待。调用时机：产出该
    // 标量的主帧（flush_batch/end_batch）已提交之后——同队列 FIFO 保证拷贝
    // 执行在生产命令之后。poll 非阻塞，就绪即取（见 compute_engine.hpp）。
    [[nodiscard]] Result<void> submit_scalar_readback_impl(
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

    [[nodiscard]] Tensor create_tensor_impl(std::size_t rows, std::size_t cols, Precision P) override
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

    [[nodiscard]] Result<Tensor> from_matrix_impl(const Matrix& m, Precision P) override
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
    [[nodiscard]] Result<Tensor> cast_impl(const Tensor& src, Precision dst) override
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

    [[nodiscard]] Result<void> copy_from_impl(Tensor& dst, const Matrix& src) override
    {
        if (dst.rows() != src.rows() || dst.cols() != src.cols())
            return std::unexpected(Error{"copy_from: shape mismatch"});
        if (dst.is_cpu())
        {
            // 防御性：dst 应为 GPU Tensor，但若为 CPU 则直接拷贝
            dst = Tensor::from_matrix(Matrix(src));
            return {};
        }
        // batch 模式下必须先 drain（提交不等待）：dst 是当前帧已引用的既有
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

    [[nodiscard]] Result<Tensor> clone_impl(const Tensor& src) override
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
    [[nodiscard]] Result<Tensor> slice_rows_impl(
        const Tensor& src, std::size_t start_row, std::size_t count) override
    {
        auto src_gpu = import(src);
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
    // 纯 GPU 架构：dst 必须为 GPU Tensor，传 CPU Tensor 直接报错（无 CPU 回退路径）。
    [[nodiscard]] Result<void> insert_rows_impl(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (dst.is_cpu())
            return std::unexpected(Error{"insert_rows: dst must be GPU tensor in pure-GPU architecture"});
        auto src_gpu = import(src);
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
    [[nodiscard]] Result<Tensor> gather_rows_impl(
        const Tensor& table, const Tensor& indices) override
    {
        auto tbl_gpu = import(table);
        if (!tbl_gpu) return std::unexpected(tbl_gpu.error());
        auto idx_gpu = import(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());

        // Phase C1 f16 直读直写（table/out f16；indices 恒 f32——行号整数值
        // 精度要求）。无 f16 pipeline → 引擎内边界 cast 回退。
        if (tbl_gpu->precision() == Precision::F16)
        {
            if (backend_.has_gather_f16_pipeline() &&
                idx_gpu->precision() == Precision::F32)
            {
                auto r = backend_.gather_gpu(f16_view(*tbl_gpu),
                                             idx_gpu->gpu_tensor(),
                                             /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto t32 = cast(*tbl_gpu, Precision::F32);
            if (!t32) return std::unexpected(t32.error());
            std::optional<Tensor> i32;
            if (idx_gpu->precision() != Precision::F32)
            {
                auto c = cast(*idx_gpu, Precision::F32);
                if (!c) return std::unexpected(c.error());
                i32 = std::move(*c);
            }
            const Tensor& idx_ref = i32 ? *i32 : *idx_gpu;
            auto r = backend_.gather_gpu(t32->gpu_tensor(), idx_ref.gpu_tensor());
            if (!r) return std::unexpected(r.error());
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(out32, Precision::F16);
        }
        auto r = backend_.gather_gpu(tbl_gpu->gpu_tensor(), idx_gpu->gpu_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst ──
    // GPU-native 实现：使用 CAS 循环实现 float atomicAdd，无 PCIe 传输。
    // Phase C2b：dst/grad 均 f16 且 f16 pipeline 可用 → 打包 half CAS 原生
    // （dst 每 word 2 个 half、grad half 直读；indices 恒 float），免去为
    // dst/grad 物化 f32 副本。任一非 f16 或无 pipeline → 引擎内边界 cast
    // 回退（正确性不变）。
    [[nodiscard]] Result<void> scatter_add_rows_impl(
        Tensor& dst, const Tensor& indices, const Tensor& grad) override
    {
        if (!dst.is_gpu())
            return std::unexpected(Error{"scatter_add_rows: dst must be GPU tensor"});

        auto idx_gpu = import(indices);
        if (!idx_gpu) return std::unexpected(idx_gpu.error());
        auto grad_gpu = import(grad);
        if (!grad_gpu) return std::unexpected(grad_gpu.error());

        const bool f16_io = dst.precision() == Precision::F16 &&
                            grad_gpu->precision() == Precision::F16 &&
                            backend_.has_scatter_add_f16_pipeline();
        if (f16_io)
        {
            GpuTensor dst_ref = f16_view(dst);       // 局部保活（禁 &临时）
            const GpuTensor grad_ref = f16_view(*grad_gpu);
            return backend_.scatter_add_gpu(dst_ref, idx_gpu->gpu_tensor(),
                                            grad_ref, /*f16_io=*/true);
        }
        if (dst.precision() == Precision::F16 || grad_gpu->precision() == Precision::F16)
        {
            // 引擎内 cast 回退：dst 若 f16 → cast_into 写回原存储（§8.3）
            auto d32 = (dst.precision() == Precision::F16)
                ? cast(dst, Precision::F32) : Result<Tensor>(dst);
            if (!d32) return std::unexpected(d32.error());
            auto g32 = (grad_gpu->precision() == Precision::F16)
                ? cast(*grad_gpu, Precision::F32) : Result<Tensor>(*grad_gpu);
            if (!g32) return std::unexpected(g32.error());
            auto g32_gpu = import(*g32);
            if (!g32_gpu) return std::unexpected(g32_gpu.error());
            auto r = backend_.scatter_add_gpu(d32->gpu_tensor(), idx_gpu->gpu_tensor(),
                                              g32_gpu->gpu_tensor());
            if (!r) return std::unexpected(r.error());
            if (dst.precision() == Precision::F16)
                return cast_into(*d32, dst);
            return {};
        }
        return backend_.scatter_add_gpu(
            dst.gpu_tensor(), idx_gpu->gpu_tensor(), grad_gpu->gpu_tensor());
    }

private:
    // batch 模式查询（内部使用，to_matrix/from_matrix 需检查）
    [[nodiscard]] bool in_batch() const noexcept { return backend_.in_batch(); }

public:

    // ── 3D 维度转置：(M, B, N) ↔ (B, M, N) ──
    [[nodiscard]] Result<Tensor> rearrange_3d_impl(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse) override
    {
        auto x_gpu = import(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());

        // Phase C1 f16 直读直写；无 f16 pipeline → 引擎内边界 cast 回退
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_.has_rearrange_3d_f16_pipeline())
            {
                auto r = backend_.rearrange_3d_gpu(
                    f16_view(*x_gpu), static_cast<uint32_t>(M),
                    static_cast<uint32_t>(B), static_cast<uint32_t>(N),
                    inverse ? 1u : 0u, /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            if (!x32) return std::unexpected(x32.error());
            auto r = backend_.rearrange_3d_gpu(
                x32->gpu_tensor(), static_cast<uint32_t>(M),
                static_cast<uint32_t>(B), static_cast<uint32_t>(N),
                inverse ? 1u : 0u);
            if (!r) return std::unexpected(r.error());
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(out32, Precision::F16);
        }
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
    [[nodiscard]] Result<Tensor> transpose_impl(const Tensor& A) override
    {
        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        // Phase C1 f16 直读直写；无 f16 pipeline（设备能力不足）→ 引擎内
        // 边界 cast 回退（f16→f32→算→f16），正确性不变——绝不把 f16 buffer
        // 绑到 f32 pipeline（静默错值）。
        if (a_gpu->precision() == Precision::F16)
        {
            if (backend_.has_transpose_f16_pipeline())
            {
                auto r = backend_.transpose_gpu(f16_view(*a_gpu), /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            auto r32 = backend_.transpose_gpu(a32->gpu_tensor());
            if (!r32) return std::unexpected(r32.error());
            Tensor out32 = Tensor::from_gpu(std::move(*r32));
            return cast(out32, Precision::F16);
        }
        auto r = backend_.transpose_gpu(a_gpu->gpu_tensor());
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 卷积/池化窗口展开（纯数据搬运，无算法；契约见 compute_engine.hpp）──
    [[nodiscard]] Result<Tensor> im2col_impl(
        const Tensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        auto x_gpu = import(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        // Phase C2d：f16 输入/输出 → f16 窗口变体（in/out 均 half，语义
        // "输出精度 = 源精度" §8.4）；无 pipeline → 引擎内边界 cast 回退
        //（绝不把 f16 buffer 绑到 f32 pipeline；CNN 路径与 move_ 直通保障）
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_.has_im2col_f16_pipeline())
            {
                auto r = backend_.im2col_gpu(f16_view(*x_gpu),
                                             C, H, W, k, stride, pad, OH, OW,
                                             /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            if (!x32) return std::unexpected(x32.error());
            auto r = backend_.im2col_gpu(x32->gpu_tensor(),
                                         C, H, W, k, stride, pad, OH, OW);
            if (!r) return std::unexpected(r.error());
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(out32, Precision::F16);
        }
        auto r = backend_.im2col_gpu(x_gpu->gpu_tensor(),
                                     C, H, W, k, stride, pad, OH, OW);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col2im_impl(
        const Tensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        auto c_gpu = import(col);
        if (!c_gpu) return std::unexpected(c_gpu.error());
        // Phase C2d：同 im2col——f16 窗口变体优先，无 pipeline 引擎内回退
        if (c_gpu->precision() == Precision::F16)
        {
            if (backend_.has_col2im_f16_pipeline())
            {
                auto r = backend_.col2im_gpu(f16_view(*c_gpu),
                                             C, H, W, k, stride, pad, OH, OW,
                                             /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(
                    GpuTensorF16(r->shared_buffer(), r->rows(), r->cols()));
            }
            auto c32 = cast(*c_gpu, Precision::F32);
            if (!c32) return std::unexpected(c32.error());
            auto r = backend_.col2im_gpu(c32->gpu_tensor(),
                                         C, H, W, k, stride, pad, OH, OW);
            if (!r) return std::unexpected(r.error());
            Tensor out32 = Tensor::from_gpu(std::move(*r));
            return cast(out32, Precision::F16);
        }
        auto r = backend_.col2im_gpu(c_gpu->gpu_tensor(),
                                     C, H, W, k, stride, pad, OH, OW);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    // ── 分组归约（契约见 compute_engine.hpp）──────────────────────────
    // Phase C2c：f16 输入 → f16 分组归约 pipeline（half 直读、f32 归约、
    // 输出 f32 (G,N)）；无 pipeline → 引擎内边界 cast 回退。
    [[nodiscard]] Result<Tensor> grouped_reduce_f16_or_cast_(
        const Tensor& x, std::size_t G, std::size_t R, bool is_max)
    {
        auto x_gpu = import(x);
        if (!x_gpu) return std::unexpected(x_gpu.error());
        if (x_gpu->precision() == Precision::F16)
        {
            if (backend_.has_group_reduce_f16_pipeline())
            {
                auto r = backend_.grouped_reduce_gpu(
                    f16_view(*x_gpu), G, R, is_max, /*f16_in=*/true);
                if (!r) return std::unexpected(r.error());
                return Tensor::from_gpu(std::move(*r));
            }
            auto x32 = cast(*x_gpu, Precision::F32);
            if (!x32) return std::unexpected(x32.error());
            auto x32_gpu = import(*x32);
            if (!x32_gpu) return std::unexpected(x32_gpu.error());
            auto r = backend_.grouped_reduce_gpu(x32_gpu->gpu_tensor(), G, R, is_max);
            if (!r) return std::unexpected(r.error());
            return Tensor::from_gpu(std::move(*r));
        }
        auto r = backend_.grouped_reduce_gpu(x_gpu->gpu_tensor(), G, R, is_max);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> grouped_reduce_sum_impl(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision = Precision::F32) override
    {
        return grouped_reduce_f16_or_cast_(x, G, R, /*is_max=*/false);
    }

    [[nodiscard]] Result<Tensor> grouped_reduce_max_impl(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision = Precision::F32) override
    {
        return grouped_reduce_f16_or_cast_(x, G, R, /*is_max=*/true);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> matmul_impl(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB,
        Precision P = Precision::F32) override
    {
        // 确保 A、B 在 GPU 上
        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = import(B);
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
        // （f16 直读 + f32 累加 + f16 写出），省掉"每个操作数一份整份 f32
        // 副本"的边界 cast——f32 副本正是训练 transient 膨胀的主因。小 N
        // （n_out≤8）走 f16 GEMV（C2e），大 N 走 f16 tiled；均不可用才回退
        // 边界 cast。
        if (P == Precision::F16)
        {
            const std::size_t n_out = transB ? B.rows() : B.cols();
            // 大 N：tiled f16；小 N（n_out≤8，GEMV 热路径）：gemv f16
            //（C2e：变体 pipeline 缺失时 f16_ok 为假 → 落到下方边界 cast）
            const bool f16_ok =
                a_gpu->precision() == Precision::F16 &&
                b_gpu->precision() == Precision::F16 &&
                ((n_out > 8 && backend_.has_matmul_tiled_f16_pipeline()) ||
                 (n_out <= 8 && backend_.has_gemv_f16_pipeline()));
            if (f16_ok)
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

            auto a32_gpu = import(*a32);
            if (!a32_gpu) return std::unexpected(a32_gpu.error());
            auto b32_gpu = import(*b32);
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

    // ── matmul + 行广播 bias：经 DSL 融合（单一事实源）──────────────
    // 走 eval_expr → AOT fusion shader 精确匹配（闭合世界）。同时让
    // scan_exprs dry-run 收集该"Linear 结构"spec，gen_fused 生成融合 kernel，
    // 使 GPU Linear::forward 全程在 GPU（基类 matmul_with_bias 的默认实现
    // 是 to_matrix/from_matrix 的 CPU 往返路径）。
    //
    // 多精度（Phase 2 in-kernel f16）：P 必须下传——f16 时**先试带类型变体**
    // （f16 直读 / f32 累加 / f16 写回，零边界 cast：Linear::forward 是 cast
    // 临时量最大的单一来源），无变体才在本函数内回退"抬 f32 → f32 融合 →
    // 落回"（**绝不把缺变体变成硬报错**，闭合世界原则只在 f32 结构缺失时适用）。
    [[nodiscard]] Result<Tensor> matmul_with_bias_impl(
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
    // P: 计算精度（§8.1，同 matmul）
    [[nodiscard]] Result<Tensor> batched_matmul_impl(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha,
        Precision P = Precision::F32) override
    {
        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = import(B);
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

            auto a32_gpu = import(*a32);
            if (!a32_gpu) return std::unexpected(a32_gpu.error());
            auto b32_gpu = import(*b32);
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
    [[nodiscard]] Result<void> add_inplace_impl(Tensor& A, const Tensor& B) override
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return std::unexpected(Error{"add_inplace: shape mismatch"});

        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        auto b_gpu = import(B);
        if (!b_gpu) return std::unexpected(b_gpu.error());

        // Phase C1b：目标 A 是 f16 → f16 pipeline 原地直加（A 自身永不物化
        // f32 副本，A 的 f32 副本正是 in-place 路径的瞬态开销大头）。
        //   · B 同 f16 → 零副本直加；
        //   · B 是 f32（梯度混合精度）→ 只物化 B 的 f16 小副本再加；
        //   · 无 f16 pipeline → 引擎内回退（A 抬 f32 加完写回，正确性不变）。
        if (a_gpu->precision() == Precision::F16 &&
            backend_.has_elementwise_v2_f16_pipeline())
        {
            std::optional<Tensor> b16;
            if (b_gpu->precision() != Precision::F16)
            {
                auto c = cast(*b_gpu, Precision::F16);
                if (!c) return std::unexpected(c.error());
                b16 = std::move(*c);
            }
            const GpuTensor& b_ref = b16 ? f16_view(*b16) : f16_view(*b_gpu);
            const GpuTensor a_ref = f16_view(*a_gpu);   // 局部保活（禁 &临时）
            const uint32_t count16 = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_.elementwise_v2_gpu(
                a_ref, &b_ref, nullptr,
                count16, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
                &a_ref, /*f16_io=*/true);  // BINARY, Add, 原地写回 A
            if (!r)
                return std::unexpected(r.error());
            if (A.is_cpu())
                A = std::move(*a_gpu);
            return {};
        }
        if (a_gpu->precision() == Precision::F16)
        {
            // 无 f16 pipeline：A 抬 f32 → 加 → 写回 A（边界 cast 路径内聚在引擎）
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            std::optional<Tensor> b32;
            if (b_gpu->precision() != Precision::F32)
            {
                auto c = cast(*b_gpu, Precision::F32);
                if (!c) return std::unexpected(c.error());
                b32 = std::move(*c);
            }
            const GpuTensor& b_ref2 = b32 ? b32->gpu_tensor() : b_gpu->gpu_tensor();
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_.elementwise_v2_gpu(
                a32->gpu_tensor(), &b_ref2, nullptr,
                cnt, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
                &a32->gpu_tensor());
            if (!r)
                return std::unexpected(r.error());
            return cast_into(*a32, A);   // f32 结果写回 A 的 f16 原存储
        }

        // A 是 f32：B 若为 f16（直通后可能出现）→ 先抬 f32 再加（B 的边界
        // cast 只在引擎内发生，A 永不物化）
        if (b_gpu->precision() == Precision::F16)
        {
            auto b32 = cast(*b_gpu, Precision::F32);
            if (!b32) return std::unexpected(b32.error());
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            auto r = backend_.elementwise_v2_gpu(
                a_gpu->gpu_tensor(), &b32->gpu_tensor(), nullptr,
                cnt, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
                &a_gpu->gpu_tensor());
            if (!r) return std::unexpected(r.error());
            if (A.is_cpu()) A = std::move(*a_gpu);
            return {};
        }

        const uint32_t count = static_cast<uint32_t>(A.rows() * A.cols());
        auto r = backend_.elementwise_v2_gpu(
            a_gpu->gpu_tensor(), &b_gpu->gpu_tensor(), nullptr,
            count, 1u, 0u, 0u, 0u, 0.0f, 0.0f, 0.0f,
            &a_gpu->gpu_tensor());  // BINARY, Add, 原地写回 A
        if (!r)
            return std::unexpected(r.error());
        // 原地模式下 A 的 buffer 已被更新；若 import 上传了新 Tensor（防御路径），替换 A
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // A *= s：真原地，直接写回 A 的 buffer（与 CpuEngine 语义一致）
    [[nodiscard]] Result<void> scale_inplace_impl(Tensor& A, Scalar s) override
    {
        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());

        // Phase C1b：f16 → f16 pipeline 原地缩放（A 永不物化 f32 副本）；
        // 无 pipeline → 引擎内回退（抬 f32 → 缩放 → 写回）
        if (a_gpu->precision() == Precision::F16)
        {
            const uint32_t cnt = static_cast<uint32_t>(A.rows() * A.cols());
            if (backend_.has_elementwise_v2_f16_pipeline())
            {
                const GpuTensor a_ref = f16_view(*a_gpu);   // 局部保活（禁 &临时）
                auto r = backend_.elementwise_v2_gpu(
                    a_ref, nullptr, nullptr,
                    cnt, 1u, 2u, 0u, 1u, static_cast<float>(s), 0.0f, 0.0f,
                    &a_ref, /*f16_io=*/true);
                if (!r) return std::unexpected(r.error());
                if (A.is_cpu()) A = std::move(*a_gpu);
                return {};
            }
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            auto r = backend_.elementwise_v2_gpu(
                a32->gpu_tensor(), nullptr, nullptr,
                cnt, 1u, 2u, 0u, 1u, static_cast<float>(s), 0.0f, 0.0f,
                &a32->gpu_tensor());
            if (!r) return std::unexpected(r.error());
            return cast_into(*a32, A);
        }

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

    // A = 0：使用 vkCmdFillBuffer 真原地清零（不分配新 buffer）
    [[nodiscard]] Result<void> zero_impl(Tensor& A) override
    {
        auto a_gpu = import(A);
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
        // 但若 import 上传了新 Tensor，需要替换 A
        if (A.is_cpu())
            A = std::move(*a_gpu);
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 扫描级原语（RLA）——手写原语 shader + GpuBackend 接线
    // 形状契约见 compute_engine.hpp 扫描级原语注释；纯 GPU 架构：
    // 无 pipeline 时硬报错，不做 CPU 回退。
    // ══════════════════════════════════════════════════════════════════════

    // Phase C2f 辅助：f16 张量 → f32 副本（引擎内边界 cast 回退用）；
    // 已是 f32 → 共享所有权直通（零拷贝）。
    [[nodiscard]] Result<Tensor> cast_if_f16(const Tensor& t)
    {
        if (t.precision() != Precision::F16)
            return t;
        return cast(t, Precision::F32);
    }

    // Phase C2f：全 f16 输入 + f16 pipeline 可用 → 原生扫描（输入输出均
    // half，消灭适配层全尺寸 f32 副本——输出 (rows*5/3/dk, seq) 是大头）；
    // 混合精度（如 boundary 仍 f32）→ 引擎内把 f16 侧 cast 到 f32 走 f32
    // pipeline（正确性不变）。输出重贴 GpuTensorF16（缓冲按 2B/元素分配）。
    [[nodiscard]] Result<Tensor> scan_prefix_outer_impl(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32) override
    {
        auto k = import(K); if (!k) return std::unexpected(k.error());
        auto v = import(V); if (!v) return std::unexpected(v.error());
        auto p = import(P); if (!p) return std::unexpected(p.error());
        auto r = import(R); if (!r) return std::unexpected(r.error());
        auto a = import(A0); if (!a) return std::unexpected(a.error());
        auto b = import(B0); if (!b) return std::unexpected(b.error());
        auto bn = import(boundary); if (!bn) return std::unexpected(bn.error());
        const bool all_f16 = k->precision() == Precision::F16 &&
            v->precision() == Precision::F16 && p->precision() == Precision::F16 &&
            r->precision() == Precision::F16 && a->precision() == Precision::F16 &&
            b->precision() == Precision::F16 &&
            bn->precision() == Precision::F16;
        // 原生 f16 仅在请求输出也是 f16 时启用（输出契约：prec=F32 必须回
        // f32——走下方引擎内 cast 路径，输出 f32 恰好免适配层落回）
        if (prec == Precision::F16 && all_f16 && backend_.has_scan_f16_pipelines())
        {
            auto res = backend_.scan_prefix_outer_gpu(
                f16_view(*k), f16_view(*v), f16_view(*p), f16_view(*r),
                f16_view(*a), f16_view(*b), has_state,
                static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
                f16_view(*bn), has_bnd, /*f16_io=*/true);
            if (!res) return std::unexpected(res.error());
            return Tensor::from_gpu(
                GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        // 引擎内 cast 回退（混合精度或无 f16 pipeline）：f16 侧抬 f32
        auto k32 = cast_if_f16(*k); if (!k32) return std::unexpected(k32.error());
        auto v32 = cast_if_f16(*v); if (!v32) return std::unexpected(v32.error());
        auto p32 = cast_if_f16(*p); if (!p32) return std::unexpected(p32.error());
        auto r32 = cast_if_f16(*r); if (!r32) return std::unexpected(r32.error());
        auto a32 = cast_if_f16(*a); if (!a32) return std::unexpected(a32.error());
        auto b32 = cast_if_f16(*b); if (!b32) return std::unexpected(b32.error());
        auto n32 = cast_if_f16(*bn); if (!n32) return std::unexpected(n32.error());
        auto res = backend_.scan_prefix_outer_gpu(
            k32->gpu_tensor(), v32->gpu_tensor(), p32->gpu_tensor(), r32->gpu_tensor(),
            a32->gpu_tensor(), b32->gpu_tensor(), has_state,
            static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
            n32->gpu_tensor(), has_bnd);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    [[nodiscard]] Result<Tensor> scan_suffix_outer_impl(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32) override
    {
        auto d = import(D); if (!d) return std::unexpected(d.error());
        auto x = import(X); if (!x) return std::unexpected(x.error());
        auto y = import(Y); if (!y) return std::unexpected(y.error());
        auto bn = import(boundary); if (!bn) return std::unexpected(bn.error());
        const bool all_f16 = d->precision() == Precision::F16 &&
            x->precision() == Precision::F16 && y->precision() == Precision::F16 &&
            bn->precision() == Precision::F16;
        if (prec == Precision::F16 && all_f16 && backend_.has_scan_f16_pipelines())
        {
            auto res = backend_.scan_suffix_outer_gpu(
                f16_view(*d), f16_view(*x), f16_view(*y),
                static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
                f16_view(*bn), has_bnd, /*f16_io=*/true);
            if (!res) return std::unexpected(res.error());
            return Tensor::from_gpu(
                GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        auto d32 = cast_if_f16(*d); if (!d32) return std::unexpected(d32.error());
        auto x32 = cast_if_f16(*x); if (!x32) return std::unexpected(x32.error());
        auto y32 = cast_if_f16(*y); if (!y32) return std::unexpected(y32.error());
        auto n32 = cast_if_f16(*bn); if (!n32) return std::unexpected(n32.error());
        auto res = backend_.scan_suffix_outer_gpu(
            d32->gpu_tensor(), x32->gpu_tensor(), y32->gpu_tensor(),
            static_cast<uint32_t>(dk), static_cast<uint32_t>(heads), causal,
            n32->gpu_tensor(), has_bnd);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    [[nodiscard]] Result<Tensor> outer_col_impl(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale,
        Precision prec = Precision::F32) override
    {
        auto p = import(P); if (!p) return std::unexpected(p.error());
        auto r = import(R); if (!r) return std::unexpected(r.error());
        auto s = import(S); if (!s) return std::unexpected(s.error());
        const bool all_f16 = p->precision() == Precision::F16 &&
            r->precision() == Precision::F16 && s->precision() == Precision::F16;
        if (prec == Precision::F16 && all_f16 &&
            backend_.has_outer_col_f16_pipeline())
        {
            auto res = backend_.outer_col_gpu(
                f16_view(*p), f16_view(*r), f16_view(*s),
                static_cast<uint32_t>(dk), has_scale, /*f16_io=*/true);
            if (!res) return std::unexpected(res.error());
            return Tensor::from_gpu(
                GpuTensorF16(res->shared_buffer(), res->rows(), res->cols()));
        }
        auto p32 = cast_if_f16(*p); if (!p32) return std::unexpected(p32.error());
        auto r32 = cast_if_f16(*r); if (!r32) return std::unexpected(r32.error());
        auto s32 = cast_if_f16(*s); if (!s32) return std::unexpected(s32.error());
        auto res = backend_.outer_col_gpu(
            p32->gpu_tensor(), r32->gpu_tensor(), s32->gpu_tensor(),
            static_cast<uint32_t>(dk), has_scale);
        if (!res) return std::unexpected(res.error());
        return Tensor::from_gpu(std::move(*res));
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════

    // Phase C2：f16 输入 → f16 归约 pipeline（输入 float16_t 直读、f32 归约、
    // 输出 f32 向量）——省掉"每个 f16 输入一份整份 f32 副本"的边界 cast。
    // 设备无 16bit 存储时 pipeline 未创建 → 引擎内 cast 回退（正确性不变）。
    [[nodiscard]] Result<GpuTensor> reduce_f16_or_cast_(
        const Tensor& A, uint32_t mode, uint32_t reduce_op)
    {
        auto a_gpu = import(A);
        if (!a_gpu) return std::unexpected(a_gpu.error());
        if (a_gpu->precision() == Precision::F16)
        {
            // f16 张量必须经 f16_view（gpu_tensor() 默认 P=F32 → 读 f16 buffer
            // 的 f32 view = 断言失败 [gpu_tensor-fail]）
            if (backend_.has_reduce_f16_pipeline())
                return backend_.reduce_gpu(f16_view(*a_gpu), mode, reduce_op,
                                           /*f16_in=*/true);
            // 引擎内边界 cast（无 f16 pipeline 的设备）
            auto a32 = cast(*a_gpu, Precision::F32);
            if (!a32) return std::unexpected(a32.error());
            auto a32_gpu = import(*a32);
            if (!a32_gpu) return std::unexpected(a32_gpu.error());
            return backend_.reduce_gpu(a32_gpu->gpu_tensor(), mode, reduce_op);
        }
        return backend_.reduce_gpu(a_gpu->gpu_tensor(), mode, reduce_op);
    }

    [[nodiscard]] Result<Tensor> row_reduce_sum_impl(const Tensor& A, Precision = Precision::F32) override
    {
        auto r = reduce_f16_or_cast_(A, /*mode=*/0u, /*reduce_op=*/0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_sum_impl(const Tensor& A, Precision = Precision::F32) override
    {
        auto r = reduce_f16_or_cast_(A, /*mode=*/1u, /*reduce_op=*/0u);
        if (!r) return std::unexpected(r.error());
        return Tensor::from_gpu(std::move(*r));
    }

    [[nodiscard]] Result<Tensor> col_reduce_max_impl(const Tensor& A, Precision = Precision::F32) override
    {
        auto r = reduce_f16_or_cast_(A, /*mode=*/1u, /*reduce_op=*/1u);
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
            auto g = import(t);
            if (!g) return std::unexpected(g.error());
            if (g->precision() == Precision::F16)
                out.bufs.push_back(&g->gpu_tensor<Precision::F16>().buffer());
            else
                out.bufs.push_back(&g->gpu_tensor().buffer());
            out.owners.push_back(std::move(*g));   // 保活（见上）
        }
        return out;
    }

#ifdef NN_FUSED_REGISTRY_EMBEDDED
    // ── 变体选择：native16（`key#a`）优先，运行期精度分派（`key#x`）回退 ──
    //   · psig == 0 → 全 f32 结构键 `key`（V0）。
    //   · 结构谓词 expr_prec_sig_native16 通过 → `key#a`（V2b 原生 f16 算术）。
    //     **必须在运行期用真实 spec 重跑谓词**：常量值已不进 key（同结构可带
    //     不同常量），生成期是对"生成那一份时的常量"判定，运行期实例的常量
    //     可能不满足谓词（如 0 < |c| < 6e-8 的 eps 在 f16 下 flush-to-0）。
    //     生成期判定与运行期判定不一致 = 静默错值。
    //   · 其余 → `key#x`（V1 运行期精度分派）：一份 shader 覆盖任意签名。
    //     ⚠ "任何非零签名都必定命中"只在**生成端产出了 `#x`** 且设备启用了
    //     16bit 存储时成立；生成被跳过（`[skip]`）/ `NN_SCAN_NO_DISPATCH=1` /
    //     设备无 16bit storage 时仍会 miss → 基类 NVI 入口回退边界 cast。
    // registry 与 pipeline **都**命中才选（设备无 shaderFloat16 时 `#a` 的
    // pipeline 不创建 → 自动落 `#x`）。
    //
    // 诊断（NN_PREC_TRACE=1）：记**实际选中的变体**（V1/V2），不记 V0 默认
    // 路径（f32 热路径，逐调用打日志会淹没输出）。两个约束：
    //   ① 只在**确认命中**后打（此前 V1 分支先打后查，miss 也报 alu-hit）；
    //   ② 只由真正 dispatch 的调用方传 log_hit=true ——
    //      `supports_expr_precision_variant` 只是查询（NVI 入口每次都调），
    //      它打日志会让计数翻倍（实测 12040 行 = 6020 次真实分派 × 2）。
    [[nodiscard]] const nn::fused::FusedShader* find_prec_variant_(
        const std::string& key, nn::ExprPrecSig psig, const nn::ExprSpec& spec,
        bool log_hit = false) const
    {
        if (psig == 0)
            return nn::fused::find_fused(key);      // V0：全 f32 结构键
        const bool trace = log_hit && nn::dsl::env_flag("NN_PREC_TRACE");
        const auto hit = [&](const std::string& k) -> const nn::fused::FusedShader* {
            const auto* fs = nn::fused::find_fused(k);
            if (!fs || !backend_.has_fused_shader(k))
                return nullptr;
            if (trace)
                std::fprintf(stderr, "[prec][alu-hit] %s\n", k.c_str());
            return fs;
        };
        // V2b：native16 原生 f16 算术（运行期用真实 spec 复核结构谓词）
        if (nn::expr_prec_sig_native16(spec, psig))
            if (const auto* fs = hit(key + nn::EXPR_PREC_ALU_SUFFIX))
                return fs;
        // V1：运行期精度分派
        const std::string xkey = key + nn::EXPR_PREC_DISPATCH_SUFFIX;
        if (const auto* fs = hit(xkey))
            return fs;
        if (trace)
            std::fprintf(stderr,
                         "[prec][fallback] %s 未预生成 → 回退基类边界 cast\n",
                         xkey.c_str());
        return nullptr;
    }
#endif
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> eval_expr_impl(
        const ExprSpec& raw_spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32) override
    {
        // ── fold 段：canonicalize 对 fold 恒等（expr_opt 入口
        //    early-return），scan/runtime 两端 key 同源 → 直接用 raw_spec 查表。
        //    PC/分派形态由 run_fused_gpu 的 fold_k 参数走（见其形态校验）。
        if (raw_spec.fold)
        {
            const std::string fkey = nn::expr_spec_key(raw_spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
            // fold 变体：native16 谓词排除 fold，find_prec_variant_ 的 `#a`
            // 探测必然落空 → 命中 `#x`（或有真实位图的存储变体）
            const nn::ExprPrecSig fpsig = nn::expr_prec_sig_of(inputs, P);
            const nn::fused::FusedShader* ffs =
                find_prec_variant_(fkey, fpsig, raw_spec, /*log_hit=*/true);
            if (ffs && backend_.has_fused_shader(ffs->key))
            {
                auto fi_r = fused_buffers_(inputs);
                if (!fi_r) return std::unexpected(fi_r.error());
                std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;
                const auto fvp = nn::expr_spec_runtime_view_params(raw_spec);
                const bool fout_f16 = nn::expr_variant_out_f16(ffs->prec_sig, fpsig);
                auto out = backend_.run_fused_gpu(
                    ffs->key, gpu_inputs, raw_spec.consts, rows, cols,
                    /*vector_out=*/false, fvp, raw_spec.rparams,
                    /*output_override=*/nullptr,
                    // 双域 fold 自带 matmul 段的 k/batch（7 槽 PC 的 slot5/6）；
                    // fold 的转置仍在 key 里（生成期定死）→ 此处恒 0（无该槽）
                    nn::expr_spec_runtime_matmul_k(raw_spec),
                    nn::expr_spec_runtime_matmul_batch(raw_spec),
                    /*matmul_trans=*/0u,
                    nn::expr_spec_runtime_fold_k(raw_spec), fout_f16, fpsig);
                if (!out) return std::unexpected(out.error());
                // f16 输出：按 2B/元素分配后重贴 GpuTensorF16（同非 fold 路径）
                if (fout_f16)
                    return Tensor::from_gpu(
                        GpuTensorF16(out->shared_buffer(), out->rows(), out->cols()));
                return Tensor::from_gpu(std::move(*out));
            }
#endif
            return std::unexpected(Error{
                "GpuEngine::eval_expr: fold 表达式未命中 AOT 融合 shader"
                "（闭合世界）；请将该 fold 结构纳入 scan_exprs；key=" + fkey});
        }
        // ── canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        //    key 与 shader 合成两端一致；dispatch 用 canonical 的 consts ──
        const ExprSpec spec = nn::canonicalize_expr_spec(raw_spec);
        // ── AOT 匹配：按规范结构 key 查预编译融合 shader ──────────────
        const std::string key = nn::expr_spec_key(spec);
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 变体优先（native16 `#a` → 运行期精度分派 `#x`）：按**真实输入精度**
        // + 目标输出精度取变体（find_prec_variant_）。命中即 shader 直接半精度
        // 读/写，无需边界 cast；未命中则退回全 f32 key —— 此时输入必须已全 f32
        // （由基类 NVI 入口 cast）。
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
        const nn::fused::FusedShader* fs =
            find_prec_variant_(key, psig, spec, /*log_hit=*/true);
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
            // 输出存储精度：分派变体按**运行期真实签名**（bit16），其余按变体身份
            const bool out_f16 = nn::expr_variant_out_f16(fs->prec_sig, psig);
            auto out = backend_.run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols,
                /*vector_out=*/false, vp, spec.rparams, /*output_override=*/nullptr,
                nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec),
                nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, out_f16, psig);
            if (!out) return std::unexpected(out.error());
            if (out_f16)
                return Tensor::from_gpu(GpuTensorF16(out->shared_buffer(), rows, cols));
            return Tensor::from_gpu(std::move(*out));
        }
#endif

        // f16 存储进了原生引擎却没命中带类型变体：**不能**把 f16 buffer 绑到
        // f32 shader 上（静默错值）→ 明确报错，明确报错（NVI 入口本应已拦截——能力查询
        // 与 shader 查询不一致时才会到这里）。
        if (psig != 0)
            return std::unexpected(Error{
                "GpuEngine::eval_expr: 该 (结构,精度) 变体未预生成（in-kernel f16 覆盖不足）"
                "；NVI 入口本应在此前抬到 f32（supports_expr_precision_variant 与 shader 查询不一致）"
                "；key=" + key});
        // ── 闭合世界：未命中任何 AOT 融合 shader → 硬报错（绝不静默回退） ──
        // 带上 key：闭合世界报错必须可定位——key 是登记/查表两侧的唯一标识，
        // 可用来查 fused_registry.hpp 里有没有该条目、以及它属于哪个 Layer 路径。
        return std::unexpected(Error{
            "GpuEngine::eval_expr: 未找到该内联表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）；"
            "key=" + key});
    }

    // ── 归约向量原生形状输出（LayerNorm/RMSNorm 小向量缓存） ────────
    // ── 精度变体能力查询（Phase 2 in-kernel f16）──────────────────────────
    // 只有 (结构, 真实输入精度, 目标输出精度) 的带类型 shader 已注册时才为 true：
    // 适配层据此决定"直接吃 f16"还是"边界 cast 回退"。
    [[nodiscard]] bool supports_native_data_move() const noexcept override { return true; }
    // Phase C2：f16 归约——reduce_f16_pipeline_ 创建成功即原生（设备无
    // 16bit 存储时为 false → 适配层维持边界 cast，与引擎内回退等价但
    // 归因表仍可见）。
    [[nodiscard]] bool supports_native_f16_reduce() const noexcept override
    {
        return backend_.has_reduce_f16_pipeline();
    }

    [[nodiscard]] bool supports_expr_precision_variant(
        const ExprSpec& raw_spec, std::span<const Tensor> inputs,
        Precision P = Precision::F32) const override
    {
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // psig == 0（全 f32）：基类快路径直通，无需变体。
        // psig != 0：选路与 eval_expr_impl 完全同源（`#a` native16 优先 →
        // `#x` 运行期分派）。**本函数只是查询**（NVI 入口每次都调）→ 传
        // log_hit=false，否则 NN_PREC_TRACE 的命中计数会翻倍。
        const nn::ExprPrecSig psig = nn::expr_prec_sig_of(inputs, P);
        if (psig == 0)
            return false;
        // 查表 key 与 native16 谓词都必须在 **canonical** 形态上求值：登记端
        // 存的是 canonical spec，生成期的谓词判定也基于它（如常量折叠会改
        // 常量值）。对 raw_spec 求值可能与生成端不一致 → 静默选错变体。
        const ExprSpec canon = nn::canonicalize_expr_spec(raw_spec);
        const std::string key = nn::expr_spec_key(canon);
        const auto* fs = find_prec_variant_(key, psig, canon);
        return fs != nullptr && backend_.has_fused_shader(fs->key);
#else
        (void)raw_spec; (void)inputs; (void)P;
        return false;
#endif
    }
    // 与 eval_expr 相同，但以 vector_out=1 调度归约融合 shader（thread 0 写
    // (rows,1)/(1,cols) 归约向量，不写全尺寸广播）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce_impl(
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
        // 生成器补齐该形态后可在此放开。
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
        // 变体优先（`#a` native16 → `#x` 运行期分派）：与 eval_expr 同款
        // (key,sig) 匹配；归约形态的 native16 谓词恒 false → 自动落 `#x`
        const nn::fused::FusedShader* fs = (psig != 0)
            ? find_prec_variant_(key, psig, spec, /*log_hit=*/true)
            : nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            if (!fi_r) return std::unexpected(fi_r.error());
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            const auto vp = nn::expr_spec_runtime_view_params(spec);
            const bool out_f16 = nn::expr_variant_out_f16(fs->prec_sig, psig);
            auto out = backend_.run_fused_gpu(
                fs->key, gpu_inputs, spec.consts, rows, cols, /*vector_out=*/true, vp,
                spec.rparams,
                /*output_override=*/nullptr, nn::expr_spec_runtime_matmul_k(spec),
                nn::expr_spec_runtime_matmul_batch(spec),
                nn::expr_spec_runtime_matmul_trans(spec),
                std::nullopt, out_f16, psig);
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
                "；NVI 入口本应在此前抬到 f32（supports_expr_precision_variant 与 shader 查询不一致）"});
        // ── 闭合世界：未命中归约融合 shader → 硬报错（绝不静默回退） ──
        return std::unexpected(Error{
            "GpuEngine::eval_expr_reduce: 未找到该归约表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）；"
            "key=" + key});
    }

    // ── 目标传递（destination-passing）：结果直接写回已有张量 ─────────────
    // 走 run_fused_gpu 的 output_override —— dst 的 buffer 作为写only输出绑定，
    // 若表达式引用了 leaf(dst) 则同一 buffer 同时作为 readonly 输入绑定：
    // 逐元素同索引"先读后写"，无跨调用危害（与 GPU 原地原语的既有做法一致）。
    // 语义与限制同 CpuEngine::eval_expr_into（仅逐元素表达式；无分配）。
    [[nodiscard]] Result<void> eval_expr_into_impl(
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
        // 变体优先（`#a` native16 → `#x` 运行期分派）：目标精度 = dst.precision()
        const nn::ExprPrecSig psig =
            nn::expr_prec_sig_of(inputs, dst.precision());
#ifdef NN_FUSED_REGISTRY_EMBEDDED
        const nn::fused::FusedShader* fs = (psig != 0)
            ? find_prec_variant_(key, psig, spec, /*log_hit=*/true)
            : nn::fused::find_fused(key);
        if (fs && backend_.has_fused_shader(fs->key))
        {
            auto fi_r = fused_buffers_(inputs);
            if (!fi_r) return std::unexpected(fi_r.error());
            std::vector<const GpuBuffer*>& gpu_inputs = fi_r->bufs;   // owners 随 fi_r 存活到本作用域末
            auto dst_gpu = import(dst);
            if (!dst_gpu) return std::unexpected(dst_gpu.error());
            // ── output_override 必须按目标存储精度取视图 ──────────────────
            // f16 目标必须经 f16_view 包 f16 buffer（同 run_fused_gpu 的
            // out_f16 做法）：按默认 F32 取 gpu_tensor() 会得到空
            // shared_ptr → Debug NN_ASSERT 引爆；Release 空解引用 = UB
            // （MSVC 下拿到空 override → 结果落临时 buffer 被丢弃 →
            //  **参数静默冻结（loss 恒 ln V）**）。存储/标签不一致时
            // 显式报错，绝不静默。
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
                nn::expr_spec_runtime_matmul_batch(spec),
                nn::expr_spec_runtime_matmul_trans(spec),
                /*fold_k=*/std::nullopt, /*out_f16=*/false, psig);
            if (!out) return std::unexpected(out.error());
            // dst 原为 CPU staging 时，import 上传了新 buffer（结果在它上面）
            // → 用 upload 后的张量替换 dst，保证调用方看到更新后的数据
            if (dst.is_cpu())
                dst = std::move(*dst_gpu);
            return {};
        }
#endif
        if (psig != 0)
            return std::unexpected(Error{
                "GpuEngine::eval_expr_into: 该 (结构,精度) 变体未预生成（in-kernel f16 覆盖不足）"
                "；NVI 入口本应在此前抬到 f32（supports_expr_precision_variant 与 shader 查询不一致）"});
        // ── 闭合世界：未命中 AOT 融合 shader → 硬报错（绝不静默回退） ──
        return std::unexpected(Error{
            "GpuEngine::eval_expr_into: 未找到该表达式的 AOT 融合 shader（闭合世界）；"
            "请将对应表达式纳入构建期扫描（scan_exprs dry-run 需覆盖该 Layer 路径）；"
            "key=" + key});
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
    // ── 跨设备拉取：CPU → 设备直传（M6，17 §4.3 / 15 §3.2 P3）────────────
    // 原 `ensure_gpu`（16 §2 的 43 处调用点已全部改为公共入口 `import`）。
    // 语义：
    //   已在设备：同精度 = 零拷贝别名；异精度 = 引擎内 cast；
    //   CPU 源 + f32→f32：直接从宿主存储上载（与原 ensure_gpu 同一条快路径，
    //     不经额外宿主拷贝）；
    //   其余精度组合（如 f16 源）：经宿主中转（升 f32 再按 P 落回）——
    //     原 ensure_gpu 硬取 `cpu_matrix()`（F32 槽）对 f16 源会取到空指针。
    [[nodiscard]] Result<Tensor> import_impl(const Tensor& src, Precision P) override
    {
        if (src.is_gpu())
        {
            if (P == src.precision())
                return src;                 // 同设备同精度 = 零拷贝别名（共享拷贝）
            return cast(src, P);            // 同设备异精度 = 引擎内 cast
        }
        if (!src.valid())
            return std::unexpected(Error{"import: invalid tensor"});
        if (src.precision() == Precision::F32 && P == Precision::F32)
        {
            auto r = GpuTensor::from_matrix(src.cpu_matrix(), backend_);
            if (!r)
                return std::unexpected(r.error());
            return Tensor::from_gpu(std::move(*r));
        }
        auto m = to_matrix(src, Precision::F32);
        if (!m)
            return std::unexpected(m.error());
        return from_matrix(*m, P);
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN