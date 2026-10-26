// ── compute_gpu_storage.hpp ───────────────────────────────────────────────────
// GPU 存储层：GpuBuffer（显存块 RAII）+ GpuTensorT<P>（形状 + 共享句柄）。
//
// 层级位置：**低于** Vulkan 后端与代数层。
//   · 只依赖 Vulkan 句柄类型 + MemoryPool，不认识 GpuBackend / ComputeEngine；
//   · 因此 algebra_matrix.hpp / compute_tensor.hpp 可以只看到"存储是什么"，
//     不必被迫解析整个后端（含 SPIR-V 嵌入与融合注册表）。
//
// 需要 GpuBackend 的方法（from_matrix / to_matrix / create_*）定义在
// compute_gpu_tensor_impl.hpp，由 compute_vk_backend.hpp 在类定义之后包含。
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <neuralnet.cpp/core_errors.hpp>
#include "../core_observer_ptr.hpp"
#include "../core_domain.hpp"
#include <neuralnet.cpp/precision.hpp>
#include "compute_memory_pool.hpp"

namespace nn
{

class GpuBackend;

template <Precision P> class MatrixT;
template <Precision P> class GpuTensorT;

using GpuTensor = GpuTensorT<Precision::F32>;
using GpuTensorF32 = GpuTensorT<Precision::F32>;
using GpuTensorF16 = GpuTensorT<Precision::F16>;

// ══════════════════════════════════════════════════════════════════════════
// GpuBuffer — GPU 缓冲区 RAII 封装
// ══════════════════════════════════════════════════════════════════════════

class GpuBuffer
{
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    MemoryPool::Allocation alloc_;
    observer_ptr<MemoryPool> pool_;
    // backend keepalive：GpuBuffer 可能晚于 GpuEngine 析构。持有强引用后，
    // 析构时可以安全访问 backend（延迟销毁队列 / memory pool）。
    std::shared_ptr<GpuBackend> backend_keepalive_;

    // ── 写版本（write epoch）──────────────────────────────────────────────
    // 与 MatrixT 的 write_epoch_ 同义（见 algebra_matrix.hpp 的说明）：内容
    // "可能已经变了"的单调计数，唯一推进入口 = ComputeEngine 的写入口护栏。
    // 放在 GpuBuffer（而不是 GpuTensorT）上：GpuTensorT 存在零拷贝重贴标签
    // （with_shape）与 import 别名，同一块显存可能有多个 GpuTensorT 句柄 ——
    // 版本必须随真正存放数据的 buffer 走。
    std::uint64_t write_epoch_{0};

public:
    // 设备域标识（生命周期 v3）：backend 对象地址。buffer 持 backend keepalive
    // ⇒ 张量存活期内该 backend 不会被析构、其地址不会被新 backend 复用。
    [[nodiscard]] DomainId backend_identity() const noexcept
    {
        return reinterpret_cast<DomainId>(backend_keepalive_.get());
    }

    [[nodiscard]] std::uint64_t write_epoch() const noexcept { return write_epoch_; }
    void mark_content_changed() noexcept { ++write_epoch_; }

    GpuBuffer() = default;

    GpuBuffer(VkDevice device, VkBuffer buffer, const MemoryPool::Allocation& alloc,
              MemoryPool& pool, std::shared_ptr<GpuBackend> backend_keepalive)
        : device_(device), buffer_(buffer), alloc_(alloc), pool_(pool),
          backend_keepalive_(std::move(backend_keepalive)) {}

    // 析构定义在 GpuBackend 之后：batch 录制期间已录制的 descriptor 仍引用
    // 本 buffer，需延迟到 end_batch 提交完成后再销毁（见 defer_buffer_destroy）。
    ~GpuBuffer();

    GpuBuffer(GpuBuffer&& o) noexcept
        : device_(o.device_), buffer_(o.buffer_), alloc_(o.alloc_), pool_(o.pool_),
          backend_keepalive_(std::move(o.backend_keepalive_)),
          write_epoch_(o.write_epoch_)
    {
        o.device_ = VK_NULL_HANDLE;
        o.buffer_ = VK_NULL_HANDLE;
        o.alloc_ = {};
        o.pool_.reset();
        o.backend_keepalive_.reset();
    }

    GpuBuffer& operator=(GpuBuffer&& o) noexcept
    {
        if (this != &o)
        {
            std::swap(device_, o.device_);
            std::swap(buffer_, o.buffer_);
            std::swap(alloc_, o.alloc_);
            pool_.swap(o.pool_);
            std::swap(backend_keepalive_, o.backend_keepalive_);
            std::swap(write_epoch_, o.write_epoch_);
        }
        return *this;
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // 创建 Device Local 缓冲区（尺寸语义 = 字节数，§6.3）
    // usage 恒补 SHADER_DEVICE_ADDRESS（P3 实例表按设备地址引用张量缓冲）：
    // 与 MemoryPool 分配侧的 DEVICE_ADDRESS 标志成对（都随 bda_available 门控）。
    [[nodiscard]] static Result<GpuBuffer> create_device_local(
        VkDevice device, MemoryPool& pool,
        std::size_t byte_count, VkBufferUsageFlags usage,
        std::shared_ptr<GpuBackend> backend_keepalive)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = byte_count;
        buf_info.usage = usage;
        if (pool.device_address_alloc())
            buf_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
        if (res != VK_SUCCESS)
            NN_FAIL("vkCreateBuffer failed: " + std::to_string(res));
        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

        auto alloc_r = pool.allocate(
            mem_reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!alloc_r)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            NN_TRY_CHECK(alloc_r);
        }

        res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
        if (res != VK_SUCCESS)
        {
            pool.free(*alloc_r);
            vkDestroyBuffer(device, buffer, nullptr);
            NN_FAIL("vkBindBufferMemory failed: " + std::to_string(res));
        }

        return GpuBuffer(device, buffer, *alloc_r, pool,
                         std::move(backend_keepalive));
    }

    // 创建 Host Visible 缓冲区（尺寸语义 = 字节数，§6.3）
    [[nodiscard]] static Result<GpuBuffer> create_host_visible(
        VkDevice device, MemoryPool& pool,
        std::size_t byte_count, VkBufferUsageFlags usage,
        std::shared_ptr<GpuBackend> backend_keepalive)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = byte_count;
        buf_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        // P3：与 create_device_local 同款补 SHADER_DEVICE_ADDRESS（池分配侧
        // 已配对 DEVICE_ADDRESS 标志，见 MemoryPool::create_block）
        if (pool.device_address_alloc())
            buf_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
        if (res != VK_SUCCESS)
            NN_FAIL("vkCreateBuffer failed: " + std::to_string(res));
        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

        VkMemoryPropertyFlags preferred, fallback;
        if (pool.is_uma())
        {
            // 真 UMA/共享显存：slab 落在统一内存池（device-local 即 host-visible）
            preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                      | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            fallback  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        else
        {
            // 独显：纯 HOST_VISIBLE（系统 RAM），避免 slab 落在 GPU 可驻留内存
            preferred = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            fallback  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        }

        auto alloc_r = pool.allocate(mem_reqs, preferred, fallback);
        if (!alloc_r)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            NN_TRY_CHECK(alloc_r);
        }

        res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
        if (res != VK_SUCCESS)
        {
            pool.free(*alloc_r);
            vkDestroyBuffer(device, buffer, nullptr);
            NN_FAIL("vkBindBufferMemory failed: " + std::to_string(res));
        }

        return GpuBuffer(device, buffer, *alloc_r, pool,
                         std::move(backend_keepalive));
    }

    [[nodiscard]] VkBuffer impl() const noexcept { return buffer_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }
};

// ══════════════════════════════════════════════════════════════════════════
// GpuTensorT<P> — GPU 矩阵抽象（多精度，docs/development/05-mixed-precision.md §6.3）
// buffer 字节数 = rows * cols * sizeof(elem<P>)（f32=4B / f16=2B）
// ══════════════════════════════════════════════════════════════════════════

template <Precision P>
class GpuTensorT
{
private:
    std::shared_ptr<GpuBuffer> buffer_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    GpuTensorT() = default;

    GpuTensorT(std::shared_ptr<GpuBuffer> buffer, std::size_t rows, std::size_t cols)
        : buffer_(std::move(buffer)), rows_(rows), cols_(cols) {}

    // 从 CPU Matrix 创建（上传数据；同精度跨设备 = 原始字节拷贝，§6.5）
    [[nodiscard]] static Result<GpuTensorT<P>> from_matrix(
        const MatrixT<P>& cpu_mat, GpuBackend& backend);

    // 创建空的 GPU Tensor（用于输出）
    [[nodiscard]] static Result<GpuTensorT<P>> create_empty(
        std::size_t rows, std::size_t cols, GpuBackend& backend);

    // 创建空的 Host Visible Tensor（activation offload 存储，§6.6）
    [[nodiscard]] static Result<GpuTensorT<P>> create_host_visible_empty(
        std::size_t rows, std::size_t cols, GpuBackend& backend);

    // 转换为 CPU Matrix（下载数据）
    [[nodiscard]] Result<MatrixT<P>> to_matrix(GpuBackend& backend) const;

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && buffer_->valid(); }
    [[nodiscard]] const GpuBuffer& buffer() const noexcept { return *buffer_; }
    // 底层 buffer 的共享句柄（把同一 buffer 以另一种元素类型重贴标签用，
    // 见 run_fused_gpu 的 out_f16：in-kernel f16 的输出是 f16 字节布局）
    [[nodiscard]] std::shared_ptr<GpuBuffer> shared_buffer() const noexcept
    { return buffer_; }

    // 零拷贝 reshape：共享底层 buffer，只改变形状元数据
    [[nodiscard]] GpuTensorT<P> with_shape(std::size_t new_rows, std::size_t new_cols) const
    {
        return GpuTensorT<P>(buffer_, new_rows, new_cols);
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN
