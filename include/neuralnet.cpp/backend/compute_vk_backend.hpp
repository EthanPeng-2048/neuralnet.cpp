// ── compute_vk_backend.hpp ──────────────────────────────────────────────────────
// Vulkan Compute Backend
//
// 职责：
//   - 管理 Vulkan 实例、设备、队列（RAII）
//   - 管理 MemoryPool 子分配器
//   - 管理 StagingRing 环形缓冲区
//   - 提供 GPU 操作 API（matmul、elementwise）
//
// 同步机制：
//   - 每个 matmul_gpu/elementwise_gpu 调用都创建 fence
//   - 提交后立即等待 fence，确保 GPU 计算完成
//   - 完成后清理资源（command buffer、descriptor set）
//
// 双轨制架构：
//   - Staging Path：CPU span → Staging → GPU → 计算 → GPU → Staging → CPU span
//   - GPU-Resident Path：GpuTensor → 计算 → GpuTensor（全程 GPU 显存）
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core_errors.hpp"
#include "../core_observer_ptr.hpp"
#include "../core_config.hpp"
#include "compute_memory_pool.hpp"
#include "compute_staging_ring.hpp"

// SPIR-V 嵌入头文件（由 CMake 生成）
#if __has_include("matmul_spv.hpp")
#include "matmul_spv.hpp"
#define NN_MATMUL_SPV_EMBEDDED
#endif

#if __has_include("matmul_tiled_spv.hpp")
#include "matmul_tiled_spv.hpp"
#define NN_MATMUL_TILED_SPV_EMBEDDED
#endif

#if __has_include("batched_matmul_spv.hpp")
#include "batched_matmul_spv.hpp"
#define NN_BATCHED_MATMUL_SPV_EMBEDDED
#endif

#if __has_include("rearrange_3d_spv.hpp")
#include "rearrange_3d_spv.hpp"
#define NN_REARRANGE_3D_SPV_EMBEDDED
#endif

#if __has_include("elementwise_v2_spv.hpp")
#include "elementwise_v2_spv.hpp"
#define NN_ELEMENTWISE_V2_SPV_EMBEDDED
#endif

#if __has_include("reduce_spv.hpp")
#include "reduce_spv.hpp"
#define NN_REDUCE_SPV_EMBEDDED
#endif

#if __has_include("broadcast_spv.hpp")
#include "broadcast_spv.hpp"
#define NN_BROADCAST_SPV_EMBEDDED
#endif

#if __has_include("transpose_spv.hpp")
#include "transpose_spv.hpp"
#define NN_TRANSPOSE_SPV_EMBEDDED
#endif

#if __has_include("gather_spv.hpp")
#include "gather_spv.hpp"
#define NN_GATHER_SPV_EMBEDDED
#endif

#if __has_include("scatter_add_spv.hpp")
#include "scatter_add_spv.hpp"
#define NN_SCATTER_ADD_SPV_EMBEDDED
#endif

#if __has_include("scan_prefix_outer_spv.hpp")
#include "scan_prefix_outer_spv.hpp"
#define NN_SCAN_PREFIX_OUTER_SPV_EMBEDDED
#endif

#if __has_include("scan_suffix_outer_spv.hpp")
#include "scan_suffix_outer_spv.hpp"
#define NN_SCAN_SUFFIX_OUTER_SPV_EMBEDDED
#endif

#if __has_include("scan_prefix_outer_gen_spv.hpp")
#include "scan_prefix_outer_gen_spv.hpp"
#define NN_SCAN_PREFIX_OUTER_GEN_SPV_EMBEDDED
#endif

#if __has_include("scan_suffix_outer_gen_spv.hpp")
#include "scan_suffix_outer_gen_spv.hpp"
#define NN_SCAN_SUFFIX_OUTER_GEN_SPV_EMBEDDED
#endif

#if __has_include("outer_col_spv.hpp")
#include "outer_col_spv.hpp"
#define NN_OUTER_COL_SPV_EMBEDDED
#endif

#if __has_include("cast_spv.hpp")
#include "cast_spv.hpp"
#define NN_CAST_SPV_EMBEDDED
#endif

// AOT 融合 shader 注册表（构建期 scan_exprs 收集 + gen_fused 合成；表达式
// 只出现在 Layer，本表是折叠后的派生物）。运行时按 expr_spec_key 匹配 dispatch。
#if __has_include("fused_registry.hpp")
#include "fused_registry.hpp"
#endif

// Vulkan 设备/Pipeline 辅助类（detail::vk_check/convert + VulkanDevice + VulkanPipeline）
#include "compute_vk_device.hpp"

namespace nn
{

// 前向声明（模板 + 别名，匹配 algebra_matrix.hpp 的 MatrixT<P>）
template <Precision P> class MatrixT;
using Matrix = MatrixT<Precision::F32>;

// 前向声明（模板 + 别名，匹配本文件末尾的 GpuTensorT<P> 定义，§6.3）
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

public:
    GpuBuffer() = default;

    GpuBuffer(VkDevice device, VkBuffer buffer, const MemoryPool::Allocation& alloc, MemoryPool& pool)
        : device_(device), buffer_(buffer), alloc_(alloc), pool_(pool) {}

    // 析构定义在 GpuBackend 之后：batch 录制期间已录制的 descriptor 仍引用
    // 本 buffer，需延迟到 end_batch 提交完成后再销毁（见 defer_buffer_destroy）。
    ~GpuBuffer();

    // 移动语义
    GpuBuffer(GpuBuffer&& o) noexcept
        : device_(o.device_), buffer_(o.buffer_), alloc_(o.alloc_), pool_(o.pool_)
    {
        o.device_ = VK_NULL_HANDLE;
        o.buffer_ = VK_NULL_HANDLE;
        o.alloc_ = {};
        o.pool_.reset();
    }

    GpuBuffer& operator=(GpuBuffer&& o) noexcept
    {
        if (this != &o)
        {
            std::swap(device_, o.device_);
            std::swap(buffer_, o.buffer_);
            std::swap(alloc_, o.alloc_);
            pool_.swap(o.pool_);
        }
        return *this;
    }

    // 禁止拷贝
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // 创建 Device Local 缓冲区（尺寸语义 = 字节数，§6.3）
    [[nodiscard]] static Result<GpuBuffer> create_device_local(
        VkDevice device, MemoryPool& pool,
        std::size_t byte_count, VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = byte_count;
        buf_info.usage = usage;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkCreateBuffer failed: " + std::to_string(res)});

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

        auto alloc_r = pool.allocate(
            mem_reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!alloc_r)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(alloc_r.error());
        }

        res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
        if (res != VK_SUCCESS)
        {
            pool.free(*alloc_r);
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(Error{"vkBindBufferMemory failed: " + std::to_string(res)});
        }

        return GpuBuffer(device, buffer, *alloc_r, pool);
    }

    // 创建 Host Visible 缓冲区（尺寸语义 = 字节数，§6.3）
    [[nodiscard]] static Result<GpuBuffer> create_host_visible(
        VkDevice device, MemoryPool& pool,
        std::size_t byte_count, VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = byte_count;
        buf_info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult res = vkCreateBuffer(device, &buf_info, nullptr, &buffer);
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkCreateBuffer failed: " + std::to_string(res)});

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
            return std::unexpected(alloc_r.error());
        }

        res = vkBindBufferMemory(device, buffer, alloc_r->memory, alloc_r->offset);
        if (res != VK_SUCCESS)
        {
            pool.free(*alloc_r);
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(Error{"vkBindBufferMemory failed: " + std::to_string(res)});
        }

        return GpuBuffer(device, buffer, *alloc_r, pool);
    }

    [[nodiscard]] VkBuffer impl() const noexcept { return buffer_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }
};

// ══════════════════════════════════════════════════════════════════════════
// GpuTensorT<P> — GPU 矩阵抽象（多精度，docs/23 §6.3）
// buffer 字节数 = rows * cols * sizeof(elem<P>)（f32=4B / f16=2B）
// 别名：GpuTensor = GpuTensorT<F32>（现有调用点零改动）
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
        const MatrixT<P>& cpu_mat, class GpuBackend& backend);

    // 创建空的 GPU Tensor（用于输出）
    [[nodiscard]] static Result<GpuTensorT<P>> create_empty(
        std::size_t rows, std::size_t cols, class GpuBackend& backend);

    // 创建空的 Host Visible Tensor（activation offload 存储，§6.6）
    [[nodiscard]] static Result<GpuTensorT<P>> create_host_visible_empty(
        std::size_t rows, std::size_t cols, class GpuBackend& backend);

    // 转换为 CPU Matrix（下载数据）
    [[nodiscard]] Result<MatrixT<P>> to_matrix(class GpuBackend& backend) const;

    // 访问器
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && buffer_->valid(); }
    [[nodiscard]] const GpuBuffer& buffer() const noexcept { return *buffer_; }

    // 零拷贝 reshape：共享底层 buffer，只改变形状元数据
    [[nodiscard]] GpuTensorT<P> with_shape(std::size_t new_rows, std::size_t new_cols) const
    {
        return GpuTensorT<P>(buffer_, new_rows, new_cols);
    }
};

// 别名（现有调用点零改动；f16 显式使用 GpuTensorF16）
using GpuTensor = GpuTensorT<Precision::F32>;
using GpuTensorF32 = GpuTensorT<Precision::F32>;
using GpuTensorF16 = GpuTensorT<Precision::F16>;

// ══════════════════════════════════════════════════════════════════════════
// GpuBackend — Vulkan 计算后端单例
// ══════════════════════════════════════════════════════════════════════════

class GpuBackend
{
private:
    VulkanDevice device_;
    VulkanPipeline matmul_pipeline_;
    VulkanPipeline matmul_tiled_pipeline_;
    VulkanPipeline batched_matmul_pipeline_;
    VulkanPipeline elementwise_v2_pipeline_;
    VulkanPipeline reduce_pipeline_;
    VulkanPipeline broadcast_pipeline_;
    VulkanPipeline rearrange_3d_pipeline_;
    VulkanPipeline transpose_pipeline_;
    VulkanPipeline gather_pipeline_;
    VulkanPipeline scatter_add_pipeline_;
    // RLA 扫描原语（手写原语，不进 AOT 融合注册表；铁律 3：shader 不含算法）
    VulkanPipeline scan_prefix_outer_pipeline_;
    VulkanPipeline scan_suffix_outer_pipeline_;
    VulkanPipeline scan_prefix_outer_gen_pipeline_;  // 通用 d_k（>64）前缀扫描
    VulkanPipeline scan_suffix_outer_gen_pipeline_;  // 通用 d_k（>64）后缀扫描
    VulkanPipeline outer_col_pipeline_;
    // 通用精度转换原语（engine.cast 的 GPU 实现；f16↔f32，kind 分派精度对）
    VulkanPipeline cast_pipeline_;
    // AOT 融合 shader pipelines（key = expr_spec_key → pipeline；由构建期
    // fused_registry.hpp 注册，运行时按 key 匹配后直接 dispatch）
    std::unordered_map<std::string, VulkanPipeline> fused_pipelines_;
    // 归约轴：-1=逐元素, 0=行归约, 1=列归约（决定 push constants 与 dispatch）
    std::unordered_map<std::string, int> fused_reduce_axis_;
    // 是否含前置 matmul 段（push constants 多 rows + mm_k 两个 uint）
    std::unordered_map<std::string, bool> fused_has_matmul_;
    // 运行时视图参数个数（RowMod/RotateHalf 的 vp push constant 槽数）
    std::unordered_map<std::string, std::uint32_t> fused_view_param_counts_;
    // 运行时标量参数个数（优化器 lr/eps/β 等的 rp push constant 槽数）
    std::unordered_map<std::string, std::uint32_t> fused_rparam_counts_;
    // 每线程处理元素数（1=标量, 4=vec4；决定 dispatch 宽度缩放）
    std::unordered_map<std::string, std::uint32_t> fused_vec_width_;

    std::unique_ptr<MemoryPool> memory_pool_;
    // 第二阶段（P3-2）：瞬态/持久分池。batch 录制期（batch_mode_=true）创建的
    // 张量（融合临时、每步激活、输入上传等）走 transient_pool_，参数/梯度/权重
    // （构建期 batch_mode_=false）走 memory_pool_。此拆分**纯组织性**——两池均
    // 不强制释放，仅按生命周期隔离，避免频繁临时分配在参数常驻块里切出碎片。
    std::unique_ptr<MemoryPool> transient_pool_;
    std::unique_ptr<StagingRing> staging_ring_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool gpu_tensor_pool_ = VK_NULL_HANDLE;

    // ── Command Buffer Batching + 多帧流水线（P0-1）─────────────────────
    // 旧实现：单一 batch_cmd_ + 单 fence，end_batch/flush_batch 每次
    // vkQueueSubmit + 阻塞 vkWaitForFences——GPU 执行 step N 时 host 被 fence
    // 卡死无法录制 step N+1（GPU 占用率 = GPU 忙时间 / (GPU 忙 + 全部 host
    // 录制 + 传输)，见《显存 & 负载不均衡分析报告》§2.1）。
    //
    // 新模型（N 帧环）：
    //   - PIPELINE_FRAMES 个 (command buffer + fence) 在 initialize() 预分配，
    //     轮转复用；描述符集按帧归属（帧的 fence 信号后才可释放）
    //   - begin_batch：取下一帧；该帧若仍在飞行则等其 fence（**只在环槽
    //     复用前等待**），reap 后开始录制
    //   - end_batch / flush_batch：提交不等待（host 继续录制，GPU 在队列
    //     上先行执行；同队列 FIFO 保证数据依赖）。空帧不提交（省 submit）
    //   - wait_in_flight：**"真正要结果"的阻塞点**（to_matrix/copy_from 读
    //     GPU 内存前）——等待所有在飞帧并 reap
    //   - release_idle_pool_blocks：非阻塞 reap 已完成帧 + 归还空闲块
    // 效果：host 录制 step N+1 与 GPU 执行 step N 重叠；host 只在真正要
    // 读 GPU 数据（logits 探针 / loss 标量 / 日志采样）时阻塞。
    static constexpr std::size_t PIPELINE_FRAMES = 6;  // 环深度（P0-1: 3→6，加深流水线提重叠度）
    struct Frame
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;  // 创建即为 unsignaled
        std::vector<VkDescriptorSet> desc_sets;  // 本帧使用的描述符集
        bool in_flight = false;                  // 已提交、fence 尚未被等待
    };
    std::vector<Frame> frames_;
    std::size_t frame_next_ = 0;         // 下一轮转帧（begin/flush 取用）
    std::size_t last_active_frame_ = 0;  // 最近录制/提交的帧（非 batch 销毁决策用）
    bool batch_mode_ = false;
    std::size_t batch_frame_ = 0;        // 当前录制中的帧
    bool batch_has_ops_ = false;         // 当前帧是否已录制 op（空帧不提交）
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;  // 当前帧的 command buffer

    std::mutex init_mutex_;
    std::mutex queue_mutex_;
    bool initialized_ = false;

    // ── f16 硬件能力（D4，§7.1）────────────────────────────────────────
    // GPU shaderFloat16：运行期一次查定，运行期不变。
    // true  → f16 GEMM 变体②（f16vec 加载，需该特性）
    // false → f16 GEMM 变体①（u8 软件解码，设备无关，默认路径）
    bool has_shader_float16_ = false;

    // ── 延迟销毁：batch 录制期间 copy-on-write 替换旧 buffer 时，旧 buffer
    // 仍被已录制的 descriptor set 引用，不能立即 vkDestroyBuffer。
    // end_batch/flush_batch 提交完成并释放 descriptor sets 后统一销毁。
    //
    // ⚠ 内存归还也须延迟到 vkDestroyBuffer 之后（D1 修复）：
    //   若立即 pool_->free()，新 buffer 可能分配到旧 buffer 尚未销毁的
    //   同一区间 → 两个存活 buffer 内存重叠，违反 Vulkan 规范
    //   （VUID-vkBindBufferMemory-memory-01988 类约束）。
    //   数据竞争方面：同一 command buffer 内命令按录制顺序执行，旧 buffer
    //   的已录制命令先于复用同一区间的新 buffer 命令执行，无数据竞争。
    //   但规范合规要求 buffer 对象本身不得重叠，故内存归还必须等 buffer
    //   销毁完成。
    //
    //   多帧流水线（P0-1）：锁窗从"整个 batch"缩短到"所属帧"——每条延迟
    //   销毁打上帧标签（frame），该帧的 fence 信号后（环槽复用 / drain /
    //   非阻塞 reap）即销毁并归还内存。
    struct PendingDestroy
    {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        MemoryPool::Allocation alloc;
        observer_ptr<MemoryPool> pool;
        std::size_t frame = 0;  // 允许销毁的帧索引（该帧完成后才可释放）
    };
    std::mutex pending_mutex_;
    std::vector<PendingDestroy> pending_destroys_;

    static constexpr uint32_t WORKGROUP_SIZE = 16;

    GpuBackend() = default;

    // 获取 SPIR-V 字节码
    [[nodiscard]] static const std::vector<uint32_t>& get_matmul_spirv()
    {
#ifdef NN_MATMUL_SPV_EMBEDDED
        return nn_matmul_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_matmul_tiled_spirv()
    {
#ifdef NN_MATMUL_TILED_SPV_EMBEDDED
        return nn_matmul_tiled_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_batched_matmul_spirv()
    {
#ifdef NN_BATCHED_MATMUL_SPV_EMBEDDED
        return nn_batched_matmul_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_rearrange_3d_spirv()
    {
#ifdef NN_REARRANGE_3D_SPV_EMBEDDED
        return nn_rearrange_3d_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_elementwise_v2_spirv()
    {
#ifdef NN_ELEMENTWISE_V2_SPV_EMBEDDED
        return nn_elementwise_v2_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_reduce_spirv()
    {
#ifdef NN_REDUCE_SPV_EMBEDDED
        return nn_reduce_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_broadcast_spirv()
    {
#ifdef NN_BROADCAST_SPV_EMBEDDED
        return nn_broadcast_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_transpose_spirv()
    {
#ifdef NN_TRANSPOSE_SPV_EMBEDDED
        return nn_transpose_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_gather_spirv()
    {
#ifdef NN_GATHER_SPV_EMBEDDED
        return nn_gather_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_scatter_add_spirv()
    {
#ifdef NN_SCATTER_ADD_SPV_EMBEDDED
        return nn_scatter_add_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_scan_prefix_outer_spirv()
    {
#ifdef NN_SCAN_PREFIX_OUTER_SPV_EMBEDDED
        return nn_scan_prefix_outer_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_scan_suffix_outer_spirv()
    {
#ifdef NN_SCAN_SUFFIX_OUTER_SPV_EMBEDDED
        return nn_scan_suffix_outer_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_scan_prefix_outer_gen_spirv()
    {
#ifdef NN_SCAN_PREFIX_OUTER_GEN_SPV_EMBEDDED
        return nn_scan_prefix_outer_gen_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_scan_suffix_outer_gen_spirv()
    {
#ifdef NN_SCAN_SUFFIX_OUTER_GEN_SPV_EMBEDDED
        return nn_scan_suffix_outer_gen_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_outer_col_spirv()
    {
#ifdef NN_OUTER_COL_SPV_EMBEDDED
        return nn_outer_col_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_cast_spirv()
    {
#ifdef NN_CAST_SPV_EMBEDDED
        return nn_cast_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    // ── AOT 融合 shader SPIR-V 来自构建期生成的 fused_registry.hpp ────────
    // （FusedShader 直接携带内联 SPIR-V，无需按名 getter；注册在 init 时遍历）

public:
    ~GpuBackend()
    {
        if (device_.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_.device());

            flush_pending_destroys();

            staging_ring_.reset();

            // 释放帧环 fences（环内 command buffers 随 command pool 一起释放）
            for (auto& f : frames_)
            {
                if (f.fence != VK_NULL_HANDLE)
                    vkDestroyFence(device_.device(), f.fence, nullptr);
            }

            if (gpu_tensor_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_.device(), gpu_tensor_pool_, nullptr);
            if (command_pool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_.device(), command_pool_, nullptr);
        }

        memory_pool_.reset();
        transient_pool_.reset();
    }

    // 设备丢失状态
    bool device_lost_ = false;
    [[nodiscard]] bool is_device_lost() const noexcept { return device_lost_; }

    // 禁止拷贝和移动
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;
    GpuBackend(GpuBackend&&) = delete;
    GpuBackend& operator=(GpuBackend&&) = delete;

    [[nodiscard]] static GpuBackend& instance()
    {
        // 故意泄漏（进程退出时由 OS 回收）：避免静态析构顺序问题——
        // 全局 GpuTensor 的析构可能晚于 backend，届时访问 instance() 会 UB。
        // TODO(1.1, L1): 若作为库被长驻进程 embed，此泄漏会持续累积
        //   （vkDestroyDevice/vkDestroyInstance/vkFreeMemory 永不调用）。
        //   恢复方案：显式 shutdown() + 引用计数，或进程级一次清理。
        static GpuBackend* backend = new GpuBackend();
        return *backend;
    }

    // ── 延迟销毁决策（GpuBuffer 析构时调用）────────────────────────────
    // 返回 true = 必须延迟（调用 defer_buffer_destroy）；false = 可立即销毁。
    //   - batch 录制中：延迟，打当前帧标签（buffer 仍被本帧已录制命令引用）
    //   - 非 batch：最近一次提交的帧若仍在飞行则延迟（buffer 可能被该帧
    //     命令引用，帧 fence 信号即可证明安全）；否则立即销毁
    // 内存归还与 buffer 销毁同批（D1）：buffer 销毁前不得把区间还给池。
    [[nodiscard]] bool needs_deferred_destroy() const noexcept
    {
        if (batch_mode_)
            return true;
        return last_active_frame_ < frames_.size() &&
               frames_[last_active_frame_].in_flight;
    }

    // ── 延迟销毁入口（GpuBuffer 析构时调用）──────────────────────────
    // 打上帧标签：batch 录制中 → 当前帧；非 batch → 最近活跃帧。
    void defer_buffer_destroy(VkDevice device, VkBuffer buffer,
                              const MemoryPool::Allocation& alloc,
                              observer_ptr<MemoryPool> pool)
    {
        const std::size_t frame =
            batch_mode_ ? batch_frame_ : last_active_frame_;
        std::lock_guard lock(pending_mutex_);
        pending_destroys_.push_back({device, buffer, alloc, pool, frame});
    }

    // ── 统一执行延迟销毁（必须在对应帧的 fence 信号之后调用）─────────
    // 先 vkDestroyBuffer 再归还内存到池（D1）：保证 buffer 对象销毁后
    // 其区间才可被新分配复用，杜绝存活 buffer 间的内存重叠。
    // flush 全部：用于"所有帧已完成"的 drain 点（wait_in_flight / 析构）。
    void flush_pending_destroys()
    {
        std::lock_guard lock(pending_mutex_);
        for (auto& pd : pending_destroys_)
        {
            if (pd.buffer != VK_NULL_HANDLE && pd.device != VK_NULL_HANDLE)
                vkDestroyBuffer(pd.device, pd.buffer, nullptr);
            if (pd.pool && pd.alloc.valid())
                pd.pool->free(pd.alloc);
        }
        pending_destroys_.clear();
    }

    // ── reap 单帧（多帧流水线 P0-1）──────────────────────────────────
    // 等待该帧 fence（wait=true；wait=false 要求调用方已用零超时
    // vkWaitForFences 确认 VK_SUCCESS）→ 销毁该帧的延迟 buffer 并归还
    // 内存 → 释放该帧描述符集（帧提交期间被已录制命令引用，fence 信号
    // 前不得释放）→ in_flight=false
    [[nodiscard]] Result<void> reap_frame(std::size_t i, bool wait)
    {
        auto& f = frames_[i];
        if (f.in_flight && wait)
        {
            constexpr uint64_t kFrameTimeoutNs = 60'000'000'000ULL;  // 60s
            const VkResult wr = vkWaitForFences(
                device_.device(), 1, &f.fence, VK_TRUE, kFrameTimeoutNs);
            if (wr == VK_ERROR_DEVICE_LOST)
            {
                // 设备丢失（TDR 触发）：GPU 已死亡，无法恢复
                device_lost_ = true;
                return std::unexpected(Error{
                    "GPU 设备丢失 (VK_ERROR_DEVICE_LOST): Windows TDR 已重置 GPU 驱动。"
                    "\n模型已自动保存，请使用 --resume <save-path> 重启训练。"
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)"});
            }
            if (wr != VK_SUCCESS)
                return std::unexpected(Error{
                    std::string("GPU 帧等待失败: Vulkan error ") +
                    std::to_string(static_cast<int>(wr)) +
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)"});
        }
        // 该帧的延迟销毁：先 vkDestroyBuffer 再归还内存（D1）
        {
            std::lock_guard lock(pending_mutex_);
            std::vector<PendingDestroy> rest;
            rest.reserve(pending_destroys_.size());
            for (auto& pd : pending_destroys_)
            {
                if (pd.frame == i)
                {
                    if (pd.buffer != VK_NULL_HANDLE && pd.device != VK_NULL_HANDLE)
                        vkDestroyBuffer(pd.device, pd.buffer, nullptr);
                    if (pd.pool && pd.alloc.valid())
                        pd.pool->free(pd.alloc);
                }
                else
                {
                    rest.push_back(std::move(pd));
                }
            }
            pending_destroys_ = std::move(rest);
        }
        if (!f.desc_sets.empty())
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_,
                static_cast<uint32_t>(f.desc_sets.size()), f.desc_sets.data());
        f.desc_sets.clear();
        f.in_flight = false;
        return {};
    }

    // 初始化
    // device_selector：手动指定计算设备（索引 "2" 或名称子串 "40HX"/"NVIDIA"）。
    // 空 = 自动选择（设备类型 + apiVersion 打分，见 VulkanDevice::initialize）。
    [[nodiscard]] Result<void> initialize(std::string device_selector = {})
    {
        std::lock_guard lock(init_mutex_);
        if (initialized_)
            return {};

        // 1. 检查 SPIR-V 是否可用
        const auto& spirv = get_matmul_spirv();
        if (spirv.empty())
            return std::unexpected(Error{"matmul SPIR-V bytecode not embedded"});

        // 2. 初始化 Vulkan 设备（先登记设备选择器）
        if (!device_selector.empty())
            device_.set_device_selector(std::move(device_selector));
        auto dev_r = device_.initialize();
        if (!dev_r)
            return dev_r;

        // 3. 创建 matmul pipeline
        auto pl_r = VulkanPipeline::create_matmul(device_.device(), spirv);
        if (!pl_r)
            return std::unexpected(pl_r.error());
        matmul_pipeline_ = std::move(*pl_r);

        // 4. 创建 memory pool（持久 + 瞬态）
        memory_pool_ = std::make_unique<MemoryPool>(
            device_.device(), device_.physical_device());
        transient_pool_ = std::make_unique<MemoryPool>(
            device_.device(), device_.physical_device());

        // 5. 创建 command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = device_.queue_family_index();
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        auto r = detail::vk_check(
            vkCreateCommandPool(device_.device(), &pool_info, nullptr, &command_pool_),
            __FILE__, __LINE__);
        if (!r)
            return r;

        // 6. 创建 staging ring
        //    use_timeline = 设备是否支持时间线信号量（跨 submit 依赖的正确
        //    原语；不支持则 staging 环走 host 等 fence 回退，见 StagingRing）
        staging_ring_ = std::make_unique<StagingRing>();
        auto st_r = staging_ring_->initialize(
            device_.device(), device_.physical_device(), command_pool_, *memory_pool_,
            device_.has_timeline_semaphores());
        if (!st_r)
            return st_r;

        // 7. 创建 descriptor pool for GPU-resident path
        // elementwise_v2 需要 4 个描述符/次，按最大值计算
        // batch 模式下描述符集延迟到 end_batch 释放，需足够大以容纳整个 batch
        //   ViT forward+backward ≈ 230 ops/样本，batch_size=512 时累积 ~118K sets
        //   增大到 262144 以留出余量，对应 desc 总量 1M
        constexpr std::size_t TENSOR_POOL_SETS = 262144;
        constexpr std::size_t TENSOR_POOL_DESCS = TENSOR_POOL_SETS * 4;  // elementwise_v2 需要 4

        VkDescriptorPoolSize tensor_pool_size{};
        tensor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        tensor_pool_size.descriptorCount = static_cast<uint32_t>(TENSOR_POOL_DESCS);

        VkDescriptorPoolCreateInfo tensor_pool_info{};
        tensor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        tensor_pool_info.maxSets = static_cast<uint32_t>(TENSOR_POOL_SETS);
        tensor_pool_info.poolSizeCount = 1;
        tensor_pool_info.pPoolSizes = &tensor_pool_size;
        tensor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        r = detail::vk_check(
            vkCreateDescriptorPool(device_.device(), &tensor_pool_info, nullptr, &gpu_tensor_pool_),
            __FILE__, __LINE__);
        if (!r)
            return r;

        // 7b. 预分配流水线帧环（P0-1）：N 个 command buffer + N 个 fence。
        //     旧实现每步 vkAllocateCommandBuffers + vkCreateFence（host 侧
        //     每步固定开销）；现在环内复用，全程零分配。
        frames_.resize(PIPELINE_FRAMES);
        {
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = static_cast<uint32_t>(PIPELINE_FRAMES);
            std::vector<VkCommandBuffer> cmds(PIPELINE_FRAMES, VK_NULL_HANDLE);
            r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, cmds.data()),
                __FILE__, __LINE__);
            if (!r)
                return r;
            for (std::size_t i = 0; i < PIPELINE_FRAMES; ++i)
                frames_[i].cmd = cmds[i];

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            // 初始 unsignaled：vkQueueSubmit 要求 fence 未 signal
            for (std::size_t i = 0; i < PIPELINE_FRAMES; ++i)
            {
                r = detail::vk_check(
                    vkCreateFence(device_.device(), &fence_info, nullptr, &frames_[i].fence),
                    __FILE__, __LINE__);
                if (!r)
                    return r;
            }
        }
        frame_next_ = 0;
        last_active_frame_ = 0;

        // 9. 创建 tiled matmul pipeline（可选）
        const auto& tiled_spirv = get_matmul_tiled_spirv();
        if (!tiled_spirv.empty())
        {
            auto tp_r = VulkanPipeline::create_matmul(device_.device(), tiled_spirv);
            if (tp_r)
                matmul_tiled_pipeline_ = std::move(*tp_r);
        }

        // 9b. 创建 batched matmul pipeline（3 bindings, 6*4=24B push constants）
        // push constants: M, N, K, transA, transB, alpha（比 matmul 多一个输出缩放系数）
        const auto& batched_spirv = get_batched_matmul_spirv();
        if (!batched_spirv.empty())
        {
            auto bp_r = VulkanPipeline::create_generic(
                device_.device(), batched_spirv, 3, 6 * sizeof(uint32_t));
            if (bp_r)
                batched_matmul_pipeline_ = std::move(*bp_r);
        }

        // 9c. 创建 rearrange_3d pipeline（2 bindings, 5*4=20B push constants）
        const auto& rearrange_spirv = get_rearrange_3d_spirv();
        if (!rearrange_spirv.empty())
        {
            auto rp_r = VulkanPipeline::create_generic(
                device_.device(), rearrange_spirv, 2, 5 * sizeof(uint32_t));
            if (rp_r)
                rearrange_3d_pipeline_ = std::move(*rp_r);
        }

        // 10. 创建 elementwise_v2 pipeline（4 bindings, 32B push constants）
        const auto& elem_v2_spirv = get_elementwise_v2_spirv();
        if (!elem_v2_spirv.empty())
        {
            auto ep_r = VulkanPipeline::create_generic(
                device_.device(), elem_v2_spirv, 4, 8 * sizeof(uint32_t));
            if (ep_r)
                elementwise_v2_pipeline_ = std::move(*ep_r);
        }

        // 11. 创建 reduce pipeline（2 bindings, 16B push constants）
        const auto& reduce_spirv = get_reduce_spirv();
        if (!reduce_spirv.empty())
        {
            auto rp_r = VulkanPipeline::create_generic(
                device_.device(), reduce_spirv, 2, 4 * sizeof(uint32_t));
            if (rp_r)
                reduce_pipeline_ = std::move(*rp_r);
        }

        // 12. 创建 broadcast pipeline（3 bindings, 16B push constants）
        const auto& broadcast_spirv = get_broadcast_spirv();
        if (!broadcast_spirv.empty())
        {
            auto bp_r = VulkanPipeline::create_generic(
                device_.device(), broadcast_spirv, 3, 4 * sizeof(uint32_t));
            if (bp_r)
                broadcast_pipeline_ = std::move(*bp_r);
        }

        // 13. 创建 transpose pipeline（2 bindings, 8B push constants）
        const auto& transpose_spirv = get_transpose_spirv();
        if (!transpose_spirv.empty())
        {
            auto tp_r = VulkanPipeline::create_generic(
                device_.device(), transpose_spirv, 2, 2 * sizeof(uint32_t));
            if (tp_r)
                transpose_pipeline_ = std::move(*tp_r);
        }

        // 14. 创建 gather pipeline（3 bindings, 12B push constants）
        const auto& gather_spirv = get_gather_spirv();
        if (!gather_spirv.empty())
        {
            auto gp_r = VulkanPipeline::create_generic(
                device_.device(), gather_spirv, 3, 3 * sizeof(uint32_t));
            if (gp_r)
                gather_pipeline_ = std::move(*gp_r);
        }

        // 15. 创建 scatter_add pipeline（3 bindings, 12B push constants）
        const auto& scatter_add_spirv = get_scatter_add_spirv();
        if (!scatter_add_spirv.empty())
        {
            auto sp_r = VulkanPipeline::create_generic(
                device_.device(), scatter_add_spirv, 3, 3 * sizeof(uint32_t));
            if (sp_r)
                scatter_add_pipeline_ = std::move(*sp_r);
        }

        // 16. 创建 RLA 扫描原语 pipelines（手写原语，不进融合注册表）
        const auto& spf_spirv = get_scan_prefix_outer_spirv();
        if (!spf_spirv.empty())
        {
            auto spf_r = VulkanPipeline::create_generic(
                device_.device(), spf_spirv, 8, 7 * sizeof(uint32_t));
            if (spf_r)
                scan_prefix_outer_pipeline_ = std::move(*spf_r);
        }
        const auto& sfs_spirv = get_scan_suffix_outer_spirv();
        if (!sfs_spirv.empty())
        {
            auto sfs_r = VulkanPipeline::create_generic(
                device_.device(), sfs_spirv, 5, 6 * sizeof(uint32_t));
            if (sfs_r)
                scan_suffix_outer_pipeline_ = std::move(*sfs_r);
        }
        // 16b. 通用 d_k（>64）前缀/后缀扫描 pipelines（状态驻全局 scratch）
        const auto& spfg_spirv = get_scan_prefix_outer_gen_spirv();
        if (!spfg_spirv.empty())
        {
            auto spfg_r = VulkanPipeline::create_generic(
                device_.device(), spfg_spirv, 9, 7 * sizeof(uint32_t));
            if (spfg_r)
                scan_prefix_outer_gen_pipeline_ = std::move(*spfg_r);
        }
        const auto& sfsg_spirv = get_scan_suffix_outer_gen_spirv();
        if (!sfsg_spirv.empty())
        {
            auto sfsg_r = VulkanPipeline::create_generic(
                device_.device(), sfsg_spirv, 6, 6 * sizeof(uint32_t));
            if (sfsg_r)
                scan_suffix_outer_gen_pipeline_ = std::move(*sfsg_r);
        }
        const auto& oc_spirv = get_outer_col_spirv();
        if (!oc_spirv.empty())
        {
            auto oc_r = VulkanPipeline::create_generic(
                device_.device(), oc_spirv, 4, 4 * sizeof(uint32_t));
            if (oc_r)
                outer_col_pipeline_ = std::move(*oc_r);
        }
        // 15. 通用精度转换 pipeline（f16↔f32；2 缓冲 + push{count,kind}）
        const auto& cast_spirv = get_cast_spirv();
        if (!cast_spirv.empty())
        {
            auto cast_r = VulkanPipeline::create_generic(
                device_.device(), cast_spirv, 2, 2u * sizeof(uint32_t));
            if (cast_r)
                cast_pipeline_ = std::move(*cast_r);
        }

#ifdef NN_FUSED_REGISTRY_EMBEDDED
        // 16. 注册 AOT 融合 shader pipelines（构建期 scan_exprs 收集 +
        //     gen_fused 合成；每个条目：N 输入 + 1 输出 binding；
        //     push constants = count + cols + 常量池；key = expr_spec_key）
        for (const auto& fs : nn::fused::kFusedShaders)
        {
            if (!fs.spirv || fs.spirv_words == 0)
                continue;  // 空 SPIR-V：跳过（构建配置缺失）
            const std::uint32_t num_bindings =
                static_cast<std::uint32_t>(fs.spec.views.size()) + 1;  // 输入 + 输出
            // 归约 kernel 的 push constants 多 uint rows + uint vector_out；
            // matmul 融合 kernel 多 uint rows + uint mm_k + uint mm_batch；
            // matmul+归约组合再多 uint mm_k + uint mm_batch（6 槽）；
            // 另加 fs.view_param_count 个运行时视图参数槽（RowMod/RotateHalf）
            const std::uint32_t pc_base =
                (fs.reduce_axis >= 0 && fs.has_matmul) ? 6u
              : ((fs.reduce_axis >= 0 || fs.has_matmul) ? 5u : 2u);
            const std::uint32_t pc_uints = pc_base + fs.view_param_count;
            const std::uint32_t pc_size =
                static_cast<std::uint32_t>(pc_uints * sizeof(std::uint32_t) +
                                           sizeof(Scalar) * fs.spec.consts.size() +
                                           sizeof(Scalar) * fs.spec.rparams.size());
            auto fp_r = VulkanPipeline::create_generic(
                device_.device(), std::span<const std::uint32_t>(fs.spirv, fs.spirv_words),
                num_bindings, pc_size);
            if (fp_r)
            {
                fused_pipelines_.emplace(fs.key, std::move(*fp_r));
                fused_reduce_axis_.emplace(fs.key, fs.reduce_axis);
                fused_has_matmul_.emplace(fs.key, fs.has_matmul != 0);
                fused_view_param_counts_.emplace(fs.key, fs.view_param_count);
                fused_rparam_counts_.emplace(fs.key, fs.rparam_count);
                fused_vec_width_.emplace(fs.key, fs.vec_width);
            }
        }
#endif

        // ── D4：查询 GPU f16 硬件能力（§7.1）─────────────────────────────
        // shaderFloat16：Vulkan 1.2 核心特性 / VK_KHR_shader_float16_int8 扩展。
        // 查询 VkPhysicalDeviceShaderFloat16Int8Features（扩展 pNext 链）。
        // 未找到扩展 → has_shader_float16_ = false（兼容路径，变体①）。
        {
            VkPhysicalDeviceShaderFloat16Int8Features f16_features{};
            f16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

            VkPhysicalDeviceFeatures2 device_features{};
            device_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            device_features.pNext = &f16_features;

            vkGetPhysicalDeviceFeatures2(device_.physical_device(), &device_features);
            has_shader_float16_ = (f16_features.shaderFloat16 == VK_TRUE);
        }

        initialized_ = true;
        return {};
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] bool gpu_available() const noexcept { return initialized_; }
    [[nodiscard]] bool has_tiled_pipeline() const noexcept { return matmul_tiled_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_batched_matmul_pipeline() const noexcept { return batched_matmul_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_rearrange_3d_pipeline() const noexcept { return rearrange_3d_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_elementwise_v2_pipeline() const noexcept { return elementwise_v2_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_reduce_pipeline() const noexcept { return reduce_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_broadcast_pipeline() const noexcept { return broadcast_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_transpose_pipeline() const noexcept { return transpose_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_gather_pipeline() const noexcept { return gather_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_scatter_add_pipeline() const noexcept { return scatter_add_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_scan_prefix_outer_pipeline() const noexcept { return scan_prefix_outer_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_scan_suffix_outer_pipeline() const noexcept { return scan_suffix_outer_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_scan_prefix_outer_gen_pipeline() const noexcept { return scan_prefix_outer_gen_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_scan_suffix_outer_gen_pipeline() const noexcept { return scan_suffix_outer_gen_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_outer_col_pipeline() const noexcept { return outer_col_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_cast_pipeline() const noexcept { return cast_pipeline_.handle() != VK_NULL_HANDLE; }

    // ── D4：f16 硬件能力查询（§7.1）────────────────────────────────────
    [[nodiscard]] bool has_shader_float16() const noexcept { return has_shader_float16_; }

    [[nodiscard]] VulkanDevice& device() noexcept { return device_; }
    [[nodiscard]] MemoryPool& memory_pool() noexcept { return *memory_pool_; }
    [[nodiscard]] MemoryPool& transient_pool() noexcept { return *transient_pool_; }
    // 瞬态/持久分池选择器：batch 录制期→瞬态池，否则→持久池（纯组织性）。
    // 供 GpuTensorT 分配（create_empty / from_matrix / create_host_visible_empty）
    // 使用，使批量内的临时/激活与构建期的参数/权重分池驻留。
    [[nodiscard]] MemoryPool& alloc_pool() noexcept
    {
        return batch_mode_ ? *transient_pool_ : *memory_pool_;
    }
    [[nodiscard]] StagingRing& staging_ring() noexcept { return *staging_ring_; }
    [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
    [[nodiscard]] VkDescriptorPool gpu_tensor_pool() const noexcept { return gpu_tensor_pool_; }

    // ── 显存回收（L2，P0-1 非阻塞版）：归还完全空闲的内存池底材 ─────
    // 非阻塞 reap 已完成帧（vkGetFenceStatus 查询，不等待）：销毁其延迟
    // buffer + 释放描述符集，使底材可被归还；未完成的帧留给环槽复用 /
    // 下次 drain reap——不在 step 边界阻塞流水线（GPU 仍在执行本 step
    // 帧时，host 可继续录制下一 step）。
    [[nodiscard]] Result<void> release_idle_pool_blocks()
    {
        if (!initialized_ || !memory_pool_)
            return {};
        for (std::size_t i = 0; i < frames_.size(); ++i)
        {
            if (!frames_[i].in_flight)
                continue;
            // 非阻塞查询：vkWaitForFences(timeout=0) 返回 VK_SUCCESS
            // （已就绪）或 VK_TIMEOUT（未就绪）。VK_NOT_READY 是
            // vkGetFenceStatus 的返回值，vkWaitForFences 不使用它。
            const VkResult rs =
                vkWaitForFences(device_.device(), 1, &frames_[i].fence,
                                VK_TRUE, 0);
            if (rs == VK_ERROR_DEVICE_LOST)
            {
                // 设备丢失（TDR 触发）：GPU 已死亡，无法恢复
                device_lost_ = true;
                return std::unexpected(Error{
                    "GPU 设备丢失 (VK_ERROR_DEVICE_LOST): Windows TDR 已重置 GPU 驱动。"
                    "\n模型已自动保存，请使用 --resume <save-path> 重启训练。"
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)"});
            }
            if (rs != VK_TIMEOUT && rs != VK_SUCCESS)
                return std::unexpected(Error{
                    std::string("GPU fence 非阻塞查询失败: Vulkan error ") +
                    std::to_string(static_cast<int>(rs))});
            if (rs == VK_SUCCESS)
            {
                auto rr = reap_frame(i, /*wait=*/false);
                if (!rr)
                    return std::unexpected(rr.error());
            }
        }
        memory_pool_->release_idle_blocks();
        if (transient_pool_)
            transient_pool_->release_idle_blocks();
        return {};
    }


    // ══════════════════════════════════════════════════════════════════
    // Command Buffer Batching API（多帧流水线 P0-1）
    // ══════════════════════════════════════════════════════════════════
    // 用法：
    //   backend.begin_batch();
    //   // ... 多次 matmul_gpu / elementwise_gpu / layernorm_gpu 调用 ...
    //   backend.end_batch();
    // 效果：所有操作录制到同一个 command buffer，一次提交（**不等待**）；
    // host 继续录制下一段，GPU 在队列上先行执行（host 录制与 GPU 执行
    // 重叠）。真正要读 GPU 数据时调 wait_in_flight()（to_matrix 内部）。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] bool in_batch() const noexcept { return batch_mode_; }

    // ── 取下一帧并开始录制（begin_batch / flush_batch 共用）──────────
    // 环槽复用前等待：该帧若仍在飞行，等其 fence（这是唯一的常规等待点；
    // 环深度 N 限制 host 至多超前 N 帧）。
    [[nodiscard]] Result<void> start_frame()
    {
        const std::size_t i = frame_next_;
        frame_next_ = (i + 1) % PIPELINE_FRAMES;

        auto rr = reap_frame(i, /*wait=*/true);
        if (!rr)
        {
            batch_mode_ = false;
            return std::unexpected(rr.error());
        }

        auto& f = frames_[i];
        auto r = detail::vk_check(vkResetCommandBuffer(f.cmd, 0), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(f.cmd, &begin_info), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        batch_cmd_ = f.cmd;
        batch_frame_ = i;
        last_active_frame_ = i;
        batch_has_ops_ = false;
        batch_mode_ = true;
        return {};
    }

    // ── 提交当前帧（不等待）────────────────────────────────────────────
    // 空帧不提交（无 op 的帧跳过，省一次 vkQueueSubmit + 一个环槽）。
    [[nodiscard]] Result<void> submit_frame_no_wait()
    {
        auto& f = frames_[batch_frame_];
        if (!batch_has_ops_)
            return {};  // 空帧：不提交

        // reap 已确保 fence 处于 unsignaled（或从未提交）状态
        auto r = detail::vk_check(
            vkResetFences(device_.device(), 1, &f.fence), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 跨 submit 数据依赖：帧命令的输入可能刚由 batch 内的 from_matrix
        // 独立上传提交写入——等 in-flight region 信号量（GPU 队列内等待，
        // host 不阻塞）
        const auto sw = collect_staging_waits();

        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &f.cmd;
            VkTimelineSemaphoreSubmitInfo timeline_info{};
            attach_staging_waits(submit_info, sw, timeline_info);
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, f.fence),
                __FILE__, __LINE__);
        }
        if (!r) return std::unexpected(r.error());

        f.in_flight = true;
        last_active_frame_ = batch_frame_;
        return {};
    }

    [[nodiscard]] Result<void> begin_batch()
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (batch_mode_)
            return std::unexpected(Error{"Already in batch mode"});
        return start_frame();
    }

    // ── 结束 batch：提交当前帧，**不等待**（P0-1）─────────────────────
    // 帧在队列上执行；其延迟销毁 / 描述符集在 reap 时释放。
    // 设备丢失等错误在 wait_in_flight / 环槽复用 reap 时浮出。
    [[nodiscard]] Result<void> end_batch()
    {
        if (!batch_mode_)
            return {};  // 容错：非 batch 模式时 no-op

        // 1. 结束录制
        auto r = detail::vk_check(vkEndCommandBuffer(batch_cmd_), __FILE__, __LINE__);
        if (!r)
        {
            batch_mode_ = false;
            return std::unexpected(r.error());
        }

        // 2. 提交（不等待）
        r = submit_frame_no_wait();
        batch_mode_ = false;
        return r;
    }

    // ── 批处理中点刷新（防 TDR，P0-1 不等待版）────────────────────────
    // 提交当前帧（不等待），然后开始录制下一帧。用于拆分大 batch（如
    // forward 与 backward 之间），避免单次提交时间过长触发 Windows TDR。
    // 当前帧为空（无 op 录制）时 no-op——不提交、不换帧（避免浪费环槽
    // 与 submit，如 to_matrix 的 drain 已换过帧的场景）。
    [[nodiscard]] Result<void> flush_batch()
    {
        if (!batch_mode_)
            return {};  // 非 batch 模式时 no-op
        if (!batch_has_ops_)
            return {};  // 空帧：无需提交，继续在同一帧录制

        auto r = detail::vk_check(vkEndCommandBuffer(batch_cmd_), __FILE__, __LINE__);
        if (!r)
        {
            batch_mode_ = false;
            return std::unexpected(r.error());
        }

        r = submit_frame_no_wait();  // 不等待
        if (!r)
        {
            batch_mode_ = false;
            return std::unexpected(r.error());
        }

        // 仍处于 batch 模式：开始录制下一帧
        return start_frame();
    }

    // ── "真正要结果"的阻塞点（P0-1）───────────────────────────────────
    // 等待所有在飞帧完成并 reap（延迟销毁 + 描述符集释放）。
    // to_matrix / copy_from 读 GPU 内存前必须调用：要读的数据可能刚由
    // 在飞帧写入，队列 FIFO 只保证顺序、不保证完成。
    [[nodiscard]] Result<void> wait_in_flight()
    {
        for (std::size_t i = 0; i < frames_.size(); ++i)
        {
            if (frames_[i].in_flight)
            {
                auto r = reap_frame(i, /*wait=*/true);
                if (!r)
                    return std::unexpected(r.error());
            }
        }
        // 所有帧已完成：兜底清理剩余延迟销毁（正常应为空）
        flush_pending_destroys();
        return {};
    }

    // ── 阻塞式上传：CPU → GPU（支持分块传输大矩阵）─────────────────────
    // 当数据超过 staging region 大小时，自动分块上传。
    // 每块大小不超过 staging region，通过多次 memcpy + vkCmdCopyBuffer 完成。
    // 元素类型由 P 决定（f32=4B / f16=2B），memcpy 为字节级操作，§6.5。
    template <Precision P>
    [[nodiscard]] Result<void> upload_blocking(
        GpuTensorT<P>& dst, std::span<const elem<P>> cpu_data)
    {
        using ElemType = elem<P>;
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!dst.valid())
            return std::unexpected(Error{"Invalid destination GpuTensor"});

        const std::size_t elem_count = dst.rows() * dst.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Upload size mismatch"});

        const std::size_t total_bytes = elem_count * sizeof(ElemType);
        const std::size_t staging_cap = staging_ring_->region_size();

        // ── 小矩阵快速路径：单次传输 ──────────────────────────────────
        // region 专属 command buffer（P0-1 修复）：不再每次上传
        // vkAllocateCommandBuffers + 提交后立即可复用/释放——VUID-
        // vkFreeCommandBuffers-pCommandBuffers-00058 禁止释放 pending
        // （已提交、fence 未 signal）的 command buffer；旧代码"submit 后
        // 立即 free"是规范违规（实测导致驱动通道排序失效、fence 提前
        // signal、跨 submit 乱序）。acquire 已等在飞 fence → 该 cmd 的
        // 上一次使用完成（invalid 状态）→ vkResetCommandBuffer 复用合法。
        if (total_bytes <= staging_cap)
        {
            auto ri = staging_ring_->acquire();

            auto r = staging_ring_->upload(ri, cpu_data, 0);
            if (!r) return r;

            auto cmd = staging_ring_->command_buffer(ri);

            r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) return r;

            VkBufferCopy cp{0, 0, total_bytes};
            vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl(), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) return r;

            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                // 跨 submit 数据依赖（P0-1 修复）：时间线信号量 signal 本次
                // 上传的 value。读取"本 copy 写入的 buffer"的后续 submit 必须
                // wait 该 value（见 collect_staging_waits）。
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                VkSemaphore sem = VK_NULL_HANDLE;
                std::uint64_t signal_value = 0;
                attach_upload_signal(submit_info, *staging_ring_, ri,
                                     timeline_info, sem, signal_value);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
            }
            if (!r) return r;

            // P0-1：提交后不等待——host 可立即继续录制。标记 region
            // in use：下次 acquire 到该 region 时等其 fence（staging 安全
            // + 等待后该 cmd 离开 pending，可 reset 复用）。**不释放 cmd**
            // （pending 状态，见上方 VUID 注释）。
            staging_ring_->mark_in_flight(ri);
            return {};
        }

        // ── 大矩阵分块上传 ───────────────────────────────────────────
        // 每块大小对齐到 sizeof(ElemType)，确保元素边界对齐
        const std::size_t chunk_bytes = (staging_cap / sizeof(ElemType)) * sizeof(ElemType);
        std::size_t dst_offset = 0;

        while (dst_offset < total_bytes)
        {
            const std::size_t this_chunk = std::min(chunk_bytes, total_bytes - dst_offset);
            const std::size_t this_elems = this_chunk / sizeof(ElemType);
            const std::size_t src_elem_offset = dst_offset / sizeof(ElemType);

            // 1. 获取 staging region 并上传当前块
            auto ri = staging_ring_->acquire();
            auto r = staging_ring_->upload(
                ri,
                std::span<const ElemType>(cpu_data.data() + src_elem_offset, this_elems),
                0);
            if (!r) return r;

            // 2. 录制 copy 命令（region 专属 cmd buffer：禁止 pending 释放，
            //    见小矩阵路径的 VUID 注释；acquire 已等在飞 fence，reset 合法）
            auto cmd = staging_ring_->command_buffer(ri);

            r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) return r;

            VkBufferCopy cp{0, dst_offset, this_chunk};
            vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl(), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) return r;

            // 3. 提交（P0-1：不等待；region 标记 in use，下次 acquire 到该
            //    region 时等其 fence；不释放 cmd——pending 状态）+ 信号本
            //    region 信号量（跨 submit 数据依赖，见 collect_staging_waits）
            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                // 跨 submit 数据依赖：时间线信号量 signal 本次上传的 value
                // （读取本 copy 写入 buffer 的后续 submit 会 wait 该 value）
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                VkSemaphore sem = VK_NULL_HANDLE;
                std::uint64_t signal_value = 0;
                attach_upload_signal(submit_info, *staging_ring_, ri,
                                     timeline_info, sem, signal_value);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
            }
            if (!r) return r;

            staging_ring_->mark_in_flight(ri);
            dst_offset += this_chunk;
        }

        return {};
    }

    // ── 阻塞式下载：GPU → CPU（支持分块传输大矩阵）─────────────────────
    // 当数据超过 staging region 大小时，自动分块下载。
    // 元素类型由 P 决定（f32=4B / f16=2B），memcpy 为字节级操作，§6.5。
    template <Precision P>
    [[nodiscard]] Result<void> download_blocking(
        const GpuTensorT<P>& src, std::span<elem<P>> cpu_data)
    {
        using ElemType = elem<P>;
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!src.valid())
            return std::unexpected(Error{"Invalid source GpuTensor"});

        const std::size_t elem_count = src.rows() * src.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Download size mismatch"});

        const std::size_t total_bytes = elem_count * sizeof(ElemType);
        const std::size_t staging_cap = staging_ring_->region_size();

        // ── 小矩阵快速路径：单次传输 ──────────────────────────────────
        if (total_bytes <= staging_cap)
        {
            auto ri = staging_ring_->acquire();

            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            auto r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            VkBufferCopy cp{0, 0, total_bytes};
            vkCmdCopyBuffer(cmd, src.buffer().impl(), staging_ring_->buffer(ri), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            // 跨 submit 数据依赖：源 buffer 可能刚由在飞上传的 copy 写入
            // （from_matrix → to_matrix 无中间 op 即此场景）——等 in-flight
            // region 信号量（本下载自用的 region 已在 acquire 中等待并重置，
            // 不在 in-flight 集合内）。时间线信号量允许同一 value 被多个
            // submit 等待，故这里与前面的 matmul 帧重复等待也合法。
            const auto sw = collect_staging_waits();

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                attach_staging_waits(submit_info, sw, timeline_info);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
                __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            r = staging_ring_->download(ri, cpu_data, 0);

            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            return r;
        }

        // ── 大矩阵分块下载 ───────────────────────────────────────────
        const std::size_t chunk_bytes = (staging_cap / sizeof(ElemType)) * sizeof(ElemType);
        std::size_t src_offset = 0;

        while (src_offset < total_bytes)
        {
            const std::size_t this_chunk = std::min(chunk_bytes, total_bytes - src_offset);
            const std::size_t this_elems = this_chunk / sizeof(ElemType);
            const std::size_t dst_elem_offset = src_offset / sizeof(ElemType);

            // 1. 获取 staging region
            auto ri = staging_ring_->acquire();

            // 2. 录制 copy 命令：GPU → staging
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            auto r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            VkBufferCopy cp{src_offset, 0, this_chunk};
            vkCmdCopyBuffer(cmd, src.buffer().impl(), staging_ring_->buffer(ri), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            // 3. 提交并等待（+ 跨 submit 数据依赖：等 in-flight region
            //    信号量，源 buffer 可能刚由在飞上传的 copy 写入）
            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            const auto sw = collect_staging_waits();

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                attach_staging_waits(submit_info, sw, timeline_info);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
                __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            // 4. 从 staging 下载当前块到 CPU
            r = staging_ring_->download(
                ri,
                std::span<ElemType>(cpu_data.data() + dst_elem_offset, this_elems),
                0);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            src_offset += this_chunk;
        }

        return {};
    }

    // ── 通用 compute dispatch（P1-13 matmul 去重核心）──────────────────
    // 统一的 descriptor/cmd/barrier/dispatch/submit 样板，供 matmul /
    // batched_matmul / rearrange 等所有"N 输入 + 1 输出"的 compute kernel 复用。
    //
    // 参数：
    //   pipeline : 目标 compute pipeline
    //   inputs   : 输入张量（写入 binding 0..n-1）
    //   output   : 输出张量（写入最后一个 binding）
    //   pc       : push constants 字节流（按 pipeline_layout 布局）
    //   wg_x/y/z : dispatch 工作组数（三维）
    [[nodiscard]] Result<void> dispatch_compute(
        const VulkanPipeline& pipeline, std::span<const GpuTensor> inputs,
        const GpuTensor& output, const std::vector<std::uint8_t>& pc,
        std::uint32_t wg_x, std::uint32_t wg_y, std::uint32_t wg_z)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        auto ds_r = alloc_desc_set(pipeline.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        const std::size_t n = inputs.size() + 1;
        std::vector<VkDescriptorBufferInfo> buf_infos(n);
        std::vector<VkWriteDescriptorSet> writes(n);
        for (std::size_t i = 0; i < inputs.size(); ++i)
            buf_infos[i] = {inputs[i].buffer().impl(), 0, VK_WHOLE_SIZE};
        buf_infos[n - 1] = {output.buffer().impl(), 0, VK_WHOLE_SIZE};
        for (std::size_t i = 0; i < n; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(),
            static_cast<uint32_t>(n), writes.data(), 0, nullptr);

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        std::vector<VkBuffer> in_bufs;
        in_bufs.reserve(inputs.size());
        for (const auto& t : inputs)
            in_bufs.push_back(t.buffer().impl());
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);
        if (!pc.empty())
            vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                static_cast<uint32_t>(pc.size()), pc.data());
        vkCmdDispatch(cmd, wg_x, wg_y, wg_z);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // ── 纯 GPU 矩阵乘法（阻塞等待完成）──────────────────────────────
    // batch_mode_ 时：录制到 batch_cmd_，不提交不等待
    // transA: A 存储为 (K,M)，按 A^T 使用
    // transB: B 存储为 (N,K)，按 B^T 使用
    [[nodiscard]] Result<GpuTensor> matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t transA = 0, uint32_t transB = 0)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        // 计算有效维度（考虑转置）
        const auto M = static_cast<uint32_t>(transA ? A.cols() : A.rows());
        const auto K = static_cast<uint32_t>(transA ? A.rows() : A.cols());
        const auto N = static_cast<uint32_t>(transB ? B.rows() : B.cols());
        const auto K_B = static_cast<uint32_t>(transB ? B.cols() : B.rows());
        if (K != K_B)
            return std::unexpected(Error{"Dimension mismatch"});

        // 1. 分配输出 Tensor
        auto C_res = GpuTensor::create_empty(M, N, *this);
        if (!C_res)
            return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);

        // 2. 选择 Pipeline（优先使用粗化分块版本）
        const bool use_tiled = has_tiled_pipeline();
        auto& pipeline = use_tiled ? matmul_tiled_pipeline_ : matmul_pipeline_;

        // 3. 复用通用 dispatch：2 输入 + 1 输出，push {M,N,K,transA,transB}
        //   Dispatch：按 shader 输出块尺寸计算工作组数
        //     matmul.comp（naive）：16×16 线程网格 = 16×16 输出块
        //     matmul_tiled.comp：64×64 输出块（BM/BN）
        //   ⚠ 曾误用 WORKGROUP_SIZE=16 统一计算 → tiled 版 dispatch 出 16 倍
        //   冗余工作组（每 16×16 一个组而非 64×64），GPU 做 16 倍无效计算，
        //   matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。此处按实际块尺寸修复。
        const uint32_t tile = use_tiled ? 64u : 16u;
        const uint32_t push_data[5] = {M, N, K, transA, transB};
        std::vector<std::uint8_t> pc(sizeof(push_data));
        std::memcpy(pc.data(), push_data, sizeof(push_data));

        std::vector<GpuTensor> inputs{A, B};
        auto r = dispatch_compute(pipeline, inputs, C, pc,
            (N + tile - 1u) / tile, (M + tile - 1u) / tile, 1u);
        if (!r)
            return std::unexpected(r.error());
        return C;
    }

    // ── 纯 GPU 批量矩阵乘法 ──────────────────────────────────────────
    // 对每个 batch b 计算 C_b = alpha * op(A_b, B_b)，结果垂直堆叠为 (batch*M, N)
    // A: (batch * A_rows_per_batch, A_cols)，B: (batch * B_rows_per_batch, B_cols)
    // batch 步长由 shader 从 M*K / K*N / M*N 推导，无需额外传入
    // alpha: 输出缩放系数（cuBLAS sgemm 语义），在 shader 写出时一次完成
    [[nodiscard]] Result<GpuTensor> batched_matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t batch,
        uint32_t transA = 0, uint32_t transB = 0,
        float alpha = 1.0f)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_gpu: batch must be > 0"});
        if (!has_batched_matmul_pipeline())
            return std::unexpected(Error{"batched_matmul_gpu: pipeline not available"});

        // 校验：A.rows() 必须能被 batch 整除
        if (A.rows() % batch != 0 || B.rows() % batch != 0)
            return std::unexpected(Error{"batched_matmul_gpu: rows not divisible by batch"});

        // 计算每个 batch 的逻辑维度
        const auto a_rows_per = static_cast<uint32_t>(A.rows() / batch);
        const auto b_rows_per = static_cast<uint32_t>(B.rows() / batch);
        const auto M = transA ? static_cast<uint32_t>(A.cols()) : a_rows_per;
        const auto K = transA ? a_rows_per : static_cast<uint32_t>(A.cols());
        const auto K_B = transB ? static_cast<uint32_t>(B.cols()) : b_rows_per;
        const auto N = transB ? b_rows_per : static_cast<uint32_t>(B.cols());
        if (K != K_B)
            return std::unexpected(Error{"batched_matmul_gpu: K dimension mismatch"});

        // 1. 分配输出 Tensor: (batch * M, N)
        auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(batch) * M, N, *this);
        if (!C_res)
            return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);

        // 2. 复用通用 dispatch：2 输入 + 1 输出
        //   Push constants: M, N, K, transA, transB, alpha（24B，含输出缩放系数）
        //   shader 采用 64×64 寄存器分块（与 matmul_tiled 相同），
        //   每个 WorkGroup 计算 64×64 输出块，Z 维度索引 batch
        struct PushDataBmm {
            uint32_t M, N, K, transA, transB;
            float alpha;
        } push{M, N, K, transA, transB, alpha};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));

        // Dispatch: X/Y 覆盖 64×64 输出块，Z 维度 = batch
        constexpr uint32_t BM = 64, BN = 64;
        std::vector<GpuTensor> inputs{A, B};
        auto r = dispatch_compute(batched_matmul_pipeline_, inputs, C, pc,
            (N + BN - 1) / BN, (M + BM - 1) / BM, batch);
        if (!r)
            return std::unexpected(r.error());
        return C;
    }

    // ══════════════════════════════════════════════════════════════════
    // RLA 扫描原语（手写原语，形状契约见 compute_engine.hpp 扫描级原语注释）
    // 1 workgroup/头：dispatch (1, B*H, 1)；标量块在头块内逐行重复存放
    // ══════════════════════════════════════════════════════════════════

    // ── 前缀扫描 + matvec 读出：输出 (rows*5, seq) ─────────────────────
    [[nodiscard]] Result<GpuTensor> scan_prefix_outer_gpu(
        const GpuTensor& K, const GpuTensor& V, const GpuTensor& P, const GpuTensor& R,
        const GpuTensor& A0, const GpuTensor& B0, bool has_state,
        uint32_t dk, uint32_t heads, bool causal,
        const GpuTensor& boundary, bool has_bnd)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (dk == 0u || heads == 0u)
            return std::unexpected(Error{"scan_prefix_outer_gpu: dk/heads must be > 0"});
        const bool generic = dk > 64u;  // 通用路径（状态驻全局 scratch，无 dk 上限）
        if (!generic && !has_scan_prefix_outer_pipeline())
            return std::unexpected(Error{"scan_prefix_outer_gpu: pipeline not available"});
        if (generic && !has_scan_prefix_outer_gen_pipeline())
            return std::unexpected(Error{"scan_prefix_outer_gpu: generic pipeline not available"});
        const auto rows = static_cast<uint32_t>(K.rows());
        if (rows % (dk * heads) != 0)
            return std::unexpected(Error{"scan_prefix_outer_gpu: rows not divisible by H*dk"});
        if (V.rows() != rows || V.cols() != K.cols() ||
            P.rows() != rows || P.cols() != K.cols() ||
            R.rows() != rows || R.cols() != K.cols())
            return std::unexpected(Error{"scan_prefix_outer_gpu: K/V/P/R shape mismatch"});
        if (has_state && (A0.rows() != heads * dk || A0.cols() != dk ||
                          B0.rows() != heads * dk || B0.cols() != dk))
            return std::unexpected(Error{"scan_prefix_outer_gpu: A0/B0 must be (H*dk, dk)"});
        const auto seq = static_cast<uint32_t>(K.cols());
        if (has_bnd)
        {
            if (boundary.rows() != 1 || boundary.cols() != (rows / (dk * heads)) * seq)
                return std::unexpected(Error{"scan_prefix_outer_gpu: boundary must be (1, B*seq)"});
        }
        const auto BH = rows / dk;
        auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(rows) * 5, seq, *this);
        if (!C_res)
            return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);
        struct PushPrefix { uint32_t dk, heads, seq, causal, has_state, has_bnd, rows; };
        PushPrefix push{dk, heads, seq, causal ? 1u : 0u, has_state ? 1u : 0u,
                        has_bnd ? 1u : 0u, rows};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));
        if (generic)
        {
            // 通用 d_k > 64：状态驻全局 scratch（B*H*2*dk²），共享内存 O(1)，
            // 对任意 dk 安全。GpuTensor 析构走 pending_destroys 延迟归还，
            // batch 录制期被引用（descriptor set）也不会过早释放。
            auto St_res = GpuTensor::create_empty(
                static_cast<std::size_t>(BH) * 2u * dk * dk, 1u, *this);
            if (!St_res)
                return std::unexpected(St_res.error());
            GpuTensor St = std::move(*St_res);
            std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary, St};
            auto r = dispatch_compute(scan_prefix_outer_gen_pipeline_, inputs, C, pc, 1u, BH, 1u);
            if (!r)
                return std::unexpected(r.error());
            return C;
        }
        std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary};
        auto r = dispatch_compute(scan_prefix_outer_pipeline_, inputs, C, pc, 1u, BH, 1u);
        if (!r)
            return std::unexpected(r.error());
        return C;
    }

    // ── 后缀扫描 + matvec 读出：输出 (rows*3, seq) ─────────────────────
    [[nodiscard]] Result<GpuTensor> scan_suffix_outer_gpu(
        const GpuTensor& D, const GpuTensor& X, const GpuTensor& Y,
        uint32_t dk, uint32_t heads, bool causal,
        const GpuTensor& boundary, bool has_bnd)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (dk == 0u || heads == 0u)
            return std::unexpected(Error{"scan_suffix_outer_gpu: dk/heads must be > 0"});
        const bool generic = dk > 64u;  // 通用路径（状态驻全局 scratch，无 dk 上限）
        if (!generic && !has_scan_suffix_outer_pipeline())
            return std::unexpected(Error{"scan_suffix_outer_gpu: pipeline not available"});
        if (generic && !has_scan_suffix_outer_gen_pipeline())
            return std::unexpected(Error{"scan_suffix_outer_gpu: generic pipeline not available"});
        const auto rows = static_cast<uint32_t>(X.rows());
        if (rows % (dk * heads) != 0)
            return std::unexpected(Error{"scan_suffix_outer_gpu: X rows not divisible by H*dk"});
        if (D.rows() != rows * dk || D.cols() != X.cols() ||
            Y.rows() != rows || Y.cols() != X.cols())
            return std::unexpected(Error{"scan_suffix_outer_gpu: D/X/Y shape mismatch"});
        const auto seq = static_cast<uint32_t>(X.cols());
        if (has_bnd)
        {
            if (boundary.rows() != 1 || boundary.cols() != (rows / (dk * heads)) * seq)
                return std::unexpected(Error{"scan_suffix_outer_gpu: boundary must be (1, B*seq)"});
        }
        const auto BH = rows / dk;
        auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(rows) * 3, seq, *this);
        if (!C_res)
            return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);
        struct PushSuffix { uint32_t dk, heads, seq, causal, has_bnd, rows; };
        PushSuffix push{dk, heads, seq, causal ? 1u : 0u, has_bnd ? 1u : 0u, rows};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));
        if (generic)
        {
            auto St_res = GpuTensor::create_empty(
                static_cast<std::size_t>(BH) * dk * dk, 1u, *this);
            if (!St_res)
                return std::unexpected(St_res.error());
            GpuTensor St = std::move(*St_res);
            std::vector<GpuTensor> inputs{D, X, Y, boundary, St};
            auto r = dispatch_compute(scan_suffix_outer_gen_pipeline_, inputs, C, pc, 1u, BH, 1u);
            if (!r)
                return std::unexpected(r.error());
            return C;
        }
        std::vector<GpuTensor> inputs{D, X, Y, boundary};
        auto r = dispatch_compute(scan_suffix_outer_pipeline_, inputs, C, pc, 1u, BH, 1u);
        if (!r)
            return std::unexpected(r.error());
        return C;
    }

    // ── 逐列外积：输出 (rows*dk, seq) ──────────────────────────────────
    [[nodiscard]] Result<GpuTensor> outer_col_gpu(
        const GpuTensor& P, const GpuTensor& R, const GpuTensor& S,
        uint32_t dk, bool has_scale)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_outer_col_pipeline())
            return std::unexpected(Error{"outer_col_gpu: pipeline not available"});
        if (dk == 0u)
            return std::unexpected(Error{"outer_col_gpu: dk must be > 0"});
        const auto rows = static_cast<uint32_t>(P.rows());
        if (rows % dk != 0)
            return std::unexpected(Error{"outer_col_gpu: rows not divisible by dk"});
        if (R.rows() != rows || R.cols() != P.cols())
            return std::unexpected(Error{"outer_col_gpu: P/R shape mismatch"});
        if (has_scale && (S.rows() != rows || S.cols() != P.cols()))
            return std::unexpected(Error{"outer_col_gpu: S must be (B*H*dk, seq)"});
        const auto seq = static_cast<uint32_t>(P.cols());
        auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(rows) * dk, seq, *this);
        if (!C_res)
            return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);
        struct PushOuter { uint32_t dk, seq, rows, has_scale; };
        PushOuter push{dk, seq, rows, has_scale ? 1u : 0u};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));
        std::vector<GpuTensor> inputs{P, R, S};
        const auto total = static_cast<uint32_t>(rows) * dk * seq;
        auto r = dispatch_compute(outer_col_pipeline_, inputs, C, pc,
            (total + 255u) / 256u, 1u, 1u);
        if (!r)
            return std::unexpected(r.error());
        return C;
    }

    // ══════════════════════════════════════════════════════════════════
    // 纯 GPU 原语方法（纯 GPU 架构核心）
    //
    // 所有方法遵循同一模式：
    //   1. 分配输出 Tensor（如有输出）
    //   2. 分配描述符集，写入 binding
    //   3. 获取 command buffer（batch 共享 / 独立分配）
    //   4. 录制：输入屏障 → bind pipeline → push constants → dispatch → 输出屏障
    //   5. 独立模式：end → submit → wait → cleanup
    // ══════════════════════════════════════════════════════════════════

    // ── 辅助：分配描述符集 ──────────────────────────────────────────────
    [[nodiscard]] Result<VkDescriptorSet> alloc_desc_set(VkDescriptorSetLayout layout)
    {
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        desc_alloc.pSetLayouts = &layout;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());
        // batch 模式：描述符集被当前帧的已录制命令引用，归属该帧，
        // 帧的 fence 信号后（reap_frame）才释放
        if (batch_mode_) frames_[batch_frame_].desc_sets.push_back(desc_set);
        return desc_set;
    }

    // ── 辅助：跨 submit 数据依赖（P0-1 修复核心）────────────────────────
    // 背景：单队列 FIFO 只是执行顺序保证。实测本驱动（NVIDIA + Windows）
    // 下，消费方 submit（download/matmul/batch 帧）紧跟上传 submit
    // （<~2ms）时，消费方 GPU 操作会读到上传写入的旧值（零）；host 侧
    // sleep ≥5ms 或 fence 等待可规避——即隐式跨 submit 数据依赖不可靠，
    // 必须用队列级原语显式建立：
    //   上传 submit:  pSignalSemaphores = {region 信号量} + value
    //   消费 submit:  pWaitSemaphores   = {所有 in-flight region 信号量}
    //                 pWaitSemaphoreValues = {各自的 value}
    //                 pWaitDstStageMask  = {ALL_COMMANDS}（waitSemaphoreCount>0
    //                                        时规范要求非空，见
    //                                        VUID-VkSubmitInfo-pWaitDstStageMask）
    // **必须用时间线信号量**：同一 value 可被任意多个 submit 等待且等待已达成
    // 的 value 是 no-op。二进制信号量一次 signal 只能被一个 wait 消费，第二
    // 个消费者会永久阻塞（VUID-vkQueueSubmit-pWaitSemaphores-03238），AMD
    // 老驱动实测直接死锁。
    // host 永不阻塞（P0-1 流水线收益保留），GPU 在队列内等待数据就绪。
    // 设备不支持时间线信号量时不创建信号量，改由 drain_in_flight() 在 host
    // 侧阻塞兜底（正确性优先）。
    struct StagingWait
    {
        std::vector<VkSemaphore> sems;
        std::vector<VkPipelineStageFlags> stages;
        std::vector<std::uint64_t> values;
        [[nodiscard]] bool any() const noexcept { return !sems.empty(); }
    };
    [[nodiscard]] StagingWait collect_staging_waits()
    {
        StagingWait w;
        if (!staging_ring_)
            return w;
        if (!staging_ring_->timeline())
        {
            // 回退路径：host 等在飞上传完成（阻塞），后续 submit 无跨 submit
            // 依赖，无需信号量
            staging_ring_->drain_in_flight();
            return w;
        }
        for (std::size_t i = 0; i < staging_ring_->num_regions(); ++i)
        {
            if (!staging_ring_->in_flight(i))
                continue;
            w.sems.push_back(staging_ring_->semaphore(i));
            w.stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            w.values.push_back(staging_ring_->signal_value(i));
        }
        return w;
    }

    // 把跨 submit 等待挂到 submit 上（pNext = VkTimelineSemaphoreSubmitInfo）
    static void attach_staging_waits(
        VkSubmitInfo& si, const StagingWait& sw, VkTimelineSemaphoreSubmitInfo& tsi)
    {
        if (!sw.any())
            return;
        si.waitSemaphoreCount = static_cast<std::uint32_t>(sw.sems.size());
        si.pWaitSemaphores = sw.sems.data();
        si.pWaitDstStageMask = sw.stages.data();
        tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsi.waitSemaphoreValueCount = static_cast<std::uint32_t>(sw.values.size());
        tsi.pWaitSemaphoreValues = sw.values.data();
        si.pNext = &tsi;
    }

    // 上传 submit 的信号挂载（时间线信号量模式）。
    // 返回 false = 当前设备非时间线模式：不 signal，消费者走 drain 回退。
    // 调用方提供的 sem_storage/value_storage 必须活到 vkQueueSubmit（
    // pSignalSemaphores/pSignalSemaphoreValues 是借用指针）。
    // 必须在 mark_in_flight() 之前调用：消费者只认"in-flight region 的
    // signal_value"，value 必须先写好。
    static bool attach_upload_signal(
        VkSubmitInfo& si, StagingRing& ring, std::size_t region_idx,
        VkTimelineSemaphoreSubmitInfo& tsi,
        VkSemaphore& sem_storage, std::uint64_t& value_storage)
    {
        if (!ring.timeline())
            return false;
        value_storage = ring.next_signal_value(region_idx);
        sem_storage = ring.semaphore(region_idx);
        tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &value_storage;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sem_storage;
        si.pNext = &tsi;
        return true;
    }

    // ── 辅助：获取 command buffer（batch 或独立）──────────────────────
    // 返回 (cmd, owns_cmd)。owns_cmd=true 时调用方需 submit_and_wait。
    [[nodiscard]] Result<std::pair<VkCommandBuffer, bool>> acquire_cmd()
    {
        if (batch_mode_)
        {
            batch_has_ops_ = true;  // P0-1：当前帧已含 op（空帧不提交）
            return std::make_pair(batch_cmd_, false);
        }

        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        auto r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            return std::unexpected(r.error());
        }
        return std::make_pair(cmd, true);
    }

    // ── 辅助：独立模式提交+等待+清理 ──────────────────────────────────
    // owns_cmd=true 时调用：end → submit → wait → free cmd + desc_set
    // desc_set 可为 VK_NULL_HANDLE（fill_zero/copy_buffer 无描述符集）
    [[nodiscard]] Result<void> submit_and_wait(
        VkCommandBuffer cmd, VkDescriptorSet desc_set)
    {
        auto r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            if (desc_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(r.error());
        }

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        r = detail::vk_check(
            vkCreateFence(device_.device(), &fence_info, nullptr, &fence),
            __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            if (desc_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(r.error());
        }

        // 跨 submit 数据依赖：输入 buffer 可能刚由在飞上传的 copy 写入
        // （from_matrix 非阻塞提交后本原语立即执行）——等 in-flight region
        // 信号量。host 不阻塞；GPU 在队列内等数据就绪。
        const auto sw = collect_staging_waits();

        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            VkTimelineSemaphoreSubmitInfo timeline_info{};
            attach_staging_waits(submit_info, sw, timeline_info);
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                __FILE__, __LINE__);
            if (!r)
            {
                vkDestroyFence(device_.device(), fence, nullptr);
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                if (desc_set != VK_NULL_HANDLE)
                    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
        }

        // 30 秒超时（单个原语，比 batch 短）
        constexpr uint64_t kSingleOpTimeoutNs = 30'000'000'000ULL;
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, kSingleOpTimeoutNs),
            __FILE__, __LINE__);

        vkDestroyFence(device_.device(), fence, nullptr);
        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
        if (desc_set != VK_NULL_HANDLE)
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
        return r;
    }

    // ── 辅助：录制输入屏障（多个输入 buffer）──────────────────────────
    void record_input_barriers(VkCommandBuffer cmd,
                                std::initializer_list<VkBuffer> buffers)
    {
        record_input_barriers(cmd,
            std::span<const VkBuffer>(buffers.begin(), buffers.size()));
    }

    void record_input_barriers(VkCommandBuffer cmd,
                                std::span<const VkBuffer> buffers)
    {
        std::vector<VkBufferMemoryBarrier> barriers;
        barriers.reserve(buffers.size());
        for (auto buf : buffers)
        {
            VkBufferMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.buffer = buf;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            barriers.push_back(b);
        }
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data(),
            0, nullptr);
    }

    // ── 辅助：录制输出屏障 ─────────────────────────────────────────────
    void record_output_barrier(VkCommandBuffer cmd, VkBuffer output_buf)
    {
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = output_buf;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);
    }

    // ══════════════════════════════════════════════════════════════════
    // elementwise_v2_gpu — 逐元素原语（unary/binary/select/axpy）
    //
    // Push Constants (32 bytes):
    //   count, mode, op, cmp_op, flags, scalar_b, scalar_then, scalar_else
    // Bindings: A(0), B(1), C(2), OUT(3)
    //
    // 对未使用的 binding（如 unary 模式的 B/C），绑定 A 的 buffer（无害占位）。
    //
    // out != nullptr 时为原地模式：OUT 直接绑定 out 的 buffer（通常即 A），
    // 免分配 + 免全量写出。逐元素 kernel 每线程只读写自己下标一次，
    // read-before-write 天然成立，别名安全。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> elementwise_v2_gpu(
        const GpuTensor& A, const GpuTensor* B, const GpuTensor* C,
        uint32_t count, uint32_t mode, uint32_t op, uint32_t cmp_op,
        uint32_t flags, float scalar_b, float scalar_then, float scalar_else,
        const GpuTensor* out = nullptr)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_elementwise_v2_pipeline())
            return std::unexpected(Error{"elementwise_v2 pipeline not available"});

        // 1. 输出 Tensor：原地模式直接复用 out 的 buffer，否则新分配
        GpuTensor output;
        if (out)
        {
            output = GpuTensor(*out);
        }
        else
        {
            auto output_res = GpuTensor::create_empty(A.rows(), A.cols(), *this);
            if (!output_res) return std::unexpected(output_res.error());
            output = std::move(*output_res);
        }

        // 2. 分配描述符集
        auto ds_r = alloc_desc_set(elementwise_v2_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 3. 写入描述符集（未使用的 binding 绑定 A 作为占位）
        VkBuffer a_buf = A.buffer().impl();
        VkDescriptorBufferInfo buf_infos[4]{
            {a_buf, 0, VK_WHOLE_SIZE},
            {B ? B->buffer().impl() : a_buf, 0, VK_WHOLE_SIZE},
            {C ? C->buffer().impl() : a_buf, 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[4]{};
        for (int i = 0; i < 4; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 4, writes, 0, nullptr);

        // 4. 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 5. 录制
        record_input_barriers(cmd, {a_buf});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            elementwise_v2_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            elementwise_v2_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // Push constants: count, mode, op, cmp_op, flags (5×uint32) + scalar_b, scalar_then, scalar_else (3×float)
        struct PushData {
            uint32_t count, mode, op, cmp_op, flags;
            float scalar_b, scalar_then, scalar_else;
        } push{count, mode, op, cmp_op, flags, scalar_b, scalar_then, scalar_else};
        vkCmdPushConstants(cmd, elementwise_v2_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        const uint32_t wg_count = (count + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        // 6. 独立模式提交
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // run_fused_gpu — AOT 融合 shader 通用执行入口
    //
    // 与 glsl_gen.hpp 生成的 shader 布局严格对应：
    //   Bindings: 输入 0..N-1，输出 N（全部 storage buffer）
    //   Push Constants（逐元素）: uint count, uint cols, float c0..（常量池）
    //   Push Constants（归约）:   uint count, uint cols, uint rows, uint vector_out, ...
    //   Push Constants（matmul 融合）: uint count, uint cols, uint rows, uint mm_k, ...
    //   local_size_x = 256
    //
    // shader_name 必须是已注册的融合 shader 的 key（= expr_spec_key，见
    // 构建期 fused_registry.hpp）。
    // 所有输入同形状 (rows, cols)（matmul 段除外：A/B 按 matmul 形状解释，
    // 但 Vulkan buffer 无形状概念，绑定即用）；同一 buffer 可绑定到多个输入。
    // matmul_k：前置 matmul 段的求和维度（形状参数，运行时填充 mm_k 槽；
    // 非 matmul shader 传 nullopt）。
    // matmul_batch：matmul 段批量数（形状参数，填充 mm_batch 槽 + dispatch z）。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> run_fused_gpu(
        const std::string& shader_name,
        std::span<const GpuTensor> inputs,
        std::span<const Scalar> consts,
        std::size_t rows, std::size_t cols,
        bool vector_out = false,
        std::span<const std::uint32_t> view_params = {},
        std::span<const Scalar> rparams = {},
        GpuTensor* output_override = nullptr,
        std::optional<std::uint32_t> matmul_k = std::nullopt,
        std::uint32_t matmul_batch = 1)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        const auto it = fused_pipelines_.find(shader_name);
        if (it == fused_pipelines_.end())
            return std::unexpected(Error{"fused shader not registered: " + shader_name});
        const VulkanPipeline& pipeline = it->second;
        // 归约轴：-1=逐元素, 0=行归约, 1=列归约
        const int raxis = fused_reduce_axis_.count(shader_name)
            ? fused_reduce_axis_.at(shader_name) : -1;
        // 是否含前置 matmul 段（push constants 多 rows + mm_k）
        const bool has_mm = fused_has_matmul_.count(shader_name)
            ? fused_has_matmul_.at(shader_name) : false;
        // 运行时视图参数个数（RowMod/RotateHalf 的 vp 槽）
        const std::uint32_t n_vp = fused_view_param_counts_.count(shader_name)
            ? fused_view_param_counts_.at(shader_name) : 0u;
        // 运行时标量参数个数（优化器 lr/eps/β 的 rp 槽）
        const std::uint32_t n_rp = fused_rparam_counts_.count(shader_name)
            ? fused_rparam_counts_.at(shader_name) : 0u;

        if (inputs.size() + 1 > EXPR_MAX_INPUTS + 1)
            return std::unexpected(Error{"run_fused_gpu: too many inputs"});
        if (consts.size() > EXPR_MAX_CONSTS)
            return std::unexpected(Error{"run_fused_gpu: too many constants"});
        if (view_params.size() != n_vp)
            return std::unexpected(Error{
                "run_fused_gpu: view_params count mismatch for " + shader_name});
        if (rparams.size() != n_rp)
            return std::unexpected(Error{
                "run_fused_gpu: rparams count mismatch for " + shader_name});
        if (has_mm && !matmul_k)
            return std::unexpected(Error{
                "run_fused_gpu: matmul shader 缺少 matmul_k（求和维度）"});
        if (!has_mm && matmul_k)
            return std::unexpected(Error{
                "run_fused_gpu: 非 matmul shader 收到了 matmul_k"});
        // matmul+归约（S5）：raxis >= 0 时 mm_k 填入 PC 第 5 槽（见下方填充）

        // count 以 uint32 传入 shader（gl_GlobalInvocationID / push constant），
        // 必须保证 rows*cols 不溢出 uint32，否则分派与索引会静默截断。
        if (rows > 0 && cols > UINT32_MAX / rows)
            return std::unexpected(Error{
                "run_fused_gpu: rows*cols exceeds uint32 range"});
        const std::uint32_t count = static_cast<std::uint32_t>(rows * cols);

        // 1. 分配输出 Tensor（vector_out：归约向量原生形状 (rows,1)/(1,cols)）
        //    若调用方指定 output_override（IR-C 录制占位 buffer），则复用其
        //    buffer（形状必须匹配），结果直接写入占位 → Layer 持有的 Tensor
        //    在 end_expr 后即物化，避免额外拷贝。
        const std::size_t out_rows = (vector_out && raxis == 0) ? rows
                                  : (vector_out && raxis == 1) ? 1 : rows;
        const std::size_t out_cols = (vector_out && raxis == 1) ? cols
                                  : (vector_out && raxis == 0) ? 1 : cols;
        std::optional<GpuTensor> owned_output;
        GpuTensor* output_ptr = output_override;
        if (output_override)
        {
            if (output_override->rows() != out_rows || output_override->cols() != out_cols)
                return std::unexpected(Error{
                    "run_fused_gpu: output_override shape mismatch (" +
                    std::to_string(out_rows) + "x" + std::to_string(out_cols) + ")"});
        }
        else
        {
            auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
            if (!output_res) return std::unexpected(output_res.error());
            owned_output.emplace(std::move(*output_res));
            output_ptr = &*owned_output;
        }
        GpuTensor output = *output_ptr;

        // 2. 分配描述符集（N 输入 + 1 输出）
        auto ds_r = alloc_desc_set(pipeline.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        const std::size_t n_bindings = inputs.size() + 1;
        std::vector<VkDescriptorBufferInfo> buf_infos(n_bindings);
        std::vector<VkWriteDescriptorSet> writes(n_bindings);
        for (std::size_t i = 0; i < inputs.size(); ++i)
            buf_infos[i] = {inputs[i].buffer().impl(), 0, VK_WHOLE_SIZE};
        buf_infos[inputs.size()] = {output.buffer().impl(), 0, VK_WHOLE_SIZE};
        for (std::size_t i = 0; i < n_bindings; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<std::uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(),
            static_cast<std::uint32_t>(n_bindings), writes.data(), 0, nullptr);

        // 3. 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 4. 录制：输入屏障 → bind → push constants → dispatch → 输出屏障
        std::vector<VkBuffer> in_bufs;
        in_bufs.reserve(inputs.size());
        for (const auto& t : inputs)
            in_bufs.push_back(t.buffer().impl());
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // Push constants 布局与 glsl_gen.hpp 一致：
        //   逐元素: count, cols, [vp0..], c0..
        //   归约:   count, cols, rows, vector_out, [vp0..], c0..
        //   matmul: count, cols, rows, mm_k, mm_batch, [vp0..], c0..
        //   matmul+归约: count, cols, rows, vector_out, mm_k, mm_batch, [vp0..], c0..
        const std::uint32_t pc_base =
            (raxis >= 0 && has_mm) ? 6u
          : ((raxis >= 0 || has_mm) ? 5u : 2u);
        const std::uint32_t pc_uints = pc_base;
        std::vector<std::uint8_t> pc(
            (pc_uints + n_vp) * sizeof(std::uint32_t) + sizeof(Scalar) * consts.size()
            + sizeof(Scalar) * rparams.size());
        std::memcpy(pc.data(), &count, sizeof(std::uint32_t));
        const std::uint32_t cols32 = static_cast<std::uint32_t>(cols);
        std::memcpy(pc.data() + sizeof(std::uint32_t), &cols32, sizeof(std::uint32_t));
        if (raxis >= 0)
        {
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32,
                        sizeof(std::uint32_t));
            const std::uint32_t vo = vector_out ? 1u : 0u;
            std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &vo,
                        sizeof(std::uint32_t));
            if (has_mm)
            {
                // matmul+归约（S5/S7）：mm_k + mm_batch（形状参数，运行时填充）
                std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &*matmul_k,
                            sizeof(std::uint32_t));
                std::memcpy(pc.data() + 5 * sizeof(std::uint32_t), &matmul_batch,
                            sizeof(std::uint32_t));
            }
        }
        else if (has_mm)
        {
            // matmul 融合：rows + mm_k + mm_batch（形状参数，运行时按实际 spec
            // 填充 → 同结构不同 K/batch 共享一个融合 shader）
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32,
                        sizeof(std::uint32_t));
            std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &*matmul_k,
                        sizeof(std::uint32_t));
            std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &matmul_batch,
                        sizeof(std::uint32_t));
        }
        // 运行时视图参数（RowMod 周期 / RotateHalf 块大小），置于固定头之后、常量池之前
        for (std::uint32_t i = 0; i < n_vp; ++i)
            std::memcpy(pc.data() + (pc_uints + i) * sizeof(std::uint32_t),
                        &view_params[i], sizeof(std::uint32_t));
        if (!consts.empty())
            std::memcpy(pc.data() + (pc_uints + n_vp) * sizeof(std::uint32_t), consts.data(),
                        sizeof(Scalar) * consts.size());
        // 运行时标量参数（优化器 lr/eps/β 等 float rp 槽），置于常量池之后
        if (!rparams.empty())
            std::memcpy(pc.data() + (pc_uints + n_vp) * sizeof(std::uint32_t) +
                            sizeof(Scalar) * consts.size(),
                        rparams.data(), sizeof(Scalar) * rparams.size());
        vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(pc.size()), pc.data());

        // dispatch：逐元素 = ceil(count/(256*vec_width))；行归约 = rows 个工作组；
        // 列归约(tile) = ceil(cols/256) 个工作组（每工作组 256 列）；
        // matmul 分块（S5）= (ceil(cols/BLOCK), ceil(rows/BLOCK), 1)，
        // BLOCK 与 glsl_gen 生成的输出块一致（EXPR_MATMUL_BLOCK：每工作组
        // 32×32 输出块、16×16 线程、每线程 2×2 寄存器分块）
        const std::uint32_t vec_width = fused_vec_width_.count(shader_name)
            ? fused_vec_width_.at(shader_name) : 1u;
        if (has_mm && raxis < 0)
        {
            const std::uint32_t wg_x =
                (static_cast<std::uint32_t>(cols) + nn::EXPR_MATMUL_BLOCK - 1u)
                / nn::EXPR_MATMUL_BLOCK;
            const std::uint32_t wg_y =
                (static_cast<std::uint32_t>(rows) + nn::EXPR_MATMUL_BLOCK - 1u)
                / nn::EXPR_MATMUL_BLOCK;
            // batch（S7）：dispatch z = 批次，A/B 按 batch 垂直切分
            vkCmdDispatch(cmd, wg_x, wg_y, matmul_batch);
        }
        else
        {
            const std::uint32_t wg_count =
                (raxis == 0) ? static_cast<std::uint32_t>(rows)
                : (raxis == 1) ? (static_cast<std::uint32_t>(cols) + 255u) / 256u
                : (count + 256u * vec_width - 1u) / (256u * vec_width);
            vkCmdDispatch(cmd, wg_count, 1, 1);
        }
        record_output_barrier(cmd, output.buffer().impl());

        // 5. 独立模式提交
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // 查询融合 shader 是否已注册（GpuEngine AOT 匹配前置判断）
    [[nodiscard]] bool has_fused_shader(const std::string& shader_name) const noexcept
    {
        return fused_pipelines_.find(shader_name) != fused_pipelines_.end();
    }

    // ══════════════════════════════════════════════════════════════════
    // reduce_gpu — 行/列归约原语（sum/max）
    //
    // Push Constants (16 bytes): rows, cols, mode, reduce_op
    // Bindings: In(0), Out(1)
    // mode: 0=row_reduce → out(rows,1), 1=col_reduce → out(1,cols)
    // reduce_op: 0=sum, 1=max
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> reduce_gpu(
        const GpuTensor& input, uint32_t mode, uint32_t reduce_op)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_reduce_pipeline())
            return std::unexpected(Error{"reduce pipeline not available"});

        const uint32_t rows = static_cast<uint32_t>(input.rows());
        const uint32_t cols = static_cast<uint32_t>(input.cols());

        // 1. 分配输出 Tensor
        std::size_t out_rows = (mode == 0) ? rows : 1;
        std::size_t out_cols = (mode == 0) ? 1 : cols;
        auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
        if (!output_res) return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

        // 2. 分配描述符集
        auto ds_r = alloc_desc_set(reduce_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[2]{
            {input.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[2]{};
        for (int i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);

        // 4. 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 5. 录制
        record_input_barriers(cmd, {input.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            reduce_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            reduce_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[4] = {rows, cols, mode, reduce_op};
        vkCmdPushConstants(cmd, reduce_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // 每个工作组 256 线程协作归约一行/一列
        const uint32_t workgroups = (mode == 0) ? rows : cols;
        vkCmdDispatch(cmd, workgroups, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        // 6. 独立模式提交
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // broadcast_gpu — 行/列广播原语
    //
    // Push Constants (16 bytes): rows, cols, mode, op
    // Bindings: A(0), Vec(1), Out(2)
    // mode: 0=row_broadcast (vec indexed by row), 1=col_broadcast (vec indexed by col)
    // op: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min
    //
    // out != nullptr 时为原地模式：Out 直接绑定 out 的 buffer（通常即 A），
    // 每线程只读写自己下标一次，别名安全。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> broadcast_gpu(
        const GpuTensor& A, const GpuTensor& vec,
        uint32_t mode, uint32_t op,
        const GpuTensor* out = nullptr)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_broadcast_pipeline())
            return std::unexpected(Error{"broadcast pipeline not available"});

        const uint32_t rows = static_cast<uint32_t>(A.rows());
        const uint32_t cols = static_cast<uint32_t>(A.cols());

        // 1. 输出 Tensor：原地模式直接复用 out 的 buffer，否则新分配
        GpuTensor output;
        if (out)
        {
            output = GpuTensor(*out);
        }
        else
        {
            auto output_res = GpuTensor::create_empty(A.rows(), A.cols(), *this);
            if (!output_res) return std::unexpected(output_res.error());
            output = std::move(*output_res);
        }

        // 2. 分配描述符集
        auto ds_r = alloc_desc_set(broadcast_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[3]{
            {A.buffer().impl(), 0, VK_WHOLE_SIZE},
            {vec.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 3, writes, 0, nullptr);

        // 4. 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 5. 录制
        record_input_barriers(cmd, {A.buffer().impl(), vec.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            broadcast_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            broadcast_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[4] = {rows, cols, mode, op};
        vkCmdPushConstants(cmd, broadcast_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t total = rows * cols;
        const uint32_t wg_count = (total + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        // 6. 独立模式提交
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // rearrange_3d_gpu — 3D 维度转置 (M, B, N) ↔ (B, M, N)
    //
    // Push Constants (20 bytes): M, B, N, inverse, total
    // Bindings: In(0), Out(1)
    //   inverse=0: (M, B*N) → (B*M, N)
    //   inverse=1: (B*M, N) → (M, B*N)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> rearrange_3d_gpu(
        const GpuTensor& input,
        uint32_t M, uint32_t B, uint32_t N, uint32_t inverse)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_rearrange_3d_pipeline())
            return std::unexpected(Error{"rearrange_3d pipeline not available"});

        const uint32_t total = M * B * N;
        if (total != input.rows() * input.cols())
            return std::unexpected(Error{"rearrange_3d: element count mismatch"});

        // 1. 分配输出 Tensor
        const std::size_t out_rows = inverse ? M : (static_cast<std::size_t>(B) * M);
        const std::size_t out_cols = inverse ? (static_cast<std::size_t>(B) * N) : N;
        auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
        if (!output_res) return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

        // 2. 分配描述符集
        auto ds_r = alloc_desc_set(rearrange_3d_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[2]{
            {input.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[2]{};
        for (int i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);

        // 4. 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 5. 录制
        record_input_barriers(cmd, {input.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            rearrange_3d_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            rearrange_3d_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[5] = {M, B, N, inverse, total};
        vkCmdPushConstants(cmd, rearrange_3d_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t wg_count = (total + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        // 6. 独立模式提交
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // fill_zero_gpu — 清零 GPU buffer（使用 vkCmdFillBuffer，字节级操作，§6.3）
    // ══════════════════════════════════════════════════════════════════
    template <Precision P>
    [[nodiscard]] Result<void> fill_zero_gpu(GpuTensorT<P>& tensor)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        auto cmd_r = acquire_cmd();
        if (!cmd_r) return std::unexpected(cmd_r.error());
        auto [cmd, owns_cmd] = *cmd_r;

        vkCmdFillBuffer(cmd, tensor.buffer().impl(), 0, VK_WHOLE_SIZE, 0);

        // 屏障确保 fill 完成后可见
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = tensor.buffer().impl();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);

        if (owns_cmd)
        {
            // fill_zero 不使用描述符集，传 VK_NULL_HANDLE 跳过释放
            auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // ══════════════════════════════════════════════════════════════════
    // copy_buffer_gpu — GPU 内 buffer 拷贝（使用 vkCmdCopyBuffer）
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> copy_buffer_gpu(
        VkBuffer src, VkBuffer dst, VkDeviceSize size)
    {
        return copy_buffer_region_gpu(src, 0, dst, 0, size);
    }

    // ══════════════════════════════════════════════════════════════════
    // copy_buffer_region_gpu — 带偏移的 buffer 拷贝（activation offload slab 用）
    // 从 src[src_offset, src_offset+size) 拷到 dst[dst_offset, dst_offset+size)。
    // batch 模式只录制不提交；非 batch 独立提交等待。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> copy_buffer_region_gpu(
        VkBuffer src, VkDeviceSize src_offset,
        VkBuffer dst, VkDeviceSize dst_offset,
        VkDeviceSize size)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        auto cmd_r = acquire_cmd();
        if (!cmd_r) return std::unexpected(cmd_r.error());
        auto [cmd, owns_cmd] = *cmd_r;

        VkBufferCopy cp{src_offset, dst_offset, size};
        vkCmdCopyBuffer(cmd, src, dst, 1, &cp);

        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = dst;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // ══════════════════════════════════════════════════════════════════
    // clone_gpu — GPU 内深拷贝（分配新 buffer + copy，无 PCIe 传输）
    // ══════════════════════════════════════════════════════════════════
    template <Precision P>
    [[nodiscard]] Result<GpuTensorT<P>> clone_gpu(const GpuTensorT<P>& src)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        // 1. 分配同形状的新 buffer
        auto dst_res = GpuTensorT<P>::create_empty(src.rows(), src.cols(), *this);
        if (!dst_res) return std::unexpected(dst_res.error());
        GpuTensorT<P> dst = std::move(*dst_res);

        // 2. GPU 内拷贝
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            src.rows() * src.cols() * sizeof(elem<P>));
        auto r = copy_buffer_gpu(src.buffer().impl(), dst.buffer().impl(), size);
        if (!r) return std::unexpected(r.error());

        return dst;
    }

    // ── 通用精度转换原语（engine.cast 的 GPU 实现，无 PCIe 往返）────────
    // 源/目标均为原始 32-bit word 缓冲；kind 分派精度对：
    //   0 = f16→f32（src 2B/pair 打包，dst 4B/word）
    //   1 = f32→f16（src 4B/word，dst 2B/pair 打包，round-half-to-even）
    // 元素数保持不变（count = rows*cols），仅字节宽度差。后续精度对
    // （BF16/F64/f8 等）扩 kind 枚举即可，绝不改引擎 cast API。
    [[nodiscard]] Result<void> cast_gpu(
        const GpuBuffer& src, const GpuBuffer& dst,
        std::size_t count, std::uint32_t kind)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_cast_pipeline())
            return std::unexpected(Error{"cast_gpu: cast pipeline not available"});

        auto ds_r = alloc_desc_set(cast_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        VkDescriptorBufferInfo binfo[2] = {
            {src.impl(), 0, VK_WHOLE_SIZE},
            {dst.impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[2];
        for (std::uint32_t i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].pNext = nullptr;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = i;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pImageInfo = nullptr;
            writes[i].pBufferInfo = &binfo[i];
            writes[i].pTexelBufferView = nullptr;
        }
        vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        std::vector<VkBuffer> in_bufs{src.impl()};
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            cast_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        struct Push { std::uint32_t count; std::uint32_t kind; };
        Push push{static_cast<std::uint32_t>(count), kind};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));
        vkCmdPushConstants(cmd, cast_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0,
            static_cast<uint32_t>(pc.size()), pc.data());

        // 每线程处理 2 元素（一对一 half 或单词对）
        const auto pairs = (count + 1u) / 2u;
        const std::uint32_t wg = static_cast<std::uint32_t>((pairs + 255u) / 256u);
        vkCmdDispatch(cmd, wg < 1u ? 1u : wg, 1u, 1u);
        record_output_barrier(cmd, dst.impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // ── 创建逻辑形状 (rows,cols) 但存储为偶数槽位的 f16 张量 ───────────
    // f32→f16 的 cast 每线程按整 word（两 half）写入；奇数元素 count 时最后一个
    // word 会越界 2 字节。这里把底层缓冲按偶数元素分配（多 1 槽），逻辑形状不变，
    // 写 word 永不越界；to_matrix 仍只读前 rows*cols 个元素。
    [[nodiscard]] Result<GpuTensorF16> create_f16_tensor(
        std::size_t rows, std::size_t cols)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        const std::size_t cnt = rows * cols;
        const std::size_t cnt_pad = cnt + (cnt & 1u);         // 偶数槽位
        auto b = GpuBuffer::create_device_local(
            device_.device(), *memory_pool_, cnt_pad * 2u,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!b) return std::unexpected(b.error());
        return GpuTensorF16(std::make_shared<GpuBuffer>(std::move(*b)), rows, cols);
    }
    template <Precision P>
    [[nodiscard]] Result<GpuTensorT<P>> slice_rows_gpu(
        const GpuTensorT<P>& src, std::size_t start_row, std::size_t count)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        const std::size_t cols = src.cols();
        if (start_row + count > src.rows())
            return std::unexpected(Error{"slice_rows_gpu: range out of bounds"});

        // 分配目标 buffer
        auto dst_res = GpuTensorT<P>::create_empty(count, cols, *this);
        if (!dst_res) return std::unexpected(dst_res.error());
        GpuTensorT<P> dst = std::move(*dst_res);

        // 行区间在行主序下连续：[start_row * cols, (start_row + count) * cols)
        const std::size_t elem_size = sizeof(elem<P>);
        const VkDeviceSize src_offset = static_cast<VkDeviceSize>(
            start_row * cols * elem_size);
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            count * cols * elem_size);

        auto cmd_r = acquire_cmd();
        if (!cmd_r) return std::unexpected(cmd_r.error());
        auto [cmd, owns_cmd] = *cmd_r;

        VkBufferCopy cp{src_offset, 0, size};
        vkCmdCopyBuffer(cmd, src.buffer().impl(), dst.buffer().impl(), 1, &cp);

        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = dst.buffer().impl();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
            if (!r) return std::unexpected(r.error());
        }
        return dst;
    }

    // ══════════════════════════════════════════════════════════════════
    // insert_rows_gpu — 行插入（GPU 内就地写入，无 PCIe 传输）
    // 将 src 的所有行写入 dst 的行 [dst_start_row, dst_start_row + src.rows())
    // 真·就地修改（vkCmdCopyBuffer with dstOffset）。
    // 注意：dst 必须以 TRANSFER_DST_BIT 创建（create_empty 已包含）。
    // ══════════════════════════════════════════════════════════════════
    template <Precision P>
    [[nodiscard]] Result<void> insert_rows_gpu(
        GpuTensorT<P>& dst, std::size_t dst_start_row, const GpuTensorT<P>& src)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        const std::size_t cols = dst.cols();
        if (src.cols() != cols)
            return std::unexpected(Error{"insert_rows_gpu: column count mismatch"});
        if (dst_start_row + src.rows() > dst.rows())
            return std::unexpected(Error{"insert_rows_gpu: range out of bounds"});

        const std::size_t elem_size = sizeof(elem<P>);
        const VkDeviceSize dst_offset = static_cast<VkDeviceSize>(
            dst_start_row * cols * elem_size);
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            src.rows() * cols * elem_size);

        auto cmd_r = acquire_cmd();
        if (!cmd_r) return std::unexpected(cmd_r.error());
        auto [cmd, owns_cmd] = *cmd_r;

        VkBufferCopy cp{0, dst_offset, size};
        vkCmdCopyBuffer(cmd, src.buffer().impl(), dst.buffer().impl(), 1, &cp);

        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = dst.buffer().impl();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    // ── Staging Path：CPU span → GPU → 计算 → GPU → CPU span ─────────
    [[nodiscard]] Result<void> matmul_direct(
        std::span<const Scalar> a, std::span<const Scalar> b,
        std::span<Scalar> c, std::size_t M, std::size_t N, std::size_t K,
        uint32_t transA = 0, uint32_t transB = 0)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        // 1. 创建临时 GPU buffers（byte count = 元素数 × sizeof(float)，§6.3）
        auto a_buf = GpuBuffer::create_device_local(
            device_.device(), *memory_pool_, M * K * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!a_buf)
            return std::unexpected(a_buf.error());

        auto b_buf = GpuBuffer::create_device_local(
            device_.device(), *memory_pool_, K * N * sizeof(float),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!b_buf)
            return std::unexpected(b_buf.error());

        // 2. 上传数据
        auto a_tensor = GpuTensor(std::make_shared<GpuBuffer>(std::move(*a_buf)), M, K);
        auto b_tensor = GpuTensor(std::make_shared<GpuBuffer>(std::move(*b_buf)), K, N);

        auto r = upload_blocking(a_tensor, a);
        if (!r)
            return r;

        r = upload_blocking(b_tensor, b);
        if (!r)
            return r;

        // 3. 执行 matmul
        auto c_tensor_res = matmul_gpu(a_tensor, b_tensor, transA, transB);
        if (!c_tensor_res)
            return std::unexpected(c_tensor_res.error());

        // 4. 下载结果
        r = download_blocking(*c_tensor_res, c);
        if (!r)
            return r;

        return {};
    }

    // ══════════════════════════════════════════════════════════════════
    // transpose_gpu — 矩阵转置 (R, C) → (C, R)
    //
    // Push Constants (8 bytes): rows, cols
    // Bindings: In(0), Out(1)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> transpose_gpu(const GpuTensor& A)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_transpose_pipeline())
            return std::unexpected(Error{"transpose pipeline not available"});

        const uint32_t R = static_cast<uint32_t>(A.rows());
        const uint32_t C = static_cast<uint32_t>(A.cols());

        // 输出 (C, R)
        auto out_res = GpuTensor::create_empty(C, R, *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        // 分配描述符集
        auto ds_r = alloc_desc_set(transpose_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 写入描述符集
        VkDescriptorBufferInfo buf_infos[2]{
            {A.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[2]{};
        for (int i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);

        // 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 录制
        record_input_barriers(cmd, {A.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            transpose_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            transpose_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[2] = {R, C};
        vkCmdPushConstants(cmd, transpose_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t total = R * C;
        const uint32_t wg_count = (total + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // gather_gpu — 按行索引查表 (GPU-native)
    //
    // table: (vocab, D), indices: (num,) → output: (num, D)
    // output[i] = table[indices[i]]，越界索引返回零行
    //
    // Push Constants (12 bytes): vocab, D, num
    // Bindings: Table(0), Indices(1), Output(2)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> gather_gpu(
        const GpuTensor& table, const GpuTensor& indices)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_gather_pipeline())
            return std::unexpected(Error{"gather pipeline not available"});

        const uint32_t vocab = static_cast<uint32_t>(table.rows());
        const uint32_t D = static_cast<uint32_t>(table.cols());
        const uint32_t num = static_cast<uint32_t>(indices.rows() * indices.cols());

        // 输出 (num, D)
        auto out_res = GpuTensor::create_empty(num, D, *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        // 分配描述符集
        auto ds_r = alloc_desc_set(gather_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 写入描述符集
        VkDescriptorBufferInfo buf_infos[3]{
            {table.buffer().impl(), 0, VK_WHOLE_SIZE},
            {indices.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 3, writes, 0, nullptr);

        // 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 录制
        record_input_barriers(cmd, {table.buffer().impl(), indices.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            gather_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            gather_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[3] = {vocab, D, num};
        vkCmdPushConstants(cmd, gather_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t total = num * D;
        const uint32_t wg_count = (total + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // scatter_add_gpu — 按行索引原子累加梯度 (GPU-native)
    //
    // dst: (vocab, D) 原地修改，indices: (num,), grad: (num, D)
    // dst[indices[i]][d] += grad[i][d]
    // 使用 CAS 循环实现 float atomicAdd
    //
    // Push Constants (12 bytes): vocab, D, num
    // Bindings: Dst(0), Indices(1), Grad(2)
    //
    // 注意：dst buffer 内容以 uint 视角访问用于 atomicCompSwap。
    // dst 初始值必须为合法 float 位模式（如 0x00000000 = 0.0f）。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> scatter_add_gpu(
        GpuTensor& dst, const GpuTensor& indices, const GpuTensor& grad)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_scatter_add_pipeline())
            return std::unexpected(Error{"scatter_add pipeline not available"});

        const uint32_t vocab = static_cast<uint32_t>(dst.rows());
        const uint32_t D = static_cast<uint32_t>(dst.cols());
        const uint32_t num = static_cast<uint32_t>(indices.rows() * indices.cols());

        // 分配描述符集
        auto ds_r = alloc_desc_set(scatter_add_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        // 写入描述符集（dst 使用 uint 视图用于原子操作）
        VkDescriptorBufferInfo buf_infos[3]{
            {dst.buffer().impl(), 0, VK_WHOLE_SIZE},
            {indices.buffer().impl(), 0, VK_WHOLE_SIZE},
            {grad.buffer().impl(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 3, writes, 0, nullptr);

        // 获取 command buffer
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // 录制（dst 需要读+写屏障）
        {
            std::vector<VkBufferMemoryBarrier> barriers;
            VkBufferMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            b.buffer = dst.buffer().impl();
            barriers.push_back(b);
            b.buffer = indices.buffer().impl();
            barriers.push_back(b);
            b.buffer = grad.buffer().impl();
            barriers.push_back(b);
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr,
                static_cast<uint32_t>(barriers.size()), barriers.data(),
                0, nullptr);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            scatter_add_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            scatter_add_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[3] = {vocab, D, num};
        vkCmdPushConstants(cmd, scatter_add_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t total = num * D;
        const uint32_t wg_count = (total + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);

        // 输出屏障（dst 被修改，需要 memory barrier）
        {
            VkBufferMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.buffer = dst.buffer().impl();
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 1, &b, 0, nullptr);
        }

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }
};  // GpuBackend

// ══════════════════════════════════════════════════════════════════════
// GpuBuffer 析构（类外定义）：batch 录制期间延迟销毁，避免已录制的
// descriptor 引用已销毁的 buffer（VUID-vkDestroyBuffer-buffer-00922）。
// ══════════════════════════════════════════════════════════════════════
GpuBuffer::~GpuBuffer()
{
    if (buffer_ == VK_NULL_HANDLE)
        return;

    auto& backend = GpuBackend::instance();
    if (backend.is_initialized() && backend.needs_deferred_destroy())
    {
        // 延迟 vkDestroyBuffer（Vulkan 规范要求 buffer 在引用它的命令
        // 提交完成前不能被销毁）。两种延迟场景（needs_deferred_destroy）：
        //   1. batch 录制中：本 buffer 被当前帧已录制的命令引用
        //   2. 非 batch 但最近提交的帧仍在飞行：本 buffer 可能被该帧命令
        //      引用（fence 信号即可证明安全）
        // 延迟销毁打上帧标签，该帧完成（环槽复用 / drain / 非阻塞 reap）
        // 时立即销毁 + 归还内存——锁窗从"整个 batch"缩短到"所属帧"
        // （P0-1）。
        // D1 修复：内存归还也一并延迟到 buffer 销毁之后。若此处立即
        // pool_->free()，新 buffer 可能分配到本 buffer 尚未销毁的同一区间
        // → 两个存活 buffer 内存重叠，违反 Vulkan 规范。
        // GpuBuffer 只经 MemoryPool 创建（create_device_local /
        // create_host_visible），pool_ 恒有效；alloc_ 无效时 free 为 no-op。
        backend.defer_buffer_destroy(device_, buffer_, alloc_, pool_);
    }
    else
    {
        // 无在飞帧引用：立即销毁 + 归还内存
        vkDestroyBuffer(device_, buffer_, nullptr);
        if (pool_ && alloc_.valid())
            pool_->free(alloc_);
    }

    buffer_ = VK_NULL_HANDLE;
    alloc_ = {};
    pool_.reset();
}

} // namespace nn

#endif // NN_HAS_VULKAN
