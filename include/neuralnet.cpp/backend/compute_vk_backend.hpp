// ── compute_vk_backend.hpp ──────────────────────────────────────────────────────
// Vulkan Compute Backend
//
// 职责：
//   - 管理 Vulkan 实例、设备、队列（RAII）
//   - 管理 MemoryPool 子分配器
//   - 管理 StagingRing 环形缓冲区
//   - 提供 GPU 操作 API（matmul、elementwise）
//
// 同步机制（独立/非 batch 模式）：
//   - 复用 initialize() 预分配的 solo fence + command buffer（submit 前
//     vkResetFences；旧实现每算子 create/destroy fence + alloc/free cmd
//     是逐元素算子 ≈0.16ms 固定开销的主要构成之一）
//   - 提交后立即等待 fence，确保 GPU 计算完成
//   - 完成后仅归还 descriptor set（gpu_tensor_pool_ 池化复用）
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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#if __has_include("matmul_gemv_spv.hpp")
#include "matmul_gemv_spv.hpp"
#define NN_MATMUL_GEMV_SPV_EMBEDDED
#endif

#if __has_include("batched_matmul_spv.hpp")
#include "batched_matmul_spv.hpp"
#define NN_BATCHED_MATMUL_SPV_EMBEDDED
#endif

// 精度变体（Phase 2 in-kernel f16）：同一份 .comp 用 -DNN_SHADER_F16=1 编出的
// 第二份 SPIR-V（f16 缓冲载入 + f32 累加 + f16 写出）。
#if __has_include("batched_matmul_f16_spv.hpp")
#include "batched_matmul_f16_spv.hpp"
#define NN_BATCHED_MATMUL_F16_SPV_EMBEDDED
#endif

#if __has_include("matmul_tiled_f16_spv.hpp")
#include "matmul_tiled_f16_spv.hpp"
#define NN_MATMUL_TILED_F16_SPV_EMBEDDED
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

#if __has_include("im2col_spv.hpp")
#include "im2col_spv.hpp"
#define NN_IM2COL_SPV_EMBEDDED
#endif

#if __has_include("col2im_spv.hpp")
#include "col2im_spv.hpp"
#define NN_COL2IM_SPV_EMBEDDED
#endif

#if __has_include("group_reduce_spv.hpp")
#include "group_reduce_spv.hpp"
#define NN_GROUP_REDUCE_SPV_EMBEDDED
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
    VulkanPipeline matmul_gemv_pipeline_;
    VulkanPipeline batched_matmul_pipeline_;
    // 精度变体（f16 存储；设备无 storageBuffer16BitAccess 时保持空句柄）
    VulkanPipeline batched_matmul_f16_pipeline_;
    VulkanPipeline matmul_tiled_f16_pipeline_;
    VulkanPipeline elementwise_v2_pipeline_;
    VulkanPipeline reduce_pipeline_;
    // 列归约两段式 partials scratch（成员复用：batch 录制期被 cmd 引用，
    // 局部销毁会踩铁律 6；形状变化时旧缓冲走 pending_destroys_ 延迟销毁，
    // 对在飞录制安全。同一 batch 内多次使用由 cmd 内屏障串行化）
    std::optional<GpuTensor> reduce_partial_;
    VulkanPipeline broadcast_pipeline_;
    VulkanPipeline rearrange_3d_pipeline_;
    VulkanPipeline transpose_pipeline_;
    VulkanPipeline gather_pipeline_;
    VulkanPipeline scatter_add_pipeline_;
    // 卷积/池化窗口展开（纯数据搬运；Conv2D/MaxPool2D 的 im2col/col2im）
    VulkanPipeline im2col_pipeline_;
    VulkanPipeline col2im_pipeline_;
    // 分组归约（沿行方向按固定长度 R 分组求和/求最大）
    VulkanPipeline group_reduce_pipeline_;
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
    // fold 形态元数据（P-C1/P-C2，与生成器分派同源判定）：
    //   out       = 输出列数（vec_state_len 或 1）——调用约定 cols==out 校验
    //   per_thread_row = 1 → v1 标量 fold：每线程一行（local 256，wg=ceil(rows/256)）；
    //               0 → v2 双域 fold：每 WG EXPR_FOLD_ROWS_PER_WG 行
    //               （row = wg*NR + ri，wg = ceil(rows/NR)，行循环同生成器）
    //               ——dispatch 公式两者不同（v2 若按 ceil(count/256) 会丢行！）
    struct FoldMeta { std::uint32_t out; std::uint8_t per_thread_row; };
    std::unordered_map<std::string, FoldMeta> fused_fold_meta_;
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
    // 帧提交后是否立即非阻塞收割已完成帧（显存峰值优化；NN_NO_EARLY_REAP=1 关闭）
    bool early_reap_enabled_ = true;
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;  // 当前帧的 command buffer

    // ── 异步标量回读（P0-2：非阻塞 loss 取值）─────────────────────────
    // 动机：forward 末尾下载 loss 标量若走 to_matrix，会 end_batch +
    // wait_in_flight（等全部在飞帧）→ 每个 step 一次全流水线 drain，host 与
    // GPU 无法重叠（锯齿）。这里给标量回读一条"不等"的路径：
    //   每个槽位 = 独立 command buffer + fence + 持久 mapped host 缓冲（4B）。
    //   submit：录制一次 4B D2H 拷贝并提交（**不等待**），调用方须在主帧
    //           提交之后调用（同队列 FIFO ⇒ 拷贝排在生产者之后，不会读到旧值）。
    //   poll ：vkGetFenceStatus 非阻塞查询；就绪则从 mapped 内存读值。
    // 槽位不参与 frames_ 环（环槽会被复用、fence 会被 reset），故自带 fence。
    static constexpr std::size_t SCALAR_READBACK_SLOTS = 8;

    struct ScalarReadbackSlot
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkBuffer host = VK_NULL_HANDLE;
        MemoryPool::Allocation alloc{};
        void* mapped = nullptr;
        bool pending = false;    // 已提交、fence 未确认
    };
    std::vector<ScalarReadbackSlot> rb_slots_;

    // ── 独立（非 batch）模式复用资源 ────────────────────────────────────
    // 逐元素算子固定开销实测 ≈0.16ms/次（4096² kernel 本身已达 370GB/s，
    // 瓶颈全在 host 侧）：每算子 vkCreateFence/vkDestroyFence +
    // vkAllocateCommandBuffers/vkFreeCommandBuffers 是可消除项 →
    // initialize() 预分配，acquire_cmd/submit_and_wait 内 reset 复用。
    // 独立模式「录制→提交→等待」严格串行（等完才返回）→ 单实例安全；
    // batch 模式走 frames_ 环，不经过这里。单线程录制假设同 batch_cmd_。
    VkFence solo_fence_ = VK_NULL_HANDLE;        // 创建即 unsignaled
    VkCommandBuffer solo_cmd_ = VK_NULL_HANDLE;   // 每次 reset + begin 复用

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

    [[nodiscard]] static const std::vector<uint32_t>& get_matmul_gemv_spirv()
    {
#ifdef NN_MATMUL_GEMV_SPV_EMBEDDED
        return nn_matmul_gemv_spirv_bytecode();
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

    [[nodiscard]] static const std::vector<uint32_t>& get_batched_matmul_f16_spirv()
    {
#ifdef NN_BATCHED_MATMUL_F16_SPV_EMBEDDED
        return nn_batched_matmul_f16_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_matmul_tiled_f16_spirv()
    {
#ifdef NN_MATMUL_TILED_F16_SPV_EMBEDDED
        return nn_matmul_tiled_f16_spirv_bytecode();
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

    [[nodiscard]] static const std::vector<uint32_t>& get_im2col_spirv()
    {
#ifdef NN_IM2COL_SPV_EMBEDDED
        return nn_im2col_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_col2im_spirv()
    {
#ifdef NN_COL2IM_SPV_EMBEDDED
        return nn_col2im_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_group_reduce_spirv()
    {
#ifdef NN_GROUP_REDUCE_SPV_EMBEDDED
        return nn_group_reduce_spirv_bytecode();
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

            destroy_scalar_readback_slots();

            staging_ring_.reset();

            // 释放帧环 fences（环内 command buffers 随 command pool 一起释放）
            for (auto& f : frames_)
            {
                if (f.fence != VK_NULL_HANDLE)
                    vkDestroyFence(device_.device(), f.fence, nullptr);
            }

            // 独立模式复用 fence（solo command buffer 随 command pool 释放）
            if (solo_fence_ != VK_NULL_HANDLE)
            {
                vkDestroyFence(device_.device(), solo_fence_, nullptr);
                solo_fence_ = VK_NULL_HANDLE;
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

        // 显存峰值优化开关：默认在帧提交后非阻塞收割已完成帧，让步内已
        // 析构张量的内存尽早还池；NN_NO_EARLY_REAP=1 关闭（A/B 对照用）。
        early_reap_enabled_ = VulkanDevice::get_env("NN_NO_EARLY_REAP").empty();

        // 3. 创建 matmul pipeline
        auto pl_r = VulkanPipeline::create_matmul(device_.device(), spirv);
        if (!pl_r)
            return std::unexpected(pl_r.error());
        matmul_pipeline_ = std::move(*pl_r);

        // 4. 创建 memory pool（持久 + 瞬态）
        //    底材粒度可用 NN_POOL_BLOCK_MB 覆盖（显存峰值调参/探针用）：
        //    128MB 大底材在混合尺寸+混合生命周期下会产生不可归还的内部碎片。
        VkDeviceSize block_bytes = MemoryPool::DEFAULT_BLOCK_SIZE;
        {
            const std::string v = VulkanDevice::get_env("NN_POOL_BLOCK_MB");
            if (!v.empty())
            {
                const unsigned long long mb = std::strtoull(v.c_str(), nullptr, 10);
                if (mb > 0) block_bytes = static_cast<VkDeviceSize>(mb) * 1024ull * 1024ull;
            }
        }
        // 自适应阶梯上限（NN_POOL_LADDER_MAX_MB>0 启用）：中等分配按最小
        // 2 的幂类别分池，兼顾小负载抗碎片与大负载少分配次数。
        VkDeviceSize ladder_max = 0;
        {
            const std::string v = VulkanDevice::get_env("NN_POOL_LADDER_MAX_MB");
            if (!v.empty())
            {
                const unsigned long long mb = std::strtoull(v.c_str(), nullptr, 10);
                if (mb > 0) ladder_max = static_cast<VkDeviceSize>(mb) * 1024ull * 1024ull;
            }
        }
        memory_pool_ = std::make_unique<MemoryPool>(
            device_.device(), device_.physical_device(), block_bytes,
            MemoryPool::DEFAULT_SMALL_BLOCK_SIZE, ladder_max);
        transient_pool_ = std::make_unique<MemoryPool>(
            device_.device(), device_.physical_device(), block_bytes,
            MemoryPool::DEFAULT_SMALL_BLOCK_SIZE, ladder_max);

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

        // 6.5 异步标量回读槽（P0-2）：非阻塞取 loss 标量，避免每 step drain
        auto rb_r = init_scalar_readback_slots();
        if (!rb_r)
            return rb_r;

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

        // 7c. 独立模式复用 fence + command buffer（见成员注释；省去
        //     每算子 create/destroy + alloc/free 的 host/driver 开销）
        {
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;
            r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &solo_cmd_),
                __FILE__, __LINE__);
            if (!r)
                return r;

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            // 初始 unsignaled：vkQueueSubmit 要求 fence 未 signal
            r = detail::vk_check(
                vkCreateFence(device_.device(), &fence_info, nullptr, &solo_fence_),
                __FILE__, __LINE__);
            if (!r)
                return r;
        }

        // 9. 创建 tiled matmul pipeline（可选）
        const auto& tiled_spirv = get_matmul_tiled_spirv();
        if (!tiled_spirv.empty())
        {
            auto tp_r = VulkanPipeline::create_matmul(device_.device(), tiled_spirv);
            if (tp_r)
                matmul_tiled_pipeline_ = std::move(*tp_r);
        }

        // 9a. 创建小 N GEMV pipeline（可选；与 tiled 同 5×4B push / 3 binding，
        //     复用 create_matmul 布局）
        const auto& gemv_spirv = get_matmul_gemv_spirv();
        if (!gemv_spirv.empty())
        {
            auto gp_r = VulkanPipeline::create_matmul(device_.device(), gemv_spirv);
            if (gp_r)
                matmul_gemv_pipeline_ = std::move(*gp_r);
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

        // 9b'. 精度变体（Phase 2 in-kernel f16）：同一 shader 的 f16 存储版。
        //   设备未启用 SSBO 16 位存储时跳过创建 → 句柄为空 → 引擎走 f32 边界
        //   cast 回退（正确性不变，只是拿不到存储折半的带宽/显存收益）。
        if (device_.has_16bit_storage())
        {
            const auto& b16 = get_batched_matmul_f16_spirv();
            if (!b16.empty())
            {
                auto r16 = VulkanPipeline::create_generic(
                    device_.device(), b16, 3, 6 * sizeof(uint32_t));
                if (r16)
                    batched_matmul_f16_pipeline_ = std::move(*r16);
            }
            const auto& t16 = get_matmul_tiled_f16_spirv();
            if (!t16.empty())
            {
                auto r16 = VulkanPipeline::create_matmul(device_.device(), t16);
                if (r16)
                    matmul_tiled_f16_pipeline_ = std::move(*r16);
            }
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

        // 11. 创建 reduce pipeline（2 bindings, 20B push constants：
        //     rows/cols/mode/reduce_op/chunk_rows——chunk_rows>0 触发列归约
        //     两段式 pass1，见 reduce_gpu）
        const auto& reduce_spirv = get_reduce_spirv();
        if (!reduce_spirv.empty())
        {
            auto rp_r = VulkanPipeline::create_generic(
                device_.device(), reduce_spirv, 2, 5 * sizeof(uint32_t));
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

        // 13b. im2col / col2im pipeline（2 bindings, 9 uint = 36B push constants）
        {
            const auto& im_spirv = get_im2col_spirv();
            if (!im_spirv.empty())
            {
                auto r = VulkanPipeline::create_generic(
                    device_.device(), im_spirv, 2, 9 * sizeof(uint32_t));
                if (r) im2col_pipeline_ = std::move(*r);
            }
            const auto& c2_spirv = get_col2im_spirv();
            if (!c2_spirv.empty())
            {
                auto r = VulkanPipeline::create_generic(
                    device_.device(), c2_spirv, 2, 9 * sizeof(uint32_t));
                if (r) col2im_pipeline_ = std::move(*r);
            }
            const auto& gr_spirv = get_group_reduce_spirv();
            if (!gr_spirv.empty())
            {
                auto r = VulkanPipeline::create_generic(
                    device_.device(), gr_spirv, 2, 4 * sizeof(uint32_t));
                if (r) group_reduce_pipeline_ = std::move(*r);
            }
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
            // 精度变体（key 含 '#'，Phase 2 in-kernel f16）需要 SSBO 16 位存储；
            // 设备未启用 → 跳过注册（运行时按 (key,sig) 查不到 → 回退边界 cast）。
            if (std::string_view(fs.key).find('#') != std::string_view::npos &&
                !device_.has_16bit_storage())
                continue;
            const std::uint32_t num_bindings =
                static_cast<std::uint32_t>(fs.spec.views.size()) + 1;  // 输入 + 输出
            // 归约 kernel 的 push constants 多 uint rows + uint vector_out；
            // matmul 融合 kernel 多 uint rows + uint mm_k + uint mm_batch；
            // matmul+归约组合再多 uint mm_k + uint mm_batch（6 槽）；
            // fold（P-C1）= count, cols, rows, vector_out, fold_k（5 槽）——
            //   **必须与 run_fused_gpu 的 pc_base 逐形态一致**：漏分支会让
            //   range 少算，vkCmdPushConstants 超 range 部分被驱动丢弃 →
            //   fold_k 读未定义残留（曾致滑窗式错值，且残留随前序 op 漂移、
            //   表现为时对时错的假 PASS——教训 4.10 同类，改动 PC 形态时
            //   创建侧(本处)与写入侧(run_fused_gpu)必须同改）；
            // 另加 fs.view_param_count 个运行时视图参数槽（RowMod/RotateHalf）
            const std::uint32_t pc_base =
                (fs.spec.fold && fs.spec.fold->matmul)    ? 7u  // fold+mm 7 槽
              : (fs.spec.fold)                            ? 5u
              : (fs.reduce_axis >= 0 && fs.has_matmul)    ? 6u
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
                fused_fold_meta_.emplace(fs.key, FoldMeta{
                    fs.spec.fold
                        ? (fs.spec.fold->vec_state_len > 0 ? fs.spec.fold->vec_state_len : 1u)
                        : 0u,
                    // per_thread_row 判据与 generate_glsl 分派同源（v2 = 有
                    // vec 态或 mm 段；v1 = 纯标量）
                    static_cast<std::uint8_t>(fs.spec.fold &&
                        !(fs.spec.fold->vec_state_len > 0 || fs.spec.fold->matmul))});
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
    [[nodiscard]] bool has_gemv_pipeline() const noexcept { return matmul_gemv_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_batched_matmul_pipeline() const noexcept { return batched_matmul_pipeline_.handle() != VK_NULL_HANDLE; }
    // 精度变体（f16 存储）可用性：引擎据此决定"直读 f16"还是"边界 cast 回退"
    [[nodiscard]] bool has_batched_matmul_f16_pipeline() const noexcept { return batched_matmul_f16_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_matmul_tiled_f16_pipeline() const noexcept { return matmul_tiled_f16_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_rearrange_3d_pipeline() const noexcept { return rearrange_3d_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_elementwise_v2_pipeline() const noexcept { return elementwise_v2_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_reduce_pipeline() const noexcept { return reduce_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_broadcast_pipeline() const noexcept { return broadcast_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_transpose_pipeline() const noexcept { return transpose_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_im2col_pipeline() const noexcept { return im2col_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_col2im_pipeline() const noexcept { return col2im_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_group_reduce_pipeline() const noexcept { return group_reduce_pipeline_.handle() != VK_NULL_HANDLE; }
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
    // 延迟销毁队列总字节（探针归因用）：已析构但因帧未完成而尚未归还池的
    // 内存。解释"Tensor 已置空但 live 不降"的账目缺口。
    [[nodiscard]] VkDeviceSize pending_destroy_bytes()
    {
        std::lock_guard lock(pending_mutex_);
        VkDeviceSize total = 0;
        for (const auto& pd : pending_destroys_) total += pd.alloc.size;
        return total;
    }
    // 瞬态/持久分池选择器：batch 录制期→瞬态池，否则→持久池（纯组织性）。
    // 供 GpuTensorT 分配（create_empty / from_matrix / create_host_visible_empty）
    // 使用，使批量内的临时/激活与构建期的参数/权重分池驻留。
    [[nodiscard]] MemoryPool& alloc_pool() noexcept
    {
        return batch_mode_ ? *transient_pool_ : *memory_pool_;
    }
    // 异步标量回读槽位数（P0-2）：供 GpuEngine 暴露给调用方做环形复用。
    [[nodiscard]] static constexpr std::size_t scalar_readback_slot_count() noexcept
    {
        return SCALAR_READBACK_SLOTS;
    }
    [[nodiscard]] StagingRing& staging_ring() noexcept { return *staging_ring_; }
    [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
    [[nodiscard]] VkDescriptorPool gpu_tensor_pool() const noexcept { return gpu_tensor_pool_; }

    // ── 非阻塞收割所有已完成帧（不含空闲块归还）────────────────────
    // 显存峰值关键项（2026-09 探针实测）：原先只在 step 边界调用本逻辑，
    // 步内已析构中间张量的延迟销毁一直挂在"已完成但未被收割"的帧上，
    // 内存不还池 → backward 只能继续申请新底材；bench 配置实测 transient
    // live 3837MB 里 pending（已析构未还池）高达 3000MB，步内峰值 ≈ 稳态
    // 2.8×。故把收割逻辑提前到每个帧提交点（submit_frame_no_wait）。
    [[nodiscard]] Result<void> reap_completed_frames()
    {
        if (!initialized_)
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
        return {};
    }

    // ── 显存回收（L2）：先非阻塞收割已完成帧，再归还完全空闲的池底材 ──
    // 调用时机：step 边界（end_batch 提交完成、延迟销毁已收割之后）。
    // 未完成的帧留给环槽复用 / 下次 reap——不在 step 边界阻塞流水线。
    [[nodiscard]] Result<void> release_idle_pool_blocks()
    {
        auto r = reap_completed_frames();
        if (!r)
            return r;
        if (memory_pool_)
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
        // 提交后立即非阻塞收割其它已完成帧：让步内已析构张量的内存尽早
        // 还池，被后续分配复用（否则要等到环槽复用或 step 边界）。
        // NN_NO_EARLY_REAP=1 可关闭（A/B 对照用，见 bench/run_mem_ab.ps1）。
        if (early_reap_enabled_)
        {
            auto rr = reap_completed_frames();
            if (!rr)
                return std::unexpected(rr.error());
        }
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
        const auto t_disp_start = std::chrono::steady_clock::now();
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
            const bool prof = profile_ops_enabled();
            const auto t_rec = std::chrono::steady_clock::now();
            auto r = submit_and_wait(cmd, desc_set);
            if (prof)
                std::fprintf(stderr,
                    "[gpu-profile] record+setup=%lldus（提交/等待分段见上一条 solo_op）\n",
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                        t_rec - t_disp_start).count()));
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
        uint32_t transA = 0, uint32_t transB = 0,
        bool f16_io = false)
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

        // 1. 分配输出 Tensor（f16_io：按 2B/元素分配 + 纯绑定视图，见
        //    batched_matmul_gpu 的同款做法）
        const bool use_f16 = f16_io;
        if (use_f16 && !has_matmul_tiled_f16_pipeline())
            return std::unexpected(Error{
                "matmul_gpu: f16 pipeline 不可用（设备无 SSBO 16 位存储？）"});
        std::optional<GpuTensor> owned;
        if (use_f16)
        {
            auto f16r = GpuTensorF16::create_empty(M, N, *this);
            if (!f16r)
                return std::unexpected(f16r.error());
            owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
        }
        else
        {
            auto C_res = GpuTensor::create_empty(M, N, *this);
            if (!C_res)
                return std::unexpected(C_res.error());
            owned.emplace(std::move(*C_res));
        }
        GpuTensor C = *owned;

        // 2. 选择 Pipeline（优先级：f16 存储版 > 小 N GEMV > 粗化分块 > naive）
        //   GEMV：N ≤ GEMV_MAX_N（shader ROWS=4 行/WG，grid.x=ceil(M/4)）。
        //   tiled 在小 N 下 64×64 块 (64-N)/64 列空转，GEMV 每线程只算
        //   真实 N 列（借鉴 ggml mul_mat_vec 分派）。
        constexpr uint32_t GEMV_MAX_N = 8;
        constexpr uint32_t GEMV_ROWS = 4;   // 与 matmul_gemv.comp 的 ROWS 一致
        // subgroup≥4 门禁：shader red[..][64] 的容量假设（256/4 = 64 槽）——
        //   更小 subgroup 会溢出 64 槽上限、部分和不落表 → 静默错值
        //   （4.10 同类"只有 GPU 错"；实测桌面卡恒 ≥8，门禁是保险丝）
        const bool use_gemv = !use_f16 && has_gemv_pipeline() && N <= GEMV_MAX_N
                              && device_.subgroup_size() >= 4u;
        const bool use_tiled = (use_f16 || has_tiled_pipeline()) && !use_gemv;
        auto& pipeline = use_f16 ? matmul_tiled_f16_pipeline_
                       : use_gemv ? matmul_gemv_pipeline_
                       : use_tiled ? matmul_tiled_pipeline_
                                   : matmul_pipeline_;

        // 3. 复用通用 dispatch：2 输入 + 1 输出，push {M,N,K,transA,transB}
        //   Dispatch：按 shader 输出块尺寸计算工作组数
        //     matmul.comp（naive）：16×16 线程网格 = 16×16 输出块
        //     matmul_tiled.comp：64×64 输出块（BM/BN）
        //     matmul_gemv.comp：每 WG ROWS=4 行 × N 列（单维 grid.x）
        //   ⚠ 曾误用 WORKGROUP_SIZE=16 统一计算 → tiled 版 dispatch 出 16 倍
        //   冗余工作组（每 16×16 一个组而非 64×64），GPU 做 16 倍无效计算，
        //   matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。此处按实际块尺寸修复。
        const uint32_t push_data[5] = {M, N, K, transA, transB};
        std::vector<std::uint8_t> pc(sizeof(push_data));
        std::memcpy(pc.data(), push_data, sizeof(push_data));

        std::vector<GpuTensor> inputs{A, B};
        if (use_gemv)
        {
            auto r = dispatch_compute(pipeline, inputs, C, pc,
                (M + GEMV_ROWS - 1u) / GEMV_ROWS, 1u, 1u);
            if (!r)
                return std::unexpected(r.error());
            return C;
        }
        const uint32_t tile = use_tiled ? 64u : 16u;
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
    //
    // f16_io（Phase 2 in-kernel f16）：A/B 是 **f16 缓冲**、C 也按 f16 分配
    //   （2B/元素）→ 用 f16 存储版 pipeline（f16 载入 + f32 累加 + f16 写出），
    //   彻底消除"每个操作数抬 f32 的整份副本"。此时 A/B 只作**缓冲视图**使用
    //   （GpuTensorT<F32> 包裹 f16 buffer，字节布局由 pipeline 决定 —— 与
    //   run_fused_gpu 的 out_f16 同一套做法）。
    [[nodiscard]] Result<GpuTensor> batched_matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t batch,
        uint32_t transA = 0, uint32_t transB = 0,
        float alpha = 1.0f,
        bool f16_io = false)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (batch == 0)
            return std::unexpected(Error{"batched_matmul_gpu: batch must be > 0"});
        const bool use_f16 = f16_io && has_batched_matmul_f16_pipeline();
        if (f16_io && !use_f16)
            return std::unexpected(Error{
                "batched_matmul_gpu: f16 pipeline 不可用（设备无 SSBO 16 位存储？）"});
        if (!use_f16 && !has_batched_matmul_pipeline())
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
        //    f16：按 2B/元素分配，再用 GpuTensor(shared_buffer,...) 包一层纯
        //    绑定视图交给 dispatch（调用方按 f16_io 重贴 GpuTensorF16）。
        std::optional<GpuTensor> owned;
        if (use_f16)
        {
            auto f16r = GpuTensorF16::create_empty(
                static_cast<std::size_t>(batch) * M, N, *this);
            if (!f16r)
                return std::unexpected(f16r.error());
            owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
        }
        else
        {
            auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(batch) * M, N, *this);
            if (!C_res)
                return std::unexpected(C_res.error());
            owned.emplace(std::move(*C_res));
        }
        GpuTensor C = *owned;

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
        auto r = dispatch_compute(
            use_f16 ? batched_matmul_f16_pipeline_ : batched_matmul_pipeline_,
            inputs, C, pc, (N + BN - 1) / BN, (M + BM - 1) / BM, batch);
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
    // ── 算子 host 开销剖析开关（NN_GPU_PROFILE=1 时 stderr 打印分段耗时；
    //    未设置时进程内仅查询一次，之后纯 bool 读取）────────────────────
    [[nodiscard]] static bool profile_ops_enabled()
    {
        static const bool on = []() {
            // 复用 VulkanDevice::get_env（MSVC 下 _dupenv_s，绕开 getenv 弃用）
            const std::string v = VulkanDevice::get_env("NN_GPU_PROFILE");
            return !v.empty() && v[0] != '0';
        }();
        return on;
    }

    // 返回 (cmd, owns_cmd)。owns_cmd=true 时调用方需 submit_and_wait。
    [[nodiscard]] Result<std::pair<VkCommandBuffer, bool>> acquire_cmd()
    {
        if (batch_mode_)
        {
            batch_has_ops_ = true;  // P0-1：当前帧已含 op（空帧不提交）
            return std::make_pair(batch_cmd_, false);
        }

        // 独立模式复用 initialize() 预分配的 solo_cmd_：reset + begin，
        // 免去每算子 vkAllocateCommandBuffers/vkFreeCommandBuffers。
        // solo_cmd_ 仅在 submit_and_wait 等完 fence 后才会被再次取用，
        // 不存在 pending 期 reset；begin 失败时 buffer 留在 reset 后初态
        // （规范允许 reset 从错误态恢复），下次调用可重试。
        VkCommandBuffer cmd = solo_cmd_;
        if (cmd == VK_NULL_HANDLE)
            return std::unexpected(Error{"acquire_cmd: solo_cmd_ not initialized"});

        auto r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        return std::make_pair(cmd, true);
    }

    // ── 辅助：独立模式提交+等待（solo fence/cmd 复用，仅归还描述符集）────
    // cmd：acquire_cmd 返回的 solo_cmd_（非 batch 模式才会走到这里）
    // desc_set 可为 VK_NULL_HANDLE（fill_zero/copy_buffer 无描述符集）
    // NN_GPU_PROFILE=1：stderr 打印 host 分段耗时（µs）用于开销归因
    [[nodiscard]] Result<void> submit_and_wait(
        VkCommandBuffer cmd, VkDescriptorSet desc_set)
    {
        const bool prof = profile_ops_enabled();
        const auto t_begin = std::chrono::steady_clock::now();

        auto r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r)
        {
            if (desc_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(r.error());
        }
        const auto t_end = std::chrono::steady_clock::now();

        if (solo_fence_ == VK_NULL_HANDLE)
        {
            if (desc_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(Error{"submit_and_wait: solo_fence_ not initialized"});
        }
        // 上一次 submit 已 wait 到 signaled → 这里 reset 回 unsignaled 复用
        //（首次调用时是 init 创建的 fresh unsignaled，reset 为合法 no-op）
        r = detail::vk_check(
            vkResetFences(device_.device(), 1, &solo_fence_), __FILE__, __LINE__);
        if (!r)
        {
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
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, solo_fence_),
                __FILE__, __LINE__);
            if (!r)
            {
                if (desc_set != VK_NULL_HANDLE)
                    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
        }
        const auto t_submit = std::chrono::steady_clock::now();

        // 30 秒超时（单个原语，比 batch 短）
        constexpr uint64_t kSingleOpTimeoutNs = 30'000'000'000ULL;
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &solo_fence_, VK_TRUE, kSingleOpTimeoutNs),
            __FILE__, __LINE__);
        const auto t_wait = std::chrono::steady_clock::now();

        // 只归还描述符集；solo fence/cmd 留待下次复用（不销毁/不释放）
        if (desc_set != VK_NULL_HANDLE)
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
        const auto t_done = std::chrono::steady_clock::now();

        if (prof)
        {
            const auto us = [](auto a, auto b) {
                return static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
            };
            std::fprintf(stderr,
                "[gpu-profile] solo_op: end=%lldus submit=%lldus wait=%lldus "
                "cleanup=%lldus total=%lldus\n",
                us(t_begin, t_end), us(t_end, t_submit), us(t_submit, t_wait),
                us(t_wait, t_done), us(t_begin, t_done));
        }
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

        // vec4 kernel（与 DSL count/(256*vec_width) 同口径）：每线程 4 元素
        const uint32_t wg_count = (count + 256u * 4u - 1u) / (256u * 4u);
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
        // 输入按**底层 buffer** 绑定（元素类型由 shader 声明决定）→ 同一结构可在
        // f32 / f16 两种存储上运行（Phase 2 in-kernel f16 的带类型变体）。
        std::span<const GpuBuffer* const> inputs,
        std::span<const Scalar> consts,
        std::size_t rows, std::size_t cols,
        bool vector_out = false,
        std::span<const std::uint32_t> view_params = {},
        std::span<const Scalar> rparams = {},
        GpuTensor* output_override = nullptr,
        std::optional<std::uint32_t> matmul_k = std::nullopt,
        std::uint32_t matmul_batch = 1,
        // fold 收缩轴长度（P-C1：fold shader 传入填 fold_k 槽；非 fold 传 nullopt）
        std::optional<std::uint32_t> fold_k = std::nullopt,
        // 输出存储精度（Phase 2 in-kernel f16）：true → 分配 f16 字节布局的
        // 输出缓冲（返回的 GpuTensor 仅是**占位标签**，调用方按此标志重贴
        // GpuTensorF16 —— 见 GpuEngine::eval_expr）。
        bool out_f16 = false)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        const auto prof_t0 = std::chrono::steady_clock::now();
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
        // fold 形态（注册时从 FusedShader.spec.fold 取，与生成器同源）
        const bool has_fold_meta = fused_fold_meta_.count(shader_name) > 0;
        const FoldMeta fmeta = has_fold_meta
            ? fused_fold_meta_.at(shader_name) : FoldMeta{0u, 0u};
        const std::uint32_t fold_out = fmeta.out;
        const bool is_fold = has_fold_meta && fold_out > 0;
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
        if (!has_mm && matmul_k && !is_fold)
            return std::unexpected(Error{
                "run_fused_gpu: 非 matmul shader 收到了 matmul_k"});
        // fold 形态校验（与 matmul_k 同款对称防呆 + 调用约定）：
        //   fold 输出网格 = (rows, out_cols)，out_cols = vec_state_len（或 1），
        //   cols 参数必须等于它（GpuEngine/层调用约定）；PC cols 槽填 fold_k
        //   （视图/块轴列数=键长，见下方 cols32），不填输出列。
        //   fold+matmul 时 matmul_k/matmul_batch 同传（PC slot5/6）。
        if (is_fold && !fold_k)
            return std::unexpected(Error{
                "run_fused_gpu: fold shader 缺少 fold_k（收缩轴长度）"});
        if (!is_fold && fold_k)
            return std::unexpected(Error{
                "run_fused_gpu: 非 fold shader 收到了 fold_k"});
        if (is_fold && vector_out)
            return std::unexpected(Error{
                "run_fused_gpu: fold shader 以 vector_out 调度（应走 eval_expr）"});
        if (is_fold && cols == 0)
            return std::unexpected(Error{
                "run_fused_gpu: fold 调用约定 cols（=输出列数 out_cols）必须 > 0"});
        // （形状无关：out_cols 不做 shader 侧比对——veclen 不进 key，同一
        //   shader 服务任意 d_k，PC vector_out 槽按调用方 cols 运行时填充）
        // matmul+归约（S5）：raxis >= 0 时 mm_k 填入 PC 第 5 槽（见下方填充）

        // count 以 uint32 传入 shader（gl_GlobalInvocationID / push constant），
        // 必须保证 rows*cols 不溢出 uint32，否则分派与索引会静默截断。
        if (rows > 0 && cols > UINT32_MAX / rows)
            return std::unexpected(Error{
                "run_fused_gpu: rows*cols exceeds uint32 range"});
        const std::uint32_t count = static_cast<std::uint32_t>(rows * cols);

        // 1. 分配输出 Tensor（vector_out：归约向量原生形状 (rows,1)/(1,cols)）
        //    若调用方指定 output_override（dsl::compute_into 的原地目标），
        //    则复用其 buffer（形状必须匹配），结果直接写入目标 → 不额外分配、
        //    不额外拷贝。
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
        else if (out_f16)
        {
            // f16 输出：按 f16 字节布局分配（2B/元素，偶数槽位对齐见 create_f16_tensor）
            auto f16r = GpuTensorF16::create_empty(out_rows, out_cols, *this);
            if (!f16r) return std::unexpected(f16r.error());
            owned_output.emplace(f16r->shared_buffer(), out_rows, out_cols);
            output_ptr = &*owned_output;
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
            buf_infos[i] = {inputs[i]->impl(), 0, VK_WHOLE_SIZE};
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
        for (const auto* b : inputs)
            in_bufs.push_back(b->impl());
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // Push constants 布局与 glsl_gen.hpp 一致：
        //   逐元素: count, cols, [vp0..], c0..
        //   归约:   count, cols, rows, vector_out, [vp0..], c0..
        //   matmul: count, cols, rows, mm_k, mm_batch, [vp0..], c0..
        //   matmul+归约: count, cols, rows, vector_out, mm_k, mm_batch, [vp0..], c0..
        //
        // ⚠ 固定头长度必须**逐形态**与生成器的 PC 声明一致（下面四档）。历史 bug：
        //   "归约但无 matmul" 曾按 5 个 uint 计算（真实头部只有 4 个），导致常量池
        //   整体后移一个 uint → shader 从错位处读常量（实测把 select(cond,1,0) 的
        //   常量读成垃圾，GPU 上归约结果静默错值，而 CPU 正常）。
        const std::uint32_t pc_base =
            (is_fold && matmul_k)       ? 7u   // fold+mm: …, fold_k, mm_k, mm_batch
          : (is_fold)                   ? 5u   // fold: count,cols,rows,vector_out,fold_k
          : (raxis >= 0 && has_mm) ? 6u   // count, cols, rows, vector_out, mm_k, mm_batch
          : (raxis >= 0)           ? 4u   // count, cols, rows, vector_out
          : (has_mm)               ? 5u   // count, cols, rows, mm_k, mm_batch
          :                          2u;  // count, cols
        const std::uint32_t pc_uints = pc_base;
        std::vector<std::uint8_t> pc(
            (pc_uints + n_vp) * sizeof(std::uint32_t) + sizeof(Scalar) * consts.size()
            + sizeof(Scalar) * rparams.size());
        std::memcpy(pc.data(), &count, sizeof(std::uint32_t));
        // fold：PC cols 槽 = fold_k（生成器视图/块轴用输入列数 K 索引；
        //   输出网格列恒 1 不进 PC）。其余形态 = 调用方 cols。
        const std::uint32_t cols32 = is_fold
            ? *fold_k : static_cast<std::uint32_t>(cols);
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
        else if (is_fold)
        {
            // fold（P-C1/P-C2）：rows + **vector_out = 调用方输出列数**
            //   （形状无关——veclen 不进 key，PC 运行时填充；P-C1 v1 生成器
            //   不读此槽、填 1 无害）+ fold_k + [mm_k, mm_batch]（7 槽形态）。
            //   与生成器 PC 声明逐字段一致（4.10：创建侧 range 必须同改）
            const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
            const std::uint32_t vo = static_cast<std::uint32_t>(cols);
            std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32,
                        sizeof(std::uint32_t));
            std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &vo,
                        sizeof(std::uint32_t));
            std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &*fold_k,
                        sizeof(std::uint32_t));
            if (matmul_k)
            {
                std::memcpy(pc.data() + 5 * sizeof(std::uint32_t), &*matmul_k,
                            sizeof(std::uint32_t));
                std::memcpy(pc.data() + 6 * sizeof(std::uint32_t), &matmul_batch,
                            sizeof(std::uint32_t));
            }
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
        // 列归约 = ceil(cols/32) 个工作组（每工作组 32 列 tile、warp=行块，
        // 与生成器 l/wb/col 结构及 OP 级 reduce.comp 契约同源）；
        // matmul 分块（S5）= (ceil(cols/BLOCK), ceil(m_per/BLOCK), matmul_batch)，
        // BLOCK 与 glsl_gen 生成的输出块一致（EXPR_MATMUL_BLOCK=64：每工作组
        // 64×64 输出块、16×16 线程、每线程 4×4 寄存器分块）
        const std::uint32_t vec_width = fused_vec_width_.count(shader_name)
            ? fused_vec_width_.at(shader_name) : 1u;
        if (has_mm && raxis < 0 && !is_fold)
        {
            const std::uint32_t wg_x =
                (static_cast<std::uint32_t>(cols) + nn::EXPR_MATMUL_BLOCK - 1u)
                / nn::EXPR_MATMUL_BLOCK;
            // batch（S7）：dispatch z = 批次，A/B 按 batch 垂直切分 → y 只覆盖
            // **批内**行 m_per = rows/mm_batch（生成器 main/load_tiles 均以 m_per
            // 为界、写回守卫 rr < m_per）。按总 rows 派 y 会把 batch 在 y/z 数
            // 两遍 → 工作量 ∝ batch²：多余 (BH−1)/BH 的 WG 跑完整条 mm_k 流水
            // 后整块丢弃（结果仍正确 → 对拍测不出；AGENTS §12 ⑤ batch=32
            // train 异常的根因）。rows = batch*M 按契约整除。
            const std::uint32_t rows_u = static_cast<std::uint32_t>(rows);
            const std::uint32_t m_per  = rows_u / matmul_batch;
            const std::uint32_t wg_y =
                (m_per + nn::EXPR_MATMUL_BLOCK - 1u) / nn::EXPR_MATMUL_BLOCK;
            // batch（S7）：dispatch z = 批次，A/B 按 batch 垂直切分
            vkCmdDispatch(cmd, wg_x, wg_y, matmul_batch);
        }
        else
        {
            // fold 的 dispatch 分两种（FoldMeta，与生成器分派同源）：
            //   v1 标量 fold（每线程一行，row=gl_GlobalInvocationID.x）：
            //     count=rows×1 → ceil(count/256) 恰 = ceil(rows/256) ✓
            //   v2 双域 fold（每 WG NR 行，row=wg*NR+ri，EXPR_FOLD_ROWS_PER_WG
            //     与生成器行循环同源）：**wg = ceil(rows/NR)**
            //     （按 ceil(count/256)=ceil(rows×out/256) 派会丢行！）
            const std::uint32_t wg_count =
                (is_fold && !fmeta.per_thread_row)
                    ? (static_cast<std::uint32_t>(rows)
                       + nn::EXPR_FOLD_ROWS_PER_WG - 1u)
                        / nn::EXPR_FOLD_ROWS_PER_WG
                : (raxis == 0) ? static_cast<std::uint32_t>(rows)
                : (raxis == 1) ? (static_cast<std::uint32_t>(cols) + 31u) / 32u
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
        if (profile_ops_enabled())
        {
            const long long dt = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - prof_t0).count();
            // fold 按形态如实打印（is_fold；旧表达式对 v1 fold 打 0、对非
            // fold 打 1）；out= 运行时列数（veclen 不进 key，注册侧值可能
            // 与调用方实际 cols 不同——按注册侧打印曾误导归因）
            std::fprintf(stderr,
                "[gpu-profile] fused fold=%d out=%u total=%lldus key=%.16s\n",
                static_cast<int>(is_fold),
                static_cast<std::uint32_t>(cols), dt, shader_name.c_str());
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
    // Push Constants (20 bytes): rows, cols, mode, reduce_op, chunk_rows
    // Bindings: In(0), Out(1)
    // mode: 0=row_reduce → out(rows,1), 1=col_reduce → out(1,cols)
    // reduce_op: 0=sum, 1=max
    // 列归约大行数走两段式 partials（同 cmd 双 dispatch 单次提交，见分支内注释）
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

        // ── 列归约大行数：两段式 partials（ggml rms_norm_partials 模式）──
        // 单段 dispatch = ceil(cols/32) 个 WG（5244² 仅 164 vs 136 并发槽 →
        // 1.2 波，末波只剩 ~20% 机器喂内存 → 带宽被波尾压约两成）。
        // pass1 把行向切 nchunk 块（dispatch y = nchunk）→ WG 数 ×nchunk，
        // 每 WG 归约一行块写 partials(nchunk, cols)；pass2 复用整表路径合并
        // （行数 = nchunk ≤ 8，成本可忽略）。两 pass 录进同一 cmd 单次提交
        // → 固定提交开销不翻倍。小行数保持单段（零开销）。
        uint32_t nchunk = 1u;
        if (mode == 1u)
        {
            const uint32_t tiles = (cols + 31u) / 32u;
            nchunk = (rows + 1023u) / 1024u;
            if (nchunk > 8u) nchunk = 8u;
            const uint32_t want = (512u + tiles - 1u) / tiles;   // ≥512 WG ≈ 4 波
            if (want > nchunk) nchunk = want;
            if (nchunk > 8u) nchunk = 8u;
            // 小形状回落单段：512² 交错 A/B 实测两段 −16%（第二遍+屏障对
            // ~10µs kernel 净亏），1024² 起两者持平或两段转优 → 边界 512K 元素
            if (rows < 256u ||
                static_cast<std::uint64_t>(rows) * cols < 524'288ull)
                nchunk = 1u;
        }
        // 两段式启用条件（同负载窗交错 A/B 定案：TP/SP 两二进制逐轮换序、
        // warmup 30、best-of-10；同码对照噪声地板 ±5%）：
        //   5244² TP 4/5 胜（中位 0.405 vs 0.430，−6%；单段 1.2 波尾欠喂被消除）
        //   4096² SP 5/5 胜 ~2%（0.277 vs 0.283；单波本已喂满，TP 纯付 ~6µs）
        //   512² TP −16%（对 ~10µs kernel 第二遍+屏障净亏）→ 尺寸护栏只排除它
        const bool two_pass = (nchunk > 1u);
        const uint32_t chunk_rows = two_pass ? (rows + nchunk - 1u) / nchunk : 0u;

        if (two_pass)
        {
            if (!reduce_partial_ ||
                static_cast<uint32_t>(reduce_partial_->rows()) != nchunk ||
                static_cast<uint32_t>(reduce_partial_->cols()) != cols)
            {
                auto pres = GpuTensor::create_empty(nchunk, cols, *this);
                if (!pres) return std::unexpected(pres.error());
                reduce_partial_ = std::move(*pres);
            }
            const GpuTensor& partial = *reduce_partial_;

            auto ds1_r = alloc_desc_set(reduce_pipeline_.descriptor_layout());
            if (!ds1_r) return std::unexpected(ds1_r.error());
            VkDescriptorSet ds1 = *ds1_r;
            auto ds2_r = alloc_desc_set(reduce_pipeline_.descriptor_layout());
            if (!ds2_r) return std::unexpected(ds2_r.error());
            VkDescriptorSet ds2 = *ds2_r;

            VkDescriptorBufferInfo infos1[2]{
                {input.buffer().impl(), 0, VK_WHOLE_SIZE},
                {partial.buffer().impl(), 0, VK_WHOLE_SIZE},
            };
            VkDescriptorBufferInfo infos2[2]{
                {partial.buffer().impl(), 0, VK_WHOLE_SIZE},
                {output.buffer().impl(), 0, VK_WHOLE_SIZE},
            };
            VkWriteDescriptorSet writes[4]{};
            for (int i = 0; i < 2; ++i)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = ds1;
                writes[i].dstBinding = static_cast<uint32_t>(i);
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos1[i];
                writes[2 + i] = writes[i];
                writes[2 + i].dstSet = ds2;
                writes[2 + i].pBufferInfo = &infos2[i];
            }
            vkUpdateDescriptorSets(device_.device(), 4, writes, 0, nullptr);

            auto cmd_r = acquire_cmd();
            if (!cmd_r)
            {
                VkDescriptorSet free_sets[2] = {ds1, ds2};
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 2, free_sets);
                return std::unexpected(cmd_r.error());
            }
            auto [cmd, owns_cmd] = *cmd_r;

            // pass1：行块级部分归约 → partials
            record_input_barriers(cmd, {input.buffer().impl()});
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                reduce_pipeline_.handle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                reduce_pipeline_.pipeline_layout(), 0, 1, &ds1, 0, nullptr);
            const uint32_t push1[5] = {rows, cols, 1u, reduce_op, chunk_rows};
            vkCmdPushConstants(cmd, reduce_pipeline_.pipeline_layout(),
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push1), push1);
            vkCmdDispatch(cmd, (cols + 31u) / 32u, nchunk, 1);

            // partials：写 → 读（同 cmd 内跨 dispatch 依赖）
            record_input_barriers(cmd, {partial.buffer().impl()});

            // pass2：合并 partials（rows = nchunk ≤ 8，复用 mode1 整表路径）
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                reduce_pipeline_.pipeline_layout(), 0, 1, &ds2, 0, nullptr);
            const uint32_t push2[5] = {nchunk, cols, 1u, reduce_op, 0u};
            vkCmdPushConstants(cmd, reduce_pipeline_.pipeline_layout(),
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push2), push2);
            vkCmdDispatch(cmd, (cols + 31u) / 32u, 1, 1);
            record_output_barrier(cmd, output.buffer().impl());

            if (owns_cmd)
            {
                auto r = submit_and_wait(cmd, ds1);   // 等 fence 后归还 ds1
                if (!r) return std::unexpected(r.error());
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &ds2);
            }
            return output;
        }

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

        const uint32_t push_data[5] = {rows, cols, mode, reduce_op, 0u};
        vkCmdPushConstants(cmd, reduce_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // 行：每 WG 256 线程协作归约一行 → rows 个 WG；
        // 列：每 WG 32 列 tile（lane=列、warp=行块，合并访问；与
        //     reduce.comp 的 WG 结构契约同源）→ (cols+31)/32 个 WG
        const uint32_t workgroups = (mode == 0) ? rows : (cols + 31u) / 32u;
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

        // vec4 kernel（与 elementwise_v2 / DSL 同口径）：每线程 4 元素
        const uint32_t total = rows * cols;
        const uint32_t wg_count = (total + 256u * 4u - 1u) / (256u * 4u);
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
    // 异步标量回读（P0-2）—— 语义见成员声明处
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> init_scalar_readback_slots()
    {
        rb_slots_.resize(SCALAR_READBACK_SLOTS);
        for (auto& s : rb_slots_)
        {
            VkCommandBufferAllocateInfo ca{};
            ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ca.commandPool = command_pool_;
            ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ca.commandBufferCount = 1;
            auto r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &ca, &s.cmd), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());

            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            r = detail::vk_check(
                vkCreateFence(device_.device(), &fi, nullptr, &s.fence), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());

            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = sizeof(float);
            bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            r = detail::vk_check(
                vkCreateBuffer(device_.device(), &bi, nullptr, &s.host), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());

            VkMemoryRequirements mr{};
            vkGetBufferMemoryRequirements(device_.device(), s.host, &mr);
            constexpr VkMemoryPropertyFlags kVis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            constexpr VkMemoryPropertyFlags kCoh = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            auto ar = memory_pool().allocate(mr, kVis | kCoh, kVis);
            if (!ar) return std::unexpected(ar.error());
            s.alloc = *ar;
            r = detail::vk_check(
                vkBindBufferMemory(device_.device(), s.host, s.alloc.memory, s.alloc.offset),
                __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
            r = detail::vk_check(
                vkMapMemory(device_.device(), s.alloc.memory, s.alloc.offset,
                            sizeof(float), 0, &s.mapped),
                __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
        }
        return {};
    }

    void destroy_scalar_readback_slots()
    {
        for (auto& s : rb_slots_)
        {
            if (s.mapped != nullptr && s.alloc.valid())
            {
                vkUnmapMemory(device_.device(), s.alloc.memory);
                s.mapped = nullptr;
            }
            if (s.host != VK_NULL_HANDLE)
                vkDestroyBuffer(device_.device(), s.host, nullptr);
            if (s.fence != VK_NULL_HANDLE)
                vkDestroyFence(device_.device(), s.fence, nullptr);
            if (s.alloc.valid())
                memory_pool().free(s.alloc);
            s.host = VK_NULL_HANDLE;
            s.fence = VK_NULL_HANDLE;
            s.cmd = VK_NULL_HANDLE;  // command buffer 随 command pool 释放
            s.pending = false;
        }
        rb_slots_.clear();
    }

    // 录制一次 4B D2H 拷贝并提交（**不等待**）。
    // 调用约定：必须在产出该标量的主帧已提交之后调用（同队列 FIFO ⇒
    // 拷贝排在生产者之后；若在主帧提交前提交，会读到上一轮的旧值）。
    [[nodiscard]] Result<void> submit_scalar_readback(std::size_t slot, VkBuffer src)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (slot >= rb_slots_.size())
            return std::unexpected(Error{"scalar readback: slot out of range"});
        auto& s = rb_slots_[slot];

        // 上一轮的值尚未取走：等它就绪（防御路径；正常调用方先 poll）
        if (s.pending)
        {
            constexpr uint64_t kTimeoutNs = 60'000'000'000ULL;
            const VkResult wr =
                vkWaitForFences(device_.device(), 1, &s.fence, VK_TRUE, kTimeoutNs);
            if (wr == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return std::unexpected(Error{
                    "scalar readback: VK_ERROR_DEVICE_LOST（GPU 已被 TDR 重置）"});
            }
            if (wr != VK_SUCCESS)
                return std::unexpected(
                    Error{"scalar readback: fence wait failed " +
                          std::to_string(static_cast<int>(wr))});
            s.pending = false;
        }

        auto r = detail::vk_check(vkResetCommandBuffer(s.cmd, 0), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(s.cmd, &cbi), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkBufferCopy cp{};
        cp.srcOffset = 0;
        cp.dstOffset = 0;
        cp.size = sizeof(float);  // (1,1) f32 标量
        vkCmdCopyBuffer(s.cmd, src, s.host, 1, &cp);

        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = s.host;
        b.offset = 0;
        b.size = sizeof(float);
        vkCmdPipelineBarrier(s.cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &b, 0, nullptr);

        r = detail::vk_check(vkEndCommandBuffer(s.cmd), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());
        r = detail::vk_check(vkResetFences(device_.device(), 1, &s.fence), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &s.cmd;
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &si, s.fence), __FILE__, __LINE__);
        }
        if (!r) return std::unexpected(r.error());
        s.pending = true;
        return {};
    }

    // 非阻塞查询：就绪则写出 out 并返回 true；未就绪返回 false（不阻塞）。
    [[nodiscard]] Result<bool> poll_scalar_readback(std::size_t slot, float& out)
    {
        if (slot >= rb_slots_.size())
            return std::unexpected(Error{"scalar readback: slot out of range"});
        auto& s = rb_slots_[slot];
        if (!s.pending)
            return false;
        const VkResult st = vkGetFenceStatus(device_.device(), s.fence);
        if (st == VK_NOT_READY)
            return false;
        if (st == VK_ERROR_DEVICE_LOST)
        {
            device_lost_ = true;
            return std::unexpected(Error{
                "scalar readback: VK_ERROR_DEVICE_LOST（GPU 已被 TDR 重置）"});
        }
        if (st != VK_SUCCESS)
            return std::unexpected(
                Error{"scalar readback: fence status " +
                      std::to_string(static_cast<int>(st))});

        // 非 coherent 内存需 invalidate；coherent 下该调用是合法的 no-op。
        // size = VK_WHOLE_SIZE 时 offset 无对齐要求（VU 只约束非 WHOLE_SIZE）。
        VkMappedMemoryRange mr{};
        mr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mr.memory = s.alloc.memory;
        mr.offset = s.alloc.offset;
        mr.size = VK_WHOLE_SIZE;
        if (vkInvalidateMappedMemoryRanges(device_.device(), 1, &mr) != VK_SUCCESS)
            return std::unexpected(Error{"scalar readback: invalidate mapped range failed"});

        std::memcpy(&out, s.mapped, sizeof(float));
        s.pending = false;
        return true;
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

        // 砖块化 dispatch：(8, 8, n_bricks)——每 WG 覆盖 64×64 输出块，
        // 8×8 个 WG 组成一砖（并发足迹聚集，见 transpose.comp 头注释）。
        // 越界 tile 由 shader 早退（dispatch 固定 8×8，末砖可不满）。
        // ⚠️ x 必须是 8 而非 16：shader 按 8 宽砖解算 tile_c =
        // (bz%bx_count)*8 + gl_WorkGroupID.x，假定 gl_WorkGroupID.x ∈ [0,8)。
        // 曾误派发 16 宽 → tile_c 跨两砖且越界 tile 只早退一半，
        // 行>512 且列>512 时静默只写前 512 行（issue #13 P0-①）。
        const uint32_t tx = (C + 63u) / 64u;
        const uint32_t ty = (R + 63u) / 64u;
        const uint32_t bx = (tx + 7u) / 8u;
        const uint32_t by = (ty + 7u) / 8u;
        vkCmdDispatch(cmd, 8, 8, bx * by);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    // ══════════════════════════════════════════════════════════════════
    // ══════════════════════════════════════════════════════════════════
    // im2col_gpu / col2im_gpu — 卷积/池化窗口展开（纯数据搬运，无算法）
    //   Push Constants (36 bytes): C,H,W,k,stride,pad,OH,OW,B
    //   Bindings: In(0), Out(1)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> im2col_gpu(
        const GpuTensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_im2col_pipeline())
            return std::unexpected(Error{"im2col pipeline not available"});
        if (C == 0 || H == 0 || W == 0 || k == 0 || stride == 0 || OH == 0 || OW == 0)
            return std::unexpected(Error{"im2col_gpu: C/H/W/k/stride/OH/OW must be > 0"});
        if (x.rows() != C * H * W)
            return std::unexpected(Error{"im2col_gpu: x must be (C*H*W, B)"});

        const std::size_t B    = x.cols();
        const std::size_t rows = C * k * k;
        const std::size_t cols = B * OH * OW;

        auto out_res = GpuTensor::create_empty(rows, cols, *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        auto ds_r = alloc_desc_set(im2col_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        VkDescriptorBufferInfo buf_infos[2]{
            {x.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        record_input_barriers(cmd, {x.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            im2col_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            im2col_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[9] = {
            static_cast<uint32_t>(C),      static_cast<uint32_t>(H),
            static_cast<uint32_t>(W),      static_cast<uint32_t>(k),
            static_cast<uint32_t>(stride), static_cast<uint32_t>(pad),
            static_cast<uint32_t>(OH),     static_cast<uint32_t>(OW),
            static_cast<uint32_t>(B)};
        vkCmdPushConstants(cmd, im2col_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const std::size_t total = rows * cols;
        const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

    [[nodiscard]] Result<GpuTensor> col2im_gpu(
        const GpuTensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_col2im_pipeline())
            return std::unexpected(Error{"col2im pipeline not available"});
        if (C == 0 || H == 0 || W == 0 || k == 0 || stride == 0 || OH == 0 || OW == 0)
            return std::unexpected(Error{"col2im_gpu: C/H/W/k/stride/OH/OW must be > 0"});
        const std::size_t P = OH * OW;
        if (col.rows() != C * k * k || col.cols() == 0 || col.cols() % P != 0)
            return std::unexpected(Error{"col2im_gpu: col must be (C*k*k, B*OH*OW)"});

        const std::size_t B    = col.cols() / P;
        const std::size_t rows = C * H * W;
        const std::size_t cols = B;

        auto out_res = GpuTensor::create_empty(rows, cols, *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        auto ds_r = alloc_desc_set(col2im_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        VkDescriptorBufferInfo buf_infos[2]{
            {col.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        record_input_barriers(cmd, {col.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            col2im_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            col2im_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[9] = {
            static_cast<uint32_t>(C),      static_cast<uint32_t>(H),
            static_cast<uint32_t>(W),      static_cast<uint32_t>(k),
            static_cast<uint32_t>(stride), static_cast<uint32_t>(pad),
            static_cast<uint32_t>(OH),     static_cast<uint32_t>(OW),
            static_cast<uint32_t>(B)};
        vkCmdPushConstants(cmd, col2im_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const std::size_t total = rows * cols;
        const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
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
    // grouped_reduce_gpu — 分组归约（沿行方向按固定长度 R 分组求和/求最大）
    //   x (G*R, N) → out (G, N)
    //   Push Constants (16 bytes): G, R, N, reduce_op
    //   Bindings: In(0), Out(1)；Dispatch: ceil(G*N / 256)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> grouped_reduce_gpu(
        const GpuTensor& x, std::size_t G, std::size_t R, bool is_max)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_group_reduce_pipeline())
            return std::unexpected(Error{"group_reduce pipeline not available"});
        if (G == 0 || R == 0)
            return std::unexpected(Error{"grouped_reduce_gpu: G/R must be > 0"});
        if (x.rows() != G * R)
            return std::unexpected(Error{"grouped_reduce_gpu: x must be (G*R, N)"});
        const std::size_t N = x.cols();
        if (N == 0)
            return std::unexpected(Error{"grouped_reduce_gpu: N must be > 0"});

        auto out_res = GpuTensor::create_empty(G, N, *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        auto ds_r = alloc_desc_set(group_reduce_pipeline_.descriptor_layout());
        if (!ds_r) return std::unexpected(ds_r.error());
        VkDescriptorSet desc_set = *ds_r;

        VkDescriptorBufferInfo buf_infos[2]{
            {x.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            return std::unexpected(cmd_r.error());
        }
        auto [cmd, owns_cmd] = *cmd_r;

        record_input_barriers(cmd, {x.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            group_reduce_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            group_reduce_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[4] = {
            static_cast<uint32_t>(G), static_cast<uint32_t>(R),
            static_cast<uint32_t>(N), is_max ? 1u : 0u};
        vkCmdPushConstants(cmd, group_reduce_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const std::size_t total = G * N;
        const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
        vkCmdDispatch(cmd, wg_count, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, desc_set);
            if (!r) return std::unexpected(r.error());
        }
        return output;
    }

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
        // vec4 kernel（与 elementwise/broadcast/DSL 同口径）：每线程 4 元素
        const uint32_t wg_count = (total + 256u * 4u - 1u) / (256u * 4u);
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
