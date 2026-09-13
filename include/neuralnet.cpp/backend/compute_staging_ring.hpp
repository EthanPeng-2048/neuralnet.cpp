// ── compute_staging_ring.hpp ─────────────────────────────────────────────────────
// 环形 Staging 缓冲区
//
// 职责：
//   - 管理 HOST_VISIBLE 缓冲区用于 CPU↔GPU 数据传输
//   - 环形分配，避免频繁创建/销毁
//   - Fence 同步确保数据安全
//
// 设计（依据性能审查报告优化）：
//   - 预分配 N 个 region（默认 2 个，每个 64MB → 总 128MB）
//   - 旧默认 4×256MB=1GB 预分配过大；实际训练单次 PCIe 传输量
//     通常 << 64MB（最大 token_emb 上传约 5MB）
//   - 内存预算上限保留 max_host_visible / 16，避免在小显存机器上过度分配
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../core_errors.hpp"
#include "../core_observer_ptr.hpp"
#include "../core_config.hpp"
#include "compute_memory_pool.hpp"

namespace nn
{

class StagingRing
{
public:
    // 默认 64MB × 2 region（实际分配大小由 initialize() 动态计算，
    // 会根据 host-visible 显存大小自动扩容，此值仅作为下限参考）
    static constexpr std::size_t DEFAULT_REGION_SIZE = 64ull * 1024 * 1024; // 64MB
    static constexpr std::size_t DEFAULT_NUM_REGIONS = 4;  // 2→4：降低 acquire 阻塞概率
    // host-visible 显存预算占比的倒数（实际取 1/HOST_VISIBLE_FRACTION）
    static constexpr VkDeviceSize HOST_VISIBLE_FRACTION = 16;
    // 动态计算 staging 大小的下限，避免小显存机器分配过小
    static constexpr VkDeviceSize MIN_REGION_SIZE = 32ull * 1024 * 1024;
    // 动态计算 staging 大小的上限（借鉴 llama.cpp 按需小 staging）：
    // 大上传/下载已由分块逻辑（每块 ≤ region）自动切分，故 staging 无需按
    // 整机 host heap 的 1/16 常驻（V100 上曾达 2×2GB=4GB host 预算）。
    // 封顶 256MB/region → 2 regions 共 512MB，共享显存架构下大幅释放预算。
    static constexpr VkDeviceSize MAX_REGION_SIZE = 256ull * 1024 * 1024;

private:
    // Staging region 内部结构（外部通过索引访问，无需直接使用此类型）
    struct Region
    {
        MemoryPool::Allocation alloc;
        VkBuffer buffer = VK_NULL_HANDLE;
        void* mapped_ptr = nullptr;
        VkFence fence = VK_NULL_HANDLE;
        // 跨 submit 数据依赖（P0-1 修复）：本 region 的上传 copy 以该信号量
        // 为提交期信号；后续读取"由该上传写入的 buffer"的 submit（download /
        // matmul / batch 帧）在 VkSubmitInfo.pWaitSemaphores 中等它。
        // 背景：单队列 FIFO 只是执行顺序保证，实测本驱动（NVIDIA + Windows）
        // 在消费方 submit 紧跟上传 submit（<~2ms）时，消费方 GPU 操作会读到
        // 上传写入的旧值（零），必须显式建立跨 submit 依赖。
        //
        // 必须用**时间线信号量**（timeline_=true）：
        //   - 同一 value 可以被任意多个 submit 等待（binary 一次 signal 只能
        //     被一个 wait 消费 → 第二个消费者等到"永远不会来的信号"→ 队列
        //     永久阻塞，AMD 老驱动实测直接挂死，校验层报
        //     VUID-vkQueueSubmit-pWaitSemaphores-03238）；
        //   - 等待已达成/更小的 value 是 no-op，重复等待天然幂等。
        // 设备不支持时间线信号量时（timeline_=false）不创建信号量，改由
        // collect_staging_waits() → drain_in_flight() 在 host 侧阻塞等待，
        // 正确性优先、放弃 P0-1 的非阻塞流水线。
        VkSemaphore semaphore = VK_NULL_HANDLE;
        // 本 region 最近一次上传的信号 value（时间线信号量用；单调递增）
        std::uint64_t signal_value = 0;
        // 专属 command buffer（P0-1 修复）：上传 copy 命令录在这里，**永不
        // 在 pending 状态释放**——VUID-vkFreeCommandBuffers-pCommandBuffers-
        // 00058 禁止释放 pending（已提交未 signal）的 command buffer。旧实
        // 现"submit 后立即可复用/释放"是规范违规：实测导致驱动通道排序失
        // 效（fence 提前 signal、跨 submit 乱序执行）。acquire 等完 fence
        // 后该 cmd 离开 pending（invalid），vkResetCommandBuffer 复用合法。
        // 新分配时为 invalid 状态，首次使用前 vkResetCommandBuffer。
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool in_flight = false;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    observer_ptr<MemoryPool> pool_;
    std::vector<Region> regions_;
    std::size_t region_size_;
    std::atomic<std::size_t> current_{0};
    bool timeline_ = false;  // 用时间线信号量（否则 host 等 fence 回退）

public:
    StagingRing() = default;

    [[nodiscard]] Result<void> initialize(
        VkDevice device, VkPhysicalDevice physical_device,
        VkCommandPool cmd_pool, MemoryPool& pool,
        bool use_timeline = false,
        std::size_t region_size = DEFAULT_REGION_SIZE,
        std::size_t num_regions = DEFAULT_NUM_REGIONS)
    {
        device_ = device;
        physical_device_ = physical_device;
        cmd_pool_ = cmd_pool;
        pool_.reset(&pool);
        timeline_ = use_timeline;

        // 动态计算 Staging 大小：取 host-visible 显存的 1/16，
        // 夹在 [MIN_REGION_SIZE, MAX_REGION_SIZE] 之间（下限防过小、
        // 上限防大内存机器预占整块 host 预算；超大传输自动分块）。
        // 用户传入的 region_size 作为额外参考（取两者中较大值）。
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
        VkDeviceSize max_host_visible = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
        {
            if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                uint32_t heap_idx = mem_props.memoryTypes[i].heapIndex;
                if (mem_props.memoryHeaps[heap_idx].size > max_host_visible)
                    max_host_visible = mem_props.memoryHeaps[heap_idx].size;
            }
        }

        VkDeviceSize calculated_size = max_host_visible / HOST_VISIBLE_FRACTION;
        calculated_size = std::clamp(calculated_size, MIN_REGION_SIZE, MAX_REGION_SIZE);

        // 最终取 用户请求值 与 动态计算值 的较大者
        // （上限已封顶，避免小显存机器被 staging 占满预算）
        region_size_ = std::max(
            static_cast<VkDeviceSize>(region_size), calculated_size);

        regions_.resize(num_regions);
        for (std::size_t i = 0; i < num_regions; ++i)
        {
            auto& r = regions_[i];

            // 创建 buffer
            VkBufferCreateInfo buf_info{};
            buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.size = region_size_;
            buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkResult res = vkCreateBuffer(device_, &buf_info, nullptr, &r.buffer);
            if (res != VK_SUCCESS)
                return std::unexpected(Error{"vkCreateBuffer failed: " + std::to_string(res)});

            // 分配内存
            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device_, r.buffer, &mem_reqs);

            auto alloc_r = pool_->allocate(
                mem_reqs,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!alloc_r)
                return std::unexpected(alloc_r.error());
            r.alloc = *alloc_r;

            // 绑定内存
            res = vkBindBufferMemory(device_, r.buffer, r.alloc.memory, r.alloc.offset);
            if (res != VK_SUCCESS)
                return std::unexpected(Error{"vkBindBufferMemory failed: " + std::to_string(res)});

            // 映射内存
            res = vkMapMemory(device_, r.alloc.memory, r.alloc.offset, r.alloc.size, 0, &r.mapped_ptr);
            if (res != VK_SUCCESS)
                return std::unexpected(Error{"vkMapMemory failed: " + std::to_string(res)});

            // 创建 fence
            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // 初始状态为 signaled
            res = vkCreateFence(device_, &fence_info, nullptr, &r.fence);
            if (res != VK_SUCCESS)
                return std::unexpected(Error{"vkCreateFence failed: " + std::to_string(res)});

            // 创建跨 submit 信号量（仅时间线模式；初始 value 0 = 未 signal，
            // 见 Region::semaphore 注释）
            if (timeline_)
            {
                VkSemaphoreTypeCreateInfo type_info{};
                type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
                type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                type_info.initialValue = 0;

                VkSemaphoreCreateInfo sema_info{};
                sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                sema_info.pNext = &type_info;
                res = vkCreateSemaphore(device_, &sema_info, nullptr, &r.semaphore);
                if (res != VK_SUCCESS)
                    return std::unexpected(Error{
                        "vkCreateSemaphore(timeline) failed: " + std::to_string(res)});
            }

            // 分配专属 command buffer（见 Region::cmd 注释：禁止 pending 释放）
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = cmd_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;
            res = vkAllocateCommandBuffers(device_, &cmd_alloc, &r.cmd);
            if (res != VK_SUCCESS)
                return std::unexpected(Error{"vkAllocateCommandBuffers failed: " + std::to_string(res)});
        }

        return {};
    }

    ~StagingRing()
    {
        if (device_ == VK_NULL_HANDLE)
            return;

        for (auto& r : regions_)
        {
            if (r.fence != VK_NULL_HANDLE)
                vkDestroyFence(device_, r.fence, nullptr);
            if (r.semaphore != VK_NULL_HANDLE)
                vkDestroySemaphore(device_, r.semaphore, nullptr);
            if (r.cmd != VK_NULL_HANDLE)
                vkFreeCommandBuffers(device_, cmd_pool_, 1, &r.cmd);
            if (r.buffer != VK_NULL_HANDLE)
                vkDestroyBuffer(device_, r.buffer, nullptr);
            // MemoryPool 会自动释放内存
        }
    }

    // 禁止拷贝和移动
    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;
    StagingRing(StagingRing&&) = delete;
    StagingRing& operator=(StagingRing&&) = delete;

    // 获取下一个可用 region 的索引
    [[nodiscard]] std::size_t acquire()
    {
        std::size_t idx = current_.fetch_add(1, std::memory_order_relaxed) % regions_.size();
        auto& r = regions_[idx];

        // 等待该 region 的 fence（如果正在使用中）
        if (r.in_flight)
        {
            vkWaitForFences(device_, 1, &r.fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device_, 1, &r.fence);
            r.in_flight = false;
            // 时间线信号量无需销毁重建：value 单调递增，旧 value 的等待合法
            // 且已达成（no-op），销毁反而会踩到"已提交 submit 仍引用它"。
        }

        return idx;
    }

    // 非时间线回退：host 阻塞等待所有在飞上传完成并释放 region。
    // 等完之后"后续 submit 读上传 buffer"不再构成跨 submit 依赖，消费者
    // 无需等信号量。代价是 host 阻塞（正是 P0-1 想避免的路径），仅在设备
    // 不支持时间线信号量时启用——正确性优先。
    void drain_in_flight()
    {
        for (auto& r : regions_)
        {
            if (!r.in_flight)
                continue;
            vkWaitForFences(device_, 1, &r.fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device_, 1, &r.fence);
            r.in_flight = false;
        }
    }

    // 上传数据到 staging region（元素类型无关，memcpy 字节级操作，§6.3）
    template <typename T>
    [[nodiscard]] Result<void> upload(
        std::size_t region_idx, std::span<const T> data, VkDeviceSize offset = 0)
    {
        if (region_idx >= regions_.size())
            return std::unexpected(Error{"Invalid region index"});

        auto& r = regions_[region_idx];
        const std::size_t byte_size = data.size() * sizeof(T);

        if (offset + byte_size > region_size_)
            return std::unexpected(Error{"Upload exceeds staging region size"});

        std::memcpy(static_cast<char*>(r.mapped_ptr) + offset, data.data(), byte_size);
        return {};
    }

    // 从 staging region 下载数据（元素类型无关，memcpy 字节级操作，§6.3）
    template <typename T>
    [[nodiscard]] Result<void> download(
        std::size_t region_idx, std::span<T> data, VkDeviceSize offset = 0)
    {
        if (region_idx >= regions_.size())
            return std::unexpected(Error{"Invalid region index"});

        auto& r = regions_[region_idx];
        const std::size_t byte_size = data.size() * sizeof(T);

        if (offset + byte_size > region_size_)
            return std::unexpected(Error{"Download exceeds staging region size"});

        std::memcpy(data.data(), static_cast<char*>(r.mapped_ptr) + offset, byte_size);
        return {};
    }

    // 获取缓冲区句柄
    [[nodiscard]] VkBuffer buffer(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].buffer;
    }

    // 获取 fence
    [[nodiscard]] VkFence fence(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].fence;
    }

    // 获取 region 的跨 submit 信号量（消费方 submit 的 pWaitSemaphores 用）
    [[nodiscard]] VkSemaphore semaphore(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].semaphore;
    }

    // 是否使用时间线信号量（false = 走 drain_in_flight 阻塞回退）
    [[nodiscard]] bool timeline() const noexcept { return timeline_; }

    // 本 region 最近一次上传的信号 value（消费方 pWaitSemaphoreValues 用）
    [[nodiscard]] std::uint64_t signal_value(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].signal_value;
    }

    // 为本次上传分配下一个信号 value（严格递增；必须在 mark_in_flight 前调用）
    [[nodiscard]] std::uint64_t next_signal_value(std::size_t region_idx) noexcept
    {
        auto& r = regions_[region_idx];
        r.signal_value = r.signal_value + 1;
        return r.signal_value;
    }

    // 获取 region 的专属 command buffer（上传 copy 录制用；acquire 等完
    // fence 后 vkResetCommandBuffer 复用，禁止在 pending 状态释放/重建）
    [[nodiscard]] VkCommandBuffer command_buffer(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].cmd;
    }

    // region 是否 in flight（已提交、fence 尚未被等待）
    [[nodiscard]] bool in_flight(std::size_t region_idx) const noexcept
    {
        return regions_[region_idx].in_flight;
    }

    // 标记 region 为正在使用
    void mark_in_flight(std::size_t region_idx) noexcept
    {
        regions_[region_idx].in_flight = true;
    }

    // 获取 region 大小
    [[nodiscard]] std::size_t region_size() const noexcept { return region_size_; }

    // 获取 region 数量
    [[nodiscard]] std::size_t num_regions() const noexcept { return regions_.size(); }
};

} // namespace nn

#endif // NN_HAS_VULKAN
