// ── staging_ring.hpp ─────────────────────────────────────────────────────
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

#ifndef NN_STAGING_RING_HPP
#define NN_STAGING_RING_HPP

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../core_errors.hpp"
#include "../core_observer_ptr.hpp"
#include "../core_config.hpp"
#include "memory_pool.hpp"

namespace nn
{

class StagingRing
{
public:
    // 默认 64MB × 2 region（实际分配大小由 initialize() 动态计算，
    // 会根据 host-visible 显存大小自动扩容，此值仅作为下限参考）
    static constexpr std::size_t DEFAULT_REGION_SIZE = 64ull * 1024 * 1024; // 64MB
    static constexpr std::size_t DEFAULT_NUM_REGIONS = 2;
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
        bool in_flight = false;
    };

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    observer_ptr<MemoryPool> pool_;
    std::vector<Region> regions_;
    std::size_t region_size_;
    std::atomic<std::size_t> current_{0};

public:
    StagingRing() = default;

    [[nodiscard]] Result<void> initialize(
        VkDevice device, VkPhysicalDevice physical_device,
        VkCommandPool cmd_pool, MemoryPool& pool,
        std::size_t region_size = DEFAULT_REGION_SIZE,
        std::size_t num_regions = DEFAULT_NUM_REGIONS)
    {
        device_ = device;
        physical_device_ = physical_device;
        cmd_pool_ = cmd_pool;
        pool_.reset(&pool);

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
        }

        return idx;
    }

    // 上传数据到 staging region
    [[nodiscard]] Result<void> upload(
        std::size_t region_idx, std::span<const Scalar> data, VkDeviceSize offset = 0)
    {
        if (region_idx >= regions_.size())
            return std::unexpected(Error{"Invalid region index"});

        auto& r = regions_[region_idx];
        const std::size_t byte_size = data.size() * sizeof(Scalar);

        if (offset + byte_size > region_size_)
            return std::unexpected(Error{"Upload exceeds staging region size"});

        std::memcpy(static_cast<char*>(r.mapped_ptr) + offset, data.data(), byte_size);
        return {};
    }

    // 从 staging region 下载数据
    [[nodiscard]] Result<void> download(
        std::size_t region_idx, std::span<Scalar> data, VkDeviceSize offset = 0)
    {
        if (region_idx >= regions_.size())
            return std::unexpected(Error{"Invalid region index"});

        auto& r = regions_[region_idx];
        const std::size_t byte_size = data.size() * sizeof(Scalar);

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
#endif // NN_STAGING_RING_HPP
