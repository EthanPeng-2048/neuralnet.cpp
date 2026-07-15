#ifndef VK_BACKEND_HPP
#define VK_BACKEND_HPP
// ── Vulkan Compute Backend (v3 - Optimized & Secure) ──────────────────────
#ifdef NN_HAS_VULKAN
#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <cassert>
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
#include <tuple>
#include <vector>
#include "nn_config.hpp"

#if __has_include("matmul_spv.hpp")
#include "matmul_spv.hpp"
#define NN_MATMUL_SPV_EMBEDDED
#endif

#if __has_include("matmul_tiled_spv.hpp")
#include "matmul_tiled_spv.hpp"
#define NN_MATMUL_TILED_SPV_EMBEDDED
#endif

#if __has_include("elementwise_spv.hpp")
#include "elementwise_spv.hpp"
#define NN_ELEMENTWISE_SPV_EMBEDDED
#endif

namespace nn
{
namespace detail
{
[[nodiscard]] inline Result<void> vk_check(VkResult res, const char* file, int line)
{
    if (res != VK_SUCCESS)
    {
        return std::unexpected(Error{
            "Vulkan error " + std::to_string(static_cast<int>(res)) +
            " at " + std::string(file) + ":" + std::to_string(line)});
    }
    return {};
}

inline void convert_double_to_float(
    std::span<const double> src, float* __restrict dst) noexcept
{
    std::transform(src.begin(), src.end(), dst,
                   [](double v) { return static_cast<float>(v); });
}

inline void convert_float_to_double(
    const float* __restrict src, std::span<double> dst) noexcept
{
    std::transform(src, src + dst.size(), dst.begin(),
                   [](float v) { return static_cast<double>(v); });
}
} // namespace detail

// ══════════════════════════════════════════════════════════════════════════
// MemoryPool — 内存子分配器 (O(log n) 优化版)
// ══════════════════════════════════════════════════════════════════════════
class MemoryPool
{
public:
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 128ull * 1024 * 1024;

    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkMemoryPropertyFlags property_flags = 0;
        [[nodiscard]] bool valid() const noexcept { return memory != VK_NULL_HANDLE; }
    };

private:
    struct FreeRegion {
        VkDeviceSize offset;
        VkDeviceSize size;
        bool operator<(const FreeRegion& o) const noexcept { return offset < o.offset; }
    };

    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        uint32_t memory_type_index = 0;
        VkMemoryPropertyFlags property_flags = 0;
        std::set<FreeRegion> free_regions; // 使用 set 保证 O(log n) 合并
        std::size_t allocation_count = 0;
        VkDevice owning_device = VK_NULL_HANDLE;

        Block() = default;
        Block(Block&& o) noexcept
            : memory(o.memory), size(o.size), memory_type_index(o.memory_type_index),
              property_flags(o.property_flags), free_regions(std::move(o.free_regions)),
              allocation_count(o.allocation_count), owning_device(o.owning_device)
        {
            o.memory = VK_NULL_HANDLE; o.size = 0; o.allocation_count = 0; o.owning_device = VK_NULL_HANDLE;
        }
        // 禁用移动赋值，防止 vector 操作引发意外释放
        Block& operator=(Block&&) = delete; 
        
        ~Block() {
            if (memory != VK_NULL_HANDLE && owning_device != VK_NULL_HANDLE)
                vkFreeMemory(owning_device, memory, nullptr);
        }
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
    };

    struct SuballocKey {
        VkDeviceMemory memory;
        VkDeviceSize offset;
        bool operator==(const SuballocKey& o) const noexcept { return memory == o.memory && offset == o.offset; }
    };
    struct SuballocKeyHash {
        std::size_t operator()(const SuballocKey& k) const noexcept {
            return std::hash<VkDeviceMemory>{}(k.memory) ^ (std::hash<VkDeviceSize>{}(k.offset) * 2654435761ULL);
        }
    };

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkDeviceSize block_size_;
    VkPhysicalDeviceMemoryProperties mem_props_;
    
    // 【关键修复】使用 unique_ptr 避免 vector 扩容/删除时触发 Block 的移动和析构
    std::vector<std::unique_ptr<Block>> blocks_; 
    std::unordered_map<SuballocKey, VkDeviceSize, SuballocKeyHash> active_allocs_;
    std::mutex mutex_;

    [[nodiscard]] static std::optional<VkDeviceSize> find_free(
        Block& block, VkDeviceSize alignment, VkDeviceSize size)
    {
        for (auto it = block.free_regions.begin(); it != block.free_regions.end(); ++it)
        {
            VkDeviceSize aligned = (it->offset + alignment - 1) & ~(alignment - 1);
            VkDeviceSize padding = aligned - it->offset;
            if (it->size >= padding + size)
            {
                VkDeviceSize result = aligned;
                VkDeviceSize remaining = it->size - padding - size;
                VkDeviceSize orig_offset = it->offset;
                block.free_regions.erase(it);
                if (padding > 0) block.free_regions.insert({orig_offset, padding});
                if (remaining > 0) block.free_regions.insert({aligned + size, remaining});
                return result;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] Result<Block> create_block(uint32_t memory_type_index,
                                             VkMemoryPropertyFlags flags, VkDeviceSize size = 0)
    {
        VkDeviceSize alloc_size = (size > 0) ? size : block_size_;
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = alloc_size;
        alloc_info.memoryTypeIndex = memory_type_index;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        auto r = detail::vk_check(vkAllocateMemory(device_, &alloc_info, nullptr, &memory), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        Block block;
        block.owning_device = device_;
        block.memory = memory;
        block.size = alloc_size;
        block.memory_type_index = memory_type_index;
        block.property_flags = flags;
        block.free_regions.insert({0, alloc_size});
        return block;
    }

    [[nodiscard]] std::optional<uint32_t> find_memory_type(
        uint32_t type_bits, VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags fallback = 0) const
    {
        std::optional<uint32_t> best, fb;
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
            if (!(type_bits & (1u << i))) continue;
            auto flags = mem_props_.memoryTypes[i].propertyFlags;
            if ((flags & preferred) == preferred && !best) best = i;
            if (fallback && (flags & fallback) == fallback && !fb) fb = i;
        }
        return best ? best : fb;
    }

public:
    MemoryPool(VkDevice device, VkPhysicalDevice physical_device, VkDeviceSize block_size = DEFAULT_BLOCK_SIZE)
        : device_(device), physical_device_(physical_device), block_size_(block_size) 
    {
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props_);
    }
    ~MemoryPool() { blocks_.clear(); }

    [[nodiscard]] Result<Allocation> allocate(
        VkMemoryRequirements requirements, VkMemoryPropertyFlags preferred_flags, VkMemoryPropertyFlags fallback_flags = 0)
    {
        std::lock_guard lock(mutex_);
        auto mem_type = find_memory_type(requirements.memoryTypeBits, preferred_flags, fallback_flags);
        if (!mem_type) return std::unexpected(Error{"No suitable GPU memory type found"});

        VkDeviceSize alignment = requirements.alignment > 0 ? requirements.alignment : 1;
        const VkDeviceSize alloc_size = requirements.size;
        VkDeviceSize effective_block_size = (alloc_size > block_size_) ? alloc_size : block_size_;

        for (auto& block_ptr : blocks_) {
            auto& block = *block_ptr;
            if (block.memory_type_index != *mem_type) continue;
            auto offset = find_free(block, alignment, alloc_size);
            if (offset) {
                block.allocation_count++;
                active_allocs_[{block.memory, *offset}] = alloc_size;
                return Allocation{block.memory, *offset, alloc_size, block.property_flags};
            }
        }

        auto block_result = create_block(*mem_type, preferred_flags, effective_block_size);
        if (!block_result) return std::unexpected(block_result.error());
        
        auto new_block = std::make_unique<Block>(std::move(*block_result));
        auto offset = find_free(*new_block, alignment, alloc_size);
        if (!offset) return std::unexpected(Error{"Suballocation failed"});

        VkDeviceMemory mem = new_block->memory;
        VkMemoryPropertyFlags flags = new_block->property_flags;
        new_block->allocation_count++;
        active_allocs_[{mem, *offset}] = alloc_size;
        blocks_.push_back(std::move(new_block));
        return Allocation{mem, *offset, alloc_size, flags};
    }

    void free(const Allocation& alloc)
    {
        if (!alloc.valid()) return;
        std::lock_guard lock(mutex_);
        SuballocKey key{alloc.memory, alloc.offset};
        auto active_it = active_allocs_.find(key);
        if (active_it == active_allocs_.end()) return;

        for (auto& block_ptr : blocks_) {
            auto& block = *block_ptr;
            if (block.memory != alloc.memory) continue;
            
            FreeRegion new_reg{alloc.offset, alloc.size};
            auto ins_it = block.free_regions.insert(new_reg).first;
            
            // 向前合并
            if (ins_it != block.free_regions.begin()) {
                auto prev = std::prev(ins_it);
                if (prev->offset + prev->size == ins_it->offset) {
                    FreeRegion merged{prev->offset, prev->size + ins_it->size};
                    block.free_regions.erase(prev);
                    block.free_regions.erase(ins_it);
                    ins_it = block.free_regions.insert(merged).first;
                }
            }
            // 向后合并
            auto next = std::next(ins_it);
            if (next != block.free_regions.end()) {
                if (ins_it->offset + ins_it->size == next->offset) {
                    FreeRegion merged{ins_it->offset, ins_it->size + next->size};
                    block.free_regions.erase(next);
                    block.free_regions.erase(ins_it);
                    block.free_regions.insert(merged);
                }
            }

            block.allocation_count--;
            active_allocs_.erase(active_it);
            
            // 【关键修复】：不要 erase block！
            // 让 Block 留在池中复用，避免 vector 移动元素导致的 Double Free。
            // 内存池的设计初衷就是保留大块内存，直到 MemoryPool 析构时统一释放。
            return;
        }
    }
};

// ══════════════════════════════════════════════════════════════════════════
// StagingRing — 全局 Staging 环形缓冲区 (线程安全 & 动态大小)
// ══════════════════════════════════════════════════════════════════════════
class StagingRing
{
public:
    static constexpr std::size_t DEFAULT_REGION_SIZE = 256ull * 1024 * 1024;
    static constexpr std::size_t DEFAULT_NUM_REGIONS = 4;

    struct Region {
        MemoryPool::Allocation alloc;
        VkBuffer buffer = VK_NULL_HANDLE;
        void* mapped_ptr = nullptr;
        VkFence fence = VK_NULL_HANDLE;
        bool in_flight = false;
    };

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    MemoryPool* pool_ = nullptr;
    std::vector<Region> regions_;
    std::size_t region_size_;
    std::atomic<std::size_t> current_{0}; // 原子变量修复数据竞争

public:
    StagingRing() = default;

    [[nodiscard]] Result<void> initialize(
        VkDevice device, VkPhysicalDevice physical_device,
        VkCommandPool cmd_pool, MemoryPool* pool,
        std::size_t region_size = DEFAULT_REGION_SIZE,
        std::size_t num_regions = DEFAULT_NUM_REGIONS)
    {
        device_ = device;
        physical_device_ = physical_device;
        cmd_pool_ = cmd_pool;
        pool_ = pool;

        // 动态计算合理的 Staging 大小，避免盲目占用 1GB HOST_VISIBLE 内存
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
        VkDeviceSize max_host_visible = 0;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                uint32_t heap_idx = mem_props.memoryTypes[i].heapIndex;
                if (mem_props.memoryHeaps[heap_idx].size > max_host_visible) {
                    max_host_visible = mem_props.memoryHeaps[heap_idx].size;
                }
            }
        }
        VkDeviceSize calculated_size = max_host_visible / 8;
        if (calculated_size < 64ull * 1024 * 1024) calculated_size = 64ull * 1024 * 1024;
        if (calculated_size > 256ull * 1024 * 1024) calculated_size = 256ull * 1024 * 1024;
        region_size_ = std::min(static_cast<VkDeviceSize>(region_size), calculated_size);

        regions_.resize(num_regions);
        for (std::size_t i = 0; i < num_regions; ++i)
        {
            auto& r = regions_[i];
            VkBufferCreateInfo buf_info{};
            buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buf_info.size = region_size_;
            buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto res = detail::vk_check(
                vkCreateBuffer(device_, &buf_info, nullptr, &r.buffer), __FILE__, __LINE__);
            if (!res) return res;

            VkMemoryRequirements mem_reqs;
            vkGetBufferMemoryRequirements(device_, r.buffer, &mem_reqs);
            auto alloc_r = pool_->allocate(
                mem_reqs,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!alloc_r) return std::unexpected(alloc_r.error());
            r.alloc = *alloc_r;

            res = detail::vk_check(
                vkBindBufferMemory(device_, r.buffer, r.alloc.memory, r.alloc.offset),
                __FILE__, __LINE__);
            if (!res) return res;

            if (vkMapMemory(device_, r.alloc.memory, r.alloc.offset,
                            r.alloc.size, 0, &r.mapped_ptr) != VK_SUCCESS)
                return std::unexpected(Error{"Failed to map staging region " + std::to_string(i)});

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            res = detail::vk_check(
                vkCreateFence(device_, &fence_info, nullptr, &r.fence), __FILE__, __LINE__);
            if (!res) return res;
        }
        return {};
    }

    ~StagingRing()
    {
        if (device_ == VK_NULL_HANDLE) return;
        for (auto& r : regions_)
        {
            if (r.fence != VK_NULL_HANDLE) vkDestroyFence(device_, r.fence, nullptr);
            if (r.mapped_ptr) vkUnmapMemory(device_, r.alloc.memory);
            if (r.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, r.buffer, nullptr);
            if (pool_ && r.alloc.valid()) pool_->free(r.alloc);
        }
    }

    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;

    [[nodiscard]] std::size_t acquire()
    {
        std::size_t idx = current_.fetch_add(1, std::memory_order_relaxed) % regions_.size();
        auto& r = regions_[idx];
        if (r.in_flight)
        {
            vkWaitForFences(device_, 1, &r.fence, VK_TRUE, 10'000'000'000ULL);
            vkResetFences(device_, 1, &r.fence);
            r.in_flight = false;
        }
        return idx;
    }

    [[nodiscard]] Result<void> upload(std::size_t idx, std::span<const double> data,
                                      VkDeviceSize byte_offset = 0)
    {
        if (byte_offset + data.size() * sizeof(float) > region_size_)
            return std::unexpected(Error{"Staging upload out of bounds"});
            
        auto& r = regions_[idx];
        if (!r.mapped_ptr) return std::unexpected(Error{"Staging region not mapped"});
        auto* dst = reinterpret_cast<float*>(
            static_cast<std::byte*>(r.mapped_ptr) + byte_offset);
        detail::convert_double_to_float(data, dst);
        return {};
    }

    [[nodiscard]] Result<void> download(std::size_t idx, std::span<double> dst,
                                        VkDeviceSize byte_offset = 0)
    {
        if (byte_offset + dst.size() * sizeof(float) > region_size_)
            return std::unexpected(Error{"Staging download out of bounds"});
            
        auto& r = regions_[idx];
        if (!r.mapped_ptr) return std::unexpected(Error{"Staging region not mapped"});
        const auto* src = reinterpret_cast<const float*>(
            static_cast<const std::byte*>(r.mapped_ptr) + byte_offset);
        detail::convert_float_to_double(src, dst);
        return {};
    }

    [[nodiscard]] VkFence fence(std::size_t idx) const noexcept { return regions_[idx].fence; }
    [[nodiscard]] VkBuffer buffer(std::size_t idx) const noexcept { return regions_[idx].buffer; }
    void mark_in_flight(std::size_t idx) noexcept { regions_[idx].in_flight = true; }
    [[nodiscard]] std::size_t region_size() const noexcept { return region_size_; }
    [[nodiscard]] std::size_t num_regions() const noexcept { return regions_.size(); }
};

// ══════════════════════════════════════════════════════════════════════════
// MappedMemoryGuard — RAII 内存映射守卫
// ══════════════════════════════════════════════════════════════════════════
class MappedMemoryGuard
{
private:
    VkDevice device_;
    VkDeviceMemory memory_;
    VkDeviceSize offset_;
    void* ptr_;
    std::size_t byte_size_;
    VkMemoryPropertyFlags property_flags_;
    mutable bool dirty_;

public:
    MappedMemoryGuard(VkDevice device, VkDeviceMemory mem, VkDeviceSize offset,
                      std::size_t size, VkMemoryPropertyFlags flags)
        : device_(device), memory_(mem), offset_(offset),
          ptr_(nullptr), byte_size_(size), property_flags_(flags), dirty_(false)
    {
        if (vkMapMemory(device_, memory_, offset_, byte_size_, 0, &ptr_) != VK_SUCCESS)
            ptr_ = nullptr;
    }
    ~MappedMemoryGuard()
    {
        if (ptr_ && device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE)
        {
            if (dirty_ && !(property_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                VkMappedMemoryRange range{};
                range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                range.memory = memory_; range.offset = offset_; range.size = byte_size_;
                vkFlushMappedMemoryRanges(device_, 1, &range);
            }
            vkUnmapMemory(device_, memory_);
        }
    }
    MappedMemoryGuard(const MappedMemoryGuard&) = delete;
    MappedMemoryGuard& operator=(const MappedMemoryGuard&) = delete;
    MappedMemoryGuard(MappedMemoryGuard&& o) noexcept
        : device_(o.device_), memory_(o.memory_), offset_(o.offset_),
          ptr_(o.ptr_), byte_size_(o.byte_size_),
          property_flags_(o.property_flags_), dirty_(o.dirty_)
    {
        o.ptr_ = nullptr; o.device_ = VK_NULL_HANDLE; o.memory_ = VK_NULL_HANDLE;
    }
    MappedMemoryGuard& operator=(MappedMemoryGuard&& o) noexcept
    {
        if (this != &o)
        {
            if (ptr_ && device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE)
            {
                if (dirty_ && !(property_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                {
                    VkMappedMemoryRange range{};
                    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
                    range.memory = memory_; range.offset = offset_; range.size = byte_size_;
                    vkFlushMappedMemoryRanges(device_, 1, &range);
                }
                vkUnmapMemory(device_, memory_);
            }
            device_ = o.device_; memory_ = o.memory_; offset_ = o.offset_;
            ptr_ = o.ptr_; byte_size_ = o.byte_size_;
            property_flags_ = o.property_flags_; dirty_ = o.dirty_;
            o.ptr_ = nullptr; o.device_ = VK_NULL_HANDLE; o.memory_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
    void upload(std::span<const double> data) noexcept
    {
        if (!ptr_) return;
        detail::convert_double_to_float(data, static_cast<float*>(ptr_));
        dirty_ = true;
    }
    void download(std::span<double> dst) const noexcept
    {
        if (!ptr_) return;
        if (!(property_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = memory_; range.offset = offset_; range.size = byte_size_;
            vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }
        detail::convert_float_to_double(static_cast<const float*>(ptr_), dst);
    }
    [[nodiscard]] std::span<std::byte> bytes() noexcept
    { return {static_cast<std::byte*>(ptr_), byte_size_}; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    { return {static_cast<const std::byte*>(ptr_), byte_size_}; }
};

// ══════════════════════════════════════════════════════════════════════════
// VkBufferWrapper — RAII 缓冲区封装
// ══════════════════════════════════════════════════════════════════════════
class VkBufferWrapper
{
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    MemoryPool::Allocation alloc_;
    MemoryPool* pool_ = nullptr;

    VkBufferWrapper(VkDevice dev, VkBuffer buf, MemoryPool::Allocation alloc, MemoryPool* pool)
        : device_(dev), buffer_(buf), alloc_(alloc), pool_(pool) {}

public:
    VkBufferWrapper() = default;
    ~VkBufferWrapper()
    {
        if (device_ != VK_NULL_HANDLE)
        {
            if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
            if (pool_ && alloc_.valid()) pool_->free(alloc_);
        }
    }
    VkBufferWrapper(const VkBufferWrapper&) = delete;
    VkBufferWrapper& operator=(const VkBufferWrapper&) = delete;
    VkBufferWrapper(VkBufferWrapper&& o) noexcept
        : device_(o.device_), buffer_(o.buffer_), alloc_(o.alloc_), pool_(o.pool_)
    {
        o.buffer_ = VK_NULL_HANDLE; o.device_ = VK_NULL_HANDLE;
        o.alloc_.memory = VK_NULL_HANDLE; o.pool_ = nullptr;
    }
    VkBufferWrapper& operator=(VkBufferWrapper&& o) noexcept
    {
        if (this != &o)
        {
            if (device_ != VK_NULL_HANDLE)
            {
                if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
                if (pool_ && alloc_.valid()) pool_->free(alloc_);
            }
            device_ = o.device_; buffer_ = o.buffer_; alloc_ = o.alloc_; pool_ = o.pool_;
            o.buffer_ = VK_NULL_HANDLE; o.device_ = VK_NULL_HANDLE;
            o.alloc_.memory = VK_NULL_HANDLE; o.pool_ = nullptr;
        }
        return *this;
    }
    [[nodiscard]] static Result<VkBufferWrapper> create(
        VkDevice device, MemoryPool& pool,
        std::size_t element_count, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags preferred_flags,
        VkMemoryPropertyFlags fallback_flags = 0)
    {
        const std::size_t buf_size = element_count * sizeof(float);
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = buf_size;
        buf_info.usage = usage;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        auto r = detail::vk_check(
            vkCreateBuffer(device, &buf_info, nullptr, &buffer), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
        auto alloc_result = pool.allocate(mem_reqs, preferred_flags, fallback_flags);
        if (!alloc_result)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(alloc_result.error());
        }
        r = detail::vk_check(
            vkBindBufferMemory(device, buffer, alloc_result->memory, alloc_result->offset),
            __FILE__, __LINE__);
        if (!r)
        {
            pool.free(*alloc_result);
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(r.error());
        }
        return VkBufferWrapper(device, buffer, *alloc_result, &pool);
    }
    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceMemory memory() const noexcept { return alloc_.memory; }
    [[nodiscard]] VkDeviceSize memory_offset() const noexcept { return alloc_.offset; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return alloc_.size; }
    [[nodiscard]] MappedMemoryGuard map() noexcept
    {
        return MappedMemoryGuard(device_, alloc_.memory, alloc_.offset,
                                 alloc_.size, alloc_.property_flags);
    }
};

class GpuBuffer
{
private:
    std::size_t element_count_ = 0;
    std::unique_ptr<VkBufferWrapper> impl_;
    explicit GpuBuffer(std::size_t count, std::unique_ptr<VkBufferWrapper> wrapper)
        : element_count_(count), impl_(std::move(wrapper)) {}
public:
    GpuBuffer() = default;
    [[nodiscard]] static Result<GpuBuffer> create(
        VkDevice device, MemoryPool& pool, std::size_t element_count, VkBufferUsageFlags usage)
    {
        auto w = VkBufferWrapper::create(device, pool, element_count, usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!w) return std::unexpected(w.error());
        return GpuBuffer(element_count, std::make_unique<VkBufferWrapper>(std::move(*w)));
    }
    [[nodiscard]] static Result<GpuBuffer> create_device_local(
        VkDevice device, MemoryPool& pool, std::size_t element_count, VkBufferUsageFlags usage)
    {
        auto w = VkBufferWrapper::create(device, pool, element_count, usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (!w) return std::unexpected(w.error());
        return GpuBuffer(element_count, std::make_unique<VkBufferWrapper>(std::move(*w)));
    }
    [[nodiscard]] std::size_t element_count() const noexcept { return element_count_; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return impl_ ? impl_->byte_size() : 0; }
    [[nodiscard]] bool empty() const noexcept { return !impl_ || element_count_ == 0; }
    [[nodiscard]] VkBufferWrapper& impl() { assert(impl_); return *impl_; }
    [[nodiscard]] const VkBufferWrapper& impl() const { assert(impl_); return *impl_; }
};

// ══════════════════════════════════════════════════════════════════════════
// GpuTensor — 显存常驻的张量载体（双轨制架构核心）
// ══════════════════════════════════════════════════════════════════════════
// CPU 端继续使用 Matrix (double) 做逻辑控制，GPU 端使用 GpuTensor (float)
// 做计算。数据只在进入网络时 Upload 一次，输出时 Download 一次，
// 中间所有矩阵乘法和激活函数全程在 GPU 显存中流转。
class GpuBackend; // 前置声明
class Matrix;     // 前置声明（完整定义在 matrix.hpp，实现文件在 gpu_tensor_impl.hpp）

class GpuTensor
{
private:
    std::shared_ptr<GpuBuffer> buffer_; // shared_ptr 支持权重共享和计算图分支
    std::size_t rows_{0};
    std::size_t cols_{0};

    GpuTensor(std::shared_ptr<GpuBuffer> buf, std::size_t r, std::size_t c)
        : buffer_(std::move(buf)), rows_(r), cols_(c) {}

public:
    GpuTensor() = default;

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && !buffer_->empty(); }
    [[nodiscard]] GpuBuffer& buffer() { return *buffer_; }
    [[nodiscard]] const GpuBuffer& buffer() const { return *buffer_; }

    // ── 工厂函数 1：从 CPU Matrix 上传（仅在输入层/权重加载时调用）──────
    [[nodiscard]] static Result<GpuTensor> from_matrix(
        const Matrix& cpu_mat, GpuBackend& backend);

    // ── 工厂函数 2：分配未初始化的显存（用于接收计算结果）──────────────
    [[nodiscard]] static Result<GpuTensor> create_empty(
        std::size_t rows, std::size_t cols, GpuBackend& backend);

    // ── 导出到 CPU Matrix（仅在输出层/Loss 计算时调用）────────────────
    [[nodiscard]] Result<Matrix> to_matrix(GpuBackend& backend) const;
};

class VulkanDevice
{
private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = 0;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    bool valid_ = false;

    [[nodiscard]] static std::optional<uint32_t> find_compute_queue_family(VkPhysicalDevice dev)
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
        for (uint32_t i = 0; i < count; ++i)
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
        return std::nullopt;
    }

public:
    VulkanDevice() = default;
    ~VulkanDevice()
    {
        if (device_ != VK_NULL_HANDLE) vkDestroyDevice(device_, nullptr);
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
    }
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    [[nodiscard]] Result<void> initialize()
    {
        if (valid_) return {};
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "neuralnet.cpp";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "neuralnet.cpp";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo inst_info{};
        inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_info.pApplicationInfo = &app_info;
#ifndef NDEBUG
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> avail_layers(layer_count);
        if (layer_count > 0)
            vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());
        bool has_validation = false;
        for (uint32_t li = 0; li < layer_count; ++li)
            if (std::strcmp(avail_layers[li].layerName, "VK_LAYER_KHRONOS_validation") == 0)
            { has_validation = true; break; }
        static const char* validation_layer = "VK_LAYER_KHRONOS_validation";
        if (has_validation) { inst_info.enabledLayerCount = 1; inst_info.ppEnabledLayerNames = &validation_layer; }
#endif
        auto r = detail::vk_check(vkCreateInstance(&inst_info, nullptr, &instance_), __FILE__, __LINE__);
        if (!r) return r;

        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0) return std::unexpected(Error{"No Vulkan-capable GPU found"});
        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

        std::optional<VkPhysicalDevice> best_discrete, best_integrated;
        uint32_t best_dq = 0, best_iq = 0;
        for (auto& dev : devices)
        {
            auto qf = find_compute_queue_family(dev);
            if (!qf) continue;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && !best_discrete)
            { best_discrete = dev; best_dq = *qf; }
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && !best_integrated)
            { best_integrated = dev; best_iq = *qf; }
        }
        if (best_discrete) { physical_device_ = *best_discrete; queue_family_index_ = best_dq; }
        else if (best_integrated) { physical_device_ = *best_integrated; queue_family_index_ = best_iq; }
        else return std::unexpected(Error{"No GPU with compute queue found"});

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        r = detail::vk_check(vkCreateDevice(physical_device_, &device_info, nullptr, &device_), __FILE__, __LINE__);
        if (!r) return r;

        vkGetDeviceQueue(device_, queue_family_index_, 0, &compute_queue_);
        valid_ = true;
        return {};
    }
    [[nodiscard]] bool is_valid() const noexcept { return valid_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] uint32_t queue_family_index() const noexcept { return queue_family_index_; }
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
};

class VulkanPipeline
{
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

public:
    VulkanPipeline() = default;
    ~VulkanPipeline() { destroy(); }
    void destroy()
    {
        if (device_ != VK_NULL_HANDLE)
        {
            if (pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline_, nullptr);
            if (pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (descriptor_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
            if (shader_module_ != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader_module_, nullptr);
        }
        device_ = VK_NULL_HANDLE; shader_module_ = VK_NULL_HANDLE;
        descriptor_layout_ = VK_NULL_HANDLE; pipeline_layout_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE;
    }
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;
    VulkanPipeline(VulkanPipeline&& o) noexcept
        : device_(o.device_), shader_module_(o.shader_module_),
          descriptor_layout_(o.descriptor_layout_),
          pipeline_layout_(o.pipeline_layout_), pipeline_(o.pipeline_)
    {
        o.device_ = VK_NULL_HANDLE; o.shader_module_ = VK_NULL_HANDLE;
        o.descriptor_layout_ = VK_NULL_HANDLE; o.pipeline_layout_ = VK_NULL_HANDLE;
        o.pipeline_ = VK_NULL_HANDLE;
    }
    VulkanPipeline& operator=(VulkanPipeline&& o) noexcept
    {
        if (this != &o) { 
            destroy(); 
            device_ = o.device_; shader_module_ = o.shader_module_;
            descriptor_layout_ = o.descriptor_layout_; pipeline_layout_ = o.pipeline_layout_;
            pipeline_ = o.pipeline_;
            o.device_ = VK_NULL_HANDLE; o.shader_module_ = VK_NULL_HANDLE;
            o.descriptor_layout_ = VK_NULL_HANDLE; o.pipeline_layout_ = VK_NULL_HANDLE;
            o.pipeline_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    [[nodiscard]] static Result<VulkanPipeline> create_matmul(
        VkDevice device, std::span<const uint32_t> spirv_code)
    {
        VulkanPipeline pl;
        pl.device_ = device;
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv_code.size_bytes();
        module_info.pCode = spirv_code.data();
        auto r = detail::vk_check(
            vkCreateShaderModule(device, &module_info, nullptr, &pl.shader_module_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; ++i)
        {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        r = detail::vk_check(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 3 * sizeof(uint32_t);
        VkPipelineLayoutCreateInfo pl_layout_info{};
        pl_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_layout_info.setLayoutCount = 1;
        pl_layout_info.pSetLayouts = &pl.descriptor_layout_;
        pl_layout_info.pushConstantRangeCount = 1;
        pl_layout_info.pPushConstantRanges = &push_range;
        r = detail::vk_check(
            vkCreatePipelineLayout(device, &pl_layout_info, nullptr, &pl.pipeline_layout_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = pl.shader_module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pl.pipeline_layout_;
        r = detail::vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pl.pipeline_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());
        return pl;
    }
    // ── 逐元素运算 Pipeline（ReLU / GeLU / BiasAdd，3 绑定 + 4 Push Constant）──
    [[nodiscard]] static Result<VulkanPipeline> create_elementwise(
        VkDevice device, std::span<const uint32_t> spirv_code)
    {
        VulkanPipeline pl;
        pl.device_ = device;
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv_code.size_bytes();
        module_info.pCode = spirv_code.data();
        auto r = detail::vk_check(
            vkCreateShaderModule(device, &module_info, nullptr, &pl.shader_module_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // binding 0 = primary input, binding 1 = secondary input (bias), binding 2 = output
        VkDescriptorSetLayoutBinding bindings[3]{};
        for (int i = 0; i < 3; ++i)
        {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 3;
        layout_info.pBindings = bindings;
        r = detail::vk_check(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // push constants: uint count, uint op, uint rows, uint cols
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4 * sizeof(uint32_t);
        VkPipelineLayoutCreateInfo pl_layout_info{};
        pl_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_layout_info.setLayoutCount = 1;
        pl_layout_info.pSetLayouts = &pl.descriptor_layout_;
        pl_layout_info.pushConstantRangeCount = 1;
        pl_layout_info.pPushConstantRanges = &push_range;
        r = detail::vk_check(
            vkCreatePipelineLayout(device, &pl_layout_info, nullptr, &pl.pipeline_layout_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = pl.shader_module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pl.pipeline_layout_;
        r = detail::vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pl.pipeline_), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());
        return pl;
    }
    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] VkDescriptorSetLayout descriptor_layout() const noexcept { return descriptor_layout_; }
};

struct MatmulKey {
    uint32_t M, N, K;
    bool operator==(const MatmulKey& o) const noexcept { return M == o.M && N == o.N && K == o.K; }
};
struct MatmulKeyHash {
    std::size_t operator()(const MatmulKey& k) const noexcept {
        std::size_t h = std::hash<uint32_t>{}(k.M);
        h ^= std::hash<uint32_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedDispatch {
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    GpuBuffer buf_a, buf_b, buf_c;
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> pre_recorded_cmds; // 预录制命令缓冲
    std::atomic<int> ref_count{0};

    ~CachedDispatch() { reset(); }
    CachedDispatch() = default;
    CachedDispatch(CachedDispatch&& o) noexcept
        : descriptor_set(o.descriptor_set), buf_a(std::move(o.buf_a)),
          buf_b(std::move(o.buf_b)), buf_c(std::move(o.buf_c)),
          device(o.device), desc_pool(o.desc_pool), cmd_pool(o.cmd_pool),
          pre_recorded_cmds(std::move(o.pre_recorded_cmds)),
          ref_count(o.ref_count.load(std::memory_order_relaxed))
    {
        o.descriptor_set = VK_NULL_HANDLE; o.device = VK_NULL_HANDLE; o.cmd_pool = VK_NULL_HANDLE;
        o.ref_count.store(0, std::memory_order_relaxed);
    }
    CachedDispatch& operator=(CachedDispatch&& o) noexcept
    {
        if (this != &o)
        {
            reset();
            descriptor_set = o.descriptor_set; o.descriptor_set = VK_NULL_HANDLE;
            buf_a = std::move(o.buf_a); buf_b = std::move(o.buf_b); buf_c = std::move(o.buf_c);
            device = o.device; o.device = VK_NULL_HANDLE;
            desc_pool = o.desc_pool; cmd_pool = o.cmd_pool; o.cmd_pool = VK_NULL_HANDLE;
            pre_recorded_cmds = std::move(o.pre_recorded_cmds);
            ref_count.store(o.ref_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            o.ref_count.store(0, std::memory_order_relaxed);
        }
        return *this;
    }
    CachedDispatch(const CachedDispatch&) = delete;
    CachedDispatch& operator=(const CachedDispatch&) = delete;

    void reset() noexcept
    {
        if (device != VK_NULL_HANDLE) {
            if (!pre_recorded_cmds.empty() && cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(device, cmd_pool, 
                    static_cast<uint32_t>(pre_recorded_cmds.size()), 
                    pre_recorded_cmds.data());
            }
            if (descriptor_set != VK_NULL_HANDLE && desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(device, desc_pool, 1, &descriptor_set);
            }
        }
        pre_recorded_cmds.clear();
        descriptor_set = VK_NULL_HANDLE;
        buf_a = GpuBuffer{}; buf_b = GpuBuffer{}; buf_c = GpuBuffer{};
        device = VK_NULL_HANDLE; desc_pool = VK_NULL_HANDLE; cmd_pool = VK_NULL_HANDLE;
        ref_count.store(0, std::memory_order_relaxed);
    }
};

class GpuBackend
{
private:
    VulkanDevice device_;
    VulkanPipeline matmul_pipeline_;
    std::mutex init_mutex_;
    std::mutex cache_mutex_;
    std::mutex queue_mutex_;
    bool initialized_ = false;
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<StagingRing> staging_ring_;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool dispatch_pool_ = VK_NULL_HANDLE;
    static constexpr std::size_t MAX_CACHED_DISPATCHES = 256;
    static constexpr std::size_t DESCRIPTORS_PER_DISPATCH = 3;
    static constexpr uint32_t WORKGROUP_SIZE = 16;

    std::unordered_map<MatmulKey, CachedDispatch, MatmulKeyHash> dispatch_cache_;
    std::list<MatmulKey> dispatch_lru_;
    std::unordered_map<MatmulKey, std::list<MatmulKey>::iterator, MatmulKeyHash> dispatch_lru_map_;

    // ── 双轨制架构：新增 Pipeline 和资源追踪 ─────────────────────────
    VulkanPipeline matmul_tiled_pipeline_;   // 4×4 粗化分块矩阵乘法
    VulkanPipeline elementwise_pipeline_;    // 逐元素运算（ReLU / GeLU）
    VkDescriptorPool gpu_tensor_pool_ = VK_NULL_HANDLE;  // GpuTensor 专用描述符池

    // 非阻塞 GPU 操作的待清理资源
    struct PendingGpuOps {
        std::vector<VkCommandBuffer> cmd_buffers;
        std::vector<VkDescriptorSet> desc_sets;
        VkFence fence = VK_NULL_HANDLE;
        bool has_fence = false;
    };
    PendingGpuOps pending_ops_;
    static constexpr std::size_t TILED_WORKGROUP_SIZE = 16; // 16×16 = 256 线程/WorkGroup
    static constexpr std::size_t TILE_MN = 64;              // 每 WorkGroup 计算 64×64 输出

    [[nodiscard]] Result<CachedDispatch> create_cached_dispatch(const MatmulKey& key)
    {
        CachedDispatch d;
        d.device = device_.device();
        d.desc_pool = dispatch_pool_;
        d.cmd_pool = command_pool_;
        const auto a_elems = static_cast<std::size_t>(key.M) * key.K;
        const auto b_elems = static_cast<std::size_t>(key.K) * key.N;
        const auto c_elems = static_cast<std::size_t>(key.M) * key.N;

        auto a = GpuBuffer::create_device_local(d.device, *memory_pool_, a_elems,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!a) return std::unexpected(a.error());
        d.buf_a = std::move(*a);

        auto b = GpuBuffer::create_device_local(d.device, *memory_pool_, b_elems,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!b) return std::unexpected(b.error());
        d.buf_b = std::move(*b);

        auto c = GpuBuffer::create_device_local(d.device, *memory_pool_, c_elems,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!c) return std::unexpected(c.error());
        d.buf_c = std::move(*c);

        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = d.desc_pool;
        desc_alloc.descriptorSetCount = 1;
        auto dl = matmul_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;
        auto r = detail::vk_check(
            vkAllocateDescriptorSets(d.device, &desc_alloc, &d.descriptor_set), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkDescriptorBufferInfo buf_infos[3]{
            {d.buf_a.impl().handle(), 0, VK_WHOLE_SIZE},
            {d.buf_b.impl().handle(), 0, VK_WHOLE_SIZE},
            {d.buf_c.impl().handle(), 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = d.descriptor_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(d.device, 3, writes, 0, nullptr);

        // 预录制命令缓冲 (性能优化)
        d.pre_recorded_cmds.resize(staging_ring_->num_regions());
        for (std::size_t i = 0; i < staging_ring_->num_regions(); ++i) {
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;
            
            r = detail::vk_check(
                vkAllocateCommandBuffers(d.device, &cmd_alloc, &d.pre_recorded_cmds[i]), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
            
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
            r = detail::vk_check(vkBeginCommandBuffer(d.pre_recorded_cmds[i], &begin_info), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
            
            auto staging_buf = staging_ring_->buffer(i);
            VkBufferCopy cp_a{0, 0, a_elems * sizeof(float)};
            VkBufferCopy cp_b{a_elems * sizeof(float), 0, b_elems * sizeof(float)};
            vkCmdCopyBuffer(d.pre_recorded_cmds[i], staging_buf, d.buf_a.impl().handle(), 1, &cp_a);
            vkCmdCopyBuffer(d.pre_recorded_cmds[i], staging_buf, d.buf_b.impl().handle(), 1, &cp_b);

            VkBufferMemoryBarrier barriers[2]{};
            for (int j = 0; j < 2; ++j) {
                barriers[j].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barriers[j].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barriers[j].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barriers[j].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[j].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[j].offset = 0; barriers[j].size = VK_WHOLE_SIZE;
            }
            barriers[0].buffer = d.buf_a.impl().handle();
            barriers[1].buffer = d.buf_b.impl().handle();
            vkCmdPipelineBarrier(d.pre_recorded_cmds[i],
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 2, barriers, 0, nullptr);

            vkCmdBindPipeline(d.pre_recorded_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE, matmul_pipeline_.handle());
            vkCmdBindDescriptorSets(d.pre_recorded_cmds[i], VK_PIPELINE_BIND_POINT_COMPUTE,
                matmul_pipeline_.pipeline_layout(), 0, 1, &d.descriptor_set, 0, nullptr);
            const uint32_t push_data[3] = {key.M, key.N, key.K};
            vkCmdPushConstants(d.pre_recorded_cmds[i], matmul_pipeline_.pipeline_layout(),
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);
            vkCmdDispatch(d.pre_recorded_cmds[i], (key.N + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE, 
                                        (key.M + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE, 1);

            VkBufferMemoryBarrier c_barrier{};
            c_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            c_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            c_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            c_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            c_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            c_barrier.buffer = d.buf_c.impl().handle();
            c_barrier.offset = 0; c_barrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(d.pre_recorded_cmds[i],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 1, &c_barrier, 0, nullptr);

            VkBufferCopy cp_c{0, (a_elems + b_elems) * sizeof(float), c_elems * sizeof(float)};
            vkCmdCopyBuffer(d.pre_recorded_cmds[i], d.buf_c.impl().handle(), staging_buf, 1, &cp_c);
            
            r = detail::vk_check(vkEndCommandBuffer(d.pre_recorded_cmds[i]), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
        }
        return d;
    }

    void evict_oldest_dispatch()
    {
        for (auto rit = dispatch_lru_.rbegin(); rit != dispatch_lru_.rend(); ++rit)
        {
            auto cache_it = dispatch_cache_.find(*rit);
            if (cache_it == dispatch_cache_.end()) continue;
            if (cache_it->second.ref_count.load(std::memory_order_acquire) > 0) continue;
            auto key = *rit;
            dispatch_lru_map_.erase(key);
            dispatch_cache_.erase(key);
            dispatch_lru_.erase(std::next(rit).base());
            return;
        }
    }

    GpuBackend() = default;
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
    [[nodiscard]] static const std::vector<uint32_t>& get_elementwise_spirv()
    {
#ifdef NN_ELEMENTWISE_SPV_EMBEDDED
        return nn_elementwise_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    // ── 清理待处理的非阻塞 GPU 操作 ─────────────────────────────────
    void flush_pending_ops()
    {
        auto& p = pending_ops_;
        if (!p.cmd_buffers.empty() || p.has_fence)
        {
            if (p.has_fence)
            {
                vkWaitForFences(device_.device(), 1, &p.fence, VK_TRUE, 10'000'000'000ULL);
                vkDestroyFence(device_.device(), p.fence, nullptr);
                p.fence = VK_NULL_HANDLE;
                p.has_fence = false;
            }
            if (!p.cmd_buffers.empty())
                vkFreeCommandBuffers(device_.device(), command_pool_,
                    static_cast<uint32_t>(p.cmd_buffers.size()), p.cmd_buffers.data());
            p.cmd_buffers.clear();
            if (!p.desc_sets.empty() && gpu_tensor_pool_ != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_,
                    static_cast<uint32_t>(p.desc_sets.size()), p.desc_sets.data());
            p.desc_sets.clear();
        }
    }

public:
    [[nodiscard]] static GpuBackend& instance()
    {
        static GpuBackend backend;
        return backend;
    }
    ~GpuBackend()
    {
        if (device_.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_.device());
            flush_pending_ops();
            if (staging_ring_) {
                for (std::size_t i = 0; i < staging_ring_->num_regions(); ++i) {
                    auto f = staging_ring_->fence(i);
                    if (f != VK_NULL_HANDLE)
                        vkWaitForFences(device_.device(), 1, &f, VK_TRUE, 10'000'000'000ULL);
                }
            }
        }
        dispatch_lru_map_.clear();
        dispatch_lru_.clear();
        dispatch_cache_.clear();
        staging_ring_.reset();
        if (device_.device() != VK_NULL_HANDLE) {
            if (gpu_tensor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_.device(), gpu_tensor_pool_, nullptr);
            if (dispatch_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_.device(), dispatch_pool_, nullptr);
            if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_.device(), command_pool_, nullptr);
        }
        memory_pool_.reset();
    }
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;

    [[nodiscard]] Result<void> initialize()
    {
        std::lock_guard lock(init_mutex_);
        if (initialized_) return {};
        const auto& spirv = get_matmul_spirv();
        if (spirv.empty())
            return std::unexpected(Error{"matmul SPIR-V bytecode not embedded."});
        auto dev_r = device_.initialize();
        if (!dev_r) return dev_r;
        auto pl_r = VulkanPipeline::create_matmul(device_.device(), spirv);
        if (!pl_r) return std::unexpected(pl_r.error());
        matmul_pipeline_ = std::move(*pl_r);
        memory_pool_ = std::make_unique<MemoryPool>(device_.device(), device_.physical_device());
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = device_.queue_family_index();
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        auto r = detail::vk_check(
            vkCreateCommandPool(device_.device(), &pool_info, nullptr, &command_pool_), __FILE__, __LINE__);
        if (!r) return r;
        staging_ring_ = std::make_unique<StagingRing>();
        auto st_r = staging_ring_->initialize(
            device_.device(), device_.physical_device(), command_pool_, memory_pool_.get());
        if (!st_r) return st_r;

        constexpr std::size_t total_descriptors = MAX_CACHED_DISPATCHES * DESCRIPTORS_PER_DISPATCH;
        VkDescriptorPoolSize dispatch_pool_size{};
        dispatch_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dispatch_pool_size.descriptorCount = static_cast<uint32_t>(total_descriptors);
        VkDescriptorPoolCreateInfo dispatch_pool_info{};
        dispatch_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dispatch_pool_info.maxSets = static_cast<uint32_t>(MAX_CACHED_DISPATCHES);
        dispatch_pool_info.poolSizeCount = 1;
        dispatch_pool_info.pPoolSizes = &dispatch_pool_size;
        dispatch_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        r = detail::vk_check(
            vkCreateDescriptorPool(device_.device(), &dispatch_pool_info, nullptr, &dispatch_pool_), __FILE__, __LINE__);
        if (!r) return r;
        // ── GpuTensor 专用描述符池（供 matmul_gpu / elementwise_gpu 使用）──
        constexpr std::size_t TENSOR_POOL_SETS = 1024;
        constexpr std::size_t TENSOR_POOL_DESCS = TENSOR_POOL_SETS * 3; // 最多 3 绑定/集
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
            vkCreateDescriptorPool(device_.device(), &tensor_pool_info, nullptr, &gpu_tensor_pool_), __FILE__, __LINE__);
        if (!r) return r;

        // ── 4×4 粗化分块矩阵乘法 Pipeline ──────────────────────────
        const auto& tiled_spirv = get_matmul_tiled_spirv();
        if (!tiled_spirv.empty())
        {
            auto tp_r = VulkanPipeline::create_matmul(device_.device(), tiled_spirv);
            if (tp_r) matmul_tiled_pipeline_ = std::move(*tp_r);
        }

        // ── 逐元素运算 Pipeline ─────────────────────────────────────
        const auto& elem_spirv = get_elementwise_spirv();
        if (!elem_spirv.empty())
        {
            auto ep_r = VulkanPipeline::create_elementwise(device_.device(), elem_spirv);
            if (ep_r) elementwise_pipeline_ = std::move(*ep_r);
        }

        initialized_ = true;
        return {};
    }
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] bool has_tiled_pipeline() const noexcept { return matmul_tiled_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_elementwise_pipeline() const noexcept { return elementwise_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] VulkanDevice& device() noexcept { return device_; }
    [[nodiscard]] MemoryPool& memory_pool() noexcept { return *memory_pool_; }
    [[nodiscard]] StagingRing& staging_ring() noexcept { return *staging_ring_; }

    // ══════════════════════════════════════════════════════════════════════
    // 双轨制架构：GpuTensor 纯 GPU 流水线 API
    // ══════════════════════════════════════════════════════════════════════

    // ── 阻塞式上传：CPU Matrix → GpuTensor（仅在输入层调用）──────────
    [[nodiscard]] Result<void> upload_blocking(
        GpuTensor& dst, std::span<const double> cpu_data)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (!dst.valid()) return std::unexpected(Error{"Invalid destination GpuTensor"});
        const std::size_t elem_count = dst.rows() * dst.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Upload size mismatch"});

        auto ri = staging_ring_->acquire();
        auto r = staging_ring_->upload(ri, cpu_data, 0);
        if (!r) return r;

        // 录制 Copy 命令：Staging → Device Local
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd), __FILE__, __LINE__);
        if (!r) return r;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r) return r;

        VkBufferCopy cp{0, 0, elem_count * sizeof(float)};
        vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl().handle(), 1, &cp);

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
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence), __FILE__, __LINE__);
            if (!r) return r;
        }
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL), __FILE__, __LINE__);
        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
        return r;
    }

    // ── 阻塞式下载：GpuTensor → CPU span（仅在输出层调用）──────────
    [[nodiscard]] Result<void> download_blocking(
        const GpuTensor& src, std::span<double> cpu_data)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (!src.valid()) return std::unexpected(Error{"Invalid source GpuTensor"});
        const std::size_t elem_count = src.rows() * src.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Download size mismatch"});

        // 先 flush 所有待处理的 GPU 操作
        flush_pending_ops();

        auto ri = staging_ring_->acquire();

        // 录制 Copy 命令：Device Local → Staging
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        auto r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd), __FILE__, __LINE__);
        if (!r) return r;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r) return r;

        VkBufferCopy cp{0, 0, elem_count * sizeof(float)};
        vkCmdCopyBuffer(cmd, src.buffer().impl().handle(), staging_ring_->buffer(ri), 1, &cp);

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
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence), __FILE__, __LINE__);
            if (!r) return r;
        }
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL), __FILE__, __LINE__);
        if (!r) return r;

        r = staging_ring_->download(ri, cpu_data, 0);
        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
        return r;
    }

    // ── 纯 GPU 矩阵乘法（零 PCIe 开销，提交后不等待）──────────────
    // 使用 4×4 粗化分块 Shader（如可用），否则回退到原始 Shader
    [[nodiscard]] Result<GpuTensor> matmul_gpu(const GpuTensor& A, const GpuTensor& B)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (A.cols() != B.rows()) return std::unexpected(Error{"Dimension mismatch"});

        const auto M = static_cast<uint32_t>(A.rows());
        const auto K = static_cast<uint32_t>(A.cols());
        const auto N = static_cast<uint32_t>(B.cols());

        // 1. 分配输出 Tensor
        auto C_res = GpuTensor::create_empty(M, N, *this);
        if (!C_res) return std::unexpected(C_res.error());
        GpuTensor C = std::move(*C_res);

        // 2. 选择 Pipeline（优先使用粗化分块版本）
        const bool use_tiled = has_tiled_pipeline();
        auto& pipeline = use_tiled ? matmul_tiled_pipeline_ : matmul_pipeline_;

        // 3. 分配描述符集
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = pipeline.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;
        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 4. 写入描述符集（绑定 A, B, C 的 Device Local Buffer）
        VkDescriptorBufferInfo buf_infos[3]{
            {A.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
            {B.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
            {C.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
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

        // 5. 录制 Command Buffer
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 管线屏障：确保输入数据就绪
        // A 可能来自上一层的 SHADER_WRITE 或初始的 TRANSFER_WRITE
        // B（权重）来自最初的 TRANSFER_WRITE
        VkBufferMemoryBarrier input_barriers[2]{};
        for (int i = 0; i < 2; ++i)
        {
            input_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            input_barriers[i].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            input_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            input_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].offset = 0;
            input_barriers[i].size = VK_WHOLE_SIZE;
        }
        input_barriers[0].buffer = A.buffer().impl().handle();
        input_barriers[1].buffer = B.buffer().impl().handle();
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 2, input_barriers, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);
        const uint32_t push_data[3] = {M, N, K};
        vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // Dispatch 维度取决于使用哪个 Shader
        if (use_tiled)
            vkCmdDispatch(cmd, (N + TILE_MN - 1u) / TILE_MN, (M + TILE_MN - 1u) / TILE_MN, 1);
        else
            vkCmdDispatch(cmd, (N + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE,
                              (M + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE, 1);

        // 管线屏障：确保输出数据就绪（供下一层或下载使用）
        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = C.buffer().impl().handle();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 6. 提交到 Compute Queue（不等待！CPU 立即返回）
        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
        }

        // 7. 追踪待清理资源
        pending_ops_.cmd_buffers.push_back(cmd);
        pending_ops_.desc_sets.push_back(desc_set);

        return C;
    }

    // ── 纯 GPU 逐元素运算（ReLU / GeLU / BiasAdd，提交后不等待）──────
    // op: 0=ReLU, 1=QuickGeLU, 2=BiasAdd
    // 对于 op=2（BiasAdd）：primary 是 matmul 结果，secondary 是 bias 向量
    [[nodiscard]] Result<GpuTensor> elementwise_gpu(
        const GpuTensor& primary, const GpuTensor* secondary, uint32_t op,
        uint32_t rows = 0, uint32_t cols = 0)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_elementwise_pipeline())
            return std::unexpected(Error{"Elementwise pipeline not available"});

        const auto elem_count = static_cast<uint32_t>(primary.rows() * primary.cols());

        // 分配输出 Tensor
        auto out_res = GpuTensor::create_empty(primary.rows(), primary.cols(), *this);
        if (!out_res) return std::unexpected(out_res.error());
        GpuTensor output = std::move(*out_res);

        // secondary 为空时用 primary 自身（unary op 场景）
        const GpuTensor& sec = secondary ? *secondary : primary;

        // 分配描述符集
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = elementwise_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;
        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkDescriptorBufferInfo buf_infos[3]{
            {primary.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
            {sec.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl().handle(), 0, VK_WHOLE_SIZE},
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

        // 录制 Command Buffer
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 输入屏障（primary 和 secondary）
        VkBufferMemoryBarrier in_barriers[2]{};
        for (int i = 0; i < 2; ++i)
        {
            in_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            in_barriers[i].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            in_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            in_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            in_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            in_barriers[i].offset = 0; in_barriers[i].size = VK_WHOLE_SIZE;
        }
        in_barriers[0].buffer = primary.buffer().impl().handle();
        in_barriers[1].buffer = sec.buffer().impl().handle();
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 2, in_barriers, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, elementwise_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            elementwise_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);
        const uint32_t push_data[4] = {elem_count, op, rows, cols};
        vkCmdPushConstants(cmd, elementwise_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);
        vkCmdDispatch(cmd, (elem_count + 255u) / 256u, 1, 1);

        // 输出屏障
        VkBufferMemoryBarrier out_barrier{};
        out_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        out_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        out_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        out_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        out_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        out_barrier.buffer = output.buffer().impl().handle();
        out_barrier.offset = 0; out_barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &out_barrier, 0, nullptr);

        r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 提交（不等待）
        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE), __FILE__, __LINE__);
            if (!r) return std::unexpected(r.error());
        }

        pending_ops_.cmd_buffers.push_back(cmd);
        pending_ops_.desc_sets.push_back(desc_set);

        return output;
    }

    [[nodiscard]] Result<void> matmul_direct(
        std::span<const double> a_data,
        std::span<const double> b_data,
        std::span<double> c_data,
        std::size_t M, std::size_t N, std::size_t K)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        MatmulKey key{static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K)};

        CachedDispatch* dp = nullptr;
        {
            std::lock_guard lock(cache_mutex_);
            auto it = dispatch_cache_.find(key);
            if (it == dispatch_cache_.end())
            {
                if (dispatch_cache_.size() >= MAX_CACHED_DISPATCHES) evict_oldest_dispatch();
                auto result = create_cached_dispatch(key);
                if (!result) return std::unexpected(result.error());
                auto [inserted_it, _] = dispatch_cache_.emplace(key, std::move(*result));
                it = inserted_it;
                dispatch_lru_.push_front(key);
                dispatch_lru_map_[key] = dispatch_lru_.begin();
            }
            else
            {
                auto lru_it = dispatch_lru_map_[key];
                dispatch_lru_.erase(lru_it);
                dispatch_lru_.push_front(key);
                dispatch_lru_map_[key] = dispatch_lru_.begin();
            }
            dp = &it->second;
            dp->ref_count.fetch_add(1, std::memory_order_release);
        }

        struct RefGuard {
            CachedDispatch* d;
            ~RefGuard() { if (d) d->ref_count.fetch_sub(1, std::memory_order_release); }
        } ref_guard{dp};
        CachedDispatch& d = *dp;

        const std::size_t a_sz = M * K;
        const std::size_t b_sz = K * N;
        const std::size_t c_sz = M * N;
        const std::size_t total_bytes = (a_sz + b_sz + c_sz) * sizeof(float);
        if (total_bytes > staging_ring_->region_size())
            return std::unexpected(Error{"Matrix too large for staging region"});

        auto ri = staging_ring_->acquire();
        auto r = staging_ring_->upload(ri, a_data, 0);
        if (!r) return r;
        r = staging_ring_->upload(ri, b_data, a_sz * sizeof(float));
        if (!r) return r;

        auto cmd = d.pre_recorded_cmds[ri]; // 直接使用预录制命令
        auto fence = staging_ring_->fence(ri);

        vkResetFences(device_.device(), 1, &fence);
        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                __FILE__, __LINE__);
            if (!r) return r;
        }
        
        staging_ring_->mark_in_flight(ri); // 修复：正确标记 in_flight
        
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
            __FILE__, __LINE__);
        if (!r) return r;

        r = staging_ring_->download(ri, c_data, (a_sz + b_sz) * sizeof(float));
        if (!r) return r;
        return {};
    }
};

// GpuTensor 方法实现已移至 gpu_tensor_impl.hpp（在 matrix.hpp 末尾自动包含）
} // namespace nn
#endif // NN_HAS_VULKAN
#endif // VK_BACKEND_HPP