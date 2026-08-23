// ── memory_pool.hpp ─────────────────────────────────────────────────────
// GPU 内存子分配器
//
// 职责：
//   - 预分配大块 GPU 内存（128MB+）
//   - 提供 O(log n) 子分配/释放
//   - 自动合并空闲区域
//   - RAII 管理 VkDeviceMemory
//
// 设计：
//   - 每个 Block 是一块大的 VkDeviceMemory
//   - 使用 std::set<FreeRegion> 维护空闲区域（按 offset 排序）
//   - 分配时找第一个合适的空闲区域（first fit）
//   - 释放时自动合并相邻空闲区域
// ─────────────────────────────────────────────────────────────────────────

#ifndef NN_MEMORY_POOL_HPP
#define NN_MEMORY_POOL_HPP

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core_errors.hpp"

namespace nn
{

class MemoryPool
{
public:
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 128ull * 1024 * 1024; // 128MB

    struct Allocation
    {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkMemoryPropertyFlags property_flags = 0;

        [[nodiscard]] bool valid() const noexcept { return memory != VK_NULL_HANDLE; }
    };

private:
    struct FreeRegion
    {
        VkDeviceSize offset;
        VkDeviceSize size;

        bool operator<(const FreeRegion& o) const noexcept { return offset < o.offset; }
    };

    struct Block
    {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        uint32_t memory_type_index = 0;
        VkMemoryPropertyFlags property_flags = 0;
        std::set<FreeRegion> free_regions;
        std::size_t allocation_count = 0;
        VkDevice owning_device = VK_NULL_HANDLE;

        Block() = default;

        Block(Block&& o) noexcept
            : memory(o.memory), size(o.size), memory_type_index(o.memory_type_index),
              property_flags(o.property_flags), free_regions(std::move(o.free_regions)),
              allocation_count(o.allocation_count), owning_device(o.owning_device)
        {
            o.memory = VK_NULL_HANDLE;
            o.size = 0;
            o.allocation_count = 0;
            o.owning_device = VK_NULL_HANDLE;
        }

        // 禁用移动赋值，防止 vector 操作引发意外释放
        Block& operator=(Block&&) = delete;

        ~Block()
        {
            if (memory != VK_NULL_HANDLE && owning_device != VK_NULL_HANDLE)
                vkFreeMemory(owning_device, memory, nullptr);
        }

        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
    };

    struct SuballocKey
    {
        VkDeviceMemory memory;
        VkDeviceSize offset;

        bool operator==(const SuballocKey& o) const noexcept
        {
            return memory == o.memory && offset == o.offset;
        }
    };

    struct SuballocKeyHash
    {
        std::size_t operator()(const SuballocKey& k) const noexcept
        {
            // Knuth 黄金比例散列常数（64-bit），用于打散相邻 key 的散列值
            constexpr std::size_t KNUTH_GOLDEN = 0x9E3779B97F4A7C15ULL;
            return std::hash<VkDeviceMemory>{}(k.memory) ^
                   (std::hash<VkDeviceSize>{}(k.offset) * KNUTH_GOLDEN);
        }
    };

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkDeviceSize block_size_;
    VkPhysicalDeviceMemoryProperties mem_props_;

    // L2：整块归还时保留的最小空闲字节数（避免频繁整块释放/重建抖动）
    VkDeviceSize retain_free_bytes_ = 0;

    // 使用 unique_ptr 避免 vector 扩容/删除时触发 Block 的移动和析构
    std::vector<std::unique_ptr<Block>> blocks_;
    std::unordered_map<SuballocKey, VkDeviceSize, SuballocKeyHash> active_allocs_;
    mutable std::mutex mutex_;

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
                if (padding > 0)
                    block.free_regions.insert({orig_offset, padding});
                if (remaining > 0)
                    block.free_regions.insert({aligned + size, remaining});
                return result;
            }
        }
        return std::nullopt;
    }

    // 注：Block 的移动赋值被 delete（防止 vector 操作引发意外释放），
    // 但移动构造可用。此处 `return block;` 经由移动构造构造 std::expected<Block, Error>，
    // 不触发移动赋值，故可正常编译。
    [[nodiscard]] Result<Block> create_block(
        uint32_t memory_type_index, VkMemoryPropertyFlags flags, VkDeviceSize size = 0)
    {
        VkDeviceSize alloc_size = (size > 0) ? size : block_size_;
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = alloc_size;
        alloc_info.memoryTypeIndex = memory_type_index;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkResult res = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkAllocateMemory failed: " + std::to_string(res)});

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
        uint32_t type_bits, VkMemoryPropertyFlags preferred,
        VkMemoryPropertyFlags fallback = 0) const
    {
        std::optional<uint32_t> best, fb;
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i)
        {
            if (!(type_bits & (1u << i)))
                continue;
            auto flags = mem_props_.memoryTypes[i].propertyFlags;
            if ((flags & preferred) == preferred && !best)
                best = i;
            if (fallback && (flags & fallback) == fallback && !fb)
                fb = i;
        }
        return best ? best : fb;
    }

public:
    MemoryPool(VkDevice device, VkPhysicalDevice physical_device,
               VkDeviceSize block_size = DEFAULT_BLOCK_SIZE)
        : device_(device), physical_device_(physical_device), block_size_(block_size)
    {
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props_);
    }

    ~MemoryPool() { blocks_.clear(); }

    // 禁止拷贝和移动
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;

    [[nodiscard]] Result<Allocation> allocate(
        VkMemoryRequirements requirements,
        VkMemoryPropertyFlags preferred_flags,
        VkMemoryPropertyFlags fallback_flags = 0)
    {
        std::lock_guard lock(mutex_);

        auto mem_type = find_memory_type(requirements.memoryTypeBits, preferred_flags, fallback_flags);
        if (!mem_type)
            return std::unexpected(Error{"No suitable GPU memory type found"});

        VkDeviceSize alignment = requirements.alignment > 0 ? requirements.alignment : 1;
        const VkDeviceSize alloc_size = requirements.size;
        VkDeviceSize effective_block_size = (alloc_size > block_size_) ? alloc_size : block_size_;

        // 尝试在现有 block 中分配
        for (auto& block_ptr : blocks_)
        {
            auto& block = *block_ptr;
            if (block.memory_type_index != *mem_type)
                continue;
            auto offset = find_free(block, alignment, alloc_size);
            if (offset)
            {
                block.allocation_count++;
                active_allocs_[{block.memory, *offset}] = alloc_size;
                return Allocation{block.memory, *offset, alloc_size, block.property_flags};
            }
        }

        // 创建新 block
        auto block_result = create_block(*mem_type, preferred_flags, effective_block_size);
        if (!block_result)
            return std::unexpected(block_result.error());

        auto new_block = std::make_unique<Block>(std::move(*block_result));
        auto offset = find_free(*new_block, alignment, alloc_size);
        if (!offset)
            return std::unexpected(Error{"Suballocation failed in new block"});

        VkDeviceMemory mem = new_block->memory;
        VkMemoryPropertyFlags flags = new_block->property_flags;
        new_block->allocation_count++;
        active_allocs_[{mem, *offset}] = alloc_size;
        blocks_.push_back(std::move(new_block));

        return Allocation{mem, *offset, alloc_size, flags};
    }

    void free(const Allocation& alloc)
    {
        if (!alloc.valid())
            return;

        std::lock_guard lock(mutex_);
        SuballocKey key{alloc.memory, alloc.offset};
        auto active_it = active_allocs_.find(key);
        if (active_it == active_allocs_.end())
            return;

        for (auto& block_ptr : blocks_)
        {
            auto& block = *block_ptr;
            if (block.memory != alloc.memory)
                continue;

            FreeRegion new_reg{alloc.offset, alloc.size};
            auto ins_it = block.free_regions.insert(new_reg).first;

            // 向前合并
            if (ins_it != block.free_regions.begin())
            {
                auto prev = std::prev(ins_it);
                if (prev->offset + prev->size == ins_it->offset)
                {
                    FreeRegion merged{prev->offset, prev->size + ins_it->size};
                    block.free_regions.erase(prev);
                    block.free_regions.erase(ins_it);
                    ins_it = block.free_regions.insert(merged).first;
                }
            }

            // 向后合并
            auto next = std::next(ins_it);
            if (next != block.free_regions.end())
            {
                if (ins_it->offset + ins_it->size == next->offset)
                {
                    FreeRegion merged{ins_it->offset, ins_it->size + next->size};
                    block.free_regions.erase(next);
                    block.free_regions.erase(ins_it);
                    block.free_regions.insert(merged);
                }
            }

            block.allocation_count--;
            active_allocs_.erase(active_it);
            return;
        }
    }

    // ── UMA/共享显存检测 ─────────────────────────────────────────────
    // 判断设备是否为统一内存架构：所有 DEVICE_LOCAL 内存类型都可被 HOST_VISIBLE
    // 访问（iGPU/APU：device-local 即 host-visible，共享同一内存池）。
    // 独立显卡（如 V100）存在"纯 device-local、不可 host 映射"的专用 VRAM
    // heap，此时即使驱动报告 DEVICE_LOCAL|HOST_VISIBLE 组合类型，那也是
    // GPU 可驻留/Zero-copy 内存（仍占专用显存）——offload 到它无效。
    // 返回 true 表示可安全地把 offload slab 放 DEVICE_LOCAL|HOST_VISIBLE。
    [[nodiscard]] bool is_uma() const noexcept
    {
        bool has_device_local = false;
        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i)
        {
            const auto flags = mem_props_.memoryTypes[i].propertyFlags;
            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            {
                has_device_local = true;
                if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
                    return false;  // 存在纯 device-local（独显专用 VRAM）
            }
        }
        return has_device_local;  // 全部 device-local 均可 host 映射 → UMA
    }

    // ── 池统计（L2 仪器化，供显存采样/逐项归因） ──────────────────────
    struct PoolStats
    {
        std::size_t block_count = 0;       // 活跃 block 数
        std::size_t allocation_count = 0;  // 活跃子分配数
        VkDeviceSize total_bytes = 0;      // 所有 block 底材总大小
        VkDeviceSize free_bytes = 0;       // 空闲区域总大小
        VkDeviceSize allocated_bytes = 0;  // total - free
        VkDeviceSize max_contiguous_free = 0;  // 最大连续空闲块
        double fragmentation = 0.0;        // 1 - 最大连续空闲/总空闲
        VkDeviceSize device_bytes = 0;     // DEVICE_LOCAL 块（真实显存）
        VkDeviceSize host_bytes = 0;       // HOST_VISIBLE 块（host RAM，offload 用）

        [[nodiscard]] std::string to_string() const
        {
            constexpr std::size_t MB = 1024 * 1024;
            std::string s;
            s += "blocks=" + std::to_string(block_count);
            s += " allocs=" + std::to_string(allocation_count);
            s += " total=" + std::to_string(total_bytes / MB) + "MB";
            s += " dev=" + std::to_string(device_bytes / MB) + "MB";
            s += " host=" + std::to_string(host_bytes / MB) + "MB";
            s += " free=" + std::to_string(free_bytes / MB) + "MB";
            s += " frag=" + std::to_string(fragmentation);
            return s;
        }
    };

    [[nodiscard]] PoolStats pool_debug_stats() const
    {
        std::lock_guard lock(mutex_);
        PoolStats s;
        s.block_count = blocks_.size();
        for (const auto& bp : blocks_)
        {
            const auto& b = *bp;
            s.allocation_count += b.allocation_count;
            s.total_bytes += b.size;
            if (b.property_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                s.device_bytes += b.size;
            if (b.property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                s.host_bytes += b.size;
            for (const auto& r : b.free_regions)
            {
                s.free_bytes += r.size;
                if (r.size > s.max_contiguous_free)
                    s.max_contiguous_free = r.size;
            }
        }
        s.allocated_bytes = s.total_bytes - s.free_bytes;
        if (s.free_bytes > 0)
            s.fragmentation = 1.0 - static_cast<double>(s.max_contiguous_free)
                                   / static_cast<double>(s.free_bytes);
        return s;
    }

    // ── 整块归还（L2）───────────────────────────────────────────────
    // 设置整块归还时保留的最小空闲字节（默认 0 = 尽量回收）。
    void set_retain_free_bytes(VkDeviceSize bytes) { retain_free_bytes_ = bytes; }

    // 释放“完全空闲且超出保留阈值”的 block 底材（调用 vkFreeMemory 并移出池）。
    // 调用时机：训练 step 边界（end_batch 提交完成、延迟销毁已 flush 之后），
    // 避免在 batch 录制中释放仍被引用/延迟销毁的 buffer 底材。
    void release_idle_blocks()
    {
        std::lock_guard lock(mutex_);
        // 当前总空闲（用于保留阈值判断）
        VkDeviceSize total_free = 0;
        for (const auto& bp : blocks_)
            for (const auto& r : bp->free_regions)
                total_free += r.size;

        for (auto it = blocks_.begin(); it != blocks_.end(); )
        {
            const auto& b = **it;
            // 完全空闲：无活跃子分配，且整个 block 被单个空闲区覆盖
            if (b.allocation_count == 0 &&
                b.free_regions.size() == 1 &&
                b.free_regions.begin()->offset == 0 &&
                b.free_regions.begin()->size == b.size)
            {
                // 仅当释放后总空闲仍 ≥ 保留目标时才释放（避免抖动）
                if (total_free - b.size >= retain_free_bytes_)
                {
                    total_free -= b.size;
                    // 通过 erase 触发 Block 析构（析构中 vkFreeMemory）
                    it = blocks_.erase(it);
                    continue;
                }
            }
            ++it;
        }
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // NN_MEMORY_POOL_HPP
