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
            return std::hash<VkDeviceMemory>{}(k.memory) ^
                   (std::hash<VkDeviceSize>{}(k.offset) * 2654435761ULL);
        }
    };

    VkDevice device_;
    VkPhysicalDevice physical_device_;
    VkDeviceSize block_size_;
    VkPhysicalDeviceMemoryProperties mem_props_;

    // 使用 unique_ptr 避免 vector 扩容/删除时触发 Block 的移动和析构
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
                if (padding > 0)
                    block.free_regions.insert({orig_offset, padding});
                if (remaining > 0)
                    block.free_regions.insert({aligned + size, remaining});
                return result;
            }
        }
        return std::nullopt;
    }

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
};

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // NN_MEMORY_POOL_HPP
