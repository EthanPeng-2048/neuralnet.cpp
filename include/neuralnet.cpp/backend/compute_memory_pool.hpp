// ── compute_memory_pool.hpp ─────────────────────────────────────────────────────
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

#pragma once

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
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
    // ── 底材尺寸（2026-09 探针实测调参，跨负载数据见下）─────────────────
    // 128MB → 12MB。实测峰值对块尺寸在 8–32MB 是平台区、最优 ≈12MB，且
    // **与模型规模无关**（真正的"自适应"是另一条规则：>底材的分配自动独占
    // 精确尺寸块，不产生内部碎片）。跨负载实测（MiB）：
    //   GPT d64/L4/b64 : 8MB=3411 12MB=3399 16MB=3411 32MB=3427 128MB=3523
    //   GPT d128/L8/b32: 8MB=4449 12MB=4445 16MB=4457 32MB=4457 128MB=4557
    //   MNIST MLP      : 8MB=82   12MB=94   16MB=90   32MB=122  128MB=314
    //   MNIST CNN      :          12MB=142  16MB=154            128MB=314
    // 另测过"最小 2 的幂阶梯类"（NN_POOL_LADDER_MAX_MB 启用）：d128 上
    // ladder16=4497 比 fixed16=4457 差 40MB（配对 3/3），故阶梯默认关闭。
    // 可用 NN_POOL_BLOCK_MB 覆盖做调参。
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 12ull * 1024 * 1024; // 12MB
    // 尺寸分类分池（抗碎片，P3）：大块底材只服务大分配，小块底材只服务小分配，
    // 避免大量高频的小临时分配在 128MB 底材里切出不可复用碎片、破坏大分配的
    // 连续性（见《显存&负载不均衡分析》）。
    static constexpr VkDeviceSize DEFAULT_SMALL_BLOCK_SIZE = 4ull * 1024 * 1024;  // 4MB
    // 小于该阈值视为"小分配"，走小块底材池。
    static constexpr VkDeviceSize SMALL_ALLOC_THRESHOLD = 256ull * 1024;           // 256KB

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
    VkDeviceSize small_block_size_;
    VkPhysicalDeviceMemoryProperties mem_props_;

    // L2：整块归还时保留的最小空闲字节数（避免频繁整块释放/重建抖动）
    VkDeviceSize retain_free_bytes_ = 0;

    // 自适应阶梯上限（0 = 关闭，走固定 block_size_ 单类别路径）：
    // 启用时中等分配按"最小 2 的幂类别"分池，见 allocate()。
    VkDeviceSize ladder_max_ = 0;

    // 使用 unique_ptr 避免 vector 扩容/删除时触发 Block 的移动和析构
    std::vector<std::unique_ptr<Block>> blocks_;
    std::unordered_map<SuballocKey, VkDeviceSize, SuballocKeyHash> active_allocs_;
    mutable std::mutex mutex_;

    // ── 诊断计数器（NN_MEM_STATS / pool_stats 打印；每条只是整数自增）──────
    // 用途：把"池粒度调细后耗时翻倍"归因到具体机制（线性扫描 vs 底材申请），
    // 而不是靠猜。c_block_scans_ = allocate 里被检查的 block 数（Σ），
    // c_region_scans_ = find_best 里被检查的空闲区数（Σ），
    // c_blocks_created_ / c_vkalloc_ms_ = vkAllocateMemory 次数与累计耗时。
    mutable std::size_t c_alloc_calls_ = 0;
    mutable std::size_t c_free_calls_ = 0;
    mutable std::size_t c_block_scans_ = 0;
    mutable std::size_t c_region_scans_ = 0;
    mutable std::size_t c_blocks_created_ = 0;
    mutable std::size_t c_blocks_released_ = 0;
    mutable double c_vkalloc_ms_ = 0.0;

    // 空闲区查找：best-fit（P3 抗碎片）。
    // 在候选块内选"对齐后剩余碎片最小 && 对齐 padding 不超预算"的空闲区。
    // 相比 first-fit，best-fit 降低把一个大小合适的洞切成两个小洞的概率，
    // 且跳过对齐 padding 过大的候选（避免产生不可复用的残片）。
    // 返回 (offset, padding_bytes)；无合适候选返回 nullopt。
    [[nodiscard]] static std::optional<std::pair<VkDeviceSize, VkDeviceSize>> find_best(
        Block& block, VkDeviceSize alignment, VkDeviceSize size)
    {
        const VkDeviceSize align_mask = alignment - 1;
        // 对齐 padding 预算：超过该值即视为"残片"，跳过该候选。
        const VkDeviceSize max_padding =
            std::max<VkDeviceSize>(64, (size >= 16384) ? (size >> 4) : (size >> 1));

        std::optional<VkDeviceSize> best_offset;
        VkDeviceSize best_padding = 0;
        VkDeviceSize best_region_start = 0;
        VkDeviceSize best_waste = std::numeric_limits<VkDeviceSize>::max();

        for (const auto& r : block.free_regions)
        {
            const VkDeviceSize aligned = (r.offset + align_mask) & ~align_mask;
            const VkDeviceSize padding = aligned - r.offset;
            if (padding > max_padding)
                continue;  // 对齐残片过大，跳过
            if (r.size < padding + size)
                continue;  // 放不下
            // 剩余碎片 = 该区分配后剩下的字节数（越小越贴合）
            const VkDeviceSize waste = r.size - (padding + size);
            if (waste < best_waste)
            {
                best_waste = waste;
                best_offset = aligned;
                best_padding = padding;
                best_region_start = r.offset;
            }
        }
        if (!best_offset)
            return std::nullopt;

        // 定位被选中的区域并从集合中剔除，按 padding/remaining 重插
        auto it = block.free_regions.lower_bound(FreeRegion{best_region_start, 1});
        if (it == block.free_regions.end() || it->offset != best_region_start)
            return std::nullopt;  // 防御：理论不可达
        const VkDeviceSize remaining = it->size - best_padding - size;
        block.free_regions.erase(it);
        if (best_padding > 0)
            block.free_regions.insert({best_region_start, best_padding});
        if (remaining > 0)
            block.free_regions.insert({*best_offset + size, remaining});
        return std::make_pair(*best_offset, best_padding);
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
        const auto t0 = std::chrono::steady_clock::now();
        VkResult res = vkAllocateMemory(device_, &alloc_info, nullptr, &memory);
        c_vkalloc_ms_ += std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count();
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkAllocateMemory failed: " + std::to_string(res)});
        c_blocks_created_++;

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
               VkDeviceSize block_size = DEFAULT_BLOCK_SIZE,
               VkDeviceSize small_block_size = DEFAULT_SMALL_BLOCK_SIZE,
               VkDeviceSize ladder_max = 0)
        : device_(device), physical_device_(physical_device), block_size_(block_size),
          small_block_size_(small_block_size), retain_free_bytes_(0),
          ladder_max_(ladder_max)
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
        c_alloc_calls_++;

        auto mem_type = find_memory_type(requirements.memoryTypeBits, preferred_flags, fallback_flags);
        if (!mem_type)
            return std::unexpected(Error{"No suitable GPU memory type found"});

        VkDeviceSize alignment = requirements.alignment > 0 ? requirements.alignment : 1;
        // matmul_tiled 等 vec4 SSBO 快路径要求基址 16B 对齐（std430 vec4 读的
        // 对齐门槛）。buffer requirements 可能小于 16，这里统一抬到 ≥16；
        // 仍保持 2 的幂，find_best 的位掩码取整不受影响。代价：每次分配
        // 最多 12B padding。
        if (alignment < 16u)
            alignment = 16u;
        const VkDeviceSize alloc_size = requirements.size;
        // 尺寸分类分池：超大分配独占整块；小分配走小块池；其余走大块池。
        // 顺序：alloc > 大块阈值 → 独占超大块；alloc < 小块阈值 → 小块池。
        VkDeviceSize pool_size;
        if (ladder_max_ > 0 && alloc_size >= SMALL_ALLOC_THRESHOLD)
        {
            // ── 自适应阶梯（2026-09 实测调参）──────────────────────────
            // 固定类别无法同时服务大小负载：小负载要细（抗内部碎片），大
            // 负载要粗（少 vkAllocateMemory）。改为"类别 = 不小于分配尺寸的
            // 最小 2 的幂"，夹在 [small_block_size_, ladder_max_]；超出上限
            // 则独占精确尺寸块。等价于把原来的单一中等类别拆成一串类别。
            VkDeviceSize c = small_block_size_;
            while (c < alloc_size && c < ladder_max_) c <<= 1;
            pool_size = (alloc_size > ladder_max_) ? alloc_size : c;
        }
        else if (alloc_size > block_size_)
            pool_size = alloc_size;                 // 超出大块底材：独占整块
        else if (alloc_size < SMALL_ALLOC_THRESHOLD)
            pool_size = small_block_size_;          // 小分配：小块底材池（抗碎片）
        else
            pool_size = block_size_;

        // 尝试在现有同类块中分配（best-fit）
        for (auto& block_ptr : blocks_)
        {
            auto& block = *block_ptr;
            c_block_scans_++;
            if (block.memory_type_index != *mem_type)
                continue;
            if (block.size != pool_size)
                continue;  // 只服务同尺寸分类的块，保持大块连续性
            c_region_scans_ += block.free_regions.size();
            auto best = find_best(block, alignment, alloc_size);
            if (best)
            {
                block.allocation_count++;
                active_allocs_[{block.memory, best->first}] = alloc_size;
                return Allocation{block.memory, best->first, alloc_size, block.property_flags};
            }
        }

        // 创建新块
        auto block_result = create_block(*mem_type, preferred_flags, pool_size);
        if (!block_result)
            return std::unexpected(block_result.error());

        auto new_block = std::make_unique<Block>(std::move(*block_result));
        auto best = find_best(*new_block, alignment, alloc_size);
        if (!best)
            return std::unexpected(Error{"Suballocation failed in new block"});

        VkDeviceMemory mem = new_block->memory;
        VkMemoryPropertyFlags flags = new_block->property_flags;
        new_block->allocation_count++;
        active_allocs_[{mem, best->first}] = alloc_size;
        blocks_.push_back(std::move(new_block));

        return Allocation{mem, best->first, alloc_size, flags};
    }

    void free(const Allocation& alloc)
    {
        if (!alloc.valid())
            return;

        std::lock_guard lock(mutex_);
        c_free_calls_++;
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

        // 池账本计数器（诊断，见 MemoryPool::c_* 注释）
        std::size_t c_alloc_calls = 0;
        std::size_t c_free_calls = 0;
        std::size_t c_block_scans = 0;
        std::size_t c_region_scans = 0;
        std::size_t c_blocks_created = 0;
        std::size_t c_blocks_released = 0;
        double vkalloc_ms = 0.0;

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
            // 池账本计数器（诊断；把"粒度变细 → 耗时翻倍"归因到线性扫描/底材申请）
            s += " | calls a/f=" + std::to_string(c_alloc_calls) + "/" +
                 std::to_string(c_free_calls);
            s += " scans blk=" + std::to_string(c_block_scans) + " reg=" +
                 std::to_string(c_region_scans);
            s += " blk_new/free=" + std::to_string(c_blocks_created) + "/" +
                 std::to_string(c_blocks_released);
            s += " vkalloc=" + std::to_string(vkalloc_ms) + "ms";
            return s;
        }
    };

    [[nodiscard]] PoolStats pool_debug_stats() const
    {
        std::lock_guard lock(mutex_);
        PoolStats s;
        s.block_count = blocks_.size();
        s.c_alloc_calls = c_alloc_calls_;
        s.c_free_calls = c_free_calls_;
        s.c_block_scans = c_block_scans_;
        s.c_region_scans = c_region_scans_;
        s.c_blocks_created = c_blocks_created_;
        s.c_blocks_released = c_blocks_released_;
        s.vkalloc_ms = c_vkalloc_ms_;
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

    // ── 活跃分配逐项尺寸（探针归因用）：降序返回每个在用分配的字节数 ──
    // 供 mem_probe 在阶段采样点打印 top-N 张量，把池增量精确落到具体分配。
    [[nodiscard]] std::vector<VkDeviceSize> live_alloc_sizes() const
    {
        std::lock_guard lock(mutex_);
        std::vector<VkDeviceSize> sizes;
        sizes.reserve(active_allocs_.size());
        for (const auto& [key, sz] : active_allocs_)
        {
            (void)key;
            sizes.push_back(sz);
        }
        std::sort(sizes.rbegin(), sizes.rend());
        return sizes;
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
                    c_blocks_released_++;
                    continue;
                }
            }
            ++it;
        }
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN
