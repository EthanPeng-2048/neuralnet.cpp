// ── vk_backend.hpp ──────────────────────────────────────────────────────
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

#ifndef NN_VK_BACKEND_HPP
#define NN_VK_BACKEND_HPP

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
#include "../config.hpp"
#include "memory_pool.hpp"
#include "staging_ring.hpp"

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

namespace nn
{

// 前向声明
class GpuTensor;

namespace detail
{

// ── Vulkan 错误检查 ──────────────────────────────────────────────────────
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

// ── CPU Scalar ↔ GPU float 转换 ─────────────────────────────────────────
inline void convert_scalar_to_float(
    std::span<const Scalar> src, float* __restrict dst) noexcept
{
    if constexpr (std::is_same_v<Scalar, float>)
        std::memcpy(dst, src.data(), src.size() * sizeof(float));
    else
        std::transform(src.begin(), src.end(), dst,
                       [](Scalar v) { return static_cast<float>(v); });
}

inline void convert_float_to_scalar(
    const float* __restrict src, std::span<Scalar> dst) noexcept
{
    if constexpr (std::is_same_v<Scalar, float>)
        std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    else
        std::transform(src, src + dst.size(), dst.begin(),
                       [](float v) { return static_cast<Scalar>(v); });
}

} // namespace detail

// ══════════════════════════════════════════════════════════════════════════
// VulkanDevice — 设备与队列管理（RAII）
// ══════════════════════════════════════════════════════════════════════════

class VulkanDevice
{
private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = 0;
    bool initialized_ = false;

public:
    VulkanDevice() = default;

    ~VulkanDevice()
    {
        if (device_ != VK_NULL_HANDLE)
            vkDestroyDevice(device_, nullptr);
        if (instance_ != VK_NULL_HANDLE)
            vkDestroyInstance(instance_, nullptr);
    }

    // 禁止拷贝和移动
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) = delete;
    VulkanDevice& operator=(VulkanDevice&&) = delete;

    [[nodiscard]] Result<void> initialize()
    {
        if (initialized_)
            return {};

        // 1. 创建 VkInstance
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "neuralnet.cpp";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "neuralnet.cpp";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkResult res = vkCreateInstance(&instance_info, nullptr, &instance_);
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkCreateInstance failed: " + std::to_string(res)});

        // 2. 选择物理设备
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0)
            return std::unexpected(Error{"No Vulkan devices found"});

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

        // 优先选择独立显卡
        for (const auto& dev : devices)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                physical_device_ = dev;
                break;
            }
        }
        // 如果没有独立显卡，使用第一个设备
        if (physical_device_ == VK_NULL_HANDLE)
            physical_device_ = devices[0];

        // 3. 查找计算队列族
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

        bool found_queue = false;
        for (uint32_t i = 0; i < queue_family_count; ++i)
        {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                queue_family_index_ = i;
                found_queue = true;
                break;
            }
        }
        if (!found_queue)
            return std::unexpected(Error{"No compute queue family found"});

        // 4. 创建逻辑设备
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

        res = vkCreateDevice(physical_device_, &device_info, nullptr, &device_);
        if (res != VK_SUCCESS)
            return std::unexpected(Error{"vkCreateDevice failed: " + std::to_string(res)});

        // 5. 获取计算队列
        vkGetDeviceQueue(device_, queue_family_index_, 0, &compute_queue_);

        initialized_ = true;
        return {};
    }

    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
    [[nodiscard]] uint32_t queue_family_index() const noexcept { return queue_family_index_; }
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
};

// ══════════════════════════════════════════════════════════════════════════
// VulkanPipeline — Pipeline 管理（RAII）
// ══════════════════════════════════════════════════════════════════════════

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

    ~VulkanPipeline()
    {
        if (pipeline_ != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, pipeline_, nullptr);
        if (pipeline_layout_ != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_layout_ != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        if (shader_module_ != VK_NULL_HANDLE)
            vkDestroyShaderModule(device_, shader_module_, nullptr);
    }

    // 移动语义
    VulkanPipeline(VulkanPipeline&& o) noexcept
        : device_(o.device_), shader_module_(o.shader_module_),
          descriptor_layout_(o.descriptor_layout_),
          pipeline_layout_(o.pipeline_layout_), pipeline_(o.pipeline_)
    {
        o.device_ = VK_NULL_HANDLE;
        o.shader_module_ = VK_NULL_HANDLE;
        o.descriptor_layout_ = VK_NULL_HANDLE;
        o.pipeline_layout_ = VK_NULL_HANDLE;
        o.pipeline_ = VK_NULL_HANDLE;
    }

    VulkanPipeline& operator=(VulkanPipeline&& o) noexcept
    {
        if (this != &o)
        {
            std::swap(device_, o.device_);
            std::swap(shader_module_, o.shader_module_);
            std::swap(descriptor_layout_, o.descriptor_layout_);
            std::swap(pipeline_layout_, o.pipeline_layout_);
            std::swap(pipeline_, o.pipeline_);
        }
        return *this;
    }

    // 禁止拷贝
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    // 创建 matmul pipeline
    [[nodiscard]] static Result<VulkanPipeline> create_matmul(
        VkDevice device, std::span<const uint32_t> spirv_code)
    {
        VulkanPipeline pl;
        pl.device_ = device;

        // 创建 shader module
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv_code.size_bytes();
        module_info.pCode = spirv_code.data();

        auto r = detail::vk_check(
            vkCreateShaderModule(device, &module_info, nullptr, &pl.shader_module_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 descriptor set layout（3 个 storage buffer）
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
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 pipeline layout（push constants: M, N, K, transA, transB）
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 5 * sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pl_layout_info{};
        pl_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_layout_info.setLayoutCount = 1;
        pl_layout_info.pSetLayouts = &pl.descriptor_layout_;
        pl_layout_info.pushConstantRangeCount = 1;
        pl_layout_info.pPushConstantRanges = &push_range;

        r = detail::vk_check(
            vkCreatePipelineLayout(device, &pl_layout_info, nullptr, &pl.pipeline_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 compute pipeline
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = pl.shader_module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pl.pipeline_layout_;

        r = detail::vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pl.pipeline_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        return pl;
    }

    // 创建通用 pipeline（可变 binding 数和 push constant 大小）
    // 用于 elementwise_v2 / reduce / broadcast 等纯原语着色器
    [[nodiscard]] static Result<VulkanPipeline> create_generic(
        VkDevice device, std::span<const uint32_t> spirv_code,
        uint32_t num_bindings, uint32_t push_constant_size)
    {
        VulkanPipeline pl;
        pl.device_ = device;

        // 创建 shader module
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv_code.size_bytes();
        module_info.pCode = spirv_code.data();

        auto r = detail::vk_check(
            vkCreateShaderModule(device, &module_info, nullptr, &pl.shader_module_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 descriptor set layout（num_bindings 个 storage buffer）
        std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
        for (uint32_t i = 0; i < num_bindings; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = num_bindings;
        layout_info.pBindings = bindings.data();

        r = detail::vk_check(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 pipeline layout（push constants）
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = push_constant_size;

        VkPipelineLayoutCreateInfo pl_layout_info{};
        pl_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_layout_info.setLayoutCount = 1;
        pl_layout_info.pSetLayouts = &pl.descriptor_layout_;
        pl_layout_info.pushConstantRangeCount = (push_constant_size > 0) ? 1u : 0u;
        pl_layout_info.pPushConstantRanges = &push_range;

        r = detail::vk_check(
            vkCreatePipelineLayout(device, &pl_layout_info, nullptr, &pl.pipeline_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 compute pipeline
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = pl.shader_module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pl.pipeline_layout_;

        r = detail::vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pl.pipeline_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        return pl;
    }

    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] VkDescriptorSetLayout descriptor_layout() const noexcept { return descriptor_layout_; }
};

// 前向声明
class Matrix;

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

    ~GpuBuffer()
    {
        if (buffer_ != VK_NULL_HANDLE)
            vkDestroyBuffer(device_, buffer_, nullptr);
        if (pool_ && alloc_.valid())
            pool_->free(alloc_);
    }

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

    // 创建 Device Local 缓冲区
    [[nodiscard]] static Result<GpuBuffer> create_device_local(
        VkDevice device, MemoryPool& pool,
        std::size_t elem_count, VkBufferUsageFlags usage)
    {
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = elem_count * sizeof(float);
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

    [[nodiscard]] VkBuffer impl() const noexcept { return buffer_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ != VK_NULL_HANDLE; }
};

// ══════════════════════════════════════════════════════════════════════════
// GpuTensor — GPU 矩阵抽象
// ══════════════════════════════════════════════════════════════════════════

class GpuTensor
{
private:
    std::shared_ptr<GpuBuffer> buffer_;
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;

public:
    GpuTensor() = default;

    GpuTensor(std::shared_ptr<GpuBuffer> buffer, std::size_t rows, std::size_t cols)
        : buffer_(std::move(buffer)), rows_(rows), cols_(cols) {}

    // 从 CPU Matrix 创建（上传数据）
    [[nodiscard]] static Result<GpuTensor> from_matrix(
        const Matrix& cpu_mat, class GpuBackend& backend);

    // 创建空的 GPU Tensor（用于输出）
    [[nodiscard]] static Result<GpuTensor> create_empty(
        std::size_t rows, std::size_t cols, class GpuBackend& backend);

    // 转换为 CPU Matrix（下载数据）
    [[nodiscard]] Result<Matrix> to_matrix(class GpuBackend& backend) const;

    // 访问器
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool valid() const noexcept { return buffer_ && buffer_->valid(); }
    [[nodiscard]] const GpuBuffer& buffer() const noexcept { return *buffer_; }

    // 零拷贝 reshape：共享底层 buffer，只改变形状元数据
    [[nodiscard]] GpuTensor with_shape(std::size_t new_rows, std::size_t new_cols) const
    {
        return GpuTensor(buffer_, new_rows, new_cols);
    }
};

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

    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<StagingRing> staging_ring_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool dispatch_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool gpu_tensor_pool_ = VK_NULL_HANDLE;

    // ── Command Buffer Batching：将所有 GPU 操作录制到同一个 command buffer ──
    // 消除 per-layer fence wait，只在 end_batch() 时提交一次、等一次
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;
    VkFence batch_fence_ = VK_NULL_HANDLE;
    bool batch_mode_ = false;
    std::vector<VkDescriptorSet> batch_desc_sets_;  // batch 期间分配的描述符集

    std::mutex init_mutex_;
    std::mutex queue_mutex_;
    bool initialized_ = false;

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

public:
    ~GpuBackend()
    {
        if (device_.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_.device());

            staging_ring_.reset();

            if (gpu_tensor_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_.device(), gpu_tensor_pool_, nullptr);
            if (dispatch_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_.device(), dispatch_pool_, nullptr);
            if (command_pool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_.device(), command_pool_, nullptr);
        }

        memory_pool_.reset();
    }

    // 禁止拷贝和移动
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;
    GpuBackend(GpuBackend&&) = delete;
    GpuBackend& operator=(GpuBackend&&) = delete;

    [[nodiscard]] static GpuBackend& instance()
    {
        static GpuBackend backend;
        return backend;
    }

    // 初始化
    [[nodiscard]] Result<void> initialize()
    {
        std::lock_guard lock(init_mutex_);
        if (initialized_)
            return {};

        // 1. 检查 SPIR-V 是否可用
        const auto& spirv = get_matmul_spirv();
        if (spirv.empty())
            return std::unexpected(Error{"matmul SPIR-V bytecode not embedded"});

        // 2. 初始化 Vulkan 设备
        auto dev_r = device_.initialize();
        if (!dev_r)
            return dev_r;

        // 3. 创建 matmul pipeline
        auto pl_r = VulkanPipeline::create_matmul(device_.device(), spirv);
        if (!pl_r)
            return std::unexpected(pl_r.error());
        matmul_pipeline_ = std::move(*pl_r);

        // 4. 创建 memory pool
        memory_pool_ = std::make_unique<MemoryPool>(
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
        staging_ring_ = std::make_unique<StagingRing>();
        auto st_r = staging_ring_->initialize(
            device_.device(), device_.physical_device(), command_pool_, *memory_pool_);
        if (!st_r)
            return st_r;

        // 7. 创建 descriptor pool for staging path
        constexpr std::size_t MAX_CACHED_DISPATCHES = 256;
        constexpr std::size_t DESCRIPTORS_PER_DISPATCH = 3;
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
            vkCreateDescriptorPool(device_.device(), &dispatch_pool_info, nullptr, &dispatch_pool_),
            __FILE__, __LINE__);
        if (!r)
            return r;

        // 8. 创建 descriptor pool for GPU-resident path
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

        // 9. 创建 tiled matmul pipeline（可选）
        const auto& tiled_spirv = get_matmul_tiled_spirv();
        if (!tiled_spirv.empty())
        {
            auto tp_r = VulkanPipeline::create_matmul(device_.device(), tiled_spirv);
            if (tp_r)
                matmul_tiled_pipeline_ = std::move(*tp_r);
        }

        // 9b. 创建 batched matmul pipeline（3 bindings, 5*4=20B push constants）
        // 与 matmul 相同的 descriptor/push-constant 布局，可复用 create_matmul
        const auto& batched_spirv = get_batched_matmul_spirv();
        if (!batched_spirv.empty())
        {
            auto bp_r = VulkanPipeline::create_matmul(device_.device(), batched_spirv);
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

    [[nodiscard]] VulkanDevice& device() noexcept { return device_; }
    [[nodiscard]] MemoryPool& memory_pool() noexcept { return *memory_pool_; }
    [[nodiscard]] StagingRing& staging_ring() noexcept { return *staging_ring_; }
    [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
    [[nodiscard]] VkDescriptorPool gpu_tensor_pool() const noexcept { return gpu_tensor_pool_; }

    // ══════════════════════════════════════════════════════════════════
    // Command Buffer Batching API
    // ══════════════════════════════════════════════════════════════════
    // 用法：
    //   backend.begin_batch();
    //   // ... 多次 matmul_gpu / elementwise_gpu / layernorm_gpu 调用 ...
    //   backend.end_batch();
    // 效果：所有操作录制到同一个 command buffer，一次提交、一次 fence wait。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] bool in_batch() const noexcept { return batch_mode_; }

    [[nodiscard]] Result<void> begin_batch()
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (batch_mode_)
            return std::unexpected(Error{"Already in batch mode"});

        // 1. 分配 command buffer
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        auto r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &batch_cmd_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 2. 开始录制
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        r = detail::vk_check(vkBeginCommandBuffer(batch_cmd_, &begin_info), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &batch_cmd_);
            batch_cmd_ = VK_NULL_HANDLE;
            return std::unexpected(r.error());
        }

        // 3. 创建 fence
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = detail::vk_check(
            vkCreateFence(device_.device(), &fence_info, nullptr, &batch_fence_),
            __FILE__, __LINE__);
        if (!r)
        {
            vkEndCommandBuffer(batch_cmd_);
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &batch_cmd_);
            batch_cmd_ = VK_NULL_HANDLE;
            return std::unexpected(r.error());
        }

        batch_mode_ = true;
        batch_desc_sets_.clear();
        return {};
    }

    [[nodiscard]] Result<void> end_batch()
    {
        if (!batch_mode_)
            return {};  // 容错：非 batch 模式时 no-op

        // 1. 结束录制
        auto r = detail::vk_check(vkEndCommandBuffer(batch_cmd_), __FILE__, __LINE__);
        if (!r)
        {
            vkDestroyFence(device_.device(), batch_fence_, nullptr);
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &batch_cmd_);
            batch_cmd_ = VK_NULL_HANDLE;
            batch_fence_ = VK_NULL_HANDLE;
            batch_mode_ = false;
            return std::unexpected(r.error());
        }

        // 2. 提交
        vkResetFences(device_.device(), 1, &batch_fence_);
        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &batch_cmd_;

            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, batch_fence_),
                __FILE__, __LINE__);
            if (!r)
            {
                vkDestroyFence(device_.device(), batch_fence_, nullptr);
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &batch_cmd_);
                batch_cmd_ = VK_NULL_HANDLE;
                batch_fence_ = VK_NULL_HANDLE;
                batch_mode_ = false;
                return std::unexpected(r.error());
            }
        }

        // 3. 等待完成
        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &batch_fence_, VK_TRUE, UINT64_MAX),
            __FILE__, __LINE__);

        // 4. 清理
        vkDestroyFence(device_.device(), batch_fence_, nullptr);
        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &batch_cmd_);
        // 释放 batch 期间分配的所有描述符集
        if (!batch_desc_sets_.empty())
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_,
                static_cast<uint32_t>(batch_desc_sets_.size()), batch_desc_sets_.data());
        batch_desc_sets_.clear();
        batch_cmd_ = VK_NULL_HANDLE;
        batch_fence_ = VK_NULL_HANDLE;
        batch_mode_ = false;

        return r;
    }

    // ── 阻塞式上传：CPU → GPU ─────────────────────────────────────────
    [[nodiscard]] Result<void> upload_blocking(
        GpuTensor& dst, std::span<const Scalar> cpu_data)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!dst.valid())
            return std::unexpected(Error{"Invalid destination GpuTensor"});

        const std::size_t elem_count = dst.rows() * dst.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Upload size mismatch"});

        // 1. 获取 staging region
        auto ri = staging_ring_->acquire();

        // 2. 上传数据到 staging
        auto r = staging_ring_->upload(ri, cpu_data, 0);
        if (!r)
            return r;

        // 3. 录制 copy 命令
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
            __FILE__, __LINE__);
        if (!r)
            return r;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r)
            return r;

        VkBufferCopy cp{0, 0, elem_count * sizeof(float)};
        vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl(), 1, &cp);

        r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r)
            return r;

        // 4. 提交并等待
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
            if (!r)
                return r;
        }

        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
            __FILE__, __LINE__);

        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
        return r;
    }

    // ── 阻塞式下载：GPU → CPU ─────────────────────────────────────────
    [[nodiscard]] Result<void> download_blocking(
        const GpuTensor& src, std::span<Scalar> cpu_data)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!src.valid())
            return std::unexpected(Error{"Invalid source GpuTensor"});

        const std::size_t elem_count = src.rows() * src.cols();
        if (cpu_data.size() != elem_count)
            return std::unexpected(Error{"Download size mismatch"});

        // 1. 获取 staging region
        auto ri = staging_ring_->acquire();

        // 2. 录制 copy 命令
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        auto r = detail::vk_check(
            vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
            __FILE__, __LINE__);
        if (!r)
            return r;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
        if (!r)
            return r;

        VkBufferCopy cp{0, 0, elem_count * sizeof(float)};
        vkCmdCopyBuffer(cmd, src.buffer().impl(), staging_ring_->buffer(ri), 1, &cp);

        r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
        if (!r)
            return r;

        // 3. 提交并等待
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
            if (!r)
                return r;
        }

        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
            __FILE__, __LINE__);
        if (!r)
            return r;

        // 4. 下载数据
        r = staging_ring_->download(ri, cpu_data, 0);

        vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
        return r;
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

        // 3. 分配描述符集
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = pipeline.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);

        // 4. 写入描述符集
        VkDescriptorBufferInfo buf_infos[3]{
            {A.buffer().impl(), 0, VK_WHOLE_SIZE},
            {B.buffer().impl(), 0, VK_WHOLE_SIZE},
            {C.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        // 5. 获取 command buffer（batch 或独立）
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool owns_cmd = false;
        VkFence fence = VK_NULL_HANDLE;

        if (batch_mode_)
        {
            cmd = batch_cmd_;
        }
        else
        {
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r)
            {
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
            owns_cmd = true;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
        }

        // 6. 录制命令（barrier + dispatch + barrier）
        // 管线屏障：确保输入数据就绪
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
        input_barriers[0].buffer = A.buffer().impl();
        input_barriers[1].buffer = B.buffer().impl();

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 2, input_barriers, 0, nullptr);

        // 绑定 pipeline 和描述符集
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // 推送常量（M, N, K, transA, transB）
        const uint32_t push_data[5] = {M, N, K, transA, transB};
        vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // Dispatch
        const uint32_t wg_x = (N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        const uint32_t wg_y = (M + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        vkCmdDispatch(cmd, wg_x, wg_y, 1);

        // 输出屏障
        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = C.buffer().impl();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        // 7. 独立模式：结束录制 → 提交 → 等待 → 清理
        if (owns_cmd)
        {
            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            r = detail::vk_check(
                vkCreateFence(device_.device(), &fence_info, nullptr, &fence),
                __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r)
                {
                    vkDestroyFence(device_.device(), fence, nullptr);
                    vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                    return std::unexpected(r.error());
                }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX),
                __FILE__, __LINE__);

            vkDestroyFence(device_.device(), fence, nullptr);
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);

            if (!r)
                return std::unexpected(r.error());
        }
        // batch 模式：不结束录制，不提交，desc_set 需要在 end_batch 后释放
        // 为简化实现，batch 模式下 desc_set 使用 per-operation 临时分配
        // end_batch() 统一提交和等待

        return C;
    }

    // ── 纯 GPU 批量矩阵乘法 ──────────────────────────────────────────
    // 对每个 batch b 计算 C_b = op(A_b, B_b)，结果垂直堆叠为 (batch*M, N)
    // A: (batch * A_rows_per_batch, A_cols)，B: (batch * B_rows_per_batch, B_cols)
    // batch 步长由 shader 从 M*K / K*N / M*N 推导，无需额外传入
    [[nodiscard]] Result<GpuTensor> batched_matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t batch,
        uint32_t transA = 0, uint32_t transB = 0)
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

        auto& pipeline = batched_matmul_pipeline_;

        // 2. 分配描述符集（与 matmul 相同的 3-binding 布局）
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = pipeline.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[3]{
            {A.buffer().impl(), 0, VK_WHOLE_SIZE},
            {B.buffer().impl(), 0, VK_WHOLE_SIZE},
            {C.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        // 4. 获取 command buffer（batch 共享 / 独立分配）
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool owns_cmd = false;
        VkFence fence = VK_NULL_HANDLE;

        if (batch_mode_)
        {
            cmd = batch_cmd_;
        }
        else
        {
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r)
            {
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
            owns_cmd = true;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }
        }

        // 5. 录制：输入屏障 → bind → push constants → dispatch(Z=batch) → 输出屏障
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
        input_barriers[0].buffer = A.buffer().impl();
        input_barriers[1].buffer = B.buffer().impl();

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 2, input_barriers, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // Push constants: M, N, K, transA, transB（与 matmul 相同布局）
        const uint32_t push_data[5] = {M, N, K, transA, transB};
        vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // Dispatch: Z 维度 = batch（每个 batch 一个 workgroup 层）
        const uint32_t wg_x = (N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        const uint32_t wg_y = (M + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        vkCmdDispatch(cmd, wg_x, wg_y, batch);

        // 输出屏障
        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = C.buffer().impl();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        // 6. 独立模式：结束录制 → 提交 → 等待 → 清理
        if (owns_cmd)
        {
            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            r = detail::vk_check(
                vkCreateFence(device_.device(), &fence_info, nullptr, &fence),
                __FILE__, __LINE__);
            if (!r)
            {
                vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                return std::unexpected(r.error());
            }

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r)
                {
                    vkDestroyFence(device_.device(), fence, nullptr);
                    vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
                    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
                    return std::unexpected(r.error());
                }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX),
                __FILE__, __LINE__);

            vkDestroyFence(device_.device(), fence, nullptr);
            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);

            if (!r)
                return std::unexpected(r.error());
        }

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
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);
        return desc_set;
    }

    // ── 辅助：获取 command buffer（batch 或独立）──────────────────────
    // 返回 (cmd, owns_cmd)。owns_cmd=true 时调用方需 submit_and_wait。
    [[nodiscard]] Result<std::pair<VkCommandBuffer, bool>> acquire_cmd()
    {
        if (batch_mode_)
            return std::make_pair(batch_cmd_, false);

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

        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &cmd;
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

        r = detail::vk_check(
            vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX),
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
    // elementwise_v2_gpu — 逐元素原语（unary/binary/select）
    //
    // Push Constants (32 bytes):
    //   count, mode, op, cmp_op, flags, scalar_b, scalar_then, scalar_else
    // Bindings: A(0), B(1), C(2), OUT(3)
    //
    // 对未使用的 binding（如 unary 模式的 B/C），绑定 A 的 buffer（无害占位）。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> elementwise_v2_gpu(
        const GpuTensor& A, const GpuTensor* B, const GpuTensor* C,
        uint32_t count, uint32_t mode, uint32_t op, uint32_t cmp_op,
        uint32_t flags, float scalar_b, float scalar_then, float scalar_else)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_elementwise_v2_pipeline())
            return std::unexpected(Error{"elementwise_v2 pipeline not available"});

        // 1. 分配输出 Tensor
        auto output_res = GpuTensor::create_empty(A.rows(), A.cols(), *this);
        if (!output_res) return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

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

        // 每线程处理一行/一列，dispatch 足够线程
        const uint32_t max_dim = (mode == 0) ? rows : cols;
        const uint32_t wg_count = (max_dim + 255) / 256;
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
    // broadcast_gpu — 行/列广播原语
    //
    // Push Constants (16 bytes): rows, cols, mode, op
    // Bindings: A(0), Vec(1), Out(2)
    // mode: 0=row_broadcast (vec indexed by row), 1=col_broadcast (vec indexed by col)
    // op: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> broadcast_gpu(
        const GpuTensor& A, const GpuTensor& vec,
        uint32_t mode, uint32_t op)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_broadcast_pipeline())
            return std::unexpected(Error{"broadcast pipeline not available"});

        const uint32_t rows = static_cast<uint32_t>(A.rows());
        const uint32_t cols = static_cast<uint32_t>(A.cols());

        // 1. 分配输出 Tensor
        auto output_res = GpuTensor::create_empty(A.rows(), A.cols(), *this);
        if (!output_res) return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

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
    // fill_zero_gpu — 清零 GPU buffer（使用 vkCmdFillBuffer）
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> fill_zero_gpu(GpuTensor& tensor)
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
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        auto cmd_r = acquire_cmd();
        if (!cmd_r) return std::unexpected(cmd_r.error());
        auto [cmd, owns_cmd] = *cmd_r;

        VkBufferCopy cp{0, 0, size};
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
    [[nodiscard]] Result<GpuTensor> clone_gpu(const GpuTensor& src)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        // 1. 分配同形状的新 buffer
        auto dst_res = GpuTensor::create_empty(src.rows(), src.cols(), *this);
        if (!dst_res) return std::unexpected(dst_res.error());
        GpuTensor dst = std::move(*dst_res);

        // 2. GPU 内拷贝
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            src.rows() * src.cols() * sizeof(float));
        auto r = copy_buffer_gpu(src.buffer().impl(), dst.buffer().impl(), size);
        if (!r) return std::unexpected(r.error());

        return dst;
    }

    // ══════════════════════════════════════════════════════════════════
    // slice_rows_gpu — 行切片（GPU 内拷贝连续行区间，无 PCIe 传输）
    // 返回 (count, cols) 的新 GpuTensor，内容为 src 行 [start_row, start_row + count)
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<GpuTensor> slice_rows_gpu(
        const GpuTensor& src, std::size_t start_row, std::size_t count)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        const std::size_t cols = src.cols();
        if (start_row + count > src.rows())
            return std::unexpected(Error{"slice_rows_gpu: range out of bounds"});

        // 分配目标 buffer
        auto dst_res = GpuTensor::create_empty(count, cols, *this);
        if (!dst_res) return std::unexpected(dst_res.error());
        GpuTensor dst = std::move(*dst_res);

        // 行区间在行主序下连续：[start_row * cols, (start_row + count) * cols)
        const VkDeviceSize src_offset = static_cast<VkDeviceSize>(
            start_row * cols * sizeof(float));
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            count * cols * sizeof(float));

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
    [[nodiscard]] Result<void> insert_rows_gpu(
        GpuTensor& dst, std::size_t dst_start_row, const GpuTensor& src)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        const std::size_t cols = dst.cols();
        if (src.cols() != cols)
            return std::unexpected(Error{"insert_rows_gpu: column count mismatch"});
        if (dst_start_row + src.rows() > dst.rows())
            return std::unexpected(Error{"insert_rows_gpu: range out of bounds"});

        const VkDeviceSize dst_offset = static_cast<VkDeviceSize>(
            dst_start_row * cols * sizeof(float));
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            src.rows() * cols * sizeof(float));

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

        // 1. 创建临时 GPU buffers
        auto a_buf = GpuBuffer::create_device_local(
            device_.device(), *memory_pool_, M * K,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!a_buf)
            return std::unexpected(a_buf.error());

        auto b_buf = GpuBuffer::create_device_local(
            device_.device(), *memory_pool_, K * N,
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
};

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // NN_VK_BACKEND_HPP
