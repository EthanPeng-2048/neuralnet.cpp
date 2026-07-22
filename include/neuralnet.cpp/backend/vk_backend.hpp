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

#if __has_include("elementwise_spv.hpp")
#include "elementwise_spv.hpp"
#define NN_ELEMENTWISE_SPV_EMBEDDED
#endif

#if __has_include("layernorm_spv.hpp")
#include "layernorm_spv.hpp"
#define NN_LAYERNORM_SPV_EMBEDDED
#endif

#if __has_include("layernorm_backward_spv.hpp")
#include "layernorm_backward_spv.hpp"
#define NN_LAYERNORM_BACKWARD_SPV_EMBEDDED
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
            this->~VulkanPipeline();
            new (this) VulkanPipeline(std::move(o));
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

    // 创建 elementwise pipeline
    [[nodiscard]] static Result<VulkanPipeline> create_elementwise(
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

        // 创建 pipeline layout（push constants: count, op, rows, cols）
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

    // 创建 layernorm pipeline（4 个 storage buffer: input, gamma, beta, output）
    [[nodiscard]] static Result<VulkanPipeline> create_layernorm(
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

        // 创建 descriptor set layout（4 个 storage buffer）
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (int i = 0; i < 4; ++i)
        {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 4;
        layout_info.pBindings = bindings;

        r = detail::vk_check(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 pipeline layout（push constants: rows, cols, eps_bits）
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

    // 创建 layernorm backward pipeline（5 个 storage buffer）
    [[nodiscard]] static Result<VulkanPipeline> create_layernorm_backward(
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

        // 创建 descriptor set layout（5 个 storage buffer）
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (int i = 0; i < 5; ++i)
        {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = 5;
        layout_info.pBindings = bindings;

        r = detail::vk_check(
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &pl.descriptor_layout_),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());

        // 创建 pipeline layout（push constants: rows, cols）
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 2 * sizeof(uint32_t);

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
    MemoryPool* pool_ = nullptr;

public:
    GpuBuffer() = default;

    GpuBuffer(VkDevice device, VkBuffer buffer, const MemoryPool::Allocation& alloc, MemoryPool* pool)
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
        o.pool_ = nullptr;
    }

    GpuBuffer& operator=(GpuBuffer&& o) noexcept
    {
        if (this != &o)
        {
            this->~GpuBuffer();
            new (this) GpuBuffer(std::move(o));
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

        return GpuBuffer(device, buffer, *alloc_r, &pool);
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
    VulkanPipeline elementwise_pipeline_;
    VulkanPipeline layernorm_pipeline_;
    VulkanPipeline layernorm_backward_pipeline_;

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

    [[nodiscard]] static const std::vector<uint32_t>& get_elementwise_spirv()
    {
#ifdef NN_ELEMENTWISE_SPV_EMBEDDED
        return nn_elementwise_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_layernorm_spirv()
    {
#ifdef NN_LAYERNORM_SPV_EMBEDDED
        return nn_layernorm_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

    [[nodiscard]] static const std::vector<uint32_t>& get_layernorm_backward_spirv()
    {
#ifdef NN_LAYERNORM_BACKWARD_SPV_EMBEDDED
        return nn_layernorm_backward_spirv_bytecode();
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
            device_.device(), device_.physical_device(), command_pool_, memory_pool_.get());
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
        // layernorm 需要 4 个描述符/次，其他操作需要 3 个，按最大值计算
        constexpr std::size_t TENSOR_POOL_SETS = 1024;
        constexpr std::size_t TENSOR_POOL_DESCS = TENSOR_POOL_SETS * 4;  // layernorm 需要 4

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

        // 10. 创建 elementwise pipeline（可选）
        const auto& elem_spirv = get_elementwise_spirv();
        if (!elem_spirv.empty())
        {
            auto ep_r = VulkanPipeline::create_elementwise(device_.device(), elem_spirv);
            if (ep_r)
                elementwise_pipeline_ = std::move(*ep_r);
        }

        // 11. 创建 layernorm pipeline（可选）
        const auto& ln_spirv = get_layernorm_spirv();
        if (!ln_spirv.empty())
        {
            auto lp_r = VulkanPipeline::create_layernorm(device_.device(), ln_spirv);
            if (lp_r)
                layernorm_pipeline_ = std::move(*lp_r);
        }

        // 12. 创建 layernorm backward pipeline（可选）
        const auto& lnb_spirv = get_layernorm_backward_spirv();
        if (!lnb_spirv.empty())
        {
            auto lnb_r = VulkanPipeline::create_layernorm_backward(device_.device(), lnb_spirv);
            if (lnb_r)
                layernorm_backward_pipeline_ = std::move(*lnb_r);
        }

        initialized_ = true;
        return {};
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] bool gpu_available() const noexcept { return initialized_; }
    [[nodiscard]] bool has_tiled_pipeline() const noexcept { return matmul_tiled_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_elementwise_pipeline() const noexcept { return elementwise_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_layernorm_pipeline() const noexcept { return layernorm_pipeline_.handle() != VK_NULL_HANDLE; }
    [[nodiscard]] bool has_layernorm_backward_pipeline() const noexcept { return layernorm_backward_pipeline_.handle() != VK_NULL_HANDLE; }

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
            return std::unexpected(Error{"Not in batch mode"});

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

    // ── 纯 GPU 逐元素运算（阻塞等待完成）──────────────────────────────
    // op: 0=ReLU, 1=QuickGeLU, 2=BiasAdd
    [[nodiscard]] Result<GpuTensor> elementwise_gpu(
        const GpuTensor& primary, const GpuTensor* secondary, uint32_t op,
        uint32_t rows = 0, uint32_t cols = 0)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_elementwise_pipeline())
            return std::unexpected(Error{"Elementwise pipeline not available"});

        const auto elem_count = static_cast<uint32_t>(primary.rows() * primary.cols());

        // 1. 分配输出 Tensor
        auto output_res = GpuTensor::create_empty(primary.rows(), primary.cols(), *this);
        if (!output_res)
            return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

        // 2. 分配描述符集
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = elementwise_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[3]{
            {primary.buffer().impl(), 0, VK_WHOLE_SIZE},
            {secondary ? secondary->buffer().impl() : primary.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        // 4. 获取 command buffer（batch 或独立）
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool owns_cmd = false;

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

        // 5. 录制命令
        // 管线屏障
        VkBufferMemoryBarrier input_barrier{};
        input_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        input_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        input_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        input_barrier.buffer = primary.buffer().impl();
        input_barrier.offset = 0;
        input_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &input_barrier, 0, nullptr);

        // 绑定 pipeline 和描述符集
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, elementwise_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            elementwise_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        // 推送常量
        const uint32_t push_data[4] = {elem_count, op, rows, cols};
        vkCmdPushConstants(cmd, elementwise_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // Dispatch
        const uint32_t wg_count = (elem_count + 255) / 256;
        vkCmdDispatch(cmd, wg_count, 1, 1);

        // 输出屏障
        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = output.buffer().impl();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        // 6. 独立模式：结束 → 提交 → 等待 → 清理
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
            VkFence fence = VK_NULL_HANDLE;
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

        return output;
    }

    // ── 纯 GPU LayerNorm（阻塞等待完成）──────────────────────────────
    // output[f][b] = gamma[f] * (input[f][b] - mean[b]) / sqrt(var[b] + eps) + beta[f]
    [[nodiscard]] Result<GpuTensor> layernorm_gpu(
        const GpuTensor& input, const GpuTensor& gamma, const GpuTensor& beta,
        uint32_t rows, uint32_t cols, float epsilon)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_layernorm_pipeline())
            return std::unexpected(Error{"LayerNorm pipeline not available"});

        // 1. 分配输出 Tensor
        auto output_res = GpuTensor::create_empty(rows, cols, *this);
        if (!output_res)
            return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

        // 2. 分配描述符集（4 个 storage buffer）
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = layernorm_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[4]{
            {input.buffer().impl(), 0, VK_WHOLE_SIZE},
            {gamma.buffer().impl(), 0, VK_WHOLE_SIZE},
            {beta.buffer().impl(), 0, VK_WHOLE_SIZE},
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

        // 4. 获取 command buffer（batch 或独立）
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool owns_cmd = false;

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

        // 5. 录制命令
        VkBufferMemoryBarrier input_barriers[3]{};
        for (int i = 0; i < 3; ++i)
        {
            input_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            input_barriers[i].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            input_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            input_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].offset = 0;
            input_barriers[i].size = VK_WHOLE_SIZE;
        }
        input_barriers[0].buffer = input.buffer().impl();
        input_barriers[1].buffer = gamma.buffer().impl();
        input_barriers[2].buffer = beta.buffer().impl();

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 3, input_barriers, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layernorm_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            layernorm_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        uint32_t eps_bits = 0;
        std::memcpy(&eps_bits, &epsilon, sizeof(float));
        const uint32_t push_data[3] = {rows, cols, eps_bits};
        vkCmdPushConstants(cmd, layernorm_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        vkCmdDispatch(cmd, cols, 1, 1);

        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = output.buffer().impl();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        // 6. 独立模式：结束 → 提交 → 等待 → 清理
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
            VkFence fence = VK_NULL_HANDLE;
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

        return output;
    }

    // ── 纯 GPU LayerNorm Backward（阻塞等待完成）──────────────────────
    // grad_input[f][b] = (grad_out[f][b]*gamma[f] - mean_g[b] - normalized[f][b]*mean_gn[b]) * std_inv[b]
    [[nodiscard]] Result<GpuTensor> layernorm_backward_gpu(
        const GpuTensor& grad_output, const GpuTensor& gamma,
        const GpuTensor& normalized, const GpuTensor& std_inv,
        uint32_t rows, uint32_t cols)
    {
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});
        if (!has_layernorm_backward_pipeline())
            return std::unexpected(Error{"LayerNorm backward pipeline not available"});

        // 1. 分配输出 Tensor
        auto output_res = GpuTensor::create_empty(rows, cols, *this);
        if (!output_res)
            return std::unexpected(output_res.error());
        GpuTensor output = std::move(*output_res);

        // 2. 分配描述符集（5 个 storage buffer）
        VkDescriptorSet desc_set = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = gpu_tensor_pool_;
        desc_alloc.descriptorSetCount = 1;
        auto dl = layernorm_backward_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &dl;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
            __FILE__, __LINE__);
        if (!r)
            return std::unexpected(r.error());
        if (batch_mode_) batch_desc_sets_.push_back(desc_set);

        // 3. 写入描述符集
        VkDescriptorBufferInfo buf_infos[5]{
            {grad_output.buffer().impl(), 0, VK_WHOLE_SIZE},
            {gamma.buffer().impl(), 0, VK_WHOLE_SIZE},
            {normalized.buffer().impl(), 0, VK_WHOLE_SIZE},
            {std_inv.buffer().impl(), 0, VK_WHOLE_SIZE},
            {output.buffer().impl(), 0, VK_WHOLE_SIZE},
        };

        VkWriteDescriptorSet writes[5]{};
        for (int i = 0; i < 5; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(), 5, writes, 0, nullptr);

        // 4. 获取 command buffer（batch 或独立）
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        bool owns_cmd = false;

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

        // 5. 录制命令
        VkBufferMemoryBarrier input_barriers[4]{};
        for (int i = 0; i < 4; ++i)
        {
            input_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            input_barriers[i].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            input_barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            input_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            input_barriers[i].offset = 0;
            input_barriers[i].size = VK_WHOLE_SIZE;
        }
        input_barriers[0].buffer = grad_output.buffer().impl();
        input_barriers[1].buffer = gamma.buffer().impl();
        input_barriers[2].buffer = normalized.buffer().impl();
        input_barriers[3].buffer = std_inv.buffer().impl();

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 4, input_barriers, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layernorm_backward_pipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            layernorm_backward_pipeline_.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);

        const uint32_t push_data[2] = {rows, cols};
        vkCmdPushConstants(cmd, layernorm_backward_pipeline_.pipeline_layout(),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        vkCmdDispatch(cmd, cols, 1, 1);

        VkBufferMemoryBarrier output_barrier{};
        output_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_barrier.buffer = output.buffer().impl();
        output_barrier.offset = 0;
        output_barrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &output_barrier, 0, nullptr);

        // 6. 独立模式：结束 → 提交 → 等待 → 清理
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
            VkFence fence = VK_NULL_HANDLE;
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

        return output;
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
};

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // NN_VK_BACKEND_HPP
