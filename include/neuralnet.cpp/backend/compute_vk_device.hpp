#pragma once

// ── compute_vk_device.hpp — Vulkan 设备与 Pipeline 辅助（RAII） ─────────
//
// 从 compute_vk_backend.hpp 拆出：detail::vk_check/convert 辅助 + VulkanDevice
// + VulkanPipeline。GpuBackend / GpuBuffer / GpuTensor 保留在 compute_vk_backend.hpp。
// ─────────────────────────────────────────────────────────────────────────

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "../core_errors.hpp"
#include "../core_config.hpp"

namespace nn
{
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
} // namespace nn

