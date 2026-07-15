#ifndef VK_BACKEND_HPP
#define VK_BACKEND_HPP

// ── Vulkan Compute Backend ─────────────────────────────────────────────────
// 为 neuralnet.cpp 提供 GPU 加速的矩阵乘法。
//
// 设计原则：
//   - 公共接口零裸指针（使用 std::span 传递数据）
//   - 所有 Vulkan 句柄由 RAII 守卫管理（自动释放）
//   - GPU 使用 float32 获得最大硬件兼容性，CPU double 自动转换
//   - 单例模式：GpuBackend::instance() 获取全局后端
//
// 依赖：
//   - Vulkan SDK（运行时）
//   - Shader 使用 float32，无需任何可选设备特性
//   - CMake 构建时编译 SPIR-V 着色器
// ─────────────────────────────────────────────────────────────────────────

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>

#include "nn_config.hpp"

// 尝试包含 CMake 生成的 SPIR-V 嵌入头文件
#if __has_include("matmul_spv.hpp")
#include "matmul_spv.hpp"
#define NN_MATMUL_SPV_EMBEDDED
#endif

namespace nn
{

// ── 内部错误检查辅助 ─────────────────────────────────────────────────────
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
} // namespace detail

// ── MappedMemoryGuard ─────────────────────────────────────────────────────
// RAII 内存映射守卫：构造时 vkMapMemory，析构时自动 vkUnmapMemory。
// 只暴露 std::span<std::byte>，不暴露 void*。
// 提供 double↔float 安全转换的上传/下载方法。
// ─────────────────────────────────────────────────────────────────────────
class MappedMemoryGuard
{
private:
    VkDevice device_;
    VkDeviceMemory memory_;
    void* ptr_;
    std::size_t byte_size_;

public:
    MappedMemoryGuard(VkDevice device, VkDeviceMemory mem, std::size_t size)
        : device_(device), memory_(mem), ptr_(nullptr), byte_size_(size)
    {
        if (vkMapMemory(device_, memory_, 0, byte_size_, 0, &ptr_) != VK_SUCCESS)
            ptr_ = nullptr;
    }

    ~MappedMemoryGuard()
    {
        if (ptr_ && device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE)
            vkUnmapMemory(device_, memory_);
    }

    MappedMemoryGuard(const MappedMemoryGuard&) = delete;
    MappedMemoryGuard& operator=(const MappedMemoryGuard&) = delete;

    MappedMemoryGuard(MappedMemoryGuard&& other) noexcept
        : device_(other.device_), memory_(other.memory_),
          ptr_(other.ptr_), byte_size_(other.byte_size_)
    {
        other.ptr_ = nullptr;
        other.device_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
    }

    MappedMemoryGuard& operator=(MappedMemoryGuard&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr_ && device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE)
                vkUnmapMemory(device_, memory_);
            device_ = other.device_;
            memory_ = other.memory_;
            ptr_ = other.ptr_;
            byte_size_ = other.byte_size_;
            other.ptr_ = nullptr;
            other.device_ = VK_NULL_HANDLE;
            other.memory_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    // ── 类型安全的数据传输（double ↔ float 自动转换）────────────────────
    // 上传：CPU double → GPU float
    void upload(std::span<const double> data) noexcept
    {
        if (!ptr_) return;
        auto* dst = static_cast<float*>(ptr_);
        for (std::size_t i = 0; i < data.size(); ++i)
            dst[i] = static_cast<float>(data[i]);
    }

    // 下载：GPU float → CPU double
    void download(std::span<double> dst) const noexcept
    {
        if (!ptr_) return;
        const auto* src = static_cast<const float*>(ptr_);
        for (std::size_t i = 0; i < dst.size(); ++i)
            dst[i] = static_cast<double>(src[i]);
    }

    [[nodiscard]] std::span<std::byte> bytes() noexcept
    {
        return {static_cast<std::byte*>(ptr_), byte_size_};
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept
    {
        return {static_cast<const std::byte*>(ptr_), byte_size_};
    }
};

// ── VkBufferWrapper ───────────────────────────────────────────────────────
// RAII 封装：VkBuffer + VkDeviceMemory。
// 构造时创建缓冲区并分配绑定内存，析构时自动释放。
// ─────────────────────────────────────────────────────────────────────────
class VkBufferWrapper
{
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    std::size_t byte_size_ = 0;

    VkBufferWrapper(VkDevice dev, VkBuffer buf, VkDeviceMemory mem, std::size_t size)
        : device_(dev), buffer_(buf), memory_(mem), byte_size_(size) {}

public:
    VkBufferWrapper() = default;

    ~VkBufferWrapper()
    {
        if (device_ != VK_NULL_HANDLE)
        {
            if (buffer_ != VK_NULL_HANDLE)
                vkDestroyBuffer(device_, buffer_, nullptr);
            if (memory_ != VK_NULL_HANDLE)
                vkFreeMemory(device_, memory_, nullptr);
        }
    }

    VkBufferWrapper(const VkBufferWrapper&) = delete;
    VkBufferWrapper& operator=(const VkBufferWrapper&) = delete;

    VkBufferWrapper(VkBufferWrapper&& other) noexcept
        : device_(other.device_), buffer_(other.buffer_),
          memory_(other.memory_), byte_size_(other.byte_size_)
    {
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.device_ = VK_NULL_HANDLE;
        other.byte_size_ = 0;
    }

    VkBufferWrapper& operator=(VkBufferWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (device_ != VK_NULL_HANDLE)
            {
                if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
                if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
            }
            device_ = other.device_;
            buffer_ = other.buffer_;
            memory_ = other.memory_;
            byte_size_ = other.byte_size_;
            other.buffer_ = VK_NULL_HANDLE;
            other.memory_ = VK_NULL_HANDLE;
            other.device_ = VK_NULL_HANDLE;
            other.byte_size_ = 0;
        }
        return *this;
    }

    // ── 工厂方法：host-visible 暂存缓冲区 ─────────────────────────────
    [[nodiscard]] static Result<VkBufferWrapper> create(
        VkDevice device, VkPhysicalDevice physical_device,
        std::size_t element_count, VkBufferUsageFlags usage)
    {
        return create_internal(device, physical_device, element_count, usage, false);
    }

    // ── 工厂方法：device-local GPU 计算缓冲区 ────────────────────────
    [[nodiscard]] static Result<VkBufferWrapper> create_device_local(
        VkDevice device, VkPhysicalDevice physical_device,
        std::size_t element_count, VkBufferUsageFlags usage)
    {
        return create_internal(device, physical_device, element_count, usage, true);
    }

private:
    [[nodiscard]] static Result<VkBufferWrapper> create_internal(
        VkDevice device, VkPhysicalDevice physical_device,
        std::size_t element_count, VkBufferUsageFlags usage, bool prefer_device_local)
    {
        const std::size_t buf_size = element_count * sizeof(float);

        // 1. 创建缓冲区
        VkBufferCreateInfo buf_info{};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = buf_size;
        buf_info.usage = usage;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        auto r = detail::vk_check(vkCreateBuffer(device, &buf_info, nullptr, &buffer),
                                   __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 2. 查询内存需求
        VkMemoryRequirements mem_reqs;
        vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

        // 3. 查找内存类型
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

        std::optional<uint32_t> best_index;       // 最优匹配
        std::optional<uint32_t> fallback_index;   // 降级匹配

        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
        {
            const bool type_match = (mem_reqs.memoryTypeBits & (1u << i)) != 0;
            if (!type_match) continue;

            const auto flags = mem_props.memoryTypes[i].propertyFlags;
            const bool host_visible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            const bool host_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            const bool device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;

            if (prefer_device_local)
            {
                // 优先：DEVICE_LOCAL（GPU 本地显存，计算最快）
                if (device_local && !best_index) { best_index = i; }
                // 降级：任何匹配类型
                if (!fallback_index) { fallback_index = i; }
            }
            else
            {
                // 优先：HOST_VISIBLE + HOST_COHERENT（CPU 可直接读写）
                if (host_visible && host_coherent)
                {
                    if (device_local && !best_index) { best_index = i; break; }
                    if (!best_index) { best_index = i; }
                }
            }
        }

        auto mem_type_index = best_index.value_or(
            fallback_index.value_or(std::numeric_limits<uint32_t>::max()));

        if (mem_type_index == std::numeric_limits<uint32_t>::max())
        {
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(Error{"No suitable GPU memory type found"});
        }

        // 4. 分配内存
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_reqs.size;
        alloc_info.memoryTypeIndex = mem_type_index;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        r = detail::vk_check(vkAllocateMemory(device, &alloc_info, nullptr, &memory),
                              __FILE__, __LINE__);
        if (!r)
        {
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(r.error());
        }

        // 5. 绑定缓冲区和内存
        r = detail::vk_check(vkBindBufferMemory(device, buffer, memory, 0),
                              __FILE__, __LINE__);
        if (!r)
        {
            vkFreeMemory(device, memory, nullptr);
            vkDestroyBuffer(device, buffer, nullptr);
            return std::unexpected(r.error());
        }

        return VkBufferWrapper(device, buffer, memory, buf_size);
    }

public:
    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceMemory memory() const noexcept { return memory_; }
    [[nodiscard]] std::size_t byte_size() const noexcept { return byte_size_; }

    // 创建 RAII 内存映射
    [[nodiscard]] MappedMemoryGuard map() noexcept
    {
        return MappedMemoryGuard(device_, memory_, byte_size_);
    }
};

// ── GpuBuffer ─────────────────────────────────────────────────────────────
// 公共 API 层的 GPU 缓冲区句柄。不暴露底层 Vulkan 类型。
// ─────────────────────────────────────────────────────────────────────────
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
        VkDevice device, VkPhysicalDevice physical_device,
        std::size_t element_count, VkBufferUsageFlags usage)
    {
        auto wrapper_result = VkBufferWrapper::create(
            device, physical_device, element_count, usage);
        if (!wrapper_result) return std::unexpected(wrapper_result.error());

        return GpuBuffer(element_count,
                         std::make_unique<VkBufferWrapper>(std::move(*wrapper_result)));
    }

    [[nodiscard]] static Result<GpuBuffer> create_device_local(
        VkDevice device, VkPhysicalDevice physical_device,
        std::size_t element_count, VkBufferUsageFlags usage)
    {
        auto wrapper_result = VkBufferWrapper::create_device_local(
            device, physical_device, element_count, usage);
        if (!wrapper_result) return std::unexpected(wrapper_result.error());

        return GpuBuffer(element_count,
                         std::make_unique<VkBufferWrapper>(std::move(*wrapper_result)));
    }

    [[nodiscard]] std::size_t element_count() const noexcept { return element_count_; }
    [[nodiscard]] std::size_t byte_size() const noexcept
    {
        return impl_ ? impl_->byte_size() : 0;
    }
    [[nodiscard]] bool empty() const noexcept
    {
        return !impl_ || element_count_ == 0;
    }

    [[nodiscard]] VkBufferWrapper& impl()
    {
        assert(impl_ && "GpuBuffer: accessing null impl");
        return *impl_;
    }

    [[nodiscard]] const VkBufferWrapper& impl() const
    {
        assert(impl_ && "GpuBuffer: accessing null impl");
        return *impl_;
    }
};

// ── VulkanDevice ──────────────────────────────────────────────────────────
// 管理 VkInstance、物理设备选择、逻辑设备和计算队列。
// 优先选择独立显卡，fallback 到集成显卡。
// ─────────────────────────────────────────────────────────────────────────
class VulkanDevice
{
private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t queue_family_index_ = 0;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    bool valid_ = false;

    // 查找设备上的计算队列族
    [[nodiscard]] static std::optional<uint32_t> find_compute_queue_family(
        VkPhysicalDevice dev)
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

        for (uint32_t i = 0; i < count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                return i;
        }
        return std::nullopt;
    }

public:
    VulkanDevice() = default;

    ~VulkanDevice()
    {
        if (device_ != VK_NULL_HANDLE)
            vkDestroyDevice(device_, nullptr);
        if (instance_ != VK_NULL_HANDLE)
            vkDestroyInstance(instance_, nullptr);
    }

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    [[nodiscard]] Result<void> initialize()
    {
        if (valid_) return {};

        // ── 1. 创建 VkInstance ───────────────────────────────────────
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
        // 检查校验层是否可用；不可用则静默跳过（用户未装 Vulkan SDK 时）
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> avail_layers(layer_count);
        if (layer_count > 0)
            vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());

        bool has_validation = false;
        for (uint32_t li = 0; li < layer_count; ++li)
        {
            if (std::strcmp(avail_layers[li].layerName,
                            "VK_LAYER_KHRONOS_validation") == 0)
            {
                has_validation = true;
                break;
            }
        }

        static const char* validation_layer = "VK_LAYER_KHRONOS_validation";
        if (has_validation)
        {
            inst_info.enabledLayerCount = 1;
            inst_info.ppEnabledLayerNames = &validation_layer;
        }
#endif

        auto r = detail::vk_check(
            vkCreateInstance(&inst_info, nullptr, &instance_), __FILE__, __LINE__);
        if (!r) return r;

        // ── 2. 枚举物理设备 ─────────────────────────────────────────
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
        if (device_count == 0)
            return std::unexpected(Error{"No Vulkan-capable GPU found"});

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

        // 优先选择独显，fallback 到集显
        std::optional<VkPhysicalDevice> best_discrete;
        std::optional<VkPhysicalDevice> best_integrated;
        uint32_t best_discrete_queue = 0;
        uint32_t best_integrated_queue = 0;

        for (auto& dev : devices)
        {
            auto qf = find_compute_queue_family(dev);
            if (!qf) continue;

            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                if (!best_discrete)
                {
                    best_discrete = dev;
                    best_discrete_queue = *qf;
                }
            }
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            {
                if (!best_integrated)
                {
                    best_integrated = dev;
                    best_integrated_queue = *qf;
                }
            }
        }

        if (best_discrete)
        {
            physical_device_ = *best_discrete;
            queue_family_index_ = best_discrete_queue;
        }
        else if (best_integrated)
        {
            physical_device_ = *best_integrated;
            queue_family_index_ = best_integrated_queue;
        }
        else
        {
            return std::unexpected(Error{"No GPU with compute queue found"});
        }

        // ── 3. 创建逻辑设备 ─────────────────────────────────────────
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

        r = detail::vk_check(
            vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
            __FILE__, __LINE__);
        if (!r) return r;

        // ── 4. 获取计算队列句柄 ─────────────────────────────────────
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

// ── VulkanPipeline ────────────────────────────────────────────────────────
// 管理计算着色器模块、描述符集布局和计算管线。
// ─────────────────────────────────────────────────────────────────────────
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
        if (device_ != VK_NULL_HANDLE)
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
    }

    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    VulkanPipeline(VulkanPipeline&& other) noexcept
        : device_(other.device_), shader_module_(other.shader_module_),
          descriptor_layout_(other.descriptor_layout_),
          pipeline_layout_(other.pipeline_layout_), pipeline_(other.pipeline_)
    {
        other.device_ = VK_NULL_HANDLE;
        other.shader_module_ = VK_NULL_HANDLE;
        other.descriptor_layout_ = VK_NULL_HANDLE;
        other.pipeline_layout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }

    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept
    {
        if (this != &other)
        {
            // 释放当前资源
            if (device_ != VK_NULL_HANDLE)
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
            // 转移所有权
            device_ = other.device_;
            shader_module_ = other.shader_module_;
            descriptor_layout_ = other.descriptor_layout_;
            pipeline_layout_ = other.pipeline_layout_;
            pipeline_ = other.pipeline_;
            other.device_ = VK_NULL_HANDLE;
            other.shader_module_ = VK_NULL_HANDLE;
            other.descriptor_layout_ = VK_NULL_HANDLE;
            other.pipeline_layout_ = VK_NULL_HANDLE;
            other.pipeline_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    // ── 创建 matmul 计算管线 ─────────────────────────────────────────
    [[nodiscard]] static Result<VulkanPipeline> create_matmul(
        VkDevice device, std::span<const uint32_t> spirv_code)
    {
        VulkanPipeline pl;
        pl.device_ = device;

        // 1. 创建 Shader Module
        VkShaderModuleCreateInfo module_info{};
        module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_info.codeSize = spirv_code.size_bytes();
        module_info.pCode = spirv_code.data();

        auto r = detail::vk_check(
            vkCreateShaderModule(device, &module_info, nullptr, &pl.shader_module_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 2. 描述符集布局：3 个 Storage Buffer（A, B, C）
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
            vkCreateDescriptorSetLayout(device, &layout_info, nullptr,
                                         &pl.descriptor_layout_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 3. Pipeline Layout（含 push constants: M, N, K）
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
            vkCreatePipelineLayout(device, &pl_layout_info, nullptr,
                                    &pl.pipeline_layout_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 4. 创建计算管线
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = pl.shader_module_;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = pl.pipeline_layout_;

        r = detail::vk_check(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                      nullptr, &pl.pipeline_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        return pl;
    }

    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] VkDescriptorSetLayout descriptor_layout() const noexcept { return descriptor_layout_; }
};

// ── VulkanCommandManager ──────────────────────────────────────────────────
// 管理命令池和描述符池，执行 GPU 计算命令的录制与提交。
// ─────────────────────────────────────────────────────────────────────────
class VulkanCommandManager
{
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

public:
    VulkanCommandManager() = default;

    ~VulkanCommandManager()
    {
        if (device_ != VK_NULL_HANDLE)
        {
            if (descriptor_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            if (command_pool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
    }

    VulkanCommandManager(const VulkanCommandManager&) = delete;
    VulkanCommandManager& operator=(const VulkanCommandManager&) = delete;

    VulkanCommandManager(VulkanCommandManager&& other) noexcept
        : device_(other.device_), command_pool_(other.command_pool_),
          descriptor_pool_(other.descriptor_pool_)
    {
        other.device_ = VK_NULL_HANDLE;
        other.command_pool_ = VK_NULL_HANDLE;
        other.descriptor_pool_ = VK_NULL_HANDLE;
    }

    VulkanCommandManager& operator=(VulkanCommandManager&& other) noexcept
    {
        if (this != &other)
        {
            if (device_ != VK_NULL_HANDLE)
            {
                if (descriptor_pool_ != VK_NULL_HANDLE)
                    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
                if (command_pool_ != VK_NULL_HANDLE)
                    vkDestroyCommandPool(device_, command_pool_, nullptr);
            }
            device_ = other.device_;
            command_pool_ = other.command_pool_;
            descriptor_pool_ = other.descriptor_pool_;
            other.device_ = VK_NULL_HANDLE;
            other.command_pool_ = VK_NULL_HANDLE;
            other.descriptor_pool_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    [[nodiscard]] static Result<VulkanCommandManager> create(
        VkDevice device, uint32_t queue_family_index)
    {
        VulkanCommandManager mgr;
        mgr.device_ = device;

        // 命令池（支持命令缓冲区重置）
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family_index;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        auto r = detail::vk_check(
            vkCreateCommandPool(device, &pool_info, nullptr, &mgr.command_pool_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 描述符池（支持批量分配）
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 3072;  // 3 × 1024

        VkDescriptorPoolCreateInfo desc_pool_info{};
        desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        desc_pool_info.maxSets = 1024;
        desc_pool_info.poolSizeCount = 1;
        desc_pool_info.pPoolSizes = &pool_size;
        desc_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        r = detail::vk_check(
            vkCreateDescriptorPool(device, &desc_pool_info, nullptr,
                                    &mgr.descriptor_pool_),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        return mgr;
    }

    [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }
    [[nodiscard]] VkDescriptorPool descriptor_pool() const noexcept { return descriptor_pool_; }

    // ── 执行一次矩阵乘法 ─────────────────────────────────────────────
    //   C[M×N] = A[M×K] × B[K×N]
    //   所有缓冲区使用 float32 格式
    [[nodiscard]] Result<void> execute_matmul(
        VkPipeline pipeline, VkPipelineLayout pipeline_layout,
        VkDescriptorSetLayout descriptor_layout,
        VkBuffer buf_a, VkBuffer buf_b, VkBuffer buf_c,
        uint32_t M, uint32_t N, uint32_t K,
        VkQueue compute_queue)
    {
        // ── 1. 分配描述符集 ─────────────────────────────────────────
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = descriptor_pool_;
        desc_alloc.descriptorSetCount = 1;
        desc_alloc.pSetLayouts = &descriptor_layout;

        VkDescriptorSet descriptor_set;
        auto r = detail::vk_check(
            vkAllocateDescriptorSets(device_, &desc_alloc, &descriptor_set),
            __FILE__, __LINE__);
        if (!r) return r;

        // ── 2. 更新描述符：绑定 A, B, C 缓冲区 ─────────────────────
        VkDescriptorBufferInfo buf_infos[3]{};
        buf_infos[0] = {buf_a, 0, VK_WHOLE_SIZE};
        buf_infos[1] = {buf_b, 0, VK_WHOLE_SIZE};
        buf_infos[2] = {buf_c, 0, VK_WHOLE_SIZE};

        VkWriteDescriptorSet writes[3]{};
        for (int i = 0; i < 3; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

        // ── 3. 分配命令缓冲区 ───────────────────────────────────────
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        VkCommandBuffer cmd_buffer;
        r = detail::vk_check(
            vkAllocateCommandBuffers(device_, &cmd_alloc, &cmd_buffer),
            __FILE__, __LINE__);
        if (!r) return r;

        // ── 4. 录制命令 ─────────────────────────────────────────────
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        r = detail::vk_check(vkBeginCommandBuffer(cmd_buffer, &begin_info), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
            return r;
        }

        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

        // Push constants: M, N, K
        const uint32_t push_data[3] = {M, N, K};
        vkCmdPushConstants(cmd_buffer, pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        // Dispatch：每个 WorkGroup 处理 16×16 的输出分块
        const uint32_t group_x = (N + 15u) / 16u;
        const uint32_t group_y = (M + 15u) / 16u;
        vkCmdDispatch(cmd_buffer, group_x, group_y, 1);

        r = detail::vk_check(vkEndCommandBuffer(cmd_buffer), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
            return r;
        }

        // ── 5. 提交并同步等待 ───────────────────────────────────────
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence;
        r = detail::vk_check(
            vkCreateFence(device_, &fence_info, nullptr, &fence), __FILE__, __LINE__);
        if (!r)
        {
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
            return r;
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer;

        r = detail::vk_check(vkQueueSubmit(compute_queue, 1, &submit_info, fence), __FILE__, __LINE__);
        if (!r)
        {
            vkDestroyFence(device_, fence, nullptr);
            vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
            return r;
        }

        // 等待计算完成（10 秒超时）
        r = detail::vk_check(
            vkWaitForFences(device_, 1, &fence, VK_TRUE, 10'000'000'000ULL),
            __FILE__, __LINE__);

        // ── 6. 清理临时资源 ─────────────────────────────────────────
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, command_pool_, 1, &cmd_buffer);
        vkFreeDescriptorSets(device_, descriptor_pool_, 1, &descriptor_set);

        return r;
    }
};

// ── MatmulKey: 调度缓存键 ────────────────────────────────────────────────
struct MatmulKey {
    uint32_t M, N, K;
    bool operator==(const MatmulKey& o) const noexcept {
        return M == o.M && N == o.N && K == o.K;
    }
};

struct MatmulKeyHash {
    std::size_t operator()(const MatmulKey& k) const noexcept {
        return (static_cast<uint64_t>(k.M) << 32) ^
               (static_cast<uint64_t>(k.N) << 16) ^
               static_cast<uint64_t>(k.K);
    }
};

// ── CachedDispatch: 预录制的 GPU 调度 ────────────────────────────────────
// 使用 device-local 显存做计算 + host-visible staging 做传输。
// 首次调用时创建全部资源并预录制命令缓冲（含拷贝+屏障+调度），
// 后续相同维度调用直接复用，仅上传 staging → 提交 → 下载 staging。
// ─────────────────────────────────────────────────────────────────────────
struct CachedDispatch {
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // Device-local 计算缓冲区（GPU 本地显存，计算快）
    GpuBuffer buf_a;
    GpuBuffer buf_b;
    GpuBuffer buf_c;
    // Host-visible staging 缓冲区（CPU 可映射，用于上传/下载）
    GpuBuffer staging_a;
    GpuBuffer staging_b;
    GpuBuffer staging_c;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    ~CachedDispatch()
    {
        if (device != VK_NULL_HANDLE)
        {
            if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
            if (cmd_buffer != VK_NULL_HANDLE)
                vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buffer);
            if (descriptor_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device, desc_pool, 1, &descriptor_set);
        }
    }

    CachedDispatch() = default;

    CachedDispatch(CachedDispatch&& o) noexcept
        : cmd_buffer(o.cmd_buffer), descriptor_set(o.descriptor_set),
          fence(o.fence), buf_a(std::move(o.buf_a)), buf_b(std::move(o.buf_b)),
          buf_c(std::move(o.buf_c)),
          staging_a(std::move(o.staging_a)), staging_b(std::move(o.staging_b)),
          staging_c(std::move(o.staging_c)),
          device(o.device), cmd_pool(o.cmd_pool), desc_pool(o.desc_pool)
    {
        o.cmd_buffer = VK_NULL_HANDLE;
        o.descriptor_set = VK_NULL_HANDLE;
        o.fence = VK_NULL_HANDLE;
        o.device = VK_NULL_HANDLE;
    }

    CachedDispatch& operator=(CachedDispatch&& o) noexcept
    {
        if (this != &o)
        {
            this->~CachedDispatch();
            cmd_buffer = o.cmd_buffer; o.cmd_buffer = VK_NULL_HANDLE;
            descriptor_set = o.descriptor_set; o.descriptor_set = VK_NULL_HANDLE;
            fence = o.fence; o.fence = VK_NULL_HANDLE;
            buf_a = std::move(o.buf_a);
            buf_b = std::move(o.buf_b);
            buf_c = std::move(o.buf_c);
            staging_a = std::move(o.staging_a);
            staging_b = std::move(o.staging_b);
            staging_c = std::move(o.staging_c);
            device = o.device; o.device = VK_NULL_HANDLE;
            cmd_pool = o.cmd_pool;
            desc_pool = o.desc_pool;
        }
        return *this;
    }

    CachedDispatch(const CachedDispatch&) = delete;
    CachedDispatch& operator=(const CachedDispatch&) = delete;
};

// ── GpuBackend ────────────────────────────────────────────────────────────
// 单例 GPU 计算后端。管理所有 Vulkan 资源的生命周期。
//
// 使用方法：
//   auto& backend = GpuBackend::instance();
//   auto r = backend.initialize();
//   if (r) {
//       auto mm = backend.matmul_direct(a_data, b_data, c_data, M, N, K);
//       if (mm) { /* 成功 */ }
//   }
// ─────────────────────────────────────────────────────────────────────────
class GpuBackend
{
private:
    VulkanDevice device_;
    VulkanPipeline matmul_pipeline_;
    VulkanCommandManager cmd_manager_;
    std::mutex mutex_;  // 保护 GPU 操作的线程安全
    bool initialized_ = false;

    // ── 调度专用描述符池：与 cmd_manager_ 的临时池隔离，避免争抢 ─
    VkDescriptorPool dispatch_pool_ = VK_NULL_HANDLE;

    // ── 缓冲区缓存：按元素数量索引，避免每次调用重新分配 ───────────
    std::unordered_map<std::size_t, std::vector<GpuBuffer>> buffer_cache_;

    // ── 调度缓存：按 (M,N,K) 维度索引，缓存预录制命令缓冲+描述符+围栏 ─
    static constexpr std::size_t MAX_CACHED_DISPATCHES = 256;
    std::unordered_map<MatmulKey, CachedDispatch, MatmulKeyHash> dispatch_cache_;
    std::list<MatmulKey> dispatch_lru_;  // front=最近使用, back=最久未使用

    // 从缓存获取或创建缓冲区（调用时已持有 mutex_）
    [[nodiscard]] Result<GpuBuffer> acquire_buffer(std::size_t element_count)
    {
        auto it = buffer_cache_.find(element_count);
        if (it != buffer_cache_.end() && !it->second.empty())
        {
            auto buf = std::move(it->second.back());
            it->second.pop_back();
            return buf;
        }
        return GpuBuffer::create(device_.device(), device_.physical_device(),
                                 element_count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    // 归还缓冲区到缓存（调用时已持有 mutex_）
    void release_buffer(GpuBuffer&& buf)
    {
        if (!buf.empty())
            buffer_cache_[buf.element_count()].push_back(std::move(buf));
    }

    // ── 创建预录制调度（调用时已持有 mutex_）────────────────────────
    // 创建 device-local 计算缓冲区 + host-visible staging 缓冲区，
    // 预录制包含拷贝、屏障、计算调度的完整命令缓冲。
    [[nodiscard]] Result<CachedDispatch> create_cached_dispatch(const MatmulKey& key)
    {
        CachedDispatch d;
        d.device = device_.device();
        d.cmd_pool = cmd_manager_.command_pool();
        d.desc_pool = dispatch_pool_;  // 使用调度专用池，不与临时操作争抢

        const auto a_elems = static_cast<std::size_t>(key.M) * key.K;
        const auto b_elems = static_cast<std::size_t>(key.K) * key.N;
        const auto c_elems = static_cast<std::size_t>(key.M) * key.N;

        // 1. 创建 device-local 计算缓冲区（GPU 本地显存）
        auto a_buf = GpuBuffer::create_device_local(d.device, device_.physical_device(),
            a_elems, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!a_buf) return std::unexpected(a_buf.error());
        d.buf_a = std::move(*a_buf);

        auto b_buf = GpuBuffer::create_device_local(d.device, device_.physical_device(),
            b_elems, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!b_buf) return std::unexpected(b_buf.error());
        d.buf_b = std::move(*b_buf);

        auto c_buf = GpuBuffer::create_device_local(d.device, device_.physical_device(),
            c_elems, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!c_buf) return std::unexpected(c_buf.error());
        d.buf_c = std::move(*c_buf);

        // 2. 创建 host-visible staging 缓冲区（CPU 可直接映射）
        auto sa_buf = GpuBuffer::create(d.device, device_.physical_device(),
            a_elems, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!sa_buf) return std::unexpected(sa_buf.error());
        d.staging_a = std::move(*sa_buf);

        auto sb_buf = GpuBuffer::create(d.device, device_.physical_device(),
            b_elems, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!sb_buf) return std::unexpected(sb_buf.error());
        d.staging_b = std::move(*sb_buf);

        auto sc_buf = GpuBuffer::create(d.device, device_.physical_device(),
            c_elems, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!sc_buf) return std::unexpected(sc_buf.error());
        d.staging_c = std::move(*sc_buf);

        // 3. 分配描述符集并绑定到 device-local 计算缓冲区
        VkDescriptorSetAllocateInfo desc_alloc{};
        desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        desc_alloc.descriptorPool = d.desc_pool;
        desc_alloc.descriptorSetCount = 1;
        auto desc_layout = matmul_pipeline_.descriptor_layout();
        desc_alloc.pSetLayouts = &desc_layout;

        auto r = detail::vk_check(
            vkAllocateDescriptorSets(d.device, &desc_alloc, &d.descriptor_set),
            __FILE__, __LINE__);
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

        // 4. 分配并预录制命令缓冲（含 staging→device 拷贝 + 屏障 + 计算）
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = d.cmd_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        r = detail::vk_check(
            vkAllocateCommandBuffers(d.device, &cmd_alloc, &d.cmd_buffer),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        r = detail::vk_check(
            vkBeginCommandBuffer(d.cmd_buffer, &begin_info), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // ── 阶段 1：staging → device 拷贝 ──────────────────────────
        VkBufferCopy copy_a{};
        copy_a.size = a_elems * sizeof(float);
        vkCmdCopyBuffer(d.cmd_buffer, d.staging_a.impl().handle(),
                        d.buf_a.impl().handle(), 1, &copy_a);

        VkBufferCopy copy_b{};
        copy_b.size = b_elems * sizeof(float);
        vkCmdCopyBuffer(d.cmd_buffer, d.staging_b.impl().handle(),
                        d.buf_b.impl().handle(), 1, &copy_b);

        // ── 屏障：TRANSFER → COMPUTE ───────────────────────────────
        VkMemoryBarrier transfer_to_compute{};
        transfer_to_compute.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        transfer_to_compute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transfer_to_compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(d.cmd_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &transfer_to_compute, 0, nullptr, 0, nullptr);

        // ── 阶段 2：矩阵乘法计算 ──────────────────────────────────
        vkCmdBindPipeline(d.cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          matmul_pipeline_.handle());
        vkCmdBindDescriptorSets(d.cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                matmul_pipeline_.pipeline_layout(),
                                0, 1, &d.descriptor_set, 0, nullptr);

        const uint32_t push_data[3] = {key.M, key.N, key.K};
        vkCmdPushConstants(d.cmd_buffer, matmul_pipeline_.pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_data), push_data);

        const uint32_t group_x = (key.N + 15u) / 16u;
        const uint32_t group_y = (key.M + 15u) / 16u;
        vkCmdDispatch(d.cmd_buffer, group_x, group_y, 1);

        // ── 屏障：COMPUTE → TRANSFER ───────────────────────────────
        VkMemoryBarrier compute_to_transfer{};
        compute_to_transfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        compute_to_transfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        compute_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(d.cmd_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &compute_to_transfer, 0, nullptr, 0, nullptr);

        // ── 阶段 3：device → staging 拷贝（下载结果）───────────────
        VkBufferCopy copy_c{};
        copy_c.size = c_elems * sizeof(float);
        vkCmdCopyBuffer(d.cmd_buffer, d.buf_c.impl().handle(),
                        d.staging_c.impl().handle(), 1, &copy_c);

        r = detail::vk_check(vkEndCommandBuffer(d.cmd_buffer), __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        // 5. 创建持久围栏
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        r = detail::vk_check(
            vkCreateFence(d.device, &fence_info, nullptr, &d.fence),
            __FILE__, __LINE__);
        if (!r) return std::unexpected(r.error());

        return d;
    }

    GpuBackend() = default;

    // 获取 matmul SPIR-V 字节码
    [[nodiscard]] static const std::vector<uint32_t>& get_matmul_spirv()
    {
#ifdef NN_MATMUL_SPV_EMBEDDED
        return nn_matmul_spirv_bytecode();
#else
        static const std::vector<uint32_t> empty;
        return empty;
#endif
    }

public:
    // Meyer's singleton
    [[nodiscard]] static GpuBackend& instance()
    {
        static GpuBackend backend;
        return backend;
    }

    ~GpuBackend()
    {
        // 先清空调度缓存（释放其中的描述符集回 dispatch_pool_），再销毁池
        dispatch_lru_.clear();
        dispatch_cache_.clear();
        if (device_.device() != VK_NULL_HANDLE)
        {
            if (dispatch_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_.device(), dispatch_pool_, nullptr);
        }
    }
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;

    // ── 初始化 GPU 后端（惰性，可安全重复调用）──────────────────────
    [[nodiscard]] Result<void> initialize()
    {
        std::lock_guard lock(mutex_);
        if (initialized_) return {};

        const auto& spirv = get_matmul_spirv();
        if (spirv.empty())
            return std::unexpected(Error{
                "matmul SPIR-V bytecode not embedded. "
                "Run CMake build with Vulkan SDK to compile shaders."});

        auto dev_r = device_.initialize();
        if (!dev_r) return dev_r;

        auto pl_r = VulkanPipeline::create_matmul(device_.device(), spirv);
        if (!pl_r) return std::unexpected(pl_r.error());
        matmul_pipeline_ = std::move(*pl_r);

        auto cmd_r = VulkanCommandManager::create(
            device_.device(), device_.queue_family_index());
        if (!cmd_r) return std::unexpected(cmd_r.error());
        cmd_manager_ = std::move(*cmd_r);

        // 调度专用描述符池（大容量，供 CachedDispatch 持久使用）
        VkDescriptorPoolSize dispatch_pool_size{};
        dispatch_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dispatch_pool_size.descriptorCount = 49152;  // 3 × 16384

        VkDescriptorPoolCreateInfo dispatch_pool_info{};
        dispatch_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dispatch_pool_info.maxSets = 16384;
        dispatch_pool_info.poolSizeCount = 1;
        dispatch_pool_info.pPoolSizes = &dispatch_pool_size;
        dispatch_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        auto dp_r = detail::vk_check(
            vkCreateDescriptorPool(device_.device(), &dispatch_pool_info, nullptr,
                                    &dispatch_pool_),
            __FILE__, __LINE__);
        if (!dp_r) return dp_r;

        initialized_ = true;
        return {};
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    // ── 上传 CPU 数据到 GPU 缓冲区 ─────────────────────────────────
    [[nodiscard]] Result<void> upload(std::span<const double> src, GpuBuffer& dst)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (dst.empty()) return std::unexpected(Error{"destination buffer is empty"});
        if (src.size() != dst.element_count())
            return std::unexpected(Error{"upload size mismatch"});

        auto mapped = dst.impl().map();
        if (!mapped.valid())
            return std::unexpected(Error{"failed to map GPU buffer for upload"});

        mapped.upload(src);
        return {};
    }

    // ── 下载 GPU 缓冲区到 CPU ───────────────────────────────────────
    [[nodiscard]] Result<void> download(GpuBuffer& src, std::span<double> dst)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});
        if (src.empty()) return std::unexpected(Error{"source buffer is empty"});
        if (src.element_count() != dst.size())
            return std::unexpected(Error{"download size mismatch"});

        auto mapped = src.impl().map();
        if (!mapped.valid())
            return std::unexpected(Error{"failed to map GPU buffer for download"});

        mapped.download(dst);
        return {};
    }

    // ── GPU 矩阵乘法（底层接口）──────────────────────────────────────
    //   C = A × B，M×K @ K×N → M×N
    [[nodiscard]] Result<void> matmul(
        GpuBuffer& a, GpuBuffer& b, GpuBuffer& c,
        std::size_t M, std::size_t N, std::size_t K)
    {
        if (!initialized_) return std::unexpected(Error{"GPU backend not initialized"});

        std::lock_guard lock(mutex_);
        return cmd_manager_.execute_matmul(
            matmul_pipeline_.handle(),
            matmul_pipeline_.pipeline_layout(),
            matmul_pipeline_.descriptor_layout(),
            a.impl().handle(), b.impl().handle(), c.impl().handle(),
            static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K),
            device_.compute_queue());
    }

    // ── 端到端 GPU 矩阵乘法（高层接口）───────────────────────────────
    //   首次调用：创建 device-local 计算缓冲 + staging 缓冲 + 预录制命令
    //   后续调用：直接复用缓存，仅 上传staging → 提交 → 下载staging
    //   LRU 淘汰：缓存超过 MAX_CACHED_DISPATCHES 时淘汰最久未使用的条目
    [[nodiscard]] Result<void> matmul_direct(
        std::span<const double> a_data,
        std::span<const double> b_data,
        std::span<double> c_data,
        std::size_t M, std::size_t N, std::size_t K)
    {
        std::lock_guard lock(mutex_);
        if (!initialized_)
            return std::unexpected(Error{"GPU backend not initialized"});

        MatmulKey key{static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K)};

        // 查找或创建缓存调度
        auto it = dispatch_cache_.find(key);
        if (it == dispatch_cache_.end())
        {
            // 缓存满时淘汰最久未使用的条目
            if (dispatch_cache_.size() >= MAX_CACHED_DISPATCHES)
            {
                auto oldest = dispatch_lru_.back();
                dispatch_lru_.pop_back();
                dispatch_cache_.erase(oldest);
            }

            auto result = create_cached_dispatch(key);
            if (!result) return std::unexpected(result.error());
            it = dispatch_cache_.emplace(key, std::move(*result)).first;
            dispatch_lru_.push_front(key);
        }
        else
        {
            // 命中：将 key 移到 LRU 最前端
            dispatch_lru_.remove(key);
            dispatch_lru_.push_front(key);
        }

        auto& d = it->second;

        // 上传 A 到 staging（double→float 转换）
        {
            auto mapped = d.staging_a.impl().map();
            if (!mapped.valid())
                return std::unexpected(Error{"failed to map staging A for upload"});
            mapped.upload(a_data);
        }

        // 上传 B 到 staging
        {
            auto mapped = d.staging_b.impl().map();
            if (!mapped.valid())
                return std::unexpected(Error{"failed to map staging B for upload"});
            mapped.upload(b_data);
        }

        // 重置围栏，提交预录制命令缓冲
        // （内部含 staging→device 拷贝 + 屏障 + 计算 + device→staging 拷贝）
        vkResetFences(d.device, 1, &d.fence);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &d.cmd_buffer;

        auto r = detail::vk_check(
            vkQueueSubmit(device_.compute_queue(), 1, &submit_info, d.fence),
            __FILE__, __LINE__);
        if (!r) return r;

        r = detail::vk_check(
            vkWaitForFences(d.device, 1, &d.fence, VK_TRUE, 10'000'000'000ULL),
            __FILE__, __LINE__);
        if (!r) return r;

        // 从 staging 下载结果
        {
            auto mapped = d.staging_c.impl().map();
            if (!mapped.valid())
                return std::unexpected(Error{"failed to map staging C for download"});
            mapped.download(c_data);
        }

        return {};
    }
};

} // namespace nn

#endif // NN_HAS_VULKAN
#endif // VK_BACKEND_HPP
