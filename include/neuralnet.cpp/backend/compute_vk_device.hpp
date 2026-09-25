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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
    std::string device_name_;   // 所选物理设备名（诊断用）
    std::string device_selector_;  // 手动指定设备（空 = 自动选择）
    bool timeline_semaphores_ = false;  // 设备支持时间线信号量（见 initialize）
    // 设备支持 SSBO 里存 16 位（float16_t）+ 已启用 storageBuffer16BitAccess：
    // Phase 2 in-kernel f16 的带类型融合 shader 前提。不支持 → 后端跳过带类型
    // 变体，运行时回退"边界 cast 适配层"（正确性不受影响）。
    bool has_16bit_storage_ = false;
    uint32_t subgroup_size_ = 4;  // 计算队列 subgroup 尺寸——matmul_gemv 的
                                  // red[..][64] 容量前提（256/subgroup≤64）；
                                  // VK1.1 查询失败按 4 兜底，见 initialize

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

    // 读取环境变量（MSVC CRT 把 getenv 标记弃用，-Werror 下必须用 _dupenv_s）
    [[nodiscard]] static std::string get_env(const char* name)
    {
#if defined(_MSC_VER)
        char* buf = nullptr;
        std::size_t len = 0;
        _dupenv_s(&buf, &len, name);
        const std::unique_ptr<char, decltype(&std::free)> guard(buf, &std::free);
        return buf != nullptr ? std::string(buf) : std::string{};
#else
        const char* value = std::getenv(name);
        return value != nullptr ? std::string(value) : std::string{};
#endif
    }

    // ── 手动指定计算设备（必须在 initialize() 之前调用）────────────────────
    // selector = 枚举索引（"2"）或设备名子串（"40HX" / "NVIDIA"）；
    // 空字符串 = 自动选择（设备类型 + apiVersion 打分）。
    // 优先级高于环境变量 NN_VULKAN_DEVICE；未命中时 initialize() 返回
    // 列出全部候选设备的错误。
    void set_device_selector(std::string selector)
    {
        device_selector_ = std::move(selector);
    }

    [[nodiscard]] Result<void> initialize()
    {
        if (initialized_)
            return {};

        // 1. 创建 VkInstance
        // 实例版本请求到 1.2：时间线信号量（跨 submit 依赖的正确原语）是
        // 1.2 核心特性，且 1.2 目标环境下 SPIR-V 1.5 才合法。loader 不支持
        // 1.2 时退回收到的最高版本（后续自动降级为"host 等在飞上传"）。
        uint32_t loader_version = VK_API_VERSION_1_0;
        if (auto enum_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
                vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion")))
        {
            uint32_t supported = VK_API_VERSION_1_0;
            if (enum_version(&supported) == VK_SUCCESS)
                loader_version = supported;
        }

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "neuralnet.cpp";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "neuralnet.cpp";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = std::min(loader_version, VK_API_VERSION_1_2);

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

        // ── 物理设备选择：按能力打分取最优，而不是"取第一个独显" ─────────
        // 实测教训（本机枚举顺序）：
        //   [0] AMD Radeon R5 240      独显 / 老专有驱动, api 1.2.170
        //   [1] Microsoft Direct3D12 (AMD R5 240)   ← Mesa Dozen 转译层
        //   [2] NVIDIA CMP 40HX        独显 / 驱动 616.92, api 1.4.351
        // 旧的"第一个 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"会选中 [0]：
        // 该驱动的 maxComputeSharedMemorySize 仅 32768（matmul 分块要 34560，
        // 校验层直接报 VUID-RuntimeSpirv-Workgroup-06530），且对跨 submit
        // 信号量的重复 wait 直接死锁（vkWaitForFences 超时 → "Vulkan error 2"）。
        // 打分 = (设备类型权重, apiVersion)，同分取先枚举者；D3D12 转译层
        // 降权（非原生驱动，且常与原生条目重复枚举同一张卡）。
        // 可用环境变量 NN_VULKAN_DEVICE 强制指定：索引（"2"）或名称子串（"NVIDIA"）。
        // 选择器来源优先级：
        //   显式 API（set_device_selector / GpuBackend::initialize(sel)）
        //   > 环境变量 NN_VULKAN_DEVICE > 自动打分（见下）
        std::string selector = device_selector_;
        if (selector.empty())
            selector = get_env("NN_VULKAN_DEVICE");

        // selector = 枚举索引（"2"）或设备名称子串（"40HX"/"NVIDIA"）
        if (!selector.empty())
        {
            const std::string_view want(selector);
            // 纯数字只按索引匹配：否则 "2" 会命中 "R5 240" 这类名称子串
            const bool want_is_index =
                !want.empty() &&
                want.find_first_not_of("0123456789") == std::string_view::npos;
            for (std::size_t i = 0; i < devices.size(); ++i)
            {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(devices[i], &props);
                const std::string_view name(props.deviceName);
                const bool matched = want_is_index
                    ? (want == std::string_view(std::to_string(i)))
                    : (name.find(want) != std::string_view::npos);
                if (matched)
                {
                    physical_device_ = devices[i];
                    break;
                }
            }
            if (physical_device_ == VK_NULL_HANDLE)
            {
                // 未命中：列出全部候选，避免"猜索引"
                std::string avail;
                for (std::size_t i = 0; i < devices.size(); ++i)
                {
                    VkPhysicalDeviceProperties props;
                    vkGetPhysicalDeviceProperties(devices[i], &props);
                    avail += "\n  [" + std::to_string(i) + "] " + props.deviceName;
                }
                return std::unexpected(Error{
                    "未找到匹配的计算设备 \"" + selector + "\"，可用设备:" + avail});
            }
        }

        if (physical_device_ == VK_NULL_HANDLE)
        {
            int best_rank = -1;
            uint32_t best_api = 0;
            for (const auto& dev : devices)
            {
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev, &props);
                const std::string_view name(props.deviceName);

                int rank = 1;
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                    rank = 3;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
                    rank = 2;
                if (name.rfind("Microsoft Direct3D12", 0) == 0 ||
                    name == "Microsoft Basic Render Driver")
                    rank = 0;

                if (rank > best_rank ||
                    (rank == best_rank && props.apiVersion > best_api))
                {
                    best_rank = rank;
                    best_api = props.apiVersion;
                    physical_device_ = dev;
                }
            }
        }
        // 兜底：无可打分设备时使用第一个设备
        if (physical_device_ == VK_NULL_HANDLE)
            physical_device_ = devices[0];

        // 记录所选设备名（诊断"到底跑在哪块卡上"用）
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_device_, &props);
            device_name_ = props.deviceName;
        }

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
        // 先探测时间线信号量能力（Vulkan 1.2 核心特性）：跨 submit 数据依赖
        // 需要"同一信号量可被多个 submit 等待"的语义，二进制信号量做不到
        // （一次 signal 只能被一个 wait 消费 → 第二个消费者永久阻塞，
        // VUID-vkQueueSubmit-pWaitSemaphores-03238）。不支持的设备退回
        // "host 等在飞上传"的阻塞路径。
        timeline_semaphores_ = false;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physical_device_, &props);
            if (app_info.apiVersion >= VK_API_VERSION_1_2 &&
                props.apiVersion >= VK_API_VERSION_1_2)
            {
                timeline_features.sType =
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
                VkPhysicalDeviceFeatures2 features2{};
                features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                features2.pNext = &timeline_features;
                vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
                timeline_semaphores_ = (timeline_features.timelineSemaphore == VK_TRUE);
            }
            // 逃生阀：某些驱动的 timeline 实现有问题时，用
            // NN_VULKAN_NO_TIMELINE=1 强制走"host 等在飞上传"回退路径
            // （正确性优先，牺牲 P0-1 非阻塞流水线）。
            if (!get_env("NN_VULKAN_NO_TIMELINE").empty())
                timeline_semaphores_ = false;
        }

        // 计算 subgroup 尺寸（VK1.1 核心，查询链式同上方 features2 模式）：
        //   matmul_gemv 的部分和表 red[ROWS][MAX_N][64] 容量前提 = 256 线程
        //   / subgroup ≥ 4 → ≤64 个 subgroup；查询失败按 4 兜底（前提视为
        //   成立），实测 <4 时后端会关掉 GEMV 分派——否则 subgroup ≥64 不
        //   落表、tid0 只和前 64 个 → 静默错值（4.10 同类"只有 GPU 错"）
        if (app_info.apiVersion >= VK_API_VERSION_1_1)
        {
            VkPhysicalDeviceSubgroupProperties sg{};
            sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &sg;
            vkGetPhysicalDeviceProperties2(physical_device_, &props2);
            if (sg.subgroupSize > 0) subgroup_size_ = sg.subgroupSize;
        }

        // 16 位存储（Phase 2 in-kernel f16）：**查询 + 启用** storageBuffer16BitAccess，
        // 使融合 shader 能把 SSBO 声明为 float16_t（算术仍 f32，见 glsl_gen）。
        // 不支持 → has_16bit_storage_ = false → 后端跳过带类型变体（回退边界 cast）。
        VkPhysicalDevice16BitStorageFeatures storage16{};
        storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        if (app_info.apiVersion >= VK_API_VERSION_1_1)
        {
            VkPhysicalDeviceFeatures2 f16q{};
            f16q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            f16q.pNext = &storage16;
            vkGetPhysicalDeviceFeatures2(physical_device_, &f16q);
            has_16bit_storage_ = (storage16.storageBuffer16BitAccess == VK_TRUE);
        }
        // 逃生阀：某些驱动/校验层对 16 位存储支持不佳时强制走边界 cast 路径
        if (!get_env("NN_VULKAN_NO_16BIT_STORAGE").empty())
            has_16bit_storage_ = false;

        float queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        // pNext 链：16 位存储（若支持）→ 时间线信号量（若支持）
        VkBaseInStructure* chain = nullptr;
        storage16.pNext = nullptr;
        if (timeline_semaphores_)
        {
            timeline_features.pNext = nullptr;
            // Vulkan 的 pNext 为 void*（非 const）→ 用非 const 指针链入
            storage16.pNext = reinterpret_cast<VkBaseInStructure*>(&timeline_features);
        }
        if (has_16bit_storage_)
            chain = reinterpret_cast<VkBaseInStructure*>(&storage16);
        else if (timeline_semaphores_)
            chain = reinterpret_cast<VkBaseInStructure*>(&timeline_features);
        device_info.pNext = chain;
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
    // SSBO 16 位存储（float16_t）是否已启用：in-kernel f16 带类型融合 shader 前提
    [[nodiscard]] bool has_16bit_storage() const noexcept { return has_16bit_storage_; }
    // 所选物理设备名（"NVIDIA CMP 40HX" 等；初始化前为空）
    [[nodiscard]] const std::string& device_name() const noexcept { return device_name_; }
    // 计算队列 subgroup 尺寸（初始化前 = 兜底值 4；GEMV 分派门禁用）
    [[nodiscard]] uint32_t subgroup_size() const noexcept { return subgroup_size_; }
    // 已启用时间线信号量（决定跨 submit 依赖走信号量还是 host 等 fence）
    [[nodiscard]] bool has_timeline_semaphores() const noexcept
    {
        return timeline_semaphores_;
    }
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

