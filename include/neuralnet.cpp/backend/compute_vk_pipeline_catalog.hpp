// ── compute_vk_pipeline_catalog.hpp ─────────────────────────────────────────
// 原语 pipeline 目录：把"有哪些 pipeline、各自布局、需要什么设备能力"
// 从 GpuBackend 的成员列表与 initialize() 样板里抽出来。
//
// 读者收益：
//   · 加一个原语变体 = 在 GpuBackend::pipeline_specs() 表里加一行；
//   · 查"这个 shader 的 binding/push 布局" = 看同一行；
//   · 查"设备没有 16 位存储时哪些变体不存在" = 看 requirement 列；
//   · 不再有 44 个成员 + 40 个 has_* 转发需要同步维护。
//
// 约束：
//   · 目录只负责**创建/查找**，不参与选路（选路仍在 GpuEngine / 算子内）；
//   · 创建失败保持既有语义：可选 pipeline 缺失 = 空句柄 = 该变体不可用。
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "../core_errors.hpp"
#include "compute_vk_device.hpp"

namespace nn
{

// 原语 pipeline 的身份。新增变体时只在这里加名字，并在
// GpuBackend::pipeline_specs() 里加对应的布局行。
enum class PipelineId : std::uint8_t
{
    // 矩阵乘法族
    matmul,
    matmul_tiled,
    matmul_tiled_fine,
    matmul_gemv,
    matmul_gemv_f16,
    batched_matmul,
    batched_matmul_f16,
    matmul_tiled_f16,

    // 逐元素 / 归约
    elementwise_v2,
    elementwise_v2_f16,
    reduce,
    reduce_f16,
    group_reduce,
    group_reduce_f16,

    // 数据搬运
    rearrange_3d,
    rearrange_3d_f16,
    transpose,
    transpose_f16,
    gather,
    gather_f16,
    scatter_add,
    scatter_add_f16,
    im2col,
    im2col_f16,
    col2im,
    col2im_f16,

    // RLA 扫描原语
    scan_prefix_outer,
    scan_prefix_outer_d32,
    scan_suffix_outer,
    scan_prefix_outer_gen,
    scan_suffix_outer_gen,
    outer_col,
    scan_prefix_outer_f16,
    scan_prefix_outer_d32_f16,
    scan_suffix_outer_f16,
    scan_prefix_outer_gen_f16,
    scan_suffix_outer_gen_f16,
    outer_col_f16,

    // 精度转换 / 三值打包
    cast,
    trit_pack,
    matmul_q,
    matmul_q_f16,
    matmul_q_gemv,
    matmul_q_gemv_f16,

    count
};

inline constexpr std::size_t kPipelineCount =
    static_cast<std::size_t>(PipelineId::count);

// 设备能力要求：不满足时**不创建**该 pipeline（调用方看到空句柄后回退）。
enum class PipelineRequirement : std::uint8_t
{
    always,
    needs_16bit_storage,
    needs_shader_float16,   // 同时隐含 needs_16bit_storage
};

// 一行 = 一个 compute pipeline 的完整布局。
struct PipelineSpec
{
    PipelineId id;
    std::string_view name;                 // 诊断/文档用，与 shader 文件同名
    const std::vector<std::uint32_t>& (*spirv)();  // SPIR-V getter（RV-08 注册表）
    std::uint32_t bindings;                // storage buffer 绑定数
    std::uint32_t push_constant_bytes;     // push constant 范围
    PipelineRequirement requirement;
};

class PipelineCatalog
{
public:
    // 按表创建。语义与旧 initialize() 一致：
    //   · SPIR-V 未嵌入（纯 CPU 构建）→ 跳过；
    //   · 设备不满足 requirement → 跳过；
    //   · 创建失败 → 跳过（该变体不可用，由上层回退）。
    void create_all(VkDevice device, const VulkanDevice& caps,
                    std::span<const PipelineSpec> specs)
    {
        for (const auto& spec : specs)
        {
            if (!supports(caps, spec.requirement))
                continue;
            const auto& code = spec.spirv();
            if (code.empty())
                continue;
            auto created = VulkanPipeline::create_generic(
                device, code, spec.bindings, spec.push_constant_bytes);
            if (created)
                entries_[index(spec.id)] = std::move(*created);
        }
    }

    [[nodiscard]] bool has(PipelineId id) const noexcept
    {
        return entries_[index(id)].handle() != VK_NULL_HANDLE;
    }

    // 调用方契约：先 has(id) 再 get(id)。缺失时返回空句柄的 pipeline，
    // Vulkan 校验层会报"绑定空 pipeline"，不会静默用错变体。
    [[nodiscard]] const VulkanPipeline& get(PipelineId id) const noexcept
    {
        return entries_[index(id)];
    }

private:
    static constexpr std::size_t index(PipelineId id) noexcept
    {
        return static_cast<std::size_t>(id);
    }

    static bool supports(const VulkanDevice& caps, PipelineRequirement req) noexcept
    {
        switch (req)
        {
        case PipelineRequirement::always:
            return true;
        case PipelineRequirement::needs_16bit_storage:
            return caps.has_16bit_storage();
        case PipelineRequirement::needs_shader_float16:
            return caps.has_16bit_storage() && caps.has_shader_float16();
        }
        return false;
    }

    std::array<VulkanPipeline, kPipelineCount> entries_{};
};

} // namespace nn

#endif // NN_HAS_VULKAN
