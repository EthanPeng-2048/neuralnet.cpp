// ── compute_vk_pipeline_specs.hpp ───────────────────────────────────────────
// GpuBackend::pipeline_specs() 的定义：**一张表**描述全部原语 pipeline 的
// SPIR-V、绑定数、push constant 大小与设备能力要求。
//
// 这张表取代了旧 initialize() 里 40 段"取字节码 → 判空 → 创建 → 移动"的
// 样板，也是"加一个 pipeline 变体要改哪几处"的答案：**这里加一行**，
// 再到 compute_vk_pipeline_catalog.hpp 的 PipelineId 加一个名字。
//
// 本文件由 compute_vk_backend.hpp 在 GpuBackend 定义之后包含（需要访问私有
// 的 get_*_spirv()），不是独立头。
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

namespace nn
{

inline std::span<const PipelineSpec> GpuBackend::pipeline_specs()
{
    using R = PipelineRequirement;
    static constexpr PipelineSpec kSpecs[] = {
        // ── 矩阵乘法族 ─────────────────────────────────────────────────
        {PipelineId::matmul, "matmul",
         get_matmul_spirv, 3, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::matmul_tiled, "matmul_tiled",
         get_matmul_tiled_spirv, 3, 5 * sizeof(std::uint32_t), R::always},
        // 细 tile（16×16 输出块，wgrad 网格饥饿；f32 only）
        {PipelineId::matmul_tiled_fine, "matmul_tiled_fine",
         get_matmul_tiled_fine_spirv, 3, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::matmul_gemv, "matmul_gemv",
         get_matmul_gemv_spirv, 3, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::matmul_gemv_f16, "matmul_gemv_f16",
         get_matmul_gemv_f16_spirv, 3, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // 批量 matmul：push 多一个输出缩放 alpha（6 uint）
        {PipelineId::batched_matmul, "batched_matmul",
         get_batched_matmul_spirv, 3, 6 * sizeof(std::uint32_t), R::always},
        {PipelineId::batched_matmul_f16, "batched_matmul_f16",
         get_batched_matmul_f16_spirv, 3, 6 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::matmul_tiled_f16, "matmul_tiled_f16",
         get_matmul_tiled_f16_spirv, 3, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},

        // ── 逐元素 / 归约 ──────────────────────────────────────────────
        // elementwise：A/B/C/OUT 四绑定；8 个 uint 的 push
        {PipelineId::elementwise_v2, "elementwise_v2",
         get_elementwise_v2_spirv, 4, 8 * sizeof(std::uint32_t), R::always},
        {PipelineId::elementwise_v2_f16, "elementwise_v2_f16",
         get_elementwise_v2_f16_spirv, 4, 8 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // reduce：rows/cols/mode/reduce_op/chunk_rows
        {PipelineId::reduce, "reduce",
         get_reduce_spirv, 2, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::reduce_f16, "reduce_f16",
         get_reduce_f16_spirv, 2, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // 分组归约：rows/cols/R/op
        {PipelineId::group_reduce, "group_reduce",
         get_group_reduce_spirv, 2, 4 * sizeof(std::uint32_t), R::always},
        {PipelineId::group_reduce_f16, "group_reduce_f16",
         get_group_reduce_f16_spirv, 2, 4 * sizeof(std::uint32_t), R::needs_16bit_storage},

        // ── 数据搬运 ───────────────────────────────────────────────────
        // rearrange_3d：五个形状参数
        {PipelineId::rearrange_3d, "rearrange_3d",
         get_rearrange_3d_spirv, 2, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::rearrange_3d_f16, "rearrange_3d_f16",
         get_rearrange_3d_f16_spirv, 2, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // transpose：rows/cols
        {PipelineId::transpose, "transpose",
         get_transpose_spirv, 2, 2 * sizeof(std::uint32_t), R::always},
        // transpose_f16 的 shared float16_t 需要 SPIR-V Float16 capability
        {PipelineId::transpose_f16, "transpose_f16",
         get_transpose_f16_spirv, 2, 2 * sizeof(std::uint32_t), R::needs_shader_float16},
        // gather / scatter_add：3 个形状参数
        {PipelineId::gather, "gather",
         get_gather_spirv, 3, 3 * sizeof(std::uint32_t), R::always},
        {PipelineId::gather_f16, "gather_f16",
         get_gather_f16_spirv, 3, 3 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::scatter_add, "scatter_add",
         get_scatter_add_spirv, 3, 3 * sizeof(std::uint32_t), R::always},
        {PipelineId::scatter_add_f16, "scatter_add_f16",
         get_scatter_add_f16_spirv, 3, 3 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // im2col / col2im：9 个卷积窗口参数
        {PipelineId::im2col, "im2col",
         get_im2col_spirv, 2, 9 * sizeof(std::uint32_t), R::always},
        {PipelineId::im2col_f16, "im2col_f16",
         get_im2col_f16_spirv, 2, 9 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::col2im, "col2im",
         get_col2im_spirv, 2, 9 * sizeof(std::uint32_t), R::always},
        {PipelineId::col2im_f16, "col2im_f16",
         get_col2im_f16_spirv, 2, 9 * sizeof(std::uint32_t), R::needs_16bit_storage},

        // ── RLA 扫描原语（手写原语，不进 AOT 融合注册表）────────────────
        // 前缀扫描：8 绑定 / 7 个参数（含 St scratch）
        {PipelineId::scan_prefix_outer, "scan_prefix_outer",
         get_scan_prefix_outer_spirv, 8, 7 * sizeof(std::uint32_t), R::always},
        // P4-2b 窄共享变体（dk ≤ 32；同布局）
        {PipelineId::scan_prefix_outer_d32, "scan_prefix_outer_d32",
         get_scan_prefix_outer_d32_spirv, 8, 7 * sizeof(std::uint32_t), R::always},
        // 后缀扫描：5 绑定 / 6 个参数
        {PipelineId::scan_suffix_outer, "scan_suffix_outer",
         get_scan_suffix_outer_spirv, 5, 6 * sizeof(std::uint32_t), R::always},
        // 通用 d_k（>64）：状态驻全局 scratch，多一档绑定
        {PipelineId::scan_prefix_outer_gen, "scan_prefix_outer_gen",
         get_scan_prefix_outer_gen_spirv, 9, 7 * sizeof(std::uint32_t), R::always},
        {PipelineId::scan_suffix_outer_gen, "scan_suffix_outer_gen",
         get_scan_suffix_outer_gen_spirv, 6, 6 * sizeof(std::uint32_t), R::always},
        {PipelineId::outer_col, "outer_col",
         get_outer_col_spirv, 4, 4 * sizeof(std::uint32_t), R::always},
        // f16 扫描变体（输入输出均 half；St scratch 仍 f32）
        {PipelineId::scan_prefix_outer_f16, "scan_prefix_outer_f16",
         get_scan_prefix_outer_f16_spirv, 8, 7 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::scan_prefix_outer_d32_f16, "scan_prefix_outer_d32_f16",
         get_scan_prefix_outer_d32_f16_spirv, 8, 7 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::scan_suffix_outer_f16, "scan_suffix_outer_f16",
         get_scan_suffix_outer_f16_spirv, 5, 6 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::scan_prefix_outer_gen_f16, "scan_prefix_outer_gen_f16",
         get_scan_prefix_outer_gen_f16_spirv, 9, 7 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::scan_suffix_outer_gen_f16, "scan_suffix_outer_gen_f16",
         get_scan_suffix_outer_gen_f16_spirv, 6, 6 * sizeof(std::uint32_t), R::needs_16bit_storage},
        {PipelineId::outer_col_f16, "outer_col_f16",
         get_outer_col_f16_spirv, 4, 4 * sizeof(std::uint32_t), R::needs_16bit_storage},

        // ── 精度转换 / 三值打包 ────────────────────────────────────────
        // cast：{count, kind}
        {PipelineId::cast, "cast",
         get_cast_spirv, 2, 2 * sizeof(std::uint32_t), R::always},
        // trit_pack：{rows, k, wpr}
        {PipelineId::trit_pack, "trit_pack",
         get_trit_pack_spirv, 3, 3 * sizeof(std::uint32_t), R::always},
        // matmul_q：{M, N, K, transA, wpr}
        {PipelineId::matmul_q, "matmul_q",
         get_matmul_q_spirv, 4, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::matmul_q_f16, "matmul_q_f16",
         get_matmul_q_f16_spirv, 4, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},
        // 窄 N（N ≤ 8）打包 GEMV：解码主路径
        {PipelineId::matmul_q_gemv, "matmul_q_gemv",
         get_matmul_q_gemv_spirv, 4, 5 * sizeof(std::uint32_t), R::always},
        {PipelineId::matmul_q_gemv_f16, "matmul_q_gemv_f16",
         get_matmul_q_gemv_f16_spirv, 4, 5 * sizeof(std::uint32_t), R::needs_16bit_storage},
    };
    return kSpecs;
}

} // namespace nn

#endif // NN_HAS_VULKAN
