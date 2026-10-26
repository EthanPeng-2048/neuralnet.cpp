// ── compute_vk_backend.hpp ──────────────────────────────────────────────────────
// Vulkan Compute Backend
//
// 职责：
//   - 管理 Vulkan 实例、设备、队列（RAII）
//   - 管理 MemoryPool 子分配器
//   - 管理 StagingRing 环形缓冲区
//   - 提供 GPU 操作 API（matmul、elementwise）
//
// 同步机制（独立/非 batch 模式）：
//   - 复用 initialize() 预分配的 solo fence + command buffer（submit 前
//     vkResetFences）——fence/cmd 整个后端生命周期只创建一次，host 侧每算子
//     固定开销里没有创建/销毁项
//   - 提交后立即等待 fence，确保 GPU 计算完成
//   - 完成后仅归还 descriptor set（gpu_tensor_pool_ 池化复用）
//
// 双轨制架构：
//   - Staging Path：CPU span → Staging → GPU → 计算 → GPU → Staging → CPU span
//   - GPU-Resident Path：GpuTensor → 计算 → GpuTensor（全程 GPU 显存）
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#ifdef NN_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

#include <neuralnet.cpp/core_errors.hpp>
#include "../core_observer_ptr.hpp"
#include <neuralnet.cpp/core_config.hpp>
#include "../core_profile.hpp"   // 设备侧计时回填（NN_PROFILE=1 门控）
// 本文件直接使用 expr_spec 的符号（run_fused_gpu 签名的 ExprPrecSig、
// EXPR_PREC_DISPATCH_SUFFIX/EXPR_MAX_INPUTS/EXPR_MAX_CONSTS/
// EXPR_MATMUL_BLOCK/EXPR_FOLD_ROWS_PER_WG）——**必须显式 include**：
// include 链 algebra_matrix.hpp → 本文件发生在 expr_spec.hpp 之前，此前
// 只靠 fused_registry.hpp 传递带入 → 注册表缺失（干净构建 / NN_EXPR_SCAN
// 收集器 TU）时编译断裂（clang 与 MSVC 同批报错，实测）。
#include "../expr_spec.hpp"
#include "compute_memory_pool.hpp"
#include "compute_gpu_storage.hpp"
#include "../core_domain.hpp"
#include "compute_vk_pipeline_catalog.hpp"
#include "compute_staging_ring.hpp"

// SPIR-V 嵌入头文件（由 CMake 生成）

// ── 原语 shader 单一注册表（RV-08）：新增一个原语 shader 只改两处 ——
//    ① 文件头加一个 `#if __has_include("<name>_spv.hpp")` 探测块（顺带绑
//       NN_SPV_FN_<name>）；② 在下面 NN_SPV_GETTER_LIST 加一个 token。
//    所有 get_<name>_spirv() 由列表统一展开（见 GpuBackend 内）。
#define NN_SPV_FN_MISSING nn::detail::spirv_missing

#define NN_SPV_GETTER_LIST(X) \
X(matmul) X(matmul_tiled) X(matmul_gemv) X(matmul_gemv_f16)\
X(batched_matmul) X(batched_matmul_f16) X(matmul_tiled_f16)\
X(matmul_tiled_fine) X(rearrange_3d) X(elementwise_v2) X(reduce)\
X(reduce_f16) X(transpose) X(im2col) X(im2col_f16) X(col2im)\
X(col2im_f16) X(group_reduce) X(group_reduce_f16) X(gather)\
X(transpose_f16) X(rearrange_3d_f16) X(gather_f16) X(elementwise_v2_f16)\
X(scatter_add) X(scatter_add_f16) X(scan_prefix_outer)\
X(scan_prefix_outer_f16) X(scan_suffix_outer) X(scan_suffix_outer_f16)\
X(scan_prefix_outer_d32) X(scan_prefix_outer_d32_f16)\
X(scan_prefix_outer_gen) X(scan_prefix_outer_gen_f16)\
X(scan_suffix_outer_gen) X(scan_suffix_outer_gen_f16) X(outer_col)\
X(outer_col_f16) X(cast) X(trit_pack) X(matmul_q) X(matmul_q_f16)\
X(matmul_q_gemv) X(matmul_q_gemv_f16)
#if __has_include("matmul_spv.hpp")
#include "matmul_spv.hpp"
#define NN_SPV_FN_matmul nn_matmul_spirv_bytecode
#else
#define NN_SPV_FN_matmul NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_tiled_spv.hpp")
#include "matmul_tiled_spv.hpp"
#define NN_SPV_FN_matmul_tiled nn_matmul_tiled_spirv_bytecode
#else
#define NN_SPV_FN_matmul_tiled NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_gemv_spv.hpp")
#include "matmul_gemv_spv.hpp"
#define NN_SPV_FN_matmul_gemv nn_matmul_gemv_spirv_bytecode
#else
#define NN_SPV_FN_matmul_gemv NN_SPV_FN_MISSING
#endif

// Phase C2e 小 N GEMV 的 f16 存储变体（f16 直读直写 + f32 累加）
#if __has_include("matmul_gemv_f16_spv.hpp")
#include "matmul_gemv_f16_spv.hpp"
#define NN_SPV_FN_matmul_gemv_f16 nn_matmul_gemv_f16_spirv_bytecode
#else
#define NN_SPV_FN_matmul_gemv_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("batched_matmul_spv.hpp")
#include "batched_matmul_spv.hpp"
#define NN_SPV_FN_batched_matmul nn_batched_matmul_spirv_bytecode
#else
#define NN_SPV_FN_batched_matmul NN_SPV_FN_MISSING
#endif

// 精度变体（Phase 2 in-kernel f16）：同一份 .comp 用 -DNN_SHADER_F16=1 编出的
// 第二份 SPIR-V（f16 缓冲载入 + f32 累加 + f16 写出）。
#if __has_include("batched_matmul_f16_spv.hpp")
#include "batched_matmul_f16_spv.hpp"
#define NN_SPV_FN_batched_matmul_f16 nn_batched_matmul_f16_spirv_bytecode
#else
#define NN_SPV_FN_batched_matmul_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_tiled_f16_spv.hpp")
#include "matmul_tiled_f16_spv.hpp"
#define NN_SPV_FN_matmul_tiled_f16 nn_matmul_tiled_f16_spirv_bytecode
#else
#define NN_SPV_FN_matmul_tiled_f16 NN_SPV_FN_MISSING
#endif

// P0 wgrad 网格饥饿：同一份 matmul_tiled.comp 用 -DNN_MM_FINE=1 编的
// 16×16 输出块变体（f32 only；逐输出元素 K 累加序与 64-tile 逐位一致）。
#if __has_include("matmul_tiled_fine_spv.hpp")
#include "matmul_tiled_fine_spv.hpp"
#define NN_SPV_FN_matmul_tiled_fine nn_matmul_tiled_fine_spirv_bytecode
#else
#define NN_SPV_FN_matmul_tiled_fine NN_SPV_FN_MISSING
#endif

#if __has_include("rearrange_3d_spv.hpp")
#include "rearrange_3d_spv.hpp"
#define NN_SPV_FN_rearrange_3d nn_rearrange_3d_spirv_bytecode
#else
#define NN_SPV_FN_rearrange_3d NN_SPV_FN_MISSING
#endif

#if __has_include("elementwise_v2_spv.hpp")
#include "elementwise_v2_spv.hpp"
#define NN_SPV_FN_elementwise_v2 nn_elementwise_v2_spirv_bytecode
#else
#define NN_SPV_FN_elementwise_v2 NN_SPV_FN_MISSING
#endif

#if __has_include("reduce_spv.hpp")
#include "reduce_spv.hpp"
#define NN_SPV_FN_reduce nn_reduce_spirv_bytecode
#else
#define NN_SPV_FN_reduce NN_SPV_FN_MISSING
#endif

// Phase C2 op-level f16 归约变体（输入 float16_t 直读、f32 归约；capability
// 仅 StorageBuffer16BitAccess，无 Float16 → 只需 16bit 存储门控）
#if __has_include("reduce_f16_spv.hpp")
#include "reduce_f16_spv.hpp"
#define NN_SPV_FN_reduce_f16 nn_reduce_f16_spirv_bytecode
#else
#define NN_SPV_FN_reduce_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("transpose_spv.hpp")
#include "transpose_spv.hpp"
#define NN_SPV_FN_transpose nn_transpose_spirv_bytecode
#else
#define NN_SPV_FN_transpose NN_SPV_FN_MISSING
#endif

#if __has_include("im2col_spv.hpp")
#include "im2col_spv.hpp"
#define NN_SPV_FN_im2col nn_im2col_spirv_bytecode
#else
#define NN_SPV_FN_im2col NN_SPV_FN_MISSING
#endif

// Phase C2d f16 卷积窗口变体（数据搬运，in/out 均 half；仅 16bit 存储）
#if __has_include("im2col_f16_spv.hpp")
#include "im2col_f16_spv.hpp"
#define NN_SPV_FN_im2col_f16 nn_im2col_f16_spirv_bytecode
#else
#define NN_SPV_FN_im2col_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("col2im_spv.hpp")
#include "col2im_spv.hpp"
#define NN_SPV_FN_col2im nn_col2im_spirv_bytecode
#else
#define NN_SPV_FN_col2im NN_SPV_FN_MISSING
#endif

#if __has_include("col2im_f16_spv.hpp")
#include "col2im_f16_spv.hpp"
#define NN_SPV_FN_col2im_f16 nn_col2im_f16_spirv_bytecode
#else
#define NN_SPV_FN_col2im_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("group_reduce_spv.hpp")
#include "group_reduce_spv.hpp"
#define NN_SPV_FN_group_reduce nn_group_reduce_spirv_bytecode
#else
#define NN_SPV_FN_group_reduce NN_SPV_FN_MISSING
#endif

// Phase C2c f16 分组归约变体（输入 half 直读、输出 float；仅 16bit 存储）
#if __has_include("group_reduce_f16_spv.hpp")
#include "group_reduce_f16_spv.hpp"
#define NN_SPV_FN_group_reduce_f16 nn_group_reduce_f16_spirv_bytecode
#else
#define NN_SPV_FN_group_reduce_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("gather_spv.hpp")
#include "gather_spv.hpp"
#define NN_SPV_FN_gather nn_gather_spirv_bytecode
#else
#define NN_SPV_FN_gather NN_SPV_FN_MISSING
#endif

// Phase C1 op-level f16 变体（-DNN_SHADER_F16=1 第二份 SPIR-V；纯位拷贝搬运）
#if __has_include("transpose_f16_spv.hpp")
#include "transpose_f16_spv.hpp"
#define NN_SPV_FN_transpose_f16 nn_transpose_f16_spirv_bytecode
#else
#define NN_SPV_FN_transpose_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("rearrange_3d_f16_spv.hpp")
#include "rearrange_3d_f16_spv.hpp"
#define NN_SPV_FN_rearrange_3d_f16 nn_rearrange_3d_f16_spirv_bytecode
#else
#define NN_SPV_FN_rearrange_3d_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("gather_f16_spv.hpp")
#include "gather_f16_spv.hpp"
#define NN_SPV_FN_gather_f16 nn_gather_f16_spirv_bytecode
#else
#define NN_SPV_FN_gather_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("elementwise_v2_f16_spv.hpp")
#include "elementwise_v2_f16_spv.hpp"
#define NN_SPV_FN_elementwise_v2_f16 nn_elementwise_v2_f16_spirv_bytecode
#else
#define NN_SPV_FN_elementwise_v2_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("scatter_add_spv.hpp")
#include "scatter_add_spv.hpp"
#define NN_SPV_FN_scatter_add nn_scatter_add_spirv_bytecode
#else
#define NN_SPV_FN_scatter_add NN_SPV_FN_MISSING
#endif

// Phase C2b f16 原子累加变体（dst 打包 half CAS + grad half 直读；
// capability 仅 StorageBuffer16BitAccess）
#if __has_include("scatter_add_f16_spv.hpp")
#include "scatter_add_f16_spv.hpp"
#define NN_SPV_FN_scatter_add_f16 nn_scatter_add_f16_spirv_bytecode
#else
#define NN_SPV_FN_scatter_add_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("scan_prefix_outer_spv.hpp")
#include "scan_prefix_outer_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer nn_scan_prefix_outer_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer NN_SPV_FN_MISSING
#endif

// Phase C2f RLA 扫描原语的 f16 存储变体（输入输出均 half；仅 16bit 存储）
#if __has_include("scan_prefix_outer_f16_spv.hpp")
#include "scan_prefix_outer_f16_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer_f16 nn_scan_prefix_outer_f16_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer_f16 NN_SPV_FN_MISSING
#endif

// P4-2b 窄共享变体（dk ≤ 32；同一份 .comp 以 -DNN_SCAN_DK_MAX=32 编出）
#if __has_include("scan_prefix_outer_d32_spv.hpp")
#include "scan_prefix_outer_d32_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer_d32 nn_scan_prefix_outer_d32_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer_d32 NN_SPV_FN_MISSING
#endif

#if __has_include("scan_prefix_outer_d32_f16_spv.hpp")
#include "scan_prefix_outer_d32_f16_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer_d32_f16 nn_scan_prefix_outer_d32_f16_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer_d32_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("scan_suffix_outer_spv.hpp")
#include "scan_suffix_outer_spv.hpp"
#define NN_SPV_FN_scan_suffix_outer nn_scan_suffix_outer_spirv_bytecode
#else
#define NN_SPV_FN_scan_suffix_outer NN_SPV_FN_MISSING
#endif

#if __has_include("scan_suffix_outer_f16_spv.hpp")
#include "scan_suffix_outer_f16_spv.hpp"
#define NN_SPV_FN_scan_suffix_outer_f16 nn_scan_suffix_outer_f16_spirv_bytecode
#else
#define NN_SPV_FN_scan_suffix_outer_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("scan_prefix_outer_gen_spv.hpp")
#include "scan_prefix_outer_gen_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer_gen nn_scan_prefix_outer_gen_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer_gen NN_SPV_FN_MISSING
#endif

#if __has_include("scan_prefix_outer_gen_f16_spv.hpp")
#include "scan_prefix_outer_gen_f16_spv.hpp"
#define NN_SPV_FN_scan_prefix_outer_gen_f16 nn_scan_prefix_outer_gen_f16_spirv_bytecode
#else
#define NN_SPV_FN_scan_prefix_outer_gen_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("scan_suffix_outer_gen_spv.hpp")
#include "scan_suffix_outer_gen_spv.hpp"
#define NN_SPV_FN_scan_suffix_outer_gen nn_scan_suffix_outer_gen_spirv_bytecode
#else
#define NN_SPV_FN_scan_suffix_outer_gen NN_SPV_FN_MISSING
#endif

#if __has_include("scan_suffix_outer_gen_f16_spv.hpp")
#include "scan_suffix_outer_gen_f16_spv.hpp"
#define NN_SPV_FN_scan_suffix_outer_gen_f16 nn_scan_suffix_outer_gen_f16_spirv_bytecode
#else
#define NN_SPV_FN_scan_suffix_outer_gen_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("outer_col_spv.hpp")
#include "outer_col_spv.hpp"
#define NN_SPV_FN_outer_col nn_outer_col_spirv_bytecode
#else
#define NN_SPV_FN_outer_col NN_SPV_FN_MISSING
#endif

#if __has_include("outer_col_f16_spv.hpp")
#include "outer_col_f16_spv.hpp"
#define NN_SPV_FN_outer_col_f16 nn_outer_col_f16_spirv_bytecode
#else
#define NN_SPV_FN_outer_col_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("cast_spv.hpp")
#include "cast_spv.hpp"
#define NN_SPV_FN_cast nn_cast_spirv_bytecode
#else
#define NN_SPV_FN_cast NN_SPV_FN_MISSING
#endif

// P2/F3：三值打包权重（打包字容器 + 量化 GEMM；见 shaders/trit_pack.comp /
// matmul_q.comp 与 include/neuralnet.cpp/ternary_pack.hpp）
#if __has_include("trit_pack_spv.hpp")
#include "trit_pack_spv.hpp"
#define NN_SPV_FN_trit_pack nn_trit_pack_spirv_bytecode
#else
#define NN_SPV_FN_trit_pack NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_q_spv.hpp")
#include "matmul_q_spv.hpp"
#define NN_SPV_FN_matmul_q nn_matmul_q_spirv_bytecode
#else
#define NN_SPV_FN_matmul_q NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_q_f16_spv.hpp")
#include "matmul_q_f16_spv.hpp"
#define NN_SPV_FN_matmul_q_f16 nn_matmul_q_f16_spirv_bytecode
#else
#define NN_SPV_FN_matmul_q_f16 NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_q_gemv_spv.hpp")
#include "matmul_q_gemv_spv.hpp"
#define NN_SPV_FN_matmul_q_gemv nn_matmul_q_gemv_spirv_bytecode
#else
#define NN_SPV_FN_matmul_q_gemv NN_SPV_FN_MISSING
#endif

#if __has_include("matmul_q_gemv_f16_spv.hpp")
#include "matmul_q_gemv_f16_spv.hpp"
#define NN_SPV_FN_matmul_q_gemv_f16 nn_matmul_q_gemv_f16_spirv_bytecode
#else
#define NN_SPV_FN_matmul_q_gemv_f16 NN_SPV_FN_MISSING
#endif

// AOT 融合 shader 注册表（构建期 scan_exprs 收集 + gen_fused 合成；表达式
// 只出现在 Layer，本表是折叠后的派生物）。运行时按 expr_spec_key 匹配 dispatch。
//
// ⚠ NN_EXPR_SCAN（扫描/收集期 TU，含 nn_enable_gpu_fusion 的收集器）**不嵌入**：
//   ① 注册表是收集器自身的**产物**——收集期哪怕只让编译器**探测**到它，
//     depfile 也会记录「fused_registry.hpp → collect.exe → .obj → fused_registry.hpp」
//     依赖环，下次构建 ninja 图加载即失败（实测）；
//   ② 收集走 CPU 干跑，运行期 dispatch 表与收集无关。
//   因此不变量：**NN_EXPR_SCAN ⇒ 不含 fused_registry**（应用/样例目标不受影响）。
// ⚠ 守卫必须写成**嵌套 #ifndef**、不能写成 `__has_include(...) && !defined(...)`
//   （或反过来短路）：clang 21 实测 `#if` 条件里的 `__has_include` **无论是否被
//   短路求值、都会被 -MD 记进 depfile**（该写法会触发上述依赖环）；
//   只有把整组条件编译**跳过**（组内不出词法）才不产生探测记录。
#ifndef NN_EXPR_SCAN
#if __has_include("fused_registry.hpp")
#include "fused_registry.hpp"
#endif
#endif

// Vulkan 设备/Pipeline 辅助类（detail::vk_check/convert + VulkanDevice + VulkanPipeline）
#include "compute_vk_device.hpp"

namespace nn
{

// 前向声明（模板 + 别名，匹配 algebra_matrix.hpp 的 MatrixT<P>）
template <Precision P> class MatrixT;
using Matrix = MatrixT<Precision::F32>;

namespace detail
{
// 纯 CPU 构建（无嵌入 SPIR-V）时 get_*_spirv() 的占位返回（RV-08 单一注册表）
[[nodiscard]] inline const std::vector<uint32_t>& spirv_missing()
{
    static const std::vector<uint32_t> empty;
    return empty;
}
}  // namespace detail

// ── P3 跨链批量派发：单实例描述（run_fused_gpu_batch 的输入）──────────────
// 与 run_fused_gpu 的单 kernel 调用同一组形参的实例化：每实例自带输入/输出
// 缓冲与元素数，共享一份 push constants（consts/rparams/vp 同值是分组前提）。
// 实例表布局与生成器（expr_glsl_gen batched 分支）同源：uvec2 数组，每实例
// 一段、stride **S = inputs.size() + 1 + extra_outs.size() + 1**——
//   [0, n_in) 输入地址 / [n_in] 主输出 / [n_in+1+k] extras / [S-1] = (count, cols)
// 尾槽 = (本实例元素数, 本实例列数)：组键与 rows/cols 无关（同组形状可不同），
// count/cols 必须逐实例携带（shader 侧 `#define cols` 与 count_l 同读尾槽）。
struct FusedBatchInst
{
    std::vector<const GpuBuffer*> inputs;      // 与 spec.views 同序
    const GpuBuffer*              out = nullptr;    // 主输出（槽 n_in）
    std::vector<const GpuBuffer*> extra_outs;  // extras 输出（槽 n_in+1+k）
    std::uint32_t                 count = 0;   // rows*cols（元素数）
    std::uint32_t                 cols = 0;    // 每实例列数（row/col 推导）
};

// ══════════════════════════════════════════════════════════════════════════
// GpuBackend — Vulkan 计算后端单例
// ══════════════════════════════════════════════════════════════════════════

class GpuBackend : public std::enable_shared_from_this<GpuBackend>
{
private:
    VulkanDevice device_;
    // ── 原语 pipeline 目录 ──────────────────────────────────────────────
    // 44 个原语/变体的"身份 + 布局 + 设备能力要求"是 pipeline_specs() 的
    // 一行表格；这里只保留目录句柄。缺失的变体 = 空句柄 = 上层回退。
    PipelineCatalog pipelines_;
    // 列归约两段式 partials scratch（backend 自有的裸 VkBuffer）。
    //
    // 为什么不再是 GpuTensor：GpuBuffer 现在持有 shared_ptr<GpuBackend>
    // keepalive；若 backend 再持有 GpuTensor，就会形成
    // backend → GpuTensor → GpuBuffer → backend 的引用环，backend 永不析构。
    // scratch 的生命周期严格短于 backend，直接由 VkBuffer + Allocation 管理。
    struct ReduceScratch
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        MemoryPool::Allocation alloc{};
        MemoryPool* pool = nullptr;
        std::uint32_t rows = 0;
        std::uint32_t cols = 0;
    };
    ReduceScratch reduce_partial_;
    // ── AOT 融合 kernel 目录 ────────────────────────────────────────────
    // 一个 key（expr_spec_key）对应一个 FusedKernel：pipeline + 该结构在
    // 运行期需要的全部形状/布局元数据。**注册时一次写定，运行期只读**。
    //
    // 旧实现是 9 张并行 map（pipeline/reduce_axis/has_matmul/fold_meta/
    // view_param/rparam/vec_width/pc_base_fixed/batch_stride），每次派发要
    // 9 次查找且必须手工保持 key 同步；漏写一张表 = 静默读默认值。
    struct FoldMeta
    {
        // 输出列数（vec_state_len 或 1）；0 = 非 fold 形态
        std::uint32_t out = 0;
        // 1 → v1 标量 fold：每线程一行（wg = ceil(rows/256)）
        // 0 → v2 双域 fold：每 WG EXPR_FOLD_ROWS_PER_WG 行（wg = ceil(rows/NR)）
        // ——dispatch 公式两者不同（v2 若按 ceil(count/256) 会丢行）
        std::uint8_t per_thread_row = 0;
    };

    struct FusedKernel
    {
        VulkanPipeline pipeline;
        int reduce_axis = -1;          // -1=逐元素, 0=行归约, 1=列归约
        bool has_matmul = false;       // 含前置 matmul 段（PC 多 rows + mm_k）
        FoldMeta fold;
        std::uint32_t view_param_count = 0;   // vp push constant 槽数
        std::uint32_t rparam_count = 0;       // rp push constant 槽数
        std::uint32_t vec_width = 1;          // 每线程元素数（1 / 4）
        // push constant 固定头长度（uint 槽数，不含 vp/常量池）。
        // 生成器与写入侧共用这一个值：重复推导一旦漂移，vkCmdPushConstants
        // 会超 range 被驱动丢弃 → shader 读未定义残留（假 PASS）。
        std::uint32_t pc_base_fixed = 2;
        // `#b` 批量变体的实例表 stride（uvec2 槽数）；0 = 非批量变体
        std::uint32_t batch_stride = 0;

        [[nodiscard]] bool is_fold() const noexcept { return fold.out > 0; }
    };

    std::unordered_map<std::string, FusedKernel> fused_kernels_;
    // P3 批量派发观测计数（测试/诊断用：确认批量路径真的走到、而非静默降级
    // 为逐 kernel 后 A/B 自比恒真）
    std::uint64_t fused_batch_dispatches_ = 0;

    std::unique_ptr<MemoryPool> memory_pool_;
    // 第二阶段：瞬态/持久分池。batch 录制期（batch_mode_=true）创建的
    // 张量（融合临时、每步激活、输入上传等）走 transient_pool_，参数/梯度/权重
    // （构建期 batch_mode_=false）走 memory_pool_。此拆分**纯组织性**——两池均
    // 不强制释放，仅按生命周期隔离，避免频繁临时分配在参数常驻块里切出碎片。
    std::unique_ptr<MemoryPool> transient_pool_;
    std::unique_ptr<StagingRing> staging_ring_;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool gpu_tensor_pool_ = VK_NULL_HANDLE;

    // ── 设备侧计时（NN_PROFILE=1 且所选队列支持时间戳时启用）───────────
    // ts_dispatch 在每个 vkCmdDispatch 前后各写一个 timestamp，按 (帧|solo)
    // 分块分配查询区间：帧块在 reap_frame（fence 已等）resolve、solo 块在
    // submit_and_wait（已等）resolve，逐标号聚合进 nn::prof::device_add。
    // begin/end 均用 ALL_COMMANDS：begin 在此前命令全部完成后落笔、end 在
    // 本 dispatch 完成后落笔 ⇒ 每对标记间隔 ≈ 该 kernel 的 GPU 执行时间；
    // dispatch 之间的屏障/空隙不计入任何标号（即"kernel 时间 vs 空隙"口径）。
    static constexpr std::uint32_t kTsPairsPerBlock = 4096;
    struct TsMark
    {
        const char* key;      // 聚合键 "label@stage"（表 4）
        const char* detail;   // per-key 归因键 "detail@label@stage"（表 5；可空）
        std::uint32_t pair;   // 查询对号
    };
    struct TsCursor
    {
        std::uint32_t base = 0;   // 本块首查询号
        std::uint32_t next = 0;   // 已用 pair 数
        std::vector<TsMark> marks;
    };

    // ── Command Buffer Batching + 多帧流水线─────────────────────
    // 单帧模型（一次提交 + 阻塞等待）会让 GPU 执行 step N 时 host 卡在 fence 上、
    // 录不了 step N+1——GPU 占用率 = GPU 忙时间 / (GPU 忙 + 全部 host 录制 +
    // 传输)，host 录制越长 GPU 空转越多。故 end_batch/flush_batch 提交不等待。
    //
    // N 帧环：
    //   - PIPELINE_FRAMES 个 (command buffer + fence) 在 initialize() 预分配，
    //     轮转复用；描述符集按帧归属（帧的 fence 信号后才可释放）
    //   - begin_batch：取下一帧；该帧若仍在飞行则等其 fence（**只在环槽
    //     复用前等待**），reap 后开始录制
    //   - end_batch / flush_batch：提交不等待（host 继续录制，GPU 在队列
    //     上先行执行；同队列 FIFO 保证数据依赖）。空帧不提交（省 submit）
    //   - wait_in_flight：**"真正要结果"的阻塞点**（to_matrix/copy_from 读
    //     GPU 内存前）——等待所有在飞帧并 reap
    //   - release_idle_pool_blocks：非阻塞 reap 已完成帧 + 归还空闲块
    // 效果：host 录制 step N+1 与 GPU 执行 step N 重叠；host 只在真正要
    // 读 GPU 数据（logits 探针 / loss 标量 / 日志采样）时阻塞。
    static constexpr std::size_t PIPELINE_FRAMES = 6;  // 环深度（加深流水线提 host/GPU 重叠度）
    struct Frame
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;  // 创建即为 unsignaled
        std::vector<VkDescriptorSet> desc_sets;  // 本帧使用的描述符集
        bool in_flight = false;                  // 已提交、fence 尚未被等待
        TsCursor ts;                             // 设备侧计时块（NN_PROFILE）
    };
    std::vector<Frame> frames_;
    std::size_t frame_next_ = 0;         // 下一轮转帧（begin/flush 取用）
    std::size_t last_active_frame_ = 0;  // 最近录制/提交的帧（非 batch 销毁决策用）
    bool batch_mode_ = false;
    std::size_t batch_frame_ = 0;        // 当前录制中的帧
    bool batch_has_ops_ = false;         // 当前帧是否已录制 op（空帧不提交）
    // 帧提交后是否立即非阻塞收割已完成帧（显存峰值优化；NN_NO_EARLY_REAP=1 关闭）
    bool early_reap_enabled_ = true;
    VkCommandBuffer batch_cmd_ = VK_NULL_HANDLE;  // 当前帧的 command buffer

    // ── 异步标量回读（非阻塞 loss 取值）─────────────────────────
    // 动机：forward 末尾下载 loss 标量若走 to_matrix，会 end_batch +
    // wait_in_flight（等全部在飞帧）→ 每个 step 一次全流水线 drain，host 与
    // GPU 无法重叠（锯齿）。这里给标量回读一条"不等"的路径：
    //   每个槽位 = 独立 command buffer + fence + 持久 mapped host 缓冲（4B）。
    //   submit：录制一次 4B D2H 拷贝并提交（**不等待**），调用方须在主帧
    //           提交之后调用（同队列 FIFO ⇒ 拷贝排在生产者之后，不会读到旧值）。
    //   poll ：vkGetFenceStatus 非阻塞查询；就绪则从 mapped 内存读值。
    // 槽位不参与 frames_ 环（环槽会被复用、fence 会被 reset），故自带 fence。
    static constexpr std::size_t SCALAR_READBACK_SLOTS = 8;

    struct ScalarReadbackSlot
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkBuffer host = VK_NULL_HANDLE;
        MemoryPool::Allocation alloc{};
        void* mapped = nullptr;
        bool pending = false;    // 已提交、fence 未确认
    };
    std::vector<ScalarReadbackSlot> rb_slots_;

    // ── 独立（非 batch）模式复用资源 ────────────────────────────────────
    // 逐元素算子固定开销实测 ≈0.14ms/次，几乎全在 host 侧（4096² kernel
    // 本身已达 370GB/s，计算远快于提交）：故 fence/cmd 不按算子新建，而是
    // initialize() 预分配，acquire_cmd/submit_and_wait 内 reset 复用。
    // 独立模式「录制→提交→等待」严格串行（等完才返回）→ 单实例安全；
    // batch 模式走 frames_ 环，不经过这里。单线程录制假设同 batch_cmd_。
    VkFence solo_fence_ = VK_NULL_HANDLE;        // 创建即 unsignaled
    VkCommandBuffer solo_cmd_ = VK_NULL_HANDLE;   // 每次 reset + begin 复用

    // ── 设备侧计时：运行期状态（TsCursor 定义见帧环段）────────────────
    VkQueryPool ts_pool_ = VK_NULL_HANDLE;
    bool ts_enabled_ = false;
    float ts_period_ns_ = 1.0f;
    TsCursor ts_solo_;                     // solo 块（base = PIPELINE_FRAMES × 块长）
    TsCursor* ts_cur_ = &ts_solo_;         // 当前录制块（acquire_cmd 切换）

    std::mutex init_mutex_;
    std::mutex queue_mutex_;
    bool initialized_ = false;

    // ── f16 硬件能力（§7.1）────────────────────────────────────────
    // GPU shaderFloat16：运行期一次查定，运行期不变。
    // true  → f16 GEMM 变体②（f16vec 加载，需该特性）
    // false → f16 GEMM 变体①（u8 软件解码，设备无关，默认路径）
    bool has_shader_float16_ = false;

    // ── 延迟销毁：batch 录制期间 copy-on-write 替换旧 buffer 时，旧 buffer
    // 仍被已录制的 descriptor set 引用，不能立即 vkDestroyBuffer。
    // 待引用它的帧执行完（该帧 fence 信号 → reap_frame，descriptor set 同帧
    // 释放）后销毁并归还内存；drain 点（wait_in_flight / 析构）走
    // flush_pending_destroys() 一次性清空。
    //
    // ⚠ 内存归还也须延迟到 vkDestroyBuffer 之后：
    //   若立即 pool_->free()，新 buffer 可能分配到旧 buffer 尚未销毁的
    //   同一区间 → 两个存活 buffer 内存重叠，违反 Vulkan 规范
    //   （VUID-vkBindBufferMemory-memory-01988 类约束）。
    //   数据竞争方面：同一 command buffer 内命令按录制顺序执行，旧 buffer
    //   的已录制命令先于复用同一区间的新 buffer 命令执行，无数据竞争。
    //   但规范合规要求 buffer 对象本身不得重叠，故内存归还必须等 buffer
    //   销毁完成。
    //
    //   多帧流水线：每条延迟销毁打上帧标签（frame），该帧的 fence
    //   信号后（环槽复用 / drain / 非阻塞 reap）即销毁并归还内存——锁窗
    //   只到所属帧，不跨越整个 batch。
    struct PendingDestroy
    {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        MemoryPool::Allocation alloc;
        observer_ptr<MemoryPool> pool;
        std::size_t frame = 0;  // 允许销毁的帧索引（该帧完成后才可释放）
    };
    std::mutex pending_mutex_;
    std::vector<PendingDestroy> pending_destroys_;

    static constexpr uint32_t WORKGROUP_SIZE = 16;

    GpuBackend() = default;

    // ── 原语 pipeline 布局表（单一真源；定义见 compute_vk_pipeline_specs.hpp）
    [[nodiscard]] static std::span<const PipelineSpec> pipeline_specs();

    // ── SPIR-V 字节码 getter（由文件头的单一注册表展开；RV-08）────────────
    // 每个 shader 的字节码访问函数在文件头绑定为 NN_SPV_FN_<name>（有嵌入头 = 真实
    // 函数；纯 CPU 构建 = 占位空向量）；此处由列表统一展开为 get_<name>_spirv()。
#define NN_SPV_GETTER(NAME)                                                    \
    [[nodiscard]] static const std::vector<uint32_t>& get_##NAME##_spirv()     \
    {                                                                         \
        return NN_SPV_FN_##NAME();                                            \
    }
    NN_SPV_GETTER_LIST(NN_SPV_GETTER)
#undef NN_SPV_GETTER

    // ── AOT 融合 shader SPIR-V 来自构建期生成的 fused_registry.hpp ────────
    // （FusedShader 直接携带内联 SPIR-V，无需按名 getter；注册在 init 时遍历）

public:
    ~GpuBackend()
    {
        if (device_.device() != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_.device());

            // scratch 必须先于 memory pool 释放；它只由 backend 自己持有，
            // 不经过 GpuBuffer 的延迟销毁队列。
            destroy_reduce_partial_now_();
            flush_pending_destroys();

            destroy_scalar_readback_slots();

            staging_ring_.reset();

            // 释放帧环 fences（环内 command buffers 随 command pool 一起释放）
            for (auto& f : frames_)
            {
                if (f.fence != VK_NULL_HANDLE)
                    vkDestroyFence(device_.device(), f.fence, nullptr);
            }

            // 独立模式复用 fence（solo command buffer 随 command pool 释放）
            if (solo_fence_ != VK_NULL_HANDLE)
            {
                vkDestroyFence(device_.device(), solo_fence_, nullptr);
                solo_fence_ = VK_NULL_HANDLE;
            }

            if (gpu_tensor_pool_ != VK_NULL_HANDLE)
                vkDestroyDescriptorPool(device_.device(), gpu_tensor_pool_, nullptr);
            if (ts_pool_ != VK_NULL_HANDLE)
                vkDestroyQueryPool(device_.device(), ts_pool_, nullptr);
            if (command_pool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_.device(), command_pool_, nullptr);
        }

        memory_pool_.reset();
        transient_pool_.reset();
    }

    // 设备丢失状态
    bool device_lost_ = false;
    [[nodiscard]] bool is_device_lost() const noexcept { return device_lost_; }

    // ── 设备能力直读（P3 门禁等：int8 点积特性在设备创建时已查询/启用）────
    // 只读句柄：调用方（GpuEngine）据此暴露 `supports_int8_dot()` 等能力位。
    [[nodiscard]] const VulkanDevice& device() const noexcept { return device_; }

    // 禁止拷贝和移动
    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;
    GpuBackend(GpuBackend&&) = delete;
    GpuBackend& operator=(GpuBackend&&) = delete;

    // 设备域标识（生命周期 v3）：backend 对象地址。GpuBuffer 持 keepalive ⇒
    // 张量存活期内该地址不会被新 backend 复用（P3 改 weak 缓存后依然成立）。
    [[nodiscard]] DomainId identity() const noexcept
    {
        return reinterpret_cast<DomainId>(this);
    }

    // 进程共享 backend（生命周期 v3 / D5）：weak 缓存 + mutex。
    //   · 有活使用者（GpuEngine / GpuBuffer / 调用方句柄）时共享同一实例；
    //   · 全部释放后 backend 及其 Vulkan 资源**确定性回收**（地址可被新
    //     backend 复用——GpuBuffer 的 keepalive 保证张量存活期内不复用）；
    //   · 下一次 shared() 重新创建（无设备初始化，调用方仍需 initialize()）。
    // 需要独立生命周期（测试/多设备）用 create()。
    [[nodiscard]] static std::shared_ptr<GpuBackend> shared()
    {
        static std::mutex mu;
        static std::weak_ptr<GpuBackend> weak;
        std::lock_guard<std::mutex> lock(mu);
        if (auto alive = weak.lock())
            return alive;
        std::shared_ptr<GpuBackend> backend(new GpuBackend());
        weak = backend;
        return backend;
    }

    // 独立实例（测试/多设备）：由调用方决定生命周期。
    [[nodiscard]] static std::shared_ptr<GpuBackend> create()
    {
        return std::shared_ptr<GpuBackend>(new GpuBackend());
    }

    [[nodiscard]] std::shared_ptr<GpuBackend> keepalive() noexcept
    {
        return weak_from_this().lock();
    }

    // ── 成员定义按主题拆分为 4 段（RV-08 物理拆分；纯移动，顺序即成员序）──
#include "compute_vk_backend_destroy.inc"      // 延迟销毁 / reap / 收帧 / 显存回收
#include "compute_vk_backend_frame.inc"        // 帧环 / batch / 上传下载 / dispatch / matmul
#include "compute_vk_backend_pipelines.inc"    // 扫描 + 辅助（描述符/依赖/solo fence/屏障）
#include "compute_vk_backend_ops.inc"          // 各原语 dispatch
};  // GpuBackend

// ══════════════════════════════════════════════════════════════════════
// GpuBuffer 析构（类外定义）：batch 录制期间延迟销毁，避免已录制的
// descriptor 引用已销毁的 buffer（VUID-vkDestroyBuffer-buffer-00922）。
//
// **必须 inline**：本库是 header-only，任何包含本头的翻译单元都会发射本定义
// ——缺失 inline 会让"≥2 个 TU 的程序"链接期 duplicate symbol（实测
// `examples/fusion_custom_layer_*` 两 TU 即撞）。库外使用者的常规工程
// （多 TU）因此不可用。
// ══════════════════════════════════════════════════════════════════════
inline GpuBuffer::~GpuBuffer()
{
    if (buffer_ == VK_NULL_HANDLE)
        return;

    // 本 keepalive 保证 backend 在析构全程存活；不再经全局 instance() 取
    // 一个可能已经销毁的单例。
    auto backend = std::move(backend_keepalive_);
    if (backend && backend->is_initialized() && backend->needs_deferred_destroy())
    {
        // 延迟 vkDestroyBuffer（Vulkan 规范要求 buffer 在引用它的命令
        // 提交完成前不能被销毁）。两种延迟场景（needs_deferred_destroy）：
        //   1. batch 录制中：本 buffer 被当前帧已录制的命令引用
        //   2. 非 batch 但最近提交的帧仍在飞行：本 buffer 可能被该帧命令
        //      引用（fence 信号即可证明安全）
        // 延迟销毁打上帧标签，该帧完成（环槽复用 / drain / 非阻塞 reap）
        // 时立即销毁 + 归还内存——锁窗只到所属帧，不跨越整个 batch
        // （多帧流水线）。
        // 内存归还也一并延迟到 buffer 销毁之后。若此处立即
        // pool_->free()，新 buffer 可能分配到本 buffer 尚未销毁的同一区间
        // → 两个存活 buffer 内存重叠，违反 Vulkan 规范。
        // GpuBuffer 只经 MemoryPool 创建（create_device_local /
        // create_host_visible），pool_ 恒有效；alloc_ 无效时 free 为 no-op。
        backend->defer_buffer_destroy(device_, buffer_, alloc_, pool_);
    }
    else
    {
        // 无在飞帧引用：立即销毁 + 归还内存
        vkDestroyBuffer(device_, buffer_, nullptr);
        if (pool_ && alloc_.valid())
            pool_->free(alloc_);
    }

    buffer_ = VK_NULL_HANDLE;
    alloc_ = {};
    pool_.reset();
}

} // namespace nn

// 原语 pipeline 布局表（GpuBackend::pipeline_specs 的定义）
#include "compute_vk_pipeline_specs.hpp"

// GpuTensorT<P> 的 backend 依赖实现（from_matrix / to_matrix / create_*）：
// 放在 GpuBackend 与 GpuBuffer 析构之后——存储层（compute_gpu_storage.hpp）
// 本身不认识后端，只有这些"经后端上传/下载"的动作需要完整定义。
// algebra_matrix.hpp 不再包含它，从而只依赖存储层。
#include "compute_gpu_tensor_impl.hpp"

#endif // NN_HAS_VULKAN
