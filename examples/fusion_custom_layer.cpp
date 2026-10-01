// ═══════════════════════════════════════════════════════════════════════════
//  examples/fusion_custom_layer.cpp — 自研层的翻译单元（A2 验收用）
//
//  存在的唯一目的：让 `nn_enable_gpu_fusion` 编收集器时**编译到本 TU**，
//  从而让 fusion_custom_layer.hpp 里的 `dsl::compute` 调用点在静态初始化期
//  经 `FusedAnchor<Expr>` 自登记——这正是"收集器必须看到使用者的调用点"的
//  落地方式（收集器 = `tools/scan_exprs.cpp` 的库内收集 + 本应用的全部 TU）。
// ═══════════════════════════════════════════════════════════════════════════

#include "fusion_custom_layer.hpp"

namespace nn_example
{

// 非 inline：odr-use 该函数 → 其函数体被发射 → 内部的 `dsl::compute<E>` 实例化
// → `FusedAnchor<E>` 静态初始化登记结构。不调用它、也不改任何库内文件。
void fusion_anchor_touch()
{
    (void)&fused_custom_op;
}

} // namespace nn_example
