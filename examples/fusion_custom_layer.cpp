// ═══════════════════════════════════════════════════════════════════════════
//  examples/fusion_custom_layer.cpp — 自研层的翻译单元（A2 验收用）
//
//  存在的唯一目的：让 `nn_enable_gpu_fusion` 编收集器时**编译到一个包含本层
//  的非 main TU**（收集器排除 MAIN 源），从而让 fusion_custom_layer.hpp 里的
//  `dsl::compute` 调用点在静态初始化期经 `FusedAnchor<Expr>` 自登记——这正是
//  "收集器必须看到使用者的调用点"的落地方式（收集器 = tools/scan_exprs.cpp
//  的库内收集 + 本应用的全部非 main TU）。
//
//  本 TU 不需要额外函数：包含头文件即会编译 `fused_custom_op` /
//  `fused_custom_norm` 的函数体，其中的 `dsl::compute<E>` 实例化会 odr-use
//  `FusedAnchor<E>::reg`（静态数据成员 → 静态初始化期登记）。
// ═══════════════════════════════════════════════════════════════════════════

#include "fusion_custom_layer.hpp"
