#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_registry.hpp — 可融合表达式注册表（AOT 收集的载体）
//
//  AOT 收集架构（表达式只在 Layer 里，别处一律不出现）：
//    ① 构建期 `scan_exprs` 收集结构：`FusedAnchor<Expr>` 在静态初始化期按
//       表达式**类型**登记（编译期可达，覆盖 dry-run 跑不到的分支），再由
//       dry-run / 模型 pass 补精度签名——每个 dsl::compute* 在记录模式下把
//       折叠出的 ExprSpec 登记进本注册表（按 expr_spec_key 去重）。
//    ② 同一次运行内直接生成：`tools/fused_generate.hpp` 对每条 spec 出
//       GLSL（glsl_gen.hpp）→ glslc → SPIR-V → 内联进 fused_registry.hpp
//       （key → spirv）。**不经 .bin 中间序列化**（原 gen_fused 已并入）。
//    ③ 运行时 eval_expr 折叠内联表达式 → expr_spec_key → 查嵌入映射
//       → dispatch。未命中硬报错（闭合世界，见 expr_dsl.hpp 的锚点说明）。
//
//  表达式**文本**只出现在 Layer；生成头是折叠后的**派生物**，
//  非手写定义，故不违反"表达式只在 Layer"的约束。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "expr_spec.hpp"
#include "expr_opt.hpp"

namespace nn::fused
{

// ── 注册表：收集折叠出的 ExprSpec **结构**，按规范 key 去重 ───────────────
//
// 只装**结构**：精度签名已不是构建期集合（生成阶段对每个结构发运行期精度
// 分派 shader，键 `key#x`；见 expr_spec.hpp 的 EXPR_PREC_SIG_DISPATCH）。
// 历史上有过 `variants`（(结构, 签名) 表）与 `contains_variant`——随签名
// 降级为运行期参数一并删除（docs/history.md）。
struct ExprRegistry
{
    std::vector<ExprSpec>       specs;
    std::unordered_set<std::string> keys;

    void add(const ExprSpec& s)
    {
        // 登记 canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        // key 建立在 canonical 形态上（scan 与 runtime 两端一致）。
        const ExprSpec canon = canonicalize_expr_spec(s);
        const std::string k = expr_spec_key(canon);
        if (keys.insert(k).second)
            specs.push_back(canon);
    }
    [[nodiscard]] bool contains(const ExprSpec& s) const
    { return keys.count(expr_spec_key(canonicalize_expr_spec(s))) != 0; }
};

// 全局注册表（scan_exprs 记录模式写入；普通构建不含 NN_EXPR_SCAN，零开销）
[[nodiscard]] inline ExprRegistry& global_registry()
{
    static ExprRegistry reg;
    return reg;
}

} // namespace nn::fused

