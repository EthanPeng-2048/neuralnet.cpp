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

// ── 精度变体（Phase 2 in-kernel f16）：同一结构 + 精度签名 ────────────────
struct ExprVariant
{
    ExprSpec    spec;
    ExprPrecSig sig = 0;
};

// ── 注册表：收集折叠出的 ExprSpec 结构，按规范 key 去重 ─────────────────
struct ExprRegistry
{
    // sig == 0（全 f32）：按结构 key 去重进基础表 specs —— bin / 生成头的
    // 基础段只承载这些条目（key 不加后缀）。
    std::vector<ExprSpec>       specs;
    std::unordered_set<std::string> keys;
    // sig != 0（带类型变体）：按 (结构 key, 精度签名) 去重。bin 只序列化
    // {sig, 基础结构下标}：变体与基础结构**同结构同 key**，故只需下标引用。
    // 同结构同 key，spec 体复用基础表条目，不重复落盘。
    std::vector<ExprVariant>    variants;
    std::unordered_set<std::string> variant_keys;

    void add(const ExprSpec& s, ExprPrecSig sig = 0)
    {
        // 登记 canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        // bin 与 key 建立在 canonical 形态上（scan 与 runtime 两端一致）。
        const ExprSpec canon = canonicalize_expr_spec(s);
        const std::string k = expr_spec_key(canon);
        if (sig == 0u)
        {
            if (keys.insert(k).second)
                specs.push_back(canon);
            return;
        }
        const std::string vk = expr_prec_sig_key(k, sig);
        if (variant_keys.insert(vk).second)
            variants.push_back(ExprVariant{canon, sig});
    }
    [[nodiscard]] bool contains(const ExprSpec& s) const
    { return keys.count(expr_spec_key(canonicalize_expr_spec(s))) != 0; }
    // 该 (结构, 签名) 是否已登记（sig==0 走结构表）
    [[nodiscard]] bool contains_variant(const ExprSpec& s, ExprPrecSig sig) const
    {
        const std::string k = expr_spec_key(canonicalize_expr_spec(s));
        if (sig == 0u)
            return keys.count(k) != 0;
        return variant_keys.count(expr_prec_sig_key(k, sig)) != 0;
    }
};

// 全局注册表（scan_exprs 记录模式写入；普通构建不含 NN_EXPR_SCAN，零开销）
[[nodiscard]] inline ExprRegistry& global_registry()
{
    static ExprRegistry reg;
    return reg;
}

} // namespace nn::fused

