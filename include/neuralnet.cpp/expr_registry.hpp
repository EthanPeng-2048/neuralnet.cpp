#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_registry.hpp — 可融合表达式注册表（AOT 收集的载体）
//
//  AOT 收集架构（表达式只在 Layer 里，别处一律不出现）：
//    ① 构建期 `scan_exprs` 收集结构：`FusedAnchor<Expr>` 在静态初始化期按
//       表达式**类型**登记（编译期可达，覆盖 dry-run 跑不到的分支），再由
//       dry-run / 模型 pass 补"只有执行 Layer 才拿得到"的结构——每个
//       dsl::compute* 在记录模式下把折叠出的 ExprSpec 登记进本注册表
//       （按 expr_spec_key 去重）。**只收集结构**：精度签名不是构建期集合。
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

// ── 注册表：收集折叠出的 ExprSpec **结构**（按规范 key 去重）+ 观察到的
//    **精度签名**（供编译期特化变体 `key#xxxx` 的生成；见 add_variant）──
//
// 结构仍是身份：运行期进度签名有兜底载体 `key#x`（一个 shader 覆盖任意
// 签名）。但 `#x` 的代价是热循环里每次读写都带 `prec` 三元——实测（850M，
// GPT d64/h4/L4/ff256/seq256/b16，f16）把 CE 融合注意力从 146 ms 降到
// 120 ms（=−18%，f32 基线 117 ms）**只**因为换了编译期特化 shader。
// 因此这里额外收集"扫描期真实出现过的 (结构, 签名)"，生成阶段为每条发一份
// 静态特化 variant（f16 存储 + f32 算术，键 = `key#xxxx`，与 ggml/torch 的
// per-type kernel 特化同思路）。**它是纯优化**：未枚举到的签名照旧落 `#x`。
//
// 历史：早期有过 `variants` 表，随"签名降级为运行期参数"一并删除；本次
// 以"特化优化（非身份）"的名义恢复，语义与那时不同（那时是唯一载体）。
struct ExprRegistry
{
    struct VariantRef
    {
        std::string  key;   // 结构 key（canonical 形态）
        ExprPrecSig  sig;   // 观察到的精度签名（≠0、≠ EXPR_PREC_SIG_DISPATCH）
    };

    std::vector<ExprSpec>           specs;
    std::unordered_set<std::string> keys;
    std::vector<VariantRef>         variants;
    std::unordered_set<std::string> variant_keys;

    void add(const ExprSpec& s)
    {
        // 登记 canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        // key 建立在 canonical 形态上（scan 与 runtime 两端一致；运行时侧
        // 同一规则见 expr_opt.hpp 的 expr_spec_lookup_key）。
        const ExprSpec canon = canonicalize_expr_spec(s);
        const std::string k = expr_spec_key(canon);
        if (keys.insert(k).second)
            specs.push_back(canon);
    }

    // 记录"某结构在某签名下真的被调用过"（扫描期 = dry-run / 模型 pass /
    // 显式 pass，均为真实张量精度 + 真实 P）。全 f32（sig==0）与分派哨兵
    // 不记：前者与 V0 共用条目，后者不是签名。
    void add_variant(const ExprSpec& s, ExprPrecSig sig)
    {
        if (sig == 0u || expr_prec_sig_is_dispatch(sig))
            return;
        const ExprSpec canon = canonicalize_expr_spec(s);
        const std::string k = expr_spec_key(canon);
        const std::string vk = expr_prec_sig_key(k, sig);
        if (variant_keys.insert(vk).second)
            variants.push_back(VariantRef{k, sig});
    }
};

// 全局注册表（scan_exprs 记录模式写入；普通构建不含 NN_EXPR_SCAN，零开销）
[[nodiscard]] inline ExprRegistry& global_registry()
{
    static ExprRegistry reg;
    return reg;
}

} // namespace nn::fused

