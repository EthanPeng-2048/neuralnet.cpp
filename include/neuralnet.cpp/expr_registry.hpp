#ifndef NN_EXPR_REGISTRY_HPP
#define NN_EXPR_REGISTRY_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_registry.hpp — 可融合表达式注册表（AOT 收集的两端共用）
//
//  AOT 收集架构（表达式只在 Layer 里，别处一律不出现）：
//    ① 构建期 scan_exprs 用假张量 dry-run Layer 的 forward/backward，
//       每个 dsl::compute / end_expr 在记录模式下把折叠出的 ExprSpec
//       **结构**登记进全局注册表（按 expr_spec_key 去重）→ dump 成 bin。
//    ② 构建期 gen_fused 读 bin → 每 spec 生成 GLSL（glsl_gen.hpp）→
//       glslc → SPIR-V → 内联进生成头 fused_registry.hpp（key → spirv）。
//    ③ 运行时 eval_expr 折叠内联表达式 → expr_spec_key → 查嵌入映射
//       → dispatch。未命中硬报错（闭合世界，提示扫描未覆盖该路径）。
//
//  表达式**文本**只出现在 Layer；bin / 生成头都是折叠后的**派生物**，
//  非手写定义，故不违反"表达式只在 Layer"的约束。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "expr_spec.hpp"
#include "expr_opt.hpp"

namespace nn::fused
{

// ── 注册表：收集折叠出的 ExprSpec 结构，按规范 key 去重 ─────────────────
struct ExprRegistry
{
    std::vector<ExprSpec>       specs;
    std::unordered_set<std::string> keys;

    void add(const ExprSpec& s)
    {
        // 登记 canonical IR：canonicalize 为引擎内部优化（IR-A/IR-B），
        // bin 与 key 建立在 canonical 形态上（scan 与 runtime 两端一致）。
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

// ── 二进制序列化（dump/load 共用同一格式）───────────────────────────────
// 格式（小端，x86/ARM 通用）：
//   magic "NNEXP" (5B) + version (u8=2)
//   count (u32)
//   每 spec：num_regs(u32)
//            instrs: count(u32) × {op(u8) dst(u8) a.kind a.idx b.kind b.idx c.kind c.idx}
//            views:  count(u32) × {kind(u8) negate(u8) param(u32)}
//            consts: count(u32) × Scalar
//            matmul: has(u8=0/1)；1 时 {a_input(u8) b_input(u8) transA(u8)
//                    transB(u8) k(u32)}
//  v2 起支持 matmul 段（v1 无 matmul，读 v1 等价 has=0）。
inline constexpr std::uint8_t kExprBinVersion = 2;

[[nodiscard]] inline bool write_registry(const std::string& path,
                                         const ExprRegistry& reg)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("NNEXP", 5);
    f.write(reinterpret_cast<const char*>(&kExprBinVersion), 1);
    const std::uint32_t count = static_cast<std::uint32_t>(reg.specs.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& s : reg.specs)
    {
        f.write(reinterpret_cast<const char*>(&s.num_regs), sizeof(s.num_regs));
        std::uint32_t n = static_cast<std::uint32_t>(s.instrs.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& in : s.instrs)
        {
            std::uint8_t bytes[8] = { in.op, in.dst, in.a.kind, in.a.idx,
                                      in.b.kind, in.b.idx, in.c.kind, in.c.idx };
            f.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        }
        n = static_cast<std::uint32_t>(s.views.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& v : s.views)
        {
            f.write(reinterpret_cast<const char*>(&v.kind), 1);
            f.write(reinterpret_cast<const char*>(&v.negate_first_half), 1);
            f.write(reinterpret_cast<const char*>(&v.param), sizeof(v.param));
        }
        n = static_cast<std::uint32_t>(s.consts.size());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& c : s.consts)
            f.write(reinterpret_cast<const char*>(&c), sizeof(c));
        const std::uint8_t has_mm = s.matmul ? 1 : 0;
        f.write(reinterpret_cast<const char*>(&has_mm), 1);
        if (s.matmul)
        {
            std::uint8_t mbytes[4] = { s.matmul->a_input, s.matmul->b_input,
                                       s.matmul->transA, s.matmul->transB };
            f.write(reinterpret_cast<const char*>(mbytes), sizeof(mbytes));
            f.write(reinterpret_cast<const char*>(&s.matmul->k), sizeof(s.matmul->k));
        }
    }
    return static_cast<bool>(f);
}

// 读回注册表（gen_fused 用）。返回 false 表示格式/路径错误。
[[nodiscard]] inline bool read_registry(const std::string& path, ExprRegistry& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[5];
    f.read(magic, 5);
    if (std::string(magic, 5) != "NNEXP") return false;
    std::uint8_t ver = 0;
    f.read(reinterpret_cast<char*>(&ver), 1);
    if (ver != kExprBinVersion) return false;
    std::uint32_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));
    out.specs.clear();
    out.keys.clear();
    out.specs.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        ExprSpec s;
        f.read(reinterpret_cast<char*>(&s.num_regs), sizeof(s.num_regs));
        std::uint32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        s.instrs.resize(n);
        for (auto& in : s.instrs)
        {
            std::uint8_t bytes[8];
            f.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
            in.op = bytes[0]; in.dst = bytes[1];
            in.a.kind = bytes[2]; in.a.idx = bytes[3];
            in.b.kind = bytes[4]; in.b.idx = bytes[5];
            in.c.kind = bytes[6]; in.c.idx = bytes[7];
        }
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        s.views.resize(n);
        for (auto& v : s.views)
        {
            f.read(reinterpret_cast<char*>(&v.kind), 1);
            f.read(reinterpret_cast<char*>(&v.negate_first_half), 1);
            f.read(reinterpret_cast<char*>(&v.param), sizeof(v.param));
        }
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        s.consts.resize(n);
        for (auto& c : s.consts)
            f.read(reinterpret_cast<char*>(&c), sizeof(c));
        std::uint8_t has_mm = 0;
        f.read(reinterpret_cast<char*>(&has_mm), 1);
        if (has_mm)
        {
            MatmulSpec mm;
            std::uint8_t mbytes[4];
            f.read(reinterpret_cast<char*>(mbytes), sizeof(mbytes));
            mm.a_input = mbytes[0]; mm.b_input = mbytes[1];
            mm.transA  = mbytes[2]; mm.transB  = mbytes[3];
            f.read(reinterpret_cast<char*>(&mm.k), sizeof(mm.k));
            s.matmul = mm;
        }
        out.add(s);
    }
    return true;
}

} // namespace nn::fused

#endif // NN_EXPR_REGISTRY_HPP
