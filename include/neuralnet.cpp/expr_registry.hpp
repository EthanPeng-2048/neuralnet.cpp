#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_registry.hpp — 可融合表达式注册表（AOT 收集的两端共用）
//
//  AOT 收集架构（表达式只在 Layer 里，别处一律不出现）：
//    ① 构建期 scan_exprs 用假张量 dry-run Layer 的 forward/backward，
//       每个 dsl::compute / compute_reduce / compute_into 在记录模式下把折叠出的 ExprSpec
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
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "core_file.hpp"   // 二进制 POD 读写（cast 边界收敛点，docs/17 §2.1）
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
//   magic "NNEXP" (5B) + version (u8=3)
//   count (u32)
//   每 spec：num_regs(u32)
//            instrs: count(u32) × {op(u8) dst(u8) a.kind a.idx b.kind b.idx c.kind c.idx}
//            views:  count(u32) × {kind(u8) negate(u8) param(u32)}
//            consts: count(u32) × Scalar
//            rparams: count(u32) × Scalar   （v3 起支持运行时标量参数）
//            matmul: has(u8=0/1)；1 时 {a_input(u8) b_input(u8) transA(u8)
//                    transB(u8) k(u32) batch(u32)}
//            fold:   has(u8=0/1)（v6 起）；1 时 {num_state(u8), k(u32),
//                    inits: count(u32) × Scalar,
//                    body: count(u32) × 指令(8B),
//                    finalize: count(u32) × 指令(8B)
//                    -- v7 追加：vec_state_len(u32),
//                    matmul: has(u8)；1 时 {a,b,tA,tB(4B) k(u32) batch(u32)},
//                    vecacc: has(u8)；1 时 {vec_state,weight_reg,b_input,
//                           scale_reg,has_scale(5B)}}
//  v2 起支持 matmul 段（v1 无 matmul，读 v1 等价 has=0）；v3 起支持 rparams。
inline constexpr std::uint8_t kExprBinVersion = 7;  // v5：MatmulSpec 补 batch；v6：FoldSpec；v7：FoldSpec 双域字段（vec_state_len/matmul/vecacc——丢段=结构损坏）

[[nodiscard]] inline bool write_registry(const std::string& path,
                                         const ExprRegistry& reg)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("NNEXP", 5);
    if (!write_pod(f, kExprBinVersion)) return false;
    const std::uint32_t count = static_cast<std::uint32_t>(reg.specs.size());
    if (!write_pod(f, count)) return false;
    for (const auto& s : reg.specs)
    {
        if (!write_pod(f, s.num_regs)) return false;
        std::uint32_t n = static_cast<std::uint32_t>(s.instrs.size());
        if (!write_pod(f, n)) return false;
        for (const auto& in : s.instrs)
        {
            std::uint8_t bytes[8] = { in.op, in.dst, in.a.kind, in.a.idx,
                                      in.b.kind, in.b.idx, in.c.kind, in.c.idx };
            if (!write_pod_span(f, std::span(bytes, 8))) return false;
        }
        n = static_cast<std::uint32_t>(s.views.size());
        if (!write_pod(f, n)) return false;
        for (const auto& v : s.views)
        {
            if (!write_pod(f, v.kind)) return false;
            if (!write_pod(f, v.negate_first_half)) return false;
            if (!write_pod(f, v.param)) return false;
            if (!write_pod(f, v.param2)) return false;  // v4：RowAccess offset
        }
        n = static_cast<std::uint32_t>(s.consts.size());
        if (!write_pod(f, n)) return false;
        for (const auto& c : s.consts)
            if (!write_pod(f, c)) return false;
        n = static_cast<std::uint32_t>(s.rparams.size());
        if (!write_pod(f, n)) return false;
        for (const auto& r : s.rparams)
            if (!write_pod(f, r)) return false;
        const std::uint8_t has_mm = s.matmul ? 1 : 0;
        if (!write_pod(f, has_mm)) return false;
        if (s.matmul)
        {
            std::uint8_t mbytes[4] = { s.matmul->a_input, s.matmul->b_input,
                                       s.matmul->transA, s.matmul->transB };
            if (!write_pod_span(f, std::span(mbytes, 4))) return false;
            if (!write_pod(f, s.matmul->k)) return false;
            if (!write_pod(f, s.matmul->batch)) return false;  // v5：batch 必须持久化
        }
        // v6：fold 段（P-C1）——丢段会让读回 spec 变"空指令表无段"，
        //     多个 fold key 撞车合并 + gen_fused 对空表 UB 崩溃
        const auto write_instr_seq = [&](const std::vector<ExprInstr>& seq) -> bool
        {
            std::uint32_t m = static_cast<std::uint32_t>(seq.size());
            if (!write_pod(f, m)) return false;
            for (const auto& in : seq)
            {
                std::uint8_t bytes[8] = { in.op, in.dst, in.a.kind, in.a.idx,
                                          in.b.kind, in.b.idx, in.c.kind, in.c.idx };
                if (!write_pod_span(f, std::span(bytes, 8))) return false;
            }
            return true;
        };
        const std::uint8_t has_fold = s.fold ? 1 : 0;
        if (!write_pod(f, has_fold)) return false;
        if (s.fold)
        {
            if (!write_pod(f, s.fold->num_state)) return false;
            if (!write_pod(f, s.fold->k)) return false;
            std::uint32_t m = static_cast<std::uint32_t>(s.fold->inits.size());
            if (!write_pod(f, m)) return false;
            for (const auto& iv : s.fold->inits)
                if (!write_pod(f, iv)) return false;
            if (!write_instr_seq(s.fold->body)) return false;
            if (!write_instr_seq(s.fold->finalize)) return false;
            // v7：双域字段（vec_state_len + 自带 matmul 段 + vecacc）——
            //   与 struct 声明序一致；漏写=读回结构损坏（同 v6 教训）
            if (!write_pod(f, s.fold->vec_state_len)) return false;
            const std::uint8_t has_fmm = s.fold->matmul ? 1 : 0;
            if (!write_pod(f, has_fmm)) return false;
            if (s.fold->matmul)
            {
                std::uint8_t mbytes[4] = { s.fold->matmul->a_input,
                                           s.fold->matmul->b_input,
                                           s.fold->matmul->transA,
                                           s.fold->matmul->transB };
                if (!write_pod_span(f, std::span(mbytes, 4))) return false;
                if (!write_pod(f, s.fold->matmul->k)) return false;
                if (!write_pod(f, s.fold->matmul->batch)) return false;
            }
            const std::uint8_t has_va = s.fold->vecacc ? 1 : 0;
            if (!write_pod(f, has_va)) return false;
            if (s.fold->vecacc)
            {
                const VecAccSpec& va = *s.fold->vecacc;
                std::uint8_t vbytes[5] = { va.vec_state, va.weight_reg,
                                           va.b_input, va.scale_reg,
                                           va.has_scale };
                if (!write_pod_span(f, std::span(vbytes, 5))) return false;
            }
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
    if (!read_pod(f, ver)) return false;
    if (ver != kExprBinVersion) return false;
    std::uint32_t count = 0;
    if (!read_pod(f, count)) return false;
    out.specs.clear();
    out.keys.clear();
    out.specs.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        ExprSpec s;
        if (!read_pod(f, s.num_regs)) return false;
        std::uint32_t n = 0;
        if (!read_pod(f, n)) return false;
        s.instrs.resize(n);
        for (auto& in : s.instrs)
        {
            std::uint8_t bytes[8];
            if (!read_pod_span(f, std::span(bytes, 8))) return false;
            in.op = bytes[0]; in.dst = bytes[1];
            in.a.kind = bytes[2]; in.a.idx = bytes[3];
            in.b.kind = bytes[4]; in.b.idx = bytes[5];
            in.c.kind = bytes[6]; in.c.idx = bytes[7];
        }
        if (!read_pod(f, n)) return false;
        s.views.resize(n);
        for (auto& v : s.views)
        {
            if (!read_pod(f, v.kind)) return false;
            if (!read_pod(f, v.negate_first_half)) return false;
            if (!read_pod(f, v.param)) return false;
            if (!read_pod(f, v.param2)) return false;  // v4：RowAccess offset
        }
        if (!read_pod(f, n)) return false;
        s.consts.resize(n);
        for (auto& c : s.consts)
            if (!read_pod(f, c)) return false;
        if (!read_pod(f, n)) return false;
        s.rparams.resize(n);
        for (auto& r : s.rparams)
            if (!read_pod(f, r)) return false;
        std::uint8_t has_mm = 0;
        if (!read_pod(f, has_mm)) return false;
        if (has_mm)
        {
            MatmulSpec mm;
            std::uint8_t mbytes[4];
            if (!read_pod_span(f, std::span(mbytes, 4))) return false;
            mm.a_input = mbytes[0]; mm.b_input = mbytes[1];
            mm.transA  = mbytes[2]; mm.transB  = mbytes[3];
            if (!read_pod(f, mm.k)) return false;
            if (!read_pod(f, mm.batch)) return false;  // v5：batch 必须读回
            s.matmul = mm;
        }
        // v6：fold 段读回（与 write 对称——不对称会让后续 spec 错位读废）
        std::uint8_t has_fold = 0;
        if (!read_pod(f, has_fold)) return false;
        if (has_fold)
        {
            FoldSpec fs;
            if (!read_pod(f, fs.num_state)) return false;
            if (!read_pod(f, fs.k)) return false;
            std::uint32_t m = 0;
            if (!read_pod(f, m)) return false;
            fs.inits.resize(m);
            for (auto& iv : fs.inits)
                if (!read_pod(f, iv)) return false;
            const auto read_instr_seq = [&](std::vector<ExprInstr>& seq) -> bool
            {
                std::uint32_t c = 0;
                if (!read_pod(f, c)) return false;
                seq.resize(c);
                for (auto& in : seq)
                {
                    std::uint8_t bytes[8];
                    if (!read_pod_span(f, std::span(bytes, 8))) return false;
                    in.op = bytes[0]; in.dst = bytes[1];
                    in.a.kind = bytes[2]; in.a.idx = bytes[3];
                    in.b.kind = bytes[4]; in.b.idx = bytes[5];
                    in.c.kind = bytes[6]; in.c.idx = bytes[7];
                }
                return true;
            };
            if (!read_instr_seq(fs.body)) return false;
            if (!read_instr_seq(fs.finalize)) return false;
            // v7：双域字段读回（与 write 对称——不对称=后续 spec 错位读废）
            if (!read_pod(f, fs.vec_state_len)) return false;
            std::uint8_t has_fmm = 0;
            if (!read_pod(f, has_fmm)) return false;
            if (has_fmm)
            {
                MatmulSpec mm2;
                std::uint8_t mbytes[4];
                if (!read_pod_span(f, std::span(mbytes, 4))) return false;
                mm2.a_input = mbytes[0]; mm2.b_input = mbytes[1];
                mm2.transA  = mbytes[2]; mm2.transB = mbytes[3];
                if (!read_pod(f, mm2.k)) return false;
                if (!read_pod(f, mm2.batch)) return false;
                fs.matmul = mm2;
            }
            std::uint8_t has_va = 0;
            if (!read_pod(f, has_va)) return false;
            if (has_va)
            {
                VecAccSpec va;
                std::uint8_t vbytes[5];
                if (!read_pod_span(f, std::span(vbytes, 5))) return false;
                va.vec_state  = vbytes[0]; va.weight_reg = vbytes[1];
                va.b_input    = vbytes[2]; va.scale_reg  = vbytes[3];
                va.has_scale  = vbytes[4];
                fs.vecacc = va;
            }
            s.fold = fs;
        }
        out.add(s);
    }
    return true;
}

} // namespace nn::fused

