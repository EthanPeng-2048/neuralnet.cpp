#ifndef NN_GLSL_GEN_HPP
#define NN_GLSL_GEN_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  glsl_gen.hpp — AOT 算子融合：ExprSpec → GLSL 计算 shader 源码
//
//  在构建期由 tools/gen_fused 调用：把 scan_exprs 收集的内联表达式结构
//  （ExprSpec，派生物）展开为单个融合 .comp 文件。
//    - 指令序列展开为直线浮点代码（无解释器开销，GPU 无分支）
//    - 视图（RotateHalf/RowMod）的索引映射内联（不物化中间张量）
//    - 常量以 push constant 传入（运行时可变）
//  产物走现有管线：glslc → .spv → cmake/embed_spirv.cmake → C++ 数组
//
//  本头仅供构建期生成器使用，运行时无需包含。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <sstream>
#include <string>

#include "expr_spec.hpp"

namespace nn
{

// ── 算子 → GLSL 表达式片段 ────────────────────────────────────────────────

// 二元/比较（输出 1.0/0.0 用于比较）
inline const char* glsl_binary_op(ExprOp op, bool& is_compare)
{
    is_compare = false;
    switch (op)
    {
    case ExprOp::Add: return "+";
    case ExprOp::Sub: return "-";
    case ExprOp::Mul: return "*";
    case ExprOp::Div: return "/";
    case ExprOp::Lt:  is_compare = true; return "<";
    case ExprOp::Le:  is_compare = true; return "<=";
    case ExprOp::Gt:  is_compare = true; return ">";
    case ExprOp::Ge:  is_compare = true; return ">=";
    case ExprOp::Eq:  is_compare = true; return "==";
    case ExprOp::Ne:  is_compare = true; return "!=";
    default: return "";
    }
}

inline const char* glsl_unary_op(ExprOp op)
{
    switch (op)
    {
    case ExprOp::Neg:   return "(-";    // 前缀，后接表达式与 ")"
    case ExprOp::Exp:   return "exp";
    case ExprOp::Log:   return "log";
    case ExprOp::Sqrt:  return "sqrt";
    case ExprOp::Rsqrt: return "inversesqrt";
    case ExprOp::Abs:   return "abs";
    case ExprOp::Tanh:  return "tanh";
    default: return "";
    }
}

// ── 视图 → 读取第 i 个元素的索引/表达式 ──────────────────────────────────
// 布局约定：row = i/cols, col = i%cols（与 C++ 端 eval_expr 一致）
// vp_slot：该视图的运行时参数槽位（仅 RowMod/RotateHalf 有效；在视图序列中
//   此前出现的运行时参数视图个数）。RowMod 周期 / RotateHalf 块大小从
//   push constant vpN 读取——同结构不同形状（不同 d_k）共享一个融合 shader。
inline void glsl_view_read(std::ostringstream& os,
                           const ExprView& v, std::uint32_t buf_id,
                           const std::string& idx_var,
                           const std::string& row_var, const std::string& col_var,
                           std::uint32_t vp_slot = 0)
{
    const std::string buf = "b" + std::to_string(buf_id);
    switch (v.kind)
    {
    default:
    case static_cast<uint8_t>(ExprViewKind::Linear):
        os << buf << "[" << idx_var << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::RotateHalf):
    {
        // block 为运行时视图参数（push constant vpN），形状无关融合
        const std::string blk = "(" + row_var + " / vp" + std::to_string(vp_slot) + ")";
        const std::string rl  = "(" + row_var + " % vp" + std::to_string(vp_slot) + ")";
        const std::string half = "(vp" + std::to_string(vp_slot) + " / 2u)";
        // 列主序索引 = 源行 * cols + 列号（cols 为 push constant 列数）
        const std::string hi  = "((" + blk + " * vp" + std::to_string(vp_slot) + " + "
                                + rl + " + " + half + ") * cols + " + col_var + ")";
        const std::string lo  = "((" + blk + " * vp" + std::to_string(vp_slot) + " + "
                                + rl + " - " + half + ") * cols + " + col_var + ")";
        const std::string neg = v.negate_first_half ? "-" : "";
        os << "((" << rl << " < " << half << ") ? "
           << neg << buf << "[" << hi << "] : " << buf << "[" << lo << "])";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowMod):
    {
        // 周期为运行时视图参数（push constant vpN），形状无关融合
        os << buf << "[(" << row_var << " % vp" << std::to_string(vp_slot) << ") * cols + "
           << col_var << "]";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowBroadcast):
        os << buf << "[" << row_var << "]";   // 输入 (rows,1)：每行一个值
        return;
    case static_cast<uint8_t>(ExprViewKind::ColBroadcast):
        os << buf << "[" << col_var << "]";   // 输入 (1,cols)：每列一个值
        return;
    }
}

// ── 生成器主入口：ExprSpec → GLSL 源码 ────────────────────────────────────
// 生成的 shader 绑定：输入 buffer binding=0..N-1，输出 binding=N；
// push constants: uint count, uint cols, [float c0..]（视图索引需要 cols）。
inline std::string generate_glsl(const std::string& name, const ExprSpec& spec)
{
    const std::size_t n_inputs = spec.views.size();
    std::ostringstream L;

    L << "// ── 自动生成（AOT 算子融合），请勿手动编辑 ──\n";
    L << "// 表达式: " << name << "\n";
    L << "#version 450\n\n";
    L << "layout(local_size_x = 256) in;\n\n";

    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { float b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { float bout[]; };\n\n";

    // push constants：count + cols（视图行/列）+ 运行时视图参数 vp + 常量
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n";
    L << "    uint cols;\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    L << "};\n\n";

    L << "void main()\n{\n";
    L << "    const uint i = gl_GlobalInvocationID.x;\n";
    L << "    if (i >= count) return;\n";
    L << "    const uint row = i / cols;\n";
    L << "    const uint col = i % cols;\n";

    // 输入读取变量（每输入缓存一次）；vp 槽 = 此前运行时参数视图个数
    for (std::size_t i = 0; i < n_inputs; ++i)
    {
        std::uint32_t vp = 0;
        for (std::size_t j = 0; j < i; ++j)
            if (expr_view_has_runtime_param(
                    static_cast<ExprViewKind>(spec.views[j].kind)))
                ++vp;
        L << "    const float v" << i << " = ";
        glsl_view_read(L, spec.views[i], static_cast<std::uint32_t>(i),
                       "i", "row", "col", vp);
        L << ";\n";
    }

    // 操作数求值
    auto operand = [&](const ExprOperand& op) -> std::string
    {
        switch (op.kind)
        {
        default:
        case static_cast<uint8_t>(ExprOperandKind::Input):
            return "v" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Reg):
        case static_cast<uint8_t>(ExprOperandKind::Fanout):  // Fanout 语义同 Reg
            return "r" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Const):
            return "c" + std::to_string(op.idx);
        }
    };

    // 指令展开（直线代码）
    for (const auto& ins : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string dst = "r" + std::to_string(ins.dst);
        const std::string a = operand(ins.a);
        switch (op)
        {
        case ExprOp::Add: case ExprOp::Sub: case ExprOp::Mul: case ExprOp::Div:
        {
            bool cmp = false;
            const char* s = glsl_binary_op(op, cmp);
            L << "    float " << dst << " = " << a << " " << s << " " << operand(ins.b) << ";\n";
            break;
        }
        case ExprOp::Max: case ExprOp::Min:
        {
            const char* s = (op == ExprOp::Max) ? "max" : "min";
            L << "    float " << dst << " = " << s << "(" << a << ", " << operand(ins.b) << ");\n";
            break;
        }
        case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
        case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
        {
            bool cmp = false;
            const char* s = glsl_binary_op(op, cmp);
            L << "    float " << dst << " = (" << a << " " << s << " " << operand(ins.b)
              << ") ? 1.0 : 0.0;\n";
            break;
        }
        case ExprOp::Neg:
            L << "    float " << dst << " = -(" << a << ");\n";
            break;
        case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
        case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
        {
            const char* s = glsl_unary_op(op);
            L << "    float " << dst << " = " << s << "(" << a << ");\n";
            break;
        }
        case ExprOp::Select:
            L << "    float " << dst << " = (" << a << " != 0.0) ? "
              << operand(ins.b) << " : " << operand(ins.c) << ";\n";
            break;
        default:
            return std::string{};  // 未知算子：让上层报错
        }
    }

    // 输出 = 最后一条指令的目标寄存器
    const std::uint32_t last_dst = spec.instrs.back().dst;
    L << "    bout[i] = r" << last_dst << ";\n";
    L << "}\n";
    return L.str();
}

// ═══════════════════════════════════════════════════════════════════════════
//  归约融合 shader 生成（M3）：含归约视图/归约指令的表达式
//
//  处理 expr_spec_reduce_axis(spec) >= 0 的表达式（全部归约同轴）：
//    - 工作组级归约：每个工作组（256 线程）协作处理一行（行归约）或一列
//      （列归约），shared memory 树形归约，随后输出该行/列的全部元素。
//    - 布局：bindings 输入 0..N-1 + 输出 N；
//      push constants: uint count, uint cols, uint rows, [float c0..]；
//      dispatch: 行归约 (rows,1,1)，列归约 (cols,1,1)。
//    - 归约槽：每个归约视图/归约指令占一个共享槽 s_red[slot][256]；
//      归约视图经 Input 操作数（槽），归约指令经 Reduce 操作数（槽）访问，
//      最终值读取 s_red[slot][0]（完成屏障后全线程可见）。
//    - 限制：归约槽 ≤ 8（共享内存 ≤ 8KB）；混合轴（行+列）不支持 → 返回 ""。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::string generate_glsl_reduce(
    const std::string& name, const ExprSpec& spec)
{
    const int axis = expr_spec_reduce_axis(spec);
    if (axis < 0)
        return {};  // 无归约（走逐元素生成器）或混合轴（不支持）
    const bool is_row = (axis == 0);

    const std::size_t n_inputs = spec.views.size();

    // ── 归约槽分配 ──
    // slot_of_view[k]：归约视图 k → 槽号；slot_of_instr[dst]：归约指令 dst → 槽号
    std::vector<int> slot_of_view(n_inputs, -1);
    std::vector<int> slot_of_instr(EXPR_MAX_REGS, -1);
    std::vector<bool> slot_is_max;
    int n_slots = 0;
    for (std::size_t k = 0; k < n_inputs; ++k)
    {
        const ExprView& v = spec.views[k];
        if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
            continue;
        slot_of_view[k] = n_slots++;
        slot_is_max.push_back(
            static_cast<ExprViewKind>(v.kind) == ExprViewKind::RowReduceMax ||
            static_cast<ExprViewKind>(v.kind) == ExprViewKind::ColReduceMax);
    }
    for (const auto& ins : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        if (!expr_op_is_reduce(op))
            continue;
        slot_of_instr[ins.dst] = n_slots++;
        slot_is_max.push_back(op == ExprOp::RowMax || op == ExprOp::ColMax);
    }
    if (n_slots == 0 || n_slots > 8)
        return {};  // 无归约或槽过多（共享内存超限）

    const bool last_is_reduce =
        expr_op_is_reduce(static_cast<ExprOp>(spec.instrs.back().op));

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · 归约 kernel），请勿手动编辑 ──\n";
    L << "// 表达式: " << name << "\n";
    L << "#version 450\n\n";
    L << "layout(local_size_x = 256) in;\n\n";
    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { float b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { float bout[]; };\n\n";
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n";
    L << "    uint cols;\n";
    L << "    uint rows;\n";
    L << "    uint vector_out;   // 1=输出归约向量（(rows,1)/(1,cols)），0=广播\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    L << "};\n\n";
    L << "shared float s_red[" << n_slots << "][256];\n\n";

    L << "void main()\n{\n";
    L << "    const uint tid = gl_LocalInvocationID.x;\n";
    L << "    const uint idx = gl_WorkGroupID.x;\n";
    L << (is_row ? "    if (idx >= rows) return;\n"
                 : "    if (idx >= cols) return;\n");

    // ── 操作数求值（当前元素由 row/col 变量给出） ──
    const auto operand = [&](const ExprOperand& op) -> std::string
    {
        switch (op.kind)
        {
        default:
        case static_cast<uint8_t>(ExprOperandKind::Reg):
        case static_cast<uint8_t>(ExprOperandKind::Fanout):
            return "r" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Const):
            return "c" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Reduce):
            return "s_red[" + std::to_string(slot_of_instr[op.idx]) + "][0]";
        case static_cast<uint8_t>(ExprOperandKind::Input):
        {
            const ExprView& v = spec.views[op.idx];
            const ExprViewKind vk = static_cast<ExprViewKind>(v.kind);
            if (expr_view_is_reduce(vk))
                return "s_red[" + std::to_string(slot_of_view[op.idx]) + "][0]";
            if (vk == ExprViewKind::RowBroadcast)
                return "b" + std::to_string(op.idx) + "[row]";
            if (vk == ExprViewKind::ColBroadcast)
                return "b" + std::to_string(op.idx) + "[col]";
            // Linear / RotateHalf / RowMod：索引映射内联；vp 槽 = 此前运行时参数视图个数
            std::ostringstream os;
            std::uint32_t vp = 0;
            for (std::size_t j = 0; j < op.idx; ++j)
                if (expr_view_has_runtime_param(
                        static_cast<ExprViewKind>(spec.views[j].kind)))
                    ++vp;
            glsl_view_read(os, v, static_cast<std::uint32_t>(op.idx),
                           "(row*cols + col)", "row", "col", vp);
            return os.str();
        }
        }
    };

    // 直线指令展开（跳过归约指令；归约结果经 s_red 访问）
    const auto emit_instrs = [&](std::size_t begin, std::size_t end)
    {
        for (std::size_t j = begin; j < end; ++j)
        {
            const ExprInstr& ins = spec.instrs[j];
            const ExprOp op = static_cast<ExprOp>(ins.op);
            if (expr_op_is_reduce(op))
                continue;
            const std::string dst = "r" + std::to_string(ins.dst);
            const std::string a = operand(ins.a);
            switch (op)
            {
            case ExprOp::Add: case ExprOp::Sub: case ExprOp::Mul: case ExprOp::Div:
            {
                bool cmp = false;
                const char* s = glsl_binary_op(op, cmp);
                L << "        float " << dst << " = " << a << " " << s << " "
                  << operand(ins.b) << ";\n";
                break;
            }
            case ExprOp::Max: case ExprOp::Min:
            {
                const char* s = (op == ExprOp::Max) ? "max" : "min";
                L << "        float " << dst << " = " << s << "(" << a
                  << ", " << operand(ins.b) << ");\n";
                break;
            }
            case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
            case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
            {
                bool cmp = false;
                const char* s = glsl_binary_op(op, cmp);
                L << "        float " << dst << " = (" << a << " " << s << " "
                  << operand(ins.b) << ") ? 1.0 : 0.0;\n";
                break;
            }
            case ExprOp::Neg:
                L << "        float " << dst << " = -(" << a << ");\n";
                break;
            case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
            case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
            {
                const char* s = glsl_unary_op(op);
                L << "        float " << dst << " = " << s << "(" << a << ");\n";
                break;
            }
            case ExprOp::Select:
                L << "        float " << dst << " = (" << a << " != 0.0) ? "
                  << operand(ins.b) << " : " << operand(ins.c) << ";\n";
                break;
            default:
                break;  // 归约指令已 continue
            }
        }
    };

    // 组合表达式：sum → "a + b"（中缀），max → "max(a, b)"（函数）
    const auto combine = [](bool is_max, const std::string& a, const std::string& b)
    {
        return is_max ? "max(" + a + ", " + b + ")" : "(" + a + " + " + b + ")";
    };

    // 归约树形归约（barrier 在每步内，最后一步后全线程可见 s_red[slot][0]）
    const auto emit_tree_reduce = [&](int slot, bool is_max)
    {
        const std::string s = "s_red[" + std::to_string(slot) + "]";
        L << "    barrier();\n";
        L << "    for (uint s = 128u; s > 0u; s >>= 1u) {\n";
        L << "        if (tid < s) " << s << "[tid] = "
          << combine(is_max, s + "[tid]", s + "[tid + s]") << ";\n";
        L << "        barrier();\n";
        L << "    }\n";
    };

    // 元素索引声明（行归约：固定行、循环列；列归约：循环行、固定列）
    const std::string loop_decl = is_row
        ? "    for (uint c = tid; c < cols; c += 256u) {\n        const uint row = idx;\n        const uint col = c;\n"
        : "    for (uint r = tid; r < rows; r += 256u) {\n        const uint row = r;\n        const uint col = idx;\n";
    const std::string out_idx = "row*cols + col";

    // ── 归约视图归约 pass ──
    for (std::size_t k = 0; k < n_inputs; ++k)
    {
        const ExprView& v = spec.views[k];
        if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
            continue;
        const int slot = slot_of_view[k];
        const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
        L << "\n    // 归约视图 " << k << " (槽 " << slot << ")\n";
        L << "    {\n";
        L << "        float acc = "
          << (is_max ? "uintBitsToFloat(0xff800000u)" : "0.0") << ";\n";
        L << loop_decl;
        L << "            acc = " << combine(is_max, "acc",
            "b" + std::to_string(k) + "[row*cols + col]") << ";\n";
        L << "        }\n";
        L << "        s_red[" << slot << "][tid] = acc;\n";
        L << "    }\n";
        emit_tree_reduce(slot, is_max);
    }

    // ── 归约指令 pass（按指令序，源可引用更早归约结果） ──
    // 寄存器声明直接放在 for 循环体内（无需额外 {} 包裹），使源寄存器在
    // combine 处可见；不同 pass 各自循环体作用域隔离，无重声明冲突。
    for (std::size_t ri = 0; ri < spec.instrs.size(); ++ri)
    {
        const ExprInstr& R = spec.instrs[ri];
        const ExprOp rop = static_cast<ExprOp>(R.op);
        if (!expr_op_is_reduce(rop))
            continue;
        const int slot = slot_of_instr[R.dst];
        const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
        const bool src_is_reg =
            (R.a.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
             R.a.kind == static_cast<uint8_t>(ExprOperandKind::Fanout));
        const std::string src = src_is_reg
            ? "r" + std::to_string(static_cast<int>(R.a.idx))
            : operand(R.a);
        L << "\n    // 归约指令 " << ri << " (槽 " << slot << ")\n";
        L << "    {\n";
        L << "        float acc = "
          << (is_max ? "uintBitsToFloat(0xff800000u)" : "0.0") << ";\n";
        L << loop_decl;
        emit_instrs(0, ri);
        L << "        acc = " << combine(is_max, "acc", src) << ";\n";
        L << "        }\n";
        L << "        s_red[" << slot << "][tid] = acc;\n";
        L << "    }\n";
        emit_tree_reduce(slot, is_max);
    }

    // ── 输出 pass ──
    // vector_out=1：输出归约向量（thread 0 写代表元素，输出形状由后端分配
    //   (rows,1)/(1,cols)）；vector_out=0：广播到 (rows,cols)。
    L << "\n    // 输出\n";
    L << "    if (vector_out == 1u) {\n";
    L << "        const uint row = " << (is_row ? "idx" : "0u") << ";\n";
    L << "        const uint col = " << (is_row ? "0u" : "idx") << ";\n";
    L << "        if (tid == 0u) {\n";
    if (last_is_reduce)
    {
        L << "            bout[" << (is_row ? "row" : "col") << "] = s_red["
          << slot_of_instr[spec.instrs.back().dst] << "][0];\n";
    }
    else
    {
        emit_instrs(0, spec.instrs.size());
        L << "            bout[" << (is_row ? "row" : "col") << "] = r"
          << static_cast<int>(spec.instrs.back().dst) << ";\n";
    }
    L << "        }\n";
    L << "        return;\n";
    L << "    }\n";
    L << loop_decl;
    if (last_is_reduce)
    {
        L << "        bout[" << out_idx << "] = s_red["
          << slot_of_instr[spec.instrs.back().dst] << "][0];\n";
    }
    else
    {
        emit_instrs(0, spec.instrs.size());
        L << "        bout[" << out_idx << "] = r"
          << static_cast<int>(spec.instrs.back().dst) << ";\n";
    }
    L << "    }\n";
    L << "}\n";
    return L.str();
}

} // namespace nn

#endif // NN_GLSL_GEN_HPP
