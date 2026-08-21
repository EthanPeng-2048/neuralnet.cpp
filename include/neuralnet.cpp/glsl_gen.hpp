#ifndef NN_GLSL_GEN_HPP
#define NN_GLSL_GEN_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  glsl_gen.hpp — AOT 算子融合：ExprSpec → GLSL 计算 shader 源码
//
//  在构建期由 tools/gen_fused 调用：把 C++ 端定义的表达式（ExprSpec，
//  单一事实来源见 fused_exprs.hpp）展开为单个融合 .comp 文件。
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
inline void glsl_view_read(std::ostringstream& os,
                           const ExprView& v, std::uint32_t buf_id,
                           const std::string& idx_var,
                           const std::string& row_var, const std::string& col_var)
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
        const std::uint32_t block = v.param;
        const std::uint32_t half = block / 2;
        const std::string blk = "(" + row_var + " / " + std::to_string(block) + "u)";
        const std::string rl  = "(" + row_var + " % " + std::to_string(block) + "u)";
        // 列主序索引 = 源行 * cols + 列号（cols 为 push constant 列数）
        const std::string hi  = "((" + blk + " * " + std::to_string(block) + "u + "
                                + rl + " + " + std::to_string(half) + "u) * cols + " + col_var + ")";
        const std::string lo  = "((" + blk + " * " + std::to_string(block) + "u + "
                                + rl + " - " + std::to_string(half) + "u) * cols + " + col_var + ")";
        const std::string neg = v.negate_first_half ? "-" : "";
        os << "((" << rl << " < " << std::to_string(half) << "u) ? "
           << neg << buf << "[" << hi << "] : " << buf << "[" << lo << "])";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowMod):
    {
        const std::uint32_t mod = v.param;
        os << buf << "[(" << row_var << " % " << std::to_string(mod) << "u) * cols + "
           << col_var << "]";
        return;
    }
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

    // push constants：count + cols（视图行/列）+ 常量
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n";
    L << "    uint cols;\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    L << "};\n\n";

    L << "void main()\n{\n";
    L << "    const uint i = gl_GlobalInvocationID.x;\n";
    L << "    if (i >= count) return;\n";
    L << "    const uint row = i / cols;\n";
    L << "    const uint col = i % cols;\n";

    // 输入读取变量（每输入缓存一次）
    for (std::size_t i = 0; i < n_inputs; ++i)
    {
        L << "    const float v" << i << " = ";
        glsl_view_read(L, spec.views[i], static_cast<std::uint32_t>(i), "i", "row", "col");
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

} // namespace nn

#endif // NN_GLSL_GEN_HPP
