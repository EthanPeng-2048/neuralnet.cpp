#ifndef NN_EXPR_CPU_EMITTER_HPP
#define NN_EXPR_CPU_EMITTER_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_cpu_emitter.hpp — CPU 后端 emitter（IR-D 第二实现）
//
//  对应文档 `docs/11-ir-optimization.md` IR-D：一份 canonical IR → 多后端。
//  本 emitter 把 ExprSpec 展开为一段可直接嵌入 C++ 的**逐元素直线代码**
//  （独立函数，无解释器开销），作为 GlslEmitter 之外的第二实现，验证
//  emitter 抽象确实支持"同一 IR 多后端"。
//
//  生成的函数签名：
//      void <name>(const float* b0, ..., const float* bN-1, float* bout,
//                  std::size_t count, std::size_t cols,
//                  std::uint32_t vp0, ..., float c0, ...);
//    - 输入/输出为扁平 float 缓冲（row-major，与 Tensor 布局一致）
//    - vpN：运行时视图参数（RowMod 周期 / RotateHalf 块大小，形状无关）
//    - cN：常量池
//  归约表达式（generate_reduce）：本 emitter 暂不生成归约 kernel（返回空串），
//  归约语义由 CPU eval_expr 解释器承担（先正确后优化）。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <sstream>
#include <string>

#include "expr_spec.hpp"
#include "expr_emitter.hpp"

namespace nn
{

// ── 二元算子 → C++ 运算符片段 ───────────────────────────────────────────
inline const char* cpu_binary_op(ExprOp op)
{
    switch (op)
    {
    case ExprOp::Add: return "+";
    case ExprOp::Sub: return "-";
    case ExprOp::Mul: return "*";
    case ExprOp::Div: return "/";
    case ExprOp::Lt:  return "<";
    case ExprOp::Le:  return "<=";
    case ExprOp::Gt:  return ">";
    case ExprOp::Ge:  return ">=";
    case ExprOp::Eq:  return "==";
    case ExprOp::Ne:  return "!=";
    default: return "";
    }
}

// ── 视图 → C++ 索引表达式 ────────────────────────────────────────────────
// 布局约定与 GLSL/CPU eval_expr 一致：row = i/cols, col = i%cols。
// vp_slot：该视图的运行时参数槽位（同 glsl_gen 的 vp 约定）。
inline void cpu_view_read(std::ostringstream& os,
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
        // 与 GLSL 生成器一致的索引映射（block 为运行时视图参数 vpN）
        const std::string blk  = "(" + row_var + " / vp" + std::to_string(vp_slot) + ")";
        const std::string rl   = "(" + row_var + " % vp" + std::to_string(vp_slot) + ")";
        const std::string half = "(vp" + std::to_string(vp_slot) + " / 2u)";
        const std::string hi   = "((" + blk + " * vp" + std::to_string(vp_slot) + " + "
                                 + rl + " + " + half + ") * cols + " + col_var + ")";
        const std::string lo   = "((" + blk + " * vp" + std::to_string(vp_slot) + " + "
                                 + rl + " - " + half + ") * cols + " + col_var + ")";
        const std::string neg  = v.negate_first_half ? "-" : "";
        os << "((" << rl << " < " << half << ") ? "
           << neg << buf << "[" << hi << "] : " << buf << "[" << lo << "])";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowMod):
        os << buf << "[(" << row_var << " % vp" << std::to_string(vp_slot)
           << ") * cols + " << col_var << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::RowBroadcast):
        os << buf << "[" << row_var << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::ColBroadcast):
        os << buf << "[" << col_var << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::BatchMod):
        // S7：按批次取模索引：data[batch % vpN]（取模数 num_heads 为运行时视图参数）
        os << buf << "[batch % vp" << std::to_string(vp_slot) << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::BatchCol):
        // S7：按 (batch, col) 切片：data[batch*vpN + col]（每批列数 seq 为运行时视图参数）
        os << buf << "[batch * vp" << std::to_string(vp_slot) << " + " << col_var << "]";
        return;
    }
}

// ── 指令展开（直线代码，逐条输出到 os） ──────────────────────────────────
inline void cpu_emit_instrs(std::ostringstream& os, const ExprSpec& spec)
{
    const auto operand = [&](const ExprOperand& op) -> std::string
    {
        switch (op.kind)
        {
        default:
        case static_cast<uint8_t>(ExprOperandKind::Input):
            return "v" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Reg):
        case static_cast<uint8_t>(ExprOperandKind::Fanout):
            return "r" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Const):
            return "c" + std::to_string(op.idx);
        }
    };

    for (const auto& ins : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string dst = "r" + std::to_string(ins.dst);
        const std::string a = operand(ins.a);
        switch (op)
        {
        case ExprOp::Add: case ExprOp::Sub: case ExprOp::Mul: case ExprOp::Div:
        {
            const char* s = cpu_binary_op(op);
            os << "        " << dst << " = " << a << " " << s << " " << operand(ins.b) << ";\n";
            break;
        }
        case ExprOp::Max:
            os << "        " << dst << " = std::max(" << a << ", " << operand(ins.b) << ");\n";
            break;
        case ExprOp::Min:
            os << "        " << dst << " = std::min(" << a << ", " << operand(ins.b) << ");\n";
            break;
        case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
        case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
        {
            const char* s = cpu_binary_op(op);
            os << "        " << dst << " = (" << a << " " << s << " " << operand(ins.b)
               << ") ? 1.0f : 0.0f;\n";
            break;
        }
        case ExprOp::Neg:
            os << "        " << dst << " = -(" << a << ");\n";
            break;
        case ExprOp::Exp:
            os << "        " << dst << " = std::exp(" << a << ");\n";
            break;
        case ExprOp::Log:
            os << "        " << dst << " = std::log(" << a << ");\n";
            break;
        case ExprOp::Sqrt:
            os << "        " << dst << " = std::sqrt(" << a << ");\n";
            break;
        case ExprOp::Rsqrt:
            os << "        " << dst << " = 1.0f / std::sqrt(" << a << ");\n";
            break;
        case ExprOp::Abs:
            os << "        " << dst << " = std::fabs(" << a << ");\n";
            break;
        case ExprOp::Tanh:
            os << "        " << dst << " = std::tanh(" << a << ");\n";
            break;
        case ExprOp::Select:
            os << "        " << dst << " = (" << a << " != 0.0f) ? "
               << operand(ins.b) << " : " << operand(ins.c) << ";\n";
            break;
        default:
            os << "        // 不支持的算子（归约指令不在此 emitter 展开）\n";
            break;
        }
    }
}

// ── CpuEmitter — 生成 C++ 逐元素直线代码 ────────────────────────────────
class CpuEmitter final : public ExprEmitter
{
public:
    [[nodiscard]] std::string_view name() const noexcept override
    { return "cpu"; }

    [[nodiscard]] std::string generate(
        const std::string& name_, const ExprSpec& spec) override
    {
        // 归约语义不在本 emitter 展开（见 generate_reduce）
        if (expr_spec_reduce_axis(spec) >= 0)
            return {};
        const std::size_t n_inputs = spec.views.size();
        std::ostringstream L;

        L << "// ── 自动生成（CpuEmitter），请勿手动编辑 ──\n";
        L << "// 表达式: " << name_ << "\n";
        L << "inline void " << name_ << "(\n";
        for (std::size_t i = 0; i < n_inputs; ++i)
            L << "    const float* b" << i << ",  // 输入 " << i << "\n";
        L << "    float* bout,\n";
        L << "    std::size_t count, std::size_t cols";
        const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
        for (std::uint32_t i = 0; i < n_vp; ++i)
            L << ",\n    std::uint32_t vp" << i;
        for (std::size_t i = 0; i < spec.consts.size(); ++i)
            L << ",\n    float c" << i;
        L << ")\n{\n";
        L << "    for (std::size_t i = 0; i < count; ++i) {\n";
        L << "        const std::size_t row = i / cols;\n";
        L << "        const std::size_t col = i % cols;\n";

        // 输入读取变量（每输入缓存一次；vp 槽 = 此前运行时参数视图个数）
        for (std::size_t i = 0; i < n_inputs; ++i)
        {
            std::uint32_t vp = 0;
            for (std::size_t j = 0; j < i; ++j)
                if (expr_view_has_runtime_param(
                        static_cast<ExprViewKind>(spec.views[j].kind)))
                    ++vp;
            L << "        const float v" << i << " = ";
            cpu_view_read(L, spec.views[i], static_cast<std::uint32_t>(i),
                          "i", "row", "col", vp);
            L << ";\n";
        }

        // 寄存器声明（先声明后赋值，兼容 IR-B liveness 复用同号寄存器）
        if (spec.num_regs > 0)
        {
            L << "        float r0";
            for (std::uint32_t r = 1; r < spec.num_regs; ++r) L << ", r" << r;
            L << ";\n";
        }

        cpu_emit_instrs(L, spec);

        const std::uint32_t last_dst = spec.instrs.back().dst;
        L << "        bout[i] = r" << last_dst << ";\n";
        L << "    }\n";
        L << "}\n";
        return L.str();
    }

    [[nodiscard]] std::string generate_reduce(
        const std::string&, const ExprSpec&) override
    {
        // 归约 kernel 由 CPU eval_expr 解释器承担；本 emitter 暂不生成
        return {};
    }
};

// 登记到 emitter 注册表
inline const bool kCpuEmitterRegistered =
    emitter_registry::register_backend("cpu",
        []() -> ExprEmitter* { return new CpuEmitter(); });

} // namespace nn

#endif // NN_EXPR_CPU_EMITTER_HPP
