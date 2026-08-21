#ifndef NN_EXPR_SPEC_HPP
#define NN_EXPR_SPEC_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_spec.hpp — 逐元素表达式 DSL：数据结构定义
//
//  纯数据结构，无引擎/执行器依赖（仅 <cstdint>/<vector> 与 config.hpp）。
//  ComputeEngine 接口（compute_engine.hpp）依赖本头声明 eval_expr；
//  执行器（expr_eval.hpp）依赖本头 + ComputeEngine 原语。
//
//  动机：将"函数式逐元素原语"升级为统一的表达式描述，使单行内多次计算
//  （如 RoPE 的 q*cos + rotate(q)*sin）只遍历一次、少产生中间 Tensor。
//  ExprSpec 是运行时可序列化的稳定表示：未来替换执行策略（统一 VM /
//  运行时 JIT 生成 shader）时，DSL、Layer、引擎接口均保持不变。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <vector>

#include "config.hpp"
#include "core_errors.hpp"

namespace nn
{

// ── 算子 ───────────────────────────────────────────────────────────────────
// 覆盖现有逐元素原语的完整语义 + 比较/选择，可表达任意逐元素融合。
enum class ExprOp : uint8_t
{
    // 一元
    Neg   = 0,  // -x
    Exp   = 1,  // e^x
    Log   = 2,  // ln(x)
    Sqrt  = 3,  // √x
    Rsqrt = 4,  // 1/√x
    Abs   = 5,  // |x|
    Tanh  = 6,  // tanh(x)
    // 二元
    Add   = 7,  // a + b
    Sub   = 8,  // a - b
    Mul   = 9,  // a * b
    Div   = 10, // a / b
    Max   = 11, // max(a, b)
    Min   = 12, // min(a, b)
    // 比较（输出 1.0 / 0.0 到寄存器，供 Select 使用）
    Lt    = 13, // a <  b
    Le    = 14, // a <= b
    Gt    = 15, // a >  b
    Ge    = 16, // a >= b
    Eq    = 17, // a == b
    Ne    = 18, // a != b
    // 选择：dst = (a != 0) ? b : c
    Select = 19,
};

// ── 操作数 ─────────────────────────────────────────────────────────────────
enum class ExprOperandKind : uint8_t
{
    Reg    = 0,  // 引用前驱指令的输出寄存器
    Input  = 1,  // 引用第 idx 个输入 Tensor（按 views[idx] 视图访问）
    Const  = 2,  // 引用常量池第 idx 项
    Fanout = 3,  // 引用前驱结果寄存器（语义同 Reg，显式表达 fanout 语义）
};

// 操作数：2 字节（kind + idx），指令布局紧凑、可序列化、未来可入 push constant
struct ExprOperand
{
    uint8_t kind = 0;  // ExprOperandKind
    uint8_t idx  = 0;  // 寄存器号 / 输入下标 / 常量池下标

    friend bool operator==(const ExprOperand&, const ExprOperand&) = default;
};

// ── 输入视图（索引映射）────────────────────────────────────────────────────
// 视图描述"如何按 (row, col) 访问一个输入 buffer"，使布局操作（行交换、
// 行广播）可以融入表达式而**不物化**中间张量。这是 RoPE 等场景省临时
// Tensor 的关键；未来 VM 中视图即廉价索引映射，无需物化。
enum class ExprViewKind : uint8_t
{
    Linear       = 0,  // data[r*cols + c]（默认）
    RotateHalf   = 1,  // 按 param(block_rows) 分块，块内前后半行交换；
                       //   negate_first_half=true 时，目标行 rl<block_rows/2
                       //   读取源 rl+block_rows/2 并取负（LLaMA 式 rotate_half）
    RowMod       = 2,  // 行取模广播：读取 data[(r % param)*cols + c]
                       //   用于 cos/sin 频率表 ((d_k,seq)) 平铺到 (batch*H*d_k,seq)
};

struct ExprView
{
    uint8_t kind = 0;                 // ExprViewKind
    uint8_t negate_first_half = 0;    // 仅 RotateHalf 有效
    uint32_t param = 0;               // RotateHalf: block_rows；RowMod: modulo

    friend bool operator==(const ExprView&, const ExprView&) = default;
};

// ── 指令 ───────────────────────────────────────────────────────────────────
struct ExprInstr
{
    uint8_t op   = 0;  // ExprOp
    uint8_t dst  = 0;  // 目标寄存器号
    ExprOperand a;     // 源操作数 1
    ExprOperand b;     // 源操作数 2
    ExprOperand c;     // 仅 Select 使用

    friend bool operator==(const ExprInstr&, const ExprInstr&) = default;
};

// ── 表达式规格（运行时可序列化，跨后端）─────────────────────────────────
// 语义：
//   - 所有输入 Tensor 同形状 (rows, cols)，views[i] 与 inputs[i] 一一对应
//   - 顺序执行 instrs，每指令结果写入 regs[dst]（寄存器数组 num_regs 个）
//   - 输出 = instrs.back().dst 寄存器的值，逐元素写入 (rows, cols) 输出
// 上限（校验保证，亦约束 GPU 路径资源）：
//   - 指令 ≤ 64，寄存器 ≤ 16，输入 ≤ 8，常量 ≤ 16
struct ExprSpec
{
    std::vector<ExprInstr>      instrs;
    std::vector<ExprView>       views;   // 与 inputs 一一对应
    std::vector<Scalar>         consts;
    std::uint32_t               num_regs = 0;
};

// ── 表达式规格相等比较（GPU AOT 匹配用）────────────────────────────────
// 两个 ExprSpec 相等 ⟺ 指令序列、输入视图、常量池、寄存器数全部一致。
// 用于运行时 eval_expr 判断"该表达式是否有预生成融合 shader"。
[[nodiscard]] inline bool expr_spec_equal(const ExprSpec& a, const ExprSpec& b)
{
    return a.instrs == b.instrs && a.views == b.views &&
           a.consts == b.consts && a.num_regs == b.num_regs;
}

// ── 上限（GPU 资源 / 校验共用）───────────────────────────────────────────
inline constexpr std::size_t EXPR_MAX_INPUTS = 8;
inline constexpr std::size_t EXPR_MAX_CONSTS = 16;
inline constexpr std::size_t EXPR_MAX_REGS   = 16;
inline constexpr std::size_t EXPR_MAX_INSTRS = 64;

// ── 校验 ──────────────────────────────────────────────────────────────────
// 返回 unexpected(Error) 描述首个非法点。Layer 侧可在提交前调用以尽早报错。
[[nodiscard]] inline Result<void> validate_expr_spec(const ExprSpec& spec,
                                                     std::size_t num_inputs)
{
    if (spec.instrs.empty())
        return std::unexpected(Error{"validate_expr_spec: empty instruction list"});
    if (spec.instrs.size() > EXPR_MAX_INSTRS)
        return std::unexpected(Error{"validate_expr_spec: too many instructions"});
    if (spec.num_regs == 0 || spec.num_regs > EXPR_MAX_REGS)
        return std::unexpected(Error{"validate_expr_spec: invalid num_regs"});
    if (num_inputs > EXPR_MAX_INPUTS)
        return std::unexpected(Error{"validate_expr_spec: too many inputs"});
    if (spec.consts.size() > EXPR_MAX_CONSTS)
        return std::unexpected(Error{"validate_expr_spec: too many constants"});
    if (spec.views.size() != num_inputs)
        return std::unexpected(Error{"validate_expr_spec: views count != inputs count"});
    for (const auto& ins : spec.instrs)
    {
        if (ins.dst >= spec.num_regs)
            return std::unexpected(Error{"validate_expr_spec: dst reg out of range"});
        for (const ExprOperand* op : {&ins.a, &ins.b, &ins.c})
        {
            if ((op->kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                 op->kind == static_cast<uint8_t>(ExprOperandKind::Fanout)) && op->idx >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: src reg out of range"});
            if (op->kind == static_cast<uint8_t>(ExprOperandKind::Input) && op->idx >= num_inputs)
                return std::unexpected(Error{"validate_expr_spec: input index out of range"});
            if (op->kind == static_cast<uint8_t>(ExprOperandKind::Const) && op->idx >= spec.consts.size())
                return std::unexpected(Error{"validate_expr_spec: const index out of range"});
        }
    }
    return {};
}

// ── 便捷构造（避免 Layer 写魔法数）──────────────────────────────────────
namespace expr
{
    inline constexpr ExprOperand reg(std::uint8_t r)   { return {0, r}; }
    inline constexpr ExprOperand input(std::uint8_t i) { return {1, i}; }
    inline constexpr ExprOperand cst(std::uint8_t c)   { return {2, c}; }
    inline constexpr ExprOperand fanout(std::uint8_t r){ return {3, r}; }
    inline constexpr ExprView linear()                 { return {0, 0, 0}; }
    inline constexpr ExprView rotate_half(std::uint32_t block_rows, bool negate_first_half = true)
    { return {1, negate_first_half ? std::uint8_t{1} : std::uint8_t{0}, block_rows}; }
    inline constexpr ExprView row_mod(std::uint32_t modulo)
    { return {2, 0, modulo}; }
} // namespace expr

} // namespace nn

#endif // NN_EXPR_SPEC_HPP
