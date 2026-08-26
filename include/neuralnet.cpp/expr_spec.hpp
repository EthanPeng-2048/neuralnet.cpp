#ifndef NN_EXPR_SPEC_HPP
#define NN_EXPR_SPEC_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_spec.hpp — 逐元素表达式 DSL：数据结构定义
//
//  纯数据结构，无引擎/执行器依赖（仅 <cstdint>/<vector> 与 config.hpp）。
//  ComputeEngine 接口（compute_engine.hpp）依赖本头声明 eval_expr；
//  表达式求值入口（expr_dsl.hpp 的 dsl::compute / 各引擎 eval_expr）依赖本头。
//
//  动机：将"函数式逐元素原语"升级为统一的表达式描述，使单行内多次计算
//  （如 RoPE 的 q*cos + rotate(q)*sin）只遍历一次、少产生中间 Tensor。
//  ExprSpec 是运行时可序列化的稳定表示：未来替换执行策略（统一 VM /
//  运行时 JIT 生成 shader）时，DSL、Layer、引擎接口均保持不变。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core_config.hpp"
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
    // ── 新增：归约指令（dst 为隐式"每行/每列一个标量"的归约向量，
    //          仅能经 ExprOperandKind::Reduce 操作数按行/列广播访问）──
    ColSum = 20,  // dst[c] = Σ_r a[r][c]
    ColMax = 21,  // dst[c] = max_r a[r][c]
    RowSum = 22,  // dst[r] = Σ_c a[r][c]
    RowMax = 23,  // dst[r] = max_c a[r][c]
};

// ── 归约语义辅助（引擎/校验共用）──────────────────────────────────────
// 归约指令：dst 是"每行/每列一个标量"的隐式向量，供 Reduce 操作数广播引用。
[[nodiscard]] inline constexpr bool expr_op_is_reduce(ExprOp op) noexcept
{
    return op == ExprOp::ColSum || op == ExprOp::ColMax ||
           op == ExprOp::RowSum || op == ExprOp::RowMax;
}
// 该归约指令产出的向量方向：true=按列归约 → (1, cols)，false=按行归约 → (rows, 1)
[[nodiscard]] inline constexpr bool expr_op_reduces_cols(ExprOp op) noexcept
{
    return op == ExprOp::ColSum || op == ExprOp::ColMax;
}
// 指令实际使用的操作数个数（未用的 b/c 保持默认 {0,0}，不可当作 Reg(0) 引用）
[[nodiscard]] inline std::size_t expr_instr_num_operands(ExprOp op) noexcept
{
    switch (op)
    {
    case ExprOp::Neg: case ExprOp::Exp: case ExprOp::Log:
    case ExprOp::Sqrt: case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
    case ExprOp::ColSum: case ExprOp::ColMax: case ExprOp::RowSum: case ExprOp::RowMax:
        return 1;
    case ExprOp::Select:
        return 3;
    default:
        return 2;
    }
}

// ── 操作数 ─────────────────────────────────────────────────────────────────
enum class ExprOperandKind : uint8_t
{
    Reg    = 0,  // 引用前驱指令的输出寄存器
    Input  = 1,  // 引用第 idx 个输入 Tensor（按 views[idx] 视图访问）
    Const  = 2,  // 引用常量池第 idx 项
    Fanout = 3,  // 引用前驱结果寄存器（语义同 Reg，显式表达 fanout 语义）
    // ── 新增：引用一个"行/列归约结果"，自动按行或按列广播 ──
    Reduce = 4,  // 引用某归约指令 dst（idx 指向归约指令的目标寄存器号）
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
    // ── 新增：归约视图（该输入被归约为每列/每行一个标量，自动广播）──
    // 求值期代表一个"每行/每列一个标量"的广播向量：
    //   对输出元素 (r,c)，行归约读取向量在 r 处的标量，列归约读取 c 处标量。
    ColReduceSum = 3,  // 该输入按列求和 → (1, cols)
    ColReduceMax = 4,  // 该输入按列求 max → (1, cols)
    RowReduceSum = 5,  // 该输入按行求和 → (rows, 1)
    RowReduceMax = 6,  // 该输入按行求 max → (rows, 1)
    // ── 新增：广播视图（输入本身已是 (rows,1)/(1,cols) 小向量，按行/列广播）──
    RowBroadcast = 7,  // 输入 (rows, 1)：读 b[r]（gamma/beta 等逐行参数）
    ColBroadcast = 8,  // 输入 (1, cols)：读 b[c]（std_inv 等逐列统计量）
};

// ── 归约视图辅助（引擎/校验共用）──────────────────────────────────────
[[nodiscard]] inline constexpr bool expr_view_is_reduce(ExprViewKind k) noexcept
{
    return k == ExprViewKind::ColReduceSum || k == ExprViewKind::ColReduceMax ||
           k == ExprViewKind::RowReduceSum || k == ExprViewKind::RowReduceMax;
}
// 该归约视图按行归约（输出 (rows,1)，按 r 广播）；false=按列归约（输出 (1,cols)，按 c 广播）
[[nodiscard]] inline constexpr bool expr_view_reduces_rows(ExprViewKind k) noexcept
{
    return k == ExprViewKind::RowReduceSum || k == ExprViewKind::RowReduceMax;
}
[[nodiscard]] inline constexpr bool expr_view_is_broadcast(ExprViewKind k) noexcept
{
    return k == ExprViewKind::RowBroadcast || k == ExprViewKind::ColBroadcast;
}

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

// ── 运行时视图参数（形状无关融合的关键）───────────────────────────────
// RowMod（周期）与 RotateHalf（块大小）的 param 是**运行时形状数据**（如
// RoPE 的 d_k），不是表达式结构：同结构不同 param（不同 d_k）应共享一个
// 融合 shader。因此：
//   - expr_spec_key **不**把这两个 param 折进 key（结构相同 → 同 key）
//   - glsl_gen 把它们作为 push constant（vp 槽）读取，dispatch 时按实际
//     spec 填充 → 一个 shader 适配所有形状（任何 d_k）
[[nodiscard]] inline constexpr bool expr_view_has_runtime_param(ExprViewKind k) noexcept
{
    return k == ExprViewKind::RowMod || k == ExprViewKind::RotateHalf;
}
// 该 spec 的运行时视图参数个数（= 融合 shader 的 push constant vp 槽位数）
[[nodiscard]] inline std::uint32_t expr_spec_runtime_view_param_count(
    const ExprSpec& s) noexcept
{
    std::uint32_t n = 0;
    for (const auto& v : s.views)
        if (expr_view_has_runtime_param(static_cast<ExprViewKind>(v.kind)))
            ++n;
    return n;
}
// 按视图顺序提取运行时视图参数（RowMod 周期 / RotateHalf 块大小），
// 运行时 eval_expr 用它填充融合 shader 的 push constant vp 槽。
[[nodiscard]] inline std::vector<std::uint32_t> expr_spec_runtime_view_params(
    const ExprSpec& s)
{
    std::vector<std::uint32_t> out;
    for (const auto& v : s.views)
        if (expr_view_has_runtime_param(static_cast<ExprViewKind>(v.kind)))
            out.push_back(v.param);
    return out;
}

// ── 表达式归约轴（融合 shader 生成用）────────────────────────────────────
// 返回 -1=无归约（逐元素）、0=全部行归约、1=全部列归约、-2=混合（不支持单 kernel 融合）
[[nodiscard]] inline int expr_spec_reduce_axis(const ExprSpec& s)
{
    int axis = -1;
    for (const auto& v : s.views)
    {
        if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
            continue;
        const int a = expr_view_reduces_rows(static_cast<ExprViewKind>(v.kind)) ? 0 : 1;
        if (axis == -1) axis = a;
        else if (axis != a) return -2;
    }
    for (const auto& in : s.instrs)
    {
        if (!expr_op_is_reduce(static_cast<ExprOp>(in.op)))
            continue;
        const int a = expr_op_reduces_cols(static_cast<ExprOp>(in.op)) ? 1 : 0;
        if (axis == -1) axis = a;
        else if (axis != a) return -2;
    }
    return axis;
}

// ── 表达式规格相等比较（GPU AOT 匹配用）────────────────────────────────
// 两个 ExprSpec 相等 ⟺ 指令序列、输入视图、常量池、寄存器数全部一致。
// 用于运行时 eval_expr 判断"该表达式是否有预生成融合 shader"。
[[nodiscard]] inline bool expr_spec_equal(const ExprSpec& a, const ExprSpec& b)
{
    return a.instrs == b.instrs && a.views == b.views &&
           a.consts == b.consts && a.num_regs == b.num_regs;
}

// ── 规范结构 key（AOT 收集/匹配的单一依据）──────────────────────────────
// 把 ExprSpec 的**结构**（指令、视图、常量、寄存器数；不含输入张量）确定性地
// 哈希成 16 位十六进制字符串。同一结构跨构建/跨调用恒得同 key，不同结构
// 以极大概率不同。用于：
//   1. 构建期 scan_exprs 注册表去重（identical 表达式只合成一个 shader）
//   2. gen_fused 产物命名（fused_<key>）与嵌入注册
//   3. 运行时 eval_expr 折叠内联表达式 → key → 查预编译 shader（闭合世界）
// 逐字段字节级串接（不用 memcpy 整个 struct，避免 padding/平台差异）。
[[nodiscard]] inline std::string expr_spec_key(const ExprSpec& s)
{
    std::uint64_t h = 0xcbf29ce484222325ull;  // FNV-1a 64 offset basis
    const auto feed = [&h](const void* p, std::size_t n)
    {
        const auto* b = static_cast<const std::uint8_t*>(p);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= static_cast<std::uint64_t>(b[i]);
            h *= 0x100000001b3ull;
        }
    };
    const auto feed_u32 = [&](std::uint32_t v) { feed(&v, sizeof(v)); };

    feed_u32(s.num_regs);
    feed_u32(static_cast<std::uint32_t>(s.instrs.size()));
    for (const auto& in : s.instrs)
    {
        feed(&in.op, 1);
        feed(&in.dst, 1);
        feed(&in.a, sizeof(in.a));  // ExprOperand: 2 字节 POD
        feed(&in.b, sizeof(in.b));
        feed(&in.c, sizeof(in.c));
    }
    feed_u32(static_cast<std::uint32_t>(s.views.size()));
    for (const auto& v : s.views)
    {
        feed(&v.kind, 1);
        feed(&v.negate_first_half, 1);
        // RowMod/RotateHalf 的 param（周期/块大小）是**运行时形状数据**，
        // 不进 key：同结构不同形状（如不同 d_k）共享一个融合 shader
        // （glsl_gen 把 param 作为 push constant 读取，dispatch 时按实际
        // spec 填充）。其余视图 param=0 固定，feed 与否不影响。
        if (!expr_view_has_runtime_param(static_cast<ExprViewKind>(v.kind)))
            feed_u32(v.param);
    }
    feed_u32(static_cast<std::uint32_t>(s.consts.size()));
    for (const auto& c : s.consts)
        feed(&c, sizeof(c));

    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
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
        const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
        const std::size_t nops = expr_instr_num_operands(static_cast<ExprOp>(ins.op));
        for (std::size_t oi = 0; oi < nops; ++oi)
        {
            const ExprOperand& op = *ops[oi];
            if ((op.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                 op.kind == static_cast<uint8_t>(ExprOperandKind::Fanout)) && op.idx >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: src reg out of range"});
            if (op.kind == static_cast<uint8_t>(ExprOperandKind::Input) && op.idx >= num_inputs)
                return std::unexpected(Error{"validate_expr_spec: input index out of range"});
            if (op.kind == static_cast<uint8_t>(ExprOperandKind::Const) && op.idx >= spec.consts.size())
                return std::unexpected(Error{"validate_expr_spec: const index out of range"});
        }
    }

    // ── 归约语义校验 ─────────────────────────────────────────────────────
    //   1. Reduce 操作数只能引用"已出现的归约指令 dst"（隐式标量向量），
    //      且禁止自引用（归约指令的源引用自己）。
    //   2. 普通 Reg/Fanout 引用归约 dst 无意义（归约结果只能经 Reduce 操作数
    //      按行/列广播访问）——拒绝，避免引擎侧产生歧义。
    //   3. 归约指令只需源操作数 a，b/c 忽略。
    {
        std::vector<uint8_t> reduce_dst(spec.num_regs, 0);  // dst -> 是否为归约结果
        std::vector<uint8_t> elem_dst(spec.num_regs, 0);    // dst -> 是否为逐元素结果
        for (const auto& ins : spec.instrs)
        {
            const ExprOp op = static_cast<ExprOp>(ins.op);
            const bool is_reduce = expr_op_is_reduce(op);
            (is_reduce ? reduce_dst : elem_dst)[ins.dst] = 1;

            const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
            const std::size_t nops = is_reduce ? 1
                : expr_instr_num_operands(op);
            for (std::size_t oi = 0; oi < nops; ++oi)
            {
                const ExprOperand& opnd = *ops[oi];
                if (opnd.kind == static_cast<uint8_t>(ExprOperandKind::Reduce))
                {
                    if (opnd.idx >= spec.num_regs)
                        return std::unexpected(Error{"validate_expr_spec: reduce ref out of range"});
                    if (opnd.idx == ins.dst)
                        return std::unexpected(Error{"validate_expr_spec: reduce self-reference"});
                    if (!reduce_dst[opnd.idx] || elem_dst[opnd.idx])
                        return std::unexpected(Error{"validate_expr_spec: Reduce operand must reference a prior reduce instruction"});
                }
                else if ((opnd.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                          opnd.kind == static_cast<uint8_t>(ExprOperandKind::Fanout)) &&
                         opnd.idx < spec.num_regs && reduce_dst[opnd.idx])
                {
                    return std::unexpected(Error{"validate_expr_spec: register ref to reduce dst (use Reduce operand)"});
                }
            }
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
    inline constexpr ExprOperand reduce(std::uint8_t r){ return {4, r}; }  // 引用归约指令 dst
    inline constexpr ExprView linear()                 { return {0, 0, 0}; }
    inline constexpr ExprView rotate_half(std::uint32_t block_rows, bool negate_first_half = true)
    { return {1, negate_first_half ? std::uint8_t{1} : std::uint8_t{0}, block_rows}; }
    inline constexpr ExprView row_mod(std::uint32_t modulo)
    { return {2, 0, modulo}; }
    inline constexpr ExprView col_reduce_sum() { return {3, 0, 0}; }
    inline constexpr ExprView col_reduce_max() { return {4, 0, 0}; }
    inline constexpr ExprView row_reduce_sum() { return {5, 0, 0}; }
    inline constexpr ExprView row_reduce_max() { return {6, 0, 0}; }
    inline constexpr ExprView row_broadcast()  { return {7, 0, 0}; }
    inline constexpr ExprView col_broadcast()  { return {8, 0, 0}; }
} // namespace expr

} // namespace nn

#endif // NN_EXPR_SPEC_HPP
