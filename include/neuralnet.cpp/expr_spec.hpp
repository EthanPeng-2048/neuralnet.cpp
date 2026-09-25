#pragma once

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
#include <optional>
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
    // ── 新增：引用前置 matmul 段（MatmulSpec）的输出 ──
    // 语义：等价于把 matmul 输出 C(r,c) = Σ_k opA(r,k)*opB(k,c) 当作一个
    // "虚拟输入寄存器"，按当前输出网格 (r,c) 读取（MatmulSpec 结构见下）。
    // idx 恒为 0（一个 spec 至多一个 matmul 段）。
    Matmul = 5,
    // ── 新增：网格索引操作数（位置相关掩码/偏置，S7）──
    // 把当前输出元素的行/列/批次下标作为标量值参与算术（uint → float）。
    //   Row   = 输出元素在 batch 内的行号（batched 网格 r % m_per；非 batched = r）
    //   Col   = 输出列号 c
    //   Batch = 批次下标（batched matmul 段网格的 batch 维；无 matmul 段 = 0）
    // 用于 causal 掩码 select(Col > Row, -inf, 0)、ALiBi -slope*(Row-Col) 等。
    Row   = 6,
    Col   = 7,
    Batch = 8,
    // 运行时标量参数（RParam，形状无关融合的标量推广）──
    // 引用 spec.rparams 第 idx 项。与视图参数（RowMod 周期 / RotateHalf
    // 块大小）同思想：**值本身是运行时数据**（如优化器的 lr/eps/β、偏差
    // 修正系数），不是表达式结构 → 不进 expr_spec_key；同结构不同值的
    // 表达式（如不同 lr、每步变化的 inv_bc）共享一个融合 shader，运行时
    // 按实际 spec 填充 push constant。
    // 与 Const 的区别：Const 值进 key（结构）、编译期不变；RParam 值不进
    // key（运行时）、CPU 求值用 spec.rparams[idx]、GPU 经 PC 传入。
    RParam = 9,
    // ── fold 行向量态操作数（P-C2）：按当前输出列 d 读行向量态[idx][d]──
    // 仅允许出现在 fold 的 finalize（向量域逐列求值）；idx 恒 0（单槽，
    // 多槽留位）。veclen=0（P-C1 fold）时校验拒绝。
    VecState = 10,
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
    // ── 新增（S7）：标签行收集 / 按批次索引 / 按批次列切片 ──
    // RowGather = 9：读取 data[uint(labels[col]) * cols + col]，其中 labels
    //   是 param 指向的输入槽（(1, cols) 浮点打包的类别索引）。用于稀疏
    //   交叉熵的标签位置 log_softmax：logits[label[c]][c]。
    // BatchMod = 10：读取 data[uint(batch) % param]（param = 取模数）。
    //   用于按批次索引的小向量（如 ALiBi 按头斜率 slopes[head]，
    //   head = batch % num_heads）。
    // BatchCol = 11：读取 data[uint(batch) * param + col]（param = 每批列数，
    //   输入 (1, batch*param)，如 doc_ids (1, batch*seq) 按 (batch, col) 切片）。
    RowGather = 9,
    BatchMod  = 10,
    BatchCol  = 11,
    // RowAccess = 12：行偏移+取模访问：读取 data[(offset + r % mod)*cols + c]。
    //   param = mod（取模数），param2 = offset（行偏移）。
    //   用于共享内存 SwiGLU 等"同一 (2·d_ff) 输入按半个偏移读取"的行切分：
    //     gate = RowAccess(in, mod=d_ff, offset=0)   → in[r % d_ff]
    //     up   = RowAccess(in, mod=d_ff, offset=d_ff) → in[d_ff + r % d_ff]
    //   覆盖 RowMod（offset=0）且支持跨半偏移；offset/mod 均为运行时形状数据。
    RowAccess = 12,
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
    uint32_t param = 0;               // RotateHalf: block_rows；RowMod: modulo；RowAccess: mod
    uint32_t param2 = 0;              // 仅 RowAccess：行偏移 offset

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

// ── 前置 matmul 段（算子融合二期 S1：matmul 参与 IR 融合）─────────────────
// 可选的 matmul 段（位于逐元素指令之前），表达 C(rows,cols) = op(A,B) 作为
// 逐元素链的起始"虚拟寄存器 0"：
//   C[r][c] = Σ_{k<mm_k} opA(r,k) * opB(k,c)
// 逐元素链经 ExprOperandKind::Matmul 操作数按 (r,c) 读取该结果。
//
// 形状语义（对"所有输入同形状"假设的定向放宽，见 docs/14 §3.1）：
//   - a_input/b_input 指向的两个输入按 matmul 语义解释：
//       A 存储 (ar, ac)：transA=0 → 逻辑 (M, K)（M=ar, K=ac）
//                         transA=1 → 存储 (K, M)（K=ar, M=ac，按 A^T 使用）
//       B 存储 (br, bc)：transB=0 → 逻辑 (K, N)（K=br, N=bc）
//                         transB=1 → 存储 (N, K)（N=br, K=bc，按 B^T 使用）
//     输出网格 (M, N) = 逐元素输出网格 (rows, cols)。
//   - 其余逐元素输入的视图/形状仍要求 (M, N)（如 bias、残差）。
//   - k（求和维度）是**形状参数**：不进 expr_spec_key（同结构不同 K 共享
//     一个融合 shader），运行时作为 push constant 填充。
//   - transA/transB/a_input/b_input 是**结构**：进 expr_spec_key。
//   - batch（S7）：批量数，A/B 按 batch 垂直切分为连续行块（与
//     batched_matmul 原语同布局），输出网格 (batch*M, N)；
//     **形状参数**：不进 key（同结构不同 batch 共享一个融合 shader），
//     运行时作为 push constant 填充，dispatch 的 z 维 = batch。
struct MatmulSpec
{
    std::uint8_t a_input = 0;  // A 是第几个输入（0-based，指向 views/inputs）
    std::uint8_t b_input = 0;  // B 是第几个输入
    std::uint8_t transA  = 0;  // 1 = A 存储为 (K, M)，按 A^T 使用
    std::uint8_t transB  = 0;  // 1 = B 存储为 (N, K)，按 B^T 使用
    std::uint32_t k      = 0;  // 求和维度（形状参数，运行时 push constant）
    std::uint32_t batch  = 1;  // 批量数（形状参数，运行时 push constant）

    friend bool operator==(const MatmulSpec&, const MatmulSpec&) = default;
};

// ── 行向量状态的块内更新（VecAccSpec：通用"双线性块累加"，P-C2）──────────
// 语义（每个收缩块执行一次，键域完成后）：
//   if (has_scale) vec[row][:] *= scale_reg(row);      // 行标量广播 rescale
//   vec[row][d] += Σ_{j∈valid} weight(row, j0+j) * b(j0+j, d);
// 其中 weight 是 body 产出的 (row, j) 网格寄存器，b_input 是 (k, vec_state_len)
// 布局的收缩侧输入。**通用结构**（流式线性组合 / 带 rescale 的加权累加），
// 引擎只认此结构、不认任何算法名。
struct VecAccSpec
{
    std::uint8_t vec_state  = 0;  // 目标行向量态槽（P-C2 恒 0；多槽留位）
    std::uint8_t weight_reg = 0;  // body 产出的 (row, j) 权重寄存器号
    std::uint8_t b_input    = 0;  // 收缩侧输入槽（(k, vec_state_len) 行主序）
    std::uint8_t scale_reg  = 0;  // 行标量缩放寄存器（rescale α）
    std::uint8_t has_scale  = 1;  // 0=仅累加；1=先 vec*=scale 再累加

    friend bool operator==(const VecAccSpec&, const VecAccSpec&) = default;
};

// ── 分块状态归约段（FoldSpec：分块流式求值地基，P-C1 起，P-C2 双域）──────
// 抽象：一切计算 = 沿某轴迭代 + 跨迭代状态（分块/硬件友好思想的通用承载，
// 不针对任何具体算法）。
// 域结构（P-C2 起）：
//   **键域**（收缩轴 k，块循环）：
//     每块在网格 (row, j∈[j0, j0+valid)) 上执行 body：
//       - 可选 matmul 段（内层收缩：网格 (row, j) 上求值 C(row,j)=Σ_d…，
//         经 Matmul 操作数引用——块局部、j 为**全局**列；a/b 索引与既有
//         MatmulSpec 宥全一致：cols=PC cols=k、batch 分解同 batched_matmul）
//       - 寄存器 0..num_state-1 = 行标量态（跨块持久，初值 inits[]）
//       - 归约指令 = 沿块内 j 归约（每行一块值，Reduce 操作数按行广播）
//       - Row/Col/Batch 操作数（需 matmul 段存在；语义与既有 batched 网格
//         一致：Row=row%（rows/batch)、Col=全局 j、Batch=row/(rows/batch)）
//       - Body 可写行标量态（源须全行标量——类别分析静态保证）
//     键域后执行 vecacc（若在）：行向量态按上式块更新（rescale + 双线性累加）
//   **向量域**（输出列 ∈ [0, out_cols)，out_cols = vec_state_len 或 1）：
//     finalize 指令序列逐 (row, d) 求值：可读行标量态、行向量态（VecState
//     操作数按当前 d 读）→ 末指令 dst = 输出元素；输出网格 (rows, out_cols)。
// P-C1 兼容：vec_state_len=0（无 vecacc/matmul/VecState）时退化为单列标量
//   fold，语义与行为逐字节不变。
// key：结构字段全进（inits/两段指令/matmul 转置与槽位/vecacc/causal_skip）；
//   k、matmul 的 k/batch、vec_state_len 不进（形状参数 → 运行时 push
//   constant；veclen 进 key 会让每个 dk 一个 shader，见 expr_spec_key 注释）。
struct FoldSpec
{
    std::uint8_t        num_state = 1;  // 行标量态数（前缀寄存器 0..num_state-1）
    std::uint32_t       k = 0;          // 收缩轴长度（形状参数，运行时 push constant）
    std::vector<Scalar> inits;          // 标量态初值，size == num_state（进 key）
    std::vector<ExprInstr> body;        // 键域块指令序列
    std::vector<ExprInstr> finalize;    // 向量域收尾指令（veclen=0 时单列执行）
    // ── P-C2 双域扩展（默认值 = 纯 P-C1 行为）──
    std::uint32_t                vec_state_len = 0;  // 行向量态长度（0=无；输出列数）
    std::optional<MatmulSpec>    matmul;   // 键域内层收缩段（块局部，N=k 全轴）
    std::optional<VecAccSpec>    vecacc;   // 行向量态块更新
    // causal 整块跳过（仅 make_fold_attn_o 非 Plain 置位；进 key——结构/
    //   codegen 分歧点）：生成器把块内 valid 钳到
    //   min(BLOCK, fold_k-k0, qt+1-k0)，k0>qt 的整块空转。被跳过的 j 恰为
    //   链内 select 屏蔽项（-inf/0）→ max 加 -inf、sum 加 0、w=0 时 +0·V=+0
    //   均为恒等 → 与全量计算逐位一致。CPU 不钳（全量算，等价性同上）。
    bool                            causal_skip = false;

    friend bool operator==(const FoldSpec&, const FoldSpec&) = default;
};

// ── fold body 寄存器类别流分析（校验 / CPU 执行 / GLSL 生成共用）──────────
// 类别：**行标量**（每行一值：状态、块归约 dst、纯标量源派生）vs **元素**
// （依赖块内列 kb：Input 或元素类派生）。写入状态（或行标量寄存器）的指令
// 源必须全为行标量——否则同一状态被各 kb/各线程覆盖（未定义）。静态单遍
// 传递即可判定（指令序内 src 类别已知）。
// 返回 reg_elem[r]（1=元素类）；状态收到元素源 → unexpected。
[[nodiscard]] inline Result<std::vector<uint8_t>> expr_fold_classify(
    const FoldSpec& f, std::uint32_t num_regs)
{
    std::vector<uint8_t> is_elem(num_regs, 0);
    std::vector<uint8_t> is_reduce_dst(num_regs, 0);
    const auto src_elem = [&](const ExprOperand& op) -> int
    {
        const auto k = static_cast<ExprOperandKind>(op.kind);
        if (k == ExprOperandKind::Input)
            return 1;
        // P-C2：依赖网格列（=块内 kb）的都算元素类——Matmul 段（N=全轴列）、
        //   Col（全局列索引）。Row / Batch 派生自行号（行固定 → 行标量）。
        //   BatchMod/BatchCol 是**视图 kind**（经 Input 操作数访问，Input 已=1，
        //   保守偏元素类：掩码链只进权重/归约源、不直接写状态，无碍）。
        //   VecState 不允许出现在 body（validate 拒）。
        if (k == ExprOperandKind::Matmul || k == ExprOperandKind::Col)
            return 1;
        if (k == ExprOperandKind::Row || k == ExprOperandKind::Batch ||
            k == ExprOperandKind::Const || k == ExprOperandKind::RParam)
            return 0;
        if (k == ExprOperandKind::VecState)
            return -1;
        if (k == ExprOperandKind::Reg || k == ExprOperandKind::Fanout)
        {
            if (op.idx >= num_regs) return -1;
            if (is_reduce_dst[op.idx]) return 0;   // 块归约 dst = 行标量
            return is_elem[op.idx];
        }
        if (k == ExprOperandKind::Reduce)
            return (op.idx < num_regs) ? 0 : -1;   // 归约结果 = 行标量
        return -1;                                 // 未知 kind → 域错误
    };
    for (const auto& ins : f.body)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::size_t nops = expr_op_is_reduce(op) ? 1
            : expr_instr_num_operands(op);
        const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
        int e = 0;
        for (std::size_t oi = 0; oi < nops; ++oi)
        {
            const int se = src_elem(*ops[oi]);
            if (se < 0)
                return std::unexpected(Error{"expr_fold_classify: operand domain error"});
            if (se > e) e = se;
        }
        if (ins.dst >= num_regs)
            return std::unexpected(Error{"expr_fold_classify: dst out of range"});
        if (expr_op_is_reduce(op))
        {
            is_reduce_dst[ins.dst] = 1;
            is_elem[ins.dst] = 0;                  // 归约输出恒为行标量
        }
        else
        {
            if (ins.dst < f.num_state && e != 0)
                return std::unexpected(Error{
                    "expr_fold_classify: state written from element-class source (kb-dependent overwrite)"});
            if (e != 0) is_elem[ins.dst] = 1;      // 类别单调升为元素（跨指令保守）
        }
    }
    return is_elem;
}

// ── 表达式规格（运行时可序列化，跨后端）─────────────────────────────────
// 语义：
//   - 所有输入 Tensor 同形状 (rows, cols)，views[i] 与 inputs[i] 一一对应
//   - 顺序执行 instrs，每指令结果写入 regs[dst]（寄存器数组 num_regs 个）
//   - 输出 = instrs.back().dst 寄存器的值，逐元素写入 (rows, cols) 输出
//   - matmul 段（可选）：若存在，先计算 C = op(A,B)（输出网格 (rows,cols)），
//     逐元素链经 Matmul 操作数读取（matmul 不消耗逐元素寄存器）；
//     instrs 可为空（输出 = matmul 结果本身）。
//   - fold 段（可选）：见 FoldSpec；存在时 instrs 必须为空，输出 = fold 输出。
// 上限（校验保证，亦约束 GPU 路径资源）：
//   - 指令 ≤ 64，寄存器 ≤ 16，输入 ≤ 8，常量 ≤ 16
struct ExprSpec
{
    std::vector<ExprInstr>      instrs;
    std::vector<ExprView>       views;   // 与 inputs 一一对应
    std::vector<Scalar>         consts;
    std::vector<Scalar>         rparams; // 运行时标量参数（不进 key，运行时按实际值填充）
    std::uint32_t               num_regs = 0;
    std::optional<MatmulSpec>   matmul;  // 前置 matmul 段（可选；缺省=无）
    std::optional<FoldSpec>     fold;    // 分块状态归约段（可选；缺省=无）
};

// ── matmul 段辅助（引擎/校验/生成器共用）──────────────────────────────
[[nodiscard]] inline bool expr_spec_has_matmul(const ExprSpec& s) noexcept
{ return s.matmul.has_value(); }
// fold 段（P-C1）：存在时该 spec 走分块状态归约求值路径
[[nodiscard]] inline bool expr_spec_has_fold(const ExprSpec& s) noexcept
{ return s.fold.has_value(); }
// 运行时 matmul 形状参数（k：求和维度；batch：批量数）。形状无关融合：
// k/batch 不进 key，作为 push constant 运行时填充（同 P2-13 的
// RowMod/RotateHalf 处理）。**P-C2：fold 自带 matmul 段同语义**（spec.fold
// 的段优先于顶层——fold spec 顶层恒无 matmul，两处不冲突）。
[[nodiscard]] inline std::optional<std::uint32_t> expr_spec_runtime_matmul_k(
    const ExprSpec& s) noexcept
{
    if (s.matmul) return s.matmul->k;
    if (s.fold && s.fold->matmul) return s.fold->matmul->k;
    return std::nullopt;
}
[[nodiscard]] inline std::uint32_t expr_spec_runtime_matmul_batch(
    const ExprSpec& s) noexcept
{
    if (s.matmul) return s.matmul->batch;
    if (s.fold && s.fold->matmul) return s.fold->matmul->batch;
    return 1u;
}
// fold 收缩轴长度（形状参数，不进 key → push constant 运行时填充）
[[nodiscard]] inline std::optional<std::uint32_t> expr_spec_runtime_fold_k(
    const ExprSpec& s) noexcept
{
    return s.fold ? std::optional<std::uint32_t>{s.fold->k} : std::nullopt;
}
// batched 网格的"每 batch 行数"（M）：rows 为输出总行（batch*M）
[[nodiscard]] inline std::size_t expr_spec_rows_per_batch(const ExprSpec& s,
                                                          std::size_t rows) noexcept
{
    const std::uint32_t b = expr_spec_runtime_matmul_batch(s);
    return (b > 0 && rows % b == 0) ? rows / b : rows;
}

// ── 运行时视图参数（形状无关融合的关键）───────────────────────────────
// RowMod（周期）与 RotateHalf（块大小）的 param 是**运行时形状数据**（如
// RoPE 的 d_k），不是表达式结构：同结构不同 param（不同 d_k）应共享一个
// 融合 shader。因此：
//   - expr_spec_key **不**把这两个 param 折进 key（结构相同 → 同 key）
//   - glsl_gen 把它们作为 push constant（vp 槽）读取，dispatch 时按实际
//     spec 填充 → 一个 shader 适配所有形状（任何 d_k）
[[nodiscard]] inline constexpr bool expr_view_has_runtime_param(ExprViewKind k) noexcept
{
    // RowMod/RotateHalf 的形状参数（d_k）与 BatchMod/BatchCol 的形状参数
    // （num_heads / seq）都是**运行时形状数据**：不进 expr_spec_key，作为
    // push constant vp 槽由 dispatch 按实际 spec 填充 → 同结构不同形状
    // （不同 d_k / num_heads / seq_len）共享一个融合 shader（形状无关融合）。
    return k == ExprViewKind::RowMod || k == ExprViewKind::RotateHalf ||
           k == ExprViewKind::BatchMod || k == ExprViewKind::BatchCol ||
           k == ExprViewKind::RowAccess;
}
// 该视图消耗的运行时视图参数槽位数（RowAccess 用 offset+mod 两个 vp 槽）
[[nodiscard]] inline constexpr std::uint32_t expr_view_runtime_param_slots(
    ExprViewKind k) noexcept
{
    return (k == ExprViewKind::RowAccess) ? 2u : 1u;
}
// 该 spec 的运行时视图参数个数（= 融合 shader 的 push constant vp 槽位数）
[[nodiscard]] inline std::uint32_t expr_spec_runtime_view_param_count(
    const ExprSpec& s) noexcept
{
    std::uint32_t n = 0;
    for (const auto& v : s.views)
        if (expr_view_has_runtime_param(static_cast<ExprViewKind>(v.kind)))
            n += expr_view_runtime_param_slots(static_cast<ExprViewKind>(v.kind));
    return n;
}
// 按视图顺序提取运行时视图参数（RowMod 周期 / RotateHalf 块大小 / RowAccess
// 的 mod+offset），运行时 eval_expr 用它填充融合 shader 的 push constant vp 槽。
[[nodiscard]] inline std::vector<std::uint32_t> expr_spec_runtime_view_params(
    const ExprSpec& s)
{
    std::vector<std::uint32_t> out;
    for (const auto& v : s.views)
    {
        const auto k = static_cast<ExprViewKind>(v.kind);
        if (!expr_view_has_runtime_param(k))
            continue;
        out.push_back(v.param);               // RowMod/RotateHalf: mod/block；RowAccess: mod
        if (k == ExprViewKind::RowAccess)
            out.push_back(v.param2);          // RowAccess 额外 offset 槽
    }
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
// 两个 ExprSpec 相等 ⟺ 指令序列、输入视图、常量池、运行时参数、寄存器数、
// matmul 段全部一致。
// 用于运行时 eval_expr 判断"该表达式是否有预生成融合 shader"。
[[nodiscard]] inline bool expr_spec_equal(const ExprSpec& a, const ExprSpec& b)
{
    return a.instrs == b.instrs && a.views == b.views &&
           a.consts == b.consts && a.rparams == b.rparams &&
           a.num_regs == b.num_regs &&
           a.matmul == b.matmul &&
           a.fold == b.fold;
}

// 该 spec 的运行时标量参数个数（= 融合 shader 的 push constant 浮点 p 槽位数）
[[nodiscard]] inline std::uint32_t expr_spec_runtime_param_count(
    const ExprSpec& s) noexcept
{
    return static_cast<std::uint32_t>(s.rparams.size());
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
        // RowMod/RotateHalf 的 param（周期/块大小）与 RowAccess 的 mod/offset
        // 是**运行时形状数据**，不进 key：同结构不同形状（如不同 d_k）共享
        // 一个融合 shader（glsl_gen 把 param 作为 push constant 读取，dispatch
        // 时按实际 spec 填充）。其余视图 param=0 固定，feed 与否不影响。
        if (!expr_view_has_runtime_param(static_cast<ExprViewKind>(v.kind)))
        {
            feed_u32(v.param);
            feed_u32(v.param2);
        }
    }
    feed_u32(static_cast<std::uint32_t>(s.consts.size()));
    for (const auto& c : s.consts)
        feed(&c, sizeof(c));
    // 运行时标量参数（RParam）：只喂**个数**（结构），不喂值——值本身是
    // 运行时数据（如优化器的 lr/eps/β），不进 key；同结构不同值的表达式
    // 共享一个融合 shader（glsl_gen 把 rparams 作为 push constant 读取，
    // dispatch 时按实际 spec 填充）。与 RowMod/RotateHalf 的 param 同处理。
    feed_u32(static_cast<std::uint32_t>(s.rparams.size()));
    // matmul 段（可选）：transA/transB/a_input/b_input 是**结构** → 进 key；
    // k（求和维度）是**形状参数** → 不进 key（同 RowMod/RotateHalf 的 param
    // 处理）：同结构不同 K 共享一个融合 shader（glsl_gen 把 k 作为 push
    // constant 读取，dispatch 时按实际 spec 填充）。
    if (s.matmul)
    {
        feed(&s.matmul->a_input, 1);
        feed(&s.matmul->b_input, 1);
        feed(&s.matmul->transA, 1);
        feed(&s.matmul->transB, 1);
    }
    // fold 段（P-C1）：结构字段全进 key（num_state、inits 初值——-inf/0 属
    // 结构、body/finalize 指令序列）；k 是形状参数不进（同 matmul.k 处理，
    // 运行时 push constant）。指令喂法与顶层 instrs 一致（字节级 POD）。
    if (s.fold)
    {
        feed(&s.fold->num_state, 1);
        feed_u32(static_cast<std::uint32_t>(s.fold->inits.size()));
        for (const auto& v : s.fold->inits)
            feed(&v, sizeof(v));
        const auto feed_seq = [&](const std::vector<ExprInstr>& seq)
        {
            feed_u32(static_cast<std::uint32_t>(seq.size()));
            for (const auto& in : seq)
            {
                feed(&in.op, 1);
                feed(&in.dst, 1);
                feed(&in.a, sizeof(in.a));
                feed(&in.b, sizeof(in.b));
                feed(&in.c, sizeof(in.c));
            }
        };
        feed_seq(s.fold->body);
        feed_seq(s.fold->finalize);
        // P-C2 双域字段：matmul 段（tA/tB/槽位进 key，k/batch 形状参数不进
        //   ——同顶层 MatmulSpec 规则）、vecacc（全进：weight/scale 槽位是结构）。
        //   **vec_state_len 不进 key**（形状参数——输出列数=模型 d_k，进 key 会让
        //   每个 dk 一个 shader、闭合世界永远缺登记（offload/attn 测试 dk miss
        //   实证）；运行时经 PC 的 vector_out 槽填充——同 k/batch/vp 形状无关
        //   融合先例）。vecacc 的存在性（⇔veclen>0，成对校验）由 if 包裹隐式编码。
        if (s.fold->matmul)
        {
            feed(&s.fold->matmul->a_input, 1);
            feed(&s.fold->matmul->b_input, 1);
            feed(&s.fold->matmul->transA, 1);
            feed(&s.fold->matmul->transB, 1);
        }
        if (s.fold->vecacc)
        {
            feed(&s.fold->vecacc->vec_state, 1);
            feed(&s.fold->vecacc->weight_reg, 1);
            feed(&s.fold->vecacc->b_input, 1);
            feed(&s.fold->vecacc->scale_reg, 1);
            feed(&s.fold->vecacc->has_scale, 1);
        }
        // causal 跳块：codegen 分歧点 → 结构进 key（同 vecacc 槽位先例）
        feed(&s.fold->causal_skip, 1);
    }

    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
}

// ── 精度签名（多精度 AOT 变体索引，Phase 2 in-kernel f16）─────────────────
// 同一个**结构** ExprSpec 可以按不同精度组合求值：第 i 个输入张量是 f16、
// 输出是 f16。AOT 以 (结构 key, 精度签名) 为索引 → 同一结构可生成多个**带类型**
// 的 shader 变体（in-kernel f16 = 半精度直读直写 + f32 参考算术），从而取代
// "边界 cast 适配层"为每个算子物化 f32 副本的做法（实测后者使训练 transient
// 膨胀 2.4×，见 docs/development/05-mixed-precision.md §12.5）。
//
// 位布局（uint32）：
//   bit 0..15  第 i 个输入是 F16（i = 输入槽位，与 views 顺序一一对应）
//   bit 16     输出是 F16
//   全 0 = 全 f32 = **旧行为**：key 不加后缀、registry 不生成额外变体（零回归）。
inline constexpr std::uint32_t EXPR_PREC_SIG_OUT_BIT    = 16u;
inline constexpr std::uint32_t EXPR_PREC_SIG_INPUT_MASK = 0xFFFFu;
using ExprPrecSig = std::uint32_t;

[[nodiscard]] inline constexpr ExprPrecSig expr_prec_sig_make(
    std::uint32_t input_bits, bool out_f16) noexcept
{
    return (out_f16 ? (1u << EXPR_PREC_SIG_OUT_BIT) : 0u)
         | (input_bits & EXPR_PREC_SIG_INPUT_MASK);
}

// 全 f32（旧行为）？
[[nodiscard]] inline constexpr bool expr_prec_sig_is_f32(ExprPrecSig s) noexcept
{ return s == 0u; }

// 第 i 个输入是否为 f16 / 输出是否为 f16
[[nodiscard]] inline constexpr bool expr_prec_sig_in_f16(ExprPrecSig s,
                                                         std::size_t i) noexcept
{ return i < 32u && ((s >> i) & 1u) != 0u; }
[[nodiscard]] inline constexpr bool expr_prec_sig_out_f16(ExprPrecSig s) noexcept
{ return ((s >> EXPR_PREC_SIG_OUT_BIT) & 1u) != 0u; }

// 变体索引 key：全 f32 → 结构 key 本身（逐字节等同旧行为）；否则加 "#xxxx"
[[nodiscard]] inline std::string expr_prec_sig_key(const std::string& spec_key,
                                                   ExprPrecSig sig)
{
    if (sig == 0u)
        return spec_key;
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%04x", static_cast<unsigned>(sig));
    return spec_key + buf;
}

// 诊断用：把签名渲染成 "in=[f16,f32,...] out=f16"
[[nodiscard]] inline std::string expr_prec_sig_str(ExprPrecSig sig,
                                                   std::size_t num_inputs)
{
    std::string s = "in=[";
    for (std::size_t i = 0; i < num_inputs; ++i)
    {
        if (i) s += ",";
        s += expr_prec_sig_in_f16(sig, i) ? "f16" : "f32";
    }
    s += "] out=";
    s += expr_prec_sig_out_f16(sig) ? "f16" : "f32";
    return s;
}

// ── 上限（GPU 资源 / 校验共用）───────────────────────────────────────────
inline constexpr std::size_t EXPR_MAX_INPUTS = 16;  // 树型 DSL 重复叶子上限（绑定按 spec 实际 views 动态创建，非固定）
inline constexpr std::size_t EXPR_MAX_CONSTS = 16;
inline constexpr std::size_t EXPR_MAX_REGS   = 32;  // 全融合分支 select 表达式（如 SwiGLU backward）所需；shader 仅声明实际 num_regs
inline constexpr std::size_t EXPR_MAX_INSTRS = 64;
// fold 段上限（P-C1：body/finalize 独立计数，不占顶层 instrs 预算——
//   两者在 GPU 上分别生成块循环体与收尾段，资源约束与主链解耦）
inline constexpr std::size_t FOLD_MAX_BODY    = 48;
inline constexpr std::size_t FOLD_MAX_FINALIZE = 16;
inline constexpr std::size_t FOLD_MAX_STATE   = 8;
inline constexpr std::size_t FOLD_MAX_VEC     = 1024;  // 行向量态长度上限（=输出列数）
inline constexpr std::uint32_t FOLD_MAX_MMK   = 1024;  // fold mm 段内层 k 上限
    // （Q 行预载 shared Qsh[1024] 的编译期尺寸——d_k 超限在 validate 静态拒）
// fold 块大小：CPU 执行器与 GPU 生成器**共用**的常量（不进 key——分块是
// 实现细节，但两侧必须同值以对齐分块边界与 max 类逐位；sum 类 GPU subgroup
// 蝶形结合序异于 CPU 串行 → 对拍仍走小容差（1e-4~1e-6），并非全逐位）。
// 历次调整均以交错 bench 实测裁决：32→64（每块协议减半，fwd −6%）；64→128
// （协议再减半 + 链/归约活跃线程翻倍（tid<BLOCK）；shared 6.75→~8.9KB、驻留
// 8→7 WG 的占用代价被收益盖过——mha fwd 5.77→5.41，18/18 绿）
inline constexpr std::uint32_t EXPR_FOLD_BLOCK = 128;
// fold v2 每 WG 行数（NR）：把"按 WG 数计费"的固定成本（Qsh 预载、入退场、
//   调度人头）摊到 NR 行——四形状拟合实测该类占 fold ~54%（γ·rows）。
//   生成器行循环与后端 dispatch ceil(rows/NR) **同源共用**；结构常量不进
//   key（两侧同值即可，shader 同 key 重生成，闭合世界无感）。行序外层循环
//   复用全部 shared（零增长 → 占用不掉档，Qsh-shrink 探针已证 shared 非
//   约束）；越界行 clamp 到末行重复算、写回由 row_ok 统一挡（末 WG 尾部
//   padding；ri 循环全 WG 均匀 → 体内屏障无发散）。
inline constexpr std::uint32_t EXPR_FOLD_ROWS_PER_WG = 2;

// ── matmul 融合分块尺寸（S5：glsl_gen 生成与后端 dispatch 共用）─────────
// 生成器把 matmul 段展开为共享内存分块 kernel：
//   - EXPR_MATMUL_TILE：local_size 每维线程数（16×16 = 256 线程）
//   - EXPR_MATMUL_BLOCK：每工作组计算的输出块每维元素数（64×64，
//     每线程 4×4 寄存器分块；共享内存 AshT/BshT = 8×64×16B×2 = 16KB）
//   - K 方向每块 32（BK=32，barrier 减半）
// 后端 dispatch 按输出块缩放：(ceil(cols/BLOCK), ceil(rows/BLOCK), 1)，
// 与生成器必须保持一致。
inline constexpr std::uint32_t EXPR_MATMUL_TILE   = 16;
inline constexpr std::uint32_t EXPR_MATMUL_BLOCK  = 64;

// ── 校验 ──────────────────────────────────────────────────────────────────
// 返回 unexpected(Error) 描述首个非法点。Layer 侧可在提交前调用以尽早报错。
[[nodiscard]] inline Result<void> validate_expr_spec(const ExprSpec& spec,
                                                     std::size_t num_inputs)
{
    // matmul 段或 fold 段存在时允许空指令表（输出 = 段结果本身；
    //   fold 的 finalize 序列即其尾链）
    if (spec.instrs.empty() && !spec.matmul && !spec.fold)
        return std::unexpected(Error{"validate_expr_spec: empty instruction list"});
    if (spec.instrs.size() > EXPR_MAX_INSTRS)
        return std::unexpected(Error{"validate_expr_spec: too many instructions"});
    if (spec.num_regs > EXPR_MAX_REGS)
        return std::unexpected(Error{"validate_expr_spec: too many registers"});
    if (num_inputs > EXPR_MAX_INPUTS)
        return std::unexpected(Error{"validate_expr_spec: too many inputs"});
    if (spec.consts.size() > EXPR_MAX_CONSTS)
        return std::unexpected(Error{"validate_expr_spec: too many constants"});
    // 运行时标量参数同样受限（PC 浮点槽位有限）
    if (spec.rparams.size() > EXPR_MAX_CONSTS)
        return std::unexpected(Error{"validate_expr_spec: too many runtime params"});
    if (spec.views.size() != num_inputs)
        return std::unexpected(Error{"validate_expr_spec: views count != inputs count"});
    // matmul 段：A/B 输入下标必须在输入范围内（形状由引擎按实际张量推导）
    if (spec.matmul)
    {
        if (spec.matmul->a_input >= num_inputs || spec.matmul->b_input >= num_inputs)
            return std::unexpected(Error{"validate_expr_spec: matmul input out of range"});
        if (spec.matmul->batch == 0)
            return std::unexpected(Error{"validate_expr_spec: matmul batch must be > 0"});
    }

    // ── fold 段校验（P-C1 起；P-C2 双域：可带自带 matmul 段/行向量态）────
    if (spec.fold)
    {
        const FoldSpec& f = *spec.fold;
        if (spec.matmul)
            return std::unexpected(Error{
                "validate_expr_spec: fold + 顶层 matmul not allowed（fold 用自带段 f.matmul）"});
        if (!spec.instrs.empty())
            return std::unexpected(Error{
                "validate_expr_spec: fold requires empty top-level instrs"});
        if (f.num_state == 0 || f.num_state > FOLD_MAX_STATE ||
            f.inits.size() != static_cast<std::size_t>(f.num_state))
            return std::unexpected(Error{
                "validate_expr_spec: fold state count/init mismatch"});
        if (f.k == 0)
            return std::unexpected(Error{"validate_expr_spec: fold k must be > 0"});
        if (f.body.empty() || f.body.size() > FOLD_MAX_BODY)
            return std::unexpected(Error{"validate_expr_spec: fold body size out of range"});
        if (f.finalize.empty() || f.finalize.size() > FOLD_MAX_FINALIZE)
            return std::unexpected(Error{"validate_expr_spec: fold finalize size out of range"});
        if (spec.num_regs < f.num_state)
            return std::unexpected(Error{"validate_expr_spec: fold num_regs < num_state"});
        // P-C2 双域字段
        if (f.vec_state_len > FOLD_MAX_VEC)
            return std::unexpected(Error{"validate_expr_spec: fold vec_state_len out of range"});
        if ((f.vec_state_len > 0) != f.vecacc.has_value())
            return std::unexpected(Error{
                "validate_expr_spec: fold vec_state_len 与 vecacc 必须成对"});
        if (f.matmul)
        {
            if (f.matmul->a_input >= num_inputs || f.matmul->b_input >= num_inputs)
                return std::unexpected(Error{"validate_expr_spec: fold matmul input out of range"});
            if (f.matmul->batch == 0)
                return std::unexpected(Error{"validate_expr_spec: fold matmul batch must be > 0"});
            if (f.matmul->k > FOLD_MAX_MMK)
                return std::unexpected(Error{
                    "validate_expr_spec: fold matmul k exceeds Qsh shared preload cap (1024)"});
        }
        // causal_skip 的 Row/m_per 网格语义取自 mm.batch——无 mm 段时生成器
        //   会引用未声明的 m_per（glslc 报错但定位差），此处静态拒绝
        if (f.causal_skip && !f.matmul)
            return std::unexpected(Error{
                "validate_expr_spec: fold causal_skip requires fold matmul segment"});
        if (f.vecacc)
        {
            const VecAccSpec& va = *f.vecacc;
            if (va.vec_state != 0)
                return std::unexpected(Error{"validate_expr_spec: fold vecacc: multi vec slot unsupported"});
            if (va.weight_reg >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: fold vecacc weight_reg out of range"});
            if (va.b_input >= num_inputs)
                return std::unexpected(Error{"validate_expr_spec: fold vecacc b_input out of range"});
            if (va.scale_reg >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: fold vecacc scale_reg out of range"});
        }

        const bool has_mm = f.matmul.has_value();
        const auto check_operand_domain = [&](const ExprOperand& opnd) -> Result<void>
        {
            const auto k = static_cast<ExprOperandKind>(opnd.kind);
            // Matmul 段操作数 / 网格索引操作数：P-C2 起随 fold 自带 matmul 段
            //   放开（掩码折叠与内层收缩消费；语义与既有 batched 网格一致）
            if (k == ExprOperandKind::Matmul && !has_mm)
                return std::unexpected(Error{"validate_expr_spec: fold: Matmul operand without fold matmul segment"});
            if ((k == ExprOperandKind::Row || k == ExprOperandKind::Col ||
                 k == ExprOperandKind::Batch) && !has_mm)
                return std::unexpected(Error{
                    "validate_expr_spec: fold: index operands require fold matmul segment (batch 网格语义)"});
            // VecState 的 body/finalize 区分不在这里做：本 lambda 两处共用——
            //   body 侧由 expr_fold_classify 拒（src_elem(VecState) = -1 域错误），
            //   finalize 侧由下方主链做域校验（veclen/单槽）。
            return {};
        };

        // body：归约限行轴、Reduce 访问顺序规则、寄存器域、临时 def-before-use
        //   （状态前缀恒 defined；归约 dst 只能经 Reduce 操作数访问——沿用
        //   全局归约语义规则，此处对 fold body 独立执行）
        std::vector<uint8_t> defined(spec.num_regs, 0);
        for (std::uint8_t s = 0; s < f.num_state; ++s)
            defined[s] = 1;
        std::vector<uint8_t> body_reduce_dst(spec.num_regs, 0);
        for (const auto& ins : f.body)
        {
            const ExprOp op = static_cast<ExprOp>(ins.op);
            if (expr_op_is_reduce(op) && expr_op_reduces_cols(op))
                return std::unexpected(Error{
                    "validate_expr_spec: fold body: col-reduce not allowed (block axis is the row-reduce axis)"});
            if (ins.dst >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: fold body dst out of range"});
            const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
            const std::size_t nops = expr_op_is_reduce(op) ? 1
                : expr_instr_num_operands(op);
            for (std::size_t oi = 0; oi < nops; ++oi)
            {
                const ExprOperand& opnd = *ops[oi];
                const auto k = static_cast<ExprOperandKind>(opnd.kind);
                if (auto d = check_operand_domain(opnd); !d)
                    return std::unexpected(d.error());
                if (k == ExprOperandKind::Reduce)
                {
                    if (opnd.idx >= spec.num_regs || opnd.idx == ins.dst)
                        return std::unexpected(Error{"validate_expr_spec: fold body reduce ref invalid"});
                    if (!body_reduce_dst[opnd.idx])
                        return std::unexpected(Error{
                            "validate_expr_spec: fold body Reduce operand must reference a prior block-reduce instr"});
                }
                else if (k == ExprOperandKind::Reg || k == ExprOperandKind::Fanout)
                {
                    if (opnd.idx >= spec.num_regs)
                        return std::unexpected(Error{"validate_expr_spec: fold body reg out of range"});
                    if (body_reduce_dst[opnd.idx])
                        return std::unexpected(Error{
                            "validate_expr_spec: fold body reg ref to block-reduce dst (use Reduce operand)"});
                    if (!defined[opnd.idx])
                        return std::unexpected(Error{
                            "validate_expr_spec: fold body use before def (or block-local reg read)"});
                }
                else if (k == ExprOperandKind::Input)
                {
                    if (opnd.idx >= num_inputs)
                        return std::unexpected(Error{"validate_expr_spec: fold body input out of range"});
                    const auto vk = static_cast<ExprViewKind>(spec.views[opnd.idx].kind);
                    if (expr_view_is_reduce(vk))
                        return std::unexpected(Error{
                            "validate_expr_spec: fold body: reduce-view input not allowed"});
                    if (vk == ExprViewKind::Linear || vk == ExprViewKind::RowMod ||
                        vk == ExprViewKind::RowBroadcast)
                        ;  // P-C1 恒允许；RowBroadcast = 行参数向量（b[row]，
                           //   无列依赖——doc_col 等行级掩码参数恒可用）
                    else if ((vk == ExprViewKind::BatchMod ||
                              vk == ExprViewKind::BatchCol) && has_mm)
                        ;  // P-C2：掩码钩子消费（ALiBi 斜率 / doc_ids），需 batch 网格
                    else
                        return std::unexpected(Error{
                            "validate_expr_spec: fold body: view kind outside supported scope"});
                }
                else if (k == ExprOperandKind::Const)
                {
                    if (opnd.idx >= spec.consts.size())
                        return std::unexpected(Error{"validate_expr_spec: fold body const out of range"});
                }
                else if (k == ExprOperandKind::RParam)
                {
                    if (opnd.idx >= spec.rparams.size())
                        return std::unexpected(Error{"validate_expr_spec: fold body rparam out of range"});
                }
            }
            if (expr_op_is_reduce(op))
                body_reduce_dst[ins.dst] = 1;
            defined[ins.dst] = 1;
        }

        // finalize：只读状态/常量/rparam/Row——禁归约、禁 Input、禁块内临时
        for (const auto& ins : f.finalize)
        {
            const ExprOp op = static_cast<ExprOp>(ins.op);
            if (expr_op_is_reduce(op))
                return std::unexpected(Error{"validate_expr_spec: fold finalize: reduce not allowed"});
            if (ins.dst >= spec.num_regs)
                return std::unexpected(Error{"validate_expr_spec: fold finalize dst out of range"});
            const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
            const std::size_t nops = expr_instr_num_operands(op);
            for (std::size_t oi = 0; oi < nops; ++oi)
            {
                const ExprOperand& opnd = *ops[oi];
                const auto k = static_cast<ExprOperandKind>(opnd.kind);
                if (auto d = check_operand_domain(opnd); !d)
                    return std::unexpected(d.error());
                if (k == ExprOperandKind::Reduce)
                    return std::unexpected(Error{"validate_expr_spec: fold finalize: Reduce operand not allowed"});
                if (k == ExprOperandKind::Input)
                    return std::unexpected(Error{"validate_expr_spec: fold finalize cannot read inputs"});
                if (k == ExprOperandKind::VecState)
                {
                    if (f.vec_state_len == 0)
                        return std::unexpected(Error{
                            "validate_expr_spec: fold finalize: VecState without vec_state_len"});
                    if (opnd.idx != 0)
                        return std::unexpected(Error{
                            "validate_expr_spec: fold finalize: VecState multi-slot unsupported"});
                    continue;  // 单槽域校验通过
                }
                if (k == ExprOperandKind::Reg || k == ExprOperandKind::Fanout)
                {
                    if (opnd.idx >= spec.num_regs)
                        return std::unexpected(Error{"validate_expr_spec: fold finalize reg out of range"});
                    if (opnd.idx >= f.num_state)
                        return std::unexpected(Error{
                            "validate_expr_spec: fold finalize reads block-local register"});
                }
                else if (k == ExprOperandKind::Const && opnd.idx >= spec.consts.size())
                    return std::unexpected(Error{"validate_expr_spec: fold finalize const out of range"});
                else if (k == ExprOperandKind::RParam && opnd.idx >= spec.rparams.size())
                    return std::unexpected(Error{"validate_expr_spec: fold finalize rparam out of range"});
            }
        }

        // 类别流分析：状态不得从元素类（kb 依赖）源更新
        auto cls = expr_fold_classify(f, spec.num_regs);
        if (!cls)
            return std::unexpected(cls.error());
        // vecacc 缩放必须是行标量（广播 rescale；吃元素源=跨 kb 覆盖）
        if (f.vecacc && (*cls)[f.vecacc->scale_reg] != 0)
            return std::unexpected(Error{
                "validate_expr_spec: fold vecacc: scale_reg must be row-scalar class"});
        // 向量域多列循环的**写后读跨迭代污染**静态拒：finalize 被写的寄存器
        //   不得再被任何源读取（如 dst 复用被读状态 l——首列输出覆盖除数，
        //   次列起全错，实测 0.4=10/25）。单列输出（vec_state_len=0）无跨迭代，
        //   不受限（P-C1 的状态自复制 finalize 合法）。
        if (f.vec_state_len > 0)
        {
            std::vector<uint8_t> fin_written(spec.num_regs, 0);
            std::vector<uint8_t> fin_read(spec.num_regs, 0);
            for (const auto& ins : f.finalize)
            {
                fin_written[ins.dst] = 1;
                const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
                const std::size_t nops = expr_instr_num_operands(
                    static_cast<ExprOp>(ins.op));
                for (std::size_t oi = 0; oi < nops; ++oi)
                {
                    const auto k = static_cast<ExprOperandKind>(ops[oi]->kind);
                    if ((k == ExprOperandKind::Reg ||
                         k == ExprOperandKind::Fanout) && ops[oi]->idx < spec.num_regs)
                        fin_read[ops[oi]->idx] = 1;
                }
            }
            for (std::uint32_t i = 0; i < spec.num_regs; ++i)
                if (fin_written[i] && fin_read[i])
                    return std::unexpected(Error{
                        "validate_expr_spec: fold finalize: written reg also read "
                        "(multi-column loop would read its own prior output)"});
        }
    }
    // S7 视图：RowGather 的标签槽（param）必须在输入范围内
    for (std::size_t k = 0; k < spec.views.size(); ++k)
    {
        if (static_cast<ExprViewKind>(spec.views[k].kind) == ExprViewKind::RowGather &&
            spec.views[k].param >= num_inputs)
            return std::unexpected(Error{"validate_expr_spec: RowGather label slot out of range"});
    }
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
            if (op.kind == static_cast<uint8_t>(ExprOperandKind::RParam) && op.idx >= spec.rparams.size())
                return std::unexpected(Error{"validate_expr_spec: rparam index out of range"});
            if (op.kind == static_cast<uint8_t>(ExprOperandKind::Matmul) && !spec.matmul)
                return std::unexpected(Error{
                    "validate_expr_spec: Matmul operand without matmul segment"});
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
    inline constexpr ExprOperand matmul_op()           { return {5, 0}; }  // 引用前置 matmul 段输出
    inline constexpr ExprOperand row()                 { return {6, 0}; }  // 当前行号（batch 内）
    inline constexpr ExprOperand col()                 { return {7, 0}; }  // 当前列号
    inline constexpr ExprOperand batch()               { return {8, 0}; }  // 当前批次下标
    // 运行时标量参数（RParam）：运行时按实际 spec.rparams[idx] 填充
    inline constexpr ExprOperand rval(std::uint8_t r)  { return {9, r}; }
    // fold 行向量态（P-C2）：按向量域当前输出列读单槽向量态
    inline constexpr ExprOperand vec_state(std::uint8_t slot = 0) { return {10, slot}; }
    inline constexpr ExprView linear()                 { return {0, 0, 0}; }
    inline constexpr ExprView rotate_half(std::uint32_t block_rows, bool negate_first_half = true)
    { return {1, negate_first_half ? std::uint8_t{1} : std::uint8_t{0}, block_rows}; }
    inline constexpr ExprView row_mod(std::uint32_t modulo)
    { return {2, 0, modulo}; }
    // RowAccess(offset, mod)：source_row = offset + (r % mod)；param=mod, param2=offset
    inline constexpr ExprView row_access(std::uint32_t offset, std::uint32_t modulo)
    { return {12, 0, modulo, offset}; }
    inline constexpr ExprView col_reduce_sum() { return {3, 0, 0}; }
    inline constexpr ExprView col_reduce_max() { return {4, 0, 0}; }
    inline constexpr ExprView row_reduce_sum() { return {5, 0, 0}; }
    inline constexpr ExprView row_reduce_max() { return {6, 0, 0}; }
    inline constexpr ExprView row_broadcast()  { return {7, 0, 0}; }
    inline constexpr ExprView col_broadcast()  { return {8, 0, 0}; }
    // S7：标签行收集（param = 标签输入槽）与按批次索引/切片
    inline constexpr ExprView row_gather(std::uint8_t label_slot)
    { return {9, 0, label_slot}; }
    inline constexpr ExprView batch_mod(std::uint32_t modulo)
    { return {10, 0, modulo}; }
    inline constexpr ExprView batch_col(std::uint32_t per_batch_cols)
    { return {11, 0, per_batch_cols}; }
} // namespace expr

} // namespace nn

