#ifndef NN_EXPR_GLSL_GEN_HPP
#define NN_EXPR_GLSL_GEN_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_glsl_gen.hpp — AOT 算子融合：ExprSpec → GLSL 计算 shader 源码
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
#include "expr_emitter.hpp"   // IR-D：emitter 抽象（GlslEmitter 实现）

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
    case static_cast<uint8_t>(ExprViewKind::RowGather):
        // S7：按标签行收集：data[uint(labels[col]) * cols + col]（labels 槽 = v.param）
        os << buf << "[uint(b" << v.param << "[" << col_var << "]) * cols + "
           << col_var << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::BatchMod):
        // S7：按批次取模索引：data[batch % vpN]（需要 batch 变量；取模数
        // num_heads 为运行时视图参数 → 同结构不同头数共享一个融合 shader）
        os << buf << "[batch % vp" << std::to_string(vp_slot) << "]";
        return;
    case static_cast<uint8_t>(ExprViewKind::BatchCol):
        // S7：按 (batch, col) 切片：data[batch*vpN + col]（需要 batch 变量；
        //    每批列数 seq 为运行时视图参数 → 同结构不同 seq 共享一个融合 shader）
        os << buf << "[batch * vp" << std::to_string(vp_slot) << " + " << col_var << "]";
        return;
    }
}

// ── 视图是否使用行/列索引（用于按需发射 row/col，省去每元素整数除法/取模）──
// Linear 用扁平索引、RowBroadcast 用 row、ColBroadcast 用 col、
// RowMod/RotateHalf 用 row+col（索引映射逐元素变化，见 glsl_view_read）。
inline bool glsl_view_uses_row(ExprViewKind k)
{
    return k != ExprViewKind::Linear && k != ExprViewKind::ColBroadcast &&
           k != ExprViewKind::RowGather && k != ExprViewKind::BatchMod &&
           k != ExprViewKind::BatchCol;
}
inline bool glsl_view_uses_col(ExprViewKind k)
{
    return k != ExprViewKind::Linear && k != ExprViewKind::RowBroadcast &&
           k != ExprViewKind::BatchMod && k != ExprViewKind::BatchCol;
}
// 表达式是否需要 batch 变量（BatchMod/BatchCol 视图 / Batch 操作数）
inline bool glsl_spec_uses_batch(const ExprSpec& spec)
{
    for (const auto& v : spec.views)
        if (static_cast<ExprViewKind>(v.kind) == ExprViewKind::BatchMod ||
            static_cast<ExprViewKind>(v.kind) == ExprViewKind::BatchCol)
            return true;
    for (const auto& ins : spec.instrs)
    {
        const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
        const std::size_t nops = expr_instr_num_operands(static_cast<ExprOp>(ins.op));
        for (std::size_t oi = 0; oi < nops; ++oi)
            if (ops[oi]->kind == static_cast<uint8_t>(ExprOperandKind::Batch))
                return true;
    }
    return false;
}

// ── vec4 向量化资格 ───────────────────────────────────────────────────────
// 仅当全部视图 ∈ {Linear, RowBroadcast, ColBroadcast} 时才可把 4 个相邻元素
// 打包成 vec4（RowMod/RotateHalf 的索引映射逐通道变化，v1 不向量化）。
// 含前置 matmul 段（MatmulSpec）的表达式也不向量化（每元素 K 循环点积）。
// 注意：cols 是运行时 push constant，故向量化路径用运行时 `cols%4==0` 守卫，
// 任意形状仍走同 kernel 内的标量回退——正确性不依赖编译期形状。
inline bool glsl_vec4_eligible(const ExprSpec& spec)
{
    if (spec.matmul)
        return false;
    for (const auto& v : spec.views)
    {
        const auto k = static_cast<ExprViewKind>(v.kind);
        if (k != ExprViewKind::Linear && k != ExprViewKind::RowBroadcast &&
            k != ExprViewKind::ColBroadcast)
            return false;
    }
    // 索引操作数（Row/Col/Batch）逐通道变化 → 不向量化
    for (const auto& ins : spec.instrs)
    {
        const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
        const std::size_t nops = expr_instr_num_operands(static_cast<ExprOp>(ins.op));
        for (std::size_t oi = 0; oi < nops; ++oi)
            if (ops[oi]->kind == static_cast<uint8_t>(ExprOperandKind::Row) ||
                ops[oi]->kind == static_cast<uint8_t>(ExprOperandKind::Col) ||
                ops[oi]->kind == static_cast<uint8_t>(ExprOperandKind::Batch))
                return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  matmul 融合 shader 生成（S3 起 + S5 分块升级：融合分块矩阵乘法）
//
//  处理含前置 matmul 段（ExprSpec.matmul）的**逐元素**表达式：
//    - 共享内存分块矩阵乘（结构驱动，复用 matmul_tiled.comp 思路）：
//        WorkGroup = TILE×TILE 线程（16×16 = 256），计算 BLOCK×BLOCK 输出块
//        （32×32，每线程 2×2 寄存器分块 acc00/01/10/11）；
//        K 方向按 32 分块：协作加载 A/B 分块到 vec4 共享内存（转置感知、
//        合并访问），barrier 后每线程每 k4 读 4 个 vec4 + 16 FMA（4:1）。
//    - 尾逐元素链编译为 eval_tail(mm, row, col) 函数（GLSL 内联零开销），
//      写回时每个输出元素调用一次（Matmul 操作数 → mm，"虚拟寄存器 0"）。
//    - transA/transB 是**结构**（进 key）→ 索引表达式硬编码进 shader；
//      mm_k 是**形状参数**（不进 key）→ push constant，运行时填充
//      （同结构不同 K 共享一个融合 shader）。
//    - 布局：bindings 输入 0..N-1 + 输出 N（A/B 就是其中的两个输入槽）；
//      push constants: uint count, uint cols, uint rows, uint mm_k,
//                      [uint vp0..], [float c0..]；
//      dispatch: (ceil(cols/BLOCK), ceil(rows/BLOCK), 1)（见
//      EXPR_MATMUL_TILE/BLOCK，后端 run_fused_gpu 与生成器共用）。
//    - 归约指令（RowSum/RowMax/...）出现在尾链时属 S5 归约组合，由
//      generate_glsl_reduce 处理（本生成器遇到归约指令返回空 → 上层报错）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::string generate_glsl_matmul(
    const std::string& name, const ExprSpec& spec)
{
    const MatmulSpec& mm = *spec.matmul;
    const std::size_t n_inputs = spec.views.size();
    const std::uint32_t a_slot = mm.a_input;
    const std::uint32_t b_slot = mm.b_input;
    const bool trA = (mm.transA != 0);
    const bool trB = (mm.transB != 0);
    constexpr std::uint32_t T = EXPR_MATMUL_TILE;    // 线程（16）
    constexpr std::uint32_t B = EXPR_MATMUL_BLOCK;   // 输出块（32 = 2*T）
    constexpr std::uint32_t BK = 32;                 // K 分块宽度
    constexpr std::uint32_t BK4 = BK / 4;            // vec4 数/行（8）

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · matmul 分块 2×2），请勿手动编辑 ──\n";
    L << "// 表达式: " << name << "\n";
    L << "#version 450\n\n";
    L << "layout(local_size_x = " << T << ", local_size_y = " << T << ") in;\n\n";
    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { float b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { float bout[]; };\n\n";
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n";
    L << "    uint cols;\n";
    L << "    uint rows;\n";
    L << "    uint mm_k;      // matmul 求和维度（形状参数，运行时填充）\n";
    L << "    uint mm_batch;  // matmul 批量数（形状参数，运行时填充；dispatch z）\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    L << "};\n\n";
    // 共享内存分块（vec4 布局，16B 对齐，转置 [k4][m] 消除 bank 冲突）：
    //   AshT[k4][m]：A 分块（32 行 × 32 k，8 个 vec4/行），k4 在前 → 内层
    //     读 AshT[k4][ty*2] 时同一 k4 下连续线程读连续 m（无 bank 冲突）；
    //     加载写 AshT[k4][m] 时 tid 连续 → k4 连续（8 路写冲突，一次/块，
    //     远小于内层 8 次 k4 循环的读冲突代价）。
    //   BK=32：barrier 减半；共享内存 8KB。
    L << "shared vec4 AshT[" << BK4 << "][" << B << "];\n";
    L << "shared vec4 BshT[" << BK4 << "][" << B << "];\n\n";

    // transA/transB 感知的全局内存加载表达式（S7 batch：A/B 按 batch 垂直
    // 切分，batch*m_per 为 A 行偏移 / batch*mm_k 为 B k 偏移；row/col 为
    // **batch 内**坐标，block 偏移由调用方并入）
    const auto a_load = [&](const std::string& row_e, const std::string& k_e)
    {
        return trA
            ? "b" + std::to_string(a_slot) + "[((batch*mm_k + (" + k_e + "))*m_per + (" + row_e + "))]"
            : "b" + std::to_string(a_slot) + "[((batch*m_per + (" + row_e + "))*mm_k + (" + k_e + "))]";
    };
    const auto b_load = [&](const std::string& col_e, const std::string& k_e)
    {
        return trB
            ? "b" + std::to_string(b_slot) + "[((batch*cols + (" + col_e + "))*mm_k + (" + k_e + "))]"
            : "b" + std::to_string(b_slot) + "[((batch*mm_k + (" + k_e + "))*cols + (" + col_e + "))]";
    };

    // ── 尾逐元素链：eval_tail(mm, row, col, batch) 函数（GLSL 内联）──
    // row 为 batch 内行号（全局行 = batch*m_per + row），与 Row 操作数一致。
    const std::uint32_t last_dst = spec.instrs.empty()
        ? 0u : spec.instrs.back().dst;
    L << "float eval_tail(float mm, uint row, uint col, uint batch)\n{\n";
    if (!spec.instrs.empty())
    {
        const auto operand = [&](const ExprOperand& op) -> std::string
        {
            switch (op.kind)
            {
            default:
            case static_cast<uint8_t>(ExprOperandKind::Matmul):
                return "mm";
            case static_cast<uint8_t>(ExprOperandKind::Input):
                return "v" + std::to_string(op.idx);
            case static_cast<uint8_t>(ExprOperandKind::Reg):
            case static_cast<uint8_t>(ExprOperandKind::Fanout):
                return "r" + std::to_string(op.idx);
            case static_cast<uint8_t>(ExprOperandKind::Const):
                return "c" + std::to_string(op.idx);
            case static_cast<uint8_t>(ExprOperandKind::Row):
                return "float(row)";  // row 参数已是 batch 内行号
            case static_cast<uint8_t>(ExprOperandKind::Col):
                return "float(col)";
            case static_cast<uint8_t>(ExprOperandKind::Batch):
                return "float(batch)";
            }
        };
        // 输入读取（跳过 matmul A/B 槽：其形状不是 (rows,cols) 网格）
        // 行索引视图（Linear/RowBroadcast/RowMod/RotateHalf）必须用**全局行**
        // grow = batch*m_per + row——CPU 参考（compute_cpu_engine read_input）
        // 以全局行 r 索引 (rows,·) 全网格输入（如注意力 m/l 的 (BH·seq,1)）；
        // Row 操作数保持 batch 内行号（causal 掩码 col > row 语义，CPU 端
        // row() = r % m_per 一致）。
        L << "    const uint m_per = rows / mm_batch;\n";
        L << "    const uint grow = batch * m_per + row;\n";
        for (std::size_t i = 0; i < n_inputs; ++i)
        {
            if (i == a_slot || i == b_slot)
                continue;
            std::uint32_t vp = 0;
            for (std::size_t j = 0; j < i; ++j)
                if (expr_view_has_runtime_param(
                        static_cast<ExprViewKind>(spec.views[j].kind))) ++vp;
            L << "    const float v" << i << " = ";
            glsl_view_read(L, spec.views[i], static_cast<std::uint32_t>(i),
                           "grow*cols + col", "grow", "col", vp);
            L << ";\n";
        }
        if (spec.num_regs > 0)
        {
            L << "    float r0";
            for (std::uint32_t r = 1; r < spec.num_regs; ++r) L << ", r" << r;
            L << ";\n";
        }
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
                L << "    " << dst << " = " << a << " " << s << " "
                  << operand(ins.b) << ";\n";
                break;
            }
            case ExprOp::Max: case ExprOp::Min:
            {
                const char* s = (op == ExprOp::Max) ? "max" : "min";
                L << "    " << dst << " = " << s << "(" << a << ", "
                  << operand(ins.b) << ");\n";
                break;
            }
            case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
            case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
            {
                bool cmp = false;
                const char* s = glsl_binary_op(op, cmp);
                L << "    " << dst << " = (" << a << " " << s << " "
                  << operand(ins.b) << ") ? 1.0 : 0.0;\n";
                break;
            }
            case ExprOp::Neg:
                L << "    " << dst << " = -(" << a << ");\n";
                break;
            case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
            case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
            {
                const char* s = glsl_unary_op(op);
                L << "    " << dst << " = " << s << "(" << a << ");\n";
                break;
            }
            case ExprOp::Select:
                L << "    " << dst << " = (" << a << " != 0.0) ? "
                  << operand(ins.b) << " : " << operand(ins.c) << ";\n";
                break;
            default:
                return {};  // 归约指令：由 generate_glsl_reduce 处理 → 上层报错
            }
        }
        L << "    return r" << last_dst << ";\n";
    }
    else
    {
        // 纯 matmul（无逐元素链）：输出 = matmul 结果
        L << "    return mm;\n";
    }
    L << "}\n\n";

    // ── main：4×4 寄存器分块 batched matmul（dispatch (x,y,z) = (列块,行块,批次)）
    L << "void main()\n{\n";
    L << "    const uint tx = gl_LocalInvocationID.x;\n";
    L << "    const uint ty = gl_LocalInvocationID.y;\n";
    L << "    const uint tid = ty * " << T << "u + tx;\n";
    L << "    const uint batch = gl_WorkGroupID.z;   // 批次下标\n";
    L << "    const uint m_per = rows / mm_batch;    // 每批输出行数\n";
    L << "    const uint block_row = gl_WorkGroupID.y * " << B << "u;\n";
    L << "    const uint block_col = gl_WorkGroupID.x * " << B << "u;\n";
    // 4×4 寄存器累加器（本线程负责 16 个输出元素；dot() = 4 FMA）
    L << "    float acc[4][4];\n";
    L << "    for (uint i = 0u; i < 4u; ++i)\n";
    L << "        for (uint j = 0u; j < 4u; ++j) acc[i][j] = 0.0;\n";

    L << "    const uint num_tiles = (mm_k + " << BK - 1 << "u) / " << BK << "u;\n";
    L << "    for (uint t = 0u; t < num_tiles; ++t)\n";
    L << "    {\n";
    // 协作加载：B×BK = 2048 元素 = 512 vec4/矩阵，每线程 2 个 A vec4 + 2 个 B vec4
    L << "        for (uint l = 0u; l < 2u; ++l)\n";
    L << "        {\n";
    L << "            const uint e = tid * 2u + l;\n";
    L << "            const uint k4 = e % " << BK4 << "u;\n";
    L << "            const uint m  = e / " << BK4 << "u;\n";
    L << "            const uint kk = t * " << BK << "u + k4 * 4u;\n";
    L << "            const uint ar = block_row + m;\n";
    L << "            const uint bc = block_col + m;\n";
    L << "            vec4 av = vec4(0.0);\n";
    L << "            if (ar < m_per && kk < mm_k)\n";
    L << "            {\n";
    L << "                av.x = " << a_load("ar", "kk") << ";\n";
    L << "                if (kk + 1u < mm_k) av.y = " << a_load("ar", "kk + 1u") << ";\n";
    L << "                if (kk + 2u < mm_k) av.z = " << a_load("ar", "kk + 2u") << ";\n";
    L << "                if (kk + 3u < mm_k) av.w = " << a_load("ar", "kk + 3u") << ";\n";
    L << "            }\n";
    L << "            vec4 bv = vec4(0.0);\n";
    L << "            if (bc < cols && kk < mm_k)\n";
    L << "            {\n";
    L << "                bv.x = " << b_load("bc", "kk") << ";\n";
    L << "                if (kk + 1u < mm_k) bv.y = " << b_load("bc", "kk + 1u") << ";\n";
    L << "                if (kk + 2u < mm_k) bv.z = " << b_load("bc", "kk + 2u") << ";\n";
    L << "                if (kk + 3u < mm_k) bv.w = " << b_load("bc", "kk + 3u") << ";\n";
    L << "            }\n";
    L << "            AshT[k4][m] = av;\n";
    L << "            BshT[k4][m] = bv;\n";
    L << "        }\n";
    L << "        barrier();\n";
    // 内层：每 k4 读 8 个 vec4（4 行 A × 4 列 B）+ 16 dot（64 FMA）；
    // 转置布局下同一 k4 的连续线程读连续 m → 无 bank 冲突
    L << "        for (uint k4 = 0u; k4 < " << BK4 << "u; ++k4)\n";
    L << "        {\n";
    L << "            vec4 a[4];\n";
    L << "            #pragma unroll\n";
    L << "            for (uint i = 0u; i < 4u; ++i)\n";
    L << "                a[i] = AshT[k4][ty * 4u + i];\n";
    L << "            vec4 b[4];\n";
    L << "            #pragma unroll\n";
    L << "            for (uint j = 0u; j < 4u; ++j)\n";
    L << "                b[j] = BshT[k4][tx * 4u + j];\n";
    L << "            #pragma unroll\n";
    L << "            for (uint i = 0u; i < 4u; ++i)\n";
    L << "                for (uint j = 0u; j < 4u; ++j)\n";
    L << "                    acc[i][j] += dot(a[i], b[j]);\n";
    L << "        }\n";
    L << "        barrier();\n";
    L << "    }\n";

    // ── 写回：16 个输出元素，各执行一次尾链（边界守卫；rr 全局行，
    //    eval_tail 的 row 参数 = batch 内行号）──
    L << "    #pragma unroll\n";
    L << "    for (uint i = 0u; i < 4u; ++i)\n";
    L << "        for (uint j = 0u; j < 4u; ++j)\n";
    L << "        {\n";
    L << "            const uint rr = block_row + ty * 4u + i;\n";
    L << "            const uint cc = block_col + tx * 4u + j;\n";
    L << "            if (rr < m_per && cc < cols)\n";
    L << "                bout[(batch * m_per + rr) * cols + cc]\n";
    L << "                    = eval_tail(acc[i][j], rr, cc, batch);\n";
    L << "        }\n";
    L << "}\n";
    return L.str();
}

// ── 生成器主入口：ExprSpec → GLSL 源码 ────────────────────────────────────
// 生成的 shader 绑定：输入 buffer binding=0..N-1，输出 binding=N；
// push constants: uint count, uint cols, [float c0..]（视图索引需要 cols）。
// 逐元素表达式若 glsl_vec4_eligible，发射 vec4 kernel：每线程处理 4 个相邻元素
// （vec4 加载/运算/存储），dispatch 宽度 4（run_fused_gpu 按 vec_width 缩放）；
// 运行时 cols%4!=0 或尾部回退到同 kernel 内标量循环。其余表达式发纯标量 kernel。
inline std::string generate_glsl(const std::string& name, const ExprSpec& spec)
{
    // 含前置 matmul 段（S3）：matmul + 尾逐元素链融合 shader
    if (spec.matmul)
        return generate_glsl_matmul(name, spec);

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

    const bool vec4_ok = glsl_vec4_eligible(spec);
    // 索引操作数（Row/Col/Batch）也按需发射 row/col/batch 变量
    const auto instr_uses = [&](uint8_t kind) {
        for (const auto& ins : spec.instrs)
        {
            const ExprOperand* ops[3] = {&ins.a, &ins.b, &ins.c};
            const std::size_t nops = expr_instr_num_operands(static_cast<ExprOp>(ins.op));
            for (std::size_t oi = 0; oi < nops; ++oi)
                if (ops[oi]->kind == kind) return true;
        }
        return false;
    };
    const bool need_row = [&]{
        for (const auto& v : spec.views)
            if (glsl_view_uses_row(static_cast<ExprViewKind>(v.kind))) return true;
        return instr_uses(static_cast<uint8_t>(ExprOperandKind::Row)); }();
    const bool need_col = [&]{
        for (const auto& v : spec.views)
            if (glsl_view_uses_col(static_cast<ExprViewKind>(v.kind))) return true;
        return instr_uses(static_cast<uint8_t>(ExprOperandKind::Col)); }();
    const bool need_batch = glsl_spec_uses_batch(spec);
    const std::uint32_t last_dst = spec.instrs.back().dst;

    // ── 标量操作数求值 ──
    const auto operand = [&](const ExprOperand& op) -> std::string
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
        // S7 索引操作数：当前网格下标（uint → float 参与算术）
        case static_cast<uint8_t>(ExprOperandKind::Row):
            return "float(row)";
        case static_cast<uint8_t>(ExprOperandKind::Col):
            return "float(col)";
        case static_cast<uint8_t>(ExprOperandKind::Batch):
            return "float(batch)";
        }
    };

    // 发射一段标量指令链（输入读取 + 寄存器声明 + 指令展开，直线代码；
    // 寄存器先声明、后赋值——兼容 IR-B liveness 复用同号寄存器）。
    // idx_var：扁平元素索引；row_var/col_var：行/列索引表达式（可为 "i/cols" 等）。
    const auto emit_scalar_chain = [&](std::ostringstream& o,
                                       const std::string& indent,
                                       const std::string& idx_var,
                                       const std::string& row_var,
                                       const std::string& col_var)
    {
        // 输入读取变量（每输入缓存一次）；vp 槽 = 此前运行时参数视图个数
        for (std::size_t i = 0; i < n_inputs; ++i)
        {
            std::uint32_t vp = 0;
            for (std::size_t j = 0; j < i; ++j)
                if (expr_view_has_runtime_param(
                        static_cast<ExprViewKind>(spec.views[j].kind))) ++vp;
            o << indent << "const float v" << i << " = ";
            glsl_view_read(o, spec.views[i], static_cast<std::uint32_t>(i),
                           idx_var, row_var, col_var, vp);
            o << ";\n";
        }
        if (spec.num_regs > 0)
        {
            o << indent << "float r0";
            for (std::uint32_t r = 1; r < spec.num_regs; ++r) o << ", r" << r;
            o << ";\n";
        }
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
                o << indent << dst << " = " << a << " " << s << " "
                  << operand(ins.b) << ";\n";
                break;
            }
            case ExprOp::Max: case ExprOp::Min:
            {
                const char* s = (op == ExprOp::Max) ? "max" : "min";
                o << indent << dst << " = " << s << "(" << a << ", "
                  << operand(ins.b) << ");\n";
                break;
            }
            case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
            case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
            {
                bool cmp = false;
                const char* s = glsl_binary_op(op, cmp);
                o << indent << dst << " = (" << a << " " << s << " "
                  << operand(ins.b) << ") ? 1.0 : 0.0;\n";
                break;
            }
            case ExprOp::Neg:
                o << indent << dst << " = -(" << a << ");\n";
                break;
            case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
            case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
            {
                const char* s = glsl_unary_op(op);
                o << indent << dst << " = " << s << "(" << a << ");\n";
                break;
            }
            case ExprOp::Select:
                o << indent << dst << " = (" << a << " != 0.0) ? "
                  << operand(ins.b) << " : " << operand(ins.c) << ";\n";
                break;
            default:
                return;  // 未知算子：让上层报错
            }
        }
    };

    if (!vec4_ok)
    {
        // ── 纯标量 kernel（含 RowMod/RotateHalf 等无法向量化的视图）──
        L << "void main()\n{\n";
        L << "    const uint i = gl_GlobalInvocationID.x;\n";
        L << "    if (i >= count) return;\n";
        if (need_row && need_col)
            L << "    const uint row = i / cols;\n    const uint col = i % cols;\n";
        else if (need_row)
            L << "    const uint row = i / cols;\n";
        else if (need_col)
            L << "    const uint col = i % cols;\n";
        if (need_batch)
            L << "    const uint batch = 0u;\n";
        emit_scalar_chain(L, "    ", "i",
                          need_row ? "row" : "", need_col ? "col" : "");
        L << "    bout[i] = r" << last_dst << ";\n";
        L << "}\n";
        return L.str();
    }

    // ── vec4 kernel：每线程处理 4 个相邻元素 ────────────────────────────
    // 快速路径（cols%4==0 且组内 4 元素齐全）→ vec4 加载/运算/存储；
    // 否则回退到同 kernel 内标量循环（任意 cols / 尾部）。dispatch 宽度 4。
    L << "void main()\n{\n";
    L << "    const uint i = gl_GlobalInvocationID.x;\n";
    L << "    const uint base = i * 4u;\n";
    L << "    if (base >= count) return;\n";
    L << "    if (cols % 4u == 0u && base + 3u < count) {\n";
    if (need_row && need_col)
        L << "        const uint row = base / cols;\n        const uint col0 = base % cols;\n";
    else if (need_row)
        L << "        const uint row = base / cols;\n";
    else if (need_col)
        L << "        const uint col0 = base % cols;\n";

    // vec4 输入读取（Linear/ColBroadcast 相邻 4 元素 → glslc 合并为 vec4 加载；
    // RowBroadcast → splat）
    for (std::size_t i = 0; i < n_inputs; ++i)
    {
        const auto k = static_cast<ExprViewKind>(spec.views[i].kind);
        if (k == ExprViewKind::Linear)
            L << "        const vec4 v" << i << " = vec4(b" << i << "[base], b" << i
              << "[base+1u], b" << i << "[base+2u], b" << i << "[base+3u]);\n";
        else if (k == ExprViewKind::RowBroadcast)
            L << "        const vec4 v" << i << " = vec4(b" << i << "[row]);\n";
        else // ColBroadcast
            L << "        const vec4 v" << i << " = vec4(b" << i << "[col0], b" << i
              << "[col0+1u], b" << i << "[col0+2u], b" << i << "[col0+3u]);\n";
    }

    // vec4 指令展开（GLSL 对 vec4 重载运算/内置函数；常量广播为 vec4）
    const auto vec4_operand = [&](const ExprOperand& op) -> std::string
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
            return "vec4(c" + std::to_string(op.idx) + ")";
        }
    };
    if (spec.num_regs > 0)
    {
        L << "        vec4 r0";
        for (std::uint32_t r = 1; r < spec.num_regs; ++r) L << ", r" << r;
        L << ";\n";
    }
    for (const auto& ins : spec.instrs)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string dst = "r" + std::to_string(ins.dst);
        const std::string a = vec4_operand(ins.a);
        switch (op)
        {
        case ExprOp::Add: case ExprOp::Sub: case ExprOp::Mul: case ExprOp::Div:
        {
            bool cmp = false;
            const char* s = glsl_binary_op(op, cmp);
            L << "        " << dst << " = " << a << " " << s << " "
              << vec4_operand(ins.b) << ";\n";
            break;
        }
        case ExprOp::Max: case ExprOp::Min:
        {
            const char* s = (op == ExprOp::Max) ? "max" : "min";
            L << "        " << dst << " = " << s << "(" << a << ", "
              << vec4_operand(ins.b) << ");\n";
            break;
        }
        case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
        case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
        {
            const char* s = nullptr;
            switch (op)
            {
            case ExprOp::Lt: s = "lessThan"; break;
            case ExprOp::Le: s = "lessThanEqual"; break;
            case ExprOp::Gt: s = "greaterThan"; break;
            case ExprOp::Ge: s = "greaterThanEqual"; break;
            case ExprOp::Eq: s = "equal"; break;
            case ExprOp::Ne: s = "notEqual"; break;
            default: break;
            }
            L << "        " << dst << " = vec4(" << s << "(" << a << ", "
              << vec4_operand(ins.b) << "));\n";
            break;
        }
        case ExprOp::Neg:
            L << "        " << dst << " = -(" << a << ");\n";
            break;
        case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
        case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
        {
            const char* s = glsl_unary_op(op);
            L << "        " << dst << " = " << s << "(" << a << ");\n";
            break;
        }
        case ExprOp::Select:
            // mix(x,y,t) = x(1-t) + y·t：t=(a≠0) 时须取 b → b 在 y 位
            L << "        " << dst << " = mix(" << vec4_operand(ins.c) << ", "
              << vec4_operand(ins.b) << ", notEqual(" << a
              << ", vec4(0.0)));\n";
            break;
        default:
            return std::string{};  // 未知算子：让上层报错
        }
    }
    L << "        bout[base] = r" << last_dst << ".x;\n";
    L << "        bout[base+1u] = r" << last_dst << ".y;\n";
    L << "        bout[base+2u] = r" << last_dst << ".z;\n";
    L << "        bout[base+3u] = r" << last_dst << ".w;\n";
    L << "        return;\n";
    L << "    }\n";

    // 标量回退：处理本线程负责的至多 4 个元素（任意 cols / 尾部）
    L << "    for (uint e = base; e < count && e < base + 4u; ++e) {\n";
    if (need_row && need_col)
        L << "        const uint row = e / cols;\n        const uint col = e % cols;\n";
    else if (need_row)
        L << "        const uint row = e / cols;\n";
    else if (need_col)
        L << "        const uint col = e % cols;\n";
    if (need_batch)
        L << "        const uint batch = 0u;\n";
    emit_scalar_chain(L, "        ", "e",
                      need_row ? "row" : "", need_col ? "col" : "");
    L << "        bout[e] = r" << last_dst << ";\n";
    L << "    }\n";
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
    L << "#version 450\n";
    L << "#extension GL_KHR_shader_subgroup_basic : enable\n";
    L << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n\n";
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
    // matmul+归约（S5/S7，注意力结构）：mm_k 求和维度 + mm_batch 批量数
    // （均为形状参数，运行时填充；dispatch 不变，batch 由 idx 分解）
    if (spec.matmul)
    {
        L << "    uint mm_k;\n";
        L << "    uint mm_batch;\n";
    }
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    L << "};\n\n";
    L << "shared float s_red[" << n_slots << "][256];\n\n";

    // ── matmul 段（S5 + S7 batch）：当前元素 (row,col) 的 matmul 值（内联
    //     K 循环点积，不物化 (batch*M,N) 中间矩阵；transA/transB 硬编码，
    //     mm_k/mm_batch 运行时填充；row 为全局行 = batch*m_per + 块内行）──
    const MatmulSpec* mm = spec.matmul ? &*spec.matmul : nullptr;
    const auto emit_mm_decl = [&]()
    {
        if (!mm)
            return;
        const std::uint32_t a_slot = mm->a_input;
        const std::uint32_t b_slot = mm->b_input;
        const bool trA = (mm->transA != 0);
        const bool trB = (mm->transB != 0);
        const std::string a_idx = trA
            ? "(batch*mm_k + kk)*m_per + row_in_batch"
            : "(batch*m_per + row_in_batch)*mm_k + kk";
        const std::string b_idx = trB
            ? "(batch*cols + col)*mm_k + kk"
            : "(batch*mm_k + kk)*cols + col";
        L << "        const uint row_in_batch = row % m_per;\n";
        L << "        float mm = 0.0;\n";
        L << "        for (uint kk = 0u; kk < mm_k; ++kk)\n";
        L << "            mm += b" << a_slot << "[" << a_idx << "] * b"
          << b_slot << "[" << b_idx << "];\n";
    };

    L << "void main()\n{\n";
    L << "    const uint tid = gl_LocalInvocationID.x;\n";
    L << "    const uint idx = gl_WorkGroupID.x;\n";
    // 行归约：工作组=单行，idx<rows 守卫。列归约(tile)：工作组=256 列 tile，
    // 每线程一列，守卫/列号在 else 分支内给出（col = idx*256 + tid）。
    L << (is_row ? "    if (idx >= rows) return;\n" : "");
    // matmul+行归约（S7 batch）：idx 为全局行（batch*m_per + row），
    // m_per = rows/mm_batch；列归约+matmul 组合不支持（注意力只用行归约）
    if (mm)
    {
        if (!is_row)
            return {};  // matmul+列归约：不支持（生成器保守放弃 → 上层报错）
        L << "    const uint m_per = rows / mm_batch;\n";
        L << "    const uint batch = idx / m_per;\n";
    }

    // 归约结果索引：行=工作组单行（s_red[slot][0]）；列 tile=每线程一列（s_red[slot][tid]）
    const std::string red_idx = is_row ? "0" : "tid";

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
            return "s_red[" + std::to_string(slot_of_instr[op.idx]) + "][" + red_idx + "]";
        case static_cast<uint8_t>(ExprOperandKind::Matmul):
            return "mm";  // 前置 matmul 段输出（emit_mm_decl 声明）
        // S7 索引操作数：Row 为 batch 内行号（row 变量是全局行 idx）；
        // batch 由 idx 分解（仅 matmul 段存在时）
        case static_cast<uint8_t>(ExprOperandKind::Row):
            return mm ? "float(row % m_per)" : "float(row)";
        case static_cast<uint8_t>(ExprOperandKind::Col):
            return "float(col)";
        case static_cast<uint8_t>(ExprOperandKind::Batch):
            return mm ? "float(batch)" : "0.0";
        case static_cast<uint8_t>(ExprOperandKind::Input):
        {
            const ExprView& v = spec.views[op.idx];
            const ExprViewKind vk = static_cast<ExprViewKind>(v.kind);
            if (expr_view_is_reduce(vk))
                return "s_red[" + std::to_string(slot_of_view[op.idx]) + "][" + red_idx + "]";
            if (vk == ExprViewKind::RowBroadcast)
                return "b" + std::to_string(op.idx) + "[row]";
            if (vk == ExprViewKind::ColBroadcast)
                return "b" + std::to_string(op.idx) + "[col]";
            // Linear / RotateHalf / RowMod / RowGather / BatchMod：索引映射内联；
            // vp 槽 = 此前运行时参数视图个数
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

    // 寄存器声明（先声明后赋值，兼容 IR-B liveness 复用同号寄存器）；
    // 每个 emit_instrs 调用点所在作用域内须先调用一次。
    const auto emit_reg_decl = [&]()
    {
        if (spec.num_regs == 0)
            return;
        L << "        float r0";
        for (std::uint32_t r = 1; r < spec.num_regs; ++r) L << ", r" << r;
        L << ";\n";
    };

    // 直线指令展开（跳过归约指令；归约结果经 s_red 访问；纯赋值，寄存器已声明）
    const auto emit_instrs = [&](std::size_t begin, std::size_t end)
    {
        // matmul 段：当前元素 (row,col) 的 mm 值（每元素求值作用域一次）
        emit_mm_decl();
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
                L << "        " << dst << " = " << a << " " << s << " "
                  << operand(ins.b) << ";\n";
                break;
            }
            case ExprOp::Max: case ExprOp::Min:
            {
                const char* s = (op == ExprOp::Max) ? "max" : "min";
                L << "        " << dst << " = " << s << "(" << a
                  << ", " << operand(ins.b) << ");\n";
                break;
            }
            case ExprOp::Lt: case ExprOp::Le: case ExprOp::Gt:
            case ExprOp::Ge: case ExprOp::Eq: case ExprOp::Ne:
            {
                bool cmp = false;
                const char* s = glsl_binary_op(op, cmp);
                L << "        " << dst << " = (" << a << " " << s << " "
                  << operand(ins.b) << ") ? 1.0 : 0.0;\n";
                break;
            }
            case ExprOp::Neg:
                L << "        " << dst << " = -(" << a << ");\n";
                break;
            case ExprOp::Exp: case ExprOp::Log: case ExprOp::Sqrt:
            case ExprOp::Rsqrt: case ExprOp::Abs: case ExprOp::Tanh:
            {
                const char* s = glsl_unary_op(op);
                L << "        " << dst << " = " << s << "(" << a << ");\n";
                break;
            }
            case ExprOp::Select:
                L << "        " << dst << " = (" << a << " != 0.0) ? "
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

    // 归约：warp shuffle 蝴蝶归约（subgroup）替代共享内存树形归约。
    //   第 1 步：warp 内 subgroupAdd/subgroupMax → 每 warp 一个部分和（零共享/屏障）
    //   第 2 步：首 warp 归约全部 warp 部分和 → s_red[slot][0]（屏障后全线程可见）
    // 真机(NVIDIA, prime-run)大矩阵实测有效（combine 步骤不再是多步 barrier 串行）。
    const auto emit_tree_reduce = [&](int slot, bool is_max)
    {
        const std::string s = "s_red[" + std::to_string(slot) + "]";
        const char* sub = is_max ? "subgroupMax" : "subgroupAdd";
        const std::string idt = is_max ? "uintBitsToFloat(0xff800000u)" : "0.0";
        const std::string vn = "v" + std::to_string(slot);   // 按槽唯一命名，避免重定义
        const std::string wn = "w" + std::to_string(slot);
        L << "    barrier();\n";
        L << "    float " << vn << " = " << s << "[tid];\n";
        L << "    " << vn << " = " << sub << "(" << vn << ");\n";
        L << "    if (gl_SubgroupInvocationID == 0u) " << s << "[gl_SubgroupID] = " << vn << ";\n";
        L << "    barrier();\n";
        L << "    if (gl_SubgroupID == 0u) {\n";
        L << "        float " << wn << " = (gl_SubgroupInvocationID < gl_NumSubgroups) ? "
          << s << "[gl_SubgroupInvocationID] : " << idt << ";\n";
        L << "        " << wn << " = " << sub << "(" << wn << ");\n";
        L << "        if (gl_SubgroupInvocationID == 0u) " << s << "[0] = " << wn << ";\n";
        L << "    }\n";
        L << "    barrier();\n";
    };

    // 组合 4 路累加器（sum/max）
    const auto combine4 = [&](bool is_max, const std::string& a0, const std::string& a1,
                              const std::string& a2, const std::string& a3) {
        return combine(is_max, combine(is_max, combine(is_max, a0, a1), a2), a3);
    };

    if (is_row)
    {
        // ═══ 行归约（工作组单行、跨线程跨步 + 树形归约；合并访问）═══
        const std::string loop_decl =
            "    for (uint c = tid; c < cols; c += 256u) {\n        const uint row = idx;\n        const uint col = c;\n";
        const std::string out_idx = "row*cols + col";

        // ── 归约视图 pass（多累加器，跨步 4*256）──
        for (std::size_t k = 0; k < n_inputs; ++k)
        {
            const ExprView& v = spec.views[k];
            if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
                continue;
            const int slot = slot_of_view[k];
            const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
            const std::string init = is_max ? "uintBitsToFloat(0xff800000u)" : "0.0";
            L << "\n    // 归约视图 " << k << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        uint i = tid;\n";
            L << "        for (; i + 3u*256u < cols; i += 4u*256u) {\n";
            L << "            acc0 = " << combine(is_max, "acc0", "b" + std::to_string(k) + "[idx*cols + i]") << ";\n";
            L << "            acc1 = " << combine(is_max, "acc1", "b" + std::to_string(k) + "[idx*cols + i+256u]") << ";\n";
            L << "            acc2 = " << combine(is_max, "acc2", "b" + std::to_string(k) + "[idx*cols + i+512u]") << ";\n";
            L << "            acc3 = " << combine(is_max, "acc3", "b" + std::to_string(k) + "[idx*cols + i+768u]") << ";\n";
            L << "        }\n";
            L << "        for (; i < cols; i += 256u)\n";
            L << "            acc0 = " << combine(is_max, "acc0", "b" + std::to_string(k) + "[idx*cols + i]") << ";\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
            emit_tree_reduce(slot, is_max);
        }

        // ── 归约指令 pass（多累加器跨步循环 + 树形归约）──
        // 用 4 路独立标量累加器（非数组索引，见 matmul-opt.md 教训）打破 acc 串行依赖链，
        // 提升 ILP；跨步 4*256 与归约视图 pass 一致。归约结合序改变（与视图 pass 相同），
        // 测试容差已覆盖。
        for (std::size_t ri = 0; ri < spec.instrs.size(); ++ri)
        {
            const ExprInstr& R = spec.instrs[ri];
            if (!expr_op_is_reduce(static_cast<ExprOp>(R.op)))
                continue;
            const int slot = slot_of_instr[R.dst];
            const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
            const bool src_is_reg =
                (R.a.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                 R.a.kind == static_cast<uint8_t>(ExprOperandKind::Fanout));
            const std::string src = src_is_reg
                ? "r" + std::to_string(static_cast<int>(R.a.idx)) : operand(R.a);
            const std::string init = is_max ? "uintBitsToFloat(0xff800000u)" : "0.0";
            L << "\n    // 归约指令 " << ri << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        uint c = tid;\n";
            L << "        for (; c + 3u*256u < cols; c += 4u*256u) {\n";
            for (int k = 0; k < 4; ++k)
            {
                L << "            { const uint row = idx; const uint col = c + "
                  << std::to_string(256 * k) << "u; ";
                emit_reg_decl();
                emit_instrs(0, ri);
                L << " acc" << k << " = " << combine(is_max, "acc" + std::to_string(k), src)
                  << "; }\n";
            }
            L << "        }\n";
            L << "        for (; c < cols; c += 256u) {\n";
            L << "            { const uint row = idx; const uint col = c; ";
            emit_reg_decl();
            emit_instrs(0, ri);
            L << " acc0 = " << combine(is_max, "acc0", src) << "; }\n";
            L << "        }\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
            emit_tree_reduce(slot, is_max);
        }

        // ── 输出 pass（行）──
        L << "\n    // 输出\n";
        L << "    if (vector_out == 1u) {\n";
        L << "        if (tid == 0u) {\n";
        if (last_is_reduce)
            L << "            bout[idx] = s_red[" << slot_of_instr[spec.instrs.back().dst] << "][0];\n";
        else
        {
            L << "            const uint row = idx;\n            const uint col = 0u;\n";
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "            bout[idx] = r" << static_cast<int>(spec.instrs.back().dst) << ";\n";
        }
        L << "        }\n";
        L << "        return;\n";
        L << "    }\n";
        L << loop_decl;
        if (last_is_reduce)
            L << "        bout[" << out_idx << "] = s_red[" << slot_of_instr[spec.instrs.back().dst] << "][0];\n";
        else
        {
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "        bout[" << out_idx << "] = r" << static_cast<int>(spec.instrs.back().dst) << ";\n";
        }
        L << "    }\n";
    }
    else
    {
        // ═══ 列归约 tile（合并访问）：每工作组 256 列、每线程一整列，无需跨线程归约 ═══
        L << "    const uint col = idx * 256u + tid;\n";
        L << "    if (col >= cols) return;\n";

        // ── 归约视图 pass（tile：每线程一整列，多累加器跨行顺序读，合并访问）──
        for (std::size_t k = 0; k < n_inputs; ++k)
        {
            const ExprView& v = spec.views[k];
            if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
                continue;
            const int slot = slot_of_view[k];
            const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
            const std::string init = is_max ? "uintBitsToFloat(0xff800000u)" : "0.0";
            L << "\n    // 归约视图 " << k << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        uint i = 0u;\n";
            L << "        for (; i + 3u < rows; i += 4u) {\n";
            L << "            acc0 = " << combine(is_max, "acc0", "b" + std::to_string(k) + "[i*cols + col]") << ";\n";
            L << "            acc1 = " << combine(is_max, "acc1", "b" + std::to_string(k) + "[(i+1u)*cols + col]") << ";\n";
            L << "            acc2 = " << combine(is_max, "acc2", "b" + std::to_string(k) + "[(i+2u)*cols + col]") << ";\n";
            L << "            acc3 = " << combine(is_max, "acc3", "b" + std::to_string(k) + "[(i+3u)*cols + col]") << ";\n";
            L << "        }\n";
            L << "        for (; i < rows; ++i)\n";
            L << "            acc0 = " << combine(is_max, "acc0", "b" + std::to_string(k) + "[i*cols + col]") << ";\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
        }

        // ── 归约指令 pass（tile：多累加器，跨行顺序读，合并访问）──
        // 4 路独立标量累加器打破 acc 串行依赖链，提升 ILP（与行归约一致）。
        for (std::size_t ri = 0; ri < spec.instrs.size(); ++ri)
        {
            const ExprInstr& R = spec.instrs[ri];
            if (!expr_op_is_reduce(static_cast<ExprOp>(R.op)))
                continue;
            const int slot = slot_of_instr[R.dst];
            const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
            const bool src_is_reg =
                (R.a.kind == static_cast<uint8_t>(ExprOperandKind::Reg) ||
                 R.a.kind == static_cast<uint8_t>(ExprOperandKind::Fanout));
            const std::string src = src_is_reg
                ? "r" + std::to_string(static_cast<int>(R.a.idx)) : operand(R.a);
            const std::string init = is_max ? "uintBitsToFloat(0xff800000u)" : "0.0";
            L << "\n    // 归约指令 " << ri << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        uint i = 0u;\n";
            L << "        for (; i + 3u < rows; i += 4u) {\n";
            for (int k = 0; k < 4; ++k)
            {
                L << "            { const uint row = i + " << std::to_string(k) << "u; ";
                emit_reg_decl();
                emit_instrs(0, ri);
                L << " acc" << k << " = " << combine(is_max, "acc" + std::to_string(k), src)
                  << "; }\n";
            }
            L << "        }\n";
            L << "        for (; i < rows; ++i) {\n";
            L << "            { const uint row = i; ";
            emit_reg_decl();
            emit_instrs(0, ri);
            L << " acc0 = " << combine(is_max, "acc0", src) << "; }\n";
            L << "        }\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
        }

        // ── 输出 pass（列 tile）──
        L << "\n    // 输出\n";
        L << "    if (vector_out == 1u) {\n";
        if (last_is_reduce)
            L << "        bout[col] = s_red[" << slot_of_instr[spec.instrs.back().dst] << "][tid];\n";
        else
        {
            L << "        const uint row = 0u;\n";
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "        bout[col] = r" << static_cast<int>(spec.instrs.back().dst) << ";\n";
        }
        L << "        return;\n";
        L << "    }\n";
        if (last_is_reduce)
        {
            L << "    for (uint row = 0u; row < rows; ++row)\n";
            L << "        bout[row*cols + col] = s_red[" << slot_of_instr[spec.instrs.back().dst] << "][tid];\n";
        }
        else
        {
            L << "    for (uint row = 0u; row < rows; ++row) {\n";
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "        bout[row*cols + col] = r" << static_cast<int>(spec.instrs.back().dst) << ";\n";
            L << "    }\n";
        }
    }
    L << "}\n";
    return L.str();
}

// ═══════════════════════════════════════════════════════════════════════════
//  GlslEmitter — GLSL 后端 emitter（IR-D）
//
//  把上述 generate_glsl / generate_glsl_reduce 封装为 ExprEmitter 接口实现，
//  使生成器工具（gen_fused）可经统一接口选择后端，不再与 GLSL 绑定。
//  自由函数 generate_glsl / generate_glsl_reduce 保留为便捷入口
//  （内部转发到 GlslEmitter），向后兼容既有调用方。
// ═══════════════════════════════════════════════════════════════════════════
class GlslEmitter final : public ExprEmitter
{
public:
    [[nodiscard]] std::string_view name() const noexcept override
    { return "glsl"; }

    [[nodiscard]] std::string generate(
        const std::string& name_, const ExprSpec& spec) override
    { return nn::generate_glsl(name_, spec); }

    [[nodiscard]] std::string generate_reduce(
        const std::string& name_, const ExprSpec& spec) override
    { return nn::generate_glsl_reduce(name_, spec); }
};

// 登记到 emitter 注册表（静态初始化；重复包含无害——同名拒绝覆盖）
inline const bool kGlslEmitterRegistered =
    emitter_registry::register_backend("glsl",
        []() -> std::unique_ptr<ExprEmitter> {
            return std::make_unique<GlslEmitter>();
        });

} // namespace nn

#endif // NN_EXPR_GLSL_GEN_HPP
