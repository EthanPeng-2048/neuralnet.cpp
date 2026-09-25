#pragma once

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

#include <bit>
#include <cmath>
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
                           std::uint32_t vp_slot = 0,
                           ExprPrecSig sig = 0)
{
    const std::string buf = "b" + std::to_string(buf_id);
    // 该输入的存储类型：f16 时**视图内部**的算术/索引读取必须显式转 f32
    // （GL_EXT_shader_16bit_storage 只允许存储，不允许 f16 算术：
    //  `-b0[i]` / `uint(b0[i])` 都会被 glslc 拒绝）。视图外层的整体转换由
    // 调用方（emit_scalar_chain / vec4 读取）负责。
    const bool f16_buf = expr_prec_sig_in_f16(sig, buf_id);
    const auto rw = [&](const std::string& idx) -> std::string {
        return f16_buf ? ("float(" + buf + "[" + idx + "])")
                       : (buf + "[" + idx + "]");
    };
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
           << neg << rw(hi) << " : " << rw(lo) << ")";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowMod):
    {
        // 周期为运行时视图参数（push constant vpN），形状无关融合
        os << buf << "[(" << row_var << " % vp" << std::to_string(vp_slot) << ") * cols + "
           << col_var << "]";
        return;
    }
    case static_cast<uint8_t>(ExprViewKind::RowAccess):
    {
        // 行偏移+取模：data[(offset + r % mod)*cols + c]。
        // vp_slot=mod（op），vp_slot+1=offset（param2），双 vp 槽（形状无关）。
        os << buf << "[(" << "vp" << std::to_string(vp_slot + 1) << " + ("
           << row_var << " % vp" << std::to_string(vp_slot) << ")) * cols + "
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
        // f16 标签槽（少见）需先转 f32 再取整：uint(float16_t) 不合法。
        os << buf << "[uint("
           << (expr_prec_sig_in_f16(sig, v.param) ? "float(b" : "b") << v.param
           << "[" << col_var << "]"
           << (expr_prec_sig_in_f16(sig, v.param) ? ")" : "") << ") * cols + "
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
//        （64×64，每线程 4×4 寄存器分块 acc[4][4]，dot() = 4 FMA）；
//        K 方向按 BK=32 分块（双缓冲）：load_tiles 协作加载 A/B 分块到
//        vec4 共享内存（转置感知、合并访问），tile t+1 的全局加载与 tile t
//        的计算并行发射、每 tile 单 barrier——暴露的加载延迟被计算覆盖。
//        40HX 变体实测 trade-off（深=feedforward/大 batch 训练，浅=小 batch 推理）：
//          · 原版 BK=32 单缓冲(16KB)：深网格基线，浅网格无流水
//          · BK=16 双缓冲(16KB占用率不变)（% = vs 原版**收益**，正=更快，
//            故深点 -2% 即回退）：浅 +2~6%，深 -2%（barrier 频率翻倍
//            而深网格延迟已被跨块调度掩盖 → 流水无用只剩 barrier 成本）
//          · BK=32 双缓冲(32KB)：barrier 节奏=原版，32KB 占用率砍半由深网格
//            WG 余量吸收——**四点 A/B 最终采用**（对照=单缓冲原版；数字为
//            **耗时**变化、负=更快；与 AGENTS §12 ② 同口径）：浅 linear
//            1024³ -1.9%、batch512 -4.9%，深 feedforward/batch4096 ±0（18/18 测试绿）
//    - 尾逐元素链编译为 eval_tail(mm, row, col) 函数（GLSL 内联零开销），
//      写回时每个输出元素调用一次（Matmul 操作数 → mm，"虚拟寄存器 0"）。
//    - transA/transB 是**结构**（进 key）→ 索引表达式硬编码进 shader；
//      mm_k 是**形状参数**（不进 key）→ push constant，运行时填充
//      （同结构不同 K 共享一个融合 shader）。
//    - 布局：bindings 输入 0..N-1 + 输出 N（A/B 就是其中的两个输入槽）；
//      push constants: uint count, uint cols, uint rows, uint mm_k,
//                      uint mm_batch, [uint vp0..], [float c0..], [float rp0..]；
//      dispatch: (ceil(cols/BLOCK), ceil(rows/BLOCK), mm_batch)（见
//      EXPR_MATMUL_TILE/BLOCK，后端 run_fused_gpu 与生成器共用）。
//    - 归约指令（RowSum/RowMax/...）出现在尾链时属 S5 归约组合，由
//      generate_glsl_reduce 处理（本生成器遇到归约指令返回空 → 上层报错）。
// ═══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::string generate_glsl_matmul(
    const std::string& name, const ExprSpec& spec, ExprPrecSig sig = 0)
{
    const MatmulSpec& mm = *spec.matmul;
    const std::size_t n_inputs = spec.views.size();
    const std::uint32_t a_slot = mm.a_input;
    const std::uint32_t b_slot = mm.b_input;
    const bool trA = (mm.transA != 0);
    const bool trB = (mm.transB != 0);
    // ── 带类型变体（Phase 2 in-kernel f16）───────────────────────────────
    // sig==0 → 全 f32，GLSL 与迁移前逐字节相同（零回归）。sig!=0 → 对应槽
    // 的缓冲区声明为 float16_t，**在全局加载处**统一转 f32（共享 tile / 内层
    // VFMA 累加 / 尾逐元素链全部保持 f32，符合 §7.2「f32 参考 + 输出舍入」）。
    // 形状/分块/双缓冲/barrier 节奏与 f32 版完全一致（只有加载粒度变化）。
    const bool sign_f16 = (sig != 0);
    const bool a_f16 = sign_f16 && expr_prec_sig_in_f16(sig, a_slot);
    const bool b_f16 = sign_f16 && expr_prec_sig_in_f16(sig, b_slot);
    const bool out_f16 = sign_f16 && expr_prec_sig_out_f16(sig);
    constexpr std::uint32_t T = EXPR_MATMUL_TILE;    // 线程（16）
    constexpr std::uint32_t B = EXPR_MATMUL_BLOCK;   // 输出块（64）
    // 变体实验（40HX，见头注释 trade-off 表）：
    //   BK=16+双缓冲(16KB)：浅网格 +2~6%，深网格 barrier 频率翻倍 -2%（真回退）
    //   BK=32+双缓冲(32KB)：barrier 节奏=原版单缓冲，代价是共享 32KB
    //   → blocks/SM 砍半（原版 16KB 单缓冲 = 深网格基线）
    // 2026-09-24 OP/融合统一 A/B：BK=32 (32KB) vs BK=16 (16KB)，四点取
    // best（3 样本）——融合侧 linear 浅(b1024)/深(b4096 fwd+train) 与
    // OP 级四点一并复测（历史"深网格 BK16 -2%"结论本轮推翻）：
    constexpr std::uint32_t BK = 16;                 // K 分块宽度
    constexpr std::uint32_t BK4 = BK / 4;            // vec4 数/行（8）
    // 每线程协作加载 vec4 数/矩阵（B×BK4 slots / 256 线程）
    constexpr std::uint32_t VPT = (B * BK4) / (T * T);  // BK=32 → 2；BK=16 → 1

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · matmul 分块 4×4 双缓冲），请勿手动编辑 ──\n";
    L << "// 表达式: " << name;
    if (sign_f16)
        L << "   [精度 " << expr_prec_sig_str(sig, n_inputs) << "]";
    L << "\n";
    L << "#version 450\n\n";
    if (sign_f16)
        L << "#extension GL_EXT_shader_16bit_storage : require\n\n";
    L << "layout(local_size_x = " << T << ", local_size_y = " << T << ") in;\n\n";
    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { " << ((sign_f16 && expr_prec_sig_in_f16(sig, i)) ? "float16_t" : "float")
          << " b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { " << (out_f16 ? "float16_t" : "float")
      << " bout[]; };\n";
    // vec4 别名视图（A/B 全局快路径专用；同 binding 双声明 std430，op 级
    // matmul_tiled 同款）。a_slot==b_slot 时不发别名、快路径整体关闭。
    // f16 槽的别名用 uvec2（4×half = 8B，std430 步长恰好 8）→ 由
    // unpackHalf2x16 解出 4 个 f32；f32 槽仍是 vec4。
    const bool vec4_ok = (a_slot != b_slot);
    if (vec4_ok)
    {
        const auto alias_type = [&](std::uint32_t slot) -> const char* {
            return (sign_f16 && expr_prec_sig_in_f16(sig, slot)) ? "uvec2" : "vec4";
        };
        L << "layout(std430, binding = " << a_slot << ") readonly buffer BufAv4 { "
          << alias_type(a_slot) << " b" << a_slot << "v4[]; };\n";
        L << "layout(std430, binding = " << b_slot << ") readonly buffer BufBv4 { "
          << alias_type(b_slot) << " b" << b_slot << "v4[]; };\n";
    }
    L << "\n";
    // 别名槽的 vec4 读取表达式（f16 → unpackHalf2x16；f32 → 直接别名）
    const auto vec4_load = [&](std::uint32_t slot, const std::string& idx) -> std::string {
        const std::string arr = "b" + std::to_string(slot) + "v4[" + idx + "]";
        if (sign_f16 && expr_prec_sig_in_f16(sig, slot))
            return "vec4(unpackHalf2x16(" + arr + ".x), unpackHalf2x16(" + arr + ".y))";
        return arr;
    };
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
    const std::uint32_t n_rp = static_cast<std::uint32_t>(spec.rparams.size());
    for (std::uint32_t i = 0; i < n_rp; ++i)
        L << "    float rp" << i << ";\n";
    L << "};\n\n";
    // 共享内存分块（op 级 matmul_tiled 配方：vec4 沿 m/n，16B 对齐，双缓冲 [stage]）：
    //   Ash[stage][k][m/4]：vec4 = 同 k 的 4 个连续行（Bsh 同构沿列）→ 主循环
    //     每 k 仅 2 次 LDS + 4 VFMA.128。旧版 vec4 沿 k + 16 dot ≈5× 指令发射，
    //     是融合 matmul 实测落后 op 级 ~2× 的主因（2026-09-24 由 fused_49fa
    //     与 op 级同形状对拍 7.2 vs 3.15ms 定位）；同 ty/tx 广播读无 bank 冲突。
    //   BK=32 + 双缓冲：[2][32][16]vec4×2 = 32KB（布局改向后**总量不变**，
    //   旧 [2][8][64]vec4 同为 32KB；BK16 变体已 A/B 深网格 -2% 不取，
    //   barrier 节奏保持原版）；>原版单缓冲 16KB → blocks/SM 砍半，
    //   深网格靠 WG 余量补、浅网格靠流水补，实测定夺；
    //   需设备 maxComputeSharedMemorySize ≥ 32KB（Vulkan 规范下限仅 16KB，
    //   合规低端设备会在 pipeline 创建时响亮失败——40HX 等目标卡无虞）；
    //   stage 交替：tile t 用 t&1。
    L << "shared vec4 Ash[2][" << BK << "][" << B / 4u << "];\n";
    L << "shared vec4 Bsh[2][" << BK << "][" << B / 4u << "];\n\n";

    // transA/transB 感知的全局内存加载表达式（S7 batch：A/B 按 batch 垂直
    // 切分，batch*m_per 为 A 行偏移 / batch*mm_k 为 B k 偏移；row/col 为
    // **batch 内**坐标，block 偏移由调用方并入）
    const auto a_load = [&](const std::string& row_e, const std::string& k_e)
    {
        const std::string raw = trA
            ? "b" + std::to_string(a_slot) + "[((batch*mm_k + (" + k_e + "))*m_per + (" + row_e + "))]"
            : "b" + std::to_string(a_slot) + "[((batch*m_per + (" + row_e + "))*mm_k + (" + k_e + "))]";
        return a_f16 ? ("float(" + raw + ")") : raw;
    };
    const auto b_load = [&](const std::string& col_e, const std::string& k_e)
    {
        const std::string raw = trB
            ? "b" + std::to_string(b_slot) + "[((batch*cols + (" + col_e + "))*mm_k + (" + k_e + "))]"
            : "b" + std::to_string(b_slot) + "[((batch*mm_k + (" + k_e + "))*cols + (" + col_e + "))]";
        return b_f16 ? ("float(" + raw + ")") : raw;
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
            case static_cast<uint8_t>(ExprOperandKind::RParam):
                return "rp" + std::to_string(op.idx);
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
                        static_cast<ExprViewKind>(spec.views[j].kind)))
                    vp += expr_view_runtime_param_slots(
                        static_cast<ExprViewKind>(spec.views[j].kind));
            L << "    const float v" << i << " = ";
            const bool tail_f16 = sign_f16 && expr_prec_sig_in_f16(sig, i);
            if (tail_f16) L << "float(";   // f16 读 → 显式转 f32（算术仍在 f32）
            glsl_view_read(L, spec.views[i], static_cast<std::uint32_t>(i),
                           "grow*cols + col", "grow", "col", vp, sig);
            if (tail_f16) L << ")";
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

    // ── load_tiles：协作加载 A/B 分块（BK×B）→ 指定 stage。双缓冲流水的
    //    发射端：前奏调 load_tiles(0,0)，主循环内 if (t+1<num_tiles)
    //    load_tiles(t+1, nstage) 与 tile t 的计算并行 ─────────────────────
    //    分组 = 沿内存连续维取 vec4（op 级 matmul_tiled v3 同款，但 trans 进
    //    key → 生成期定死单路径、无运行时 trans 分支）；%4 对齐与边界是
    //    uniform 分支（mm_k/m_per/cols 运行时才知道），未对齐/尾块走同分组
    //    标量回退 → 每个 (k,m) 恰被写一次，语义与旧版一致。共享槽位恒用
    //    tile 内坐标，越界判/全局地址才用 block_row/col + tile 偏移。
    L << "void load_tiles(uint t, uint stage)\n{\n";
    L << "    const uint tid = gl_LocalInvocationID.y * " << T
      << "u + gl_LocalInvocationID.x;\n";
    L << "    const uint batch = gl_WorkGroupID.z;\n";
    L << "    const uint m_per = rows / mm_batch;\n";
    L << "    const uint block_row = gl_WorkGroupID.y * " << B << "u;\n";
    L << "    const uint block_col = gl_WorkGroupID.x * " << B << "u;\n";
    L << "    for (uint l = 0u; l < " << VPT << "u; ++l)\n";
    L << "    {\n";
    L << "    const uint e = tid * " << VPT << "u + l;\n";
    if (trA)
    {
        // A 存 (K,M)：沿行连续 → vec4 整读 4 行同 k，直写共享
        L << "    const uint kAL = e / " << (B / 4u) << "u;\n";
        L << "    const uint kAG = t * " << BK << "u + kAL;\n";
        L << "    const uint mA4 = e % " << (B / 4u) << "u;\n";
        L << "    const uint arG = block_row + mA4 * 4u;\n";
        L << "    if (" << (vec4_ok ? "((m_per & 3u) == 0u)" : "false")
          << " && kAG < mm_k && arG < m_per)\n";
        L << "    {\n";
        L << "        Ash[stage][kAL][mA4] = "
          << vec4_load(a_slot, "(((batch*mm_k + (kAG))*m_per + (arG))) >> 2") << ";\n";
        L << "    }\n    else\n    {\n";
        L << "        for (uint j = 0u; j < 4u; ++j)\n        {\n";
        L << "            const float v = (arG + j < m_per && kAG < mm_k) ? "
          << a_load("arG + j", "kAG") << " : 0.0;\n";
        L << "            Ash[stage][kAL][mA4][j] = v;\n";
        L << "        }\n    }\n";
    }
    else
    {
        // A 行主序：沿 k 连续 → vec4 散写 4 个共享行（op 级 k_contig 同构）
        L << "    const uint kAL = (e % " << BK4 << "u) * 4u;\n";
        L << "    const uint kAG = t * " << BK << "u + kAL;\n";
        L << "    const uint arT = e / " << BK4 << "u;\n";
        L << "    const uint arG = block_row + arT;\n";
        L << "    if (" << (vec4_ok ? "((mm_k & 3u) == 0u)" : "false")
          << " && arG < m_per && kAG < mm_k)\n";
        L << "    {\n";
        L << "        const vec4 v = "
          << vec4_load(a_slot, "(((batch*m_per + (arG))*mm_k + (kAG))) >> 2") << ";\n";
        L << "        Ash[stage][kAL    ][arT >> 2][arT & 3u] = v.x;\n";
        L << "        Ash[stage][kAL + 1u][arT >> 2][arT & 3u] = v.y;\n";
        L << "        Ash[stage][kAL + 2u][arT >> 2][arT & 3u] = v.z;\n";
        L << "        Ash[stage][kAL + 3u][arT >> 2][arT & 3u] = v.w;\n";
        L << "    }\n    else\n    {\n";
        L << "        for (uint j = 0u; j < 4u; ++j)\n        {\n";
        L << "            const float v = (arG < m_per && kAG + j < mm_k) ? "
          << a_load("arG", "kAG + j") << " : 0.0;\n";
        L << "            Ash[stage][kAL + j][arT >> 2][arT & 3u] = v;\n";
        L << "        }\n    }\n";
    }
    if (trB)
    {
        // B 存 (N,K)：沿 k 连续 → vec4 散写 4 个共享行
        L << "    const uint kBL = (e % " << BK4 << "u) * 4u;\n";
        L << "    const uint kBG = t * " << BK << "u + kBL;\n";
        L << "    const uint nT = e / " << BK4 << "u;\n";
        L << "    const uint bcG = block_col + nT;\n";
        L << "    if (" << (vec4_ok ? "((mm_k & 3u) == 0u)" : "false")
          << " && bcG < cols && kBG < mm_k)\n";
        L << "    {\n";
        L << "        const vec4 v = "
          << vec4_load(b_slot, "(((batch*cols + (bcG))*mm_k + (kBG))) >> 2") << ";\n";
        L << "        for (uint j = 0u; j < 4u; ++j)\n";
        L << "            Bsh[stage][kBL + j][nT >> 2][nT & 3u] = v[j];\n";
        L << "    }\n    else\n    {\n";
        L << "        for (uint j = 0u; j < 4u; ++j)\n        {\n";
        L << "            const float v = (bcG < cols && kBG + j < mm_k) ? "
          << b_load("bcG", "kBG + j") << " : 0.0;\n";
        L << "            Bsh[stage][kBL + j][nT >> 2][nT & 3u] = v;\n";
        L << "        }\n    }\n";
    }
    else
    {
        // B (K,N) 行主序：沿列连续 → vec4 整读 4 列同 k，直写共享
        L << "    const uint kBL = e / " << (B / 4u) << "u;\n";
        L << "    const uint kBG = t * " << BK << "u + kBL;\n";
        L << "    const uint nB4 = e % " << (B / 4u) << "u;\n";
        L << "    const uint bcG = block_col + nB4 * 4u;\n";
        L << "    if (" << (vec4_ok ? "((cols & 3u) == 0u)" : "false")
          << " && kBG < mm_k && bcG < cols)\n";
        L << "    {\n";
        L << "        Bsh[stage][kBL][nB4] = "
          << vec4_load(b_slot, "(((batch*mm_k + (kBG))*cols + (bcG))) >> 2") << ";\n";
        L << "    }\n    else\n    {\n";
        L << "        for (uint j = 0u; j < 4u; ++j)\n        {\n";
        L << "            const float v = (kBG < mm_k && bcG + j < cols) ? "
          << b_load("bcG + j", "kBG") << " : 0.0;\n";
        L << "            Bsh[stage][kBL][nB4][j] = v;\n";
        L << "        }\n    }\n";
    }
    L << "    }\n";
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
    // 4×4 寄存器累加器（本线程负责 16 个输出元素；vec4 acc[i] = 第 i 行的
    // 4 列 → 内层每 k 4 条 VFMA.128，尾链 acc[i][j] 索引语义不变）
    L << "    vec4 acc[4];\n";
    L << "    for (uint i = 0u; i < 4u; ++i)\n";
    L << "        acc[i] = vec4(0.0);\n";

    // 前奏：加载 tile 0 → stage 0；此后每迭代发射 t+1 的加载（stage^1），
    // 与 tile t 的计算并行，单 barrier 收口（与 matmul_tiled.comp v2 同构：
    // 暴露的全局加载延迟被 tile t 的计算覆盖）
    L << "    const uint num_tiles = (mm_k + " << BK - 1 << "u) / " << BK << "u;\n";
    L << "    load_tiles(0u, 0u);\n";
    L << "    barrier();\n";
    L << "    uint stage = 0u;\n";
    L << "    for (uint t = 0u; t < num_tiles; ++t)\n";
    L << "    {\n";
    L << "        const uint nstage = stage ^ 1u;\n";
    L << "        if (t + 1u < num_tiles)\n";
    L << "            load_tiles(t + 1u, nstage);\n";
    // 内层：每 k 读 2 个 vec4（Ash[k][ty] = 4 行、Bsh[k][tx] = 4 列）+ 4
    // VFMA.128（op 级 matmul_tiled 同款；同槽广播读无 bank 冲突）
    L << "        for (uint k = 0u; k < " << BK << "u; ++k)\n";
    L << "        {\n";
    L << "            const vec4 av = Ash[stage][k][ty];\n";
    L << "            const vec4 bv = Bsh[stage][k][tx];\n";
    L << "            acc[0] += av.x * bv;\n";
    L << "            acc[1] += av.y * bv;\n";
    L << "            acc[2] += av.z * bv;\n";
    L << "            acc[3] += av.w * bv;\n";
    L << "        }\n";
    L << "        barrier();\n";
    L << "        stage = nstage;\n";
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
    L << "                    = " << (out_f16 ? "float16_t(" : "")
      << "eval_tail(acc[i][j], rr, cc, batch)" << (out_f16 ? ")" : "") << ";\n";
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
// ═══════════════════════════════════════════════════════════════════════════
//  fold 生成器（P-C1 分块状态归约 FoldSpec）
//
//  线程映射：**每线程一行**（WG=256，dispatch = ceil(rows/256)）——状态是
//  每行标量、块循环由该线程串行推进。与 CpuEngine::eval_fold_impl 完全同序
//  （同分块 EXPR_FOLD_BLOCK / 同指令序 / 同归约结合序 / 同尾块 gate）→
//  GPU 对拍**期望逐位一致**（P-C2 向量化时才引入 subgroup 归约差异）。
//  PC 形态：{count, cols, rows, vector_out, fold_k} + vp… + consts… + rparams…
//    cols = 输入列数 K（glsl_view_read 索引用）；fold_k = K（块循环上界）。
//    与后端 run_fused_gpu 的 fold 形态组包必须逐字段一致（PC 固定头教训 4.10）。
//  寄存器物化：状态/标量类 = float r{n}；元素类 = float e{n}[BLOCK]（块内每
//    kb 一值）；块归约 dst = float red{n}（行标量，经 Reduce 操作数广播）。
//  状态初值一律 bit_cast → uintBitsToFloat 字面量（±inf/任意 float 零精度损失，
//    同 reduce 生成器 -inf 风格）。
// ═══════════════════════════════════════════════════════════════════════════
inline std::string generate_glsl_fold(const std::string& name, const ExprSpec& spec)
{
    // 形态由 validate_expr_spec 保证；防御性拒绝一切越界组合
    if (!spec.fold || spec.matmul || !spec.instrs.empty())
        return {};
    const FoldSpec& f = *spec.fold;
    auto cls = expr_fold_classify(f, spec.num_regs);
    if (!cls)
        return {};
    const std::vector<uint8_t>& is_elem = *cls;
    const std::size_t n_inputs = spec.views.size();

    std::vector<uint8_t> is_block_reduce(spec.num_regs, 0);
    for (const auto& ins : f.body)
        if (expr_op_is_reduce(static_cast<ExprOp>(ins.op)))
            is_block_reduce[ins.dst] = 1;

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · fold 分块状态归约），请勿手动编辑 ──\n";
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
    L << "    uint vector_out;\n";
    L << "    uint fold_k;\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    for (std::size_t i = 0; i < spec.rparams.size(); ++i)
        L << "    float rp" << i << ";\n";
    L << "};\n\n";

    L << "void main()\n{\n";
    L << "    const uint row = gl_GlobalInvocationID.x;\n";
    L << "    if (row >= rows) return;\n";

    // 寄存器物化（元素类 = 块内数组；其余 = 标量）+ 块归约行标量
    for (std::uint32_t r = 0; r < spec.num_regs; ++r)
    {
        if (is_elem[r])
            L << "    float e" << r << "[" << EXPR_FOLD_BLOCK << "];\n";
        else
            L << "    float r" << r << ";\n";
    }
    for (std::uint32_t r = 0; r < spec.num_regs; ++r)
        if (is_block_reduce[r])
            L << "    float red" << r << ";\n";
    // 状态初值（bit_cast → uintBitsToFloat：±inf 与任意 float 均零精度损失）
    for (std::uint8_t s = 0; s < f.num_state; ++s)
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(f.inits[s]);
        L << "    r" << static_cast<int>(s) << " = uintBitsToFloat(0x"
          << std::hex << bits << std::dec << "u);\n";
    }

    // 视图运行时参数槽（同 operand 序数：此前出现的 runtime-param 视图槽数）
    const auto vp_of = [&](std::size_t k) -> std::uint32_t
    {
        std::uint32_t vp = 0;
        for (std::size_t j = 0; j < k; ++j)
            if (expr_view_has_runtime_param(static_cast<ExprViewKind>(spec.views[j].kind)))
                vp += expr_view_runtime_param_slots(static_cast<ExprViewKind>(spec.views[j].kind));
        return vp;
    };
    // 操作数 → GLSL 片段；kbv = 块内下标变量（标量上下文传 "0u"——类别分析
    //   保证标量指令无元素源，该分支不可达；归约/元素上下文传 "kb"）
    const auto operand = [&](const ExprOperand& op, const std::string& kbv) -> std::string
    {
        switch (op.kind)
        {
        default:
        case static_cast<uint8_t>(ExprOperandKind::Reg):
        case static_cast<uint8_t>(ExprOperandKind::Fanout):
            return is_elem[op.idx]
                ? ("e" + std::to_string(op.idx) + "[" + kbv + "]")
                : ("r" + std::to_string(op.idx));
        case static_cast<uint8_t>(ExprOperandKind::Const):
            return "c" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::RParam):
            return "rp" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Reduce):
            return "red" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Input):
        {
            const std::string gk = "(k0 + " + kbv + ")";   // 全局收缩下标
            std::ostringstream os;
            glsl_view_read(os, spec.views[op.idx], op.idx,
                           "(row * cols + " + gk + ")", "row", gk, vp_of(op.idx));
            return os.str();
        }
        }
    };
    // 指令 → 赋值语句（dst_str 已含下标/变量名），直接写入 L
    const auto emit_assign = [&](const ExprInstr& ins, const std::string& dst_str,
                                 const std::string& kbv)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string a = operand(ins.a, kbv);
        if (op == ExprOp::Select)
        {
            L << dst_str << " = (" << a << " != 0.0) ? "
              << operand(ins.b, kbv) << " : " << operand(ins.c, kbv) << ";\n";
            return;
        }
        if (op == ExprOp::Max || op == ExprOp::Min)
        {
            L << dst_str << " = " << (op == ExprOp::Max ? "max" : "min")
              << "(" << a << ", " << operand(ins.b, kbv) << ");\n";
            return;
        }
        bool cmp = false;
        const char* s = glsl_binary_op(op, cmp);
        if (s && *s)
        {
            const std::string b = operand(ins.b, kbv);
            if (cmp)
                L << dst_str << " = (" << a << " " << s << " " << b
                  << ") ? 1.0 : 0.0;\n";
            else
                L << dst_str << " = " << a << " " << s << " " << b << ";\n";
            return;
        }
        if (op == ExprOp::Neg)
        {
            L << dst_str << " = -(" << a << ");\n";
            return;
        }
        L << dst_str << " = " << glsl_unary_op(op) << "(" << a << ");\n";
    };

    // ── 块循环（k0 步进 BLOCK；valid gate 尾块——与 CPU 同序）──────────
    L << "\n    // 分块收缩轴（与 CPU eval_fold 同分块同序 → 逐位一致）\n";
    L << "    for (uint k0 = 0u; k0 < fold_k; k0 += " << EXPR_FOLD_BLOCK << "u) {\n";
    L << "        const uint valid = min(" << EXPR_FOLD_BLOCK << "u, fold_k - k0);\n";
    for (const auto& ins : f.body)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string dst_r = std::to_string(ins.dst);
        if (expr_op_is_reduce(op))
        {
            const bool is_max = (op == ExprOp::RowMax || op == ExprOp::ColMax);
            // max 恒等元 = lowest() 直出（0xFF7FFFFF 即 -FLT_MAX 位型，**禁加
            // 负号**：-(-FLT_MAX)=+FLT_MAX 会让 acc 恒为 inf；亦**禁用 -inf**）：
            //   与 CPU 基准逐位对齐（eval_fold_impl 一律 numeric_limits::lowest()）。
            //   -inf 会在「块内元素全 -inf（如文档掩码屏蔽整块）且状态 m_old=-inf」
            //   时算出 blk_m=-inf → dm = m_old − m = −inf−−inf = **NaN** →
            //   l/O 全污染（fold 分块状态进位特有；两趟式对整行求 m 不触发）。
            //   lowest() 有限 → dm=−inf → exp(−inf)=0，全屏蔽块贡献恒 0，语义正确。
            L << "        { float acc = "
              << (is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0")
              << ";\n";
            L << "          for (uint kb = 0u; kb < valid; ++kb) {\n";
            const std::string src = operand(ins.a, "kb");
            L << "            acc = "
              << (is_max ? ("max(acc, " + src + ")")
                         : ("(acc + " + src + ")"))
              << ";\n";
            L << "          }\n";
            L << "          red" << dst_r << " = acc; }\n";
        }
        else if (!is_elem[ins.dst])
        {
            L << "        ";
            emit_assign(ins, "r" + dst_r, "0u");
        }
        else
        {
            L << "        for (uint kb = 0u; kb < valid; ++kb) {\n";
            L << "          ";
            emit_assign(ins, "e" + dst_r + "[kb]", "kb");
            L << "        }\n";
        }
    }
    L << "    }\n";

    // ── finalize（每行一次；源仅状态/常量/rparam——validate 保证）────────
    L << "\n    // finalize\n";
    for (const auto& ins : f.finalize)
    {
        L << "    ";
        emit_assign(ins, "r" + std::to_string(ins.dst), "0u");
    }
    L << "    bout[row] = r" << static_cast<int>(f.finalize.back().dst) << ";\n";
    L << "}\n";
    return L.str();
}

// ═══ fold v2 生成器（P-C2 双域：键域[mm+掩码+归约] / vecacc / 向量域）══════
// 线程映射：local_size=256；v2 = 每 WG NR 行（EXPR_FOLD_ROWS_PER_WG），
//   row = gl_WorkGroupID.x*NR + ri（后端 dispatch 同源 ceil(rows/NR)；
//   FoldMeta{out, per_thread_row}，ceil(count/256) 只属 v1）；
//   行循环外置 → Qsh/sreg/spart/smm 跨行复用（shared 零增长，NR 摊薄按
//   WG 计费的固定成本）；越界行 clamp+row_ok 写回守卫（全 WG 均匀无发散）。
//   - 键域元素段（A 段）：j 分片 `j = tid; j < valid; j += 256`（valid ≤
//     EXPR_FOLD_BLOCK）；QKᵀ 在块首 Qsh 预载 + spart/smm 协作归约求得
//     （Matmul 操作数读 smm[j]；原现场串行 Σ_d 已删——吞吐回退主因）；
//   - 元素类寄存器跨段传递 = shared 全量读写 `sreg[NELEM][BLOCK]` + 每段后
//     barrier（免 live 分析；def-before-use 由 validate 静态保证）；
//   - 归约本体：subgroup shuffleDown 蝶式（lane0 落 sred → barrier 广播）。
//     规范保证 subgroup ⊆ workgroup，AMD wave64 经 gl_SubgroupSize 泛化安全
//     （旧"串行扫免 subgroup"顾虑源自跨 WG 混 lane 的担心，对合规驱动不
//     成立；串行扫描实测占 fold ~11%，弃用）；
//   - 行向量态 O：线程按 `dd = tid; dd < vector_out; dd += 256` 步进分片
//     （寄存器跨块持久；输出列数=veclen **运行时读 PC 的 vector_out 槽**——
//     形状无关，同 key 服务任意 d_k）；oi 上界 4（veclen ≤ FOLD_MAX_VEC）。
// PC 形态（fold+matmul = 7 槽 / 纯 fold = 5 槽）：
//   count, cols(=键长), rows, vector_out(=输出列数，运行时), fold_k,
//   [mm_k, mm_batch], vp.., c.., rp..
//   ——创建侧 push range / 写入侧 pc_base / 本声明必须三处同改（4.10 复发记录）。
inline std::string generate_glsl_fold_v2(const std::string& name, const ExprSpec& spec)
{
    if (!spec.fold || spec.matmul || !spec.instrs.empty())
        return {};
    const FoldSpec& f = *spec.fold;
    auto cls = expr_fold_classify(f, spec.num_regs);
    if (!cls)
        return {};
    const std::vector<uint8_t>& is_elem = *cls;
    std::vector<int> eidx(spec.num_regs, -1);
    int ne = 0;
    for (std::uint32_t r = 0; r < spec.num_regs; ++r)
        if (is_elem[r]) eidx[r] = ne++;
    // OUTC_MAX 由单一事实源 FOLD_MAX_VEC 派生（1024/256=4；手写 4 会在
    //   抬高 FOLD_MAX_VEC 时让生成的 o[(dd-tid)/256] 静默越界）
    constexpr std::uint32_t OUTC_MAX = (FOLD_MAX_VEC + 255u) / 256u;
    const std::size_t n_inputs = spec.views.size();
    const bool has_mm = f.matmul.has_value();
    // 输出列数 **运行时读 PC 的 vector_out 槽**（形状无关：同 key 服务任意
    // d_k——veclen 曾进 key 致每个 dk 一个 shader、闭合世界缺登记实证）

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · fold v2 双域），请勿手动编辑 ──\n";
    L << "// 表达式: " << name << "\n";
    L << "#version 450\n";
    // 前提：gl_SubgroupSize 为 2 的幂（桌面 wave32/64 恒真）——非 2 幂时
    //   shuffleDown 蝶形 off>>=1 折叠不全会漏部分和（无运行时守护，记为
    //   书面前提）；`: require` 与 matmul_gemv 的限定符口径统一
    L << "#extension GL_KHR_shader_subgroup_basic : require\n";              // gl_SubgroupSize
    L << "#extension GL_KHR_shader_subgroup_shuffle_relative : require\n\n"; // subgroupShuffleDown
    L << "layout(local_size_x = 256) in;\n\n";
    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { float b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { float bout[]; };\n\n";
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n    uint cols;\n    uint rows;\n"
         "    uint vector_out;\n    uint fold_k;\n";
    if (has_mm)
        L << "    uint mm_k;\n    uint mm_batch;\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    for (std::size_t i = 0; i < spec.rparams.size(); ++i)
        L << "    float rp" << i << ";\n";
    L << "};\n\n";
    if (ne > 0)
    {
        L << "shared float sreg[" << ne << "][" << EXPR_FOLD_BLOCK << "];\n";
        L << "shared float sred[1];\n";   // subgroup 归约 lane0 落点 → barrier 广播
    }
    if (has_mm)
    {
        // QKᵀ 吞吐化：Q 行预载 shared（32+ 个 j 复用 → 消 Q 重读）
        // + 每块 spart/smm 协作归约。尺寸/映射随 EXPR_FOLD_BLOCK 参数化：
        //   jm = tid & (BLOCK-1)、grp = tid >> log2(BLOCK)（组数 = 256/BLOCK）、
        //   d 步进 = 256/BLOCK。
        L << "shared float Qsh[" << FOLD_MAX_MMK << "];\n";
        L << "shared float spart[" << EXPR_FOLD_BLOCK << "]["
          << (256u / EXPR_FOLD_BLOCK) << "];\n";
        L << "shared float smm[" << EXPR_FOLD_BLOCK << "];\n";
    }

    L << "void main()\n{\n";
    L << "    const uint tid = gl_LocalInvocationID.x;\n";
    // 标量寄存器声明（一次；仅行标量类——元素类寄存器在链 j 循环内就地
    //   声明（float r{n} = sreg[..][j]），validate 保证 S/F 上下文不读元素
    //   寄存器 → 外层声明必成死变量；状态初值/o 初始化下沉到行循环内按行重置）
    for (std::uint32_t r = 0; r < spec.num_regs; ++r)
        if (!is_elem[r]) L << "    float r" << r << ";\n";
    // ── NR 行/WG 外层循环（同一行结构原样按 ri 重放；Qsh/sreg/spart/smm
    //    跨行复用 → shared 零增长。ri 循环边界对全 WG 均匀（体内含屏障，
    //    禁分支发散）；越界行 clamp 到末行重复算、写回由 row_ok 挡掉。──
    L << "    for (uint ri = 0u; ri < " << EXPR_FOLD_ROWS_PER_WG << "u; ++ri) {\n";
    L << "      const uint row_r = gl_WorkGroupID.x * " << EXPR_FOLD_ROWS_PER_WG
      << "u + ri;\n";
    L << "      const uint row = min(row_r, rows - 1u);\n";
    L << "      const bool row_ok = row_r < rows;\n";
    if (has_mm)
    {
        L << "      const uint m_per = rows / mm_batch;\n";
        L << "      const uint batch = row / m_per;\n";
        L << "      const uint row_in = row - batch * m_per;\n";
        // Q 行按行预载 shared（整行 dk 个，256 线程步进；A 的 d 维布局按 transA）
        L << "      for (uint d = tid; d < mm_k; d += 256u) {\n";
        if (f.matmul->transA)
            L << "        Qsh[d] = b" << static_cast<int>(f.matmul->a_input)
              << "[(batch * mm_k + d) * m_per + row_in];\n";
        else
            L << "        Qsh[d] = b" << static_cast<int>(f.matmul->a_input)
              << "[(batch * m_per + row_in) * mm_k + d];\n";
        L << "      }\n";
        L << "      barrier();   // Qsh 就绪（每行预载，块循环前一次）\n";
    }
    // 状态初值（每行重置）
    for (std::uint8_t s = 0; s < f.num_state; ++s)
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(f.inits[s]);
        L << "      r" << static_cast<int>(s) << " = uintBitsToFloat(0x"
          << std::hex << bits << std::dec << "u);\n";
    }
    // 行向量态（步进分片寄存器；初值恒 0——VecAccSpec 语义）
    // 数组固定上界 OUTC_MAX=4（veclen ≤ FOLD_MAX_VEC=1024 → oi ≤ 3）
    if (f.vecacc)
    {
        L << "      float o[" << OUTC_MAX << "];\n";
        L << "      for (uint oi = 0u; oi < " << OUTC_MAX << "u; ++oi) o[oi] = 0.0;\n";
    }

    // ── 操作数 → GLSL 片段。ctx：'E'=链(j 上下文) 'R'=归约源(内层 j)
    //    'S'=标量行(无 j) 'F'=finalize(dd 上下文)。──
    const auto vp_of = [&](std::size_t k) -> std::uint32_t
    {
        std::uint32_t vp = 0;
        for (std::size_t j = 0; j < k; ++j)
            if (expr_view_has_runtime_param(
                    static_cast<ExprViewKind>(spec.views[j].kind)))
                vp += expr_view_runtime_param_slots(
                    static_cast<ExprViewKind>(spec.views[j].kind));
        return vp;
    };
    // （QKᵀ 的 mm 物化已上移为块首 smm[] 协作计算——原 mm_prelude 现场
    //   串行 Σ_d 结构删除，见块循环头）
    const auto operand = [&](const ExprOperand& op, char ctx,
                             const std::string& jv) -> std::string
    {
        switch (op.kind)
        {
        default:
        case static_cast<uint8_t>(ExprOperandKind::Reg):
        case static_cast<uint8_t>(ExprOperandKind::Fanout):
            if (is_elem[op.idx] && (ctx == 'R'))
                return "sreg[" + std::to_string(eidx[op.idx]) + "][" + jv + "]";
            return "r" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Const):
            return "c" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::RParam):
            return "rp" + std::to_string(op.idx);
        case static_cast<uint8_t>(ExprOperandKind::Reduce):
            return "r" + std::to_string(op.idx);   // 归约 dst 每线程副本
        case static_cast<uint8_t>(ExprOperandKind::Matmul):
            // QKᵀ 已在块首协作算入 smm[]（见块循环头）——链内直接读，
            //   不再现场串行 Σ_d（原 mm_prelude 结构是吞吐回退主因）
            return "smm[" + jv + "]";
        case static_cast<uint8_t>(ExprOperandKind::Row):
            return "float(row % m_per)";
        case static_cast<uint8_t>(ExprOperandKind::Col):
            return "float(k0 + " + jv + ")";
        case static_cast<uint8_t>(ExprOperandKind::Batch):
            return "float(batch)";
        case static_cast<uint8_t>(ExprOperandKind::VecState):
        {
            // finalize：dd 步进分片 → oi = (dd - tid)/256
            (void)ctx;
            return "o[(dd - tid) / 256u]";
        }
        case static_cast<uint8_t>(ExprOperandKind::Input):
        {
            const std::string gk = "(k0 + " + jv + ")";
            std::ostringstream os;
            glsl_view_read(os, spec.views[op.idx], op.idx,
                           "(row * cols + " + gk + ")", "row", gk, vp_of(op.idx));
            return os.str();
        }
        }
    };
    // 指令 → 赋值语句（dst_str = 目标表达式）
    const auto emit_assign = [&](const ExprInstr& ins, const std::string& dst_str,
                                 char ctx, const std::string& jv) -> std::string
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        const std::string a = operand(ins.a, ctx, jv);
        std::ostringstream s;
        if (op == ExprOp::Select)
        {
            const std::string bt = operand(ins.b, ctx, jv);
            const std::string ce = operand(ins.c, ctx, jv);
            s << dst_str << " = (" << a << " != 0.0) ? " << bt << " : " << ce << ";";
            return "          " + s.str() + "\n";
        }
        if (op == ExprOp::Max || op == ExprOp::Min)
        {
            s << dst_str << " = " << (op == ExprOp::Max ? "max" : "min")
              << "(" << a << ", " << operand(ins.b, ctx, jv) << ");";
            return "          " + s.str() + "\n";
        }
        bool cmp = false;
        const char* bin = glsl_binary_op(op, cmp);
        if (bin && *bin)
        {
            const std::string bv = operand(ins.b, ctx, jv);
            if (cmp)
                s << dst_str << " = (" << a << " " << bin << " " << bv
                  << ") ? 1.0 : 0.0;";
            else
                s << dst_str << " = " << a << " " << bin << " " << bv << ";";
            return "          " + s.str() + "\n";
        }
        if (op == ExprOp::Neg)
        {
            s << dst_str << " = -(" << a << ");";
            return "          " + s.str() + "\n";
        }
        s << dst_str << " = " << glsl_unary_op(op) << "(" << a << ");";
        return "          " + s.str() + "\n";
    };

    // ── 块循环 + body 段扫描发射 ─────────────────────────────────────────
    L << "\n    for (uint k0 = 0u; k0 < fold_k; k0 += " << EXPR_FOLD_BLOCK
      << "u) {\n";
    if (f.causal_skip)
    {
        // causal 整块/边界跳过（qt = row%m_per，与链内 select 谓词同源）：
        //   k0>qt 整块 valid=0 空转（smm/链/归约/vecacc 全由 valid 门控），边界
        //   块钳到 qt+1-k0。跳过项恰为链内 -inf/0 屏蔽值 → 恒等，与 CPU 全量
        //   逐位一致；valid 仅依赖 row/k0 → 全 WG 均匀，体内屏障无发散。
        //   注：causal_skip 只由 make_fold_attn_o（恒带 mm）置位，m_per 必有定义。
        L << "        const uint qt = row % m_per;\n";
        L << "        const uint valid = min(min(" << EXPR_FOLD_BLOCK
          << "u, fold_k - k0), (k0 > qt ? 0u : qt + 1u - k0));\n";
    }
    else
    {
        L << "        const uint valid = min(" << EXPR_FOLD_BLOCK
          << "u, fold_k - k0);\n";
    }
    if (has_mm)
    {
        // QKᵀ 协作化（映射随 BLOCK 参数化，见 shared 声明处注释）
        const MatmulSpec& m = *f.matmul;
        constexpr std::uint32_t LG = []() {
            std::uint32_t v = EXPR_FOLD_BLOCK, g = 0;
            while (v > 1) { v >>= 1; ++g; }
            return g;
        }();
        constexpr std::uint32_t NG = 256u / EXPR_FOLD_BLOCK;  // d 组数
        L << "        const uint jm = tid & " << (EXPR_FOLD_BLOCK - 1) << "u;\n";
        L << "        const uint grp = tid >> " << LG << "u;\n";
        L << "        float part = 0.0;\n";
        L << "        if (jm < valid) {\n";
        L << "          for (uint d = grp; d < mm_k; d += " << NG << "u) {\n";
        if (m.transB)
            L << "            part += Qsh[d] * b" << static_cast<int>(m.b_input)
              << "[(batch * cols + (k0 + jm)) * mm_k + d];\n";
        else
            L << "            part += Qsh[d] * b" << static_cast<int>(m.b_input)
              << "[(batch * mm_k + d) * cols + (k0 + jm)];\n";
        L << "          }\n";
        L << "        }\n";
        L << "        spart[jm][grp] = part;\n";
        L << "        barrier();\n";
        L << "        if (grp == 0u) {\n";
        L << "          float s = 0.0;\n";
        L << "          for (uint g = 0u; g < " << NG << "u; ++g) s += spart[jm][g];\n";
        L << "          smm[jm] = s;\n";
        L << "        }\n";
        L << "        barrier();   // smm 就绪（链/归约的 Matmul 操作数读它）\n";
    }
    std::vector<ExprInstr> chain;
    const auto flush_chain = [&]()
    {
        if (chain.empty())
            return;
        // 元素 j 循环：全量载入元素变量 ← shared、执行链、全量写回 + barrier
        //   （对称全量读写免 live 分析；读到未初始化 shared 仅在 validate
        //    漏网时发生且随即被覆盖——链读未定义由 def-before-use 静态拒）
        L << "        for (uint j = tid; j < valid; j += 256u) {\n";
        for (std::uint32_t r = 0; r < spec.num_regs; ++r)
            if (is_elem[r])
                L << "          float r" << r << " = sreg[" << eidx[r]
                  << "][j];\n";
        for (const auto& ins : chain)
        {
            const std::string dst = "r" + std::to_string(ins.dst);
            L << emit_assign(ins, dst, 'E', "j");
        }
        for (std::uint32_t r = 0; r < spec.num_regs; ++r)
            if (is_elem[r])
            {
                bool written = false;
                for (const auto& ins : chain)
                    if (ins.dst == r) written = true;
                if (written)
                    L << "          sreg[" << eidx[r] << "][j] = r" << r << ";\n";
            }
        L << "        }\n";
        L << "        barrier();\n";
        chain.clear();
    };
    for (const auto& ins : f.body)
    {
        const ExprOp op = static_cast<ExprOp>(ins.op);
        if (expr_op_is_reduce(op))
        {
            flush_chain();
            const bool is_max = (op == ExprOp::RowMax || op == ExprOp::ColMax);
            const auto skind = static_cast<ExprOperandKind>(ins.a.kind);
            const bool elem_src = (skind == ExprOperandKind::Reg ||
                                   skind == ExprOperandKind::Fanout)
                               && is_elem[ins.a.idx];
            if (elem_src)
            {
                // 归约本体（subgroup shuffle 蝶式，fold v2 吞吐路径）：
                //   旧实现 256 线程各串行扫 valid（64 级 shared 依赖链 ×256 冗余，
                //   窗口探针量化占 fold ~11%）。新结构：
                //   ① tid < BLOCK 的线程按 gl_SubgroupSize 步进各持若干 sreg
                //      （flush 屏障已保证跨线程可见；subgroup0 恰好完整覆盖
                //      [0, valid) 且不重不漏，任意 subgroup 尺寸成立）；
                //   ② subgroup 内 shuffleDown 蝶式 log2 归约；lane0 落 sred[0]；
                //   ③ barrier 广播——r{dst} 是每线程副本，标量段全 WG 读。
                //   跨设备：规范保证 subgroup ⊆ workgroup（AMD wave64 经
                //   gl_SubgroupSize 泛化，16/32/64/128 皆正确）——旧头注释
                //   "免 subgroup 防混邻 WG lane" 对合规驱动不成立，故此改。
                //   数值：浮点求和结合序异于 CPU 串行 → run_fold_attn_gpu
                //   容差 1e-4 兜；max 结合序无关，位级一致。
                L << "        {\n";
                // max 恒等元 = lowest()（0xFF7FFFFF）而非 -inf：与 CPU 基准
                //   eval_fold_impl(numeric_limits::lowest()) 对齐——块内全 -inf
                //   （doc 掩码整块屏蔽）时 m_old=blk_m=-inf 会让 dm=−inf−−inf
                //   =NaN 污染 l/O（2026-09 训练 -nan 根因），见 fold v1 同款注释。
                const std::string init =
                    is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
                const std::string eis = std::to_string(eidx[ins.a.idx]);
                L << "          float v = " << init << ";\n";
                L << "          if (tid < " << EXPR_FOLD_BLOCK << "u) {\n";
                L << "            for (uint e = tid; e < valid; e += gl_SubgroupSize)\n";
                L << "              v = "
                  << (is_max ? ("max(v, sreg[" + eis + "][e])")
                             : ("(v + sreg[" + eis + "][e])"))
                  << ";\n";
                L << "          }\n";
                L << "          for (uint off = gl_SubgroupSize >> 1; off > 0u; off >>= 1)\n";
                L << "            v = "
                  << (is_max ? "max(v, subgroupShuffleDown(v, off))"
                             : "(v + subgroupShuffleDown(v, off))")
                  << ";\n";
                L << "          if (tid == 0u) sred[0] = v;\n";
                L << "          barrier();   // sred 广播（跨 subgroup → 全 WG）\n";
                L << "          r" << std::to_string(ins.dst) << " = sred[0];\n";
                L << "        }\n";
            }
            else
            {
                // 标量源归约（罕见路径）：保持原串行扫描语义
                L << "        {\n";
                // max init = lowest()（0xFF7FFFFF），对齐 CPU 基准/防 −inf−−inf=NaN
                //   ——与上面元素源路径同款理由，两处必须同改
                L << "          float acc = "
                  << (is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0") << ";\n";
                L << "          for (uint j = 0u; j < valid; ++j) {\n";
                const std::string src = operand(ins.a, 'R', "j");
                L << "            acc = "
                  << (is_max ? ("max(acc, " + src + ")") : ("(acc + " + src + ")"))
                  << ";\n";
                L << "          }\n";
                L << "          r" << std::to_string(ins.dst) << " = acc;\n";
                L << "        }\n";
            }
        }
        else if (is_elem[ins.dst])
        {
            chain.push_back(ins);   // 元素链（遇标量/归约/尾冲）
        }
        else
        {
            flush_chain();
            // 标量行（每线程副本；源全行标量——validate 保证不读元素）
            L << emit_assign(ins, "r" + std::to_string(ins.dst), 'S', "j");
        }
    }
    flush_chain();

    // ── vecacc：O *= scale；O += Σ_j w(j)·b(j, d)（d 步进分片）──
    if (f.vecacc)
    {
        const VecAccSpec& va = *f.vecacc;
        if (va.has_scale)
        {
            L << "        for (uint dd = tid; dd < vector_out; dd += 256u)"
              << " o[(dd - tid) / 256u] *= r"
              << static_cast<int>(va.scale_reg) << ";\n";
        }
        const std::string b_row = has_mm ? "(row / m_per)" : "0u";
        // vecacc：O += Σ_j w(j)·b(j, d) —— 4 路软件流水。
        //   旧版单发串行读 V（GPU 无硬件预取 → 64 级 L2 延迟链完全暴露，
        //   探针实测占 fold 55%：5694→2567µs）。每拍同批发出 j..j+3 四个
        //   读 → 延迟重叠；四条独立累加语句保持 j 升序求和 → 与旧版逐位一致。
        //   runtime valid 由尾循环兜余数；所有权/屏障/shared 布局不动。
        L << "        const uint jr0 = " << b_row << " * fold_k + k0;\n";
        L << "        for (uint dd = tid; dd < vector_out; dd += 256u) {\n";
        L << "          uint j = 0u;\n";
        L << "          for (; j + 4u <= valid; j += 4u) {\n";
        for (int t = 0; t < 4; ++t)
            L << "            const float v" << t << " = b"
              << static_cast<int>(va.b_input) << "[(jr0 + j + " << t
              << "u) * vector_out + dd];\n";
        for (int t = 0; t < 4; ++t)
            L << "            o[(dd - tid) / 256u] += sreg["
              << eidx[va.weight_reg] << "][j + " << t << "u] * v" << t << ";\n";
        L << "          }\n";
        L << "          for (; j < valid; ++j)\n";
        L << "            o[(dd - tid) / 256u] += sreg["
          << eidx[va.weight_reg] << "][j] * b" << static_cast<int>(va.b_input)
          << "[(jr0 + j) * vector_out + dd];\n";
        L << "        }\n";
    }
    L << "    }\n";   // 块循环尾

    // ── finalize（向量域）：dd 步进分片逐列执行 → bout[row*vector_out + dd] ──
    L << "\n    for (uint dd = tid; dd < vector_out; dd += 256u) {\n";
    for (const auto& ins : f.finalize)
    {
        L << emit_assign(ins, "r" + std::to_string(ins.dst), 'F', "dd");
    }
    L << "      if (row_ok) bout[row * vector_out + dd] = r"
      << static_cast<int>(f.finalize.back().dst) << ";\n";   // 越界 ri 不写回
    L << "    }\n";
    L << "    }\n";   // NR 行循环尾
    L << "}\n";
    return L.str();
}

// ── 带类型变体（Phase 2 in-kernel f16）───────────────────────────────────
// sig 标注每个输入/输出的存储精度（见 expr_spec.hpp 的 ExprPrecSig）。
//   sig == 0 → 全 f32：**GLSL 与迁移前逐字节相同**（零回归）。
//   sig != 0 → 该精度上有 f16 的元素：缓冲区声明为 float16_t（GL_EXT_shader_16bit_storage
//              + 显式 float(...)/float16_t(...) 转换），算术仍在 f32（§7.2：f32 参考 +
//              输出舍入）→ 读一次 2B、写一次 2B，不再需要"边界 cast"为每个算子
//              物化 f32 副本（那正是 transient 膨胀 2.4× 的根因）。
// 目前只支持**纯逐元素**形态；reduce / matmul / fold 拿到 sig != 0 时返回空串
// （生成器不支持 → 上层跳过该变体，运行时回退边界 cast，正确性不受影响）。
inline std::string generate_glsl(const std::string& name, const ExprSpec& spec,
                                 ExprPrecSig sig = 0)
{
    if (sig != 0)
    {
        // 已支持的带类型形态：纯逐元素 + matmul 段（本函数尾部的分派）。
        // fold 段 / 含归约指令的形态尚未支持 → 返回空串（上层跳过该变体，
        // 运行时回退边界 cast，正确性不受影响）。
        if (spec.fold || expr_spec_reduce_axis(spec) != -1)
            return {};
    }
    // fold 段（P-C1/P-C2）：双域形态（向量态/mm 段）走 v2，纯标量走 v1
    if (spec.fold)
        return (spec.fold->vec_state_len > 0 || spec.fold->matmul)
            ? generate_glsl_fold_v2(name, spec)
            : generate_glsl_fold(name, spec);
    // 含前置 matmul 段（S3）：matmul + 尾逐元素链融合 shader
    if (spec.matmul)
        return generate_glsl_matmul(name, spec, sig);

    const std::size_t n_inputs = spec.views.size();
    std::ostringstream L;

    // 逐输入/输出的存储类型
    const auto in_type = [&](std::size_t i) -> const char*
    {
        return (sig != 0 && expr_prec_sig_in_f16(sig, i)) ? "float16_t" : "float";
    };
    const bool out_f16 = sig != 0 && expr_prec_sig_out_f16(sig);
    const bool any_f16 = sig != 0;

    L << "// ── 自动生成（AOT 算子融合），请勿手动编辑 ──\n";
    L << "// 表达式: " << name;
    if (sig != 0)
        L << "   [精度 " << expr_prec_sig_str(sig, n_inputs) << "]";
    L << "\n";
    L << "#version 450\n\n";
    if (any_f16)
        L << "#extension GL_EXT_shader_16bit_storage : require\n\n";
    L << "layout(local_size_x = 256) in;\n\n";

    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { " << in_type(i) << " b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { " << (out_f16 ? "float16_t" : "float")
      << " bout[]; };\n\n";

    // push constants：count + cols（视图行/列）+ 运行时视图参数 vp + 常量
    L << "layout(push_constant) uniform PC {\n";
    L << "    uint count;\n";
    L << "    uint cols;\n";
    const std::uint32_t n_vp = expr_spec_runtime_view_param_count(spec);
    for (std::uint32_t i = 0; i < n_vp; ++i)
        L << "    uint vp" << i << ";\n";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
        L << "    float c" << i << ";\n";
    const std::uint32_t n_rp = static_cast<std::uint32_t>(spec.rparams.size());
    for (std::uint32_t i = 0; i < n_rp; ++i)
        L << "    float rp" << i << ";\n";
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
        case static_cast<uint8_t>(ExprOperandKind::RParam):
            return "rp" + std::to_string(op.idx);
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
                        static_cast<ExprViewKind>(spec.views[j].kind)))
                    vp += expr_view_runtime_param_slots(
                        static_cast<ExprViewKind>(spec.views[j].kind));
            o << indent << "const float v" << i << " = ";
            const bool f16_in = (sig != 0 && expr_prec_sig_in_f16(sig, i));
            if (f16_in) o << "float(";   // f16 读 → 显式转 f32（算术仍在 f32）
            glsl_view_read(o, spec.views[i], static_cast<std::uint32_t>(i),
                           idx_var, row_var, col_var, vp, sig);
            if (f16_in) o << ")";
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
        L << "    bout[i] = "
          << (out_f16 ? "float16_t(r" : "r") << last_dst
          << (out_f16 ? ")" : "") << ";\n";
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
        const std::string cvt = (sig != 0 && expr_prec_sig_in_f16(sig, i)) ? "float" : "";
        const auto ld = [&](const std::string& idx) {
            return cvt.empty() ? ("b" + std::to_string(i) + "[" + idx + "]")
                               : ("float(b" + std::to_string(i) + "[" + idx + "])");
        };
        if (k == ExprViewKind::Linear)
            L << "        const vec4 v" << i << " = vec4(" << ld("base") << ", "
              << ld("base+1u") << ", " << ld("base+2u") << ", " << ld("base+3u")
              << ");\n";
        else if (k == ExprViewKind::RowBroadcast)
            L << "        const vec4 v" << i << " = vec4(" << ld("row") << ");\n";
        else // ColBroadcast
            L << "        const vec4 v" << i << " = vec4(" << ld("col0") << ", "
              << ld("col0+1u") << ", " << ld("col0+2u") << ", " << ld("col0+3u")
              << ");\n";
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
        case static_cast<uint8_t>(ExprOperandKind::RParam):
            return "vec4(rp" + std::to_string(op.idx) + ")";
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
    const auto st = [&](const std::string& comp) {
        return out_f16 ? ("float16_t(r" + std::to_string(last_dst) + "." + comp + ")")
                       : ("r" + std::to_string(last_dst) + "." + comp);
    };
    L << "        bout[base] = " << st("x") << ";\n";
    L << "        bout[base+1u] = " << st("y") << ";\n";
    L << "        bout[base+2u] = " << st("z") << ";\n";
    L << "        bout[base+3u] = " << st("w") << ";\n";
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
    L << "        bout[e] = "
      << (out_f16 ? "float16_t(r" : "r") << last_dst
      << (out_f16 ? ")" : "") << ";\n";
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
    const std::string& name, const ExprSpec& spec, ExprPrecSig sig = 0)
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

    // ── 带类型变体（Phase 2 in-kernel f16）───────────────────────────────
    // sig==0 → 全 f32，GLSL 与迁移前逐字节相同（零回归）。sig!=0 → 对应槽的
    // 缓冲区声明为 float16_t，读取处统一 float(...) 转 f32（累加/合并/尾链
    // 全在 f32，符合 §7.2），输出写回按 out 位 float16_t(...)。
    const bool sign_f16 = (sig != 0);
    const bool out_f16 = sign_f16 && expr_prec_sig_out_f16(sig);
    const auto rd = [&](std::size_t k, const std::string& idx) -> std::string {
        const std::string raw = "b" + std::to_string(k) + "[" + idx + "]";
        return (sign_f16 && expr_prec_sig_in_f16(sig, k)) ? ("float(" + raw + ")") : raw;
    };
    const auto wr = [&](const std::string& v) -> std::string {
        return out_f16 ? ("float16_t(" + v + ")") : v;
    };

    std::ostringstream L;
    L << "// ── 自动生成（AOT 算子融合 · 归约 kernel），请勿手动编辑 ──\n";
    L << "// 表达式: " << name;
    if (sign_f16)
        L << "   [精度 " << expr_prec_sig_str(sig, n_inputs) << "]";
    L << "\n";
    L << "#version 450\n";
    L << "#extension GL_KHR_shader_subgroup_basic : enable\n";
    L << "#extension GL_KHR_shader_subgroup_arithmetic : enable\n";
    if (sign_f16)
        L << "#extension GL_EXT_shader_16bit_storage : require\n";
    L << "\n";
    L << "layout(local_size_x = 256) in;\n\n";
    for (std::size_t i = 0; i < n_inputs; ++i)
        L << "layout(std430, binding = " << i << ") readonly buffer Buf" << i
          << " { " << ((sign_f16 && expr_prec_sig_in_f16(sig, i)) ? "float16_t" : "float")
          << " b" << i << "[]; };\n";
    L << "layout(std430, binding = " << n_inputs
      << ") writeonly buffer BufOut { " << (out_f16 ? "float16_t" : "float")
      << " bout[]; };\n\n";
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
    const std::uint32_t n_rp = static_cast<std::uint32_t>(spec.rparams.size());
    for (std::uint32_t i = 0; i < n_rp; ++i)
        L << "    float rp" << i << ";\n";
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
        L << "            mm += " << rd(a_slot, a_idx) << " * " << rd(b_slot, b_idx) << ";\n";
    };

    L << "void main()\n{\n";
    L << "    const uint tid = gl_LocalInvocationID.x;\n";
    L << "    const uint idx = gl_WorkGroupID.x;\n";
    // 行归约：工作组=单行，idx<rows 守卫。列归约：工作组=32 列 tile，
    // lane=列内偏移、warp=行块（col = idx*32 + l；越界列不早退、读写守卫）。
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
        case static_cast<uint8_t>(ExprOperandKind::RParam):
            return "rp" + std::to_string(op.idx);
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
                return rd(op.idx, "row");
            if (vk == ExprViewKind::ColBroadcast)
                return rd(op.idx, "col");
            // Linear / RotateHalf / RowMod / RowGather / BatchMod：索引映射内联；
            // vp 槽 = 此前运行时参数视图个数
            std::ostringstream os;
            std::uint32_t vp = 0;
            for (std::size_t j = 0; j < op.idx; ++j)
                if (expr_view_has_runtime_param(
                        static_cast<ExprViewKind>(spec.views[j].kind)))
                    vp += expr_view_runtime_param_slots(
                        static_cast<ExprViewKind>(spec.views[j].kind));
            glsl_view_read(os, v, static_cast<std::uint32_t>(op.idx),
                           "(row*cols + col)", "row", "col", vp, sig);
            const std::string raw = os.str();
            // f16 槽：视图读取返回 float16_t 左值 → 显式转 f32（算术仍在 f32）
            return (sign_f16 && expr_prec_sig_in_f16(sig, op.idx))
                ? ("float(" + raw + ")") : raw;
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
        // max init = lowest()（0xFF7FFFFF）对齐 CPU 基准，理由同归约视图处注释
        const std::string idt = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
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
            // max init = lowest()（0xFF7FFFFF）对齐 CPU 基准（全库归约一律
            //   numeric_limits::lowest()）：-inf 作恒等元在全 -inf 输入下与
            //   状态进位/减法组合会出 −inf−−inf=NaN（2026-09 fold doc 掩码
            //   GPU -nan 同类根因），全库统一 lowest 根除该风险类
            const std::string init = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
            L << "\n    // 归约视图 " << k << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        uint i = tid;\n";
            L << "        for (; i + 3u*256u < cols; i += 4u*256u) {\n";
            L << "            acc0 = " << combine(is_max, "acc0", rd(k, "idx*cols + i")) << ";\n";
            L << "            acc1 = " << combine(is_max, "acc1", rd(k, "idx*cols + i+256u")) << ";\n";
            L << "            acc2 = " << combine(is_max, "acc2", rd(k, "idx*cols + i+512u")) << ";\n";
            L << "            acc3 = " << combine(is_max, "acc3", rd(k, "idx*cols + i+768u")) << ";\n";
            L << "        }\n";
            L << "        for (; i < cols; i += 256u)\n";
            L << "            acc0 = " << combine(is_max, "acc0", rd(k, "idx*cols + i")) << ";\n";
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
            // max init = lowest()（0xFF7FFFFF）对齐 CPU 基准（全库归约一律
            //   numeric_limits::lowest()）：-inf 作恒等元在全 -inf 输入下与
            //   状态进位/减法组合会出 −inf−−inf=NaN（2026-09 fold doc 掩码
            //   GPU -nan 同类根因），全库统一 lowest 根除该风险类
            const std::string init = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
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
            L << "            bout[idx] = " << wr("s_red[" + std::to_string(slot_of_instr[spec.instrs.back().dst]) + "][0]") << ";\n";
        else
        {
            L << "            const uint row = idx;\n            const uint col = 0u;\n";
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "            bout[idx] = " << wr("r" + std::to_string(static_cast<int>(spec.instrs.back().dst))) << ";\n";
        }
        L << "        }\n";
        L << "        return;\n";
        L << "    }\n";
        L << loop_decl;
        if (last_is_reduce)
            L << "        bout[" << out_idx << "] = " << wr("s_red[" + std::to_string(slot_of_instr[spec.instrs.back().dst]) + "][0]") << ";\n";
        else
        {
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "        bout[" << out_idx << "] = " << wr("r" + std::to_string(static_cast<int>(spec.instrs.back().dst))) << ";\n";
        }
        L << "    }\n";
    }
    else
    {
        // ═══ 列归约（2026-09-24 重构，与 OP 级 reduce.comp 列模式同构）═══
        // 旧结构「每 WG 256 列、每线程一整列」：读取按行合并，但 WG 数 =
        // ceil(cols/256)——CE 场景 cols = total 常为数百 → 仅个位数 WG，
        // 行循环串行度极高；且与 OP 级（WG=单列、lane 跨行步进、完全不
        // 合并）算法不一致。两路现统一为 OP 级重写版结构：
        //   lane = 列内偏移 l（同 32 连续列 → 同行读 128B 合并事务）
        //   warp = 行块 wb（8 个，row = wb 步进 8）→ 每 WG 覆盖 32 列
        //   部分和 s_red[slot][tid]（tid = wb*32+l）→ 跨 8 行块合并就地覆写
        //   dispatch：ceil(cols/32)（backend raxis==1 同源改）
        //   4 路累加器跨步 32（8 行 × 4），ILP 形态与行 pass 一致
        // 越界列**不早退**（barrier 需全 WG 参与）：读跳过、部分和置恒等元、
        // 输出写守卫。
        // emit_col_merge：每个归约 pass 后跨 8 行块合并本槽（与行 pass 的
        // emit_tree_reduce 调用位置契约一致——后续 pass/输出读到的都是
        // 整列值；s_red[w*32+l] 跨 lane 连续读、[tid] 连续写，零 bank 冲突）
        const auto emit_col_merge = [&](int slot, bool is_max)
        {
            const std::string s = "s_red[" + std::to_string(slot) + "]";
            const std::string init = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
            const std::string m = "m" + std::to_string(slot);
            L << "    barrier();\n";   // ① 各行块部分和写完
            L << "    { float " << m << " = " << init << ";\n";
            L << "        for (uint w = 0u; w < 8u; ++w)\n";
            L << "            " << m << " = "
              << combine(is_max, m, s + "[w*32u + l]") << ";\n";
            // ② 读/写分离：所有 lane 读完 8 项后才允许覆写自身槽位
            //    （写 s_red[tid=(wb*32+l)] 与他 lane 读 [w*32+l] 同址，
            //     无此 barrier 会读到已覆写值 → sum 双计）
            L << "        barrier();\n";
            L << "        " << s << "[tid] = " << m << "; }\n";
            L << "    barrier();\n";   // ③ 覆写完成，后续 pass/输出才可读
        };
        L << "    const uint l  = tid & 31u;\n";
        L << "    const uint wb = tid >> 5u;\n";
        L << "    const uint col = idx * 32u + l;\n";

        // ── 归约视图 pass（tile：每线程一整列，多累加器跨行顺序读，合并访问）──
        for (std::size_t k = 0; k < n_inputs; ++k)
        {
            const ExprView& v = spec.views[k];
            if (!expr_view_is_reduce(static_cast<ExprViewKind>(v.kind)))
                continue;
            const int slot = slot_of_view[k];
            const bool is_max = slot_is_max[static_cast<std::size_t>(slot)];
            // max init = lowest()（0xFF7FFFFF）对齐 CPU 基准（全库归约一律
            //   numeric_limits::lowest()）：-inf 作恒等元在全 -inf 输入下与
            //   状态进位/减法组合会出 −inf−−inf=NaN（2026-09 fold doc 掩码
            //   GPU -nan 同类根因），全库统一 lowest 根除该风险类
            const std::string init = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
            L << "\n    // 归约视图 " << k << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        if (col < cols) {\n";
            L << "            uint i = wb;\n";
            L << "            for (; i + 24u < rows; i += 32u) {\n";
            L << "                acc0 = " << combine(is_max, "acc0", rd(k, "i*cols + col")) << ";\n";
            L << "                acc1 = " << combine(is_max, "acc1", rd(k, "(i+8u)*cols + col")) << ";\n";
            L << "                acc2 = " << combine(is_max, "acc2", rd(k, "(i+16u)*cols + col")) << ";\n";
            L << "                acc3 = " << combine(is_max, "acc3", rd(k, "(i+24u)*cols + col")) << ";\n";
            L << "            }\n";
            L << "            for (; i < rows; i += 8u)\n";
            L << "                acc0 = " << combine(is_max, "acc0", rd(k, "i*cols + col")) << ";\n";
            L << "        }\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
            emit_col_merge(slot, is_max);
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
            // max init = lowest()（0xFF7FFFFF）对齐 CPU 基准（全库归约一律
            //   numeric_limits::lowest()）：-inf 作恒等元在全 -inf 输入下与
            //   状态进位/减法组合会出 −inf−−inf=NaN（2026-09 fold doc 掩码
            //   GPU -nan 同类根因），全库统一 lowest 根除该风险类
            const std::string init = is_max ? "uintBitsToFloat(0xFF7FFFFFu)" : "0.0";
            L << "\n    // 归约指令 " << ri << " (槽 " << slot << ")\n";
            L << "    {\n";
            L << "        float acc0 = " << init << ", acc1 = " << init
              << ", acc2 = " << init << ", acc3 = " << init << ";\n";
            L << "        if (col < cols) {\n";
            L << "            uint i = wb;\n";
            L << "            for (; i + 24u < rows; i += 32u) {\n";
            for (int k = 0; k < 4; ++k)
            {
                L << "                { const uint row = i + " << std::to_string(8 * k)
                  << "u; ";
                emit_reg_decl();
                emit_instrs(0, ri);
                L << " acc" << k << " = " << combine(is_max, "acc" + std::to_string(k), src)
                  << "; }\n";
            }
            L << "            }\n";
            L << "            for (; i < rows; i += 8u) {\n";
            L << "                { const uint row = i; ";
            emit_reg_decl();
            emit_instrs(0, ri);
            L << " acc0 = " << combine(is_max, "acc0", src) << "; }\n";
            L << "            }\n";
            L << "        }\n";
            L << "        s_red[" << slot << "][tid] = "
              << combine4(is_max, "acc0", "acc1", "acc2", "acc3") << ";\n";
            L << "    }\n";
            emit_col_merge(slot, is_max);
        }

        // ── 输出 pass（列：lane=列 / warp=行块，只写自己行块的行）──
        L << "\n    // 输出\n";
        L << "    if (vector_out == 1u) {\n";
        if (last_is_reduce)
            L << "        if (col < cols) bout[col] = " << wr("s_red[" + std::to_string(slot_of_instr[spec.instrs.back().dst]) + "][tid]") << ";\n";
        else
        {
            L << "        if (col < cols) {\n";
            L << "            const uint row = 0u;\n";
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "            bout[col] = " << wr("r" + std::to_string(static_cast<int>(spec.instrs.back().dst))) << ";\n";
            L << "        }\n";
        }
        L << "        return;\n";
        L << "    }\n";
        L << "    if (col < cols) for (uint row = wb; row < rows; row += 8u) {\n";
        if (last_is_reduce)
            L << "        bout[row*cols + col] = " << wr("s_red[" + std::to_string(slot_of_instr[spec.instrs.back().dst]) + "][tid]") << ";\n";
        else
        {
            emit_reg_decl();
            emit_instrs(0, spec.instrs.size());
            L << "        bout[row*cols + col] = " << wr("r" + std::to_string(static_cast<int>(spec.instrs.back().dst))) << ";\n";
        }
        L << "    }\n";
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
        const std::string& name_, const ExprSpec& spec,
        ExprPrecSig sig = 0) override
    { return nn::generate_glsl(name_, spec, sig); }

    [[nodiscard]] std::string generate_reduce(
        const std::string& name_, const ExprSpec& spec,
        ExprPrecSig sig = 0) override
    { return nn::generate_glsl_reduce(name_, spec, sig); }
};

// 登记到 emitter 注册表（静态初始化；重复包含无害——同名拒绝覆盖）
inline const bool kGlslEmitterRegistered =
    emitter_registry::register_backend("glsl",
        []() -> std::unique_ptr<ExprEmitter> {
            return std::make_unique<GlslEmitter>();
        });

} // namespace nn

