#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_dsl.hpp — 统一表达式 DSL（编译期模板，唯一前端）
//
//  目标：所有逐元素计算都用"普通数学写法"写出，编译期融合：
//      auto e = leaf(q) * row_mod(cos, dk) + rotate_half(leaf(q), dk) * row_mod(sin, dk);
//      engine.compute(e, rows, cols);
//
//  单一事实来源 = 这一整段表达式本身（编译期类型）。两个消费方共用：
//    1. CPU：直接把表达式当作编译期 AST，逐元素求值 → 编译器内联 + SIMD 融合，
//       等价手写 for 循环（零开销抽象）。
//    2. GPU（AOT，无运行时生成）：to_expr_spec(expr) 在编译期把同一表达式折叠
//       成扁平 ExprSpec → 与预生成融合 shader 比对 → dispatch。
//
//  闭合世界（closed-world）约定：代码里用到的表达式集合 = AOT 生成的 shader 集合。
//  运行时若遇到未预生成 shader 的表达式，GPU 直接**硬报错**，绝不静默回退
//  （即彻底删除 eager 路径，把"难以察觉的漂移"变成"立即暴露的错误"）。
//
//  设计要点：
//    - 算子复用 nn::ops（唯一的算子来源，op_id() 统一映射到 ExprOp）。
//    - 叶子持有 Tensor（零拷贝 shared_ptr）；视图（RotateHalf/RowMod）是索引
//      映射叶子，不物化中间张量。
//    - 运算符重载/概念约束：普通 + - * / 与比较、select 均照常书写。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>

#include "compute_tensor.hpp"
#include "compute_engine.hpp"
#include "core_assert.hpp"
#include "expr_spec.hpp"
#include "expr_registry.hpp"
#include "algebra_expr.hpp"   // nn::Expression / nn::BoolExpression 概念
#include "algebra_ops.hpp"    // nn::ops（唯一算子来源，含 op_id()）
#include "algebra_matrix.hpp" // Matrix

namespace nn::dsl
{

// ── DSL 逐元素模板路径的并行门控 ────────────────────────────────────────
// 低于全局 nn::PARALLEL_THRESHOLD(524288)。全局值按"裸逐元素 add 的内存
// 带宽回本点"标定；而本路径折叠的是多算子融合表达式（每元素 2~6 次算术，
// 还可能含 exp/tanh/sqrt），per-element 成本更高，回本点更低。
// LayerNorm/RMSNorm 的典型形状 768×512 = 393216 恰在旧门控之外 → 整段串行。
// 131072 元素 × ~2ns ≈ 0.26ms，远超并行区启动开销（实测 ~100µs）。
inline constexpr std::size_t kDslParallelThreshold = 131072;

// ══════════════════════════════════════════════════════════════════════════
// 内部求值/折叠构建器：把编译期表达式折叠成扁平 ExprSpec + 输入张量列表
// ══════════════════════════════════════════════════════════════════════════
struct SpecBuilder
{
    ExprSpec spec;
    std::vector<Tensor> inputs;

    ExprOperand add_const(Scalar v)
    {
        spec.consts.push_back(v);
        return expr::cst(static_cast<std::uint8_t>(spec.consts.size() - 1));
    }
    // 运行时标量参数（RParam）：值作为运行时数据存入 spec.rparams，不进
    // expr_spec_key；同结构不同值共享一个融合 shader（glsl_gen 把 rparams
    // 作为 push constant 读取，dispatch 时按实际 spec 填充）。用于优化器的
    // lr/eps/β、偏差修正系数等每步会变、但结构固定的标量。
    ExprOperand add_rparam(Scalar v)
    {
        spec.rparams.push_back(v);
        return expr::rval(static_cast<std::uint8_t>(spec.rparams.size() - 1));
    }
    ExprOperand add_input_linear(const Tensor& t)
    {
        spec.views.push_back(expr::linear());
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_rotate(const Tensor& t, std::uint32_t block)
    {
        spec.views.push_back(expr::rotate_half(block, /*negate_first_half=*/true));
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_rowmod(const Tensor& t, std::uint32_t mod)
    {
        spec.views.push_back(expr::row_mod(mod));
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_rowaccess(const Tensor& t, std::uint32_t offset, std::uint32_t mod)
    {
        spec.views.push_back(expr::row_access(offset, mod));
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    // 归约视图输入：该输入被归约为每行/每列一个标量（(rows,1)/(1,cols)），
    // 求值期自动广播——参与算术时按当前 (r,c) 读取对应标量。
    ExprOperand add_reduce_input(const Tensor& t, ExprViewKind kind)
    {
        ExprView v;
        v.kind = static_cast<std::uint8_t>(kind);
        spec.views.push_back(v);
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    // 归约指令：对某表达式结果（操作数 a）归约 → 隐式"每行/每列一个标量"
    // 的归约向量，后续指令经 Reduce 操作数按行/列广播访问。
    ExprOperand add_reduce_instr(ExprOp op, ExprOperand a)
    {
        ExprInstr in;
        in.op = static_cast<std::uint8_t>(op);
        in.dst = static_cast<std::uint8_t>(spec.num_regs++);
        in.a = a;
        spec.instrs.push_back(in);
        return expr::reduce(in.dst);
    }
    // 广播视图：输入本身已是 (rows,1)/(1,cols) 小向量，按行/列广播
    ExprOperand add_input_rowbroadcast(const Tensor& t)
    {
        spec.views.push_back(expr::row_broadcast());
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_input_colbroadcast(const Tensor& t)
    {
        spec.views.push_back(expr::col_broadcast());
        inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(inputs.size() - 1));
    }
    ExprOperand add_instr(ExprOp op, ExprOperand a, ExprOperand b = {}, ExprOperand c = {})
    {
        ExprInstr in;
        in.op = static_cast<std::uint8_t>(op);
        in.dst = static_cast<std::uint8_t>(spec.num_regs++);
        in.a = a; in.b = b; in.c = c;
        spec.instrs.push_back(in);
        return expr::reg(in.dst);
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 编译期求值的"输出列数感知"分发：eval_with_cols
//
// nn::Expression 的契约是 eval(i)：只有一个扁平下标，**不知道输出列数**。
// 大部分叶子是列数无关的（Linear/RotateHalf/RowMod/RowAccess 都能从自身
// 张量推出列数），但有一类视图必须知道"输出列数"才能把扁平下标还原成
// (row, col)：RowBroadcast（输入 (rows,1)，读第 i/cols 个标量）。这类节点
// 额外提供 eval(i, cols)；复合节点把 cols 透传给子节点：
//   eval_with_cols(child, i, cols)：子节点有 2 参 eval 就用它，否则退回 eval(i)。
//
// 只有真正需要"全行/全列"信息（归约视图/归约指令/matmul/网格索引）的表达式
// 才走引擎解释器（由 has_reduction_v 分流）；纯索引映射（含广播视图）留在
// 模板求值路径 → 编译器内联 + 与其它逐元素路径共用同一并行门控。
// ══════════════════════════════════════════════════════════════════════════
template <typename T>
concept ColsAwareEval = requires(const T& t, std::size_t i, std::size_t cols)
{
    { t.eval(i, cols) } -> std::convertible_to<Scalar>;
};

template <typename E>
[[nodiscard]] Scalar eval_with_cols(const E& e, std::size_t i, std::size_t cols)
{
    if constexpr (ColsAwareEval<E>)
        return static_cast<Scalar>(e.eval(i, cols));
    else
        return static_cast<Scalar>(e.eval(i));
}

// ══════════════════════════════════════════════════════════════════════════
// eval_into_span — 把逐元素表达式写进既有 span（eval_cpu / compute_into 共用）
//
// 与逐元素原语（algebra_matrix 的 apply/transform/原地运算）**同构**：
//   - n < kDslParallelThreshold：串行计数循环 + NN_VECTORIZE_PRAGMA
//   - n ≥ kDslParallelThreshold：自建分块并行，块内同样是**带向量化提示的裸
//     指针计数循环**（不用 nn::for_each(iota)：其分块内层在池里，无法为该
//     循环单独加向量化提示——池被多种读写模式复用，不能全局 assume_safety；
//     实测大张量下慢 1.3–1.5 倍）
// NN_VECTORIZE_PRAGMA（clang `loop vectorize(assume_safety)` / GCC ivdep）是
// 关键：内层循环读写的是多个"来源未知"的指针（目标 + 各叶子），编译器无法
// 证明互不别名；不给提示时不向量化，实测逐元素 add 慢 3 倍以上。
// 语义安全性：本循环是纯 map —— 每个输出元素只依赖**同一下标**的输入，
// 先读完再写回。无论目标是否与某输入同 buffer，向量化后的"载入同下标向量
// →计算→存回"与串行逐一读改写逐字节一致（无跨下标依赖）。
//
// 门控说明：这里用 kDslParallelThreshold（131072）而非全局 PARALLEL_THRESHOLD
// （524288）。全局值是按"裸 add 的内存带宽回本点"标的；而本路径折叠的是
// 多算子融合表达式（每元素 2~6 次算术 + 可能的 exp/tanh），per-element 成本
// 更高，回本点更低。典型 LayerNorm/RMSNorm 形状 768×512=393216 旧门控下
// **整段串行**（正好低于 524288），是本项目 CPU 归一化层只用 ~1 核的直接原因。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
inline void eval_into_span(const E& e, Span d_span, std::size_t cols) noexcept
{
    Scalar* d = d_span.data();
    const std::size_t n = d_span.size();
    if (n == 0)
        return;
    if (n < kDslParallelThreshold)
    {
        NN_VECTORIZE_PRAGMA
        for (std::size_t i = 0; i < n; ++i)
            d[i] = eval_with_cols(e, i, cols);
        return;
    }

    const std::size_t hw = static_cast<std::size_t>(std::thread::hardware_concurrency());
    // 分块数：以 ~64K 元素/块为目标（≈256KB，L2 友好），上限 hw*4 以保证
    // 负载均衡（块太少会让大量线程闲置，实测 2.36M 元素时退化近 10 倍）。
    constexpr std::size_t TARGET_CHUNK = std::size_t{1} << 16;
    const std::size_t n_chunks = std::clamp(n / TARGET_CHUNK, std::size_t{1},
                                           std::max<std::size_t>(hw, 1) * 4);
    const std::size_t base = n / n_chunks;
    const std::size_t rem  = n % n_chunks;
    nn::parallel_for_samples(n_chunks, [&e, d, cols, base, rem](std::size_t c) noexcept
    {
        const std::size_t off = c * base + std::min(c, rem);
        const std::size_t len = base + (c < rem ? 1 : 0);
        Scalar* dc = d + off;
        NN_VECTORIZE_PRAGMA
        for (std::size_t k = 0; k < len; ++k)
            dc[k] = eval_with_cols(e, off + k, cols);
    });
}
// ══════════════════════════════════════════════════════════════════════════
// CpuViewCache — 叶子节点的 CPU 数据视图缓存
//
// 逐元素模板求值会**按元素**读取叶子（e.eval(i)）。若每个元素都重新
// t.cpu_matrix().span()，会引入固定开销：variant 槽检查 + span 构造 ——
// 实测使模板路径比等价手写循环慢 1.5–4 倍（原地原语只做一次指针遍历）。
// 因此在叶子构造时缓存一次数据视图（ConstSpan 只是 (指针, 长度)，16 字节）。
//
// 非 CPU 张量（GPU）**不**触发 cpu_matrix()：GPU 路径只走 to_spec 折叠，
// 从不调用 eval；此时缓存留空，eval 走兜底（理论上不会到达该分支）。
// ══════════════════════════════════════════════════════════════════════════
struct CpuViewCache
{
    // CPU 张量的数据指针与列数（构造时缓存一次）。非 CPU 张量（GPU）留空：
    // GPU 路径只走 to_spec 折叠，从不调用 eval。
    // 用裸指针而非 ConstSpan：at() 必须**无分支**，否则编译器无法向量化
    // 内层循环（实测带 `empty() ? ... : ...` 兜底分支时原地 add 慢 5 倍）。
    const Scalar* data = nullptr;
    std::size_t cols = 0;

    void cache_cpu_view(const Tensor& t)
    {
        if (!t.is_cpu())
            return;
        const auto& m = t.cpu_matrix();
        data = m.span().data();
        cols = t.cols();
    }
    [[nodiscard]] Scalar at(std::size_t i) const
    {
        NN_ASSERT(data != nullptr,
                  "DSL 叶子 eval 仅适用于 CPU 张量（GPU 路径走 to_spec 折叠）");
        return data[i];
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 叶子节点（同时满足 nn::Expression：可 eval(i)；以及可 to_spec(SpecBuilder&)）
// ══════════════════════════════════════════════════════════════════════════

// 标量常量
struct ConstLeaf
{
    Scalar value;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return value; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_const(value); }
};

// 运行时标量参数叶子（RParam）：常量与运行时的折中。值参与算术但**不进
// expr_spec_key**（非表达式结构），glsl_gen 把 rparams 作为 push constant
// 读取；CPU 求值用 spec.rparams[idx]。适配"每步会变的标量"（优化器超参）。
struct RParamLeaf
{
    Scalar value;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return value; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_rparam(value); }
};

// 线性叶子：直接读取 Tensor 数据（row-major 扁平）
struct TensorRef : CpuViewCache
{
    Tensor t;

    TensorRef(Tensor tt) : t(std::move(tt)) { cache_cpu_view(t); }
    [[nodiscard]] Scalar eval(std::size_t i) const { return at(i); }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_linear(t); }
};

// 视图：RotateHalf —— 按 block 分块，块内前后半行交换 + 前半取负（LLaMA rotate_half）
struct RotateHalfRef : CpuViewCache
{
    Tensor t;
    std::uint32_t block;

    RotateHalfRef(Tensor tt, std::uint32_t blk) : t(std::move(tt)), block(blk)
    { cache_cpu_view(t); }

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        const std::size_t c = i % cols, r = i / cols;
        const std::size_t blk = block;
        const std::size_t rl = r % blk;
        const std::size_t rr = (r / blk) * blk
            + ((rl < blk / 2) ? (rl + blk / 2) : (rl - blk / 2));
        const Scalar v = at(rr * cols + c);
        return (rl < blk / 2) ? -v : v;
    }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_rotate(t, block); }
};

// 视图：RowMod —— 行取模广播（频率表平铺到多行块）
struct RowModRef : CpuViewCache
{
    Tensor t;
    std::uint32_t mod;

    RowModRef(Tensor tt, std::uint32_t m) : t(std::move(tt)), mod(m) { cache_cpu_view(t); }

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        const std::size_t c = i % cols, r = i / cols;
        return at((r % mod) * cols + c);
    }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_rowmod(t, mod); }
};

// 视图：RowAccess —— 行偏移+取模访问：data[(offset + r % mod)*cols + c]
// 用于 SwiGLU 等"同一 (2·d_ff) 输入按半个偏移读取"的行切分：
//   gate = row_access(in, offset=0,      mod=d_ff) → in[r % d_ff]
//   up   = row_access(in, offset=d_ff,   mod=d_ff) → in[d_ff + r % d_ff]
// offset/mod 均为运行时形状数据（共享一个融合 shader，运行时经 vp 槽填充）。
struct RowAccessRef : CpuViewCache
{
    Tensor t;
    std::uint32_t offset;
    std::uint32_t mod;

    RowAccessRef(Tensor tt, std::uint32_t off, std::uint32_t m)
        : t(std::move(tt)), offset(off), mod(m) { cache_cpu_view(t); }

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        const std::size_t c = i % cols, r = i / cols;
        return at((offset + (r % mod)) * cols + c);
    }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_input_rowaccess(t, offset, mod); }
};

// ══════════════════════════════════════════════════════════════════════════
// 归约叶子：把"按行/按列归约出标量向量"接入表达式
//
// 两类（对应 expr_spec.hpp 的归约视图与归约指令两种机制）：
//   1. ReduceViewRef<Kind>：对输入 Tensor 直接归约 → 归约**视图**。
//      求值期代表一个 (rows,1)/(1,cols) 的广播向量，参与算术时自动广播。
//   2. ReduceRef<E, Rop>：对子表达式结果归约 → 归约**指令**，
//      用于 exp(x) 求和这类"表达式内部归约"（见 DslExpr 概念定义之后）。
//
// 含归约的表达式**不实例化**模板 CPU 求值路径（has_reduction_v 把 dsl::compute
// 分流到引擎 eval_expr，由引擎实现归约视图/指令语义），故 eval 仅需满足
// nn::Expression 概念约束，返回占位值。
// ══════════════════════════════════════════════════════════════════════════
template <ExprViewKind Kind>
struct ReduceViewRef
{
    Tensor t;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_reduce_input(t, Kind); }
};

using RowReduceSumRef = ReduceViewRef<ExprViewKind::RowReduceSum>;
using RowReduceMaxRef = ReduceViewRef<ExprViewKind::RowReduceMax>;
using ColReduceSumRef = ReduceViewRef<ExprViewKind::ColReduceSum>;
using ColReduceMaxRef = ReduceViewRef<ExprViewKind::ColReduceMax>;

// ══════════════════════════════════════════════════════════════════════════
// matmul 叶子（算子融合二期 S1/S2：matmul 参与 IR 融合）
//
// matmul(A, B) 折叠成 ExprSpec 的**前置 matmul 段**（MatmulSpec）：
//   逐元素链经 Matmul 操作数按 (r,c) 读取 C = op(A,B)，中间结果不物化。
// A/B 在折叠时登记为输入（Linear 视图），槽位记录进 MatmulSpec.a_input/b_input
// （折叠顺序决定槽位，MatmulRef::to_spec 记录实际槽位，不假设 0/1）。
// k（求和维度）是形状参数：不进 expr_spec_key，同结构不同 K 共享融合 shader。
//
// 模板 CPU 求值路径不实例化（has_reduction_v<MatmulRef>=true 把 dsl::compute
// 分流到引擎 eval_expr，由引擎做 matmul 预计算 + 逐元素链），eval 仅满足
// nn::Expression 概念约束，返回占位值。
// ══════════════════════════════════════════════════════════════════════════
struct MatmulRef
{
    Tensor a, b;
    bool transA = false;
    bool transB = false;
    std::uint32_t batch = 1;  // S7：批量数（形状参数，不进 key）

    // ── CPU 模板求值路径的预绑定状态（引擎内部，Layer 无感知）───────────
    // C 由引擎通用 matmul 原语物化一次并绑定指针；随后尾链（如 +bias）在
    // 编译期模板路径内联求值 → 编译器融合 + 向量化，不再经 ExprSpec 解释器。
    // 详见本节末尾 "CPU 预绑定" 一节的说明。GPU/scan 不使用本状态。
    mutable Tensor c_cache_{};
    mutable const Scalar* c_data_ = nullptr;

    [[nodiscard]] Result<void> prepare_cpu(ComputeEngine& eng) const
    {
        if (c_data_ != nullptr)
            return {};  // 幂等：同一棵树重复求值只物化一次
        constexpr Precision P = Precision::F32;
        Result<Tensor> c = (batch > 1)
            ? eng.batched_matmul(a, b, batch, transA, transB, Scalar{1}, P)
            : eng.matmul(a, b, transA, transB, P);
        if (!c)
            return std::unexpected(c.error());
        if (!c->is_cpu())
            return std::unexpected(Error{"dsl matmul prepare: result not on CPU"});
        c_cache_ = std::move(*c);
        c_data_ = c_cache_.cpu_matrix().span().data();
        return {};
    }

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    // CPU 模板路径（2 参）：读预绑定的 C（形状与输出网格一致）
    [[nodiscard]] Scalar eval(std::size_t i, std::size_t) const noexcept
    {
        return c_data_ != nullptr ? c_data_[i] : Scalar{0};
    }
    ExprOperand to_spec(SpecBuilder& sb) const
    {
        // 显式固定登记顺序（A 先 B 后），保证跨编译器确定（同 Binary 约定）
        const std::uint8_t ai = static_cast<std::uint8_t>(sb.inputs.size());
        (void)sb.add_input_linear(a);
        const std::uint8_t bi = static_cast<std::uint8_t>(sb.inputs.size());
        (void)sb.add_input_linear(b);
        // k = 求和维度：transA=0 → A.cols()（A 存储 (batch*M, K)）；
        // transA=1 → A.rows()/batch（A 存储 (batch*K, M)，每批 (K, M)）
        const std::uint32_t k = static_cast<std::uint32_t>(
            transA ? (a.rows() / batch) : a.cols());
        sb.spec.matmul = MatmulSpec{ai, bi, transA ? std::uint8_t{1} : std::uint8_t{0},
                                    transB ? std::uint8_t{1} : std::uint8_t{0}, k, batch};
        return expr::matmul_op();
    }
};

// ══════════════════════════════════════════════════════════════════════════
// 网格索引叶子（S7）：行号/列号/批次下标作为标量参与算术
//
//   row()   = 当前输出元素在 batch 内的行号
//   col()   = 当前输出列号
//   batch() = 当前批次下标
// 用于位置相关掩码/偏置（causal select(Col > Row, -inf, 0)、ALiBi 等）。
// 走引擎 eval_expr（索引由引擎按网格推导），eval 仅满足概念约束。
// ══════════════════════════════════════════════════════════════════════════
struct RowIdxLeaf  { [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; } ExprOperand to_spec(SpecBuilder&) const { return expr::row(); } };
struct ColIdxLeaf  { [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; } ExprOperand to_spec(SpecBuilder&) const { return expr::col(); } };
struct BatchIdxLeaf{ [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; } ExprOperand to_spec(SpecBuilder&) const { return expr::batch(); } };

// ══════════════════════════════════════════════════════════════════════════
// S7 视图叶子：标签行收集 / 按批次索引
//   row_gather(logits, labels)：读取 logits[label[c]][c]（稀疏 CE loss 收集）；
//     labels 登记为 ColBroadcast 视图（(1,cols)），RowGather 视图的 param
//     指向 labels 输入槽。
//   batch_mod(t, modulo)：读取 t[batch % modulo]（ALiBi 按头斜率等）。
// ══════════════════════════════════════════════════════════════════════════
struct RowGatherRef
{
    Tensor t;      // 主输入（rows, cols），如 logits (classes, total)
    Tensor labels; // 标签 (1, cols) 浮点打包

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& sb) const
    {
        const std::uint8_t ti = static_cast<std::uint8_t>(sb.inputs.size());
        (void)sb.add_input_linear(t);
        const std::uint8_t li = static_cast<std::uint8_t>(sb.inputs.size());
        (void)sb.add_input_colbroadcast(labels);
        sb.spec.views[ti] = expr::row_gather(li);  // RowGather 视图，param=标签槽
        return expr::input(ti);
    }
};

struct BatchModRef
{
    Tensor t;
    std::uint32_t modulo;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& sb) const
    {
        // 直接登记 BatchMod 视图（输入 (1,mod) 小向量，按 batch 取模索引）
        sb.spec.views.push_back(expr::batch_mod(modulo));
        sb.inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(sb.inputs.size() - 1));
    }
};

// 按 (batch, col) 切片（S7）：doc_ids (1, batch*seq) → data[batch*seq + col]
struct BatchColRef
{
    Tensor t;
    std::uint32_t per_batch_cols;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& sb) const
    {
        sb.spec.views.push_back(expr::batch_col(per_batch_cols));
        sb.inputs.push_back(t);
        return expr::input(static_cast<std::uint8_t>(sb.inputs.size() - 1));
    }
};

// 广播视图叶子：输入 (rows,1)/(1,cols) 小向量，按行/列广播参与算术。
//
// 两者都是**纯索引映射**（不物化、无跨元素依赖），因此留在模板求值路径，
// 不经引擎解释器（见 has_reduction_v<BroadcastRef> = false）：
//   - 列广播：输入 (1,cols)，输出列数 = 输入列数 → 1 参 eval(i) 即可正确求值。
//   - 行广播：输入 (rows,1)，读第 i/cols 个标量 —— 需要"输出列数"，输入自身
//     推不出，故 1 参 eval(i)（无 cols）无法给出正确值，返回占位值；行广播
//     一律经 2 参 eval(i, cols)（eval_with_cols 自动选择）求值。
template <ExprViewKind Kind>
struct BroadcastRef : CpuViewCache
{
    Tensor t;

    BroadcastRef(Tensor tt) : t(std::move(tt)) { cache_cpu_view(t); }

    [[nodiscard]] Scalar eval(std::size_t i) const
    {
        if constexpr (Kind == ExprViewKind::ColBroadcast)
            return at(i % cols);   // 输入 (1,cols) → cols 即输出列数
        else
            return Scalar{0};  // 行广播需输出列数，见 eval(i, out_cols)
    }
    [[nodiscard]] Scalar eval(std::size_t i, std::size_t out_cols) const
    {
        if constexpr (Kind == ExprViewKind::ColBroadcast)
            return at(i % cols);
        else
            return at(i / out_cols);  // 输入 (rows,1)：读第 r 个标量
    }
    ExprOperand to_spec(SpecBuilder& b) const
    {
        if constexpr (Kind == ExprViewKind::RowBroadcast)
            return b.add_input_rowbroadcast(t);
        else
            return b.add_input_colbroadcast(t);
    }
};

using RowBroadcastRef = BroadcastRef<ExprViewKind::RowBroadcast>;
using ColBroadcastRef = BroadcastRef<ExprViewKind::ColBroadcast>;

// ══════════════════════════════════════════════════════════════════════════
// DSL 表达式概念（比 nn::Expression 更严格：额外要求可折叠成 ExprSpec）
//
// 用于让 DSL 运算符在约束偏序上严格优先于旧代数 nn::operator* 等，从而
// 在 namespace nn（Layer）里**直接写内联数学表达式**时消除重载歧义（旧
// 代数与 DSL 共用 nn::Expression；若不区分二者，二者对 DSL 叶子同为候选）。
// ══════════════════════════════════════════════════════════════════════════
template <typename T>
concept DslExpr = nn::Expression<T> && requires(SpecBuilder& b, const T& t)
{
    { t.to_spec(b) } -> std::convertible_to<ExprOperand>;
};

// ══════════════════════════════════════════════════════════════════════════
// 节点模板（每个都同时支持 CPU 求值 eval(i) 与 GPU 折叠 to_spec）
// ══════════════════════════════════════════════════════════════════════════

template <typename Op, nn::Expression C>
struct Unary
{
    C child;
    [[nodiscard]] auto eval(std::size_t i) const { return Op::apply(child.eval(i)); }
    [[nodiscard]] auto eval(std::size_t i, std::size_t cols) const
    { return Op::apply(eval_with_cols(child, i, cols)); }
    ExprOperand to_spec(SpecBuilder& b) const
    { return b.add_instr(Op::op_id(), child.to_spec(b)); }
};

// 比较算子（ops::Gt 等）使 eval 返回 bool → 该 Binary 是 BoolExpression，
// 供 select() 使用；to_spec 仍折叠成单条比较指令（输出 0.0/1.0 到寄存器）。
template <typename Op, nn::Expression L, nn::Expression R>
struct Binary
{
    L l; R r;
    [[nodiscard]] auto eval(std::size_t i) const { return Op::apply(l.eval(i), r.eval(i)); }
    [[nodiscard]] auto eval(std::size_t i, std::size_t cols) const
    { return Op::apply(eval_with_cols(l, i, cols), eval_with_cols(r, i, cols)); }
    ExprOperand to_spec(SpecBuilder& b) const
    {
        // 显式固定操作数折叠顺序（l 先 r 后）：C++ 函数实参求值顺序未指定，
        // 直接 b.add_instr(.., l.to_spec(b), r.to_spec(b)) 会让 views/inputs 的
        // 登记顺序随编译器（MSVC 左到右 / Clang 右到左）漂移 → expr_spec_key
        // 跨编译器不稳定，破坏 AOT 匹配。先求值到局部变量以固定顺序。
        const ExprOperand lo = l.to_spec(b);
        const ExprOperand ro = r.to_spec(b);
        return b.add_instr(Op::op_id(), lo, ro);
    }
};

// select(cond, then, else) —— cond 为 BoolExpression，then/else 为值表达式
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
struct Select
{
    C cond; T then_e; E else_e;
    [[nodiscard]] Scalar eval(std::size_t i) const
    { return cond.eval(i) ? then_e.eval(i) : else_e.eval(i); }
    [[nodiscard]] Scalar eval(std::size_t i, std::size_t cols) const
    {
        return (eval_with_cols(cond, i, cols) != Scalar{0})
            ? eval_with_cols(then_e, i, cols) : eval_with_cols(else_e, i, cols);
    }
    ExprOperand to_spec(SpecBuilder& b) const
    {
        // 同 Binary：显式固定折叠顺序（cond → then → else），保证跨编译器确定。
        const ExprOperand co = cond.to_spec(b);
        const ExprOperand to = then_e.to_spec(b);
        const ExprOperand eo = else_e.to_spec(b);
        return b.add_instr(ExprOp::Select, co, to, eo);
    }
};

// 归约节点：对子表达式结果做归约（归约**指令**）→ 隐式标量向量（Reduce 操作数）。
//   row_reduce_sum(exp(x)) 这类"表达式内部归约"；对输入 Tensor 直接归约请用
//   ReduceViewRef（归约视图，GPU 融合更友好）。
template <nn::dsl::DslExpr E, ExprOp Rop>
struct ReduceRef
{
    E child;

    [[nodiscard]] constexpr Scalar eval(std::size_t) const noexcept { return Scalar{0}; }
    ExprOperand to_spec(SpecBuilder& b) const { return b.add_reduce_instr(Rop, child.to_spec(b)); }
};

// ══════════════════════════════════════════════════════════════════════════
// has_reduction_v：表达式树是否**必须**走引擎解释器
//
// 用于 dsl::compute 的 CPU 路径分流：需要"全行/全列"信息、或需要引擎按网格
// 推导的表达式（归约视图/归约指令/matmul/网格索引/标签收集）无法用逐元素
// 模板求值表达 → 折叠成 ExprSpec 走引擎 eval_expr（CPU 扩展语义处理）；
// 其余表达式（含广播视图这类纯索引映射）保持编译期模板求值（内联 + 并行）。
// 该标志只影响 CPU 分支；GPU/scan 一律走 to_expr_spec，与本标志无关。
// ══════════════════════════════════════════════════════════════════════════
template <typename T> inline constexpr bool has_reduction_v = false;

// 归约叶子需走 eval_expr（模板求值无法表达全行/全列归约）
template <ExprViewKind K>
inline constexpr bool has_reduction_v<ReduceViewRef<K>> = true;
// 广播视图**不需要**走 eval_expr：它是纯索引映射（无跨元素依赖），2 参
// eval(i, cols) 即可在模板求值路径上正确求值（见 BroadcastRef 与
// eval_with_cols）→ 留在编译期模板路径（内联 + 并行）。
template <ExprViewKind K>
inline constexpr bool has_reduction_v<BroadcastRef<K>> = false;
// matmul 叶子同样需走 eval_expr（matmul 预计算 + 逐元素链，引擎实现）
template <>
inline constexpr bool has_reduction_v<MatmulRef> = true;
// 索引/收集叶子（S7）：行/列/批次下标与标签收集由引擎按网格推导
template <>
inline constexpr bool has_reduction_v<RowIdxLeaf> = true;
template <>
inline constexpr bool has_reduction_v<ColIdxLeaf> = true;
template <>
inline constexpr bool has_reduction_v<BatchIdxLeaf> = true;
template <>
inline constexpr bool has_reduction_v<RowGatherRef> = true;
template <>
inline constexpr bool has_reduction_v<BatchModRef> = true;
template <>
inline constexpr bool has_reduction_v<BatchColRef> = true;
template <nn::dsl::DslExpr E, ExprOp Rop>
inline constexpr bool has_reduction_v<ReduceRef<E, Rop>> = true;
template <typename Op, nn::Expression C>
inline constexpr bool has_reduction_v<Unary<Op, C>> = has_reduction_v<C>;
template <typename Op, nn::Expression L, nn::Expression R>
inline constexpr bool has_reduction_v<Binary<Op, L, R>> = has_reduction_v<L> || has_reduction_v<R>;
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
inline constexpr bool has_reduction_v<Select<C, T, E>>
    = has_reduction_v<C> || has_reduction_v<T> || has_reduction_v<E>;

// ══════════════════════════════════════════════════════════════════════════
// 叶子构造（普通写法入口）
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline TensorRef leaf(Tensor t) { return TensorRef{std::move(t)}; }

// 运行时标量参数（RParam）：值不限、结构固定 → 同结构共享融合 shader。
// 用于优化器超参（lr/eps/β）、偏差修正系数等每步会变但结构不变的标量。
[[nodiscard]] inline constexpr RParamLeaf rparam(Scalar v) { return RParamLeaf{v}; }

// matmul 叶子（S1/S2）：C = op(A,B)，折叠为 ExprSpec 的前置 matmul 段。
// 逐元素链（如 +bias、激活）自动与 matmul 融合成一个 kernel（GPU AOT）。
// batch（S7）：A/B 按 batch 垂直切分（batched_matmul 同布局），形状参数。
[[nodiscard]] inline MatmulRef matmul(Tensor a, Tensor b,
                                      bool transA = false, bool transB = false,
                                      std::uint32_t batch = 1)
{ return MatmulRef{std::move(a), std::move(b), transA, transB, batch}; }

// 网格索引叶子（S7）：行号/列号/批次下标
[[nodiscard]] inline RowIdxLeaf row()   { return {}; }
[[nodiscard]] inline ColIdxLeaf col()   { return {}; }
[[nodiscard]] inline BatchIdxLeaf batch() { return {}; }

// S7 视图叶子：标签行收集（稀疏 CE loss）/ 按批次取模索引（ALiBi 斜率）/
// 按 (batch,col) 切片（doc_ids）
[[nodiscard]] inline RowGatherRef row_gather(Tensor t, Tensor labels)
{ return RowGatherRef{std::move(t), std::move(labels)}; }
[[nodiscard]] inline BatchModRef batch_mod(Tensor t, std::uint32_t modulo)
{ return BatchModRef{std::move(t), modulo}; }
[[nodiscard]] inline BatchColRef batch_col(Tensor t, std::uint32_t per_batch_cols)
{ return BatchColRef{std::move(t), per_batch_cols}; }

[[nodiscard]] inline RotateHalfRef rotate_half(Tensor t, std::uint32_t block)
{ return RotateHalfRef{std::move(t), block}; }

[[nodiscard]] inline RowModRef row_mod(Tensor t, std::uint32_t mod)
{ return RowModRef{std::move(t), mod}; }

// 行偏移+取模访问（SwiGLU 半切分等）：data[(offset + r % mod)*cols + c]
[[nodiscard]] inline RowAccessRef row_access(Tensor t, std::uint32_t offset, std::uint32_t mod)
{ return RowAccessRef{std::move(t), offset, mod}; }

// ══════════════════════════════════════════════════════════════════════════
// 归约自由函数：对输入 Tensor 直接归约 → 归约**视图**（GPU 融合更友好）；
// 对表达式结果归约 → 归约**指令**（重载按参数类型自动选择，见 ReduceRef）。
// 返回的归约叶子参与算术时自动按行/按列广播。
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline RowReduceSumRef row_reduce_sum(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline RowReduceMaxRef row_reduce_max(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColReduceSumRef col_reduce_sum(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColReduceMaxRef col_reduce_max(Tensor t) { return {std::move(t)}; }

template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_sum(const E& e) { return ReduceRef<E, ExprOp::RowSum>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto row_reduce_max(const E& e) { return ReduceRef<E, ExprOp::RowMax>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_sum(const E& e) { return ReduceRef<E, ExprOp::ColSum>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto col_reduce_max(const E& e) { return ReduceRef<E, ExprOp::ColMax>{e}; }

// 广播视图自由函数：gamma/beta (F,1) 按行广播、std_inv (1,B) 按列广播
[[nodiscard]] inline RowBroadcastRef row_broadcast(Tensor t) { return {std::move(t)}; }
[[nodiscard]] inline ColBroadcastRef col_broadcast(Tensor t) { return {std::move(t)}; }

// ══════════════════════════════════════════════════════════════════════════
// 一元函数（普通数学写法）
// ══════════════════════════════════════════════════════════════════════════
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto neg(const E& e) { return Unary<ops::Neg, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto abs(const E& e)   { return Unary<ops::Abs, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto exp(const E& e)   { return Unary<ops::Exp, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto log(const E& e)   { return Unary<ops::Log, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto sqrt(const E& e)  { return Unary<ops::Sqrt, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto rsqrt(const E& e) { return Unary<ops::Rsqrt, E>{e}; }
template <nn::dsl::DslExpr E> [[nodiscard]] auto tanh(const E& e)  { return Unary<ops::Tanh, E>{e}; }

// ══════════════════════════════════════════════════════════════════════════
// 二元函数：max / min / select
// ══════════════════════════════════════════════════════════════════════════
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> [[nodiscard]] constexpr auto max(const L& l, const R& r) { return Binary<ops::Max, L, R>{l, r}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto max(const E& e, Scalar s) { return Binary<ops::Max, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto max(Scalar s, const E& e) { return Binary<ops::Max, ConstLeaf, E>{ConstLeaf{s}, e}; }
template <nn::dsl::DslExpr L, nn::dsl::DslExpr R> [[nodiscard]] constexpr auto min(const L& l, const R& r) { return Binary<ops::Min, L, R>{l, r}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto min(const E& e, Scalar s) { return Binary<ops::Min, E, ConstLeaf>{e, ConstLeaf{s}}; }
template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto min(Scalar s, const E& e) { return Binary<ops::Min, ConstLeaf, E>{ConstLeaf{s}, e}; }

/// relu(x) = max(x, 0)
template <nn::dsl::DslExpr E> [[nodiscard]] auto relu(const E& e) { return nn::dsl::max(e, ConstLeaf{Scalar{0}}); }

template <nn::BoolExpression C, nn::dsl::DslExpr T, nn::dsl::DslExpr E>
[[nodiscard]] constexpr auto select(const C& c, const T& t, const E& e)
{ return Select<C, T, E>{c, t, e}; }
template <nn::BoolExpression C, nn::dsl::DslExpr T>
[[nodiscard]] constexpr auto select(const C& c, const T& t, Scalar ev)
{ return Select<C, T, ConstLeaf>{c, t, ConstLeaf{ev}}; }
template <nn::BoolExpression C, nn::dsl::DslExpr E>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, const E& e)
{ return Select<C, ConstLeaf, E>{c, ConstLeaf{tv}, e}; }
template <nn::BoolExpression C>
[[nodiscard]] constexpr auto select(const C& c, Scalar tv, Scalar ev)
{ return Select<C, ConstLeaf, ConstLeaf>{c, ConstLeaf{tv}, ConstLeaf{ev}}; }

// ══════════════════════════════════════════════════════════════════════════
// 运算符（Sym 生成，覆盖 Expr×Expr / Expr×Scalar / Scalar×Expr）
// 值运算（+ - * /）返回值表达式；比较（> < >= <= == !=）返回布尔表达式。
// ══════════════════════════════════════════════════════════════════════════
#define NN_DSL_BINARY(OP, OpT)                                                            \
    template <nn::dsl::DslExpr L, nn::dsl::DslExpr R>                                     \
    [[nodiscard]] constexpr auto operator OP(const L& l, const R& r)                      \
    { return Binary<OpT, L, R>{l, r}; }                                                   \
    template <nn::dsl::DslExpr E>                                                         \
    [[nodiscard]] constexpr auto operator OP(const E& e, Scalar s)                        \
    { return Binary<OpT, E, ConstLeaf>{e, ConstLeaf{s}}; }                                \
    template <nn::dsl::DslExpr E>                                                         \
    [[nodiscard]] constexpr auto operator OP(Scalar s, const E& e)                        \
    { return Binary<OpT, ConstLeaf, E>{ConstLeaf{s}, e}; }

NN_DSL_BINARY(+, ops::Add)
NN_DSL_BINARY(-, ops::Sub)
NN_DSL_BINARY(*, ops::Mul)
NN_DSL_BINARY(/, ops::Div)
NN_DSL_BINARY(>, ops::Gt)
NN_DSL_BINARY(<, ops::Lt)
NN_DSL_BINARY(>=, ops::Ge)
NN_DSL_BINARY(<=, ops::Le)
NN_DSL_BINARY(==, ops::Eq)
NN_DSL_BINARY(!=, ops::Ne)
#undef NN_DSL_BINARY

template <nn::dsl::DslExpr E> [[nodiscard]] constexpr auto operator-(const E& e)
{ return Unary<ops::Neg, E>{e}; }

// ══════════════════════════════════════════════════════════════════════════
// GPU AOT：编译期表达式 → 扁平 ExprSpec + 输入张量（闭合世界分发用）
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] std::pair<ExprSpec, std::vector<Tensor>> to_expr_spec(const E& e)
{
    SpecBuilder b;
    (void)e.to_spec(b);
    return {std::move(b.spec), std::move(b.inputs)};
}

// ══════════════════════════════════════════════════════════════════════════
// CPU 预绑定（cpu_prepare）—— 让"含 matmul/归约"的表达式也走编译期内联
//
// 背景：has_reduction_v 仅用于 CPU 分支分流（GPU/scan 一律 to_expr_spec）。
// matmul / 归约视图 / 归约指令 / 网格索引这类"需要 GEMM 或全行全列才能算出"
// 的节点，此前把**整棵表达式**丢给运行时 ExprSpec 解释器（逐元素
// Scalar regs[16]={} + switch(op) + lambda 间接调用）。
//
// 本节与文件头 §1 的既有设计意图一致（"CPU 直接把表达式当作编译期 AST，
// 逐元素求值 → 编译器内联 + SIMD 融合，等价手写 for 循环"）：把真正需要
// 全局信息的节点**先算出来并绑定**，其余部分仍交给 eval_into_span 内联求值。
// 即：融合交给编译器，而不是运行时解释器。
//
// 红线：本节只做"通用结构"的预绑定（matmul / 归约 / 索引映射），不含任何
// 算法名或算法逻辑；公式文本仍只在 Layer。GPU 路径完全不经过本段。
//
// ⚠ 白名单必须**fail-safe**：占位 eval() 的节点（ReduceViewRef/ReduceRef/
//   RowIdxLeaf/ColIdxLeaf/BatchIdxLeaf/RowGatherRef/BatchModRef/BatchColRef 的
//   eval 目前返回占位值，真实语义只由解释器提供）一旦被放进模板路径就会
//   **静默算错**。因此：默认只有 has_reduction_v 为假（本就是模板路径原生
//   节点）才可预绑定，含归约的节点必须逐个显式加入白名单。
//   教训：曾把默认值设为 true，导致 SwiGLU::backward 的 select(row()<d_ff,…)
//   在模板路径下 row() 恒为 0，两半梯度选错（layer_gradcheck 的 fc1.w 全红）。
// ══════════════════════════════════════════════════════════════════════════

// 含归约节点的白名单：已确认可"预绑定 + 模板内联"的才列在此（逐个迁移）
template <typename T> inline constexpr bool cpu_preparable_leaf_v = false;

// matmul：用引擎通用 GEMM 原语物化 C 后绑定指针，尾链交模板路径内联
template <> inline constexpr bool cpu_preparable_leaf_v<MatmulRef> = true;

// 树级判定：不含归约的节点本就是模板路径原生节点，可直接预绑定（no-op）；
// 含归约的节点只有白名单内的才可预绑定
template <typename T>
inline constexpr bool cpu_preparable_v = !has_reduction_v<T> || cpu_preparable_leaf_v<T>;

template <typename Op, nn::Expression C>
inline constexpr bool cpu_preparable_v<Unary<Op, C>> = cpu_preparable_v<C>;
template <typename Op, nn::Expression L, nn::Expression R>
inline constexpr bool cpu_preparable_v<Binary<Op, L, R>>
    = cpu_preparable_v<L> && cpu_preparable_v<R>;
template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
inline constexpr bool cpu_preparable_v<Select<C, T, E>>
    = cpu_preparable_v<C> && cpu_preparable_v<T> && cpu_preparable_v<E>;

// 默认：普通叶子 / 视图 / 索引叶子无需预绑定
template <typename T>
[[nodiscard]] inline Result<void> cpu_prepare(const T&, ComputeEngine&,
                                              std::size_t, std::size_t)
{ return {}; }

template <typename Op, nn::Expression C>
[[nodiscard]] inline Result<void> cpu_prepare(const Unary<Op, C>& u, ComputeEngine& eng,
                                              std::size_t rows, std::size_t cols)
{ return cpu_prepare(u.child, eng, rows, cols); }

template <typename Op, nn::Expression L, nn::Expression R>
[[nodiscard]] inline Result<void> cpu_prepare(const Binary<Op, L, R>& b, ComputeEngine& eng,
                                              std::size_t rows, std::size_t cols)
{
    if (auto r = cpu_prepare(b.l, eng, rows, cols); !r)
        return r;
    return cpu_prepare(b.r, eng, rows, cols);
}

template <nn::BoolExpression C, nn::Expression T, nn::Expression E>
[[nodiscard]] inline Result<void> cpu_prepare(const Select<C, T, E>& s, ComputeEngine& eng,
                                              std::size_t rows, std::size_t cols)
{
    if (auto r = cpu_prepare(s.cond, eng, rows, cols); !r)
        return r;
    if (auto r = cpu_prepare(s.then_e, eng, rows, cols); !r)
        return r;
    return cpu_prepare(s.else_e, eng, rows, cols);
}

// matmul：把 C 物化一次（引擎通用 GEMM 原语）并绑定指针
[[nodiscard]] inline Result<void> cpu_prepare(const MatmulRef& m, ComputeEngine& eng,
                                              std::size_t, std::size_t)
{ return m.prepare_cpu(eng); }

// ══════════════════════════════════════════════════════════════════════════
// CPU：编译期模板直接求值（编译器内联 + SIMD 融合，等价手写循环）
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Tensor eval_cpu(const E& e, std::size_t rows, std::size_t cols)
{
    // 输出会被 eval_into_span 完整覆盖（每个下标恰好写一次）→ 用未初始化构造，
    // 省掉"分配 + 写满一遍零 + 马上被全覆盖"里的那一遍全尺寸零写。
    // 实测本机单线程写满 1.57MB 要 0.50ms（~3.2 GB/s），是纯浪费。
    Matrix out = Matrix::make_uninitialized(rows, cols);
    // 与 eager 逐元素原语同构（串行+向量化提示 / 阈值以上并行）→ 同门控下
    // DSL 取代 elementwise_* 时 CPU 性能不倒退。
    eval_into_span(e, out.span(), cols);
    return Tensor::from_matrix(std::move(out));
}

// ══════════════════════════════════════════════════════════════════════════
// 统一入口：engine.compute(expr, rows, cols)
//   CPU：编译期模板求值（SIMD 融合）。
//   GPU：闭合世界 AOT —— to_expr_spec 折叠 → engine.eval_expr 匹配预生成
//        shader；未命中由 eval_expr 硬报错（无 eager）。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Result<Tensor> compute(ComputeEngine& eng, const E& e,
                                     std::size_t rows, std::size_t cols)
{
#ifdef NN_EXPR_SCAN
    // 构建期扫描模式：折叠内联表达式的**结构**并登记进全局注册表，
    // 返回占位张量让 dry-run 流程继续（scan 只关心表达式集合，不真算）。
    // 表达式文本仍只出现在 Layer；这里登记的是派生物 ExprSpec。
    (void)eng;
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    fused::global_registry().add(spec);
    return Tensor::cpu(rows, cols);
#else
    if (eng.device() == Device::CPU)
    {
        if constexpr (nn::dsl::has_reduction_v<E>)
        {
            if constexpr (nn::dsl::cpu_preparable_v<E>)
            {
                // 可预绑定：先物化需要全局信息的节点（如 matmul C），
                // 其余交编译期模板内联求值（同一个 eval_into_span 循环）。
                if (auto r = cpu_prepare(e, eng, rows, cols); !r)
                    return std::unexpected(r.error());
                return eval_cpu(e, rows, cols);
            }
            // 含归约且暂不可预绑定：模板求值无法表达"全行/全列归约"，
            // 折叠成 ExprSpec 走引擎 eval_expr（CPU 扩展语义处理归约视图/指令）
            auto [spec, inputs] = to_expr_spec(e);
            if (auto v = validate_expr_spec(spec, inputs.size()); !v)
                return std::unexpected(v.error());
            return eng.eval_expr(spec, inputs, rows, cols);
        }
        return eval_cpu(e, rows, cols);
    }

    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr(spec, inputs, rows, cols);  // 闭合世界：GPU 未命中即报错
#endif
}

// ══════════════════════════════════════════════════════════════════════════
// compute_into — 目标传递入口：dst = expr（不分配新张量）
//
// 与 compute() 共用同一前端/同一 IR/同一 AOT 匹配，唯一差别是**输出落点**：
// 结果直接写进调用方提供的 dst。用于把"原地更新"语义纳入 DSL（此前只能靠
// 引擎的原地原语 add_inplace / scale_inplace / axpy_inplace / broadcast_*）：
//   dst += expr            → compute_into(eng, leaf(dst) + expr, dst)
//   dst *= k               → compute_into(eng, leaf(dst) * rparam(k), dst)
//   dst += k * other       → compute_into(eng, leaf(dst) + leaf(other) * rparam(k), dst)
//   dst += row_broadcast(v)→ compute_into(eng, leaf(dst) + row_broadcast(v), dst)
// dst 与某个输入是同一 buffer 是安全的（逐元素先读后写）。
//
// CPU 走编译期模板求值（与 compute() 的 eval_cpu 同一路径 + 同一并行门控），
// 因此"原地"不引入任何分配/拷贝；GPU 走 AOT 融合 shader 的 output_override。
// 仅支持逐元素表达式（无归约）；归约向量输出用 compute_reduce。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Result<void> compute_into(ComputeEngine& eng, const E& e, Tensor& dst)
{
#ifdef NN_EXPR_SCAN
    // 构建期扫描：与 compute() 一样只登记结构（不真算、不关心 dst 的值）
    (void)eng;
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    (void)dst;
    fused::global_registry().add(spec);
    return {};
#else
    if (eng.device() == Device::CPU)
    {
        if (!dst.is_cpu())
            return std::unexpected(Error{"dsl::compute_into: dst not on CPU"});
        if constexpr (nn::dsl::has_reduction_v<E>)
        {
            if constexpr (nn::dsl::cpu_preparable_v<E>)
            {
                // 可预绑定：物化需要全局信息的节点后，直接内联写进 dst
                // （与上面纯逐元素分支同一循环结构；dst 与输入同 buffer 安全）
                if (auto r = cpu_prepare(e, eng, dst.rows(), dst.cols()); !r)
                    return std::unexpected(r.error());
                eval_into_span(e, dst.cpu_matrix().span(), dst.cols());
                return {};
            }
            // 含归约：折叠成 ExprSpec 走引擎（与 compute() 的 CPU 分支一致）
            auto [spec, inputs] = to_expr_spec(e);
            if (auto v = validate_expr_spec(spec, inputs.size()); !v)
                return std::unexpected(v.error());
            return eng.eval_expr_into(spec, inputs, dst.rows(), dst.cols(), dst);
        }
        // 纯逐元素：编译期模板直接写进 dst 的 span（零解释器开销、零分配），
        // 与 eval_cpu / 逐元素原语同一循环结构 + 同一并行门控。
        eval_into_span(e, dst.cpu_matrix().span(), dst.cols());
        return {};
    }

    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr_into(spec, inputs, dst.rows(), dst.cols(), dst);
#endif
}

// ══════════════════════════════════════════════════════════════════════════
// 归约向量原生形状输出：compute_reduce(engine, expr, rows, cols)
//
// 与 compute() 等价，但输出为归约向量本身（(rows,1)/(1,cols)），而非广播到
// (rows,cols)。用于 LayerNorm/RMSNorm 的 (1,B) 统计量缓存（mean/var/rms_inv）
// 与 (F,1) 梯度归约（grad_gamma/grad_beta）——只产出小向量，避免写全尺寸广播。
// 要求表达式归约轴为 0/1（否则引擎报错）。
// ══════════════════════════════════════════════════════════════════════════
template <typename E>
[[nodiscard]] Result<Tensor> compute_reduce(ComputeEngine& eng, const E& e,
                                            std::size_t rows, std::size_t cols)
{
#ifdef NN_EXPR_SCAN
    // 构建期扫描：同 compute()，登记结构（归约轴由 gen_fused 判定）。
    // 占位张量按归约轴取向量形状 (rows,1)/(1,cols)，使 Layer 后续
    // add_inplace 等形状相关操作在 dry-run 中不因形状失配而中断。
    (void)eng;
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    const int raxis = expr_spec_reduce_axis(spec);
    fused::global_registry().add(spec);
    return (raxis == 0) ? Tensor::cpu(rows, 1)
         : (raxis == 1) ? Tensor::cpu(1, cols)
         : Tensor::cpu(rows, cols);
#else
    auto [spec, inputs] = to_expr_spec(e);
    if (auto v = validate_expr_spec(spec, inputs.size()); !v)
        return std::unexpected(v.error());
    return eng.eval_expr_reduce(spec, inputs, rows, cols);
#endif
}

} // namespace nn::dsl

