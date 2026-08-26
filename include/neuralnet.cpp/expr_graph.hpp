#ifndef NN_EXPR_GRAPH_HPP
#define NN_EXPR_GRAPH_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_graph.hpp — 图 IR + 融合分析（IR-C）
//
//  对应文档 `docs/11-ir-optimization.md` IR-C：把扁平 ExprSpec 演进为图 IR
//  （DAG），为 begin_expr/end_expr 提供多表达式融合分析。
//
//  图结构：
//    - 节点 = 一次 eval_expr / eval_expr_reduce（一个 canonical ExprSpec）。
//    - 边 = 数据依赖：节点的某个输入槽引用前序节点的输出（虚拟寄存器）。
//    - 录制：begin_expr 开启录制图；期间每次表达式求值把节点加入图并返回
//      携带 virtual_tag 的占位 Tensor；end_expr 做融合分析并产出 kernel 序列。
//
//  融合策略（本阶段落地：逐元素链拼接）：
//    - 相邻节点 A → B（B 以 Linear 视图消费 A 的输出，且 A 无其他消费者）
//      且 A、B 均为逐元素（无归约）、形状相同 → 把 A 的指令内联进 B，
//      A 的输出寄存器直接作为 B 的操作数 → 中间结果不落显存，一次 dispatch。
//    - 归约表达式 / 归约输出 / matmul 等作为融合边界 → 独立 kernel。
//    - 拼接后 canonicalize + validate；超限（instrs/regs/inputs/consts）
//      或非法 → 放弃融合（保守，保证正确性优先）。
//
//  确定性铁律（与 expr_opt.hpp 一致）：贪心按节点序从前到后；拼接重映射
//  顺序固定；结果经 canonicalize_expr_spec 统一规范化（两端一致）。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "compute_tensor.hpp"
#include "expr_spec.hpp"
#include "expr_opt.hpp"

namespace nn
{

// ── 图 IR 节点 ────────────────────────────────────────────────────────────
struct ExprGraphNode
{
    ExprSpec    spec;          // canonical IR（录制时已 canonicalize）
    std::size_t rows = 0;      // 输出形状（统一网格 (rows, cols)）
    std::size_t cols = 0;
    bool        vector_out = false;  // 输出为归约向量（(rows,1)/(1,cols)）
    // 依赖：dep_of_input[k] = 该输入槽依赖的图节点下标；-1 = 外部输入。
    // 长度 == spec.views.size()（与输入一一对应）。
    std::vector<int> dep_of_input;
    // 外部输入张量：dep_of_input[k] < 0 时有效（Layer 传入的真实输入）。
    std::vector<Tensor> input_tensors;
};

// ── 图（录制期累积） ─────────────────────────────────────────────────────
// D3 修复：node_outputs 移入 ExprGraph（随图一起堆分配、thread-local 指针
// 持有），消除 GpuEngine 成员 node_outputs_ 的跨线程共享数据竞争。
struct ExprGraph
{
    std::vector<ExprGraphNode> nodes;
    // 占位 Tensor 的 virtual_tag → 节点下标（录制时建立，供依赖识别）
    std::unordered_map<std::uint64_t, int> node_by_tag;
    // 节点下标 → 占位输出 Tensor（end_expr 时写入真实结果）
    std::unordered_map<int, Tensor> node_outputs;

    // 录制：把一个表达式加入图，返回节点下标。
    // inputs 中带 virtual_tag 的占位 Tensor 会被识别为前序节点输出（依赖边）。
    [[nodiscard]] int add_node(const ExprSpec& raw_spec,
                               std::span<const Tensor> inputs,
                               std::size_t rows, std::size_t cols,
                               bool vector_out)
    {
        const ExprSpec spec = canonicalize_expr_spec(raw_spec);
        const int idx = static_cast<int>(nodes.size());
        ExprGraphNode nd;
        nd.spec        = spec;
        nd.rows        = rows;
        nd.cols        = cols;
        nd.vector_out  = vector_out;
        nd.dep_of_input.reserve(spec.views.size());
        nd.input_tensors.reserve(spec.views.size());
        for (std::size_t k = 0; k < spec.views.size(); ++k)
        {
            const std::uint64_t tag = (k < inputs.size())
                ? inputs[k].virtual_tag() : 0;
            const auto it = (tag != 0) ? node_by_tag.find(tag) : node_by_tag.end();
            if (it != node_by_tag.end())
            {
                nd.dep_of_input.push_back(it->second);
                nd.input_tensors.emplace_back();  // 占位（依赖节点输出）
            }
            else
            {
                nd.dep_of_input.push_back(-1);
                nd.input_tensors.push_back(
                    (k < inputs.size()) ? inputs[k] : Tensor{});
            }
        }
        nodes.push_back(std::move(nd));
        // 登记输出标记（tag = 节点下标 + 1，0 保留给"非节点输出"）
        node_by_tag[static_cast<std::uint64_t>(idx) + 1ull] = idx;
        return idx;
    }
};

// ── 融合 kernel（融合分析输出） ──────────────────────────────────────────
// 一个 kernel = 一个（可能为多个节点拼接的）复合 ExprSpec + 外部输入来源。
struct FusedKernelInput
{
    int     node = -1;   // >= 0：该输入是图节点 node 的输出（作为外部 buffer 绑定）
    Tensor  external;    // node < 0：Layer 外部输入
};

struct FusedKernel
{
    ExprSpec    spec;             // 复合 spec（canonical）
    std::vector<FusedKernelInput> inputs;  // 与 spec.views 一一对应
    std::size_t rows = 0;
    std::size_t cols = 0;
    bool        vector_out = false;
    int         tail = -1;        // 末尾节点（输出节点）在图中下标
    std::vector<int> members;     // kernel 包含的节点（顺序 = 融合顺序）
};

// ── 全局录制状态（线程局部，D3 修复） ──────────────────────────────────
// begin_expr/end_expr（GPU 引擎执行 / scan 引擎登记）开启/关闭录制图。
// dsl::compute 的 scan 分支与 GpuEngine::eval_expr 通过 recording_graph()
// 判断"当前是否在录制"以及把表达式加入录制图。
//
// D3 修复：录制图由 thread_local unique_ptr 持有（堆分配），而非引擎成员
// （std::optional<ExprGraph> recording_）。原设计中 recording_ 是引擎成员
// （跨线程共享），而 thread_local 指针指向它——两个线程同时 begin_expr 会
// 让各自的 thread_local 指针指向同一个引擎成员，且 node_outputs_ 也是共享
// 成员 → 数据竞争。改为每线程独立堆分配后，各线程的录制图完全隔离。
namespace fused
{

[[nodiscard]] inline std::unique_ptr<ExprGraph>& recording_graph_owner()
{
    static thread_local std::unique_ptr<ExprGraph> g;
    return g;
}
[[nodiscard]] inline ExprGraph* recording_graph() noexcept
{ return recording_graph_owner().get(); }
[[nodiscard]] inline bool is_recording() noexcept
{ return recording_graph_owner() != nullptr; }

} // namespace fused

// ── 融合分析（IR-C 核心，纯函数） ───────────────────────────────────────
// 贪心按节点序：维护"当前打开的 kernel"（其末尾节点为 tail）。节点 i 若
// 满足"并入"条件则并入，否则新开 kernel。
[[nodiscard]] inline std::vector<FusedKernel> fuse_expr_graph(const ExprGraph& g)
{
    std::vector<FusedKernel> kernels;
    if (g.nodes.empty())
        return kernels;

    // 消费者表：节点 j 被哪些节点消费（dep_of_input 引用 j）
    std::vector<std::vector<int>> consumers(g.nodes.size());
    for (std::size_t i = 0; i < g.nodes.size(); ++i)
        for (const int dep : g.nodes[i].dep_of_input)
            if (dep >= 0)
                consumers[static_cast<std::size_t>(dep)].push_back(static_cast<int>(i));

    // ── 尝试把节点 bi 并入当前 kernel（cur） ───────────────────────────
    const auto try_append = [&](FusedKernel& cur, int bi) -> bool
    {
        const ExprGraphNode& B = g.nodes[static_cast<std::size_t>(bi)];
        const int tail = cur.tail;
        const ExprSpec& A = cur.spec;   // 当前复合 spec（含 tail 及之前节点）
        const ExprSpec& Bs = B.spec;

        // 1) 前置：均逐元素（无归约）、非归约输出、形状相同
        if (cur.vector_out || B.vector_out) return false;
        if (expr_spec_reduce_axis(A) != -1) return false;
        if (expr_spec_reduce_axis(Bs) != -1) return false;
        if (cur.rows != B.rows || cur.cols != B.cols) return false;

        // 2) tail 无其他消费者，且被 B 以 Linear 视图消费
        if (consumers[static_cast<std::size_t>(tail)].size() != 1 ||
            consumers[static_cast<std::size_t>(tail)][0] != bi)
            return false;
        std::vector<int> tail_slots;   // B 中引用 tail 的输入槽
        std::vector<int> other_slots;  // B 中其他输入槽
        for (std::size_t k = 0; k < Bs.views.size(); ++k)
        {
            if (B.dep_of_input[k] == tail)
            {
                if (static_cast<ExprViewKind>(Bs.views[k].kind) != ExprViewKind::Linear)
                    return false;  // 仅支持 Linear 引用（寄存器直接替换）
                tail_slots.push_back(static_cast<int>(k));
            }
            else
            {
                other_slots.push_back(static_cast<int>(k));
            }
        }
        if (tail_slots.empty())
            return false;

        // 3) 拼接（确定性：tail 的 views/consts/instrs 在前，B 的在后）
        const std::size_t out_reg_A = A.instrs.back().dst;

        // 3a) 寄存器偏移溢出防护：B 的指令 dst 经 +A.num_regs 重映射后必须 < 256
        // （ExprInstr.dst 为 uint8_t）。超限则放弃融合（保守，正确性优先）。
        // ⚠ 寄存器偏移必须用 A.num_regs（A 占用的寄存器数），而非 A 的指令数。
        // canonical 后指令数与寄存器数可能不同（liveness 复用），用指令数会产生
        // 寄存器空洞（A 的寄存器 0..num_regs-1 之后空出 instrs.size()-num_regs 个号），
        // 浪费寄存器并可能使本可融合的表达式超出 EXPR_MAX_REGS=16 而被放弃。
        for (const auto& ins : Bs.instrs)
            if (static_cast<std::size_t>(ins.dst) + A.num_regs >= 256)
                return false;

        std::vector<ExprView> views = A.views;
        std::vector<int> b_view_map(Bs.views.size(), -1);
        for (const int k : other_slots)
        {
            b_view_map[static_cast<std::size_t>(k)] = static_cast<int>(views.size());
            views.push_back(Bs.views[static_cast<std::size_t>(k)]);
        }

        std::vector<Scalar> consts = A.consts;
        const std::size_t b_const_base = A.consts.size();
        // 关键：B 的常量池必须追加在 A 之后（B 的 Const 引用经偏移指向这里）
        consts.insert(consts.end(), Bs.consts.begin(), Bs.consts.end());

        // 寄存器偏移基准：A 占用的寄存器数（canonical 后 A 的寄存器号为
        // 0..A.num_regs-1）。B 的寄存器整体平移 A.num_regs，避免与 A 冲突。
        // （用 A.num_regs 而非 A.instrs.size()：canonical 后指令数 ≥ 寄存器数，
        //  用指令数会留下寄存器空洞；虽然后续 canonicalize 会重新紧凑编号，
        //  但用寄存器数更精确，且溢出检查更宽松，允许更多融合。）
        const std::size_t reg_base = A.num_regs;

        std::vector<ExprInstr> instrs = A.instrs;
        for (const auto& ins : Bs.instrs)
        {
            ExprInstr ni = ins;
            ni.dst = static_cast<std::uint8_t>(ni.dst + reg_base);
            const auto remap_op = [&](ExprOperand o) -> ExprOperand
            {
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Input))
                {
                    const std::size_t k = o.idx;
                    const auto it = std::find(tail_slots.begin(), tail_slots.end(),
                                              static_cast<int>(k));
                    if (it != tail_slots.end())
                        return expr::reg(static_cast<std::uint8_t>(out_reg_A));
                    return expr::input(static_cast<std::uint8_t>(
                        b_view_map[k]));
                }
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
                    return expr::cst(static_cast<std::uint8_t>(
                        o.idx + b_const_base));
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                    o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                    o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce))
                    return {o.kind, static_cast<std::uint8_t>(o.idx + reg_base)};
                return o;
            };
            const std::size_t nops =
                expr_instr_num_operands(static_cast<ExprOp>(ins.op));
            ni.a = remap_op(ins.a);
            if (nops >= 2) ni.b = remap_op(ins.b);
            if (nops >= 3) ni.c = remap_op(ins.c);
            instrs.push_back(ni);
        }

        ExprSpec fused;
        fused.views   = std::move(views);
        fused.consts  = std::move(consts);
        fused.instrs  = std::move(instrs);
        fused.num_regs = static_cast<std::uint32_t>(reg_base + Bs.num_regs);
        fused = canonicalize_expr_spec(fused);
        if (auto v = validate_expr_spec(fused, fused.views.size()); !v)
            return false;  // 超限/非法 → 放弃融合（保守）

        // 4) 提交：更新 cur
        std::vector<FusedKernelInput> new_inputs = cur.inputs;
        for (const int k : other_slots)
        {
            const std::size_t kk = static_cast<std::size_t>(k);
            if (B.dep_of_input[kk] < 0)
                new_inputs.push_back(FusedKernelInput{
                    /*node=*/-1, /*external=*/B.input_tensors[kk]});
            else
                new_inputs.push_back(FusedKernelInput{
                    /*node=*/B.dep_of_input[kk], /*external=*/Tensor{}});
        }
        cur.spec       = std::move(fused);
        cur.inputs     = std::move(new_inputs);
        cur.rows       = B.rows;
        cur.cols       = B.cols;
        cur.tail       = bi;
        cur.members.push_back(bi);
        return true;
    };

    // ── 贪心主循环 ─────────────────────────────────────────────────────
    for (std::size_t i = 0; i < g.nodes.size(); ++i)
    {
        const auto& N = g.nodes[i];
        if (!kernels.empty() && try_append(kernels.back(), static_cast<int>(i)))
            continue;

        FusedKernel k;
        k.spec       = N.spec;
        k.rows       = N.rows;
        k.cols       = N.cols;
        k.vector_out = N.vector_out;
        k.tail       = static_cast<int>(i);
        k.members.push_back(static_cast<int>(i));
        for (std::size_t kk = 0; kk < N.spec.views.size(); ++kk)
        {
            if (N.dep_of_input[kk] < 0)
                k.inputs.push_back(FusedKernelInput{
                    /*node=*/-1, /*external=*/N.input_tensors[kk]});
            else
                k.inputs.push_back(FusedKernelInput{
                    /*node=*/N.dep_of_input[kk], /*external=*/Tensor{}});
        }
        kernels.push_back(std::move(k));
    }
    return kernels;
}

} // namespace nn

#endif // NN_EXPR_GRAPH_HPP
