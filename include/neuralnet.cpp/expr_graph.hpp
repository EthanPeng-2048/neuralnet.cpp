#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_graph.hpp — 图 IR + 融合分析（IR-C）
//
//  2026-10-06 恢复：本文件按 git `8f2990f^`（e936f4d）的旧实现移植回主干
//  （用户裁定：IR-C 非否决，系当年实现不完整而暂时废弃；取舍记录与恢复立项
//  见 docs/history.md「IR-C 定位修正与恢复立项」）。相对旧版的差异只有两处：
//    1. 图节点带 Precision，融合要求成员精度一致（旧版无精度维度，混精度链
//       拼接会把不同精度语义塞进单 kernel = 静默错值）；
//    2. Tensor 占位标记经 M1 后的公共元数据访问器（virtual_tag/
//       set_virtual_tag），存储访问仍收口在引擎域内。
//
//  对应文档 `docs/development/03-ir-optimization.md` IR-C：把扁平 ExprSpec 演进为图 IR
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
//      且 A、B 均为逐元素（无归约）、形状相同、精度相同 → 把 A 的指令内联进 B，
//      A 的输出寄存器直接作为 B 的操作数 → 中间结果不落显存，一次 dispatch。
//    - 归约表达式 / 归约输出 / matmul 等作为融合边界 → 独立 kernel。
//    - 拼接后 canonicalize + validate；超限（instrs/regs/inputs/consts）
//      或非法 → 放弃融合（保守，保证正确性优先）。
//
//  确定性铁律（与 expr_opt.hpp 一致）：贪心按节点序从前到后；拼接重映射
//  顺序固定；结果经 canonicalize_expr_spec 统一规范化（两端一致）。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "compute_tensor.hpp"
#include "precision.hpp"
#include "expr_spec.hpp"
#include "expr_opt.hpp"

namespace nn
{

// ── 占位标记发号器（跨图全局唯一）────────────────────────────────────────
// P2 实测坑：tag 若用节点局部编号（node+1），跨录制图会撞车——Adam 多张量
// 步进里第 2 个张量的 K1 带着**上一张量/上一步**的 tag=1，被新图误识别为
// 本图节点 0 的输出（GPU 锚位偏移、静默错值；CPU 无录制路径不受影响）。
// 全局唯一发号后，陈旧 tag 永不命中（node_by_tag 只登记本图的号）。
[[nodiscard]] inline std::uint64_t next_virtual_tag() noexcept
{
    static std::atomic<std::uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// ── 图 IR 节点 ────────────────────────────────────────────────────────────
struct ExprGraphNode
{
    ExprSpec    spec;          // canonical IR（录制时已 canonicalize）
    std::size_t rows = 0;      // 输出形状（统一网格 (rows, cols)）
    std::size_t cols = 0;
    bool        vector_out = false;  // 输出为归约向量（(rows,1)/(1,cols)）
    Precision   prec = Precision::F32;  // 求值精度（恢复时新增：融合成员须一致）
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
    // 节点下标 → 本图分配的唯一占位标记（调用方 stamp 到占位 Tensor；
    // 与 node_by_tag 同源，杜绝跨图撞车——见 next_virtual_tag）
    std::vector<std::uint64_t> tag_of_node;
    // 录制发起引擎（非拥有观察者；恢复时新增）：录制段内的表达式必须经
    // **同一**引擎求值——占位 buffer 是该引擎的存储，混引擎消费会静默读到
    // 未写入数据。NVI 录制拦截据此做跨引擎 fail-fast。
    ComputeEngine* owner = nullptr;

    // 录制：把一个表达式加入图，返回节点下标。
    // inputs 中带 virtual_tag 的占位 Tensor 会被识别为前序节点输出（依赖边）。
    [[nodiscard]] int add_node(const ExprSpec& raw_spec,
                               std::span<const Tensor> inputs,
                               std::size_t rows, std::size_t cols,
                               bool vector_out,
                               Precision prec = Precision::F32)
    {
        const ExprSpec spec = canonicalize_expr_spec(raw_spec);
        const int idx = static_cast<int>(nodes.size());
        ExprGraphNode nd;
        nd.spec        = spec;
        nd.rows        = rows;
        nd.cols        = cols;
        nd.vector_out  = vector_out;
        nd.prec        = prec;
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
        // 登记输出标记（全局唯一发号；0 保留给"非节点输出"）
        const std::uint64_t tag = next_virtual_tag();
        node_by_tag[tag] = idx;
        tag_of_node.push_back(tag);
        return idx;
    }
};

// ── 融合 kernel（融合分析输出） ──────────────────────────────────────────
// 一个 kernel = 一个（可能为多个节点拼接的）复合 ExprSpec + 外部输入来源。
struct FusedKernelInput
{
    int     node = -1;   // >= 0：该输入是图节点 node 的输出（作为外部 buffer 绑定）
    Tensor  external;    // node < 0：Layer 外部输入（运行时张量）
    // P2-12（图级缓存跨 step 复用）：node < 0 时记录该外部输入的来源，
    // 以便从缓存计划反查当前图对应节点的输入槽（input_tensors[ext_slot]）。
    int     ext_node = -1;  // 来源原图节点 index
    int     ext_slot = -1;  // 来源节点输入槽 index
};

struct FusedKernel
{
    ExprSpec    spec;             // 复合 spec（canonical）
    std::vector<FusedKernelInput> inputs;  // 与 spec.views 一一对应
    std::size_t rows = 0;
    std::size_t cols = 0;
    bool        vector_out = false;
    Precision   prec = Precision::F32;   // 成员共同精度（恢复时新增）
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

        // 1) 前置：均逐元素（无归约）、非归约输出、形状相同、精度相同
        if (cur.vector_out || B.vector_out) return false;
        if (cur.prec != B.prec) return false;   // 恢复时新增：混精度不拼
        // 1a) matmul 段（S5）：matmul 是"前置段"（唯一、位于链首）——
        //   - B 含 matmul → 不能拼入当前 kernel（matmul 必须位于链首）→ 新 kernel
        //   - A 含 matmul → 允许拼接 B：fused 保留 A.matmul，B 的指令拼在
        //     A 的尾链之后（A 纯 matmul 时 B 对 A 输出的引用替换为 Matmul 操作数）
        if (Bs.matmul) return false;
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
        // A 纯 matmul（空链，S5）时无输出寄存器：B 对 A 输出的引用替换为
        // Matmul 操作数（读取前置 matmul 段输出），否则替换为 A 的输出寄存器。
        const bool a_pure_matmul = A.matmul && A.instrs.empty();
        const std::size_t out_reg_A = a_pure_matmul ? 0 : A.instrs.back().dst;

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
        // ⚠ RParam 槽同理偏移拼接（漏掉 = 运行期 rp 槽错位/为 0，静默错值）
        std::vector<Scalar> rparams = A.rparams;
        const std::size_t b_rp_base = rparams.size();
        rparams.insert(rparams.end(), Bs.rparams.begin(), Bs.rparams.end());

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
                    {
                        if (a_pure_matmul)
                            return expr::matmul_op();  // 引用前置 matmul 段输出
                        return expr::reg(static_cast<std::uint8_t>(out_reg_A));
                    }
                    return expr::input(static_cast<std::uint8_t>(
                        b_view_map[k]));
                }
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
                    return expr::cst(static_cast<std::uint8_t>(
                        o.idx + b_const_base));
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::RParam))
                    return {o.kind, static_cast<std::uint8_t>(o.idx + b_rp_base)};
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
        fused.rparams = std::move(rparams);
        fused.instrs  = std::move(instrs);
        fused.num_regs = static_cast<std::uint32_t>(reg_base + Bs.num_regs);
        fused.matmul  = A.matmul;   // 保留前置 matmul 段（S5：matmul 参与图融合）
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
                    /*node=*/-1, /*external=*/B.input_tensors[kk],
                    /*ext_node=*/bi, /*ext_slot=*/static_cast<int>(kk)});
            else
                new_inputs.push_back(FusedKernelInput{
                    /*node=*/B.dep_of_input[kk], /*external=*/Tensor{}});
        }
        cur.spec       = std::move(fused);
        cur.inputs     = std::move(new_inputs);
        cur.rows       = B.rows;
        cur.cols       = B.cols;
        cur.prec       = B.prec;
        cur.tail       = bi;
        cur.members.push_back(bi);
        return true;
    };

    // ── P2 分量标记（写穿物化）──────────────────────────────────────────
    // 纯逐元素 F32 同形状同精度节点经**依赖边**连通成组（含多消费者/菱形）：
    // 组内全部成员物化（多输出 kernel，spec.extras = 除尾成员外的输出寄存器）
    // → 中间量逃逸/多消费者不再是融合边界。非 F32 / matmul / fold / 归约 /
    // 归约输出走 P1 链语义（单输出，中间量契约不变）。
    const auto p2_eligible = [&](std::size_t i) -> bool
    {
        const auto& N = g.nodes[i];
        return !N.vector_out && N.prec == Precision::F32 &&
               !N.spec.instrs.empty() && !N.spec.matmul && !N.spec.fold &&
               expr_spec_reduce_axis(N.spec) == -1;
    };
    std::vector<int> uf_parent(g.nodes.size());
    for (std::size_t i = 0; i < uf_parent.size(); ++i)
        uf_parent[i] = static_cast<int>(i);
    const auto uf_find = [&](int x) -> int
    {
        while (uf_parent[static_cast<std::size_t>(x)] != x)
        {
            uf_parent[static_cast<std::size_t>(x)] =
                uf_parent[static_cast<std::size_t>(
                    uf_parent[static_cast<std::size_t>(x)])];
            x = uf_parent[static_cast<std::size_t>(x)];
        }
        return x;
    };
    for (std::size_t i = 0; i < g.nodes.size(); ++i)
    {
        if (!p2_eligible(i))
            continue;
        for (const int d : g.nodes[i].dep_of_input)
        {
            if (d < 0 || !p2_eligible(static_cast<std::size_t>(d)))
                continue;
            const auto& A = g.nodes[static_cast<std::size_t>(d)];
            const auto& B = g.nodes[i];
            if (A.rows != B.rows || A.cols != B.cols || A.prec != B.prec)
                continue;
            const int ra = uf_find(d);
            const int rb = uf_find(static_cast<int>(i));
            if (ra != rb)
                uf_parent[static_cast<std::size_t>(std::max(ra, rb))] =
                    std::min(ra, rb);   // 小根为根：确定性
        }
    }

    // 每 kernel 的构造期簿记（成员序输出寄存器；P1 kernel 不使用）
    std::vector<std::vector<int>> k_members;
    std::vector<std::vector<std::uint8_t>> k_out_reg;
    std::vector<std::uint8_t> k_is_comp;
    std::unordered_map<int, std::size_t> comp_kernel;   // 分量根 → kernels 下标

    // 新 kernel 单成员起步（P1/分量共用的初始化）
    const auto open_kernel = [&](std::size_t i, bool comp) -> std::size_t
    {
        const auto& N = g.nodes[i];
        FusedKernel k;
        k.spec       = N.spec;
        k.rows       = N.rows;
        k.cols       = N.cols;
        k.vector_out = N.vector_out;
        k.prec       = N.prec;
        k.tail       = static_cast<int>(i);
        k.members.push_back(static_cast<int>(i));
        for (std::size_t kk = 0; kk < N.spec.views.size(); ++kk)
        {
            if (kk >= N.dep_of_input.size() || N.dep_of_input[kk] < 0)
                k.inputs.push_back(FusedKernelInput{
                    /*node=*/-1,
                    /*external=*/(kk < N.input_tensors.size()) ? N.input_tensors[kk]
                                                               : Tensor{},
                    /*ext_node=*/static_cast<int>(i), /*ext_slot=*/static_cast<int>(kk)});
            else
                k.inputs.push_back(FusedKernelInput{
                    /*node=*/N.dep_of_input[kk], /*external=*/Tensor{}});
        }
        kernels.push_back(std::move(k));
        k_members.push_back({static_cast<int>(i)});
        k_out_reg.push_back(N.spec.instrs.empty()
            ? std::vector<std::uint8_t>{}
            : std::vector<std::uint8_t>{N.spec.instrs.back().dst});
        k_is_comp.push_back(comp ? 1 : 0);
        return kernels.size() - 1;
    };

    // 分量成员并入（广义拼接）：成员对**任意已并入成员**的 Linear 引用替换成
    // 该成员输出寄存器（拼接编号下），其余槽追加为 kernel 输入。成员输出按
    // 成员序簿记，extras = 除尾成员外全部（写穿物化）。拼接**不做 canonicalize**
    // （寄存器号按拼接编号稳定）——registry 与运行期都按 canonical 键查表
    // （ExprRegistry::add / eval_expr_multi_into 内部 canonicalize，extras 随
    // canonicalize 重映射），键两端同源。先拷贝构造、预算预检、成功才提交。
    const auto comp_append = [&](std::size_t ki, std::size_t bi) -> bool
    {
        const auto& mem0 = k_members[ki];
        const auto& outs0 = k_out_reg[ki];
        const ExprGraphNode& B = g.nodes[bi];
        const ExprSpec& Bs = B.spec;
        if (Bs.instrs.empty() || Bs.matmul || Bs.fold)
            return false;
        const auto member_idx = [&](int dep) -> int
        {
            for (std::size_t m = 0; m < mem0.size(); ++m)
                if (mem0[m] == dep)
                    return static_cast<int>(m);
            return -1;
        };

        FusedKernel cur = kernels[ki];
        std::vector<int> mem = mem0;
        std::vector<std::uint8_t> outs = outs0;

        std::vector<ExprView> views = cur.spec.views;
        std::vector<FusedKernelInput> inputs = cur.inputs;
        std::vector<int> slot_map(Bs.views.size(), -2);   // -2=成员引用，≥0=新输入槽
        for (std::size_t k = 0; k < Bs.views.size(); ++k)
        {
            const int dep = (k < B.dep_of_input.size()) ? B.dep_of_input[k] : -1;
            const int m = (dep >= 0) ? member_idx(dep) : -1;
            if (m >= 0)
            {
                if (static_cast<ExprViewKind>(Bs.views[k].kind) != ExprViewKind::Linear)
                    return false;   // 仅 Linear 引用可寄存器直替
            }
            else
            {
                slot_map[k] = static_cast<int>(views.size());
                views.push_back(Bs.views[k]);
                if (dep < 0)
                    inputs.push_back(FusedKernelInput{
                        /*node=*/-1,
                        /*external=*/(k < B.input_tensors.size()) ? B.input_tensors[k]
                                                                 : Tensor{},
                        /*ext_node=*/static_cast<int>(bi),
                        /*ext_slot=*/static_cast<int>(k)});
                else
                    inputs.push_back(FusedKernelInput{
                        /*node=*/dep, /*external=*/Tensor{}});
            }
        }

        // 预算预检（保守：按拼接后原始计数；上限 = validate_expr_spec 同源）
        if (cur.spec.instrs.size() + Bs.instrs.size() > EXPR_MAX_INSTRS)
            return false;
        if (cur.spec.num_regs + Bs.num_regs > 255)
            return false;
        if (cur.spec.consts.size() + Bs.consts.size() > EXPR_MAX_CONSTS)
            return false;
        if (cur.spec.rparams.size() + Bs.rparams.size() > EXPR_MAX_CONSTS)
            return false;
        if (views.size() > EXPR_MAX_INPUTS)
            return false;

        std::vector<Scalar> consts = cur.spec.consts;
        const std::size_t b_const_base = consts.size();
        consts.insert(consts.end(), Bs.consts.begin(), Bs.consts.end());
        // ⚠ RParam 与 Const 同理：B 的槽必须偏移拼接（漏掉 = 运行期 rp 槽全 0，
        //   融合 kernel 静默用错超参——P2 实测把 GPU 字节锚打偏的真凶）
        std::vector<Scalar> rparams = cur.spec.rparams;
        const std::size_t b_rp_base = rparams.size();
        rparams.insert(rparams.end(), Bs.rparams.begin(), Bs.rparams.end());
        const std::size_t reg_base = cur.spec.num_regs;
        std::vector<ExprInstr> instrs = cur.spec.instrs;
        for (const auto& ins : Bs.instrs)
        {
            ExprInstr ni = ins;
            ni.dst = static_cast<std::uint8_t>(ni.dst + reg_base);
            const auto remap = [&](ExprOperand o) -> ExprOperand
            {
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Input))
                {
                    const std::size_t k = o.idx;
                    if (slot_map[k] == -2)
                    {
                        const int m = member_idx(B.dep_of_input[k]);
                        return expr::reg(outs[static_cast<std::size_t>(m)]);
                    }
                    return expr::input(static_cast<std::uint8_t>(slot_map[k]));
                }
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Const))
                    return expr::cst(static_cast<std::uint8_t>(
                        o.idx + b_const_base));
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::RParam))
                    return {o.kind, static_cast<std::uint8_t>(o.idx + b_rp_base)};
                if (o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reg) ||
                    o.kind == static_cast<std::uint8_t>(ExprOperandKind::Fanout) ||
                    o.kind == static_cast<std::uint8_t>(ExprOperandKind::Reduce))
                    return {o.kind, static_cast<std::uint8_t>(o.idx + reg_base)};
                return o;
            };
            const std::size_t nops =
                expr_instr_num_operands(static_cast<ExprOp>(ins.op));
            ni.a = remap(ins.a);
            if (nops >= 2) ni.b = remap(ins.b);
            if (nops >= 3) ni.c = remap(ins.c);
            instrs.push_back(ni);
        }

        ExprSpec fused;
        fused.views   = std::move(views);
        fused.consts  = std::move(consts);
        fused.rparams = std::move(rparams);
        fused.instrs  = std::move(instrs);
        fused.num_regs = static_cast<std::uint32_t>(reg_base + Bs.num_regs);
        outs.push_back(static_cast<std::uint8_t>(Bs.instrs.back().dst + reg_base));
        fused.extras.assign(outs.begin(), outs.end() - 1);
        if (auto v = validate_expr_spec(fused, fused.views.size()); !v)
            return false;   // 超限/非法 → 放弃并入（保守）

        cur.spec       = std::move(fused);
        cur.inputs     = std::move(inputs);
        cur.rows       = B.rows;
        cur.cols       = B.cols;
        cur.prec       = B.prec;
        cur.tail       = static_cast<int>(bi);
        cur.members.push_back(static_cast<int>(bi));
        mem.push_back(static_cast<int>(bi));   // 簿记同步（供后续成员的依赖查找）
        kernels[ki]    = std::move(cur);
        k_members[ki]  = std::move(mem);
        k_out_reg[ki]  = std::move(outs);
        return true;
    };

    // ── 主循环（双规则）────────────────────────────────────────────────
    // ① S5 例外：matmul 头链优先按 P1 链语义拼（逐元素尾链附在 matmul 段后，
    //    单输出——matmul 输出无寄存器可写穿）；② P2 分量（写穿全输出，支持
    //    多消费者/菱形合并与超预算分裂）；③ 其余（非 F32/matmul/fold/归约）
    //    按 P1 链语义。分量 kernel 与 P1 kernel 互不混拼。
    for (std::size_t i = 0; i < g.nodes.size(); ++i)
    {
        if (!kernels.empty() && kernels.back().spec.matmul &&
            !k_is_comp.back() && try_append(kernels.back(), static_cast<int>(i)))
        {
            k_members.back().push_back(static_cast<int>(i));
            k_out_reg.back().clear();   // P1 单输出（不写穿）
            continue;
        }

        if (p2_eligible(i))
        {
            const int root = uf_find(static_cast<int>(i));
            const auto it = comp_kernel.find(root);
            if (it == comp_kernel.end())
            {
                comp_kernel.emplace(root, open_kernel(i, /*comp=*/true));
                continue;
            }
            if (comp_append(it->second, i))
                continue;
            // 预算不足：分裂——同分量余下成员进新 kernel（跨 kernel 依赖经
            // 写穿输出读取），映射顶替为新 kernel
            it->second = open_kernel(i, /*comp=*/true);
            continue;
        }

        if (!kernels.empty() && !k_is_comp.back() &&
            try_append(kernels.back(), static_cast<int>(i)))
        {
            k_members.back().push_back(static_cast<int>(i));
            k_out_reg.back().clear();   // P1 单输出（不写穿）
            continue;
        }
        (void)open_kernel(i, /*comp=*/false);
    }
    return kernels;
}

// ═══════════════════════════════════════════════════════════════════════════
//  P2-12 图级缓存跨 step 复用
//
//  训练每 step 的表达式结构高度重复（Layer 结构固定），但 `fuse_expr_graph`
//  每 step 都重跑：融合分析 + 拼接 canonicalize + validate。这属于训练热路径
//  上的纯 CPU 开销。为消除它，这里缓存"整图结构 key → 融合 kernel 计划"：
//    - key 覆盖所有影响融合决策的因素（节点 canonical key + 形状 + 拓扑），
//      同一结构必然产出同一 kernel 计划（确定性铁律 8 保证）。
//    - 计划只存结构（spec/shape/tail/members/输入来源），不存运行时张量；
//      命中后由 instantiate_plan 绑定当前图的 node_outputs 与外部输入。
//  不改变任何融合决策或计算，故不引入数值差异，不影响逐字节确定性。
// ═══════════════════════════════════════════════════════════════════════════

// 缓存的 kernel 计划（不含运行时张量）
struct FusedKernelPlan
{
    ExprSpec           spec;             // 复合 spec（canonical）
    std::vector<int>   input_node;       // 每输入槽：来源节点 idx（>=0）或 -1（外部）
    std::vector<int>   input_ext_node;   // 外部时：来源原图节点 idx
    std::vector<int>   input_ext_slot;   // 外部时：来源节点输入槽 index
    std::size_t        rows = 0, cols = 0;
    bool               vector_out = false;
    Precision          prec = Precision::F32;
    int                tail = -1;
    std::vector<int>   members;
};

// 整图结构 key（FNV-1a）：覆盖节点 spec key、形状、vector_out、精度、依赖拓扑。
[[nodiscard]] inline std::uint64_t graph_cache_key(const ExprGraph& g)
{
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    std::uint64_t h = kFnvOffset;
    const auto mix_u64 = [&](std::uint64_t v)
    {
        for (int b = 0; b < 8; ++b)
        {
            h ^= static_cast<std::uint8_t>(v >> (8 * b));
            h *= kFnvPrime;
        }
    };
    const auto mix_str = [&](const std::string& s)
    {
        for (const unsigned char c : s)
        {
            h ^= static_cast<std::uint8_t>(c);
            h *= kFnvPrime;
        }
    };
    mix_u64(static_cast<std::uint64_t>(g.nodes.size()));
    for (const auto& n : g.nodes)
    {
        mix_u64(static_cast<std::uint64_t>(n.rows));
        mix_u64(static_cast<std::uint64_t>(n.cols));
        mix_u64(n.vector_out ? 1ull : 0ull);
        mix_u64(static_cast<std::uint64_t>(n.prec));
        mix_str(expr_spec_key(n.spec));
        mix_u64(static_cast<std::uint64_t>(n.dep_of_input.size()));
        for (const int d : n.dep_of_input)
            mix_u64(static_cast<std::uint64_t>(d));
    }
    return h;
}

// 从 fused kernel 提取计划（不含运行时张量）。
[[nodiscard]] inline FusedKernelPlan plan_from_kernel(const FusedKernel& k)
{
    FusedKernelPlan p;
    p.spec        = k.spec;
    p.rows        = k.rows;
    p.cols        = k.cols;
    p.vector_out  = k.vector_out;
    p.prec        = k.prec;
    p.tail        = k.tail;
    p.members     = k.members;
    p.input_node.reserve(k.inputs.size());
    p.input_ext_node.reserve(k.inputs.size());
    p.input_ext_slot.reserve(k.inputs.size());
    for (const auto& in : k.inputs)
    {
        p.input_node.push_back(in.node);
        p.input_ext_node.push_back(in.ext_node);
        p.input_ext_slot.push_back(in.ext_slot);
    }
    return p;
}

// 从计划实例化 kernel，绑定当前图的运行时张量（node 输出 / 外部输入）。
[[nodiscard]] inline FusedKernel instantiate_plan(const FusedKernelPlan& p, const ExprGraph& g)
{
    FusedKernel k;
    k.spec        = p.spec;
    k.rows        = p.rows;
    k.cols        = p.cols;
    k.vector_out  = p.vector_out;
    k.prec        = p.prec;
    k.tail        = p.tail;
    k.members     = p.members;
    k.inputs.reserve(p.input_node.size());
    for (std::size_t i = 0; i < p.input_node.size(); ++i)
    {
        FusedKernelInput in;
        in.node = p.input_node[i];
        if (in.node < 0)
        {
            in.ext_node = p.input_ext_node[i];
            in.ext_slot = p.input_ext_slot[i];
            in.external = (in.ext_node >= 0 && in.ext_slot >= 0)
                ? g.nodes[static_cast<std::size_t>(in.ext_node)]
                      .input_tensors[static_cast<std::size_t>(in.ext_slot)]
                : Tensor{};
        }
        k.inputs.push_back(std::move(in));
    }
    // ⚠ rparams 是**每步变化的运行时值**（Adam 的 inv_bc1/inv_bc2 等）：
    // 计划缓存只复用结构，值必须按成员序从当前图节点重拼（拼接规则 =
    // 成员序各池顺序追加，与 fuse 侧 concat 同源；canonicalize 对 rparams
    // 原样透传、顺序稳定）。否则缓存命中的 kernel 永远用第 1 步的超参
    // （P2 实测：GPU 字节锚偏移的真凶）。consts 是结构字面量、不随步变，
    // 随计划缓存安全。
    {
        std::vector<Scalar> rp;
        for (const int m : k.members)
        {
            const auto& nd = g.nodes[static_cast<std::size_t>(m)];
            rp.insert(rp.end(), nd.spec.rparams.begin(), nd.spec.rparams.end());
        }
        k.spec.rparams = std::move(rp);
    }
    return k;
}

// ── 图级缓存容器（thread_local，与录制图同线程隔离，D3 一致） ──────────
// 容量受限：结构种类（每 Layer 一种图结构）通常很少，但防御性地设上限，
// 超限清空（等价于重建缓存，不影响正确性）。
namespace fused
{

[[nodiscard]] inline std::unordered_map<std::uint64_t,
                                        std::vector<FusedKernelPlan>>&
graph_plan_cache()
{
    static thread_local std::unordered_map<std::uint64_t,
                                           std::vector<FusedKernelPlan>> c;
    return c;
}
inline constexpr std::size_t GRAPH_PLAN_CACHE_MAX = 512;

} // namespace fused

} // namespace nn
