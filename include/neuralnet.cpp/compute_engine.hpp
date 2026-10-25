#pragma once

// ── compute_engine.hpp — 计算引擎抽象接口 ─────────────────────────────────
// ComputeEngine 是与底层硬件接触的唯一抽象层。
//
// 设计原则（铁律）：
//   1. 本接口只提供 op-level 原语（矩阵乘法、加法、转置、归约、广播、
//      逐元素运算等），绝不包含任何算法。
//   2. ReLU、GeLU、LayerNorm、Softmax、Attention 等算法由 Layer 层
//      通过组合原语表达。
//   3. Layer 持有 ComputeEngine 引用，forward/backward 只写一次，
//      CPU/GPU 由引擎实现自动分发。
//
// 原语分类：
//   - 矩阵级：matmul, batched_matmul, transpose, add_inplace, scale_inplace, zero
//   - 归约级：row_reduce_sum, col_reduce_sum
//   - 分组归约：grouped_reduce_sum, grouped_reduce_max
//   - 条件选择：由表达式 DSL 的 select 承担（engine 不暴露该原语）
//
// 批处理控制：
//   - begin_batch / end_batch：CPU 引擎为 no-op；GPU 引擎录制到
//     command buffer，end_batch 时统一提交。
//
// 多精度（docs/development/05-mixed-precision.md §8）：
//   - **运算类**原语（归约 / 扫描 / eval_expr）带显式形参
//     `Precision P = Precision::F32`：P = 该算子输出（及计算）精度。
//   - **纯数据搬运**原语（transpose / slice_rows / insert_rows / gather_rows /
//     scatter_add_rows / rearrange_3d / im2col / col2im / clone）**不带 P**：
//     输出精度 = 源精度（§8.4 的自然语义）。
//   - **in-place** 原语（add_inplace / scale_inplace / zero / eval_expr_into）**
//     存储精度不可变**（§8.3）。
//   - **精度策略住在各引擎的 `*_impl` 里**：基类公共入口只做 bind 检查 +
//     设备域检查后**直通** `*_impl`，**不做任何能力分支**；"原生
//     f16 直读直写"还是"抬 f32 算 → 按 P 落回"由引擎自己决定（CPU 引擎基于
//     f32 Matrix → 边界 cast；GPU 引擎有 f16 变体/字节级原语 → 原生）。
//     基类只保留**通用 cast 助手**（`to_f32` / `to_prec` / `move_` /
//     `unary_` / `binary_` / `inplace*_`）供各引擎的 `*_impl` 复用。
//   - **能力查询不是接口的一部分**：引擎能力是各实现类的私有细节；对外只经
//     `describe()` 返回的 `EngineInfo` 披露**事实**（禁止用于选路）。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>   // RV-02：create_tensor 的形状溢出检查
#include <memory>
#include <random>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <neuralnet.cpp/core_config.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_assert.hpp>
#include "core_env.hpp"       // RV-09：环境变量读取唯一入口
#include "core_domain.hpp"
#include "core_profile.hpp"   // 训练 step 剖析（NN_PROFILE=1 门控；默认零开销）
#include "compute_tensor.hpp"
#include "expr_opt.hpp"   // canonicalize_expr_spec（trace 归因用，与 gen_fused 同源 key）
#include "expr_graph.hpp" // IR-C 录制图（begin_expr/end_expr 图融合）
#include "expr_spec.hpp"

namespace nn
{

namespace dsl
{
    // 前置声明（compute_engine.hpp 不能反向 include expr_dsl.hpp）：
    // 默认实参（P = F32）在 expr_dsl.hpp 的定义处。
    // 生命周期 v3 起不再有库内 stamp/adopt 通道：DSL 静态工厂产出的 CPU
    // 张量天然属于宿主域，设备域从存储派生。
    template <typename E>
    [[nodiscard]] Result<Tensor> compute(ComputeEngine& eng, const E& e,
                                         std::size_t rows, std::size_t cols,
                                         Precision P = Precision::F32);
    template <typename E>
    [[nodiscard]] Result<Tensor> compute_reduce(ComputeEngine& eng, const E& e,
                                                std::size_t rows, std::size_t cols,
                                                Precision P = Precision::F32);
} // namespace nn::dsl

// ══════════════════════════════════════════════════════════════════════════
// 算子枚举（op-level，不含算法语义）
// ══════════════════════════════════════════════════════════════════════════

// 归约算子（matmul 融合原语用，op-level 无算法语义）
enum class ReduceOp : uint32_t
{
    Sum = 0,
    Max = 1,
    Min = 2,
};

// ── 多精度变体签名（AOT in-kernel f16 索引；类型见 expr_spec.hpp）────────
// 由**实际张量精度** + 目标输出精度算出：位 i = 第 i 个输入是 f16，
// bit16 = 输出是 f16。全 0 = 全 f32：不追加变体后缀，与基础结构共用同一条目
// （融合算术保持 f32）。运行时用它选 (结构 key, 签名) 对应的带类型 shader；
// 构建期 scan/gen 两端同源。
[[nodiscard]] inline ExprPrecSig expr_prec_sig_of(
    std::span<const Tensor> inputs, Precision P = Precision::F32) noexcept
{
    std::uint32_t bits = 0;
    const std::size_t n = inputs.size() < 32u ? inputs.size() : 32u;
    for (std::size_t i = 0; i < n; ++i)
        if (inputs[i].precision() == Precision::F16)
            bits |= (1u << i);
    return expr_prec_sig_make(bits, P == Precision::F16);
}

// ── InitSpec — 声明式张量初始化（M2，docs/todo/17 §4.4）──────────────
// 分工（铁律 #3）：**层算分布参数**（limit / mean / stddev 这类层语义留在
// Layer），**引擎负责填数**（只管把 shape×precision×分布的数填出来）。
// 初始化策略（host 生成后上传 vs 设备端原生生成）由引擎自选，调用方不可见
// （D6）。seed 必填（17 §8 U1 裁定：层传显式默认值，漏传 = 编译错误）——
// 引擎内部把 seed 与本次创建的序号混流，同 seed 的不同张量不撞流（同形状
// 多层不互为镜像）；创建顺序 = 模型构造顺序 → 跨进程初值逐字节确定。
struct InitSpec
{
    enum class Kind
    {
        Uninitialized, // 不写初值（契约沿用 cpu_uninitialized：读前必须全写满）
        Zero,          // 全零
        Constant,      // 全 value
        Uniform,       // U[lo, hi]
        Normal,        // N(mean, stddev)
    };

    Kind kind = Kind::Zero;
    Scalar value = Scalar{0};                      // Constant
    Scalar lo = Scalar{0}, hi = Scalar{0};         // Uniform
    Scalar mean = Scalar{0}, stddev = Scalar{0};   // Normal
    std::uint64_t seed = 42; // 分布种子（Uninitialized/Zero/Constant 不消费）

    [[nodiscard]] static InitSpec uninitialized()
    {
        InitSpec s;
        s.kind = Kind::Uninitialized;
        return s;
    }
    [[nodiscard]] static InitSpec zero()
    {
        InitSpec s;
        s.kind = Kind::Zero;
        return s;
    }
    [[nodiscard]] static InitSpec constant(Scalar v)
    {
        InitSpec s;
        s.kind = Kind::Constant;
        s.value = v;
        return s;
    }
    // 分布类 seed 必填（U1）
    [[nodiscard]] static InitSpec uniform(Scalar lo, Scalar hi, std::uint64_t seed)
    {
        InitSpec s;
        s.kind = Kind::Uniform;
        s.lo = lo;
        s.hi = hi;
        s.seed = seed;
        return s;
    }
    [[nodiscard]] static InitSpec normal(Scalar mean, Scalar stddev, std::uint64_t seed)
    {
        InitSpec s;
        s.kind = Kind::Normal;
        s.mean = mean;
        s.stddev = stddev;
        s.seed = seed;
        return s;
    }
};

// 层传的显式默认 seed（17 §8 U1：seed 必填、层传显式默认值）。
// 同 seed 不同张量由引擎按创建序号混流，不撞流。
inline constexpr std::uint64_t kInitSeed = 42;

// ══════════════════════════════════════════════════════════════════════════
// EngineInfo — 引擎身份 / 能力摘要（唯一对外查询通道）
//
// 一句话：能力**信息**（CLI / 探针 / 测试 / 日志要看的那一份）只能经这里
// 披露；它**只陈述事实**，**禁止用于选路** —— 选路必须落在各引擎内部
// （见 AGENTS 铁律「引擎能力体系」）。字段描述的是 init 期已解析完成的
// 设备事实，读取无副作用、无计算。
// ⚠ 打印/记录（`text()`）**只用于调试**：它是给人看的诊断快照，**不是**用来
//   在运行时判断"该不该降级 / 该走哪条路径"的开关。按能力值分支 = 违反
//   铁律 #13（选路走引擎内部谓词）；"打印出来就据此改行为"是错误用法。
// ══════════════════════════════════════════════════════════════════════════
struct EngineInfo
{
    std::string engine;             // 引擎类别："cpu" / "vulkan"
    std::string device;             // 设备名（CPU 引擎 = "cpu"）
    bool native_f16_move = false;   // 非 f32 数据搬运/原地原语可直读直写
    bool native_f16_reduce = false; // 非 f32 归约原语可直读输入
    bool fused_prec_variant = false; // 具备带类型（f16）融合 shader 机制
    bool ternary_gemm = false;      // 三值打包 GEMM（trit_pack / matmul_q）
    bool int8_dot = false;          // 设备侧 4×int8 打包点积（DP4A 语义）
    bool int8_dot_accelerated = false;  // 上者是硬件指令（非驱动软件展开）

    // 稳定、可机读的一行式摘要（字段顺序固定；供日志/审计正则匹配）
    [[nodiscard]] std::string text() const
    {
        std::string s = "engine=" + engine + " device=" + device;
        const auto bit = [&s](const char* name, bool v)
        {
            s += ' ';
            s += name;
            s += (v ? "=1" : "=0");
        };
        bit("native_f16_move", native_f16_move);
        bit("native_f16_reduce", native_f16_reduce);
        bit("fused_prec_variant", fused_prec_variant);
        bit("ternary_gemm", ternary_gemm);
        bit("int8_dot", int8_dot);
        bit("int8_dot_accelerated", int8_dot_accelerated);
        return s;
    }
};

// ══════════════════════════════════════════════════════════════════════════
// ComputeEngine — 计算引擎抽象接口
// ══════════════════════════════════════════════════════════════════════════
//
// 分层（NVI：Non-Virtual Interface）：
//   public  非虚入口 = 薄适配层：设备域检查 + 参数检查 → 直通 `*_impl`。
//           **不做能力分支**（精度策略下沉到各引擎的 `*_impl`）。
//   protected 虚 `*_impl` = 引擎实现：既管算，也管自己的精度边界策略。
//   能力：各引擎私有事实（`*_` 私有成员 / 编译期常量）；对外只经 `describe()`。
class ComputeEngine : public std::enable_shared_from_this<ComputeEngine>
{
public:
    // ── 生命周期与设备域（docs/todo/28-lifecycle-v3.md）──────────────────
    // 引擎必须由 shared_ptr 管理（CpuEngine::create / GpuEngine::create）。
    // 会解引用引擎的 Layer / Model / Optimizer / ExprGraph / ExprSegment 保存
    // keepalive()；Tensor 不保存引擎身份——设备域从**存储**派生。
    virtual ~ComputeEngine() = default;

    ComputeEngine(const ComputeEngine&) = delete;
    ComputeEngine& operator=(const ComputeEngine&) = delete;

    // 本引擎的设备域：CpuEngine = kHostDomain；GpuEngine = backend 标识。
    // 每个入口逐个操作数比较 `Tensor::device_domain_()` 与本值。
    [[nodiscard]] DomainId domain() const noexcept { return domain_; }

    // 取得本引擎的强引用。未由 shared_ptr 管理时返回空指针（调用方必须
    // 把它当错误处理，不能继续解引用 this 之外的任何生命周期假设）。
    [[nodiscard]] std::shared_ptr<ComputeEngine> keepalive() noexcept
    {
        return weak_from_this().lock();
    }

protected:
    explicit ComputeEngine(DomainId domain) : domain_(domain) {}

public:

    // ── 设备查询 ──────────────────────────────────────────────────────────
    [[nodiscard]] virtual Device device() const noexcept = 0;

    // ── 引擎身份 / 能力摘要（唯一对外查询通道）───────────────────────────
    // 只披露事实；**禁止**据此选路（选路在各引擎内部）。默认 = 未知引擎。
    [[nodiscard]] virtual EngineInfo describe() const { return EngineInfo{}; }

    // ── 批处理控制 ────────────────────────────────────────────────────────
    // CPU 引擎：no-op（操作立即同步执行）
    // GPU 引擎：begin 开始录制，end 统一提交 + fence wait
    [[nodiscard]] virtual Result<void> begin_batch() = 0;
    [[nodiscard]] virtual Result<void> end_batch() = 0;

    // ── 表达式录制（IR-C 图融合；见 docs/history/history.md
    //    「IR-C 定位修正与恢复立项」）──────────────────────────────────────
    // begin_expr 进入录制；期间 eval_expr/eval_expr_reduce 把节点加入录制图
    // （expr_graph.hpp）并返回携带 virtual_tag 的占位 Tensor；end_expr 融合
    // 分析（fuse_expr_graph：纯逐元素单消费者链拼接 + S5 matmul 前置段），
    // 复合 kernel 直接写回各 tail 占位存储（中间量内联为寄存器）。
    //   · CPU 引擎：普通运行 no-op（表达式照常逐个求值）；NN_EXPR_SCAN 构建期
    //     开启录制并在 end_expr 登记复合 spec（闭合世界两端一致）。
    //   · GPU 引擎：录制 + 融合执行；复合 spec 未命中 AOT shader 硬报错。
    // 契约（P1）：录制段内中间量**不得逃逸**（不落 backward 缓存 / 状态张量、
    // 段外不读中间值）；段内表达式一律经开启录制的引擎求值（跨引擎 fail-fast）。
    // P2「写穿物化/多输出」将解除逃逸限制。
    [[nodiscard]] virtual Result<void> begin_expr() = 0;
    [[nodiscard]] virtual Result<void> end_expr() = 0;

    // ── 批处理中点刷新（防 TDR）────────────────────────────────────────────
    // GPU 引擎：提交当前 command buffer 并等待完成，然后自动开始新的录制。
    // 可在 forward 与 backward 之间调用，将一次大提交拆分为多次小提交，
    // 避免单次提交时间过长触发 Windows TDR。
    // CPU 引擎：no-op。
    [[nodiscard]] virtual Result<void> flush_batch() { return {}; }

    // ── 显存回收（L2）─────────────────────────────────────────────────
    // GPU 引擎：在 end_batch（提交完成、延迟销毁已 flush）之后归还完全
    // 空闲的内存池底材给 GPU。CPU 引擎：**语义成立的 no-op**——直配模式
    // 没有可归还的整块（17 §4.5 D5），文档写明而非假装有池。
    [[nodiscard]] virtual Result<void> release_idle_pool_blocks() { return {}; }

    // ── 内存统计（L2 仪器化）──────────────────────────────────────
    // GPU 引擎：池统计字符串（persist{…} transient{…} pending=…，见 GpuEngine）。
    // **无池引擎（CPU）**：如实上报宿主直配账本 `nn::host_alloc_ledger()`
    // （17 §4.5 D5"直配模式统计"）——活分配数/字节 + 峰值 + 累计，单位字节。
    // 两引擎同一动词、都非空，GUI/训练日志一套代码读两个引擎。
    // 用于训练中内存采样与逐项归因（`text_train` 的 NN_MEM_STATS 打点）。
    [[nodiscard]] virtual std::string pool_stats() const
    {
        const auto& led = host_alloc_ledger();
        return "direct{ blocks=" +
               std::to_string(led.live_blocks.load(std::memory_order_relaxed)) +
               " live_bytes=" +
               std::to_string(led.live_bytes.load(std::memory_order_relaxed)) +
               " peak_bytes=" +
               std::to_string(led.peak_bytes.load(std::memory_order_relaxed)) +
               " total_blocks=" +
               std::to_string(led.total_blocks.load(std::memory_order_relaxed)) +
               " total_bytes=" +
               std::to_string(led.total_bytes.load(std::memory_order_relaxed)) + " }";
    }

    // ── 激活 offload（L1-offload）───────────────────────────────────
    // ── activation offload slab（L1-offload，持久复用缓冲） ────────────
    // 每个 GPTBlock / RAPTBlock 持有一块持久 host-visible slab，所有激活按
    // float 偏移写入/读出，跨 step 复用 → RAM = 激活实际体积（避免每 tensor
    // 独立 128MB 块导致的碎片膨胀）。CPU 引擎 no-op（开启只会得到 1×1 张量）。
    [[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes)
    {
        NN_PROF_OP("create_offload_buffer");
        return create_offload_buffer_impl(bytes);
    }
    // 把 src 复制到 buffer 的 offset（float 单位）处
    // ── 边界 cast 入口：f16 激活在写入 slab 前抬到 f32（slab 恒 f32 存）──
    // restore 出来是 f32，后续运算由入口层适配（正确性不受影响，代价是该份
    // 激活按 f32 存、不享 f16 存储折半）。
    [[nodiscard]] Result<void> offload_save(
        const Tensor& buffer, std::size_t offset, const Tensor& src)
    {
        NN_PROF_OP("offload_save");
        NN_TRY(ec, bind_check_({&buffer, &src}));
        auto s = to_f32(src);
        NN_TRY_CHECK(s);
        return offload_save_impl(buffer, offset, *s);
    }
    // 从 buffer 的 offset（float 单位）处复制 rows×cols 到新 GPU tensor
    [[nodiscard]] Result<Tensor> offload_restore(
        const Tensor& buffer, std::size_t offset,
        std::size_t rows, std::size_t cols)
    {
        NN_PROF_OP("offload_restore");
        NN_TRY(ec, bind_check_({&buffer}));
        return offload_restore_impl(buffer, offset, rows, cols);
    }

    // ── 异步标量回读（非阻塞取 loss）──────────────────────────────
    // 动机：若用 to_matrix 取每步 loss，GPU 引擎会 end_batch + wait_in_flight
    // （等全部在飞帧）→ 每 step 一次全流水线 drain，host/GPU 无法重叠。
    //
    // submit_scalar_readback(slot, t)：把 t 的头 4 字节排入一次 D2H 拷贝
    //   并提交，**不等待**（源统一抬到 f32——GPU 只排 f32 拷贝）。
    //   ⚠ 调用约定：必须在产出 t 的主帧已提交之后调用——同一队列 FIFO 保证
    //   拷贝执行在生产者之后（在主帧提交前提交会读到上一轮旧值）。
    //   ⚠ t 的宿主 Tensor 必须存活到 poll 返回就绪（buffer 生命周期跨越提交）。
    // poll_scalar_readback(slot, out)：非阻塞查询。就绪写值并返回 true；
    //   未就绪返回 false（调用方稍后重试）。
    // scalar_readback_slots()：可用槽位数（调用方据此做环形复用）。
    [[nodiscard]] Result<void> submit_scalar_readback(std::size_t slot, const Tensor& t)
    {
        NN_PROF_OP("submit_scalar_readback");
        NN_TRY(ec, bind_check_({&t}));
        auto s = to_f32(t);
        NN_TRY_CHECK(s);
        return submit_scalar_readback_impl(slot, *s);
    }

    [[nodiscard]] virtual Result<bool> poll_scalar_readback(
        std::size_t slot, Scalar& out)
    {
        NN_PROF_OP("poll_scalar_readback");
        if (slot >= sync_readback_slots_.size())
            return false;
        out = sync_readback_slots_[slot];
        return true;
    }

    [[nodiscard]] virtual std::size_t scalar_readback_slots() const { return 1; }

    // ── 张量工厂（统一接口，§6.4, §6.5）────────────────────────────────
    // P 由调用方显式指定（§8.5）：无隐式推导，无 Auto
    // 公共非虚入口在尾部统一做设备域检查（v3，docs/todo/28 §4.3），
    // 引擎只实现 create_tensor_impl / from_matrix_impl（protected）——避免
    // 每个 override 各自漏 stamp。to_matrix 输出 Matrix（宿主），不参与 stamp。
    // RV-02（工厂错误协议收敛）：错误**结构化**上抛，不再"空张量 + stderr"。
    //   · 非"存储精度"（BF16/F64 保留值、T1_58 三值声明）→ 拒绝建张量
    //     （docs 21 §4.1 的落地保证：T1_58 只是层的精度声明，绝不能拿它当
    //      存储标签）；
    //   · 形状非法（rows*cols 溢出 size_t）→ 拒绝；
    //   · 分配失败 → 把实现侧原因（GpuTensor::create_empty 的设备/显存错误）
    //     原样带上（旧实现连 stderr 都不打印就返回空 Tensor）。
    // 旧签名 `create_tensor` 保留为 Result 版 + NN_UNWRAP 解包（失败即终止），
    // 库内 100+ 调用点无需一次性迁移（分期走 try_create_tensor）。
    [[nodiscard]] Result<Tensor> try_create_tensor(std::size_t rows, std::size_t cols,
                                                   Precision P = Precision::F32)
    {
        NN_PROF_OP("create_tensor");
        NN_TRY(pc, check_precision_supported(P));
        NN_TRY(shape_ok, check_shape_(rows, cols));
        return create_tensor_impl(rows, cols, P);
    }
    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols,
                                       Precision P = Precision::F32)
    {
        return NN_UNWRAP(try_create_tensor(rows, cols, P), "engine.create_tensor");
    }
    // ── 声明式创建/初始化（M2，17 §4.3/§4.4）────────────────────────────
    // InitSpec 指定初值；策略 = 引擎 host 生成后上传（D6：两设备同序列、
    // 调用方不可见；将来设备端原生生成是引擎内部优化，不改本签名）。
    // 失败返回 Result 错误（与 3 参 try_create_tensor 同协议）。
    // 语义分工：3 参 = 纯分配（沿用现状：CPU Matrix 零填充 / GPU 分配未初始
    // 化，需要零的调用方显式 zero）；"要什么初值"一律走本重载显式声明。
    [[nodiscard]] Result<Tensor> try_create_tensor(std::size_t rows, std::size_t cols,
                                                   Precision P, const InitSpec& spec)
    {
        NN_PROF_OP("create_tensor");
        NN_TRY(pc, check_precision_supported(P));
        NN_TRY(shape_ok, check_shape_(rows, cols));
        switch (spec.kind)
        {
        case InitSpec::Kind::Uninitialized:
            return create_tensor_impl(rows, cols, P);
        case InitSpec::Kind::Zero:
        {
            // 分配 + 整张清零（GPU 分配未初始化，zero() 是必要写；CPU 分配
            // 已零填充仍走一遍 zero() 统一契约）。
            NN_TRY(t_r, create_tensor_impl(rows, cols, P));
            NN_TRY(z, zero(*t_r));
            return std::move(*t_r);
        }
        default:
            break;
        }
        // Constant / Uniform / Normal：host 生成 + 上传（from_matrix 内部
        // f16 由 from_matrix 做 RHE 舍入，与层手填路径一致）。
        Matrix m(rows, cols, Matrix::uninitialized_tag{});
        fill_host_(m, spec, init_seq_++);
        return from_matrix(m, P);
    }
    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols, Precision P,
                                       const InitSpec& spec)
    {
        return NN_UNWRAP(try_create_tensor(rows, cols, P, spec),
                        "engine.create_tensor(spec)");
    }
    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m, Precision P = Precision::F32)
    {
        NN_PROF_OP("from_matrix");
        // 同上：非存储精度不得作为张量的存储标签（返回 Result，故直接上抛原因）
        NN_TRY(pc, check_precision_supported(P));
        return from_matrix_impl(m, P);
    }
    [[nodiscard]] virtual Result<Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) = 0;

    // ── 跨设备拉取（M6，17 §4.3 / 15 §3.2 P3；v3 起是唯一跨域入口）────────
    // 目标侧拉取语义（15 §3.2）：
    //   同域 + 同精度 = 零拷贝别名；同域 + 异精度 = 引擎内 cast；
    //   CPU → GPU = 上载（GpuEngine 原生上传路径，省一次宿主拷贝）；
    //   GPU → CPU **不走这里**（用 to_matrix 落宿主）；GPU 异 backend = 报错。
    // 返回**新句柄**，绝不改写调用方的 `src`——需要就地重绑定必须在调用点
    // 显式写 `t = std::move(*import(t))`（与 `cast_into` 的不替换红线同族）。
    [[nodiscard]] Result<Tensor> import(
        const Tensor& src,
        std::source_location loc = std::source_location::current())
    {
        NN_PROF_OP("import");
        return import(src, src.precision(), loc);
    }

    [[nodiscard]] Result<Tensor> import(
        const Tensor& src, Precision P,
        std::source_location loc = std::source_location::current())
    {
        NN_PROF_OP("import");
        // 热路径快路：同一设备域且精度一致时，import_impl 只会原样返回共享
        // 句柄。optimizer 每 step 会对每个 DSL leaf 反复走此入口（profile 约
        // 3.4 万次/20 steps）；在这里提前返回可跳过下方检查与 virtual dispatch。
        // 域从存储派生（只读、无原子操作）。
        if (src.valid() && src.precision() == P && src.device_domain_() == domain_)
            return src;
        if (!src.valid())
            NN_FAIL("import: invalid tensor");
        const DomainId d = src.device_domain_();
        // 唯一允许的跨域方向：宿主域 → 本引擎（GPU 上载）。CPU 引擎收到 GPU
        // 张量、或 GPU 引擎收到另一个 backend 的张量，一律明确报错。
        if (d != domain_ && d != kHostDomain)
            NN_FAIL(domain_error_(src, loc));
        return import_impl(src, P);
    }

    // ══════════════════════════════════════════════════════════════════════
    // I/O 分组（M3，docs/todo/17 §4.3 D3/D9 + §3 D11）：
    //   from_matrix / to_matrix / import / read / write / get_index / set_index
    // 宿主数据进出张量只准经本组动词；D9：from_matrix/to_matrix 保留原名
    // （Matrix = 宿主 I/O 载体），批量 read/write 为**本体**，索引级
    // get_index/set_index 是语法糖，`import` 是跨设备/引擎拉取入口（M6，
    // 原 `ensure_gpu` 迁入并升为引擎契约）。
    // 分层口径（M4 D11 / 铁律 #12）：库外（tests/tools/CLI）只准出现本组动词；
    // 计算路径（Layer/Loss/Optimizer/Model）禁用 **Matrix 型三动词**
    //（from_matrix/to_matrix/copy_from），span 级 read/write 与
    // `detail::upload_span/download_span` 宿主桥受控可用；序列化编解码层除外。
    //
    // 调用约定（17 §7-1/§7-4，铁律 #6）：
    //   - read 在 GPU 上**隐含 flush + 同步**（与 to_matrix 同路）：录制窗口
    //     内调用会 end_batch 提交 → wait_in_flight → 开新帧。逐 step loss
    //     回读继续走 submit_scalar_readback 快路，不被 read 取代。
    //   - write 覆盖 t 的**既有存储、不替换对象**（沿 copy_into 红线）；GPU
    //     录制窗口内走 copy_from 的既有 drain 语义。写入的张量必须存活到
    //     end_batch() 之后（铁律 #6）。
    //   - U2 元素类型：span<T> 的 T 必须与 t.precision() 精确匹配（float↔F32、
    //     f16↔F16），错配 = 运行期错误，类型本身非法 = 编译期 static_assert
    //     ——防 f16/f32 槽错位。f16/f32 换算由引擎边界做（RHE），不经调用方。
    //   - 批量是本体，index API 不给热循环承诺（GPU 上每次调用 = 一整轮
    //     staging 往返；热路径请攒批量）。
    // ══════════════════════════════════════════════════════════════════════

    // 批量读：张量全量 → dst（行主序），GPU 隐含 flush + 同步。
    template <class T>
    [[nodiscard]] Result<void> read(const Tensor& t, std::span<T> dst)
    {
        NN_PROF_OP("read");
        static_assert(!std::is_const_v<T> &&
                          (std::is_same_v<T, Scalar> || std::is_same_v<T, f16>),
                      "read: span 元素类型必须是 nn::Scalar(float) 或 nn::f16（U2 精确匹配）");
        NN_TRY(ec, bind_check_({&t}));
        constexpr Precision want =
            std::is_same_v<T, f16> ? Precision::F16 : Precision::F32;
        if (t.precision() != want)
            NN_FAIL("read: span 元素类型与张量精度不匹配（U2 精确匹配）");
        if (dst.size() != t.rows() * t.cols())
            NN_FAIL("read: 元素数与张量形状不匹配");
        // 统一经 to_matrix（GPU 分支自带 flush + wait_in_flight；f16 存储
        // 升 f32 精确无损，T=f16 时下面再 RHE 落回——往返位不变）。
        auto m = to_matrix(t, Precision::F32);
        NN_TRY_CHECK(m);
        const auto src = m->span();
        if constexpr (std::is_same_v<T, Scalar>)
        {
            std::copy(src.begin(), src.end(), dst.begin());
        }
        else
        {
            for (std::size_t i = 0; i < src.size(); ++i)
                dst[i] = static_cast<f16>(src[i]);
        }
        return {};
    }

protected:
    // ══════════════════════════════════════════════════════════════════════
    // 写版本护栏（WriteEpochGuard）—— 参数派生缓冲的失效依据
    //
    // 问题：Layer 的"参数派生物"（BitLinear 的 τ/wq）只在**参数变过**时才需要
    // 重算，而 forward 每步都会被调用（梯度累积 N 次、自回归生成每 token 一次
    // ——见 docs/todo/21 §4.11）。判断"变没变"必须有一个便宜且**不漏
    // 报**的信号。
    //
    // 机制：Tensor 的存储对象（MatrixT / GpuBuffer）持有一个单调计数
    // `write_epoch`；**每一个"经引擎媒介写入张量存储"的公共入口**在首行放一个
    // Guard，函数返回（成功或失败）时把目标张量的写版本 +1。派生缓冲记下
    // "我算它时的版本"，版本相同即内容未变。
    //
    // 三条性质（为什么可信）：
    //   1. **不遗漏**：库内所有写入都经引擎（M1/铁律 #11 由编译器保证存储
    //      访问器私有）；本类里每一个写入口都有 Guard（`bench/doc_inventory.ps1`
    //      的审计 + `src/write_epoch_test.cpp` 逐入口断言）。
    //   2. **别名安全**：版本记在存储对象上（不是 Tensor 上），Tensor 拷贝 /
    //      GPU with_shape 重贴标签 / import 零拷贝别名共享同一计数。
    //   3. **单调**：只前进（整对象替换走 `inherit_write_epoch_` 抬到两端较大
    //      者之后），故"替换后恰好等于旧值"的假命中不可能发生。
    // 代价：写入路径多一次自增；失败返回也会推进（保守方向 —— 多算一次派生
    // 缓冲，不会用旧值）。
    // ──────────────────────────────────────────────────────────────────────
    static void mark_written_(Tensor& t) noexcept { t.touch_write_epoch_(); }

    class WriteEpochGuard
    {
    public:
        explicit WriteEpochGuard(Tensor& t) noexcept : t_(&t) {}
        WriteEpochGuard(const WriteEpochGuard&) = delete;
        WriteEpochGuard& operator=(const WriteEpochGuard&) = delete;
        ~WriteEpochGuard() { mark_written_(*t_); }

    private:
        Tensor* t_;
    };

    // 多输出版（P2 写穿物化 / IR-C 图执行）：一次调用可能写多张张量
    // （outs[0] + spec.extras 的各槽），逐个推进写版本。span 只持引用，
    // 生命周期由调用方保证（与 WriteEpochGuard 同契约，零分配）。
    class WriteEpochGuardMulti
    {
    public:
        explicit WriteEpochGuardMulti(std::span<Tensor> ts) noexcept : ts_(ts) {}
        WriteEpochGuardMulti(const WriteEpochGuardMulti&) = delete;
        WriteEpochGuardMulti& operator=(const WriteEpochGuardMulti&) = delete;
        ~WriteEpochGuardMulti()
        {
            for (auto& t : ts_)
                mark_written_(t);
        }

    private:
        std::span<Tensor> ts_;
    };

public:
    // 批量写：src 覆盖 t 的既有存储（不替换对象、不改形状/精度）。
    // 与 copy_from 的分工：copy_from 收 f32 Matrix（含 f32→f16 转换填充，
    // 序列化加载用）；write 收与张量精度严格一致的 span（U2）。
    // 形参用 span<T> 而非 span<const T>：模板实参推导中 "const T" 无法与
    // 非 const 的 span<Scalar> 匹配（推导失败），故这里收 span<T>、静态断言
    // 只校验去 cv 后的元素类型；src 只读。
    //
    // ⚠ 本函数（以及下面每一个"写张量存储"的公共入口）首行的
    //   `WriteEpochGuard` 是**契约的一部分**，不要删：参数派生缓冲
    //   （BitLinear 的三值权重）靠写版本判断"要不要重算"，见下方说明。
    // fresh_dst 语义见 copy_from（默认 false；仅 detail::upload_span 传 true）。
    template <class T>
    [[nodiscard]] Result<void> write(Tensor& t, std::span<T> src,
                                     bool fresh_dst = false)
    {
        NN_PROF_OP("write");
        using Elem = std::remove_cv_t<T>;
        static_assert(std::is_same_v<Elem, Scalar> || std::is_same_v<Elem, f16>,
                      "write: span 元素类型必须是 nn::Scalar(float) 或 nn::f16（U2 精确匹配）");
        NN_TRY(ec, bind_check_({&t}));
        WriteEpochGuard wg{t};   // 写版本（见 guard 注释）
        constexpr Precision want =
            std::is_same_v<Elem, f16> ? Precision::F16 : Precision::F32;
        if (t.precision() != want)
            NN_FAIL("write: span 元素类型与张量精度不匹配（U2 精确匹配）");
        if (src.size() != t.rows() * t.cols())
            NN_FAIL("write: 元素数与张量形状不匹配");
        if (t.is_cpu())
        {
            // 宿主张量直写既有存储（保留对象身份）。不走 copy_from：GPU 引擎
            // 的 copy_from CPU 回退分支会替换 dst 对象，不满足 write 红线；
            // CPU 引擎的 F16 目标则要经 f32 转换，而 write 的 T 已与存储
            // 精确一致（U2），此处按存储元素类型直拷，零换算。
            if constexpr (std::is_same_v<Elem, f16>)
            {
                MatrixT<Precision::F16> m16(t.rows(), t.cols(),
                                            MatrixT<Precision::F16>::uninitialized_tag{});
                std::copy(src.begin(), src.end(), m16.span().begin());
                t.cpu_matrix<Precision::F16>() = std::move(m16);
            }
            else
            {
                Matrix m(t.rows(), t.cols(), Matrix::uninitialized_tag{});
                std::copy(src.begin(), src.end(), m.span().begin());
                t.cpu_matrix() = std::move(m);
            }
            return {};
        }

        // GPU 目标：经 copy_from（既有录制窗口 drain 语义；F16 目标 =
        // f32 上传 + cast_into 写回 dst 原存储，对象身份不替换）。
        // f16 → f32 提升精确无损，落回 dst 时 RHE，与边界 cast 同口径。
        Matrix m(t.rows(), t.cols(), Matrix::uninitialized_tag{});
        const auto d = m.span();
        if constexpr (std::is_same_v<Elem, Scalar>)
        {
            std::copy(src.begin(), src.end(), d.begin());
        }
        else
        {
            for (std::size_t i = 0; i < src.size(); ++i)
                d[i] = static_cast<Scalar>(src[i]);
        }
        return copy_from(t, m, fresh_dst);
    }

    // 索引读（语法糖：宿主直读 / GPU = 一次批量 read）。行主序 (row, col)；
    // f16 存储提升为 Scalar（精确无损）。GPU 每次调用一整轮 staging 往返，
    // 不承诺热循环性能（17 §7-1）——热路径请攒批量用 read。
    [[nodiscard]] Result<Scalar> get_index(const Tensor& t, std::size_t row,
                                           std::size_t col)
    {
        NN_TRY(ec, bind_check_({&t}));
        if (row >= t.rows() || col >= t.cols())
            NN_FAIL("get_index: (row, col) 越界");
        const std::size_t idx = row * t.cols() + col;
        if (t.is_cpu())
        {
            if (t.precision() == Precision::F16)
                return static_cast<Scalar>(t.cpu_matrix<Precision::F16>().span()[idx]);
            return t.cpu_matrix().span()[idx];
        }
        if (t.precision() == Precision::F16)
        {
            std::vector<f16> buf(t.rows() * t.cols());
            NN_TRY(r, read(t, std::span<f16>(buf)));
            return static_cast<Scalar>(buf[idx]);
        }
        std::vector<Scalar> buf(t.rows() * t.cols());
        NN_TRY(r, read(t, std::span<Scalar>(buf)));
        return buf[idx];
    }

    // 索引写（语法糖：宿主直写 / GPU = read → 改一格 → write）。
    // f16 存储：Scalar 值经 RHE 舍入（与 from_matrix/边界 cast 同口径）。
    [[nodiscard]] Result<void> set_index(Tensor& t, std::size_t row, std::size_t col,
                                         Scalar v)
    {
        NN_TRY(ec, bind_check_({&t}));
        WriteEpochGuard wg{t};
        if (row >= t.rows() || col >= t.cols())
            NN_FAIL("set_index: (row, col) 越界");
        const std::size_t idx = row * t.cols() + col;
        if (t.is_cpu())
        {
            if (t.precision() == Precision::F16)
                t.cpu_matrix<Precision::F16>().span()[idx] = static_cast<f16>(v);
            else
                t.cpu_matrix().span()[idx] = v;
            return {};
        }
        if (t.precision() == Precision::F16)
        {
            std::vector<f16> buf(t.rows() * t.cols());
            NN_TRY(r, read(t, std::span<f16>(buf)));
            buf[idx] = static_cast<f16>(v);
            return write(t, std::span<const f16>(buf));
        }
        std::vector<Scalar> buf(t.rows() * t.cols());
        NN_TRY(r, read(t, std::span<Scalar>(buf)));
        buf[idx] = v;
        return write(t, std::span<const Scalar>(buf));
    }

    // ── reshape（M1，docs/todo/17 §3 D10：修改类操作一律引擎方法）──
    // 语义同原 Tensor::reshape：GPU 共享底层 buffer（零拷贝视图）、CPU 按
    // precision_ 复制数据到新形状；设备域随存储派生（v3），空槽保持无效
    // 的输入形状由引擎校验。元素数不匹配返回错误（原 NN_ASSERT 升级）。
    [[nodiscard]] Result<Tensor> reshape(const Tensor& t, std::size_t new_rows,
                                          std::size_t new_cols)
    {
        NN_PROF_OP("reshape");
        NN_TRY(ec, bind_check_({&t}));
        if (t.rows() * t.cols() != new_rows * new_cols)
            NN_FAIL("reshape: element count mismatch");
        return t.reshape(new_rows, new_cols);
    }

    // ── cast 原语（§7.5，唯一"变精度"算子，永远显式）────────────────
    // 升 cast（f16→f32）精确无损；降 cast（f32→f16）round-half-to-even。
    // 默认实现：同精度 = 返回 src（共享所有权，零拷贝）；跨精度 = 错误（引擎覆盖）。
    [[nodiscard]] Result<Tensor> cast(const Tensor& src, Precision dst)
    {
        NN_PROF_OP("cast");
        NN_TRY(ec, bind_check_({&src}));
        return cast_impl(src, dst);
    }

    // ── cast_into：把 src 按精度转换后写入 **dst 的既有存储**（不替换对象）
    // 与 cast 的区别是"落点"：cast 返回新张量，cast_into 保留 dst 的
    // 对象身份与底层 buffer（in-place 语义必需——边界 cast 做 f16 原地更新时，
    // 若替换 dst 对象，其它持有同一张量句柄的缓存会静默失联）。
    // 要求 rows/cols 一致。默认实现：同精度走 copy_into，跨精度报错。
    [[nodiscard]] virtual Result<void> cast_into(const Tensor& src, Tensor& dst)
    {
        if (src.rows() != dst.rows() || src.cols() != dst.cols())
            NN_FAIL("cast_into: shape mismatch");
        if (src.precision() == dst.precision())
            return copy_into(dst, src);
        NN_FAIL("cast_into: 该引擎不支持跨精度转换");
    }

    // ── copy_into：同精度同形状拷贝（dst 的既有存储被完整覆盖，不替换对象）
    // 用于 cast_into 的同精度分支与边界 cast 的"写回原存储"路径。
    [[nodiscard]] virtual Result<void> copy_into(Tensor& dst, const Tensor& src)
    {
        (void)dst; (void)src;
        NN_FAIL("copy_into: 该引擎未实现");
    }

    // ══════════════════════════════════════════════════════════════════════
    // 精度策略（原 PrecisionEngine 逻辑）—— 现已**下沉到各引擎的 `*_impl`**
    //
    // 一句话：**f16 只改变"数据存哪儿"，不改变"算子怎么算"**——算术始终以
    // f32 为参考精度。两种落地方式（由引擎自己选，基类入口不做能力分支）：
    //   原生：引擎有带类型融合 shader / 字节级原语 → 直接把原张量交给内核
    //         （读 f16 / 写 f16，算术 f32），零边界临时量；
    //   边界 cast：入 = 把 f16 操作数抬到 f32（已是 f32 则零拷贝直通），
    //         算 = 调用既有 f32 实现（零分歧），
    //         出 = 结果按目标精度落回（f16 写出 = round-half-to-even）。
    // in-place 入口用 `cast_into` 写回**原存储**：保留张量对象身份与底层
    // buffer（§8.3：in-place 存储精度不可变）。基类仍提供通用 cast 助手
    // （`to_f32` / `to_prec` / `move_` / `unary_` / `binary_` / `inplace*_`）。
    //
    // 全 f32 配置下本层纯直通（每入口一条快速判定分支），行为与直接调用
    // 引擎实现逐字节一致（G5 零回归）。
    // ══════════════════════════════════════════════════════════════════════

    // ── 边界 cast 临时量归因（诊断；NN_PREC_TRACE=1）───────────────────
    // 记录每个"被物化的临时量"的**张量形状**（to_f32 = 抬到 f32 的副本，
    // 落回 = 输出按 P 舍入的副本）。用途：把 f16 路径的 transient 膨胀精确
    // 落到具体形状/算子（避免"猜哪个算子在 cast"）。零成本：未开 trace 时
    // note_temp_ 直接返回。
    struct TempStat
    {
        std::size_t rows = 0;
        std::size_t cols = 0;
        bool to_f32 = true;       // true = 抬到 f32 的副本；false = 输出落回
        std::size_t count = 0;
        std::size_t bytes = 0;    // 临时量实际字节（f16 副本按 2B/元素）
        std::uint32_t line = 0;   // 调用点行号（source_location，见 note_temp_）
    };

    [[nodiscard]] static std::vector<TempStat>& temp_stats()
    {
        static std::vector<TempStat> v;
        return v;
    }

    [[nodiscard]] static std::string dump_temp_stats()
    {
        auto& v = temp_stats();
        if (v.empty())
            return {};
        std::vector<const TempStat*> order;
        order.reserve(v.size());
        for (const auto& e : v) order.push_back(&e);
        std::sort(order.begin(), order.end(),
                  [](const TempStat* a, const TempStat* b) { return a->bytes > b->bytes; });
        std::string s = "── 边界 cast 临时量归因（按字节降序）──\n";
        for (const auto* e : order)
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "  %-11s (%zu,%zu)  x%-5zu %8.1f MB  L%u\n",
                          e->to_f32 ? "->f32 副本" : "->按P落回",
                          e->rows, e->cols, e->count,
                          static_cast<double>(e->bytes) / (1024.0 * 1024.0),
                          static_cast<unsigned>(e->line));
            s += buf;
        }
        return s;
    }

    static void reset_temp_stats() { temp_stats().clear(); }

    // ══════════════════════════════════════════════════════════════════════
    // 纯数据搬运原语（无 P：输出精度 = 源精度，§8.4）
    // ══════════════════════════════════════════════════════════════════════

    // ── 深拷贝 Tensor（CPU 矩阵拷贝 / GPU buffer 拷贝，无 PCIe 传输） ──
    // 用于需要修改中间结果但不影响原 Tensor 的场景
    [[nodiscard]] Result<Tensor> clone(const Tensor& src)
    {
        NN_PROF_OP("clone");
        NN_TRY(ec, bind_check_({&src}));
        // 精度策略（原生 f16 字节拷贝 / 抬 f32 再落回）在引擎的 clone_impl 内。
        return clone_impl(src);
    }

    // 将 CPU Matrix 数据写入已有 Tensor（CPU 拷贝 / GPU 上传）
    // 用于序列化加载、Optimizer 参数写回等场景。
    // f16 目标：先按 f32 上传，再 cast_into 写进 dst 的原存储
    //
    // fresh_dst（默认 false）= 调用方声明 **dst 是刚创建、当前录制帧与在飞
    // 帧都不可能引用的全新张量**（detail::upload_span 的 create→write 相邻
    // 形态即此类）。此时 GPU 侧**跳过录制窗口的整条流水线排空**：
    //   · 不存在"当前帧先前命令写 dst、被先执行的上传覆盖"的反向依赖——
    //     全新 buffer 无任何已录命令引用（池化复用只在写出方帧完成后还块，
    //     见 pending_destroys / reap_frame），故无需 wait_in_flight；
    //   · 上传 → 后续帧读 dst 的正向依赖由 staging 时间线信号量显式保证
    //     （帧提交时 collect_staging_waits 等待——与 from_matrix 批内上传
    //     同一条既有机制，GPU 测试已覆盖）。
    // 训练热路径 labels/mask/doc_ids 全走 upload_span → fresh 路径，每步 ~6
    // 次全排空被消除（历史画像 copy_from 占墙钟 71.8%，见 docs/history）。
    // fresh_dst=false（默认）保留原排空语义 → load_model / 测试直写既有张量
    // 零回归。
    // ⚠ 契约：fresh_dst=true 仅当 dst 未被任何**已录制或在飞**命令引用；
    // 误传 = 数据竞争。库内唯一承诺方是 detail::upload_span。
    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src,
                                         bool fresh_dst = false)
    {
        NN_PROF_OP("copy_from");
        NN_TRY(ec, bind_check_({&dst}));
        WriteEpochGuard wg{dst};
        if (dst.precision() == Precision::F32)
            return copy_from_impl(dst, src, fresh_dst);
        auto t = from_matrix(src, Precision::F32);
        NN_TRY_CHECK(t);
        return cast_into(*t, dst);
    }

    // ── 行切片原语（op-level 数据操作，不含算法语义） ──────────────────
    // 返回 src 的行 [start_row, start_row + count) 的连续拷贝。
    // 用于多头注意力中 per-head Q/K/V 切片等场景。
    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count)
    {
        NN_PROF_OP("slice_rows");
        NN_TRY(ec, bind_check_({&src}));
        return slice_rows_impl(src, start_row, count);
    }

    // 将 src 的所有行写入 dst 的行 [dst_start_row, dst_start_row + src.rows())。
    // 真·就地修改（GPU 用 vkCmdCopyBuffer with dstOffset）。
    // 用于多头注意力中 per-head 输出拼接等场景。
    // insert_rows：就地写入 dst 的行区间 → dst 精度即目标精度
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src)
    {
        NN_PROF_OP("insert_rows");
        NN_TRY(ec, bind_check_({&dst, &src}));
        WriteEpochGuard wg{dst};
        return insert_rows_impl(dst, dst_start_row, src);
    }

    // ── 行 gather / scatter-add 原语（op-level 数据操作，不含算法语义） ──
    // gather_rows: 按 indices 从 table 中按行查表，等价于 tf.gather / torch.index_select
    //   table: (vocab, D)
    //   indices: (num_indices,) — 行索引；越界索引返回零行（防御性，不抛错）
    //   输出: (num_indices, D)，out[i] = table[indices[i]]
    // 典型用途：Token embedding 查表（避免 Layer 内手动 to_matrix + at_unchecked）
    [[nodiscard]] Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices)
    {
        NN_PROF_OP("gather_rows");
        NN_TRY(ec, bind_check_({&table, &indices}));
        return gather_rows_impl(table, indices);
    }

    // scatter_add_rows: 按 indices 把 grad 的行原子累加到 dst 的对应行
    //   dst: (vocab, D)，原地修改
    //   indices: (num_indices,)
    //   grad: (num_indices, D)
    //   语义: dst[indices[i]] += grad[i]  (重复 indices 会被多次累加)
    // 典型用途：Embedding 反向梯度按 token ID 累加（替代 Layer 内手动循环）
    // scatter_add_rows：就地累加到 dst → 在 f32 空间累加后写回 dst 存储
    [[nodiscard]] Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad)
    {
        NN_PROF_OP("scatter_add_rows");
        NN_TRY(ec, bind_check_({&dst, &indices, &grad}));
        WriteEpochGuard wg{dst};
        return scatter_add_rows_impl(dst, indices, grad);
    }

    // 3D 维度转置：(M, B, N) ↔ (B, M, N)
    //   inverse=false: 输入 (M, B*N) → 输出 (B*M, N)
    //     out[b*M + m, n] = in[m, b*N + n]
    //   inverse=true:  输入 (B*M, N) → 输出 (M, B*N)
    //     out[m, b*N + n] = in[b*M + m, n]
    // 典型用途：MHA 批量化时把 (H*d_k, batch*seq) 重排为 (batch*H*d_k, seq)，
    //   使 batched_matmul 能按 batch*H 切分行块。
    [[nodiscard]] Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse = false)
    {
        NN_PROF_OP("rearrange_3d");
        NN_TRY(ec, bind_check_({&x}));
        return rearrange_3d_impl(x, M, B, N, inverse);
    }

    // ── 矩阵转置：A (R, C) → out (C, R) ──
    // 纯 layout 操作，零算法语义。用于 embedding 列布局转换等场景。
    [[nodiscard]] Result<Tensor> transpose(const Tensor& A)
    {
        NN_PROF_OP("transpose");
        NN_TRY(ec, bind_check_({&A}));
        return transpose_impl(A);
    }

    // ── 卷积/池化窗口展开原语（op-level 数据搬运，零算法语义）─────────────
    // im2col：把 (C, H, W) 输入按滑动窗展开成 GEMM 的列矩阵。
    //   x:   (C*H*W, B)
    //   out: (C*k*k, B*OH*OW)
    //   out[(ci*k + kh)*k + kw, (oh*OW + ow)*B + b]
    //       = x[ci*H*W + (oh*stride + kh - pad)*W + (ow*stride + kw - pad), b]
    // 越界（padding 区）取 0。OH/OW 由调用方按 (H + 2*pad - k)/stride + 1 给出。
    // **列序为「位置优先」(oh, ow, b)**：这样 Layer 只需 rearrange_3d + gather_rows
    //   各一次即可完成 (C, B*P) → (C*P, B) 的 samples 布局转换。
    // 用途：Conv2D 把卷积变成 matmul（W × col）；MaxPool2D 展开窗口做列归约
    //   （取 k=pool, stride=stride, pad=0）。
    [[nodiscard]] Result<Tensor> im2col(
        const Tensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW)
    {
        NN_PROF_OP("im2col");
        NN_TRY(ec, bind_check_({&x}));
        return im2col_impl(x, C, H, W, k, stride, pad, OH, OW);
    }

    // col2im：im2col 的伴随（adjoint / 反向散射）。
    //   col: (C*k*k, B*OH*OW) → out: (C*H*W, B)
    //   out[ci*H*W + ih*W + iw, b]
    //       = Σ_{kh,kw} col[(ci*k + kh)*k + kw, (oh*OW + ow)*B + b]
    //   其中 oh = (ih + pad - kh)/stride（须整除且落在 [0, OH)），ow 同理。
    // **重叠窗口（stride < k）的贡献在此累加**；每个输出元素由单个线程/循环
    //   完整求和，故无需原子操作/预清零。
    [[nodiscard]] Result<Tensor> col2im(
        const Tensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW)
    {
        NN_PROF_OP("col2im");
        NN_TRY(ec, bind_check_({&col}));
        return col2im_impl(col, C, H, W, k, stride, pad, OH, OW);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════

    // C = A × B（支持转置标志 + 精度参数，§8.1）
    // transA: 使用 A^T，transB: 使用 B^T
    // P: 计算精度（F32 = 默认；F16 = f16 GEMM，§7.4）：由 Layer 显式传入
    //   （PrecisionProfile.compute；无隐式推导，§8.5）
    // 非 f32 P 直接下传（不经本层 cast）：引擎有 **f16 存储版 GEMM**
    // （f16 直读 + f32 累加 + f16 写出）→ 不为每个操作数物化整份 f32 副本。
    // 引擎在"f16 pipeline 不可用 / 小 N GEMV / 操作数精度混合"时自行回退
    // f32 空间 + 边界，正确性不变。
    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("matmul");
        NN_TRY(ec, bind_check_({&A, &B}));
        if (P != Precision::F32)
            return matmul_impl(A, B, transA, transB, P);
        return binary_(A, B, P, [this, transA, transB](const Tensor& a, const Tensor& b)
        {
            return matmul_impl(a, b, transA, transB, Precision::F32);
        });
    }

    // 批量矩阵乘法：对每个 batch b 计算 C_b = alpha * op(A_b, B_b)，结果垂直堆叠
    // A: (batch * A_rows_per_batch, A_cols) — 按 batch 切分为连续行块
    // B: (batch * B_rows_per_batch, B_cols)
    // 输出: (batch * M, N)，M/N 为每个 batch 的逻辑输出维度
    //   transA=0: A_b 为 (M, K)，transA=1: A_b 存储为 (K, M) 按 A_b^T 使用
    //   transB=0: B_b 为 (K, N)，transB=1: B_b 存储为 (N, K) 按 B_b^T 使用
    // alpha: 输出缩放系数（cuBLAS sgemm 语义），GPU 在 shader 写出时一次完成，
    //   供上层折叠 1/sqrt(d_k) 等系数，省去额外全矩阵 scale pass
    // P: 计算精度（§8.1，同 matmul）
    // 典型用途：多头注意力的 Q^T×K 和 V×A 批量化（消除 per-head 循环）
    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA = false, bool transB = false,
        Scalar alpha = Scalar{1},
        Precision P = Precision::F32)
    {
        NN_PROF_OP("batched_matmul");
        NN_TRY(ec, bind_check_({&A, &B}));
        if (P != Precision::F32)
            return batched_matmul_impl(A, B, batch, transA, transB, alpha, P);
        return binary_(A, B, P,
            [this, batch, transA, transB, alpha](const Tensor& a, const Tensor& b)
        {
            return batched_matmul_impl(a, b, batch, transA, transB, alpha,
                                       Precision::F32);
        });
    }

    // ── matmul + broadcast bias（统一精度处理）────────────────────────────
    // out = A × B + bias（broadcast add，bias (out,1) → (out,batch)）
    // 入口按 P/操作数精度分派：非 f32 直接下传（经 DSL 融合 matmul 段，带类型
    // 变体可用时 f16 直读直写、零边界 cast；无变体时引擎自行回退 f32 空间）；
    // 全 f32 直通；混合输入抬 f32 计算后按 P 落回。
    [[nodiscard]] Result<Tensor> matmul_with_bias(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("matmul_with_bias");
        NN_TRY(ec, bind_check_({&A, &B, &bias}));
        if (P != Precision::F32 &&
            (A.precision() != Precision::F32 || B.precision() != Precision::F32 ||
             bias.precision() != Precision::F32))
            return matmul_with_bias_impl(A, B, bias, transA, transB, P);

        if (P == Precision::F32 && A.precision() == Precision::F32 &&
            B.precision() == Precision::F32 && bias.precision() == Precision::F32)
            return matmul_with_bias_impl(A, B, bias, transA, transB,
                                         Precision::F32);
        auto a = to_f32(A);
        NN_TRY_CHECK(a);
        auto b = to_f32(B);
        NN_TRY_CHECK(b);
        auto bi = to_f32(bias);
        NN_TRY_CHECK(bi);
        auto r = matmul_with_bias_impl(*a, *b, *bi, transA, transB,
                                       Precision::F32);
        NN_TRY_CHECK(r);
        return to_prec(std::move(*r), P);
    }

    // 梯度累加：dst += src（dst 存储精度不可变，§8.3）
    [[nodiscard]] Result<void> accumulate(Tensor& dst, const Tensor& src)
    {
        NN_PROF_OP("accumulate");
        NN_TRY(ec, bind_check_({&dst, &src}));
        WriteEpochGuard wg{dst};
        const bool dbg = prec_env_flag("NN_F16_DEBUG");
        const auto mx = [this](const Tensor& t)
        {
            auto m = to_matrix(t, Precision::F32);
            if (!m) return -1.0;
            double x = 0.0;
            for (auto v : m->span())
            {
                if (!std::isfinite(v)) return -2.0;
                x = std::max(x, std::fabs(static_cast<double>(v)));
            }
            return x;
        };
        const double pre_d = dbg ? mx(dst) : 0.0;
        const double pre_s = dbg ? mx(src) : 0.0;
        auto r = add_inplace_impl(dst, src);
        if (dbg)
        {
            const double post_d = mx(dst);
            if (pre_d > 10.0 || pre_s > 10.0 || post_d > 10.0)
                std::fprintf(stderr,
                             "[dbg][accumulate] %s pre_dst=%.6g pre_src=%.6g post_dst=%.6g\n",
                             dst.shape_str().c_str(), pre_d, pre_s, post_d);
        }
        return r;
    }

    // A += B（逐元素，同形状）
    [[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B)
    {
        NN_PROF_OP("add_inplace");
        NN_TRY(ec, bind_check_({&A, &B}));
        WriteEpochGuard wg{A};
        return add_inplace_impl(A, B);
    }

    // A *= scalar
    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s)
    {
        NN_PROF_OP("scale_inplace");
        NN_TRY(ec, bind_check_({&A}));
        WriteEpochGuard wg{A};
        return scale_inplace_impl(A, s);
    }

    // A = 0
    [[nodiscard]] Result<void> zero(Tensor& A)
    {
        NN_PROF_OP("zero");
        NN_TRY(ec, bind_check_({&A}));
        WriteEpochGuard wg{A};
        return zero_impl(A);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 三值打包权重（P2 / F3）：trit_pack + matmul_q
    //
    // BitLinear 的"有效权重"是 τ∘W（W 三值、τ 逐行 absmean）。打包后权重容器
    // 只占 **0.25 B/权重**（f16 稠密缓冲 2 B/权重），代价是必须由专用内核解码
    // —— docs/todo/21 §4.6 的"T1_58 张量只进量化 GEMM"在这里落地为
    // "打包容器只进 matmul_q"（布局见 ternary_pack.hpp）。
    //
    //   trit_pack(w (R,K) f32, τ (R,1) f32) → 容器 (R, ceil(K/16)) f32
    //   matmul_q  (容器, τ, B)          → C = (τ∘W)·B   （W = (M,K)，M=R）
    //   matmul_q_t(容器, τ, B, out_rows) → C = (τ∘W)^T·B（A 存为 (K,M)，即 W
    //                                         本身 (out,in) 按 A^T 用 ——
    //                                         反向的 dX 走这条）
    //   τ 恒按**存储行**（= W 的输出通道）取：两条路径同一口径，内核无需分支。
    //   入口不带 bias（BitLinear 在调用点用一次 row_broadcast 加）—— 保持原语
    //   最小；把 bias 折进 shader 是后续可选优化（省一次 (out,batch) dispatch）。
    //
    // 能力：不支持打包的引擎由 `*_impl` 自己 fail-fast（基类默认实现即
    // NN_FAIL），公共入口不做能力分支（策略在引擎内部）。
    // ══════════════════════════════════════════════════════════════════════

    [[nodiscard]] Result<Tensor> trit_pack(const Tensor& w, const Tensor& scale)
    {
        NN_TRY(ec, bind_check_({&w, &scale}));
        return trit_pack_impl(w, scale);
    }

    [[nodiscard]] Result<Tensor> matmul_q(const Tensor& packed, const Tensor& scale,
                                          const Tensor& B,
                                          Precision P = Precision::F32)
    {
        return matmul_q_dispatch_(packed, scale, B, /*transA=*/false, /*out_rows=*/0,
                                  P);
    }

    // C = (τ∘W)^T·B：A 存储为 (K, M)（= W 的 (out, in)），out_rows = M。
    [[nodiscard]] Result<Tensor> matmul_q_t(const Tensor& packed, const Tensor& scale,
                                            const Tensor& B, std::size_t out_rows,
                                            Precision P = Precision::F32)
    {
        return matmul_q_dispatch_(packed, scale, B, /*transA=*/true, out_rows, P);
    }

private:
    [[nodiscard]] Result<Tensor> matmul_q_dispatch_(const Tensor& packed,
                                                    const Tensor& scale,
                                                    const Tensor& B, bool transA,
                                                    std::size_t out_rows, Precision P)
    {
        NN_TRY(ec, bind_check_({&packed, &scale, &B}));
        // 精度策略与 matmul 同款：非 f32 下传引擎自行决定（GPU 有 in-kernel f16
        // 变体 → 直读 f16 的 B / 直写 f16 的 C，零边界 cast；CPU/无变体时由
        // 引擎内部抬 f32 计算 + 按 P 落回，见各 *matmul_q_impl*）。
        return matmul_q_impl(packed, scale, B, transA, out_rows, P);
    }

public:

    // ══════════════════════════════════════════════════════════════════════
    // 扫描级原语（带状态的顺序归约 + matvec 读出；RLA 线性注意力积木）
    //
    // 引擎只提供"前缀/后缀顺序归约 + matvec 读出"；RLA 算法（L2 归一化
    // 分母 / ReLU 门控 / 梯度公式 / 文档重置策略）全部由 Layer 用这些原语
    // 与逐元素原语组合表达（铁律 3：shader 永不含算法）。
    //
    // 形状约定（batch-major，列序 i = b*seq + t；头 (b,h) 的行块起点
    //   r0 = (b*H + h)*d_k，每头 d_k 行）：
    //   K/V/P/R（X/Y）: (B·H·d_k, seq)
    //   D             : (B·H·d_k², seq)，(b,h) 的 (a,b') 元素在
    //                    行 (b*H*d_k + a)*d_k + b'
    //   A0/B0         : (H·d_k, d_k) 初始运行态，行块 h = 第 h 头
    //                    （B>1 时按头循环）；has_state=false → 按零
    //                    处理（传 (1,1) dummy，规避 0 字节 buffer）
    //   boundary      : (1, B·seq)，1 = 文档起点（t==0 或与前一位置
    //                    文档不同）；has_bnd=false → 无文档感知
    //                    （传 (1,1) dummy）
    //   标量块（s/r）: 每 (b,h,t) 一个标量，在头块内 d_k 行重复存放
    //                    （避免块级广播原语）；实现写全部 d_k 行
    //                    的同一值，Layer 读任一行均可。
    //
    // 原生 f16 扫描（GPU）→ f16 输入直接下传，**不物化全尺寸 f32 输入副本**
    // （scan 输出是 RLA 训练的大头临时量）。CPU 能力 false → 边界 cast。
    // 输出按 prec 落回。
    // ══════════════════════════════════════════════════════════════════════

    // 前缀扫描：
    //   causal=true : 含自身前缀（i<=t）：A_t = A0 + Σ_{i≤t, 与 t 同文档}
    //                 k_i·k_i^T，B_t = B0 + Σ_{i≤t, 同文档} v_i·k_i^T；
    //                 文档边界处运行态清零（A0/B0 仅首个文档生效）。
    //   causal=false: 全集常数 A = A0 + Σ_all k·k^T，B = B0 + Σ_all v·k^T
    //                 （无边界重置）。
    // 输出 (B·H·5·d_k, seq)，行块（每块 (B·H·d_k, seq)）：
    //   [0) B·P   [1) A·P   [2) B^T·R   [3) s = P·(A·P)   [4) r = R·(B·P)
    //   其中 [3)/[4) 为逐列标量（头内逐行重复）。
    [[nodiscard]] Result<Tensor> scan_prefix_outer(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32)
    {
        NN_PROF_OP("scan_prefix_outer");
        NN_TRY(ec, bind_check_({&K, &V, &P, &R, &A0, &B0, &boundary}));
        // 精度策略（原生 f16 扫描 / 引擎内边界 cast）在引擎的 *_impl 内。
        return scan_prefix_outer_impl(K, V, P, R, A0, B0, has_state, dk, heads,
                                             causal, boundary, has_bnd, prec);
    }

    // 后缀扫描（RLA 反向 pass 2）：
    //   causal=true : S_i = Σ_{t≥i, 与 i 同文档} D_t（i+1 为文档起点时
    //                 先清零再累加 D_i）；
    //   causal=false: S_i = D_i（Layer 预先把全集梯度沿 seq 广播）。
    // D (B·H·d_k², seq)，X/Y (B·H·d_k, seq)
    // 输出 (B·H·3·d_k, seq)，行块：[0) S·X   [1) S·Y   [2) S^T·Y
    [[nodiscard]] Result<Tensor> scan_suffix_outer(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32)
    {
        NN_PROF_OP("scan_suffix_outer");
        NN_TRY(ec, bind_check_({&D, &X, &Y, &boundary}));
        return scan_suffix_outer_impl(D, X, Y, dk, heads, causal, boundary,
                                            has_bnd, prec);
    }

    // 逐列外积（RLA 反向的 dL/dA、dL/dB 物化）：
    //   out[(b,h): (a,b'), t] = P[a,t]·R[b',t] (· S[t] if has_scale)
    // P/R: (B·H·d_k, seq)；S: (B·H·d_k, seq)（标量头内逐行重复，
    // 实现读头块首行；has_scale=false → 传 dummy）
    // 输出: (B·H·d_k², seq)
    [[nodiscard]] Result<Tensor> outer_col(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale,
        Precision prec = Precision::F32)
    {
        NN_PROF_OP("outer_col");
        NN_TRY(ec, bind_check_({&P, &R, &S}));
        return outer_col_impl(P, R, S, dk, has_scale, prec);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语
    // ══════════════════════════════════════════════════════════════════════
    // §7.3：累加恒 f32，输出舍入到 P。原生 f16 归约（GPU reduce f16 变体，
    // 归约全程 f32、输出 f32 向量）→ f16 输入直接下传，不物化整份 f32 输入
    // 副本；CPU 能力 false → 边界 cast。输出仍按 P 落回（(rows,1)/(1,cols)
    // 归约向量极小，开销可忽略）。

    // 按行求和：A (rows, cols) → out (rows, 1)
    // out[r] = Σ_c A[r][c]
    // P：输出（与累加）精度（§7.3：归约恒 f32 累加 + 输出舍入到 P）
    [[nodiscard]] Result<Tensor> row_reduce_sum(
        const Tensor& A, Precision P = Precision::F32)
    {
        NN_PROF_OP("row_reduce_sum");
        NN_TRY(ec, bind_check_({&A}));
        return row_reduce_sum_impl(A, P);
    }

    // 按列求和：A (rows, cols) → out (1, cols)
    // out[c] = Σ_r A[r][c]
    [[nodiscard]] Result<Tensor> col_reduce_sum(
        const Tensor& A, Precision P = Precision::F32)
    {
        NN_PROF_OP("col_reduce_sum");
        NN_TRY(ec, bind_check_({&A}));
        return col_reduce_sum_impl(A, P);
    }

    // 按列求最大值：A (rows, cols) → out (1, cols)
    // out[c] = max_r A[r][c]
    [[nodiscard]] Result<Tensor> col_reduce_max(
        const Tensor& A, Precision P = Precision::F32)
    {
        NN_PROF_OP("col_reduce_max");
        NN_TRY(ec, bind_check_({&A}));
        return col_reduce_max_impl(A, P);
    }

    // ── 分组归约（segmented reduce，沿行方向按固定长度分组）───────────────
    // 与 row/col_reduce 的区别：归约轴不是"整行/整列"，而是**每连续 R 行为一组**。
    //   x: (G*R, N) → out: (G, N)
    //   grouped_reduce_sum: out[g, n] = Σ_{i<R} x[g*R + i, n]
    //   grouped_reduce_max: out[g, n] = max_i  x[g*R + i, n]
    // 用途：把"逐通道/逐头一次 dispatch"的层内循环压成**单次**原语调用
    //   （池化窗口归约、多头分组统计、分组归一化等）。
    [[nodiscard]] Result<Tensor> grouped_reduce_sum(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("grouped_reduce_sum");
        NN_TRY(ec, bind_check_({&x}));
        return grouped_reduce_sum_impl(x, G, R, P);
    }
    [[nodiscard]] Result<Tensor> grouped_reduce_max(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("grouped_reduce_max");
        NN_TRY(ec, bind_check_({&x}));
        return grouped_reduce_max_impl(x, G, R, P);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（逐元素融合的统一入口）
    // ══════════════════════════════════════════════════════════════════════

    // 对一个逐元素表达式求值，输出 (rows, cols)。所有输入同形状。
    //
    // 这是"函数式逐元素原语"的表达式升级：单行内多次计算（如 RoPE 的
    // q*cos + rotate(q)*sin、残差、激活）可合并为一次调用，减少临时 Tensor。
    // 执行策略由后端决定（CPU 编译期模板求值；Vulkan AOT 融合 shader，闭合世界），
    // Layer 侧无需关心——表达式是唯一逐元素编程模型。
    //
    // 语义与上限见 expr_spec.hpp（ExprSpec）。输出 = 最后一条指令的目标寄存器。
    // in-kernel f16 优先：引擎有该 (结构, 输入精度, 输出精度) 的**带类型**
    // 融合 shader → 直接把原张量交给它（读 f16 / 写 f16，算术 f32）；
    // 无变体 → 抬 f32 计算后按 P 落回（trace_miss_ 是唯一可见信号）。
    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("eval_expr");
        NN_TRY(ec, bind_check_(inputs));
        if (!spec.extras.empty())
            NN_FAIL("eval_expr: 多输出 spec（P2 extras）请走 eval_expr_multi_into");
        // IR-C 录制段：入图不求值（占位 Tensor，end_expr 融合执行）
        if (auto* g = fused::recording_graph())
            return record_graph_node_(*g, spec, inputs, rows, cols,
                                      /*vector_out=*/false, P);
        // 变体选路 / 边界 cast 归因**只做一次**：整段下沉到引擎的
        // eval_expr_impl（它本来就要算 key/精度签名）——F2 修复。
        return eval_expr_impl(spec, inputs, rows, cols, P);
    }

    // ── 归约向量原生形状输出（LayerNorm/RMSNorm 小向量缓存等） ──────────
    // 语义同 eval_expr，但输出为归约向量本身（非广播）：
    //   行归约轴 → (rows,1)；列归约轴 → (1,cols)。
    // 要求表达式归约轴为 0/1（expr_spec_reduce_axis）。
    [[nodiscard]] Result<Tensor> eval_expr_reduce(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32)
    {
        NN_PROF_OP("eval_expr_reduce");
        NN_TRY(ec, bind_check_(inputs));
        if (!spec.extras.empty())
            NN_FAIL("eval_expr_reduce: 多输出 spec（P2 extras）请走 eval_expr_multi_into");
        // IR-C 录制段：入图不求值（归约向量占位，end_expr 融合执行）
        if (auto* g = fused::recording_graph())
            return record_graph_node_(*g, spec, inputs, rows, cols,
                                      /*vector_out=*/true, P);
        // 变体选路 / 边界 cast 归因整段下沉到引擎的 eval_expr_reduce_impl（F2 修复）
        return eval_expr_reduce_impl(spec, inputs, rows, cols, P);
    }

    // ── 目标传递（destination-passing）：结果写入已有张量 ────────────────
    // 语义同 eval_expr，但输出**直接写进 out**（不分配新张量），用于表达
    // "原地更新"语义：dst = f(dst, ...)（梯度累加 / 参数更新 / 就地缩放 /
    // 就地按行广播等）。out 允许与某个输入是同一 buffer：逐元素"先读完全部
    // 输入再写 out[i]"，就地安全。
    // 仅支持逐元素表达式（无归约）；归约向量输出用 eval_expr_reduce。
    [[nodiscard]] Result<void> eval_expr_into(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out)
    {
        NN_PROF_OP("eval_expr_into");
        NN_TRY(ec, bind_check_(inputs));
        NN_TRY(ec2, bind_check_({&out}));
        if (!spec.extras.empty())
            NN_FAIL("eval_expr_into: 多输出 spec（P2 extras）请走 eval_expr_multi_into");
        WriteEpochGuard wg{out};
        // IR-C 录制段（P2 写穿物化）：目标传递节点——输出即 dst 既有张量，
        // end_expr 的复合 kernel 直接写回其存储（多输出槽 0）。物化发生在
        // end_expr 且不经本入口，故写版本在**录制点**就推进（保守方向）。
        if (auto* g = fused::recording_graph())
            return record_graph_into_node_(*g, spec, inputs, rows, cols, out);
        // 变体选路 / 边界 cast 归因整段下沉到引擎的 eval_expr_into_impl（F2 修复）
        return eval_expr_into_impl(spec, inputs, rows, cols, out);
    }

    // ── P2 多输出（IR-C 写穿物化）：一次求值写多个输出张量 ────────────────
    // outs[0] = 主输出（instrs.back().dst），outs[1+k] = spec.extras[k] 的值
    // 各自写进对应张量的既有存储。仅纯逐元素 **V0（全 f32）** spec——生成器
    // 对带 extras 的结构只发 V0 变体（`#x`/typed/native16 跳过），且多输出
    // 融合由 fuse_expr_graph 的精度门禁限定 F32。不支持录制段（图执行是
    // end_expr 的职责）。
    [[nodiscard]] Result<void> eval_expr_multi_into(
        const ExprSpec& spec, std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, std::span<Tensor> outs)
    {
        NN_PROF_OP("eval_expr_multi_into");
        NN_TRY(ec, bind_check_(inputs));
        NN_TRY(ec2, bind_check_(outs));
        WriteEpochGuardMulti wgs{outs};   // 写版本：多输出逐个推进
        if (fused::is_recording())
            NN_FAIL("eval_expr_multi_into: 不支持在 begin_expr/end_expr 录制段内使用");
        if (outs.size() != spec.extras.size() + 1)
            NN_FAIL("eval_expr_multi_into: outs 数量 != 1 + extras 数量");
        for (const auto& t : inputs)
            if (t.precision() != Precision::F32)
                NN_FAIL("eval_expr_multi_into: 多输出仅支持全 f32（V0）输入");
        for (const auto& t : outs)
            if (t.precision() != Precision::F32)
                NN_FAIL("eval_expr_multi_into: 多输出仅支持全 f32（V0）输出");
        return eval_expr_multi_into_impl(spec, inputs, rows, cols, outs);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 引擎实现接口（protected 虚函数，NVI 的内层）—— 引擎只实现这些：
    // **既管算，也管自己的精度边界策略**（原生 f16 直通 / 抬 f32 再落回）。
    // 默认实现仅覆盖"引擎未提供"的少数入口；其余为纯虚。
    // ══════════════════════════════════════════════════════════════════════
protected:
    // ── P1 NVI 的引擎侧实现（原公共虚入口下沉至此，stamp 在公共入口统一做）──
    // RV-02：实现侧返回 Result —— 分配失败（GPU 显存/设备）的原因必须能被
    // 上层拿到，旧形态 `return Tensor();` 把原因整块丢弃。
    [[nodiscard]] virtual Result<Tensor> create_tensor_impl(std::size_t rows, std::size_t cols, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> from_matrix_impl(const Matrix& m, Precision P) = 0;
    // 跨设备拉取（M6；v3 起是唯一 CPU→GPU 跨域入口，公共入口做设备域检查）。
    // 默认实现 = 经宿主中转；同设备快路（别名 / 引擎内 cast）也在默认实现内。
    // GpuEngine override 为"CPU → 设备直传"（原 ensure_gpu 路径）。
    [[nodiscard]] virtual Result<Tensor> import_impl(const Tensor& src, Precision P)
    {
        if (src.device() == device())
        {
            if (P == src.precision())
                return src;                 // 同设备同精度 = 零拷贝别名
            return cast(src, P);            // 同设备异精度 = 引擎内 cast
        }
        auto m = to_matrix(src, Precision::F32);   // 跨设备：宿主中转
        NN_TRY_CHECK(m);
        return from_matrix(*m, P);
    }
    // cast 默认实现：同精度 = 返回 src（共享所有权，零拷贝）；跨精度 = 错误（引擎覆盖）
    [[nodiscard]] virtual Result<Tensor> cast_impl(const Tensor& src, Precision dst)
    {
        if (src.precision() == dst)
            return src;  // 同精度 = 返回共享所有权（零拷贝）
        NN_FAIL("cast: 该引擎不支持跨精度转换");
    }
    // offload 占位默认（激活 offload 是 GPU 特性；CPU 开启只得到 1×1 张量）
    // D8（15 §4.8）：走 `create_tensor` 而非 `Tensor::cpu` 静态工厂——出生即经
    // 引擎入口，形状/语义不变（仍是 1×1 占位）。
    [[nodiscard]] virtual Result<Tensor> create_offload_buffer_impl(std::size_t /*bytes*/)
    {
        return create_tensor(1, 1);
    }
    [[nodiscard]] virtual Result<Tensor> offload_restore_impl(
        const Tensor& /*buffer*/, std::size_t /*offset*/,
        std::size_t /*rows*/, std::size_t /*cols*/)
    {
        return create_tensor(1, 1);
    }

    // ── offload / 回读 ────────────────────────────────────────────────────
    // CPU no-op 默认（激活 offload 是 GPU 特性；CPU 开启只得到 1×1 张量）
    [[nodiscard]] virtual Result<void> offload_save_impl(
        const Tensor& /*buffer*/, std::size_t /*offset*/, const Tensor& /*src*/)
    {
        return {};
    }
    // 默认实现（CPU / 其他同步引擎）：submit 立刻取标量存值，poll 恒立刻就绪。
    // （入口已统一抬到 f32，这里只读 f32；f16 分支保留作健壮性兜底。）
    [[nodiscard]] virtual Result<void> submit_scalar_readback_impl(
        std::size_t slot, const Tensor& t)
    {
        if (!t.is_cpu())
            NN_FAIL("submit_scalar_readback: 默认实现仅支持 CPU 张量");
        Scalar v = Scalar{0};
        if (t.precision() == Precision::F32)
        {
            const auto& m = t.cpu_matrix();
            v = m.span()[0];
        }
        else if (t.precision() == Precision::F16)
        {
            const auto& m = t.cpu_matrix<Precision::F16>();
            v = static_cast<Scalar>(m.span()[0]);
        }
        else
        {
            NN_FAIL("submit_scalar_readback: 不支持的精度");
        }
        if (sync_readback_slots_.size() <= slot)
            sync_readback_slots_.resize(slot + 1, Scalar{0});
        sync_readback_slots_[slot] = v;
        return {};
    }

    // ── 纯数据搬运 ────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> clone_impl(const Tensor& src) = 0;
    [[nodiscard]] virtual Result<void> copy_from_impl(Tensor& dst, const Matrix& src,
                                                      bool fresh_dst) = 0;
    [[nodiscard]] virtual Result<Tensor> slice_rows_impl(
        const Tensor& src, std::size_t start_row, std::size_t count) = 0;
    [[nodiscard]] virtual Result<void> insert_rows_impl(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) = 0;
    [[nodiscard]] virtual Result<Tensor> gather_rows_impl(
        const Tensor& table, const Tensor& indices) = 0;
    [[nodiscard]] virtual Result<void> scatter_add_rows_impl(
        Tensor& dst, const Tensor& indices, const Tensor& grad) = 0;
    [[nodiscard]] virtual Result<Tensor> rearrange_3d_impl(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse) = 0;
    [[nodiscard]] virtual Result<Tensor> transpose_impl(const Tensor& A) = 0;
    [[nodiscard]] virtual Result<Tensor> im2col_impl(
        const Tensor& x,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) = 0;
    [[nodiscard]] virtual Result<Tensor> col2im_impl(
        const Tensor& col,
        std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) = 0;

    // ── 矩阵级 ────────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> matmul_impl(
        const Tensor& A, const Tensor& B,
        bool transA, bool transB, Precision P) = 0;

    // 三值打包权重（P2 / F3）：语义见公共入口 trit_pack / matmul_q 的说明。
    // 默认 = 引擎未接打包路径（fail-fast）；不支持打包的引擎由这里的默认
    // 实现或各自的覆盖 fail-fast（能力不下沉到上层）。
    [[nodiscard]] virtual Result<Tensor> trit_pack_impl(const Tensor& w,
                                                        const Tensor& scale)
    {
        (void)w; (void)scale;
        NN_FAIL("trit_pack: 该引擎未实现三值打包");
    }
    [[nodiscard]] virtual Result<Tensor> matmul_q_impl(const Tensor& packed,
                                                      const Tensor& scale,
                                                      const Tensor& B, bool transA,
                                                      std::size_t out_rows,
                                                      Precision P)
    {
        (void)packed; (void)scale; (void)B; (void)transA; (void)out_rows; (void)P;
        NN_FAIL("matmul_q: 该引擎未实现三值 GEMM");
    }
    [[nodiscard]] virtual Result<Tensor> batched_matmul_impl(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha,
        Precision P) = 0;

    // 默认实现：matmul + 逐行 add bias（兼容所有引擎）
    // D6-3（15 §4.6-3，M6 裁定=删除宿主兜底）：原默认实现是
    // `to_matrix → 主机逐列加 bias → from_matrix` 的 PCIe 往返兜底，两引擎
    // （CpuEngine/GpuEngine）均已 override，属死码且会在第三后端上静默退化。
    // 改为纯虚：新引擎必须给出真现值（编译期强制）。
    [[nodiscard]] virtual Result<Tensor> matmul_with_bias_impl(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA, bool transB, Precision P) = 0;

    [[nodiscard]] virtual Result<void> add_inplace_impl(Tensor& A, const Tensor& B) = 0;
    [[nodiscard]] virtual Result<void> scale_inplace_impl(Tensor& A, Scalar s) = 0;
    [[nodiscard]] virtual Result<void> zero_impl(Tensor& A) = 0;

    // ── 扫描级 ────────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> scan_prefix_outer_impl(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec) = 0;
    [[nodiscard]] virtual Result<Tensor> scan_suffix_outer_impl(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec) = 0;
    [[nodiscard]] virtual Result<Tensor> outer_col_impl(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale,
        Precision prec) = 0;

    // ── 归约 ──────────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> row_reduce_sum_impl(
        const Tensor& A, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> col_reduce_sum_impl(
        const Tensor& A, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> col_reduce_max_impl(
        const Tensor& A, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> grouped_reduce_sum_impl(
        const Tensor& x, std::size_t G, std::size_t R, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> grouped_reduce_max_impl(
        const Tensor& x, std::size_t G, std::size_t R, Precision P) = 0;

    // ── 表达式求值 ────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> eval_expr_impl(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P) = 0;

    // 默认实现返回错误（未支持的引擎）；CPU/GPU 覆盖。
    [[nodiscard]] virtual Result<Tensor> eval_expr_reduce_impl(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols; (void)P;
        NN_FAIL("eval_expr_reduce: 该引擎不支持归约向量输出");
    }

    // 默认实现返回错误（未支持的引擎）；CPU/GPU 覆盖为真原地实现。
    [[nodiscard]] virtual Result<void> eval_expr_into_impl(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols; (void)out;
        NN_FAIL("eval_expr_into: 该引擎不支持原地表达式求值");
    }

    // P2 多输出（IR-C 写穿物化）；默认实现返回错误（未支持的引擎）。
    [[nodiscard]] virtual Result<void> eval_expr_multi_into_impl(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, std::span<Tensor> outs)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols; (void)outs;
        NN_FAIL("eval_expr_multi_into: 该引擎不支持多输出表达式求值");
    }

protected:
    // 下面这些是**引擎实现可复用的 cast 助手**（各引擎的 `*_impl` 在
    // "原生路径不可用"时用它们表达边界 cast）。对外不可见。
    //
    // 注意（生命周期 v3）：不再有 `stamp_`/`adopt` 出生绑定通道。设备域从
    // 存储派生（CPU → kHostDomain，GPU → GpuBuffer 的 backend 标识），静态工厂
    // 产出的 CPU 张量天然属于宿主域，引擎内部中转无需补任何身份。

    // ── 形状合法性（RV-02）：rows*cols 溢出 size_t 直接拒绝 ────────────────
    // 溢出后按"回绕后的元素数"分配 = 立即越界，属参数错误而非分配失败，故在
    // 工厂前置拦下（零尺寸合法：(1,1) dummy 与空批次路径都用到）。
    [[nodiscard]] static Result<void> check_shape_(std::size_t rows, std::size_t cols)
    {
        if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows)
            NN_FAIL("create_tensor: 形状溢出（rows*cols 超过 size_t 上限）");
        return {};
    }

    // ── IR-C 录制拦截──────────────────────────────────
    // 录制图归属校验（唯一入口）：同一录制段内的表达式必须经**同一**引擎求值。
    // owner 是 shared_ptr：录制图存在期间引擎必然存活。
    [[nodiscard]] Result<void> check_graph_owner_(const ExprGraph& g,
                                                  const char* what) const
    {
        if (!g.owner)
            NN_FAIL(std::string(what) + ": 录制图未绑定引擎");
        if (g.owner.get() != this)
            NN_FAIL(std::string(what) +
                    ": 录制段内跨引擎求值（段内表达式必须经开启录制的引擎）");
        return {};
    }

    // begin_expr/end_expr 录制段内，eval_expr/eval_expr_reduce 不求值：表达式
    // 入图（add_node 建依赖边），返回带 virtual_tag 的占位 Tensor（真实存储、
    // 出生即绑定——end_expr 的复合 kernel 经 output_override 直接写回该存储，
    // 调用方持有的句柄即物化结果）。占位精度 = P（输出存储精度）。
    [[nodiscard]] Result<Tensor> record_graph_node_(
        ExprGraph& g, const ExprSpec& spec, std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, bool vector_out, Precision P)
    {
        NN_TRY_CHECK(check_graph_owner_(g, "eval_expr"));
        const int node = g.add_node(spec, inputs, rows, cols, vector_out, P);
        std::size_t orows = rows, ocols = cols;
        if (vector_out)
        {
            const int raxis = expr_spec_reduce_axis(spec);
            orows = (raxis == 0) ? rows : 1;
            ocols = (raxis == 1) ? cols : 1;
        }
        // RV-02：工厂错误上抛（分配失败原因不再被吞成空张量）
        NN_TRY(t_r, try_create_tensor(orows, ocols, P));
        Tensor t = std::move(*t_r);
        t.set_virtual_tag(g.tag_of_node[static_cast<std::size_t>(node)]);
        g.node_outputs[node] = t;
        return t;
    }

    // 目标传递的录制节点（P2 写穿物化）：输出 = dst 既有张量（无占位）。
    [[nodiscard]] Result<void> record_graph_into_node_(
        ExprGraph& g, const ExprSpec& spec, std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out)
    {
        NN_TRY_CHECK(check_graph_owner_(g, "eval_expr_into"));
        const int node = g.add_node(spec, inputs, rows, cols,
                                    /*vector_out=*/false, out.precision(),
                                    // RV-03：记录写目标存储 —— 同目标的多次写
                                    // （AdamW 的 decay→update 等）据此建显式顺序边
                                    out.storage_id());
        g.node_outputs[node] = out;   // 输出 = dst 张量（end_expr 写回其存储）
        return {};
    }

    // ── InitSpec host 填数（M2）：seed 与创建序号混流 ─────────────────────
    // splitmix64 定点混淆：序号按模型构造顺序推进（每引擎独立、进程内从 0
    // 起）→ 跨进程同序 → 初值逐字节确定；同 seed 的不同张量（同形状多层）
    // 不撞流。单线程顺序填充，无并行不确定性（铁律 #8）。
    static std::uint64_t mix_init_seed_(std::uint64_t seed, std::uint64_t seq)
    {
        std::uint64_t z = seed + 0x9E3779B97F4A7C15ull * (seq + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    static void fill_host_(Matrix& m, const InitSpec& spec, std::uint64_t seq)
    {
        std::mt19937_64 rng(mix_init_seed_(spec.seed, seq));
        auto sp = m.span();
        switch (spec.kind)
        {
        case InitSpec::Kind::Constant:
            std::fill(sp.begin(), sp.end(), spec.value);
            break;
        case InitSpec::Kind::Uniform:
        {
            std::uniform_real_distribution<Scalar> dist(spec.lo, spec.hi);
            for (std::size_t i = 0; i < sp.size(); ++i) sp[i] = dist(rng);
            break;
        }
        case InitSpec::Kind::Normal:
        {
            std::normal_distribution<Scalar> dist(spec.mean, spec.stddev);
            for (std::size_t i = 0; i < sp.size(); ++i) sp[i] = dist(rng);
            break;
        }
        default:
            std::fill(sp.begin(), sp.end(), Scalar{0});
            break;
        }
    }

    // ── 设备域检查（生命周期 v3，docs/todo/28-lifecycle-v3.md §4.3）──────
    // 每个 public NVI 入口逐个操作数比较 `Tensor::device_domain_()` 与本引擎
    // 域：空/无效张量（域 = 0）豁免（交由各入口既有的无效/形状检查拒绝）；
    // 其余必须**等于本引擎域**。跨域入口只有 `import`（CPU→GPU 上载）与
    // `to_matrix`（GPU→CPU 落宿主）。
    // 错误带调用点 source_location（默认参在调用处求值）与张量形状。
    [[nodiscard]] Error domain_error_(const Tensor& t, std::source_location loc) const
    {
        std::string s = "设备域不匹配 at " + std::string(loc.file_name()) + ":"
                      + std::to_string(loc.line()) + "  " + t.shape_str()
                      + "  tensor_domain=" + std::to_string(t.device_domain_())
                      + " engine_domain=" + std::to_string(domain_);
        const DomainId d = t.device_domain_();
        if (d == kHostDomain && domain_ != kHostDomain)
            s += "\n  CPU 张量不能直接进入 GPU 引擎：请先经 engine.import(t) 上载";
        else if (d != kHostDomain && domain_ == kHostDomain)
            s += "\n  GPU 张量不能直接进入 CPU 引擎：请先经 engine.to_matrix(t) 落宿主";
        else
            s += "\n  跨 GPU backend：请先 to_matrix 落宿主，再 import 到目标 backend";
        return Error{std::move(s)};
    }

    [[nodiscard]] Result<void> bind_check_one_(const Tensor& t,
                                               std::source_location loc) const
    {
        const DomainId d = t.device_domain_();
        if (d == 0 || d == domain_)
            return {};
        NN_FAIL(domain_error_(t, loc));
    }

    [[nodiscard]] Result<void> bind_check_(std::initializer_list<const Tensor*> ts,
                                           std::source_location loc = std::source_location::current()) const
    {
        for (const Tensor* t : ts)
        {
            if (!t)
                continue;
            if (auto r = bind_check_one_(*t, loc); !r)
                return r;
        }
        return {};
    }

    [[nodiscard]] Result<void> bind_check_(std::span<const Tensor> ts,
                                           std::source_location loc = std::source_location::current()) const
    {
        for (const Tensor& t : ts)
            if (auto r = bind_check_one_(t, loc); !r)
                return r;
        return {};
    }

    // ══════════════════════════════════════════════════════════════════════
    // 边界 cast 内部辅助（原 PrecisionEngine 私有工具，随下沉迁入基类）
    // ══════════════════════════════════════════════════════════════════════

    // 全部 f32？（快速直通判定：不物化任何中间量）
    [[nodiscard]] static bool all_f32(std::span<const Tensor> ts) noexcept
    {
        for (const auto& t : ts)
            if (t.precision() != Precision::F32)
                return false;
        return true;
    }

    // 抬到 f32：已是 f32 → 共享所有权直通（零拷贝）；f16 → 引擎 cast
    // loc：调用点（source_location 默认参在**调用处**求值 → 归因表能落到
    // 具体算子行号，而非统一落在本函数）。
    [[nodiscard]] Result<Tensor> to_f32(const Tensor& t,
                                        std::source_location loc = std::source_location::current())
    {
        if (t.precision() == Precision::F32)
            return t;
        note_temp_(t.rows(), t.cols(), /*to_f32=*/true,
                   static_cast<std::uint32_t>(loc.line()));
        return cast(t, Precision::F32);
    }

    // 落回目标精度：目标 f32 或已同精度 → 原样返回
    [[nodiscard]] Result<Tensor> to_prec(Tensor t, Precision P,
                                         std::source_location loc = std::source_location::current())
    {
        if (P == Precision::F32 || t.precision() == P)
            return t;
        note_temp_(t.rows(), t.cols(), /*to_f32=*/false,
                   static_cast<std::uint32_t>(loc.line()));
        // 入参常是 `*_impl` 的产物（unary_/binary_/move_ 等）。v3 起设备域
        // 从存储派生，基类内部中转无需补任何身份；cast 的 bind_check_ 直接
        // 比较存储域与本引擎域。
        return cast(std::move(t), P);
    }

    // 一批操作数抬到 f32（顺序与输入一致；ExprSpec 的 views 依赖该顺序）
    [[nodiscard]] Result<std::vector<Tensor>> to_f32_all(
        std::span<const Tensor> ts,
        std::source_location loc = std::source_location::current())
    {
        std::vector<Tensor> out;
        out.reserve(ts.size());
        for (const auto& t : ts)
        {
            auto c = to_f32(t, loc);
            NN_TRY_CHECK(c);
            out.push_back(std::move(*c));
        }
        return out;
    }

    // in-place 双操作数通用：抬 f32 → 算 → 必要时 cast_into 写回 A 的原存储
    // loc：转发给 to_f32/to_prec（归因落到调用本模板的算子行）。
    template <typename Fn>
    [[nodiscard]] Result<void> inplace2_(Tensor& A, const Tensor& B, Fn&& fn,
                                         std::source_location loc = std::source_location::current())
    {
        if (A.precision() == Precision::F32 && B.precision() == Precision::F32)
            return fn(A, B);
        // 纯边界 cast 助手（不做能力分支）：需要"原生 f16 直通"的引擎在自己
        // 的 `*_impl` 里先处理 f16，再落到这里（或根本不进来）。
        auto b = to_f32(B, loc);
        NN_TRY_CHECK(b);
        if (A.precision() == Precision::F32)
            return fn(A, *b);
        auto a = to_f32(A, loc);
        NN_TRY_CHECK(a);
        auto r = fn(*a, *b);
        NN_TRY_CHECK(r);
        return cast_into(*a, A);   // 存储精度不可变（§8.3）
    }

    // in-place 单操作数通用
    template <typename Fn>
    [[nodiscard]] Result<void> inplace1_(Tensor& A, Fn&& fn,
                                         std::source_location loc = std::source_location::current())
    {
        if (A.precision() == Precision::F32)
            return fn(A);
        auto a = to_f32(A, loc);
        NN_TRY_CHECK(a);
        auto r = fn(*a);
        NN_TRY_CHECK(r);
        return cast_into(*a, A);
    }

    // 运算类通用：抬 f32 → 算 → 按 P 落回（单操作数）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> unary_(const Tensor& A, Precision P, Fn&& fn,
                                        std::source_location loc = std::source_location::current())
    {
        if (P == Precision::F32 && A.precision() == Precision::F32)
            return fn(A);
        auto a = to_f32(A, loc);
        NN_TRY_CHECK(a);
        auto r = fn(*a);
        NN_TRY_CHECK(r);
        return to_prec(std::move(*r), P, loc);
    }

    // 运算类通用：抬 f32 → 算 → 按 P 落回（双操作数）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> binary_(const Tensor& A, const Tensor& B,
                                         Precision P, Fn&& fn,
                                         std::source_location loc = std::source_location::current())
    {
        if (P == Precision::F32 && A.precision() == Precision::F32 &&
            B.precision() == Precision::F32)
            return fn(A, B);
        auto a = to_f32(A, loc);
        NN_TRY_CHECK(a);
        auto b = to_f32(B, loc);
        NN_TRY_CHECK(b);
        auto r = fn(*a, *b);
        NN_TRY_CHECK(r);
        return to_prec(std::move(*r), P, loc);
    }

    // 数据搬运类通用：输出精度 = 源精度（§8.4，无 P 形参）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> move_(const Tensor& A, Fn&& fn,
                                       std::source_location loc = std::source_location::current())
    {
        const Precision out_p = A.precision();
        if (out_p == Precision::F32)
            return fn(A);
        auto a = to_f32(A, loc);
        NN_TRY_CHECK(a);
        auto r = fn(*a);
        NN_TRY_CHECK(r);
        return to_prec(std::move(*r), out_p, loc);
    }

    // 不同精度行插入（供引擎 `insert_rows_impl` 在非 f32 时复用）：在 f32 空间
    // 插入后写回 dst 的原存储（§8.3 in-place 存储精度不可变）。
    [[nodiscard]] Result<void> insert_rows_cast_(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src)
    {
        auto d = to_f32(dst);
        NN_TRY_CHECK(d);
        auto s = to_f32(src);
        NN_TRY_CHECK(s);
        auto r = insert_rows_impl(*d, dst_start_row, *s);
        NN_TRY_CHECK(r);
        if (dst.precision() == Precision::F32)
            return {};
        return cast_into(*d, dst);
    }

    // 不同精度 scatter-add（供引擎 `scatter_add_rows_impl` 在非 f32 时复用）。
    [[nodiscard]] Result<void> scatter_add_rows_cast_(
        Tensor& dst, const Tensor& indices, const Tensor& grad)
    {
        auto d = to_f32(dst);
        NN_TRY_CHECK(d);
        auto g = to_f32(grad);
        NN_TRY_CHECK(g);
        auto r = scatter_add_rows_impl(*d, indices, *g);
        NN_TRY_CHECK(r);
        if (dst.precision() == Precision::F32)
            return {};
        return cast_into(*d, dst);
    }

    // ── 多精度变体发现 / 边界 cast 归因（诊断；NN_PREC_TRACE=1）────────────
    // trace_variant_：打印每个 (结构 key, 精度签名) —— 这就是 in-kernel f16
    // 需要生成的变体集合。本层是所有原语调用的必经点、能看到每次调用的
    // **真实输入精度**，故变体发现放在这里（scan 的 f16 dry-run 是另一条同源路径）。
    // note_temp_：记录一次"物化临时量"（形状 + 调用点行号 → 次数/字节）；见
    // dump_temp_stats()。
    // line = source_location 行号（区分同一形状来自哪个算子路径——只有形状
    // 时 matmul/unary_/eval_expr 的 cast 无法分辨）。
    // 探针门：NN_PROBES_ENABLED=0（默认）时 note_temp_ 是空实现、
    // prec_env_flag 恒 false —— 逐算子归因与变体日志**整块不编译进二进制**。
    static void note_temp_(std::size_t rows, std::size_t cols, bool to_f32,
                           std::uint32_t line = 0)
    {
#if !NN_PROBES_ENABLED
        (void)rows; (void)cols; (void)to_f32; (void)line;
        return;
#else
        if (!prec_trace_enabled_())
            return;
        const std::size_t bytes = rows * cols * (to_f32 ? 4u : 2u);
        for (auto& e : temp_stats())
            if (e.rows == rows && e.cols == cols && e.to_f32 == to_f32 && e.line == line)
            {
                e.count++;
                e.bytes += bytes;
                return;
            }
        temp_stats().push_back(TempStat{rows, cols, to_f32, 1, bytes, line});
#endif
    }

    [[nodiscard]] static bool prec_trace_enabled_()
    {
        // 诊断变体归因开关：探针门下的**读一次 + 缓存**（core_env 已含门控）。
        static const bool on = ::nn::detail::probe_flag_raw("NN_PREC_TRACE");
        return on;
    }

    // 通用诊断开关（RV-09：实现收敛到 core_env.hpp，不再各写 getenv/_dupenv_s）。
    // 每次读取、零缓存（与 dsl::env_flag 同语义）；探针门：默认构建恒 false
    // （**不求值 getenv**），"NN_F16_DEBUG" 等诊断开关失效。
    [[nodiscard]] static bool prec_env_flag(const char* name)
    {
        return ::nn::detail::probe_flag_raw(name);
    }

    // 诊断（NN_PREC_TRACE=1）：请求了非零精度签名却**没有命中带类型变体** —— 这是
    // "静默回退边界 cast"的唯一可见信号，也是下一期该补哪些变体的清单（去重打印）。
    static void trace_miss_(const ExprSpec& spec,
                            std::span<const Tensor> inputs, Precision P)
    {
        static std::unordered_set<std::string> seen;
        const ExprSpec canon = canonicalize_expr_spec(spec);
        const std::string k = expr_spec_key(canon);
        const ExprPrecSig sig = expr_prec_sig_of(inputs, P);
        const std::string vk = expr_prec_sig_key(k, sig);
        if (!seen.insert(vk).second)
            return;
        const int raxis = expr_spec_reduce_axis(canon);
        std::fprintf(stderr,
                     "[prec][miss] key=%s sig=0x%04x %s instrs=%zu mm=%d fold=%d raxis=%d\n",
                     k.c_str(), static_cast<unsigned>(sig),
                     expr_prec_sig_str(sig, inputs.size()).c_str(),
                     canon.instrs.size(), canon.matmul ? 1 : 0,
                     canon.fold ? 1 : 0, raxis);
    }
    static void trace_variant_(const ExprSpec& spec,
                               std::span<const Tensor> inputs, Precision P)
    {
        if (!prec_trace_enabled_())
            return;
        static std::unordered_set<std::string> seen;
        const std::string k = expr_spec_key(canonicalize_expr_spec(spec));
        const ExprPrecSig sig = expr_prec_sig_of(inputs, P);
        const std::string vk = expr_prec_sig_key(k, sig);
        if (!seen.insert(vk).second)
            return;
        std::fprintf(stderr,
                     "[prec] key=%s sig=0x%04x %s n_in=%zu instrs=%zu mm=%d fold=%d\n",
                     k.c_str(), static_cast<unsigned>(sig),
                     expr_prec_sig_str(sig, inputs.size()).c_str(),
                     inputs.size(), spec.instrs.size(),
                     spec.matmul ? 1 : 0, spec.fold ? 1 : 0);
    }

    // 默认（CPU / 同步引擎）标量回读槽：submit 立即存值，poll 立即就绪。
    // GPU 引擎覆写为异步槽位（见 GpuEngine / GpuBackend::rb_slots_）。
    std::vector<Scalar> sync_readback_slots_;

    // InitSpec 分布创建序号（M2）：每个引擎独立从 0 起，host 填数时混流用。
    std::uint64_t init_seq_ = 0;

private:
    // 设备域 token（v3）：派生类构造时写入；入口检查据此拒绝跨域操作数。
    // 0 = 未设置（仅基类抽象态，派生引擎必须给非零域）。
    DomainId domain_ = 0;
};

namespace detail {

// ── L2+ 宿主桥（docs/todo/17 §3 D11 / §4.6，AGENTS.md 铁律 #12）────
// L2（Layer/Loss/Optimizer/Model）计算路径**不得出现 Matrix 类型**，因此层自算
// 的辅助数据（索引/位置/掩码/斜率/编码表等——没有对应的引擎侧生成原语）以 span
// 形态经这三个函数进出；它们内部走 M3 的批量 `write` / `read`（GPU 自动 staging、
// 隐含 flush + 同步），不新增任何原语，也不进引擎虚表（49 个虚函数骨架不变）。
//
//   upload_span  ：标量缓冲 → 新建 (rows, cols, P) 张量。f16 目标按
//                  round-half-to-even 舍入，与 `from_matrix(m, P)` 逐位同口径
//                  ——它是 L2 里 `from_matrix` 的 span 形态对应物。
//   download_span：张量 → 既有标量缓冲；f16 存储先升 f32，与
//                  `to_matrix(t, F32)` 同值（升 cast 精确无损）。
//   download_vector：download_span 的取值便捷形态（返回值语义缓冲）。
//
// 适用范围仅限**层自算的小规模辅助数据**；数据集、预训练权重、对拍/落盘等大批量
// I/O 仍走 I/O 层的 `from_matrix` / `to_matrix`（那是宿主 I/O 载体的职责）。
[[nodiscard]] inline Result<Tensor> upload_span(ComputeEngine& engine, std::size_t rows,
                                                std::size_t cols, Precision P,
                                                std::span<const Scalar> src)
{
    if (src.size() != rows * cols)
        NN_FAIL("upload_span: 元素数与形状不匹配");
    // RV-02：工厂返回 Result —— 分配失败原因原样上抛（含 GPU 侧原因）
    NN_TRY(t_r, engine.try_create_tensor(rows, cols, P));
    Tensor t = std::move(*t_r);
    if (P == Precision::F32)
    {
        // fresh_dst=true：t 刚创建、无任何已录/在飞命令引用 → GPU 侧跳过
        // 录制窗口排空（见 copy_from 契约）。本函数是库内唯一承诺方。
        NN_TRY(r, engine.write(t, std::span<const Scalar>(src), /*fresh_dst=*/true));
        return t;
    }
    std::vector<f16> buf(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        buf[i] = f16{src[i]};   // RHE 舍入，与 from_matrix 的 f32→f16 同路径
    NN_TRY(r, engine.write(t, std::span<f16>(buf), /*fresh_dst=*/true));
    return t;
}

[[nodiscard]] inline Result<void> download_span(ComputeEngine& engine, const Tensor& t,
                                                std::span<Scalar> dst)
{
    if (dst.size() != t.rows() * t.cols())
        NN_FAIL("download_span: 元素数与张量形状不匹配");
    if (t.precision() == Precision::F32)
        return engine.read(t, dst);
    std::vector<f16> buf(dst.size());
    NN_TRY(r, engine.read(t, std::span<f16>(buf)));
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = static_cast<Scalar>(buf[i]);
    return {};
}

[[nodiscard]] inline Result<std::vector<Scalar>> download_vector(ComputeEngine& engine,
                                                                const Tensor& t)
{
    std::vector<Scalar> v(t.rows() * t.cols());
    NN_TRY(r, download_span(engine, t, std::span(v)));
    return v;
}

} // namespace detail

// ── IR-C 录制段 RAII 守卫（P2）───────────────────────────────
// 作用域退出（含 NN_TRY 错误早退）自动 end_expr 收口，防录制图悬挂；
// 正常路径显式 end() 收口并把 end_expr 的错误照常传播（守卫只兜底）。
// 持**强引用**：录制段存活期内引擎不会被销毁（原裸引用在作用域内释放最后
// 一个引擎句柄时会在析构里解引用已销毁对象 = UAF）。
class ExprSegment
{
public:
    explicit ExprSegment(std::shared_ptr<ComputeEngine> e) noexcept
        : e_(std::move(e))
    {
        if (!e_)
            NN_FATAL("ExprSegment: 引擎句柄为空（必须由 shared_ptr 管理）");
    }
    ExprSegment(const ExprSegment&) = delete;
    ExprSegment& operator=(const ExprSegment&) = delete;
    ~ExprSegment()
    {
        if (!done_)
            (void)e_->end_expr();   // 兜底收口（错误已被上层传播或忽略）
    }
    [[nodiscard]] Result<void> end()
    {
        done_ = true;
        return e_->end_expr();
    }

private:
    std::shared_ptr<ComputeEngine> e_;
    bool done_ = false;
};

} // namespace nn
