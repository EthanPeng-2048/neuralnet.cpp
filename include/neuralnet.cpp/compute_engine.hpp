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
//   - **边界 cast 层（NVI 公共入口）**：f16 无变体算子的"抬 f32 算 → 按 P
//     落回"逻辑住在基类非虚入口里（原 PrecisionEngine 装饰器已下沉删除，
//     见 docs/development/15-computeengine-refresh.md §4.1）：基类问能力
//     （supports_native_* / supports_expr_precision_variant）→ 基类决定
//     cast → 引擎只实现 `*_impl`。全 f32 配置下每个入口都是快速直通分支，
//     与原生引擎逐字节一致。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <source_location>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core_config.hpp"
#include "core_errors.hpp"
#include "compute_tensor.hpp"
#include "expr_opt.hpp"   // canonicalize_expr_spec（trace 归因用，与 gen_fused 同源 key）
#include "expr_spec.hpp"

namespace nn
{

namespace dsl
{
    // 前置声明（compute_engine.hpp 不能反向 include expr_dsl.hpp）：
    // nn::dsl::compute / compute_reduce 是 ComputeEngine::adopt 的 friend
    // 调用方——DSL 静态工厂产物的库内 stamp 通道（15 §4.2 D2）。
    // 默认实参（P = F32）在 expr_dsl.hpp 的定义处。
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

// ══════════════════════════════════════════════════════════════════════════
// ComputeEngine — 计算引擎抽象接口
// ══════════════════════════════════════════════════════════════════════════
//
// 分层（NVI：Non-Virtual Interface）：
//   public  非虚入口 = 边界 cast 层（原 PrecisionEngine 逻辑）：
//           变体优先 / 全 f32 快速直通 / 无变体时"抬 f32 → *_impl → 按 P 落回"。
//   protected 虚 `*_impl` = 引擎原生实现（只管算，不管精度边界）。
//   能力查询（supports_*）：基类决策用的虚函数，CPU 默认 false / GPU 覆盖。
class ComputeEngine
{
public:
    virtual ~ComputeEngine() = default;

    // ── 设备查询 ──────────────────────────────────────────────────────────
    [[nodiscard]] virtual Device device() const noexcept = 0;

    // ── 批处理控制 ────────────────────────────────────────────────────────
    // CPU 引擎：no-op（操作立即同步执行）
    // GPU 引擎：begin 开始录制，end 统一提交 + fence wait
    [[nodiscard]] virtual Result<void> begin_batch() = 0;
    [[nodiscard]] virtual Result<void> end_batch() = 0;

    // ── 批处理中点刷新（防 TDR）────────────────────────────────────────────
    // GPU 引擎：提交当前 command buffer 并等待完成，然后自动开始新的录制。
    // 可在 forward 与 backward 之间调用，将一次大提交拆分为多次小提交，
    // 避免单次提交时间过长触发 Windows TDR。
    // CPU 引擎：no-op。
    [[nodiscard]] virtual Result<void> flush_batch() { return {}; }

    // ── 显存回收（L2）─────────────────────────────────────────────────
    // GPU 引擎：在 end_batch（提交完成、延迟销毁已 flush）之后归还完全
    // 空闲的内存池底材给 GPU。CPU 引擎：no-op。
    [[nodiscard]] virtual Result<void> release_idle_pool_blocks() { return {}; }

    // ── 显存池统计（L2 仪器化）──────────────────────────────────────
    // GPU 引擎返回池统计字符串（块数/占用/空闲/碎片）；CPU 引擎返回空。
    // 用于训练中显存采样与逐项归因。
    [[nodiscard]] virtual std::string pool_stats() const { return {}; }

    // ── 激活 offload（L1-offload）───────────────────────────────────
    // ── activation offload slab（L1-offload，持久复用缓冲） ────────────
    // 每个 GPTBlock / RAPTBlock 持有一块持久 host-visible slab，所有激活按
    // float 偏移写入/读出，跨 step 复用 → RAM = 激活实际体积（避免每 tensor
    // 独立 128MB 块导致的碎片膨胀）。CPU 引擎 no-op（开启只会得到 1×1 张量）。
    [[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes)
    {
        return stamp_(create_offload_buffer_impl(bytes));
    }
    // 把 src 复制到 buffer 的 offset（float 单位）处
    // ── 边界 cast 入口：f16 激活在写入 slab 前抬到 f32（slab 恒 f32 存）──
    // restore 出来是 f32，后续运算由入口层适配（正确性不受影响，代价是该份
    // 激活按 f32 存、不享 f16 存储折半）。
    [[nodiscard]] Result<void> offload_save(
        const Tensor& buffer, std::size_t offset, const Tensor& src)
    {
        if (auto ec = bind_check_({&buffer, &src}); !ec)
            return std::unexpected(ec.error());
        auto s = to_f32(src);
        if (!s)
            return std::unexpected(s.error());
        return offload_save_impl(buffer, offset, *s);
    }
    // 从 buffer 的 offset（float 单位）处复制 rows×cols 到新 GPU tensor
    [[nodiscard]] Result<Tensor> offload_restore(
        const Tensor& buffer, std::size_t offset,
        std::size_t rows, std::size_t cols)
    {
        return stamp_(offload_restore_impl(buffer, offset, rows, cols));
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
        if (auto ec = bind_check_({&t}); !ec)
            return std::unexpected(ec.error());
        auto s = to_f32(t);
        if (!s)
            return std::unexpected(s.error());
        return submit_scalar_readback_impl(slot, *s);
    }

    [[nodiscard]] virtual Result<bool> poll_scalar_readback(
        std::size_t slot, Scalar& out)
    {
        if (slot >= sync_readback_slots_.size())
            return false;
        out = sync_readback_slots_[slot];
        return true;
    }

    [[nodiscard]] virtual std::size_t scalar_readback_slots() const { return 1; }

    // ── 张量工厂（统一接口，§6.4, §6.5）────────────────────────────────
    // P 由调用方显式指定（§8.5）：无隐式推导，无 Auto
    // P1（docs/development/15 §3.1）：公共非虚入口在尾部统一 stamp 出生绑定，
    // 引擎只实现 create_tensor_impl / from_matrix_impl（protected）——避免
    // 每个 override 各自漏 stamp。to_matrix 输出 Matrix（宿主），不参与 stamp。
    [[nodiscard]] Tensor create_tensor(std::size_t rows, std::size_t cols, Precision P = Precision::F32)
    {
        return stamp_(create_tensor_impl(rows, cols, P));
    }
    [[nodiscard]] Result<Tensor> from_matrix(const Matrix& m, Precision P = Precision::F32)
    {
        return stamp_(from_matrix_impl(m, P));
    }
    [[nodiscard]] virtual Result<Matrix> to_matrix(const Tensor& t, Precision P = Precision::F32) = 0;

    // ── cast 原语（§7.5，唯一"变精度"算子，永远显式）────────────────
    // 升 cast（f16→f32）精确无损；降 cast（f32→f16）round-half-to-even。
    // 默认实现：同精度 = 返回 src（共享所有权，零拷贝）；跨精度 = 错误（引擎覆盖）。
    [[nodiscard]] Result<Tensor> cast(const Tensor& src, Precision dst)
    {
        if (auto ec = bind_check_({&src}); !ec)
            return std::unexpected(ec.error());
        return stamp_(cast_impl(src, dst));
    }

    // ── cast_into：把 src 按精度转换后写入 **dst 的既有存储**（不替换对象）
    // 与 cast 的区别是"落点"：cast 返回新张量，cast_into 保留 dst 的
    // 对象身份与底层 buffer（in-place 语义必需——边界 cast 做 f16 原地更新时，
    // 若替换 dst 对象，其它持有同一张量句柄的缓存会静默失联）。
    // 要求 rows/cols 一致。默认实现：同精度走 copy_into，跨精度报错。
    [[nodiscard]] virtual Result<void> cast_into(const Tensor& src, Tensor& dst)
    {
        if (src.rows() != dst.rows() || src.cols() != dst.cols())
            return std::unexpected(Error{"cast_into: shape mismatch"});
        if (src.precision() == dst.precision())
            return copy_into(dst, src);
        return std::unexpected(Error{"cast_into: 该引擎不支持跨精度转换"});
    }

    // ── copy_into：同精度同形状拷贝（dst 的既有存储被完整覆盖，不替换对象）
    // 用于 cast_into 的同精度分支与边界 cast 的"写回原存储"路径。
    [[nodiscard]] virtual Result<void> copy_into(Tensor& dst, const Tensor& src)
    {
        (void)dst; (void)src;
        return std::unexpected(Error{"copy_into: 该引擎未实现"});
    }

    // ══════════════════════════════════════════════════════════════════════
    // 边界 cast 层（NVI 公共入口）—— 原 PrecisionEngine 逻辑
    //
    // 一句话：**f16 只改变"数据存哪儿"，不改变"算子怎么算"**——算术始终以
    // f32 为参考精度，f16 的精度边界处理全部集中在本层入口。每个入口先问
    // 能力查询（supports_native_data_move / supports_native_f16_reduce /
    // supports_expr_precision_variant），没有原生路径才走边界 cast：
    //   变体优先：引擎有带类型融合 shader → 直接把 f16 原张量交给 *_impl
    //           （读 f16 / 写 f16，算术 f32），零边界临时量；
    //   边界 cast：入 = 把 f16 操作数抬到 f32（已是 f32 则零拷贝直通），
    //           算 = 调用 *_impl 的既有 f32 实现（零分歧），
    //           出 = 结果按目标精度落回（f16 写出 = round-half-to-even）。
    // in-place 入口用 `cast_into` 写回**原存储**：保留张量对象身份与底层
    // buffer（§8.3：in-place 存储精度不可变）。
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
        if (auto ec = bind_check_({&src}); !ec)
            return std::unexpected(ec.error());
        // 原生 f16 数据搬运：引擎的 clone 是模板化字节拷贝 → 直接放行，
        // 省掉"抬 f32 → 拷贝 → 落回 f16"的 2 份全尺寸临时量。
        if (src.precision() != Precision::F32 && supports_native_data_move())
            return stamp_(clone_impl(src));
        return stamp_(move_(src, [this](const Tensor& s) { return clone_impl(s); }));
    }

    // 将 CPU Matrix 数据写入已有 Tensor（CPU 拷贝 / GPU 上传）
    // 用于序列化加载、Optimizer 参数写回等场景。
    // f16 目标：先按 f32 上传，再 cast_into 写进 dst 的原存储
    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src)
    {
        if (auto ec = bind_check_({&dst}); !ec)
            return std::unexpected(ec.error());
        if (dst.precision() == Precision::F32)
            return copy_from_impl(dst, src);
        auto t = from_matrix(src, Precision::F32);
        if (!t)
            return std::unexpected(t.error());
        return cast_into(*t, dst);
    }

    // ── 行切片原语（op-level 数据操作，不含算法语义） ──────────────────
    // 返回 src 的行 [start_row, start_row + count) 的连续拷贝。
    // 用于多头注意力中 per-head Q/K/V 切片等场景。
    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count)
    {
        if (auto ec = bind_check_({&src}); !ec)
            return std::unexpected(ec.error());
        if (src.precision() != Precision::F32 && supports_native_data_move())
            return stamp_(slice_rows_impl(src, start_row, count));
        return stamp_(move_(src, [this, start_row, count](const Tensor& s)
        {
            return slice_rows_impl(s, start_row, count);
        }));
    }

    // 将 src 的所有行写入 dst 的行 [dst_start_row, dst_start_row + src.rows())。
    // 真·就地修改（GPU 用 vkCmdCopyBuffer with dstOffset）。
    // 用于多头注意力中 per-head 输出拼接等场景。
    // insert_rows：就地写入 dst 的行区间 → dst 精度即目标精度
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src)
    {
        if (auto ec = bind_check_({&dst, &src}); !ec)
            return std::unexpected(ec.error());
        if (dst.precision() != Precision::F32 && dst.precision() == src.precision() &&
            supports_native_data_move())
            return insert_rows_impl(dst, dst_start_row, src);
        if (dst.precision() == Precision::F32 && src.precision() == Precision::F32)
            return insert_rows_impl(dst, dst_start_row, src);
        // 不同精度（或 f16 dst）：统一在 f32 空间插入后写回 dst 的原存储
        auto d = to_f32(dst);
        if (!d)
            return std::unexpected(d.error());
        auto s = to_f32(src);
        if (!s)
            return std::unexpected(s.error());
        auto r = insert_rows_impl(*d, dst_start_row, *s);
        if (!r)
            return std::unexpected(r.error());
        if (dst.precision() == Precision::F32)
            return {};
        return cast_into(*d, dst);
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
        if (auto ec = bind_check_({&table, &indices}); !ec)
            return std::unexpected(ec.error());
        return stamp_(move_(table, [this, &indices](const Tensor& t)
        {
            return gather_rows_impl(t, indices);
        }));
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
        if (auto ec = bind_check_({&dst, &indices, &grad}); !ec)
            return std::unexpected(ec.error());
        if (dst.precision() == Precision::F32 && grad.precision() == Precision::F32)
            return scatter_add_rows_impl(dst, indices, grad);
        // 原生 f16（GPU：打包 half CAS 变体；无 pipeline 时引擎内 cast 回退）
        // → 直通，不为 dst/grad 物化 f32 副本。CPU 能力 false → 边界 cast。
        if (supports_native_data_move())
            return scatter_add_rows_impl(dst, indices, grad);
        auto d = to_f32(dst);
        if (!d)
            return std::unexpected(d.error());
        auto g = to_f32(grad);
        if (!g)
            return std::unexpected(g.error());
        auto r = scatter_add_rows_impl(*d, indices, *g);
        if (!r)
            return std::unexpected(r.error());
        if (dst.precision() == Precision::F32)
            return {};
        return cast_into(*d, dst);
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
        if (auto ec = bind_check_({&x}); !ec)
            return std::unexpected(ec.error());
        return stamp_(move_(x, [this, M, B, N, inverse](const Tensor& t)
        {
            return rearrange_3d_impl(t, M, B, N, inverse);
        }));
    }

    // ── 矩阵转置：A (R, C) → out (C, R) ──
    // 纯 layout 操作，零算法语义。用于 embedding 列布局转换等场景。
    [[nodiscard]] Result<Tensor> transpose(const Tensor& A)
    {
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        return stamp_(move_(A, [this](const Tensor& t) { return transpose_impl(t); }));
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
        if (auto ec = bind_check_({&x}); !ec)
            return std::unexpected(ec.error());
        return stamp_(move_(x, [this, C, H, W, k, stride, pad, OH, OW](const Tensor& t)
        {
            return im2col_impl(t, C, H, W, k, stride, pad, OH, OW);
        }));
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
        if (auto ec = bind_check_({&col}); !ec)
            return std::unexpected(ec.error());
        return stamp_(move_(col, [this, C, H, W, k, stride, pad, OH, OW](const Tensor& t)
        {
            return col2im_impl(t, C, H, W, k, stride, pad, OH, OW);
        }));
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
        if (auto ec = bind_check_({&A, &B}); !ec)
            return std::unexpected(ec.error());
        if (P != Precision::F32)
            return stamp_(matmul_impl(A, B, transA, transB, P));
        return stamp_(binary_(A, B, P, [this, transA, transB](const Tensor& a, const Tensor& b)
        {
            return matmul_impl(a, b, transA, transB, Precision::F32);
        }));
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
        if (auto ec = bind_check_({&A, &B}); !ec)
            return std::unexpected(ec.error());
        if (P != Precision::F32)
            return stamp_(batched_matmul_impl(A, B, batch, transA, transB, alpha, P));
        return stamp_(binary_(A, B, P,
            [this, batch, transA, transB, alpha](const Tensor& a, const Tensor& b)
        {
            return batched_matmul_impl(a, b, batch, transA, transB, alpha,
                                       Precision::F32);
        }));
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
        if (auto ec = bind_check_({&A, &B, &bias}); !ec)
            return std::unexpected(ec.error());
        if (P != Precision::F32 &&
            (A.precision() != Precision::F32 || B.precision() != Precision::F32 ||
             bias.precision() != Precision::F32))
            return stamp_(matmul_with_bias_impl(A, B, bias, transA, transB, P));

        if (P == Precision::F32 && A.precision() == Precision::F32 &&
            B.precision() == Precision::F32 && bias.precision() == Precision::F32)
            return stamp_(matmul_with_bias_impl(A, B, bias, transA, transB,
                                         Precision::F32));
        auto a = to_f32(A);
        if (!a) return std::unexpected(a.error());
        auto b = to_f32(B);
        if (!b) return std::unexpected(b.error());
        auto bi = to_f32(bias);
        if (!bi) return std::unexpected(bi.error());
        auto r = matmul_with_bias_impl(*a, *b, *bi, transA, transB,
                                       Precision::F32);
        if (!r) return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), P));
    }

    // 梯度累加：dst += src（dst 存储精度不可变，§8.3）
    [[nodiscard]] Result<void> accumulate(Tensor& dst, const Tensor& src)
    {
        if (auto ec = bind_check_({&dst, &src}); !ec)
            return std::unexpected(ec.error());
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
        auto r = inplace2_(dst, src, [this](Tensor& d, const Tensor& s)
        {
            return add_inplace_impl(d, s);
        });
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
        if (auto ec = bind_check_({&A, &B}); !ec)
            return std::unexpected(ec.error());
        return inplace2_(A, B, [this](Tensor& a, const Tensor& b)
        {
            return add_inplace_impl(a, b);
        });
    }

    // A *= scalar
    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s)
    {
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        return inplace1_(A, [this, s](Tensor& a) { return scale_inplace_impl(a, s); });
    }

    // A = 0
    [[nodiscard]] Result<void> zero(Tensor& A)
    {
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        // 原生 f16 清零：fill_zero 是字节级原语（每步 zero_grad 调用 N 次）
        if (A.precision() != Precision::F32 && supports_native_data_move())
            return zero_impl(A);
        return inplace1_(A, [this](Tensor& a) { return zero_impl(a); });
    }

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
        if (auto ec = bind_check_({&K, &V, &P, &R, &A0, &B0, &boundary}); !ec)
            return std::unexpected(ec.error());
        if (prec == Precision::F32 && K.precision() == Precision::F32 &&
            V.precision() == Precision::F32 && P.precision() == Precision::F32 &&
            R.precision() == Precision::F32 && A0.precision() == Precision::F32 &&
            B0.precision() == Precision::F32 && boundary.precision() == Precision::F32)
            return stamp_(scan_prefix_outer_impl(K, V, P, R, A0, B0, has_state, dk, heads,
                                          causal, boundary, has_bnd,
                                          Precision::F32));
        if (supports_native_data_move())
        {
            auto r = scan_prefix_outer_impl(K, V, P, R, A0, B0, has_state, dk, heads,
                                            causal, boundary, has_bnd, prec);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), prec));
        }
        const std::vector<Tensor> ts{K, V, P, R, A0, B0, boundary};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = scan_prefix_outer_impl((*c)[0], (*c)[1], (*c)[2], (*c)[3],
                                        (*c)[4], (*c)[5], has_state, dk, heads,
                                        causal, (*c)[6], has_bnd, Precision::F32);
        if (!r) return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), prec));
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
        if (auto ec = bind_check_({&D, &X, &Y, &boundary}); !ec)
            return std::unexpected(ec.error());
        if (prec == Precision::F32 && D.precision() == Precision::F32 &&
            X.precision() == Precision::F32 && Y.precision() == Precision::F32 &&
            boundary.precision() == Precision::F32)
            return stamp_(scan_suffix_outer_impl(D, X, Y, dk, heads, causal, boundary,
                                          has_bnd, Precision::F32));
        if (supports_native_data_move())
        {
            auto r = scan_suffix_outer_impl(D, X, Y, dk, heads, causal, boundary,
                                            has_bnd, prec);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), prec));
        }
        const std::vector<Tensor> ts{D, X, Y, boundary};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = scan_suffix_outer_impl((*c)[0], (*c)[1], (*c)[2], dk, heads,
                                        causal, (*c)[3], has_bnd, Precision::F32);
        if (!r) return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), prec));
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
        if (auto ec = bind_check_({&P, &R, &S}); !ec)
            return std::unexpected(ec.error());
        if (prec == Precision::F32 && P.precision() == Precision::F32 &&
            R.precision() == Precision::F32 && S.precision() == Precision::F32)
            return stamp_(outer_col_impl(P, R, S, dk, has_scale, Precision::F32));
        if (supports_native_data_move())
        {
            auto r = outer_col_impl(P, R, S, dk, has_scale, prec);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), prec));
        }
        const std::vector<Tensor> ts{P, R, S};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = outer_col_impl((*c)[0], (*c)[1], (*c)[2], dk, has_scale,
                                Precision::F32);
        if (!r) return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), prec));
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
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        if (A.precision() != Precision::F32 && supports_native_f16_reduce())
        {
            auto r = row_reduce_sum_impl(A, P);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), P));
        }
        return stamp_(unary_(A, P, [this](const Tensor& a)
        {
            return row_reduce_sum_impl(a, Precision::F32);
        }));
    }

    // 按列求和：A (rows, cols) → out (1, cols)
    // out[c] = Σ_r A[r][c]
    [[nodiscard]] Result<Tensor> col_reduce_sum(
        const Tensor& A, Precision P = Precision::F32)
    {
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        if (A.precision() != Precision::F32 && supports_native_f16_reduce())
        {
            auto r = col_reduce_sum_impl(A, P);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), P));
        }
        return stamp_(unary_(A, P, [this](const Tensor& a)
        {
            return col_reduce_sum_impl(a, Precision::F32);
        }));
    }

    // 按列求最大值：A (rows, cols) → out (1, cols)
    // out[c] = max_r A[r][c]
    [[nodiscard]] Result<Tensor> col_reduce_max(
        const Tensor& A, Precision P = Precision::F32)
    {
        if (auto ec = bind_check_({&A}); !ec)
            return std::unexpected(ec.error());
        if (A.precision() != Precision::F32 && supports_native_f16_reduce())
        {
            auto r = col_reduce_max_impl(A, P);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), P));
        }
        return stamp_(unary_(A, P, [this](const Tensor& a)
        {
            return col_reduce_max_impl(a, Precision::F32);
        }));
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
        if (auto ec = bind_check_({&x}); !ec)
            return std::unexpected(ec.error());
        if (x.precision() != Precision::F32 && supports_native_f16_reduce())
        {
            auto r = grouped_reduce_sum_impl(x, G, R, P);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), P));
        }
        return stamp_(unary_(x, P, [this, G, R](const Tensor& t)
        {
            return grouped_reduce_sum_impl(t, G, R, Precision::F32);
        }));
    }
    [[nodiscard]] Result<Tensor> grouped_reduce_max(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision P = Precision::F32)
    {
        if (auto ec = bind_check_({&x}); !ec)
            return std::unexpected(ec.error());
        if (x.precision() != Precision::F32 && supports_native_f16_reduce())
        {
            auto r = grouped_reduce_max_impl(x, G, R, P);
            if (!r) return std::unexpected(r.error());
            return stamp_(to_prec(std::move(*r), P));
        }
        return stamp_(unary_(x, P, [this, G, R](const Tensor& t)
        {
            return grouped_reduce_max_impl(t, G, R, Precision::F32);
        }));
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
        if (auto ec = bind_check_(inputs); !ec)
            return std::unexpected(ec.error());
        trace_variant_(spec, inputs, P);
        if (P == Precision::F32 && all_f32(inputs))
            return stamp_(eval_expr_impl(spec, inputs, rows, cols, Precision::F32));
        // ── in-kernel f16 优先（Phase 2）────────────────────────────────────
        if (supports_expr_precision_variant(spec, inputs, P))
            return stamp_(eval_expr_impl(spec, inputs, rows, cols, P));
        if (prec_trace_enabled_())
            trace_miss_(spec, inputs, P);
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto r = eval_expr_impl(spec, *in32, rows, cols, Precision::F32);
        if (!r)
            return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), P));
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
        if (auto ec = bind_check_(inputs); !ec)
            return std::unexpected(ec.error());
        trace_variant_(spec, inputs, P);
        if (P == Precision::F32 && all_f32(inputs))
            return stamp_(eval_expr_reduce_impl(spec, inputs, rows, cols, Precision::F32));
        // in-kernel f16 优先（与 eval_expr 同款；归约带类型变体由 gen_fused 生成）
        if (supports_expr_precision_variant(spec, inputs, P))
            return stamp_(eval_expr_reduce_impl(spec, inputs, rows, cols, P));
        if (prec_trace_enabled_())
            trace_miss_(spec, inputs, P);
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto r = eval_expr_reduce_impl(spec, *in32, rows, cols, Precision::F32);
        if (!r)
            return std::unexpected(r.error());
        return stamp_(to_prec(std::move(*r), P));
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
        if (auto ec = bind_check_(inputs); !ec)
            return std::unexpected(ec.error());
        if (auto ec = bind_check_({&out}); !ec)
            return std::unexpected(ec.error());
        trace_variant_(spec, inputs, out.precision());
        if (out.precision() == Precision::F32 && all_f32(inputs))
        {
            if (prec_trace_enabled_())
                std::fprintf(stderr, "[into] branch=native_f32\n");
            return eval_expr_into_impl(spec, inputs, rows, cols, out);
        }
        // in-kernel f16 优先：目标传递 + 带类型变体 = 直接读写原存储，零临时量
        if (supports_expr_precision_variant(spec, inputs, out.precision()))
        {
            if (prec_trace_enabled_())
                std::fprintf(stderr, "[into] branch=native_variant out_p=%d\n",
                             static_cast<int>(out.precision()));
            return eval_expr_into_impl(spec, inputs, rows, cols, out);
        }
        // 目标可能是 f16：在 f32 空间求值后 cast_into 写回 out 的原存储
        // （out 常同时是输入之一——in32 里的 f32 拷贝即"读旧值"，语义正确）
        if (prec_trace_enabled_())
        {
            trace_miss_(spec, inputs, out.precision());
            std::fprintf(stderr, "[into] branch=cast_fallback out_p=%d\n",
                         static_cast<int>(out.precision()));
        }
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto tmp = create_tensor(rows, cols, Precision::F32);
        auto r = eval_expr_into_impl(spec, *in32, rows, cols, tmp);
        if (!r)
            return std::unexpected(r.error());
        return cast_into(tmp, out);
    }

    // ── 原生 f16 数据路径能力（Phase 2 / C1 钩子）──────────────────────────
    // GPU 引擎为 true：数据搬运原语（clone/slice/insert/zero = 模板化字节
    // 拷贝；transpose/rearrange/gather/im2col/col2im = f16 pipeline 或引擎内
    // cast 回退）与原地算术（add_inplace/scale_inplace = f16 pipeline 或
    // 引擎内回退）都能**直接消费 f16 张量**——无 pipeline 时引擎内部完成
    // 边界 cast，基类入口无需物化 f32 副本（归因表搬运/inplace 大头由此消灭）。
    // CPU 引擎为 false：其原语基于 f32 Matrix → 基类入口维持边界 cast。
    // 返回 true 时基类直接放行 f16 张量（省掉"抬 f32 → 算 → 落回 f16"的
    // 全尺寸临时量与 3 倍流量）。
    [[nodiscard]] virtual bool supports_native_data_move() const noexcept { return false; }
    // ── 原生 f16 归约能力（Phase C2）────────────────────────────────────────
    // GPU 引擎为 true：row/col_reduce_sum/max 可直接消费 f16 输入张量
    // （reduce.comp f16 变体：输入 float16_t 直读、f32 归约、输出 f32 向量；
    // 设备无 16bit 存储时引擎内部 cast 回退）。基类据此跳过"抬 f32 副本 →
    // 归约"的全尺寸临时量（归因 L756 大头：Linear::backward grad_bias 直调
    // row_reduce_sum 的 (64,8192)×38）。
    // CPU 引擎为 false：其归约基于 f32 Matrix → 基类入口维持边界 cast。
    [[nodiscard]] virtual bool supports_native_f16_reduce() const noexcept { return false; }
    // ── 精度变体能力查询（Phase 2 in-kernel f16）──────────────────────────
    // 该 (结构, 输入精度, 目标输出精度) 是否有预生成的**带类型**融合 shader？
    // 默认 false（原生引擎无变体概念）→ 基类入口退回边界 cast（把 f16 输入
    // 抬成 f32 副本再调用 *_impl）：正确性不变，只多花带宽。
    // 返回 true 的引擎承诺 `*_impl(spec, inputs, rows, cols, P)` 能直接消费
    // 非 f32 存储的 inputs（读 f16 / 写 f16，算术仍在 f32）。
    [[nodiscard]] virtual bool supports_expr_precision_variant(
        const ExprSpec& spec, std::span<const Tensor> inputs,
        Precision P = Precision::F32) const
    {
        (void)spec; (void)inputs; (void)P;
        return false;
    }

    // ══════════════════════════════════════════════════════════════════════
    // 引擎实现接口（protected 虚函数，NVI 的内层）—— 引擎只实现这些：
    // 只管算，不管精度边界（边界由上面的公共入口处理）。
    // 默认实现仅覆盖"引擎未提供"的少数入口；其余为纯虚。
    // ══════════════════════════════════════════════════════════════════════
protected:
    // ── P1 NVI 的引擎侧实现（原公共虚入口下沉至此，stamp 在公共入口统一做）──
    [[nodiscard]] virtual Tensor create_tensor_impl(std::size_t rows, std::size_t cols, Precision P) = 0;
    [[nodiscard]] virtual Result<Tensor> from_matrix_impl(const Matrix& m, Precision P) = 0;
    // cast 默认实现：同精度 = 返回 src（共享所有权，零拷贝）；跨精度 = 错误（引擎覆盖）
    [[nodiscard]] virtual Result<Tensor> cast_impl(const Tensor& src, Precision dst)
    {
        if (src.precision() == dst)
            return src;  // 同精度 = 返回共享所有权（零拷贝）
        return std::unexpected(Error{"cast: 该引擎不支持跨精度转换"});
    }
    // offload 占位默认（激活 offload 是 GPU 特性；CPU 开启只得到 1×1 张量）
    [[nodiscard]] virtual Result<Tensor> create_offload_buffer_impl(std::size_t /*bytes*/)
    {
        return Tensor::cpu(1, 1);
    }
    [[nodiscard]] virtual Result<Tensor> offload_restore_impl(
        const Tensor& /*buffer*/, std::size_t /*offset*/,
        std::size_t /*rows*/, std::size_t /*cols*/)
    {
        return Tensor::cpu(1, 1);
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
            return std::unexpected(Error{
                "submit_scalar_readback: 默认实现仅支持 CPU 张量"});
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
            return std::unexpected(Error{
                "submit_scalar_readback: 不支持的精度"});
        }
        if (sync_readback_slots_.size() <= slot)
            sync_readback_slots_.resize(slot + 1, Scalar{0});
        sync_readback_slots_[slot] = v;
        return {};
    }

    // ── 纯数据搬运 ────────────────────────────────────────────────────────
    [[nodiscard]] virtual Result<Tensor> clone_impl(const Tensor& src) = 0;
    [[nodiscard]] virtual Result<void> copy_from_impl(Tensor& dst, const Matrix& src) = 0;
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
    [[nodiscard]] virtual Result<Tensor> batched_matmul_impl(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA, bool transB,
        Scalar alpha,
        Precision P) = 0;

    // 默认实现：matmul + 逐行 add bias（兼容所有引擎）
    [[nodiscard]] virtual Result<Tensor> matmul_with_bias_impl(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA, bool transB, Precision P)
    {
        auto result = matmul(A, B, transA, transB, P);
        if (!result) return std::unexpected(result.error());

        // broadcast bias: (out,1) → (out,batch)
        auto bias_mat = to_matrix(bias, P);
        if (!bias_mat) return std::unexpected(bias_mat.error());

        auto res_mat = to_matrix(*result, P);
        if (!res_mat) return std::unexpected(res_mat.error());

        for (std::size_t row = 0; row < A.rows(); ++row)
        {
            float b_val = bias_mat->at(row, 0);
            for (std::size_t col = 0; col < B.cols(); ++col)
                res_mat->set_value(row, col, res_mat->at(row, col) + b_val);
        }

        return from_matrix(*res_mat, P);
    }

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
        return std::unexpected(Error{"eval_expr_reduce: 该引擎不支持归约向量输出"});
    }

    // 默认实现返回错误（未支持的引擎）；CPU/GPU 覆盖为真原地实现。
    [[nodiscard]] virtual Result<void> eval_expr_into_impl(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out)
    {
        (void)spec; (void)inputs; (void)rows; (void)cols; (void)out;
        return std::unexpected(Error{
            "eval_expr_into: 该引擎不支持原地表达式求值"});
    }

private:
    // 库内 stamp 通道（15 §4.2 D2）：有引擎在场、但产物走静态工厂的位置
    // （dsl::compute 的 eval_cpu 出口与扫描占位、compute_reduce 的归约向量
    // 出口）统一经此补绑定。非用户 API——friend 限定给 nn::dsl 两个入口。
    template <typename E>
    friend Result<Tensor> dsl::compute(ComputeEngine&, const E&, std::size_t,
                                       std::size_t, Precision);
    template <typename E>
    friend Result<Tensor> dsl::compute_reduce(ComputeEngine&, const E&, std::size_t,
                                              std::size_t, Precision);
    [[nodiscard]] Tensor adopt(Tensor t) { return stamp_(std::move(t)); }

    // ══════════════════════════════════════════════════════════════════════
    // P1 出生 stamp（docs/development/15 §3.1）：未绑定的产物补上 this；
    // 已绑定 → 原样（cast 同精度返回 src 等传播场景沿用 src 绑定，不改写）。
    // 只 stamp 有效张量：失败路径返回的 Tensor() 空槽保持不绑定
    // （§3.1：库内非绑定态仅剩空槽；库外直构不受本函数管辖）。
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Tensor stamp_(Tensor t)
    {
        if (t.valid() && !t.engine_)
            t.engine_ = make_observer(*this);
        return t;
    }
    [[nodiscard]] Result<Tensor> stamp_(Result<Tensor>&& r)
    {
        if (r && r->valid() && !r->engine_)
            r->engine_ = make_observer(*this);
        return std::move(r);
    }

    // ── P1 跨引擎检查（15 §4.3 D3）──────────────────────────────────────
    // 判定（D3 字面）：操作数指针判等——**双方都 bound 且不同 → Result 硬错误**；
    // 单侧未绑定按库外豁免放行（P1 只加检查、零行为变化，ctest 不红）。
    // NN_BIND_DEBUG=1（进程启动前设置）时"未绑定输入进引擎"也报错：用于抓
    // 库内 stamp 漏网 + 产出 P2 库外迁移清单；P3 视情况转默认开并升 import。
    // 错误带调用点 source_location（默认参在调用处求值）与张量形状。
    [[nodiscard]] static bool bind_debug_enabled_()
    {
        static const bool on = [] {
#if defined(_MSC_VER)
            char* buf = nullptr; std::size_t len = 0;
            _dupenv_s(&buf, &len, "NN_BIND_DEBUG");
            const bool v = (buf != nullptr && buf[0] != '\0' && buf[0] != '0');
            std::free(buf);
            return v;
#else
            const char* v = std::getenv("NN_BIND_DEBUG");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
        }();
        return on;
    }

    [[nodiscard]] static Error bind_error_(const char* kind, const Tensor* t,
                                           const Tensor* other,
                                           std::source_location loc)
    {
        std::string s = std::string("bind_check_ ") + kind + " at "
                      + loc.file_name() + ":" + std::to_string(loc.line()) + "  "
                      + (t ? t->shape_str() : std::string("?"));
        if (t && other)
            s += " vs " + other->shape_str();
        return Error{std::move(s)};
    }

    // 单操作数判定核心：维护"首个 bound 引擎"基准 ref，后续 bound 必须与之相等
    [[nodiscard]] Result<void> bind_check_one_(const Tensor*& ref, const Tensor* t,
                                               std::source_location loc) const
    {
        if (!t)
            return {};
        if (!t->bound())
        {
            if (bind_debug_enabled_())
                return std::unexpected(
                    bind_error_("unbound input into engine", t, nullptr, loc));
            return {};   // 库外豁免（D3）
        }
        if (!ref)
        {
            ref = t;
            return {};
        }
        if (t->engine_.get() != ref->engine_.get())
            return std::unexpected(
                bind_error_("mixed engines (both bound, pointers differ)", ref, t, loc));
        return {};
    }

    [[nodiscard]] Result<void> bind_check_(std::initializer_list<const Tensor*> ts,
                                           std::source_location loc = std::source_location::current()) const
    {
        const Tensor* ref = nullptr;
        for (const Tensor* t : ts)
            if (auto r = bind_check_one_(ref, t, loc); !r)
                return r;
        return {};
    }

    [[nodiscard]] Result<void> bind_check_(std::span<const Tensor> ts,
                                           std::source_location loc = std::source_location::current()) const
    {
        const Tensor* ref = nullptr;
        for (const Tensor& t : ts)
            if (auto r = bind_check_one_(ref, &t, loc); !r)
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
            if (!c)
                return std::unexpected(c.error());
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
        // Phase C1b：引擎原生支持 f16（GPU：f16 pipeline 原地直加/缩放，
        // 无 pipeline 引擎内回退）→ 直通，A 不物化 f32 副本（inplace 归因
        // 大头由此消灭）。CPU 钩子 false → 维持边界 cast。
        if (supports_native_data_move())
            return fn(A, B);
        auto b = to_f32(B, loc);
        if (!b)
            return std::unexpected(b.error());
        if (A.precision() == Precision::F32)
            return fn(A, *b);
        auto a = to_f32(A, loc);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a, *b);
        if (!r)
            return std::unexpected(r.error());
        return cast_into(*a, A);   // 存储精度不可变（§8.3）
    }

    // in-place 单操作数通用
    template <typename Fn>
    [[nodiscard]] Result<void> inplace1_(Tensor& A, Fn&& fn,
                                         std::source_location loc = std::source_location::current())
    {
        if (A.precision() == Precision::F32)
            return fn(A);
        // Phase C1b：同 inplace2_——GPU 原生 f16 原地（scale_inplace 的
        // (2048,256) 归因大头由此消灭）；CPU 钩子 false → 维持边界 cast
        if (supports_native_data_move())
            return fn(A);
        auto a = to_f32(A, loc);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
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
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
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
        if (!a)
            return std::unexpected(a.error());
        auto b = to_f32(B, loc);
        if (!b)
            return std::unexpected(b.error());
        auto r = fn(*a, *b);
        if (!r)
            return std::unexpected(r.error());
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
        // Phase C1：引擎原生支持 f16 数据搬运（GPU：f16 pipeline 直读直写；
        // 无 pipeline 时引擎内自行 cast 回退）→ 直通，**不物化 f32 副本**。
        // CPU 引擎此钩子为 false → 维持边界 cast（其原语基于 f32 Matrix）。
        if (supports_native_data_move())
            return fn(A);
        auto a = to_f32(A, loc);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), out_p, loc);
    }

    // ── 多精度变体发现 / 边界 cast 归因（诊断；NN_PREC_TRACE=1）────────────
    // trace_variant_：打印每个 (结构 key, 精度签名) —— 这就是 in-kernel f16
    // 需要生成的变体集合。本层是所有原语调用的必经点、能看到每次调用的
    // **真实输入精度**，故变体发现放在这里（scan 的 f16 dry-run 是另一条同源路径）。
    // note_temp_：记录一次"物化临时量"（形状 + 调用点行号 → 次数/字节）；见
    // dump_temp_stats()。
    // line = source_location 行号（区分同一形状来自哪个算子路径——只有形状
    // 时 matmul/unary_/eval_expr 的 cast 无法分辨）。
    static void note_temp_(std::size_t rows, std::size_t cols, bool to_f32,
                           std::uint32_t line = 0)
    {
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
    }

    [[nodiscard]] static bool prec_trace_enabled_()
    {
        static const bool on = [] {
#if defined(_MSC_VER)
            char* buf = nullptr; std::size_t len = 0;
            _dupenv_s(&buf, &len, "NN_PREC_TRACE");
            const bool v = (buf != nullptr && buf[0] != '\0' && buf[0] != '0');
            std::free(buf);
            return v;
#else
            const char* v = std::getenv("NN_PREC_TRACE");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
        }();
        return on;
    }

    // 通用环境变量开关（原 nn::dsl::env_flag 同语义：每次 getenv、零缓存；
    // compute_engine.hpp 不能反向依赖 expr_dsl.hpp，故保留同款实现）。
    [[nodiscard]] static bool prec_env_flag(const char* name)
    {
#if defined(_MSC_VER)
        char* buf = nullptr; std::size_t len = 0;
        _dupenv_s(&buf, &len, name);
        const bool v = (buf != nullptr && buf[0] != '\0' && buf[0] != '0');
        std::free(buf);
        return v;
#else
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
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
};

} // namespace nn
