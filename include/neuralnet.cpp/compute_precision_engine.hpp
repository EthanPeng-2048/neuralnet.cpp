#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  compute_precision_engine.hpp — 多精度适配层（PrecisionEngine）
//
//  设计依据：docs/development/05-mixed-precision.md（Q3-A / Q4 / Q6 / §8 / §11.1）
//
//  一句话：**f16 只改变"数据存哪儿"，不改变"算子怎么算"**。融合世界（AOT
//  闭合世界 + 手写 shader）保持全 f32，f16 的边界 cast 全部集中在本层：
//
//      入：把 f16 操作数抬到 f32（cast；已是 f32 则零拷贝直通）
//      算：调用内层引擎的既有 f32 实现（零改动、零分歧、零 shader 变体）
//      出：结果按目标精度落回（f16 写出 = round-half-to-even，与硬件一致）
//
//  in-place 原语（add/scale/axpy/zero/broadcast_*/eval_expr_into）用
//  `cast_into` 写回**原存储**：保留张量对象身份与底层 buffer，避免"替换对象"
//  导致其它持有同一句柄的缓存静默失联（§8.3：in-place 存储精度不可变）。
//
//  为什么不做"引擎内逐原语 f16 分支"（§11.1 的 Phase 2 才做）：
//    · 需要 CPU/GPU 各写一遍（两份易漂移的实现，违背"改一处只改一个头文件"）；
//    · 手写 shader（elementwise/reduce/broadcast/matmul_tiled/gather…）全是 f32
//      类型，逐条改 f16 变体会让 shader 与 AOT 注册表规模翻倍（每个 ExprSpec
//      都要两份），而收益仅在带宽（显存占用由**存储**决定，边界 cast 已拿到）。
//  边界 cast 以"设备内多一次带宽"换"零 shader 变体、零注册表膨胀、零回归"。
//
//  精度语义（本层的唯一规则，显式、可预测）：
//    · 带 P 形参的运算类原语：输出/计算精度 = P（**不做 Auto 推导**；Layer 按
//      §8.5 约定总是显式传 `p_.compute` / `p_.stable`，未传 = F32 = 现状）。
//    · 不带 P 的纯数据搬运原语（transpose/slice/insert/gather/scatter/
//      rearrange_3d/im2col/col2im/clone）：输出精度 = 源精度（§8.4）。
//    · in-place 原语：存储精度不可变，显式 P 只改计算路径（§8.3）。
//
//  用法（CLI / 测试）：
//      nn::GpuEngine gpu;
//      nn::PrecisionEngine engine(gpu);          // 默认 profile = 全 f32
//      auto model = build_gpt_model(engine, ...);  // 传适配层即可
//  全 f32 配置下本层是**纯直通**（每个原语一条 if 判断），行为与直接使用内层
//  引擎逐字节一致（G5 零回归）。
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compute_engine.hpp"
#include "expr_opt.hpp"   // canonicalize_expr_spec（与 gen_fused 同源 key）

namespace nn
{

class PrecisionEngine final : public ComputeEngine
{
private:
    ComputeEngine& inner_;

    // ── 内部辅助 ──────────────────────────────────────────────────────────
    // 全部 f32？（快速直通判定：不物化任何中间量）
    [[nodiscard]] static bool all_f32(std::span<const Tensor> ts) noexcept
    {
        for (const auto& t : ts)
            if (t.precision() != Precision::F32)
                return false;
        return true;
    }

    // 抬到 f32：已是 f32 → 共享所有权直通（零拷贝）；f16 → 引擎 cast
    [[nodiscard]] Result<Tensor> to_f32(const Tensor& t)
    {
        if (t.precision() == Precision::F32)
            return t;
        note_temp_(t.rows(), t.cols(), /*to_f32=*/true);
        return inner_.cast(t, Precision::F32);
    }

    // 落回目标精度：目标 f32 或已同精度 → 原样返回
    [[nodiscard]] Result<Tensor> to_prec(Tensor t, Precision P)
    {
        if (P == Precision::F32 || t.precision() == P)
            return t;
        note_temp_(t.rows(), t.cols(), /*to_f32=*/false);
        return inner_.cast(std::move(t), P);
    }

    // 一批操作数抬到 f32（顺序与输入一致；ExprSpec 的 views 依赖该顺序）
    [[nodiscard]] Result<std::vector<Tensor>> to_f32_all(std::span<const Tensor> ts)
    {
        std::vector<Tensor> out;
        out.reserve(ts.size());
        for (const auto& t : ts)
        {
            auto c = to_f32(t);
            if (!c)
                return std::unexpected(c.error());
            out.push_back(std::move(*c));
        }
        return out;
    }

    // in-place 双操作数通用：抬 f32 → 算 → 必要时 cast_into 写回 A 的原存储
    template <typename Fn>
    [[nodiscard]] Result<void> inplace2_(Tensor& A, const Tensor& B, Fn&& fn)
    {
        if (A.precision() == Precision::F32 && B.precision() == Precision::F32)
            return fn(A, B);
        auto b = to_f32(B);
        if (!b)
            return std::unexpected(b.error());
        if (A.precision() == Precision::F32)
            return fn(A, *b);
        auto a = to_f32(A);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a, *b);
        if (!r)
            return std::unexpected(r.error());
        return inner_.cast_into(*a, A);   // 存储精度不可变（§8.3）
    }

    // in-place 单操作数通用
    template <typename Fn>
    [[nodiscard]] Result<void> inplace1_(Tensor& A, Fn&& fn)
    {
        if (A.precision() == Precision::F32)
            return fn(A);
        auto a = to_f32(A);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
        return inner_.cast_into(*a, A);
    }

    // 运算类通用：抬 f32 → 算 → 按 P 落回（单操作数）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> unary_(const Tensor& A, Precision P, Fn&& fn)
    {
        if (P == Precision::F32 && A.precision() == Precision::F32)
            return fn(A);
        auto a = to_f32(A);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), P);
    }

    // 运算类通用：抬 f32 → 算 → 按 P 落回（双操作数）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> binary_(const Tensor& A, const Tensor& B,
                                         Precision P, Fn&& fn)
    {
        if (P == Precision::F32 && A.precision() == Precision::F32 &&
            B.precision() == Precision::F32)
            return fn(A, B);
        auto a = to_f32(A);
        if (!a)
            return std::unexpected(a.error());
        auto b = to_f32(B);
        if (!b)
            return std::unexpected(b.error());
        auto r = fn(*a, *b);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), P);
    }

    // 数据搬运类通用：输出精度 = 源精度（§8.4，无 P 形参）
    template <typename Fn>
    [[nodiscard]] Result<Tensor> move_(const Tensor& A, Fn&& fn)
    {
        const Precision out_p = A.precision();
        if (out_p == Precision::F32)
            return fn(A);
        auto a = to_f32(A);
        if (!a)
            return std::unexpected(a.error());
        auto r = fn(*a);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), out_p);
    }

    // ── 多精度变体发现 / 边界 cast 归因（诊断；NN_PREC_TRACE=1）────────────
    // 打印每个 (结构 key, 精度签名) —— 这就是 in-kernel f16 需要生成的变体集合。
    // 只有适配层看得到**真实输入精度**（内层引擎收到的永远是 f32 副本），故
    // 变体发现必须放在这里（scan 的 f16 dry-run 是另一条同源路径）。
    // ── 多精度变体发现 / 边界 cast 归因（诊断；NN_PREC_TRACE=1）────────────
    // 记录一次"物化临时量"（形状 → 次数/字节）；见 dump_temp_stats()。
    static void note_temp_(std::size_t rows, std::size_t cols, bool to_f32)
    {
        if (!prec_trace_enabled_())
            return;
        const std::size_t bytes = rows * cols * (to_f32 ? 4u : 2u);
        for (auto& e : temp_stats())
            if (e.rows == rows && e.cols == cols && e.to_f32 == to_f32)
            {
                e.count++;
                e.bytes += bytes;
                return;
            }
        temp_stats().push_back(TempStat{rows, cols, to_f32, 1, bytes});
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

public:
    explicit PrecisionEngine(ComputeEngine& inner) : inner_(inner) {}

    // ── 边界 cast 归因（诊断；NN_PREC_TRACE=1）───────────────────────────
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
                          "  %-11s (%zu,%zu)  x%-5zu %8.1f MB\n",
                          e->to_f32 ? "->f32 副本" : "->按P落回",
                          e->rows, e->cols, e->count,
                          static_cast<double>(e->bytes) / (1024.0 * 1024.0));
            s += buf;
        }
        return s;
    }

    static void reset_temp_stats() { temp_stats().clear(); }

    // 内层引擎（需要直连 f32 路径或查询后端状态时用；诊断/测试）
    [[nodiscard]] ComputeEngine& inner() noexcept { return inner_; }
    [[nodiscard]] const ComputeEngine& inner() const noexcept { return inner_; }

    // ══════════════════════════════════════════════════════════════════════
    // 设备 / 批处理 / 诊断：原样转发
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Device device() const noexcept override { return inner_.device(); }
    [[nodiscard]] Result<void> begin_batch() override { return inner_.begin_batch(); }
    [[nodiscard]] Result<void> end_batch() override { return inner_.end_batch(); }
    [[nodiscard]] Result<void> flush_batch() override { return inner_.flush_batch(); }
    [[nodiscard]] Result<void> release_idle_pool_blocks() override
    {
        return inner_.release_idle_pool_blocks();
    }
    [[nodiscard]] std::string pool_stats() const override { return inner_.pool_stats(); }

    // ── 激活 offload：slab 统一按 f32 存（offset 以 float 为单位）──────────
    // f16 激活在写入 slab 前抬到 f32：restore 出来是 f32，后续运算由本层
    // 适配（正确性不受影响，只是该份激活不再享 f16 存储折半）。
    [[nodiscard]] Result<Tensor> offload_store(const Tensor& src) override
    {
        auto s = to_f32(src);
        if (!s)
            return std::unexpected(s.error());
        return inner_.offload_store(*s);
    }
    [[nodiscard]] Result<Tensor> offload_load(
        const Tensor& handle, std::size_t rows, std::size_t cols) override
    {
        return inner_.offload_load(handle, rows, cols);
    }
    [[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes) override
    {
        return inner_.create_offload_buffer(bytes);
    }
    [[nodiscard]] Result<void> offload_save(
        const Tensor& buffer, std::size_t offset, const Tensor& src) override
    {
        auto s = to_f32(src);
        if (!s)
            return std::unexpected(s.error());
        return inner_.offload_save(buffer, offset, *s);
    }
    [[nodiscard]] Result<Tensor> offload_restore(
        const Tensor& buffer, std::size_t offset,
        std::size_t rows, std::size_t cols) override
    {
        return inner_.offload_restore(buffer, offset, rows, cols);
    }

    // ── 标量回读：GPU 引擎要求源为 f32（先 cast 再排 D2H 拷贝）────────────
    [[nodiscard]] Result<void> submit_scalar_readback(
        std::size_t slot, const Tensor& t) override
    {
        auto s = to_f32(t);
        if (!s)
            return std::unexpected(s.error());
        return inner_.submit_scalar_readback(slot, *s);
    }
    [[nodiscard]] Result<bool> poll_scalar_readback(
        std::size_t slot, Scalar& out) override
    {
        return inner_.poll_scalar_readback(slot, out);
    }
    [[nodiscard]] std::size_t scalar_readback_slots() const override
    {
        return inner_.scalar_readback_slots();
    }

    // ══════════════════════════════════════════════════════════════════════
    // 张量工厂 / 精度转换：原样转发（内层引擎已实现 f16 存储与原生 cast）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Tensor create_tensor(
        std::size_t rows, std::size_t cols, Precision P = Precision::F32) override
    {
        return inner_.create_tensor(rows, cols, P);
    }
    [[nodiscard]] Result<Tensor> from_matrix(
        const Matrix& m, Precision P = Precision::F32) override
    {
        return inner_.from_matrix(m, P);
    }
    [[nodiscard]] Result<Matrix> to_matrix(
        const Tensor& t, Precision P = Precision::F32) override
    {
        return inner_.to_matrix(t, P);
    }
    [[nodiscard]] Result<Tensor> cast(const Tensor& src, Precision dst) override
    {
        return inner_.cast(src, dst);
    }
    [[nodiscard]] Result<void> cast_into(const Tensor& src, Tensor& dst) override
    {
        return inner_.cast_into(src, dst);
    }
    [[nodiscard]] Result<void> copy_into(Tensor& dst, const Tensor& src) override
    {
        return inner_.copy_into(dst, src);
    }

    // host Matrix → dst（序列化加载 / 优化器参数写回）
    // f16 目标：先按 f32 上传，再 cast_into 写进 dst 的原存储
    [[nodiscard]] Result<void> copy_from(Tensor& dst, const Matrix& src) override
    {
        if (dst.precision() == Precision::F32)
            return inner_.copy_from(dst, src);
        auto t = inner_.from_matrix(src, Precision::F32);
        if (!t)
            return std::unexpected(t.error());
        return inner_.cast_into(*t, dst);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 纯数据搬运原语（无 P：输出精度 = 源精度，§8.4）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> clone(const Tensor& src) override
    {
        // 原生 f16 数据搬运：内层引擎的 clone 是模板化字节拷贝 → 直接放行，
        // 省掉"抬 f32 → 拷贝 → 落回 f16"的 2 份全尺寸临时量（§12.8 归因）。
        if (src.precision() != Precision::F32 && inner_.supports_native_data_move())
            return inner_.clone(src);
        return move_(src, [this](const Tensor& s) { return inner_.clone(s); });
    }

    [[nodiscard]] Result<Tensor> slice_rows(
        const Tensor& src, std::size_t start_row, std::size_t count) override
    {
        if (src.precision() != Precision::F32 && inner_.supports_native_data_move())
            return inner_.slice_rows(src, start_row, count);
        return move_(src, [this, start_row, count](const Tensor& s)
        {
            return inner_.slice_rows(s, start_row, count);
        });
    }

    // insert_rows：就地写入 dst 的行区间 → dst 精度即目标精度
    [[nodiscard]] Result<void> insert_rows(
        Tensor& dst, std::size_t dst_start_row, const Tensor& src) override
    {
        if (dst.precision() != Precision::F32 && dst.precision() == src.precision() &&
            inner_.supports_native_data_move())
            return inner_.insert_rows(dst, dst_start_row, src);
        if (dst.precision() == Precision::F32 && src.precision() == Precision::F32)
            return inner_.insert_rows(dst, dst_start_row, src);
        // 不同精度（或 f16 dst）：统一在 f32 空间插入后写回 dst 的原存储
        auto d = to_f32(dst);
        if (!d)
            return std::unexpected(d.error());
        auto s = to_f32(src);
        if (!s)
            return std::unexpected(s.error());
        auto r = inner_.insert_rows(*d, dst_start_row, *s);
        if (!r)
            return std::unexpected(r.error());
        if (dst.precision() == Precision::F32)
            return {};
        return inner_.cast_into(*d, dst);
    }

    [[nodiscard]] Result<Tensor> gather_rows(
        const Tensor& table, const Tensor& indices) override
    {
        return move_(table, [this, &indices](const Tensor& t)
        {
            return inner_.gather_rows(t, indices);
        });
    }

    // scatter_add_rows：就地累加到 dst → 在 f32 空间累加后写回 dst 存储
    [[nodiscard]] Result<void> scatter_add_rows(
        Tensor& dst, const Tensor& indices, const Tensor& grad) override
    {
        if (dst.precision() == Precision::F32 && grad.precision() == Precision::F32)
            return inner_.scatter_add_rows(dst, indices, grad);
        auto d = to_f32(dst);
        if (!d)
            return std::unexpected(d.error());
        auto g = to_f32(grad);
        if (!g)
            return std::unexpected(g.error());
        auto r = inner_.scatter_add_rows(*d, indices, *g);
        if (!r)
            return std::unexpected(r.error());
        if (dst.precision() == Precision::F32)
            return {};
        return inner_.cast_into(*d, dst);
    }

    [[nodiscard]] Result<Tensor> rearrange_3d(
        const Tensor& x, std::size_t M, std::size_t B, std::size_t N,
        bool inverse = false) override
    {
        return move_(x, [this, M, B, N, inverse](const Tensor& t)
        {
            return inner_.rearrange_3d(t, M, B, N, inverse);
        });
    }

    [[nodiscard]] Result<Tensor> transpose(const Tensor& A) override
    {
        return move_(A, [this](const Tensor& t) { return inner_.transpose(t); });
    }

    [[nodiscard]] Result<Tensor> im2col(
        const Tensor& x, std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        return move_(x, [this, C, H, W, k, stride, pad, OH, OW](const Tensor& t)
        {
            return inner_.im2col(t, C, H, W, k, stride, pad, OH, OW);
        });
    }

    [[nodiscard]] Result<Tensor> col2im(
        const Tensor& col, std::size_t C, std::size_t H, std::size_t W,
        std::size_t k, std::size_t stride, std::size_t pad,
        std::size_t OH, std::size_t OW) override
    {
        return move_(col, [this, C, H, W, k, stride, pad, OH, OW](const Tensor& t)
        {
            return inner_.col2im(t, C, H, W, k, stride, pad, OH, OW);
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 矩阵级原语
    // ══════════════════════════════════════════════════════════════════════
    // 非 f32 直接下传（不经本层 cast）：内层有 **f16 存储版 GEMM**
    // （f16 直读 + f32 累加 + f16 写出）→ 不再为每个操作数物化整份 f32 副本
    // （实测这是训练 transient 膨胀的头号来源）。内层在"f16 pipeline 不可用 /
    // 小 N GEMV / 操作数精度混合"时自行回退 f32 空间 + 边界 cast，正确性不变。
    [[nodiscard]] Result<Tensor> matmul(
        const Tensor& A, const Tensor& B,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32) override
    {
        if (P != Precision::F32)
            return inner_.matmul(A, B, transA, transB, P);
        return binary_(A, B, P, [this, transA, transB](const Tensor& a, const Tensor& b)
        {
            return inner_.matmul(a, b, transA, transB, Precision::F32);
        });
    }

    [[nodiscard]] Result<Tensor> batched_matmul(
        const Tensor& A, const Tensor& B,
        std::size_t batch,
        bool transA = false, bool transB = false,
        Scalar alpha = Scalar{1},
        Precision P = Precision::F32) override
    {
        if (P != Precision::F32)
            return inner_.batched_matmul(A, B, batch, transA, transB, alpha, P);
        return binary_(A, B, P,
            [this, batch, transA, transB, alpha](const Tensor& a, const Tensor& b)
        {
            return inner_.batched_matmul(a, b, batch, transA, transB, alpha,
                                         Precision::F32);
        });
    }

    [[nodiscard]] Result<Tensor> matmul_with_bias(
        const Tensor& A, const Tensor& B, const Tensor& bias,
        bool transA = false, bool transB = false,
        Precision P = Precision::F32) override
    {
        // 非 f32 直接下传（不经本层 cast）：内层 matmul_with_bias 经 DSL 融合
        // matmul 段，带类型变体可用时 **f16 直读直写、零边界 cast**；无变体时
        // 内层自行回退 f32 空间。scan 的 f16 dry-run 也经此路径 → DSL 钩子按
        // 真实操作数精度 + P 登记 (结构,签名)，带类型变体才可能被发现。
        if (P != Precision::F32 &&
            (A.precision() != Precision::F32 || B.precision() != Precision::F32 ||
             bias.precision() != Precision::F32))
            return inner_.matmul_with_bias(A, B, bias, transA, transB, P);

        if (P == Precision::F32 && A.precision() == Precision::F32 &&
            B.precision() == Precision::F32 && bias.precision() == Precision::F32)
            return inner_.matmul_with_bias(A, B, bias, transA, transB);
        auto a = to_f32(A);
        if (!a) return std::unexpected(a.error());
        auto b = to_f32(B);
        if (!b) return std::unexpected(b.error());
        auto bi = to_f32(bias);
        if (!bi) return std::unexpected(bi.error());
        auto r = inner_.matmul_with_bias(*a, *b, *bi, transA, transB);
        if (!r) return std::unexpected(r.error());
        return to_prec(std::move(*r), P);
    }

    // 梯度累加：dst += src（dst 存储精度不可变，§8.3）
    [[nodiscard]] Result<void> accumulate(Tensor& dst, const Tensor& src) override
    {
        return inplace2_(dst, src, [this](Tensor& d, const Tensor& s)
        {
            return inner_.add_inplace(d, s);
        });
    }

    [[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B) override
    {
        return inplace2_(A, B, [this](Tensor& a, const Tensor& b)
        {
            return inner_.add_inplace(a, b);
        });
    }

    [[nodiscard]] Result<void> scale_inplace(Tensor& A, Scalar s) override
    {
        return inplace1_(A, [this, s](Tensor& a) { return inner_.scale_inplace(a, s); });
    }

    [[nodiscard]] Result<void> axpy_inplace(
        Tensor& A, Scalar scalar, const Tensor& B) override
    {
        return inplace2_(A, B, [this, scalar](Tensor& a, const Tensor& b)
        {
            return inner_.axpy_inplace(a, scalar, b);
        });
    }

    [[nodiscard]] Result<void> zero(Tensor& A) override
    {
        // 原生 f16 清零：fill_zero 是字节级原语（每步 zero_grad 调用 N 次）
        if (A.precision() != Precision::F32 && inner_.supports_native_data_move())
            return inner_.zero(A);
        return inplace1_(A, [this](Tensor& a) { return inner_.zero(a); });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 扫描级原语（RLA/RAPT）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> scan_prefix_outer(
        const Tensor& K, const Tensor& V, const Tensor& P, const Tensor& R,
        const Tensor& A0, const Tensor& B0, bool has_state,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32) override
    {
        if (prec == Precision::F32 && K.precision() == Precision::F32 &&
            V.precision() == Precision::F32 && P.precision() == Precision::F32 &&
            R.precision() == Precision::F32 && A0.precision() == Precision::F32 &&
            B0.precision() == Precision::F32 && boundary.precision() == Precision::F32)
            return inner_.scan_prefix_outer(K, V, P, R, A0, B0, has_state, dk, heads,
                                            causal, boundary, has_bnd);
        const std::vector<Tensor> ts{K, V, P, R, A0, B0, boundary};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = inner_.scan_prefix_outer((*c)[0], (*c)[1], (*c)[2], (*c)[3],
                                          (*c)[4], (*c)[5], has_state, dk, heads,
                                          causal, (*c)[6], has_bnd);
        if (!r) return std::unexpected(r.error());
        return to_prec(std::move(*r), prec);
    }

    [[nodiscard]] Result<Tensor> scan_suffix_outer(
        const Tensor& D, const Tensor& X, const Tensor& Y,
        std::size_t dk, std::size_t heads, bool causal,
        const Tensor& boundary, bool has_bnd,
        Precision prec = Precision::F32) override
    {
        if (prec == Precision::F32 && D.precision() == Precision::F32 &&
            X.precision() == Precision::F32 && Y.precision() == Precision::F32 &&
            boundary.precision() == Precision::F32)
            return inner_.scan_suffix_outer(D, X, Y, dk, heads, causal, boundary,
                                            has_bnd);
        const std::vector<Tensor> ts{D, X, Y, boundary};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = inner_.scan_suffix_outer((*c)[0], (*c)[1], (*c)[2], dk, heads,
                                          causal, (*c)[3], has_bnd);
        if (!r) return std::unexpected(r.error());
        return to_prec(std::move(*r), prec);
    }

    [[nodiscard]] Result<Tensor> outer_col(
        const Tensor& P, const Tensor& R, const Tensor& S,
        std::size_t dk, bool has_scale,
        Precision prec = Precision::F32) override
    {
        if (prec == Precision::F32 && P.precision() == Precision::F32 &&
            R.precision() == Precision::F32 && S.precision() == Precision::F32)
            return inner_.outer_col(P, R, S, dk, has_scale);
        const std::vector<Tensor> ts{P, R, S};
        auto c = to_f32_all(ts);
        if (!c) return std::unexpected(c.error());
        auto r = inner_.outer_col((*c)[0], (*c)[1], (*c)[2], dk, has_scale);
        if (!r) return std::unexpected(r.error());
        return to_prec(std::move(*r), prec);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 归约原语（§7.3：累加恒 f32，输出舍入到 P）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> row_reduce_sum(
        const Tensor& A, Precision P = Precision::F32) override
    {
        return unary_(A, P, [this](const Tensor& a) { return inner_.row_reduce_sum(a); });
    }
    [[nodiscard]] Result<Tensor> col_reduce_sum(
        const Tensor& A, Precision P = Precision::F32) override
    {
        return unary_(A, P, [this](const Tensor& a) { return inner_.col_reduce_sum(a); });
    }
    [[nodiscard]] Result<Tensor> row_reduce_max(
        const Tensor& A, Precision P = Precision::F32) override
    {
        return unary_(A, P, [this](const Tensor& a) { return inner_.row_reduce_max(a); });
    }
    [[nodiscard]] Result<Tensor> col_reduce_max(
        const Tensor& A, Precision P = Precision::F32) override
    {
        return unary_(A, P, [this](const Tensor& a) { return inner_.col_reduce_max(a); });
    }
    [[nodiscard]] Result<Tensor> grouped_reduce_sum(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision P = Precision::F32) override
    {
        return unary_(x, P, [this, G, R](const Tensor& t)
        {
            return inner_.grouped_reduce_sum(t, G, R);
        });
    }
    [[nodiscard]] Result<Tensor> grouped_reduce_max(
        const Tensor& x, std::size_t G, std::size_t R,
        Precision P = Precision::F32) override
    {
        return unary_(x, P, [this, G, R](const Tensor& t)
        {
            return inner_.grouped_reduce_max(t, G, R);
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 广播原语（in-place：A 的存储精度不变）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<void> broadcast_row_inplace(
        Tensor& A, const Tensor& row_vec, BinaryOp op) override
    {
        return inplace2_(A, row_vec, [this, op](Tensor& a, const Tensor& v)
        {
            return inner_.broadcast_row_inplace(a, v, op);
        });
    }
    [[nodiscard]] Result<void> broadcast_col_inplace(
        Tensor& A, const Tensor& col_vec, BinaryOp op) override
    {
        return inplace2_(A, col_vec, [this, op](Tensor& a, const Tensor& v)
        {
            return inner_.broadcast_col_inplace(a, v, op);
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 逐元素原语
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> elementwise_unary(
        UnaryOp op, const Tensor& A, Precision P = Precision::F32) override
    {
        return unary_(A, P, [this, op](const Tensor& a)
        {
            return inner_.elementwise_unary(op, a);
        });
    }

    [[nodiscard]] Result<Tensor> elementwise_binary(
        BinaryOp op, const Tensor& A, const Tensor& B,
        Precision P = Precision::F32) override
    {
        return binary_(A, B, P, [this, op](const Tensor& a, const Tensor& b)
        {
            return inner_.elementwise_binary(op, a, b);
        });
    }

    [[nodiscard]] Result<Tensor> elementwise_binary_scalar(
        BinaryOp op, const Tensor& A, Scalar s, bool scalar_first = false,
        Precision P = Precision::F32) override
    {
        return unary_(A, P, [this, op, s, scalar_first](const Tensor& a)
        {
            return inner_.elementwise_binary_scalar(op, a, s, scalar_first);
        });
    }

    [[nodiscard]] Result<Tensor> elementwise_select_scalar_cond(
        CompareOp cmp, const Tensor& A, Scalar scalar_b,
        const Tensor& then_t, Scalar scalar_else,
        Precision P = Precision::F32) override
    {
        return binary_(A, then_t, P,
            [this, cmp, scalar_b, scalar_else](const Tensor& a, const Tensor& t)
        {
            return inner_.elementwise_select_scalar_cond(cmp, a, scalar_b, t,
                                                         scalar_else);
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    // 表达式求值（融合世界入口；P = 输出精度）
    // ══════════════════════════════════════════════════════════════════════
    [[nodiscard]] Result<Tensor> eval_expr(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32) override
    {
        trace_variant_(spec, inputs, P);
        if (P == Precision::F32 && all_f32(inputs))
            return inner_.eval_expr(spec, inputs, rows, cols);
        // ── in-kernel f16 优先（Phase 2）────────────────────────────────────
        // 内层引擎有该 (结构, 输入精度, 输出精度) 的**带类型**融合 shader →
        // 直接把原张量交给它（读 f16 / 写 f16，算术 f32）：不再为每个算子物化
        // f32 副本（那正是训练 transient 膨胀 2.4× 的根因）。
        if (inner_.supports_expr_precision_variant(spec, inputs, P))
            return inner_.eval_expr(spec, inputs, rows, cols, P);
        if (prec_trace_enabled_())
            trace_miss_(spec, inputs, P);
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto r = inner_.eval_expr(spec, *in32, rows, cols);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), P);
    }

    [[nodiscard]] Result<Tensor> eval_expr_reduce(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols,
        Precision P = Precision::F32) override
    {
        trace_variant_(spec, inputs, P);
        if (P == Precision::F32 && all_f32(inputs))
            return inner_.eval_expr_reduce(spec, inputs, rows, cols);
        // in-kernel f16 优先（与 eval_expr 同款；归约带类型变体由 gen_fused 生成）
        if (inner_.supports_expr_precision_variant(spec, inputs, P))
            return inner_.eval_expr_reduce(spec, inputs, rows, cols, P);
        if (prec_trace_enabled_())
            trace_miss_(spec, inputs, P);
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto r = inner_.eval_expr_reduce(spec, *in32, rows, cols);
        if (!r)
            return std::unexpected(r.error());
        return to_prec(std::move(*r), P);
    }

    [[nodiscard]] Result<void> eval_expr_into(
        const ExprSpec& spec,
        std::span<const Tensor> inputs,
        std::size_t rows, std::size_t cols, Tensor& out) override
    {
        trace_variant_(spec, inputs, out.precision());
        if (out.precision() == Precision::F32 && all_f32(inputs))
            return inner_.eval_expr_into(spec, inputs, rows, cols, out);
        // in-kernel f16 优先：目标传递 + 带类型变体 = 直接读写原存储，零临时量
        if (inner_.supports_expr_precision_variant(spec, inputs, out.precision()))
            return inner_.eval_expr_into(spec, inputs, rows, cols, out);
        // 目标可能是 f16：在 f32 空间求值后 cast_into 写回 out 的原存储
        // （out 常同时是输入之一——in32 里的 f32 拷贝即"读旧值"，语义正确）
        if (prec_trace_enabled_())
            trace_miss_(spec, inputs, out.precision());
        auto in32 = to_f32_all(inputs);
        if (!in32)
            return std::unexpected(in32.error());
        auto tmp = inner_.create_tensor(rows, cols, Precision::F32);
        auto r = inner_.eval_expr_into(spec, *in32, rows, cols, tmp);
        if (!r)
            return std::unexpected(r.error());
        return inner_.cast_into(tmp, out);
    }
};

} // namespace nn
