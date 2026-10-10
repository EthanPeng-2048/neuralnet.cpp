#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "compute_engine.hpp"
#include "compute_tensor.hpp"
#include "model_spec.hpp"
#include "precision.hpp"
#include "expr_dsl.hpp"

namespace nn
{

// ── NN_F16_DEBUG：中间量扫描（CPU f16 发散定位诊断）─────────────────────
// 探针门（core_config.hpp 的 NN_PROBES_ENABLED）：默认构建里本函数是**空内联**，
// 20 个反向调用点零指令、零字符串；要排障时用 `-DNN_ENABLE_PROBES=ON` 重新配置
// 构建，再设 `NN_F16_DEBUG=1` 选择性地开（同一份探针构建里逐算子开关）。
#if NN_PROBES_ENABLED
inline void nn_dbg_scan(const char* tag, ComputeEngine& eng, const Tensor& t)
{
    static const bool on = nn::dsl::env_flag("NN_F16_DEBUG");
    if (!on) return;
    std::vector<Scalar> buf(t.rows() * t.cols());   // 宿主桥（17 §3 D11）：不经 Matrix
    if (auto er = detail::download_span(eng, t, std::span(buf)); !er)
    {
        std::fprintf(stderr, "[bwd][%s] download fail: %s\n", tag,
                     er.error().message.c_str());
        return;
    }
    double mx = 0.0;
    bool bad = false;
    for (Scalar v : buf)
    {
        if (!std::isfinite(v)) bad = true;
        else mx = std::max(mx, std::fabs(static_cast<double>(v)));
    }
    std::fprintf(stderr, "[bwd][%s] %s max=%.6g%s\n", tag, t.shape_str().c_str(),
                 mx, bad ? "  <<< NONFINITE" : "");
    std::fflush(stderr);
}
#else
inline void nn_dbg_scan(const char*, ComputeEngine&, const Tensor&) noexcept {}
#endif

// ══════════════════════════════════════════════════════════════════════════
// Layer — 引擎化计算层基类
// ══════════════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════════════
// PrecisionSet / PrecisionSupport — 层能力声明（docs 21 §4.2，裁决 R1）
//
// 背景（docs 21 §2）：`Precision` 是"精度词汇表"，加一个枚举值等于给**所有模型
// 的所有槽位**多一个可选项。若不声明"谁能用"，就会制造 18 §3.1 C2/C3 那类缺陷
// （"可组合但必崩，且无任何启动期提示"、"静默回落另一个实现"）。
//
// 解法：**层按槽声明自己支持哪些精度**，框架在 `Layer::init` 这个唯一咽喉处校验，
// 不兼容组合 = 初始化期报错（带层名 + 槽位 + 取值 + 该槽允许集合）。
//
// 为什么**按槽**而不是一个扁平集合：`BitLinear` 需要 `param ∈ {T1_58}` 而
// `compute/stable/optimizer ∈ {f16,f32}`；扁平 `{T1_58}` 会让
// `{param:T1_58, compute:T1_58}` 通过校验，然后**每个激活算子都被要求输出三值**。
//
// 位掩码实现：无分配、可平凡拷贝、可 constexpr，且留足扩展（uint32 = 32 种精度）。
// ══════════════════════════════════════════════════════════════════════════
struct PrecisionSet
{
    std::uint32_t mask = 0;

    [[nodiscard]] constexpr bool has(Precision p) const noexcept
    {
        return (mask & (1u << static_cast<std::uint32_t>(p))) != 0;
    }
    constexpr void add(Precision p) noexcept
    {
        mask |= (1u << static_cast<std::uint32_t>(p));
    }
    [[nodiscard]] static constexpr PrecisionSet of(std::initializer_list<Precision> list) noexcept
    {
        PrecisionSet s;
        for (Precision p : list) s.add(p);
        return s;
    }

    // 诊断用：按枚举序罗列（"f16|f32"）——进错误信息，不进任何热路径。
    [[nodiscard]] std::string to_string() const
    {
        std::string s;
        for (const Precision p : {Precision::F16, Precision::F32, Precision::BF16,
                                 Precision::F64, Precision::T1_58})
        {
            if (!has(p)) continue;
            if (!s.empty()) s += '|';
            s += precision_name(p);
        }
        return s.empty() ? std::string("(空集)") : s;
    }
};

// 舍入精度集合 {f16, f32}：**一切既有层的默认能力**（§4.2 —— 这是"不影响其他层"
// 的落实方式：默认声明 = 现状，既有层一行不改）。
[[nodiscard]] constexpr PrecisionSet rounding_precision_set() noexcept
{
    return PrecisionSet::of({Precision::F16, Precision::F32});
}

// 按槽的精度能力（四槽与 PrecisionProfile 一一对应；语义见 §4.3.2：
// param = "该层 forward 的**有效权重精度**"，不是"参数张量的存储布局"）。
struct PrecisionSupport
{
    PrecisionSet param, compute, stable, optimizer;

    [[nodiscard]] static constexpr PrecisionSupport rounding_all_slots() noexcept
    {
        const PrecisionSet s = rounding_precision_set();
        return PrecisionSupport{s, s, s, s};
    }
};

// ── P1.5：三值（T1_58）profile 的「复合层分流」──────────────────────────
// 语义（docs 21 §4.3.2 + §4.8.3）：`param = T1_58` 描述的是**线性子层
// （BitLinear）的权重是三值**，不是"本层自己的存储布局是三值"——
// T1_58 不是存储精度，拿它建张量会被引擎拒绝。
// 因此持有线性子层的复合层（Attention / FeedForward / GPTBlock / GPTModel /
// RAPT*）必须把一次 profile 拆成两份：
//   · `self`   —— param 归一化为 f32（复合层自身不持有三值权重；它自建的张量
//                 如 token 嵌入按 f32 创建），其余三槽原样；
//   · `linear` —— 三值模式（weight_quant=T1_58）下 param 置为 T1_58，
//                 否则 == self → 交给 BitLinear / Linear。
// 非三值入参（param != T1_58）时 `self == linear == 入参` → 既有路径逐位不变。
//
// **配置自相矛盾 = fail-fast**：本层未以三值模式构造（weight_quant=None）却
// 收到 param=T1_58，说明有人把三值声明打到了普通 Linear 上——普通 Linear
// 承载不了三值权重，此时**不静默降级成 f32**，直接报错（否则会得到"配置了
// 三值、实际跑 f32"的静默错值路径）。
struct TernaryProfileSplit
{
    PrecisionProfile self;    // 自身 / 非三值子层
    PrecisionProfile linear;  // 直接持有的线性子层（BitLinear 或 Linear）
};

[[nodiscard]] inline TernaryProfileSplit split_ternary_profile(
    const char* layer_name, const PrecisionProfile& profile, bool ternary)
{
    if (!ternary && profile.param == Precision::T1_58)
    {
        NN_CHECK(false,
                 std::string("层 ") + layer_name + "：收到 param=t1_58，但该层不是"
                 "三值模式（weight_quant=None）—— 三值权重只对 BitLinear 有意义；"
                 "请以 weight_quant=t1_58 构造该层，或把 param 改回 f16/f32"
                 "（docs/development/21-quantized-weights.md §4.8.3）");
    }
    TernaryProfileSplit out{profile, profile};
    if (out.self.param == Precision::T1_58) out.self.param = Precision::F32;
    if (ternary) out.linear.param = Precision::T1_58;
    return out;
}

// 「param 归一化」单用版（给自建张量的复合层用：`GPTModel` 的 token 嵌入）。
[[nodiscard]] inline PrecisionProfile without_ternary_param(
    PrecisionProfile profile) noexcept
{
    if (profile.param == Precision::T1_58) profile.param = Precision::F32;
    return profile;
}

// 「profile → 线性层种类」的**唯一**约定（与 MLP 工厂 `build_mnist_mlp_model`
// 的 `ternary = (precision.param == T1_58)` 同源）：模型层构造时若 profile 的
// param 槽声明了 T1_58，就等价于 weight_quant=T1_58（显式开关优先保留）。
// 这样 CLI `--precision-param t1_58`、`ModelSpec.weight_quant` 与库 API 三条
// 路径收敛到同一个判据，不存在"声明了三值却构造出普通 Linear"的组合。
[[nodiscard]] inline WeightQuant effective_weight_quant(
    const PrecisionProfile& profile, WeightQuant explicit_quant) noexcept
{
    return (profile.param == Precision::T1_58) ? WeightQuant::T1_58 : explicit_quant;
}

class Layer
{
protected:
    // 梯度检查点模式：为 true 时 forward 不保留逐层中间激活（供激活重计算）。
    bool checkpoint_mode_ = false;

    // D7：精度配置（§9.2，Model 在 add/构造 Layer 时注入，每层一份）
    // 默认全 F32 = 现状行为，零回归（G5）
    PrecisionProfile p_;

    // ── M6 段 C（17 §5 M6 / 15 §3.3）：每调用 engine 形参删除 ─────────────
    // 层在 `init(engine)` 时绑定（公共 init 是 NVI 包装，绑定后转 init_impl），
    // forward/backward/zero_grad/forward_recompute 不再收 engine 形参。
    // 方法体里继续用局部名 `engine`：函数首行 `ComputeEngine& engine = *engine_;`
    // （DSL 入口按 15 §3.3 保留形参，层传的就是这个局部引用）。
    ComputeEngine* engine_ = nullptr;

    // 绑定读取：未 init 就跑 forward 是调用方错误 → fail-fast 前先打**调用点**
    //（source_location 默认参在调用处求值 → 能直接定位到是哪个层的哪个方法）。
    [[nodiscard]] ComputeEngine& engine_ref(
        std::source_location loc = std::source_location::current()) const
    {
        if (!engine_)
        {
            std::fprintf(stderr,
                         "[layer-init] Layer: engine 未绑定（init(engine) 未被调用）"
                         " —— 调用点 %s:%d\n",
                         loc.file_name(), static_cast<int>(loc.line()));
            NN_ASSERT(false, "Layer: engine 未绑定——先调 init(engine)（Model::add 会自动调）");
        }
        return *engine_;
    }

public:
    virtual ~Layer() = default;

    // ── 层名（诊断设施）─────────────────────────────────────────────────
    // 能力校验的错误信息必须能定位到"哪一层"（此前 Layer 无任何名称设施，
    // 诊断只能靠 file:line）。各层 override 返回自己的名字；复合层返回自身名。
    [[nodiscard]] virtual const char* layer_name() const noexcept { return "Layer"; }

    // ── D7：精度配置注入（§9.2）─────────────────────────────────────────
    // virtual：复合层/持有辅助对象（RoPE、位置编码器）的层需要把 profile 继续
    // 下传（否则辅助对象内的 DSL 求值退回 F32，静默丢掉 f16 存储收益）。
    //
    // **契约（D3(b)）**：profile 在 `init(engine)` 之后**不可变**——已 init 时
    // 调用即 fail-fast。"profile 何时有效"因此从"三条注入路径的交集"收敛成一条
    // 可判定规则（docs 21 §4.3.2）：校验（唯一的、发生在 init 内的）不可能被旁路。
    // 复合层的 override 必须先调本函数（先校验），再逐子层下传。
    virtual void set_precision_profile(const PrecisionProfile& profile)
    {
        if (engine_ != nullptr)
        {
            NN_CHECK(false,
                     std::string("set_precision_profile: 层已 init，精度 profile 之后不可变（层=")
                         + layer_name() + "）—— 请在 init(engine) 之前注入"
                         "（docs/development/21-quantized-weights.md §4.3.2(b)）");
        }
        p_ = profile;
    }
    [[nodiscard]] const PrecisionProfile& precision_profile() const noexcept { return p_; }

    // ── 层能力声明（R1，§4.2）──────────────────────────────────────────
    // 默认 = 现状（f16/f32 × 四槽）→ 既有层零改动。声明支持的层 override。
    [[nodiscard]] virtual PrecisionSupport precision_support() const
    {
        return PrecisionSupport::rounding_all_slots();
    }

    // forward/backward 只写一次，CPU/GPU 由引擎实现自动分发（引擎来自 init 绑定）
    [[nodiscard]] virtual Result<Tensor> forward(const Tensor& input) = 0;

    [[nodiscard]] virtual Result<Tensor> backward(const Tensor& grad_output) = 0;

    // 参数访问（供 optimizer 使用）— 使用 reference_wrapper 替代裸指针，明确表达非拥有语义
    [[nodiscard]] virtual std::vector<TensorRef> parameters() { return {}; }
    [[nodiscard]] virtual std::vector<TensorRef> param_gradients() { return {}; }

protected:
    // ── 复合层参数收集（消除各复合层 parameters()/param_gradients() 的同构级联）──
    // 按实参顺序拼接若干列表；实参可以是 `Tensor`（隐式变成单元素）或
    // `std::vector<TensorRef>`（子层级联结果）。用法：
    //     return collect_refs(token_emb_, pos_encoder_->parameters(),
    //                         forward_block_refs_(&GPTBlock::parameters),
    //                         lm_head_.parameters());
    // 语义与手写的 `p.insert(p.end(), x.begin(), x.end())` 序列完全一致
    // （顺序就是实参顺序，不改变优化器看到的参数顺序）。
    [[nodiscard]] static std::size_t ref_count_impl_(const Tensor&) noexcept { return 1; }
    [[nodiscard]] static std::size_t ref_count_impl_(const std::vector<TensorRef>& v) noexcept
    { return v.size(); }
    static void append_refs_impl_(std::vector<TensorRef>& out, Tensor& t)
    { out.push_back(t); }
    static void append_refs_impl_(std::vector<TensorRef>& out, const std::vector<TensorRef>& v)
    { out.insert(out.end(), v.begin(), v.end()); }

    template <typename... Groups>
    [[nodiscard]] static std::vector<TensorRef> collect_refs(Groups&&... groups)
    {
        std::vector<TensorRef> out;
        out.reserve((std::size_t{0} + ... + ref_count_impl_(groups)));
        (append_refs_impl_(out, std::forward<Groups>(groups)), ...);
        return out;
    }

    // 把「对块容器逐块调用某成员函数」的结果拼起来（三个模型的 blocks_ 级联共用）。
    //   RefGetter = 指向 Layer 子类 parameters / param_gradients 的成员函数指针。
    template <typename Blocks, typename RefGetter>
    [[nodiscard]] static std::vector<TensorRef> collect_block_refs_(
        Blocks& blocks, RefGetter getter)
    {
        std::vector<TensorRef> out;
        for (auto& b : blocks)
        {
            auto r = (b.*getter)();
            out.insert(out.end(), r.begin(), r.end());
        }
        return out;
    }

    // ── Pre-Norm 残差块反向（GPTBlock / TransformerEncoderLayer / RAPTBlock 共用）──
    // 前向：x2 = x1 + sub1(norm1(x))；out = x2 + sub2(norm2(x2))
    // 反向：grad_s2  = sub2.bwd(grad_out)
    //       grad_x2  = grad_out + norm2.bwd(grad_s2)     ← 残差 2 分流
    //       grad_s1  = sub1.bwd(grad_x2)
    //       grad_x1  = grad_x2 + norm1.bwd(grad_s1)      ← 残差 1 分流
    // 返回 grad_x1。
    //   add_prec = 两处残差相加的**表达式输出精度**。各块历史行为不同（GPT/
    //   Transformer 传 p.compute；RAPT 的残差相加历史上未传 profile、恒 F32），
    //   故显式传入以保持逐位不变——不要"顺手统一"成 p.compute。
    [[nodiscard]] static Result<Tensor> prenorm_residual_backward_(
        ComputeEngine& engine, const Tensor& grad_output,
        Layer& norm1, Layer& sub1, Layer& norm2, Layer& sub2,
        Precision add_prec)
    {
        NN_TRY(grad_s2, sub2.backward(grad_output));
        NN_TRY(b_n2, norm2.backward(*grad_s2));
        auto grad_x2 = dsl::compute(engine,
            dsl::leaf(grad_output) + dsl::leaf(*b_n2),
            grad_output.rows(), grad_output.cols(), add_prec);
        NN_TRY_CHECK(grad_x2);
        NN_TRY(grad_s1, sub1.backward(*grad_x2));
        NN_TRY(b_n1, norm1.backward(*grad_s1));
        return dsl::compute(engine,
            dsl::leaf(*grad_x2) + dsl::leaf(*b_n1),
            grad_x2->rows(), grad_x2->cols(), add_prec);
    }

    // ── Pre-Norm 残差块前向的一个残差分支 ─────────────────────────────────
    // 返回 x + sub(norm(x))（GPT/Transformer/RAPT 三块的两个残差分支共用）。
    //   add_prec 同上：显式传以保持各块历史精度行为逐位不变。
    [[nodiscard]] static Result<Tensor> prenorm_residual_forward_(
        ComputeEngine& engine, const Tensor& x, Layer& norm, Layer& sub,
        Precision add_prec)
    {
        NN_TRY(n, norm.forward(x));
        NN_TRY(s, sub.forward(*n));
        return dsl::compute(engine,
            dsl::leaf(x) + dsl::leaf(*s),
            x.rows(), x.cols(), add_prec);
    }

public:

    // 梯度清零（每个训练 step 开始前调用）
    [[nodiscard]] virtual Result<void> zero_grad()
    {
        ComputeEngine& engine = engine_ref();
        for (auto& grad : param_gradients())
        {
            NN_TRY(r, engine.zero(grad));
        }
        return {};
    }

    // batch 录制粒度控制（默认 no-op，仅 GPTModel override）
    virtual void set_flush_interval(std::size_t /*interval*/) {}

    // 文档感知：设置当前 step 每样本文档 id（默认 no-op，GPTModel override）
    virtual void set_doc_ids(std::span<const std::size_t> /*ids*/) {}

    // 训练/推理模式切换（默认 no-op，BatchNorm 等需要 override）
    virtual void set_training(bool /*training*/) {}

    // 非可学习状态收集（默认空，BatchNorm 的 running 统计量等需要 override）
    [[nodiscard]] virtual std::vector<TensorRef> extra_state() { return {}; }

    // 引擎相关初始化（创建/上传权重张量），替换构造函数中的 NN_ASSERT 模式。
    // **NVI（M6 段 C）**：公共入口负责绑定 engine_，各层 override `init_impl`
    //（层代码不感知绑定，改名由编译器穷尽驱动：`init(...) override` → `init_impl`）。
    // 默认实现空操作；各层在构造后由 Model::add<T>() 调用（测试里也必须调一次）。
    //
    // **精度能力校验（R3，docs 21 §4.3）**：这是全仓**唯一**的校验咽喉 ——
    // `Model::add<T>()` 先注入 profile 再 init；工厂路径（构造器注入）也在 init 前
    // 完成注入；复合层的 `init_impl` 逐个 init 子层 → 叶子层各自校验即可，
    // **不需要任何聚合/递归求交逻辑**（那会把静态声明变成递归求交，复杂度升一个量级）。
    // 校验必须在 init_impl **之前**：init_impl 会用 p_.param 建权重张量
    //（T1_58 走到那里才会报"不是存储精度"，离根因太远）。
    [[nodiscard]] Result<void> init(ComputeEngine& engine)
    {
        engine_ = &engine;
        NN_TRY_CHECK(check_precision_support_());
        return init_impl(engine);
    }

protected:
    // ── 能力校验实现：把 profile 的四个槽逐个对照 precision_support() ────────
    [[nodiscard]] Result<void> check_precision_support_() const
    {
        const PrecisionSupport sup = precision_support();
        const PrecisionProfile& p = p_;
        if (!sup.param.has(p.param))
            return precision_slot_error_("param", p.param, sup.param);
        if (!sup.compute.has(p.compute))
            return precision_slot_error_("compute", p.compute, sup.compute);
        if (!sup.stable.has(p.stable))
            return precision_slot_error_("stable", p.stable, sup.stable);
        if (!sup.optimizer.has(p.optimizer))
            return precision_slot_error_("optimizer", p.optimizer, sup.optimizer);
        return {};
    }

    // 错误信息口径（§7 验收 2）：层名 + 槽位 + 取值 + 该槽**允许的集合**
    [[nodiscard]] Result<void> precision_slot_error_(const char* slot, Precision got,
                                                     const PrecisionSet& allowed) const
    {
        NN_FAIL(std::string("精度能力校验失败: 层 ") + layer_name()
                + " 不支持 " + slot + "=" + precision_name(got)
                + "（该槽允许: " + allowed.to_string()
                + "）—— 见 docs/development/21-quantized-weights.md §4.2");
    }

public:

    // 梯度检查点（激活重计算）契约 ──────────────────────────────────
    // checkpoint_mode_ = true 时，forward 不保留中间激活（供 L1 激活重计算）；
    // forward_recompute 重算 forward 并重建缓存（供 backward 使用）。
    virtual void set_checkpoint_mode(bool enabled) { checkpoint_mode_ = enabled; }
    [[nodiscard]] bool checkpoint_mode() const noexcept { return checkpoint_mode_; }

    // 释放本层为 backward 保留的中间激活缓存（清空成员缓存 Tensor，归还显存）。
    // 由 GPTModel 在 checkpoint 块 backward 之后调用，避免重算的激活跨块累积
    // （否则所有块缓存会在 backward 末尾同时驻留，抵消检查点的显存收益）。
    // 默认 no-op；各缓存持有层 override。
    virtual void clear_cache() {}

    // 返回本层 backward 所需的中间激活缓存引用（供 activation offload 导出/导入）。
    // 仅返回 valid 的张量；掩码等小而常驻的缓存不在此列（不参与 offload，保持常驻）。
    [[nodiscard]] virtual std::vector<TensorRef> activation_cache() { return {}; }

    // 该层是否可作为“重计算单元”（即 forward_recompute 有实际意义）
    [[nodiscard]] virtual bool recompute_supported() const { return false; }

    // 从保存的输入重算 forward，重建本层 backward 所需的中间缓存。
    // 默认实现：临时关闭 checkpoint 模式重跑 forward（保留缓存）再恢复。
    // 复合层（GPTBlock 等）override 以同时关闭子层的 checkpoint 模式。
    [[nodiscard]] virtual Result<Tensor> forward_recompute(const Tensor& saved_input)
    {
        const bool prev = checkpoint_mode_;
        checkpoint_mode_ = false;
        auto r = forward(saved_input);
        checkpoint_mode_ = prev;
        return r;
    }

protected:
    // 各层的初始化实现（原 `init(ComputeEngine&)` override 全部改名至此）
    [[nodiscard]] virtual Result<void> init_impl(ComputeEngine& /*engine*/) { return {}; }

public:

    // 梯度检查点粒度（默认 0 = 不启用；由 GPTModel 等 override）
    virtual void set_checkpoint_every(std::size_t /*stride*/) {}

    // activation offload（L1-offload）开关（默认 no-op；GPTModel override）
    virtual void set_activation_offload(bool /*enabled*/) {}

    // 理论 offload RAM 字节数（各层累计；默认 0，GPTModel override）
    [[nodiscard]] virtual std::size_t offload_ram_bytes() { return 0; }
};

// ── 辅助：深拷贝 Tensor（通过 engine.clone()，无 PCIe 传输） ──────────────
// 用于需要修改中间结果但不影响原 Tensor 的场景（如 LayerNorm 中的 diff）
// ══════════════════════════════════════════════════════════════════════════
// ActivationOffloader — 通用激活 offload（L1-offload）
//
// 把「本层 backward 所需的中间激活」（由 Layer::activation_cache() 枚举）
// 逐个写进一块持久 host-visible slab，随后把成员张量置空以释放 device-local
// 显存；backward 入口再从 slab 恢复。与具体层无关：GPTBlock / RAPTBlock 共用
// 同一实现（避免两份易漂移的拷贝）。
//
// 用法：
//   offloader.set_enabled(true);
//   ... forward ...
//   offloader.export_activations(engine, activation_cache());  // 导出并释放
//   ... backward 入口 ...
//   offloader.import_activations(engine);                      // 恢复
//
// 注意：export 依赖 forward 后成员地址稳定（refs_ 持有成员引用）；
//       无有效缓存时（如 checkpoint 模式 forward 不驻留激活）静默跳过。
// ══════════════════════════════════════════════════════════════════════════
class ActivationOffloader
{
private:
    bool enabled_ = false;
    bool offloaded_ = false;
    Tensor slab_;                                   // 持久 host-visible 缓冲（跨 step 复用）
    std::vector<TensorRef> refs_;                   // 各激活成员引用（地址稳定）
    std::vector<std::pair<std::size_t, std::size_t>> shapes_;  // 各激活形状
    std::vector<std::size_t> offsets_;              // 各激活在 slab 中的 float 偏移

public:
    void set_enabled(bool enabled) noexcept
    {
        enabled_ = enabled;
        if (!enabled) { slab_ = Tensor{}; offloaded_ = false; }  // 释放持久缓冲
    }
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] bool offloaded() const noexcept { return offloaded_; }

    // 导出：逐个写入 slab，写入后置空成员张量（数据已在 host slab）
    [[nodiscard]] Result<void> export_activations(
        ComputeEngine& engine, std::vector<TensorRef> refs)
    {
        if (!enabled_ || offloaded_) return {};
        refs_ = std::move(refs);
        offsets_.clear();
        shapes_.clear();
        // slab 容量校验：slab 必须容纳**本次**导出的激活总量（批大小/
        // 序列长度变化、--resume 后续训、最后一个不满 batch 之后的 step 等
        // 都会改变总量）；若容量不足，offload_save 会按新 offset 越界写
        // slab → 缓冲区破坏/设备丢失。因此每次导出都按当前总量校验，
        // 不足则重建。
        std::size_t needed = 0;
        for (auto& ref : refs_)
            if (ref.get().valid()) needed += ref.get().size();
        if (needed == 0) { offloaded_ = false; return {}; }
        if (!slab_.valid() || slab_.size() < needed)
        {
            NN_TRY(slab, engine.create_offload_buffer(needed));
            slab_ = std::move(*slab);
        }
        std::size_t offset = 0;
        for (auto& ref : refs_)
        {
            if (!ref.get().valid()) continue;
            NN_TRY(r, engine.offload_save(slab_, offset, ref.get()));
            shapes_.push_back({ref.get().rows(), ref.get().cols()});
            offsets_.push_back(offset);
            offset += ref.get().size();
            ref.get() = Tensor{};   // 释放 GPU 版（数据已在 host slab）
        }
        offloaded_ = true;
        return {};
    }

    // 导入：从 host slab 恢复激活到缓存成员（backward 前调用，替代重计算）
    [[nodiscard]] Result<void> import_activations(ComputeEngine& engine)
    {
        if (!offloaded_) return {};
        for (std::size_t i = 0; i < offsets_.size(); ++i)
        {
            auto t = engine.offload_restore(slab_, offsets_[i],
                                            shapes_[i].first, shapes_[i].second);
            NN_TRY_CHECK(t);
            refs_[i].get() = std::move(*t);
        }
        offloaded_ = false;
        return {};
    }

    // 实际 slab 字节数（诊断用；未创建时 0）
    [[nodiscard]] std::size_t slab_bytes() const noexcept
    {
        return slab_.valid() ? slab_.size() * sizeof(float) : 0;
    }
};

[[nodiscard]] inline Result<Tensor> clone_tensor(
    ComputeEngine& engine, const Tensor& src)
{
    return engine.clone(src);
}

// ══════════════════════════════════════════════════════════════════════════
// 采样：temperature 缩放 → 数值稳定 softmax → 随机采样 / 贪心（argmax）
//
// GPTModel / RAPTModel 的 generate() 各自逐 token 维护不同的运行态
// （KV cache / RLA 运行态），但"从末位 logits 选下一 token"这一步两者
// **逐字相同** → 收敛到本函数（曾有三份拷贝，改一处要改三处；第三份属
// 于已移除的 ZiPTModel，见 docs/history.md）。
//
//   logits        就地变为概率（调用方不需要原值）
//   temperature   >0 随机采样（1.0 = 不缩放但仍采样）；<=0 贪心
//   消耗恰好一次 dist(rng)（仅采样分支）——RNG 消耗序与旧实现逐位一致
// ══════════════════════════════════════════════════════════════════════════
[[nodiscard]] inline std::size_t sample_next_token_(
    std::vector<Scalar>& logits, Scalar temperature,
    std::mt19937_64& rng, std::uniform_real_distribution<Scalar>& dist)
{
    const std::size_t n = logits.size();
    NN_ASSERT(n > 0, "sample_next_token_: empty logits");
    if (temperature > 0.0 && temperature != 1.0)
        for (auto& v : logits) v /= temperature;

    // softmax（数值稳定）
    Scalar max_val = logits[0];
    for (std::size_t v = 1; v < n; ++v)
        max_val = std::max(max_val, logits[v]);
    Scalar sum_exp = 0.0;
    for (auto& v : logits)
    {
        v = std::exp(v - max_val);
        sum_exp += v;
    }
    for (auto& v : logits) v /= sum_exp;

    if (temperature > 0.0)
    {
        const Scalar r = dist(rng);
        Scalar cumulative = 0.0;
        std::size_t next = n - 1;
        for (std::size_t v = 0; v < n; ++v)
        {
            cumulative += logits[v];
            if (r <= cumulative) { next = v; break; }
        }
        return next;
    }
    std::size_t next = 0;
    Scalar best = logits[0];
    for (std::size_t v = 1; v < n; ++v)
        if (logits[v] > best) { best = logits[v]; next = v; }
    return next;
}

} // namespace nn
