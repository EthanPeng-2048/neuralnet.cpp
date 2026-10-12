#pragma once

// ── compute_layer_bitlinear.hpp — 三值（1.58-bit）线性层 BitLinear ──────────
//
// 设计：docs/todo/21-quantized-weights.md §4.4（P1 形态）。
// 前置证据：research/ternary_ste/REPORT.md（同一套 DSL 组合的 STE 训练 PoC）、
//           research/ternary_scale/REPORT.md（尺度 = 逐行 absmean 的实测依据）。
//
// 名词（§4.3.2 语义，务必先读）：
//   · `param = T1_58` 表示"本层 forward 的**有效权重精度**是三值"，
//     **不是**"参数张量的存储布局是三值" —— 所以：
//       latent_w_ : f32 参数（优化器更新对象 + STE 梯度落点），**不落盘为三值**；
//       wq_       : f16 量化缓冲（{-1,0,+1} 在 f16 下精确可表示），每步重算；
//       tau_      : (out,1) f32 逐行尺度（absmean），同样是派生物。
//   · forward 用三值权重 + 尺度做**去量化**点积；backward 用 STE
//     （把量化器当恒等映射求导）把梯度送回 f32 latent。
//
// 算法（只在此处，不在 Engine/Shader）：
//   τ_o   = (1/K)·Σ_k |W[o,k]|                      （逐输出通道 absmean）
//   wq    = select(|W|>0.5τ → ±1, 否则 0)           （= RoundClip(W/τ,-1,1)；等值点差异见下）
//   Y     = (wq · X)·τ + b                          （τ 逐行广播，b 逐行广播）
//   dX    = wqᵀ · (dY∘τ)                            （等价于 (τ∘wq)ᵀ·dY）
//   dW    = (dY · Xᵀ)∘τ                             （STE：∂wq/∂W ≡ 1，含去量化尺度）
//   db    = Σ_batch dY
//
// 与 f16/f32 的**正交性**（§8）：三值是"换一种权重表示"（离散化 + 需要 γ），
// 不是"换更窄的容器"（舍入）。因此它只占 param 槽，且不进提升序
//（precision_rank 无定义）——见 precision.hpp。
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "compute_engine.hpp"
#include "compute_layer_base.hpp"
#include "compute_tensor.hpp"
#include "expr_dsl.hpp"
#include "precision.hpp"

namespace nn
{

class BitLinear final : public Layer
{
private:
    std::size_t in_features_;
    std::size_t out_features_;

    // ── 可训练参数（f32 latent；优化器与 STE 的作用点）────────────────────
    Tensor latent_w_;    // (out_features, in_features)
    Tensor b_;           // (out_features, 1) —— 偏置不做三值量化（BitNet 同款）
    Tensor grad_w_;      // 与 latent_w_ 同形
    Tensor grad_b_;      // 与 b_ 同形

    // ── 量化派生物（每步重算；不落盘、不进序列化）────────────────────────
    Tensor wq_;          // (out_features, in_features) f16，值 ∈ {-1,0,+1}
    Tensor wq_packed_;   // (out, ceil(in/16)) f32 容器，2 bit/权重（P2/F3 路径）
    Tensor tau_;         // (out_features, 1) f32，逐行 absmean

    // 缓冲形态：**打包容器是默认形态**（训练与解码同一条路）。层不查引擎
    // 能力 —— 不支持打包的引擎由 trit_pack/matmul_q 自行 fail-fast。
    // f16 稠密缓冲仅作为 `force_pack_ = 0` 的 A/B 对照形态保留（数值口径由
    // src/ternary_gemm_test.cpp 对拍）。
    bool use_packed_ = false;

    // 打包路径开关（F3 的实测结论，2026-10-11，102/40HX）：
    //   · 默认 = 打包（-1）；0/1 = 强制关/开（A/B 与回退用）。
    // 为什么不再按激活宽度选址：**分块解码版** `matmul_q.comp` 落地后，打包路径在
    // 宽 batch（训练）下也与 f16 稠密分块 GEMM 持平 —— 实测 d256/L4 60 步
    // 2.0 s vs 2.0 s、d512/L8 30 步 5.2 s vs 5.1 s；同时省下 wq 的
    // 2 B → 0.25 B/权重（整卡峰值 −7.5%~8.4%）。此前"每输出元素独立解码"的
    // 初版在宽 batch 下慢 2×（4.0 s），那条启发式已被分块版取代。
    int force_pack_ = -1;

    // ── 部署态（P2 部署格式，ternary_deploy.hpp）──────────────────────────
    // true = 本层只有"打包字 + τ"（来自部署文件），**没有 latent / 没有梯度**：
    // 推理足迹 4.25 B/权重 → 0.25 B/权重。代价是只读 —— 反向 / 保存 / 取参数
    // 全部 fail-fast（见下面的守卫），避免"以为在训练其实在推一个残层"。
    bool packed_only_ = false;

    Tensor input_cache_; // forward 输入缓存（backward 用）

    // ── 量化缓存（F1）：只在**权重变过**时重算 τ/wq ──────────────────────
    // 判据 = `latent_w_.write_epoch()`（存储级写版本，任何经引擎媒介的参数
    // 写入都会推进它，见 ComputeEngine::WriteEpochGuard）。为什么必须有：
    // forward 的调用频率远高于参数更新频率 ——
    //   · 梯度累积 `--accum-steps N`：一个 optimizer step 有 N 次 forward；
    //   · 自回归生成：每个 token 每层一次 forward（KV cache 只省注意力）；
    //   · 激活重计算（checkpoint）会再跑一次 forward。
    // 缓存不改变数值：同一权重算出的 τ/wq 逐位相同（见
    // src/write_epoch_test.cpp 的"缓存重算 = 从零重算"逐位断言）。
    std::uint64_t quant_epoch_ = 0;   // 最近一次量化时 latent_w_ 的写版本
    bool quant_valid_ = false;        // 首次 forward 必然量化（epoch 0 不参与判定）
    std::uint64_t quantize_calls_ = 0;  // 量化实际执行次数（诊断/测试用）

public:
    BitLinear(std::size_t in_features, std::size_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    [[nodiscard]] const char* layer_name() const noexcept override { return "BitLinear"; }

    // ── 能力声明（R1，§4.2）────────────────────────────────────────────
    // param **只**接受 T1_58（T1_58 层面"该层权重是三值"的语义锚）；
    // compute/stable/optimizer 仍是 {f16,f32}——激活与数值敏感链绝不能被三值化。
    [[nodiscard]] PrecisionSupport precision_support() const override
    {
        PrecisionSupport sup = PrecisionSupport::rounding_all_slots();
        sup.param = PrecisionSet::of({Precision::T1_58});
        return sup;
    }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // Xavier 均匀分布（与 Linear 同款）；latent/偏置**恒 f32**
        //（§4.3.2：param=T1_58 只是一句声明，绝不拿它去 create_tensor）。
        const Scalar limit = std::sqrt(6.0 / static_cast<Scalar>(in_features_ + out_features_));
        latent_w_ = engine.create_tensor(out_features_, in_features_, Precision::F32,
                                         InitSpec::uniform(-limit, limit, kInitSeed));
        if (!latent_w_.valid()) NN_FAIL("BitLinear: latent 权重初始化失败");
        b_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        if (!b_.valid()) NN_FAIL("BitLinear: 偏置初始化失败");
        // 梯度缓冲**不在此分配**（F2）：惰性 + 只训练时分配，见 ensure_grad_()。
        // 量化缓冲：**打包容器**（2 bit/权重，F3）是默认也是唯一形态 ——
        // 能力判断不下沉到层（引擎不支持时由 matmul_q/trit_pack 自行 fail-fast）。
        use_packed_ = true;
        ensure_quant_buffer_();
        tau_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        if (!tau_.valid()) NN_FAIL("BitLinear: 尺度缓冲初始化失败");
        return {};
    }

    // ── 打包路径开关（逃生阀 / A-B 用）────────────────────────────────────
    // 引擎支持三值 GEMM 时默认开启；显式传 false 可退回 f16 稠密缓冲（P0/A-B
    // 对比与回退用）。切换会换缓冲形态 → 必须作废量化缓存、下一次 forward 重算。
    // 需在 init 之后调用（缓冲分配与 shape 有关）。
    void set_ternary_pack_enabled(bool enabled)
    {
        force_pack_ = enabled ? 1 : 0;
        const bool target = enabled;
        if (target == use_packed_) return;
        use_packed_ = target;
        quant_valid_ = false;
        ensure_quant_buffer_();
    }

    // 恢复"按形状自动选"（默认策略）
    void set_ternary_pack_auto()
    {
        force_pack_ = -1;
        const bool target = true;
        if (target == use_packed_) return;
        use_packed_ = target;
        quant_valid_ = false;
        ensure_quant_buffer_();
    }

    // ── 参数视图：latent（f32）+ 偏置（f32）──────────────────────────────
    [[nodiscard]] std::vector<TensorRef> parameters() override
    {
        return {latent_w_, b_};
    }
    [[nodiscard]] std::vector<TensorRef> param_gradients() override
    {
        ensure_grad_();
        return {grad_w_, grad_b_};
    }

    // ── 参数形态协议（v7）────────────────────────────────────────────────
    // 索引空间恒为 {0 = 权重, 1 = 偏置}（与实例状态无关：打包只读态下 latent
    // 为空张量，但下标不变）。kind 随**文件角色**分流：
    //   Checkpoint(.nnckpt) → TernaryLatent（f32 latent，续训用）
    //   Model(.nnmodel)     → TernaryPacked2（2 bit/权重容器 + τ）
    // 序列化层负责把容器在盘上编码成 base-3（TernaryPacked3，1.6 bpw）并在载入
    // 时解回 2-bit 容器——层只认运行期形态（L2 不碰宿主字节/Matrix）。
    [[nodiscard]] std::size_t param_count() override { return 2; }

    [[nodiscard]] Result<ParamEntry> export_param(std::size_t index, ModelRole role) override
    {
        if (index == 1)   // 偏置：恒 Dense（不做三值量化，BitNet 同款）
            return dense_entry_(b_);
        if (index != 0)
            NN_FAIL("BitLinear::export_param: 参数下标越界（期望 0/1，实际 " +
                    std::to_string(index) + "）");
        ParamEntry e;
        e.rows = out_features_;
        e.cols = in_features_;
        if (role == ModelRole::Checkpoint)
        {
            if (!latent_w_.valid())
                NN_FAIL("BitLinear::export_param: 打包只读态没有 latent —— "
                        "本模型来自推理文件（.nnmodel），不能按断点角色（.nnckpt）导出");
            e.kind = ParamKind::TernaryLatent;
            e.precision = Precision::F32;
            e.payload = latent_w_;
            return e;
        }
        // 推理角色：打包形态（0.25 B/权重容器 + τ）
        NN_TRY(q, ensure_quantized());
        if (!use_packed_ || !wq_packed_.valid() || !tau_.valid())
            NN_FAIL("BitLinear::export_param: 打包派生物不可用（缓冲形态被切到 f16 稠密）；"
                    "推理文件要求打包形态");
        e.kind = ParamKind::TernaryPacked2;
        e.precision = Precision::F32;   // 容器按 f32 张量承载原始字
        e.payload = wq_packed_;
        e.aux = tau_;
        return e;
    }

    [[nodiscard]] Result<void> import_param(std::size_t index, const ParamEntry& entry) override
    {
        if (index == 1)
        {
            if (entry.kind != ParamKind::Dense)
                NN_FAIL(std::string("BitLinear::import_param: 偏置必须是 Dense，实际 kind=") +
                        param_kind_name(entry.kind));
            return import_into_(b_, entry);
        }
        if (index != 0)
            NN_FAIL("BitLinear::import_param: 参数下标越界（期望 0/1，实际 " +
                    std::to_string(index) + "）");
        switch (entry.kind)
        {
            case ParamKind::TernaryLatent:
            {
                if (!latent_w_.valid())
                {
                    // 从打包只读态恢复可训练：重建 f32 latent（零初始化）并回到
                    // 默认打包缓冲形态；量化缓存作废（首次 forward 会重算）。
                    latent_w_ = engine_ref().create_tensor(
                        out_features_, in_features_, Precision::F32, InitSpec::zero());
                    NN_UNWRAP(latent_w_.valid(), "BitLinear: latent 恢复失败");
                    packed_only_ = false;
                    use_packed_ = true;
                    ensure_quant_buffer_();
                }
                quant_valid_ = false;
                return import_into_(latent_w_, entry);
            }
            case ParamKind::TernaryPacked2:
            {
                if (!entry.aux.valid())
                    NN_FAIL("BitLinear::import_param: 打包形态缺少 τ（aux）");
                const std::size_t wpr = nn::trit_pack::words_per_row(in_features_);
                if (entry.rows != out_features_ ||
                    entry.payload.rows() != out_features_ ||
                    entry.payload.cols() != wpr)
                    NN_FAIL("BitLinear::import_param: 打包字形状与层不匹配（期望 " +
                            std::to_string(out_features_) + "x" + std::to_string(wpr) + "）");
                if (entry.aux.rows() != out_features_ || entry.aux.cols() != 1)
                    NN_FAIL("BitLinear::import_param: τ 形状与层不匹配（期望 " +
                            std::to_string(out_features_) + "x1）");
                // 打包只读态：只保留打包字 + τ（推理足迹 0.25 B/权重 + τ）
                force_pack_ = 1;
                use_packed_ = true;
                wq_packed_ = entry.payload;
                tau_ = entry.aux;
                quant_valid_ = true;
                packed_only_ = true;
                latent_w_ = Tensor{};
                wq_ = Tensor{};
                grad_w_ = Tensor{};
                grad_b_ = Tensor{};
                return {};
            }
            default:
                NN_FAIL(std::string("BitLinear::import_param: 不接受该参数形态（kind=") +
                        param_kind_name(entry.kind) +
                        "，可接受 {TernaryLatent, TernaryPacked2}）");
        }
    }

    [[nodiscard]] bool grad_buffers_allocated() const noexcept override
    {
        return grad_w_.valid();
    }

    // ── 量化派生物只读视图（诊断 / 测试 / 序列化对拍用）────────────────────
    [[nodiscard]] const Tensor& latent_weights() const noexcept { return latent_w_; }
    [[nodiscard]] const Tensor& quantized_weights() const noexcept { return wq_; }
    [[nodiscard]] const Tensor& row_scales() const noexcept { return tau_; }
    // 打包路径（P2/F3）：本层是否用 2 bit/权重的打包缓冲（`quantized_weights()`
    // 在打包路径下为空 —— 用 `quantized_weights_packed()` + ternary_pack.hpp 解码）
    [[nodiscard]] bool uses_packed_weights() const noexcept { return use_packed_; }
    [[nodiscard]] const Tensor& quantized_weights_packed() const noexcept
    {
        return wq_packed_;
    }
    // 量化实际执行次数（诊断/测试：缓存是否生效看它，不看 forward 次数）
    [[nodiscard]] std::uint64_t quantize_calls() const noexcept { return quantize_calls_; }

    // ── 部署态（P2 部署格式）──────────────────────────────────────────────
    // 载入"打包字 + τ"并把本层切成**只读推理**态：释放 latent（这一项就是部署态
    // 的全部收益：4 B/权重 → 0）、释放梯度与 f16 稠密缓冲。τ 一并载入是为了
    // **逐位复现**训练时的有效权重 τ∘W（重新量化会得到 mean|trit|·τ 的比例差异）。
    [[nodiscard]] Result<void> load_packed_weights(Tensor packed_words, Tensor tau)
    {
        if (!packed_words.valid() || !tau.valid())
            NN_FAIL("BitLinear: 部署权重为空张量");
        if (packed_words.rows() != out_features_ ||
            packed_words.cols() != nn::trit_pack::words_per_row(in_features_))
            NN_FAIL("BitLinear: 部署打包字形状与层不匹配（期望 " +
                    std::to_string(out_features_) + "x" +
                    std::to_string(nn::trit_pack::words_per_row(in_features_)) + "）");
        if (tau.rows() != out_features_ || tau.cols() != 1)
            NN_FAIL("BitLinear: 部署 τ 形状与层不匹配（期望 " +
                    std::to_string(out_features_) + "x1）");
        force_pack_ = 1;
        use_packed_ = true;
        wq_packed_ = std::move(packed_words);
        tau_ = std::move(tau);
        quant_valid_ = true;
        packed_only_ = true;
        latent_w_ = Tensor{};   // 免 latent（部署态的核心收益）
        wq_ = Tensor{};
        grad_w_ = Tensor{};
        grad_b_ = Tensor{};
        return {};
    }

    // 本层是否处于部署态（无 latent/梯度，只读）
    [[nodiscard]] bool packed_only() const noexcept { return packed_only_; }
    [[nodiscard]] std::size_t in_features() const noexcept { return in_features_; }
    [[nodiscard]] std::size_t out_features() const noexcept { return out_features_; }

    // 导出（部署落盘）用：确保打包派生物与当前 latent 同步。**不改**推理语义，
    // 只在量化缓存失效时补一次量化（与 forward 里的那条路径同一实现）。
    [[nodiscard]] Result<void> ensure_quantized()
    {
        if (needs_quantize_()) return quantize_weights_();
        return {};
    }

    void clear_cache() override
    {
        // 只清**激活**缓存。wq_/τ 是**权重派生物**：有效性由写版本决定
        // （needs_quantize_），清掉只会让下一次 forward 白算一遍全量权重。
        input_cache_ = Tensor{};
    }

    [[nodiscard]] std::vector<TensorRef> activation_cache() override
    {
        // backward 需要 input_cache_（X）。τ **不在**此列：它是权重派生物
        // （随写版本缓存），不是激活 —— 列进 offload 会被搬到 host 再搬回
        // （白付一次搬运），清空后还会让缓存判定失效、触发一次全量重算；
        // 它只有 (out,1)，常驻代价可忽略。
        std::vector<TensorRef> r;
        if (input_cache_.valid()) r.emplace_back(input_cache_);
        return r;
    }

    // ── forward：量化（τ + wq）→ 去量化点积 ─────────────────────────────
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        if (input.rows() != in_features_)
            NN_FAIL("bitlinear forward: input shape mismatch");
        // 部署态：只允许"打包字 + τ"这条路（引擎不支持三值 GEMM 就没法跑）
        if (packed_only_ && !(use_packed_ && wq_packed_.valid() && tau_.valid()))
            NN_FAIL("BitLinear(packed-only)：部署态缺少打包权重/τ（载入失败或缓冲被清）");
        if (!checkpoint_mode_)
            input_cache_ = input;
        // 缓冲形态：默认跟随引擎能力（分块解码版下训练/解码都不亏，见成员注释）；
        // `set_ternary_pack_enabled` 之后走强制模式。换形态即作废量化缓存。
        sync_buffer_mode_();
        // τ / wq 只在 latent 权重变过时重算（同权重下的 N 次 forward 共享）。
        // 重算失败：直接上抛（不静默沿用上一步的旧缓冲）。
        if (needs_quantize_())
        {
            NN_TRY(q, quantize_weights_());
        }
        // 打包路径：Y = (τ∘W)·X 是一趟量化 GEMM（权重读 0.25 B/元素），偏置用
        // 一次行广播加（原语不带 bias，见 compute_engine.hpp 的说明）。
        if (use_packed_)
        {
            NN_TRY(y, engine.matmul_q(wq_packed_, tau_, input, p_.compute));
            return dsl::compute(engine,
                dsl::leaf(*y) + dsl::row_broadcast(b_),
                out_features_, input.cols(), p_.compute);
        }
        return dsl::compute(engine,
            dsl::matmul(wq_, input, false, false) * dsl::row_broadcast(tau_)
                + dsl::row_broadcast(b_),
            out_features_, input.cols(), p_.compute);
    }

    // ── backward：STE（把 wq 当常数；梯度回 f32 latent）──────────────────
    [[nodiscard]] Result<Tensor> backward(const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (grad_output.rows() != out_features_)
            NN_FAIL("bitlinear backward: grad_output shape mismatch");
        if (packed_only_)
            NN_FAIL("BitLinear(packed-only)：部署态只读（没有 latent 可更新）—— "
                    "本文件是推理部署文件，请用训练文件恢复训练");
        ensure_grad_();   // 训练路径：首次反向才建梯度缓冲（F2）
        if (!tau_.valid() || !input_cache_.valid())
            NN_FAIL("bitlinear backward: 必须先 forward（τ/输入缓存未建立）");

        // dX = (τ∘W)ᵀ·dY：τ 是逐输出通道的（落在求和维内），必须在内核里按
        // 存储行取 —— 打包路径直接一趟量化 GEMM（顺带省掉了 dy_scaled 中间量）。
        Tensor grad_input;
        if (use_packed_)
        {
            NN_TRY(gi, engine.matmul_q_t(wq_packed_, tau_, grad_output,
                                         in_features_, p_.compute));
            grad_input = std::move(*gi);
        }
        else
        {
            // f16 稠密缓冲路径：τ 先落到 dY 的行上再左乘（等价形式）
            NN_TRY(dy_scaled, dsl::compute(engine,
                dsl::leaf(grad_output) * dsl::row_broadcast(tau_),
                out_features_, grad_output.cols(), p_.compute));
            auto gi = dsl::compute(engine,
                dsl::matmul(wq_, *dy_scaled, /*transA=*/true, false),
                in_features_, grad_output.cols(), p_.compute);
            NN_TRY_CHECK(gi);
            grad_input = std::move(*gi);
        }

        // dW += (dY·Xᵀ)∘τ（STE：∂wq/∂W ≡ 1，再乘去量化尺度 τ）
        // matmul 段 + 逐行缩放 + 原地累加融合为单次 dispatch（与 Linear 同构）。
        auto grad_w_acc = dsl::compute_into(engine,
            dsl::leaf(grad_w_)
                + dsl::matmul(grad_output, input_cache_, false, true)
                      * dsl::row_broadcast(tau_),
            grad_w_);
        NN_TRY_CHECK(grad_w_acc);

        // db += Σ_batch dY（归约向量 + 原地累加两步；与 Linear 同款理由：
        // 归约向量输出契约要求输出链只经归约/广播视图访问输入）
        NN_TRY(gb, dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::leaf(grad_output)),
            grad_output.rows(), grad_output.cols(), p_.compute));
        auto r2 = dsl::compute_into(engine,
            dsl::leaf(grad_b_) + dsl::leaf(*gb), grad_b_);
        NN_TRY_CHECK(r2);

        return grad_input;
    }

private:
    // 按当前模式策略（force_pack_ / 引擎能力）选定缓冲形态。
    void sync_buffer_mode_()
    {
        // 部署态：形态由文件固定（打包字 + τ）
        if (packed_only_) return;
        const bool want = (force_pack_ == 0) ? false : true;
        if (want == use_packed_) return;
        use_packed_ = want;
        quant_valid_ = false;   // 缓冲形态变了 → 派生物必须重算
        ensure_quant_buffer_();
    }

    // 按当前模式准备量化缓冲（另一形态释放：两种容器不共存，避免推理期为
    // 用不到的形态付显存）。
    void ensure_quant_buffer_()
    {
        if (packed_only_) return;   // 部署态：缓冲来自文件，不由本函数建/清
        ComputeEngine& engine = engine_ref();
        if (use_packed_)
        {
            if (!wq_packed_.valid())
                wq_packed_ = engine.create_tensor(
                    out_features_, nn::trit_pack::words_per_row(in_features_),
                    Precision::F32, InitSpec::zero());
            // 终止式检查（本函数返回 void）
            NN_UNWRAP(wq_packed_.valid(), "BitLinear: 打包量化缓冲初始化失败");
            wq_ = Tensor{};
        }
        else
        {
            if (!wq_.valid())
                wq_ = engine.create_tensor(out_features_, in_features_, Precision::F16,
                                           InitSpec::zero());
            NN_UNWRAP(wq_.valid(), "BitLinear: 量化缓冲初始化失败");
            wq_packed_ = Tensor{};
        }
    }

    // ── 梯度缓冲：惰性分配（F2）────────────────────────────────────────────
    // 三值层的梯度**恒 f32**（STE 落点 + 优化器状态，见 §4.3.2），即 4 B/权重
    // —— 在推理足迹（latent 4 + wq 2 + grad 4 = 10 B/权重）里占 40%；2B 参数
    // 规模下是 8 GB 级死显存。梯度的唯一消费者是 `param_gradients()`（优化器
    // 构造/诊断）与 `backward()`，都在训练路径上；TensorRef 是成员引用，故
    // "先给引用、后建存储"对优化器透明。
    void ensure_grad_()
    {
        if (grad_w_.valid()) return;
        ComputeEngine& engine = engine_ref();
        grad_w_ = engine.create_tensor(out_features_, in_features_, Precision::F32,
                                       InitSpec::zero());
        grad_b_ = engine.create_tensor(out_features_, 1, Precision::F32, InitSpec::zero());
        // 终止式检查（本函数返回 void，不能用 NN_FAIL 的 return 语义）
        NN_UNWRAP(grad_w_.valid() && grad_b_.valid(),
                 "BitLinear: 梯度缓冲（惰性）初始化失败");
    }

    // 量化派生物是否还有效：写版本相等 ⇒ latent 权重未被写过。
    // 另外兜底"存储被清"（offload 的 export 会把成员置空、clear_cache 等）：
    // 那种情况下算一次比用空张量安全。
    [[nodiscard]] bool needs_quantize_() const noexcept
    {
        // 部署态：只有文件载入的那一份（永不重算 —— latent 已不存在）
        if (packed_only_) return !(wq_packed_.valid() && tau_.valid());
        if (!quant_valid_) return true;
        if (!tau_.valid()) return true;
        if (use_packed_ ? !wq_packed_.valid() : !wq_.valid()) return true;
        return quant_epoch_ != latent_w_.write_epoch();
    }

    // ── 量化：τ = 逐行 absmean；wq = RoundClip(W/τ, -1, 1) ───────────────
    // 全程在引擎内（铁律 #12：L2+ 不得把权重读回宿主）：
    //   1) τ = row_reduce_sum(|W|·(1/K))        —— **一趟**归约出 τ：1/K 折进
    //      归约操作数（RParam 承载，值不进 key；归约器会重放操作数前缀的
    //      逐元素指令）。此前是"归约 + 一次 (out,1) 标量缩放"两次 dispatch。
    //      注：GLSL emitter 不支持把**归约视图**内联进逐元素 kernel，故归约
    //      必须单独出向量（这条限制与 RParam 无关）。
    //   2) wq ← select(|W|>0.5τ 的正负两侧)      —— compute_into 原地写 f16 缓冲。
    // 合计每步每层 2 次 dispatch（原 3 次）、2 趟权重读（4+4 B/元素）+ 1 趟
    // f16 写（2 B/元素）；见 docs/todo/21 §4.11 的字节账。
    [[nodiscard]] Result<void> quantize_weights_()
    {
        ComputeEngine& engine = engine_ref();
        const Scalar inv_in = Scalar{1} / static_cast<Scalar>(in_features_);

        NN_TRY(tau, dsl::compute_reduce(engine,
            dsl::row_reduce_sum(dsl::abs(dsl::leaf(latent_w_)) * dsl::rparam(inv_in)),
            out_features_, in_features_, Precision::F32));
        tau_ = std::move(*tau);

        // 打包路径（F3）：一趟 trit_pack 直接出 2 bit/权重容器（8.25 B/元素
        // 流量 vs f16 缓冲路径的 10 B/元素，且常驻从 2 B → 0.25 B/权重）。
        if (use_packed_)
        {
            NN_TRY(packed, engine.trit_pack(latent_w_, tau_));
            wq_packed_ = std::move(*packed);
            quant_epoch_ = latent_w_.write_epoch();
            quant_valid_ = true;
            ++quantize_calls_;
            return {};
        }

        // RoundClip(W/τ, -1, 1) 的等价实现（BitNet 定义形式）：DSL 没有
        // round/floor，故用**严格**阈值 select —— 判据是 |W| > 0.5τ，恰好
        // |W| == 0.5τ **不**取正负号 → 归 0（半值向零）。与 RoundClip 的
        // 唯一差异正在这个测度零的等值点（RoundClip 按 round-half-* 取值）。
        auto lo = dsl::row_broadcast(tau_) * dsl::rparam(Scalar{-0.5});
        auto hi = dsl::row_broadcast(tau_) * dsl::rparam(Scalar{0.5});
        auto r = dsl::compute_into(engine,
            dsl::select(dsl::leaf(latent_w_) > hi,
                        Scalar{1},
                        dsl::select(dsl::leaf(latent_w_) < lo, Scalar{-1}, Scalar{0})),
            wq_);
        NN_TRY_CHECK(r);
        // 只有全部成功才登记"派生缓冲对应该写版本"
        quant_epoch_ = latent_w_.write_epoch();
        quant_valid_ = true;
        ++quantize_calls_;
        return {};
    }
};

} // namespace nn
