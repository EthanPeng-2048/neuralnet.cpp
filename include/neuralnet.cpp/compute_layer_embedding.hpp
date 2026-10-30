#pragma once

// ── compute_layer_embedding.hpp — Embedding：可学习查表嵌入（issue #32）───
//
// 此前库内没有可复用的查表嵌入层：GPT 的 token embedding 是模型私有成员
// （compute_layer_gpt.hpp 的 token_emb_ + 内联 gather_rows/scatter_add_rows），
// 任何"可学习查表"的自定义复合层（点 id 查表、动作空间、多模态 ID）都要手工
// 重做 forward + backward 全套与配套样板。本层把这套语义收敛为标准 Layer。
//
// 语义（= tf.gather / torch.nn.Embedding）：
//   forward:  out[i] = table[ids[i]]，输出 (d_model, N) —— 列布局（铁律 #5：
//             每列一个嵌入向量，直接对接 Linear / 注意力的 (in_features, N) 约定）
//   backward: grad_table[ids[i]] += grad_out[:, i]（scatter_add_rows 伴随，
//             重复 id 自动累加 = 同 id 多处出现的正确梯度语义）
//             索引无梯度 → 返回零张量（与 GPTModel token_emb backward 同口径）
//
// 索引类型：`forward(const Tensor&)` 按引擎原语约定接受标量索引张量（f32 存
//   整数）；整数不落地 —— `forward_ids(std::span<const std::size_t>)` 是整数
//   宿主入口（宿主桥上传，宿主侧 fail-fast 越界）。⚠ f32 只能精确表示
//   |id| ≤ 2²⁴ 的整数：vocab > 2²⁴ 时请勿用 Tensor 入口（forward_ids 的宿主
//   上传同样经 f32 —— 该上界是引擎索引张量的存储口径，不是本层的选择）。
//
// padding_idx（可选，<0 = 无）：与 torch 同语义 —— 该行**初值恒零**且**每步
//   梯度清零**（优化器拿不到非零梯度 → 行永不移动；AdamW 的乘性 decay 也
//   保持零）。forward 不做特殊处理（表行本来就是零）。
//
// 分层（铁律 #3/#12）：算法（查表 + 伴随散射 + padding 语义）只在本层；
//   gather_rows / scatter_add_rows / insert_rows 是 op-level 原语；索引表经
//   span 宿主桥上传，不经 Matrix。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "compute_engine.hpp"
#include "compute_layer_base.hpp"
#include "compute_tensor.hpp"

namespace nn
{

class Embedding final : public Layer
{
private:
    std::size_t vocab_size_;
    std::size_t d_model_;
    std::ptrdiff_t padding_idx_;   // <0 = 无 padding 行
    Tensor table_;                 // (vocab, d_model) 嵌入表
    Tensor grad_table_;            // (vocab, d_model) 梯度（惰性分配，F2）
    Tensor ids_cache_;             // 本步索引（散射用；离散索引非浮点激活，
                                   //  checkpoint 模式同样保留 —— 不缓存就无法
                                   //  scatter，与 Linear 的 input_cache_ 不同）
    Tensor padding_zero_row_;      // (1, d_model) 零行（init 与每步清 padding 梯度复用）

    // 梯度缓冲惰性分配（F2：推理路径不为梯度付显存）
    void ensure_grad_()
    {
        if (grad_table_.valid()) return;
        ComputeEngine& engine = engine_ref();
        grad_table_ = engine.create_tensor(vocab_size_, d_model_, p_.param,
                                           InitSpec::zero());
        NN_UNWRAP(grad_table_.valid(),
                 "Embedding: 梯度缓冲（惰性）初始化失败");
    }

    // 清零 padding 行（覆盖语义：insert_rows 直接写目标行）
    [[nodiscard]] Result<void> clear_padding_row_(Tensor& t)
    {
        if (padding_idx_ < 0) return {};
        return engine_ref().insert_rows(
            t, static_cast<std::size_t>(padding_idx_), padding_zero_row_);
    }

public:
    // padding_idx：<0 = 无；否则 ∈ [0, vocab)
    Embedding(std::size_t vocab_size, std::size_t d_model,
              std::ptrdiff_t padding_idx = -1)
        : vocab_size_(vocab_size), d_model_(d_model), padding_idx_(padding_idx)
    {
        // 前置条件校验（铁律 #1）：构造后仍会使用这些成员 → 无条件终止口径
        NN_ASSERT_ALWAYS(vocab_size > 0, "Embedding: vocab_size 必须 > 0");
        NN_ASSERT_ALWAYS(d_model > 0, "Embedding: d_model 必须 > 0");
        NN_ASSERT_ALWAYS(padding_idx < 0 ||
                             static_cast<std::size_t>(padding_idx) < vocab_size,
                         "Embedding: padding_idx 越界（须 < vocab_size 或负数表示无）");
    }

    [[nodiscard]] const char* layer_name() const noexcept override
    { return "Embedding"; }

    [[nodiscard]] std::size_t vocab_size() const noexcept { return vocab_size_; }
    [[nodiscard]] std::size_t d_model() const noexcept { return d_model_; }
    [[nodiscard]] std::ptrdiff_t padding_idx() const noexcept { return padding_idx_; }

    [[nodiscard]] Result<void> init_impl(ComputeEngine& engine) override
    {
        // N(0, 1)（torch.nn.Embedding 同分布）；seed 显式（U1）
        table_ = engine.create_tensor(vocab_size_, d_model_, p_.param,
                                      InitSpec::normal(Scalar{0}, Scalar{1}, kInitSeed));
        if (!table_.valid()) NN_FAIL("Embedding: 嵌入表初始化失败");
        padding_zero_row_ =
            engine.create_tensor(1, d_model_, p_.param, InitSpec::zero());
        if (!padding_zero_row_.valid())
            NN_FAIL("Embedding: padding 零行缓冲初始化失败");
        // padding 行强制清零（分布初始化会填值）
        return clear_padding_row_(table_);
    }

    std::vector<TensorRef> parameters() override { return {table_}; }

    std::vector<TensorRef> param_gradients() override
    {
        ensure_grad_();
        return {grad_table_};
    }

    [[nodiscard]] bool grad_buffers_allocated() const noexcept override
    { return grad_table_.valid(); }

    void clear_cache() override { ids_cache_ = Tensor{}; }

    std::vector<TensorRef> activation_cache() override
    {
        std::vector<TensorRef> r;
        if (ids_cache_.valid()) r.emplace_back(ids_cache_);
        return r;
    }

    // ── forward：查表 ──────────────────────────────────────────────────
    // input = 索引张量（任意形状；按 flat 遍历全部元素 = gather_rows 原语口径）。
    // 输出 (d_model, N)，N = input 元素数（列布局，对接下游层约定）。
    // 越界索引沿引擎防御语义返回零行（不抛错）——整数入口 forward_ids 在宿主
    // 侧 fail-fast，需要严格校验的调用方用它。
    [[nodiscard]] Result<Tensor> forward(const Tensor& input) override
    {
        ComputeEngine& engine = engine_ref();
        NN_TRY(rows, engine.gather_rows(table_, input));   // (N, d_model)
        ids_cache_ = input;   // Tensor 引用语义（存储共享）；GPU 全程设备端
        return engine.transpose(*rows);                    // (d_model, N)
    }

    // ── forward_ids：整数宿主入口（issue #32 的"索引接受整数类型"）──────
    // ids 经 span 宿主桥上传（f32 存储，|id| ≤ 2²⁴ 精确）；越界 fail-fast。
    // 输出 (d_model, N)。
    [[nodiscard]] Result<Tensor> forward_ids(std::span<const std::size_t> ids)
    {
        for (std::size_t id : ids)
        {
            if (id >= vocab_size_)
                NN_FAIL("Embedding::forward_ids: 索引 " + std::to_string(id) +
                        " 越界（vocab_size=" + std::to_string(vocab_size_) + "）");
        }
        std::vector<Scalar> f(ids.begin(), ids.end());
        ComputeEngine& engine = engine_ref();
        NN_TRY(idx, upload_span(engine, 1, ids.size(), Precision::F32,
                                std::span(f)));
        return forward(*idx);
    }

    // ── backward：scatter_add 伴随 + 零输入梯度 ────────────────────────
    // grad_output (d_model, N) → 转置 (N, d_model) → 按索引原子累加进表梯度。
    // 索引不可导 → 返回与输入 ids 同形状的零张量（接口一致性，同 GPTModel）。
    [[nodiscard]] Result<Tensor> backward(const Tensor& grad_output) override
    {
        ComputeEngine& engine = engine_ref();
        if (grad_output.rows() != d_model_)
            NN_FAIL("Embedding backward: grad_output 形状不匹配（rows != d_model）");
        if (!ids_cache_.valid())
            NN_FAIL("Embedding backward: 无缓存索引（backward 前必须先 forward）");
        ensure_grad_();
        // (d_model, N) → (N, d_model)：scatter_add 的 grad 形状须与表同行宽
        NN_TRY(gT, engine.transpose(grad_output));
        // dst[ids[i]] += grad_T[i]（重复 id 累加；越界忽略 —— 与 forward 的
        // 防御语义对称：越界 id 取了零行、也不该有梯度）
        NN_TRY(sr, engine.scatter_add_rows(grad_table_, ids_cache_, *gT));
        // padding 行梯度清零（每步；torch 同语义）
        NN_TRY(cpr, clear_padding_row_(grad_table_));
        // 索引梯度 = 零（形状镜像输入）
        Tensor zero = engine.create_tensor(ids_cache_.rows(), ids_cache_.cols(),
                                           Precision::F32, InitSpec::zero());
        if (!zero.valid()) NN_FAIL("Embedding backward: 零梯度张量分配失败");
        return zero;
    }
};

} // namespace nn
