#pragma once

// ── domain_gpt.hpp — GPT 领域构建层（引擎化架构） ──────────────────────────
//
// 依赖：Model + ComputeEngine + Tokenizer
//   （model_container.hpp / compute_engine.hpp / domain_tokenizer.hpp）
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>

#include "core_errors.hpp"
#include "compute_engine.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"
#include "precision.hpp"

namespace nn {

// ── GPT 默认超参数 ────────────────────────────────────────────────────────
inline constexpr std::size_t GPT_VOCAB_SIZE    = 10000;
inline constexpr std::size_t GPT_D_MODEL       = 128;
inline constexpr std::size_t GPT_NUM_HEADS     = 4;
inline constexpr std::size_t GPT_D_FF          = 512;
inline constexpr std::size_t GPT_NUM_LAYERS    = 4;
inline constexpr std::size_t GPT_SEQ_LEN       = 256;

// ── GPT 配置结构体 ────────────────────────────────────────────────────────
// 将 build_gpt_model 的多个位置参数收拢为一个结构体，避免调用方签名过长。
struct GptConfig {
    std::size_t vocab_size  = GPT_VOCAB_SIZE;
    std::size_t d_model     = GPT_D_MODEL;
    std::size_t seq_len     = GPT_SEQ_LEN;
    std::size_t num_heads   = GPT_NUM_HEADS;
    std::size_t d_ff        = GPT_D_FF;
    std::size_t num_layers  = GPT_NUM_LAYERS;
    PosEncodingType pos_enc = PosEncodingType::Learned;
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type = NormType::LayerNorm;
    PrecisionProfile precision;  // 模型级精度配置（§9.1，默认全 F32）
    bool subln = false;          // SubLN（BitNet 2B4T 子层归一化，docs 22 §3.1）
    std::size_t n_head_kv = 0;   // GQA 的 KV 头数（0 = num_heads = MHA，docs 22 §3.2）
    bool tie_embeddings = false; // head 复用 token embedding（docs 22 §3.3）
};

// ── 构建 GPT 模型（GptConfig 版本，推荐使用） ────────────────────────────
// GPTModel 是一个单一 Layer（内含 TokenEmb+PosEmb+N×GPTBlock+LN+LM Head），
// 作为 Model 的唯一层。engine 决定权重张量驻留设备。
// cfg.pos_enc 控制位置编码：Learned(默认)/Sinusoidal/ALiBi
[[nodiscard]] inline Result<Model> build_gpt_model(
    ComputeEngine& engine, const GptConfig& cfg)
{
    if (cfg.d_model == 0 || cfg.num_heads == 0 || cfg.seq_len == 0 || cfg.vocab_size == 0)
        NN_FAIL("GPT model parameters must be positive");
    if (cfg.d_ff == 0 || cfg.num_layers == 0)
        NN_FAIL("GPT d_ff and num_layers must be positive");
    if (cfg.d_model % cfg.num_heads != 0)
        NN_FAIL("GPT d_model must be divisible by num_heads");
    if (cfg.n_head_kv != 0 && (cfg.n_head_kv > cfg.num_heads ||
                               cfg.num_heads % cfg.n_head_kv != 0))
        NN_FAIL("GPT n_head_kv must be a divisor of num_heads (0 = MHA)");
    Model model(engine);
    {
        auto r = model.add<GPTModel>(cfg.vocab_size, cfg.d_model, cfg.seq_len,
                                     cfg.num_heads, cfg.d_ff, cfg.num_layers, cfg.pos_enc,
                                     cfg.activation, cfg.norm_type, cfg.precision,
                                     cfg.subln, cfg.n_head_kv, cfg.tie_embeddings);
        NN_TRY_CHECK(r);
    }
    return model;
}

// ── 构建 GPT 模型（位置参数版本，转发至 GptConfig 版本） ───────
[[nodiscard]] inline Result<Model> build_gpt_model(
    ComputeEngine& engine,
    std::size_t vocab_size  = GPT_VOCAB_SIZE,
    std::size_t d_model     = GPT_D_MODEL,
    std::size_t seq_len     = GPT_SEQ_LEN,
    std::size_t num_heads   = GPT_NUM_HEADS,
    std::size_t d_ff        = GPT_D_FF,
    std::size_t num_layers  = GPT_NUM_LAYERS,
    PosEncodingType pos_enc_type = PosEncodingType::Learned,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm,
    PrecisionProfile precision = PrecisionProfile{},
    bool subln = false,
    std::size_t n_head_kv = 0,
    bool tie_embeddings = false)
{
    return build_gpt_model(engine, GptConfig{
        vocab_size, d_model, seq_len, num_heads, d_ff, num_layers,
        pos_enc_type, activation, norm_type, precision, subln, n_head_kv,
        tie_embeddings});
}

// ── 从 ModelSpec 构建 GPT 模型 ──────────────────────────────────────────
// 用于从二进制文件加载时自动还原架构。
// 统一的 GPTModel 通过 spec.pos_encoding 区分 Learned/Sinusoidal/ALiBi 模式，
// 因此无需为 ALiBi 单独提供构建函数。
[[nodiscard]] inline Result<Model> build_gpt_model_from_spec(
    ComputeEngine& engine, const ModelSpec &spec,
    PrecisionProfile precision = PrecisionProfile{})
{
    // 接受 GPT 类型，或 ALiBi_GPT 类型（两种 spec 类型走同一构建路径）
    if (!spec.is_gpt() && !spec.is_alibi_gpt())
        NN_FAIL("Invalid ModelSpec type for GPT: expected GPT or ALiBi_GPT");
    // ── 三值（T1_58）权重由**规格**决定（权威来源，docs 21 §4.7/§4.8.3）──
    // 与 build_mnist_mlp_model 的加载路径同款：保存过的三值模型必须还原成
    // BitLinear（参数形状相同、语义不同），命令行传进来的 precision 会被规格覆盖。
    if (spec.weight_quant == WeightQuant::T1_58)
        precision.param = Precision::T1_58;
    auto model = build_gpt_model(
        engine,
        spec.vocab_size, spec.d_model, spec.seq_len,
        spec.num_heads, spec.d_ff, spec.num_layers,
        spec.pos_encoding, spec.activation, spec.norm_type, precision,
        spec.subln, spec.n_head_kv, spec.tie_embeddings);
    if (model)
        model->set_spec(spec);  // 记录架构规格，供 load_model 校验
    return model;
}

// ── 构造 GPT ModelSpec ──────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_gpt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers,
    PosEncodingType pos_encoding = PosEncodingType::Learned,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm,
    bool subln = false,
    std::size_t n_head_kv = 0,
    bool tie_embeddings = false)
{
    ModelSpec spec;
    spec.type         = ModelType::GPT;
    spec.vocab_size   = vocab_size;
    spec.d_model      = d_model;
    spec.seq_len      = seq_len;
    spec.num_heads    = num_heads;
    spec.d_ff         = d_ff;
    spec.num_layers   = num_layers;
    spec.pos_encoding = pos_encoding;
    spec.activation   = activation;
    spec.norm_type    = norm_type;
    spec.subln        = subln;
    spec.n_head_kv    = n_head_kv;
    spec.tie_embeddings = tie_embeddings;
    return spec;
}

} // namespace nn
