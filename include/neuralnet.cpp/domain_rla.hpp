#ifndef NN_DOMAIN_RLA_HPP
#define NN_DOMAIN_RLA_HPP

// ── domain_rla.hpp — RAPT 领域构建层（ReLU 激活线性注意力语言模型） ──────────
//
// 依赖：Model + ComputeEngine + ModelSpec
//   （model_container.hpp / compute_engine.hpp / model_spec.hpp）
//
// RAPT（ReLU-activated Attention Pretrained Transformer）：用 ReLU 线性注意力
// （RLA）替代 softmax 自注意力。RLA 用 ReLU 门控做动态稀疏检索，分子/分母都靠
// 运行态前缀和实现 O(L·d²) 复杂度，无 O(L²) 物化。RLA 强约束：位置编码必须
// RoPE（或 ALiBi）且 RoPE 施加在 Q/K 进 ReLU 之前——本实现强制 RoPE。
// 复用 domain_gpt.hpp 的默认超参常量（GPT_VOCAB_SIZE 等）。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <string>

#include "core_errors.hpp"
#include "compute_engine.hpp"
#include "model_container.hpp"
#include "model_spec.hpp"
#include "domain_gpt.hpp"  // GPT_VOCAB_SIZE 等共享默认超参

namespace nn {

// ── RAPT 配置结构体 ─────────────────────────────────────────────────────
struct RAPTConfig {
    std::size_t vocab_size = GPT_VOCAB_SIZE;
    std::size_t d_model    = GPT_D_MODEL;
    std::size_t seq_len    = GPT_SEQ_LEN;
    std::size_t num_heads  = GPT_NUM_HEADS;
    std::size_t d_ff       = GPT_D_FF;
    std::size_t num_layers = GPT_NUM_LAYERS;
    PosEncodingType pos_enc   = PosEncodingType::RoPE;   // RLA 约束：RoPE（或 ALiBi）
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type        = NormType::LayerNorm;
    bool causal = true;                                  // RAPT 是 causal LM
};

// ── 构建 RAPT 模型（推荐使用） ─────────────────────────────────────────
[[nodiscard]] inline Result<Model> build_rapt_model(
    ComputeEngine& engine, const RAPTConfig& cfg)
{
    if (cfg.d_model == 0 || cfg.num_heads == 0 || cfg.seq_len == 0 ||
        cfg.vocab_size == 0 || cfg.d_ff == 0 || cfg.num_layers == 0)
        return std::unexpected(Error{"RAPT model parameters must be positive"});
    if (cfg.d_model % cfg.num_heads != 0)
        return std::unexpected(Error{"RAPT d_model must be divisible by num_heads"});
    // RLA 强约束：RoPE 需偶数 d_k（旋转块）；v1 只支持 RoPE
    if (cfg.pos_enc != PosEncodingType::RoPE)
        return std::unexpected(Error{"RAPT v1 requires pos_enc == RoPE (RLA constraint)"});
    if ((cfg.d_model / cfg.num_heads) % 2 != 0)
        return std::unexpected(Error{"RAPT requires even d_k (RoPE) — d_model/num_heads must be even"});

    Model model(engine);
    {
        auto r = model.add<RAPTModel>(
            cfg.vocab_size, cfg.d_model, cfg.seq_len,
            cfg.num_heads, cfg.d_ff, cfg.num_layers,
            cfg.pos_enc, cfg.activation, cfg.norm_type, cfg.causal);
        if (!r) return std::unexpected(r.error());
    }
    return model;
}

// ── 从 ModelSpec 构建 RAPT 模型 ─────────────────────────────────────────
[[nodiscard]] inline Result<Model> build_rapt_model_from_spec(
    ComputeEngine& engine, const ModelSpec& spec)
{
    if (!spec.is_rapt())
        return std::unexpected(Error{"Invalid ModelSpec type for RAPT: expected RAPT"});

    auto model = build_rapt_model(engine, RAPTConfig{
        spec.vocab_size, spec.d_model, spec.seq_len,
        spec.num_heads, spec.d_ff, spec.num_layers,
        spec.pos_encoding, spec.activation, spec.norm_type, /*causal=*/true});
    if (model)
        model->set_spec(spec);
    return model;
}

// ── 构造 RAPT ModelSpec ─────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_rapt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers,
    PosEncodingType pos_encoding = PosEncodingType::RoPE,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm)
{
    ModelSpec spec;
    spec.type         = ModelType::RAPT;
    spec.vocab_size   = vocab_size;
    spec.d_model      = d_model;
    spec.seq_len      = seq_len;
    spec.num_heads    = num_heads;
    spec.d_ff         = d_ff;
    spec.num_layers   = num_layers;
    spec.pos_encoding = pos_encoding;
    spec.activation   = activation;
    spec.norm_type    = norm_type;
    return spec;
}

} // namespace nn

#endif // NN_DOMAIN_RLA_HPP
