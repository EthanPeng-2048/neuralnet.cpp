#pragma once

// ── domain_zipt.hpp — ZiPT 领域构建层（AttnZip 记忆压缩解码器） ─────────────
//
// 依赖：Model + ComputeEngine
//   （model_container.hpp / compute_engine.hpp / model_spec.hpp）
//
// ZiPT（zip + GPT）：先把长上下文 X 经 CrossAttention 压缩为 M 个记忆 token
// C，随后每个 ZiPTBlock 对 [C ; 局部] 做联合注意力。复杂度对上下文长度线性。
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

// ── ZiPT 默认记忆 token 数 ───────────────────────────────────────────────
inline constexpr std::size_t ZIPT_MEMORY_TOKENS = 32;  // M（≪ seq_len，建议偶数规避 GPU matmul 奇数列后端 bug）

// ── ZiPT 配置结构体 ─────────────────────────────────────────────────────
struct ZiPTConfig {
    std::size_t vocab_size    = GPT_VOCAB_SIZE;
    std::size_t d_model       = GPT_D_MODEL;
    std::size_t seq_len       = GPT_SEQ_LEN;        // L：总上下文长度（一次 forward 处理的最大长度）
    std::size_t window        = 0;                  // W：局部窗口（直接注意力；0=默认=seq_len，旧行为 W=L 无压缩）
    std::size_t num_heads     = GPT_NUM_HEADS;
    std::size_t d_ff          = GPT_D_FF;
    std::size_t num_layers    = GPT_NUM_LAYERS;
    std::size_t memory_tokens = ZIPT_MEMORY_TOKENS; // M：记忆 token 数（≪ seq_len）
    PosEncodingType pos_enc   = PosEncodingType::Learned;
    ActivationType activation = ActivationType::GeLU;
    NormType norm_type        = NormType::LayerNorm;
};

// ── 构建 ZiPT 模型（推荐使用） ─────────────────────────────────────────
[[nodiscard]] inline Result<Model> build_zipt_model(
    ComputeEngine& engine, const ZiPTConfig& cfg)
{
    if (cfg.d_model == 0 || cfg.num_heads == 0 || cfg.seq_len == 0 ||
        cfg.vocab_size == 0 || cfg.memory_tokens == 0)
        return std::unexpected(Error{"ZiPT model parameters must be positive"});
    if (cfg.d_ff == 0 || cfg.num_layers == 0)
        return std::unexpected(Error{"ZiPT d_ff and num_layers must be positive"});
    if (cfg.d_model % cfg.num_heads != 0)
        return std::unexpected(Error{"ZiPT d_model must be divisible by num_heads"});
    if (cfg.memory_tokens > cfg.seq_len)
        return std::unexpected(Error{"ZiPT memory_tokens must be <= seq_len"});
    ZiPTConfig c = cfg;
    if (c.window == 0) c.window = c.seq_len;   // 默认 W=L（旧行为，向后兼容）
    if (c.window > c.seq_len)
        return std::unexpected(Error{"ZiPT window must be <= seq_len"});

    Model model(engine);
    {
        auto r = model.add<ZiPTModel>(
            c.vocab_size, c.d_model, c.seq_len, c.window,
            c.num_heads, c.d_ff, c.num_layers, c.memory_tokens,
            c.pos_enc, c.activation, c.norm_type);
        if (!r) return std::unexpected(r.error());
    }
    return model;
}

// ── 从 ModelSpec 构建 ZiPT 模型 ─────────────────────────────────────────
[[nodiscard]] inline Result<Model> build_zipt_model_from_spec(
    ComputeEngine& engine, const ModelSpec& spec)
{
    if (!spec.is_zipt())
        return std::unexpected(Error{"Invalid ModelSpec type for ZiPT: expected ZiPT"});

    auto model = build_zipt_model(engine, ZiPTConfig{
        spec.vocab_size, spec.d_model, spec.seq_len, spec.window,
        spec.num_heads, spec.d_ff, spec.num_layers, spec.memory_tokens,
        spec.pos_encoding, spec.activation, spec.norm_type});
    if (model)
        model->set_spec(spec);
    return model;
}

// ── 构造 ZiPT ModelSpec ─────────────────────────────────────────────────
[[nodiscard]] inline ModelSpec make_zipt_spec(
    std::size_t vocab_size,
    std::size_t d_model,
    std::size_t seq_len,
    std::size_t num_heads,
    std::size_t d_ff,
    std::size_t num_layers,
    std::size_t memory_tokens,
    std::size_t window = 0,
    PosEncodingType pos_encoding = PosEncodingType::Learned,
    ActivationType activation = ActivationType::GeLU,
    NormType norm_type = NormType::LayerNorm)
{
    ModelSpec spec;
    spec.type           = ModelType::ZiPT;
    spec.vocab_size     = vocab_size;
    spec.d_model        = d_model;
    spec.seq_len        = seq_len;
    spec.num_heads      = num_heads;
    spec.d_ff           = d_ff;
    spec.num_layers     = num_layers;
    spec.memory_tokens  = memory_tokens;
    spec.window         = window;
    spec.pos_encoding   = pos_encoding;
    spec.activation     = activation;
    spec.norm_type      = norm_type;
    return spec;
}

} // namespace nn

