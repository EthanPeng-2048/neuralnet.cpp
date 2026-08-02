// ── Attention 一致性验证：forward vs forward_step ──────────────────────────
//
// 目的：验证 KV cache 增量推理（forward_step）与整段推理（forward）在相同
//       输入下产生相同的输出。若一致则证明 Attention 语义修复正确，
//       生成乱码是旧模型权重的问题（用 buggy 代码训练得到），需重训。
//
// 原理：
//   - forward: 输入整段 token 序列，带因果掩码，输出 (vocab, seq) logits
//   - forward_step: 逐 token 增量推理，KV cache 复用前文，输出 (vocab, 1) logits
//   - forward 在位置 i 的 logits（带因果掩码，只能看到 0..i）
//     应与 forward_step 在 cur_len=i 时的 logits 完全一致
//
// 用法：attn_consistency_test [--gpu] [--seq N] [--layers N]
//   --gpu       启用 GPU（Vulkan）
//   --seq N     测试序列长度（默认 16）
//   --layers N  Transformer 层数（默认 2）
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/engine_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::GPTModel;
using nn::PosEncodingType;

namespace {

// ── 辅助：从 (vocab, seq) logits 矩阵取第 col 列 → 向量 ──
std::vector<Scalar> extract_logits_column(
    const Matrix& logits, std::size_t col)
{
    const std::size_t vocab = logits.rows();
    std::vector<Scalar> out(vocab);
    for (std::size_t v = 0; v < vocab; ++v)
        out[v] = logits.at_unchecked(v, col);
    return out;
}

// ── 辅助：从 (vocab, 1) logits 矩阵取第 0 列 → 向量 ──
std::vector<Scalar> extract_logits_single(const Matrix& logits)
{
    return extract_logits_column(logits, 0);
}

// ── 辅助：两个向量的最大绝对差 ──
Scalar max_abs_diff(const std::vector<Scalar>& a, const std::vector<Scalar>& b)
{
    Scalar m = Scalar{0};
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        Scalar d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ── 辅助：打印向量前 N 个元素 ──
void print_vec_head(const std::vector<Scalar>& v, std::size_t n = 8)
{
    std::cout << "[";
    const std::size_t m = std::min(n, v.size());
    for (std::size_t i = 0; i < m; ++i)
    {
        std::cout << std::scientific << std::setprecision(4) << v[i];
        if (i + 1 < m) std::cout << ", ";
    }
    if (v.size() > m) std::cout << ", ...";
    std::cout << "]";
}

// ── 单引擎一致性测试 ──
// 返回 true 表示通过（forward 与 forward_step 输出一致）
bool test_consistency(
    ComputeEngine& engine,
    const std::string& engine_name,
    std::size_t seq_len,
    std::size_t num_layers,
    PosEncodingType pos_enc,
    const std::string& pos_enc_name,
    bool use_batch = true)
{
    // 模型超参（小规模，加速测试）
    const std::size_t vocab_size = 100;
    const std::size_t d_model    = 32;
    const std::size_t num_heads  = 4;
    const std::size_t d_ff       = 64;

    std::cout << "\n────────────────────────────────────────────────────────\n"
              << "引擎: " << engine_name
              << " | 位置编码: " << pos_enc_name
              << " | seq=" << seq_len
              << " | layers=" << num_layers
              << " | batch=" << (use_batch ? "on" : "off") << "\n"
              << "────────────────────────────────────────────────────────\n";

    // ── 1. 构建模型 ──
    std::cout << "[1/4] 构建 GPTModel..." << std::flush;
    GPTModel model(
        engine, vocab_size, d_model, seq_len,
        num_heads, d_ff, num_layers, pos_enc);
    std::cout << " 完成\n";

    // ── 2. 生成随机 token 序列 ──
    std::cout << "[2/4] 生成随机 token 序列..." << std::flush;
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<std::size_t> tok_dist(0, vocab_size - 1);
    std::vector<std::size_t> tokens(seq_len);
    for (auto& t : tokens) t = tok_dist(rng);
    std::cout << " 完成 (tokens:";
    for (std::size_t i = 0; i < std::min<std::size_t>(8, seq_len); ++i)
        std::cout << " " << tokens[i];
    if (seq_len > 8) std::cout << " ...";
    std::cout << ")\n";

    // ── 3. 整段 forward ──
    std::cout << "[3/4] 运行 forward（整段推理）..." << std::flush;
    Matrix tok_mat(seq_len, 1);
    for (std::size_t t = 0; t < seq_len; ++t)
        tok_mat.set_value_unchecked(t, 0, static_cast<Scalar>(tokens[t]));
    auto tok_t = engine.from_matrix(tok_mat);
    if (!tok_t)
    {
        std::cout << "\n  ❌ from_matrix 失败: " << tok_t.error().message << "\n";
        return false;
    }

    if (use_batch)
    {
        auto begin_batch_r = engine.begin_batch();
        if (!begin_batch_r)
        {
            std::cout << "\n  ❌ begin_batch 失败: " << begin_batch_r.error().message << "\n";
            return false;
        }
    }
    auto fwd_r = model.forward(engine, *tok_t);
    if (!fwd_r)
    {
        std::cout << "\n  ❌ forward 失败: " << fwd_r.error().message << "\n";
        return false;
    }
    if (use_batch)
    {
        auto end_batch_r = engine.end_batch();
        if (!end_batch_r)
        {
            std::cout << "\n  ❌ end_batch 失败: " << end_batch_r.error().message << "\n";
            return false;
        }
    }

    auto fwd_logits_m = engine.to_matrix(*fwd_r);
    if (!fwd_logits_m)
    {
        std::cout << "\n  ❌ to_matrix 失败: " << fwd_logits_m.error().message << "\n";
        return false;
    }
    std::cout << " 完成 (logits: " << fwd_logits_m->rows()
              << " × " << fwd_logits_m->cols() << ")\n";

    // ── 4. 逐 token forward_step ──
    std::cout << "[4/4] 运行 forward_step（增量推理）并对比..." << std::flush;

    // 预分配 KV cache
    std::vector<Tensor> k_caches, v_caches;
    k_caches.reserve(num_layers);
    v_caches.reserve(num_layers);
    for (std::size_t i = 0; i < num_layers; ++i)
    {
        k_caches.push_back(engine.create_tensor(seq_len, d_model));
        v_caches.push_back(engine.create_tensor(seq_len, d_model));
        auto r1 = engine.zero(k_caches.back());
        if (!r1) { std::cout << "\n  ❌ zero(k_cache) 失败\n"; return false; }
        auto r2 = engine.zero(v_caches.back());
        if (!r2) { std::cout << "\n  ❌ zero(v_cache) 失败\n"; return false; }
    }

    Scalar total_max_diff = Scalar{0};
    std::size_t worst_pos = 0;
    bool all_ok = true;

    for (std::size_t i = 0; i < seq_len; ++i)
    {
        // 增量推理：cur_len = i（cache 中已有 i 个 token，准备写入第 i 个）
        if (use_batch)
        {
            auto br = engine.begin_batch();
            if (!br) { std::cout << "\n  ❌ begin_batch 失败\n"; return false; }
        }

        auto step_r = model.forward_step(
            engine, tokens[i], i, k_caches, v_caches, i);
        if (!step_r)
        {
            std::cout << "\n  ❌ forward_step 在 pos=" << i
                      << " 失败: " << step_r.error().message << "\n";
            return false;
        }

        if (use_batch)
        {
            auto er = engine.end_batch();
            if (!er) { std::cout << "\n  ❌ end_batch 失败\n"; return false; }
        }

        auto step_logits_m = engine.to_matrix(*step_r);
        if (!step_logits_m)
        {
            std::cout << "\n  ❌ to_matrix 失败\n";
            return false;
        }

        // 对比 forward 的第 i 列
        auto fwd_col = extract_logits_column(*fwd_logits_m, i);
        auto step_col = extract_logits_single(*step_logits_m);
        Scalar diff = max_abs_diff(fwd_col, step_col);

        if (diff > total_max_diff)
        {
            total_max_diff = diff;
            worst_pos = i;
        }

        // 容差：ffast-math 下数值误差会放大，但同一引擎内 forward 与
        // forward_step 的数学等价性应保持在 1e-2 量级以内
        const Scalar tolerance = 5e-2f;
        if (diff > tolerance)
        {
            all_ok = false;
            std::cout << "\n  ❌ pos=" << i << " 不一致 (max_diff="
                      << std::scientific << std::setprecision(4) << diff << ")\n";
            std::cout << "    forward:    ";
            print_vec_head(fwd_col);
            std::cout << "\n    forward_step: ";
            print_vec_head(step_col);
            std::cout << "\n";
        }
    }

    std::cout << "\n";
    std::cout << "  最大绝对误差: " << std::scientific << std::setprecision(6)
              << total_max_diff << " (位置 " << worst_pos << ")\n";
    std::cout << "  结果: " << (all_ok ? "✅ 通过" : "❌ 失败") << "\n";

    return all_ok;
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_gpu = false;
    bool use_batch = true;
    std::size_t seq_len = 16;
    std::size_t num_layers = 2;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--gpu") use_gpu = true;
        else if (arg == "--no-batch") use_batch = false;
        else if (arg == "--seq" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --seq\n"; return 1; }
            seq_len = *v;
        }
        else if (arg == "--layers" && i + 1 < argc)
        {
            auto v = nn::parse_number<std::size_t>(argv[++i]);
            if (!v) { std::cerr << "无效 --layers\n"; return 1; }
            num_layers = *v;
        }
        else if (arg == "--help")
        {
            std::cout << "用法: " << argv[0] << " [--gpu] [--no-batch] [--seq N] [--layers N]\n";
            return 0;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 1;
        }
    }

    std::cout << "========================================\n"
              << "  Attention 一致性验证\n"
              << "  forward (整段) vs forward_step (KV cache)\n"
              << "========================================\n"
              << "  序列长度: " << seq_len << "\n"
              << "  层数:     " << num_layers << "\n"
              << "  引擎:     " << (use_gpu ? "GPU (Vulkan)" : "CPU") << "\n"
              << "  batch:    " << (use_batch ? "on" : "off") << "\n"
              << "========================================\n";

    // ── 创建引擎 ──
    nn::cli::EngineConfig cfg;
    cfg.use_gpu = use_gpu;
    auto engine = nn::cli::create_engine(cfg, std::cout);
    if (!engine)
    {
        std::cerr << "引擎创建失败\n";
        return 1;
    }

    bool all_pass = true;

    // 测试 1: Learned 位置编码（GPT 默认）
    all_pass &= test_consistency(
        *engine, use_gpu ? "GPU" : "CPU",
        seq_len, num_layers,
        PosEncodingType::Learned, "Learned", use_batch);

    // 测试 2: Sinusoidal 位置编码
    all_pass &= test_consistency(
        *engine, use_gpu ? "GPU" : "CPU",
        seq_len, num_layers,
        PosEncodingType::Sinusoidal, "Sinusoidal", use_batch);

    // 测试 3: ALiBi（无位置嵌入，因果 + 线性偏置）
    all_pass &= test_consistency(
        *engine, use_gpu ? "GPU" : "CPU",
        seq_len, num_layers,
        PosEncodingType::ALiBi, "ALiBi", use_batch);

    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  总体结果: " << (all_pass ? "✅ 全部通过" : "❌ 存在失败");
    std::cout << "\n════════════════════════════════════════\n";

    return all_pass ? 0 : 1;
}
