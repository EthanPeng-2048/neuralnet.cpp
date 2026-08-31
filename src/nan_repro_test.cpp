// ── nan_repro_test.cpp — 复现 step-474 NaN 批次（doc 边界在窗口中部） ─────
//
// 数据文件 build/nan_repro_data.txt（由 training 日志中的窗口索引从 tokcache
// 提取）：
//   第 1 行: batch seq
//   第 2..(1+batch) 行: 每行一个窗口的 token IDs
//   其后 batch 行: 每行一个窗口的 doc ids（已重标号，1 起）
//
// 用法：nan_repro_test [path] [--gpu]
//   CPU: 期望无 NaN（算法参考）
//   GPU: 若复现 → 首个非有限值位置 + 所在 sample/pos/doc 打印
//
// 该批次中 sample 6 的 doc 边界在 t=6（doc A x6 + doc B x250），
// 用户/复现运行中的 NaN 恰在 (b=6, t=6) = 新文档首 token。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::GPTModel;
using nn::PosEncodingType;
using nn::ActivationType;
using nn::NormType;

int main(int argc, char** argv)
{
    std::string path = "build/nan_repro_data.txt";
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else path = a;
    }

    std::ifstream f(path);
    if (!f) { std::cerr << "无法打开数据文件: " << path << "\n"; return 2; }
    int batch = 0, seq = 0;
    f >> batch >> seq;
    std::vector<std::vector<std::size_t>> toks(batch), docs(batch);
    for (int b = 0; b < batch; ++b)
    {
        toks[b].resize(seq);
        for (int t = 0; t < seq; ++t) f >> toks[b][t];
    }
    for (int b = 0; b < batch; ++b)
    {
        docs[b].resize(seq);
        for (int t = 0; t < seq; ++t) f >> docs[b][t];
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    // 与用户训练同构（掩码 IR 与尺寸无关，但保持网格尺寸一致: BH*seq=8192）
    const std::size_t vocab = 32784, d_model = 256, heads = 4, d_ff = 1024, layers = 4;
    GPTModel model(vocab, d_model, seq, heads, d_ff, layers,
                   PosEncodingType::Learned, ActivationType::SwiGLU, NormType::RMSNorm);
    { auto r = model.init(eng); if (!r) { std::cerr << "GPTModel init 失败: " << r.error().message << "\n"; return 1; } }

    Matrix x_m(seq, batch);
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < seq; ++t)
            x_m.set_value_unchecked(t, b, static_cast<Scalar>(toks[b][t]));

    std::vector<std::size_t> flat_docs(static_cast<std::size_t>(batch) * seq);
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < seq; ++t)
            flat_docs[b * seq + t] = docs[b][t];
    model.set_doc_ids(std::span(flat_docs));

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "from_matrix 失败\n"; return 1; }
    auto y = model.forward(eng, *x);
    if (!y) { std::cerr << "forward 失败: " << y.error().message << "\n"; return 1; }
    auto m = eng.to_matrix(*y);
    if (!m) { std::cerr << "to_matrix 失败\n"; return 1; }

    const auto sp = m->span();
    const std::size_t cols = m->cols();
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < sp.size(); ++i)
        if (!std::isfinite(sp[i])) { if (bad == 0) first = i; ++bad; }

    std::printf("logits: %zu non-finite, first at flat=%zu (r=%zu c=%zu)\n",
                bad, first, first / cols, first % cols);
    if (bad > 0)
    {
        const std::size_t c = first % cols;            // b*seq + t
        const std::size_t b = c / seq;
        const std::size_t t = c % seq;
        std::printf("→ sample %zu pos %zu, doc=%zu, token=%zu\n",
                    b, t, docs[b][t], toks[b][t]);
        // 统计所有非有限列（同一 c 的 vocab 行应全非有限）
        std::vector<std::size_t> bad_cols;
        for (std::size_t i = 0; i < sp.size(); ++i)
            if (!std::isfinite(sp[i]))
            {
                const std::size_t cc = i % cols;
                if (bad_cols.empty() || bad_cols.back() != cc) bad_cols.push_back(cc);
            }
        std::printf("non-finite columns (b*seq+t):");
        for (auto cc : bad_cols) std::printf(" %zu", cc);
        std::printf("\n");
        return 1;
    }
    std::printf("OK: no non-finite\n");
    return 0;
}
