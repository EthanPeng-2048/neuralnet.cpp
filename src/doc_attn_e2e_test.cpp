// ── 文档感知掩码端到端正确性测试（doc_attn_test） ─────────────────────
//
// 原理（不变性验证）：
//   若块对角掩码正确生效，一个窗口内 doc B 的位置只能注意同文档内的位置，
//   因此 doc B 的输出应完全不受 doc A 内容影响。
//
// 测试步骤：
//   1) 构造单窗口 (seq=8, batch=1)，doc A 占 0-3，doc B 占 4-7，doc_ids=[1,1,1,1,2,2,2,2]
//   2) 变体1: doc A 用 token A1；变体2: doc A 用完全不同的 token A2（doc B 相同）
//   3) 设 doc_ids 跑两次 forward → doc B 位置 (4..7) 的 logits 必须【相同】
//   4) 负对照：清空 doc_ids（纯因果）跑同样两组 → doc B 位置 logits 必须【不同】
//      （证明测试有灵敏度，能抓住"掩码没生效"的 bug）
//
// 用法：doc_attn_test [--cuda|--gpu]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <cstdlib>
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

namespace {

// 运行一次 forward，返回 logits 的 Matrix 视图（vocab × seq*batch）
Matrix run_forward(ComputeEngine& eng, GPTModel& model,
                   const std::vector<std::size_t>& tokens,
                   const std::vector<std::size_t>& doc_ids)
{
    const std::size_t seq = tokens.size();
    Matrix x_m(seq, 1);
    for (std::size_t t = 0; t < seq; ++t)
        x_m.set_value_unchecked(t, 0, static_cast<Scalar>(tokens[t]));

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "  from_matrix 失败\n"; std::abort(); }

    model.set_doc_ids(doc_ids);              // 空 vector = 纯因果
    auto y = model.forward(eng, *x);
    if (!y) { std::cerr << "  forward 失败: " << y.error().message << "\n"; std::abort(); }

    auto m = eng.to_matrix(*y);
    if (!m) { std::cerr << "  to_matrix 失败\n"; std::abort(); }
    return *m;
}

// 比较两矩阵的指定列区间，返回最大绝对差
Scalar max_col_diff(const Matrix& a, const Matrix& b,
                    std::size_t start_col, std::size_t end_col)
{
    Scalar worst{0};
    for (std::size_t r = 0; r < a.rows(); ++r)
        for (std::size_t c = start_col; c <= end_col; ++c)
        {
            const Scalar d = std::fabs(a.at_unchecked(r, c) - b.at_unchecked(r, c));
            if (d > worst) worst = d;
        }
    return worst;
}

}  // namespace

int main(int argc, char* argv[])
{
    bool use_cuda = false, use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--cuda") use_cuda = true;
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: doc_attn_test [--cuda|--gpu]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_cuda = use_cuda;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    const std::size_t vocab = 64, d_model = 16, seq = 8, heads = 2, d_ff = 32, layers = 2;
    GPTModel model(vocab, d_model, seq, heads, d_ff, layers,
                   PosEncodingType::Learned, ActivationType::GeLU, NormType::LayerNorm);
    { auto r = model.init(eng); if (!r) { std::cerr << "GPTModel init 失败: " << r.error().message << "\n"; return 1; } }

    // 单窗口：doc A 占 0..3，doc B 占 4..7
    const std::vector<std::size_t> doc_ids{1, 1, 1, 1, 2, 2, 2, 2};
    // 变体1 / 变体2：仅 doc A 的 token 不同，doc B 相同
    const std::vector<std::size_t> tok_v1{ 5,  7,  9, 11, 20, 21, 22, 23};
    const std::vector<std::size_t> tok_v2{50, 60, 70, 80, 20, 21, 22, 23};

    std::cout << "========================================\n";
    std::cout << "  文档感知掩码端到端正确性测试\n";
    std::cout << "========================================\n";

    // ── 1) 文档感知：doc B 输出应不受 doc A 影响（不变性） ──
    auto l1_doc = run_forward(eng, model, tok_v1, doc_ids);
    auto l2_doc = run_forward(eng, model, tok_v2, doc_ids);
    const Scalar diff_docB_doc = max_col_diff(l1_doc, l2_doc, 4, 7);

    std::cout << "  [文档感知] doc A 变化后，doc B 位置 logits 最大差 = "
              << diff_docB_doc << "\n";
    const bool invariant_ok = (diff_docB_doc <= Scalar{1e-4});

    // ── 2) 负对照：纯因果下，doc B 输出应随 doc A 变化（灵敏度） ──
    auto l1_causal = run_forward(eng, model, tok_v1, {});
    auto l2_causal = run_forward(eng, model, tok_v2, {});
    const Scalar diff_docB_causal = max_col_diff(l1_causal, l2_causal, 4, 7);

    std::cout << "  [纯因果]   doc A 变化后，doc B 位置 logits 最大差 = "
              << diff_docB_causal << "\n";
    const bool sensitive_ok = (diff_docB_causal > Scalar{1e-2});

    // ── 3) 顺带确认：文档感知与纯因果在 doc B 上确实不同（掩码确实生效） ──
    const Scalar diff_doc_vs_causal = max_col_diff(l1_doc, l1_causal, 4, 7);
    std::cout << "  [对照] 文档感知 vs 纯因果（同输入）doc B 最大差 = "
              << diff_doc_vs_causal << "\n";

    bool pass = true;
    if (!invariant_ok)
    {
        std::cout << "  ❌ 不变性失败：块对角掩码未阻止 doc A 影响 doc B\n";
        pass = false;
    }
    else std::cout << "  ✅ 不变性通过：doc B 输出与 doc A 无关\n";

    if (!sensitive_ok)
    {
        std::cout << "  ❌ 灵敏度失败：纯因果下 doc B 也未随 doc A 变化（测试无效）\n";
        pass = false;
    }
    else std::cout << "  ✅ 灵敏度通过：纯因果对照确实会受 doc A 影响\n";

    std::cout << (pass ? "  → ALL PASS\n" : "  → TEST FAILED\n");
    return pass ? 0 : 1;
}
