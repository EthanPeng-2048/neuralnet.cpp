// ── 文档感知掩码端到端正确性测试（doc_attn_test） ─────────────────────
//
// 原理（不变性 + 跨样本隔离）：
//   若块对角掩码正确生效，窗口内 doc B 的位置只能注意同文档内的位置，
//   因此 doc B 的输出应完全不受 doc A 内容影响；batch=2 下样本1 的输出
//   还必须与样本0 的变化完全无关（隔离——batch=1 时两种布局重合测不出，
//   见铁律 5）。
//
// 测试步骤（batch=2，样本0=文档窗口、样本1=固定序列）：
//   1) 样本0：doc A 占 0-3，doc B 占 4-7，doc_ids=[1,1,1,1,2,2,2,2]
//   2) 变体1: 样本0 的 doc A 用 token A1；变体2: 用完全不同的 A2
//      （doc B 与样本1 的输入两次运行完全相同）
//   3) 设 doc_ids 跑两次 forward → 样本0 doc B 列必须【相同】（不变性）；
//      样本1 的列必须【逐位相同】（跨样本隔离）
//   4) 负对照：清空 doc_ids（纯因果）跑同样两组 → 样本0 doc B 必须
//      【不同】（灵敏度——证明测试能抓住"掩码没生效"）；样本1 仍须相同
//
// 用法：doc_attn_test [--gpu]
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

// 运行一次 forward（batch=2），返回 logits（vocab × seq*batch，batch-major
// 列区间：样本 b = [b*seq, b*seq+seq)）。空 doc 向量 = 纯因果。
Matrix run_forward(ComputeEngine& eng, GPTModel& model,
                   const std::vector<std::size_t>& tokens0,
                   const std::vector<std::size_t>& tokens1,
                   const std::vector<std::size_t>& doc0,
                   const std::vector<std::size_t>& doc1)
{
    const std::size_t seq = tokens0.size();
    const std::size_t batch = 2;
    Matrix x_m(seq, batch);              // (位置, 样本)：元素 = token id
    for (std::size_t b = 0; b < batch; ++b)
    {
        const std::vector<std::size_t>& tk = (b == 0) ? tokens0 : tokens1;
        for (std::size_t t = 0; t < seq; ++t)
            x_m.set_value_unchecked(t, b, static_cast<Scalar>(tk[t]));
    }

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "  from_matrix 失败\n"; std::abort(); }

    // set_doc_ids 收 batch-major (b*seq+t) 全量；两样本皆空 = 关闭
    std::vector<std::size_t> ids(doc0);
    ids.insert(ids.end(), doc1.begin(), doc1.end());
    model.set_doc_ids(ids);
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
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: doc_attn_test [--gpu]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    if (!engine_res) { std::cerr << "引擎创建失败: " << engine_res.error().message << "\n"; return 1; }
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    const std::size_t vocab = 64, d_model = 16, seq = 8, heads = 2, d_ff = 32, layers = 2;
    GPTModel model(vocab, d_model, seq, heads, d_ff, layers,
                   PosEncodingType::Learned, ActivationType::GeLU, NormType::LayerNorm);
    { auto r = model.init(eng); if (!r) { std::cerr << "GPTModel init 失败: " << r.error().message << "\n"; return 1; } }

    // batch=2（铁律 5：注意力测试必须覆盖 batch>1）：
    //   样本0 = 文档窗口（doc A 占 0..3、doc B 占 4..7）；
    //   样本1 = 固定序列（两次运行完全相同）→ 其 logits 只可能因跨样本
    //   串扰而变（position-major 类历史 bug 在 batch=1 下不可见）。
    const std::vector<std::size_t> doc_ids0{1, 1, 1, 1, 2, 2, 2, 2};
    const std::vector<std::size_t> doc_ids1{3, 3, 3, 3, 4, 4, 4, 4};
    // 变体1 / 变体2：仅样本0 的 doc A token 不同，doc B 与样本1 相同
    const std::vector<std::size_t> tok_v1{ 5,  7,  9, 11, 20, 21, 22, 23};
    const std::vector<std::size_t> tok_v2{50, 60, 70, 80, 20, 21, 22, 23};
    const std::vector<std::size_t> tok_s{ 6,  8, 10, 12, 30, 31, 32, 33};
    // logits 列区间（batch-major (vocab, seq*batch)）：样本0=0..7，样本1=8..15

    std::cout << "========================================\n";
    std::cout << "  文档感知掩码端到端正确性测试\n";
    std::cout << "========================================\n";

    // ── 1) 文档感知：样本0 的 doc B 输出应不受 doc A 影响（不变性） ──
    auto l1_doc = run_forward(eng, model, tok_v1, tok_s, doc_ids0, doc_ids1);
    auto l2_doc = run_forward(eng, model, tok_v2, tok_s, doc_ids0, doc_ids1);
    const Scalar diff_docB_doc = max_col_diff(l1_doc, l2_doc, 4, 7);

    std::cout << "  [文档感知] doc A 变化后，doc B 位置 logits 最大差 = "
              << diff_docB_doc << "\n";
    const bool invariant_ok = (diff_docB_doc <= Scalar{1e-4});

    // ── 1b) 跨样本隔离：样本1 输入两次完全相同 → 列 8..15 必须逐位不变
    //        （样本0 变了还能传进来 = batch 串扰，batch=1 测不出）
    const Scalar diff_iso_doc = max_col_diff(l1_doc, l2_doc, 8, 15);
    std::cout << "  [隔离]     样本0 doc A 变化后，样本1 logits 最大差 = "
              << diff_iso_doc << "\n";
    const bool isolation_ok = (diff_iso_doc <= Scalar{1e-4});

    // ── 2) 负对照：纯因果下，doc B 输出应随 doc A 变化（灵敏度） ──
    auto l1_causal = run_forward(eng, model, tok_v1, tok_s, {}, {});
    auto l2_causal = run_forward(eng, model, tok_v2, tok_s, {}, {});
    const Scalar diff_docB_causal = max_col_diff(l1_causal, l2_causal, 4, 7);

    std::cout << "  [纯因果]   doc A 变化后，doc B 位置 logits 最大差 = "
              << diff_docB_causal << "\n";
    const bool sensitive_ok = (diff_docB_causal > Scalar{1e-2});

    // ── 2b) 纯因果下的跨样本隔离（同样应成立） ──
    const Scalar diff_iso_causal = max_col_diff(l1_causal, l2_causal, 8, 15);
    std::cout << "  [隔离·因果] 样本0 变化后，样本1 logits 最大差 = "
              << diff_iso_causal << "\n";
    const bool isolation_causal_ok = (diff_iso_causal <= Scalar{1e-4});

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

    if (!isolation_ok || !isolation_causal_ok)
    {
        std::cout << "  ❌ 隔离失败：样本0 的变化泄漏到样本1（batch 串扰，"
                  << "铁律 5 盲区）\n";
        pass = false;
    }
    else std::cout << "  ✅ 隔离通过：样本1 输出与样本0 无关（batch=2 无串扰）\n";

    if (!sensitive_ok)
    {
        std::cout << "  ❌ 灵敏度失败：纯因果下 doc B 也未随 doc A 变化（测试无效）\n";
        pass = false;
    }
    else std::cout << "  ✅ 灵敏度通过：纯因果对照确实会受 doc A 影响\n";

    std::cout << (pass ? "  → ALL PASS\n" : "  → TEST FAILED\n");
    return pass ? 0 : 1;
}
