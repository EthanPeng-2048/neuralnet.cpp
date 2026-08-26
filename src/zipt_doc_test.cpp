// ── ZiPT 块级文档感知掩码端到端正确性测试（zipt_doc_test） ────────────────
//
// 验证 ZiPTBlock 的文档边界掩码（build_mask_ 叠加 doc_ids）正确生效。
// 在块级做测试：记忆输入固定、局部窗口跨两篇文档，因此无全局记忆串扰，
// 可直接观测"局部 query 是否被禁止注意跨文档局部 key"。
//
// 原理（不变性验证，同 doc_attn_test）：
//   单窗口 (window=8, batch=1)，doc A 占 0-3，doc B 占 4-7，doc_ids=[1,1,1,1,2,2,2,2]。
//   - 设 doc_ids：doc B 位置只能注意同文档（doc B）局部 key + 固定记忆，
//     因此改变 doc A 局部 token 不应改变 doc B 位置的输出。
//   - 负对照：清空 doc_ids（纯因果），doc B 位置可注意 doc A 局部 key，
//     doc B 输出应随 doc A 改变（证明测试有灵敏度，能抓住"掩码没生效"）。
//
// 用法：zipt_doc_test
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::CpuEngine;
using nn::ZiPTBlock;
using nn::NormType;
using nn::ActivationType;

namespace {

constexpr std::size_t kDModel = 8, kHeads = 2, kDFF = 16, kWindow = 8, kMemory = 2;

// 跑一次块前向，返回输出 Matrix（d_model × window）。
Matrix run_forward(CpuEngine& eng, ZiPTBlock& blk,
                   const std::vector<Scalar>& x_flat,
                   const std::vector<Scalar>& mem_flat,
                   const std::vector<std::size_t>& doc_ids)
{
    Matrix x_m(kDModel, kWindow);
    for (std::size_t r = 0; r < kDModel; ++r)
        for (std::size_t c = 0; c < kWindow; ++c)
            x_m.set_value_unchecked(r, c, x_flat[r * kWindow + c]);
    Matrix mem_m(kDModel, kMemory);
    for (std::size_t r = 0; r < kDModel; ++r)
        for (std::size_t c = 0; c < kMemory; ++c)
            mem_m.set_value_unchecked(r, c, mem_flat[r * kMemory + c]);

    auto x = eng.from_matrix(x_m);
    if (!x) { std::cerr << "  from_matrix(x) 失败\n"; std::abort(); }
    auto m = eng.from_matrix(mem_m);
    if (!m) { std::cerr << "  from_matrix(mem) 失败\n"; std::abort(); }

    blk.set_doc_ids(doc_ids);   // 空 = 纯因果局部掩码
    auto y = blk.forward(eng, *x, *m);
    if (!y) { std::cerr << "  ZiPTBlock forward 失败: " << y.error().message << "\n"; std::abort(); }
    auto ym = eng.to_matrix(*y);
    if (!ym) { std::cerr << "  to_matrix 失败\n"; std::abort(); }
    return *ym;
}

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

int main()
{
    CpuEngine eng;
    ZiPTBlock blk(kDModel, kHeads, kDFF, kWindow, kMemory,
                  NormType::LayerNorm, ActivationType::GeLU);
    { auto r = blk.init(eng); if (!r) { std::cerr << "ZiPTBlock init 失败\n"; return 1; } }

    std::mt19937_64 rng{3};
    std::normal_distribution<Scalar> nd(0.0, 1.0);

    // 记忆固定（跨变体不变）
    std::vector<Scalar> mem_flat(kDModel * kMemory);
    for (auto& v : mem_flat) v = nd(rng);

    // 变体1 / 变体2：局部窗口列 4-7（doc B）相同，列 0-3（doc A）不同
    std::vector<Scalar> x_v1(kDModel * kWindow), x_v2(kDModel * kWindow);
    for (std::size_t r = 0; r < kDModel; ++r)
        for (std::size_t c = 0; c < kWindow; ++c)
        {
            const Scalar v = nd(rng);
            x_v1[r * kWindow + c] = v;
            x_v2[r * kWindow + c] = v;
        }
    for (std::size_t r = 0; r < kDModel; ++r)
        for (std::size_t c = 0; c < 4; ++c)   // doc A 大幅扰动
            x_v2[r * kWindow + c] += static_cast<Scalar>((r + 1) * (c + 2)) * 0.5f + 3.0f;

    const std::vector<std::size_t> doc_ids{1, 1, 1, 1, 2, 2, 2, 2};

    std::cout << "========================================\n";
    std::cout << "  ZiPT 块级文档感知掩码正确性测试\n";
    std::cout << "========================================\n";

    // ── 1) 文档感知：doc B 输出应不受 doc A 局部变化影响（不变性） ──
    const auto l1_doc = run_forward(eng, blk, x_v1, mem_flat, doc_ids);
    const auto l2_doc = run_forward(eng, blk, x_v2, mem_flat, doc_ids);
    const Scalar diff_docB_doc = max_col_diff(l1_doc, l2_doc, 4, 7);
    std::cout << "  [文档感知] doc A 变化后，doc B 位置输出最大差 = "
              << diff_docB_doc << "\n";
    const bool invariant_ok = (diff_docB_doc <= Scalar{1e-4});

    // ── 2) 负对照：纯因果下，doc B 应随 doc A 变化（灵敏度） ──
    const auto l1_causal = run_forward(eng, blk, x_v1, mem_flat, {});
    const auto l2_causal = run_forward(eng, blk, x_v2, mem_flat, {});
    const Scalar diff_docB_causal = max_col_diff(l1_causal, l2_causal, 4, 7);
    std::cout << "  [纯因果]   doc A 变化后，doc B 位置输出最大差 = "
              << diff_docB_causal << "\n";
    const bool sensitive_ok = (diff_docB_causal > Scalar{1e-2});

    // ── 3) 顺带：文档感知与纯因果在 doc B 上确实不同（掩码生效） ──
    const Scalar diff_doc_vs_causal = max_col_diff(l1_doc, l1_causal, 4, 7);
    std::cout << "  [对照] 文档感知 vs 纯因果（同输入）doc B 最大差 = "
              << diff_doc_vs_causal << "\n";

    bool pass = true;
    if (!invariant_ok)
    {
        std::cout << "  ❌ 不变性失败：ZiPT 局部文档掩码未阻止 doc A 影响 doc B\n";
        pass = false;
    }
    else std::cout << "  ✅ 不变性通过：doc B 输出与 doc A 无关（掩码生效）\n";

    if (!sensitive_ok)
    {
        std::cout << "  ❌ 灵敏度失败：纯因果下 doc B 也未随 doc A 变化（测试无效）\n";
        pass = false;
    }
    else std::cout << "  ✅ 灵敏度通过：纯因果对照确实会受 doc A 影响\n";

    std::cout << (pass ? "  → ALL PASS\n" : "  → TEST FAILED\n");
    return pass ? 0 : 1;
}
