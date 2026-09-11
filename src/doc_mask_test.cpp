// ── 文档感知注意力掩码构建单测（doc_mask_test） ─────────────────────────
//
// 目的：验证 build_attention_mask 的块对角 ∧ 因果 ∧ ALiBi 组合逻辑。
//       纯 CPU / Matrix 测试，无需引擎，可独立运行验证掩码数学正确性。
//
// 用法：doc_mask_test
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

using nn::Scalar;
using nn::Matrix;

namespace {

int failures = 0;

void check_row(const Matrix& m, std::size_t row,
               const std::vector<Scalar>& expected)
{
    for (std::size_t j = 0; j < expected.size(); ++j)
    {
        const Scalar got = m.at_unchecked(row, j);
        const Scalar exp = expected[j];
        bool ok;
        if (std::isinf(exp) && exp < 0)            // 期望 -inf
            ok = std::isinf(got) && got < 0;
        else
            ok = std::fabs(got - exp) <= Scalar{1e-6}
                 * (Scalar{1} + std::fabs(got) + std::fabs(exp));
        if (!ok)
        {
            ++failures;
            std::printf("  FAIL row=%zu col=%zu got=%g exp=%g\n",
                        row, j, static_cast<double>(got), static_cast<double>(exp));
        }
    }
}

// 纯因果：mask[i][j] = 0 if j<=i else -inf，所有 batch/head 平铺一致
void test_causal()
{
    const std::size_t B = 2, S = 4, H = 2;
    auto m = nn::build_attention_mask(B, S, H, false, {});
    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    for (std::size_t b = 0; b < B; ++b)
        for (std::size_t h = 0; h < H; ++h)
            for (std::size_t i = 0; i < S; ++i)
            {
                std::vector<Scalar> exp(S);
                for (std::size_t j = 0; j < S; ++j)
                    exp[j] = (j <= i) ? Scalar{0} : neg_inf;
                check_row(m, (b * H + h) * S + i, exp);
            }
    std::printf("[causal-no-doc] done\n");
}

// 块对角：跨文档位置 -inf，同文档内仍因果
void test_block_diag()
{
    const std::size_t S = 6;
    const std::vector<std::size_t> doc{0, 0, 1, 1, 1, 2};  // 3 篇文档
    auto m = nn::build_attention_mask(1, S, 1, false, {}, doc);
    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    for (std::size_t i = 0; i < S; ++i)
    {
        std::vector<Scalar> exp(S);
        for (std::size_t j = 0; j < S; ++j)
            exp[j] = (j <= i && doc[i] == doc[j]) ? Scalar{0} : neg_inf;
        check_row(m, i, exp);
    }
    std::printf("[block-diag] done\n");
}

// ALiBi：允许位置叠加 -m_h*(i-j)，未来位置 -inf
void test_alibi()
{
    const std::size_t S = 3;
    const std::vector<Scalar> slopes{1.0, 0.5};
    auto m = nn::build_attention_mask(1, S, 2, true, slopes);
    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    for (std::size_t h = 0; h < 2; ++h)
        for (std::size_t i = 0; i < S; ++i)
        {
            std::vector<Scalar> exp(S);
            for (std::size_t j = 0; j < S; ++j)
                exp[j] = (j <= i) ? -slopes[h] * static_cast<Scalar>(i - j) : neg_inf;
            check_row(m, h * S + i, exp);
        }
    std::printf("[alibi] done\n");
}

// 块对角 + ALiBi 组合
void test_block_diag_alibi()
{
    const std::size_t S = 4;
    const std::vector<std::size_t> doc{0, 0, 1, 1};
    const std::vector<Scalar> slopes{0.25};
    auto m = nn::build_attention_mask(1, S, 1, true, slopes, doc);
    const Scalar neg_inf = -std::numeric_limits<Scalar>::infinity();
    for (std::size_t i = 0; i < S; ++i)
    {
        std::vector<Scalar> exp(S);
        for (std::size_t j = 0; j < S; ++j)
        {
            if (j <= i && doc[i] == doc[j])
                exp[j] = -slopes[0] * static_cast<Scalar>(i - j);
            else
                exp[j] = neg_inf;
        }
        check_row(m, i, exp);
    }
    std::printf("[block-diag+alibi] done\n");
}

}  // namespace

int main()
{
    test_causal();
    test_block_diag();
    test_alibi();
    test_block_diag_alibi();

    if (failures == 0) { std::printf("ALL PASS\n"); return 0; }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
