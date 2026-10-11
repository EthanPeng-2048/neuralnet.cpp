// ── probe_scale.cpp — 三值量化「尺度规则 × 粒度」对真实权重的影响 ──────────────
// 配套报告：research/ternary_scale/REPORT.md
// 设计文档：docs/development/21-quantized-weights.md（这条对应待定项 D1）
//
// 问题：三值化 w -> q*tau（q ∈ {-1,0,+1}）里，tau 取
//   · 统计量：absmean（BitNet b1.58）还是 absmax（llama.cpp TQ1_0）
//   · 粒度：per-tensor / per-row（按输出通道）/ per-256 / per-128 / per-64
// 对「多少权重被量成 0」与「重构误差」造成多大差别。
//
// 指标（raw 求和后跨矩阵加权，最后统一算比值）：
//   zero%   = 被量成 0 的权重比例
//   relerr% = ||W - tau*q||_F / ||W||_F
//   outerr% = ||(tau*q)x - Wx||_F / ||Wx||_F，x ~ N(0,1)（模拟该层输出被扰动多少）
//
// 用法（可给多个模型文件）：
//   probe_scale <model.bin> [model2.bin ...]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using nn::Matrix;
using nn::Precision;
using nn::Scalar;

namespace
{

enum class Rule { AbsMean, AbsMax };
enum class Gran { Tensor, Row, Blk256, Blk128, Blk64 };

constexpr std::size_t kOutCols = 4;   // 输出误差用的随机 x 列数
constexpr int kNumRules = 2;
constexpr int kNumGrans = 5;

const char* gran_name(Gran g)
{
    switch (g)
    {
        case Gran::Tensor: return "tensor";
        case Gran::Row:    return "row";
        case Gran::Blk256: return "blk256";
        case Gran::Blk128: return "blk128";
        case Gran::Blk64:  return "blk64";
    }
    return "?";
}

std::size_t gran_block(Gran g)
{
    switch (g)
    {
        case Gran::Blk256: return 256;
        case Gran::Blk128: return 128;
        case Gran::Blk64:  return 64;
        default:           return 0;
    }
}

// 原始累加量（不预先算比值，便于跨矩阵加权）
struct Raw
{
    double zero = 0.0;   // 零元素个数
    double pos = 0.0;    // +1 个数
    double neg = 0.0;    // -1 个数
    double wnum = 0.0;   // Σ (w - ŵ)^2
    double wden = 0.0;   // Σ w^2
    double onum = 0.0;   // Σ (ŷ - y)^2
    double oden = 0.0;   // Σ y^2
};

// 逐元素三值化：q = (w > 0.5*tau) ? +1 : (w < -0.5*tau ? -1 : 0)
inline int ternarize(Scalar w, double tau)
{
    if (tau <= 0.0) return 0;
    if (static_cast<double>(w) > 0.5 * tau) return 1;
    if (static_cast<double>(w) < -0.5 * tau) return -1;
    return 0;
}

// 对单个矩阵做一次 (规则, 粒度) 的评估
Raw evaluate(const Matrix& m, const std::vector<Scalar>& x, Rule rule, Gran gran)
{
    const auto w = m.span();
    const std::size_t n = w.size();
    const std::size_t cols = m.cols();
    const std::size_t block = (gran == Gran::Row) ? cols
                            : (gran == Gran::Tensor) ? n : gran_block(gran);

    std::vector<Scalar> wq(n, Scalar{0});
    Raw r;

    for (std::size_t begin = 0; begin < n; begin += block)
    {
        const std::size_t end = std::min(begin + block, n);
        double tau = 0.0;
        if (rule == Rule::AbsMean)
        {
            double s = 0.0;
            for (std::size_t i = begin; i < end; ++i) s += std::fabs(static_cast<double>(w[i]));
            tau = s / static_cast<double>(end - begin);
        }
        else
        {
            for (std::size_t i = begin; i < end; ++i)
                tau = std::max(tau, std::fabs(static_cast<double>(w[i])));
        }
        for (std::size_t i = begin; i < end; ++i)
        {
            const int q = ternarize(w[i], tau);
            wq[i] = static_cast<Scalar>(static_cast<double>(q) * tau);
            const double d = static_cast<double>(w[i]) - static_cast<double>(wq[i]);
            if (q == 0) r.zero += 1.0;
            else if (q > 0) r.pos += 1.0;
            else r.neg += 1.0;
            r.wnum += d * d;
            r.wden += static_cast<double>(w[i]) * static_cast<double>(w[i]);
        }
    }

    // 输出误差：y = W x，ŷ = Wq x（x 按 cols × kOutCols 列存）
    for (std::size_t rr = 0; rr < m.rows(); ++rr)
    {
        for (std::size_t j = 0; j < kOutCols; ++j)
        {
            double y = 0.0, yh = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                const double xv = static_cast<double>(x[c * kOutCols + j]);
                y  += static_cast<double>(w[rr * cols + c]) * xv;
                yh += static_cast<double>(wq[rr * cols + c]) * xv;
            }
            const double d = yh - y;
            r.onum += d * d;
            r.oden += y * y;
        }
    }
    return r;
}

constexpr Gran kGrans[kNumGrans] = {Gran::Tensor, Gran::Row, Gran::Blk256, Gran::Blk128, Gran::Blk64};
constexpr Rule kRules[kNumRules] = {Rule::AbsMean, Rule::AbsMax};

inline double pct(double num, double den)
{
    return den > 0.0 ? 100.0 * std::sqrt(num / den) : 0.0;
}

// 三值分布的信息熵（bit/权重）：把"多少权重存活"翻译成"实际携带多少信息"
inline double entropy_bits(const Raw& r)
{
    const double n = r.zero + r.pos + r.neg;
    if (n <= 0.0) return 0.0;
    double h = 0.0;
    for (double c : {r.zero, r.pos, r.neg})
    {
        if (c <= 0.0) continue;
        const double p = c / n;
        h -= p * std::log2(p);
    }
    return h;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("用法: probe_scale <model.bin> [model2.bin ...]\n");
        return 1;
    }

    for (int ai = 1; ai < argc; ++ai)
    {
        const std::string path = argv[ai];
        std::printf("\n================ %s ================\n", path.c_str());

        nn::CpuEngine eng;
        auto spec_r = nn::peek_model_spec(path);
        if (!spec_r) { std::printf("peek 失败: %s\n", spec_r.error().message.c_str()); continue; }
        const nn::ModelSpec& spec = *spec_r;
        std::printf("spec: type=%u d_model=%zu d_ff=%zu layers=%zu vocab=%zu layer_dims=%zu\n",
                    static_cast<unsigned>(spec.type), spec.d_model, spec.d_ff,
                    spec.num_layers, spec.vocab_size, spec.layer_dims.size());

        nn::Result<nn::Model> built = (spec.is_gpt() || spec.is_alibi_gpt())
            ? nn::build_gpt_model_from_spec(eng, spec)
            : nn::build_mnist_model_from_spec(eng, spec);
        if (!built) { std::printf("build 失败: %s\n", built.error().message.c_str()); continue; }
        nn::Model model = std::move(*built);

        auto loaded = nn::load_model(path, model);
        if (!loaded) { std::printf("load 失败: %s\n", loaded.error().message.c_str()); continue; }

        auto params = model.parameters();

        Raw agg[kNumRules][kNumGrans];
        std::size_t total_w = 0, n_mat = 0, n_skip = 0;

        std::printf("\n%-4s %-13s %-11s %-10s %-9s  [absmean@row]  [absmax@row]  [absmean@256] [absmax@256]\n",
                    "#", "shape", "mean|w|", "max|w|", "max/mean");

        for (std::size_t pi = 0; pi < params.size(); ++pi)
        {
            const nn::Tensor& t = params[pi].get();
            if (t.cols() < 2) { ++n_skip; continue; }   // 跳过 bias（(out,1)）
            auto mr = eng.to_matrix(t, Precision::F32);
            if (!mr) continue;
            const Matrix& m = *mr;

            std::mt19937 rng(20261010u + static_cast<unsigned>(pi));
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<Scalar> x(m.cols() * kOutCols);
            for (auto& v : x) v = nd(rng);

            double s = 0.0, mx = 0.0;
            for (auto wv : m.span())
            {
                const double a = std::fabs(static_cast<double>(wv));
                s += a;
                mx = std::max(mx, a);
            }
            const double mean = s / static_cast<double>(m.size());

            double z[kNumRules][kNumGrans], rl[kNumRules][kNumGrans];
            for (int ri = 0; ri < kNumRules; ++ri)
                for (int gi = 0; gi < kNumGrans; ++gi)
                {
                    const Raw r = evaluate(m, x, kRules[ri], kGrans[gi]);
                    z[ri][gi]  = 100.0 * r.zero / static_cast<double>(m.size());
                    rl[ri][gi] = pct(r.wnum, r.wden);
                    agg[ri][gi].zero += r.zero;
                    agg[ri][gi].pos += r.pos;
                    agg[ri][gi].neg += r.neg;
                    agg[ri][gi].wnum += r.wnum;
                    agg[ri][gi].wden += r.wden;
                    agg[ri][gi].onum += r.onum;
                    agg[ri][gi].oden += r.oden;
                }

            total_w += m.size();
            ++n_mat;

            const std::string shape =
                std::to_string(m.rows()) + "x" + std::to_string(m.cols());
            std::printf("%-4zu %-13s %-11.5f %-10.5f %-9.2f  "
                        "%5.1f%%/%5.1f%%  %5.1f%%/%5.1f%%  %5.1f%%/%5.1f%%  %5.1f%%/%5.1f%%\n",
                        pi, shape.c_str(), mean, mx, mean > 0.0 ? mx / mean : 0.0,
                        z[0][1], rl[0][1], z[1][1], rl[1][1],
                        z[0][2], rl[0][2], z[1][2], rl[1][2]);
        }

        std::printf("\n-- 汇总（按元素数加权；矩阵 %zu 个 / 权重 %zu 个，跳过 bias %zu 个）--\n",
                    n_mat, total_w, n_skip);
        std::printf("%-8s %-30s %-30s\n", "粒度",
                    "absmean zero%/relerr%/bits", "absmax zero%/relerr%/bits");
        for (int gi = 0; gi < kNumGrans; ++gi)
        {
            std::printf("%-8s %8.1f%% / %8.1f%% / %5.3f      %8.1f%% / %8.1f%% / %5.3f\n",
                        gran_name(kGrans[gi]),
                        agg[0][gi].zero * 100.0 / static_cast<double>(total_w),
                        pct(agg[0][gi].wnum, agg[0][gi].wden),
                        entropy_bits(agg[0][gi]),
                        agg[1][gi].zero * 100.0 / static_cast<double>(total_w),
                        pct(agg[1][gi].wnum, agg[1][gi].wden),
                        entropy_bits(agg[1][gi]));
        }

        std::printf("\n-- 层输出相对误差（%%，全模型加权）--\n");
        std::printf("%-8s %-12s %-12s\n", "粒度", "absmean", "absmax");
        for (int gi = 0; gi < kNumGrans; ++gi)
            std::printf("%-8s %11.2f%% %11.2f%%\n", gran_name(kGrans[gi]),
                        pct(agg[0][gi].onum, agg[0][gi].oden),
                        pct(agg[1][gi].onum, agg[1][gi].oden));
    }
    return 0;
}
