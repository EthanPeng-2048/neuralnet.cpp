// ── MaxPool2D 正确性检查（独立参考实现比对）───────────────────────────────
//
// 目的：把层实现与一份独立写法的窗口 max + argmax 散射逐元素比对，覆盖：
//         forward : 每窗口取 max
//         backward: 梯度散射回 argmax（窗口重叠时在同一位置累加）
//       并覆盖两条防御性契约（不显式断言就会静默通过的陷阱）：
//         ① clear_cache() 后直接 backward 必须报错 —— 缓存为空时不得读取；
//            警告：vector::clear() 保留容量，越界读未必崩溃（表现为"静默
//            沿用陈旧索引"而不报错），只有断言"必须报错"才抓得住
//         ② checkpoint 模式 forward 不驻留 argmax → backward 必须报错；
//            先 forward_recompute 重建缓存后必须与参考一致
//
// 用法：maxpool_gradcheck [--gpu]
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::MaxPool2D;

namespace {

// 独立参考：窗口 max + argmax 散射（不依赖层内任何实现）。
// 平局取"先出现者"，与层的 `if (v > best)` 判据一致。
struct PoolRef
{
    std::size_t C, H, W, P, S, OH, OW, B;
    Matrix x;   // (C*H*W, B)

    Scalar xv(std::size_t c, std::size_t ih, std::size_t iw, std::size_t b) const
    {
        return x.at_unchecked(c * H * W + ih * W + iw, b);
    }
    Matrix fwd() const
    {
        Matrix out(C * OH * OW, B);
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t b = 0; b < B; ++b)
                for (std::size_t oh = 0; oh < OH; ++oh)
                    for (std::size_t ow = 0; ow < OW; ++ow)
                    {
                        Scalar best = -std::numeric_limits<Scalar>::infinity();
                        for (std::size_t dh = 0; dh < P; ++dh)
                            for (std::size_t dw = 0; dw < P; ++dw)
                                best = std::max(best, xv(c, oh * S + dh, ow * S + dw, b));
                        out.set_value_unchecked(c * OH * OW + oh * OW + ow, b, best);
                    }
        return out;
    }
    // 独立参考的反向：按契约「窗口内并列最大值均分梯度」实现
    // （无并列最大值时 == argmax 散射，总梯度守恒）
    Matrix bwd(const Matrix& go) const
    {
        Matrix gin(C * H * W, B, Scalar{0});
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t b = 0; b < B; ++b)
                for (std::size_t oh = 0; oh < OH; ++oh)
                    for (std::size_t ow = 0; ow < OW; ++ow)
                    {
                        Scalar best = -std::numeric_limits<Scalar>::infinity();
                        for (std::size_t dh = 0; dh < P; ++dh)
                            for (std::size_t dw = 0; dw < P; ++dw)
                                best = std::max(best, xv(c, oh * S + dh, ow * S + dw, b));
                        std::size_t cnt = 0;
                        for (std::size_t dh = 0; dh < P; ++dh)
                            for (std::size_t dw = 0; dw < P; ++dw)
                                if (xv(c, oh * S + dh, ow * S + dw, b) == best) ++cnt;
                        if (cnt == 0) continue;
                        const std::size_t orr = c * OH * OW + oh * OW + ow;
                        const Scalar share =
                            go.at_unchecked(orr, b) / static_cast<Scalar>(cnt);
                        for (std::size_t dh = 0; dh < P; ++dh)
                            for (std::size_t dw = 0; dw < P; ++dw)
                            {
                                const std::size_t ih = oh * S + dh, iw = ow * S + dw;
                                if (xv(c, ih, iw, b) == best)
                                    gin.span()[(c * H * W + ih * W + iw) * B + b] += share;
                            }
                    }
        return gin;
    }
};

Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return Scalar{1e9f};
    Scalar e{0};
    for (std::size_t i = 0; i < a.span().size(); ++i)
        e = std::max(e, std::fabs(a.span()[i] - b.span()[i]));
    return e;
}

} // namespace

int main(int argc, char* argv[])
{
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--gpu") use_gpu = true;
        else if (a == "--help")
        {
            std::cout << "用法: maxpool_gradcheck [--gpu]\n";
            return 0;
        }
    }

    nn::cli::EngineConfig ecfg;
    ecfg.use_gpu = use_gpu;
    auto engine_res = nn::cli::create_engine(ecfg, std::cout);
    NN_EXIT(engine_res, 1, "引擎创建失败: ");
    auto engine = std::move(*engine_res);
    ComputeEngine& eng = *engine;

    std::cout << "========================================\n";
    std::cout << "  MaxPool2D 正确性检查（独立参考实现比对）\n";
    std::cout << "========================================\n";

    struct Case { std::size_t C, H, W, P, S, B; };
    const Case cases[] = {
        {2, 6, 6, 2, 2, 2},   // 标准（stride == pool）
        {2, 6, 6, 2, 1, 2},   // 重叠窗口（梯度在同一位置累加）
        {3, 5, 7, 2, 2, 2},   // 非整除尺寸（尾部余数）
        // pool=3（R = 9）：分组归约的 R 是**运行期**视图参数（不进 expr_spec_key）
        // → 任意池化窗口共享同一个融合 shader。本用例提供**非扫描窗口尺寸**
        // 的数值覆盖（库内扫描/模型 pass 只跑 pool∈{2,3}；真正锁死"R 不进 key"
        // 的契约断言在 expr_fused_key_test / expr_matmul_test 的 key 相等断言，
        // 本用例不是那条门禁——R 若被折回 key，扫描仍会注册 pool=3 的结构）。
        {2, 7, 7, 3, 3, 2},
    };
    const Scalar tol = 1e-6f;
    int failures = 0;

    std::mt19937_64 rng(20260919);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    for (const Case& cs : cases)
    {
        const std::size_t OH = (cs.H - cs.P) / cs.S + 1;
        const std::size_t OW = (cs.W - cs.P) / cs.S + 1;
        std::printf("── C=%zu H=%zu W=%zu pool=%zu stride=%zu B=%zu (out %zux%zu) ──\n",
                    cs.C, cs.H, cs.W, cs.P, cs.S, cs.B, OH, OW);

        MaxPool2D pool(cs.C, cs.H, cs.W, cs.P, cs.S);
        (void)pool.init(eng);   // M6 段 C：层由 init 绑定引擎
        Matrix xm(cs.C * cs.H * cs.W, cs.B);
        for (std::size_t i = 0; i < xm.size(); ++i) xm.span()[i] = dist(rng);
        Matrix gom(cs.C * OH * OW, cs.B);
        for (std::size_t i = 0; i < gom.size(); ++i) gom.span()[i] = dist(rng);
        PoolRef ref{cs.C, cs.H, cs.W, cs.P, cs.S, OH, OW, cs.B, xm};

        auto x = eng.from_matrix(xm);
        auto go = eng.from_matrix(gom);
        if (!x || !go) { std::cerr << "  上传失败\n"; return 1; }

        const auto report = [&](const char* what, const Matrix& got, const Matrix& want) {
            const Scalar e = max_abs_diff(got, want);
            const bool ok = e < tol;
            std::printf("  %-22s %s  max_abs_diff=%.3e\n", what, ok ? "OK" : "FAIL",
                        static_cast<double>(e));
            if (!ok) ++failures;
        };

        // ① forward / backward 与独立参考一致
        auto out = pool.forward(*x);
        NN_EXIT(out, 1, "  forward 失败: ");
        auto om = eng.to_matrix(*out);
        report("forward", *om, ref.fwd());

        auto gx = pool.backward(*go);
        NN_EXIT(gx, 1, "  backward 失败: ");
        auto gxm = eng.to_matrix(*gx);
        report("backward(grad_x)", *gxm, ref.bwd(gom));

        // ② clear_cache 后直接 backward 必须被拒绝
        pool.clear_cache();
        auto gx2 = pool.backward(*go);
        std::printf("  %-22s %s\n", "clear_cache→backward",
                    gx2 ? "FAIL（未拦截）" : "OK（已拒绝）");
        if (gx2) ++failures;

        // ③ checkpoint 模式：forward 不驻留 → backward 必须被拒绝；
        //    forward_recompute 重建缓存后必须与参考一致
        pool.set_checkpoint_mode(true);
        auto out3 = pool.forward(*x);
        NN_EXIT(out3, 1, "  ckpt forward 失败\n");
        auto gx3 = pool.backward(*go);
        std::printf("  %-22s %s\n", "ckpt→backward",
                    gx3 ? "FAIL（未拦截）" : "OK（已拒绝）");
        if (gx3) ++failures;

        auto rec = pool.forward_recompute(*x);
        NN_EXIT(rec, 1, "  forward_recompute 失败: ");
        auto gx4 = pool.backward(*go);
        if (!gx4)
        {
            std::cerr << "  recompute 后 backward 失败: " << gx4.error().message << "\n";
            ++failures;
        }
        else
        {
            auto gx4m = eng.to_matrix(*gx4);
            report("recompute 后 backward", *gx4m, ref.bwd(gom));
        }
        pool.set_checkpoint_mode(false);
        std::printf("\n");
    }

    // ── 平局语义（显式用例）：窗口内 3 路并列 → 梯度均分且总梯度守恒 ──
    // x 窗口 = [1,1;1,0]（max=1，三路并列），grad_out = 2
    // 期望 grad_x = [2/3, 2/3, 2/3, 0]，Σ = 2（守恒）
    {
        const std::size_t C = 1, H = 2, W = 2, P = 2, S = 2, B = 1;
        std::printf("── 平局用例：C=1 H=2 W=2 pool=2 stride=2（窗口 [1,1;1,0]）──\n");
        MaxPool2D pool(C, H, W, P, S);
        (void)pool.init(eng);   // M6 段 C：层由 init 绑定引擎
        Matrix xm(C * H * W, B);
        xm.span()[0] = 1; xm.span()[1] = 1; xm.span()[2] = 1; xm.span()[3] = 0;
        Matrix gom(C * 1 * 1, B);
        gom.span()[0] = 2;
        auto x = eng.from_matrix(xm);
        auto go = eng.from_matrix(gom);
        if (!x || !go) { std::cerr << "  上传失败\n"; return 1; }

        auto out = pool.forward(*x);
        if (!out)
        {
            std::cerr << "  平局用例 forward 失败: " << out.error().message << "\n";
            ++failures;
        }
        else
        {
            auto om = eng.to_matrix(*out);
            auto gx = pool.backward(*go);
            if (!gx || !om)
            {
                std::cerr << "  平局用例 backward 失败\n";
                ++failures;
            }
            else
            {
                auto gm = eng.to_matrix(*gx);
                const Scalar want = Scalar{2} / Scalar{3};
                const bool ok = std::fabs(om->at_unchecked(0, 0) - Scalar{1}) < 1e-6f &&
                                std::fabs(gm->at_unchecked(0, 0) - want) < 1e-6f &&
                                std::fabs(gm->at_unchecked(1, 0) - want) < 1e-6f &&
                                std::fabs(gm->at_unchecked(2, 0) - want) < 1e-6f &&
                                std::fabs(gm->at_unchecked(3, 0)) < 1e-6f;
                Scalar sum = 0;
                for (std::size_t i = 0; i < gm->size(); ++i) sum += gm->span()[i];
                const bool conserved = std::fabs(sum - Scalar{2}) < 1e-6f;
                std::printf("  %-22s %s（期望 [2/3,2/3,2/3,0]）\n", "并列均分",
                            ok ? "OK" : "FAIL");
                std::printf("  %-22s %s（Σgrad=%.6f，期望 2）\n", "总梯度守恒",
                            conserved ? "OK" : "FAIL", double(sum));
                if (!ok || !conserved) ++failures;
            }
        }
        std::printf("\n");
    }

    std::cout << "----------------------------------------\n";
    std::cout << "  结果: " << (failures == 0 ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return failures == 0 ? 0 : 1;
}
