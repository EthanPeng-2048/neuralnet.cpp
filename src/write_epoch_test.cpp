// ── write_epoch_test — 写版本（write epoch）契约 + BitLinear 量化缓存 ──────
// 设计依据：docs/development/21-quantized-weights.md §4.11（F1/F4/F6 落地）
//
// 覆盖：
//   [1] 逐写入口：每一个"经引擎媒介写入张量存储"的公共入口都推进写版本
//       （write / set_index / copy_from / cast_into / copy_into / insert_rows /
//        scatter_add_rows / add_inplace / scale_inplace / zero / accumulate /
//        eval_expr_into）—— 这是"版本没变 ⇒ 内容没变"的前提
//   [2] 只读路径**不**推进（read / to_matrix / dsl::compute / matmul / clone）
//   [3] 别名可见性：Tensor 拷贝共享存储 → 经别名写入，原句柄的版本也前进
//   [4] BitLinear 缓存：同权重下多次 forward 只量化一次；输入变化不影响缓存
//   [5] BitLinear 失效：权重经引擎写入（含优化器的 compute_into 路径）后
//       必须重算，且**重算结果与从零重算逐位相同**（缓存不改数值）
//   [6] clear_cache 只清激活，不清权重派生物（τ/wq 保留，不白算）
//   [7] 梯度累积/生成期的等价性：N 次 forward 的输出与"每次都重算"逐位相同
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_layer_bitlinear.hpp"
#include "neuralnet.cpp/compute_optimizer.hpp"
#include "neuralnet.cpp/expr_dsl.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

// ── 便捷：引擎上建一个已填零的 f32 张量 ───────────────────────────────────
nn::Tensor zeros(nn::ComputeEngine& eng, std::size_t rows, std::size_t cols)
{
    nn::Tensor t = eng.create_tensor(rows, cols, nn::Precision::F32, nn::InitSpec::zero());
    NN_ASSERT(t.valid(), "create_tensor(zero) failed");
    return t;
}

nn::Tensor filled(nn::ComputeEngine& eng, std::size_t rows, std::size_t cols, float v)
{
    nn::Matrix m(rows, cols);
    for (auto& x : m.span()) x = v;
    return upload(eng, m);
}

// 每次写调用前后比对版本号：严格前进 = 契约成立
template <class Fn>
void expect_bump(nn::ComputeEngine& eng, const char* what, nn::Tensor& t, Fn&& fn)
{
    const std::uint64_t before = t.write_epoch();
    auto r = fn();
    if (!r.has_value())
        std::printf("[FAIL] %s: 调用失败 —— %s\n", what, r.error().message.c_str());
    CHECK(r.has_value(), what);
    const std::uint64_t after = t.write_epoch();
    if (!(after > before))
        std::printf("[FAIL] %s: 写版本未前进（before=%llu after=%llu）\n", what,
                    static_cast<unsigned long long>(before),
                    static_cast<unsigned long long>(after));
    CHECK(after > before, what);
    (void)eng;
}

template <class Fn>
void expect_no_bump(const char* what, const nn::Tensor& t, Fn&& fn)
{
    const std::uint64_t before = t.write_epoch();
    auto r = fn();
    if (!r.has_value())
        std::printf("[FAIL] %s: 调用失败 —— %s\n", what, r.error().message.c_str());
    CHECK(r.has_value(), what);
    if (t.write_epoch() != before)
        std::printf("[FAIL] %s: 只读路径推进了写版本（before=%llu after=%llu）\n", what,
                    static_cast<unsigned long long>(before),
                    static_cast<unsigned long long>(t.write_epoch()));
    CHECK(t.write_epoch() == before, what);
}

// ══════════════════════════════════════════════════════════════════════════
// [1] 逐写入口推进写版本
// ══════════════════════════════════════════════════════════════════════════
void test_write_entries(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[1] 写入口逐条推进写版本（%s）\n", tag);

    nn::Tensor t = zeros(eng, 3, 2);
    nn::Tensor src = filled(eng, 3, 2, 1.0f);
    nn::Tensor idx = filled(eng, 1, 1, 0.0f);       // 行索引 0
    nn::Matrix m(3, 2);
    for (auto& x : m.span()) x = 0.5f;

    std::vector<nn::Scalar> flat(6, 0.25f);
    expect_bump(eng, "write 推进写版本", t, [&]
    {
        return eng.write(t, std::span<nn::Scalar>(flat));
    });
    expect_bump(eng, "set_index 推进写版本", t, [&]
    {
        return eng.set_index(t, 0, 0, 1.0f);
    });
    expect_bump(eng, "copy_from 推进写版本", t, [&]
    {
        return eng.copy_from(t, m);
    });
    expect_bump(eng, "accumulate 推进写版本", t, [&]
    {
        return eng.accumulate(t, src);
    });
    expect_bump(eng, "add_inplace 推进写版本", t, [&]
    {
        return eng.add_inplace(t, src);
    });
    expect_bump(eng, "scale_inplace 推进写版本", t, [&]
    {
        return eng.scale_inplace(t, 2.0f);
    });
    {
        // insert_rows 要求 dst 能容纳 src 的全部行
        nn::Tensor tall = zeros(eng, 4, 2);
        expect_bump(eng, "insert_rows 推进写版本", tall, [&]
        {
            return eng.insert_rows(tall, 1, src);
        });
    }
    expect_bump(eng, "scatter_add_rows 推进写版本", t, [&]
    {
        return eng.scatter_add_rows(t, idx, src);
    });
    expect_bump(eng, "eval_expr_into（DSL）推进写版本", t, [&]
    {
        // 优化器更新参数走的就是这条（dsl::compute_into → eval_expr_into）
        return nn::dsl::compute_into(eng,
            nn::dsl::leaf(t) + nn::dsl::leaf(src), t);
    });
    {
        // copy_into / cast_into：目标传递的其余两个出口
        nn::Tensor dst = zeros(eng, 3, 2);
        expect_bump(eng, "copy_into 推进写版本", dst, [&]
        {
            return eng.copy_into(dst, src);
        });
        nn::Tensor f16src = upload(eng, m, nn::Precision::F16);
        nn::Tensor f32dst = zeros(eng, 3, 2);
        expect_bump(eng, "cast_into 推进写版本", f32dst, [&]
        {
            return eng.cast_into(f16src, f32dst);
        });
    }
    expect_bump(eng, "zero 推进写版本", t, [&]
    {
        return eng.zero(t);
    });

    // ── [2] 只读路径不推进 ───────────────────────────────────────────────
    std::printf("[2] 只读路径不推进写版本（%s）\n", tag);
    expect_no_bump("read 不推进写版本", t, [&]
    {
        std::vector<nn::Scalar> buf(6);
        return eng.read(t, std::span<nn::Scalar>(buf));
    });
    expect_no_bump("to_matrix 不推进写版本", t, [&]
    {
        return eng.to_matrix(t, nn::Precision::F32);
    });
    expect_no_bump("clone 不推进源写版本", t, [&]
    {
        return eng.clone(t);
    });
    {
        // 合法形状：A (2,3) × B (3,2)（t/src 是 (3,2)，不足以相乘）
        nn::Tensor a = filled(eng, 2, 3, 1.0f);
        nn::Tensor b = filled(eng, 3, 2, 1.0f);
        expect_no_bump("matmul 不推进 A 操作数写版本", a, [&]
        {
            return eng.matmul(a, b);
        });
        expect_no_bump("matmul 不推进 B 操作数写版本", b, [&]
        {
            return eng.matmul(a, b);
        });
    }
    expect_no_bump("dsl::compute 不推进输入写版本", t, [&]
    {
        return nn::dsl::compute(eng, nn::dsl::leaf(t) * nn::dsl::leaf(src), 3, 2);
    });

    // ── [3] 别名可见：拷贝共享存储 ───────────────────────────────────────
    std::printf("[3] 别名（Tensor 拷贝）上的写入对原句柄可见（%s）\n", tag);
    {
        nn::Tensor alias = t;   // shared storage
        const std::uint64_t before = t.write_epoch();
        CHECK(alias.write_epoch() == before, "别名初始版本一致");
        auto r = eng.scale_inplace(alias, 0.5f);
        CHECK(r.has_value(), "经别名写入成功");
        CHECK(t.write_epoch() > before, "经别名写入后原句柄版本前进");
    }
}

// ══════════════════════════════════════════════════════════════════════════
// [4]~[7] BitLinear 量化缓存
// ══════════════════════════════════════════════════════════════════════════
nn::BitLinear make_bitlinear(nn::ComputeEngine& eng, std::size_t in, std::size_t out,
                             const nn::Matrix& w, bool packed = false)
{
    nn::BitLinear bl(in, out);
    nn::PrecisionProfile prof;
    prof.param = nn::Precision::T1_58;
    bl.set_precision_profile(prof);
    auto r = bl.init(eng);
    NN_ASSERT(r, "BitLinear init failed");
    auto wr = eng.copy_from(bl.parameters()[0].get(), w);
    NN_ASSERT(wr, "BitLinear 权重写入失败");
    // 本节测的是**缓存**（与缓冲形态正交）：默认强制 f16 稠密路径，好让
    // `quantized_weights()` 直接可读；打包路径的缓存另有 [7b] 一段。
    bl.set_ternary_pack_enabled(packed);
    return bl;
}

// (τ, wq) 的宿主快照，用于逐位比较
nn::Matrix wq_of(nn::ComputeEngine& eng, nn::BitLinear& bl)
{
    return download(eng, bl.quantized_weights());
}
nn::Matrix tau_of(nn::ComputeEngine& eng, nn::BitLinear& bl)
{
    return download(eng, bl.row_scales());
}

bool mat_same(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (std::size_t r = 0; r < a.rows(); ++r)
        for (std::size_t c = 0; c < a.cols(); ++c)
            if (a.at(r, c) != b.at(r, c)) return false;
    return true;
}

void test_bitlinear_cache(nn::ComputeEngine& eng, const char* tag)
{
    constexpr std::size_t IN = 8, OUT = 4, B = 3;
    std::printf("[4] BitLinear 量化缓存：同权重多次 forward 只量化一次（%s）\n", tag);

    nn::Matrix W(OUT, IN);
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t k = 0; k < IN; ++k)
            W.set_value(o, k, 0.37f * static_cast<float>((o + 1) * (k + 2) % 7) - 0.9f);

    nn::BitLinear bl = make_bitlinear(eng, IN, OUT, W);
    nn::Matrix X(IN, B);
    for (std::size_t k = 0; k < IN; ++k)
        for (std::size_t b = 0; b < B; ++b)
            X.set_value(k, b, 0.11f * static_cast<float>(k + b) - 0.3f);
    nn::Tensor Xt = upload(eng, X);

    auto y1 = bl.forward(Xt);
    CHECK(y1.has_value(), "首次 forward 成功");
    if (!y1.has_value()) return;
    CHECK(bl.quantize_calls() == 1, "首次 forward 量化一次");

    const nn::Matrix wq0 = wq_of(eng, bl);
    const nn::Matrix tau0 = tau_of(eng, bl);
    const nn::Matrix y0 = download(eng, *y1);

    // 同权重、同输入 → 命中缓存（且输出逐位不变）
    auto y2 = bl.forward(Xt);
    CHECK(y2.has_value(), "第二次 forward 成功");
    if (y2.has_value())
        CHECK(mat_same(y0, download(eng, *y2)), "缓存命中时输出逐位不变");
    CHECK(bl.quantize_calls() == 1, "同权重下第二次 forward 不重算 τ/wq");

    // 换输入（权重没变）→ 仍不重算
    nn::Tensor Xt2 = filled(eng, IN, B, 0.2f);
    auto y3 = bl.forward(Xt2);
    CHECK(y3.has_value(), "换输入 forward 成功");
    CHECK(bl.quantize_calls() == 1, "仅输入变化不触发重算（缓存只跟权重走）");

    // ── [5] 权重写入 → 必须重算，且与从零重算逐位相同 ──────────────────
    // 用**真实优化器**（Adam）走一遍参数更新：这就是训练循环里 latent 权重
    // 唯一的变更路径（dsl::compute_into 原地写），也是"缓存必须失效"的现场。
    std::printf("[5] 优化器/引擎写入权重后重算，且与从零重算逐位相同（%s）\n", tag);
    nn::Matrix W2(OUT, IN);
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t k = 0; k < IN; ++k)
            W2.set_value(o, k, 0.25f * static_cast<float>((o * 3 + k * 5) % 9) - 1.1f);

    {
        // 造一份假梯度（STE 梯度的形态：与 latent 同形），交给真实 Adam
        nn::Matrix G(OUT, IN);
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
                G.set_value(o, k, 0.03f * static_cast<float>(k + 1) - 0.05f);
        auto gw = eng.copy_from(bl.param_gradients()[0].get(), G);
        CHECK(gw.has_value(), "写入假梯度");
        nn::Adam opt(eng, bl.parameters(), bl.param_gradients(), /*lr=*/0.05f);
        auto st = opt.step();
        CHECK(st.has_value(), "Adam step（真实参数更新路径）");
    }
    auto y4 = bl.forward(Xt);
    CHECK(y4.has_value(), "权重变化后 forward 成功");
    CHECK(bl.quantize_calls() == 2, "优化器更新权重后重算一次");
    const nn::Matrix wq1 = wq_of(eng, bl);
    const nn::Matrix tau1 = tau_of(eng, bl);
    CHECK(!mat_same(wq0, wq1) || !mat_same(tau0, tau1),
          "权重变化后 τ/wq 至少之一变化");

    // 另一条写入路径：engine.copy_from（序列化加载 / 宿主直写）
    {
        nn::BitLinear ref = make_bitlinear(eng, IN, OUT, W2);
        auto ry = ref.forward(Xt);
        CHECK(ry.has_value(), "参考层 forward 成功");
        // 同一份权重（W2）经 copy_from 写入被测层 → 必须重算并逐位一致
        auto wr = eng.copy_from(bl.parameters()[0].get(), W2);
        CHECK(wr.has_value(), "copy_from 写入新权重");
        auto y5 = bl.forward(Xt);
        CHECK(y5.has_value(), "copy_from 后 forward 成功");
        CHECK(bl.quantize_calls() == 3, "copy_from 写入后重算一次");
        const nn::Matrix wq2 = wq_of(eng, bl);
        const nn::Matrix tau2 = tau_of(eng, bl);
        CHECK(mat_same(wq2, wq_of(eng, ref)), "缓存重算 = 从零重算（wq 逐位）");
        CHECK(mat_same(tau2, tau_of(eng, ref)), "缓存重算 = 从零重算（τ 逐位）");
        if (ry.has_value() && y5.has_value())
            CHECK(mat_same(download(eng, *ry), download(eng, *y5)),
                  "同权重同输入 → 输出逐位相同（缓存不改变数值）");
    }

    // ── [6] clear_cache 只清激活 ─────────────────────────────────────────
    std::printf("[6] clear_cache 保留权重派生物（%s）\n", tag);
    {
        const std::uint64_t calls_before = bl.quantize_calls();
        bl.clear_cache();
        auto y6 = bl.forward(Xt);
        CHECK(y6.has_value(), "clear_cache 后 forward 成功");
        CHECK(bl.quantize_calls() == calls_before,
              "clear_cache 不清 τ/wq（权重派生物，重算只是白算）");
        auto br = bl.backward(filled(eng, OUT, B, 1.0f));
        CHECK(br.has_value(), "clear_cache 后 forward → backward 正常");
    }

    // ── [7] N 次 forward（梯度累积/生成期形态）与"每次重算"等价 ──────────
    std::printf("[7] 缓存路径与逐次重算路径数值等价（%s）\n", tag);
    {
        auto sc = eng.scale_inplace(bl.parameters()[0].get(), 0.5f);
        CHECK(sc.has_value(), "再改一次权重（引擎原地缩放 → 缓存失效）");
        std::vector<nn::Matrix> outs;
        for (int i = 0; i < 4; ++i)
        {
            auto y = bl.forward(Xt);
            CHECK(y.has_value(), "累积期 forward 成功");
            if (y.has_value()) outs.push_back(download(eng, *y));
        }
        CHECK(outs.size() == 4, "拿到 4 次输出");
        for (std::size_t i = 1; i < outs.size(); ++i)
            CHECK(mat_same(outs[0], outs[i]), "同权重下多次 forward 输出逐位相同");
        CHECK(bl.quantize_calls() == 4, "该轮只量化一次（缓存命中 3 次）");
    }

    // ── [7b] 打包路径同样命中缓存（F1 × F3 的交叉）────────────────────────
    if (eng.describe().ternary_gemm)
    {
        nn::BitLinear pk = make_bitlinear(eng, IN, OUT, W, /*packed=*/true);
        CHECK(pk.uses_packed_weights(), "打包路径已启用");
        auto a = pk.forward(Xt);
        auto b = pk.forward(Xt);
        CHECK(a.has_value() && b.has_value(), "打包路径 forward ×2 成功");
        CHECK(pk.quantize_calls() == 1, "打包路径：同权重两次 forward 只量化一次");
        if (a.has_value() && b.has_value())
            CHECK(mat_same(download(eng, *a), download(eng, *b)),
                  "打包路径缓存命中时输出逐位相同");
    }
}

} // namespace

// ── [8] F2：惰性梯度缓冲（推理路径不为梯度付存储）──────────────────────────
namespace
{
void test_lazy_grads(nn::ComputeEngine& eng, const char* tag)
{
    constexpr std::size_t IN = 6, OUT = 5, B = 2;
    std::printf("[8] 惰性梯度：init/forward 后未分配，param_gradients/backward 才分配（%s）\n",
                tag);

    // Linear（f16 配置：梯度 = 权重等体积）
    {
        nn::Linear lin(IN, OUT);
        lin.set_precision_profile(nn::profile_f16());
        auto r = lin.init(eng);
        CHECK(r.has_value(), "Linear init 成功");
        CHECK(!lin.grad_buffers_allocated(), "init 后梯度缓冲未分配");
        auto y = lin.forward(filled(eng, IN, B, 0.5f));
        CHECK(y.has_value(), "Linear forward 成功");
        CHECK(!lin.grad_buffers_allocated(), "只 forward（推理）后梯度缓冲仍未分配");
        (void)lin.param_gradients();   // 优化器构造路径（nodiscard：显式丢弃）
        CHECK(lin.grad_buffers_allocated(), "param_gradients() 后梯度缓冲已分配");
    }

    // BitLinear（三值配置：梯度恒 f32，是推理足迹的 40%）
    {
        nn::Matrix W(OUT, IN);
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
                W.set_value(o, k, 0.2f * static_cast<float>((o + k) % 5) - 0.4f);
        nn::BitLinear bl = make_bitlinear(eng, IN, OUT, W);
        CHECK(!bl.grad_buffers_allocated(), "BitLinear init 后梯度缓冲未分配");
        auto y = bl.forward(filled(eng, IN, B, 0.5f));
        CHECK(y.has_value(), "BitLinear forward 成功");
        CHECK(!bl.grad_buffers_allocated(), "只 forward（推理）后梯度缓冲仍未分配");
        auto br = bl.backward(filled(eng, OUT, B, 1.0f));
        CHECK(br.has_value(), "backward 成功");
        CHECK(bl.grad_buffers_allocated(), "backward 后梯度缓冲已分配（训练路径）");
    }
}

} // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        test_write_entries(**eng_r, "GPU");
        test_bitlinear_cache(**eng_r, "GPU");
        test_lazy_grads(**eng_r, "GPU");
    }
    else
    {
        nn::CpuEngine eng;
        test_write_entries(eng, "CPU");
        test_bitlinear_cache(eng, "CPU");
        test_lazy_grads(eng, "CPU");
    }

    if (g_failures == 0)
        std::puts("write_epoch_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
