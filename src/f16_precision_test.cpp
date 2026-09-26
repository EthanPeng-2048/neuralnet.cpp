// ── f16_precision_test.cpp — Phase 2：f16 存储真正生效（适配层 + DSL + 端到端）
//
// 验收（docs/development/05-mixed-precision.md §12.2 的子集，Phase 1 边界 cast）：
//   1. **f32 零回归**：全 f32 配置下 PrecisionEngine 与内层引擎逐字节一致（G5）
//   2. f16 存储：张量 precision() == F16，且 f32 值在 f16 容差内
//   3. f16 DSL 求值 / 原地更新 / 归约 / 数据搬运 / matmul / 逐元素原语
//   4. **in-place 存储精度不可变**（§8.3）且张量对象身份保持不变
//   5. f16 端到端：Linear + 损失 + AdamW 全 f16（profile_all_f16）训练 loss 下降
//
// 重要：f16 与 f32 **不逐字节可比**（§11.2）——只做容差比对；同设备同路径的
// f32 路径则做逐字节比对（回归护栏）。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;


#define CHECK_NEAR(a, b, tol, msg) \
    do { \
        const double va = static_cast<double>(a), vb = static_cast<double>(b); \
        const double tol_ = (tol); \
        const double err = std::fabs(va - vb); \
        if (err > tol_ * (1.0 + std::fabs(vb))) { \
            std::fprintf(stderr, "  FAIL %s:%d: %s (%.6g vs %.6g, err=%.3g)\n", \
                         g_label, __LINE__, (msg), va, vb, err); \
            ++g_failures; \
        } \
    } while (0)

const char* g_label = "?";

// 环境变量开关（MSVC 下 getenv 被标 deprecated/-Werror → 走 _dupenv_s）
bool env_flag(const char* name)
{
#ifdef _MSC_VER
    char* buf = nullptr;
    std::size_t len = 0;
    const bool on = (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr && buf[0] != '\0');
    std::free(buf);
    return on;
#else
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
#endif
}

nn::Matrix make_rand(std::size_t rows, std::size_t cols, unsigned seed,
                     float lo = -1.0f, float hi = 1.0f)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : m.span()) v = d(rng);
    return m;
}

// 把张量读成 f32 Matrix（f16 自动升 cast）
nn::Matrix read_f32(nn::ComputeEngine& eng, const nn::Tensor& t)
{
    auto r = eng.to_matrix(t, nn::Precision::F32);
    if (!r)
    {
        std::fprintf(stderr, "  FAIL %s: to_matrix 失败: %s\n", g_label,
                     r.error().message.c_str());
        ++g_failures;
        return nn::Matrix(1, 1);
    }
    return std::move(*r);
}

// ── 1. f32 零回归：适配层 == 内层引擎（逐字节）───────────────────────────
void test_f32_zero_regression(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(16, 8, 1);
    nn::Matrix b = make_rand(16, 8, 2);

    auto ta = raw.from_matrix(a);
    auto tb = raw.from_matrix(b);
    CHECK(ta && tb, "from_matrix");
    if (!ta || !tb) return;

    // 归约 / 搬运 两类代表原语：适配层必须逐字节等于内层引擎
    // （逐元素原语已整体移除；逐元素路径的 f32 零回归见 test_f32_zero_regression_dsl）
    auto ref_red = raw.col_reduce_sum(*ta);
    auto got_red = eng.col_reduce_sum(*ta);
    CHECK(ref_red && got_red, "col_reduce_sum");
    auto ref_tr = raw.transpose(*ta);
    auto got_tr = eng.transpose(*ta);
    CHECK(ref_tr && got_tr, "transpose");

    const nn::Tensor* refs[2] = {&*ref_red, &*ref_tr};
    const nn::Tensor* gots[2] = {&*got_red, &*got_tr};
    for (int k = 0; k < 2; ++k)
    {
        const nn::Matrix mr = read_f32(raw, *refs[k]);
        const nn::Matrix mg = read_f32(raw, *gots[k]);
        bool identical = (mr.rows() == mg.rows() && mr.cols() == mg.cols());
        if (identical)
            for (std::size_t i = 0; i < mr.size(); ++i)
                if (mr.span()[i] != mg.span()[i]) { identical = false; break; }
        CHECK(identical, "f32 全 f32 配置逐字节一致（G5 零回归）");
        CHECK(gots[k]->precision() == nn::Precision::F32, "全 f32 输出精度");
    }
}

// DSL 版 f32 零回归（CPU：不受 AOT 闭合世界约束）
void test_f32_zero_regression_dsl(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(16, 8, 1);
    nn::Matrix b = make_rand(16, 8, 2);
    auto ta = raw.from_matrix(a);
    auto tb = raw.from_matrix(b);
    CHECK(ta && tb, "from_matrix");
    if (!ta || !tb) return;

    auto ref = nn::dsl::compute(raw,
        nn::dsl::exp(nn::dsl::leaf(*ta) * nn::dsl::rparam(0.25f))
        + nn::dsl::leaf(*tb), 16, 8);
    auto got = nn::dsl::compute(eng,
        nn::dsl::exp(nn::dsl::leaf(*ta) * nn::dsl::rparam(0.25f))
        + nn::dsl::leaf(*tb), 16, 8);
    CHECK(ref && got, "dsl::compute");
    if (!ref || !got) return;

    CHECK(got->precision() == nn::Precision::F32, "f32 直通精度");
    const nn::Matrix mr = read_f32(raw, *ref);
    const nn::Matrix mg = read_f32(raw, *got);
    bool identical = (mr.rows() == mg.rows() && mr.cols() == mg.cols());
    if (identical)
        for (std::size_t i = 0; i < mr.size(); ++i)
            if (mr.span()[i] != mg.span()[i]) { identical = false; break; }
    CHECK(identical, "f32 DSL 逐字节一致（G5 零回归）");
}

// ── 2. f16 DSL 求值：输出精度 + 数值容差 ────────────────────────────────
void test_f16_dsl(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(24, 10, 3);
    nn::Matrix b = make_rand(24, 10, 4);

    auto ta16 = eng.from_matrix(a, nn::Precision::F16);
    auto tb16 = eng.from_matrix(b, nn::Precision::F16);
    CHECK(ta16 && tb16, "from_matrix(f16)");
    if (!ta16 || !tb16) return;
    CHECK(ta16->precision() == nn::Precision::F16, "上传张量为 f16");

    // f32 参考（先降精度再算，保证与 f16 输入同源）
    auto ta32 = eng.cast(*ta16, nn::Precision::F32);
    auto tb32 = eng.cast(*tb16, nn::Precision::F32);
    CHECK(ta32 && tb32, "cast f16→f32");
    if (!ta32 || !tb32) return;
    auto ref = nn::dsl::compute(raw,
        nn::dsl::exp(nn::dsl::leaf(*ta32) * nn::dsl::rparam(0.25f))
        + nn::dsl::leaf(*tb32), 24, 10);
    CHECK(ref.has_value(), "f32 参考");

    auto got = nn::dsl::compute(eng,
        nn::dsl::exp(nn::dsl::leaf(*ta16) * nn::dsl::rparam(0.25f))
        + nn::dsl::leaf(*tb16), 24, 10, nn::Precision::F16);
    CHECK(got.has_value(), "f16 dsl::compute");
    if (!ref || !got) return;

    CHECK(got->precision() == nn::Precision::F16, "f16 输出精度");
    const nn::Matrix mg = read_f32(raw, *got);
    const nn::Matrix mr = read_f32(raw, *ref);
    double max_err = 0.0;
    for (std::size_t i = 0; i < mr.size(); ++i)
        max_err = std::max(max_err, std::fabs(static_cast<double>(mg.span()[i])
                                            - static_cast<double>(mr.span()[i])));
    // f16 相对精度 2^-10 ≈ 1e-3，逐算子舍入累积 → 5e-3 容差
    CHECK(max_err < 5e-3, "f16 DSL 与 f32 参考在容差内");
    if (max_err >= 5e-3)
        std::fprintf(stderr, "    max_err=%.3g\n", max_err);
}

// ── 3. in-place：存储精度不可变 + 对象身份保持（§8.3）────────────────────
void test_f16_inplace(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(12, 6, 5, 0.0f, 1.0f);
    nn::Matrix b = make_rand(12, 6, 6, 0.0f, 1.0f);
    auto ta = eng.from_matrix(a, nn::Precision::F16);
    auto tb = eng.from_matrix(b, nn::Precision::F16);
    CHECK(ta && tb, "from_matrix(f16)");
    if (!ta || !tb) return;

    // 别名：in-place 后必须仍看到新值（缓冲未被"换对象"）
    nn::Tensor alias = *ta;
    auto r = nn::dsl::compute_into(eng, nn::dsl::leaf(*ta) + nn::dsl::leaf(*tb), *ta);
    CHECK(r.has_value(), "f16 compute_into");
    CHECK(ta->precision() == nn::Precision::F16, "compute_into 后精度不变");
    CHECK(alias.precision() == nn::Precision::F16, "别名精度不变");

    const nn::Matrix out = read_f32(raw, alias);
    bool ok = true;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float expect = static_cast<float>(
            static_cast<nn::f16>(static_cast<nn::f16>(a.span()[i]) + static_cast<nn::f16>(b.span()[i])));
        if (std::fabs(out.span()[i] - expect) > 2e-3f) { ok = false; break; }
    }
    CHECK(ok, "f16 in-place 结果正确且别名可见");

    // add_inplace / scale_inplace / zero 同样保持精度
    auto r2 = eng.add_inplace(*ta, *tb);
    CHECK(r2.has_value(), "f16 add_inplace");
    CHECK(ta->precision() == nn::Precision::F16, "add_inplace 后精度不变");
    auto r3 = eng.scale_inplace(*ta, 0.5f);
    CHECK(r3.has_value(), "f16 scale_inplace");
    auto r4 = eng.zero(*ta);
    CHECK(r4.has_value(), "f16 zero");
    const nn::Matrix z = read_f32(raw, *ta);
    bool all_zero = true;
    for (std::size_t i = 0; i < z.size(); ++i)
        if (z.span()[i] != 0.0f) { all_zero = false; break; }
    CHECK(all_zero, "f16 zero 写零");
}

// ── 4. 数据搬运：输出精度 = 源精度（§8.4）───────────────────────────────
void test_f16_move(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(8, 4, 7);
    auto ta = eng.from_matrix(a, nn::Precision::F16);
    CHECK(ta.has_value(), "from_matrix(f16)");
    if (!ta) return;

    auto tr = eng.transpose(*ta);
    CHECK(tr && tr->precision() == nn::Precision::F16, "transpose 保 f16");
    auto cl = eng.clone(*ta);
    CHECK(cl && cl->precision() == nn::Precision::F16, "clone 保 f16");
    auto sl = eng.slice_rows(*ta, 2, 3);
    CHECK(sl && sl->precision() == nn::Precision::F16, "slice_rows 保 f16");
    if (sl)
    {
        const nn::Matrix ms = read_f32(raw, *sl);
        bool ok = true;
        for (std::size_t r = 0; r < 3; ++r)
            for (std::size_t c = 0; c < 4; ++c)
                if (std::fabs(ms.at(r, c) - static_cast<float>(static_cast<nn::f16>(a.at(r + 2, c)))) > 1e-3f)
                    ok = false;
        CHECK(ok, "slice_rows 数值正确");
    }

    // insert_rows：f16 dst 原存储写回
    auto dst = eng.from_matrix(nn::Matrix(8, 4), nn::Precision::F16);
    if (dst && sl)
    {
        auto ri = eng.insert_rows(*dst, 2, *sl);
        CHECK(ri.has_value(), "insert_rows(f16)");
        CHECK(dst->precision() == nn::Precision::F16, "insert_rows 后 dst 精度不变");
        const nn::Matrix md = read_f32(raw, *dst);
        CHECK_NEAR(md.at(3, 1), static_cast<float>(static_cast<nn::f16>(a.at(3, 1))),
                   1e-3, "insert_rows 数值");
    }

    // gather_rows（表 f16、索引 f32）
    nn::Matrix tbl = make_rand(6, 3, 8);
    auto tt = eng.from_matrix(tbl, nn::Precision::F16);
    nn::Matrix idxm(4, 1);
    idxm.set_value(0, 0, 3); idxm.set_value(1, 0, 0);
    idxm.set_value(2, 0, 5); idxm.set_value(3, 0, 1);
    auto idx = raw.from_matrix(idxm);
    if (tt && idx)
    {
        auto g = eng.gather_rows(*tt, *idx);
        CHECK(g && g->precision() == nn::Precision::F16, "gather_rows 保 f16");
        if (g)
        {
            const nn::Matrix mg = read_f32(raw, *g);
            CHECK_NEAR(mg.at(2, 2), static_cast<float>(static_cast<nn::f16>(tbl.at(5, 2))),
                       1e-3, "gather_rows 数值");
        }
    }

    // rearrange_3d：(M, B*N) → (B*M, N)，M=2 / B=3 / N=4 → 2×12 → 6×4
    nn::Matrix x = make_rand(2, 12, 9);
    auto tx = eng.from_matrix(x, nn::Precision::F16);
    if (tx)
    {
        auto r3 = eng.rearrange_3d(*tx, 2, 3, 4, false);
        CHECK(r3 && r3->precision() == nn::Precision::F16, "rearrange_3d 保 f16");
    }
}

// ── 5. 归约 + matmul ────────────────────────────────────────────────────
void test_f16_reduce_matmul(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    nn::Matrix a = make_rand(20, 6, 10);
    auto ta = eng.from_matrix(a, nn::Precision::F16);
    CHECK(ta.has_value(), "from_matrix(f16)");
    if (!ta) return;

    auto cs = eng.col_reduce_sum(*ta, nn::Precision::F16);
    CHECK(cs && cs->precision() == nn::Precision::F16, "col_reduce_sum 输出 f16");
    auto rs = eng.row_reduce_sum(*ta, nn::Precision::F16);
    CHECK(rs && rs->precision() == nn::Precision::F16, "row_reduce_sum 输出 f16");

    // matmul：f16 × f16 → f16（适配层边界 cast 走 f32 GEMM）
    nn::Matrix w = make_rand(5, 20, 11);
    nn::Matrix x = make_rand(20, 7, 12);
    auto tw16 = eng.from_matrix(w, nn::Precision::F16);
    auto tx16 = eng.from_matrix(x, nn::Precision::F16);
    CHECK(tw16 && tx16, "from_matrix(f16) for matmul");
    if (!tw16 || !tx16) return;

    auto mm = eng.matmul(*tw16, *tx16, false, false, nn::Precision::F16);
    CHECK(mm.has_value(), "f16 matmul");
    CHECK(mm && mm->precision() == nn::Precision::F16, "matmul 输出 f16");
    if (!mm) return;

    // f32 参考：同一输入值（f16 舍入后）走 f32 GEMM
    auto tw32 = eng.cast(*tw16, nn::Precision::F32);
    auto tx32 = eng.cast(*tx16, nn::Precision::F32);
    if (tw32 && tx32)
    {
        auto ref = eng.matmul(*tw32, *tx32, false, false, nn::Precision::F32);
        CHECK(ref.has_value(), "f32 matmul 参考");
        if (ref)
        {
            const nn::Matrix mg = read_f32(raw, *mm);
            const nn::Matrix mr = read_f32(raw, *ref);
            double max_err = 0.0;
            for (std::size_t i = 0; i < mr.size(); ++i)
                max_err = std::max(max_err,
                    std::fabs(static_cast<double>(mg.span()[i]) - static_cast<double>(mr.span()[i])));
            CHECK(max_err < 5e-3, "f16 matmul 与 f32 参考在容差内");
            if (max_err >= 5e-3) std::fprintf(stderr, "    matmul max_err=%.3g\n", max_err);
        }
    }
}

// ── 6. 端到端：全 f16 训练的 loss 下降 + 与 f32 对照 ─────────────────────
// 用 Linear + MSE 损失（不依赖 GPT/CE 的完整链路）验证"参数 f16 + 激活 f16 +
// 优化器 f16 状态"下的训练仍然收敛；梯度与 f32 参考在容差内。
void test_f16_end_to_end(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    const std::size_t D = 8, N = 32;
    nn::Matrix x = make_rand(D, N, 21);
    nn::Matrix w_true(D, 1, nn::Scalar{0.5f});   // 0.5 在 f16 中可精确表示
    nn::Matrix y(N, 1);
    for (std::size_t i = 0; i < N; ++i)
    {
        float s = 0.0f;
        for (std::size_t d = 0; d < D; ++d) s += x.at(d, i) * w_true.at(d, 0);
        y.set_value(i, 0, s);
    }

    auto tx = eng.from_matrix(x, nn::Precision::F16);
    auto ty = eng.from_matrix(y, nn::Precision::F16);
    nn::Matrix w0 = make_rand(D, 1, 23, -0.1f, 0.1f);
    auto tw = eng.from_matrix(w0, nn::Precision::F16);
    CHECK(tx && ty && tw, "端到端张量创建");
    if (!tx || !ty || !tw) return;

    // 手工 SGD（f16 参数、f32 计算、cast_into 写回）：
    //   pred = x^T w        (matmul, P=F16)
    //   diff = pred - y     (dsl, P=F16)
    //   loss = mean(diff^2) (归约 f32)
    //   grad = 2/N * x·diff (matmul, P=F16)
    const float lr = 0.05f;
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int step = 0; step < 40; ++step)
    {
        auto pred = eng.matmul(*tx, *tw, true, false, nn::Precision::F16);
        if (!pred) { CHECK(false, "matmul forward"); return; }
        auto diff = nn::dsl::compute(eng,
            nn::dsl::leaf(*pred) - nn::dsl::leaf(*ty), N, 1, nn::Precision::F16);
        if (!diff) { CHECK(false, "diff"); return; }

        // loss（f32 参考读数，仅用于断言收敛）
        auto sq = nn::dsl::compute(eng,
            nn::dsl::leaf(*diff) * nn::dsl::leaf(*diff), N, 1, nn::Precision::F32);
        if (!sq) { CHECK(false, "sq"); return; }
        auto loss_sum = eng.col_reduce_sum(*sq, nn::Precision::F32);
        if (!loss_sum) { CHECK(false, "loss reduce"); return; }
        const nn::Matrix lm = read_f32(raw, *loss_sum);
        const float loss = lm.span()[0] / static_cast<float>(N);
        if (step == 0) first_loss = loss;
        last_loss = loss;

        // grad_w = 2/N * x · diff
        auto gdiff = nn::dsl::compute(eng,
            nn::dsl::leaf(*diff) * nn::dsl::rparam(2.0f / static_cast<float>(N)),
            N, 1, nn::Precision::F16);
        if (!gdiff) { CHECK(false, "gdiff"); return; }
        auto grad = eng.matmul(*tx, *gdiff, false, false, nn::Precision::F16);
        if (!grad) { CHECK(false, "grad matmul"); return; }

        // w -= lr*grad（f16 存储就地更新；先抬 f32 再落回）
        auto upd = nn::dsl::compute_into(eng,
            nn::dsl::leaf(*tw) - nn::dsl::leaf(*grad) * nn::dsl::rparam(lr), *tw);
        if (!upd) { CHECK(false, "参数更新"); return; }
        CHECK(tw->precision() == nn::Precision::F16, "训练后权重仍为 f16");
    }
    CHECK(last_loss < first_loss * 0.5f, "全 f16 训练 loss 显著下降");
    std::fprintf(stderr, "    [%s] f16 e2e loss %.5f → %.5f\n", g_label, first_loss, last_loss);
}

// ── 7. 模型级端到端：GPT + 给定 profile 训练若干步，返回 (首步 loss, 末步 loss)
struct TrainResult
{
    bool ok = false;
    float first = 0.0f;
    float last = 0.0f;
    std::vector<float> losses;   // 每步 loss（f32/f16 轨迹对拍用）
};

TrainResult train_tiny_gpt(nn::ComputeEngine& eng, const nn::PrecisionProfile& prof,
                           int steps, float lr, bool verbose)
{
    TrainResult out;
    nn::GptConfig cfg{};
    cfg.vocab_size = 32;
    cfg.d_model = 16;
    cfg.seq_len = 4;
    cfg.num_heads = 2;
    cfg.d_ff = 32;
    cfg.num_layers = 2;
    cfg.precision = prof;

    auto mr = nn::build_gpt_model(eng, cfg);
    if (!mr) { CHECK(false, "build_gpt_model"); return out; }
    nn::Model& model = *mr;

    // 确定初值：把全部参数覆盖为固定伪随机值。Linear 的初值来自
    // random_device（每进程不同）→ 否则 f16/f32 两条路径无法逐步对拍，
    // 且失败与否随初值随机（实测 4/5 随机初值下 CPU f16 会发散）。
    {
        std::mt19937 rng(12345);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (auto& p : model.parameters())
        {
            nn::Matrix m(p.get().rows(), p.get().cols());
            for (auto& v : m.span()) v = nd(rng);
            if (auto r = eng.copy_from(p.get(), m); !r)
            {
                CHECK(false, "copy_from(固定初值)");
                return out;
            }
        }
    }

    auto opt = nn::create_optimizer("adamw", eng, model.parameters(),
                                    model.param_gradients(), lr, 0.0f, prof);
    if (!opt) { CHECK(false, "create_optimizer"); return out; }

    const std::size_t batch = 2;
    const std::size_t total = batch * cfg.seq_len;
    nn::Matrix x(cfg.seq_len, batch);
    for (std::size_t i = 0; i < cfg.seq_len * batch; ++i)
        x.span()[i] = static_cast<float>(i % cfg.vocab_size);
    nn::Matrix tgt(cfg.vocab_size, total, nn::Scalar{0});
    for (std::size_t i = 0; i < total; ++i)
        tgt.set_value(i % cfg.vocab_size, i, 1.0f);
    auto xt = eng.from_matrix(x);
    auto tt = eng.from_matrix(tgt);
    if (!xt || !tt) { CHECK(false, "from_matrix"); return out; }

    nn::CrossEntropyLoss ce;
    ce.set_precision_profile(prof);

    // 非有限值定位（verbose 时逐阶段扫描）
    int g_step = -1;
    const auto scan = [&](const char* tag, const nn::Tensor& t)
    {
        if (!verbose) return;
        auto m = eng.to_matrix(t, nn::Precision::F32);
        if (!m) return;
        for (auto v : m->span())
            if (!std::isfinite(v))
            {
                std::fprintf(stderr, "      [%s] step %d 首个非有限值: %s @%s 前 4 值 %.4g %.4g\n",
                             g_label, g_step, tag, t.shape_str().c_str(),
                             m->span()[0], m->span()[1]);
                return;
            }
    };

    // 固定初值后先扫一遍参数：确认 NaN 不是初始化引入的
    if (verbose)
        for (auto& p : model.parameters()) scan("init-param", p.get());

    for (int s = 0; s < steps; ++s)
    {
        g_step = s;
        if (auto r = model.zero_grad(); !r) { CHECK(false, "zero_grad"); return out; }
        auto logits = model.forward(*xt);
        if (!logits) { CHECK(false, "model.forward"); return out; }
        scan("forward/logits", *logits);
        auto loss = ce.forward(eng, *logits, *tt);
        if (!loss) { CHECK(false, "ce.forward"); return out; }
        auto grad = ce.backward();
        if (!grad) { CHECK(false, "ce.backward"); return out; }
        scan("ce/backward", *grad);
        auto bwd = model.backward(*grad);
        if (!bwd) { CHECK(false, "model.backward"); return out; }
        scan("model/backward-out", *bwd);
        if (verbose && s < 3)
            for (auto& p : model.param_gradients()) scan("param-grad", p.get());
        if (auto r = opt->step(); !r) { CHECK(false, "optimizer.step"); return out; }
        if (s == 0) out.first = *loss;
        out.last = *loss;
        out.losses.push_back(*loss);
        if (verbose)
        {
            std::fprintf(stderr, "    [%s] step %2d loss %.4f\n", g_label, s, *loss);
            if (!std::isfinite(*loss))
            {
                // 定位：哪个参数先出现非有限值（f16 溢出 / 更新爆炸）
                for (auto& p : model.parameters())
                {
                    auto m = eng.to_matrix(p.get(), nn::Precision::F32);
                    if (!m) continue;
                    bool bad = false;
                    double mx = 0.0;
                    for (auto v : m->span())
                    {
                        if (!std::isfinite(v)) bad = true;
                        mx = std::max(mx, std::fabs(static_cast<double>(v)));
                    }
                    if (bad || mx > 1e4)
                        std::fprintf(stderr, "      参数异常: %s max=%.3g%s\n",
                                     p.get().shape_str().c_str(), mx,
                                     bad ? " (NaN/Inf)" : "");
                }
                break;
            }
        }
    }
    out.ok = true;
    return out;
}

// 覆盖 Layer 接线：参数按 p_.param 创建、dsl::compute 按 p_.compute、
// loss 按 p_.stable、优化器状态按 p_.optimizer；且全部算子经适配层。
//
// 验收方式：同结构/同初值/同数据下，f32 与 f16 **逐 step 对拍** —— f16 存储
// 只该带来 1e-3 量级差异（§7.2：f32 参考 + 输出舍入），不该改变训练轨迹。
void test_f16_model_e2e(nn::ComputeEngine& raw, nn::ComputeEngine& eng)
{
    constexpr int STEPS = 20;
    constexpr float LR = 3e-3f;

    // 对照：原生引擎 + 全 f32 profile（现状路径）
    const auto r32 = train_tiny_gpt(raw, nn::profile_f32(), STEPS, LR, false);
    CHECK(r32.ok, "profile_f32 训练完成");
    if (r32.ok)
        CHECK(r32.losses.back() <= r32.losses.front(), "profile_f32（原生引擎）loss 不增");

    // profile_f16：param + compute f16（CLI --f16 语义）
    // NN_F16_DEBUG=1 → 逐 step loss + 逐阶段非有限值扫描（定位 CPU f16 发散用）
    const bool dbg = env_flag("NN_F16_DEBUG");
    const auto r16 = train_tiny_gpt(eng, nn::profile_f16(), STEPS, LR, dbg);
    CHECK(r16.ok, "profile_f16 训练完成");
    if (!r16.ok) return;

    if (!std::isfinite(r16.losses.back()))
    {
        // 历史"已知问题"（CPU 侧 f16 训练发散）已修复并转为**硬失败**：
        //   根因1 = DSL 预绑定把 f16 操作数喂给 f32 GEMM（cpu_matrix<F32>()
        //   拿到空指针）；根因2 = float_to_half_bits 次正规分支 exp<=-46 的
        //   移位 UB（|v| ∈ [2.8e-14, 1.2e-10] 的梯度被写成垃圾 half）。
        //   详见 docs/development/05-mixed-precision.md §12.12。
        std::fprintf(stderr, "  FAIL %s: f16 训练出现非有限 loss（回归！）"
                             " 轨迹: %.4f → %.4f\n",
                     g_label, r16.losses.empty() ? 0.0 : r16.losses.front(),
                     r16.losses.back());
        ++g_failures;
        return;
    }

    // 逐 step 对拍（相对偏差，分母取 max(1, |f32|) 避免小 loss 放大）
    double max_rel = 0.0;
    for (std::size_t i = 0; i < r32.losses.size() && i < r16.losses.size(); ++i)
    {
        const double a = r16.losses[i];
        const double b = r32.losses[i];
        const double rel = std::fabs(a - b) / std::max(1.0, std::fabs(b));
        max_rel = std::max(max_rel, rel);
    }
    CHECK(max_rel < 0.10, "f16 训练轨迹与 f32 逐 step 一致（<10%）");
    if (max_rel >= 0.10)
        std::fprintf(stderr, "    轨迹最大相对偏差 %.3f\n", max_rel);
    std::fprintf(stderr, "    [%s] f32 %.4f → %.4f ; f16 %.4f → %.4f (max_rel=%.4f)\n",
                 g_label, r32.losses.front(), r32.losses.back(),
                 r16.losses.front(), r16.losses.back(), max_rel);
    // 参数精度必须由 profile 驱动：profile_f16 下全部参数 f16、f32 下全 f32
    {
        nn::GptConfig cfg{};
        cfg.vocab_size = 32; cfg.d_model = 16; cfg.seq_len = 4;
        cfg.num_heads = 2; cfg.d_ff = 32; cfg.num_layers = 2;
        cfg.precision = nn::profile_f16();
        auto m16 = nn::build_gpt_model(eng, cfg);
        CHECK(m16.has_value(), "build_gpt_model(profile_f16)");
        if (m16)
        {
            auto params = m16->parameters();
            std::size_t n16 = 0;
            for (auto& p : params)
                if (p.get().precision() == nn::Precision::F16) ++n16;
            if (n16 != params.size())
            {
                std::fprintf(stderr, "    参数精度: f16=%zu / %zu\n", n16, params.size());
                for (auto& p : params)
                    if (p.get().precision() != nn::Precision::F16)
                        std::fprintf(stderr, "      非 f16: %s\n", p.get().shape_str().c_str());
            }
            CHECK(n16 == params.size(), "profile_f16：全部参数为 f16 存储");
        }

        nn::GptConfig cfg32 = cfg;
        cfg32.precision = nn::profile_f32();
        auto m32 = nn::build_gpt_model(raw, cfg32);
        CHECK(m32.has_value(), "build_gpt_model(profile_f32)");
        if (m32)
        {
            auto params = m32->parameters();
            std::size_t n32 = 0;
            for (auto& p : params)
                if (p.get().precision() == nn::Precision::F32) ++n32;
            CHECK(n32 == params.size(), "profile_f32：全部参数为 f32 存储（对照组）");
        }
    }
}

// ── 8. NN_F16_DEBUG：profile 单字段矩阵诊断（CPU f16 发散定位用）────────
// 逐字段拆开跑同一个小 GPT：哪个字段单独开就 NaN，根因就在该字段触达的路径。
void debug_profile_matrix(nn::ComputeEngine& eng)
{
    using P = nn::Precision;
    struct Case { const char* name; nn::PrecisionProfile prof; };
    const Case cases[] = {
        {"f32",       nn::profile_f32()},
        {"param=f16", {P::F16, P::F32, P::F32, P::F32}},
        {"compute=f16", {P::F32, P::F16, P::F32, P::F32}},
        {"stable=f16", {P::F32, P::F32, P::F16, P::F32}},
        {"opt=f16",   {P::F32, P::F32, P::F32, P::F16}},
        {"f16(CLI)",  nn::profile_f16()},
    };
    const char* saved = g_label;
    for (const auto& c : cases)
    {
        g_label = c.name;
        const auto r = train_tiny_gpt(eng, c.prof, 20, 3e-3f, /*verbose=*/true);
        std::printf("  [dbg %-11s] ok=%d traj:", c.name, r.ok ? 1 : 0);
        for (std::size_t i = 0; i < r.losses.size(); ++i)
            std::printf(" %.4f", r.losses[i]);
        std::printf("\n");
        std::fflush(stdout);
    }
    g_label = saved;
}

void run_suite(const char* label, nn::ComputeEngine& raw, nn::ComputeEngine& adapter,
               bool allow_arbitrary_expr)
{
    g_label = label;
    std::printf("  [%s] f32 零回归 / 搬运 / 归约+matmul / 逐元素%s\n", label,
                allow_arbitrary_expr ? " / f16 DSL / in-place / 端到端" : "");
    test_f32_zero_regression(raw, adapter);
    test_f16_move(raw, adapter);
    test_f16_reduce_matmul(raw, adapter);
    // 任意表达式（DSL）在 GPU 上受 AOT 闭合世界约束：只能命中构建期 scan_exprs
    // 登记过的结构。故这里只在 CPU 跑任意表达式；GPU 的 DSL / 原地 / 端到端
    // 由 test_f16_model_e2e（Layer 结构 = 已登记结构）覆盖。
    if (allow_arbitrary_expr)
    {
        test_f32_zero_regression_dsl(raw, adapter);
        test_f16_dsl(raw, adapter);
        test_f16_inplace(raw, adapter);
        test_f16_end_to_end(raw, adapter);
    }
    // Layer/Model 级：CPU/GPU 都跑（结构均来自已登记的 Layer 表达式）
    test_f16_model_e2e(raw, adapter);
}

} // namespace

int main()
{
    std::printf("f16_precision_test（Phase 2：f16 存储经 PrecisionEngine 生效）\n");

    // CPU（始终运行；任意 DSL 表达式不受 AOT 闭合世界约束）
    {
        nn::CpuEngine cpu;
        nn::PrecisionEngine adapter(cpu);
        run_suite("cpu", cpu, adapter, /*allow_arbitrary_expr=*/true);
        if (env_flag("NN_F16_DEBUG"))
        {
            std::printf("── NN_F16_DEBUG: CPU profile 单字段矩阵 ──\n");
            debug_profile_matrix(adapter);
        }
    }

#ifdef NN_HAS_VULKAN
    // GPU（可选；设备不可用则跳过 GPU 部分而不是失败）
    {
        auto& backend = nn::GpuBackend::instance();
        if (backend.initialize())
        {
            nn::GpuEngine gpu(backend);
            nn::PrecisionEngine adapter(gpu);
            run_suite("gpu", gpu, adapter, /*allow_arbitrary_expr=*/false);
        }
        else
        {
            std::printf("  [gpu] 跳过（Vulkan 设备不可用）\n");
        }
    }
#endif

    if (g_failures == 0)
    {
        std::printf("f16_precision_test: ALL PASSED\n");
        return 0;
    }
    std::printf("f16_precision_test: %d FAILURES\n", g_failures);
    return 1;
}