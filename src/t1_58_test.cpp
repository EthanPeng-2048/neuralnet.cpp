// ── t1_58_test — 三值（T1_58）精度与 BitLinear ──────────────────────────────
// 设计依据：docs/development/21-quantized-weights.md（P1 验收 §5/§7）
//
// 覆盖：
//   [1] PrecisionSet / PrecisionSupport 语义（默认集合 = {f16,f32} × 四槽）
//   [2] 能力校验**反例**：普通 Linear + param=T1_58 → 初始化期报错
//       （错误信息必须含 层名 + 槽位 + 取值），且是 Result 错误而非终止
//   [3] 能力校验**正例**：默认 / f16 / master-weights 配置全部照常通过
//   [4] Model::add 路径：default_precision_profile(param=T1_58) + add<Linear> → 报错
//   [5] D3(b)：init 之后 set_precision_profile → fail-fast（子进程断言非零退出）
//   [6] BitLinear：量化输出逐位符合 τ 规则（absmean + per-row）
//   [7] BitLinear：latent 权重的 STE 梯度 = 解析式（数值梯度只对 latent 做）
//   [8] BitLinear MLP 端到端小训练（CPU；`--gpu` 时在 GPU 上跑）
//
// 子进程模式：--expect-post-init-abort（父进程 std::system 自举）
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

std::string g_self;   // argv[0]：子进程自举用

// ── 便捷：随机矩阵 → 引擎张量（均匀分布 [lo,hi)）────────────────────────────
[[maybe_unused]] nn::Tensor upload_random(nn::ComputeEngine& eng, std::size_t rows,
                                          std::size_t cols, float lo, float hi, unsigned seed)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : m.span()) v = d(rng);
    return upload(eng, m);
}

// ── [1] PrecisionSet / PrecisionSupport ──────────────────────────────────
void test_precision_set()
{
    std::puts("[1] PrecisionSet / PrecisionSupport 语义");

    const nn::PrecisionSet r = nn::rounding_precision_set();
    CHECK(r.has(nn::Precision::F16), "默认集合含 f16");
    CHECK(r.has(nn::Precision::F32), "默认集合含 f32");
    CHECK(!r.has(nn::Precision::T1_58), "默认集合不含 t1_58");
    CHECK(r.to_string() == "f16|f32", "to_string() 按枚举序罗列");

    const nn::PrecisionSet t = nn::PrecisionSet::of({nn::Precision::T1_58});
    CHECK(t.has(nn::Precision::T1_58), "显式集合含 t1_58");
    CHECK(!t.has(nn::Precision::F32), "显式集合不含 f32");
    CHECK(t.to_string() == "t1_58", "t1_58 集合名");

    // 默认能力 = 现状（既有层零改动）
    nn::CpuEngine eng;
    nn::Linear probe(4, 3);
    const nn::PrecisionSupport sup = probe.precision_support();
    CHECK(sup.param.mask == r.mask && sup.compute.mask == r.mask &&
          sup.stable.mask == r.mask && sup.optimizer.mask == r.mask,
          "默认 precision_support() = {f16,f32} × 四槽");
    CHECK(std::string(probe.layer_name()) == "Linear", "Linear::layer_name()");
    (void)eng;
}

// ── [2] 能力校验反例（构建/初始化期报错）─────────────────────────────────
nn::Result<void> add_linear_with_profile(nn::ComputeEngine& eng, const nn::PrecisionProfile& p)
{
    nn::Model model(eng);
    model.set_default_precision_profile(p);
    return model.add<nn::Linear>(4, 3);
}

void test_capability_reject()
{
    std::puts("[2] 能力校验反例：普通 Linear 不支持 param=T1_58");
    nn::CpuEngine eng;

    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;

    // (a) 手工构造 + init
    {
        nn::Linear ln(4, 3);
        ln.set_precision_profile(t1);
        auto r = ln.init(eng);
        CHECK(!r.has_value(), "Linear + param=T1_58 → init 失败");
        if (!r.has_value())
        {
            const std::string& m = r.error().message;
            CHECK(m.find("Linear") != std::string::npos, "错误含层名（Linear）");
            CHECK(m.find("param") != std::string::npos, "错误含槽位（param）");
            CHECK(m.find("t1_58") != std::string::npos, "错误含取值（t1_58）");
            CHECK(m.find("f16|f32") != std::string::npos, "错误含该槽允许集合");
        }
    }
    // (b) Model::add 路径（工厂/容器路径同样在 init 期拦下）
    {
        auto r = add_linear_with_profile(eng, t1);
        CHECK(!r.has_value(), "Model::add<Linear> + param=T1_58 → 失败");
    }
    // (c) compute 槽不允许 T1_58（激活被三值化必然崩）
    {
        nn::PrecisionProfile p;
        p.compute = nn::Precision::T1_58;
        nn::Linear ln(4, 3);
        ln.set_precision_profile(p);
        auto r = ln.init(eng);
        CHECK(!r.has_value(), "Linear + compute=T1_58 → init 失败");
        if (!r.has_value())
            CHECK(r.error().message.find("compute") != std::string::npos, "错误含槽位（compute）");
    }
}

// ── [3] 能力校验正例（既有配置零回归）───────────────────────────────────
void test_capability_accept()
{
    std::puts("[3] 能力校验正例：既有配置全部照常通过");
    nn::CpuEngine eng;

    const nn::PrecisionProfile cases[] = {
        nn::profile_f32(),
        nn::profile_f16(),
        nn::profile_master_weights(),
    };
    for (const auto& p : cases)
    {
        nn::Linear ln(4, 3);
        ln.set_precision_profile(p);
        auto r = ln.init(eng);
        CHECK(r.has_value(), "既有 profile 通过能力校验");
    }
    // 默认（未注入 profile）= 全 f32，同样通过
    {
        nn::Linear ln(4, 3);
        auto r = ln.init(eng);
        CHECK(r.has_value(), "未注入 profile（默认全 f32）通过");
    }
}

// ── [4] D3(b)：init 后 profile 不可变 ────────────────────────────────────
void test_post_init_immutable()
{
    std::puts("[4] D3(b)：init 之后 set_precision_profile → fail-fast");
    nn::CpuEngine eng;
    nn::Linear ln(4, 3);
    auto r = ln.init(eng);
    CHECK(r.has_value(), "init 成功");
    // 子进程断言：这里的调用会终止进程（不能在本进程里做）
    const std::string cmd = "\"" + g_self + "\" --expect-post-init-abort";
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return;
    }
#ifndef _WIN32
    const int code = WIFEXITED(rc) ? WEXITSTATUS(rc) : (WIFSIGNALED(rc) ? 128 + WTERMSIG(rc) : rc);
#else
    const int code = rc;
#endif
    CHECK(code != 0, "init 后 set_precision_profile → 非零退出（abort）");
}

// ── [5][6] BitLinear：τ / wq 规则（逐位）+ 去量化 forward + STE backward ────
// 参考值全部在宿主上用双精度独立算一遍（不复用被测实现）。
// 已知 latent 权重：三行的结构各不相同
//   行0 {1.0, 0.3, -0.3, 0.3}  → absmean τ=0.475（absmax τ=1.0 会把 0.3 全归零）
//   行1 {0.4,-0.4, 0.1,-0.1}   → τ=0.25，阈值 0.125 → 两个 0.1 归零
//   行2 {0.7, 0.6, 0.5, 0.1}   → τ=0.475 → 只有 0.1 归零
constexpr std::size_t kIN = 4, kOUT = 3, kB = 2;
constexpr float kRawW[kOUT][kIN] = {
    {1.0f, 0.3f, -0.3f, 0.3f},
    {0.4f, -0.4f, 0.1f, -0.1f},
    {0.7f, 0.6f, 0.5f, 0.1f},
};
constexpr float kX[kIN][kB] = {{1.f, -2.f}, {0.5f, 1.f}, {-1.f, 0.25f}, {2.f, -0.5f}};

nn::Matrix known_weight_matrix()
{
    nn::Matrix W(kOUT, kIN);
    for (std::size_t o = 0; o < kOUT; ++o)
        for (std::size_t k = 0; k < kIN; ++k)
            W.set_value(o, k, kRawW[o][k]);
    return W;
}

nn::Matrix known_input_matrix()
{
    nn::Matrix X(kIN, kB);
    for (std::size_t k = 0; k < kIN; ++k)
        for (std::size_t b = 0; b < kB; ++b)
            X.set_value(k, b, kX[k][b]);
    return X;
}

// 宿主参考：τ（逐行 absmean）与 wq（RoundClip(W/τ,-1,1)）
void ref_quant(double (&tau_ref)[kOUT], double (&wq_ref)[kOUT][kIN])
{
    for (std::size_t o = 0; o < kOUT; ++o)
    {
        double s = 0.0;
        for (std::size_t k = 0; k < kIN; ++k) s += std::fabs(static_cast<double>(kRawW[o][k]));
        tau_ref[o] = s / static_cast<double>(kIN);
        for (std::size_t k = 0; k < kIN; ++k)
        {
            const double w = static_cast<double>(kRawW[o][k]);
            wq_ref[o][k] = (w > 0.5 * tau_ref[o]) ? 1.0
                         : (w < -0.5 * tau_ref[o]) ? -1.0 : 0.0;
        }
    }
}

void test_bitlinear_rules()
{
    std::puts("[5] BitLinear：τ = 逐行 absmean，wq 逐位符合 RoundClip(W/τ,-1,1)");
    nn::CpuEngine eng;
    constexpr std::size_t IN = kIN, OUT = kOUT, B = kB;

    nn::BitLinear bl(IN, OUT);
    nn::PrecisionProfile prof;
    prof.param = nn::Precision::T1_58;
    bl.set_precision_profile(prof);
    auto init_r = bl.init(eng);
    CHECK(init_r.has_value(), "BitLinear + param=T1_58 → init 成功");
    if (!init_r.has_value()) return;

    const nn::Matrix W = known_weight_matrix();
    {
        auto wr = eng.copy_from(bl.parameters()[0].get(), W);
        CHECK(wr.has_value(), "写入已知 latent 权重");
    }

    // 输入
    const nn::Matrix X = known_input_matrix();
    const nn::Tensor Xt = upload(eng, X);

    auto y_r = bl.forward(Xt);
    CHECK(y_r.has_value(), "BitLinear forward 成功");
    if (!y_r.has_value()) return;

    // ── 宿主参考：τ / wq ───────────────────────────────────────────────
    double tau_ref[OUT] = {0, 0, 0};
    double wq_ref[OUT][IN] = {};
    ref_quant(tau_ref, wq_ref);
    {
        const nn::Matrix tau = download(eng, bl.row_scales());
        CHECK(tau.rows() == OUT && tau.cols() == 1, "τ 形状 (out,1)");
        for (std::size_t o = 0; o < OUT; ++o)
            CHECK(std::fabs(static_cast<double>(tau.at(o, 0)) - tau_ref[o]) < 1e-6,
                  "τ = 逐行 absmean（absmax 不是尺度）");
    }
    {
        const nn::Matrix wq = download(eng, bl.quantized_weights());
        bool exact = true, in_set = true;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
            {
                const double q = static_cast<double>(wq.at(o, k));
                if (q != wq_ref[o][k]) exact = false;
                if (!(q == 1.0 || q == 0.0 || q == -1.0)) in_set = false;
            }
        CHECK(exact, "wq 逐位等于 RoundClip(W/τ,-1,1)");
        CHECK(in_set, "wq 取值 ⊂ {-1,0,+1}（f16 精确可表示）");
        // 行0 的三值码位：absmean 保留 0.3（absmax 会把它们全归零）
        CHECK(wq.at(0, 1) == 1.0f && wq.at(0, 2) == -1.0f,
              "absmean 尺度下 0.3 不被归零（与 absmax 的关键差别）");
    }
    // ── forward：Y = (wq·X)∘τ + b（b 初值 0）────────────────────────────
    {
        const nn::Matrix Y = download(eng, *y_r);
        double ref[OUT][B] = {};
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t b = 0; b < B; ++b)
            {
                double acc = 0.0;
                for (std::size_t k = 0; k < IN; ++k)
                    acc += wq_ref[o][k] * static_cast<double>(kX[k][b]);
                ref[o][b] = acc * tau_ref[o];
            }
        double mx = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t b = 0; b < B; ++b)
                mx = std::max(mx, std::fabs(static_cast<double>(Y.at(o, b)) - ref[o][b]));
        CHECK(mx < 1e-5, "forward = (wq·X)∘τ（去量化点积）");
    }

    // ── STE backward：dX = wqᵀ(dY∘τ)；dW = (dY·Xᵀ)∘τ；db = ΣdY ──────────
    std::puts("[6] BitLinear：STE 梯度 = 解析式（量化器导数当恒等）");
    {
        auto zr = bl.zero_grad();
        CHECK(zr.has_value(), "zero_grad");
        nn::Matrix dY(OUT, B);
        const float dv[OUT][B] = {{0.5f, -0.25f}, {1.0f, 0.75f}, {-0.5f, 0.25f}};
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t b = 0; b < B; ++b)
                dY.set_value(o, b, dv[o][b]);
        const nn::Tensor dYt = upload(eng, dY);

        auto gi_r = bl.backward(dYt);
        CHECK(gi_r.has_value(), "BitLinear backward 成功");
        if (!gi_r.has_value()) return;

        // dW[o,k] = τ_o · Σ_b dY[o,b]·X[k,b]
        double dw_ref[OUT][IN] = {};
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
            {
                double acc = 0.0;
                for (std::size_t b = 0; b < B; ++b)
                    acc += static_cast<double>(dv[o][b]) * static_cast<double>(kX[k][b]);
                dw_ref[o][k] = acc * tau_ref[o];
            }
        const nn::Matrix gw = download(eng, bl.param_gradients()[0].get());
        double mxw = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t k = 0; k < IN; ++k)
                mxw = std::max(mxw, std::fabs(static_cast<double>(gw.at(o, k)) - dw_ref[o][k]));
        CHECK(mxw < 1e-5, "dW_lat = (dY·Xᵀ)∘τ（STE 含去量化尺度）");

        // db[o] = Σ_b dY[o,b]
        const nn::Matrix gb = download(eng, bl.param_gradients()[1].get());
        double mxb = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
        {
            double s = 0.0;
            for (std::size_t b = 0; b < B; ++b) s += static_cast<double>(dv[o][b]);
            mxb = std::max(mxb, std::fabs(static_cast<double>(gb.at(o, 0)) - s));
        }
        CHECK(mxb < 1e-5, "db = Σ_batch dY");

        // dX[k,b] = Σ_o wq[o,k]·τ_o·dY[o,b]
        const nn::Matrix gi = download(eng, *gi_r);
        double mxi = 0.0;
        for (std::size_t k = 0; k < IN; ++k)
            for (std::size_t b = 0; b < B; ++b)
            {
                double acc = 0.0;
                for (std::size_t o = 0; o < OUT; ++o)
                    acc += wq_ref[o][k] * tau_ref[o] * static_cast<double>(dv[o][b]);
                mxi = std::max(mxi, std::fabs(static_cast<double>(gi.at(k, b)) - acc));
            }
        CHECK(mxi < 1e-5, "dX = wqᵀ(dY∘τ)（τ 在求和维内，不能提到 matmul 外）");
    }
}

// ── [7] 端到端：T1_58 MLP 小训练收敛（CPU / GPU 同一份代码）──────────────
// ── [7a] compute=F16：混合精度槽（param 三值 + 激活 f16）───────────────
// 三值只占 param 槽；激活/稳定链仍是 f16/f32 → 同一层在 compute=f16 下
// 必须给出与 f32 路径**同一数学结果**（差异仅 f16 舍入）。
void test_bitlinear_f16_compute()
{
    std::puts("[7a] BitLinear：compute=F16（param 三值 + 激活 f16）与 f32 一致");
    nn::CpuEngine eng;
    constexpr std::size_t IN = kIN, OUT = kOUT, B = kB;

    nn::PrecisionProfile prof;
    prof.param     = nn::Precision::T1_58;
    prof.compute   = nn::Precision::F16;
    prof.stable    = nn::Precision::F32;
    prof.optimizer = nn::Precision::F32;
    nn::BitLinear bl(IN, OUT);
    bl.set_precision_profile(prof);
    auto init_r = bl.init(eng);
    CHECK(init_r.has_value(), "BitLinear + compute=F16 → init 成功（能力校验只锁 param 槽）");
    if (!init_r.has_value()) return;

    const nn::Matrix W = known_weight_matrix();
    auto wr = eng.copy_from(bl.parameters()[0].get(), W);
    CHECK(wr.has_value(), "写入已知 latent 权重");

    const nn::Matrix X = known_input_matrix();
    auto y_r = bl.forward(upload(eng, X));
    CHECK(y_r.has_value(), "f16 compute forward 成功");
    if (!y_r.has_value()) return;

    double tau_ref[OUT] = {0, 0, 0};
    double wq_ref[OUT][IN] = {};
    ref_quant(tau_ref, wq_ref);
    const nn::Matrix Y = download(eng, *y_r);
    double mx = 0.0;
    for (std::size_t o = 0; o < OUT; ++o)
        for (std::size_t b = 0; b < B; ++b)
        {
            double acc = 0.0;
            for (std::size_t k = 0; k < IN; ++k)
                acc += wq_ref[o][k] * static_cast<double>(kX[k][b]);
            mx = std::max(mx, std::fabs(static_cast<double>(Y.at(o, b)) - acc * tau_ref[o]));
        }
    CHECK(mx < 2e-3, "f16 compute 下结果与 f32 参考一致（f16 舍入内）");
    // 量化缓冲恒 f16（与 compute 槽无关）
    CHECK(bl.quantized_weights().precision() == nn::Precision::F16, "wq 缓冲恒 f16");
    CHECK(bl.latent_weights().precision() == nn::Precision::F32, "latent 恒 f32（STE 落点）");
}

// ── [8] 规格/序列化往返：weight_quant 决定重建 BitLinear ─────────────────
// 保存 → peek → 用规格重建 → 加载 → 同一输入的前向逐位一致；且该前向必须等于
// "三值量化"的宿主参考（这就是"重建出来的是 BitLinear 而非 Linear"的判据）。
void test_spec_roundtrip()
{
    std::puts("[8] ModelSpec.weight_quant 往返（save → peek → 重建 → load → 前向一致）");
    constexpr std::size_t IN = kIN, OUT = kOUT, B = kB;
    const std::string file = "t1_58_spec_roundtrip.bin";
    const nn::Matrix W = known_weight_matrix();
    const nn::Matrix X = known_input_matrix();

    nn::ModelSpec spec;
    spec.type         = nn::ModelType::MLP;
    spec.layer_dims   = {IN, OUT};
    spec.norm_type    = nn::NormType::LayerNorm;
    spec.weight_quant = nn::WeightQuant::T1_58;

    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;

    nn::Matrix Y_ref;
    {   // 建模型（from_spec：与 mnist_train 生产路径一致）→ 写已知权重 → 保存
        nn::CpuEngine eng;
        auto m = nn::build_mnist_model_from_spec(eng, spec, t1);
        CHECK(m.has_value(), "from_spec 构建 T1_58 单层模型");
        if (!m.has_value())
        {
            std::printf("    构建失败: %s\n", m.error().message.c_str());
            return;
        }
        CHECK(m->num_layers() == 1, "单层模型（{IN,OUT} → 一个 BitLinear）");
        auto wr = eng.copy_from(m->parameters()[0].get(), W);
        CHECK(wr.has_value(), "写入已知 latent 权重");

        auto y = m->forward(upload(eng, X));
        CHECK(y.has_value(), "forward 成功");
        if (!y.has_value()) return;
        Y_ref = download(eng, *y);

        double tau[kOUT] = {0, 0, 0};
        double wq[kOUT][kIN] = {};
        ref_quant(tau, wq);
        double mx = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t b = 0; b < B; ++b)
            {
                double acc = 0.0;
                for (std::size_t k = 0; k < IN; ++k)
                    acc += wq[o][k] * static_cast<double>(kX[k][b]);
                mx = std::max(mx, std::fabs(static_cast<double>(Y_ref.at(o, b)) - acc * tau[o]));
            }
        CHECK(mx < 1e-5, "该层前向 = 三值量化参考 ⇒ 确实是 BitLinear（不是 Linear）");

        auto sr = nn::save_model(file, *m, spec);
        CHECK(sr.has_value(), "save_model");
        if (!sr.has_value()) std::printf("    save 失败: %s\n", sr.error().message.c_str());
    }

    {   // peek：规格必须记住三值（缺键 = None → 旧文件零破坏）
        auto peeked = nn::peek_model_spec(file);
        CHECK(peeked.has_value(), "peek_model_spec");
        if (peeked.has_value())
        {
            CHECK(peeked->weight_quant == nn::WeightQuant::T1_58, "规格记住 weight_quant=t1_58");
            CHECK(peeked->layer_dims == spec.layer_dims, "layer_dims 往返一致");
            CHECK(nn::spec_matches(*peeked, spec), "spec_matches 含 weight_quant 比对");
            CHECK(nn::spec_summary(*peeked).find("wq=t1_58") != std::string::npos,
                  "spec_summary 披露 wq=t1_58");
        }
        // 反例：把同一文件加载进"普通 MLP"（weight_quant=None）必须被拒
        nn::CpuEngine eng;
        nn::ModelSpec plain = spec;
        plain.weight_quant = nn::WeightQuant::None;
        auto plain_model = nn::build_mnist_model_from_spec(eng, plain, nn::profile_f32());
        CHECK(plain_model.has_value(), "普通 MLP 构建成功");
        if (plain_model.has_value())
        {
            auto lr = nn::load_model(file, *plain_model);
            CHECK(!lr.has_value(), "三值权重**不能**加载进普通 Linear 模型（架构不匹配）");
        }
    }

    {   // 重建 + 加载：同一输入 → 与前向参考一致
        nn::CpuEngine eng;
        auto m2 = nn::build_mnist_model_from_spec(eng, spec, t1);
        CHECK(m2.has_value(), "按规格重建成功");
        if (!m2.has_value()) return;
        auto lr = nn::load_model(file, *m2);
        CHECK(lr.has_value(), "load_model 成功（参数条数/形状一致）");
        if (!lr.has_value())
        {
            std::printf("    load 失败: %s\n", lr.error().message.c_str());
            return;
        }
        auto y2 = m2->forward(upload(eng, X));
        CHECK(y2.has_value(), "重建模型 forward 成功");
        if (!y2.has_value()) return;
        const nn::Matrix Y2 = download(eng, *y2);
        double mx = 0.0;
        for (std::size_t o = 0; o < OUT; ++o)
            for (std::size_t b = 0; b < B; ++b)
                mx = std::max(mx, std::fabs(static_cast<double>(Y2.at(o, b))
                                            - static_cast<double>(Y_ref.at(o, b))));
        CHECK(mx == 0.0, "重建 + 加载后前向与保存前**逐位一致**");
    }
    std::remove(file.c_str());
}

bool test_training(nn::ComputeEngine& eng, const char* dev)
{
    std::printf("[7] BitLinear MLP 端到端小训练（%s）：loss 收敛\n", dev);
    constexpr std::size_t IN = 8, HID = 16, OUT = 4, B = 32;

    // 教师：三值权重线性映射 + 固定偏置（学生是"BitLinear → Norm → 激活 → BitLinear"）
    std::mt19937 rng(20261010u);
    std::uniform_int_distribution<int> td(0, 2);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    nn::Matrix Wt(OUT, IN);
    for (auto& v : Wt.span())
        v = static_cast<float>(td(rng) - 1) * 0.5f;
    nn::Matrix Xm(IN, B);
    for (auto& v : Xm.span()) v = ud(rng);
    nn::Matrix Ym(OUT, B);
    Wt.multiply_to(Ym, Xm);   // Ym = Wt·Xm（教师权重**逐元素固定**，不随列变化）

    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;
    auto m_r = nn::build_mnist_mlp_model(eng, {IN, HID, OUT}, nn::NormType::LayerNorm, t1);
    CHECK(m_r.has_value(), "T1_58 MLP 构建成功（线性层 = BitLinear）");
    if (!m_r.has_value())
    {
        std::printf("    构建失败: %s\n", m_r.error().message.c_str());
        return false;
    }
    nn::Model& model = *m_r;

    auto opt = nn::create_optimizer("adam", eng, model.parameters(), model.param_gradients(),
                                    /*lr=*/0.05f, /*wd=*/0.0f, t1);
    CHECK(opt != nullptr, "优化器创建成功");
    if (!opt) return false;

    nn::MSELoss mse;
    mse.set_precision_profile(t1);
    const nn::Tensor Xt = upload(eng, Xm);
    const nn::Tensor Yt = upload(eng, Ym);

    float first = -1.f, last = -1.f;
    constexpr int STEPS = 300;
    // 失败时打印引擎错误原文（GPU 上多为"闭合世界未命中/精度签名不支持"）
    const auto report = [](const char* what, const std::string& msg)
    {
        std::printf("    [FAIL] %s: %s\n", what, msg.c_str());
        CHECK(false, what);
    };
    for (int step = 0; step < STEPS; ++step)
    {
        if (auto r = model.zero_grad(); !r) { report("zero_grad", r.error().message); return false; }
        auto pred = model.forward(Xt);
        if (!pred) { report("forward", pred.error().message); return false; }
        auto loss = mse.forward(eng, *pred, Yt);
        if (!loss) { report("loss", loss.error().message); return false; }
        if (!std::isfinite(*loss)) { CHECK(false, "loss 非有限"); return false; }
        if (step == 0) first = *loss;
        last = *loss;
        auto grad = mse.backward();
        if (!grad) { report("loss.backward", "no grad"); return false; }
        if (auto r = model.backward(*grad); !r) { report("model.backward", r.error().message); return false; }
        if (auto r = opt->step(); !r) { report("optimizer.step", r.error().message); return false; }
    }
    std::printf("    loss: %.6f → %.6f（%d 步）\n", first, last, STEPS);
    CHECK(last < first * 0.7f, "loss 显著下降（量化权重仍可训练）");
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    // ── 子进程自举：init 后注入 profile → 期望终止 ────────────────────────
    if (argc >= 2 && std::string(argv[1]) == "--expect-post-init-abort")
    {
        nn::CpuEngine eng;
        nn::Linear ln(4, 3);
        (void)ln.init(eng);
        ln.set_precision_profile(nn::profile_f16());   // 必须 fail-fast
        std::printf("不应到达：profile 在 init 后被接受\n");
        return 0;
    }

    test_precision_set();
    test_capability_reject();
    test_capability_accept();
    test_post_init_immutable();
    test_bitlinear_rules();
    test_bitlinear_f16_compute();
    test_spec_roundtrip();

    // ── 端到端训练：--gpu 时用 GPU（无设备/未编译 Vulkan → 退出码 77 = skip）──
    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        if (!test_training(**eng_r, "GPU")) g_failures++;
    }
    else
    {
        nn::CpuEngine eng;
        if (!test_training(eng, "CPU")) g_failures++;
    }

    if (g_failures == 0)
    {
        std::printf("\nt1_58_test: ALL PASSED\n");
        return 0;
    }
    std::printf("\nt1_58_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
