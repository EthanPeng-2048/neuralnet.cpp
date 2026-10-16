// ── model_roles_test — 模型文件角色（v7：.nnmodel / .nnckpt）+ 参数形态协议 ──
// 设计依据：docs/todo/23-library-design-cleanup-and-model-file-roles.md §2.1/§2.2
//
// 覆盖（取代原 ternary_deploy_test 的部署格式面 —— NNQT 独立格式已删除）：
//   [1] Model 角色（推理）：三值层的权重以**打包形态**落盘（盘上 base-3）；
//       载入后层进只读推理态（无 latent），前向与保存前逐位一致。
//   [2] Checkpoint 角色（训练）：三值层写 f32 latent；载入后 latent 逐位一致，
//       且可继续反向（梯度回 latent）。
//   [3] 载荷字节账：Model 文件 < Checkpoint 文件（base-3 1.6 bpw + 无 latent）。
//   [4] 角色只读查询：peek_model_role 与文件头一致。
//   [5] 反例：只读态下按 Checkpoint 角色导出 → 明确错误（不静默）。
//   [6] **P1-7**：`WeightLoadIntent::Trainable` 载入三值 Model 文件 ⇒ 按文件 τ
//       反量化成 f32 latent（可训练）：打包字/τ 与文件逐位一致、首次 forward 与
//       源模型逐位一致（播种缓存，不重算）、latent == 量化器不动点、8 步 Adam
//       latent 确实移动；dense 文件两种意图逐位相同。
//
// 退出码：0 = 通过，1 = 失败
// ───────────────────────────────────────────────────────────────────────────

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "neuralnet.cpp/compute_layer_bitlinear.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

constexpr std::size_t IN = 32;
constexpr std::size_t H = 16;
constexpr std::size_t OUT = 8;
constexpr std::size_t B = 4;

// 单隐层三值 MLP（权重走 BitLinear）
nn::ModelSpec ternary_spec()
{
    nn::ModelSpec spec;
    spec.type = nn::ModelType::MLP;
    spec.layer_dims = {IN, H, OUT};
    spec.weight_quant = nn::WeightQuant::T1_58;
    return spec;
}

[[nodiscard]] nn::Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : m.span()) v = d(rng);
    return m;
}

// 找到第一个 BitLinear（按参数顺序遍历顶层层的子层）
[[nodiscard]] nn::BitLinear* first_bitlinear(nn::Model& model)
{
    for (std::size_t li = 0; li < model.num_layers(); ++li)
    {
        nn::Layer& L = model.layer_at(li);
        if (auto* bl = dynamic_cast<nn::BitLinear*>(&L)) return bl;
    }
    return nullptr;
}

[[nodiscard]] std::uint64_t file_size(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return static_cast<std::uint64_t>(f.tellg());
}

bool matrices_equal(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (sa[i] != sb[i]) return false;
    return true;
}

// 位级比较：**打包容器专用**（容器是"f32 张量承载原始字"，字里可能是 NaN 位型
// —— 例如 0xffda75eb —— 而 NaN != NaN，用浮点比较会误报不等）。
bool matrices_bit_identical(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (std::bit_cast<std::uint32_t>(sa[i]) != std::bit_cast<std::uint32_t>(sb[i]))
            return false;
    return true;
}

void test_model_and_checkpoint_roles()
{
    std::puts("[1] Model 角色：打包形态落盘 + 只读态载入 + 前向逐位一致");
    nn::CpuEngine eng;
    const nn::ModelSpec spec = ternary_spec();
    const nn::PrecisionProfile t1 = nn::profile_t1_58();

    auto m_r = nn::build_mnist_model_from_spec(eng, spec, t1);
    CHECK(m_r.has_value(), "三值模型构建成功");
    if (!m_r.has_value()) return;
    nn::Model& model = *m_r;

    const nn::Tensor x = upload(eng, random_matrix(IN, B, 7));
    auto y_r = model.forward(x);
    CHECK(y_r.has_value(), "原模型前向成功");
    if (!y_r.has_value()) return;
    const nn::Matrix y_ref = download(eng, *y_r);

    const std::string f_model = "model_roles_test.nnmodel";
    const std::string f_ckpt = "model_roles_test.nnckpt";
    auto sr = nn::save_model(f_model, model, spec, {}, nn::ModelRole::Model);
    CHECK(sr.has_value(), "save_model(Model) 成功");
    auto sc = nn::save_model(f_ckpt, model, spec, {}, nn::ModelRole::Checkpoint);
    CHECK(sc.has_value(), "save_model(Checkpoint) 成功");
    if (!sr.has_value() || !sc.has_value()) return;

    // ── [4] 角色只读查询 ──
    {
        auto r1 = nn::peek_model_role(f_model);
        auto r2 = nn::peek_model_role(f_ckpt);
        CHECK(r1.has_value() && *r1 == nn::ModelRole::Model, "peek_model_role(.nnmodel) = Model");
        CHECK(r2.has_value() && *r2 == nn::ModelRole::Checkpoint,
              "peek_model_role(.nnckpt) = Checkpoint");
    }

    // ── [3] 字节账：Model 文件显著小于 Checkpoint 文件 ──
    {
        const std::uint64_t sm = file_size(f_model);
        const std::uint64_t scp = file_size(f_ckpt);
        std::printf("    .nnmodel = %llu B, .nnckpt = %llu B\n",
                    static_cast<unsigned long long>(sm),
                    static_cast<unsigned long long>(scp));
        CHECK(sm > 0 && scp > 0 && sm < scp,
              "推理文件小于断点文件（base-3 打包 + 无 latent）");
    }

    // ── [1] Model 角色载入：只读态 + 前向逐位一致 ──
    {
        auto m2_r = nn::build_mnist_model_from_spec(eng, spec, t1);
        CHECK(m2_r.has_value(), "按规格重建模型");
        if (!m2_r.has_value()) return;
        nn::Model& m2 = *m2_r;
        auto lr = nn::load_model(f_model, m2);
        CHECK(lr.has_value(), "load_model(.nnmodel) 成功");
        if (!lr.has_value())
        {
            std::printf("    载入失败: %s\n", lr.error().message.c_str());
            return;
        }
        nn::BitLinear* bl = first_bitlinear(m2);
        CHECK(bl != nullptr, "重建模型里有 BitLinear");
        if (bl != nullptr)
        {
            CHECK(!bl->latent_weights().valid(),
                  "推理文件载入后没有 latent（只读推理态）");
            auto g = bl->backward(bl->latent_weights());   // 只读态 → 错误（不静默）
            CHECK(!g.has_value(), "只读态 backward() 明确报错");
        }
        auto y2_r = m2.forward(x);
        CHECK(y2_r.has_value(), "载入模型前向成功");
        if (y2_r.has_value())
            CHECK(matrices_equal(y_ref, download(eng, *y2_r)),
                  "前向输出与保存前**逐位一致**");
    }

    // ── [2] Checkpoint 角色载入：latent 逐位一致 + 可继续训练 ──
    {
        auto m3_r = nn::build_mnist_model_from_spec(eng, spec, t1);
        CHECK(m3_r.has_value(), "按规格重建模型（断点）");
        if (!m3_r.has_value()) return;
        nn::Model& m3 = *m3_r;
        auto lr = nn::load_model(f_ckpt, m3);
        CHECK(lr.has_value(), "load_model(.nnckpt) 成功");
        if (!lr.has_value())
        {
            std::printf("    载入失败: %s\n", lr.error().message.c_str());
            return;
        }
        nn::BitLinear* bl_src = first_bitlinear(model);
        nn::BitLinear* bl_dst = first_bitlinear(m3);
        CHECK(bl_src != nullptr && bl_dst != nullptr, "两侧都有 BitLinear");
        if (bl_src != nullptr && bl_dst != nullptr)
        {
            CHECK(bl_dst->latent_weights().valid(), "断点载入后 latent 存在（可续训）");
            CHECK(matrices_equal(download(eng, bl_src->latent_weights()),
                                 download(eng, bl_dst->latent_weights())),
                  "latent 逐位一致");
            // 可继续训练：一次前向 + 反向必须都成功（梯度回 f32 latent）
            auto y3 = m3.forward(x);
            CHECK(y3.has_value(), "断点载入后前向成功");
            if (y3.has_value())
            {
                // 该 BitLinear 是隐层投影（out=H）→ 梯度形状 (H,B)
                const nn::Tensor dy = upload(eng, random_matrix(H, B, 11));
                auto g = bl_dst->backward(dy);
                CHECK(g.has_value(), "断点载入后可反向（梯度回 latent）");
                if (!g.has_value())
                    std::printf("    backward 失败: %s\n", g.error().message.c_str());
            }
        }
    }

    // ── [5] 反例：只读态按 Checkpoint 角色导出 → 明确错误 ──
    {
        auto m4_r = nn::build_mnist_model_from_spec(eng, spec, t1);
        if (m4_r.has_value())
        {
            nn::Model& m4 = *m4_r;
            if (nn::load_model(f_model, m4).has_value())
            {
                nn::BitLinear* bl = first_bitlinear(m4);
                if (bl != nullptr)
                {
                    auto e = bl->export_param(0, nn::ModelRole::Checkpoint);
                    CHECK(!e.has_value(),
                          "只读推理态按断点角色导出 → 明确错误（不静默）");
                }
            }
        }
    }

    std::remove(f_model.c_str());
    std::remove(f_ckpt.c_str());
}

// ── [6] P1-7：三值 Model 文件的 **warm start**（打包 → latent 反量化）────────
// 载入意图（`nn::WeightLoadIntent`）与文件角色正交：同一个 `.nnmodel` 既能进
// 部署只读态（Inference，[1] 已覆盖），也能按文件 τ 反量化成可训练 latent
// （Trainable）——后者就是三值模型的 SFT / QAT 续训入口。
void test_ternary_warm_start()
{
    nn::CpuEngine eng;
    const nn::ModelSpec spec = ternary_spec();
    const nn::PrecisionProfile t1 = nn::profile_t1_58();

    auto m_r = nn::build_mnist_model_from_spec(eng, spec, t1);
    CHECK(m_r.has_value(), "warm start 源模型构建成功");
    if (!m_r.has_value()) return;
    nn::Model& src = *m_r;

    const nn::Tensor x = upload(eng, random_matrix(IN, B, 21));
    auto y_ref_r = src.forward(x);
    CHECK(y_ref_r.has_value(), "源模型前向成功");
    if (!y_ref_r.has_value()) return;
    const nn::Matrix y_ref = download(eng, *y_ref_r);

    nn::BitLinear* bl_src = first_bitlinear(src);
    CHECK(bl_src != nullptr, "源模型有 BitLinear");
    if (bl_src == nullptr) return;
    // 源模型的打包派生物 = 落进文件的那份（Model 角色原样写打包字 + τ）
    const nn::Matrix src_packed = download(eng, bl_src->quantized_weights_packed());
    const nn::Matrix src_tau = download(eng, bl_src->row_scales());

    const std::string f = "model_roles_warmstart.nnmodel";
    auto sr = nn::save_model(f, src, spec, {}, nn::ModelRole::Model);
    CHECK(sr.has_value(), "save_model(Model) 成功");
    if (!sr.has_value()) return;

    auto m2_r = nn::build_mnist_model_from_spec(eng, spec, t1);
    CHECK(m2_r.has_value(), "按规格重建模型（warm start）");
    if (!m2_r.has_value()) return;
    nn::Model& m2 = *m2_r;
    auto lr = nn::load_model(f, m2, nullptr, nullptr, nn::WeightLoadIntent::Trainable);
    CHECK(lr.has_value(), "Trainable 意图载入三值 .nnmodel 成功");
    if (!lr.has_value())
    {
        std::printf("    载入失败: %s\n", lr.error().message.c_str());
        std::remove(f.c_str());
        return;
    }
    nn::BitLinear* bl = first_bitlinear(m2);
    CHECK(bl != nullptr, "重建模型里有 BitLinear");
    if (bl == nullptr)
    {
        std::remove(f.c_str());
        return;
    }

    // ① 可训练态（不是只读部署态）
    CHECK(!bl->packed_only(), "Trainable 载入后不是 packed_only（可训练）");
    CHECK(bl->latent_weights().valid(), "latent 已建立");
    CHECK(bl->quantized_weights_packed().valid() && bl->row_scales().valid(),
          "打包字 + τ 都在（播种的量化缓存）");
    // ② wq/τ 与文件里的那份**逐位一致**（播种自文件，不经"用自己的 absmean 重算"）
    {
        const nn::Matrix packed = download(eng, bl->quantized_weights_packed());
        const nn::Matrix tau = download(eng, bl->row_scales());
        CHECK(matrices_bit_identical(src_packed, packed), "打包字与文件里的逐位一致");
        CHECK(matrices_equal(src_tau, tau), "τ 与文件里的逐位一致");
        CHECK(bl->quantize_calls() == 1, "载入时播种打包派生物（trit_pack 恰 1 次）");
    }
    // ③ 首次 forward 与源模型逐位一致，且用的是文件里的 wq/τ（不重算）
    {
        auto y2_r = m2.forward(x);
        CHECK(y2_r.has_value(), "warm start 前向成功");
        if (y2_r.has_value())
            CHECK(matrices_equal(y_ref, download(eng, *y2_r)),
                  "首次 forward 与推理文件里的模型逐位一致");
        CHECK(bl->quantize_calls() == 1, "首次 forward 不重算 τ/wq（缓存命中）");
    }
    // ④ 反量化 latent = **量化器的不动点**：L == trit·(τ·K/n) 逐位，
    //    且逐行 absmean(L) == τ（否则续训一开步就是尺度跳变，见 ternary_pack.hpp）
    {
        const nn::Matrix lat = download(eng, bl->latent_weights());
        const nn::Matrix tau = download(eng, bl->row_scales());
        const std::size_t wpr = nn::trit_pack::words_per_row(IN);
        std::vector<std::uint32_t> words(wpr);
        bool formula_ok = true, fixed_point_ok = true;
        for (std::size_t o = 0; o < OUT; ++o)
        {
            for (std::size_t i = 0; i < wpr; ++i)
                words[i] = std::bit_cast<std::uint32_t>(src_packed.span()[o * wpr + i]);
            const float s =
                nn::trit_pack::latent_row_scale(words.data(), IN, tau.span()[o]);
            for (std::size_t c = 0; c < IN; ++c)
                if (lat.span()[o * IN + c] != nn::trit_pack::at(words.data(), c) * s)
                    formula_ok = false;
            double acc = 0.0;
            for (std::size_t c = 0; c < IN; ++c)
                acc += std::fabs(static_cast<double>(lat.span()[o * IN + c]));
            const double mean = acc / static_cast<double>(IN);
            const double tol =
                2e-6 * std::max(1.0, std::fabs(static_cast<double>(tau.span()[o])));
            if (std::fabs(mean - static_cast<double>(tau.span()[o])) > tol)
                fixed_point_ok = false;
        }
        CHECK(formula_ok, "反量化 latent == trit·(τ·K/n)（逐位）");
        CHECK(fixed_point_ok, "逐行 absmean(latent) == τ（量化器不动点）");
    }
    // ⑤ 8 步 Adam：backward 可跑、latent（f32 主权重）确实移动
    {
        auto opt = nn::create_optimizer("adam", eng, m2.parameters(), m2.param_gradients(),
                                        1e-2f, 0.0f, t1);
        CHECK(opt != nullptr, "warm start 后能建 Adam 优化器");
        if (opt != nullptr)
        {
            const nn::Matrix lat_before = download(eng, bl->latent_weights());
            nn::CrossEntropyLoss ce;
            ce.set_precision_profile(t1);
            nn::Matrix tgt(OUT, B, nn::Scalar{0});
            for (std::size_t b = 0; b < B; ++b)
                tgt.set_value(b % OUT, b, 1.0f);
            const nn::Tensor tgt_t = upload(eng, tgt);
            bool steps_ok = true;
            for (int step = 0; step < 8 && steps_ok; ++step)
            {
                steps_ok = m2.zero_grad().has_value();
                auto logits = m2.forward(x);
                if (!logits) { steps_ok = false; break; }
                auto loss = ce.forward(eng, *logits, tgt_t);
                if (!loss) { steps_ok = false; break; }
                auto grad = ce.backward();
                if (!grad) { steps_ok = false; break; }
                if (!m2.backward(*grad)) { steps_ok = false; break; }
                if (!opt->step()) { steps_ok = false; break; }
            }
            CHECK(steps_ok, "warm start 后 8 步 Adam（forward/loss/backward/step）全部成功");
            const nn::Matrix lat_after = download(eng, bl->latent_weights());
            std::size_t moved = 0;
            for (std::size_t i = 0; i < lat_before.size() && i < lat_after.size(); ++i)
                if (lat_before.span()[i] != lat_after.span()[i]) ++moved;
            std::printf("    三值 latent 8 步 Adam：移动 %zu/%zu\n", moved,
                        lat_before.size());
            CHECK(moved > 0, "8 步 Adam 后 latent 确实移动（三值续训可用）");
            CHECK(bl->quantize_calls() > 1, "latent 被写过 ⇒ 量化缓存按写版本重算");
        }
    }
    std::remove(f.c_str());

    // ⑥ dense 文件：两种载入意图**逐位相同**（P1-7 不改变 dense 路径）
    {
        nn::ModelSpec dspec;
        dspec.type = nn::ModelType::MLP;
        dspec.layer_dims = {IN, H, OUT};
        const nn::PrecisionProfile f32p = nn::profile_f32();
        auto d_r = nn::build_mnist_model_from_spec(eng, dspec, f32p);
        CHECK(d_r.has_value(), "dense 模型构建成功");
        if (d_r.has_value())
        {
            nn::Model& dense = *d_r;
            const std::string df = "model_roles_warmstart_dense.nnmodel";
            auto dsr = nn::save_model(df, dense, dspec, {}, nn::ModelRole::Model);
            CHECK(dsr.has_value(), "save_model(dense) 成功");
            if (dsr.has_value())
            {
                auto a_r = nn::build_mnist_model_from_spec(eng, dspec, f32p);
                auto b_r = nn::build_mnist_model_from_spec(eng, dspec, f32p);
                CHECK(a_r.has_value() && b_r.has_value(), "dense 重建两份");
                if (a_r.has_value() && b_r.has_value())
                {
                    nn::Model& a = *a_r;
                    nn::Model& b = *b_r;
                    auto la = nn::load_model(df, a);   // 默认 Inference
                    auto lb = nn::load_model(df, b, nullptr, nullptr,
                                             nn::WeightLoadIntent::Trainable);
                    CHECK(la.has_value() && lb.has_value(), "两种意图载入 dense 均成功");
                    if (la.has_value() && lb.has_value())
                    {
                        auto pa = a.parameters();
                        auto pb = b.parameters();
                        bool same = (pa.size() == pb.size());
                        for (std::size_t i = 0; same && i < pa.size(); ++i)
                            same = matrices_equal(download(eng, pa[i].get()),
                                                  download(eng, pb[i].get()));
                        CHECK(same, "dense 文件两种意图载入的参数逐位一致（零回归）");
                    }
                }
                std::remove(df.c_str());
            }
        }
    }
}

}   // namespace

int main()
{
    test_model_and_checkpoint_roles();
    test_ternary_warm_start();
    if (g_failures == 0)
    {
        std::printf("\nmodel_roles_test: ALL PASSED\n");
        return 0;
    }
    std::printf("\nmodel_roles_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
