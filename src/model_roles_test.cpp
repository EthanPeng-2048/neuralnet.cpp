// ── model_roles_test — 模型文件角色（v7：.nnmodel / .nnckpt）+ 参数形态协议 ──
// 设计依据：docs/development/23-library-design-cleanup-and-model-file-roles.md §2.1/§2.2
//
// 覆盖（取代原 ternary_deploy_test 的部署格式面 —— NNQT 独立格式已删除）：
//   [1] Model 角色（推理）：三值层的权重以**打包形态**落盘（盘上 base-3）；
//       载入后层进只读推理态（无 latent），前向与保存前逐位一致。
//   [2] Checkpoint 角色（训练）：三值层写 f32 latent；载入后 latent 逐位一致，
//       且可继续反向（梯度回 latent）。
//   [3] 载荷字节账：Model 文件 < Checkpoint 文件（base-3 1.6 bpw + 无 latent）。
//   [4] 角色只读查询：peek_model_role 与文件头一致。
//   [5] 反例：只读态下按 Checkpoint 角色导出 → 明确错误（不静默）。
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

}   // namespace

int main()
{
    test_model_and_checkpoint_roles();
    if (g_failures == 0)
    {
        std::printf("\nmodel_roles_test: ALL PASSED\n");
        return 0;
    }
    std::printf("\nmodel_roles_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
