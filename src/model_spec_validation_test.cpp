// ── model_spec_validation_test.cpp — ModelSpec 架构校验一致性测试 ──────────
//
// 目的：验证 ModelSpec 架构校验逻辑（load_model 时文件头与模型自身规格比对）。
//
// 覆盖：
//   1. spec_matches 单元断言：GPT/MLP/Transformer/CNN 各类型的匹配/不匹配，
//      以及 GPT ↔ ALiBi_GPT 家族兼容（统一 GPTModel）。
//   2. round-trip：build_gpt_model_from_spec → save_model → load_model，
//      匹配 spec 应加载成功。
//   3. 不匹配 spec：用不同架构构建的 model 加载同一文件应报错。
//   4. 无 spec 模型（build_gpt_model 直接构建）：跳过校验，向后兼容。
// ─────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"

using nn::ActivationType;
using nn::ModelSpec;
using nn::ModelType;
using nn::NormType;
using nn::PosEncodingType;

namespace
{

// 记录断言结果，全部通过时 run_test 返回 0。
void expect(bool cond, const std::string& name, bool& all)
{
    if (cond)
        std::cout << "  ✓ " << name << "\n";
    else
    {
        std::cout << "  ✗ " << name << "\n";
        all = false;
    }
}

int run_test()
{
    bool all = true;
    std::cout << "========================================\n";
    std::cout << "  ModelSpec 架构校验一致性测试\n";
    std::cout << "========================================\n";

    // ── 1. spec_matches 单元断言 ──────────────────────────────────────
    std::cout << "\n── spec_matches 单元断言 ──\n";

    // GPT
    const ModelSpec g1 = nn::make_gpt_spec(100, 64, 32, 4, 128, 2);
    const ModelSpec g2 = nn::make_gpt_spec(100, 64, 32, 4, 128, 2);
    const ModelSpec g3 = nn::make_gpt_spec(101, 64, 32, 4, 128, 2);
    expect(nn::spec_matches(g1, g2),   "GPT 相同 spec 匹配", all);
    expect(!nn::spec_matches(g1, g3),  "GPT 不同 vocab 不匹配", all);

    // GPT ↔ ALiBi_GPT 家族兼容（统一 GPTModel，type 允许不同）
    // 构造 type 不同、其余字段（含 pos_encoding）相同的两份 spec。
    ModelSpec g_fam_a = g1;  g_fam_a.pos_encoding = PosEncodingType::ALiBi;              // type=GPT
    ModelSpec g_fam_b = g1;  g_fam_b.type = ModelType::ALiBi_GPT; g_fam_b.pos_encoding = PosEncodingType::ALiBi;
    expect(nn::spec_matches(g_fam_a, g_fam_b),   "GPT 与 ALiBi_GPT 家族兼容（同字段）", all);

    // GPT 跨类型不匹配
    ModelSpec g_other = g1;
    g_other.type = ModelType::MLP;
    expect(!nn::spec_matches(g1, g_other), "GPT 与 MLP 跨类型不匹配", all);

    // MLP
    ModelSpec m1;  m1.type = ModelType::MLP; m1.layer_dims = {128, 10};
    ModelSpec m2;  m2.type = ModelType::MLP; m2.layer_dims = {128, 10};
    ModelSpec m3;  m3.type = ModelType::MLP; m3.layer_dims = {64, 10};
    expect(nn::spec_matches(m1, m2),   "MLP 相同 layer_dims 匹配", all);
    expect(!nn::spec_matches(m1, m3),  "MLP 不同 layer_dims 不匹配", all);

    // Transformer
    ModelSpec t1;  t1.type = ModelType::Transformer;
    t1.d_model = 64; t1.num_heads = 4; t1.d_ff = 128; t1.num_layers = 2; t1.patch_size = 7;
    ModelSpec t2;  t2.type = ModelType::Transformer;
    t2.d_model = 64; t2.num_heads = 4; t2.d_ff = 128; t2.num_layers = 2; t2.patch_size = 7;
    ModelSpec t3 = t1;  t3.num_heads = 8;
    expect(nn::spec_matches(t1, t2),   "Transformer 相同 spec 匹配", all);
    expect(!nn::spec_matches(t1, t3),  "Transformer 不同 heads 不匹配", all);

    // CNN
    ModelSpec c1;  c1.type = ModelType::CNN;
    c1.cnn_in_channels = 1; c1.cnn_in_size = 28; c1.cnn_pool = 2;
    c1.cnn_channels = {16, 32}; c1.cnn_kernels = {5, 5};
    c1.cnn_strides = {1, 1};    c1.cnn_paddings = {0, 0};
    c1.layer_dims = {10};
    ModelSpec c2 = c1;
    ModelSpec c3 = c1;  c3.cnn_channels = {16, 64};
    expect(nn::spec_matches(c1, c2),   "CNN 相同 spec 匹配", all);
    expect(!nn::spec_matches(c1, c3),  "CNN 不同 channels 不匹配", all);

    // ── 2. round-trip：匹配 spec 加载成功 ─────────────────────────────
    std::cout << "\n── round-trip（匹配 spec 加载成功） ──\n";
    {
        nn::CpuEngine eng;
        const std::string file = "arch_check_matching.bin";
        const ModelSpec spec = nn::make_gpt_spec(100, 64, 32, 4, 128, 2);

        auto m = nn::build_gpt_model_from_spec(eng, spec);
        if (!m) { expect(false, "build_gpt_model_from_spec: " + m.error().message, all); return all ? 0 : 1; }
        expect(m->spec().has_value(), "build_gpt_model_from_spec 记录 spec", all);

        if (auto r = nn::save_model(file, *m, spec); !r)
        { expect(false, "save_model: " + r.error().message, all); return all ? 0 : 1; }

        auto m2 = nn::build_gpt_model_from_spec(eng, spec);
        auto lr = nn::load_model(file, *m2);
        if (lr)
            expect(true, "匹配 spec 的 load_model 成功", all);
        else
            expect(false, "匹配 spec 应加载成功: " + lr.error().message, all);
        std::remove(file.c_str());
    }

    // ── 3. 不匹配 spec：加载同一文件应报错 ────────────────────────────
    std::cout << "\n── 不匹配 spec（应报错） ──\n";
    {
        nn::CpuEngine eng;
        const std::string file = "arch_check_mismatch.bin";
        const ModelSpec spec = nn::make_gpt_spec(100, 64, 32, 4, 128, 2);

        auto m = nn::build_gpt_model_from_spec(eng, spec);
        if (!m) { expect(false, "build_gpt_model_from_spec: " + m.error().message, all); return all ? 0 : 1; }
        if (auto r = nn::save_model(file, *m, spec); !r)
        { expect(false, "save_model: " + r.error().message, all); return all ? 0 : 1; }

        // 用不同 vocab 的架构构建，加载同一文件应被校验拦截
        const ModelSpec bad = nn::make_gpt_spec(999, 64, 32, 4, 128, 2);
        auto m2 = nn::build_gpt_model_from_spec(eng, bad);
        auto lr = nn::load_model(file, *m2);
        if (lr)
        {
            expect(false, "不匹配 spec 应加载失败（但成功了）", all);
        }
        else
        {
            expect(true, "不匹配 spec 的 load_model 正确报错", all);
            std::cout << "      错误信息: " << lr.error().message << "\n";
        }
        std::remove(file.c_str());
    }

    // ── 4. 无 spec 模型（build_gpt_model 直接构建）：跳过校验 ─────────
    std::cout << "\n── 无 spec 模型（跳过校验，向后兼容） ──\n";
    {
        nn::CpuEngine eng;
        const std::string file = "arch_check_nospec.bin";
        const ModelSpec spec = nn::make_gpt_spec(100, 64, 32, 4, 128, 2);

        auto m = nn::build_gpt_model_from_spec(eng, spec);
        if (!m) { expect(false, "build_gpt_model_from_spec: " + m.error().message, all); return all ? 0 : 1; }
        if (auto r = nn::save_model(file, *m, spec); !r)
        { expect(false, "save_model: " + r.error().message, all); return all ? 0 : 1; }

        // 用与文件相同架构直接构建（无 spec，跳过架构校验），加载应成功
        // （build_gpt_model 不记录 spec，load_model 不会触发 spec_matches 拦截）。
        auto m2 = nn::build_gpt_model(eng, 100, 64, 32, 4, 128, 2);
        expect(!m2->spec().has_value(), "build_gpt_model 不记录 spec", all);
        auto lr = nn::load_model(file, *m2);
        if (lr)
            expect(true, "无 spec 模型 load_model 成功（跳过校验）", all);
        else
            expect(false, "无 spec 模型应加载成功: " + lr.error().message, all);
        std::remove(file.c_str());
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "  结果: " << (all ? "✅ 全部通过" : "❌ 存在失败") << "\n";
    return all ? 0 : 1;
}

} // namespace

int main()
{
    return run_test();
}
