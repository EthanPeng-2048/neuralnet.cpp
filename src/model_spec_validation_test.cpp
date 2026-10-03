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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"

using nn::ActivationType;
using nn::ModelSpec;
using nn::ModelType;
using nn::NormType;
using nn::NormPlace;
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
    ModelSpec m4 = m1;  m4.norm_type = NormType::RMSNorm;
    expect(!nn::spec_matches(m1, m4),  "MLP 不同 norm_type 不匹配", all);

    // Transformer
    ModelSpec t1;  t1.type = ModelType::Transformer;
    t1.d_model = 64; t1.num_heads = 4; t1.d_ff = 128; t1.num_layers = 2; t1.patch_size = 7;
    ModelSpec t2;  t2.type = ModelType::Transformer;
    t2.d_model = 64; t2.num_heads = 4; t2.d_ff = 128; t2.num_layers = 2; t2.patch_size = 7;
    ModelSpec t3 = t1;  t3.num_heads = 8;
    expect(nn::spec_matches(t1, t2),   "Transformer 相同 spec 匹配", all);
    expect(!nn::spec_matches(t1, t3),  "Transformer 不同 heads 不匹配", all);
    ModelSpec t4 = t1;  t4.norm_place = NormPlace::Final;
    expect(!nn::spec_matches(t1, t4),  "Transformer 不同 norm_place 不匹配", all);
    ModelSpec t5 = t1;  t5.norm_type = NormType::BatchNorm;
    expect(!nn::spec_matches(t1, t5),  "Transformer 不同 norm_type 不匹配", all);

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
    ModelSpec c4 = c1;  c4.norm_place = NormPlace::Both;
    expect(!nn::spec_matches(c1, c4),  "CNN 不同 norm_place 不匹配", all);
    ModelSpec c5 = c1;  c5.norm_type = NormType::RMSNorm;
    expect(!nn::spec_matches(c1, c5),  "CNN 不同 norm_type 不匹配", all);

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

    // ── 5. v5 矩阵形状未信任输入防御（内存安全审计 2026-09）────────────
    // rows/cols 直接来自文件字节：溢出回绕（小缓冲+巨大元数据 → 后续按行列
    // 索引越界）与巨量分配（内存耗尽）必须在分配前被拒绝；正常矩阵不得误杀。
    std::cout << "\n── v5 未信任形状防御（应拒绝恶意、放行正常） ──\n";
    {
        // 5.1 rows*cols 溢出回绕（2^40 × 2^40），无矩阵数据
        {
            const std::string file = "arch_bad_overflow.bin";
            std::ofstream ofs(file, std::ios::binary);
            const std::uint8_t tag = 0;  // f32
            const std::uint64_t rows = 1ULL << 40, cols = 1ULL << 40;
            ofs.write(reinterpret_cast<const char*>(&tag), 1);
            ofs.write(reinterpret_cast<const char*>(&rows), 8);
            ofs.write(reinterpret_cast<const char*>(&cols), 8);
            ofs.close();
            std::ifstream ifs(file, std::ios::binary);
            auto r = nn::detail::read_matrix_v5(ifs);
            expect(!r, "溢出形状 2^40x2^40 被拒绝", all);
            std::remove(file.c_str());
        }
        // 5.2 形状声明 4x4 但数据只剩 4 字节（远小于 64B）
        {
            const std::string file = "arch_bad_short.bin";
            std::ofstream ofs(file, std::ios::binary);
            const std::uint8_t tag = 0;
            const std::uint64_t rows = 4, cols = 4;
            const float one = 1.0f;
            ofs.write(reinterpret_cast<const char*>(&tag), 1);
            ofs.write(reinterpret_cast<const char*>(&rows), 8);
            ofs.write(reinterpret_cast<const char*>(&cols), 8);
            ofs.write(reinterpret_cast<const char*>(&one), 4);
            ofs.close();
            std::ifstream ifs(file, std::ios::binary);
            auto r = nn::detail::read_matrix_v5(ifs);
            expect(!r, "数据不足的形状被拒绝", all);
            std::remove(file.c_str());
        }
        // 5.3 正常 2x3 f32（数据恰好用满剩余字节）→ 接受且数值正确
        {
            const std::string file = "arch_ok_matrix.bin";
            std::ofstream ofs(file, std::ios::binary);
            const std::uint8_t tag = 0;
            const std::uint64_t rows = 2, cols = 3;
            const float vals[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
            ofs.write(reinterpret_cast<const char*>(&tag), 1);
            ofs.write(reinterpret_cast<const char*>(&rows), 8);
            ofs.write(reinterpret_cast<const char*>(&cols), 8);
            ofs.write(reinterpret_cast<const char*>(vals), sizeof(vals));
            ofs.close();
            std::ifstream ifs(file, std::ios::binary);
            auto r = nn::detail::read_matrix_v5(ifs);
            bool ok = r && r->second.rows() == 2 && r->second.cols() == 3;
            if (ok)
                for (std::size_t i = 0; i < 6; ++i)
                    if (r->second.span()[i] != vals[i]) ok = false;
            expect(ok, "正常矩阵读回（形状+数值，无误杀）", all);
            std::remove(file.c_str());
        }
    }

    // ── 6. 已移除模型类型（ZiPT，type=6）必须被明确拒绝 ────────────────
    // 2026-10-01 AttnZip/ZiPT 整体移除；ModelType 的 6 号取值保留为
    // Reserved_ZiPT 占位，旧 .bin（type=6）应得到**可读的**错误
    // （而非含糊的 "Unknown" 或静默误解）。见 docs/history.md「ZiPT 移除」。
    std::cout << "\n── 已移除模型类型（type=6）拒绝 ──\n";
    {
        nn::KeyValueRecord kv;
        kv.set("type", static_cast<uint64_t>(6));
        auto r = nn::detail::spec_from_kv(kv);
        if (r)
        {
            expect(false, "type=6（ZiPT）应被拒绝（但解析成功了）", all);
        }
        else
        {
            const bool readable = r.error().message.find("ZiPT") != std::string::npos;
            expect(readable, "type=6 拒绝信息含 'ZiPT'（可读）", all);
            std::cout << "      错误信息: " << r.error().message << "\n";
        }
    }

    // ── 7. 归一化挂载（NormPlace）：默认值 / 显式挂载 / 规格往返 ───────
    // **各架构默认（2026-10-03 起）**：CNN = BatchNorm@Conv、ViT = LayerNorm@Final、
    // MLP = LayerNorm + 结构内置挂载；**旧文件缺 norm_place 键 → None**（旧布局逐位不变）；
    // 显式挂载时 norm 必须真的在模型里，且 save→load 往返成功、错配被拦截。
    std::cout << "\n── 归一化挂载（NormPlace） ──\n";
    {
        nn::CpuEngine eng;

        // 7.0 旧文件兼容：缺 norm_place / norm_type 键 → 架构无关的 None / LayerNorm
        //（**不**回落到新默认 CNN=BatchNorm@Conv、ViT=LayerNorm@Final——
        //  否则旧 checkpoint 的参数布局会错位，加载必败）
        {
            nn::KeyValueRecord kv;
            kv.set("type", static_cast<uint64_t>(nn::ModelType::CNN));
            auto r = nn::detail::spec_from_kv(kv);
            expect(bool(r) && r->norm_place == nn::NormPlace::None &&
                            r->norm_type == nn::NormType::LayerNorm,
                   "缺 norm_place/norm_type 键的旧文件 → None/LayerNorm（旧布局）", all);
        }

        // 7.0b 各架构默认（回归锁，改默认值必须同步文档与锚点）
        {
            const nn::ModelSpec c = nn::make_cnn_spec(1, 12, 2, {{4, 3, 1, 0}}, {16, 10});
            expect(c.norm_type == NormType::BatchNorm && c.norm_place == NormPlace::Conv,
                   "make_cnn_spec 默认 = BatchNorm@Conv", all);
            const nn::ModelSpec t = nn::make_mnist_transformer_spec();
            expect(t.norm_type == NormType::LayerNorm && t.norm_place == NormPlace::Final,
                   "make_mnist_transformer_spec 默认 = LayerNorm@Final", all);
        }

        // 7.1 CNN：conv+head 挂 RMSNorm
        {
            nn::CnnConfig ccfg;
            ccfg.in_channels = 1; ccfg.in_size = 12; ccfg.pool = 2;
            ccfg.convs = {{4, 3, 1, 0}, {8, 3, 1, 0}};
            ccfg.fc_dims = {16, 10};
            ccfg.norm_type = NormType::RMSNorm;
            ccfg.norm_place = NormPlace::Both;

            const nn::ModelSpec spec = nn::make_cnn_spec(
                ccfg.in_channels, ccfg.in_size, ccfg.pool, ccfg.convs,
                ccfg.fc_dims, ccfg.norm_type, ccfg.norm_place);
            auto m = nn::build_cnn_model_from_spec(eng, spec);
            if (!m) { expect(false, "build_cnn_model(norm=both): " + m.error().message, all); }
            else
            {
                std::size_t n_rms = 0;
                for (std::size_t i = 0; i < m->num_layers(); ++i)
                    if (dynamic_cast<nn::RMSNorm*>(&m->layer_at(i))) ++n_rms;
                // 卷积后 2 个 + FC 隐藏层 1 个（fc_dims 尾元素是 logits 不挂）
                expect(n_rms == 3, "CNN norm_place=both → 3 个 RMSNorm（conv 2 + head 1）", all);

                expect(m->spec().has_value() && nn::spec_matches(spec, *m->spec()),
                       "CNN norm spec 记录且一致", all);

                const std::string file = "arch_check_cnn_norm.bin";
                if (auto r = nn::save_model(file, *m, spec); !r)
                    expect(false, "CNN save_model: " + r.error().message, all);
                else
                {
                    auto m2 = nn::build_cnn_model_from_spec(eng, spec);
                    if (!m2)
                        expect(false, "CNN build_from_spec: " + m2.error().message, all);
                    else
                    {
                        auto lr = nn::load_model(file, *m2);
                        expect(static_cast<bool>(lr), "CNN norm spec 往返 load 成功", all);
                    }
                    // norm_place 不同的架构加载同一文件 → 必须被拦截
                    nn::ModelSpec bad = spec;
                    bad.norm_place = NormPlace::None;
                    auto m3 = nn::build_cnn_model_from_spec(eng, bad);
                    if (!m3)
                        expect(false, "CNN build(norm_place=None): " + m3.error().message, all);
                    else
                    {
                        auto lr3 = nn::load_model(file, *m3);
                        expect(!lr3, "CNN norm_place 错配的 load 被拦截", all);
                    }
                }
                std::remove(file.c_str());
            }
        }

        // 7.2 ViT：final norm（+ 一轮 forward/backward，梯度有限）
        {
            const nn::ModelSpec spec = nn::make_mnist_transformer_spec(
                nn::MNIST_PATCH_SIZE, nn::MNIST_TF_D_MODEL, nn::MNIST_TF_NUM_HEADS,
                nn::MNIST_TF_D_FF, nn::MNIST_TF_NUM_LAYERS,
                NormType::LayerNorm, NormPlace::Final);
            auto m = nn::build_mnist_model_from_spec(eng, spec);
            if (!m) { expect(false, "build ViT(final norm): " + m.error().message, all); }
            else
            {
                const std::size_t B = 2;   // batch>1：批内布局（铁律 #5）
                auto x = eng.create_tensor(nn::MNIST_INPUT_DIM, B, nn::Precision::F32,
                                           nn::InitSpec::normal(0.0f, 1.0f, 42));
                auto out = m->forward(x);
                bool ok = out.has_value();
                if (ok)
                {
                    auto g = eng.create_tensor(out->rows(), out->cols(), nn::Precision::F32,
                                               nn::InitSpec::constant(1));
                    auto grad = m->backward(g);
                    ok = grad.has_value();
                    if (ok)
                    {
                        auto gm = eng.to_matrix(*grad);
                        ok = static_cast<bool>(gm);
                        if (ok)
                            for (auto v : gm->span())
                                if (!std::isfinite(v)) { ok = false; break; }
                    }
                }
                expect(ok, "ViT(final norm) forward/backward 梯度全有限", all);

                const std::string file = "arch_check_vit_norm.bin";
                if (auto r = nn::save_model(file, *m, spec); !r)
                    expect(false, "ViT save_model: " + r.error().message, all);
                else
                {
                    auto m2 = nn::build_mnist_model_from_spec(eng, spec);
                    if (!m2)
                        expect(false, "ViT build_from_spec: " + m2.error().message, all);
                    else
                    {
                        auto lr = nn::load_model(file, *m2);
                        expect(static_cast<bool>(lr), "ViT final-norm 往返 load 成功", all);
                    }
                    nn::ModelSpec bad = spec;
                    bad.norm_place = NormPlace::None;
                    auto m3 = nn::build_mnist_model_from_spec(eng, bad);
                    if (!m3)
                        expect(false, "ViT build(norm_place=None): " + m3.error().message, all);
                    else
                    {
                        auto lr3 = nn::load_model(file, *m3);
                        expect(!lr3, "ViT norm_place 错配的 load 被拦截", all);
                    }
                }
                std::remove(file.c_str());
            }
        }
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
