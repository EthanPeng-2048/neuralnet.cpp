// ── ternary_deploy_test — 三值部署格式（P2：packed-only 推理文件）──────────
// 设计依据：docs/development/21-quantized-weights.md §5 的 P2 剩余项
//   （"序列化打包 / 推理部署格式"）+ §4.11 ② 的 9.4× 名义收益。
//
// 覆盖：
//   [1] 文件类型识别：magic 分派（部署文件 vs 训练文件），peek 两种都读得出
//   [2] 逐位等价：部署文件载入的模型前向输出与原模型**逐位相同**（同打包字 +
//       同 τ ⇒ 同内核同输入 ⇒ 必须逐位；2-bit 与 base-3 两种载荷都要过）
//   [3] packed-only 语义：载入后三值层没有 latent/梯度（`packed_only()`），
//       反向 fail-fast 而不是静默用残层
//   [4] 尺寸账：部署文件 < 训练文件的 1/8（权重 32 bit → 1.6~2.0 bit），
//       且 base-3 载荷 ≤ 2-bit 载荷
//   [5] 训练文件路径零回归：`load_any_model` 载训练文件 → 层不是 packed-only，
//       可继续训练（对部署格式的"别把训练文件变成只读"反向断言）
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_layer_bitlinear.hpp"
#include "neuralnet.cpp/domain_gpt.hpp"
#include "neuralnet.cpp/model_container.hpp"
#include "neuralnet.cpp/model_serialization.hpp"
#include "neuralnet.cpp/ternary_deploy.hpp"
#include "neuralnet.cpp/ternary_pack.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

nn::ModelSpec ternary_spec()
{
    nn::ModelSpec s;
    s.type = nn::ModelType::MLP;
    s.layer_dims = {70, 33};   // 与 build_ternary_mlp 的两层形状同源（仅作规格往返）
    s.weight_quant = nn::WeightQuant::T1_58;
    return s;
}

// 小 MLP：两层 BitLinear（含偏置）——足够覆盖"打包权重 + 普通矩阵（偏置）"
// 两种条目共存，且不依赖某个具体 domain 工厂。
std::unique_ptr<nn::Model> build_ternary_mlp(nn::ComputeEngine& eng, std::size_t in,
                                             std::size_t hidden, std::size_t out)
{
    const nn::ModelSpec spec = ternary_spec();   // 与 save/load 用**同一份**规格
    auto model = std::make_unique<nn::Model>(eng);
    // 三值 profile：param = T1_58（BitLinear 的能力集合只收它；compute 保持 f32 →
    // CPU/GPU 同一口径，本测试只关心"部署格式的往返"而不是数值配方）
    nn::PrecisionProfile prof;
    prof.param = nn::Precision::T1_58;
    NN_CHECK(model->add_layer(std::make_unique<nn::BitLinear>(in, hidden), prof),
             "BitLinear#1 add 失败");
    NN_CHECK(model->add_layer(std::make_unique<nn::BitLinear>(hidden, out), prof),
             "BitLinear#2 add 失败");
    model->set_spec(spec);
    return model;
}

// 写入确定的 latent（避免依赖随机初始化）：两层都填同一套可复现的花样
void seed_latents(nn::ComputeEngine& eng, nn::Model& model)
{
    for (std::size_t li = 0; li < model.num_layers(); ++li)
    {
        auto& layer = model.layer_at(li);
        auto params = layer.parameters();
        for (std::size_t pi = 0; pi < params.size(); ++pi)
        {
            const std::size_t rows = params[pi].get().rows();
            const std::size_t cols = params[pi].get().cols();
            nn::Matrix m(rows, cols);
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t c = 0; c < cols; ++c)
                    m.set_value(r, c, 0.19f * static_cast<float>((r * 7 + c * 5 + li * 3) % 13)
                                          - 1.1f);
            NN_CHECK(eng.copy_from(params[pi], m), "写入 latent 失败");
        }
    }
}

std::size_t file_size(const std::string& path)
{
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    return ec ? 0u : static_cast<std::size_t>(n);
}

// ── 主流程 ────────────────────────────────────────────────────────────────
void test_deploy(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[1..5] 三值部署格式：载入逐位等价 / packed-only / 尺寸账（%s）\n", tag);
    if (!eng.describe().ternary_gemm)
    {
        std::puts("    （该引擎不支持三值 GEMM → 跳过；部署格式要求打包路径）");
        return;
    }
    constexpr std::size_t IN = 70, HID = 150, OUT = 33;
    const std::string dir = std::string("deploy_test_") + tag;
    std::filesystem::create_directories(dir);
    const std::string train_path = dir + "/train.bin";
    const std::string dep_b3 = dir + "/deploy_b3.nnqt";
    const std::string dep_2b = dir + "/deploy_2b.nnqt";

    auto a = build_ternary_mlp(eng, IN, HID, OUT);
    seed_latents(eng, *a);

    // 输入（确定性）
    nn::Matrix X(IN, 5);
    for (std::size_t r = 0; r < IN; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            X.set_value(r, c, 0.23f * static_cast<float>((r + 3 * c) % 7) - 0.7f);
    const nn::Tensor Xt = upload(eng, X);
    auto y_a = a->forward(Xt);
    CHECK(y_a.has_value(), "原模型 forward 成功");
    if (!y_a.has_value()) return;
    const nn::Matrix Y_ref = download(eng, *y_a);

    // ── [1] 训练文件（v6）+ 两个部署文件（base-3 / 2-bit）───────────────
    const std::string tok = "\x4e\x4e\x56\x43";   // 占位 tokenizer 字节（镜像语义）
    NN_CHECK(nn::save_model(train_path, *a, ternary_spec(), tok), "save_model 成功");
    NN_CHECK(nn::save_deployment(dep_b3, *a, ternary_spec(), /*base3=*/true, tok),
             "save_deployment(base-3) 成功");
    NN_CHECK(nn::save_deployment(dep_2b, *a, ternary_spec(), /*base3=*/false, tok),
             "save_deployment(2-bit) 成功");
    {
        auto d1 = nn::is_deployment_file(dep_b3);
        auto d2 = nn::is_deployment_file(train_path);
        CHECK(d1.has_value() && *d1, "is_deployment_file(部署) = true");
        CHECK(d2.has_value() && !*d2, "is_deployment_file(训练) = false");
        auto s1 = nn::peek_any_model_spec(dep_b3);
        auto s2 = nn::peek_any_model_spec(train_path);
        CHECK(s1.has_value() && s2.has_value() &&
                  s1->weight_quant == nn::WeightQuant::T1_58 &&
                  s2->weight_quant == nn::WeightQuant::T1_58,
              "peek_any_model_spec：两种文件都读得出 weight_quant=T1_58");
        auto ok = nn::load_any_model(dep_b3, *a, nullptr);   // 反向：往已载模型上再载
        CHECK(ok.has_value(), "load_any_model(部署文件) 成功");
    }

    // ── [2][3] 载入部署文件（新模型）→ 逐位等价 + packed-only ───────────
    const std::string paths[2] = {dep_b3, dep_2b};
    const char* names[2] = {"base-3", "2-bit"};
    for (int k = 0; k < 2; ++k)
    {
        auto b = build_ternary_mlp(eng, IN, HID, OUT);
        std::string tok_out;
        NN_CHECK(nn::load_deployment(paths[k], *b, &tok_out), "load_deployment 成功");
        CHECK(tok_out == tok, "tokenizer 随文件往返");

        bool all_packed_only = true, no_latent = true;
        for (std::size_t li = 0; li < b->num_layers(); ++li)
        {
            auto* bl = dynamic_cast<nn::BitLinear*>(&b->layer_at(li));
            CHECK(bl != nullptr, "部署载入后仍是最值层（BitLinear）");
            if (bl == nullptr) { all_packed_only = false; break; }
            if (!bl->packed_only()) all_packed_only = false;
            if (!bl->uses_packed_weights()) all_packed_only = false;
            if (bl->latent_weights().valid()) no_latent = false;   // latent 必须已释放
        }
        CHECK(all_packed_only, "三值层进入 packed-only（只读推理态）");
        CHECK(no_latent, "部署态不持有 latent（推理足迹 4.25 → 0.25 B/权重）");

        auto y_b = b->forward(Xt);
        CHECK(y_b.has_value(), "部署态 forward 成功");
        if (y_b.has_value())
        {
            const nn::Matrix Y = download(eng, *y_b);
            double mx = 0.0;
            for (std::size_t i = 0; i < Y.size(); ++i)
                mx = std::max(mx, std::fabs(static_cast<double>(Y.span()[i]) -
                                            static_cast<double>(Y_ref.span()[i])));
            std::printf("    [%s] 前向 vs 原模型最大差 = %.3e\n", names[k], mx);
            CHECK(mx == 0.0, "部署态前向与原模型**逐位相同**（同打包字 + 同 τ）");
        }

        // [3] 只读语义：反向必须报错（而不是静默用残层）
        nn::Matrix G(Y_ref.rows(), Y_ref.cols());
        G.zero();
        const nn::Tensor g = upload(eng, G);
        auto back = b->backward(g);
        CHECK(!back.has_value(), "部署态 backward fail-fast（只读推理）");
    }

    // ── [4] 尺寸账 ──────────────────────────────────────────────────────
    {
        const std::size_t sz_train = file_size(train_path);
        const std::size_t sz_b3 = file_size(dep_b3);
        const std::size_t sz_2b = file_size(dep_2b);
        std::printf("    尺寸：训练 %zu B / 部署 base-3 %zu B / 部署 2-bit %zu B\n",
                    sz_train, sz_b3, sz_2b);
        CHECK(sz_train > 0 && sz_b3 > 0 && sz_2b > 0, "三个文件都非空");
        // 权重是 f32 latent（32 bit/权重）→ 2.0 bit/权重 ⇒ 约 1/16；留 1/8 的余量
        CHECK(sz_b3 * 8u < sz_train, "部署文件 < 训练文件的 1/8（同架构同权重）");
        CHECK(sz_b3 < sz_2b, "base-3 载荷比 2-bit 载荷更小");
    }

    // ── [5] 训练文件路径零回归 ─────────────────────────────────────────
    {
        auto c = build_ternary_mlp(eng, IN, HID, OUT);
        std::string tok_out;
        NN_CHECK(nn::load_any_model(train_path, *c, &tok_out), "load_any_model(训练) 成功");
        bool any_packed_only = false, latent_ok = true;
        for (std::size_t li = 0; li < c->num_layers(); ++li)
        {
            auto* bl = dynamic_cast<nn::BitLinear*>(&c->layer_at(li));
            if (bl == nullptr) continue;
            if (bl->packed_only()) any_packed_only = true;
            if (!bl->latent_weights().valid()) latent_ok = false;
        }
        CHECK(!any_packed_only && latent_ok,
              "训练文件载入后**不是** packed-only（可继续训练，与部署文件区分）");
        auto y_c = c->forward(Xt);
        CHECK(y_c.has_value(), "训练文件载入的模型 forward 成功");

        // [5b] 两个入口不互相串：训练文件不能被当作部署文件读（magic 校验）
        auto bad = nn::peek_deployment_spec(train_path);
        CHECK(!bad.has_value(), "peek_deployment_spec(训练文件) 明确失败（magic 不匹配）");
    }

    std::filesystem::remove_all(dir);
}

// ── [6] GPT 嵌套三值层：Attention/FeedForward/Block 的层级遍历 ──────────────
// 手搭的两层 MLP 是"顶层即 BitLinear"，而真实 GPT 的三值层**藏在复合层内部**
// （Block → Attention/FF → BitLinear）——部署格式的遍历契约（自身参数在前 +
// 子层按 sub_layers 顺序）必须在这条链上一次不差，否则打包条目会错位。
std::string walk_signature(nn::Model& model)
{
    std::string sig;
    std::function<void(nn::Layer&, int)> rec = [&](nn::Layer& L, int depth) {
        auto children = L.sub_layers();
        std::size_t child_params = 0;
        for (auto* c : children) child_params += c->parameters().size();
        const std::size_t total = L.parameters().size();
        sig += std::string(static_cast<std::size_t>(depth), ' ') + L.layer_name() + ":" +
               std::to_string(total) + "-" + std::to_string(child_params) + "\n";
        for (auto* c : children) rec(*c, depth + 1);
    };
    for (std::size_t li = 0; li < model.num_layers(); ++li)
        rec(model.layer_at(li), 0);
    return sig;
}

void test_gpt_deploy(nn::ComputeEngine& eng, const char* tag)
{
    std::printf("[6] GPT 嵌套三值层：部署往返（%s）\n", tag);
    if (!eng.describe().ternary_gemm)
    {
        std::puts("    （该引擎不支持三值 GEMM → 跳过）");
        return;
    }
    nn::ModelSpec spec;
    spec.type = nn::ModelType::GPT;
    spec.d_model = 32;
    spec.num_heads = 2;
    spec.num_layers = 2;
    spec.d_ff = 64;
    spec.seq_len = 8;
    spec.vocab_size = 48;
    spec.pos_encoding = nn::PosEncodingType::RoPE;
    spec.weight_quant = nn::WeightQuant::T1_58;
    spec.activation = nn::ActivationType::ReLU2;
    spec.norm_type = nn::NormType::RMSNorm;
    spec.tie_embeddings = true;

    const std::string dir = std::string("deploy_gpt_") + tag;
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/gpt.nnqt";

    auto a_res = nn::build_gpt_model_from_spec(eng, spec);
    CHECK(a_res.has_value(), "GPT 构建成功（写端）");
    if (!a_res.has_value()) return;
    nn::Model& a = *a_res;
    seed_latents(eng, a);
    nn::Matrix X(spec.d_model, 3);   // 3 个 token
    for (std::size_t r = 0; r < spec.d_model; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            X.set_value(r, c, 0.31f * static_cast<float>((r + 2 * c) % 11) - 0.9f);
    const nn::Tensor Xt = upload(eng, X);
    auto y_a = a.forward(Xt);
    CHECK(y_a.has_value(), "写端 forward 成功");
    if (!y_a.has_value()) return;
    const nn::Matrix Y_ref = download(eng, *y_a);
    CHECK(nn::save_deployment(path, a, spec, /*base3=*/true, {}), "GPT 部署导出成功");

    auto b_res = nn::build_gpt_model_from_spec(eng, spec);
    CHECK(b_res.has_value(), "GPT 构建成功（读端）");
    if (!b_res.has_value()) return;
    nn::Model& b = *b_res;
    // 结构契约：两端的遍历签名必须逐字符相同（否则打包条目必错位）
    const std::string sig_a = walk_signature(a);
    const std::string sig_b = walk_signature(b);
    if (sig_a != sig_b)
    {
        std::printf("---- 写端 walk ----\n%s---- 读端 walk ----\n%s", sig_a.c_str(),
                    sig_b.c_str());
    }
    CHECK(sig_a == sig_b, "遍历签名一致（自身参数 + 子层顺序）");
    {
        auto lr = nn::load_deployment(path, b, nullptr);
        if (!lr)
            std::printf("    load_deployment 失败: %s\n", lr.error().message.c_str());
        CHECK(lr.has_value(), "GPT 部署载入成功");
    }

    std::size_t packed_layers = 0;
    std::function<void(nn::Layer&)> count = [&](nn::Layer& L) {
        if (auto* bl = dynamic_cast<nn::BitLinear*>(&L); bl && bl->packed_only())
            ++packed_layers;
        for (auto* c : L.sub_layers()) count(*c);
    };
    for (std::size_t li = 0; li < b.num_layers(); ++li) count(b.layer_at(li));
    std::printf("    嵌套三值层进入 packed-only：%zu 个\n", packed_layers);
    CHECK(packed_layers > 0, "GPT 的嵌套三值层确实被部署态接管");

    auto y_b = b.forward(Xt);
    CHECK(y_b.has_value(), "读端 forward 成功");
    if (y_b.has_value())
    {
        const nn::Matrix Y = download(eng, *y_b);
        double mx = 0.0;
        for (std::size_t i = 0; i < Y.size(); ++i)
            mx = std::max(mx, std::fabs(static_cast<double>(Y.span()[i]) -
                                        static_cast<double>(Y_ref.span()[i])));
        std::printf("    GPT 前向 vs 原模型最大差 = %.3e\n", mx);
        CHECK(mx == 0.0, "GPT 部署态前向与原模型逐位相同");
    }
    std::filesystem::remove_all(dir);
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
        test_deploy(**eng_r, "GPU");
        test_gpt_deploy(**eng_r, "GPU");
    }
    else
    {
        nn::CpuEngine eng;
        test_deploy(eng, "CPU");
        test_gpt_deploy(eng, "CPU");
    }

    if (g_failures == 0)
        std::puts("ternary_deploy_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
