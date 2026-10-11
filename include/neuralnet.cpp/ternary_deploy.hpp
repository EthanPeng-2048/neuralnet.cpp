#pragma once

// ── ternary_deploy.hpp — 三值权重的**部署格式**（P2）──────────────────────
//
// **动机**（docs/development/21 §4.11 ②、§5 的 P2 剩余项）：训练/检查点文件里
// 三值层存的是 f32 latent（32 bit/权重），而推理根本用不到它 —— 推理只需要
// (打包字, 逐行 τ)。推理态因此一直背着 4 B/权重 的 latent（10 B/权重 → 6 B/权重
// 的"三值推理足迹"里最大的那一项）。本头提供**只读推理**的独立文件格式：
//
//   [magic "NNQT" u32][version u32][precision tag u8][spec kvrec]
//   [每个 layer 的每个参数一条：tag u8]
//       · tag 0 / 2（f32 / f16 矩阵）：[rows u64][cols u64][数据]（与 v5/v6 逐字节同款）
//       · tag 4（打包三值；`precision_tag(T1_58)` 早就为它留了号）：
//         [encoding u8][rows u64][k u64][τ f32 × rows][payload]，其中
//           encoding 0 = 2-bit 容器（运行期编码，ceil(K/16) 字/行 = 0.25 B/权重）
//           encoding 1 = base-3（部署编码，5 位/字节 = ceil(K/5) B/行 ≈ 1.6 bpw）
//   [extra state 条目（同 tag 0/2）][tokenizer_len u64][tokenizer 数据]
//
// **为什么独立 magic 而不是升 MODEL_VERSION**：① 版本号已抢号（v6 = .nnvocab 镜像，
// roadmap 的 checkpoint 优化器状态还排在后面）；② 部署文件是**只读**语义（载入后
// `BitLinear` 进 packed-only，没有 latent/梯度），与"训练检查点"混在一个版本里会让
// `--resume` 误载一个不能训练的文件。独立 magic 让两类文件在**读第一行**时就能分开，
// 既有模型文件的读写路径零改动。
//
// **载入语义**：三值层 → `BitLinear::load_packed_weights`（packed-only：释放 latent /
// 梯度 / 稠密缓冲，权重足迹 0.25 B/权重 + 4 B/行）；其余参数（偏置 / 非三值层 / extra
// state）走普通矩阵路径。τ **随文件保存并按位载入** —— 重新量化会得到
// `absmean(trit)·τ` 的比例差异，只有载入 τ 才能逐位复现训练时的有效权重 τ∘W。
// ─────────────────────────────────────────────────────────────────────────

#include <bit>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "compute_engine.hpp"
#include "compute_layer_bitlinear.hpp"   // 部署态 = 三值层专属（packed-only 载入）
#include "model_container.hpp"
#include "model_serialization.hpp"       // detail:: 读写助手（spec kvrec / 矩阵 / tokenizer）
#include "model_spec.hpp"
#include "precision.hpp"
#include "ternary_pack.hpp"

namespace nn
{

// "NNQT"：小端字节序写成 u32 → 文件首 4 字节 = 'N','N','Q','T'
inline constexpr std::uint32_t DEPLOY_MAGIC   = 0x54514E4Eu;
inline constexpr std::uint32_t DEPLOY_VERSION = 1u;

namespace detail_deploy
{

// encoding 字节（打包条目内）
inline constexpr std::uint8_t kEnc2Bit  = 0u;
inline constexpr std::uint8_t kEncBase3 = 1u;

[[nodiscard]] inline std::size_t payload_bytes_for(std::uint8_t enc, std::size_t rows,
                                                   std::size_t k) noexcept
{
    const std::size_t per_row = (enc == kEncBase3)
        ? nn::trit_pack::base3_bytes_per_row(k)
        : nn::trit_pack::bytes_per_row(k);
    return rows * per_row;
}

// 单行：2-bit 容器字 → 落盘 payload（编码由 enc 选）
inline void encode_row(std::uint8_t enc, const std::uint32_t* words, std::size_t k,
                       std::uint8_t* out) noexcept
{
    if (enc == kEncBase3)
    {
        nn::trit_pack::pack_row_base3(words, k, out);
        return;
    }
    std::memcpy(out, words, nn::trit_pack::bytes_per_row(k));
}

// 单行：落盘 payload → 2-bit 容器字（**运行期一律转回 2-bit**，内核只认它）
inline void decode_row(std::uint8_t enc, const std::uint8_t* in, std::size_t k,
                       std::uint32_t* words) noexcept
{
    if (enc == kEncBase3)
    {
        nn::trit_pack::unpack_row_base3(in, k, words);
        return;
    }
    std::memcpy(words, in, nn::trit_pack::bytes_per_row(k));
}

// 读 n 个字节到 buf（binary；EOF/短读 → 错误）
[[nodiscard]] inline Result<void> read_blob(std::ifstream& ifs, void* buf, std::size_t n)
{
    if (n == 0) return {};
    ifs.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(n));
    if (!ifs)
        NN_FAIL("部署文件读取失败（payload 短读）");
    return {};
}

// ── 完整参数遍历（部署格式的"逐个参数"口径）──────────────────────────────
// 顺序必须与 `Model::parameters()` **完全一致**（两类文件共享这一个契约）：
//   本层"自身参数"（`parameters()` 的前 own 个）→ 按 `sub_layers()` 顺序递归子层，
//   其中 own = 本层参数总数 − 子层参数数。
// 为什么不是"直接遍历 Model::parameters()"：那条路径只给得到 Tensor，给不到
// **拥有它的层** —— 而部署格式要把打包字 + τ 落到具体 BitLinear（并把它切成
// packed-only）。GPT 的三值层藏在 Block/Attention/FeedForward 内部，只有递归才够。
// 不一致（复合层没实现 sub_layers / own 为负）→ fail-fast，绝不静默错位。
template <typename Fn>
[[nodiscard]] Result<void> visit_params_rec(Layer& layer, Fn&& fn)
{
    auto children = layer.sub_layers();
    std::size_t child_params = 0;
    for (auto* c : children)
    {
        if (c == nullptr)
            NN_FAIL(std::string("sub_layers() 返回空指针（层=") + layer.layer_name() + "）");
        child_params += c->parameters().size();
    }
    // 本层参数**一次取出**再交给回调：回调可能把本层切成 packed-only
    // （部署载入），那之后再调 `parameters()` 就是非法（fail-fast）。
    // 取值的这一刻，本层与所有子层都还没被改动。
    auto all_refs = layer.parameters();
    const std::size_t total = all_refs.size();
    if (child_params > total)
        NN_FAIL(std::string("子层参数数 > 本层参数数（复合层未实现 sub_layers？层=") +
                layer.layer_name() + "）");
    const std::size_t own = total - child_params;
    std::vector<TensorRef> own_refs(all_refs.begin(), all_refs.begin() +
                                    static_cast<std::ptrdiff_t>(own));
    for (std::size_t i = 0; i < own; ++i)
    {
        NN_TRY(r, fn(layer, own_refs, i));
    }
    for (auto* c : children)
    {
        NN_TRY(r2, visit_params_rec(*c, fn));
    }
    return {};
}

// 顶层：对模型的每一层递归（`Model::layer_at` 顺序 = `Model::parameters()` 的顺序）
template <typename Fn>
[[nodiscard]] Result<void> visit_model_params(Model& model, Fn&& fn)
{
    for (std::size_t li = 0; li < model.num_layers(); ++li)
    {
        NN_TRY(r, visit_params_rec(model.layer_at(li), fn));
    }
    return {};
}

} // namespace detail_deploy

// ── 是不是部署文件（只读前 4 字节 magic）──────────────────────────────────
[[nodiscard]] inline Result<bool> is_deployment_file(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto m = detail::read_bytes<std::uint32_t>(ifs);
    if (!m)
        NN_FAIL("文件过短（读不到 magic）: " + filename);
    return *m == DEPLOY_MAGIC;
}

// ── 保存部署文件 ─────────────────────────────────────────────────────────
// base3 = true 用 base-3 载荷（1.6 bpw），false 用 2-bit 容器（2.0 bpw）。
// tokenizer_data 与训练文件同款（v6 的 .nnvocab 镜像字节）。
[[nodiscard]] inline Result<void> save_deployment(const std::string& filename, Model& model,
                                                 const ModelSpec& spec, bool base3 = true,
                                                 const std::string& tokenizer_data = {})
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        NN_FAIL("Cannot open file for writing: " + filename);
    NN_TRY(w0, detail::write_bytes<std::uint32_t>(ofs, DEPLOY_MAGIC));
    NN_TRY(w1, detail::write_bytes<std::uint32_t>(ofs, DEPLOY_VERSION));
    NN_TRY(w2, detail::write_bytes<std::uint8_t>(ofs, PRECISION_TAG));
    NN_TRY(w3, detail::write_spec_header(ofs, spec));

    const std::uint8_t enc = base3 ? detail_deploy::kEncBase3 : detail_deploy::kEnc2Bit;
    ComputeEngine& engine = model.engine();

    // 逐参数（递归到叶子；顺序 = `Model::parameters()`，见 visit_params_rec 的契约）
    std::size_t visited = 0;
    auto write_param = [&](Layer& L, const std::vector<TensorRef>& own,
                           std::size_t pi) -> Result<void>
    {
        ++visited;
        auto* bl = dynamic_cast<BitLinear*>(&L);
        {
            // 三值层的权重（out×in）→ 打包条目；其余（偏置等）走普通矩阵
            const bool is_ternary_w = (bl != nullptr) && pi == 0;
            if (!is_ternary_w)
            {
                auto param = own[pi];
                Precision p = param.get().precision();
                auto m32 = engine.to_matrix(param, Precision::F32);
                NN_TRY_CHECK(m32);
                if (p == Precision::F16)
                {
                    MatrixT<Precision::F16> m16(m32->rows(), m32->cols());
                    const auto src = m32->span();
                    auto dst = m16.span();
                    for (std::size_t i = 0; i < src.size(); ++i) dst[i] = src[i];
                    NN_TRY(wm, detail::write_matrix_fp16(ofs, m16));
                }
                else
                {
                    NN_TRY(wm, detail::write_matrix_v5(ofs, *m32, Precision::F32));
                }
                return {};
            }

            // 打包条目：确保量化派生物有效（部署格式的收益全在这条分支上）
            if (!engine.supports_ternary_gemm())
                NN_FAIL("save_deployment: 该引擎不支持三值 GEMM，无法导出打包权重"
                        "（部署文件要求打包形态）");
            if (!bl->uses_packed_weights()) bl->set_ternary_pack_enabled(true);
            NN_TRY(q, bl->ensure_quantized());
            auto cont_r = engine.to_matrix(bl->quantized_weights_packed(), Precision::F32);
            NN_TRY_CHECK(cont_r);
            auto tau_r = engine.to_matrix(bl->row_scales(), Precision::F32);
            NN_TRY_CHECK(tau_r);
            const Matrix& cont = *cont_r;
            const Matrix& tau = *tau_r;
            const std::size_t rows = cont.rows();
            const std::size_t k = bl->in_features();
            const std::size_t wpr = nn::trit_pack::words_per_row(k);
            if (cont.cols() != wpr || tau.rows() != rows || tau.cols() != 1)
                NN_FAIL("save_deployment: 打包派生物形状与层不一致");

            NN_TRY(t0, detail::write_bytes<std::uint8_t>(ofs, precision_tag(Precision::T1_58)));
            NN_TRY(t1, detail::write_bytes<std::uint8_t>(ofs, enc));
            NN_TRY(t2, detail::write_fields(ofs, static_cast<std::uint64_t>(rows),
                                            static_cast<std::uint64_t>(k)));
            const auto tau_span = tau.span();
            if (!nn::write_pod_span(ofs, tau_span))
                NN_FAIL("save_deployment: τ 写入失败");

            // payload：逐行从容器字编码（宿主侧，峰值额外内存 = 1 行）
            std::vector<std::uint32_t> words(wpr);
            std::vector<std::uint8_t> payload(detail_deploy::payload_bytes_for(enc, 1, k));
            const auto cont_span = cont.span();
            for (std::size_t r = 0; r < rows; ++r)
            {
                for (std::size_t i = 0; i < wpr; ++i)
                    words[i] = std::bit_cast<std::uint32_t>(cont_span[r * wpr + i]);
                detail_deploy::encode_row(enc, words.data(), k, payload.data());
                if (!nn::write_pod_span(
                        ofs, std::span<const std::uint8_t>(payload.data(), payload.size())))
                    NN_FAIL("save_deployment: payload 写入失败");
            }
            return {};
        }
    };
    NN_TRY(v0, detail_deploy::visit_model_params(model, write_param));
    // 覆盖自检：遍历到的参数数必须与 `Model::parameters()` 一致（防结构漏接）
    const std::size_t total_params = model.parameters().size();
    if (visited != total_params)
        NN_FAIL("save_deployment: 参数遍历覆盖 " + std::to_string(visited) + " 个，模型有 " +
                std::to_string(total_params) + " 个（复合层未实现 sub_layers？）");

    // 非可学习状态（BatchNorm running 统计等）：与 v6 同款（普通矩阵）
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        auto m32 = engine.to_matrix(e_tensor, Precision::F32);
        NN_TRY_CHECK(m32);
        NN_TRY(we, detail::write_matrix_v5(ofs, *m32, e_tensor.get().precision()));
    }

    NN_TRY(wt, detail::write_tokenizer(ofs, tokenizer_data));
    return {};
}

// ── 载入部署文件 ─────────────────────────────────────────────────────────
// 模型必须已按 spec 建好（`peek_deployment_spec` + 工厂）；三值层进 packed-only。
[[nodiscard]] inline Result<void> load_deployment(const std::string& filename, Model& model,
                                                 std::string* tokenizer_out = nullptr)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto magic_r = detail::read_bytes<std::uint32_t>(ifs);
    NN_TRY_CHECK(magic_r);
    if (*magic_r != DEPLOY_MAGIC)
        NN_FAIL("不是部署文件（magic 不匹配）: " + filename);
    auto ver_r = detail::read_bytes<std::uint32_t>(ifs);
    NN_TRY_CHECK(ver_r);
    if (*ver_r != DEPLOY_VERSION)
        NN_FAIL("部署文件版本不支持: " + std::to_string(*ver_r));
    auto pt_r = detail::read_bytes<std::uint8_t>(ifs);
    NN_TRY_CHECK(pt_r);
    if (*pt_r != PRECISION_TAG)
        NN_FAIL("部署文件的 Scalar 精度与本构建不一致（f32/f64 混用）");
    auto spec_r = detail::read_spec_header(ifs, DEPLOY_VERSION);
    NN_TRY_CHECK(spec_r);
    if (auto stored = model.spec(); stored && !spec_matches(*stored, *spec_r))
        NN_FAIL("部署文件与模型架构不匹配：" + spec_summary(*spec_r) + " vs " +
                spec_summary(*stored));

    ComputeEngine& engine = model.engine();
    // 载入前先记下参数总数（packed-only 之后 `parameters()` 会 fail-fast）
    const std::size_t total_params = model.parameters().size();
    std::size_t visited = 0;
    auto read_param = [&](Layer& L, const std::vector<TensorRef>& own,
                          std::size_t pi) -> Result<void>
    {
        ++visited;
        auto tag_r = detail::read_bytes<std::uint8_t>(ifs);
        NN_TRY_CHECK(tag_r);
        const std::uint8_t tag = *tag_r;

        if (tag != precision_tag(Precision::T1_58))
        {
            // 普通矩阵条目：退回 1 字节 tag 后交给 v5 读取器（同一份实现）
            ifs.seekg(-1, std::ios::cur);
            auto m_r = detail::read_matrix_v5(ifs);
            NN_TRY_CHECK(m_r);
            // f32 宿主值（f16 数据读入时已精确升位）→ copy_from（同 v6 口径）
            NN_TRY(cm, engine.copy_from(own[pi], m_r->second));
            return {};
        }

        {
            auto* bl = dynamic_cast<BitLinear*>(&L);
            if (bl == nullptr || pi != 0)
                NN_FAIL(std::string("部署文件与模型不匹配：打包条目落在非三值层/非权重槽（层=") +
                        L.layer_name() + "，参数索引=" + std::to_string(pi) + "）");
            auto enc_r = detail::read_bytes<std::uint8_t>(ifs);
            NN_TRY_CHECK(enc_r);
            const std::uint8_t enc = *enc_r;
            auto dims_r = detail::read_fields<std::uint64_t, std::uint64_t>(ifs);
            NN_TRY_CHECK(dims_r);
            const std::size_t rows = static_cast<std::size_t>(std::get<0>(*dims_r));
            const std::size_t k = static_cast<std::size_t>(std::get<1>(*dims_r));
            if (rows != bl->out_features() || k != bl->in_features())
                NN_FAIL("部署文件与模型不匹配：打包权重形状 " + std::to_string(rows) + "x" +
                        std::to_string(k) + " vs 层 " + std::to_string(bl->out_features()) +
                        "x" + std::to_string(bl->in_features()));

            Matrix tau_m(rows, 1);
            if (!nn::read_pod_span(ifs, tau_m.span()))
                NN_FAIL("部署文件读取失败（τ 短读）");

            const std::size_t wpr = nn::trit_pack::words_per_row(k);
            const std::size_t per_row = detail_deploy::payload_bytes_for(enc, 1, k);
            Matrix cont(rows, wpr);   // f32 容器：每元素 = 一个 32-bit 打包字
            std::vector<std::uint8_t> payload(per_row);
            std::vector<std::uint32_t> words(wpr);
            auto cont_span = cont.span();
            for (std::size_t r = 0; r < rows; ++r)
            {
                NN_TRY(pb, detail_deploy::read_blob(ifs, payload.data(), per_row));
                detail_deploy::decode_row(enc, payload.data(), k, words.data());
                for (std::size_t i = 0; i < wpr; ++i)
                    cont_span[r * wpr + i] = std::bit_cast<float>(words[i]);
            }

            auto cont_t = engine.from_matrix(std::move(cont), Precision::F32);
            NN_TRY_CHECK(cont_t);
            auto tau_t = engine.from_matrix(std::move(tau_m), Precision::F32);
            NN_TRY_CHECK(tau_t);
            NN_TRY(lw, bl->load_packed_weights(std::move(*cont_t), std::move(*tau_t)));
        }
        return {};
    };
    NN_TRY(v0, detail_deploy::visit_model_params(model, read_param));
    if (visited != total_params)
        NN_FAIL("load_deployment: 参数遍历覆盖 " + std::to_string(visited) + " 个，模型有 " +
                std::to_string(total_params) + " 个（复合层未实现 sub_layers？）");

    // 非可学习状态
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        if (ifs.peek() == EOF) break;
        auto m_r = detail::read_matrix_v5(ifs);
        NN_TRY_CHECK(m_r);
        NN_TRY(cm, engine.copy_from(e_tensor, m_r->second));
    }

    if (tokenizer_out)
    {
        auto tok_r = detail::read_tokenizer(ifs);
        NN_TRY_CHECK(tok_r);
        *tokenizer_out = std::move(*tok_r);
    }
    return {};
}

// ── 只读 spec / 统一入口（推理 CLI 用：按 magic 分派，两类文件同一入口）────
[[nodiscard]] inline Result<ModelSpec> peek_deployment_spec(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto magic_r = detail::read_bytes<std::uint32_t>(ifs);
    NN_TRY_CHECK(magic_r);
    if (*magic_r != DEPLOY_MAGIC)
        NN_FAIL("不是部署文件（magic 不匹配）: " + filename);
    auto ver_r = detail::read_bytes<std::uint32_t>(ifs);
    NN_TRY_CHECK(ver_r);
    auto pt_r = detail::read_bytes<std::uint8_t>(ifs);
    NN_TRY_CHECK(pt_r);
    return detail::read_spec_header(ifs, DEPLOY_VERSION);
}

[[nodiscard]] inline Result<ModelSpec> peek_any_model_spec(const std::string& filename)
{
    auto is_dep = is_deployment_file(filename);
    NN_TRY_CHECK(is_dep);
    if (*is_dep) return peek_deployment_spec(filename);
    return peek_model_spec(filename);
}

[[nodiscard]] inline Result<void> load_any_model(const std::string& filename, Model& model,
                                                std::string* tokenizer_out = nullptr)
{
    auto is_dep = is_deployment_file(filename);
    NN_TRY_CHECK(is_dep);
    if (*is_dep) return load_deployment(filename, model, tokenizer_out);
    // `load_model` 的返回值是嵌入的 tokenizer（v5 老签名），此处只需把参数载好
    auto tok_r = load_model(filename, model);
    NN_TRY_CHECK(tok_r);
    if (tokenizer_out) *tokenizer_out = std::move(*tok_r);
    return {};
}

} // namespace nn

