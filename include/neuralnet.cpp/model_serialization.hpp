#ifndef NN_MODEL_SERIALIZATION_HPP
#define NN_MODEL_SERIALIZATION_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  model_io.hpp — 模型二进制序列化
//
//  职责：将 Model + ModelSpec 保存为二进制文件，或从文件加载参数。
//  设计：
//    - 使用固定宽度整数 (uint64_t) 保证跨平台/跨位宽兼容性
//    - 纯库函数，无 I/O 副作用（调用方自行决定是否打印日志）
//    - 向后兼容 V1 格式读取，新文件统一写入 V2 格式
//    - 使用 Result<T> 返回错误，不抛异常
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "config.hpp"
#include "model_spec.hpp"
#include "compute_layer.hpp"
#include "model_container.hpp"

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  二进制文件格式
//
//  Version 3 (当前格式，含精度标记 + 可选 tokenizer):
//    [magic 4B] [version=3 4B] [precision 1B] [model_type 4B]
//    [spec data...]
//    [param matrices...]
//    [tokenizer_len 8B] [tokenizer_json...]     ← V3 新增
//
//  Version 2 (含架构规格):
//    [magic 4B] [version=2 4B] [model_type 4B] [spec data...] [matrices...]
//
//  Version 1 (仅参数):
//    [magic 4B] [version=1 4B] [matrices...]
//
//  precision 字节: 0 = f32, 1 = f64（用于校验保存时与加载时的 Scalar 类型一致）
//  tokenizer_len = 0 表示未嵌入分词器（向后兼容）
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr uint32_t MODEL_MAGIC    = 0x4E4E4E4E;  // "NNNN"
inline constexpr uint32_t MODEL_VERSION  = 3;

// ── 精度标记（写入文件头，加载时校验） ──────────────────────────────────
// 0 = float (f32), 1 = double (f64)
inline constexpr uint8_t PRECISION_TAG = sizeof(Scalar) == 4 ? 0 : 1;
static_assert(sizeof(Scalar) == 4 || sizeof(Scalar) == 8,
              "Scalar must be float (4 bytes) or double (8 bytes)");

// ═══════════════════════════════════════════════════════════════════════════
//  detail — 内部读写工具
//
//  所有函数使用固定宽度整数 (uint64_t) 替代 std::size_t，保证跨平台兼容。
//  所有函数返回 Result<T>，失败时返回 Error 错误信息。
// ═══════════════════════════════════════════════════════════════════════════

namespace detail
{

// ── 基础类型读写 ──────────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_u32(std::ofstream &ofs, uint32_t v)
{
    ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
    if (!ofs)
        return std::unexpected(Error{"Write error while writing u32"});
    return {};
}

[[nodiscard]] inline Result<uint32_t> read_u32(std::ifstream &ifs)
{
    uint32_t v;
    ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!ifs)
        return std::unexpected(Error{"Unexpected end of file while reading u32"});
    return v;
}

[[nodiscard]] inline Result<void> write_u64(std::ofstream &ofs, uint64_t v)
{
    ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
    if (!ofs)
        return std::unexpected(Error{"Write error while writing u64"});
    return {};
}

[[nodiscard]] inline Result<uint64_t> read_u64(std::ifstream &ifs)
{
    uint64_t v;
    ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!ifs)
        return std::unexpected(Error{"Unexpected end of file while reading u64"});
    return v;
}

// ── 矩阵读写 ──────────────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_matrix(std::ofstream &ofs, const Matrix &m)
{
    if (auto r = write_u64(ofs, static_cast<uint64_t>(m.rows())); !r)
        return std::unexpected(r.error());
    if (auto r = write_u64(ofs, static_cast<uint64_t>(m.cols())); !r)
        return std::unexpected(r.error());
    const auto s = m.span();
    ofs.write(reinterpret_cast<const char *>(s.data()),
              static_cast<std::streamsize>(s.size() * sizeof(Scalar)));
    if (!ofs)
        return std::unexpected(Error{"Write error while writing matrix data"});
    return {};
}

[[nodiscard]] inline Result<void> read_matrix(std::ifstream &ifs, Matrix &m)
{
    auto rows_r = read_u64(ifs);
    if (!rows_r) return std::unexpected(rows_r.error());
    auto cols_r = read_u64(ifs);
    if (!cols_r) return std::unexpected(cols_r.error());
    const auto rows = static_cast<std::size_t>(*rows_r);
    const auto cols = static_cast<std::size_t>(*cols_r);

    if (rows != m.rows() || cols != m.cols())
    {
        return std::unexpected(Error{
            "Matrix shape mismatch: expected (" + std::to_string(m.rows())
            + ", " + std::to_string(m.cols()) + "), got ("
            + std::to_string(rows) + ", " + std::to_string(cols) + ")"});
    }

    auto s = m.span();
    ifs.read(reinterpret_cast<char *>(s.data()),
             static_cast<std::streamsize>(s.size() * sizeof(Scalar)));
    if (!ifs)
        return std::unexpected(Error{"Unexpected end of file while reading matrix data"});
    return {};
}

// ── ModelSpec 序列化 ──────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_spec(std::ofstream &ofs, const ModelSpec &spec)
{
    if (auto r = write_u32(ofs, static_cast<uint32_t>(spec.type)); !r)
        return std::unexpected(r.error());

    switch (spec.type)
    {
    case ModelType::MLP:
    {
        if (auto r = write_u32(ofs, static_cast<uint32_t>(spec.layer_dims.size())); !r)
            return std::unexpected(r.error());
        for (auto d : spec.layer_dims)
            if (auto r = write_u64(ofs, static_cast<uint64_t>(d)); !r)
                return std::unexpected(r.error());
        break;
    }

    case ModelType::Transformer:
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.d_model)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.num_heads)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.d_ff)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.num_layers)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.patch_size)); !r)
            return std::unexpected(r.error());
        break;

    case ModelType::GPT:
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.vocab_size)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.d_model)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.seq_len)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.num_heads)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.d_ff)); !r)
            return std::unexpected(r.error());
        if (auto r = write_u64(ofs, static_cast<uint64_t>(spec.num_layers)); !r)
            return std::unexpected(r.error());
        break;

    default:
        return std::unexpected(Error{"Cannot write unknown ModelType: "
                           + std::to_string(static_cast<uint32_t>(spec.type))});
    }
    return {};
}

[[nodiscard]] inline Result<ModelSpec> read_spec(std::ifstream &ifs)
{
    ModelSpec spec;
    auto type_r = read_u32(ifs);
    if (!type_r) return std::unexpected(type_r.error());
    spec.type = static_cast<ModelType>(*type_r);

    switch (spec.type)
    {
    case ModelType::MLP:
    {
        auto nd_r = read_u32(ifs);
        if (!nd_r) return std::unexpected(nd_r.error());
        const auto nd = *nd_r;
        spec.layer_dims.resize(nd);
        for (uint32_t i = 0; i < nd; ++i)
        {
            auto d_r = read_u64(ifs);
            if (!d_r) return std::unexpected(d_r.error());
            spec.layer_dims[i] = static_cast<std::size_t>(*d_r);
        }
        break;
    }

    case ModelType::Transformer:
    {
        auto v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.d_model    = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.num_heads  = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.d_ff       = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.num_layers = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.patch_size = static_cast<std::size_t>(*v);
        break;
    }

    case ModelType::GPT:
    {
        auto v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.vocab_size = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.d_model    = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.seq_len    = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.num_heads  = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.d_ff       = static_cast<std::size_t>(*v);
        v = read_u64(ifs); if (!v) return std::unexpected(v.error());
        spec.num_layers = static_cast<std::size_t>(*v);
        break;
    }

    default:
        return std::unexpected(Error{"Unknown ModelType in file: "
                           + std::to_string(static_cast<uint32_t>(spec.type))});
    }

    return spec;
}

// ── 文件头读写 ──────────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_header(std::ofstream &ofs)
{
    if (auto r = write_u32(ofs, MODEL_MAGIC); !r)
        return std::unexpected(r.error());
    if (auto r = write_u32(ofs, MODEL_VERSION); !r)
        return std::unexpected(r.error());
    ofs.write(reinterpret_cast<const char *>(&PRECISION_TAG), 1);
    if (!ofs)
        return std::unexpected(Error{"Write error while writing precision tag"});
    return {};
}

// 返回读到的 version，同时校验 magic number 和精度
[[nodiscard]] inline Result<uint32_t> read_and_validate_header(std::ifstream &ifs)
{
    auto magic_r = read_u32(ifs);
    if (!magic_r) return std::unexpected(magic_r.error());
    if (*magic_r != MODEL_MAGIC)
        return std::unexpected(Error{"Invalid model file: bad magic number (0x"
                           + std::to_string(*magic_r) + ")"});

    auto version_r = read_u32(ifs);
    if (!version_r) return std::unexpected(version_r.error());
    if (*version_r == 0 || *version_r > MODEL_VERSION)
        return std::unexpected(Error{"Unsupported model file version: "
                           + std::to_string(*version_r)});

    // V3: 读取并校验精度标记
    if (*version_r >= 3)
    {
        uint8_t precision_tag;
        ifs.read(reinterpret_cast<char *>(&precision_tag), 1);
        if (!ifs)
            return std::unexpected(Error{"Unexpected end: missing precision tag"});
        if (precision_tag != PRECISION_TAG)
            return std::unexpected(Error{
                "Precision mismatch: file uses " + std::string(precision_tag == 0 ? "f32" : "f64")
                + ", but build uses " + std::string(PRECISION_TAG == 0 ? "f32" : "f64")});
    }

    return *version_r;
}

// ── Tokenizer JSON 读写（V3 新增） ──────────────────────────────────

[[nodiscard]] inline Result<void> write_tokenizer(std::ofstream &ofs, const std::string &json)
{
    if (auto r = write_u64(ofs, static_cast<uint64_t>(json.size())); !r)
        return std::unexpected(r.error());
    if (!json.empty())
    {
        ofs.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!ofs)
            return std::unexpected(Error{"Write error while writing tokenizer"});
    }
    return {};
}

[[nodiscard]] inline Result<std::string> read_tokenizer(std::ifstream &ifs)
{
    auto len_r = read_u64(ifs);
    if (!len_r) return std::unexpected(len_r.error());
    const auto len = static_cast<std::size_t>(*len_r);
    if (len == 0) return std::string{};  // 未嵌入

    std::string json(len, '\0');
    ifs.read(json.data(), static_cast<std::streamsize>(len));
    if (!ifs)
        return std::unexpected(Error{"Unexpected end while reading tokenizer data"});
    return json;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
//  公开 API
//
//  save_model      — 保存 Model + ModelSpec（可选嵌入 tokenizer JSON）
//  load_model      — 从文件加载参数，返回嵌入的 tokenizer JSON（如有）
//  peek_model_spec — 只读文件头，返回 ModelSpec（不读参数）
//
//  向后兼容 V1/V2/V3 读取，写入统一为 V3。
// ═══════════════════════════════════════════════════════════════════════════

// ── 保存模型（V3 格式，含精度标记 + 可选 tokenizer） ────────────────────
// 新架构：参数为 Tensor*，通过 engine.to_matrix 下载到 CPU Matrix 后写入。
[[nodiscard]] inline Result<void> save_model(const std::string &filename,
    Model &model, const ModelSpec &spec, const std::string &tokenizer_json = {})
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        return std::unexpected(Error{"Cannot open file for writing: " + filename});

    if (auto r = detail::write_header(ofs); !r)
        return std::unexpected(r.error());
    if (auto r = detail::write_spec(ofs, spec); !r)
        return std::unexpected(r.error());

    auto& engine = model.engine();
    auto params = model.parameters();
    for (auto *p_tensor : params)
    {
        auto m = engine.to_matrix(*p_tensor);
        if (!m) return std::unexpected(m.error());
        if (auto r = detail::write_matrix(ofs, *m); !r)
            return std::unexpected(r.error());
    }

    // V3: 写入 tokenizer JSON（长度前缀，0 表示无）
    if (auto r = detail::write_tokenizer(ofs, tokenizer_json); !r)
        return std::unexpected(r.error());

    if (!ofs)
        return std::unexpected(Error{"Write error while saving model to: " + filename});
    return {};
}

// ── 读取模型规格（只读头部，不读参数/tokenizer） ──────────────────────────
//    V1 文件返回 ModelType::Unknown
[[nodiscard]] inline Result<ModelSpec> peek_model_spec(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());

    // V2/V3 有规格头
    if (*version_r >= 2)
        return detail::read_spec(ifs);

    // V1 文件没有规格信息
    return ModelSpec{ModelType::Unknown, {}, 0, 0, 0, 0, 0, 0, 0};
}

// ── 加载参数 + tokenizer（兼容 V1/V2/V3） ─────────────────────────────────
//    返回嵌入的 tokenizer JSON 字符串（空串 = 未嵌入或旧格式）
// 新架构：先 read_matrix 读入临时 CPU Matrix，再通过 engine.copy_from
// 上传到参数 Tensor（CPU 拷贝 / GPU 上传由引擎实现决定）。
[[nodiscard]] inline Result<std::string> load_model(const std::string &filename, Model &model)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());
    const auto version = *version_r;

    // V2/V3: 跳过规格头（规格已隐含在构建好的 model 中）
    if (version >= 2)
    {
        auto spec_r = detail::read_spec(ifs);
        if (!spec_r) return std::unexpected(spec_r.error());
    }

    auto& engine = model.engine();
    auto params = model.parameters();
    for (auto *p_tensor : params)
    {
        // 先读入临时 Matrix（按参数 Tensor 的形状）
        Matrix tmp(p_tensor->rows(), p_tensor->cols());
        if (auto r = detail::read_matrix(ifs, tmp); !r)
            return std::unexpected(r.error());
        // 通过 engine 上传到参数 Tensor
        if (auto r = engine.copy_from(*p_tensor, tmp); !r)
            return std::unexpected(r.error());
    }

    // V3: 读取嵌入的 tokenizer JSON
    std::string tokenizer_json;
    if (version >= 3)
    {
        auto tok_r = detail::read_tokenizer(ifs);
        if (!tok_r) return std::unexpected(tok_r.error());
        tokenizer_json = std::move(*tok_r);
    }

    if (!ifs)
        return std::unexpected(Error{"Read error while loading model from: " + filename});
    return tokenizer_json;
}

} // namespace nn

#endif // NN_MODEL_SERIALIZATION_HPP