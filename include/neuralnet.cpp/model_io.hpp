#ifndef MODEL_IO_HPP
#define MODEL_IO_HPP

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

#include "nn_config.hpp"
#include "layer.hpp"
#include "model.hpp"

namespace nn
{



// ═══════════════════════════════════════════════════════════════════════════
//  ModelSpec — 模型架构描述，嵌入到 V2 文件头部
//
//  加载时可先调用 peek_model_spec() 读取规格，再据此构建模型。
// ═══════════════════════════════════════════════════════════════════════════

enum class ModelType : uint32_t
{
    Unknown     = 0,
    MLP         = 1,
    Transformer = 2,
    GPT         = 3,
};

struct ModelSpec
{
    ModelType type = ModelType::Unknown;

    // ── MLP ──
    std::vector<std::size_t> layer_dims;

    // ── Transformer (MNIST ViT) ──
    std::size_t d_model    = 0;
    std::size_t num_heads  = 0;
    std::size_t d_ff       = 0;
    std::size_t num_layers = 0;
    std::size_t patch_size = 0;

    // ── GPT ──
    std::size_t vocab_size = 0;
    std::size_t seq_len    = 0;

    [[nodiscard]] bool is_mlp()         const noexcept { return type == ModelType::MLP; }
    [[nodiscard]] bool is_transformer() const noexcept { return type == ModelType::Transformer; }
    [[nodiscard]] bool is_gpt()         const noexcept { return type == ModelType::GPT; }
};

// ═══════════════════════════════════════════════════════════════════════════
//  二进制文件格式
//
//  Version 1 (旧格式，仅参数):
//    [magic 4B] [version=1 4B] [矩阵数据...]
//
//  Version 2 (当前格式，含架构规格):
//    [magic 4B] [version=2 4B] [model_type 4B] [规格数据...] [矩阵数据...]
//
//  规格编码 (所有多字节整数均为小端序 uint64_t):
//    MLP:         [num_dims 4B] [dim_0 8B] ... [dim_N 8B]
//    Transformer: [d_model 8B] [num_heads 8B] [d_ff 8B] [num_layers 8B] [patch_size 8B]
//    GPT:         [vocab_size 8B] [d_model 8B] [seq_len 8B] [num_heads 8B] [d_ff 8B] [num_layers 8B]
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr uint32_t MODEL_MAGIC   = 0x4E4E4E4E;  // "NNNN"
inline constexpr uint32_t MODEL_VERSION = 2;

// ═══════════════════════════════════════════════════════════════════════════
//  detail — 内部读写工具
//
//  所有函数使用固定宽度整数 (uint64_t) 替代 std::size_t，保证跨平台兼容。
//  所有函数返回 Result<T>，失败时返回 Error 错误信息。
// ═══════════════════════════════════════════════════════════════════════════

namespace detail
{

// ── 基础类型读写 ──────────────────────────────────────────────────────

inline Result<void> write_u32(std::ofstream &ofs, uint32_t v)
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

inline Result<void> write_u64(std::ofstream &ofs, uint64_t v)
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

inline Result<void> write_matrix(std::ofstream &ofs, const Matrix &m)
{
    if (auto r = write_u64(ofs, static_cast<uint64_t>(m.rows())); !r)
        return std::unexpected(r.error());
    if (auto r = write_u64(ofs, static_cast<uint64_t>(m.cols())); !r)
        return std::unexpected(r.error());
    const auto &data = m.data();
    ofs.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(double)));
    if (!ofs)
        return std::unexpected(Error{"Write error while writing matrix data"});
    return {};
}

inline Result<void> read_matrix(std::ifstream &ifs, Matrix &m)
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

    auto &data = m.data();
    ifs.read(reinterpret_cast<char *>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(double)));
    if (!ifs)
        return std::unexpected(Error{"Unexpected end of file while reading matrix data"});
    return {};
}

// ── ModelSpec 序列化 ──────────────────────────────────────────────────

inline Result<void> write_spec(std::ofstream &ofs, const ModelSpec &spec)
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

inline Result<void> write_header(std::ofstream &ofs)
{
    if (auto r = write_u32(ofs, MODEL_MAGIC); !r)
        return std::unexpected(r.error());
    if (auto r = write_u32(ofs, MODEL_VERSION); !r)
        return std::unexpected(r.error());
    return {};
}

// 返回读到的 version (1 或 2)，同时校验 magic number
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

    return *version_r;
}

} // namespace detail

// ═══════════════════════════════════════════════════════════════════════════
//  公开 API
//
//  save_model      — 保存 Model + ModelSpec 为 V2 格式二进制文件
//  load_model      — 从文件加载参数到已有 Model（兼容 V1/V2）
//  peek_model_spec — 只读文件头，返回 ModelSpec（不读参数，V1 返回 Unknown）
//
//  所有函数在失败时返回 Result 错误，无 std::cout 副作用。
//  调用方如需日志，可检查返回值并打印。
// ═══════════════════════════════════════════════════════════════════════════

// ── 保存模型 + 规格 ──────────────────────────────────────────────────────
inline Result<void> save_model(const std::string &filename,
                       Model &model, const ModelSpec &spec)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        return std::unexpected(Error{"Cannot open file for writing: " + filename});

    if (auto r = detail::write_header(ofs); !r)
        return std::unexpected(r.error());
    if (auto r = detail::write_spec(ofs, spec); !r)
        return std::unexpected(r.error());

    auto params = model.parameters();
    for (auto &p_ref : params)
    {
        if (auto r = detail::write_matrix(ofs, p_ref.get()); !r)
            return std::unexpected(r.error());
    }

    if (!ofs)
        return std::unexpected(Error{"Write error while saving model to: " + filename});
    return {};
}

// ── 读取模型规格（只读头部，不读参数） ──────────────────────────────────
//    V1 文件返回 ModelType::Unknown
[[nodiscard]] inline Result<ModelSpec> peek_model_spec(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());

    if (*version_r == 2)
        return detail::read_spec(ifs);

    // V1 文件没有规格信息
    return ModelSpec{ModelType::Unknown, {}, 0, 0, 0, 0, 0, 0, 0};
}

// ── 加载参数到已有模型 ──────────────────────────────────────────────────
//    兼容 V1 和 V2 文件（V2 会跳过规格头，只读参数）
inline Result<void> load_model(const std::string &filename, Model &model)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());

    // V2 有规格头，跳过（规格已隐含在构建好的 model 中）
    if (*version_r == 2)
    {
        auto spec_r = detail::read_spec(ifs);
        if (!spec_r) return std::unexpected(spec_r.error());
    }

    auto params = model.parameters();
    for (auto &p_ref : params)
    {
        if (auto r = detail::read_matrix(ifs, p_ref.get()); !r)
            return std::unexpected(r.error());
    }

    if (!ifs)
        return std::unexpected(Error{"Read error while loading model from: " + filename});
    return {};
}

} // namespace nn

#endif // MODEL_IO_HPP
