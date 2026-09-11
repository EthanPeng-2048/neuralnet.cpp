#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  model_io.hpp — 模型二进制序列化
//
//  职责：将 Model + ModelSpec 保存为二进制文件，或从文件加载参数。
//  设计：
//    - 使用固定宽度整数 (uint64_t) 保证跨平台/跨位宽兼容性
//    - 纯库函数，无 I/O 副作用（调用方自行决定是否打印日志）
//    - 规格头为自描述 KeyValueRecord（长度前缀，无偏移量假设）
//    - 使用 Result<T> 返回错误，不抛异常
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "core_config.hpp"
#include "core_file.hpp"    // 二进制 POD 读写（cast 边界收敛点，docs/17 §2.1）
#include "model_spec.hpp"
#include "model_keyvalue_record.hpp"
#include "compute_layer.hpp"
#include "model_container.hpp"

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  二进制文件格式（v4，自描述规格头）
//
//    [magic 4B]                 "NNNN" —— 标识「nn.cpp 的模型文件」
//    [version 4B]               模型格式版本号（v[x]；>=4 为自描述格式）
//    [precision 1B]             f32/f64 精度标记
//    [spec_len 8B]              规格头长度前缀（解决内容大小不确定问题）
//    [spec: KeyValueRecord]     字段记录模型信息（自描述，无偏移量假设）
//    [param matrices...]        模型权重
//    [extra state matrices...]  非可学习状态（如 BatchNorm running 统计）
//    [tokenizer_len 8B]         0 = 未嵌入
//    [tokenizer data...]        分词表数据
//
//  v1/v2/v3 为旧的偏移量定长格式，已移除支持（无有意义的旧模型）。
//
//  precision 字节: 0 = f32, 1 = f64（用于校验保存时与加载时的 Scalar 类型一致）
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr uint32_t MODEL_MAGIC    = 0x4E4E4E4E;  // "NNNN"
inline constexpr uint32_t MODEL_VERSION  = 5;            // v5: per-tensor precision tags

// ── 序列化待办（1.1，代码审查项 S3 / M1 / M2）────────────────────────────
// S3: read_spec_header / read_tokenizer 直接用文件里的 uint64 长度预分配
//     std::string(len,'\0')，无上限校验；配合 -fno-exceptions，恶意/损坏文件
//     可触发 bad_alloc → 进程崩溃。恢复前需加长度上限与读取完整性校验。
//     （1.0 决定暂不处理：仅影响本地加载自己/下载的模型，用户自行把关。）
// M1: .bin 全文件无校验和（.nnpkg 有 sha256）。内容损坏会被静默载入错误权重
//     且无感知。建议 MODEL_VERSION 5 增加整文件校验和与尾部完整性标记。
// M2: extra_state 的注释称"旧文件读到 EOF 保持默认（running_mean=0 等）"，
//     但实现是直接返回错误，注释与实现不符。需统一为按版本回退默认值。
// ═══════════════════════════════════════════════════════════════════════

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

// ── 安全二进制 I/O 辅助（转发 L0 收敛点 core_file.hpp::write_pod/read_pod）
// 字节级 reinterpret_cast 只出现在 core_file.hpp（docs/17 §2.1）。

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] inline Result<void> write_bytes(std::ofstream &ofs, const T &v)
{
    if (!nn::write_pod(ofs, v))
        return std::unexpected(Error{"Write error"});
    return {};
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] inline Result<T> read_bytes(std::ifstream &ifs)
{
    T v{};
    if (!nn::read_pod(ifs, v))
        return std::unexpected(Error{"Unexpected end of file"});
    return v;
}

// ── 基础类型读写 ──────────────────────────────────────────────────────
// 直接使用 write_bytes<T> / read_bytes<T>，不再提供 write_u32/u64 等冗余包装。

// variadic 字段批量写入：依次写入每个字段，遇错即停
template <typename... Ts>
[[nodiscard]] inline Result<void> write_fields(std::ofstream &ofs, const Ts &...fields)
{
    Result<void> result = {};
    bool stop = false;
    auto write_one = [&](const auto &field) {
        if (stop) return;
        result = write_bytes(ofs, field);
        if (!result) stop = true;
    };
    (write_one(fields), ...);
    return result;
}

// variadic 字段批量读取：依次读出每个字段，遇错即停
// 注意：实现使用递归模板逐个读取，避免 std::apply 在 MSVC 上的推导问题。
namespace detail_read
{
// 递归终止：所有字段已读取
template <typename Tuple>
[[nodiscard]] inline Result<void> read_fields_rec(
    std::ifstream & /*ifs*/, Tuple & /*values*/, std::index_sequence<>)
{
    return {};
}

// 递归读取第 I 个字段，然后读取剩余字段
template <typename Tuple, std::size_t I, std::size_t... Rest>
[[nodiscard]] inline Result<void> read_fields_rec(
    std::ifstream &ifs, Tuple &values, std::index_sequence<I, Rest...>)
{
    using T = std::tuple_element_t<I, Tuple>;
    auto r = read_bytes<T>(ifs);
    if (!r) return std::unexpected(r.error());
    std::get<I>(values) = *r;
    return read_fields_rec(ifs, values, std::index_sequence<Rest...>{});
}
}  // namespace detail_read

template <typename... Ts>
[[nodiscard]] inline Result<std::tuple<Ts...>> read_fields(std::ifstream &ifs)
{
    std::tuple<Ts...> values{};
    auto result = detail_read::read_fields_rec<std::tuple<Ts...>>(
        ifs, values, std::index_sequence_for<Ts...>{});
    if (!result) return std::unexpected(result.error());
    return values;
}

// ── 矩阵读写（v5：per-tensor precision tag）──────────────────────────────

// v5：写入矩阵时附带 precision tag（1B），tag 值见 precision_tag()
[[nodiscard]] inline Result<void> write_matrix_v5(std::ofstream &ofs, const Matrix &m, Precision p)
{
    // 写入 precision tag（§11.3）
    if (auto r = write_bytes<uint8_t>(ofs, precision_tag(p)); !r)
        return std::unexpected(r.error());
    // 写入形状
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())); !r)
        return std::unexpected(r.error());
    // 写入数据
    if (!nn::write_pod_span(ofs, m.span()))
        return std::unexpected(Error{"Write error while writing matrix data"});
    return {};
}

// v5：写入 f16 矩阵
[[nodiscard]] inline Result<void> write_matrix_fp16(std::ofstream &ofs, const MatrixT<Precision::F16> &m)
{
    if (auto r = write_bytes<uint8_t>(ofs, precision_tag(Precision::F16)); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())); !r)
        return std::unexpected(r.error());
    if (!nn::write_pod_span(ofs, m.span()))
        return std::unexpected(Error{"Write error while writing f16 matrix data"});
    return {};
}

// v5：读取矩阵（自动识别 precision tag）
[[nodiscard]] inline Result<std::pair<Precision, Matrix>> read_matrix_v5(std::ifstream &ifs)
{
    // 读取 precision tag
    auto tag_r = read_bytes<uint8_t>(ifs);
    if (!tag_r) return std::unexpected(tag_r.error());
    Precision p = precision_from_tag(*tag_r);
    if (p != Precision::F32 && p != Precision::F16)
        return std::unexpected(Error{"Unsupported precision tag in v5: " + std::to_string(*tag_r)});

    // 读取形状
    auto rows_r = read_bytes<uint64_t>(ifs);
    if (!rows_r) return std::unexpected(rows_r.error());
    auto cols_r = read_bytes<uint64_t>(ifs);
    if (!cols_r) return std::unexpected(cols_r.error());
    const auto rows = static_cast<std::size_t>(*rows_r);
    const auto cols = static_cast<std::size_t>(*cols_r);

    if (p == Precision::F16)
    {
        // 读取 f16 数据到临时 buffer，然后转为 f32 Matrix
        std::vector<f16> buf(rows * cols);
        if (!nn::read_pod_span(ifs, std::span<f16>(buf.data(), buf.size())))
            return std::unexpected(Error{"Unexpected end of file while reading f16 matrix data"});
        // 转为 f32（升 cast，精确无损）
        Matrix m(rows, cols);
        auto span = m.span();
        for (std::size_t i = 0; i < buf.size(); ++i)
            span[i] = static_cast<float>(buf[i]);
        return std::make_pair(Precision::F16, std::move(m));
    }

    // F32 路径
    Matrix m(rows, cols);
    if (!nn::read_pod_span(ifs, m.span()))
        return std::unexpected(Error{"Unexpected end of file while reading matrix data"});
    return std::make_pair(Precision::F32, std::move(m));
}

// ── v4 兼容：无 precision tag 的矩阵读写 ─────────────────────────────────
[[nodiscard]] inline Result<void> write_matrix(std::ofstream &ofs, const Matrix &m)
{
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())); !r)
        return std::unexpected(r.error());
    if (!nn::write_pod_span(ofs, m.span()))
        return std::unexpected(Error{"Write error while writing matrix data"});
    return {};
}

[[nodiscard]] inline Result<void> read_matrix(std::ifstream &ifs, Matrix &m)
{
    auto rows_r = read_bytes<uint64_t>(ifs);
    if (!rows_r) return std::unexpected(rows_r.error());
    auto cols_r = read_bytes<uint64_t>(ifs);
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

    if (!nn::read_pod_span(ifs, m.span()))
        return std::unexpected(Error{"Unexpected end of file while reading matrix data"});
    return {};
}

// ── ModelSpec ↔ KeyValueRecord（自描述规格头） ─────────────────────────

[[nodiscard]] inline KeyValueRecord spec_to_kv(const ModelSpec &spec)
{
    KeyValueRecord kv;
    kv.set("type", static_cast<uint64_t>(spec.type));

    std::vector<uint64_t> dims;
    dims.reserve(spec.layer_dims.size());
    for (auto d : spec.layer_dims)
        dims.push_back(static_cast<uint64_t>(d));
    kv.set("layer_dims", dims);

    kv.set("d_model",      static_cast<uint64_t>(spec.d_model));
    kv.set("num_heads",    static_cast<uint64_t>(spec.num_heads));
    kv.set("d_ff",         static_cast<uint64_t>(spec.d_ff));
    kv.set("num_layers",   static_cast<uint64_t>(spec.num_layers));
    kv.set("patch_size",   static_cast<uint64_t>(spec.patch_size));
    kv.set("vocab_size",   static_cast<uint64_t>(spec.vocab_size));
    kv.set("seq_len",      static_cast<uint64_t>(spec.seq_len));
    kv.set("pos_encoding", static_cast<uint64_t>(spec.pos_encoding));
    kv.set("activation",   static_cast<uint64_t>(spec.activation));
    kv.set("norm_type",    static_cast<uint64_t>(spec.norm_type));
    // ZiPT 记忆 token 数（可选字段：GPT/旧文件缺失时回落 0，不影响加载）
    kv.set("memory_tokens", static_cast<uint64_t>(spec.memory_tokens));
    // ZiPT 局部窗口 W（可选字段：缺失时回落 0 = 旧行为 W=L，即无压缩）
    kv.set("window", static_cast<uint64_t>(spec.window));

    // ── CNN ──
    kv.set("cnn_in_channels", static_cast<uint64_t>(spec.cnn_in_channels));
    kv.set("cnn_in_size",     static_cast<uint64_t>(spec.cnn_in_size));
    kv.set("cnn_pool",        static_cast<uint64_t>(spec.cnn_pool));

    auto to_u64_vec = [](const std::vector<std::size_t>& src) {
        std::vector<uint64_t> out;
        out.reserve(src.size());
        for (auto v : src) out.push_back(static_cast<uint64_t>(v));
        return out;
    };
    kv.set("cnn_channels",  to_u64_vec(spec.cnn_channels));
    kv.set("cnn_kernels",   to_u64_vec(spec.cnn_kernels));
    kv.set("cnn_strides",   to_u64_vec(spec.cnn_strides));
    kv.set("cnn_paddings",  to_u64_vec(spec.cnn_paddings));
    return kv;
}

// ══════════════════════════════════════════════════════════════════════
// 版本默认值表 —— 新增字段的唯一维护点
//
// 约定：文件版本号更新（v[x]）意味着规格字段可能新增/缺失。读取时，
// 早于「字段引入版本」的文件缺失该字段，在此按默认值回落，保证永久兼容。
// 未来新增字段只需：
//   1. spec_to_kv 中写入；
//   2. 此处追加一行（引入版本 + 缺失默认值）。
// ══════════════════════════════════════════════════════════════════════
inline void apply_spec_version_defaults(KeyValueRecord &kv, uint32_t version)
{
    // norm_type 于 v4 引入；更早版本缺失时默认 LayerNorm。
    // 注：v1/v2/v3 旧偏移量格式已在 read_and_validate_header 拒绝，此处为语义兜底。
    if (version < 4 && !kv.has("norm_type"))
        kv.set("norm_type", static_cast<uint64_t>(NormType::LayerNorm));
    // ── 未来字段在此追加，例如：
    // if (version < 5 && !kv.has("max_norm")) kv.set("max_norm", 0);
}

[[nodiscard]] inline Result<ModelSpec> spec_from_kv(const KeyValueRecord &kv)
{
    ModelSpec spec;  // 缺失字段保持 ModelSpec 默认值（即版本默认值）
    uint64_t v = 0;
    std::vector<uint64_t> dims;

    if (kv.get("type", v))        spec.type         = static_cast<ModelType>(v);
    if (kv.get("layer_dims", dims))
    {
        spec.layer_dims.clear();
        spec.layer_dims.reserve(dims.size());
        for (auto d : dims)
            spec.layer_dims.push_back(static_cast<std::size_t>(d));
    }
    if (kv.get("d_model", v))     spec.d_model      = static_cast<std::size_t>(v);
    if (kv.get("num_heads", v))   spec.num_heads    = static_cast<std::size_t>(v);
    if (kv.get("d_ff", v))        spec.d_ff         = static_cast<std::size_t>(v);
    if (kv.get("num_layers", v))  spec.num_layers   = static_cast<std::size_t>(v);
    if (kv.get("patch_size", v))  spec.patch_size   = static_cast<std::size_t>(v);
    if (kv.get("vocab_size", v))  spec.vocab_size   = static_cast<std::size_t>(v);
    if (kv.get("seq_len", v))     spec.seq_len      = static_cast<std::size_t>(v);
    if (kv.get("pos_encoding", v)) spec.pos_encoding = static_cast<PosEncodingType>(v);
    if (kv.get("activation", v))  spec.activation   = static_cast<ActivationType>(v);
    if (kv.get("norm_type", v))   spec.norm_type    = static_cast<NormType>(v);
    if (kv.get("memory_tokens", v)) spec.memory_tokens = static_cast<std::size_t>(v);
    if (kv.get("window", v))       spec.window       = static_cast<std::size_t>(v);

    // ── CNN ──
    if (kv.get("cnn_in_channels", v)) spec.cnn_in_channels = static_cast<std::size_t>(v);
    if (kv.get("cnn_in_size", v))     spec.cnn_in_size     = static_cast<std::size_t>(v);
    if (kv.get("cnn_pool", v))        spec.cnn_pool        = static_cast<std::size_t>(v);

    auto from_u64_vec = [](const std::vector<uint64_t>& src) {
        std::vector<std::size_t> out;
        out.reserve(src.size());
        for (auto v : src) out.push_back(static_cast<std::size_t>(v));
        return out;
    };
    if (kv.get("cnn_channels", dims)) spec.cnn_channels  = from_u64_vec(dims);
    if (kv.get("cnn_kernels", dims))  spec.cnn_kernels   = from_u64_vec(dims);
    if (kv.get("cnn_strides", dims))  spec.cnn_strides   = from_u64_vec(dims);
    if (kv.get("cnn_paddings", dims)) spec.cnn_paddings  = from_u64_vec(dims);

    if (spec.type == ModelType::Unknown)
        return std::unexpected(Error{"模型文件规格缺少有效的 type 字段"});
    return spec;
}

// ── 规格头读写：长度前缀 + KeyValueRecord（无偏移量假设） ────────────────

[[nodiscard]] inline Result<void> write_spec_header(std::ofstream &ofs, const ModelSpec &spec)
{
    auto bytes = spec_to_kv(spec).serialize();
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(bytes.size())); !r)
        return std::unexpected(r.error());
    if (!bytes.empty())
    {
        ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!ofs)
            return std::unexpected(Error{"Write error while writing spec header"});
    }
    return {};
}

[[nodiscard]] inline Result<ModelSpec> read_spec_header(std::ifstream &ifs, uint32_t version)
{
    auto len_r = read_bytes<uint64_t>(ifs);
    if (!len_r) return std::unexpected(len_r.error());
    const auto len = static_cast<std::size_t>(*len_r);
    std::string bytes(len, '\0');
    if (len > 0)
    {
        ifs.read(bytes.data(), static_cast<std::streamsize>(len));
        if (!ifs)
            return std::unexpected(Error{"Unexpected end while reading spec header"});
    }
    auto kv_r = KeyValueRecord::parse(bytes);
    if (!kv_r) return std::unexpected(kv_r.error());
    apply_spec_version_defaults(*kv_r, version);
    return spec_from_kv(*kv_r);
}

// ── 文件头读写 ──────────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_header(std::ofstream &ofs)
{
    if (auto r = write_bytes<uint32_t>(ofs, MODEL_MAGIC); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint32_t>(ofs, MODEL_VERSION); !r)
        return std::unexpected(r.error());
    if (auto r = write_bytes<uint8_t>(ofs, PRECISION_TAG); !r)
        return std::unexpected(r.error());
    return {};
}

// 返回读到的 version，同时校验 magic number、格式版本和精度
// v1/v2/v3 为旧的偏移量定长格式（已移除支持），仅接受自描述格式 v4+。
[[nodiscard]] inline Result<uint32_t> read_and_validate_header(std::ifstream &ifs)
{
    auto magic_r = read_bytes<uint32_t>(ifs);
    if (!magic_r) return std::unexpected(magic_r.error());
    if (*magic_r != MODEL_MAGIC)
        return std::unexpected(Error{"Invalid model file: bad magic number (0x"
                           + std::to_string(*magic_r) + ")"});

    auto version_r = read_bytes<uint32_t>(ifs);
    if (!version_r) return std::unexpected(version_r.error());
    const auto version = *version_r;
    if (version < 4)
        return std::unexpected(Error{"模型文件为旧格式 (v" + std::to_string(version)
                           + ")，已不再支持；请用当前版本重新训练/保存。"});
    if (version > MODEL_VERSION)
        return std::unexpected(Error{"模型文件版本过新 (v" + std::to_string(version)
                           + " > v" + std::to_string(MODEL_VERSION) + ")，请升级程序。"});

    // 读取并校验精度标记（v4/v5 通用）
    auto pt_result = read_bytes<uint8_t>(ifs);
    if (!pt_result)
        return std::unexpected(Error{"Unexpected end: missing precision tag"});
    uint8_t precision_tag = *pt_result;
    if (precision_tag != PRECISION_TAG)
        return std::unexpected(Error{
            "Precision mismatch: file uses " + std::string(precision_tag == 0 ? "f32" : "f64")
            + ", but build uses " + std::string(PRECISION_TAG == 0 ? "f32" : "f64")});

    return version;
}

// ── Tokenizer JSON 读写（V3 新增） ──────────────────────────────────

[[nodiscard]] inline Result<void> write_tokenizer(std::ofstream &ofs, const std::string &json)
{
    if (auto r = write_bytes<uint64_t>(ofs, static_cast<uint64_t>(json.size())); !r)
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
    auto len_r = read_bytes<uint64_t>(ifs);
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
//  save_model      — 保存 Model + ModelSpec（可选嵌入 tokenizer 数据）
//  load_model      — 从文件加载参数，返回嵌入的 tokenizer 数据（如有）
//  peek_model_spec — 只读文件头，返回 ModelSpec（不读参数）
// ═══════════════════════════════════════════════════════════════════════════

// ── 保存模型（v5：per-tensor precision tag）──────────────────────────────
// 参数为 Tensor*，通过 engine.to_matrix 下载到 CPU 后写入。
// v5 格式：每个矩阵前加 1B precision tag（0=f32, 2=f16）。
[[nodiscard]] inline Result<void> save_model(const std::string &filename,
    Model &model, const ModelSpec &spec, const std::string &tokenizer_json = {})
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        return std::unexpected(Error{"Cannot open file for writing: " + filename});

    if (auto r = detail::write_header(ofs); !r)
        return std::unexpected(r.error());
    if (auto r = detail::write_spec_header(ofs, spec); !r)
        return std::unexpected(r.error());

    auto& engine = model.engine();
    auto params = model.parameters();
    for (auto& p_tensor : params)
    {
        Precision p = p_tensor.get().precision();
        if (p == Precision::F16)
        {
            // f16 参数：写入带 f16 tag
            // 对于 CPU f16 tensor，直接访问底层 f16 数据
            if (p_tensor.get().is_cpu())
            {
                const auto& m16 = p_tensor.get().cpu_matrix<Precision::F16>();
                if (auto r = detail::write_matrix_fp16(ofs, m16); !r)
                    return std::unexpected(r.error());
            }
            else
            {
                // GPU f16：download 到 CPU（f32 升 cast），再写 f16
                auto m32_r = engine.to_matrix(p_tensor, Precision::F32);
                if (!m32_r) return std::unexpected(m32_r.error());
                MatrixT<Precision::F16> m16(m32_r->rows(), m32_r->cols());
                const auto src = m32_r->span();
                auto dst = m16.span();
                for (std::size_t i = 0; i < src.size(); ++i)
                    dst[i] = src[i];
                if (auto r = detail::write_matrix_fp16(ofs, m16); !r)
                    return std::unexpected(r.error());
            }
        }
        else
        {
            // f32 参数（默认路径）
            auto m = engine.to_matrix(p_tensor, Precision::F32);
            if (!m) return std::unexpected(m.error());
            if (auto r = detail::write_matrix_v5(ofs, *m, p); !r)
                return std::unexpected(r.error());
        }
    }

    // 非可学习状态（如 BatchNorm 的 running_mean/running_var）
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        auto m = engine.to_matrix(e_tensor);
        if (!m) return std::unexpected(m.error());
        if (auto r = detail::write_matrix_v5(ofs, *m, e_tensor.get().precision()); !r)
            return std::unexpected(r.error());
    }

    // tokenizer JSON（长度前缀，0 表示无）
    if (auto r = detail::write_tokenizer(ofs, tokenizer_json); !r)
        return std::unexpected(r.error());

    if (!ofs)
        return std::unexpected(Error{"Write error while saving model to: " + filename});
    return {};
}

// ── 读取模型规格（只读头部，不读参数/tokenizer） ──────────────────────────
[[nodiscard]] inline Result<ModelSpec> peek_model_spec(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());

    return detail::read_spec_header(ifs, *version_r);
}

// ── 加载参数 + tokenizer ───────────────────────────────────────────────
//    返回嵌入的 tokenizer 数据字符串（空串 = 未嵌入）
// v5：per-tensor precision tag（读取时自动识别 f32/f16）
// v4：无 precision tag（所有矩阵为 f32）
[[nodiscard]] inline Result<std::string> load_model(const std::string &filename, Model &model)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return std::unexpected(Error{"Cannot open file for reading: " + filename});

    auto version_r = detail::read_and_validate_header(ifs);
    if (!version_r) return std::unexpected(version_r.error());
    const uint32_t version = *version_r;

    // 读取并校验规格头
    auto spec_r = detail::read_spec_header(ifs, version);
    if (!spec_r) return std::unexpected(spec_r.error());

    if (auto stored = model.spec(); stored)
    {
        if (!spec_matches(*stored, *spec_r))
        {
            return std::unexpected(Error{
                "Model architecture mismatch while loading '" + filename + "': "
                "model expects " + spec_summary(*stored) +
                " but file contains " + spec_summary(*spec_r)});
        }
    }

    auto& engine = model.engine();
    auto params = model.parameters();

    if (version >= 5)
    {
        // ── v5：per-tensor precision tag ─────────────────────────────
        for (auto& p_tensor : params)
        {
            auto result_r = detail::read_matrix_v5(ifs);
            if (!result_r) return std::unexpected(result_r.error());
            auto& [file_prec, file_matrix] = *result_r;

            // 统一接口：from_matrix(m, P) 上传到目标精度
            Precision target_p = p_tensor.get().precision();
            if (file_prec == target_p)
            {
                // 同精度：直接上传
                auto uploaded = engine.from_matrix(file_matrix, target_p);
                if (!uploaded) return std::unexpected(uploaded.error());
                // copy_from 从上传结果写入目标 tensor
                if (auto r = engine.copy_from(p_tensor, file_matrix); !r)
                    return std::unexpected(r.error());
            }
            else
            {
                // 跨精度：cast 后上传
                auto uploaded = engine.from_matrix(file_matrix, file_prec);
                if (!uploaded) return std::unexpected(uploaded.error());
                auto casted = engine.cast(*uploaded, target_p);
                if (!casted) return std::unexpected(casted.error());
                // 从 cast 结果下载为 f32，再 copy_from
                auto m32 = engine.to_matrix(*casted, Precision::F32);
                if (!m32) return std::unexpected(m32.error());
                if (auto r = engine.copy_from(p_tensor, *m32); !r)
                    return std::unexpected(r.error());
            }
        }
    }
    else
    {
        // ── v4：无 precision tag（所有矩阵为 f32）───────────────────
        for (auto& p_tensor : params)
        {
            Matrix tmp(p_tensor.get().rows(), p_tensor.get().cols());
            if (auto r = detail::read_matrix(ifs, tmp); !r)
                return std::unexpected(r.error());
            if (auto r = engine.copy_from(p_tensor, tmp); !r)
                return std::unexpected(r.error());
        }
    }

    // 非可学习状态
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        if (ifs.peek() == EOF) break;

        if (version >= 5)
        {
            auto result_r = detail::read_matrix_v5(ifs);
            if (!result_r) return std::unexpected(result_r.error());
            auto& [file_prec, file_matrix] = *result_r;
            if (auto r = engine.copy_from(e_tensor, file_matrix); !r)
                return std::unexpected(r.error());
        }
        else
        {
            Matrix tmp(e_tensor.get().rows(), e_tensor.get().cols());
            auto mr = detail::read_matrix(ifs, tmp);
            if (!mr) return std::unexpected(mr.error());
            if (auto r = engine.copy_from(e_tensor, tmp); !r)
                return std::unexpected(r.error());
        }
    }

    // 读取 tokenizer
    std::string tokenizer_json;
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

