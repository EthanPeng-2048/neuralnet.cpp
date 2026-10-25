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
#include <streambuf>
#include <string>
#include <tuple>
#include <type_traits>
#include <array>
#include <vector>

#include <neuralnet.cpp/core_config.hpp>
#include <neuralnet.cpp/core_file.hpp>    // 二进制 POD 读写（cast 边界收敛点）
#include <neuralnet.cpp/core_sha256.hpp>  // 整文件校验和（P2-1）
#include "model_spec.hpp"
#include <neuralnet.cpp/model_keyvalue_record.hpp>
#include "train_state.hpp"  // 训练状态段（P1-2）
#include "compute_layer.hpp"
#include "model_container.hpp"

namespace nn
{

// ═══════════════════════════════════════════════════════════════════════════
//  二进制文件格式（v8，自描述规格头）
//
//    [magic 4B]                 "NNNN" —— 标识「nn.cpp 的模型文件」
//    [version 4B]               模型格式版本号（v[x]；>=4 为自描述格式）
//    [precision 1B]             f32/f64 精度标记
//    [role 1B]                  文件角色（v7 起；Model=推理 / Checkpoint=训练）
//    [spec_len 8B]              规格头长度前缀（解决内容大小不确定问题）
//    [spec: KeyValueRecord]     字段记录模型信息（自描述，无偏移量假设）
//    [state_len 8B]             训练状态段长度（**v8 起**；0 = 无该段）
//    [state: kvrec]             训练状态（TrainState：计数器 / 配方 / 数据集身份）
//    [optimizer blobs...]       优化器状态张量（参数条目形态；数量见 kvrec 的
//                               optimizer_blob_count；仅 Checkpoint 角色非零）
//    [param entries...]         模型权重（v7 起为参数级 kind 条目）
//    [extra state entries...]   非可学习状态（如 BatchNorm running 统计）
//    [tokenizer_len 8B]         0 = 未嵌入
//    [tokenizer data...]        词表数据（**按模型版本分派**：
//                               v5 及以下 = 内嵌 JSON 字符串（模型格式的一部分，
//                               照常读取、零破坏）；v6 起 = .nnvocab 文件镜像
//                               字节（[label "NNVC"][词表 kvrec v2]））
//    [sha256 32B]               整文件校验和（**v8 起**；覆盖本字段之前的全部字节）
//    ["NNEND" 5B]               尾部完整性标记（**v8 起**）
//
//  文件格式 v4+ 自描述（长度前缀 + KeyValueRecord）；v1/v2/v3 偏移量定长
//  格式在 header 校验阶段（read_and_validate_header）即拒绝。
//
//  precision 字节: 0 = f32, 1 = f64（用于校验保存时与加载时的 Scalar 类型一致）
//
//  角色语义（v8 起冻结，docs/todo/23 §2.1.1）：checkpoint = model 的**超集** ——
//  同一容器 + role 字节，训练状态段 role-gated（Model 写 0 长度）。`.nnckpt`
//  仍需能被推理入口直接读（裁决第 3 条的"测试推理"用途），故不做独立格式。
// ═══════════════════════════════════════════════════════════════════════════

inline constexpr uint32_t MODEL_MAGIC    = 0x4E4E4E4E;  // "NNNN"
// v7（docs/todo/23）：**文件角色**（role 字节：Model=推理 / Checkpoint=训练）
//   + **参数级 kind**（[kind][precision][rows][cols][aux_len][payload_len][aux][payload]）。
//   · 角色决定"写哪些参数"：`.nnmodel` 写推理所需全部参数（三值层写打包形态）；
//     `.nnckpt` 写模型 + 训练参数（三值层写 f32 latent）。
//   · v5/v6 的扁平矩阵序列仍**可读**（role 缺失按 Model 处理）；**写一律 v7**。
// v6: 内嵌词表改为 .nnvocab kvrec（v5 及以下仍读其内嵌 JSON，见上）。
// v8（docs/todo/18 §4.2 P1-2 + §4.3 P2-1 合批）：
//   · **训练状态段**（role 门控；spec 之后）：`TrainState` 的 kvrec（计数器 /
//     训练配方含精度 profile 与 lr 曲线形状 / 数据集身份）+ 优化器状态张量条目。
//     Model 角色写 0 长度；老文件（v7 及以下）缺段 ⇒ 冷启动语义（零破坏）。
//   · **整文件校验和 + 尾部完整性标记**：`[sha256 32B]["NNEND" 5B]`（摘要覆盖
//     其前的全部字节，边写边算：见 `detail::HashingStreamBuf`）；截断/损坏的
//     文件在读入前即明确报错（`verify_model_integrity`），不再静默载入错误权重。
inline constexpr uint32_t MODEL_VERSION  = 8;

// ── 尾部完整性标记（v8；见上）────────────────────────────────────────────
inline constexpr std::size_t MODEL_DIGEST_BYTES = 32;          // sha256
inline constexpr std::string_view MODEL_TAIL_MAGIC = "NNEND";  // 5 字节
inline constexpr std::size_t MODEL_TAIL_BYTES = MODEL_DIGEST_BYTES + MODEL_TAIL_MAGIC.size();

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
// 字节级 reinterpret_cast 只出现在 core_file.hpp。

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] inline Result<void> write_bytes(std::ostream &ofs, const T &v)
{
    if (!nn::write_pod(ofs, v))
        NN_FAIL("Write error");
    return {};
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
[[nodiscard]] inline Result<T> read_bytes(std::ifstream &ifs)
{
    T v{};
    if (!nn::read_pod(ifs, v))
    {
        // 诊断增强：带上失败位置（跨版本格式排查时先看"读到哪断了"）
        const auto pos = ifs.tellg();
        NN_FAIL("Unexpected end of file（读 " + std::to_string(sizeof(T)) +
                " 字节失败，流位置 " +
                (pos < 0 ? std::string("<eof>")
                         : std::to_string(static_cast<long long>(pos))) + "）");
    }
    return v;
}

// ── 基础类型读写 ──────────────────────────────────────────────────────
// 统一使用 write_bytes<T> / read_bytes<T>，不另设 write_u32/u64 等冗余包装。

// variadic 字段批量写入：依次写入每个字段，遇错即停
template <typename... Ts>
[[nodiscard]] inline Result<void> write_fields(std::ostream &ofs, const Ts &...fields)
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
    NN_TRY_CHECK(r);
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
    NN_TRY_CHECK(result);
    return values;
}

// ── 边写边算 sha256 的流缓冲（v8 整文件校验和；P2-1）────────────────────
// 所有写路径都走 std::ostream 接口（write_pod / os.write），故把哈希塞进
// streambuf 就能零侵入地覆盖"校验和该覆盖的每一个字节"——不需要把整文件
// 在内存里拼一遍，也不需要写完再读回来算一遍。
// 用法：`HashingStreamBuf hb(ofs.rdbuf(), st); std::ostream os(&hb);` 之后
// 所有写入交给 `os`；摘要本体与尾部标记**不经** os（它们不算进摘要）。
class HashingStreamBuf : public std::streambuf
{
public:
    HashingStreamBuf(std::streambuf &sink, Sha256State &hasher) noexcept
        : sink_(sink), hasher_(hasher) {}

protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override
    {
        if (n > 0)
            sha256_update(hasher_, std::string_view(s, static_cast<std::size_t>(n)));
        return sink_.sputn(s, n);
    }

    int_type overflow(int_type c) override
    {
        if (traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
        const char ch = traits_type::to_char_type(c);
        if (xsputn(&ch, 1) != 1) return traits_type::eof();
        return static_cast<int_type>(static_cast<unsigned char>(ch));
    }

private:
    std::streambuf &sink_;
    Sha256State &hasher_;
};

// ── 矩阵读写（v5：per-tensor precision tag）──────────────────────────────

// v5：写入矩阵时附带 precision tag（1B），tag 值见 precision_tag()
[[nodiscard]] inline Result<void> write_matrix_v5(std::ostream &ofs, const Matrix &m, Precision p)
{
    // 写入 precision tag（§11.3）
    NN_TRY(r, write_bytes<uint8_t>(ofs, precision_tag(p)));
    // 写入形状
    NN_TRY(r2, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())));
    NN_TRY(r3, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())));
    // 写入数据
    if (!nn::write_pod_span(ofs, m.span()))
        NN_FAIL("Write error while writing matrix data");
    return {};
}

// v5：写入 f16 矩阵
[[nodiscard]] inline Result<void> write_matrix_fp16(std::ostream &ofs, const MatrixT<Precision::F16> &m)
{
    NN_TRY(r, write_bytes<uint8_t>(ofs, precision_tag(Precision::F16)));
    NN_TRY(r2, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())));
    NN_TRY(r3, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())));
    if (!nn::write_pod_span(ofs, m.span()))
        NN_FAIL("Write error while writing f16 matrix data");
    return {};
}

// v5：读取矩阵（自动识别 precision tag）
[[nodiscard]] inline Result<std::pair<Precision, Matrix>> read_matrix_v5(std::ifstream &ifs)
{
    // 读取 precision tag
    auto tag_r = read_bytes<uint8_t>(ifs);
    NN_TRY_CHECK(tag_r);
    Precision p = precision_from_tag(*tag_r);
    if (p != Precision::F32 && p != Precision::F16)
        NN_FAIL("Unsupported precision tag in v5: " + std::to_string(*tag_r));
    // 读取形状
    auto rows_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(rows_r);
    auto cols_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(cols_r);
    const auto rows = static_cast<std::size_t>(*rows_r);
    const auto cols = static_cast<std::size_t>(*cols_r);

    // ── 未信任形状防御（与 kMaxSerializedStringBytes 同一威胁模型）──────
    // rows/cols 直接来自文件，必须在任何分配前校验，否则两类事故：
    //   ① rows*cols 不溢出但巨量 → 先物化巨缓冲 → 内存耗尽，
    //      -fno-exceptions 下 bad_alloc 直接 terminate，连 Result 都回不去；
    //   ② rows*cols 溢出回绕成小值 → 小缓冲 + 巨大 rows_/cols_ 元数据，
    //      后续按行列索引（row*cols_+col）越界读写。
    // 判据：矩阵数据字节数不得超过文件剩余字节数（用除法比较，
    // 全程不产生 rows*cols 或 ×elem 的乘法溢出）。
    {
        const std::streampos pos = ifs.tellg();
        ifs.seekg(0, std::ios::end);
        const std::streampos end = ifs.tellg();
        if (pos < 0 || end < pos)
            NN_FAIL("Cannot determine file size while reading matrix");
        ifs.seekg(pos);
        if (!ifs)
            NN_FAIL("Stream seek failed while reading matrix");
        const auto remaining = static_cast<std::uint64_t>(end - pos);
        const std::uint64_t elem_bytes = (p == Precision::F16) ? 2u : 4u;
        const std::uint64_t max_elems = remaining / elem_bytes;
        if (rows != 0 && cols != 0 && static_cast<std::uint64_t>(rows) > max_elems / cols)
            NN_FAIL("Matrix shape " + std::to_string(rows) + "x" + std::to_string(cols)                 + " exceeds remaining file bytes; file is corrupt or malicious");
    }

    if (p == Precision::F16)
    {
        // 读取 f16 数据到临时 buffer，然后转为 f32 Matrix
        std::vector<f16> buf(rows * cols);
        if (!nn::read_pod_span(ifs, std::span<f16>(buf.data(), buf.size())))
            NN_FAIL("Unexpected end of file while reading f16 matrix data");
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
        NN_FAIL("Unexpected end of file while reading matrix data");
    return std::make_pair(Precision::F32, std::move(m));
}

// ── v4 兼容：无 precision tag 的矩阵读写 ─────────────────────────────────
[[nodiscard]] inline Result<void> write_matrix(std::ostream &ofs, const Matrix &m)
{
    NN_TRY(r, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.rows())));
    NN_TRY(r2, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(m.cols())));
    if (!nn::write_pod_span(ofs, m.span()))
        NN_FAIL("Write error while writing matrix data");
    return {};
}

[[nodiscard]] inline Result<void> read_matrix(std::ifstream &ifs, Matrix &m)
{
    auto rows_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(rows_r);
    auto cols_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(cols_r);
    const auto rows = static_cast<std::size_t>(*rows_r);
    const auto cols = static_cast<std::size_t>(*cols_r);

    if (rows != m.rows() || cols != m.cols())
    {
        NN_FAIL("Matrix shape mismatch: expected (" + std::to_string(m.rows())             + ", " + std::to_string(m.cols()) + "), got ("             + std::to_string(rows) + ", " + std::to_string(cols) + ")");
    }

    if (!nn::read_pod_span(ifs, m.span()))
        NN_FAIL("Unexpected end of file while reading matrix data");
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
    kv.set("norm_place",   static_cast<uint64_t>(spec.norm_place));
    // subln（BitNet 2B4T 的子层归一化，docs 22）：缺失 → false（旧布局逐位一致）。
    kv.set("subln",        static_cast<uint64_t>(spec.subln ? 1 : 0));
    // n_head_kv（GQA，docs 22）：缺失 → 0（= num_heads = MHA，旧布局逐位一致）。
    kv.set("n_head_kv",    static_cast<uint64_t>(spec.n_head_kv));
    // tie_embeddings（docs 22）：缺失 → false（独立 head，旧布局逐位一致）。
    kv.set("tie_embeddings", static_cast<uint64_t>(spec.tie_embeddings ? 1 : 0));
    // weight_quant（线性层是否 BitLinear/三值）于 P1 引入。旧文件**无此键** →
    // 读回 WeightQuant::None（= 普通 Linear，参数布局逐位一致）→ 无需升版本。
    kv.set("weight_quant", static_cast<uint64_t>(spec.weight_quant));
    // output_schema / output_packing（RL-0）：旧文件缺键 = 单输出模型。
    if (spec.output_schema)
        kv.set("output_schema", encode_tensor_schema(*spec.output_schema));
    if (!spec.output_packing.empty())
        kv.set("output_packing", encode_tensor_name_list(spec.output_packing));
    // ZiPT（AttnZip）已于 2026-10-01 整体移除：不再写 memory_tokens / window 字段。
    // 旧文件（含这两个键）由 spec_from_kv 的 type 分支给出明确错误。

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
    // 注：v1/v2/v3 偏移量格式在 read_and_validate_header 已拒绝，此处为语义兜底。
    if (version < 4 && !kv.has("norm_type"))
        kv.set("norm_type", static_cast<uint64_t>(NormType::LayerNorm));
    // norm_place（CNN/ViT 归一化挂载）于 v5 引入；缺失时保持 ModelSpec 默认
    // NormPlace::None = 不加额外归一化，与旧文件的参数布局逐位一致 → 无需回落动作。
    // if (version < 5 && !kv.has("norm_place")) kv.set("norm_place", 0);
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
    if (kv.get("norm_place", v))  spec.norm_place   = static_cast<NormPlace>(v);
    // subln（docs 22 新增）：缺键 → 保持 ModelSpec 默认 subln = false
    if (kv.get("subln", v))       spec.subln        = (v != 0);
    // n_head_kv（docs 22 新增）：缺键 → 保持默认 0（= MHA）
    if (kv.get("n_head_kv", v))   spec.n_head_kv    = static_cast<std::size_t>(v);
    // tie_embeddings（docs 22 新增）：缺键 → 保持默认 false
    if (kv.get("tie_embeddings", v)) spec.tie_embeddings = (v != 0);
    // weight_quant（P1 新增）：缺键 → 保持 ModelSpec 默认 WeightQuant::None
    if (kv.get("weight_quant", v)) spec.weight_quant = static_cast<WeightQuant>(v);
    // output_schema / output_packing（RL-0 新增）：缺键 → nullopt/空（旧行为）。
    std::string encoded;
    if (kv.get("output_schema", encoded))
    {
        NN_TRY(schema, decode_tensor_schema(encoded));
        spec.output_schema = std::move(*schema);
    }
    if (kv.get("output_packing", encoded))
    {
        NN_TRY(names, decode_tensor_name_list(encoded));
        spec.output_packing = std::move(*names);
    }
    if (!spec.output_packing.empty() && !spec.output_schema)
        NN_FAIL("模型文件含 output_packing 但缺少 output_schema");
    if (spec.output_schema && !spec.output_packing.empty())
    {
        if (spec.output_packing.size() != spec.output_schema->size())
            NN_FAIL("模型文件 output_packing 未完整覆盖 output_schema");
        std::vector<bool> seen(spec.output_schema->size(), false);
        for (const std::string& name : spec.output_packing)
        {
            NN_TRY(handle, spec.output_schema->field(name));
            if (seen[handle->index]) NN_FAIL("模型文件 output_packing 含重复端口 '" + name + "'");
            seen[handle->index] = true;
            const TensorFieldSpec& field = spec.output_schema->fields()[handle->index];
            if (!field.shape.rows)
                NN_FAIL("模型文件 output_packing 的端口 '" + name + "' 未声明固定 rows");
        }
    }
    // 旧文件的 memory_tokens / window（ZiPT 专用）不再读取，随 type=6 一并拒绝。

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
        NN_FAIL("模型文件规格缺少有效的 type 字段");
    if (spec.type == ModelType::Reserved_ZiPT)
        NN_FAIL("该模型文件为 AttnZip/ZiPT 架构，已整体移除"             "（代码保留在 legacy/zipt 分支；恢复前提见 docs/history/history.md）");
    return spec;
}

// ── 规格头读写：长度前缀 + KeyValueRecord（无偏移量假设） ────────────────

[[nodiscard]] inline Result<void> write_spec_header(std::ostream &ofs, const ModelSpec &spec)
{
    auto bytes = spec_to_kv(spec).serialize();
    NN_TRY(r, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(bytes.size())));
    if (!bytes.empty())
    {
        ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!ofs)
            NN_FAIL("Write error while writing spec header");
    }
    return {};
}

// 反序列化字符串段的硬上限：文件中的长度字段不可信（64 位，可能是损坏/恶意值），
// 直接 std::string(len,'\0') 会先做巨量分配 → bad_alloc/terminate（-fno-exceptions
// 下连 Result 都回不去）。模型 spec / tokenizer JSON 远小于此值。
inline constexpr std::size_t kMaxSerializedStringBytes = 64u * 1024u * 1024u;

[[nodiscard]] inline Result<ModelSpec> read_spec_header(std::ifstream &ifs, uint32_t version)
{
    auto len_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(len_r);
    if (*len_r > kMaxSerializedStringBytes)
        NN_FAIL("spec header 长度越界（超过 64 MiB），文件可能损坏或恶意");
    const auto len = static_cast<std::size_t>(*len_r);
    std::string bytes(len, '\0');
    if (len > 0)
    {
        ifs.read(bytes.data(), static_cast<std::streamsize>(len));
        if (!ifs)
            NN_FAIL("Unexpected end while reading spec header");
    }
    auto kv_r = KeyValueRecord::parse(bytes);
    NN_TRY_CHECK(kv_r);
    apply_spec_version_defaults(*kv_r, version);
    return spec_from_kv(*kv_r);
}

// ── 训练状态段读取（v8）：长度前缀 + KeyValueRecord → TrainState ─────────
// `state_len == 0` = 无该段（Model 角色，或断点由未接线路径写出）⇒ 默认值，
// 续训路径按"冷启动语义"处理。
[[nodiscard]] inline Result<TrainState> read_train_state_section(
    std::ifstream &ifs, std::uint64_t state_len)
{
    if (state_len == 0) return TrainState{};
    if (state_len > kMaxSerializedStringBytes)
        NN_FAIL("训练状态段长度越界（超过 64 MiB），文件可能损坏或恶意");
    std::string bytes(static_cast<std::size_t>(state_len), '\0');
    ifs.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!ifs)
        NN_FAIL("Unexpected end while reading train state section");
    NN_TRY(kv, KeyValueRecord::parse(bytes));
    return train_state_from_kv(*kv);
}

// ── 文件头读写 ──────────────────────────────────────────────────────

[[nodiscard]] inline Result<void> write_header(std::ostream &ofs, ModelRole role)
{
    NN_TRY(r, write_bytes<uint32_t>(ofs, MODEL_MAGIC));
    NN_TRY(r2, write_bytes<uint32_t>(ofs, MODEL_VERSION));
    NN_TRY(r3, write_bytes<uint8_t>(ofs, PRECISION_TAG));
    NN_TRY(r4, write_bytes<uint8_t>(ofs, static_cast<uint8_t>(role)));
    return {};
}

// 返回读到的 version，同时校验 magic number、格式版本和精度
// 仅接受自描述格式 v4+；v1/v2/v3 偏移量定长格式在此处拒绝。
[[nodiscard]] inline Result<uint32_t> read_and_validate_header(
    std::ifstream &ifs, ModelRole* role_out = nullptr)
{
    auto magic_r = read_bytes<uint32_t>(ifs);
    NN_TRY_CHECK(magic_r);
    if (*magic_r != MODEL_MAGIC)
        NN_FAIL("Invalid model file: bad magic number (0x"
                           + std::to_string(*magic_r) + ")");
    auto version_r = read_bytes<uint32_t>(ifs);
    NN_TRY_CHECK(version_r);
    const auto version = *version_r;
    if (version < 4)
        NN_FAIL("模型文件为旧格式 (v" + std::to_string(version)
                           + ")，已不再支持；请用当前版本重新训练/保存。");
    if (version > MODEL_VERSION)
        NN_FAIL("模型文件版本过新 (v" + std::to_string(version)
                           + " > v" + std::to_string(MODEL_VERSION) + ")，请升级程序。");
    auto pt_result = read_bytes<uint8_t>(ifs);
    if (!pt_result)
        NN_FAIL("Unexpected end: missing precision tag");
    uint8_t precision_tag = *pt_result;
    if (precision_tag != PRECISION_TAG)
        NN_FAIL("Precision mismatch: file uses " + std::string(precision_tag == 0 ? "f32" : "f64")             + ", but build uses " + std::string(PRECISION_TAG == 0 ? "f32" : "f64"));

    // ── v7：文件角色字节（v5/v6 无该字段 → 一律按 Model 处理，零破坏）──────
    ModelRole role = ModelRole::Model;
    if (version >= 7)
    {
        auto role_r = read_bytes<uint8_t>(ifs);
        if (!role_r)
            NN_FAIL("Unexpected end: missing file role byte (v7)");
        if (*role_r > static_cast<uint8_t>(ModelRole::Checkpoint))
            NN_FAIL("模型文件 role 字段非法（" + std::to_string(*role_r) + "）");
        role = static_cast<ModelRole>(*role_r);
    }
    if (role_out) *role_out = role;
    return version;
}

// ── Tokenizer 数据读写（长度前缀字节块；内容按模型版本分派）────────────
// v5 及以下 = JSON 字符串；v6 起 = .nnvocab 文件镜像字节。
// 本层只搬字节，解释权在调用方（load_tokenizer_from_model_blob）。

[[nodiscard]] inline Result<void> write_tokenizer(std::ostream &ofs, const std::string &data)
{
    NN_TRY(r, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(data.size())));
    if (!data.empty())
    {
        ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!ofs)
            NN_FAIL("Write error while writing tokenizer");
    }
    return {};
}

[[nodiscard]] inline Result<std::string> read_tokenizer(std::ifstream &ifs)
{
    auto len_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(len_r);
    if (*len_r > kMaxSerializedStringBytes)
        NN_FAIL("tokenizer 长度越界（超过 64 MiB），文件可能损坏或恶意");
    const auto len = static_cast<std::size_t>(*len_r);
    if (len == 0) return std::string{};  // 未嵌入

    std::string data(len, '\0');
    ifs.read(data.data(), static_cast<std::streamsize>(len));
    if (!ifs)
        NN_FAIL("Unexpected end while reading tokenizer data");
    return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// v7 参数条目（参数级 kind）
//   [kind 1B][precision 1B][rows 8B][cols 8B][aux_len 8B][payload_len 8B][aux][payload]
// 载荷一律按**原始字节**落盘：
//   · Dense / TernaryLatent：rows*cols 个元素（precision = f32/f16）
//   · TernaryPacked2：rows * ceil(K/16) 个 f32 字（2 bit/权重容器）
//   · TernaryPacked3：rows * ceil(K/5) 字节（base-3，1.6 bpw）
//   · aux：形态附加（三值：tau，rows 个 f32）
// base-3 只是**盘上编码**（由 L3 决定），运行期永远是 2-bit 容器 —— 层只认后者。
// ═══════════════════════════════════════════════════════════════════════════
static_assert(sizeof(Scalar) == 4, "v7 打包载荷按 f32 字承载（Scalar 必须是 float）");

// 单行：2-bit 容器字 -> 盘上 payload（2-bit / base-3）
inline void encode_packed_row(bool base3, const std::uint32_t *words, std::size_t k,
                              std::uint8_t *out) noexcept
{
    if (base3)
    {
        nn::trit_pack::pack_row_base3(words, k, out);
        return;
    }
    std::memcpy(out, words, nn::trit_pack::bytes_per_row(k));
}

// 单行：盘上 payload -> 2-bit 容器字（运行期一律 2-bit）
inline void decode_packed_row(bool base3, const std::uint8_t *in, std::size_t k,
                              std::uint32_t *words) noexcept
{
    if (base3)
    {
        nn::trit_pack::unpack_row_base3(in, k, words);
        return;
    }
    std::memcpy(words, in, nn::trit_pack::bytes_per_row(k));
}

struct ParamBlob
{
    ParamKind kind = ParamKind::Dense;
    Precision precision = Precision::F32;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::vector<std::uint8_t> aux;
    std::vector<std::uint8_t> payload;
};

// 未信任长度防御：条目字节数不得超过文件剩余长度
// （与 read_matrix_v5 同一威胁模型：先校验再分配/跳过）
[[nodiscard]] inline Result<void> ensure_blob_fits_(
    std::ifstream &ifs, std::uint64_t aux_len, std::uint64_t payload_len)
{
    const std::streampos pos = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    const std::streampos end = ifs.tellg();
    if (pos < 0 || end < pos)
        NN_FAIL("Cannot determine file size while reading param entry");
    ifs.seekg(pos);
    const auto remain = static_cast<std::uint64_t>(end - pos);
    if (aux_len > remain || payload_len > remain - aux_len)
        NN_FAIL("参数条目长度越界（文件可能损坏或恶意）");
    return {};
}

[[nodiscard]] inline Result<void> write_param_blob(std::ostream &ofs, const ParamBlob &b)
{
    NN_TRY(r0, write_bytes<uint8_t>(ofs, static_cast<uint8_t>(b.kind)));
    NN_TRY(r1, write_bytes<uint8_t>(ofs, precision_tag(b.precision)));
    NN_TRY(r2, write_bytes<uint64_t>(ofs, b.rows));
    NN_TRY(r3, write_bytes<uint64_t>(ofs, b.cols));
    NN_TRY(r4, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(b.aux.size())));
    NN_TRY(r5, write_bytes<uint64_t>(ofs, static_cast<uint64_t>(b.payload.size())));
    if (!b.aux.empty())
    {
        ofs.write(reinterpret_cast<const char *>(b.aux.data()),
                  static_cast<std::streamsize>(b.aux.size()));
        if (!ofs) NN_FAIL("Write error while writing param aux");
    }
    if (!b.payload.empty())
    {
        ofs.write(reinterpret_cast<const char *>(b.payload.data()),
                  static_cast<std::streamsize>(b.payload.size()));
        if (!ofs) NN_FAIL("Write error while writing param payload");
    }
    return {};
}

[[nodiscard]] inline Result<ParamBlob> read_param_blob(std::ifstream &ifs)
{
    auto k_r = read_bytes<uint8_t>(ifs);
    NN_TRY_CHECK(k_r);
    if (*k_r > static_cast<uint8_t>(ParamKind::I8Packed4))
        NN_FAIL("参数条目 kind 非法（" + std::to_string(*k_r) + "）");
    auto pt_r = read_bytes<uint8_t>(ifs);
    NN_TRY_CHECK(pt_r);
    auto rows_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(rows_r);
    auto cols_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(cols_r);
    auto alen_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(alen_r);
    auto plen_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(plen_r);
    NN_TRY_CHECK(ensure_blob_fits_(ifs, *alen_r, *plen_r));
    ParamBlob b;
    b.kind = static_cast<ParamKind>(*k_r);
    b.precision = precision_from_tag(*pt_r);
    b.rows = *rows_r;
    b.cols = *cols_r;
    b.aux.resize(static_cast<std::size_t>(*alen_r));
    b.payload.resize(static_cast<std::size_t>(*plen_r));
    if (!b.aux.empty())
    {
        ifs.read(reinterpret_cast<char *>(b.aux.data()),
                 static_cast<std::streamsize>(b.aux.size()));
        if (!ifs) NN_FAIL("Unexpected end while reading param aux");
    }
    if (!b.payload.empty())
    {
        ifs.read(reinterpret_cast<char *>(b.payload.data()),
                 static_cast<std::streamsize>(b.payload.size()));
        if (!ifs) NN_FAIL("Unexpected end while reading param payload");
    }
    return b;
}

// ── 跳过参数条目（只读条目头，载荷 seek 过去）────────────────────────────
// 读训练状态段（在参数之前）与读参数（状态段在前）时各需跳过对方；把载荷
// 读进内存再丢掉在大模型上是纯浪费（一条 embedding 就几百 MB）。
[[nodiscard]] inline Result<void> skip_param_blob(std::ifstream &ifs)
{
    auto k_r = read_bytes<uint8_t>(ifs);
    NN_TRY_CHECK(k_r);
    if (*k_r > static_cast<uint8_t>(ParamKind::I8Packed4))
        NN_FAIL("参数条目 kind 非法（" + std::to_string(*k_r) + "）");
    NN_TRY_CHECK(read_bytes<uint8_t>(ifs));   // precision tag
    NN_TRY_CHECK(read_bytes<uint64_t>(ifs));  // rows
    NN_TRY_CHECK(read_bytes<uint64_t>(ifs));  // cols
    auto alen_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(alen_r);
    auto plen_r = read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(plen_r);
    NN_TRY_CHECK(ensure_blob_fits_(ifs, *alen_r, *plen_r));
    const auto jump = static_cast<std::streamoff>(*alen_r + *plen_r);
    ifs.seekg(jump, std::ios::cur);
    if (!ifs)
        NN_FAIL("参数条目跳过失败（文件可能损坏或截断）");
    return {};
}

} // namespace detail

// ── ParamEntry ⇄ ParamBlob（宿主字节编解码；base-3 只在这里出现）─────────
[[nodiscard]] inline Result<detail::ParamBlob> param_entry_to_blob(
    ComputeEngine &engine, const ParamEntry &e, ModelRole role)
{
    detail::ParamBlob b;
    b.rows = e.rows;
    b.cols = e.cols;
    switch (e.kind)
    {
        case ParamKind::Dense:
        case ParamKind::TernaryLatent:
        {
            b.kind = e.kind;
            b.precision = e.precision;
            auto m32 = engine.to_matrix(e.payload, Precision::F32);
            NN_TRY_CHECK(m32);
            if (m32->rows() != e.rows || m32->cols() != e.cols)
                NN_FAIL("序列化：参数张量形状与条目不一致");
            if (e.precision == Precision::F16)
            {
                MatrixT<Precision::F16> m16(e.rows, e.cols);
                const auto s = m32->span();
                auto d = m16.span();
                for (std::size_t i = 0; i < s.size(); ++i) d[i] = s[i];
                const auto bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t *>(m16.span().data()),
                    m16.span().size() * sizeof(std::uint16_t));
                b.payload.assign(bytes.begin(), bytes.end());
            }
            else
            {
                const auto bytes = std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t *>(m32->span().data()),
                    m32->span().size() * sizeof(Scalar));
                b.payload.assign(bytes.begin(), bytes.end());
            }
            return b;
        }
        case ParamKind::TernaryPacked2:
        {
            auto cont = engine.to_matrix(e.payload, Precision::F32);
            NN_TRY_CHECK(cont);
            auto tau = engine.to_matrix(e.aux, Precision::F32);
            NN_TRY_CHECK(tau);
            const std::size_t k = e.cols;
            const std::size_t rows = e.rows;
            const std::size_t wpr = nn::trit_pack::words_per_row(k);
            if (cont->rows() != rows || cont->cols() != wpr)
                NN_FAIL("序列化：打包容器形状与条目不一致");
            if (tau->rows() != rows || tau->cols() != 1)
                NN_FAIL("序列化：tau 形状与条目不一致");
            const auto tsp = tau->span();
            b.aux.resize(tsp.size() * sizeof(Scalar));
            std::memcpy(b.aux.data(), tsp.data(), b.aux.size());
            // 推理角色默认落盘为 base-3（1.6 bpw）；断点角色不会走到这里
            const bool base3 = (role == ModelRole::Model);
            b.kind = base3 ? ParamKind::TernaryPacked3 : ParamKind::TernaryPacked2;
            b.precision = Precision::F32;
            const std::size_t per_row = base3
                ? nn::trit_pack::base3_bytes_per_row(k)
                : nn::trit_pack::bytes_per_row(k);
            b.payload.resize(rows * per_row);
            const auto cs = cont->span();
            std::vector<std::uint32_t> words(wpr);
            for (std::size_t r = 0; r < rows; ++r)
            {
                for (std::size_t i = 0; i < wpr; ++i)
                    words[i] = std::bit_cast<std::uint32_t>(cs[r * wpr + i]);
                detail::encode_packed_row(base3, words.data(), k,
                                          b.payload.data() + r * per_row);
            }
            return b;
        }
        default:
            NN_FAIL(std::string("序列化：不支持的参数形态（kind=") +
                    param_kind_name(e.kind) + "）");
    }
}

[[nodiscard]] inline Result<ParamEntry> param_blob_to_entry(
    ComputeEngine &engine, const detail::ParamBlob &b,
    WeightLoadIntent intent = WeightLoadIntent::Inference)
{
    ParamEntry e;
    e.rows = static_cast<std::size_t>(b.rows);
    e.cols = static_cast<std::size_t>(b.cols);
    switch (b.kind)
    {
        case ParamKind::Dense:
        case ParamKind::TernaryLatent:
        {
            Matrix m(e.rows, e.cols);
            if (b.precision == Precision::F16)
            {
                MatrixT<Precision::F16> m16(e.rows, e.cols);
                if (b.payload.size() != m16.span().size() * sizeof(std::uint16_t))
                    NN_FAIL("序列化：f16 载荷长度与形状不一致");
                std::memcpy(m16.span().data(), b.payload.data(), b.payload.size());
                const auto s = m16.span();
                auto d = m.span();
                for (std::size_t i = 0; i < s.size(); ++i)
                    d[i] = static_cast<Scalar>(s[i]);
            }
            else
            {
                if (b.payload.size() != m.size() * sizeof(Scalar))
                    NN_FAIL("序列化：载荷长度与形状不一致");
                std::memcpy(m.span().data(), b.payload.data(), b.payload.size());
            }
            // 载荷恒物化为 f32 张量（f16 升位精确无损）→ import_into_ 负责回写
            auto t = engine.from_matrix(m, Precision::F32);
            NN_TRY_CHECK(t);
            e.kind = b.kind;
            e.precision = b.precision;
            e.payload = std::move(*t);
            return e;
        }
        case ParamKind::TernaryPacked2:
        case ParamKind::TernaryPacked3:
        {
            const std::size_t k = static_cast<std::size_t>(b.cols);
            const std::size_t rows = static_cast<std::size_t>(b.rows);
            const std::size_t wpr = nn::trit_pack::words_per_row(k);
            const bool base3 = (b.kind == ParamKind::TernaryPacked3);
            const std::size_t per_row = base3
                ? nn::trit_pack::base3_bytes_per_row(k)
                : nn::trit_pack::bytes_per_row(k);
            if (b.payload.size() != rows * per_row)
                NN_FAIL("序列化：打包载荷长度与形状不一致");
            Matrix tm(rows, 1);
            if (b.aux.size() != tm.size() * sizeof(Scalar))
                NN_FAIL("序列化：tau（aux）长度与形状不一致");
            std::memcpy(tm.span().data(), b.aux.data(), b.aux.size());
            // ── P1-7：Trainable ⇒ 反量化成 f32 latent（warm start / QAT 续训）──
            // 载荷 → latent 的转换在**宿主侧**（L3 是宿主字节/Matrix 的唯一落点；
            // L2 层不得碰 Matrix）。τ 随条目一起交给层（aux）：层按文件 τ 重建
            // 打包派生物，故首次 forward 的 wq/τ 与文件里的模型逐位一致。
            // 公式与两条不变量的依据见 ternary_pack.hpp 的 `latent_row_scale`。
            if (intent == WeightLoadIntent::Trainable)
            {
                Matrix lat(rows, k);
                std::vector<std::uint32_t> words(wpr);
                const auto ts = tm.span();
                auto ls = lat.span();
                for (std::size_t r = 0; r < rows; ++r)
                {
                    detail::decode_packed_row(base3, b.payload.data() + r * per_row,
                                              k, words.data());
                    nn::trit_pack::dequant_row(words.data(), k, ts[r],
                                               ls.data() + r * k);
                }
                auto lt = engine.from_matrix(lat, Precision::F32);
                NN_TRY_CHECK(lt);
                auto tt = engine.from_matrix(tm, Precision::F32);
                NN_TRY_CHECK(tt);
                e.kind = ParamKind::TernaryLatent;
                e.precision = Precision::F32;
                e.payload = std::move(*lt);
                e.aux = std::move(*tt);
                return e;
            }
            Matrix cont(rows, wpr);
            std::vector<std::uint32_t> words(wpr);
            auto cs = cont.span();
            for (std::size_t r = 0; r < rows; ++r)
            {
                detail::decode_packed_row(base3, b.payload.data() + r * per_row, k,
                                          words.data());
                for (std::size_t i = 0; i < wpr; ++i)
                    cs[r * wpr + i] = std::bit_cast<Scalar>(words[i]);
            }
            auto ct = engine.from_matrix(cont, Precision::F32);
            NN_TRY_CHECK(ct);
            e.kind = ParamKind::TernaryPacked2;   // 运行期形态：内核只认 2-bit 容器
            e.payload = std::move(*ct);
            auto tt = engine.from_matrix(tm, Precision::F32);
            NN_TRY_CHECK(tt);
            e.aux = std::move(*tt);
            return e;
        }
        default:
            NN_FAIL(std::string("序列化：文件里是不支持的参数形态（kind=") +
                    param_kind_name(b.kind) + "）");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  公开 API
//
//  save_model          — 保存 Model + ModelSpec（可选嵌入 tokenizer / 训练状态）
//  load_model          — 从文件加载参数，返回嵌入的 tokenizer 数据（如有）
//  peek_model_spec     — 只读文件头，返回 ModelSpec（不读参数）
//  verify_model_integrity — 整文件校验和 + 尾部标记校验（v8；P2-1）
//  peek_train_state    — 只读训练状态段（计数器/配方/数据集身份；不读权重）
//  load_train_state    — 读训练状态段 + 优化器状态张量（续训路径）
// ═══════════════════════════════════════════════════════════════════════════

// ── 整文件完整性校验（v8；P2-1）──────────────────────────────────────────
// 校验尾部 `[sha256 32B]["NNEND" 5B]`：摘要覆盖其前的**全部字节**。
// 截断/损坏的文件在解析参数之前就明确报错（旧缺陷 C5：静默载入错误权重）。
// 成本 = 一次顺序读（分块哈希，不把文件读进内存）；调用方是
// load_model / load_train_state，`peek_*` 只读头部故不校验。
[[nodiscard]] inline Result<void> verify_model_integrity(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    ifs.seekg(0, std::ios::end);
    const std::streampos end = ifs.tellg();
    if (end < 0)
        NN_FAIL("Cannot determine file size: " + filename);
    const auto total = static_cast<std::uint64_t>(end);
    if (total < MODEL_TAIL_BYTES)
        NN_FAIL("模型文件过小（" + std::to_string(total) + " 字节 < 尾部标记 " +
                std::to_string(MODEL_TAIL_BYTES) + " 字节）：文件被截断或不是模型文件: " + filename);
    ifs.seekg(static_cast<std::streamoff>(total - MODEL_TAIL_BYTES), std::ios::beg);
    if (!ifs)
        NN_FAIL("Seek failed while verifying: " + filename);
    std::array<std::uint8_t, MODEL_TAIL_BYTES> tail{};
    ifs.read(reinterpret_cast<char *>(tail.data()),
             static_cast<std::streamsize>(MODEL_TAIL_BYTES));
    if (ifs.gcount() != static_cast<std::streamsize>(MODEL_TAIL_BYTES))
        NN_FAIL("尾部完整性标记读取失败（文件被截断）: " + filename);
    if (std::string_view(reinterpret_cast<const char *>(tail.data()) + MODEL_DIGEST_BYTES,
                         MODEL_TAIL_MAGIC.size()) != MODEL_TAIL_MAGIC)
        NN_FAIL("模型文件缺少尾部完整性标记（\"NNEND\"）：文件被截断、损坏，"
                "或是旧版本（v7 及以下）文件: " + filename);

    std::array<std::uint8_t, MODEL_DIGEST_BYTES> stored{};
    std::memcpy(stored.data(), tail.data(), MODEL_DIGEST_BYTES);
    NN_TRY(digest, nn::sha256_file_prefix(filename, total - MODEL_TAIL_BYTES));
    if (*digest != stored)
    {
        NN_FAIL("模型文件校验和不匹配（内容损坏）: " + filename +
                "\n  文件记录的摘要: " + nn::sha256_hex(stored).substr(0, 16) +
                "\n  实测摘要      : " + nn::sha256_hex(*digest).substr(0, 16) +
                "\n  该文件不可用，请用上游断点/模型重新产出。");
    }
    return {};
}

// ── 训练状态段（v8；role 门控）──────────────────────────────────────────
// 仅 Checkpoint 角色可携带；Model 角色必须为空（推理文件不带训练状态，
// docs/todo/23 §2.1.1）。`optimizer_blobs` 的顺序由优化器定义
// （`Optimizer::save_state` 的产出顺序即载入时的期望顺序）。
struct CheckpointState
{
    const TrainState *state = nullptr;          // 状态本体（空 = 写 0 长度段）
    std::span<const Tensor> optimizer_blobs{};  // 优化器状态张量（Dense 形态落盘）
};

// ── 保存模型（v8：文件角色 + 参数级 kind + 训练状态段 + 尾部校验和）──────
// role 决定"写哪些参数"：Model = 推理模型（三值层写打包形态，默认 base-3 载荷）；
// Checkpoint = 训练断点（三值层写 f32 latent）+ 可选训练状态段。
[[nodiscard]] inline Result<void> save_model(const std::string &filename,
    Model &model, const ModelSpec &spec, const std::string &tokenizer_data = {},
    ModelRole role = ModelRole::Model, const CheckpointState &ckpt = {})
{
    if (ckpt.state != nullptr && role != ModelRole::Checkpoint)
        NN_FAIL("训练状态段只能写入 Checkpoint 角色（.nnckpt）：推理模型（.nnmodel）"
                "不得携带训练状态（docs/todo/23 §2.1.1）");
    if (ckpt.state == nullptr && !ckpt.optimizer_blobs.empty())
        NN_FAIL("优化器状态张量必须与训练状态一同写入（CheckpointState::state 为空）");

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        NN_FAIL("Cannot open file for writing: " + filename);

    // v8 整文件校验和：所有内容字节经哈希流写入；摘要与尾部标记最后**不经**该流
    detail::Sha256State hasher;
    detail::HashingStreamBuf hbuf(*ofs.rdbuf(), hasher);
    std::ostream os(&hbuf);

    NN_TRY(r, detail::write_header(os, role));
    NN_TRY(r2, detail::write_spec_header(os, spec));

    // ── 训练状态段（v8）：[state_len][kvrec][优化器状态张量条目 × N] ──────
    // 无状态（.nnmodel / 未接线断点）时写 0 长度，布局保持统一。
    {
        TrainState snapshot;
        if (ckpt.state != nullptr)
        {
            snapshot = *ckpt.state;
            snapshot.optimizer_blob_count = ckpt.optimizer_blobs.size();
        }
        const std::string state_bytes = train_state_to_kv(snapshot).serialize();
        const std::uint64_t state_len =
            (ckpt.state != nullptr) ? static_cast<std::uint64_t>(state_bytes.size()) : 0u;
        NN_TRY(r3, detail::write_bytes<uint64_t>(os, state_len));
        if (state_len > 0)
        {
            os.write(state_bytes.data(), static_cast<std::streamsize>(state_bytes.size()));
            if (!os)
                NN_FAIL("Write error while writing train state to: " + filename);
        }
    }
    auto& engine = model.engine();

    // 优化器状态张量（Dense 条目；形状/精度随张量自描述，载入方自行校验）
    for (const Tensor &t : ckpt.optimizer_blobs)
    {
        if (!t.valid())
            NN_FAIL("优化器状态张量非法（未绑定存储）");
        ParamEntry e;
        e.kind = ParamKind::Dense;
        e.precision = t.precision();
        e.rows = t.rows();
        e.cols = t.cols();
        e.payload = t;
        NN_TRY(b, param_entry_to_blob(engine, e, role));
        NN_TRY(w, detail::write_param_blob(os, *b));
    }

    // 逐层逐参数（索引 = 本层参数扁平顺序；`Model::parameters()` 即各层的级联）
    for (std::size_t li = 0; li < model.num_layers(); ++li)
    {
        Layer &L = model.layer_at(li);
        const std::size_t n = L.param_count();
        for (std::size_t i = 0; i < n; ++i)
        {
            NN_TRY(e, L.export_param(i, role));
            NN_TRY(b, param_entry_to_blob(engine, *e, role));
            NN_TRY(w, detail::write_param_blob(os, *b));
        }
    }

    // 非可学习状态（如 BatchNorm 的 running_mean/running_var）：Dense 条目
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        ParamEntry e;
        e.kind = ParamKind::Dense;
        e.precision = e_tensor.get().precision();
        e.rows = e_tensor.get().rows();
        e.cols = e_tensor.get().cols();
        e.payload = e_tensor;
        NN_TRY(b, param_entry_to_blob(engine, e, role));
        NN_TRY(w, detail::write_param_blob(os, *b));
    }

    // tokenizer 数据（长度前缀，0 = 未嵌入；v5=JSON / v6=.nnvocab，按版本分派）
    NN_TRY(r4, detail::write_tokenizer(os, tokenizer_data));
    if (!os)
        NN_FAIL("Write error while saving model to: " + filename);

    // ── 尾部（v8）：整文件校验和 + 完整性标记 ─────────────────────────────
    // 注意：摘要与标记本身**不算进**校验和 ⇒ 直接写 ofs（绕过哈希流）。
    const auto digest = detail::sha256_finalize(hasher);
    ofs.write(reinterpret_cast<const char *>(digest.data()),
              static_cast<std::streamsize>(digest.size()));
    ofs.write(MODEL_TAIL_MAGIC.data(),
              static_cast<std::streamsize>(MODEL_TAIL_MAGIC.size()));
    if (!ofs)
        NN_FAIL("Write error while writing file tail to: " + filename);
    return {};
}

// ── 读取模型规格（只读头部，不读参数/tokenizer） ──────────────────────────
[[nodiscard]] inline Result<ModelSpec> peek_model_spec(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto version_r = detail::read_and_validate_header(ifs);
    NN_TRY_CHECK(version_r);
    return detail::read_spec_header(ifs, *version_r);
}

// ── 只读文件角色（v7；v5/v6 无该字段 → Model）────────────────────────────
[[nodiscard]] inline Result<ModelRole> peek_model_role(const std::string &filename,
                                                       uint32_t *version_out = nullptr)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    ModelRole role = ModelRole::Model;
    auto v = detail::read_and_validate_header(ifs, &role);
    NN_TRY_CHECK(v);
    if (version_out) *version_out = *v;
    return role;
}

// ── 只读训练状态段（v8）：计数器 / 训练配方 / 数据集身份（不读权重）──────
// 供续训路径在建模型**之前**取精度 profile 与计数器、供 CLI/GUI 显示断点状态。
// 没有该段（role=Model，或 v7 及以下 / 未接线的断点）→ 明确报错，调用方据此
// 回落"冷启动语义"（docs/todo/18 §4.2 P1-2）。
[[nodiscard]] inline Result<TrainState> peek_train_state(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto version_r = detail::read_and_validate_header(ifs);
    NN_TRY_CHECK(version_r);
    if (*version_r < 8)
        NN_FAIL("训练状态段自 v8 起才有：" + filename + " 是 v" +
                std::to_string(*version_r) + "（按冷启动语义处理）");
    auto spec_r = detail::read_spec_header(ifs, *version_r);
    NN_TRY_CHECK(spec_r);
    auto len_r = detail::read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(len_r);
    if (*len_r == 0)
        NN_FAIL("文件没有训练状态段（推理模型 .nnmodel，或该断点由旧版本写出）: " +
                filename);
    NN_TRY(state, detail::read_train_state_section(ifs, *len_r));
    return *state;
}

// ── 读训练状态段 + 优化器状态张量（v8；续训路径）────────────────────────
// 优化器状态张量按 `optimizer_blob_count` 顺序读出，**物化为 f32**（与参数
// 条目同一口径），并记录每个张量在文件里的精度（载入方据此校验/转换）。
struct LoadedTrainState
{
    TrainState state;
    std::vector<Tensor> optimizer_blobs;
    std::vector<Precision> optimizer_blob_precisions;
};

[[nodiscard]] inline Result<LoadedTrainState> load_train_state(
    ComputeEngine &engine, const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    auto version_r = detail::read_and_validate_header(ifs);
    NN_TRY_CHECK(version_r);
    if (*version_r < 8)
        NN_FAIL("训练状态段自 v8 起才有：" + filename + " 是 v" +
                std::to_string(*version_r) + "（按冷启动语义处理）");
    NN_TRY_CHECK(verify_model_integrity(filename));
    auto spec_r = detail::read_spec_header(ifs, *version_r);
    NN_TRY_CHECK(spec_r);
    auto len_r = detail::read_bytes<uint64_t>(ifs);
    NN_TRY_CHECK(len_r);
    if (*len_r == 0)
        NN_FAIL("文件没有训练状态段（推理模型 .nnmodel，或该断点由旧版本写出）: " +
                filename);

    LoadedTrainState out;
    NN_TRY(st, detail::read_train_state_section(ifs, *len_r));
    out.state = std::move(*st);
    out.optimizer_blobs.reserve(out.state.optimizer_blob_count);
    out.optimizer_blob_precisions.reserve(out.state.optimizer_blob_count);
    for (std::size_t i = 0; i < out.state.optimizer_blob_count; ++i)
    {
        NN_TRY(b, detail::read_param_blob(ifs));
        if (b->kind != ParamKind::Dense)
            NN_FAIL(std::string("优化器状态条目必须是 Dense 形态，实际 kind=") +
                    param_kind_name(b->kind));
        NN_TRY(e, param_blob_to_entry(engine, *b));
        out.optimizer_blob_precisions.push_back(b->precision);
        out.optimizer_blobs.push_back(std::move(e->payload));
    }
    return out;
}

// ── 加载参数 + tokenizer ───────────────────────────────────────────────
//    返回嵌入的 tokenizer 数据（空串 = 未嵌入）；version_out 可取回文件
//    版本，用于词表数据的 v5(JSON)/v6(.nnvocab kvrec) 分派。
// v5：per-tensor precision tag（读取时自动识别 f32/f16）
// v4：无 precision tag（所有矩阵为 f32）
// v8：读参数前先做整文件校验和校验；并跳过训练状态段（读它用 load_train_state）
// intent（P1-7）：三值层的**打包载荷**是"进部署只读态"还是"反量化成可训练
//    latent"——推理侧默认 `Inference`（0.25 B/权重）；训练侧的 warm start /
//    QAT 续训传 `Trainable`（`.nnckpt` 的 TernaryLatent 载荷两种意图同结果）。
[[nodiscard]] inline Result<std::string> load_model(const std::string &filename, Model &model,
                                                   uint32_t *version_out = nullptr,
                                                   ModelRole *role_out = nullptr,
                                                   WeightLoadIntent intent =
                                                       WeightLoadIntent::Inference)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        NN_FAIL("Cannot open file for reading: " + filename);
    ModelRole role = ModelRole::Model;
    auto version_r = detail::read_and_validate_header(ifs, &role);
    NN_TRY_CHECK(version_r);
    const uint32_t version = *version_r;
    if (version_out) *version_out = version;
    if (role_out) *role_out = role;

    // v8：整文件校验和（截断/损坏在读参数之前明确报错）
    if (version >= 8)
        NN_TRY_CHECK(verify_model_integrity(filename));

    // 读取并校验规格头
    auto spec_r = detail::read_spec_header(ifs, version);
    NN_TRY_CHECK(spec_r);
    if (auto stored = model.spec(); stored)
    {
        if (!spec_matches(*stored, *spec_r))
        {
            NN_FAIL("Model architecture mismatch while loading '" + filename + "': "                 "model expects " + spec_summary(*stored) +                 " but file contains " + spec_summary(*spec_r));
        }
    }

    // v8：跳过训练状态段（[state_len][kvrec][优化器状态张量条目 × N]）——
    // 读权重不需要它；kvrec 仍要解析出 blob 数才知道跳过几条张量条目。
    if (version >= 8)
    {
        auto state_len_r = detail::read_bytes<uint64_t>(ifs);
        NN_TRY_CHECK(state_len_r);
        NN_TRY(st, detail::read_train_state_section(ifs, *state_len_r));
        for (std::size_t i = 0; i < st->optimizer_blob_count; ++i)
            NN_TRY_CHECK(detail::skip_param_blob(ifs));
    }

    auto& engine = model.engine();
    auto params = model.parameters();

    if (version >= 7)
    {
        // ── v7：参数级 kind（由拥有该参数的那个层决定形态）─────────────
        for (std::size_t li = 0; li < model.num_layers(); ++li)
        {
            Layer &L = model.layer_at(li);
            const std::size_t n = L.param_count();
            for (std::size_t i = 0; i < n; ++i)
            {
                NN_TRY(b, detail::read_param_blob(ifs));
                NN_TRY(e, param_blob_to_entry(engine, *b, intent));
                NN_TRY(r, L.import_param(i, *e));
            }
        }
        // 非可学习状态：Dense 条目
        auto extras7 = model.extra_state();
        for (auto& e_tensor : extras7)
        {
            NN_TRY(b, detail::read_param_blob(ifs));
            if (b->kind != ParamKind::Dense)
                NN_FAIL(std::string("非可学习状态必须是 Dense 条目，实际 kind=") +
                        param_kind_name(b->kind));
            NN_TRY(e, param_blob_to_entry(engine, *b));
            auto m = engine.to_matrix(e->payload, Precision::F32);
            NN_TRY_CHECK(m);
            NN_TRY(cf, engine.copy_from(e_tensor, *m));
        }
    }
    else if (version >= 5)
    {
        // ── v5：per-tensor precision tag ─────────────────────────────
        for (auto& p_tensor : params)
        {
            auto result_r = detail::read_matrix_v5(ifs);
            NN_TRY_CHECK(result_r);
            // file_prec 由 read_matrix_v5 校验（F32/F16）；值路径不再分精度
            // 分支，故只取矩阵本体（file_matrix 恒为 f32 宿主值）。
            const Matrix& file_matrix = result_r->second;

            // 统一接口：`copy_from` 把宿主 Matrix 写进目标张量。
            // M6 D6-1（15 §4.6-1）整改：原实现分两支且都白做——同精度支
            // `from_matrix(file_matrix, target_p)` 上传后**丢弃结果**再 copy_from
            // （每参数一次全量上传白做）；跨精度支 `上传→cast→to_matrix 下载→
            // 再 copy_from` （多两个 PCIe 往返）。file_matrix 恒为 f32 宿主值
            // （f16 文件数据读入时已精确升 f32，f16→f32→f16 位不变），故四种
            // (file_prec × target) 组合与原路径逐位一致，收敛为一行。
            NN_TRY(r, engine.copy_from(p_tensor, file_matrix));
        }
    }
    else
    {
        // ── v4：无 precision tag（所有矩阵为 f32）───────────────────
        for (auto& p_tensor : params)
        {
            Matrix tmp(p_tensor.get().rows(), p_tensor.get().cols());
            NN_TRY(r, detail::read_matrix(ifs, tmp));
            NN_TRY(r2, engine.copy_from(p_tensor, tmp));
        }
    }

    // 非可学习状态（**v7 已在上面按参数条目读过** → 这里只服务 v5/v6 老文件）
    auto extras = model.extra_state();
    for (auto& e_tensor : extras)
    {
        if (version >= 7) break;
        if (ifs.peek() == EOF) break;

        if (version >= 5)
        {
            auto result_r = detail::read_matrix_v5(ifs);
            NN_TRY_CHECK(result_r);
            auto& [file_prec, file_matrix] = *result_r;
            NN_TRY(r, engine.copy_from(e_tensor, file_matrix));
        }
        else
        {
            Matrix tmp(e_tensor.get().rows(), e_tensor.get().cols());
            auto mr = detail::read_matrix(ifs, tmp);
            NN_TRY_CHECK(mr);
            NN_TRY(r, engine.copy_from(e_tensor, tmp));
        }
    }

    // 读取 tokenizer 数据（v5=JSON / v6=.nnvocab kvrec，解释权在调用方）
    std::string tokenizer_data;
    {
        auto tok_r = detail::read_tokenizer(ifs);
        NN_TRY_CHECK(tok_r);
        tokenizer_data = std::move(*tok_r);
    }

    if (!ifs)
        NN_FAIL("Read error while loading model from: " + filename);
    return tokenizer_data;
}

// ── 统一入口（v5/v6/v7 同一 magic，自动按版本分流）───────────────────────
[[nodiscard]] inline Result<ModelSpec> peek_any_model_spec(const std::string &filename)
{
    return peek_model_spec(filename);
}

[[nodiscard]] inline Result<void> load_any_model(const std::string &filename, Model &model,
                                                 std::string *tokenizer_out = nullptr,
                                                 ModelRole *role_out = nullptr)
{
    auto tok_r = load_model(filename, model, nullptr, role_out);
    NN_TRY_CHECK(tok_r);
    if (tokenizer_out) *tokenizer_out = std::move(*tok_r);
    return {};
}

} // namespace nn
