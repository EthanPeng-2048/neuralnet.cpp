#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  expr_emitter.hpp — 后端 emitter 抽象（IR-D）
//
//  对应文档 `docs/internal/03-ir-optimization.md` IR-D：把后端代码生成从 GLSL 专用
//  抽象为 emitter 接口，实现"一份 canonical IR → 多后端代码"：
//
//      IR (canonical ExprSpec) → GlslEmitter（当前唯一注册后端）
//
//  设计：
//    - ExprEmitter 是纯接口：给定 name + canonical ExprSpec，产出目标后端
//      源码字符串（失败返回空串，语义与 glsl_gen 的约定一致）。
//    - 现有 generate_glsl / generate_glsl_reduce 成为 GlslEmitter 的实现
//      （见 expr_glsl_gen.hpp），gen_fused 等消费方经接口调用，与具体后端解耦。
//    - 具体 emitter 类在各自头文件（expr_glsl_gen.hpp）提供，
//      本头只定义接口 + 注册表（按后端名选择 emitter 工厂）。
//  注意：本头供构建期生成器/工具使用，运行时无需包含（与 glsl_gen 一致）。
// ═══════════════════════════════════════════════════════════════════════════

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "expr_spec.hpp"

namespace nn
{

// ── emitter 抽象接口 ──────────────────────────────────────────────────────
// 实现约定（与现有 glsl_gen 语义对齐）：
//   - generate：逐元素 kernel（无归约视图/归约指令）。
//   - generate_reduce：归约 kernel（expr_spec_reduce_axis >= 0）；
//     混合轴（== -2）或不支持的归约结构返回空串。
//   - 产物均以"自动生成，请勿手动编辑"注释开头，便于区分手工代码。
class ExprEmitter
{
public:
    virtual ~ExprEmitter() = default;

    // 后端名（如 "glsl"），供注册表/日志识别
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // 逐元素 kernel 源码；失败返回空串（调用方报错）
    // sig（可选）：存储精度签名（0 = 全 f32 变体；
    // 见 expr_spec.hpp 的 ExprPrecSig）。带类型变体（in-kernel f16）不支持时
    // 返回空串 → 上层跳过该变体，运行时回退边界 cast。
    // native16：原生 f16 算术变体（GL_EXT_shader_explicit_arithmetic_types_
    // float16，全链 float16_t 计算）。仅当 expr_prec_sig_native16(spec, sig)
    // 为真时请求；否则返回空串（上层跳过）。f32 算术变体（native16=false）
    // 与 native16 变体**并存**——设备按 shaderFloat16 能力选 pipeline。
    [[nodiscard]] virtual std::string generate(
        const std::string& name_, const ExprSpec& spec,
        ExprPrecSig sig = 0, bool native16 = false) = 0;

    // 归约 kernel 源码；不支持（混合轴/超槽/带类型变体）返回空串
    [[nodiscard]] virtual std::string generate_reduce(
        const std::string& name_, const ExprSpec& spec,
        ExprPrecSig sig = 0) = 0;

    // 批量变体 kernel 源码（键 = key#b，P3 跨链批量派发）：同结构多实例
    // 经实例表（uvec2 数组，stride = 输入+输出+尾槽）+ buffer device address
    // 一次 dispatch。资格 = expr_spec_batchable（纯逐元素）。sig = 变体身份
    // 签名：0 = V0 全 f32（**含 extras 复合结构**，extras 引用恒 f32）；具体
    // 签名 = 静态特化变体，仅 **extras 为空**的结构（extras 的元素类型不在签名
    // 编码里 → 带 extras 的 f16 结构仍逐 kernel 降级）。运行期分派哨兵不支持。
    // 默认返回空串 = 该后端不支持 → 运行期回退逐 kernel 派发（闭合世界降级，
    // 不是错误）。
    [[nodiscard]] virtual std::string generate_batched(
        const std::string&, const ExprSpec&, ExprPrecSig = 0) const { return {}; }

    // matmul 细 tile 变体 kernel 源码（键 = key#f，P0 wgrad 网格饥饿）：
    // 输出块 64×64 → 16×16、16×16 线程 → 8×8、每线程 4×4 → 2×2 输出
    // （R=2：网格摊满 SM × 线程内 ILP=4 的甜点，取舍数据见
    // EXPR_MATMUL_BLOCK_FINE），供 64-tile 网格工作组过少的大 K 反传形状
    // 派发；每输出元素 K 累加序与 64-tile 变体逐位一致。
    // 资格 = 纯 matmul 形态（spec.matmul、无 fold、raxis<0）；sig = 变体身份
    // 签名（0 = V0 全 f32；具体签名 = 该结构的静态特化变体，见
    // EXPR_PREC_STORAGE_SUFFIX / expr_prec_sig_key）。运行期分派哨兵不支持。
    // 默认返回空串 = 该后端不支持 → 构建期不产出条目、运行期留在 64-tile
    // 路径（降级，不是错误）。
    [[nodiscard]] virtual std::string generate_matmul_fine(
        const std::string&, const ExprSpec&, ExprPrecSig = 0) const { return {}; }

    // matmul K 切分变体 kernel 源码（键 = key#ks，wgrad 二次归约）：
    // 细 tile 网格仍低于 EXPR_MATMUL_KSPLIT_MIN_WGS 时，把求和维 K 切成
    // S 段、dispatch x × S——每 WG 只跑一段 K 并写 (S·rows, cols) 部分和
    // （**不跑尾链**：尾链非线性，须在求和之后统一执行）。PC 比 V0 多一槽
    // ks_span（16 对齐）。⚠ 跨段 K 累加序与未切分不逐位一致 → 移动 GPU
    // 字节锚（已裁定放行）。资格 = 细 tile 同判据（纯 matmul、静态签名）,
    // **输出恒 f32**（写的是部分和缓冲，与 sig 的输出精度无关），
    // 由伴生 #kr（generate_matmul_ksplit_reduce）完成二次归约 + 尾链。
    // 默认返回空串 = 该后端不支持 → 构建期不产出条目、运行期留细 tile
    // 路径（降级，不是错误）。
    [[nodiscard]] virtual std::string generate_matmul_ksplit(
        const std::string&, const ExprSpec&, ExprPrecSig = 0) const { return {}; }

    // K 切分伴生归约 kernel 源码（键 = key#kr）：绑定 = 原输入 0..N-1 +
    // 部分和（binding N）+ 输出（binding N+1）；读段数经 PC（mm_k 槽位）。
    // 求和按段号升序左折（确定性），随后跑与 V0 同源的尾链。
    // 与 generate_matmul_ksplit 配对产出（有 #ks 才有 #kr）。
    // 默认返回空串 = 该后端不支持 → 构建期不产出条目（#ks 同步不产出）。
    [[nodiscard]] virtual std::string generate_matmul_ksplit_reduce(
        const std::string&, const ExprSpec&, ExprPrecSig = 0) const { return {}; }
};

// ── 简单注册表：按后端名选择 emitter 工厂 ────────────────────────────────
// 供生成器工具（gen_fused 等）按 --backend 选择目标 emitter。
// 具体 emitter 类型在各自头文件中向本注册表登记（见 GlslEmitter）。
// 线程安全：初始化期单线程登记（静态初始化），运行期只读查询。
namespace emitter_registry
{

// 工厂返回 unique_ptr（铁律 2：无裸指针所有权）；make() 直接转发。
using Factory = std::unique_ptr<ExprEmitter>(*)();

// 后端表（静态初始化，见各 emitter 头文件的登记语句）
inline std::unordered_map<std::string, Factory>& backends()
{
    static std::unordered_map<std::string, Factory> m;
    return m;
}

// 登记后端（返回 false 表示已存在同名后端，拒绝覆盖）
inline bool register_backend(std::string_view name, Factory factory)
{
    auto& m = backends();
    return m.emplace(std::string(name), factory).second;
}

// 按名创建 emitter；未登记返回 nullptr
[[nodiscard]] inline std::unique_ptr<ExprEmitter> make(std::string_view name)
{
    const auto& m = backends();
    const auto it = m.find(std::string(name));
    if (it == m.end())
        return nullptr;
    return it->second();
}

// 所有已登记后端名（调试/日志）
[[nodiscard]] inline std::vector<std::string> names()
{
    std::vector<std::string> out;
    out.reserve(backends().size());
    for (const auto& [k, v] : backends())
        out.push_back(k);
    return out;
}

} // namespace emitter_registry

} // namespace nn

