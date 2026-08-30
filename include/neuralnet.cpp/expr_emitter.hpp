#ifndef NN_EXPR_EMITTER_HPP
#define NN_EXPR_EMITTER_HPP

// ═══════════════════════════════════════════════════════════════════════════
//  expr_emitter.hpp — 后端 emitter 抽象（IR-D）
//
//  对应文档 `docs/11-ir-optimization.md` IR-D：把后端代码生成从 GLSL 专用
//  抽象为 emitter 接口，实现"一份 canonical IR → 多后端代码"：
//
//      IR (canonical ExprSpec) → GlslEmitter / CudaEmitter / …
//
//  设计：
//    - ExprEmitter 是纯接口：给定 name + canonical ExprSpec，产出目标后端
//      源码字符串（失败返回空串，语义与 glsl_gen 的约定一致）。
//    - 现有 generate_glsl / generate_glsl_reduce 成为 GlslEmitter 的实现
//      （见 glsl_gen.hpp），gen_fused 等消费方经接口调用，不再与 GLSL 绑定。
//    - 具体 emitter 类在各自头文件（glsl_gen.hpp / cpu_emitter.hpp）提供，
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

    // 后端名（"glsl" / "cpu" / ...），供注册表/日志识别
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    // 逐元素 kernel 源码；失败返回空串（调用方报错）
    [[nodiscard]] virtual std::string generate(
        const std::string& name_, const ExprSpec& spec) = 0;

    // 归约 kernel 源码；不支持（混合轴/超槽）返回空串
    [[nodiscard]] virtual std::string generate_reduce(
        const std::string& name_, const ExprSpec& spec) = 0;
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

#endif // NN_EXPR_EMITTER_HPP
