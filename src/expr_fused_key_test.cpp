// ── expr_fused_key_test — "结构 = 表达式类型" 不变量（AOT 自登记的前提）────────
// 背景（docs/development/02 §自登记锚点）：`FusedAnchor<Expr>` 在构建期按表达式
// **类型**默认构造一个"符号实例"折叠出结构并登记；运行时同一调用点折叠出的 key
// 必须与它一致，否则闭合世界 miss。成立条件 = **key 只描述结构**：
//   ① 常量池**值**不进 key（只喂个数；值经 push constant `c<i>` 传给 shader）
//   ② matmul `transA/transB` 不进 key（运行期 operand layout，PC `mm_trans`）
//   ③ 分组归约 `R` 不进 key（运行期视图参数，PC `vp`）
// 本测试同时锁死**符号实例 key ≡ 真实实例 key** —— 这是自登记机制的全部依据。
// 若有人重新引入"运行期值决定结构"，这里会立刻红。
//
// 纯 CPU、无 GPU 依赖。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <string>
#include <type_traits>

#include <neuralnet.cpp/compute_cpu_engine.hpp>
#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/expr_registry.hpp>
#include <neuralnet.cpp/expr_spec.hpp>

namespace
{
int g_fail_fk = 0;

[[nodiscard]] std::string fk_key(const nn::ExprSpec& s)
{ return nn::expr_spec_key(nn::canonicalize_expr_spec(s)); }

void fk_eq(const char* name, const std::string& a, const std::string& b,
           const char* why)
{
    if (a == b)
    {
        std::printf("[PASS] %s（%s）key=%s\n", name, why, a.c_str());
        return;
    }
    ++g_fail_fk;
    std::printf("[FAIL] %s：key 应相同但不同（%s）\n        a=%s\n        b=%s\n",
                name, why, a.c_str(), b.c_str());
}
} // namespace

int main()
{
    nn::CpuEngine eng;
    // 真实实例的操作数（形状/精度都真实）
    const nn::Tensor x  = eng.create_tensor(8, 4, nn::Precision::F32);
    const nn::Tensor w  = eng.create_tensor(5, 8, nn::Precision::F32);
    const nn::Tensor b  = eng.create_tensor(5, 1, nn::Precision::F32);
    const nn::Tensor g  = eng.create_tensor(5, 4, nn::Precision::F32);
    const nn::Tensor S{};   // 符号张量（默认构造：未绑定、无存储）

    using namespace nn;

    // ── ① 符号实例 key ≡ 真实实例 key（FusedAnchor 的全部依据）──────────────
    struct Pair { const char* name; std::string real; std::string sym; };
    const Pair pairs[] = {
        {"relu_fwd", fk_key(dsl::to_expr_spec(
                         dsl::max(dsl::leaf(x), Scalar{0})).first),
                     fk_key(dsl::to_expr_spec(
                         dsl::max(dsl::leaf(S), Scalar{0})).first)},
        {"linear_fwd(matmul+row_broadcast)",
                     fk_key(dsl::to_expr_spec(
                         dsl::matmul(w, x, false, false) + dsl::row_broadcast(b)).first),
                     fk_key(dsl::to_expr_spec(
                         dsl::matmul(S, S, false, false) + dsl::row_broadcast(S)).first)},
        {"matmul_transA",
                     fk_key(dsl::to_expr_spec(dsl::matmul(w, g, true, false)).first),
                     fk_key(dsl::to_expr_spec(dsl::matmul(S, S, true, false)).first)},
        {"matmul_transB",
                     fk_key(dsl::to_expr_spec(dsl::matmul(g, x, false, true)).first),
                     fk_key(dsl::to_expr_spec(dsl::matmul(S, S, false, true)).first)},
        {"row_reduce_sum",
                     fk_key(dsl::to_expr_spec(dsl::row_reduce_sum(dsl::leaf(g))).first),
                     fk_key(dsl::to_expr_spec(dsl::row_reduce_sum(dsl::leaf(S))).first)},
        {"grouped_reduce_max",
                     fk_key(dsl::to_expr_spec(
                         dsl::grouped_reduce_max(g, 4u) + dsl::rparam(Scalar{0})).first),
                     fk_key(dsl::to_expr_spec(
                         dsl::grouped_reduce_max(S, 1u) + dsl::rparam(Scalar{0})).first)},
        {"rotate_half",
                     fk_key(dsl::to_expr_spec(dsl::rotate_half(x, 16u)).first),
                     fk_key(dsl::to_expr_spec(dsl::rotate_half(S, 1u)).first)},
        {"row_access",
                     fk_key(dsl::to_expr_spec(dsl::row_access(x, 8u, 8u)).first),
                     fk_key(dsl::to_expr_spec(dsl::row_access(S, 1u, 1u)).first)},
    };
    for (const Pair& p : pairs)
        fk_eq(p.name, p.real, p.sym, "符号实例 key ≡ 真实实例 key");

    // 默认构造可用性（锚点要求整棵表达式树可符号构造）
    static_assert(std::is_default_constructible_v<nn::dsl::TensorRef>);
    static_assert(std::is_default_constructible_v<nn::dsl::GroupedReduceSumRef>);
    static_assert(std::is_default_constructible_v<nn::dsl::GroupedReduceMaxRef>);

    // ── ② 常量池**值**不进 key（只喂个数）────────────────────────────────
    fk_eq("const_value_not_in_key",
          fk_key(dsl::to_expr_spec(dsl::max(dsl::leaf(x), Scalar{1e-6f})).first),
          fk_key(dsl::to_expr_spec(dsl::max(dsl::leaf(x), Scalar{1e-5f})).first),
          "常量池只喂个数");

    // ── ③ matmul 转置不进 key（运行期 operand layout）─────────────────────
    {
        const std::string k00 = fk_key(dsl::to_expr_spec(
            dsl::matmul(w, x, false, false) + dsl::row_broadcast(b)).first);
        const std::string k01 = fk_key(dsl::to_expr_spec(
            dsl::matmul(w, x, false, true) + dsl::row_broadcast(b)).first);
        const std::string k10 = fk_key(dsl::to_expr_spec(
            dsl::matmul(w, x, true, false) + dsl::row_broadcast(b)).first);
        const std::string k11 = fk_key(dsl::to_expr_spec(
            dsl::matmul(w, x, true, true) + dsl::row_broadcast(b)).first);
        fk_eq("matmul_trans_not_in_key", k00, k01, "4 种转置组合共享一个 shader");
        fk_eq("matmul_trans_not_in_key", k00, k10, "4 种转置组合共享一个 shader");
        fk_eq("matmul_trans_not_in_key", k00, k11, "4 种转置组合共享一个 shader");

        // 运行期描述子位编码（GPU 端据此选加载路径；编码错 = 静默读错布局）
        const auto trans_of = [](bool a, bool c) {
            nn::ExprSpec s;
            s.views  = {nn::expr::linear(), nn::expr::linear()};
            s.matmul = nn::MatmulSpec{0, 1, static_cast<std::uint8_t>(a ? 1u : 0u),
                                      static_cast<std::uint8_t>(c ? 1u : 0u), 1};
            return nn::expr_spec_runtime_matmul_trans(s);
        };
        const bool enc_ok = trans_of(false, false) == 0u && trans_of(true, false) == 1u
                         && trans_of(false, true) == 2u && trans_of(true, true) == 3u;
        if (enc_ok) std::printf("[PASS] expr_spec_runtime_matmul_trans 位编码\n");
        else { ++g_fail_fk; std::printf("[FAIL] matmul trans 位编码不是 bit0/bit1\n"); }
    }

    // ── ④ 分组归约 R 不进 key（运行期视图参数）────────────────────────────
    fk_eq("grouped_reduce_R_not_in_key",
          fk_key(dsl::to_expr_spec(
              dsl::grouped_reduce_max(x, 4u) + dsl::rparam(Scalar{0})).first),
          fk_key(dsl::to_expr_spec(
              dsl::grouped_reduce_max(x, 9u) + dsl::rparam(Scalar{0})).first),
          "R 经 vp 槽运行期传入");
    // sum / max 是**两种视图**（结构）→ 必须仍然区分
    {
        const std::string ks = fk_key(dsl::to_expr_spec(
            dsl::grouped_reduce_sum(x, 4u) + dsl::rparam(Scalar{0})).first);
        const std::string km = fk_key(dsl::to_expr_spec(
            dsl::grouped_reduce_max(x, 4u) + dsl::rparam(Scalar{0})).first);
        if (ks != km) std::printf("[PASS] sum/max 是不同视图（key 不同）\n");
        else { ++g_fail_fk; std::printf("[FAIL] grouped_reduce sum/max key 应不同\n"); }
    }

    std::printf("\nexpr_fused_key_test: %d failure(s)\n", g_fail_fk);
    return g_fail_fk != 0 ? 1 : 0;
}
