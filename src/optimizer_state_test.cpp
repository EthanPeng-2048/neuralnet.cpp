// ── optimizer_state_test — 优化器断点状态导出/载入（P1-2）────────────────────
//
// 验收（docs/internal/18-roadmap.md §4.2 P1-2）：
//   "TDR 模拟重启后动量非零且 loss 轨迹连续"。本测试把它机械化：
//   ① 状态存在性（每个优化器要么有状态、要么被显式登记为无状态）；
//   ② 重启后状态非零；
//   ③ **中途存状态 → 重启续跑** 与 **不中断直跑** 的参数**逐位一致**（轨迹连续）；
//   ④ 负例：不恢复状态（冷启动）续跑必须**不同**（否则 ③ 不具判别力）；
//   ⑤ 张量数/形状不符 → 明确报错（不静默接受部分状态）。
//
// 覆盖优化器 = `nn::optimizer_table()` 全表（sgd / sgd_momentum / adam / adamw /
// adamwquant / muon），名单与期望状态数取自该表（RV-04 唯一真源）。
//
// 退出码：0 = 通过，1 = 失败。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"

#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

namespace
{
int g_fail = 0;

constexpr std::size_t kParams = 2;   // 两个参数张量（含一个 (n,1)，覆盖 Muon 的非 2D 分支）

// ── 固定夹具：参数/梯度张量 + 确定性梯度序列 ─────────────────────────────
struct Fixture
{
    std::shared_ptr<nn::CpuEngine> engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    std::vector<nn::Tensor> params;
    std::vector<nn::Tensor> grads;
    std::vector<nn::TensorRef> pref;
    std::vector<nn::TensorRef> gref;

    void build()
    {
        params.reserve(kParams);
        grads.reserve(kParams);
        params.push_back(engine.create_tensor(4, 3, nn::Precision::F32, nn::InitSpec::zero()));
        params.push_back(engine.create_tensor(3, 1, nn::Precision::F32, nn::InitSpec::zero()));
        grads.push_back(engine.create_tensor(4, 3, nn::Precision::F32, nn::InitSpec::zero()));
        grads.push_back(engine.create_tensor(3, 1, nn::Precision::F32, nn::InitSpec::zero()));
        for (auto& p : params) pref.push_back(std::ref(p));
        for (auto& g : grads) gref.push_back(std::ref(g));
        set_params_from_seed(/*seed=*/1);
        set_grads(0);
    }

    // 参数初值：确定性、非对称、非零（重启对比的前提是两次起点相同）
    void set_params_from_seed(int seed)
    {
        for (std::size_t pi = 0; pi < params.size(); ++pi)
        {
            nn::Matrix m(params[pi].rows(), params[pi].cols());
            auto s = m.span();
            for (std::size_t k = 0; k < s.size(); ++k)
                s[k] = 0.05f * static_cast<float>((static_cast<int>(k) * 7 + seed * 3) % 11 - 5)
                     + 0.001f * static_cast<float>(pi);
            NN_ASSERT_ALWAYS(engine.copy_from(params[pi], m), "upload params");
        }
    }

    // 梯度：确定性序列（与 step 号绑定；同一 step 号给同一批梯度）
    void set_grads(std::size_t step)
    {
        for (std::size_t pi = 0; pi < grads.size(); ++pi)
        {
            nn::Matrix m(grads[pi].rows(), grads[pi].cols());
            auto s = m.span();
            for (std::size_t k = 0; k < s.size(); ++k)
            {
                const int v = static_cast<int>((k * 13 + step * 29 + pi * 5) % 19) - 9;
                s[k] = 0.01f * static_cast<float>(v);
            }
            NN_ASSERT_ALWAYS(engine.copy_from(grads[pi], m), "upload grads");
        }
    }

    [[nodiscard]] std::vector<nn::Matrix> snapshot()
    {
        std::vector<nn::Matrix> out;
        for (const auto& p : params) out.push_back(download(engine, p));
        return out;
    }

    // 单步：设梯度 → step（参数/梯度/nn 序列完全确定）
    void step(nn::Optimizer& opt, std::size_t step_id)
    {
        set_grads(step_id);
        NN_ASSERT_ALWAYS(opt.step(), "optimizer step");
    }
};

[[nodiscard]] bool same_params(const std::vector<nn::Matrix>& a,
                               const std::vector<nn::Matrix>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].rows() != b[i].rows() || a[i].cols() != b[i].cols()) return false;
        const auto sa = a[i].span();
        const auto sb = b[i].span();
        for (std::size_t k = 0; k < sa.size(); ++k)
            if (sa[k] != sb[k]) return false;
    }
    return true;
}

[[nodiscard]] std::unique_ptr<nn::Optimizer> make(Fixture& fx, const std::string& name)
{
    return nn::create_optimizer(name, fx.engine, fx.pref, fx.gref,
                                /*lr=*/0.01f, /*weight_decay=*/0.01f,
                                nn::PrecisionProfile{}, 0.9f, 0.999f);
}

// 表里声明的"逻辑状态缓冲数" → 期望的张量数（每参数）
[[nodiscard]] std::size_t expected_tensors_per_param(std::string_view name)
{
    if (name == "sgd") return 0;
    if (name == "adamwquant") return 4;   // m16/v16 载荷 + sm/sv 尺度
    if (name == "adam" || name == "adamw") return 2;
    return 1;                              // sgd_momentum / muon
}

// ── [1][2] 状态存在性 + 非零 ───────────────────────────────────────────
void test_state_presence()
{
    std::cout << "\n── [1] 全表状态存在性（kOptimizerTable 唯一真源）──\n";
    for (const auto& meta : nn::optimizer_table())
    {
        Fixture fx;
        fx.build();
        auto opt = make(fx, std::string(meta.name));
        CHECK(opt != nullptr, ("create_optimizer: " + std::string(meta.name)).c_str());
        if (!opt) continue;

        for (std::size_t s = 0; s < 3; ++s) fx.step(*opt, s);

        std::vector<nn::Tensor> state;
        auto sr = opt->save_state(state);
        CHECK(sr.has_value(), ("save_state: " + std::string(meta.name)).c_str());
        const std::size_t want = expected_tensors_per_param(meta.name) * kParams;
        CHECK(state.size() == want,
              ("状态张量数 = " + std::to_string(want) + "（" + std::string(meta.name) +
               "），实际 " + std::to_string(state.size())).c_str());
        if (meta.state_bufs == 0 && meta.name != "adamwquant")
            CHECK(state.empty(), "无状态优化器（sgd）导出的状态为空");
        else if (state.empty())
            CHECK(false, "有状态优化器不得导出空状态（新优化器必须实现 save_state）");

        // 步计数器：Adam 家族非零；无状态优化器恒 0
        const bool adam_family = meta.name == "adam" || meta.name == "adamw" ||
                                 meta.name == "adamwquant";
        CHECK(opt->state_step() == (adam_family ? 3u : 0u),
              ("state_step: " + std::string(meta.name)).c_str());

        // 动量非零（TDR 重启的验收口径）
        if (!state.empty())
        {
            bool nonzero = false;
            for (std::size_t i = 0; i < state.size() && !nonzero; ++i)
            {
                const auto m = download(fx.engine, state[i]);
                for (float v : m.span())
                    if (v != 0.0f) { nonzero = true; break; }
            }
            CHECK(nonzero, ("重启后状态非零（" + std::string(meta.name) + "）").c_str());
        }
    }
}

// ── [3][4] 重启续跑 = 直跑（逐位）；冷启动续跑 ≠ 直跑 ───────────────────
void test_resume_continuity()
{
    std::cout << "\n── [3] 中途存状态 → 重启续跑 vs 直跑（逐位一致）──\n";
    constexpr std::size_t kTotal = 10;
    constexpr std::size_t kSplit = 5;

    for (const auto& meta : nn::optimizer_table())
    {
        const std::string name(meta.name);

        // A：不中断直跑
        Fixture a;
        a.build();
        auto opt_a = make(a, name);
        if (!opt_a) { CHECK(false, "create_optimizer (A)"); continue; }
        for (std::size_t s = 0; s < kTotal; ++s) a.step(*opt_a, s);
        const auto params_a = a.snapshot();

        // B：跑到中途 → 存状态 → 新建优化器（模拟进程重启）→ 载入续跑
        Fixture b;
        b.build();
        b.set_params_from_seed(1);   // 与 A 同一初值
        auto opt_b = make(b, name);
        if (!opt_b) { CHECK(false, "create_optimizer (B)"); continue; }
        for (std::size_t s = 0; s < kSplit; ++s) b.step(*opt_b, s);
        std::vector<nn::Tensor> saved;
        auto sr = opt_b->save_state(saved);
        CHECK(sr.has_value(), ("save_state（中断点）: " + name).c_str());
        const std::uint64_t saved_step = opt_b->state_step();

        opt_b.reset();                       // "进程死了"：优化器连同状态一起消失
        auto opt_b2 = make(b, name);         // 重启后重新构造（状态为零初始化）
        if (!opt_b2) { CHECK(false, "create_optimizer (B2)"); continue; }
        opt_b2->set_state_step(saved_step);
        auto lr2 = opt_b2->load_state(saved);
        CHECK(lr2.has_value(), ("load_state（重启）: " + name).c_str());
        for (std::size_t s = kSplit; s < kTotal; ++s) b.step(*opt_b2, s);
        CHECK(same_params(params_a, b.snapshot()),
              ("重启续跑与直跑参数逐位一致（" + name + "）").c_str());

        // 负例：不恢复状态（冷启动续跑）必须与直跑**不同** —— 否则上一条不具判别力
        // （无状态优化器是例外：没有状态可丢，冷启动续跑**天然**与直跑一致）
        const bool stateful = !saved.empty();
        Fixture c;
        c.build();
        c.set_params_from_seed(1);
        auto opt_c = make(c, name);
        if (!opt_c) { CHECK(false, "create_optimizer (C)"); continue; }
        for (std::size_t s = 0; s < kSplit; ++s) c.step(*opt_c, s);
        opt_c.reset();
        auto opt_c2 = make(c, name);         // 不载入状态、不恢复 t
        if (!opt_c2) { CHECK(false, "create_optimizer (C2)"); continue; }
        for (std::size_t s = kSplit; s < kTotal; ++s) c.step(*opt_c2, s);
        if (stateful)
            CHECK(!same_params(params_a, c.snapshot()),
                  ("冷启动续跑与直跑不同（负例具判别力：" + name + "）").c_str());
        else
            CHECK(same_params(params_a, c.snapshot()),
                  ("无状态优化器（" + name + "）冷启动续跑 = 直跑（无状态可丢）").c_str());
    }
}

// ── [5] 张量数/形状不符 → 报错 ────────────────────────────────────────
void test_state_mismatch()
{
    std::cout << "\n── [5] 状态不匹配 → 明确报错 ──\n";
    Fixture fx;
    fx.build();
    auto opt = make(fx, "adam");
    if (!opt) { CHECK(false, "create_optimizer"); return; }
    for (std::size_t s = 0; s < 2; ++s) fx.step(*opt, s);
    std::vector<nn::Tensor> state;
    NN_ASSERT_ALWAYS(opt->save_state(state), "save_state");

    // 数量少了
    auto too_few = opt->load_state(std::span<const nn::Tensor>(state.data(), state.size() - 1));
    CHECK(!too_few.has_value(), "状态张量数偏少 → 报错");
    // 形状不对（Adam 顺序 = m0(4,3), v0(4,3), m1(3,1), v1(3,1)；
    // 拿 m0 顶替 m1 的槽位 = (4,3) → (3,1) 不匹配）
    std::vector<nn::Tensor> wrong_shape = state;
    wrong_shape[2] = state[0];
    auto bad_shape = opt->load_state(wrong_shape);
    CHECK(!bad_shape.has_value(), "状态张量形状不符 → 报错");
    // sgd 无状态：给非空状态 → 报错
    auto sgd = make(fx, "sgd");
    if (sgd)
        CHECK(!sgd->load_state(state).has_value(), "无状态优化器收到状态 → 报错");
}
} // namespace

int main()
{
    std::printf("optimizer_state_test（优化器断点状态导出/载入；P1-2）\n");
    test_state_presence();
    test_resume_continuity();
    test_state_mismatch();
    if (g_fail == 0)
    {
        std::printf("[结果] 全部通过\n");
        return 0;
    }
    std::printf("[结果] %d 项失败\n", g_fail);
    return 1;
}
