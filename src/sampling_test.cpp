// ── sampling_test — 采样器（P0-5：top-k / top-p / 重复惩罚 / 种子）──────────
// 覆盖两段：
//   A. 采样器单测（直接喂 logits，不需要模型；确定性、无浮点随机性）
//      · **默认配置 ≡ 旧 temperature 路径**：返回下标逐位一致 +
//        RNG 消耗序一致（mt19937_64 状态相等）——采样器重构的回归锁
//      · top-k：只在（值降序, 下标升序）的前 k 个里选；k=1 = argmax
//      · top-p：核采样只留"累计概率 ≥ 阈值"的头部；阈值→0 至少留 1 个
//      · 重复惩罚：打压 history 中的去重 token（能改变贪心结果）；
//        repeat_penalty=1 无操作；repeat_window 只看最近 N 个
//      · 贪心（temperature<=0）不消耗 RNG
//   B. 模型级（GPTModel::generate）
//      · **同 seed + 同 prompt ⇒ 输出逐 token 相同**（P0-5 验收）
//      · 不同 seed 在高温度下确实给出不同序列（种子真的接上了）
//      · top_k=1 ≡ temperature=0 贪心（截断到 1 个候选 = argmax）
//
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <limits>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <neuralnet.cpp/nn.hpp>

#include "test_common.hpp"

using nn::SamplingConfig;
using nn::Scalar;

int g_fail = 0;

namespace
{

std::mt19937_64 make_rng()
{ return std::mt19937_64{0xC0FFEEULL}; }

// 旧签名（无截断）与新签名默认配置的等价性：同 seed 下返回同一下标、
// 概率向量逐位相同、RNG 状态相同
void test_default_equals_legacy()
{
    const std::vector<std::vector<Scalar>> cases = {
        {0.0, 0.0, 0.0, 0.0},                 // 完全均匀
        {1.0, -1.0, 0.5, 3.0, -2.5, 0.0},     // 一般形状
        {5.0, 5.0, 1.0, 1.0, 1.0},            // 并列
        {-10.0, -20.0, -30.0},                // 全负
        {100.0, 0.0, -100.0},                 // 极端尺度
    };
    const std::vector<Scalar> temps = {1.0, 0.7, 0.0, 2.5};
    bool ok = true;
    for (const auto& base : cases)
    {
        for (const Scalar t : temps)
        {
            auto rng_a = make_rng();
            auto rng_b = make_rng();
            std::uniform_real_distribution<Scalar> d1(0.0, 1.0), d2(0.0, 1.0);
            std::vector<Scalar> la = base, lb = base;
            const std::size_t ia = nn::sample_next_token_(la, t, rng_a, d1);
            SamplingConfig cfg;
            cfg.temperature = t;
            const std::size_t ib = nn::sample_next_token_(lb, cfg, rng_b, d2);
            if (ia != ib || rng_a != rng_b || la != lb) ok = false;
        }
    }
    CHECK(ok, "默认配置 ≡ 旧 temperature 路径（下标 / 概率 / RNG 状态逐位一致）");
}

// 多次采样同一 logits：返回出现过的下标集合
std::vector<std::size_t> draw_support(const std::vector<Scalar>& logits, const SamplingConfig& cfg,
                                      std::size_t draws)
{
    std::vector<std::size_t> seen;
    auto rng = make_rng();
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    for (std::size_t i = 0; i < draws; ++i)
    {
        std::vector<Scalar> l = logits;
        const std::size_t t = nn::sample_next_token_(l, cfg, rng, dist, {});
        bool found = false;
        for (const std::size_t s : seen)
            if (s == t) found = true;
        if (!found) seen.push_back(t);
    }
    return seen;
}

bool in_set(const std::vector<std::size_t>& set, std::size_t v)
{
    for (const std::size_t s : set)
        if (s == v) return true;
    return false;
}

void test_top_k()
{
    const std::vector<Scalar> logits = {1.0, 4.0, 2.0, 3.0};
    SamplingConfig cfg;
    cfg.top_k = 2;   // 期望候选 = {下标 1(4.0), 下标 3(3.0)}
    std::mt19937_64 rng = make_rng();
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    bool only_top2 = true;
    for (std::size_t i = 0; i < 500; ++i)
    {
        std::vector<Scalar> l = logits;
        const std::size_t t = nn::sample_next_token_(l, cfg, rng, dist, {});
        if (t != 1 && t != 3) only_top2 = false;
    }
    CHECK(only_top2, "top_k=2 只在前 2 个候选里采样");

    SamplingConfig k1;
    k1.top_k = 1;   // 单候选 ⇒ 恒 argmax（= 4.0 的下标 1）
    bool k1_ok = true;
    for (std::size_t i = 0; i < 50; ++i)
    {
        std::vector<Scalar> l = logits;
        if (nn::sample_next_token_(l, k1, rng, dist, {}) != 1) k1_ok = false;
    }
    CHECK(k1_ok, "top_k=1 ⇒ 恒取 argmax 下标");

    // 并列：值相同时取小下标（确定性）
    const std::vector<Scalar> tie = {2.0, 2.0, 2.0};
    k1.top_k = 2;
    bool tie_ok = true;
    for (std::size_t i = 0; i < 50; ++i)
    {
        std::vector<Scalar> l = tie;
        const std::size_t t = nn::sample_next_token_(l, k1, rng, dist, {});
        if (t > 1) tie_ok = false;
    }
    CHECK(tie_ok, "top_k 并列时按小下标优先（确定性）");
}

void test_top_p()
{
    // 近 one-hot：{10,0,0,0} 的 softmax ≈ {0.99995, 1.7e-5, ...}
    const std::vector<Scalar> peaked = {10.0, 0.0, 0.0, 0.0};
    SamplingConfig cfg;
    cfg.top_p = 0.5;
    bool p_ok = true;
    auto seen = draw_support(peaked, cfg, 300);
    for (const std::size_t s : seen)
        if (s != 0) p_ok = false;
    CHECK(p_ok, "top_p=0.5 在近 one-hot 上只留峰（核采样）");

    // 阈值极小 ⇒ 至少留 1 个（不会空集）
    SamplingConfig tiny;
    tiny.top_p = 1e-9;
    auto seen_tiny = draw_support(peaked, tiny, 50);
    CHECK(!seen_tiny.empty() && in_set(seen_tiny, 0), "top_p→0 仍保留至少 1 个候选（峰自身）");

    // 均匀分布 + top_p=0.5：4 个等概率 ⇒ 留 2 个
    SamplingConfig half;
    half.top_p = 0.5;
    auto seen_flat = draw_support({1.0, 1.0, 1.0, 1.0}, half, 400);
    CHECK(seen_flat.size() <= 2, "top_p=0.5 在均匀分布上只留 2 个候选");
}

void test_repeat_penalty()
{
    auto rng = make_rng();
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    const std::vector<std::size_t> hist0 = {0};
    const std::vector<std::size_t> hist10 = {1, 0};   // 最近一个是 0

    // 惩罚 = 关：贪心取 0（logit 5.0 > 4.0）
    SamplingConfig off;
    off.temperature = 0.0;
    std::vector<Scalar> l = {5.0, 4.0};
    CHECK(nn::sample_next_token_(l, off, rng, dist, hist0) == 0,
          "repeat_penalty=1 ⇒ 无操作（贪心取原 argmax）");

    // 惩罚 10×：5.0/10 = 0.5 < 4.0 ⇒ 贪心改取 1
    SamplingConfig on;
    on.temperature = 0.0;
    on.repeat_penalty = 10.0;
    l = {5.0, 4.0};
    CHECK(nn::sample_next_token_(l, on, rng, dist, hist0) == 1,
          "repeat_penalty 打压 history 中的 token（改变 argmax）");

    // 窗口：history = {1, 0}，窗口 1 → 只罚 0 → 0.5 vs 4.0 ⇒ 取 1；
    // 窗口 0（全部）→ 两个都被罚 ⇒ 0.5 vs 0.4 ⇒ 取 0
    on.repeat_window = 1;
    l = {5.0, 4.0};
    const std::size_t win1 = nn::sample_next_token_(l, on, rng, dist, hist10);
    on.repeat_window = 0;
    l = {5.0, 4.0};
    const std::size_t win0 = nn::sample_next_token_(l, on, rng, dist, hist10);
    CHECK(win1 == 1 && win0 == 0, "repeat_window=1 只看最近一个 token；0 = 全部上下文");

    // 负 logit 走乘法分支（HF 语义）：{-1, -5} + 罚下标 1 ⇒ {-1, -50} ⇒ 取 0；
    // 若误用"除法"则为 {-1, -0.5} ⇒ 取 1（本断言语义专属，能区分两者）。
    // 注意：采样器就地覆写 logits 为概率，故这里只能从"选出的下标"反推。
    SamplingConfig neg;
    neg.temperature = 0.0;
    neg.repeat_penalty = 10.0;
    l = {-1.0, -5.0};
    // ⚠ 不能写 `{1}` 直接当 span 实参：libstdc++ 接受，libc++（clang 24）报
    // "no matching function"（span 无 initializer_list 构造）——用显式数组。
    const std::size_t hist_neg[1] = {1};
    const std::size_t nt = nn::sample_next_token_(l, neg, rng, dist, hist_neg);
    CHECK(nt == 0, "重复惩罚对负 logit 用乘法（-5 → -50 ⇒ 未被罚的 -1 胜出）");
}

// ── logits_transform 钩子（issue #34）────────────────────────────────────
void test_logits_transform()
{
    const std::vector<Scalar> logits = {1.0, 4.0, 2.0, 3.0};
    std::mt19937_64 rng = make_rng();
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);

    // [a] 钩子把非法位置置 -inf：约束解码语义（棋类合法着 / 语法约束）
    {
        SamplingConfig cfg;
        cfg.temperature = 0.0;   // 贪心：结果由掩码后的 argmax 唯一决定
        cfg.logits_transform = [](std::span<Scalar> l)
        {
            l[1] = -std::numeric_limits<Scalar>::infinity();   // 屏蔽原 argmax(4.0)
        };
        std::vector<Scalar> l = logits;
        const std::size_t t = nn::sample_next_token_(l, cfg, rng, dist, {});
        CHECK(t == 3, "钩子置 -inf 的位置不被贪心选中（次大 3.0 的下标 3 上位）");
    }

    // [b] 钩子加偏置：改变贪心结果
    {
        SamplingConfig cfg;
        cfg.temperature = 0.0;
        cfg.logits_transform = [](std::span<Scalar> l) { l[0] += 10.0f; };
        std::vector<Scalar> l = logits;
        CHECK(nn::sample_next_token_(l, cfg, rng, dist, {}) == 0,
              "钩子加偏置可把任意位置推为贪心选择");
    }

    // [c] 未设置钩子：与旧行为逐位一致（RNG 消耗序不被钩子机制扰动）
    {
        SamplingConfig cfg;   // 全默认（含无钩子）
        std::mt19937_64 r1 = make_rng(), r2 = make_rng();
        std::uniform_real_distribution<Scalar> d1(0.0, 1.0), d2(0.0, 1.0);
        std::vector<Scalar> l1 = logits, l2 = logits;
        const std::size_t t1 = nn::sample_next_token_(l1, cfg, r1, d1, {});
        const std::size_t t2 = nn::sample_next_token_(l2, cfg, r2, d2, {});
        CHECK(t1 == t2, "无钩子时输出与旧路径一致");
        CHECK(r1 == r2, "无钩子时 RNG 状态逐位一致");
    }

    // [d] 采样模式下 -inf 位置概率为 0（softmax 自然兼容，永不选中）
    {
        SamplingConfig cfg;
        cfg.temperature = 1.0;
        cfg.has_seed = true;
        cfg.seed = 12345;
        cfg.logits_transform = [](std::span<Scalar> l)
        {
            l[0] = -std::numeric_limits<Scalar>::infinity();
            l[2] = -std::numeric_limits<Scalar>::infinity();
        };
        std::mt19937_64 r{777};
        std::uniform_real_distribution<Scalar> dd(0.0, 1.0);
        bool never_masked = true;
        for (int i = 0; i < 300; ++i)
        {
            std::vector<Scalar> l = logits;
            const std::size_t t = nn::sample_next_token_(l, cfg, r, dd, {});
            if (t == 0 || t == 2) never_masked = false;   // 被掩码的两个位置
        }
        CHECK(never_masked, "采样模式下 -inf 位置 300 次全部不被选中");
    }
}

void test_greedy_consumes_no_rng()
{
    SamplingConfig greedy;
    greedy.temperature = 0.0;
    greedy.top_k = 3;
    auto rng = make_rng();
    const auto before = rng;
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    std::vector<Scalar> l = {1.0, 5.0, 2.0, 3.0};
    (void)nn::sample_next_token_(l, greedy, rng, dist, {});
    CHECK(rng == before, "贪心路径不消耗 RNG（与旧实现一致）");
}

void test_gpt_model_generate()
{
    auto eng_owner = nn::CpuEngine::create();
    nn::CpuEngine& eng = *eng_owner;
    const std::size_t vocab = 32, d_model = 16, seq = 8, heads = 2, d_ff = 32;
    nn::GPTModel model(vocab, d_model, seq, heads, d_ff, /*layers=*/1, nn::PosEncodingType::Learned,
                       nn::ActivationType::SwiGLU, nn::NormType::RMSNorm);
    NN_EXIT(model.init(eng), 1, "GPTModel init 失败: ");

    const std::vector<std::size_t> prompt = {1, 2, 3};
    auto run = [&](const SamplingConfig& cfg)
    {
        auto r = model.generate(eng, prompt, /*max_new_tokens=*/5, cfg);
        NN_EXIT(r, 1, "GPTModel::generate 失败: ");
        return *r;
    };

    // ① 同 seed 可复现
    SamplingConfig seeded;
    seeded.temperature = 1.0;
    seeded.has_seed = true;
    seeded.seed = 12345;
    const auto a = run(seeded);
    const auto b = run(seeded);
    CHECK(a == b && !a.empty(), "同 seed + 同 prompt ⇒ 输出逐 token 相同");

    // ② 种子真的接上了（高温下不同 seed 至少给出两种不同序列）
    SamplingConfig hot = seeded;
    hot.temperature = 5.0;
    std::vector<std::vector<std::size_t>> outs;
    for (std::uint64_t s : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}, std::uint64_t{4},
                            std::uint64_t{5}, std::uint64_t{6}})
    {
        hot.seed = s;
        outs.push_back(run(hot));
    }
    std::size_t distinct = 0;
    for (std::size_t i = 0; i < outs.size(); ++i)
    {
        bool dup = false;
        for (std::size_t j = 0; j < i; ++j)
            if (outs[i] == outs[j]) dup = true;
        if (!dup) ++distinct;
    }
    CHECK(distinct >= 2, "不同 seed 在高温下给出不同序列（种子生效）");

    // ③ top_k=1 ≡ temperature=0 贪心
    SamplingConfig greedy;
    greedy.temperature = 0.0;
    SamplingConfig top1;
    top1.temperature = 1.0;
    top1.top_k = 1;
    CHECK(run(greedy) == run(top1), "top_k=1 ≡ temperature=0 贪心（逐 token 相同）");
}

} // namespace

int main()
{
    std::puts("=== sampling_test: 采样器（top-k / top-p / 重复惩罚 / 种子）===");
    test_default_equals_legacy();
    test_top_k();
    test_top_p();
    test_repeat_penalty();
    test_logits_transform();
    test_greedy_consumes_no_rng();
    test_gpt_model_generate();
    std::printf("\nsampling_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
