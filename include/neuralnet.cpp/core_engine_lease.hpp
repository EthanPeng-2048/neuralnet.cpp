// ── core_engine_lease.hpp — 引擎非拥有借用 + 存活令牌 ─────────────────────
//
// 问题：Layer / Model / Tensor 都以非拥有方式借用 ComputeEngine。裸指针或
// observer_ptr 只能表达"不拥有"，无法表达"借用是否仍然有效"——引擎先析构、
// 借用方后使用时，`engine_ref()` 的空指针检查会放行一个悬垂指针，随后任何
// 虚调用都是 UB（实测：pure virtual call 终止 / 挂死 / 静默错值都可能）。
//
// 本头给出两个类型：
//   · EngineLiveness —— 与引擎对象**分离分配**的存活令牌。引擎析构时把
//     alive 置 false；令牌本身仍可安全查询，因此"引擎已销毁"是可判定的。
//   · EngineLease    —— 8 字节、可平凡拷贝的借用句柄。拷贝零原子操作；
//     每次 require() 只做一次 acquire load。
//
// 生命周期契约（由类型强制，不靠注释约定）：
//   1. 借用方只能经 EngineLease 访问引擎；`require()` 在不安全时终止并打印
//      借用方名字 + 调用点；
//   2. EngineLease 不延长引擎寿命（非拥有）；
//   3. 令牌地址**永不复用**（见下方令牌池）⇒ 陈旧 lease 永远读到 false。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>

#include "core_assert.hpp"
#include "core_errors.hpp"

namespace nn
{

class ComputeEngine;

// ══════════════════════════════════════════════════════════════════════════
// EngineLiveness —— 一次引擎实例的存活令牌
// 与 ComputeEngine **分开分配**：引擎析构只把 alive 置 false，令牌内存仍有效。
// ══════════════════════════════════════════════════════════════════════════
class EngineLiveness
{
public:
    explicit EngineLiveness(ComputeEngine* engine) noexcept : engine_(engine) {}

    EngineLiveness(const EngineLiveness&) = delete;
    EngineLiveness& operator=(const EngineLiveness&) = delete;

    [[nodiscard]] bool alive() const noexcept
    {
        return alive_.load(std::memory_order_acquire);
    }
    [[nodiscard]] ComputeEngine* engine() const noexcept { return engine_; }

private:
    friend class ComputeEngine;
    void retire() noexcept { alive_.store(false, std::memory_order_release); }

    ComputeEngine* engine_ = nullptr;
    std::atomic<bool> alive_{true};
};

namespace detail
{

// ── 进程级令牌池：只增不减 ────────────────────────────────────────────────
// 为什么不用 shared_ptr/weak_ptr 或"回收复用"：
//   · EngineLease 必须是一个指针大小、拷贝零原子操作 —— Tensor 每次拷贝都会
//     复制它（reshape / 表达式 leaf / 图节点输出都在热路径上）；
//   · 令牌地址**永不复用**是正确性要求：复用会让一个陈旧 lease 把新引擎误认
//     成旧引擎（alive 又是 true）——那是比悬垂更危险的静默错值。
// 代价 = 每个引擎一个 ~16B 的块，进程退出时随地址空间回收（与
// GpuBackend::instance() 的故意泄漏同源）。引擎是重量级对象，一个进程通常
// 只有 1 个；测试里按引擎数增长，量级仍是 KB。
class EngineTokenPool
{
public:
    [[nodiscard]] EngineLiveness* create(ComputeEngine* engine)
    {
        std::lock_guard lock(mu_);
        blocks_.emplace_back(engine);
        return &blocks_.back();
    }

    [[nodiscard]] std::size_t created() const
    {
        std::lock_guard lock(mu_);
        return blocks_.size();
    }

private:
    mutable std::mutex mu_;
    // deque：push_back 不移动已有元素 ⇒ 已发出的 EngineLiveness* 恒稳定。
    std::deque<EngineLiveness> blocks_;
};

[[nodiscard]] inline EngineTokenPool& engine_token_pool() noexcept
{
    // 故意泄漏：避免静态析构顺序——全局 Tensor / 引擎的析构可能晚于本池。
    static EngineTokenPool* pool = new EngineTokenPool();
    return *pool;
}

} // namespace detail

// ══════════════════════════════════════════════════════════════════════════
// EngineLease —— 可失效的引擎借用句柄（8B，平凡拷贝）
// ══════════════════════════════════════════════════════════════════════════
class EngineLease
{
public:
    EngineLease() noexcept = default;
    explicit EngineLease(EngineLiveness* live) noexcept : live_(live) {}

    // 是否曾经绑定过（出生绑定仍存在）
    [[nodiscard]] bool bound() const noexcept { return live_ != nullptr; }
    // 绑定且引擎仍存活
    [[nodiscard]] bool alive() const noexcept
    {
        return live_ != nullptr && live_->alive();
    }
    // 绑定过但引擎已销毁
    [[nodiscard]] bool expired() const noexcept
    {
        return live_ != nullptr && !live_->alive();
    }

    // 引擎存活时返回指针；未绑定 / 已销毁 → nullptr（永不返回悬垂指针）
    [[nodiscard]] ComputeEngine* get() const noexcept
    {
        return alive() ? live_->engine() : nullptr;
    }

    // 身份 = 令牌地址（每个引擎唯一且永不复用）。用于"是否同一引擎"的比较：
    // 引擎销毁后，两个同源 Tensor 仍应判为同源；不同引擎永不相等。
    [[nodiscard]] const EngineLiveness* identity() const noexcept { return live_; }

    // 访问咽喉：不安全时终止，并打印借用方与调用点。
    [[nodiscard]] ComputeEngine& require(
        std::string_view owner,
        std::source_location loc = std::source_location::current()) const
    {
        if (live_ == nullptr)
        {
            const std::string msg = describe(owner, "未绑定引擎（先调 init(engine)）");
            detail::fatal_at("NN_FATAL", loc.file_name(),
                             static_cast<int>(loc.line()), msg.c_str());
        }
        if (!live_->alive())
        {
            const std::string msg = describe(
                owner, "绑定的引擎已销毁——engine 必须比借用方活得久");
            detail::fatal_at("NN_FATAL", loc.file_name(),
                             static_cast<int>(loc.line()), msg.c_str());
        }
        return *live_->engine();
    }

    void reset() noexcept { live_ = nullptr; }

private:
    [[nodiscard]] static std::string describe(std::string_view owner,
                                              std::string_view why)
    {
        std::string msg = "[engine-lease] ";
        if (!owner.empty())
        {
            msg += owner;
            msg += ": ";
        }
        msg += why;
        return msg;
    }

    EngineLiveness* live_ = nullptr;
};

// Tensor 逐次拷贝都会复制它：必须保持一个指针大小、平凡可拷贝。
static_assert(sizeof(EngineLease) == sizeof(void*),
              "EngineLease 必须保持 8B（Tensor 热路径）");
static_assert(std::is_trivially_copyable_v<EngineLease>,
              "EngineLease 必须平凡可拷贝（拷贝零原子操作）");

} // namespace nn
