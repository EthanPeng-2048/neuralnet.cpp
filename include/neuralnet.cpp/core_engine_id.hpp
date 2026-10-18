// ── core_engine_id.hpp — 引擎身份（Tensor 只保存的 8 字节值）─────────────
//
// Tensor 需要知道"自己出生在哪个引擎"，用于跨引擎混用的诊断；但它**不应**
// 保存任何引擎指针，否则 Tensor 晚于引擎析构时就会形成悬垂。
//
// EngineId 是进程内单调递增、永不复用的 64 位整数：
//   · 0 = 未绑定；
//   · 复制 = 一次整数复制（平凡可复制、零原子操作）；
//   · 旧 Tensor 的 ID 不会与新引擎相同，因此不会把新引擎误认成旧引擎。
//
// 生命周期保证不靠这个类型：真正会解引用引擎的 Layer / Model / Optimizer /
// ExprGraph 持有 std::shared_ptr<ComputeEngine>。详见
// docs/development/09-engine-ownership.md。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace nn
{

class EngineId
{
public:
    constexpr EngineId() noexcept = default;
    explicit constexpr EngineId(std::uint64_t value) noexcept : value_(value) {}

    [[nodiscard]] constexpr bool bound() const noexcept { return value_ != 0; }
    [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

    friend constexpr bool operator==(EngineId a, EngineId b) noexcept
    {
        return a.value_ == b.value_;
    }
    friend constexpr bool operator!=(EngineId a, EngineId b) noexcept
    {
        return !(a == b);
    }

private:
    std::uint64_t value_ = 0;
};

namespace detail
{

// 进程级单调计数器：从 1 开始，0 保留给"未绑定"。
// 64 位回绕在现实中不可达；真发生时直接终止，绝不把 0 当作有效身份。
[[nodiscard]] inline EngineId next_engine_id() noexcept
{
    static std::atomic<std::uint64_t> next{1};
    const std::uint64_t value =
        next.fetch_add(1, std::memory_order_relaxed);
    if (value == 0)
        std::abort();
    return EngineId(value);
}

} // namespace detail

static_assert(sizeof(EngineId) == sizeof(std::uint64_t),
              "EngineId 必须保持 8B（Tensor 逐次拷贝都会复制它）");
static_assert(std::is_trivially_copyable_v<EngineId>,
              "EngineId 必须平凡可复制（Tensor 拷贝零原子操作）");

} // namespace nn
