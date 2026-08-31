// ── core_observer_ptr.hpp — 非拥有型观察者指针 ──────────────────────────
//
// observer_ptr<T> 是对 T* 的类型安全封装，明确表达"观察但不拥有"语义。
// 等价于 C++26 std::observer_ptr<T> 的轻量替代品（无需 libc++ 支持）。
//
// 设计原则：
//   - 零开销：与裸指针相同的内存布局和性能
//   - 可空：默认构造为 nullptr，支持 if (ptr) 检查
//   - 类型自文档化：observer_ptr<T> 明确声明"不负责释放"
//   - 不提供 RAII 语义：生命周期管理由外部负责
//
// 使用场景：
//   - 类成员：持有非拥有的引擎/池引用（如 Model::engine_, GpuBuffer::pool_）
//   - 函数参数：表示可选的观察者输入
//   - 返回值：表示可能为空的查询结果
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include <cstddef>
#include <utility>

namespace nn
{

template <typename T>
class observer_ptr
{
    T* ptr_ = nullptr;

public:
    // ── 构造 ─────────────────────────────────────────────────────────────
    constexpr observer_ptr() noexcept = default;
    constexpr observer_ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}
    constexpr explicit observer_ptr(T& obj) noexcept : ptr_(&obj) {}
    constexpr explicit observer_ptr(T* ptr) noexcept : ptr_(ptr) {}

    // 从另一个 observer_ptr<U> 转换（允许隐式向上转型）
    template <typename U>
        requires std::is_convertible_v<U*, T*>
    constexpr observer_ptr(const observer_ptr<U>& other) noexcept : ptr_(other.get()) {}

    // ── 访问 ─────────────────────────────────────────────────────────────
    [[nodiscard]] constexpr T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] constexpr T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] constexpr T* get() const noexcept { return ptr_; }

    // ── 布尔转换 ─────────────────────────────────────────────────────────
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // ── 修改 ─────────────────────────────────────────────────────────────
    constexpr void reset(T* p = nullptr) noexcept { ptr_ = p; }
    constexpr void swap(observer_ptr& other) noexcept { std::swap(ptr_, other.ptr_); }

    // ── 比较 ─────────────────────────────────────────────────────────────
    friend constexpr bool operator==(const observer_ptr&, const observer_ptr&) = default;
    friend constexpr bool operator==(const observer_ptr& p, std::nullptr_t) noexcept { return !p; }
    friend constexpr auto operator<=>(const observer_ptr& a, const observer_ptr& b) noexcept
    {
        return a.ptr_ <=> b.ptr_;
    }
};

// ── 便捷工厂函数 ───────────────────────────────────────────────────────
template <typename T>
[[nodiscard]] constexpr observer_ptr<T> make_observer(T& obj) noexcept
{
    return observer_ptr<T>(obj);
}

template <typename T>
[[nodiscard]] constexpr observer_ptr<T> make_observer(T* ptr) noexcept
{
    return observer_ptr<T>(ptr);
}

} // namespace nn
