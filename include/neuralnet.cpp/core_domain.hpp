#pragma once

// ── core_domain.hpp — 设备域（对象生命周期 v3，docs/internal/28-lifecycle-v3.md）──
//
// 域从**存储**派生，不保存身份：
//   · CPU 存储    → `kHostDomain`（进程级常量；所有 CPU 引擎共享宿主内存，
//                   无设备依赖，因此 CPU 存储不需要 keepalive）
//   · GPU 存储    → 其 `GpuBuffer` 的 backend 标识（backend 对象地址；buffer
//                   持有 `shared_ptr<GpuBackend>` keepalive ⇒ 张量存活期内该
//                   地址不会被新 backend 复用）
//   · 空/无效张量 → `0`（唯一豁免：不参与入口检查）
//
// `ComputeEngine` 的派生类构造时把本引擎域 token 写入基类 protected 成员
// （非虚函数，虚表口径不变）；每个 public NVI 入口逐个操作数比较
// `Tensor::device_domain_()` 与本引擎域。跨域入口只有 `import`（CPU→GPU 上载）
// 与 `to_matrix`（GPU→CPU 落宿主）。
//
// 这是 v2"出生引擎身份（8 字节 ID）+ 传播机制（stamp/adopt/bound/绑定调试开关）"
// 的替代：**减少一个概念**——张量不需要知道自己由谁创建，只需要知道自己现在
// 在哪块设备上。
// ─────────────────────────────────────────────────────────────────────────

#include <cstdint>

namespace nn
{

// 设备域标识：0 保留给"空/无效张量"（唯一豁免）。
using DomainId = std::uintptr_t;

// 宿主域：进程级常量。取 1（不是地址）——避免与任何 backend 地址碰撞；
// CPU 引擎无状态、无资源域，因此不需要"多 host backend"。
inline constexpr DomainId kHostDomain = 1;

} // namespace nn
