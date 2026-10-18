// ── core_engine_lease.hpp — 兼容转发头（旧 EngineLease 已删除）──────────
//
// v2 生命周期模型不再使用存活令牌 / 借用句柄：
//   · Tensor 只保存 8 字节 EngineId；
//   · Layer / Model / Optimizer / ExprGraph 持有 shared_ptr<ComputeEngine>。
//
// 保留本头是为了让仍 include 旧路径的下游代码得到明确的迁移入口；新代码
// 应直接 include "core_engine_id.hpp"。
// ─────────────────────────────────────────────────────────────────────────

#pragma once

#include "core_engine_id.hpp"
