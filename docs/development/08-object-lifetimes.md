# 对象生命周期契约（v3：设备域模型）

> 本页是转发页。**唯一真源 = [`09-engine-ownership.md`](09-engine-ownership.md)**
> （v3，2026-10-20，随 `v1.10.0-rc0` 发布）。
>
> v2 的"Tensor 保存 8 字节 `EngineId` + `stamp_`/`adopt`/`bound`/`NN_BIND_DEBUG`"
> 模型已删除：设备域**从存储派生**，入口逐操作数检查，`import` / `to_matrix` 是仅有的
> 跨域通道。设计与审计见 [`todo/28-lifecycle-v3.md`](../todo/28-lifecycle-v3.md)。
