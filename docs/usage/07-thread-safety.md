# 线程安全契约

> **一句话结论**：本库**不提供任何跨线程并发调用保证**——服务化/多线程下游请用
> 「**每 worker 一线程一引擎**」模型，或对共享引擎做**外部串行化**。
> 本文给出每个对象的明确口径与依据，让下游不必靠猜（issue #38）。
> **读者**：要用本库跑训练/推理的调用者。总索引见 [`../README.md`](../README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
> **状态**：现行。

## 1. 总契约

| # | 规则 | 依据 |
|---|---|---|
| 1 | **同一 `ComputeEngine` 实例上的并发调用（forward/backward/read/write/begin_batch…）不安全**，必须外部同步 | 引擎持有可变槽（见 §2）；GPU 录制单命令缓冲（见 §3） |
| 2 | **同一 `Model` 实例上的并发 forward 不安全**，即使引擎不同也不行（一个 Model 只绑一个引擎，等价于规则 1） | 层在 forward 时写内部缓存（`input_cache_` / `col_cache_` 等，为 backward 保留） |
| 3 | **同一 `Tensor` 的并发读写不安全**（含"A 线程录命令、B 线程 read"——必须有 happens-before 边界） | `Tensor` 无任何内部同步 |
| 4 | **不同引擎实例之间互不干扰**（每引擎一套状态） | 引擎状态不跨实例共享（例外：全局线程池，见 §4） |
| 5 | `GpuBackend::shared()` 的**创建/回收**线程安全；拿到之后的引擎操作仍受规则 1 约束 | weak 缓存 + mutex（v1.10.0 起） |

## 2. CPU 引擎（`CpuEngine` / `CpuModel`）

- **op 内部已并行**：大张量算子经全局 `ThreadPool`（`hardware_concurrency` 线程）按块并行。
  用户线程再并发调 forward **不会加速**，反而 oversubscribe（N 用户线程 × M 池线程）。
- **并发 forward 不安全的具体根源**（不止"大概率不行"）：
  - 引擎的标量回读槽 `sync_readback_slots_`（`compute_engine.hpp`）是共享可变
    `std::vector`——并发 `submit_scalar_readback` = 数据竞争；
  - 层的 backward 缓存（如 `Linear::input_cache_`）在 forward 中被覆写——同模型
    并发 forward 会让 backward 用错缓存（静默错误，不报错）。
- **可并行的形态**：每线程一个 `CpuEngine` + 一个 `Model`（权重各自持有；
  模型大时显存/内存翻倍，见 §5 的取舍）。

## 3. GPU 引擎（`GpuEngine` / `GpuBackend`）

- **录制是单命令缓冲**：`begin_batch()` 后所有原语录进当前帧的 `batch_cmd_`
  （`compute_vk_backend_frame.inc`）——**无锁的裸录制**，并发录 = 数据竞争
  （不会稳定崩溃，会静默录坏命令流）。同一引擎必须串行。
- **已加锁的部分**（只保证内部一致性，不构成"可并发调用"的口径）：
  队列提交（`queue_mutex_`）、延迟销毁/描述符回收（`pending_mutex_`）、
  内存池（`compute_memory_pool.hpp` 的 `mutex_`）、staging 环（atomic 游标）。
- **表达式录制图是 `thread_local`**（`expr_graph.hpp`，D3 修复）：A 线程
  `begin_expr` 不影响 B 线程的录制图指针——但录进去的求值仍走同一引擎，
  规则 1 照旧。
- **可并行的形态**：
  - 每连接一线程 + 每线程一个 `GpuBackend::shared()`？——**不行**：`shared()`
    返回**同一个** backend 实例（按设备缓存）。真正的隔离需要每 worker 进程，
    或外部串行化（mutex/队列），或等待未来的多引擎多队列支持。
  - 现实建议：GPU 推理服务用**单推理线程 + 请求队列**（这正是多数下游已采用
    的形态），CPU 评估等旁路任务另开 CPU 引擎。

## 4. 全局资源（跨引擎也共享）

| 资源 | 口径 |
|---|---|
| 全局 `ThreadPool`（`extern/threadpool`，`core_threadpool.hpp` 单例） | 池本身线程安全；所有 CPU 引擎共享 → 多引擎并发 forward 会 oversubscribe，可通过环境/配置调池大小（见该头文件） |
| 精度编译期特化表 / 剖析计数器（`core_profile.hpp`） | 内部有 mutex，线程安全（只影响诊断输出，不影响计算语义） |
| AOT 融合注册表（闭合世界） | 构建期生成、运行期只读 → 安全 |

## 5. 服务化下游的设计模板

```
✅ 推荐（HTTP 服务、每连接一线程的宿主）：
   主线程 / 池：接受请求 → 无状态分发
   推理：单推理线程 + mutex 或请求队列（GPU）；
         或每 worker 线程一个 CpuEngine+Model（CPU，注意 oversubscribe）

❌ 不要：
   多个 HTTP 线程直接并发调同一 Model::forward（GPU 命令流静默录坏 /
   CPU 缓存互相覆写——两者都**不报错**，输出错但不崩）
```

**迁移提示**（对已在用全量串行化的下游）：并行化的正确粒度是**引擎实例级**
（规则 4），不是调用级。想提升 GPU 吞吐，目前的路径是多进程
（每进程一卡）而非多线程。

## 6. 版本与演进

- 本文档口径自 **v1.13.1 / v1.14.0** 时期成立（issue #38 立档）。
- 若未来引入"引擎级并发安全"承诺（如 GPU 多命令缓冲录制、CPU 引擎无状态化），
  将以 API 变化（版本第 2 位递增）+ 本文档更新的形式发布，不会静默改变。
