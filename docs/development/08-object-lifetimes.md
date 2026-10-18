# 对象生命周期契约（v2）

> 本文记录 2026-10-18 之后生效的所有权模型。旧版“EngineLease 存活令牌”
> 已删除；完整设计、改动范围与性能口径见
> [`09-engine-ownership.md`](09-engine-ownership.md)。

## 1. 一句话

- **Tensor 只保存 8 字节 `EngineId`**：无指针、无控制块、拷贝零原子操作；
- **会解引用引擎的对象持有 `shared_ptr<ComputeEngine>`**：`Layer`、`Model`、
  `Optimizer`、`ExprGraph` / `ExprBlock`；
- **`GpuBuffer` 持有 `shared_ptr<GpuBackend>` keepalive**：GPU 张量可以晚于
  `GpuEngine` 析构；
- 引擎与 backend 都由 `shared_ptr` 管理，最后一个强引用释放时按 RAII 析构。

因此不再需要“引擎必须比借用方活得久”的调用约定，也不再需要进程级令牌池。

## 2. 引擎创建

```cpp
auto cpu = nn::CpuEngine::create();          // shared_ptr<CpuEngine>

#ifdef NN_HAS_VULKAN
auto backend = nn::GpuBackend::shared();     // 进程共享；RAII 析构
auto gpu = nn::GpuEngine::create(backend);
#endif
```

`CpuEngine` / `GpuEngine` 的构造函数私有，栈上构造是编译错误。这保证
`Layer::init(engine)` 里面的 `engine.weak_from_this().lock()` 一定能成功。

## 3. 各对象职责

| 对象 | 持有的生命周期信息 | 说明 |
|---|---|---|
| `Tensor` | `EngineId`（8 B） | 只用于跨引擎诊断；不提供 `engine()` / `engine_alive()` |
| `Layer` | `shared_ptr<ComputeEngine>` | `engine_ref()` 直接解引用；层存活则引擎存活 |
| `Model` | `shared_ptr<ComputeEngine>` | `Model::engine()` 返回 `*engine_` |
| `Optimizer` | `shared_ptr<ComputeEngine>` | 优化器晚于 Model 存活也安全 |
| `ExprGraph` | `shared_ptr<ComputeEngine>` | 录制图存在期间引擎不会析构 |
| `ExprBlock` | `shared_ptr<ComputeEngine>` | `end_expr` 直接使用强引用 |
| `GpuBuffer` | `shared_ptr<GpuBackend>` | 延迟销毁 / 归还 pool 时 backend 必然存活 |

## 4. 失败语义

| 场景 | 行为 |
|---|---|
| 用未绑定的 `EngineId` 做跨引擎检查 | 未绑定输入按库外豁免放行（`NN_BIND_DEBUG=1` 时报错） |
| Tensor 由引擎 A 创建，交给引擎 B | `Result` 错误：`mixed engines` |
| Tensor 的出生引擎已析构 | Tensor 仍可复制/读元数据/析构；不会解引用旧引擎 |
| `Layer` 已绑定 A，再 `init(B)` | `Result` 错误，保持 A |
| 引擎不是 `shared_ptr` 管理 | `Layer::init` / `Model` / `Optimizer` 返回构造期错误 |
| `GpuEngine` 析构但 GPU Tensor 仍存活 | `GpuBuffer` 的 backend keepalive 保证 backend 存活 |

## 5. 迁移检查清单

1. 不要写 `nn::CpuEngine engine;`；改用 `auto engine = nn::CpuEngine::create();`
   并传 `*engine` 给 `Layer::init` / `Model` / `create_optimizer`；
2. 不要保存 `ComputeEngine*` 作为长期成员；成员类型用
   `std::shared_ptr<ComputeEngine>`；
3. 不要从 Tensor 取引擎；把引擎作为显式参数传入；
4. 判等用 `Tensor::engine_id()`，不要比较地址；
5. GPU buffer/backend 的析构顺序由 `GpuBuffer` keepalive 保证，不要再调用
   `GpuBackend::instance()` 来“续命”一个可能已销毁的单例。

## 6. 回归

`src/layer_lifetime_test.cpp` 覆盖：

1. `EngineId` 8 B / 平凡复制 / 不复用；
2. Tensor 晚于引擎析构仍可复制、可析构；
3. Layer / Model 在外部引擎句柄释放后仍可 forward；
4. Optimizer 在外部引擎句柄释放后仍可 step；
5. 跨引擎重绑返回 `Result` 错误；
6. init 失败回滚与 profile 注入；
7. `NN_GPU_LIFETIME_FULL=1` 下验证 GpuBuffer 晚于 GpuEngine/backend 析构。
