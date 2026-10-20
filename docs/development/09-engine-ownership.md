# 引擎所有权与对象生命周期（v3：设备域模型）

> 适用范围：`ComputeEngine`、`GpuBackend`、`CpuEngine` / `GpuEngine`、`Layer`、
> `Model`、`Optimizer`、`TrainingSession`、`Tensor`、`ExprGraph` / `ExprSegment`。
>
> **本文是生命周期契约的唯一真源**（v3，2026-10-20，随 `v1.10.0-rc0` 发布）。
> [`08-object-lifetimes.md`](08-object-lifetimes.md) 已降为 3 行转发。设计与审计证据见
> [`todo/28-lifecycle-v3.md`](../todo/28-lifecycle-v3.md)；Session 层见
> [`todo/29-session-architecture.md`](../todo/29-session-architecture.md)。

## 1. 一句话

- **Tensor 只拥有存储，不保存引擎身份**：设备域从存储**派生**（CPU → `kHostDomain`；
  GPU → 其 `GpuBuffer` 的 backend 标识；空/无效 → 0 唯一豁免）；
- **会解引用引擎的对象持有 `shared_ptr<ComputeEngine>`**：`Layer`、`Model`、
  `Optimizer`、`ExprGraph`、`ExprSegment`；
- **`GpuBuffer` 持有 `shared_ptr<GpuBackend>` keepalive**：GPU 张量可以晚于
  `GpuEngine` 析构；
- **每个引擎入口逐个操作数检查设备域**：跨域一律 `Result` 错误；`import`（CPU→GPU）
  与 `to_matrix`（GPU→CPU）是仅有的跨域通道；
- **`TrainingSession` 拥有 Model + Optimizer**：非拥有 `TensorRef` 的 owner 与引用者
  同生共死（推荐写法；自由工厂保留并文档化前置条件）。

## 2. 设备域（DomainId）

```cpp
using DomainId = std::uintptr_t;              // 0 保留给空/无效张量
inline constexpr DomainId kHostDomain = 1;    // 宿主域（进程级常量）
```

`Tensor` 的域从存储派生，**没有字段、没有传播机制**：

| 存储 | 域 |
|---|---|
| CPU `MatrixT<P>` | `kHostDomain`（所有 CPU 引擎共享；CPU 引擎无状态、宿主内存无设备依赖 ⇒ 不需要 keepalive） |
| GPU `GpuBuffer` | `reinterpret_cast<DomainId>(backend)`（buffer 持 backend keepalive ⇒ 张量存活期内地址不被新 backend 复用） |
| 空/无效 | `0`（唯一豁免；交由各入口既有的无效/形状检查拒绝） |

`ComputeEngine` 在派生类构造时写入本引擎域（`CpuEngine` = `kHostDomain`，`GpuEngine` =
backend 标识），`domain()` 只读。**非虚函数**：引擎虚表口径（52 方法）不变。

**为什么不是 v2 的 EngineId**：v2 让 Tensor 保存"出生引擎身份"，于是每个产出张量的
路径都要 `stamp_`、每个跨引擎入口都要比较身份，回退路径（f16 边界 cast 等）两次漏
stamp（RV-06、v1.9 f16 回退 cast）。v3 把"身份"换成"当前所在设备"——这是 Tensor
真正需要知道的信息，且从存储**必然正确**，不存在漏传播。

## 3. 入口检查与跨域

每个 public NVI 入口对**每个非空操作数**检查 `t.device_domain_() == domain_`：

| 场景 | 结果 |
|---|---|
| 同域张量 | 通过 |
| 空/无效张量 | 豁免（既有无效/形状检查拒绝） |
| CPU 张量 → GPU 引擎 op | `Result` 错误：提示先 `engine.import(t)` |
| GPU 张量 → CPU 引擎 op | `Result` 错误：提示先 `engine.to_matrix(t)` |
| GPU 张量 → 另一 backend 的引擎 | `Result` 错误：提示先 `to_matrix` 落宿主 |
| 跨 backend `import` | `Result` 错误 |

错误信息带 `source_location`、张量形状与两侧域值（可诊断，不静默）。

`import` 是**唯一跨域入口**：同域 = 零拷贝别名；同域异精度 = 引擎内 cast；
CPU → GPU = 上载；GPU → CPU 不走 import（用 `to_matrix`）；GPU 异 backend = 错误。
GPU impl 不再有 CPU 回退分支（`cast`/`copy_from`/`clone`/`copy_into`/`cast_into`）。

## 4. 各对象职责

| 对象 | 生命周期信息 | 说明 |
|---|---|---|
| `Tensor` | 无（域从存储派生） | 只拥有存储；可活过引擎、可跨 CPU 引擎使用 |
| `Layer` | `shared_ptr<ComputeEngine>` | `engine_ref()` 直接解引用；层存活则引擎存活 |
| `Model` | `shared_ptr<ComputeEngine>` | `engine()` 返回 `*engine_`；`engine_bound()` 只读查询 |
| `Optimizer` | `shared_ptr<ComputeEngine>` | **只保证引擎存活**；`params_/grads_` 是 Model 内 Tensor 的非拥有引用 ⇒ **必须早于 Model 销毁** |
| `TrainingSession` | 同上（同时持有 Model 与 Optimizer） | 成员声明顺序保证 optimizer 先析构；推荐训练入口 |
| `ExprGraph` | `shared_ptr<ComputeEngine>` | 录制图存在期间引擎不会析构 |
| `ExprSegment` | `shared_ptr<ComputeEngine>` | 录制段 RAII 守卫；作用域内释放外部句柄也安全 |
| `GpuBuffer` | `shared_ptr<GpuBackend>` | 延迟销毁 / 归还 pool 时 backend 必然存活 |

## 5. 引擎创建与 backend 生命周期

```cpp
auto cpu = nn::CpuEngine::create();                     // 私有构造，shared_ptr 管理

#ifdef NN_HAS_VULKAN
auto backend_owner = nn::GpuBackend::shared();          // 进程共享（weak 缓存）
nn::GpuBackend& backend = *backend_owner;
if (auto r = backend.initialize(); !r) { /* 无设备：SKIP */ }
auto gpu = nn::GpuEngine::create(backend_owner);
#endif
```

- 引擎必须由 `shared_ptr` 管理（`Layer::init` 内 `weak_from_this().lock()` 必须成功）；
  栈上构造 = 编译错误。
- `GpuBackend::shared()` = **weak 缓存 + mutex**：有活使用者时复用；全部释放后 backend
  与 Vulkan 资源**确定性回收**；下一次 `shared()` 重建（不自动 `initialize()`）。
- `GpuBackend::create()` = 独立实例（测试/多设备），生命周期由调用方决定。
- `GpuBuffer` 持有 backend keepalive，因此"张量晚于引擎析构"安全；backend 析构只会在
  所有 buffer 释放后发生。

## 6. 失败语义（汇总）

| 场景 | 行为 |
|---|---|
| 引擎不是 `shared_ptr` 管理 | `Layer::init` / `Model` 构造 / `Optimizer` 构造 fail-fast |
| 层已绑定引擎 A，再 `init(B)` | `Result` 错误，保持 A |
| 跨域操作数 | `Result` 错误（域不匹配，带调用点/形状/域值） |
| 张量的引擎已析构 | Tensor 仅持有存储 ⇒ 复制/读元数据/析构安全；不能参与原引擎 op |
| `GpuEngine` 析构但 GPU Tensor 仍存活 | `GpuBuffer` 的 backend keepalive 保证 backend 存活 |
| Model 先于 Optimizer 析构 | **UAF（契约违约）**：用 `TrainingSession` 或保证销毁顺序 |

## 7. 性能口径（v3 实测）

| 项 | v2（v1.9.0） | v3（v1.10.0-rc0） |
|---|---|---|
| `sizeof(Tensor)` CPU / GPU | 64 B / 88 B | **56 B / 80 B** |
| Tensor 拷贝 | 一次 8 B 身份复制 + 存储原子对 | 少一次 8 B 复制（原子行为不变） |
| 引擎身份比较 | 8 B EngineId | 一次存储域读取（无原子操作） |
| 机制点 | 101（ID/stamp/adopt/bind/调试开关） | **42**（只留入口域检查调用点） |

## 8. 迁移检查清单（v1.9 → v1.10）

1. 删除 `t.engine_id()` / `t.bound()` 的使用：不要从张量反查引擎；需要"能否在本引擎用"
   就看操作返回的 `Result`。
2. `GpuBackend::instance()` → `auto owner = GpuBackend::shared(); GpuBackend& b = *owner;`
   （owner 必须活到使用者结束）。
3. GPU 引擎不再隐式接收 CPU 张量：显式 `auto t_gpu = NN_UNWRAP(gpu.import(t_cpu));`。
4. 结果回宿主：`NN_UNWRAP(gpu.to_matrix(t_gpu))`（`to_matrix` 是 GPU→CPU 的正常出口）。
5. 训练入口优先用 `TrainingSession`（`Model` + `Optimizer` 同生共死）；继续用
   `create_optimizer` 时保证 optimizer 早于参数所有者销毁。文本/GPT 训练再优先用
   `nn::TextTrainer`（`text_trainer.hpp`：**数据集 + 训练循环**，内部持有
   `TrainingSession`，并由它装配/持有 `TrainState`）。
6. 不再需要 `NN_BIND_DEBUG`：域检查是默认行为。

## 9. 回归

`src/layer_lifetime_test.cpp`（域模型 / 跨域拒绝 / import 唯一入口 / backend weak 缓存 /
强引用 / init 回滚）、`src/lifetime_contract_test.cpp`（ExprSegment 强引用 / offload 状态机
与精度门 / optimizer 先于 Model）、`src/training_session_test.cpp`（Session 所有权/移动/
精度透传/错误路径）。ASan 下三者在 Windows clang 22 全绿（LSan 不支持）。
