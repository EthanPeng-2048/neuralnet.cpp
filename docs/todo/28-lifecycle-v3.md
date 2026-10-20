# 28 — 生命周期 v3：设备域模型与所有权收敛

> **读者**：项目开发者（本仓内部工作文档）。分册定位见 [`README.md`](README.md)。
> **状态**：✅ **已实施（2026-10-20，分支 `dev/lifetime_refresh`，发布 `v1.10.0-rc0`）**。
> P0（安全修复）/ P1（v3 core）/ P3（backend 弱缓存）已完成并全绿；Session 层 P2a 见
> [`29`](29-session-architecture.md)（P2b 未做）；P4 文档收尾随本分支。决策记录见 §3；Session 层（原第 3 项）拆到
> [`29-session-architecture.md`](29-session-architecture.md)。
> **替代关系**：本文取代 [`../development/08-object-lifetimes.md`](../development/08-object-lifetimes.md)
> 与 [`../development/09-engine-ownership.md`](../development/09-engine-ownership.md) 里"Tensor 保存
> EngineId provenance"的部分；实施完成后 08/09 按本文重写（见 §5 P4）。
> 执行排序见 [`27-execution-plan.md`](27-execution-plan.md)；裁定台账见 [`18-roadmap.md`](18-roadmap.md) §4.2 P1-8。

## 1. 一句话

**删掉 Tensor 的"出生引擎身份"（EngineId）与全套传播机制（`stamp_`/`adopt`/`bound`/unbound 三态/
`NN_BIND_DEBUG`），把引擎兼容性改为"张量实际所在的设备域、域从存储派生"的统一模型**；同时修复本轮审计
确认的 owner 侧、offload 状态机与 GPU backend 生命周期缺口。

这不是"再换一种句柄"，而是**减少一个概念**：张量不需要知道自己由谁创建，只需要知道自己现在在哪块设备上。

## 2. 立项依据（现状审计，2026-10-20）

审计方法：全仓 grep + 逐文件走查 + 本机 sizeof/copy 探针（clang++ -O3，Windows）。债务编号 D1–D9：

| # | 债 | 严重度 | 证据 / 触发条件 |
|---|---|---|---|
| D1 | **绑定检查没有比较 `this`**：`bind_check_` 只让操作数互相比较（首个 bound 作基准），从不验证操作数属于当前引擎 ⇒ 单操作数入口可让外来 GPU 张量进入任意引擎；`GpuEngine::import_impl` 对 `src.is_gpu()` 直接返回别名，跨 backend 传张量是 UB | 高 | `compute_engine.hpp` 的 `bind_check_one_`；`compute_gpu_engine.hpp` 的 `import_impl`；全仓无 mixed-engine 拒绝测试 |
| D2 | **provenance 传播机器是持续维护税**：core_engine_id.hpp 83 行；`stamp_` 35 处；`adopt` 18 处；`bind_check_` 42 处；`NN_BIND_DEBUG` 3 处 + CI 第二轮 ctest。新增回退路径必须记得 stamp（历史上 RV-06、v1.9 f16 回退 cast 两次补漏） | 高 | 计数见 `bench/doc_inventory.ps1` 之外的 grep 口径；`docs/history/history.md` 对应条目 |
| D3 | **Optimizer 的 owner 契约是假的**：`params_/grads_` 是非拥有 `TensorRef` 指向 Model/Layer 成员 Tensor；Model 先析构 ⇒ `step()` UAF；两份文档却写"优化器晚于 Model 存活也安全" | 高 | `compute_optimizer.hpp`；`09-engine-ownership.md`、`08-object-lifetimes.md`；`layer_lifetime_test` 未覆盖 |
| D4 | **activation offload 状态机缺口**：`export_activations` 在 `offloaded_` 为真时静默 `return {}`，只有 import 清标志 ⇒ 两次 forward 之间可能静默用上一步 slab；slab 以 f32 记账，f16/t1_58 + offload 无配置期拒绝、无测试 | 中高 | `compute_layer_base.hpp` 的 `ActivationOffloader`；`cli/text_train_config.hpp` 未校验组合 |
| D5 | **GPU 后端生命周期不具确定性**：`GpuBackend::shared()` 的函数局部 static 强引用使 Vulkan 资源活到进程退出；`instance()` 叠加第二层 static，16 个调用点 | 中 | `backend/compute_vk_backend.hpp`；`09-engine-ownership.md` 声称 RAII 即时回收 |
| D6 | **录制段生命周期**：`ExprSegment` 持裸 `ComputeEngine&`（作用域内丢引擎句柄 ⇒ 析构 UAF）；跨线程 `end_expr` 静默 no-op | 中 | `compute_engine.hpp` 的 `ExprSegment`；`compute_gpu_engine.hpp` 的 `end_expr` |
| D7 | **API 易用性**：~130 处 `create()` 后立刻退化成 `Engine&`（`raw_engine.get()` 别名 idiom）；工厂/`create_optimizer` 只收 `ComputeEngine&`；`create_optimizer` 返回 `unique_ptr`/`nullptr` 与全库 `Result` 不一致 | 中 | 全仓用法审计 |
| D8 | **文档双源/过期 + 门禁补丁税**：08/09 互相矛盾且重复；`usage/03` 自相矛盾；`cpp-api-manual` 示例返回 `unique_ptr` 但工厂返回 `Result<shared_ptr>`；`NN_BIND_DEBUG` 第二轮 CI 是 stamp 机制的补丁税 | 中 | 对应文件 |
| D9 | 洁癖项：`Layer::BindingState` 三态；`nn::TensorRef` 与 `nn::dsl::TensorRef` 重名；`Model` unbound 走 abort、错误通道混用 | 低 | `compute_layer_base.hpp`、`model_container.hpp` |

**量化基线（本机探针，2026-10-20）**：`sizeof(Tensor)` = 64 B（CPU-only 构建；GPU 构建 88 B）；
`EngineId` = 8 B；活张量 copy+destroy ≈ 12.5 ns（含存储 `shared_ptr` 原子对）。v3 后 CPU 构建
`sizeof(Tensor)` = 56 B、GPU 构建 80 B；Tensor 拷贝少一次 8 B 复制（原子行为不变）。

## 3. 决策记录（2026-10-20 已确认）

| 决策 | 结论 |
|---|---|
| 破坏性 hard-cut | **接受**：删 `Tensor::engine_id()/bound()`、删 `NN_BIND_DEBUG`、删 `GpuBackend::instance()`；**分支 `dev/lifetime_refresh` 发布 `v1.10.0-rc0`（2026-10-20 用户裁定）**，迁移表进 release notes |
| 域模型 | **统一**：CPU/host 域 = 进程级常量；GPU 域 = backend 标识；空/无效张量 = 唯一豁免。**不给 CPU 开中性特例** |
| CPU↔GPU 交互 | 一律显式 `import`/`to_matrix`；**GPU 引擎拒绝 CPU 张量操作数**；据此删除 GPU impl 的 CPU 回退分支与逐 op 自动 `import`（审计后逐项删除） |
| CPU 引擎 | 与 GPU **统一**，但独立轨道 **P-CPU**（CpuBackend 同形重构），GPU 侧零依赖，不阻塞 GPU |
| GPU backend | **`shared()` 改 weak 缓存 + 删除 `instance()`**（确定性回收；设备重建成本只在"全部使用者归零之后"发生） |
| Tensor 持 `shared_ptr<Engine>` | **不做**（上一轮已论证：不修任何真实 bug，增加 8 B + 原子对 + 环/析构风险） |
| Optimizer 归属 | **不让 Model 拥有**；训练状态统一进 Session 层，见 [`29`](29-session-architecture.md) |
| 不做 | 全局引擎注册表；move-only Tensor；`TrainingSession` 以外的额外编排层（RL/推理先留占位设计） |

## 4. 目标模型

### 4.1 不变式

1. **存储自保活**：Tensor 只拥有存储；GPU 存储经 `GpuBuffer → shared_ptr<GpuBackend>` 保活。
   Tensor 可活过引擎、可跨 CPU 引擎使用。
2. **域从存储派生**：`DomainId`（`uintptr_t`）：
   - CPU 存储 → `kHostDomain`（进程级常量，所有 CPU 引擎共享）；
   - GPU 存储 → 其 `GpuBuffer` 的 backend 标识（地址；buffer 的 keepalive 保证地址不被复用）；
   - 空/无效张量 → 0（唯一豁免）。
   **不新增 Tensor 字段**，因此不存在 any 形式的传播/漏 stamp。
3. **入口检查逐操作数对本引擎**：每个非空操作数的域必须等于本引擎域。引擎域 token 由派生类构造时写入
   `ComputeEngine` 的 protected 成员（**非虚函数**，虚函数口径不变）：`CpuEngine` = `kHostDomain`，
   `GpuEngine` = backend。
4. **`import` 是唯一跨域入口**：CPU→GPU 上载；同域 = 别名/cast；**GPU 异域 fail-fast**（跨 backend
   请先落 CPU）；GPU→CPU 走 `to_matrix`（宿主 I/O）。
5. **会解引用引擎的对象全部持强引用**：Layer/Model/Optimizer/ExprGraph/**ExprSegment**；非拥有
   `TensorRef` 的 owner 契约由 Session 层结构化保证（见 29），自由工厂保留并文档化前置条件。

### 4.2 删除 / 新增清单（预估）

| 删除 | 新增 |
|---|---|
| core_engine_id.hpp（70 行）、core_engine_lease.hpp（13 行）；`EngineId` 字段/`id()`/ID 分配器 | `DomainId` 概念 + `kHostDomain`；`Tensor::device_domain_()`（friend 引擎，raw 访问器，避免 `shared_ptr` 原子） |
| `stamp_` 35 处 + `adopt` 18 处 + `to_prec` 内部 stamp；`reshape` 的 id 传播 | `GpuBuffer::backend_identity()` + 裸 buffer 访问器（import 热路约 1 700 次/step，禁止新增原子） |
| `bound()`/unbound 三态；`NN_BIND_DEBUG` 3 处 + CI 第二轮 ctest；`bind_check_` 的旧比较逻辑 | `bind_check_` 新实现（调用点 42 处不变）；`import` 快路设备守卫；错误信息"设备域不匹配" |
| core_engine_id.hpp 的 include（`compute_tensor`/`expr_dsl`/`expr_graph`/`model_container`） | 文档：单源生命周期文档（09 重写、08 删除/转发）、usage/03、cpp-api-manual、AGENTS 索引 |

净代码约 **−120 行**；机制点 **101 → 42**（只留 bind 调用点）。**预期数值路径零改动**：CPU/GPU 字节锚
必须逐位不变。

### 4.3 失败语义（目标）

| 场景 | 结果 |
|---|---|
| 张量活过引擎（含 GPU） | 安全；GPU 由存储 keepalive 保证 |
| 跨 CPU 引擎使用张量 | 允许（host 域相同；CPU 引擎无状态） |
| GPU 张量进入不同 backend 的引擎 | `Result` 错误（设备域不匹配），带 `source_location` 与形状 |
| CPU 张量进入 GPU 引擎 op | `Result` 错误，提示先 `import` |
| 跨 backend `import` | `Result` 错误，提示先落 CPU |
| 空张量参与运算 | 由各入口既有的无效/形状检查拒绝（域检查跳过空槽，不改变既有错误） |

## 5. 阶段与门禁

> P2（Session 层）见 [`29`](29-session-architecture.md)；本文只列 P0/P1/P3/P4 与 P-CPU。

### P0 — 契约/安全修复（不破坏 API，0.5–1 d）

- D3 文档纠错（08/09 的"优化器晚于 Model"假承诺）+ 负例测试（ASan 下 Model 先析构的行为在
  29 号 Session 落地前先按"契约 + 明确错误"处理）。
- D4 offload：`offloaded_` 在 `export_activations` 入口 fail-fast（不再静默 `return {}`）；f16/t1_58 +
  offload 的组合在配置期显式拒绝（最短路径；精度协议化另立）。
- D6 `ExprSegment` 改持 `shared_ptr<ComputeEngine>`；GPU `end_expr` 在无录制时返回明确错误（不再
  静默 no-op）。
- D8：08/09 矛盾、`usage/03` 自相矛盾、`cpp-api-manual` 过期示例；新增 CPU 可跑的负例测试。
- 门禁：CPU ctest 全绿、ASan 跑新增生命周期测试、`doc_align_audit` exit 0。

### P1 — v3 core（breaking，2–3 d）

- 实现 §4 的域模型；删 §4.2 全部删除项；重写 `bind_check_`、`import` 快路；审计并删除 GPU impl 的
  CPU 回退分支与逐 op 自动 `import`（用 GPU ctest 暴露现存的"CPU 张量喂 GPU 引擎"调用点，逐个改成显式
  `import`）；重写 `layer_lifetime_test`/`expr_fuse_test`；删 `NN_BIND_DEBUG` 与 CI 第二轮；更新
  文档/注释/AGENTS。
- **验收门禁（必须全过）**：
  1. `-Werror` 零告警（clang/GCC/MSVC 口径按现有 CI）；
  2. CPU ctest 全绿（含新增域/生命周期测试）；
  3. GPU ctest 全绿（850M + 40HX）；
  4. CPU/GPU 字节锚**逐位不变**；
  5. `layer_bench` 配对 A/B 无回退；
  6. `grep` 门禁：全仓无 `EngineId`/`NN_BIND_DEBUG` 残留；
  7. `bench/doc_align_audit.ps1`、`bench/doc_inventory.ps1` exit 0。
- 回滚：P1 可整体 revert（不依赖 P3/P4）；发布按 `v1.10.0-rc0` 处理。

### P3 — GPU backend 确定性生命周期（1 d）

- `GpuBackend::shared()` 改 `static std::weak_ptr` + mutex 缓存；删除 `instance()`，迁移 16 处调用点；
  文档明确"进程共享当且仅当有活使用者；确定性回收走 `create()` 或最后一个使用者释放"。
- 验收：GPU ctest 全绿；新增"全部释放 → 再 `shared()` 重建"回归；downstream 样例加销毁/重建；
  850M/40HX 各跑一轮。

### P4 — 文档与发布收尾（0.5 d）

- 09 重写为 v3 唯一真源；08 删除或 3 行转发；`docs/README.md`/AGENTS 索引同步；`history.md` 追加
  v3 条目；release-notes `v1.10.0-rc0` + 迁移表；`cpp-api-manual` 更新。

### P-CPU — CPU 引擎统一重构（独立轨道，1–2 d，P1 后）

- 目标：`CpuEngine` 与 `GpuEngine` 同形——"engine（门面）→ backend（资源/服务）→ domain"；`CpuBackend`
  承接线程池配置、host 分配账本、scratch、`describe()`/`pool_stats()`。
- 硬约束：**不得要求 GPU 侧改动**；`kHostDomain` 为进程常量 ⇒ CPU 存储不需要 keepalive。这是
  **资源依赖上的有意不对称**（宿主内存无设备依赖），不是 API 语义上的特殊化，文档写明。
- 不阻塞 GPU：P1 只冻结 `DomainId` 接口，P-CPU 在 P1 之后/并行推进。

## 6. 风险

| 风险 | 处置 |
|---|---|
| 破坏性 API 影响外部调用方 | `v1.10.0-rc0` + release notes 迁移表；`engine_id()/bound()` 无仓内调用方（仅测试/文档） |
| 删除 GPU impl 的 CPU 回退时漏掉真实路径 | 逐项 grep + 全量 GPU ctest（850M/40HX）；P1 先跑普通 ctest 再跑 `NN_BIND_DEBUG` 旧口径对照 |
| CPU/GPU 统一后出现"本该自动 import"的调用点 | P1 用 GPU ctest 暴露，逐个改为显式 `import`；这是预期内的一次性修正 |
| 跨 CPU 引擎诊断消失 | 接受：CPU 引擎无状态，跨 CPU 张量本身安全；GPU 域检查比现状更强 |
| P-CPU 范围蔓延 | 只做"同形重构 + 资源归位"，不做新原语/新后端；接口由 P1 冻结 |

## 7. 开放问题

1. P-CPU 是否需要支持"多 host backend"（多个 CPU 资源域）？当前判定：**不需要**，`kHostDomain`
   进程常量足够；若未来 CPU 侧出现需要隔离的资源（如独立线程池配额），再评估。
2. `Tensor::device_domain_()` 的访问边界：经 friend 给 `ComputeEngine` 与 `detail::TensorAccess`；
   是否需要暴露只读诊断接口（`domain()` 公开）？暂定不公开，避免重新引入"从张量找引擎"的用法。
3. P1 是否顺带做 D9 的 `BindingState` 两态化与错误通道统一？暂定**不在 P1**，随 29 号 P2 一起。

## 8. 参考

- Session 层：[`29-session-architecture.md`](29-session-architecture.md)
- 执行排序：[`27-execution-plan.md`](27-execution-plan.md)
- 裁定台账：[`18-roadmap.md`](18-roadmap.md) §4.2 P1-8
- 现行底层契约：[`17-unified-tensor-engine.md`](17-unified-tensor-engine.md)
- 现行生命周期文档（将被本文取代）：[`../development/08-object-lifetimes.md`](../development/08-object-lifetimes.md)、
  [`../development/09-engine-ownership.md`](../development/09-engine-ownership.md)
