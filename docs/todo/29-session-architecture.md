# 29 — Session 层：TrainingSession / InferenceSession / RLTrainingSession

> **读者**：项目开发者（本仓内部工作文档）。分册定位见 [`README.md`](README.md)。
> **状态**：🟡 **设计稿 + 待实施（2026-10-20 命名与方案已确认）**。
> **依赖**：[`28-lifecycle-v3.md`](28-lifecycle-v3.md)（P1 生命周期 v3 先落；本文对应执行计划 P2）。
> 执行排序见 [`27-execution-plan.md`](27-execution-plan.md)；裁定台账见 [`18-roadmap.md`](18-roadmap.md) §4.2 P1-8。

## 1. 动机

- **Model 是值对象**：推理只需要 `Model`（forward/backward/generate），不需要 Optimizer、不需要训练
  计数器；把 Optimizer 放进 Model 会让推理为训练付出语义代价。
- **D3 的真实修复不是"给 Model 加优化器"，而是"让非拥有引用的 owner 与引用者同生共死"**：
  Optimizer 的 `params_/grads_` 指向 Model 内 Layer 的 Tensor 成员，只要有一个对象同时拥有
  Model 与 Optimizer，UAF 就结构性地不可能发生。
- **为后续形态留同一模式**：推理服务需要会话缓存（KV cache / prompt cache / 命中统计），RL 需要
  多模型 + rollout + 多优化器——它们都是"**Session 拥有模型与派生状态，引擎是共享资源**"。现在把
  Session 层立起来，后续不需要再改 Model 的语义。

## 2. 命名与分层

**统一使用 `*Session` 后缀**。不使用 `*Engine`：`ComputeEngine` / `CpuEngine` / `GpuEngine` 已经占用了
"Engine = 计算后端门面"这个词，再用 `TrainEngine`/`InferenceEngine` 会让"engine"同时指后端与编排层，
参数名与方法名都会歧义。

| 层 | 名字 | 职责 | 状态 |
|---|---|---|---|
| L2 | `ComputeEngine`（`CpuEngine`/`GpuEngine`） | 计算原语/设备门面 | 现行，不动 |
| L3 | `Model` | 网络容器 + forward/backward | 现行，**保持纯值对象** |
| L4 | **`TrainingSession`** | 拥有 Model + Optimizer + 训练状态；训练生命周期容器 | 本文 P2 |
| L5 | `InferenceSession`（占位） | 拥有 Model + 推理会话缓存（KV/prompt/命中统计） | 未来 |
| L5 | `RLTrainingSession`（占位） | 拥有 policy/reference 多模型 + 多优化器 + rollout | 未来 |

命名细则：P2 先落 `TrainingSession`；`InferenceSession` / `RLTrainingSession` 只写接口占位，不实现，
避免提前抽象（触发条件 = 真实调用方出现）。

## 3. 所有权与不变式

```
TrainingSession
 ├─ Model model_;                    // 值语义，move 进来；内部持 shared_ptr<ComputeEngine>
 │    └─ vector<unique_ptr<Layer>> → 参数/梯度 Tensor 成员（对象地址稳定）
 ├─ unique_ptr<Optimizer> opt_;      // 声明在 model_ 之后 → 先于 model_ 析构
 │    ├─ shared_ptr<ComputeEngine>
 │    └─ vector<TensorRef> → model_ 的 Tensor 成员
 └─ TrainState state_;               // 计数器 / lr / 精度配方 / 断点状态（P2b）
```

不变式：

1. **Session 是 Optimizer 的唯一创建者**：`opt_` 只从 `model_` 取 `parameters()/param_gradients()`，
   引用永远有效。
2. **成员声明顺序 = 析构顺序的正确性**：`model_` 声明在前、`opt_` 在后 ⇒ optimizer 先析构；
   Session 析构时 optimizer 不会解引用已销毁的 Tensor。
3. **不可拷贝、可移动**：移动 Model 不搬动 Layer 对象（`unique_ptr` 指向堆），所以 Session 移动后
   optimizer 的 TensorRef 仍有效——用测试锁住。
4. **没有 "Session 建成后再加层" 的路径**：Model 已 move 进 Session，不存在"optimizer 看不到新参数"
   的静默漏训。
5. **引擎由 Model（与 Optimizer）的 `shared_ptr<ComputeEngine>` 持有**；Session 不持裸引擎引用。
6. **Loss 不归 Session**：Loss 是任务相关的，保持调用方按 step 组合；若日后证明必要，再加
   `attach_loss(std::unique_ptr<Loss>)` + 便捷 `train_step`，不提前塞训练语义。

## 4. `TrainingSession` API 草案（P2a）

```cpp
namespace nn {

struct TrainingConfig
{
    std::string optimizer = "adamw";
    Scalar lr = 1e-4f;
    Scalar weight_decay = 0.01f;
    PrecisionProfile precision{};
    Scalar beta1 = 0.9f, beta2 = 0.999f;
    Scalar clip_grad_norm = 0;          // 0 = 不裁剪
    bool activation_offload = false;
    std::size_t checkpoint_every = 0;
};

class TrainingSession
{
public:
    // 唯一入口：Model 必须已构建（engine 已绑定）；内部按 cfg 创建 optimizer。
    [[nodiscard]] static Result<TrainingSession> create(Model model,
                                                        TrainingConfig cfg = {});
    TrainingSession(TrainingSession&&) noexcept = default;
    TrainingSession& operator=(TrainingSession&&) noexcept = default;
    TrainingSession(const TrainingSession&) = delete;

    [[nodiscard]] Model& model() noexcept;
    [[nodiscard]] const Model& model() const noexcept;
    [[nodiscard]] Optimizer& optimizer() noexcept;
    [[nodiscard]] ComputeEngine& engine() noexcept;      // = model().engine()

    // 训练原语（epoch/step 循环与数据集仍留在调用方/CLI）
    [[nodiscard]] Result<void> zero_grad();
    [[nodiscard]] Result<void> clip_grads();             // cfg.clip_grad_norm
    [[nodiscard]] Result<void> step();                   // optimizer.step() + t++
    void set_lr(Scalar lr);
    [[nodiscard]] std::uint64_t step_count() const noexcept;

    // P2b（训练状态段）
    // [[nodiscard]] Result<void> save(std::string_view path, ...);
    // [[nodiscard]] Result<void> load_state(std::string_view path);

private:
    Model model_{};
    std::unique_ptr<Optimizer> opt_{};
    TrainingConfig cfg_{};
    // TrainState state_{};   // P2b
};

} // namespace nn
```

**刻意不做的事**（防 god object）：

- 不碰数据集 / epoch / CLI 参数 / 日志排版；
- 不替代 `Model::forward/backward`（推理可直接用 Model，不需要 Session）；
- 不封装 Loss / 多损失 / 自定义 step；
- 不新增计算原语、不改 `ComputeEngine` 虚表。

## 5. 生命周期保证与失败语义

| 场景 | 结果 |
|---|---|
| Model 与 Optimizer 同属 Session | 参数永远活到 optimizer 之后；无 UAF |
| Session 移动 | 安全（Layer 对象在堆上，TensorRef 仍有效）；测试锁住 |
| 需要推理 | `session.model()` 可直接 forward/generate；或独立构建 Model（不建 Session） |
| 外部仍走自由工厂 `create_optimizer` | 保留；文档写"optimizer 必须在参数所有者存活期内销毁"；**不**加 weak-token（无真实外部调用方前不加机制；触发条件见 §10） |
| `Model::add` 在 Session 之后 | 不存在：Model 已 move 进 Session，没有变更入口 |
| 断点恢复（P2b） | `TrainingSession::load_state` 只能在本 Session 创建后调用；计数器/lr/优化器状态由状态段决定 |

## 6. `InferenceSession`（占位设计，不实现）

动机：推理服务需要"模型 + 每会话缓存"的生命周期绑定，缓存命中率/淘汰策略属于会话状态，不属于 Model。

职责草案：

- 拥有 `Model`；持有 `shared_ptr<ComputeEngine>`；
- 会话缓存（KV cache、prompt/prefix cache、命中统计、可选的持久化）；
- API 草案：`create(Model, InferenceConfig)`、`generate/forward`、`cache_stats()`、`reset_cache()`；
- 生命周期不变式与 TrainingSession 相同：Session 拥有 Model，缓存随会话销毁；引擎共享。
- **不现在做**：触发条件 = 推理服务化/多轮缓存有真实调用方（见 [`18`](18-roadmap.md) §6 触发清单口径）。

## 7. `RLTrainingSession`（占位设计，不实现）

职责草案：

- 拥有 policy Model（+ 可选 reference Model）、各自 Optimizer、rollout buffer、reward/advantage 计算所需状态；
- **组合而非继承**：可以是两个模型 + 两个 optimizer 的聚合，不要求复用 TrainingSession 的子类关系；
- 生命周期规则同一条：Session 拥有模型与派生状态，引擎是共享资源。
- **不现在做**：触发条件 = RL 训练入口/算法落地。

## 8. 迁移计划

### P2a — Session 核心（不改 CLI，1–1.5 d）

- 新头文件 `include/neuralnet.cpp/training_session.hpp`（或按 29 落地时确定的文件名）；
- `TrainingSession::create` 内部调用现有 `create_optimizer`；`TrainingConfig` 映射现有参数；
- 测试（P2a 新增测试目标，随 CMake 注册）：
  1. 创建 → 8 步 step 参数移动（复用 `optimizer_test` 思路）；
  2. Session 移动后 step 正常；
  3. ASan：Session 作用域结束时 optimizer 与 Model 析构顺序正确；
  4. 推理：`session.model()` 可 forward；独立 Model（无 Session）也可 forward；
  5. `--f16` / 三值 profile 透传正确。

### P2b — CLI 迁移 + 训练状态（1–1.5 d）

- `text_train` / `mnist_train` / `cli/text_train_loop.hpp` 改用 `TrainingSession`；
- `TrainState`（计数器 / lr 曲线 / 精度配方）、`.nnckpt` 保存/恢复收进 Session；
- 删除各入口重复的 "build model → create optimizer → state" 样板；
- 验收：同 seed/同数据下训练输出与现行**逐字节一致**（loss 序列、`.nnmodel`/`.nnckpt` md5）；
  `text_train_modules_test` / `model_roles_test` / `optimizer_state_test` 全绿；
- 文档：`usage/` 训练示例切到 Session；`AGENTS.md` 训练范式更新（P4）。

## 9. 验收（P2 整体）

1. ASan 下 Session 生命周期测试全绿；
2. Session 移动后 TensorRef 有效（测试）；
3. CLI 端到端逐字节一致（见 P2b）；
4. 推理路径不依赖 Session（`Model` 独立可用）；
5. 文档：29 状态更新；`usage/`/`AGENTS` 训练示例同步。

## 10. 风险与开放问题

| 风险 / 问题 | 处置 |
|---|---|
| Session 变成 god object | 明确 §4 "刻意不做"；数据/CLI/Loss 全留在调用方 |
| Loss 是否归 Session | 暂不归；触发 = 出现第三种训练循环且重复代码显著 |
| TrainState 范围 | P2b 只搬现有 `TrainState` 字段，不新增语义；不改变 v8 文件格式 |
| 自由工厂的 UAF 可能 | 文档前置条件；真实外部调用方出现后再加 weak-token（低成本、可后补） |
| 命名细节：`RLTrainingSession` 是否太长 | 先用全称，实现时再定简称（`RLSession` 等）；不影响 P2 |
| 未来推理/RL 的接口是否被现在锁死 | 只冻结"Session 拥有模型 + 引擎共享"这一条模式；具体 API 留到对应立项 |

## 11. 参考

- 生命周期 v3：[`28-lifecycle-v3.md`](28-lifecycle-v3.md)
- 执行排序：[`27-execution-plan.md`](27-execution-plan.md)
- 裁定台账：[`18-roadmap.md`](18-roadmap.md) §4.2 P1-8
- 现行优化器/训练状态：[`17-unified-tensor-engine.md`](17-unified-tensor-engine.md)、
  [`23-library-design-cleanup-and-model-file-roles.md`](23-library-design-cleanup-and-model-file-roles.md)
