# 29 — Session 层：TrainingSession / TrainingGroupSession / InferenceSession

> **读者**：项目开发者（本仓内部工作文档）。分册定位见 [`README.md`](README.md)。
> **状态**：✅ **P2a 已实施（2026-10-20，`training_session.hpp` + `training_session_test`）**；
> ✅ **P2b 已实施（2026-10-21，v1.10.0-rc0.1）**：`TrainingSession` 持有 `TrainState`；
> 数据集经 `TextSampleSet`（样本 = 窗口）**直接接入训练工厂 `TextTrainer`**，CLI 退出训练循环
> 实现（见 §8）。`InferenceSession` 仍为设计占位；`TrainingGroupSession` 的通用架构已冻结、
> 尚未实施（§6/§7）。
> **2026-10-22 修订**：[`30-training-session-step.md`](30-training-session-step.md) 已冻结
> `TrainingSession` 的单步协议、内部梯度累积、global/micro batch 语义与 `TrainState`
> 所有权拆分；与本文 §3 第 6 条、§4 “刻意不做”及 §8 TrainState 归属冲突处，以 30 号为准。
> **2026-10-23 库化修订**：[`31-rl-training-primitives.md`](31-rl-training-primitives.md)
> 已冻结通用多模型会话设计；`RLTrainingSession` 不再是核心 Session 类型，RL 多模型场景
> 使用通用 `TrainingGroupSession`，算法与 rollout 留在 RL 编排层。
> **依赖**：[`28-lifecycle-v3.md`](28-lifecycle-v3.md)（P1 生命周期 v3 先落；本文对应执行计划 P2）。
> 执行排序见 [`27-execution-plan.md`](27-execution-plan.md)；裁定台账见 [`18-roadmap.md`](18-roadmap.md) §4.2 P1-8。

## 1. 动机

- **Model 是值对象**：推理只需要 `Model`（forward/backward/generate），不需要 Optimizer、不需要训练
  计数器；把 Optimizer 放进 Model 会让推理为训练付出语义代价。
- **D3 的真实修复不是"给 Model 加优化器"，而是"让非拥有引用的 owner 与引用者同生共死"**：
  Optimizer 的 `params_/grads_` 指向 Model 内 Layer 的 Tensor 成员，只要有一个对象同时拥有
  Model 与 Optimizer，UAF 就结构性地不可能发生。
- **为后续形态留同一模式**：推理服务需要会话缓存（KV cache / prompt cache / 命中统计），RL 等
  多模型训练需要多个可训练/只读角色——它们都是"**Session 拥有模型与派生状态，引擎是共享资源**"。
  但 rollout、算法和 I/O 不进入 Session；它们属于 Run Driver / Pipeline。

## 2. 命名与分层

**统一使用 `*Session` 后缀**。不使用 `*Engine`：`ComputeEngine` / `CpuEngine` / `GpuEngine` 已经占用了
"Engine = 计算后端门面"这个词，再用 `TrainEngine`/`InferenceEngine` 会让"engine"同时指后端与编排层，
参数名与方法名都会歧义。

| 层 | 名字 | 职责 | 状态 |
|---|---|---|---|
| L2 | `ComputeEngine`（`CpuEngine`/`GpuEngine`） | 计算原语/设备门面 | 现行，不动 |
| L3 | `Model` | 网络容器 + forward/backward | 现行，**保持纯值对象** |
| L3 | `TextSampleSet`（`dataset_samples.hpp`） | 数据集子集 → **训练样本**（窗口）与 batch 装配 | 本文 P2b，现行 |
| L4 | **`TrainingSession`** | 拥有 Model + Optimizer + 单步/累积状态；执行一次 update | 本文 P2a，30 号修订 |
| L4 | `TrainingGroupSession`（架构已冻结、未实施） | 以稳定名称拥有多个可训练 Session / 只读 Model；只管角色与生命周期 | [`31`](31-rl-training-primitives.md) §5.2 |
| L5 | **`TextTrainer`**（`text_trainer.hpp`） | Run Driver：数据集 → global/micro batch → Session；epoch/step、lr、评估、运行态 `TrainState` | 本文 P2b，30 号修订 |
| L5 | `InferenceSession`（占位） | 拥有 Model + 推理会话缓存（KV/prompt/命中统计） | 未来 |

命名细则：P2 先落 `TrainingSession`；`TrainingGroupSession` 是多模型生命周期的通用容器，
`InferenceSession` 是多轮推理缓存容器。**不新增核心 `RLTrainingSession`**：RL 的模型角色交给
`TrainingGroupSession`，rollout/算法/buffer/checkpoint 策略交给 RL 编排层。

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
6. **Loss 不归 Session**：Loss 是任务相关的，经 `TrainingSession::train_step` 的 Step 回调进入；
   Session 只执行累积/裁剪/update（30 号）。

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

- Session **不碰**数据集 / epoch / CLI 参数 / 日志排版；这些属于训练工厂 `TextTrainer`（见 P2b 段）；
- 不替代 `Model::forward/backward`（推理可直接用 Model，不需要 Session）；
- 不封装 Loss / 多损失 / 自定义 step（Loss 仍由训练工厂按任务持有）；
- 不新增计算原语、不改 `ComputeEngine` 虚表。

> P2b 的边界补充：**数据集不归 Session**——`TextTrainer` 接收 `Dataset`（或已切好的
> `TextSampleSet`）并驱动循环；Session 只持有 Model/Optimizer/单步状态。这样
> 推理/RL 复用 Session 时不会被文本数据语义污染。

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

## 7. `TrainingGroupSession`（通用多模型会话，架构已冻结、未实施）

库化后不能假设只有 RL 的 actor/critic/reference 会需要多模型生命周期。通用容器以稳定名称
持有若干角色，每个角色可以是：

- 一个可训练的 `TrainingSession`（Model + Optimizer）；
- 一个只读 `Model`（reference / teacher / frozen generator）。

职责仅限：

- 保证所有角色及其 optimizer 的生命周期正确；
- 校验角色名称、重复绑定、engine/schema 兼容性；
- 提供 role 查询，不解释算法。

明确不做：

- 不拥有 rollout、environment、replay buffer、RNG、PPO epoch 或 checkpoint I/O；
- 不把 reward/advantage 计算塞进 Session；
- 不特别绑定 RL；actor-critic、GAN、ensemble、多智能体都可复用。

`RLTrainingSession` 不再是核心类型。若以后为易用性保留同名工厂，它只能组装
`TrainingGroupSession` 和 RL 标准配置，不拥有第二套生命周期或训练更新语义。
完整职责边界与验收见 [`31`](31-rl-training-primitives.md) §5。

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

### P2b — CLI 迁移 + 训练状态 + 数据集接入训练工厂（✅ 2026-10-21，v1.10.0-rc0.1）

已落地（`text_train`；`mnist_train` 同步迁移到 Session）：

- **样本概念显式化**（`dataset_samples.hpp`）：存储单位 = doc，训练单位 = **sample**
  （一个 `seq_len` 窗口）；`step = ceil(sample_count / batch_size)` 是派生量。
  `TextSampleSet::create(dataset, subset, TextWindowConfig)` 一次给出切窗 + batch 装配
  （PAD、目标 = 下一 token、掩码按目标位置、doc_ids PAD=0、batch-major 扁平化）。
- **数据集直接接入训练工厂**（`text_trainer.hpp`）：

  ```cpp
  auto dataset = NN_UNWRAP(nn::Dataset::open(path));
  auto session = NN_UNWRAP(nn::TrainingSession::create(std::move(model), session_cfg));
  auto trainer = NN_UNWRAP(nn::TextTrainer::create(std::move(session), dataset, recipe));
  auto report  = NN_UNWRAP(trainer.fit());   // epoch×step / 累积 / 裁剪 / 评估
  ```

  `TextTrainer` 拥有 Loss 与循环，接受 hooks（进度输出、梯度日志、探针、定期断点）。
- **`TrainState` 收进 Session**：`train_state()/set_train_state()`；写入方是训练工厂
  （epoch/step 位置只有循环知道），Session 保证生命周期；断点状态装配唯一实现 =
  `nn::make_text_train_state`。`fit()` 结束时状态 = **真实下一步位置**（正常跑完
  `(epochs,0)`；`--max-steps` 提前停为实际停止位置；探针中止不更新 Session 状态）。
- **配方一致性**：优化器侧字段（optimizer/lr/weight_decay/beta/precision/max_norm）的
  真源是 `TrainingConfig`；`TextTrainConfig` 同名副本必须一致（`TextTrainer::create`
  逐项校验），起步用 `nn::make_text_train_config(session)`。
- **CLI 退出循环实现**：`cli/text_train_loop.hpp` 只留 CLI 侧 hooks（梯度日志、
  `NN_NAN_TRACE`/`NN_MEM_STATS`、`.nnckpt` 写盘），`text_train.cpp` 不再手工拼 corpus/
  切窗/填 batch；`mnist_train` 不再用自由工厂 `create_optimizer`。
- **配方单一真源**：`nn::TextTrainConfig`（库层）；CLI `TrainConfig` 继承它，只额外保留
  入口字段；`lr_scheduler.hpp` 上收到库层（`cli/cli_lr_scheduler.hpp` 变转发头）。
- **命名归一**：`DatasetIdentity.num_windows → num_samples`（断点键 `dataset_samples`，
  旧键 `dataset_windows` 兼容读）；v8 文件格式与版本号不变。
- 验收：新增 `dataset_samples_test` / `text_trainer_test`；`text_train_modules_test` /
  `train_state_test` / `training_session_test` 全绿；CPU ctest 57/57。
- 文档：`usage/` 训练示例、API 手册、AGENTS 训练范式同步（P4 同批完成）。
- **仍未做（明示）**：`P2-4` 的流式 DataLoader（`TextSampleSet` 仍整段物化
  `TextCorpus`）；`InferenceSession` 仍为设计占位，`TrainingGroupSession` 架构已冻结但未实施；
  `create_optimizer` 自由工厂签名未收口（D7）。

## 9. 验收（P2 整体）

1. ASan 下 Session 生命周期测试全绿；
2. Session 移动后 TensorRef / 累积状态有效（测试）；
3. CLI 端到端：循环由 `TextTrainer` 与旧 CLI 循环逐行等价迁移（代码同构 + 正常路径文案/顺序抽查）；CPU/GPU ctest 全绿 + 真实文本冒烟（无 seed API，不承诺"同 seed 轨迹逐位一致"）；
4. 推理路径不依赖 Session（`Model` 独立可用）；
5. **样本概念**：数据集样本数 / 派生 step 数 / batch 逐位约定由 `dataset_samples_test` 锁住；
6. 文档：29 状态更新；`usage/`/`AGENTS` 训练示例同步。

## 10. 风险与开放问题

| 风险 / 问题 | 处置 |
|---|---|
| Session 变成 god object | 30 号冻结为 micro-batch 单步执行器；数据/CLI/Loss/epoch/lr 全留 Run Driver |
| `TextTrainer` 变成 god object | 只做"数据顺序 + 训练循环 + 状态装配"；输出/断点/探针走 hooks；Loss 仍按任务注入（当前内置 CE，后续可泛化） |
| Loss 是否归 Session | 不归；归训练工厂（P2b），触发 = 出现第三种训练循环且重复代码显著 |
| TrainState 范围 | P2b 只搬现有 `TrainState` 字段，不新增语义；v8 文件格式不变（仅键 `dataset_windows → dataset_samples`，旧键兼容读） |
| 自由工厂的 UAF 可能 | 文档前置条件；真实外部调用方出现后再加 weak-token（低成本、可后补） |
| 多模型 Session 是否绑定 RL | 不绑定；通用 `TrainingGroupSession` 只管角色与生命周期，RL 算法/rollout 留在 31 号的编排层 |
| 未来推理/RL 的接口是否被现在锁死 | 只冻结"Session 拥有模型 + 引擎共享"这一条模式；具体 API 留到对应立项 |

## 11. 参考

- 生命周期 v3：[`28-lifecycle-v3.md`](28-lifecycle-v3.md)
- 执行排序：[`27-execution-plan.md`](27-execution-plan.md)
- 裁定台账：[`18-roadmap.md`](18-roadmap.md) §4.2 P1-8
- RL 与通用训练基础设施设计：[`31-rl-training-primitives.md`](31-rl-training-primitives.md)
- 现行优化器/训练状态：[`17-unified-tensor-engine.md`](17-unified-tensor-engine.md)、
  [`23-library-design-cleanup-and-model-file-roles.md`](23-library-design-cleanup-and-model-file-roles.md)
