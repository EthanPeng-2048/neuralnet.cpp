# 30 — TrainingSession 单步协议与梯度累积

> **读者**：项目开发者。
> **状态**：现行（已实施，随 `v1.11.0` 发布）。
> **关系**：本文修订 [`29-session-architecture.md`](29-session-architecture.md) 中
> “Session 只提供 zero_grad/clip/step 低层原语、Loss 全留训练工厂”的旧边界。
> `Model` 所有权、Optimizer 非拥有引用的生命周期修复不变。

## 1. 定位

`TrainingSession` 是**模型与优化器的训练生命周期容器**，也是**一次参数更新**的执行器；
它不是完整训练流程容器。

职责：

- 拥有 `Model`、`Optimizer` 和最小会话状态；
- 接收一个已经组装好的 **micro-batch**；
- 通过任务 Step 回调完成 forward / loss / backward；
- 内部维护梯度累积；
- 在更新边界执行归一化、裁剪、optimizer step、zero_grad；
- 返回本次 micro-step 是否真正更新了参数，以及可异步回读的 loss 汇总。

不负责：

- 数据集、样本集合、shuffle、样本顺序；
- epoch、lr 曲线、warmup、评估；
- checkpoint 触发时机、产物路径、日志排版；
- 判断哪些 token 参与 loss。

`TextTrainer` 仍是完整文本训练 Run Driver。它拥有数据与运行状态，逐个 micro-batch
驱动 Session；Loss 与文本语义通过任务 Step 回调进入，不进入 Session 本体。

## 2. Batch 与 Step 语义

术语固定如下：

| 术语 | 定义 |
|---|---|
| `global_batch_size` | 一次 optimizer update 覆盖的理论样本数；CLI `--batch-size` 采用此语义 |
| `grad_accum_steps` | 一次 update 包含的 micro-step 数；CLI `--accum-steps` |
| `micro_batch_size` | `global_batch_size / grad_accum_steps` |
| micro-step | 一次 forward + loss + backward，只累加梯度 |
| step | 一次 optimizer update；计数器、lr 调度、checkpoint 均以它为单位 |

约束：

- `global_batch_size > 0`、`grad_accum_steps > 0`；
- 正常批次必须满足 `global_batch_size % grad_accum_steps == 0`，否则 fail-fast；
- 数据集尾部 global batch 可以小于理论值；其 micro-batch 按剩余样本切分；
- 每个 global batch 的最后一个 micro-step 使用 `force_update`，避免尾部梯度悬挂；
- `steps_per_epoch = ceil(sample_count / global_batch_size)`；
- `max_steps`、`start_step`、`global_step`、`step_in_epoch`、lr step 调度均按 update step 计数。

## 3. Loss mask

“哪些 token 参与 loss”是样本语义，不是 Session 语义。

- `TextSample.loss_mask` 是样本内最终掩码；
- `TextTrainer` 在切样本时固化文档边界、窗口边界、assistant 区段等规则；
- `TextSampleBatch.flat_loss_mask` 只携带已计算好的逐位置 mask；
- Session 不识别 token、doc、assistant 或 CE，只接收 Step 回调给出的有效单位数。

## 4. Global-batch 归一化

不能把每个 micro-batch 的均值简单乘以 `1 / grad_accum_steps`。当 mask 导致各
micro-batch 的有效 token 数不同、或末批不满时，这会变成“micro 均值之均值”，
改变理论 global batch 的目标函数。

统一契约：

- Run Driver 在拆分 micro-batch 前算出本次 update 的
  `normalization_units`（文本 sparse-CE = 全 global batch 有效 token 数）；
- Step 回调返回的梯度已经按其 `gradient_units` 归一化；
- Session 对每个 micro 梯度缩放
  `gradient_units / normalization_units` 后 `model.backward()`；
- 等价于所有 micro-batch 按同一有效单位总和归一化；
- Step 回调同时返回设备侧未归一化 loss 分母量（sparse-CE 为 NLL sum）和
  `loss_units`，Session 在 update 完成时返回汇总，Run Driver 可继续异步回读。

如果 `normalization_units == 0`，本次 update 明确报错，不执行无意义参数更新。

## 5. API 草案

```cpp
struct TrainingConfig
{
    std::string optimizer;
    Scalar lr;
    // ... optimizer hyperparameters / precision / model memory switches
    std::size_t grad_accum_steps = 1;
};

struct TrainingStepOutput
{
    Tensor grad;                    // 对模型输出的梯度
    std::size_t gradient_units;     // grad 已采用的归一化单位数
    Tensor loss_sum;                // 可选：设备侧 loss 分子
    std::size_t loss_units = 0;
};

struct TrainingStepOptions
{
    std::optional<Scalar> lr;
    Scalar clip_grad_norm = 0;
    std::size_t normalization_units = 0;
    bool force_update = false;
    std::function<Result<bool>()> before_update;  // true = 中止且不更新
    std::function<Result<void>()> after_update;   // step 后、zero_grad 前
};

struct TrainingStepReport
{
    bool parameters_updated = false;
    std::uint64_t optimizer_step = 0;
    std::size_t microsteps_accumulated = 0;
    Tensor loss_sum;
    std::size_t loss_units = 0;
};

template <class Batch, class StepFn>
Result<TrainingStepReport>
TrainingSession::train_step(const Batch& batch,
                            TrainingStepOptions options,
                            StepFn&& step_fn);
```

`step_fn` 是任务适配边界：

```cpp
Result<TrainingStepOutput> step_fn(Model&, ComputeEngine&, const Batch&);
```

它是模板调用点，不进入会话长期状态；Text 路径捕获 `CrossEntropyLoss` 与词表信息。

## 6. 会话状态拆分

旧 `TrainingSession::train_state()` 同时承载 optimizer、epoch、lr 曲线和数据集身份，
违反 Session 定位。新边界：

| 状态 | Owner | 内容 |
|---|---|---|
| `TrainingSessionState` | Session | 当前 lr、optimizer step、累积 micro-step、loss 分子、累积是否有效 |
| `TrainingRunState` | `TextTrainer` / Run Driver | epoch、epoch 内 update step、样本位置、数据集身份、调度配方 |
| `TrainState` | 序列化 DTO | `.nnckpt` 状态段；保存时由 Run Driver 与 Session 状态共同装配 |

GPU 录制期间按铁律 6 管理张量生命周期：Session 的 pending loss、任务回调输出和梯度
均活到 `end_batch()`；任一步错误均结束录制并把累积标记为无效，调用方不得继续沿用
半完成的 update。

## 7. Batch-size 兼容

这是语义变更，不只是改名：

- 旧：`batch_size` = 每次 forward/backward 的 micro-batch，等效 update batch =
  `batch_size × accum_steps`；
- 新：`batch_size` / `global_batch_size` = 每次 update 的理论 batch，
  micro-batch = `batch_size / accum_steps`。

`.nnckpt` 兼容：

- 新状态段写 `batch_size_semantics="global"`；
- 缺少该键的旧状态按 legacy micro-batch 解释，恢复时转换为
  `global_batch_size = legacy_batch_size × accum_steps`；
- 旧断点若落在累积中间，pending gradients 本来未进入文件；迁移只恢复到最近的
  update 边界并明确文档化；
- 模型文件 v8 外层格式不变，只增加训练状态段键；缺键走 legacy 默认。

## 8. TextTrainer 新流程

每个 update step：

1. 从 shuffle 后样本下标取 `global_batch_size` 个样本；
2. 统计整个 global batch 的有效 token 数 `normalization_units`；
3. 按 `micro_batch_size` 切分；
4. 每个 micro 调 `TrainingSession::train_step`；
5. Session 返回 `parameters_updated=true` 时：
   - 把 `loss_sum/loss_units` 送入现有非阻塞 scalar readback；
   - 推进 TrainState / lr schedule / checkpoint 计数；
6. epoch 末、`max_steps`、数据集尾部强制 flush。

`TextTrainer` 不再手写 zero_grad / clip / optimizer.step / zero_grad，也不再手工乘
`1/accum_steps`。

## 9. 验收矩阵

### 单元

- `TrainingSession`：
  - 2 个 micro-step 后只执行 1 次 optimizer step；
  - 未到更新边界 `parameters_updated=false`；
  - `force_update` 可提交尾批；
  - global normalization 与手工构造的单批结果一致；
  - 错误/hook 中止后累积状态不被静默复用；
  - move 后 optimizer TensorRef 与累积状态有效。
- `TextSample` / `TextSampleBatch`：
  - mask 在 Sample 中确定，batch 逐位携带；
  - 不同 micro 有效 token 数不等时，梯度等于全 global batch 一次 backward。

### 集成

- `TextTrainer`：
  - `global_batch_size=8, accum=4` 的每 update 样本数 = 8；
  - `steps_per_epoch = ceil(samples/8)`；
  - 末批不足时仍执行一次 update；
  - `max_steps` / checkpoint / lr step 按 update 计数；
  - loss 异步回读数量与 update 数一致。
- CPU ctest 全绿；GPU 至少跑 text trainer + loss readback 冒烟。

### 兼容

- 新旧 batch-size 语义的 checkpoint 恢复；
- 旧状态缺 `batch_size_semantics` 时按 legacy 转换；
- 文档同步 `AGENTS.md`、API 手册、usage 与 todo 索引。

## 10. 实施顺序

1. Session：`TrainingStepOutput/Options/Report`、内部累积、全局归一化；
2. 状态：`TrainingSessionState`，把 `TrainState` 所有权移回 TextTrainer；
3. Data：Sample 有效单位计数、global/micro batch 切分；
4. TextTrainer：迁移训练循环与非阻塞 loss 回读；
5. CLI / checkpoint：语义键、resume 转换、帮助文案；
6. 测试与文档收口。
