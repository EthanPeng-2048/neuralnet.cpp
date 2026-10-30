# 强化学习与可组合训练基础设施

> **读者**：需要在 neuralnet.cpp 之上实现 policy gradient、PPO、GRPO，
> 或把模型/rollout/buffer 拆成可替换组件的库使用者。
> **状态**：现行（已实施，随 `v1.12.0` 发布）。

本库提供的是**RL 基础设施**，不是内置环境或游戏框架。它包含：

- 具名多输出模型协议；
- schema 化 experience 与 advantage 变换；
- categorical policy distribution、可替换分布接口；
- REINFORCE、PPO、GRPO 标准 preset 与低层 objective 组件；
- 通用多模型 `TrainingGroupSession`；
- 可替换 rollout source、buffer、sampler、metrics、checkpoint 和 hooks。

环境、棋规、MCTS、自对局 worker 和领域 replay 策略仍在消费者侧。

## 1. 引入

完整入口：

```cpp
#include <neuralnet.cpp/nn.hpp>
```

只使用 RL 子系统：

```cpp
#include <neuralnet.cpp/rl.hpp>
```

所有公共失败路径返回 `Result<T>`；不要在校验失败后继续解引用。

## 2. 形状与字段约定

| 数据 | 推荐形状 | 说明 |
|---|---|---|
| policy logits | `(actions, T)` | categorical softmax 的输入 |
| action / old_log_prob / advantage / mask | `(1, T)` | 每个决策位置一项 |
| value / old_value / return | `(1, T)` | value head 输出一致 |
| GAE reward / value / done / mask | `(seq, batch)` | row=时间，col=独立轨迹 |
| GRPO returns | `(N,1)` | 配合宿主 `GroupSpan` |

批量序列全局仍遵守 batch-major：`i = batch * seq + time`。经验 schema 只校验
声明的形状/精度/设备约束，不把文本、token 或棋局语义写进核心。

## 3. 具名多输出

`TensorSchema` 是有序字段表。名称只在 setup 期解析为 `FieldHandle`，热路径按
handle 下标访问。

```cpp
auto schema = NN_UNWRAP(nn::TensorSchema::create(
    "policy_value",
    {
        nn::tensor_field(
            "policy",
            nn::TensorShapeSpec{.rows = 362, .cols = std::nullopt}),
        nn::tensor_field(
            "value",
            nn::TensorShapeSpec{.rows = 1, .cols = std::nullopt}),
    }));

nn::Model model(engine);
NN_UNWRAP(model.add<nn::Linear>(256, 512));

auto bundle = std::make_unique<nn::OutputBundleLayer>(schema);
NN_UNWRAP(bundle->add_port(
    "policy", std::make_unique<nn::Linear>(512, 362)));
NN_UNWRAP(bundle->add_port(
    "value", std::make_unique<nn::Linear>(512, 1)));

// 仅当旧调用方仍需要单 Tensor 时显式启用。
NN_UNWRAP(bundle->set_row_band_packing({"policy", "value"}));
NN_UNWRAP(model.add_layer(std::move(bundle)));

auto outputs = NN_UNWRAP(model.forward_named(input));
auto policy = NN_UNWRAP(outputs.require("policy"));
auto value = NN_UNWRAP(outputs.require("value"));
```

`Model::forward_named` 让中间层保持普通 `Tensor` 数据流，最后一层可以展开成多个
端口。`Model::backward_named` 按端口反向，再把共享 trunk 的梯度求和。

`RowBandPacking` 把端口映射到连续行带，例如 `(362+1,T)`。它只是兼容单 Tensor
ABI 的适配器；named API 从不依赖私有行号约定。需要行带时，schema 的每个端口
必须有固定 `rows`。

`ModelSpec.output_schema` / `output_packing` 会随模型文件保存；旧文件缺键仍是
单输出模型。

## 4. Experience schema

```cpp
auto exp = NN_UNWRAP(nn::ExperienceSchema::create(
    {"action", "action_mask", "advantage", "return", "old_value",
     "old_log_prob"},
    {
        nn::tensor_field("consumer_score"),  // 自定义字段
    }));

auto record = NN_UNWRAP(nn::TensorRecord::create(
    exp.schema(),
    {
        nn::NamedTensor{.name = "action", .value = action},
        nn::NamedTensor{.name = "action_mask", .value = action_mask},
        nn::NamedTensor{.name = "value_mask", .value = value_mask},
        nn::NamedTensor{.name = "advantage", .value = advantage},
        nn::NamedTensor{.name = "return", .value = ret},
        nn::NamedTensor{.name = "old_value", .value = old_value},
        nn::NamedTensor{.name = "old_log_prob", .value = old_log_prob},
        nn::NamedTensor{.name = "consumer_score", .value = score},
    }));

auto view = NN_UNWRAP(nn::PolicyBatchView::create(record));
```

标准字段始终登记在 schema 中但默认可选；`create` 的 required 列表负责收紧。
未知字段、重复字段、缺必需字段以及 shape/precision/placement 不符都会返回错误。

`PolicyBatchView` 是只读强类型视图。自定义 objective 仍可经 `TensorRecord::require`
读取自己的字段，核心代码不需要分支。

## 5. GAE、return 与 normalization

GAE 的递推状态留在设备侧：

```cpp
nn::GaeAdvantage gae({.gamma = 0.99f, .lambda = 0.95f});
auto gae_out = NN_UNWRAP(gae.apply(
    engine, reward, value, &done, &mask, &bootstrap_value));

nn::MonteCarloReturn mc(0.99f);
auto ret = NN_UNWRAP(mc.apply(engine, reward, &done, &mask));

nn::MaskedNormalizer normalizer;
auto normalized = NN_UNWRAP(normalizer.apply(engine, advantage, mask));
```

- `done=1` 清空 bootstrap 与后续 advantage 传播；
- mask 外的时间步不参与递推和统计；
- normalization 使用**总体标准差**；
- 全 mask 或零方差返回 0，不产生 NaN。

GRPO 的分组标准化：

```cpp
const std::vector<nn::GroupSpan> groups{
    {.first_row = 0, .row_count = group_size},
    {.first_row = group_size, .row_count = group_size},
};
auto grouped = NN_UNWRAP(group.apply(engine, returns, groups));
```

## 6. Policy distribution 与低层 objectives

`PolicyDistribution` 是扩展接口；首版提供 `CategoricalDistribution`。Objective
只依赖该接口的 log-prob、entropy、sample 和梯度入口，因此 Gaussian、
multi-discrete 或消费者自定义分布可复用同一目标组件。

```cpp
nn::CategoricalDistribution policy;
auto log_prob = NN_UNWRAP(policy.log_prob(engine, logits, actions));
auto entropy = NN_UNWRAP(policy.entropy(engine, logits));
auto sample = NN_UNWRAP(policy.sample(engine, logits, seed));
```

REINFORCE：

```cpp
nn::ReinforceObjective reinforce;
auto loss = NN_UNWRAP(reinforce.forward(engine, logits, view));
auto grad_logits = NN_UNWRAP(reinforce.backward());
```

`ReinforceObjective(true)` 使用 `return - old_value` 作为 baseline。低层组件还包括：

- `ValueRegressionObjective`；
- `EntropyBonusObjective`；
- `SampledKlObjective`。

它们都适配 `Loss`，但 context-only 入口是
`forward(engine, prediction, PolicyBatchView)`。直接调用
`forward(engine, pred, target)` 会明确报错，不会猜字段名。

## 7. PPO

低层组件：

```cpp
nn::ClippedPolicyObjective ppo_policy(0.2f);
nn::ClippedValueObjective ppo_value(0.2f, 0.5f);
```

高层 preset：

```cpp
nn::PpoConfig cfg;
cfg.clip_epsilon = 0.2f;
cfg.value_clip_epsilon = 0.2f;
cfg.value_coefficient = 0.5f;
cfg.entropy_coefficient = 0.01f;

nn::PpoObjective ppo(cfg);
auto report = NN_UNWRAP(
    ppo.evaluate(engine, model_output_schema, policy_logits, value, view));

// 推荐：让 TrainingSession 统一处理 global normalization / 累积 / clip / step。
nn::TrainingStepOptions options;
options.normalization_units = valid_action_count;
auto step = NN_UNWRAP(session.train_named_step(
    micro_batch, options,
    [&](nn::Model& model, nn::ComputeEngine& engine,
        const MicroBatch& input) -> nn::Result<nn::NamedTrainingStepOutput> {
        auto outputs = NN_UNWRAP(model.forward_named(input.tensor));
        auto policy = NN_UNWRAP(outputs.require("policy"));
        auto value = NN_UNWRAP(outputs.require("value"));
        auto report = NN_UNWRAP(
            ppo.evaluate(engine, model_output_schema,
                         *policy, *value, view));
        return nn::NamedTrainingStepOutput{
            .gradients = std::move(report.gradients),
            .gradient_units = valid_action_count,
            .loss_sum = nn::Tensor{},
            .loss_units = 0,
        };
    }));
```

`PpoReport` 同时提供 policy/value loss、entropy、approx KL、policy/value clip
fraction。缺失 `old_log_prob`、`old_value` 或显式 mask 会返回可定位错误。
`train_named_step` 与单输出 `train_step` 使用同一 update/累积/归一化语义，只把
最后一层 backward 换成 `Model::backward_named`。

## 8. GRPO

GRPO 先对每个 group 的 return 做总体标准化，再复用 `ClippedPolicyObjective`
与 entropy 组件：

```cpp
nn::GrpoObjective grpo;
auto report = NN_UNWRAP(
    grpo.evaluate(engine, policy_only_schema, logits, view, group_spans));
NN_UNWRAP(model.backward_named(report.gradients));
```

record 已有 `advantage` 时直接使用；否则要求 `return` 与完整 group spans。
零方差组的 advantage 为 0。

## 9. 多模型角色

`TrainingGroupSession` 是通用容器，不绑定 actor-critic 名称：

```cpp
nn::TrainingGroupSession group;
NN_UNWRAP(group.add_training_role("actor", std::move(actor_session)));
NN_UNWRAP(group.add_training_role("critic", std::move(critic_session)));
NN_UNWRAP(group.add_readonly_role("reference", std::move(reference_model)));

auto* actor = NN_UNWRAP(group.training_session("actor"));
auto* ref = NN_UNWRAP(group.model("reference"));
```

约束：

- role 名非空且唯一；
- 所有角色共享同一 `ComputeEngine`；
- 可训练角色保持 `TrainingSession` 的 optimizer 生命周期；
- 不同角色可有不同 output schema；
- `TrainingSession` 的 micro/global batch 归一化语义不变。

## 10. Rollout pipeline

实现 `RolloutSource`、`BatchSampler`、`MetricSink`、`CheckpointProvider` 后装配：

```cpp
nn::RLPipelineConfig cfg;
cfg.batch_size = 32;
cfg.collect_per_update = 32;
cfg.buffer_capacity = 4096;
cfg.clear_buffer_after_update = true;  // on-policy
cfg.checkpoint_every = 100;

auto pipeline = NN_UNWRAP(nn::RLTrainingPipeline::create(
    cfg,
    std::make_unique<MyRolloutSource>(...),
    std::make_unique<nn::UniformBatchSampler>(),
    std::make_shared<MyMetrics>(),
    std::make_shared<MyCheckpointProvider>(),
    /*seed=*/42));

NN_UNWRAP(pipeline.run(
    1000,
    [&](std::span<const nn::TensorRecord* const> batch,
        nn::DeterministicRng& rng,
        nn::MetricSink& metrics) -> nn::Result<void> {
        // PPO/GRPO 组件 + TrainingSession/TrainingGroupSession 更新
        // metrics.emit({...});
        return {};
    }));
```

固定顺序是 `rollout → buffer → sample → update`。sampler 返回的 batch size 和
索引都会校验。`DeterministicRng` 状态及 update/rollout 计数可保存与恢复；
custom section 直接通过 `CheckpointProvider` 读写。通用 pipeline 不规定
buffer 和 replay 的文件布局。

`UniformBatchSampler` 支持有/无放回采样。消费者可替换为 prioritized、
on-policy epoch 或自定义 sampler。

## 11. 测试与 AOT

门禁：

| 测试 | 覆盖 |
|---|---|
| `tensor_schema_test` | schema/handle/具名张量/梯度包 |
| `rl_output_test` | named/packed 等价、共享 trunk 梯度、ModelSpec 往返 |
| `rl_experience_test` | GAE/return/mask/group normalization |
| `rl_policy_test` | categorical、REINFORCE、组件、有限差分、自定义分布 |
| `rl_ppo_test` | clip 三分支、value clip、preset 等价 |
| `rl_grpo_test` | 多组、零方差、preset 等价 |
| `training_group_session_test` | 多 role 生命周期/引擎校验/move |
| `rl_pipeline_test` | source/buffer/sampler/metrics/checkpoint/hooks/seed |
| `rl_training_integration_test` | PPO named gradients → TrainingSession optimizer update |
| `rl_gpu_smoke_test` | Vulkan AOT 闭合世界 |

新增 RL 内联表达式时，`tools/scan_exprs.cpp` 必须触达对应运行路径；GPU 未登记
结构会在执行时返回带 key 的闭合世界错误。

## 12. 当前边界

- 环境 `reset/step`、MCTS、棋规、自对局 worker 不在库内；
- 首版 policy 预设是 categorical；连续/多离散可通过 `PolicyDistribution` 扩展；
- 通用 pipeline 不强制 replay/RNG/checkpoint 的单一文件格式；
- 不新增引擎原语或 shader；objective 使用 DSL 组合；
- f16 和性能优化不是本功能的前置条件。
