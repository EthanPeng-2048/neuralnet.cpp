# 31 — RL 训练原语与可组合训练基础设施

> **读者**：项目开发者（本仓内部设计文档）。
> **状态**：✅ **已实施（2026-10-23，分支 `dev/rl_training`）**。
> **需求来源**：[GitHub issue #23](https://github.com/EthanPeng-2048/neuralnet.cpp/issues/23)。
> **用户裁定（2026-10-23）**：当前主线切到 RL；f16 与训练性能项继续后移，不作为 RL 的前置。
> **2026-10-23 库化修正**：issue #23 是能力**下界**，不是范围上界。公共 API 不得以
> `TextTrainer`、go.neuralnet.cpp 或任何已知内部调用方作为设计边界；未知库消费者需要时，
> 应能复用低层组件、替换算法环节或增加自定义字段。
> **执行排序**见 [`27-execution-plan.md`](27-execution-plan.md) §2；裁定台账见
> [`18-roadmap.md`](18-roadmap.md) §4.6。

## 1. 定位与设计判断

本专项不只实现“策略/价值双头 + PPO/GRPO”，而是补齐可复用的训练基础设施：

```text
基础契约
  具名张量 / schema / gradient bundle / checkpoint section / metrics
       ↓
训练核心
  TrainingSession（单模型单优化器）
  TrainingGroupSession（多模型、多优化器、具名角色）
       ↓
RL 组件
  experience record / advantage transform / policy distribution
  policy-objective components / standard REINFORCE-PPO-GRPO presets
       ↓
编排层
  rollout source / buffer / sampler / pipeline / trainer hooks
```

设计原则：

1. **公共语义通用**：核心头文件不出现围棋、MCTS、文本、token 或 `TextTrainer` 语义。
2. **低层可组合、高层有预设**：消费者可用单个组件，也可使用 PPO/GRPO 预置流水线。
3. **语义名优先，packing 只是编码**：多输出的公共语义是具名张量端口；行带打包只作为
   兼容现有单 `Tensor` ABI 的适配器。
4. **数据上下文 schema 化**：不把算法上下文塞进含义不明的 `target`，也不把 RL 数据写死成
   只适用于 PPO 的封闭结构。
5. **状态按所有者拆分**：optimizer/epsilon 状态、运行态、rollout/RNG、checkpoint I/O、
   数据和算法配置分别归属，不引入 god object。
6. **热路径无名称查找**：名称只在 setup/反序列化期解析；训练循环使用已解析的 field handle。
7. **复用现有计算栈**：RL 目标继续由 `Loss` + DSL 组合；不新增 ComputeEngine 原语或 shader。

### 1.1 已冻结决策

| 编号 | 决策 |
|---|---|
| D1 | issue #23 是能力下界；公共 API 以未知库消费者为前提，不围绕 `TextTrainer` 或 go.neuralnet.cpp 设计 |
| D2 | 多输出的公共语义使用 `TensorSchema` / `NamedTensorSet`；`RowBandPacking` 只是兼容单 Tensor ABI 的优化适配器 |
| D3 | RL 上下文使用 schema 化 `TensorRecord`，`PolicyBatchView` 只是标准字段视图，不是封闭数据模型 |
| D4 | 不新增核心 `RLTrainingSession`；多模型/多优化器由通用 `TrainingGroupSession` 管理，算法和 rollout 留在编排层 |
| D5 | REINFORCE/PPO/GRPO 由 distribution、advantage transform 和低层 objective components 组合；标准 preset 不限制自定义组合 |
| D6 | checkpoint、metrics、buffer、sampler、rollout source 均提供可替换接口；核心不强制统一 replay/RNG 文件布局 |
| D7 | RL-2 不是“可选能力”；其接口面在本专项内冻结，但实现可与 RL-0/1 分批交付 |

## 2. 需求清单（issue #23 是最小集合）

### RL-0 — 多输出模型协议（必需）

- 任意数量的具名输出端口，端口可有不同形状；
- 公共 API 能直接取得具名张量，不要求调用方先约定私有行区间；
- `forward/backward` 的端口与梯度一一对应；
- 提供可选 row-band packing，使现有单 `Tensor` ABI、优化器和 DSL 路径可复用；
- `ModelSpec` 能恢复并校验输出 schema；
- `PrecisionProfile` 继续逐子层转发；
- 策略/价值双头、辅助分类头、GAN 多输出等都能使用同一协议。

### RL-1 — 经验处理与策略目标组件（必需）

issue 要求的最小算法集合：

- REINFORCE（可选 baseline）；
- PPO-clip（policy clip + value clip）；
- GRPO；
- GAE、return/advantage normalization；
- episode return、clip fraction、explained variance、entropy、approx KL 等指标。

公共设计额外要求：

- experience 使用可扩展 schema，可附加消费者自定义字段；
- advantage 估计、normalization、policy distribution、policy/value/entropy/KL 目标可独立替换；
- REINFORCE/PPO/GRPO 是组件预设，不是不可绕过的单体实现；
- 首版可只实现离散 categorical action，但数据与 objective 接口不得封死连续/多离散动作。

### RL-2 — 通用编排与多模型会话（规划项）

- `TrainingGroupSession` 管理 actor/critic/reference 等具名模型角色；
- rollout source、experience buffer、batch sampler、metrics sink、checkpoint provider 可替换；
- 支持 on-policy rollout 与离线 replay 的接入，不把 PPO 循环写死在 Session；
- 固定 seed 可复现，状态可按 section 恢复；
- 环境、MCTS、棋规、自对局 worker 仍留在消费者。

实现可以分批，但上述公共扩展面应在 RL-0/1 定型时一起冻结，避免后续破坏性重构。

## 3. 公共基础契约

### 3.1 具名张量与 schema

```cpp
struct NamedTensor
{
    std::string name;
    Tensor value;
};

class NamedTensorSet
{
public:
    [[nodiscard]] const Tensor* find(std::string_view name) const;
    [[nodiscard]] Result<TensorRef> require(FieldHandle field);
    // 有序、可遍历；move-only 或严格控制拷贝
};
```

`TensorSchema` 是有序字段描述：

- 名称；
- 必需 / 可选；
- 形状及 batch/step 轴语义；
- precision / device 约束；
- 可选 packing 编码。

第一阶段采用有序字段，不实现任意 DAG：

- schema 名称必须唯一且稳定；
- setup 期把名称解析为 `FieldHandle`；
- 热路径用 handle 下标，不查字符串 map；
- 缺失必需字段、重复字段、形状不兼容均返回可定位的 `Result` 错误。

`NamedTensorSet` 用于输出、experience、metrics 和 checkpoint 扩展 section，避免为每一类上下文
重复发明封闭结构。

### 3.2 输出 schema 与打包适配

`OutputSchema` 是 `TensorSchema` 的输出侧特化。公共语义是：

```text
policy: (A, T)
value : (1, T)
aux   : (K, T)
```

每一端口保留独立名称、shape 和 precision。`GradientBundle` 使用同一 schema 表达各端口梯度。

为避免现有 `Tensor -> Tensor` 栈被整体改写成多输出图，提供可选：

```text
RowBandPacking
  policy: rows [0, A)
  value : rows [A, A + 1)
  packed: (A + 1, T)
```

约束：

1. 有序、互不重叠、无空洞并覆盖全部打包行；
2. 名称在 schema 初始化/序列化期唯一；
3. packing 与 schema 在构造期双向校验；
4. 现有单输出模型可显式登记为单端口 schema；
5. 只有消费者主动选择 packing 时，才生成单张量兼容表示。

计划落点：

- `OutputBundleLayer`：同一输入 fan-out 到若干输出分支；named backward 按端口分发梯度，
  再把各分支对输入的梯度求和；
- 实现现有 `Layer::forward/backward` 时通过 `RowBandPacking` 兼容；
- `parameters()` / `param_children()` / `extra_state()` 继续遵守扁平参数顺序；
- 精度、training mode、checkpoint、offload 等能力向子层转发；
- `ModelSpec` 增加**可选** `output_schema` 与 packing 描述；缺键表示旧单输出模型。

第一版不要求任意多阶段 DAG；但输出端口协议、gradient bundle 和序列化格式不能与“双头”
或“固定行区间”绑定。

### 3.3 Experience schema

RL 数据按“可扩展记录 + 标准视图”表达：

```text
ExperienceSchema
  standard fields:
    action, old_log_prob, advantage, return, old_value
    action_mask, value_mask, reward, done, group_id
  custom fields:
    consumer-defined names
```

底层：

```cpp
class TensorRecord;       // NamedTensorSet + schema
class PolicyBatchView;    // 对标准字段的强类型只读视图
```

算法在 create 期声明依赖字段，例如：

```text
REINFORCE: action, advantage, action_mask
PPO:       action, old_log_prob, advantage, return, action_mask
PPO+value: 追加 old_value, value_mask
GRPO:      追加 group_id / group_spans
```

约定：

- 多环境、多 token、`(batch, seq)` 统一按规范顺序压平成 `T`；不要求 Tensor 增加 N 维；
- 所有数值字段驻留设备侧，group span 等纯结构元数据可留在宿主侧；
- mask 显式传入，不把缺失 mask 隐式解释为全 1；
- `action` 的编码由 `PolicyDistribution` 解释；
- 自定义字段不要求核心 loss 理解，可由消费者自己的 objective/transform 读取。

## 4. 算法组件

### 4.1 Advantage 与 normalization

独立、无优化器状态的 transform：

- `GaeAdvantage`；
- `MonteCarloReturn`；
- `GroupAdvantage`（GRPO）；
- return/advantage normalization；
- mask、done、group-span 校验。

GAE 的逆序循环由宿主控制调度，但每步计算与状态保留在设备侧，不逐 step 下载标量。实现继续使用
现有引擎/DSL 原语；如果以后需要融合扫描，另行优化，不改变 API。

### 4.2 Policy distribution

公共扩展点：

```text
PolicyDistribution
  sample()
  log_prob()
  entropy()
  distribution-specific parameter schema
```

首批预设：

- categorical；
- 后续可加 diagonal Gaussian、multi-discrete 或消费者自定义分布。

PPO/GRPO 的目标只依赖 `log_prob_new`、`log_prob_old`、advantage 和 mask，因此不会因 action kind
变化而复制整套算法。

### 4.3 Objective components 与标准预设

不建议只提供三个单体类。公共能力应包含：

```text
PolicySurrogate
ClippedPolicySurrogate
ValueRegression
ClippedValueLoss
EntropyBonus
KLRegularizer
MaskedReducer
```

标准 preset 由这些组件组合：

```text
make_reinforce_objective(...)
make_ppo_objective(...)
make_grpo_objective(...)
```

约束：

- objective 继续派生/适配现有 `Loss` 协议；
- 需要 experience 上下文的入口必须强类型接收 `TensorRecord` / `PolicyBatchView`；
- 基类 `Loss::forward(pred,target)` 不得靠魔法字段名猜 RL 上下文；不满足上下文时明确报错；
- loss component 只产生该输出端口的梯度，`GradientBundle` 负责组合；
- 新增算法不需要修改 `TrainingSession` 或 ComputeEngine。

### 4.4 数学与数值契约

记号：`π_new` / `π_old` 为动作概率，`r = π_new(a|s) / π_old(a|s)`，
`A` 为 advantage，`R` 为 return，`V` 为 value prediction。

```text
REINFORCE:     L_pi = -Σ log π(a|s) · A
PPO policy:    L_pi = -mean(min(rA, clip(r, 1-ε, 1+ε)A))
PPO value:     L_v  = mean(max((V-R)^2, (clip(V,V_old-εv,V_old+εv)-R)^2))
GRPO:          A_i  = (R_i - mean(R_group)) / (std(R_group) + eps)
GAE:           δ_t  = r_t + γV_{t+1} - V_t
               A_t  = δ_t + γλA_{t+1}
```

约束：

- log-prob 使用数值稳定的 log-softmax（categorical）或分布的稳定实现；
- clip 三分支、value clip 两侧均须有解析值测试；
- ratio 使用 `exp(logπ_new - logπ_old)`，非有限值必须可诊断；
- return normalization 与 GRPO 使用总体标准差；方差为 0 时输出 0，不产生 NaN；
- `done=1` 清空 GAE bootstrap；mask 外时间步不参与递推；
- loss reduction 默认保持在现有 stable 精度；
- f16 不是前置；先以 f32 完成正确性，再补 `--f16` smoke。

## 5. Session 与编排层

### 5.1 `TrainingSession`

保持当前职责不变：

- 拥有一个 Model + 一个 Optimizer；
- 执行 micro-step、梯度累积、global normalization、裁剪和 update；
- 不识别文本、RL、rollout、advantage 或环境。

RL-0/1 的单 actor 或共享 trunk + policy/value 模型直接使用它，不需要 RL 专用子类。

### 5.2 `TrainingGroupSession`

新增通用多模型生命周期容器：

```cpp
struct ModelRoleSlot
{
    std::string name;                    // actor / critic / reference / custom
    std::variant<TrainingSession, Model>  // 可训练或只读，具体 API 实现期定
};

class TrainingGroupSession;
```

职责仅限：

- 以稳定名称拥有多个可训练 Session 或只读 Model；
- 保证各 role 的模型、optimizer 和引擎生命周期正确；
- 在创建期校验同一 role 的 engine/schema 兼容性；
- 提供 role 查询，不包含算法和 rollout。

它可服务 RL、actor-critic、GAN、ensemble、多智能体等场景。**`RLTrainingSession` 不是核心类型**；
如果以后为了易用性提供该名字，它只能是调用 `TrainingGroupSession` + RL 标准配置的 convenience
factory，不拥有独立语义。

### 5.3 RL 编排

计划提供可替换流水线：

```text
RolloutSource
  → Trajectory transforms
  → AdvantageEstimator
  → ExperienceBuffer
  → BatchSampler
  → Objective
  → TrainingSession / TrainingGroupSession
```

接口草案：

```text
RolloutSource
BatchSampler
MetricSink
CheckpointProvider
TrainerHooks
```

`RLTrainer` / `RLTrainingPipeline` 是上述组件的标准装配；消费者可以绕过它，直接调用低层接口。
环境、MCTS、自对局、规则判断和领域 buffer policy 均不在库内实现。

### 5.4 Metrics 与 checkpoint

指标使用通用 `MetricRecord`，由 `MetricSink` 消费。建议提供标准 RL 字段，但允许自定义字段：

```text
policy_loss, value_loss, entropy, approx_kl,
clip_fraction, explained_variance, episode_return
```

指标不能改变梯度。设备侧累计，在 update 边界统一回读。

checkpoint 采用可扩展 section：

```text
CheckpointEnvelope
  ├─ model/output/optimizer/training sections
  ├─ RL state section
  │    算法配置 / rollout cursor / RNG / replay metadata
  └─ custom sections
```

核心提供最小模型/optimizer 状态；RL 和消费者状态按 section 注册。不得要求所有消费者接受同一
套 buffer 或 RNG 格式。

## 6. 分阶段验收

### RL-0 — 输出基础设施

1. `TensorSchema` / `NamedTensorSet` / `GradientBundle` 的名称、顺序、缺失字段和 shape 错误测试；
2. 双输出模型通过 named API 取得 `policy (A,T)` 与 `value (1,T)`；
3. named backward 只把梯度送到对应端口，共享 trunk 收到正确梯度和；
4. 同一模型经 `RowBandPacking` 产生 `(A+1,T)`，与 named 路径数值一致；
5. `CE(policy)+λ·MSE(value)` 组合梯度通过有限差分/解析梯度检查；
6. 非法 row band、重名、shape/schema 不匹配全部返回 `Result`；
7. `ModelSpec` 的 output schema/packing save-load 往返一致，缺键旧文件可载入；
8. f32 CPU 为硬门禁，GPU / f16 至少 smoke；三值不支持处初始化期明确报错。

### RL-1 — Experience 与 objectives

1. experience schema 的必需/可选/自定义字段、device/precision/shape 校验；
2. GAE 3–5 步手算、`done` 截断、bootstrap、mask 边界；
3. return/advantage/group normalization 的零方差行为；
4. REINFORCE 单样本、baseline、全 mask；
5. PPO ratio 三分支、value clip 两侧、缺失 `old_log_prob`/`old_value` 的明确错误；
6. GRPO 单组、多组、零方差组；
7. categorical log-prob/entropy 与有限差分；
8. 标准 preset 与手工组件组合在相同输入上数值一致；
9. 自定义 experience 字段可被自定义 objective 读取，核心代码零分支；
10. `scan_exprs` 覆盖所有运行期表达式结构；CPU 必跑，Vulkan 有 smoke；
11. 全部失败路径返回 `Result`，不依赖探针构建。

### RL-2 — 通用编排与多模型

1. `TrainingGroupSession` 可同时持有两个可训练 role 和一个只读 role，移动/析构安全；
2. 非法 role 名称、重复角色、不同引擎/schema 的不兼容组合明确报错；
3. rollout → transform → buffer → sampler → update 的样本数/update 数一致；
4. 固定 seed 下采样、rollout 和 PPO epoch 顺序可复现；
5. checkpoint 可分别恢复 optimizer、rollout cursor、RNG 和自定义 section；
6. 不改变 `TrainingSession` 的 micro/global normalization 语义；
7. 不引入环境、MCTS 或游戏规则依赖。

一期不以“loss 下降”作为 RL 正确性的唯一证据；解析值与梯度检查优先。

## 7. 预计落点

| 职责 | 预计落点 |
|---|---|
| Tensor schema / named set / gradient bundle | 计划新增通用 schema 头 |
| Output schema / row-band packing / bundle layer | 计划新增 output 系列头 + `compute_layer.hpp` 聚合 |
| ModelSpec 兼容键 | `model_spec.hpp` / `model_serialization.hpp` |
| experience schema / transforms / advantage | 计划新增 experience / advantage 头 |
| policy distribution / objectives / presets | 计划新增 policy 系列头 |
| `TrainingGroupSession` | 计划新增 group session 头 |
| RL pipeline / buffer / state | 计划新增 rl trainer / rl state 头 |
| 门禁 | schema/output + experience/objective + group/pipeline 三组测试，按 CPU/GPU 分变体 |

不把这些能力塞进 `TextTrainer`、`TextSampleSet` 或 `Dataset`。文本训练只是 `TrainingSession`
的一个高层使用者，不是公共训练架构。

## 8. 实施顺序与硬依赖

1. 冻结公共基础：schema、named tensor、gradient bundle、output schema 与 checkpoint section；
2. RL-0：named output + row-band adapter + bundle layer + ModelSpec；
3. RL-1a：experience schema、GAE/return/group transforms；
4. RL-1b：distribution + low-level objective components + REINFORCE preset；
5. RL-1c：PPO components/preset + value clip；
6. RL-1d：GRPO preset；
7. RL-2a：`TrainingGroupSession`；
8. RL-2b：buffer/sampler/pipeline、metrics sink、可扩展 checkpoint provider。

```text
公共基础 → RL-0 → RL-1a/b → RL-1c → RL-1d
                              └──────→ RL-2a → RL-2b
```

f16 / 训练性能与这条链无正确性依赖：可以后做，不能成为 RL 的前置。

## 9. 非目标

- MCTS、PUCT、游戏规则、局面编码、环境 `reset/step`、自对局 worker 池；
- 新后端、CUDA、分布式训练作为当前交付；
- 为某个 loss 新增 shader / 引擎虚函数；
- 任意动态图、自动微分框架或完整 DAG 调度器；
- 用单一 `RLTrainingSession` / `RLTrainer` 把模型、优化器、rollout、buffer 和 I/O 合成 god object；
- 强制所有消费者采用同一套 replay、RNG 或 checkpoint 布局。

协议应给上述能力留扩展点，但实现范围可以分批交付。

## 10. 下游适配示例：go.neuralnet.cpp

go.neuralnet.cpp 只作为消费者样例：

- 用 `OutputSchema` 声明 `policy` 与 `value`；
- 需要兼容旧单 Tensor 时启用 `RowBandPacking`，得到 `(362+1,T)`；
- AlphaZero 搜索目标用通用 `CE + λ·MSE` 多 objective 组合；
- 自对局 loop、MCTS、棋规和 replay policy 仍留在衍生项目；
- 下游 smoke 至少证明 named output、组合梯度和一次 optimizer update。

该示例不进入上游公共 API 命名，也不成为其他消费者的能力上限。

## 11. 实施结果（2026-10-23）

RL-0、RL-1a/b/c/d、RL-2a/b 已按依赖顺序落地：

| 阶段 | 代码落点 | 门禁 |
|---|---|---|
| 公共契约 | `tensor_schema.hpp`、`named_tensor_set.hpp` | `tensor_schema_test` |
| RL-0 | `compute_layer_bundle.hpp`、`Model::forward_named/backward_named`、ModelSpec schema/packing | `rl_output_test` |
| RL-1a | `rl_experience.hpp`：TensorRecord/PolicyBatchView/GAE/return/group normalization | `rl_experience_test` |
| RL-1b | `rl_policy.hpp`：PolicyDistribution、categorical、REINFORCE、value/entropy/KL 组件 | `rl_policy_test` |
| RL-1c | ClippedPolicy/ClippedValue/PpoObjective | `rl_ppo_test` + `rl_gpu_smoke_test` |
| RL-1d | GroupAdvantage + GrpoObjective | `rl_grpo_test` + `rl_gpu_smoke_test` |
| RL-2a | `training_group_session.hpp` | `training_group_session_test` |
| RL-2b | `rl_pipeline.hpp`：source/buffer/sampler/metrics/checkpoint/hooks | `rl_pipeline_test` |

实现边界与设计一致：没有新增引擎原语/shader；environment/MCTS/规则仍在消费者；
首版分布是 categorical，但 objective 依赖 `PolicyDistribution` 接口；通用 checkpoint
provider 只固定 pipeline 自身 RNG/计数 section，消费者 buffer/replay 可保存自定义 section。

调用文档：[`../usage/06-reinforcement-learning.md`](../usage/06-reinforcement-learning.md)。

### 11.1 全量验收（2026-10-23，`ethan@192.168.1.102`）

- Windows / clang 23.1 / MSVC STL / Debug；
- CPU-only（`NN_ENABLE_GPU=OFF`，tests ON）：**66/66**，其中 **45 pass + 21 skip**；
- Vulkan（40HX）：**68/68**，其中 **65 pass + 3 skip**；
- `scan_exprs`：库内 **158 条**融合表达式，RL 的 PPO clip/value、梯度选择和 reduction
  路径均已进入闭合世界；
- `rl_gpu_smoke_test` 覆盖 GAE、PPO preset、GRPO preset 的设备执行，通过。

skip 均为既有条件门禁（无 GPU 构建的 GPU 用例、缺失 CLI app 的发现类用例、以及
`scan_bitwise_test` 的平台口径），没有 RL 用例被跳过。
