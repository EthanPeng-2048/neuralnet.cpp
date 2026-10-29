# 32 — 领域无关训练骨架（RunDriver）与开放模型规格（ModelSpecBase）

> **读者**：项目开发者。
> **状态**：📐 RFC（待实施）。
> **关系**：本文修订 [`30-training-session-step.md`](30-training-session-step.md) 的层次边界，
> 在「单步 Session（`30`）」与「文本 Run Driver（`TextTrainer`）」之间补上缺失的
> **领域无关骨架**层，并把封闭枚举 `ModelType` 的规格扩展为开放多态。
> 涵盖 issue **#39**（Trainer 骨架）与 **#26**（ModelSpec 扩展点）。

## 0. 问题

训练栈的可复用层次目前是断层的：

| 层 | 载体 | 职责 | 状态 |
|---|---|---|---|
| 单步编排 | `TrainingSession`（`30`） | Model+Optimizer 同生共死、micro-step、梯度累积、global 归一化、裁剪、update | ✅ 完备 |
| **领域无关骨架** | **—** | epoch/step 循环、lr 接线、断点节奏、指标、评估节奏 | ❌ **空洞** |
| 文本域驱动 | `TextTrainer` | 上层 + 文本专属（滑窗、sparse-CE、非阻塞 loss 槽） | ✅ 但焊死文本 |
| 应用驱动 | `go_train_cli.cpp`（1456 行） | 重写了一遍"骨架" | ❌ 每个下游 1400 行起步 |

空洞层里的逻辑其实是**领域无关**的，只有数据源与指标是领域相关的。现在两处
（`TextTrainer::fit` 内部 + go CLI）各自维护这套逻辑，其中最危险的是**断点交叉校验**
（数据集身份、recipe 一致性、counter 越界）——这类"错了很难发现"的库口径逻辑
被下游手写，每重写一次就多一份偏离风险。

同时，模型规格是**封闭枚举**：衍生项目要加自定义模型必须改主仓 `ModelType`
并往 `spec_compatible` / `spec_summary` / `spec_to_kv` 三处 `switch` 加分支。
这与自定义训练域是同一枚硬币的两面。

## 1. 设计原则（裁决依据）

所有决策必须符合：**自由、模块化、代码复用、分层、统一**。

- **自由**：骨架不绑架域。核心不可关，但所有"能力"可关/可换，域可只用核心。
- **模块化**：每个扩展点是独立虚基类，域只注入差异。
- **复用**：库口径（断点/归一化/校验）只写一份；文本/MNIST 等域复用预置适配器。
- **分层**：公共在父、特化在子；上层只依赖下层的抽象，不反向。
- **统一**：四个扩展点用同一心智模型——「虚基类定义契约 + 主仓持有基类引用 + 子类注入特化」。

## 2. 架构总览

```
┌─ 应用（go_train_cli / mnist_cli / 自定义域）─────────────────┐
│    提供：TrainState 子类 + BatchSource + （可选）Spec 子类      │
└──────────────────────────────────────────────────────────┘
┌─ RunDriver（本 RFC 新增，领域无关骨架）──────────────────────┐
│    必选核心：epoch/step 循环 · lr 接线 · 断点时机 · 报告聚合      │
│    可选能力：checkpoint 节奏 · 指标落盘 · 评估（loss 回读归域）│
└──────────────────────────────────────────────────────────┘
┌─ TrainState 基类（本 RFC 新增）＋ ModelSpecBase（本 RFC 新增）┐
│    状态与规格的扩展契约 + 库口径编解码/校验骨架                │
└──────────────────────────────────────────────────────────┘
┌─ TrainingSession（`30`，不变）─────────────────────────────┐
│    单步执行 · 梯度累积 · global 归一化 · 裁剪 · update          │
└──────────────────────────────────────────────────────────┘
┌─ Model / Layer / Engine（不变）───────────────────────────┐
└──────────────────────────────────────────────────────────┘
```

四个领域扩展点，同一心智模型：

| 扩展对象 | 机制（虚基类 + 子类注入） |
|---|---|
| 训练状态域 | `TrainState` 虚基类 + `write_extra_kv` / `read_extra_kv` |
| 模型架构域 | `ModelSpecBase` 虚基类 + `write_kv` / `read_kv` / `build` |
| 数据供给域 | `BatchSource` 抽象 + `IndexedBatchSource` 预置适配器 |
| 训练编排域 | `RunDriver` 核心 + 可选能力（hook / 策略） |

## 3. 扩展点一：`TrainState`（状态域）

### 3.1 现状

`train_state.hpp` 的 `TrainState` 是单一静态结构，文本字段（`recipe.stride`、
`DatasetIdentity.num_docs/loss_scope`）直接焊在里面，没有类型化扩展槽位。

### 3.2 方案

`TrainState` 虚基类承载计数器/配方/优化器 + 唯一的编解码骨架（模板方法）；
子类只提供域特化字段的编解码钩子。

```cpp
namespace nn {

// 通用训练配方（去掉文本专属 stride；stride 下沉到 TextTrainState）
struct TrainRecipe {
    PrecisionProfile precision{};
    std::string optimizer = "adam";
    std::string lr_schedule = "fixed";   // fixed / cosine / multistep / step_cosine
    double lr = 0.0, min_lr = 0.0;
    double weight_decay = 0.0, beta1 = 0.9, beta2 = 0.999, max_norm = 0.0;
    std::uint64_t epochs = 0, total_steps = 0;
    std::uint64_t warmup_epochs = 0, warmup_steps = 0;
    std::uint64_t batch_size = 0, micro_batch_size = 0;
    bool batch_size_is_global = false;
    std::uint64_t accum_steps = 0;
    std::vector<double> lr_per_epoch;
};

struct TrainCounters {
    std::uint64_t global_step = 0, epoch = 0, step_in_epoch = 0;
    double lr = 0.0;
};

class TrainState {                       // 虚基类
public:
    virtual ~TrainState() = default;
    [[nodiscard]] virtual const char* state_kind() const noexcept = 0;  // "text"/"mnist"/…

    TrainCounters counters;
    TrainRecipe   recipe;
    std::uint64_t optimizer_step = 0;
    std::uint64_t optimizer_blob_count = 0;

    // 编解码骨架（模板方法；库口径唯一实现）
    [[nodiscard]] KeyValueRecord to_kv() const;         // 基类字段 + write_extra_kv
    [[nodiscard]] static Result<std::unique_ptr<TrainState>>
        from_kv(const KeyValueRecord& kv);              // 读 kind → 工厂构造子类 → read_extra_kv

protected:
    virtual void write_extra_kv(KeyValueRecord& kv) const {}   // 子类：域字段落盘
    virtual Result<void> read_extra_kv(const KeyValueRecord& kv) { return {}; }
};

} // namespace nn
```

### 3.3 文本子类（向后兼容）

```cpp
class TextTrainState : public TrainState {
public:
    [[nodiscard]] const char* state_kind() const noexcept override { return "text"; }
    std::uint64_t stride = 0;              // 从 recipe 下沉
    DatasetIdentity dataset;               // 保留在文本域
protected:
    void write_extra_kv(KeyValueRecord& kv) const override;   // stride + dataset 各键
    Result<void> read_extra_kv(const KeyValueRecord& kv) override;
};
```

### 3.4 兼容口径

- **仅向下兼容（新版读旧版），且不作承诺**——该能力本身是 **v4 模型格式引入 kvrec**
  所带来的特性（自描述键值 ⇒ 新版读旧版），不是本 RFC 额外新增的保证；
- 旧 `.nnckpt` 的 `state_kind` 键缺失 ⇒ `from_kv` 回退构造 `TextTrainState`；
- 旧 `recipe.stride` 键仍在 `TextTrainState::read_extra_kv` 里回退读取；
- 反向（旧读器读新文件）与跨版本逐位一致性均**不在承诺范围**。

### 3.5 断点交叉校验上收（库口径唯一）

go CLI 里那段"错了很难发现"的校验逻辑上收到基类：

```cpp
enum class ResumePolicy { Strict, Relaxed };

[[nodiscard]] Result<void> verify_resume(const TrainState& saved,
                                         const TrainState& current,
                                         ResumePolicy policy = ResumePolicy::Strict);
// Strict：kinds 相等、recipe 关键维度相等（optimizer/lr_schedule/epochs/
//         total_steps/warmup/batch/accum/precision）、counter 不越界；
//         再调用两态的 verify_extra()。
// Relaxed：只校验计数器单调性（研究期换数据集/改配方）。
```

子类补域校验：

```cpp
// TextTrainState override
Result<void> verify_extra(const TrainState& current) const;  // dataset.sha256/num_samples 一致
```

第三方域零重写即得库口径校验（前提：走基类）。

## 4. 扩展点二：`BatchSource`（数据供给域）

### 4.1 Batch 语义评估

| 方案 | 描述 | 自由 | 复用 | 结论 |
|---|---|---|---|---|
| **Pull（回调式）** | 域按 `step_hint` 供给一个 batch | **最高** | 高 | ✅ **基座** |
| **Indexed（索引式）** | 域持样本表，骨架按 `indices` 取 `make_batch` | 中 | 中 | ✅ 预置适配器 |
| Iterator（迭代器式） | 域给 `next_batch()` | 中 | 中 | ✗ 难表达 epoch 重置/断点定位 |
| 固定 epoch 网格 | 骨架强制 `steps_per_epoch` | 低 | 低 | ✗ 排除 |

### 4.2 `BatchSource`：Pull 契约（分层最深、最自由）

```cpp
// 一个训练步的产物（域无关；Batch 是不透明域类型，由域自己的 StepFn 消费）
struct BatchAt {
    std::size_t samples = 0;       // 本 batch 有效样本数（末批可能不满）
    std::size_t units = 0;         // 归一化单位（文本=有效 token；视觉=样本数）
    bool epoch_done = true;        // 本 batch 后是否结束当前 epoch
};

class BatchSource {                  // 领域无关：只有顺序，没有索引/shuffle 概念
public:
    virtual ~BatchSource() = default;
    // 供给第 step_hint 个 batch（域自己决定来源：数据表 / 在线采集 / rollout）
    virtual Result<BatchAt> next(std::size_t step_hint) = 0;
    // 可选：无固定 epoch 的域返回 nullopt（骨架退化为步进驱动）
    [[nodiscard]] virtual std::optional<std::size_t> steps_per_epoch() const { return std::nullopt; }
    // 断点定位：域把内部游标移到 (epoch, start_step)；无固定 epoch 的域可空实现
    virtual Result<void> rewind(std::size_t epoch, std::size_t start_step) { return {}; }
};

class IndexedBatchSource : public BatchSource {   // 预置适配器：文本/MNIST 复用
public:
    // 持样本表 + 每 epoch shuffle + make_batch 装配；steps_per_epoch 固定
    // ...
};
```

**关键裁决**：`steps_per_epoch` 可选。有值 ⇒ 骨架驱动 epoch 网格 + epoch 级 lr；
`nullopt` ⇒ 骨架退化为步进驱动（lr 走 step 级），域自行决定何时算一轮。
同一骨架覆盖文本（固定 epoch）与 RL/在线（无固定 epoch）。

### 4.3 领域无关 Batch 载体

`BatchAt` 只带元数据；真正的张量载体由域定义（文本是 `TextSampleBatch`，
视觉是图像+标签）。骨架不解释张量内容，只透传给域自己的 `StepFn`：

```cpp
// 骨架持一个类型擦除的 batch 载体（域自解释）
using AnyBatch = std::shared_ptr<void>;   // 或 std::any；域 StepFn 自行 static_cast
```

文本路径用 `IndexedTextBatchSource : BatchSource`，内部产出 `TextSampleBatch`，
塞进 `AnyBatch`；`TextTrainer` 的 StepFn 取回原类型。零拷贝、零语义泄漏。

## 5. 扩展点三：`RunDriver`（训练编排域）

### 5.1 必选核心 vs 可选能力（解决"内置影响自由度"）

| 类别 | 内容 | 可关？ |
|---|---|---|
| **必选核心** | epoch/step 计数、断点时机（epoch 边界归一化）、lr 调度接线（`compute_epoch_lr`/`compute_step_lr`）、`train_step` 调用、报告聚合 | 否（保证库口径统一） |
| **可选能力** | checkpoint 节奏、指标落盘、评估循环 | **是**（每个可关/可换/默认开） |

非阻塞 loss 回读槽**归域**（`do_update` 内：域拥有 batch 与 `train_step`，自然拥有
loss 产出与回读时机），driver 不内置——这是实现与 §5.2 草案的差异，以本条为准。

### 5.2 接口草案

```cpp
namespace nn {

// 一次训练的结果（域无关）
struct RunReport {
    Scalar seconds = 0;
    std::uint64_t elapsed_ns = 0;
    std::size_t steps_executed = 0, samples_executed = 0;
    std::size_t ckpt_failures = 0;
    int abort_code = 0;
    Scalar final_lr = 0;
    std::size_t final_epoch = 0, final_step_in_epoch = 0;
};

struct RunDriverConfig {
    int epochs = 1;              // epochs<=0 = 步进驱动（无固定 epoch）
    std::size_t max_steps = 0;   // 0 = 按 epoch 跑完
    int stop_after_epoch = 0;    // >0 = 训到该 epoch（1-based，不含）存档退出
    std::string lr_schedule = "fixed";
    Scalar lr = 0, min_lr = 0;
    int warmup_epochs = 0;
    std::size_t warmup_steps = 0;
    std::vector<Scalar> lr_per_epoch;
    std::size_t log_interval = 50, save_interval = 100;
    // 可选能力开关（默认全开）
    bool enable_loss_readback = true;   // 关 = 同步 loss（简单域）
    bool enable_checkpoint   = true;
    bool enable_metrics      = true;
    bool enable_eval         = true;
};

// 领域注入点（全部可空 = 对应能力不启用；空 = 该能力不启用）
struct RunHooks {
    // 必选：域的单步前向/损失/反向（文本 = sparse-CE，视觉 = CE/...）
    std::function<Result<TrainingStepOutput>(
        TrainingSession&, ComputeEngine&, const AnyBatch&)> step_fn;
    // 可选：评估（一个 epoch 结束时）
    std::function<Result<Scalar>(TrainingSession&, std::size_t epoch)> evaluate;
    // 可选：指标/落盘
    std::function<void(const RunReport&, std::size_t epoch)> on_epoch_end;
    // 可选：断点请求（域自己落地路径/格式）
    std::function<Result<void>(TrainingSession&, const TrainState&)> on_checkpoint;
    // 可选：诊断探针（batch 装配后 / 更新后 / 相位）
    std::function<void(TrainingSession&, std::size_t epoch, std::size_t step)> on_update;
    std::function<bool(TrainingSession&, std::size_t step)> on_probe_check;  // true=中止
    std::function<void(const char*)> on_phase;
};

class RunDriver {                  // 领域无关骨架
public:
    [[nodiscard]] static Result<RunDriver> create(
        TrainingSession& session,
        BatchSource& source,
        std::unique_ptr<TrainState> state,
        RunDriverConfig cfg);

    // 注入钩子（生命周期：活到下一次 fit 返回）
    void set_hooks(RunHooks hooks);

    // 驱动：epoch×step 循环 + lr 接线 + 断点节奏 + 报告
    [[nodiscard]] Result<RunReport> fit();

private:
    TrainingSession& session_;
    BatchSource& source_;
    std::unique_ptr<TrainState> state_;
    RunDriverConfig cfg_;
    RunHooks hooks_;
};

} // namespace nn
```

### 5.3 与 `TrainingSession` 的边界（`30` 不变）

`RunDriver` 只做编排，不碰梯度/归一化/裁剪——那些仍是 `TrainingSession` 的职责。
`RunDriver::fit` 内层对每个 global batch：

1. `source_.next(step_hint)` → `BatchAt`（含 units）；
2. 按 `recipe.accum_steps` 切 micro-batch，逐个 `session_.train_step(...)`，
   `normalization_units = BatchAt::units`；
3. 非阻塞 loss 回读槽（可选能力）收割 + 日志；
4. epoch 边界：lr 接线、评估（可选）、断点（可选）、指标（可选）、状态归一化。

## 6. 扩展点四：`ModelSpecBase`（模型架构域，#26）

### 6.1 现状

`ModelType` 是封闭枚举，`spec_compatible` / `spec_summary` / `spec_to_kv`
三处 `switch(type)` 是封闭世界硬约束。

### 6.2 方案：组合多态（判别符随多态对象走）

`type` **不在** `ModelSpec` 外壳，而在 `ModelSpecBase`；外壳只持域无关公共字段。

```cpp
namespace nn {

struct ModelSpecBase {                  // 类型参数 + 判别符 + 分发虚方法
    virtual ~ModelSpecBase() = default;
    [[nodiscard]] virtual const char* type_name() const noexcept = 0;   // 字符串判别符
    virtual bool arch_compatible(const ModelSpecBase& other) const = 0; // 替换 switch
    [[nodiscard]] virtual std::string summary() const = 0;              // 替换 switch
    virtual void write_kv(KeyValueRecord& kv) const = 0;                // 类型字段落盘
    virtual Result<void> read_kv(const KeyValueRecord& kv) = 0;
    // 类型参数→模型的转换内聚到子类（GPT 自己知道怎么造自己）
    [[nodiscard]] virtual Result<Model> build(ComputeEngine& engine) const = 0;
};

struct ModelSpec {                      // 外壳：域无关公共字段
    std::optional<TensorSchema> output_schema;   // 通用多输出，不属于任何类型
    std::unique_ptr<ModelSpecBase> arch;         // 判别符 + 类型参数随多态走
};

// 各类型子类（判别符内聚；类型参数从外壳下沉到子类）
struct GptSpec : ModelSpecBase {
    [[nodiscard]] const char* type_name() const noexcept override { return "gpt"; }
    std::size_t vocab_size, d_model, seq_len, num_heads, d_ff, num_layers;
    PosEncodingType pos_encoding; ActivationType activation; NormType norm_type;
    bool subln; std::size_t n_head_kv; bool tie_embeddings;
    // arch_compatible / summary / write_kv / read_kv / build 全部内聚
    [[nodiscard]] Result<Model> build(ComputeEngine& e) const override;   // → build_gpt_model
};
struct CnnSpec  : ModelSpecBase { /* ... */ };
struct MlpSpec  : ModelSpecBase { /* ... */ };
struct RaptSpec : ModelSpecBase { /* ... */ };
struct TransformerSpec : ModelSpecBase { /* ... */ };

} // namespace nn
```

### 6.3 开放性：内建工厂表 + registry 合流

- **判别符 = 字符串**（`type_name()`），`ModelType` 枚举**不再进** spec；
- **内建类型**：主仓持编译期已知的 `builtin_specs()` 工厂表
  （`"gpt"→make GptSpec`，`"cnn"→make CnnSpec`，…）；
- **自定义类型**：查内建表未命中 ⇒ 查 `ModelRegistry` 注册的 builder；
  都没有 ⇒ 明确报"未知模型类型"。内建/自定义在"查表构造子类"这一步合流；
- 外壳 `ModelSpec` 完全无感判别逻辑。

```cpp
class ModelRegistry {                   // 自定义模型的子类工厂解析器（可选便利层）
public:
    static ModelRegistry& instance();
    using Factory = std::function<std::unique_ptr<ModelSpecBase>()>;
    void register_type(std::string type_name, Factory f);
    [[nodiscard]] std::unique_ptr<ModelSpecBase> make(const std::string& type_name) const;
};
```

### 6.4 三处 switch 的命运

| 现有 switch | 去向 |
|---|---|
| `spec_compatible` 类型分发 | `arch->arch_compatible()`（虚） |
| `spec_summary` 类型分发 | `arch->summary()`（虚） |
| `spec_to_kv` / `kv_to_spec` 字段循环 | `arch->write_kv/read_kv`（虚）+ 工厂表 |

### 6.5 文件头兼容（仅向下，不承诺）

- `type` 字段从**数值枚举改为字符串类型名**；
- 向下兼容走数值回落：`kv_to_spec` 检测到数值（`3`=GPT 等）时用一张
  **旧枚举→类型名**只读映射表还原（内建类型全部可逆）——同样是 v4/kvrec
  自描述特性的既有口径：只保证新读旧，反向不承诺；
- `is_gpt()`/`is_cnn()` 等判别调用点改为 `type_name()` 比较或
  `dynamic_cast<GptSpec*>`（热路径用 `type_name()`，诊断用 cast）。

## 7. 迁移路径

### 阶段 A：`TrainState` → 虚基类（先做，独立可测）
1. 提虚基类 + `TextTrainState` 子类；`TextTrainState` = 现有 `TrainState` 的全部字段；
2. `to_kv`/`from_kv` 骨架 + 旧文件回退（缺 `state_kind` ⇒ text）；
3. 上收 `verify_resume` + `TextTrainState::verify_extra`；
4. 全部现有 `TrainState` 调用点改为 `TrainState&`/`TextTrainState`；断点逐位回归。

### 阶段 B：`BatchSource` + `IndexedTextBatchSource`
1. 定义 `BatchSource`/`BatchAt`/`AnyBatch`；
2. `IndexedTextBatchSource : BatchSource` 包装 `TextSampleSet`（shuffle/装配/units）；
3. 现有 `TextSampleSet::make_batch` 路径不变，只加一层适配。

### 阶段 C：`RunDriver` 骨架 + `TextTrainer` 收敛
1. 实现 `RunDriver::fit`（核心 + 可选能力）；
2. `TextTrainer::fit` 改写为「`RunDriver` + 文本钩子」的实例（向后兼容 + dogfood）；
3. go CLI 收敛到 ~200 行（提供 `TextTrainState` 子类 + `IndexedTextBatchSource` + 钩子）。

### 阶段 D：`ModelSpecBase` 开放规格（#26，可与 A–C 并行）
1. 提 `ModelSpecBase` + `ModelSpec` 外壳 + 内建子类（Gpt/Cnn/Mlp/Rapt/Transformer）；
2. 内建工厂表 + `ModelRegistry`；
3. `spec_to_kv`/`kv_to_spec`/`spec_compatible`/`spec_summary` 虚化；
4. 文件头 `type` 字符串化 + 旧枚举映射回退；`build_*_from_spec` 收敛到子类 `build`；
5. 模型文件逐位回归 + 旧文件加载回归。

## 8. 验收标准

- [ ] 自定义域只写 `TrainState` 子类 + `BatchSource` + `StepFn` 即可训练，不碰骨架内部；
- [ ] `TextTrainer` 行为逐位不变（可观测锚：进度文案、seed 语义、断点计数器语义）；
- [ ] go CLI 从 1456 行收敛到 ~200 行，且断点交叉校验零手写（走 `verify_resume`）；
- [ ] 加一个自定义模型不改主仓任何 `switch`（子类 + 注册即可）；
- [ ] 新版读旧 `.nnckpt` / 模型文件可读（向下；源自 v4/kvrec，不作额外承诺）；
- [ ] `RunDriver` 关闭全部可选能力后 = 纯 epoch×step 循环（高自由度验证）；
- [ ] 现有 ctest 68/68 保持绿 + 新增骨架/Spec 单测。

## 9. 版本口径

- `dev` **从不定档版本号**：版本号只在本工作经审核合入 `main` 时确定（当前最新版 `v1.13.0`）；
- 届时按 `v1.<API 变化>.<漏洞修复>` 判定档位：`TrainState` 虚基类 / `RunDriver` /
  `BatchSource` / `ModelSpecBase` 属 API 变化面，定档由审核裁定，本稿不预判具体号；
- `TextTrainer` 内部改写、文件头 `type` 字符串化 ⇒ 行为不变，不构成 API 变化面。

## 10. 拍板记录

1. ✅ `AnyBatch` 载体 = `std::any`（指针形式读取，`-fno-exceptions` 安全）；
2. ✅ 步进驱动与 epoch 驱动共存（`epochs<=0` + `max_steps>0` = 步进驱动）；
3. ✅ `ModelSpecBase::build` **不进基类**——裁定：build 是域层职责（`build_*_from_spec`
   自由函数），理由是分层而非成本：`model_spec.hpp` 是纯数据头（无 L2 依赖），
   而 build 需要外壳公共字段（output_schema/packing）+ 运行期 precision + Model 类型；
   且仓内不存在需要收敛的中央 build 分发 switch（加载点全是显式按类型调用）。
   `ModelSpecBase` 只收敛真正的封闭点：`arch_compatible` / `summary` / `write_kv` /
   `read_kv` 四个分发 + `make_spec` 工厂（含 ModelRegistry 自定义注册）；
4. ✅ `verify_resume` 不校验 `lr`/`min_lr`/`weight_decay`/`beta*`/`lr_per_epoch`
   （只校验曲线形状类字段；比 go CLI 口径宽，接受）；
5. ✅ `fit` 吞掉 `evaluate` 错误（与"评估失败不中止训练"一致；留痕由域钩子负责）。
