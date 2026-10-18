# 对象生命周期契约：引擎与借用方

> 适用范围：`ComputeEngine` 与 `Layer` / `Model` / `Tensor` / `ExprGraph` /
> `ExprBlock` 之间的借用关系。实现见
> [`core_engine_lease.hpp`](../../include/neuralnet.cpp/core_engine_lease.hpp)。

## 1. 契约

**引擎必须比所有借用它的对象活得久。** 这是所有权设计（引擎由调用方拥有，
层/模型/张量只借用），不是实现细节。

与旧实现的关键区别：违反契约时**不会**解引用悬垂指针。借用方在第一次访问
引擎时终止，并打印：

- 借用方名字（`Linear` / `Model` / `Tensor` / `dsl::end_expr` / ...）；
- 失败原因（未绑定 / init 失败 / 引擎已销毁）；
- 调用点（`file:line`）。

示例：

```text
[engine-lease] Linear: 绑定的引擎已销毁——engine 必须比借用方活得久 at include/neuralnet.cpp/compute_layer_mlp.hpp:109
```

正确顺序（引擎最后销毁）：

```cpp
nn::CpuEngine engine;          // 先创建
nn::Model model(engine);       // 借用
model.add<nn::Linear>(4, 3);
// ... 训练/推理 ...
// model 先于 engine 析构（逆声明序）
```

CLI 的 `create_engine()` 返回 `unique_ptr<ComputeEngine>`，调用方应保证它在
模型/层/张量之后销毁；`--gpu` 路径的 `GpuEngine` 还借用进程级
`GpuBackend::instance()`（后者故意泄漏，天然满足）。

## 2. 类型

### 2.1 `EngineLiveness`（存活令牌）

- 与 `ComputeEngine` **分开分配**；引擎析构时只把 `alive` 置 false；
- 令牌内存仍然有效，因此“引擎已销毁”是可安全判定的；
- 每引擎一个，`ComputeEngine` 构造时从进程级令牌池取得。

### 2.2 `EngineLease`（借用句柄，8 字节）

```cpp
[[nodiscard]] bool bound()   const noexcept;  // 是否曾绑定
[[nodiscard]] bool alive()   const noexcept;  // 绑定且引擎存活
[[nodiscard]] bool expired() const noexcept;  // 绑定过但引擎已销毁
[[nodiscard]] ComputeEngine* get() const noexcept;   // 不安全时 nullptr
[[nodiscard]] const EngineLiveness* identity() const noexcept; // 同一引擎判等
[[nodiscard]] ComputeEngine& require(std::string_view owner,
                                     std::source_location = current()) const;
```

约束（由 `static_assert` 钉死）：

- `sizeof(EngineLease) == sizeof(void*)`；
- 平凡可拷贝（`Tensor` 的 reshape / 表达式 leaf / 图节点输出都会复制它）。

### 2.3 进程级令牌池

`detail::engine_token_pool()` 是**只增不减**的 `std::deque<EngineLiveness>`：

- `deque::push_back` 不移动已有元素 → 已发出的令牌地址恒稳定；
- 令牌地址**永不复用**：陈旧 lease 永远读到 `alive=false`，不会把新引擎
  误认成旧引擎；
- 池本身故意泄漏（与 `GpuBackend::instance()` 同源），避免静态析构顺序问题；
- 成本：每引擎约 16 B。引擎是重量级对象，一个进程通常只有 1 个；测试按
  引擎数增长，量级仍是 KB。

**为什么不用 `shared_ptr`/`weak_ptr`**：那会让 `EngineLease` 变成 16 B，
并在每次拷贝时做原子引用计数——`Tensor` 的拷贝在热路径上。当前方案把成本
压到“每次 `require()` 一次 acquire load”。

## 3. Layer 状态机

`Layer::BindingState`：

| 状态 | 含义 | `engine_ref()` | `set_precision_profile` | `init()` |
|---|---|---|---|---|
| `unbound` | 从未 init | 终止：先调 `init(engine)` | 允许 | 绑定并执行 |
| `ready` | init 成功 | 引擎存活则返回；已销毁则终止 | 拒绝（profile 不可变） | 同引擎可重入；换引擎报错 |
| `failed` | 上一次 init 失败 | 终止：半初始化，修复后重新 init | 允许 | 允许重试 |

`init()` 的失败回滚是显式的：

```cpp
NN_TRY_CHECK(check_precision_support_());   // 纯静态校验，失败不留绑定
engine_ = engine.lease();
binding_state_ = BindingState::ready;        // init_impl 内可用 engine_ref()
if (auto r = init_impl(engine); !r)  { binding_state_ = failed; return r; }
if (auto r = check_param_precision_(); !r) { binding_state_ = failed; return r; }
```

“是否已 init”不再用“指针非空”推断——后者会把失败 init 误判成成功，并让
后续 `set_precision_profile` 直接 abort（旧行为，已由
`layer_lifetime_test` 回归覆盖）。

## 4. 借用点清单

| 借用方 | 成员 | 说明 |
|---|---|---|
| `Layer` | `EngineLease engine_` + `BindingState` | 访问咽喉 `engine_ref()` |
| `Model` | `EngineLease engine_` | `engine()` 走 `require`；forward/backward 先查 `alive()` 并返回错误 |
| `Tensor` | `EngineLease engine_` | `bound()` = 出生绑定存在；`engine_alive()` 判定存活；`engine()` 走 `require` |
| `ExprGraph` | `EngineLease owner` | 录制段归属；`expired()` 时给出明确错误 |
| `ExprBlock` | `EngineLease eng` | `end_expr` 走 `require` |
| 函数局部指针（如 `text_train_loop`） | 裸指针 | 仅在函数作用域内使用，不存储；不参与本契约 |

`Tensor` 的两个语义要分清：

- `bound()`：是否 stamp 过引擎（出生绑定，身份用 `identity()` 比较）；
- `engine_alive()`：该引擎是否仍存活。

`ComputeEngine::stamp_` 只替换**未绑定或已失效**的绑定；绑定到另一个**存活**
引擎的张量保持原绑定（跨引擎 import 的既有语义）。

## 5. 新增借用方的检查清单

1. 成员类型用 `EngineLease`，不要用 `ComputeEngine*` / `observer_ptr`；
2. 从引擎取得 lease：`engine.lease()`（不要保存 `&engine`）；
3. 访问引擎只经 `require(owner, loc)`；需要“可恢复错误”时用
   `alive()`/`expired()` 先判定；
4. 判等用 `identity()`，不要用 `get()`（引擎销毁后 `get()` 为 nullptr）；
5. 不要试图延长引擎寿命（不要改成 `shared_ptr`）；
6. 新增测试：正常顺序 + 引擎先析构（子进程断言非零退出）。

## 6. 测试

`src/layer_lifetime_test.cpp`（ctest 名 `layer_lifetime_test`）覆盖：

1. `EngineLease` 三态与令牌地址不复用；
2. `Layer` 状态机、init 失败回滚、失败后注入 profile；
3. 跨引擎重绑返回 `Result` 错误；
4. 引擎先析构：`binding_state()==ready`、`engine_alive()==false`，
   `forward` 在子进程中终止且消息含 `[engine-lease]`；
5. 正常顺序 forward 不回归；
6. `Model::forward` 返回明确错误、`Model::engine()` 终止；
7. `Tensor::engine()` 终止、`bound()`/`engine_alive()` 语义。

## 7. 性能与尺寸

- `EngineLease`：8 B（与旧 `observer_ptr` 相同）；
- `Layer` / `Model`：多 1 B 状态字段（对齐后通常不增加实际尺寸）；
- `Tensor`：尺寸不变；
- 每次 `engine_ref()` / `require()`：一次 acquire load；
- 引擎构造：一次池分配（deque 追加）；引擎析构：一次 release store。

不引入 LTO、不引入异常、不改变 header-only 打包方式。
