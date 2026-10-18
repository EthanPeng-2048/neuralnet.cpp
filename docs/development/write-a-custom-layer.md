# 写一个自己的层（自定义 Layer 指南）

> **读者**：调用者（库使用者）。想在内置层之外加自己的结构时读本文。
> 背景实测报告（接入契约、7 个摩擦点的库外视角）见 [`20-custom-layer-ergonomics.md`](20-custom-layer-ergonomics.md)；
> 引擎原语清单见 [`../usage/03-compute-engine-usage.md`](../usage/03-compute-engine-usage.md)；
> 多精度语义契约见 [`05-mixed-precision.md`](05-mixed-precision.md)。
>
> 本文的三步路径与示例层**经实测**：示例 `LowRankAdapter` 通过数值梯度检查
> （`max|num−an| = 4.7e-06`）并能训练收敛（`loss 0.0615 → 0.000116`，50 步 Adam）。

---

## 0. 最短路径（三步）

1. 写一个 `final : public nn::Layer` 的头文件，覆写 `init_impl` / `forward` / `backward`，
   需要可学习参数时再覆写 `parameters()` / `param_gradients()`。
2. `model.add<MyLayer>(args...)`（或 `model.add_layer(std::make_unique<MyLayer>(...))`）。
3. 训练循环、优化器、多精度、序列化**一并生效**——自定义层与内置层没有特殊分支。

---

## 1. 完整示例：`LowRankAdapter`

残差低秩适配器（LoRA 形态）：`out = x + B·(A·x)`，`A:(rank,dim)`、`B:(dim,rank)`，
`B` 零初始化 ⇒ 初始为恒等映射。选它是因为它同时压到五条契约：有可学习参数、
forward 需要缓存、backward 有转置 matmul、表达式受限（一个表达式至多一个 matmul）、
多精度免费。

```cpp
#include <neuralnet.cpp/nn.hpp>

namespace demo
{

class LowRankAdapter final : public nn::Layer
{
private:
    std::size_t dim_;
    std::size_t rank_;
    nn::Tensor a_;             // (rank, dim)
    nn::Tensor b_;             // (dim, rank)，零初始化 ⇒ 初始为恒等映射
    nn::Tensor grad_a_;
    nn::Tensor grad_b_;
    nn::Tensor x_cache_;       // forward 的输入（backward 用）
    nn::Tensor h_cache_;       // forward 的中间量 h = A·x（backward 用）

    // 梯度缓冲惰性分配：只推理的路径不该为梯度付存储。
    void ensure_grad_()
    {
        if (grad_a_.valid()) return;
        nn::ComputeEngine& engine = engine_ref();
        grad_a_ = engine.create_tensor(rank_, dim_, p_.param, nn::InitSpec::zero());
        grad_b_ = engine.create_tensor(dim_, rank_, p_.param, nn::InitSpec::zero());
        // 本函数返回 void → 用 NN_UNWRAP（终止式），不能用 NN_FAIL（返回语义）
        NN_UNWRAP(grad_a_.valid() && grad_b_.valid(),
                 "LowRankAdapter: 梯度缓冲初始化失败");
    }

public:
    // 构造函数只存形状；权重在 init_impl 里创建（那里才有引擎）
    LowRankAdapter(std::size_t dim, std::size_t rank) : dim_(dim), rank_(rank) {}

    [[nodiscard]] const char* layer_name() const noexcept override { return "LowRankAdapter"; }

    // ⚠ 覆写的是 init_impl（不是 init）：init 是 NVI 公共入口，先绑定引擎、
    //   再做精度能力校验，最后才转到 init_impl
    [[nodiscard]] nn::Result<void> init_impl(nn::ComputeEngine& engine) override
    {
        const nn::Scalar limit =
            std::sqrt(6.0f / static_cast<nn::Scalar>(dim_ + rank_));
        a_ = engine.create_tensor(rank_, dim_, p_.param,           // 精度用 p_.param
                                  nn::InitSpec::uniform(-limit, limit, nn::kInitSeed));
        if (!a_.valid()) NN_FAIL("LowRankAdapter: a 初始化失败");
        b_ = engine.create_tensor(dim_, rank_, p_.param, nn::InitSpec::zero());
        if (!b_.valid()) NN_FAIL("LowRankAdapter: b 初始化失败");
        return {};
    }

    [[nodiscard]] std::vector<nn::TensorRef> parameters() override { return {a_, b_}; }

    [[nodiscard]] std::vector<nn::TensorRef> param_gradients() override
    {
        ensure_grad_();
        return {grad_a_, grad_b_};
    }

    [[nodiscard]] bool grad_buffers_allocated() const noexcept override
    {
        return grad_a_.valid();
    }

    void clear_cache() override { x_cache_ = nn::Tensor{}; h_cache_ = nn::Tensor{}; }

    [[nodiscard]] std::vector<nn::TensorRef> activation_cache() override
    {
        std::vector<nn::TensorRef> r;
        if (x_cache_.valid()) r.emplace_back(x_cache_);
        if (h_cache_.valid()) r.emplace_back(h_cache_);
        return r;
    }

    // 声明"可重算单元"：checkpoint 模式下由基类默认 forward_recompute 重建缓存
    [[nodiscard]] bool recompute_supported() const override { return true; }

    // ⚠ forward 不收 engine 形参（新签名）：engine 在 init 时已绑定，经 engine_ref() 取回
    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& input) override
    {
        nn::ComputeEngine& engine = engine_ref();
        if (input.rows() != dim_) NN_FAIL("LowRankAdapter: input 形状不匹配");
        // checkpoint 模式：不得保留中间激活（否则梯度检查点白开）
        if (!checkpoint_mode_) x_cache_ = input;

        // 表达式①：h = A·x —— 一个表达式至多一个 matmul，故必须分两次 compute
        auto h = nn::dsl::compute(engine, nn::dsl::matmul(a_, input, false, false),
                                  rank_, input.cols(), p_.compute);
        NN_TRY_CHECK(h);
        if (!checkpoint_mode_) h_cache_ = *h;

        // 表达式②：out = x + B·h（残差恒等路径）
        return nn::dsl::compute(engine,
            nn::dsl::leaf(input) + nn::dsl::matmul(b_, *h, false, false),
            dim_, input.cols(), p_.compute);
    }

    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& grad_output) override
    {
        nn::ComputeEngine& engine = engine_ref();
        if (!x_cache_.valid() || !h_cache_.valid())
            NN_FAIL("LowRankAdapter: 缓存缺失（checkpoint 模式需先 forward_recompute）");
        ensure_grad_();

        // gb = B^T·g → (rank, B)
        NN_TRY(gb, nn::dsl::compute(engine, nn::dsl::matmul(b_, grad_output, true, false),
                                    rank_, grad_output.cols(), p_.compute));
        // grad_a += gb·x^T → (rank, dim)
        auto acc_a = nn::dsl::compute_into(engine,
            nn::dsl::leaf(grad_a_) + nn::dsl::matmul(*gb, x_cache_, false, true), grad_a_);
        NN_TRY_CHECK(acc_a);
        // grad_b += g·h^T → (dim, rank)
        auto acc_b = nn::dsl::compute_into(engine,
            nn::dsl::leaf(grad_b_) + nn::dsl::matmul(grad_output, h_cache_, false, true), grad_b_);
        NN_TRY_CHECK(acc_b);
        // grad_x = g + A^T·gb → (dim, B)
        return nn::dsl::compute(engine,
            nn::dsl::leaf(grad_output) + nn::dsl::matmul(a_, *gb, true, false),
            dim_, grad_output.cols(), p_.compute);
    }
};

} // namespace demo
```

> **输出形状可省**：上例的 `dsl::compute(engine, expr, rows, cols, P)` 可写成
> `dsl::compute(engine, expr, P)` —— 输出网格由表达式推导（逐元素/广播/归约视图/
> matmul 段各按广播规则合并，口径见 [`cpp-api-manual.md`](cpp-api-manual.md) §6）。
> 推导不出的网格（如裸广播向量）会返回错误并提示改用显式形式，不会猜形状；
> 两种写法在同一表达式上**结果逐位一致**（库内 6 处调用点已改为省略形式）。

### 接入与训练（与内置层逐字一致）

```cpp
auto engine_owner = nn::CpuEngine::create();
nn::CpuEngine& engine = *engine_owner;
nn::Model model(engine);
NN_UNWRAP(model.add<nn::Linear>(28, 3));
NN_UNWRAP(model.add<demo::LowRankAdapter>(3, 2));   // dim=3, rank=2

auto opt = nn::create_optimizer("adam", engine, model.parameters(),
                                model.param_gradients(), 1e-2f);
// 循环体与内置层完全一样：zero_grad → forward → loss → backward → step
```

---

## 2. 接入契约（逐条）

这些点**编译器兜不住**（写了才对、不写也能过编译），是本指南的核心。

| # | 契约 | 不遵守的后果 |
|---|---|---|
| 1 | 覆写 **`init_impl(engine)`**，不是 `init` | `init` 是 NVI（绑定引擎 + 精度能力校验），覆写它 = 绕过校验 |
| 2 | `forward` / `backward` **不带 engine 形参**；用 `engine_ref()` 取引擎 | 旧签名编译失败（这是有意的） |
| 3 | `checkpoint_mode_ == true` 时 forward **不得**保留中间激活 | 编译通过、普通训练正常，**只有开梯度检查点才崩** |
| 4 | 覆写 `clear_cache()` / `activation_cache()` | 不报错，但 activation offload 与本层缓存释放**静默失效**（显存泄漏） |
| 5 | 参数精度用 `p_.param`、算子输出精度用 `p_.compute`、数值敏感链用 `p_.stable` | 硬编码 F32 = 编译过、测试过、**静默失去 f16 收益** |
| 6 | `parameters()` 与 `param_gradients()` **同序同数量** | 优化器把梯度写到错的参数上 |
| 7 | `p_` 在 `init` 之后不可变（`set_precision_profile` 已 init 时 fail-fast） | 触发 fail-fast；只要保证 profile 在 add 之前设好即可 |

### 2.1 复合层（持子层）额外两条

如果层内部持有别的 `Layer`：

- 覆写**受保护的** `param_children()` 按顺序返回子层指针——参数的**下标转发**（`export_param` /
  `import_param`）靠它，序列化才认得你的子层参数；
- 覆写 `set_precision_profile`：先调基类（先校验），再逐个把 profile 下传给子层与辅助对象
  （RoPE / 位置编码器），否则子层内的 DSL 求值退回 F32。

### 2.2 精度能力声明（可选）

默认声明 `PrecisionSupport::rounding_all_slots()` = `{f16, f32}` × 四槽，所以普通层**一行不用写**。
只有当你的层需要**别的**精度（例如像 `BitLinear` 那样只接受 `param=T1_58`）时才覆写：

```cpp
[[nodiscard]] nn::PrecisionSupport precision_support() const override
{
    nn::PrecisionSupport s = nn::PrecisionSupport::rounding_all_slots();
    // s.param = nn::PrecisionSet::of({nn::Precision::T1_58});   // 例：只接受三值权重槽
    return s;
}
```

`nn::Layer::init` 会在初始化期把不兼容组合拦下，报错含层名 + 槽位 + 取值 + 允许集合。
若层同时持线性子层又想支持三值权重，用 `nn::split_ternary_profile(layer_name, profile, ternary)`
分流（自身拿 `param` 归一化为 f32 的那份，线性子层拿 `param=T1_58` 的那份）。

---

## 3. 数值梯度自检（backward 写错时唯一的兜底）

新层的转置标志最容易写错，而库**不提供**公共梯度检查工具——把下面这段抄进你的测试
（自包含，约 40 行）。它遍历 `model.parameters()` / `model.param_gradients()`，
对每个参数抽样若干元素做中心差分：

```cpp
auto engine_owner = nn::CpuEngine::create();
nn::CpuEngine& engine = *engine_owner;
nn::Model model(engine);
NN_UNWRAP(model.add<nn::Linear>(4, 3));
NN_UNWRAP(model.add<demo::LowRankAdapter>(3, 2));   // 换成你的层

const std::size_t in_dim = 4, dim = 3, batch = 4;
nn::Tensor x = NN_UNWRAP(engine.try_create_tensor(in_dim, batch, nn::Precision::F32,
                                                 nn::InitSpec::uniform(-1.f, 1.f, 7)));
nn::Tensor zero = NN_UNWRAP(engine.try_create_tensor(dim, batch, nn::Precision::F32,
                                                    nn::InitSpec::zero()));
nn::MSELoss mse;
auto loss_once = [&]() -> double {
    nn::Tensor out = NN_UNWRAP(model.forward(x));
    return static_cast<double>(NN_UNWRAP(mse.forward(engine, out, zero)));
};

// 1. 解析梯度：一次 forward + backward
NN_UNWRAP(model.zero_grad());
nn::Tensor out = NN_UNWRAP(model.forward(x));
(void)NN_UNWRAP(mse.forward(engine, out, zero));
nn::Tensor gout = NN_UNWRAP(mse.backward());
(void)NN_UNWRAP(model.backward(gout));

// 2. 逐参数抽样做中心差分
auto params = model.parameters();
auto grads  = model.param_gradients();
const double eps = 1e-3;
double max_err = 0.0;
for (std::size_t pi = 0; pi < params.size(); ++pi)
{
    nn::Tensor& P = params[pi].get();
    nn::Tensor& G = grads[pi].get();
    std::vector<nn::Scalar> pv(P.size()), gv(G.size());
    NN_UNWRAP(engine.read(P, std::span(pv)));
    NN_UNWRAP(engine.read(G, std::span(gv)));
    for (std::size_t k = 0; k < std::min<std::size_t>(P.size(), 4); ++k)
    {
        const std::size_t idx = (k * 5 + 1) % P.size();
        const nn::Scalar orig = pv[idx];
        pv[idx] = nn::Scalar(static_cast<double>(orig) + eps);
        NN_UNWRAP(engine.write(P, std::span(pv)));
        const double lp = loss_once();
        pv[idx] = nn::Scalar(static_cast<double>(orig) - eps);
        NN_UNWRAP(engine.write(P, std::span(pv)));
        const double lm = loss_once();
        pv[idx] = orig;
        NN_UNWRAP(engine.write(P, std::span(pv)));
        const double num = (lp - lm) / (2.0 * eps);
        max_err = std::max(max_err, std::fabs(num - static_cast<double>(gv[idx])));
    }
}
std::printf("gradcheck: max|num-an| = %.3e\n", max_err);   // 应 << 1e-3
```

实测（上面这个层、CPU、f32）：`max|num-an| = 4.7e-06`。
把 `model.add<demo::LowRankAdapter>` 换成你的层、`target` 换成真实标签即可复用。

---

## 4. GPU 检查清单（CPU 通过 ≠ GPU 通过）

GPU 表达式 shader 是**构建期**生成的（AOT 闭合世界，铁律 #7），自定义层的新结构不在库内
注册表里。因此：

1. 先保证 **CPU 上通过**（含上面的数值梯度检查）；
2. 在 CMake 里加一行，把你的表达式结构编进目标专属注册表：

   ```cmake
   nn_enable_gpu_fusion(my_app MAIN ${CMAKE_CURRENT_SOURCE_DIR}/my_app_main.cpp)
   ```

   `MAIN` 必须指向含 `main` 的源文件（收集器自带 main，需从目标源列表剔除）。
   该函数只在 GPU 启用时可用；未启用时调用会给出指向根因的报错。
3. 训练/推理时用 `engine.begin_batch()` / `engine.end_batch()` 包住 forward（可选，消提交开销）——
   注意**录制期引用的张量必须活到 `end_batch()` 之后**（铁律 #6）。

漏掉第 2 步的现象是：**构建期不报错，运行期按 `expr_spec_key` 硬报错**。这是设计使然
（无 eager 回退），不是 bug。

---

## 5. 常见编译错误与正确写法

| 你写的 | 报错 | 正确写法 |
|---|---|---|
| `if (!ok) NN_FAIL("...")` 在 `void` 函数 / `main` 里 | `-Wreturn-mismatch`（`NN_FAIL` = `return std::unexpected`） | `NN_UNWRAP(ok, "...")` 或 `NN_FATAL("...")` |
| `auto x = NN_UNWRAP(engine.from_matrix(m)); *x;` | `member reference type 'Tensor' is not a pointer` | `NN_UNWRAP` **已解包**，`x` 就是 `Tensor` |
| `NN_UNWRAP(model.backward(g));`（丢弃 `[[nodiscard]]`） | `ignoring return value ... nodiscard` | `(void)NN_UNWRAP(model.backward(g));` |
| `opt->zero_grad();` 忽略返回值 | `-Wunused-result` | `NN_UNWRAP(opt->zero_grad());` |
| 覆写 `init(engine)` 而不是 `init_impl` | 行为异常（绕过 NVI 校验） | 覆写 `init_impl` |
| 一个表达式里写两个 `dsl::matmul` | 折叠失败（不是清晰断言） | 拆成两次 `dsl::compute` |
| forward 无条件 `x_cache_ = input;` | 梯度检查点下崩 | 加 `if (!checkpoint_mode_)` |

---

## 6. 免费获得的四件事（无需写代码）

1. **优化器接入**：`parameters()` / `param_gradients()` 各返回一个列表，
   `create_optimizer` / `zero_grad` / `clip_grad_norm` 全部自动生效。
2. **多精度**：全程用 `p_.param` / `p_.compute`，`--f16` 配置下自动获得 f16 存储 + f32 计算。
3. **序列化**：默认按 `Dense` 形态导出/导入（`export_param` / `import_param` 基类已实现）；
   复合层只要正确实现 `param_children()` 就能整树落盘。
4. **剖析**：`NN_PROFILE=1` 时，`Model` 的逐层 scope 已包含你的层（无需额外插桩）。

> 相关样例：`examples/fusion_custom_layer.hpp`（带自研融合表达式的层 + GPU 注册形态）。
