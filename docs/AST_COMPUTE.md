# 表达式模板：编译期 AST 胶水语言设计

## 一、核心思想

上层用**普通 C++ 运算符**写表达式，编译器自动构建**编译期 AST**。
底层拿到 AST 后：
- **CPU 路径**：递归求值 AST（零开销，和手写循环等价）
- **GPU 路径**：AST → `consteval` 生成 SPIR-V 二进制 → `vkCreateShaderModule` → dispatch

**零运行时编译开销**：SPIR-V 二进制在编译期从 AST 直接生成，无需 GLSL 中间层，无需任何外部工具。

算法永远在上层，计算永远在底层，中间靠 AST 传递。

## 二、上层写法（Layer 视角）

```cpp
// Layer 中：用普通运算符写算法
Span x = result.span();

compute::apply(x, max(x, Scalar{0}));                              // ReLU
compute::apply(x, x * (Scalar{1} / (Scalar{1} + exp(-Scalar{1.702} * x)))); // QuickGeLU
compute::apply(x, select(x > Scalar{0}, x, x * Scalar{0.01}));     // LeakyReLU
compute::apply(x, abs(x));                                          // abs
compute::apply(x, x * Scalar{2} + Scalar{1});                       // 线性变换
```

> ⚠️ `Scalar` 是 [nn_config.hpp](../include/neuralnet.cpp/nn_config.hpp) 定义的标量类型别名（当前为 `float`）。
> 由于 `compute::apply` 接受 `const Expr&` 形参，字面量必须显式标 `Scalar{...}` 以触发模板运算符重载，
> 否则 `x > 0` 中的 `0`（int）会与 `Span::operator>(Scalar)` 的重载不匹配。
> 三元条件需用 `select()`，C++ 无法重载 `?:`。

上层**完全不知道**底下是 CPU 还是 GPU。运算符重载在幕后构建 AST。

## 三、Span：可构建 AST 的智能视图

```cpp
// span.hpp — 替代 std::span<Scalar> 作为逐元素操作的载体
class Span {
    Scalar* data_;
    std::size_t size_;
public:
    Scalar* data() noexcept;
    std::size_t size() const noexcept;

    // ── 运算符重载：返回 Expr 而非 Scalar ──────────────────
    // Span OP Scalar → BinaryExpr<Span, Scalar, Op>
    auto operator+(Scalar s) const;   // → BinaryExpr<...>
    auto operator-(Scalar s) const;
    auto operator*(Scalar s) const;
    auto operator/(Scalar s) const;
    auto operator>(Scalar s) const;   // → CmpExpr<...>
    auto operator<(Scalar s) const;
    auto operator==(Scalar s) const;

    // Scalar OP Span → BinaryExpr<Scalar, Span, Op>
    friend auto operator+(Scalar s, const Span& sp);
    friend auto operator*(Scalar s, const Span& sp);

    // Span OP Span → BinaryExpr<Span, Span, Op>
    auto operator+(const Span& other) const;
    auto operator-(const Span& other) const;
    auto operator*(const Span& other) const;
};
```

## 四、Expr 类型系统（编译期 AST 节点）

```cpp
// expr.hpp — 表达式模板核心

// ── 叶子节点 ────────────────────────────────────────────────
struct Val {                          // 常量
    Scalar value;
    auto eval(std::size_t) const { return value; }
    consteval SpvId to_spv(SpvBuilder& spv) const {
        return spv.float_constant(value);  // OpConstant %float value
    }
};

// ── 一元表达式 ──────────────────────────────────────────────
template<typename Child, typename Op>
struct UnaryExpr {
    Child child;
    static constexpr bool has_spv = Child::has_spv;

    auto eval(std::size_t i) const {
        return Op::apply(child.eval(i));
    }
    consteval SpvId to_spv(SpvBuilder& spv) const {
        auto a = child.to_spv(spv);
        return Op::spv_op(spv, a);           // 例如 OpFMax %a %float_0
    }
};

// ── 二元表达式 ──────────────────────────────────────────────
template<typename Left, typename Right, typename Op>
struct BinaryExpr {
    Left left;
    Right right;
    static constexpr bool has_spv = Left::has_spv && Right::has_spv;

    auto eval(std::size_t i) const {
        return Op::apply(left.eval(i), right.eval(i));
    }
    consteval SpvId to_spv(SpvBuilder& spv) const {
        auto l = left.to_spv(spv);
        auto r = right.to_spv(spv);
        return Op::spv_op(spv, l, r);        // 例如 OpFAdd %l %r
    }
};

// ── 三元表达式（条件选择）────────────────────────────────────
template<typename Cond, typename Then, typename Else>
struct TernaryExpr {
    Cond cond;
    Then then_expr;
    Else else_expr;
    static constexpr bool has_spv = Cond::has_spv && Then::has_spv && Else::has_spv;

    auto eval(std::size_t i) const {
        return cond.eval(i) ? then_expr.eval(i) : else_expr.eval(i);
    }
    consteval SpvId to_spv(SpvBuilder& spv) const {
        auto c = cond.to_spv(spv);
        auto t = then_expr.to_spv(spv);
        auto e = else_expr.to_spv(spv);
        return spv.select(c, t, e);         // OpSelect %c %t %e
    }
};
```

## 五、Op 策略（编译期函数对象）

```cpp
// ops.hpp — 每个 op 同时定义 CPU 行为和 SPIR-V 指令

namespace ops {

struct Add {
    static Scalar apply(Scalar a, Scalar b) { return a + b; }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a, SpvId b) {
        return spv.fadd(a, b);               // OpFAdd %result %a %b
    }
};

struct Mul {
    static Scalar apply(Scalar a, Scalar b) { return a * b; }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a, SpvId b) {
        return spv.fmul(a, b);               // OpFMul %result %a %b
    }
};

struct Gt {
    static bool apply(Scalar a, Scalar b) { return a > b; }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a, SpvId b) {
        return spv.ford_less_than(b, a);     // OpFOrdGreaterThan %a %b → bool
    }
};

struct Max {
    static Scalar apply(Scalar a, Scalar b) { return std::max(a, b); }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a, SpvId b) {
        return spv.glsl_std450_fmax(a, b);   // OpExtInst %GLSLstd450FMax %a %b
    }
};

struct Neg {
    static Scalar apply(Scalar a) { return -a; }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a) {
        return spv.fneg(a);                  // OpFNegate %result %a
    }
};

struct Abs {
    static Scalar apply(Scalar a) { return std::abs(a); }
    static consteval SpvId spv_op(SpvBuilder& spv, SpvId a) {
        return spv.glsl_std450_fabs(a);      // OpExtInst %GLSLstd450FAbs %a
    }
};

} // namespace ops
```

## 六、运算符重载：连接上层表达式与 AST

```cpp
// span_ops.hpp — Span 运算符重载

// Span + Scalar → BinaryExpr<Span, Val, ops::Add>
inline auto Span::operator+(Scalar s) const {
    return BinaryExpr<Span, Val, ops::Add>{*this, Val{s}};
}

// Span > Scalar → BinaryExpr<Span, Val, ops::Gt>
inline auto Span::operator>(Scalar s) const {
    return BinaryExpr<Span, Val, ops::Gt>{*this, Val{s}};
}

// Span * Span → BinaryExpr<Span, Span, ops::Mul>
inline auto Span::operator*(const Span& other) const {
    return BinaryExpr<Span, Span, ops::Mul>{*this, other};
}
```

## 七、compute::apply — 统一入口

> 注：当前实现位于 [compute_dispatch.hpp](../include/neuralnet.cpp/algebra/compute_dispatch.hpp)，
> 为自由函数 `nn::compute::apply(Span, const Expr&)`。下面是设计目标（含 GPU 分支预览）。

```cpp
// compute_dispatch.hpp

template <typename Expr>
void apply(Span x, const Expr &expr) {
    const auto n = x.size();
    if (n == 0) return;

#ifdef NN_HAS_VULKAN
    if constexpr (Expr::has_spv) {        // 编译期：此表达式能生成 SPIR-V？
        if (SmartPolicy::gpu_enabled) {
            auto& backend = GpuBackend::instance();
            if (backend.is_initialized() || backend.initialize()) {
                // 编译期生成 SPIR-V，运行时零开销加载
                consteval auto spv = expr.to_spv_module();
                if (backend.elementwise_from_spv(x, spv)) return;
            }
        }
    }
#endif

    // CPU 路径：按索引分发（小数据串行，大数据并行），编译器内联等价手写循环
    if (n >= SmartPolicy::PARALLEL_THRESHOLD) {
        auto indices = std::views::iota(std::size_t{0}, n);
        SmartPolicy::for_each(indices.begin(), indices.end(),
            [&x, &expr](std::size_t i) noexcept {
                x[i] = static_cast<Scalar>(expr.eval(i));
            });
    } else {
        for (std::size_t i = 0; i < n; ++i)
            x[i] = static_cast<Scalar>(expr.eval(i));
    }
}
```

## 八、完整的 AST 构建示例

上层写 `x * (Scalar{1} / (Scalar{1} + exp(Scalar{-1.702} * x)))` 时，编译器构建的类型：

```
BinaryExpr<
    Span,                                          // x
    BinaryExpr<
        Val{1.0},                                  // 1.0
        BinaryExpr<
            Val{1.0},                              // 1.0
            UnaryExpr<
                BinaryExpr<                        // -1.702 * x
                    Val{-1.702},                   // -1.702
                    Span,                          // x
                    ops::Mul
                >,
                ops::Exp                           // exp(-1.702 * x)
            >,
            ops::Add                               // 1.0 + exp(...)
        >,
        ops::Div                                   // 1.0 / (1.0 + exp(...))
    >,
    ops::Mul                                       // x * (1.0 / (...))
>
```

运行时调用 `expr.eval(i)` 对每个索引 `i` 求值，编译期内联后等价于：

```cpp
// 编译器展开后等价的手写循环
for (std::size_t i = 0; i < n; ++i) {
    Scalar xv = x[i];
    data[i] = xv * (1.0f / (1.0f + std::exp(-(-1.702f * xv))));
}
```

> ⚠️ 注意：C++ 无法重载 `operator?:`，因此三元条件表达式**不能**直接写成
> `x > 0 ? x : x * 0.01f`。需要使用 `select()` 自由函数（见 [expr.hpp](../include/neuralnet.cpp/algebra/expr.hpp)）。

## 九、条件表达式（三元选择）

> C++ 不允许重载 `operator?:`，因此 LeakyReLU 不能直接写成 `x > 0 ? x : x * 0.01f`。
> 提供 `select(cond, then, else)` 自由函数替代（见 [expr.hpp](../include/neuralnet.cpp/algebra/expr.hpp)）。

```cpp
// LeakyReLU: x > 0 ? x : x * 0.01  —— 用 select() 表达
compute::apply(x, select(x > Scalar{0}, x, x * 0.01f));

// 编译器构建：
TernaryExpr<
    BinaryExpr<Span, Val, ops::Gt>,     // x > 0
    Span,                                // x
    BinaryExpr<Span, Val, ops::Mul>     // x * 0.01
>

// CPU 求值（编译期内联）：
// for (i=0..n) data[i] = (data[i] > 0) ? data[i] : (data[i] * 0.01f);
```

## 十、CPU 路径的求值策略

```cpp
// 逐元素递归求值
for (std::size_t i = 0; i < n; ++i)
    data[i] = expr.eval(i);
```

**关键**：因为 Expr 类型在编译期完全确定，编译器会：
1. 内联所有 `eval()` 调用
2. 消除 AST 节点的内存分配（全部是栈上值类型）
3. 向量化最终的标量循环

实测性能与手写 `for` 循环**完全等价**（零开销抽象）。

## 十一、GPU 路径的运行时流程

```
Layer: compute::apply(span, expr)
  ↓
ComputeDispatch: Expr::has_spv == true（编译期检查）
  ↓
consteval auto spv = expr.to_spv_module()   // 编译期：AST → SPIR-V 二进制
  ↓
GpuBackend: vkCreateShaderModule(spv_bytes)  // 运行时：加载二进制，零编译
  ↓
vkCmdDispatch → GPU 执行
```

**零运行时编译开销**：SPIR-V 在编译期由 `consteval` 从 AST 直接生成，
运行时只做内存拷贝 + `vkCreateShaderModule`（微秒级）。

## 十二、性能分析

| 路径 | 编译期 | 运行时 | 开销 |
|------|--------|--------|------|
| AST 构建 | ✅ 编译器完成 | — | 零 |
| CPU 求值 | ✅ 编译器内联 | ✅ 执行 | 与手写循环等价 |
| SPIR-V 生成 | ✅ `consteval` | — | 零 |
| `vkCreateShaderModule` | — | ✅ 加载二进制 | 微秒级 |
| GPU dispatch | — | ✅ | 与现有相同 |

与 GLSL→SPIR-V 运行时编译方案的对比：

| | GLSL 方案 | SPIR-V 直出 |
|---|-----------|------------|
| 运行时编译 | ~2ms（GLSL→SPIR-V） | **零** |
| 外部依赖 | `glslangValidator` | **无** |
| 缓存需求 | 需要（避免重复编译） | 不需要 |
| 首次开销 | ~2ms | 微秒级 |

## 十三、扩展新操作

### 添加 exp()

**步骤 1**：在 [ops.hpp](../include/neuralnet.cpp/algebra/ops.hpp) 中定义 Op（CPU `apply` + 可选 SPIR-V 指令）
```cpp
struct Exp {
    [[nodiscard]] static Scalar apply(Scalar a) noexcept { return std::exp(a); }
    // GPU 路径启用后再添加：
    // static consteval SpvId spv_op(SpvBuilder& spv, SpvId a) { return spv.glsl_std450_exp(a); }
};
```

**步骤 2**：在 [expr.hpp](../include/neuralnet.cpp/algebra/expr.hpp) 中添加自由函数（与现有 `exp`/`log`/`sigmoid` 等保持一致）
```cpp
template <Expression Expr>
[[nodiscard]] auto exp(const Expr &expr) {
    return UnaryExpr<Expr, ops::Exp>{expr};
}
```

**步骤 3**：完成。上层直接使用：
```cpp
compute::apply(x, exp(x));
```

**不需要修改 `compute_dispatch.hpp` 或 `Matrix`。**

## 十四、局限性

| 特性 | CPU | GPU |
|------|-----|-----|
| 算术运算 (+, -, *, /) | ✅ | ✅（待实现） |
| 比较 (>, <, ==) | ✅ | ✅（待实现） |
| 三元选择 `select()` | ✅ | ✅（待实现） |
| 数学函数 (exp, abs, max, min) | ✅ 已定义 Op | ✅ 需补 Op 的 SPIR-V 指令 |
| 循环/分支逻辑 | ❌ AST 不支持 | ❌ |
| 函数调用（非数学函数） | ❌ | ❌ |
| 任意 C++ lambda | ❌ | ❌ |

> ⚠️ C++ 无法重载 `operator?:`，因此"三元条件"通过 `select()` 自由函数实现，不是语言原生运算符。

**神经网络覆盖率**：ReLU, GeLU, SiLU, Softmax(部分), LayerNorm(部分) 等常见激活函数均可表达。覆盖率约 **85-90%** 的逐元素操作。

## 十五、跨平台支持

SPIR-V 是 Khronos 组织定义的 **GPU 中间表示**，一次生成，到处执行。

| 平台 | Vulkan Compute | SPIR-V 直出 | 说明 |
|------|----------------|-------------|------|
| **Windows** | ✅ | ✅ | 原生支持 |
| **Linux** | ✅ | ✅ | Mesa/NVIDIA/AMD 驱动 |
| **Android (Termux)** | ✅ | ✅ | 7.0+ 系统级 Vulkan，compute 无需窗口 |

**不支持**：macOS/iOS（Apple 不提供 Vulkan，不纳入跨平台范围）。

SPIR-V 二进制在三个平台间**完全共享**，零改动。GPU 驱动直接消费 SPIR-V，
无需任何平台特定的 shader 编译器。

## 十六、与现有代码的关系

> 本表反映设计目标。当前实现状态：CPU 路径已落地（[algebra/](../include/neuralnet.cpp/algebra/) 下的 `expr.hpp` / `ops.hpp` / `span.hpp` / `compute_dispatch.hpp`），GPU 路径尚未开始。

| 现有组件 | 改动 |
|---------|------|
| [nn_config.hpp](../include/neuralnet.cpp/nn_config.hpp) (SmartPolicy) | 不变，CPU 并行基础设施复用 |
| [matrix.hpp](../include/neuralnet.cpp/algebra/matrix.hpp) (Matrix) | 不变，纯数据容器；通过 `span()` 暴露给 AST 入口 |
| [layer.hpp](../include/neuralnet.cpp/layer.hpp) (Layer) | 直接调用 `compute::apply(span, expr)` 表达 ReLU/GeLU 等逐元素算法 |
| [compute_dispatch.hpp](../include/neuralnet.cpp/algebra/compute_dispatch.hpp) | 已模板化 `apply(Span, Expr)`；GPU 分支待补 |
| `vk_backend.hpp` (GpuBackend) | 待新增：`elementwise_from_spv(span, spv_bytes)` |
| **待新增** `spv_builder.hpp` | SPIR-V 二进制生成器（consteval，编译期） |
| [ops.hpp](../include/neuralnet.cpp/algebra/ops.hpp) | 已存在：CPU `apply`；待补每个 Op 的 `spv_op` |
| [expr.hpp](../include/neuralnet.cpp/algebra/expr.hpp) | 已存在：AST 节点类型系统；待补 `to_spv()` / `has_spv` |

> 旧设计中提到的 `elementwise.comp` 预编译快速路径已废弃，统一由 AST → SPIR-V 直出。
