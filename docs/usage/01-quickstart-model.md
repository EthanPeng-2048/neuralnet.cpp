# 快速上手：构建自己的模型

**本教程教你如何使用 `ComputeEngine` + `Layer` + `Model` 三件套从零构建神经网络模型。**

---

## 三件套概念

```
ComputeEngine  — 硬件抽象（CPU/GPU），提供计算原语
Layer          — 神经网络层，组合原语表达算法
Model          — 层容器，管理 Layer 的 forward/backward 链
```

**关系：**

```cpp
nn::Model model(engine);              // Model 绑定一个 Engine
model.add<nn::Linear>(784, 128);      // 添加 Layer（add<T>(args...) 返回 Result<void>，
model.add<nn::ReLU>();                //   构造后自动 init(engine)）
model.add<nn::Linear>(128, 10);

auto result = model.forward(input);   // engine 在 Model 内部自动传递
```

---

## 第一步：选择引擎

```cpp
#include <neuralnet.cpp/nn.hpp>

// CPU 引擎（默认，无需额外依赖）
nn::CpuEngine engine;

// GPU 引擎（需要 Vulkan SDK，条件编译 NN_HAS_VULKAN）
// nn::GpuEngine gpu_engine;
```

---

## 第二步：构建 Model

### 方式一：模板 add（推荐，适合 MLP）

```cpp
nn::Model model(engine);

model.add<nn::Linear>(784, 256);   // 输入层: 784 → 256
model.add<nn::ReLU>();             // 激活函数
model.add<nn::Linear>(256, 128);   // 隐藏层: 256 → 128
model.add<nn::ReLU>();
model.add<nn::Linear>(128, 10);    // 输出层: 128 → 10
```

> `add<LayerType>(args...)` 返回 `Result<void>`（层构造后自动调用 `Layer::init(engine)`，
> 失败经 Result 上抛）；它不返回 `*this`，**不能点链调用**——逐行添加、需要时逐个检查错误。

### 方式二：使用工厂函数（MNIST/GPT 预设）

```cpp
// MNIST MLP：784 → 512 → 256 → 128 → 64 → 10
auto model_result = nn::build_mnist_mlp_model(engine);
nn::Model model = std::move(*model_result);

// MNIST Transformer (ViT)：28×28 图像，patch_size=7
// 默认含编码器末端 final norm（NormPlace::Final，原版 ViT 的 ln_f）；
// 块内 pre-norm 默认 LayerNorm，可用 NormType 参数换 RMSNorm/BatchNorm
auto model_result = nn::build_mnist_transformer_model(engine);
nn::Model model = std::move(*model_result);

// GPT 语言模型
auto model_result = nn::build_gpt_model(
    engine,
    10000,  // vocab_size
    128,    // d_model
    256,    // seq_len
    4,      // num_heads
    512,    // d_ff
    4       // num_layers
);
nn::Model model = std::move(*model_result);
```

### 方式三：添加其他内置层

```cpp
nn::Model model(engine);
model.add<nn::GeLU>();                  // QuickGeLU 激活
model.add<nn::LayerNorm>(256);          // 层归一化
model.add<nn::Softmax>();               // Softmax
model.add<nn::PositionalEncoding>(128, 1024); // 位置编码
model.add<nn::FeedForward>(128, 512);   // FFN
model.add<nn::TransformerEncoderLayer>(128, 4, 512, 256); // Transformer 编码器层
model.add<nn::CausalSelfAttention>(128, 4, 1024);  // 因果自注意力
model.add<nn::GPTBlock>(128, 4, 512, 1024);        // GPT 块
```

---

## Tensor 的使用

`Tensor` 是所有数据的统一容器：

```cpp
// M1（docs/development/17 §4.1）起：创建/读写张量一律经 Engine——
// Tensor 的静态工厂与存储访问器已私有化，绕过引擎 = 编译错误。

// 通过 Engine 创建（统一入口：自动分配、出生绑定到引擎）
nn::Tensor t1 = engine.create_tensor(784, 32);  // 纯分配
nn::Tensor t3 = engine.create_tensor(784, 32);
(void)engine.zero(t3);  // 清零（分配后手动）

// M2（docs/development/17 §4.4）：声明式初始化——层算分布参数，引擎填数；
// 分布类 seed 必填（同 seed 不同张量由引擎混流，不撞流）
nn::Tensor t_w = engine.create_tensor(32, 10, nn::Precision::F32,
                                      nn::InitSpec::uniform(-0.1f, 0.1f, nn::kInitSeed));
nn::Tensor t_b = engine.create_tensor(10, 1, nn::Precision::F32,
                                      nn::InitSpec::zero());

// 从 Matrix 上传（Matrix = 宿主 I/O 载体）
nn::Matrix m(784, 32);
auto t_result = engine.from_matrix(m);
nn::Tensor t2 = std::move(*t_result);

// 转换为 Matrix（GPU 张量会先等待在飞命令完成，再下载回 CPU）
auto m_result = engine.to_matrix(t3);
nn::Matrix m2 = std::move(*m_result);
```

**张量布局（重要！）：**

```
Matrix/Tensor 行主序:  (rows, cols)
  data_[row * cols + col]

神经网络约定（列主序 batch-major）:
  输入: (feature_dim, batch_size)    — 每列一个样本
  输出: (out_dim, batch_size)
  权重: (out_features, in_features)
```

---

## 完整前向传播示例

```cpp
#include <neuralnet.cpp/nn.hpp>

int main() {
    nn::CpuEngine engine;

    // 1. 构建模型
    nn::Model model(engine);
    model.add<nn::Linear>(784, 256);
    model.add<nn::ReLU>();
    model.add<nn::Linear>(256, 10);

    // 2. 准备输入 (784 像素, 32 样本)
    nn::Tensor input = engine.create_tensor(784, 32);
    // ... 填充数据 ...

    // 3. 前向传播
    auto out_result = model.forward(input);
    if (!out_result) {
        std::cerr << "Error: " << out_result.error().message << "\n";
        return 1;
    }
    nn::Tensor output = std::move(*out_result);
    // output: (10, 32) — 每列一个样本的 10 类 logits

    // 4. 下载到 CPU 做 argmax
    auto m = engine.to_matrix(output);
    for (std::size_t b = 0; b < 32; ++b) {
        std::size_t best = 0;
        for (std::size_t c = 1; c < 10; ++c) {
            if (m->at(c, b) > m->at(best, b))
                best = c;
        }
        std::cout << "Sample " << b << ": predicted " << best << "\n";
    }
}
```

---

## 自定义 Layer

如果内置层不够用，可以继承 `Layer` 实现自己的层：

```cpp
class MyLayer final : public nn::Layer {
private:
    nn::Tensor weight_;
    nn::Tensor grad_weight_;
    nn::Tensor input_cache_;
    std::size_t in_dim_;
    std::size_t out_dim_;

public:
    // 构造函数只存形状（Model::add 构造后会自动调用 init(engine)）
    MyLayer(std::size_t in_dim, std::size_t out_dim)
        : in_dim_(in_dim), out_dim_(out_dim) {}

    // 权重初始化：M2 声明式——层算分布参数（Xavier limit），引擎填数
    [[nodiscard]] nn::Result<void> init(nn::ComputeEngine& engine) override
    {
        const nn::Scalar limit =
            std::sqrt(6.0f / static_cast<nn::Scalar>(in_dim_ + out_dim_));
        weight_ = engine.create_tensor(out_dim_, in_dim_, nn::Precision::F32,
                                       nn::InitSpec::uniform(-limit, limit, nn::kInitSeed));
        if (!weight_.valid())
            return std::unexpected(nn::Error{"MyLayer: 权重初始化失败"});

        grad_weight_ = engine.create_tensor(out_dim_, in_dim_, nn::Precision::F32,
                                            nn::InitSpec::zero());
        if (!grad_weight_.valid())
            return std::unexpected(nn::Error{"MyLayer: 梯度缓冲初始化失败"});
        return {};
    }

    // 参数访问（供 Optimizer 使用）：TensorRef = reference_wrapper<Tensor>
    std::vector<nn::TensorRef> parameters() override {
        return {weight_};
    }
    std::vector<nn::TensorRef> param_gradients() override {
        return {grad_weight_};
    }

    // 前向传播
    [[nodiscard]] nn::Result<nn::Tensor> forward(
        nn::ComputeEngine& engine, const nn::Tensor& input) override
    {
        input_cache_ = input;
        // 组合原语表达算法
        return engine.matmul(weight_, input);
    }

    // 反向传播
    [[nodiscard]] nn::Result<nn::Tensor> backward(
        nn::ComputeEngine& engine, const nn::Tensor& grad_output) override
    {
        // grad_input = weight^T × grad_output
        auto grad_input = engine.matmul(weight_, grad_output, /*transA=*/true);

        // grad_weight += grad_output × input^T
        auto gw = engine.matmul(grad_output, input_cache_, /*transA=*/false, /*transB=*/true);
        if (!gw) return std::unexpected(gw.error());
        auto r = engine.add_inplace(grad_weight_, *gw);
        if (!r) return std::unexpected(r.error());

        return grad_input;
    }
};
```

**添加到 Model：**

```cpp
nn::Model model(engine);
model.add<MyLayer>(784, 256);
model.add<nn::ReLU>();
model.add<MyLayer>(256, 10);
```

---

## 常见陷阱

### 1. Tensor 布局错误

```cpp
// ❌ 错误：(batch, feature) — 行主序 batch-major
nn::Tensor input = engine.create_tensor(32, 784);

// ✅ 正确：(feature, batch) — 列主序 batch-major
nn::Tensor input = engine.create_tensor(784, 32);
```

### 2. 忘记绑定 Engine

```cpp
// ❌ 错误：Model 未绑定 Engine
nn::Model model;                    // 默认构造
model.add<nn::Linear>(784, 10);     // NN_ASSERT(engine_, "Model: engine not bound") 断言失败

// ✅ 正确
nn::CpuEngine engine;
nn::Model model(engine);
model.add<nn::Linear>(784, 10);
```

### 3. 跨设备操作绕过 Engine

```cpp
// ❌ 错误：直接摸 GPU 张量的 CPU 存储（M1 起存储访问器已私有 → 这行编译不过）
nn::GpuEngine engine;                       // 需 NN_HAS_VULKAN
nn::Tensor t = engine.create_tensor(10, 10);
auto& m = t.cpu_matrix();                   // 编译错误：cpu_matrix 是私有成员

// ✅ 正确：跨设备读写统一走 engine（to_matrix 下载，GPU 会等待在飞命令完成）
auto m_result = engine.to_matrix(t);
```

### 4. 梯度清零遗漏

```cpp
// ❌ 错误：忘记清零梯度
auto out = model.forward(input);
auto loss = loss_fn.forward(engine, *out, target);
auto grad = loss_fn.backward();
model.backward(*grad);
optimizer->step();  // 梯度累积！

// ✅ 正确：每步清零
optimizer->zero_grad();   // 或 model.zero_grad()（两者都不带 engine 参数）
auto out = model.forward(input);
auto loss = loss_fn.forward(engine, *out, target);
auto grad = loss_fn.backward();
model.backward(*grad);
optimizer->step();
```

---

## 可用层速查表

| 层 | 构造参数 | 用途 |
|----|---------|------|
| `Linear(in, out)` | 输入/输出维度 | 全连接层 |
| `ReLU()` | 无 | ReLU 激活 |
| `GeLU()` | 无 | QuickGeLU 激活 |
| `LayerNorm(shape, eps)` | 归一化维度 + epsilon | 层归一化 |
| `Softmax()` | 无 | 行级 Softmax |
| `PositionalEncoding(d_model, max_len)` | 模型维度 + 最大长度 | 正弦位置编码 |
| `FeedForward(d_model, d_ff)` | 模型维度 + FFN 中间维度 | FFN = Linear₂(GeLU(Linear₁(x))) |
| `TransformerEncoderLayer(d_model, heads, d_ff, seq_len)` | 全部 | Pre-Norm 编码器层 |
| `TransformerEncoder(d_model, heads, d_ff, layers, patches)` | 全部 | ViT 编码器（含池化） |
| `PatchEmbedding(img, patch, d_model)` | 图像/patch/模型维度 | 图像 patch 嵌入 |
| `CausalSelfAttention(d_model, heads, max_len)` | 全部 | 因果自注意力 |
| `GPTBlock(d_model, heads, d_ff, max_len)` | 全部 | Pre-Norm 解码器块 |
| `GPTModel(vocab, d_model, seq, heads, d_ff, layers)` | 全部 | 完整 GPT 模型 |