# 🛠️ 快速上手：构建自己的模型

> 本教程教你如何使用 `ComputeEngine` + `Layer` + `Model` 三件套从零构建神经网络模型。

---

## 📐 三件套概念

```
ComputeEngine  — 硬件抽象（CPU/GPU），提供计算原语
Layer          — 神经网络层，组合原语表达算法
Model          — 层容器，管理 Layer 的 forward/backward 链
```

**关系：**

```cpp
Model model(engine);             // Model 绑定一个 Engine
model.add_linear(784, 128);     // 添加 Layer（内部持有 engine 引用）
model.add_relu();
model.add_linear(128, 10);

auto result = model.forward(input);  // engine 在 Model 内部自动传递
```

---

## 🚀 第一步：选择引擎

```cpp
#include <neuralnet.cpp/nn.hpp>

// CPU 引擎（默认，无需额外依赖）
nn::CpuEngine engine;

// GPU 引擎（需要 Vulkan SDK，条件编译 NN_HAS_VULKAN）
// nn::GpuEngine gpu_engine;
```

---

## 🔨 第二步：构建 Model

### 方式一：链式构建（推荐，适合 MLP）

```cpp
nn::Model model(engine);

model.add_linear(784, 256)     // 输入层: 784 → 256
     .add_relu()                // 激活函数
     .add_linear(256, 128)     // 隐藏层: 256 → 128
     .add_relu()
     .add_linear(128, 10);     // 输出层: 128 → 10
```

### 方式二：使用工厂函数（MNIST/GPT 预设）

```cpp
// MNIST MLP：784 → 512 → 256 → 128 → 64 → 10
auto model_result = nn::build_mnist_mlp_model(engine);
nn::Model model = std::move(*model_result);

// MNIST Transformer (ViT)：28×28 图像，patch_size=7
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

### 方式三：模板 add（自定义层）

```cpp
nn::Model model(engine);
model.add<nn::Linear>(784, 256);        // 等价于 add_linear
model.add<nn::ReLU>();                  // 等价于 add_relu
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

## 📊 Tensor 的使用

`Tensor` 是所有数据的统一容器：

```cpp
// 创建 CPU 张量
nn::Tensor t = nn::Tensor::cpu(784, 32);  // 784×32，零初始化

// 从 Matrix 创建（共享所有权）
nn::Matrix m(784, 32);
nn::Tensor t = nn::Tensor::from_matrix(std::move(m));

// 通过 Engine 创建（自动分配到目标设备）
nn::Tensor t = engine.create_tensor(784, 32);
engine.zero(t);  // 清零

// 转换为 Matrix（仅 CPU 张量）
auto m_result = engine.to_matrix(t);
nn::Matrix m = std::move(*m_result);

// 从 Matrix 上传到 Engine
auto t_result = engine.from_matrix(m);
nn::Tensor t = std::move(*t_result);
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

## 🔄 完整前向传播示例

```cpp
#include <neuralnet.cpp/nn.hpp>

int main() {
    nn::CpuEngine engine;

    // 1. 构建模型
    nn::Model model(engine);
    model.add_linear(784, 256)
         .add_relu()
         .add_linear(256, 10);

    // 2. 准备输入 (784 像素, 32 样本)
    nn::Tensor input = nn::Tensor::cpu(784, 32);
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

## 🏗️ 自定义 Layer

如果内置层不够用，可以继承 `Layer` 实现自己的层：

```cpp
class MyLayer final : public nn::Layer {
private:
    nn::Tensor weight_;
    nn::Tensor grad_weight_;
    nn::Tensor input_cache_;

public:
    MyLayer(nn::ComputeEngine& engine, std::size_t in_dim, std::size_t out_dim)
    {
        // 在 CPU 初始化权重，通过 engine 上传
        nn::Matrix w_cpu = nn::Matrix::random(out_dim, in_dim);  // 假设有此方法
        auto w = engine.from_matrix(w_cpu);
        weight_ = std::move(*w);

        grad_weight_ = engine.create_tensor(out_dim, in_dim);
        engine.zero(grad_weight_);
    }

    // 参数访问（供 Optimizer 使用）
    std::vector<nn::Tensor*> parameters() override {
        return {&weight_};
    }
    std::vector<nn::Tensor*> param_gradients() override {
        return {&grad_weight_};
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
        auto grad_input = engine.matmul(weight_, grad_output, true, false);

        // grad_weight += grad_output × input^T
        auto gw = engine.matmul(grad_output, input_cache_, false, true);
        auto r = engine.add_inplace(grad_weight_, *gw);

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

## ⚠️ 常见陷阱

### 1. Tensor 布局错误

```cpp
// ❌ 错误：(batch, feature) — 行主序 batch-major
nn::Tensor input = nn::Tensor::cpu(32, 784);

// ✅ 正确：(feature, batch) — 列主序 batch-major
nn::Tensor input = nn::Tensor::cpu(784, 32);
```

### 2. 忘记绑定 Engine

```cpp
// ❌ 错误：Model 未绑定 Engine
nn::Model model;
model.add_linear(784, 10);  // 崩溃：engine_ == nullptr

// ✅ 正确
nn::CpuEngine engine;
nn::Model model(engine);
model.add_linear(784, 10);
```

### 3. 跨设备操作

```cpp
// ❌ 错误：GPU 张量调用 to_matrix
nn::GpuEngine engine;
nn::Tensor t = engine.create_tensor(10, 10);
auto m = engine.to_matrix(t);  // 错误：tensor is not GPU

// ✅ 正确：先下载到 CPU
auto m = engine.to_matrix(t);  // 错误信息会告诉你
```

### 4. 梯度清零遗漏

```cpp
// ❌ 错误：忘记清零梯度
auto out = model.forward(input);
auto loss = loss_fn.forward(engine, out, target);
auto grad = loss_fn.backward();
model.backward(engine, grad);
optimizer.step();  // 梯度累积！

// ✅ 正确：每步清零
optimizer.zero_grad();  // 或 model.zero_grad(engine)
auto out = model.forward(input);
auto loss = loss_fn.forward(engine, out, target);
auto grad = loss_fn.backward();
model.backward(engine, grad);
optimizer.step();
```

---

## 📋 可用层速查表

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
