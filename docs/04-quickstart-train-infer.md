# 🚀 快速上手：训练与推理

> 本教程教你如何使用 neuralnet.cpp 训练和推理 MNIST 手写数字识别与 GPT 文本生成模型。

---

## 📋 前置条件

**构建项目：**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**数据准备：**

```bash
# MNIST 数据集会自动下载到 datasets/mnist_data/
# GPT 训练需要文本文件（如 datasets/llm_corpus.txt）
```

---

## 🖼️ MNIST 手写数字识别

### 训练

```bash
# MLP 架构（默认）
./build/mnist_train --epochs 10 --lr 0.001 --optimizer adam

# Transformer (ViT) 架构
./build/mnist_train --arch transformer --epochs 20 --lr 0.001

# 使用 GPU 加速
./build/mnist_train --gpu --epochs 10

# 恢复训练
./build/mnist_train --resume mnist_model.bin --epochs 5

# 快速测试（限制样本数）
./build/mnist_train --max-samples 1000 --epochs 3
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `mlp` | 架构：`mlp` / `transformer` |
| `--resume <path>` | 无 | 从已有模型恢复训练 |
| `--save <path>` | `mnist_model.bin` | 模型保存路径 |
| `--dataset <path>` | `datasets/mnist_data` | 数据集目录 |
| `--epochs <n>` | `10` | 训练轮数 |
| `--lr <lr>` | `0.001` | 学习率 |
| `--batch-size <n>` | `64` | 批大小 |
| `--optimizer <name>` | `adam` | 优化器：`sgd`/`sgd_momentum`/`adam`/`adamw`/`muon` |
| `--gpu` | 禁用 | 启用 GPU 加速 |
| `--max-samples <n>` | 全部 | 限制训练样本数 |

### 推理

```bash
# 使用已训练的模型推理
./build/mnist_infer --model mnist_model.bin --image datasets/mnist_data/test/0_1.csv

# 交互模式（输入文件路径）
./build/mnist_infer --interactive
```

---

## 📝 GPT 文本生成

### 训练

```bash
# 基础训练
./build/text_train datasets/llm_corpus.txt --epochs 10 --lr 0.001

# 自定义模型大小
./build/text_train datasets/llm_corpus.txt \
    --d-model 256 --num-heads 8 --num-layers 6 --d-ff 1024 \
    --seq-len 512 --batch-size 16 --epochs 20

# 使用自定义词表
./build/text_train datasets/llm_corpus.txt --vocab my_bpe.json

# 使用 Muon 优化器
./build/text_train datasets/llm_corpus.txt --optimizer muon --lr 0.002

# 恢复训练
./build/text_train datasets/llm_corpus.txt --resume gpt_model.bin --epochs 5
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `<text-file>` | 必需 | 训练文本文件路径 |
| `--save <path>` | `gpt_model.bin` | 模型保存路径 |
| `--resume <path>` | 无 | 从已有模型恢复训练 |
| `--vocab <path>` | `gpt_bpe.json` | 词表 JSON 路径（自动识别类型） |
| `--epochs <n>` | `10` | 训练轮数 |
| `--lr <lr>` | `0.001` | 学习率 |
| `--batch-size <n>` | `32` | 批大小 |
| `--seq-len <n>` | `256` | 序列长度 |
| `--optimizer <name>` | `adam` | 优化器 |
| `--weight-decay <w>` | `0.01` | AdamW 权重衰减 |
| `--d-model <n>` | `128` | 模型维度 |
| `--num-heads <n>` | `4` | 注意力头数 |
| `--num-layers <n>` | `4` | Transformer 层数 |
| `--d-ff <n>` | `512` | FFN 中间维度 |
| `--gpu` | 禁用 | 启用 GPU 加速 |
| `--log-interval <n>` | `50` | 日志间隔 |

### 推理

```bash
# 单次生成
./build/text_infer --prompt "Once upon a time" --max-tokens 200

# 交互模式
./build/text_infer --interactive

# 调节温度（0=贪心，>1=更随机）
./build/text_infer --prompt "Hello" --temperature 0.8

# 使用 GPU
./build/text_infer --gpu --prompt "Hello"
```

**完整参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model <path>` | `gpt_model.bin` | 模型文件路径 |
| `--vocab <path>` | `bpe_vocab.json` | 词表路径（仅模型未嵌入时） |
| `--prompt <text>` | `Hello` | 输入提示文本 |
| `--interactive` | 禁用 | 交互式生成模式 |
| `--max-tokens <n>` | `200` | 最大生成 token 数 |
| `--temperature <t>` | `1.0` | 温度参数（0=贪心） |
| `--gpu` | 禁用 | 启用 GPU 加速 |
| `--show-tokens` | 禁用 | 显示 token ID（调试） |

---

## 🔤 分词器

### 训练分词器

```bash
# 训练 BPE 分词器
./build/tokenizer_train --input datasets/llm_corpus.txt --vocab-size 10000 --output my_bpe.json

# 训练 ByteZip 分词器
./build/tokenizer_train --type bytezip --input datasets/llm_corpus.txt --output my_bytezip.json
```

### 分词器推理

```bash
# 编码文本
./build/tokenizer_infer --vocab gpt_bpe.json --encode "Hello world"

# 解码 token IDs
./build/tokenizer_infer --vocab gpt_bpe.json --decode "72,101,108,108,111"
```

### 支持的分词器类型

| 类型 | JSON `"type"` 字段 | 说明 |
|------|---------------------|------|
| BPE | `"bpe"` | Byte-Pair Encoding |
| CharBPE | `"charbpe"` | 字符级 BPE |
| WordZip | `"wordzip"` | 词频统计分词 |
| Space | `"space"` | 空格分词 |

V3 格式模型文件会自动嵌入分词器，推理时无需单独指定 `--vocab`。

---

## 💻 C++ API 编程训练

### MNIST MLP 训练

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_mnist.hpp>

int main() {
    nn::CpuEngine engine;

    // 1. 构建模型
    auto model_result = nn::build_mnist_mlp_model(engine);
    nn::Model model = std::move(*model_result);

    // 2. 优化器
    auto optimizer = nn::create_optimizer(
        "adam", engine,
        model.parameters(), model.param_gradients(),
        0.001,  // lr
        0.01    // weight_decay
    );

    nn::CrossEntropyLoss loss_fn;

    // 3. 加载数据集（CSV 格式）
    // ... 假设已有 train_images (N×784) 和 train_labels (N×1) ...

    const std::size_t batch_size = 64;
    const std::size_t epochs = 10;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        Scalar epoch_loss = 0;

        for (std::size_t batch = 0; batch < num_batches; ++batch) {
            // 清零梯度
            optimizer->zero_grad();

            // 准备 batch 数据 (784, batch_size)
            nn::Matrix x_batch = /* ... */;
            nn::Matrix y_onehot = /* ... (10, batch_size) ... */;

            // 上传到引擎
            auto x_t = engine.from_matrix(x_batch);
            auto y_t = engine.from_matrix(y_onehot);
            nn::Tensor input = std::move(*x_t);
            nn::Tensor target = std::move(*y_t);

            // 前向传播
            auto out = model.forward(input);
            auto loss = loss_fn.forward(engine, *out, target);

            // 反向传播
            auto grad = loss_fn.backward();
            model.backward(engine, *grad);

            // 参数更新
            optimizer->step();

            epoch_loss += *loss;
        }

        std::cout << "Epoch " << epoch + 1
                  << "  loss=" << epoch_loss / num_batches << "\n";
    }

    // 4. 保存模型
    auto spec = nn::make_mlp_spec({784, 512, 256, 128, 64, 10});
    auto save_result = nn::save_model("mnist_model.bin", model, spec);
}
```

### GPT 文本生成训练

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>

int main() {
    nn::CpuEngine engine;

    // 1. 加载分词器
    auto tokenizer = nn::load_tokenizer_from_file("gpt_bpe.json");

    // 2. 构建 GPT 模型
    auto model_result = nn::build_gpt_model(
        engine,
        tokenizer->vocab_size(),  // vocab_size
        128,    // d_model
        256,    // seq_len
        4,      // num_heads
        512,    // d_ff
        4       // num_layers
    );
    nn::Model model = std::move(*model_result);

    // 3. 优化器
    auto optimizer = nn::create_optimizer(
        "adamw", engine,
        model.parameters(), model.param_gradients(),
        0.001, 0.01
    );

    nn::CrossEntropyLoss loss_fn;

    // 4. 编码文本 → token IDs
    std::string text = /* ... 读取文本文件 ... */;
    auto tokens = tokenizer->encode(text);

    // 5. 训练循环（next-token prediction）
    const std::size_t seq_len = 256;
    const std::size_t batch_size = 32;

    for (int epoch = 0; epoch < 10; ++epoch) {
        for (std::size_t step = 0; step < steps_per_epoch; ++step) {
            optimizer->zero_grad();

            // 采样 batch: x = tokens[t:t+seq_len], y = tokens[t+1:t+seq_len+1]
            nn::Matrix x_tokens(seq_len, batch_size);  // (seq, batch)
            nn::Matrix y_onehot(vocab_size, seq_len * batch_size);
            // ... 填充数据 ...

            auto x_t = engine.from_matrix(x_tokens);
            auto y_t = engine.from_matrix(y_onehot);
            nn::Tensor input = std::move(*x_t);
            nn::Tensor target = std::move(*y_t);

            // 前向传播
            auto logits = model.forward(input);
            auto loss = loss_fn.forward(engine, *logits, target);

            // 反向传播 + 更新
            auto grad = loss_fn.backward();
            model.backward(engine, *grad);
            optimizer->step();

            std::cout << "Step " << step << "  loss=" << *loss << "\n";
        }
    }

    // 6. 保存模型（含嵌入词表）
    auto spec = nn::make_gpt_spec(
        tokenizer->vocab_size(), 128, 256, 4, 512, 4);
    nn::save_model("gpt_model.bin", model, spec,
                   tokenizer.get());  // 嵌入分词器
}
```

### GPT 推理（自回归生成）

```cpp
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/model_serialization.hpp>
#include <neuralnet.cpp/domain_gpt.hpp>

int main() {
    nn::CpuEngine engine;

    // 1. 加载模型（V3 格式自动读取规格 + 嵌入词表）
    auto load_result = nn::load_model_with_spec("gpt_model.bin", engine);
    auto& [model, spec, tokenizer] = *load_result;

    // 2. 编码提示文本
    auto prompt_tokens = tokenizer->encode("Once upon a time");

    // 3. 自回归生成
    const int max_tokens = 200;
    const double temperature = 1.0;

    std::vector<std::size_t> generated(prompt_tokens.begin(), prompt_tokens.end());

    for (int i = 0; i < max_tokens; ++i) {
        // 构造输入: (seq_len, 1) — 取最后 seq_len 个 token
        nn::Matrix input(spec.seq_len, 1);
        // ... 填充 token（不足部分用 pad_id=0）...

        auto x_t = engine.from_matrix(input);
        auto logits = model.forward(*x_t);

        // 取最后一个位置的 logits
        auto logits_m = engine.to_matrix(*logits);
        // logits_m: (vocab_size, seq_len*1)
        // 最后一个位置: col = seq_len - 1

        // 温度采样
        std::vector<Scalar> probs(vocab_size);
        Scalar max_logit = *std::max_element(/* last position logits */);
        Scalar sum = 0;
        for (std::size_t v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp((logits_m->at(v, seq_len-1) - max_logit) / temperature);
            sum += probs[v];
        }
        for (auto& p : probs) p /= sum;

        // 采样下一个 token
        std::size_t next_token = /* argmax 或随机采样 */;
        generated.push_back(next_token);

        if (next_token == tokenizer->eos_id()) break;
    }

    // 4. 解码输出
    std::string output = tokenizer->decode(generated);
    std::cout << output << "\n";
}
```

---

## 📊 训练监控

### 控制台输出

训练过程中会打印：

```
[Step 50]  loss=2.345  lr=0.001000
[Step 100] loss=1.987  lr=0.001000
[Step 150] loss=1.654  lr=0.001000
...
```

### 模型保存格式

| 版本 | 格式 | 说明 |
|------|------|------|
| V1 | `[magic][version][matrices...]` | 仅参数 |
| V2 | `[magic][version][model_type][spec][matrices...]` | 含架构规格 |
| V3 | `[magic][version][precision][model_type][spec][matrices...][tokenizer]` | 含精度标记 + 嵌入词表 |

- `load_model` 同时支持 V1/V2/V3 读取
- `save_model` 统一写入 V3 格式
- V3 模型推理时无需单独指定 `--vocab` 参数
