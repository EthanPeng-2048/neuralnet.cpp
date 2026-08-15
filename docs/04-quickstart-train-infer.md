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

# 使用 Vulkan GPU 加速
./build/mnist_train --gpu --epochs 10

# 使用 CUDA GPU 加速
./build/mnist_train --cuda --epochs 10

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
| `--gpu` | 禁用 | 启用 Vulkan GPU 加速 |
| `--cuda` | 禁用 | 启用 CUDA GPU 加速（优先于 --gpu） |
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
| `--gpu` | 禁用 | 启用 Vulkan GPU 加速 |
| `--cuda` | 禁用 | 启用 CUDA GPU 加速（优先于 --gpu） |
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
| `--gpu` | 禁用 | 启用 Vulkan GPU 加速 |
| `--cuda` | 禁用 | 启用 CUDA GPU 加速 |
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

---

## 🖥️ 图形化界面 (GUI) 操作指南

项目提供基于 tkinter 的全功能 GUI（`gui.py`），覆盖全部训练/推理/分词器操作，无需记忆命令行参数。

```bash
# 启动 GUI
python gui.py
```

> **前提**: 需要先构建 C++ 项目（`cmake -B build -G Ninja`），GUI 会调用 `build/` 目录下的可执行文件。
> **依赖**: Python 3.8+，tkinter（Python 内置），Pillow（`pip install Pillow`，用于手写板功能）。

### GUI 功能一览

| Tab | 功能 |
|-----|------|
| 🏋️ **训练** | MNIST 模型训练：支持 MLP / Transformer 架构切换、超参数调节、恢复训练、学习率调度（fixed / cosine）、GPU 加速、评估样本数；实时训练曲线 |
| 🔍 **推理** | MNIST 图片推理：Top-K 预测 + 置信度条形图 + 图片预览；内置 ✍️ **手写板**，鼠标书写数字即刻识别 |
| 🖼️ **图片查看** | 浏览 CSV 格式手写数字图片，前后翻页导航 |
| 📝 **GPT 训练** | GPT 语言模型训练：位置编码（learned / sinusoidal / alibi）、激活（GeLU / SwiGLU）、归一化（LayerNorm / RMSNorm）、梯度累积、滑动窗口 stride、学习率调度（fixed / cosine / step_cosine）、TDR 防护、梯度裁剪；实时 loss 曲线 |
| 💬 **GPT 推理** | GPT 文本生成：温度调节、交互模式、Token ID 调试输出 |
| 🔤 **分词器训练** | 训练 BPE / CharBPE 分词器，配置词表大小和最小合并频率 |
| 🔡 **分词器推理** | 分词器编码/解码测试，支持文本编码、ID 解码、文件编码 |

---

### 🏋️ MNIST 训练 Tab

```
┌──────────────────────────────────────────────────┐
│  🏋️ 训练 Tab                                      │
│                                                    │
│  数据集目录:  [datasets/mnist_data    ] [浏览…]     │
│  模型保存路径: [mnist_model.bin        ] [浏览…]     │
│  ☑ 从已有模型恢复训练                                │
│                                                    │
│  ┌─ 超参数 ────────────────────────────────────┐   │
│  │ 轮数: [10]  学习率: [0.001]  批大小: [64]    │   │
│  │ 优化器: [adam ▾]  权重衰减: [0.01]           │   │
│  │ 模型类型: [mlp ▾]                            │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 模型架构 ──────────────────────────────────┐   │
│  │ MLP: 层维度 [784,512,256,128,10]             │   │
│  │ 或 Transformer: 模型维度/头数/层数/FFN/Patch  │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 学习率调度 ────────────────────────────────┐   │
│  │ 调度模式: [fixed ▾] / [cosine ▾]             │   │
│  │ cosine: 预热轮数 + 最低学习率                 │   │
│  │ 手动: 逗号分隔每轮 lr                         │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 训练选项 ──────────────────────────────────┐   │
│  │ GPU: [None ▾]  最大样本数: [0]  评估样本数: [200] │   │
│  │ ☑ 每 epoch 打乱 batch 顺序                    │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  [▶ 开始训练]  [⏹ 停止]                            │
│                                                    │
│  ┌─ 训练日志 ──────────────────────────────────┐   │
│  │ Epoch 1/10  loss=0.4523  train_acc=90.12%   │   │
│  │ Epoch 2/10  loss=0.2134  train_acc=94.56%   │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 📊 训练曲线 ───────────────────────────────┐   │
│  │  ╭──╮                                        │   │
│  │ ╭╯  ╰──╮   step_loss / epoch_loss            │   │
│  │╭╯      ╰─────────   train_acc / test_acc     │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

**操作步骤：**

1. **选择数据集** — 点击"浏览…"选择 `datasets/mnist_data` 目录（默认已填好）
2. **设置保存路径** — 模型训练完成后保存的 `.bin` 文件路径
3. **配置超参数** — 轮数、学习率、批大小、优化器（`sgd`/`adam`/`adamw`/`muon`）、权重衰减
4. **选择模型类型**：
   - `mlp`：可自定义层维度（如 `784,512,256,128,10`）
   - `transformer`：可配置模型维度、注意力头数、FFN 维度、层数、Patch 大小
5. **可选功能**：
   - ✅ 恢复训练 — 从已有模型继续训练
   - ✅ GPU 加速 — 支持 Vulkan / CUDA（下拉选择）
   - ✅ 评估样本数 — 仅 Transformer 架构生效，限制评估样本以加快训练（0=自动）
   - ✅ 学习率调度 — 支持 cosine 衰减（含预热）和手动指定每轮 lr
6. **点击 "▶ 开始训练"** — 日志面板实时输出训练进度，下方曲线图实时绘制 step_loss、epoch_loss、train_acc、test_acc、lr

---

### 🔍 MNIST 推理 Tab

**基本推理：**

1. **选择模型文件** — 点击"浏览…"选择训练好的 `.bin` 模型
2. **选择图片** — 输入单张 CSV 图片路径或包含多张图片的目录
3. **设置 Top-K** — 控制显示前 K 个预测结果
4. **点击 "▶ 开始推理"** — 左侧显示图片预览，右侧显示 Top-K 置信度条形图

**✍️ 手写板识别：**

推理 Tab 内置了手写板功能，无需准备 CSV 文件即可实时识别：

1. 在左侧白色画布上用鼠标**书写一个数字**（0-9）
2. 右侧 **28×28 预览** 窗口实时显示下采样后的效果
3. 点击 **"🔍 识别手写数字"** 按钮
4. 下方显示预测结果（数字 + 置信度百分比）
5. 点击 **"🗑️ 清除"** 重写

```
┌──────────────────────────────────────────────────┐
│  🔍 推理 Tab                                      │
│                                                    │
│  模型文件:  [pretrained/model.bin    ] [浏览…]     │
│  图片/目录: [                        ] [浏览…]     │
│  Top-K: [3]   架构: 自动识别   ☐ GPU  ☐ 显示像素   │
│                                                    │
│  [▶ 开始推理]                                       │
│                                                    │
│  ┌─ ✍️ 手写识别 ──────────────────────────────┐   │
│  │ ┌──────────┐  28×28 预览:                    │   │
│  │ │          │  ┌──────┐                       │   │
│  │ │   ✍️     │  │▓▓▓▓▓▓│                       │   │
│  │ │  画布    │  │▓  ▓▓▓│  预测: 3 (98.2%)     │   │
│  │ │          │  │  ▓▓▓▓│  条形图:              │   │
│  │ └──────────┘  └──────┘  ████░ 3  98.2%      │   │
│  │ [🗑️ 清除] [🔍 识别手写数字]  █░░░░ 7   1.2%   │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 图片预览 ────────┐  ┌─ 预测结果 ───────────┐   │
│  │                   │  │ filename.csv          │   │
│  │   28×28 灰度图    │  │ ████░░ 3  95.3%      │   │
│  │                   │  │ █░░░░░ 5   3.1%      │   │
│  └───────────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────┘
```

---

### 🖼️ 图片查看 Tab

浏览 CSV 格式的手写数字图片：

1. 选择单张 CSV 图片文件或包含多张图片的目录
2. 使用 **上一张 / 下一张** 按钮翻页导航
3. 左侧显示 28×28 灰度图片预览

---

### 📝 GPT 训练 Tab

```
┌──────────────────────────────────────────────────┐
│  📝 GPT 训练 Tab                                   │
│                                                    │
│  训练文本文件: [datasets/llm_corpus.txt ] [浏览…]   │
│  模型保存路径: [gpt_model.bin           ] [浏览…]   │
│  词表 JSON:   [bpe_vocab.json          ] [浏览…]   │
│  ☑ 从已有模型恢复训练                                │
│                                                    │
│  ┌─ 超参数 ────────────────────────────────────┐   │
│  │ 轮数: [10]  学习率: [0.001]  批大小: [32]    │   │
│  │ 序列长度: [256]  优化器: [adam ▾]            │   │
│  │ 权重衰减: [0.01]  日志间隔: [50]             │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  ┌─ 模型架构 ──────────────────────────────────┐   │
│  │ 模型维度: [128]  注意力头: [4]  层数: [4]     │   │
│  │ FFN 维度: [512]                              │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  位置编码 / 激活 / 归一化: [learned ▾] [gelu ▾] [layernorm ▾] │
│  梯度累积: [1]  stride: [0]   GPU: [None ▾]      │
│                                                    │
│  ▶ 高级选项（学习率调度 / TDR 防护 / 梯度日志 / 梯度裁剪） │
│    - 调度: fixed / cosine / step_cosine + 预热 + min-lr │
│    - TDR 重试 + flush 间隔；梯度裁剪 max-norm       │
│                                                    │
│  ☑ 显示梯度统计                                    │
│                                                    │
│  [▶ 开始训练]  [⏹ 停止]                            │
│                                                    │
│  ┌─ 📊 训练曲线 ───────────────────────────────┐   │
│  │  ╭──╮   step_loss / epoch_loss / lr          │   │
│  │ ╭╯  ╰──╮                                     │   │
│  │╭╯      ╰─────────                            │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

**操作步骤：**

1. **选择训练文本** — 点击"浏览…"选择 `.txt` 文本文件（如 `datasets/llm_corpus.txt`）
2. **设置词表** — 指定 BPE 词表 JSON 路径（如 `bpe_vocab.json`）
3. **配置模型架构** — 模型维度、注意力头数、Transformer 层数、FFN 维度
4. **选择位置编码 / 激活 / 归一化** — `learned`/`sinusoidal`/`alibi`，`gelu`/`swiglu`，`layernorm`/`rmsnorm`
5. **配置超参数** — 轮数、学习率、批大小、序列长度、优化器；可开启梯度累积与滑动窗口 stride
6. **高级选项** — 学习率调度（cosine / step_cosine）、TDR 防护、梯度统计、梯度裁剪
7. **点击 "▶ 开始训练"** — 实时显示 loss 曲线

---

### 💬 GPT 推理 Tab

**操作步骤：**

1. **选择模型** — 指定训练好的 GPT 模型 `.bin` 文件
2. **选择词表** — 指定对应的词表 JSON
3. **设置生成参数** — 最大 token 数（默认 200）、温度（0=贪心，>1=更随机，默认 0.8）
4. **输入提示文本** — 在输入框中输入起始文本（如 "Hello"）
5. **点击 "▶ 生成文本"** — 在输出区域显示生成结果

**可选项：**
- ✅ **交互式生成模式** — 启用后通过 stdin 进行多轮对话式生成
- ✅ **显示 Token ID** — 调试用，显示每个 token 的 ID
- ✅ **GPU 加速** — 支持 Vulkan / CUDA（下拉选择）

---

### 🔤 分词器训练 Tab

**操作步骤：**

1. **选择训练文本** — 用于训练分词器的 `.txt` 文件
2. **设置输出路径** — 分词器模型保存的 JSON 文件
3. **选择类型** — `bpe`（字节级 BPE）或 `charbpe`（字符级 BPE，中文推荐）
4. **配置参数** — 词表大小（默认 5000）、最小合并频率（默认 2）
5. **点击 "▶ 开始训练"**

---

### 🔡 分词器推理 Tab

**操作步骤：**

1. **选择词表** — 加载分词器 JSON 文件
2. **选择操作模式**：
   - **编码+解码验证** — 输入文本，查看编码再解码的完整流程
   - **仅编码** — 将文本编码为 token ID 序列
   - **仅解码** — 将 token ID 序列解码为文本
3. **输入内容** — 文本或逗号分隔的 ID 列表（按 Enter 快速执行）
4. **可选** — 勾选"显示 Token 详情"查看字节级信息
5. **点击 "▶ 执行"** — 输出区域显示结果
