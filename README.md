# neuralnet.cpp

一个轻量级 C++26 神经网络库，从头实现（无第三方深度学习框架依赖），支持并行计算。

## 项目结构

```
neuralnet.cpp/
├── .gitignore
├── CMakeLists.txt
├── ARCHITECTURE.md          ← 架构文档
├── DEVELOPMENT_STANDARDS.md ← 开发规范
├── README.md
├── gui.py                   ← 图形化操作界面 (Tkinter)
├── csv_png.py               ← CSV 转 PNG 图像
├── extract_digits.py        ← 从 MNIST CSV 提取单个数字
├── save_dataset.py          ← 下载 MNIST 数据集
├── build/                   ← CMake 构建输出
│   ├── mnist_train.exe
│   └── mnist_infer.exe
├── include/
│   └── neuralnet.cpp/
│       ├── nn.hpp              ← 统一入口头文件
│       ├── nn_config.hpp       ← SmartPolicy、BLOCK_SIZE 等配置
│       ├── layer.hpp           ← Layer 基类 + Linear/ReLU/GeLU/LayerNorm/Softmax/MultiHeadAttention/CausalSelfAttention/GPTModel
│       ├── loss.hpp            ← Loss 基类 + MSELoss/CrossEntropyLoss
│       ├── optimizer.hpp       ← SGD / SGDWithMomentum / Adam
│       ├── model.hpp           ← Model 容器（链式 add<>）
│       ├── model_io.hpp        ← 二进制模型序列化
│       ├── model_spec.hpp      ← ModelSpec 纯数据描述
│       ├── mnist_common.hpp    ← MNIST 常量与 build_mnist_model()
│       ├── gpt_common.hpp      ← GPT 工厂与超参数
│       ├── tokenizer.hpp       ← BPE/Space 分词器
│       ├── algebra/            ← L1 代数层
│       │   ├── matrix.hpp      ← Matrix 类 + 块并行 matmul
│       │   ├── expr.hpp        ← 表达式模板
│       │   ├── ops.hpp         ← ReLU/GeLU 等逐元素算子
│       │   ├── span.hpp        ← Span 抽象
│       │   └── compute_dispatch.hpp ← 计算分派
│       └── core/               ← L0 硬件层
│           ├── thread_pool.hpp ← 全局线程池
│           ├── errors.hpp      ← Result<T> = std::expected<T, Error>
│           └── assert.hpp      ← 断言宏
├── src/
│   ├── mnist_train.cpp         ← MNIST 训练
│   ├── mnist_infer.cpp         ← MNIST 推理
│   ├── text_train.cpp          ← GPT 文本训练
│   ├── text_infer.cpp          ← GPT 文本推理
│   ├── tokenizer_train.cpp     ← BPE 分词器训练
│   ├── tokenizer_infer.cpp     ← BPE 分词器推理
│   └── compute_bench.cpp       ← 计算性能基准测试
├── datasets/
│   ├── mnist_data/          ← MNIST CSV 数据
│   └── test/                ← 按数字分类的测试图片
└── pretrained/
    └── mnist_model.bin      ← 预训练模型
```

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/EthanPeng-2048/neuralnet.cpp)

## 依赖

- **编译器**: LLVM Clang++ 22.1+（`C:/Program Files/LLVM/bin/clang++.exe`）
- **C++ 标准**: C++26
- **构建工具**: CMake 4.x Ninja
- **GUI** (可选): Python 3.8+，tkinter（内置），Pillow（`pip install Pillow`）

## 构建与运行

### 构建 C++ 项目

```powershell
cmake -B build -G Ninja
```

### 准备数据

```bash
#运行下载脚本
pip install pillow
python save_dataset.py
```

### 运行训练

```bash
# 从头开始训练 MLP（默认: 10轮, 学习率0.001, 批大小64, Adam优化器）
./build/mnist_train

# 训练 Transformer 模型
./build/mnist_train --model-type transformer --epochs 20

# 从已有模型恢复训练
./build/mnist_train --resume pretrained/mnist_model.bin

# 自定义参数
./build/mnist_train --epochs 20 --lr 0.001 --batch-size 32 --optimizer adam --save my_model.bin

# 支持的优化器: sgd / sgd_momentum / adam
./build/mnist_train --optimizer sgd_momentum
```

### 运行推理

```bash
# 推理单张图片
./build/mnist_infer datasets/test/0/digit_0001.csv

# 批量推理目录下所有 CSV
./build/mnist_infer datasets/test/0/

# 指定模型文件和类型
./build/mnist_infer image.csv --model pretrained/model.bin --model-type transformer

# 显示 Top-5 预测结果
./build/mnist_infer image.csv --topk 5

# 调试模式：显示像素矩阵
./build/mnist_infer image.csv --show-pixels
```

### 🖥️ 图形化界面 (GUI)

提供了一个基于 tkinter 的图形化界面，方便进行训练、推理和图片查看操作。

```bash
# 启动 GUI
python gui.py
```

> **前提**: 需要先构建 C++ 项目（`cmake -B build -G Ninja`），GUI 会调用 build 目录下的可执行文件。

#### GUI 功能一览

| Tab | 功能 |
|-----|------|
| 🏋️ **训练** | 配置数据集路径、模型保存路径、轮数、学习率、批大小、优化器、模型类型（MLP/Transformer），支持恢复训练；实时显示训练日志 |
| 🔍 **推理** | 选择模型文件和图片 CSV 文件/目录，支持 MLP/Transformer 模型类型切换，显示 Top-K 预测结果置信度条形图和手写数字图片预览；内置 ✍️ 手写板，可直接在画布上书写数字进行实时识别 |
| 🖼️ **图片查看** | 浏览单张 CSV 图片或批量加载目录，支持前后翻页导航 |

#### GUI 操作流程

**训练模式：**
1. 配置数据集路径、模型保存路径等参数
2. 选择模型类型（`mlp` 或 `transformer`）和优化器
3. 可选勾选"从已有模型恢复训练"并指定模型路径
4. 点击"▶ 开始训练"，训练日志实时滚动输出
5. 训练过程中可随时点击"⏹ 停止"终止训练

**推理模式：**
1. 选择模型文件（`.bin`）并指定模型类型（需与训练时一致）
2. 通过文件选择或直接输入路径指定 CSV 图片文件/目录
3. 设置 Top-K 值控制显示预测结果数量
4. 点击"▶ 开始推理"，结果以置信度条形图 + 图片预览形式展示
5. 也可使用内置手写板：在画布上书写数字 → 点击"🔍 识别手写数字" → 自动下采样为 28×28 并推理

## 网络结构

支持两种模型架构（通过 `--model-type` 切换），定义在 `mnist_common.hpp`：

### MLP（默认）

```
输入 (784)
→ Linear(784 → 512) + LayerNorm + GeLU
→ Linear(512 → 256) + LayerNorm + GeLU
→ Linear(256 → 128) + LayerNorm + GeLU
→ Linear(128 → 64)  + LayerNorm + GeLU
→ Linear(64  → 10)
→ CrossEntropy Loss (含 Softmax)
```

### Transformer（ViT-like）

```
输入 (784) — 展平的 28×28 图像
→ PatchEmbedding: 28×28 → 16 个 7×7 patch → 投影到 d_model=64
→ TransformerEncoder ×2 层:
    ├── MultiHeadAttention(d_model=64, heads=4)
    ├── LayerNorm + FeedForward(d_model=64 → d_ff=128 → 64)
    └── 残差连接 + 位置编码
→ 池化 (seq_len → 1)
→ Linear(64 → 10)
→ CrossEntropy Loss (含 Softmax)
```

## 提供的组件

| 组件 | 说明 |
|------|------|
| `nn::Matrix` | 行主序矩阵，支持并行加/减/乘/转置、预分配缓冲区、`std::span` 访问 |
| `nn::Model` | 网络容器，支持链式 `add<LayerType>()` 构建 |
| `nn::Linear` | 全连接层（含 Xavier 均匀初始化、融合 bias 计算） |
| `nn::ReLU` | ReLU 激活函数 |
| `nn::GeLU` | QuickGeLU 激活函数（`x · σ(1.702x)`） |
| `nn::LayerNorm` | 层归一化（含可学习 γ/β 参数） |
| `nn::Softmax` | Softmax 激活层（数值稳定，含反向传播） |
| `nn::MultiHeadAttention` | 多头注意力层（Transformer 用，`d_model × seq_len` 输入输出） |
| `nn::PositionalEncoding` | 正弦/余弦位置编码（Transformer 用） |
| `nn::MSELoss` | 均方误差损失 |
| `nn::CrossEntropyLoss` | 交叉熵损失（含数值稳定 Softmax，`loss.hpp` 中定义） |
| `nn::SGD` | 随机梯度下降优化器 |
| `nn::SGDWithMomentum` | 动量 SGD 优化器 |
| `nn::Adam` | Adam 优化器（一阶/二阶矩估计） |
| `nn::SmartPolicy` | 自适应并行策略：小矩阵串行，大矩阵线程池并行 |
| `nn::ThreadPool` | 全局单例线程池（懒初始化） |
| `nn::save_model` / `nn::load_model` | 二进制模型序列化 |

## 📐 开发规范

本项目遵循严格的 C++ 开发规范，详见 [DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)。

### 核心原则

1. **零手动内存管理** - 完全消除显式指针操作和手动内存管理
2. **模块化设计** - 清晰的职责分离，简洁的接口设计
3. **高性能优先** - 预分配、缓存友好、并行化
4. **紧跟最新标准** - 始终使用最新的 C++ 标准（当前：C++26）

### 快速参考

| 规范 | 要求 |
|------|------|
| 内存管理 | 禁止 `new`/`delete`，使用 `std::vector`、`std::unique_ptr`、`std::span` |
| 接口设计 | 流式 API、两级访问（安全 + 快速） |
| 性能优化 | 预分配缓冲区、缓存友好分块算法、智能并行化 |
| C++ 标准 | C++26，积极使用 ranges、concepts、std::print 等新特性 |
| 命名规范 | 类名 CamelCase，函数/变量 snake_case，私有成员尾部下划线 |
