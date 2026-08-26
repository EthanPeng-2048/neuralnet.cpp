# neuralnet.cpp

一个从零实现的 C++26 神经网络库，支持 CPU/GPU 双后端、多层 Transformer、GPT 语言模型训练与推理。

## 许可证

本项目采用 [MIT License](LICENSE)，版权归 © 2026 EthanPeng-2048 所有。你可以自由使用、修改、分发和商用本项目的代码，只需保留版权声明。

> **数据集说明**：`scripts/` 下的下载脚本仅负责从第三方来源拉取数据（如 TinyStories、OpenWebText、中文 wiki 等），这些数据集各自拥有独立的许可证，不随本项目代码的 MIT 许可分发。使用时请自行遵守相应数据集的许可条款。

## 🤖 AI 工具致谢

本项目的部分代码由 AI 辅助生成。当前使用（且目前限于）以下 AI 工具：

- **DeepSeek V4 系列**
- **Mi Mimo v2.5 系列**
- **VS Code Copilot**

> AI 仅作为辅助工具，所有代码的所有权与最终版权归本项目作者（EthanPeng-2048）所有，不影响上述 MIT 许可证的效力。

## �📚 文档

| 文档 | 说明 |
|------|------|
| [架构设计](docs/01-architecture.md) | 项目分层架构、引擎化设计、数据流、模块详解 |
| [性能优化](docs/02-performance.md) | SmartPolicy、线程池、缓存分块、GPU 加速、算子融合等 |
| [快速上手：构建模型](docs/03-quickstart-model.md) | ComputeEngine/Layer/Model 三件套使用教程 |
| [快速上手：训练与推理](docs/04-quickstart-train-infer.md) | MNIST/GPT 训练推理命令行 + C++ API 示例 + GUI 操作指南 |
| [算法解析](docs/05-algorithm-reference.md) | 每个 Layer/Loss/Optimizer 的数学原理与原语分解 |
| [CUDA 后端](docs/06-cuda-backend.md) | CUDA GPU 加速后端设计、构建方法、编译器兼容性 |
| [训练包](docs/07-train-package.md) | 用 `.nnpkg` 打包超参+训练集，跨设备一键复现训练 |
| [开发规范](docs/DEVELOPMENT_STANDARDS.md) | C++ 编码规范、模块隔离、内存管理 |

## 项目结构

```
neuralnet.cpp/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── 01-architecture.md       ← 架构设计文档
│   ├── 02-performance.md        ← 性能优化文档
│   ├── 03-quickstart-model.md   ← 模型构建教程
│   ├── 04-quickstart-train-infer.md ← 训练推理教程
│   ├── 05-algorithm-reference.md    ← 算法解析参考
│   ├── 06-cuda-backend.md       ← CUDA 后端设计
│   └── DEVELOPMENT_STANDARDS.md ← 开发规范
├── gui.py                       ← 图形化操作界面 (CustomTkinter)
├── include/neuralnet.cpp/
│   ├── nn.hpp                   ← 统一入口头文件
│   ├── core_config.hpp          ← SmartPolicy、BLOCK_SIZE
│   ├── compute_tensor.hpp       ← 统一跨设备张量
│   ├── compute_engine.hpp       ← 引擎抽象接口
│   ├── compute_cpu_engine.hpp   ← CPU 引擎
│   ├── compute_gpu_engine.hpp   ← GPU 引擎 (Vulkan)
│   ├── compute_cuda_engine.hpp  ← GPU 引擎 (CUDA, 已停用)
│   ├── compute_layer.hpp        ← Layer 聚合头
│   ├── compute_layer_{base,mlp,conv,softmax,attention,feedforward,transformer,gpt,zipt,rapt}.hpp ← 各层域
│   ├── compute_loss.hpp         ← 损失函数
│   ├── compute_optimizer.hpp    ← 优化器 (SGD/Adam/AdamW/Muon)
│   ├── model_container.hpp      ← Model 容器
│   ├── model_spec.hpp           ← 架构描述
│   ├── model_serialization.hpp  ← 二进制序列化
│   ├── model_keyvalue_record.hpp ← 自描述键值记录
│   ├── domain_mnist.hpp         ← MNIST 模型工厂
│   ├── domain_gpt.hpp           ← GPT 模型工厂
│   ├── domain_tokenizer.hpp     ← 分词器
│   ├── algebra_matrix.hpp       ← 矩阵类
│   ├── algebra_expr.hpp         ← 表达式模板
│   ├── algebra_ops.hpp          ← 逐元素算子
│   ├── algebra_span.hpp         ← Span 抽象
│   ├── algebra_compute.hpp      ← 计算分派
│   ├── core_threadpool.hpp      ← 全局线程池
│   ├── core_errors.hpp          ← Result<T>
│   ├── core_assert.hpp          ← 断言宏
│   ├── core_file.hpp            ← 文件工具
│   ├── core_observer_ptr.hpp    ← 观察者指针
│   └── backend/                 ← GPU 后端 (Vulkan)
├── src/
│   ├── mnist_train.cpp          ← MNIST 训练
│   ├── mnist_infer.cpp          ← MNIST 推理
│   ├── text_train.cpp           ← GPT 文本训练
│   ├── text_infer.cpp           ← GPT 文本推理
│   ├── tokenizer_train.cpp      ← 分词器训练
│   ├── tokenizer_infer.cpp      ← 分词器推理
│   └── gpu_test.cpp             ← GPU 后端测试
├── datasets/                    ← 训练数据
├── pretrained/                  ← 预训练模型
├── scripts/                     ← 数据处理/训练辅助脚本
├── shaders/                     ← GPU Compute Shader
└── compare_with_torch/          ← PyTorch 对比实现
```

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/EthanPeng-2048/neuralnet.cpp)

## 依赖

- **编译器**: LLVM Clang++ 22.1+（`C:/Program Files/LLVM/bin/clang++.exe`）
- **C++ 标准**: C++26
- **构建工具**: CMake 4.x Ninja
- **GUI** (可选): Python 3.10+，customtkinter（`pip install customtkinter`），Pillow（`pip install Pillow`）

## 构建与运行

### 构建 C++ 项目

```powershell
cmake -B build -G Ninja
```

### 准备数据

- MNIST 数据：`datasets/mnist_data/`（`train.csv` / `test.csv`）
- 文本语料：使用 `scripts/` 下的 `download_*.py` 脚本下载

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

基于 CustomTkinter 的全功能图形化界面，覆盖 MNIST 训练/推理、GPT 训练/推理、分词器训练/推理，无需记忆命令行参数。

```bash
# 启动 GUI
python gui.py
```

> **前提**: 需要先构建 C++ 项目（`cmake -B build -G Ninja`），GUI 会调用 `build/` 目录下的可执行文件。

#### GUI 功能一览

| Tab | 功能 |
|-----|------|
| 🏋️ **MNIST 训练** | MNIST 模型训练：MLP / Transformer 架构、超参数调节、恢复训练、学习率调度（fixed / cosine）、GPU 加速、评估样本数；实时 loss / acc 曲线 |
| 🔍 **MNIST 推理** | MNIST 图片推理 + 图片查看：推理某张图片自动显示该图、Top-K 置信度条形图、目录翻页导航；内置 ✍️ **手写板**，鼠标书写数字即刻识别 |
| 📝 **GPT 训练** | GPT 语言模型训练：位置编码（learned / sinusoidal / alibi）、激活（GeLU / SwiGLU）、归一化（LayerNorm / RMSNorm）、梯度累积、滑动窗口 stride、学习率调度（fixed / cosine / step_cosine）、TDR 防护、梯度裁剪；实时 loss 曲线 |
| 💬 **GPT 推理** | GPT 文本生成：温度调节、交互模式、Token ID 调试输出 |
| 🔤 **分词器训练** | 训练 BPE / CharBPE 分词器，配置词表大小和最小合并频率 |
| 🔡 **分词器推理** | 分词器编码/解码测试，支持文本编码、ID 解码、文件编码 |

#### GUI 操作流程

**MNIST 训练：**
1. 选择数据集目录（默认 `datasets/mnist_data`）和模型保存路径
2. 配置超参数：轮数、学习率、批大小、优化器（`sgd`/`adam`/`adamw`/`muon`）
3. 选择模型类型：`mlp`（自定义层维度）或 `transformer`（模型维度/头数/层数/FFN/Patch）
4. 可选：恢复训练、GPU 加速、学习率调度、振荡抑制
5. 点击 **"▶ 开始训练"**，日志面板实时输出，下方曲线图绘制 loss / accuracy
6. 训练中可随时点击 **"⏹ 停止"** 终止

**MNIST 推理：**
1. 选择训练好的模型文件（`.bin`），设置 Top-K 与 GPU
2. 选择单张 CSV 图片或包含多张图片的目录
3. 点击 **"🔍 推理当前图片"** → 自动显示该图片 + Top-K 置信度条形图；目录模式可用 **上一张 / 下一张** 翻页浏览（已推理过的图片会缓存预测）
4. 点击 **"▶ 推理整个目录"** 可批量推理并逐个自动显示
5. **手写板模式**：在白色画布上书写数字 → 点击 **"🔍 识别手写数字"**，即刻得到预测

**GPT 训练：**
1. 选择训练文本文件（`.txt`）和词表 JSON
2. 配置模型架构（维度/头数/层数/FFN）和位置编码
3. 点击 **"▶ 开始训练"**，实时显示 loss 曲线

**GPT 推理：**
1. 选择模型文件和词表，设置温度和最大 token 数
2. 输入提示文本，点击 **"▶ 生成文本"**
3. 支持勾选"交互式生成模式"进行多轮生成

**分词器训练/推理：**
- 训练：选择文本文件 → 设置类型（`bpe`/`charbpe`）和词表大小 → 点击训练
- 推理：选择词表 → 输入文本或 ID → 选择编码/解码 → 点击执行

## 网络结构

支持两种模型架构（通过 `--model-type` 切换），定义在 `mnist_common.hpp`：

### MLP（默认）

归一化层可通过 `--norm layernorm|rmsnorm|batchnorm` 切换（默认 LayerNorm）：

```
输入 (784)
→ Linear(784 → 512) + Norm + GeLU
→ Linear(512 → 256) + Norm + GeLU
→ Linear(256 → 128) + Norm + GeLU
→ Linear(128 → 64)  + Norm + GeLU
→ Linear(64  → 10)
→ CrossEntropy Loss (含 Softmax)
```

- `LayerNorm`：按特征维归一化（GPT-2 风格，默认）
- `RMSNorm`：无均值、更轻量（LLaMA 风格）
- `BatchNorm`：按 batch 维归一化，训练时更新 running 统计、推理时使用（MLP 专用）

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
