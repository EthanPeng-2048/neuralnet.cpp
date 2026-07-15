# 🧠 neuralnet.cpp — 代码架构与运行流程

> 一个从零实现的 C++26 MNIST 手写数字识别神经网络，支持多线程加速、Adam/SGD 优化器、以及 Python GUI 交互界面。

---

## 📐 系统总体架构

```mermaid
graph TB
    subgraph "🖥️ 用户入口层"
        A["gui.py<br/>Python GUI 入口"]
        B["src/train.cpp<br/>训练入口"]
        C["src/infer.cpp<br/>推理入口"]
    end

    subgraph "🧩 核心库 (include/neuralnet.cpp/)"
        D["nn.hpp<br/>统一头文件"]
        E["model.hpp<br/>网络容器"]
        F["layer.hpp<br/>层定义"]
        G["loss.hpp<br/>损失函数"]
        H["optimizer.hpp<br/>优化器"]
        I["matrix.hpp<br/>矩阵运算"]
        J["model_io.hpp<br/>模型序列化"]
        K["nn_config.hpp<br/>配置策略"]
        L["thread_pool.hpp<br/>线程池"]
    end

    subgraph "💾 数据层"
        M["datasets/mnist_data/<br/>MNIST CSV 数据"]
        N["pretrained/<br/>预训练模型"]
    end

    A -->|"subprocess 调用"| B
    A -->|"subprocess 调用"| C
    B --> D
    C --> D
    D --> E
    D --> F
    D --> G
    D --> H
    E --> F
    F --> I
    G --> I
    H --> I
    I --> K
    K --> L
    J --> E
    B --> M
    B -->|"save_model()"| N
    C -->|"load_model()"| N
    C --> M
```

---

## 🏗️ 分层架构详解

```mermaid
graph LR
    subgraph "应用层"
        T["train.cpp<br/>训练流程"]
        I["infer.cpp<br/>推理流程"]
        P["gui.py<br/>图形界面"]
    end

    subgraph "接口层"
        M["Model<br/>网络容器"]
        IO["ModelIO<br/>模型持久化"]
    end

    subgraph "计算层"
        L["Linear Layer<br/>全连接层"]
        R["ReLU Layer<br/>激活层"]
        CE["CrossEntropy<br/>交叉熵损失"]
        OPT["Adam / SGD<br/>优化器"]
    end

    subgraph "基础层"
        MX["Matrix<br/>矩阵运算"]
        TP["ThreadPool<br/>并行执行"]
        CFG["SmartPolicy<br/>自适应调度"]
    end

    T --> M
    T --> IO
    T --> CE
    T --> OPT
    I --> M
    I --> IO
    P --> T
    P --> I
    M --> L
    M --> R
    L --> MX
    R --> MX
    CE --> MX
    OPT --> MX
    MX --> CFG
    CFG --> TP
```

---

## 🔄 训练流程 (train.cpp)

```mermaid
flowchart TD
    START(["🚀 程序启动"]) --> PARSE["解析命令行参数<br/>--epochs, --batch-size,<br/>--lr, --optimizer"]
    PARSE --> LOAD_DATA["加载 CSV 数据<br/>MNIST 784 维像素"]
    LOAD_DATA --> SPLIT["分割训练集/测试集"]
    SPLIT --> BUILD["构建网络模型<br/>Linear(784→512)→LayerNorm→GeLU<br/>→ Linear(512→256)→LayerNorm→GeLU<br/>→ Linear(256→128)→LayerNorm→GeLU<br/>→ Linear(128→64)→LayerNorm→GeLU<br/>→ Linear(64→10)"]
    BUILD --> LOAD_PREV{"有预训练模型?"}
    LOAD_PREV -->|"是"| LOAD_M["加载预训练权重"]
    LOAD_PREV -->|"否"| INIT["Xavier 初始化权重"]
    LOAD_M --> TRAIN_LOOP
    INIT --> TRAIN_LOOP

    subgraph TRAIN_LOOP["🔁 训练循环 (per epoch)"]
        BATCH["取一个 batch<br/>(默认 64 样本)"]
        BATCH --> FORWARD["⚡ 前向传播<br/>model.forward(X)"]
        FORWARD --> LOSS["📊 计算损失<br/>CrossEntropyLoss"]
        LOSS --> BACKWARD["📉 反向传播<br/>model.backward(grad)"]
        BACKWARD --> UPDATE["🔄 更新参数<br/>optimizer.step()"]
        UPDATE --> CLR["清除梯度<br/>optimizer.zero_grad()"]
        CLR --> CHECK{"还有 batch?"}
        CHECK -->|"是"| BATCH
        CHECK -->|"否"| EVAL
    end

    EVAL["📈 评估测试集准确率"]
    EVAL --> SAVE["💾 保存模型<br/>save_model()"]
    SAVE --> DONE(["✅ 训练完成"])
```

---

## 🔮 推理流程 (infer.cpp)

```mermaid
flowchart TD
    START(["🚀 推理启动"]) --> LOAD["加载模型<br/>load_model()"]
    LOAD --> INPUT{"输入来源?"}
    INPUT -->|"单文件"| READ_CSV["读取 CSV 图像"]
    INPUT -->|"目录"| READ_DIR["批量读取 CSV"]
    READ_CSV --> PREDICT
    READ_DIR --> PREDICT

    subgraph PREDICT["🔮 推理过程"]
        FORWARD["前向传播<br/>model.forward(x)"]
        FORWARD --> SOFTMAX["Softmax 概率"]
        SOFTMAX --> ARGMAX["取最大概率类别<br/>argmax(output)"]
    end

    ARGMAX --> RESULT["输出预测结果<br/>数字 0-9"]
    RESULT --> DONE(["✅ 完成"])
```

---

## 🧩 核心组件关系图

### Matrix — 数据基石

```mermaid
classDiagram
    class Matrix {
        -vector~double~ data_
        -size_t rows_
        -size_t cols_
        +Matrix(rows, cols)
        +transpose() Matrix
        +operator+(Matrix) Matrix
        -operator-(Matrix) Matrix
        +operator*(Matrix) Matrix
        +multiply_to(other, result)
        +scale_inplace(scalar)
        +add_inplace(other)
        +zero()
        +at(r, c) double
        +set_value(r, c, val)
        +rows() size_t
        +cols() size_t
        +data_ptr() double*
    }
```

> 📝 **行主序存储，列式批处理**：内存按行主序排列（`index = row * cols + col`），但批处理中每个样本存储为一列，`(784, N)` 表示 N 张 28×28 图像。

### Layer — 计算层

```mermaid
classDiagram
    class Layer {
        <<abstract>>
        +forward(input) Matrix
        +backward(grad_output) Matrix
        +parameters() vector~Matrix~&
        +param_gradients() vector~Matrix~&
    }

    class Linear {
        -Matrix W
        -Matrix b
        -Matrix grad_W
        -Matrix grad_b
        -Matrix cached_input
        +forward(input) Matrix
        +backward(grad_output) Matrix
        +parameters() vector~Matrix~&
        +param_gradients() vector~Matrix~&
    }

    class ReLU {
        -Matrix cached_input
        +forward(input) Matrix
        +backward(grad_output) Matrix
    }

    class GeLU {
        -Matrix input_cache_
        -Matrix sigmoid_cache_
        -constexpr double BETA = 1.702
        +forward(input) Matrix
        +backward(grad_output) Matrix
    }

    class LayerNorm {
        -size_t normalized_shape_
        -double epsilon_
        -Matrix gamma_
        -Matrix beta_
        -Matrix grad_gamma_
        -Matrix grad_beta_
        +forward(input) Matrix
        +backward(grad_output) Matrix
    }

    class PositionalEncoding {
        -size_t max_len_
        -size_t d_model_
        -Matrix pe_
        +forward(input) Matrix
    }

    Layer <|-- Linear
    Layer <|-- ReLU
    Layer <|-- GeLU
    Layer <|-- LayerNorm
    Layer <|-- PositionalEncoding
```

### Model — 网络容器

```mermaid
classDiagram
    class Model {
        -vector~unique_ptr~Layer~~ layers_
        +add~T~(args...) void
        +forward(input) Matrix
        +backward(grad_output) void
        +parameters() vector~Matrix*~
        +param_gradients() vector~Matrix*~
        +num_layers() size_t
    }
    Model o-- Layer : contains *
```

### Optimizer — 参数更新

```mermaid
classDiagram
    class Optimizer {
        <<abstract>>
        +step(params, grads) void
        +zero_grad(grads) void
    }

    class SGD {
        -double learning_rate
        +step(params, grads)
    }

    class SGD_w_Momentum {
        -double learning_rate
        -double momentum
        -vector~Matrix~ velocities
        +step(params, grads)
    }

    class Adam {
        -double learning_rate
        -double beta1, beta2
        -double epsilon
        -int t
        -vector~Matrix~ m  // 一阶矩
        -vector~Matrix~ v  // 二阶矩
        +step(params, grads)
    }

    Optimizer <|-- SGD
    Optimizer <|-- SGD_w_Momentum
    Optimizer <|-- Adam
```

---

## ⚡ 并行执行策略

```mermaid
flowchart LR
    OP["矩阵操作请求"] --> SP["SmartPolicy"]
    SP -->|"元素数 < 100K"| SERIAL["串行执行"]
    SP -->|"元素数 ≥ 100K"| POOL["ThreadPool 并行"]
    POOL --> W1["Worker 1"]
    POOL --> W2["Worker 2"]
    POOL --> W3["Worker 3"]
    POOL --> WN["Worker N"]
```

| 组件 | 作用 |
|------|------|
| `ThreadPool` | 单例线程池，懒初始化，管理 worker 线程和任务队列 |
| `SmartPolicy` | 自适应调度：小矩阵串行避免开销，大矩阵自动并行 |
| `parallel_for_each` | 并行遍历元素 |
| `parallel_transform` | 并行变换（如 ReLU、矩阵加减） |
| `parallel_transform_reduce` | 并行归约（如求和、求均值） |

---

## 📦 模型序列化格式 (model_io.hpp)

### V2 格式（当前）

```mermaid
flowchart LR
    subgraph "V2 二进制模型文件"
        A["Magic: 0x4E4E4E4E<br/>(uint32)"] --> B["Version: 2<br/>(uint32)"]
        B --> C["ModelType<br/>(uint32)"]
        C --> D["ModelSpec<br/>(架构参数)"]
        D --> E["Matrix 0<br/>rows(u64)+cols(u64)+data"]
        E --> F["Matrix 1 ..."]
        F --> G["..."]
    end
```

### ModelSpec 编码

| 模型类型 | 字段 |
|----------|------|
| MLP | `layer_dims`: num_dims(u32) + dim_0..dim_N(u64) |
| Transformer | d_model, num_heads, d_ff, num_layers, patch_size (各 u64) |
| GPT | vocab_size, d_model, seq_len, num_heads, d_ff, num_layers (各 u64) |

### 公开 API

| 函数 | 说明 |
|------|------|
| `save_model(path, model, spec)` | 保存 Model + ModelSpec 为 V2 格式 |
| `load_model(path, model)` | 加载参数到已有 Model（兼容 V1/V2） |
| `peek_model_spec(path)` | 只读文件头，返回 ModelSpec（V1 返回 Unknown） |

所有函数失败时抛出 `ModelIOError`，无 `std::cout` 副作用。

---

## 🐍 Python GUI 工作流

```mermaid
flowchart TD
    GUI["gui.py<br/>Tkinter GUI"] -->|"训练模式<br/>subprocess.run()"| TRAIN["train.cpp"]
    GUI -->|"推理模式<br/>subprocess.run()"| INFER["infer.cpp"]

    GUI -->|"画布绘制"| CANVAS["28×28 画布"]
    CANVAS -->|"鼠标事件"| DRAW["捕获笔画"]
    DRAW -->|"保存 CSV"| CSV["digit_xxxx.csv"]
    CSV --> INFER

    GUI -->|"实时显示"| LOG["训练日志输出"]
    LOG -->|"解析进度条"| PROGRESS["进度条更新"]
```

**辅助 Python 脚本：**
| 脚本 | 用途 |
|------|------|
| `save_dataset.py` | 从 raw MNIST 生成训练/测试 CSV |
| `extract_digits.py` | 从 CSV 中提取单个数字图像 |
| `csv_png.py` | 将 CSV 像素数据转为 PNG 图像预览 |

---

## 🗂️ 构建系统 (CMakeLists.txt)

```mermaid
flowchart TD
    CMAKE["CMakeLists.txt<br/>C++26, Clang++"] --> TARGET1["mnist_train<br/>训练可执行文件"]
    CMAKE --> TARGET2["mnist_infer<br/>推理可执行文件"]
    CMAKE --> DEPS["编译选项: -O3 -ffast-math<br/>-fexperimental-library<br/>-Wall -Wextra -Wpedantic -Werror"]
```

---

## 📊 数据流全景

```mermaid
flowchart LR
    subgraph "训练阶段"
        CSV_IN["train.csv<br/>60000 张图"] --> PARSE["解析像素值"]
        PARSE --> MAT["Matrix(784, N)"]
        MAT --> LABEL["one_hot(labels)<br/>Matrix(10, N)"]
        LABEL --> NET["前向→损失→反向→更新"]
        NET --> WT["训练后权重"]
        WT --> BIN["model.bin<br/>二进制模型"]
    end

    subgraph "推理阶段"
        BIN --> LOAD["加载权重"]
        CSV_TEST["digit_xxxx.csv<br/>单张图像"] --> PARSE2["解析像素值"]
        PARSE2 --> MAT2["Matrix(784, 1)"]
        MAT2 --> LOAD
        LOAD --> NET2["前向传播"]
        NET2 --> PROB["Softmax 概率"]
        PROB --> DIGIT["预测数字 0-9"]
    end
```

---

## 🎯 网络结构总结

默认 MNIST 架构（定义在 `mnist_common.hpp` 的 `MNIST_LAYER_DIMS`）：

```
┌──────────────────────────────────────────────────────────┐
│                 MNIST 手写数字识别网络                      │
├──────────────────────────────────────────────────────────┤
│  Input:    784 (28×28 像素展平)                           │
│  ────────────────────────────────────────────────────── │
│  Layer 1:  Linear(784 → 512) + LayerNorm + GeLU          │
│  Layer 2:  Linear(512 → 256) + LayerNorm + GeLU          │
│  Layer 3:  Linear(256 → 128) + LayerNorm + GeLU          │
│  Layer 4:  Linear(128 → 64)  + LayerNorm + GeLU          │
│  Layer 5:  Linear(64  → 10)                              │
│  ────────────────────────────────────────────────────── │
│  Loss:     CrossEntropy (含数值稳定 Softmax)              │
│  Optimizer: Adam / SGD / SGD+Momentum                    │
│  输出:      10 维向量 → argmax → 数字 0-9                  │
└──────────────────────────────────────────────────────────┘
```

> 📝 网络架构由 `MNIST_LAYER_DIMS = {784, 512, 256, 128, 64, 10}` 驱动，
> 自动在隐藏层之间插入 LayerNorm + GeLU 激活。

---

*Generated by GitHub Copilot — MiMo V2.5*
