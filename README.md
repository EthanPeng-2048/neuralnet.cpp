# neuralnet.cpp

一个轻量级 C++26 神经网络库，从头实现（无第三方深度学习框架依赖），**列主序矩阵**（column-major）存储，支持并行计算。

## 项目结构

```
neuralnet.cpp/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── gui.py                  ← 图形化操作界面
├── csv_png.py
├── extract_digits.py
├── save_dataset.py
├── build/
│   ├── mnist_infer.exe
│   ├── mnist_train.exe
│   └── mnist_data/
│       ├── test.csv
│       └── train.csv
├── include/
│   └── neuralnet.cpp/
│       ├── layer.hpp
│       ├── loss.hpp
│       ├── matrix.hpp
│       └── optimizer.hpp
└── src/
    ├── infer.cpp
    └── train.cpp
```

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
# 从头开始训练
./build/mnist_train

# 从已有模型恢复训练
./build/mnist_train --load mnist_model.bin

# 指定保存路径
./build/mnist_train --save my_model.bin
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
| 🏋️ **训练** | 配置数据集路径、模型保存路径、轮数、学习率、批大小、优化器，支持恢复训练；实时显示训练日志 |
| 🔍 **推理** | 选择模型文件和图片 CSV 文件/目录，显示预测结果置信度条形图和手写数字图片预览 |
| 🖼️ **图片查看** | 浏览单张 CSV 图片或批量加载目录，支持前后翻页导航 |

## 网络结构

```
输入 (784) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(64) → ReLU → Linear(10)
```

## 提供的组件

| 组件 | 说明 |
|------|------|
| `nn::Matrix` | 列主序矩阵，支持并行加/减/乘/转置 |
| `nn::Linear` | 全连接层（含 Xavier 初始化） |
| `nn::ReLU` | ReLU 激活函数 |
| `nn::MSELoss` | 均方误差损失 |
| `CrossEntropyLoss` | 交叉熵损失（含数值稳定 Softmax，在 `train.cpp` 中定义） |
| `nn::SGD` | 随机梯度下降优化器 |
| `save_model` / `load_model` | 二进制模型序列化 |

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
