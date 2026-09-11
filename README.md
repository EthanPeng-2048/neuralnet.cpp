# neuralnet.cpp

一个轻量级 C++26 神经网络库，从头实现（无第三方深度学习框架依赖），**列主序矩阵**（column-major）存储，支持并行计算。

## 项目结构

```
neuralnet.cpp/
├── .gitignore
├── CMakeLists.txt
├── README.md
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

## 构建与运行

### build.cmake

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
