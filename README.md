<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/icon-dark.svg">
  <img src="assets/icon-light.svg" alt="neuralnet.cpp logo" width="96">
</picture>

# neuralnet.cpp

**从零实现的 C++26 神经网络库**——不依赖 PyTorch/CUDA，CPU 与 Vulkan GPU 双后端，
覆盖 MLP 到 GPT 的训练与推理全流程，header-only，可两行嵌进你自己的工程。

[![CMake (Clang, C++26)](https://github.com/neuralnet-cpp-dev/neuralnet.cpp/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/neuralnet-cpp-dev/neuralnet.cpp/actions/workflows/cmake-single-platform.yml)
[![C++26](https://img.shields.io/badge/C%2B%2B-26-00599C?logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/26)
[![Vulkan](https://img.shields.io/badge/GPU-Vulkan%20Compute-AC162C?logo=vulkan&logoColor=white)](https://www.khronos.org/vulkan/)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](LICENSE)

[快速开始](#-快速-start) · [为什么选它](#为什么选它) · [它不做什么](#它不做什么) · [文档](#文档) · [参与](#参与)

</div>

---

## 🚀 快速开始

```bash
git clone --recurse-submodules git@github.com:neuralnet-cpp-dev/neuralnet.cpp.git
cd neuralnet.cpp
cmake --preset dev && cmake --build --preset dev        # 需要 CMake ≥3.30 + Ninja + C++26 编译器
```

训练你的第一个模型（MNIST，默认数据目录 `datasets/mnist_data/`）：

```bash
./build/dev/mnist_train --gpu --epochs 10               # GPU 训练（无 GPU 去掉 --gpu）
./build/dev/mnist_infer my_digit.csv --model mnist_model.nnmodel
```

把它当库用——三行核心代码构建一个 MLP：

```cpp
#include <neuralnet.cpp/nn.hpp>

auto engine = nn::CpuEngine::create();                  // GPU: nn::GpuBackend::shared()
nn::Model model(*engine);
model.add<nn::Linear>(784, 128);                        // add<T>(args...) 返回 Result<void>，
model.add<nn::ReLU>();                                 //   构造后自动 init(engine)
model.add<nn::Linear>(128, 10);
auto pred = model.forward(input);                       // engine 在 Model 内自动传递
```

<details>
<summary>准备训练数据（MNIST / 文本语料）</summary>

- MNIST：从 [Releases](https://github.com/EthanPeng-2048/neuralnet.cpp/releases) 下载
  `mnist_data` 压缩包，解压到 `datasets/mnist_data/`（`train.csv` / `test.csv`）。
- 文本语料：`scripts/` 下的 `download_*.py` 脚本；GPT 训练数据统一为 `.nndataset`：

  ```bash
  ./build/dev/tokenizer_train corpus.txt --vocab-size 10000 --output my_bpe.nnvocab
  ./build/dev/dataset_gen corpus.txt --vocab my_bpe.nnvocab -o corpus.nndataset
  ./build/dev/text_train corpus.nndataset --gpu
  ```

</details>

<details>
<summary>克隆后 extern/ 为空？</summary>

v1.13.0 起 core / threadpool / kvrecord / dataset / tokenizer 是 `extern/` 下的嵌套 submodule。
已 clone 但漏了子仓时补救：

```bash
git submodule update --init --recursive
```

</details>

## 为什么选它

| 亮点 | 说明 |
|------|------|
| **从零实现，全程可读** | 计算引擎、表达式 DSL、AOT kernel 融合、Vulkan 后端、BPE 分词器、训练循环全部自研——想搞懂"框架底层发生了什么"，每一层都有源码与文档 |
| **CPU / Vulkan GPU 双后端** | 同一套模型代码切换后端；GPU 侧表达式录制 → AOT 融合单 kernel；f16 混合精度 |
| **C++26 header-only** | 单一聚合头 `nn.hpp`；`FetchContent` / `add_subdirectory` 两行接入；零手动内存管理 |
| **训练 + 推理闭环** | MLP / CNN / ViT / Transformer / GPT；SGD/Adam/AdamW/Muon；`.nnpkg` 训练包跨设备复现；GUI 图形界面 |
| **RL 与可组合训练** | GAE、REINFORCE/PPO/GRPO、具名多输出、多模型 Session 与可替换 rollout pipeline |
| **严谨的工程口径** | 数值梯度自检、CPU/GPU 位级一致门禁、可引用的研究档案（每个结论带证据 + 重跑命令 + 边界，见 [`research/`](research/README.md)） |

## 它不做什么

诚实设定边界，避免不匹配的预期：

- **不做分布式训练**——单机单卡/CPU 为主，没有多机多卡、模型并行或流水线并行；
- **不追吞吐排名**——与 PyTorch 的 f32 训练吞吐差距约 2.1×（见 [基准报告](docs/benchmarks/2026-10-06-vulkan-vs-torch.md)），
  本项目的价值在**可读、自研、可嵌入**，不是 benchmark 榜单；
- **不是 Python 生态替代品**——没有 autograd 图、没有 pip 包、不兼容 ONNX/checkpoint 格式生态；
- **GPU 只走 Vulkan**——不做 CUDA/Metal 后端；部分精度路径（全 f16 训练）有已知数值限制，
  现状见 [`research/`](research/README.md) 与 [多精度文档](docs/development/05-mixed-precision.md)。

## 文档

**总索引见 [`docs/README.md`](docs/README.md)**（按读者分册）。最常去的入口：

| 我想… | 去看 |
|-------|------|
| 跑通训练/推理（CLI + GUI） | [docs/usage/02-quickstart-train-infer.md](docs/usage/02-quickstart-train-infer.md) |
| 用 C++ API 写自己的模型 | [docs/development/02-cpp-api-manual.md](docs/development/02-cpp-api-manual.md) |
| 把库嵌进自己的工程 | [docs/development/01-cmake-integration.md](docs/development/01-cmake-integration.md) |
| 了解架构与数据流 | [docs/introduction/01-architecture.md](docs/introduction/01-architecture.md) |
| 引用本仓的实测结论 | [research/README.md](research/README.md) |

## 依赖

| 组件 | 要求 |
|------|------|
| 编译器 | 任意 C++26 编译器（Clang / GCC / MSVC 均可；CI 主矩阵 = Clang） |
| 构建 | CMake ≥ 3.30 + Ninja（预设：`cmake --preset dev` 等，见 [CMakePresets.json](CMakePresets.json)） |
| GPU（可选） | Vulkan SDK；无 GPU 用 `--preset cpu` 纯 CPU 构建 |
| GUI（可选） | Python 3.10+ + `pip install customtkinter pillow` |

<details>
<summary>项目结构</summary>

```
neuralnet.cpp/
├── include/neuralnet.cpp/   ← header-only 库（nn.hpp 统一聚合入口；backend/ = Vulkan 后端）
├── src/                     ← 可执行入口：mnist/text 训练与推理、tokenizer、数据集工具
├── shaders/                 ← GPU compute shader（AOT 融合产物 + 手写原语）
├── docs/                    ← 文档总索引（docs/README.md）：usage / development / contributors / internal / history
├── research/                ← 可引用的研究档案（定义基线 / 实测结论 / 明确标注的 TODO）
├── gui.py                   ← 图形化操作界面（CustomTkinter）
├── extern/                  ← 子仓：core / threadpool / kvrecord / dataset / tokenizer
├── datasets/ · pretrained/  ← 训练数据 · 预训练模型
├── scripts/ · bench/ · compare_with_torch/  ← 数据脚本 · 基准测试 · torch 对拍
└── CMakePresets.json        ← 预设构建（dev / portable / cpu / probes ...）
```

</details>

## 参与

- **用库遇到问题**：提 issue，附最小复现与构建环境（编译器版本 + preset）。
- **想贡献代码**：入口是 [CONTRIBUTING.md](CONTRIBUTING.md)（构建、测试口径、PR 自查清单）；
  改代码前先读 [C++ 开发规范](docs/contributors/cpp-standards.md) 与 [踩坑警示录](docs/contributors/pitfalls.md)。
- **维护者**：@EthanPeng-2048。发布与版本号规则见 [release-policy](docs/contributors/release-policy.md)
  （**版本号由人提出，agent 不得擅自发布**）。
- 所有参与方须遵守 [行为准则](CODE_OF_CONDUCT.md)。

## 许可证

[GNU Lesser General Public License v3.0](LICENSE)（LGPL-3.0，含 GPL-3.0 条款），
© 2026 EthanPeng-2048。闭源应用可以链接本库；对**本库本身**的修改须以本许可证继续开源。

> **数据集说明**：`scripts/` 的下载脚本仅从第三方拉取数据（TinyStories、OpenWebText、中文 wiki 等），
> 数据集各自持有独立许可证，不随本项目 LGPL 分发，请自行遵守其条款。

## 🤖 AI 工具致谢

本项目的部分代码由 AI 辅助生成，当前使用：DeepSeek V4 / V4.1、Mi Mimo v2.5 / v2.6、
Qwen3.8、VS Code Copilot、ChatGPT Codex。AI 仅作为辅助工具，代码所有权与最终版权
归 EthanPeng-2048，不影响 LGPL 许可证效力。
