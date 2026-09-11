# ⚡ CUDA GPU 加速后端

> **⚠️ 已停用（v1.0.0）**
>
> **CUDA 后端已正式停用**：M4/M5/M6 融合原语与 DSL 表达式在 CUDA 上未实现，无真实回退，导致 CUDA 上 GPT/MNIST 训练推理均无法运行。为避免"文档声称支持但实际损坏"，自 v1.0.0 起**不提供 `NN_ENABLE_CUDA` 开关**（CMake 不再定义该 option），`cuda_engine.hpp` 仅在 `NN_HAS_CUDA` 下编译但该宏永不定义。CLI 的 `--cuda` 参数仍会被解析，但传入后返回错误"请求 --cuda 但未编译 CUDA 支持；不回退 CPU"。
>
> **本页保留作为恢复 CUDA 后端时的参考文档**，其中的构建命令、CMake 选项、运行方式在当前版本均**不可用**。恢复路线见 `docs/13-optimize-proposal-list.md` §5（P5-01/P5-02/P5-03）。

---

> **版本**：1.0（已停用）  
> **GPU**：Tesla V100-PCIE-32GB (sm_70)  
> **CUDA Toolkit**：12.8  
> **依赖**：nvcc + MSVC 2022 BuildTools（编译期），cudart / cuda / nvrtc（运行时）

---

## 📋 目录

1. [设计概览](#-设计概览)
2. [构建方法](#-构建方法)
3. [架构详解](#-架构详解)
4. [与 Vulkan 后端对比](#-与-vulkan-后端对比)
5. [内核清单](#-内核清单)
6. [编译器兼容性备注](#-编译器兼容性备注)
7. [使用方式](#-使用方式)
8. [故障排除](#-故障排除)

---

## 🎯 设计概览

CUDA 后端与 Vulkan 后端架构对称，共享同一套 `ComputeEngine` 接口：

```
ComputeEngine (抽象接口)
├── CpuEngine          ── 纯 CPU 实现
├── GpuEngine          ── Vulkan compute shader
└── CudaEngine         ── CUDA 内核 (本后端)
    └── CudaBackend    ── 管理设备/内存/内核调用
```

**关键决策**：CUDA 内核由 **nvcc 独立编译为 .obj**，再通过 C ABI 链接到 Clang 主程序。这是因为 nvcc 12.8 与 MSVC 2026 (VS 18) 的 `cudafe++` 不兼容，但与 VS 2022 BuildTools (MSVC 14.44) 可正常工作。

```
cuda/cuda_kernels.cu  ──nvcc (MSVC 2022)──▶  cuda_kernels.obj
                                                  │
src/*.cpp ──Clang──▶ *.obj ────────────────────────┤
                                                  │
include/.../cuda_backend.hpp  (extern "C")     ◀──┘
         .../cuda_engine.hpp  (ComputeEngine)
```

---

## 🔨 构建方法

### 前置条件

1. **CUDA Toolkit 12.8**：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
2. **VS 2022 BuildTools**：`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
   - nvcc 需要 MSVC 2017–2022 作为 host compiler
   - VS 2026 (MSVC 14.51) **不兼容**——`cudafe++` 会 crash (ACCESS_VIOLATION)
3. **LLVM/Clang**：主项目编译器

### 构建命令

```bash
# 配置（启用 CUDA）
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_CUDA=ON

# 可选：指定 SM 架构（默认 50 = GTX 850M；V100 = 70）
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_CUDA=ON -DNN_CUDA_ARCH=80

# 构建
cmake --build build --parallel
```

### CMake 选项

> **⚠️ 以下选项在当前版本已不存在**（CUDA 已停用，CMake 不再提供 `NN_ENABLE_CUDA` option）。仅作为恢复 CUDA 时的历史参考。

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `NN_ENABLE_CUDA` | `OFF` | 启用 CUDA GPU 加速（已停用，不再提供） |
| `NN_CUDA_ARCH` | `50` | SM 架构版本（50=GTX850M/Maxwell, 70=V100, 80=A100, 86=RTX30xx, 89=RTX40xx） |

---

## 🏗️ 架构详解

### 文件结构

```
cuda/
├── CMakeLists.txt           ← nvcc 编译脚本（.bat 包装 vcvars + nvcc）
├── cuda_kernels.cu          ← 15 个 CUDA 内核实现
└── cuda_kernels.h           ← extern "C" 接口声明

include/neuralnet.cpp/
├── backend/
│   └── cuda_backend.hpp     ← CudaBackend 单例 + CudaTensor + CudaBuffer
└── cuda_engine.hpp          ← CudaEngine（ComputeEngine 实现）
```

### 编译流程

1. **CMake 配置阶段**：生成 `compile_cuda_kernels.bat` 脚本
   - 调用 `vcvars64.bat` 初始化 MSVC 2022 环境
   - 运行 `nvcc -c -O2 --use_fast_math -allow-unsupported-compiler -Xcompiler /MD -arch=sm_XX`
2. **构建阶段**：
   - Ninja 调用 .bat 脚本编译 `cuda_kernels.cu` → `cuda_kernels.obj`
   - Clang 编译所有 `.cpp` 文件（含 cuda_backend.hpp）
   - 链接：`cuda_kernels.obj` + `cudart.lib` + `cuda.lib` + `nvrtc.lib`
3. **运行阶段**：
   - `CudaEngine` 调用 `CudaBackend` 的方法
   - `CudaBackend` 通过 extern "C" 函数调用 nvcc 编译的内核

### 数据流

```
CPU Matrix ──CudaBackend::upload_blocking──▶ CudaTensor (GPU)
    │
    ├── CudaBackend::matmul_gpu()     ──▶ CudaTensor
    ├── CudaBackend::elementwise_*()  ──▶ CudaTensor
    ├── CudaBackend::reduce_gpu()     ──▶ CudaTensor
    └── ...
    │
CudaTensor ──CudaBackend::download_blocking──▶ CPU Matrix
```

所有计算在 GPU 上原生执行，仅在 `from_matrix` / `to_matrix` 时进行 PCIe 传输。

---

## ⚖️ 与 Vulkan 后端对比

| 特性 | Vulkan | CUDA |
|------|--------|------|
| 编译器 | glslc (GLSL → SPIR-V) | nvcc (.cu → .obj) |
| 内核语言 | GLSL compute shader | CUDA C++ |
| 设备管理 | VkInstance / VkDevice / VkQueue | cudaSetDevice / cudaStream |
| 内存管理 | MemoryPool 子分配器 | cudaMalloc / cudaMemcpy |
| 同步模型 | Command buffer + fence | cudaDeviceSynchronize (同步流) |
| 矩阵乘法 | 自定义 tiled shader | 自定义 tiled kernel |
| 批处理 | begin_batch / end_batch (录制) | 预留（当前 no-op） |
| 构建依赖 | Vulkan SDK + glslc | CUDA Toolkit + MSVC 2022 |
| 运行时依赖 | vulkan-1.dll | cudart64_12.dll + cuda64_12.dll |

### 内核对照

两套后端的内核语义完全一致，可互相替换：

| 原语 | Vulkan shader | CUDA kernel |
|------|---------------|-------------|
| matmul | matmul.comp / matmul_tiled.comp | matmul_kernel / batched_matmul_kernel |
| elementwise | elementwise_v2.comp | elementwise_{unary,binary,scalar,axpy,select}_kernel |
| reduce | reduce.comp | reduce_{row,col}_{sum,max}_kernel |
| broadcast | broadcast.comp | broadcast_{row,col}_kernel |
| transpose | transpose.comp | transpose_kernel |
| gather | gather.comp | gather_kernel |
| scatter_add | scatter_add.comp | scatter_add_kernel |
| rearrange_3d | rearrange_3d.comp | rearrange_3d_kernel |

---

## 📦 内核清单

### cuda_kernels.h — extern "C" 接口

```c
// 矩阵乘法
int cuda_matmul(const float* A, const float* B, float* C, ...);
int cuda_batched_matmul(const float* A, const float* B, float* C, ...);

// 逐元素运算
int cuda_elementwise_unary(const float* A, float* out, ...);
int cuda_elementwise_binary(const float* A, const float* B, float* out, ...);
int cuda_elementwise_binary_scalar(const float* A, float* out, ...);
int cuda_axpy(const float* A, const float* B, float* out, ...);
int cuda_elementwise_select_scalar_cond(const float* A, const float* then_v, float* out, ...);

// 归约
int cuda_reduce_row(const float* A, float* out, ...);
int cuda_reduce_col(const float* A, float* out, ...);

// 广播
int cuda_broadcast_row(const float* A, const float* vec, float* out, ...);
int cuda_broadcast_col(const float* A, const float* vec, float* out, ...);

// 转置与重排
int cuda_transpose(const float* A, float* out, ...);
int cuda_rearrange_3d(const float* A, float* out, ...);

// Gather / Scatter
int cuda_gather(const float* table, const int* indices, float* out, ...);
int cuda_scatter_add(float* dst, const int* indices, const float* grad, ...);
```

所有函数返回 `int`（0=成功，非0=cudaError_t），最后一个参数为 `void* stream`（nullptr=默认流）。

---

## ⚠️ 编译器兼容性备注

### 问题

nvcc 12.8 的内部组件 `cudafe++` 在解析 **MSVC 2026 (VS 18, v14.51)** 头文件时会 crash：

```
nvcc error: 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
```

`-allow-unsupported-compiler` 只绕过版本检查，但 `cudafe++` 仍然会因无法解析新头文件而崩溃。

### 解决方案

使用 **VS 2022 BuildTools (MSVC 14.44)** 作为 nvcc 的 host compiler：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -c -O2 --use_fast_math -allow-unsupported-compiler -Xcompiler /MD -arch=sm_70 cuda_kernels.cu -o cuda_kernels.obj
```

### CRT 匹配

nvcc 默认使用 `/MT`（静态 CRT），但 Clang 主程序使用 `/MD`（动态 CRT）。必须通过 `-Xcompiler /MD` 强制 nvcc 使用动态 CRT，否则链接时报错：

```
lld-link: error: /failifmismatch: mismatch detected for 'RuntimeLibrary'
```

---

## 🎮 使用方式

> **⚠️ 当前版本（v1.0.0+）不可用**：以下 `--cuda` 命令在传入后都会返回错误"请求 --cuda 但未编译 CUDA 支持（NN_HAS_CUDA）；不回退 CPU"。此处仅保留为恢复 CUDA 时的历史参考。

### 命令行（历史参考，当前不可用）

```bash
# MNIST 训练 + CUDA 加速
./build/mnist_train --cuda --epochs 10

# GPT 训练 + CUDA 加速
./build/text_train corpus.txt --cuda --epochs 10 --lr 0.001

# 推理 + CUDA 加速
./build/text_infer --model gpt_model.bin --cuda --prompt "Hello"

# MNIST 推理 + CUDA 加速
./build/mnist_infer --model mnist_model.bin --cuda --input test.csv
```

### 优先级

`--cuda` 优先于 `--gpu`（Vulkan）。如果同时指定两者，CUDA 生效。

### C++ API（历史参考，当前不可用）

```cpp
#include <neuralnet.cpp/nn.hpp>

// 创建 CUDA 引擎
auto& backend = nn::CudaBackend::instance();
backend.initialize();
nn::CudaEngine engine(backend);

// 与 CpuEngine / GpuEngine 完全相同的 API
nn::Model model(engine);
model.add_linear(784, 256).add_relu().add_linear(256, 10);
```

---

## 🔧 故障排除

### "No CUDA devices found"

确认 NVIDIA 驱动已安装：`nvidia-smi`

### nvcc 编译失败："unsupported Microsoft Visual Studio version"

确保 VS 2022 BuildTools 已安装。CMakeLists.txt 中的 `.bat` 脚本会自动调用 `vcvars64.bat`。

### cudafe++ ACCESS_VIOLATION

MSVC 版本太新。使用 VS 2022 BuildTools（MSVC 14.44），不要用 VS 2026（MSVC 14.51）。

### 链接错误：RuntimeLibrary mismatch

nvcc 需要 `-Xcompiler /MD` 标志。检查 `build/cuda/compile_cuda_kernels.bat` 是否包含此标志。

### 找不到 cuda_kernels.h

CMakeLists.txt 中 `NN_CUDA_INCLUDE_DIR` 指向 `cuda/` 目录。确认 `add_subdirectory(cuda)` 存在。
