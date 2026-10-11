# 构建集成：把库接进你的工程

> **读者**：调用者（库使用者）——要在自己的 CMake 工程里用 neuralnet.cpp，或要在本仓
> 加文件 / 加 shader / 加测试。总索引见 [`README.md`](README.md)。
>
> 消费方的最小接入也可以看 [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md)；
> 本文是**完整版**：全部构建开关、GPU 语义、自研算子融合注册、跨平台与故障排查。

---

## 1. 两种角色

| 角色 | 你做什么 | 用哪套 CMake |
|---|---|---|
| **消费方**（把库嵌进自己的工程） | `add_subdirectory` + 链 `neuralnet::nn` | 本文 §2–§6 |
| **本仓开发**（改库本身 / 跑测试） | 在仓库内构建 app / 测试 | 本文 §7 |

目标就一个：**门面目标 `neuralnet::nn`**。把 C++ 标准、include 目录、线程库、GPU 宏与
shader 生成顺序全部打包传递，你只写两行。

---

## 2. 最小接入（两行 CMake + 一个 include）

```cmake
cmake_minimum_required(VERSION 3.30)
project(my_app LANGUAGES CXX)

# ① 嵌入库（把仓库放到源码树任意位置；下例为 3rd_party/）
add_subdirectory(3rd_party/neuralnet.cpp)

# ② 链接门面目标
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

```cpp
// main.cpp
#include <neuralnet.cpp/nn.hpp>

int main()
{
    nn::CpuEngine engine;
    nn::Model model = NN_CHECK(nn::build_mnist_mlp_model(engine));
    (void)model;
    return 0;
}
```

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

**FetchContent** 同理，把 ① 换成：

```cmake
include(FetchContent)
FetchContent_Declare(neuralnet.cpp GIT_REPOSITORY <url> GIT_TAG <tag>)
FetchContent_MakeAvailable(neuralnet.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

独立最小样例工程见 [`examples/downstream/`](../../examples/downstream/)（CPU / GPU 双路径 + ctest 注册）。

---

## 3. `neuralnet::nn` 替你传递了什么

链接它之后，你的目标自动获得：

- **C++ 标准**：`cxx_std_26`（Clang/GNU）；MSVC `cl` 不认识 `cxx_std_26` →
  自动降级 `cxx_std_23`（映射 `/std:c++latest`）；
- **include 目录**：`include/` +（GPU 时）SPIR-V 嵌入头目录 `generated/`；
- **线程库**：`Threads::Threads`；
- **GPU（若启用）**：宏 `NN_HAS_VULKAN`、`Vulkan::Vulkan` 链接、以及
  「shader 生成先于你的 TU 编译」的顺序依赖；
- **库头需要的接口选项**：Clang 的 `-Wno-pass-failed` / `-fexperimental-library`、
  MSVC 的 `/utf-8`、`/Zc:preprocessor`（否则 `NN_CHECK` / `NN_EXIT` 会被
  传统预处理器静默展开错）。

> 你**不要**自己写任何 Vulkan 相关 CMake。源码里 `#ifdef NN_HAS_VULKAN` 分支即可。
> ⚠ `Vulkan::Vulkan` 是 imported 目标，**不可跨目录直接引用**——一律经 `neuralnet::nn` 传递。

---

## 4. 构建开关（在 `add_subdirectory` **之前**设置）

| 变量 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `NN_ENABLE_GPU` | `AUTO` / `ON` / `OFF` | `AUTO` | `AUTO` = 探测到 Vulkan + glslc 就启用 GPU；`ON` = 强制要求，缺依赖则**配置期失败**（不静默降级）；`OFF` = 完全跳过（不探测、不编 `scan_exprs`、不生成 shader → 纯 CPU 最快路径） |
| `NN_BUILD_APPS` | `ON` / `OFF` | **嵌入时 OFF**（顶层工程 ON） | 是否构建本仓 app / 基准可执行文件。嵌入时默认 OFF —— 消费方只拿库目标 |
| `NN_ENABLE_TESTS` | `ON` / `OFF` | `OFF` | 是否构建并注册本仓测试目标（`list(APPEND NN_TEST_TARGETS ...)` 收集） |
| `NN_ENABLE_PCH` | `ON` / `OFF` | `ON` | 本仓共享 PCH（`nn.hpp` 预编译一次，多目标 `REUSE_FROM`）。无 app 目标时自动无候选、无副作用 |
| `NN_ENABLE_NATIVE` | `ON` / `OFF` | `ON` | `-march=native`（只作用于本仓目标，不影响消费方编译选项）。分发 / CI 用 `OFF` 生成可移植基线 |
| `NN_ENABLE_PROBES` | `ON` / `OFF` | `OFF` | **探针编译门**：打开才编译库内热路径探针并创建探针目标（`mem_probe` / `f16_cpu_probe` / `gpu_stability_probe`）。会改变库头语义 → **不可与默认构建的对象 / 缓存混用** |
| `NN_COMPILER_LAUNCHER` | path | 自动探测 ccache / sccache | 编译器缓存启动器；显式传空串 = 禁用 |

```cmake
# 例：纯 CPU、不编本仓应用
set(NN_ENABLE_GPU OFF)
add_subdirectory(3rd_party/neuralnet.cpp)
```

命令行等价：`cmake -B build -DNN_ENABLE_GPU=OFF`。

> **探针门语义**：`NN_ENABLE_PROBES` 经 `nn_core` 的 **INTERFACE** 定义传播
> （`NN_ENABLE_PROBES=1`），保证所有 TU 取值一致（同头两种语义 = ODR 隐患）。
> 关时 `nn_dbg_scan` / `NN_PREC_TRACE` / `NN_NAN_TRACE` / `NN_MEM_STATS` 等**不参与编译**，
> 相关环境变量一律无效。默认关 → 正常训练/推理没有额外分支与字符串。

---

## 5. GPU 语义

- **能力随目标传递**：链接 `neuralnet::nn` 且配置期 GPU 启用，你的目标就自动拿到
  `NN_HAS_VULKAN`、SPIR-V 嵌入头目录、`Vulkan::Vulkan`、shader 生成顺序。
- **`NN_ENABLE_GPU=AUTO` 的静默降级**：找不到 Vulkan / glslc 时走纯 CPU。
  想要「没 GPU 就配置失败」用 `ON`。
- **闭合世界**：库内层的融合 shader 构建期全部生成，你的目标直接跑库内层**开箱即用**。
  但若你在自己的层里写了 `nn::dsl::compute` 表达式，必须再加一行（见 §6），
  否则 GPU 运行期按 `expr_spec_key` **硬报错**（无 eager 回退，设计使然）。

---

## 6. 自研算子的融合注册：`nn_enable_gpu_fusion`

```cmake
nn_enable_gpu_fusion(<target> MAIN <含 main 的源文件> [OUT_DIR <目录>])
```

| 参数 | 说明 |
|---|---|
| `<target>` | 你的可执行/库目标（函数会为其创建 `<target>_nnfusion_collect`） |
| `MAIN` | **必须**指向含 `main` 的源文件——收集器自带 main，会从目标源列表剔除它（漏写 / 写错路径会在配置期报错） |
| `OUT_DIR` | 可选，生成 `fused_registry.hpp` 的目录（默认 `${CMAKE_CURRENT_BINARY_DIR}/nn_fusion/<target>`） |

它在构建期为你的目标收集全部 `nn::dsl::compute` 调用点（含 `#ifdef` 内的），生成**目标专属**
的 `fused_registry.hpp` 与融合 shader，并把该目录 `BEFORE` 放到 include 路径最前
（让 `__has_include("fused_registry.hpp")` 解析到你这份）。

```cmake
add_executable(my_app main.cpp my_layer.cpp)          # 自研层在 my_layer.cpp
target_link_libraries(my_app PRIVATE neuralnet::nn)
nn_enable_gpu_fusion(my_app MAIN ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
```

- 该函数**只在 GPU 启用时可用**；未启用时调用会得到指向根因的明确报错
  （也提供了一个 stub，报 "需要 GPU 构建"）。
- 样例：`examples/fusion_custom_layer*`（库内形态）、
  `examples/downstream`（`-DNN_DOWNSTREAM_FUSION=ON`，**从消费方 CMakeLists 调用**，
  验证函数在调用方作用域求值的路径）。

---

## 7. 本仓开发：构建与测试

```bash
# 默认 Release：-O3 -funroll-loops -march=native
cmake -B build -G Ninja && cmake --build build

# 开测试（测试程序在 build/，注册进 ctest）
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build && ctest --test-dir build
```

构建出的应用入口：`build/{mnist_train,mnist_infer,text_train,text_infer,tokenizer_train,
tokenizer_infer,dataset_gen,dataset_convert}`，另有 `layer_bench`（性能）与测试类可执行文件。
`NN_BUILD_APPS` 控制这批 app；探针目标（`mem_probe` / `f16_cpu_probe` / `gpu_stability_probe`）
另受 `NN_ENABLE_PROBES` 控制。

**已知构建坑**：链接报 `permission denied`（`build/` 下 exe 正在运行被占用 → 构建中止）
→ 追加 `-- -k 0` 让 ninja 跳过失败目标继续建其余目标（被占用目标仍需关掉进程后重链）。

---

## 8. 编译耗时

本工程 header-only，单 TU 首次 parse `nn.hpp`（含 36 份原语 SPIR-V 嵌入头 + AOT 融合注册表，
约 7MB 头文本）较慢——本机实测约 17.5s 无 PCH。已落地四项提速（默认开启，无需配置）：

1. `NN_ENABLE_PCH`：`nn.hpp` 预编译一次，由编译条件相同的目标 `REUSE_FROM` 共享
   （本机 Debug 全量 125.5s → 72.7s）；
2. `CMAKE_CXX_SCAN_FOR_MODULES OFF`：本项目零 C++ 模块；
3. `scan_exprs` 的 glslc **并行**（环境变量 `NN_SCAN_JOBS=<n>` 覆盖，`=1` 回串行）
   + `fused_registry.hpp` 内容稳定写入（内容不变则不动 mtime，靠 `restat` 掐断下游重编）；
4. 编译器缓存探测扩到 ccache / sccache（可用 `-DNN_COMPILER_LAUNCHER=<path>` 指定）。

**消费方**加速：对自己的目标用
`target_precompile_headers(my_app PRIVATE <path>/neuralnet.cpp/nn.hpp)`。

> ⚠ 两条不变量：① 收集器目标（`scan_exprs`、`*_nnfusion_collect`）**不套 PCH**；
> ② 新写「生成物参与 PCH」的场景必须显式 `add_dependencies(<pch_target>, <生成边>)`。

---

## 9. 跨平台与编译器

- **标准要求 C++26**（接口以 `target_compile_features` 传递）。编译器：Clang 22+ / GCC 15+ /
  MSVC（VS2026+，走 `/std:c++latest`）。
- **CMake ≥ 3.30**。`CMakeLists.txt` 不指定编译器（由 CMake 探测或调用方
  `CXX=...` / `-DCMAKE_CXX_COMPILER` 决定），**标准库编译/链接两侧必须一致**。
- **MSVC**：必须带 `/Zc:preprocessor`（已作为接口选项随门面传递）。`cl.exe` 默认的传统
  预处理器会把 `NN_CHECK(x, msg)` 静默退化成逗号表达式（条件恒真、检查永不触发），
  `NN_EXIT` 直接 `C3861`。
- **Clang + libc++**：`-Werror` 下 libc++ 会对「记录类型但从未使用」的局部变量报
  `-Wunused-variable`（libstdc++ 不报）——属真实死变量，删掉即可。
- **`-fno-exceptions` 不传递**：库头不使用异常，消费方按自己的异常策略编译即可。
- 已验证四组「编译器 × 标准库」各自全新构建 + ctest：clang + libc++、
  g++ + libstdc++（CPU-only）、clang-cl + MSVC STL、MSVC cl + MSVC STL。

---

## 10. 故障排查

| 症状 | 原因 / 处理 |
|---|---|
| `target was not found: Vulkan::Vulkan` | 在自己的 CMakeLists 里直接链了 `Vulkan::Vulkan`（imported 目标不可跨目录引用）→ 改链 `neuralnet::nn` |
| GPU 运行期 `expr_spec_key ... closed world` 硬报错 | 自研 `dsl::compute` 未注册 → 加 `nn_enable_gpu_fusion(...)`（§6） |
| `nn_enable_gpu_fusion` 报 "Unknown CMake command" | 构建缓存过旧 → 重新 configure（当前版本在 GPU 不可用时也定义了 stub，会给出指向根因的报错） |
| `nn_enable_gpu_fusion` 报 "MAIN=... 不在目标的源文件列表里" | `MAIN` 路径写错 / 目标里没有该文件（防链接期 duplicate main） |
| 编译报 `cxx_std_26 not known to CXX compiler` | 编译器过旧 |
| 找不到 `neuralnet.cpp/nn.hpp` | 确认链接的是 `neuralnet::nn`，且在 `add_subdirectory` **之后** |
| 找到了 Vulkan 但想要纯 CPU | `-DNN_ENABLE_GPU=OFF`（在 add_subdirectory 之前或命令行） |
| 想「没 GPU 就失败」 | `-DNN_ENABLE_GPU=ON` |
| 探针环境变量（`NN_F16_DEBUG` 等）无效 | 默认构建不含探针 → 重新配置 `-DNN_ENABLE_PROBES=ON` |
