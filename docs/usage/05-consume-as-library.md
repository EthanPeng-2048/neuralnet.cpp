# 作为库使用（add_subdirectory / FetchContent 嵌入）

> 面向**把 neuralnet.cpp 放进自己工程**的使用者。目标形态：CMake 两行 + 源码一个 include。
> 本仓自身的构建/测试见 `AGENTS.md` §2；本文只讲消费侧。

## 1. 快速开始

把仓库放到自己的源码树里（任意路径，下例为 `3rd_party/`）：

```cmake
cmake_minimum_required(VERSION 3.30)
project(my_app LANGUAGES CXX)

# ① 嵌入库（路径按你的摆放方式调整）
add_subdirectory(3rd_party/neuralnet.cpp)

# ② 链接门面目标——C++ 标准、include 目录、线程库、（GPU 时）NN_HAS_VULKAN、
#    SPIR-V 嵌入头目录、Vulkan 链接、shader 生成顺序依赖，全部由它传递
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

```cpp
// ③ 源码侧唯一的入口
#include <neuralnet.cpp/nn.hpp>

int main() {
    nn::CpuEngine engine;
    // ... 构建模型、训练/推理，见 docs/usage/01、02
}
```

最小可运行样例见 [`examples/downstream/`](../../examples/downstream/)（独立工程，
CPU/GPU 双路径 + ctest 注册）。

FetchContent 同理，把 ① 换成：

```cmake
include(FetchContent)
FetchContent_Declare(neuralnet.cpp GIT_REPOSITORY <url> GIT_TAG <tag>)
FetchContent_MakeAvailable(neuralnet.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

## 2. 构建开关（嵌入方在 `add_subdirectory` 之前设置）

| 变量 | 取值 | 默认 | 含义 |
|------|------|------|------|
| `NN_ENABLE_GPU` | `AUTO` / `ON` / `OFF` | `AUTO` | `AUTO` = 探测到 Vulkan+glslc 就启用 GPU（本仓历史行为）；`ON` = 强制要求，缺 Vulkan 或 glslc **配置期直接失败**（不静默降级 CPU）；`OFF` = 完全跳过——不探测 Vulkan、不编 `scan_exprs`、不生成 shader（纯 CPU 最快路径） |
| `NN_BUILD_APPS` | `ON` / `OFF` | **嵌入时 OFF** | 是否构建本仓 9 个 app/探针/基准可执行文件。顶层工程默认 ON，`add_subdirectory` 嵌入默认 OFF——消费方只拿到库目标，不被迫编译整个应用集 |
| `NN_ENABLE_TESTS` | `ON` / `OFF` | `OFF` | 本仓 23 个 ctest 用例（嵌入场景一般不开） |
| `NN_ENABLE_PCH` | `ON` / `OFF` | `ON` | 本仓共享 PCH（无 app 目标时自动无候选、无副作用） |
| `NN_ENABLE_NATIVE` | `ON` / `OFF` | `ON` | `-march=native`（只作用于本仓目标，不影响消费方编译选项） |
| `NEURALNET_CPP_DIR` | 路径 | `../..` | 仅 `examples/downstream` 样例用：库位置（可指向 3rd_party 摆放处） |

```cmake
# 例：纯 CPU、不编本仓应用
set(NN_ENABLE_GPU OFF)
add_subdirectory(3rd_party/neuralnet.cpp)
```

命令行等价：`cmake -B build -DNN_ENABLE_GPU=OFF`。

## 3. GPU 语义（重要）

- **GPU 能力随目标传递**：只要链接 `neuralnet::nn` 且配置期 GPU 启用，你的目标就自动获得
  `NN_HAS_VULKAN` 宏、36 份原语 SPIR-V 嵌入头 + 库自身 `fused_registry.hpp` 的 include
  目录、`Vulkan::Vulkan` 链接、以及「shader 生成先于你的 TU 编译」的顺序依赖。
  源码里 `#ifdef NN_HAS_VULKAN` 即可分支，无需自己写任何 Vulkan 相关 CMake。
- **闭合世界**：库内层的融合 shader 在库构建期全部生成；你的目标直接跑库内层**开箱即用**。
  但若你在自己的层里写了 `nn::dsl::compute` 表达式，必须再加一行（见 §4）——否则 GPU
  运行期会按 `expr_spec_key` **硬报错**（无 eager 回退，这是 AOT 闭合世界的设计）。
- **`NN_ENABLE_GPU=AUTO` 的静默降级**：找不到 Vulkan 时走纯 CPU（与本仓行为一致）。
  想要"没 GPU 就失败"用 `ON`。

## 4. 自研算子的融合注册（可选，一行 CMake）

在自己的层里用 `nn::dsl::compute` 写表达式后，让构建期收集器把你的结构编进**你的目标专属**
的 `fused_registry.hpp`：

```cmake
nn_enable_gpu_fusion(my_app MAIN ${CMAKE_CURRENT_SOURCE_DIR}/my_app_main.cpp)
```

- 收集器 = 库内收集逻辑（锚点 + dry-run + 模型 pass）+ **你的目标全部 TU**
  （以 `-DNN_EXPR_SCAN` 编译，表达式调用点在静态初始化期自登记）。
- `MAIN` 必须指向含 `main` 的源文件（收集器自带 main，需从目标源列表剔除）。
- 该函数**只在 GPU 启用时可用**；未启用时调用会得到指向根因的明确报错。
- 样例：`examples/fusion_custom_layer*`（库内形态）、`examples/downstream`
  （`-DNN_DOWNSTREAM_FUSION=ON`，消费方形态——从你的 CMakeLists 里调用，验证了
  函数在调用方作用域求值的路径）。

## 5. 已知限制

- **编译耗时**：header-only，单 TU 首次 parse `nn.hpp`（含全部层 + SPIR-V 嵌入 +
  融合注册表，约 7MB 头文本）较慢（本机实测约 17.5s 无 PCH）。消费方可对**自己的**
  目标用 `target_precompile_headers(my_app PRIVATE <path>/neuralnet.cpp/nn.hpp)` 加速。
- **`-Wno-pass-failed` / `-fexperimental-library`（Clang）与 `/utf-8`（MSVC）**
  已作为接口选项随 `neuralnet::nn` 传递——库头文件的 `#pragma clang loop` 与
  UTF-8 中文注释需要它们，否则消费方开 `-Werror` 会编译失败。
- **`-fno-exceptions` 不传递**：库头文件不使用异常（铁律 #1），消费方按自己的
  异常策略编译即可。
- **标准要求 C++26**（接口以 `target_compile_features` 传递：Clang/GNU 传
  `cxx_std_26`；MSVC cl 不识别 `cxx_std_26`，降级传 `cxx_std_23`——其映射即
  `/std:c++latest`，C++26 预览全开）。MSVC 下消费方源码也须为 UTF-8（`/utf-8` 已传）。
- **install / `find_package` 尚不支持**：当前只支持源码嵌入（add_subdirectory /
  FetchContent）。`install()`/export 与 `CMakePresets.json` 见
  `docs/development/18-roadmap.md` P0-4 的剩余部分。
- **只支持 CPU / Vulkan 双后端**（无 CUDA）。

## 6. 故障排查

| 症状 | 原因 / 处理 |
|------|------------|
| `target was not found: Vulkan::Vulkan` | 在自己的 CMakeLists 里直接 `target_link_libraries(... Vulkan::Vulkan)`——imported 目标不可跨目录引用。改为链接 `neuralnet::nn`（Vulkan 作为接口需求传递） |
| GPU 运行期 `expr_spec_key ... closed world` 硬报错 | 自研 `dsl::compute` 未注册 → 加 `nn_enable_gpu_fusion(...)`（§4） |
| `nn_enable_gpu_fusion` 报 "Unknown CMake command" | 版本较旧的构建缓存；重新 configure。当前版本在 GPU 不可用时也定义了 stub，会给出指向根因的报错 |
| 编译报 `cxx_std_26 not known to CXX compiler` | 编译器过旧；本库需 Clang 22+ / GCC 15+ / MSVC（VS2026+，走 `/std:c++latest`） |
| 消费方目标找不到 `neuralnet.cpp/nn.hpp` | 确认链接的是 `neuralnet::nn`（include 目录经它传递），且在 `add_subdirectory` **之后** |
| 找到了 Vulkan 但想要纯 CPU | `-DNN_ENABLE_GPU=OFF`（在 add_subdirectory 之前或命令行） |
