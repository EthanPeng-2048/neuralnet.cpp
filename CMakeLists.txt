# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)

# ── 编译器与标准库：一律使用默认，不由本文件指定 ──────────────────────────
# 编译器：用 CMake 默认探测结果，或调用方显式 -DCMAKE_CXX_COMPILER / CXX 环境变量。
# 标准库：用编译器默认（Linux clang/gcc 默认 libstdc++）。如需 libc++，由调用方
# 通过 -DCMAKE_CXX_FLAGS / -DCMAKE_EXE_LINKER_FLAGS 统一指定（编译与链接必须一致，
# 否则链接期会出现 std 符号 ABI 不匹配）。
# 本文件不再隐式挑选编译器或 stdlib，避免与调用方/CI 的设置冲突。

# VERSION 以 git tag 为准（当前 v1.6.1；AGENTS §变更此文件时务必同步 git 状态）
project(neuralnet.cpp VERSION 1.7.0 LANGUAGES CXX)

# ══════════════════════════════════════════════════════════════════════════
#  作为库被消费（add_subdirectory / FetchContent 嵌入 3rd_party）的门控
# ══════════════════════════════════════════════════════════════════════════
# PROJECT_IS_TOP_LEVEL：本文件是顶层工程（正常开发/CI）时 ON；被消费方
# add_subdirectory 嵌入时 OFF。据此默认值翻转，让嵌入方**只拿到库目标**，
# 不被迫编译本仓 12 个 app/探针/基准（可显式 -DNN_BUILD_APPS=ON 打开）。
option(NN_BUILD_APPS "Build neuralnet.cpp application/bench executables"
       ${PROJECT_IS_TOP_LEVEL})

# ── GPU（Vulkan）开关：消费方在自己工程里一行控制 NN_HAS_VULKAN ───────────
#   AUTO（默认）= 探测到 Vulkan+glslc 就启用，否则纯 CPU（本仓历史行为）
#   ON          = 强制要求，缺 Vulkan 或 glslc 直接配置失败（fail-fast）
#   OFF         = 完全跳过：不探测 Vulkan、不生成 shader、不编 scan_exprs
# 嵌入方写法：在 add_subdirectory **之前** set(NN_ENABLE_GPU OFF)（或 -D）。
set(NN_ENABLE_GPU "AUTO" CACHE STRING "GPU support: AUTO (detect) / ON (require) / OFF (disable)")
set_property(CACHE NN_ENABLE_GPU PROPERTY STRINGS AUTO ON OFF)
if(NOT NN_ENABLE_GPU MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "NN_ENABLE_GPU must be AUTO/ON/OFF, got '${NN_ENABLE_GPU}'")
endif()

# ── 本仓源/构建根（CACHE，全局可见）─────────────────────────────────────
# CMake 函数体在**调用方作用域**求值：消费方调用 nn_enable_gpu_fusion 时，
# CMAKE_SOURCE_DIR/CMAKE_BINARY_DIR 会解析到消费方的目录（实测报错/路径错）。
# 因此把本仓的根路径烘焙进 CACHE（INTERNAL 隐含 FORCE，每次配置刷新），
# 函数体内一律引用这两个变量，与调用方是谁无关。
set(NN_LIB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL
    "neuralnet.cpp source root (stable under add_subdirectory)")
set(NN_LIB_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL
    "neuralnet.cpp binary root (stable under add_subdirectory)")

# ── C++ 模块：本项目不使用 ────────────────────────────────────────────────
# C++20 起 CMake 默认为每个源文件追加一次 `clang-scan-deps` 依赖扫描
# （build.ninja 里的 *.obj.ddi / CXX.dd / *.obj.modmap 三条边），本项目
# **没有任何 `import`/`export module`**（header-only 库），扫描是纯浪费：
# 实测每个 TU 约 2.1s（占单文件编译时间 ~12%）。关闭后 `.ddi` 边整体消失，
# 编译命令里的 `@*.obj.modmap` 响应文件也不再生成。
# 若将来真的要引入 C++ 模块，删掉这一行即可恢复默认行为。
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ── C++26 标准 ──────────────────────────────────────────────────────────
# CMake 4.3 尚未内置 MSVC 的 CXX26 方言标志映射（VS2026 cl 19.51 也不认
# /std:c++26，C++26 特性走 /std:c++latest），故对 MSVC 用 CMAKE_CXX_STANDARD=23
# 让方言探测通过，再显式传 /std:c++latest。Clang/GCC 走标准 CMAKE_CXX_STANDARD=26。
if(MSVC)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(/std:c++latest)
else()
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ── 默认 Release 模式，可通过 -DCMAKE_BUILD_TYPE=Debug 切换 ────────────────
# 仅顶层工程时设置：嵌入（add_subdirectory）时不得强改消费方的构建类型。
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 公共编译标志 ─────────────────────────────────────────────────────────
# 项目规范：禁止异常、开启严格警告。按编译器选择对应标志：
# - Clang/GCC：-fno-exceptions -Wall -Wextra -Wpedantic -Werror
# - MSVC：     /W4 /permissive-（严格一致性）；保留 /EHsc（ppltasks.h 需要），
#              不强制 /WX（MSVC 与 clang 警告集不同，代码未按 MSVC 零警告编写）
if(MSVC)
    add_compile_options(/W4 /permissive- /source-charset:utf-8 /execution-charset:utf-8)
else()
    add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)
endif()

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
    # f16 矩阵乘法（MatrixT<F16>）无原生 x86 SIMD → 优化器无法向量化是预期行为
    # （§7.2 兼容路径：f32 参考计算 + f16 舍入），抑制 -Wpass-failed 避免 -Werror 下编译失败
    add_compile_options(-Wno-pass-failed)
endif()
# C++ 标准库由编译器默认决定（见文件开头的说明），此处不再显式指定 -stdlib。
#
# ── 编译器缓存（可选，强烈建议装了就用）──────────────────────────────────
# 自动探测 ccache / sccache（Windows 上 ccache 常缺席、sccache 更常见），也可用
# `-DNN_COMPILER_LAUNCHER=<path>` 显式指定（显式指定为空串 = 禁用）。
# 对本 header-only 工程收益很大：改动只影响单个 TU 时，未变 TU 直接命中缓存。
# ⚠ 调用方若已给 CMake 原生的 `-DCMAKE_CXX_COMPILER_LAUNCHER=...`（CI 就是这么
#   配的），这里**不再重复包一层** RULE_LAUNCH_COMPILE——两套机制叠加会变成
#   `ccache ccache clang++`。
# ⚠ RULE_LAUNCH_* 是 **GLOBAL 属性**，会作用于整个构建树——嵌入（add_subdirectory）
#   消费方的构建时**自动探测一律不生效**（避免污染消费方）；消费方想要就自己设
#   CMAKE_CXX_COMPILER_LAUNCHER，或显式传 -DNN_COMPILER_LAUNCHER=<path>（显式 =
#   调用方主动要求，允许生效于整个构建树）。
set(NN_COMPILER_LAUNCHER "" CACHE STRING
    "Compiler launcher prefix (e.g. ccache / sccache); empty = auto-detect (top-level only)")
set(NN_COMPILER_LAUNCHER_EXPLICIT "OFF" CACHE INTERNAL "NN_COMPILER_LAUNCHER passed by caller")
if(NN_COMPILER_LAUNCHER)
    set(NN_COMPILER_LAUNCHER_EXPLICIT "ON")
endif()
if(NOT NN_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER AND PROJECT_IS_TOP_LEVEL)
    find_program(NN_CCACHE_PROGRAM ccache)
    find_program(NN_SCCACHE_PROGRAM sccache)
    if(NN_CCACHE_PROGRAM)
        set(NN_COMPILER_LAUNCHER "${NN_CCACHE_PROGRAM}")
    elseif(NN_SCCACHE_PROGRAM)
        set(NN_COMPILER_LAUNCHER "${NN_SCCACHE_PROGRAM}")
    endif()
endif()
if(NN_COMPILER_LAUNCHER AND (NN_COMPILER_LAUNCHER_EXPLICIT OR PROJECT_IS_TOP_LEVEL))
    message(STATUS "Compiler launcher: ${NN_COMPILER_LAUNCHER}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${NN_COMPILER_LAUNCHER}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${NN_COMPILER_LAUNCHER}")
elseif(CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "Compiler launcher: ${CMAKE_CXX_COMPILER_LAUNCHER} "
                   "(来自 CMAKE_CXX_COMPILER_LAUNCHER，用 CMake 原生机制)")
else()
    message(STATUS "Compiler launcher: (none found; install ccache or sccache for faster rebuilds)")
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
# 注意：不使用 -ffast-math，因为它会禁用 NaN/Infinity 支持，影响训练稳定性
# MSVC 对应 /fp:fast 同样禁用 NaN 语义（实测：precision_type_test 的
# "NaN != NaN" / "NaN 比较均为 false" 断言在 /fp:fast 下被编译器折叠为
# false → 假失败；/fp:precise 下全绿）→ 按同一铁律用 /fp:precise。
# Clang/GCC 用 -O3 + 向量化（自带 -fno-math-errno/-fno-trapping-math）。
#
# -march=native 会让二进制只能在编译机同代 CPU 上运行，分发/换机可能 SIGILL。
# 默认开启（本地追求最大性能）；发布/CI 构建请用 -DNN_ENABLE_NATIVE=OFF 生成
# 通用可移植基线（见 .github/workflows 的 Configure 步骤）。
option(NN_ENABLE_NATIVE "Enable -march=native (max perf, non-portable binary)" ON)
if(MSVC)
    set(RELEASE_FLAGS /O2 /fp:precise)
    set(DEBUG_FLAGS /Od /Zi)
else()
    if(NN_ENABLE_NATIVE)
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native)
    else()
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops)
    endif()
    set(DEBUG_FLAGS -O0 -g)
endif()

add_compile_options(
    "$<$<CONFIG:Release>:${RELEASE_FLAGS}>"
    "$<$<CONFIG:Debug>:${DEBUG_FLAGS}>"
)

# ── 线程库（Linux 需要 pthread） ─────────────────────────────────────────
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ── std::stacktrace 运行时库探测 (GCC & Clang) ──────────────────────────
# GCC 14+ 和 Clang (使用 libstdc++) 都需要链接实验性库才能使用 std::stacktrace。
# GCC 使用 stdc++exp，Clang 可能需要 stdc++exp 或 stdc++_libbacktrace。
# 探测一次，成功后通过 nn_core INTERFACE 传播给所有目标。
# ⚠ **只有 `tools/scan_exprs.cpp` 用 std::stacktrace**（库头文件零使用，已 grep
#   确认）；嵌入（add_subdirectory）消费方编译头文件根本不需要它。因此 GNU 下
#   探测失败时：顶层工程仍 FATAL_ERROR（保住既有行为），嵌入方降级为警告 +
#   留空——否则消费方会为了一个用不到的库在配置期被卡死。

set(NN_STACKTRACE_FATAL ${PROJECT_IS_TOP_LEVEL})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # ── GCC 探测逻辑 ──────────────────────────────────────────────
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_HAS_STDCXXEXP)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if(NN_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        elseif(NN_STACKTRACE_FATAL)
            message(FATAL_ERROR "std::stacktrace 需要 stdc++exp 或 stdc++_libbacktrace")
        else()
            message(WARNING "std::stacktrace 库未找到（嵌入模式降级：仅 tools/scan_exprs 受影响）")
            set(NN_STACKTRACE_LIB "")
        endif()
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # ── Clang 探测逻辑 ─────────────────────────────────────────────
    # Clang 可能链接 libstdc++ 或 libc++。我们通过尝试链接来探测。
    # 如果 Clang 使用 libc++ 且其版本尚不支持 stacktrace，则此处会失败。
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_CLANG_HAS_STDCXXEXP)

    if(NN_CLANG_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        # 回退尝试旧版 libbacktrace 库
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        else()
            # Clang 可能在使用 libc++，其 stacktrace 支持可能不完整。
            message(WARNING "Clang 无法链接 std::stacktrace。如果使用 libc++，请确保版本 >= 18。CI 构建可能会失败。")
            set(NN_STACKTRACE_LIB "")
        endif()
    endif()
endif()

if(NN_STACKTRACE_LIB)
    message(STATUS "std::stacktrace link library: ${NN_STACKTRACE_LIB}")
else()
    message(STATUS "std::stacktrace link library: (none found,可能链接失败)")
endif()

# ── 公共 INTERFACE 库：所有目标共享 include 目录和链接库 ──────────────
# 通过 INTERFACE 库消除 7 个目标的重复设置
add_library(nn_core INTERFACE)
target_include_directories(nn_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(nn_core INTERFACE Threads::Threads)
if(NN_STACKTRACE_LIB)
    target_link_libraries(nn_core INTERFACE ${NN_STACKTRACE_LIB})
endif()

# ── C++ 标准（消费方使用需求，实测于 CMake 4.3）─────────────────────────────
# Clang/GNU（GNU 前端）认识 cxx_std_26 → 传 -std=c++26；
# **MSVC cl 不认识 cxx_std_26**（实测报 "not known to CXX compiler"，其
# CMAKE_CXX23_STANDARD_COMPILE_OPTION 就映射为 /std:c++latest = C++26 预览
# 全开，与本文件顶部 MSVC 分支同策略）→ 降级请求 cxx_std_23；
# clang-cl 认识 cxx_std_26（映射 /std:c++latest）→ 走 26。
# 判据用「特征列表实测」而非 if(MSVC)，因为 MSVC 变量对 clang-cl 也为真。
if("cxx_std_26" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(nn_core INTERFACE cxx_std_26)
elseif("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(nn_core INTERFACE cxx_std_23)
endif()

# ── 消费门面目标：下游只写 target_link_libraries(app PRIVATE neuralnet::nn) ──
# 嵌入（add_subdirectory / FetchContent）用法：
#     add_subdirectory(3rd_party/neuralnet.cpp)
#     target_link_libraries(app PRIVATE neuralnet::nn)
#     #include <neuralnet.cpp/nn.hpp>
# 门面 = nn_core（include/标准/线程）+ **编译本库头文件所需**的选项（见下）
#      + **GPU 开启时**的四项附加（在下方 `if(Vulkan_FOUND)` 块内补：
#      NN_HAS_VULKAN 宏 + generated/ 嵌入头目录 + Vulkan::Vulkan + shader
#      生成依赖）。
# ⚠ 刻意与 nn_core 分离的原因：
#   ① `scan_exprs` / 收集器必须保持**无 NN_HAS_VULKAN**（AOT 闭合世界，
#      AGENTS §7 铁律），它们只链 nn_core；
#   ② 下方这些「编译选项」只该给**真正的库使用者**（库内目标已从本文件的
#      目录级 add_compile_options 拿到，重复传递会让命令行出现两份相同 flag）。
add_library(neuralnet_cpp INTERFACE)
target_link_libraries(neuralnet_cpp INTERFACE nn_core)

# ── 编译本库头文件所需的消费方选项（目录级 add_compile_options 传不过去）──
# 实测缺口：下游 Release 构建库头文件时出现 6 个 `-Wpass-failed` 警告
# （`#pragma clang loop` 向量化未达成），库内靠 -Wno-pass-failed 压住；
# 消费方一旦开 -Werror 就直接编译失败——必须随目标传播。
target_compile_options(neuralnet_cpp INTERFACE
    # Clang：库内同款（-fexperimental-library 与 -Wno-pass-failed 都在本文件
    # 顶部的 Clang 分支加给库内目标；作用域限定 CXX 避免影响消费方 C 代码）
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-fexperimental-library>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-pass-failed>
    # MSVC：库头文件是 UTF-8 + 大量中文注释/字符串，不指定字符集时 cl 按
    # 系统 codepage 读 → C4819/C4828 甚至误解析；/utf-8 = source+execution
    # 都按 UTF-8（= 库内 /source-charset:utf-8 /execution-charset:utf-8 的
    # 合并等价物）。⚠ 这同时要求消费方源码也是 UTF-8（现代工程默认如此）。
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

add_library(neuralnet::nn ALIAS neuralnet_cpp)

# ── 预编译头（PCH）：本项目编译时间的最大头 ──────────────────────────────
# 库是 header-only，每个 TU 都要把 `nn.hpp`（→ 全部 Layer/Loss/Optimizer/
# Engine + 构建期嵌入的 36 份原语 SPIR-V + AOT 融合注册表，合计 ~7MB 头文本）
# 从头 parse 一遍。实测（Debug / clang / 本机）单 TU：
#   无 PCH 17.5s → 有 PCH 9.8s（-44%）。
# 所有候选目标的编译条件完全相同 → 只编**一份** PCH 供它们复用（REUSE_FROM）：
# 31 份逐目标 PCH 合计约 780s CPU，共享后只剩 1 份。见文件末尾的落地块。
# 关掉：-DNN_ENABLE_PCH=OFF。
# ⚠ **收集器目标不套 PCH**：`scan_exprs.cpp` 是在**源文件内**（include 之前）
#   自行 `#define NN_EXPR_SCAN` 的——PCH 会在该宏生效前把 nn.hpp 预编译进 AST，
#   导致 DSL 收集钩子整体失效（注册表空 → 构建硬失败）。故只有经
#   `nn_add_executable` 建的应用/测试目标进入候选。
option(NN_ENABLE_PCH "Precompile nn.hpp once and reuse it across targets" ON)

# ── 便捷函数：统一添加可执行目标 ────────────────────────────────────────
# PCH **不在这里挂**：目标编译条件（NN_HAS_VULKAN / include 目录）要到文件
# 末尾的 GPU 循环才定型，提前挂会与共享 PCH 目标的属性对不上（REUSE_FROM
# 要求两边完全一致）。这里只登记候选，末尾统一落地。
function(nn_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE nn_core)
    if(NN_ENABLE_PCH)
        set(NN_PCH_CANDIDATES ${NN_PCH_CANDIDATES} ${name} PARENT_SCOPE)
    endif()
endfunction()

# ── 测试/验证目标开关（默认关闭） ──────────────────────────────────────
# 默认只构建应用与基准工具目标，避免生成大量验证程序；
# 需要时用 -DNN_ENABLE_TESTS=ON 开启全部 gradcheck / *_test / *_probe / *_debug。
option(NN_ENABLE_TESTS "Build test/verification executables" OFF)

# 收集实际构建的目标，供 GPU 统一启用（避免引用未定义目标）
set(NN_APP_TARGETS "")
set(NN_TEST_TARGETS "")
set(NN_BENCH_TARGETS "")   # 性能基准已移除（2026 清理：bench 工具不注册 ctest 且无法在 -Werror 下编译）

# ── 应用 / 探针 / 基准目标（NN_BUILD_APPS，默认 = 是否顶层工程）───────────
# 嵌入消费方（add_subdirectory）默认 OFF：只拿库目标（neuralnet::nn），
# 不被迫编译本仓 12 个可执行文件。需要复现应用时显式 -DNN_BUILD_APPS=ON。
# ⚠ f16_writeback_probe 同时是 ctest 用例（见 NN_ENABLE_TESTS 块），故它
#   单独按「任一开关」创建；其余 app 只看 NN_BUILD_APPS。
if(NN_BUILD_APPS)
    # ── MNIST 目标（标准构建核心应用） ─────────────────────────────────────
    nn_add_executable(mnist_train src/mnist_train.cpp)
    nn_add_executable(mnist_infer src/mnist_infer.cpp)
    list(APPEND NN_APP_TARGETS mnist_train mnist_infer)

    # ── 文本目标（GPT 训练/推理 + BPE 分词器训练/推理，标准构建核心应用） ──
    nn_add_executable(text_train src/text_train.cpp)
    nn_add_executable(text_infer src/text_infer.cpp)
    nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
    nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)
    list(APPEND NN_APP_TARGETS text_train text_infer tokenizer_train tokenizer_infer)

    # ── 数据集工具（生成器 + 旧格式转换器）────────────────────────────────
    nn_add_executable(dataset_gen src/dataset_gen.cpp)
    nn_add_executable(dataset_convert src/dataset_convert.cpp)
    list(APPEND NN_APP_TARGETS dataset_gen dataset_convert)

    # ── 层吞吐量基准（CPU/Vulkan GPU；--all 或 --layer <name> 指定） ────────
    nn_add_executable(layer_bench src/layer_bench.cpp)
    list(APPEND NN_APP_TARGETS layer_bench)

    # ── 运行时显存分项探针（双池采样 + nvidia-smi + 逻辑分项归因） ─────────────
    nn_add_executable(mem_probe src/mem_probe.cpp)
    list(APPEND NN_APP_TARGETS mem_probe)

    # ── CPU f16 组件级诊断探针（A-I 阶段定位发散；NN_F16_DEBUG=1 开逐中间量扫描）──
    nn_add_executable(f16_cpu_probe src/f16_cpu_probe.cpp)
    list(APPEND NN_APP_TARGETS f16_cpu_probe)

    # ── CPU/GPU 训练 run-to-run 确定性探针（ComputeEngine Refresh P0 / D9 实测；
    #    手动运行不注册 ctest，退出码 0=逐位一致 1=非确定 2=运行错误）──────────
    nn_add_executable(gpu_stability_probe src/gpu_stability_probe.cpp)
    list(APPEND NN_APP_TARGETS gpu_stability_probe)
endif()

# ── GPU f16 参数写回探针（issue #13 P0-②：MSVC 下权重冻结定位）──────────────
# app 与 ctest 双身份 → 任一开关打开就创建（目标重复创建会报错，故统一在此）。
if(NN_BUILD_APPS OR NN_ENABLE_TESTS)
    nn_add_executable(f16_writeback_probe src/f16_writeback_probe.cpp)
    if(NN_BUILD_APPS)
        list(APPEND NN_APP_TARGETS f16_writeback_probe)
    endif()
endif()

# ── 测试 / 验证目标（默认关闭，NN_ENABLE_TESTS=ON 时构建）─────────────
if(NN_ENABLE_TESTS)

    # ══════════════════════════════════════════════════════════════════════
    # 测试目标（19 个，按功能域分组合并）
    # 合并策略：wrapper .cpp 通过 #define main + #include 聚合原测试源文件
    # ══════════════════════════════════════════════════════════════════════

    # ── GPU 集成测试（需要 Vulkan）────────────────────────────────────────
    nn_add_executable(gpu_test src/gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_test)

    # ── GPU f16 参数写回回归（issue #13 P0-②：MSVC 下权重冻结）───────────
    # 复用 app 目标 f16_writeback_probe（已在 NN_APP_TARGETS 构建）：
    # 断言 in-place f16 写回值正确 + 8 步 Adam 后参数必须移动（防"静默冻结"）
    list(APPEND NN_TEST_TARGETS f16_writeback_probe)

    # ── GPU f16 能力检查 + cast/matmul 验证 ──────────────────────────────
    nn_add_executable(gpu_f16_test src/gpu_f16_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_f16_test)

    # ── 多精度类型系统（f16 转换 / 舍入 / 张量精度 / f16 计算）──────────
    nn_add_executable(precision_test src/precision_test.cpp)
    list(APPEND NN_TEST_TARGETS precision_test)

    # ── 多精度适配层（f16 存储真正生效：PrecisionEngine + DSL + 端到端）──
    nn_add_executable(f16_precision_test src/f16_precision_test.cpp)
    list(APPEND NN_TEST_TARGETS f16_precision_test)

    # ── 分词器正确性（BPE pre_tokenize 对拍 + 标记往返）─────────────────
    nn_add_executable(tokenizer_test src/tokenizer_test.cpp)
    list(APPEND NN_TEST_TARGETS tokenizer_test)

    # ── Attention（梯度检查 + forward 一致性 + batch 行号回归）──────────
    nn_add_executable(attn_test src/attn_test.cpp)
    list(APPEND NN_TEST_TARGETS attn_test)

    # ── 基础层数值梯度检查（SwiGLU / RMSNorm / Softmax）─────────────────
    nn_add_executable(layer_gradcheck_test src/layer_gradcheck_test.cpp)
    list(APPEND NN_TEST_TARGETS layer_gradcheck_test)

    # ── BatchNorm（工厂不回落 + 训练/推理双态 fwd/bwd + running EMA
    #    + Model::set_training 转发 + save/load 往返逐位一致）────────────
    nn_add_executable(batchnorm_test src/batchnorm_test.cpp)
    list(APPEND NN_TEST_TARGETS batchnorm_test)

    # ── CNN（MaxPool2D 独立参考比对 + 缓存/checkpoint 契约 + 端到端冒烟）──
    nn_add_executable(cnn_test src/cnn_test.cpp)
    list(APPEND NN_TEST_TARGETS cnn_test)

    # ── GPT 模型（整链梯度检查 + 梯度检查点一致性）──────────────────────
    nn_add_executable(gpt_test src/gpt_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_test)

    # ── Activation Offload（原语往返 + GPT/RAPT 模型级一致性）──────────
    # 模型级 offload 一致性（GPU-only：无 Vulkan 构建时自身返回 77 = SKIP）
    nn_add_executable(gpt_offload_test src/gpt_offload_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_offload_test)
    nn_add_executable(rapt_offload_test src/rapt_offload_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_offload_test)

    # ── 文档感知注意力（端到端不变性 + 灵敏度/跨样本隔离对照；掩码构建
    #    单测已随 build_attention_mask 删除）────────────────────────────────
    nn_add_executable(doc_attn_test src/doc_attn_test.cpp)
    list(APPEND NN_TEST_TARGETS doc_attn_test)

    # ── ZiPT / AttnZip 已于 2026-10-01 移除（保留在 legacy/zipt 分支）──
    #     恢复前提见 docs/history.md「ZiPT 移除」条。

    # ── RAPT / ReLU 线性注意力（梯度 + 冒烟 + 梯度检查点）──────────────
    nn_add_executable(rapt_test src/rapt_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_test)

    # ── 表达式/IR 纯 CPU（DSL + 归约 + matmul + 优化 pass + 图 IR）────
    nn_add_executable(expr_cpu_test src/expr_cpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_cpu_test)

    # ── 表达式/IR GPU（图融合端到端 + AOT shader 验证 + 表达式求值）────
    nn_add_executable(expr_gpu_test src/expr_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_gpu_test)

    # ── 融合 shader **逐形态** GPU 对拍（rope/swiglu/gelu/softmax/matmul/
    #    matmul+reduce/norm/reduce_consts/fold v1/fold attn/回退）──────────
    # ⚠ 此前该文件**未被任何 CMake 目标引用**（未编译、未注册）——它是融合
    #    shader 逐形态的回归面，重新接入。
    # 覆盖口径：**V0（全 f32）**逐形态对拍（该文件无 f16/精度引用）。
    # V1（`key#x` 运行期精度分派）的覆盖在 `f16_precision_test`（NN_PREC_TRACE
    # 实测 20 条 #x / 4 条 #a）；本测试不覆盖 V1。
    nn_add_executable(fused_gpu_test src/fused_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS fused_gpu_test)

    # ── ModelSpec 架构校验（load_model 文件头 vs 模型规格比对）──────────
    nn_add_executable(model_spec_validation_test src/model_spec_validation_test.cpp)
    list(APPEND NN_TEST_TARGETS model_spec_validation_test)

    # ── 稀疏交叉熵融合（loss/grad vs 参考，CPU+GPU）─────────────────────
    nn_add_executable(ce_fusion_test src/ce_fusion_test.cpp)
    list(APPEND NN_TEST_TARGETS ce_fusion_test)

    # ── 错误处理宏族（NN_CHECK / NN_EXIT / NN_TRY_MSG / NN_FAIL）────────────
    # 覆盖：解值三形态 + 打印格式 + 传播语境前缀 + 终止/退出码
    # （终止行为经子进程 --expect-abort / --expect-exit <n> 断言，本进程不终止）
    nn_add_executable(error_macro_test src/error_macro_test.cpp)
    list(APPEND NN_TEST_TARGETS error_macro_test)

    # ── KVRecord v2（哨兵自举 / 嵌套 / 未知类型跳过 / 搬迁地址校验）─────────
    nn_add_executable(kvrec_test src/kvrec_test.cpp)
    list(APPEND NN_TEST_TARGETS kvrec_test)

    # ── .nnvocab 词表文件（json→nnvocab→加载语义一致 + 旧 json 明确拒绝）────
    nn_add_executable(nnvocab_test src/nnvocab_test.cpp)
    list(APPEND NN_TEST_TARGETS nnvocab_test)

    # ── .nndataset / nn::Dataset（生成→读取与旧 tokcache 产物逐位一致）─────
    nn_add_executable(dataset_test src/dataset_test.cpp)
    list(APPEND NN_TEST_TARGETS dataset_test)

endif()

# ── 测试目标统一处理：输出到 <默认可执行目录>/test，并注册到 ctest ──────────
# 单一开关 NN_ENABLE_TESTS 同时负责「编译测试程序」与「注册 ctest 测试」：
#   编译：上面 if(NN_ENABLE_TESTS) 内的 nn_add_executable(...)
#   注册：这里对 NN_TEST_TARGETS 逐一 add_test
# 可执行文件位置不硬编码生成器差异，统一设为 ${CMAKE_CURRENT_BINARY_DIR}/test，
# 由 CMake 自动适配：
#   - 单配置生成器（Ninja / Make）：默认可执行目录为 build 根目录 → build/test
#   - 多配置生成器（MSVC）：CMake 自动追加配置子目录 → build/test/Release(或 Debug)
# add_test 直接传目标名，CTest 会按当前配置自动解析出真实可执行文件路径。
if(NN_ENABLE_TESTS)
    enable_testing()
    foreach(TARGET ${NN_TEST_TARGETS})
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
        # 退出码 77 = 依约定"跳过"（如纯 CPU 构建下需要 Vulkan 的测试）：
        # 记 Skipped 而非 FAILED，否则 CPU-only 构建/CI 出现假红。
        set_tests_properties(${TARGET} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    # CNN 的 GPU 变体：DSL 表达式校验只在 GPU/scan 路径发生（CPU 模板路径不校验），
    # 默认的纯 CPU cnn_test 测不出"只在 GPU 触发"的表达式结构回归
    # （如 MaxPool2D 裸视图根 → 空指令表）。无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET cnn_test)
        add_test(NAME cnn_test_gpu COMMAND cnn_test --gpu)
        set_tests_properties(cnn_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # BatchNorm 的 GPU 变体：推理态/EMA 表达式的 ExprSpec 校验只在 GPU
    # 路径发生（CPU 模板路径不校验），且闭合世界 miss 只在 GPU 运行期暴露。
    # 无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET batchnorm_test)
        add_test(NAME batchnorm_test_gpu COMMAND batchnorm_test --gpu)
        set_tests_properties(batchnorm_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 基准工具已移除（2026 清理：无法在 -Werror 下编译，且无 ctest 覆盖）
    foreach(TARGET ${NN_BENCH_TARGETS})
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
    endforeach()
endif()

# ── Vulkan GPU 加速（NN_ENABLE_GPU = AUTO/ON/OFF，见文件头） ───────────────
if(NN_ENABLE_GPU STREQUAL "OFF")
    message(STATUS "Vulkan: disabled (NN_ENABLE_GPU=OFF); pure CPU build")
    set(Vulkan_FOUND OFF)
elseif(NN_ENABLE_GPU STREQUAL "ON")
    # 强制模式：缺 Vulkan 直接配置失败（fail-fast，不静默降级 CPU）
    find_package(Vulkan REQUIRED)
else()
    find_package(Vulkan QUIET)
endif()
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")

    # 查找 SPIR-V 编译器（Vulkan SDK 自带）
    find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
    if(NOT GLSLC AND NN_ENABLE_GPU STREQUAL "ON")
        # ON = 强制模式：缺 glslc 同样 fail-fast（否则 GPU 静默降级 CPU）
        message(FATAL_ERROR "NN_ENABLE_GPU=ON but glslc not found. Install the Vulkan SDK "
                            "or use NN_ENABLE_GPU=AUTO/OFF.")
    endif()

    # ── 铁律：原语 shader 永不含任何算法 ──────────────────────────────
    # 原语 shader 只做 op-level 计算（矩阵乘、逐元素、归约、广播、搬移）。
    # ReLU/GeLU/Softmax/LayerNorm/Attention/RoPE/SwiGLU 等算法一律来自
    # Layer（组合原语）或 DSL（nn::dsl，见下），绝不写进原语 shader。
    set(GPU_SHADERS
        matmul
        matmul_tiled
        batched_matmul
        rearrange_3d
        elementwise_v2
        transpose
        gather
        scatter_add
        im2col
        col2im
        group_reduce
        scan_prefix_outer
        scan_suffix_outer
        scan_prefix_outer_gen
        scan_suffix_outer_gen
        outer_col
        cast)
    # 融合 shader：构建期从 Layer 内联表达式自动合成（**单步**）──────────
    # 表达式**文本只出现在 Layer**（compute / compute_reduce / compute_into）。
    # scan_exprs 同进程内完成：收集折叠出的 ExprSpec 结构（锚点自登记 + dry-run
    # + 模型 pass + 显式登记，按 key 去重）→ glsl_gen → glslc → 内联 SPIR-V
    # → fused_registry.hpp（key=expr_spec_key[#x|#a] → 融合 shader）。
    # 不经 .bin 中间序列化（原独立工具 gen_fused 已并入 tools/fused_generate.hpp）。
    # 运行时按 key 精确匹配 dispatch；未命中硬报错（闭合世界）。
    set(GPU_SHADER_HPPS "")

    # 便捷函数：glslc → .spv → embed_spirv.cmake → C++ 头文件
    # 可选附加 glslc 参数走 ARGN（如 --target-env=...；不传则保持默认，
    # 现有 shader 的编译参数完全不变）
    # ⚠ 函数体内用 CMAKE_CURRENT_SOURCE_DIR/CMAKE_CURRENT_BINARY_DIR（= 本仓
    #   源/构建根，add_subdirectory 嵌入时自动指向 3rd_party 子树），不能用
    #   CMAKE_SOURCE_DIR/CMAKE_BINARY_DIR（嵌入时会指到消费方工程的根）。
    #   本函数只在本文件作用域内被调用，故两变量在定义/调用期同值。
    function(nn_embed_shader SHADER_NAME SRC_FILE)
        set(SPV ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        set(GPU_SHADER_SPVS ${GPU_SHADER_SPVS} ${SPV} PARENT_SCOPE)
        set(GPU_SHADER_HPPS ${GPU_SHADER_HPPS} ${HPP} PARENT_SCOPE)

        # 编译 GLSL → SPIR-V
        #   -I shaders：手写 shader 的 #include "nn_f16_rne.glsl"（f16 写出的
        #   RNE 预舍入，见该文件头注释）；所有 shader 都依赖它 → 改动时全量重编。
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${ARGN} -o ${SPV} ${SRC_FILE}
            DEPENDS ${SRC_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/nn_f16_rne.glsl
            COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv"
        )

        # 嵌入 SPIR-V → C++ 头文件
        add_custom_command(
            OUTPUT ${HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SPV}
                -DHPP_OUTPUT=${HPP}
                -DFUNC_NAME=${FUNC}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SPV} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header"
        )
    endfunction()

    # 1. 普通（手写）纯原语 shader
    foreach(SHADER_NAME ${GPU_SHADERS})
        nn_embed_shader(${SHADER_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
    endforeach()

    # 1b. 小 N GEMV（matmul_gemv）：用 subgroup 归约（GL_KHR_shader_subgroup_*）
    #     需 SPIR-V ≥1.3 → 显式 --target-env=vulkan1.2（与 gen_fused 一致）；
    #     其余 shader 编译参数不变。
    nn_embed_shader(matmul_gemv ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_gemv.comp --target-env=vulkan1.2)

    # 1c. reduce：行列归约用 subgroup 蝶式（GL_KHR_shader_subgroup_*，与
    #     DSL 融合归约同配方）→ 同样需 SPIR-V ≥1.3 → --target-env=vulkan1.2
    nn_embed_shader(reduce ${CMAKE_CURRENT_SOURCE_DIR}/shaders/reduce.comp --target-env=vulkan1.2)

    # 1d. 精度变体（Phase 2 in-kernel f16）：**同一份原语 shader** 用
    #     -DNN_SHADER_F16=1 编出第二份 SPIR-V（缓冲声明 float16_t + 别名槽
    #     uvec2/unpackHalf2x16，f32 累加，f16 写出）。后端按操作数精度选
    #     pipeline；设备无 storageBuffer16BitAccess 时不创建（回退 f32 边界
    #     cast，正确性不变）。f32 分支文本逐字未改 → 零回归。
    nn_embed_shader(batched_matmul_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/batched_matmul.comp -DNN_SHADER_F16=1)
    nn_embed_shader(matmul_tiled_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_tiled.comp -DNN_SHADER_F16=1)
    # C2e：小 N GEMV 的 f16 变体（n_out≤8 推理热路径；subgroup → vulkan1.2）
    nn_embed_shader(matmul_gemv_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_gemv.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
    # C2f：RLA 扫描原语的 f16 存储变体（输入输出均 half；输出 (·,seq) 全
    # 尺寸——消灭适配层 scan 的全尺寸 f32 副本；scratch St 仍 f32）。
    nn_embed_shader(scan_prefix_outer_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_prefix_outer.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_suffix_outer_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_suffix_outer.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_prefix_outer_gen_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_prefix_outer_gen.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_suffix_outer_gen_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_suffix_outer_gen.comp -DNN_SHADER_F16=1)
    nn_embed_shader(outer_col_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/outer_col.comp -DNN_SHADER_F16=1)

    # 1e. Phase C1 op-level f16 变体（消灭边界 cast 的搬运大头）：
    #     transpose / rearrange_3d / gather 的 f16 存储版（纯位拷贝零算术）。
    #     - gather：indices 保持 float（行号整数值 > f16 精度，禁转）；
    #     - transpose：shared float16_t → SPIR-V Float16 capability →
    #       运行时要求 shaderFloat16（否则后端跳过该 pipeline，回退边界 cast）；
    #     - rearrange_3d：仅 StorageBuffer16BitAccess。
    #     与 matmul 同款：f32 分支逐字未改 → 零回归。
    nn_embed_shader(transpose_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/transpose.comp -DNN_SHADER_F16=1)
    nn_embed_shader(rearrange_3d_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rearrange_3d.comp -DNN_SHADER_F16=1)
    nn_embed_shader(gather_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gather.comp -DNN_SHADER_F16=1)
    # C1b：add_inplace/scale_inplace/accumulate 的 f16 存储变体
    # （归因表 (256,8192)/(64,8192)/(2048,256) inplace 大头；仅需16位存储）
    nn_embed_shader(elementwise_v2_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/elementwise_v2.comp -DNN_SHADER_F16=1)
    # C2：行列归约的 f16 存储变体（归因 L756 大头：Linear::backward grad_bias
    # 直调 row_reduce_sum 的 (64,8192)×38 + (256,8192)×8 = 140MB；输入直读
    # f16、输出归约向量保持 float）。subgroup → 同 reduce 需 vulkan1.2。
    nn_embed_shader(reduce_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/reduce.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
    # C2b：embedding 梯度原子累加的 f16 存储变体（dst 打包 half CAS、
    # grad half 直读、indices 保持 float）——归因 L518/521 (8208,64)×2 +
    # (8192,64)×4；仅需 16bit 存储（CAS 对 uint word，无 f16 原子）。
    nn_embed_shader(scatter_add_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scatter_add.comp -DNN_SHADER_F16=1)
    # C2c：分组归约的 f16 存储变体（CNN 池化路径：输入 half 直读、输出
    # float——调用方 P=F32；仅需 16bit 存储）
    nn_embed_shader(group_reduce_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/group_reduce.comp -DNN_SHADER_F16=1)
    # C2d：卷积窗口展开/伴随散射的 f16 存储变体（数据搬运，in/out 均 half）
    nn_embed_shader(im2col_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/im2col.comp -DNN_SHADER_F16=1)
    nn_embed_shader(col2im_f16 ${CMAKE_CURRENT_SOURCE_DIR}/shaders/col2im.comp -DNN_SHADER_F16=1)

    # 2. AOT 融合 shader：scan_exprs（**单步** = 收集 + 生成）
    #    收集：FusedAnchor 自登记（按表达式类型）+ dry-run + 模型 pass
    #    生成：expr_emitter（默认 GLSL）→ glslc → 内联 SPIR-V → fused_registry.hpp
    #    原 gen_fused 已并入 scan_exprs 的生成阶段（tools/fused_generate.hpp），
    #    不再经 expr_specs.bin 中间文件。
    set(NN_FUSED_REGISTRY_HPP ${CMAKE_CURRENT_BINARY_DIR}/generated/fused_registry.hpp)

    set(SCAN_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/tools/scan_exprs.cpp)
    set(FUSED_GEN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tools/fused_generate.hpp)
    add_executable(scan_exprs ${SCAN_SRC})
    target_include_directories(scan_exprs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/neuralnet.cpp)
    target_link_libraries(scan_exprs PRIVATE nn_core Vulkan::Vulkan)
    add_custom_command(
        OUTPUT ${NN_FUSED_REGISTRY_HPP}
        # scan_exprs 是原生可执行文件，直接写文件、不会自建目录。
        # Ninja 会自动创建自定义命令的输出目录，但 Makefiles 生成器不会，
        # 全新构建时 generated/ 尚未存在会写入失败（见 08-pitfalls）。
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
        COMMAND scan_exprs ${CMAKE_CURRENT_BINARY_DIR}/generated ${GLSLC}
        DEPENDS scan_exprs ${SCAN_SRC} ${FUSED_GEN_SRC}
        COMMENT "[GPU] Scan + synthesize fused shaders (scan_exprs) → fused_registry.hpp"
    )

    list(APPEND GPU_SHADER_HPPS ${NN_FUSED_REGISTRY_HPP})

    # ── 库自带的融合能力：为**任意目标**收集 + 生成它自己的 fused_registry.hpp ──
    # 用法：nn_enable_gpu_fusion(<target> MAIN <含 main 的源文件> [OUT_DIR <目录>])
    #
    # 语义（对应"写 dsl::compute 之后只跑构建即可"）：
    #   收集器 = 库内收集逻辑（`tools/scan_exprs.cpp`：锚点自登记 + dry-run +
    #   模型 pass + 显式登记）+ **本目标的全部 TU**（用 -DNN_EXPR_SCAN 编，
    #   让使用者的 `dsl::compute` 调用点在静态初始化期自登记）。
    #   产物目录只作为该目标的私有 include 目录——"`fused_registry.hpp` 在
    #   include path 上即生效"这一既有机制不变。
    #
    # ⚠ 收集器**必须保留库内收集逻辑**：实测 84 个结构里只有 13 个来自
    #   编译期锚点，71 个来自 dry-run/模型 pass（见 docs/history.md 与
    #   docs/development/02）。纯锚点收集器会漏掉库自身的结构。
    #
    # ── 树外可用性（CMake 函数作用域）────────────────────────────────────
    # CMake 函数体在**调用方作用域**求值：消费方在自己的 CMakeLists 里调用
    # 本函数时，CMAKE_SOURCE_DIR/CMAKE_BINARY_DIR/SCAN_SRC 等都会解析到
    # **消费方**的目录（实测报错）。因此函数体内一律用：
    #   NN_LIB_SOURCE_DIR / NN_LIB_BINARY_DIR —— 本仓路径，configure 期
    #     写入 CACHE（INTERNAL 隐含 FORCE，每次配置刷新；全局可见）。
    #   GLSLC —— find_program 的结果本身是 CACHE 变量，天然全局可见。
    #   neuralnet::nn —— 收集器只链**门面目标**（见下方链接处的两作用说明）。
    #     ⚠ 不能在函数体内直接写 Vulkan::Vulkan：IMPORTED 目标**不可跨目录
    #     直接引用**（实测消费方作用域报 "target was not found"）；而门面是
    #     普通/ALIAS 目标（全局可见，实测可用），其 INTERFACE 里的
    #     Vulkan::Vulkan 条目在**定义侧**解析（实测可用，见探针）。
    function(nn_enable_gpu_fusion target)
        cmake_parse_arguments(NNF "" "MAIN;OUT_DIR" "" ${ARGN})
        if(NOT TARGET compile_gpu_shaders)
            message(FATAL_ERROR
                "nn_enable_gpu_fusion: GPU 构建不可用（未找到 Vulkan/glslc，"
                "或 NN_ENABLE_GPU=OFF）。该函数只在 GPU 启用时有定义。")
        endif()
        if(NOT TARGET ${target})
            message(FATAL_ERROR "nn_enable_gpu_fusion: 目标 ${target} 不存在")
        endif()
        if(TARGET ${target}_nnfusion_collect)
            message(FATAL_ERROR
                "nn_enable_gpu_fusion: 目标 ${target} 已调用过（${target}_nnfusion_collect 已存在）")
        endif()
        if(NOT NNF_OUT_DIR)
            set(NNF_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/nn_fusion/${target}")
        endif()
        if(NNF_MAIN)
            get_filename_component(NNF_MAIN "${NNF_MAIN}" ABSOLUTE)
        endif()
        get_target_property(_nnf_srcs ${target} SOURCES)
        set(_nnf_inputs "")
        set(_nnf_main_found FALSE)
        foreach(_s IN LISTS _nnf_srcs)
            if(_s MATCHES "^\\$<")          # 生成器表达式源（跳过，罕见）
                continue()
            endif()
            get_filename_component(_abs "${_s}" ABSOLUTE)
            if(NNF_MAIN AND _abs STREQUAL NNF_MAIN)
                set(_nnf_main_found TRUE)
                continue()                 # 收集器自带 main（SCAN_SRC），排除目标的
            endif()
            list(APPEND _nnf_inputs "${_abs}")
        endforeach()
        # MAIN 拼写/路径不一致会让目标的 main 进入收集器 → 链接期 duplicate main。
        # 静默通过不了（链接会报），但报错点离根因很远，这里提前拦住。
        if(NNF_MAIN AND NOT _nnf_main_found)
            message(FATAL_ERROR
                "nn_enable_gpu_fusion: MAIN=${NNF_MAIN} 不在 ${target} 的源文件列表里")
        endif()
        set(_nnf_scan_src  "${NN_LIB_SOURCE_DIR}/tools/scan_exprs.cpp")
        set(_nnf_gen_src   "${NN_LIB_SOURCE_DIR}/tools/fused_generate.hpp")
        set(_nnf_lib_inc   "${NN_LIB_SOURCE_DIR}/include/neuralnet.cpp")
        set(_nnf_lib_gen   "${NN_LIB_BINARY_DIR}/generated")
        set(_nnf_reg "${NNF_OUT_DIR}/fused_registry.hpp")
        add_executable(${target}_nnfusion_collect ${_nnf_scan_src} ${_nnf_inputs})
        target_include_directories(${target}_nnfusion_collect
            PRIVATE ${_nnf_lib_inc})
        # 收集器必须按**与目标相同的编译条件**编译其 TU：使用者的层代码常在
        # `#ifdef NN_HAS_VULKAN` / 私有 include / 自定义宏之内，条件缺失会让那些
        # 调用点静默消失 → 收集不到 → GPU 运行期闭合世界硬报错（离根因很远）。
        target_include_directories(${target}_nnfusion_collect PRIVATE
            $<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>)
        target_compile_options(${target}_nnfusion_collect PRIVATE
            $<TARGET_PROPERTY:${target},COMPILE_OPTIONS>)
        target_compile_definitions(${target}_nnfusion_collect PRIVATE
            NN_EXPR_SCAN $<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>)
        # 链门面目标（而非 nn_core）——两个作用：
        # ① 消费方目标的 NN_HAS_VULKAN/generated/ 等是经 neuralnet::nn 的
        #   **INTERFACE** 传入的，`$<TARGET_PROPERTY:...,COMPILE_DEFINITIONS>`
        #   只取目标**自身**属性、拿不到 INTERFACE 项（实测缺口）；收集器直接
        #   链门面即可拿到同款条件——否则使用者 `#ifdef NN_HAS_VULKAN` 里的
        #   层代码在收集器 TU 中被编译掉 → 收集不到 → GPU 运行期闭合世界硬报错。
        # ② include 顺序：收集器**自身**目录（含 NNF_OUT_DIR，经上一行属性
        #   复制）先于门面 INTERFACE 目录（库 generated/），注册表解析优先级
        #   正确（实测：自身目录恒先于 INTERFACE 目录）。
        # 库内目标不受影响：NN_HAS_VULKAN 本就经其自身属性复制进来，链门面
        # 只是同值重复（编译器对重复 -D 同值无警告）。
        target_link_libraries(${target}_nnfusion_collect PRIVATE
            neuralnet::nn)
        add_custom_command(
            OUTPUT ${_nnf_reg}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${NNF_OUT_DIR}
            COMMAND ${target}_nnfusion_collect ${NNF_OUT_DIR} ${GLSLC}
            DEPENDS ${target}_nnfusion_collect ${_nnf_scan_src} ${_nnf_gen_src} ${_nnf_inputs}
            COMMENT "[nn] 为 ${target} 收集 + 合成融合 shader → ${_nnf_reg}")
        add_custom_target(${target}_nnfusion_registry DEPENDS ${_nnf_reg})
        # ⚠ include 顺序有意义：本目标自己的 `fused_registry.hpp` 必须排在
        #   库的 `generated` **之前**——后者含库自身的注册表（不含使用者的
        #   自定义结构）与手写原语 shader 的嵌入头。放在前面才能让
        #   `__has_include("fused_registry.hpp")`（见 backend/compute_vk_backend.hpp）
        #   解析到本目标专属的那一份。
        # **必须用 BEFORE**：CMake 的 INCLUDE_DIRECTORIES 属性对重复项保留
        #   首次出现的位置，若目标此前已（由 GPU 统一启用循环）拿到
        #   `generated`，普通 append 会把它排在前面，顺序语义反而失效。
        # 实测顺序：目标自身目录（BEFORE 置前）**全部先于** INTERFACE 传入的
        # 目录，故消费方经 neuralnet::nn 拿到的库 generated/ 也排在本目录之后。
        target_include_directories(${target} BEFORE PRIVATE
            ${NNF_OUT_DIR} ${_nnf_lib_gen})
        add_dependencies(${target} ${target}_nnfusion_registry compile_gpu_shaders)
    endfunction()

    if(NOT GLSLC)
        message(WARNING "No SPIR-V compiler found (glslc). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        set(NN_GPU_ENABLED ON)   # 供文件末尾的共享 PCH 目标对齐编译条件
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})

        # ── 门面目标的 GPU 使用需求（消费方只链 neuralnet::nn 即得全部）──────
        # ① NN_HAS_VULKAN 宏：门控 nn.hpp 里的 GPU 引擎编译；
        # ② generated/ 目录：36 份原语 SPIR-V 嵌入头 + 库自身 fused_registry.hpp
        #    （nn.hpp 靠 `__has_include` 探测，目录必须在 include path 上）；
        # ③ Vulkan::Vulkan：imported 目标在**本目录作用域**内链入 INTERFACE 属性
        #    ——实测消费方经传递链接可正常解析（与直接 target_link_libraries
        #    到消费方目标不同，后者会报 target not found）；
        # ④ add_dependencies 到 compile_gpu_shaders：实测 INTERFACE 库的依赖
        #    **传递挂到消费方目标的编译边**（cmake_object_order_depends_target_*），
        #    保证 generated/ 头先于消费方 TU 编译生成。
        target_compile_definitions(neuralnet_cpp INTERFACE NN_HAS_VULKAN)
        target_include_directories(neuralnet_cpp INTERFACE
            ${CMAKE_CURRENT_BINARY_DIR}/generated)
        target_link_libraries(neuralnet_cpp INTERFACE Vulkan::Vulkan)
        add_dependencies(neuralnet_cpp compile_gpu_shaders)

        # 为所有实际构建的目标启用 Vulkan（测试目标按 NN_ENABLE_TESTS 决定）
        set(NN_GPU_TARGETS ${NN_APP_TARGETS})
        if(NN_ENABLE_TESTS)
            list(APPEND NN_GPU_TARGETS ${NN_TEST_TARGETS})
        endif()

        foreach(TARGET ${NN_GPU_TARGETS})
            target_compile_definitions(${TARGET} PRIVATE NN_HAS_VULKAN)
            target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
            target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
            add_dependencies(${TARGET} compile_gpu_shaders)
        endforeach()

        # ── A2 验收样例：一个「库外使用者」形态的目标 ──────────────────────
        # 它有自己的自研融合层（examples/fusion_custom_layer.hpp），**不使用**
        # 库内 build/generated 的注册表，而是靠 nn_enable_gpu_fusion 生成自己
        # 的那一份 —— 证明"写 dsl::compute + 一行 CMake 即可"。
        # 只在 NN_ENABLE_TESTS 下构建：它的收集器会为 ~85 个库内结构各跑一遍
        # glslc（三份变体），是纯验收成本，不该加到默认构建上。
        if(NN_ENABLE_TESTS)
            nn_add_executable(fusion_custom_layer_example
                examples/fusion_custom_layer_main.cpp
                examples/fusion_custom_layer.cpp)
            target_compile_definitions(fusion_custom_layer_example PRIVATE NN_HAS_VULKAN)
            target_link_libraries(fusion_custom_layer_example PRIVATE Vulkan::Vulkan)
            nn_enable_gpu_fusion(fusion_custom_layer_example
                MAIN ${CMAKE_CURRENT_SOURCE_DIR}/examples/fusion_custom_layer_main.cpp)

            # 注册为**常驻 ctest**（注册循环在 Vulkan 探测之前跑，故此处手工注册，
            # 口径与循环一致：输出目录 + SKIP_RETURN_CODE 77 = 无设备时 skip）。
            # 这是"库外使用者写 dsl::compute + 一行 CMake → GPU 能跑"的唯一自动
            # 门禁（含 f16 运行期精度分派 `key#x` 路径与归一化形态）。
            set_target_properties(fusion_custom_layer_example PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
            add_test(NAME fusion_custom_layer_example
                     COMMAND fusion_custom_layer_example)
            set_tests_properties(fusion_custom_layer_example
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
    # stub：GPU 不可用时仍定义 nn_enable_gpu_fusion，让消费方调用得到一句
    # 指向根因的报错，而不是难定位的 "Unknown CMake command"。
    function(nn_enable_gpu_fusion target)
        message(FATAL_ERROR
            "nn_enable_gpu_fusion 需要 GPU 构建，但本次配置 Vulkan 不可用"
            "（未找到 Vulkan/glslc，或 NN_ENABLE_GPU=OFF）。安装 Vulkan SDK "
            "或改用 -DNN_ENABLE_GPU=ON 强制要求。")
    endfunction()
endif()

# ══════════════════════════════════════════════════════════════════════════
#  PCH 落地（所有目标创建完、编译条件定型之后再挂）
# ══════════════════════════════════════════════════════════════════════════
# 候选目标的编译条件逐项相同（nn_core 的 include + NN_HAS_VULKAN +
# generated/ + Vulkan include），因此**一份 PCH 全目标共享**（REUSE_FROM）。
# 代价对比（本机 Debug 全量构建）：31 份逐目标 PCH ≈ 780s CPU → 1 份。
# ⚠ 例外：`fusion_custom_layer_example` 经 `nn_enable_gpu_fusion` 多出
#   自己的产物目录（BEFORE include），条件不匹配 → 单独一份，不能 REUSE。
if(NN_ENABLE_PCH AND NN_PCH_CANDIDATES)
    add_library(nn_pch OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/nn_pch_stub.cpp)
    target_link_libraries(nn_pch PRIVATE nn_core)
    if(NN_GPU_ENABLED)
        target_compile_definitions(nn_pch PRIVATE NN_HAS_VULKAN)
        target_include_directories(nn_pch PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
        target_link_libraries(nn_pch PRIVATE Vulkan::Vulkan)
        # ⚠ 必须等 `compile_gpu_shaders`（手写原语 SPIR-V 嵌入头 + fused_registry）
        #   生成完再编 PCH：nn.hpp 里那 36 个 `#if __has_include("..._spv.hpp")`
        #   与 `#if __has_include("fused_registry.hpp")` 是在**编 PCH 那一刻**求值
        #   的——早于生成完成就会整块落空，且从此不再重新求值（没 include 就没
        #   depfile 记录），表现为运行期 "matmul SPIR-V bytecode not embedded" /
        #   融合闭合世界硬报错。逐目标 PCH 天然带目标依赖，共享 PCH 必须显式补。
        add_dependencies(nn_pch compile_gpu_shaders)
    endif()
    target_precompile_headers(nn_pch PRIVATE [["neuralnet.cpp/nn.hpp"]])

    set(NN_PCH_OWN_TARGETS "")
    foreach(_pch_tgt IN LISTS NN_PCH_CANDIDATES)
        if(_pch_tgt STREQUAL "fusion_custom_layer_example")
            target_precompile_headers(${_pch_tgt} PRIVATE [["neuralnet.cpp/nn.hpp"]])
            list(APPEND NN_PCH_OWN_TARGETS ${_pch_tgt})
        else()
            target_precompile_headers(${_pch_tgt} REUSE_FROM nn_pch)
        endif()
    endforeach()
    list(LENGTH NN_PCH_CANDIDATES _nn_pch_n)
    list(LENGTH NN_PCH_OWN_TARGETS _nn_pch_own)
    message(STATUS "PCH: nn.hpp precompiled once, shared by "
                   "${_nn_pch_n} targets (${_nn_pch_own} own PCH)")
elseif(NN_ENABLE_PCH)
    message(STATUS "PCH: no candidate targets")
endif()