# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 默认 Release 模式，可通过 -DCMAKE_BUILD_TYPE=Debug 切换 ────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 公共编译标志 ─────────────────────────────────────────────────────────
# -fno-exceptions: 项目规范禁止异常，编译器层面强制执行
# -Wall -Wextra -Wpedantic -Werror: 严格警告
add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -ffast-math -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
set(RELEASE_FLAGS -O3 -ffast-math -funroll-loops -march=native)
set(DEBUG_FLAGS -O0 -g)

add_compile_options(
    "$<$<CONFIG:Release>:${RELEASE_FLAGS}>"
    "$<$<CONFIG:Debug>:${DEBUG_FLAGS}>"
)

# -ffast-math 会禁用 NaN/Infinity 支持，Clang 会对此产生警告
# 项目代码依赖 infinity()（如 Softmax 的数值稳定化），需要抑制此警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-Wno-nan-infinity-disabled)
endif()

# ── 线程库（Linux 需要 pthread） ─────────────────────────────────────────
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ── 公共 INTERFACE 库：所有目标共享 include 目录和链接库 ──────────────
# 通过 INTERFACE 库消除 7 个目标的重复设置
add_library(nn_core INTERFACE)
target_include_directories(nn_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nn_core INTERFACE Threads::Threads)

# ── 便捷函数：统一添加可执行目标 ────────────────────────────────────────
function(nn_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE nn_core)
endfunction()

# ── MNIST 目标 ──────────────────────────────────────────────────────────────
nn_add_executable(mnist_train src/mnist_train.cpp)
nn_add_executable(mnist_infer src/mnist_infer.cpp)
nn_add_executable(mnist_bench src/mnist_bench.cpp)

# ── 文本目标（GPT 训练/推理 + BPE 分词器训练/推理） ───────────────────
nn_add_executable(text_train src/text_train.cpp)
nn_add_executable(text_infer src/text_infer.cpp)
nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)

# ── 计算性能基准测试 ─────────────────────────────────────────────────────
nn_add_executable(compute_bench src/compute_bench.cpp)
nn_add_executable(bench_thresholds src/bench_thresholds.cpp)

# ── GPU 测试（需要 Vulkan） ──────────────────────────────────────────────
nn_add_executable(gpu_test src/gpu_test.cpp)

# ── Vulkan GPU 加速（可选） ────────────────────────────────────────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")

    # 查找 SPIR-V 编译器（Vulkan SDK 自带）
    find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})

    # 定义着色器列表（仅保留纯原语着色器，不含任何算法）
    # - matmul / matmul_tiled: 矩阵乘法原语
    # - batched_matmul: 批量矩阵乘法原语（Z 维度索引 batch，消除 per-head 循环）
    # - elementwise_v2: 逐元素原语（unary/binary/select）— 纯 GPU 架构核心
    # - reduce: 行/列归约原语（sum/max）— LayerNorm、CrossEntropy 等
    # - broadcast: 行/列广播原语（add/sub/mul/div/max/min）— Linear 偏置、
    #   LayerNorm 缩放、Softmax 归一化等
    # 旧的 elementwise / layernorm / layernorm_backward 着色器（含算法语义）
    # 已随旧架构删除。
    set(GPU_SHADERS
        matmul
        matmul_tiled
        batched_matmul
        rearrange_3d
        elementwise_v2
        reduce
        broadcast
        transpose
        gather
        scatter_add)
    set(GPU_SHADER_HPPS "")

    foreach(SHADER_NAME ${GPU_SHADERS})
        set(SRC ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
        set(SPV ${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        list(APPEND GPU_SHADER_SPVS ${SPV})
        list(APPEND GPU_SHADER_HPPS ${HPP})

        # 编译 GLSL → SPIR-V
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} -o ${SPV} ${SRC}
            DEPENDS ${SRC}
            COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv"
        )

        # 嵌入 SPIR-V → C++ 头文件
        add_custom_command(
            OUTPUT ${HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SPV}
                -DHPP_OUTPUT=${HPP}
                -DFUNC_NAME=${FUNC}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SPV} ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header"
        )
    endforeach()

    if(NOT GLSLC)
        message(WARNING "No SPIR-V compiler found (glslc). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})

        # 为所有目标启用 Vulkan
        set(NN_GPU_TARGETS
            mnist_train mnist_infer mnist_bench
            text_train text_infer
            tokenizer_train tokenizer_infer
            compute_bench bench_thresholds gpu_test
        )

        foreach(TARGET ${NN_GPU_TARGETS})
            target_compile_definitions(${TARGET} PRIVATE NN_HAS_VULKAN)
            target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/generated)
            target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
            add_dependencies(${TARGET} compile_gpu_shaders)
        endforeach()

        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
endif()

# ── 测试目标 ─────────────────────────────────────────────────────────────
# 启用测试（默认 OFF，可通过 -DNN_BUILD_TESTS=ON 开启）
option(NN_BUILD_TESTS "Build unit tests" OFF)
if(NN_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_const_span.cpp")
        nn_add_executable(test_const_span tests/test_const_span.cpp)
        add_test(NAME test_const_span COMMAND test_const_span)
    endif()
endif()
