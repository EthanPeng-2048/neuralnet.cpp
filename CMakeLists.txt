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
# Release: -O3 -ffast-math -march=native（性能优先）
# Debug:   -O0 -g（调试优先，禁用 -march=native 保证可移植性）
set(RELEASE_FLAGS -O3 -ffast-math)
set(DEBUG_FLAGS -O0 -g)

add_compile_options(
    "$<$<CONFIG:Release>:${RELEASE_FLAGS}>"
    "$<$<CONFIG:Debug>:${DEBUG_FLAGS}>"
)

# -march=native 仅在显式开启时使用（生成不可移植二进制）
option(NN_USE_NATIVE_ARCH "Use -march=native for maximum performance on this machine" OFF)
if(NN_USE_NATIVE_ARCH)
    add_compile_options(-march=native)
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

# ── 文本生成目标 ────────────────────────────────────────────────────────────
nn_add_executable(text_train src/text_train.cpp)
nn_add_executable(text_infer src/text_infer.cpp)

# ── 分词器目标 ──────────────────────────────────────────────────────────────
nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)

# ── 计算性能基准测试 ─────────────────────────────────────────────────────
nn_add_executable(compute_bench src/compute_bench.cpp)

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
