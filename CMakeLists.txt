cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 为所有目标添加编译标志
add_compile_options(-fexperimental-library -O3 -ffast-math -Wall -Wextra -Wpedantic -Werror)

# ── MNIST 目标 ──────────────────────────────────────────────────────────────
add_executable(mnist_train src/mnist_train.cpp)
add_executable(mnist_infer src/mnist_infer.cpp)

target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mnist_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── 文本生成目标 ────────────────────────────────────────────────────────────
add_executable(text_train src/text_train.cpp)
add_executable(text_infer src/text_infer.cpp)

target_include_directories(text_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(text_infer PRIVATE ${CMAKE_SOURCE_DIR}/include)
