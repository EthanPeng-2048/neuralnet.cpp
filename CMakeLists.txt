cmake_minimum_required(VERSION 3.20)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

# C++26 标准
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++26 -Wall -Wextra")

# 直接使用 clang++（非 VS 工具链）
# cmake -B build -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe"
# 然后 cmake --build build
#
# 或直接用 build.cmake 脚本（纯 cmake + clang++，无需任何构建工具）：
# cmake -P build.cmake

add_executable(mnist_train src/train.cpp)
add_executable(mnist_infer src/infer.cpp)

target_include_directories(mnist_train PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(mnist_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)